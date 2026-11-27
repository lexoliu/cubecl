use alloc::sync::Weak;
use core::hash::Hash;
use core::ops::Deref;
use cubecl_core::zspace::SmallVec;
use cubecl_environment::collections::{HashMap, hash_map::Entry};
use cubecl_server::storage::StorageId;
use wgpu::ComputePipeline;

/// Maximum bind groups retained per stream. Covers steady-state model layers
/// while bounding GPU-process bind-group metadata on browsers (notably `WebKit`).
pub(crate) const DEFAULT_MAX_BIND_GROUPS: usize = 2048;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BindGroupKey {
    pub(crate) pipeline: usize,
    pub(crate) bindings: SmallVec<[(StorageId, u64, u64); 8]>,
}

pub(crate) enum BindGroupRef<'a, G> {
    Cached(&'a G),
    Owned(G),
}

impl<G> Deref for BindGroupRef<'_, G> {
    type Target = G;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Cached(group) => group,
            Self::Owned(group) => group,
        }
    }
}

#[derive(Debug)]
struct CachedBindGroup<G> {
    group: G,
    last_used: u64,
}

#[derive(Debug)]
struct CachedLayout<P, L> {
    owner: Weak<P>,
    layout: L,
}

/// Per-stream cache of pipeline bind-group layouts and bind groups, invalidated
/// whenever a referenced [`StorageId`] is deallocated.
#[derive(Debug)]
pub(crate) struct BindGroupCache<L, G, P = ComputePipeline> {
    layouts: HashMap<usize, CachedLayout<P, L>>,
    entries: HashMap<BindGroupKey, CachedBindGroup<G>>,
    by_storage: HashMap<StorageId, SmallVec<[BindGroupKey; 4]>>,
    clock: u64,
    max_entries: usize,
}

impl<L, G, P> Default for BindGroupCache<L, G, P> {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BIND_GROUPS)
    }
}

impl<L, G, P> BindGroupCache<L, G, P> {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            layouts: HashMap::new(),
            entries: HashMap::new(),
            by_storage: HashMap::new(),
            clock: 0,
            max_entries: max_entries.max(1),
        }
    }

    pub(crate) fn pipeline_ptr(pipeline: &alloc::sync::Arc<P>) -> usize {
        alloc::sync::Arc::as_ptr(pipeline) as usize
    }

    fn ensure_layout(
        &mut self,
        pipeline: &alloc::sync::Arc<P>,
        create: impl FnOnce() -> L,
    ) -> usize {
        let ptr = Self::pipeline_ptr(pipeline);
        let needs_refresh = match self.layouts.get(&ptr) {
            Some(cached) => cached.owner.strong_count() == 0,
            None => true,
        };
        if needs_refresh {
            if self.layouts.remove(&ptr).is_some() {
                self.evict_pipeline(ptr);
            }
            self.layouts.insert(
                ptr,
                CachedLayout {
                    owner: alloc::sync::Arc::downgrade(pipeline),
                    layout: create(),
                },
            );
        }
        ptr
    }

    /// Return the cached group-0 bind-group layout for `pipeline`, building it
    /// once per live pipeline. If an older pipeline at the same address was
    /// dropped (for instance after a compilation-cache reset), any bind groups
    /// recorded under that address are evicted before inserting the new layout.
    pub(crate) fn layout_for(
        &mut self,
        pipeline: &alloc::sync::Arc<P>,
        create: impl FnOnce() -> L,
    ) -> &L {
        let ptr = self.ensure_layout(pipeline, create);
        &self.layouts.get(&ptr).expect("layout inserted").layout
    }

    /// Return a cached bind group for `(pipeline, key)`, or build one using the
    /// cached layout for `pipeline`. When `key` is `None` (for example, a
    /// cross-stream binding whose storage lifecycle is tracked by another
    /// stream), the group is created uncached while still reusing the layout.
    pub(crate) fn get_or_create(
        &mut self,
        pipeline: &alloc::sync::Arc<P>,
        key: Option<BindGroupKey>,
        create_layout: impl FnOnce() -> L,
        create_group: impl FnOnce(&L) -> G,
    ) -> BindGroupRef<'_, G> {
        let ptr = self.ensure_layout(pipeline, create_layout);
        let Some(key) = key else {
            let layout = &self.layouts.get(&ptr).expect("layout inserted").layout;
            return BindGroupRef::Owned(create_group(layout));
        };

        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            self.evict_oldest();
        }
        let layout = &self.layouts.get(&ptr).expect("layout inserted").layout;
        match self.entries.entry(key) {
            Entry::Occupied(mut occupied) => {
                self.clock = self.clock.wrapping_add(1);
                occupied.get_mut().last_used = self.clock;
                BindGroupRef::Cached(&occupied.into_mut().group)
            }
            Entry::Vacant(vacant) => {
                for &(storage_id, _, _) in &vacant.key().bindings {
                    let list = self.by_storage.entry(storage_id).or_default();
                    if !list.contains(vacant.key()) {
                        list.push(vacant.key().clone());
                    }
                }
                let group = create_group(layout);
                self.clock = self.clock.wrapping_add(1);
                let entry = vacant.insert(CachedBindGroup {
                    group,
                    last_used: self.clock,
                });
                BindGroupRef::Cached(&entry.group)
            }
        }
    }

    /// Drop every cached bind group that references `storage_id`.
    pub(crate) fn evict_storage(&mut self, storage_id: StorageId) {
        let Some(keys) = self.by_storage.remove(&storage_id) else {
            return;
        };
        for key in keys {
            if self.entries.remove(&key).is_some() {
                for &(other_id, _, _) in &key.bindings {
                    if other_id == storage_id {
                        continue;
                    }
                    self.remove_key_from_storage_index(other_id, &key);
                }
            }
        }
    }

    /// Drop all cached bind groups while retaining layouts for live pipelines.
    pub(crate) fn clear_entries(&mut self) {
        self.entries.clear();
        self.by_storage.clear();
        self.layouts
            .retain(|_, cached| cached.owner.strong_count() > 0);
    }

    fn evict_pipeline(&mut self, pipeline: usize) {
        let doomed: alloc::vec::Vec<BindGroupKey> = self
            .entries
            .keys()
            .filter(|key| key.pipeline == pipeline)
            .cloned()
            .collect();
        for key in doomed {
            self.remove_entry(&key);
        }
    }

    fn evict_oldest(&mut self) {
        let evict_count = (self.max_entries / 8).max(1).min(self.entries.len());
        let mut by_age: alloc::vec::Vec<(u64, BindGroupKey)> = self
            .entries
            .iter()
            .map(|(k, v)| (v.last_used, k.clone()))
            .collect();
        if evict_count < by_age.len() {
            by_age.select_nth_unstable_by_key(evict_count, |(age, _)| *age);
        }
        for (_, key) in by_age.into_iter().take(evict_count) {
            self.remove_entry(&key);
        }
    }

    fn remove_entry(&mut self, key: &BindGroupKey) {
        if self.entries.remove(key).is_some() {
            for &(storage_id, _, _) in &key.bindings {
                self.remove_key_from_storage_index(storage_id, key);
            }
        }
    }

    fn remove_key_from_storage_index(&mut self, storage_id: StorageId, key: &BindGroupKey) {
        let mut empty = false;
        if let Some(list) = self.by_storage.get_mut(&storage_id) {
            if let Some(pos) = list.iter().position(|candidate| candidate == key) {
                list.swap_remove(pos);
            }
            empty = list.is_empty();
        }
        if empty {
            self.by_storage.remove(&storage_id);
        }
    }
}

pub(crate) type WgpuBindGroupCache = BindGroupCache<wgpu::BindGroupLayout, wgpu::BindGroup>;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;

    fn key(pipeline: &Arc<u32>, ids: &[StorageId]) -> BindGroupKey {
        BindGroupKey {
            pipeline: BindGroupCache::<(), (), u32>::pipeline_ptr(pipeline),
            bindings: ids.iter().map(|&id| (id, 0, 64)).collect(),
        }
    }

    #[test]
    fn evict_storage_drops_only_referencing_entries_and_cleans_sibling_indices() {
        let mut cache: BindGroupCache<u32, u32, u32> = BindGroupCache::new(16);
        let p = Arc::new(1);
        let s1 = StorageId::new();
        let s2 = StorageId::new();
        let s3 = StorageId::new();
        let k12 = key(&p, &[s1, s2]);
        let k23 = key(&p, &[s2, s3]);

        assert_eq!(
            *cache.get_or_create(&p, Some(k12.clone()), || 1, |_| 12),
            12
        );
        assert_eq!(
            *cache.get_or_create(&p, Some(k23.clone()), || 1, |_| 23),
            23
        );
        cache.evict_storage(s1);

        assert!(!cache.entries.contains_key(&k12));
        assert!(cache.entries.contains_key(&k23));
        assert!(!cache.by_storage.contains_key(&s1));
        assert_eq!(cache.by_storage.get(&s2).map(|v| v.len()), Some(1));
        assert_eq!(
            *cache.get_or_create(
                &p,
                Some(k23),
                || panic!("layout cached"),
                |_| panic!("group cached")
            ),
            23
        );
    }

    #[test]
    fn capacity_limit_evicts_least_recently_used_entries() {
        let mut cache: BindGroupCache<u32, u32, u32> = BindGroupCache::new(4);
        let p = Arc::new(1);
        let ids: alloc::vec::Vec<StorageId> = (0..5).map(|_| StorageId::new()).collect();
        let keys: alloc::vec::Vec<BindGroupKey> = ids.iter().map(|&id| key(&p, &[id])).collect();

        for (i, key) in keys.iter().enumerate().take(4) {
            assert_eq!(
                *cache.get_or_create(&p, Some(key.clone()), || 1, |_| i as u32),
                i as u32
            );
        }
        // Refresh entry 0 so entry 1 is now the oldest.
        assert_eq!(
            *cache.get_or_create(
                &p,
                Some(keys[0].clone()),
                || panic!("layout cached"),
                |_| panic!("entry 0 cached")
            ),
            0
        );
        assert_eq!(
            *cache.get_or_create(&p, Some(keys[4].clone()), || panic!("layout cached"), |_| 4),
            4
        );

        assert!(!cache.entries.contains_key(&keys[1]));
        assert!(cache.entries.contains_key(&keys[0]));
        assert!(cache.entries.contains_key(&keys[4]));
    }

    #[test]
    fn dropped_pipeline_invalidates_layout_and_entries() {
        let mut cache: BindGroupCache<u32, u32, u32> = BindGroupCache::new(8);
        let s1 = StorageId::new();
        let p1 = Arc::new(1);
        let mut layout_builds = 0;
        let mut group_builds = 0;
        let k1 = key(&p1, &[s1]);

        let g = *cache.get_or_create(
            &p1,
            Some(k1.clone()),
            || {
                layout_builds += 1;
                layout_builds
            },
            |_| {
                group_builds += 1;
                group_builds
            },
        );
        assert_eq!((g, layout_builds, group_builds), (1, 1, 1));

        // Simulate a dead owner recorded at the address of a live replacement pipeline.
        let p2 = Arc::new(2);
        let ptr2 = BindGroupCache::<u32, u32, u32>::pipeline_ptr(&p2);
        let dead = Arc::new(99);
        let dead_weak = Arc::downgrade(&dead);
        drop(dead);
        cache.layouts.insert(
            ptr2,
            CachedLayout {
                owner: dead_weak,
                layout: 99,
            },
        );
        let stale_key = BindGroupKey {
            pipeline: ptr2,
            bindings: SmallVec::from_slice(&[(s1, 0, 64)]),
        };
        cache.entries.insert(
            stale_key.clone(),
            CachedBindGroup {
                group: 99,
                last_used: 0,
            },
        );

        let rebuilt = *cache.get_or_create(
            &p2,
            Some(stale_key),
            || {
                layout_builds += 1;
                layout_builds
            },
            |_| {
                group_builds += 1;
                group_builds
            },
        );
        assert_eq!((rebuilt, layout_builds, group_builds), (2, 2, 2));

        drop(p1);
        cache.clear_entries();
        assert_eq!(cache.layouts.len(), 1);
    }

    #[test]
    fn uncacheable_key_reuses_layout_without_caching_group() {
        let mut cache: BindGroupCache<u32, u32, u32> = BindGroupCache::new(8);
        let p = Arc::new(1);
        let mut layout_builds = 0;
        let mut group_builds = 0;

        for expected in 1..=2 {
            let group = cache.get_or_create(
                &p,
                None,
                || {
                    layout_builds += 1;
                    layout_builds
                },
                |_| {
                    group_builds += 1;
                    group_builds
                },
            );
            assert!(matches!(group, BindGroupRef::Owned(v) if v == expected));
        }
        assert_eq!((layout_builds, group_builds), (1, 2));
        assert!(cache.entries.is_empty());
    }
}
