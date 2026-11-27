use super::storage::MIN_BUFFER_SIZE;
use crate::{WgpuResource, WgpuStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_server::storage::{ComputeStorage, StorageId};

/// Backing buffer size for each dynamic-offset metadata page.
///
/// Typical `info` payloads are 80–512 bytes aligned to 256 bytes, so a single
/// 512 KiB page serves ~1,000–2,000 distinct entries from one `wgpu::Buffer`,
/// letting every launch of a pipeline share one cached `BindGroup`.
const INFO_PAGE_SIZE: u64 = 512 * 1024;

#[derive(Debug)]
struct InfoSlot {
    offset: u64,
    capacity: u64,
    token: Arc<()>,
}

#[derive(Debug)]
struct InfoPage {
    storage_id: StorageId,
    capacity: u64,
    cursor: u64,
    slots: Vec<InfoSlot>,
}

impl InfoPage {
    fn new(storage_id: StorageId, capacity: u64) -> Self {
        Self {
            storage_id,
            capacity,
            cursor: 0,
            slots: Vec::new(),
        }
    }

    fn is_idle(&self) -> bool {
        self.slots
            .iter()
            .all(|slot| Arc::strong_count(&slot.token) == 1)
    }

    fn reuse_slot(&self, aligned: u64, allow_oversized: bool) -> Option<(u64, Arc<()>)> {
        self.slots
            .iter()
            .find(|slot| {
                Arc::strong_count(&slot.token) == 1
                    && if allow_oversized {
                        slot.capacity >= aligned && slot.capacity <= aligned.saturating_mul(2)
                    } else {
                        slot.capacity == aligned
                    }
            })
            .map(|slot| (slot.offset, slot.token.clone()))
    }

    fn bump_slot(&mut self, aligned: u64) -> Option<(u64, Arc<()>)> {
        let end = self.cursor.checked_add(aligned)?;
        if end > self.capacity {
            return None;
        }
        let offset = self.cursor;
        self.cursor = end;
        let token = Arc::new(());
        let handle = token.clone();
        self.slots.push(InfoSlot {
            offset,
            capacity: aligned,
            token,
        });
        Some((offset, handle))
    }
}

/// Sub-allocating pool for kernel `info` metadata buffers, backed by a small
/// number of `WgpuStorage` pages aligned for dynamic bind-group offsets.
#[derive(Debug)]
pub(crate) struct InfoPool<S = WgpuStorage> {
    storage: S,
    alignment: u64,
    pages: Vec<InfoPage>,
    /// Tokens retained for launches in the current unsubmitted encoder.
    queued: Vec<Arc<()>>,
}

impl<S: ComputeStorage> InfoPool<S> {
    pub(crate) fn new(storage: S, alignment: u64) -> Self {
        Self {
            storage,
            alignment: alignment.max(MIN_BUFFER_SIZE),
            pages: Vec::new(),
            queued: Vec::new(),
        }
    }

    pub(crate) fn bytes_allocated(&self) -> u64 {
        self.storage.bytes_allocated()
    }

    pub(crate) fn pin_queued(&mut self, token: Arc<()>) {
        self.queued.push(token);
    }

    pub(crate) fn release_queued(&mut self) {
        self.queued.clear();
    }

    /// Reserve an `alignment`-aligned slice for a `size`-byte metadata payload
    /// and pin it until [`Self::release_queued`] runs after submission.
    fn reserve_slot(&mut self, size: u64) -> (StorageId, u64, Arc<()>) {
        let aligned = size.max(MIN_BUFFER_SIZE).next_multiple_of(self.alignment);

        for page in &self.pages {
            if let Some((offset, token)) = page.reuse_slot(aligned, false) {
                self.queued.push(token.clone());
                return (page.storage_id, offset, token);
            }
        }
        for page in &mut self.pages {
            if let Some((offset, token)) = page.bump_slot(aligned) {
                self.queued.push(token.clone());
                return (page.storage_id, offset, token);
            }
        }
        for page in &self.pages {
            if let Some((offset, token)) = page.reuse_slot(aligned, true) {
                self.queued.push(token.clone());
                return (page.storage_id, offset, token);
            }
        }

        let page_size = INFO_PAGE_SIZE.max(aligned).next_multiple_of(self.alignment);
        let handle = self
            .storage
            .alloc(page_size)
            .expect("Must have enough memory for kernel metadata");
        let mut page = InfoPage::new(handle.id, page_size);
        let (offset, token) = page
            .bump_slot(aligned)
            .expect("fresh info page fits requested slice");
        let storage_id = page.storage_id;
        self.pages.push(page);
        self.queued.push(token.clone());
        (storage_id, offset, token)
    }

    /// Release every page whose slots are all idle (neither queued nor cached).
    pub(crate) fn cleanup(&mut self) {
        let mut kept = Vec::with_capacity(self.pages.len());
        for page in self.pages.drain(..) {
            if page.is_idle() {
                self.storage.dealloc(page.storage_id);
            } else {
                kept.push(page);
            }
        }
        self.pages = kept;
    }
}

impl InfoPool<WgpuStorage> {
    pub(crate) fn reserve(&mut self, size: u64) -> (Arc<()>, WgpuResource) {
        let (storage_id, offset, token) = self.reserve_slot(size);
        let resource = self
            .storage
            .get_at(storage_id, offset, size)
            .expect("info page present in storage");
        (token, resource)
    }

    pub(crate) fn contains(&self, id: StorageId) -> bool {
        self.storage.contains(id)
    }

    pub(crate) fn take_released_ids(&mut self) -> Vec<StorageId> {
        self.storage.take_released_ids()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn destroy_pending(&mut self) {
        self.storage.destroy_pending();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::server::IoError;
    use cubecl_server::storage::{StorageHandle, StorageUtilization};

    struct FakeStorage {
        allocated: u64,
        deallocated: Vec<StorageId>,
    }

    impl FakeStorage {
        fn new() -> Self {
            Self {
                allocated: 0,
                deallocated: Vec::new(),
            }
        }
    }

    impl ComputeStorage for FakeStorage {
        type Resource = ();

        fn alignment(&self) -> usize {
            256
        }

        fn get(&mut self, _handle: &StorageHandle) -> Result<Self::Resource, IoError> {
            Ok(())
        }

        fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
            self.allocated += size;
            Ok(StorageHandle::new(
                StorageId::new(),
                StorageUtilization { offset: 0, size },
            ))
        }

        fn dealloc(&mut self, id: StorageId) {
            self.deallocated.push(id);
        }

        fn flush(&mut self) {}

        fn bytes_allocated(&self) -> u64 {
            self.allocated
        }
    }

    #[test]
    fn reuses_single_page_and_recycles_released_slots() {
        let mut pool = InfoPool::new(FakeStorage::new(), 256);

        let (id1, off1, pin1) = pool.reserve_slot(160);
        let (id2, off2, pin2) = pool.reserve_slot(304);
        assert_eq!(id1, id2);
        assert_eq!(off1, 0);
        assert_eq!(off2, 256);

        // Dropping only the caller pin while still queued must not recycle the slot.
        drop(pin1);
        let (id3, off3, _pin3) = pool.reserve_slot(160);
        assert_eq!(id3, id1);
        assert_eq!(off3, 768);

        // After submission releases queued pins, slot 0 is free while slot 1 stays pinned.
        pool.release_queued();
        let (id4, off4, _pin4) = pool.reserve_slot(160);
        assert_eq!(id4, id1);
        assert_eq!(off4, 0);
        assert_eq!(pool.pages.len(), 1);
        drop(pin2);
    }

    #[test]
    fn cleanup_releases_only_idle_pages() {
        let mut pool = InfoPool::new(FakeStorage::new(), 256);
        let (id, _, pin) = pool.reserve_slot(128);
        pool.release_queued();
        pool.cleanup();
        assert_eq!(pool.pages.len(), 1);
        assert!(pool.storage.deallocated.is_empty());

        drop(pin);
        pool.cleanup();
        assert!(pool.pages.is_empty());
        assert_eq!(pool.storage.deallocated, vec![id]);
    }
}
