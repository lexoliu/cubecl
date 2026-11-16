use alloc::boxed::Box;
use core::fmt::Display;

pub use cubecl_environment::time::{Duration, Instant};

use cubecl_environment::future::DynFut;

/// How a benchmark's execution times are measured.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum TimingMethod {
    /// Time measurements come from full timing of execution + sync
    /// calls.
    System,
    /// Time measurements come from hardware reported timestamps
    /// coming from a sync call.
    Device,
}

impl Display for TimingMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimingMethod::System => f.write_str("system"),
            TimingMethod::Device => f.write_str("device"),
        }
    }
}

/// Start and end point for a profile. Can be turned into a duration.
#[derive(Debug)]
pub struct ProfileTicks {
    start: Instant,
    end: Instant,
}

impl ProfileTicks {
    /// Create a new `ProfileTicks` from a start and end time.
    pub fn from_start_end(start: Instant, end: Instant) -> Self {
        Self { start, end }
    }

    /// Get the duration contained in this `ProfileTicks`.
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }

    /// Get the duration since the epoch start of this `ProfileTicks`.
    pub fn start_duration_since(&self, epoch: Instant) -> Duration {
        self.start.duration_since(epoch)
    }

    /// Get the duration since the epoch end of this `ProfileTicks`.
    pub fn end_duration_since(&self, epoch: Instant) -> Duration {
        self.end.duration_since(epoch)
    }
}

/// Result from profiling between two measurements. This can either be a duration or a future that resolves to a duration.
///
/// The future resolves to [`None`] when the window turned out to carry no
/// measurement, which a backend can only discover once the device has answered.
/// That absence is not a zero: zero is the fastest duration there is, so a
/// caller comparing candidates would let an unmeasured one win every comparison
/// it enters.
pub struct ProfileDuration {
    // The future to read profiling data. For System profiling,
    // this should be entirely synchronous.
    future: DynFut<Option<ProfileTicks>>,
    method: TimingMethod,
}

impl ProfileDuration {
    /// The method used to measure the execution time.
    pub fn timing_method(&self) -> TimingMethod {
        self.method
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn new(future: DynFut<Option<ProfileTicks>>, method: TimingMethod) -> ProfileDuration {
        Self { future, method }
    }

    /// Create a new `ProfileDuration` straight from a duration.
    pub fn new_system_time(start: Instant, end: Instant) -> Self {
        Self::new(
            Box::pin(async move { Some(ProfileTicks::from_start_end(start, end)) }),
            TimingMethod::System,
        )
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn new_device_time(
        future: impl Future<Output = ProfileTicks> + Send + 'static,
    ) -> ProfileDuration {
        Self::new(
            Box::pin(async move { Some(future.await) }),
            TimingMethod::Device,
        )
    }

    /// Create a new `ProfileDuration` from a future that may resolve to no
    /// measurement at all, for a backend that only learns the window carried
    /// none once the device has answered.
    pub fn new_device_time_maybe(
        future: impl Future<Output = Option<ProfileTicks>> + Send + 'static,
    ) -> ProfileDuration {
        Self::new(Box::pin(future), TimingMethod::Device)
    }

    /// Average a measured batch without inventing a duration for an unmeasured window.
    pub fn per_iteration(self, repetitions: core::num::NonZeroU32) -> Self {
        let method = self.method;
        Self::new(
            Box::pin(async move {
                self.resolve().await.map(|ticks| ProfileTicks {
                    start: ticks.start,
                    end: ticks.start + ticks.duration() / repetitions.get(),
                })
            }),
            method,
        )
    }

    /// Retrieve the future that resolves the profile.
    pub fn into_future(self) -> DynFut<Option<ProfileTicks>> {
        self.future
    }

    /// Resolve the actual duration of the profile, possibly by waiting for the future to complete.
    ///
    /// [`None`] when the window carried no measurement. See the type's docs for
    /// why that is not reported as a zero duration.
    pub async fn resolve(self) -> Option<ProfileTicks> {
        self.future.await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;
    use cubecl_environment::future::read_sync;

    #[test]
    fn averages_a_batch_and_preserves_its_start_and_method() {
        let start = Instant::now();
        let duration = ProfileDuration::new_device_time(async move {
            ProfileTicks::from_start_end(start, start + Duration::from_micros(640))
        })
        .per_iteration(NonZeroU32::new(32).unwrap());
        assert_eq!(duration.timing_method(), TimingMethod::Device);
        let ticks = read_sync(duration.resolve()).unwrap();
        assert_eq!(ticks.start_duration_since(start), Duration::ZERO);
        assert_eq!(ticks.duration(), Duration::from_micros(20));
    }

    #[test]
    fn averaging_does_not_turn_a_missing_measurement_into_zero() {
        let duration = ProfileDuration::new_device_time_maybe(async { None })
            .per_iteration(NonZeroU32::new(32).unwrap());
        assert!(read_sync(duration.resolve()).is_none());
    }
}
