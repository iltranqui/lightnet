//! Opt-in phase timers for the detector training data loader.
//!
//! This is a diagnostic aid for the training throughput work, not part of the
//! runtime contract.  Nothing is measured unless a benchmark/test calls
//! [`enable`] (or the CLI calls [`enable_from_env`]).
//!
//! Timers use relaxed atomics.  Under a parallel loader a few concurrent
//! updates may be lost, which is acceptable for a coarse performance probe and
//! avoids any synchronisation on the hot path.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

static ENABLED: AtomicBool = AtomicBool::new(false);

static SAMPLES: AtomicU64 = AtomicU64::new(0);
static LOADS: AtomicU64 = AtomicU64::new(0);
static READ_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_NS: AtomicU64 = AtomicU64::new(0);
static PLANAR_NS: AtomicU64 = AtomicU64::new(0);
static LABELS_NS: AtomicU64 = AtomicU64::new(0);
static AUGMENT_NS: AtomicU64 = AtomicU64::new(0);
static COPY_NS: AtomicU64 = AtomicU64::new(0);
static TO_RGB8_NS: AtomicU64 = AtomicU64::new(0);
static MEAN_NS: AtomicU64 = AtomicU64::new(0);
static RESIZE_NS: AtomicU64 = AtomicU64::new(0);
static HSV_NS: AtomicU64 = AtomicU64::new(0);
static ENCODE_NS: AtomicU64 = AtomicU64::new(0);

/// Raw accumulated loader timings (all values in nanoseconds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoaderProfileSnapshot {
    /// Number of training samples (augmented images) produced.
    pub samples: u64,
    /// Number of source images decoded (mosaic/mixup can exceed `samples`).
    pub loads: u64,
    pub read_ns: u64,
    pub decode_ns: u64,
    pub planar_ns: u64,
    pub labels_ns: u64,
    pub augment_ns: u64,
    pub copy_ns: u64,
    /// Augmentation internals (all inside `augment_ns`).
    pub to_rgb8_ns: u64,
    pub mean_ns: u64,
    pub resize_ns: u64,
    pub hsv_ns: u64,
    pub encode_ns: u64,
}

impl LoaderProfileSnapshot {
    /// `fs::read` + image decode + u8->f32 planar conversion.
    pub fn load_ns(&self) -> u64 {
        self.read_ns + self.decode_ns + self.planar_ns
    }

    /// Everything the loader spent per produced sample except the caller-owned
    /// `encode`/prefetch plumbing.
    pub fn total_ns(&self) -> u64 {
        self.load_ns() + self.labels_ns + self.augment_ns + self.copy_ns
    }

    fn ms_per(&self, total: u64) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            total as f64 / self.samples as f64 / 1.0e6
        }
    }

    /// Per-sample breakdown in milliseconds (read/decode/planar/labels/augment/copy/load/total).
    pub fn per_sample_ms(&self) -> [f64; 8] {
        [
            self.ms_per(self.read_ns),
            self.ms_per(self.decode_ns),
            self.ms_per(self.planar_ns),
            self.ms_per(self.labels_ns),
            self.ms_per(self.augment_ns),
            self.ms_per(self.copy_ns),
            self.ms_per(self.load_ns()),
            self.ms_per(self.total_ns()),
        ]
    }

    pub fn load_ms_per_image(&self) -> f64 {
        if self.loads == 0 {
            0.0
        } else {
            self.load_ns() as f64 / self.loads as f64 / 1.0e6
        }
    }

    /// Per-sample augmentation internals in milliseconds
    /// (to_rgb8/mean/resize/hsv/encode).
    pub fn per_sample_augment_ms(&self) -> [f64; 5] {
        [
            self.ms_per(self.to_rgb8_ns),
            self.ms_per(self.mean_ns),
            self.ms_per(self.resize_ns),
            self.ms_per(self.hsv_ns),
            self.ms_per(self.encode_ns),
        ]
    }
}

/// Enable profiling for the remainder of the process.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Enable profiling when `DARKNET_LOADER_PROFILE=1`.
pub fn enable_from_env() {
    if std::env::var("DARKNET_LOADER_PROFILE").as_deref() == Ok("1") {
        enable();
    }
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub fn reset() {
    for counter in [
        &SAMPLES,
        &LOADS,
        &READ_NS,
        &DECODE_NS,
        &PLANAR_NS,
        &LABELS_NS,
        &AUGMENT_NS,
        &COPY_NS,
        &TO_RGB8_NS,
        &MEAN_NS,
        &RESIZE_NS,
        &HSV_NS,
        &ENCODE_NS,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

pub fn snapshot() -> LoaderProfileSnapshot {
    LoaderProfileSnapshot {
        samples: SAMPLES.load(Ordering::Relaxed),
        loads: LOADS.load(Ordering::Relaxed),
        read_ns: READ_NS.load(Ordering::Relaxed),
        decode_ns: DECODE_NS.load(Ordering::Relaxed),
        planar_ns: PLANAR_NS.load(Ordering::Relaxed),
        labels_ns: LABELS_NS.load(Ordering::Relaxed),
        augment_ns: AUGMENT_NS.load(Ordering::Relaxed),
        copy_ns: COPY_NS.load(Ordering::Relaxed),
        to_rgb8_ns: TO_RGB8_NS.load(Ordering::Relaxed),
        mean_ns: MEAN_NS.load(Ordering::Relaxed),
        resize_ns: RESIZE_NS.load(Ordering::Relaxed),
        hsv_ns: HSV_NS.load(Ordering::Relaxed),
        encode_ns: ENCODE_NS.load(Ordering::Relaxed),
    }
}

/// Snapshot and reset in one call.
pub fn take() -> LoaderProfileSnapshot {
    let snapshot = snapshot();
    reset();
    snapshot
}

#[inline]
fn nanos(duration: Duration) -> u64 {
    duration.as_nanos() as u64
}

pub fn add_sample() {
    if enabled() {
        SAMPLES.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn add_load() {
    if enabled() {
        LOADS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn add_read(duration: Duration) {
    READ_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_decode(duration: Duration) {
    DECODE_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_planar(duration: Duration) {
    PLANAR_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_labels(duration: Duration) {
    LABELS_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_augment(duration: Duration) {
    AUGMENT_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_copy(duration: Duration) {
    COPY_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_to_rgb8(duration: Duration) {
    TO_RGB8_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_mean(duration: Duration) {
    MEAN_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_resize(duration: Duration) {
    RESIZE_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_hsv(duration: Duration) {
    HSV_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}

pub fn add_encode(duration: Duration) {
    ENCODE_NS.fetch_add(nanos(duration), Ordering::Relaxed);
}
