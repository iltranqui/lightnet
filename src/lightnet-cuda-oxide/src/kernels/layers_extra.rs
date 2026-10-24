//! Shared device helpers (`dropout_unit`, `marker_position`), the `constrain` and
//! `gradient_centralize` kernels, and the residual epilogue. The per-layer kernels that used
//! to live here are in their own files (`wmhf`, `vit`, `attention`, `moe`, `laya`,
//! `dense_loss`, `softmax`, `reorg`, `channel_shuffle`, `dropout`, `scale_channels`,
//! `shortcut`, `recurrent`, `eml`, `normalize_channels`), with kernel names and signatures
//! unchanged.
//!
//! Every kernel owns its output elements exclusively (no atomics), so results
//! are deterministic.  The host references live in `lightnet_cuda::host_ops`.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Device twin of `lightnet_cuda::dropout_unit`: must stay bit-identical.
#[device]
pub fn dropout_unit(seed: u64, index: u64) -> f32 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0
}

/// Match CPU half-away rounding without f32's (x+0.5) halfway rounding error.
/// Clamp before converting so NaN/infinities cannot poison integer indexing.
#[device]
pub fn marker_position(value: f32, sequence: usize) -> usize {
    let value = value as f64;
    if !(value >= 0.0) {
        0
    } else if value >= (sequence - 1) as f64 {
        sequence - 1
    } else {
        (value + 0.5) as usize
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Darknet `constrain_ongpu`: `out = min(limit, max(-limit, x))`, NaN -> `-limit`.
    #[kernel]
    pub fn constrain(x: &[f32], mut out: DisjointSlice<f32>, limit: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            let mut v = x[i];
            if !(v > -limit) {
                v = -limit;
            }
            if v > limit {
                v = limit;
            }
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Darknet `gradient_centralization_gpu`, one thread per filter of
    /// `f_size` contiguous values (sequential sum, = the host reference); a
    /// plain copy when `enabled == 0`.
    #[kernel]
    pub fn gradient_centralize(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        f_size: usize,
        filters: usize,
        enabled: u32,
    ) {
        let f = thread::index_1d().get();
        if f >= filters {
            return;
        }
        let base = f * f_size;
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < f_size {
            sum += x[base + i];
            i += 1;
        }
        let mean = if enabled != 0 {
            sum / f_size as f32
        } else {
            0.0
        };
        // SAFETY: this thread owns the `f_size` values of filter `f`.
        unsafe {
            let dst = out.as_mut_ptr().add(base);
            let mut i = 0usize;
            while i < f_size {
                *dst.add(i) = x[base + i] - mean;
                i += 1;
            }
        }
    }

}

/// Fused residual + bias epilogue for a transformer block:
/// `out[i] = residual[i] + y[i] + bias[i % cols]`.
///
/// The bias is per **column**: the row index is the token, so it must not shift
/// the offset.  `has_bias = 0` makes it a plain residual add.
#[cuda_module]
pub mod residual {
    use super::*;
    use cuda_device::{kernel, thread};

    #[kernel]
    pub fn residual_bias(
        residual: &[f32],
        y: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        cols: usize,
        has_bias: u32,
    ) {
        let i = thread::index_1d();
        let idx = i.get();
        if idx >= out.len() {
            return;
        }
        let b = if has_bias != 0 { bias[idx % cols] } else { 0.0 };
        if let Some(slot) = out.get_mut(i) {
            *slot = residual[idx] + y[idx] + b;
        }
    }
}
