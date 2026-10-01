//! Batch-norm training kernels (batch statistics, normalise + affine, and the
//! exact input gradient).  Deterministic: every per-channel reduction is a
//! fixed-order two-stage sum (256 strided partials per channel, then a serial
//! fold by `bias_scale::bias_backward_reduce`), never a float atomic.
//!
//! Layout: NCHW flattened; element `i` belongs to channel
//! `(i / channel_stride) % channels`.  Semantics follow the Rust CPU runtime
//! (`backward_batchnorm`) and `src-lib/blas_kernels.cu`; the host references
//! live in `darknet_cuda::host_ops`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Must equal `darknet_cuda::host_ops::BATCH_NORM_EPSILON`.
const EPSILON: f32 = 0.000_01;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Stage 1 of `sum(x)` per channel: block `c`, thread `t` writes its
    /// strided partial to `partial[c * 256 + t]`.
    #[kernel]
    pub fn bn_sum_partial(
        x: &[f32],
        mut partial: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
        row_groups: usize,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get();
        let c = gid / 256;
        let t = gid % 256;
        if let Some(p) = partial.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut g = 0usize;
            while g < row_groups {
                let base = (g * channels + c) * channel_stride;
                let mut s = t;
                while s < channel_stride {
                    sum += x[base + s];
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// Stage 1 of `sum((x - mean[c])^2)` per channel.
    #[kernel]
    pub fn bn_sq_dev_partial(
        x: &[f32],
        mean: &[f32],
        mut partial: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
        row_groups: usize,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get();
        let c = gid / 256;
        let t = gid % 256;
        if let Some(p) = partial.get_mut(idx) {
            let m = mean[c];
            let mut sum = 0.0f32;
            let mut g = 0usize;
            while g < row_groups {
                let base = (g * channels + c) * channel_stride;
                let mut s = t;
                while s < channel_stride {
                    let d = x[base + s] - m;
                    sum += d * d;
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// Stage 1 of `sum(a * b)` per channel (the `dscale` reduction).
    #[kernel]
    pub fn bn_dot_partial(
        a: &[f32],
        b: &[f32],
        mut partial: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
        row_groups: usize,
    ) {
        let idx = thread::index_1d();
        let gid = idx.get();
        let c = gid / 256;
        let t = gid % 256;
        if let Some(p) = partial.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut g = 0usize;
            while g < row_groups {
                let base = (g * channels + c) * channel_stride;
                let mut s = t;
                while s < channel_stride {
                    sum += a[base + s] * b[base + s];
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// `mean[c] = sum[c] / n`.
    #[kernel]
    pub fn bn_mean_finalize(sum: &[f32], mut mean: DisjointSlice<f32>, inv_n: f32) {
        let idx = thread::index_1d();
        let c = idx.get();
        if let Some(m) = mean.get_mut(idx) {
            *m = sum[c] * inv_n;
        }
    }

    /// Biased batch variance (used to normalise) and the rolling-stat EMA
    /// `new = keep * old + (1 - keep) * batch`; the rolling variance takes the
    /// unbiased estimate (`sum_sq / (n - 1)`).
    #[kernel]
    pub fn bn_var_finalize(
        sum_sq: &[f32],
        mean: &[f32],
        rolling_mean: &[f32],
        rolling_variance: &[f32],
        mut variance: DisjointSlice<f32>,
        mut new_rolling_mean: DisjointSlice<f32>,
        mut new_rolling_variance: DisjointSlice<f32>,
        inv_n: f32,
        inv_n_unbiased: f32,
        keep: f32,
    ) {
        let idx = thread::index_1d();
        let c = idx.get();
        if c >= sum_sq.len() {
            return;
        }
        let update = 1.0 - keep;
        if let Some(v) = variance.get_mut(idx) {
            *v = sum_sq[c] * inv_n;
        }
        if let Some(r) = new_rolling_mean.get_mut(thread::index_1d()) {
            *r = keep * rolling_mean[c] + update * mean[c];
        }
        if let Some(r) = new_rolling_variance.get_mut(thread::index_1d()) {
            *r = keep * rolling_variance[c] + update * (sum_sq[c] * inv_n_unbiased);
        }
    }

    /// `x_hat = (x - mean) / sqrt(var + eps)`, `out = x_hat * scale + bias`.
    #[kernel]
    pub fn bn_normalize(
        x: &[f32],
        mean: &[f32],
        variance: &[f32],
        scale: &[f32],
        bias: &[f32],
        mut x_hat: DisjointSlice<f32>,
        mut out: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x.len() {
            return;
        }
        let c = (i / channel_stride) % channels;
        let h = (x[i] - mean[c]) / (variance[c] + EPSILON).sqrt();
        if let Some(o) = x_hat.get_mut(idx) {
            *o = h;
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = h * scale[c] + bias[c];
        }
    }

    /// `dx = scale / sqrt(var + eps) / n * (n * dy - sum_dy - x_hat * sum_dy_xhat)`.
    #[kernel]
    pub fn bn_backward_dx(
        dy: &[f32],
        x_hat: &[f32],
        variance: &[f32],
        scale: &[f32],
        sum_dy: &[f32],
        sum_dy_xhat: &[f32],
        mut dx: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
        n: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dy.len() {
            return;
        }
        let c = (i / channel_stride) % channels;
        let inv_std = 1.0 / (variance[c] + EPSILON).sqrt();
        if let Some(o) = dx.get_mut(idx) {
            *o = scale[c] * inv_std / n * (n * dy[i] - sum_dy[c] - x_hat[i] * sum_dy_xhat[c]);
        }
    }
}
