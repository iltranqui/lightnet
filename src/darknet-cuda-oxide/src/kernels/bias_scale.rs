//! Per-channel bias / scale kernels and the bias-gradient reduction.
//!
//! Layout: tensors are `[rows, channel_stride]` flattened, and element `i`
//! belongs to channel `(i / channel_stride) % channels`.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Threads per channel in the bias-gradient partial sums.
pub const BIAS_BACKWARD_THREADS: usize = 256;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// out[i] = x[i] + bias[channel(i)]
    #[kernel]
    pub fn bias_add(
        x: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i] + bias[(i / channel_stride) % channels];
        }
    }

    /// out[i] = activate(kind, x[i] + bias[channel(i)])  (bias_add + activation fused).
    #[kernel]
    pub fn bias_activation(
        x: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
        kind: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let v = x[i] + bias[(i / channel_stride) % channels];
            *o = crate::kernels::activations::activate(kind, v);
        }
    }

    /// out[i] = x[i] * scale[channel(i)]
    #[kernel]
    pub fn channel_scale(
        x: &[f32],
        scale: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        channel_stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i] * scale[(i / channel_stride) % channels];
        }
    }

    /// Stage 1 of the bias gradient: block `c` owns channel `c`, thread `t`
    /// writes its strided partial sum to `partial[c * 256 + t]`.  Avoids
    /// atomics (deterministic) and shared memory.
    #[kernel]
    pub fn bias_backward_partial(
        grad: &[f32],
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
                    sum += grad[base + s];
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// Stage 2: one thread per channel folds its 256 partial sums.
    #[kernel]
    pub fn bias_backward_reduce(partial: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let c = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut t = 0usize;
            while t < 256 {
                sum += partial[c * 256 + t];
                t += 1;
            }
            *o = sum;
        }
    }
}
