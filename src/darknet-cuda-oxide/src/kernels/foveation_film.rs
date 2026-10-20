//! Fused per-sample FiLM channel affine and adjoint.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `y[b,c,s] = x[b,c,s] * (1 + gamma[b,c]) + beta[b,c]`.
    #[kernel]
    pub fn film_affine(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
    ) {
        let i = thread::index_1d().get();
        if i < x.len() {
            let channel = i / spatial % channels;
            let batch_channel = i / (channels * spatial) * channels + channel;
            if let Some(value) = output.get_mut(thread::index_1d()) {
                *value = x[i] * (1.0 + gamma[batch_channel]) + beta[batch_channel];
            }
        }
    }

    /// `d_input = d_output * (1 + gamma)`, one owned output per element.
    #[kernel]
    pub fn film_affine_dx(
        dy: &[f32],
        gamma: &[f32],
        mut dx: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
    ) {
        let i = thread::index_1d().get();
        if i < dy.len() {
            let channel = i / spatial % channels;
            let batch_channel = i / (channels * spatial) * channels + channel;
            if let Some(value) = dx.get_mut(thread::index_1d()) {
                *value = dy[i] * (1.0 + gamma[batch_channel]);
            }
        }
    }

    /// Spatially reduce `d_gamma = sum(dy*x)` and `d_beta = sum(dy)`.
    /// One thread owns each `(batch, channel)` reduction result.
    #[kernel]
    pub fn film_affine_param_grads(
        x: &[f32],
        dy: &[f32],
        mut dgamma: DisjointSlice<f32>,
        mut dbeta: DisjointSlice<f32>,
        spatial: usize,
    ) {
        let bc = thread::index_1d();
        let idx = bc.get();
        if idx < dgamma.len() {
            let offset = idx * spatial;
            let mut scale_sum = 0.0f32;
            let mut bias_sum = 0.0f32;
            let mut s = 0usize;
            while s < spatial {
                scale_sum += dy[offset + s] * x[offset + s];
                bias_sum += dy[offset + s];
                s += 1;
            }
            if let Some(value) = dgamma.get_mut(bc) {
                *value = scale_sum;
            }
            if let Some(value) = dbeta.get_mut(thread::index_1d()) {
                *value = bias_sum;
            }
        }
    }
}
