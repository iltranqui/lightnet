//! ScaleChannels forward and gradient kernels.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[i] = scale[si(i)] * src[i]`, `si` = `[batch, hw]` (`scale_wh != 0`)
    /// or `[batch, channels]` index of element `i`.
    #[kernel]
    pub fn scale_channels_forward(
        scale: &[f32],
        src: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = out.get_mut(idx) {
            *o = scale[si] * src[i];
        }
    }

    /// `d_source[i] = delta[i] * scale[si(i)]`.
    #[kernel]
    pub fn scale_channels_dsource(
        delta: &[f32],
        scale: &[f32],
        mut d_src: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 {
            b * hw + r % hw
        } else {
            b * channels + r / hw
        };
        if let Some(o) = d_src.get_mut(idx) {
            *o = delta[i] * scale[si];
        }
    }

    /// `d_scale[j] = sum delta * src` over the elements sharing scale `j`;
    /// one thread per scale element, fixed summation order.
    #[kernel]
    pub fn scale_channels_dscale(
        delta: &[f32],
        src: &[f32],
        mut d_scale: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= d_scale.len() {
            return;
        }
        let mut sum = 0.0f32;
        if scale_wh != 0 {
            let b = j / hw;
            let s = j % hw;
            let mut c = 0usize;
            while c < channels {
                let i = (b * channels + c) * hw + s;
                sum += delta[i] * src[i];
                c += 1;
            }
        } else {
            let base = j * hw;
            let mut s = 0usize;
            while s < hw {
                sum += delta[base + s] * src[base + s];
                s += 1;
            }
        }
        if let Some(o) = d_scale.get_mut(idx) {
            *o = sum;
        }
    }
}
