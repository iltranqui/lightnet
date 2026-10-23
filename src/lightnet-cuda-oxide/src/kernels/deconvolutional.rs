//! Deterministic direct transposed-convolution kernels for NCHW tensors.
//! Weights use Darknet `[input channel, output filter, ky, kx]` layout.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn deconv_fwd(
        x: &[f32],
        weights: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        output_height: usize,
        output_width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= batch * filters * output_height * output_width {
            return;
        }
        let ox = i % output_width;
        let oy = (i / output_width) % output_height;
        let f = (i / (output_width * output_height)) % filters;
        let b = i / (output_width * output_height * filters);
        let mut sum = 0.0f32;
        let mut c = 0usize;
        while c < channels {
            let mut ky = 0usize;
            while ky < kernel_size {
                let iy_num = oy as i64 - ky as i64;
                if iy_num >= 0 && (iy_num as usize) % stride == 0 {
                    let iy = iy_num as usize / stride;
                    if iy < input_height {
                        let mut kx = 0usize;
                        while kx < kernel_size {
                            let ix_num = ox as i64 - kx as i64;
                            if ix_num >= 0 && (ix_num as usize) % stride == 0 {
                                let ix = ix_num as usize / stride;
                                if ix < input_width {
                                    let xi =
                                        ((b * channels + c) * input_height + iy) * input_width + ix;
                                    let wi =
                                        ((c * filters + f) * kernel_size + ky) * kernel_size + kx;
                                    sum += x[xi] * weights[wi];
                                }
                            }
                            kx += 1;
                        }
                    }
                }
                ky += 1;
            }
            c += 1;
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = sum;
        }
    }

    #[kernel]
    pub fn deconv_dgrad(
        output_grad: &[f32],
        weights: &[f32],
        mut input_grad: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        output_height: usize,
        output_width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= batch * channels * input_height * input_width {
            return;
        }
        let ix = i % input_width;
        let iy = (i / input_width) % input_height;
        let c = (i / (input_width * input_height)) % channels;
        let b = i / (input_width * input_height * channels);
        let mut sum = 0.0f32;
        let mut f = 0usize;
        while f < filters {
            let mut ky = 0usize;
            while ky < kernel_size {
                let oy = iy * stride + ky;
                if oy < output_height {
                    let mut kx = 0usize;
                    while kx < kernel_size {
                        let ox = ix * stride + kx;
                        if ox < output_width {
                            let yi = ((b * filters + f) * output_height + oy) * output_width + ox;
                            let wi = ((c * filters + f) * kernel_size + ky) * kernel_size + kx;
                            sum += weights[wi] * output_grad[yi];
                        }
                        kx += 1;
                    }
                }
                ky += 1;
            }
            f += 1;
        }
        if let Some(dst) = input_grad.get_mut(idx) {
            *dst = sum;
        }
    }

    /// One thread owns each weight element and reduces in a stable serial order.
    #[kernel]
    pub fn deconv_wgrad(
        x: &[f32],
        output_grad: &[f32],
        mut weight_grad: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        output_height: usize,
        output_width: usize,
    ) {
        let idx = thread::index_1d();
        let wi = idx.get();
        let weight_count = channels * filters * kernel_size * kernel_size;
        if wi >= weight_count {
            return;
        }
        let kx = wi % kernel_size;
        let ky = (wi / kernel_size) % kernel_size;
        let f = (wi / (kernel_size * kernel_size)) % filters;
        let c = wi / (kernel_size * kernel_size * filters);
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut iy = 0usize;
            while iy < input_height {
                let oy = iy * stride + ky;
                if oy < output_height {
                    let mut ix = 0usize;
                    while ix < input_width {
                        let ox = ix * stride + kx;
                        if ox < output_width {
                            let xi = ((b * channels + c) * input_height + iy) * input_width + ix;
                            let yi = ((b * filters + f) * output_height + oy) * output_width + ox;
                            sum += x[xi] * output_grad[yi];
                        }
                        ix += 1;
                    }
                }
                iy += 1;
            }
            b += 1;
        }
        if let Some(dst) = weight_grad.get_mut(idx) {
            *dst = sum;
        }
    }

    #[kernel]
    pub fn deconv_bgrad(
        output_grad: &[f32],
        mut bias_grad: DisjointSlice<f32>,
        batch: usize,
        filters: usize,
        output_height: usize,
        output_width: usize,
    ) {
        let idx = thread::index_1d();
        let f = idx.get();
        if f >= filters {
            return;
        }
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let start = (b * filters + f) * output_height * output_width;
            let mut p = 0usize;
            while p < output_height * output_width {
                sum += output_grad[start + p];
                p += 1;
            }
            b += 1;
        }
        if let Some(dst) = bias_grad.get_mut(idx) {
            *dst = sum;
        }
    }
}
