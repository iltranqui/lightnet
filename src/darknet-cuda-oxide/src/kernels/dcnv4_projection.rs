//! DCNv4 coarse predictor and lattice interpolation, NHWC offsets/NCHW input.
//! Explicit coarse extents preserve dilated and ceil-divided output geometry.
//! Straightforward correctness kernels; backward scatter uses f32 atomics and
//! is not bitwise deterministic. No performance claim.

use super::dcnv4::dcnv4_atomic_add;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One thread per [batch, coarse_y, coarse_x, padded_offset_channel].
    #[kernel]
    pub fn dcnv4_project(
        input: &[f32],
        weights: &[f32],
        biases: &[f32],
        mut coarse: DisjointSlice<f32>,
        channels: usize,
        in_h: usize,
        in_w: usize,
        coarse_h: usize,
        coarse_w: usize,
        offset_dim: usize,
        size: usize,
        stride_y: usize,
        stride_x: usize,
        padding: usize,
        dilation: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= coarse.len() {
            return;
        }
        let p = i % offset_dim;
        let q = i / offset_dim;
        let cx = q % coarse_w;
        let cy = (q / coarse_w) % coarse_h;
        let b = q / (coarse_h * coarse_w);
        let mut value = 0.0f32;
        let mut c = 0;
        while c < channels {
            let mut ky = 0;
            while ky < size {
                let yy = cy * stride_y + ky * dilation;
                let mut kx = 0;
                while kx < size {
                    let xx = cx * stride_x + kx * dilation;
                    if yy >= padding && xx >= padding && yy - padding < in_h && xx - padding < in_w
                    {
                        let x = ((b * channels + c) * in_h + yy - padding) * in_w + xx - padding;
                        let w = (p * channels + c) * size * size + ky * size + kx;
                        value += input[x] * weights[w];
                    }
                    kx += 1;
                }
                ky += 1;
            }
            c += 1;
        }
        // SAFETY: this thread owns i, checked against the full coarse length.
        unsafe {
            *coarse.as_mut_ptr().add(i) = value + biases[p];
        }
    }

    /// Source lattice coordinates y/d_stride, x/d_stride, replicated border.
    #[kernel]
    pub fn dcnv4_offset_upsample(
        coarse: &[f32],
        mut dense: DisjointSlice<f32>,
        coarse_h: usize,
        coarse_w: usize,
        dense_h: usize,
        dense_w: usize,
        offset_dim: usize,
        d_stride: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= dense.len() {
            return;
        }
        let p = i % offset_dim;
        let q = i / offset_dim;
        let x = q % dense_w;
        let y = (q / dense_w) % dense_h;
        let b = q / (dense_h * dense_w);
        let sy = y as f32 / d_stride as f32;
        let sx = x as f32 / d_stride as f32;
        let y0 = sy as usize;
        let x0 = sx as usize;
        let ly = sy - y0 as f32;
        let lx = sx - x0 as f32;
        let mut value = 0.0f32;
        let mut iy = 0;
        while iy < 2 {
            let yy = if y0 + iy < coarse_h {
                y0 + iy
            } else {
                coarse_h - 1
            };
            let wy = if iy == 0 { 1.0 - ly } else { ly };
            let mut ix = 0;
            while ix < 2 {
                let xx = if x0 + ix < coarse_w {
                    x0 + ix
                } else {
                    coarse_w - 1
                };
                let wx = if ix == 0 { 1.0 - lx } else { lx };
                value += wy * wx * coarse[((b * coarse_h + yy) * coarse_w + xx) * offset_dim + p];
                ix += 1;
            }
            iy += 1;
        }
        // SAFETY: this thread owns i, checked against the full dense length.
        unsafe {
            *dense.as_mut_ptr().add(i) = value;
        }
    }

    /// One dense adjoint scatters to its four clamped coarse neighbors.
    #[kernel]
    pub fn dcnv4_offset_upsample_backward(
        dense_grad: &[f32],
        coarse_grad: *mut f32,
        coarse_h: usize,
        coarse_w: usize,
        dense_h: usize,
        dense_w: usize,
        offset_dim: usize,
        d_stride: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= dense_grad.len() {
            return;
        }
        let p = i % offset_dim;
        let q = i / offset_dim;
        let x = q % dense_w;
        let y = (q / dense_w) % dense_h;
        let b = q / (dense_h * dense_w);
        let sy = y as f32 / d_stride as f32;
        let sx = x as f32 / d_stride as f32;
        let y0 = sy as usize;
        let x0 = sx as usize;
        let ly = sy - y0 as f32;
        let lx = sx - x0 as f32;
        let mut iy = 0;
        while iy < 2 {
            let yy = if y0 + iy < coarse_h {
                y0 + iy
            } else {
                coarse_h - 1
            };
            let wy = if iy == 0 { 1.0 - ly } else { ly };
            let mut ix = 0;
            while ix < 2 {
                let xx = if x0 + ix < coarse_w {
                    x0 + ix
                } else {
                    coarse_w - 1
                };
                let wx = if ix == 0 { 1.0 - lx } else { lx };
                let index = ((b * coarse_h + yy) * coarse_w + xx) * offset_dim + p;
                // SAFETY: caller validates full extents and zeroes the output;
                // all writes are atomic, including replicated border aliases.
                unsafe {
                    dcnv4_atomic_add(coarse_grad.add(index), dense_grad[i] * wy * wx);
                }
                ix += 1;
            }
            iy += 1;
        }
    }

    /// Adjoint of project; adds predictor d_input to aggregation d_input.
    /// Weight/bias gradients are positive sums (no batch/optimizer scaling).
    #[kernel]
    pub fn dcnv4_project_backward(
        input: &[f32],
        weights: &[f32],
        coarse_grad: &[f32],
        d_input: *mut f32,
        d_weights: *mut f32,
        d_biases: *mut f32,
        channels: usize,
        in_h: usize,
        in_w: usize,
        coarse_h: usize,
        coarse_w: usize,
        offset_dim: usize,
        size: usize,
        stride_y: usize,
        stride_x: usize,
        padding: usize,
        dilation: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= coarse_grad.len() {
            return;
        }
        let p = i % offset_dim;
        let q = i / offset_dim;
        let cx = q % coarse_w;
        let cy = (q / coarse_w) % coarse_h;
        let b = q / (coarse_h * coarse_w);
        let grad = coarse_grad[i];
        // SAFETY: padded channel p is in bounds; all writers are atomic.
        unsafe {
            dcnv4_atomic_add(d_biases.add(p), grad);
        }
        let mut c = 0;
        while c < channels {
            let mut ky = 0;
            while ky < size {
                let yy = cy * stride_y + ky * dilation;
                let mut kx = 0;
                while kx < size {
                    let xx = cx * stride_x + kx * dilation;
                    if yy >= padding && xx >= padding && yy - padding < in_h && xx - padding < in_w
                    {
                        let x = ((b * channels + c) * in_h + yy - padding) * in_w + xx - padding;
                        let w = (p * channels + c) * size * size + ky * size + kx;
                        // SAFETY: launcher validates all extents, output buffers
                        // are distinct and all scatter writes are atomic.
                        unsafe {
                            dcnv4_atomic_add(d_input.add(x), grad * weights[w]);
                            dcnv4_atomic_add(d_weights.add(w), grad * input[x]);
                        }
                    }
                    kx += 1;
                }
                ky += 1;
            }
            c += 1;
        }
    }
}
