//! Mixed-precision (AMP) training kernels: elementwise, batch-norm, pooling,
//! upsample and col2im on f16 storage (raw `u16` binary16 bit patterns).
//!
//! Every kernel widens its f16 operands to f32 in registers, does all
//! arithmetic and every reduction in f32 (the per-channel reductions use the
//! same fixed-order two-stage scheme as `batchnorm.rs`, so they are
//! deterministic) and rounds to nearest even only when storing an f16 result.
//! Per-channel statistics, scales, biases and weight gradients stay f32.

use super::activations::{activate, gradient};
use super::convert::{f16_bits_to_f32 as h2f, f32_to_f16_bits};
use super::pool::first_window;
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Must equal `batchnorm.rs` / `darknet_cuda::host_ops::BATCH_NORM_EPSILON`.
const EPSILON: f32 = 0.000_01;

#[device]
fn f2h(x: f32) -> u16 {
    f32_to_f16_bits(x, 0)
}

/// `window_argmax` of `pool.rs` reading f16 (same window rule, strict `>`).
#[device]
fn window_argmax_h(
    x: &[u16],
    plane: usize,
    h: usize,
    w: usize,
    oy: usize,
    ox: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    off: usize,
) -> usize {
    let y0 = (oy * stride_y) as i64 - off as i64;
    let x0 = (ox * stride_x) as i64 - off as i64;
    let mut best = f32::NEG_INFINITY;
    let mut best_i = usize::MAX;
    let mut ky = 0usize;
    while ky < size {
        let iy = y0 + ky as i64;
        if iy >= 0 && iy < h as i64 {
            let mut kx = 0usize;
            while kx < size {
                let ix = x0 + kx as i64;
                if ix >= 0 && ix < w as i64 {
                    let idx = plane + iy as usize * w + ix as usize;
                    let v = h2f(x[idx]);
                    if v > best {
                        best = v;
                        best_i = idx;
                    }
                }
                kx += 1;
            }
        }
        ky += 1;
    }
    best_i
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out = f16(x + bias[c])`, `c = (i / channel_stride) % channels`.
    #[kernel]
    pub fn amp_bias_add(
        x: &[u16],
        bias: &[f32],
        mut out: DisjointSlice<u16>,
        channels: usize,
        channel_stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let c = (i / channel_stride) % channels;
            *o = f2h(h2f(x[i]) + bias[c]);
        }
    }

    #[kernel]
    pub fn amp_activation(x: &[u16], mut out: DisjointSlice<u16>, kind: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f2h(activate(kind, h2f(x[i])));
        }
    }

    /// `out = f16(delta * gradient(x))`, `x` the (f16) activation input.
    #[kernel]
    pub fn amp_activation_backward(
        x: &[u16],
        delta: &[u16],
        mut out: DisjointSlice<u16>,
        kind: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f2h(h2f(delta[i]) * gradient(kind, h2f(x[i])));
        }
    }

    #[kernel]
    pub fn amp_vector_add(x: &[u16], y: &[u16], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f2h(h2f(x[i]) + h2f(y[i]));
        }
    }

    /// Stage 1 of `sum(x)` per channel (f32 partials, 256 per channel).
    #[kernel]
    pub fn amp_sum_partial(
        x: &[u16],
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
                    sum += h2f(x[base + s]);
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// Stage 1 of `sum((x - mean[c])^2)` per channel.
    #[kernel]
    pub fn amp_sq_dev_partial(
        x: &[u16],
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
                    let d = h2f(x[base + s]) - m;
                    sum += d * d;
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// Stage 1 of `sum(a * b)` per channel (both f16).
    #[kernel]
    pub fn amp_dot_partial(
        a: &[u16],
        b: &[u16],
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
                    sum += h2f(a[base + s]) * h2f(b[base + s]);
                    s += 256;
                }
                g += 1;
            }
            *p = sum;
        }
    }

    /// `x_hat = f16((x - mean) / sqrt(var + eps))`; `out = f16(x_hat_f32 * scale + bias)`
    /// (the output is computed from the unrounded f32 `x_hat`).
    #[kernel]
    pub fn amp_bn_normalize(
        x: &[u16],
        mean: &[f32],
        variance: &[f32],
        scale: &[f32],
        bias: &[f32],
        mut x_hat: DisjointSlice<u16>,
        mut out: DisjointSlice<u16>,
        channels: usize,
        channel_stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x.len() {
            return;
        }
        let c = (i / channel_stride) % channels;
        let h = (h2f(x[i]) - mean[c]) / (variance[c] + EPSILON).sqrt();
        if let Some(o) = x_hat.get_mut(idx) {
            *o = f2h(h);
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = f2h(h * scale[c] + bias[c]);
        }
    }

    /// `dx = f16(scale / sqrt(var + eps) / n * (n * dy - sum_dy - x_hat * sum_dy_xhat))`.
    #[kernel]
    pub fn amp_bn_backward_dx(
        dy: &[u16],
        x_hat: &[u16],
        variance: &[f32],
        scale: &[f32],
        sum_dy: &[f32],
        sum_dy_xhat: &[f32],
        mut dx: DisjointSlice<u16>,
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
            *o = f2h(
                scale[c] * inv_std / n * (n * h2f(dy[i]) - sum_dy[c] - h2f(x_hat[i]) * sum_dy_xhat[c]),
            );
        }
    }

    /// Gather col2im: f32 columns (the GEMM accumulates in f32) -> f16 `dx`.
    /// `images` consecutive images, column block `[image][c*k*k][oh*ow]`, written
    /// to `dx[dx_base + ..]`.
    #[kernel]
    pub fn amp_col2im(
        col: &[f32],
        mut dx: DisjointSlice<u16>,
        dx_base: usize,
        images: usize,
        channels: usize,
        h: usize,
        w: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let gi = thread::index_1d().get();
        let plane = channels * h * w;
        if gi >= images * plane || dx_base + gi >= dx.len() {
            return;
        }
        let img = gi / plane;
        let i = gi % plane;
        let c = i / (h * w);
        let rem = i % (h * w);
        let iy = rem / w;
        let ix = rem % w;
        let ohw = out_h * out_w;
        let kk = ksize * ksize;
        let col_base = img * channels * kk * ohw;
        let mut sum = 0.0f32;
        let mut ky = 0usize;
        while ky < ksize {
            let ny = iy as i64 + padding as i64 - (ky * dilation) as i64;
            if ny >= 0 && (ny as usize) % stride_y == 0 {
                let oy = ny as usize / stride_y;
                if oy < out_h {
                    let mut kx = 0usize;
                    while kx < ksize {
                        let nx = ix as i64 + padding as i64 - (kx * dilation) as i64;
                        if nx >= 0 && (nx as usize) % stride_x == 0 {
                            let ox = nx as usize / stride_x;
                            if ox < out_w {
                                sum += col[col_base + (c * kk + ky * ksize + kx) * ohw + oy * out_w + ox];
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        // SAFETY: one thread per input element; `gi < dx.len()`.
        unsafe {
            *dx.as_mut_ptr().add(dx_base + gi) = f2h(sum);
        }
    }

    /// Max-pool backward gather on f16 (same argmax rule as the f32 kernel).
    #[kernel]
    pub fn amp_maxpool_backward(
        x: &[u16],
        dy: &[u16],
        mut dx: DisjointSlice<u16>,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        off: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        let plane = h * w;
        let bc = i / plane;
        let r = i % plane;
        let iy = r / w;
        let ix = r % w;
        let oy_lo = first_window(iy, stride_y, size, off);
        let ox_lo = first_window(ix, stride_x, size, off);
        let mut oy_hi = (iy + off) as i64 / stride_y as i64;
        if oy_hi > out_h as i64 - 1 {
            oy_hi = out_h as i64 - 1;
        }
        let mut ox_hi = (ix + off) as i64 / stride_x as i64;
        if ox_hi > out_w as i64 - 1 {
            ox_hi = out_w as i64 - 1;
        }
        let mut sum = 0.0f32;
        let mut oy = oy_lo;
        while oy <= oy_hi {
            let mut ox = ox_lo;
            while ox <= ox_hi {
                let best = window_argmax_h(
                    x, bc * plane, h, w, oy as usize, ox as usize, size, stride_x, stride_y, off,
                );
                if best == i {
                    sum += h2f(dy[(bc * out_h + oy as usize) * out_w + ox as usize]);
                }
                ox += 1;
            }
            oy += 1;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = f2h(sum);
        }
    }

    /// Nearest-upsample backward (expanding layers): `scale * sum` of the
    /// `stride x stride` output block, f32 sum.
    #[kernel]
    pub fn amp_upsample_backward(
        dy: &[u16],
        mut dx: DisjointSlice<u16>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        let plane = in_h * in_w;
        let bc = i / plane;
        let r = i % plane;
        let iy = r / in_w;
        let ix = r % in_w;
        let mut sum = 0.0f32;
        let mut ky = 0usize;
        while ky < stride {
            let oy = iy * stride + ky;
            if oy < out_h {
                let mut kx = 0usize;
                while kx < stride {
                    let ox = ix * stride + kx;
                    if ox < out_w {
                        sum += h2f(dy[(bc * out_h + oy) * out_w + ox]);
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = f2h(scale * sum);
        }
    }
}
