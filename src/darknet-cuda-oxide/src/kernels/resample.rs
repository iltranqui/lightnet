//! Upsample (nearest / reverse) and bilinear crop.
//!
//! Upsample follows `src-lib/blas_kernels.cu::upsample_kernel` and the
//! `darknet-core` reference.  The forward direction of a `reverse` layer is a
//! strided sum; it is written as a gather per output element (the C++ kernel
//! uses `atomicAdd`), so both directions are deterministic.
//!
//! Crop bilinearly samples per-tap affine-adjusted values while out-of-image
//! taps remain zero, matching the Rust CPU sampler.  Its backward is a scatter
//! with arbitrary source coordinates and therefore uses float atomics
//! (summation order, not the result set, is nondeterministic).

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// `floor(v)` as i64 for coordinates that are far from i64 limits.
#[device]
pub fn floor_i64(v: f32) -> i64 {
    let t = v as i64;
    if (t as f32) > v { t - 1 } else { t }
}

/// Relaxed device-scope `*p += v`.
///
/// # Safety
/// `p` must be a live, aligned device f32 that is only accessed atomically
/// while the kernel runs.
#[device]
pub unsafe fn atomic_add_f32(p: *mut f32, v: f32) {
    let cell = unsafe { DeviceAtomicF32::from_ptr(p) };
    let _old = cell.fetch_add(v, AtomicOrdering::Relaxed);
}

#[device]
fn crop_hsv_component(r: f32, g: f32, b: f32, saturation_scale: f32, exposure_scale: f32, channel: usize) -> f32 {
    let maximum = r.max(g).max(b);
    let minimum = r.min(g).min(b);
    let delta = maximum - minimum;
    let hue = if maximum == 0.0 {
        -1.0
    } else if delta == 0.0 {
        0.0
    } else if r == maximum {
        (g - b) / delta + if g < b { 6.0 } else { 0.0 }
    } else if g == maximum {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };
    let hsv_saturation = if maximum == 0.0 { 0.0 } else { delta / maximum };
    let hsv_saturation = hsv_saturation * saturation_scale;
    let value = maximum * exposure_scale;
    let converted = if hsv_saturation == 0.0 {
        value
    } else {
        let index = hue as i32;
        let fraction = hue - index as f32;
        let p = value * (1.0 - hsv_saturation);
        let q = value * (1.0 - hsv_saturation * fraction);
        let t = value * (1.0 - hsv_saturation * (1.0 - fraction));
        match (index, channel) {
            (0, 0) => value,
            (0, 1) => t,
            (0, _) => p,
            (1, 0) => q,
            (1, 1) => value,
            (1, _) => p,
            (2, 0) => p,
            (2, 1) => value,
            (2, _) => t,
            (3, 0) => p,
            (3, 1) => q,
            (3, _) => value,
            (4, 0) => t,
            (4, 1) => p,
            (4, _) => value,
            (_, 0) => value,
            (_, 1) => p,
            (_, _) => q,
        }
    };
    converted.max(0.0).min(1.0)
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[bc, oy, ox] = scale * x[bc, oy / stride, ox / stride]`.
    #[kernel]
    pub fn upsample_nearest(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let bc = i / oplane;
        let r = i % oplane;
        let iy = (r / out_w) / stride;
        let ix = (r % out_w) / stride;
        if let Some(o) = out.get_mut(idx) {
            *o = scale * x[bc * in_h * in_w + iy * in_w + ix];
        }
    }

    /// Forward of a `reverse` upsample layer: each small output pixel sums the
    /// `stride x stride` block of the big input it covers (taps past the input
    /// edge are skipped, which also covers `out = floor(in / stride)` where
    /// the C++ kernel ignores the ragged tail).
    #[kernel]
    pub fn upsample_reverse(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let bc = i / oplane;
        let r = i % oplane;
        let oy = r / out_w;
        let ox = r % out_w;
        let mut sum = 0.0f32;
        let mut dy = 0usize;
        while dy < stride {
            let iy = oy * stride + dy;
            if iy < in_h {
                let mut dx = 0usize;
                while dx < stride {
                    let ix = ox * stride + dx;
                    if ix < in_w {
                        sum += x[bc * in_h * in_w + iy * in_w + ix];
                    }
                    dx += 1;
                }
            }
            dy += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = scale * sum;
        }
    }

    /// Gradient w.r.t. the upsample input, one thread per input element.
    /// `reverse == 0`: `scale * sum` of the `stride x stride` output block.
    /// `reverse != 0` (layer ran in reverse): `scale * dy[y/stride, x/stride]`,
    /// or 0 where that output pixel does not exist.
    #[kernel]
    pub fn upsample_backward(
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
        reverse: u32,
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
        let mut v = 0.0f32;
        if reverse != 0 {
            let oy = iy / stride;
            let ox = ix / stride;
            if oy < out_h && ox < out_w {
                v = scale * dy[(bc * out_h + oy) * out_w + ox];
            }
        } else {
            let mut sum = 0.0f32;
            let mut ky = 0usize;
            while ky < stride {
                let oy = iy * stride + ky;
                if oy < out_h {
                    let mut kx = 0usize;
                    while kx < stride {
                        let ox = ix * stride + kx;
                        if ox < out_w {
                            sum += dy[(bc * out_h + oy) * out_w + ox];
                        }
                        kx += 1;
                    }
                }
                ky += 1;
            }
            v = scale * sum;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = v;
        }
    }

    /// Bilinear crop, one thread per output element.  `source_{x,y}` hold one
    /// coordinate per (batch, output pixel); `scales` / `offsets` one per batch.
    #[kernel]
    pub fn crop_bilinear(
        x: &[f32],
        source_x: &[f32],
        source_y: &[f32],
        scales: &[f32],
        offsets: &[f32],
        saturations: &[f32],
        exposures: &[f32],
        apply_hsv: bool,
        mut out: DisjointSlice<f32>,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let b = i / (channels * oplane);
        let rem = i % (channels * oplane);
        let c = rem / oplane;
        let p = rem % oplane;
        let coord = b * oplane + p;
        let sx = source_x[coord];
        let sy = source_y[coord];
        let ix = floor_i64(sx);
        let iy = floor_i64(sy);
        let dx = sx - ix as f32;
        let dy = sy - iy as f32;
        let base = (b * channels + c) * in_h * in_w;
        let mut value = 0.0f32;
        let mut t = 0usize;
        while t < 4 {
            // t bit0 -> x + 1, bit1 -> y + 1; weights as in the C++ sampler.
            let tx = ix + (t & 1) as i64;
            let ty = iy + (t >> 1) as i64;
            let wx = if t & 1 == 1 { dx } else { 1.0 - dx };
            let wy = if t >> 1 == 1 { dy } else { 1.0 - dy };
            if tx >= 0 && tx < in_w as i64 && ty >= 0 && ty < in_h as i64 {
                let spatial = ty as usize * in_w + tx as usize;
                let pixel = base + spatial;
                let image_base = b * channels * in_h * in_w;
                let raw = if apply_hsv && channels >= 3 && c < 3 {
                    crop_hsv_component(
                        x[image_base + spatial],
                        x[image_base + in_h * in_w + spatial],
                        x[image_base + 2 * in_h * in_w + spatial],
                        saturations[b],
                        exposures[b],
                        c,
                    )
                } else {
                    x[pixel]
                };
                let sample = raw * scales[b] + offsets[b * channels + c];
                value += wy * wx * sample;
            }
            t += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = value;
        }
    }

    /// Scatter of the crop gradient into the input (`dx` must be zeroed).
    /// `dx` is a raw device pointer because the updates are atomic adds.
    /// Tile-average NCHW source input into the HF predictor's pooled layout.
    #[kernel]
    pub fn hyperbolic_predictor_pool(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        factor: usize,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= output.len() {
            return;
        }
        let output_spatial = output_height * output_width;
        let batch = i / (channels * output_spatial);
        let rem = i % (channels * output_spatial);
        let channel = rem / output_spatial;
        let pixel = rem % output_spatial;
        let oy = pixel / output_width;
        let ox = pixel % output_width;
        let base = (batch * channels + channel) * input_height * input_width;
        let mut sum = 0.0f32;
        let mut samples = 0usize;
        let mut ky = 0usize;
        while ky < factor {
            let y = oy * factor + ky;
            let mut kx = 0usize;
            while kx < factor {
                let x = ox * factor + kx;
                if y < input_height && x < input_width {
                    sum += input[base + y * input_width + x];
                    samples += 1;
                }
                kx += 1;
            }
            ky += 1;
        }
        if let Some(value) = output.get_mut(index) {
            *value = if samples == 0 { 0.0 } else { sum / samples as f32 };
        }
    }

    /// Apply the source constraining functions to raw per-sample HF predictor values.
    #[kernel]
    pub fn hyperbolic_predictor_constrain(
        raw: &[f32],
        mut output: DisjointSlice<f32>,
        param_count: usize,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= output.len() {
            return;
        }
        let parameter = i % param_count;
        let value = raw[i];
        let constrained = if parameter < 2 {
            0.9 * value.tanh()
        } else if parameter + 1 == param_count {
            if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp = value.exp();
                exp / (1.0 + exp)
            }
        } else {
            let softplus = if value > 20.0 { value } else { (1.0 + value.exp()).ln() };
            (2.0 * softplus + 0.1).max(0.1).min(2.0)
        };
        if let Some(output) = output.get_mut(index) {
            *output = constrained;
        }
    }

    /// Build the inverse sampling grid for isotropic Newton HF mode.
    #[kernel]
    pub fn hyperbolic_newton_grid(
        params: &[f32],
        mut grid: DisjointSlice<f32>,
        height: usize,
        width: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= grid.len() {
            return;
        }
        let spatial = height * width;
        let pixel = i / 2;
        let batch = pixel / spatial;
        let p = pixel % spatial;
        let ox = p % width;
        let oy = p / width;
        let qx = if width > 1 { 2.0 * ox as f32 / (width - 1) as f32 - 1.0 } else { 0.0 };
        let qy = if height > 1 { 2.0 * oy as f32 / (height - 1) as f32 - 1.0 } else { 0.0 };
        let cx = params[batch * 4].max(-0.9).min(0.9);
        let cy = params[batch * 4 + 1].max(-0.9).min(0.9);
        let radius = params[batch * 4 + 2].max(0.1).min(2.0).max(1.0e-4);
        let strength = params[batch * 4 + 3].max(0.0).min(1.0);
        let alpha = alpha.max(1.0e-4);
        let p_exp = p_exp.max(1.0e-4);
        let mut px = qx.max(-1.0).min(1.0);
        let mut py = qy.max(-1.0).min(1.0);
        let mut iteration = 0usize;
        while iteration < iterations {
            let dx = px - cx;
            let dy = py - cy;
            let r = (dx * dx + dy * dy).sqrt();
            let (u, du_dr) = if r > 1.0e-6 {
                let ar = alpha * r;
                let tanh = ar.tanh();
                let sech2 = 1.0 - tanh * tanh;
                (tanh / r, (alpha * sech2 * r - tanh) / (r * r))
            } else {
                (alpha, 0.0)
            };
            let inside = r < radius;
            let (weight, dw_dr) = if inside {
                let base = (1.0 - r / radius).max(0.0);
                let weight = base.powf(p_exp);
                let dw = if r > 1.0e-6 {
                    -(p_exp / radius) * base.powf((p_exp - 1.0).max(0.0))
                } else {
                    0.0
                };
                (weight, dw)
            } else {
                (0.0, 0.0)
            };
            let gain = 1.0 + strength * weight * (u - 1.0);
            let dg_dr = strength * (dw_dr * (u - 1.0) + weight * du_dr);
            let mut j00 = gain;
            let mut j01 = 0.0;
            let mut j10 = 0.0;
            let mut j11 = gain;
            if r > 1.0e-6 {
                let d_gain_over_r = dg_dr / r;
                j00 += d_gain_over_r * dx * dx;
                j01 += d_gain_over_r * dx * dy;
                j10 += d_gain_over_r * dy * dx;
                j11 += d_gain_over_r * dy * dy;
            }
            let mapped_x = cx + gain * dx;
            let mapped_y = cy + gain * dy;
            let residual_x = qx - mapped_x;
            let residual_y = qy - mapped_y;
            let determinant = j00 * j11 - j01 * j10;
            if determinant.abs() < 1.0e-8 {
                px = (px + eta * residual_x).max(-1.0).min(1.0);
                py = (py + eta * residual_y).max(-1.0).min(1.0);
            } else {
                let step_x = (j11 * residual_x - j01 * residual_y) / determinant;
                let step_y = (-j10 * residual_x + j00 * residual_y) / determinant;
                px = (px + eta * step_x).max(-1.0).min(1.0);
                py = (py + eta * step_y).max(-1.0).min(1.0);
            }
            iteration += 1;
        }
        let value = if i & 1 == 0 {
            (px + 1.0) * 0.5 * width.saturating_sub(1) as f32
        } else {
            (py + 1.0) * 0.5 * height.saturating_sub(1) as f32
        };
        if let Some(output) = grid.get_mut(index) {
            *output = value;
        }
    }

    /// Build the inverse sampling grid for isotropic fixed-point HF mode.
    /// One thread owns one scalar coordinate; x/y are recomputed deterministically.
    #[kernel]
    pub fn hyperbolic_fixed_point_grid(
        params: &[f32],
        mut grid: DisjointSlice<f32>,
        height: usize,
        width: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
    ) {
        let index = thread::index_1d();
        let i = index.get();
        if i >= grid.len() {
            return;
        }
        let spatial = height * width;
        let pixel = i / 2;
        let batch = pixel / spatial;
        let p = pixel % spatial;
        let ox = p % width;
        let oy = p / width;
        let qx = if width > 1 {
            2.0 * ox as f32 / (width - 1) as f32 - 1.0
        } else {
            0.0
        };
        let qy = if height > 1 {
            2.0 * oy as f32 / (height - 1) as f32 - 1.0
        } else {
            0.0
        };
        let cx = params[batch * 4].max(-0.9).min(0.9);
        let cy = params[batch * 4 + 1].max(-0.9).min(0.9);
        let radius = params[batch * 4 + 2].max(0.1).min(2.0);
        let strength = params[batch * 4 + 3].max(0.0).min(1.0);
        let mut px = qx.max(-1.0).min(1.0);
        let mut py = qy.max(-1.0).min(1.0);
        let safe_alpha = alpha.max(1.0e-4);
        let safe_exp = p_exp.max(1.0e-4);
        let mut step = 0usize;
        while step < iterations {
            let dx = px - cx;
            let dy = py - cy;
            let radius_from_center = (dx * dx + dy * dy).sqrt();
            let u = if radius_from_center > 1.0e-6 {
                (safe_alpha * radius_from_center).tanh() / radius_from_center
            } else {
                safe_alpha
            };
            let weight = if radius_from_center < radius {
                (1.0 - radius_from_center / radius).max(0.0).powf(safe_exp)
            } else {
                0.0
            };
            let gain = 1.0 + strength * weight * (u - 1.0);
            px = (px + eta * (qx - (cx + gain * dx))).max(-1.0).min(1.0);
            py = (py + eta * (qy - (cy + gain * dy))).max(-1.0).min(1.0);
            step += 1;
        }
        let value = if i & 1 == 0 {
            (px + 1.0) * 0.5 * width.saturating_sub(1) as f32
        } else {
            (py + 1.0) * 0.5 * height.saturating_sub(1) as f32
        };
        if let Some(output) = grid.get_mut(index) {
            *output = value;
        }
    }

    /// Gather an NCHW tensor through an HF sampling grid in pixel coordinates.
    /// Out-of-bounds taps contribute zero, matching the CPU HF bilinear sampler.
    #[kernel]
    pub fn hyperbolic_warp(
        x: &[f32],
        grid: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= output.len() {
            return;
        }
        let spatial = height * width;
        let batch = i / (channels * spatial);
        let channel = (i / spatial) % channels;
        let pixel = i % spatial;
        let grid_index = (batch * spatial + pixel) * 2;
        let sx = grid[grid_index];
        let sy = grid[grid_index + 1];
        let ix = floor_i64(sx);
        let iy = floor_i64(sy);
        let fx = sx - ix as f32;
        let fy = sy - iy as f32;
        let input_base = (batch * channels + channel) * spatial;
        let mut value = 0.0f32;
        let mut tap = 0usize;
        while tap < 4 {
            let tx = ix + (tap & 1) as i64;
            let ty = iy + (tap >> 1) as i64;
            let wx = if tap & 1 == 1 { fx } else { 1.0 - fx };
            let wy = if tap >> 1 == 1 { fy } else { 1.0 - fy };
            if tx >= 0 && tx < width as i64 && ty >= 0 && ty < height as i64 {
                value += wy * wx * x[input_base + ty as usize * width + tx as usize];
            }
            tap += 1;
        }
        if let Some(out) = output.get_mut(idx) {
            *out = value;
        }
    }

    #[kernel]
    pub fn crop_bilinear_backward(
        dy: &[f32],
        source_x: &[f32],
        source_y: &[f32],
        scales: &[f32],
        dx: *mut f32,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= dy.len() {
            return;
        }
        let oplane = out_h * out_w;
        let b = i / (channels * oplane);
        let rem = i % (channels * oplane);
        let c = rem / oplane;
        let p = rem % oplane;
        let coord = b * oplane + p;
        let sx = source_x[coord];
        let sy = source_y[coord];
        let ix = floor_i64(sx);
        let iy = floor_i64(sy);
        let fx = sx - ix as f32;
        let fy = sy - iy as f32;
        let base = (b * channels + c) * in_h * in_w;
        let g = dy[i] * scales[b];
        let mut t = 0usize;
        while t < 4 {
            let tx = ix + (t & 1) as i64;
            let ty = iy + (t >> 1) as i64;
            let wx = if t & 1 == 1 { fx } else { 1.0 - fx };
            let wy = if t >> 1 == 1 { fy } else { 1.0 - fy };
            if tx >= 0 && tx < in_w as i64 && ty >= 0 && ty < in_h as i64 {
                // SAFETY: the tap is inside the image, so the address is inside
                // the `batch*channels*in_h*in_w` buffer the host validated; all
                // concurrent accesses to it are atomic.
                unsafe {
                    atomic_add_f32(dx.add(base + ty as usize * in_w + tx as usize), g * wy * wx);
                }
            }
            t += 1;
        }
    }
}
