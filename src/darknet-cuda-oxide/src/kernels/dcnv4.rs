//! DCNv4 deformable aggregation and its adjoint.
//!
//! Contract: `darknet_cuda::host_ops::dcnv4` (transcribed from
//! `darknet_core::dcnv4`). One thread per `(batch, group, output cell)` owns that
//! cell's channel outputs on the forward pass; the backward pass atomically
//! scatters into `d_input` and owns its row's `d_offsets` (no atomics there).
//! Input-gradient atomic summation order is not bitwise deterministic.
//!
//! Kernel points are held in a fixed-size buffer, so `size * size` must not
//! exceed [`MAX_POINTS`]; the launcher rejects larger kernels explicitly rather
//! than silently truncating. The current bound covers kernels through 27x27.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Largest supported kernel area (`size <= 27`).
pub const MAX_POINTS: usize = 729;

/// Relaxed device-scope `*p += v`.
///
/// # Safety
/// `p` must be a live, aligned device f32 only accessed atomically.
#[device]
pub unsafe fn dcnv4_atomic_add(p: *mut f32, v: f32) {
    let cell = unsafe { DeviceAtomicF32::from_ptr(p) };
    let _old = cell.fetch_add(v, AtomicOrdering::Relaxed);
}

#[device]
fn dcnv4_floor_i64(v: f32) -> i64 {
    let t = v as i64;
    if (t as f32) > v { t - 1 } else { t }
}

#[device]
fn dcnv4_in_bounds(v: i64, extent: usize) -> bool {
    v >= 0 && v < extent as i64
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Aggregates `[B, C, H, W]` at `[B, OH*OW, offset_dim]` offsets.
    /// One thread per `(batch, group, output cell)`; it writes only that cell's
    /// channels in its group, so no output element has two writers.
    #[kernel]
    pub fn dcnv4_aggregate(
        input: &[f32],
        offsets: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        groups: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        softmax: u32,
        remove_center: u32,
        offset_scale: f32,
    ) {
        let spatial = out_h * out_w;
        let points = size * size - remove_center as usize;
        let cpg = channels / groups;
        let offset_dim = (groups * points * 3 + 7) / 8 * 8;
        let i = thread::index_1d().get();
        if i >= out.len() / cpg {
            return;
        }
        let b = i / (groups * spatial);
        let rem = i % (groups * spatial);
        let g = rem / spatial;
        let q = rem % spatial;
        let (out_y, out_x) = (q / out_w, q % out_w);
        let center = (dilation * (size - 1)) >> 1;
        let center_f = center as f32;
        let p0_x = center as f32 - padding as f32 + out_x as f32 * stride_x as f32;
        let p0_y = center as f32 - padding as f32 + out_y as f32 * stride_y as f32;
        let p0_x_scaled = p0_x - center_f * offset_scale;
        let p0_y_scaled = p0_y - center_f * offset_scale;
        let row_start = (b * spatial + q) * offset_dim;
        let group_offset = g * points * 3;
        let mask_start = group_offset + 2 * points;
        let mut masks = [0.0f32; MAX_POINTS];
        let mut k = 0usize;
        while k < points {
            masks[k] = offsets[row_start + mask_start + k];
            k += 1;
        }
        if softmax != 0 {
            let mut maximum = -3.4e38f32;
            k = 0;
            while k < points {
                if masks[k] > maximum {
                    maximum = masks[k];
                }
                k += 1;
            }
            let mut sum = 0.0f32;
            k = 0;
            while k < points {
                masks[k] = (masks[k] - maximum).exp();
                sum += masks[k];
                k += 1;
            }
            let denominator = sum + 1.0e-6;
            k = 0;
            while k < points {
                masks[k] /= denominator;
                k += 1;
            }
        }
        let mut point = 0usize;
        let mut kw = 0usize;
        while kw < size {
            let mut kh = 0usize;
            while kh < size {
                if remove_center != 0 && kw == size / 2 && kh == size / 2 {
                    kh += 1;
                    continue;
                }
                let ox = offsets[row_start + group_offset + 2 * point];
                let oy = offsets[row_start + group_offset + 2 * point + 1];
                let sx = p0_x_scaled + (kw as f32 * dilation as f32 + ox) * offset_scale;
                let sy = p0_y_scaled + (kh as f32 * dilation as f32 + oy) * offset_scale;
                let mask = masks[point];
                let mut d = 0usize;
                while d < cpg {
                    let channel = g * cpg + d;
                    let base = (b * channels + channel) * in_h * in_w;
                    let mut value = 0.0f32;
                    if !(sy < -1.0 || sy > in_h as f32 || sx < -1.0 || sx > in_w as f32) {
                        let y0 = dcnv4_floor_i64(sy);
                        let x0 = dcnv4_floor_i64(sx);
                        let ly = sy - y0 as f32;
                        let lx = sx - x0 as f32;
                        let hy = 1.0 - ly;
                        let hx = 1.0 - lx;
                        if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0, in_w) {
                            value += hy * hx * input[base + y0 as usize * in_w + x0 as usize];
                        }
                        if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                            value += hy * lx * input[base + y0 as usize * in_w + (x0 + 1) as usize];
                        }
                        if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0, in_w) {
                            value += ly * hx * input[base + (y0 + 1) as usize * in_w + x0 as usize];
                        }
                        if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                            value += ly
                                * lx
                                * input[base + (y0 + 1) as usize * in_w + (x0 + 1) as usize];
                        }
                    }
                    // SAFETY: this thread owns `(b, group, q)`, so the address is
                    // one of its own channel outputs and no other thread writes it.
                    unsafe {
                        *out.as_mut_ptr().add((b * channels + channel) * spatial + q) +=
                            mask * value;
                    }
                    d += 1;
                }
                point += 1;
                kh += 1;
            }
            kw += 1;
        }
    }
}

/// Adjoint of `dcnv4_aggregate`. One thread per `(batch, group, cell)`; it owns
/// its row's offset slots (no atomics there) and atomically scatters `d_input`.
#[cuda_module]
pub mod backward {
    use super::*;

    #[kernel]
    pub fn dcnv4_aggregate_backward(
        input: &[f32],
        offsets: &[f32],
        dy: &[f32],
        d_input: *mut f32,
        d_offsets: *mut f32,
        channels: usize,
        groups: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        softmax: u32,
        remove_center: u32,
        offset_scale: f32,
    ) {
        let spatial = out_h * out_w;
        let points = size * size - remove_center as usize;
        let cpg = channels / groups;
        let offset_dim = (groups * points * 3 + 7) / 8 * 8;
        let i = thread::index_1d().get();
        if i >= dy.len() / cpg {
            return;
        }
        let b = i / (groups * spatial);
        let rem = i % (groups * spatial);
        let g = rem / spatial;
        let q = rem % spatial;
        let (out_y, out_x) = (q / out_w, q % out_w);
        let center = (dilation * (size - 1)) >> 1;
        let center_f = center as f32;
        let p0_x = center as f32 - padding as f32 + out_x as f32 * stride_x as f32;
        let p0_y = center as f32 - padding as f32 + out_y as f32 * stride_y as f32;
        let p0_x_scaled = p0_x - center_f * offset_scale;
        let p0_y_scaled = p0_y - center_f * offset_scale;
        let row_start = (b * spatial + q) * offset_dim;
        let group_offset = g * points * 3;
        let mask_start = group_offset + 2 * points;
        let mut masks = [0.0f32; MAX_POINTS];
        let mut k = 0usize;
        while k < points {
            masks[k] = offsets[row_start + mask_start + k];
            k += 1;
        }
        if softmax != 0 {
            let mut maximum = -3.4e38f32;
            k = 0;
            while k < points {
                if masks[k] > maximum {
                    maximum = masks[k];
                }
                k += 1;
            }
            let mut sum = 0.0f32;
            k = 0;
            while k < points {
                masks[k] = (masks[k] - maximum).exp();
                sum += masks[k];
                k += 1;
            }
            let denominator = sum + 1.0e-6;
            k = 0;
            while k < points {
                masks[k] /= denominator;
                k += 1;
            }
        }
        let mut mask_gradient = [0.0f32; MAX_POINTS];
        let mut point = 0usize;
        let mut kw = 0usize;
        while kw < size {
            let mut kh = 0usize;
            while kh < size {
                if remove_center != 0 && kw == size / 2 && kh == size / 2 {
                    kh += 1;
                    continue;
                }
                let ox = offsets[row_start + group_offset + 2 * point];
                let oy = offsets[row_start + group_offset + 2 * point + 1];
                let sx = p0_x_scaled + (kw as f32 * dilation as f32 + ox) * offset_scale;
                let sy = p0_y_scaled + (kh as f32 * dilation as f32 + oy) * offset_scale;
                let mask = masks[point];
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;
                let mut gm = 0.0f32;
                let mut d = 0usize;
                while d < cpg {
                    let channel = g * cpg + d;
                    let base = (b * channels + channel) * in_h * in_w;
                    let top = dy[(b * channels + channel) * spatial + q];
                    let y0 = dcnv4_floor_i64(sy);
                    let x0 = dcnv4_floor_i64(sx);
                    let ly = sy - y0 as f32;
                    let lx = sx - x0 as f32;
                    let hy = 1.0 - ly;
                    let hx = 1.0 - lx;
                    let v00 = if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0, in_w) {
                        input[base + y0 as usize * in_w + x0 as usize]
                    } else {
                        0.0
                    };
                    let v01 = if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                        input[base + y0 as usize * in_w + (x0 + 1) as usize]
                    } else {
                        0.0
                    };
                    let v10 = if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0, in_w) {
                        input[base + (y0 + 1) as usize * in_w + x0 as usize]
                    } else {
                        0.0
                    };
                    let v11 = if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                        input[base + (y0 + 1) as usize * in_w + (x0 + 1) as usize]
                    } else {
                        0.0
                    };
                    let value = hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
                    let dvdx = hy * (v01 - v00) + ly * (v11 - v10);
                    let dvdy = hx * (v10 - v00) + lx * (v11 - v01);
                    let scale = top * mask;
                    // SAFETY: `d_input` is a zeroed caller buffer; every tap is
                    // bounds-checked and all writers are atomic.
                    unsafe {
                        if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0, in_w) {
                            dcnv4_atomic_add(
                                d_input.add(base + y0 as usize * in_w + x0 as usize),
                                scale * hy * hx,
                            );
                        }
                        if dcnv4_in_bounds(y0, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                            dcnv4_atomic_add(
                                d_input.add(base + y0 as usize * in_w + (x0 + 1) as usize),
                                scale * hy * lx,
                            );
                        }
                        if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0, in_w) {
                            dcnv4_atomic_add(
                                d_input.add(base + (y0 + 1) as usize * in_w + x0 as usize),
                                scale * ly * hx,
                            );
                        }
                        if dcnv4_in_bounds(y0 + 1, in_h) && dcnv4_in_bounds(x0 + 1, in_w) {
                            dcnv4_atomic_add(
                                d_input.add(base + (y0 + 1) as usize * in_w + (x0 + 1) as usize),
                                scale * ly * lx,
                            );
                        }
                    }
                    gx += top * mask * dvdx * offset_scale;
                    gy += top * mask * dvdy * offset_scale;
                    gm += top * value;
                    d += 1;
                }
                // SAFETY: this thread owns `(b, g, q)`, hence its row's slots.
                unsafe {
                    *d_offsets.add(row_start + group_offset + 2 * point) += gx;
                    *d_offsets.add(row_start + group_offset + 2 * point + 1) += gy;
                }
                mask_gradient[point] = gm;
                point += 1;
                kh += 1;
            }
            kw += 1;
        }
        if softmax != 0 {
            let mut dot = 0.0f32;
            let mut i2 = 0usize;
            while i2 < points {
                dot += mask_gradient[i2] * masks[i2];
                i2 += 1;
            }
            let mut i2 = 0usize;
            while i2 < points {
                // SAFETY: as above, owned row slots.
                unsafe {
                    *d_offsets.add(row_start + mask_start + i2) +=
                        masks[i2] * (mask_gradient[i2] - dot);
                }
                i2 += 1;
            }
        } else {
            let mut i2 = 0usize;
            while i2 < points {
                unsafe {
                    *d_offsets.add(row_start + mask_start + i2) += mask_gradient[i2];
                }
                i2 += 1;
            }
        }
    }
}
