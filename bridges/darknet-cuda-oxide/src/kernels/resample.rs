//! Upsample (nearest / reverse) and bilinear crop.
//!
//! Upsample follows `src-lib/blas_kernels.cu::upsample_kernel` and the
//! `darknet-core` reference.  The forward direction of a `reverse` layer is a
//! strided sum; it is written as a gather per output element (the C++ kernel
//! uses `atomicAdd`), so both directions are deterministic.
//!
//! Crop follows the cuTile bridge: bilinear sample with zero taps outside the
//! image, then a per-batch affine `value * scale + offset`.  Its backward is a
//! scatter with arbitrary source coordinates and therefore uses float atomics
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
                value += wy * wx * x[base + ty as usize * in_w + tx as usize];
            }
            t += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = value * scales[b] + offsets[b];
        }
    }

    /// Scatter of the crop gradient into the input (`dx` must be zeroed).
    /// `dx` is a raw device pointer because the updates are atomic adds.
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
