//! Direct (no im2col) convolution kernels: forward, data gradient, weight
//! gradient.  Weights are `[filters, channels / groups, k, k]`, tensors NCHW,
//! one symmetric `padding`, per-axis strides, dilation applied to the taps.
//!
//! All three kernels are deterministic: forward and dgrad give every output
//! element to exactly one thread (dgrad gathers, it never scatters), and wgrad
//! reduces each weight element inside one block with a fixed-order tree.

use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

/// Threads per block in the weight-gradient reduction (power of two).
pub const WGRAD_THREADS: u32 = 128;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn conv_direct_fwd(
        x: &[f32],
        w: &[f32],
        mut out: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        h: usize,
        wd: usize,
        filters: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= batch * filters * out_h * out_w {
            return;
        }
        let ox = i % out_w;
        let oy = (i / out_w) % out_h;
        let f = (i / (out_w * out_h)) % filters;
        let b = i / (out_w * out_h * filters);
        let cpg = channels / groups;
        let g = f / (filters / groups);
        let mut acc = 0.0f32;
        let mut ic = 0usize;
        while ic < cpg {
            let c = g * cpg + ic;
            let mut ky = 0usize;
            while ky < ksize {
                let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
                if iy >= 0 && iy < h as i64 {
                    let mut kx = 0usize;
                    while kx < ksize {
                        let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
                        if ix >= 0 && ix < wd as i64 {
                            acc += x[((b * channels + c) * h + iy as usize) * wd + ix as usize]
                                * w[((f * cpg + ic) * ksize + ky) * ksize + kx];
                        }
                        kx += 1;
                    }
                }
                ky += 1;
            }
            ic += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = acc;
        }
    }

    /// dx[b,c,iy,ix] = sum over filters of the group, taps hitting (iy, ix).
    #[kernel]
    pub fn conv_direct_dgrad(
        dy: &[f32],
        w: &[f32],
        mut dx: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        h: usize,
        wd: usize,
        filters: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        out_h: usize,
        out_w: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= batch * channels * h * wd {
            return;
        }
        let ix = i % wd;
        let iy = (i / wd) % h;
        let c = (i / (wd * h)) % channels;
        let b = i / (wd * h * channels);
        let cpg = channels / groups;
        let fpg = filters / groups;
        let g = c / cpg;
        let ic = c % cpg;
        let mut acc = 0.0f32;
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
                                let mut fo = 0usize;
                                while fo < fpg {
                                    let f = g * fpg + fo;
                                    acc += dy[((b * filters + f) * out_h + oy) * out_w + ox]
                                        * w[((f * cpg + ic) * ksize + ky) * ksize + kx];
                                    fo += 1;
                                }
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = acc;
        }
    }

    /// One block per weight element `(f, ic, ky, kx)`; its threads stride over
    /// `(b, oy, ox)` and fold through shared memory in a fixed order.
    #[kernel]
    pub fn conv_direct_wgrad(
        x: &[f32],
        dy: &[f32],
        mut dw: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        h: usize,
        wd: usize,
        filters: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        out_h: usize,
        out_w: usize,
    ) {
        static mut RED: SharedArray<f32, 128> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let widx = thread::blockIdx_x() as usize;
        let kx = widx % ksize;
        let ky = (widx / ksize) % ksize;
        let cpg = channels / groups;
        let ic = (widx / (ksize * ksize)) % cpg;
        let f = widx / (ksize * ksize * cpg);
        let g = f / (filters / groups);
        let c = g * cpg + ic;

        let total = batch * out_h * out_w;
        let mut acc = 0.0f32;
        let mut t = tid;
        while t < total {
            let ox = t % out_w;
            let oy = (t / out_w) % out_h;
            let b = t / (out_w * out_h);
            let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
            let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
            if iy >= 0 && iy < h as i64 && ix >= 0 && ix < wd as i64 {
                acc += x[((b * channels + c) * h + iy as usize) * wd + ix as usize]
                    * dy[((b * filters + f) * out_h + oy) * out_w + ox];
            }
            t += 128;
        }
        unsafe {
            RED[tid] = acc;
        }
        thread::sync_threads();
        let mut s = 64usize;
        while s > 0 {
            if tid < s {
                unsafe {
                    RED[tid] = RED[tid] + RED[tid + s];
                }
            }
            thread::sync_threads();
            s /= 2;
        }
        if tid == 0 {
            // SAFETY: block `widx` is the only writer of `dw[widx]`; the host
            // launches exactly `dw.len()` blocks.
            unsafe {
                *dw.as_mut_ptr().add(widx) = RED[0];
            }
        }
    }
}
