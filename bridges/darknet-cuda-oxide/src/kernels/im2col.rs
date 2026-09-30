//! im2col / col2im for one batch element, with stride, dilation, padding and
//! groups.  Ported from `src-lib/im2col_kernels.cu` / `col2im_kernels.cu`.
//!
//! Column layout: `col[r][p]` with `r = c * k * k + ky * k + kx` over *all*
//! input channels `c` and `p = oy * out_w + ox`.  Group `g` therefore owns the
//! contiguous row block `[g * kg, (g + 1) * kg)` with `kg = (C / groups) * k * k`,
//! which is exactly the `K x OHW` matrix the per-group GEMM consumes.
//!
//! col2im is written as a gather (one thread per input element) so it needs no
//! atomics and is bit-for-bit deterministic.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `x` is read at `x_base + (c * h + iy) * w + ix`.
    #[kernel]
    pub fn im2col(
        x: &[f32],
        mut col: DisjointSlice<f32>,
        x_base: usize,
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
        let idx = thread::index_1d();
        let i = idx.get();
        let ohw = out_h * out_w;
        let kk = ksize * ksize;
        if i >= channels * kk * ohw {
            return;
        }
        let r = i / ohw;
        let p = i % ohw;
        let c = r / kk;
        let rem = r % kk;
        let ky = rem / ksize;
        let kx = rem % ksize;
        let oy = p / out_w;
        let ox = p % out_w;
        let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
        let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
        let mut v = 0.0f32;
        if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
            v = x[x_base + (c * h + iy as usize) * w + ix as usize];
        }
        if let Some(o) = col.get_mut(idx) {
            *o = v;
        }
    }

    /// Gather form of col2im: `dx[dx_base + (c*h+iy)*w+ix] = sum of every col
    /// entry that im2col read from that input element`.
    #[kernel]
    pub fn col2im(
        col: &[f32],
        mut dx: DisjointSlice<f32>,
        dx_base: usize,
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
        let i = thread::index_1d().get();
        if i >= channels * h * w {
            return;
        }
        let c = i / (h * w);
        let rem = i % (h * w);
        let iy = rem / w;
        let ix = rem % w;
        let ohw = out_h * out_w;
        let kk = ksize * ksize;
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
                                sum += col[(c * kk + ky * ksize + kx) * ohw + oy * out_w + ox];
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        // SAFETY: one thread per input element; `i < channels*h*w` and the
        // host checks `dx_base + channels*h*w <= dx.len()`.
        unsafe {
            *dx.as_mut_ptr().add(dx_base + i) = sum;
        }
    }
}
