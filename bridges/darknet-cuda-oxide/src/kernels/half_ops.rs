//! f16-storage inference ops (P6): im2col with padded GEMM layout, maxpool,
//! route and nearest upsample on raw `u16` half bit patterns.  Compare /
//! arithmetic happens in f32 after an exact widening, results are rounded back
//! to nearest even (data movement ops are bit copies).

use super::convert::{f16_bits_to_f32, f32_to_f16_bits};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// im2col of one batch element into the GEMM operand layout used by the
    /// f16 conv: group `g` owns a `kpad x npad` block
    /// `col[(g * kpad + rr) * npad + p]` where `rr < kg` is the row inside the
    /// group (`r = g * kg + rr = c * k * k + ky * k + kx`) and `p < oh * ow`;
    /// the K / N padding is written as zero.
    #[kernel]
    pub fn im2col_f16(
        x: &[u16],
        mut col: DisjointSlice<u16>,
        x_base: usize,
        h: usize,
        w: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        out_h: usize,
        out_w: usize,
        kg: usize,
        kpad: usize,
        npad: usize,
        groups: usize,
        images: usize,
        img_stride: usize,
    ) {
        let idx = thread::index_1d();
        let gi = idx.get();
        let blk = kpad * npad;
        if gi >= images * groups * blk {
            return;
        }
        // Column buffer layout `[image][group][kpad][npad]`; image `m` reads
        // `x` from `x_base + m * img_stride`.
        let img = gi / (groups * blk);
        let i = gi % (groups * blk);
        let x_base = x_base + img * img_stride;
        let grp = i / blk;
        let rem = i % blk;
        let rr = rem / npad;
        let p = rem % npad;
        let ohw = out_h * out_w;
        let mut v = 0u16;
        if rr < kg && p < ohw {
            let r = grp * kg + rr;
            let kk = ksize * ksize;
            let c = r / kk;
            let kr = r % kk;
            let ky = kr / ksize;
            let kx = kr % ksize;
            let oy = p / out_w;
            let ox = p % out_w;
            let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
            let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
            if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                v = x[x_base + (c * h + iy as usize) * w + ix as usize];
            }
        }
        if let Some(o) = col.get_mut(idx) {
            *o = v;
        }
    }

    /// Same window rule as the f32 `maxpool_forward` (taps outside the image
    /// skipped, strict `>`, empty window -> -inf).
    #[kernel]
    pub fn maxpool_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
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
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let plane = (i / oplane) * h * w;
        let r = i % oplane;
        let y0 = ((r / out_w) * stride_y) as i64 - off as i64;
        let x0 = ((r % out_w) * stride_x) as i64 - off as i64;
        let mut best = f32::NEG_INFINITY;
        let mut best_bits = 0xFC00u16;
        let mut ky = 0usize;
        while ky < size {
            let iy = y0 + ky as i64;
            if iy >= 0 && iy < h as i64 {
                let mut kx = 0usize;
                while kx < size {
                    let ix = x0 + kx as i64;
                    if ix >= 0 && ix < w as i64 {
                        let bits = x[plane + iy as usize * w + ix as usize];
                        let v = f16_bits_to_f32(bits);
                        if v > best {
                            best = v;
                            best_bits = bits;
                        }
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = best_bits;
        }
    }

    /// Nearest upsample; `scale != 1` multiplies in f32 and rounds back.
    #[kernel]
    pub fn upsample_nearest_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
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
        let bits = x[bc * in_h * in_w + iy * in_w + ix];
        if let Some(o) = out.get_mut(idx) {
            *o = if scale == 1.0 { bits } else { f32_to_f16_bits(scale * f16_bits_to_f32(bits), 0) };
        }
    }

    /// Same strided channel-slice copy as the f32 `route_copy`.
    #[kernel]
    pub fn route_copy_f16(
        src: &[u16],
        mut dst: DisjointSlice<u16>,
        batch: usize,
        channels: usize,
        src_total: usize,
        src_offset: usize,
        dst_total: usize,
        dst_offset: usize,
        stride: usize,
    ) {
        let i = thread::index_1d().get();
        let per_batch = channels * stride;
        if i >= batch * per_batch {
            return;
        }
        let b = i / per_batch;
        let r = i % per_batch;
        let c = r / stride;
        let s = r % stride;
        let from = (b * src_total + src_offset + c) * stride + s;
        let to = (b * dst_total + dst_offset + c) * stride + s;
        // SAFETY: each thread writes a distinct `to`; the host checks that
        // `dst` covers `batch * dst_total * stride` elements.
        unsafe {
            *dst.as_mut_ptr().add(to) = src[from];
        }
    }
}
