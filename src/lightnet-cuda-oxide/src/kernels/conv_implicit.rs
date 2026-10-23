//! Implicit-GEMM f32 convolution forward: the classic `sgemm_tiled` tiling
//! (64x64x16, 4x4 per thread, `cp.async` double buffer) but the `B` operand is
//! generated on the fly from the NCHW input instead of being read from a
//! materialised im2col column buffer.
//!
//! The GEMM is `Y[f, p] = W[f, r] * col[r, p]` with `r = (c_local, ky, kx)` and
//! `p = oy * out_w + ox`; `col[r, p]` is `x[image, channel(r), iy(p, ky),
//! ix(p, kx)]` or zero outside the padded image.  Instances are `(image,
//! group)`, image-major, so the instance index maps to `A`/`C` with simple
//! strides exactly like the materialised forward path.

use super::unroll::unroll;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

pub const BM: usize = 64;
pub const BN: usize = 64;
pub const GEMM_THREADS: u32 = 256;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_conv_implicit(
        a: &[f32],
        x: &[f32],
        mut c: DisjointSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldc: usize,
        alpha: f32,
        beta: f32,
        a_base: usize,
        a_stride: usize,
        a_mod: usize,
        c_base: usize,
        c_stride: usize,
        c_mod: usize,
        count: usize,
        kchunk: usize,
        x_base: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        cpg: usize,
        groups: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        _out_h: usize,
        out_w: usize,
    ) {
        // Same double-buffered shared tiles as `sgemm_tiled`.
        static mut AS: SharedArray<f32, 2048, 16> = SharedArray::UNINIT; // 2 x 64 x 16
        static mut BS: SharedArray<f32, 2048, 16> = SharedArray::UNINIT; // 2 x 16 x 64
        let asp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };

        let tid = thread::threadIdx_x() as usize;
        let tx = tid % 16;
        let ty = tid / 16;

        let tiles_n = n.div_ceil(64);
        let tiles_m = m.div_ceil(64);
        let block = thread::blockIdx_x() as usize;
        let per_instance = tiles_m * tiles_n;
        let inst_all = block / per_instance;
        let inst = inst_all % count;
        let kbeg = (inst_all / count) * kchunk;
        let kend = if kbeg + kchunk < k { kbeg + kchunk } else { k };
        let tile = block % per_instance;
        let row0 = (tile / tiles_n) * 64;
        let col0 = (tile % tiles_n) * 64;

        let a_off = a_base + (inst % a_mod) * a_stride;
        let c_off = c_base + (inst_all % c_mod) * c_stride;

        // Implicit column decode: instance -> (image, group).
        let group = inst % groups;
        let image = inst / groups;
        let x_img = x_base + image * channels * in_h * in_w;
        let cbase = group * cpg;
        let kk = ksize * ksize;

        let vec_a = lda % 4 == 0 && a_off % 4 == 0;

        let mut acc = [0.0f32; 16];

        let nk = (kend - kbeg).div_ceil(16);
        let mut it = 0usize;
        while it <= nk {
            if it < nk {
                // Issue tile `it` into stage `it & 1`.
                let k0 = kbeg + it * 16;
                let stage = it & 1;
                if vec_a {
                    let ml = tid / 4;
                    let kl = (tid % 4) * 4;
                    let gm = row0 + ml;
                    let gk = k0 + kl;
                    let rem = if kend > gk { kend - gk } else { 0 };
                    let ok = gm < m && rem > 0;
                    let src = if ok { a_off + gm * lda + gk } else { 0 };
                    let bytes = if rem >= 4 { 16 } else { rem * 4 };
                    unsafe {
                        cuda_device::async_copy::cp_async_cg_zfill_16(
                            asp.add(stage * 1024 + ml * 16 + kl) as *mut u32,
                            a.as_ptr().add(src) as *const u8,
                            if ok { bytes as u32 } else { 0 },
                        );
                    }
                }
                // B tile from the input: one scalar `cp.async` per element.
                unroll!(4, r, {
                    let e = tid + 256 * r;
                    let kl2 = e / 64;
                    let nl = e % 64;
                    let gk = k0 + kl2;
                    let gn = col0 + nl;
                    let mut src = 0usize;
                    let mut ok = false;
                    if gk < kend && gn < n {
                        let cl = gk / kk;
                        let rem = gk % kk;
                        let ky = rem / ksize;
                        let kx = rem % ksize;
                        let oy = gn / out_w;
                        let ox = gn % out_w;
                        let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
                        let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
                        if iy >= 0 && iy < in_h as i64 && ix >= 0 && ix < in_w as i64 {
                            src = x_img + ((cbase + cl) * in_h + iy as usize) * in_w + ix as usize;
                            ok = true;
                        }
                    }
                    unsafe {
                        cuda_device::async_copy::cp_async_ca_zfill_4(
                            bsp.add(stage * 1024 + kl2 * 64 + nl) as *mut u32,
                            x.as_ptr().add(src) as *const u8,
                            if ok { 4 } else { 0 },
                        );
                    }
                });
                if !vec_a {
                    unroll!(4, r, {
                        let e = tid + 256 * r;
                        // A tile: AS[m_local * 16 + k_local] (row-major weights).
                        let ml = e / 16;
                        let kl = e % 16;
                        let gm = row0 + ml;
                        let gk = k0 + kl;
                        let ok = gm < m && gk < kend;
                        let idx = gm * lda + gk;
                        let src = if ok { a_off + idx } else { 0 };
                        unsafe {
                            cuda_device::async_copy::cp_async_ca_zfill_4(
                                asp.add(stage * 1024 + ml * 16 + kl) as *mut u32,
                                a.as_ptr().add(src) as *const u8,
                                if ok { 4 } else { 0 },
                            );
                        }
                    });
                }
            }
            unsafe {
                cuda_device::async_copy::cp_async_commit_group();
                cuda_device::async_copy::cp_async_wait_group(1);
            }
            thread::sync_threads();

            if it > 0 {
                let ao = ((it - 1) & 1) * 1024;
                let mut kk_i = 0usize;
                while kk_i < 16 {
                    let mut ar = [0.0f32; 4];
                    let mut br = [0.0f32; 4];
                    unroll!(4, i, {
                        unsafe {
                            ar[i] = AS[ao + (ty + 16 * i) * 16 + kk_i];
                            br[i] = BS[ao + kk_i * 64 + tx + 16 * i];
                        }
                    });
                    unroll!(4, i, {
                        unroll!(4, j, {
                            acc[i * 4 + j] += ar[i] * br[j];
                        });
                    });
                    kk_i += 1;
                }
            }
            thread::sync_threads();
            it += 1;
        }

        unroll!(4, i, {
            let gm = row0 + ty + 16 * i;
            unroll!(4, j, {
                let gn = col0 + tx + 16 * j;
                if gm < m && gn < n {
                    let idx = c_off + gm * ldc + gn;
                    // SAFETY: each (instance, row, col) is owned by exactly one
                    // thread; the host launcher validated the extents.
                    unsafe {
                        let dst = c.as_mut_ptr().add(idx);
                        let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                        *dst = alpha * acc[i * 4 + j] + prev;
                    }
                }
            });
        });
    }
}
