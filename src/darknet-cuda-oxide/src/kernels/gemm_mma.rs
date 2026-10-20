//! Tensor-core GEMM: `C = alpha * op(A) * op(B) + beta * C` with TF32 inputs
//! and f32 accumulation, using warp-level
//! `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (sm_80+, runs on
//! Ada sm_89).  Same `GemmArgs` semantics as [`super::gemm`]: row-major,
//! NN / NT / TN / TT, batched `count` with `(base, stride, modulus)` views,
//! arbitrary `m`, `n`, `k` (out-of-range elements are loaded as zero and
//! stores are guarded), `beta == 0` never reads `C`.
//!
//! Accuracy: operands are rounded to TF32 (`cvt.rna`, 10-bit mantissa, relative
//! error <= 2^-11 per operand); products and accumulation are f32.
//!
//! Tiling: 64x64 block tile, BK = 16, 128 threads = 4 warps in a 2x2 grid, each
//! warp owning a 32x32 sub-tile = 2 (m16) x 4 (n8) mma tiles per k8 step.
//! Two shared-memory stages are filled with `cp.async` (16-byte when the rows
//! are contiguous and aligned, 4-byte otherwise, zero-filled out of range)
//! while the previous stage feeds the tensor cores; operands are rounded to
//! TF32 (`cvt.rna`) when the fragments are read.  `ldmatrix` is not used: it
//! moves 16-bit elements only, and a TF32 fragment register is one 32-bit
//! element (row strides of 20 / 72 words make the reads bank-conflict free).
//! All small loops go through `unroll!`: un-unrolled accumulator arrays are
//! demoted to local memory by the device pipeline (2.5x slower).
//! Split-K (`kchunk` < `k`) is driven by the host (`OxideBackend::gemm`).
//!
//! Fragment layout (PTX m16n8k8 tf32, `g = lane / 4`, `t = lane % 4`):
//!   A regs: (g, t) (g+8, t) (g, t+4) (g+8, t+4);  B regs: (t, g) (t+4, g);
//!   C regs: (g, 2t) (g, 2t+1) (g+8, 2t) (g+8, 2t+1).

use super::unroll::unroll;
use cuda_device::convert::cvt_rna_tf32_f32;
use cuda_device::wmma::mma_m16n8k8_f32_tf32;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

pub const MMA_BM: usize = 64;
pub const MMA_BN: usize = 64;
pub const MMA_THREADS: u32 = 128;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sgemm_tf32_mma(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        trans_a: u32,
        trans_b: u32,
        alpha: f32,
        beta: f32,
        a_base: usize,
        a_stride: usize,
        a_mod: usize,
        b_base: usize,
        b_stride: usize,
        b_mod: usize,
        c_base: usize,
        c_stride: usize,
        c_mod: usize,
        count: usize,
        kchunk: usize,
    ) {
        // Two stages of raw f32 words (converted to TF32 when the fragments are
        // read), filled with 4-byte `cp.async` (zero-filled outside the matrix).
        static mut AS: SharedArray<u32, 2560, 16> = SharedArray::UNINIT; // 2 x 64 x 20
        static mut BS: SharedArray<u32, 2304, 16> = SharedArray::UNINIT; // 2 x 16 x 72
        let asp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };

        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let wm = (warp / 2) * 32;
        let wn = (warp % 2) * 32;

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
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst_all % c_mod) * c_stride;

        // acc[(mi * 4 + ni) * 4 + j]
        let mut acc = [0.0f32; 32];

        let nk = (kend - kbeg).div_ceil(16);
        let mut it = 0usize;
        while it <= nk {
            if it < nk {
                // Issue tile `it` into stage `it & 1`: 1024 elements per
                // operand, 8 per thread; fast index chosen so global reads stay
                // coalesced for each transpose case.
                let k0 = kbeg + it * 16;
                let stage = it & 1;
                // 16-byte copies when the operand rows are contiguous and 16-byte
                // aligned (NN/NT A rows, NN/TN B rows); element copies otherwise.
                let vec_a = trans_a == 0 && lda % 4 == 0 && a_off % 4 == 0;
                let vec_b = trans_b == 0 && ldb % 4 == 0 && b_off % 4 == 0;
                if vec_a {
                    let mut r = 0usize;
                    while r < 2 {
                        let e = tid + 128 * r;
                        let ml = e / 4;
                        let kl = (e % 4) * 4;
                        let gm = row0 + ml;
                        let gk = k0 + kl;
                        let rem = if kend > gk { kend - gk } else { 0 };
                        let ok = gm < m && rem > 0;
                        let src = if ok { a_off + gm * lda + gk } else { 0 };
                        let bytes = if rem >= 4 { 16 } else { rem * 4 };
                        // SAFETY: aligned in-bounds shared slot; `src` is a valid
                        // element of `a`, with `bytes` bounded by the row end.
                        unsafe {
                            cuda_device::async_copy::cp_async_cg_zfill_16(
                                asp.add(stage * 1280 + ml * 20 + kl),
                                a.as_ptr().add(src) as *const u8,
                                if ok { bytes as u32 } else { 0 },
                            );
                        }
                        r += 1;
                    }
                }
                if vec_b {
                    let mut r = 0usize;
                    while r < 2 {
                        let e = tid + 128 * r;
                        let kl2 = e / 16;
                        let nl = (e % 16) * 4;
                        let gk2 = k0 + kl2;
                        let gn = col0 + nl;
                        let rem = if n > gn { n - gn } else { 0 };
                        let ok = gk2 < kend && rem > 0;
                        let src = if ok { b_off + gk2 * ldb + gn } else { 0 };
                        let bytes = if rem >= 4 { 16 } else { rem * 4 };
                        unsafe {
                            cuda_device::async_copy::cp_async_cg_zfill_16(
                                bsp.add(stage * 1152 + kl2 * 72 + nl),
                                b.as_ptr().add(src) as *const u8,
                                if ok { bytes as u32 } else { 0 },
                            );
                        }
                        r += 1;
                    }
                }
                if !vec_a || !vec_b {
                    let mut r = 0usize;
                    while r < 8 {
                        let e = tid + 128 * r;
                        if !vec_a {
                            let (ml, kl) = if trans_a == 0 {
                                (e / 16, e % 16)
                            } else {
                                (e % 64, e / 64)
                            };
                            let gm = row0 + ml;
                            let gk = k0 + kl;
                            let ok = gm < m && gk < kend;
                            let idx = if trans_a == 0 {
                                gm * lda + gk
                            } else {
                                gk * lda + gm
                            };
                            let src = if ok { a_off + idx } else { 0 };
                            // SAFETY: in-bounds shared slot; the source is a valid
                            // element of `a` (index 0 with size 0 when out of range).
                            unsafe {
                                cuda_device::async_copy::cp_async_ca_zfill_4(
                                    asp.add(stage * 1280 + ml * 20 + kl),
                                    a.as_ptr().add(src) as *const u8,
                                    if ok { 4 } else { 0 },
                                );
                            }
                        }
                        if !vec_b {
                            let (kl2, nl) = if trans_b == 0 {
                                (e / 64, e % 64)
                            } else {
                                (e % 16, e / 16)
                            };
                            let gk2 = k0 + kl2;
                            let gn = col0 + nl;
                            let ok2 = gk2 < kend && gn < n;
                            let idx2 = if trans_b == 0 {
                                gk2 * ldb + gn
                            } else {
                                gn * ldb + gk2
                            };
                            let src2 = if ok2 { b_off + idx2 } else { 0 };
                            unsafe {
                                cuda_device::async_copy::cp_async_ca_zfill_4(
                                    bsp.add(stage * 1152 + kl2 * 72 + nl),
                                    b.as_ptr().add(src2) as *const u8,
                                    if ok2 { 4 } else { 0 },
                                );
                            }
                        }
                        r += 1;
                    }
                }
            }
            unsafe {
                cuda_device::async_copy::cp_async_commit_group();
                // Tile `it` may still be in flight; tile `it - 1` must have landed.
                cuda_device::async_copy::cp_async_wait_group(1);
            }
            thread::sync_threads();

            if it > 0 {
                let stage = (it - 1) & 1;
                let ao = stage * 1280;
                let bo = stage * 1152;
                let mut ks = 0usize;
                while ks < 16 {
                    let mut bf = [[0u32; 2]; 4];
                    unroll!(4, ni, {
                        let col = wn + ni * 8 + g;
                        unsafe {
                            bf[ni][0] =
                                cvt_rna_tf32_f32(f32::from_bits(BS[bo + (ks + t) * 72 + col]));
                            bf[ni][1] =
                                cvt_rna_tf32_f32(f32::from_bits(BS[bo + (ks + t + 4) * 72 + col]));
                        }
                    });
                    unroll!(2, mi, {
                        let row = wm + mi * 16 + g;
                        let af = unsafe {
                            [
                                cvt_rna_tf32_f32(f32::from_bits(AS[ao + row * 20 + ks + t])),
                                cvt_rna_tf32_f32(f32::from_bits(AS[ao + (row + 8) * 20 + ks + t])),
                                cvt_rna_tf32_f32(f32::from_bits(AS[ao + row * 20 + ks + t + 4])),
                                cvt_rna_tf32_f32(f32::from_bits(
                                    AS[ao + (row + 8) * 20 + ks + t + 4],
                                )),
                            ]
                        };
                        unroll!(4, ni, {
                            let o = (mi * 4 + ni) * 4;
                            let cin = [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]];
                            // SAFETY: all 32 lanes of every warp reach this call
                            // together (no divergent control flow around it);
                            // fragments follow the documented layout and A/B hold
                            // valid TF32 bit patterns from `cvt.rna.tf32`.
                            let d = unsafe { mma_m16n8k8_f32_tf32(cin, af, bf[ni]) };
                            acc[o] = d[0];
                            acc[o + 1] = d[1];
                            acc[o + 2] = d[2];
                            acc[o + 3] = d[3];
                        });
                    });
                    ks += 8;
                }
            }
            // Stage `(it - 1) & 1` is refilled by the next iteration's issue.
            thread::sync_threads();
            it += 1;
        }

        unroll!(2, mi, {
            unroll!(4, ni, {
                unroll!(4, j, {
                    let gm = row0 + wm + mi * 16 + g + if j >= 2 { 8 } else { 0 };
                    let gn = col0 + wn + ni * 8 + 2 * t + (j & 1);
                    if gm < m && gn < n {
                        let idx = c_off + gm * ldc + gn;
                        // SAFETY: each (instance, row, col) is owned by exactly
                        // one thread; extents are validated by the host launcher.
                        unsafe {
                            let dst = c.as_mut_ptr().add(idx);
                            let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                            *dst = alpha * acc[(mi * 4 + ni) * 4 + j] + prev;
                        }
                    }
                });
            });
        });
    }
}
