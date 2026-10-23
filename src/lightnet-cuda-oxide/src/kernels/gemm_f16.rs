//! Tensor-core GEMM with f16 inputs and f32 accumulation:
//! `C = alpha * op(A) * op(B) + beta * C` through warp-level
//! `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (sm_80+).  Operands are
//! raw f16 bit patterns (`u16`).  Same `GemmArgs` semantics as
//! [`super::gemm_mma`]: row-major, NN / NT / TN / TT, batched `count` with
//! `(base, stride, modulus)` views, arbitrary `m`, `n`, `k`, `beta == 0` never
//! reads `C`, split-K driven by the host (`kchunk`, a multiple of 32).
//!
//! Tiling: 64x64 block tile, BK = 32, 128 threads = 4 warps (2x2), each warp a
//! 32x32 sub-tile = 2 (m16) x 4 (n8) mma tiles per k16 step, two `cp.async`
//! stages (the same pipeline as `gemm_mma`).  Fragments come from `ldmatrix`
//! (`.trans` where the shared tile is stored the other way round), so all four
//! transpose cases use the same mma code:
//!
//! * A, `trans_a == 0`: tile `[64 m][32 k]`, pitch 40 halves, `ldmatrix.x4`;
//!   `trans_a == 1`: tile `[32 k][64 m]`, pitch 72, `ldmatrix.x4.trans`.
//! * B, `trans_b == 0`: tile `[32 k][64 n]`, pitch 72, `ldmatrix.x4.trans`;
//!   `trans_b == 1`: tile `[64 n][32 k]`, pitch 40, `ldmatrix.x4`.
//!
//! (Pitches of 80 / 144 bytes make every 8-row ldmatrix phase bank-conflict
//! free.)  Operand tiles move with 16-byte `cp.async` when the contiguous
//! dimension is a multiple of 8 and 16-byte aligned (always true for the conv
//! path, which pads K and N to multiples of 8), else with a plain element
//! loop (correct for every shape, slower).
//!
//! `bf16 != 0` reads the operands as bfloat16 instead (same tiles and
//! fragments, `mma.sync ... .bf16.bf16.f32`); only the f32 epilogue is used then.
//!
//! Epilogue `epi == 0`: f32 `alpha * acc + beta * C`.  `epi == 1` (inference
//! conv): `v = alpha * acc + bias[(inst % bias_mod) * bias_stride + row]`, activation `act`,
//! round to nearest even and store f16 into `c16` (columns `>= n_store`, the
//! N padding, are not stored).

use super::unroll::unroll;
use super::convert::f32_to_half_bits;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::wmma::{
    ldmatrix_x4_shared_u32, ldmatrix_x4_trans_shared_u32, mma_m16n8k16_f32_bf16,
    mma_m16n8k16_f32_f16,
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

pub const HMMA_BM: usize = 64;
pub const HMMA_BN: usize = 64;
pub const HMMA_BK: usize = 32;
pub const HMMA_THREADS: u32 = 128;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn hgemm_f16_mma(
        a: &[u16],
        b: &[u16],
        bias: &[f32],
        mut c: DisjointSlice<f32>,
        mut c16: DisjointSlice<u16>,
        m: usize,
        n: usize,
        k: usize,
        n_store: usize,
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
        epi: u32,
        act: u32,
        bias_stride: usize,
        bias_mod: usize,
        bf16: u32,
    ) {
        // 2 stages x 2560 halves (1280 words) per operand.
        static mut AS: SharedArray<u32, 2560, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 2560, 16> = SharedArray::UNINIT;
        let asp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };
        let ash = asp as *mut u16;
        let bsh = bsp as *mut u16;
        let sa = unsafe { cvta_generic_to_shared_u32(asp as *const u8) };
        let sb = unsafe { cvta_generic_to_shared_u32(bsp as *const u8) };

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

        let vec_a =
            (if trans_a == 0 { k % 8 == 0 } else { m % 8 == 0 }) && lda % 8 == 0 && a_off % 8 == 0;
        let vec_b =
            (if trans_b == 0 { n % 8 == 0 } else { k % 8 == 0 }) && ldb % 8 == 0 && b_off % 8 == 0;

        let idx8 = lane / 8;
        let r8 = lane % 8;

        // acc[(mi * 4 + ni) * 4 + j]
        let mut acc = [0.0f32; 32];

        let nk = (kend - kbeg).div_ceil(32);
        let mut it = 0usize;
        while it <= nk {
            if it < nk {
                let k0 = kbeg + it * 32;
                let stage = it & 1;
                if vec_a {
                    let mut r = 0usize;
                    while r < 2 {
                        let e = tid + 128 * r;
                        let (ml, kl, gm, gk) = if trans_a == 0 {
                            let ml = e / 4;
                            let kl = (e % 4) * 8;
                            (ml, kl, row0 + ml, k0 + kl)
                        } else {
                            let kl = e / 8;
                            let ml = (e % 8) * 8;
                            (ml, kl, row0 + ml, k0 + kl)
                        };
                        let ok = gm < m && gk < kend;
                        let (src, dst) = if trans_a == 0 {
                            (a_off + gm * lda + gk, stage * 1280 + (ml * 40 + kl) / 2)
                        } else {
                            (a_off + gk * lda + gm, stage * 1280 + (kl * 72 + ml) / 2)
                        };
                        // SAFETY: aligned in-bounds shared slot; the source is
                        // a valid 16-byte chunk of `a` when `ok`, else 0 bytes
                        // are read (zero fill) from element 0.
                        unsafe {
                            cuda_device::async_copy::cp_async_cg_zfill_16(
                                asp.add(dst),
                                a.as_ptr().add(if ok { src } else { 0 }) as *const u8,
                                if ok { 16 } else { 0 },
                            );
                        }
                        r += 1;
                    }
                }
                if vec_b {
                    let mut r = 0usize;
                    while r < 2 {
                        let e = tid + 128 * r;
                        let (kl, nl, gk, gn) = if trans_b == 0 {
                            let kl = e / 8;
                            let nl = (e % 8) * 8;
                            (kl, nl, k0 + kl, col0 + nl)
                        } else {
                            let nl = e / 4;
                            let kl = (e % 4) * 8;
                            (kl, nl, k0 + kl, col0 + nl)
                        };
                        let ok = gn < n && gk < kend;
                        let (src, dst) = if trans_b == 0 {
                            (b_off + gk * ldb + gn, stage * 1280 + (kl * 72 + nl) / 2)
                        } else {
                            (b_off + gn * ldb + gk, stage * 1280 + (nl * 40 + kl) / 2)
                        };
                        unsafe {
                            cuda_device::async_copy::cp_async_cg_zfill_16(
                                bsp.add(dst),
                                b.as_ptr().add(if ok { src } else { 0 }) as *const u8,
                                if ok { 16 } else { 0 },
                            );
                        }
                        r += 1;
                    }
                }
                if !vec_a {
                    let mut r = 0usize;
                    while r < 16 {
                        let e = tid + 128 * r;
                        let (ml, kl) = if trans_a == 0 {
                            (e / 32, e % 32)
                        } else {
                            (e % 64, e / 64)
                        };
                        let gm = row0 + ml;
                        let gk = k0 + kl;
                        let ok = gm < m && gk < kend;
                        let v = if ok {
                            if trans_a == 0 {
                                a[a_off + gm * lda + gk]
                            } else {
                                a[a_off + gk * lda + gm]
                            }
                        } else {
                            0u16
                        };
                        let sidx = if trans_a == 0 {
                            ml * 40 + kl
                        } else {
                            kl * 72 + ml
                        };
                        // SAFETY: in-bounds shared slot (< 2560 per stage).
                        unsafe {
                            *ash.add(stage * 2560 + sidx) = v;
                        }
                        r += 1;
                    }
                }
                if !vec_b {
                    let mut r = 0usize;
                    while r < 16 {
                        let e = tid + 128 * r;
                        let (kl, nl) = if trans_b == 0 {
                            (e / 64, e % 64)
                        } else {
                            (e % 32, e / 32)
                        };
                        let gk = k0 + kl;
                        let gn = col0 + nl;
                        let ok = gn < n && gk < kend;
                        let v = if ok {
                            if trans_b == 0 {
                                b[b_off + gk * ldb + gn]
                            } else {
                                b[b_off + gn * ldb + gk]
                            }
                        } else {
                            0u16
                        };
                        let sidx = if trans_b == 0 {
                            kl * 72 + nl
                        } else {
                            nl * 40 + kl
                        };
                        unsafe {
                            *bsh.add(stage * 2560 + sidx) = v;
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

            macro_rules! compute_tile {
                ($mma:ident) => {
                    if it > 0 {
                        let so = ((it - 1) & 1) * 2560;
                        unroll!(2, kk, {
                            let ks = kk * 16;
                            let mut bf = [[0u32; 2]; 4];
                            unroll!(2, nj, {
                                let regs = if trans_b == 0 {
                                    let kr = ks + (idx8 & 1) * 8 + r8;
                                    let nc = wn + nj * 16 + (idx8 >> 1) * 8;
                                    // SAFETY: all lanes participate; addresses are
                                    // 16-byte aligned rows inside the stage.
                                    unsafe {
                                        ldmatrix_x4_trans_shared_u32(
                                            sb + 2 * (so + kr * 72 + nc) as u32,
                                        )
                                    }
                                } else {
                                    let nr = wn + nj * 16 + (idx8 >> 1) * 8 + r8;
                                    let kc = ks + (idx8 & 1) * 8;
                                    unsafe {
                                        ldmatrix_x4_shared_u32(sb + 2 * (so + nr * 40 + kc) as u32)
                                    }
                                };
                                bf[nj * 2] = [regs[0], regs[1]];
                                bf[nj * 2 + 1] = [regs[2], regs[3]];
                            });
                            unroll!(2, mi, {
                                let af = if trans_a == 0 {
                                    let row = wm + mi * 16 + (lane % 16);
                                    let col = ks + (lane / 16) * 8;
                                    unsafe {
                                        ldmatrix_x4_shared_u32(
                                            sa + 2 * (so + row * 40 + col) as u32,
                                        )
                                    }
                                } else {
                                    let kr = ks + (idx8 >> 1) * 8 + r8;
                                    let mc = wm + mi * 16 + (idx8 & 1) * 8;
                                    unsafe {
                                        ldmatrix_x4_trans_shared_u32(
                                            sa + 2 * (so + kr * 72 + mc) as u32,
                                        )
                                    }
                                };
                                unroll!(4, ni, {
                                    let o = (mi * 4 + ni) * 4;
                                    let cin = [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]];
                                    // SAFETY: all 32 lanes reach this call together;
                                    // fragments follow the documented m16n8k16 layout.
                                    let d = unsafe { $mma(cin, af, bf[ni]) };
                                    acc[o] = d[0];
                                    acc[o + 1] = d[1];
                                    acc[o + 2] = d[2];
                                    acc[o + 3] = d[3];
                                });
                            });
                        });
                    }
                };
            }
            if bf16 == 0 {
                compute_tile!(mma_m16n8k16_f32_f16);
            } else {
                compute_tile!(mma_m16n8k16_f32_bf16);
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
                    if epi == 0 {
                        if gm < m && gn < n {
                            let idx = c_off + gm * ldc + gn;
                            // SAFETY: each (instance, row, col) is owned by one
                            // thread; extents are validated by the host.
                            unsafe {
                                let dst = c.as_mut_ptr().add(idx);
                                let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                                *dst = alpha * acc[(mi * 4 + ni) * 4 + j] + prev;
                            }
                        }
                    } else if gm < m && gn < n_store {
                        let idx = c_off + gm * ldc + gn;
                        let v = alpha * acc[(mi * 4 + ni) * 4 + j]
                            + bias[(inst % bias_mod) * bias_stride + gm];
                        let y = crate::kernels::activations::activate(act, v);
                        // SAFETY: as above, for the f16 output.
                        unsafe {
                            *c16.as_mut_ptr().add(idx) = f32_to_half_bits(y, bf16);
                        }
                    }
                });
            });
        });
    }

    /// Split-K epilogue of the fused conv path: fixed-order sum of the `splits`
    /// dense `m x n` partial products (`ws[(s * count + inst) * m * n + ..]`),
    /// `+ bias`, activation, f16 store (columns `>= n_store` skipped).
    #[kernel]
    pub fn hgemm_splitk_reduce_act16(
        ws: &[f32],
        bias: &[f32],
        mut c16: DisjointSlice<u16>,
        m: usize,
        n: usize,
        n_store: usize,
        ldc: usize,
        count: usize,
        splits: usize,
        alpha: f32,
        c_base: usize,
        c_stride: usize,
        c_mod: usize,
        act: u32,
        bias_stride: usize,
        bias_mod: usize,
        fmt: u32,
    ) {
        let e = thread::index_1d().get();
        let mn = m * n;
        if e < count * mn {
            let inst = e / mn;
            let r = e % mn;
            let col = r % n;
            if col < n_store {
                let row = r / n;
                let mut sum = 0.0f32;
                let mut s = 0usize;
                while s < splits {
                    sum += ws[(s * count + inst) * mn + r];
                    s += 1;
                }
                let v = alpha * sum + bias[(inst % bias_mod) * bias_stride + row];
                let y = crate::kernels::activations::activate(act, v);
                let idx = c_base + (inst % c_mod) * c_stride + row * ldc + col;
                // SAFETY: one thread per (instance, row, col); extents are
                // validated by the host launcher.
                unsafe {
                    *c16.as_mut_ptr().add(idx) = f32_to_half_bits(y, fmt);
                }
            }
        }
    }
}
