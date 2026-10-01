//! Shared-memory tiled f32 GEMM: `C = alpha * op(A) * op(B) + beta * C`
//! (row-major; `op` = identity or transpose, so NN / NT / TN / TT).
//!
//! Ported from the classic `gemm_nn / gemm_nt / gemm_tn` kernels of
//! `src-lib/blas_kernels.cu`, but with a 64x64 block tile, 16-deep K slices,
//! a 4x4 register tile per thread (256 threads per block), `cp.async`
//! double-buffered shared stages and unrolled (register-resident) accumulators.
//!
//! The kernel is batched: `count` independent instances share one launch (one
//! per group in a grouped conv).  Instance `i` uses
//! `a[a_base + (i % a_mod) * a_stride ..]`, and likewise for `b` and `c`, which
//! lets the caller broadcast one matrix across instances (`mod = 1`) or
//! advance it (`mod = count`).  `beta == 0` never reads `C`.
//!
//! Future tensor-core / wgmma variants slot in behind [`crate::GemmArgs`]
//! (see `OxideBackend::gemm`) without touching the conv drivers.

use super::unroll::unroll;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;
pub const GEMM_THREADS: u32 = 256;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sgemm_tiled(
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
        // Two stages, filled with `cp.async` (16-byte when the operand rows are
        // contiguous and aligned, 4-byte otherwise; zero-filled outside the
        // matrix) while the previous stage is consumed.
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
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst_all % c_mod) * c_stride;

        let vec_a = trans_a == 0 && lda % 4 == 0 && a_off % 4 == 0;
        let vec_b = trans_b == 0 && ldb % 4 == 0 && b_off % 4 == 0;

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
                    // SAFETY: aligned in-bounds shared slot; `src` is a valid
                    // element of `a`, `bytes` is bounded by the row end.
                    unsafe {
                        cuda_device::async_copy::cp_async_cg_zfill_16(
                            asp.add(stage * 1024 + ml * 16 + kl) as *mut u32,
                            a.as_ptr().add(src) as *const u8,
                            if ok { bytes as u32 } else { 0 },
                        );
                    }
                }
                if vec_b {
                    let kl2 = tid / 16;
                    let nl = (tid % 16) * 4;
                    let gk2 = k0 + kl2;
                    let gn = col0 + nl;
                    let rem = if n > gn { n - gn } else { 0 };
                    let ok = gk2 < kend && rem > 0;
                    let src = if ok { b_off + gk2 * ldb + gn } else { 0 };
                    let bytes = if rem >= 4 { 16 } else { rem * 4 };
                    unsafe {
                        cuda_device::async_copy::cp_async_cg_zfill_16(
                            bsp.add(stage * 1024 + kl2 * 64 + nl) as *mut u32,
                            b.as_ptr().add(src) as *const u8,
                            if ok { bytes as u32 } else { 0 },
                        );
                    }
                }
                if !vec_a || !vec_b {
                    unroll!(4, r, {
                        let e = tid + 256 * r;
                        if !vec_a {
                            // A tile: AS[m_local * 16 + k_local]
                            let (ml, kl) = if trans_a == 0 { (e / 16, e % 16) } else { (e % 64, e / 64) };
                            let gm = row0 + ml;
                            let gk = k0 + kl;
                            let ok = gm < m && gk < kend;
                            let idx = if trans_a == 0 { gm * lda + gk } else { gk * lda + gm };
                            let src = if ok { a_off + idx } else { 0 };
                            unsafe {
                                cuda_device::async_copy::cp_async_ca_zfill_4(
                                    asp.add(stage * 1024 + ml * 16 + kl) as *mut u32,
                                    a.as_ptr().add(src) as *const u8,
                                    if ok { 4 } else { 0 },
                                );
                            }
                        }
                        if !vec_b {
                            // B tile: BS[k_local * 64 + n_local]
                            let (kl2, nl) = if trans_b == 0 { (e / 64, e % 64) } else { (e % 16, e / 16) };
                            let gk2 = k0 + kl2;
                            let gn = col0 + nl;
                            let ok = gk2 < kend && gn < n;
                            let idx = if trans_b == 0 { gk2 * ldb + gn } else { gn * ldb + gk2 };
                            let src = if ok { b_off + idx } else { 0 };
                            unsafe {
                                cuda_device::async_copy::cp_async_ca_zfill_4(
                                    bsp.add(stage * 1024 + kl2 * 64 + nl) as *mut u32,
                                    b.as_ptr().add(src) as *const u8,
                                    if ok { 4 } else { 0 },
                                );
                            }
                        }
                    });
                }
            }
            unsafe {
                cuda_device::async_copy::cp_async_commit_group();
                // Tile `it` may still be in flight; tile `it - 1` must have landed.
                cuda_device::async_copy::cp_async_wait_group(1);
            }
            thread::sync_threads();

            if it > 0 {
                let ao = ((it - 1) & 1) * 1024;
                let mut kk = 0usize;
                while kk < 16 {
                    let mut ar = [0.0f32; 4];
                    let mut br = [0.0f32; 4];
                    unroll!(4, i, {
                        unsafe {
                            ar[i] = AS[ao + (ty + 16 * i) * 16 + kk];
                            br[i] = BS[ao + kk * 64 + tx + 16 * i];
                        }
                    });
                    unroll!(4, i, {
                        unroll!(4, j, {
                            acc[i * 4 + j] += ar[i] * br[j];
                        });
                    });
                    kk += 1;
                }
            }
            // Stage `(it - 1) & 1` is refilled by the next iteration's issue.
            thread::sync_threads();
            it += 1;
        }

        unroll!(4, i, {
            let gm = row0 + ty + 16 * i;
            unroll!(4, j, {
                let gn = col0 + tx + 16 * j;
                if gm < m && gn < n {
                    let idx = c_off + gm * ldc + gn;
                    // SAFETY: each (instance, row, col) is owned by exactly
                    // one thread; `idx` is bounds-checked by the host launcher
                    // (`OxideBackend::gemm` validates buffer extents).
                    unsafe {
                        let dst = c.as_mut_ptr().add(idx);
                        let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                        *dst = alpha * acc[i * 4 + j] + prev;
                    }
                }
            });
        });
    }

    /// Split-K epilogue: `C[inst] = alpha * sum_s ws[s * count + inst] + beta * C[inst]`
    /// (deterministic fixed-order sum).  `ws` holds `splits * count` dense
    /// `m x n` partial products written by the GEMM kernels with `ldc = n`.
    #[kernel]
    pub fn gemm_splitk_reduce(
        ws: &[f32],
        mut c: DisjointSlice<f32>,
        m: usize,
        n: usize,
        ldc: usize,
        count: usize,
        splits: usize,
        alpha: f32,
        beta: f32,
        c_base: usize,
        c_stride: usize,
        c_mod: usize,
    ) {
        let e = thread::index_1d().get();
        let mn = m * n;
        if e < count * mn {
            let inst = e / mn;
            let r = e % mn;
            let mut sum = 0.0f32;
            let mut s = 0usize;
            while s < splits {
                sum += ws[(s * count + inst) * mn + r];
                s += 1;
            }
            let idx = c_base + (inst % c_mod) * c_stride + (r / n) * ldc + r % n;
            // SAFETY: one thread per (instance, row, col); extents validated by
            // the host launcher (`OxideBackend::gemm`).
            unsafe {
                let dst = c.as_mut_ptr().add(idx);
                let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                *dst = alpha * sum + prev;
            }
        }
    }
}
