//! Shared-memory tiled f32 GEMM: `C = alpha * op(A) * op(B) + beta * C`
//! (row-major; `op` = identity or transpose, so NN / NT / TN / TT).
//!
//! Ported from the classic `gemm_nn / gemm_nt / gemm_tn` kernels of
//! `src-lib/blas_kernels.cu`, but with a 64x64 block tile, 16-deep K slices
//! and a 4x4 register tile per thread (256 threads per block).
//!
//! The kernel is batched: `count` independent instances share one launch (one
//! per group in a grouped conv).  Instance `i` uses
//! `a[a_base + (i % a_mod) * a_stride ..]`, and likewise for `b` and `c`, which
//! lets the caller broadcast one matrix across instances (`mod = 1`) or
//! advance it (`mod = count`).  `beta == 0` never reads `C`.
//!
//! Future tensor-core / wgmma variants slot in behind [`crate::GemmArgs`]
//! (see `OxideBackend::gemm`) without touching the conv drivers.

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
    ) {
        static mut AS: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 1024> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let tx = tid % 16;
        let ty = tid / 16;

        let tiles_n = n.div_ceil(64);
        let tiles_m = m.div_ceil(64);
        let block = thread::blockIdx_x() as usize;
        let per_instance = tiles_m * tiles_n;
        let inst = block / per_instance;
        let tile = block % per_instance;
        let row0 = (tile / tiles_n) * 64;
        let col0 = (tile % tiles_n) * 64;

        let a_off = a_base + (inst % a_mod) * a_stride;
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst % c_mod) * c_stride;

        let mut acc = [0.0f32; 16];

        let mut k0 = 0usize;
        while k0 < k {
            // Cooperative loads: 1024 elements per tile, 4 per thread, with
            // the fast-varying index chosen so global reads stay coalesced.
            let mut r = 0usize;
            while r < 4 {
                let e = tid + 256 * r;
                // A tile: AS[m_local * 16 + k_local]
                let (ml, kl) = if trans_a == 0 {
                    (e / 16, e % 16)
                } else {
                    (e % 64, e / 64)
                };
                let gm = row0 + ml;
                let gk = k0 + kl;
                let mut av = 0.0f32;
                if gm < m && gk < k {
                    let idx = if trans_a == 0 {
                        gm * lda + gk
                    } else {
                        gk * lda + gm
                    };
                    av = a[a_off + idx];
                }
                // B tile: BS[k_local * 64 + n_local]
                let (kl2, nl) = if trans_b == 0 {
                    (e / 64, e % 64)
                } else {
                    (e % 16, e / 16)
                };
                let gk2 = k0 + kl2;
                let gn = col0 + nl;
                let mut bv = 0.0f32;
                if gk2 < k && gn < n {
                    let idx = if trans_b == 0 {
                        gk2 * ldb + gn
                    } else {
                        gn * ldb + gk2
                    };
                    bv = b[b_off + idx];
                }
                unsafe {
                    AS[ml * 16 + kl] = av;
                    BS[kl2 * 64 + nl] = bv;
                }
                r += 1;
            }
            thread::sync_threads();

            let mut kk = 0usize;
            while kk < 16 {
                let mut ar = [0.0f32; 4];
                let mut br = [0.0f32; 4];
                let mut i = 0usize;
                while i < 4 {
                    unsafe {
                        ar[i] = AS[(ty + 16 * i) * 16 + kk];
                        br[i] = BS[kk * 64 + tx + 16 * i];
                    }
                    i += 1;
                }
                i = 0;
                while i < 4 {
                    let mut j = 0usize;
                    while j < 4 {
                        acc[i * 4 + j] += ar[i] * br[j];
                        j += 1;
                    }
                    i += 1;
                }
                kk += 1;
            }
            thread::sync_threads();
            k0 += 16;
        }

        let mut i = 0usize;
        while i < 4 {
            let gm = row0 + ty + 16 * i;
            let mut j = 0usize;
            while j < 4 {
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
                j += 1;
            }
            i += 1;
        }
    }
}
