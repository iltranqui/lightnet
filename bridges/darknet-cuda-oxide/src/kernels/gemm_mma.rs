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
//! Tiles are staged in shared memory as TF32 bit patterns.  `ldmatrix` is not
//! used: it moves 16-bit elements only, and a TF32 fragment register is one
//! 32-bit element, so the fragments are read directly from shared memory
//! (row strides of 20 / 72 words make those reads bank-conflict free).
//!
//! Fragment layout (PTX m16n8k8 tf32, `g = lane / 4`, `t = lane % 4`):
//!   A regs: (g, t) (g+8, t) (g, t+4) (g+8, t+4);  B regs: (t, g) (t+4, g);
//!   C regs: (g, 2t) (g, 2t+1) (g+8, 2t) (g+8, 2t+1).

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
    ) {
        static mut AS: SharedArray<u32, 1280> = SharedArray::UNINIT; // 64 x 20
        static mut BS: SharedArray<u32, 1152> = SharedArray::UNINIT; // 16 x 72

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
        let inst = block / per_instance;
        let tile = block % per_instance;
        let row0 = (tile / tiles_n) * 64;
        let col0 = (tile % tiles_n) * 64;

        let a_off = a_base + (inst % a_mod) * a_stride;
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst % c_mod) * c_stride;

        // acc[(mi * 4 + ni) * 4 + j]
        let mut acc = [0.0f32; 32];

        let mut k0 = 0usize;
        while k0 < k {
            // 1024 elements per tile, 8 per thread; fast index chosen so
            // global reads stay coalesced for each transpose case.
            let mut r = 0usize;
            while r < 8 {
                let e = tid + 128 * r;
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
                    AS[ml * 20 + kl] = cvt_rna_tf32_f32(av);
                    BS[kl2 * 72 + nl] = cvt_rna_tf32_f32(bv);
                }
                r += 1;
            }
            thread::sync_threads();

            let mut ks = 0usize;
            while ks < 16 {
                let mut bf = [[0u32; 2]; 4];
                let mut ni = 0usize;
                while ni < 4 {
                    let col = wn + ni * 8 + g;
                    unsafe {
                        bf[ni][0] = BS[(ks + t) * 72 + col];
                        bf[ni][1] = BS[(ks + t + 4) * 72 + col];
                    }
                    ni += 1;
                }
                let mut mi = 0usize;
                while mi < 2 {
                    let row = wm + mi * 16 + g;
                    let af = unsafe {
                        [
                            AS[row * 20 + ks + t],
                            AS[(row + 8) * 20 + ks + t],
                            AS[row * 20 + ks + t + 4],
                            AS[(row + 8) * 20 + ks + t + 4],
                        ]
                    };
                    ni = 0;
                    while ni < 4 {
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
                        ni += 1;
                    }
                    mi += 1;
                }
                ks += 8;
            }
            thread::sync_threads();
            k0 += 16;
        }

        let mut mi = 0usize;
        while mi < 2 {
            let mut ni = 0usize;
            while ni < 4 {
                let mut j = 0usize;
                while j < 4 {
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
                    j += 1;
                }
                ni += 1;
            }
            mi += 1;
        }
    }
}
