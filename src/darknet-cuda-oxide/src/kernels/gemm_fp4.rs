//! Block-scaled NVFP4 GEMM emulated on the CUDA cores (`__dp4a`, no tensor
//! cores), ported from `raptor-fp4:src-fp4sim/fp4_sim_gemm.cu`
//! (`fp4sim_gemm_nt_packed_kernel`).
//!
//! Native FP4 `mma` needs sm_120a/sm_121a, which Ada does not have, so the
//! Blackwell NVFP4 recipe is emulated: for every 16-element block along K the
//! E2M1 codes are decoded to `{0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}` (twice
//! the real values, so `E2M1_MAX = 6` becomes 12) and combined four lanes at a
//! time by `__dp4a` into an *exact* i32 dot product.  The block contributes
//! `dot * 0.25 * decode(scale_a) * decode(scale_b)` to the f32 accumulator; the
//! `0.25` undoes the doubling and every factor is exactly representable in
//! f32, so the block dot itself is bit exact.
//!
//! Operand layout (both operands are "NT", i.e. contiguous along K):
//!
//! ```text
//! C[M, N] = alpha * gA * gB * sum_blocks dot_b * 0.25 * sA[m, b] * sB[n, b] + beta * C
//! ```
//!
//! `a_codes` is `M x blocks_per_row * 8` bytes (two 4-bit codes per byte, low
//! nibble = even K), `a_scales` is `M x blocks_per_row` raw e4m3 bytes, and the
//! same for `B` with `N` rows.  `k` is the true reduction extent; a ragged
//! `k % 16` tail is zero coded.  `alpha`, the global scales and `beta` follow
//! the C++ contract: the global scales are *device* floats read inside the
//! kernel, `beta == 0` never reads `C`.
//!
//! Two kernels: the tiled one and the one-thread-per-output reference that
//! the tests use as the oracle.

use super::quant_fp4::{BLOCK, decode_e4m3_magnitude, e2m1x2_lut};
use super::unroll::unroll;
use cuda_device::dotprod::dp4a_s32;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread};

/// Output tile rows / columns per block.
pub const FP4_BM: usize = 64;
pub const FP4_BN: usize = 64;
/// Threads per block: a 16x16 grid, 4x4 outputs each.
pub const FP4_THREADS: u32 = 256;
/// Bytes of packed codes per 16-element block.
pub const CODE_BYTES_PER_BLOCK: usize = 8;
/// Shared code tile: `FP4_BM x CODE_BYTES_PER_BLOCK` for each operand plus the
/// per-row scale bytes.  One `k` block is processed at a time (the `dp4a` dot
/// is over a whole block).
const CODE_TILE: usize = FP4_BM * CODE_BYTES_PER_BLOCK;

/// Decodes 16 packed E2M1 codes (`bytes[0..8]`, low nibble first) into four
/// i32 words of four packed `dp4a` lanes each.
#[device]
pub fn decode_row_words(bytes: &[u8; CODE_BYTES_PER_BLOCK], words: &mut [i32; 4]) {
    unroll!(4, w, {
        let mut packed = 0u32;
        unroll!(4, lane, {
            let nibble_index = w * 4 + lane;
            let byte = bytes[nibble_index / 2];
            let code = if nibble_index & 1 != 0 {
                byte >> 4
            } else {
                byte & 0x0F
            };
            packed |= ((e2m1x2_lut(code as u32) as u32) & 0xFF) << (8 * lane);
        });
        words[w] = packed as i32;
    });
}

/// Gathers one row of 8 packed code bytes out of a packed operand (shared tile
/// or global buffer), where consecutive rows are `row_stride` bytes apart.
///
/// # Safety
/// `base.add(row * row_stride + col_base + 8)` must be a readable range.
#[device]
pub unsafe fn gather_code_row(
    out: &mut [u8; CODE_BYTES_PER_BLOCK],
    base: *const u8,
    row_stride: usize,
    row: usize,
    col_base: usize,
) {
    let start = row * row_stride + col_base;
    let mut b = 0usize;
    while b < CODE_BYTES_PER_BLOCK {
        // SAFETY: the caller guarantees the whole row is readable.
        out[b] = unsafe { *base.add(start + b) };
        b += 1;
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Tiled packed NVFP4 GEMM, CUDA cores only (`__dp4a`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp4_gemm_nt_packed(
        a_codes: &[u8],
        a_scales: &[u8],
        b_codes: &[u8],
        b_scales: &[u8],
        mut c: DisjointSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        ldc: usize,
        global_a: &[f32],
        global_b: &[f32],
    ) {
        static mut AS: SharedArray<u8, CODE_TILE, 16> = SharedArray::UNINIT;
        static mut AS_SCALE: SharedArray<u8, FP4_BM, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u8, CODE_TILE, 16> = SharedArray::UNINIT;
        static mut BS_SCALE: SharedArray<u8, FP4_BN, 16> = SharedArray::UNINIT;
        let asp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let assp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS_SCALE) };
        let bsp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };
        let bssp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS_SCALE) };

        let tid = thread::threadIdx_x() as usize;
        let ty = tid / 16;
        let tx = tid % 16;
        let tiles_n = n.div_ceil(FP4_BN);
        let tile = thread::blockIdx_x() as usize;
        let row0 = (tile / tiles_n) * FP4_BM;
        let col0 = (tile % tiles_n) * FP4_BN;
        let blocks_per_row = k.div_ceil(BLOCK);

        // acc[mi * 4 + ni]
        let mut acc = [0.0f32; 16];

        let mut bk = 0usize;
        while bk < blocks_per_row {
            // Stage the code tiles (64 x 8 bytes each) and the scales.
            let mut i = tid;
            while i < FP4_BM * CODE_BYTES_PER_BLOCK {
                let r = i / CODE_BYTES_PER_BLOCK;
                let b = i % CODE_BYTES_PER_BLOCK;
                let gr = row0 + r;
                // SAFETY: `gr < m` implies the index is inside `a_codes`
                // (sized `m * blocks_per_row * 8`); the shared slot is in
                // bounds because `i < CODE_TILE`.
                let byte = if gr < m {
                    unsafe {
                        *a_codes.as_ptr().add(
                            gr * blocks_per_row * CODE_BYTES_PER_BLOCK
                                + bk * CODE_BYTES_PER_BLOCK
                                + b,
                        )
                    }
                } else {
                    0
                };
                unsafe { asp.add(i).write(byte) };
                i += FP4_THREADS as usize;
            }
            let mut i = tid;
            while i < FP4_BN * CODE_BYTES_PER_BLOCK {
                let r = i / CODE_BYTES_PER_BLOCK;
                let b = i % CODE_BYTES_PER_BLOCK;
                let gr = col0 + r;
                let byte = if gr < n {
                    unsafe {
                        *b_codes.as_ptr().add(
                            gr * blocks_per_row * CODE_BYTES_PER_BLOCK
                                + bk * CODE_BYTES_PER_BLOCK
                                + b,
                        )
                    }
                } else {
                    0
                };
                unsafe { bsp.add(i).write(byte) };
                i += FP4_THREADS as usize;
            }
            if tid < FP4_BM {
                let gr = row0 + tid;
                // SAFETY: `tid < 64` is inside `AS_SCALE`, and `gr < m` puts
                // the global index inside `a_scales`.
                unsafe {
                    assp.add(tid).write(if gr < m {
                        *a_scales.as_ptr().add(gr * blocks_per_row + bk)
                    } else {
                        0
                    });
                }
            }
            if tid < FP4_BN {
                let gr = col0 + tid;
                // SAFETY: as above for `BS_SCALE` / `b_scales`.
                unsafe {
                    bssp.add(tid).write(if gr < n {
                        *b_scales.as_ptr().add(gr * blocks_per_row + bk)
                    } else {
                        0
                    });
                }
            }
            thread::sync_threads();

            let mut aw = [[0i32; 4]; 4];
            let mut bw = [[0i32; 4]; 4];
            let mut ascale = [0.0f32; 4];
            let mut bscale = [0.0f32; 4];
            unroll!(4, mi, {
                let r = ty * 4 + mi;
                let mut bytes = [0u8; CODE_BYTES_PER_BLOCK];
                // SAFETY: the staged tile holds `FP4_BM` rows of 8 bytes, so
                // row `r < 64` is fully readable.
                unsafe {
                    gather_code_row(&mut bytes, asp as *const u8, CODE_BYTES_PER_BLOCK, r, 0)
                };
                decode_row_words(&bytes, &mut aw[mi]);
                // SAFETY: `r < 64` is inside `AS_SCALE`.
                ascale[mi] = decode_e4m3_magnitude(unsafe { *assp.add(r) } as u32);
            });
            unroll!(4, ni, {
                let col = tx * 4 + ni;
                let mut bytes = [0u8; CODE_BYTES_PER_BLOCK];
                // SAFETY: as above for the B tile.
                unsafe {
                    gather_code_row(&mut bytes, bsp as *const u8, CODE_BYTES_PER_BLOCK, col, 0)
                };
                decode_row_words(&bytes, &mut bw[ni]);
                // SAFETY: `col < 64` is inside `BS_SCALE`.
                bscale[ni] = decode_e4m3_magnitude(unsafe { *bssp.add(col) } as u32);
            });
            unroll!(4, mi, {
                unroll!(4, ni, {
                    let mut dot = 0i32;
                    unroll!(4, w, {
                        dot = dp4a_s32(aw[mi][w] as u32, bw[ni][w] as u32, dot);
                    });
                    let o = mi * 4 + ni;
                    // (2a)*(2b) = 4ab, so 0.25 undoes the doubling exactly.
                    acc[o] += dot as f32 * (0.25 * ascale[mi] * bscale[ni]);
                });
            });
            thread::sync_threads();
            bk += 1;
        }

        let scale_a = if global_a.is_empty() {
            1.0
        } else {
            global_a[0]
        };
        let scale_b = if global_b.is_empty() {
            1.0
        } else {
            global_b[0]
        };
        let factor = alpha * scale_a * scale_b;
        unroll!(4, mi, {
            let gm = row0 + ty * 4 + mi;
            unroll!(4, ni, {
                let gn = col0 + tx * 4 + ni;
                if gm < m && gn < n {
                    let idx = gm * ldc + gn;
                    // SAFETY: each (row, column) has exactly one owner thread;
                    // the extents are validated by the host launcher.
                    unsafe {
                        let dst = c.as_mut_ptr().add(idx);
                        let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                        *dst = factor * acc[mi * 4 + ni] + prev;
                    }
                }
            });
        });
    }

    /// One thread per output element, same math as [`fp4_gemm_nt_packed`];
    /// the correctness oracle the tiled kernel is tested against.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp4_gemm_nt_packed_reference(
        a_codes: &[u8],
        a_scales: &[u8],
        b_codes: &[u8],
        b_scales: &[u8],
        mut c: DisjointSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        ldc: usize,
        global_a: &[f32],
        global_b: &[f32],
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= m * n {
            return;
        }
        let gm = i / n;
        let gn = i % n;
        let blocks_per_row = k.div_ceil(BLOCK);
        let a_stride = blocks_per_row * CODE_BYTES_PER_BLOCK;
        let mut total = 0.0f32;
        let mut bk = 0usize;
        while bk < blocks_per_row {
            let mut abytes = [0u8; CODE_BYTES_PER_BLOCK];
            let mut bbytes = [0u8; CODE_BYTES_PER_BLOCK];
            // SAFETY: the host sizes both code buffers at
            // `rows * blocks_per_row * 8` and `gm < m`, `gn < n`.
            unsafe {
                gather_code_row(
                    &mut abytes,
                    a_codes.as_ptr(),
                    a_stride,
                    gm,
                    bk * CODE_BYTES_PER_BLOCK,
                );
                gather_code_row(
                    &mut bbytes,
                    b_codes.as_ptr(),
                    a_stride,
                    gn,
                    bk * CODE_BYTES_PER_BLOCK,
                );
            }
            let mut aw = [0i32; 4];
            let mut bw = [0i32; 4];
            decode_row_words(&abytes, &mut aw);
            decode_row_words(&bbytes, &mut bw);
            let mut dot = 0i32;
            unroll!(4, w, {
                dot = dp4a_s32(aw[w] as u32, bw[w] as u32, dot);
            });
            // SAFETY: `gm < m` / `gn < n` put both indices inside the scale
            // buffers, which the host sizes at `rows * blocks_per_row`.
            let sa =
                decode_e4m3_magnitude(
                    unsafe { *a_scales.as_ptr().add(gm * blocks_per_row + bk) } as u32
                );
            let sb =
                decode_e4m3_magnitude(
                    unsafe { *b_scales.as_ptr().add(gn * blocks_per_row + bk) } as u32
                );
            total += dot as f32 * (0.25 * sa * sb);
            bk += 1;
        }
        let scale_a = if global_a.is_empty() {
            1.0
        } else {
            global_a[0]
        };
        let scale_b = if global_b.is_empty() {
            1.0
        } else {
            global_b[0]
        };
        let factor = alpha * scale_a * scale_b;
        let dst = c.as_mut_ptr();
        // SAFETY: `gm * ldc + gn` is inside `c` (sized `m * ldc`).
        let dst = unsafe { dst.add(gm * ldc + gn) };
        // SAFETY: `gm * ldc + gn` is inside `c` (sized `m * ldc`).
        unsafe {
            let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
            *dst = factor * total + prev;
        }
    }
}
