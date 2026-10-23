//! FP8 tensor-core GEMM for `sm_89+` (Ada), plus the f32 -> FP8 quantisation
//! helpers the backend needs.
//!
//! `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (and the e5m2 /
//! mixed combinations) is the first FP8 shape Ada exposes; it keeps the
//! fragment shape of the existing int8 `mma_m16n8k32_s32_s8` kernel (A = 4
//! registers, B = 2, C = 4), so the fragment loads here are 32-bit shared
//! loads.  (`ldmatrix` also works: two adjacent 8-bit values are one 16-bit
//! element in this layout, and the tuning experiment of `plan_fp8.md` Task 5
//! sub-step D verified it, but it was 14% slower than the word loads.)
//!
//! Layout (`mma.sync` m16n8k32, 8-bit operands).  With `group = lane / 4`
//! and `thread = lane % 4`:
//!
//! ```text
//! A (16 x 32, row-major, `a[i]` packs bytes 4i..4i+4):
//!   row = group + (i in 8..16 ? 8 : 0), col = thread * 4 + (i & 3) + (i in 8..16 ? 16 : 0)
//! B (32 x 8, column-major i.e. `b[k][n]`, `b[i]` packs bytes 4i..4i+4):
//!   row = thread * 4 + (i & 3) + (i in 4..8 ? 16 : 0), col = group
//! C (16 x 8, f32): row = group + (i in 2..4 ? 8 : 0), col = thread * 2 + (i & 1)
//! ```
//!
//! Tiling: 64x64 output block tile, BK = 32 (one mma k step), 128 threads =
//! 4 warps in a 2x2 grid, each warp owning a 32x32 sub-tile = 2 (m16) x 4
//! (n8) mma tiles.  Shared memory holds two stages of both operand tiles with
//! a 16-byte pad on the K row stride (32 -> 48, bank-conflict free and
//! 16-byte aligned for `cp.async`), and one `sync_threads` per K step: A is
//! copied with `cp.async` (when K-contiguous and 16-byte aligned) while B goes
//! through registers, the `[k][n]` case as register-transposed 4x4 byte blocks
//! (`fp8_stage_t4`).  `NN` / `NT` / `TN` / `TT` are all supported, batched
//! through the same `(base, stride, modulus)` views as [`super::gemm_mma`],
//! `beta == 0` never reads `C`, and out-of-range loads are zero filled.
//! `sgemm_fp8_mma_e4m3` is a copy of the generic kernel with the e4m3 x e4m3
//! mma hard-wired (the runtime format `match` cost 38%); keep the two bodies
//! in sync.
//!
//! Accuracy: operands are rounded to the format by the caller-supplied scale
//! (see [`crate::fp8`]); products and accumulation are f32.  e4m3 carries 3
//! mantissa bits (relative error <= 2^-4 per operand), e5m2 carries 2.

use super::convert::half_bits_to_f32;
use super::unroll::unroll;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::convert::{
    cvt_f32_f16x2_lo, cvt_rn_f16x2_e4m3x2, cvt_rn_f16x2_e5m2x2, cvt_rn_satfinite_e4m3x2_f32,
    cvt_rn_satfinite_e5m2x2_f32,
};
use cuda_device::wmma::{
    mma_m16n8k16_f32_f16, mma_m16n8k16_fp8_f32_e4m3_e4m3, mma_m16n8k16_fp8_f32_e4m3_e5m2,
    mma_m16n8k16_fp8_f32_e5m2_e4m3, mma_m16n8k16_fp8_f32_e5m2_e5m2, mma_m16n8k32_fp8_f32_e4m3_e4m3,
    mma_m16n8k32_fp8_f32_e4m3_e5m2, mma_m16n8k32_fp8_f32_e5m2_e4m3, mma_m16n8k32_fp8_f32_e5m2_e5m2,
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread};

/// Output tile rows per block.
pub const F8_BM: usize = 64;
/// Output tile columns per block.
pub const F8_BN: usize = 64;
/// K elements staged per mma step (the m16n8k32 shape).
pub const F8_BK: usize = 32;
/// Threads per block (4 warps, 2x2 over the 64x64 tile).
pub const F8_THREADS: u32 = 128;
/// Shared row stride of the A tile in bytes (A is staged `[m][k]`): 32 data
/// bytes + 8 bytes of bank padding.  A fragment reads are 4 consecutive K
/// bytes at `m * LD_A + 4t`, so the stride must be a multiple of 4 for the
/// 32-bit access to stay aligned.
const LD_A: usize = F8_BK + 16;
/// Shared row stride of the B tile in bytes (B is staged `[n][k]`): 32 data
/// bytes + 4 bytes of padding.  B is *not* stored `[k][n]` like in global
/// memory: an mma B register holds four K values of one column, which in the
/// global layout are 32 bytes apart, so a `[k][n]` tile could only be read one
/// byte at a time.  Transposing it in shared memory makes the four bytes
/// consecutive again.
const LD_B: usize = F8_BK + 4;
/// 16-byte chunks per operand row when staging a tile (`F8_BK / 16`).
const CHUNKS_PER_ROW: usize = F8_BK / 16;

/// f32 -> e4m3 bits, round to nearest even, saturating (`|x| > 448` -> 448).
#[device]
pub fn f32_to_e4m3_bits(value: f32) -> u8 {
    (cvt_rn_satfinite_e4m3x2_f32(value, 0.0) & 0xFF) as u8
}

/// f32 -> e5m2 bits, round to nearest even, saturating (`|x| > 57344` -> 57344).
#[device]
pub fn f32_to_e5m2_bits(value: f32) -> u8 {
    (cvt_rn_satfinite_e5m2x2_f32(value, 0.0) & 0xFF) as u8
}

/// e4m3 bits -> f32 (exact).
#[device]
pub fn e4m3_bits_to_f32(bits: u8) -> f32 {
    cvt_f32_f16x2_lo(cvt_rn_f16x2_e4m3x2(bits as u16) & 0xFFFF)
}

/// e5m2 bits -> f32 (exact).
#[device]
pub fn e5m2_bits_to_f32(bits: u8) -> f32 {
    cvt_f32_f16x2_lo(cvt_rn_f16x2_e5m2x2(bits as u16) & 0xFFFF)
}

/// f32 -> FP8 byte in `fmt` (`0` = e4m3, `1` = e5m2), after multiplying by
/// `scale` (the quantisation multiply; `1.0` for a pure format cast).
#[device]
pub fn quantize_f32(value: f32, scale: f32, fmt: u32) -> u8 {
    let scaled = value * scale;
    if fmt == 0 {
        f32_to_e4m3_bits(scaled)
    } else {
        f32_to_e5m2_bits(scaled)
    }
}

/// Logical element count `len` of a possibly larger, reused buffer of
/// `buf_len` elements (never beyond the buffer).
#[device]
pub fn bounded_len(len: u32, buf_len: usize) -> usize {
    let len = len as usize;
    if len < buf_len { len } else { buf_len }
}

/// Quantisation factor `max_finite(fmt) / amax` (`fmt` 0 = e4m3, 1 = e5m2);
/// `1.0` when amax is zero or non-finite.
#[device]
pub fn fp8_factor(amax: f32, fmt: u32) -> f32 {
    let max_finite = if fmt == 0 { 448.0 } else { 57344.0 };
    if amax > 0.0 && amax.is_finite() {
        max_finite / amax
    } else {
        1.0
    }
}

/// One `mma.m16n8k32` with the FP8 pair selected by `fmt_a` / `fmt_b`.
/// All 32 lanes must call it; `a` / `b` follow the layout documented above.
#[device]
pub fn mma_f8_k32(c: [f32; 4], a: [u32; 4], b: [u32; 2], fmt_a: u32, fmt_b: u32) -> [f32; 4] {
    // SAFETY: the caller guarantees whole-warp participation and the
    // documented fragment layout; `a` / `b` hold valid FP8 bit patterns.
    unsafe {
        match (fmt_a, fmt_b) {
            (0, 0) => mma_m16n8k32_fp8_f32_e4m3_e4m3(c, a, b),
            (0, 1) => mma_m16n8k32_fp8_f32_e4m3_e5m2(c, a, b),
            (1, 0) => mma_m16n8k32_fp8_f32_e5m2_e4m3(c, a, b),
            _ => mma_m16n8k32_fp8_f32_e5m2_e5m2(c, a, b),
        }
    }
}

/// [`mma_f8_k32`] fixed to e4m3 x e4m3: no runtime format `match` in the hot loop.
#[device]
pub fn mma_f8_k32_e4m3(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    // SAFETY: as [`mma_f8_k32`].
    unsafe { mma_m16n8k32_fp8_f32_e4m3_e4m3(c, a, b) }
}

/// One `mma.m16n8k16` with the FP8 pair selected by `fmt_a` / `fmt_b`.
/// All 32 lanes must call it; `a` is two registers and `b` one.
#[device]
pub fn mma_f8_k16(c: [f32; 4], a: [u32; 2], b: u32, fmt_a: u32, fmt_b: u32) -> [f32; 4] {
    // SAFETY: the caller guarantees whole-warp participation and valid
    // fragments for the documented layout.
    unsafe {
        match (fmt_a, fmt_b) {
            (0, 0) => mma_m16n8k16_fp8_f32_e4m3_e4m3(c, a, b),
            (0, 1) => mma_m16n8k16_fp8_f32_e4m3_e5m2(c, a, b),
            (1, 0) => mma_m16n8k16_fp8_f32_e5m2_e4m3(c, a, b),
            _ => mma_m16n8k16_fp8_f32_e5m2_e5m2(c, a, b),
        }
    }
}

/// Stages one 16-byte chunk of an operand tile into registers as four u32
/// words (little-endian byte order), zero-filling out-of-range elements.
///
/// `row` is the tile-local row (0..64), `kb` the K offset inside the tile
/// (0 or 16).  `trans == 0`: element (row, k) lives at `grow * ldr + gk`
/// (the tile is contiguous along K); `trans == 1`: at `gk * ldr + grow`.
/// Note that this is *not* `trans_a` / `trans_b`: a "NN" GEMM stores B as
/// `[k][n]`, which is contiguous along N, so B is staged with `trans = 1`.
#[allow(clippy::too_many_arguments)]
#[device]
pub fn fp8_stage16(
    dst: &mut [u32; 4],
    buf: &[u8],
    base: usize,
    ldr: usize,
    k: usize,
    rows: usize,
    row0: usize,
    row: usize,
    k0: usize,
    kb: usize,
    trans: u32,
) {
    let grow = row0 + row;
    let gk = k0 + kb;
    let remaining = if gk < k { k - gk } else { 0 };
    let bytes = if remaining > 16 { 16 } else { remaining };
    if grow >= rows || bytes == 0 {
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 0;
        return;
    }
    let index = if trans == 0 {
        grow * ldr + gk
    } else {
        gk * ldr + grow
    };
    let src = unsafe { buf.as_ptr().add(base + index) };
    // Only the K-contiguous case can be read as four aligned words; with
    // `trans == 1` the 16 elements are `ldr` bytes apart.
    let aligned = (src as usize) % 4 == 0;
    if trans == 0 && bytes == 16 && aligned {
        // SAFETY: `src` points at 16 valid bytes of `buf` (the row is inside
        // the matrix and 16 K elements remain) and is 4-byte aligned.
        unsafe {
            let p = src as *const u32;
            dst[0] = p.read();
            dst[1] = p.add(1).read();
            dst[2] = p.add(2).read();
            dst[3] = p.add(3).read();
        }
        return;
    }
    dst[0] = 0;
    dst[1] = 0;
    dst[2] = 0;
    dst[3] = 0;
    for j in 0..16usize {
        if j >= bytes {
            break;
        }
        let byte = if trans == 0 {
            // SAFETY: `index + j` is inside the row extent guarded above.
            unsafe { *buf.as_ptr().add(base + index + j) }
        } else {
            // SAFETY: element (k0 + kb + j, grow) of the stored operand.
            unsafe { *buf.as_ptr().add(base + (gk + j) * ldr + grow) }
        };
        dst[j / 4] |= (byte as u32) << (8 * (j % 4));
    }
}

/// Stages a `trans == 1` operand tile (stored `[k][n]`, contiguous along N)
/// as one 4 (k) x 4 (n) byte block per thread, transposed in registers.
///
/// `ng` (0..16) picks four columns `col0 + 4 ng ..`, `kg` (0..8) four K rows
/// `k0 + 4 kg ..`.  `dst[j]` is the word of K-consecutive bytes (bit `8 r` =
/// k offset `r`) for column `4 ng + j`, ready to store at `[n][k]`.  A full
/// aligned block takes four word loads instead of sixteen byte loads; edge
/// blocks fall back to guarded byte loads and zero fill.
#[allow(clippy::too_many_arguments)]
#[device]
pub fn fp8_stage_t4(
    dst: &mut [u32; 4],
    buf: &[u8],
    base: usize,
    ldr: usize,
    k: usize,
    cols: usize,
    col0: usize,
    ng: usize,
    k0: usize,
    kg: usize,
) {
    let gn = col0 + 4 * ng;
    let gk = k0 + 4 * kg;
    let at = unsafe { buf.as_ptr().add(base) };
    if gn + 4 <= cols && gk + 4 <= k && ldr % 4 == 0 && ((at as usize) + gn) % 4 == 0 {
        // SAFETY: rows `gk..gk + 4` are inside the K extent and the four
        // columns inside N, so each word is 4 valid, 4-byte aligned bytes.
        let (w0, w1, w2, w3) = unsafe {
            (
                (at.add(gk * ldr + gn) as *const u32).read(),
                (at.add((gk + 1) * ldr + gn) as *const u32).read(),
                (at.add((gk + 2) * ldr + gn) as *const u32).read(),
                (at.add((gk + 3) * ldr + gn) as *const u32).read(),
            )
        };
        let lo02 = (w0 & 0x00FF_00FF) | ((w1 & 0x00FF_00FF) << 8);
        let lo13 = ((w0 >> 8) & 0x00FF_00FF) | (w1 & 0xFF00_FF00);
        let hi02 = (w2 & 0x00FF_00FF) | ((w3 & 0x00FF_00FF) << 8);
        let hi13 = ((w2 >> 8) & 0x00FF_00FF) | (w3 & 0xFF00_FF00);
        dst[0] = (lo02 & 0xFFFF) | (hi02 << 16);
        dst[1] = (lo13 & 0xFFFF) | (hi13 << 16);
        dst[2] = (lo02 >> 16) | (hi02 & 0xFFFF_0000);
        dst[3] = (lo13 >> 16) | (hi13 & 0xFFFF_0000);
        return;
    }
    dst[0] = 0;
    dst[1] = 0;
    dst[2] = 0;
    dst[3] = 0;
    for r in 0..4usize {
        if gk + r >= k {
            break;
        }
        for j in 0..4usize {
            if gn + j < cols {
                // SAFETY: element (gk + r, gn + j) of the stored operand.
                let byte = unsafe { *at.add((gk + r) * ldr + gn + j) };
                dst[j] |= (byte as u32) << (8 * r);
            }
        }
    }
}

/// Issues one thread's 16-byte chunk of an operand tile into shared memory
/// at `dst` (a 16-byte aligned shared word pointer).  `vec != 0`: the operand
/// is K-contiguous with 16-byte aligned rows (`trans == 0`), so the chunk is
/// one `cp.async.cg` that zero-fills past the K edge and past the last row;
/// otherwise the chunk goes through [`fp8_stage16`] and plain shared stores.
///
/// # Safety
/// `dst` points at 4 writable shared words; `vec != 0` requires
/// `ldr % 16 == 0`, `(buf + base) % 16 == 0` and `k0 + kb` a multiple of 16.
#[allow(clippy::too_many_arguments)]
#[device]
pub unsafe fn fp8_issue16(
    dst: *mut u32,
    buf: &[u8],
    base: usize,
    ldr: usize,
    k: usize,
    rows: usize,
    row0: usize,
    row: usize,
    k0: usize,
    kb: usize,
    trans: u32,
    vec: u32,
) {
    if vec != 0 {
        let grow = row0 + row;
        let gk = k0 + kb;
        let remaining = if gk < k { k - gk } else { 0 };
        let ok = grow < rows && remaining > 0;
        let bytes = if remaining > 16 { 16 } else { remaining };
        let src = if ok { base + grow * ldr + gk } else { 0 };
        // SAFETY: when `ok`, `src` addresses `bytes` valid bytes of `buf`;
        // otherwise 0 bytes are read (zero fill) from element 0.
        unsafe {
            cuda_device::async_copy::cp_async_cg_zfill_16(
                dst,
                buf.as_ptr().add(src),
                if ok { bytes as u32 } else { 0 },
            );
        }
    } else {
        let mut r = [0u32; 4];
        fp8_stage16(&mut r, buf, base, ldr, k, rows, row0, row, k0, kb, trans);
        // SAFETY: the caller guarantees 4 writable shared words at `dst`.
        unsafe {
            dst.write(r[0]);
            dst.add(1).write(r[1]);
            dst.add(2).write(r[2]);
            dst.add(3).write(r[3]);
        }
    }
}

/// Block size of `fp8_amax_pair_kernel` (one tile per block).
pub const F8_AMAX_BLOCK: usize = 256;
/// Amax history slots per quantised tensor role (delayed scaling).
pub const F8_HIST_LEN: usize = 16;
/// Word offset of operand B's amax slot in the pair kernels: 32 floats = 128
/// bytes, a separate cache line from A's slot at word 0.
pub const F8_AMAX_B_SLOT: usize = 32;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One warp, one `mma.m16n8k32`: `c[i * 32 + lane] = D[lane][i]` with
    /// `A` and `B` given as *raw fragments* (16 and 8 bytes per lane).  Exists
    /// so the fragment layout can be validated on its own, without the tiling
    /// around it.  `a.len() >= 128`, `b.len() >= 64`, `c.len() >= 128`.
    #[kernel]
    pub fn fp8_mma_probe(a: &[u8], b: &[u8], mut c: DisjointSlice<f32>, fmt_a: u32, fmt_b: u32) {
        let lane = thread::threadIdx_x() as usize;
        if lane >= 32 {
            return;
        }
        let mut af = [0u32; 4];
        let mut bf = [0u32; 2];
        // SAFETY: the host guarantees `a.len() >= 512` and `b.len() >= 256`.
        unsafe {
            let pa = a.as_ptr().add(lane * 16) as *const u32;
            for i in 0..4 {
                af[i] = pa.add(i).read();
            }
            let pb = b.as_ptr().add(lane * 8) as *const u32;
            bf[0] = pb.read();
            bf[1] = pb.add(1).read();
        }
        let d = mma_f8_k32([0.0f32; 4], af, bf, fmt_a, fmt_b);
        let out = c.as_mut_ptr();
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: `i * 32 + lane < 128 <= c.len()` for every lane, checked
            // by the host launcher; each (i, lane) slot has one owner.
            unsafe { out.add(i * 32 + lane).write(d[i]) };
            i += 1;
        }
    }

    /// `m16n8k16` twin of [`fp8_mma_probe`]: `a` is 32 lanes x 8 bytes of A
    /// fragments (256 bytes), `b` is 32 lanes x 4 bytes of B fragments (128
    /// bytes), `c.len() >= 128`.
    #[kernel]
    pub fn fp8_mma_probe16(a: &[u8], b: &[u8], mut c: DisjointSlice<f32>, fmt_a: u32, fmt_b: u32) {
        let lane = thread::threadIdx_x() as usize;
        let mut af = [0u32; 2];
        let bf;
        // SAFETY: the host guarantees `a.len() >= 256` and `b.len() >= 128`.
        unsafe {
            let pa = a.as_ptr().add(lane * 8) as *const u32;
            af[0] = pa.read();
            af[1] = pa.add(1).read();
            bf = (b.as_ptr().add(lane * 4) as *const u32).read();
        }
        let d = mma_f8_k16([0.0f32; 4], af, bf, fmt_a, fmt_b);
        let out = c.as_mut_ptr();
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: `i * 32 + lane < 128 <= c.len()` for every lane.
            unsafe { out.add(i * 32 + lane).write(d[i]) };
            i += 1;
        }
    }

    /// Control probe with the f16 `mma.m16n8k16`: identical harness (single
    /// warp, raw fragments, `out[i * 32 + lane]`), f16 operands.  `a` is 32
    /// lanes x 8 halves (256), `b` is 32 lanes x 4 halves (128).  Used to tell a harness
    /// bug apart from an FP8-hardware problem.
    #[kernel]
    pub fn f16_mma_probe16(a: &[u16], b: &[u16], mut c: DisjointSlice<f32>) {
        let lane = thread::threadIdx_x() as usize;
        let mut af = [0u32; 4];
        let mut bf = [0u32; 2];
        // SAFETY: the host guarantees `a.len() >= 256` and `b.len() >= 128`;
        // one lane owns 8 A halves and 4 B halves.
        unsafe {
            let pa = a.as_ptr().add(lane * 8) as *const u32;
            let mut i = 0usize;
            while i < 4 {
                af[i] = pa.add(i).read();
                i += 1;
            }
            let pb = b.as_ptr().add(lane * 4) as *const u32;
            bf[0] = pb.read();
            bf[1] = pb.add(1).read();
        }
        // SAFETY: single warp, documented f16 fragment layout.
        let d = unsafe { mma_m16n8k16_f32_f16([0.0f32; 4], af, bf) };
        let out = c.as_mut_ptr();
        let mut i = 0usize;
        while i < 4 {
            // SAFETY: `i * 32 + lane < 128 <= c.len()` for every lane.
            unsafe { out.add(i * 32 + lane).write(d[i]) };
            i += 1;
        }
    }

    /// `out[off] = max(|x[i]|)` over the first `len` elements of `x` (`len` is
    /// the logical element count: `x` may be a larger, reused buffer whose tail
    /// holds stale data from an earlier use), ignoring non-finite values.  `out` is
    /// a single f32 slot (read as `u32` bits) that must already hold 0.  Each
    /// 256-thread block reduces in shared memory and issues one `atomicMax` on the
    /// bit pattern (valid for non-negative floats); launch whole 256-thread blocks.
    #[kernel]
    pub fn fp8_amax_f32_kernel(x: &[f32], mut out: DisjointSlice<f32>, off: u32, len: u32) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let i = thread::blockIdx_x() as usize * 256 + tid;
        let mut bits = 0u32;
        if i < bounded_len(len, x.len()) {
            let value = x[i];
            if value.is_finite() {
                bits = value.abs().to_bits();
            }
        }
        // SAFETY: `tid < 256`, the shared array's length; every access between
        // two barriers touches a distinct or read-only slot.
        unsafe { sh.add(tid).write(bits) };
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                let other = unsafe { sh.add(tid + stride).read() };
                let mine = unsafe { sh.add(tid).read() };
                if other > mine {
                    unsafe { sh.add(tid).write(other) };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: the host guarantees `out.len() >= 1`; slot 0 is only ever
            // touched through this atomic while the kernel runs.  The slot is an
            // f32 buffer; an all-zero word is both 0.0 and the zero bit pattern
            // `fetch_max` needs.
            let slot = unsafe { out.as_mut_ptr().add(off as usize) } as *mut u32;
            let atomic = unsafe { DeviceAtomicU32::from_ptr(slot) };
            let _ = atomic.fetch_max(unsafe { sh.read() }, AtomicOrdering::Relaxed);
        }
    }

    /// [`fp8_amax_f32_kernel`] on 16-bit storage: `hfmt` (`0` = f16, `1` =
    /// bf16) selects how each `x[i]` is widened (exact), so the slot gets the
    /// value the f32 kernel would get from the widened tensor.
    #[kernel]
    pub fn fp8_amax_half_kernel(x: &[u16], mut out: DisjointSlice<f32>, off: u32, hfmt: u32, len: u32) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let i = thread::blockIdx_x() as usize * 256 + tid;
        let mut bits = 0u32;
        if i < bounded_len(len, x.len()) {
            let value = half_bits_to_f32(x[i], hfmt);
            if value.is_finite() {
                bits = value.abs().to_bits();
            }
        }
        // SAFETY: `tid < 256`, the shared array's length; every access between
        // two barriers touches a distinct or read-only slot.
        unsafe { sh.add(tid).write(bits) };
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                let other = unsafe { sh.add(tid + stride).read() };
                let mine = unsafe { sh.add(tid).read() };
                if other > mine {
                    unsafe { sh.add(tid).write(other) };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: as `fp8_amax_f32_kernel`.
            let slot = unsafe { out.as_mut_ptr().add(off as usize) } as *mut u32;
            let atomic = unsafe { DeviceAtomicU32::from_ptr(slot) };
            let _ = atomic.fetch_max(unsafe { sh.read() }, AtomicOrdering::Relaxed);
        }
    }

    /// [`fp8_quantize_kernel`] reading 16-bit storage (`hfmt` as in
    /// [`fp8_amax_half_kernel`]): the same factor, scale pair and bytes as the
    /// f32 kernel gives on the widened tensor.
    #[kernel]
    pub fn fp8_quantize_half_kernel(
        x: &[u16],
        amax: &[f32],
        mut out: DisjointSlice<u8>,
        mut scale: DisjointSlice<f32>,
        fmt: u32,
        off: u32,
        hfmt: u32,
        len: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let factor = fp8_factor(amax[off as usize], fmt);
        if let Some(o) = scale.get_mut(thread::index_1d()) {
            *o = if i == 0 { factor } else { 1.0 / factor };
        }
        if i >= bounded_len(len, x.len()) {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = quantize_f32(half_bits_to_f32(x[i], hfmt), factor, fmt);
        }
    }

    /// `out[i] = quantize(x[i] * factor, fmt)` with `factor = max_finite(fmt) /
    /// amax` taken from the amax slot (1.0 when amax is zero or non-finite).
    /// The first two threads also write the operand's device scale pair
    /// `scale = [factor, 1 / factor]`, which the mma epilogue consumes, so no
    /// separate scale launch is needed and the host never reads either word.
    /// The launch must cover at least two threads (block sizes always do).
    #[kernel]
    pub fn fp8_quantize_kernel(
        x: &[f32],
        amax: &[f32],
        mut out: DisjointSlice<u8>,
        mut scale: DisjointSlice<f32>,
        fmt: u32,
        off: u32,
        len: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let factor = fp8_factor(amax[off as usize], fmt);
        if let Some(o) = scale.get_mut(thread::index_1d()) {
            *o = if i == 0 { factor } else { 1.0 / factor };
        }
        if i >= bounded_len(len, x.len()) {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = quantize_f32(x[i], factor, fmt);
        }
    }

    /// Delayed-scaling quantiser: one pass over `x`.  `hist[base..base + 16]`
    /// is the key's amax history.  Call `n` of a key has `w = n % 16` and
    /// `z = (n + 2) % 16` (host-computed): the factor is `max_finite(fmt) /
    /// max(hist[s])` over every slot except `w` and `z`, so nothing read here
    /// is written by this launch; the block-reduced amax of `x` (non-finite
    /// ignored) goes into slot `w` with one `atomicMax` per block (the slot was
    /// zeroed two calls ago), and thread 0 of block 0 zeroes slot `z` for the
    /// call after next.  An all-zero history gives factor 1.0.  Elements are
    /// quantised with saturation, so a value above the history max clamps to
    /// the format's max finite.  The first two threads write the scale pair
    /// `[factor, 1 / factor]` exactly like [`fp8_quantize_kernel`].  Launch
    /// whole 256-thread blocks covering `max(x.len(), 2)` threads.
    #[kernel]
    pub fn fp8_quantize_delayed_kernel(
        x: &[f32],
        mut hist: DisjointSlice<f32>,
        mut out: DisjointSlice<u8>,
        mut scale: DisjointSlice<f32>,
        fmt: u32,
        base: u32,
        w: u32,
        z: u32,
        len: u32,
    ) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let idx = thread::index_1d();
        let i = idx.get();
        let hp = hist.as_mut_ptr();
        let mut hmax = 0.0f32;
        let mut s = 0u32;
        while s < 16 {
            if s != w && s != z {
                // SAFETY: the host passes `base + 16 <= hist.len()`; slots other
                // than `w` / `z` are not written during this launch.
                let v = unsafe { hp.add((base + s) as usize).read() };
                if v > hmax {
                    hmax = v;
                }
            }
            s += 1;
        }
        let factor = fp8_factor(hmax, fmt);
        let mut bits = 0u32;
        if i < bounded_len(len, x.len()) {
            let value = x[i];
            if value.is_finite() {
                bits = value.abs().to_bits();
            }
            if let Some(o) = out.get_mut(idx) {
                *o = quantize_f32(value, factor, fmt);
            }
        }
        if let Some(o) = scale.get_mut(thread::index_1d()) {
            *o = if i == 0 { factor } else { 1.0 / factor };
        }
        // SAFETY: `tid < 256`, the shared array's length; every access between
        // two barriers touches a distinct or read-only slot.
        unsafe { sh.add(tid).write(bits) };
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                let other = unsafe { sh.add(tid + stride).read() };
                let mine = unsafe { sh.add(tid).read() };
                if other > mine {
                    unsafe { sh.add(tid).write(other) };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: slot `w` is only touched through this atomic during the
            // launch and holds 0 on entry (zeroed two calls ago).
            let slot = unsafe { hp.add((base + w) as usize) } as *mut u32;
            let atomic = unsafe { DeviceAtomicU32::from_ptr(slot) };
            let _ = atomic.fetch_max(unsafe { sh.read() }, AtomicOrdering::Relaxed);
            if thread::blockIdx_x() == 0 {
                // SAFETY: slot `z` is neither read nor written by any other
                // thread of this launch.
                unsafe { hp.add((base + z) as usize).write(0.0) };
            }
        }
    }

    /// [`fp8_quantize_delayed_kernel`] reading 16-bit storage (`hfmt`: `0` = f16, `1` = bf16, widened exactly by
    /// `half_bits_to_f32`): the same history protocol, factor, scale pair and bytes as the f32 kernel gives on the
    /// widened tensor.  Delayed-scaling quantiser: one pass over `x`.  `hist[base..base + 16]`
    /// is the key's amax history.  Call `n` of a key has `w = n % 16` and
    /// `z = (n + 2) % 16` (host-computed): the factor is `max_finite(fmt) /
    /// max(hist[s])` over every slot except `w` and `z`, so nothing read here
    /// is written by this launch; the block-reduced amax of `x` (non-finite
    /// ignored) goes into slot `w` with one `atomicMax` per block (the slot was
    /// zeroed two calls ago), and thread 0 of block 0 zeroes slot `z` for the
    /// call after next.  An all-zero history gives factor 1.0.  Elements are
    /// quantised with saturation, so a value above the history max clamps to
    /// the format's max finite.  The first two threads write the scale pair
    /// `[factor, 1 / factor]` exactly like [`fp8_quantize_kernel`].  Launch
    /// whole 256-thread blocks covering `max(x.len(), 2)` threads.
    #[kernel]
    pub fn fp8_quantize_delayed_half_kernel(
        x: &[u16],
        mut hist: DisjointSlice<f32>,
        mut out: DisjointSlice<u8>,
        mut scale: DisjointSlice<f32>,
        fmt: u32,
        base: u32,
        w: u32,
        z: u32,
        hfmt: u32,
        len: u32,
    ) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let idx = thread::index_1d();
        let i = idx.get();
        let hp = hist.as_mut_ptr();
        let mut hmax = 0.0f32;
        let mut s = 0u32;
        while s < 16 {
            if s != w && s != z {
                // SAFETY: the host passes `base + 16 <= hist.len()`; slots other
                // than `w` / `z` are not written during this launch.
                let v = unsafe { hp.add((base + s) as usize).read() };
                if v > hmax {
                    hmax = v;
                }
            }
            s += 1;
        }
        let factor = fp8_factor(hmax, fmt);
        let mut bits = 0u32;
        if i < bounded_len(len, x.len()) {
            let value = half_bits_to_f32(x[i], hfmt);
            if value.is_finite() {
                bits = value.abs().to_bits();
            }
            if let Some(o) = out.get_mut(idx) {
                *o = quantize_f32(value, factor, fmt);
            }
        }
        if let Some(o) = scale.get_mut(thread::index_1d()) {
            *o = if i == 0 { factor } else { 1.0 / factor };
        }
        // SAFETY: `tid < 256`, the shared array's length; every access between
        // two barriers touches a distinct or read-only slot.
        unsafe { sh.add(tid).write(bits) };
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                let other = unsafe { sh.add(tid + stride).read() };
                let mine = unsafe { sh.add(tid).read() };
                if other > mine {
                    unsafe { sh.add(tid).write(other) };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            // SAFETY: slot `w` is only touched through this atomic during the
            // launch and holds 0 on entry (zeroed two calls ago).
            let slot = unsafe { hp.add((base + w) as usize) } as *mut u32;
            let atomic = unsafe { DeviceAtomicU32::from_ptr(slot) };
            let _ = atomic.fetch_max(unsafe { sh.read() }, AtomicOrdering::Relaxed);
            if thread::blockIdx_x() == 0 {
                // SAFETY: slot `z` is neither read nor written by any other
                // thread of this launch.
                unsafe { hp.add((base + z) as usize).write(0.0) };
            }
        }
    }

    /// `slots[0] = max|a[..a_len]|`, `slots[32] = max|b[..b_len]|` (non-finite
    /// ignored) in one launch of `ceil(a_len / 256) + ceil(b_len / 256)`
    /// blocks of 256 threads; whole blocks belong to one operand.  Each block
    /// reduces its tile in shared memory and issues one atomic.  `slots` is 33
    /// f32 words, zero on entry; the atomics act on the bit patterns, which
    /// order like the values for non-negative floats.
    #[kernel]
    pub fn fp8_amax_pair_kernel(
        a: &[f32],
        b: &[f32],
        a_len: u32,
        b_len: u32,
        mut slots: DisjointSlice<f32>,
        off: u32,
    ) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let a_blocks = (a_len as usize).div_ceil(256);
        let (is_b, i, len) = if block < a_blocks {
            (false, block * 256 + tid, a_len as usize)
        } else {
            (true, (block - a_blocks) * 256 + tid, b_len as usize)
        };
        let mut bits = 0u32;
        if i < len {
            let value = if is_b { b[i] } else { a[i] };
            if value.is_finite() {
                bits = value.abs().to_bits();
            }
        }
        // SAFETY: `tid < 256`, the shared array's length; every access between
        // two barriers touches a distinct or read-only slot.
        unsafe { sh.add(tid).write(bits) };
        thread::sync_threads();
        let mut stride = 128usize;
        while stride > 0 {
            if tid < stride {
                let other = unsafe { sh.add(tid + stride).read() };
                let mine = unsafe { sh.add(tid).read() };
                if other > mine {
                    unsafe { sh.add(tid).write(other) };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let word = off as usize + if is_b { 32 } else { 0 };
            // SAFETY: the host passes 33 words; each slot is only touched
            // through this atomic while the kernel runs.
            let slot = unsafe { slots.as_mut_ptr().add(word) } as *mut u32;
            let atomic = unsafe { DeviceAtomicU32::from_ptr(slot) };
            let _ = atomic.fetch_max(unsafe { sh.read() }, AtomicOrdering::Relaxed);
        }
    }

    /// Quantise `a[..a_len]` and `b[..b_len]` in one launch.  Threads
    /// `[0, max(a_len, 2))` handle `a`, the following `max(b_len, 2)` threads
    /// handle `b`; the first two threads of each side also write that
    /// operand's scale pair `[factor, 1 / factor]`.  Factors come from
    /// `amax[0]` / `amax[32]`.
    #[kernel]
    pub fn fp8_quantize_pair_kernel(
        a: &[f32],
        b: &[f32],
        amax: &[f32],
        mut out_a: DisjointSlice<u8>,
        mut out_b: DisjointSlice<u8>,
        mut scale_a: DisjointSlice<f32>,
        mut scale_b: DisjointSlice<f32>,
        a_len: u32,
        b_len: u32,
        fmt_a: u32,
        fmt_b: u32,
        off: u32,
    ) {
        let i = thread::index_1d().get();
        let off = off as usize;
        let (a_len, b_len) = (a_len as usize, b_len as usize);
        let a_span = if a_len > 2 { a_len } else { 2 };
        // SAFETY (all raw writes below): the host sizes `out_*` to the operand
        // lengths and `scale_*` to two words; each thread writes only the slots
        // its own index selects, so no two threads write the same address.
        if i < a_span {
            let factor = fp8_factor(amax[off], fmt_a);
            if i < 2 {
                unsafe { scale_a.as_mut_ptr().add(i).write(if i == 0 { factor } else { 1.0 / factor }) };
            }
            if i < a_len {
                unsafe { out_a.as_mut_ptr().add(i).write(quantize_f32(a[i], factor, fmt_a)) };
            }
        } else {
            let j = i - a_span;
            let factor = fp8_factor(amax[off + 32], fmt_b);
            if j < 2 {
                unsafe { scale_b.as_mut_ptr().add(j).write(if j == 0 { factor } else { 1.0 / factor }) };
            }
            if j < b_len {
                unsafe { out_b.as_mut_ptr().add(j).write(quantize_f32(b[j], factor, fmt_b)) };
            }
        }
    }

    /// `out[i] = quantize(x[i] * scale[0], fmt)` with a caller-supplied factor
    /// (the host-side `quantize_f32_to_fp8` path).
    #[kernel]
    pub fn fp8_quantize_scaled_kernel(x: &[f32], mut out: DisjointSlice<u8>, scale: &[f32], fmt: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = quantize_f32(x[i], scale[0], fmt);
        }
    }

    /// `out[i] = f32(quantized x[i])` (the scale is *not* reapplied).
    #[kernel]
    pub fn fp8_dequantize_kernel(x: &[u8], mut out: DisjointSlice<f32>, fmt: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= x.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = if fmt == 0 {
                e4m3_bits_to_f32(x[i])
            } else {
                e5m2_bits_to_f32(x[i])
            };
        }
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with 8-bit operands in `a` /
    /// `b` and an f32 accumulator in `c`.  Same view/split-K conventions as
    /// [`super::gemm_mma::kernels::sgemm_tf32_mma`].
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_fp8_mma(
        a: &[u8],
        b: &[u8],
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
        fmt_a: u32,
        fmt_b: u32,
        scale_a: &[f32],
        scale_b: &[f32],
    ) {
        // Two stages per operand; A is 16-byte aligned for `cp.async`.
        static mut AS: SharedArray<u32, { 2 * F8_BM * LD_A / 4 }, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, { 2 * F8_BN * LD_B / 4 }, 16> = SharedArray::UNINIT;
        let asw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };
        let asp = asw as *mut u8;
        let bsp = bsw as *mut u8;

        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let wm = (warp / 2) * 32;
        let wn = (warp % 2) * 32;

        let tiles_n = n.div_ceil(F8_BN);
        let tiles_m = m.div_ceil(F8_BM);
        let block = thread::blockIdx_x() as usize;
        let per_instance = tiles_m * tiles_n;
        let inst_all = block / per_instance;
        let inst = inst_all % count;
        let kbeg = (inst_all / count) * kchunk;
        let kend = if kbeg + kchunk < k { kbeg + kchunk } else { k };
        let tile = block % per_instance;
        let row0 = (tile / tiles_n) * F8_BM;
        let col0 = (tile % tiles_n) * F8_BN;

        // `trans_*` describe the logical GEMM operand ("NN" means B is
        // stored [k][n]); the staging helper wants "contiguous along K", which
        // is the complement for B and the same for A.
        let stage_trans_a = trans_a;
        let stage_trans_b = 1 - trans_b;
        let a_off = a_base + (inst % a_mod) * a_stride;
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst_all % c_mod) * c_stride;

        // This thread's 16-byte chunk of each operand tile: `F8_THREADS`
        // threads cover `F8_BM` rows x `CHUNKS_PER_ROW` chunks exactly.
        let stage_row = tid / CHUNKS_PER_ROW;
        let stage_kb = (tid % CHUNKS_PER_ROW) * 16;

        // acc[(mi * 4 + ni) * 4 + j]
        let mut acc = [0.0f32; 32];

        let vec_a = (stage_trans_a == 0 && lda % 16 == 0 && (a.as_ptr() as usize + a_off) % 16 == 0)
            as u32;
        let nk = (kend - kbeg).div_ceil(F8_BK);
        let mut breg = [0u32; 4];
        // A goes global -> shared (cp.async when aligned), B through registers.
        macro_rules! issue_a {
            ($tile:expr) => {{
                let tile_i: usize = $tile;
                // SAFETY: row < 64 and kb + 16 <= 32 <= LD_A inside the stage;
                // 16-byte aligned (LD_A and the stage size are multiples of 16).
                unsafe {
                    fp8_issue16(
                        asw.add((tile_i & 1) * (F8_BM * LD_A / 4) + (stage_row * LD_A + stage_kb) / 4),
                        a,
                        a_off,
                        lda,
                        kend,
                        m,
                        row0,
                        stage_row,
                        kbeg + tile_i * F8_BK,
                        stage_kb,
                        stage_trans_a,
                        vec_a,
                    );
                    cuda_device::async_copy::cp_async_commit_group();
                }
            }};
        }
        macro_rules! load_b {
            ($tile:expr) => {{
                let k0 = kbeg + ($tile) * F8_BK;
                if stage_trans_b == 1 {
                    fp8_stage_t4(&mut breg, b, b_off, ldb, kend, n, col0, tid % 16, k0, tid / 16);
                } else {
                    fp8_stage16(
                        &mut breg,
                        b,
                        b_off,
                        ldb,
                        kend,
                        n,
                        col0,
                        stage_row,
                        k0,
                        stage_kb,
                        stage_trans_b,
                    );
                }
            }};
        }
        macro_rules! store_b {
            ($tile:expr) => {{
                let base = (($tile) & 1) * (F8_BN * LD_B / 4);
                unroll!(4, w, {
                    // SAFETY: the offsets stay inside this stage's B tile and
                    // are word aligned (`LD_B` and `stage_kb` are multiples of 4).
                    unsafe {
                        // `trans == 1` blocks hold column `4 ng + w`, K bytes
                        // `4 kg ..`; the others a 16-byte K run of one row.
                        let boff = if stage_trans_b == 1 {
                            ((tid % 16) * 4 + w) * LD_B / 4 + tid / 16
                        } else {
                            (stage_row * LD_B + stage_kb) / 4 + w
                        };
                        bsw.add(base + boff).write(breg[w]);
                    }
                });
            }};
        }
        if nk > 0 {
            issue_a!(0);
            load_b!(0);
            store_b!(0);
        }

        let mut it = 0usize;
        while it < nk {
            // Tile `it` of A is the only group in flight here.
            unsafe { cuda_device::async_copy::cp_async_wait_group(0) };
            // Publishes tile `it` (A and B); also proves every warp is done
            // reading the stage that tile `it + 1` is about to overwrite.
            thread::sync_threads();
            let has_next = it + 1 < nk;
            if has_next {
                issue_a!(it + 1);
                load_b!(it + 1);
            }
            let asp = unsafe { asp.add((it & 1) * (F8_BM * LD_A)) };
            let bsp = unsafe { bsp.add((it & 1) * (F8_BN * LD_B)) };

            // One mma k step per 32-element tile (F8_BK == 32).
            let mut bf = [[0u32; 2]; 4];
            unroll!(4, ni, {
                let col = wn + ni * 8 + g;
                unroll!(2, r, {
                    let kk = t * 4 + r * 16;
                    // SAFETY: `col < F8_BN` and `kk + 3 < F8_BK`, so the four
                    // bytes are inside the staged B tile; `LD_B` is a multiple
                    // of 4 and `kk` is too, so the word access is aligned.
                    unsafe {
                        bf[ni][r] = (bsp.add(col * LD_B + kk) as *const u32).read();
                    }
                });
            });
            unroll!(2, mi, {
                let row = wm + mi * 16 + g;
                let mut af = [0u32; 4];
                unroll!(2, r, {
                    let kk = t * 4 + r * 16;
                    // SAFETY: as above for the A tile.
                    unsafe {
                        af[r * 2] = (asp.add(row * LD_A + kk) as *const u32).read();
                        af[r * 2 + 1] = (asp.add((row + 8) * LD_A + kk) as *const u32).read();
                    }
                });
                unroll!(4, ni, {
                    let o = (mi * 4 + ni) * 4;
                    let cin = [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]];
                    let d = mma_f8_k32(cin, af, bf[ni], fmt_a, fmt_b);
                    acc[o] = d[0];
                    acc[o + 1] = d[1];
                    acc[o + 2] = d[2];
                    acc[o + 3] = d[3];
                });
            });
            if has_next {
                store_b!(it + 1);
            }
            it += 1;
        }

        unroll!(2, mi, {
            unroll!(4, ni, {
                unroll!(4, j, {
                    let gm = row0 + wm + mi * 16 + g + if j >= 2 { 8 } else { 0 };
                    let gn = col0 + wn + ni * 8 + 2 * t + (j & 1);
                    if gm < m && gn < n {
                        let idx = c_off + gm * ldc + gn;
                        // SAFETY: each (instance, row, column) has exactly one
                        // owner thread; the extents are checked by the host.
                        unsafe {
                            let dst = c.as_mut_ptr().add(idx);
                            let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                            // `scale_*[1]` is the reciprocal dequantisation factor:
                            // the mma works in the quantised domain.
                            let alpha = alpha * scale_a[1] * scale_b[1];
                            *dst = alpha * acc[(mi * 4 + ni) * 4 + j] + prev;
                        }
                    }
                });
            });
        });
    }

    /// [`sgemm_fp8_mma`] for e4m3 x e4m3 operands: a verbatim copy of its body
    /// (keep the two in sync; `fp8_e4m3_specialised_kernel_matches_generic` pins
    /// them bitwise) whose tensor-core step has no runtime format `match`.
    /// `fmt_a` / `fmt_b` are accepted for signature parity and ignored.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_fp8_mma_e4m3(
        a: &[u8],
        b: &[u8],
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
        _fmt_a: u32,
        _fmt_b: u32,
        scale_a: &[f32],
        scale_b: &[f32],
    ) {
        // Two stages per operand; A is 16-byte aligned for `cp.async`.
        static mut AS: SharedArray<u32, { 2 * F8_BM * LD_A / 4 }, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, { 2 * F8_BN * LD_B / 4 }, 16> = SharedArray::UNINIT;
        let asw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsw = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };
        let asp = asw as *mut u8;
        let bsp = bsw as *mut u8;

        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let g = lane / 4;
        let t = lane % 4;
        let wm = (warp / 2) * 32;
        let wn = (warp % 2) * 32;

        let tiles_n = n.div_ceil(F8_BN);
        let tiles_m = m.div_ceil(F8_BM);
        let block = thread::blockIdx_x() as usize;
        let per_instance = tiles_m * tiles_n;
        let inst_all = block / per_instance;
        let inst = inst_all % count;
        let kbeg = (inst_all / count) * kchunk;
        let kend = if kbeg + kchunk < k { kbeg + kchunk } else { k };
        let tile = block % per_instance;
        let row0 = (tile / tiles_n) * F8_BM;
        let col0 = (tile % tiles_n) * F8_BN;

        // `trans_*` describe the logical GEMM operand ("NN" means B is
        // stored [k][n]); the staging helper wants "contiguous along K", which
        // is the complement for B and the same for A.
        let stage_trans_a = trans_a;
        let stage_trans_b = 1 - trans_b;
        let a_off = a_base + (inst % a_mod) * a_stride;
        let b_off = b_base + (inst % b_mod) * b_stride;
        let c_off = c_base + (inst_all % c_mod) * c_stride;

        // This thread's 16-byte chunk of each operand tile: `F8_THREADS`
        // threads cover `F8_BM` rows x `CHUNKS_PER_ROW` chunks exactly.
        let stage_row = tid / CHUNKS_PER_ROW;
        let stage_kb = (tid % CHUNKS_PER_ROW) * 16;

        // acc[(mi * 4 + ni) * 4 + j]
        let mut acc = [0.0f32; 32];

        let vec_a = (stage_trans_a == 0 && lda % 16 == 0 && (a.as_ptr() as usize + a_off) % 16 == 0)
            as u32;
        let nk = (kend - kbeg).div_ceil(F8_BK);
        let mut breg = [0u32; 4];
        // A goes global -> shared (cp.async when aligned), B through registers.
        macro_rules! issue_a {
            ($tile:expr) => {{
                let tile_i: usize = $tile;
                // SAFETY: row < 64 and kb + 16 <= 32 <= LD_A inside the stage;
                // 16-byte aligned (LD_A and the stage size are multiples of 16).
                unsafe {
                    fp8_issue16(
                        asw.add((tile_i & 1) * (F8_BM * LD_A / 4) + (stage_row * LD_A + stage_kb) / 4),
                        a,
                        a_off,
                        lda,
                        kend,
                        m,
                        row0,
                        stage_row,
                        kbeg + tile_i * F8_BK,
                        stage_kb,
                        stage_trans_a,
                        vec_a,
                    );
                    cuda_device::async_copy::cp_async_commit_group();
                }
            }};
        }
        macro_rules! load_b {
            ($tile:expr) => {{
                let k0 = kbeg + ($tile) * F8_BK;
                if stage_trans_b == 1 {
                    fp8_stage_t4(&mut breg, b, b_off, ldb, kend, n, col0, tid % 16, k0, tid / 16);
                } else {
                    fp8_stage16(
                        &mut breg,
                        b,
                        b_off,
                        ldb,
                        kend,
                        n,
                        col0,
                        stage_row,
                        k0,
                        stage_kb,
                        stage_trans_b,
                    );
                }
            }};
        }
        macro_rules! store_b {
            ($tile:expr) => {{
                let base = (($tile) & 1) * (F8_BN * LD_B / 4);
                unroll!(4, w, {
                    // SAFETY: the offsets stay inside this stage's B tile and
                    // are word aligned (`LD_B` and `stage_kb` are multiples of 4).
                    unsafe {
                        // `trans == 1` blocks hold column `4 ng + w`, K bytes
                        // `4 kg ..`; the others a 16-byte K run of one row.
                        let boff = if stage_trans_b == 1 {
                            ((tid % 16) * 4 + w) * LD_B / 4 + tid / 16
                        } else {
                            (stage_row * LD_B + stage_kb) / 4 + w
                        };
                        bsw.add(base + boff).write(breg[w]);
                    }
                });
            }};
        }
        if nk > 0 {
            issue_a!(0);
            load_b!(0);
            store_b!(0);
        }

        let mut it = 0usize;
        while it < nk {
            // Tile `it` of A is the only group in flight here.
            unsafe { cuda_device::async_copy::cp_async_wait_group(0) };
            // Publishes tile `it` (A and B); also proves every warp is done
            // reading the stage that tile `it + 1` is about to overwrite.
            thread::sync_threads();
            let has_next = it + 1 < nk;
            if has_next {
                issue_a!(it + 1);
                load_b!(it + 1);
            }
            let asp = unsafe { asp.add((it & 1) * (F8_BM * LD_A)) };
            let bsp = unsafe { bsp.add((it & 1) * (F8_BN * LD_B)) };

            // One mma k step per 32-element tile (F8_BK == 32).
            let mut bf = [[0u32; 2]; 4];
            unroll!(4, ni, {
                let col = wn + ni * 8 + g;
                unroll!(2, r, {
                    let kk = t * 4 + r * 16;
                    // SAFETY: `col < F8_BN` and `kk + 3 < F8_BK`, so the four
                    // bytes are inside the staged B tile; `LD_B` is a multiple
                    // of 4 and `kk` is too, so the word access is aligned.
                    unsafe {
                        bf[ni][r] = (bsp.add(col * LD_B + kk) as *const u32).read();
                    }
                });
            });
            unroll!(2, mi, {
                let row = wm + mi * 16 + g;
                let mut af = [0u32; 4];
                unroll!(2, r, {
                    let kk = t * 4 + r * 16;
                    // SAFETY: as above for the A tile.
                    unsafe {
                        af[r * 2] = (asp.add(row * LD_A + kk) as *const u32).read();
                        af[r * 2 + 1] = (asp.add((row + 8) * LD_A + kk) as *const u32).read();
                    }
                });
                unroll!(4, ni, {
                    let o = (mi * 4 + ni) * 4;
                    let cin = [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]];
                    let d = mma_f8_k32_e4m3(cin, af, bf[ni]);
                    acc[o] = d[0];
                    acc[o + 1] = d[1];
                    acc[o + 2] = d[2];
                    acc[o + 3] = d[3];
                });
            });
            if has_next {
                store_b!(it + 1);
            }
            it += 1;
        }

        unroll!(2, mi, {
            unroll!(4, ni, {
                unroll!(4, j, {
                    let gm = row0 + wm + mi * 16 + g + if j >= 2 { 8 } else { 0 };
                    let gn = col0 + wn + ni * 8 + 2 * t + (j & 1);
                    if gm < m && gn < n {
                        let idx = c_off + gm * ldc + gn;
                        // SAFETY: each (instance, row, column) has exactly one
                        // owner thread; the extents are checked by the host.
                        unsafe {
                            let dst = c.as_mut_ptr().add(idx);
                            let prev = if beta == 0.0 { 0.0 } else { beta * *dst };
                            // `scale_*[1]` is the reciprocal dequantisation factor:
                            // the mma works in the quantised domain.
                            let alpha = alpha * scale_a[1] * scale_b[1];
                            *dst = alpha * acc[(mi * 4 + ni) * 4 + j] + prev;
                        }
                    }
                });
            });
        });
    }
}
