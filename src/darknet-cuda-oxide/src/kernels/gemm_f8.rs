//! FP8 tensor-core GEMM for `sm_89+` (Ada), plus the f32 -> FP8 quantisation
//! helpers the backend needs.
//!
//! `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` (and the e5m2 /
//! mixed combinations) is the first FP8 shape Ada exposes; it keeps the
//! fragment shape of the existing int8 `mma_m16n8k32_s32_s8` kernel (A = 4
//! registers, B = 2, C = 4), so the fragment loads are byte-wide rather than
//! `ldmatrix`-based: `ldmatrix` only moves 16-bit elements.
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
//! (n8) mma tiles.  Shared memory holds both operand tiles with an 8-byte pad
//! on the K row stride (32 -> 40) so the four lanes of a group hit different
//! banks.  Operands are staged through registers: the next K tile is fetched
//! into per-thread registers while the current one feeds the tensor cores.
//! `NN` / `NT` / `TN` / `TT` are all supported, batched through the same
//! `(base, stride, modulus)` views as [`super::gemm_mma`], `beta == 0` never
//! reads `C`, and out-of-range loads are zero filled.
//!
//! Accuracy: operands are rounded to the format by the caller-supplied scale
//! (see [`crate::fp8`]); products and accumulation are f32.  e4m3 carries 3
//! mantissa bits (relative error <= 2^-4 per operand), e5m2 carries 2.

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
const LD_A: usize = F8_BK + 8;
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

/// Block size of `fp8_amax_pair_kernel` (one tile per block).
pub const F8_AMAX_BLOCK: usize = 256;
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

    /// `out[0] = max(|x[i]|)` over `x`, ignoring non-finite values.  `out` is
    /// a single f32 slot (read as `u32` bits) that must already hold 0.  Each
    /// 256-thread block reduces in shared memory and issues one `atomicMax` on the
    /// bit pattern (valid for non-negative floats); launch whole 256-thread blocks.
    #[kernel]
    pub fn fp8_amax_f32_kernel(x: &[f32], mut out: DisjointSlice<f32>, off: u32) {
        static mut SH: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SH) };
        let tid = thread::threadIdx_x() as usize;
        let i = thread::blockIdx_x() as usize * 256 + tid;
        let mut bits = 0u32;
        if i < x.len() {
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
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let amax = amax[off as usize];
        let max_finite = if fmt == 0 { 448.0 } else { 57344.0 };
        let factor = if amax > 0.0 && amax.is_finite() {
            max_finite / amax
        } else {
            1.0
        };
        if let Some(o) = scale.get_mut(thread::index_1d()) {
            *o = if i == 0 { factor } else { 1.0 / factor };
        }
        if i >= x.len() {
            return;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = quantize_f32(x[i], factor, fmt);
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
        static mut AS: SharedArray<u8, { F8_BM * LD_A }, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u8, { F8_BN * LD_B }, 16> = SharedArray::UNINIT;
        let asp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut AS) };
        let bsp = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BS) };

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

        let nk = (kend - kbeg).div_ceil(F8_BK);
        let mut areg = [0u32; 4];
        let mut breg = [0u32; 4];
        if nk > 0 {
            fp8_stage16(
                &mut areg,
                a,
                a_off,
                lda,
                kend,
                m,
                row0,
                stage_row,
                kbeg,
                stage_kb,
                stage_trans_a,
            );
            fp8_stage16(
                &mut breg,
                b,
                b_off,
                ldb,
                kend,
                n,
                col0,
                stage_row,
                kbeg,
                stage_kb,
                stage_trans_b,
            );
        }

        let mut it = 0usize;
        while it < nk {
            // Publish the staged tile.
            thread::sync_threads();
            unroll!(4, w, {
                // SAFETY: `stage_row < 64` and `stage_kb + 15` is inside both
                // strides, so each 4-byte store lands inside its tile; both
                // strides are multiples of 4 and the shared bases are
                // 16-byte aligned, so every store is aligned.
                unsafe {
                    (asp as *mut u32)
                        .add((stage_row * LD_A + stage_kb) / 4 + w)
                        .write(areg[w]);
                    (bsp as *mut u32)
                        .add((stage_row * LD_B + stage_kb) / 4 + w)
                        .write(breg[w]);
                }
            });
            thread::sync_threads();

            // Prefetch the next tile into registers.
            if it + 1 < nk {
                let k0 = kbeg + (it + 1) * F8_BK;
                fp8_stage16(
                    &mut areg,
                    a,
                    a_off,
                    lda,
                    kend,
                    m,
                    row0,
                    stage_row,
                    k0,
                    stage_kb,
                    stage_trans_a,
                );
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
