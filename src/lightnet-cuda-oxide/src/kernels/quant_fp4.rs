//! NVFP4 (E2M1 codes + E4M3 block scales) quantiser, ported from the C++
//! reference `raptor-fp4:src-fp4sim/fp4_sim_kernels.cu`.
//!
//! Recipe (the `TwoLevel` scale mode of the C++ file, which is the NVFP4 one):
//!
//! ```text
//! global_scale = amax / (6 * 448)              (1 when amax is 0 or non-finite)
//! block_scale  = E4M3(min(448, block_amax / (global_scale * 6)))   (1 for a zero block)
//! q            = E2M1(x / (global_scale * decode(block_scale)))
//! ```
//!
//! The E2M1 grid is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign bit; a code
//! carries `sign << 3 | magnitude_index`.  The stored layout is tightly
//! packed: `codes[row, block * 8 + b]` holds the two E2M1 codes of elements
//! `block * 16 + 2b` (low nibble) and `block * 16 + 2b + 1` (high nibble), and
//! `scales[row * blocks_per_row + block]` is the *raw* E4M3 code of that
//! block's scale.  `blocks_per_row = ceil(K / 16)`; a ragged tail is zero
//! coded, i.e. `+0`.
//!
//! Rounding is round-to-nearest with ties to the even code (matching
//! `fp4sim_encode_nearest`), non-finite inputs are ignored by amax, NaN encodes
//! as `+6` and +-Inf saturates.  `tasks/rust/03_plans/lowprec/fp8_fp4.md` Phase C step 7.

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;
/// Largest finite e4m3 magnitude.
pub const E4M3_MAX: f32 = 448.0;
/// `E2M1_MAX * E4M3_MAX`: the global scale divisor of the NVFP4 recipe.
pub const GLOBAL_DIVISOR: f32 = 2688.0;
/// Elements per E4M3 scale block along K.
pub const BLOCK: usize = 16;
/// Threads per quantise launch.
pub const QUANT_THREADS: u32 = 128;

/// `decode_e2m1(code) * 2`, i.e. the E2M1 grid doubled to integers.  Bit 3 is
/// the sign, bits 2:0 index the grid.  Also the `__dp4a` operand encoding:
/// every lane holds values in `[-12, 12]`, so the product of two lanes plus
/// the rest of the word never overflows i32.
#[device]
pub fn e2m1x2_lut(code: u32) -> i32 {
    // {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
    let magnitude = code & 7;
    let value = if magnitude == 0 {
        0
    } else if magnitude == 1 {
        1
    } else if magnitude == 2 {
        2
    } else if magnitude == 3 {
        3
    } else if magnitude == 4 {
        4
    } else if magnitude == 5 {
        6
    } else if magnitude == 6 {
        8
    } else {
        12
    };
    if code & 8 != 0 { -value } else { value }
}

/// Signed e4m3 decode; bit 7 is the sign.
#[device]
pub fn decode_e4m3(code: u32) -> f32 {
    let value = decode_e4m3_magnitude(code & 0x7F);
    if code & 0x80 != 0 { -value } else { value }
}

/// Magnitude of an e4m3 code (`fp4sim_decode_e4m3` on the C++ side).
#[device]
pub fn decode_e4m3_magnitude(code: u32) -> f32 {
    let exponent = code >> 3;
    let mantissa = code & 7;
    if exponent == 0 {
        (mantissa as f32) * exp2i(-9)
    } else {
        (1.0 + (mantissa as f32) / 8.0) * exp2i(exponent as i32 - 7)
    }
}

/// Exact `2^e`; the e4m3 grid only needs `e` in `-9..=8`.
#[device]
fn exp2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Nearest e4m3 code over `0..=0x7e` for a non-negative finite magnitude,
/// ties to the even code (`fp4sim_encode_e4m3`).
#[device]
pub fn encode_e4m3(magnitude: f32) -> u8 {
    let mut best = 0u32;
    let mut best_distance = f32::MAX;
    let mut code = 0u32;
    while code <= 0x7E {
        let distance = (magnitude - decode_e4m3_magnitude(code)).abs();
        if distance < best_distance || (distance == best_distance && (code & 1) == 0) {
            best = code;
            best_distance = distance;
        }
        code += 1;
    }
    best as u8
}

/// `decode_e2m1(code)` (the halved grid value, not the doubled LUT value).
#[device]
pub fn decode_e2m1(code: u32) -> f32 {
    (e2m1x2_lut(code) as f32) * 0.5
}

/// Nearest E2M1 code for `value`: sign bit plus the grid index of the nearest
/// magnitude, ties to the even index (`fp4sim_encode_nearest`).
#[device]
pub fn encode_e2m1(value: f32) -> u8 {
    if value.is_nan() {
        return 0x07;
    }
    let sign: u8 = if value.to_bits() >> 31 != 0 {
        0x08
    } else {
        0x00
    };
    let magnitude = if value.is_infinite() {
        E2M1_MAX
    } else {
        let m = value.abs();
        if m > E2M1_MAX { E2M1_MAX } else { m }
    };
    let mut best = 0u32;
    let mut best_distance = f32::MAX;
    let mut idx = 0u32;
    while idx < 8 {
        let grid = decode_e2m1(idx);
        let distance = (magnitude - grid).abs();
        if distance < best_distance || (distance == best_distance && (idx & 1) == 0) {
            best = idx;
            best_distance = distance;
        }
        idx += 1;
    }
    sign | best as u8
}

/// `amax / (6 * 448)`, or 1 for a zero / non-finite amax
/// (`fp4sim_choose_global_scale`).
#[device]
pub fn choose_global_scale(amax: f32) -> f32 {
    if amax > 0.0 && amax.is_finite() {
        amax / GLOBAL_DIVISOR
    } else {
        1.0
    }
}

/// Decoded E4M3 block scale for a block amax, `TwoLevel` mode
/// (`fp4sim_local_scale`).
#[device]
pub fn local_scale(block_amax: f32, global_scale: f32) -> f32 {
    if !(block_amax > 0.0) || !(global_scale > 0.0) {
        return 1.0;
    }
    let raw = block_amax / (global_scale * E2M1_MAX);
    let clamped = if raw > E4M3_MAX { E4M3_MAX } else { raw };
    decode_e4m3_magnitude(encode_e4m3(clamped) as u32)
}

/// Fills `codes[0..8]` (two codes per byte, low nibble first) for the 16
/// elements of one block, and returns the raw E4M3 code of the block scale.
/// Elements past `K` are coded `+0`.
#[allow(clippy::too_many_arguments)]
#[device]
pub fn quantize_block(
    values: &[f32],
    k: usize,
    block_start: usize,
    global_scale: f32,
    codes: &mut [u8; 8],
    scale_code: &mut u8,
) {
    let mut amax = 0.0f32;
    let mut i = 0usize;
    while i < BLOCK {
        let kk = block_start + i;
        let v = if kk < k {
            let v = values[kk];
            if v.is_finite() { v.abs() } else { 0.0 }
        } else {
            0.0
        };
        if v > amax {
            amax = v;
        }
        i += 1;
    }
    let scale = local_scale(amax, global_scale);
    *scale_code = encode_e4m3(scale);
    let divisor = global_scale * scale;
    let mut b = 0usize;
    while b < 8 {
        let mut byte = 0u8;
        let mut half = 0usize;
        while half < 2 {
            let ii = b * 2 + half;
            let kk = block_start + ii;
            if kk < k {
                let code = encode_e2m1(values[kk] / divisor);
                byte |= code << (4 * half);
            }
            half += 1;
        }
        codes[b] = byte;
        b += 1;
    }
}

/// Host-free reference decode of a packed E2M1 nibble (`decode_e2m1`), used
/// by the tests through the device `decode_e2m1`.
pub fn decode_e2m1_host(code: u8) -> f32 {
    const GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = GRID[(code & 7) as usize];
    if code & 8 != 0 { -magnitude } else { magnitude }
}

/// Host-free reference decode of an e4m3 byte, matching [`decode_e4m3`].
pub fn decode_e4m3_host(code: u8) -> f32 {
    let exponent = (code >> 3) as i32;
    let mantissa = (code & 7) as f32;
    let value = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if code & 0x80 != 0 { -value } else { value }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[0] = max(|x[i]|)` over a row-major `[rows, K]` matrix, ignoring
    /// non-finite values.  `out` is a single zeroed `u32` slot.
    #[kernel]
    pub fn fp4_amax_f32_kernel(x: &[f32], rows: usize, k: usize, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let total = rows * k;
        if i >= total {
            return;
        }
        let value = x[i];
        let magnitude = if value.is_finite() { value.abs() } else { 0.0 };
        // SAFETY: the host guarantees `out.len() >= 1`; slot 0 is only
        // touched through this atomic while the kernel runs.
        let atomic = unsafe { DeviceAtomicU32::from_ptr(out.as_mut_ptr()) };
        let _ = atomic.fetch_max(magnitude.to_bits(), AtomicOrdering::Relaxed);
    }

    /// `scale[0] = choose_global_scale(amax[0])`.  One thread.
    #[kernel]
    pub fn fp4_finalize_global_scale_kernel(amax: &[f32], mut scale: DisjointSlice<f32>) {
        if thread::threadIdx_x() != 0 {
            return;
        }
        if amax.is_empty() {
            return;
        }
        let out = scale.as_mut_ptr();
        // SAFETY: the host guarantees `scale.len() >= 1`.
        unsafe { out.write(choose_global_scale(amax[0])) };
    }

    /// Packed NVFP4 quantiser: one thread per 16-element block along K.
    /// `global` is a single device f32 slot holding the global scale.
    /// `out_codes` must hold `rows * blocks_per_row * 8` bytes and
    /// `out_scales` `rows * blocks_per_row` bytes.
    #[kernel]
    pub fn fp4_quantize_1x16_kernel(
        x: &[f32],
        rows: usize,
        k: usize,
        global: &[f32],
        mut out_codes: DisjointSlice<u8>,
        mut out_scales: DisjointSlice<u8>,
    ) {
        let idx = thread::index_1d();
        let id = idx.get();
        let blocks_per_row = k.div_ceil(BLOCK);
        if id >= rows * blocks_per_row || global.is_empty() {
            return;
        }
        let row = id / blocks_per_row;
        let block = id % blocks_per_row;
        let global_scale = global[0];
        let base = row * k;
        let mut codes = [0u8; 8];
        let mut scale_code = 0u8;
        // SAFETY: `base + block * 16 + 15 < x.len()` is checked by the host
        // through the `rows * k` bound on `id`.
        quantize_block(
            &x[base..base + k],
            k,
            block * BLOCK,
            global_scale,
            &mut codes,
            &mut scale_code,
        );
        let code_ptr = out_codes.as_mut_ptr();
        let scale_ptr = out_scales.as_mut_ptr();
        let mut b = 0usize;
        while b < 8 {
            // SAFETY: the host sizes `out_codes` at `rows * blocks_per_row * 8`.
            unsafe {
                code_ptr
                    .add(row * blocks_per_row * 8 + block * 8 + b)
                    .write(codes[b])
            };
            b += 1;
        }
        // SAFETY: the host sizes `out_scales` at `rows * blocks_per_row`.
        unsafe { scale_ptr.add(id).write(scale_code) };
    }

    /// Dequantises a packed operand back to f32 (`decode(code) *
    /// decode(scale)`); `out` has the `[rows, K]` shape of the input.
    #[kernel]
    pub fn fp4_dequantize_1x16_kernel(
        codes: &[u8],
        scales: &[u8],
        rows: usize,
        k: usize,
        global: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= rows * k {
            return;
        }
        let blocks_per_row = k.div_ceil(BLOCK);
        let row = i / k;
        let kk = i % k;
        let block = kk / BLOCK;
        let code = if kk % 2 == 0 {
            codes[row * blocks_per_row * 8 + block * 8 + (kk % 16) / 2] & 0x0F
        } else {
            codes[row * blocks_per_row * 8 + block * 8 + (kk % 16) / 2] >> 4
        };
        let scale_code = scales[row * blocks_per_row + block] as u32;
        let value = decode_e2m1(code as u32) * decode_e4m3(scale_code) * global;
        if let Some(o) = out.get_mut(idx) {
            *o = value;
        }
    }
}
