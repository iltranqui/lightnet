//! f32 <-> 16-bit float conversions (P6 inference storage).
//!
//! 16-bit values travel as raw `u16` bit patterns.  f32 -> f16 / bf16 use the
//! hardware `cvt.rn` instructions (round to nearest, ties to even); NaN stays
//! NaN, overflow becomes +-Inf unless `sat != 0`, which clamps every |x| above
//! 65504 (f16 max, also +-Inf) to +-65504.  f16 -> f32 / bf16 -> f32 are exact.
//! Subnormal f16 results are produced (no flush to zero).

use cuda_device::convert::{cvt_bf16x2_f32, cvt_f16x2_f32, cvt_f32_f16x2_lo};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

pub const F16_MAX: f32 = 65504.0;

/// f32 -> f16 bits, round to nearest even; `sat != 0` clamps to +-65504.
#[device]
pub fn f32_to_f16_bits(x: f32, sat: u32) -> u16 {
    let mut v = x;
    if sat != 0 {
        if v > F16_MAX {
            v = F16_MAX;
        } else if v < -F16_MAX {
            v = -F16_MAX;
        }
    }
    (cvt_f16x2_f32(v, 0.0) & 0xFFFF) as u16
}

/// f16 bits -> f32 (exact).
#[device]
pub fn f16_bits_to_f32(h: u16) -> f32 {
    cvt_f32_f16x2_lo(h as u32)
}

/// f32 -> bf16 bits, round to nearest even; NaN stays NaN, overflow stays
/// finite up to the f32 range (bf16 has the f32 exponent).
#[device]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    (cvt_bf16x2_f32(x, 0.0) & 0xFFFF) as u16
}

/// bf16 bits -> f32 (exact).
#[device]
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// 16-bit bits -> f32 for `fmt` (`0` = f16, `1` = bf16), exact.
#[device]
pub fn half_bits_to_f32(bits: u16, fmt: u32) -> f32 {
    if fmt == 0 {
        f16_bits_to_f32(bits)
    } else {
        bf16_bits_to_f32(bits)
    }
}

/// f32 -> 16-bit bits for `fmt` (`0` = f16, `1` = bf16), round to nearest even.
/// f16 overflow becomes +-Inf (no saturation), so a scaled gradient that
/// overflows stays visible to the loss scaler.
#[device]
pub fn f32_to_half_bits(x: f32, fmt: u32) -> u16 {
    if fmt == 0 {
        f32_to_f16_bits(x, 0)
    } else {
        f32_to_bf16_bits(x)
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn cvt_f32_to_f16(x: &[f32], mut out: DisjointSlice<u16>, sat: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_f16_bits(x[i], sat);
        }
    }

    #[kernel]
    pub fn cvt_f16_to_f32(x: &[u16], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f16_bits_to_f32(x[i]);
        }
    }

    /// `out[r * ld + c] = f16(x[r * cols + c])` for `c < cols`, zero for the
    /// padding columns `cols..ld` (GEMM operands whose rows are padded to a
    /// multiple of 8 elements).
    #[kernel]
    pub fn cvt_f32_to_f16_rows(
        x: &[f32],
        mut out: DisjointSlice<u16>,
        cols: usize,
        ld: usize,
        sat: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let r = i / ld;
            let c = i % ld;
            *o = if c < cols {
                f32_to_f16_bits(x[r * cols + c], sat)
            } else {
                0
            };
        }
    }

    /// `out[i] = f32_to_half_bits(x[i], fmt)`; `fmt` 0 = f16, 1 = bf16.
    #[kernel]
    pub fn cvt_f32_to_half(x: &[f32], mut out: DisjointSlice<u16>, fmt: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_half_bits(x[i], fmt);
        }
    }

    /// `out[i] = half_bits_to_f32(x[i], fmt)`; `fmt` 0 = f16, 1 = bf16.
    #[kernel]
    pub fn cvt_half_to_f32(x: &[u16], mut out: DisjointSlice<f32>, fmt: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = half_bits_to_f32(x[i], fmt);
        }
    }

    #[kernel]
    pub fn cvt_f32_to_bf16(x: &[f32], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = (cvt_bf16x2_f32(x[i], 0.0) & 0xFFFF) as u16;
        }
    }

    #[kernel]
    pub fn cvt_bf16_to_f32(x: &[u16], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f32::from_bits((x[i] as u32) << 16);
        }
    }
}
