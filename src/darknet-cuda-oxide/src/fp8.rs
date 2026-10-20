//! Host-side reference for the two 8-bit float formats the GPU kernels use.
//!
//! Both formats are 1 sign bit + 4 exponent bits + 3 (e4m3) or 2 (e5m2)
//! mantissa bits, stored in one byte.  The decoders below are bit-exact
//! descriptions of the formats; the encoders reproduce
//! `cvt.rn.satfinite.e4m3x2.f32` / `cvt.rn.satfinite.e5m2x2.f32` (round to
//! nearest, ties to even, overflow clamped to the largest finite magnitude,
//! NaN kept as NaN) by a brute-force search over the code space.  The device
//! kernels in [`crate::kernels::gemm_f8`] use the hardware instructions
//! instead; `tests/oxide_gemm_f8.rs` pins the two against each other.
//!
//! Values the formats cannot represent:
//!
//! | format | max finite | smallest normal | smallest subnormal |
//! |---|---|---|---|
//! | e4m3 | 448 | 2^-6 | 2^-9 |
//! | e5m2 | 57344 | 2^-14 | 2^-16 |
//!
//! e4m3 has no infinities: code `0x7f`/`0xff` is NaN.  e5m2 follows IEEE 754
//! for the exponent: `0x7c` is the largest finite, `0x7d` is infinity and
//! `0x7f` is NaN.  e5m2 relative precision is 2^-3 (one mantissa bit worse
//! than e4m3) but its dynamic range is 128x wider, which is why the C++
//! reference (`v6-dev-fp8`) quantises gradients to e5m2.

/// 8-bit float format selected by the kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Format {
    /// 4-bit exponent, 3-bit mantissa, max 448 (no infinities).
    E4M3,
    /// 5-bit exponent, 2-bit mantissa, max 57344 (IEEE-like infinities).
    E5M2,
}

impl Fp8Format {
    /// Largest finite magnitude representable in the format.
    pub fn max_finite(self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }

    /// Mantissa bits (excluding the implicit leading 1).
    pub fn mantissa_bits(self) -> u32 {
        match self {
            Self::E4M3 => 3,
            Self::E5M2 => 2,
        }
    }

    /// Exponent bias.  Note that e4m3 uses 7, *not* the `2^(bits-1) - 1` that
    /// a plain IEEE-style field would imply: the OCP/NVIDIA e4m3 format trades
    /// exponent range for a wider subnormal grid, which is why its smallest
    /// subnormal is `2^-9` and not `2^-6`.
    pub fn exponent_bias(self) -> i32 {
        match self {
            Self::E4M3 => 7,
            Self::E5M2 => 15,
        }
    }

    /// Exponent bits (the byte layout is 1 sign + `exponent_bits` + mantissa).
    pub fn exponent_bits(self) -> u32 {
        7 - self.mantissa_bits()
    }

    /// Exponent field value reserved for infinity / NaN.
    pub fn reserved_exponent(self) -> u32 {
        (1u32 << self.exponent_bits()) - 1
    }

    /// Largest finite *code*: e4m3 spends the top binade on finite values (its
    /// maximum is `0x7e` = 448 and only `0x7f` is NaN, there is no infinity),
    /// while e5m2 keeps IEEE behaviour and reserves the whole top binade.
    pub fn max_finite_code(self) -> u8 {
        match self {
            Self::E4M3 => 0x7E,
            Self::E5M2 => 0x7B,
        }
    }

    /// Whether a magnitude byte encodes NaN (sign already stripped).
    pub fn is_nan_code(self, magnitude: u8) -> bool {
        let mb = self.mantissa_bits();
        let exponent = (magnitude >> mb) as u32;
        let mantissa = magnitude & ((1u8 << mb) - 1);
        match self {
            // Only the all-ones code is NaN; there is no infinity.
            Self::E4M3 => magnitude == 0x7F,
            Self::E5M2 => exponent == self.reserved_exponent() && mantissa != 0,
        }
    }

    /// Whether a magnitude byte encodes infinity (sign already stripped).
    pub fn is_inf_code(self, magnitude: u8) -> bool {
        matches!(self, Self::E5M2) && magnitude == self.max_finite_code() + 1
    }

    /// Wire code used by the kernels to select the conversion instruction.
    pub fn code(self) -> u32 {
        match self {
            Self::E4M3 => 0,
            Self::E5M2 => 1,
        }
    }

    /// Inverse of [`Self::code`].
    pub fn from_code(code: u32) -> Self {
        if code == 0 { Self::E4M3 } else { Self::E5M2 }
    }
}

/// Magnitude (`sign` stripped) of the code.  Subnormal codes (exponent 0) step
/// by `2^(1 - bias - mantissa_bits)`, e.g. `2^-9` for e4m3 and `2^-16` for e5m2.
fn decode_magnitude(code: u8, fmt: Fp8Format) -> f32 {
    let mb = fmt.mantissa_bits();
    let bias = fmt.exponent_bias();
    let mantissa = (code & ((1u8 << mb) - 1)) as f32;
    let exponent = (code >> mb) as i32;
    if exponent == 0 {
        mantissa * exp2f(1 - bias - mb as i32)
    } else {
        (1.0 + mantissa / (1u32 << mb) as f32) * exp2f(exponent - bias)
    }
}

/// Exact `2^e` for the exponent range both formats use (-16..=16).
fn exp2f(e: i32) -> f32 {
    if (-126..=127).contains(&e) {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e < -126 {
        0.0
    } else {
        f32::INFINITY
    }
}

/// Byte -> f32, exact.  `0x7f` (and `0xff` for e4m3) decode as NaN.
pub fn fp8_to_f32(bits: u8, fmt: Fp8Format) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let magnitude = bits & 0x7F;
    if fmt.is_nan_code(magnitude) {
        return f32::NAN;
    }
    if fmt.is_inf_code(magnitude) {
        return sign * f32::INFINITY;
    }
    sign * decode_magnitude(magnitude, fmt)
}

/// f32 -> byte, matching `cvt.rn.satfinite.<fmt>x2.f32`: round to nearest
/// with ties to the even code, clamp `|x|` to the largest finite magnitude,
/// and map NaN to the NaN code.
pub fn f32_to_fp8(value: f32, fmt: Fp8Format) -> u8 {
    if value.is_nan() {
        // The hardware `cvt.rn.satfinite.<fmt>x2.f32` maps every NaN to the
        // *unsigned* NaN code; the sign is not preserved.
        return 0x7F;
    }
    let sign: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    let max_code = fmt.max_finite_code();
    // Saturation happens before the search: `max - x` loses every code to f32
    // rounding once `x` is far above the grid, which would otherwise make the
    // tie rule pick a sub-maximal code.
    if magnitude > decode_magnitude(max_code, fmt) {
        return sign | max_code;
    }
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0..=max_code {
        let distance = (magnitude - decode_magnitude(code, fmt)).abs();
        // Ties go to the even code: with `mb` mantissa bits the parity of the
        // code is the parity of the mantissa, which is exactly the IEEE
        // round-to-nearest-even tie rule on this grid.
        if distance < best_distance || (distance == best_distance && (code & 1) == 0) {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

/// Scale that maps `amax` onto the format's largest finite magnitude
/// (the C++ `v6-dev-fp8` recipe: `scale = amax / 448` for e4m3).  A zero or
/// non-finite amax gives 1.0, matching `fp8_scaling.cpp`.
pub fn amax_to_scale(amax: f32, fmt: Fp8Format) -> f32 {
    if amax > 0.0 && amax.is_finite() {
        amax / fmt.max_finite()
    } else {
        1.0
    }
}

/// Largest finite `|x|` in `values`; non-finite entries are ignored (0), the
/// same rule the C++ amax kernels use.  All-zero / all-non-finite gives 0.
pub fn amax_f32(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|v| if v.is_finite() { v.abs() } else { 0.0 })
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_decode_matches_documented_grid() {
        let cases = [
            (0x00, 0.0f32),
            (0x80, -0.0),
            (0x01, 2f32.powi(-9)),
            (0x07, 7.0 * 2f32.powi(-9)),
            (0x08, 2f32.powi(-6)),
            (0x38, 1.0),
            (0x39, 1.125),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0x7E, 448.0),
            (0xFE, -448.0),
        ];
        for (bits, want) in cases {
            assert_eq!(
                fp8_to_f32(bits, Fp8Format::E4M3),
                want,
                "e4m3 code {bits:#04x}"
            );
        }
        assert!(fp8_to_f32(0x7F, Fp8Format::E4M3).is_nan());
    }

    #[test]
    fn e5m2_decode_matches_documented_grid() {
        let cases = [
            (0x00, 0.0f32),
            (0x01, 2f32.powi(-16)),
            (0x03, 3.0 * 2f32.powi(-16)),
            (0x04, 2f32.powi(-14)),
            (0x3C, 1.0),
            (0x3E, 1.5),
            (0x42, 3.0),
            (0x7B, 57344.0),
            (0xFB, -57344.0),
        ];
        for (bits, want) in cases {
            assert_eq!(
                fp8_to_f32(bits, Fp8Format::E5M2),
                want,
                "e5m2 code {bits:#04x}"
            );
        }
        assert!(fp8_to_f32(0x7C, Fp8Format::E5M2).is_infinite());
        assert!(fp8_to_f32(0x7F, Fp8Format::E5M2).is_nan());
    }

    #[test]
    fn encode_is_exact_on_representable_values() {
        for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
            for code in 0u8..=fmt.max_finite_code() {
                let v = fp8_to_f32(code, fmt);
                if !v.is_finite() {
                    continue;
                }
                assert_eq!(
                    f32_to_fp8(v, fmt),
                    code,
                    "{fmt:?} round trip of code {code:#04x}"
                );
                assert_eq!(
                    f32_to_fp8(-v, fmt),
                    code | 0x80,
                    "{fmt:?} negative round trip"
                );
            }
        }
    }

    #[test]
    fn encode_saturates_and_keeps_nan() {
        assert_eq!(f32_to_fp8(1e9, Fp8Format::E4M3), 0x7E);
        assert_eq!(f32_to_fp8(-1e9, Fp8Format::E4M3), 0xFE);
        assert_eq!(f32_to_fp8(f32::INFINITY, Fp8Format::E4M3), 0x7E);
        assert_eq!(f32_to_fp8(1e30, Fp8Format::E5M2), 0x7B);
        assert_eq!(f32_to_fp8(f32::NAN, Fp8Format::E4M3), 0x7F);
        // The hardware drops the sign of NaN in both formats.
        assert_eq!(f32_to_fp8(-f32::NAN, Fp8Format::E5M2), 0x7F);
    }

    #[test]
    fn encode_rounds_to_nearest_even() {
        // Halfway between e4m3 1.0 (0x38) and 1.125 (0x39) -> even code 0x38.
        assert_eq!(f32_to_fp8(1.0625, Fp8Format::E4M3), 0x38);
        // Halfway between 1.125 (0x39) and 1.25 (0x3A) -> even code 0x3A.
        assert_eq!(f32_to_fp8(1.1875, Fp8Format::E4M3), 0x3A);
        // Halfway between 1.25 (0x3A) and 1.375 (0x3B) -> even code 0x3A.
        assert_eq!(f32_to_fp8(1.3125, Fp8Format::E4M3), 0x3A);
    }

    #[test]
    fn scale_and_amax_helpers_follow_the_cpp_recipe() {
        assert_eq!(amax_to_scale(0.0, Fp8Format::E4M3), 1.0);
        assert_eq!(amax_to_scale(f32::NAN, Fp8Format::E4M3), 1.0);
        assert_eq!(amax_to_scale(448.0, Fp8Format::E4M3), 1.0);
        assert_eq!(amax_to_scale(89.6, Fp8Format::E4M3), 0.2);
        assert_eq!(amax_f32(&[1.0, -3.0, f32::NAN, f32::INFINITY, 2.0]), 3.0);
        assert_eq!(amax_f32(&[f32::NAN]), 0.0);
    }
}
