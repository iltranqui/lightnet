//! Host-side VC-attention quantization primitives, transcribed from the
//! historical `flash_expcast.cuh` source. These helpers are not wired into the
//! ViT cfg/runtime path yet.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpCastMode {
    E4M3,
    Order0,
    Order2,
}

impl TryFrom<u8> for ExpCastMode {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::E4M3),
            1 => Ok(Self::Order0),
            2 => Ok(Self::Order2),
            _ => Err(format!(
                "unsupported VC ExpCast mode {value}; expected 0, 1, or 2"
            )),
        }
    }
}

/// Encode an f32 as saturating E4M3 with round-to-nearest-even, matching the
/// source `Darknet::vc::e4m3_encode_bits` bit-level conversion.
pub(crate) fn e4m3_encode_bits(value: f32) -> u8 {
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let raw_exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if raw_exp == 0xff {
        // Source saturates both infinities and NaNs to signed max-finite.
        return sign | 0x7e;
    }
    if raw_exp == 0 {
        // f32 subnormals are all below the E4M3 half-min-subnormal threshold.
        return sign;
    }

    let unbiased_exp = raw_exp - 127;
    let mut e4m3_exp = unbiased_exp + 7;
    if e4m3_exp <= 0 {
        let shift = 14 - unbiased_exp;
        if shift > 24 {
            return sign;
        }
        let full = 0x80_0000 | mantissa;
        let mut significand = full >> shift;
        let remainder = full & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && significand & 1 != 0) {
            significand += 1;
        }
        if significand >= 8 {
            return sign | 0x08;
        }
        return sign | significand as u8;
    }
    if e4m3_exp > 15 {
        return sign | 0x7e;
    }

    let mut e4m3_mantissa = mantissa >> 20;
    let remainder = mantissa & 0x0f_ffff;
    let halfway = 0x08_0000;
    if remainder > halfway || (remainder == halfway && e4m3_mantissa & 1 != 0) {
        e4m3_mantissa += 1;
        if e4m3_mantissa >= 8 {
            e4m3_mantissa = 0;
            e4m3_exp += 1;
        }
    }
    if e4m3_exp >= 16 || (e4m3_exp == 15 && e4m3_mantissa >= 7) {
        return sign | 0x7e;
    }
    sign | ((e4m3_exp as u8) << 3) | e4m3_mantissa as u8
}

/// Decode E4M3, including the source convention that the reserved NaN code
/// decodes to signed max-finite.
pub(crate) fn e4m3_to_f32(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0 {
        return sign * f32::from(mantissa) * (1.0 / 512.0);
    }
    if exponent == 15 && mantissa == 7 {
        return sign * 448.0;
    }
    let base = (1u32 << (exponent - 1)) as f32 * (1.0 / 64.0);
    sign * base * (1.0 + f32::from(mantissa) * 0.125)
}

/// Quantize a log2 probability exponent into the source's 2^8-scaled E4M3
/// probability code. Inputs are finite in the VC attention score path.
pub(crate) fn expcast_probability_code(u: f32, mode: ExpCastMode) -> Result<u8, String> {
    if !u.is_finite() {
        return Err("VC ExpCast requires a finite log2 probability".into());
    }
    let code = match mode {
        ExpCastMode::E4M3 => e4m3_encode_bits(u.exp2() * 256.0),
        ExpCastMode::Order0 => {
            // round-to-nearest-even of 8*u + 120 + beta, beta = -0.35.
            clip_probability_code((8.0 * u + 120.0 - 0.35).round_ties_even())
        }
        ExpCastMode::Order2 => {
            // Schraudolph order-2 mantissa correction used by the source.
            if u <= -15.0 {
                0
            } else if u >= 0.0 {
                120
            } else {
                let exponent = u.floor();
                let fraction = u - exponent;
                let correction = fraction * (0.6565 + 0.3435 * fraction);
                let raw = (exponent + 15.0) * 8.0 + (correction * 8.0).round_ties_even();
                clip_probability_code(raw)
            }
        }
    };
    Ok(code)
}

fn clip_probability_code(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 120.0 {
        120
    } else {
        value as u8
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpCastMode, e4m3_encode_bits, e4m3_to_f32, expcast_probability_code};

    #[test]
    fn e4m3_decoded_codes_roundtrip_except_reserved_nan_codes() {
        for code in 0u8..=u8::MAX {
            if code & 0x7f == 0x7f {
                continue;
            }
            assert_eq!(e4m3_encode_bits(e4m3_to_f32(code)), code, "{code:#04x}");
        }
    }

    #[test]
    fn e4m3_covers_subnormal_normal_saturation_and_special_values() {
        assert_eq!(e4m3_encode_bits(0.0), 0x00);
        assert_eq!(e4m3_encode_bits(-0.0), 0x80);
        assert_eq!(e4m3_encode_bits(2.0_f32.powi(-9)), 0x01);
        assert_eq!(e4m3_encode_bits(2.0_f32.powi(-10)), 0x00);
        assert_eq!(e4m3_encode_bits(3.0 * 2.0_f32.powi(-10)), 0x02);
        assert_eq!(e4m3_encode_bits(2.0_f32.powi(-6)), 0x08);
        assert_eq!(e4m3_encode_bits(1.0), 0x38);
        assert_eq!(e4m3_encode_bits(256.0), 0x78);
        assert_eq!(e4m3_encode_bits(448.0), 0x7e);
        assert_eq!(e4m3_encode_bits(f32::INFINITY), 0x7e);
        assert_eq!(e4m3_encode_bits(f32::NEG_INFINITY), 0xfe);
        assert_eq!(e4m3_encode_bits(f32::NAN), 0x7e);
        assert_eq!(e4m3_to_f32(0x7f), 448.0);
        assert_eq!(e4m3_to_f32(0xff), -448.0);
    }

    #[test]
    fn expcast_modes_match_source_hand_calculations() {
        let inputs = [-15.0, -8.0, -1.0, 0.0];
        let expected = [0, 56, 112, 120];
        for (u, code) in inputs.into_iter().zip(expected) {
            assert_eq!(
                expcast_probability_code(u, ExpCastMode::Order0).unwrap(),
                code
            );
            assert_eq!(
                expcast_probability_code(u, ExpCastMode::Order2).unwrap(),
                code
            );
        }
        assert_eq!(
            expcast_probability_code(-1.0, ExpCastMode::E4M3).unwrap(),
            112
        );
        assert_eq!(
            expcast_probability_code(-0.5, ExpCastMode::Order0).unwrap(),
            116
        );
        assert_eq!(
            expcast_probability_code(-0.5, ExpCastMode::Order2).unwrap(),
            115
        );
    }

    #[test]
    fn expcast_rejects_invalid_mode_and_non_finite_input() {
        assert!(ExpCastMode::try_from(3).is_err());
        assert!(expcast_probability_code(f32::NAN, ExpCastMode::Order0).is_err());
        assert!(expcast_probability_code(f32::INFINITY, ExpCastMode::E4M3).is_err());
    }
}
