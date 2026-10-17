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

const V_SMOOTH_TILE_SIZE: usize = 64;
const E4M3_MAX_FINITE_CODE: u8 = 0x7e;

#[derive(Clone, Debug)]
pub(crate) struct VsmoothCache {
    tokens: usize,
    channels: usize,
    permutation: Vec<usize>,
    means: Vec<f32>,
    residual_codes: Vec<u8>,
    normalized_residuals: Vec<f32>,
    scale_max_positions: Vec<usize>,
    scale_max_signs: Vec<f32>,
    scale_ties: Vec<bool>,
    zero_scales: Vec<bool>,
}

/// Apply source-style V-Smooth to one batch-head tensor. The permutation maps
/// each new position to its original token; `None` uses sequence order.
pub(crate) fn vsmooth_forward(
    values: &[f32],
    tokens: usize,
    channels: usize,
    permutation: Option<&[usize]>,
) -> Result<(Vec<f32>, VsmoothCache), String> {
    if tokens == 0 || channels == 0 {
        return Err("VC V-Smooth requires positive token and channel counts".into());
    }
    let expected = tokens
        .checked_mul(channels)
        .ok_or("VC V-Smooth tensor size overflow")?;
    if values.len() != expected {
        return Err(format!(
            "VC V-Smooth input has {} values, expected {expected}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("VC V-Smooth input must contain only finite values".into());
    }
    let permutation = match permutation {
        Some(permutation) => {
            if permutation.len() != tokens {
                return Err(format!(
                    "VC V-Smooth permutation has {} entries, expected {tokens}",
                    permutation.len()
                ));
            }
            let mut seen = vec![false; tokens];
            for &token in permutation {
                if token >= tokens || seen[token] {
                    return Err("VC V-Smooth permutation is not a token bijection".into());
                }
                seen[token] = true;
            }
            permutation.to_vec()
        }
        None => (0..tokens).collect(),
    };

    let tile_count = tokens.div_ceil(V_SMOOTH_TILE_SIZE);
    let mut means = vec![0.0; tile_count * channels];
    for tile in 0..tile_count {
        let begin = tile * V_SMOOTH_TILE_SIZE;
        let end = (begin + V_SMOOTH_TILE_SIZE).min(tokens);
        let rows = (end - begin) as f32;
        for channel in 0..channels {
            let mut sum = 0.0;
            for position in begin..end {
                sum += values[permutation[position] * channels + channel];
            }
            means[tile * channels + channel] = sum / rows;
        }
    }

    let mut scales = vec![1.0; channels];
    let mut scale_max_positions = vec![0; channels];
    let mut scale_max_signs = vec![0.0; channels];
    let mut scale_ties = vec![false; channels];
    let mut zero_scales = vec![false; channels];
    for channel in 0..channels {
        let mut maximum = 0.0f32;
        let mut maximum_position = 0;
        let mut maxima = 0usize;
        for position in 0..tokens {
            let tile = position / V_SMOOTH_TILE_SIZE;
            let value = values[permutation[position] * channels + channel];
            let residual = value - means[tile * channels + channel];
            let magnitude = residual.abs();
            if magnitude > maximum {
                maximum = magnitude;
                maximum_position = position;
                maxima = 1;
            } else if magnitude == maximum {
                maxima += 1;
            }
        }
        if maximum == 0.0 {
            zero_scales[channel] = true;
        } else {
            scales[channel] = maximum / 448.0;
            scale_max_positions[channel] = maximum_position;
            let tile = maximum_position / V_SMOOTH_TILE_SIZE;
            let value = values[permutation[maximum_position] * channels + channel];
            scale_max_signs[channel] = (value - means[tile * channels + channel]).signum();
            scale_ties[channel] = maxima > 1;
        }
    }

    let mut residual_codes = vec![0; expected];
    let mut normalized_residuals = vec![0.0; expected];
    let mut output = vec![0.0; expected];
    for position in 0..tokens {
        let tile = position / V_SMOOTH_TILE_SIZE;
        for channel in 0..channels {
            let value = values[permutation[position] * channels + channel];
            let residual = value - means[tile * channels + channel];
            let normalized = residual / scales[channel];
            let index = position * channels + channel;
            let code = e4m3_encode_bits(normalized);
            residual_codes[index] = code;
            normalized_residuals[index] = normalized;
            output[index] = means[tile * channels + channel] + scales[channel] * e4m3_to_f32(code);
        }
    }

    Ok((
        output,
        VsmoothCache {
            tokens,
            channels,
            permutation,
            means,
            residual_codes,
            normalized_residuals,
            scale_max_positions,
            scale_max_signs,
            scale_ties,
            zero_scales,
        },
    ))
}

/// Exact piecewise derivative of V-Smooth at points where its E4M3 codes and
/// unique nonzero residual-scale maxima are locally stable. Returns an error
/// where the classical derivative is undefined.
pub(crate) fn vsmooth_backward(
    output_grad: &[f32],
    cache: &VsmoothCache,
) -> Result<Vec<f32>, String> {
    let expected = cache
        .tokens
        .checked_mul(cache.channels)
        .ok_or("VC V-Smooth gradient size overflow")?;
    if output_grad.len() != expected {
        return Err(format!(
            "VC V-Smooth output gradient has {} values, expected {expected}",
            output_grad.len()
        ));
    }
    for channel in 0..cache.channels {
        if cache.zero_scales[channel] {
            return Err(format!(
                "VC V-Smooth scale derivative is undefined for zero-residual channel {channel}"
            ));
        }
        if cache.scale_ties[channel] {
            return Err(format!(
                "VC V-Smooth scale derivative is undefined for tied maxima in channel {channel}"
            ));
        }
    }
    if cache
        .normalized_residuals
        .iter()
        .copied()
        .any(e4m3_quantization_boundary)
    {
        return Err("VC V-Smooth derivative is undefined at an E4M3 rounding boundary".into());
    }

    let tile_count = cache.means.len() / cache.channels;
    let mut input_grad = vec![0.0; expected];
    for channel in 0..cache.channels {
        let mut scale_grad = 0.0;
        for position in 0..cache.tokens {
            let index = position * cache.channels + channel;
            scale_grad += output_grad[index] * e4m3_to_f32(cache.residual_codes[index]);
        }

        for tile in 0..tile_count {
            let begin = tile * V_SMOOTH_TILE_SIZE;
            let end = (begin + V_SMOOTH_TILE_SIZE).min(cache.tokens);
            let mut mean_grad = 0.0;
            for position in begin..end {
                mean_grad += output_grad[position * cache.channels + channel];
            }
            let contribution = mean_grad / (end - begin) as f32;
            for position in begin..end {
                let original_token = cache.permutation[position];
                input_grad[original_token * cache.channels + channel] += contribution;
            }
        }

        let maximum_position = cache.scale_max_positions[channel];
        let maximum_tile = maximum_position / V_SMOOTH_TILE_SIZE;
        let tile_begin = maximum_tile * V_SMOOTH_TILE_SIZE;
        let tile_end = (tile_begin + V_SMOOTH_TILE_SIZE).min(cache.tokens);
        let mean_derivative = 1.0 / (tile_end - tile_begin) as f32;
        let scale_derivative = cache.scale_max_signs[channel] / 448.0;
        for position in 0..cache.tokens {
            let direct = if position == maximum_position {
                1.0
            } else {
                0.0
            };
            let centered_mean = if position >= tile_begin && position < tile_end {
                mean_derivative
            } else {
                0.0
            };
            let original_token = cache.permutation[position];
            input_grad[original_token * cache.channels + channel] +=
                scale_grad * scale_derivative * (direct - centered_mean);
        }
    }
    Ok(input_grad)
}

fn e4m3_quantization_boundary(value: f32) -> bool {
    let magnitude = value.abs();
    let code = e4m3_encode_bits(magnitude).min(E4M3_MAX_FINITE_CODE);
    let lower_boundary = code > 0 && magnitude == (e4m3_to_f32(code - 1) + e4m3_to_f32(code)) * 0.5;
    let upper_boundary = code < E4M3_MAX_FINITE_CODE
        && magnitude == (e4m3_to_f32(code) + e4m3_to_f32(code + 1)) * 0.5;
    lower_boundary || upper_boundary
}

fn probability_code_boundary(u: f32, mode: ExpCastMode) -> bool {
    match mode {
        ExpCastMode::E4M3 => e4m3_quantization_boundary(u.exp2() * 256.0),
        ExpCastMode::Order0 => {
            let raw = 8.0 * u + 120.0 - 0.35;
            let lower = raw.floor();
            lower >= 0.0 && lower < 120.0 && raw - lower == 0.5
        }
        ExpCastMode::Order2 => {
            if u <= -15.0 || u >= 0.0 {
                return false;
            }
            let exponent = u.floor();
            let fraction = u - exponent;
            let correction = fraction * (0.6565 + 0.3435 * fraction) * 8.0;
            let lower = correction.floor();
            let base = (exponent + 15.0) * 8.0;
            base + lower >= 0.0 && base + lower < 120.0 && correction - lower == 0.5
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct QuantizedOnlineRowCache {
    tokens: usize,
    channels: usize,
    scores_log2: Vec<f32>,
    values: Vec<f32>,
    context: Vec<f32>,
    probability_codes: Vec<u8>,
    prefix_maxes: Vec<f32>,
    prefix_owners: Vec<usize>,
    prefix_ties: Vec<bool>,
    mode: ExpCastMode,
}

/// Source-shaped online ExpCast attention for one query row. Scores are
/// finite log2 logits after score clamping; values are the dequantized V-Smooth
/// rows in KV permutation order. The retained row state is O(T + T*C).
pub(crate) fn quantized_online_row_forward(
    scores_log2: &[f32],
    values: &[f32],
    channels: usize,
    mode: ExpCastMode,
) -> Result<(Vec<f32>, f32, QuantizedOnlineRowCache), String> {
    let tokens = scores_log2.len();
    if tokens == 0 || channels == 0 {
        return Err("VC online attention requires positive token and channel counts".into());
    }
    let expected_values = tokens
        .checked_mul(channels)
        .ok_or("VC online attention value size overflow")?;
    if values.len() != expected_values {
        return Err(format!(
            "VC online attention values have {} entries, expected {expected_values}",
            values.len()
        ));
    }
    if scores_log2.iter().any(|score| !score.is_finite())
        || values.iter().any(|value| !value.is_finite())
    {
        return Err("VC online attention requires finite scores and values".into());
    }

    let tile_count = tokens.div_ceil(V_SMOOTH_TILE_SIZE);
    let mut prefix_maxes = vec![0.0; tile_count];
    let mut prefix_owners = vec![0; tile_count];
    let mut prefix_ties = vec![false; tile_count];
    let mut probability_codes = vec![0; tokens];
    let mut running_max = f32::NEG_INFINITY;
    let mut running_owner = 0;
    let mut running_tie = false;

    for tile in 0..tile_count {
        let begin = tile * V_SMOOTH_TILE_SIZE;
        let end = (begin + V_SMOOTH_TILE_SIZE).min(tokens);
        let mut tile_max = f32::NEG_INFINITY;
        let mut tile_owner = begin;
        let mut tile_max_count = 0usize;
        for position in begin..end {
            let score = scores_log2[position];
            if score > tile_max {
                tile_max = score;
                tile_owner = position;
                tile_max_count = 1;
            } else if score == tile_max {
                tile_max_count += 1;
            }
        }
        if tile_max > running_max {
            running_max = tile_max;
            running_owner = tile_owner;
            running_tie = tile_max_count > 1;
        } else if tile_max == running_max {
            running_tie = true;
        }
        prefix_maxes[tile] = running_max;
        prefix_owners[tile] = running_owner;
        prefix_ties[tile] = running_tie;
        for position in begin..end {
            let u = scores_log2[position] - running_max;
            probability_codes[position] = expcast_probability_code(u, mode)?;
        }
    }

    let final_max = prefix_maxes[tile_count - 1];
    let mut normalizer = 0.0;
    for position in 0..tokens {
        let tile = position / V_SMOOTH_TILE_SIZE;
        let prefix_scale = (prefix_maxes[tile] - final_max).exp2();
        normalizer += e4m3_to_f32(probability_codes[position]) * prefix_scale;
    }
    if !(normalizer > 0.0 && normalizer.is_finite()) {
        return Err("VC online attention produced an invalid probability sum".into());
    }
    let mut context = vec![0.0; channels];
    for position in 0..tokens {
        let tile = position / V_SMOOTH_TILE_SIZE;
        let prefix_scale = (prefix_maxes[tile] - final_max).exp2();
        let weight = e4m3_to_f32(probability_codes[position]) * prefix_scale / normalizer;
        for channel in 0..channels {
            context[channel] += weight * values[position * channels + channel];
        }
    }
    let lse = (final_max + normalizer.log2() - 8.0) * 0.6931471805599453;
    Ok((
        context.clone(),
        lse,
        QuantizedOnlineRowCache {
            tokens,
            channels,
            scores_log2: scores_log2.to_vec(),
            values: values.to_vec(),
            context,
            probability_codes,
            prefix_maxes,
            prefix_owners,
            prefix_ties,
            mode,
        },
    ))
}

/// Exact context derivative of the quantized online row away from ExpCast
/// transitions and prefix-max ties. Probability codes are held fixed, but the
/// online inter-tile rescaling through prefix maxima remains differentiable.
pub(crate) fn quantized_online_row_backward(
    output_grad: &[f32],
    cache: &QuantizedOnlineRowCache,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if output_grad.len() != cache.channels {
        return Err(format!(
            "VC online attention output gradient has {} entries, expected {}",
            output_grad.len(),
            cache.channels
        ));
    }
    if cache.prefix_ties.iter().any(|&tied| tied) {
        return Err("VC online attention derivative is undefined at a prefix-max tie".into());
    }
    if (0..cache.tokens).any(|position| {
        let tile = position / V_SMOOTH_TILE_SIZE;
        probability_code_boundary(
            cache.scores_log2[position] - cache.prefix_maxes[tile],
            cache.mode,
        )
    }) {
        return Err("VC online attention derivative is undefined at an ExpCast boundary".into());
    }

    let tile_count = cache.prefix_maxes.len();
    let final_max = cache.prefix_maxes[tile_count - 1];
    let mut tile_sums = vec![0.0; tile_count];
    let mut tile_values = vec![vec![0.0; cache.channels]; tile_count];
    let mut normalizer = 0.0;
    for position in 0..cache.tokens {
        let tile = position / V_SMOOTH_TILE_SIZE;
        let probability = e4m3_to_f32(cache.probability_codes[position]);
        let prefix_scale = (cache.prefix_maxes[tile] - final_max).exp2();
        tile_sums[tile] += probability;
        normalizer += probability * prefix_scale;
        for channel in 0..cache.channels {
            tile_values[tile][channel] +=
                probability * cache.values[position * cache.channels + channel];
        }
    }

    let mut values_grad = vec![0.0; cache.tokens * cache.channels];
    let mut prefix_max_grad = vec![0.0; tile_count];
    for tile in 0..tile_count {
        let prefix_scale = (cache.prefix_maxes[tile] - final_max).exp2();
        let mut centered_context_dot = 0.0;
        for channel in 0..cache.channels {
            centered_context_dot += output_grad[channel]
                * (tile_values[tile][channel] - cache.context[channel] * tile_sums[tile]);
        }
        prefix_max_grad[tile] =
            0.6931471805599453 * prefix_scale * centered_context_dot / normalizer;
        let begin = tile * V_SMOOTH_TILE_SIZE;
        let end = (begin + V_SMOOTH_TILE_SIZE).min(cache.tokens);
        for position in begin..end {
            let weight = e4m3_to_f32(cache.probability_codes[position]) * prefix_scale / normalizer;
            for channel in 0..cache.channels {
                values_grad[position * cache.channels + channel] = weight * output_grad[channel];
            }
        }
    }

    let total_prefix_grad: f32 = prefix_max_grad.iter().sum();
    prefix_max_grad[tile_count - 1] -= total_prefix_grad;
    let mut scores_grad = vec![0.0; cache.tokens];
    for (tile, &owner) in cache.prefix_owners.iter().enumerate() {
        scores_grad[owner] += prefix_max_grad[tile];
    }
    Ok((scores_grad, values_grad))
}

#[cfg(test)]
mod tests {
    use super::{
        ExpCastMode, e4m3_encode_bits, e4m3_quantization_boundary, e4m3_to_f32,
        expcast_probability_code, quantized_online_row_backward, quantized_online_row_forward,
        vsmooth_backward, vsmooth_forward,
    };

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

    #[test]
    fn e4m3_boundary_detector_finds_rounding_midpoints() {
        let boundary = 2.0_f32.powi(-10);
        assert!(e4m3_quantization_boundary(boundary));
        assert!(!e4m3_quantization_boundary(f32::from_bits(
            boundary.to_bits() + 1
        )));
        assert!(e4m3_quantization_boundary(432.0));
        assert!(!e4m3_quantization_boundary(448.0));
    }

    #[test]
    fn vsmooth_backward_matches_finite_differences_away_from_quantization_edges() {
        let tokens = 70;
        let channels = 2;
        let values: Vec<f32> = (0..tokens)
            .flat_map(|token| {
                let t = token as f32;
                [
                    (t * 0.173).sin() * 0.73 + t * 0.011,
                    (t * 0.119).cos() * 0.41 - t * 0.007,
                ]
            })
            .collect();
        let permutation: Vec<usize> = (0..tokens)
            .map(|position| (position + 23) % tokens)
            .collect();
        let (output, cache) =
            vsmooth_forward(&values, tokens, channels, Some(&permutation)).unwrap();
        let output_grad: Vec<f32> = (0..output.len())
            .map(|index| (index as f32 * 0.37).cos())
            .collect();
        let analytic = vsmooth_backward(&output_grad, &cache).unwrap();
        let objective = |input: &[f32]| {
            let (output, _) = vsmooth_forward(input, tokens, channels, Some(&permutation)).unwrap();
            output
                .iter()
                .zip(&output_grad)
                .map(|(value, grad)| value * grad)
                .sum::<f32>()
        };
        let epsilon = 1.0e-5;
        let mut checked = 0;
        for index in 0..values.len() {
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let (_, plus_cache) =
                vsmooth_forward(&plus, tokens, channels, Some(&permutation)).unwrap();
            let (_, minus_cache) =
                vsmooth_forward(&minus, tokens, channels, Some(&permutation)).unwrap();
            if plus_cache.residual_codes != cache.residual_codes
                || minus_cache.residual_codes != cache.residual_codes
                || plus_cache.scale_max_positions != cache.scale_max_positions
                || minus_cache.scale_max_positions != cache.scale_max_positions
            {
                continue;
            }
            let numeric = (objective(&plus) - objective(&minus)) / (2.0 * epsilon);
            let error = (analytic[index] - numeric).abs();
            assert!(
                error <= 0.015 + 0.025 * numeric.abs(),
                "gradient {index}: analytic {}, numeric {numeric}, error {error}",
                analytic[index]
            );
            checked += 1;
        }
        assert!(
            checked > 80,
            "only {checked} stable coordinates were checked"
        );
    }

    #[test]
    fn vsmooth_backward_rejects_undefined_scale_derivatives() {
        let grad = vec![1.0; 64];
        let tied: Vec<f32> = (0..64)
            .map(|position| match position {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        let (_, tied_cache) = vsmooth_forward(&tied, 64, 1, None).unwrap();
        assert!(vsmooth_backward(&grad, &tied_cache).is_err());

        let constant = vec![2.0; 64];
        let (_, constant_cache) = vsmooth_forward(&constant, 64, 1, None).unwrap();
        assert!(vsmooth_backward(&grad, &constant_cache).is_err());
    }

    #[test]
    fn online_expcast_backward_matches_finite_differences_through_alpha() {
        let tokens = 70;
        let channels = 2;
        let mut scores: Vec<f32> = (0..tokens)
            .map(|position| -2.17 - (position % 13) as f32 * 0.293)
            .collect();
        scores[0] = 0.237;
        scores[64] = 0.713;
        let values: Vec<f32> = (0..tokens)
            .flat_map(|position| {
                if position < 64 {
                    [1.0 + position as f32 * 0.001, -0.5]
                } else {
                    [-0.8, 1.3 + position as f32 * 0.001]
                }
            })
            .collect();
        let (context, _, cache) =
            quantized_online_row_forward(&scores, &values, channels, ExpCastMode::Order0).unwrap();
        let output_grad = [0.8, -0.45];
        let (scores_grad, values_grad) =
            quantized_online_row_backward(&output_grad, &cache).unwrap();
        assert!(
            scores_grad[0].abs() > 1.0e-5,
            "old prefix max must affect alpha"
        );
        assert!(
            scores_grad[64].abs() > 1.0e-5,
            "final max must affect alpha"
        );
        assert_eq!(cache.probability_codes.len(), tokens);
        assert_eq!(cache.prefix_maxes.len(), tokens.div_ceil(64));

        let objective = |row_scores: &[f32], row_values: &[f32]| {
            let (output, _, _) =
                quantized_online_row_forward(row_scores, row_values, channels, ExpCastMode::Order0)
                    .unwrap();
            output
                .iter()
                .zip(output_grad)
                .map(|(value, grad)| value * grad)
                .sum::<f32>()
        };
        let epsilon = 1.0e-4;
        let mut checked_scores = 0;
        for index in 0..tokens {
            let mut plus = scores.clone();
            let mut minus = scores.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let (_, _, plus_cache) =
                quantized_online_row_forward(&plus, &values, channels, ExpCastMode::Order0)
                    .unwrap();
            let (_, _, minus_cache) =
                quantized_online_row_forward(&minus, &values, channels, ExpCastMode::Order0)
                    .unwrap();
            if plus_cache.probability_codes != cache.probability_codes
                || minus_cache.probability_codes != cache.probability_codes
                || plus_cache.prefix_owners != cache.prefix_owners
                || minus_cache.prefix_owners != cache.prefix_owners
            {
                continue;
            }
            let numeric =
                (objective(&plus, &values) - objective(&minus, &values)) / (2.0 * epsilon);
            let error = (scores_grad[index] - numeric).abs();
            assert!(
                error <= 0.02 + 0.04 * numeric.abs(),
                "score gradient {index}: analytic {}, numeric {numeric}, error {error}",
                scores_grad[index]
            );
            checked_scores += 1;
        }
        assert!(
            checked_scores > 60,
            "only {checked_scores} score entries checked"
        );

        let mut checked_values = 0;
        for index in 0..values.len() {
            let mut plus = values.clone();
            let mut minus = values.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric =
                (objective(&scores, &plus) - objective(&scores, &minus)) / (2.0 * epsilon);
            let error = (values_grad[index] - numeric).abs();
            assert!(
                error <= 0.005,
                "value gradient {index}: analytic {}, numeric {numeric}, error {error}",
                values_grad[index]
            );
            checked_values += 1;
        }
        assert_eq!(checked_values, values.len());
        assert!(context.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn online_expcast_score_derivative_is_zero_for_a_single_kv_tile() {
        let scores = [0.237, -1.13, -2.71, -3.19];
        let values = [0.1, 0.7, -0.3, 1.4];
        let (_, _, cache) =
            quantized_online_row_forward(&scores, &values, 1, ExpCastMode::Order0).unwrap();
        let (scores_grad, _) = quantized_online_row_backward(&[1.0], &cache).unwrap();
        assert_eq!(scores_grad, vec![0.0; scores.len()]);
    }

    #[test]
    fn online_expcast_backward_rejects_prefix_max_ties() {
        let scores = [0.0, 0.0];
        let values = [1.0, -1.0];
        let (_, _, cache) =
            quantized_online_row_forward(&scores, &values, 1, ExpCastMode::Order0).unwrap();
        assert!(quantized_online_row_backward(&[1.0], &cache).is_err());
    }
}
