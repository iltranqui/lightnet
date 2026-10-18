//! `[laya_marker_gather]`: the LAYA decision-model per-option gather (Phase 1b
//! of `tasks/rust/layers.md`, unblocked by Open Question 2), ported from
//! `Jev_laya:src-lib/laya_marker_gather_layer.cpp`.
//!
//! The layer has no weights.  It reads two inputs, route-style:
//!
//! * **Input A** - the hidden-state tensor, channel-major `[batch, hidden, 1, T]`
//!   (in practice `[laya_head]`'s output).
//! * **Input B** - the marker-info tensor, channel-major
//!   `[batch, marker_channels, 1, T]` with `marker_channels >= 2 * max_options`.
//!   Channel `2*j` holds `marker_pos[j]` broadcast as a constant across the `T`
//!   positions, channel `2*j+1` holds `marker_mask[j]` (1/0).  The mask is **not**
//!   read here: masking is deferred to CLI-side post-processing, and garbage in
//!   a masked-out slot is fine.
//!
//! Per sample `b`, per option `j`:
//!
//! ```text
//! pos = clamp(round(inputB[b, 2*j, 0, 0]), 0, T-1)
//! output[b, :, 0, j] = inputA[b, :, 0, pos]        (all `hidden` channels)
//! ```
//!
//! Output is channel-major `[batch, hidden, 1, max_options]`.
//!
//! The C++ header calls the backward a "no-op" but the `.cpp` implements the
//! exact adjoint (a scatter-add of the output gradient into input A at each
//! marker position), which is the derivative of the forward above; we port that.
//! Input B carries integer indices, so it has no gradient.

/// C++ `std::lround(static_cast<double>(value))` clamped to `[0, t-1]`.
///
/// `f64::round` rounds half away from zero, matching C++ `lround`; going through
/// `f64` first matches `static_cast<double>` before the round.
pub fn marker_position(value: f32, t: usize) -> usize {
    debug_assert!(t > 0);
    let rounded = (value as f64).round();
    if rounded < 0.0 {
        0
    } else if rounded >= t as f64 {
        t - 1
    } else {
        rounded as usize
    }
}

/// C++ `forward_laya_marker_gather_layer`.
///
/// `input_a` is `[batch, hidden, 1, T]` and `input_b` is
/// `[batch, marker_channels, 1, T]`, both channel-major; the result is
/// `[batch, hidden, 1, max_options]`.
pub fn forward(
    input_a: &[f32],
    input_b: &[f32],
    batch: usize,
    hidden: usize,
    max_options: usize,
    seq_len: usize,
    marker_channels: usize,
) -> Result<Vec<f32>, String> {
    if hidden == 0 || max_options == 0 || seq_len == 0 {
        return Err(format!(
            "laya_marker_gather: hidden={hidden} max_options={max_options} seq_len={seq_len} must all be > 0"
        ));
    }
    if marker_channels < 2 * max_options {
        return Err(format!(
            "laya_marker_gather: marker source has {marker_channels} channels, needs >= 2*max_options={}",
            2 * max_options
        ));
    }
    if input_a.len() != batch * hidden * seq_len {
        return Err(format!(
            "laya_marker_gather: input A has {} values, expected batch*hidden*seq_len={}",
            input_a.len(),
            batch * hidden * seq_len
        ));
    }
    if input_b.len() != batch * marker_channels * seq_len {
        return Err(format!(
            "laya_marker_gather: input B has {} values, expected batch*marker_channels*seq_len={}",
            input_b.len(),
            batch * marker_channels * seq_len
        ));
    }

    let mut output = vec![0.0f32; batch * hidden * max_options];
    for b in 0..batch {
        for j in 0..max_options {
            // Channel `2*j` of B, column 0 of the sequence: every T column holds
            // the same broadcast value.
            let pos = marker_position(input_b[(b * marker_channels + 2 * j) * seq_len], seq_len);
            for c in 0..hidden {
                output[(b * hidden + c) * max_options + j] =
                    input_a[(b * hidden + c) * seq_len + pos];
            }
        }
    }
    Ok(output)
}

/// C++ `backward_laya_marker_gather_layer`: the exact adjoint of [`forward`].
///
/// Returns input A's gradient, `[batch, hidden, 1, T]`.  Duplicate marker
/// positions accumulate (the scatter is a `+=`), so options that point at the
/// same position share that position's gradient.  Input B gets no gradient.
pub fn backward(
    input_b: &[f32],
    delta: &[f32],
    batch: usize,
    hidden: usize,
    max_options: usize,
    seq_len: usize,
    marker_channels: usize,
) -> Result<Vec<f32>, String> {
    if hidden == 0 || max_options == 0 || seq_len == 0 {
        return Err(format!(
            "laya_marker_gather: hidden={hidden} max_options={max_options} seq_len={seq_len} must all be > 0"
        ));
    }
    if marker_channels < 2 * max_options {
        return Err(format!(
            "laya_marker_gather: marker source has {marker_channels} channels, needs >= 2*max_options={}",
            2 * max_options
        ));
    }
    if delta.len() != batch * hidden * max_options {
        return Err(format!(
            "laya_marker_gather: output gradient has {} values, expected batch*hidden*max_options={}",
            delta.len(),
            batch * hidden * max_options
        ));
    }
    if input_b.len() != batch * marker_channels * seq_len {
        return Err(format!(
            "laya_marker_gather: input B has {} values, expected batch*marker_channels*seq_len={}",
            input_b.len(),
            batch * marker_channels * seq_len
        ));
    }

    let mut delta_a = vec![0.0f32; batch * hidden * seq_len];
    for b in 0..batch {
        for j in 0..max_options {
            let pos = marker_position(input_b[(b * marker_channels + 2 * j) * seq_len], seq_len);
            for c in 0..hidden {
                delta_a[(b * hidden + c) * seq_len + pos] +=
                    delta[(b * hidden + c) * max_options + j];
            }
        }
    }
    Ok(delta_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Channel-major `[batch, hidden, 1, seq_len]`.
    fn a_data(batch: usize, hidden: usize, seq_len: usize) -> Vec<f32> {
        (0..batch * hidden * seq_len)
            .map(|i| i as f32 * 0.5 - 1.0)
            .collect()
    }

    /// Channel-major `[batch, marker_channels, 1, seq_len]` with per-option
    /// constant `marker_pos` in channel `2*j` and a mask in `2*j+1`.
    fn b_data(
        batch: usize,
        marker_channels: usize,
        seq_len: usize,
        positions: &[f32],
        mask: &[f32],
    ) -> Vec<f32> {
        let options = positions.len();
        let mut data = vec![0.0f32; batch * marker_channels * seq_len];
        for b in 0..batch {
            for j in 0..options {
                for t in 0..seq_len {
                    data[(b * marker_channels + 2 * j) * seq_len + t] = positions[j];
                    data[(b * marker_channels + 2 * j + 1) * seq_len + t] = mask[j];
                }
            }
        }
        data
    }

    #[test]
    fn gather_selects_the_marker_column_per_option() {
        // H=2, K=2, T=3, batch=1. A row-major (c, t):
        //   c0: [0, 1, 2]   c1: [3, 4, 5]
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let b = b_data(1, 4, 3, &[1.0, 2.0], &[1.0, 1.0]);
        let out = forward(&a, &b, 1, 2, 2, 3, 4).unwrap();
        // output [hidden=2, max_options=2]: c0 -> [a[c0,pos0], a[c0,pos1]] = [1, 2]
        //                                    c1 -> [a[c1,pos0], a[c1,pos1]] = [4, 5]
        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn marker_gather_axis_layout_uses_sample_specific_positions() {
        // NCHW A=[B,H,T], with B=2,H=1,T=3; each sample points at a
        // different column of its own sequence.
        let a = [10., 11., 12., 20., 21., 22.];
        let b = [1., 1., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1.];
        let out = forward(&a, &b, 2, 1, 1, 3, 2).unwrap();
        assert_eq!(out, vec![11., 22.]);
    }

    #[test]
    fn marker_positions_are_rounded_and_clamped_like_cxx() {
        // C++ std::lround: half away from zero.
        assert_eq!(marker_position(1.4, 5), 1);
        assert_eq!(marker_position(1.5, 5), 2);
        assert_eq!(marker_position(2.5, 5), 3);
        assert_eq!(marker_position(-0.4, 5), 0);
        assert_eq!(marker_position(-3.0, 5), 0);
        assert_eq!(marker_position(4.6, 5), 4);
        assert_eq!(marker_position(99.0, 5), 4);
        // A 1-wide sequence always clamps to column 0.
        assert_eq!(marker_position(7.0, 1), 0);
    }

    #[test]
    fn negative_and_out_of_range_positions_clamp() {
        let a = vec![0.0, 10.0, 20.0];
        let b = b_data(1, 2, 3, &[-5.0], &[1.0]);
        let out = forward(&a, &b, 1, 1, 1, 3, 2).unwrap();
        assert_eq!(out, vec![0.0]);
        let b = b_data(1, 2, 3, &[100.0], &[1.0]);
        let out = forward(&a, &b, 1, 1, 1, 3, 2).unwrap();
        assert_eq!(out, vec![20.0]);
    }

    #[test]
    fn extra_marker_channels_are_ignored() {
        // marker_channels = 6 >= 2*max_options = 4; only channels 0 and 2 are read.
        let a = vec![0.0, 10.0]; // [hidden=1, seq_len=2]
        let mut b = vec![0.0f32; 1 * 6 * 2];
        for t in 0..2 {
            b[0 * 2 + t] = 1.0; // channel 0 -> pos 1
            b[2 * 2 + t] = 0.0; // channel 2 -> pos 0
        }
        let out = forward(&a, &b, 1, 1, 2, 2, 6).unwrap();
        assert_eq!(out, vec![10.0, 0.0]);
    }

    #[test]
    fn duplicate_positions_accumulate_in_the_backward() {
        // Both options point at column 1, so its A-gradient is the sum.
        let b = b_data(1, 4, 3, &[1.0, 1.0], &[1.0, 1.0]);
        let delta = vec![2.0, 3.0]; // [hidden=1, max_options=2]
        let grad = backward(&b, &delta, 1, 1, 2, 3, 4).unwrap();
        assert_eq!(grad, vec![0.0, 5.0, 0.0]);
    }

    #[test]
    fn backward_is_the_exact_adjoint_of_forward() {
        let (batch, hidden, max_options, seq_len, marker_channels) = (2, 3, 4, 5, 9);
        let a = a_data(batch, hidden, seq_len);
        let positions = [0.0f32, 1.4, 2.5, 4.0];
        let mask = [1.0f32, 0.0, 1.0, 1.0];
        let b = b_data(batch, marker_channels, seq_len, &positions, &mask);
        let delta: Vec<f32> = (0..batch * hidden * max_options)
            .map(|i| (i as f32).sin())
            .collect();

        let out = forward(&a, &b, batch, hidden, max_options, seq_len, marker_channels).unwrap();
        let grad = backward(
            &b,
            &delta,
            batch,
            hidden,
            max_options,
            seq_len,
            marker_channels,
        )
        .unwrap();

        // <forward(a), delta> == <a, backward(delta)> for every a.
        let lhs: f32 = out.iter().zip(&delta).map(|(o, d)| o * d).sum();
        let rhs: f32 = a.iter().zip(&grad).map(|(x, g)| x * g).sum();
        assert!((lhs - rhs).abs() < 1e-4, "adjoint mismatch: {lhs} vs {rhs}");
    }

    #[test]
    fn shape_errors_are_reported() {
        let a = vec![0.0f32; 6];
        let b = b_data(1, 4, 3, &[1.0, 2.0], &[1.0, 1.0]);
        // Too few marker channels.
        assert!(forward(&a, &b, 1, 2, 3, 3, 4).is_err());
        // Wrong A length.
        assert!(forward(&a[..5], &b, 1, 2, 2, 3, 4).is_err());
        // Wrong B length.
        assert!(forward(&a, &b[..8], 1, 2, 2, 3, 4).is_err());
        // Bad backward shapes.
        assert!(backward(&b, &[0.0; 3], 1, 2, 2, 3, 4).is_err());
        assert!(backward(&b[..8], &[0.0; 4], 1, 2, 2, 3, 4).is_err());
    }
}
