//! `[deform_conv]` offset envelope and mask sigmoid: device vs an independently
//! written host reference of the source `sanitize`, `centered_lhtan` and
//! `stable_sigmoid` functions. Covers the clamp saturates, the per-batch
//! normalization switch (on/off), and non-finite inputs.
//!
//! Scope: these are the gradient-free envelope stages only. The per-batch
//! mean/stddev that decide `normalize` are the caller's responsibility, the
//! predictor columns are not built here, and the runtime still uses the CPU
//! forward.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

const TOL: f32 = 2e-6;

/// Tolerance for the *composed* forward. The device predictor runs through a
/// GEMM whose summation order differs from the reference's sequential loop, so
/// offsets differ in the last f32 bits and the bilinear sampling amplifies that
/// by the local image gradient. A single stage stays at `TOL`; the composition
/// needs a few ulp of headroom. Measured worst case in these fixtures is
/// ~2.4e-6 absolute on values ~0.16 (~1.5e-5 relative).
const TOL_COMPOSED: f32 = 1e-4;

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-10_000.0, 10_000.0)
    } else {
        0.0
    }
}

fn centered_lhtan(value: f32, max_offset: f32) -> f32 {
    let scaled = value / (2.0 * max_offset) + 0.5;
    let mapped = if scaled < 0.0 {
        0.001 * scaled
    } else if scaled > 1.0 {
        1.0 + 0.001 * (scaled - 1.0)
    } else {
        scaled
    };
    sanitize((2.0 * mapped - 1.0) * max_offset)
}

fn stable_sigmoid(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    let clipped = value.clamp(-80.0, 80.0);
    if clipped >= 0.0 {
        1.0 / (1.0 + (-clipped).exp())
    } else {
        let z = clipped.exp();
        z / (1.0 + z)
    }
}

fn assert_close_tol(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_close_with(name, got, want, tol)
}

fn assert_close_with(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{name}[{i}]: got {g}, want NaN");
            continue;
        }
        let err = (g - w).abs();
        assert!(
            err <= tol + tol * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{name}[{i}]: got {g}, want NaN");
            continue;
        }
        let err = (g - w).abs();
        assert!(
            err <= TOL + TOL * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

#[test]
fn envelope_normalize_off_matches_the_source_piecewise_map() {
    let backend = backend();
    let (planes, spatial, max_offset) = (2usize, 4usize, 6.0f32);
    let per_batch = planes * spatial;
    let batch = 2;
    // Deliberately includes both saturate sides, exact endpoints, NaN and inf.
    let raw: Vec<f32> = (0..batch * per_batch)
        .map(|i| match i % 8 {
            0 => -50.0,
            1 => 50.0,
            2 => -6.0,
            3 => 6.0,
            4 => 0.0,
            5 => f32::NAN,
            6 => f32::INFINITY,
            _ => (i as f32) * 0.375 - 2.0,
        })
        .collect();
    let mean = vec![0.0f32; batch];
    let inv_std = vec![1.0f32; batch];
    let normalize = vec![0.0f32; batch];
    let want: Vec<f32> = raw
        .iter()
        .map(|&v| centered_lhtan(sanitize(v), max_offset))
        .collect();

    let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let t_mean = backend.upload_f32(&[batch], &mean).unwrap();
    let t_inv = backend.upload_f32(&[batch], &inv_std).unwrap();
    let t_norm = backend.upload_f32(&[batch], &normalize).unwrap();
    let out = backend
        .deform_offset_envelope(
            &t_raw,
            &t_mean,
            &t_inv,
            &t_norm,
            planes,
            spatial,
            max_offset,
            cfg(),
        )
        .unwrap();
    assert_close("envelope/off", &backend.download_f32(&out).unwrap(), &want);
}

#[test]
fn envelope_normalize_on_uses_the_per_batch_stats() {
    let backend = backend();
    let (planes, spatial, max_offset) = (3usize, 5usize, 4.0f32);
    let per_batch = planes * spatial;
    let batch = 2;
    let raw: Vec<f32> = (0..batch * per_batch)
        .map(|i| ((i * 13 % 29) as f32) * 0.5 - 6.0)
        .collect();
    // Per-batch stats differ, so a batch-indexing bug changes the result.
    let mean = vec![-1.5f32, 2.25];
    let inv_std = vec![0.5f32, 2.0];
    let normalize = vec![1.0f32, 1.0];
    let want: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let b = i / per_batch;
            centered_lhtan(sanitize((sanitize(v) - mean[b]) * inv_std[b]), max_offset)
        })
        .collect();

    let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let t_mean = backend.upload_f32(&[batch], &mean).unwrap();
    let t_inv = backend.upload_f32(&[batch], &inv_std).unwrap();
    let t_norm = backend.upload_f32(&[batch], &normalize).unwrap();
    let out = backend
        .deform_offset_envelope(
            &t_raw,
            &t_mean,
            &t_inv,
            &t_norm,
            planes,
            spatial,
            max_offset,
            cfg(),
        )
        .unwrap();
    assert_close("envelope/on", &backend.download_f32(&out).unwrap(), &want);
}

#[test]
fn envelope_rejects_bad_operands() {
    let backend = backend();
    let t_raw = backend.upload_f32(&[8], &[0.0; 8]).unwrap();
    let t_ok = backend.upload_f32(&[2], &[0.0; 2]).unwrap();
    let t_short = backend.upload_f32(&[1], &[0.0]).unwrap();
    // Stats length must equal the derived batch.
    assert!(
        backend
            .deform_offset_envelope(&t_raw, &t_short, &t_ok, &t_ok, 2, 4, 1.0, cfg())
            .is_err()
    );
    // max_offset must be finite and positive.
    assert!(
        backend
            .deform_offset_envelope(&t_raw, &t_ok, &t_ok, &t_ok, 2, 4, 0.0, cfg())
            .is_err()
    );
    // Raw length must be a multiple of planes*spatial.
    assert!(
        backend
            .deform_offset_envelope(&t_raw, &t_ok, &t_ok, &t_ok, 3, 3, 1.0, cfg())
            .is_err()
    );
}

#[test]
fn mask_sigmoid_matches_the_source_stable_form() {
    let backend = backend();
    let raw: Vec<f32> = vec![
        0.0,
        1.0,
        -1.0,
        79.0,
        -79.0,
        80.0,
        -80.0,
        1e9,
        -1e9,
        20_000.0,
        -20_000.0,
        f32::NAN,
    ];
    let want: Vec<f32> = raw.iter().map(|&v| stable_sigmoid(sanitize(v))).collect();
    let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let out = backend.deform_mask_sigmoid(&t_raw, cfg()).unwrap();
    assert_close("mask", &backend.download_f32(&out).unwrap(), &want);
}

#[test]
fn mask_sigmoid_clamps_the_extremes_and_stays_symmetric() {
    let backend = backend();
    // The source clamps the logit to +/-80 and uses the symmetric exponential.
    // At exactly 80 the f32 `1/(1+exp(-80))` rounds to 1.0, so the property to
    // assert is clamping + symmetry + no NaN, not a strict (0,1) bound.
    let raw = vec![80.0f32, -80.0, 1e6, -1e6];
    let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let got = backend
        .download_f32(&backend.deform_mask_sigmoid(&t_raw, cfg()).unwrap())
        .unwrap();
    assert!(got.iter().all(|v| !v.is_nan()), "NaN in {got:?}");
    assert!(
        (0.0..=1.0).contains(&got[0]) && got[0] >= 0.5,
        "got {}",
        got[0]
    );
    assert!(
        (0.0..=1.0).contains(&got[1]) && got[1] <= 0.5,
        "got {}",
        got[1]
    );
    // Beyond the clamp the value is exactly the clamped value, not a new one.
    assert_eq!(got[2].to_bits(), got[0].to_bits(), "got {got:?}");
    assert_eq!(got[3].to_bits(), got[1].to_bits(), "got {got:?}");
    assert!((got[0] - (1.0 - got[1])).abs() < 1e-6, "got {got:?}");
}

// ---------------------------------------------------------------------------
// Per-batch offset statistics
// ---------------------------------------------------------------------------

/// Independent f64 reference of the source statistics rule.
fn stats(raw: &[f32], per_batch: usize, max_offset: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let batch = raw.len() / per_batch;
    let mut mean = Vec::with_capacity(batch);
    let mut inv_std = Vec::with_capacity(batch);
    let mut normalize = Vec::with_capacity(batch);
    for b in 0..batch {
        let slice = &raw[b * per_batch..(b + 1) * per_batch];
        let values: Vec<f64> = slice.iter().map(|&v| sanitize(v) as f64).collect();
        let n = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let sum_sq: f64 = values.iter().map(|v| v * v).sum();
        let m = sum / n;
        let variance = (sum_sq / n - m * m).max(0.0);
        let stddev = variance.sqrt();
        let norm = stddev > 1e-12 && (stddev > max_offset as f64 || m.abs() > max_offset as f64);
        mean.push(m as f32);
        inv_std.push(if norm { (1.0 / stddev) as f32 } else { 1.0 });
        normalize.push(if norm { 1.0 } else { 0.0 });
    }
    (mean, inv_std, normalize)
}

#[test]
fn offset_stats_match_the_source_and_are_per_batch() {
    let backend = backend();
    let (per_batch, batch, max_offset) = (6usize, 3, 3.0f32);
    // Batch 0 is tight (normalize off); batch 1 is wide (normalize on via the
    // spread gate); batch 2 has a large mean with small but nonzero spread, so
    // it normalizes through the mean gate only. A constant batch would NOT
    // normalize: the source requires `stddev > 1e-12` before either gate.
    let mut raw = vec![
        0.1, -0.1, 0.05, 0.0, -0.05, 0.02, // b0
        100.0, -100.0, 50.0, -50.0, 0.0, 25.0, // b1
        7.0, 7.01, 6.99, 7.02, 6.98, 7.0, // b2: large mean, tiny (nonzero) spread
    ];
    assert_eq!(raw.len(), per_batch * batch);
    let (want_mean, want_inv, want_norm) = stats(&raw, per_batch, max_offset);

    let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let (m, i_s, n) = backend
        .deform_offset_stats(&t_raw, per_batch, max_offset, cfg())
        .unwrap();
    assert_close("stats/mean", &backend.download_f32(&m).unwrap(), &want_mean);
    assert_close(
        "stats/inv_std",
        &backend.download_f32(&i_s).unwrap(),
        &want_inv,
    );
    assert_eq!(
        backend.download_f32(&n).unwrap(),
        want_norm,
        "normalize flags must match, and differ between batches"
    );
    assert_eq!(
        want_norm,
        vec![0.0, 1.0, 1.0],
        "fixture must span both gates"
    );

    // Sanitizing before the reduction matters: an out-of-range value must not
    // move the stats beyond the clamp.
    raw[1] = 1.0e9;
    let (want_mean2, _, _) = stats(&raw, per_batch, max_offset);
    let t_raw2 = backend.upload_f32(&[raw.len()], &raw).unwrap();
    let (m2, _, _) = backend
        .deform_offset_stats(&t_raw2, per_batch, max_offset, cfg())
        .unwrap();
    assert_close(
        "stats/mean-clamped",
        &backend.download_f32(&m2).unwrap(),
        &want_mean2,
    );
}

#[test]
fn offset_stats_reject_bad_geometry() {
    let backend = backend();
    let t_raw = backend.upload_f32(&[6], &[0.0; 6]).unwrap();
    assert!(backend.deform_offset_stats(&t_raw, 0, 1.0, cfg()).is_err());
    assert!(backend.deform_offset_stats(&t_raw, 4, 1.0, cfg()).is_err());
    assert!(backend.deform_offset_stats(&t_raw, 3, 0.0, cfg()).is_err());
}

// ---------------------------------------------------------------------------
// Offset / mask predictor (conv-based)
// ---------------------------------------------------------------------------

/// Independent reference of `predictor_columns` followed by the dense
/// predictor matmul and bias: `out[f, oy, ox] = b[f] + sum_{c,ky,kx} W[f,c,ky,kx]
/// * x[c, oy*sy - pad + ky, ox*sx - pad + kx]` with zero fill outside.
#[allow(clippy::too_many_arguments)]
fn predict(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    filters: usize,
    size: usize,
    stride: usize,
    padding: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * filters * out_h * out_w];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut sum = biases[f];
                    for c in 0..channels {
                        for ky in 0..size {
                            for kx in 0..size {
                                let y =
                                    oy as isize * stride as isize - padding as isize + ky as isize;
                                let x =
                                    ox as isize * stride as isize - padding as isize + kx as isize;
                                if y >= 0 && y < in_h as isize && x >= 0 && x < in_w as isize {
                                    let w = weights[((f * channels + c) * size + ky) * size + kx];
                                    let v = input[(b * channels + c) * in_h * in_w
                                        + y as usize * in_w
                                        + x as usize];
                                    sum += w * v;
                                }
                            }
                        }
                    }
                    out[(b * filters + f) * out_h * out_w + oy * out_w + ox] = sum;
                }
            }
        }
    }
    out
}

#[test]
fn predictor_matches_columns_times_weights_plus_bias() {
    let backend = backend();
    let (batch, channels, in_h, in_w) = (2usize, 3usize, 3usize, 4usize);
    let (size, stride, padding) = (3usize, 1usize, 1usize);
    let (out_h, out_w) = (3, 4);
    // 2*size*size offset filters, like the real layer.
    let filters = 2 * size * size;
    let input: Vec<f32> = (0..batch * channels * in_h * in_w)
        .map(|i| ((i * 11 % 23) as f32) * 0.25 - 2.0)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * size * size)
        .map(|i| ((i * 5 % 17) as f32) * 0.125 - 1.0)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| (i as f32) * 0.01 - 0.1).collect();
    let want = predict(
        &input, &weights, &biases, batch, channels, in_h, in_w, filters, size, stride, padding,
        out_h, out_w,
    );

    let t_in = backend
        .upload_f32(&[batch, channels, in_h, in_w], &input)
        .unwrap();
    let t_w = backend
        .upload_f32(&[filters, channels, size, size], &weights)
        .unwrap();
    let t_b = backend.upload_f32(&[filters], &biases).unwrap();
    let out = backend
        .deform_offset_predict(
            &t_in,
            &t_w,
            &t_b,
            channels,
            in_h,
            in_w,
            filters,
            size,
            stride,
            stride,
            padding,
            1,
            out_h,
            out_w,
            cfg(),
        )
        .unwrap();
    assert_close("predictor", &backend.download_f32(&out).unwrap(), &want);
}

#[test]
fn predictor_matches_with_stride_and_no_padding() {
    let backend = backend();
    let (batch, channels, in_h, in_w) = (1usize, 2usize, 5usize, 6usize);
    let (size, stride, padding) = (3usize, 2usize, 0usize);
    let (out_h, out_w) = (2, 2);
    let filters = size * size; // mask-shaped predictor
    let input: Vec<f32> = (0..batch * channels * in_h * in_w)
        .map(|i| (i as f32) * 0.0625 - 1.0)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * size * size)
        .map(|i| 0.5 - ((i % 7) as f32) * 0.125)
        .collect();
    let biases = vec![0.25f32; filters];
    let want = predict(
        &input, &weights, &biases, batch, channels, in_h, in_w, filters, size, stride, padding,
        out_h, out_w,
    );
    let t_in = backend
        .upload_f32(&[batch, channels, in_h, in_w], &input)
        .unwrap();
    let t_w = backend
        .upload_f32(&[filters, channels, size, size], &weights)
        .unwrap();
    let t_b = backend.upload_f32(&[filters], &biases).unwrap();
    let out = backend
        .deform_offset_predict(
            &t_in,
            &t_w,
            &t_b,
            channels,
            in_h,
            in_w,
            filters,
            size,
            stride,
            stride,
            padding,
            1,
            out_h,
            out_w,
            cfg(),
        )
        .unwrap();
    assert_close(
        "predictor/strided",
        &backend.download_f32(&out).unwrap(),
        &want,
    );
}

#[test]
fn predictor_supports_source_extent_dilation_and_rejects_bad_operands() {
    let backend = backend();
    let (batch, channels, in_h, in_w, filters, size) =
        (1usize, 2usize, 3usize, 4usize, 9usize, 3usize);
    let (stride, padding, dilation) = (1usize, 1usize, 2usize);
    let (out_h, out_w) = (1usize, 2usize);
    let input: Vec<f32> = (0..batch * channels * in_h * in_w)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.13)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * size * size)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.007)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| i as f32 * 0.03 - 0.1).collect();
    let t_in = backend
        .upload_f32(&[batch, channels, in_h, in_w], &input)
        .unwrap();
    let t_w = backend
        .upload_f32(&[filters, channels, size, size], &weights)
        .unwrap();
    let t_b = backend.upload_f32(&[filters], &biases).unwrap();
    let got = backend
        .deform_offset_predict(
            &t_in,
            &t_w,
            &t_b,
            channels,
            in_h,
            in_w,
            filters,
            size,
            stride,
            stride,
            padding,
            dilation,
            out_h,
            out_w,
            cfg(),
        )
        .unwrap();
    let mut want = vec![0.0; batch * filters * out_h * out_w];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut value = biases[f];
                    for c in 0..channels {
                        for ky in 0..size {
                            for kx in 0..size {
                                let y = (oy * stride) as isize - (padding * dilation) as isize
                                    + (ky * dilation) as isize;
                                let x = (ox * stride) as isize - (padding * dilation) as isize
                                    + (kx * dilation) as isize;
                                if y >= 0 && y < in_h as isize && x >= 0 && x < in_w as isize {
                                    let xi = ((b * channels + c) * in_h + y as usize) * in_w
                                        + x as usize;
                                    let wi = ((f * channels + c) * size + ky) * size + kx;
                                    value += input[xi] * weights[wi];
                                }
                            }
                        }
                    }
                    want[((b * filters + f) * out_h + oy) * out_w + ox] = value;
                }
            }
        }
    }
    assert_close(
        "dilated predictor",
        &backend.download_f32(&got).unwrap(),
        &want,
    );
    // Wrong bias length.
    assert!(
        backend
            .deform_offset_predict(
                &t_in,
                &t_w,
                &backend
                    .upload_f32(&[filters - 1], &biases[..filters - 1])
                    .unwrap(),
                channels,
                in_h,
                in_w,
                filters,
                size,
                stride,
                stride,
                padding,
                dilation,
                out_h,
                out_w,
                cfg(),
            )
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Composed native forward (all five stages)
// ---------------------------------------------------------------------------

/// Deformable accumulation reference (same contract as the accumulation suite).
#[allow(clippy::too_many_arguments)]
fn accumulate_ref(
    input: &[f32],
    offsets: &[f32],
    masks: Option<&[f32]>,
    weights: &[f32],
    batch: usize,
    channels: usize,
    filters: usize,
    groups: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let plane = out_h * out_w;
    let kernels = size * size;
    let cpg = channels / groups;
    let fpg = filters / groups;
    let mut out = vec![0.0f32; batch * filters * plane];
    for b in 0..batch {
        for f in 0..filters {
            let group = f / fpg;
            for pos in 0..plane {
                let (oy, ox) = (pos / out_w, pos % out_w);
                let mut sum = 0.0f32;
                for lc in 0..cpg {
                    let c = group * cpg + lc;
                    let base = (b * channels + c) * in_h * in_w;
                    for k in 0..kernels {
                        let (ky, kx) = (k / size, k % size);
                        let dy = offsets[b * 2 * kernels * plane + (2 * k) * plane + pos];
                        let dx = offsets[b * 2 * kernels * plane + (2 * k + 1) * plane + pos];
                        let y = (oy * stride) as f32 - (padding as f32) + ky as f32 + dy;
                        let x = (ox * stride) as f32 - (padding as f32) + kx as f32 + dx;
                        let m = masks.map_or(1.0, |m| m[b * kernels * plane + k * plane + pos]);
                        let w = weights[(f * cpg + lc) * kernels + k];
                        // bilinear with zero padding + far-outside early-out
                        let v = if y < -1.0 || y > in_h as f32 || x < -1.0 || x > in_w as f32 {
                            0.0
                        } else {
                            let (y0, x0) = (y.floor(), x.floor());
                            let (ly, lx) = (y - y0, x - x0);
                            let mut acc = 0.0f32;
                            for (dyi, wy) in [(0.0f32, 1.0 - ly), (1.0, ly)] {
                                for (dxi, wx) in [(0.0f32, 1.0 - lx), (1.0, lx)] {
                                    let iy = y0 + dyi;
                                    let ix = x0 + dxi;
                                    if iy >= 0.0
                                        && iy < in_h as f32
                                        && ix >= 0.0
                                        && ix < in_w as f32
                                    {
                                        acc += wy
                                            * wx
                                            * input[base + iy as usize * in_w + ix as usize];
                                    }
                                }
                            }
                            acc
                        };
                        sum += w * v * m;
                    }
                }
                out[(b * filters + f) * plane + pos] = if sum.is_finite() {
                    sum.clamp(-10_000.0, 10_000.0)
                } else {
                    0.0
                };
            }
        }
    }
    out
}

struct Composed {
    input: Vec<f32>,
    main_w: Vec<f32>,
    off_w: Vec<f32>,
    off_b: Vec<f32>,
    mask_w: Vec<f32>,
    mask_b: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn composed_reference(
    f: &Composed,
    batch: usize,
    channels: usize,
    filters: usize,
    groups: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride: usize,
    padding: usize,
    use_mask: bool,
) -> Vec<f32> {
    let kernels = size * size;
    let plane = out_h * out_w;
    let max_offset = size as f32 * 2.0;
    let input_s: Vec<f32> = f.input.iter().map(|&v| sanitize(v)).collect();
    let off_raw = predict(
        &input_s,
        &f.off_w,
        &f.off_b,
        batch,
        channels,
        in_h,
        in_w,
        2 * kernels,
        size,
        stride,
        padding,
        out_h,
        out_w,
    );
    let (mean, inv, norm) = stats(&off_raw, 2 * kernels * plane, max_offset);
    let offsets: Vec<f32> = off_raw
        .iter()
        .enumerate()
        .map(|(i, &raw)| {
            let b = i / (2 * kernels * plane);
            let v = sanitize(raw);
            let v = if norm[b] != 0.0 {
                sanitize((v - mean[b]) * inv[b])
            } else {
                v
            };
            centered_lhtan(v, max_offset)
        })
        .collect();
    let masks = if use_mask {
        Some(
            predict(
                &input_s, &f.mask_w, &f.mask_b, batch, channels, in_h, in_w, kernels, size, stride,
                padding, out_h, out_w,
            )
            .iter()
            .map(|&v| stable_sigmoid(sanitize(v)))
            .collect::<Vec<f32>>(),
        )
    } else {
        None
    };
    accumulate_ref(
        &input_s,
        &offsets,
        masks.as_deref(),
        &f.main_w,
        batch,
        channels,
        filters,
        groups,
        in_h,
        in_w,
        out_h,
        out_w,
        size,
        stride,
        padding,
    )
}

fn composed_backend_run(
    f: &Composed,
    batch: usize,
    channels: usize,
    filters: usize,
    groups: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride: usize,
    padding: usize,
    use_mask: bool,
) -> Vec<f32> {
    let backend = backend();
    let kernels = size * size;
    let t_in = backend
        .upload_f32(&[batch, channels, in_h, in_w], &f.input)
        .unwrap();
    let t_main = backend
        .upload_f32(&[filters * (channels / groups) * kernels], &f.main_w)
        .unwrap();
    let t_off_w = backend
        .upload_f32(&[2 * kernels * channels * kernels], &f.off_w)
        .unwrap();
    let t_off_b = backend.upload_f32(&[2 * kernels], &f.off_b).unwrap();
    let t_mask_w = backend
        .upload_f32(&[kernels * channels * kernels], &f.mask_w)
        .unwrap();
    let t_mask_b = backend.upload_f32(&[kernels], &f.mask_b).unwrap();
    let masks = if use_mask {
        Some((&t_mask_w, &t_mask_b))
    } else {
        None
    };
    let out = backend
        .deform_conv_forward_native(
            &t_in,
            &t_main,
            &t_off_w,
            &t_off_b,
            masks,
            channels,
            filters,
            groups,
            in_h,
            in_w,
            out_h,
            out_w,
            size,
            stride,
            stride,
            padding,
            1,
            cfg(),
        )
        .unwrap();
    backend.download_f32(&out).unwrap()
}

#[test]
fn composed_forward_matches_the_stage_by_stage_reference() {
    let (batch, channels, filters, groups) = (2usize, 2usize, 3usize, 1usize);
    let (in_h, in_w, out_h, out_w) = (3usize, 4usize, 3usize, 4usize);
    let (size, stride, padding) = (3usize, 1usize, 1usize);
    let kernels = size * size;
    let f = Composed {
        input: (0..batch * channels * in_h * in_w)
            .map(|i| ((i * 7 % 19) as f32) * 0.5 - 3.0)
            .collect(),
        main_w: (0..filters * channels * kernels)
            .map(|i| 0.5 - ((i * 3 % 11) as f32) * 0.125)
            .collect(),
        off_w: (0..2 * kernels * channels * kernels)
            .map(|i| ((i * 13 % 17) as f32) * 0.25 - 2.0)
            .collect(),
        off_b: (0..2 * kernels).map(|i| (i as f32) * 0.05 - 0.4).collect(),
        mask_w: (0..kernels * channels * kernels)
            .map(|i| ((i * 5 % 7) as f32) * 0.5 - 1.5)
            .collect(),
        mask_b: (0..kernels).map(|i| 0.1 * i as f32).collect(),
    };
    for use_mask in [false, true] {
        let want = composed_reference(
            &f, batch, channels, filters, groups, in_h, in_w, out_h, out_w, size, stride, padding,
            use_mask,
        );
        let got = composed_backend_run(
            &f, batch, channels, filters, groups, in_h, in_w, out_h, out_w, size, stride, padding,
            use_mask,
        );
        assert_close_with(
            &format!("composed/mask={use_mask}"),
            &got,
            &want,
            TOL_COMPOSED,
        );
    }
}

#[test]
fn composed_forward_with_groups_matches() {
    let (batch, channels, filters, groups) = (1usize, 4usize, 4usize, 2usize);
    let (in_h, in_w, out_h, out_w) = (2usize, 3usize, 2usize, 3usize);
    let (size, stride, padding) = (3usize, 1usize, 1usize);
    let kernels = size * size;
    let f = Composed {
        input: (0..batch * channels * in_h * in_w)
            .map(|i| (i as f32) * 0.25 - 2.0)
            .collect(),
        main_w: (0..filters * (channels / groups) * kernels)
            .map(|i| 0.75 - ((i % 5) as f32) * 0.25)
            .collect(),
        off_w: (0..2 * kernels * channels * kernels)
            .map(|i| ((i * 3 % 13) as f32) * 0.125 - 1.0)
            .collect(),
        off_b: vec![0.05f32; 2 * kernels],
        mask_w: vec![0.0f32; kernels * channels * kernels],
        mask_b: vec![0.0f32; kernels],
    };
    let want = composed_reference(
        &f, batch, channels, filters, groups, in_h, in_w, out_h, out_w, size, stride, padding,
        false,
    );
    let got = composed_backend_run(
        &f, batch, channels, filters, groups, in_h, in_w, out_h, out_w, size, stride, padding,
        false,
    );
    assert_close_with("composed/groups", &got, &want, TOL_COMPOSED);
}

// ---------------------------------------------------------------------------
// Envelope / mask slope adjoints (device)
// ---------------------------------------------------------------------------

#[test]
fn envelope_pre_backward_matches_the_host_reference_both_modes() {
    let backend = backend();
    let (planes, spatial, max_offset) = (2usize, 4usize, 3.0f32);
    let per_batch = planes * spatial;
    let batch = 2;
    let raw: Vec<f32> = (0..batch * per_batch)
        .map(|i| ((i * 9 % 17) as f32) * 0.5 - 4.0 + 0.17)
        .collect();
    let mean = vec![-0.5f32, 1.25];
    let inv_std = vec![0.75f32, 1.5];
    let seed: Vec<f32> = (0..raw.len())
        .map(|i| 0.2 + ((i % 5) as f32) * 0.15)
        .collect();
    for normalize in [[0.0f32, 0.0], [1.0, 1.0]] {
        let want = lightnet_cuda::host_ops::deform_conv::offset_envelope_pre_backward(
            &raw, &mean, &inv_std, &normalize, &seed, planes, spatial, max_offset,
        )
        .unwrap();
        let t_raw = backend.upload_f32(&[raw.len()], &raw).unwrap();
        let t_mean = backend.upload_f32(&[batch], &mean).unwrap();
        let t_inv = backend.upload_f32(&[batch], &inv_std).unwrap();
        let t_norm = backend.upload_f32(&[batch], &normalize).unwrap();
        let t_seed = backend.upload_f32(&[seed.len()], &seed).unwrap();
        let out = backend
            .deform_envelope_pre_backward(
                &t_raw,
                &t_mean,
                &t_inv,
                &t_norm,
                &t_seed,
                planes,
                spatial,
                max_offset,
                cfg(),
            )
            .unwrap();
        assert_close_tol(
            &format!("env_pre_bwd/norm={}", normalize[0]),
            &backend.download_f32(&out).unwrap(),
            &want,
            2e-6,
        );
    }
}

#[test]
fn mask_sigmoid_backward_matches_the_host_reference() {
    let backend = backend();
    let logits = vec![
        0.0f32,
        1.5,
        -2.25,
        20.0,
        -20.0,
        79.0,
        -79.0,
        80.0,
        -80.0,
        1e6,
        f32::NAN,
    ];
    let seed: Vec<f32> = (0..logits.len()).map(|i| 0.5 + i as f32 * 0.125).collect();
    let want = lightnet_cuda::host_ops::deform_conv::mask_sigmoid_backward(&logits, &seed).unwrap();
    let t_logits = backend.upload_f32(&[logits.len()], &logits).unwrap();
    let t_seed = backend.upload_f32(&[seed.len()], &seed).unwrap();
    let out = backend
        .deform_mask_sigmoid_backward(&t_logits, &t_seed, cfg())
        .unwrap();
    assert_close_tol(
        "mask_sigmoid_bwd",
        &backend.download_f32(&out).unwrap(),
        &want,
        2e-6,
    );
}
