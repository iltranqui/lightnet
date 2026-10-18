//! Native `[deform_conv]` accumulation stage: device vs an independently
//! written host reference (the reference below is derived from the source loop
//! order, not from `darknet_core`), covering groups, stride/padding/dilation,
//! optional sigmoid masks, displaced samples past the feature-map edge, and the
//! source `sanitize` clamp.
//!
//! Scope: this tests the deformable accumulation only. The predictor columns,
//! conditional per-batch offset normalization and lhtan/sigmoid envelopes are
//! not part of the kernel, and the runtime still uses the CPU forward.

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

const TOL: f32 = 2e-5;

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

#[derive(Clone, Copy)]
struct Geom {
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
    dilation: usize,
}

impl Geom {
    fn channels_per_group(&self) -> usize {
        self.channels / self.groups
    }
    fn filters_per_group(&self) -> usize {
        self.filters / self.groups
    }
    fn kernels(&self) -> usize {
        self.size * self.size
    }
    fn plane(&self) -> usize {
        self.out_h * self.out_w
    }
}

/// Independent bilinear sample, written from the documented contract.
fn sample(input: &[f32], base: usize, g: &Geom, y: f32, x: f32) -> f32 {
    if y < -1.0 || y > g.in_h as f32 || x < -1.0 || x > g.in_w as f32 {
        return 0.0;
    }
    let y0 = y.floor();
    let x0 = x.floor();
    let (ly, lx) = (y - y0, x - x0);
    let mut value = 0.0f32;
    for (dy, wy) in [(0.0f32, 1.0 - ly), (1.0, ly)] {
        for (dx, wx) in [(0.0f32, 1.0 - lx), (1.0, lx)] {
            let iy = y0 + dy;
            let ix = x0 + dx;
            if iy >= 0.0 && iy < g.in_h as f32 && ix >= 0.0 && ix < g.in_w as f32 {
                value += wy * wx * input[base + iy as usize * g.in_w + ix as usize];
            }
        }
    }
    value
}

fn reference(
    input: &[f32],
    offsets: &[f32],
    masks: &[f32],
    weights: &[f32],
    g: &Geom,
    use_mask: bool,
) -> Vec<f32> {
    let plane = g.plane();
    let mut out = vec![0.0f32; g.batch * g.filters * plane];
    for b in 0..g.batch {
        for filter in 0..g.filters {
            let group = filter / g.filters_per_group();
            for pos in 0..plane {
                let (oy, ox) = (pos / g.out_w, pos % g.out_w);
                let mut sum = 0.0f32;
                for lc in 0..g.channels_per_group() {
                    let channel = group * g.channels_per_group() + lc;
                    let base = (b * g.channels + channel) * g.in_h * g.in_w;
                    for ky in 0..g.size {
                        for kx in 0..g.size {
                            let k = ky * g.size + kx;
                            let off = b * 2 * g.kernels() * plane + pos;
                            let dy = offsets[off + 2 * k * plane];
                            let dx = offsets[off + (2 * k + 1) * plane];
                            let y = (oy * g.stride) as f32 - (g.padding * g.dilation) as f32
                                + (ky * g.dilation) as f32
                                + dy;
                            let x = (ox * g.stride) as f32 - (g.padding * g.dilation) as f32
                                + (kx * g.dilation) as f32
                                + dx;
                            let m = if use_mask {
                                masks[b * g.kernels() * plane + k * plane + pos]
                            } else {
                                1.0
                            };
                            let w =
                                weights[(filter * g.channels_per_group() + lc) * g.kernels() + k];
                            sum += w * sample(input, base, g, y, x) * m;
                        }
                    }
                }
                out[(b * g.filters + filter) * plane + pos] = if sum.is_finite() {
                    sum.clamp(-10_000.0, 10_000.0)
                } else {
                    0.0
                };
            }
        }
    }
    out
}

struct Fixture {
    input: Vec<f32>,
    offsets: Vec<f32>,
    masks: Vec<f32>,
    weights: Vec<f32>,
}

/// Deterministic operands with fractional offsets so the bilinear path and the
/// out-of-range zero padding are both exercised.
fn fixture(g: &Geom, displace: bool, use_mask: bool) -> Fixture {
    let input: Vec<f32> = (0..g.batch * g.channels * g.in_h * g.in_w)
        .map(|i| ((i * 7 % 19) as f32) * 0.125 - 1.0)
        .collect();
    let offset_len = g.batch * 2 * g.kernels() * g.plane();
    let offsets: Vec<f32> = (0..offset_len)
        .map(|i| {
            if !displace {
                0.0
            } else {
                // Alternate large displacements so some taps fall outside.
                let step = (i % 5) as f32;
                0.25 + step * 0.4 - 0.75
            }
        })
        .collect();
    let masks: Vec<f32> = if use_mask {
        (0..g.batch * g.kernels() * g.plane())
            .map(|i| 0.2 + (i % 4) as f32 * 0.2)
            .collect()
    } else {
        Vec::new()
    };
    let weights: Vec<f32> = (0..g.filters * g.channels_per_group() * g.kernels())
        .map(|i| 0.5 - (i % 3) as f32 * 0.25)
        .collect();
    Fixture {
        input,
        offsets,
        masks,
        weights,
    }
}

fn run(g: &Geom, displace: bool, use_mask: bool) -> (Vec<f32>, Vec<f32>) {
    let backend = backend();
    let f = fixture(g, displace, use_mask);
    let want = reference(&f.input, &f.offsets, &f.masks, &f.weights, g, use_mask);
    let t_in = backend.upload_f32(&[f.input.len()], &f.input).unwrap();
    let t_off = backend.upload_f32(&[f.offsets.len()], &f.offsets).unwrap();
    let t_masks = backend
        .upload_f32(
            &[f.masks.len().max(1)],
            &if f.masks.is_empty() {
                vec![0.0]
            } else {
                f.masks.clone()
            },
        )
        .unwrap();
    let t_w = backend.upload_f32(&[f.weights.len()], &f.weights).unwrap();
    let out = backend
        .deform_conv_accumulate(
            &t_in,
            &t_off,
            &t_masks,
            &t_w,
            g.channels,
            g.filters,
            g.channels_per_group(),
            g.filters_per_group(),
            g.in_h,
            g.in_w,
            g.out_h,
            g.out_w,
            g.size,
            g.stride,
            g.stride,
            g.padding,
            g.dilation,
            use_mask,
            cfg(),
        )
        .unwrap();
    (backend.download_f32(&out).unwrap(), want)
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL + TOL * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

#[test]
fn zero_offset_matches_a_plain_convolution() {
    // 1x1 channel, 3x3 kernel, pad 1, stride 1, no masks: this must equal a
    // normal 3x3 convolution, so a displacement bug cannot hide.
    let g = Geom {
        batch: 1,
        channels: 1,
        filters: 1,
        groups: 1,
        in_h: 2,
        in_w: 3,
        out_h: 2,
        out_w: 3,
        size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    };
    let (got, want) = run(&g, false, false);
    assert_close("zero-offset conv", &got, &want);
    // Hand-computed 3x3 dot product for output (oy=1, ox=1) directly from the
    // fixture formulas: base = (1-1+ky, 1-1+kx), and only ky < 2 is inside the
    // 2-row input. weights[k] = 0.5 - (k%3)*0.25, input[i] = ((i*7)%19)*0.125 - 1.
    //   k0 -0.5000, k1 -0.03125, k2 0, k3 -0.375, k4 +0.03125, k5 0
    let expected: f32 = -0.875;
    assert!(
        (got[4] - expected).abs() < 1e-6,
        "got {} want {expected}",
        got[4]
    );
}

#[test]
fn grouped_displaced_and_masked_accumulation_matches() {
    let g = Geom {
        batch: 2,
        channels: 4,
        filters: 4,
        groups: 2,
        in_h: 3,
        in_w: 4,
        out_h: 3,
        out_w: 4,
        size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    };
    let (got, want) = run(&g, true, true);
    assert_close("grouped+masked", &got, &want);
}

#[test]
fn strided_dilated_accumulation_matches() {
    // `deform_conv`'s own extent includes the dilated kernel extent:
    //   oh = (H + 2*pad - dilation*(size-1) - 1) / stride + 1
    // so 5x6 with size 3, stride 2, pad 1, dilation 2 gives exactly 2x2.
    // (A `[convolutional]`-style extent that ignores dilation would not, which
    // is why the predictor columns cannot simply reuse `conv2d_f32` here.)
    let g = Geom {
        batch: 1,
        channels: 2,
        filters: 2,
        groups: 1,
        in_h: 5,
        in_w: 6,
        out_h: 2,
        out_w: 2,
        size: 3,
        stride: 2,
        padding: 1,
        dilation: 2,
    };
    let oh = (g.in_h + 2 * g.padding - g.dilation * (g.size - 1) - 1) / g.stride + 1;
    let ow = (g.in_w + 2 * g.padding - g.dilation * (g.size - 1) - 1) / g.stride + 1;
    assert_eq!((oh, ow), (g.out_h, g.out_w), "fixture must be self-consistent");
    let (got, want) = run(&g, true, false);
    assert_close("strided+dilated", &got, &want);
}

#[test]
fn samples_pushed_past_the_edge_are_zero() {
    // Large offsets on a 1x1 feature map: every tap lands far outside, so the
    // accumulation is exactly zero, not a clamped or wrapped sample.
    let backend = backend();
    let input = vec![5.0f32];
    let offsets = vec![9.0f32; 2 * 9];
    let weights = vec![1.0f32; 9];
    let t_in = backend.upload_f32(&[1], &input).unwrap();
    let t_off = backend.upload_f32(&[18], &offsets).unwrap();
    let t_m = backend.upload_f32(&[1], &[0.0]).unwrap();
    let t_w = backend.upload_f32(&[9], &weights).unwrap();
    let out = backend
        .deform_conv_accumulate(
            &t_in,
            &t_off,
            &t_m,
            &t_w,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            3,
            1,
            1,
            1,
            1,
            false,
            cfg(),
        )
        .unwrap();
    assert_eq!(backend.download_f32(&out).unwrap(), vec![0.0]);
}

#[test]
fn mismatched_operands_are_rejected() {
    let backend = backend();
    let g = Geom {
        batch: 1,
        channels: 2,
        filters: 2,
        groups: 1,
        in_h: 2,
        in_w: 2,
        out_h: 2,
        out_w: 2,
        size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    };
    let f = fixture(&g, false, false);
    let t_in = backend.upload_f32(&[f.input.len()], &f.input).unwrap();
    let t_off_short = backend
        .upload_f32(&[f.offsets.len() - 1], &f.offsets[1..])
        .unwrap();
    let t_off = backend.upload_f32(&[f.offsets.len()], &f.offsets).unwrap();
    let t_w = backend.upload_f32(&[f.weights.len()], &f.weights).unwrap();
    let t_m = backend.upload_f32(&[1], &[1.0]).unwrap();
    // Short offsets.
    assert!(
        backend
            .deform_conv_accumulate(
                &t_in,
                &t_off_short,
                &t_m,
                &t_w,
                g.channels,
                g.filters,
                g.channels_per_group(),
                g.filters_per_group(),
                g.in_h,
                g.in_w,
                g.out_h,
                g.out_w,
                g.size,
                g.stride,
                g.stride,
                g.padding,
                g.dilation,
                false,
                cfg(),
            )
            .is_err()
    );
    // Mask requested but not supplied at the right length.
    assert!(
        backend
            .deform_conv_accumulate(
                &t_in,
                &t_off,
                &t_m,
                &t_w,
                g.channels,
                g.filters,
                g.channels_per_group(),
                g.filters_per_group(),
                g.in_h,
                g.in_w,
                g.out_h,
                g.out_w,
                g.size,
                g.stride,
                g.stride,
                g.padding,
                g.dilation,
                true,
                cfg(),
            )
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Accumulation adjoint
// ---------------------------------------------------------------------------

fn assert_close_tol(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= tol + tol * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

fn accumulate_backward_case(use_mask: bool) {
    let backend = backend();
    let g = Geom {
        batch: 2,
        channels: 2,
        filters: 2,
        groups: 1,
        in_h: 2,
        in_w: 3,
        out_h: 2,
        out_w: 3,
        size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
    };
    let f = fixture(&g, true, use_mask);
    let seed: Vec<f32> = (0..g.batch * g.filters * g.plane())
        .map(|i| 0.3 + ((i % 5) as f32) * 0.1)
        .collect();
    // Host reference: the FD-verified adjoint in `darknet_cuda`.
    let geometry = darknet_cuda::host_ops::deform_conv::DeformGeometry {
        batch: g.batch,
        channels: g.channels,
        filters: g.filters,
        channels_per_group: g.channels_per_group(),
        filters_per_group: g.filters_per_group(),
        input_height: g.in_h,
        input_width: g.in_w,
        output_height: g.out_h,
        output_width: g.out_w,
        size: g.size,
        stride_x: g.stride,
        stride_y: g.stride,
        padding: g.padding,
        dilation: g.dilation,
    };
    let want = darknet_cuda::host_ops::deform_conv::accumulate_backward(
        &f.input,
        &f.offsets,
        if use_mask { Some(f.masks.as_slice()) } else { None },
        &f.weights,
        &seed,
        &geometry,
    )
    .unwrap();

    let t_in = backend.upload_f32(&[f.input.len()], &f.input).unwrap();
    let t_off = backend.upload_f32(&[f.offsets.len()], &f.offsets).unwrap();
    let t_masks = backend
        .upload_f32(
            &[f.masks.len().max(1)],
            &if f.masks.is_empty() { vec![0.0] } else { f.masks.clone() },
        )
        .unwrap();
    let t_w = backend.upload_f32(&[f.weights.len()], &f.weights).unwrap();
    let t_seed = backend.upload_f32(&[seed.len()], &seed).unwrap();
    let (d_in, d_off, d_masks, d_w) = backend
        .deform_conv_accumulate_backward(
            &t_in, &t_off, &t_masks, &t_w, &t_seed, geometry, use_mask, cfg(),
        )
        .unwrap();
    // Atomic scatter: compare with accumulation-order tolerance.
    assert_close_tol(
        &format!("accum_bwd/d_input mask={use_mask}"),
        &backend.download_f32(&d_in).unwrap(),
        &want.d_input,
        2e-5,
    );
    assert_close_tol(
        &format!("accum_bwd/d_offsets mask={use_mask}"),
        &backend.download_f32(&d_off).unwrap(),
        &want.d_offsets,
        2e-5,
    );
    assert_close_tol(
        &format!("accum_bwd/d_weights mask={use_mask}"),
        &backend.download_f32(&d_w).unwrap(),
        &want.d_weights,
        2e-5,
    );
    if use_mask {
        assert_close_tol(
            &format!("accum_bwd/d_masks mask={use_mask}"),
            &backend.download_f32(&d_masks).unwrap(),
            &want.d_masks,
            2e-5,
        );
    } else {
        assert!(want.d_masks.is_empty());
    }
}

#[test]
fn accumulation_backward_matches_the_host_reference_without_masks() {
    accumulate_backward_case(false);
}

#[test]
fn accumulation_backward_matches_the_host_reference_with_masks() {
    accumulate_backward_case(true);
}

// ---------------------------------------------------------------------------
// Predictor / column-gather adjoint
// ---------------------------------------------------------------------------

#[test]
fn predictor_backward_matches_the_host_reference() {
    let backend = backend();
    let (batch, channels, filters) = (2usize, 3usize, 4usize);
    let (in_h, in_w) = (3usize, 4usize);
    let (size, stride, padding, dilation) = (3usize, 1usize, 1usize, 1usize);
    let (out_h, out_w) = (3usize, 4usize);
    let kernels = size * size;
    let column_count = channels * kernels;
    let plane = out_h * out_w;
    let input: Vec<f32> = (0..batch * channels * in_h * in_w)
        .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
        .collect();
    let weights: Vec<f32> = (0..filters * column_count)
        .map(|i| ((i * 5 % 11) as f32) * 0.125 - 0.6)
        .collect();
    let seed: Vec<f32> = (0..batch * filters * plane)
        .map(|i| 0.2 + ((i % 5) as f32) * 0.1)
        .collect();
    let columns = darknet_cuda::host_ops::deform_conv::predictor_columns(
        &input, batch, channels, in_h, in_w, size, stride, padding, dilation, out_h, out_w,
    )
    .unwrap();
    let want = darknet_cuda::host_ops::deform_conv::predictor_backward(
        &input, &columns, &weights, &seed, batch, channels, filters, in_h, in_w, out_h, out_w,
        size, stride, padding, dilation,
    )
    .unwrap();

    // The device kernel takes the sanitized input and rebuilds each column
    // value itself, so no columns buffer is uploaded.
    let t_input = backend.upload_f32(&[input.len()], &input).unwrap();
    let t_w = backend.upload_f32(&[weights.len()], &weights).unwrap();
    let t_seed = backend.upload_f32(&[seed.len()], &seed).unwrap();
    let (d_in, d_w, d_b) = backend
        .deform_predictor_backward(
            &t_input, &t_w, &t_seed, channels, filters, in_h, in_w, out_h, out_w, size, stride,
            stride, padding, dilation, batch, cfg(),
        )
        .unwrap();
    assert_close_tol(
        "predictor_bwd/d_input",
        &backend.download_f32(&d_in).unwrap(),
        &want.d_input,
        2e-5,
    );
    assert_close_tol(
        "predictor_bwd/d_weights",
        &backend.download_f32(&d_w).unwrap(),
        &want.d_weights,
        2e-5,
    );
    assert_close_tol(
        "predictor_bwd/d_biases",
        &backend.download_f32(&d_b).unwrap(),
        &want.d_biases,
        2e-5,
    );
}
