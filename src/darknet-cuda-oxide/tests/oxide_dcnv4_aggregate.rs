//! Device DCNv4 aggregation vs the FD-verified host reference, across the four
//! `softmax`/`remove_center` combinations and two group configurations.
//!
//! The fixture keeps every sample near a cell centre: a finite difference that
//! pushes `sx` across an integer compares a one-sided derivative with a central
//! one and reports a false mismatch (see `tasks/rust/lessons.md`).

use darknet_cuda::host_ops::dcnv4 as host;
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

const TOL: f32 = 2e-5;

fn backend() -> OxideBackend {
    OxideBackend::new().expect("oxide backend")
}

fn geometry(
    channels: usize,
    groups: usize,
    softmax: bool,
    remove_center: bool,
) -> host::Dcnv4Geometry {
    host::Dcnv4Geometry {
        batch: 2,
        channels,
        groups,
        input_height: 3,
        input_width: 4,
        output_height: 2,
        output_width: 3,
        size: 3,
        stride_x: 1,
        stride_y: 1,
        padding: 1,
        dilation: 1,
        softmax,
        remove_center,
    }
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
fn device_aggregation_matches_the_host_reference() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    let offset_scale = 0.75f32;
    for (channels, groups) in [(2usize, 1usize), (4, 2)] {
        for softmax in [false, true] {
            for remove_center in [false, true] {
                let g = geometry(channels, groups, softmax, remove_center);
                let input: Vec<f32> = (0..g.batch * g.channels * g.input_height * g.input_width)
                    .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
                    .collect();
                let offsets: Vec<f32> = (0..g.batch * g.spatial() * g.offset_dim())
                    .map(|i| 0.5 + ((i * 3 % 5) as f32) * 0.02)
                    .collect();
                let want = host::aggregate(&input, &offsets, &g, offset_scale).unwrap();
                let t_in = backend.upload_f32(&[input.len()], &input).unwrap();
                let t_off = backend.upload_f32(&[offsets.len()], &offsets).unwrap();
                let got = backend
                    .dcnv4_aggregate_native(&t_in, &t_off, g, offset_scale, cfg)
                    .unwrap();
                assert_close(
                    &format!("dcnv4 sm={softmax} rc={remove_center} c={channels} g={groups}"),
                    &backend.download_f32(&got).unwrap(),
                    &want,
                );
            }
        }
    }
}

#[test]
fn device_aggregation_rejects_oversized_kernels_and_bad_operands() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    let mut g = geometry(2, 1, false, false);
    g.size = 7; // 49 > MAX_POINTS
    let input = vec![0.0f32; g.batch * g.channels * g.input_height * g.input_width];
    let offsets = vec![0.0f32; g.batch * g.spatial() * g.offset_dim()];
    let t_in = backend.upload_f32(&[input.len()], &input).unwrap();
    let t_off = backend.upload_f32(&[offsets.len()], &offsets).unwrap();
    assert!(
        backend
            .dcnv4_aggregate_native(&t_in, &t_off, g, 1.0, cfg)
            .is_err()
    );
    let g = geometry(2, 1, false, false);
    // Short offsets: rejected by the length check.
    let short = vec![0.0f32; offsets.len() - 1];
    let t_short = backend.upload_f32(&[short.len()], &short).unwrap();
    assert!(
        backend
            .dcnv4_aggregate_native(&t_in, &t_short, g, 1.0, cfg)
            .is_err()
    );
}

#[test]
fn device_aggregation_backward_matches_the_host_reference() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    let offset_scale = 0.75f32;
    for (channels, groups) in [(2usize, 1usize), (4, 2)] {
        for softmax in [false, true] {
            for remove_center in [false, true] {
                let g = geometry(channels, groups, softmax, remove_center);
                let input: Vec<f32> = (0..g.batch * g.channels * g.input_height * g.input_width)
                    .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
                    .collect();
                let offsets: Vec<f32> = (0..g.batch * g.spatial() * g.offset_dim())
                    .map(|i| 0.5 + ((i * 3 % 5) as f32) * 0.02)
                    .collect();
                let seed: Vec<f32> = (0..g.batch * g.channels * g.spatial())
                    .map(|i| ((i * 7 % 17) as f32 - 8.0) * 0.3)
                    .collect();
                let (want_in, want_off) =
                    host::aggregate_backward(&input, &offsets, &seed, &g, offset_scale).unwrap();
                let t_in = backend.upload_f32(&[input.len()], &input).unwrap();
                let t_off = backend.upload_f32(&[offsets.len()], &offsets).unwrap();
                let t_seed = backend.upload_f32(&[seed.len()], &seed).unwrap();
                let (d_in, d_off) = backend
                    .dcnv4_aggregate_backward_native(&t_in, &t_off, &t_seed, g, offset_scale, cfg)
                    .unwrap();
                assert_close(
                    &format!(
                        "dcnv4_bwd/in sm={softmax} rc={remove_center} c={channels} g={groups}"
                    ),
                    &backend.download_f32(&d_in).unwrap(),
                    &want_in,
                );
                assert_close(
                    &format!(
                        "dcnv4_bwd/off sm={softmax} rc={remove_center} c={channels} g={groups}"
                    ),
                    &backend.download_f32(&d_off).unwrap(),
                    &want_off,
                );
            }
        }
    }
}

#[test]
fn device_adjoint_covers_strided_dilated_and_far_outside_samples() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    for size in [1, 3, 5] {
        for softmax in [false, true] {
            for remove_center in [false, true] {
                let mut g = geometry(6, 2, softmax, remove_center);
                g.size = size;
                g.stride_x = 2;
                g.stride_y = 3;
                g.padding = 2;
                g.dilation = 2;
                let input: Vec<_> = (0..g.batch * g.channels * g.input_height * g.input_width)
                    .map(|i| (i % 19) as f32 * 0.13 - 1.0)
                    .collect();
                // Include signed masks and large displacements outside every edge.
                let offsets: Vec<_> = (0..g.batch * g.spatial() * g.offset_dim())
                    .map(|i| match i % 7 {
                        0 => -30.5,
                        1 => 30.5,
                        _ => (i % 11) as f32 * 0.27 - 1.17,
                    })
                    .collect();
                let seed: Vec<_> = (0..g.batch * g.channels * g.spatial())
                    .map(|i| (i % 7) as f32 * 0.4 - 1.2)
                    .collect();
                let want = host::aggregate_backward(&input, &offsets, &seed, &g, 0.8).unwrap();
                let x = backend.upload_f32(&[input.len()], &input).unwrap();
                let off = backend.upload_f32(&[offsets.len()], &offsets).unwrap();
                let dy = backend.upload_f32(&[seed.len()], &seed).unwrap();
                let (dx, doff) = backend
                    .dcnv4_aggregate_backward_native(&x, &off, &dy, g, 0.8, cfg)
                    .unwrap();
                assert_close("edge dx", &backend.download_f32(&dx).unwrap(), &want.0);
                assert_close("edge doff", &backend.download_f32(&doff).unwrap(), &want.1);
                let out = backend
                    .dcnv4_aggregate_native(&x, &off, g, 0.8, cfg)
                    .unwrap();
                assert_close(
                    "edge forward",
                    &backend.download_f32(&out).unwrap(),
                    &host::aggregate(&input, &offsets, &g, 0.8).unwrap(),
                );
            }
        }
    }
}

#[test]
fn device_adjoint_rejects_bad_geometry_and_each_short_operand() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    let g = geometry(2, 1, false, false);
    let x_len = g.batch * g.channels * g.input_height * g.input_width;
    let off_len = g.batch * g.spatial() * g.offset_dim();
    let dy_len = g.batch * g.channels * g.spatial();
    let upload = |len| backend.upload_f32(&[len], &vec![0.0; len]).unwrap();
    let x = upload(x_len);
    let off = upload(off_len);
    let dy = upload(dy_len);
    for (a, b, c) in [
        (x_len - 1, off_len, dy_len),
        (x_len, off_len - 1, dy_len),
        (x_len, off_len, dy_len - 1),
    ] {
        assert!(
            backend
                .dcnv4_aggregate_backward_native(&upload(a), &upload(b), &upload(c), g, 1.0, cfg)
                .is_err()
        );
    }
    for bad in [
        host::Dcnv4Geometry { batch: 0, ..g },
        host::Dcnv4Geometry { groups: 0, ..g },
        host::Dcnv4Geometry { groups: 3, ..g },
        host::Dcnv4Geometry { dilation: 0, ..g },
        host::Dcnv4Geometry { size: 7, ..g },
        host::Dcnv4Geometry {
            size: usize::MAX,
            ..g
        },
        host::Dcnv4Geometry {
            batch: usize::MAX,
            ..g
        },
        host::Dcnv4Geometry {
            input_width: usize::MAX,
            ..g
        },
        host::Dcnv4Geometry {
            output_height: usize::MAX,
            ..g
        },
    ] {
        assert!(
            backend
                .dcnv4_aggregate_backward_native(&x, &off, &dy, bad, 1.0, cfg)
                .is_err()
        );
    }
}

#[test]
fn device_adjoint_matches_device_forward_finite_differences() {
    let backend = backend();
    let cfg = ElementwiseLaunchConfig::default();
    // With scale=1 and offsets=0.5, every sampled coordinate has fractional
    // part 0.5. The +/-0.01 perturbation cannot straddle a floor-cell seam.
    let eps = 0.01f32;
    for softmax in [false, true] {
        for remove_center in [false, true] {
            let g = host::Dcnv4Geometry {
                batch: 2,
                channels: 2,
                groups: 2,
                input_height: 2,
                input_width: 3,
                output_height: 1,
                output_width: 1,
                ..geometry(2, 2, softmax, remove_center)
            };
            let input: Vec<_> = (0..g.batch * g.channels * g.input_height * g.input_width)
                .map(|i| (i % 7) as f32 * 0.2 - 0.6)
                .collect();
            let offsets = vec![0.5; g.batch * g.spatial() * g.offset_dim()];
            let seed = vec![0.7, -1.3, -0.4, 1.1];
            let x = backend.upload_f32(&[input.len()], &input).unwrap();
            let off = backend.upload_f32(&[offsets.len()], &offsets).unwrap();
            let dy = backend.upload_f32(&[seed.len()], &seed).unwrap();
            let (dx, doff) = backend
                .dcnv4_aggregate_backward_native(&x, &off, &dy, g, 1.0, cfg)
                .unwrap();
            let dx = backend.download_f32(&dx).unwrap();
            let doff = backend.download_f32(&doff).unwrap();
            let objective = |input: &[f32], offsets: &[f32]| -> f32 {
                let x = backend.upload_f32(&[input.len()], input).unwrap();
                let off = backend.upload_f32(&[offsets.len()], offsets).unwrap();
                let y = backend
                    .dcnv4_aggregate_native(&x, &off, g, 1.0, cfg)
                    .unwrap();
                backend
                    .download_f32(&y)
                    .unwrap()
                    .iter()
                    .zip(&seed)
                    .map(|(y, s)| y * s)
                    .sum()
            };
            for is_offset in [false, true] {
                let (values, grads) = if is_offset {
                    (&offsets, &doff)
                } else {
                    (&input, &dx)
                };
                for i in 0..values.len() {
                    let mut plus = values.clone();
                    let mut minus = values.clone();
                    plus[i] += eps;
                    minus[i] -= eps;
                    let fd = if is_offset {
                        (objective(&input, &plus) - objective(&input, &minus)) / (2.0 * eps)
                    } else {
                        (objective(&plus, &offsets) - objective(&minus, &offsets)) / (2.0 * eps)
                    };
                    assert!(
                        (grads[i] - fd).abs() < 2e-3,
                        "offset={is_offset} sm={softmax} rc={remove_center} i={i}: analytic={} fd={fd}",
                        grads[i]
                    );
                }
            }
        }
    }
}
