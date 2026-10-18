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

fn geometry(channels: usize, groups: usize, softmax: bool, remove_center: bool) -> host::Dcnv4Geometry {
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
    assert!(backend
        .dcnv4_aggregate_native(&t_in, &t_off, g, 1.0, cfg)
        .is_err());
    let g = geometry(2, 1, false, false);
    // Short offsets: rejected by the length check.
    let short = vec![0.0f32; offsets.len() - 1];
    let t_short = backend.upload_f32(&[short.len()], &short).unwrap();
    assert!(backend
        .dcnv4_aggregate_native(&t_in, &t_short, g, 1.0, cfg)
        .is_err());
}
