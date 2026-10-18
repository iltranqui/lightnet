//! Native deformable bilinear gather: device vs host-reference parity for both
//! directions, including the two conventions that are easy to get wrong (the
//! far-outside early-out and the floor-cell one-sided derivative at integer
//! coordinates), plus a device finite-difference check.
//!
//! The reference is `darknet_cuda::host_ops::bilinear_gather`, which is the
//! contract the CPU layers already use. `d_input` is a float-atomic scatter, so
//! the comparison is tolerance-based, not bit-exact.

use darknet_cuda::{
    BilinearGatherGeometry, ElementwiseLaunchConfig, F32TensorOpsBackend,
    host_ops::bilinear_gather as host,
};
use darknet_cuda_oxide::OxideBackend;

const TOL: f32 = 2e-5;

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        // NaN must stay NaN (the CPU contract lets it propagate); never compare
        // it with `==`.
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

/// B=2, C=3, 3x4 input, S=2 sample planes, 2x3 output cells. Coordinates are
/// chosen to cover: an exact pixel, a fractional interior, a partially
/// out-of-bounds cell, a far-outside sample and NaN.
fn fixture() -> (
    BilinearGatherGeometry,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let g = BilinearGatherGeometry::new(2, 3, 3, 4, 2, 2, 3).unwrap();
    let input: Vec<f32> = (0..g.input_len())
        .map(|i| (i as f32) * 0.125 - 1.5)
        .collect();
    let mut sample_y = vec![0.0f32; g.sample_len()];
    let mut sample_x = vec![0.0f32; g.sample_len()];
    // Per (b, s) plane, six cells with distinct coordinate characters.
    let patterns = [
        (1.0, 2.0),      // exact pixel (right-sided cell)
        (0.5, 1.5),      // fractional interior
        (2.75, -0.5),    // partially outside on both axes
        (-2.0, 0.5),     // far outside (y)
        (1.25, 3.25),    // last row/column, one corner past the edge
        (f32::NAN, 0.5), // NaN propagates like the CPU reference
    ];
    for b in 0..g.batch {
        for s in 0..g.samples {
            for (cell, (y, x)) in patterns.iter().enumerate() {
                let index = g.sample_index(b, s, cell / g.output_width, cell % g.output_width);
                sample_y[index] = *y;
                sample_x[index] = *x;
            }
        }
    }
    let grad: Vec<f32> = (0..g.output_len())
        .map(|i| {
            if i % 5 == 0 {
                -1.0
            } else {
                0.5 + (i % 3) as f32 * 0.25
            }
        })
        .collect();
    (g, input, sample_y, sample_x, grad)
}

#[test]
fn forward_matches_the_host_reference() {
    let backend = backend();
    let (g, input, y, x, _) = fixture();
    let want = host::forward(&input, &y, &x, &g).unwrap();
    let t_in = backend.upload_f32(&[g.input_len()], &input).unwrap();
    let t_y = backend.upload_f32(&[g.sample_len()], &y).unwrap();
    let t_x = backend.upload_f32(&[g.sample_len()], &x).unwrap();
    let out = backend
        .deform_bilinear_gather_f32(&t_in, &t_y, &t_x, g, cfg())
        .unwrap();
    assert_eq!(out.len(), g.output_len());
    let got = backend.download_f32(&out).unwrap();
    assert_close("forward", &got, &want, TOL);
}

#[test]
fn backward_matches_the_host_reference() {
    let backend = backend();
    let (g, input, y, x, grad) = fixture();
    let (want_din, want_dy, want_dx) = host::backward(&input, &y, &x, &grad, &g).unwrap();
    let t_in = backend.upload_f32(&[g.input_len()], &input).unwrap();
    let t_y = backend.upload_f32(&[g.sample_len()], &y).unwrap();
    let t_x = backend.upload_f32(&[g.sample_len()], &x).unwrap();
    let t_grad = backend.upload_f32(&[g.output_len()], &grad).unwrap();
    let (d_in, d_y, d_x) = backend
        .deform_bilinear_gather_backward_f32(&t_in, &t_y, &t_x, &t_grad, g, cfg())
        .unwrap();
    assert_close(
        "d_input",
        &backend.download_f32(&d_in).unwrap(),
        &want_din,
        TOL,
    );
    assert_close(
        "d_sample_y",
        &backend.download_f32(&d_y).unwrap(),
        &want_dy,
        TOL,
    );
    assert_close(
        "d_sample_x",
        &backend.download_f32(&d_x).unwrap(),
        &want_dx,
        TOL,
    );
}

#[test]
fn far_outside_samples_are_zero_in_both_directions() {
    let backend = backend();
    let g = BilinearGatherGeometry::new(1, 2, 3, 3, 1, 1, 2).unwrap();
    let input: Vec<f32> = (0..g.input_len()).map(|i| i as f32 + 1.0).collect();
    let y = vec![-3.0f32, 4.0];
    let x = vec![0.5f32, 0.5];
    let grad = vec![1.0f32; g.output_len()];
    let t_in = backend.upload_f32(&[g.input_len()], &input).unwrap();
    let t_y = backend.upload_f32(&[g.sample_len()], &y).unwrap();
    let t_x = backend.upload_f32(&[g.sample_len()], &x).unwrap();
    let t_grad = backend.upload_f32(&[g.output_len()], &grad).unwrap();
    let out = backend
        .deform_bilinear_gather_f32(&t_in, &t_y, &t_x, g, cfg())
        .unwrap();
    assert!(
        backend
            .download_f32(&out)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    let (d_in, d_y, d_x) = backend
        .deform_bilinear_gather_backward_f32(&t_in, &t_y, &t_x, &t_grad, g, cfg())
        .unwrap();
    assert!(
        backend
            .download_f32(&d_in)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        backend
            .download_f32(&d_y)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
    assert!(
        backend
            .download_f32(&d_x)
            .unwrap()
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn device_backward_matches_finite_differences_of_device_forward() {
    let backend = backend();
    // A central difference needs a locally differentiable point: no NaN, no
    // integer coordinate (the floor-cell convention is one-sided there) and no
    // sample within `eps` of the far-outside cliff.
    let g = BilinearGatherGeometry::new(2, 3, 3, 4, 2, 2, 3).unwrap();
    let input: Vec<f32> = (0..g.input_len())
        .map(|i| ((i * 37 % 29) as f32) * 0.25 - 1.0)
        .collect();
    let mut y = vec![0.0f32; g.sample_len()];
    let mut x = vec![0.0f32; g.sample_len()];
    for i in 0..g.sample_len() {
        y[i] = (i % g.input_height) as f32 + 0.3 + 0.4 * ((i * 7 % 11) as f32 / 11.0);
        x[i] = (i % g.input_width) as f32 + 0.3 + 0.4 * ((i * 5 % 13) as f32 / 13.0);
    }
    let seed: Vec<f32> = (0..g.output_len())
        .map(|i| 0.25 + (i % 7) as f32 * 0.1)
        .collect();
    let t_y = backend.upload_f32(&[g.sample_len()], &y).unwrap();
    let t_x = backend.upload_f32(&[g.sample_len()], &x).unwrap();
    let t_seed = backend.upload_f32(&[g.output_len()], &seed).unwrap();
    let objective = |values: &[f32]| -> f32 {
        let t = backend.upload_f32(&[g.input_len()], values).unwrap();
        let out = backend
            .deform_bilinear_gather_f32(&t, &t_y, &t_x, g, cfg())
            .unwrap();
        backend
            .download_f32(&out)
            .unwrap()
            .iter()
            .zip(&seed)
            .map(|(v, s)| v * s)
            .sum()
    };
    let t_in = backend.upload_f32(&[g.input_len()], &input).unwrap();
    let (d_in, _, _) = backend
        .deform_bilinear_gather_backward_f32(&t_in, &t_y, &t_x, &t_seed, g, cfg())
        .unwrap();
    let analytic = backend.download_f32(&d_in).unwrap();
    let eps = 1e-2f32;
    for i in (0..g.input_len()).step_by(11) {
        let mut plus = input.clone();
        plus[i] += eps;
        let mut minus = input.clone();
        minus[i] -= eps;
        let fd = (objective(&plus) - objective(&minus)) / (2.0 * eps);
        assert!(
            (analytic[i] - fd).abs() < 2e-2,
            "d_input[{i}]: analytic {} vs fd {fd}",
            analytic[i]
        );
    }
}

#[test]
fn mismatched_operands_are_rejected() {
    let backend = backend();
    let (g, input, y, x, _) = fixture();
    let t_in = backend.upload_f32(&[g.input_len()], &input).unwrap();
    let t_y = backend.upload_f32(&[g.sample_len()], &y).unwrap();
    let t_y_short = backend.upload_f32(&[g.sample_len() - 1], &y[1..]).unwrap();
    let t_x = backend.upload_f32(&[g.sample_len()], &x).unwrap();
    assert!(
        backend
            .deform_bilinear_gather_f32(&t_in, &t_y_short, &t_x, g, cfg())
            .is_err()
    );
    assert!(
        backend
            .deform_bilinear_gather_backward_f32(&t_in, &t_y, &t_x, &t_y_short, g, cfg())
            .is_err()
    );
}
