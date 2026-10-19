use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, host_ops::local};
use darknet_cuda_oxide::OxideBackend;

fn close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() < 3e-5 + 3e-5 * w.abs(),
            "{name}[{i}]: {g} != {w}"
        );
    }
}

#[test]
fn local_forward_and_backward_match_host_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    for (batch, channels, height, width, filters, out_h, out_w, size, sx, sy, padding) in [
        (2, 2, 4, 5, 3, 3, 2, 2, 2, 1, 0),
        (1, 1, 3, 4, 2, 3, 4, 3, 1, 1, 1),
        (2, 2, 4, 4, 3, 3, 3, 2, 1, 1, 0),
    ] {
        let input: Vec<_> = (0..batch * channels * height * width)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.04)
            .collect();
        let locations = out_h * out_w;
        let kernel = channels * size * size;
        let weights: Vec<_> = (0..locations * filters * kernel)
            .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.015)
            .collect();
        let biases: Vec<_> = (0..locations * filters)
            .map(|i| (i as f32 - 2.0) * 0.025)
            .collect();
        let delta: Vec<_> = (0..batch * filters * locations)
            .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.07)
            .collect();
        let x_shape = [batch, channels, height, width];
        let y_shape = [batch, filters, out_h, out_w];
        let w_shape = [locations, filters, channels, size, size];
        let x = backend.upload_f32(&x_shape, &input).unwrap();
        let w = backend.upload_f32(&w_shape, &weights).unwrap();
        let b = backend.upload_f32(&[locations, filters], &biases).unwrap();
        let dy = backend.upload_f32(&y_shape, &delta).unwrap();
        let y = backend
            .local_f32(
                &x, &w, &b, &x_shape, &y_shape, filters, size, sx, sy, padding, config,
            )
            .expect("native local forward");
        let expected_y = local::forward(
            &input, &weights, &biases, batch, channels, height, width, filters, out_h, out_w, size,
            sx, sy, padding,
        )
        .unwrap();
        close("forward", &backend.download_f32(&y).unwrap(), &expected_y);

        let (dx, dw, db) = backend
            .local_backward_f32(
                &x, &w, &dy, &x_shape, &y_shape, filters, size, sx, sy, padding, config,
            )
            .expect("native local backward");
        let (expected_dx, expected_dw, expected_db) = local::backward(
            &input, &weights, &delta, batch, channels, height, width, filters, out_h, out_w, size,
            sx, sy, padding,
        )
        .unwrap();
        close(
            "input gradient",
            &backend.download_f32(&dx).unwrap(),
            &expected_dx,
        );
        close(
            "weight gradient",
            &backend.download_f32(&dw).unwrap(),
            &expected_dw,
        );
        close(
            "bias gradient",
            &backend.download_f32(&db).unwrap(),
            &expected_db,
        );
    }
}

#[test]
fn local_rejects_bad_weight_storage() {
    let backend = OxideBackend::new().unwrap();
    let x = backend.upload_f32(&[1, 1, 1, 1], &[1.0]).unwrap();
    let w = backend.upload_f32(&[0], &[]).unwrap();
    let b = backend.upload_f32(&[1, 1], &[0.0]).unwrap();
    assert!(
        backend
            .local_f32(
                &x,
                &w,
                &b,
                &[1, 1, 1, 1],
                &[1, 1, 1, 1],
                1,
                1,
                1,
                1,
                0,
                ElementwiseLaunchConfig::default(),
            )
            .is_err()
    );
}
