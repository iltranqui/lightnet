use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, host_ops::deconvolutional};
use darknet_cuda_oxide::OxideBackend;

fn close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-5 + 2e-5 * w.abs(),
            "{name}[{i}]: got {g}, want {w}"
        );
    }
}

#[test]
fn native_deconvolution_forward_and_backward_match_host_reference() {
    let backend = OxideBackend::new().expect("create Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    for (batch, channels, ih, iw, filters, kernel, stride) in
        [(2, 2, 2, 3, 3, 2, 2), (1, 3, 3, 2, 2, 3, 1)]
    {
        let oh = stride * (ih - 1) + kernel;
        let ow = stride * (iw - 1) + kernel;
        let input: Vec<_> = (0..batch * channels * ih * iw)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.07)
            .collect();
        let weights: Vec<_> = (0..channels * filters * kernel * kernel)
            .map(|i| ((i * 11 % 23) as f32 - 10.0) * 0.03)
            .collect();
        let dy: Vec<_> = (0..batch * filters * oh * ow)
            .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.09)
            .collect();
        let x_shape = [batch, channels, ih, iw];
        let y_shape = [batch, filters, oh, ow];
        let w_shape = [channels, filters, kernel, kernel];
        let x = backend.upload_f32(&x_shape, &input).unwrap();
        let w = backend.upload_f32(&w_shape, &weights).unwrap();
        let dy_gpu = backend.upload_f32(&y_shape, &dy).unwrap();

        let output = backend
            .deconvolutional_f32(
                &x, &w, &y_shape, channels, ih, iw, filters, kernel, stride, config,
            )
            .expect("native deconvolution forward");
        let expected = deconvolutional::forward(
            &input, &weights, batch, channels, ih, iw, filters, kernel, stride, oh, ow,
        );
        close(
            "forward",
            &backend.download_f32(&output).unwrap(),
            &expected,
        );

        let (dx, dw, db) = backend
            .deconvolutional_backward_f32(
                &x, &w, &dy_gpu, &x_shape, &y_shape, filters, kernel, stride, config,
            )
            .expect("native deconvolution backward");
        let (expected_dx, expected_dw, expected_db) = deconvolutional::backward(
            &input, &weights, &dy, batch, channels, ih, iw, filters, kernel, stride, oh, ow,
        );
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
fn native_deconvolution_rejects_inconsistent_output_shape() {
    let backend = OxideBackend::new().expect("create Oxide backend");
    let x = backend.upload_f32(&[1, 1, 2, 2], &[1.0; 4]).unwrap();
    let w = backend.upload_f32(&[1, 1, 2, 2], &[1.0; 4]).unwrap();
    let error = match backend.deconvolutional_f32(
        &x,
        &w,
        &[1, 1, 3, 4],
        1,
        2,
        2,
        1,
        2,
        2,
        ElementwiseLaunchConfig::default(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("invalid transposed-convolution extent was accepted"),
    };
    assert!(error.to_string().contains("output"));
}
