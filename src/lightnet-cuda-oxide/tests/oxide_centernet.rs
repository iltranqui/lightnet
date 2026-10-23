use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

#[test]
fn centernet_nchw_heatmap_sigmoid_preserves_regression_planes_and_sanitizes() {
    let backend = OxideBackend::new().expect("create actual Oxide backend");
    let shape = [2, 6, 2, 3];
    let mut input = (0..shape.iter().product::<usize>())
        .map(|i| (i as f32 - 17.0) * 0.13)
        .collect::<Vec<_>>();
    // Deliberately include distinct non-finite behavior in both regions.
    input[0] = f32::INFINITY; // class sigmoid saturates to one
    input[6] = f32::NAN; // class NaN sanitizes to zero
    input[12] = f32::INFINITY; // regression infinity sanitizes to zero
    input[18] = f32::NEG_INFINITY; // next batch's class sigmoid is zero

    let x = backend.upload_f32(&shape, &input).expect("upload input");
    let output = backend
        .centernet_forward_f32(&x, &shape, 2, ElementwiseLaunchConfig::default())
        .expect("native CenterNet forward");
    let got = backend.download_f32(&output).expect("download output");
    let spatial = shape[2] * shape[3];
    let channels = shape[1];
    for i in 0..input.len() {
        let channel = (i / spatial) % channels;
        let value = if channel < 2 {
            1.0 / (1.0 + (-(input[i] as f64)).exp())
        } else {
            input[i] as f64
        };
        let want = if value.is_finite() { value as f32 } else { 0.0 };
        assert!(
            (got[i] - want).abs() <= 2.0e-6,
            "NCHW index {i}: got {}, expected {want}",
            got[i]
        );
    }
}

#[test]
fn centernet_rejects_invalid_shape_and_class_extent() {
    let backend = OxideBackend::new().expect("create actual Oxide backend");
    let x = backend.upload_f32(&[1, 4, 1, 2], &[0.0; 8]).unwrap();
    assert!(
        backend
            .centernet_forward_f32(&x, &[1, 4, 1, 2], 5, ElementwiseLaunchConfig::default())
            .is_err()
    );
    assert!(
        backend
            .centernet_forward_f32(&x, &[1, 4, 2], 2, ElementwiseLaunchConfig::default())
            .is_err()
    );
}
