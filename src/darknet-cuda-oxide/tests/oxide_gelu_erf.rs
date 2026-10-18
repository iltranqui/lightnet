//! Exact-erf GELU prerequisites for LAYA; not act-head runtime integration.
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            if w.is_nan() {
                g.is_nan()
            } else if w.is_infinite() {
                g == w
            } else {
                (g - w).abs() <= tol * (1.0 + w.abs())
            },
            "{i}: {g} vs {w}"
        );
    }
}

#[test]
fn gelu_erf_native_forward_backward_matches_hand_values_and_cpu_not_tanh() {
    let backend = OxideBackend::new().unwrap();
    let shape = [2, 2, 3, 5];
    let values = [-3.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let x: Vec<f32> = (0..60).map(|i| values[i % 6]).collect();
    let seed: Vec<f32> = (0..60).map(|i| [-1.5, 0.0, 0.25, 2.0][i % 4]).collect();
    let want: Vec<f32> = x
        .iter()
        .map(|&v| darknet_core::laya_scorer::gelu_erf(v))
        .collect();
    let want_grad: Vec<f32> = x
        .iter()
        .zip(&seed)
        .map(|(&v, &s)| s * darknet_core::laya_scorer::gelu_erf_grad(v))
        .collect();
    let host: Vec<_> = x
        .iter()
        .map(|&v| darknet_cuda::host_ops::gelu_erf::value(v))
        .collect();
    let host_grad: Vec<_> = x
        .iter()
        .zip(&seed)
        .map(|(&v, &s)| s * darknet_cuda::host_ops::gelu_erf::slope(v))
        .collect();
    close(&host, &want, 1e-7);
    close(&host_grad, &want_grad, 1e-7);
    close(
        &want[1..5],
        &[-0.15865525, 0.0, 0.84134475, 1.9544997],
        1e-6,
    );
    assert!((darknet_core::laya_scorer::gelu_erf_grad(1.0) - 1.0833155).abs() < 1e-6);
    assert!((darknet_core::laya_scorer::gelu_erf_grad(-1.0) + 0.08331547).abs() < 1e-6);
    let input = backend.upload_f32(&shape, &x).unwrap();
    let seeds = backend.upload_f32(&shape, &seed).unwrap();
    for tile_len in [32, 128, 256] {
        let config = ElementwiseLaunchConfig { tile_len };
        let out = backend.gelu_erf_f32(&input, &shape, config).unwrap();
        let grad = backend
            .gelu_erf_backward_f32(&input, &seeds, &shape, config)
            .unwrap();
        assert_eq!(out.shape(), shape);
        close(&backend.download_f32(&out).unwrap(), &want, 2e-6);
        close(&backend.download_f32(&grad).unwrap(), &want_grad, 2e-6);
    }
    let old = backend
        .activation_f32(
            &input,
            F32ActivationKind::Gelu,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let old = backend.download_f32(&old).unwrap();
    assert!(
        (old[3] - want[3]).abs() > 1e-4,
        "existing tanh GELU must remain distinct"
    );
}

#[test]
fn gelu_erf_native_backward_matches_actual_gpu_forward_central_differences() {
    let backend = OxideBackend::new().unwrap();
    let shape = [2, 2, 3, 5];
    let x: Vec<f32> = (0..60).map(|i| i as f32 * 0.1 - 3.0).collect();
    let seed: Vec<f32> = (0..60).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
    let input = backend.upload_f32(&shape, &x).unwrap();
    let seeds = backend.upload_f32(&shape, &seed).unwrap();
    let config = ElementwiseLaunchConfig::default();
    let grad = backend
        .gelu_erf_backward_f32(&input, &seeds, &shape, config)
        .unwrap();
    let grad = backend.download_f32(&grad).unwrap();
    for i in [0, 7, 13, 20, 29, 30, 31, 39, 48, 59] {
        let score = |offset: f32| {
            let mut perturbed = x.clone();
            perturbed[i] += offset;
            let a = backend.upload_f32(&shape, &perturbed).unwrap();
            let y = backend.gelu_erf_f32(&a, &shape, config).unwrap();
            backend
                .download_f32(&y)
                .unwrap()
                .iter()
                .zip(&seed)
                .map(|(&v, &s)| f64::from(v) * f64::from(s))
                .sum::<f64>()
        };
        let fd = (score(0.001) - score(-0.001)) / 0.002;
        assert!(
            (fd - f64::from(grad[i])).abs() < 0.0005,
            "{i}: {fd} vs {}",
            grad[i]
        );
    }
}

#[test]
fn gelu_erf_native_extremes_and_nonfinite_follow_literal_cpu_formula() {
    let backend = OxideBackend::new().unwrap();
    let x = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        100.0,
        -100.0,
        15.0,
        -15.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        1.0,
        100.0,
        -100.0,
    ];
    let mut seed = vec![1.0; x.len()];
    seed[11] = f32::NAN;
    seed[12] = f32::INFINITY;
    seed[13] = f32::INFINITY;
    let shape = [x.len()];
    let input = backend.upload_f32(&shape, &x).unwrap();
    let seeds = backend.upload_f32(&shape, &seed).unwrap();
    let want: Vec<_> = x
        .iter()
        .map(|&v| darknet_core::laya_scorer::gelu_erf(v))
        .collect();
    let want_grad: Vec<_> = x
        .iter()
        .zip(&seed)
        .map(|(&v, &s)| s * darknet_core::laya_scorer::gelu_erf_grad(v))
        .collect();
    assert!(want[2].is_nan()); // literal -Inf * 0, not a substituted limiting value
    let out = backend
        .gelu_erf_f32(&input, &shape, ElementwiseLaunchConfig::default())
        .unwrap();
    let grad = backend
        .gelu_erf_backward_f32(&input, &seeds, &shape, ElementwiseLaunchConfig::default())
        .unwrap();
    close(&backend.download_f32(&out).unwrap(), &want, 2e-6);
    close(&backend.download_f32(&grad).unwrap(), &want_grad, 2e-6);
}

#[test]
fn gelu_erf_native_rejects_bad_shape_storage_and_seed_metadata() {
    let backend = OxideBackend::new().unwrap();
    let shape = [2, 2, 3, 5];
    let config = ElementwiseLaunchConfig::default();
    let a = backend.upload_f32(&shape, &vec![1.0; 60]).unwrap();
    for bad in [
        vec![],
        vec![0],
        vec![usize::MAX, 2],
        vec![2, 2, 3, 4],
        vec![3, 2, 2, 5],
    ] {
        assert!(backend.gelu_erf_f32(&a, &bad, config).is_err());
        assert!(backend.gelu_erf_backward_f32(&a, &a, &bad, config).is_err());
    }
    let short = backend.upload_f32(&[1], &[0.0]).unwrap();
    let flat = backend.upload_f32(&[60], &vec![0.0; 60]).unwrap();
    assert!(backend.gelu_erf_f32(&short, &shape, config).is_err());
    assert!(
        backend
            .gelu_erf_backward_f32(&a, &short, &shape, config)
            .is_err()
    );
    assert!(
        backend
            .gelu_erf_backward_f32(&a, &flat, &shape, config)
            .is_err()
    );
    assert!(
        backend
            .gelu_erf_backward_f32(&short, &a, &shape, config)
            .is_err()
    );
}
