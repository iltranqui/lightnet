use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, host_ops::foveation_film};
use darknet_cuda_oxide::OxideBackend;

fn close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

fn objective(input: &[f32], gamma: &[f32], beta: &[f32], shape: &[usize], seed: &[f32]) -> f32 {
    foveation_film::affine(input, gamma, beta, shape)
        .iter()
        .zip(seed)
        .map(|(output, seed)| output * seed)
        .sum()
}

#[test]
fn foveation_film_affine_forward_backward_and_finite_difference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [2, 3, 2, 3];
    let batch_channels = [shape[0], shape[1]];
    let input: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| (i as f32 * 0.17).sin() - 0.2)
        .collect();
    let gamma = vec![0.2, -0.4, 0.7, -0.1, 0.35, 0.05];
    let beta = vec![-0.3, 0.1, 0.45, 0.2, -0.15, 0.6];
    let seed: Vec<f32> = (0..input.len())
        .map(|i| (i as f32 * 0.11).cos() * 0.3)
        .collect();
    let x = backend.upload_f32(&shape, &input).unwrap();
    let gamma_tensor = backend.upload_f32(&batch_channels, &gamma).unwrap();
    let beta_tensor = backend.upload_f32(&batch_channels, &beta).unwrap();
    let dy = backend.upload_f32(&shape, &seed).unwrap();
    let config = ElementwiseLaunchConfig::default();

    let output = backend
        .foveation_film_affine_f32(&x, &gamma_tensor, &beta_tensor, &shape, config)
        .unwrap();
    let output = backend.download_f32(&output).unwrap();
    let expected = foveation_film::affine(&input, &gamma, &beta, &shape);
    close(&output, &expected, 1e-6, "forward");

    let (dx, dgamma, dbeta) = backend
        .foveation_film_affine_backward_f32(&x, &dy, &gamma_tensor, &shape, config)
        .unwrap();
    let dx = backend.download_f32(&dx).unwrap();
    let dgamma = backend.download_f32(&dgamma).unwrap();
    let dbeta = backend.download_f32(&dbeta).unwrap();
    let (expected_dx, expected_dgamma, expected_dbeta) =
        foveation_film::affine_backward(&input, &seed, &gamma, &shape);
    close(&dx, &expected_dx, 1e-6, "input adjoint");
    close(&dgamma, &expected_dgamma, 1e-6, "gamma adjoint");
    close(&dbeta, &expected_dbeta, 1e-6, "beta adjoint");

    let step = 1e-3;
    for index in 0..input.len() {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += step;
        minus[index] -= step;
        let numeric = (objective(&plus, &gamma, &beta, &shape, &seed)
            - objective(&minus, &gamma, &beta, &shape, &seed))
            / (2.0 * step);
        assert!(
            (numeric - dx[index]).abs() < 2e-4,
            "input FD[{index}]: numeric={numeric}, analytic={}",
            dx[index]
        );
    }
    for index in 0..gamma.len() {
        let mut gamma_plus = gamma.clone();
        let mut gamma_minus = gamma.clone();
        gamma_plus[index] += step;
        gamma_minus[index] -= step;
        let numeric_gamma = (objective(&input, &gamma_plus, &beta, &shape, &seed)
            - objective(&input, &gamma_minus, &beta, &shape, &seed))
            / (2.0 * step);
        assert!(
            (numeric_gamma - dgamma[index]).abs() < 2e-4,
            "gamma FD[{index}]: numeric={numeric_gamma}, analytic={}",
            dgamma[index]
        );

        let mut beta_plus = beta.clone();
        let mut beta_minus = beta.clone();
        beta_plus[index] += step;
        beta_minus[index] -= step;
        let numeric_beta = (objective(&input, &gamma, &beta_plus, &shape, &seed)
            - objective(&input, &gamma, &beta_minus, &shape, &seed))
            / (2.0 * step);
        assert!(
            (numeric_beta - dbeta[index]).abs() < 2e-4,
            "beta FD[{index}]: numeric={numeric_beta}, analytic={}",
            dbeta[index]
        );
    }
}
