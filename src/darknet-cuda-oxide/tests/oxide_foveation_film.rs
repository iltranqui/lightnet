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
fn foveation_film_forward_projection_composes_native_linear_and_affine() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [2, 3, 2, 2];
    let batch = shape[0];
    let channels = shape[1];
    let param_count = 2;
    let embed_dim = 3;
    let input: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| (i as f32 * 0.23).sin())
        .collect();
    let hf_values = vec![0.7, -0.2, -0.4, 0.9];
    let w_embed = vec![0.4, -0.1, -0.2, 0.3, 0.15, 0.7];
    let b_embed = vec![0.1, -0.05, 0.2];
    let w_gamma = vec![0.2, -0.1, 0.4, -0.3, 0.25, 0.1, 0.05, 0.2, -0.15];
    let b_gamma = vec![0.01, -0.03, 0.04];
    let w_beta = vec![-0.1, 0.3, 0.2, 0.15, -0.2, 0.35, 0.4, 0.05, -0.25];
    let b_beta = vec![-0.02, 0.06, 0.03];
    let x = backend.upload_f32(&shape, &input).unwrap();
    let hf = backend
        .upload_f32(&[batch, param_count], &hf_values)
        .unwrap();
    let w_embed_t = backend
        .upload_f32(&[embed_dim, param_count], &w_embed)
        .unwrap();
    let b_embed_t = backend.upload_f32(&[embed_dim], &b_embed).unwrap();
    let w_gamma_t = backend
        .upload_f32(&[channels, embed_dim], &w_gamma)
        .unwrap();
    let b_gamma_t = backend.upload_f32(&[channels], &b_gamma).unwrap();
    let w_beta_t = backend.upload_f32(&[channels, embed_dim], &w_beta).unwrap();
    let b_beta_t = backend.upload_f32(&[channels], &b_beta).unwrap();

    let (output, embedding, gamma, beta) = backend
        .foveation_film_forward_f32(
            &x,
            &hf,
            &w_embed_t,
            &b_embed_t,
            &w_gamma_t,
            &b_gamma_t,
            &w_beta_t,
            &b_beta_t,
            &shape,
            param_count,
            embed_dim,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let (expected, expected_embedding, expected_gamma, expected_beta) = foveation_film::forward(
        &input,
        &hf_values,
        batch,
        channels,
        shape[2],
        shape[3],
        param_count,
        embed_dim,
        &w_embed,
        &b_embed,
        &w_gamma,
        &b_gamma,
        &w_beta,
        &b_beta,
    );
    close(
        &backend.download_f32(&output).unwrap(),
        &expected,
        2e-6,
        "output",
    );
    close(
        &backend.download_f32(&embedding).unwrap(),
        &expected_embedding,
        1e-6,
        "embedding",
    );
    close(
        &backend.download_f32(&gamma).unwrap(),
        &expected_gamma,
        2e-6,
        "gamma",
    );
    close(
        &backend.download_f32(&beta).unwrap(),
        &expected_beta,
        2e-6,
        "beta",
    );

    let seed: Vec<f32> = (0..input.len())
        .map(|i| (i as f32 * 0.19).cos() * 0.25)
        .collect();
    let dy = backend.upload_f32(&shape, &seed).unwrap();
    let gradients = backend
        .foveation_film_backward_f32(
            &x,
            &dy,
            &hf,
            &embedding,
            &gamma,
            &w_embed_t,
            &w_gamma_t,
            &w_beta_t,
            &shape,
            param_count,
            embed_dim,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let expected_gradients = foveation_film::backward(
        &input,
        &seed,
        &hf_values,
        &expected_embedding,
        &expected_gamma,
        batch,
        channels,
        shape[2],
        shape[3],
        param_count,
        embed_dim,
        &w_embed,
        &w_gamma,
        &w_beta,
    );
    for (name, actual, expected) in [
        (
            "input gradient",
            &gradients.input,
            &expected_gradients.input,
        ),
        (
            "HF gradient",
            &gradients.hf_params,
            &expected_gradients.hf_params,
        ),
        (
            "embedding weight gradient",
            &gradients.w_embed,
            &expected_gradients.w_embed,
        ),
        (
            "embedding bias gradient",
            &gradients.b_embed,
            &expected_gradients.b_embed,
        ),
        (
            "gamma weight gradient",
            &gradients.w_gamma,
            &expected_gradients.w_gamma,
        ),
        (
            "gamma bias gradient",
            &gradients.b_gamma,
            &expected_gradients.b_gamma,
        ),
        (
            "beta weight gradient",
            &gradients.w_beta,
            &expected_gradients.w_beta,
        ),
        (
            "beta bias gradient",
            &gradients.b_beta,
            &expected_gradients.b_beta,
        ),
    ] {
        close(&backend.download_f32(actual).unwrap(), expected, 3e-5, name);
    }
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
