//! Host references for per-sample FiLM affine application and its adjoint.

/// Apply channel-wise FiLM parameters laid out as `[batch, channels]` to an
/// NCHW tensor: `y[b,c,s] = x[b,c,s] * (1 + gamma[b,c]) + beta[b,c]`.
pub fn affine(input: &[f32], gamma: &[f32], beta: &[f32], shape: &[usize]) -> Vec<f32> {
    assert!(shape.len() == 4, "FiLM affine shape must be NCHW");
    let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
    let spatial = height * width;
    assert!(batch > 0 && channels > 0 && spatial > 0);
    assert_eq!(input.len(), batch * channels * spatial);
    assert_eq!(gamma.len(), batch * channels);
    assert_eq!(beta.len(), batch * channels);
    let mut output = vec![0.0; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            let affine = b * channels + c;
            let offset = affine * spatial;
            let scale = 1.0 + gamma[affine];
            for s in 0..spatial {
                output[offset + s] = input[offset + s] * scale + beta[affine];
            }
        }
    }
    output
}

/// Exact adjoint of [`affine`]. Returns `(d_input, d_gamma, d_beta)`.
pub fn affine_backward(
    input: &[f32],
    output_gradient: &[f32],
    gamma: &[f32],
    shape: &[usize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(shape.len() == 4, "FiLM affine shape must be NCHW");
    let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
    let spatial = height * width;
    assert!(batch > 0 && channels > 0 && spatial > 0);
    assert_eq!(input.len(), batch * channels * spatial);
    assert_eq!(output_gradient.len(), input.len());
    assert_eq!(gamma.len(), batch * channels);
    let mut input_gradient = vec![0.0; input.len()];
    let mut gamma_gradient = vec![0.0; batch * channels];
    let mut beta_gradient = vec![0.0; batch * channels];
    for b in 0..batch {
        for c in 0..channels {
            let affine = b * channels + c;
            let offset = affine * spatial;
            let scale = 1.0 + gamma[affine];
            for s in 0..spatial {
                let index = offset + s;
                let gradient = output_gradient[index];
                input_gradient[index] = gradient * scale;
                gamma_gradient[affine] += gradient * input[index];
                beta_gradient[affine] += gradient;
            }
        }
    }
    (input_gradient, gamma_gradient, beta_gradient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_and_adjoint_are_batch_channel_and_spatial_independent() {
        let shape = [2, 2, 1, 2];
        let input = [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let gamma = [2.0, 0.0, 1.0, 3.0];
        let beta = [-2.0, 0.0, -1.0, -1.5];
        assert_eq!(
            affine(&input, &gamma, &beta, &shape),
            vec![1.0, 4.0, 10.0, 20.0, 5.0, 7.0, 118.5, 158.5]
        );

        let dy = [0.5, -0.25, 0.2, 0.1, -0.3, 0.4, 0.05, -0.2];
        let (dx, dgamma, dbeta) = affine_backward(&input, &dy, &gamma, &shape);
        assert_eq!(dx, vec![1.5, -0.75, 0.2, 0.1, -0.6, 0.8, 0.2, -0.8]);
        assert_eq!(dgamma, vec![0.0, 4.0, 0.7, -6.5]);
        for (actual, expected) in dbeta.iter().zip([0.25, 0.3, 0.1, -0.15]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
