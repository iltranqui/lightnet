//! Host references for per-sample FiLM affine application and its adjoint.

/// Full FiLM projection reference. Weight layouts are `[embed_dim, param_count]`
/// and `[channels, embed_dim]`; `input` is NCHW and `hf_params` is `[B,P]`.
pub fn forward(
    input: &[f32],
    hf_params: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    param_count: usize,
    embed_dim: usize,
    w_embed: &[f32],
    b_embed: &[f32],
    w_gamma: &[f32],
    b_gamma: &[f32],
    w_beta: &[f32],
    b_beta: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(batch > 0 && channels > 0 && height > 0 && width > 0);
    assert!(param_count > 0 && embed_dim > 0);
    assert_eq!(input.len(), batch * channels * height * width);
    assert_eq!(hf_params.len(), batch * param_count);
    assert_eq!(w_embed.len(), embed_dim * param_count);
    assert_eq!(b_embed.len(), embed_dim);
    assert_eq!(w_gamma.len(), channels * embed_dim);
    assert_eq!(b_gamma.len(), channels);
    assert_eq!(w_beta.len(), channels * embed_dim);
    assert_eq!(b_beta.len(), channels);

    let mut embedding = vec![0.0; batch * embed_dim];
    let mut gamma = vec![0.0; batch * channels];
    let mut beta = vec![0.0; batch * channels];
    for b in 0..batch {
        for j in 0..embed_dim {
            let mut value = b_embed[j];
            for i in 0..param_count {
                value += w_embed[j * param_count + i] * hf_params[b * param_count + i];
            }
            embedding[b * embed_dim + j] = value.max(0.0);
        }
        for c in 0..channels {
            let mut g = b_gamma[c];
            let mut be = b_beta[c];
            for j in 0..embed_dim {
                let embedded = embedding[b * embed_dim + j];
                g += w_gamma[c * embed_dim + j] * embedded;
                be += w_beta[c * embed_dim + j] * embedded;
            }
            gamma[b * channels + c] = g;
            beta[b * channels + c] = be;
        }
    }
    let output = affine(input, &gamma, &beta, &[batch, channels, height, width]);
    (output, embedding, gamma, beta)
}

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

/// Gradients of the full FiLM projection, including the optional source input.
#[derive(Clone, Debug, PartialEq)]
pub struct Gradients {
    pub input: Vec<f32>,
    pub hf_params: Vec<f32>,
    pub w_embed: Vec<f32>,
    pub b_embed: Vec<f32>,
    pub w_gamma: Vec<f32>,
    pub b_gamma: Vec<f32>,
    pub w_beta: Vec<f32>,
    pub b_beta: Vec<f32>,
}

/// Exact adjoint of [`forward`] for the piecewise-linear ReLU branch.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    input: &[f32],
    output_gradient: &[f32],
    hf_params: &[f32],
    embedding: &[f32],
    gamma: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    param_count: usize,
    embed_dim: usize,
    w_embed: &[f32],
    w_gamma: &[f32],
    w_beta: &[f32],
) -> Gradients {
    let spatial = height * width;
    assert!(batch > 0 && channels > 0 && spatial > 0);
    assert!(param_count > 0 && embed_dim > 0);
    assert_eq!(input.len(), batch * channels * spatial);
    assert_eq!(output_gradient.len(), input.len());
    assert_eq!(hf_params.len(), batch * param_count);
    assert_eq!(embedding.len(), batch * embed_dim);
    assert_eq!(gamma.len(), batch * channels);
    assert_eq!(w_embed.len(), embed_dim * param_count);
    assert_eq!(w_gamma.len(), channels * embed_dim);
    assert_eq!(w_beta.len(), channels * embed_dim);

    let (input_gradient, gamma_gradient, beta_gradient) = affine_backward(
        input,
        output_gradient,
        gamma,
        &[batch, channels, height, width],
    );
    let mut embedding_gradient = vec![0.0; batch * embed_dim];
    for b in 0..batch {
        for c in 0..channels {
            for j in 0..embed_dim {
                let dg = gamma_gradient[b * channels + c];
                let db = beta_gradient[b * channels + c];
                embedding_gradient[b * embed_dim + j] +=
                    dg * w_gamma[c * embed_dim + j] + db * w_beta[c * embed_dim + j];
            }
        }
        for j in 0..embed_dim {
            if embedding[b * embed_dim + j] <= 0.0 {
                embedding_gradient[b * embed_dim + j] = 0.0;
            }
        }
    }

    let mut gradients = Gradients {
        input: input_gradient,
        hf_params: vec![0.0; batch * param_count],
        w_embed: vec![0.0; embed_dim * param_count],
        b_embed: vec![0.0; embed_dim],
        w_gamma: vec![0.0; channels * embed_dim],
        b_gamma: vec![0.0; channels],
        w_beta: vec![0.0; channels * embed_dim],
        b_beta: vec![0.0; channels],
    };
    for c in 0..channels {
        for j in 0..embed_dim {
            let mut dgamma_weight = 0.0;
            let mut dbeta_weight = 0.0;
            for b in 0..batch {
                dgamma_weight += gamma_gradient[b * channels + c] * embedding[b * embed_dim + j];
                dbeta_weight += beta_gradient[b * channels + c] * embedding[b * embed_dim + j];
            }
            gradients.w_gamma[c * embed_dim + j] = dgamma_weight;
            gradients.w_beta[c * embed_dim + j] = dbeta_weight;
        }
        for b in 0..batch {
            gradients.b_gamma[c] += gamma_gradient[b * channels + c];
            gradients.b_beta[c] += beta_gradient[b * channels + c];
        }
    }
    for j in 0..embed_dim {
        for b in 0..batch {
            let de = embedding_gradient[b * embed_dim + j];
            gradients.b_embed[j] += de;
            for i in 0..param_count {
                gradients.w_embed[j * param_count + i] += de * hf_params[b * param_count + i];
                gradients.hf_params[b * param_count + i] += de * w_embed[j * param_count + i];
            }
        }
    }
    gradients
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
