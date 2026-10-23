//! Pure CPU reference math for source-derived FiLM modulation from HF parameters.

use crate::error::{Error, Result};
use crate::runtime::FoveationFilmParams;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FoveationFilmForwardCache {
    /// ReLU embedding, row-major `[batch, embed_dim]`.
    pub embedding: Vec<f32>,
    /// Per-sample channel scales before adding the identity term.
    pub gamma: Vec<f32>,
    /// Per-sample channel shifts.
    pub beta: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FoveationFilmGradients {
    pub input: Vec<f32>,
    pub params: FoveationFilmParams,
    /// Per-batch gradient of the selected HF parameter vector.
    pub hf_params: Vec<f32>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[foveation_film]: {}", message.into()))
}

fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{what} element count overflows")))
    })
}

fn validate_layout(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    param_count: usize,
    embed_dim: usize,
    params: &FoveationFilmParams,
) -> Result<(usize, usize)> {
    if [batch, channels, height, width, param_count, embed_dim].contains(&0) {
        return Err(invalid(
            "NCHW dimensions, param_count and embed_dim must be positive",
        ));
    }
    let total = checked_product(&[batch, channels, height, width], "NCHW tensor")?;
    let embedding_count = checked_product(&[batch, embed_dim], "embedding")?;
    let batch_channels = checked_product(&[batch, channels], "channel cache")?;
    let embed_weights = checked_product(&[embed_dim, param_count], "embedding weights")?;
    let channel_weights = checked_product(&[channels, embed_dim], "channel weights")?;
    if params.w_embed.len() != embed_weights
        || params.b_embed.len() != embed_dim
        || params.w_gamma.len() != channel_weights
        || params.b_gamma.len() != channels
        || params.w_beta.len() != channel_weights
        || params.b_beta.len() != channels
    {
        return Err(invalid(
            "parameter layout does not match embed_dim and channels",
        ));
    }
    if embedding_count == 0 || batch_channels == 0 {
        return Err(invalid("cache dimensions overflow"));
    }
    Ok((total, height * width))
}

pub(crate) fn forward_nchw(
    input: &[f32],
    hf_params: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    param_count: usize,
    embed_dim: usize,
    params: &FoveationFilmParams,
) -> Result<(Vec<f32>, FoveationFilmForwardCache)> {
    let (total, spatial) = validate_layout(
        batch,
        channels,
        height,
        width,
        param_count,
        embed_dim,
        params,
    )?;
    if input.len() != total {
        return Err(invalid(format!(
            "input contains {} values, expected {total}",
            input.len()
        )));
    }
    let expected_hf = checked_product(&[batch, param_count], "HF parameter batch")?;
    if hf_params.len() != expected_hf {
        return Err(invalid(format!(
            "HF parameter cache contains {} values, expected {expected_hf}",
            hf_params.len()
        )));
    }

    let mut embedding = vec![0.0; batch * embed_dim];
    let mut gamma = vec![0.0; batch * channels];
    let mut beta = vec![0.0; batch * channels];
    for b in 0..batch {
        let hf = &hf_params[b * param_count..(b + 1) * param_count];
        for j in 0..embed_dim {
            let mut value = params.b_embed[j];
            for (i, &parameter) in hf.iter().enumerate() {
                value += params.w_embed[j * param_count + i] * parameter;
            }
            embedding[b * embed_dim + j] = if value > 0.0 { value } else { 0.0 };
        }
        for c in 0..channels {
            let mut scale = params.b_gamma[c];
            let mut shift = params.b_beta[c];
            for j in 0..embed_dim {
                let embedded = embedding[b * embed_dim + j];
                scale += params.w_gamma[c * embed_dim + j] * embedded;
                shift += params.w_beta[c * embed_dim + j] * embedded;
            }
            gamma[b * channels + c] = scale;
            beta[b * channels + c] = shift;
        }
    }

    let mut output = vec![0.0; total];
    for b in 0..batch {
        for c in 0..channels {
            let channel = b * channels + c;
            let scale = 1.0 + gamma[channel];
            let shift = beta[channel];
            let offset = channel * spatial;
            for s in 0..spatial {
                output[offset + s] = scale * input[offset + s] + shift;
            }
        }
    }
    Ok((
        output,
        FoveationFilmForwardCache {
            embedding,
            gamma,
            beta,
        },
    ))
}

pub(crate) fn backward_nchw(
    input: &[f32],
    output_gradient: &[f32],
    hf_params: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    param_count: usize,
    embed_dim: usize,
    params: &FoveationFilmParams,
    cache: &FoveationFilmForwardCache,
) -> Result<FoveationFilmGradients> {
    let (total, spatial) = validate_layout(
        batch,
        channels,
        height,
        width,
        param_count,
        embed_dim,
        params,
    )?;
    if input.len() != total || output_gradient.len() != total {
        return Err(invalid(
            "input/output-gradient shape does not match NCHW dimensions",
        ));
    }
    let expected_hf = checked_product(&[batch, param_count], "HF parameter batch")?;
    let expected_embedding = checked_product(&[batch, embed_dim], "embedding cache")?;
    let expected_channels = checked_product(&[batch, channels], "channel cache")?;
    if hf_params.len() != expected_hf
        || cache.embedding.len() != expected_embedding
        || cache.gamma.len() != expected_channels
        || cache.beta.len() != expected_channels
    {
        return Err(invalid(
            "forward cache layout does not match the configured layer",
        ));
    }

    let mut input_gradient = vec![0.0; total];
    let mut gradient = FoveationFilmParams {
        w_embed: vec![0.0; params.w_embed.len()],
        b_embed: vec![0.0; params.b_embed.len()],
        w_gamma: vec![0.0; params.w_gamma.len()],
        b_gamma: vec![0.0; params.b_gamma.len()],
        w_beta: vec![0.0; params.w_beta.len()],
        b_beta: vec![0.0; params.b_beta.len()],
    };
    let mut hf_parameter_gradient = vec![0.0; expected_hf];

    for b in 0..batch {
        let mut gamma_gradient = vec![0.0; channels];
        let mut beta_gradient = vec![0.0; channels];
        for c in 0..channels {
            let channel = b * channels + c;
            let offset = channel * spatial;
            let scale = 1.0 + cache.gamma[channel];
            for s in 0..spatial {
                let seed = output_gradient[offset + s];
                input_gradient[offset + s] = seed * scale;
                gamma_gradient[c] += seed * input[offset + s];
                beta_gradient[c] += seed;
            }
        }

        let mut embedding_gradient = vec![0.0; embed_dim];
        for c in 0..channels {
            let embed_offset = b * embed_dim;
            gradient.b_gamma[c] += gamma_gradient[c];
            gradient.b_beta[c] += beta_gradient[c];
            for j in 0..embed_dim {
                let embed = cache.embedding[embed_offset + j];
                gradient.w_gamma[c * embed_dim + j] += gamma_gradient[c] * embed;
                gradient.w_beta[c * embed_dim + j] += beta_gradient[c] * embed;
                embedding_gradient[j] += gamma_gradient[c] * params.w_gamma[c * embed_dim + j]
                    + beta_gradient[c] * params.w_beta[c * embed_dim + j];
            }
        }

        let hf_offset = b * param_count;
        for j in 0..embed_dim {
            if cache.embedding[b * embed_dim + j] <= 0.0 {
                embedding_gradient[j] = 0.0;
            }
            gradient.b_embed[j] += embedding_gradient[j];
            for i in 0..param_count {
                gradient.w_embed[j * param_count + i] +=
                    embedding_gradient[j] * hf_params[hf_offset + i];
                hf_parameter_gradient[hf_offset + i] +=
                    embedding_gradient[j] * params.w_embed[j * param_count + i];
            }
        }
    }

    Ok(FoveationFilmGradients {
        input: input_gradient,
        params: gradient,
        hf_params: hf_parameter_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BATCH: usize = 2;
    const CHANNELS: usize = 2;
    const HEIGHT: usize = 2;
    const WIDTH: usize = 3;
    const EMBED: usize = 2;
    const PARAMS: usize = 2;

    fn params() -> FoveationFilmParams {
        FoveationFilmParams {
            w_embed: vec![1.0, 0.0, -1.0, 1.0],
            b_embed: vec![0.0, 0.0],
            w_gamma: vec![0.1, 0.2, 0.3, -0.1],
            b_gamma: vec![0.05, -0.2],
            w_beta: vec![0.4, -0.5, 0.1, 0.2],
            b_beta: vec![0.01, 0.02],
        }
    }

    fn input() -> Vec<f32> {
        (1..=BATCH * CHANNELS * HEIGHT * WIDTH)
            .map(|value| value as f32)
            .collect()
    }

    fn hf_params() -> Vec<f32> {
        vec![1.0, 2.0, -1.0, 0.5]
    }

    fn forward(values: &FoveationFilmParams, input: &[f32], hf: &[f32]) -> Vec<f32> {
        forward_nchw(
            input, hf, BATCH, CHANNELS, HEIGHT, WIDTH, PARAMS, EMBED, values,
        )
        .expect("valid FiLM forward")
        .0
    }

    #[test]
    fn foveation_film_axis_layout_maps_batch_hf_embedding_and_channel_affines() {
        let input = input();
        let output = forward(&params(), &input, &hf_params());

        // Batch 0: embed=[1,1], gamma=[0.35,0], beta=[-0.09,0.32].
        assert!((output[0] - 1.26).abs() < 1e-6);
        assert!((output[6] - 7.32).abs() < 1e-6);
        // Batch 1: embed=[0,1.5], gamma=[0.35,-0.35], beta=[-0.74,0.32].
        assert!(
            (output[12] - 16.81).abs() < 2e-6,
            "output[12]={}",
            output[12]
        );
        assert!((output[18] - 12.67).abs() < 1e-6);
        // Broadcast values are constant across each NCHW channel plane.
        assert!((output[5] - (6.0 * 1.35 - 0.09)).abs() < 1e-6);
        assert!((output[11] - (12.0 + 0.32)).abs() < 1e-6);
        assert!((output[17] - (18.0 * 1.35 - 0.74)).abs() < 1e-6);
        assert!((output[23] - (24.0 * 0.65 + 0.32)).abs() < 1e-6);
    }

    fn objective(values: &FoveationFilmParams, input: &[f32], hf: &[f32], seed: &[f32]) -> f32 {
        forward(values, input, hf)
            .iter()
            .zip(seed)
            .map(|(value, seed)| value * seed)
            .sum()
    }

    fn check_finite_difference(
        name: &str,
        analytic: &[f32],
        values: &[f32],
        loss: impl Fn(&[f32]) -> f32,
    ) {
        assert_eq!(analytic.len(), values.len(), "{name} shape");
        let step = 1.0e-3;
        let mut candidate = values.to_vec();
        for index in 0..candidate.len() {
            let original = candidate[index];
            candidate[index] = original + step;
            let plus = loss(&candidate);
            candidate[index] = original - step;
            let minus = loss(&candidate);
            candidate[index] = original;
            let numeric = (plus - minus) / (2.0 * step);
            assert!(
                (numeric - analytic[index]).abs() < 2.5e-3,
                "{name}[{index}]: numeric={numeric}, analytic={}",
                analytic[index]
            );
        }
    }

    #[test]
    fn backward_matches_finite_differences_for_input_all_params_and_hf_source() {
        let input: Vec<f32> = vec![
            0.3, -0.7, 1.1, 0.2, 0.8, -0.4, 1.2, -0.3, 0.5, -0.9, 0.6, 1.4, -0.5, 0.7, 1.3, -0.2,
            0.9, -1.1, 0.4, 1.0, -0.6, 0.8, -0.1, 1.5,
        ];
        let hf = vec![0.6, 0.9, 0.8, 0.7];
        let values = FoveationFilmParams {
            w_embed: vec![0.22, -0.13, 0.17, 0.31],
            b_embed: vec![0.8, 0.9],
            w_gamma: vec![0.12, -0.18, 0.21, 0.11],
            b_gamma: vec![0.03, -0.04],
            w_beta: vec![-0.09, 0.15, 0.07, -0.12],
            b_beta: vec![0.05, -0.02],
        };
        let seed: Vec<f32> = (0..input.len())
            .map(|index| (index as f32 - 11.0) * 0.07)
            .collect();
        let (output, cache) = forward_nchw(
            &input, &hf, BATCH, CHANNELS, HEIGHT, WIDTH, PARAMS, EMBED, &values,
        )
        .expect("valid FiLM forward");
        let gradients = backward_nchw(
            &input, &seed, &hf, BATCH, CHANNELS, HEIGHT, WIDTH, PARAMS, EMBED, &values, &cache,
        )
        .expect("valid FiLM backward");
        let base = output.iter().zip(&seed).map(|(x, g)| x * g).sum::<f32>();
        assert!(base.is_finite());

        check_finite_difference("input", &gradients.input, &input, |candidate| {
            objective(&values, candidate, &hf, &seed)
        });
        check_finite_difference(
            "w_embed",
            &gradients.params.w_embed,
            &values.w_embed,
            |candidate| {
                let mut trial = values.clone();
                trial.w_embed.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference(
            "b_embed",
            &gradients.params.b_embed,
            &values.b_embed,
            |candidate| {
                let mut trial = values.clone();
                trial.b_embed.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference(
            "w_gamma",
            &gradients.params.w_gamma,
            &values.w_gamma,
            |candidate| {
                let mut trial = values.clone();
                trial.w_gamma.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference(
            "b_gamma",
            &gradients.params.b_gamma,
            &values.b_gamma,
            |candidate| {
                let mut trial = values.clone();
                trial.b_gamma.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference(
            "w_beta",
            &gradients.params.w_beta,
            &values.w_beta,
            |candidate| {
                let mut trial = values.clone();
                trial.w_beta.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference(
            "b_beta",
            &gradients.params.b_beta,
            &values.b_beta,
            |candidate| {
                let mut trial = values.clone();
                trial.b_beta.copy_from_slice(candidate);
                objective(&trial, &input, &hf, &seed)
            },
        );
        check_finite_difference("hf_params", &gradients.hf_params, &hf, |candidate| {
            objective(&values, &input, candidate, &seed)
        });
    }

    #[test]
    fn invalid_layouts_are_rejected() {
        let values = params();
        assert!(forward_nchw(&[1.0], &[0.0, 1.0], 1, 1, 1, 1, 2, 2, &values).is_err());
    }
}
