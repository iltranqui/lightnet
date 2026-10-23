//! CPU reference for the guarded C++ `[clifford]` layer.
//!
//! The C++ implementation is disabled by `LAYER_CLIFFORD`; this module is
//! hand-derived from `Jev_laya:src-lib/clifford_layer.cpp`. Layout is NCHW and
//! every backward below differentiates this Rust forward.

use crate::error::{Error, Result};
use crate::model::{
    Activation, CliffordContextMode, CliffordGffnMode, CliffordGlobalOp, CliffordHigherMode,
    CliffordInteractionMode, CliffordSpec, ConvolutionalSpec, TensorShape,
};
use crate::runtime::{
    BatchNormParams, BatchNormTrainBuffers, ConvolutionalParams, ConvolutionalTrainBuffers, Tensor,
    WmhfConvRuntimeCache, backward_wmhf_convolution, forward_wmhf_convolution,
};

type CliffordConvRuntimeCache = WmhfConvRuntimeCache;

const LN_EPSILON: f32 = 1.0e-5;
const TWO_PI: f32 = std::f32::consts::TAU;

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordBranchParams {
    /// Row-major `[C, raw_channels]` pointwise projection.
    pub projection_weights: Vec<f32>,
    pub projection_biases: Vec<f32>,
    /// Row-major `[C, 2*C]` projection of concatenated normalized input/features.
    pub gate_weights: Vec<f32>,
    pub gate_biases: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordFourierParams {
    /// Elementwise complex multiplier, `[C, modes_h, modes_w]`.
    pub real: Vec<f32>,
    pub imaginary: Vec<f32>,
}

/// Trainable Clifford weights in their logical form. Serialization order is
/// defined by `Layer::weights` and `weights.rs`, not this struct's field order.
#[derive(Clone, Debug, PartialEq)]
pub struct CliffordParams {
    pub layer_norm_gamma: Vec<f32>,
    pub layer_norm_beta: Vec<f32>,
    /// Row-major `[C,C]` determinant projection.
    pub determinant_weights: Vec<f32>,
    pub determinant_biases: Vec<f32>,
    /// `num_dwconv` ordinary BN depthwise convolutions, in execution order.
    pub depthwise: Vec<ConvolutionalParams>,
    pub local: CliffordBranchParams,
    pub layer_scale: Vec<f32>,
    pub global: Option<CliffordBranchParams>,
    pub fourier: Option<CliffordFourierParams>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordBranchTrainBuffers {
    pub projection_weight_updates: Vec<f32>,
    pub projection_bias_updates: Vec<f32>,
    pub gate_weight_updates: Vec<f32>,
    pub gate_bias_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordFourierTrainBuffers {
    pub real_updates: Vec<f32>,
    pub imaginary_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordTrainBuffers {
    pub layer_norm_gamma_updates: Vec<f32>,
    pub layer_norm_beta_updates: Vec<f32>,
    pub determinant_weight_updates: Vec<f32>,
    pub determinant_bias_updates: Vec<f32>,
    pub depthwise: Vec<ConvolutionalTrainBuffers>,
    pub local: CliffordBranchTrainBuffers,
    pub layer_scale_updates: Vec<f32>,
    pub global: Option<CliffordBranchTrainBuffers>,
    pub fourier: Option<CliffordFourierTrainBuffers>,
}

impl CliffordParams {
    pub fn zeroed(channels: usize, spec: &CliffordSpec) -> Self {
        let kernel_area = spec.dwconv_size * spec.dwconv_size;
        let depthwise = (0..spec.num_dwconv)
            .map(|_| ConvolutionalParams {
                biases: vec![0.0; channels],
                batch_norm: Some(BatchNormParams {
                    biases: vec![0.0; channels],
                    scales: vec![1.0; channels],
                    rolling_mean: vec![0.0; channels],
                    rolling_variance: vec![1.0; channels],
                }),
                weights: vec![0.0; channels * kernel_area],
            })
            .collect();
        let raw = raw_channels(spec, channels);
        let local = zero_branch(channels, raw);
        let global = has_global(spec).then(|| zero_branch(channels, raw));
        let fourier = has_fourier(spec).then(|| {
            let count = channels * spec.fno_modes_h * spec.fno_modes_w;
            CliffordFourierParams {
                real: vec![0.0; count],
                imaginary: vec![0.0; count],
            }
        });
        Self {
            layer_norm_gamma: vec![0.0; channels],
            layer_norm_beta: vec![0.0; channels],
            determinant_weights: vec![0.0; channels * channels],
            determinant_biases: vec![0.0; channels],
            depthwise,
            local,
            layer_scale: vec![0.0; channels],
            global,
            fourier,
        }
    }

    pub(crate) fn zero_gradients_like(&self) -> Self {
        let depthwise = self.depthwise.iter().map(zero_convolutional_like).collect();
        Self {
            layer_norm_gamma: vec![0.0; self.layer_norm_gamma.len()],
            layer_norm_beta: vec![0.0; self.layer_norm_beta.len()],
            determinant_weights: vec![0.0; self.determinant_weights.len()],
            determinant_biases: vec![0.0; self.determinant_biases.len()],
            depthwise,
            local: zero_branch_like(&self.local),
            layer_scale: vec![0.0; self.layer_scale.len()],
            global: self.global.as_ref().map(zero_branch_like),
            fourier: self.fourier.as_ref().map(|weights| CliffordFourierParams {
                real: vec![0.0; weights.real.len()],
                imaginary: vec![0.0; weights.imaginary.len()],
            }),
        }
    }
}

impl CliffordTrainBuffers {
    pub fn zeroed(channels: usize, spec: &CliffordSpec) -> Self {
        let params = CliffordParams::zeroed(channels, spec);
        Self {
            layer_norm_gamma_updates: vec![0.0; channels],
            layer_norm_beta_updates: vec![0.0; channels],
            determinant_weight_updates: vec![0.0; channels * channels],
            determinant_bias_updates: vec![0.0; channels],
            depthwise: params
                .depthwise
                .iter()
                .map(convolutional_train_buffers_like)
                .collect(),
            local: zero_branch_train_like(&params.local),
            layer_scale_updates: vec![0.0; channels],
            global: params.global.as_ref().map(zero_branch_train_like),
            fourier: params
                .fourier
                .as_ref()
                .map(|weights| CliffordFourierTrainBuffers {
                    real_updates: vec![0.0; weights.real.len()],
                    imaginary_updates: vec![0.0; weights.imaginary.len()],
                }),
        }
    }
}

fn zero_branch(channels: usize, raw_channels: usize) -> CliffordBranchParams {
    CliffordBranchParams {
        projection_weights: vec![0.0; channels * raw_channels],
        projection_biases: vec![0.0; channels],
        gate_weights: vec![0.0; 2 * channels * channels],
        gate_biases: vec![0.0; channels],
    }
}

fn zero_branch_like(params: &CliffordBranchParams) -> CliffordBranchParams {
    CliffordBranchParams {
        projection_weights: vec![0.0; params.projection_weights.len()],
        projection_biases: vec![0.0; params.projection_biases.len()],
        gate_weights: vec![0.0; params.gate_weights.len()],
        gate_biases: vec![0.0; params.gate_biases.len()],
    }
}

fn zero_branch_train_like(params: &CliffordBranchParams) -> CliffordBranchTrainBuffers {
    CliffordBranchTrainBuffers {
        projection_weight_updates: vec![0.0; params.projection_weights.len()],
        projection_bias_updates: vec![0.0; params.projection_biases.len()],
        gate_weight_updates: vec![0.0; params.gate_weights.len()],
        gate_bias_updates: vec![0.0; params.gate_biases.len()],
    }
}

fn zero_convolutional_like(params: &ConvolutionalParams) -> ConvolutionalParams {
    ConvolutionalParams {
        biases: vec![0.0; params.biases.len()],
        batch_norm: params.batch_norm.as_ref().map(|bn| BatchNormParams {
            biases: vec![0.0; bn.biases.len()],
            scales: vec![0.0; bn.scales.len()],
            rolling_mean: vec![0.0; bn.rolling_mean.len()],
            rolling_variance: vec![0.0; bn.rolling_variance.len()],
        }),
        weights: vec![0.0; params.weights.len()],
    }
}

fn convolutional_train_buffers_like(params: &ConvolutionalParams) -> ConvolutionalTrainBuffers {
    ConvolutionalTrainBuffers {
        bias_updates: vec![0.0; params.biases.len()],
        batch_norm: params.batch_norm.as_ref().map(|bn| BatchNormTrainBuffers {
            scale_updates: vec![0.0; bn.scales.len()],
            mean_delta: vec![0.0; bn.rolling_mean.len()],
            variance_delta: vec![0.0; bn.rolling_variance.len()],
        }),
        weight_updates: vec![0.0; params.weights.len()],
    }
}

pub(crate) fn raw_channels(spec: &CliffordSpec, channels: usize) -> usize {
    let shifts = match spec.interaction_mode {
        CliffordInteractionMode::Inner => spec.inner_shifts.len(),
        CliffordInteractionMode::Wedge => spec.wedge_shifts.len(),
        CliffordInteractionMode::Full => spec.wedge_shifts.len() + spec.inner_shifts.len(),
    };
    channels * shifts
}

pub(crate) fn has_local(spec: &CliffordSpec) -> bool {
    spec.gffn_mode != CliffordGffnMode::Global
}

pub(crate) fn has_global(spec: &CliffordSpec) -> bool {
    spec.gffn_mode != CliffordGffnMode::Local
}

pub(crate) fn has_fourier(spec: &CliffordSpec) -> bool {
    has_global(spec) && spec.global_op == CliffordGlobalOp::Fourier
}

fn has_higher_local(spec: &CliffordSpec) -> bool {
    has_local(spec)
        && spec.higher_mode == CliffordHigherMode::VectorBivector
        && spec.interaction_mode != CliffordInteractionMode::Inner
}

fn conv_spec(spec: &CliffordSpec, channels: usize) -> ConvolutionalSpec {
    ConvolutionalSpec {
        filters: channels,
        size: spec.dwconv_size,
        stride_x: 1,
        stride_y: 1,
        dilation: 1,
        padding: spec.dwconv_size / 2,
        groups: channels,
        share_index: None,
        batch_normalize: true,
        activation: Activation::Linear,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CliffordLocalCache {
    pub determinant: Tensor,
    pub context_pre_diff: Tensor,
    pub context: Tensor,
    pub depthwise: Vec<CliffordConvRuntimeCache>,
    pub raw: Tensor,
    pub features: Tensor,
    pub gate_logits: Tensor,
    pub gate: Tensor,
    pub higher: Option<Tensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CliffordFourierCache {
    pub coefficients_real: Vec<f32>,
    pub coefficients_imaginary: Vec<f32>,
    pub context: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CliffordGlobalCache {
    pub pooled: Option<Vec<f32>>,
    pub fourier: Option<CliffordFourierCache>,
    pub raw: Tensor,
    pub features: Tensor,
    pub gate_logits: Tensor,
    pub gate: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CliffordRuntimeCache {
    pub input_shape: TensorShape,
    pub normalized: Tensor,
    pub xhat: Vec<f32>,
    pub inv_std: Vec<f32>,
    pub local: Option<CliffordLocalCache>,
    pub global: Option<CliffordGlobalCache>,
    pub mixed: Tensor,
    pub drop_path_mask: Vec<f32>,
    pub training: bool,
}

pub(crate) fn forward(
    spec: &CliffordSpec,
    params: &CliffordParams,
    input: &Tensor,
    training: bool,
    drop_path_mask: Option<&[f32]>,
) -> Result<(Tensor, CliffordRuntimeCache)> {
    validate_params(spec, params, input.shape)?;
    let channels = input.shape.channels;
    let spatial = input.shape.width * input.shape.height;
    let (normalized, xhat, inv_std) =
        layer_norm_forward(input, &params.layer_norm_gamma, &params.layer_norm_beta);

    let local = if has_local(spec) {
        let determinant = linear_forward(
            &normalized,
            &params.determinant_weights,
            &params.determinant_biases,
            channels,
        )?;
        let mut current = normalized.clone();
        let conv = conv_spec(spec, channels);
        let mut depthwise = Vec::with_capacity(spec.num_dwconv);
        for conv_params in &params.depthwise {
            let cache =
                forward_wmhf_convolution(&conv, conv_params, &current, input.shape, training)?;
            current = cache.output.clone();
            depthwise.push(cache);
        }
        let context_pre_diff = current;
        let mut context = context_pre_diff.clone();
        for value in &mut context.data {
            *value = silu(*value);
        }
        if spec.context_mode == CliffordContextMode::Diff {
            for (value, det) in context.data.iter_mut().zip(&determinant.data) {
                *value -= det;
            }
        }
        let raw =
            rolling_local_forward(spec, &determinant, &context, raw_channels(spec, channels))?;
        let features = linear_forward(
            &raw,
            &params.local.projection_weights,
            &params.local.projection_biases,
            channels,
        )?;
        let gate_logits = gate_forward(
            &normalized,
            &features,
            &params.local.gate_weights,
            &params.local.gate_biases,
        )?;
        let mut gate = gate_logits.clone();
        for value in &mut gate.data {
            *value = sigmoid(*value);
        }
        let higher = has_higher_local(spec)
            .then(|| higher_forward(&determinant, &features, &spec.wedge_shifts));
        Some(CliffordLocalCache {
            determinant,
            context_pre_diff,
            context,
            depthwise,
            raw,
            features,
            gate_logits,
            gate,
            higher,
        })
    } else {
        None
    };

    let global = if has_global(spec) {
        let (pooled, fourier_context, fourier_cache) = match spec.global_op {
            CliffordGlobalOp::AvgPool => {
                let pooled = average_pool(&normalized);
                (Some(pooled), None, None)
            }
            CliffordGlobalOp::Fourier => {
                let fourier = params.fourier.as_ref().ok_or_else(|| {
                    Error::InvalidConfig("[clifford] Fourier weights are missing".to_owned())
                })?;
                let (context, cache) = fourier_forward(&normalized, fourier, spec)?;
                (None, Some(context), Some(cache))
            }
        };
        let raw = match spec.global_op {
            CliffordGlobalOp::AvgPool => rolling_global_forward(
                spec,
                &normalized,
                pooled.as_deref().expect("avgpool context"),
                raw_channels(spec, channels),
            )?,
            CliffordGlobalOp::Fourier => rolling_local_forward(
                spec,
                &normalized,
                fourier_context.as_ref().expect("Fourier context"),
                raw_channels(spec, channels),
            )?,
        };
        let params_global = params.global.as_ref().ok_or_else(|| {
            Error::InvalidConfig("[clifford] global branch weights are missing".to_owned())
        })?;
        let features = linear_forward(
            &raw,
            &params_global.projection_weights,
            &params_global.projection_biases,
            channels,
        )?;
        let gate_logits = gate_forward(
            &normalized,
            &features,
            &params_global.gate_weights,
            &params_global.gate_biases,
        )?;
        let mut gate = gate_logits.clone();
        for value in &mut gate.data {
            *value = sigmoid(*value);
        }
        Some(CliffordGlobalCache {
            pooled,
            fourier: fourier_cache,
            raw,
            features,
            gate_logits,
            gate,
        })
    } else {
        None
    };

    let mut mixed = normalized.clone();
    for value in &mut mixed.data {
        *value = silu(*value);
    }
    if let Some(local) = &local {
        for index in 0..mixed.data.len() {
            mixed.data[index] += local.gate.data[index] * local.features.data[index];
            if let Some(higher) = &local.higher {
                mixed.data[index] += higher.data[index];
            }
        }
    }
    if let Some(global) = &global {
        for index in 0..mixed.data.len() {
            mixed.data[index] += global.gate.data[index] * global.features.data[index];
        }
    }

    let mask = if training && spec.drop_path > 0.0 {
        let provided = drop_path_mask.ok_or_else(|| {
            Error::InvalidConfig(
                "[clifford] training DropPath requires a cached sample mask".to_owned(),
            )
        })?;
        if provided.len() != input.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "[clifford] DropPath mask has {} values, expected batch {}",
                provided.len(),
                input.shape.batch
            )));
        }
        let inv_keep = 1.0 / (1.0 - spec.drop_path);
        if provided.iter().any(|value| {
            !value.is_finite() || (*value != 0.0 && (*value - inv_keep).abs() > 1.0e-6)
        }) {
            return Err(Error::InvalidConfig(
                "[clifford] DropPath mask values must be zero or the inverted keep probability"
                    .to_owned(),
            ));
        }
        provided.to_vec()
    } else {
        vec![1.0; input.shape.batch]
    };
    let mut output = input.clone();
    for batch in 0..input.shape.batch {
        for channel in 0..channels {
            let scale = params.layer_scale[channel] * mask[batch];
            let start = (batch * channels + channel) * spatial;
            for position in 0..spatial {
                output.data[start + position] += mixed.data[start + position] * scale;
            }
        }
    }
    let cache = CliffordRuntimeCache {
        input_shape: input.shape,
        normalized,
        xhat,
        inv_std,
        local,
        global,
        mixed,
        drop_path_mask: mask,
        training,
    };
    Ok((output, cache))
}

pub(crate) fn backward(
    spec: &CliffordSpec,
    params: &CliffordParams,
    cache: &CliffordRuntimeCache,
    output_gradient: &Tensor,
) -> Result<(Tensor, CliffordParams)> {
    validate_params(spec, params, cache.input_shape)?;
    if output_gradient.shape != cache.input_shape {
        return Err(Error::InvalidConfig(format!(
            "[clifford] backward gradient shape mismatch: expected {}, got {}",
            cache.input_shape, output_gradient.shape
        )));
    }
    let shape = cache.input_shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let mut gradients = params.zero_gradients_like();
    let mut input_gradient = output_gradient.clone();
    let mut d_normalized = Tensor::zeros(shape);
    let mut d_mixed = Tensor::zeros(shape);
    for batch in 0..shape.batch {
        for channel in 0..channels {
            let start = (batch * channels + channel) * spatial;
            let mask = cache.drop_path_mask[batch];
            let scale = params.layer_scale[channel];
            for position in 0..spatial {
                let index = start + position;
                let dy = output_gradient.data[index];
                gradients.layer_scale[channel] += dy * mask * cache.mixed.data[index];
                d_mixed.data[index] = dy * mask * scale;
                let x = cache.normalized.data[index];
                let sig = sigmoid(x);
                d_normalized.data[index] += d_mixed.data[index] * (sig + x * sig * (1.0 - sig));
            }
        }
    }

    if let Some(local) = &cache.local {
        let local_grad = &mut gradients.local;
        let mut d_determinant = Tensor::zeros(shape);
        let mut d_features = Tensor::zeros(shape);
        let mut d_gate_logits = Tensor::zeros(shape);
        for index in 0..d_features.data.len() {
            let alpha = local.gate.data[index];
            d_features.data[index] += d_mixed.data[index] * alpha;
            d_gate_logits.data[index] =
                d_mixed.data[index] * local.features.data[index] * alpha * (1.0 - alpha);
        }
        if local.higher.is_some() {
            higher_backward(
                &local.determinant,
                &local.features,
                &spec.wedge_shifts,
                &d_mixed,
                &mut d_determinant,
                &mut d_features,
            );
        }
        let (gate_input_gradient, gate_feature_gradient) = gate_backward(
            &cache.normalized,
            &local.features,
            &d_gate_logits,
            &params.local.gate_weights,
            &mut local_grad.gate_weights,
            &mut local_grad.gate_biases,
        );
        add_tensor(&mut d_normalized, &gate_input_gradient);
        add_tensor(&mut d_features, &gate_feature_gradient);
        let mut d_raw = Tensor::zeros(local.raw.shape);
        let projection_input_gradient = linear_backward(
            &local.raw,
            &d_features,
            &params.local.projection_weights,
            &mut local_grad.projection_weights,
            &mut local_grad.projection_biases,
        )?;
        d_raw.data.copy_from_slice(&projection_input_gradient.data);
        let mut d_context = Tensor::zeros(shape);
        rolling_local_backward(
            spec,
            &local.determinant,
            &local.context,
            &d_raw,
            &mut d_determinant,
            &mut d_context,
        );
        if spec.context_mode == CliffordContextMode::Diff {
            for (d_det, d_ctx) in d_determinant.data.iter_mut().zip(&d_context.data) {
                *d_det -= d_ctx;
            }
        }
        for (gradient, pre) in d_context.data.iter_mut().zip(&local.context_pre_diff.data) {
            let sig = sigmoid(*pre);
            *gradient *= sig + *pre * sig * (1.0 - sig);
        }
        let conv = conv_spec(spec, channels);
        let mut conv_gradient = d_context;
        for index in (0..spec.num_dwconv).rev() {
            let (previous_gradient, param_gradient) = backward_wmhf_convolution(
                &conv,
                &params.depthwise[index],
                &local.depthwise[index],
                &conv_gradient,
                cache.training,
            )?;
            gradients.depthwise[index] = param_gradient;
            conv_gradient = previous_gradient;
        }
        add_tensor(&mut d_normalized, &conv_gradient);
        let det_input_gradient = linear_backward(
            &cache.normalized,
            &d_determinant,
            &params.determinant_weights,
            &mut gradients.determinant_weights,
            &mut gradients.determinant_biases,
        )?;
        add_tensor(&mut d_normalized, &det_input_gradient);
    }

    if let (Some(global), Some(global_grad), Some(global_params)) =
        (&cache.global, &mut gradients.global, &params.global)
    {
        let mut d_features = Tensor::zeros(shape);
        let mut d_gate_logits = Tensor::zeros(shape);
        for index in 0..d_features.data.len() {
            let alpha = global.gate.data[index];
            d_features.data[index] += d_mixed.data[index] * alpha;
            d_gate_logits.data[index] =
                d_mixed.data[index] * global.features.data[index] * alpha * (1.0 - alpha);
        }
        let (gate_input_gradient, gate_feature_gradient) = gate_backward(
            &cache.normalized,
            &global.features,
            &d_gate_logits,
            &global_params.gate_weights,
            &mut global_grad.gate_weights,
            &mut global_grad.gate_biases,
        );
        add_tensor(&mut d_normalized, &gate_input_gradient);
        add_tensor(&mut d_features, &gate_feature_gradient);
        let d_raw = linear_backward(
            &global.raw,
            &d_features,
            &global_params.projection_weights,
            &mut global_grad.projection_weights,
            &mut global_grad.projection_biases,
        )?;
        match spec.global_op {
            CliffordGlobalOp::AvgPool => {
                let pooled = global.pooled.as_ref().expect("avgpool cache");
                let (mut d_state, d_pooled) =
                    rolling_global_backward(spec, &cache.normalized, pooled, &d_raw);
                let scale = 1.0 / spatial as f32;
                for batch in 0..shape.batch {
                    for channel in 0..channels {
                        let grad = d_pooled[batch * channels + channel] * scale;
                        let start = (batch * channels + channel) * spatial;
                        for position in 0..spatial {
                            d_state.data[start + position] += grad;
                        }
                    }
                }
                add_tensor(&mut d_normalized, &d_state);
            }
            CliffordGlobalOp::Fourier => {
                let fourier_cache = global.fourier.as_ref().expect("Fourier cache");
                let fourier_params = params.fourier.as_ref().expect("Fourier params");
                let fourier_grad = gradients.fourier.as_mut().expect("Fourier gradients");
                let mut d_fourier_context = Tensor::zeros(shape);
                rolling_local_backward(
                    spec,
                    &cache.normalized,
                    &fourier_cache.context,
                    &d_raw,
                    &mut d_normalized,
                    &mut d_fourier_context,
                );
                fourier_backward(
                    spec,
                    &fourier_cache.coefficients_real,
                    &fourier_cache.coefficients_imaginary,
                    fourier_params,
                    &d_fourier_context,
                    &mut d_normalized,
                    &mut fourier_grad.real,
                    &mut fourier_grad.imaginary,
                );
            }
        }
    }

    let (layer_norm_input_gradient, gamma_gradient, beta_gradient) = layer_norm_backward(
        &d_normalized,
        &cache.xhat,
        &cache.inv_std,
        &params.layer_norm_gamma,
    );
    add_tensor(&mut input_gradient, &layer_norm_input_gradient);
    gradients.layer_norm_gamma = gamma_gradient;
    gradients.layer_norm_beta = beta_gradient;
    Ok((input_gradient, gradients))
}

fn validate_params(spec: &CliffordSpec, params: &CliffordParams, shape: TensorShape) -> Result<()> {
    let channels = shape.channels;
    let raw = raw_channels(spec, channels);
    let bad = |name: &str| {
        Error::InvalidConfig(format!(
            "[clifford] parameter `{name}` has an invalid shape"
        ))
    };
    if params.layer_norm_gamma.len() != channels || params.layer_norm_beta.len() != channels {
        return Err(bad("layer_norm"));
    }
    if params.determinant_weights.len() != channels * channels
        || params.determinant_biases.len() != channels
        || params.depthwise.len() != spec.num_dwconv
    {
        return Err(bad("determinant/depthwise"));
    }
    for conv in &params.depthwise {
        if conv.biases.len() != channels
            || conv.weights.len() != channels * spec.dwconv_size * spec.dwconv_size
            || conv.batch_norm.as_ref().is_none_or(|bn| {
                bn.biases.len() != channels
                    || bn.scales.len() != channels
                    || bn.rolling_mean.len() != channels
                    || bn.rolling_variance.len() != channels
            })
        {
            return Err(bad("depthwise"));
        }
    }
    if !valid_branch(&params.local, channels, raw) || params.layer_scale.len() != channels {
        return Err(bad("local/layer_scale"));
    }
    if params.global.is_some() != has_global(spec) {
        return Err(bad("global"));
    }
    if params
        .global
        .as_ref()
        .is_some_and(|branch| !valid_branch(branch, channels, raw))
    {
        return Err(bad("global"));
    }
    if params.fourier.is_some() != has_fourier(spec) {
        return Err(bad("Fourier"));
    }
    if params.fourier.as_ref().is_some_and(|fno| {
        let expected = channels * spec.fno_modes_h * spec.fno_modes_w;
        fno.real.len() != expected || fno.imaginary.len() != expected
    }) {
        return Err(bad("Fourier"));
    }
    if shape.batch == 0 || shape.channels == 0 || shape.height == 0 || shape.width == 0 {
        return Err(Error::InvalidConfig(
            "[clifford] input must be a non-empty NCHW image".to_owned(),
        ));
    }
    Ok(())
}

fn valid_branch(branch: &CliffordBranchParams, channels: usize, raw: usize) -> bool {
    branch.projection_weights.len() == channels * raw
        && branch.projection_biases.len() == channels
        && branch.gate_weights.len() == 2 * channels * channels
        && branch.gate_biases.len() == channels
}

fn layer_norm_forward(input: &Tensor, gamma: &[f32], beta: &[f32]) -> (Tensor, Vec<f32>, Vec<f32>) {
    let shape = input.shape;
    let spatial = shape.width * shape.height;
    let mut normalized = Tensor::zeros(shape);
    let mut xhat = vec![0.0; input.data.len()];
    let mut inv_std = vec![0.0; shape.batch * spatial];
    for batch in 0..shape.batch {
        for position in 0..spatial {
            let mut mean = 0.0;
            for channel in 0..shape.channels {
                mean += input.data[(batch * shape.channels + channel) * spatial + position];
            }
            mean /= shape.channels as f32;
            let mut variance = 0.0;
            for channel in 0..shape.channels {
                let value =
                    input.data[(batch * shape.channels + channel) * spatial + position] - mean;
                variance += value * value;
            }
            variance /= shape.channels as f32;
            let inv = 1.0 / (variance + LN_EPSILON).sqrt();
            inv_std[batch * spatial + position] = inv;
            for channel in 0..shape.channels {
                let index = (batch * shape.channels + channel) * spatial + position;
                let value = (input.data[index] - mean) * inv;
                xhat[index] = value;
                normalized.data[index] = value * gamma[channel] + beta[channel];
            }
        }
    }
    (normalized, xhat, inv_std)
}

fn layer_norm_backward(
    output_gradient: &Tensor,
    xhat: &[f32],
    inv_std: &[f32],
    gamma: &[f32],
) -> (Tensor, Vec<f32>, Vec<f32>) {
    let shape = output_gradient.shape;
    let spatial = shape.width * shape.height;
    let mut input_gradient = Tensor::zeros(shape);
    let mut gamma_gradient = vec![0.0; shape.channels];
    let mut beta_gradient = vec![0.0; shape.channels];
    for batch in 0..shape.batch {
        for position in 0..spatial {
            let mut sum = 0.0;
            let mut sum_xhat = 0.0;
            for channel in 0..shape.channels {
                let index = (batch * shape.channels + channel) * spatial + position;
                let dy = output_gradient.data[index];
                let dxhat = dy * gamma[channel];
                sum += dxhat;
                sum_xhat += dxhat * xhat[index];
                gamma_gradient[channel] += dy * xhat[index];
                beta_gradient[channel] += dy;
            }
            let scale = inv_std[batch * spatial + position] / shape.channels as f32;
            for channel in 0..shape.channels {
                let index = (batch * shape.channels + channel) * spatial + position;
                let dxhat = output_gradient.data[index] * gamma[channel];
                input_gradient.data[index] =
                    scale * (shape.channels as f32 * dxhat - sum - xhat[index] * sum_xhat);
            }
        }
    }
    (input_gradient, gamma_gradient, beta_gradient)
}

fn linear_forward(
    input: &Tensor,
    weights: &[f32],
    biases: &[f32],
    outputs: usize,
) -> Result<Tensor> {
    let inputs = input.shape.channels;
    if weights.len() != outputs * inputs || biases.len() != outputs {
        return Err(Error::InvalidConfig(
            "[clifford] linear parameter shape mismatch".to_owned(),
        ));
    }
    let shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        outputs,
        input.shape.batch,
    );
    let spatial = input.shape.width * input.shape.height;
    let mut output = Tensor::zeros(shape);
    for batch in 0..shape.batch {
        for out in 0..outputs {
            let out_start = (batch * outputs + out) * spatial;
            output.data[out_start..out_start + spatial].fill(biases[out]);
            for channel in 0..inputs {
                let weight = weights[out * inputs + channel];
                let input_start = (batch * inputs + channel) * spatial;
                for position in 0..spatial {
                    output.data[out_start + position] +=
                        weight * input.data[input_start + position];
                }
            }
        }
    }
    Ok(output)
}

fn linear_backward(
    input: &Tensor,
    output_gradient: &Tensor,
    weights: &[f32],
    weight_gradient: &mut [f32],
    bias_gradient: &mut [f32],
) -> Result<Tensor> {
    let outputs = output_gradient.shape.channels;
    let inputs = input.shape.channels;
    if output_gradient.shape.width != input.shape.width
        || output_gradient.shape.height != input.shape.height
        || output_gradient.shape.batch != input.shape.batch
        || weights.len() != outputs * inputs
        || weight_gradient.len() != weights.len()
        || bias_gradient.len() != outputs
    {
        return Err(Error::InvalidConfig(
            "[clifford] linear backward shape mismatch".to_owned(),
        ));
    }
    let spatial = input.shape.width * input.shape.height;
    let mut input_gradient = Tensor::zeros(input.shape);
    for batch in 0..input.shape.batch {
        for out in 0..outputs {
            let dy_start = (batch * outputs + out) * spatial;
            for position in 0..spatial {
                bias_gradient[out] += output_gradient.data[dy_start + position];
            }
            for channel in 0..inputs {
                let input_start = (batch * inputs + channel) * spatial;
                let weight_index = out * inputs + channel;
                for position in 0..spatial {
                    let dy = output_gradient.data[dy_start + position];
                    weight_gradient[weight_index] += dy * input.data[input_start + position];
                    input_gradient.data[input_start + position] += weights[weight_index] * dy;
                }
            }
        }
    }
    Ok(input_gradient)
}

fn gate_forward(
    input: &Tensor,
    features: &Tensor,
    weights: &[f32],
    biases: &[f32],
) -> Result<Tensor> {
    if input.shape != features.shape {
        return Err(Error::InvalidConfig(
            "[clifford] gate input shape mismatch".to_owned(),
        ));
    }
    let channels = input.shape.channels;
    if weights.len() != 2 * channels * channels || biases.len() != channels {
        return Err(Error::InvalidConfig(
            "[clifford] gate parameter shape mismatch".to_owned(),
        ));
    }
    let spatial = input.shape.width * input.shape.height;
    let mut output = Tensor::zeros(input.shape);
    for batch in 0..input.shape.batch {
        for out in 0..channels {
            let out_start = (batch * channels + out) * spatial;
            output.data[out_start..out_start + spatial].fill(biases[out]);
            for channel in 0..channels {
                let input_start = (batch * channels + channel) * spatial;
                let wx = weights[out * 2 * channels + channel];
                let wf = weights[out * 2 * channels + channels + channel];
                for position in 0..spatial {
                    output.data[out_start + position] += wx * input.data[input_start + position]
                        + wf * features.data[input_start + position];
                }
            }
        }
    }
    Ok(output)
}

fn gate_backward(
    input: &Tensor,
    features: &Tensor,
    gate_gradient: &Tensor,
    weights: &[f32],
    weight_updates: &mut [f32],
    bias_updates: &mut [f32],
) -> (Tensor, Tensor) {
    let shape = input.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let mut input_gradient = Tensor::zeros(shape);
    let mut feature_gradient = Tensor::zeros(shape);
    for batch in 0..shape.batch {
        for out in 0..channels {
            let dy_start = (batch * channels + out) * spatial;
            for position in 0..spatial {
                bias_updates[out] += gate_gradient.data[dy_start + position];
            }
            for channel in 0..channels {
                let input_start = (batch * channels + channel) * spatial;
                let wx_index = out * 2 * channels + channel;
                let wf_index = wx_index + channels;
                for position in 0..spatial {
                    let dy = gate_gradient.data[dy_start + position];
                    weight_updates[wx_index] += dy * input.data[input_start + position];
                    weight_updates[wf_index] += dy * features.data[input_start + position];
                    input_gradient.data[input_start + position] += weights[wx_index] * dy;
                    feature_gradient.data[input_start + position] += weights[wf_index] * dy;
                }
            }
        }
    }
    (input_gradient, feature_gradient)
}

fn rolling_local_forward(
    spec: &CliffordSpec,
    determinant: &Tensor,
    context: &Tensor,
    raw_channels: usize,
) -> Result<Tensor> {
    rolling_forward(spec, determinant, Some(context), None, raw_channels)
}

fn rolling_global_forward(
    spec: &CliffordSpec,
    state: &Tensor,
    pooled: &[f32],
    raw_channels: usize,
) -> Result<Tensor> {
    rolling_forward(spec, state, None, Some(pooled), raw_channels)
}

fn rolling_forward(
    spec: &CliffordSpec,
    state: &Tensor,
    local_context: Option<&Tensor>,
    global_context: Option<&[f32]>,
    raw_channels: usize,
) -> Result<Tensor> {
    let shape = state.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    if raw_channels != raw_channels_for_spec(spec, channels) {
        return Err(Error::InvalidConfig(
            "[clifford] raw interaction layout mismatch".to_owned(),
        ));
    }
    let mut output = Tensor::zeros(TensorShape::new(
        shape.width,
        shape.height,
        raw_channels,
        shape.batch,
    ));
    let wedge_on = spec.interaction_mode != CliffordInteractionMode::Inner;
    let inner_on = spec.interaction_mode != CliffordInteractionMode::Wedge;
    let wedge_count = if wedge_on { spec.wedge_shifts.len() } else { 0 };
    let inner_count = if inner_on { spec.inner_shifts.len() } else { 0 };
    for batch in 0..shape.batch {
        for (shift_index, &shift) in spec.wedge_shifts.iter().enumerate().take(wedge_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Wedge => shift_index,
                CliffordInteractionMode::Full => wedge_base(shift_index, inner_count),
                CliffordInteractionMode::Inner => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let current_start = (batch * channels + channel) * spatial;
                let rolled_start = (batch * channels + rolled) * spatial;
                let out_start = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let a = state.data[current_start + position];
                    let b = state.data[rolled_start + position];
                    let (ctx_a, ctx_b) = if let Some(ctx) = local_context {
                        (
                            ctx.data[current_start + position],
                            ctx.data[rolled_start + position],
                        )
                    } else if let Some(ctx) = global_context {
                        (
                            ctx[batch * channels + channel],
                            ctx[batch * channels + rolled],
                        )
                    } else {
                        unreachable!()
                    };
                    output.data[out_start + position] = a * ctx_b - ctx_a * b;
                }
            }
        }
        for (shift_index, &shift) in spec.inner_shifts.iter().enumerate().take(inner_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Inner => shift_index,
                CliffordInteractionMode::Full => inner_base(shift_index, wedge_count),
                CliffordInteractionMode::Wedge => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let current_start = (batch * channels + channel) * spatial;
                let rolled_start = (batch * channels + rolled) * spatial;
                let out_start = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let a = state.data[current_start + position];
                    let b = if let Some(ctx) = local_context {
                        ctx.data[rolled_start + position]
                    } else {
                        global_context.expect("global context")[batch * channels + rolled]
                    };
                    output.data[out_start + position] = silu(a * b);
                }
            }
        }
    }
    Ok(output)
}

fn raw_channels_for_spec(spec: &CliffordSpec, channels: usize) -> usize {
    raw_channels(spec, channels)
}

fn wedge_base(shift_index: usize, inner_count: usize) -> usize {
    if shift_index < inner_count {
        2 * shift_index
    } else {
        inner_count + shift_index
    }
}

fn inner_base(shift_index: usize, wedge_count: usize) -> usize {
    if shift_index < wedge_count {
        2 * shift_index + 1
    } else {
        wedge_count + shift_index
    }
}

fn rolling_local_backward(
    spec: &CliffordSpec,
    state: &Tensor,
    context: &Tensor,
    raw_gradient: &Tensor,
    state_gradient: &mut Tensor,
    context_gradient: &mut Tensor,
) {
    let shape = state.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let raw_channels = raw_gradient.shape.channels;
    let wedge_on = spec.interaction_mode != CliffordInteractionMode::Inner;
    let inner_on = spec.interaction_mode != CliffordInteractionMode::Wedge;
    let wedge_count = if wedge_on { spec.wedge_shifts.len() } else { 0 };
    let inner_count = if inner_on { spec.inner_shifts.len() } else { 0 };
    for batch in 0..shape.batch {
        for (shift_index, &shift) in spec.wedge_shifts.iter().enumerate().take(wedge_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Wedge => shift_index,
                CliffordInteractionMode::Full => wedge_base(shift_index, inner_count),
                CliffordInteractionMode::Inner => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let rot = (batch * channels + rolled) * spatial;
                let out = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let dy = raw_gradient.data[out + position];
                    state_gradient.data[cur + position] += dy * context.data[rot + position];
                    context_gradient.data[rot + position] += dy * state.data[cur + position];
                    context_gradient.data[cur + position] -= dy * state.data[rot + position];
                    state_gradient.data[rot + position] -= dy * context.data[cur + position];
                }
            }
        }
        for (shift_index, &shift) in spec.inner_shifts.iter().enumerate().take(inner_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Inner => shift_index,
                CliffordInteractionMode::Full => inner_base(shift_index, wedge_count),
                CliffordInteractionMode::Wedge => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let rot = (batch * channels + rolled) * spatial;
                let out = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let a = state.data[cur + position];
                    let b = context.data[rot + position];
                    let product = a * b;
                    let sig = sigmoid(product);
                    let d_product =
                        raw_gradient.data[out + position] * (sig + product * sig * (1.0 - sig));
                    state_gradient.data[cur + position] += d_product * b;
                    context_gradient.data[rot + position] += d_product * a;
                }
            }
        }
    }
}

fn rolling_global_backward(
    spec: &CliffordSpec,
    state: &Tensor,
    global_context: &[f32],
    raw_gradient: &Tensor,
) -> (Tensor, Vec<f32>) {
    let shape = state.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let raw_channels = raw_gradient.shape.channels;
    let mut state_gradient = Tensor::zeros(shape);
    let mut context_gradient = vec![0.0; shape.batch * channels];
    let wedge_on = spec.interaction_mode != CliffordInteractionMode::Inner;
    let inner_on = spec.interaction_mode != CliffordInteractionMode::Wedge;
    let wedge_count = if wedge_on { spec.wedge_shifts.len() } else { 0 };
    let inner_count = if inner_on { spec.inner_shifts.len() } else { 0 };
    for batch in 0..shape.batch {
        for (shift_index, &shift) in spec.wedge_shifts.iter().enumerate().take(wedge_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Wedge => shift_index,
                CliffordInteractionMode::Full => wedge_base(shift_index, inner_count),
                CliffordInteractionMode::Inner => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let rot = (batch * channels + rolled) * spatial;
                let out = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let dy = raw_gradient.data[out + position];
                    state_gradient.data[cur + position] +=
                        dy * global_context[batch * channels + rolled];
                    context_gradient[batch * channels + rolled] += dy * state.data[cur + position];
                    context_gradient[batch * channels + channel] -= dy * state.data[rot + position];
                    state_gradient.data[rot + position] -=
                        dy * global_context[batch * channels + channel];
                }
            }
        }
        for (shift_index, &shift) in spec.inner_shifts.iter().enumerate().take(inner_count) {
            let block = match spec.interaction_mode {
                CliffordInteractionMode::Inner => shift_index,
                CliffordInteractionMode::Full => inner_base(shift_index, wedge_count),
                CliffordInteractionMode::Wedge => unreachable!(),
            };
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let out = (batch * raw_channels + block * channels + channel) * spatial;
                for position in 0..spatial {
                    let a = state.data[cur + position];
                    let b = global_context[batch * channels + rolled];
                    let product = a * b;
                    let sig = sigmoid(product);
                    let d_product =
                        raw_gradient.data[out + position] * (sig + product * sig * (1.0 - sig));
                    state_gradient.data[cur + position] += d_product * b;
                    context_gradient[batch * channels + rolled] += d_product * a;
                }
            }
        }
    }
    (state_gradient, context_gradient)
}

fn average_pool(input: &Tensor) -> Vec<f32> {
    let channels = input.shape.channels;
    let spatial = input.shape.width * input.shape.height;
    let mut pooled = vec![0.0; input.shape.batch * channels];
    for batch in 0..input.shape.batch {
        for channel in 0..channels {
            let start = (batch * channels + channel) * spatial;
            let sum: f32 = input.data[start..start + spatial].iter().sum();
            pooled[batch * channels + channel] = sum / spatial as f32;
        }
    }
    pooled
}

fn higher_forward(determinant: &Tensor, features: &Tensor, shifts: &[usize]) -> Tensor {
    let shape = determinant.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let mut output = Tensor::zeros(shape);
    if shifts.is_empty() {
        return output;
    }
    let scale = 1.0 / shifts.len() as f32;
    for batch in 0..shape.batch {
        for &shift in shifts {
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let rot = (batch * channels + rolled) * spatial;
                for position in 0..spatial {
                    output.data[cur + position] += scale
                        * (determinant.data[cur + position] * features.data[rot + position]
                            - features.data[cur + position] * determinant.data[rot + position]);
                }
            }
        }
    }
    output
}

fn higher_backward(
    determinant: &Tensor,
    features: &Tensor,
    shifts: &[usize],
    output_gradient: &Tensor,
    determinant_gradient: &mut Tensor,
    feature_gradient: &mut Tensor,
) {
    if shifts.is_empty() {
        return;
    }
    let shape = determinant.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let scale = 1.0 / shifts.len() as f32;
    for batch in 0..shape.batch {
        for &shift in shifts {
            for channel in 0..channels {
                let rolled = (channel + shift) % channels;
                let cur = (batch * channels + channel) * spatial;
                let rot = (batch * channels + rolled) * spatial;
                for position in 0..spatial {
                    let grad = output_gradient.data[cur + position] * scale;
                    determinant_gradient.data[cur + position] +=
                        grad * features.data[rot + position];
                    feature_gradient.data[rot + position] +=
                        grad * determinant.data[cur + position];
                    feature_gradient.data[cur + position] -=
                        grad * determinant.data[rot + position];
                    determinant_gradient.data[rot + position] -=
                        grad * features.data[cur + position];
                }
            }
        }
    }
}

fn fourier_forward(
    input: &Tensor,
    params: &CliffordFourierParams,
    spec: &CliffordSpec,
) -> Result<(Tensor, CliffordFourierCache)> {
    let shape = input.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let modes = spec.fno_modes_h * spec.fno_modes_w;
    if params.real.len() != channels * modes || params.imaginary.len() != channels * modes {
        return Err(Error::InvalidConfig(
            "[clifford] Fourier parameter shape mismatch".to_owned(),
        ));
    }
    let mut coefficients_real = vec![0.0; shape.batch * channels * modes];
    let mut coefficients_imaginary = vec![0.0; coefficients_real.len()];
    let mut context = Tensor::zeros(shape);
    let inv_spatial = 1.0 / spatial as f32;
    for batch in 0..shape.batch {
        for channel in 0..channels {
            let input_start = (batch * channels + channel) * spatial;
            let coefficient_start = (batch * channels + channel) * modes;
            let weight_start = channel * modes;
            for kh in 0..spec.fno_modes_h {
                for kw in 0..spec.fno_modes_w {
                    let mode = kh * spec.fno_modes_w + kw;
                    let mut re = 0.0;
                    let mut im = 0.0;
                    for y in 0..shape.height {
                        for x in 0..shape.width {
                            let theta = TWO_PI
                                * ((kh * y) as f32 / shape.height as f32
                                    + (kw * x) as f32 / shape.width as f32);
                            let value = input.data[input_start + y * shape.width + x];
                            re += value * theta.cos();
                            im -= value * theta.sin();
                        }
                    }
                    coefficients_real[coefficient_start + mode] = re;
                    coefficients_imaginary[coefficient_start + mode] = im;
                    let y_re = re * params.real[weight_start + mode]
                        - im * params.imaginary[weight_start + mode];
                    let y_im = re * params.imaginary[weight_start + mode]
                        + im * params.real[weight_start + mode];
                    let multiplicity = if kw == 0 { 1.0 } else { 2.0 };
                    for y in 0..shape.height {
                        for x in 0..shape.width {
                            let theta = TWO_PI
                                * ((kh * y) as f32 / shape.height as f32
                                    + (kw * x) as f32 / shape.width as f32);
                            let index = input_start + y * shape.width + x;
                            context.data[index] += multiplicity
                                * (y_re * theta.cos() - y_im * theta.sin())
                                * inv_spatial;
                        }
                    }
                }
            }
        }
    }
    let cache = CliffordFourierCache {
        coefficients_real,
        coefficients_imaginary,
        context: context.clone(),
    };
    Ok((context, cache))
}

fn fourier_backward(
    spec: &CliffordSpec,
    coefficients_real: &[f32],
    coefficients_imaginary: &[f32],
    params: &CliffordFourierParams,
    output_gradient: &Tensor,
    input_gradient: &mut Tensor,
    real_gradient: &mut [f32],
    imaginary_gradient: &mut [f32],
) {
    let shape = output_gradient.shape;
    let channels = shape.channels;
    let spatial = shape.width * shape.height;
    let modes = spec.fno_modes_h * spec.fno_modes_w;
    let inv_spatial = 1.0 / spatial as f32;
    for batch in 0..shape.batch {
        for channel in 0..channels {
            let start = (batch * channels + channel) * spatial;
            let coefficient_start = (batch * channels + channel) * modes;
            let weight_start = channel * modes;
            for kh in 0..spec.fno_modes_h {
                for kw in 0..spec.fno_modes_w {
                    let mode = kh * spec.fno_modes_w + kw;
                    let mut d_re = 0.0;
                    let mut d_im = 0.0;
                    for y in 0..shape.height {
                        for x in 0..shape.width {
                            let theta = TWO_PI
                                * ((kh * y) as f32 / shape.height as f32
                                    + (kw * x) as f32 / shape.width as f32);
                            let grad = output_gradient.data[start + y * shape.width + x];
                            d_re += grad * theta.cos();
                            d_im -= grad * theta.sin();
                        }
                    }
                    let multiplicity = if kw == 0 { 1.0 } else { 2.0 };
                    let d_y_re = d_re * multiplicity * inv_spatial;
                    let d_y_im = d_im * multiplicity * inv_spatial;
                    let re = coefficients_real[coefficient_start + mode];
                    let im = coefficients_imaginary[coefficient_start + mode];
                    let w_re = params.real[weight_start + mode];
                    let w_im = params.imaginary[weight_start + mode];
                    real_gradient[weight_start + mode] += d_y_re * re + d_y_im * im;
                    imaginary_gradient[weight_start + mode] += -d_y_re * im + d_y_im * re;
                    let d_coef_re = d_y_re * w_re + d_y_im * w_im;
                    let d_coef_im = -d_y_re * w_im + d_y_im * w_re;
                    for y in 0..shape.height {
                        for x in 0..shape.width {
                            let theta = TWO_PI
                                * ((kh * y) as f32 / shape.height as f32
                                    + (kw * x) as f32 / shape.width as f32);
                            input_gradient.data[start + y * shape.width + x] +=
                                d_coef_re * theta.cos() - d_coef_im * theta.sin();
                        }
                    }
                }
            }
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn add_tensor(destination: &mut Tensor, source: &Tensor) {
    debug_assert_eq!(destination.shape, source.shape);
    for (dst, src) in destination.data.iter_mut().zip(&source.data) {
        *dst += src;
    }
}

#[cfg(test)]
mod tests {
    use crate::cfg::ConfigFile;
    use crate::model::{CliffordSpec, LayerKind, Network};

    use super::{CliffordParams, Tensor, backward, forward, layer_norm_forward};

    fn build_spec(
        global_op: &str,
        num_dwconv: usize,
        interaction_mode: &str,
        context_mode: &str,
        gffn_mode: &str,
        higher_mode: &str,
        drop_path: f32,
    ) -> CliffordSpec {
        let cfg = ConfigFile::parse_str(
            "clifford-finite-difference.cfg",
            &format!(
                r#"[net]
width=2
height=1
channels=2
batch=2
[clifford]
filters=2
shifts=1
ctx_mode={context_mode}
cli_mode={interaction_mode}
gffn_mode={gffn_mode}
cli_higher_mode={higher_mode}
global_op={global_op}
fno_modes_h=1
fno_modes_w=2
dwconv_size=1
num_dwconv={num_dwconv}
activation=swish
drop_path={drop_path}
"#
            ),
        )
        .expect("valid Clifford cfg");
        let network = Network::from_config(cfg).expect("valid Clifford network");
        let LayerKind::Clifford(spec) = &network.layers[0].kind else {
            panic!("expected [clifford]");
        };
        spec.clone()
    }

    fn fill(values: &mut [f32], center: f32, scale: f32) {
        let midpoint = (values.len() as f32 - 1.0) * 0.5;
        for (index, value) in values.iter_mut().enumerate() {
            *value = center + (index as f32 - midpoint) * scale;
        }
    }

    fn test_params(spec: &CliffordSpec) -> CliffordParams {
        let mut params = CliffordParams::zeroed(2, spec);
        fill(&mut params.layer_norm_gamma, 1.0, 0.03);
        fill(&mut params.layer_norm_beta, 0.01, 0.02);
        fill(&mut params.determinant_weights, 0.08, 0.03);
        fill(&mut params.determinant_biases, -0.02, 0.01);
        for (slot, conv) in params.depthwise.iter_mut().enumerate() {
            fill(&mut conv.biases, 0.0, 0.01);
            fill(&mut conv.weights, 0.07 + slot as f32 * 0.01, 0.04);
            let bn = conv.batch_norm.as_mut().expect("depthwise BN");
            fill(&mut bn.scales, 0.9, 0.05);
            fill(&mut bn.biases, 0.0, 0.01);
            fill(&mut bn.rolling_mean, 0.0, 0.01);
            fill(&mut bn.rolling_variance, 1.0, 0.1);
        }
        fill(&mut params.local.projection_weights, 0.06, 0.02);
        fill(&mut params.local.projection_biases, 0.01, 0.01);
        fill(&mut params.local.gate_weights, 0.04, 0.015);
        fill(&mut params.local.gate_biases, -0.03, 0.01);
        fill(&mut params.layer_scale, 0.2, 0.02);
        if let Some(global) = &mut params.global {
            fill(&mut global.projection_weights, -0.05, 0.02);
            fill(&mut global.projection_biases, 0.02, 0.01);
            fill(&mut global.gate_weights, -0.03, 0.015);
            fill(&mut global.gate_biases, 0.04, 0.01);
        }
        if let Some(fourier) = &mut params.fourier {
            fill(&mut fourier.real, 0.3, 0.05);
            fill(&mut fourier.imaginary, -0.12, 0.04);
        }
        params
    }

    fn for_each_param_vector(params: &CliffordParams, mut visit: impl FnMut(&[f32])) {
        visit(&params.layer_norm_gamma);
        visit(&params.layer_norm_beta);
        visit(&params.determinant_weights);
        visit(&params.determinant_biases);
        for conv in &params.depthwise {
            visit(&conv.biases);
            if let Some(bn) = &conv.batch_norm {
                visit(&bn.scales);
            }
            visit(&conv.weights);
        }
        visit(&params.local.projection_weights);
        visit(&params.local.projection_biases);
        visit(&params.local.gate_weights);
        visit(&params.local.gate_biases);
        visit(&params.layer_scale);
        if let Some(global) = &params.global {
            visit(&global.projection_weights);
            visit(&global.projection_biases);
            visit(&global.gate_weights);
            visit(&global.gate_biases);
        }
        if let Some(fourier) = &params.fourier {
            visit(&fourier.real);
            visit(&fourier.imaginary);
        }
    }

    fn for_each_param_vector_mut(
        params: &mut CliffordParams,
        mut visit: impl FnMut(&mut Vec<f32>),
    ) {
        visit(&mut params.layer_norm_gamma);
        visit(&mut params.layer_norm_beta);
        visit(&mut params.determinant_weights);
        visit(&mut params.determinant_biases);
        for conv in &mut params.depthwise {
            visit(&mut conv.biases);
            if let Some(bn) = &mut conv.batch_norm {
                visit(&mut bn.scales);
            }
            visit(&mut conv.weights);
        }
        visit(&mut params.local.projection_weights);
        visit(&mut params.local.projection_biases);
        visit(&mut params.local.gate_weights);
        visit(&mut params.local.gate_biases);
        visit(&mut params.layer_scale);
        if let Some(global) = &mut params.global {
            visit(&mut global.projection_weights);
            visit(&mut global.projection_biases);
            visit(&mut global.gate_weights);
            visit(&mut global.gate_biases);
        }
        if let Some(fourier) = &mut params.fourier {
            visit(&mut fourier.real);
            visit(&mut fourier.imaginary);
        }
    }

    fn flatten_params(params: &CliffordParams) -> Vec<f32> {
        let mut values = Vec::new();
        for_each_param_vector(params, |vector| values.extend_from_slice(vector));
        values
    }

    fn perturb_param(params: &mut CliffordParams, index: usize, amount: f32) {
        let mut base = 0;
        for_each_param_vector_mut(params, |vector| {
            if (base..base + vector.len()).contains(&index) {
                vector[index - base] += amount;
            }
            base += vector.len();
        });
    }

    fn scalar_loss(
        spec: &CliffordSpec,
        params: &CliffordParams,
        input: &Tensor,
        upstream: &Tensor,
        mask: &[f32],
    ) -> f32 {
        let (output, _) = forward(spec, params, input, true, Some(mask)).expect("Clifford forward");
        output
            .data
            .iter()
            .zip(&upstream.data)
            .map(|(output, upstream)| output * upstream)
            .sum()
    }

    fn check_finite_differences(
        global_op: &str,
        num_dwconv: usize,
        interaction_mode: &str,
        context_mode: &str,
        gffn_mode: &str,
        higher_mode: &str,
        drop_path: f32,
    ) {
        let spec = build_spec(
            global_op,
            num_dwconv,
            interaction_mode,
            context_mode,
            gffn_mode,
            higher_mode,
            drop_path,
        );
        let params = test_params(&spec);
        let input = Tensor {
            shape: crate::model::TensorShape::new(2, 1, 2, 2),
            data: vec![0.13, -0.21, 0.31, 0.07, -0.17, 0.23, 0.09, -0.29],
        };
        let upstream = Tensor {
            shape: input.shape,
            data: vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.15, -0.35],
        };
        let mask = vec![0.0, 1.0 / (1.0 - spec.drop_path)];
        let (output, cache) =
            forward(&spec, &params, &input, true, Some(&mask)).expect("Clifford forward");
        let (input_gradient, parameter_gradient) =
            backward(&spec, &params, &cache, &upstream).expect("Clifford backward");
        let epsilon = 1.0e-3;
        let tolerance = 4.0e-3;

        for index in 0..input.data.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus.data[index] += epsilon;
            minus.data[index] -= epsilon;
            let numeric = (scalar_loss(&spec, &params, &plus, &upstream, &mask)
                - scalar_loss(&spec, &params, &minus, &upstream, &mask))
                / (2.0 * epsilon);
            let analytic = input_gradient.data[index];
            assert!(
                (numeric - analytic).abs() <= tolerance * (1.0 + numeric.abs()),
                "{global_op} input gradient[{index}] analytic={analytic}, numeric={numeric}"
            );
        }

        let analytic_params = flatten_params(&parameter_gradient);
        let base_params = flatten_params(&params);
        assert_eq!(analytic_params.len(), base_params.len());
        for index in 0..base_params.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            perturb_param(&mut plus, index, epsilon);
            perturb_param(&mut minus, index, -epsilon);
            let numeric = (scalar_loss(&spec, &plus, &input, &upstream, &mask)
                - scalar_loss(&spec, &minus, &input, &upstream, &mask))
                / (2.0 * epsilon);
            let analytic = analytic_params[index];
            assert!(
                (numeric - analytic).abs() <= tolerance * (1.0 + numeric.abs()),
                "{global_op} parameter gradient[{index}] analytic={analytic}, numeric={numeric}"
            );
        }

        assert_eq!(output.shape, input.shape);
    }

    #[test]
    fn clifford_axis_layout_pins_cpp_nchw_token_channel_normalization() {
        use crate::axis_fixture::{axis_tensor, nchw_index, probe_shape};

        // The C++ layer calls layernorm_nchw_forward with B,C,HW. It reduces
        // channels independently at each batch/spatial token, then writes NCHW.
        // Source: Jev_laya:src-lib/clifford_layer.cpp:311-346,1578-1580.
        let shape = probe_shape(); // B=2,C=2,H=2,W=3
        let input_values = axis_tensor(shape);
        let input = Tensor::from_data(shape, input_values.clone()).unwrap();
        let gamma = [1.25, -0.75];
        let beta = [0.2, -0.4];
        let (normalized, _, _) = layer_norm_forward(&input, &gamma, &beta);
        let mut expected = vec![0.0; input_values.len()];
        let spatial = shape.height * shape.width;
        for n in 0..shape.batch {
            for y in 0..shape.height {
                for x in 0..shape.width {
                    let values = (0..shape.channels)
                        .map(|c| input_values[nchw_index(shape, n, c, y, x)])
                        .collect::<Vec<_>>();
                    let mean = values.iter().sum::<f32>() / shape.channels as f32;
                    let variance = values
                        .iter()
                        .map(|value| (value - mean) * (value - mean))
                        .sum::<f32>()
                        / shape.channels as f32;
                    let inv_std = 1.0 / (variance + 1.0e-5).sqrt();
                    for c in 0..shape.channels {
                        let index = nchw_index(shape, n, c, y, x);
                        expected[index] = (values[c] - mean) * inv_std * gamma[c] + beta[c];
                    }
                }
            }
        }
        assert_eq!(normalized.shape, shape);
        assert_eq!(spatial, 6);
        for (index, (actual, expected)) in normalized.data.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-6,
                "Clifford C++ NCHW token norm mismatch at flat index {index}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn clifford_avgpool_drop_path_backward_matches_finite_differences() {
        check_finite_differences("avgpool", 2, "full", "diff", "hybrid", "vb", 0.25);
    }

    #[test]
    fn clifford_fourier_backward_matches_finite_differences() {
        check_finite_differences("fourier", 1, "full", "diff", "hybrid", "off", 0.0);
    }

    #[test]
    fn clifford_inner_abs_local_backward_matches_finite_differences() {
        check_finite_differences("avgpool", 1, "inner", "abs", "local", "vb", 0.0);
    }

    #[test]
    fn clifford_wedge_global_backward_matches_finite_differences() {
        check_finite_differences("avgpool", 1, "wedge", "abs", "global", "off", 0.0);
    }
}
