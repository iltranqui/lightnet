//! Clifford parameter and optimizer buffers.

use crate::params::{BatchNormParams, BatchNormTrainBuffers, ConvolutionalParams, ConvolutionalTrainBuffers};
use lightnet_model::clifford::geometry::raw_channels;
use lightnet_model::model::{CliffordGffnMode, CliffordGlobalOp, CliffordSpec};

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

    #[doc(hidden)]
    pub fn zero_gradients_like(&self) -> Self {
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



#[doc(hidden)]
pub fn has_local(spec: &CliffordSpec) -> bool {
    spec.gffn_mode != CliffordGffnMode::Global
}

#[doc(hidden)]
pub fn has_global(spec: &CliffordSpec) -> bool {
    spec.gffn_mode != CliffordGffnMode::Local
}

#[doc(hidden)]
pub fn has_fourier(spec: &CliffordSpec) -> bool {
    has_global(spec) && spec.global_op == CliffordGlobalOp::Fourier
}

