use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::model::{
    Activation, ConnectedSpec, ConvolutionalSpec, LayerKind, LrPolicy, NetConfig, Network,
    RouteSpec, TensorShape,
};
use crate::weights::WeightsHeader;
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

const BATCH_NORM_EPSILON: f32 = 0.000_01;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: TensorShape,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: TensorShape) -> Self {
        Self {
            shape,
            data: vec![0.0; shape.batch * shape.element_count()],
        }
    }

    pub fn filled(shape: TensorShape, value: f32) -> Self {
        Self {
            shape,
            data: vec![value; shape.batch * shape.element_count()],
        }
    }

    pub fn from_data(shape: TensorShape, data: Vec<f32>) -> Result<Self> {
        let expected = shape.batch * shape.element_count();
        if data.len() != expected {
            return Err(Error::InvalidConfig(format!(
                "tensor data length mismatch: expected {expected}, got {}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    fn channel_area(&self) -> usize {
        self.shape.width * self.shape.height
    }

    fn batch_stride(&self) -> usize {
        self.shape.channels * self.channel_area()
    }

    fn offset(&self, batch: usize, channel: usize, y: usize, x: usize) -> usize {
        batch * self.batch_stride() + channel * self.channel_area() + y * self.shape.width + x
    }

    pub fn get(&self, batch: usize, channel: usize, y: usize, x: usize) -> f32 {
        self.data[self.offset(batch, channel, y, x)]
    }

    pub fn set(&mut self, batch: usize, channel: usize, y: usize, x: usize, value: f32) {
        let idx = self.offset(batch, channel, y, x);
        self.data[idx] = value;
    }
}

impl From<PlanarRgbImage> for Tensor {
    fn from(value: PlanarRgbImage) -> Self {
        Self {
            shape: value.shape,
            data: value.data,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchNormParams {
    pub scales: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_variance: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionalParams {
    pub biases: Vec<f32>,
    pub batch_norm: Option<BatchNormParams>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConnectedParams {
    pub biases: Vec<f32>,
    pub batch_norm: Option<BatchNormParams>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerParams {
    None,
    Convolutional(ConvolutionalParams),
    Connected(ConnectedParams),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchNormTrainBuffers {
    pub scale_updates: Vec<f32>,
    pub mean_delta: Vec<f32>,
    pub variance_delta: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionalTrainBuffers {
    pub bias_updates: Vec<f32>,
    pub batch_norm: Option<BatchNormTrainBuffers>,
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConnectedTrainBuffers {
    pub bias_updates: Vec<f32>,
    pub batch_norm: Option<BatchNormTrainBuffers>,
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerTrainBuffers {
    None,
    Convolutional(ConvolutionalTrainBuffers),
    Connected(ConnectedTrainBuffers),
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloHeadOutput {
    pub layer_index: usize,
    pub output: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardResult {
    pub layer_outputs: Vec<Tensor>,
    pub layer_activation_inputs: Vec<Option<Tensor>>,
    pub yolo_heads: Vec<YoloHeadOutput>,
}

impl ForwardResult {
    pub fn final_output(&self) -> Option<&Tensor> {
        self.layer_outputs.last()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackwardResult {
    pub input_gradient: Tensor,
    pub layer_output_gradients: Vec<Tensor>,
    pub param_gradients: Vec<LayerParams>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuForwardSummary {
    pub final_output: Option<Tensor>,
    pub yolo_heads: Vec<YoloHeadOutput>,
}

#[derive(Clone, Debug)]
pub struct GpuForwardState<T>
where
    T: Clone,
{
    pub input_gpu: T,
    pub gpu_layer_outputs: Vec<T>,
    pub gpu_layer_activation_inputs: Vec<Option<T>>,
    pub cpu_forward: ForwardResult,
}

/// Pre-allocate the shared backward scratch buffer.
/// Sized to max(filters + nweights) across all conv layers — mirrors C++ `net->workspace`.
pub fn compute_cpu_workspace_size(network: &Network) -> usize {
    network
        .layers
        .iter()
        .map(|l| match &l.kind {
            LayerKind::Convolutional(spec) => {
                let groups = spec.groups.max(1);
                let in_ch = l.input_shape.channels / groups;
                let nweights = spec.filters * in_ch * spec.size * spec.size;
                spec.filters + nweights
            }
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableNetwork {
    pub network: Network,
    pub params: Vec<LayerParams>,
    pub train_buffers: Vec<LayerTrainBuffers>,
    /// Shared scratch buffer for backward pass — avoids per-call Vec allocation.
    /// Equivalent to C++ `net->workspace`. Private; threaded through backward_accumulate paths.
    workspace: Vec<f32>,
}

impl ExecutableNetwork {
    pub fn new(network: Network) -> Self {
        let params = network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        let train_buffers = network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect::<Vec<_>>();
        let workspace = vec![0.0f32; compute_cpu_workspace_size(&network)];
        Self {
            network,
            params,
            train_buffers,
            workspace,
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(Network::from_path(path)?))
    }

    pub fn zero_input(&self) -> Tensor {
        Tensor::zeros(self.network.net.input_shape())
    }

    pub fn load_weights_file(&mut self, path: impl AsRef<Path>) -> Result<WeightsHeader> {
        load_weights_into_params(&self.network, &mut self.params, path)
    }

    pub fn zero_train_buffers(&mut self) {
        self.train_buffers = self
            .network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect();
    }

    pub fn update_weights(&mut self, lr: f32, momentum: f32, decay: f32, batch: usize) {
        for (params, train) in self.params.iter_mut().zip(&mut self.train_buffers) {
            match (params, train) {
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) => {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    if let (Some(params_bn), Some(train_bn)) =
                        (&mut params.batch_norm, &mut train.batch_norm)
                    {
                        sgd_update(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    }
                }
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    if let (Some(params_bn), Some(train_bn)) =
                        (&mut params.batch_norm, &mut train.batch_norm)
                    {
                        sgd_update(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    }
                }
                _ => {}
            }
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }

        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                input
            } else {
                layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let (output, activation_input) = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    forward_convolutional_with_cache(spec, conv, current_input, layer.output_shape)
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => (
                    forward_connected(spec, connected, current_input, layer.output_shape),
                    None,
                ),
                (LayerKind::MaxPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    false,
                )
                .map(|output| (output, None))?,
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    true,
                )
                .map(|output| (output, None))?,
                (LayerKind::AvgPool, LayerParams::None) => (forward_avgpool(current_input), None),
                (LayerKind::Route(spec), LayerParams::None) => (
                    forward_route(spec, layer.index, &layer_outputs, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    forward_shortcut_with_cache(spec, current_input, layer.index, &layer_outputs)?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => (
                    forward_upsample(spec, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Softmax { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::ScaleChannels(_), LayerParams::None)
                | (LayerKind::Sam(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => {
                    (current_input.clone(), None)
                }
                (LayerKind::Reorg(spec), LayerParams::None) => (
                    forward_reorg(spec.stride, spec.reverse, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Rnn { .. }, _)
                | (LayerKind::Lstm { .. }, _)
                | (LayerKind::Crnn { .. }, _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: layer.kind.name().to_owned(),
                    });
                }
                (LayerKind::Unsupported(name), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: name.clone(),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            };

            if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }

            layer_outputs.push(output);
            layer_activation_inputs.push(activation_input);
        }

        Ok(ForwardResult {
            layer_outputs,
            layer_activation_inputs,
            yolo_heads,
        })
    }

    pub fn backward(
        &self,
        input: &Tensor,
        forward: &ForwardResult,
        output_gradient: &Tensor,
    ) -> Result<BackwardResult> {
        let mut layer_output_gradient_seeds = vec![None; self.network.layers.len()];
        let last_index = self
            .network
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()))?;
        layer_output_gradient_seeds[last_index] = Some(output_gradient.clone());
        self.backward_seeded(input, forward, &layer_output_gradient_seeds)
    }

    pub fn backward_seeded(
        &self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<BackwardResult> {
        // Allocate a local workspace for this standalone (non-accumulate) backward call.
        // backward_accumulate_seeded reuses the pre-allocated self.workspace instead.
        let mut local_workspace = vec![0.0f32; compute_cpu_workspace_size(&self.network)];
        self.backward_seeded_core(
            &mut local_workspace,
            input,
            forward,
            layer_output_gradient_seeds,
        )
    }

    fn backward_seeded_core(
        &self,
        workspace: &mut Vec<f32>,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<BackwardResult> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        if forward.layer_outputs.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward result layer count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.layer_outputs.len()
            )));
        }
        if forward.layer_activation_inputs.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward activation cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.layer_activation_inputs.len()
            )));
        }
        if layer_output_gradient_seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "backward seed count mismatch: expected {}, got {}",
                self.network.layers.len(),
                layer_output_gradient_seeds.len()
            )));
        }

        let mut input_gradient = Tensor::zeros(input.shape);
        let mut layer_output_gradients = self
            .network
            .layers
            .iter()
            .map(|layer| Tensor::zeros(layer.output_shape))
            .collect::<Vec<_>>();
        let mut param_gradients = self
            .network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        for (layer, seed) in self.network.layers.iter().zip(layer_output_gradient_seeds) {
            if let Some(seed) = seed {
                if seed.shape != layer.output_shape {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} backward seed shape mismatch: expected {}, got {}",
                        layer.index, layer.output_shape, seed.shape
                    )));
                }
                layer_output_gradients[layer.index] = seed.clone();
            }
        }

        for (layer, params) in self.network.layers.iter().zip(&self.params).rev() {
            let grad = layer_output_gradients[layer.index].clone();
            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing saved forward output for layer #{} input",
                        layer.index
                    ))
                })?
            };

            match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let (input_grad, layer_param_gradients) = backward_convolutional(
                        spec,
                        conv,
                        current_input,
                        layer.output_shape,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                        workspace,
                    )?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    accumulate_convolutional_param_gradients(
                        param_gradients.get_mut(owner).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing convolutional gradient slot for owner layer #{owner}"
                            ))
                        })?,
                        &layer_param_gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    let input_grad = backward_maxpool(
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        current_input,
                        &grad,
                        false,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    let input_grad = backward_maxpool(
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        current_input,
                        &grad,
                        true,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::AvgPool, LayerParams::None) => {
                    let input_grad = backward_avgpool(&grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    backward_route(
                        spec,
                        layer.index,
                        &grad,
                        &mut layer_output_gradients,
                        &self.network.layers,
                    )?;
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let accum_grad = backward_shortcut(
                        spec,
                        layer.index,
                        &forward.layer_outputs,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &accum_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        accumulate_tensor(&mut layer_output_gradients[source_index], &accum_grad)?;
                    }
                }
                (LayerKind::Upsample(spec), LayerParams::None) => {
                    let input_grad = backward_upsample(spec, &grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Softmax { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::ScaleChannels(_), LayerParams::None)
                | (LayerKind::Sam(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => {
                    accumulate_previous_gradient(
                        layer.index,
                        &grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Connected(_), _)
                | (LayerKind::Reorg(_), _)
                | (LayerKind::Rnn { .. }, _)
                | (LayerKind::Lstm { .. }, _)
                | (LayerKind::Crnn { .. }, _)
                | (LayerKind::Unsupported(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{}_backward", layer.kind.name()),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            }
        }

        Ok(BackwardResult {
            input_gradient,
            layer_output_gradients,
            param_gradients,
        })
    }

    pub fn backward_accumulate(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        output_gradient: &Tensor,
    ) -> Result<BackwardResult> {
        // Build seed vec and delegate to backward_accumulate_seeded so both paths
        // share the pre-allocated self.workspace (C++ net->workspace parity).
        let mut seeds = vec![None; self.network.layers.len()];
        let last = self
            .network
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()))?;
        seeds[last] = Some(output_gradient.clone());
        self.backward_accumulate_seeded(input, forward, &seeds)
    }

    pub fn backward_accumulate_seeded(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<BackwardResult> {
        // Pull out the pre-allocated workspace to avoid the borrow-split problem:
        // mem::take replaces self.workspace with an empty Vec, then we restore it after.
        let mut ws = std::mem::take(&mut self.workspace);
        let result =
            self.backward_seeded_core(&mut ws, input, forward, layer_output_gradient_seeds)?;
        self.workspace = ws;
        self.update_convolution_batch_norm_running_stats(input, forward)?;
        self.accumulate_train_buffers(&result.param_gradients)?;
        Ok(result)
    }

    fn update_convolution_batch_norm_running_stats(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
    ) -> Result<()> {
        for layer in &self.network.layers {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize || spec.share_index.is_some() {
                continue;
            }

            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let (mean, variance) = {
                let params =
                    resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                let raw_output =
                    raw_convolution_output(spec, params, current_input, layer.output_shape);
                channel_mean_variance(&raw_output)
            };

            let Some(LayerParams::Convolutional(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing convolutional params for layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing batch-norm params for convolutional layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(batch_norm, &mean, &variance, 0.9)?;
        }
        Ok(())
    }

    fn accumulate_train_buffers(&mut self, param_gradients: &[LayerParams]) -> Result<()> {
        if param_gradients.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "parameter gradient count mismatch: expected {}, got {}",
                self.network.layers.len(),
                param_gradients.len()
            )));
        }

        for (layer, gradient) in self.network.layers.iter().zip(param_gradients) {
            match (&layer.kind, gradient) {
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv_grad)) => {
                    let owner = spec.share_index.unwrap_or(layer.index);
                    let Some(LayerTrainBuffers::Convolutional(train)) =
                        self.train_buffers.get_mut(owner)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing convolutional train buffers for layer #{} owner #{}",
                            layer.index, owner
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &conv_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &conv_grad.weights)?;
                    match (&mut train.batch_norm, &conv_grad.batch_norm) {
                        (Some(train_bn), Some(grad_bn)) => {
                            accumulate_vec(&mut train_bn.scale_updates, &grad_bn.scales)?;
                            accumulate_vec(&mut train_bn.mean_delta, &grad_bn.rolling_mean)?;
                            accumulate_vec(
                                &mut train_bn.variance_delta,
                                &grad_bn.rolling_variance,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "batch norm train buffer mismatch while accumulating layer #{}",
                                layer.index
                            )));
                        }
                    }
                }
                (LayerKind::Connected(_), LayerParams::Connected(connected_grad)) => {
                    let Some(LayerTrainBuffers::Connected(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing connected train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &connected_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &connected_grad.weights)?;
                    match (&mut train.batch_norm, &connected_grad.batch_norm) {
                        (Some(train_bn), Some(grad_bn)) => {
                            accumulate_vec(&mut train_bn.scale_updates, &grad_bn.scales)?;
                            accumulate_vec(&mut train_bn.mean_delta, &grad_bn.rolling_mean)?;
                            accumulate_vec(
                                &mut train_bn.variance_delta,
                                &grad_bn.rolling_variance,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "batch norm train buffer mismatch while accumulating connected layer #{}",
                                layer.index
                            )));
                        }
                    }
                }
                (_, LayerParams::None) => {}
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} produced gradients incompatible with train buffers",
                        layer.index
                    )));
                }
            }
        }

        Ok(())
    }
}

pub fn compute_learning_rate(net: &NetConfig, batch_num: usize) -> f32 {
    let mut lr = net.learning_rate;

    if net.burn_in > 0 && batch_num < net.burn_in {
        let progress = batch_num as f32 / net.burn_in as f32;
        return net.learning_rate * progress.powi(4);
    }

    match net.policy {
        LrPolicy::Constant => lr,
        LrPolicy::Steps => {
            for (step, scale) in net.steps.iter().zip(&net.scales) {
                if batch_num >= *step {
                    lr *= *scale;
                }
            }
            lr
        }
        LrPolicy::Poly => {
            if net.max_batches == 0 || batch_num >= net.max_batches {
                0.0
            } else {
                let progress = 1.0 - batch_num as f32 / net.max_batches as f32;
                net.learning_rate * progress.powi(4)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
{
    pub backend: B,
    pub network: Network,
    pub params: Vec<LayerParams>,
    device_conv_params: RefCell<Vec<Option<GpuConvolutionLayerParams<B::Tensor>>>>,
    device_train_buffers: RefCell<Vec<Option<GpuConvolutionTrainBuffers<B::Tensor>>>>,
}

#[derive(Clone, Debug)]
struct GpuConvolutionLayerParams<T> {
    raw_weights: T,
    raw_biases: T,
    weights: T,
    biases: T,
    batch_norm_scales: Option<T>,
}

#[derive(Clone, Debug)]
struct GpuConvolutionTrainBuffers<T> {
    weight_updates: T,
    bias_updates: T,
    scale_updates: Option<T>,
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn new(network: Network, backend: B) -> Self {
        let params = network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        let device_conv_params = RefCell::new(vec![None; network.layers.len()]);
        let device_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        Self {
            backend,
            network,
            params,
            device_conv_params,
            device_train_buffers,
        }
    }

    pub fn from_path(path: impl AsRef<Path>, backend: B) -> Result<Self> {
        Ok(Self::new(Network::from_path(path)?, backend))
    }

    pub fn zero_input(&self) -> Tensor {
        Tensor::zeros(self.network.net.input_shape())
    }

    pub fn load_weights_file(&mut self, path: impl AsRef<Path>) -> Result<WeightsHeader> {
        let header = load_weights_into_params(&self.network, &mut self.params, path)?;
        self.clear_device_conv_params();
        self.clear_device_train_buffers();
        self.warm_device_conv_params()?;
        Ok(header)
    }

    pub fn init_train_buffers_gpu(&mut self) -> Result<()> {
        self.warm_device_conv_params()?;
        let mut train_buffers = Vec::with_capacity(self.network.layers.len());
        for layer in &self.network.layers {
            match &layer.kind {
                LayerKind::Convolutional(spec) if spec.share_index.is_none() => {
                    let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
                    let weight_updates = self
                        .backend
                        .zeros_f32(&[spec.filters, grouped_channels, spec.size, spec.size])
                        .map_err(|err| backend_error("allocate GPU weight updates", err))?;
                    let bias_updates = self
                        .backend
                        .zeros_f32(&[spec.filters])
                        .map_err(|err| backend_error("allocate GPU bias updates", err))?;
                    let scale_updates = spec
                        .batch_normalize
                        .then(|| {
                            self.backend
                                .zeros_f32(&[spec.filters])
                                .map_err(|err| backend_error("allocate GPU scale updates", err))
                        })
                        .transpose()?;
                    train_buffers.push(Some(GpuConvolutionTrainBuffers {
                        weight_updates,
                        bias_updates,
                        scale_updates,
                    }));
                }
                _ => train_buffers.push(None),
            }
        }
        *self.device_train_buffers.get_mut() = train_buffers;
        Ok(())
    }

    pub fn zero_train_buffers_gpu(&mut self) -> Result<()> {
        self.init_train_buffers_gpu()
    }

    pub fn update_weights_gpu(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    ) -> Result<()> {
        let config = ElementwiseLaunchConfig::default();
        for layer_index in 0..self.network.layers.len() {
            let Some(owner) = (match &self.network.layers[layer_index].kind {
                LayerKind::Convolutional(spec) => {
                    let owner = spec.share_index.unwrap_or(layer_index);
                    (owner == layer_index).then_some(owner)
                }
                _ => None,
            }) else {
                continue;
            };

            let cached = self
                .device_conv_params
                .borrow()
                .get(owner)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing cached GPU convolution params for layer #{owner}"
                    ))
                })?;
            let train = self
                .device_train_buffers
                .borrow()
                .get(owner)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing cached GPU train buffers for layer #{owner}"
                    ))
                })?;

            let (raw_weights, weight_updates) = self
                .backend
                .sgd_update_f32(
                    &cached.raw_weights,
                    &train.weight_updates,
                    lr,
                    momentum,
                    decay,
                    batch,
                    config,
                )
                .map_err(|err| backend_error("GPU weight update", err))?;
            let (raw_biases, bias_updates) = self
                .backend
                .sgd_update_f32(
                    &cached.raw_biases,
                    &train.bias_updates,
                    lr,
                    momentum,
                    0.0,
                    batch,
                    config,
                )
                .map_err(|err| backend_error("GPU bias update", err))?;
            let scale_updates = self.update_gpu_batch_norm_scales(
                owner,
                train.scale_updates,
                lr,
                momentum,
                batch,
                config,
            )?;

            let refreshed = self.refresh_device_convolution_params(
                owner,
                raw_weights,
                raw_biases,
                weight_updates,
                bias_updates,
                scale_updates,
            )?;
            self.sync_convolution_owner_from_device(owner, &refreshed)?;
        }
        Ok(())
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        Ok(self.forward_with_state(input)?.cpu_forward)
    }

    pub fn forward_with_state(&self, input: &Tensor) -> Result<GpuForwardState<B::Tensor>> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f32(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload input tensor", err))?;

        let mut gpu_layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut gpu_layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                &input_gpu
            } else {
                gpu_layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let mut activation_input_gpu = None;
            let output_gpu = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = self.cached_convolution_params(
                        owner,
                        layer.input_shape.channels,
                        spec,
                        conv,
                    )?;
                    let conv_output = self
                        .backend
                        .conv2d_f32(
                            current_input,
                            &cached.weights,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution", err))?;
                    let biased = self
                        .backend
                        .bias_add_f32(
                            &conv_output,
                            &cached.biases,
                            spec.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution bias add", err))?;
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    activation_input_gpu = Some(biased.clone());
                    self.backend
                        .activation_f32(&biased, activation, launch_config)
                        .map_err(|err| backend_error("convolution activation", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "maxpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .maxpool_f32(
                            current_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            launch_config,
                        )
                        .map_err(|err| backend_error("maxpool", err))?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f32(
                            current_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            launch_config,
                        )
                        .map_err(|err| backend_error("local_avgpool", err))?
                }
                (LayerKind::AvgPool, LayerParams::None) => self
                    .backend
                    .avgpool_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("avgpool", err))?,
                (LayerKind::Route(spec), LayerParams::None) => {
                    let mut route_inputs = Vec::with_capacity(spec.layers.len());
                    let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
                    let mut route_channels = Vec::with_capacity(spec.layers.len());
                    let mut route_total_channels = Vec::with_capacity(spec.layers.len());
                    for raw_index in &spec.layers {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source =
                            gpu_layer_outputs
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for route layer #{}",
                                    layer.index
                                ))
                            })?;
                        let chunk_channels =
                            source_layer.output_shape.channels / spec.groups.max(1);
                        route_inputs.push(source);
                        route_channel_offsets.push(chunk_channels * spec.group_id);
                        route_channels.push(chunk_channels);
                        route_total_channels.push(source_layer.output_shape.channels);
                    }

                    self.backend
                        .route_concat_f32(
                            &route_inputs,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("route concat", err))?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let mut accumulated = current_input.clone();
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source =
                            gpu_layer_outputs
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for shortcut layer #{}",
                                    layer.index
                                ))
                            })?;
                        if source_layer.output_shape != layer.output_shape {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        accumulated = self
                            .backend
                            .shortcut_add_f32(&accumulated, source, launch_config)
                            .map_err(|err| backend_error("shortcut add", err))?;
                    }

                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    activation_input_gpu = Some(accumulated.clone());
                    self.backend
                        .activation_f32(&accumulated, activation, launch_config)
                        .map_err(|err| backend_error("shortcut activation", err))?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.stride.unsigned_abs(),
                        spec.scale,
                        spec.reverse,
                        launch_config,
                    )
                    .map_err(|err| backend_error("upsample", err))?,
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Softmax { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::ScaleChannels(_), LayerParams::None)
                | (LayerKind::Sam(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => current_input.clone(),
                (LayerKind::Connected(_), _)
                | (LayerKind::Reorg(_), _)
                | (LayerKind::Rnn { .. }, _)
                | (LayerKind::Lstm { .. }, _)
                | (LayerKind::Crnn { .. }, _)
                | (LayerKind::Unsupported(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: layer.kind.name().to_owned(),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            };

            let output = self
                .download_tensor(&output_gpu, layer.output_shape)
                .map_err(|err| backend_error("download layer output", err))?;
            let activation_input = match activation_input_gpu.as_ref() {
                Some(tensor) => Some(
                    self.download_tensor(tensor, layer.output_shape)
                        .map_err(|err| backend_error("download activation cache", err))?,
                ),
                None => None,
            };

            if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }

            gpu_layer_outputs.push(output_gpu);
            gpu_layer_activation_inputs.push(activation_input_gpu);
            layer_outputs.push(output);
            layer_activation_inputs.push(activation_input);
        }

        Ok(GpuForwardState {
            input_gpu,
            gpu_layer_outputs,
            gpu_layer_activation_inputs,
            cpu_forward: ForwardResult {
                layer_outputs,
                layer_activation_inputs,
                yolo_heads,
            },
        })
    }

    pub fn forward_compact(&self, input: &Tensor) -> Result<GpuForwardSummary> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f32(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload input tensor", err))?;
        let mut gpu_layer_outputs = vec![None; self.network.layers.len()];
        let mut remaining_uses = gpu_layer_output_use_counts(&self.network)?;
        let mut final_output = None;
        let mut yolo_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                Some(&input_gpu)
            } else if layer_uses_previous_input(&layer.kind) {
                Some(
                    gpu_layer_outputs
                        .get(layer.index - 1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU layer output for layer #{}",
                                layer.index
                            ))
                        })?,
                )
            } else {
                None
            };

            let output_gpu = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = self.cached_convolution_params(
                        owner,
                        layer.input_shape.channels,
                        spec,
                        conv,
                    )?;
                    let conv_output = self
                        .backend
                        .conv2d_f32(
                            current_input.expect("convolution requires current input"),
                            &cached.weights,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution", err))?;
                    let biased = self
                        .backend
                        .bias_add_f32(
                            &conv_output,
                            &cached.biases,
                            spec.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution bias add", err))?;
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .activation_f32(&biased, activation, launch_config)
                        .map_err(|err| backend_error("convolution activation", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "maxpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .maxpool_f32(
                            current_input.expect("maxpool requires current input"),
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            launch_config,
                        )
                        .map_err(|err| backend_error("maxpool", err))?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f32(
                            current_input.expect("local_avgpool requires current input"),
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            launch_config,
                        )
                        .map_err(|err| backend_error("local_avgpool", err))?
                }
                (LayerKind::AvgPool, LayerParams::None) => self
                    .backend
                    .avgpool_f32(
                        current_input.expect("avgpool requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("avgpool", err))?,
                (LayerKind::Route(spec), LayerParams::None) => {
                    let mut route_inputs = Vec::with_capacity(spec.layers.len());
                    let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
                    let mut route_channels = Vec::with_capacity(spec.layers.len());
                    let mut route_total_channels = Vec::with_capacity(spec.layers.len());
                    for raw_index in &spec.layers {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source = gpu_layer_outputs
                            .get(source_index)
                            .and_then(Option::as_ref)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for route layer #{}",
                                    layer.index
                                ))
                            })?;
                        let chunk_channels =
                            source_layer.output_shape.channels / spec.groups.max(1);
                        route_inputs.push(source);
                        route_channel_offsets.push(chunk_channels * spec.group_id);
                        route_channels.push(chunk_channels);
                        route_total_channels.push(source_layer.output_shape.channels);
                    }

                    self.backend
                        .route_concat_f32(
                            &route_inputs,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("route concat", err))?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let mut accumulated = current_input
                        .expect("shortcut requires current input")
                        .clone();
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source = gpu_layer_outputs
                            .get(source_index)
                            .and_then(Option::as_ref)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for shortcut layer #{}",
                                    layer.index
                                ))
                            })?;
                        if source_layer.output_shape != layer.output_shape {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        accumulated = self
                            .backend
                            .shortcut_add_f32(&accumulated, source, launch_config)
                            .map_err(|err| backend_error("shortcut add", err))?;
                    }

                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .activation_f32(&accumulated, activation, launch_config)
                        .map_err(|err| backend_error("shortcut activation", err))?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f32(
                        current_input.expect("upsample requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.stride.unsigned_abs(),
                        spec.scale,
                        spec.reverse,
                        launch_config,
                    )
                    .map_err(|err| backend_error("upsample", err))?,
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Softmax { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::ScaleChannels(_), LayerParams::None)
                | (LayerKind::Sam(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => current_input
                    .expect("staging layer requires current input")
                    .clone(),
                (LayerKind::Connected(_), _)
                | (LayerKind::Reorg(_), _)
                | (LayerKind::Rnn { .. }, _)
                | (LayerKind::Lstm { .. }, _)
                | (LayerKind::Crnn { .. }, _)
                | (LayerKind::Unsupported(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: layer.kind.name().to_owned(),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            };

            let should_download_final = layer.index + 1 == self.network.layers.len();
            let should_collect_yolo =
                matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_));
            let downloaded = if should_download_final || should_collect_yolo {
                Some(
                    self.download_tensor(&output_gpu, layer.output_shape)
                        .map_err(|err| backend_error("download layer output", err))?,
                )
            } else {
                None
            };

            if should_collect_yolo {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: downloaded
                        .as_ref()
                        .expect("yolo output should have been downloaded")
                        .clone(),
                });
            }
            if should_download_final {
                final_output = downloaded;
            }

            gpu_layer_outputs[layer.index] = Some(output_gpu);
            release_gpu_sources_for_layer(
                layer.index,
                &layer.kind,
                &mut remaining_uses,
                &mut gpu_layer_outputs,
            )?;

            if remaining_uses[layer.index] == 0 {
                gpu_layer_outputs[layer.index] = None;
            }
        }

        Ok(GpuForwardSummary {
            final_output,
            yolo_heads,
        })
    }

    pub fn backward_accumulate_seeded_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        if layer_output_gradient_seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "GPU backward seed count mismatch: expected {}, got {}",
                self.network.layers.len(),
                layer_output_gradient_seeds.len()
            )));
        }
        if self.device_train_buffers.borrow().len() != self.network.layers.len()
            || self
                .device_train_buffers
                .borrow()
                .iter()
                .all(Option::is_none)
        {
            self.init_train_buffers_gpu()?;
        }

        let config = ElementwiseLaunchConfig::default();
        let mut layer_output_gradients: Vec<Option<B::Tensor>> =
            vec![None; self.network.layers.len()];
        for (layer_index, seed) in layer_output_gradient_seeds.iter().enumerate() {
            if let Some(seed) = seed {
                layer_output_gradients[layer_index] = Some(
                    self.backend
                        .upload_f32(&tensor_backend_shape(seed.shape), &seed.data)
                        .map_err(|err| backend_error("upload GPU backward seed", err))?,
                );
            }
        }

        for layer in self.network.layers.iter().rev() {
            let Some(grad) = layer_output_gradients[layer.index].take() else {
                continue;
            };

            match (&layer.kind, self.params.get(layer.index)) {
                (
                    LayerKind::Convolutional(spec),
                    Some(LayerParams::Convolutional(_) | LayerParams::None),
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = self.cached_convolution_params(
                        owner,
                        layer.input_shape.channels,
                        spec,
                        conv,
                    )?;
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for convolutional layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| backend_error("GPU convolution activation backward", err))?;
                    let bias_grad = self
                        .backend
                        .bias_backward_f32(
                            &delta,
                            spec.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| backend_error("GPU convolution bias backward", err))?;
                    let current_input =
                        if layer.index == 0 {
                            &forward.input_gpu
                        } else {
                            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU layer output for convolutional layer #{}",
                                layer.index
                            ))
                        })?
                        };
                    let scale_grad = if let Some(batch_norm) = conv.batch_norm.as_ref() {
                        let raw_output = self
                            .backend
                            .conv2d_f32(
                                current_input,
                                &cached.raw_weights,
                                &tensor_backend_shape(layer.output_shape),
                                layer.input_shape.channels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                spec.filters,
                                spec.size,
                                spec.stride_x,
                                spec.stride_y,
                                spec.dilation,
                                spec.padding,
                                spec.groups.max(1),
                                config,
                            )
                            .map_err(|err| backend_error("GPU batch-norm raw convolution", err))?;
                        let scale_grad = self.batch_norm_scale_gradient_gpu(
                            &raw_output,
                            &delta,
                            batch_norm,
                            layer.output_shape,
                        )?;
                        Some(scale_grad)
                    } else {
                        None
                    };
                    let grad_for_conv = if let Some(scale) = cached.batch_norm_scales.as_ref() {
                        self.backend
                            .channel_scale_f32(
                                &delta,
                                scale,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                config,
                            )
                            .map_err(|err| backend_error("GPU batch-norm delta scale", err))?
                    } else {
                        delta.clone()
                    };
                    let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
                    let weight_grad = self
                        .backend
                        .conv2d_backward_weights_f32(
                            current_input,
                            &grad_for_conv,
                            &[spec.filters, grouped_channels, spec.size, spec.size],
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            config,
                        )
                        .map_err(|err| backend_error("GPU convolution weight backward", err))?;
                    let input_grad = self
                        .backend
                        .conv2d_backward_input_f32(
                            &grad_for_conv,
                            &cached.raw_weights,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            config,
                        )
                        .map_err(|err| backend_error("GPU convolution input backward", err))?;
                    self.accumulate_device_train_buffer(owner, weight_grad, bias_grad, scale_grad)?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), Some(LayerParams::None)) => {
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        forward
                            .gpu_layer_outputs
                            .get(layer.index - 1)
                            .ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing previous GPU layer output for maxpool layer #{}",
                                    layer.index
                                ))
                            })?
                    };
                    let input_grad = self
                        .backend
                        .maxpool_backward_f32(
                            current_input,
                            &grad,
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            config,
                        )
                        .map_err(|err| backend_error("GPU maxpool backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LocalAvgPool(spec), Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .local_avgpool_backward_f32(
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            config,
                        )
                        .map_err(|err| backend_error("GPU local_avgpool backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::AvgPool, Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .avgpool_backward_f32(
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            config,
                        )
                        .map_err(|err| backend_error("GPU avgpool backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Route(spec), Some(LayerParams::None)) => {
                    let mut input_shapes = Vec::with_capacity(spec.layers.len());
                    let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
                    let mut route_channels = Vec::with_capacity(spec.layers.len());
                    let mut route_total_channels = Vec::with_capacity(spec.layers.len());
                    let mut source_indices = Vec::with_capacity(spec.layers.len());
                    for raw_index in &spec.layers {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source_layer =
                            self.network
                                .layers
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let chunk_channels =
                            source_layer.output_shape.channels / spec.groups.max(1);
                        input_shapes.push(tensor_backend_shape(source_layer.output_shape).to_vec());
                        route_channel_offsets.push(chunk_channels * spec.group_id);
                        route_channels.push(chunk_channels);
                        route_total_channels.push(source_layer.output_shape.channels);
                        source_indices.push(source_index);
                    }
                    let grads = self
                        .backend
                        .route_backward_f32(
                            &grad,
                            &input_shapes,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| backend_error("GPU route backward", err))?;
                    for (source_index, source_grad) in source_indices.into_iter().zip(grads) {
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            source_grad,
                        )?;
                    }
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::None)) => {
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for shortcut layer #{}",
                                layer.index
                            ))
                        })?;
                    let accum_grad = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| backend_error("GPU shortcut activation backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        accum_grad.clone(),
                        &mut layer_output_gradients,
                    )?;
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            accum_grad.clone(),
                        )?;
                    }
                }
                (LayerKind::Upsample(spec), Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .upsample_backward_f32(
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.output_shape.height,
                            layer.output_shape.width,
                            spec.stride.unsigned_abs(),
                            spec.scale,
                            config,
                        )
                        .map_err(|err| backend_error("GPU upsample backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), Some(LayerParams::None))
                | (LayerKind::GaussianYolo(_), Some(LayerParams::None))
                | (LayerKind::Region(_), Some(LayerParams::None))
                | (LayerKind::Softmax { .. }, Some(LayerParams::None))
                | (LayerKind::Cost { .. }, Some(LayerParams::None))
                | (LayerKind::Dropout { .. }, Some(LayerParams::None))
                | (LayerKind::ScaleChannels(_), Some(LayerParams::None))
                | (LayerKind::Sam(_), Some(LayerParams::None))
                | (LayerKind::Contrastive { .. }, Some(LayerParams::None)) => {
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        grad,
                        &mut layer_output_gradients,
                    )?;
                }
                _ => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{}_gpu_backward", layer.kind.name()),
                    });
                }
            }
        }

        self.update_gpu_convolution_batch_norm_running_stats_from_forward(forward)?;
        Ok(())
    }

    fn download_tensor(
        &self,
        tensor: &B::Tensor,
        shape: TensorShape,
    ) -> std::result::Result<Tensor, B::Error> {
        let data = self.backend.download_f32(tensor)?;
        Ok(Tensor { shape, data })
    }

    fn accumulate_previous_gpu_gradient(
        &self,
        layer_index: usize,
        gradient: B::Tensor,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        if layer_index == 0 {
            return Ok(());
        }
        self.accumulate_gpu_gradient(&mut layer_output_gradients[layer_index - 1], gradient)
    }

    fn accumulate_gpu_gradient(
        &self,
        slot: &mut Option<B::Tensor>,
        gradient: B::Tensor,
    ) -> Result<()> {
        *slot = Some(match slot.take() {
            Some(existing) => self
                .backend
                .vector_add_f32(&existing, &gradient, ElementwiseLaunchConfig::default())
                .map_err(|err| backend_error("GPU gradient accumulation", err))?,
            None => gradient,
        });
        Ok(())
    }

    fn accumulate_device_train_buffer(
        &self,
        owner: usize,
        weight_grad: B::Tensor,
        bias_grad: B::Tensor,
        scale_grad: Option<B::Tensor>,
    ) -> Result<()> {
        let Some(train) = self
            .device_train_buffers
            .borrow()
            .get(owner)
            .and_then(Clone::clone)
        else {
            return Err(Error::InvalidConfig(format!(
                "missing GPU train buffers for layer #{owner}"
            )));
        };
        let weight_updates = self
            .backend
            .saxpy_f32(
                1.0,
                &weight_grad,
                &train.weight_updates,
                ElementwiseLaunchConfig::default(),
            )
            .map_err(|err| backend_error("GPU weight gradient accumulation", err))?;
        let bias_updates = self
            .backend
            .saxpy_f32(
                1.0,
                &bias_grad,
                &train.bias_updates,
                ElementwiseLaunchConfig::default(),
            )
            .map_err(|err| backend_error("GPU bias gradient accumulation", err))?;
        let scale_updates = match (train.scale_updates, scale_grad) {
            (Some(scale_updates), Some(scale_grad)) => Some(
                self.backend
                    .saxpy_f32(
                        1.0,
                        &scale_grad,
                        &scale_updates,
                        ElementwiseLaunchConfig::default(),
                    )
                    .map_err(|err| {
                        backend_error("GPU batch-norm scale gradient accumulation", err)
                    })?,
            ),
            (None, None) => None,
            (Some(scale_updates), None) => Some(scale_updates),
            (None, Some(_)) => {
                return Err(Error::InvalidConfig(format!(
                    "GPU batch-norm scale gradient produced for non-BN layer #{owner}"
                )));
            }
        };
        self.device_train_buffers.borrow_mut()[owner] = Some(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
        });
        Ok(())
    }

    fn batch_norm_scale_gradient_gpu(
        &self,
        raw_output: &B::Tensor,
        delta: &B::Tensor,
        batch_norm: &BatchNormParams,
        output_shape: TensorShape,
    ) -> Result<B::Tensor> {
        let raw_output = self
            .backend
            .download_f32(raw_output)
            .map_err(|err| backend_error("download GPU batch-norm raw output", err))?;
        let delta = self
            .backend
            .download_f32(delta)
            .map_err(|err| backend_error("download GPU batch-norm delta", err))?;
        if raw_output.len() != delta.len() {
            return Err(Error::InvalidConfig(format!(
                "GPU batch-norm scale gradient shape mismatch: raw output has {} values, delta has {}",
                raw_output.len(),
                delta.len()
            )));
        }
        if batch_norm.rolling_mean.len() != output_shape.channels
            || batch_norm.rolling_variance.len() != output_shape.channels
        {
            return Err(Error::InvalidConfig(format!(
                "GPU batch-norm rolling stat mismatch: expected {} channels",
                output_shape.channels
            )));
        }

        let channel_stride = output_shape.width * output_shape.height;
        let mut scale_grad = vec![0.0; output_shape.channels];
        for b in 0..output_shape.batch {
            for (channel, grad) in scale_grad.iter_mut().enumerate() {
                let inv_std =
                    1.0 / (batch_norm.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt();
                let base = (b * output_shape.channels + channel) * channel_stride;
                for idx in 0..channel_stride {
                    let offset = base + idx;
                    let normalized =
                        (raw_output[offset] - batch_norm.rolling_mean[channel]) * inv_std;
                    *grad += delta[offset] * normalized;
                }
            }
        }

        let scale_grad = self
            .backend
            .upload_f32(&[output_shape.channels], &scale_grad)
            .map_err(|err| backend_error("upload GPU batch-norm scale gradient", err))?;
        Ok(scale_grad)
    }

    fn update_gpu_batch_norm_scales(
        &mut self,
        owner: usize,
        scale_updates: Option<B::Tensor>,
        lr: f32,
        momentum: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Option<B::Tensor>> {
        let Some(scale_updates) = scale_updates else {
            return Ok(None);
        };
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        let Some(batch_norm) = params.batch_norm.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "GPU batch-norm scale updates exist for non-BN layer #{owner}"
            )));
        };

        let scales = self
            .backend
            .upload_f32(&[batch_norm.scales.len()], &batch_norm.scales)
            .map_err(|err| backend_error("upload GPU batch-norm scales", err))?;
        let (updated_scales, updated_scale_updates) = self
            .backend
            .sgd_update_f32(&scales, &scale_updates, lr, momentum, 0.0, batch, config)
            .map_err(|err| backend_error("GPU batch-norm scale update", err))?;
        batch_norm.scales = self
            .backend
            .download_f32(&updated_scales)
            .map_err(|err| backend_error("download GPU batch-norm scales", err))?;
        Ok(Some(updated_scale_updates))
    }

    fn update_gpu_convolution_batch_norm_running_stats(
        &mut self,
        owner: usize,
        mean: &[f32],
        variance: &[f32],
    ) -> Result<()> {
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        let Some(batch_norm) = params.batch_norm.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing batch-norm params for convolutional layer #{owner}"
            )));
        };
        update_batch_norm_running_stats(batch_norm, mean, variance, 0.9)?;

        let cached = self
            .device_conv_params
            .borrow()
            .get(owner)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached GPU convolution params for layer #{owner}"
                ))
            })?;
        let layer =
            self.network.layers.get(owner).ok_or_else(|| {
                Error::InvalidConfig(format!("missing convolutional layer #{owner}"))
            })?;
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Err(Error::InvalidConfig(format!(
                "layer #{owner} is not convolutional"
            )));
        };
        let params = resolve_convolutional_params(&self.network, &self.params, owner)?;
        let refreshed = self.cached_convolution_upload(
            spec,
            layer.input_shape.channels,
            params,
            Some((cached.raw_weights, cached.raw_biases)),
        )?;
        self.device_conv_params.borrow_mut()[owner] = Some(refreshed);
        Ok(())
    }

    fn update_gpu_convolution_batch_norm_running_stats_from_forward(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
    ) -> Result<()> {
        let config = ElementwiseLaunchConfig::default();
        let mut batch_norm_stats = Vec::new();
        for layer in &self.network.layers {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize || spec.share_index.is_some() {
                continue;
            }

            let current_input =
                if layer.index == 0 {
                    &forward.input_gpu
                } else {
                    forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous GPU layer output for batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
                };
            let cached = self
                .device_conv_params
                .borrow()
                .get(layer.index)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing cached GPU convolution params for layer #{}",
                        layer.index
                    ))
                })?;
            let raw_output = self
                .backend
                .conv2d_f32(
                    current_input,
                    &cached.raw_weights,
                    &tensor_backend_shape(layer.output_shape),
                    layer.input_shape.channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.filters,
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.dilation,
                    spec.padding,
                    spec.groups.max(1),
                    config,
                )
                .map_err(|err| backend_error("GPU batch-norm raw convolution stats", err))?;
            let (mean, variance) =
                self.batch_norm_channel_stats_gpu(&raw_output, layer.output_shape)?;
            batch_norm_stats.push((layer.index, mean, variance));
        }

        for (owner, mean, variance) in batch_norm_stats {
            self.update_gpu_convolution_batch_norm_running_stats(owner, &mean, &variance)?;
        }
        Ok(())
    }

    fn batch_norm_channel_stats_gpu(
        &self,
        raw_output: &B::Tensor,
        output_shape: TensorShape,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let raw_output = self
            .backend
            .download_f32(raw_output)
            .map_err(|err| backend_error("download GPU batch-norm raw output stats", err))?;
        Ok(channel_mean_variance(&Tensor {
            shape: output_shape,
            data: raw_output,
        }))
    }

    fn clear_device_conv_params(&mut self) {
        for entry in self.device_conv_params.get_mut().iter_mut() {
            *entry = None;
        }
    }

    fn clear_device_train_buffers(&mut self) {
        for entry in self.device_train_buffers.get_mut().iter_mut() {
            *entry = None;
        }
    }

    fn warm_device_conv_params(&self) -> Result<()> {
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            if let (
                LayerKind::Convolutional(spec),
                LayerParams::Convolutional(_) | LayerParams::None,
            ) = (&layer.kind, params)
            {
                let conv = resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                let _ =
                    self.cached_convolution_params(owner, layer.input_shape.channels, spec, conv)?;
            }
        }
        Ok(())
    }

    fn cached_convolution_params(
        &self,
        layer_index: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        if let Some(cached) = self.device_conv_params.borrow()[layer_index].clone() {
            return Ok(cached);
        }

        let (folded_biases, folded_weights) =
            fold_convolutional_params_for_inference(spec, params, input_channels)?;
        let raw_weights = self
            .backend
            .upload_f32(
                &[
                    spec.filters,
                    input_channels / spec.groups.max(1),
                    spec.size,
                    spec.size,
                ],
                &params.weights,
            )
            .map_err(|err| backend_error("upload raw convolution weights", err))?;
        let raw_biases = self
            .backend
            .upload_f32(&[spec.filters], &params.biases)
            .map_err(|err| backend_error("upload raw convolution biases", err))?;
        let uploaded = GpuConvolutionLayerParams {
            raw_weights: raw_weights.clone(),
            raw_biases: raw_biases.clone(),
            weights: if spec.batch_normalize {
                self.backend
                    .upload_f32(
                        &[
                            spec.filters,
                            input_channels / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                        &folded_weights,
                    )
                    .map_err(|err| backend_error("upload folded convolution weights", err))?
            } else {
                raw_weights
            },
            biases: if spec.batch_normalize {
                self.backend
                    .upload_f32(&[spec.filters], &folded_biases)
                    .map_err(|err| backend_error("upload folded convolution biases", err))?
            } else {
                raw_biases
            },
            batch_norm_scales: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    let scales = batch_norm
                        .scales
                        .iter()
                        .zip(&batch_norm.rolling_variance)
                        .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
                        .collect::<Vec<_>>();
                    self.backend
                        .upload_f32(&[spec.filters], &scales)
                        .map_err(|err| backend_error("upload batch-norm scales", err))
                })
                .transpose()?,
        };

        let mut cached = self.device_conv_params.borrow_mut();
        let entry = cached.get_mut(layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing cached device params for layer #{layer_index}"
            ))
        })?;
        if entry.is_none() {
            *entry = Some(uploaded.clone());
        }
        Ok(entry.clone().expect("device params were just initialized"))
    }

    fn refresh_device_convolution_params(
        &self,
        owner: usize,
        raw_weights: B::Tensor,
        raw_biases: B::Tensor,
        weight_updates: B::Tensor,
        bias_updates: B::Tensor,
        scale_updates: Option<B::Tensor>,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let layer =
            self.network.layers.get(owner).ok_or_else(|| {
                Error::InvalidConfig(format!("missing convolutional layer #{owner}"))
            })?;
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Err(Error::InvalidConfig(format!(
                "layer #{owner} is not convolutional"
            )));
        };
        let weights_host = self
            .backend
            .download_f32(&raw_weights)
            .map_err(|err| backend_error("download raw convolution weights", err))?;
        let biases_host = self
            .backend
            .download_f32(&raw_biases)
            .map_err(|err| backend_error("download raw convolution biases", err))?;
        let params = resolve_convolutional_params(&self.network, &self.params, owner)?;
        let host_params = ConvolutionalParams {
            biases: biases_host,
            batch_norm: params.batch_norm.clone(),
            weights: weights_host,
        };
        let refreshed = self.cached_convolution_upload(
            spec,
            layer.input_shape.channels,
            &host_params,
            Some((raw_weights, raw_biases)),
        )?;
        self.device_conv_params.borrow_mut()[owner] = Some(refreshed.clone());
        self.device_train_buffers.borrow_mut()[owner] = Some(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
        });
        Ok(refreshed)
    }

    fn cached_convolution_upload(
        &self,
        spec: &ConvolutionalSpec,
        input_channels: usize,
        params: &ConvolutionalParams,
        raw: Option<(B::Tensor, B::Tensor)>,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let (folded_biases, folded_weights) =
            fold_convolutional_params_for_inference(spec, params, input_channels)?;
        let (raw_weights, raw_biases) = match raw {
            Some(existing) => existing,
            None => (
                self.backend
                    .upload_f32(
                        &[
                            spec.filters,
                            input_channels / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                        &params.weights,
                    )
                    .map_err(|err| backend_error("upload raw convolution weights", err))?,
                self.backend
                    .upload_f32(&[spec.filters], &params.biases)
                    .map_err(|err| backend_error("upload raw convolution biases", err))?,
            ),
        };
        Ok(GpuConvolutionLayerParams {
            raw_weights: raw_weights.clone(),
            raw_biases: raw_biases.clone(),
            weights: if spec.batch_normalize {
                self.backend
                    .upload_f32(
                        &[
                            spec.filters,
                            input_channels / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                        &folded_weights,
                    )
                    .map_err(|err| backend_error("upload folded convolution weights", err))?
            } else {
                raw_weights
            },
            biases: if spec.batch_normalize {
                self.backend
                    .upload_f32(&[spec.filters], &folded_biases)
                    .map_err(|err| backend_error("upload folded convolution biases", err))?
            } else {
                raw_biases
            },
            batch_norm_scales: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    let scales = batch_norm
                        .scales
                        .iter()
                        .zip(&batch_norm.rolling_variance)
                        .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
                        .collect::<Vec<_>>();
                    self.backend
                        .upload_f32(&[spec.filters], &scales)
                        .map_err(|err| backend_error("upload batch-norm scales", err))
                })
                .transpose()?,
        })
    }

    fn sync_convolution_owner_from_device(
        &mut self,
        owner: usize,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let raw_weights = self
            .backend
            .download_f32(&cached.raw_weights)
            .map_err(|err| backend_error("download synced raw convolution weights", err))?;
        let raw_biases = self
            .backend
            .download_f32(&cached.raw_biases)
            .map_err(|err| backend_error("download synced raw convolution biases", err))?;
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        params.weights = raw_weights;
        params.biases = raw_biases;
        Ok(())
    }
}

fn default_layer_params(layer: &crate::model::Layer) -> LayerParams {
    match &layer.kind {
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerParams::None;
            }
            let filters = spec.filters;
            let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
            let weight_count = filters * grouped_channels * spec.size * spec.size;
            LayerParams::Convolutional(ConvolutionalParams {
                biases: vec![0.0; filters],
                batch_norm: spec.batch_normalize.then(|| BatchNormParams {
                    scales: vec![0.0; filters],
                    rolling_mean: vec![0.0; filters],
                    rolling_variance: vec![0.0; filters],
                }),
                weights: vec![0.0; weight_count],
            })
        }
        LayerKind::Connected(spec) => {
            let output = spec.output;
            let input = layer.input_shape.element_count();
            LayerParams::Connected(ConnectedParams {
                biases: vec![0.0; output],
                batch_norm: spec.batch_normalize.then(|| BatchNormParams {
                    scales: vec![0.0; output],
                    rolling_mean: vec![0.0; output],
                    rolling_variance: vec![0.0; output],
                }),
                weights: vec![0.0; output * input],
            })
        }
        _ => LayerParams::None,
    }
}

fn default_layer_train_buffers(layer: &crate::model::Layer) -> LayerTrainBuffers {
    match &layer.kind {
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerTrainBuffers::None;
            }
            let filters = spec.filters;
            let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
            let weight_count = filters * grouped_channels * spec.size * spec.size;
            LayerTrainBuffers::Convolutional(ConvolutionalTrainBuffers {
                bias_updates: vec![0.0; filters],
                batch_norm: spec.batch_normalize.then(|| BatchNormTrainBuffers {
                    scale_updates: vec![0.0; filters],
                    mean_delta: vec![0.0; filters],
                    variance_delta: vec![0.0; filters],
                }),
                weight_updates: vec![0.0; weight_count],
            })
        }
        LayerKind::Connected(spec) => {
            let output = spec.output;
            let input = layer.input_shape.element_count();
            LayerTrainBuffers::Connected(ConnectedTrainBuffers {
                bias_updates: vec![0.0; output],
                batch_norm: spec.batch_normalize.then(|| BatchNormTrainBuffers {
                    scale_updates: vec![0.0; output],
                    mean_delta: vec![0.0; output],
                    variance_delta: vec![0.0; output],
                }),
                weight_updates: vec![0.0; output * input],
            })
        }
        _ => LayerTrainBuffers::None,
    }
}

fn accumulate_vec(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(Error::InvalidConfig(format!(
            "gradient accumulation length mismatch: expected {}, got {}",
            dst.len(),
            src.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
    Ok(())
}

fn sgd_update(
    weights: &mut [f32],
    updates: &mut [f32],
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    let batch = batch.max(1) as f32;
    let step = lr / batch;
    for (weight, update) in weights.iter_mut().zip(updates.iter_mut()) {
        *update += -decay * batch * *weight;
        *weight += step * *update;
        *update *= momentum;
    }
}

fn accumulate_convolutional_param_gradients(
    dst: &mut LayerParams,
    src: &ConvolutionalParams,
) -> Result<()> {
    let LayerParams::Convolutional(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected convolutional gradient destination".to_owned(),
        ));
    };
    accumulate_vec(&mut dst.biases, &src.biases)?;
    accumulate_vec(&mut dst.weights, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scales, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.rolling_mean, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.rolling_variance, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn resolve_convolutional_owner(network: &Network, layer_index: usize) -> Result<usize> {
    let Some(layer) = network.layers.get(layer_index) else {
        return Err(Error::InvalidConfig(format!(
            "missing convolutional layer #{layer_index}"
        )));
    };
    let LayerKind::Convolutional(spec) = &layer.kind else {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} is not convolutional"
        )));
    };
    Ok(spec.share_index.unwrap_or(layer_index))
}

fn resolve_convolutional_params<'a>(
    network: &Network,
    params: &'a [LayerParams],
    layer_index: usize,
) -> Result<&'a ConvolutionalParams> {
    let owner = resolve_convolutional_owner(network, layer_index)?;
    match params.get(owner) {
        Some(LayerParams::Convolutional(conv)) => Ok(conv),
        _ => Err(Error::InvalidConfig(format!(
            "missing convolutional parameters for layer #{} owner #{}",
            layer_index, owner
        ))),
    }
}

fn fold_convolutional_params_for_inference(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input_channels: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let groups = spec.groups.max(1);
    if spec.filters % groups != 0 {
        return Err(Error::InvalidConfig(format!(
            "convolution filters {} are not divisible by groups {}",
            spec.filters, groups
        )));
    }
    if input_channels % groups != 0 {
        return Err(Error::InvalidConfig(format!(
            "input channels {input_channels} are not divisible by groups {groups}"
        )));
    }
    if params.biases.len() != spec.filters {
        return Err(Error::InvalidConfig(format!(
            "convolution bias length mismatch: expected {}, got {}",
            spec.filters,
            params.biases.len()
        )));
    }

    let channels_per_group = input_channels / groups;
    let weights_per_filter = channels_per_group * spec.size * spec.size;
    let expected_weight_count = spec.filters * weights_per_filter;
    if params.weights.len() != expected_weight_count {
        return Err(Error::InvalidConfig(format!(
            "convolution weight length mismatch: expected {expected_weight_count}, got {}",
            params.weights.len()
        )));
    }

    let mut folded_biases = params.biases.clone();
    let mut folded_weights = params.weights.clone();
    if let Some(batch_norm) = params.batch_norm.as_ref() {
        if batch_norm.scales.len() != spec.filters
            || batch_norm.rolling_mean.len() != spec.filters
            || batch_norm.rolling_variance.len() != spec.filters
        {
            return Err(Error::InvalidConfig(format!(
                "batch norm parameter length mismatch for {} filters",
                spec.filters
            )));
        }

        for filter in 0..spec.filters {
            let scale = batch_norm.scales[filter]
                / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
            folded_biases[filter] = folded_biases[filter] - batch_norm.rolling_mean[filter] * scale;
            let weight_base = filter * weights_per_filter;
            for weight in &mut folded_weights[weight_base..weight_base + weights_per_filter] {
                *weight *= scale;
            }
        }
    }

    Ok((folded_biases, folded_weights))
}

fn load_weights_into_params(
    network: &Network,
    params: &mut [LayerParams],
    path: impl AsRef<Path>,
) -> Result<WeightsHeader> {
    let mut file = File::open(path)?;
    let header = read_header(&mut file)?;
    let transpose_connected = header.major > 1000 || header.minor > 1000;

    for (layer, params) in network.layers.iter().zip(params.iter_mut()) {
        match params {
            LayerParams::None => {}
            LayerParams::Convolutional(conv) => {
                read_f32_into(&mut file, &mut conv.biases)?;
                if let Some(batch_norm) = conv.batch_norm.as_mut() {
                    read_f32_into(&mut file, &mut batch_norm.scales)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
                }
                read_f32_into(&mut file, &mut conv.weights)?;
            }
            LayerParams::Connected(connected) => {
                read_f32_into(&mut file, &mut connected.biases)?;
                read_f32_into(&mut file, &mut connected.weights)?;
                if transpose_connected {
                    transpose_connected_weights(layer, connected)?;
                }
                if let Some(batch_norm) = connected.batch_norm.as_mut() {
                    read_f32_into(&mut file, &mut batch_norm.scales)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
                }
            }
        }
    }

    Ok(header)
}

fn tensor_backend_shape(shape: TensorShape) -> [usize; 4] {
    [shape.batch, shape.channels, shape.height, shape.width]
}

fn layer_uses_previous_input(kind: &LayerKind) -> bool {
    !matches!(kind, LayerKind::Route(_))
}

fn gpu_layer_output_use_counts(network: &Network) -> Result<Vec<usize>> {
    let mut use_counts = vec![0usize; network.layers.len()];
    for layer in &network.layers {
        if layer.index > 0 && layer_uses_previous_input(&layer.kind) {
            use_counts[layer.index - 1] += 1;
        }

        match &layer.kind {
            LayerKind::Route(spec) => {
                for raw_index in &spec.layers {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            LayerKind::Shortcut(spec) => {
                for raw_index in &spec.from {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            _ => {}
        }
    }

    Ok(use_counts)
}

fn release_gpu_source<T: Clone>(
    source_index: usize,
    remaining_uses: &mut [usize],
    gpu_layer_outputs: &mut [Option<T>],
) -> Result<()> {
    let remaining = remaining_uses.get_mut(source_index).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "missing GPU use-count entry for source layer #{source_index}"
        ))
    })?;
    if *remaining == 0 {
        return Err(Error::InvalidConfig(format!(
            "GPU source layer #{source_index} was released too early"
        )));
    }
    *remaining -= 1;
    if *remaining == 0 {
        gpu_layer_outputs[source_index] = None;
    }
    Ok(())
}

fn release_gpu_sources_for_layer<T: Clone>(
    layer_index: usize,
    kind: &LayerKind,
    remaining_uses: &mut [usize],
    gpu_layer_outputs: &mut [Option<T>],
) -> Result<()> {
    if layer_index > 0 && layer_uses_previous_input(kind) {
        release_gpu_source(layer_index - 1, remaining_uses, gpu_layer_outputs)?;
    }

    match kind {
        LayerKind::Route(spec) => {
            for raw_index in &spec.layers {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        LayerKind::Shortcut(spec) => {
            for raw_index in &spec.from {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn gpu_activation_kind(layer_index: usize, activation: &Activation) -> Result<F32ActivationKind> {
    activation.to_cuda_f32_kind().ok_or_else(|| {
        Error::InvalidConfig(format!(
            "layer #{layer_index} uses activation {:?}, which the GPU f32 backend does not support",
            activation
        ))
    })
}

fn backend_error(err_context: &str, err: impl std::fmt::Display) -> Error {
    Error::Backend(format!("{err_context}: {err}"))
}

fn read_header(file: &mut File) -> Result<WeightsHeader> {
    let major = read_i32(file)?;
    let minor = read_i32(file)?;
    let revision = read_i32(file)?;
    let (seen, header_bytes) = if (major * 10 + minor) >= 2 {
        (read_u64(file)?, 3 * 4 + 8)
    } else {
        (u64::from(read_u32(file)?), 3 * 4 + 4)
    };

    Ok(WeightsHeader {
        major,
        minor,
        revision,
        seen,
        header_bytes,
    })
}

fn read_f32_into(file: &mut File, dst: &mut [f32]) -> Result<()> {
    let mut bytes = vec![0u8; dst.len() * 4];
    file.read_exact(&mut bytes)?;
    for (chunk, value) in bytes.chunks_exact(4).zip(dst.iter_mut()) {
        *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn read_i32(file: &mut File) -> Result<i32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn transpose_connected_weights(
    layer: &crate::model::Layer,
    connected: &mut ConnectedParams,
) -> Result<()> {
    let LayerKind::Connected(spec) = &layer.kind else {
        return Err(Error::InvalidConfig("expected connected layer".to_owned()));
    };

    let rows = layer.input_shape.element_count();
    let cols = spec.output;
    let mut transpose = vec![0.0; connected.weights.len()];
    for x in 0..rows {
        for y in 0..cols {
            transpose[y * rows + x] = connected.weights[x * cols + y];
        }
    }
    connected.weights = transpose;
    Ok(())
}

fn forward_convolutional(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    forward_convolutional_with_cache(spec, params, input, output_shape).0
}

fn forward_convolutional_with_cache(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> (Tensor, Option<Tensor>) {
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);

    for b in 0..input.shape.batch {
        for filter in 0..spec.filters {
            let group = filter / filters_per_group.max(1);
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut acc = 0.0;
                    for ic in 0..channels_per_group {
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let in_y = oy * spec.stride_y + ky * spec.dilation;
                                let in_x = ox * spec.stride_x + kx * spec.dilation;
                                let src_y = in_y as isize - spec.padding as isize;
                                let src_x = in_x as isize - spec.padding as isize;

                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }

                                let channel = group * channels_per_group + ic;
                                let input_value =
                                    input.get(b, channel, src_y as usize, src_x as usize);
                                let weight_index = ((filter * channels_per_group + ic) * spec.size
                                    + ky)
                                    * spec.size
                                    + kx;
                                acc += input_value * params.weights[weight_index];
                            }
                        }
                    }

                    let biased =
                        normalize_value(acc, filter, &params.biases, params.batch_norm.as_ref());
                    activation_input.set(b, filter, oy, ox, biased);
                    output.set(b, filter, oy, ox, activate(spec.activation.clone(), biased));
                }
            }
        }
    }

    (output, Some(activation_input))
}

fn raw_convolution_output(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    let mut output = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for filter in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let value =
                        convolution_raw_value(spec, &params.weights, input, b, filter, y, x);
                    output.set(b, filter, y, x, value);
                }
            }
        }
    }
    output
}

fn channel_mean_variance(tensor: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let spatial = tensor.shape.width * tensor.shape.height;
    let sample_count = tensor.shape.batch * spatial;
    let mut mean = vec![0.0; tensor.shape.channels];
    for b in 0..tensor.shape.batch {
        for (channel, slot) in mean.iter_mut().enumerate() {
            for idx in 0..spatial {
                let y = idx / tensor.shape.width;
                let x = idx % tensor.shape.width;
                *slot += tensor.get(b, channel, y, x);
            }
        }
    }
    if sample_count != 0 {
        let scale = 1.0 / sample_count as f32;
        for value in &mut mean {
            *value *= scale;
        }
    }

    let mut variance = vec![0.0; tensor.shape.channels];
    for b in 0..tensor.shape.batch {
        for (channel, slot) in variance.iter_mut().enumerate() {
            for idx in 0..spatial {
                let y = idx / tensor.shape.width;
                let x = idx % tensor.shape.width;
                let diff = tensor.get(b, channel, y, x) - mean[channel];
                *slot += diff * diff;
            }
        }
    }
    if sample_count > 1 {
        let scale = 1.0 / (sample_count - 1) as f32;
        for value in &mut variance {
            *value *= scale;
        }
    }

    (mean, variance)
}

fn update_batch_norm_running_stats(
    batch_norm: &mut BatchNormParams,
    mean: &[f32],
    variance: &[f32],
    keep: f32,
) -> Result<()> {
    if batch_norm.rolling_mean.len() != mean.len()
        || batch_norm.rolling_variance.len() != variance.len()
    {
        return Err(Error::InvalidConfig(format!(
            "batch-norm running-stat mismatch: rolling mean {}, rolling variance {}, mean {}, variance {}",
            batch_norm.rolling_mean.len(),
            batch_norm.rolling_variance.len(),
            mean.len(),
            variance.len()
        )));
    }
    let update = 1.0 - keep;
    for (dst, src) in batch_norm.rolling_mean.iter_mut().zip(mean) {
        *dst = keep * *dst + update * *src;
    }
    for (dst, src) in batch_norm.rolling_variance.iter_mut().zip(variance) {
        *dst = keep * *dst + update * *src;
    }
    Ok(())
}

fn forward_connected(
    spec: &ConnectedSpec,
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    let mut output = Tensor::zeros(output_shape);
    let inputs = input.shape.element_count();

    for b in 0..input.shape.batch {
        let src_offset = b * inputs;
        let src = &input.data[src_offset..src_offset + inputs];

        for out_idx in 0..spec.output {
            let weight_base = out_idx * inputs;
            let mut acc = 0.0;
            for (idx, src_value) in src.iter().enumerate() {
                acc += *src_value * params.weights[weight_base + idx];
            }

            let value = activate(
                spec.activation.clone(),
                normalize_value(acc, out_idx, &params.biases, params.batch_norm.as_ref()),
            );
            output.data[b * spec.output + out_idx] = value;
        }
    }

    output
}

fn backward_convolutional(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, ConvolutionalParams)> {
    if output_gradient.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "convolution backward gradient shape mismatch: expected {output_shape}, got {}",
            output_gradient.shape
        )));
    }

    let output = forward_convolutional(spec, params, input, output_shape);
    let delta = apply_output_activation_gradient(
        &output,
        output_gradient,
        &spec.activation,
        "convolutional",
        activation_input,
    )?;

    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);
    // Use the pre-allocated workspace as scratch for bias_grad and weight_grad.
    // This avoids a heap allocation per backward call (C++ net->workspace parity).
    let bias_size = spec.filters;
    let weight_size = params.weights.len();
    let needed = bias_size + weight_size;
    if workspace.len() < needed {
        workspace.resize(needed, 0.0);
    }
    workspace[..bias_size].fill(0.0);
    workspace[bias_size..needed].fill(0.0);
    // Split the workspace into two mutable views used as bias_grad / weight_grad.
    let (bias_grad, weight_grad) = workspace[..needed].split_at_mut(bias_size);
    let mut input_grad = Tensor::zeros(input.shape);
    let mut batch_norm_grad = params
        .batch_norm
        .as_ref()
        .map(|batch_norm| BatchNormParams {
            scales: vec![0.0; batch_norm.scales.len()],
            rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
            rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
        });

    for b in 0..input.shape.batch {
        for filter in 0..spec.filters {
            let group = filter / filters_per_group.max(1);
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut grad_value = delta.get(b, filter, oy, ox);
                    if let Some(batch_norm) = params.batch_norm.as_ref() {
                        let raw_value =
                            convolution_raw_value(spec, &params.weights, input, b, filter, oy, ox);
                        let inv_std =
                            1.0 / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
                        let normalized = (raw_value - batch_norm.rolling_mean[filter]) * inv_std;
                        bias_grad[filter] += grad_value;
                        if let Some(batch_norm_grad) = batch_norm_grad.as_mut() {
                            batch_norm_grad.scales[filter] += grad_value * normalized;
                        }
                        grad_value *= batch_norm.scales[filter] * inv_std;
                    } else {
                        bias_grad[filter] += grad_value;
                    }

                    for ic in 0..channels_per_group {
                        let channel = group * channels_per_group + ic;
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let in_y = oy * spec.stride_y + ky * spec.dilation;
                                let in_x = ox * spec.stride_x + kx * spec.dilation;
                                let src_y = in_y as isize - spec.padding as isize;
                                let src_x = in_x as isize - spec.padding as isize;

                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }

                                let src_y = src_y as usize;
                                let src_x = src_x as usize;
                                let weight_index = ((filter * channels_per_group + ic) * spec.size
                                    + ky)
                                    * spec.size
                                    + kx;
                                weight_grad[weight_index] +=
                                    input.get(b, channel, src_y, src_x) * grad_value;

                                let input_index = input.offset(b, channel, src_y, src_x);
                                input_grad.data[input_index] +=
                                    params.weights[weight_index] * grad_value;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((
        input_grad,
        ConvolutionalParams {
            biases: bias_grad.to_vec(),
            batch_norm: batch_norm_grad,
            weights: weight_grad.to_vec(),
        },
    ))
}

fn convolution_raw_value(
    spec: &ConvolutionalSpec,
    weights: &[f32],
    input: &Tensor,
    batch: usize,
    filter: usize,
    out_y: usize,
    out_x: usize,
) -> f32 {
    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);
    let group = filter / filters_per_group.max(1);
    let mut acc = 0.0;

    for ic in 0..channels_per_group {
        for ky in 0..spec.size {
            for kx in 0..spec.size {
                let in_y = out_y * spec.stride_y + ky * spec.dilation;
                let in_x = out_x * spec.stride_x + kx * spec.dilation;
                let src_y = in_y as isize - spec.padding as isize;
                let src_x = in_x as isize - spec.padding as isize;
                if src_y < 0
                    || src_y >= input.shape.height as isize
                    || src_x < 0
                    || src_x >= input.shape.width as isize
                {
                    continue;
                }

                let channel = group * channels_per_group + ic;
                let weight_index =
                    ((filter * channels_per_group + ic) * spec.size + ky) * spec.size + kx;
                acc += input.get(batch, channel, src_y as usize, src_x as usize)
                    * weights[weight_index];
            }
        }
    }

    acc
}

fn backward_maxpool(
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    input: &Tensor,
    output_gradient: &Tensor,
    average: bool,
) -> Result<Tensor> {
    let mut input_grad = Tensor::zeros(input.shape);
    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            for oy in 0..output_gradient.shape.height {
                for ox in 0..output_gradient.shape.width {
                    let grad_value = output_gradient.get(b, c, oy, ox);
                    if average {
                        let mut valid = Vec::new();
                        for ky in 0..size {
                            for kx in 0..size {
                                let in_y = oy * stride_y + ky;
                                let in_x = ox * stride_x + kx;
                                let src_y = in_y as isize - (padding / 2) as isize;
                                let src_x = in_x as isize - (padding / 2) as isize;
                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }
                                valid.push((src_y as usize, src_x as usize));
                            }
                        }
                        if !valid.is_empty() {
                            let share = grad_value / valid.len() as f32;
                            for (y, x) in valid {
                                let idx = input.offset(b, c, y, x);
                                input_grad.data[idx] += share;
                            }
                        }
                    } else {
                        let mut best = None;
                        let mut best_value = f32::NEG_INFINITY;
                        for ky in 0..size {
                            for kx in 0..size {
                                let in_y = oy * stride_y + ky;
                                let in_x = ox * stride_x + kx;
                                let src_y = in_y as isize - (padding / 2) as isize;
                                let src_x = in_x as isize - (padding / 2) as isize;
                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }
                                let value = input.get(b, c, src_y as usize, src_x as usize);
                                if value > best_value {
                                    best_value = value;
                                    best = Some((src_y as usize, src_x as usize));
                                }
                            }
                        }
                        if let Some((y, x)) = best {
                            let idx = input.offset(b, c, y, x);
                            input_grad.data[idx] += grad_value;
                        }
                    }
                }
            }
        }
    }
    Ok(input_grad)
}

fn backward_route(
    spec: &RouteSpec,
    layer_index: usize,
    output_gradient: &Tensor,
    layer_output_gradients: &mut [Tensor],
    layers: &[crate::model::Layer],
) -> Result<()> {
    let mut out_channel = 0usize;
    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source_layer = layers.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        let chunk_channels = source_layer.output_shape.channels / spec.groups.max(1);
        let channel_start = chunk_channels * spec.group_id;
        let source_grad =
            layer_output_gradients
                .get_mut(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: *raw_index,
                })?;

        for b in 0..source_layer.output_shape.batch {
            for c in 0..chunk_channels {
                for y in 0..source_layer.output_shape.height {
                    for x in 0..source_layer.output_shape.width {
                        let grad = output_gradient.get(b, out_channel + c, y, x);
                        let idx = source_grad.offset(b, channel_start + c, y, x);
                        source_grad.data[idx] += grad;
                    }
                }
            }
        }

        out_channel += chunk_channels;
    }
    Ok(())
}

fn backward_shortcut(
    spec: &crate::model::ShortcutSpec,
    layer_index: usize,
    forward_outputs: &[Tensor],
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<Tensor> {
    let output = forward_outputs.get(layer_index).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "missing saved forward output for shortcut layer #{layer_index}"
        ))
    })?;
    apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "shortcut",
        activation_input,
    )
}

fn backward_upsample(
    spec: &crate::model::UpsampleSpec,
    output_gradient: &Tensor,
    input_shape: TensorShape,
) -> Result<Tensor> {
    let stride = spec.stride.unsigned_abs();
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "upsample stride cannot be zero".to_owned(),
        ));
    }

    let mut input_grad = Tensor::zeros(input_shape);
    if spec.reverse {
        for b in 0..input_shape.batch {
            for c in 0..input_shape.channels {
                for y in 0..input_shape.height {
                    for x in 0..input_shape.width {
                        let src_y = y / stride;
                        let src_x = x / stride;
                        input_grad.set(
                            b,
                            c,
                            y,
                            x,
                            spec.scale * output_gradient.get(b, c, src_y, src_x),
                        );
                    }
                }
            }
        }
    } else {
        for b in 0..output_gradient.shape.batch {
            for c in 0..output_gradient.shape.channels {
                for y in 0..output_gradient.shape.height {
                    for x in 0..output_gradient.shape.width {
                        let dst_y = y / stride;
                        let dst_x = x / stride;
                        let idx = input_grad.offset(b, c, dst_y, dst_x);
                        input_grad.data[idx] += spec.scale * output_gradient.get(b, c, y, x);
                    }
                }
            }
        }
    }
    Ok(input_grad)
}

fn apply_output_activation_gradient(
    output: &Tensor,
    output_gradient: &Tensor,
    activation: &Activation,
    layer_name: &str,
    activation_input: Option<&Tensor>,
) -> Result<Tensor> {
    if output.shape != output_gradient.shape {
        return Err(Error::InvalidConfig(format!(
            "{layer_name} backward gradient shape mismatch: expected {}, got {}",
            output.shape, output_gradient.shape
        )));
    }
    if let Some(cached_input) = activation_input
        && cached_input.shape != output.shape
    {
        return Err(Error::InvalidConfig(format!(
            "{layer_name} activation cache shape mismatch: expected {}, got {}",
            output.shape, cached_input.shape
        )));
    }

    let mut delta = Tensor::zeros(output.shape);
    for (index, (dst, (out_value, grad_value))) in delta
        .data
        .iter_mut()
        .zip(output.data.iter().zip(&output_gradient.data))
        .enumerate()
    {
        let cached_input = activation_input.map(|tensor| tensor.data[index]);
        *dst = activation_gradient_from_output(activation, *out_value, cached_input, layer_name)?
            * *grad_value;
    }
    Ok(delta)
}

fn activation_gradient_from_output(
    activation: &Activation,
    output_value: f32,
    activation_input: Option<f32>,
    layer_name: &str,
) -> Result<f32> {
    match activation {
        Activation::Linear => Ok(1.0),
        Activation::Logistic => Ok(output_value * (1.0 - output_value)),
        Activation::Relu => Ok((output_value > 0.0) as i32 as f32),
        Activation::Leaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
        Activation::Tanh => Ok(1.0 - output_value * output_value),
        Activation::Swish => {
            let x = activation_input.ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "{layer_name} backward requires cached activation_input for {:?}",
                    activation
                ))
            })?;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            Ok(sigmoid + x * sigmoid * (1.0 - sigmoid))
        }
        Activation::Mish => {
            let x = activation_input.ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "{layer_name} backward requires cached activation_input for {:?}",
                    activation
                ))
            })?;
            const MISH_THRESHOLD: f32 = 20.0;
            let softplus = if x > MISH_THRESHOLD {
                x
            } else if x < -MISH_THRESHOLD {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tsp = softplus.tanh();
            let grad_tsp = (1.0 - tsp * tsp) * grad_sp;
            Ok(x * grad_tsp + tsp)
        }
        Activation::Other(name) => Err(Error::UnsupportedLayer {
            layer_index: 0,
            name: format!("{layer_name}_{name}_backward"),
        }),
    }
}

fn accumulate_previous_gradient(
    layer_index: usize,
    gradient: &Tensor,
    input_gradient: &mut Tensor,
    layer_output_gradients: &mut [Tensor],
) -> Result<()> {
    if layer_index == 0 {
        accumulate_tensor(input_gradient, gradient)
    } else {
        accumulate_tensor(
            layer_output_gradients
                .get_mut(layer_index - 1)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous layer gradient slot for layer #{layer_index}"
                    ))
                })?,
            gradient,
        )
    }
}

fn accumulate_tensor(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.shape != src.shape {
        return Err(Error::InvalidConfig(format!(
            "gradient accumulation shape mismatch: {} vs {}",
            dst.shape, src.shape
        )));
    }
    for (dst_value, src_value) in dst.data.iter_mut().zip(&src.data) {
        *dst_value += *src_value;
    }
    Ok(())
}

fn normalize_value(
    value: f32,
    index: usize,
    biases: &[f32],
    batch_norm: Option<&BatchNormParams>,
) -> f32 {
    if let Some(batch_norm) = batch_norm {
        let normalized = (value - batch_norm.rolling_mean[index])
            / (batch_norm.rolling_variance[index] + BATCH_NORM_EPSILON).sqrt();
        normalized * batch_norm.scales[index] + biases[index]
    } else {
        value + biases[index]
    }
}

fn activate(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::Linear => value,
        Activation::Logistic => 1.0 / (1.0 + (-value).exp()),
        Activation::Relu => value.max(0.0),
        Activation::Leaky => {
            if value > 0.0 {
                value
            } else {
                0.1 * value
            }
        }
        Activation::Mish => value * (1.0 + value.exp()).ln().tanh(),
        Activation::Swish => value / (1.0 + (-value).exp()),
        Activation::Tanh => value.tanh(),
        Activation::Other(_) => value,
    }
}

fn forward_maxpool(
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    input: &Tensor,
    output_shape: TensorShape,
    average: bool,
) -> Result<Tensor> {
    if output_shape.channels != input.shape.channels {
        return Err(Error::UnsupportedLayer {
            layer_index: 0,
            name: "maxpool_depth".to_owned(),
        });
    }

    let mut output = Tensor::zeros(output_shape);
    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut acc = if average { 0.0 } else { f32::NEG_INFINITY };
                    let mut count = 0usize;

                    for ky in 0..size {
                        for kx in 0..size {
                            let in_y = oy * stride_y + ky;
                            let in_x = ox * stride_x + kx;
                            let src_y = in_y as isize - (padding / 2) as isize;
                            let src_x = in_x as isize - (padding / 2) as isize;

                            if src_y < 0
                                || src_y >= input.shape.height as isize
                                || src_x < 0
                                || src_x >= input.shape.width as isize
                            {
                                continue;
                            }

                            let value = input.get(b, c, src_y as usize, src_x as usize);
                            if average {
                                acc += value;
                                count += 1;
                            } else {
                                acc = acc.max(value);
                            }
                        }
                    }

                    if average && count > 0 {
                        acc /= count as f32;
                    }

                    output.set(b, c, oy, ox, acc);
                }
            }
        }
    }
    Ok(output)
}

fn forward_avgpool(input: &Tensor) -> Tensor {
    let mut output = Tensor::zeros(TensorShape::new(
        1,
        1,
        input.shape.channels,
        input.shape.batch,
    ));
    let area = input.channel_area() as f32;

    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            let mut acc = 0.0;
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    acc += input.get(b, c, y, x);
                }
            }
            output.data[b * input.shape.channels + c] = acc / area;
        }
    }

    output
}

fn backward_avgpool(output_grad: &Tensor, input_shape: TensorShape) -> Result<Tensor> {
    let expected = TensorShape::new(1, 1, input_shape.channels, input_shape.batch);
    if output_grad.shape != expected {
        return Err(Error::InvalidConfig(format!(
            "avgpool backward gradient shape mismatch: expected {expected}, got {}",
            output_grad.shape
        )));
    }

    let mut input_grad = Tensor::zeros(input_shape);
    let area = (input_shape.width * input_shape.height) as f32;
    for b in 0..input_shape.batch {
        for c in 0..input_shape.channels {
            let grad = output_grad.data[b * input_shape.channels + c] / area;
            for y in 0..input_shape.height {
                for x in 0..input_shape.width {
                    input_grad.set(b, c, y, x, grad);
                }
            }
        }
    }

    Ok(input_grad)
}

fn forward_route(
    spec: &RouteSpec,
    layer_index: usize,
    outputs: &[Tensor],
    output_shape: TensorShape,
) -> Result<Tensor> {
    let mut output = Tensor::zeros(output_shape);
    let mut out_channel = 0usize;

    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        let chunk_channels = source.shape.channels / spec.groups.max(1);
        let channel_start = chunk_channels * spec.group_id;

        for b in 0..source.shape.batch {
            for c in 0..chunk_channels {
                for y in 0..source.shape.height {
                    for x in 0..source.shape.width {
                        let value = source.get(b, channel_start + c, y, x);
                        output.set(b, out_channel + c, y, x, value);
                    }
                }
            }
        }

        out_channel += chunk_channels;
    }

    Ok(output)
}

fn forward_shortcut_with_cache(
    spec: &crate::model::ShortcutSpec,
    current_input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<(Tensor, Option<Tensor>)> {
    let mut output = current_input.clone();
    for raw_index in &spec.from {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        if source.shape != output.shape {
            return Err(Error::InvalidConfig(format!(
                "shortcut layer #{layer_index} mixes incompatible shapes {} and {}",
                output.shape, source.shape
            )));
        }
        for (dst, src) in output.data.iter_mut().zip(&source.data) {
            *dst += *src;
        }
    }
    let activation_input = output.clone();
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok((output, Some(activation_input)))
}

fn forward_upsample(
    spec: &crate::model::UpsampleSpec,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let stride = spec.stride.unsigned_abs();
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "upsample stride cannot be zero".to_owned(),
        ));
    }

    let mut output = Tensor::zeros(output_shape);
    if spec.reverse {
        for b in 0..output.shape.batch {
            for c in 0..output.shape.channels {
                for y in 0..input.shape.height {
                    for x in 0..input.shape.width {
                        let dst_y = y / stride;
                        let dst_x = x / stride;
                        let dst = output.offset(b, c, dst_y, dst_x);
                        output.data[dst] += spec.scale * input.get(b, c, y, x);
                    }
                }
            }
        }
    } else {
        for b in 0..input.shape.batch {
            for c in 0..input.shape.channels {
                for y in 0..output.shape.height {
                    for x in 0..output.shape.width {
                        let src_y = y / stride;
                        let src_x = x / stride;
                        let value = spec.scale * input.get(b, c, src_y, src_x);
                        output.set(b, c, y, x, value);
                    }
                }
            }
        }
    }

    Ok(output)
}

fn forward_reorg(
    stride: usize,
    reverse: bool,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "reorg stride cannot be zero".to_owned(),
        ));
    }

    let mut output = Tensor::zeros(output_shape);
    if reverse {
        for b in 0..input.shape.batch {
            for c in 0..input.shape.channels {
                let out_c = c / (stride * stride);
                let offset = c % (stride * stride);
                let dy = offset / stride;
                let dx = offset % stride;
                for y in 0..input.shape.height {
                    for x in 0..input.shape.width {
                        output.set(
                            b,
                            out_c,
                            y * stride + dy,
                            x * stride + dx,
                            input.get(b, c, y, x),
                        );
                    }
                }
            }
        }
    } else {
        for b in 0..output.shape.batch {
            for c in 0..output.shape.channels {
                let in_c = c % input.shape.channels;
                let offset = c / input.shape.channels;
                let dy = offset / stride;
                let dx = offset % stride;
                for y in 0..output.shape.height {
                    for x in 0..output.shape.width {
                        output.set(
                            b,
                            c,
                            y,
                            x,
                            input.get(b, in_c, y * stride + dy, x * stride + dx),
                        );
                    }
                }
            }
        }
    }
    Ok(output)
}

fn resolve_relative_layer(layer_index: usize, raw_index: isize) -> Result<usize> {
    let absolute = if raw_index < 0 {
        layer_index
            .checked_add_signed(raw_index)
            .ok_or(Error::InvalidRoute {
                layer_index,
                reference: raw_index,
            })?
    } else {
        usize::try_from(raw_index).map_err(|_| Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        })?
    };

    if absolute >= layer_index {
        return Err(Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        });
    }

    Ok(absolute)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use crate::cfg::ConfigFile;
    use crate::model::{
        Activation, ConvolutionalSpec, LayerKind, LrPolicy, NetConfig, Network, TensorShape,
    };
    use crate::train::{YoloLabel, compute_yolo_training_loss};
    use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

    use super::{
        ConvolutionalParams, ExecutableNetwork, ForwardResult, GpuExecutableNetwork, LayerParams,
        LayerTrainBuffers, Tensor, compute_learning_rate, sgd_update,
    };

    fn repo_cfg(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../cfg")
            .join(name)
    }

    #[derive(Clone, Debug, PartialEq)]
    struct FakeTensor {
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    #[derive(Clone, Debug, Default)]
    struct FakeBackend;

    impl F32TensorOpsBackend for FakeBackend {
        type Error = String;
        type Tensor = FakeTensor;

        fn upload_f32(
            &self,
            shape: &[usize],
            host: &[f32],
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let expected = shape.iter().product::<usize>();
            if host.len() != expected {
                return Err(format!(
                    "upload_f32 expected {expected} elements for shape {:?}, got {}",
                    shape,
                    host.len()
                ));
            }
            Ok(FakeTensor {
                shape: shape.to_vec(),
                data: host.to_vec(),
            })
        }

        fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(FakeTensor {
                shape: shape.to_vec(),
                data: vec![0.0; shape.iter().product()],
            })
        }

        fn download_f32(
            &self,
            tensor: &Self::Tensor,
        ) -> std::result::Result<Vec<f32>, Self::Error> {
            Ok(tensor.data.clone())
        }

        fn bias_add_f32(
            &self,
            x: &Self::Tensor,
            bias: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if bias.data.len() != channels {
                return Err("bias_add_f32 bias length mismatch".to_owned());
            }
            let batch = x.data.len() / (channels * channel_stride);
            let mut out = x.data.clone();
            for b in 0..batch {
                for c in 0..channels {
                    let base = (b * channels + c) * channel_stride;
                    for idx in 0..channel_stride {
                        out[base + idx] += bias.data[c];
                    }
                }
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: out,
            })
        }

        fn channel_scale_f32(
            &self,
            x: &Self::Tensor,
            scale: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if scale.data.len() != channels {
                return Err("channel_scale_f32 scale length mismatch".to_owned());
            }
            let batch = x.data.len() / (channels * channel_stride);
            let mut out = x.data.clone();
            for b in 0..batch {
                for c in 0..channels {
                    let base = (b * channels + c) * channel_stride;
                    for idx in 0..channel_stride {
                        out[base + idx] *= scale.data[c];
                    }
                }
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: out,
            })
        }

        fn bias_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = output_grad.data.len() / (channels * channel_stride);
            let mut bias_grad = vec![0.0; channels];
            for b in 0..batch {
                for (c, slot) in bias_grad.iter_mut().enumerate().take(channels) {
                    let base = (b * channels + c) * channel_stride;
                    *slot += output_grad.data[base..base + channel_stride]
                        .iter()
                        .sum::<f32>();
                }
            }
            Ok(FakeTensor {
                shape: vec![channels],
                data: bias_grad,
            })
        }

        fn activation_f32(
            &self,
            x: &Self::Tensor,
            activation: F32ActivationKind,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: x
                    .data
                    .iter()
                    .copied()
                    .map(|value| apply_fake_activation(activation, value))
                    .collect(),
            })
        }

        fn activation_backward_f32(
            &self,
            activation_input: &Self::Tensor,
            delta: &Self::Tensor,
            activation: F32ActivationKind,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if activation_input.shape != delta.shape {
                return Err("activation_backward_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: activation_input.shape.clone(),
                data: activation_input
                    .data
                    .iter()
                    .zip(&delta.data)
                    .map(|(value, grad)| grad * fake_activation_gradient(activation, *value))
                    .collect(),
            })
        }

        fn shortcut_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("shortcut_add_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: x
                    .data
                    .iter()
                    .zip(&y.data)
                    .map(|(lhs, rhs)| lhs + rhs)
                    .collect(),
            })
        }

        fn route_concat_f32(
            &self,
            inputs: &[&Self::Tensor],
            input_channel_offsets: &[usize],
            input_channels: &[usize],
            input_total_channels: &[usize],
            output_shape: &[usize],
            channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if inputs.len() != input_channel_offsets.len()
                || inputs.len() != input_channels.len()
                || inputs.len() != input_total_channels.len()
            {
                return Err("route_concat_f32 metadata length mismatch".to_owned());
            }
            let batch = inputs
                .first()
                .and_then(|tensor| tensor.shape.first().copied())
                .ok_or("route_concat_f32 requires at least one input")?;
            let total_channels = input_channels.iter().sum::<usize>();
            let mut out = vec![0.0; output_shape.iter().product()];
            let mut channel_offset = 0usize;

            for (((input, source_channel_offset), src_channels), source_total_channels) in inputs
                .iter()
                .zip(input_channel_offsets.iter().copied())
                .zip(input_channels.iter().copied())
                .zip(input_total_channels.iter().copied())
            {
                for b in 0..batch {
                    for c in 0..src_channels {
                        let src_base = (b * source_total_channels + source_channel_offset + c)
                            * channel_stride;
                        let dst_base = (b * total_channels + channel_offset + c) * channel_stride;
                        out[dst_base..dst_base + channel_stride]
                            .copy_from_slice(&input.data[src_base..src_base + channel_stride]);
                    }
                }
                channel_offset += src_channels;
            }

            Ok(FakeTensor {
                shape: output_shape.to_vec(),
                data: out,
            })
        }

        fn route_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shapes: &[Vec<usize>],
            input_channel_offsets: &[usize],
            input_channels: &[usize],
            input_total_channels: &[usize],
            channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
            if input_shapes.len() != input_channel_offsets.len()
                || input_shapes.len() != input_channels.len()
                || input_shapes.len() != input_total_channels.len()
            {
                return Err("route_backward_f32 metadata length mismatch".to_owned());
            }
            let batch = output_grad.shape[0];
            let mut output_channel_offset = 0usize;
            let mut out = Vec::with_capacity(input_shapes.len());

            for (((input_shape, source_channel_offset), src_channels), source_total_channels) in
                input_shapes
                    .iter()
                    .zip(input_channel_offsets.iter().copied())
                    .zip(input_channels.iter().copied())
                    .zip(input_total_channels.iter().copied())
            {
                let mut input_grad = vec![0.0; input_shape.iter().product()];
                for b in 0..batch {
                    for c in 0..src_channels {
                        let src_base =
                            (b * output_grad.shape[1] + output_channel_offset + c) * channel_stride;
                        let dst_base = (b * source_total_channels + source_channel_offset + c)
                            * channel_stride;
                        input_grad[dst_base..dst_base + channel_stride].copy_from_slice(
                            &output_grad.data[src_base..src_base + channel_stride],
                        );
                    }
                }
                out.push(FakeTensor {
                    shape: input_shape.clone(),
                    data: input_grad,
                });
                output_channel_offset += src_channels;
            }

            Ok(out)
        }

        fn conv2d_f32(
            &self,
            x: &Self::Tensor,
            weights: &Self::Tensor,
            output_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let input = Tensor::from_data(
                TensorShape::new(input_width, input_height, input_channels, x.shape[0]),
                x.data.clone(),
            )
            .map_err(|err| err.to_string())?;
            let output_shape = TensorShape::new(
                output_shape[3],
                output_shape[2],
                output_shape[1],
                output_shape[0],
            );
            let spec = ConvolutionalSpec {
                filters,
                size: kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                share_index: None,
                batch_normalize: false,
                activation: Activation::Linear,
            };
            let params = ConvolutionalParams {
                biases: vec![0.0; filters],
                batch_norm: None,
                weights: weights.data.clone(),
            };
            let output = super::forward_convolutional(&spec, &params, &input, output_shape);
            Ok(FakeTensor {
                shape: vec![
                    output.shape.batch,
                    output.shape.channels,
                    output.shape.height,
                    output.shape.width,
                ],
                data: output.data,
            })
        }

        fn conv2d_backward_input_f32(
            &self,
            output_grad: &Self::Tensor,
            weights: &Self::Tensor,
            input_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = input_shape[0];
            let channels_per_group = input_channels / groups;
            let filters_per_group = filters / groups;
            let output_height = output_grad.shape[2];
            let output_width = output_grad.shape[3];
            let mut input_grad = vec![0.0; input_shape.iter().product()];

            for b in 0..batch {
                for channel in 0..input_channels {
                    let group = channel / channels_per_group;
                    let input_channel_in_group = channel % channels_per_group;
                    for iy in 0..input_height {
                        for ix in 0..input_width {
                            let mut acc = 0.0;
                            for filter_in_group in 0..filters_per_group {
                                let filter = group * filters_per_group + filter_in_group;
                                for ky in 0..kernel_size {
                                    let oy_num =
                                        iy as isize + padding as isize - (ky * dilation) as isize;
                                    if oy_num < 0 || oy_num % stride_y as isize != 0 {
                                        continue;
                                    }
                                    let oy = (oy_num / stride_y as isize) as usize;
                                    if oy >= output_height {
                                        continue;
                                    }
                                    for kx in 0..kernel_size {
                                        let ox_num = ix as isize + padding as isize
                                            - (kx * dilation) as isize;
                                        if ox_num < 0 || ox_num % stride_x as isize != 0 {
                                            continue;
                                        }
                                        let ox = (ox_num / stride_x as isize) as usize;
                                        if ox >= output_width {
                                            continue;
                                        }

                                        let grad_index =
                                            ((((b * filters) + filter) * output_height) + oy)
                                                * output_width
                                                + ox;
                                        let weight_index = ((filter * channels_per_group
                                            + input_channel_in_group)
                                            * kernel_size
                                            + ky)
                                            * kernel_size
                                            + kx;
                                        acc += output_grad.data[grad_index]
                                            * weights.data[weight_index];
                                    }
                                }
                            }
                            let input_index = ((((b * input_channels) + channel) * input_height)
                                + iy)
                                * input_width
                                + ix;
                            input_grad[input_index] = acc;
                        }
                    }
                }
            }

            Ok(FakeTensor {
                shape: input_shape.to_vec(),
                data: input_grad,
            })
        }

        fn conv2d_backward_weights_f32(
            &self,
            x: &Self::Tensor,
            output_grad: &Self::Tensor,
            weight_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = x.shape[0];
            let channels_per_group = input_channels / groups;
            let filters_per_group = filters / groups;
            let output_height = output_grad.shape[2];
            let output_width = output_grad.shape[3];
            let mut weight_grad = vec![0.0; weight_shape.iter().product()];

            for filter in 0..filters {
                let group = filter / filters_per_group;
                for input_channel_in_group in 0..channels_per_group {
                    let input_channel = group * channels_per_group + input_channel_in_group;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let mut acc = 0.0;
                            for b in 0..batch {
                                for oy in 0..output_height {
                                    let in_y = oy * stride_y + ky * dilation;
                                    let src_y = in_y as isize - padding as isize;
                                    if src_y < 0 || src_y >= input_height as isize {
                                        continue;
                                    }
                                    for ox in 0..output_width {
                                        let in_x = ox * stride_x + kx * dilation;
                                        let src_x = in_x as isize - padding as isize;
                                        if src_x < 0 || src_x >= input_width as isize {
                                            continue;
                                        }
                                        let input_index = ((((b * input_channels)
                                            + input_channel)
                                            * input_height)
                                            + src_y as usize)
                                            * input_width
                                            + src_x as usize;
                                        let grad_index =
                                            ((((b * filters) + filter) * output_height) + oy)
                                                * output_width
                                                + ox;
                                        acc += x.data[input_index] * output_grad.data[grad_index];
                                    }
                                }
                            }
                            let weight_index = ((filter * channels_per_group
                                + input_channel_in_group)
                                * kernel_size
                                + ky)
                                * kernel_size
                                + kx;
                            weight_grad[weight_index] = acc;
                        }
                    }
                }
            }

            Ok(FakeTensor {
                shape: weight_shape.to_vec(),
                data: weight_grad,
            })
        }

        fn upsample_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            stride: usize,
            scale: f32,
            reverse: bool,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = x.shape[0];
            let output_height = output_shape[2];
            let output_width = output_shape[3];
            let mut out = vec![0.0; output_shape.iter().product()];

            if reverse {
                for b in 0..batch {
                    for c in 0..channels {
                        for y in 0..input_height {
                            for x_idx in 0..input_width {
                                let src = ((((b * channels) + c) * input_height) + y) * input_width
                                    + x_idx;
                                let dst_y = y / stride;
                                let dst_x = x_idx / stride;
                                let dst = ((((b * channels) + c) * output_height) + dst_y)
                                    * output_width
                                    + dst_x;
                                out[dst] += scale * x.data[src];
                            }
                        }
                    }
                }
            } else {
                for b in 0..batch {
                    for c in 0..channels {
                        for y in 0..output_height {
                            for x_idx in 0..output_width {
                                let src_y = y / stride;
                                let src_x = x_idx / stride;
                                let src = ((((b * channels) + c) * input_height) + src_y)
                                    * input_width
                                    + src_x;
                                let dst = ((((b * channels) + c) * output_height) + y)
                                    * output_width
                                    + x_idx;
                                out[dst] = scale * x.data[src];
                            }
                        }
                    }
                }
            }

            Ok(FakeTensor {
                shape: output_shape.to_vec(),
                data: out,
            })
        }

        fn upsample_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            output_height: usize,
            output_width: usize,
            stride: usize,
            scale: f32,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.upsample_f32(
                output_grad,
                input_shape,
                channels,
                output_height,
                output_width,
                stride,
                scale,
                true,
                config,
            )
        }

        fn maxpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let input = Tensor::from_data(
                TensorShape::new(input_width, input_height, channels, x.shape[0]),
                x.data.clone(),
            )
            .map_err(|err| err.to_string())?;
            let output_shape = TensorShape::new(
                output_shape[3],
                output_shape[2],
                output_shape[1],
                output_shape[0],
            );
            let output = super::forward_maxpool(
                size,
                stride_x,
                stride_y,
                padding,
                &input,
                output_shape,
                false,
            )
            .map_err(|err| err.to_string())?;
            Ok(FakeTensor {
                shape: vec![
                    output.shape.batch,
                    output.shape.channels,
                    output.shape.height,
                    output.shape.width,
                ],
                data: output.data,
            })
        }

        fn maxpool_backward_f32(
            &self,
            x: &Self::Tensor,
            output_grad: &Self::Tensor,
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = x.shape[0];
            let output_height = output_grad.shape[2];
            let output_width = output_grad.shape[3];
            let pad_offset = (padding / 2) as isize;
            let mut input_grad = vec![0.0; x.data.len()];

            for b in 0..batch {
                for c in 0..channels {
                    for oy in 0..output_height {
                        for ox in 0..output_width {
                            let mut best_index = None;
                            let mut best_value = f32::NEG_INFINITY;
                            for ky in 0..size {
                                let in_y =
                                    oy as isize * stride_y as isize + ky as isize - pad_offset;
                                if !(0..input_height as isize).contains(&in_y) {
                                    continue;
                                }
                                for kx in 0..size {
                                    let in_x =
                                        ox as isize * stride_x as isize + kx as isize - pad_offset;
                                    if !(0..input_width as isize).contains(&in_x) {
                                        continue;
                                    }
                                    let input_index = ((((b * channels) + c) * input_height)
                                        + in_y as usize)
                                        * input_width
                                        + in_x as usize;
                                    let value = x.data[input_index];
                                    if value > best_value {
                                        best_value = value;
                                        best_index = Some(input_index);
                                    }
                                }
                            }

                            if let Some(input_index) = best_index {
                                let output_index = ((((b * channels) + c) * output_height) + oy)
                                    * output_width
                                    + ox;
                                input_grad[input_index] += output_grad.data[output_index];
                            }
                        }
                    }
                }
            }

            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: input_grad,
            })
        }

        fn local_avgpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let input = Tensor::from_data(
                TensorShape::new(input_width, input_height, channels, x.shape[0]),
                x.data.clone(),
            )
            .map_err(|err| err.to_string())?;
            let output_shape = TensorShape::new(
                output_shape[3],
                output_shape[2],
                output_shape[1],
                output_shape[0],
            );
            let output = super::forward_maxpool(
                size,
                stride_x,
                stride_y,
                padding,
                &input,
                output_shape,
                true,
            )
            .map_err(|err| err.to_string())?;
            Ok(FakeTensor {
                shape: vec![
                    output.shape.batch,
                    output.shape.channels,
                    output.shape.height,
                    output.shape.width,
                ],
                data: output.data,
            })
        }

        fn local_avgpool_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let input = Tensor::zeros(TensorShape::new(
                input_width,
                input_height,
                channels,
                input_shape[0],
            ));
            let output_grad = Tensor::from_data(
                TensorShape::new(
                    output_grad.shape[3],
                    output_grad.shape[2],
                    output_grad.shape[1],
                    output_grad.shape[0],
                ),
                output_grad.data.clone(),
            )
            .map_err(|err| err.to_string())?;
            let input_grad = super::backward_maxpool(
                size,
                stride_x,
                stride_y,
                padding,
                &input,
                &output_grad,
                true,
            )
            .map_err(|err| err.to_string())?;
            Ok(FakeTensor {
                shape: input_shape.to_vec(),
                data: input_grad.data,
            })
        }

        fn avgpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if output_shape.len() != 4 || output_shape[1] != channels {
                return Err("avgpool_f32 output shape mismatch".to_owned());
            }
            let batch = x.shape[0];
            let area = (input_height * input_width) as f32;
            let mut out = vec![0.0; output_shape.iter().product()];
            for b in 0..batch {
                for c in 0..channels {
                    let mut acc = 0.0;
                    for y in 0..input_height {
                        for x_idx in 0..input_width {
                            let src =
                                ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                            acc += x.data[src];
                        }
                    }
                    out[b * channels + c] = acc / area;
                }
            }
            Ok(FakeTensor {
                shape: output_shape.to_vec(),
                data: out,
            })
        }

        fn avgpool_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let batch = input_shape[0];
            let area = (input_height * input_width) as f32;
            let mut input_grad = vec![0.0; input_shape.iter().product()];
            for b in 0..batch {
                for c in 0..channels {
                    let grad = output_grad.data[b * channels + c] / area;
                    for y in 0..input_height {
                        for x_idx in 0..input_width {
                            let dst =
                                ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                            input_grad[dst] = grad;
                        }
                    }
                }
            }
            Ok(FakeTensor {
                shape: input_shape.to_vec(),
                data: input_grad,
            })
        }

        fn vector_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("vector_add_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: x
                    .data
                    .iter()
                    .zip(&y.data)
                    .map(|(lhs, rhs)| lhs + rhs)
                    .collect(),
            })
        }

        fn saxpy_f32(
            &self,
            alpha: f32,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("saxpy_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: x
                    .data
                    .iter()
                    .zip(&y.data)
                    .map(|(lhs, rhs)| alpha * lhs + rhs)
                    .collect(),
            })
        }

        fn sgd_update_f32(
            &self,
            weights: &Self::Tensor,
            updates: &Self::Tensor,
            lr: f32,
            momentum: f32,
            decay: f32,
            batch: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
            if weights.shape != updates.shape {
                return Err("sgd_update_f32 shape mismatch".to_owned());
            }
            let mut new_weights = weights.data.clone();
            let mut new_updates = updates.data.clone();
            super::sgd_update(
                &mut new_weights,
                &mut new_updates,
                lr,
                momentum,
                decay,
                batch,
            );
            Ok((
                FakeTensor {
                    shape: weights.shape.clone(),
                    data: new_weights,
                },
                FakeTensor {
                    shape: updates.shape.clone(),
                    data: new_updates,
                },
            ))
        }
    }

    fn apply_fake_activation(activation: F32ActivationKind, value: f32) -> f32 {
        let activation = match activation {
            F32ActivationKind::Linear => Activation::Linear,
            F32ActivationKind::Logistic => Activation::Logistic,
            F32ActivationKind::Relu => Activation::Relu,
            F32ActivationKind::Leaky => Activation::Leaky,
            F32ActivationKind::Mish => Activation::Mish,
            F32ActivationKind::Swish => Activation::Swish,
            F32ActivationKind::Tanh => Activation::Tanh,
        };
        super::activate(activation, value)
    }

    fn fake_activation_gradient(activation: F32ActivationKind, value: f32) -> f32 {
        match activation {
            F32ActivationKind::Linear => 1.0,
            F32ActivationKind::Logistic => {
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                sigmoid * (1.0 - sigmoid)
            }
            F32ActivationKind::Relu => (value > 0.0) as i32 as f32,
            F32ActivationKind::Leaky => {
                if value > 0.0 {
                    1.0
                } else {
                    0.1
                }
            }
            F32ActivationKind::Mish => {
                const MISH_THRESHOLD: f32 = 20.0;
                let softplus = if value > MISH_THRESHOLD {
                    value
                } else if value < -MISH_THRESHOLD {
                    value.exp()
                } else {
                    (1.0 + value.exp()).ln()
                };
                let grad_sp = 1.0 - (-softplus).exp();
                let tsp = softplus.tanh();
                let grad_tsp = (1.0 - tsp * tsp) * grad_sp;
                value * grad_tsp + tsp
            }
            F32ActivationKind::Swish => {
                let sigmoid = 1.0 / (1.0 + (-value).exp());
                sigmoid + value * sigmoid * (1.0 - sigmoid)
            }
            F32ActivationKind::Tanh => {
                let tanh = value.tanh();
                1.0 - tanh * tanh
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    struct CountingBackend {
        uploads: Arc<AtomicUsize>,
        inner: FakeBackend,
    }

    impl CountingBackend {
        fn upload_count(&self) -> usize {
            self.uploads.load(Ordering::Relaxed)
        }
    }

    impl F32TensorOpsBackend for CountingBackend {
        type Error = String;
        type Tensor = FakeTensor;

        fn upload_f32(
            &self,
            shape: &[usize],
            host: &[f32],
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.uploads.fetch_add(1, Ordering::Relaxed);
            self.inner.upload_f32(shape, host)
        }

        fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.zeros_f32(shape)
        }

        fn download_f32(
            &self,
            tensor: &Self::Tensor,
        ) -> std::result::Result<Vec<f32>, Self::Error> {
            self.inner.download_f32(tensor)
        }

        fn bias_add_f32(
            &self,
            x: &Self::Tensor,
            bias: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner
                .bias_add_f32(x, bias, channels, channel_stride, config)
        }

        fn channel_scale_f32(
            &self,
            x: &Self::Tensor,
            scale: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner
                .channel_scale_f32(x, scale, channels, channel_stride, config)
        }

        fn bias_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            channels: usize,
            channel_stride: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner
                .bias_backward_f32(output_grad, channels, channel_stride, config)
        }

        fn activation_f32(
            &self,
            x: &Self::Tensor,
            activation: F32ActivationKind,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.activation_f32(x, activation, config)
        }

        fn activation_backward_f32(
            &self,
            activation_input: &Self::Tensor,
            delta: &Self::Tensor,
            activation: F32ActivationKind,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner
                .activation_backward_f32(activation_input, delta, activation, config)
        }

        fn shortcut_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.shortcut_add_f32(x, y, config)
        }

        fn route_concat_f32(
            &self,
            inputs: &[&Self::Tensor],
            input_channel_offsets: &[usize],
            input_channels: &[usize],
            input_total_channels: &[usize],
            output_shape: &[usize],
            channel_stride: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.route_concat_f32(
                inputs,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                output_shape,
                channel_stride,
                config,
            )
        }

        fn route_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shapes: &[Vec<usize>],
            input_channel_offsets: &[usize],
            input_channels: &[usize],
            input_total_channels: &[usize],
            channel_stride: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
            self.inner.route_backward_f32(
                output_grad,
                input_shapes,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                channel_stride,
                config,
            )
        }

        fn conv2d_f32(
            &self,
            x: &Self::Tensor,
            weights: &Self::Tensor,
            output_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.conv2d_f32(
                x,
                weights,
                output_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            )
        }

        fn conv2d_backward_input_f32(
            &self,
            output_grad: &Self::Tensor,
            weights: &Self::Tensor,
            input_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.conv2d_backward_input_f32(
                output_grad,
                weights,
                input_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            )
        }

        fn conv2d_backward_weights_f32(
            &self,
            x: &Self::Tensor,
            output_grad: &Self::Tensor,
            weight_shape: &[usize],
            input_channels: usize,
            input_height: usize,
            input_width: usize,
            filters: usize,
            kernel_size: usize,
            stride_x: usize,
            stride_y: usize,
            dilation: usize,
            padding: usize,
            groups: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.conv2d_backward_weights_f32(
                x,
                output_grad,
                weight_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            )
        }

        fn maxpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.maxpool_f32(
                x,
                output_shape,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )
        }

        fn maxpool_backward_f32(
            &self,
            x: &Self::Tensor,
            output_grad: &Self::Tensor,
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.maxpool_backward_f32(
                x,
                output_grad,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )
        }

        fn local_avgpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.local_avgpool_f32(
                x,
                output_shape,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )
        }

        fn local_avgpool_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            size: usize,
            stride_x: usize,
            stride_y: usize,
            padding: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.local_avgpool_backward_f32(
                output_grad,
                input_shape,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )
        }

        fn avgpool_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner
                .avgpool_f32(x, output_shape, channels, input_height, input_width, config)
        }

        fn avgpool_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.avgpool_backward_f32(
                output_grad,
                input_shape,
                channels,
                input_height,
                input_width,
                config,
            )
        }

        fn upsample_f32(
            &self,
            x: &Self::Tensor,
            output_shape: &[usize],
            channels: usize,
            input_height: usize,
            input_width: usize,
            stride: usize,
            scale: f32,
            reverse: bool,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.upsample_f32(
                x,
                output_shape,
                channels,
                input_height,
                input_width,
                stride,
                scale,
                reverse,
                config,
            )
        }

        fn upsample_backward_f32(
            &self,
            output_grad: &Self::Tensor,
            input_shape: &[usize],
            channels: usize,
            output_height: usize,
            output_width: usize,
            stride: usize,
            scale: f32,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.upsample_backward_f32(
                output_grad,
                input_shape,
                channels,
                output_height,
                output_width,
                stride,
                scale,
                config,
            )
        }

        fn vector_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.vector_add_f32(x, y, config)
        }

        fn saxpy_f32(
            &self,
            alpha: f32,
            x: &Self::Tensor,
            y: &Self::Tensor,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.saxpy_f32(alpha, x, y, config)
        }

        fn sgd_update_f32(
            &self,
            weights: &Self::Tensor,
            updates: &Self::Tensor,
            lr: f32,
            momentum: f32,
            decay: f32,
            batch: usize,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
            self.inner
                .sgd_update_f32(weights, updates, lr, momentum, decay, batch, config)
        }
    }

    fn assert_same_forward(cpu: &ForwardResult, gpu: &ForwardResult) {
        assert_eq!(cpu.layer_outputs.len(), gpu.layer_outputs.len());
        assert_eq!(cpu.yolo_heads.len(), gpu.yolo_heads.len());
        for (lhs, rhs) in cpu.layer_outputs.iter().zip(&gpu.layer_outputs) {
            assert_eq!(lhs.shape, rhs.shape);
            assert_eq!(lhs.data, rhs.data);
        }
    }

    #[derive(Clone, Debug, Default)]
    struct ShapeOnlyBackend;

    impl F32TensorOpsBackend for ShapeOnlyBackend {
        type Error = String;
        type Tensor = FakeTensor;

        fn upload_f32(
            &self,
            shape: &[usize],
            host: &[f32],
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            let expected = shape.iter().product::<usize>();
            if host.len() != expected {
                return Err(format!(
                    "upload_f32 expected {expected} elements for shape {:?}, got {}",
                    shape,
                    host.len()
                ));
            }
            Ok(FakeTensor {
                shape: shape.to_vec(),
                data: host.to_vec(),
            })
        }

        fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(FakeTensor {
                shape: shape.to_vec(),
                data: vec![0.0; shape.iter().product()],
            })
        }

        fn download_f32(
            &self,
            tensor: &Self::Tensor,
        ) -> std::result::Result<Vec<f32>, Self::Error> {
            Ok(tensor.data.clone())
        }

        fn bias_add_f32(
            &self,
            x: &Self::Tensor,
            _bias: &Self::Tensor,
            _channels: usize,
            _channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
        }

        fn channel_scale_f32(
            &self,
            x: &Self::Tensor,
            scale: &Self::Tensor,
            _channels: usize,
            _channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if scale.shape.len() != 1 {
                return Err("channel_scale_f32 expects flat scale tensor".to_owned());
            }
            Ok(x.clone())
        }

        fn bias_backward_f32(
            &self,
            _output_grad: &Self::Tensor,
            channels: usize,
            _channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(&[channels])
        }

        fn activation_f32(
            &self,
            x: &Self::Tensor,
            _activation: F32ActivationKind,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
        }

        fn activation_backward_f32(
            &self,
            activation_input: &Self::Tensor,
            delta: &Self::Tensor,
            _activation: F32ActivationKind,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if activation_input.shape != delta.shape {
                return Err("activation_backward_f32 shape mismatch".to_owned());
            }
            Ok(delta.clone())
        }

        fn shortcut_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("shortcut_add_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: vec![0.0; x.data.len()],
            })
        }

        fn route_concat_f32(
            &self,
            _inputs: &[&Self::Tensor],
            _input_channel_offsets: &[usize],
            _input_channels: &[usize],
            _input_total_channels: &[usize],
            output_shape: &[usize],
            _channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn route_backward_f32(
            &self,
            _output_grad: &Self::Tensor,
            input_shapes: &[Vec<usize>],
            _input_channel_offsets: &[usize],
            _input_channels: &[usize],
            _input_total_channels: &[usize],
            _channel_stride: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
            input_shapes
                .iter()
                .map(|shape| self.zeros_f32(shape))
                .collect()
        }

        fn conv2d_f32(
            &self,
            _x: &Self::Tensor,
            _weights: &Self::Tensor,
            output_shape: &[usize],
            _input_channels: usize,
            _input_height: usize,
            _input_width: usize,
            _filters: usize,
            _kernel_size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _dilation: usize,
            _padding: usize,
            _groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn conv2d_backward_input_f32(
            &self,
            _output_grad: &Self::Tensor,
            _weights: &Self::Tensor,
            input_shape: &[usize],
            _input_channels: usize,
            _input_height: usize,
            _input_width: usize,
            _filters: usize,
            _kernel_size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _dilation: usize,
            _padding: usize,
            _groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(input_shape)
        }

        fn conv2d_backward_weights_f32(
            &self,
            _x: &Self::Tensor,
            _output_grad: &Self::Tensor,
            weight_shape: &[usize],
            _input_channels: usize,
            _input_height: usize,
            _input_width: usize,
            _filters: usize,
            _kernel_size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _dilation: usize,
            _padding: usize,
            _groups: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(weight_shape)
        }

        fn maxpool_f32(
            &self,
            _x: &Self::Tensor,
            output_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn maxpool_backward_f32(
            &self,
            x: &Self::Tensor,
            _output_grad: &Self::Tensor,
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(&x.shape)
        }

        fn local_avgpool_f32(
            &self,
            _x: &Self::Tensor,
            output_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn local_avgpool_backward_f32(
            &self,
            _output_grad: &Self::Tensor,
            input_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _size: usize,
            _stride_x: usize,
            _stride_y: usize,
            _padding: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(input_shape)
        }

        fn avgpool_f32(
            &self,
            _x: &Self::Tensor,
            output_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn avgpool_backward_f32(
            &self,
            _output_grad: &Self::Tensor,
            input_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(input_shape)
        }

        fn upsample_f32(
            &self,
            _x: &Self::Tensor,
            output_shape: &[usize],
            _channels: usize,
            _input_height: usize,
            _input_width: usize,
            _stride: usize,
            _scale: f32,
            _reverse: bool,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(output_shape)
        }

        fn upsample_backward_f32(
            &self,
            _output_grad: &Self::Tensor,
            input_shape: &[usize],
            _channels: usize,
            _output_height: usize,
            _output_width: usize,
            _stride: usize,
            _scale: f32,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.zeros_f32(input_shape)
        }

        fn vector_add_f32(
            &self,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("vector_add_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: vec![0.0; x.data.len()],
            })
        }

        fn saxpy_f32(
            &self,
            _alpha: f32,
            x: &Self::Tensor,
            y: &Self::Tensor,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            if x.shape != y.shape {
                return Err("saxpy_f32 shape mismatch".to_owned());
            }
            Ok(FakeTensor {
                shape: x.shape.clone(),
                data: vec![0.0; x.data.len()],
            })
        }

        fn sgd_update_f32(
            &self,
            weights: &Self::Tensor,
            updates: &Self::Tensor,
            _lr: f32,
            _momentum: f32,
            _decay: f32,
            _batch: usize,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
            if weights.shape != updates.shape {
                return Err("sgd_update_f32 shape mismatch".to_owned());
            }
            Ok((weights.clone(), updates.clone()))
        }
    }

    #[test]
    fn fake_backend_activation_backward_matches_reference() {
        let backend = FakeBackend;
        let activation_input = FakeTensor {
            shape: vec![5],
            data: vec![-2.0, -0.5, 0.0, 0.5, 2.0],
        };
        let delta = FakeTensor {
            shape: vec![5],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let config = ElementwiseLaunchConfig::default();

        for activation in [
            F32ActivationKind::Linear,
            F32ActivationKind::Logistic,
            F32ActivationKind::Relu,
            F32ActivationKind::Leaky,
            F32ActivationKind::Mish,
            F32ActivationKind::Swish,
            F32ActivationKind::Tanh,
        ] {
            let actual = backend
                .activation_backward_f32(&activation_input, &delta, activation, config)
                .expect("backward activation");
            let expected = activation_input
                .data
                .iter()
                .zip(&delta.data)
                .map(|(value, grad)| grad * fake_activation_gradient(activation, *value))
                .collect::<Vec<_>>();
            assert_eq!(actual.shape, activation_input.shape);
            assert_eq!(actual.data.len(), expected.len());
            for (actual_value, expected_value) in actual.data.iter().zip(expected) {
                assert!((actual_value - expected_value).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn fake_backend_upsample_backward_accumulates_source_gradients() {
        let backend = FakeBackend;
        let output_grad = FakeTensor {
            shape: vec![1, 1, 4, 4],
            data: (1..=16).map(|value| value as f32).collect(),
        };

        let input_grad = backend
            .upsample_backward_f32(
                &output_grad,
                &[1, 1, 2, 2],
                1,
                4,
                4,
                2,
                1.0,
                ElementwiseLaunchConfig::default(),
            )
            .expect("upsample backward");

        assert_eq!(input_grad.shape, vec![1, 1, 2, 2]);
        assert_eq!(input_grad.data, vec![14.0, 22.0, 46.0, 54.0]);
    }

    #[test]
    fn fake_backend_maxpool_backward_accumulates_overlapping_windows() {
        let backend = FakeBackend;
        let input = FakeTensor {
            shape: vec![1, 1, 3, 3],
            data: vec![1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
        };
        let output_grad = FakeTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        let input_grad = backend
            .maxpool_backward_f32(
                &input,
                &output_grad,
                1,
                3,
                3,
                2,
                1,
                1,
                0,
                ElementwiseLaunchConfig::default(),
            )
            .expect("maxpool backward");

        assert_eq!(input_grad.shape, input.shape);
        assert_eq!(
            input_grad.data,
            vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn fake_backend_route_backward_scatter_matches_forward_layout() {
        let backend = FakeBackend;
        let output_grad = FakeTensor {
            shape: vec![1, 3, 1, 2],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        let input_grads = backend
            .route_backward_f32(
                &output_grad,
                &[vec![1, 4, 1, 2], vec![1, 2, 1, 2]],
                &[1, 0],
                &[2, 1],
                &[4, 2],
                2,
                ElementwiseLaunchConfig::default(),
            )
            .expect("route backward");

        assert_eq!(input_grads.len(), 2);
        assert_eq!(
            input_grads[0],
            FakeTensor {
                shape: vec![1, 4, 1, 2],
                data: vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            }
        );
        assert_eq!(
            input_grads[1],
            FakeTensor {
                shape: vec![1, 2, 1, 2],
                data: vec![5.0, 6.0, 0.0, 0.0],
            }
        );
    }

    #[test]
    fn fake_backend_bias_backward_reduces_channel_gradients() {
        let backend = FakeBackend;
        let output_grad = FakeTensor {
            shape: vec![1, 2, 1, 3],
            data: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        };
        let bias_grad = backend
            .bias_backward_f32(&output_grad, 2, 3, ElementwiseLaunchConfig::default())
            .expect("bias backward");
        assert_eq!(bias_grad.shape, vec![2]);
        assert_eq!(bias_grad.data, vec![6.0, 60.0]);
    }

    #[test]
    fn fake_backend_conv_backward_matches_reference_for_pointwise_conv() {
        let backend = FakeBackend;
        let input = FakeTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let weights = FakeTensor {
            shape: vec![1, 1, 1, 1],
            data: vec![2.0],
        };
        let output_grad = FakeTensor {
            shape: vec![1, 1, 2, 2],
            data: vec![10.0, 20.0, 30.0, 40.0],
        };

        let bias_grad = backend
            .bias_backward_f32(&output_grad, 1, 4, ElementwiseLaunchConfig::default())
            .expect("bias backward");
        let input_grad = backend
            .conv2d_backward_input_f32(
                &output_grad,
                &weights,
                &[1, 1, 2, 2],
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                ElementwiseLaunchConfig::default(),
            )
            .expect("conv backward input");
        let weight_grad = backend
            .conv2d_backward_weights_f32(
                &input,
                &output_grad,
                &[1, 1, 1, 1],
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                ElementwiseLaunchConfig::default(),
            )
            .expect("conv backward weights");

        assert_eq!(bias_grad.data, vec![100.0]);
        assert_eq!(input_grad.shape, vec![1, 1, 2, 2]);
        assert_eq!(input_grad.data, vec![20.0, 40.0, 60.0, 80.0]);
        assert_eq!(weight_grad.shape, vec![1, 1, 1, 1]);
        assert_eq!(weight_grad.data, vec![300.0]);
    }

    #[test]
    fn runs_single_conv_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-runtime-conv.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0])
            .expect("tensor");
        let result = executable.forward(&input).expect("forward should succeed");

        assert_eq!(
            result.final_output().expect("final output").data,
            vec![2.5, 4.5]
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn runs_route_and_upsample_layers() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1

                [upsample]
                stride=2
                scale=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-runtime-route.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");
        let result = executable.forward(&input).expect("forward should succeed");
        let final_output = result.final_output().expect("final output");

        assert_eq!(final_output.shape.width, 4);
        assert_eq!(final_output.shape.height, 4);
        assert_eq!(final_output.get(0, 0, 0, 0), 1.0);
        assert_eq!(final_output.get(0, 0, 3, 3), 4.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn collects_yolo_head_outputs() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=6

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::filled(executable.network.net.input_shape(), 1.0);
        let result = executable.forward(&input).expect("forward should succeed");

        assert_eq!(result.yolo_heads.len(), 1);
        assert_eq!(result.yolo_heads[0].output.data, input.data);
    }

    #[test]
    fn backward_runs_single_conv_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path =
            PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-conv.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0])
            .expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        let output_gradient =
            Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![10.0, 20.0]).expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");

        assert_eq!(backward.input_gradient.data, vec![20.0, 40.0]);
        match &backward.param_gradients[0] {
            LayerParams::Convolutional(conv) => {
                assert_eq!(conv.biases, vec![30.0]);
                assert_eq!(conv.weights, vec![50.0]);
            }
            other => panic!("unexpected param gradients: {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn backward_runs_avgpool_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [avgpool]
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        assert_eq!(
            forward.final_output().expect("final output").data,
            vec![2.5]
        );

        let output_gradient =
            Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![8.0]).expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");
        assert_eq!(backward.input_gradient.data, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn backward_runs_local_avgpool_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [local_avgpool]
                size=2
                stride=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        assert_eq!(
            forward.final_output().expect("final output").data,
            vec![2.5, 3.0, 3.5, 4.0]
        );

        let output_gradient = Tensor::from_data(
            executable.network.layers[0].output_shape,
            vec![4.0, 6.0, 8.0, 10.0],
        )
        .expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");
        assert_eq!(backward.input_gradient.data, vec![1.0, 4.0, 5.0, 18.0]);
    }

    #[test]
    fn backward_route_scatter_matches_cpp_style_delta_flow() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path =
            PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-route.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![3.0, 4.0])
            .expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        let output_gradient =
            Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![7.0, 11.0]).expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");

        assert_eq!(backward.layer_output_gradients[1].data, vec![7.0, 11.0]);
        assert_eq!(backward.layer_output_gradients[0].data, vec![7.0, 11.0]);
        assert_eq!(backward.input_gradient.data, vec![7.0, 11.0]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn shared_convolutional_layers_count_one_parameter_block() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [convolutional]
                filters=1
                size=1
                stride=1
                share_index=0
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        assert_eq!(network.total_parameter_count(), 2);
    }

    #[test]
    fn backward_accumulate_shared_conv_updates_into_owner_buffers() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [convolutional]
                filters=1
                size=1
                stride=1
                share_index=0
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path =
            PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-shared.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![3.0]).expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        let output_gradient =
            Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![1.0]).expect("tensor");
        let backward = executable
            .backward_accumulate(&input, &forward, &output_gradient)
            .expect("backward should succeed");

        assert_eq!(backward.input_gradient.data, vec![4.0]);
        match &executable.train_buffers[0] {
            LayerTrainBuffers::Convolutional(conv) => {
                assert_eq!(conv.bias_updates, vec![3.0]);
                assert_eq!(conv.weight_updates, vec![12.5]);
            }
            other => panic!("unexpected owner train buffers: {other:?}"),
        }
        assert_eq!(executable.train_buffers[1], LayerTrainBuffers::None);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn sgd_update_applies_decay_step_and_momentum() {
        let mut weights = vec![2.0];
        let mut updates = vec![3.0];
        sgd_update(&mut weights, &mut updates, 0.1, 0.9, 0.01, 2);
        assert!((weights[0] - 2.148).abs() < 1e-6);
        assert!((updates[0] - 2.664).abs() < 1e-6);
    }

    #[test]
    fn compute_learning_rate_handles_burn_in_steps_and_poly() {
        let base = NetConfig {
            width: 1,
            height: 1,
            channels: 1,
            batch: 1,
            subdivisions: 1,
            learning_rate: 0.01,
            momentum: 0.9,
            decay: 0.0005,
            max_batches: 100,
            burn_in: 10,
            policy: LrPolicy::Constant,
            steps: Vec::new(),
            scales: Vec::new(),
            equidistant_point: 0,
            badlabels_rejection_percentage: 0.0,
            num_sigmas_reject_badlabels: 0.0,
        };
        assert_eq!(compute_learning_rate(&base, 0), 0.0);
        assert!((compute_learning_rate(&base, 5) - 0.000625).abs() < 1e-9);
        assert!((compute_learning_rate(&base, 10) - 0.01).abs() < 1e-9);

        let steps = NetConfig {
            policy: LrPolicy::Steps,
            burn_in: 0,
            steps: vec![20, 40],
            scales: vec![0.1, 0.1],
            ..base.clone()
        };
        assert!((compute_learning_rate(&steps, 10) - 0.01).abs() < 1e-9);
        assert!((compute_learning_rate(&steps, 20) - 0.001).abs() < 1e-9);
        assert!((compute_learning_rate(&steps, 40) - 0.0001).abs() < 1e-9);

        let poly = NetConfig {
            policy: LrPolicy::Poly,
            burn_in: 0,
            ..base
        };
        assert!((compute_learning_rate(&poly, 50) - 0.000625).abs() < 1e-9);
        assert_eq!(compute_learning_rate(&poly, 100), 0.0);
    }

    #[test]
    fn backward_conv_supports_swish_with_cached_activation_input() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=swish
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path =
            PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-swish.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0])
            .expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        assert!(forward.layer_activation_inputs[0].is_some());

        let output_gradient =
            Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 1.0]).expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");

        let expected_bias_grad = {
            let s1 = 1.0 / (1.0 + (-1.0f32).exp());
            let s2 = 1.0 / (1.0 + (-2.0f32).exp());
            (s1 + 1.0 * s1 * (1.0 - s1)) + (s2 + 2.0 * s2 * (1.0 - s2))
        };
        let expected_weight_grad = {
            let s1 = 1.0 / (1.0 + (-1.0f32).exp());
            let s2 = 1.0 / (1.0 + (-2.0f32).exp());
            1.0 * (s1 + 1.0 * s1 * (1.0 - s1)) + 2.0 * (s2 + 2.0 * s2 * (1.0 - s2))
        };

        match &backward.param_gradients[0] {
            LayerParams::Convolutional(conv) => {
                assert!((conv.biases[0] - expected_bias_grad).abs() < 1e-6);
                assert!((conv.weights[0] - expected_weight_grad).abs() < 1e-6);
            }
            other => panic!("unexpected param gradients: {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn backward_runs_batchnorm_conv_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-bn.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&3.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![3.0]).expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        let output_gradient =
            Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![4.0]).expect("tensor");
        let backward = executable
            .backward(&input, &forward, &output_gradient)
            .expect("backward should succeed");

        let inv_std = 1.0 / (3.0f32 + super::BATCH_NORM_EPSILON).sqrt();
        let normalized = (6.0 - 1.0) * inv_std;
        let grad_preconv = 4.0 * 2.0 * inv_std;

        assert!((backward.input_gradient.data[0] - 2.0 * grad_preconv).abs() < 1e-5);
        match &backward.param_gradients[0] {
            LayerParams::Convolutional(conv) => {
                assert!((conv.biases[0] - 4.0).abs() < 1e-6);
                assert!((conv.weights[0] - 3.0 * grad_preconv).abs() < 1e-5);
                let bn = conv.batch_norm.as_ref().expect("batchnorm grads");
                assert!((bn.scales[0] - 4.0 * normalized).abs() < 1e-5);
            }
            other => panic!("unexpected param gradients: {other:?}"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn backward_accumulate_updates_batchnorm_running_stats() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);
        match &mut executable.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 1.0;
                let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
                batch_norm.rolling_mean[0] = 10.0;
                batch_norm.rolling_variance[0] = 20.0;
            }
            other => panic!("unexpected params: {other:?}"),
        }

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
            .expect("input tensor");
        let forward = executable.forward(&input).expect("forward should succeed");
        let seeds = vec![Some(Tensor::filled(
            executable.network.layers[0].output_shape,
            0.0,
        ))];
        executable
            .backward_accumulate_seeded(&input, &forward, &seeds)
            .expect("backward should update stats");

        let params = match &executable.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
        assert!((batch_norm.rolling_mean[0] - 9.3).abs() < 1.0e-6);
        assert!((batch_norm.rolling_variance[0] - 18.2).abs() < 1.0e-6);
    }

    #[test]
    fn backward_seeded_accumulates_multiple_yolo_heads() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=6
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1

                [route]
                layers=-2

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = ExecutableNetwork::new(network);

        let path =
            PathBuf::from(std::env::temp_dir()).join("darknet-runtime-backward-multihead.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0.0f32.to_le_bytes(), 6).flatten());
        bytes.extend(std::iter::repeat_n(1.0f32.to_le_bytes(), 6).flatten());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0]).expect("tensor");
        let forward = executable.forward(&input).expect("forward should succeed");

        let mut seeds = vec![None; executable.network.layers.len()];
        seeds[1] = Some(Tensor::filled(forward.layer_outputs[1].shape, 1.0));
        seeds[3] = Some(Tensor::filled(forward.layer_outputs[3].shape, 1.0));
        let backward = executable
            .backward_seeded(&input, &forward, &seeds)
            .expect("seeded backward should succeed");

        assert_eq!(backward.layer_output_gradients[1].data, vec![1.0; 6]);
        assert_eq!(backward.layer_output_gradients[3].data, vec![1.0; 6]);
        assert_eq!(backward.layer_output_gradients[2].data, vec![1.0; 6]);
        assert_eq!(backward.layer_output_gradients[0].data, vec![2.0; 6]);
        assert_eq!(backward.input_gradient.data, vec![12.0]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gpu_activation_adapter_maps_expected_variants() {
        assert_eq!(
            Activation::Mish.to_cuda_f32_kind(),
            Some(F32ActivationKind::Mish)
        );
        assert_eq!(
            Activation::Other("hard_mish".to_owned()).to_cuda_f32_kind(),
            None
        );
    }

    #[test]
    fn gpu_runtime_runs_route_shortcut_and_upsample_layers() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [upsample]
                stride=2
                scale=1

                [route]
                layers=-1

                [shortcut]
                from=-2
                activation=relu
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = GpuExecutableNetwork::new(network, FakeBackend);
        let input =
            Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 2.0]).expect("tensor");
        let result = executable
            .forward(&input)
            .expect("gpu forward should succeed");
        let final_output = result.final_output().expect("final output");

        assert_eq!(final_output.shape, TensorShape::new(4, 2, 1, 1));
        assert_eq!(
            final_output.data,
            vec![2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0]
        );
    }

    #[test]
    fn gpu_runtime_runs_single_conv_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-gpu-runtime-conv.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0])
            .expect("tensor");
        let result = executable
            .forward(&input)
            .expect("gpu forward should succeed");

        assert_eq!(
            result.final_output().expect("final output").data,
            vec![2.5, 4.5]
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gpu_runtime_runs_maxpool_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=4
                height=4
                channels=1

                [maxpool]
                size=2
                stride=2
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = GpuExecutableNetwork::new(network, FakeBackend);
        let input = Tensor::from_data(
            TensorShape::new(4, 4, 1, 1),
            vec![
                1.0, 5.0, 2.0, 4.0, 3.0, 7.0, 6.0, 8.0, 0.0, 9.0, 2.0, 1.0, 4.0, 3.0, 5.0, 6.0,
            ],
        )
        .expect("tensor");
        let result = executable
            .forward(&input)
            .expect("gpu forward should succeed");

        assert_eq!(
            result.final_output().expect("final output").data,
            vec![7.0, 8.0, 9.0, 6.0]
        );
    }

    #[test]
    fn gpu_runtime_runs_avgpool_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [avgpool]
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = GpuExecutableNetwork::new(network, FakeBackend);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");
        let result = executable
            .forward(&input)
            .expect("gpu forward should succeed");

        assert_eq!(result.final_output().expect("final output").data, vec![2.5]);
    }

    #[test]
    fn gpu_runtime_runs_local_avgpool_layer() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [local_avgpool]
                size=2
                stride=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let executable = GpuExecutableNetwork::new(network, FakeBackend);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");
        let result = executable
            .forward(&input)
            .expect("gpu forward should succeed");

        assert_eq!(
            result.final_output().expect("final output").data,
            vec![2.5, 3.0, 3.5, 4.0]
        );
    }

    #[test]
    fn gpu_runtime_reuses_uploaded_conv_params() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let backend = CountingBackend::default();
        let mut executable = GpuExecutableNetwork::new(network, backend.clone());

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-gpu-runtime-cache.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        assert_eq!(backend.upload_count(), 2);

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0])
            .expect("tensor");
        executable
            .forward(&input)
            .expect("first gpu forward should succeed");
        executable
            .forward(&input)
            .expect("second gpu forward should succeed");

        assert_eq!(backend.upload_count(), 4);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gpu_weight_updates_preserve_momentum_between_iterations() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
        executable
            .init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0])
            .expect("input tensor");
        let seeds = vec![Some(
            Tensor::from_data(executable.network.layers[0].output_shape, vec![3.0])
                .expect("seed tensor"),
        )];

        for _ in 0..2 {
            let forward = executable
                .forward_with_state(&input)
                .expect("GPU forward should succeed");
            executable
                .backward_accumulate_seeded_gpu(&forward, &seeds)
                .expect("GPU backward should accumulate");
            executable
                .update_weights_gpu(0.1, 0.9, 0.0, 1)
                .expect("GPU update should succeed");
        }

        let params = match &executable.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        assert!((params.weights[0] - 1.74).abs() < 1.0e-6);
        assert!((params.biases[0] - 0.87).abs() < 1.0e-6);

        let train_buffers = executable.device_train_buffers.borrow();
        let train = train_buffers[0].as_ref().expect("train buffers exist");
        assert!((train.weight_updates.data[0] - 10.26).abs() < 1.0e-6);
        assert!((train.bias_updates.data[0] - 5.13).abs() < 1.0e-6);
    }

    #[test]
    fn gpu_batchnorm_conv_accumulates_and_updates_scale() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut executable.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 2.0;
                params.biases[0] = 0.5;
                let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
                batch_norm.scales[0] = 1.5;
                batch_norm.rolling_mean[0] = 1.0;
                batch_norm.rolling_variance[0] = 4.0 - super::BATCH_NORM_EPSILON;
            }
            other => panic!("unexpected params: {other:?}"),
        }
        executable
            .init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0])
            .expect("input tensor");
        let seeds = vec![Some(
            Tensor::from_data(executable.network.layers[0].output_shape, vec![3.0])
                .expect("seed tensor"),
        )];
        let forward = executable
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        executable
            .backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("GPU backward should accumulate");

        {
            let train_buffers = executable.device_train_buffers.borrow();
            let train = train_buffers[0].as_ref().expect("train buffers exist");
            let scale_updates = train.scale_updates.as_ref().expect("scale updates exist");
            assert!((scale_updates.data[0] - 4.5).abs() < 1.0e-6);
        }

        executable
            .update_weights_gpu(0.1, 0.0, 0.0, 1)
            .expect("GPU update should succeed");
        let params = match &executable.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
        assert!((batch_norm.scales[0] - 1.95).abs() < 1.0e-6);
    }

    #[test]
    fn gpu_batchnorm_conv_updates_running_stats() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut executable.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 1.0;
                let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
                batch_norm.rolling_mean[0] = 10.0;
                batch_norm.rolling_variance[0] = 20.0;
            }
            other => panic!("unexpected params: {other:?}"),
        }
        executable
            .init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
            .expect("input tensor");
        let seeds = vec![Some(Tensor::filled(
            executable.network.layers[0].output_shape,
            0.0,
        ))];
        let forward = executable
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        executable
            .backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("GPU backward should update stats");

        let params = match &executable.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
        assert!((batch_norm.rolling_mean[0] - 9.3).abs() < 1.0e-6);
        assert!((batch_norm.rolling_variance[0] - 18.2).abs() < 1.0e-6);
    }

    #[test]
    fn gpu_batchnorm_running_stats_update_without_gradient_seed() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut executable.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 1.0;
                let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
                batch_norm.rolling_mean[0] = 10.0;
                batch_norm.rolling_variance[0] = 20.0;
            }
            other => panic!("unexpected params: {other:?}"),
        }
        executable
            .init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
            .expect("input tensor");
        let seeds = vec![None; executable.network.layers.len()];
        let forward = executable
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        executable
            .backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("GPU backward should update stats");

        let params = match &executable.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
        assert!((batch_norm.rolling_mean[0] - 9.3).abs() < 1.0e-6);
        assert!((batch_norm.rolling_variance[0] - 18.2).abs() < 1.0e-6);

        let train_buffers = executable.device_train_buffers.borrow();
        let train = train_buffers[0].as_ref().expect("train buffers exist");
        assert_eq!(train.weight_updates.data, vec![0.0]);
        assert_eq!(train.bias_updates.data, vec![0.0]);
    }

    #[test]
    fn gpu_yolo_loss_backward_matches_cpu_train_buffers() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=8
                height=8
                channels=1

                [convolutional]
                filters=6
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut cpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights = vec![0.03, -0.02, 0.05, 0.07, -0.04, 0.01];
                params.biases = vec![0.2, -0.1, 0.05, -0.15, 0.3, -0.25];
            }
            other => panic!("unexpected params: {other:?}"),
        }
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(
            cpu.network.net.input_shape(),
            (0..64).map(|idx| (idx as f32 + 1.0) / 64.0).collect(),
        )
        .expect("input tensor");
        let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
        let gpu_forward = gpu
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

        let labels = vec![vec![YoloLabel {
            class_id: 0,
            x: 0.5,
            y: 0.5,
            w: 0.25,
            h: 0.25,
        }]];
        let training =
            compute_yolo_training_loss(&cpu.network, &cpu_forward, &labels).expect("loss computes");
        assert!(training.loss.is_finite());
        assert!(training.positive_matches > 0);

        cpu.backward_accumulate_seeded(&input, &cpu_forward, &training.layer_output_gradient_seeds)
            .expect("CPU backward should accumulate");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &training.layer_output_gradient_seeds)
            .expect("GPU backward should accumulate");

        let cpu_train = match &cpu.train_buffers[0] {
            LayerTrainBuffers::Convolutional(train) => train,
            other => panic!("unexpected CPU train buffers: {other:?}"),
        };
        let gpu_buffers = gpu.device_train_buffers.borrow();
        let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");

        assert_eq!(
            cpu_train.bias_updates.len(),
            gpu_train.bias_updates.data.len()
        );
        for (cpu_value, gpu_value) in cpu_train
            .bias_updates
            .iter()
            .zip(&gpu_train.bias_updates.data)
        {
            assert!((cpu_value - gpu_value).abs() < 1.0e-5);
        }
        assert_eq!(
            cpu_train.weight_updates.len(),
            gpu_train.weight_updates.data.len()
        );
        for (cpu_value, gpu_value) in cpu_train
            .weight_updates
            .iter()
            .zip(&gpu_train.weight_updates.data)
        {
            assert!((cpu_value - gpu_value).abs() < 1.0e-5);
        }
    }

    #[test]
    fn gpu_avgpool_backward_feeds_previous_convolution() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [avgpool]
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut cpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 1.0;
                params.biases[0] = 0.0;
            }
            other => panic!("unexpected params: {other:?}"),
        }
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(cpu.network.net.input_shape(), vec![1.0, 2.0, 3.0, 4.0])
            .expect("input tensor");
        let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
        let gpu_forward = gpu
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

        let mut seeds = vec![None; cpu.network.layers.len()];
        seeds[1] = Some(
            Tensor::from_data(cpu.network.layers[1].output_shape, vec![8.0]).expect("seed tensor"),
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward should accumulate");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("GPU backward should accumulate");

        let cpu_train = match &cpu.train_buffers[0] {
            LayerTrainBuffers::Convolutional(train) => train,
            other => panic!("unexpected CPU train buffers: {other:?}"),
        };
        let gpu_buffers = gpu.device_train_buffers.borrow();
        let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");
        assert_eq!(cpu_train.bias_updates, vec![8.0]);
        assert_eq!(cpu_train.weight_updates, vec![20.0]);
        assert_eq!(gpu_train.bias_updates.data, cpu_train.bias_updates);
        assert_eq!(gpu_train.weight_updates.data, cpu_train.weight_updates);
    }

    #[test]
    fn gpu_local_avgpool_backward_feeds_previous_convolution() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [local_avgpool]
                size=2
                stride=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        match &mut cpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights[0] = 1.0;
                params.biases[0] = 0.0;
            }
            other => panic!("unexpected params: {other:?}"),
        }
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");

        let input = Tensor::from_data(cpu.network.net.input_shape(), vec![1.0, 2.0, 3.0, 4.0])
            .expect("input tensor");
        let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
        let gpu_forward = gpu
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

        let mut seeds = vec![None; cpu.network.layers.len()];
        seeds[1] = Some(
            Tensor::from_data(
                cpu.network.layers[1].output_shape,
                vec![4.0, 6.0, 8.0, 10.0],
            )
            .expect("seed tensor"),
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward should accumulate");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("GPU backward should accumulate");

        let cpu_train = match &cpu.train_buffers[0] {
            LayerTrainBuffers::Convolutional(train) => train,
            other => panic!("unexpected CPU train buffers: {other:?}"),
        };
        let gpu_buffers = gpu.device_train_buffers.borrow();
        let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");
        assert_eq!(cpu_train.bias_updates, vec![28.0]);
        assert_eq!(cpu_train.weight_updates, vec![96.0]);
        assert_eq!(gpu_train.bias_updates.data, cpu_train.bias_updates);
        assert_eq!(gpu_train.weight_updates.data, cpu_train.weight_updates);
    }

    #[test]
    fn gpu_runtime_matches_cpu_for_conv_maxpool_slice() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=4
                height=4
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [maxpool]
                size=2
                stride=2

                [upsample]
                stride=2
                scale=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-runtime-parity.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        cpu.load_weights_file(&path)
            .expect("cpu weights should load");
        gpu.load_weights_file(&path)
            .expect("gpu weights should load");

        let input = Tensor::from_data(
            cpu.network.net.input_shape(),
            vec![
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0,
            ],
        )
        .expect("tensor");
        let cpu_result = cpu.forward(&input).expect("cpu forward should succeed");
        let gpu_result = gpu.forward(&input).expect("gpu forward should succeed");

        assert_same_forward(&cpu_result, &gpu_result);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gpu_forward_compact_matches_full_forward_on_small_network() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [maxpool]
                size=2
                stride=2

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-runtime-compact.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(&path, bytes).expect("weights file should write");

        executable
            .load_weights_file(&path)
            .expect("weights should load");
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("tensor");

        let full = executable
            .forward(&input)
            .expect("full forward should succeed");
        let compact = executable
            .forward_compact(&input)
            .expect("compact forward should succeed");

        assert_eq!(compact.final_output, full.final_output().cloned());
        assert_eq!(compact.yolo_heads, full.yolo_heads);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn gpu_runtime_runs_primary_yolov4_tiny_targets() {
        for cfg_name in ["yolov4-tiny.cfg", "yolov4-tiny-3l.cfg"] {
            let network = Network::from_path(repo_cfg(cfg_name)).expect("cfg should parse");
            let unsupported = network
                .layers
                .iter()
                .find(|layer| {
                    !matches!(
                        layer.kind,
                        LayerKind::Convolutional(_)
                            | LayerKind::MaxPool(_)
                            | LayerKind::Route(_)
                            | LayerKind::Shortcut(_)
                            | LayerKind::Upsample(_)
                            | LayerKind::Yolo(_)
                            | LayerKind::GaussianYolo(_)
                            | LayerKind::Region(_)
                            | LayerKind::Softmax { .. }
                            | LayerKind::Cost { .. }
                            | LayerKind::Dropout { .. }
                            | LayerKind::ScaleChannels(_)
                            | LayerKind::Sam(_)
                            | LayerKind::Contrastive { .. }
                    )
                })
                .map(|layer| (layer.index, layer.kind.name().to_owned()));

            assert_eq!(
                unsupported, None,
                "{cfg_name} contains an unsupported GPU layer"
            );
        }
    }

    #[test]
    fn gpu_forward_compact_runs_primary_yolov4_tiny_targets() {
        for (cfg_name, expected_heads) in
            [("yolov4-tiny.cfg", 2usize), ("yolov4-tiny-3l.cfg", 3usize)]
        {
            let network = Network::from_path(repo_cfg(cfg_name)).expect("cfg should parse");
            let executable = GpuExecutableNetwork::new(network, ShapeOnlyBackend);
            let input = executable.zero_input();
            let summary = executable
                .forward_compact(&input)
                .expect("compact gpu forward should succeed");

            assert_eq!(summary.yolo_heads.len(), expected_heads);
            assert!(summary.final_output.is_some());
        }
    }
}
