use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::model::{
    Activation, ConnectedSpec, ConvolutionalSpec, LayerKind, Network, RouteSpec, TensorShape,
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
pub struct YoloHeadOutput {
    pub layer_index: usize,
    pub output: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardResult {
    pub layer_outputs: Vec<Tensor>,
    pub yolo_heads: Vec<YoloHeadOutput>,
}

impl ForwardResult {
    pub fn final_output(&self) -> Option<&Tensor> {
        self.layer_outputs.last()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuForwardSummary {
    pub final_output: Option<Tensor>,
    pub yolo_heads: Vec<YoloHeadOutput>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableNetwork {
    pub network: Network,
    pub params: Vec<LayerParams>,
}

impl ExecutableNetwork {
    pub fn new(network: Network) -> Self {
        let params = network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        Self { network, params }
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

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }

        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                input
            } else {
                layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let output = match (&layer.kind, params) {
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv)) => {
                    forward_convolutional(spec, conv, current_input, layer.output_shape)
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    forward_connected(spec, connected, current_input, layer.output_shape)
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    false,
                )?,
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    true,
                )?,
                (LayerKind::AvgPool, LayerParams::None) => forward_avgpool(current_input),
                (LayerKind::Route(spec), LayerParams::None) => {
                    forward_route(spec, layer.index, &layer_outputs, layer.output_shape)?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    forward_shortcut(spec, current_input, layer.index, &layer_outputs)?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => {
                    forward_upsample(spec, current_input, layer.output_shape)?
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Softmax { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::ScaleChannels(_), LayerParams::None)
                | (LayerKind::Sam(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => current_input.clone(),
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    forward_reorg(spec.stride, spec.reverse, current_input, layer.output_shape)?
                }
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
        }

        Ok(ForwardResult {
            layer_outputs,
            yolo_heads,
        })
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
}

#[derive(Clone, Debug)]
struct GpuConvolutionLayerParams<T> {
    weights: T,
    biases: T,
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
        Self {
            backend,
            network,
            params,
            device_conv_params,
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
        self.warm_device_conv_params()?;
        Ok(header)
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
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
        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                &input_gpu
            } else {
                gpu_layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let output_gpu = match (&layer.kind, params) {
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv)) => {
                    let cached = self.cached_convolution_params(
                        layer.index,
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
                | (LayerKind::LocalAvgPool(_), _)
                | (LayerKind::AvgPool, _)
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

            if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }

            gpu_layer_outputs.push(output_gpu);
            layer_outputs.push(output);
        }

        Ok(ForwardResult {
            layer_outputs,
            yolo_heads,
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
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv)) => {
                    let cached = self.cached_convolution_params(
                        layer.index,
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
                | (LayerKind::LocalAvgPool(_), _)
                | (LayerKind::AvgPool, _)
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

    fn download_tensor(
        &self,
        tensor: &B::Tensor,
        shape: TensorShape,
    ) -> std::result::Result<Tensor, B::Error> {
        let data = self.backend.download_f32(tensor)?;
        Ok(Tensor { shape, data })
    }

    fn clear_device_conv_params(&mut self) {
        for entry in self.device_conv_params.get_mut().iter_mut() {
            *entry = None;
        }
    }

    fn warm_device_conv_params(&self) -> Result<()> {
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            if let (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv)) =
                (&layer.kind, params)
            {
                let _ = self.cached_convolution_params(
                    layer.index,
                    layer.input_shape.channels,
                    spec,
                    conv,
                )?;
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
        let uploaded = GpuConvolutionLayerParams {
            weights: self
                .backend
                .upload_f32(
                    &[
                        spec.filters,
                        input_channels / spec.groups.max(1),
                        spec.size,
                        spec.size,
                    ],
                    &folded_weights,
                )
                .map_err(|err| backend_error("upload convolution weights", err))?,
            biases: self
                .backend
                .upload_f32(&[spec.filters], &folded_biases)
                .map_err(|err| backend_error("upload convolution biases", err))?,
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
}

fn default_layer_params(layer: &crate::model::Layer) -> LayerParams {
    match &layer.kind {
        LayerKind::Convolutional(spec) => {
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
    let mut output = Tensor::zeros(output_shape);
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

                    let value = normalize_and_activate(
                        acc,
                        filter,
                        &params.biases,
                        params.batch_norm.as_ref(),
                        &spec.activation,
                    );
                    output.set(b, filter, oy, ox, value);
                }
            }
        }
    }

    output
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

            let value = normalize_and_activate(
                acc,
                out_idx,
                &params.biases,
                params.batch_norm.as_ref(),
                &spec.activation,
            );
            output.data[b * spec.output + out_idx] = value;
        }
    }

    output
}

fn normalize_and_activate(
    value: f32,
    index: usize,
    biases: &[f32],
    batch_norm: Option<&BatchNormParams>,
    activation: &Activation,
) -> f32 {
    let biased = if let Some(batch_norm) = batch_norm {
        let normalized = (value - batch_norm.rolling_mean[index])
            / (batch_norm.rolling_variance[index] + BATCH_NORM_EPSILON).sqrt();
        normalized * batch_norm.scales[index] + biases[index]
    } else {
        value + biases[index]
    };

    activate(activation.clone(), biased)
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

fn forward_shortcut(
    spec: &crate::model::ShortcutSpec,
    current_input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<Tensor> {
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
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok(output)
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
    use crate::model::{Activation, ConvolutionalSpec, LayerKind, Network, TensorShape};
    use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

    use super::{
        ConvolutionalParams, ExecutableNetwork, ForwardResult, GpuExecutableNetwork, Tensor,
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

        fn activation_f32(
            &self,
            x: &Self::Tensor,
            activation: F32ActivationKind,
            config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            self.inner.activation_f32(x, activation, config)
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

        fn activation_f32(
            &self,
            x: &Self::Tensor,
            _activation: F32ActivationKind,
            _config: ElementwiseLaunchConfig,
        ) -> std::result::Result<Self::Tensor, Self::Error> {
            Ok(x.clone())
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
