//! CPU and backend-generic Darknet execution.
//!
//! The execution lifecycle mirrors C++ Darknet without reproducing its raw
//! pointer ownership:
//!
//! 1. [`crate::model::Network`] describes layers and shapes.
//! 2. [`ExecutableNetwork`] owns parameter values, train buffers, and CPU
//!    scratch storage.
//! 3. `forward` records activation, recurrent, and dropout state needed by
//!    backward.
//! 4. `backward` walks the graph in reverse and returns input/parameter
//!    gradients; accumulation and SGD update persistent buffers separately.
//! 5. `GpuExecutableNetwork` uploads the same parameter model and uses the CPU
//!    reference for operations without a native backend kernel.
//!
//! Tensor indexing is planar and contiguous: each batch contains all channels,
//! each channel contains `height * width` values, and channels are stored in
//! channel-major order exactly like Darknet's `float *output` arrays.

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::maxpool_depth::{backward_maxpool_depth, forward_maxpool_depth};
use crate::model::{
    Activation, ConnectedSpec, ConvolutionalSpec, CrnnSpec, CropSpec, DropblockSpec, GruSpec,
    LayerKind, LocalSpec, LrPolicy, LstmSpec, NetConfig, Network, RnnSpec, RouteSpec, TensorShape,
};
use crate::weights::WeightsHeader;
use darknet_cuda::{
    AdamParams, AdamUpdateItem, ElementwiseLaunchConfig, F16TensorOpsBackend, F32ActivationKind, F32TensorOpsBackend,
    NanInfMode, SgdUpdateItem, dropout_unit,
};

const BATCH_NORM_EPSILON: f32 = 0.000_01;
/// Rolling-stat EMA keep factor of the GPU batch-statistics training path:
/// C++ CUDA `scal_ongpu(.99)` + `axpy_ongpu(.01)` / cuDNN factor `.01`.
const BATCH_NORM_ROLLING_KEEP: f32 = 0.99;
/// Rolling-stat EMA keep factor of the host (CPU) training path: C++
/// `forward_batchnorm_layer` uses `scal_cpu(.9)` + `axpy_cpu(.1)`.
const BATCH_NORM_ROLLING_KEEP_CPU: f32 = 0.9;

/// Backward-pass policy for the few ops where C++ Darknet's hand-written gradient is not the
/// exact derivative of its forward. `Analytical` (default) is the exact derivative, `Cpp`
/// reproduces the C++ shortcut for training parity. Selected by `DARKNET_GRAD_MODE=analytical|cpp`
/// and read at each backward. Affects only: softmax (C++ passes `dy` through; its cost layer
/// folds the Jacobian in) and `scale_channels` with `scale_wh=1` (C++ divides the scale gradient
/// by `w*h`). See `tasks/rust/todo.md` "Analytical gradients everywhere" for the full audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GradMode {
    #[default]
    Analytical,
    Cpp,
}

impl GradMode {
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("DARKNET_GRAD_MODE").unwrap_or_default())
    }

    pub fn parse(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "cpp" | "c++" => Self::Cpp,
            _ => Self::Analytical,
        }
    }
}

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

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn channel_area(&self) -> usize {
        self.shape.width * self.shape.height
    }

    fn batch_stride(&self) -> usize {
        self.shape.channels * self.channel_area()
    }

    fn offset(&self, batch: usize, channel: usize, y: usize, x: usize) -> usize {
        // Keep the indexing formula in one place.  CPU layers and host-side
        // GPU checks then cannot accidentally interpret this tensor as NHWC.
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
    pub biases: Vec<f32>,
    pub scales: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_variance: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StandaloneBatchNormTrainBuffers {
    pub bias_updates: Vec<f32>,
    pub scale_updates: Vec<f32>,
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

/// Parameters for a legacy Darknet local layer.  The leading dimension of
/// both arrays is output location rather than just output channel.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalParams {
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
}

/// Weighted `[shortcut]` parameters (tasks/rust/layers.md Task 6).
#[derive(Clone, Debug, PartialEq)]
pub struct ShortcutParams {
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RnnParams {
    pub input: ConnectedParams,
    pub self_params: ConnectedParams,
    pub output: ConnectedParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GruParams {
    pub input_z: ConnectedParams,
    pub state_z: ConnectedParams,
    pub input_r: ConnectedParams,
    pub state_r: ConnectedParams,
    pub input_h: ConnectedParams,
    pub state_h: ConnectedParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LstmParams {
    pub wf: ConnectedParams,
    pub wi: ConnectedParams,
    pub wg: ConnectedParams,
    pub wo: ConnectedParams,
    pub uf: ConnectedParams,
    pub ui: ConnectedParams,
    pub ug: ConnectedParams,
    pub uo: ConnectedParams,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrnnParams {
    pub input: ConvolutionalParams,
    pub self_params: ConvolutionalParams,
    pub output: ConvolutionalParams,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum LayerParams {
    None,
    BatchNorm(BatchNormParams),
    Convolutional(ConvolutionalParams),
    Connected(ConnectedParams),
    Local(LocalParams),
    Shortcut(ShortcutParams),
    Gru(GruParams),
    Rnn(RnnParams),
    Lstm(LstmParams),
    Crnn(CrnnParams),
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
pub struct LocalTrainBuffers {
    pub bias_updates: Vec<f32>,
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShortcutTrainBuffers {
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RnnTrainBuffers {
    pub input: ConnectedTrainBuffers,
    pub self_params: ConnectedTrainBuffers,
    pub output: ConnectedTrainBuffers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GruTrainBuffers {
    pub input_z: ConnectedTrainBuffers,
    pub state_z: ConnectedTrainBuffers,
    pub input_r: ConnectedTrainBuffers,
    pub state_r: ConnectedTrainBuffers,
    pub input_h: ConnectedTrainBuffers,
    pub state_h: ConnectedTrainBuffers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LstmTrainBuffers {
    pub wf: ConnectedTrainBuffers,
    pub wi: ConnectedTrainBuffers,
    pub wg: ConnectedTrainBuffers,
    pub wo: ConnectedTrainBuffers,
    pub uf: ConnectedTrainBuffers,
    pub ui: ConnectedTrainBuffers,
    pub ug: ConnectedTrainBuffers,
    pub uo: ConnectedTrainBuffers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrnnTrainBuffers {
    pub input: ConvolutionalTrainBuffers,
    pub self_params: ConvolutionalTrainBuffers,
    pub output: ConvolutionalTrainBuffers,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum LayerTrainBuffers {
    None,
    BatchNorm(StandaloneBatchNormTrainBuffers),
    Convolutional(ConvolutionalTrainBuffers),
    Connected(ConnectedTrainBuffers),
    Local(LocalTrainBuffers),
    Shortcut(ShortcutTrainBuffers),
    Gru(GruTrainBuffers),
    Rnn(RnnTrainBuffers),
    Lstm(LstmTrainBuffers),
    Crnn(CrnnTrainBuffers),
}

#[derive(Clone, Debug, PartialEq)]
pub struct RnnForwardCache {
    pub sequence_batch: usize,
    pub input_outputs: Vec<Tensor>,
    pub input_activation_inputs: Vec<Tensor>,
    pub self_outputs: Vec<Tensor>,
    pub self_activation_inputs: Vec<Tensor>,
    pub previous_states: Vec<Tensor>,
    pub states: Vec<Tensor>,
    pub output_activation_inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LstmForwardCache {
    pub sequence_batch: usize,
    pub wf_outputs: Vec<Tensor>,
    pub wf_activation_inputs: Vec<Tensor>,
    pub wi_outputs: Vec<Tensor>,
    pub wi_activation_inputs: Vec<Tensor>,
    pub wg_outputs: Vec<Tensor>,
    pub wg_activation_inputs: Vec<Tensor>,
    pub wo_outputs: Vec<Tensor>,
    pub wo_activation_inputs: Vec<Tensor>,
    pub uf_outputs: Vec<Tensor>,
    pub uf_activation_inputs: Vec<Tensor>,
    pub ui_outputs: Vec<Tensor>,
    pub ui_activation_inputs: Vec<Tensor>,
    pub ug_outputs: Vec<Tensor>,
    pub ug_activation_inputs: Vec<Tensor>,
    pub uo_outputs: Vec<Tensor>,
    pub uo_activation_inputs: Vec<Tensor>,
    pub previous_states: Vec<Tensor>,
    pub previous_cells: Vec<Tensor>,
    pub forget: Vec<Tensor>,
    pub input_gate: Vec<Tensor>,
    pub candidate: Vec<Tensor>,
    pub output_gate: Vec<Tensor>,
    pub cells: Vec<Tensor>,
    pub states: Vec<Tensor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrnnForwardCache {
    pub sequence_batch: usize,
    pub input_outputs: Vec<Tensor>,
    pub input_activation_inputs: Vec<Tensor>,
    pub self_outputs: Vec<Tensor>,
    pub self_activation_inputs: Vec<Tensor>,
    pub previous_states: Vec<Tensor>,
    pub states: Vec<Tensor>,
    pub output_activation_inputs: Vec<Tensor>,
    pub outputs: Vec<Tensor>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum RecurrentForwardCache {
    Rnn(RnnForwardCache),
    Lstm(LstmForwardCache),
    Crnn(CrnnForwardCache),
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloHeadOutput {
    pub layer_index: usize,
    pub output: Tensor,
}

/// Sampling coordinates captured by a crop forward pass. Coordinates are
/// shared by every channel in a batch image, matching Darknet's crop kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct CropForwardCache {
    pub source_x: Vec<f32>,
    pub source_y: Vec<f32>,
    pub input_scales: Vec<f32>,
    pub input_offsets: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardResult {
    pub layer_outputs: Vec<Tensor>,
    pub layer_activation_inputs: Vec<Option<Tensor>>,
    /// Inverted-dropout masks captured during a training forward pass.
    /// `None` means that the layer was evaluated in inference mode or is not
    /// a dropout layer.
    pub dropout_masks: Vec<Option<Tensor>>,
    pub crop_caches: Vec<Option<CropForwardCache>>,
    pub recurrent_caches: Vec<Option<RecurrentForwardCache>>,
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
    /// Batch-statistics batch-norm state of convolutional layers run by a
    /// training forward; `None` means rolling statistics were used (inference
    /// forward, or a network that needs the CPU backward fallback).
    pub gpu_batch_norm_caches: Vec<Option<GpuBatchNormCache<T>>>,
    pub cpu_forward: ForwardResult,
}

/// What the batch-norm backward pass and the rolling-stat update need from a
/// batch-statistics forward.
#[derive(Clone, Debug)]
pub struct GpuBatchNormCache<T>
where
    T: Clone,
{
    scales: T,
    x_hat: T,
    /// Biased batch variance.
    variance: T,
    /// Updated `(rolling_mean, rolling_variance)`; `None` for shared layers,
    /// whose owner keeps the statistics.
    rolling: Option<(T, T)>,
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
            LayerKind::Connected(spec) => {
                let nweights = spec.output * l.input_shape.element_count();
                spec.output + nweights
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
    /// Training iteration in progress (C++ `get_current_iteration`): drives the
    /// DropBlock ramp and the `dont_update`/`burnin_update` gates (see
    /// [`ExecutableNetwork::set_train_iteration`]).
    train_iteration: usize,
}

impl ExecutableNetwork {
    pub fn new(network: Network) -> Self {
        // Preserve one parameter/train-buffer slot per graph layer.  Layers
        // without weights use `None`, which keeps Rust indices aligned with
        // Darknet's C++ layer indices used by route and shortcut references.
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
            train_iteration: 0,
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

    /// Sets the 0-based index of the training iteration in progress (the
    /// CLI's `batch_num`).  The forward pass uses it for the DropBlock ramp;
    /// the update uses `iteration + 1` (C++ `seen / (batch * subdivisions)`
    /// after the iteration's batches) for `dont_update` / `burnin_update`.
    pub fn set_train_iteration(&mut self, iteration: usize) {
        self.train_iteration = iteration;
    }

    pub fn zero_train_buffers(&mut self) {
        self.train_buffers = self
            .network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect();
    }

    /// Apply one Darknet SGD step to the persistent parameter/update buffers.
    ///
    /// Darknet's backward pass stores a *descent direction* in `*_updates`:
    /// detector targets normally produce `target - prediction`, which is the
    /// negative of the mathematical loss derivative.  Consequently the
    /// update below adds `learning_rate / batch * update` to a parameter.  The
    /// convention is intentional and is shared by the CPU runtime and every
    /// backend implementing [`F32TensorOpsBackend::sgd_update_f32`].
    pub fn update_weights(&mut self, lr: f32, momentum: f32, decay: f32, batch: usize) {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        for (layer_index, (params, train)) in self
            .params
            .iter_mut()
            .zip(&mut self.train_buffers)
            .enumerate()
        {
            // C++ `update_network`: per-layer update gating.
            let opts = self.network.layers[layer_index].train;
            if !opts.updates_at(iteration_num, burn_in) {
                continue;
            }
            match (params, train) {
                (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    sgd_update(
                        &mut params.scales,
                        &mut train.scale_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) => {
                    // Per-layer `learning_rate` scale, `grad_centr` (with BN)
                    // before the step and `clip` on the weights after it.
                    let lr = lr * opts.learning_rate_scale;
                    if opts.grad_centr && params.batch_norm.is_some() {
                        let f_size = params.weights.len() / params.biases.len().max(1);
                        if f_size > 0 {
                            train.weight_updates = darknet_cuda::host_ops::gradient_centralization(
                                &train.weight_updates,
                                f_size,
                            );
                        }
                    }
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
                    if opts.clip > 0.0 {
                        params.weights = darknet_cuda::host_ops::constrain(&params.weights, opts.clip);
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
                (LayerParams::Local(params), LayerTrainBuffers::Local(train)) => {
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
                }
                (LayerParams::Shortcut(params), LayerTrainBuffers::Shortcut(train)) => {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Gru(params), LayerTrainBuffers::Gru(train)) => {
                    for (params, train) in [
                        (&mut params.input_z, &mut train.input_z),
                        (&mut params.state_z, &mut train.state_z),
                        (&mut params.input_r, &mut train.input_r),
                        (&mut params.state_r, &mut train.state_r),
                        (&mut params.input_h, &mut train.input_h),
                        (&mut params.state_h, &mut train.state_h),
                    ] {
                        update_connected_params(params, train, lr, momentum, decay, batch);
                    }
                }
                (LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
                    update_connected_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
                    update_connected_params(
                        &mut params.wf,
                        &mut train.wf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wi,
                        &mut train.wi,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wg,
                        &mut train.wg,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wo,
                        &mut train.wo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uf,
                        &mut train.uf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ui,
                        &mut train.ui,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ug,
                        &mut train.ug,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uo,
                        &mut train.uo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
                    update_convolutional_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        self.forward_with_mode(input, false)
    }

    /// Execute an inference-only graph while retaining tensors only until their
    /// final route/shortcut consumer.  This is intentionally restricted to
    /// layers whose inference state is fully represented by their output; all
    /// other networks retain the complete forward graph for compatibility.
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        if self.network.layers.iter().any(|layer| {
            !matches!(
                layer.kind,
                LayerKind::Convolutional(_)
                    | LayerKind::MaxPool(crate::model::MaxPoolSpec { depth: false, .. })
                    | LayerKind::AvgPool
                    | LayerKind::Route(_)
                    | LayerKind::Shortcut(crate::model::ShortcutSpec { weights_type: crate::model::ShortcutWeightsType::None, .. })
                    | LayerKind::Upsample(_)
                    | LayerKind::Yolo(_)
                    | LayerKind::GaussianYolo(_)
                    | LayerKind::Region(_)
                    | LayerKind::Detection(_)
                    | LayerKind::Softmax { .. }
                    | LayerKind::Cost { .. }
                    | LayerKind::Contrastive { .. }
            )
        }) {
            return self
                .forward(input)?
                .final_output()
                .cloned()
                .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()));
        }

        let mut outputs = vec![None; self.network.layers.len()];
        let mut remaining_uses = gpu_layer_output_use_counts(&self.network)?;
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                Some(input)
            } else if layer_uses_previous_input(&layer.kind) {
                Some(
                    outputs
                        .get(layer.index - 1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous inference output for layer #{}",
                                layer.index
                            ))
                        })?,
                )
            } else {
                None
            };

            let output = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    forward_convolutional(
                        spec,
                        conv,
                        current_input.expect("conv input"),
                        layer.output_shape,
                    )
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input.expect("maxpool input"),
                    layer.output_shape,
                    false,
                )?,
                (LayerKind::AvgPool, LayerParams::None) => {
                    forward_avgpool(current_input.expect("avgpool input"))
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    forward_route_compact(spec, layer.index, &outputs, layer.output_shape)?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => forward_shortcut_compact(
                    spec,
                    current_input.expect("shortcut input"),
                    layer.index,
                    &outputs,
                )?,
                (LayerKind::Upsample(spec), LayerParams::None) => forward_upsample(
                    spec,
                    current_input.expect("upsample input"),
                    layer.output_shape,
                )?,
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    forward_softmax(current_input.expect("softmax input"), *groups)?
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => {
                    current_input.expect("pass-through input").clone()
                }
                _ => unreachable!("layer-family preflight must cover compact inference"),
            };

            outputs[layer.index] = Some(output);
            release_gpu_sources_for_layer(
                layer.index,
                &layer.kind,
                &mut remaining_uses,
                &mut outputs,
            )?;
            if layer.index + 1 == self.network.layers.len() {
                return outputs[layer.index].take().ok_or_else(|| {
                    Error::InvalidConfig("missing final inference output".to_owned())
                });
            }
            if remaining_uses[layer.index] == 0 {
                outputs[layer.index] = None;
            }
        }
        Err(Error::InvalidConfig("network has no layers".to_owned()))
    }

    /// Run a training forward pass.  Darknet's dropout layer is active only
    /// during training; keeping this explicit prevents inference commands from
    /// becoming stochastic while allowing the detector trainer to use the
    /// correct inverted-dropout semantics.
    pub fn forward_training(&self, input: &Tensor) -> Result<ForwardResult> {
        self.forward_with_mode(input, true)
    }

    fn forward_with_mode(&self, input: &Tensor, training: bool) -> Result<ForwardResult> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }

        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.network.layers.len());
        let mut crop_caches = Vec::with_capacity(self.network.layers.len());
        let mut recurrent_caches = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();
        let mut rng_state = 0x9e37_79b9_u64;

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            // This match is the Rust equivalent of dispatching a C++ layer's
            // `forward` callback.  Previous outputs stay alive because later
            // route/shortcut/SAM layers and the backward pass need them.
            let current_input = if layer.index == 0 {
                input
            } else {
                layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let mut dropout_mask = None;
            let mut crop_cache = None;
            let mut recurrent_cache = None;
            let (output, activation_input) = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    forward_convolutional_with_cache(spec, conv, current_input, layer.output_shape)
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let pre = crate::deconvolutional::deconv_forward(
                        current_input,
                        &p.weights,
                        &p.biases,
                        spec.size,
                        spec.stride,
                        layer.output_shape,
                    );
                    let data = pre
                        .data
                        .iter()
                        .map(|v| activate(spec.activation.clone(), *v))
                        .collect();
                    (Tensor::from_data(pre.shape, data)?, Some(pre))
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    forward_connected_with_cache(spec, connected, current_input, layer.output_shape)
                }
                (LayerKind::Local(spec), LayerParams::Local(local)) => {
                    forward_local_with_cache(spec, local, current_input, layer.output_shape)?
                }
                (LayerKind::Gru(spec), LayerParams::Gru(gru)) => (
                    forward_gru_layer(spec, gru, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(batch_norm)) => {
                    forward_batchnorm_with_cache(batch_norm, current_input, training)?
                }
                (LayerKind::Crop(spec), LayerParams::None) => {
                    let (output, cache) = forward_crop(
                        spec,
                        current_input,
                        layer.output_shape,
                        training,
                        &mut rng_state,
                    )?;
                    crop_cache = Some(cache);
                    (output, None)
                }
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    forward_maxpool_depth(current_input, layer.output_shape).map(|output| (output, None))?
                }
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
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    forward_shortcut_weighted(spec, params, current_input, layer.index, &layer_outputs)?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    forward_shortcut_with_cache(spec, current_input, layer.index, &layer_outputs)?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => (
                    forward_upsample(spec, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Logistic, LayerParams::None) => {
                    let data = current_input
                        .data
                        .iter()
                        .map(|v| activate(Activation::Logistic, *v))
                        .collect();
                    (Tensor::from_data(current_input.shape, data)?, None)
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => {
                    (current_input.clone(), None)
                }
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    (forward_softmax(current_input, *groups)?, None)
                }
                (LayerKind::Dropout { probability, dropblock }, LayerParams::None) => {
                    let (output, mask) = match dropblock {
                        Some(spec) => forward_dropblock(
                            current_input,
                            spec,
                            probability.unwrap_or(0.5),
                            training,
                            &mut rng_state,
                            self.train_iteration,
                            self.network.net.max_batches,
                        )?,
                        None => forward_dropout(
                            current_input,
                            probability.unwrap_or(0.5),
                            training,
                            &mut rng_state,
                        )?,
                    };
                    dropout_mask = mask;
                    (output, None)
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    forward_scale_channels(spec, current_input, source, layer.output_shape)?
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    forward_sam(spec, current_input, source, layer.output_shape)?
                }
                (LayerKind::Reorg(spec), LayerParams::None) => (
                    forward_reorg(spec.stride, spec.reverse, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    (crate::channel_slice::slice_channels(source, spec.start, spec.end)?, None)
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => (
                    crate::channel_shuffle::shuffle_channels(current_input, spec.groups, false)?,
                    None,
                ),
                (LayerKind::Rnn(spec), LayerParams::Rnn(recurrent)) => {
                    let (output, cache) =
                        forward_rnn_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Rnn(cache));
                    (output, None)
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(recurrent)) => {
                    let (output, cache) =
                        forward_lstm_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Lstm(cache));
                    (output, None)
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(recurrent)) => {
                    let (output, cache) =
                        forward_crnn_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Crnn(cache));
                    (output, None)
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

            // C++ `try_fix_nan`: convolution outputs are fixed for NaN/Inf.
            let output = if self.network.net.try_fix_nan
                && matches!(layer.kind, LayerKind::Convolutional(_))
            {
                Tensor {
                    shape: output.shape,
                    data: darknet_cuda::host_ops::sanitize_nan_inf(&output.data, NanInfMode::Fix),
                }
            } else {
                output
            };

            if matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_)) {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }

            layer_outputs.push(output);
            layer_activation_inputs.push(activation_input);
            dropout_masks.push(dropout_mask);
            crop_caches.push(crop_cache);
            recurrent_caches.push(recurrent_cache);
        }

        Ok(ForwardResult {
            layer_outputs,
            layer_activation_inputs,
            dropout_masks,
            crop_caches,
            recurrent_caches,
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
        if forward.dropout_masks.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward dropout mask count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.dropout_masks.len()
            )));
        }
        if forward.crop_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward crop cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.crop_caches.len()
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
            // Accumulate, do not replace, gradients here.  Several later
            // layers can feed the same earlier layer through route/shortcut
            // fan-in, matching C++'s additive `layer.delta` behavior.
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
                    let try_fix_nan = self.network.net.try_fix_nan;
                    // C++ `try_fix_nan`: clamp the output delta to [-1, 1]
                    // before the activation gradient, reset the input delta after.
                    let grad = if try_fix_nan {
                        Tensor {
                            shape: grad.shape,
                            data: darknet_cuda::host_ops::constrain(&grad.data, 1.0),
                        }
                    } else {
                        grad
                    };
                    let (mut input_grad, layer_param_gradients) = backward_convolutional(
                        spec,
                        conv,
                        current_input,
                        layer.output_shape,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                        workspace,
                    )?;
                    if try_fix_nan {
                        input_grad.data = darknet_cuda::host_ops::sanitize_nan_inf(
                            &input_grad.data,
                            NanInfMode::Reset,
                        );
                    }
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
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let delta = apply_output_activation_gradient(
                        &forward.layer_outputs[layer.index],
                        &grad,
                        &spec.activation,
                        "deconvolutional",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    // C++ also scales the weight gradient by 1/batch (and the optimizer divides again).
                    let weight_scale = if GradMode::from_env() == GradMode::Cpp {
                        1.0 / current_input.shape.batch as f32
                    } else {
                        1.0
                    };
                    let g = crate::deconvolutional::deconv_backward(
                        current_input,
                        &p.weights,
                        &delta,
                        spec.size,
                        spec.stride,
                        weight_scale,
                    );
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deconvolutional gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Convolutional(ConvolutionalParams {
                        biases: g.biases,
                        batch_norm: None,
                        weights: g.weights,
                    });
                    accumulate_previous_gradient(
                        layer.index,
                        &g.input,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    let input_grad = backward_maxpool_depth(current_input, &grad)?;
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
                        spec.zero_nonmax,
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
                        false,
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
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    let output = &forward.layer_outputs[layer.index];
                    let delta = apply_output_activation_gradient(
                        output,
                        &grad,
                        &spec.activation,
                        "shortcut",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    let sources = shortcut_sources(spec, layer.index, &forward.layer_outputs, current_input.shape.batch)?;
                    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
                    let g = crate::shortcut_weights::weighted_backward(
                        &delta.data,
                        &current_input.data,
                        &slices,
                        current_input.shape.batch,
                        current_input.shape.element_count(),
                        &params.weights,
                        spec.weights_normalization,
                        spec.weight_slots(current_input.shape.channels),
                        GradMode::from_env() == GradMode::Cpp,
                    );
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Shortcut(ShortcutParams { weights: g.weights });
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, g.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    for (raw_index, source_grad) in spec.from.iter().zip(g.sources) {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let shape = forward.layer_outputs[source_index].shape;
                        accumulate_tensor(
                            &mut layer_output_gradients[source_index],
                            &Tensor::from_data(shape, source_grad)?,
                        )?;
                    }
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
                        let source_shape = self
                            .network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?
                            .output_shape;
                        let source_gradient = shortcut_source_gradient(&accum_grad, source_shape)?;
                        accumulate_tensor(
                            &mut layer_output_gradients[source_index],
                            &source_gradient,
                        )?;
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
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing connected output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, layer_param_gradients) = backward_connected(
                        spec,
                        connected,
                        current_input,
                        layer.output_shape,
                        &grad,
                        output,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                        workspace,
                    )?;
                    accumulate_connected_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing connected gradient slot for layer #{}",
                                layer.index
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
                (LayerKind::Local(spec), LayerParams::Local(local)) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing local output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, layer_param_gradients) = backward_local(
                        spec,
                        local,
                        current_input,
                        layer.output_shape,
                        &grad,
                        output,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing local gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Local(layer_param_gradients);
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(batch_norm)) => {
                    let (input_grad, layer_param_gradients) = backward_batchnorm(
                        batch_norm,
                        current_input,
                        &grad,
                        forward.layer_activation_inputs[layer.index].is_some(),
                    )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing batch-norm gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::BatchNorm(layer_param_gradients);
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crop(_), LayerParams::None) => {
                    let cache = forward.crop_caches[layer.index].as_ref().ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing crop forward cache for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = backward_crop(cache, current_input.shape, &grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    let input_grad = backward_reorg(spec, &grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_shape = self
                        .network
                        .layers
                        .get(source_index)
                        .ok_or(Error::InvalidRoute { layer_index: layer.index, reference: spec.from })?
                        .output_shape;
                    let source_grad =
                        crate::channel_slice::slice_channels_backward(&grad, source_shape, spec.start)?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => {
                    let input_grad =
                        crate::channel_shuffle::shuffle_channels(&grad, spec.groups, true)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing softmax output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = if GradMode::from_env() == GradMode::Cpp {
                        // C++ passes dy straight through (cost layer folds the Jacobian in).
                        grad.clone()
                    } else {
                        backward_softmax(output, &grad, *groups)?
                    };
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dropout { .. }, LayerParams::None) => {
                    let input_grad =
                        backward_dropout(&grad, forward.dropout_masks[layer.index].as_ref())?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source =
                        forward
                            .layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing scale_channels output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (mut input_grad, source_grad) = backward_scale_channels(
                        spec,
                        current_input,
                        source,
                        output,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    if spec.scale_wh && GradMode::from_env() == GradMode::Cpp {
                        let channel_size = (output.shape.height * output.shape.width) as f32;
                        for v in &mut input_grad.data {
                            *v /= channel_size;
                        }
                    }
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source =
                        forward
                            .layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing sam output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, source_grad) = backward_sam(
                        spec,
                        current_input,
                        source,
                        output,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Logistic, LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => {
                    accumulate_previous_gradient(
                        layer.index,
                        &grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Rnn(spec), LayerParams::Rnn(params)) => {
                    let Some(RecurrentForwardCache::Rnn(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing RNN forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_rnn_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_rnn_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing RNN gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(params)) => {
                    let Some(RecurrentForwardCache::Lstm(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing LSTM forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_lstm_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_lstm_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing LSTM gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(params)) => {
                    let Some(RecurrentForwardCache::Crnn(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing CRNN forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_crnn_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_crnn_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing CRNN gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Unsupported(_), _) => {
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
            update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)?;
        }

        for layer in &self.network.layers {
            let LayerKind::Connected(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }

            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for connected batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let Some(LayerParams::Connected(params)) = self.params.get(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected params for layer #{}",
                    layer.index
                )));
            };
            let raw_output = raw_connected_output(params, current_input, layer.output_shape);
            let (mean, variance) = channel_mean_variance(&raw_output);
            let Some(LayerParams::Connected(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing mutable connected params for layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing batch-norm params for connected layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)?;
        }

        for layer in &self.network.layers {
            if !matches!(layer.kind, LayerKind::BatchNorm(_)) {
                continue;
            }
            if forward.layer_activation_inputs[layer.index].is_none() {
                continue;
            }
            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for standalone batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let (mean, variance) = batchnorm_channel_stats(current_input);
            let Some(LayerParams::BatchNorm(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing standalone batch-norm params for layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(params, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)?;
        }
        update_recurrent_batch_norm_running_stats(&self.network, &mut self.params, input, forward)?;
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
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(gradient)) => {
                    let Some(LayerTrainBuffers::BatchNorm(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing standalone batch-norm train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &gradient.biases)?;
                    accumulate_vec(&mut train.scale_updates, &gradient.scales)?;
                }
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
                (LayerKind::Deconvolutional(_), LayerParams::Convolutional(conv_grad)) => {
                    let Some(LayerTrainBuffers::Convolutional(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deconvolutional train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &conv_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &conv_grad.weights)?;
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
                (LayerKind::Local(_), LayerParams::Local(local_grad)) => {
                    let Some(LayerTrainBuffers::Local(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing local train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &local_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &local_grad.weights)?;
                }
                (LayerKind::Shortcut(_), LayerParams::Shortcut(gradient)) => {
                    let Some(LayerTrainBuffers::Shortcut(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.weight_updates, &gradient.weights)?;
                }
                (LayerKind::Rnn(_), LayerParams::Rnn(gradient)) => {
                    let Some(LayerTrainBuffers::Rnn(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing RNN train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_connected_train_buffer(&mut train.input, &gradient.input)?;
                    accumulate_connected_train_buffer(
                        &mut train.self_params,
                        &gradient.self_params,
                    )?;
                    accumulate_connected_train_buffer(&mut train.output, &gradient.output)?;
                }
                (LayerKind::Lstm(_), LayerParams::Lstm(gradient)) => {
                    let Some(LayerTrainBuffers::Lstm(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing LSTM train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_connected_train_buffer(&mut train.wf, &gradient.wf)?;
                    accumulate_connected_train_buffer(&mut train.wi, &gradient.wi)?;
                    accumulate_connected_train_buffer(&mut train.wg, &gradient.wg)?;
                    accumulate_connected_train_buffer(&mut train.wo, &gradient.wo)?;
                    accumulate_connected_train_buffer(&mut train.uf, &gradient.uf)?;
                    accumulate_connected_train_buffer(&mut train.ui, &gradient.ui)?;
                    accumulate_connected_train_buffer(&mut train.ug, &gradient.ug)?;
                    accumulate_connected_train_buffer(&mut train.uo, &gradient.uo)?;
                }
                (LayerKind::Crnn(_), LayerParams::Crnn(gradient)) => {
                    let Some(LayerTrainBuffers::Crnn(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing CRNN train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(&mut train.input, &gradient.input)?;
                    accumulate_convolutional_train_buffer(
                        &mut train.self_params,
                        &gradient.self_params,
                    )?;
                    accumulate_convolutional_train_buffer(&mut train.output, &gradient.output)?;
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
        return net.learning_rate * progress.powf(net.power);
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
        LrPolicy::Step => {
            net.learning_rate
                * net
                    .scale
                    .powi((batch_num / net.step.max(1)).try_into().unwrap_or(i32::MAX))
        }
        LrPolicy::Exp => net.gamma.powi(batch_num.try_into().unwrap_or(i32::MAX)) * lr,
        LrPolicy::Sig => {
            net.learning_rate / (1.0 + (net.gamma * (batch_num as f32 - net.step as f32)).exp())
        }
        LrPolicy::Sgdr => {
            let mut cycle_start = 0usize;
            let mut cycle_size = net.sgdr_cycle.max(1);
            while cycle_start.saturating_add(cycle_size) < batch_num {
                cycle_start = cycle_start.saturating_add(cycle_size);
                cycle_size = cycle_size.saturating_mul(net.sgdr_mult.max(1)).max(1);
            }
            let phase = (batch_num.saturating_sub(cycle_start)) as f32 / cycle_size as f32;
            net.learning_rate_min
                + 0.5
                    * (net.learning_rate - net.learning_rate_min)
                    * (1.0 + (std::f32::consts::PI * phase).cos())
        }
        LrPolicy::Poly => {
            if net.max_batches == 0 || batch_num >= net.max_batches {
                0.0
            } else {
                let progress = 1.0 - batch_num as f32 / net.max_batches as f32;
                net.learning_rate * progress.powf(net.power)
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
    /// Per-owner-layer f16 conv weights of `forward_compact_f16`, type-erased
    /// because `B::TensorF16` only exists for `B: F16TensorOpsBackend`.
    f16_conv_weights: RefCell<Vec<Option<std::rc::Rc<dyn std::any::Any>>>>,
    device_train_buffers: RefCell<Vec<Option<GpuConvolutionTrainBuffers<B::Tensor>>>>,
    cpu_fallback_train_buffers: RefCell<Vec<LayerTrainBuffers>>,
    /// Per-owner-layer Adam moments (device), created lazily on the first
    /// Adam update.
    adam_state: Vec<Option<GpuAdamLayerState<B::Tensor>>>,
    /// Adam moments for host-resident (connected) parameters.
    host_adam_state: RefCell<Vec<Option<HostAdamLayerState>>>,
    /// Number of Adam updates applied so far (the 1-based bias-correction `t`).
    adam_step: u32,
    /// See [`ExecutableNetwork::set_train_iteration`].
    train_iteration: usize,
    /// Host convolution weights / biases / BN gamma and rolling stats lag the
    /// device ones (deferred updates, see `sync_host_weights_from_device`).
    host_weights_stale: bool,
    /// Set by [`Self::update_weights_gpu_deferred`]: the backward pass keeps the
    /// BN rolling stats on the device instead of downloading them per step.
    defer_host_sync: bool,
}

/// First/second moments of one tensor.
type GpuMoments<T> = (T, T);

/// Adam moments of the host-resident parameters of one layer (connected).
#[derive(Clone, Debug, Default)]
struct HostAdamLayerState {
    weights: Option<(Vec<f32>, Vec<f32>)>,
    biases: Option<(Vec<f32>, Vec<f32>)>,
}

#[derive(Clone, Debug)]
struct GpuAdamLayerState<T> {
    weights: Option<GpuMoments<T>>,
    biases: Option<GpuMoments<T>>,
    scales: Option<GpuMoments<T>>,
}

impl<T> Default for GpuAdamLayerState<T> {
    fn default() -> Self {
        Self {
            weights: None,
            biases: None,
            scales: None,
        }
    }
}

/// One optimizer step shared by every tensor of an update (see
/// [`GpuExecutableNetwork::update_weights_gpu`]).
#[derive(Clone, Copy, Debug)]
enum GpuOptimizerStep {
    /// `lr` and `decay` are already loss-scale corrected.
    Sgd {
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    },
    Adam(AdamParams),
}

impl GpuOptimizerStep {
    /// The step with the learning rate multiplied by a per-layer `learning_rate` scale.
    fn with_lr_scale(self, scale: f32) -> Self {
        match self {
            Self::Sgd {
                lr,
                momentum,
                decay,
                batch,
            } => Self::Sgd {
                lr: lr * scale,
                momentum,
                decay,
                batch,
            },
            Self::Adam(mut params) => {
                params.lr *= scale;
                Self::Adam(params)
            }
        }
    }
}

/// One parameter tensor of a layer's optimizer step.
struct GpuOptimTarget<'a, T> {
    is_weight: bool,
    w: &'a T,
    u: &'a T,
    /// A pending (not yet accumulated) gradient, see `GpuPendingGrads`.
    grad: Option<&'a T>,
    moments: &'a mut Option<GpuMoments<T>>,
    shape: &'a [usize],
}

/// Sanitises, then applies one SGD/Adam step to a layer's parameter tensors
/// (weights, biases, BN scales) in one backend call (C++
/// `update_convolutional_layer_gpu`: `reset_nan_and_inf(updates)`,
/// `fix_nan_and_inf(weights)` for weights, then the optimizer).  Returns the
/// new `(weights, updates)` per target; Adam zeroes the updates like
/// `adam_update_gpu`.
fn gpu_optimizer_multi_step<B>(
    backend: &B,
    step: &GpuOptimizerStep,
    targets: Vec<GpuOptimTarget<'_, B::Tensor>>,
    config: ElementwiseLaunchConfig,
) -> Result<Vec<(B::Tensor, B::Tensor)>>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    match step {
        GpuOptimizerStep::Sgd {
            lr,
            momentum,
            decay,
            batch,
        } => {
            let items: Vec<_> = targets
                .iter()
                .map(|t| SgdUpdateItem {
                    weights: t.w,
                    updates: t.u,
                    grad: t.grad,
                    shape: t.shape,
                    fix_weights: t.is_weight,
                    decay: if t.is_weight { *decay } else { 0.0 },
                })
                .collect();
            backend
                .sanitized_sgd_update_multi_f32(&items, *lr, *momentum, *batch, config)
                .map_err(|err| backend_error("GPU SGD update", err))
        }
        GpuOptimizerStep::Adam(params) => {
            let mut targets = targets;
            for t in targets.iter_mut() {
                if t.moments.is_none() {
                    let count = t.shape.iter().product::<usize>();
                    let zeros = || {
                        backend
                            .zeros_f32(&[count])
                            .map_err(|err| backend_error("allocate GPU Adam moments", err))
                    };
                    *t.moments = Some((zeros()?, zeros()?));
                }
            }
            let items: Vec<_> = targets
                .iter()
                .map(|t| {
                    let (m, v) = t.moments.as_ref().expect("moments allocated above");
                    // C++ decays biases and scales too under coupled Adam;
                    // decoupled (AdamW) decays weights only.
                    let mut params = *params;
                    if !t.is_weight && params.decoupled_decay {
                        params.decay = 0.0;
                    }
                    AdamUpdateItem {
                        weights: t.w,
                        updates: t.u,
                        grad: t.grad,
                        m,
                        v,
                        shape: t.shape,
                        fix_weights: t.is_weight,
                        params,
                    }
                })
                .collect();
            let results = backend
                .sanitized_adam_update_multi_f32(&items, config)
                .map_err(|err| backend_error("GPU Adam update", err))?;
            drop(items);
            let mut out = Vec::with_capacity(results.len());
            for (t, (w, m, v, zeroed)) in targets.iter_mut().zip(results) {
                *t.moments = Some((m, v));
                out.push((w, zeroed));
            }
            Ok(out)
        }
    }
}

#[derive(Clone, Debug)]
struct GpuConvolutionLayerParams<T> {
    raw_weights: T,
    raw_biases: T,
    weights: T,
    biases: T,
    batch_norm_scales: Option<T>,
    /// Device-resident BN gamma and rolling stats (the source of truth during
    /// training; the host copies lag when `host_weights_stale`).
    bn: Option<GpuBatchNormState<T>>,
    /// The raw weights/BN stats changed since `weights`/`biases`/
    /// `batch_norm_scales` (the BN-folded inference copies) were built; they
    /// are re-folded on the device the next time something needs them.
    folded_stale: bool,
}

#[derive(Clone, Debug)]
struct GpuBatchNormState<T> {
    scales: T,
    rolling_mean: T,
    rolling_variance: T,
}

#[derive(Clone, Debug)]
struct GpuConvolutionTrainBuffers<T> {
    weight_updates: T,
    bias_updates: T,
    scale_updates: Option<T>,
    /// Gradient of the latest backward that is not yet added to the `*_updates`
    /// accumulators; the next optimizer step folds it in (`updates + grad`, the
    /// same addition the separate accumulate launch did).  Only set on the
    /// deferred training path (see `defer_host_sync`).
    pending: Option<GpuPendingGrads<T>>,
}

#[derive(Clone, Debug)]
struct GpuPendingGrads<T> {
    weights: T,
    biases: T,
    scales: Option<T>,
}

/// f16-storage inference (P6).  Training and every non-inference op stay on
/// the f32 path.
impl<B> GpuExecutableNetwork<B>
where
    B: F16TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// f16 twin of [`GpuExecutableNetwork::forward_compact`]: weights and
    /// activations are stored as binary16 on the device (f32 accumulation,
    /// bias and activation), the yolo heads are widened to f32 on download.
    /// Supports convolutional, maxpool, (local) avgpool, standalone batchnorm,
    /// route, shortcut (mismatched sources are cropped / zero-padded like
    /// Darknet), upsample (also `reverse`) and yolo-style pass-through layers;
    /// any other layer returns `UnsupportedLayer` (use `forward_compact`).
    pub fn forward_compact_f16(&self, input: &Tensor) -> Result<GpuForwardSummary> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        for layer in &self.network.layers {
            match &layer.kind {
                LayerKind::Convolutional(_)
                | LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::BatchNorm(_)
                | LayerKind::Route(_)
                | LayerKind::Upsample(_)
                | LayerKind::Yolo(_)
                | LayerKind::GaussianYolo(_) => {}
                LayerKind::Shortcut(spec) => {
                    if spec.from.is_empty() {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "shortcut without sources (f16 inference)".to_owned(),
                        });
                    }
                }
                other => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{} (f16 inference)", other.name()),
                    });
                }
            }
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f16(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload f16 input tensor", err))?;
        let mut gpu_layer_outputs: Vec<Option<B::TensorF16>> = vec![None; self.network.layers.len()];
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
                    let groups = spec.groups.max(1);
                    let cached_w16 = self.f16_conv_weights.borrow()[owner]
                        .as_ref()
                        .and_then(|shared| shared.downcast_ref::<B::TensorF16>())
                        .cloned();
                    let weights16 = match cached_w16 {
                        Some(w) => w,
                        None => {
                            let per_filter =
                                layer.input_shape.channels / groups * spec.size * spec.size;
                            let w = self
                                .backend
                                .conv_weights_f16(&cached.weights, spec.filters, per_filter)
                                .map_err(|err| backend_error("convert conv weights to f16", err))?;
                            self.f16_conv_weights.borrow_mut()[owner] = Some(std::rc::Rc::new(w.clone()));
                            w
                        }
                    };
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .conv2d_bias_act_f16(
                            current_input.expect("convolution requires current input"),
                            &weights16,
                            &cached.biases,
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
                            groups,
                            activation,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 convolution", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    if spec.depth {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "maxpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .maxpool_f16(
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
                        .map_err(|err| backend_error("f16 maxpool", err))?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f16(
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
                        .map_err(|err| backend_error("f16 local_avgpool", err))?
                }
                (LayerKind::AvgPool, LayerParams::None) => self
                    .backend
                    .avgpool_f16(
                        current_input.expect("avgpool requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 avgpool", err))?,
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => {
                    let shape = layer.input_shape;
                    let (scales, biases) = folded_batchnorm_inference_params(params, shape.channels)?;
                    let scales = self
                        .backend
                        .upload_f32(&[shape.channels], &scales)
                        .map_err(|err| backend_error("upload batchnorm inference scales", err))?;
                    let biases = self
                        .backend
                        .upload_f32(&[shape.channels], &biases)
                        .map_err(|err| backend_error("upload batchnorm inference biases", err))?;
                    self.backend
                        .channel_affine_act_f16(
                            current_input.expect("batchnorm requires current input"),
                            Some(&scales),
                            &biases,
                            shape.channels,
                            shape.width * shape.height,
                            F32ActivationKind::Linear,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 batchnorm inference", err))?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let mut accumulated = current_input
                        .expect("shortcut requires current input")
                        .clone();
                    let last = spec.from.len().saturating_sub(1);
                    for (n, raw_index) in spec.from.iter().enumerate() {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source = gpu_layer_outputs
                            .get(source_index)
                            .and_then(Option::as_ref)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?;
                        let source_layer = self.network.layers.get(source_index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing source layer #{source_index} for shortcut layer #{}",
                                layer.index
                            ))
                        })?;
                        if source_layer.output_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        // Darknet resizes a mismatched source flat per sample.
                        let fitted;
                        let source = if source_layer.output_shape != layer.output_shape {
                            fitted = self
                                .backend
                                .shortcut_fit_f16(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("f16 shortcut resize source", err))?;
                            &fitted
                        } else {
                            source
                        };
                        // The activation rides on the last add (one rounding).
                        let act = if n == last { activation } else { F32ActivationKind::Linear };
                        accumulated = self
                            .backend
                            .shortcut_add_act_f16(&accumulated, source, act, launch_config)
                            .map_err(|err| backend_error("f16 shortcut add", err))?;
                    }
                    accumulated
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
                        .route_concat_f16(
                            &route_inputs,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 route concat", err))?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f16(
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
                    .map_err(|err| backend_error("f16 upsample", err))?,
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None) => current_input
                    .expect("staging layer requires current input")
                    .clone(),
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
                Some(Tensor {
                    shape: layer.output_shape,
                    data: self
                        .backend
                        .download_f16(&output_gpu)
                        .map_err(|err| backend_error("download f16 layer output", err))?,
                })
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
        let f16_conv_weights = RefCell::new((0..network.layers.len()).map(|_| None).collect());
        let device_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        let cpu_fallback_train_buffers = RefCell::new(
            network
                .layers
                .iter()
                .map(default_layer_train_buffers)
                .collect(),
        );
        Self {
            backend,
            network,
            params,
            device_conv_params,
            f16_conv_weights,
            device_train_buffers,
            cpu_fallback_train_buffers,
            adam_state: Vec::new(),
            host_adam_state: RefCell::new(Vec::new()),
            adam_step: 0,
            train_iteration: 0,
            host_weights_stale: false,
            defer_host_sync: false,
        }
    }

    /// See [`ExecutableNetwork::set_train_iteration`].
    pub fn set_train_iteration(&mut self, iteration: usize) {
        self.train_iteration = iteration;
    }

    /// Adam updates applied so far (the `t` of the next update is this + 1).
    pub fn adam_step(&self) -> u32 {
        self.adam_step
    }

    /// Restores the Adam step counter (e.g. when resuming); moments are not
    /// part of the weights file and restart from zero.
    pub fn set_adam_step(&mut self, step: u32) {
        self.adam_step = step;
    }

    /// Loss scale applied to the yolo seed gradients before backward.
    pub fn loss_scale(&self) -> f32 {
        self.network.net.loss_scale
    }

    /// Update the loss scale (dynamic loss scaling).  Values are applied to the
    /// next backward and divided back out of the next optimizer step.
    pub fn set_loss_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale > 0.0 {
            self.network.net.loss_scale = scale;
        }
    }

    /// True if any device-resident convolution weight/bias/scale gradient is
    /// non-finite, used by dynamic loss scaling to detect overflow before the
    /// optimizer step.  Host-fallback layers are not inspected.
    pub fn gradients_non_finite(&self) -> Result<bool> {
        for entry in self.device_train_buffers.borrow().iter().flatten() {
            if self
                .backend
                .any_non_finite_f32(&entry.weight_updates)
                .map_err(|err| backend_error("gradient finiteness check", err))?
                || self
                    .backend
                    .any_non_finite_f32(&entry.bias_updates)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
            {
                return Ok(true);
            }
            if let Some(scales) = &entry.scale_updates
                && self
                    .backend
                    .any_non_finite_f32(scales)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
            {
                return Ok(true);
            }
            if let Some(pending) = &entry.pending {
                if self
                    .backend
                    .any_non_finite_f32(&pending.weights)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
                    || self
                        .backend
                        .any_non_finite_f32(&pending.biases)
                        .map_err(|err| backend_error("gradient finiteness check", err))?
                {
                    return Ok(true);
                }
                if let Some(scales) = &pending.scales
                    && self
                        .backend
                        .any_non_finite_f32(scales)
                        .map_err(|err| backend_error("gradient finiteness check", err))?
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Persist the Adam step counter and per-owner moment tensors to a sidecar
    /// file.  The `.weights` format stays C++-compatible and carries only
    /// parameters plus `seen`; resuming Adam from it would restart the moments,
    /// so training uses this sidecar to continue the optimizer exactly.
    pub fn save_optimizer_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = std::io::BufWriter::with_capacity(1 << 20, File::create(path)?);
        file.write_all(b"DNOPT1\0\0")?;
        file.write_all(&self.adam_step.to_le_bytes())?;
        let entries: Vec<usize> = self
            .adam_state
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.as_ref().map(|_| index))
            .collect();
        file.write_all(&(entries.len() as u64).to_le_bytes())?;
        for index in entries {
            file.write_all(&(index as u64).to_le_bytes())?;
            let state = self.adam_state[index]
                .as_ref()
                .expect("filtered to Some above");
            let (weight_shape, filter_shape) = self.adam_parameter_shapes(index)?;
            self.write_moments(&mut file, state.weights.as_ref(), &weight_shape)?;
            self.write_moments(&mut file, state.biases.as_ref(), &filter_shape)?;
            self.write_moments(&mut file, state.scales.as_ref(), &filter_shape)?;
        }
        file.flush()?;
        Ok(())
    }

    /// Restore state written by [`Self::save_optimizer_state`].  Layers absent
    /// from the sidecar keep `None` moments (Adam will initialise them on the
    /// next step), and the step counter is restored as stored.
    pub fn load_optimizer_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != b"DNOPT1\0\0" {
            return Err(Error::Weights("bad optimizer state magic".to_owned()));
        }
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let step = u32::from_le_bytes(buf4);
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        let mut state: Vec<Option<GpuAdamLayerState<B::Tensor>>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        for _ in 0..count {
            file.read_exact(&mut buf8)?;
            let index = u64::from_le_bytes(buf8) as usize;
            if index >= self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer state layer index out of range".to_owned(),
                ));
            }
            let (weight_shape, filter_shape) = self.adam_parameter_shapes(index)?;
            let weights = self.read_moments(&mut file, &weight_shape)?;
            let biases = self.read_moments(&mut file, &filter_shape)?;
            let scales = self.read_moments(&mut file, &filter_shape)?;
            state[index] = Some(GpuAdamLayerState {
                weights,
                biases,
                scales,
            });
        }
        self.adam_state = state;
        self.adam_step = step;
        Ok(())
    }

    fn adam_parameter_shapes(&self, index: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let layer = &self.network.layers[index];
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Err(Error::Weights(format!(
                "optimizer state for non-convolutional layer #{index}"
            )));
        };
        let grouped = layer.input_shape.channels / spec.groups.max(1);
        Ok((
            vec![spec.filters, grouped, spec.size, spec.size],
            vec![spec.filters],
        ))
    }

    fn write_moments(
        &self,
        file: &mut impl Write,
        moments: Option<&GpuMoments<B::Tensor>>,
        shape: &[usize],
    ) -> Result<()> {
        match moments {
            None => {
                file.write_all(&[0u8])?;
                Ok(())
            }
            Some((first, second)) => {
                file.write_all(&[1u8])?;
                for tensor in [first, second] {
                    let values = self
                        .backend
                        .download_f32(tensor)
                        .map_err(|err| backend_error("download Adam moment", err))?;
                    if values.len() != shape.iter().product::<usize>() {
                        return Err(Error::Weights(
                            "Adam moment length does not match its parameter shape".to_owned(),
                        ));
                    }
                    crate::weights::write_f32_slice(file, &values)?;
                }
                Ok(())
            }
        }
    }

    fn read_moments(
        &self,
        file: &mut File,
        shape: &[usize],
    ) -> Result<Option<GpuMoments<B::Tensor>>> {
        let mut flag = [0u8; 1];
        file.read_exact(&mut flag)?;
        if flag[0] == 0 {
            return Ok(None);
        }
        let len = shape.iter().product::<usize>();
        let mut first = vec![0f32; len];
        let mut second = vec![0f32; len];
        for buffer in [&mut first, &mut second] {
            let mut bytes = vec![0u8; len * 4];
            file.read_exact(&mut bytes)?;
            for (slot, chunk) in buffer.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        let first = self
            .backend
            .upload_f32(shape, &first)
            .map_err(|err| backend_error("upload Adam moment", err))?;
        let second = self
            .backend
            .upload_f32(shape, &second)
            .map_err(|err| backend_error("upload Adam moment", err))?;
        Ok(Some((first, second)))
    }

    /// The tensor backend (for benchmark statistics such as workspace size).
    pub fn backend(&self) -> &B {
        &self.backend
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
                        pending: None,
                    }));
                }
                _ => train_buffers.push(None),
            }
        }
        *self.device_train_buffers.get_mut() = train_buffers;
        Ok(())
    }

    pub fn zero_train_buffers_gpu(&mut self) -> Result<()> {
        *self.cpu_fallback_train_buffers.get_mut() = self
            .network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect();
        self.init_train_buffers_gpu()
    }

    /// One optimizer step over every device-resident convolution (and, for
    /// SGD, the host-fallback layers).  The optimizer follows `[net]`: SGD with
    /// momentum by default, Adam with `adam=1` (`B1`/`B2`/`eps`, coupled L2
    /// decay like C++, or AdamW with `decoupled_decay=1`; `lr`/`decay`/`batch`
    /// are the same arguments).  `loss_scale` makes the step independent of the
    /// gradient scale applied in `backward_accumulate_seeded_gpu`: SGD divides
    /// lr and multiplies decay, Adam scales eps and the coupled decay.  Updates are reset and weights fixed for
    /// NaN/Inf first (C++ `reset_nan_and_inf` / `fix_nan_and_inf`).
    pub fn update_weights_gpu(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    ) -> Result<()> {
        self.update_weights_gpu_impl(lr, momentum, decay, batch, true)
    }

    /// Like [`Self::update_weights_gpu`] but the convolution weights, biases and
    /// BN gamma / rolling statistics stay on the device (the following backward
    /// passes keep the rolling stats there too): `self.params[..]` conv
    /// parameters are stale until [`Self::sync_host_weights_from_device`] runs
    /// (call it before reading or saving `params`).  Falls back to the eager
    /// sync when the CPU backward fallback reads host weights.
    pub fn update_weights_gpu_deferred(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    ) -> Result<()> {
        let eager = self.requires_cpu_backward_fallback();
        self.update_weights_gpu_impl(lr, momentum, decay, batch, eager)
    }

    /// Refreshes the host convolution weights, biases and BN gamma / rolling
    /// stats after deferred updates.
    pub fn sync_host_weights_from_device(&mut self) -> Result<()> {
        if !self.host_weights_stale {
            return Ok(());
        }
        for layer_index in 0..self.network.layers.len() {
            let LayerKind::Convolutional(spec) = &self.network.layers[layer_index].kind else {
                continue;
            };
            if spec.share_index.is_some() {
                continue;
            }
            let cached = self.device_conv_params.borrow()[layer_index].clone();
            if let Some(cached) = cached {
                self.sync_convolution_params_from_device(layer_index, &cached)?;
            }
        }
        self.host_weights_stale = false;
        Ok(())
    }

    fn update_weights_gpu_impl(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        sync_host_weights: bool,
    ) -> Result<()> {
        let config = ElementwiseLaunchConfig::default();
        self.defer_host_sync = !sync_host_weights;
        let net = &self.network.net;
        let loss_scale = if net.loss_scale > 0.0 && net.loss_scale.is_finite() {
            net.loss_scale
        } else {
            1.0
        };
        let step = if net.adam {
            if self.params.iter().any(|params| {
                !matches!(
                    params,
                    LayerParams::None | LayerParams::Convolutional(_) | LayerParams::Connected(_)
                )
            }) {
                return Err(Error::InvalidConfig(
                    "adam=1 on the GPU path supports convolutional and connected layers only \
                     (standalone batchnorm / recurrent / local have no Adam state)"
                        .to_string(),
                ));
            }
            self.adam_step += 1;
            GpuOptimizerStep::Adam(AdamParams {
                // Adam is invariant to the gradient scale, so lr stays as is
                // (C++ divides it by loss_scale anyway, which shrinks the
                // step); eps and the coupled decay scale with the gradients.
                lr,
                beta1: net.adam_b1,
                beta2: net.adam_b2,
                eps: net.adam_eps * loss_scale,
                decay: if net.decoupled_decay {
                    decay
                } else {
                    decay * loss_scale
                },
                batch,
                step: self.adam_step,
                decoupled_decay: net.decoupled_decay,
            })
        } else {
            GpuOptimizerStep::Sgd {
                lr: lr / loss_scale,
                momentum,
                decay: decay * loss_scale,
                batch,
            }
        };
        if self.adam_state.len() != self.network.layers.len() {
            self.adam_state = (0..self.network.layers.len()).map(|_| None).collect();
        }
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        for layer_index in 0..self.network.layers.len() {
            let opts = self.network.layers[layer_index].train;
            // C++ `update_network_gpu` gating (`dont_update`, `burnin_update`,
            // `train_only_bn`): the whole layer is skipped, buffers untouched.
            if !opts.updates_at(iteration_num, burn_in) {
                continue;
            }
            let (owner, filters, weight_shape) = match &self.network.layers[layer_index].kind {
                LayerKind::Convolutional(spec) => {
                    let owner = spec.share_index.unwrap_or(layer_index);
                    if owner != layer_index {
                        continue;
                    }
                    (
                        owner,
                        spec.filters,
                        [
                            spec.filters,
                            self.network.layers[layer_index].input_shape.channels
                                / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                    )
                }
                _ => continue,
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

            let layer_step = step.with_lr_scale(opts.learning_rate_scale);
            // `grad_centr` (with BN) acts on the sanitised, accumulated weight updates.
            let mut weight_grad = train.weight_updates.clone();
            let pending = train.pending.clone();
            if opts.grad_centr && train.scale_updates.is_some() {
                let f_size = weight_shape[1] * weight_shape[2] * weight_shape[3];
                if let Some(p) = pending.as_ref() {
                    weight_grad = self
                        .backend
                        .saxpy_f32(1.0, &p.weights, &weight_grad, config)
                        .map_err(|err| backend_error("GPU gradient accumulation", err))?;
                }
                weight_grad = self
                    .backend
                    .sanitize_nan_inf_f32(&weight_grad, &weight_shape, NanInfMode::Reset, config)
                    .map_err(|err| backend_error("GPU update reset_nan_and_inf", err))?;
                weight_grad = self
                    .backend
                    .gradient_centralization_f32(&weight_grad, &weight_shape, f_size, config)
                    .map_err(|err| backend_error("GPU gradient centralization", err))?;
            }
            let weights_pending = if opts.grad_centr && train.scale_updates.is_some() {
                None
            } else {
                pending.as_ref().map(|p| &p.weights)
            };
            let mut moments = self.adam_state[owner].take().unwrap_or_default();
            let scales_input = match (cached.bn.as_ref(), train.scale_updates.as_ref()) {
                (Some(bn), Some(updates)) => Some((&bn.scales, updates)),
                (None, Some(_)) => {
                    return Err(Error::InvalidConfig(
                        "GPU batch-norm scale updates exist for a layer without device BN state"
                            .to_owned(),
                    ));
                }
                _ => None,
            };
            let filters_shape = [filters];
            let mut targets = vec![
                GpuOptimTarget {
                    is_weight: true,
                    w: &cached.raw_weights,
                    u: &weight_grad,
                    grad: weights_pending,
                    moments: &mut moments.weights,
                    shape: &weight_shape,
                },
                GpuOptimTarget {
                    is_weight: false,
                    w: &cached.raw_biases,
                    u: &train.bias_updates,
                    grad: pending.as_ref().map(|p| &p.biases),
                    moments: &mut moments.biases,
                    shape: &filters_shape,
                },
            ];
            if let Some((scales, updates)) = scales_input {
                targets.push(GpuOptimTarget {
                    is_weight: false,
                    w: scales,
                    u: updates,
                    grad: pending.as_ref().and_then(|p| p.scales.as_ref()),
                    moments: &mut moments.scales,
                    shape: &filters_shape,
                });
            }
            let mut stepped = gpu_optimizer_multi_step(&self.backend, &layer_step, targets, config)?
                .into_iter();
            let (mut raw_weights, weight_updates) = stepped.next().expect("weights result");
            if opts.clip > 0.0 {
                raw_weights = self
                    .backend
                    .constrain_f32(&raw_weights, &weight_shape, opts.clip, config)
                    .map_err(|err| backend_error("GPU weight clip", err))?;
            }
            let (raw_biases, bias_updates) = stepped.next().expect("biases result");
            let (new_scales, scale_updates) = match stepped.next() {
                Some((w, u)) => (Some(w), Some(u)),
                None => (None, None),
            };
            self.adam_state[owner] = Some(moments);

            let refreshed = self.refresh_device_convolution_params(
                owner,
                raw_weights,
                raw_biases,
                new_scales,
                weight_updates,
                bias_updates,
                scale_updates,
            )?;
            if sync_host_weights {
                self.sync_convolution_params_from_device(owner, &refreshed)?;
            } else {
                self.host_weights_stale = true;
            }
        }
        if sync_host_weights {
            self.host_weights_stale = false;
        }
        if let GpuOptimizerStep::Adam(adam) = step {
            self.update_cpu_fallback_connected_adam(&adam)?;
        } else {
            self.update_cpu_fallback_weights(lr / loss_scale, momentum, decay * loss_scale, batch);
        }
        Ok(())
    }

    /// Adam step for host-resident connected parameters (device convolutions are
    /// handled by the main loop).  Rejects any other host parameter type.
    fn update_cpu_fallback_connected_adam(&mut self, adam: &AdamParams) -> Result<()> {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        let mut moments_all = self.host_adam_state.borrow_mut();
        if moments_all.len() != self.network.layers.len() {
            *moments_all = (0..self.network.layers.len()).map(|_| None).collect();
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        for (index, layer) in self.network.layers.iter().enumerate() {
            if matches!(layer.kind, LayerKind::Convolutional(_)) {
                continue;
            }
            if !layer.train.updates_at(iteration_num, burn_in) {
                continue;
            }
            match (&mut self.params[index], &mut buffers[index]) {
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
                    let moments = moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam(
                        &mut params.weights,
                        &mut train.weight_updates,
                        &mut moments.weights,
                        adam,
                    )?;
                    apply_host_adam(
                        &mut params.biases,
                        &mut train.bias_updates,
                        &mut moments.biases,
                        adam,
                    )?;
                }
                (LayerParams::Convolutional(_), _) if matches!(layer.kind, LayerKind::Deconvolutional(_)) => {
                    return Err(Error::InvalidConfig(format!(
                        "adam=1 on the GPU path has no optimizer state for deconvolutional layer #{index}"
                    )));
                }
                (LayerParams::None, _) | (LayerParams::Convolutional(_), _) => {}
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "adam=1 on the GPU path has no optimizer state for layer #{} ({})",
                        index,
                        layer.kind.name()
                    )));
                }
            }
        }
        Ok(())
    }

    fn update_cpu_fallback_weights(&mut self, lr: f32, momentum: f32, decay: f32, batch: usize) {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        let train_buffers = self.cpu_fallback_train_buffers.get_mut();
        for (layer_index, ((params, train), layer)) in self
            .params
            .iter_mut()
            .zip(train_buffers.iter_mut())
            .zip(&self.network.layers)
            .enumerate()
        {
            if matches!(layer.kind, LayerKind::Convolutional(_)) {
                continue;
            }
            if !layer.train.updates_at(iteration_num, burn_in) {
                continue;
            }
            match (params, train) {
                (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
                    update_batchnorm_params(params, train, lr, momentum, batch);
                }
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
                    update_connected_params(params, train, lr, momentum, decay, batch);
                }
                (LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
                    update_connected_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
                    update_connected_params(
                        &mut params.wf,
                        &mut train.wf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wi,
                        &mut train.wi,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wg,
                        &mut train.wg,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wo,
                        &mut train.wo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uf,
                        &mut train.uf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ui,
                        &mut train.ui,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ug,
                        &mut train.ug,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uo,
                        &mut train.uo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
                    update_convolutional_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train))
                    if matches!(layer.kind, LayerKind::Deconvolutional(_)) =>
                {
                    sgd_update(&mut params.weights, &mut train.weight_updates, lr, momentum, decay, batch);
                    sgd_update(&mut params.biases, &mut train.bias_updates, lr, momentum, 0.0, batch);
                }
                (LayerParams::None, LayerTrainBuffers::None)
                | (LayerParams::Convolutional(_), LayerTrainBuffers::Convolutional(_)) => {}
                _ => {
                    let _ = layer_index;
                }
            }
        }
    }

    /// Batch-statistics batch norm of a raw (bias-free) convolution output.
    /// The shift is the layer's raw biases (the BN biases); the rolling EMA is
    /// computed on the device and applied to the host params after backward.
    fn forward_batchnorm_training_gpu(
        &self,
        raw_output: &B::Tensor,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
        batch_norm: &BatchNormParams,
        shape: TensorShape,
        update_rolling: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, GpuBatchNormCache<B::Tensor>)> {
        validate_batchnorm_params(batch_norm, shape.channels)?;
        let state = cached.bn.as_ref().ok_or_else(|| {
            Error::InvalidConfig("missing device batch-norm state for a BN layer".to_owned())
        })?;
        let (scales, rolling_mean, rolling_variance) = (
            state.scales.clone(),
            state.rolling_mean.clone(),
            state.rolling_variance.clone(),
        );
        let result = self
            .backend
            .batchnorm_train_forward_f32(
                raw_output,
                &scales,
                &cached.raw_biases,
                &rolling_mean,
                &rolling_variance,
                &tensor_backend_shape(shape),
                BATCH_NORM_ROLLING_KEEP,
                config,
            )
            .map_err(|err| backend_error("GPU batch-norm training forward", err))?;
        let cache = GpuBatchNormCache {
            scales,
            x_hat: result.x_hat,
            variance: result.variance,
            rolling: update_rolling.then_some((result.rolling_mean, result.rolling_variance)),
        };
        Ok((result.output, cache))
    }

    fn forward_batchnorm_inference_gpu(
        &self,
        input: &B::Tensor,
        shape: TensorShape,
        params: &BatchNormParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let (scales, biases) = folded_batchnorm_inference_params(params, shape.channels)?;
        let scales = self
            .backend
            .upload_f32(&[shape.channels], &scales)
            .map_err(|err| backend_error("upload batchnorm inference scales", err))?;
        let biases = self
            .backend
            .upload_f32(&[shape.channels], &biases)
            .map_err(|err| backend_error("upload batchnorm inference biases", err))?;
        let scaled = self
            .backend
            .channel_scale_f32(
                input,
                &scales,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference scale", err))?;
        self.backend
            .bias_add_f32(
                &scaled,
                &biases,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference bias", err))
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        Ok(self.forward_with_state(input)?.cpu_forward)
    }

    pub fn forward_with_state(&self, input: &Tensor) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_mode(input, false)
    }

    pub fn forward_training_with_state(
        &self,
        input: &Tensor,
    ) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_mode(input, true)
    }

    /// Training forward for the GPU backward path: only the yolo head outputs
    /// (what the host loss reads) are downloaded; every other
    /// `cpu_forward.layer_outputs` entry is a shape-only tensor with empty data
    /// and `layer_activation_inputs` is `None`.  Networks that need the CPU
    /// backward fallback still get the full host copies.
    pub fn forward_training_device(
        &self,
        input: &Tensor,
    ) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_impl(input, true, false)
    }

    fn forward_with_state_mode(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_impl(input, training, true)
    }

    fn forward_with_state_impl(
        &self,
        input: &Tensor,
        training: bool,
        host_copies: bool,
    ) -> Result<GpuForwardState<B::Tensor>> {
        let host_copies = host_copies || self.requires_cpu_backward_fallback();
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
        let mut recurrent_caches = Vec::with_capacity(self.network.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.network.layers.len());
        let mut crop_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_batch_norm_caches = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();
        let mut rng_state = 0x9e37_79b9_u64;
        // The CPU backward fallback differentiates with rolling statistics, so
        // batch statistics are only used when the GPU backward handles the net.
        let batch_norm_training = training && !self.requires_cpu_backward_fallback();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                &input_gpu
            } else {
                gpu_layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let mut activation_input_gpu = None;
            let mut dropout_mask = None;
            let mut crop_cache = None;
            let mut batch_norm_cache = None;
            let output_gpu = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = if batch_norm_training && conv.batch_norm.is_some() {
                        self.cached_convolution_params_raw(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    } else {
                        self.cached_convolution_params(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    };
                    // Batch-norm training convolves with the raw weights; the
                    // folded (rolling-stat) weights serve inference.
                    let weights = if batch_norm_training && conv.batch_norm.is_some() {
                        &cached.raw_weights
                    } else {
                        &cached.weights
                    };
                    let conv_output = self
                        .backend
                        .conv2d_f32(
                            current_input,
                            weights,
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
                    let biased = match conv.batch_norm.as_ref() {
                        Some(batch_norm) if batch_norm_training => {
                            let (normalized, cache) = self.forward_batchnorm_training_gpu(
                                &conv_output,
                                &cached,
                                batch_norm,
                                layer.output_shape,
                                spec.share_index.is_none(),
                                launch_config,
                            )?;
                            batch_norm_cache = Some(cache);
                            normalized
                        }
                        _ => self
                            .backend
                            .bias_add_f32(
                                &conv_output,
                                &cached.biases,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                launch_config,
                            )
                            .map_err(|err| backend_error("convolution bias add", err))?,
                    };
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    activation_input_gpu = Some(biased.clone());
                    self.backend
                        .activation_f32(&biased, activation, launch_config)
                        .map_err(|err| backend_error("convolution activation", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    if spec.depth {
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
                (LayerKind::Crop(spec), LayerParams::None) => {
                    if training {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "crop_gpu_training_augmentation".to_owned(),
                        });
                    }
                    let (cache, _) = plan_crop_forward(
                        spec,
                        layer.input_shape,
                        layer.output_shape,
                        false,
                        &mut rng_state,
                    )?;
                    let source_x = self
                        .backend
                        .upload_f32(&[cache.source_x.len()], &cache.source_x)
                        .map_err(|err| backend_error("upload crop source x", err))?;
                    let source_y = self
                        .backend
                        .upload_f32(&[cache.source_y.len()], &cache.source_y)
                        .map_err(|err| backend_error("upload crop source y", err))?;
                    let input_scales = self
                        .backend
                        .upload_f32(&[cache.input_scales.len()], &cache.input_scales)
                        .map_err(|err| backend_error("upload crop input scales", err))?;
                    let input_offsets = self
                        .backend
                        .upload_f32(&[cache.input_offsets.len()], &cache.input_offsets)
                        .map_err(|err| backend_error("upload crop input offsets", err))?;
                    let output = self
                        .backend
                        .crop_bilinear_f32(
                            current_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            &source_x,
                            &source_y,
                            &input_scales,
                            &input_offsets,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU crop", err))?;
                    crop_cache = Some(cache);
                    output
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => {
                    if training {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "batchnorm_gpu_training".to_owned(),
                        });
                    }
                    self.forward_batchnorm_inference_gpu(
                        current_input,
                        layer.input_shape,
                        params,
                        launch_config,
                    )?
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
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(_)) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("shortcut weights (GPU, from={:?})", spec.from),
                    });
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
                        if source_layer.output_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        // Darknet resizes a mismatched source flat per sample
                        // (crop / zero-pad), then adds it.
                        let fitted;
                        let source = if source_layer.output_shape != layer.output_shape {
                            fitted = self
                                .backend
                                .shortcut_fit_f32(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("shortcut resize source", err))?;
                            &fitted
                        } else {
                            source
                        };
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
                (LayerKind::Logistic, LayerParams::None) => self
                    .backend
                    .activation_f32(
                        current_input,
                        darknet_cuda::F32ActivationKind::Logistic,
                        launch_config,
                    )
                    .map_err(|err| backend_error("logistic", err))?,
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::Contrastive { .. }, LayerParams::None) => current_input.clone(),
                (LayerKind::Dropout { probability, dropblock }, LayerParams::None) => {
                    let probability = probability.unwrap_or(0.5);
                    check_dropout_probability(probability)?;
                    let block_plan = dropblock.as_ref().map(|spec| {
                        dropblock_plan(
                            spec,
                            probability,
                            layer.output_shape,
                            self.train_iteration,
                            self.network.net.max_batches,
                        )
                    });
                    if !training || probability == 0.0 || matches!(block_plan, Some((p, _)) if p <= 0.0)
                    {
                        current_input.clone()
                    } else {
                        let seed = advance_rng(&mut rng_state);
                        let (output, mask) = match block_plan {
                            Some((block_prob, block_size)) => self
                                .backend
                                .dropblock_f32(
                                    current_input,
                                    &tensor_backend_shape(layer.output_shape),
                                    block_prob,
                                    block_size,
                                    dropblock_seed(seed, self.train_iteration),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GPU dropblock", err))?,
                            None => self
                                .backend
                                .dropout_f32(
                                    current_input,
                                    &tensor_backend_shape(layer.output_shape),
                                    probability,
                                    seed,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GPU dropout", err))?,
                        };
                        dropout_mask = Some(
                            self.download_tensor(&mask, layer.output_shape)
                                .map_err(|err| backend_error("download dropout mask", err))?,
                        );
                        output
                    }
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    let inputs = layer.input_shape.element_count();
                    if connected.weights.len() != spec.output * inputs
                        || connected.biases.len() != spec.output
                    {
                        return Err(Error::InvalidConfig(format!(
                            "connected layer #{} expects {} weights and {} biases, got {} and {}",
                            layer.index,
                            spec.output * inputs,
                            spec.output,
                            connected.weights.len(),
                            connected.biases.len()
                        )));
                    }
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        // Inference batch norm is a per-output affine map, so
                        // fold it into the uploaded weights and biases.
                        let (weights, biases) = match connected.batch_norm.as_ref() {
                            Some(bn) => {
                                let mut weights = connected.weights.clone();
                                let mut biases = connected.biases.clone();
                                for (o, row) in weights.chunks_mut(inputs).enumerate() {
                                    let k = bn.scales[o]
                                        / (bn.rolling_variance[o] + BATCH_NORM_EPSILON).sqrt();
                                    row.iter_mut().for_each(|w| *w *= k);
                                    biases[o] -= bn.rolling_mean[o] * k;
                                }
                                (weights, biases)
                            }
                            None => (connected.weights.clone(), connected.biases.clone()),
                        };
                        let weights_gpu = self
                            .backend
                            .upload_f32(&[spec.output, inputs], &weights)
                            .map_err(|err| backend_error("upload connected weights", err))?;
                        let biases_gpu = self
                            .backend
                            .upload_f32(&[spec.output], &biases)
                            .map_err(|err| backend_error("upload connected biases", err))?;
                        let pre_activation = self
                            .backend
                            .connected_f32(
                                current_input,
                                &weights_gpu,
                                &biases_gpu,
                                layer.input_shape.batch,
                                inputs,
                                spec.output,
                                launch_config,
                            )
                            .map_err(|err| backend_error("connected", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("connected activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download connected input", err))?;
                        let (host_output, host_activation_input) = forward_connected_with_cache(
                            spec,
                            connected,
                            &host_input,
                            layer.output_shape,
                        );
                        if let Some(activation_input) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &activation_input.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload connected activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload connected output", err))?
                    }
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    if spec.stride == 0 {
                        return Err(Error::InvalidConfig(
                            "reorg stride cannot be zero".to_owned(),
                        ));
                    }
                    self.backend
                        .reorg_f32(
                            current_input,
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.stride,
                            spec.reverse,
                            launch_config,
                        )
                        .map_err(|err| backend_error("reorg", err))?
                }
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = gpu_layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    let source_channels = self.network.layers[source_index].output_shape.channels;
                    self.backend
                        .route_concat_f32(
                            &[source],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_channels],
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("channel_slice", err))?
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => self
                    .backend
                    .channel_shuffle_f32(
                        current_input,
                        &tensor_backend_shape(layer.input_shape),
                        spec.groups,
                        false,
                        launch_config,
                    )
                    .map_err(|err| backend_error("channel_shuffle", err))?,
                (LayerKind::Softmax { groups }, LayerParams::None) => self
                    .backend
                    .softmax_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        softmax_group_size(layer.input_shape, *groups)?,
                        launch_config,
                    )
                    .map_err(|err| backend_error("softmax", err))?,
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer =
                        self.network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let source_gpu =
                        gpu_layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        if source_layer.output_shape != layer.output_shape {
                            return Err(Error::InvalidConfig(format!(
                                "scale_channels source/output mismatch: {} vs {}",
                                source_layer.output_shape, layer.output_shape
                            )));
                        }
                        let pre_activation = self
                            .backend
                            .scale_channels_f32(
                                current_input,
                                source_gpu,
                                &tensor_backend_shape(layer.output_shape),
                                spec.scale_wh,
                                launch_config,
                            )
                            .map_err(|err| backend_error("scale_channels", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("scale_channels activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download scale_channels input", err))?;
                        let host_source = self
                            .download_tensor(source_gpu, source_layer.output_shape)
                            .map_err(|err| backend_error("download scale_channels source", err))?;
                        let (host_output, host_activation_input) = forward_scale_channels(
                            spec,
                            &host_input,
                            &host_source,
                            layer.output_shape,
                        )?;
                        if let Some(activation_input) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &activation_input.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload scale_channels activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload scale_channels output", err))?
                    }
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer =
                        self.network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let source_gpu =
                        gpu_layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download sam input", err))?;
                    let host_source = self
                        .download_tensor(source_gpu, source_layer.output_shape)
                        .map_err(|err| backend_error("download sam source", err))?;
                    let (host_output, host_activation_input) =
                        forward_sam(spec, &host_input, &host_source, layer.output_shape)?;
                    if let Some(activation_input) = host_activation_input {
                        activation_input_gpu = Some(
                            self.backend
                                .upload_f32(
                                    &tensor_backend_shape(layer.output_shape),
                                    &activation_input.data,
                                )
                                .map_err(|err| backend_error("upload sam activation cache", err))?,
                        );
                    }
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                        .map_err(|err| backend_error("upload sam output", err))?
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download deconvolutional input", err))?;
                    let pre = crate::deconvolutional::deconv_forward(
                        &host_input,
                        &p.weights,
                        &p.biases,
                        spec.size,
                        spec.stride,
                        layer.output_shape,
                    );
                    activation_input_gpu = Some(
                        self.backend
                            .upload_f32(&tensor_backend_shape(layer.output_shape), &pre.data)
                            .map_err(|err| backend_error("upload deconvolutional activation cache", err))?,
                    );
                    let activated: Vec<f32> =
                        pre.data.iter().map(|v| activate(spec.activation.clone(), *v)).collect();
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &activated)
                        .map_err(|err| backend_error("upload deconvolutional output", err))?
                }
                (LayerKind::Rnn(spec), LayerParams::Rnn(recurrent)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download RNN input", err))?;
                    let (host_output, cache) =
                        forward_rnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
                    recurrent_caches.push(Some(RecurrentForwardCache::Rnn(cache)));
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                        .map_err(|err| backend_error("upload RNN output", err))?
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(recurrent)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download LSTM input", err))?;
                    let (host_output, cache) =
                        forward_lstm_layer(spec, recurrent, &host_input, layer.output_shape)?;
                    recurrent_caches.push(Some(RecurrentForwardCache::Lstm(cache)));
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                        .map_err(|err| backend_error("upload LSTM output", err))?
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(recurrent)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download CRNN input", err))?;
                    let (host_output, cache) =
                        forward_crnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
                    recurrent_caches.push(Some(RecurrentForwardCache::Crnn(cache)));
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                        .map_err(|err| backend_error("upload CRNN output", err))?
                }
                (LayerKind::Unsupported(_), _) => {
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

            // C++ `try_fix_nan`: convolution outputs are fixed for NaN/Inf.
            let output_gpu = if self.network.net.try_fix_nan
                && matches!(layer.kind, LayerKind::Convolutional(_))
            {
                self.backend
                    .sanitize_nan_inf_f32(
                        &output_gpu,
                        &tensor_backend_shape(layer.output_shape),
                        NanInfMode::Fix,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU try_fix_nan output fix", err))?
            } else {
                output_gpu
            };
            let is_yolo = matches!(layer.kind, LayerKind::Yolo(_) | LayerKind::GaussianYolo(_));
            let (output, activation_input) = if host_copies || is_yolo {
                let output = self
                    .download_tensor(&output_gpu, layer.output_shape)
                    .map_err(|err| backend_error("download layer output", err))?;
                let activation_input = match activation_input_gpu.as_ref() {
                    Some(tensor) if host_copies => Some(
                        self.download_tensor(tensor, layer.output_shape)
                            .map_err(|err| backend_error("download activation cache", err))?,
                    ),
                    _ => None,
                };
                (output, activation_input)
            } else {
                (
                    Tensor {
                        shape: layer.output_shape,
                        data: Vec::new(),
                    },
                    None,
                )
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
            dropout_masks.push(dropout_mask);
            crop_caches.push(crop_cache);
            gpu_batch_norm_caches.push(batch_norm_cache);
            if recurrent_caches.len() < layer_outputs.len() {
                recurrent_caches.push(None);
            }
        }

        Ok(GpuForwardState {
            input_gpu,
            gpu_layer_outputs,
            gpu_layer_activation_inputs,
            gpu_batch_norm_caches,
            cpu_forward: ForwardResult {
                layer_outputs,
                layer_activation_inputs,
                dropout_masks,
                crop_caches,
                recurrent_caches,
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
        if self.network.layers.iter().any(|layer| {
            matches!(
                layer.kind,
                LayerKind::Connected(_)
                    | LayerKind::Deconvolutional(_)
                    | LayerKind::Reorg(_)
                    | LayerKind::ChannelSlice(_)
                    | LayerKind::ChannelShuffle(_)
                    | LayerKind::Softmax { .. }
                    | LayerKind::Logistic
                    | LayerKind::Regression(_)
                    | LayerKind::ScaleChannels(_)
                    | LayerKind::Sam(_)
                    | LayerKind::Rnn(_)
                    | LayerKind::Lstm(_)
                    | LayerKind::Crnn(_)
                    | LayerKind::Shortcut(crate::model::ShortcutSpec {
                        weights_type:
                            crate::model::ShortcutWeightsType::PerFeature | crate::model::ShortcutWeightsType::PerChannel,
                        ..
                    })
            )
        }) {
            let state = self.forward_with_state(input)?;
            return Ok(GpuForwardSummary {
                final_output: state.cpu_forward.final_output().cloned(),
                yolo_heads: state.cpu_forward.yolo_heads,
            });
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
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .bias_activation_f32(
                            &conv_output,
                            &cached.biases,
                            spec.filters,
                            layer.output_shape.width * layer.output_shape.height,
                            activation,
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution bias + activation", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    if spec.depth {
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
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => self
                    .forward_batchnorm_inference_gpu(
                        current_input.expect("batchnorm requires current input"),
                        layer.input_shape,
                        params,
                        launch_config,
                    )?,
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
                        if source_layer.output_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        // Darknet resizes a mismatched source flat per sample
                        // (crop / zero-pad), then adds it.
                        let fitted;
                        let source = if source_layer.output_shape != layer.output_shape {
                            fitted = self
                                .backend
                                .shortcut_fit_f32(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("shortcut resize source", err))?;
                            &fitted
                        } else {
                            source
                        };
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
                | (LayerKind::Detection(_), LayerParams::None)
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
                | (LayerKind::Rnn(_), _)
                | (LayerKind::Lstm(_), _)
                | (LayerKind::Crnn(_), _)
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

    /// Accumulates the layer gradients into the device train buffers.  With
    /// `loss_scale != 1` every seed is multiplied by it first (the matching
    /// division happens in `update_weights_gpu`).
    pub fn backward_accumulate_seeded_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        let loss_scale = self.network.net.loss_scale;
        if loss_scale == 1.0 || !(loss_scale > 0.0 && loss_scale.is_finite()) {
            return self.backward_accumulate_gpu_unscaled(forward, layer_output_gradient_seeds);
        }
        let scaled: Vec<Option<Tensor>> = layer_output_gradient_seeds
            .iter()
            .map(|seed| {
                seed.as_ref().map(|seed| Tensor {
                    shape: seed.shape,
                    data: seed.data.iter().map(|g| g * loss_scale).collect(),
                })
            })
            .collect();
        self.backward_accumulate_gpu_unscaled(forward, &scaled)
    }

    fn backward_accumulate_gpu_unscaled(
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
        if self.requires_cpu_backward_fallback() {
            return self
                .backward_accumulate_seeded_gpu_cpu_fallback(forward, layer_output_gradient_seeds);
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
                    // The batch-statistics BN backward only needs the raw tensors.
                    let cached = if forward.gpu_batch_norm_caches[layer.index].is_some() {
                        self.cached_convolution_params_raw(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    } else {
                        self.cached_convolution_params(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    };
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for convolutional layer #{}",
                                layer.index
                            ))
                        })?;
                    let try_fix_nan = self.network.net.try_fix_nan;
                    // C++ `try_fix_nan`: clamp the output delta to [-1, 1]
                    // before the activation gradient.
                    let grad = if try_fix_nan {
                        self.backend
                            .constrain_f32(&grad, &tensor_backend_shape(layer.output_shape), 1.0, config)
                            .map_err(|err| backend_error("GPU try_fix_nan delta clamp", err))?
                    } else {
                        grad
                    };
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| backend_error("GPU convolution activation backward", err))?;
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
                    let (bias_grad, scale_grad, grad_for_conv) = if let Some(cache) =
                        forward.gpu_batch_norm_caches[layer.index].as_ref()
                    {
                        // Batch-statistics batch norm: exact dx, dscale, dbias.
                        let (dx, dscale, dbias) = self
                            .backend
                            .batchnorm_train_backward_f32(
                                &delta,
                                &cache.x_hat,
                                &cache.variance,
                                &cache.scales,
                                &tensor_backend_shape(layer.output_shape),
                                config,
                            )
                            .map_err(|err| backend_error("GPU batch-norm backward", err))?;
                        (dbias, Some(dscale), dx)
                    } else {
                        let bias_grad = self
                            .backend
                            .bias_backward_f32(
                                &delta,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                config,
                            )
                            .map_err(|err| backend_error("GPU convolution bias backward", err))?;
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
                        (bias_grad, scale_grad, grad_for_conv)
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
                    let input_grad = if try_fix_nan {
                        self.backend
                            .sanitize_nan_inf_f32(
                                &input_grad,
                                &tensor_backend_shape(layer.input_shape),
                                NanInfMode::Reset,
                                config,
                            )
                            .map_err(|err| backend_error("GPU try_fix_nan input delta reset", err))?
                    } else {
                        input_grad
                    };
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
                (LayerKind::Crop(_), Some(LayerParams::None)) => {
                    let cache = forward.cpu_forward.crop_caches[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing crop forward cache for GPU layer #{}",
                                layer.index
                            ))
                        })?;
                    let source_x = self
                        .backend
                        .upload_f32(&[cache.source_x.len()], &cache.source_x)
                        .map_err(|err| backend_error("upload crop backward source x", err))?;
                    let source_y = self
                        .backend
                        .upload_f32(&[cache.source_y.len()], &cache.source_y)
                        .map_err(|err| backend_error("upload crop backward source y", err))?;
                    let input_scales = self
                        .backend
                        .upload_f32(&[cache.input_scales.len()], &cache.input_scales)
                        .map_err(|err| backend_error("upload crop backward input scales", err))?;
                    let input_grad = self
                        .backend
                        .crop_bilinear_backward_f32(
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            &source_x,
                            &source_y,
                            &input_scales,
                            config,
                        )
                        .map_err(|err| backend_error("GPU crop backward", err))?;
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
                        let source_shape = self
                            .network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?
                            .output_shape;
                        // Mismatched source: crop / zero-pad the gradient flat
                        // per sample, like the forward add.
                        let source_grad = if source_shape != layer.output_shape {
                            self.backend
                                .shortcut_fit_f32(
                                    &accum_grad,
                                    source_shape.batch,
                                    &tensor_backend_shape(source_shape),
                                    config,
                                )
                                .map_err(|err| backend_error("GPU shortcut source gradient", err))?
                        } else {
                            accum_grad.clone()
                        };
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            source_grad,
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
                (LayerKind::Softmax { groups }, Some(LayerParams::None)) => {
                    let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing GPU softmax output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = if GradMode::from_env() == GradMode::Cpp {
                        grad.clone()
                    } else {
                        self.backend
                            .softmax_backward_f32(
                                output,
                                &grad,
                                &tensor_backend_shape(layer.output_shape),
                                softmax_group_size(layer.input_shape, *groups)?,
                                config,
                            )
                            .map_err(|err| backend_error("GPU softmax backward", err))?
                    };
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelSlice(spec), Some(LayerParams::None)) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer = &self.network.layers[source_index];
                    let grads = self
                        .backend
                        .route_backward_f32(
                            &grad,
                            &[tensor_backend_shape(source_layer.output_shape).to_vec()],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_layer.output_shape.channels],
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| backend_error("GPU channel_slice backward", err))?;
                    for source_grad in grads {
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            source_grad,
                        )?;
                    }
                }
                (LayerKind::ChannelShuffle(spec), Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .channel_shuffle_f32(
                            &grad,
                            &tensor_backend_shape(layer.output_shape),
                            spec.groups,
                            true,
                            config,
                        )
                        .map_err(|err| backend_error("GPU channel_shuffle backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Reorg(spec), Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .reorg_backward_f32(
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.stride,
                            spec.reverse,
                            config,
                        )
                        .map_err(|err| backend_error("GPU reorg backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dropout { .. }, Some(LayerParams::None)) => {
                    let input_grad = match forward.cpu_forward.dropout_masks[layer.index].as_ref() {
                        Some(mask) => {
                            let mask_gpu = self
                                .backend
                                .upload_f32(&tensor_backend_shape(mask.shape), &mask.data)
                                .map_err(|err| backend_error("upload dropout mask", err))?;
                            self.backend
                                .dropout_backward_f32(
                                    &grad,
                                    &mask_gpu,
                                    &tensor_backend_shape(mask.shape),
                                    config,
                                )
                                .map_err(|err| backend_error("GPU dropout backward", err))?
                        }
                        None => grad,
                    };
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ScaleChannels(spec), Some(LayerParams::None)) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for scale_channels layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| {
                            backend_error("GPU scale_channels activation backward", err)
                        })?;
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        &forward.gpu_layer_outputs[layer.index - 1]
                    };
                    let source = forward.gpu_layer_outputs.get(source_index).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.from,
                        },
                    )?;
                    let (mut input_grad, source_grad) = self
                        .backend
                        .scale_channels_backward_f32(
                            &delta,
                            current_input,
                            source,
                            &tensor_backend_shape(layer.output_shape),
                            spec.scale_wh,
                            config,
                        )
                        .map_err(|err| backend_error("GPU scale_channels backward", err))?;
                    if spec.scale_wh && GradMode::from_env() == GradMode::Cpp {
                        let channel_size =
                            (layer.output_shape.height * layer.output_shape.width) as f32;
                        let zeros = self
                            .backend
                            .zeros_f32(&[
                                layer.output_shape.batch,
                                1,
                                layer.output_shape.height,
                                layer.output_shape.width,
                            ])
                            .map_err(|err| backend_error("GPU scale_channels C++ mode", err))?;
                        input_grad = self
                            .backend
                            .saxpy_f32(1.0 / channel_size, &input_grad, &zeros, config)
                            .map_err(|err| backend_error("GPU scale_channels C++ mode", err))?;
                    }
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                    self.accumulate_gpu_gradient(
                        &mut layer_output_gradients[source_index],
                        source_grad,
                    )?;
                }
                (
                    LayerKind::Connected(spec),
                    Some(LayerParams::Connected(connected)),
                ) if !spec.batch_normalize && spec.activation.to_cuda_f32_kind().is_some() => {
                    // The gradient chain stays on the device; only the connected
                    // GEMM itself runs on the host (`connected_backward_f32`).
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for connected layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| backend_error("GPU connected activation backward", err))?;
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU layer output for connected layer #{}",
                                layer.index
                            ))
                        })?
                    };
                    let inputs = layer.input_shape.element_count();
                    let weights_gpu = self
                        .backend
                        .upload_f32(&[spec.output, inputs], &connected.weights)
                        .map_err(|err| backend_error("upload connected weights", err))?;
                    let (input_grad, weight_grad, bias_grad) = self
                        .backend
                        .connected_backward_f32(
                            current_input,
                            &weights_gpu,
                            &delta,
                            layer.input_shape.batch,
                            inputs,
                            spec.output,
                            config,
                        )
                        .map_err(|err| backend_error("GPU connected backward", err))?;
                    self.accumulate_connected_host_gradients(
                        layer.index,
                        &weight_grad,
                        &bias_grad,
                        inputs,
                        spec.output,
                    )?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), Some(LayerParams::None))
                | (LayerKind::GaussianYolo(_), Some(LayerParams::None))
                | (LayerKind::Region(_), Some(LayerParams::None))
                | (LayerKind::Detection(_), Some(LayerParams::None))
                | (LayerKind::Cost { .. }, Some(LayerParams::None))
                | (LayerKind::Logistic, Some(LayerParams::None))
                | (LayerKind::Regression(_), Some(LayerParams::None))
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

    fn requires_cpu_backward_fallback(&self) -> bool {
        self.network.layers.iter().any(|layer| {
            match &layer.kind {
                // Recurrent layers need device sequence state; keep the whole
                // net on the CPU backward for them.
                LayerKind::Rnn(_) | LayerKind::Lstm(_) | LayerKind::Crnn(_) => return true,
                // Deconvolution is host-resident (no device kernel); run the
                // whole backward on the CPU.
                LayerKind::Deconvolutional(_) => return true,
                // Connected layers are handled on the GPU path with a host
                // round trip unless their activation or batch norm
                // (batch-statistics training) is unsupported.
                LayerKind::Connected(spec) => {
                    if spec.batch_normalize || spec.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                _ => {}
            }
            if let LayerKind::ScaleChannels(spec) = &layer.kind
                && spec.activation.to_cuda_f32_kind().is_none()
            {
                return true;
            }
            if let LayerKind::MaxPool(spec) = &layer.kind
                && spec.zero_nonmax
            {
                // No device kernel for the non-argmax zeroing yet.
                return true;
            }
            false
        })
    }

    fn backward_accumulate_seeded_gpu_cpu_fallback(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        if self.device_train_buffers.borrow().len() != self.network.layers.len()
            || self
                .device_train_buffers
                .borrow()
                .iter()
                .all(Option::is_none)
        {
            self.init_train_buffers_gpu()?;
        }
        let input = self
            .download_tensor(&forward.input_gpu, self.network.net.input_shape())
            .map_err(|err| backend_error("download CPU fallback input", err))?;
        let mut cpu = ExecutableNetwork::new(self.network.clone());
        cpu.params = self.params.clone();
        cpu.train_buffers = self.cpu_fallback_train_buffers.borrow().clone();
        for (layer, train) in self.network.layers.iter().zip(&mut cpu.train_buffers) {
            if matches!(layer.kind, LayerKind::Convolutional(_)) {
                *train = default_layer_train_buffers(layer);
            }
        }
        cpu.backward_accumulate_seeded(&input, &forward.cpu_forward, layer_output_gradient_seeds)?;

        let mut device_conv_gradients = Vec::new();
        for (layer, train) in self.network.layers.iter().zip(&cpu.train_buffers) {
            if !matches!(layer.kind, LayerKind::Convolutional(_)) {
                continue;
            }
            let owner = match &layer.kind {
                LayerKind::Convolutional(spec) => spec.share_index.unwrap_or(layer.index),
                _ => continue,
            };
            if owner != layer.index {
                continue;
            }
            let LayerTrainBuffers::Convolutional(train) = train else {
                continue;
            };
            let Some(LayerParams::Convolutional(_)) = cpu.params.get(owner) else {
                return Err(Error::InvalidConfig(format!(
                    "missing CPU fallback convolution params for layer #{owner}"
                )));
            };
            let weight_grad = self
                .backend
                .upload_f32(&conv_weight_shape(layer), &train.weight_updates)
                .map_err(|err| backend_error("upload CPU convolution weight gradient", err))?;
            let bias_grad = self
                .backend
                .upload_f32(&[train.bias_updates.len()], &train.bias_updates)
                .map_err(|err| backend_error("upload CPU convolution bias gradient", err))?;
            let scale_grad = train
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    self.backend
                        .upload_f32(&[batch_norm.scale_updates.len()], &batch_norm.scale_updates)
                        .map_err(|err| backend_error("upload CPU batch-norm gradient", err))
                })
                .transpose()?;
            device_conv_gradients.push((owner, weight_grad, bias_grad, scale_grad));
        }
        for (owner, weight_grad, bias_grad, scale_grad) in device_conv_gradients {
            self.accumulate_device_train_buffer(owner, weight_grad, bias_grad, scale_grad)?;
        }

        let fallback_buffers = self.cpu_fallback_train_buffers.get_mut();
        for (layer_index, (layer, train)) in self
            .network
            .layers
            .iter()
            .zip(cpu.train_buffers)
            .enumerate()
        {
            if !matches!(layer.kind, LayerKind::Convolutional(_)) {
                fallback_buffers[layer_index] = train;
            }
        }
        self.params = cpu.params;
        self.clear_device_conv_params();
        self.warm_device_conv_params()?;
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

    /// Accumulate a connected layer's host weight/bias gradient into its
    /// `LayerTrainBuffers` (the layer's parameters stay on the host; the
    /// optimizer's `update_cpu_fallback_weights` step consumes them).
    fn accumulate_connected_host_gradients(
        &self,
        layer_index: usize,
        weight_grad: &B::Tensor,
        bias_grad: &B::Tensor,
        inputs: usize,
        outputs: usize,
    ) -> Result<()> {
        let dw = self
            .backend
            .download_f32(weight_grad)
            .map_err(|err| backend_error("download connected weight gradient", err))?;
        let db = self
            .backend
            .download_f32(bias_grad)
            .map_err(|err| backend_error("download connected bias gradient", err))?;
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Connected(train)) = buffers.get_mut(layer_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing connected train buffers for layer #{layer_index}"
            )));
        };
        if train.weight_updates.len() != outputs * inputs || train.bias_updates.len() != outputs {
            return Err(Error::InvalidConfig(format!(
                "connected train buffer mismatch for layer #{layer_index}"
            )));
        }
        for (slot, value) in train.weight_updates.iter_mut().zip(dw) {
            *slot += value;
        }
        for (slot, value) in train.bias_updates.iter_mut().zip(db) {
            *slot += value;
        }
        Ok(())
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
        if self.defer_host_sync && scale_grad.is_some() == train.scale_updates.is_some() {
            // Keep this gradient pending: the optimizer step adds it to the
            // accumulators in its own launch.  An earlier pending gradient (a
            // subdivision) is added first, so the summation order is unchanged.
            let mut train = train;
            if let Some(pending) = train.pending.take() {
                let flushed = self.flush_pending_grads(&train, &pending)?;
                train.weight_updates = flushed.0;
                train.bias_updates = flushed.1;
                train.scale_updates = flushed.2;
            }
            train.pending = Some(GpuPendingGrads {
                weights: weight_grad,
                biases: bias_grad,
                scales: scale_grad,
            });
            self.device_train_buffers.borrow_mut()[owner] = Some(train);
            return Ok(());
        }
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
            pending: None,
        });
        Ok(())
    }

    /// `updates + pending` for every tensor (the accumulate launches the fused
    /// optimizer step skips).
    #[allow(clippy::type_complexity)]
    fn flush_pending_grads(
        &self,
        train: &GpuConvolutionTrainBuffers<B::Tensor>,
        pending: &GpuPendingGrads<B::Tensor>,
    ) -> Result<(B::Tensor, B::Tensor, Option<B::Tensor>)> {
        let config = ElementwiseLaunchConfig::default();
        let add = |grad: &B::Tensor, acc: &B::Tensor| {
            self.backend
                .saxpy_f32(1.0, grad, acc, config)
                .map_err(|err| backend_error("GPU gradient accumulation", err))
        };
        let scales = match (&pending.scales, &train.scale_updates) {
            (Some(g), Some(acc)) => Some(add(g, acc)?),
            (None, acc) => acc.clone(),
            (Some(_), None) => {
                return Err(Error::InvalidConfig(
                    "pending batch-norm scale gradient for a layer without scale updates"
                        .to_owned(),
                ));
            }
        };
        Ok((
            add(&pending.weights, &train.weight_updates)?,
            add(&pending.biases, &train.bias_updates)?,
            scales,
        ))
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
        update_batch_norm_running_stats(batch_norm, mean, variance, BATCH_NORM_ROLLING_KEEP)?;
        self.refresh_gpu_batch_norm_cache(owner)
    }

    /// Stores the device-computed rolling statistics of the batch-statistics
    /// forward.  The host copy is refreshed now unless host syncs are deferred.
    fn set_device_batch_norm_rolling_stats(
        &mut self,
        owner: usize,
        mean: B::Tensor,
        variance: B::Tensor,
    ) -> Result<()> {
        let mut cached = self.device_conv_params.borrow()[owner]
            .clone()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached GPU convolution params for layer #{owner}"
                ))
            })?;
        let Some(bn) = cached.bn.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing device batch-norm state for layer #{owner}"
            )));
        };
        bn.rolling_mean = mean;
        bn.rolling_variance = variance;
        cached.folded_stale = true;
        self.device_conv_params.borrow_mut()[owner] = Some(cached.clone());
        if self.defer_host_sync {
            self.host_weights_stale = true;
        } else {
            self.sync_convolution_params_from_device(owner, &cached)?;
        }
        Ok(())
    }

    /// Marks the folded device params stale after the host BN stats changed and
    /// re-uploads the host BN state to the device.
    fn refresh_gpu_batch_norm_cache(&mut self, owner: usize) -> Result<()> {
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
        let mut cached = cached;
        if let Some(LayerParams::Convolutional(ConvolutionalParams {
            batch_norm: Some(batch_norm),
            ..
        })) = self.params.get(owner)
        {
            cached.bn = Some(self.upload_batch_norm_state(batch_norm)?);
        }
        cached.folded_stale = true;
        self.device_conv_params.borrow_mut()[owner] = Some(cached);
        Ok(())
    }

    fn update_gpu_convolution_batch_norm_running_stats_from_forward(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
    ) -> Result<()> {
        let config = ElementwiseLaunchConfig::default();
        let mut batch_norm_stats = Vec::new();
        let mut batch_norm_rolling = Vec::new();
        for layer in &self.network.layers {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize || spec.share_index.is_some() {
                continue;
            }
            if let Some(cache) = forward.gpu_batch_norm_caches[layer.index].as_ref() {
                // Batch-statistics forward: the device already did the EMA.
                if let Some((mean, variance)) = cache.rolling.as_ref() {
                    batch_norm_rolling.push((layer.index, mean.clone(), variance.clone()));
                }
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
        for (owner, mean, variance) in batch_norm_rolling {
            self.set_device_batch_norm_rolling_stats(owner, mean, variance)?;
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
        // Callers replace the host params right before; nothing is stale then.
        self.host_weights_stale = false;
        for entry in self.device_conv_params.get_mut().iter_mut() {
            *entry = None;
        }
        for entry in self.f16_conv_weights.get_mut().iter_mut() {
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

    /// Cached device params with the BN-folded copies up to date.
    fn cached_convolution_params(
        &self,
        layer_index: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let cached =
            self.cached_convolution_params_raw(layer_index, input_channels, spec, params)?;
        if !cached.folded_stale {
            return Ok(cached);
        }
        let refolded = self.cached_convolution_upload(
            spec,
            input_channels,
            params,
            Some((cached.raw_weights, cached.raw_biases, cached.bn)),
        )?;
        self.device_conv_params.borrow_mut()[layer_index] = Some(refolded.clone());
        Ok(refolded)
    }

    /// Cached device params whose raw tensors are current but whose folded
    /// copies may be stale (training BN forward/backward only read the raw ones).
    fn cached_convolution_params_raw(
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
            folded_stale: false,
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
            bn: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| self.upload_batch_norm_state(batch_norm))
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

    /// Stores the updated raw tensors; the BN-folded copies are re-folded lazily.
    /// Nothing is copied to the host (see `sync_convolution_params_from_device`).
    fn refresh_device_convolution_params(
        &self,
        owner: usize,
        raw_weights: B::Tensor,
        raw_biases: B::Tensor,
        scales: Option<B::Tensor>,
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
        let params = resolve_convolutional_params(&self.network, &self.params, owner)?;
        let folds = spec.batch_normalize && params.batch_norm.is_some();
        let mut refreshed = self.device_conv_params.borrow()[owner]
            .clone()
            .ok_or_else(|| {
                Error::InvalidConfig(format!("missing cached device params for layer #{owner}"))
            })?;
        refreshed.raw_weights = raw_weights.clone();
        refreshed.raw_biases = raw_biases.clone();
        if let (Some(scales), Some(bn)) = (scales, refreshed.bn.as_mut()) {
            bn.scales = scales;
        }
        if folds {
            refreshed.folded_stale = true;
        } else {
            refreshed.weights = raw_weights;
            refreshed.biases = raw_biases;
        }
        self.device_conv_params.borrow_mut()[owner] = Some(refreshed.clone());
        self.device_train_buffers.borrow_mut()[owner] = Some(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
            pending: None,
        });
        Ok(refreshed)
    }

    fn upload_batch_norm_state(
        &self,
        batch_norm: &BatchNormParams,
    ) -> Result<GpuBatchNormState<B::Tensor>> {
        let upload = |name: &str, host: &[f32]| {
            self.backend
                .upload_f32(&[host.len()], host)
                .map_err(|err| backend_error(name, err))
        };
        Ok(GpuBatchNormState {
            scales: upload("upload batch-norm scales", &batch_norm.scales)?,
            rolling_mean: upload("upload batch-norm rolling mean", &batch_norm.rolling_mean)?,
            rolling_variance: upload(
                "upload batch-norm rolling variance",
                &batch_norm.rolling_variance,
            )?,
        })
    }

    fn cached_convolution_upload(
        &self,
        spec: &ConvolutionalSpec,
        input_channels: usize,
        params: &ConvolutionalParams,
        raw: Option<(B::Tensor, B::Tensor, Option<GpuBatchNormState<B::Tensor>>)>,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let weight_shape = [
            spec.filters,
            input_channels / spec.groups.max(1),
            spec.size,
            spec.size,
        ];
        // With device-resident raw weights and BN state the fold runs on the
        // device (`scale = gamma / sqrt(var + eps)`, `w * scale` per filter,
        // `bias - mean * scale`, the same arithmetic as
        // `fold_convolutional_params_for_inference`), so the host copies may be
        // stale.
        if let Some((raw_weights, raw_biases, bn)) = raw {
            let mut weights = raw_weights.clone();
            let mut biases = raw_biases.clone();
            let mut batch_norm_scales = None;
            if let Some(bn) = bn.as_ref() {
                let (scales, folded_biases) = self
                    .backend
                    .batchnorm_fold_f32(
                        &bn.scales,
                        &bn.rolling_mean,
                        &bn.rolling_variance,
                        &raw_biases,
                        BATCH_NORM_EPSILON,
                        ElementwiseLaunchConfig::default(),
                    )
                    .map_err(|err| backend_error("fold batch-norm parameters", err))?;
                if spec.batch_normalize {
                    weights = self
                        .backend
                        .channel_scale_f32(
                            &raw_weights,
                            &scales,
                            spec.filters,
                            weight_shape[1] * spec.size * spec.size,
                            ElementwiseLaunchConfig::default(),
                        )
                        .map_err(|err| backend_error("fold convolution weights", err))?;
                    biases = folded_biases;
                }
                batch_norm_scales = Some(scales);
            }
            return Ok(GpuConvolutionLayerParams {
                raw_weights,
                raw_biases,
                weights,
                biases,
                folded_stale: false,
                batch_norm_scales,
                bn,
            });
        }
        let (folded_biases, folded_weights) =
            fold_convolutional_params_for_inference(spec, params, input_channels)?;
        let raw_weights = self
            .backend
            .upload_f32(&weight_shape, &params.weights)
            .map_err(|err| backend_error("upload raw convolution weights", err))?;
        let raw_biases = self
            .backend
            .upload_f32(&[spec.filters], &params.biases)
            .map_err(|err| backend_error("upload raw convolution biases", err))?;
        let (weights, biases) = if spec.batch_normalize {
            (
                self.backend
                    .upload_f32(&weight_shape, &folded_weights)
                    .map_err(|err| backend_error("upload folded convolution weights", err))?,
                self.backend
                    .upload_f32(&[spec.filters], &folded_biases)
                    .map_err(|err| backend_error("upload folded convolution biases", err))?,
            )
        } else {
            (raw_weights.clone(), raw_biases.clone())
        };
        Ok(GpuConvolutionLayerParams {
            raw_weights,
            raw_biases,
            weights,
            biases,
            folded_stale: false,
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
            bn: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| self.upload_batch_norm_state(batch_norm))
                .transpose()?,
        })
    }

    /// Copies the device raw weights, biases and BN gamma / rolling stats of one
    /// owner convolution to the host params.
    fn sync_convolution_params_from_device(
        &mut self,
        owner: usize,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let download = |name: &str, tensor: &B::Tensor| {
            self.backend
                .download_f32(tensor)
                .map_err(|err| backend_error(name, err))
        };
        let raw_weights = download("download synced raw convolution weights", &cached.raw_weights)?;
        let raw_biases = download("download synced raw convolution biases", &cached.raw_biases)?;
        let bn = match cached.bn.as_ref() {
            Some(bn) => Some((
                download("download synced batch-norm scales", &bn.scales)?,
                download("download synced batch-norm rolling mean", &bn.rolling_mean)?,
                download("download synced batch-norm rolling variance", &bn.rolling_variance)?,
            )),
            None => None,
        };
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        params.weights = raw_weights;
        params.biases = raw_biases;
        if let (Some((scales, mean, variance)), Some(batch_norm)) = (bn, params.batch_norm.as_mut())
        {
            batch_norm.scales = scales;
            batch_norm.rolling_mean = mean;
            batch_norm.rolling_variance = variance;
        }
        Ok(())
    }
}

fn conv_weight_shape(layer: &crate::model::Layer) -> Vec<usize> {
    match &layer.kind {
        LayerKind::Convolutional(spec) => {
            let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
            vec![spec.filters, grouped_channels, spec.size, spec.size]
        }
        _ => Vec::new(),
    }
}

fn default_layer_params(layer: &crate::model::Layer) -> LayerParams {
    match &layer.kind {
        LayerKind::BatchNorm(_) => {
            LayerParams::BatchNorm(default_batchnorm_params(layer.input_shape.channels))
        }
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerParams::None;
            }
            LayerParams::Convolutional(default_convolutional_params(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ))
        }
        LayerKind::Deconvolutional(spec) => LayerParams::Convolutional(default_convolutional_params(
            layer.input_shape.channels,
            spec.filters,
            spec.size,
            1,
            false,
        )),
        LayerKind::Connected(spec) => LayerParams::Connected(default_connected_params(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::Local(spec) => LayerParams::Local(default_local_params(
            layer.input_shape.channels,
            layer.output_shape.width * layer.output_shape.height,
            spec.filters,
            spec.size,
        )),
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerParams::None,
            len => LayerParams::Shortcut(ShortcutParams {
                weights: vec![0.0; len],
            }),
        },
        LayerKind::Gru(spec) => LayerParams::Gru(default_gru_params(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::Rnn(spec) => LayerParams::Rnn(RnnParams {
            input: default_connected_params(
                layer.input_shape.element_count(),
                spec.hidden,
                spec.batch_normalize,
            ),
            self_params: default_connected_params(spec.hidden, spec.hidden, spec.batch_normalize),
            output: default_connected_params(spec.hidden, spec.output, spec.batch_normalize),
        }),
        LayerKind::Lstm(spec) => LayerParams::Lstm(LstmParams {
            wf: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wi: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wg: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wo: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            uf: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ui: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ug: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            uo: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Crnn(spec) => LayerParams::Crnn(CrnnParams {
            input: default_convolutional_params(
                layer.input_shape.channels,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            self_params: default_convolutional_params(
                spec.hidden,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            output: default_convolutional_params(
                spec.hidden,
                spec.output,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
        }),
        _ => LayerParams::None,
    }
}

fn default_layer_train_buffers(layer: &crate::model::Layer) -> LayerTrainBuffers {
    match &layer.kind {
        LayerKind::BatchNorm(_) => LayerTrainBuffers::BatchNorm(default_batchnorm_train_buffers(
            layer.input_shape.channels,
        )),
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerTrainBuffers::None;
            }
            LayerTrainBuffers::Convolutional(default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ))
        }
        LayerKind::Deconvolutional(spec) => LayerTrainBuffers::Convolutional(
            default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                1,
                false,
            ),
        ),
        LayerKind::Connected(spec) => {
            LayerTrainBuffers::Connected(default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ))
        }
        LayerKind::Local(spec) => LayerTrainBuffers::Local(default_local_train_buffers(
            layer.input_shape.channels,
            layer.output_shape.width * layer.output_shape.height,
            spec.filters,
            spec.size,
        )),
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerTrainBuffers::None,
            len => LayerTrainBuffers::Shortcut(ShortcutTrainBuffers {
                weight_updates: vec![0.0; len],
            }),
        },
        LayerKind::Gru(spec) => LayerTrainBuffers::Gru(default_gru_train_buffers(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::Rnn(spec) => LayerTrainBuffers::Rnn(RnnTrainBuffers {
            input: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.hidden,
                spec.batch_normalize,
            ),
            self_params: default_connected_train_buffers(
                spec.hidden,
                spec.hidden,
                spec.batch_normalize,
            ),
            output: default_connected_train_buffers(spec.hidden, spec.output, spec.batch_normalize),
        }),
        LayerKind::Lstm(spec) => LayerTrainBuffers::Lstm(LstmTrainBuffers {
            wf: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wi: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wg: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wo: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            uf: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ui: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ug: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            uo: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Crnn(spec) => LayerTrainBuffers::Crnn(CrnnTrainBuffers {
            input: default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            self_params: default_convolutional_train_buffers(
                spec.hidden,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            output: default_convolutional_train_buffers(
                spec.hidden,
                spec.output,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
        }),
        _ => LayerTrainBuffers::None,
    }
}

fn default_batchnorm_params(channels: usize) -> BatchNormParams {
    BatchNormParams {
        biases: vec![0.0; channels],
        scales: vec![1.0; channels],
        rolling_mean: vec![0.0; channels],
        rolling_variance: vec![1.0; channels],
    }
}

fn default_batchnorm_train_buffers(channels: usize) -> StandaloneBatchNormTrainBuffers {
    StandaloneBatchNormTrainBuffers {
        bias_updates: vec![0.0; channels],
        scale_updates: vec![0.0; channels],
    }
}

fn default_connected_params(
    inputs: usize,
    outputs: usize,
    batch_normalize: bool,
) -> ConnectedParams {
    ConnectedParams {
        biases: vec![0.0; outputs],
        batch_norm: batch_normalize.then(|| BatchNormParams {
            biases: vec![0.0; outputs],
            scales: vec![0.0; outputs],
            rolling_mean: vec![0.0; outputs],
            rolling_variance: vec![0.0; outputs],
        }),
        weights: vec![0.0; outputs * inputs],
    }
}

fn default_gru_params(inputs: usize, output: usize, batch_normalize: bool) -> GruParams {
    GruParams {
        input_z: default_connected_params(inputs, output, batch_normalize),
        state_z: default_connected_params(output, output, batch_normalize),
        input_r: default_connected_params(inputs, output, batch_normalize),
        state_r: default_connected_params(output, output, batch_normalize),
        input_h: default_connected_params(inputs, output, batch_normalize),
        state_h: default_connected_params(output, output, batch_normalize),
    }
}

fn default_convolutional_params(
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
) -> ConvolutionalParams {
    let groups = groups.max(1);
    ConvolutionalParams {
        biases: vec![0.0; filters],
        batch_norm: batch_normalize.then(|| BatchNormParams {
            biases: vec![0.0; filters],
            scales: vec![0.0; filters],
            rolling_mean: vec![0.0; filters],
            rolling_variance: vec![0.0; filters],
        }),
        weights: vec![0.0; filters * (input_channels / groups) * size * size],
    }
}

fn default_local_params(
    input_channels: usize,
    locations: usize,
    filters: usize,
    size: usize,
) -> LocalParams {
    LocalParams {
        biases: vec![0.0; locations * filters],
        weights: vec![0.0; locations * filters * input_channels * size * size],
    }
}

fn default_connected_train_buffers(
    inputs: usize,
    outputs: usize,
    batch_normalize: bool,
) -> ConnectedTrainBuffers {
    ConnectedTrainBuffers {
        bias_updates: vec![0.0; outputs],
        batch_norm: batch_normalize.then(|| BatchNormTrainBuffers {
            scale_updates: vec![0.0; outputs],
            mean_delta: vec![0.0; outputs],
            variance_delta: vec![0.0; outputs],
        }),
        weight_updates: vec![0.0; outputs * inputs],
    }
}

fn default_gru_train_buffers(
    inputs: usize,
    output: usize,
    batch_normalize: bool,
) -> GruTrainBuffers {
    GruTrainBuffers {
        input_z: default_connected_train_buffers(inputs, output, batch_normalize),
        state_z: default_connected_train_buffers(output, output, batch_normalize),
        input_r: default_connected_train_buffers(inputs, output, batch_normalize),
        state_r: default_connected_train_buffers(output, output, batch_normalize),
        input_h: default_connected_train_buffers(inputs, output, batch_normalize),
        state_h: default_connected_train_buffers(output, output, batch_normalize),
    }
}

fn default_convolutional_train_buffers(
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
) -> ConvolutionalTrainBuffers {
    let groups = groups.max(1);
    ConvolutionalTrainBuffers {
        bias_updates: vec![0.0; filters],
        batch_norm: batch_normalize.then(|| BatchNormTrainBuffers {
            scale_updates: vec![0.0; filters],
            mean_delta: vec![0.0; filters],
            variance_delta: vec![0.0; filters],
        }),
        weight_updates: vec![0.0; filters * (input_channels / groups) * size * size],
    }
}

fn default_local_train_buffers(
    input_channels: usize,
    locations: usize,
    filters: usize,
    size: usize,
) -> LocalTrainBuffers {
    LocalTrainBuffers {
        bias_updates: vec![0.0; locations * filters],
        weight_updates: vec![0.0; locations * filters * input_channels * size * size],
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
    // This is Darknet's update rule, not the convention used by optimizers
    // that receive a positive dL/dw tensor.  `updates` is accumulated from
    // Darknet deltas, so it already points downhill and is added to weights.
    let batch = batch.max(1) as f32;
    let step = lr / batch;
    for (weight, update) in weights.iter_mut().zip(updates.iter_mut()) {
        *update += -decay * batch * *weight;
        *weight += step * *update;
        *update *= momentum;
    }
}

fn update_batchnorm_params(
    params: &mut BatchNormParams,
    train: &mut StandaloneBatchNormTrainBuffers,
    lr: f32,
    momentum: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.biases,
        &mut train.bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    sgd_update(
        &mut params.scales,
        &mut train.scale_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
}

/// One Adam step on a host-resident parameter tensor, zeroing the accumulated
/// update like the device kernel (`adam_update_gpu` clears `weight_updates`).
fn apply_host_adam(
    weights: &mut [f32],
    updates: &mut [f32],
    moments: &mut Option<(Vec<f32>, Vec<f32>)>,
    params: &AdamParams,
) -> Result<()> {
    if weights.len() != updates.len() {
        return Err(Error::InvalidConfig(
            "host Adam weight/update length mismatch".to_owned(),
        ));
    }
    let (m, v) = moments
        .take()
        .unwrap_or_else(|| (vec![0.0; weights.len()], vec![0.0; weights.len()]));
    if m.len() != weights.len() || v.len() != weights.len() {
        return Err(Error::InvalidConfig(
            "host Adam moment length mismatch".to_owned(),
        ));
    }
    let (updated, m2, v2) = darknet_cuda::host_ops::adam_update(weights, updates, &m, &v, params);
    weights.copy_from_slice(&updated);
    for value in updates.iter_mut() {
        *value = 0.0;
    }
    *moments = Some((m2, v2));
    Ok(())
}

fn update_connected_params(
    params: &mut ConnectedParams,
    train: &mut ConnectedTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
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
    if let (Some(params_bn), Some(train_bn)) = (&mut params.batch_norm, &mut train.batch_norm) {
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

fn update_convolutional_params(
    params: &mut ConvolutionalParams,
    train: &mut ConvolutionalTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
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
    if let (Some(params_bn), Some(train_bn)) = (&mut params.batch_norm, &mut train.batch_norm) {
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

fn accumulate_connected_param_gradients(
    dst: &mut LayerParams,
    src: &ConnectedParams,
) -> Result<()> {
    let LayerParams::Connected(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected connected gradient destination".to_owned(),
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
                "connected batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn accumulate_rnn_param_gradients(dst: &mut LayerParams, src: &RnnParams) -> Result<()> {
    let LayerParams::Rnn(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected RNN gradient destination".to_owned(),
        ));
    };
    accumulate_connected_gradient(&mut dst.input, &src.input)?;
    accumulate_connected_gradient(&mut dst.self_params, &src.self_params)?;
    accumulate_connected_gradient(&mut dst.output, &src.output)
}

fn accumulate_lstm_param_gradients(dst: &mut LayerParams, src: &LstmParams) -> Result<()> {
    let LayerParams::Lstm(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected LSTM gradient destination".to_owned(),
        ));
    };
    accumulate_connected_gradient(&mut dst.wf, &src.wf)?;
    accumulate_connected_gradient(&mut dst.wi, &src.wi)?;
    accumulate_connected_gradient(&mut dst.wg, &src.wg)?;
    accumulate_connected_gradient(&mut dst.wo, &src.wo)?;
    accumulate_connected_gradient(&mut dst.uf, &src.uf)?;
    accumulate_connected_gradient(&mut dst.ui, &src.ui)?;
    accumulate_connected_gradient(&mut dst.ug, &src.ug)?;
    accumulate_connected_gradient(&mut dst.uo, &src.uo)
}

fn accumulate_crnn_param_gradients(dst: &mut LayerParams, src: &CrnnParams) -> Result<()> {
    let LayerParams::Crnn(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected CRNN gradient destination".to_owned(),
        ));
    };
    accumulate_convolutional_gradient(&mut dst.input, &src.input)?;
    accumulate_convolutional_gradient(&mut dst.self_params, &src.self_params)?;
    accumulate_convolutional_gradient(&mut dst.output, &src.output)
}

fn accumulate_connected_train_buffer(
    dst: &mut ConnectedTrainBuffers,
    src: &ConnectedParams,
) -> Result<()> {
    accumulate_vec(&mut dst.bias_updates, &src.biases)?;
    accumulate_vec(&mut dst.weight_updates, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scale_updates, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.mean_delta, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.variance_delta, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "connected recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn accumulate_convolutional_train_buffer(
    dst: &mut ConvolutionalTrainBuffers,
    src: &ConvolutionalParams,
) -> Result<()> {
    accumulate_vec(&mut dst.bias_updates, &src.biases)?;
    accumulate_vec(&mut dst.weight_updates, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scale_updates, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.mean_delta, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.variance_delta, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional recurrent batch norm gradient mismatch".to_owned(),
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
    if !spec.filters.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "convolution filters {} are not divisible by groups {}",
            spec.filters, groups
        )));
    }
    if !input_channels.is_multiple_of(groups) {
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

        for (filter, folded_bias) in folded_biases.iter_mut().enumerate() {
            let scale = batch_norm.scales[filter]
                / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
            *folded_bias -= batch_norm.rolling_mean[filter] * scale;
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
            LayerParams::BatchNorm(batch_norm) => {
                read_f32_into(&mut file, &mut batch_norm.biases)?;
                read_f32_into(&mut file, &mut batch_norm.scales)?;
                read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
            }
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
                read_connected_params(
                    &mut file,
                    connected,
                    layer.input_shape.element_count(),
                    connected.biases.len(),
                    transpose_connected,
                )?;
            }
            LayerParams::Local(local) => {
                read_f32_into(&mut file, &mut local.biases)?;
                read_f32_into(&mut file, &mut local.weights)?;
            }
            LayerParams::Shortcut(shortcut) => {
                read_f32_into(&mut file, &mut shortcut.weights)?;
            }
            LayerParams::Gru(gru) => {
                let inputs = layer.input_shape.element_count();
                let output = gru.input_z.biases.len();
                for (params, params_inputs) in [
                    (&mut gru.input_z, inputs),
                    (&mut gru.state_z, output),
                    (&mut gru.input_r, inputs),
                    (&mut gru.state_r, output),
                    (&mut gru.input_h, inputs),
                    (&mut gru.state_h, output),
                ] {
                    read_connected_params(
                        &mut file,
                        params,
                        params_inputs,
                        output,
                        transpose_connected,
                    )?;
                }
            }
            LayerParams::Rnn(recurrent) => {
                let inputs = layer.input_shape.element_count();
                let hidden = recurrent.input.biases.len();
                let output = recurrent.output.biases.len();
                read_connected_params(
                    &mut file,
                    &mut recurrent.input,
                    inputs,
                    hidden,
                    transpose_connected,
                )?;
                read_connected_params(
                    &mut file,
                    &mut recurrent.self_params,
                    hidden,
                    hidden,
                    transpose_connected,
                )?;
                read_connected_params(
                    &mut file,
                    &mut recurrent.output,
                    hidden,
                    output,
                    transpose_connected,
                )?;
            }
            LayerParams::Lstm(recurrent) => {
                let inputs = layer.input_shape.element_count();
                let output = recurrent.wf.biases.len();
                for (params, params_inputs) in [
                    (&mut recurrent.wf, output),
                    (&mut recurrent.wi, output),
                    (&mut recurrent.wg, output),
                    (&mut recurrent.wo, output),
                    (&mut recurrent.uf, inputs),
                    (&mut recurrent.ui, inputs),
                    (&mut recurrent.ug, inputs),
                    (&mut recurrent.uo, inputs),
                ] {
                    read_connected_params(
                        &mut file,
                        params,
                        params_inputs,
                        output,
                        transpose_connected,
                    )?;
                }
            }
            LayerParams::Crnn(recurrent) => {
                let input_channels = layer.input_shape.channels;
                let hidden = recurrent.self_params.biases.len();
                let output = recurrent.output.biases.len();
                let size = match &layer.kind {
                    LayerKind::Crnn(spec) => spec.size,
                    _ => 1,
                };
                let groups = match &layer.kind {
                    LayerKind::Crnn(spec) => spec.groups,
                    _ => 1,
                };
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.input,
                    input_channels,
                    hidden,
                    size,
                    groups,
                )?;
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.self_params,
                    hidden,
                    hidden,
                    size,
                    groups,
                )?;
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.output,
                    hidden,
                    output,
                    size,
                    groups,
                )?;
            }
        }
    }

    Ok(header)
}

fn tensor_backend_shape(shape: TensorShape) -> [usize; 4] {
    [shape.batch, shape.channels, shape.height, shape.width]
}

fn layer_uses_previous_input(kind: &LayerKind) -> bool {
    !matches!(kind, LayerKind::Route(_) | LayerKind::ChannelSlice(_))
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
            LayerKind::ChannelSlice(spec) => {
                let source_index = resolve_relative_layer(layer.index, spec.from)?;
                use_counts[source_index] += 1;
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
        LayerKind::ChannelSlice(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
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

fn read_connected_params(
    file: &mut File,
    params: &mut ConnectedParams,
    inputs: usize,
    outputs: usize,
    transpose: bool,
) -> Result<()> {
    if params.biases.len() != outputs || params.weights.len() != inputs * outputs {
        return Err(Error::InvalidConfig(
            "connected recurrent weight shape mismatch".to_owned(),
        ));
    }
    read_f32_into(file, &mut params.biases)?;
    read_f32_into(file, &mut params.weights)?;
    if transpose {
        let mut transposed = vec![0.0; params.weights.len()];
        for input in 0..inputs {
            for output in 0..outputs {
                transposed[output * inputs + input] = params.weights[input * outputs + output];
            }
        }
        params.weights = transposed;
    }
    if let Some(batch_norm) = params.batch_norm.as_mut() {
        read_f32_into(file, &mut batch_norm.scales)?;
        read_f32_into(file, &mut batch_norm.rolling_mean)?;
        read_f32_into(file, &mut batch_norm.rolling_variance)?;
    }
    Ok(())
}

fn read_convolutional_params(
    file: &mut File,
    params: &mut ConvolutionalParams,
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
) -> Result<()> {
    let expected = filters * (input_channels / groups.max(1)) * size * size;
    if params.biases.len() != filters || params.weights.len() != expected {
        return Err(Error::InvalidConfig(
            "convolutional recurrent weight shape mismatch".to_owned(),
        ));
    }
    read_f32_into(file, &mut params.biases)?;
    if let Some(batch_norm) = params.batch_norm.as_mut() {
        read_f32_into(file, &mut batch_norm.scales)?;
        read_f32_into(file, &mut batch_norm.rolling_mean)?;
        read_f32_into(file, &mut batch_norm.rolling_variance)?;
    }
    read_f32_into(file, &mut params.weights)?;
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

fn convolution_output_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    let stride = stride.max(1);
    (input + padding * 2).saturating_sub(size) / stride + 1
}

fn forward_convolutional_with_cache(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> (Tensor, Option<Tensor>) {
    // This direct loop is the correctness reference for convolution.  A
    // backend may replace it with im2col/GEMM or a CUDA kernel, but grouped
    // channel mapping, dilation, padding, and output order must remain equal.
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);

    // The cfg compatibility runner constructs networks before loading weights,
    // so the default parameter vectors are entirely zero.  For finite input
    // values every convolution product is consequently zero; retain the bias,
    // batch-normalization, and activation work while avoiding the otherwise
    // dominant kernel loop.  Non-finite inputs preserve IEEE multiplication
    // semantics through the reference loop below.
    if params.weights.iter().all(|weight| *weight == 0.0)
        && input.data.iter().all(|value| value.is_finite())
    {
        let channel_area = output_shape.width * output_shape.height;
        for b in 0..input.shape.batch {
            for filter in 0..spec.filters {
                let biased =
                    normalize_value(0.0, filter, &params.biases, params.batch_norm.as_ref());
                let activated = activate(spec.activation.clone(), biased);
                let start = b * output.batch_stride() + filter * channel_area;
                activation_input.data[start..start + channel_area].fill(biased);
                output.data[start..start + channel_area].fill(activated);
            }
        }
        return (output, Some(activation_input));
    }

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

fn update_connected_batch_norm_from_input(
    params: &mut ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<()> {
    let raw_output = raw_connected_output(params, input, output_shape);
    let Some(batch_norm) = params.batch_norm.as_mut() else {
        return Ok(());
    };
    let (mean, variance) = channel_mean_variance(&raw_output);
    update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)
}

fn update_convolutional_batch_norm_from_input(
    params: &mut ConvolutionalParams,
    spec: &ConvolutionalSpec,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<()> {
    let raw_output = raw_convolution_output(spec, params, input, output_shape);
    let Some(batch_norm) = params.batch_norm.as_mut() else {
        return Ok(());
    };
    let (mean, variance) = channel_mean_variance(&raw_output);
    update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)
}

fn update_recurrent_batch_norm_running_stats(
    network: &Network,
    params: &mut [LayerParams],
    input: &Tensor,
    forward: &ForwardResult,
) -> Result<()> {
    for layer in &network.layers {
        let current_input = if layer.index == 0 {
            input
        } else {
            forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous layer output for recurrent batch-norm stats at layer #{}",
                    layer.index
                ))
            })?
        };
        match (&layer.kind, params.get_mut(layer.index)) {
            (LayerKind::Rnn(spec), Some(LayerParams::Rnn(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
                let output_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Rnn(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing RNN cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.input,
                        &step_input,
                        hidden_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.self_params,
                        &cache.previous_states[step],
                        hidden_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.output,
                        &cache.states[step],
                        output_shape,
                    )?;
                }
            }
            (LayerKind::Lstm(spec), Some(LayerParams::Lstm(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Lstm(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing LSTM cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wf,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wi,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wg,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wo,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.uf,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.ui,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.ug,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.uo,
                        &step_input,
                        state_shape,
                    )?;
                }
            }
            (LayerKind::Crnn(spec), Some(LayerParams::Crnn(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let width = convolution_output_dim(
                    current_input.shape.width,
                    spec.padding,
                    spec.size,
                    spec.stride,
                );
                let height = convolution_output_dim(
                    current_input.shape.height,
                    spec.padding,
                    spec.size,
                    spec.stride,
                );
                let hidden_shape = TensorShape::new(width, height, spec.hidden, sequence_batch);
                let output_shape = TensorShape::new(width, height, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Crnn(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing CRNN cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
                let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
                let output_spec = recurrent_convolutional_spec(spec, spec.output);
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.input,
                        &input_spec,
                        &step_input,
                        hidden_shape,
                    )?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.self_params,
                        &self_spec,
                        &cache.previous_states[step],
                        hidden_shape,
                    )?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.output,
                        &output_spec,
                        &cache.states[step],
                        output_shape,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn forward_local_with_cache(
    spec: &LocalSpec,
    params: &LocalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    let locations = output_shape.width * output_shape.height;
    let kernel_values = input.shape.channels * spec.size * spec.size;
    let expected_biases = locations * spec.filters;
    let expected_weights = expected_biases * kernel_values;
    if params.biases.len() != expected_biases || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "local parameter shape mismatch: expected {expected_biases} biases and {expected_weights} weights, got {} and {}",
            params.biases.len(),
            params.weights.len()
        )));
    }

    let mut activation_input = Tensor::zeros(output_shape);
    // As with convolution, unweighted cfg construction has zero local kernels.
    // Preserve the per-location bias and activation exactly, while avoiding
    // spatial kernel traversal for finite inputs.  Keep the reference path for
    // non-finite inputs so IEEE zero-times-NaN/Inf behavior is unchanged.
    if params.weights.iter().all(|weight| *weight == 0.0)
        && input.data.iter().all(|value| value.is_finite())
    {
        for batch in 0..input.shape.batch {
            for out_y in 0..output_shape.height {
                for out_x in 0..output_shape.width {
                    let location = out_y * output_shape.width + out_x;
                    for filter in 0..spec.filters {
                        let parameter = location * spec.filters + filter;
                        activation_input.set(batch, filter, out_y, out_x, params.biases[parameter]);
                    }
                }
            }
        }
        let mut output = activation_input.clone();
        for value in &mut output.data {
            *value = activate(spec.activation.clone(), *value);
        }
        return Ok((output, Some(activation_input)));
    }

    for batch in 0..input.shape.batch {
        for out_y in 0..output_shape.height {
            for out_x in 0..output_shape.width {
                let location = out_y * output_shape.width + out_x;
                for filter in 0..spec.filters {
                    let parameter = location * spec.filters + filter;
                    let weight_base = parameter * kernel_values;
                    let mut value = params.biases[parameter];
                    for channel in 0..input.shape.channels {
                        for kernel_y in 0..spec.size {
                            for kernel_x in 0..spec.size {
                                let source_x = out_x * spec.stride_x + kernel_x;
                                let source_y = out_y * spec.stride_y + kernel_y;
                                let input_x = source_x as isize - spec.padding as isize;
                                let input_y = source_y as isize - spec.padding as isize;
                                if input_x < 0
                                    || input_y < 0
                                    || input_x as usize >= input.shape.width
                                    || input_y as usize >= input.shape.height
                                {
                                    continue;
                                }
                                let kernel_index =
                                    (channel * spec.size + kernel_y) * spec.size + kernel_x;
                                value += params.weights[weight_base + kernel_index]
                                    * input.get(batch, channel, input_y as usize, input_x as usize);
                            }
                        }
                    }
                    activation_input.set(batch, filter, out_y, out_x, value);
                }
            }
        }
    }
    let mut output = activation_input.clone();
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok((output, Some(activation_input)))
}

fn backward_local(
    spec: &LocalSpec,
    params: &LocalParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    output: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, LocalParams)> {
    if output_gradient.shape != output_shape || output.shape != output_shape {
        return Err(Error::InvalidConfig(
            "local backward output shape mismatch".to_owned(),
        ));
    }
    let locations = output_shape.width * output_shape.height;
    let kernel_values = input.shape.channels * spec.size * spec.size;
    let expected_biases = locations * spec.filters;
    let expected_weights = expected_biases * kernel_values;
    if params.biases.len() != expected_biases || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(
            "local backward parameter shape mismatch".to_owned(),
        ));
    }
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "local",
        activation_input,
    )?;
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = LocalParams {
        biases: vec![0.0; expected_biases],
        weights: vec![0.0; expected_weights],
    };
    for batch in 0..input.shape.batch {
        for out_y in 0..output_shape.height {
            for out_x in 0..output_shape.width {
                let location = out_y * output_shape.width + out_x;
                for filter in 0..spec.filters {
                    let parameter = location * spec.filters + filter;
                    let weight_base = parameter * kernel_values;
                    let delta_value = delta.get(batch, filter, out_y, out_x);
                    gradients.biases[parameter] += delta_value;
                    for channel in 0..input.shape.channels {
                        for kernel_y in 0..spec.size {
                            for kernel_x in 0..spec.size {
                                let source_x = out_x * spec.stride_x + kernel_x;
                                let source_y = out_y * spec.stride_y + kernel_y;
                                let input_x = source_x as isize - spec.padding as isize;
                                let input_y = source_y as isize - spec.padding as isize;
                                if input_x < 0
                                    || input_y < 0
                                    || input_x as usize >= input.shape.width
                                    || input_y as usize >= input.shape.height
                                {
                                    continue;
                                }
                                let kernel_index =
                                    (channel * spec.size + kernel_y) * spec.size + kernel_x;
                                let weight_index = weight_base + kernel_index;
                                let input_value =
                                    input.get(batch, channel, input_y as usize, input_x as usize);
                                gradients.weights[weight_index] += input_value * delta_value;
                                let current = input_gradient.get(
                                    batch,
                                    channel,
                                    input_y as usize,
                                    input_x as usize,
                                );
                                input_gradient.set(
                                    batch,
                                    channel,
                                    input_y as usize,
                                    input_x as usize,
                                    current + params.weights[weight_index] * delta_value,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((input_gradient, gradients))
}

fn forward_connected_with_cache(
    spec: &ConnectedSpec,
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> (Tensor, Option<Tensor>) {
    // Darknet stores connected weights as [output][input].  Keeping this
    // layout makes binary weight loading and the transposed backward GEMM
    // match connected_layer.cpp without a hidden transpose.
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
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

            let normalized =
                normalize_value(acc, out_idx, &params.biases, params.batch_norm.as_ref());
            activation_input.data[b * spec.output + out_idx] = normalized;
            let value = activate(spec.activation.clone(), normalized);
            output.data[b * spec.output + out_idx] = value;
        }
    }

    (output, Some(activation_input))
}

fn batchnorm_channel_stats(input: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let sample_count = (input.shape.batch * input.shape.width * input.shape.height).max(1);
    let mut mean = vec![0.0; input.shape.channels];
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    mean[channel] += input.get(b, channel, y, x);
                }
            }
        }
    }
    for value in &mut mean {
        *value /= sample_count as f32;
    }

    let mut variance = vec![0.0; input.shape.channels];
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let diff = input.get(b, channel, y, x) - mean[channel];
                    variance[channel] += diff * diff;
                }
            }
        }
    }
    for value in &mut variance {
        *value /= sample_count as f32;
    }
    (mean, variance)
}

fn validate_batchnorm_params(params: &BatchNormParams, channels: usize) -> Result<()> {
    if params.biases.len() != channels
        || params.scales.len() != channels
        || params.rolling_mean.len() != channels
        || params.rolling_variance.len() != channels
    {
        return Err(Error::InvalidConfig(format!(
            "batchnorm parameter length mismatch: expected {channels} channels, got biases {}, scales {}, rolling mean {}, rolling variance {}",
            params.biases.len(),
            params.scales.len(),
            params.rolling_mean.len(),
            params.rolling_variance.len()
        )));
    }
    Ok(())
}

fn folded_batchnorm_inference_params(
    params: &BatchNormParams,
    channels: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    validate_batchnorm_params(params, channels)?;
    let mut scales = Vec::with_capacity(channels);
    let mut biases = Vec::with_capacity(channels);
    for channel in 0..channels {
        let scale =
            params.scales[channel] / (params.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt();
        scales.push(scale);
        biases.push(params.biases[channel] - params.rolling_mean[channel] * scale);
    }
    Ok((scales, biases))
}

fn forward_batchnorm_with_cache(
    params: &BatchNormParams,
    input: &Tensor,
    training: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    validate_batchnorm_params(params, input.shape.channels)?;
    let (mean, variance) = if training {
        batchnorm_channel_stats(input)
    } else {
        (params.rolling_mean.clone(), params.rolling_variance.clone())
    };
    let mut output = Tensor::zeros(input.shape);
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            let inv_std = 1.0 / (variance[channel] + BATCH_NORM_EPSILON).sqrt();
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let normalized = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    output.set(
                        b,
                        channel,
                        y,
                        x,
                        normalized * params.scales[channel] + params.biases[channel],
                    );
                }
            }
        }
    }
    // The cache is a training marker. Backward recomputes the batch statistics
    // from the saved graph input, while inference uses rolling statistics.
    Ok((output, training.then(|| input.clone())))
}

fn backward_batchnorm(
    params: &BatchNormParams,
    input: &Tensor,
    output_gradient: &Tensor,
    training: bool,
) -> Result<(Tensor, BatchNormParams)> {
    if output_gradient.shape != input.shape {
        return Err(Error::InvalidConfig(format!(
            "batchnorm backward gradient shape mismatch: expected {}, got {}",
            input.shape, output_gradient.shape
        )));
    }
    validate_batchnorm_params(params, input.shape.channels)?;
    let (mean, variance) = if training {
        batchnorm_channel_stats(input)
    } else {
        (params.rolling_mean.clone(), params.rolling_variance.clone())
    };
    let count = (input.shape.batch * input.shape.width * input.shape.height).max(1) as f32;
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradient = BatchNormParams {
        biases: vec![0.0; input.shape.channels],
        scales: vec![0.0; input.shape.channels],
        rolling_mean: vec![0.0; input.shape.channels],
        rolling_variance: vec![0.0; input.shape.channels],
    };

    for channel in 0..input.shape.channels {
        let inv_std = 1.0 / (variance[channel] + BATCH_NORM_EPSILON).sqrt();
        let mut sum_gradient = 0.0;
        let mut sum_gradient_xhat = 0.0;
        for b in 0..input.shape.batch {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let dy = output_gradient.get(b, channel, y, x);
                    let xhat = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    gradient.biases[channel] += dy;
                    gradient.scales[channel] += dy * xhat;
                    sum_gradient += dy;
                    sum_gradient_xhat += dy * xhat;
                }
            }
        }

        for b in 0..input.shape.batch {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let dy = output_gradient.get(b, channel, y, x);
                    let xhat = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    let dx = if training {
                        params.scales[channel] * inv_std / count
                            * (count * dy - sum_gradient - xhat * sum_gradient_xhat)
                    } else {
                        params.scales[channel] * inv_std * dy
                    };
                    input_gradient.set(b, channel, y, x, dx);
                }
            }
        }
    }
    Ok((input_gradient, gradient))
}

fn recurrent_sequence_batch(batch: usize, steps: usize) -> Result<usize> {
    if steps == 0 {
        return Err(Error::InvalidConfig(
            "recurrent layer steps cannot be zero".to_owned(),
        ));
    }
    if batch == 0 || !batch.is_multiple_of(steps) {
        return Err(Error::InvalidConfig(format!(
            "recurrent layer batch {} must be a non-zero multiple of steps {}",
            batch, steps
        )));
    }
    Ok(batch / steps)
}

fn sequence_slice(
    input: &Tensor,
    step: usize,
    sequence_batch: usize,
    step_shape: TensorShape,
) -> Result<Tensor> {
    if step_shape.batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "recurrent step shape batch mismatch".to_owned(),
        ));
    }
    let step_len = sequence_batch * step_shape.element_count();
    let start = step
        .checked_mul(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent step offset overflow".to_owned()))?;
    let end = start
        .checked_add(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent step end overflow".to_owned()))?;
    if end > input.data.len() {
        return Err(Error::InvalidConfig(
            "recurrent input does not contain all requested steps".to_owned(),
        ));
    }
    Tensor::from_data(step_shape, input.data[start..end].to_vec())
}

fn sequence_write(
    output: &mut Tensor,
    step: usize,
    sequence_batch: usize,
    step_output: &Tensor,
) -> Result<()> {
    if step_output.shape.batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "recurrent output step batch mismatch".to_owned(),
        ));
    }
    let step_len = sequence_batch * step_output.shape.element_count();
    let start = step
        .checked_mul(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent output offset overflow".to_owned()))?;
    let end = start
        .checked_add(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent output end overflow".to_owned()))?;
    if end > output.data.len() {
        return Err(Error::InvalidConfig(
            "recurrent output does not contain all requested steps".to_owned(),
        ));
    }
    output.data[start..end].copy_from_slice(&step_output.data);
    Ok(())
}

fn recurrent_connected_spec(
    output: usize,
    batch_normalize: bool,
    activation: Activation,
) -> ConnectedSpec {
    ConnectedSpec {
        output,
        batch_normalize,
        activation,
    }
}

fn recurrent_convolutional_spec(spec: &CrnnSpec, filters: usize) -> ConvolutionalSpec {
    ConvolutionalSpec {
        filters,
        size: spec.size,
        stride_x: spec.stride,
        stride_y: spec.stride,
        dilation: spec.dilation,
        padding: spec.padding,
        groups: spec.groups,
        share_index: None,
        batch_normalize: spec.batch_normalize,
        activation: spec.activation.clone(),
    }
}

fn combine_tensors(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape != rhs.shape {
        return Err(Error::InvalidConfig(format!(
            "recurrent tensors have incompatible shapes {} and {}",
            lhs.shape, rhs.shape
        )));
    }
    let mut output = lhs.clone();
    for (dst, src) in output.data.iter_mut().zip(&rhs.data) {
        *dst += *src;
    }
    Ok(output)
}

fn add_tensor_inplace(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.shape != src.shape {
        return Err(Error::InvalidConfig(format!(
            "recurrent gradient shapes do not match: {} and {}",
            dst.shape, src.shape
        )));
    }
    for (dst, src) in dst.data.iter_mut().zip(&src.data) {
        *dst += *src;
    }
    Ok(())
}

fn forward_gru_layer(
    spec: &GruSpec,
    params: &GruParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "GRU output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input.shape.element_count();
    for (name, connected, expected_inputs) in [
        ("input_z", &params.input_z, inputs),
        ("state_z", &params.state_z, spec.output),
        ("input_r", &params.input_r, inputs),
        ("state_r", &params.state_r, spec.output),
        ("input_h", &params.input_h, inputs),
        ("state_h", &params.state_h, spec.output),
    ] {
        if connected.weights.len() != expected_inputs * spec.output
            || connected.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "GRU {name} parameter shape is inconsistent with its specification"
            )));
        }
    }

    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let input_linear =
        recurrent_connected_spec(spec.output, spec.batch_normalize, Activation::Linear);
    let state_linear =
        recurrent_connected_spec(spec.output, spec.batch_normalize, Activation::Linear);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(state_shape);

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let input_z =
            forward_connected_with_cache(&input_linear, &params.input_z, &step_input, state_shape)
                .0;
        let state_z = forward_connected_with_cache(
            &state_linear,
            &params.state_z,
            &previous_state,
            state_shape,
        )
        .0;
        let input_r =
            forward_connected_with_cache(&input_linear, &params.input_r, &step_input, state_shape)
                .0;
        let state_r = forward_connected_with_cache(
            &state_linear,
            &params.state_r,
            &previous_state,
            state_shape,
        )
        .0;
        let input_h =
            forward_connected_with_cache(&input_linear, &params.input_h, &step_input, state_shape)
                .0;
        let mut z = combine_tensors(&input_z, &state_z)?;
        let mut r = combine_tensors(&input_r, &state_r)?;
        for value in &mut z.data {
            *value = activate(Activation::Logistic, *value);
        }
        for value in &mut r.data {
            *value = activate(Activation::Logistic, *value);
        }
        let mut reset_state = previous_state.clone();
        for (value, reset) in reset_state.data.iter_mut().zip(&r.data) {
            *value *= *reset;
        }
        let state_h =
            forward_connected_with_cache(&state_linear, &params.state_h, &reset_state, state_shape)
                .0;
        let mut candidate = combine_tensors(&input_h, &state_h)?;
        for value in &mut candidate.data {
            *value = activate(Activation::Logistic, *value);
        }
        let mut state = Tensor::zeros(state_shape);
        for (((dst, previous), candidate), update) in state
            .data
            .iter_mut()
            .zip(&previous_state.data)
            .zip(&candidate.data)
            .zip(&z.data)
        {
            *dst = *update * *previous + (1.0 - *update) * *candidate;
        }
        sequence_write(&mut output, step, sequence_batch, &state)?;
        previous_state = state;
    }
    Ok(output)
}

fn forward_rnn_layer(
    spec: &RnnSpec,
    params: &RnnParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, RnnForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "RNN output shape is inconsistent with its specification".to_owned(),
        ));
    }
    if params.input.weights.len() != input.shape.element_count() * spec.hidden
        || params.self_params.weights.len() != spec.hidden * spec.hidden
        || params.output.weights.len() != spec.hidden * spec.output
    {
        return Err(Error::InvalidConfig(
            "RNN parameter shapes are inconsistent with its specification".to_owned(),
        ));
    }

    let input_spec =
        recurrent_connected_spec(spec.hidden, spec.batch_normalize, spec.activation.clone());
    let self_activation = match spec.logistic {
        1 => Activation::Logistic,
        2 => Activation::Loggy,
        _ => spec.activation.clone(),
    };
    let self_spec = recurrent_connected_spec(spec.hidden, spec.batch_normalize, self_activation);
    let output_spec =
        recurrent_connected_spec(spec.output, spec.batch_normalize, spec.activation.clone());
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
    let output_step_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(hidden_shape);
    let mut cache = RnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (input_output, input_activation_input) =
            forward_connected_with_cache(&input_spec, &params.input, &step_input, hidden_shape);
        let (self_output, self_activation_input) = forward_connected_with_cache(
            &self_spec,
            &params.self_params,
            &previous_state,
            hidden_shape,
        );
        let mut state = combine_tensors(&input_output, &self_output)?;
        if spec.shortcut {
            add_tensor_inplace(&mut state, &previous_state)?;
        }
        let (step_output, output_activation_input) =
            forward_connected_with_cache(&output_spec, &params.output, &state, output_step_shape);
        sequence_write(&mut output, step, sequence_batch, &step_output)?;
        cache.input_outputs.push(input_output);
        cache.input_activation_inputs.push(
            input_activation_input.expect("connected forward always records an activation input"),
        );
        cache.self_outputs.push(self_output);
        cache.self_activation_inputs.push(
            self_activation_input.expect("connected forward always records an activation input"),
        );
        cache.previous_states.push(previous_state);
        cache.states.push(state.clone());
        cache.output_activation_inputs.push(
            output_activation_input.expect("connected forward always records an activation input"),
        );
        cache.outputs.push(step_output);
        previous_state = state;
    }

    Ok((output, cache))
}

fn forward_lstm_layer(
    spec: &LstmSpec,
    params: &LstmParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, LstmForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "LSTM output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input.shape.element_count();
    let expected = inputs * spec.output;
    for nested in [
        &params.wf, &params.wi, &params.wg, &params.wo, &params.uf, &params.ui, &params.ug,
        &params.uo,
    ] {
        if nested.weights.len() != expected && nested.weights.len() != spec.output * spec.output {
            return Err(Error::InvalidConfig(
                "LSTM parameter shapes are inconsistent with its specification".to_owned(),
            ));
        }
    }

    let linear =
        |output| recurrent_connected_spec(output, spec.batch_normalize, Activation::Linear);
    let wf_spec = linear(spec.output);
    let wi_spec = linear(spec.output);
    let wg_spec = linear(spec.output);
    let wo_spec = linear(spec.output);
    let uf_spec = linear(spec.output);
    let ui_spec = linear(spec.output);
    let ug_spec = linear(spec.output);
    let uo_spec = linear(spec.output);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(state_shape);
    let mut previous_cell = Tensor::zeros(state_shape);
    let mut cache = LstmForwardCache {
        sequence_batch,
        wf_outputs: Vec::with_capacity(spec.steps),
        wf_activation_inputs: Vec::with_capacity(spec.steps),
        wi_outputs: Vec::with_capacity(spec.steps),
        wi_activation_inputs: Vec::with_capacity(spec.steps),
        wg_outputs: Vec::with_capacity(spec.steps),
        wg_activation_inputs: Vec::with_capacity(spec.steps),
        wo_outputs: Vec::with_capacity(spec.steps),
        wo_activation_inputs: Vec::with_capacity(spec.steps),
        uf_outputs: Vec::with_capacity(spec.steps),
        uf_activation_inputs: Vec::with_capacity(spec.steps),
        ui_outputs: Vec::with_capacity(spec.steps),
        ui_activation_inputs: Vec::with_capacity(spec.steps),
        ug_outputs: Vec::with_capacity(spec.steps),
        ug_activation_inputs: Vec::with_capacity(spec.steps),
        uo_outputs: Vec::with_capacity(spec.steps),
        uo_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        previous_cells: Vec::with_capacity(spec.steps),
        forget: Vec::with_capacity(spec.steps),
        input_gate: Vec::with_capacity(spec.steps),
        candidate: Vec::with_capacity(spec.steps),
        output_gate: Vec::with_capacity(spec.steps),
        cells: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (wf_output, wf_activation_input) =
            forward_connected_with_cache(&wf_spec, &params.wf, &previous_state, state_shape);
        let (wi_output, wi_activation_input) =
            forward_connected_with_cache(&wi_spec, &params.wi, &previous_state, state_shape);
        let (wg_output, wg_activation_input) =
            forward_connected_with_cache(&wg_spec, &params.wg, &previous_state, state_shape);
        let (wo_output, wo_activation_input) =
            forward_connected_with_cache(&wo_spec, &params.wo, &previous_state, state_shape);
        let (uf_output, uf_activation_input) =
            forward_connected_with_cache(&uf_spec, &params.uf, &step_input, state_shape);
        let (ui_output, ui_activation_input) =
            forward_connected_with_cache(&ui_spec, &params.ui, &step_input, state_shape);
        let (ug_output, ug_activation_input) =
            forward_connected_with_cache(&ug_spec, &params.ug, &step_input, state_shape);
        let (uo_output, uo_activation_input) =
            forward_connected_with_cache(&uo_spec, &params.uo, &step_input, state_shape);

        let mut forget = Tensor::zeros(state_shape);
        let mut input_gate = Tensor::zeros(state_shape);
        let mut candidate = Tensor::zeros(state_shape);
        let mut output_gate = Tensor::zeros(state_shape);
        for index in 0..forget.data.len() {
            forget.data[index] = activate(
                Activation::Logistic,
                wf_output.data[index] + uf_output.data[index],
            );
            input_gate.data[index] = activate(
                Activation::Logistic,
                wi_output.data[index] + ui_output.data[index],
            );
            candidate.data[index] = activate(
                Activation::Tanh,
                wg_output.data[index] + ug_output.data[index],
            );
            output_gate.data[index] = activate(
                Activation::Logistic,
                wo_output.data[index] + uo_output.data[index],
            );
        }
        let mut cell = Tensor::zeros(state_shape);
        let mut state = Tensor::zeros(state_shape);
        for index in 0..cell.data.len() {
            cell.data[index] = forget.data[index] * previous_cell.data[index]
                + input_gate.data[index] * candidate.data[index];
            state.data[index] =
                output_gate.data[index] * activate(Activation::Tanh, cell.data[index]);
        }
        sequence_write(&mut output, step, sequence_batch, &state)?;

        cache.wf_outputs.push(wf_output);
        cache
            .wf_activation_inputs
            .push(wf_activation_input.expect("connected forward cache"));
        cache.wi_outputs.push(wi_output);
        cache
            .wi_activation_inputs
            .push(wi_activation_input.expect("connected forward cache"));
        cache.wg_outputs.push(wg_output);
        cache
            .wg_activation_inputs
            .push(wg_activation_input.expect("connected forward cache"));
        cache.wo_outputs.push(wo_output);
        cache
            .wo_activation_inputs
            .push(wo_activation_input.expect("connected forward cache"));
        cache.uf_outputs.push(uf_output);
        cache
            .uf_activation_inputs
            .push(uf_activation_input.expect("connected forward cache"));
        cache.ui_outputs.push(ui_output);
        cache
            .ui_activation_inputs
            .push(ui_activation_input.expect("connected forward cache"));
        cache.ug_outputs.push(ug_output);
        cache
            .ug_activation_inputs
            .push(ug_activation_input.expect("connected forward cache"));
        cache.uo_outputs.push(uo_output);
        cache
            .uo_activation_inputs
            .push(uo_activation_input.expect("connected forward cache"));
        cache.previous_states.push(previous_state);
        cache.previous_cells.push(previous_cell);
        cache.forget.push(forget);
        cache.input_gate.push(input_gate);
        cache.candidate.push(candidate);
        cache.output_gate.push(output_gate);
        cache.cells.push(cell.clone());
        cache.states.push(state.clone());
        previous_state = state;
        previous_cell = cell;
    }

    Ok((output, cache))
}

fn forward_crnn_layer(
    spec: &CrnnSpec,
    params: &CrnnParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, CrnnForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    let input_width =
        convolution_output_dim(input.shape.width, spec.padding, spec.size, spec.stride);
    let input_height =
        convolution_output_dim(input.shape.height, spec.padding, spec.size, spec.stride);
    let hidden_shape = TensorShape::new(input_width, input_height, spec.hidden, sequence_batch);
    let output_step_shape =
        TensorShape::new(input_width, input_height, spec.output, sequence_batch);
    if output_shape != TensorShape::new(input_width, input_height, spec.output, input.shape.batch) {
        return Err(Error::InvalidConfig(
            "CRNN output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(hidden_shape);
    let mut cache = CrnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (input_output, input_activation_input) =
            forward_convolutional_with_cache(&input_spec, &params.input, &step_input, hidden_shape);
        let (self_output, self_activation_input) = forward_convolutional_with_cache(
            &self_spec,
            &params.self_params,
            &previous_state,
            hidden_shape,
        );
        let mut state = combine_tensors(&input_output, &self_output)?;
        if spec.shortcut {
            add_tensor_inplace(&mut state, &previous_state)?;
        }
        let (step_output, output_activation_input) = forward_convolutional_with_cache(
            &output_spec,
            &params.output,
            &state,
            output_step_shape,
        );
        sequence_write(&mut output, step, sequence_batch, &step_output)?;
        cache.input_outputs.push(input_output);
        cache.input_activation_inputs.push(
            input_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.self_outputs.push(self_output);
        cache.self_activation_inputs.push(
            self_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.previous_states.push(previous_state);
        cache.states.push(state.clone());
        cache.output_activation_inputs.push(
            output_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.outputs.push(step_output);
        previous_state = state;
    }
    Ok((output, cache))
}

fn zero_connected_gradient(params: &ConnectedParams) -> ConnectedParams {
    ConnectedParams {
        biases: vec![0.0; params.biases.len()],
        batch_norm: params
            .batch_norm
            .as_ref()
            .map(|batch_norm| BatchNormParams {
                biases: vec![0.0; batch_norm.biases.len()],
                scales: vec![0.0; batch_norm.scales.len()],
                rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
                rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
            }),
        weights: vec![0.0; params.weights.len()],
    }
}

fn zero_convolutional_gradient(params: &ConvolutionalParams) -> ConvolutionalParams {
    ConvolutionalParams {
        biases: vec![0.0; params.biases.len()],
        batch_norm: params
            .batch_norm
            .as_ref()
            .map(|batch_norm| BatchNormParams {
                biases: vec![0.0; batch_norm.biases.len()],
                scales: vec![0.0; batch_norm.scales.len()],
                rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
                rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
            }),
        weights: vec![0.0; params.weights.len()],
    }
}

fn accumulate_connected_gradient(dst: &mut ConnectedParams, src: &ConnectedParams) -> Result<()> {
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
                "connected recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn accumulate_convolutional_gradient(
    dst: &mut ConvolutionalParams,
    src: &ConvolutionalParams,
) -> Result<()> {
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
                "convolutional recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

fn backward_rnn_layer(
    spec: &RnnSpec,
    params: &RnnParams,
    cache: &RnnForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, RnnParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "RNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
    let output_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    if output_gradient.shape.batch != input.shape.batch
        || output_gradient.shape.width != 1
        || output_gradient.shape.height != 1
        || output_gradient.shape.channels != spec.output
    {
        return Err(Error::InvalidConfig(
            "RNN backward gradient shape mismatch".to_owned(),
        ));
    }
    let input_spec =
        recurrent_connected_spec(spec.hidden, spec.batch_normalize, spec.activation.clone());
    let self_activation = match spec.logistic {
        1 => Activation::Logistic,
        2 => Activation::Loggy,
        _ => spec.activation.clone(),
    };
    let self_spec = recurrent_connected_spec(spec.hidden, spec.batch_normalize, self_activation);
    let output_spec =
        recurrent_connected_spec(spec.output, spec.batch_normalize, spec.activation.clone());
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = RnnParams {
        input: zero_connected_gradient(&params.input),
        self_params: zero_connected_gradient(&params.self_params),
        output: zero_connected_gradient(&params.output),
    };
    let mut future_state_gradient = Tensor::zeros(hidden_shape);

    for step in (0..spec.steps).rev() {
        let output_gradient_step =
            sequence_slice(output_gradient, step, sequence_batch, output_shape)?;
        let (mut state_gradient, output_gradient_params) = backward_connected(
            &output_spec,
            &params.output,
            &cache.states[step],
            output_shape,
            &output_gradient_step,
            &cache.outputs[step],
            Some(&cache.output_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut state_gradient, &future_state_gradient)?;
        accumulate_connected_gradient(&mut gradients.output, &output_gradient_params)?;

        let (previous_from_self, self_gradient) = backward_connected(
            &self_spec,
            &params.self_params,
            &cache.previous_states[step],
            hidden_shape,
            &state_gradient,
            &cache.self_outputs[step],
            Some(&cache.self_activation_inputs[step]),
            workspace,
        )?;
        accumulate_connected_gradient(&mut gradients.self_params, &self_gradient)?;

        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (step_input_gradient, input_gradient_params) = backward_connected(
            &input_spec,
            &params.input,
            &step_input,
            hidden_shape,
            &state_gradient,
            &cache.input_outputs[step],
            Some(&cache.input_activation_inputs[step]),
            workspace,
        )?;
        accumulate_connected_gradient(&mut gradients.input, &input_gradient_params)?;
        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;

        future_state_gradient = previous_from_self;
        if spec.shortcut {
            add_tensor_inplace(&mut future_state_gradient, &state_gradient)?;
        }
        if step == 0 {
            future_state_gradient = Tensor::zeros(hidden_shape);
        }
    }

    Ok((input_gradient, gradients))
}

fn backward_lstm_layer(
    spec: &LstmSpec,
    params: &LstmParams,
    cache: &LstmForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, LstmParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "LSTM forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let inputs_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    if output_gradient.shape != TensorShape::new(1, 1, spec.output, input.shape.batch) {
        return Err(Error::InvalidConfig(
            "LSTM backward gradient shape mismatch".to_owned(),
        ));
    }
    let linear =
        |output| recurrent_connected_spec(output, spec.batch_normalize, Activation::Linear);
    let wf_spec = linear(spec.output);
    let wi_spec = linear(spec.output);
    let wg_spec = linear(spec.output);
    let wo_spec = linear(spec.output);
    let uf_spec = linear(spec.output);
    let ui_spec = linear(spec.output);
    let ug_spec = linear(spec.output);
    let uo_spec = linear(spec.output);
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = LstmParams {
        wf: zero_connected_gradient(&params.wf),
        wi: zero_connected_gradient(&params.wi),
        wg: zero_connected_gradient(&params.wg),
        wo: zero_connected_gradient(&params.wo),
        uf: zero_connected_gradient(&params.uf),
        ui: zero_connected_gradient(&params.ui),
        ug: zero_connected_gradient(&params.ug),
        uo: zero_connected_gradient(&params.uo),
    };
    let mut next_hidden_gradient = Tensor::zeros(state_shape);
    let mut next_cell_gradient = Tensor::zeros(state_shape);

    for step in (0..spec.steps).rev() {
        let mut hidden_gradient =
            sequence_slice(output_gradient, step, sequence_batch, state_shape)?;
        add_tensor_inplace(&mut hidden_gradient, &next_hidden_gradient)?;
        let cell = &cache.cells[step];
        let previous_cell = &cache.previous_cells[step];
        let forget = &cache.forget[step];
        let input_gate = &cache.input_gate[step];
        let candidate = &cache.candidate[step];
        let output_gate = &cache.output_gate[step];

        let mut forget_gradient = Tensor::zeros(state_shape);
        let mut input_gate_gradient = Tensor::zeros(state_shape);
        let mut candidate_gradient = Tensor::zeros(state_shape);
        let mut output_gate_gradient = Tensor::zeros(state_shape);
        let mut cell_gradient = Tensor::zeros(state_shape);
        for index in 0..cell.data.len() {
            let tanh_cell = cell.data[index].tanh();
            output_gate_gradient.data[index] = hidden_gradient.data[index] * tanh_cell;
            cell_gradient.data[index] = hidden_gradient.data[index]
                * output_gate.data[index]
                * (1.0 - tanh_cell * tanh_cell)
                + next_cell_gradient.data[index];
            forget_gradient.data[index] = cell_gradient.data[index] * previous_cell.data[index];
            input_gate_gradient.data[index] = cell_gradient.data[index] * candidate.data[index];
            candidate_gradient.data[index] = cell_gradient.data[index] * input_gate.data[index];
            forget_gradient.data[index] *= forget.data[index] * (1.0 - forget.data[index]);
            input_gate_gradient.data[index] *=
                input_gate.data[index] * (1.0 - input_gate.data[index]);
            candidate_gradient.data[index] *= 1.0 - candidate.data[index] * candidate.data[index];
            output_gate_gradient.data[index] *=
                output_gate.data[index] * (1.0 - output_gate.data[index]);
        }

        let mut previous_hidden_gradient = Tensor::zeros(state_shape);
        let mut step_input_gradient = Tensor::zeros(inputs_shape);
        let previous_state = &cache.previous_states[step];
        let step_input = sequence_slice(input, step, sequence_batch, inputs_shape)?;

        let (gradient, parameter_gradient) = backward_connected(
            &wf_spec,
            &params.wf,
            previous_state,
            state_shape,
            &forget_gradient,
            &cache.wf_outputs[step],
            Some(&cache.wf_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wf, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wi_spec,
            &params.wi,
            previous_state,
            state_shape,
            &input_gate_gradient,
            &cache.wi_outputs[step],
            Some(&cache.wi_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wi, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wg_spec,
            &params.wg,
            previous_state,
            state_shape,
            &candidate_gradient,
            &cache.wg_outputs[step],
            Some(&cache.wg_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wg, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wo_spec,
            &params.wo,
            previous_state,
            state_shape,
            &output_gate_gradient,
            &cache.wo_outputs[step],
            Some(&cache.wo_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wo, &parameter_gradient)?;

        let (gradient, parameter_gradient) = backward_connected(
            &uf_spec,
            &params.uf,
            &step_input,
            state_shape,
            &forget_gradient,
            &cache.uf_outputs[step],
            Some(&cache.uf_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.uf, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &ui_spec,
            &params.ui,
            &step_input,
            state_shape,
            &input_gate_gradient,
            &cache.ui_outputs[step],
            Some(&cache.ui_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.ui, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &ug_spec,
            &params.ug,
            &step_input,
            state_shape,
            &candidate_gradient,
            &cache.ug_outputs[step],
            Some(&cache.ug_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.ug, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &uo_spec,
            &params.uo,
            &step_input,
            state_shape,
            &output_gate_gradient,
            &cache.uo_outputs[step],
            Some(&cache.uo_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.uo, &parameter_gradient)?;

        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;
        next_hidden_gradient = previous_hidden_gradient;
        for (index, (dst, value)) in next_cell_gradient
            .data
            .iter_mut()
            .zip(&cell_gradient.data)
            .enumerate()
        {
            *dst = *value * forget.data[index];
        }
    }

    Ok((input_gradient, gradients))
}

fn backward_crnn_layer(
    spec: &CrnnSpec,
    params: &CrnnParams,
    cache: &CrnnForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, CrnnParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "CRNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let input_width =
        convolution_output_dim(input.shape.width, spec.padding, spec.size, spec.stride);
    let input_height =
        convolution_output_dim(input.shape.height, spec.padding, spec.size, spec.stride);
    let hidden_shape = TensorShape::new(input_width, input_height, spec.hidden, sequence_batch);
    let output_shape = TensorShape::new(input_width, input_height, spec.output, sequence_batch);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    if output_gradient.shape
        != TensorShape::new(input_width, input_height, spec.output, input.shape.batch)
    {
        return Err(Error::InvalidConfig(
            "CRNN backward gradient shape mismatch".to_owned(),
        ));
    }
    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = CrnnParams {
        input: zero_convolutional_gradient(&params.input),
        self_params: zero_convolutional_gradient(&params.self_params),
        output: zero_convolutional_gradient(&params.output),
    };
    let mut future_state_gradient = Tensor::zeros(hidden_shape);

    for step in (0..spec.steps).rev() {
        let output_gradient_step =
            sequence_slice(output_gradient, step, sequence_batch, output_shape)?;
        let (mut state_gradient, output_gradient_params) = backward_convolutional(
            &output_spec,
            &params.output,
            &cache.states[step],
            output_shape,
            &output_gradient_step,
            Some(&cache.output_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut state_gradient, &future_state_gradient)?;
        accumulate_convolutional_gradient(&mut gradients.output, &output_gradient_params)?;

        let (previous_from_self, self_gradient) = backward_convolutional(
            &self_spec,
            &params.self_params,
            &cache.previous_states[step],
            hidden_shape,
            &state_gradient,
            Some(&cache.self_activation_inputs[step]),
            workspace,
        )?;
        accumulate_convolutional_gradient(&mut gradients.self_params, &self_gradient)?;

        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (step_input_gradient, input_gradient_params) = backward_convolutional(
            &input_spec,
            &params.input,
            &step_input,
            hidden_shape,
            &state_gradient,
            Some(&cache.input_activation_inputs[step]),
            workspace,
        )?;
        accumulate_convolutional_gradient(&mut gradients.input, &input_gradient_params)?;
        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;

        future_state_gradient = previous_from_self;
        if spec.shortcut {
            add_tensor_inplace(&mut future_state_gradient, &state_gradient)?;
        }
        if step == 0 {
            future_state_gradient = Tensor::zeros(hidden_shape);
        }
    }

    Ok((input_gradient, gradients))
}

/// Elements per softmax group in one batch item.
fn softmax_group_size(shape: TensorShape, groups: usize) -> Result<usize> {
    let groups = groups.max(1);
    let per_batch = shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    Ok(per_batch / groups)
}

fn forward_softmax(input: &Tensor, groups: usize) -> Result<Tensor> {
    let groups = groups.max(1);
    let per_batch = input.shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    let group_size = per_batch / groups;
    let mut output = Tensor::zeros(input.shape);
    for b in 0..input.shape.batch {
        let base = b * per_batch;
        for group in 0..groups {
            let start = base + group * group_size;
            let end = start + group_size;
            let max_value = input.data[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for (dst, value) in output.data[start..end]
                .iter_mut()
                .zip(&input.data[start..end])
            {
                *dst = (*value - max_value).exp();
                sum += *dst;
            }
            if sum > 0.0 {
                for value in &mut output.data[start..end] {
                    *value /= sum;
                }
            }
        }
    }
    Ok(output)
}

fn check_dropout_probability(probability: f32) -> Result<()> {
    if !(0.0..1.0).contains(&probability) {
        return Err(Error::InvalidConfig(format!(
            "dropout probability must be in [0, 1), got {probability}"
        )));
    }
    Ok(())
}

fn forward_dropout(
    input: &Tensor,
    probability: f32,
    training: bool,
    rng_state: &mut u64,
) -> Result<(Tensor, Option<Tensor>)> {
    check_dropout_probability(probability)?;
    if !training || probability == 0.0 {
        return Ok((input.clone(), None));
    }

    // One seed per dropout layer, then a counter-hash per element
    // (`darknet_cuda::dropout_unit`), so a GPU backend reproduces this mask
    // exactly from the same seed.
    let seed = advance_rng(rng_state);
    let scale = 1.0 / (1.0 - probability);
    let mut output = Tensor::zeros(input.shape);
    let mut mask = Tensor::zeros(input.shape);
    for (index, ((out, mask_value), value)) in output
        .data
        .iter_mut()
        .zip(mask.data.iter_mut())
        .zip(&input.data)
        .enumerate()
    {
        let multiplier = if dropout_unit(seed, index as u64) < probability {
            0.0
        } else {
            scale
        };
        *mask_value = multiplier;
        *out = *value * multiplier;
    }
    Ok((output, Some(mask)))
}

/// C++ `forward_dropout_layer_gpu` DropBlock schedule: during the first 85% of
/// `max_batches` the probability and block size ramp up linearly.  Returns the
/// per-cell block probability `p * ramp / block_size^2` and the square block
/// side (`min` of the clamped width/height, >= 1).
fn dropblock_plan(
    spec: &DropblockSpec,
    probability: f32,
    shape: TensorShape,
    iteration: usize,
    max_batches: usize,
) -> (f32, usize) {
    let ramp_end = max_batches as f64 * 0.85;
    let multiplier = if (iteration as f64) < ramp_end {
        iteration as f32 / ramp_end as f32
    } else {
        1.0
    };
    let (mut block_width, mut block_height) = (
        (spec.size_abs as f32 * multiplier) as i64,
        (spec.size_abs as f32 * multiplier) as i64,
    );
    if spec.size_rel != 0.0 {
        block_width = (spec.size_rel * shape.width as f32 * multiplier) as i64;
        block_height = (spec.size_rel * shape.height as f32 * multiplier) as i64;
    }
    let block_width = block_width.clamp(1, shape.width.max(1) as i64);
    let block_height = block_height.clamp(1, shape.height.max(1) as i64);
    let block_size = block_width.min(block_height) as usize;
    (
        probability * multiplier / (block_size * block_size) as f32,
        block_size,
    )
}

/// Per-iteration DropBlock seed: the layer seed mixed with the iteration so
/// every step draws new blocks.
fn dropblock_seed(layer_seed: u64, iteration: usize) -> u64 {
    layer_seed ^ (iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// DropBlock forward (see `darknet_cuda::host_ops::dropblock`); like dropout
/// the returned mask is the per-element multiplier used by the backward.
fn forward_dropblock(
    input: &Tensor,
    spec: &DropblockSpec,
    probability: f32,
    training: bool,
    rng_state: &mut u64,
    iteration: usize,
    max_batches: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    check_dropout_probability(probability)?;
    if !training || probability == 0.0 {
        return Ok((input.clone(), None));
    }
    let seed = advance_rng(rng_state);
    let (block_prob, block_size) = dropblock_plan(spec, probability, input.shape, iteration, max_batches);
    if block_prob <= 0.0 {
        return Ok((input.clone(), None));
    }
    let shape = tensor_backend_shape(input.shape);
    let (out, mask) = darknet_cuda::host_ops::dropblock(
        &input.data,
        &shape,
        block_prob,
        block_size,
        dropblock_seed(seed, iteration),
    );
    Ok((
        Tensor {
            shape: input.shape,
            data: out,
        },
        Some(Tensor {
            shape: input.shape,
            data: mask,
        }),
    ))
}

fn advance_rng(state: &mut u64) -> u64 {
    // A local generator keeps the core crate dependency-free.  Re-seeding per
    // forward call is intentional: inference remains deterministic and tests
    // can reproduce a training pass exactly.
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn next_random_unit(state: &mut u64) -> f32 {
    ((advance_rng(state) >> 40) as u32) as f32 / (u32::MAX as f32 + 1.0)
}

fn crop_sample(input: &Tensor, batch: usize, channel: usize, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x >= input.shape.width as isize || y >= input.shape.height as isize {
        0.0
    } else {
        input.get(batch, channel, y as usize, x as usize)
    }
}

#[derive(Clone, Copy)]
struct CropLevelAdjustment {
    saturation: f32,
    exposure: f32,
    rgb_shift: [f32; 3],
    scale: f32,
    translate: f32,
    apply_hsv: bool,
}

fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max - min;
    if max == 0.0 {
        return [-1.0, 0.0, 0.0];
    }
    let saturation = delta / max;
    let mut hue = if delta == 0.0 {
        0.0
    } else if rgb[0] == max {
        (rgb[1] - rgb[2]) / delta
    } else if rgb[1] == max {
        2.0 + (rgb[2] - rgb[0]) / delta
    } else {
        4.0 + (rgb[0] - rgb[1]) / delta
    };
    if hue < 0.0 {
        hue += 6.0;
    }
    [hue, saturation, max]
}

fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let [hue, saturation, value] = hsv;
    let rgb = if saturation == 0.0 {
        [value, value, value]
    } else {
        let index = hue.floor() as i32;
        let fractional = hue - index as f32;
        let p = value * (1.0 - saturation);
        let q = value * (1.0 - saturation * fractional);
        let t = value * (1.0 - saturation * (1.0 - fractional));
        match index {
            0 => [value, t, p],
            1 => [q, value, p],
            2 => [p, value, t],
            3 => [p, q, value],
            4 => [t, p, value],
            _ => [value, p, q],
        }
    };
    rgb.map(|value| value.clamp(0.0, 1.0))
}

fn crop_adjusted_sample(
    input: &Tensor,
    batch: usize,
    channel: usize,
    x: isize,
    y: isize,
    adjustment: CropLevelAdjustment,
) -> f32 {
    if x < 0 || y < 0 || x >= input.shape.width as isize || y >= input.shape.height as isize {
        return 0.0;
    }
    let mut value = crop_sample(input, batch, channel, x, y);
    if channel < 3 && input.shape.channels >= 3 {
        let mut rgb = [
            crop_sample(input, batch, 0, x, y),
            crop_sample(input, batch, 1, x, y),
            crop_sample(input, batch, 2, x, y),
        ];
        if adjustment.apply_hsv {
            let mut hsv = rgb_to_hsv(rgb);
            hsv[1] *= adjustment.saturation;
            hsv[2] *= adjustment.exposure;
            rgb = hsv_to_rgb(hsv);
        }
        value = rgb[channel];
    }
    let shift = if channel < 3 {
        adjustment.rgb_shift[channel]
    } else {
        0.0
    };
    value * adjustment.scale + adjustment.translate + shift
}

fn crop_bilinear(
    input: &Tensor,
    batch: usize,
    channel: usize,
    x: f32,
    y: f32,
    adjustment: CropLevelAdjustment,
) -> f32 {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let dx = x - ix as f32;
    let dy = y - iy as f32;
    (1.0 - dy) * (1.0 - dx) * crop_adjusted_sample(input, batch, channel, ix, iy, adjustment)
        + dy * (1.0 - dx) * crop_adjusted_sample(input, batch, channel, ix, iy + 1, adjustment)
        + (1.0 - dy) * dx * crop_adjusted_sample(input, batch, channel, ix + 1, iy, adjustment)
        + dy * dx * crop_adjusted_sample(input, batch, channel, ix + 1, iy + 1, adjustment)
}

/// Plans crop coordinates and color-level adjustments without touching tensor
/// data. The GPU executor uploads this metadata so image samples themselves
/// stay resident on the device.
fn plan_crop_forward(
    spec: &CropSpec,
    input_shape: TensorShape,
    output_shape: TensorShape,
    training: bool,
    rng_state: &mut u64,
) -> Result<(CropForwardCache, Vec<CropLevelAdjustment>)> {
    if output_shape.batch != input_shape.batch || output_shape.channels != input_shape.channels {
        return Err(Error::InvalidConfig(format!(
            "crop output shape must preserve batch/channels: input {}, output {}",
            input_shape, output_shape
        )));
    }
    let samples = output_shape.batch * output_shape.width * output_shape.height;
    let mut cache = CropForwardCache {
        source_x: vec![0.0; samples],
        source_y: vec![0.0; samples],
        input_scales: vec![0.0; input_shape.batch],
        input_offsets: vec![0.0; input_shape.batch],
    };
    let mut adjustments = Vec::with_capacity(input_shape.batch);
    let center_x = input_shape.width as f32 / 2.0;
    let center_y = input_shape.height as f32 / 2.0;

    for batch in 0..input_shape.batch {
        let random: [f32; 8] = std::array::from_fn(|_| next_random_unit(rng_state));
        let adjustment = if training {
            let mut saturation = random[0] * (spec.saturation - 1.0) + 1.0;
            if random[1] > 0.5 {
                saturation = 1.0 / saturation;
            }
            let mut exposure = random[2] * (spec.exposure - 1.0) + 1.0;
            if random[3] > 0.5 {
                exposure = 1.0 / exposure;
            }
            CropLevelAdjustment {
                saturation,
                exposure,
                rgb_shift: [
                    (random[0] - 0.5) * spec.shift,
                    (random[1] - 0.5) * spec.shift,
                    (random[2] - 0.5) * spec.shift,
                ],
                scale: if spec.noadjust { 1.0 } else { 2.0 },
                translate: if spec.noadjust { 0.0 } else { -1.0 },
                apply_hsv: true,
            }
        } else {
            CropLevelAdjustment {
                saturation: 1.0,
                exposure: 1.0,
                rgb_shift: [0.0; 3],
                scale: if spec.noadjust { 1.0 } else { 2.0 },
                translate: if spec.noadjust { 0.0 } else { -1.0 },
                apply_hsv: false,
            }
        };
        cache.input_scales[batch] = adjustment.scale;
        cache.input_offsets[batch] = adjustment.translate;
        let (dw, dh, flip, angle) = if training {
            let dw = (input_shape.width.saturating_sub(spec.width)) as f32 * random[4];
            let dh = (input_shape.height.saturating_sub(spec.height)) as f32 * random[5];
            let flip = spec.flip && random[6] > 0.5;
            let angle = (2.0 * random[7] - 1.0) * spec.angle.to_radians();
            (dw, dh, flip, angle)
        } else {
            (
                (input_shape.width.saturating_sub(spec.width)) as f32 / 2.0,
                (input_shape.height.saturating_sub(spec.height)) as f32 / 2.0,
                false,
                0.0,
            )
        };
        let (sin, cos) = angle.sin_cos();
        for oy in 0..output_shape.height {
            for ox in 0..output_shape.width {
                let x = if flip {
                    input_shape.width as f32 - dw - ox as f32 - 1.0
                } else {
                    ox as f32 + dw
                };
                let y = oy as f32 + dh;
                let source_x = cos * (x - center_x) - sin * (y - center_y) + center_x;
                let source_y = sin * (x - center_x) + cos * (y - center_y) + center_y;
                let index = (batch * output_shape.height + oy) * output_shape.width + ox;
                cache.source_x[index] = source_x;
                cache.source_y[index] = source_y;
            }
        }
        adjustments.push(adjustment);
    }
    Ok((cache, adjustments))
}

fn forward_crop(
    spec: &CropSpec,
    input: &Tensor,
    output_shape: TensorShape,
    training: bool,
    rng_state: &mut u64,
) -> Result<(Tensor, CropForwardCache)> {
    let (cache, adjustments) =
        plan_crop_forward(spec, input.shape, output_shape, training, rng_state)?;
    let mut output = Tensor::zeros(output_shape);
    for batch in 0..output_shape.batch {
        for channel in 0..output_shape.channels {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let index = (batch * output_shape.height + oy) * output_shape.width + ox;
                    output.set(
                        batch,
                        channel,
                        oy,
                        ox,
                        crop_bilinear(
                            input,
                            batch,
                            channel,
                            cache.source_x[index],
                            cache.source_y[index],
                            adjustments[batch],
                        ),
                    );
                }
            }
        }
    }
    Ok((output, cache))
}

fn accumulate_crop_gradient(
    gradient: &mut Tensor,
    batch: usize,
    channel: usize,
    x: isize,
    y: isize,
    value: f32,
) {
    if x >= 0 && y >= 0 && x < gradient.shape.width as isize && y < gradient.shape.height as isize {
        let previous = gradient.get(batch, channel, y as usize, x as usize);
        gradient.set(batch, channel, y as usize, x as usize, previous + value);
    }
}

fn backward_crop(
    cache: &CropForwardCache,
    input_shape: TensorShape,
    output_gradient: &Tensor,
) -> Result<Tensor> {
    let expected =
        output_gradient.shape.batch * output_gradient.shape.width * output_gradient.shape.height;
    if cache.source_x.len() != expected
        || cache.source_y.len() != expected
        || cache.input_scales.len() != output_gradient.shape.batch
        || cache.input_offsets.len() != output_gradient.shape.batch
    {
        return Err(Error::InvalidConfig(
            "crop cache shape does not match output gradient".to_owned(),
        ));
    }
    let mut input_gradient = Tensor::zeros(input_shape);
    for batch in 0..output_gradient.shape.batch {
        for channel in 0..output_gradient.shape.channels {
            for oy in 0..output_gradient.shape.height {
                for ox in 0..output_gradient.shape.width {
                    let index = (batch * output_gradient.shape.height + oy)
                        * output_gradient.shape.width
                        + ox;
                    let x = cache.source_x[index];
                    let y = cache.source_y[index];
                    let ix = x.floor() as isize;
                    let iy = y.floor() as isize;
                    let dx = x - ix as f32;
                    let dy = y - iy as f32;
                    let grad =
                        output_gradient.get(batch, channel, oy, ox) * cache.input_scales[batch];
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix,
                        iy,
                        grad * (1.0 - dy) * (1.0 - dx),
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix,
                        iy + 1,
                        grad * dy * (1.0 - dx),
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix + 1,
                        iy,
                        grad * (1.0 - dy) * dx,
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix + 1,
                        iy + 1,
                        grad * dy * dx,
                    );
                }
            }
        }
    }
    Ok(input_gradient)
}

fn forward_scale_channels(
    spec: &crate::model::ScaleChannelsSpec,
    input: &Tensor,
    source: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    if source.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "scale_channels source/output mismatch: {} vs {}",
            source.shape, output_shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for c in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let scale = if spec.scale_wh {
                        input.get(b, 0, y, x)
                    } else {
                        input.get(b, c, 0, 0)
                    };
                    let pre_activation = scale * source.get(b, c, y, x);
                    activation_input.set(b, c, y, x, pre_activation);
                    output.set(
                        b,
                        c,
                        y,
                        x,
                        activate(spec.activation.clone(), pre_activation),
                    );
                }
            }
        }
    }
    Ok((output, Some(activation_input)))
}

fn forward_sam(
    spec: &crate::model::SamSpec,
    input: &Tensor,
    source: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    if input.shape != output_shape || source.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "sam shape mismatch: input {}, source {}, output {}",
            input.shape, source.shape, output_shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    for ((out, pre_activation), (input_value, source_value)) in output
        .data
        .iter_mut()
        .zip(&mut activation_input.data)
        .zip(input.data.iter().zip(&source.data))
    {
        *pre_activation = *input_value * *source_value;
        *out = activate(spec.activation.clone(), *pre_activation);
    }
    Ok((output, Some(activation_input)))
}

#[allow(clippy::too_many_arguments)]
fn backward_connected(
    spec: &ConnectedSpec,
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    output: &Tensor,
    activation_input: Option<&Tensor>,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, ConnectedParams)> {
    // Return parameter-shaped gradients instead of mutating the model.  The
    // caller can use this for a standalone backward pass or merge it into
    // persistent Darknet-style train buffers for optimizer updates.
    let inputs = input.shape.element_count();
    if output.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "connected output shape mismatch: expected {}, got {}",
            output_shape, output.shape
        )));
    }
    let expected_weights = spec.output * inputs;
    if params.biases.len() != spec.output || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "connected parameter shape mismatch: expected {} biases and {} weights, got {} and {}",
            spec.output,
            expected_weights,
            params.biases.len(),
            params.weights.len()
        )));
    }
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "connected",
        activation_input,
    )?;

    let needed = spec.output + expected_weights;
    if workspace.len() < needed {
        workspace.resize(needed, 0.0);
    }
    workspace[..needed].fill(0.0);
    let (bias_grad, weight_grad) = workspace[..needed].split_at_mut(spec.output);
    let mut input_grad = Tensor::zeros(input.shape);
    let mut batch_norm_grad = params
        .batch_norm
        .as_ref()
        .map(|batch_norm| BatchNormParams {
            biases: vec![0.0; batch_norm.biases.len()],
            scales: vec![0.0; batch_norm.scales.len()],
            rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
            rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
        });

    for b in 0..input.shape.batch {
        let input_base = b * inputs;
        let output_base = b * spec.output;
        for (out_idx, bias_grad_slot) in bias_grad.iter_mut().enumerate().take(spec.output) {
            let mut grad_value = delta.data[output_base + out_idx];
            if let Some(batch_norm) = params.batch_norm.as_ref() {
                let raw_value = connected_raw_value(params, input, b, out_idx);
                let inv_std =
                    1.0 / (batch_norm.rolling_variance[out_idx] + BATCH_NORM_EPSILON).sqrt();
                let normalized = (raw_value - batch_norm.rolling_mean[out_idx]) * inv_std;
                *bias_grad_slot += grad_value;
                if let Some(batch_norm_grad) = batch_norm_grad.as_mut() {
                    batch_norm_grad.scales[out_idx] += grad_value * normalized;
                }
                grad_value *= batch_norm.scales[out_idx] * inv_std;
            } else {
                *bias_grad_slot += grad_value;
            }

            let weight_base = out_idx * inputs;
            for input_idx in 0..inputs {
                let input_value = input.data[input_base + input_idx];
                weight_grad[weight_base + input_idx] += input_value * grad_value;
                input_grad.data[input_base + input_idx] +=
                    params.weights[weight_base + input_idx] * grad_value;
            }
        }
    }

    Ok((
        input_grad,
        ConnectedParams {
            biases: bias_grad.to_vec(),
            batch_norm: batch_norm_grad,
            weights: weight_grad.to_vec(),
        },
    ))
}

fn connected_raw_value(
    params: &ConnectedParams,
    input: &Tensor,
    batch: usize,
    output_index: usize,
) -> f32 {
    let inputs = input.shape.element_count();
    let input_base = batch * inputs;
    let weight_base = output_index * inputs;
    input.data[input_base..input_base + inputs]
        .iter()
        .zip(&params.weights[weight_base..weight_base + inputs])
        .map(|(input, weight)| input * weight)
        .sum()
}

fn raw_connected_output(
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    let mut output = Tensor::zeros(output_shape);
    for batch in 0..input.shape.batch {
        for output_index in 0..output_shape.channels {
            output.data[batch * output_shape.channels + output_index] =
                connected_raw_value(params, input, batch, output_index);
        }
    }
    output
}

fn backward_softmax(output: &Tensor, output_gradient: &Tensor, groups: usize) -> Result<Tensor> {
    if output.shape != output_gradient.shape {
        return Err(Error::InvalidConfig(format!(
            "softmax backward gradient shape mismatch: expected {}, got {}",
            output.shape, output_gradient.shape
        )));
    }
    let groups = groups.max(1);
    let per_batch = output.shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    let group_size = per_batch / groups;
    let mut input_grad = Tensor::zeros(output.shape);
    for b in 0..output.shape.batch {
        let base = b * per_batch;
        for group in 0..groups {
            let start = base + group * group_size;
            let end = start + group_size;
            let dot = output.data[start..end]
                .iter()
                .zip(&output_gradient.data[start..end])
                .map(|(probability, gradient)| probability * gradient)
                .sum::<f32>();
            for index in start..end {
                input_grad.data[index] = output.data[index] * (output_gradient.data[index] - dot);
            }
        }
    }
    Ok(input_grad)
}

fn backward_dropout(output_gradient: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if let Some(mask) = mask {
        if mask.shape != output_gradient.shape {
            return Err(Error::InvalidConfig(format!(
                "dropout mask shape mismatch: expected {}, got {}",
                output_gradient.shape, mask.shape
            )));
        }
        let mut input_grad = Tensor::zeros(output_gradient.shape);
        for ((dst, gradient), multiplier) in input_grad
            .data
            .iter_mut()
            .zip(&output_gradient.data)
            .zip(&mask.data)
        {
            *dst = *gradient * *multiplier;
        }
        Ok(input_grad)
    } else {
        Ok(output_gradient.clone())
    }
}

fn backward_scale_channels(
    spec: &crate::model::ScaleChannelsSpec,
    input: &Tensor,
    source: &Tensor,
    output: &Tensor,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "scale_channels",
        activation_input,
    )?;
    let mut input_grad = Tensor::zeros(input.shape);
    let mut source_grad = Tensor::zeros(source.shape);
    for b in 0..output.shape.batch {
        for c in 0..output.shape.channels {
            for y in 0..output.shape.height {
                for x in 0..output.shape.width {
                    let index = output.offset(b, c, y, x);
                    let scale = if spec.scale_wh {
                        input.get(b, 0, y, x)
                    } else {
                        input.get(b, c, 0, 0)
                    };
                    let source_value = source.get(b, c, y, x);
                    let grad = delta.data[index];
                    let source_index = source.offset(b, c, y, x);
                    source_grad.data[source_index] = grad * scale;
                    if spec.scale_wh {
                        let input_index = input.offset(b, 0, y, x);
                        input_grad.data[input_index] += grad * source_value;
                    } else {
                        let input_index = input.offset(b, c, 0, 0);
                        input_grad.data[input_index] += grad * source_value;
                    }
                }
            }
        }
    }
    Ok((input_grad, source_grad))
}

fn backward_sam(
    spec: &crate::model::SamSpec,
    input: &Tensor,
    source: &Tensor,
    output: &Tensor,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "sam",
        activation_input,
    )?;
    if input.shape != source.shape || input.shape != output.shape {
        return Err(Error::InvalidConfig(format!(
            "sam backward shape mismatch: input {}, source {}, output {}",
            input.shape, source.shape, output.shape
        )));
    }
    let mut input_grad = Tensor::zeros(input.shape);
    let mut source_grad = Tensor::zeros(source.shape);
    for index in 0..delta.data.len() {
        input_grad.data[index] = delta.data[index] * source.data[index];
        source_grad.data[index] = delta.data[index] * input.data[index];
    }
    Ok((input_grad, source_grad))
}

fn backward_reorg(
    spec: &crate::model::ReorgSpec,
    output_gradient: &Tensor,
    input_shape: TensorShape,
) -> Result<Tensor> {
    forward_reorg(spec.stride, !spec.reverse, output_gradient, input_shape)
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
            biases: vec![0.0; batch_norm.biases.len()],
            scales: vec![0.0; batch_norm.scales.len()],
            rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
            rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
        });

    for b in 0..input.shape.batch {
        for (filter, bias_grad_value) in bias_grad.iter_mut().enumerate() {
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
                        *bias_grad_value += grad_value;
                        if let Some(batch_norm_grad) = batch_norm_grad.as_mut() {
                            batch_norm_grad.scales[filter] += grad_value * normalized;
                        }
                        grad_value *= batch_norm.scales[filter] * inv_std;
                    } else {
                        *bias_grad_value += grad_value;
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
    zero_nonmax: bool,
) -> Result<Tensor> {
    let mut input_grad = Tensor::zeros(input.shape);
    let mut is_max = vec![false; input.data.len()];
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
                            is_max[idx] = true;
                        }
                    }
                }
            }
        }
    }
    if zero_nonmax && !average {
        for (index, chosen) in is_max.iter().enumerate() {
            if !*chosen {
                input_grad.data[index] = 0.0;
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
    match activation {        Activation::Linear => Ok(1.0),
        Activation::Logistic => Ok(output_value * (1.0 - output_value)),
        Activation::Loggy => Ok(0.5 * (1.0 - output_value * output_value)),
        Activation::Relu => Ok((output_value > 0.0) as i32 as f32),
        Activation::Leaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
        Activation::Tanh => Ok(1.0 - output_value * output_value),
        // relu6(x) is in (0, 6) exactly when x is, so the output form is exact.
        Activation::Relu6 => Ok((output_value > 0.0 && output_value < 6.0) as i32 as f32),
        // ramp(x) > 0 exactly when x > 0.
        Activation::Ramp => Ok((output_value > 0.0) as i32 as f32 + 0.1),
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
        Activation::Gelu => Ok(crate::activation_math::gelu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Elu => Ok(crate::activation_math::elu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Selu => Ok(crate::activation_math::selu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Plse => Ok(crate::activation_math::plse_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Hardtan => Ok(crate::activation_math::hardtan_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Lhtan => Ok(crate::activation_math::lhtan_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Stair => Ok(crate::activation_math::stair_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::HardMish => Ok(crate::activation_math::hard_mish_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        // relie / revleaky are leaky-shaped: the output has the sign of the input.
        Activation::Relie => Ok(if output_value > 0.0 { 1.0 } else { 0.01 }),
        Activation::Revleaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
        Activation::Other(name) => Err(Error::UnsupportedLayer {
            layer_index: 0,
            name: format!("{layer_name}_{name}_backward"),
        }),
    }
}

fn cached_activation_input(
    activation: &Activation,
    activation_input: Option<f32>,
    layer_name: &str,
) -> Result<f32> {
    activation_input.ok_or_else(|| {
        Error::InvalidConfig(format!(
            "{layer_name} backward requires cached activation_input for {activation:?}"
        ))
    })
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
        Activation::Loggy => 2.0 / (1.0 + (-value).exp()) - 1.0,
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
        Activation::Relu6 => value.clamp(0.0, 6.0),
        Activation::Ramp => value * (value > 0.0) as i32 as f32 + 0.1 * value,
        Activation::Gelu => crate::activation_math::gelu(value),
        Activation::Elu => crate::activation_math::elu(value),
        Activation::Selu => crate::activation_math::selu(value),
        Activation::Relie => crate::activation_math::relie(value),
        Activation::Plse => crate::activation_math::plse(value),
        Activation::Hardtan => crate::activation_math::hardtan(value),
        Activation::Lhtan => crate::activation_math::lhtan(value),
        Activation::Stair => crate::activation_math::stair(value),
        Activation::HardMish => crate::activation_math::hard_mish(value),
        Activation::Revleaky => {
            if value > 0.0 {
                value
            } else {
                0.1 * value
            }
        }
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

fn forward_route_compact(
    spec: &RouteSpec,
    layer_index: usize,
    outputs: &[Option<Tensor>],
    output_shape: TensorShape,
) -> Result<Tensor> {
    let mut output = Tensor::zeros(output_shape);
    let mut out_channel = 0usize;

    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source =
            outputs
                .get(source_index)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidRoute {
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

fn shortcut_sources<'a>(
    spec: &crate::model::ShortcutSpec,
    layer_index: usize,
    outputs: &'a [Tensor],
    batch: usize,
) -> Result<Vec<&'a Tensor>> {
    spec.from
        .iter()
        .map(|raw_index| {
            let source_index = resolve_relative_layer(layer_index, *raw_index)?;
            let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
                layer_index,
                reference: *raw_index,
            })?;
            if source.shape.batch != batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{layer_index} mixes incompatible batch shapes"
                )));
            }
            Ok(source)
        })
        .collect()
}

fn forward_shortcut_weighted(
    spec: &crate::model::ShortcutSpec,
    params: &ShortcutParams,
    input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<(Tensor, Option<Tensor>)> {
    let sources = shortcut_sources(spec, layer_index, outputs, input.shape.batch)?;
    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
    let pre = crate::shortcut_weights::weighted_forward(
        &input.data,
        &slices,
        input.shape.batch,
        input.shape.element_count(),
        &params.weights,
        spec.weights_normalization,
        spec.weight_slots(input.shape.channels),
    );
    let output: Vec<f32> = pre.iter().map(|v| activate(spec.activation.clone(), *v)).collect();
    Ok((
        Tensor::from_data(input.shape, output)?,
        Some(Tensor::from_data(input.shape, pre)?),
    ))
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
        if source.shape.batch != output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "shortcut layer #{layer_index} mixes incompatible batch shapes {} and {}",
                output.shape, source.shape
            )));
        }
        let output_stride = output.shape.element_count();
        let source_stride = source.shape.element_count();
        let copied = output_stride.min(source_stride);
        for batch in 0..output.shape.batch {
            let output_offset = batch * output_stride;
            let source_offset = batch * source_stride;
            for index in 0..copied {
                output.data[output_offset + index] += source.data[source_offset + index];
            }
        }
    }
    let activation_input = output.clone();
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok((output, Some(activation_input)))
}

fn forward_shortcut_compact(
    spec: &crate::model::ShortcutSpec,
    current_input: &Tensor,
    layer_index: usize,
    outputs: &[Option<Tensor>],
) -> Result<Tensor> {
    let mut output = current_input.clone();
    for raw_index in &spec.from {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source =
            outputs
                .get(source_index)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: *raw_index,
                })?;
        if source.shape.batch != output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "shortcut layer #{layer_index} mixes incompatible batch shapes {} and {}",
                output.shape, source.shape
            )));
        }
        let output_stride = output.shape.element_count();
        let source_stride = source.shape.element_count();
        let copied = output_stride.min(source_stride);
        for batch in 0..output.shape.batch {
            let output_offset = batch * output_stride;
            let source_offset = batch * source_stride;
            for index in 0..copied {
                output.data[output_offset + index] += source.data[source_offset + index];
            }
        }
    }
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok(output)
}

fn shortcut_source_gradient(output_gradient: &Tensor, source_shape: TensorShape) -> Result<Tensor> {
    if output_gradient.shape.batch != source_shape.batch {
        return Err(Error::InvalidConfig(format!(
            "shortcut gradient batch {} is incompatible with source batch {}",
            output_gradient.shape, source_shape
        )));
    }

    let mut source_gradient = Tensor::zeros(source_shape);
    let output_stride = output_gradient.shape.element_count();
    let source_stride = source_shape.element_count();
    let copied = output_stride.min(source_stride);
    for batch in 0..source_shape.batch {
        let output_offset = batch * output_stride;
        let source_offset = batch * source_stride;
        source_gradient.data[source_offset..source_offset + copied]
            .copy_from_slice(&output_gradient.data[output_offset..output_offset + copied]);
    }
    Ok(source_gradient)
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
                // Inverse of the non-reverse mapping below (Darknet reorg_cpu:
                // small channel k <-> big channel k % out_c, offset k / out_c).
                let out_c = c % output.shape.channels;
                let offset = c / output.shape.channels;
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
            .join("../../cfg")
            .join(name)
    }

    #[test]
    fn legacy_detection_layer_preserves_forward_and_backward_tensors() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "detection.cfg",
                r#"
                    [net]
                    width=1
                    height=1
                    channels=7
                    batch=1

                    [detection]
                    classes=2
                    coords=4
                    side=1
                    num=1
                "#,
            )
            .expect("cfg parses"),
        )
        .expect("network constructs");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0; 7])
            .expect("input tensor");
        let forward = executable
            .forward_training(&input)
            .expect("forward succeeds");
        assert_eq!(forward.final_output(), Some(&input));

        let gradient = Tensor::filled(input.shape, 0.25);
        let backward = executable
            .backward(&input, &forward, &gradient)
            .expect("backward succeeds");
        assert_eq!(backward.input_gradient, gradient);
    }

    #[test]
    fn crop_layer_center_samples_and_scatters_bilinear_gradients() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "crop.cfg",
                r#"
                    [net]
                    width=4
                    height=3
                    channels=1
                    batch=1

                    [crop]
                    crop_width=2
                    crop_height=2
                    noadjust=1
                "#,
            )
            .expect("cfg parses"),
        )
        .expect("network constructs");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            (0..12).map(|value| value as f32).collect(),
        )
        .expect("input tensor");
        let forward = executable.forward(&input).expect("crop forward succeeds");
        assert_eq!(
            forward.final_output().expect("crop output").data,
            vec![3.0, 4.0, 7.0, 8.0]
        );

        let gradient = Tensor::filled(forward.final_output().expect("output").shape, 1.0);
        let backward = executable
            .backward(&input, &forward, &gradient)
            .expect("crop backward succeeds");
        assert_eq!(
            backward.input_gradient.data,
            vec![0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0,]
        );
    }

    #[test]
    fn crop_layer_applies_darknet_level_adjustment_before_sampling() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "crop-levels.cfg",
                r#"
                    [net]
                    width=1
                    height=1
                    channels=3
                    batch=1

                    [crop]
                    crop_width=1
                    crop_height=1
                    saturation=1
                    exposure=1
                    shift=0
                "#,
            )
            .expect("cfg parses"),
        )
        .expect("network constructs");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![0.25, 0.5, 0.75])
            .expect("input tensor");
        let output = executable
            .forward(&input)
            .expect("crop forward succeeds")
            .final_output()
            .expect("crop output")
            .clone();
        assert_eq!(output.data, vec![-0.5, 0.0, 0.5]);
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
            if activation_input.data.len() != delta.data.len() {
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
                false,
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
            F32ActivationKind::Relu6 => Activation::Relu6,
            F32ActivationKind::Ramp => Activation::Ramp,
            F32ActivationKind::Gelu => Activation::Gelu,
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
            F32ActivationKind::Relu6 => (value > 0.0 && value < 6.0) as i32 as f32,
            F32ActivationKind::Ramp => (value > 0.0) as i32 as f32 + 0.1,
            F32ActivationKind::Gelu => crate::activation_math::gelu_grad(value),
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
            if activation_input.data.len() != delta.data.len() {
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

    fn squared_error_loss(executable: &ExecutableNetwork, input: &Tensor, target: &[f32]) -> f32 {
        let forward = executable
            .forward(input)
            .expect("loss evaluation forward should succeed");
        let output = forward
            .final_output()
            .expect("loss evaluation needs a final output");
        assert_eq!(output.data.len(), target.len());
        output
            .data
            .iter()
            .zip(target)
            .map(|(prediction, target)| {
                let error = prediction - target;
                0.5 * error * error
            })
            .sum()
    }

    fn assert_slice_close(name: &str, expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len(), "{name} length mismatch");
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            assert!(
                (expected - actual).abs() <= tolerance,
                "{name}[{index}] differs: expected {expected:.8e}, actual {actual:.8e}, tolerance {tolerance:.8e}"
            );
        }
    }

    /// Independent analytical reference for a Darknet NCHW cross-correlation.
    ///
    /// The backward equations used here are closed-form chain-rule equations:
    ///
    /// ```text
    /// dL/dw[f,c,ky,kx] += x[b,c,iy,ix] * dL/dy[b,f,oy,ox]
    /// dL/db[f]          += dL/dy[b,f,oy,ox]
    /// dL/dx[b,c,iy,ix]  += w[f,c,ky,kx] * dL/dy[b,f,oy,ox]
    /// ```
    ///
    /// Keeping this oracle separate from the runtime implementation makes the
    /// test catch indexing, padding, grouping, and gradient-sign mistakes.
    struct AnalyticalConvCase<'a> {
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        filters: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        input: &'a [f32],
        weights: &'a [f32],
        biases: &'a [f32],
    }

    fn analytical_convolution_output(case: &AnalyticalConvCase<'_>) -> Vec<f32> {
        let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
        let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
        let channels_per_group = case.channels / case.groups;
        let filters_per_group = case.filters / case.groups;
        let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
            (((batch * case.channels + channel) * case.height) + y) * case.width + x
        };
        let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
            (((batch * case.filters + filter) * output_height) + y) * output_width + x
        };
        let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
            (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
        };
        let mut output = vec![0.0; case.batch * case.filters * output_height * output_width];

        for batch in 0..case.batch {
            for filter in 0..case.filters {
                let group = filter / filters_per_group;
                for y in 0..output_height {
                    for x in 0..output_width {
                        let mut value = case.biases[filter];
                        for channel_in_group in 0..channels_per_group {
                            let channel = group * channels_per_group + channel_in_group;
                            for ky in 0..case.kernel {
                                for kx in 0..case.kernel {
                                    let source_y = y * case.stride + ky;
                                    let source_x = x * case.stride + kx;
                                    if source_y < case.padding
                                        || source_x < case.padding
                                        || source_y >= case.height + case.padding
                                        || source_x >= case.width + case.padding
                                    {
                                        continue;
                                    }
                                    let source_y = source_y - case.padding;
                                    let source_x = source_x - case.padding;
                                    value += case.input
                                        [input_index(batch, channel, source_y, source_x)]
                                        * case.weights
                                            [weight_index(filter, channel_in_group, ky, kx)];
                                }
                            }
                        }
                        output[output_index(batch, filter, y, x)] = value;
                    }
                }
            }
        }
        output
    }

    fn analytical_convolution_gradients(
        case: &AnalyticalConvCase<'_>,
        output_gradient: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
        let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
        let channels_per_group = case.channels / case.groups;
        let filters_per_group = case.filters / case.groups;
        let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
            (((batch * case.channels + channel) * case.height) + y) * case.width + x
        };
        let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
            (((batch * case.filters + filter) * output_height) + y) * output_width + x
        };
        let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
            (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
        };
        let mut input_gradient = vec![0.0; case.input.len()];
        let mut weight_gradient = vec![0.0; case.weights.len()];
        let mut bias_gradient = vec![0.0; case.biases.len()];

        assert_eq!(
            output_gradient.len(),
            case.batch * case.filters * output_height * output_width
        );
        for batch in 0..case.batch {
            for filter in 0..case.filters {
                let group = filter / filters_per_group;
                for y in 0..output_height {
                    for x in 0..output_width {
                        let gradient = output_gradient[output_index(batch, filter, y, x)];
                        bias_gradient[filter] += gradient;
                        for channel_in_group in 0..channels_per_group {
                            let channel = group * channels_per_group + channel_in_group;
                            for ky in 0..case.kernel {
                                for kx in 0..case.kernel {
                                    let source_y = y * case.stride + ky;
                                    let source_x = x * case.stride + kx;
                                    if source_y < case.padding
                                        || source_x < case.padding
                                        || source_y >= case.height + case.padding
                                        || source_x >= case.width + case.padding
                                    {
                                        continue;
                                    }
                                    let source_y = source_y - case.padding;
                                    let source_x = source_x - case.padding;
                                    let input_offset =
                                        input_index(batch, channel, source_y, source_x);
                                    let weight_offset =
                                        weight_index(filter, channel_in_group, ky, kx);
                                    weight_gradient[weight_offset] +=
                                        case.input[input_offset] * gradient;
                                    input_gradient[input_offset] +=
                                        case.weights[weight_offset] * gradient;
                                }
                            }
                        }
                    }
                }
            }
        }
        (input_gradient, weight_gradient, bias_gradient)
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
    fn maxpool_zero_nonmax_zeroes_non_argmax_inputs() {
        let input = Tensor::from_data(
            TensorShape::new(3, 3, 1, 1),
            vec![1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
        )
        .expect("input");
        let mut grad = Tensor::zeros(TensorShape::new(2, 2, 1, 1));
        for (i, value) in grad.data.iter_mut().enumerate() {
            *value = (i + 1) as f32;
        }
        let exact = super::backward_maxpool(2, 1, 1, 0, &input, &grad, false, false).unwrap();
        let zeroed = super::backward_maxpool(2, 1, 1, 0, &input, &grad, false, true).unwrap();
        // Every window's argmax is the centre (1,1) -> flat index 4.
        assert_eq!(zeroed.data[4], exact.data[4]);
        for index in [0usize, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(zeroed.data[index], 0.0, "index {index}");
        }
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

        let path = std::env::temp_dir().join("darknet-runtime-conv.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-route.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-conv.weights");
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
    fn convolutional_analytical_gradients_and_sgd_step() {
        // This test covers the complete CPU training contract for a padded
        // multi-channel convolution.  The analytical reference validates the
        // forward pass and dL/dx, dL/dw, and dL/db independently of the runtime
        // implementation.  The second half then uses Darknet's negative
        // upstream-gradient convention and verifies one real SGD update.
        let cfg = ConfigFile::parse_str(
            "conv-gradient.cfg",
            r#"
                [net]
                width=3
                height=2
                channels=2
                batch=2

                [convolutional]
                filters=2
                size=3
                stride=1
                pad=1
                activation=linear
            "#,
        )
        .expect("convolution config should parse");
        let network = Network::from_config(cfg).expect("convolution network should build");
        let mut executable = ExecutableNetwork::new(network);

        let weights = (0..36)
            .map(|index| 0.03 * (index as f32 - 17.0))
            .collect::<Vec<_>>();
        let biases = vec![0.17, -0.23];
        if let LayerParams::Convolutional(params) = &mut executable.params[0] {
            params.weights.clone_from_slice(&weights);
            params.biases.clone_from_slice(&biases);
        } else {
            panic!("expected a convolutional parameter block");
        }

        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            (0..24).map(|index| (index as f32 - 11.0) / 7.0).collect(),
        )
        .expect("convolution input should have the expected shape");
        let analytical_case = AnalyticalConvCase {
            batch: 2,
            channels: 2,
            height: 2,
            width: 3,
            filters: 2,
            kernel: 3,
            stride: 1,
            padding: 1,
            groups: 1,
            input: &input.data,
            weights: &weights,
            biases: &biases,
        };
        let analytical_output = analytical_convolution_output(&analytical_case);
        let forward = executable
            .forward(&input)
            .expect("convolution forward should succeed");
        let output = forward.final_output().expect("convolution output");
        assert_slice_close(
            "CPU convolution forward",
            &analytical_output,
            &output.data,
            2.0e-6,
        );
        let target = analytical_output
            .iter()
            .enumerate()
            .map(|(index, _)| 0.07 * index as f32 - 0.4)
            .collect::<Vec<_>>();
        let upstream = analytical_output
            .iter()
            .zip(&target)
            .map(|(prediction, target)| prediction - target)
            .collect::<Vec<_>>();
        let upstream_tensor = Tensor::from_data(output.shape, upstream.clone())
            .expect("upstream gradient should have the output shape");
        let backward = executable
            .backward(&input, &forward, &upstream_tensor)
            .expect("convolution backward should succeed");
        let (weight_gradient, bias_gradient) = match &backward.param_gradients[0] {
            LayerParams::Convolutional(gradient) => {
                (gradient.weights.clone(), gradient.biases.clone())
            }
            other => panic!("unexpected convolution gradient block: {other:?}"),
        };
        // For L = 1/2 * sum((y - target)^2), dL/dy = y - target.  These
        // expected gradients come from the closed-form convolution equations
        // above, with no parameter or input perturbation.
        let (expected_input_gradient, expected_weight_gradient, expected_bias_gradient) =
            analytical_convolution_gradients(&analytical_case, &upstream);
        assert_slice_close(
            "convolution weight analytical gradient",
            &expected_weight_gradient,
            &weight_gradient,
            2.0e-6,
        );
        assert_slice_close(
            "convolution bias analytical gradient",
            &expected_bias_gradient,
            &bias_gradient,
            2.0e-6,
        );
        assert_slice_close(
            "convolution input analytical gradient",
            &expected_input_gradient,
            &backward.input_gradient.data,
            2.0e-6,
        );

        let before_loss = squared_error_loss(&executable, &input, &target);
        let before_weights = match &executable.params[0] {
            LayerParams::Convolutional(params) => params.weights.clone(),
            other => panic!("unexpected convolution parameters: {other:?}"),
        };
        let before_biases = match &executable.params[0] {
            LayerParams::Convolutional(params) => params.biases.clone(),
            other => panic!("unexpected convolution parameters: {other:?}"),
        };
        let descent_seed =
            Tensor::from_data(output.shape, upstream.iter().map(|value| -*value).collect())
                .expect("descent seed should have the output shape");
        executable.zero_train_buffers();
        let training_forward = executable
            .forward_training(&input)
            .expect("training forward should succeed");
        executable
            .backward_accumulate(&input, &training_forward, &descent_seed)
            .expect("training backward should accumulate");

        // Darknet accumulates -dL/dtheta in its update buffers and adds that
        // descent direction during update_weights().
        if let LayerTrainBuffers::Convolutional(train) = &executable.train_buffers[0] {
            let expected_weight_update = expected_weight_gradient
                .iter()
                .map(|gradient| -*gradient)
                .collect::<Vec<_>>();
            let expected_bias_update = expected_bias_gradient
                .iter()
                .map(|gradient| -*gradient)
                .collect::<Vec<_>>();
            assert_slice_close(
                "Darknet convolution weight descent buffer",
                &expected_weight_update,
                &train.weight_updates,
                2.0e-6,
            );
            assert_slice_close(
                "Darknet convolution bias descent buffer",
                &expected_bias_update,
                &train.bias_updates,
                2.0e-6,
            );
        } else {
            panic!("expected convolutional training buffers");
        }

        let learning_rate = 1.0e-3;
        let batch = executable.network.net.batch;
        executable.update_weights(learning_rate, 0.0, 0.0, batch);
        let expected_weights = before_weights
            .iter()
            .zip(&expected_weight_gradient)
            .map(|(weight, gradient)| weight - learning_rate * gradient / batch as f32)
            .collect::<Vec<_>>();
        let expected_biases = before_biases
            .iter()
            .zip(&expected_bias_gradient)
            .map(|(bias, gradient)| bias - learning_rate * gradient / batch as f32)
            .collect::<Vec<_>>();
        match &executable.params[0] {
            LayerParams::Convolutional(params) => {
                assert_slice_close(
                    "CPU convolution SGD weights",
                    &expected_weights,
                    &params.weights,
                    1.0e-6,
                );
                assert_slice_close(
                    "CPU convolution SGD biases",
                    &expected_biases,
                    &params.biases,
                    1.0e-6,
                );
            }
            other => panic!("unexpected convolution parameters after update: {other:?}"),
        }
        let after_loss = squared_error_loss(&executable, &input, &target);
        assert!(
            after_loss < before_loss,
            "one Darknet SGD step must lower the loss: before={before_loss}, after={after_loss}"
        );
    }

    #[test]
    fn fake_gpu_convolution_backward_and_sgd_match_cpu_training() {
        // This uses the same F32TensorOpsBackend seam as the real cuTile
        // bridge.  The fake backend executes the scalar reference equations,
        // which lets CI verify device dispatch, gradient accumulation, and
        // the device SGD API without requiring a CUDA driver.
        let cfg = ConfigFile::parse_str(
            "gpu-conv-training.cfg",
            r#"
                [net]
                width=3
                height=2
                channels=2
                batch=2

                [convolutional]
                filters=2
                size=3
                stride=1
                pad=1
                activation=linear
            "#,
        )
        .expect("convolution config should parse");
        let network = Network::from_config(cfg).expect("convolution network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        let weights = (0..36)
            .map(|index| 0.02 * (index as f32 - 13.0))
            .collect::<Vec<_>>();
        let biases = vec![0.11, -0.19];
        for params in [&mut cpu.params[0], &mut gpu.params[0]] {
            let LayerParams::Convolutional(params) = params else {
                panic!("expected a convolutional parameter block");
            };
            params.weights.clone_from_slice(&weights);
            params.biases.clone_from_slice(&biases);
        }

        let input = Tensor::from_data(
            cpu.network.net.input_shape(),
            (0..24).map(|index| (index as f32 - 8.0) / 9.0).collect(),
        )
        .expect("convolution input should have the expected shape");
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU training forward should succeed");
        let cpu_output = cpu_forward.final_output().expect("CPU output");
        let target = cpu_output
            .data
            .iter()
            .enumerate()
            .map(|(index, _)| 0.03 * index as f32 - 0.2)
            .collect::<Vec<_>>();
        let descent_seed = Tensor::from_data(
            cpu_output.shape,
            cpu_output
                .data
                .iter()
                .zip(&target)
                .map(|(prediction, target)| target - prediction)
                .collect(),
        )
        .expect("GPU seed should have the output shape");
        cpu.zero_train_buffers();
        cpu.backward_accumulate(&input, &cpu_forward, &descent_seed)
            .expect("CPU training backward should succeed");
        cpu.update_weights(1.0e-3, 0.0, 0.0, cpu.network.net.batch);

        gpu.zero_train_buffers_gpu()
            .expect("GPU training buffers should initialize");
        let gpu_forward = gpu
            .forward_with_state(&input)
            .expect("GPU training forward should succeed");
        assert_slice_close(
            "fake GPU convolution forward",
            &cpu_forward.final_output().expect("CPU output").data,
            &gpu_forward
                .cpu_forward
                .final_output()
                .expect("GPU output")
                .data,
            1.0e-6,
        );
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[Some(descent_seed)])
            .expect("GPU convolution backward should succeed");
        gpu.update_weights_gpu(1.0e-3, 0.0, 0.0, gpu.network.net.batch)
            .expect("GPU convolution SGD update should succeed");

        let cpu_params = match &cpu.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected CPU parameters: {other:?}"),
        };
        let gpu_params = match &gpu.params[0] {
            LayerParams::Convolutional(params) => params,
            other => panic!("unexpected GPU parameters: {other:?}"),
        };
        assert_slice_close(
            "fake GPU convolution SGD weights",
            &cpu_params.weights,
            &gpu_params.weights,
            1.0e-6,
        );
        assert_slice_close(
            "fake GPU convolution SGD biases",
            &cpu_params.biases,
            &gpu_params.biases,
            1.0e-6,
        );
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-route.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-shared.weights");
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
            time_steps: 1,
            subdivisions: 1,
            learning_rate: 0.01,
            learning_rate_min: 0.00001,
            momentum: 0.9,
            decay: 0.0005,
            max_batches: 100,
            burn_in: 10,
            power: 4.0,
            step: 1,
            scale: 1.0,
            gamma: 1.0,
            sgdr_cycle: 100,
            sgdr_mult: 2,
            policy: LrPolicy::Constant,
            steps: Vec::new(),
            scales: Vec::new(),
            equidistant_point: 0,
            badlabels_rejection_percentage: 0.0,
            num_sigmas_reject_badlabels: 0.0,
            adam: false,
            adam_b1: 0.9,
            adam_b2: 0.999,
            adam_eps: 1e-6,
            decoupled_decay: false,
            loss_scale: 1.0,
            try_fix_nan: false,
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-swish.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-bn.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-backward-multihead.weights");
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
    fn standalone_batchnorm_runs_inference_training_backward_update_and_weights() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1
                batch=2

                [batchnorm]
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        assert_eq!(
            network.layers[0]
                .weights()
                .expect("weights")
                .parameter_count(),
            4
        );
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::BatchNorm(params) = &mut executable.params[0] else {
            panic!("expected standalone batchnorm parameters");
        };
        params.biases[0] = 0.5;
        params.scales[0] = 2.0;
        params.rolling_mean[0] = 10.0;
        params.rolling_variance[0] = 4.0;

        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 3.0, 5.0, 7.0],
        )
        .expect("input tensor");
        let inference = executable
            .forward(&input)
            .expect("inference should succeed");
        let inference_inv_std = 1.0 / (4.0f32 + super::BATCH_NORM_EPSILON).sqrt();
        for (actual, expected) in inference
            .final_output()
            .expect("output")
            .data
            .iter()
            .zip([1.0f32, 3.0, 5.0, 7.0])
        {
            let expected = (expected - 10.0) * inference_inv_std * 2.0 + 0.5;
            assert!((actual - expected).abs() < 1e-5);
        }

        let training = executable
            .forward_training(&input)
            .expect("training forward should succeed");
        let output = training.final_output().expect("output");
        let inv_std = 1.0 / (5.0f32 + super::BATCH_NORM_EPSILON).sqrt();
        let expected = [
            (1.0 - 4.0) * inv_std * 2.0 + 0.5,
            (3.0 - 4.0) * inv_std * 2.0 + 0.5,
            (5.0 - 4.0) * inv_std * 2.0 + 0.5,
            (7.0 - 4.0) * inv_std * 2.0 + 0.5,
        ];
        for (actual, expected) in output.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let seed = Tensor::from_data(output.shape, vec![1.0, 2.0, 3.0, 4.0]).expect("seed");
        let backward = executable
            .backward(&input, &training, &seed)
            .expect("backward should succeed");
        let xhat = [-3.0 * inv_std, -inv_std, inv_std, 3.0 * inv_std];
        let sum_dy: f32 = (1..=4).map(|value| value as f32).sum();
        let sum_dy_xhat: f32 = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .zip(xhat)
            .map(|(dy, xhat)| dy * xhat)
            .sum();
        for ((actual, dy), xhat) in backward
            .input_gradient
            .data
            .iter()
            .zip([1.0, 2.0, 3.0, 4.0])
            .zip(xhat)
        {
            let expected = 2.0 * inv_std / 4.0 * (4.0 * dy - sum_dy - xhat * sum_dy_xhat);
            assert!((actual - expected).abs() < 1e-5);
        }
        match &backward.param_gradients[0] {
            LayerParams::BatchNorm(gradient) => {
                assert!((gradient.biases[0] - 10.0).abs() < 1e-6);
                assert!((gradient.scales[0] - sum_dy_xhat).abs() < 1e-6);
                assert_eq!(gradient.rolling_mean, vec![0.0]);
                assert_eq!(gradient.rolling_variance, vec![0.0]);
            }
            other => panic!("unexpected param gradients: {other:?}"),
        }

        executable.zero_train_buffers();
        executable
            .backward_accumulate(&input, &training, &seed)
            .expect("accumulating backward should succeed");
        match &executable.train_buffers[0] {
            LayerTrainBuffers::BatchNorm(train) => {
                assert_eq!(train.bias_updates, vec![10.0]);
                assert!((train.scale_updates[0] - sum_dy_xhat).abs() < 1e-6);
            }
            other => panic!("unexpected train buffers: {other:?}"),
        }
        executable.update_weights(0.1, 0.0, 0.0, 1);
        let params = match &executable.params[0] {
            LayerParams::BatchNorm(params) => params,
            other => panic!("unexpected params: {other:?}"),
        };
        assert!((params.biases[0] - 1.5).abs() < 1e-6);
        assert!((params.scales[0] - (2.0 + 0.1 * sum_dy_xhat)).abs() < 1e-5);
        assert!((params.rolling_mean[0] - 9.4).abs() < 1e-6);
        assert!((params.rolling_variance[0] - 4.1).abs() < 1e-6);
    }

    #[test]
    fn gpu_standalone_batchnorm_uses_rolling_statistics_for_inference() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=2
                height=1
                channels=1
                batch=2

                [batchnorm]
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
        let LayerParams::BatchNorm(params) = &mut executable.params[0] else {
            panic!("expected standalone batchnorm parameters");
        };
        params.biases[0] = 0.5;
        params.scales[0] = 2.0;
        params.rolling_mean[0] = 10.0;
        params.rolling_variance[0] = 4.0;

        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 3.0, 5.0, 7.0],
        )
        .expect("input tensor");
        let result = executable
            .forward(&input)
            .expect("GPU inference should succeed");
        let inverse_std = 1.0 / (4.0f32 + super::BATCH_NORM_EPSILON).sqrt();
        let expected =
            [1.0f32, 3.0, 5.0, 7.0].map(|value| (value - 10.0) * inverse_std * 2.0 + 0.5);
        let actual = &result.final_output().expect("output").data;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let compact = executable
            .forward_compact(&input)
            .expect("compact GPU inference should succeed");
        let compact = &compact.final_output.expect("compact output").data;
        for (actual, expected) in compact.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
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

        let path = std::env::temp_dir().join("darknet-gpu-runtime-conv.weights");
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

        let path = std::env::temp_dir().join("darknet-gpu-runtime-cache.weights");
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
    fn gpu_inference_folds_batchnorm_into_conv_like_cpu() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=3
                height=1
                channels=2

                [convolutional]
                filters=3
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        match &mut gpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights = vec![1.0, -2.0, 0.5, 0.25, -1.0, 3.0];
                params.biases = vec![0.1, -0.2, 0.3];
                let bn = params.batch_norm.as_mut().expect("batch norm params");
                bn.scales = vec![1.5, 0.5, -2.0];
                bn.rolling_mean = vec![0.2, -0.4, 1.0];
                bn.rolling_variance = vec![4.0, 0.25, 9.0];
            }
            other => panic!("unexpected params: {other:?}"),
        }
        let mut cpu = ExecutableNetwork::new(network);
        cpu.params = gpu.params.clone();
        let input = Tensor::from_data(
            gpu.network.net.input_shape(),
            vec![1.0, 2.0, -1.0, 0.5, -3.0, 4.0],
        )
        .expect("input tensor");
        let want = cpu.forward(&input).expect("CPU forward");
        let got = gpu.forward(&input).expect("GPU forward");
        let (w, g) = (&want.layer_outputs[0], &got.layer_outputs[0]);
        assert_eq!(w.data.len(), g.data.len());
        for (a, b) in w.data.iter().zip(&g.data) {
            assert!((a - b).abs() < 1.0e-5, "cpu {a} vs gpu {b}");
        }
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
        assert!((batch_norm.rolling_mean[0] - 9.93).abs() < 1.0e-6);
        assert!((batch_norm.rolling_variance[0] - 19.82).abs() < 1.0e-6);
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
        assert!((batch_norm.rolling_mean[0] - 9.93).abs() < 1.0e-6);
        assert!((batch_norm.rolling_variance[0] - 19.82).abs() < 1.0e-6);

        let train_buffers = executable.device_train_buffers.borrow();
        let train = train_buffers[0].as_ref().expect("train buffers exist");
        assert_eq!(train.weight_updates.data, vec![0.0]);
        assert_eq!(train.bias_updates.data, vec![0.0]);
    }

    fn gpu_training_bn_network() -> Network {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=2
                width=3
                height=2
                channels=2

                [convolutional]
                filters=3
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg should parse");
        Network::from_config(cfg).expect("network should build")
    }

    fn gpu_training_bn_executable(
        weights: &[f32],
        scales: &[f32],
        biases: &[f32],
    ) -> GpuExecutableNetwork<FakeBackend> {
        let mut gpu = GpuExecutableNetwork::new(gpu_training_bn_network(), FakeBackend);
        match &mut gpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights = weights.to_vec();
                params.biases = biases.to_vec();
                let bn = params.batch_norm.as_mut().expect("batch norm params");
                bn.scales = scales.to_vec();
                bn.rolling_mean = vec![0.3, -0.2, 0.5];
                bn.rolling_variance = vec![2.0, 0.5, 3.0];
            }
            other => panic!("unexpected params: {other:?}"),
        }
        gpu
    }

    const BN_TRAIN_WEIGHTS: [f32; 6] = [0.5, -1.0, 0.25, 0.75, -0.5, 1.5];
    const BN_TRAIN_SCALES: [f32; 3] = [1.5, 0.5, -2.0];
    const BN_TRAIN_BIASES: [f32; 3] = [0.1, -0.2, 0.3];

    fn bn_train_input() -> Vec<f32> {
        (0..24).map(|i| ((i * 7 % 11) as f32) * 0.37 - 1.5).collect()
    }

    /// Pins the rolling-stat EMA factors to C++: GPU `.99`/`.01` (batchnorm_layer.cpp
    /// scal_ongpu/axpy_ongpu, cuDNN factor .01), host `.9`/`.1` (scal_cpu/axpy_cpu).
    #[test]
    fn batch_norm_rolling_keep_matches_cpp() {
        assert_eq!(super::BATCH_NORM_ROLLING_KEEP, 0.99);
        assert_eq!(super::BATCH_NORM_ROLLING_KEEP_CPU, 0.9);
        let mut bn = super::BatchNormParams {
            scales: vec![1.0],
            biases: vec![0.0],
            rolling_mean: vec![10.0],
            rolling_variance: vec![20.0],
        };
        super::update_batch_norm_running_stats(&mut bn, &[3.0], &[2.0], super::BATCH_NORM_ROLLING_KEEP).unwrap();
        // .99 * 10 + .01 * 3, .99 * 20 + .01 * 2
        assert!((bn.rolling_mean[0] - 9.93).abs() < 1.0e-5);
        assert!((bn.rolling_variance[0] - 19.82).abs() < 1.0e-5);
        super::update_batch_norm_running_stats(&mut bn, &[3.0], &[2.0], super::BATCH_NORM_ROLLING_KEEP_CPU).unwrap();
        assert!((bn.rolling_mean[0] - (0.9 * 9.93 + 0.3)).abs() < 1.0e-5);
    }

    #[test]
    fn gpu_training_forward_uses_batch_statistics_and_updates_rolling_stats() {
        let mut gpu = gpu_training_bn_executable(&BN_TRAIN_WEIGHTS, &BN_TRAIN_SCALES, &BN_TRAIN_BIASES);
        gpu.init_train_buffers_gpu().expect("train buffers");
        let input = Tensor::from_data(gpu.network.net.input_shape(), bn_train_input()).expect("input");
        let forward = gpu.forward_training_with_state(&input).expect("training forward");

        // f64 reference: 1x1 conv, batch statistics over (batch, h, w).
        let (batch, spatial, in_c, out_c) = (2usize, 6usize, 2usize, 3usize);
        let mut raw = vec![0.0f64; batch * out_c * spatial];
        for b in 0..batch {
            for o in 0..out_c {
                for s in 0..spatial {
                    raw[(b * out_c + o) * spatial + s] = (0..in_c)
                        .map(|c| {
                            BN_TRAIN_WEIGHTS[o * in_c + c] as f64
                                * input.data[(b * in_c + c) * spatial + s] as f64
                        })
                        .sum();
                }
            }
        }
        let n = (batch * spatial) as f64;
        let mut want = vec![0.0f64; raw.len()];
        let mut want_rolling = [(0.0f64, 0.0f64); 3];
        for o in 0..out_c {
            let at = |b: usize, s: usize| raw[(b * out_c + o) * spatial + s];
            let mean: f64 = (0..batch).flat_map(|b| (0..spatial).map(move |s| (b, s))).map(|(b, s)| at(b, s)).sum::<f64>() / n;
            let sq: f64 = (0..batch).flat_map(|b| (0..spatial).map(move |s| (b, s))).map(|(b, s)| (at(b, s) - mean).powi(2)).sum();
            let var = sq / n;
            for b in 0..batch {
                for s in 0..spatial {
                    want[(b * out_c + o) * spatial + s] = (at(b, s) - mean) / (var + 1.0e-5).sqrt()
                        * BN_TRAIN_SCALES[o] as f64
                        + BN_TRAIN_BIASES[o] as f64;
                }
            }
            let (rm, rv) = ([0.3, -0.2, 0.5][o], [2.0, 0.5, 3.0][o]);
            want_rolling[o] = (0.99 * rm + 0.01 * mean, 0.99 * rv + 0.01 * sq / (n - 1.0));
        }
        for (g, w) in forward.cpu_forward.layer_outputs[0].data.iter().zip(&want) {
            assert!((*g as f64 - w).abs() < 1.0e-4, "output {g} vs {w}");
        }
        // Inference forward still uses the rolling statistics.
        let inference = gpu.forward(&input).expect("inference forward");
        assert!(
            inference.layer_outputs[0]
                .data
                .iter()
                .zip(&want)
                .any(|(g, w)| (*g as f64 - w).abs() > 1.0e-2),
            "inference must not use batch statistics"
        );

        let seeds = vec![None; gpu.network.layers.len()];
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
        let Some(LayerParams::Convolutional(params)) = gpu.params.first() else {
            panic!("conv params expected");
        };
        let bn = params.batch_norm.as_ref().expect("batch norm params");
        for o in 0..out_c {
            assert!((bn.rolling_mean[o] as f64 - want_rolling[o].0).abs() < 1.0e-5);
            assert!((bn.rolling_variance[o] as f64 - want_rolling[o].1).abs() < 1.0e-5);
        }
    }

    #[test]
    fn gpu_training_batch_norm_backward_matches_finite_difference() {
        let input_data = bn_train_input();
        let seed_data: Vec<f32> = (0..36).map(|i| ((i * 5 % 7) as f32) * 0.3 - 0.8).collect();
        // L = sum(seed * output) through the batch-statistics forward.
        let loss = |weights: &[f32], scales: &[f32], biases: &[f32]| -> f64 {
            let gpu = gpu_training_bn_executable(weights, scales, biases);
            let input = Tensor::from_data(gpu.network.net.input_shape(), input_data.clone()).expect("input");
            let forward = gpu.forward_training_with_state(&input).expect("training forward");
            forward.cpu_forward.layer_outputs[0]
                .data
                .iter()
                .zip(&seed_data)
                .map(|(o, s)| *o as f64 * *s as f64)
                .sum()
        };

        let mut gpu = gpu_training_bn_executable(&BN_TRAIN_WEIGHTS, &BN_TRAIN_SCALES, &BN_TRAIN_BIASES);
        gpu.init_train_buffers_gpu().expect("train buffers");
        let input = Tensor::from_data(gpu.network.net.input_shape(), input_data.clone()).expect("input");
        let forward = gpu.forward_training_with_state(&input).expect("training forward");
        let seeds = vec![Some(
            Tensor::from_data(gpu.network.layers[0].output_shape, seed_data.clone()).expect("seed"),
        )];
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
        let (weight_updates, bias_updates, scale_updates) = {
            let buffers = gpu.device_train_buffers.borrow();
            let train = buffers[0].as_ref().expect("train buffers");
            (
                train.weight_updates.data.clone(),
                train.bias_updates.data.clone(),
                train.scale_updates.as_ref().expect("scale updates").data.clone(),
            )
        };

        let eps = 1.0e-2f32;
        let fd = |which: usize, index: usize| -> f64 {
            let (mut w, mut s, mut b) = (BN_TRAIN_WEIGHTS.to_vec(), BN_TRAIN_SCALES.to_vec(), BN_TRAIN_BIASES.to_vec());
            let target = match which {
                0 => &mut w,
                1 => &mut s,
                _ => &mut b,
            };
            target[index] += eps;
            let lp = loss(&w, &s, &b);
            let target = match which {
                0 => &mut w,
                1 => &mut s,
                _ => &mut b,
            };
            target[index] -= 2.0 * eps;
            let lm = loss(&w, &s, &b);
            (lp - lm) / (2.0 * eps as f64)
        };
        for (which, name, got) in [(0, "weight", &weight_updates), (1, "scale", &scale_updates), (2, "bias", &bias_updates)] {
            for (i, g) in got.iter().enumerate() {
                let want = fd(which, i);
                assert!(
                    (*g as f64 - want).abs() < 2.0e-2 + 2.0e-2 * want.abs(),
                    "{name} update[{i}]: backward {g} vs finite difference {want}"
                );
            }
        }
    }

    #[test]
    fn gpu_yolo_head_matches_cpu_and_cpp_activated_decode() {
        // Documents P3b: the Rust runtime keeps YOLO head outputs as raw
        // (pre-activation) logits on both CPU and GPU paths and applies
        // logistic + scale_x_y on the host inside decode/loss.  C++ applies
        // them on the device in forward_yolo_layer_gpu before decoding.  For
        // new_coords=0 (LegoGears_v2) both must yield identical detections.
        use crate::detect::{DetectionOptions, decode_yolo_detections};

        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                batch=1
                width=8
                height=8
                channels=1

                [convolutional]
                filters=7
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=9,12
                classes=2
                num=1
                scale_x_y=1.05
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        match &mut cpu.params[0] {
            LayerParams::Convolutional(params) => {
                params.weights = vec![1.3, -0.9, 0.7, 1.1, 2.5, 1.7, -1.4];
                params.biases = vec![0.1, -0.2, 0.3, -0.1, -1.5, -0.8, -1.2];
            }
            other => panic!("unexpected params: {other:?}"),
        }
        gpu.params = cpu.params.clone();
        let input = Tensor::from_data(
            cpu.network.net.input_shape(),
            (0..64).map(|idx| (idx as f32 - 20.0) / 16.0).collect(),
        )
        .expect("input tensor");

        let cpu_forward = cpu.forward(&input).expect("CPU forward");
        let gpu_full = gpu.forward_with_state(&input).expect("GPU forward");
        let gpu_compact = gpu.forward_compact(&input).expect("GPU compact forward");
        assert_eq!(cpu_forward.yolo_heads, gpu_full.cpu_forward.yolo_heads);
        assert_eq!(cpu_forward.yolo_heads, gpu_compact.yolo_heads);

        let options = DetectionOptions {
            confidence_threshold: 0.1,
            nms_threshold: 0.45,
            letterbox: false,
        };
        let decode = |heads: &[super::YoloHeadOutput]| {
            let mut dets = crate::detect::decode_yolo_detections(&network, heads, 8, 8, options)
                .expect("decode");
            dets.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
            dets
        };
        let cpu_dets = decode(&cpu_forward.yolo_heads);
        let gpu_dets = decode(&gpu_compact.yolo_heads);
        assert_eq!(cpu_dets, gpu_dets);

        // C++ reference: forward_yolo_layer_gpu activation, then
        // get_yolo_detections on the activated tensor (before NMS).
        let spec = match &network.layers[1].kind {
            LayerKind::Yolo(spec) => spec.clone(),
            other => panic!("unexpected layer {other:?}"),
        };
        let raw = &cpu_forward.yolo_heads[0].output;
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let (w, h, entries) = (8usize, 8usize, spec.classes + 5);
        let mut reference = Vec::new();
        for row in 0..h {
            for col in 0..w {
                // activated output, C++ layout: x,y logistic then scal_add; obj+cls logistic
                let x = spec.scale_x_y * sig(raw.get(0, 0, row, col)) - 0.5 * (spec.scale_x_y - 1.0);
                let y = spec.scale_x_y * sig(raw.get(0, 1, row, col)) - 0.5 * (spec.scale_x_y - 1.0);
                let obj = sig(raw.get(0, 4, row, col));
                if obj <= options.confidence_threshold {
                    continue;
                }
                let (bw, bh) = (
                    raw.get(0, 2, row, col).exp() * 9.0 / 8.0,
                    raw.get(0, 3, row, col).exp() * 12.0 / 8.0,
                );
                for c in 0..spec.classes {
                    let conf = obj * sig(raw.get(0, 5 + c, row, col));
                    if conf > options.confidence_threshold {
                        reference.push((
                            conf,
                            (col as f32 + x) / w as f32 * 8.0,
                            (row as f32 + y) / h as f32 * 8.0,
                            bw * 8.0,
                            bh * 8.0,
                        ));
                    }
                }
            }
        }
        assert_eq!(entries, 7);
        assert!(!reference.is_empty(), "test must exercise detections");
        reference.sort_by(|a, b| a.0.total_cmp(&b.0));
        // decode applies NMS; compare the pre-NMS reference against a run with
        // NMS disabled via an overlap threshold of 1.0.
        let no_nms = DetectionOptions {
            nms_threshold: 1.0,
            ..options
        };
        let mut dets = decode_yolo_detections(&network, &cpu_forward.yolo_heads, 8, 8, no_nms)
            .expect("decode");
        dets.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
        assert_eq!(dets.len(), reference.len());
        for (d, r) in dets.iter().zip(&reference) {
            assert!((d.confidence - r.0).abs() < 1e-6);
            assert!((d.bbox.x - r.1).abs() < 1e-4);
            assert!((d.bbox.y - r.2).abs() < 1e-4);
            assert!((d.bbox.w - r.3).abs() < 1e-4);
            assert!((d.bbox.h - r.4).abs() < 1e-4);
        }
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

        let path = std::env::temp_dir().join("darknet-runtime-parity.weights");
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

        let path = std::env::temp_dir().join("darknet-runtime-compact.weights");
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
    fn cpu_inference_compact_releases_route_and_shortcut_sources() {
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

                [upsample]
                stride=2
                scale=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1,-2

                [shortcut]
                from=-3
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
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, -2.0])
            .expect("input tensor");

        let full = executable
            .forward(&input)
            .expect("full inference should succeed")
            .final_output()
            .cloned()
            .expect("full output");
        let compact = executable
            .forward_inference(&input)
            .expect("compact inference should succeed");
        assert_eq!(compact, full);
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
                            | LayerKind::Detection(_)
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

    #[test]
    fn rnn_forward_backward_and_weight_updates_are_wired() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "rnn.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[rnn]\nhidden=3\noutput=2\nactivation=linear\n",
            )
            .expect("RNN cfg should parse"),
        )
        .expect("RNN network should build");
        assert_eq!(network.net.input_shape().batch, 4);
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Rnn(params) = &mut executable.params[0] else {
            panic!("expected RNN parameters");
        };
        for (index, value) in params.input.weights.iter_mut().enumerate() {
            *value = 0.05 + index as f32 * 0.01;
        }
        for (index, value) in params.self_params.weights.iter_mut().enumerate() {
            *value = 0.03 + index as f32 * 0.005;
        }
        for (index, value) in params.output.weights.iter_mut().enumerate() {
            *value = 0.02 + index as f32 * 0.004;
        }
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            (0..8).map(|index| index as f32 * 0.1 - 0.2).collect(),
        )
        .expect("RNN input should have the expected shape");
        let forward = executable
            .forward(&input)
            .expect("RNN forward should succeed");
        let output = forward.final_output().expect("RNN output should exist");
        assert_eq!(output.shape, executable.network.layers[0].output_shape);
        let seed = Tensor::filled(output.shape, 1.0);
        let backward = executable
            .backward(&input, &forward, &seed)
            .expect("RNN backward should succeed");
        assert!(
            backward
                .input_gradient
                .data
                .iter()
                .all(|value| value.is_finite())
        );

        executable.zero_train_buffers();
        let forward = executable
            .forward_training(&input)
            .expect("RNN training forward should succeed");
        executable
            .backward_accumulate(&input, &forward, &seed)
            .expect("RNN train backward should succeed");
        assert!(matches!(
            executable.train_buffers[0],
            LayerTrainBuffers::Rnn(_)
        ));
    }

    #[test]
    fn lstm_forward_backward_and_weight_updates_are_wired() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "lstm.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[lstm]\noutput=3\n",
            )
            .expect("LSTM cfg should parse"),
        )
        .expect("LSTM network should build");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Lstm(params) = &mut executable.params[0] else {
            panic!("expected LSTM parameters");
        };
        for nested in [
            &mut params.wf,
            &mut params.wi,
            &mut params.wg,
            &mut params.wo,
            &mut params.uf,
            &mut params.ui,
            &mut params.ug,
            &mut params.uo,
        ] {
            for value in &mut nested.weights {
                *value = 0.04;
            }
            for value in &mut nested.biases {
                *value = 0.01;
            }
        }
        let input = Tensor::filled(executable.network.net.input_shape(), 0.2);
        let forward = executable
            .forward(&input)
            .expect("LSTM forward should succeed");
        let output = forward.final_output().expect("LSTM output should exist");
        let seed = Tensor::filled(output.shape, 1.0);
        let backward = executable
            .backward(&input, &forward, &seed)
            .expect("LSTM backward should succeed");
        assert!(
            backward
                .input_gradient
                .data
                .iter()
                .all(|value| value.is_finite())
        );
        executable
            .backward_accumulate(&input, &forward, &seed)
            .expect("LSTM train backward should succeed");
        assert!(matches!(
            executable.train_buffers[0],
            LayerTrainBuffers::Lstm(_)
        ));
    }

    #[test]
    fn crnn_forward_backward_and_weight_updates_are_wired() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "crnn.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=2\ntime_steps=2\n[crnn]\nsize=1\nhidden=2\noutput=1\nactivation=linear\n",
            )
            .expect("CRNN cfg should parse"),
        )
        .expect("CRNN network should build");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Crnn(params) = &mut executable.params[0] else {
            panic!("expected CRNN parameters");
        };
        for nested in [
            &mut params.input,
            &mut params.self_params,
            &mut params.output,
        ] {
            for value in &mut nested.weights {
                *value = 0.1;
            }
            for value in &mut nested.biases {
                *value = 0.01;
            }
        }
        let input = Tensor::filled(executable.network.net.input_shape(), 0.2);
        let forward = executable
            .forward(&input)
            .expect("CRNN forward should succeed");
        let output = forward.final_output().expect("CRNN output should exist");
        let seed = Tensor::filled(output.shape, 1.0);
        let backward = executable
            .backward(&input, &forward, &seed)
            .expect("CRNN backward should succeed");
        assert!(
            backward
                .input_gradient
                .data
                .iter()
                .all(|value| value.is_finite())
        );
        executable
            .backward_accumulate(&input, &forward, &seed)
            .expect("CRNN train backward should succeed");
        assert!(matches!(
            executable.train_buffers[0],
            LayerTrainBuffers::Crnn(_)
        ));
    }

    #[test]
    fn cpu_forward_and_backward_pass_match_connected_reference() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "cpu-forward-backward.cfg",
                "[net]\ninputs=2\nbatch=1\n[connected]\noutput=1\nactivation=linear\n",
            )
            .expect("connected cfg should parse"),
        )
        .expect("connected network should build");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Connected(params) = &mut executable.params[0] else {
            panic!("expected connected parameters");
        };
        params.weights = vec![2.0, -1.0];
        params.biases = vec![0.25];

        let input = Tensor::from_data(executable.network.net.input_shape(), vec![3.0, 4.0])
            .expect("connected input should have the expected shape");

        let inference = executable
            .forward(&input)
            .expect("CPU inference forward should succeed");
        assert_eq!(
            inference.final_output().expect("forward output").data,
            vec![2.25]
        );

        let training = executable
            .forward_training(&input)
            .expect("CPU training forward should succeed");
        let seed = Tensor::filled(training.final_output().expect("training output").shape, 1.0);
        let backward = executable
            .backward(&input, &training, &seed)
            .expect("CPU backward should succeed");

        assert_eq!(backward.input_gradient.data, vec![2.0, -1.0]);
        match &backward.param_gradients[0] {
            LayerParams::Connected(gradient) => {
                assert_eq!(gradient.weights, vec![3.0, 4.0]);
                assert_eq!(gradient.biases, vec![1.0]);
            }
            other => panic!("unexpected connected gradients: {other:?}"),
        }

        executable.zero_train_buffers();
        executable
            .backward_accumulate(&input, &training, &seed)
            .expect("CPU training backward should accumulate");
        let LayerTrainBuffers::Connected(train) = &executable.train_buffers[0] else {
            panic!("expected connected training buffers");
        };
        assert_eq!(train.weight_updates, vec![3.0, 4.0]);
        assert_eq!(train.bias_updates, vec![1.0]);
    }

    #[test]
    fn cpu_crop_center_forward_and_backward_scatter() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "cpu-crop.cfg",
                "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n[crop]\ncrop_width=2\ncrop_height=2\nflip=0\nangle=0\nnoadjust=1\n",
            )
            .expect("crop cfg should parse"),
        )
        .expect("crop network should build");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            (1..=16).map(|value| value as f32).collect(),
        )
        .expect("crop input should have the expected shape");

        let forward = executable
            .forward(&input)
            .expect("CPU crop forward should succeed");
        assert_eq!(
            forward.final_output().expect("crop output").data,
            vec![6.0, 7.0, 10.0, 11.0]
        );

        let seed = Tensor::from_data(
            forward.final_output().expect("crop output").shape,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("crop gradient should have the expected shape");
        let backward = executable
            .backward(&input, &forward, &seed)
            .expect("CPU crop backward should succeed");
        assert_eq!(
            backward.input_gradient.data,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn gpu_connected_layers_train_on_the_gpu_path() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "gpu-connected.cfg",
                "[net]\ninputs=2\nbatch=1\n[connected]\noutput=1\nactivation=linear\n",
            )
            .expect("connected cfg should parse"),
        )
        .expect("connected network should build");
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        let LayerParams::Connected(params) = &mut gpu.params[0] else {
            panic!("expected connected parameters");
        };
        params.weights = vec![2.0, 3.0];
        params.biases = vec![0.5];
        let input = Tensor::from_data(gpu.network.net.input_shape(), vec![1.0, 2.0])
            .expect("connected input should have the expected shape");
        assert!(
            !gpu.requires_cpu_backward_fallback(),
            "connected layers must use the GPU backward path"
        );
        let forward = gpu
            .forward_with_state(&input)
            .expect("GPU connected forward should succeed");
        assert_eq!(forward.cpu_forward.final_output().unwrap().data, vec![8.5]);
        let seed = vec![Some(Tensor::filled(
            gpu.network.layers[0].output_shape,
            1.0,
        ))];
        gpu.backward_accumulate_seeded_gpu(&forward, &seed)
            .expect("GPU connected backward should succeed");
        gpu.update_weights_gpu(0.1, 0.0, 0.0, 1)
            .expect("GPU connected update should succeed");
        let LayerParams::Connected(params) = &gpu.params[0] else {
            panic!("expected connected parameters after update");
        };
        assert_eq!(params.weights, vec![2.1, 3.2]);
    }

    #[test]
    fn local_layer_uses_position_specific_kernels_in_forward_and_backward() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "local-runtime.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n[local]\nfilters=1\nsize=2\nstride=1\npad=0\nactivation=linear\n",
            )
            .expect("local cfg should parse"),
        )
        .expect("local network should build");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Local(params) = &mut executable.params[0] else {
            panic!("expected local parameters");
        };
        // location 0 owns [1, 2, 3, 4], location 1 owns [5, 6, 7, 8].
        params.biases = vec![0.5, -0.5];
        params.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("local input should have the expected shape");

        let forward = executable
            .forward(&input)
            .expect("local forward should succeed");
        assert_eq!(
            forward.final_output().expect("local output").data,
            vec![37.5, 110.5]
        );
        let seed = Tensor::from_data(executable.network.layers[0].output_shape, vec![1.0, 2.0])
            .expect("local gradient should have the expected shape");
        let backward = executable
            .backward(&input, &forward, &seed)
            .expect("local backward should succeed");
        assert_eq!(
            backward.input_gradient.data,
            vec![1.0, 12.0, 12.0, 3.0, 18.0, 16.0]
        );
        let LayerParams::Local(gradient) = &backward.param_gradients[0] else {
            panic!("expected local gradients");
        };
        assert_eq!(gradient.biases, vec![1.0, 2.0]);
        assert_eq!(
            gradient.weights,
            vec![1.0, 2.0, 4.0, 5.0, 4.0, 6.0, 10.0, 12.0]
        );
    }

    #[test]
    fn gru_layer_applies_darknet_update_gate_over_sequence_steps() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "gru-runtime.cfg",
                "[net]\ninputs=1\nbatch=2\ntime_steps=2\n[gru]\noutput=1\n",
            )
            .expect("GRU cfg should parse"),
        )
        .expect("GRU network should build");
        let executable = ExecutableNetwork::new(network);
        let input = Tensor::from_data(
            executable.network.net.input_shape(),
            vec![0.0, 0.0, 0.0, 0.0],
        )
        .expect("GRU input should have the expected shape");
        let forward = executable
            .forward(&input)
            .expect("GRU forward should succeed");
        assert_eq!(
            forward.final_output().expect("GRU output").data,
            vec![0.25, 0.25, 0.375, 0.375]
        );
    }

    fn fill_conv_params(params: &mut [LayerParams]) {
        for (layer, params) in params.iter_mut().enumerate() {
            if let LayerParams::Convolutional(conv) = params {
                for (i, w) in conv.weights.iter_mut().enumerate() {
                    *w = ((i + 3 * layer) as f32 * 0.7).sin() * 0.6;
                }
                for (i, b) in conv.biases.iter_mut().enumerate() {
                    *b = ((i + layer) as f32 * 1.3).cos() * 0.2;
                }
            }
        }
    }

    fn patterned_input(shape: TensorShape) -> Tensor {
        let data = (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.37).sin() + 0.1 * (i % 5) as f32)
            .collect();
        Tensor::from_data(shape, data).expect("input tensor")
    }

    /// Runs the same training forward + seeded backward on the CPU network and
    /// on the GPU network (fake backend, default host layer ops) and compares
    /// every layer output, dropout mask and convolution gradient.
    fn assert_gpu_layer_chain_matches_cpu(cfg_text: &str, seed_layer: usize) {
        let network = Network::from_config(
            ConfigFile::parse_str("inline.cfg", cfg_text).expect("cfg should parse"),
        )
        .expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        fill_conv_params(&mut cpu.params);
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu()
            .expect("GPU train buffers should initialize");
        assert!(
            !gpu.requires_cpu_backward_fallback(),
            "chain must use the GPU backward path"
        );

        let input = patterned_input(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("GPU forward");
        for (index, (lhs, rhs)) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
            .enumerate()
        {
            assert_eq!(lhs.shape, rhs.shape);
            assert_slice_close(&format!("layer {index} output"), &lhs.data, &rhs.data, 1e-5);
        }
        assert_eq!(
            cpu_forward.dropout_masks,
            gpu_forward.cpu_forward.dropout_masks
        );

        let mut seeds = vec![None; cpu.network.layers.len()];
        let seed_shape = cpu.network.layers[seed_layer].output_shape;
        seeds[seed_layer] = Some(
            Tensor::from_data(
                seed_shape,
                (0..seed_shape.batch * seed_shape.element_count())
                    .map(|i| (i as f32 * 0.91).cos())
                    .collect(),
            )
            .expect("seed tensor"),
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("GPU backward");

        let gpu_buffers = gpu.device_train_buffers.borrow();
        let mut checked = 0;
        for (index, train) in cpu.train_buffers.iter().enumerate() {
            let LayerTrainBuffers::Convolutional(cpu_train) = train else {
                continue;
            };
            let gpu_train = gpu_buffers[index].as_ref().expect("GPU train buffers");
            assert_slice_close(
                &format!("layer {index} weight gradient"),
                &cpu_train.weight_updates,
                &gpu_train.weight_updates.data,
                1e-4,
            );
            assert_slice_close(
                &format!("layer {index} bias gradient"),
                &cpu_train.bias_updates,
                &gpu_train.bias_updates.data,
                1e-4,
            );
            assert!(cpu_train.weight_updates.iter().any(|g| g.abs() > 1e-6));
            checked += 1;
        }
        assert!(checked > 0);
    }

    #[test]
    fn grad_mode_parses_analytical_default_and_cpp() {
        assert_eq!(crate::runtime::GradMode::parse(""), crate::runtime::GradMode::Analytical);
        assert_eq!(crate::runtime::GradMode::parse("analytical"), crate::runtime::GradMode::Analytical);
        assert_eq!(crate::runtime::GradMode::parse("garbage"), crate::runtime::GradMode::Analytical);
        assert_eq!(crate::runtime::GradMode::parse("CPP"), crate::runtime::GradMode::Cpp);
        assert_eq!(crate::runtime::GradMode::parse("c++"), crate::runtime::GradMode::Cpp);
    }

    #[test]
    fn gpu_dropout_reorg_softmax_match_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [dropout]\nprobability=0.4\n\
             [reorg]\nstride=2\n\
             [softmax]\ngroups=2\n",
            3,
        );
    }

    #[test]
    fn gpu_reverse_reorg_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [reorg]\nstride=2\n\
             [reorg]\nstride=2\nreverse=1\n",
            2,
        );
    }

    #[test]
    fn gpu_scale_channels_matches_cpu_forward_and_backward() {
        // per-channel scale (avgpool output) and per-pixel scale (1-filter conv)
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [avgpool]\n\
             [scale_channels]\nfrom=-2\nactivation=leaky\n",
            2,
        );
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [convolutional]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n\
             [scale_channels]\nfrom=-2\nscale_wh=1\nactivation=linear\n",
            2,
        );
    }

    /// Darknet resizes a mismatched shortcut source flat per sample (zero-pad
    /// when the source is smaller, crop when larger); the GPU path used to fall
    /// back to the CPU for the whole network.
    #[test]
    fn gpu_mismatched_shortcut_matches_cpu_forward_and_backward() {
        // source smaller than the layer: 2-channel source into a 4-channel input.
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
             [shortcut]\nfrom=-2\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
            3,
        );
        // source larger than the layer: the 4-channel source is cropped.
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n\
             [shortcut]\nfrom=-2\nactivation=linear\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
            3,
        );
    }

    /// Chained conv + activation stacks (incl. relu6 and ramp): GPU-runtime
    /// weight/bias gradients equal the CPU `ExecutableNetwork` backward.
    #[test]
    fn gpu_activation_chains_match_cpu_forward_and_backward() {
        for act in ["leaky", "relu6", "ramp", "mish", "swish", "logistic", "relu"] {
            assert_gpu_layer_chain_matches_cpu(
                &format!(
                    "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
                     [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nactivation={act}\n\
                     [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation={act}\n"
                ),
                1,
            );
        }
    }

    #[test]
    fn relu6_and_ramp_follow_cpp_activation_and_gradient() {
        let relu6 = |x: f32| super::activate(Activation::Relu6, x);
        assert_eq!([relu6(-1.0), relu6(0.0), relu6(3.0), relu6(6.0), relu6(9.0)], [0.0, 0.0, 3.0, 6.0, 6.0]);
        let ramp = |x: f32| super::activate(Activation::Ramp, x);
        assert_eq!(ramp(-2.0), -0.2);
        assert_eq!(ramp(2.0), 2.0 + 0.2);
        let grad = |a: &Activation, out: f32| {
            super::activation_gradient_from_output(a, out, None, "t").expect("gradient")
        };
        // output form == C++ input form: (x>0 && x<6) and (x>0)+0.1
        for x in [-3.0f32, 0.0, 2.0, 5.99, 6.0, 8.0] {
            assert_eq!(grad(&Activation::Relu6, relu6(x)), (x > 0.0 && x < 6.0) as i32 as f32, "relu6 x={x}");
            assert_eq!(grad(&Activation::Ramp, ramp(x)), (x > 0.0) as i32 as f32 + 0.1, "ramp x={x}");
        }
        for (name, expected) in [("relu6", Activation::Relu6), ("ramp", Activation::Ramp)] {
            let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={name}\n"),
                )
                .expect("cfg"),
            )
            .expect("network");
            let LayerKind::Convolutional(spec) = &network.layers[0].kind else {
                panic!("expected convolution");
            };
            assert_eq!(spec.activation, expected);
            assert!(spec.activation.to_cuda_f32_kind().is_some());
        }
    }

    #[test]
    fn extra_activations_parse_dispatch_and_use_the_exact_input_gradient() {
        let all = [
            ("gelu", Activation::Gelu),
            ("elu", Activation::Elu),
            ("selu", Activation::Selu),
            ("relie", Activation::Relie),
            ("plse", Activation::Plse),
            ("hardtan", Activation::Hardtan),
            ("lhtan", Activation::Lhtan),
            ("stair", Activation::Stair),
            ("hard_mish", Activation::HardMish),
            ("revleaky", Activation::Revleaky),
        ];
        for (name, expected) in &all {
            let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={name}\n"),
                )
                .expect("cfg"),
            )
            .expect("network");
            let LayerKind::Convolutional(spec) = &network.layers[0].kind else {
                panic!("expected convolution");
            };
            assert_eq!(&spec.activation, expected, "{name}");
            // dispatch -> central difference of `activate` vs the runtime gradient
            for x in [-1.7f32, -0.6, 0.4, 0.9, 2.3] {
                let h = 1e-2;
                let f = |v: f32| super::activate(expected.clone(), v);
                let fd = (f(x + h) - f(x - h)) / (2.0 * h);
                let grad = super::activation_gradient_from_output(expected, f(x), Some(x), "t")
                    .expect("gradient");
                assert!((fd - grad).abs() <= 3e-3 + 3e-3 * fd.abs(), "{name} x={x}: fd {fd}, grad {grad}");
            }
        }
        // input-form gradients must refuse to run without the cached pre-activation
        assert!(super::activation_gradient_from_output(&Activation::Gelu, 0.5, None, "t").is_err());
    }

    #[test]
    fn gpu_logistic_head_matches_cpu_forward_and_passes_the_seed_through() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [logistic]\n",
            1,
        );
    }

    #[test]
    fn deconvolutional_layer_trains_on_cpu_and_gpu_fallback_identically() {        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation=leaky\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        fill_conv_params(&mut cpu.params);
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        assert!(gpu.requires_cpu_backward_fallback(), "deconvolutional trains through the CPU fallback");

        let input = patterned_input(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).unwrap();
        let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
        for (i, (a, b)) in cpu_forward.layer_outputs.iter().zip(&gpu_forward.cpu_forward.layer_outputs).enumerate() {
            assert_slice_close(&format!("layer {i}"), &a.data, &b.data, 1e-5);
        }
        let shape = cpu.network.layers[1].output_shape;
        let mut seeds = vec![None; 2];
        seeds[1] = Some(Tensor::from_data(shape, (0..2 * shape.element_count()).map(|i| (i as f32 * 0.91).cos()).collect()).unwrap());
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds).unwrap();
        let LayerTrainBuffers::Convolutional(cpu_train) = &cpu.train_buffers[1] else { panic!() };
        let fallback = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::Convolutional(gpu_train) = &fallback[1] else { panic!() };
        assert!(cpu_train.weight_updates.iter().any(|g| g.abs() > 1e-6));
        assert_slice_close("weight_updates", &cpu_train.weight_updates, &gpu_train.weight_updates, 1e-4);
        assert_slice_close("bias_updates", &cpu_train.bias_updates, &gpu_train.bias_updates, 1e-4);
        drop(fallback);

        // one SGD step moves the host-resident deconvolution weights on both paths alike
        cpu.update_weights(0.01, 0.9, 0.0005, 2);
        gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
        let (LayerParams::Convolutional(a), LayerParams::Convolutional(b)) = (&cpu.params[1], &gpu.params[1]) else { panic!() };
        assert_slice_close("updated weights", &a.weights, &b.weights, 1e-5);
    }

    #[test]
    fn gpu_channel_shuffle_matches_cpu_forward_and_backward() {        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\nstride=1\nactivation=linear\n\
             [channel_shuffle]\ngroups=3\n",
            1,
        );
    }

    #[test]
    fn gpu_channel_slice_matches_cpu_forward_and_backward() {        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [channel_slice]\nfrom=-1\nstart=1\nend=3\n",
            1,
        );
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [maxpool]\nsize=2\nstride=2\n\
             [channel_slice]\nfrom=-2\nstart=0\nend=2\n",
            2,
        );
    }

    #[test]
    fn gpu_gelu_convolution_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=gelu\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
            1,
        );
    }

    #[test]
    fn weighted_shortcut_layer_backward_matches_finite_differences_on_its_weights() {        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=relu\nactivation=leaky\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut net = ExecutableNetwork::new(network);
        fill_conv_params(&mut net.params);
        let LayerParams::Shortcut(p) = &mut net.params[2] else { panic!() };
        p.weights = vec![0.9, 0.4, 0.7, 0.6]; // (n+1)=2 inputs x 2 channels
        let input = patterned_input(net.network.net.input_shape());
        let seed = Tensor::from_data(
            net.network.layers[2].output_shape,
            (0..16).map(|i| (i as f32 * 0.61).cos()).collect(),
        )
        .unwrap();
        let loss = |net: &ExecutableNetwork| -> f32 {
            let out = net.forward_training(&input).unwrap();
            out.layer_outputs[2].data.iter().zip(&seed.data).map(|(o, s)| o * s).sum()
        };
        let forward = net.forward_training(&input).unwrap();
        let mut seeds = vec![None; 3];
        seeds[2] = Some(seed.clone());
        let backward = net.backward_accumulate_seeded(&input, &forward, &seeds).unwrap();
        let LayerParams::Shortcut(g) = &backward.param_gradients[2] else { panic!() };
        for k in 0..4 {
            let h = 1e-2f32;
            let (mut np, mut nm) = (net.clone(), net.clone());
            if let LayerParams::Shortcut(p) = &mut np.params[2] {
                p.weights[k] += h;
            }
            if let LayerParams::Shortcut(p) = &mut nm.params[2] {
                p.weights[k] -= h;
            }
            let fd = (loss(&np) - loss(&nm)) / (2.0 * h);
            assert!(
                (fd - g.weights[k]).abs() <= 5e-3 + 5e-3 * fd.abs(),
                "w[{k}]: fd {fd}, analytic {}",
                g.weights[k]
            );
        }
    }

    #[test]
    fn gpu_connected_forward_matches_cpu_with_and_without_batch_norm() {
        for batch_normalize in [0, 1] {
            let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!(
                        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=3\n\
                         [connected]\noutput=5\nbatch_normalize={batch_normalize}\nactivation=leaky\n"
                    ),
                )
                .expect("cfg should parse"),
            )
            .expect("network should build");
            let mut cpu = ExecutableNetwork::new(network.clone());
            let LayerParams::Connected(params) = &mut cpu.params[0] else {
                panic!("expected connected parameters");
            };
            for (i, w) in params.weights.iter_mut().enumerate() {
                *w = (i as f32 * 0.43).sin() * 0.5;
            }
            for (i, b) in params.biases.iter_mut().enumerate() {
                *b = 0.1 * i as f32 - 0.2;
            }
            if let Some(bn) = params.batch_norm.as_mut() {
                bn.scales = vec![1.5, 0.7, -0.4, 1.0, 2.0];
                bn.rolling_mean = vec![0.1, -0.2, 0.3, 0.0, 0.5];
                bn.rolling_variance = vec![0.5, 1.5, 2.0, 0.8, 1.2];
            }
            assert_eq!(params.batch_norm.is_some(), batch_normalize == 1);
            let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
            gpu.params = cpu.params.clone();
            let input = patterned_input(cpu.network.net.input_shape());
            let cpu_forward = cpu.forward(&input).expect("CPU forward");
            let gpu_forward = gpu.forward_with_state(&input).expect("GPU forward");
            assert_slice_close(
                "connected output",
                &cpu_forward.layer_outputs[0].data,
                &gpu_forward.cpu_forward.layer_outputs[0].data,
                1e-5,
            );
            assert_slice_close(
                "connected activation input",
                &cpu_forward.layer_activation_inputs[0].as_ref().unwrap().data,
                &gpu_forward.cpu_forward.layer_activation_inputs[0]
                    .as_ref()
                    .unwrap()
                    .data,
                1e-5,
            );
        }
    }

    // ---- P8 optimizer wiring: Adam/AdamW, loss scale, NaN/Inf, accumulation ----

    fn optimizer_net(net_extra: &str) -> GpuExecutableNetwork<FakeBackend> {
        let cfg = format!(
            "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n{net_extra}\n\
             [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nbatch_normalize=1\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n"
        );
        let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg"))
            .expect("network");
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        fill_conv_params(&mut gpu.params);
        if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
            let bn = p.batch_norm.as_mut().expect("bn");
            bn.scales = vec![1.2, 0.8, -0.6];
        }
        gpu.init_train_buffers_gpu().expect("train buffers");
        gpu
    }

    fn optimizer_seeds(gpu: &GpuExecutableNetwork<FakeBackend>, step: usize) -> Vec<Option<Tensor>> {
        let mut seeds = vec![None; gpu.network.layers.len()];
        let shape = gpu.network.layers[1].output_shape;
        seeds[1] = Some(
            Tensor::from_data(
                shape,
                (0..shape.batch * shape.element_count())
                    .map(|i| ((i + 7 * step) as f32 * 0.91).cos() * 0.5)
                    .collect(),
            )
            .expect("seed"),
        );
        seeds
    }

    /// (weights, biases, scales) of the two conv layers as host vectors plus
    /// the accumulated updates, from the device state.
    fn optimizer_snapshot(
        gpu: &GpuExecutableNetwork<FakeBackend>,
    ) -> Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
        (0..2)
            .map(|i| {
                let train = gpu.device_train_buffers.borrow()[i].clone().expect("train");
                (
                    train.weight_updates.data,
                    train.bias_updates.data,
                    train.scale_updates.map(|t| t.data),
                )
            })
            .collect()
    }

    fn host_params(gpu: &GpuExecutableNetwork<FakeBackend>, layer: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let LayerParams::Convolutional(p) = &gpu.params[layer] else {
            panic!("conv params")
        };
        (
            p.weights.clone(),
            p.biases.clone(),
            p.batch_norm.as_ref().map(|b| b.scales.clone()).unwrap_or_default(),
        )
    }

    /// f64 Adam / AdamW reference of one tensor (C++ `adam_update_gpu`).
    #[allow(clippy::too_many_arguments)]
    fn ref_adam_tensor(
        w: &mut [f64],
        m: &mut [f64],
        v: &mut [f64],
        u: &[f32],
        lr: f64,
        (b1, b2, eps): (f64, f64, f64),
        decay_batch: f64,
        decoupled: f64,
        t: i32,
    ) {
        for i in 0..w.len() {
            let g = u[i] as f64 - decay_batch * w[i];
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let mhat = m[i] / (1.0 - b1.powi(t));
            let vhat = v[i] / (1.0 - b2.powi(t));
            w[i] += lr * mhat / (vhat.sqrt() + eps) - decoupled * w[i];
        }
    }

    #[test]
    fn gpu_adam_and_adamw_match_f64_reference_over_steps() {
        for decoupled in [false, true] {
            let mut gpu = optimizer_net(&format!(
                "adam=1\nB1=0.8\nB2=0.95\neps=0.00001\ndecoupled_decay={}",
                decoupled as u8
            ));
            assert!(gpu.network.net.adam && gpu.network.net.decoupled_decay == decoupled);
            assert_eq!((gpu.network.net.adam_b1, gpu.network.net.adam_b2), (0.8, 0.95));
            // reference state: (w, b, s) x (m, v) per layer
            let ws: Vec<_> = (0..2).map(|l| host_params(&gpu, l)).collect();
            let mut refw: Vec<[Vec<f64>; 3]> = ws
                .iter()
                .map(|(w, b, s)| [w, b, s].map(|x| x.iter().map(|&f| f as f64).collect()))
                .collect();
            let mut refm: Vec<[Vec<f64>; 3]> = refw.iter().map(|a| a.clone().map(|x| vec![0.0; x.len()])).collect();
            let mut refv = refm.clone();
            let (lr, decay, batch) = (0.01f32, 0.05f32, 2usize);
            for step in 1..=4 {
                let input = patterned_input(gpu.network.net.input_shape());
                let forward = gpu.forward_training_with_state(&input).expect("forward");
                let seeds = optimizer_seeds(&gpu, step);
                gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
                let updates = optimizer_snapshot(&gpu);
                gpu.update_weights_gpu(lr, 0.9, decay, batch).expect("adam update");
                assert_eq!(gpu.adam_step() as usize, step);
                let (coupled, dec) = if decoupled {
                    (0.0, lr as f64 * decay as f64)
                } else {
                    (decay as f64 * batch as f64, 0.0)
                };
                for layer in 0..2 {
                    let (uw, ub, us) = &updates[layer];
                    for (k, u) in [Some(uw), Some(ub), us.as_ref()].into_iter().enumerate() {
                        let Some(u) = u else { continue };
                        // C++ decays bias/scale only for coupled Adam.
                        let (c, d) = if k == 0 { (coupled, dec) } else if decoupled { (0.0, 0.0) } else { (coupled, 0.0) };
                        ref_adam_tensor(
                            &mut refw[layer][k],
                            &mut refm[layer][k],
                            &mut refv[layer][k],
                            u,
                            lr as f64,
                            (0.8, 0.95, 1e-5),
                            c,
                            d,
                            step as i32,
                        );
                    }
                    let (w, b, s) = host_params(&gpu, layer);
                    for (k, got) in [w, b, s].iter().enumerate() {
                        for (i, g) in got.iter().enumerate() {
                            let want = refw[layer][k][i];
                            assert!(
                                (*g as f64 - want).abs() < 2e-5,
                                "decoupled={decoupled} step {step} layer {layer} tensor {k}[{i}]: {g} vs {want}"
                            );
                        }
                    }
                }
                // Adam zeroes the accumulated updates (C++ adam_update_gpu).
                for (uw, ub, us) in optimizer_snapshot(&gpu) {
                    assert!(uw.iter().chain(&ub).chain(us.iter().flatten()).all(|x| *x == 0.0));
                }
            }
        }
    }

    #[test]
    fn deferred_gpu_update_matches_eager_after_sync_and_refolds_lazily() {
        let mut eager = optimizer_net("");
        let mut lazy = optimizer_net("");
        let input = Tensor::from_data(
            eager.network.net.input_shape(),
            (0..eager.network.net.input_shape().element_count() * 2)
                .map(|i| (i as f32 * 0.37).sin())
                .collect(),
        )
        .expect("input");
        for step in 0..3 {
            for (gpu, deferred) in [(&mut eager, false), (&mut lazy, true)] {
                let forward = gpu.forward_training_device(&input).expect("forward");
                let seeds = optimizer_seeds(gpu, step);
                gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
                if deferred {
                    gpu.update_weights_gpu_deferred(0.05, 0.9, 0.001, 2).expect("update");
                } else {
                    gpu.update_weights_gpu(0.05, 0.9, 0.001, 2).expect("update");
                }
            }
        }
        lazy.sync_host_weights_from_device().expect("sync");
        for layer in 0..2 {
            assert_eq!(host_params(&eager, layer), host_params(&lazy, layer), "layer {layer}");
        }
        // Inference forward (BN-folded weights) agrees after the lazy refold.
        let want = eager.forward(&input).expect("eager inference");
        let got = lazy.forward(&input).expect("lazy inference");
        assert_eq!(want.layer_outputs.len(), got.layer_outputs.len());
        for (w, g) in want.layer_outputs.iter().zip(&got.layer_outputs) {
            assert_slice_close("inference", &g.data, &w.data, 1e-5);
        }
    }

    #[test]
    fn gpu_adam_rejects_host_fallback_layers() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nadam=1\n[local]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        gpu.init_train_buffers_gpu().expect("buffers");
        assert!(gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).is_err());
    }

    #[test]
    fn gpu_adam_supports_connected_layers() {
        // Connected parameters stay on the host but now have Adam moments, so a
        // connected layer trains with `adam=1` on the GPU path.
        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\ninputs=2\nbatch=1\nadam=1\n[connected]\noutput=1\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        let LayerParams::Connected(params) = &mut gpu.params[0] else {
            panic!("expected connected parameters");
        };
        params.weights = vec![2.0, 3.0];
        params.biases = vec![0.5];
        let input = Tensor::from_data(gpu.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = gpu.forward_with_state(&input).expect("forward");
        let seed = vec![Some(Tensor::filled(
            gpu.network.layers[0].output_shape,
            1.0,
        ))];
        gpu.backward_accumulate_seeded_gpu(&forward, &seed)
            .expect("backward");
        gpu.update_weights_gpu(0.1, 0.9, 0.0, 1).expect("adam update");
        let LayerParams::Connected(params) = &gpu.params[0] else {
            panic!("expected connected parameters");
        };
        // Adam steps by ~lr regardless of gradient magnitude.
        assert!(
            params.weights[0] > 2.0 && params.weights[1] > 3.0,
            "weights should move: {:?}",
            params.weights
        );
    }

    #[test]
    fn gpu_sgd_is_default_and_loss_scale_is_neutral() {
        // loss_scale scales the seeds and divides the lr back out: SGD and Adam
        // results are independent of it (up to rounding).
        for adam in [0, 1] {
            let run = |loss_scale: f32| {
                let mut gpu = optimizer_net(&format!("adam={adam}\nloss_scale={loss_scale}"));
                assert_eq!(gpu.network.net.loss_scale, loss_scale);
                for step in 1..=3 {
                    let input = patterned_input(gpu.network.net.input_shape());
                    let forward = gpu.forward_training_with_state(&input).expect("forward");
                    let seeds = optimizer_seeds(&gpu, step);
                    gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
                    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).expect("update");
                }
                (host_params(&gpu, 0), host_params(&gpu, 1))
            };
            let (a, b) = (run(1.0), run(256.0));
            for (x, y) in [(a.0, b.0), (a.1, b.1)] {
                assert_slice_close("weights", &x.0, &y.0, 2e-5);
                assert_slice_close("biases", &x.1, &y.1, 2e-5);
                assert_slice_close("scales", &x.2, &y.2, 2e-5);
            }
        }
        let gpu = optimizer_net("");
        assert!(!gpu.network.net.adam && gpu.network.net.loss_scale == 1.0 && !gpu.network.net.decoupled_decay);
    }

    #[test]
    fn gpu_update_resets_nan_inf_updates_and_fixes_nan_inf_weights() {
        let mut gpu = optimizer_net("");
        {
            let mut cached = gpu.device_conv_params.borrow_mut();
            let c = cached[1].as_mut().expect("cached params");
            c.raw_weights.data[1] = f32::INFINITY;
            c.raw_weights.data[2] = f32::NAN;
            c.raw_weights.data[0] = 0.25;
            let mut train = gpu.device_train_buffers.borrow_mut();
            let t = train[1].as_mut().expect("train");
            t.weight_updates.data[0] = f32::NAN;
            t.weight_updates.data[3] = f32::NEG_INFINITY;
            t.bias_updates.data[0] = f32::NAN;
        }
        // lr = 0: weights only change through the sanitizer.
        gpu.update_weights_gpu(0.0, 0.9, 0.0, 1).expect("update");
        let (w, b, _) = host_params(&gpu, 1);
        assert_eq!(w[0], 0.25, "finite weight untouched");
        assert_eq!(w[1], 1.0 / 2.0, "fix_nan_and_inf: 1/(index+1)");
        assert_eq!(w[2], 1.0 / 3.0);
        assert!(w.iter().chain(&b).all(|x| x.is_finite()));
        let train = gpu.device_train_buffers.borrow()[1].clone().expect("train");
        assert!(train.weight_updates.data.iter().chain(&train.bias_updates.data).all(|x| x.is_finite()));
    }

    #[test]
    fn loss_scale_helpers_and_non_finite_detection() {
        let mut gpu = optimizer_net("adam=1\nloss_scale=8\n");
        assert_eq!(gpu.loss_scale(), 8.0);
        gpu.set_loss_scale(4.0);
        assert_eq!(gpu.loss_scale(), 4.0);
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu.forward_training_with_state(&input).expect("forward");
        let seeds = optimizer_seeds(&gpu, 1);
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("backward");
        assert!(!gpu.gradients_non_finite().expect("finite check"));
        {
            let mut buffers = gpu.device_train_buffers.borrow_mut();
            let train = buffers[0].as_mut().expect("train buffer");
            let mut data = gpu.backend.download_f32(&train.weight_updates).expect("download");
            data[0] = f32::INFINITY;
            train.weight_updates = gpu
                .backend
                .upload_f32(&[data.len()], &data)
                .expect("upload");
        }
        assert!(gpu.gradients_non_finite().expect("finite check"));
        gpu.set_loss_scale(0.0);
        assert_eq!(gpu.loss_scale(), 4.0);
    }

    #[test]
    fn optimizer_state_round_trips_adam_moments() {
        let dir = std::env::temp_dir().join(format!("darknet-opt-state-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("opt.state");

        let step = |gpu: &mut GpuExecutableNetwork<FakeBackend>, s: usize| {
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            let seeds = optimizer_seeds(gpu, s);
            gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
                .expect("backward");
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 2).expect("update");
        };

        let mut original = optimizer_net("adam=1\nloss_scale=1\n");
        step(&mut original, 1);
        step(&mut original, 2);
        let weights_path = dir.join("model.weights");
        crate::weights::save_weights(&original.network, &original.params, &weights_path, 2)
            .expect("save weights");
        original.save_optimizer_state(&path).expect("save");
        assert_eq!(original.adam_step(), 2);

        // Resume into a fresh network: restore parameters from the weights file
        // and moments/step from the sidecar.
        let mut resumed = optimizer_net("adam=1\nloss_scale=1\n");
        resumed.load_weights_file(&weights_path).expect("load weights");
        resumed.load_optimizer_state(&path).expect("load");
        assert_eq!(resumed.adam_step(), 2);
        let path2 = dir.join("opt2.state");
        resumed.save_optimizer_state(&path2).expect("resave");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&path2).unwrap(),
            "optimizer state round-trip changed bytes"
        );

        // The next step must match, which only holds if the moments were restored.
        step(&mut original, 3);
        step(&mut resumed, 3);
        for layer in 0..2 {
            let (w0, b0, s0) = host_params(&original, layer);
            let (w1, b1, s1) = host_params(&resumed, layer);
            assert_slice_close("weights", &w0, &w1, 1e-6);
            assert_slice_close("biases", &b0, &b1, 1e-6);
            assert_slice_close("scales", &s0, &s1, 1e-6);
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn gpu_gradients_accumulate_across_subdivisions() {
        // Two backward passes into the same buffers == the sum of two fresh passes.
        let grads = |steps: &[usize]| {
            let mut gpu = optimizer_net("subdivisions=1");
            for &step in steps {
                let input = patterned_input(gpu.network.net.input_shape());
                let forward = gpu.forward_training_with_state(&input).expect("forward");
                let seeds = optimizer_seeds(&gpu, step);
                gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
            }
            optimizer_snapshot(&gpu)
        };
        let both = grads(&[1, 2]);
        let (first, second) = (grads(&[1]), grads(&[2]));
        for layer in 0..2 {
            let sum = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>();
            assert_slice_close("weight updates", &sum(&first[layer].0, &second[layer].0), &both[layer].0, 1e-5);
            assert_slice_close("bias updates", &sum(&first[layer].1, &second[layer].1), &both[layer].1, 1e-5);
        }
    }

    /// Finite-difference check of the GPU-path backward of dropout (fixed
    /// mask) and scale_channels (both scale layouts) through a conv.
    #[test]
    fn gpu_dropout_and_scale_channels_backward_match_finite_difference() {
        for cfg_tail in [
            "[dropout]\nprobability=0.4\n",
            "[avgpool]\n[scale_channels]\nfrom=-2\nactivation=leaky\n",
            "[convolutional]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n[scale_channels]\nfrom=-2\nscale_wh=1\nactivation=linear\n",
        ] {
            let cfg = format!(
                "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n{cfg_tail}"
            );
            let build = |weights: &[f32]| {
                let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg"))
                    .expect("network");
                let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
                fill_conv_params(&mut gpu.params);
                if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
                    p.weights = weights.to_vec();
                }
                gpu
            };
            let base: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).sin() * 0.6).collect();
            let last = build(&base).network.layers.len() - 1;
            let loss = |weights: &[f32]| -> f64 {
                let gpu = build(weights);
                let input = patterned_input(gpu.network.net.input_shape());
                let forward = gpu.forward_training_with_state(&input).expect("forward");
                let out = &forward.cpu_forward.layer_outputs[last];
                out.data.iter().enumerate().map(|(i, o)| *o as f64 * (i as f64 * 0.91).cos()).sum()
            };
            let mut gpu = build(&base);
            gpu.init_train_buffers_gpu().expect("buffers");
            assert!(!gpu.requires_cpu_backward_fallback(), "GPU backward path expected");
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            let shape = gpu.network.layers[last].output_shape;
            let mut seeds = vec![None; gpu.network.layers.len()];
            seeds[last] = Some(
                Tensor::from_data(
                    shape,
                    (0..shape.batch * shape.element_count()).map(|i| (i as f32 * 0.91).cos()).collect(),
                )
                .expect("seed"),
            );
            gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
            let analytic = gpu.device_train_buffers.borrow()[0].clone().expect("train").weight_updates.data;
            let h = 1e-3f32;
            for (i, a) in analytic.iter().enumerate() {
                let (mut wp, mut wm) = (base.clone(), base.clone());
                wp[i] += h;
                wm[i] -= h;
                let fd = (loss(&wp) - loss(&wm)) / (2.0 * h as f64);
                assert!(
                    (*a as f64 - fd).abs() < 2e-3 + 2e-3 * fd.abs(),
                    "[{cfg_tail:?}] dW[{i}]: analytic {a} vs finite difference {fd}"
                );
            }
        }
    }

    #[test]
    fn cpu_reorg_backward_is_the_adjoint_of_forward() {
        // <reorg(x), y> == <x, reorg_backward(y)> for forward and reverse layers
        for reverse in [false, true] {
            let spec = crate::model::ReorgSpec { stride: 2, reverse };
            let (small, big) = (
                TensorShape::new(3, 2, 8, 2),
                TensorShape::new(6, 4, 2, 2),
            );
            let (input_shape, output_shape) = if reverse { (small, big) } else { (big, small) };
            let x = patterned_input(input_shape);
            let y = patterned_input(output_shape);
            let fx = super::forward_reorg(spec.stride, spec.reverse, &x, output_shape).unwrap();
            let bty = super::backward_reorg(&spec, &y, input_shape).unwrap();
            let lhs: f32 = fx.data.iter().zip(&y.data).map(|(a, b)| a * b).sum();
            let rhs: f32 = x.data.iter().zip(&bty.data).map(|(a, b)| a * b).sum();
            assert!((lhs - rhs).abs() < 1e-3, "reverse={reverse}: {lhs} vs {rhs}");
        }
    }

    // ---- DropBlock ----

    const DROPBLOCK_CFG: &str = "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
         [dropout]\nprobability=0.9\ndropblock=1\ndropblock_size_abs=2\n";

    #[test]
    fn dropblock_cfg_parses_like_cpp() {
        let spec = |tail: &str| {
            let cfg = format!(
                "[net]\nwidth=8\nheight=6\nchannels=1\nbatch=1\n[dropout]\nprobability=0.3\n{tail}"
            );
            let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).unwrap()).unwrap();
            match &network.layers[0].kind {
                LayerKind::Dropout { probability, dropblock } => {
                    assert_eq!(*probability, Some(0.3));
                    *dropblock
                }
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(spec(""), None);
        assert_eq!(spec("dropblock=0\ndropblock_size_abs=3\n"), None);
        // neither key set: C++ defaults to 7 (the cap only applies to an explicit value).
        assert_eq!(spec("dropblock=1\n"), Some(crate::model::DropblockSpec { size_rel: 0.0, size_abs: 7 }));
        assert_eq!(spec("dropblock=1\ndropblock_size_abs=20\n"), Some(crate::model::DropblockSpec { size_rel: 0.0, size_abs: 6 }));
        assert_eq!(spec("dropblock=1\ndropblock_size_rel=0.5\n"), Some(crate::model::DropblockSpec { size_rel: 0.5, size_abs: 0 }));
        // both set: the absolute size wins
        assert_eq!(
            spec("dropblock=1\ndropblock_size_rel=0.5\ndropblock_size_abs=3\n"),
            Some(crate::model::DropblockSpec { size_rel: 0.0, size_abs: 3 })
        );
    }

    #[test]
    fn dropblock_plan_follows_the_cpp_ramp() {
        let shape = TensorShape::new(10, 8, 4, 1);
        let abs = crate::model::DropblockSpec { size_rel: 0.0, size_abs: 7 };
        // max_batches = 0: no ramp.
        assert_eq!(super::dropblock_plan(&abs, 0.5, shape, 0, 0), (0.5 / 49.0, 7));
        // ramp over the first 85% of 100 batches: iteration 0 -> nothing.
        assert_eq!(super::dropblock_plan(&abs, 0.5, shape, 0, 100), (0.0, 1));
        // halfway through the ramp: multiplier 0.5 -> int(3.5) = 3.
        let (p, size) = super::dropblock_plan(&abs, 0.5, shape, 42, 100);
        assert_eq!(size, 3);
        assert!((p - 0.5 * (42.0 / 85.0) / 9.0).abs() < 1e-7);
        // after the ramp, the size is capped by the smaller side (height 8).
        assert_eq!(super::dropblock_plan(&abs, 0.5, shape, 90, 100), (0.5 / 49.0, 7));
        let big = crate::model::DropblockSpec { size_rel: 0.0, size_abs: 9 };
        assert_eq!(super::dropblock_plan(&big, 0.5, shape, 90, 100).1, 8);
        // relative size uses width/height separately, the square is the smaller one.
        let rel = crate::model::DropblockSpec { size_rel: 0.5, size_abs: 0 };
        assert_eq!(super::dropblock_plan(&rel, 0.5, shape, 0, 0).1, 4);
    }

    #[test]
    fn dropblock_reference_zeroes_one_square_per_plane_and_rescales_per_sample() {
        let shape = [2usize, 3, 5, 6];
        let x: Vec<f32> = (0..shape.iter().product::<usize>()).map(|i| 1.0 + (i % 7) as f32).collect();
        let bs = 2;
        // block_prob 1 selects the lowest-random cell of every plane.
        let (out, mask) = darknet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 99);
        let (c, h, w) = (shape[1], shape[2], shape[3]);
        for b in 0..shape[0] {
            let dropped: usize = mask[b * c * h * w..(b + 1) * c * h * w].iter().filter(|m| **m == 0.0).count();
            assert_eq!(dropped, c * bs * bs, "one {bs}x{bs} block per channel");
            let scale = 1.0 / (1.0 - dropped as f32 / (c * h * w) as f32);
            for ch in 0..c {
                let plane = &mask[(b * c + ch) * h * w..(b * c + ch + 1) * h * w];
                let zeros: Vec<(usize, usize)> = (0..h * w)
                    .filter(|i| plane[*i] == 0.0)
                    .map(|i| (i / w, i % w))
                    .collect();
                let (y0, x0) = (zeros.iter().map(|p| p.0).min().unwrap(), zeros.iter().map(|p| p.1).min().unwrap());
                let expected: Vec<(usize, usize)> =
                    (0..bs).flat_map(|dy| (0..bs).map(move |dx| (y0 + dy, x0 + dx))).collect();
                assert_eq!(zeros, expected, "contiguous square inside the plane");
                for i in 0..h * w {
                    if plane[i] != 0.0 {
                        assert_eq!(plane[i], scale);
                    }
                }
            }
        }
        for ((o, m), v) in out.iter().zip(&mask).zip(&x) {
            assert_eq!(*o, v * m);
        }
        // probability 0 keeps everything (scale 1)
        let (out0, mask0) = darknet_cuda::host_ops::dropblock(&x, &shape, 0.0, bs, 99);
        assert_eq!(out0, x);
        assert!(mask0.iter().all(|m| *m == 1.0));
        // deterministic for a seed, different for another
        assert_eq!(darknet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 99).1, mask);
        assert_ne!(darknet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 100).1, mask);
    }

    #[test]
    fn gpu_dropblock_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(DROPBLOCK_CFG, 1);
        // the mask is not vacuous
        let network = Network::from_config(ConfigFile::parse_str("inline.cfg", DROPBLOCK_CFG).unwrap()).unwrap();
        let cpu = ExecutableNetwork::new(network);
        let forward = cpu.forward_training(&patterned_input(cpu.network.net.input_shape())).unwrap();
        let mask = forward.dropout_masks[1].as_ref().expect("dropblock mask");
        assert!(mask.data.iter().any(|m| *m == 0.0) && mask.data.iter().any(|m| *m > 1.0));
        // inference leaves the input alone
        let infer = cpu.forward(&patterned_input(cpu.network.net.input_shape())).unwrap();
        assert!(infer.dropout_masks[1].is_none());
        assert_eq!(infer.layer_outputs[1], infer.layer_outputs[0]);
    }

    /// The training iteration changes the DropBlock mask (ramp + seed mix);
    /// iteration 0 of a ramped schedule drops nothing.
    #[test]
    fn dropblock_mask_follows_the_train_iteration() {
        let cfg = DROPBLOCK_CFG.replace("batch=2\n", "batch=2\nmax_batches=100\n");
        let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network);
        let input = patterned_input(cpu.network.net.input_shape());
        cpu.set_train_iteration(0);
        assert!(cpu.forward_training(&input).unwrap().dropout_masks[1].is_none());
        cpu.set_train_iteration(60);
        let a = cpu.forward_training(&input).unwrap().dropout_masks[1].clone().expect("mask");
        cpu.set_train_iteration(61);
        let b = cpu.forward_training(&input).unwrap().dropout_masks[1].clone().expect("mask");
        assert_ne!(a, b);
    }

    #[test]
    fn gpu_dropblock_backward_matches_finite_difference() {
        let cfg = format!(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [dropout]\nprobability=0.9\ndropblock=1\ndropblock_size_abs=2\n"
        );
        let build = |weights: &[f32]| {
            let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg")).expect("network");
            let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
            fill_conv_params(&mut gpu.params);
            if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
                p.weights = weights.to_vec();
            }
            gpu
        };
        let base: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).sin() * 0.6).collect();
        let last = 1;
        let loss = |weights: &[f32]| -> f64 {
            let gpu = build(weights);
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            forward.cpu_forward.layer_outputs[last]
                .data
                .iter()
                .enumerate()
                .map(|(i, o)| *o as f64 * (i as f64 * 0.91).cos())
                .sum()
        };
        let mut gpu = build(&base);
        gpu.init_train_buffers_gpu().expect("buffers");
        assert!(!gpu.requires_cpu_backward_fallback());
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu.forward_training_with_state(&input).expect("forward");
        assert!(forward.cpu_forward.dropout_masks[last].as_ref().expect("mask").data.contains(&0.0));
        let shape = gpu.network.layers[last].output_shape;
        let mut seeds = vec![None; gpu.network.layers.len()];
        seeds[last] = Some(
            Tensor::from_data(
                shape,
                (0..shape.batch * shape.element_count()).map(|i| (i as f32 * 0.91).cos()).collect(),
            )
            .unwrap(),
        );
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds).expect("backward");
        let analytic = gpu.device_train_buffers.borrow()[0].clone().unwrap().weight_updates.data;
        let h = 1e-3f32;
        for (i, a) in analytic.iter().enumerate() {
            let (mut wp, mut wm) = (base.clone(), base.clone());
            wp[i] += h;
            wm[i] -= h;
            let fd = (loss(&wp) - loss(&wm)) / (2.0 * h as f64);
            assert!(
                (*a as f64 - fd).abs() < 2e-3 + 2e-3 * fd.abs(),
                "dW[{i}]: analytic {a} vs finite difference {fd}"
            );
        }
    }

    // ---- per-layer update keys and try_fix_nan ----

    /// Two convs (layer 0 with BN, 32-channel input so `grad_centr` applies,
    /// layer 1 without BN); `l0`/`l1` add keys to the layer sections.
    fn keyed_cfg(net_extra: &str, l0: &str, l1: &str) -> String {
        format!(
            "[net]\nwidth=2\nheight=2\nchannels=32\nbatch=1\n{net_extra}\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nbatch_normalize=1\nactivation=linear\n{l0}\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n{l1}\n"
        )
    }

    fn keyed_cpu(cfg: &str) -> ExecutableNetwork {
        let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network);
        fill_conv_params(&mut cpu.params);
        for (layer, train) in cpu.train_buffers.iter_mut().enumerate() {
            if let LayerTrainBuffers::Convolutional(t) = train {
                for (i, u) in t.weight_updates.iter_mut().enumerate() {
                    *u = ((i * 3 + layer) % 11) as f32 * 0.25 - 1.0;
                }
                for (i, u) in t.bias_updates.iter_mut().enumerate() {
                    *u = i as f32 * 0.5 - 0.3;
                }
                if let Some(bn) = t.batch_norm.as_mut() {
                    bn.scale_updates = vec![0.4, -0.2, 0.6];
                }
            }
        }
        cpu
    }

    fn keyed_gpu(cfg: &str) -> GpuExecutableNetwork<FakeBackend> {
        let cpu = keyed_cpu(cfg);
        let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        {
            let mut train = gpu.device_train_buffers.borrow_mut();
            for (layer, t) in train.iter_mut().enumerate() {
                let LayerTrainBuffers::Convolutional(c) = &cpu.train_buffers[layer] else { continue };
                let t = t.as_mut().unwrap();
                t.weight_updates.data = c.weight_updates.clone();
                t.bias_updates.data = c.bias_updates.clone();
                if let (Some(s), Some(bn)) = (t.scale_updates.as_mut(), c.batch_norm.as_ref()) {
                    s.data = bn.scale_updates.clone();
                }
            }
        }
        gpu
    }

    /// Weights of both layers after one plain step (lr 0.1, momentum 0, no
    /// decay, batch 1) from the buffers of `keyed_cpu`, on CPU and GPU.
    fn keyed_step(cfg: &str, iteration: usize) -> [(Vec<f32>, Vec<f32>); 2] {
        let mut cpu = keyed_cpu(cfg);
        cpu.set_train_iteration(iteration);
        cpu.update_weights(0.1, 0.0, 0.0, 1);
        let mut gpu = keyed_gpu(cfg);
        gpu.set_train_iteration(iteration);
        gpu.update_weights_gpu(0.1, 0.0, 0.0, 1).unwrap();
        let pick = |params: &[LayerParams], i: usize| match &params[i] {
            LayerParams::Convolutional(p) => (p.weights.clone(), p.biases.clone()),
            _ => unreachable!(),
        };
        for i in 0..2 {
            let (c, g) = (pick(&cpu.params, i), pick(&gpu.params, i));
            assert_slice_close("cpu/gpu weights", &c.0, &g.0, 1e-6);
            assert_slice_close("cpu/gpu biases", &c.1, &g.1, 1e-6);
        }
        [pick(&cpu.params, 0), pick(&cpu.params, 1)]
    }

    #[test]
    fn layer_train_keys_parse_and_train_only_bn_propagates_backwards() {
        let net = |l0: &str, l1: &str| {
            Network::from_config(ConfigFile::parse_str("inline.cfg", &keyed_cfg("", l0, l1)).unwrap()).unwrap()
        };
        let n = net("learning_rate=0.5\nclip=2\ngrad_centr=1\nburnin_update=2\n", "dont_update=7\n");
        assert_eq!(n.layers[0].train.learning_rate_scale, 0.5);
        assert_eq!((n.layers[0].train.clip, n.layers[0].train.grad_centr, n.layers[0].train.burnin_update), (2.0, true, 2));
        assert_eq!((n.layers[1].train.dont_update, n.layers[1].train.learning_rate_scale), (7, 1.0));
        assert!(!n.layers[0].train.train_only_bn && !n.layers[1].train.train_only_bn);
        // C++ set_train_only_bn: a later layer's flag covers every earlier layer
        let n = net("", "train_only_bn=1\n");
        assert!(n.layers[0].train.train_only_bn && n.layers[1].train.train_only_bn);
        let n = net("train_only_bn=1\n", "");
        assert!(n.layers[0].train.train_only_bn && !n.layers[1].train.train_only_bn);
        assert!(!Network::from_config(ConfigFile::parse_str("i.cfg", &keyed_cfg("", "", "")).unwrap()).unwrap().net.try_fix_nan);
        assert!(Network::from_config(ConfigFile::parse_str("i.cfg", &keyed_cfg("try_fix_nan=1", "", "")).unwrap()).unwrap().net.try_fix_nan);
    }

    #[test]
    fn per_layer_learning_rate_scales_only_that_layer_cpu_and_gpu() {
        let plain = keyed_step(&keyed_cfg("", "", ""), 0);
        let scaled = keyed_step(&keyed_cfg("", "learning_rate=0.5\n", ""), 0);
        let base = keyed_cpu(&keyed_cfg("", "", ""));
        let LayerParams::Convolutional(p0) = &base.params[0] else { unreachable!() };
        for i in 0..p0.weights.len() {
            let d_plain = plain[0].0[i] - p0.weights[i];
            let d_scaled = scaled[0].0[i] - p0.weights[i];
            assert!((d_scaled - 0.5 * d_plain).abs() < 1e-6, "w[{i}]: {d_scaled} vs half of {d_plain}");
        }
        assert!(plain[0].0 != p0.weights);
        assert_eq!(plain[1], scaled[1], "other layers keep the global rate");
    }

    #[test]
    fn dont_update_burnin_update_and_train_only_bn_gate_the_step() {
        let base = keyed_cpu(&keyed_cfg("", "", ""));
        let init = |i: usize| match &base.params[i] {
            LayerParams::Convolutional(p) => (p.weights.clone(), p.biases.clone()),
            _ => unreachable!(),
        };
        // dont_update=3: C++ updates once iteration_num (= iteration + 1) > 3.
        for (iteration, updated) in [(0, false), (2, false), (3, true)] {
            let r = keyed_step(&keyed_cfg("", "", "dont_update=3\n"), iteration);
            assert_eq!(r[1] != init(1), updated, "dont_update at iteration {iteration}");
            assert!(r[0] != init(0), "layer 0 is not gated");
        }
        // burnin_update=2, burn_in=10: skipped while 2*10 > iteration_num.
        for (iteration, updated) in [(0, false), (18, false), (19, true)] {
            let r = keyed_step(&keyed_cfg("burn_in=10", "burnin_update=2\n", ""), iteration);
            assert_eq!(r[0] != init(0), updated, "burnin_update at iteration {iteration}");
            assert!(r[1] != init(1));
        }
        // train_only_bn on layer 1 freezes both layers (C++ skips the update
        // of every layer up to the flagged one).
        let r = keyed_step(&keyed_cfg("", "", "train_only_bn=1\n"), 5);
        assert_eq!(r[0], init(0));
        assert_eq!(r[1], init(1));
    }

    #[test]
    fn clip_clamps_weights_after_the_step() {
        let free = keyed_step(&keyed_cfg("", "", ""), 0);
        let clipped = keyed_step(&keyed_cfg("", "clip=0.3\n", ""), 0);
        assert!(free[0].0.iter().any(|w| w.abs() > 0.3), "test must exceed the clip");
        let expected: Vec<f32> = free[0].0.iter().map(|w| w.clamp(-0.3, 0.3)).collect();
        assert_slice_close("clipped weights", &clipped[0].0, &expected, 1e-6);
        assert_eq!(clipped[0].1, free[0].1, "biases are not clipped");
        assert_eq!(clipped[1], free[1]);
    }

    #[test]
    fn grad_centr_removes_the_per_filter_mean_of_weight_updates() {
        let base = keyed_cpu(&keyed_cfg("", "", ""));
        let LayerParams::Convolutional(p0) = &base.params[0] else { unreachable!() };
        let LayerTrainBuffers::Convolutional(t0) = &base.train_buffers[0] else { unreachable!() };
        let free = keyed_step(&keyed_cfg("", "", ""), 0);
        let centred = keyed_step(&keyed_cfg("", "grad_centr=1\n", ""), 0);
        let f_size = 32;
        for f in 0..3 {
            let upd = &t0.weight_updates[f * f_size..(f + 1) * f_size];
            let mean = upd.iter().sum::<f32>() / f_size as f32;
            assert!(mean.abs() > 1e-3, "filter {f} needs a non-zero mean");
            for i in 0..f_size {
                let idx = f * f_size + i;
                let expect = p0.weights[idx] + 0.1 * (upd[i] - mean);
                assert!((centred[0].0[idx] - expect).abs() < 1e-5, "w[{idx}]");
            }
        }
        assert!(free[0].0 != centred[0].0);
        assert_eq!(free[0].1, centred[0].1, "biases untouched");
        assert_eq!(free[1], centred[1], "no BN -> no centralisation");
        // grad_centr without batch norm does nothing (C++ condition)
        let cfg = keyed_cfg("", "", "grad_centr=1\n");
        assert_eq!(keyed_step(&cfg, 0)[1], free[1]);
    }

    #[test]
    fn try_fix_nan_fixes_conv_outputs_and_clamps_deltas_cpu_and_gpu() {
        let cfg = |fix: u8| keyed_cfg(&format!("try_fix_nan={fix}"), "", "");
        let input_of = |shape| patterned_input(shape);
        // forward: a NaN weight poisons the output unless try_fix_nan is set
        for fix in [0u8, 1] {
            let mut cpu = keyed_cpu(&cfg(fix));
            if let LayerParams::Convolutional(p) = &mut cpu.params[1] {
                p.weights[0] = f32::NAN;
            }
            let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
            gpu.params = cpu.params.clone();
            let input = input_of(cpu.network.net.input_shape());
            let cpu_out = cpu.forward_training(&input).unwrap().layer_outputs[1].clone();
            let gpu_out = gpu.forward_training_with_state(&input).unwrap().cpu_forward.layer_outputs[1].clone();
            for out in [&cpu_out, &gpu_out] {
                assert_eq!(out.data.iter().all(|v| v.is_finite()), fix == 1, "try_fix_nan={fix}");
            }
            if fix == 1 {
                assert_slice_close("fixed output", &cpu_out.data, &gpu_out.data, 1e-6);
            }
        }
        // backward: output deltas are clamped to [-1, 1] (last layer is linear)
        let big: Vec<f32> = (0..8).map(|i| [5.0, -7.0, 0.25, -0.5, 3.0, -0.1, 9.0, 0.9][i]).collect();
        let clamped: Vec<f32> = big.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        let run = |fix: u8, seed: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let mut cpu = keyed_cpu(&cfg(fix));
            for t in &mut cpu.train_buffers {
                if let LayerTrainBuffers::Convolutional(t) = t {
                    t.weight_updates.fill(0.0);
                    t.bias_updates.fill(0.0);
                }
            }
            let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
            gpu.params = cpu.params.clone();
            gpu.init_train_buffers_gpu().unwrap();
            let input = input_of(cpu.network.net.input_shape());
            let shape = cpu.network.layers[1].output_shape;
            let mut seeds = vec![None; 2];
            seeds[1] = Some(Tensor::from_data(shape, seed.to_vec()).unwrap());
            let f = cpu.forward_training(&input).unwrap();
            cpu.backward_accumulate_seeded(&input, &f, &seeds).unwrap();
            let gf = gpu.forward_training_with_state(&input).unwrap();
            gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
            let LayerTrainBuffers::Convolutional(c) = &cpu.train_buffers[1] else { unreachable!() };
            let g = gpu.device_train_buffers.borrow()[1].clone().unwrap();
            assert_slice_close("cpu/gpu layer-1 gradient", &c.weight_updates, &g.weight_updates.data, 1e-4);
            (c.weight_updates.clone(), c.bias_updates.clone())
        };
        let with_fix = run(1, &big);
        let reference = run(0, &clamped);
        let unclamped = run(0, &big);
        assert_slice_close("clamped delta gradient", &with_fix.0, &reference.0, 1e-5);
        assert_slice_close("clamped bias gradient", &with_fix.1, &reference.1, 1e-5);
        assert!(with_fix.1 != unclamped.1, "without try_fix_nan the large deltas pass through");
    }
}
