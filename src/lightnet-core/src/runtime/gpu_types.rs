use super::*;
use crate::model::DeconvolutionalSpec;

/// Whether the network contains a `[deform_conv]` variant the device path does
/// not cover yet. Shared by the forward and the backward so they cannot choose
/// different paths for the same network.
pub(super) fn has_host_only_deform(network: &Network) -> bool {
    network.layers.iter().any(|layer| {
        matches!(&layer.kind, LayerKind::DeformConv(spec)
            if spec.batch_normalize)
    })
}

pub(super) fn deconvolutional_spec_is_native(spec: &DeconvolutionalSpec) -> bool {
    spec.activation.to_cuda_f32_kind().is_some()
        || spec.activation.to_channel_normalization().is_some()
}

/// Native GraphConv training forward/backward covers edge modes 0/1 and
/// supported activations, including batch-statistics BN; compact inference also
/// supports BN affine using rolling statistics.
pub(super) fn graph_conv_spec_needs_host(spec: &GraphConvSpec) -> bool {
    !matches!(spec.edge_mode, 0 | 1)
        || (spec.activation.to_channel_normalization().is_none()
            && spec.activation.to_cuda_f32_kind().is_none())
}

pub(super) fn graph_conv_spec_needs_compact_host(spec: &GraphConvSpec) -> bool {
    !matches!(spec.edge_mode, 0 | 1)
        || (spec.activation.to_channel_normalization().is_none()
            && spec.activation.to_cuda_f32_kind().is_none())
}

pub(super) fn has_host_only_graph_conv(network: &Network) -> bool {
    network.layers.iter().any(|layer| {
        matches!(&layer.kind, LayerKind::GraphConv(spec) if graph_conv_spec_needs_host(spec))
    })
}

pub(super) fn reject_gpu_stage1_cpu_layers(network: &Network, operation: &str) -> Result<()> {
    if let Some(layer) = network.layers.iter().find(|layer| {
        matches!(
            layer.kind,
            LayerKind::DepthAnything(_) | LayerKind::DptHead(_) | LayerKind::Yolo3d(_)
        )
    }) {
        return Err(Error::UnsupportedLayer {
            layer_index: layer.index,
            name: format!(
                "[{}] GPU {operation} is not implemented in Stage-1 CPU support",
                layer.kind.name()
            ),
        });
    }
    Ok(())
}

pub(super) fn reject_gpu_hyperbolic_foveation(network: &Network, operation: &str) -> Result<()> {
    if let Some(layer) = network.layers.iter().find(|layer| {
        matches!(
            layer.kind,
            LayerKind::HyperbolicFoveation(_)
                | LayerKind::FoveationFilm(_)
                | LayerKind::DepthAnything(_)
                | LayerKind::DptHead(_)
                | LayerKind::Yolo3d(_)
        )
    }) {
        return Err(Error::UnsupportedLayer {
            layer_index: layer.index,
            name: format!(
                "[{}] GPU {operation} is not implemented in Stage-1 CPU support",
                layer.kind.name()
            ),
        });
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackwardResult {
    pub input_gradient: Tensor,
    pub layer_output_gradients: Vec<Tensor>,
    pub param_gradients: Vec<LayerParams>,
    /// Incoming scan-state derivatives for each MambaVision layer.
    pub mambavision_state_gradients: Vec<Option<crate::mambavision::MambaVisionState>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuForwardSummary {
    pub final_output: Option<Tensor>,
    pub yolo_heads: Vec<YoloHeadOutput>,
    /// The modern anchor-free heads, in layer order.
    pub modern_heads: Vec<ModernHeadOutput>,
}

#[derive(Clone, Debug)]
pub struct GpuForwardState<T>
where
    T: Clone,
{
    pub input_gpu: T,
    pub gpu_layer_outputs: Vec<T>,
    pub gpu_layer_activation_inputs: Vec<Option<T>>,
    /// Batch-statistics batch-norm state for standalone BatchNorm, convolutional,
    /// and GraphConv layers
    /// run by a native training forward; `None` means rolling statistics were
    /// used (inference forward, or a network that needs CPU backward fallback).
    pub gpu_batch_norm_caches: Vec<Option<GpuBatchNormCache<T>>>,
    pub(crate) gpu_rnn_caches: Vec<Option<GpuRnnBackwardCache<T>>>,
    pub(crate) gpu_crnn_caches: Vec<Option<GpuCrnnBackwardCache<T>>>,
    pub(crate) gpu_lstm_caches: Vec<Option<GpuLstmBackwardCache<T>>>,
    pub(crate) gpu_moe_caches: Vec<Option<GpuMoeBackwardCache<T>>>,
    pub(crate) gpu_eml_caches: Vec<Option<GpuEmlBackwardCache<T>>>,
    pub(super) gpu_dcnv4_caches: Vec<Option<dcnv4_gpu::ForwardCache<T>>>,
    pub cpu_forward: ForwardResult,
}

/// Device-resident recurrent values retained for RNN backward, including the
/// host-forward-cache upload path for batch-normalized RNNs.
#[derive(Clone, Debug)]
pub struct GpuRnnBackwardCache<T>
where
    T: Clone,
{
    pub(super) sequence_batch: usize,
    pub(super) input_activation_inputs: Vec<T>,
    pub(super) self_activation_inputs: Vec<T>,
    pub(super) output_activation_inputs: Vec<T>,
    pub(super) previous_states: Vec<T>,
    pub(super) states: Vec<T>,
}

/// Device-resident recurrent values retained for CRNN backward, including the
/// host-forward-cache upload path for batch-normalized CRNNs.
#[derive(Clone, Debug)]
pub struct GpuCrnnBackwardCache<T>
where
    T: Clone,
{
    pub(super) sequence_batch: usize,
    pub(super) input_activation_inputs: Vec<T>,
    pub(super) self_activation_inputs: Vec<T>,
    pub(super) output_activation_inputs: Vec<T>,
    pub(super) previous_states: Vec<T>,
    pub(super) states: Vec<T>,
}

/// Device-resident MoE values retained for no-BN backward.
#[derive(Clone, Debug)]
pub struct GpuMoeBackwardCache<T>
where
    T: Clone,
{
    pub(super) gates: T,
    pub(super) pooled: T,
    pub(super) expert_outputs: Vec<T>,
    pub(super) expert_activation_inputs: Vec<T>,
}

/// Device-resident branch outputs used by no-BN EML backward.
#[derive(Clone, Debug)]
pub struct GpuEmlBackwardCache<T>
where
    T: Clone,
{
    pub(super) input_branch_output: T,
    pub(super) self_branch_output: T,
    pub(super) input_batch_norm: Option<GpuBatchNormCache<T>>,
    pub(super) self_batch_norm: Option<GpuBatchNormCache<T>>,
}

/// Device-resident recurrent values retained for LSTM backward, including the
/// host-forward-cache upload path for batch-normalized LSTMs.
#[derive(Clone, Debug)]
pub struct GpuLstmBackwardCache<T>
where
    T: Clone,
{
    pub(super) sequence_batch: usize,
    pub(super) previous_states: Vec<T>,
    pub(super) previous_cells: Vec<T>,
    pub(super) forget: Vec<T>,
    pub(super) input_gate: Vec<T>,
    pub(super) candidate: Vec<T>,
    pub(super) output_gate: Vec<T>,
    pub(super) cells: Vec<T>,
}

/// What the batch-norm backward pass and the rolling-stat update need from a
/// batch-statistics forward.
#[derive(Clone, Debug)]
pub struct GpuBatchNormCache<T>
where
    T: Clone,
{
    pub(super) scales: T,
    pub(super) x_hat: T,
    /// Biased batch variance.
    pub(super) variance: T,
    /// Updated `(rolling_mean, rolling_variance)`; `None` for shared layers,
    /// whose owner keeps the statistics.
    pub(super) rolling: Option<(T, T)>,
}

#[derive(Clone, Debug)]
pub struct GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
{
    pub backend: B,
    pub network: Network,
    pub params: Vec<LayerParams>,
    pub(super) device_conv_params: RefCell<Vec<Vec<Option<GpuConvolutionLayerParams<B::Tensor>>>>>,
    /// Per-owner-layer f16 conv weights of `forward_compact_f16`, type-erased
    /// because `B::TensorF16` only exists for `B: F16TensorOpsBackend`.
    pub(super) f16_conv_weights: RefCell<Vec<Option<std::rc::Rc<dyn std::any::Any>>>>,
    pub(super) device_train_buffers: RefCell<Vec<Option<GpuConvolutionTrainBuffers<B::Tensor>>>>,
    pub(super) device_moe_train_buffers: RefCell<Vec<Option<GpuMoeTrainBuffers<B::Tensor>>>>,
    pub(super) device_eml_train_buffers: RefCell<Vec<Option<GpuEmlTrainBuffers<B::Tensor>>>>,
    pub(super) device_dcnv4_train_buffers: RefCell<Vec<Option<GpuDcnv4TrainBuffers<B::Tensor>>>>,
    /// Reusable DCNv4 parameter uploads; snapshots detect public host edits.
    pub(crate) dcnv4_device_params: RefCell<Vec<Option<dcnv4_gpu::DeviceParameters<B::Tensor>>>>,
    pub(super) cpu_fallback_train_buffers: RefCell<Vec<LayerTrainBuffers>>,
    pub(super) capture_input_gradient: Cell<bool>,
    pub(super) captured_input_gradient: RefCell<Option<Tensor>>,
    /// One slot per layer; only `[deform_conv]` uses it, and only on the
    /// device-trainable path (see `DeformDeviceCache`).
    pub(super) deform_device_caches: RefCell<Vec<Option<DeformDeviceCache<B::Tensor>>>>,
    /// Per-owner-layer convolution Adam moments (device), created lazily on the
    /// first Adam update.
    pub(super) adam_state: Vec<Option<GpuAdamLayerState<B::Tensor>>>,
    /// Device Adam moments for MoE routers/experts, including BN scales in DNOPT4.
    pub(super) moe_adam_state: Vec<Option<GpuMoeAdamLayerState<B::Tensor>>>,
    /// Device Adam moments for EML's two convolution branches, persisted in DNOPT5.
    pub(super) eml_adam_state: Vec<Option<GpuEmlAdamLayerState<B::Tensor>>>,
    /// Persistent device Adam moments for native DCNv4 updates. Sidecar saves
    /// materialize them into the existing host-compatible representation.
    pub(super) dcnv4_adam_state: Vec<Option<GpuDcnv4AdamState<B::Tensor>>>,
    /// Adam moments for parameters updated by the host fallback.
    pub(super) host_adam_state: RefCell<Vec<Option<HostAdamLayerState>>>,
    /// Newton–Schulz momentum for opt-in GPU Muon updates (host sidecar mirror).
    pub(super) muon_state: Vec<Option<MuonLayerState>>,
    /// Persistent device momentum for connected-layer GPU Muon matrices.
    pub(super) device_muon_momentum: RefCell<Vec<Option<B::Tensor>>>,

    /// Number of Adam updates applied so far (the 1-based bias-correction `t`).
    pub(super) adam_step: u32,
    /// See [`ExecutableNetwork::set_train_iteration`].
    pub(super) train_iteration: usize,
    /// Host convolution/EML weights, biases, BN gamma, and rolling stats lag
    /// the device ones (deferred updates, see `sync_host_weights_from_device`).
    pub(super) host_weights_stale: bool,
    /// Set by [`Self::update_weights_gpu_deferred`]: the backward pass keeps the
    /// BN rolling stats on the device instead of downloading them per step.
    pub(super) defer_host_sync: bool,
}

/// Device-resident intermediates of one native `[deform_conv]` forward pass.
/// They are the operands the native backward needs; produced only when the whole
/// network is device-trainable (no host-cache fallback), and dropped otherwise.
#[derive(Clone, Debug)]
pub struct DeformDeviceCache<T> {
    pub input_sanitized: T,
    pub raw_offsets: T,
    pub mean: T,
    pub inv_std: T,
    pub stddev: T,
    pub normalize: T,
    pub offsets: T,
    pub masks: Option<T>,
    pub mask_logits: Option<T>,
}

/// First/second moments of one tensor.
pub(super) type GpuMoments<T> = (T, T);

#[derive(Clone, Debug)]
pub(super) struct GpuAdamLayerState<T> {
    pub(super) weights: Option<GpuMoments<T>>,
    pub(super) biases: Option<GpuMoments<T>>,
    pub(super) scales: Option<GpuMoments<T>>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct GpuMoeAdamLayerState<T> {
    pub(super) router: GpuAdamLayerState<T>,
    pub(super) experts: Vec<GpuAdamLayerState<T>>,
}

#[derive(Clone, Debug)]
pub(super) struct GpuEmlAdamLayerState<T> {
    pub(super) input: GpuAdamLayerState<T>,
    pub(super) self_params: GpuAdamLayerState<T>,
}

impl<T> Default for GpuEmlAdamLayerState<T> {
    fn default() -> Self {
        Self {
            input: GpuAdamLayerState::default(),
            self_params: GpuAdamLayerState::default(),
        }
    }
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
pub(super) enum GpuOptimizerStep {
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
    pub(super) fn with_lr_scale(self, scale: f32) -> Self {
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
pub(super) struct GpuOptimTarget<'a, T> {
    pub(super) is_weight: bool,
    pub(super) w: &'a T,
    pub(super) u: &'a T,
    /// A pending (not yet accumulated) gradient, see `GpuPendingGrads`.
    pub(super) grad: Option<&'a T>,
    pub(super) moments: &'a mut Option<GpuMoments<T>>,
    pub(super) shape: &'a [usize],
}

/// Sanitises, then applies one SGD/Adam step to a layer's parameter tensors
/// (weights, biases, BN scales) in one backend call (C++
/// `update_convolutional_layer_gpu`: `reset_nan_and_inf(updates)`,
/// `fix_nan_and_inf(weights)` for weights, then the optimizer).  Returns the
/// new `(weights, updates)` per target; Adam zeroes the updates like
/// `adam_update_gpu`.
pub(super) fn gpu_optimizer_multi_step<B>(
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
pub(super) struct GpuConvolutionLayerParams<T> {
    pub(super) raw_weights: T,
    pub(super) raw_biases: T,
    pub(super) weights: T,
    pub(super) biases: T,
    pub(super) batch_norm_scales: Option<T>,
    /// Device-resident BN gamma and rolling stats (the source of truth during
    /// training; the host copies lag when `host_weights_stale`).
    pub(super) bn: Option<GpuBatchNormState<T>>,
    /// The raw weights/BN stats changed since `weights`/`biases`/
    /// `batch_norm_scales` (the BN-folded inference copies) were built; they
    /// are re-folded on the device the next time something needs them.
    pub(super) folded_stale: bool,
}

#[derive(Clone, Debug)]
pub(super) struct GpuBatchNormState<T> {
    pub(super) scales: T,
    pub(super) rolling_mean: T,
    pub(super) rolling_variance: T,
}

#[derive(Clone, Debug)]
pub(super) struct GpuConvolutionTrainBuffers<T> {
    pub(super) weight_updates: T,
    pub(super) bias_updates: T,
    pub(super) scale_updates: Option<T>,
    /// Gradient of the latest backward that is not yet added to the `*_updates`
    /// accumulators; the next optimizer step folds it in (`updates + grad`, the
    /// same addition the separate accumulate launch did).  Only set on the
    /// deferred training path (see `defer_host_sync`).
    pub(super) pending: Option<GpuPendingGrads<T>>,
}

#[derive(Clone, Debug)]
pub(super) struct GpuPendingGrads<T> {
    pub(super) weights: T,
    pub(super) biases: T,
    pub(super) scales: Option<T>,
}

#[derive(Clone, Debug)]
pub(super) struct GpuMoeTrainBuffers<T> {
    pub(super) router_weight_updates: T,
    pub(super) router_bias_updates: T,
    pub(super) expert_weight_updates: Vec<T>,
    pub(super) expert_bias_updates: Vec<T>,
    pub(super) expert_scale_updates: Vec<Option<T>>,
    /// Accumulated device gradients not yet flushed to the host optimizer.
    pub(super) dirty: bool,
}

/// Device gradients and SGD update buffers for EML's two convolution branches.
#[derive(Clone, Debug)]
pub(super) struct GpuEmlTrainBuffers<T> {
    pub(super) input: GpuConvolutionTrainBuffers<T>,
    pub(super) self_params: GpuConvolutionTrainBuffers<T>,
}

/// Device-resident DCNv4 Adam moments. They are materialized in the existing
/// host-sidecar schema only when optimizer state is saved.
#[derive(Clone, Debug)]
pub(super) struct GpuDcnv4AdamState<T> {
    pub(super) offset_weights: Option<GpuMoments<T>>,
    pub(super) offset_biases: Option<GpuMoments<T>>,
    pub(super) biases: Option<GpuMoments<T>>,
    pub(super) scales: Option<GpuMoments<T>>,
}

/// Native f32 DCNv4 parameter-gradient accumulators. Gradients stay on-device
/// through the native optimizer boundary and flush only for host fallback.
#[derive(Clone, Debug)]
pub(crate) struct GpuDcnv4TrainBuffers<T> {
    pub(crate) bias_updates: T,
    pub(crate) offset_weight_updates: T,
    pub(crate) offset_bias_updates: T,
    pub(crate) scale_updates: Option<T>,
    pub(crate) momentum_bias: T,
    pub(crate) momentum_offset_weights: T,
    pub(crate) momentum_offset_biases: T,
    pub(crate) momentum_scale: Option<T>,
    pub(crate) dirty: bool,
}
