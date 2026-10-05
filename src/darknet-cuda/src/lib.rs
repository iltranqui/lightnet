//! Backend-neutral CUDA tensor contracts used by the Rust runtime.
//!
//! This crate does not require a CUDA driver for CPU builds.  It defines the
//! operation traits and tensor metadata needed by the core executor; the
//! optional cuTile bridge supplies a concrete device implementation.

pub mod host_ops;
pub use host_ops::{AdamParams, NanInfMode, dropout_unit};

pub const CUTILE_RS_REFERENCE: &str = "https://github.com/NVlabs/cutile-rs";
pub const DEFAULT_TILE_1D: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F32ActivationKind {
    Linear,
    Logistic,
    Relu,
    Leaky,
    Mish,
    Swish,
    Tanh,
    /// `clamp(x, 0, 6)`; gradient 1 strictly inside (0, 6) (C++ `relu6_*`).
    Relu6,
    /// `x*(x>0) + 0.1x`; gradient `(x>0) + 0.1` (C++ `ramp_*`).
    Ramp,
    /// tanh-approximation GELU `0.5x(1+tanh(0.797885x+0.035677x^3))`; exact derivative (C++ `gelu_*`).
    Gelu,
}

/// Outputs of [`F32TensorOpsBackend::batchnorm_train_forward_f32`].
#[derive(Clone, Debug)]
pub struct BatchNormTrainForward<T> {
    /// `x_hat * scale + bias`, shaped like the input.
    pub output: T,
    /// Normalised input, needed by the backward pass.
    pub x_hat: T,
    /// Per-channel batch mean.
    pub mean: T,
    /// Per-channel biased batch variance.
    pub variance: T,
    pub rolling_mean: T,
    pub rolling_variance: T,
}

/// One tensor of a multi-tensor SGD step (see
/// [`F32TensorOpsBackend::sanitized_sgd_update_multi_f32`]).
#[derive(Clone, Debug)]
pub struct SgdUpdateItem<'a, T> {
    pub weights: &'a T,
    pub updates: &'a T,
    /// A not yet accumulated gradient: the step uses `updates + grad`
    /// (exactly `saxpy_f32(1.0, grad, updates)` first).
    pub grad: Option<&'a T>,
    pub shape: &'a [usize],
    pub fix_weights: bool,
    /// Weight decay (0 for biases / BN scales).
    pub decay: f32,
}

/// One tensor of a multi-tensor Adam step (see
/// [`F32TensorOpsBackend::sanitized_adam_update_multi_f32`]).
#[derive(Clone, Debug)]
pub struct AdamUpdateItem<'a, T> {
    pub weights: &'a T,
    pub updates: &'a T,
    pub grad: Option<&'a T>,
    pub m: &'a T,
    pub v: &'a T,
    pub shape: &'a [usize],
    pub fix_weights: bool,
    pub params: AdamParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F16,
    BF16,
    F32,
    I32,
    U8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
    RowMajor,
    Flat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelLanguage {
    CudaC,
    Ptx,
    CutileDsl,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceTensorDesc {
    pub name: String,
    pub shape: [usize; 4],
    pub scalar_type: ScalarType,
    pub layout: TensorLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarArg {
    I32(i32),
    U32(u32),
    F32(f32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionStream {
    pub id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelModule {
    pub name: String,
    pub language: KernelLanguage,
    pub source: String,
    pub entry_points: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelLaunch {
    pub entry_point: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
    pub tensors: Vec<DeviceTensorDesc>,
    pub scalars: Vec<ScalarArg>,
    pub stream: ExecutionStream,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledKernel {
    pub module_name: String,
    pub entry_point: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementwiseLaunchConfig {
    pub tile_len: usize,
}

impl Default for ElementwiseLaunchConfig {
    fn default() -> Self {
        Self {
            tile_len: DEFAULT_TILE_1D,
        }
    }
}

pub trait CudaBackend {
    type Error;

    fn backend_name(&self) -> &'static str;
    fn compile(&self, module: &KernelModule) -> Result<Vec<CompiledKernel>, Self::Error>;
    fn launch(&self, kernel: &CompiledKernel, launch: &KernelLaunch) -> Result<(), Self::Error>;
}

// `KernelLaunch` describes generic launches, but a real runtime also needs typed
// device buffers. This trait is the first concrete execution seam for Rust-side
// tensor ops while larger Darknet kernels are still being ported.
//
// The operation signatures intentionally keep the complete Darknet kernel
// geometry at the call site. Grouping those values into opaque descriptors
// would make the backend API less explicit and would hide layout mistakes.
#[allow(clippy::too_many_arguments)]
pub trait F32TensorOpsBackend {
    type Error;
    type Tensor: Clone;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error>;
    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error>;
    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;
    /// True if any element of `x` is NaN or infinite.  The default downloads
    /// and scans on the host; backends may override with a device reduction.
    fn any_non_finite_f32(&self, x: &Self::Tensor) -> Result<bool, Self::Error> {
        Ok(self.download_f32(x)?.iter().any(|value| !value.is_finite()))
    }
    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// `activation(bias_add(x, bias))` (inference epilogue).  Backends may
    /// fuse it into one launch; the result must equal the composition below.
    fn bias_activation_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let biased = self.bias_add_f32(x, bias, channels, channel_stride, config)?;
        self.activation_f32(&biased, activation, config)
    }
    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    /// Bilinearly sample an NCHW tensor using one source coordinate pair per
    /// batch/output pixel. `input_scales` and `input_offsets` contain one
    /// affine level adjustment per batch and are applied after interpolation.
    fn crop_bilinear_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        input_offsets: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            input_offsets,
            config,
        );
        panic!("crop_bilinear_f32 is not implemented by this CUDA backend")
    }
    /// Scatter the bilinear crop gradient back into the NCHW input. Out of
    /// bounds taps are ignored, matching Darknet's zero-padded sampler.
    fn crop_bilinear_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = (
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            config,
        );
        panic!("crop_bilinear_backward_f32 is not implemented by this CUDA backend")
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
    ) -> Result<Self::Tensor, Self::Error>;
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
    ) -> Result<Self::Tensor, Self::Error>;
    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error>;
    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error>;

    /// Darknet shortcut resize for sources whose shape differs from the layer
    /// output: per batch sample the flat `x.len() / batch` elements are
    /// cropped or zero-padded to `output_shape`'s per-sample length (the C++
    /// `shortcut_multilayer_*` adds `src_i < min(src_outputs, outputs)` flat).
    /// Forward: `shortcut_add_f32(input, shortcut_fit_f32(source, ..))`;
    /// backward: the source gradient is `shortcut_fit_f32(layer_grad, ..)`.
    fn shortcut_fit_f32(
        &self,
        x: &Self::Tensor,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let out_len: usize = output_shape.iter().product();
        let fitted = host_ops::shortcut_fit(&host, batch, out_len / batch.max(1));
        self.upload_f32(output_shape, &fitted)
    }
    /// Row-wise LayerNorm over a flat `rows x cols` buffer:
    /// `y = gamma * (x - mean) * inv_std + beta`, population variance.
    ///
    /// `gamma` and `beta` may be `None` for the bias-free variant that
    /// ModernBERT's `attn_norm` / `mlp_norm` and every `[laya_*]` LayerNorm use.
    /// `shape` is the output tensor's shape.  Shared building block for the
    /// transformer family (see `tasks/rust/layers.md` Phase 3A prerequisite).
    fn layernorm_forward_f32(
        &self,
        x: &Self::Tensor,
        gamma: Option<&Self::Tensor>,
        beta: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let g = match gamma {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        let b = match beta {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(shape, &host_ops::layernorm(&host, rows, cols, g.as_deref(), b.as_deref(), eps))
    }
    /// Row-wise LayerNorm backward: the input gradient only, with the reduction
    /// statistics recomputed from `x` (`dx = g - mean(g) - xhat * mean(g*xhat)`).
    ///
    /// `gamma` may be `None`.  Parameter gradients are not produced here: every
    /// LayerNorm in the ported families is either frozen or trained by a path
    /// that folds them in separately.
    fn layernorm_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        x: &Self::Tensor,
        gamma: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let dy = self.download_f32(output_grad)?;
        let host = self.download_f32(x)?;
        let g = match gamma {
            Some(t) => Some(self.download_f32(t)?),
            None => None,
        };
        self.upload_f32(shape, &host_ops::layernorm_backward(&dy, &host, rows, cols, g.as_deref(), eps))
    }
    /// Softmax over every contiguous `group_size` run of `x` (max-subtracted);
    /// `shape` is the tensor shape of `x` and the output.
    fn softmax_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::softmax(&host, group_size))
    }
    /// `dx = y * (dy - sum(y * dy))` per softmax group (the full Jacobian,
    /// matching the Rust CPU runtime; Darknet C++ passes `dy` through because
    /// its cost layer already folds the softmax derivative in).
    fn softmax_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let y = self.download_f32(output)?;
        let dy = self.download_f32(output_grad)?;
        self.upload_f32(shape, &host_ops::softmax_backward(&y, &dy, group_size))
    }
    /// Darknet reorg on NCHW `[batch, c, h, w]` shapes.  Non-reverse folds
    /// `stride x stride` spatial blocks into channels (`c * stride^2`, `h /
    /// stride`); `reverse` is the inverse layout.  Channel `k` of the small
    /// tensor reads big channel `k % c_big` at block offset `k / c_big`.
    fn reorg_f32(
        &self,
        x: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let d = host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .unwrap_or_else(|err| panic!("{err}"));
        let host = self.download_f32(x)?;
        let out = if reverse { host_ops::reorg_to_big(&host, &d) } else { host_ops::reorg_to_small(&host, &d) };
        self.upload_f32(output_shape, &out)
    }
    /// Exact adjoint of [`Self::reorg_f32`]: returns the gradient of the layer
    /// input (`input_shape`) from the output gradient.
    fn reorg_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let d = host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .unwrap_or_else(|err| panic!("{err}"));
        let host = self.download_f32(output_grad)?;
        let dx = if reverse { host_ops::reorg_to_small(&host, &d) } else { host_ops::reorg_to_big(&host, &d) };
        self.upload_f32(input_shape, &dx)
    }
    /// `[channel_shuffle]` on NCHW `shape` (see [`host_ops::channel_permute`]): forward
    /// (`inverse = false`) uses rows = groups, cols = c / groups; `inverse = true` is the exact
    /// adjoint (rows = c / groups, cols = groups) used by the backward pass.
    fn channel_shuffle_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4 && groups > 0 && shape[1] % groups == 0,
            "channel_shuffle_f32: bad shape {shape:?} or groups {groups}"
        );
        let (rows, cols) = if inverse { (shape[1] / groups, groups) } else { (groups, shape[1] / groups) };
        let host = self.download_f32(x)?;
        let out = host_ops::channel_permute(&host, shape[0], rows, cols, shape[2] * shape[3]);
        self.upload_f32(shape, &out)
    }
    /// Inverted dropout: element `i` is zeroed when `dropout_unit(seed, i) <
    /// probability`, otherwise scaled by `1 / (1 - probability)`.  Returns
    /// `(output, mask)` where `mask` holds the per-element multiplier.
    fn dropout_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        probability: f32,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let (out, mask) = host_ops::dropout(&host, probability, seed);
        Ok((self.upload_f32(shape, &out)?, self.upload_f32(shape, &mask)?))
    }
    /// `dx = dy * mask`.
    fn dropout_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        mask: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let dy = self.download_f32(output_grad)?;
        let m = self.download_f32(mask)?;
        let dx: Vec<f32> = dy.iter().zip(&m).map(|(a, b)| a * b).collect();
        self.upload_f32(shape, &dx)
    }
    /// DropBlock forward (see [`host_ops::dropblock`]): `shape` is NCHW,
    /// `block_prob` the per-cell selection probability and `block_size` the
    /// square side (`1..=min(w, h)`).  Returns `(output, mask)` with the
    /// per-element multiplier, so the backward is [`Self::dropout_backward_f32`].
    fn dropblock_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        block_prob: f32,
        block_size: usize,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        let (out, mask) = host_ops::dropblock(&host, shape, block_prob, block_size, seed);
        Ok((self.upload_f32(shape, &out)?, self.upload_f32(shape, &mask)?))
    }
    /// Darknet `constrain_ongpu`: out-of-place `min(limit, max(-limit, x))`.
    fn constrain_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        limit: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::constrain(&host, limit))
    }
    /// Darknet `gradient_centralization_gpu` on weight updates laid out as
    /// `filters` contiguous runs of `f_size` values (no-op unless
    /// `f_size % 32 == 0`, like C++).
    fn gradient_centralization_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        f_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::gradient_centralization(&host, f_size))
    }
    /// Darknet scale_channels pre-activation: `out[b,c,y,x] = scale * source`
    /// with `scale` shaped `[batch, h*w]` (`scale_wh`) or `[batch, c]`.
    /// `shape` is the NCHW shape of `source` / the output.
    fn scale_channels_f32(
        &self,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(shape.len() == 4, "scale_channels_f32: shape must be NCHW");
        let s = self.download_f32(scale)?;
        let src = self.download_f32(source)?;
        let out = host_ops::scale_channels(&s, &src, shape[1], shape[2] * shape[3], scale_wh);
        self.upload_f32(shape, &out)
    }
    /// `(d_scale, d_source)` for upstream gradient `delta` (already through
    /// the activation derivative).
    fn scale_channels_backward_f32(
        &self,
        delta: &Self::Tensor,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(shape.len() == 4, "scale_channels_backward_f32: shape must be NCHW");
        let g = self.download_f32(delta)?;
        let s = self.download_f32(scale)?;
        let src = self.download_f32(source)?;
        let (d_scale, d_source) =
            host_ops::scale_channels_backward(&g, &s, &src, shape[1], shape[2] * shape[3], scale_wh);
        let scale_shape = if scale_wh {
            [shape[0], 1, shape[2], shape[3]]
        } else {
            [shape[0], shape[1], 1, 1]
        };
        Ok((
            self.upload_f32(&scale_shape, &d_scale)?,
            self.upload_f32(shape, &d_source)?,
        ))
    }
    /// Connected layer pre-activation `out[b, o] = x[b, :] . w[o, :] + bias[o]`
    /// (weights `[outputs][inputs]`, Darknet layout), output `[batch, outputs]`.
    fn connected_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let xh = self.download_f32(x)?;
        let wh = self.download_f32(weights)?;
        let bh = self.download_f32(biases)?;
        let out = host_ops::connected(&xh, &wh, &bh, batch, inputs, outputs);
        self.upload_f32(&[batch, outputs], &out)
    }
    /// `(dx, dw, db)` of [`Self::connected_f32`] for `delta` `[batch, outputs]`
    /// (already through the activation derivative).
    fn connected_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        delta: &Self::Tensor,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let xh = self.download_f32(x)?;
        let wh = self.download_f32(weights)?;
        let dh = self.download_f32(delta)?;
        let (dx, dw, db) = host_ops::connected_backward(&xh, &wh, &dh, batch, inputs, outputs);
        Ok((
            self.upload_f32(&[batch, inputs], &dx)?,
            self.upload_f32(&[outputs, inputs], &dw)?,
            self.upload_f32(&[outputs], &db)?,
        ))
    }
    /// Batch-norm training forward on NCHW `shape`: batch mean / variance over
    /// `(batch, h, w)` per channel, normalise, scale + bias, and the rolling
    /// EMA `new = keep * old + (1 - keep) * batch` (unbiased variance for the
    /// rolling estimate).  Reductions are deterministic.
    fn batchnorm_train_forward_f32(
        &self,
        x: &Self::Tensor,
        scales: &Self::Tensor,
        biases: &Self::Tensor,
        rolling_mean: &Self::Tensor,
        rolling_variance: &Self::Tensor,
        shape: &[usize],
        keep: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<BatchNormTrainForward<Self::Tensor>, Self::Error> {
        let _ = config;
        assert!(shape.len() == 4, "batchnorm_train_forward_f32: shape must be NCHW");
        let r = host_ops::batchnorm_train_forward(
            &self.download_f32(x)?,
            &self.download_f32(scales)?,
            &self.download_f32(biases)?,
            &self.download_f32(rolling_mean)?,
            &self.download_f32(rolling_variance)?,
            shape[1],
            shape[2] * shape[3],
            keep,
        );
        let c = [shape[1]];
        Ok(BatchNormTrainForward {
            output: self.upload_f32(shape, &r.output)?,
            x_hat: self.upload_f32(shape, &r.x_hat)?,
            mean: self.upload_f32(&c, &r.mean)?,
            variance: self.upload_f32(&c, &r.variance)?,
            rolling_mean: self.upload_f32(&c, &r.rolling_mean)?,
            rolling_variance: self.upload_f32(&c, &r.rolling_variance)?,
        })
    }
    /// `(dx, dscale, dbias)` of [`Self::batchnorm_train_forward_f32`] for
    /// upstream gradient `output_grad` (already through the activation
    /// derivative); `variance` is the biased batch variance it returned.
    fn batchnorm_train_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        x_hat: &Self::Tensor,
        variance: &Self::Tensor,
        scales: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        assert!(shape.len() == 4, "batchnorm_train_backward_f32: shape must be NCHW");
        let (dx, dscale, dbias) = host_ops::batchnorm_train_backward(
            &self.download_f32(output_grad)?,
            &self.download_f32(x_hat)?,
            &self.download_f32(variance)?,
            &self.download_f32(scales)?,
            shape[1],
            shape[2] * shape[3],
        );
        Ok((
            self.upload_f32(shape, &dx)?,
            self.upload_f32(&[shape[1]], &dscale)?,
            self.upload_f32(&[shape[1]], &dbias)?,
        ))
    }
    /// Darknet `reset_nan_and_inf` / `fix_nan_and_inf` (see [`NanInfMode`]),
    /// out of place.
    fn sanitize_nan_inf_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        mode: NanInfMode,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        let host = self.download_f32(x)?;
        self.upload_f32(shape, &host_ops::sanitize_nan_inf(&host, mode))
    }
    /// Darknet `adam_update_gpu` (see [`AdamParams`]): out of place, returns
    /// `(weights', m', v')`; the caller zeroes the accumulated `updates`.
    fn adam_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        m: &Self::Tensor,
        v: &Self::Tensor,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let (w, m1, v1) = host_ops::adam_update(
            &self.download_f32(weights)?,
            &self.download_f32(updates)?,
            &self.download_f32(m)?,
            &self.download_f32(v)?,
            params,
        );
        let shape = [w.len()];
        Ok((
            self.upload_f32(&shape, &w)?,
            self.upload_f32(&shape, &m1)?,
            self.upload_f32(&shape, &v1)?,
        ))
    }

    /// Fused `reset_nan_and_inf(updates)` + (`fix_weights`) `fix_nan_and_inf(weights)`
    /// + [`Self::sgd_update_f32`] (decay already `0` for biases / BN scales).
    /// Backends override it to do the three steps in one launch; the result is
    /// identical to the composition below.
    fn sanitized_sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        shape: &[usize],
        fix_weights: bool,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let u = self.sanitize_nan_inf_f32(updates, shape, NanInfMode::Reset, config)?;
        let w = if fix_weights {
            self.sanitize_nan_inf_f32(weights, shape, NanInfMode::Fix, config)?
        } else {
            weights.clone()
        };
        self.sgd_update_f32(&w, &u, lr, momentum, decay, batch, config)
    }

    /// Fused sanitize + [`Self::adam_update_f32`]; returns
    /// `(weights', m', v', zeroed updates)` (Adam clears the accumulated updates).
    fn sanitized_adam_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        m: &Self::Tensor,
        v: &Self::Tensor,
        shape: &[usize],
        fix_weights: bool,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let u = self.sanitize_nan_inf_f32(updates, shape, NanInfMode::Reset, config)?;
        let w = if fix_weights {
            self.sanitize_nan_inf_f32(weights, shape, NanInfMode::Fix, config)?
        } else {
            weights.clone()
        };
        let (w, m, v) = self.adam_update_f32(&w, &u, m, v, params, config)?;
        Ok((w, m, v, self.zeros_f32(shape)?))
    }

    /// [`Self::sanitized_sgd_update_f32`] over several tensors (one layer's
    /// weights, biases and BN scales), in the same order as `items`; results
    /// are bit-identical to per-tensor calls.  Backends may do it in one launch.
    fn sanitized_sgd_update_multi_f32(
        &self,
        items: &[SgdUpdateItem<'_, Self::Tensor>],
        lr: f32,
        momentum: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor)>, Self::Error> {
        items
            .iter()
            .map(|it| {
                let acc;
                let updates = match it.grad {
                    Some(g) => {
                        acc = self.saxpy_f32(1.0, g, it.updates, config)?;
                        &acc
                    }
                    None => it.updates,
                };
                self.sanitized_sgd_update_f32(
                    it.weights,
                    updates,
                    it.shape,
                    it.fix_weights,
                    lr,
                    momentum,
                    it.decay,
                    batch,
                    config,
                )
            })
            .collect()
    }

    /// [`Self::sanitized_adam_update_f32`] over several tensors; returns
    /// `(weights', m', v', zeroed updates)` per item.
    fn sanitized_adam_update_multi_f32(
        &self,
        items: &[AdamUpdateItem<'_, Self::Tensor>],
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor)>, Self::Error> {
        items
            .iter()
            .map(|it| {
                let acc;
                let updates = match it.grad {
                    Some(g) => {
                        acc = self.saxpy_f32(1.0, g, it.updates, config)?;
                        &acc
                    }
                    None => it.updates,
                };
                self.sanitized_adam_update_f32(
                    it.weights,
                    updates,
                    it.m,
                    it.v,
                    it.shape,
                    it.fix_weights,
                    &it.params,
                    config,
                )
            })
            .collect()
    }

    /// Per-channel inference fold of batch norm: `scale = gamma / sqrt(var + eps)`
    /// and `bias' = bias - mean * scale`.  Returns `(scale, bias')`.
    fn batchnorm_fold_f32(
        &self,
        gamma: &Self::Tensor,
        mean: &Self::Tensor,
        variance: &Self::Tensor,
        bias: &Self::Tensor,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let _ = config;
        let (g, m, v, b) = (
            self.download_f32(gamma)?,
            self.download_f32(mean)?,
            self.download_f32(variance)?,
            self.download_f32(bias)?,
        );
        let scale: Vec<f32> = g.iter().zip(&v).map(|(g, v)| g / (v + eps).sqrt()).collect();
        let folded: Vec<f32> = b.iter().zip(&m).zip(&scale).map(|((b, m), s)| b - m * s).collect();
        let shape = [scale.len()];
        Ok((self.upload_f32(&shape, &scale)?, self.upload_f32(&shape, &folded)?))
    }
}

/// Opt-in half-precision (binary16) INFERENCE surface on top of the f32 ops.
///
/// Weights and activations live on the device as f16 (`TensorF16`), GEMM
/// accumulation, bias and activation are f32, results are rounded to nearest
/// even on store.  Only the ops an inference graph needs exist; there is no
/// backward and no f16 training (training stays f32).  Weights are prepared
/// from the backend's f32 tensors, so the runtime keeps one f32 source of
/// truth (folded BN) and caches the converted f16 copy.
#[allow(clippy::too_many_arguments)]
pub trait F16TensorOpsBackend: F32TensorOpsBackend {
    /// `'static` so the runtime can cache it type-erased per layer.
    type TensorF16: Clone + 'static;

    /// Uploads host f32 data and rounds it to f16 on the device.
    fn upload_f16(&self, shape: &[usize], host: &[f32]) -> Result<Self::TensorF16, Self::Error>;
    /// Downloads an f16 tensor widened to f32.
    fn download_f16(&self, tensor: &Self::TensorF16) -> Result<Vec<f32>, Self::Error>;
    /// Converts `[filters, per_filter]` f32 conv weights to the backend's f16
    /// GEMM operand layout (backend-defined padding, opaque to the caller).
    fn conv_weights_f16(
        &self,
        weights: &Self::Tensor,
        filters: usize,
        per_filter: usize,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(conv(x, weights) + bias))`; `bias` is f32 `[filters]`.
    fn conv2d_bias_act_f16(
        &self,
        x: &Self::TensorF16,
        weights: &Self::TensorF16,
        bias: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn maxpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn upsample_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    fn route_concat_f16(
        &self,
        inputs: &[&Self::TensorF16],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Mean over the in-image taps of each window.
    fn local_avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Global average pool to `[batch, channels, 1, 1]`.
    fn avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// Flat per-sample crop / zero-pad of a mismatched shortcut source.
    fn shortcut_fit_f16(
        &self,
        x: &Self::TensorF16,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(a + b))`, same shapes (shortcut layer).
    fn shortcut_add_act_f16(
        &self,
        a: &Self::TensorF16,
        b: &Self::TensorF16,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
    /// `f16(activation(x * scale[c] + bias[c]))` per channel (`scale = None`:
    /// bias + activation only); `scale` / `bias` are f32 `[channels]`.
    fn channel_affine_act_f16(
        &self,
        x: &Self::TensorF16,
        scale: Option<&Self::Tensor>,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error>;
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CutilePlanningNote {
    pub target_sm: String,
    pub tile_shape: [usize; 2],
    pub async_stages: usize,
    pub notes: String,
}

#[derive(Clone, Debug, Default)]
pub struct StubCudaBackend;

impl CudaBackend for StubCudaBackend {
    type Error = String;

    fn backend_name(&self) -> &'static str {
        "stub"
    }

    fn compile(&self, module: &KernelModule) -> Result<Vec<CompiledKernel>, Self::Error> {
        Ok(module
            .entry_points
            .iter()
            .map(|entry_point| CompiledKernel {
                module_name: module.name.clone(),
                entry_point: entry_point.clone(),
            })
            .collect())
    }

    fn launch(&self, _kernel: &CompiledKernel, _launch: &KernelLaunch) -> Result<(), Self::Error> {
        Err("no CUDA backend is wired into this workspace yet".to_owned())
    }
}

pub fn default_cutile_plan(kernel_name: impl Into<String>) -> CutilePlanningNote {
    CutilePlanningNote {
        target_sm: "sm_80+".to_owned(),
        tile_shape: [128, 128],
        async_stages: 2,
        notes: format!(
            "plan for {name}: map tensor tiles and host-side async launches through a future cutile-rs backend",
            name = kernel_name.into()
        ),
    }
}
