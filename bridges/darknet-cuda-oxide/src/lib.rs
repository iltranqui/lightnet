//! `F32TensorOpsBackend` implemented with kernels written in Rust and
//! compiled by cuda-oxide (`cargo oxide`).  Build this crate only through
//! `cargo oxide`; plain `cargo build` cannot compile the device code.

pub mod kernels;
pub mod launch;
pub mod tensor;

use std::fmt;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use kernels::bias_scale::BIAS_BACKWARD_THREADS;
use kernels::{activations, bias_scale, elementwise, optim};
pub use tensor::OxideTensorF32;

#[derive(Debug)]
pub enum OxideError {
    Cuda(DriverError),
    Shape(String),
    /// The op has no cuda-oxide kernel yet.
    NotImplemented(&'static str),
}

impl fmt::Display for OxideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(err) => write!(f, "CUDA driver error: {err:?}"),
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
            Self::NotImplemented(op) => write!(f, "{op}: not yet implemented in oxide backend"),
        }
    }
}

impl std::error::Error for OxideError {}

impl From<DriverError> for OxideError {
    fn from(err: DriverError) -> Self {
        Self::Cuda(err)
    }
}

pub struct OxideBackend {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    elementwise: elementwise::kernels::LoadedModule,
    activations: activations::kernels::LoadedModule,
    bias_scale: bias_scale::kernels::LoadedModule,
    optim: optim::kernels::LoadedModule,
}

fn activation_code(kind: F32ActivationKind) -> u32 {
    match kind {
        F32ActivationKind::Linear => activations::ACT_LINEAR,
        F32ActivationKind::Logistic => activations::ACT_LOGISTIC,
        F32ActivationKind::Relu => activations::ACT_RELU,
        F32ActivationKind::Leaky => activations::ACT_LEAKY,
        F32ActivationKind::Mish => activations::ACT_MISH,
        F32ActivationKind::Swish => activations::ACT_SWISH,
        F32ActivationKind::Tanh => activations::ACT_TANH,
    }
}

fn same_len(op: &str, a: &OxideTensorF32, b: &OxideTensorF32) -> Result<(), OxideError> {
    if a.len() != b.len() {
        return Err(OxideError::Shape(format!(
            "{op} shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

/// Validates `[rows, channel_stride]` layout with one value per channel.
fn check_channel_layout(
    op: &str,
    x_len: usize,
    per_channel_len: Option<usize>,
    channels: usize,
    channel_stride: usize,
) -> Result<(), OxideError> {
    if channels == 0 || channel_stride == 0 {
        return Err(OxideError::Shape(format!(
            "{op} requires channels > 0 and channel_stride > 0"
        )));
    }
    if per_channel_len.is_some_and(|len| len != channels) {
        return Err(OxideError::Shape(format!(
            "{op} per-channel length mismatch: expected {channels}, got {per_channel_len:?}"
        )));
    }
    if x_len % channel_stride != 0 || (x_len / channel_stride) % channels != 0 {
        return Err(OxideError::Shape(format!(
            "{op} length {x_len} is not divisible by channel_stride {channel_stride} * channels {channels}"
        )));
    }
    Ok(())
}

impl OxideBackend {
    pub fn new() -> Result<Self, OxideError> {
        Self::with_device(0)
    }

    pub fn with_device(ordinal: usize) -> Result<Self, OxideError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        // Each `#[cuda_module]` loads the crate's embedded PTX bundle once.
        let elementwise = elementwise::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load elementwise module: {err:?}")))?;
        let activations = activations::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load activations module: {err:?}")))?;
        let bias_scale = bias_scale::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load bias_scale module: {err:?}")))?;
        let optim = optim::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load optim module: {err:?}")))?;
        Ok(Self {
            _ctx: ctx,
            stream,
            elementwise,
            activations,
            bias_scale,
            optim,
        })
    }

    fn alloc(&self, len: usize) -> Result<DeviceBuffer<f32>, OxideError> {
        Ok(DeviceBuffer::<f32>::zeroed(&self.stream, len)?)
    }
}

fn launch_err(err: impl fmt::Debug) -> OxideError {
    OxideError::Shape(format!("kernel launch failed: {err:?}"))
}

impl F32TensorOpsBackend for OxideBackend {
    type Error = OxideError;
    type Tensor = OxideTensorF32;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error> {
        let expected: usize = shape.iter().product();
        if expected != host.len() {
            return Err(OxideError::Shape(format!(
                "upload_f32 shape {shape:?} expects {expected} elements, got {}",
                host.len()
            )));
        }
        let buf = DeviceBuffer::from_host(&self.stream, host)?;
        Ok(OxideTensorF32::new(buf, shape))
    }

    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        let buf = self.alloc(shape.iter().product())?;
        Ok(OxideTensorF32::new(buf, shape))
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        Ok(tensor.buf.to_host_vec(&self.stream)?)
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        check_channel_layout(
            "bias_add_f32",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.bias_add(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        check_channel_layout(
            "channel_scale_f32",
            x.len(),
            Some(scale.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.channel_scale(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &scale.buf,
                &mut out,
                channels,
                channel_stride,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        check_channel_layout(
            "bias_backward_f32",
            output_grad.len(),
            None,
            channels,
            channel_stride,
        )?;
        let row_groups = output_grad.len() / channel_stride / channels;
        let mut partial = self.alloc(channels * BIAS_BACKWARD_THREADS)?;
        let mut out = self.alloc(channels)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.bias_scale.bias_backward_partial(
                &self.stream,
                launch::blocks_of(channels, BIAS_BACKWARD_THREADS as u32),
                &output_grad.buf,
                &mut partial,
                channels,
                channel_stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: one thread per channel; `partial` holds channels * 256 sums.
        unsafe {
            self.bias_scale.bias_backward_reduce(
                &self.stream,
                launch::elementwise(channels, ElementwiseLaunchConfig::default()),
                &partial,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[channels]))
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut out = self.alloc(x.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.activations.activation(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        same_len("activation_backward_f32", activation_input, delta)?;
        let mut out = self.alloc(delta.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.activations.activation_backward(
                &self.stream,
                launch::elementwise(delta.len(), config),
                &activation_input.buf,
                &delta.buf,
                &mut out,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, activation_input.shape()))
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.vector_add_f32(x, y, config)
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        same_len("vector_add_f32", x, y)?;
        let mut out = self.alloc(x.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.elementwise.vector_add(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &y.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, y.shape()))
    }

    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        same_len("saxpy_f32", x, y)?;
        let mut out = self.alloc(x.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.elementwise.saxpy(
                &self.stream,
                launch::elementwise(x.len(), config),
                alpha,
                &x.buf,
                &y.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, y.shape()))
    }

    /// Out of place: returns fresh `(weights, updates)` tensors and leaves the
    /// inputs untouched (tensors are shared `Arc` buffers).
    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        same_len("sgd_update_f32", weights, updates)?;
        let batch = batch.max(1) as f32;
        let mut new_weights = self.alloc(weights.len())?;
        let mut new_updates = self.alloc(weights.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.optim.sgd_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &mut new_weights,
                &mut new_updates,
                lr / batch,
                momentum,
                decay * batch,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, weights.shape()),
            OxideTensorF32::new(new_updates, updates.shape()),
        ))
    }

    // ---- Ops without an oxide kernel yet: explicit errors, no fallbacks. ----

    fn route_concat_f32(
        &self,
        _inputs: &[&Self::Tensor],
        _input_channel_offsets: &[usize],
        _input_channels: &[usize],
        _input_total_channels: &[usize],
        _output_shape: &[usize],
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("route_concat_f32"))
    }

    fn route_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        _input_shapes: &[Vec<usize>],
        _input_channel_offsets: &[usize],
        _input_channels: &[usize],
        _input_total_channels: &[usize],
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        Err(OxideError::NotImplemented("route_backward_f32"))
    }

    fn conv2d_f32(
        &self,
        _x: &Self::Tensor,
        _weights: &Self::Tensor,
        _output_shape: &[usize],
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
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("conv2d_f32"))
    }

    fn conv2d_backward_input_f32(
        &self,
        _output_grad: &Self::Tensor,
        _weights: &Self::Tensor,
        _input_shape: &[usize],
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
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("conv2d_backward_input_f32"))
    }

    fn conv2d_backward_weights_f32(
        &self,
        _x: &Self::Tensor,
        _output_grad: &Self::Tensor,
        _weight_shape: &[usize],
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
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("conv2d_backward_weights_f32"))
    }

    fn maxpool_f32(
        &self,
        _x: &Self::Tensor,
        _output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("maxpool_f32"))
    }

    fn maxpool_backward_f32(
        &self,
        _x: &Self::Tensor,
        _output_grad: &Self::Tensor,
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("maxpool_backward_f32"))
    }

    fn local_avgpool_f32(
        &self,
        _x: &Self::Tensor,
        _output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("local_avgpool_f32"))
    }

    fn local_avgpool_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        _input_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("local_avgpool_backward_f32"))
    }

    fn avgpool_f32(
        &self,
        _x: &Self::Tensor,
        _output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("avgpool_f32"))
    }

    fn avgpool_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        _input_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("avgpool_backward_f32"))
    }

    fn crop_bilinear_f32(
        &self,
        _x: &Self::Tensor,
        _output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _source_x: &Self::Tensor,
        _source_y: &Self::Tensor,
        _input_scales: &Self::Tensor,
        _input_offsets: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("crop_bilinear_f32"))
    }

    fn crop_bilinear_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        _input_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _source_x: &Self::Tensor,
        _source_y: &Self::Tensor,
        _input_scales: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("crop_bilinear_backward_f32"))
    }

    fn upsample_f32(
        &self,
        _x: &Self::Tensor,
        _output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _stride: usize,
        _scale: f32,
        _reverse: bool,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("upsample_f32"))
    }

    fn upsample_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        _input_shape: &[usize],
        _channels: usize,
        _output_height: usize,
        _output_width: usize,
        _stride: usize,
        _scale: f32,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        Err(OxideError::NotImplemented("upsample_backward_f32"))
    }
}
