//! `F32TensorOpsBackend` implemented with kernels written in Rust and
//! compiled by cuda-oxide (`cargo oxide`).  Build this crate only through
//! `cargo oxide`; plain `cargo build` cannot compile the device code.

pub mod conv;
pub mod kernels;
pub mod launch;
pub mod tensor;

use std::fmt;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use kernels::bias_scale::BIAS_BACKWARD_THREADS;
use kernels::{activations, bias_scale, conv_direct, elementwise, gemm, im2col, optim};
pub use conv::ConvMode;
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
    gemm: gemm::kernels::LoadedModule,
    im2col: im2col::kernels::LoadedModule,
    conv_direct: conv_direct::kernels::LoadedModule,
    conv_mode: ConvMode,
    /// im2col / dcol scratch, grown on demand and reused across conv calls.
    workspace: Mutex<Option<DeviceBuffer<f32>>>,
}

/// One (batched) GEMM: `C = alpha * op(A) * op(B) + beta * C`, row-major.
///
/// `op(A)` is `m x k` (stored `k x m` with leading dim `lda` when `trans_a`),
/// `op(B)` is `k x n`, `C` is `m x n` with leading dim `ldc`.  `a` / `b` / `c`
/// are `(base, stride, modulus)`: instance `i < count` starts at
/// `base + (i % modulus) * stride` elements.  Backends with tensor-core GEMMs
/// can implement the same description.
#[derive(Clone, Copy, Debug)]
pub struct GemmArgs {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
    pub a: (usize, usize, usize),
    pub b: (usize, usize, usize),
    pub c: (usize, usize, usize),
    pub count: usize,
}

impl Default for GemmArgs {
    fn default() -> Self {
        Self {
            m: 0,
            n: 0,
            k: 0,
            lda: 0,
            ldb: 0,
            ldc: 0,
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 0.0,
            a: (0, 0, 1),
            b: (0, 0, 1),
            c: (0, 0, 1),
            count: 1,
        }
    }
}

/// Highest element index (+1) touched by a stored `rows x cols` matrix.
fn extent(base: usize, stride: usize, modulus: usize, count: usize, rows: usize, cols: usize, ld: usize) -> usize {
    let inst = count.min(modulus).saturating_sub(1);
    base + inst * stride + (rows - 1) * ld + cols
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
        let gemm = gemm::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm module: {err:?}")))?;
        let im2col = im2col::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load im2col module: {err:?}")))?;
        let conv_direct = conv_direct::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load conv_direct module: {err:?}")))?;
        Ok(Self {
            _ctx: ctx,
            stream,
            elementwise,
            activations,
            bias_scale,
            optim,
            gemm,
            im2col,
            conv_direct,
            conv_mode: ConvMode::from_env(),
            workspace: Mutex::new(None),
        })
    }

    /// Overrides the conv implementation chosen from `DARKNET_OXIDE_CONV`.
    pub fn with_conv_mode(mut self, mode: ConvMode) -> Self {
        self.conv_mode = mode;
        self
    }

    pub fn conv_mode(&self) -> ConvMode {
        self.conv_mode
    }

    /// Validated GEMM launch.  This is the single seam conv (and later
    /// `connected`) goes through; alternative kernels plug in here.
    pub(crate) fn gemm(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm: {msg}"));
        if g.m == 0 || g.n == 0 || g.k == 0 || g.count == 0 {
            return Err(bad("m, n, k and count must be > 0"));
        }
        if g.a.2 == 0 || g.b.2 == 0 || g.c.2 == 0 {
            return Err(bad("instance modulus must be > 0"));
        }
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        if g.lda < ac || g.ldb < bc || g.ldc < g.n {
            return Err(bad("leading dimension smaller than row length"));
        }
        if extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda) > a.len()
            || extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb) > b.len()
            || extent(g.c.0, g.c.1, g.c.2, g.count, g.m, g.n, g.ldc) > c.len()
        {
            return Err(bad("operand buffer too small for the requested view"));
        }
        let tiles = g.m.div_ceil(gemm::BM) * g.n.div_ceil(gemm::BN);
        // SAFETY: extents validated above; each C element has one owner thread
        // and instances writing the same `c` slot are not allowed (`c` modulus
        // >= count at every call site).
        unsafe {
            self.gemm.sgemm_tiled(
                &self.stream,
                launch::blocks_of(tiles * g.count, gemm::GEMM_THREADS),
                a,
                b,
                c,
                g.m,
                g.n,
                g.k,
                g.lda,
                g.ldb,
                g.ldc,
                g.trans_a as u32,
                g.trans_b as u32,
                g.alpha,
                g.beta,
                g.a.0,
                g.a.1,
                g.a.2,
                g.b.0,
                g.b.1,
                g.b.2,
                g.c.0,
                g.c.1,
                g.c.2,
            )
        }
        .map_err(launch_err)
    }

    /// Single dense GEMM on tensors, for tests and future `connected` layers.
    /// Returns a fresh `m x n` tensor; `c` (read only when `beta != 0`) is
    /// copied via the host, so this is not a hot path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(
        &self,
        a: &OxideTensorF32,
        b: &OxideTensorF32,
        c: &OxideTensorF32,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut out = DeviceBuffer::from_host(&self.stream, &c.buf.to_host_vec(&self.stream)?)?;
        self.gemm(
            &a.buf,
            &b.buf,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: if trans_a { m } else { k },
                ldb: if trans_b { k } else { n },
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta,
                ..GemmArgs::default()
            },
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    pub(crate) fn alloc(&self, len: usize) -> Result<DeviceBuffer<f32>, OxideError> {
        Ok(DeviceBuffer::<f32>::zeroed(&self.stream, len)?)
    }
}

pub(crate) fn launch_err(err: impl fmt::Debug) -> OxideError {
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.conv_forward(x, weights, output_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config)
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.conv_backward_input(output_grad, weights, input_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config)
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.conv_backward_weights(x, output_grad, weight_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config)
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
