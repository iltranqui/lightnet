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
use kernels::{
    activations, bias_scale, conv_direct, elementwise, gemm, gemm_mma, im2col, optim, pool, resample,
    route,
};
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
    gemm_mma: gemm_mma::kernels::LoadedModule,
    gemm_mode: GemmMode,
    im2col: im2col::kernels::LoadedModule,
    conv_direct: conv_direct::kernels::LoadedModule,
    route: route::kernels::LoadedModule,
    pool: pool::kernels::LoadedModule,
    resample: resample::kernels::LoadedModule,
    conv_mode: ConvMode,
    /// im2col / dcol scratch, grown on demand and reused across conv calls.
    workspace: Mutex<Option<DeviceBuffer<f32>>>,
}

/// Which GEMM kernel conv / `gemm_f32` use.
///
/// * `Fp32`: exact f32 FMA tiled kernel (default).
/// * `Tf32`: tensor-core `mma.sync` m16n8k8 with TF32 inputs, f32 accumulate
///   (operands rounded to a 10-bit mantissa).
/// * `Auto`: `Tf32` when `m * n * k >= AUTO_TF32_MIN_MACS` (2^24, i.e. about
///   256^3) and `min(m, n) >= 32`, else `Fp32`.  Below that the launch is too
///   small to fill the GPU or to amortise the 64x64 tile padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    Fp32,
    Tf32,
    Auto,
}

/// Minimum `m * n * k` for `GemmMode::Auto` to pick the TF32 tensor-core kernel.
pub const AUTO_TF32_MIN_MACS: usize = 1 << 24;

impl GemmMode {
    /// `DARKNET_OXIDE_GEMM={fp32,tf32,auto}`; unset or unknown means `Fp32`.
    pub fn from_env() -> Self {
        match std::env::var("DARKNET_OXIDE_GEMM")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "tf32" => Self::Tf32,
            "auto" => Self::Auto,
            _ => Self::Fp32,
        }
    }

    fn use_tf32(self, g: &GemmArgs) -> bool {
        match self {
            Self::Fp32 => false,
            Self::Tf32 => true,
            Self::Auto => {
                g.m.saturating_mul(g.n).saturating_mul(g.k) >= AUTO_TF32_MIN_MACS
                    && g.m.min(g.n) >= 32
            }
        }
    }
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


fn shape_err(op: &str, msg: String) -> OxideError {
    OxideError::Shape(format!("{op}: {msg}"))
}

/// Checks an NCHW `[batch, channels, h, w]` shape and returns its batch.
fn nchw_batch(
    op: &str,
    what: &str,
    shape: &[usize],
    channels: usize,
    h: usize,
    w: usize,
) -> Result<usize, OxideError> {
    if shape.len() != 4 || shape[1] != channels || shape[2] != h || shape[3] != w {
        return Err(shape_err(
            op,
            format!("{what} shape {shape:?} is not [batch, {channels}, {h}, {w}]"),
        ));
    }
    Ok(shape[0])
}

/// Validated pooling geometry: `(batch, out_h, out_w)`.
fn pool_layout(
    op: &str,
    x_shape_len: usize,
    out_shape: &[usize],
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
) -> Result<(usize, usize, usize), OxideError> {
    if channels == 0 || in_h == 0 || in_w == 0 || size == 0 || stride_x == 0 || stride_y == 0 {
        return Err(shape_err(
            op,
            "channels, spatial dims, size and strides must be > 0".to_owned(),
        ));
    }
    if out_shape.len() != 4 || out_shape[1] != channels || out_shape[2] == 0 || out_shape[3] == 0 {
        return Err(shape_err(
            op,
            format!("output shape {out_shape:?} is not [batch, {channels}, out_h, out_w]"),
        ));
    }
    let batch = out_shape[0];
    if x_shape_len != batch * channels * in_h * in_w {
        return Err(shape_err(
            op,
            format!(
                "input has {x_shape_len} elements, expected {batch}*{channels}*{in_h}*{in_w}"
            ),
        ));
    }
    Ok((batch, out_shape[2], out_shape[3]))
}

/// Validated crop geometry: `(batch, out_h, out_w)`.
fn crop_layout(
    op: &str,
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    coords: (usize, usize),
    adjustments: &[usize],
) -> Result<(), OxideError> {
    if channels == 0 || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(shape_err(op, "channels and spatial dims must be > 0".to_owned()));
    }
    let expected = batch * out_h * out_w;
    if coords.0 != expected || coords.1 != expected {
        return Err(shape_err(
            op,
            format!(
                "expected {expected} source coordinates, got x={} y={}",
                coords.0, coords.1
            ),
        ));
    }
    if adjustments.iter().any(|&len| len != batch) {
        return Err(shape_err(
            op,
            format!("per-batch adjustments must have length {batch}, got {adjustments:?}"),
        ));
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
        let gemm_mma = gemm_mma::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_mma module: {err:?}")))?;
        let im2col = im2col::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load im2col module: {err:?}")))?;
        let conv_direct = conv_direct::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load conv_direct module: {err:?}")))?;
        let route = route::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load route module: {err:?}")))?;
        let pool = pool::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load pool module: {err:?}")))?;
        let resample = resample::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load resample module: {err:?}")))?;
        Ok(Self {
            _ctx: ctx,
            stream,
            elementwise,
            activations,
            bias_scale,
            optim,
            gemm,
            gemm_mma,
            gemm_mode: GemmMode::from_env(),
            im2col,
            conv_direct,
            route,
            pool,
            resample,
            conv_mode: ConvMode::from_env(),
            workspace: Mutex::new(None),
        })
    }

    /// Overrides the conv implementation chosen from `DARKNET_OXIDE_CONV`.
    pub fn with_conv_mode(mut self, mode: ConvMode) -> Self {
        self.conv_mode = mode;
        self
    }

    /// Overrides the GEMM kernel chosen from `DARKNET_OXIDE_GEMM`.
    pub fn with_gemm_mode(mut self, mode: GemmMode) -> Self {
        self.gemm_mode = mode;
        self
    }

    pub fn gemm_mode(&self) -> GemmMode {
        self.gemm_mode
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
        if self.gemm_mode.use_tf32(g) {
            let tiles = g.m.div_ceil(gemm_mma::MMA_BM) * g.n.div_ceil(gemm_mma::MMA_BN);
            // SAFETY: as for the fp32 kernel below (same ownership/extents).
            return unsafe {
                self.gemm_mma.sgemm_tf32_mma(
                    &self.stream,
                    launch::blocks_of(tiles * g.count, gemm_mma::MMA_THREADS),
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
            .map_err(launch_err);
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

    /// Benchmark helper: mean milliseconds per dense NN GEMM (`alpha = 1`,
    /// `beta = 0`) over `iters` launches after one warm-up, stream-synchronised.
    pub fn bench_gemm_f32(
        &self,
        a: &OxideTensorF32,
        b: &OxideTensorF32,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<f64, OxideError> {
        let mut c = self.alloc(m * n)?;
        let g = GemmArgs { m, n, k, lda: k, ldb: n, ldc: n, ..GemmArgs::default() };
        self.gemm(&a.buf, &b.buf, &mut c, &g)?;
        self.stream.synchronize()?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            self.gemm(&a.buf, &b.buf, &mut c, &g)?;
        }
        self.stream.synchronize()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / iters as f64)
    }

    /// `(out_h, out_w)` of a pooling gradient tensor, checking that it is
    /// `[batch, channels, h, w]` for the `x_len`-element input.
    fn grad_hw(
        op: &str,
        grad: &OxideTensorF32,
        x_len: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
    ) -> Result<(usize, usize), OxideError> {
        let g = grad.shape();
        if g.len() != 4 || g[1] != channels || in_h * in_w * channels == 0 || x_len % (in_h * in_w * channels) != 0 || g[0] != x_len / (in_h * in_w * channels) {
            return Err(shape_err(op, format!("gradient shape {g:?} does not match the input")));
        }
        Ok((g[2], g[3]))
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

    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "route_concat_f32";
        let n = inputs.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
        {
            return Err(shape_err(op, "needs >= 1 input and equal-length metadata".to_owned()));
        }
        if channel_stride == 0 {
            return Err(shape_err(op, "channel_stride must be > 0".to_owned()));
        }
        let out_total: usize = input_channels.iter().sum();
        let out_len: usize = output_shape.iter().product();
        let mut batch = None;
        for k in 0..n {
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
            {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k}: slice [{}..{}) outside {} channels",
                        input_channel_offsets[k],
                        input_channel_offsets[k] + input_channels[k],
                        input_total_channels[k]
                    ),
                ));
            }
            let per_batch = input_total_channels[k] * channel_stride;
            if inputs[k].len() == 0 || inputs[k].len() % per_batch != 0 {
                return Err(shape_err(
                    op,
                    format!("input {k} length {} not a multiple of {per_batch}", inputs[k].len()),
                ));
            }
            let b = inputs[k].len() / per_batch;
            if *batch.get_or_insert(b) != b {
                return Err(shape_err(op, format!("input {k} batch {b} differs from {batch:?}")));
            }
        }
        let batch = batch.unwrap_or(0);
        if out_len != batch * out_total * channel_stride {
            return Err(shape_err(
                op,
                format!(
                    "output shape {output_shape:?} needs {} elements, inputs give {}",
                    out_len,
                    batch * out_total * channel_stride
                ),
            ));
        }
        let mut out = self.alloc(out_len)?;
        let mut dst_offset = 0usize;
        for k in 0..n {
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: one thread per copied element, all with distinct
            // destinations inside `out` (validated above).
            unsafe {
                self.route.route_copy(
                    &self.stream,
                    launch::elementwise(count, config),
                    &inputs[k].buf,
                    &mut out,
                    batch,
                    input_channels[k],
                    input_total_channels[k],
                    input_channel_offsets[k],
                    out_total,
                    dst_offset,
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            dst_offset += input_channels[k];
        }
        Ok(OxideTensorF32::new(out, output_shape))
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
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        let op = "route_backward_f32";
        let n = input_shapes.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
        {
            return Err(shape_err(op, "needs >= 1 input and equal-length metadata".to_owned()));
        }
        if channel_stride == 0 {
            return Err(shape_err(op, "channel_stride must be > 0".to_owned()));
        }
        let out_total: usize = input_channels.iter().sum();
        let per_batch = out_total * channel_stride;
        if per_batch == 0 || output_grad.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!("gradient length {} not a multiple of {per_batch}", output_grad.len()),
            ));
        }
        let batch = output_grad.len() / per_batch;
        let mut grads = Vec::with_capacity(n);
        let mut src_offset = 0usize;
        for k in 0..n {
            let len: usize = input_shapes[k].iter().product();
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
                || len != batch * input_total_channels[k] * channel_stride
            {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k}: shape {:?} / slice [{}..{}) inconsistent with {} channels, batch {batch}",
                        input_shapes[k],
                        input_channel_offsets[k],
                        input_channel_offsets[k] + input_channels[k],
                        input_total_channels[k]
                    ),
                ));
            }
            let mut grad = self.alloc(len)?;
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: distinct destination per thread inside `grad`, which
            // covers `batch * input_total_channels[k] * stride` (validated).
            unsafe {
                self.route.route_copy(
                    &self.stream,
                    launch::elementwise(count, config),
                    &output_grad.buf,
                    &mut grad,
                    batch,
                    input_channels[k],
                    out_total,
                    src_offset,
                    input_total_channels[k],
                    input_channel_offsets[k],
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            grads.push(OxideTensorF32::new(grad, &input_shapes[k]));
            src_offset += input_channels[k];
        }
        Ok(grads)
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
    ) -> Result<Self::Tensor, Self::Error> {
        let (_, out_h, out_w) = pool_layout(
            "maxpool_f32", x.len(), output_shape, channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut out = self.alloc(output_shape.iter().product())?;
        // SAFETY: one thread per output element; all input reads are
        // bounds-checked against the image and `x` covers the validated extent.
        unsafe {
            self.pool.maxpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
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
    ) -> Result<Self::Tensor, Self::Error> {
        let (out_h, out_w) = Self::grad_hw("maxpool_backward_f32", output_grad, x.len(), channels, input_height, input_width)?;
        pool_layout(
            "maxpool_backward_f32", x.len(), output_grad.shape(), channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut dx = self.alloc(x.len())?;
        // SAFETY: one thread per input element (gather, no atomics); reads are
        // bounds-checked and `x` / `output_grad` cover the validated extents.
        unsafe {
            self.pool.maxpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &x.buf,
                &output_grad.buf,
                &mut dx,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, x.shape()))
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
    ) -> Result<Self::Tensor, Self::Error> {
        let (_, out_h, out_w) = pool_layout(
            "local_avgpool_f32", x.len(), output_shape, channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut out = self.alloc(output_shape.iter().product())?;
        // SAFETY: one thread per output element; reads are bounds-checked.
        unsafe {
            self.pool.local_avgpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
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
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "local_avgpool_backward_f32";
        let batch = nchw_batch(op, "input", input_shape, channels, input_height, input_width)?;
        let (out_h, out_w) = Self::grad_hw(op, output_grad, batch * channels * input_height * input_width, channels, input_height, input_width)?;
        pool_layout(
            op, input_shape.iter().product(), output_grad.shape(), channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut dx = self.alloc(input_shape.iter().product())?;
        // SAFETY: one thread per input element (gather); reads bounds-checked.
        unsafe {
            self.pool.local_avgpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "avgpool_f32";
        let batch = nchw_batch(op, "output", output_shape, channels, 1, 1)?;
        let area = input_height * input_width;
        if area == 0 || x.len() != batch * channels * area {
            return Err(shape_err(
                op,
                format!("input has {} elements, expected {batch}*{channels}*{area}", x.len()),
            ));
        }
        let mut out = self.alloc(batch * channels)?;
        // SAFETY: one thread per (batch, channel) plane, reading `area` elements.
        unsafe {
            self.pool.avgpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                area,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "avgpool_backward_f32";
        let batch = nchw_batch(op, "input", input_shape, channels, input_height, input_width)?;
        let area = input_height * input_width;
        if area == 0 || output_grad.len() != batch * channels {
            return Err(shape_err(
                op,
                format!("gradient has {} elements, expected {batch}*{channels}", output_grad.len()),
            ));
        }
        let mut dx = self.alloc(input_shape.iter().product())?;
        // SAFETY: one thread per input element reading `dy[i / area]`.
        unsafe {
            self.pool.avgpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                area,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

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
        let op = "crop_bilinear_f32";
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(op, format!("output shape {output_shape:?} is not [batch, {channels}, h, w]")));
        }
        let batch = output_shape[0];
        crop_layout(
            op,
            batch,
            channels,
            input_height,
            input_width,
            output_shape[2],
            output_shape[3],
            (source_x.len(), source_y.len()),
            &[input_scales.len(), input_offsets.len()],
        )?;
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(op, format!("input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}", x.len())));
        }
        let mut out = self.alloc(output_shape.iter().product())?;
        // SAFETY: one thread per output element; taps are bounds-checked and
        // coordinate / adjustment lengths were validated above.
        unsafe {
            self.resample.crop_bilinear(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &source_x.buf,
                &source_y.buf,
                &input_scales.buf,
                &input_offsets.buf,
                &mut out,
                channels,
                input_height,
                input_width,
                output_shape[2],
                output_shape[3],
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

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
        let op = "crop_bilinear_backward_f32";
        let g = output_grad.shape();
        if g.len() != 4 || g[1] != channels {
            return Err(shape_err(op, format!("gradient shape {g:?} is not [batch, {channels}, h, w]")));
        }
        let batch = g[0];
        crop_layout(
            op,
            batch,
            channels,
            input_height,
            input_width,
            g[2],
            g[3],
            (source_x.len(), source_y.len()),
            &[input_scales.len()],
        )?;
        nchw_batch(op, "input", input_shape, channels, input_height, input_width)
            .and_then(|b| if b == batch { Ok(()) } else { Err(shape_err(op, format!("input batch {b} != gradient batch {batch}"))) })?;
        let dx = self.alloc(input_shape.iter().product())?;
        // SAFETY: `dx` is a live zeroed buffer of the validated extent; the
        // kernel only touches it through in-image atomic adds.
        unsafe {
            self.resample.crop_bilinear_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &output_grad.buf,
                &source_x.buf,
                &source_y.buf,
                &input_scales.buf,
                dx.cu_deviceptr() as *mut f32,
                channels,
                input_height,
                input_width,
                g[2],
                g[3],
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
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
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "upsample_f32";
        if stride == 0 || channels == 0 || input_height == 0 || input_width == 0 {
            return Err(shape_err(op, "channels, spatial dims and stride must be > 0".to_owned()));
        }
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(op, format!("output shape {output_shape:?} is not [batch, {channels}, h, w]")));
        }
        let (batch, out_h, out_w) = (output_shape[0], output_shape[2], output_shape[3]);
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(op, format!("input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}", x.len())));
        }
        if reverse {
            // floor (C++) and ceil (cuTile) sized outputs are both accepted.
            if out_h == 0 || out_w == 0 || out_h > input_height.div_ceil(stride) || out_w > input_width.div_ceil(stride) || out_h < input_height / stride || out_w < input_width / stride {
                return Err(shape_err(op, format!("reverse output {out_h}x{out_w} incompatible with input {input_height}x{input_width} / stride {stride}")));
            }
        } else if out_h != input_height * stride || out_w != input_width * stride {
            return Err(shape_err(op, format!("output {out_h}x{out_w} != input {input_height}x{input_width} * stride {stride}")));
        }
        let mut out = self.alloc(output_shape.iter().product())?;
        // SAFETY: one thread per output element; input reads are in range by
        // the shape checks above (reverse additionally bounds-checks taps).
        unsafe {
            if reverse {
                self.resample.upsample_reverse(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                )
            } else {
                self.resample.upsample_nearest(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    /// The trait carries no `reverse` flag: the layer direction is inferred from
    /// the shapes.  `output_height == input_height * stride` is a normal
    /// (expanding) layer; anything smaller is a `reverse` layer (identical
    /// results when `stride == 1`).
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
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "upsample_backward_f32";
        if stride == 0 || input_shape.len() != 4 || input_shape[1] != channels {
            return Err(shape_err(op, format!("bad stride {stride} or input shape {input_shape:?}")));
        }
        let (batch, in_h, in_w) = (input_shape[0], input_shape[2], input_shape[3]);
        if in_h == 0 || in_w == 0 || output_height == 0 || output_width == 0 {
            return Err(shape_err(op, "spatial dims must be > 0".to_owned()));
        }
        if output_grad.len() != batch * channels * output_height * output_width {
            return Err(shape_err(op, format!("gradient has {} elements, expected {batch}*{channels}*{output_height}*{output_width}", output_grad.len())));
        }
        let reverse = if output_height == in_h * stride && output_width == in_w * stride {
            false
        } else if output_height >= in_h / stride
            && output_height <= in_h.div_ceil(stride)
            && output_width >= in_w / stride
            && output_width <= in_w.div_ceil(stride)
        {
            true
        } else {
            return Err(shape_err(op, format!("output {output_height}x{output_width} fits neither expand nor reverse of input {in_h}x{in_w}, stride {stride}")));
        };
        let mut dx = self.alloc(input_shape.iter().product())?;
        // SAFETY: one thread per input element (gather); dy reads are bounds-checked.
        unsafe {
            self.resample.upsample_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                in_h,
                in_w,
                output_height,
                output_width,
                stride,
                scale,
                reverse as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }
}
