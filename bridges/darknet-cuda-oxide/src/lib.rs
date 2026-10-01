//! `F32TensorOpsBackend` implemented with kernels written in Rust and
//! compiled by cuda-oxide (`cargo oxide`).  Build this crate only through
//! `cargo oxide`; plain `cargo build` cannot compile the device code.

pub mod conv;
pub mod half;
pub mod kernels;
pub mod launch;
pub mod mixed;
pub mod tensor;

use std::fmt;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use darknet_cuda::{AdamParams, BatchNormTrainForward, NanInfMode, ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use kernels::bias_scale::BIAS_BACKWARD_THREADS;
use kernels::{
    activations, amp as amp_k, batchnorm, bias_scale, conv_direct, convert, elementwise, gemm, gemm_f16, gemm_mma, half_ops, im2col,
    layers_extra, optim, pool, resample, route,
};
pub use conv::ConvMode;
pub use tensor::{OxideTensorF16, OxideTensorF32};
use tensor::PooledBuf;

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
    batchnorm: batchnorm::kernels::LoadedModule,
    optim: optim::kernels::LoadedModule,
    gemm: gemm::kernels::LoadedModule,
    gemm_mma: gemm_mma::kernels::LoadedModule,
    gemm_f16: gemm_f16::kernels::LoadedModule,
    convert: convert::kernels::LoadedModule,
    half_ops: half_ops::kernels::LoadedModule,
    amp_k: amp_k::kernels::LoadedModule,
    /// Mixed-precision mode: see [`OxideBackend::with_amp`].
    amp: bool,
    /// f16 copies of f32 master weights (keyed by the weight buffer identity).
    amp_w16: Mutex<Vec<mixed::W16Entry>>,
    /// Zero-length f32 buffer shared by every f16-storage tensor.
    empty_f32: Arc<PooledBuf>,
    /// im2col scratch of the f16 conv path and 1-element dummies for the unused
    /// output / bias operands of the f16 GEMM kernel.
    half_scratch: Mutex<half::HalfScratch>,
    half_col: Mutex<Option<DeviceBuffer<u16>>>,
    gemm_mode: GemmMode,
    im2col: im2col::kernels::LoadedModule,
    conv_direct: conv_direct::kernels::LoadedModule,
    route: route::kernels::LoadedModule,
    pool: pool::kernels::LoadedModule,
    resample: resample::kernels::LoadedModule,
    layers_extra: layers_extra::kernels::LoadedModule,
    conv_mode: ConvMode,
    /// im2col / dcol scratch, grown on demand and reused across conv calls.
    workspace: Mutex<Option<DeviceBuffer<f32>>>,
    bufs: Arc<tensor::BufPool>,
}

/// Which GEMM kernel conv / `gemm_f32` use.
///
/// * `Fp32`: exact f32 FMA tiled kernel (default).
/// * `Tf32`: tensor-core `mma.sync` m16n8k8 with TF32 inputs, f32 accumulate
///   (operands rounded to a 10-bit mantissa).
/// * `Fp16`: tensor-core `mma.sync` m16n8k16 with f16 inputs, f32 accumulate.
///   In the f32 API the operands are rounded to f16 on the fly (compute mode,
///   not f16 storage); the real f16-storage inference path is
///   [`OxideBackend::conv2d_bias_act_f16`] and friends, enabled in the CLI by
///   the same `DARKNET_OXIDE_GEMM=fp16`.  Never picked by `Auto`.
/// * `Auto`: `Tf32` when `m * n * k >= AUTO_TF32_MIN_MACS` (2^24, i.e. about
///   256^3) and `min(m, n) >= 32`, else `Fp32`.  Below that the launch is too
///   small to fill the GPU or to amortise the 64x64 tile padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    Fp32,
    Tf32,
    Fp16,
    Auto,
}

/// Minimum `m * n * k` for `GemmMode::Auto` to pick the TF32 tensor-core kernel.
pub const AUTO_TF32_MIN_MACS: usize = 1 << 24;

/// Split-K kicks in below this many output tiles (about 4 blocks per SM on a
/// 24-SM Ada laptop part), with at least `SPLITK_MIN_K` K per split.
pub(crate) const SPLITK_TARGET_BLOCKS: usize = 96;
pub(crate) const SPLITK_MIN_K: usize = 128;
pub(crate) const SPLITK_MAX: usize = 16;

impl GemmMode {
    /// `DARKNET_OXIDE_GEMM={fp32,tf32,fp16,auto}`; unset or unknown means `Fp32`.
    pub fn from_env() -> Self {
        match std::env::var("DARKNET_OXIDE_GEMM")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "tf32" => Self::Tf32,
            "fp16" | "f16" | "half" => Self::Fp16,
            "auto" => Self::Auto,
            _ => Self::Fp32,
        }
    }

    pub(crate) fn use_tf32(self, g: &GemmArgs) -> bool {
        match self {
            Self::Fp32 | Self::Fp16 => false,
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
        F32ActivationKind::Relu6 => activations::ACT_RELU6,
        F32ActivationKind::Ramp => activations::ACT_RAMP,
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
        let batchnorm = batchnorm::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load batchnorm module: {err:?}")))?;
        let optim = optim::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load optim module: {err:?}")))?;
        let gemm = gemm::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm module: {err:?}")))?;
        let gemm_mma = gemm_mma::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_mma module: {err:?}")))?;
        let gemm_f16 = gemm_f16::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_f16 module: {err:?}")))?;
        let convert = convert::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load convert module: {err:?}")))?;
        let half_ops = half_ops::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load half_ops module: {err:?}")))?;
        let amp_k = amp_k::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load amp module: {err:?}")))?;
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
        let layers_extra = layers_extra::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load layers_extra module: {err:?}")))?;
        let half_scratch = half::HalfScratch::new(&stream)?;
        let bufs = tensor::BufPool::new();
        let empty_f32 = Arc::new(PooledBuf::new(DeviceBuffer::<f32>::zeroed(&stream, 0)?, &bufs));
        let amp = mixed::amp_from_env();
        Ok(Self {
            _ctx: ctx,
            stream,
            elementwise,
            activations,
            bias_scale,
            batchnorm,
            optim,
            gemm,
            gemm_mma,
            gemm_f16,
            convert,
            half_ops,
            amp_k,
            amp,
            amp_w16: Mutex::new(Vec::new()),
            empty_f32,
            half_scratch: Mutex::new(half_scratch),
            half_col: Mutex::new(None),
            gemm_mode: if amp { GemmMode::Fp16 } else { GemmMode::from_env() },
            im2col,
            conv_direct,
            route,
            pool,
            resample,
            layers_extra,
            conv_mode: ConvMode::from_env(),
            workspace: Mutex::new(None),
            bufs,
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
        if self.gemm_mode == GemmMode::Fp16 {
            return self.gemm_f32_operands_as_f16(a, b, c, g);
        }
        // Split-K: with fewer than `SPLITK_TARGET_BLOCKS` output tiles the GPU is
        // under-filled while each block walks a long K loop (deep, small-spatial
        // convs).  Split K across extra blocks that write dense partial products
        // to a workspace, then reduce them in a fixed order (deterministic).
        let tf32 = self.gemm_mode.use_tf32(g);
        let (bm, bn) = if tf32 { (gemm_mma::MMA_BM, gemm_mma::MMA_BN) } else { (gemm::BM, gemm::BN) };
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS.div_ceil(tiles).min(g.k / SPLITK_MIN_K).min(SPLITK_MAX).max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 { g.k.div_ceil(splits).next_multiple_of(16) } else { g.k };
        let splits = g.k.div_ceil(kchunk); // drop empty trailing splits
        let mn = g.m * g.n;
        let mut ws = if splits > 1 { Some(self.alloc_uninit(splits * g.count * mn)?) } else { None };
        let (alpha, beta, ldc) = if splits > 1 { (1.0, 0.0, g.n) } else { (g.alpha, g.beta, g.ldc) };
        let (cb, cs, cm) = if splits > 1 { (0, mn, splits * g.count) } else { g.c };
        let target: &mut DeviceBuffer<f32> = match ws.as_mut() {
            Some(w) => w,
            None => c,
        };
        let blocks = tiles * splits;
        // SAFETY: extents validated above (the workspace is sized for every
        // split); each C element has one owner thread and instances writing the
        // same `c` slot are not allowed (`c` modulus >= count at every call site).
        let r = if tf32 {
            unsafe {
                self.gemm_mma.sgemm_tf32_mma(
                    &self.stream,
                    launch::blocks_of(blocks, gemm_mma::MMA_THREADS),
                    a, b, target, g.m, g.n, g.k, g.lda, g.ldb, ldc,
                    g.trans_a as u32, g.trans_b as u32, alpha, beta,
                    g.a.0, g.a.1, g.a.2, g.b.0, g.b.1, g.b.2, cb, cs, cm,
                    g.count, kchunk,
                )
            }
        } else {
            unsafe {
                self.gemm.sgemm_tiled(
                    &self.stream,
                    launch::blocks_of(blocks, gemm::GEMM_THREADS),
                    a, b, target, g.m, g.n, g.k, g.lda, g.ldb, ldc,
                    g.trans_a as u32, g.trans_b as u32, alpha, beta,
                    g.a.0, g.a.1, g.a.2, g.b.0, g.b.1, g.b.2, cb, cs, cm,
                    g.count, kchunk,
                )
            }
        };
        r.map_err(launch_err)?;
        if let Some(w) = ws {
            // SAFETY: one thread per output element; `c` extent validated above.
            unsafe {
                self.gemm.gemm_splitk_reduce(
                    &self.stream,
                    launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                    &w, c, g.m, g.n, g.ldc, g.count, splits, g.alpha, g.beta, g.c.0, g.c.1, g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
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
        let mut out = self.alloc_uninit(c.len())?;
        out.copy_from_host(&self.stream, &c.buf.to_host_vec(&self.stream)?)?;
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

    /// Benchmark helper: mean device milliseconds (CUDA events) per dense NN GEMM
    /// (`alpha = 1`, `beta = 0`) over `iters` launches after one warm-up.
    pub fn bench_gemm_f32(
        &self,
        a: &OxideTensorF32,
        b: &OxideTensorF32,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<f64, OxideError> {
        if self.gemm_mode == GemmMode::Fp16 {
            // Native f16 operands (no per-call conversion), the real fp16 GEMM cost.
            return self.bench_gemm_f16(m, n, k, iters);
        }
        let mut c = self.alloc(m * n)?;
        let g = GemmArgs { m, n, k, lda: k, ldb: n, ldc: n, ..GemmArgs::default() };
        self.time_gpu_ms(iters, || self.gemm(&a.buf, &b.buf, &mut c, &g))
    }

    /// Blocks until all queued work on the backend stream has finished.
    pub fn synchronize(&self) -> Result<(), OxideError> {
        Ok(self.stream.synchronize()?)
    }

    /// Benchmark helper: runs `f` once as warm-up, then `iters` times between two
    /// CUDA events on the backend stream and returns the mean device
    /// milliseconds per call (host launch gaps included, so launch-bound ops
    /// show their launch cost).  `f` must only enqueue work on this backend.
    pub fn time_gpu_ms<F>(&self, iters: usize, mut f: F) -> Result<f64, OxideError>
    where
        F: FnMut() -> Result<(), OxideError>,
    {
        f()?;
        self.stream.synchronize()?;
        let start = self.stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..iters {
            f()?;
        }
        let end = self.stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
    }

    /// Bytes held by the cached im2col / col2im workspace.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|b| b.num_bytes()))
            .unwrap_or(0)
    }

    /// Bytes currently parked in the size-keyed buffer pool (free list).
    pub fn pool_cached_bytes(&self) -> usize {
        self.bufs.cached_bytes()
    }

    /// `(free, total)` device bytes as reported by the driver (whole device).
    pub fn device_mem_info(&self) -> Result<(usize, usize), OxideError> {
        let (mut free, mut total) = (0usize, 0usize);
        self._ctx.bind_to_thread()?;
        // SAFETY: valid out-pointers; the context is current on this thread.
        let r = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        if r != cuda_core::sys::cudaError_enum_CUDA_SUCCESS {
            return Err(OxideError::Shape(format!("cuMemGetInfo failed: {r}")));
        }
        Ok((free, total))
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

    /// Zero-filled output buffer (for kernels that accumulate or only write
    /// part of it).
    pub(crate) fn alloc(&self, len: usize) -> Result<PooledBuf, OxideError> {
        match self.bufs.take(len) {
            Some(mut b) => {
                b.zero_async(&self.stream)?;
                Ok(PooledBuf::new(b, &self.bufs))
            }
            None => Ok(PooledBuf::new(DeviceBuffer::<f32>::zeroed(&self.stream, len)?, &self.bufs)),
        }
    }

    /// Output buffer with unspecified contents: only for kernels that write
    /// every element before anything reads it.
    pub(crate) fn alloc_uninit(&self, len: usize) -> Result<PooledBuf, OxideError> {
        match self.bufs.take(len) {
            Some(b) => Ok(PooledBuf::new(b, &self.bufs)),
            None => self.alloc(len),
        }
    }

    /// Shared reorg launcher for `src_shape -> dst_shape`.  `backward` means
    /// the shapes are (layer output, layer input), i.e. the adjoint direction.
    /// The destination is the small layout exactly when `reverse == backward`.
    fn reorg_launch(
        &self,
        op: &str,
        src: &OxideTensorF32,
        src_shape: &[usize],
        dst_shape: &[usize],
        stride: usize,
        reverse: bool,
        backward: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let to_small = reverse == backward;
        // `reorg_dims` takes (layer input, layer output).
        let (input_shape, output_shape) = if backward {
            (dst_shape, src_shape)
        } else {
            (src_shape, dst_shape)
        };
        let d = darknet_cuda::host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .map_err(|msg| shape_err(op, msg))?;
        let (src_len, dst_len) = if to_small {
            (d.big_len(), d.small_len())
        } else {
            (d.small_len(), d.big_len())
        };
        if src.len() != src_len || src_shape.iter().product::<usize>() != src_len {
            return Err(shape_err(op, format!("input has {} elements, expected {src_len}", src.len())));
        }
        let mut out = self.alloc_uninit(dst_len)?;
        // SAFETY: one thread per output element; source indices are in range
        // by the `reorg_dims` check (`hs * stride <= hb`, `ws * stride <= wb`).
        unsafe {
            if to_small {
                self.layers_extra.reorg_to_small(
                    &self.stream,
                    launch::elementwise(dst_len, config),
                    &src.buf,
                    &mut out,
                    d.cb,
                    d.hb,
                    d.wb,
                    d.hs,
                    d.ws,
                    d.stride,
                )
            } else {
                self.layers_extra.reorg_to_big(
                    &self.stream,
                    launch::elementwise(dst_len, config),
                    &src.buf,
                    &mut out,
                    d.cb,
                    d.hb,
                    d.wb,
                    d.hs,
                    d.ws,
                    d.stride,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, dst_shape))
    }

    /// Validates scale_channels operands; returns `(channels, h * w)`.
    /// Validates an NCHW batch-norm layout; returns `(channels, channel_stride, row_groups)`.
    fn batchnorm_layout(
        op: &str,
        shape: &[usize],
        len: usize,
        per_channel: &[&OxideTensorF32],
    ) -> Result<(usize, usize, usize), OxideError> {
        if shape.len() != 4 || shape.iter().product::<usize>() != len || len == 0 {
            return Err(shape_err(op, format!("shape {shape:?} does not match {len} elements")));
        }
        let channels = shape[1];
        if per_channel.iter().any(|t| t.len() != channels) {
            return Err(shape_err(op, format!("per-channel tensors must have {channels} elements")));
        }
        Ok((channels, shape[2] * shape[3], shape[0]))
    }

    /// Stage 2 of a per-channel reduction: fold the `channels * 256` partials.
    fn fold_partials(&self, partial: &PooledBuf, channels: usize) -> Result<PooledBuf, OxideError> {
        let mut out = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel; `partial` holds channels * 256 sums.
        unsafe {
            self.bias_scale.bias_backward_reduce(
                &self.stream,
                launch::elementwise(channels, ElementwiseLaunchConfig::default()),
                partial,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(out)
    }

    fn scale_channels_layout(
        &self,
        op: &str,
        scale: &OxideTensorF32,
        source: &OxideTensorF32,
        shape: &[usize],
        scale_wh: bool,
    ) -> Result<(usize, usize), OxideError> {
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(op, format!("shape {shape:?} is not a non-empty [batch, c, h, w]")));
        }
        let (batch, channels, hw) = (shape[0], shape[1], shape[2] * shape[3]);
        let scale_len = if scale_wh { batch * hw } else { batch * channels };
        if source.len() != batch * channels * hw || scale.len() != scale_len {
            return Err(shape_err(
                op,
                format!(
                    "source has {} (want {}) / scale has {} (want {scale_len}) elements",
                    source.len(),
                    batch * channels * hw,
                    scale.len()
                ),
            ));
        }
        Ok((channels, hw))
    }

    fn connected_layout(
        &self,
        op: &str,
        x_len: usize,
        w_len: usize,
        batch: usize,
        inputs: usize,
        outputs: usize,
    ) -> Result<(), OxideError> {
        if batch == 0 || inputs == 0 || outputs == 0 {
            return Err(shape_err(op, "batch, inputs and outputs must be > 0".to_owned()));
        }
        if x_len != batch * inputs || w_len != outputs * inputs {
            return Err(shape_err(
                op,
                format!("x has {x_len} (want {}), weights {w_len} (want {})", batch * inputs, outputs * inputs),
            ));
        }
        Ok(())
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
        let mut buf = self.alloc_uninit(host.len())?;
        buf.copy_from_host(&self.stream, host)?;
        Ok(OxideTensorF32::new(buf, shape))
    }

    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        let buf = self.alloc(shape.iter().product())?;
        Ok(OxideTensorF32::new(buf, shape))
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        let tensor = self.f32v(tensor)?;
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
        if x.is_half() {
            return self.bias_add_amp(x, bias, channels, channel_stride, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let bias_f = self.f32v(bias)?;
        let bias = &*bias_f;
        check_channel_layout(
            "bias_add_f32",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        check_channel_layout(
            "channel_scale_f32",
            x.len(),
            Some(scale.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
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
        if output_grad.is_half() {
            return self.bias_backward_amp(output_grad, channels, channel_stride);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        check_channel_layout(
            "bias_backward_f32",
            output_grad.len(),
            None,
            channels,
            channel_stride,
        )?;
        let row_groups = output_grad.len() / channel_stride / channels;
        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        let mut out = self.alloc_uninit(channels)?;
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
        if x.is_half() {
            return self.activation_amp(x, activation, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let mut out = self.alloc_uninit(x.len())?;
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
        if activation_input.is_half() {
            return self.activation_backward_amp(activation_input, delta, activation, config);
        }
        let activation_input_f = self.f32v(activation_input)?;
        let activation_input = &*activation_input_f;
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        same_len("activation_backward_f32", activation_input, delta)?;
        let mut out = self.alloc_uninit(delta.len())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let y_f = self.f32v(y)?;
        let y = &*y_f;
        self.vector_add_f32(x, y, config)
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        if x.is_half() && y.is_half() {
            return self.vector_add_amp(x, y, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let y_f = self.f32v(y)?;
        let y = &*y_f;
        same_len("vector_add_f32", x, y)?;
        let mut out = self.alloc_uninit(x.len())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let y_f = self.f32v(y)?;
        let y = &*y_f;
        same_len("saxpy_f32", x, y)?;
        let mut out = self.alloc_uninit(x.len())?;
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
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        same_len("sgd_update_f32", weights, updates)?;
        let batch = batch.max(1) as f32;
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
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
        if inputs.iter().any(|t| t.is_half()) {
            let halves = inputs.iter().map(|t| self.as_half(t)).collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&OxideTensorF16> = halves.iter().collect();
            let out = self.route_concat_f16(&refs, input_channel_offsets, input_channels, input_total_channels, output_shape, channel_stride, config)?;
            return Ok(self.wrap_half(out));
        }
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
        let mut out = self.alloc_uninit(out_len)?;
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
        if output_grad.is_half() {
            return self.route_backward_amp(output_grad, input_shapes, input_channel_offsets, input_channels, input_total_channels, channel_stride, config);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
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
            // A slice that covers every channel is fully overwritten; otherwise the
            // untouched channels must read zero.
            let mut grad = if input_channels[k] == input_total_channels[k] {
                self.alloc_uninit(len)?
            } else {
                self.alloc(len)?
            };
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
        if self.amp {
            return self.conv_forward_amp(x, weights, output_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
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
        if self.amp {
            return self.conv_backward_input_amp(output_grad, weights, input_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
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
        if self.amp {
            return self.conv_backward_weights_amp(x, output_grad, weight_shape, input_channels, input_height, input_width, filters, kernel_size, stride_x, stride_y, dilation, padding, groups, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
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
        if x.is_half() {
            let xh = self.as_half(x)?;
            let out = self.maxpool_f16(&xh, output_shape, channels, input_height, input_width, size, stride_x, stride_y, padding, config)?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let (_, out_h, out_w) = pool_layout(
            "maxpool_f32", x.len(), output_shape, channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
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
        if x.is_half() {
            return self.maxpool_backward_amp(x, output_grad, channels, input_height, input_width, size, stride_x, stride_y, padding, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let (out_h, out_w) = Self::grad_hw("maxpool_backward_f32", output_grad, x.len(), channels, input_height, input_width)?;
        pool_layout(
            "maxpool_backward_f32", x.len(), output_grad.shape(), channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut dx = self.alloc_uninit(x.len())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let (_, out_h, out_w) = pool_layout(
            "local_avgpool_f32", x.len(), output_shape, channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
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
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "local_avgpool_backward_f32";
        let batch = nchw_batch(op, "input", input_shape, channels, input_height, input_width)?;
        let (out_h, out_w) = Self::grad_hw(op, output_grad, batch * channels * input_height * input_width, channels, input_height, input_width)?;
        pool_layout(
            op, input_shape.iter().product(), output_grad.shape(), channels, input_height, input_width, size, stride_x, stride_y,
        )?;
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "avgpool_f32";
        let batch = nchw_batch(op, "output", output_shape, channels, 1, 1)?;
        let area = input_height * input_width;
        if area == 0 || x.len() != batch * channels * area {
            return Err(shape_err(
                op,
                format!("input has {} elements, expected {batch}*{channels}*{area}", x.len()),
            ));
        }
        let mut out = self.alloc_uninit(batch * channels)?;
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
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "avgpool_backward_f32";
        let batch = nchw_batch(op, "input", input_shape, channels, input_height, input_width)?;
        let area = input_height * input_width;
        if area == 0 || output_grad.len() != batch * channels {
            return Err(shape_err(
                op,
                format!("gradient has {} elements, expected {batch}*{channels}", output_grad.len()),
            ));
        }
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let source_x_f = self.f32v(source_x)?;
        let source_x = &*source_x_f;
        let source_y_f = self.f32v(source_y)?;
        let source_y = &*source_y_f;
        let input_scales_f = self.f32v(input_scales)?;
        let input_scales = &*input_scales_f;
        let input_offsets_f = self.f32v(input_offsets)?;
        let input_offsets = &*input_offsets_f;
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
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
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
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let source_x_f = self.f32v(source_x)?;
        let source_x = &*source_x_f;
        let source_y_f = self.f32v(source_y)?;
        let source_y = &*source_y_f;
        let input_scales_f = self.f32v(input_scales)?;
        let input_scales = &*input_scales_f;
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
        if x.is_half() && !reverse {
            let xh = self.as_half(x)?;
            let out = self.upsample_f16(&xh, output_shape, channels, input_height, input_width, stride, scale, reverse, config)?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
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
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
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
        if output_grad.is_half()
            && input_shape.len() == 4
            && output_height == input_shape[2] * stride
            && output_width == input_shape[3] * stride
        {
            return self.upsample_backward_amp(output_grad, input_shape, channels, output_height, output_width, stride, scale, config);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
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
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
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

    fn shortcut_fit_f32(
        &self,
        x: &Self::Tensor,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "shortcut_fit_f32";
        let out_len: usize = output_shape.iter().product();
        if batch == 0 || x.len() % batch != 0 || out_len % batch != 0 || out_len == 0 || x.is_empty() {
            return Err(shape_err(
                op,
                format!("batch {batch}, {} source elements and output shape {output_shape:?} are inconsistent", x.len()),
            ));
        }
        let (from_len, to_len) = (x.len() / batch, out_len / batch);
        let mut out = self.alloc_uninit(out_len)?;
        // SAFETY: one thread per output element; reads stay inside `x`
        // (`i < from_len`, `b < batch`).
        unsafe {
            self.layers_extra.shortcut_fit(
                &self.stream,
                launch::elementwise(out_len, config),
                &x.buf,
                &mut out,
                from_len,
                to_len,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    fn softmax_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "softmax_f32";
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(op, format!("shape {shape:?} does not match {} elements", x.len())));
        }
        if group_size == 0 || x.is_empty() || x.len() % group_size != 0 {
            return Err(shape_err(op, format!("length {} is not a multiple of group size {group_size}", x.len())));
        }
        let n_groups = x.len() / group_size;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.layers_extra.softmax_forward(
                &self.stream,
                launch::elementwise(n_groups, config),
                &x.buf,
                &mut out,
                group_size,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn softmax_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "softmax_backward_f32";
        same_len(op, output, output_grad)?;
        if shape.iter().product::<usize>() != output.len() {
            return Err(shape_err(op, format!("shape {shape:?} does not match {} elements", output.len())));
        }
        if group_size == 0 || output.is_empty() || output.len() % group_size != 0 {
            return Err(shape_err(op, format!("length {} is not a multiple of group size {group_size}", output.len())));
        }
        let n_groups = output.len() / group_size;
        let mut dx = self.alloc_uninit(output.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.layers_extra.softmax_backward(
                &self.stream,
                launch::elementwise(n_groups, config),
                &output.buf,
                &output_grad.buf,
                &mut dx,
                group_size,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }

    fn reorg_f32(
        &self,
        x: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        self.reorg_launch("reorg_f32", x, input_shape, output_shape, stride, reverse, false, config)
    }

    fn reorg_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        // The adjoint swaps the roles: gradient flows output -> input.
        self.reorg_launch(
            "reorg_backward_f32",
            output_grad,
            output_shape,
            input_shape,
            stride,
            reverse,
            true,
            config,
        )
    }

    fn dropout_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        probability: f32,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "dropout_f32";
        if !(0.0..1.0).contains(&probability) || x.is_empty() {
            return Err(shape_err(op, format!("probability {probability} not in [0, 1) or empty input")));
        }
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(op, format!("shape {shape:?} does not match {} elements", x.len())));
        }
        let mut out = self.alloc_uninit(x.len())?;
        let mut mask = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `out` and `mask` slot.
        unsafe {
            self.layers_extra.dropout_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                &mut mask,
                probability,
                1.0 / (1.0 - probability),
                seed,
            )
        }
        .map_err(launch_err)?;
        Ok((OxideTensorF32::new(out, shape), OxideTensorF32::new(mask, shape)))
    }

    fn dropout_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        mask: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let mask_f = self.f32v(mask)?;
        let mask = &*mask_f;
        same_len("dropout_backward_f32", output_grad, mask)?;
        if shape.iter().product::<usize>() != mask.len() {
            return Err(shape_err("dropout_backward_f32", format!("shape {shape:?} does not match {} elements", mask.len())));
        }
        let mut dx = self.alloc_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.layers_extra.dropout_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mask.buf,
                &mut dx,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }

    fn dropblock_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        block_prob: f32,
        block_size: usize,
        seed: u64,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "dropblock_f32";
        if shape.len() != 4 || shape.iter().product::<usize>() != x.len() || x.is_empty() {
            return Err(shape_err(op, format!("shape {shape:?} is not NCHW of {} elements", x.len())));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if block_size == 0 || block_size > w.min(h) || !(0.0..=1.0).contains(&block_prob) {
            return Err(shape_err(
                op,
                format!("block_size {block_size} must be in 1..={} and prob {block_prob} in [0, 1]", w.min(h)),
            ));
        }
        let planes = n * c;
        let mut keep = self.alloc_uninit(x.len())?;
        let mut dropped = self.alloc_uninit(planes)?;
        let mut scales = self.alloc_uninit(n)?;
        let mut out = self.alloc_uninit(x.len())?;
        let mut mask = self.alloc_uninit(x.len())?;
        // SAFETY: stage 1 writes only its own plane of `keep` and slot of
        // `dropped`; stage 2 one `scales` slot per sample; stage 3 one element.
        unsafe {
            self.layers_extra.dropblock_plane(
                &self.stream,
                launch::elementwise(planes, config),
                &mut keep,
                &mut dropped,
                seed,
                w,
                h,
                block_prob,
                block_size,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: see above.
        unsafe {
            self.layers_extra.dropblock_scales(
                &self.stream,
                launch::elementwise(n, config),
                &dropped,
                &mut scales,
                c,
                c * h * w,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: see above.
        unsafe {
            self.layers_extra.dropblock_apply(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &keep,
                &scales,
                &mut out,
                &mut mask,
                c * h * w,
            )
        }
        .map_err(launch_err)?;
        Ok((OxideTensorF32::new(out, shape), OxideTensorF32::new(mask, shape)))
    }

    fn constrain_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        limit: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err("constrain_f32", format!("shape {shape:?} does not match {} elements", x.len())));
        }
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own slot.
        unsafe {
            self.layers_extra.constrain(&self.stream, launch::elementwise(x.len(), config), &x.buf, &mut out, limit)
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn gradient_centralization_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        f_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "gradient_centralization_f32";
        if f_size == 0 || x.len() % f_size != 0 || shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(op, format!("filter size {f_size} does not divide {} elements", x.len())));
        }
        let filters = x.len() / f_size;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per filter writes the filter's own values.
        unsafe {
            self.layers_extra.gradient_centralize(
                &self.stream,
                launch::elementwise(filters, config),
                &x.buf,
                &mut out,
                f_size,
                filters,
                // C++ only centralises filters whose size is a multiple of the warp.
                (f_size % 32 == 0) as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn scale_channels_f32(
        &self,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        let source_f = self.f32v(source)?;
        let source = &*source_f;
        let (channels, hw) = self.scale_channels_layout("scale_channels_f32", scale, source, shape, scale_wh)?;
        let mut out = self.alloc_uninit(source.len())?;
        // SAFETY: one thread per output element; scale index is in range by
        // the layout check.
        unsafe {
            self.layers_extra.scale_channels_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &scale.buf,
                &source.buf,
                &mut out,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn scale_channels_backward_f32(
        &self,
        delta: &Self::Tensor,
        scale: &Self::Tensor,
        source: &Self::Tensor,
        shape: &[usize],
        scale_wh: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        let source_f = self.f32v(source)?;
        let source = &*source_f;
        let op = "scale_channels_backward_f32";
        let (channels, hw) = self.scale_channels_layout(op, scale, source, shape, scale_wh)?;
        same_len(op, delta, source)?;
        let mut d_src = self.alloc_uninit(source.len())?;
        let mut d_scale = self.alloc_uninit(scale.len())?;
        // SAFETY: one thread per `d_src` element / per scale element.
        unsafe {
            self.layers_extra.scale_channels_dsource(
                &self.stream,
                launch::elementwise(d_src.len(), config),
                &delta.buf,
                &scale.buf,
                &mut d_src,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: as above; each scale element is owned by one thread.
        unsafe {
            self.layers_extra.scale_channels_dscale(
                &self.stream,
                launch::elementwise(d_scale.len(), config),
                &delta.buf,
                &source.buf,
                &mut d_scale,
                channels,
                hw,
                scale_wh as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((OxideTensorF32::new(d_scale, scale.shape()), OxideTensorF32::new(d_src, shape)))
    }

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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let biases_f = self.f32v(biases)?;
        let biases = &*biases_f;
        let op = "connected_f32";
        self.connected_layout(op, x.len(), weights.len(), batch, inputs, outputs)?;
        if biases.len() != outputs {
            return Err(shape_err(op, format!("expected {outputs} biases, got {}", biases.len())));
        }
        let mut raw = self.alloc_uninit(batch * outputs)?;
        // out[b, o] = x[b, :] . w[o, :]  (NT GEMM; beta = 0 writes every element)
        self.gemm(
            &x.buf,
            &weights.buf,
            &mut raw,
            &GemmArgs {
                m: batch,
                n: outputs,
                k: inputs,
                lda: inputs,
                ldb: inputs,
                ldc: outputs,
                trans_b: true,
                ..GemmArgs::default()
            },
        )?;
        let mut out = self.alloc_uninit(batch * outputs)?;
        // SAFETY: one thread per output element; channel_stride 1 = per-column bias.
        unsafe {
            self.bias_scale.bias_add(
                &self.stream,
                launch::elementwise(out.len(), config),
                &raw,
                &biases.buf,
                &mut out,
                outputs,
                1,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[batch, outputs]))
    }

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
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        let op = "connected_backward_f32";
        self.connected_layout(op, x.len(), weights.len(), batch, inputs, outputs)?;
        if delta.len() != batch * outputs {
            return Err(shape_err(op, format!("expected {} delta elements, got {}", batch * outputs, delta.len())));
        }
        // dx = delta . W  (NN GEMM)
        let mut dx = self.alloc_uninit(batch * inputs)?;
        self.gemm(
            &delta.buf,
            &weights.buf,
            &mut dx,
            &GemmArgs {
                m: batch,
                n: inputs,
                k: outputs,
                lda: outputs,
                ldb: inputs,
                ldc: inputs,
                ..GemmArgs::default()
            },
        )?;
        // dW = delta^T . x  (TN GEMM)
        let mut dw = self.alloc_uninit(outputs * inputs)?;
        self.gemm(
            &delta.buf,
            &x.buf,
            &mut dw,
            &GemmArgs {
                m: outputs,
                n: inputs,
                k: batch,
                lda: outputs,
                ldb: inputs,
                ldc: inputs,
                trans_a: true,
                ..GemmArgs::default()
            },
        )?;
        let db = self.bias_backward_f32(delta, outputs, 1, config)?;
        Ok((
            OxideTensorF32::new(dx, &[batch, inputs]),
            OxideTensorF32::new(dw, &[outputs, inputs]),
            db,
        ))
    }

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
        if x.is_half() {
            return self.batchnorm_train_forward_amp(x, scales, biases, rolling_mean, rolling_variance, shape, keep, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let scales_f = self.f32v(scales)?;
        let scales = &*scales_f;
        let biases_f = self.f32v(biases)?;
        let biases = &*biases_f;
        let rolling_mean_f = self.f32v(rolling_mean)?;
        let rolling_mean = &*rolling_mean_f;
        let rolling_variance_f = self.f32v(rolling_variance)?;
        let rolling_variance = &*rolling_variance_f;
        let op = "batchnorm_train_forward_f32";
        let (channels, stride, row_groups) =
            Self::batchnorm_layout(op, shape, x.len(), &[scales, biases, rolling_mean, rolling_variance])?;
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;
        let per_channel = launch::elementwise(channels, config);

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.batchnorm.bn_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &x.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let mut mean = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel.
        unsafe { self.batchnorm.bn_mean_finalize(&self.stream, per_channel, &partial, &mut mean, 1.0 / n) }
            .map_err(launch_err)?;

        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.batchnorm.bn_sq_dev_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &x.buf,
                &mean,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let sum_sq = self.fold_partials(&partial, channels)?;
        let mut variance = self.alloc_uninit(channels)?;
        let mut new_mean = self.alloc_uninit(channels)?;
        let mut new_var = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel writes its own slot of each output.
        unsafe {
            self.batchnorm.bn_var_finalize(
                &self.stream,
                per_channel,
                &sum_sq,
                &mean,
                &rolling_mean.buf,
                &rolling_variance.buf,
                &mut variance,
                &mut new_mean,
                &mut new_var,
                1.0 / n,
                1.0 / (n - 1.0).max(1.0),
                keep,
            )
        }
        .map_err(launch_err)?;

        let mut x_hat = self.alloc_uninit(x.len())?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `x_hat` and `out` slot.
        unsafe {
            self.batchnorm.bn_normalize(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mean,
                &variance,
                &scales.buf,
                &biases.buf,
                &mut x_hat,
                &mut out,
                channels,
                stride,
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok(BatchNormTrainForward {
            output: OxideTensorF32::new(out, shape),
            x_hat: OxideTensorF32::new(x_hat, shape),
            mean: OxideTensorF32::new(mean, &c),
            variance: OxideTensorF32::new(variance, &c),
            rolling_mean: OxideTensorF32::new(new_mean, &c),
            rolling_variance: OxideTensorF32::new(new_var, &c),
        })
    }

    fn batchnorm_train_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        x_hat: &Self::Tensor,
        variance: &Self::Tensor,
        scales: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        if x_hat.is_half() {
            return self.batchnorm_train_backward_amp(output_grad, x_hat, variance, scales, shape, config);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let x_hat_f = self.f32v(x_hat)?;
        let x_hat = &*x_hat_f;
        let variance_f = self.f32v(variance)?;
        let variance = &*variance_f;
        let scales_f = self.f32v(scales)?;
        let scales = &*scales_f;
        let op = "batchnorm_train_backward_f32";
        same_len(op, output_grad, x_hat)?;
        let (channels, stride, row_groups) =
            Self::batchnorm_layout(op, shape, output_grad.len(), &[variance, scales])?;
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.batchnorm.bn_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &output_grad.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let dbias = self.fold_partials(&partial, channels)?;
        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.batchnorm.bn_dot_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &output_grad.buf,
                &x_hat.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let dscale = self.fold_partials(&partial, channels)?;

        let mut dx = self.alloc_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.batchnorm.bn_backward_dx(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &x_hat.buf,
                &variance.buf,
                &scales.buf,
                &dbias,
                &dscale,
                &mut dx,
                channels,
                stride,
                n,
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok((
            OxideTensorF32::new(dx, shape),
            OxideTensorF32::new(dscale, &c),
            OxideTensorF32::new(dbias, &c),
        ))
    }

    fn sanitize_nan_inf_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        mode: NanInfMode,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        if shape.iter().product::<usize>() != x.len() {
            return Err(OxideError::Shape(format!(
                "sanitize_nan_inf_f32: shape {shape:?} does not match {} elements",
                x.len()
            )));
        }
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own slot of the output.
        unsafe {
            self.optim.sanitize_nan_inf(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                (mode == NanInfMode::Fix) as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn adam_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        m: &Self::Tensor,
        v: &Self::Tensor,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let m_f = self.f32v(m)?;
        let m = &*m_f;
        let v_f = self.f32v(v)?;
        let v = &*v_f;
        let op = "adam_update_f32";
        same_len(op, weights, updates)?;
        same_len(op, weights, m)?;
        same_len(op, weights, v)?;
        let (coupled, decoupled, bc1, bc2) = params.kernel_scalars();
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_m = self.alloc_uninit(weights.len())?;
        let mut new_v = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element writes its own slot of each output.
        unsafe {
            self.optim.adam_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &m.buf,
                &v.buf,
                &mut new_weights,
                &mut new_m,
                &mut new_v,
                params.lr,
                params.beta1,
                params.beta2,
                params.eps,
                coupled,
                decoupled,
                bc1,
                bc2,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, weights.shape()),
            OxideTensorF32::new(new_m, weights.shape()),
            OxideTensorF32::new(new_v, weights.shape()),
        ))
    }

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
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let op = "sanitized_sgd_update_f32";
        same_len(op, weights, updates)?;
        if shape.iter().product::<usize>() != weights.len() {
            return Err(OxideError::Shape(format!(
                "{op}: shape {shape:?} does not match {} elements",
                weights.len()
            )));
        }
        let batch = batch.max(1) as f32;
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element; every output slot is written.
        unsafe {
            self.optim.sanitized_sgd_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &mut new_weights,
                &mut new_updates,
                lr / batch,
                momentum,
                decay * batch,
                fix_weights as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, shape),
            OxideTensorF32::new(new_updates, shape),
        ))
    }

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
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let m_f = self.f32v(m)?;
        let m = &*m_f;
        let v_f = self.f32v(v)?;
        let v = &*v_f;
        let op = "sanitized_adam_update_f32";
        same_len(op, weights, updates)?;
        same_len(op, weights, m)?;
        same_len(op, weights, v)?;
        if shape.iter().product::<usize>() != weights.len() {
            return Err(OxideError::Shape(format!(
                "{op}: shape {shape:?} does not match {} elements",
                weights.len()
            )));
        }
        let (coupled, decoupled, bc1, bc2) = params.kernel_scalars();
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_m = self.alloc_uninit(weights.len())?;
        let mut new_v = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element; every output slot is written.
        unsafe {
            self.optim.sanitized_adam_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &m.buf,
                &v.buf,
                &mut new_weights,
                &mut new_m,
                &mut new_v,
                &mut new_updates,
                params.lr,
                params.beta1,
                params.beta2,
                params.eps,
                coupled,
                decoupled,
                bc1,
                bc2,
                fix_weights as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, shape),
            OxideTensorF32::new(new_m, shape),
            OxideTensorF32::new(new_v, shape),
            OxideTensorF32::new(new_updates, shape),
        ))
    }

    fn batchnorm_fold_f32(
        &self,
        gamma: &Self::Tensor,
        mean: &Self::Tensor,
        variance: &Self::Tensor,
        bias: &Self::Tensor,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let gamma_f = self.f32v(gamma)?;
        let gamma = &*gamma_f;
        let mean_f = self.f32v(mean)?;
        let mean = &*mean_f;
        let variance_f = self.f32v(variance)?;
        let variance = &*variance_f;
        let bias_f = self.f32v(bias)?;
        let bias = &*bias_f;
        let op = "batchnorm_fold_f32";
        same_len(op, gamma, mean)?;
        same_len(op, gamma, variance)?;
        same_len(op, gamma, bias)?;
        let mut scale = self.alloc_uninit(gamma.len())?;
        let mut folded = self.alloc_uninit(gamma.len())?;
        // SAFETY: one thread per channel; every output slot is written.
        unsafe {
            self.optim.bn_fold(
                &self.stream,
                launch::elementwise(gamma.len(), config),
                &gamma.buf,
                &mean.buf,
                &variance.buf,
                &bias.buf,
                &mut scale,
                &mut folded,
                eps,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(scale, gamma.shape()),
            OxideTensorF32::new(folded, gamma.shape()),
        ))
    }
}
