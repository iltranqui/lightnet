//! `F32TensorOpsBackend` implemented with kernels written in Rust and
//! compiled by cuda-oxide (`cargo oxide`).  Build this crate only through
//! `cargo oxide`; plain `cargo build` cannot compile the device code.

pub mod conv;
mod clifford;
mod dcnv4_native;
mod deconvolutional;
mod graph_conv;
mod swin;
pub use dcnv4_native::Dcnv4NativeGradients;
pub mod fp4;
pub mod fp8;
pub mod half;
pub mod kernels;
pub mod launch;
pub mod mixed;
pub mod tensor;

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use lightnet_cuda::{
    AdamParams, AdamUpdateItem, BatchNormTrainForward, BilinearGatherGeometry,
    ElementwiseLaunchConfig, F32ActivationKind, F32ChannelNormalization,
    F32ShortcutWeightNormalization, F32TensorOpsBackend, FoveationFilmGradients, NanInfMode,
    SgdUpdateItem,
};

pub use conv::ConvMode;
pub use fp4::Fp4Packed;
pub use fp8::Fp8Format;
pub use mixed::{HalfFormat, parse_amp_format};
use kernels::bias_scale::BIAS_BACKWARD_THREADS;
use kernels::{
    activations, amp as amp_k, batchnorm, bias_scale, conv_direct, conv_implicit, convert, dcnv4,
    deform_sample, elementwise, foveation_film, gemm, gemm_f8, gemm_f16, gemm_fp4, gemm_mma,
    half_ops, im2col, layers_extra, muon, optim, pool, quant_fp4, resample, route, wmhf, vit,
    attention, moe, laya, dense_loss, softmax, reorg, channel_shuffle, dropout, scale_channels,
    shortcut, recurrent, eml, normalize_channels,
};
use tensor::{PooledBuf, PooledBuf8};
pub use tensor::{OxideTensorF16, OxideTensorF32};

/// Words per amax slot: the pair kernel uses words `0` and `32`, so a slot is
/// 64 words (256 bytes) and neighbouring slots never share a cache line.
pub(crate) const FP8_SLOT_WORDS: usize = 64;
/// Slots per ring; one memset zeroes all of them once every `FP8_RING_SLOTS`
/// quantised operands instead of one memset per operand.
pub(crate) const FP8_RING_SLOTS: usize = 256;

/// A ring of zeroed amax slots.  The amax kernels `fetch_max` into a slot that
/// must start at 0; a slot is handed out once per lap and the whole ring is
/// re-zeroed (stream-ordered, so after every kernel that read the previous lap)
/// when it wraps.
pub(crate) struct Fp8Ring {
    pub(crate) buf: DeviceBuffer<f32>,
    next: usize,
}

/// How the FP8 quantiser picks a tensor's scale.
///
/// `Current` (default) measures the tensor's own amax in a first pass and
/// quantises in a second.  `Delayed` takes the factor from the maximum of the
/// last calls' amaxes at the same call site (an amax history) and records this
/// call's amax in the same single pass; the first call of a site has no history
/// and measures its own.  Chosen by `LIGHTNET_OXIDE_FP8_SCALING=current|delayed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Fp8Scaling {
    #[default]
    Current,
    Delayed,
}

impl Fp8Scaling {
    /// Parse the env spelling (case-insensitive); anything but `delayed` is `Current`.
    pub fn from_name(name: &str) -> Self {
        if name.trim().eq_ignore_ascii_case("delayed") {
            Self::Delayed
        } else {
            Self::Current
        }
    }

    fn from_env() -> Self {
        Self::from_name(&std::env::var("LIGHTNET_OXIDE_FP8_SCALING").unwrap_or_default())
    }
}

/// Device amax histories of the delayed FP8 scaling: `FP8_HIST_KEYS` keys of
/// `F8_HIST_LEN` slots each in one zeroed buffer.  A key is (source buffer
/// device pointer, pool incarnation, element count, format code).  The
/// incarnation changes when the pool hands the address to a new tensor, so a
/// recycled buffer starts a fresh history instead of inheriting the old
/// owner's amax.  When the buffer is full, new keys fall back to current
/// scaling.
pub(crate) struct Fp8Hist {
    buf: DeviceBuffer<f32>,
    /// key -> (first slot word, calls so far)
    keys: std::collections::HashMap<(u64, u64, usize, u32), (usize, u64)>,
}

pub(crate) const FP8_HIST_KEYS: usize = 4096;

/// Stable delayed-history keys, one per logical operand role.  Unlike a buffer
/// address these survive pool reuse, so the amax history keeps accumulating
/// across training steps instead of restarting whenever a tensor is reallocated.
const FP8_SITE_FORWARD_A: u64 = 0x4650_0001;
const FP8_SITE_FORWARD_B: u64 = 0x4650_0002;
const FP8_SITE_WGRAD_A: u64 = 0x4650_0005;
const FP8_SITE_WGRAD_B: u64 = 0x4650_0006;
const FP8_SITE_WEIGHTS_B: u64 = 0x4650_0007;
const FP8_SITE_DY: u64 = 0x4650_0008;
const FP8_SITE_HALF_A: u64 = 0x4650_0009;
const FP8_SITE_HALF_B: u64 = 0x4650_000a;

/// Source of a delayed-scaling quantise: f32 storage, or 16-bit storage with
/// the format that widens it.
#[derive(Clone, Copy)]
enum Fp8Src<'a> {
    F32(&'a DeviceBuffer<f32>),
    Half(&'a DeviceBuffer<u16>, mixed::HalfFormat),
}

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
    layernorm: activations::layernorm::LoadedModule,
    bias_scale: bias_scale::kernels::LoadedModule,
    batchnorm: batchnorm::kernels::LoadedModule,
    muon: muon::kernels::LoadedModule,
    optim: optim::kernels::LoadedModule,
    gemm: gemm::kernels::LoadedModule,
    gemm_mma: gemm_mma::kernels::LoadedModule,
    gemm_f16: gemm_f16::kernels::LoadedModule,
    gemm_f8: gemm_f8::kernels::LoadedModule,
    quant_fp4: quant_fp4::kernels::LoadedModule,
    gemm_fp4: gemm_fp4::kernels::LoadedModule,
    convert: convert::kernels::LoadedModule,
    half_ops: half_ops::kernels::LoadedModule,
    amp_k: amp_k::kernels::LoadedModule,
    /// Mixed-precision mode: see [`OxideBackend::with_amp`].
    amp: bool,
    /// 16-bit format of AMP storage and of the 16-bit GEMM modes.
    half_fmt: mixed::HalfFormat,
    /// f16 copies of f32 master weights (keyed by the weight buffer identity).
    amp_w16: Mutex<Vec<mixed::W16Entry>>,
    /// Device-to-host readbacks issued by the GEMM paths; see
    /// [`OxideBackend::host_readbacks`].
    host_readbacks: AtomicU64,
    /// Quantised copies of f32 weight tensors for the FP8 GEMM (keyed by the
    /// weight buffer identity and format); see [`OxideBackend::fp8_weights`].
    fp8_w: Mutex<Vec<mixed::Fp8WEntry>>,
    /// Quantised output gradient shared by the dgrad and wgrad of one conv
    /// backward; see [`OxideBackend::fp8_dy`].
    fp8_dy: Mutex<Option<mixed::Fp8DyEntry>>,
    /// The 16-bit-storage twin of `fp8_dy`; see [`OxideBackend::fp8_dy_half`].
    fp8_dy_half: Mutex<Option<mixed::Fp8DyHalfEntry>>,
    /// Pre-zeroed amax slots (see [`Fp8Ring`]).
    fp8_ring: Mutex<Fp8Ring>,
    /// Scale selection of the activation quantiser; see [`Fp8Scaling`].
    fp8_scaling: Fp8Scaling,
    /// Amax histories of [`Fp8Scaling::Delayed`], allocated on first use.
    fp8_hist: Mutex<Option<Fp8Hist>>,
    /// Quantise launches issued by the FP8 path; see
    /// [`OxideBackend::fp8_quantize_launches`].
    fp8_quantize_launches: AtomicU64,
    /// Stand-alone amax passes of the 16-bit FP8 quantiser, see
    /// [`OxideBackend::fp8_amax_passes`].
    fp8_amax_passes: AtomicU64,
    /// Test seam: route e4m3 x e4m3 GEMMs through the generic-format kernel;
    /// see [`OxideBackend::set_fp8_generic_kernel`].
    fp8_generic_kernel: AtomicBool,
    /// Zero-length f32 buffer shared by every f16-storage tensor.
    empty_f32: Arc<PooledBuf>,
    /// im2col scratch of the f16 conv path and 1-element dummies for the unused
    /// output / bias operands of the f16 GEMM kernel.
    half_scratch: Mutex<half::HalfScratch>,
    half_col: Mutex<Option<DeviceBuffer<u16>>>,
    gemm_mode: GemmMode,
    im2col: im2col::kernels::LoadedModule,
    conv_direct: conv_direct::kernels::LoadedModule,
    local: kernels::local::kernels::LoadedModule,
    graph_conv: kernels::graph_conv::kernels::LoadedModule,
    deconvolutional: kernels::deconvolutional::kernels::LoadedModule,
    conv_implicit: conv_implicit::kernels::LoadedModule,
    route: route::kernels::LoadedModule,
    pool: pool::kernels::LoadedModule,
    resample: resample::kernels::LoadedModule,
    deform_sample: deform_sample::kernels::LoadedModule,
    dsconv: kernels::dsconv::kernels::LoadedModule,
    deform_backward: deform_sample::backward::LoadedModule,
    foveation_film: foveation_film::kernels::LoadedModule,
    dcnv4: dcnv4::kernels::LoadedModule,
    dcnv4_backward: dcnv4::backward::LoadedModule,
    dcnv4_projection: kernels::dcnv4_projection::kernels::LoadedModule,
    layers_extra: layers_extra::kernels::LoadedModule,
    wmhf: wmhf::kernels::LoadedModule,
    normalize_channels: normalize_channels::kernels::LoadedModule,
    eml: eml::kernels::LoadedModule,
    recurrent: recurrent::kernels::LoadedModule,
    shortcut: shortcut::kernels::LoadedModule,
    scale_channels: scale_channels::kernels::LoadedModule,
    dropout: dropout::kernels::LoadedModule,
    channel_shuffle: channel_shuffle::kernels::LoadedModule,
    reorg: reorg::kernels::LoadedModule,
    softmax: softmax::kernels::LoadedModule,
    dense_loss: dense_loss::kernels::LoadedModule,
    laya: laya::kernels::LoadedModule,
    moe: moe::kernels::LoadedModule,
    attention: attention::kernels::LoadedModule,
    vit: vit::kernels::LoadedModule,
    swin: kernels::swin::kernels::LoadedModule,
    clifford: kernels::clifford::kernels::LoadedModule,
    residual_epilogue: layers_extra::residual::LoadedModule,
    conv_mode: ConvMode,
    /// FP8 formats for the dgrad / wgrad GEMMs when the FP8 backward is
    /// enabled: `(A format, B format)`, i.e. (weights, dy).  `None` keeps the
    /// forward's GEMM mode.
    bwd_fp8: Option<(Fp8Format, Fp8Format)>,
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
///   the same `LIGHTNET_OXIDE_GEMM=fp16`.  Never picked by `Auto`.
/// * `Auto`: `Tf32` when `m * n * k >= AUTO_TF32_MIN_MACS` (2^24, i.e. about
///   256^3) and `min(m, n) >= 32`, else `Fp32`.  Below that the launch is too
///   small to fill the GPU or to amortise the 64x64 tile padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmMode {
    Fp32,
    Tf32,
    Fp16,
    /// bf16 operands on the 16-bit tensor-core GEMM (f32 accumulate and output).
    Bf16,
    /// Per-tensor e4m3 quantisation followed by `mma.m16n8k32` (sm_89+).
    Fp8,
    /// Packed NVFP4 (E2M1 + E4M3 block scales) emulated with `__dp4a`.
    Fp4,
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
    /// `LIGHTNET_OXIDE_GEMM={fp32,tf32,fp16,bf16,fp8,fp4,auto}`; unset or unknown
    /// means `Fp32`.
    ///
    /// `fp8` quantises each operand to e4m3 per tensor (amax / 448) and runs
    /// the Ada tensor cores; `fp4` uses the packed NVFP4 emulation, which needs
    /// both operands contiguous along K and silently falls back to `Fp32`
    /// otherwise (see [`OxideBackend::gemm`]).
    pub fn from_env() -> Self {
        match std::env::var("LIGHTNET_OXIDE_GEMM")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "tf32" => Self::Tf32,
            "fp16" | "f16" | "half" => Self::Fp16,
            "bf16" | "bfloat16" => Self::Bf16,
            "fp8" | "e4m3" => Self::Fp8,
            "fp4" | "nvfp4" => Self::Fp4,
            "auto" => Self::Auto,
            _ => Self::Fp32,
        }
    }

    pub(crate) fn use_tf32(self, g: &GemmArgs) -> bool {
        match self {
            Self::Fp32 | Self::Fp16 | Self::Bf16 | Self::Fp8 | Self::Fp4 => false,
            Self::Tf32 => true,
            Self::Auto => {
                g.m.saturating_mul(g.n).saturating_mul(g.k) >= AUTO_TF32_MIN_MACS
                    && g.m.min(g.n) >= 32
            }
        }
    }
}

/// `LIGHTNET_OXIDE_GEMM_BWD=fp8` turns the conv dgrad / wgrad GEMMs into FP8
/// ones with the C++ recipe's asymmetric formats (e4m3 for weights and
/// activations, e5m2 for the incoming gradient, which is the wider dynamic
/// range the gradients need).  Unset or anything else keeps the forward mode.
fn fp8_backward_from_env() -> Option<(Fp8Format, Fp8Format)> {
    match std::env::var("LIGHTNET_OXIDE_GEMM_BWD")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "fp8" | "e4m3_e5m2" => Some((Fp8Format::E4M3, Fp8Format::E5M2)),
        _ => None,
    }
}

/// Which conv backward GEMM a call belongs to; decides where `dy` sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackwardRole {
    /// `dy` is operand A.
    Wgrad,
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
fn extent(
    base: usize,
    stride: usize,
    modulus: usize,
    count: usize,
    rows: usize,
    cols: usize,
    ld: usize,
) -> usize {
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
        F32ActivationKind::Gelu => activations::ACT_GELU,
        F32ActivationKind::Elu => activations::ACT_ELU,
        F32ActivationKind::Selu => activations::ACT_SELU,
        F32ActivationKind::HardMish => activations::ACT_HARD_MISH,
        F32ActivationKind::Loggy => activations::ACT_LOGGY,
        F32ActivationKind::Hardtan => activations::ACT_HARDTAN,
        F32ActivationKind::Relie => activations::ACT_RELIE,
        F32ActivationKind::Lhtan => activations::ACT_LHTAN,
        F32ActivationKind::Plse => activations::ACT_PLSE,
        F32ActivationKind::Stair => activations::ACT_STAIR,
    }
}

fn channel_normalization_code(normalization: F32ChannelNormalization) -> u32 {
    match normalization {
        F32ChannelNormalization::Relu => 0,
        F32ChannelNormalization::Softmax => 1,
        F32ChannelNormalization::SoftmaxMaxval => 2,
    }
}

fn channel_normalization_size(
    op: &str,
    output_shape: &[usize],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> Result<usize, OxideError> {
    if batch == 0 || channels == 0 || spatial == 0 {
        return Err(shape_err(
            op,
            "batch, channels, and spatial must be nonzero".to_owned(),
        ));
    }
    let total = batch
        .checked_mul(channels)
        .and_then(|elements| elements.checked_mul(spatial))
        .ok_or_else(|| shape_err(op, "tensor length overflows usize".to_owned()))?;
    let shape_len = output_shape
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim))
        .ok_or_else(|| shape_err(op, "output shape overflows usize".to_owned()))?;
    if output_shape.is_empty() || shape_len != total {
        return Err(shape_err(
            op,
            format!("output shape {output_shape:?} has {shape_len} elements, expected {total}"),
        ));
    }
    Ok(total)
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

/// Conv geometry for the implicit-GEMM forward ([`OxideBackend::gemm_conv_implicit`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImplicitConv {
    pub x_base: usize,
    pub channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub cpg: usize,
    pub groups: usize,
    pub ksize: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
    pub out_h: usize,
    pub out_w: usize,
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
            format!("input has {x_shape_len} elements, expected {batch}*{channels}*{in_h}*{in_w}"),
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
        return Err(shape_err(
            op,
            "channels and spatial dims must be > 0".to_owned(),
        ));
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
    let valid_adjustments = match adjustments {
        [scales] => *scales == batch,
        [scales, offsets] => *scales == batch && *offsets == batch * channels,
        _ => false,
    };
    if !valid_adjustments {
        return Err(shape_err(
            op,
            format!(
                "expected per-batch scales ({batch}) and optional per-channel offsets ({}), got {adjustments:?}",
                batch * channels
            ),
        ));
    }
    Ok(())
}

impl OxideBackend {
    /// Native f32 WMHF four-direction scan backward and parameter gradients.
    pub fn wmhf_scan4_backward_f32(
        &self,
        input: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        weights: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "wmhf_scan4_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let total = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "tensor element count overflows".to_owned()))?;
        let direction_len = total
            .checked_mul(4)
            .ok_or_else(|| shape_err(op, "direction scratch size overflows".to_owned()))?;
        let sequences_per_channel = height
            .checked_mul(2)
            .and_then(|value| value.checked_add(width.checked_mul(2)?))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let sequence_count = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(sequences_per_channel))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let output_grad = self.f32v(output_grad)?;
        let weights = self.f32v(weights)?;
        if input.shape() != [batch, channels, height, width]
            || input.len() != total
            || output_grad.shape() != [batch, channels, height, width]
            || output_grad.len() != total
        {
            return Err(shape_err(
                op,
                "input/output-gradient shape mismatch".to_owned(),
            ));
        }
        if weights.shape() != [4, channels] || weights.len() != 4 * channels {
            return Err(shape_err(op, "scan-weight shape mismatch".to_owned()));
        }
        let states = self.alloc_uninit(direction_len)?;
        let input_parts = self.alloc_uninit(direction_len)?;
        let da_parts = self.alloc_uninit(direction_len)?;
        let db_parts = self.alloc_uninit(direction_len)?;
        let dc_parts = self.alloc_uninit(direction_len)?;
        let dd_parts = self.alloc_uninit(direction_len)?;
        let mut input_grad = self.alloc_uninit(total)?;
        let mut weight_grad = self.alloc_uninit(4 * channels)?;
        // SAFETY: one row/column sequence owns each state and partial entry in
        // its direction plane. Follow-up gathers reduce four-way overlaps and
        // channel parameter contributions.
        unsafe {
            self.wmhf.wmhf_scan4_backward_states(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &weights.buf,
                states.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_sequences(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &output_grad.buf,
                &weights.buf,
                &states,
                input_parts.cu_deviceptr() as *mut f32,
                da_parts.cu_deviceptr() as *mut f32,
                db_parts.cu_deviceptr() as *mut f32,
                dc_parts.cu_deviceptr() as *mut f32,
                dd_parts.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_input(
                &self.stream,
                launch::elementwise(total, config),
                &input_parts,
                &mut input_grad,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_backward_weights(
                &self.stream,
                launch::elementwise(4 * channels, config),
                &da_parts,
                &db_parts,
                &dc_parts,
                &dd_parts,
                &mut weight_grad,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_grad, &[batch, channels, height, width]),
            OxideTensorF32::new(weight_grad, &[4, channels]),
        ))
    }

    /// Native f32 WMHF high-frequency magnitude/nearest-upsample branch.
    pub fn wmhf_high_frequency_forward_f32(
        &self,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_high_frequency_forward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let expected_coeff_shape = [batch, channels, height2, width2];
        let lh = self.f32v(lh)?;
        let hl = self.f32v(hl)?;
        let hh = self.f32v(hh)?;
        for (name, tensor) in [("LH", &*lh), ("HL", &*hl), ("HH", &*hh)] {
            if tensor.shape() != expected_coeff_shape || tensor.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!(
                        "{name} shape/length {:?}/{}, expected {expected_coeff_shape:?}/{coeff_len}",
                        tensor.shape(),
                        tensor.len()
                    ),
                ));
            }
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: one thread writes each output pixel and reads its checked
        // nearest-upsampled coefficient cell.
        unsafe {
            self.wmhf.wmhf_high_frequency_forward(
                &self.stream,
                launch::elementwise(output_len, config),
                &lh.buf,
                &hl.buf,
                &hh.buf,
                &mut output,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, channels, height, width],
        ))
    }

    /// Native gather adjoint for the high-frequency magnitude/upsample branch.
    pub fn wmhf_high_frequency_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "wmhf_high_frequency_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let coeff_shape = [batch, channels, height2, width2];
        let output_grad = self.f32v(output_grad)?;
        let lh = self.f32v(lh)?;
        let hl = self.f32v(hl)?;
        let hh = self.f32v(hh)?;
        if output_grad.shape() != [batch, channels, height, width]
            || output_grad.len() != output_len
        {
            return Err(shape_err(
                op,
                "output gradient shape/length mismatch".to_owned(),
            ));
        }
        for (name, tensor) in [("LH", &*lh), ("HL", &*hl), ("HH", &*hh)] {
            if tensor.shape() != coeff_shape || tensor.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!("{name} coefficient shape/length mismatch"),
                ));
            }
        }
        let mut lh_grad = self.alloc_uninit(coeff_len)?;
        let mut hl_grad = self.alloc_uninit(coeff_len)?;
        let mut hh_grad = self.alloc_uninit(coeff_len)?;
        // SAFETY: one thread owns each coefficient gradient and gathers all
        // high-resolution cells mapped to that coefficient.
        unsafe {
            self.wmhf.wmhf_high_frequency_backward(
                &self.stream,
                launch::elementwise(coeff_len, config),
                &output_grad.buf,
                &lh.buf,
                &hl.buf,
                &hh.buf,
                &mut lh_grad,
                &mut hl_grad,
                &mut hh_grad,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(lh_grad, &coeff_shape),
            OxideTensorF32::new(hl_grad, &coeff_shape),
            OxideTensorF32::new(hh_grad, &coeff_shape),
        ))
    }

    /// Gather a contiguous NCHW channel range for a WMHF branch.
    pub fn wmhf_split_channels_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        begin: usize,
        count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_split_channels_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || count == 0 {
            return Err(shape_err(
                op,
                "dimensions and channel count must be positive".to_owned(),
            ));
        }
        let end = begin
            .checked_add(count)
            .ok_or_else(|| shape_err(op, "channel range overflows".to_owned()))?;
        if end > channels {
            return Err(shape_err(op, "channel range exceeds input".to_owned()));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(count)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != [batch, channels, height, width] || input.len() != input_len {
            return Err(shape_err(op, "input shape/length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: each output index gathers one valid source channel/spatial index.
        unsafe {
            self.wmhf.wmhf_split_channels(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                channels,
                height,
                width,
                begin,
                count,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, count, height, width]))
    }

    /// ViT patch gather to `[batch * tokens, patch * patch * channels]`.
    pub fn vit_patchify_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(usize, usize, OxideTensorF32), OxideError> {
        let op = "vit_patchify_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || patch_size == 0 || patch_stride == 0 {
            return Err(shape_err(op, "dimensions, patch size and stride must be positive".to_owned()));
        }
        let padded_h = height + 2 * pad;
        let padded_w = width + 2 * pad;
        if padded_h < patch_size
            || padded_w < patch_size
            || (padded_h - patch_size) % patch_stride != 0
            || (padded_w - patch_size) % patch_stride != 0
        {
            return Err(shape_err(op, "patch extent does not tile the padded input".to_owned()));
        }
        let out_h = (padded_h - patch_size) / patch_stride + 1;
        let out_w = (padded_w - patch_size) / patch_stride + 1;
        let patch_dim = patch_size * patch_size * channels;
        let rows = batch * out_h * out_w;
        let output_len = rows
            .checked_mul(patch_dim)
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != [batch, channels, height, width] {
            return Err(shape_err(op, "input shape mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: every output element is written once from a bounds-checked source.
        unsafe {
            self.vit.vit_patchify(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                channels,
                height,
                width,
                patch_size,
                patch_stride,
                pad,
                out_h,
                out_w,
            )
        }
        .map_err(launch_err)?;
        Ok((out_h, out_w, OxideTensorF32::new(output, &[rows, patch_dim])))
    }

    /// Add a `[tokens, hidden]` table to each sample of `[batch * tokens, hidden]`.
    pub fn vit_add_pos_embed_f32(
        &self,
        tokens: &OxideTensorF32,
        pos: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_add_pos_embed_f32";
        let tokens = self.f32v(tokens)?;
        let pos = self.f32v(pos)?;
        if pos.len() == 0 || tokens.len() % pos.len() != 0 {
            return Err(shape_err(op, "token length is not a multiple of the table".to_owned()));
        }
        let len = tokens.len();
        let shape = tokens.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_add_pos_embed(
                &self.stream,
                launch::elementwise(len, config),
                &tokens.buf,
                &pos.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// Finite clamp used by the ViT feature guards (non-finite becomes zero).
    pub fn vit_clamp_feature_f32(
        &self,
        input: &OxideTensorF32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_clamp_feature_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let input = self.f32v(input)?;
        let len = input.len();
        let shape = input.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_clamp_feature(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// ViT mHC residual mix with host-computed scalar coefficients.
    pub fn vit_mhc_residual_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_mhc_residual_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        if skip.len() != branch.len() {
            return Err(shape_err(op, "skip and branch lengths differ".to_owned()));
        }
        let len = skip.len();
        let shape = skip.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_mhc_residual(
                &self.stream,
                launch::elementwise(len, config),
                &skip.buf,
                &branch.buf,
                &mut output,
                skip_scale,
                branch_scale,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// `[batch * tokens, hidden]` token rows to NCHW `[batch, hidden, out_h, out_w]`.
    pub fn vit_tokens_to_nchw_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_tokens_to_nchw_f32";
        let tokens = out_h * out_w;
        if batch == 0 || tokens == 0 || hidden == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let len = batch * tokens * hidden;
        let input = self.f32v(input)?;
        if input.len() != len {
            return Err(shape_err(op, "input length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_tokens_to_nchw(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                tokens,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, hidden, out_h, out_w]))
    }

    /// NCHW `[batch, hidden, out_h, out_w]` to `[batch * tokens, hidden]` rows.
    pub fn vit_nchw_to_tokens_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_nchw_to_tokens_f32";
        let tokens = out_h * out_w;
        if batch == 0 || tokens == 0 || hidden == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let len = batch * tokens * hidden;
        let input = self.f32v(input)?;
        if input.len() != len {
            return Err(shape_err(op, "input length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_nchw_to_tokens(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                tokens,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch * tokens, hidden]))
    }

    /// Adjoint of [`Self::vit_patchify_f32`]: `[batch * tokens, patch_dim]`
    /// patch gradients accumulated into an NCHW input gradient.
    pub fn vit_patch_scatter_f32(
        &self,
        patch_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_patch_scatter_f32";
        if batch == 0
            || channels == 0
            || height == 0
            || width == 0
            || patch_size == 0
            || patch_stride == 0
            || out_h == 0
            || out_w == 0
        {
            return Err(shape_err(op, "dimensions, patch size and stride must be positive".to_owned()));
        }
        let patch_dim = patch_size * patch_size * channels;
        let patch_len = batch
            .checked_mul(out_h * out_w)
            .and_then(|n| n.checked_mul(patch_dim))
            .ok_or_else(|| shape_err(op, "patch gradient size overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let patch_grad = self.f32v(patch_grad)?;
        if patch_grad.len() != patch_len {
            return Err(shape_err(op, "patch gradient length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: one thread owns each output pixel and reads only in-range taps.
        unsafe {
            self.vit.vit_patch_scatter(
                &self.stream,
                launch::elementwise(output_len, config),
                &patch_grad.buf,
                &mut output,
                channels,
                height,
                width,
                patch_size,
                patch_stride,
                pad,
                out_h,
                out_w,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, channels, height, width]))
    }

    /// Adjoint of [`Self::vit_clamp_feature_f32`] evaluated at its raw input.
    pub fn vit_clamp_backward_f32(
        &self,
        raw: &OxideTensorF32,
        grad: &OxideTensorF32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_clamp_backward_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let raw = self.f32v(raw)?;
        let grad = self.f32v(grad)?;
        if raw.len() != grad.len() {
            return Err(shape_err(op, "raw and gradient lengths differ".to_owned()));
        }
        let len = grad.len();
        let shape = grad.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_clamp_backward(
                &self.stream,
                launch::elementwise(len, config),
                &raw.buf,
                &grad.buf,
                &mut output,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// Adjoint of [`Self::vit_mhc_residual_f32`]: `(d_skip, d_branch, sums)`,
    /// where `sums` holds the `[grad_post_skip, grad_post_branch, grad_mix]`
    /// reductions the host turns into the six mHC parameter gradients.
    pub fn vit_mhc_residual_backward_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        grad: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "vit_mhc_residual_backward_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        let grad = self.f32v(grad)?;
        if skip.len() != branch.len() || skip.len() != grad.len() || grad.len() == 0 {
            return Err(shape_err(op, "skip, branch and gradient lengths differ".to_owned()));
        }
        let len = grad.len();
        let shape = grad.shape().to_vec();
        let mut d_skip = self.alloc_uninit(len)?;
        let mut d_branch = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each element of both outputs.
        unsafe {
            self.vit.vit_mhc_residual_backward(
                &self.stream,
                launch::elementwise(len, config),
                &skip.buf,
                &branch.buf,
                &grad.buf,
                &mut d_skip,
                &mut d_branch,
                skip_scale,
                branch_scale,
                clip,
            )
        }
        .map_err(launch_err)?;
        let mut terms = self.alloc_uninit(3 * len)?;
        // SAFETY: one thread owns each of the `3 * len` planar term elements.
        unsafe {
            self.vit.vit_mhc_param_terms(
                &self.stream,
                launch::elementwise(3 * len, config),
                &skip.buf,
                &branch.buf,
                &grad.buf,
                &mut terms,
                skip_scale,
                branch_scale,
                mix,
                post_skip,
                post_branch,
                clip,
            )
        }
        .map_err(launch_err)?;
        // Planar `[3, len]` terms: three channels of stride `len`, one row group.
        let terms = OxideTensorF32::new(terms, &[3, len]);
        let sums = <Self as F32TensorOpsBackend>::bias_backward_f32(self, &terms, 3, len, config)?;
        Ok((
            OxideTensorF32::new(d_skip, &shape),
            OxideTensorF32::new(d_branch, &shape),
            sums,
        ))
    }

    /// Sum a `[batch, table]` token gradient over the batch (learned positions).
    pub fn vit_pos_embed_backward_f32(
        &self,
        grad: &OxideTensorF32,
        batch: usize,
        table: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_pos_embed_backward_f32";
        if batch == 0 || table == 0 {
            return Err(shape_err(op, "batch and table must be positive".to_owned()));
        }
        let grad = self.f32v(grad)?;
        if Some(grad.len()) != batch.checked_mul(table) {
            return Err(shape_err(op, "gradient length is not batch * table".to_owned()));
        }
        let mut output = self.alloc_uninit(table)?;
        // SAFETY: one thread owns each table entry.
        unsafe {
            self.vit.vit_pos_embed_backward(
                &self.stream,
                launch::elementwise(table, config),
                &grad.buf,
                &mut output,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[table]))
    }

    /// Concatenate the three WMHF local depthwise branch outputs in source order.
    pub fn wmhf_concat_local_f32(
        &self,
        branch3: &OxideTensorF32,
        branch5: &OxideTensorF32,
        branch7: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_concat_local_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let branch_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "branch element count overflows".to_owned()))?;
        let output_channels = channels
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "output channel count overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(output_channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let expected_shape = [batch, channels, height, width];
        let branch3 = self.f32v(branch3)?;
        let branch5 = self.f32v(branch5)?;
        let branch7 = self.f32v(branch7)?;
        for (name, branch) in [("3x", &*branch3), ("5x", &*branch5), ("7x", &*branch7)] {
            if branch.shape() != expected_shape || branch.len() != branch_len {
                return Err(shape_err(
                    op,
                    format!("{name} branch shape/length mismatch"),
                ));
            }
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: each output position selects one valid element from one validated branch.
        unsafe {
            self.wmhf.wmhf_concat_local(
                &self.stream,
                launch::elementwise(output_len, config),
                &branch3.buf,
                &branch5.buf,
                &branch7.buf,
                &mut output,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, output_channels, height, width],
        ))
    }

    /// Native WMHF gate fusion: `fuse + freq_scale * gate * projected + shortcut`.
    pub fn wmhf_apply_gate_f32(
        &self,
        fuse: &OxideTensorF32,
        gate: &OxideTensorF32,
        projected: &OxideTensorF32,
        shortcut: Option<&OxideTensorF32>,
        freq_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_apply_gate_f32";
        let fuse = self.f32v(fuse)?;
        let gate = self.f32v(gate)?;
        let projected = self.f32v(projected)?;
        let shortcut = shortcut.map(|tensor| self.f32v(tensor)).transpose()?;
        if fuse.is_empty()
            || gate.shape() != fuse.shape()
            || gate.len() != fuse.len()
            || projected.shape() != fuse.shape()
            || projected.len() != fuse.len()
        {
            return Err(shape_err(
                op,
                "fuse/gate/projected shape mismatch".to_owned(),
            ));
        }
        if let Some(shortcut) = &shortcut {
            if shortcut.shape() != fuse.shape() || shortcut.len() != fuse.len() {
                return Err(shape_err(op, "shortcut shape mismatch".to_owned()));
            }
        }
        let mut output = self.alloc_uninit(fuse.len())?;
        // SAFETY: each output element has one owner; all inputs have identical validated lengths.
        unsafe {
            self.wmhf.wmhf_apply_gate(
                &self.stream,
                launch::elementwise(fuse.len(), config),
                &fuse.buf,
                &gate.buf,
                &projected.buf,
                shortcut.as_ref().map_or(&fuse.buf, |value| &value.buf),
                &mut output,
                freq_scale,
                shortcut.is_some(),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, fuse.shape()))
    }

    /// Native exact adjoint of WMHF gate fusion.
    pub fn wmhf_apply_gate_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        gate: &OxideTensorF32,
        projected: &OxideTensorF32,
        freq_scale: f32,
        use_shortcut: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "wmhf_apply_gate_backward_f32";
        let output_grad = self.f32v(output_grad)?;
        let gate = self.f32v(gate)?;
        let projected = self.f32v(projected)?;
        if output_grad.is_empty()
            || gate.shape() != output_grad.shape()
            || gate.len() != output_grad.len()
            || projected.shape() != output_grad.shape()
            || projected.len() != output_grad.len()
        {
            return Err(shape_err(
                op,
                "gradient/gate/projected shape mismatch".to_owned(),
            ));
        }
        let mut fuse_grad = self.alloc_uninit(output_grad.len())?;
        let mut gate_grad = self.alloc_uninit(output_grad.len())?;
        let mut projected_grad = self.alloc_uninit(output_grad.len())?;
        let mut shortcut_grad = self.alloc_uninit(output_grad.len())?;
        // SAFETY: each gradient element has one owner and all input extents match.
        unsafe {
            self.wmhf.wmhf_apply_gate_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &output_grad.buf,
                &gate.buf,
                &projected.buf,
                &mut fuse_grad,
                &mut gate_grad,
                &mut projected_grad,
                &mut shortcut_grad,
                freq_scale,
                use_shortcut,
            )
        }
        .map_err(launch_err)?;
        let shape = output_grad.shape();
        Ok((
            OxideTensorF32::new(fuse_grad, shape),
            OxideTensorF32::new(gate_grad, shape),
            OxideTensorF32::new(projected_grad, shape),
            OxideTensorF32::new(shortcut_grad, shape),
        ))
    }

    /// Native f32 WMHF four-direction EMA scan for NCHW tensors and `[4,C]` weights.
    pub fn wmhf_scan4_forward_f32(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_scan4_forward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let total = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "tensor element count overflows".to_owned()))?;
        let direction_len = total
            .checked_mul(4)
            .ok_or_else(|| shape_err(op, "direction scratch size overflows".to_owned()))?;
        let sequences_per_channel = height
            .checked_mul(2)
            .and_then(|value| value.checked_add(width.checked_mul(2)?))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let sequence_count = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(sequences_per_channel))
            .ok_or_else(|| shape_err(op, "scan sequence count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let weights = self.f32v(weights)?;
        if input.shape() != [batch, channels, height, width] || input.len() != total {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected [{batch},{channels},{height},{width}]/{total}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        if weights.shape() != [4, channels] || weights.len() != 4 * channels {
            return Err(shape_err(
                op,
                format!(
                    "weight shape/length are {:?}/{}, expected [{},{}]/{}",
                    weights.shape(),
                    weights.len(),
                    4,
                    channels,
                    4 * channels
                ),
            ));
        }
        let directions = self.alloc_uninit(direction_len)?;
        let mut output = self.alloc_uninit(total)?;
        // SAFETY: each sequence has a unique (batch,channel,line,direction)
        // and writes every element of its disjoint direction plane exactly once.
        unsafe {
            self.wmhf.wmhf_scan4_forward(
                &self.stream,
                launch::elementwise(sequence_count, config),
                &input.buf,
                &weights.buf,
                directions.cu_deviceptr() as *mut f32,
                batch,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.wmhf.wmhf_scan4_combine(
                &self.stream,
                launch::elementwise(total, config),
                &directions,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, channels, height, width],
        ))
    }

    /// Native f32 WMHF Haar decomposition for `[B,C,H,W]` input.
    pub fn wmhf_haar_dwt_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "wmhf_haar_dwt_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.len() != input_len || input.shape() != [batch, channels, height, width] {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected [{batch},{channels},{height},{width}]/{input_len}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut ll = self.alloc_uninit(output_len)?;
        let mut lh = self.alloc_uninit(output_len)?;
        let mut hl = self.alloc_uninit(output_len)?;
        let mut hh = self.alloc_uninit(output_len)?;
        // SAFETY: one thread writes the same valid coefficient index in each
        // sub-band; all input and output extents were checked above.
        unsafe {
            self.wmhf.wmhf_haar_dwt(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut ll,
                &mut lh,
                &mut hl,
                &mut hh,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        let output_shape = [batch, channels, height2, width2];
        Ok((
            OxideTensorF32::new(ll, &output_shape),
            OxideTensorF32::new(lh, &output_shape),
            OxideTensorF32::new(hl, &output_shape),
            OxideTensorF32::new(hh, &output_shape),
        ))
    }

    /// Native Haar synthesis using the exact gather adjoint of DWT.
    pub fn wmhf_haar_iwt_f32(
        &self,
        ll: &OxideTensorF32,
        lh: &OxideTensorF32,
        hl: &OxideTensorF32,
        hh: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.wmhf_haar_dwt_backward_f32(ll, lh, hl, hh, batch, channels, height, width, config)
    }

    /// Native adjoint of Haar synthesis; its sub-bands are exactly the DWT.
    pub fn wmhf_haar_iwt_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        self.wmhf_haar_dwt_f32(output_grad, batch, channels, height, width, config)
    }

    /// Native gather adjoint for four WMHF Haar sub-band gradients.
    pub fn wmhf_haar_dwt_backward_f32(
        &self,
        ll_grad: &OxideTensorF32,
        lh_grad: &OxideTensorF32,
        hl_grad: &OxideTensorF32,
        hh_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "wmhf_haar_dwt_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch/channels/height/width must be positive".to_owned(),
            ));
        }
        let height2 = height.div_ceil(2);
        let width2 = width.div_ceil(2);
        let coeff_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height2))
            .and_then(|value| value.checked_mul(width2))
            .ok_or_else(|| shape_err(op, "coefficient count overflows".to_owned()))?;
        let input_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input gradient count overflows".to_owned()))?;
        let expected_shape = [batch, channels, height2, width2];
        let ll_grad = self.f32v(ll_grad)?;
        let lh_grad = self.f32v(lh_grad)?;
        let hl_grad = self.f32v(hl_grad)?;
        let hh_grad = self.f32v(hh_grad)?;
        for (name, gradient) in [
            ("LL", &*ll_grad),
            ("LH", &*lh_grad),
            ("HL", &*hl_grad),
            ("HH", &*hh_grad),
        ] {
            if gradient.shape() != expected_shape || gradient.len() != coeff_len {
                return Err(shape_err(
                    op,
                    format!(
                        "{name} gradient shape/length are {:?}/{}, expected {expected_shape:?}/{coeff_len}",
                        gradient.shape(),
                        gradient.len()
                    ),
                ));
            }
        }
        let mut input_grad = self.alloc_uninit(input_len)?;
        // SAFETY: one thread owns each input-gradient element and only reads
        // checked sub-band elements from its corresponding Haar tile.
        unsafe {
            self.wmhf.wmhf_haar_dwt_backward(
                &self.stream,
                launch::elementwise(input_len, config),
                &ll_grad.buf,
                &lh_grad.buf,
                &hl_grad.buf,
                &hh_grad.buf,
                &mut input_grad,
                channels,
                height,
                width,
                height2,
                width2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            input_grad,
            &[batch, channels, height, width],
        ))
    }

    pub fn new() -> Result<Self, OxideError> {
        Self::with_device(0)
    }

    pub fn with_device(ordinal: usize) -> Result<Self, OxideError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        // Each `#[cuda_module]` loads the crate's embedded PTX bundle once.
        let elementwise = elementwise::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load elementwise module: {err:?}")))?;
        let layernorm = activations::layernorm::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load layernorm module: {err:?}")))?;
        let activations = activations::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load activations module: {err:?}")))?;
        let bias_scale = bias_scale::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load bias_scale module: {err:?}")))?;
        let batchnorm = batchnorm::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load batchnorm module: {err:?}")))?;
        let muon = muon::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load Muon module: {err:?}")))?;
        let optim = optim::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load optim module: {err:?}")))?;
        let gemm = gemm::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm module: {err:?}")))?;
        let gemm_mma = gemm_mma::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_mma module: {err:?}")))?;
        let gemm_f16 = gemm_f16::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_f16 module: {err:?}")))?;
        let gemm_f8 = gemm_f8::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_f8 module: {err:?}")))?;
        let quant_fp4 = quant_fp4::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load quant_fp4 module: {err:?}")))?;
        let gemm_fp4 = gemm_fp4::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load gemm_fp4 module: {err:?}")))?;
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
        let local = kernels::local::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load local module: {err:?}")))?;
        let graph_conv = kernels::graph_conv::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load graph_conv module: {err:?}")))?;
        let deconvolutional = kernels::deconvolutional::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load deconvolutional module: {err:?}")))?;
        let conv_implicit = conv_implicit::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load conv_implicit module: {err:?}")))?;
        let route = route::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load route module: {err:?}")))?;
        let pool = pool::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load pool module: {err:?}")))?;
        let resample = resample::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load resample module: {err:?}")))?;
        let deform_sample = deform_sample::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load deform_sample module: {err:?}")))?;
        let deform_backward = deform_sample::backward::load(&ctx).map_err(|err| {
            OxideError::Shape(format!("load deform_sample::backward module: {err:?}"))
        })?;
        let dsconv = kernels::dsconv::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dsconv module: {err:?}")))?;
        let foveation_film = foveation_film::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load foveation_film module: {err:?}")))?;
        let dcnv4 = dcnv4::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dcnv4 module: {err:?}")))?;
        let dcnv4_backward = dcnv4::backward::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dcnv4::backward module: {err:?}")))?;
        let dcnv4_projection = kernels::dcnv4_projection::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dcnv4_projection module: {err:?}")))?;
        let residual_epilogue = layers_extra::residual::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load residual epilogue module: {err:?}")))?;
        let vit = vit::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load vit module: {err:?}")))?;
        let swin = kernels::swin::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load swin module: {err:?}")))?;
        let clifford = kernels::clifford::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load clifford module: {err:?}")))?;
        let attention = attention::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load attention module: {err:?}")))?;
        let moe = moe::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load moe module: {err:?}")))?;
        let laya = laya::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load laya module: {err:?}")))?;
        let dense_loss = dense_loss::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dense_loss module: {err:?}")))?;
        let softmax = softmax::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load softmax module: {err:?}")))?;
        let reorg = reorg::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load reorg module: {err:?}")))?;
        let channel_shuffle = channel_shuffle::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load channel_shuffle module: {err:?}")))?;
        let dropout = dropout::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load dropout module: {err:?}")))?;
        let scale_channels = scale_channels::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load scale_channels module: {err:?}")))?;
        let shortcut = shortcut::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load shortcut module: {err:?}")))?;
        let recurrent = recurrent::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load recurrent module: {err:?}")))?;
        let eml = eml::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load eml module: {err:?}")))?;
        let normalize_channels = normalize_channels::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load normalize_channels module: {err:?}")))?;
        let wmhf = wmhf::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load wmhf module: {err:?}")))?;
        let layers_extra = layers_extra::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load layers_extra module: {err:?}")))?;
        let half_scratch = half::HalfScratch::new(&stream)?;
        let fp8_ring = Fp8Ring {
            buf: DeviceBuffer::<f32>::zeroed(&stream, FP8_RING_SLOTS * FP8_SLOT_WORDS)?,
            next: 0,
        };
        let bufs = tensor::BufPool::new();
        let empty_f32 = Arc::new(PooledBuf::new(
            DeviceBuffer::<f32>::zeroed(&stream, 0)?,
            &bufs,
        ));
        let amp_format = mixed::amp_format_from_env();
        let amp = amp_format.is_some();
        Ok(Self {
            _ctx: ctx,
            stream,
            elementwise,
            activations,
            layernorm,
            bias_scale,
            batchnorm,
            muon,
            optim,
            gemm,
            gemm_mma,
            gemm_f16,
            gemm_f8,
            quant_fp4,
            gemm_fp4,
            convert,
            half_ops,
            amp_k,
            amp,
            half_fmt: amp_format.unwrap_or(mixed::HalfFormat::F16),
            amp_w16: Mutex::new(Vec::new()),
            host_readbacks: AtomicU64::new(0),
            fp8_w: Mutex::new(Vec::new()),
            fp8_dy: Mutex::new(None),
            fp8_dy_half: Mutex::new(None),
            fp8_ring: Mutex::new(fp8_ring),
            fp8_scaling: Fp8Scaling::from_env(),
            fp8_hist: Mutex::new(None),
            fp8_quantize_launches: AtomicU64::new(0),
            fp8_amax_passes: AtomicU64::new(0),
            fp8_generic_kernel: AtomicBool::new(false),
            empty_f32,
            half_scratch: Mutex::new(half_scratch),
            half_col: Mutex::new(None),
            gemm_mode: match (amp_format, GemmMode::from_env()) {
                // AMP with `LIGHTNET_OXIDE_GEMM=fp8`: the conv GEMMs run fp8 on
                // the 16-bit activations; every other AMP op keeps its path.
                (Some(_), GemmMode::Fp8) => GemmMode::Fp8,
                (Some(mixed::HalfFormat::F16), _) => GemmMode::Fp16,
                (Some(mixed::HalfFormat::Bf16), _) => GemmMode::Bf16,
                (None, mode) => mode,
            },
            im2col,
            conv_direct,
            local,
            graph_conv,
            deconvolutional,
            conv_implicit,
            route,
            pool,
            resample,
            layers_extra,
            wmhf,
            normalize_channels,
            eml,
            recurrent,
            shortcut,
            scale_channels,
            dropout,
            channel_shuffle,
            reorg,
            softmax,
            dense_loss,
            laya,
            moe,
            attention,
            vit,
            swin,
            deform_sample,
            deform_backward,
            dsconv,
            foveation_film,
            dcnv4,
            dcnv4_backward,
            dcnv4_projection,
            clifford,
            residual_epilogue,
            conv_mode: ConvMode::from_env(),
            bwd_fp8: fp8_backward_from_env(),
            workspace: Mutex::new(None),
            bufs,
        })
    }

    /// Overrides the conv implementation chosen from `LIGHTNET_OXIDE_CONV`.
    pub fn with_conv_mode(mut self, mode: ConvMode) -> Self {
        self.conv_mode = mode;
        self
    }

    /// Overrides the GEMM kernel chosen from `LIGHTNET_OXIDE_GEMM`.
    pub fn with_gemm_mode(mut self, mode: GemmMode) -> Self {
        self.gemm_mode = mode;
        self
    }

    pub fn gemm_mode(&self) -> GemmMode {
        self.gemm_mode
    }

    /// Count of device-to-host readbacks made by the GEMM paths so far.
    /// Tests compare it before and after a call to pin "no readback" behaviour.
    pub fn host_readbacks(&self) -> u64 {
        self.host_readbacks.load(Ordering::Relaxed)
    }

    /// Count of quantise kernel launches made by the FP8 path so far.  A cached
    /// weight costs no launch, so tests compare it to pin the weight cache.
    pub fn fp8_quantize_launches(&self) -> u64 {
        self.fp8_quantize_launches.load(Ordering::Relaxed)
    }

    /// Count of stand-alone amax passes made by [`Self::fp8_quantize_half`] so
    /// far (the first pass of current scaling, or of delayed scaling's call 0).
    /// Delayed scaling measures the amax inside the quantise kernel, so after
    /// call 0 a key adds none.
    pub fn fp8_amax_passes(&self) -> u64 {
        self.fp8_amax_passes.load(Ordering::Relaxed)
    }

    /// Test seam: `true` makes e4m3 x e4m3 GEMMs use the generic-format
    /// `sgemm_fp8_mma` instead of the specialised kernel, so tests can compare
    /// the two bitwise.
    pub fn set_fp8_generic_kernel(&self, generic: bool) {
        self.fp8_generic_kernel.store(generic, Ordering::Relaxed);
    }

    /// Enable (`Some`) or disable (`None`) the FP8 dgrad / wgrad GEMMs with the
    /// given `(A format, B format)`; see [`fp8_backward_from_env`] for the env
    /// spelling and the C++ recipe this mirrors.
    pub fn with_fp8_backward(mut self, formats: Option<(Fp8Format, Fp8Format)>) -> Self {
        self.bwd_fp8 = formats;
        self
    }

    /// Select current or delayed FP8 scaling (default from
    /// `LIGHTNET_OXIDE_FP8_SCALING`).  Cached weights always use current scaling.
    pub fn with_fp8_scaling(mut self, scaling: Fp8Scaling) -> Self {
        self.fp8_scaling = scaling;
        self
    }

    pub fn fp8_scaling(&self) -> Fp8Scaling {
        self.fp8_scaling
    }

    /// Test seam: quantise `x` to `fmt` through the activation quantiser
    /// (honouring [`Fp8Scaling`]) and return the bytes and the scale pair
    /// `[factor, 1 / factor]`.  `key` overrides the delayed-history key (the
    /// source buffer address plus its pool incarnation) so a test can simulate
    /// one call site whose buffer changes between calls.
    pub fn fp8_quantize_probe(
        &self,
        x: &OxideTensorF32,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        let (bytes, scale) = self.fp8_quantize_keyed(&x.buf, x.len(), fmt, key)?;
        let bytes = if bytes.len() == 0 { Vec::new() } else { bytes.to_host_vec(&self.stream)? };
        let scale = scale.to_host_vec(&self.stream)?;
        Ok((bytes, [scale[0], scale[1]]))
    }

    /// The FP8 formats used by the dgrad / wgrad GEMMs, if enabled.
    pub fn fp8_backward_formats(&self) -> Option<(Fp8Format, Fp8Format)> {
        self.bwd_fp8
    }

    /// GEMM seam for the conv backward arms: the forward's [`Self::gemm`] mode,
    /// unless the FP8 backward is on, in which case the operands are quantised
    /// to the configured formats.  The configured pair is `(activation or
    /// weight format, dy format)`; `role` places `dy` as operand B (dgrad) or
    /// operand A (wgrad) so each tensor keeps its own format.
    pub(crate) fn backward_gemm(
        &self,
        role: BackwardRole,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        match self.bwd_fp8 {
            Some((fx, fdy)) => {
                let (fa, fb) = match role {
                    BackwardRole::Wgrad => (fdy, fx),
                };
                let (key_a, key_b) = match role {
                    BackwardRole::Wgrad => (FP8_SITE_WGRAD_A, FP8_SITE_WGRAD_B),
                };
                self.gemm_f32_via_fp8_fmt(a, b, c, g, fa, fb, key_a, key_b)
            }
            None => self.gemm(a, b, c, g),
        }
    }

    pub fn conv_mode(&self) -> ConvMode {
        self.conv_mode
    }

    /// `[1, 1]` on the device: the scale pair of an operand that is not
    /// quantised here (pre-quantised bytes, or an empty operand).
    fn unit_scale(&self) -> Result<PooledBuf, OxideError> {
        let b = DeviceBuffer::<f32>::from_host(&self.stream, &[1.0, 1.0])?;
        Ok(PooledBuf::new(b, &self.bufs))
    }

    /// Word offset of the next zeroed slot of the amax ring.  Once every
    /// [`FP8_RING_SLOTS`] calls the whole ring is zero-filled again (one
    /// memset), which is safe because the stream orders it after every kernel
    /// of the previous lap.
    pub(crate) fn fp8_ring_guard(&self) -> Result<std::sync::MutexGuard<'_, Fp8Ring>, OxideError> {
        self.fp8_ring
            .lock()
            .map_err(|_| shape_err("fp8_ring", "lock poisoned".into()))
    }

    pub(crate) fn fp8_next_slot(&self) -> Result<u32, OxideError> {
        let mut ring = self.fp8_ring_guard()?;
        if ring.next == FP8_RING_SLOTS {
            ring.buf.zero_async(&self.stream)?;
            ring.next = 0;
        }
        let off = ring.next * FP8_SLOT_WORDS;
        ring.next += 1;
        Ok(off as u32)
    }

    /// Quantise the first `len` elements of a device buffer to `fmt`: one amax
    /// pass, then one quantise pass that derives `factor = max_finite / amax`
    /// from the amax slot and also writes the device scale pair
    /// `[factor, 1 / factor]` the mma epilogue consumes.  The host never reads
    /// either word.
    ///
    /// Note the direction: [`fp8::amax_to_scale`] returns the C++ *divisor*
    /// `amax / max_finite`, and the quantiser multiplies, so the factor is its
    /// reciprocal.  Getting this backwards maps every operand onto the bottom
    /// of the format's range (an e5m2 typical value lands below `2^-16` and is
    /// flushed) and silently destroys precision.
    fn fp8_quantize_operand(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
        site: u64,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        self.fp8_quantize_keyed(buf, len, fmt, Some(site))
    }

    /// [`Self::fp8_quantize_operand`] with an optional delayed-history key
    /// override; `None` keys on the buffer's device address and pool
    /// incarnation.
    fn fp8_quantize_keyed(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if self.fp8_scaling == Fp8Scaling::Delayed && len > 0 {
            if let Some(done) = self.fp8_quantize_delayed(Fp8Src::F32(buf), len, fmt, key)? {
                return Ok(done);
            }
        }
        self.fp8_quantize_operand_current(buf, len, fmt)
    }

    /// Delayed scaling for one operand; `None` when the history is full and the
    /// caller should use current scaling.  Call `n` of a key writes slot
    /// `n % 16` and zeroes slot `(n + 2) % 16`; call 0 (empty history) measures
    /// its own amax into its slot, then quantises with it (the two-pass path).
    fn fp8_quantize_delayed(
        &self,
        src: Fp8Src<'_>,
        len: usize,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<Option<(PooledBuf8, PooledBuf)>, OxideError> {
        let hist_len = gemm_f8::F8_HIST_LEN;
        let mut guard = self
            .fp8_hist
            .lock()
            .map_err(|_| shape_err("fp8_hist", "lock poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(Fp8Hist {
                buf: DeviceBuffer::<f32>::zeroed(&self.stream, FP8_HIST_KEYS * hist_len)?,
                keys: std::collections::HashMap::new(),
            });
        }
        let hist = guard.as_mut().expect("initialised above");
        // 16-bit sources fold their format into the code's upper bits, so a
        // pooled buffer that changes dtype never shares a history.
        let (ptr, code) = match src {
            Fp8Src::F32(b) => (b.cu_deviceptr() as u64, fmt.code()),
            Fp8Src::Half(b, hf) => (b.cu_deviceptr() as u64, fmt.code() | ((hf.code() + 1) << 8)),
        };
        let key = match key {
            Some(k) => (k, 0, len, code),
            None => (ptr, self.bufs.incarnation(ptr as usize), len, code),
        };
        let slots_used = hist.keys.len();
        let entry = match hist.keys.get_mut(&key) {
            Some(e) => e,
            None if slots_used < FP8_HIST_KEYS => hist.keys.entry(key).or_insert((slots_used * hist_len, 0)),
            None => return Ok(None),
        };
        let (base, n) = (entry.0, entry.1);
        entry.1 += 1;
        let w = (n % hist_len as u64) as u32;
        let z = ((n + 2) % hist_len as u64) as u32;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        if n == 0 {
            // Empty history: current scaling, with the amax landing in slot `w`.
            // SAFETY: one thread per element; slot `w` of a fresh key is zero and
            // only touched through `fetch_max`.
            let amax_grid = launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32);
            let quant_grid = launch::elementwise(len.max(2), ElementwiseLaunchConfig::default());
            match src {
                Fp8Src::F32(buf) => unsafe {
                    self.gemm_f8
                        .fp8_amax_f32_kernel(&self.stream, amax_grid, buf, &mut hist.buf, base as u32 + w, len as u32)
                        .map_err(launch_err)?;
                    // SAFETY: as `fp8_quantize_operand_current`.
                    self.gemm_f8
                        .fp8_quantize_kernel(
                            &self.stream,
                            quant_grid,
                            buf,
                            &hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32 + w,
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
                Fp8Src::Half(buf, hf) => unsafe {
                    self.fp8_amax_passes.fetch_add(1, Ordering::Relaxed);
                    self.gemm_f8
                        .fp8_amax_half_kernel(&self.stream, amax_grid, buf, &mut hist.buf, base as u32 + w, hf.code(), len as u32)
                        .map_err(launch_err)?;
                    // SAFETY: as `fp8_quantize_half`.
                    self.gemm_f8
                        .fp8_quantize_half_kernel(
                            &self.stream,
                            quant_grid,
                            buf,
                            &hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32 + w,
                            hf.code(),
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
            }
        } else {
            let blocks = len.max(2).div_ceil(gemm_f8::F8_AMAX_BLOCK);
            let grid = launch::blocks_of(blocks, gemm_f8::F8_AMAX_BLOCK as u32);
            // SAFETY: whole 256-thread blocks; the kernel reads only history
            // slots other than `w` / `z`, writes slot `w` atomically and zeroes
            // `z`, writes every byte of `out` and both scale words.
            match src {
                Fp8Src::F32(buf) => unsafe {
                    self.gemm_f8
                        .fp8_quantize_delayed_kernel(
                            &self.stream,
                            grid,
                            buf,
                            &mut hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32,
                            w,
                            z,
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
                Fp8Src::Half(buf, hf) => unsafe {
                    self.gemm_f8
                        .fp8_quantize_delayed_half_kernel(
                            &self.stream,
                            grid,
                            buf,
                            &mut hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32,
                            w,
                            z,
                            hf.code(),
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
            }
        }
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok(Some((out, scale)))
    }

    /// Current-scaling quantiser of one operand (two passes: amax, quantise).
    fn fp8_quantize_operand_current(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if len == 0 {
            return Ok((self.alloc8_uninit(0)?, self.unit_scale()?));
        }
        // Pooled: no cudaMalloc/free per operand.  The amax slot comes zeroed
        // from the ring; the scale pair and bytes are fully written by the
        // quantise kernel.
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: one thread per element; the ring slot is zeroed and only
        // touched through `fetch_max` while the kernel runs.
        unsafe {
            self.gemm_f8.fp8_amax_f32_kernel(
                &self.stream,
                launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                buf,
                &mut ring.buf,
                off,
                len as u32,
            )
        }
        .map_err(launch_err)?;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        // SAFETY: one thread per element (at least two, so both scale words are
        // written); the kernel writes every byte of `out` and both scale words.
        unsafe {
            self.gemm_f8.fp8_quantize_kernel(
                &self.stream,
                launch::elementwise(len.max(2), ElementwiseLaunchConfig::default()),
                buf,
                &ring.buf,
                &mut out,
                &mut scale,
                fmt.code(),
                off,
                len as u32,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok((out, scale))
    }

    /// True when AMP storage is on and the GEMM mode is fp8: the conv GEMMs run
    /// fp8 on the 16-bit activations ([`Self::fp8_quantize_half`]).
    pub(crate) fn amp_fp8(&self) -> bool {
        self.amp && self.gemm_mode == GemmMode::Fp8
    }

    /// [`Self::fp8_quantize_operand_current`] reading 16-bit storage (`hfmt`
    /// says how to widen it): same factor, scale pair and bytes as widening to
    /// f32 first, with no f32 intermediate.  Under [`Fp8Scaling::Delayed`] it
    /// takes the one-pass delayed path of [`Self::fp8_quantize_delayed`] (keyed
    /// on the 16-bit buffer address, its pool incarnation, its length and both
    /// formats); current scaling otherwise.
    pub(crate) fn fp8_quantize_half(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        site: u64,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        self.fp8_quantize_half_keyed(buf, len, fmt, hfmt, Some(site))
    }

    /// [`Self::fp8_quantize_half`] with an optional delayed-history key
    /// override; `None` keys on the buffer's device address and pool
    /// incarnation.
    fn fp8_quantize_half_keyed(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        key: Option<u64>,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if self.fp8_scaling == Fp8Scaling::Delayed && len > 0 {
            if let Some(done) = self.fp8_quantize_delayed(Fp8Src::Half(buf, hfmt), len, fmt, key)? {
                return Ok(done);
            }
        }
        self.fp8_quantize_half_current(buf, len, fmt, hfmt)
    }

    /// Current-scaling (two-pass) body of [`Self::fp8_quantize_half`].
    fn fp8_quantize_half_current(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if len == 0 {
            return Ok((self.alloc8_uninit(0)?, self.unit_scale()?));
        }
        self.fp8_amax_passes.fetch_add(1, Ordering::Relaxed);
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: one thread per element; the ring slot is zeroed and only
        // touched through `fetch_max` while the kernel runs.
        unsafe {
            self.gemm_f8.fp8_amax_half_kernel(
                &self.stream,
                launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                buf,
                &mut ring.buf,
                off,
                hfmt.code(),
                len as u32,
            )
        }
        .map_err(launch_err)?;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        // SAFETY: one thread per element (at least two, so both scale words are
        // written); the kernel writes every byte of `out` and both scale words.
        unsafe {
            self.gemm_f8.fp8_quantize_half_kernel(
                &self.stream,
                launch::elementwise(len.max(2), ElementwiseLaunchConfig::default()),
                buf,
                &ring.buf,
                &mut out,
                &mut scale,
                fmt.code(),
                off,
                hfmt.code(),
                len as u32,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok((out, scale))
    }

    /// Test seam for [`Self::fp8_quantize_half`]: the bytes and the scale pair
    /// `[factor, 1 / factor]`.
    pub fn fp8_quantize_half_probe(
        &self,
        x: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        self.fp8_quantize_half_probe_keyed(x, fmt, hfmt, None)
    }

    /// [`Self::fp8_quantize_half_probe`] with a delayed-history key override
    /// (see [`Self::fp8_quantize_probe`]).
    pub fn fp8_quantize_half_probe_keyed(
        &self,
        x: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        key: Option<u64>,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        let (bytes, scale) = self.fp8_quantize_half_keyed(&x.buf, x.len(), fmt, hfmt, key)?;
        let bytes = if bytes.len() == 0 { Vec::new() } else { bytes.to_host_vec(&self.stream)? };
        let scale = scale.to_host_vec(&self.stream)?;
        Ok((bytes, [scale[0], scale[1]]))
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with `A` and `B` 16-bit buffers
    /// (`hfmt`): each is quantised straight from its 16-bit storage to `fmt_a` /
    /// `fmt_b` and the result accumulates in f32 into `c`.
    pub(crate) fn gemm_fp8_half_operands(
        &self,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(), OxideError> {
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda).min(a.len());
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb).min(b.len());
        let (aq, sa) = self.fp8_quantize_half(a, a_len, fmt_a, hfmt, FP8_SITE_HALF_A)?;
        let (bq, sb) = self.fp8_quantize_half(b, b_len, fmt_b, hfmt, FP8_SITE_HALF_B)?;
        self.gemm_fp8_args(&aq, &bq, c, g, fmt_a, fmt_b, &sa, &sb)
    }

    /// Tensor-level [`Self::gemm_fp8_half_operands`]: `alpha * op(A) * op(B)`
    /// for dense 16-bit `A` (`m x k`, or `k x m` transposed) and `B`, f32
    /// result `m x n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_half(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(m * n)?;
        self.gemm_fp8_half_operands(
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
                beta: 0.0,
                ..GemmArgs::default()
            },
            fmt_a,
            fmt_b,
            hfmt,
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// Quantise two operands together: one zero-fill, one amax launch and one
    /// quantise launch instead of two of each.  Falls back to the per-operand
    /// path when either operand is empty.
    #[allow(clippy::type_complexity)]
    fn fp8_quantize_pair(
        &self,
        a: &DeviceBuffer<f32>,
        a_len: usize,
        fmt_a: Fp8Format,
        key_a: u64,
        b: &DeviceBuffer<f32>,
        b_len: usize,
        fmt_b: Fp8Format,
        key_b: u64,
    ) -> Result<((PooledBuf8, PooledBuf), (PooledBuf8, PooledBuf)), OxideError> {
        if a_len == 0 || b_len == 0 || self.fp8_scaling == Fp8Scaling::Delayed {
            return Ok((
                self.fp8_quantize_operand(a, a_len, fmt_a, key_a)?,
                self.fp8_quantize_operand(b, b_len, fmt_b, key_b)?,
            ));
        }
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: whole 256-thread blocks per operand tile; the ring slot is
        // zeroed and only touched through `fetch_max`.
        let tile = gemm_f8::F8_AMAX_BLOCK;
        let blocks = a_len.div_ceil(tile) + b_len.div_ceil(tile);
        unsafe {
            self.gemm_f8.fp8_amax_pair_kernel(
                &self.stream,
                launch::blocks_of(blocks, tile as u32),
                a,
                b,
                a_len as u32,
                b_len as u32,
                &mut ring.buf,
                off,
            )
        }
        .map_err(launch_err)?;
        let mut out_a = self.alloc8_uninit(a_len)?;
        let mut out_b = self.alloc8_uninit(b_len)?;
        let mut scale_a = self.alloc_uninit(2)?;
        let mut scale_b = self.alloc_uninit(2)?;
        let threads = a_len.max(2) + b_len.max(2);
        // SAFETY: the kernel writes every byte of both outputs and both words of
        // both scale pairs, each from a distinct thread.
        unsafe {
            self.gemm_f8.fp8_quantize_pair_kernel(
                &self.stream,
                launch::elementwise(threads, ElementwiseLaunchConfig::default()),
                a,
                b,
                &ring.buf,
                &mut out_a,
                &mut out_b,
                &mut scale_a,
                &mut scale_b,
                a_len as u32,
                b_len as u32,
                fmt_a.code(),
                fmt_b.code(),
                off,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(2, Ordering::Relaxed);
        Ok(((out_a, scale_a), (out_b, scale_b)))
    }

    /// Absolute maximum of the first `len` elements of `buf` in a ring slot
    /// (returns its word offset; read by the device, never by the host).  Used by the
    /// byte im2col, which quantises while it gathers.
    pub(crate) fn fp8_amax_slot(&self, buf: &DeviceBuffer<f32>, len: usize) -> Result<u32, OxideError> {
        let off = self.fp8_next_slot()?;
        if len > 0 {
            let mut ring = self.fp8_ring_guard()?;
            // SAFETY: one thread per element; the ring slot is zeroed and only
            // touched through `fetch_max`.
            unsafe {
                self.gemm_f8.fp8_amax_f32_kernel(
                    &self.stream,
                    launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                    buf,
                    &mut ring.buf,
                    off,
                    len as u32,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(off)
    }

    /// Quantised copy of a weight tensor, cached by buffer identity and format.
    /// Weight tensors are immutable (an optimizer step returns a new tensor), so
    /// the copy refreshes once per update.  Only weight tensors may use this:
    /// pooled activation buffers are reused under the same address, so an
    /// identity key on them would serve stale bytes.
    pub(crate) fn fp8_weights(&self, w: &OxideTensorF32, fmt: Fp8Format) -> Result<mixed::Fp8Weights, OxideError> {
        let ptr = Arc::as_ptr(&w.buf);
        {
            let cache = self
                .fp8_w
                .lock()
                .map_err(|_| shape_err("fp8_weights", "lock poisoned".into()))?;
            if let Some(e) = cache
                .iter()
                .find(|e| e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.fmt == fmt)
            {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_operand_current(&w.buf, w.len(), fmt)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        let mut cache = self
            .fp8_w
            .lock()
            .map_err(|_| shape_err("fp8_weights", "lock poisoned".into()))?;
        cache.retain(|e| e.key.strong_count() > 0);
        if cache.len() >= mixed::W16_CACHE_MAX {
            cache.remove(0);
        }
        cache.push(mixed::Fp8WEntry {
            key: Arc::downgrade(&w.buf),
            fmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// Quantised copy of a conv output gradient, shared by the dgrad and the
    /// wgrad of one backward: the first caller quantises the whole tensor (one
    /// keyed quantise, so delayed scaling advances its history once), the
    /// second gets the same bytes and scale pair.  Only the most recent
    /// gradient is kept.  Valid because tensors are immutable once built and a
    /// dead tensor's entry is skipped by the `Weak` check.
    pub(crate) fn fp8_dy(&self, dy: &OxideTensorF32, fmt: Fp8Format) -> Result<mixed::Fp8Weights, OxideError> {
        let lock_err = || shape_err("fp8_dy", "lock poisoned".into());
        let ptr = Arc::as_ptr(&dy.buf);
        {
            let cache = self.fp8_dy.lock().map_err(|_| lock_err())?;
            if let Some(e) = cache.as_ref().filter(|e| {
                e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.len == dy.len() && e.fmt == fmt
            }) {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_operand(&dy.buf, dy.len(), fmt, FP8_SITE_DY)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        *self.fp8_dy.lock().map_err(|_| lock_err())? = Some(mixed::Fp8DyEntry {
            key: Arc::downgrade(&dy.buf),
            len: dy.len(),
            fmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// [`Self::fp8_dy`] for a 16-bit-storage `dy` (AMP): dgrad and wgrad of one
    /// conv backward share one quantise of the whole gradient.  Keyed on the
    /// half buffer identity, its length and the formats.
    pub(crate) fn fp8_dy_half(
        &self,
        dy: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<mixed::Fp8Weights, OxideError> {
        let lock_err = || shape_err("fp8_dy_half", "lock poisoned".into());
        let ptr = Arc::as_ptr(&dy.buf);
        {
            let cache = self.fp8_dy_half.lock().map_err(|_| lock_err())?;
            if let Some(e) = cache.as_ref().filter(|e| {
                e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.len == dy.len() && e.fmt == fmt && e.hfmt == hfmt
            }) {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_half(&dy.buf, dy.len(), fmt, hfmt, FP8_SITE_DY)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        *self.fp8_dy_half.lock().map_err(|_| lock_err())? = Some(mixed::Fp8DyHalfEntry {
            key: Arc::downgrade(&dy.buf),
            len: dy.len(),
            fmt,
            hfmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// wgrad with the column already FP8 bytes (B) and `dy` (A) quantised once
    /// through [`Self::fp8_dy`].
    pub(crate) fn gemm_fp8_dy_a_bytes(
        &self,
        dy: &OxideTensorF32,
        fmt_a: Fp8Format,
        bq: &DeviceBuffer<u8>,
        fmt_b: Fp8Format,
        sb: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        let dq = self.fp8_dy(dy, fmt_a)?;
        self.gemm_fp8_args(&dq.bytes, bq, c, g, fmt_a, fmt_b, &dq.scale, sb)
    }

    /// dgrad GEMM: weight operand A (cached FP8 copy) times `dy` as operand B,
    /// quantised once through [`Self::fp8_dy`].
    pub(crate) fn backward_gemm_weights_dy(
        &self,
        w: &OxideTensorF32,
        dy: &OxideTensorF32,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        match self.bwd_fp8 {
            Some((fa, fb)) => {
                let wq = self.fp8_weights(w, fa)?;
                let dq = self.fp8_dy(dy, fb)?;
                self.gemm_fp8_args(&wq.bytes, &dq.bytes, c, g, fa, fb, &wq.scale, &dq.scale)
            }
            None => self.gemm_weights_a(w, &dy.buf, c, g),
        }
    }

    /// `C = alpha * W * op(B) + beta * C` where `W` is a weight tensor and its
    /// quantised copy comes from [`Self::fp8_weights`]; `B` is quantised per call.
    pub(crate) fn gemm_fp8_weights_a(
        &self,
        w: &OxideTensorF32,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<(), OxideError> {
        let wq = self.fp8_weights(w, fmt_a)?;
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        let (bq, sb) = self.fp8_quantize_operand(b, b_len, fmt_b, FP8_SITE_WEIGHTS_B)?;
        self.gemm_fp8_args(&wq.bytes, &bq, c, g, fmt_a, fmt_b, &wq.scale, &sb)
    }

    /// Forward GEMM seam for a weight operand A (conv forward).  In FP8 mode the
    /// weights come from the cache; otherwise this is [`Self::gemm`].
    pub(crate) fn gemm_weights_a(
        &self,
        w: &OxideTensorF32,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        if self.gemm_mode == GemmMode::Fp8 && !self.amp {
            return self.gemm_fp8_weights_a(w, b, c, g, Fp8Format::E4M3, Fp8Format::E4M3);
        }
        self.gemm(&w.buf, b, c, g)
    }

    /// `C = alpha * A * op(B) + beta * C` with `A` a weight tensor (cached FP8
    /// copy in FP8 mode).  The tensor-level counterpart of the conv seam.
    pub fn gemm_f32_weights_a(
        &self,
        w: &OxideTensorF32,
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
        out.copy_from_device_async(&c.buf, &self.stream)?;
        self.gemm_weights_a(
            w,
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

    /// `C = alpha * op(A) * op(B) + beta * C` with both operands quantised to
    /// `fmt_a` / `fmt_b` per tensor (`amax / max_finite`), run on the Ada FP8
    /// tensor cores.  Accumulates in f32; the error is the operand rounding
    /// (<= 2^-4 relative for e4m3, 2^-3 for e5m2, per operand).
    fn gemm_f32_via_fp8_fmt(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        key_a: u64,
        key_b: u64,
    ) -> Result<(), OxideError> {
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        let ((aq, sa), (bq, sb)) =
            self.fp8_quantize_pair(a, a_len, fmt_a, key_a, b, b_len, fmt_b, key_b)?;
        // The mma works in the quantised domain: the epilogue multiplies by the
        // device-side reciprocal factors `sa[1] * sb[1]`, so `c` needs no pass.
        self.gemm_fp8_args(&aq, &bq, c, g, fmt_a, fmt_b, &sa, &sb)
    }

    /// e4m3 x e4m3, the forward format pair.
    fn gemm_f32_via_fp8(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        self.gemm_f32_via_fp8_fmt(
            a,
            b,
            c,
            g,
            Fp8Format::E4M3,
            Fp8Format::E4M3,
            FP8_SITE_FORWARD_A,
            FP8_SITE_FORWARD_B,
        )
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with both operands quantised to
    /// packed NVFP4 (E2M1 codes, one E4M3 scale per 16 elements along K, plus
    /// an f32 global scale) and multiplied on the CUDA cores with `__dp4a`.
    ///
    /// This is an *emulation*: native FP4 `mma` needs sm_120a, so it is slower
    /// than fp16 by design and exists for parity with the C++ `raptor-fp4`
    /// reference.  The packed layout is only defined for operands contiguous
    /// along K, so this path transposes on the host (a full round trip) when a
    /// caller hands in an N-contiguous operand - acceptable for an emulation
    /// whose whole purpose is reference parity, not speed.
    fn gemm_f32_via_fp4(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp4: {msg}"));
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        if a_len < g.m * g.k || b_len < g.n * g.k {
            return Err(bad("operand buffer smaller than m*k / n*k"));
        }
        // Instance 0 only: the packed GEMM has no batched form yet.
        if g.count > 1 {
            return Err(bad("the packed NVFP4 path handles a single GEMM instance"));
        }
        let a_raw = a.to_host_vec(&self.stream)?;
        let b_raw = b.to_host_vec(&self.stream)?;
        // Pack A as [m][k] and B as [n][k], transposing whichever operand the
        // caller stored N-contiguous.
        let a_k: Vec<f32> = if g.trans_a {
            let mut v = vec![0.0f32; g.m * g.k];
            for p in 0..g.k {
                for i in 0..g.m {
                    v[i * g.k + p] = a_raw[g.a.0 + p * g.lda + i];
                }
            }
            v
        } else {
            a_raw[g.a.0..g.a.0 + g.m * g.k].to_vec()
        };
        let b_k: Vec<f32> = if g.trans_b {
            b_raw[g.b.0..g.b.0 + g.n * g.k].to_vec()
        } else {
            let mut v = vec![0.0f32; g.n * g.k];
            for p in 0..g.k {
                for j in 0..g.n {
                    v[j * g.k + p] = b_raw[g.b.0 + p * g.ldb + j];
                }
            }
            v
        };
        let pa = self::fp4::quantize_1x16(&a_k, g.m, g.k);
        let pb = self::fp4::quantize_1x16(&b_k, g.n, g.k);
        let c_host = if g.beta == 0.0 {
            Vec::new()
        } else {
            c.to_host_vec(&self.stream)?
        };
        let result = self.gemm_fp4(
            &pa,
            &pb,
            if g.beta == 0.0 { None } else { Some(&c_host) },
            g.m,
            g.n,
            g.k,
            g.alpha,
            g.beta,
            false,
        )?;
        c.copy_from_host(&self.stream, &result)?;
        Ok(())
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
        if self.amp_fp8() {
            // Only the AMP conv GEMMs are fp8; any other op that reaches the f32
            // GEMM seam keeps the 16-bit compute path.
            return self.gemm_f32_operands_as_half(a, b, c, g, self.half_fmt == mixed::HalfFormat::Bf16);
        }
        if self.gemm_mode == GemmMode::Fp16 {
            return self.gemm_f32_operands_as_half(a, b, c, g, false);
        }
        if self.gemm_mode == GemmMode::Bf16 {
            return self.gemm_f32_operands_as_half(a, b, c, g, true);
        }
        if self.gemm_mode == GemmMode::Fp8 {
            return self.gemm_f32_via_fp8(a, b, c, g);
        }
        if self.gemm_mode == GemmMode::Fp4 {
            return self.gemm_f32_via_fp4(a, b, c, g);
        }
        // Split-K: with fewer than `SPLITK_TARGET_BLOCKS` output tiles the GPU is
        // under-filled while each block walks a long K loop (deep, small-spatial
        // convs).  Split K across extra blocks that write dense partial products
        // to a workspace, then reduce them in a fixed order (deterministic).
        let tf32 = self.gemm_mode.use_tf32(g);
        let (bm, bn) = if tf32 {
            (gemm_mma::MMA_BM, gemm_mma::MMA_BN)
        } else {
            (gemm::BM, gemm::BN)
        };
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS
                .div_ceil(tiles)
                .min(g.k / SPLITK_MIN_K)
                .min(SPLITK_MAX)
                .max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 {
            g.k.div_ceil(splits).next_multiple_of(16)
        } else {
            g.k
        };
        let splits = g.k.div_ceil(kchunk); // drop empty trailing splits
        let mn = g.m * g.n;
        let mut ws = if splits > 1 {
            Some(self.alloc_uninit(splits * g.count * mn)?)
        } else {
            None
        };
        let (alpha, beta, ldc) = if splits > 1 {
            (1.0, 0.0, g.n)
        } else {
            (g.alpha, g.beta, g.ldc)
        };
        let (cb, cs, cm) = if splits > 1 {
            (0, mn, splits * g.count)
        } else {
            g.c
        };
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
                    a,
                    b,
                    target,
                    g.m,
                    g.n,
                    g.k,
                    g.lda,
                    g.ldb,
                    ldc,
                    g.trans_a as u32,
                    g.trans_b as u32,
                    alpha,
                    beta,
                    g.a.0,
                    g.a.1,
                    g.a.2,
                    g.b.0,
                    g.b.1,
                    g.b.2,
                    cb,
                    cs,
                    cm,
                    g.count,
                    kchunk,
                )
            }
        } else {
            unsafe {
                self.gemm.sgemm_tiled(
                    &self.stream,
                    launch::blocks_of(blocks, gemm::GEMM_THREADS),
                    a,
                    b,
                    target,
                    g.m,
                    g.n,
                    g.k,
                    g.lda,
                    g.ldb,
                    ldc,
                    g.trans_a as u32,
                    g.trans_b as u32,
                    alpha,
                    beta,
                    g.a.0,
                    g.a.1,
                    g.a.2,
                    g.b.0,
                    g.b.1,
                    g.b.2,
                    cb,
                    cs,
                    cm,
                    g.count,
                    kchunk,
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
                    &w,
                    c,
                    g.m,
                    g.n,
                    g.ldc,
                    g.count,
                    splits,
                    g.alpha,
                    g.beta,
                    g.c.0,
                    g.c.1,
                    g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
    }

    /// Implicit-GEMM conv forward: same math and instance layout as [`Self::gemm`]
    /// with `B = col` materialised, but the `B` tile is generated from `x` on
    /// the fly (fp32 only; callers fall back to the materialised path for
    /// tf32/fp16 modes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_conv_implicit(
        &self,
        a: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        conv: &ImplicitConv,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_conv_implicit: {msg}"));
        if g.m == 0 || g.n == 0 || g.k == 0 || g.count == 0 {
            return Err(bad("m, n, k and count must be > 0"));
        }
        if g.trans_a || g.trans_b {
            return Err(bad("only the NN layout is supported"));
        }
        if g.a.2 == 0 || g.c.2 == 0 {
            return Err(bad("instance modulus must be > 0"));
        }
        if g.lda < g.k || g.ldc < g.n {
            return Err(bad("leading dimension smaller than row length"));
        }
        if extent(g.a.0, g.a.1, g.a.2, g.count, g.m, g.k, g.lda) > a.len()
            || extent(g.c.0, g.c.1, g.c.2, g.count, g.m, g.n, g.ldc) > c.len()
        {
            return Err(bad("operand buffer too small for the requested view"));
        }
        let (bm, bn) = (conv_implicit::BM, conv_implicit::BN);
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS
                .div_ceil(tiles)
                .min(g.k / SPLITK_MIN_K)
                .min(SPLITK_MAX)
                .max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 {
            g.k.div_ceil(splits).next_multiple_of(16)
        } else {
            g.k
        };
        let splits = g.k.div_ceil(kchunk); // drop empty trailing splits
        let mn = g.m * g.n;
        let mut ws = if splits > 1 {
            Some(self.alloc_uninit(splits * g.count * mn)?)
        } else {
            None
        };
        let (alpha, beta, ldc) = if splits > 1 {
            (1.0, 0.0, g.n)
        } else {
            (g.alpha, g.beta, g.ldc)
        };
        let (cb, cs, cm) = if splits > 1 {
            (0, mn, splits * g.count)
        } else {
            g.c
        };
        let target: &mut DeviceBuffer<f32> = match ws.as_mut() {
            Some(w) => w,
            None => c,
        };
        let blocks = tiles * splits;
        // SAFETY: extents validated above; the workspace is sized for every
        // split, and every C element has one owner thread.
        unsafe {
            self.conv_implicit.sgemm_conv_implicit(
                &self.stream,
                launch::blocks_of(blocks, conv_implicit::GEMM_THREADS),
                a,
                x,
                target,
                g.m,
                g.n,
                g.k,
                g.lda,
                ldc,
                alpha,
                beta,
                g.a.0,
                g.a.1,
                g.a.2,
                cb,
                cs,
                cm,
                g.count,
                kchunk,
                conv.x_base,
                conv.channels,
                conv.in_h,
                conv.in_w,
                conv.cpg,
                conv.groups,
                conv.ksize,
                conv.stride_x,
                conv.stride_y,
                conv.dilation,
                conv.padding,
                conv.out_h,
                conv.out_w,
            )
        }
        .map_err(launch_err)?;
        if let Some(w) = ws {
            // SAFETY: one thread per output element; extents validated above.
            unsafe {
                self.gemm.gemm_splitk_reduce(
                    &self.stream,
                    launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                    &w,
                    c,
                    g.m,
                    g.n,
                    g.ldc,
                    g.count,
                    splits,
                    g.alpha,
                    g.beta,
                    g.c.0,
                    g.c.1,
                    g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
    }

    /// Native f32 Newton–Schulz orthogonalization for one row-major matrix.
    /// Uses five paper-coefficient iterations and the smaller Gram dimension.
    /// Inputs are expected to be finite; shape errors are returned explicitly.
    pub fn muon_orthogonalize_f32(
        &self,
        matrix: &OxideTensorF32,
        rows: usize,
        cols: usize,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "muon_orthogonalize_f32";
        let len = rows
            .checked_mul(cols)
            .ok_or_else(|| shape_err(op, "matrix extent overflow".to_owned()))?;
        if rows == 0 || cols == 0 || matrix.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "expected non-empty {rows}x{cols} matrix, got {} elements",
                    matrix.len()
                ),
            ));
        }
        let config = ElementwiseLaunchConfig::default();
        let norm_squared = DeviceBuffer::<f32>::zeroed(&self.stream, 1)?;
        // SAFETY: `norm_squared` is one aligned, zero-initialized scalar and
        // every input thread contributes through an atomic add.
        unsafe {
            self.muon.muon_sum_squares_f32(
                &self.stream,
                launch::elementwise(len, config),
                &matrix.buf,
                norm_squared.cu_deviceptr() as *mut f32,
            )
        }
        .map_err(launch_err)?;
        let mut normalized = self.alloc_uninit(len)?;
        // SAFETY: one thread per matrix element; norm/output extents are valid.
        unsafe {
            self.muon.muon_normalize_f32(
                &self.stream,
                launch::elementwise(len, config),
                &matrix.buf,
                &norm_squared,
                &mut normalized,
            )
        }
        .map_err(launch_err)?;

        let transposed = rows > cols;
        let (small, large) = if transposed {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let mut x = OxideTensorF32::new(normalized, &[rows, cols]);
        if transposed {
            x = self.muon_transpose_f32(&x, rows, cols, config)?;
        }
        for _ in 0..5 {
            let x_t = self.muon_transpose_f32(&x, small, large, config)?;
            let gram =
                self.linear_f32(&x, &x, None, &[small, small], small, large, small, config)?;
            let ax = self.linear_f32(
                &gram,
                &x_t,
                None,
                &[small, large],
                small,
                small,
                large,
                config,
            )?;
            let ax_t = self.muon_transpose_f32(&ax, small, large, config)?;
            let a2x = self.linear_f32(
                &gram,
                &ax_t,
                None,
                &[small, large],
                small,
                small,
                large,
                config,
            )?;
            x = self.muon_combine_f32(&x, &ax, &a2x, small, large, config)?;
        }
        if transposed {
            x = self.muon_transpose_f32(&x, small, large, config)?;
        }
        Ok(x)
    }

    /// Native device momentum and Nesterov-lookahead update for one Muon matrix.
    pub fn muon_momentum_lookahead_f32(
        &self,
        updates: &OxideTensorF32,
        momentum: &OxideTensorF32,
        shape: &[usize],
        batch: usize,
        loss_scale: f32,
        coefficient: f32,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "muon_momentum_lookahead_f32";
        if updates.len() == 0
            || momentum.len() != updates.len()
            || shape.iter().product::<usize>() != updates.len()
        {
            return Err(shape_err(
                op,
                "updates, momentum, and shape must have equal nonzero lengths".to_owned(),
            ));
        }
        if !coefficient.is_finite() || !(0.0..1.0).contains(&coefficient) {
            return Err(shape_err(
                op,
                "momentum coefficient must be finite and in [0, 1)".to_owned(),
            ));
        }
        let scale = if loss_scale.is_finite() && loss_scale > 0.0 {
            loss_scale
        } else {
            1.0
        };
        let divisor = batch.max(1) as f32 * scale;
        if !divisor.is_finite() || divisor <= 0.0 {
            return Err(shape_err(
                op,
                "gradient divisor must be finite and positive".to_owned(),
            ));
        }
        let config = ElementwiseLaunchConfig::default();
        let mut next = self.alloc_uninit(updates.len())?;
        let mut lookahead = self.alloc_uninit(updates.len())?;
        // SAFETY: all buffers have equal validated lengths; each thread writes a unique output slot.
        unsafe {
            self.muon.muon_momentum_lookahead_f32(
                &self.stream,
                launch::elementwise(updates.len(), config),
                &updates.buf,
                &momentum.buf,
                &mut next,
                &mut lookahead,
                divisor,
                coefficient,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(next, shape),
            OxideTensorF32::new(lookahead, shape),
        ))
    }

    /// Apply Muon direction and decoupled AdamW decay entirely on the device.
    pub fn muon_apply_update_f32(
        &self,
        weights: &OxideTensorF32,
        direction: &OxideTensorF32,
        shape: &[usize],
        learning_rate: f32,
        decay: f32,
        direction_scale: f32,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "muon_apply_update_f32";
        if weights.len() == 0
            || direction.len() != weights.len()
            || shape.iter().product::<usize>() != weights.len()
        {
            return Err(shape_err(
                op,
                "weights, direction, and shape must have equal nonzero lengths".to_owned(),
            ));
        }
        if !learning_rate.is_finite()
            || learning_rate < 0.0
            || !decay.is_finite()
            || decay < 0.0
            || !direction_scale.is_finite()
            || direction_scale < 0.0
        {
            return Err(shape_err(
                op,
                "learning rate, decay, and direction scale must be finite and non-negative"
                    .to_owned(),
            ));
        }
        let decay_factor = 1.0 - learning_rate * decay;
        if !decay_factor.is_finite() {
            return Err(shape_err(
                op,
                "decoupled decay factor must be finite".to_owned(),
            ));
        }
        let config = ElementwiseLaunchConfig::default();
        let mut output = self.alloc_uninit(weights.len())?;
        // SAFETY: all input/output buffers have equal validated lengths; each thread writes one slot.
        unsafe {
            self.muon.muon_apply_update_f32(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &direction.buf,
                &mut output,
                learning_rate,
                decay_factor,
                direction_scale,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn muon_transpose_f32(
        &self,
        matrix: &OxideTensorF32,
        rows: usize,
        cols: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut output = self.alloc_uninit(rows * cols)?;
        // SAFETY: one thread owns each transposed output element.
        unsafe {
            self.muon.muon_transpose_f32(
                &self.stream,
                launch::elementwise(rows * cols, config),
                &matrix.buf,
                &mut output,
                rows,
                cols,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[cols, rows]))
    }

    fn muon_combine_f32(
        &self,
        x: &OxideTensorF32,
        ax: &OxideTensorF32,
        a2x: &OxideTensorF32,
        rows: usize,
        cols: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let len = rows * cols;
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: all inputs and output have the validated matrix extent.
        unsafe {
            self.muon.muon_linear_combination_f32(
                &self.stream,
                launch::elementwise(len, config),
                3.4445,
                &x.buf,
                -4.7750,
                &ax.buf,
                2.0315,
                &a2x.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[rows, cols]))
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
        out.copy_from_device_async(&c.buf, &self.stream)?;
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

    // ---- FP8 (e4m3 / e5m2) path -----------------------------------------
    //
    // The FP8 GEMM runs on the Ada tensor cores (`mma.m16n8k32` with 8-bit
    // operands) and needs 8-bit operands, so the host has to quantise first.
    // `fp8_amax` -> `fp8::amax_to_scale` -> `quantize_f32_to_fp8` is the
    // recipe of the C++ reference (`v6-dev-fp8`, `fp8_scaling.cpp`); the
    // scale is applied to the quantised value, so the GEMM itself stays a
    // plain integer-domain mma with an f32 accumulator.

    /// Largest finite `|x|` over `values`; non-finite entries count as 0.
    /// Same rule as [`fp8::amax_f32`] and the C++ amax kernels.
    pub fn fp8_amax(&self, values: &[f32]) -> Result<f32, OxideError> {
        if values.is_empty() {
            return Ok(0.0);
        }
        let x = DeviceBuffer::<f32>::from_host(&self.stream, values)?;
        let mut slot = DeviceBuffer::<f32>::zeroed(&self.stream, 1)?;
        // SAFETY: one thread per input element; the single atomic slot is
        // zeroed above and only ever read through `fetch_max`.
        unsafe {
            self.gemm_f8.fp8_amax_f32_kernel(
                &self.stream,
                launch::blocks_of(values.len().div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                &x,
                &mut slot,
                0,
                values.len() as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(slot.to_host_vec(&self.stream)?[0])
    }

    /// `bytes[i] = quantize(values[i] * scale, fmt)`.
    pub fn quantize_f32_to_fp8(
        &self,
        values: &[f32],
        scale: f32,
        fmt: Fp8Format,
    ) -> Result<Vec<u8>, OxideError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let x = DeviceBuffer::<f32>::from_host(&self.stream, values)?;
        let s = DeviceBuffer::<f32>::from_host(&self.stream, &[scale])?;
        let mut out = DeviceBuffer::<u8>::zeroed(&self.stream, values.len())?;
        // SAFETY: one thread per element; both buffers have `values.len()`
        // elements.
        unsafe {
            self.gemm_f8.fp8_quantize_scaled_kernel(
                &self.stream,
                launch::elementwise(values.len(), ElementwiseLaunchConfig::default()),
                &x,
                &mut out,
                &s,
                fmt.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// `out[i] = f32(bytes[i])`; the quantisation scale is *not* reapplied.
    pub fn dequantize_fp8_to_f32(
        &self,
        bytes: &[u8],
        fmt: Fp8Format,
    ) -> Result<Vec<f32>, OxideError> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let x = DeviceBuffer::<u8>::from_host(&self.stream, bytes)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, bytes.len())?;
        // SAFETY: one thread per element; both buffers have `bytes.len()`
        // elements.
        unsafe {
            self.gemm_f8.fp8_dequantize_kernel(
                &self.stream,
                launch::elementwise(bytes.len(), ElementwiseLaunchConfig::default()),
                &x,
                &mut out,
                fmt.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Control probe with the f16 `mma.m16n8k16`, same harness as
    /// [`Self::fp8_mma_probe16`] but with 16-bit operands.  One warp of
    /// `m16n8k16` fragments is 32 x 8 A halves and 32 x 4 B halves.
    pub fn f16_mma_probe16(&self, a: &[u16], b: &[u16]) -> Result<Vec<f32>, OxideError> {
        if a.len() < 256 || b.len() < 128 {
            return Err(shape_err(
                "f16_mma_probe16",
                format!(
                    "expected 256 A and 128 B halves, got {} and {}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u16>::from_host(&self.stream, &a[..256])?;
        let db = DeviceBuffer::<u16>::from_host(&self.stream, &b[..128])?;
        let mut out = DeviceBuffer::<f32>::from_host(&self.stream, &vec![-999.0f32; 128])?;
        // SAFETY: a single warp; every lane's fragments are present and each
        // output slot has exactly one owner.
        unsafe {
            self.gemm_f8
                .f16_mma_probe16(&self.stream, launch::blocks_of(1, 32), &da, &db, &mut out)
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Raw-fragment `mma.m16n8k16` probe: one warp of fragments is 32 lanes x
    /// 8 A bytes and 32 lanes x 4 B bytes; the result is
    /// `out[i * 32 + lane] = D[lane][i]` for a 16x8 f32 tile.
    pub fn fp8_mma_probe16(
        &self,
        a: &[u8],
        b: &[u8],
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<Vec<f32>, OxideError> {
        if a.len() < 256 || b.len() < 128 {
            return Err(shape_err(
                "fp8_mma_probe16",
                format!(
                    "fragments must be 256 A bytes and 128 B bytes, got {} and {}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, &a[..256])?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, &b[..128])?;
        let mut out = DeviceBuffer::<f32>::from_host(&self.stream, &vec![-999.0f32; 128])?;
        // SAFETY: a single warp; every lane's fragments are present and each
        // of the 128 output slots has exactly one owner.
        unsafe {
            self.gemm_f8.fp8_mma_probe16(
                &self.stream,
                launch::blocks_of(1, 32),
                &da,
                &db,
                &mut out,
                fmt_a.code(),
                fmt_b.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Raw-fragment `mma.m16n8k32` probe: one warp of fragments is 32 lanes x
    /// 16 A bytes and 32 lanes x 8 B bytes, and the result is
    /// `out[i * 32 + lane] = D[lane][i]` for a 16x8 f32 tile.  Used by
    /// `tests/oxide_gemm_f8.rs` to pin the documented fragment layout.
    pub fn fp8_mma_probe(
        &self,
        a: &[u8],
        b: &[u8],
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<Vec<f32>, OxideError> {
        if a.len() < 512 || b.len() < 256 {
            return Err(shape_err(
                "fp8_mma_probe",
                format!(
                    "fragments must be 512 A bytes and 256 B bytes, got {} and {}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, &a[..512])?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, &b[..256])?;
        // Filled with a sentinel rather than zeros so a lane that fails to
        // write is distinguishable from a genuine zero result.
        let mut out = DeviceBuffer::<f32>::from_host(&self.stream, &vec![-999.0f32; 128])?;
        // SAFETY: a single warp, `a`/`b` cover every lane's fragments, and
        // each of the 128 output slots has exactly one owner.
        unsafe {
            self.gemm_f8.fp8_mma_probe(
                &self.stream,
                launch::blocks_of(1, 32),
                &da,
                &db,
                &mut out,
                fmt_a.code(),
                fmt_b.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Validated FP8 GEMM launch (8-bit operands, f32 accumulator) with the
    /// same view, split-K and `beta` semantics as [`Self::gemm`].
    pub(crate) fn gemm_fp8_args(
        &self,
        a: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        scale_a: &DeviceBuffer<f32>,
        scale_b: &DeviceBuffer<f32>,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp8: {msg}"));
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
        // Element extents, in bytes for the 8-bit operands.
        let a_bytes = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_bytes = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        if a_bytes > a.len() || b_bytes > b.len() {
            return Err(bad("operand buffer too small for the requested view"));
        }
        if extent(g.c.0, g.c.1, g.c.2, g.count, g.m, g.n, g.ldc) > c.len() {
            return Err(bad("output buffer too small for the requested view"));
        }
        let (bm, bn) = (gemm_f8::F8_BM, gemm_f8::F8_BN);
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS
                .div_ceil(tiles)
                .min(g.k / SPLITK_MIN_K)
                .min(SPLITK_MAX)
                .max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 {
            g.k.div_ceil(splits).next_multiple_of(16)
        } else {
            g.k
        };
        let splits = g.k.div_ceil(kchunk);
        let mn = g.m * g.n;
        let mut ws = if splits > 1 {
            Some(self.alloc_uninit(splits * g.count * mn)?)
        } else {
            None
        };
        let (alpha, beta, ldc) = if splits > 1 {
            (1.0, 0.0, g.n)
        } else {
            (g.alpha, g.beta, g.ldc)
        };
        let (cb, cs, cm) = if splits > 1 {
            (0, mn, splits * g.count)
        } else {
            g.c
        };
        let target: &mut DeviceBuffer<f32> = match ws.as_mut() {
            Some(w) => w,
            None => c,
        };
        let blocks = tiles * splits;
        // SAFETY: extents validated above; each C element has one owner
        // thread and instances writing the same `c` slot are not allowed.
        macro_rules! launch_f8 {
            ($kernel:ident) => {
        unsafe {
            self.gemm_f8.$kernel(
                &self.stream,
                launch::blocks_of(blocks, gemm_f8::F8_THREADS),
                a,
                b,
                target,
                g.m,
                g.n,
                g.k,
                g.lda,
                g.ldb,
                ldc,
                g.trans_a as u32,
                g.trans_b as u32,
                alpha,
                beta,
                g.a.0,
                g.a.1,
                g.a.2,
                g.b.0,
                g.b.1,
                g.b.2,
                cb,
                cs,
                cm,
                g.count,
                kchunk,
                fmt_a.code(),
                fmt_b.code(),
                scale_a,
                scale_b,
            )
        }
            };
        }
        // The e4m3 x e4m3 kernel has no runtime format `match` in its mma loop.
        let both_e4m3 = fmt_a == Fp8Format::E4M3
            && fmt_b == Fp8Format::E4M3
            && !self.fp8_generic_kernel.load(Ordering::Relaxed);
        let launched = if both_e4m3 { launch_f8!(sgemm_fp8_mma_e4m3) } else { launch_f8!(sgemm_fp8_mma) };
        launched.map_err(launch_err)?;
        if let Some(w) = ws {
            // SAFETY: one thread per output element; `c` extent validated above.
            unsafe {
                self.gemm.gemm_splitk_reduce(
                    &self.stream,
                    launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                    &w,
                    c,
                    g.m,
                    g.n,
                    g.ldc,
                    g.count,
                    splits,
                    g.alpha,
                    g.beta,
                    g.c.0,
                    g.c.1,
                    g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
    }

    /// Dense `C = alpha * op(A) * op(B)` (no `beta`) on FP8 operands, host
    /// round trip; `a` is `m x k` (or `k x m` when `trans_a`), `b` is `k x n`
    /// (or `n x k` when `trans_b`).  Returns a fresh `m x n` f32 tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<OxideTensorF32, OxideError> {
        let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
        let (br, bc) = if trans_b { (n, k) } else { (k, n) };
        if a.len() < ar * ac || b.len() < br * bc {
            return Err(shape_err(
                "gemm_fp8",
                format!(
                    "operands too small: {} A bytes for {ar}x{ac}, {} B bytes for {br}x{bc}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, a)?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, b)?;
        let mut out = self.alloc_uninit(m * n)?;
        let unit = self.unit_scale()?;
        self.gemm_fp8_args(
            &da,
            &db,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: ac,
                ldb: bc,
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta: 0.0,
                ..GemmArgs::default()
            },
            fmt_a,
            fmt_b,
            &unit,
            &unit,
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// Benchmark helper: mean device milliseconds per dense FP8 NN GEMM
    /// (`alpha = 1`, `beta = 0`) over `iters` launches after one warm-up.
    pub fn bench_gemm_fp8(
        &self,
        a: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<f64, OxideError> {
        let mut c = self.alloc(m * n)?;
        let unit = self.unit_scale()?;
        let g = GemmArgs {
            m,
            n,
            k,
            lda: k,
            ldb: n,
            ldc: n,
            ..GemmArgs::default()
        };
        self.time_gpu_ms(iters, || {
            self.gemm_fp8_args(&*a, &*b, &mut c, &g, fmt_a, fmt_b, &unit, &unit)
        })
    }

    /// Like [`Self::bench_gemm_fp8`] but takes host operand slices and uploads
    /// them once, so the timed region is the `mma` loop alone (the bench table
    /// in `tests/oxide_gemm_bench.rs` reports the kernel, while
    /// `LIGHTNET_OXIDE_GEMM=fp8` additionally pays for the quantisation).
    pub fn bench_gemm_fp8_host(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<f64, OxideError> {
        let da = DeviceBuffer::<u8>::from_host(&self.stream, a)?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, b)?;
        self.bench_gemm_fp8(&da, &db, m, n, k, iters, fmt_a, fmt_b)
    }

    // ---- FP4 (NVFP4 emulation on the CUDA cores) -------------------------
    //
    // Native FP4 `mma` needs sm_120a; on Ada the recipe is emulated with
    // `__dp4a` over exact i32 block dots.  This is *emulation*, not a tensor
    // core path: it is slower than FP16 by design and exists for parity with
    // the `raptor-fp4` C++ reference, not for speed.

    /// Packed NVFP4 quantiser: `[rows, k]` row-major f32 -> E2M1 codes with one
    /// E4M3 scale per 16 elements along K, plus the f32 global scale.
    pub fn fp4_quantize_1x16(
        &self,
        x: &[f32],
        rows: usize,
        k: usize,
    ) -> Result<Fp4Packed, OxideError> {
        if rows == 0 || k == 0 {
            return Err(shape_err(
                "fp4_quantize_1x16",
                "rows and k must be > 0".to_string(),
            ));
        }
        if x.len() != rows * k {
            return Err(shape_err(
                "fp4_quantize_1x16",
                format!(
                    "expected {} elements for {rows}x{k}, got {}",
                    rows * k,
                    x.len()
                ),
            ));
        }
        let blocks = k.div_ceil(quant_fp4::BLOCK);
        let dx = DeviceBuffer::<f32>::from_host(&self.stream, x)?;
        let mut amax_slot = DeviceBuffer::<u32>::zeroed(&self.stream, 1)?;
        // SAFETY: one thread per element; the single atomic slot is zeroed
        // above and only read through `fetch_max`.
        unsafe {
            self.quant_fp4.fp4_amax_f32_kernel(
                &self.stream,
                launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                &dx,
                rows,
                k,
                &mut amax_slot,
            )
        }
        .map_err(launch_err)?;
        let amax = f32::from_bits(amax_slot.to_host_vec(&self.stream)?[0]);
        let amax_slot = DeviceBuffer::<f32>::from_host(&self.stream, &[amax])?;
        let mut scale = DeviceBuffer::<f32>::zeroed(&self.stream, 1)?;
        // SAFETY: a single thread writing slot 0 of a one-element buffer.
        unsafe {
            self.quant_fp4.fp4_finalize_global_scale_kernel(
                &self.stream,
                launch::blocks_of(1, 1),
                &amax_slot,
                &mut scale,
            )
        }
        .map_err(launch_err)?;
        // `scale` now holds `choose_global_scale(amax)` and is the single
        // device f32 the quantiser divides by (the recipe's global scale).
        let global_scale = scale.to_host_vec(&self.stream)?[0];
        let mut codes = DeviceBuffer::<u8>::zeroed(&self.stream, rows * blocks * 8)?;
        let mut scales = DeviceBuffer::<u8>::zeroed(&self.stream, rows * blocks)?;
        // SAFETY: one thread per 16-element block; `codes` and `scales` are
        // sized exactly as the kernel indexes them.
        unsafe {
            self.quant_fp4.fp4_quantize_1x16_kernel(
                &self.stream,
                launch::elementwise(rows * blocks, ElementwiseLaunchConfig::default()),
                &dx,
                rows,
                k,
                &scale,
                &mut codes,
                &mut scales,
            )
        }
        .map_err(launch_err)?;
        Ok(Fp4Packed {
            rows,
            k,
            codes: codes.to_host_vec(&self.stream)?,
            scales: scales.to_host_vec(&self.stream)?,
            global_scale,
        })
    }

    /// Dequantises a packed NVFP4 operand back to the `[rows, k]` f32 matrix.
    pub fn fp4_dequantize_1x16(&self, packed: &Fp4Packed) -> Result<Vec<f32>, OxideError> {
        let blocks = packed.blocks_per_row();
        let codes = DeviceBuffer::<u8>::from_host(&self.stream, &packed.codes)?;
        let scales = DeviceBuffer::<u8>::from_host(&self.stream, &packed.scales)?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, packed.rows * packed.k)?;
        // SAFETY: one thread per element; every code / scale index is derived
        // from `row`, `kk` and `blocks`, all bounded by the buffer sizes.
        unsafe {
            self.quant_fp4.fp4_dequantize_1x16_kernel(
                &self.stream,
                launch::elementwise(packed.rows * packed.k, ElementwiseLaunchConfig::default()),
                &codes,
                &scales,
                packed.rows,
                packed.k,
                packed.global_scale,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        let _ = blocks;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Packed NVFP4 GEMM, CUDA cores only (`__dp4a`).
    ///
    /// `C[M, N] = alpha * gA * gB * sum_blocks dot * 0.25 * sA * sB + beta * C`
    /// with both operands "NT" (contiguous along K, which is how
    /// [`Fp4Packed`] stores them).  `reference` picks the one-thread-per-output
    /// oracle kernel instead of the tiled one.  `c` (read only when
    /// `beta != 0`) is supplied by the host.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp4(
        &self,
        a: &Fp4Packed,
        b: &Fp4Packed,
        c: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        reference: bool,
    ) -> Result<Vec<f32>, OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp4: {msg}"));
        if m == 0 || n == 0 || k == 0 {
            return Err(bad("m, n and k must be > 0"));
        }
        if a.rows != m || b.rows != n || a.k != k || b.k != k {
            return Err(bad("packed operand shapes do not match m, n, k"));
        }
        if let Some(c) = c {
            if c.len() < m * n {
                return Err(bad("c is smaller than m * n"));
            }
        }
        let dca = DeviceBuffer::<u8>::from_host(&self.stream, &a.codes)?;
        let dsa = DeviceBuffer::<u8>::from_host(&self.stream, &a.scales)?;
        let dcb = DeviceBuffer::<u8>::from_host(&self.stream, &b.codes)?;
        let dsb = DeviceBuffer::<u8>::from_host(&self.stream, &b.scales)?;
        let ga = DeviceBuffer::<f32>::from_host(&self.stream, &[a.global_scale])?;
        let gb = DeviceBuffer::<f32>::from_host(&self.stream, &[b.global_scale])?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, m * n)?;
        if beta != 0.0 {
            if let Some(c) = c {
                out.copy_from_host(&self.stream, c)?;
            }
        }
        let (bm, bn) = (gemm_fp4::FP4_BM, gemm_fp4::FP4_BN);
        let blocks = m.div_ceil(bm) * n.div_ceil(bn);
        let grid = if reference {
            launch::elementwise(m * n, ElementwiseLaunchConfig::default())
        } else {
            launch::blocks_of(blocks, gemm_fp4::FP4_THREADS)
        };
        // SAFETY: the packed buffers are sized `rows * ceil(k/16) * 8` and
        // `rows * ceil(k/16)` as the kernels index them; `out` is `m * n` and
        // each element has exactly one owner thread.
        unsafe {
            if reference {
                self.gemm_fp4.fp4_gemm_nt_packed_reference(
                    &self.stream,
                    grid,
                    &dca,
                    &dsa,
                    &dcb,
                    &dsb,
                    &mut out,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    n,
                    &ga,
                    &gb,
                )
            } else {
                self.gemm_fp4.fp4_gemm_nt_packed(
                    &self.stream,
                    grid,
                    &dca,
                    &dsa,
                    &dcb,
                    &dsb,
                    &mut out,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    n,
                    &ga,
                    &gb,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Benchmark helper: mean device milliseconds per packed NVFP4 GEMM over
    /// `iters` launches after one warm-up.
    pub fn bench_gemm_fp4(
        &self,
        a: &Fp4Packed,
        b: &Fp4Packed,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<f64, OxideError> {
        let dca = DeviceBuffer::<u8>::from_host(&self.stream, &a.codes)?;
        let dsa = DeviceBuffer::<u8>::from_host(&self.stream, &a.scales)?;
        let dcb = DeviceBuffer::<u8>::from_host(&self.stream, &b.codes)?;
        let dsb = DeviceBuffer::<u8>::from_host(&self.stream, &b.scales)?;
        let ga = DeviceBuffer::<f32>::from_host(&self.stream, &[a.global_scale])?;
        let gb = DeviceBuffer::<f32>::from_host(&self.stream, &[b.global_scale])?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, m * n)?;
        let blocks = m.div_ceil(gemm_fp4::FP4_BM) * n.div_ceil(gemm_fp4::FP4_BN);
        let grid = launch::blocks_of(blocks, gemm_fp4::FP4_THREADS);
        // SAFETY: buffers are sized as the kernel indexes them and every
        // output element has exactly one owner thread.
        let mut run = || unsafe {
            self.gemm_fp4.fp4_gemm_nt_packed(
                &self.stream,
                grid,
                &dca,
                &dsa,
                &dcb,
                &dsb,
                &mut out,
                m,
                n,
                k,
                1.0f32,
                0.0f32,
                n,
                &ga,
                &gb,
            )
        };
        Ok(self.time_gpu_ms(iters, || run().map_err(launch_err))?)
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
        let g = GemmArgs {
            m,
            n,
            k,
            lda: k,
            ldb: n,
            ldc: n,
            ..GemmArgs::default()
        };
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
        let start = self
            .stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for _ in 0..iters {
            f()?;
        }
        let end = self
            .stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
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
        if g.len() != 4
            || g[1] != channels
            || in_h * in_w * channels == 0
            || x_len % (in_h * in_w * channels) != 0
            || g[0] != x_len / (in_h * in_w * channels)
        {
            return Err(shape_err(
                op,
                format!("gradient shape {g:?} does not match the input"),
            ));
        }
        Ok((g[2], g[3]))
    }

    /// Zero-filled output buffer (for kernels that accumulate or only write
    /// part of it).  If the driver is out of memory the pool is flushed once
    /// and the allocation retried, so a large cache can never starve a later
    /// (larger) tensor.
    pub(crate) fn alloc(&self, len: usize) -> Result<PooledBuf, OxideError> {
        match self.bufs.take(len) {
            Some(mut b) => {
                b.zero_async(&self.stream)?;
                Ok(PooledBuf::new(b, &self.bufs))
            }
            None => self.alloc_zeroed(len),
        }
    }

    /// Fresh `len`-element zeroed allocation; flush the pool and retry once on
    /// OOM (the pool may be holding cached buffers that are not in use).
    fn alloc_zeroed(&self, len: usize) -> Result<PooledBuf, OxideError> {
        match DeviceBuffer::<f32>::zeroed(&self.stream, len) {
            Ok(b) => Ok(PooledBuf::new(b, &self.bufs)),
            Err(err) => {
                self.bufs.flush();
                match DeviceBuffer::<f32>::zeroed(&self.stream, len) {
                    Ok(b) => Ok(PooledBuf::new(b, &self.bufs)),
                    Err(_) => Err(OxideError::from(err)),
                }
            }
        }
    }

    /// Byte buffer with unspecified contents (fp8 operands); same rule as
    /// [`Self::alloc_uninit`].  Flushes the pool and retries once on OOM.
    fn alloc8_uninit(&self, len: usize) -> Result<PooledBuf8, OxideError> {
        if let Some(b) = self.bufs.take8(len) {
            return Ok(PooledBuf8::new(b, &self.bufs));
        }
        match DeviceBuffer::<u8>::zeroed(&self.stream, len) {
            Ok(b) => Ok(PooledBuf8::new(b, &self.bufs)),
            Err(err) => {
                self.bufs.flush();
                match DeviceBuffer::<u8>::zeroed(&self.stream, len) {
                    Ok(b) => Ok(PooledBuf8::new(b, &self.bufs)),
                    Err(_) => Err(OxideError::from(err)),
                }
            }
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
        let d = lightnet_cuda::host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .map_err(|msg| shape_err(op, msg))?;
        let (src_len, dst_len) = if to_small {
            (d.big_len(), d.small_len())
        } else {
            (d.small_len(), d.big_len())
        };
        if src.len() != src_len || src_shape.iter().product::<usize>() != src_len {
            return Err(shape_err(
                op,
                format!("input has {} elements, expected {src_len}", src.len()),
            ));
        }
        let mut out = self.alloc_uninit(dst_len)?;
        // SAFETY: one thread per output element; source indices are in range
        // by the `reorg_dims` check (`hs * stride <= hb`, `ws * stride <= wb`).
        unsafe {
            if to_small {
                self.reorg.reorg_to_small(
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
                self.reorg.reorg_to_big(
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
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {len} elements"),
            ));
        }
        let channels = shape[1];
        if per_channel.iter().any(|t| t.len() != channels) {
            return Err(shape_err(
                op,
                format!("per-channel tensors must have {channels} elements"),
            ));
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
            return Err(shape_err(
                op,
                format!("shape {shape:?} is not a non-empty [batch, c, h, w]"),
            ));
        }
        let (batch, channels, hw) = (shape[0], shape[1], shape[2] * shape[3]);
        let scale_len = if scale_wh {
            batch * hw
        } else {
            batch * channels
        };
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
            return Err(shape_err(
                op,
                "batch, inputs and outputs must be > 0".to_owned(),
            ));
        }
        if x_len != batch * inputs || w_len != outputs * inputs {
            return Err(shape_err(
                op,
                format!(
                    "x has {x_len} (want {}), weights {w_len} (want {})",
                    batch * inputs,
                    outputs * inputs
                ),
            ));
        }
        Ok(())
    }
}

/// Reject a row count / width the LayerNorm kernels cannot serve before
/// launching, instead of reading out of bounds.
fn check_layernorm_shape(
    op: &str,
    shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), OxideError> {
    let total: usize = shape.iter().product();
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(total) {
        return Err(OxideError::Shape(format!(
            "{op}: rows*cols must equal the tensor length ({rows}*{cols} vs {total})"
        )));
    }
    if cols > kernels::activations::layernorm::LAYERNORM_MAX_COLS {
        return Err(OxideError::Shape(format!(
            "{op}: cols={cols} exceeds the kernel limit {}",
            kernels::activations::layernorm::LAYERNORM_MAX_COLS
        )));
    }
    Ok(())
}

pub(crate) fn launch_err(err: impl fmt::Debug) -> OxideError {
    OxideError::Shape(format!("kernel launch failed: {err:?}"))
}

/// Shared operand check for the deformable gather pair. `shape_err` keeps the
/// message format used by the other Oxide overrides.
fn deform_gather_shape(
    op: &str,
    input: &OxideTensorF32,
    sample_y: &OxideTensorF32,
    sample_x: &OxideTensorF32,
    gradient: Option<&OxideTensorF32>,
    geometry: BilinearGatherGeometry,
) -> Result<(), OxideError> {
    geometry
        .checked_products()
        .map_err(|msg| shape_err(op, msg))?;
    if input.len() != geometry.input_len() {
        return Err(shape_err(
            op,
            format!(
                "input has {} elements, expected {}",
                input.len(),
                geometry.input_len()
            ),
        ));
    }
    for (name, tensor) in [("sample_y", sample_y), ("sample_x", sample_x)] {
        if tensor.len() != geometry.sample_len() {
            return Err(shape_err(
                op,
                format!(
                    "{name} has {} elements, expected {}",
                    tensor.len(),
                    geometry.sample_len()
                ),
            ));
        }
    }
    if let Some(gradient) = gradient
        && gradient.len() != geometry.output_len()
    {
        return Err(shape_err(
            op,
            format!(
                "output gradient has {} elements, expected {}",
                gradient.len(),
                geometry.output_len()
            ),
        ));
    }
    Ok(())
}

impl OxideBackend {
    /// Device `[deform_conv]` **accumulation stage** over prepared offsets and
    /// masks. Per `(b, filter, oy, ox)` it sums
    /// `weight[f, c, k] * bilinear_sample(input, base + offset) * mask` over the
    /// group's channels and kernel points, then applies the source `sanitize`
    /// clamp.
    ///
    /// Deliberately not wired into the runtime yet: the predictor columns,
    /// per-batch conditional offset normalization and the lhtan/sigmoid
    /// envelopes are the caller's responsibility, and the CPU forward/backward
    /// remains the reference. `layer-support.md` keeps `[deform_conv]` at
    /// `host` / `cpu` until the full forward is assembled and tested.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_accumulate(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: &OxideTensorF32,
        weights: &OxideTensorF32,
        channels: usize,
        filters: usize,
        channels_per_group: usize,
        filters_per_group: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_conv_accumulate";
        if channels == 0
            || filters == 0
            || in_h == 0
            || in_w == 0
            || out_h == 0
            || out_w == 0
            || size == 0
            || channels_per_group == 0
            || filters_per_group == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if channels % channels_per_group != 0 || filters % filters_per_group != 0 {
            return Err(shape_err(
                op,
                "groups must divide both input channels and filters".to_owned(),
            ));
        }
        let kernels = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let plane = out_h * out_w;
        let in_plane = in_h * in_w;
        let per_batch = channels
            .checked_mul(in_plane)
            .ok_or_else(|| shape_err(op, "input channel count overflows".to_owned()))?;
        if input.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, not a multiple of channels*H*W",
                    input.len()
                ),
            ));
        }
        let batch = input.len() / per_batch;
        let offset_len = batch * 2 * kernels * plane;
        if offsets.len() != offset_len {
            return Err(shape_err(
                op,
                format!(
                    "offsets have {} elements, expected {offset_len}",
                    offsets.len()
                ),
            ));
        }
        if use_mask && masks.len() != batch * kernels * plane {
            return Err(shape_err(
                op,
                format!(
                    "masks have {} elements, expected {}",
                    masks.len(),
                    batch * kernels * plane
                ),
            ));
        }
        let weight_len = filters
            .checked_mul(channels_per_group)
            .and_then(|v| v.checked_mul(kernels))
            .ok_or_else(|| shape_err(op, "weight count overflows".to_owned()))?;
        if weights.len() != weight_len {
            return Err(shape_err(
                op,
                format!(
                    "weights have {} elements, expected {weight_len}",
                    weights.len()
                ),
            ));
        }
        let out_len = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(plane))
            .ok_or_else(|| shape_err(op, "output count overflows".to_owned()))?;
        let mut out = self.alloc_uninit(out_len)?;
        // SAFETY: one thread per output element; every input/offset/mask/weight
        // index is inside a buffer validated above and inside the kernel's
        // own in-image checks.
        unsafe {
            self.deform_sample.deform_conv_accumulate(
                &self.stream,
                launch::elementwise(out_len, config),
                &input.buf,
                &offsets.buf,
                &masks.buf,
                &weights.buf,
                &mut out,
                channels,
                filters,
                channels_per_group,
                filters_per_group,
                in_h,
                in_w,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding,
                dilation,
                u32::from(use_mask),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[batch, filters, out_h, out_w]))
    }

    /// Build one single-template DSConv morph branch's coordinate maps from
    /// tanh-transformed `[B,2*K,H,W]` offsets. `morph=2` composes two calls.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_coordinate_map_f32(
        &self,
        offsets: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_coordinate_map_f32";
        if batch == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 1
        {
            return Err(shape_err(
                op,
                "invalid batch/spatial/kernel/morph geometry".to_owned(),
            ));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        if offsets.shape() != [batch, offset_channels, height, width] {
            return Err(shape_err(
                op,
                "offset tensor shape does not match geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "coordinate map length overflows".to_owned()))?;
        let offsets = self.f32v(offsets)?;
        let mut y_map = self.alloc_uninit(output_len)?;
        let mut x_map = self.alloc_uninit(output_len)?;
        // SAFETY: operand shape and output lengths are validated above; each
        // thread owns one y/x coordinate pair and all offset taps stay in bounds.
        unsafe {
            self.dsconv.dsconv_coordinate_map(
                &self.stream,
                launch::elementwise(output_len, config),
                &offsets.buf,
                &mut y_map,
                &mut x_map,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                morph,
            )
        }
        .map_err(launch_err)?;
        let shape = [batch, height, width, kernel_size];
        Ok((
            OxideTensorF32::new(y_map, &shape),
            OxideTensorF32::new(x_map, &shape),
        ))
    }

    /// Border-clamped bilinear sampling for the DSConv's deformed tensor,
    /// returning `[B,C,H,W,K]` in the CPU reference's axis order.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_bilinear_sample_f32(
        &self,
        input: &OxideTensorF32,
        y_map: &OxideTensorF32,
        x_map: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_bilinear_sample_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || kernel_size == 0 {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        let coord_shape = [batch, height, width, kernel_size];
        if input.shape() != [batch, channels, height, width]
            || y_map.shape() != coord_shape
            || x_map.shape() != coord_shape
        {
            return Err(shape_err(
                op,
                "input and coordinate tensor shapes do not match geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "sampled tensor length overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let y_map = self.f32v(y_map)?;
        let x_map = self.f32v(x_map)?;
        let mut sampled = self.alloc_uninit(output_len)?;
        // SAFETY: exact source/map shapes and output extent were checked above.
        unsafe {
            self.dsconv.dsconv_bilinear_sample(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &y_map.buf,
                &x_map.buf,
                &mut sampled,
                batch,
                channels,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        let shape = [batch, channels, height, width, kernel_size];
        Ok(OxideTensorF32::new(sampled, &shape))
    }

    /// Adjoint of DSConv's border-clamped bilinear gather. Input gradients use
    /// float atomics; coordinate gradients are uniquely owned per sample.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_bilinear_backward_f32(
        &self,
        input: &OxideTensorF32,
        y_map: &OxideTensorF32,
        x_map: &OxideTensorF32,
        sampled_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_bilinear_backward_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || kernel_size == 0 {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        let coord_shape = [batch, height, width, kernel_size];
        let sampled_shape = [batch, channels, height, width, kernel_size];
        if input.shape() != [batch, channels, height, width]
            || y_map.shape() != coord_shape
            || x_map.shape() != coord_shape
            || sampled_gradient.shape() != sampled_shape
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input gradient length overflows".to_owned()))?;
        let coord_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "coordinate gradient length overflows".to_owned()))?;
        let input = self.f32v(input)?;
        let y_map = self.f32v(y_map)?;
        let x_map = self.f32v(x_map)?;
        let sampled_gradient = self.f32v(sampled_gradient)?;
        let mut input_gradient = self.alloc_zeroed(input_len)?;
        let mut y_gradient = self.alloc_uninit(coord_len)?;
        let mut x_gradient = self.alloc_uninit(coord_len)?;
        // SAFETY: all tensor shapes and gradient extents are checked; output
        // coordinate writers are unique and input collisions use float atomics.
        unsafe {
            self.dsconv.dsconv_bilinear_backward(
                &self.stream,
                launch::elementwise(coord_len, config),
                &input.buf,
                &y_map.buf,
                &x_map.buf,
                &sampled_gradient.buf,
                &mut input_gradient,
                &mut y_gradient,
                &mut x_gradient,
                batch,
                channels,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_gradient, &[batch, channels, height, width]),
            OxideTensorF32::new(y_gradient, &coord_shape),
            OxideTensorF32::new(x_gradient, &coord_shape),
        ))
    }

    /// Backpropagate coordinate-map adjoints through the center-anchored
    /// cumulative offsets and tanh, preserving the configured morph axes.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_offset_map_backward_f32(
        &self,
        offsets_tanh: &OxideTensorF32,
        y_coordinate_gradient: &OxideTensorF32,
        x_coordinate_gradient: &OxideTensorF32,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_offset_map_backward_f32";
        if batch == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 2
        {
            return Err(shape_err(op, "invalid DSConv offset geometry".to_owned()));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        let coord_shape = [batch, height, width, kernel_size];
        if offsets_tanh.shape() != [batch, offset_channels, height, width]
            || y_coordinate_gradient.shape() != coord_shape
            || x_coordinate_gradient.shape() != coord_shape
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let offset_len = batch
            .checked_mul(offset_channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "offset gradient length overflows".to_owned()))?;
        let offsets_tanh = self.f32v(offsets_tanh)?;
        let y_coordinate_gradient = self.f32v(y_coordinate_gradient)?;
        let x_coordinate_gradient = self.f32v(x_coordinate_gradient)?;
        let mut offset_gradient = self.alloc_uninit(offset_len)?;
        // SAFETY: input and output layouts are validated above; one thread owns
        // each raw-offset gradient and all cumulative taps are within K.
        unsafe {
            self.dsconv.dsconv_offset_map_backward(
                &self.stream,
                launch::elementwise(offset_len, config),
                &offsets_tanh.buf,
                &y_coordinate_gradient.buf,
                &x_coordinate_gradient.buf,
                &mut offset_gradient,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                morph,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            offset_gradient,
            &[batch, offset_channels, height, width],
        ))
    }

    /// Collapse DSConv's `[B,C,H,W,K]` sampled tensor with `[F,C,K]` weights.
    pub fn dsconv_collapse_f32(
        &self,
        sampled: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_collapse_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if sampled.shape() != [batch, channels, height, width, kernel_size]
            || weights.shape() != [filters, channels, kernel_size]
            || biases.shape() != [filters]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let output_len = batch
            .checked_mul(filters)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
        let sampled = self.f32v(sampled)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: operand shapes and output length were checked above.
        unsafe {
            self.dsconv.dsconv_collapse(
                &self.stream,
                launch::elementwise(output_len, config),
                &sampled.buf,
                &weights.buf,
                &biases.buf,
                &mut output,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            output,
            &[batch, filters, height, width],
        ))
    }

    /// Exact adjoints of the linear DSConv collapse for one output-channel
    /// partition (used separately for the two branches of morph=2).
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_collapse_backward_f32(
        &self,
        sampled: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        filter_start: usize,
        filter_count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dsconv_collapse_backward_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if filter_count == 0 || filter_start > filters || filter_count > filters - filter_start {
            return Err(shape_err(op, "invalid output-channel partition".to_owned()));
        }
        if sampled.shape() != [batch, channels, height, width, kernel_size]
            || weights.shape() != [filters, channels, kernel_size]
            || output_gradient.shape() != [batch, filters, height, width]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let sampled_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "sampled gradient length overflows".to_owned()))?;
        let weight_len = filters
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(kernel_size))
            .ok_or_else(|| shape_err(op, "weight gradient length overflows".to_owned()))?;
        let sampled = self.f32v(sampled)?;
        let weights = self.f32v(weights)?;
        let output_gradient = self.f32v(output_gradient)?;
        let mut sampled_gradient = self.alloc_uninit(sampled_len)?;
        let mut weight_gradient = self.alloc_uninit(weight_len)?;
        let mut bias_gradient = self.alloc_uninit(filters)?;
        // SAFETY: all operands were shape-checked; every output element has one writer.
        unsafe {
            self.dsconv.dsconv_collapse_backward_sampled(
                &self.stream,
                launch::elementwise(sampled_len, config),
                &weights.buf,
                &output_gradient.buf,
                &mut sampled_gradient,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: shape-checked parameter/output gradients; unique parameter writers.
        unsafe {
            self.dsconv.dsconv_collapse_backward_weights(
                &self.stream,
                launch::elementwise(weight_len, config),
                &sampled.buf,
                &output_gradient.buf,
                &mut weight_gradient,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: one thread owns each bias-gradient channel.
        unsafe {
            self.dsconv.dsconv_collapse_backward_biases(
                &self.stream,
                launch::elementwise(filters, config),
                &output_gradient.buf,
                &mut bias_gradient,
                batch,
                filters,
                height,
                width,
                filter_start,
                filter_count,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(
                sampled_gradient,
                &[batch, channels, height, width, kernel_size],
            ),
            OxideTensorF32::new(weight_gradient, &[filters, channels, kernel_size]),
            OxideTensorF32::new(bias_gradient, &[filters]),
        ))
    }

    /// Composed DSConv f32 forward for the bounded single-template contract.
    /// This returns the post-activation layer output but does not retain the
    /// forward intermediates needed by runtime backward.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_forward_native_f32(
        &self,
        input: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dsconv_forward_native_f32";
        if batch == 0
            || channels == 0
            || filters == 0
            || height == 0
            || width == 0
            || kernel_size == 0
            || kernel_size % 2 == 0
            || !extend_scope.is_finite()
            || morph > 2
            || (morph == 2 && filters < 2)
        {
            return Err(shape_err(
                op,
                "invalid batch, layer, or morph geometry".to_owned(),
            ));
        }
        let offset_channels = kernel_size
            .checked_mul(2)
            .ok_or_else(|| shape_err(op, "offset channel count overflows".to_owned()))?;
        if input.shape() != [batch, channels, height, width]
            || offset_weights.shape() != [offset_channels, channels, 3, 3]
            || offset_biases.shape() != [offset_channels]
            || weights.shape() != [filters, channels, kernel_size]
            || biases.shape() != [filters]
        {
            return Err(shape_err(
                op,
                "operand shapes do not match DSConv geometry".to_owned(),
            ));
        }
        let offset_shape = [batch, offset_channels, height, width];
        let raw_offsets = self.conv2d_f32(
            input,
            offset_weights,
            &offset_shape,
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let biased_offsets = self.bias_add_f32(
            &raw_offsets,
            offset_biases,
            offset_channels,
            height * width,
            config,
        )?;
        let offsets_tanh = self.activation_f32(&biased_offsets, F32ActivationKind::Tanh, config)?;
        let (x_y_map, x_x_map) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            0,
            config,
        )?;
        let (y_y_map, y_x_map) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            1,
            config,
        )?;
        let sampled_x = self.dsconv_bilinear_sample_f32(
            input,
            &x_y_map,
            &x_x_map,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let sampled_y = self.dsconv_bilinear_sample_f32(
            input,
            &y_y_map,
            &y_x_map,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let output_len = batch
            .checked_mul(filters)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
        let sampled_x = self.f32v(&sampled_x)?;
        let sampled_y = self.f32v(&sampled_y)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut raw_output = self.alloc_uninit(output_len)?;
        // SAFETY: all input/parameter extents and output geometry were checked.
        unsafe {
            self.dsconv.dsconv_collapse_morph(
                &self.stream,
                launch::elementwise(output_len, config),
                &sampled_x.buf,
                &sampled_y.buf,
                &weights.buf,
                &biases.buf,
                &mut raw_output,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                morph,
            )
        }
        .map_err(launch_err)?;
        let raw_output = OxideTensorF32::new(raw_output, &[batch, filters, height, width]);
        self.activation_f32(&raw_output, activation, config)
    }

    /// Composed exact adjoint of the bounded single-template DSConv forward.
    /// Intermediate device values are recomputed; input-sampling scatter uses
    /// atomics, so only summation order is nondeterministic.
    #[allow(clippy::too_many_arguments)]
    pub fn dsconv_backward_native_f32(
        &self,
        input: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "dsconv_backward_native_f32";
        if output_gradient.shape() != [batch, filters, height, width] {
            return Err(shape_err(
                op,
                "output-gradient shape does not match DSConv output".to_owned(),
            ));
        }
        let raw_output = self.dsconv_forward_native_f32(
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            batch,
            channels,
            filters,
            height,
            width,
            kernel_size,
            extend_scope,
            morph,
            F32ActivationKind::Linear,
            config,
        )?;
        let output_delta =
            self.activation_backward_f32(&raw_output, output_gradient, activation, config)?;

        // Rebuild predictor offsets and both branch sampling caches.
        let offset_channels = 2 * kernel_size;
        let offset_shape = [batch, offset_channels, height, width];
        let raw_offsets = self.conv2d_f32(
            input,
            offset_weights,
            &offset_shape,
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let biased_offsets = self.bias_add_f32(
            &raw_offsets,
            offset_biases,
            offset_channels,
            height * width,
            config,
        )?;
        let offsets_tanh = self.activation_f32(&biased_offsets, F32ActivationKind::Tanh, config)?;
        let (x_y, x_x) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            0,
            config,
        )?;
        let (y_y, y_x) = self.dsconv_coordinate_map_f32(
            &offsets_tanh,
            batch,
            height,
            width,
            kernel_size,
            extend_scope,
            1,
            config,
        )?;
        let sampled_x = self.dsconv_bilinear_sample_f32(
            input,
            &x_y,
            &x_x,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;
        let sampled_y = self.dsconv_bilinear_sample_f32(
            input,
            &y_y,
            &y_x,
            batch,
            channels,
            height,
            width,
            kernel_size,
            config,
        )?;

        let (x_start, x_count, y_start, y_count) = match morph {
            0 => (0, filters, 0, 0),
            1 => (0, 0, 0, filters),
            2 => (0, filters / 2, filters / 2, filters - filters / 2),
            _ => return Err(shape_err(op, "morph must be 0, 1, or 2".to_owned())),
        };
        let mut grad_weights: Option<OxideTensorF32> = None;
        let mut grad_biases: Option<OxideTensorF32> = None;
        let mut grad_input_sampling: Option<OxideTensorF32> = None;
        let mut grad_offset_map: Option<OxideTensorF32> = None;
        for (start, count, sampled, y_map, x_map, branch_morph) in [
            (x_start, x_count, &sampled_x, &x_y, &x_x, 0usize),
            (y_start, y_count, &sampled_y, &y_y, &y_x, 1usize),
        ] {
            if count == 0 {
                continue;
            }
            let (sampled_delta, weight_delta, bias_delta) = self.dsconv_collapse_backward_f32(
                sampled,
                weights,
                &output_delta,
                batch,
                channels,
                filters,
                height,
                width,
                kernel_size,
                start,
                count,
                config,
            )?;
            grad_weights = Some(match grad_weights {
                Some(ref existing) => self.vector_add_f32(existing, &weight_delta, config)?,
                None => weight_delta,
            });
            grad_biases = Some(match grad_biases {
                Some(ref existing) => self.vector_add_f32(existing, &bias_delta, config)?,
                None => bias_delta,
            });
            let (input_delta, y_coordinate_delta, x_coordinate_delta) = self
                .dsconv_bilinear_backward_f32(
                    input,
                    y_map,
                    x_map,
                    &sampled_delta,
                    batch,
                    channels,
                    height,
                    width,
                    kernel_size,
                    config,
                )?;
            grad_input_sampling = Some(match grad_input_sampling {
                Some(ref existing) => self.vector_add_f32(existing, &input_delta, config)?,
                None => input_delta,
            });
            let offset_delta = self.dsconv_offset_map_backward_f32(
                &offsets_tanh,
                &y_coordinate_delta,
                &x_coordinate_delta,
                batch,
                height,
                width,
                kernel_size,
                extend_scope,
                branch_morph,
                config,
            )?;
            grad_offset_map = Some(match grad_offset_map {
                Some(ref existing) => self.vector_add_f32(existing, &offset_delta, config)?,
                None => offset_delta,
            });
        }
        let grad_weights =
            grad_weights.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_biases =
            grad_biases.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_input_sampling = grad_input_sampling
            .ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_offset_map =
            grad_offset_map.ok_or_else(|| shape_err(op, "no active morph branch".to_owned()))?;
        let grad_offset_weights = self.conv2d_backward_weights_f32(
            input,
            &grad_offset_map,
            &[offset_channels, channels, 3, 3],
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let grad_offset_biases =
            self.bias_backward_f32(&grad_offset_map, offset_channels, height * width, config)?;
        let grad_input_predictor = self.conv2d_backward_input_f32(
            &grad_offset_map,
            offset_weights,
            &[batch, channels, height, width],
            channels,
            height,
            width,
            offset_channels,
            3,
            1,
            1,
            1,
            1,
            1,
            config,
        )?;
        let grad_input =
            self.vector_add_f32(&grad_input_sampling, &grad_input_predictor, config)?;
        Ok((
            grad_input,
            grad_weights,
            grad_biases,
            grad_offset_weights,
            grad_offset_biases,
        ))
    }

    /// `[deform_conv]` offset/mask predictor: a plain convolution of the input
    /// with the predictor weights plus its bias — i.e. `W * im2col(x) + b`, the
    /// same quantity `deform_conv::predictor_columns` produces. `groups` is not
    /// a parameter: the source predictor is always dense over all input
    /// channels (`offset_filters = 2*size*size`, `mask_filters = size*size`).
    ///
    /// Dilation greater than one uses an explicitly-shaped native predictor
    /// kernel because DeformConv's output extent differs from `[convolutional]`.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_offset_predict(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        input_channels: usize,
        in_h: usize,
        in_w: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_offset_predict";
        if input_channels == 0
            || in_h == 0
            || in_w == 0
            || filters == 0
            || size == 0
            || out_h == 0
            || out_w == 0
            || stride_x == 0
            || stride_y == 0
            || dilation == 0
        {
            return Err(shape_err(
                op,
                "all dimensions, strides and dilation must be positive".to_owned(),
            ));
        }
        let per_batch = input_channels
            .checked_mul(in_h)
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        if input.len() == 0 || input.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, not a multiple of C*H*W = {per_batch}",
                    input.len()
                ),
            ));
        }
        let batch = input.len() / per_batch;
        if input.shape() != [batch, input_channels, in_h, in_w] {
            return Err(shape_err(
                op,
                format!(
                    "input tensor shape {:?} does not match predictor geometry",
                    input.shape()
                ),
            ));
        }
        let weight_len = filters
            .checked_mul(input_channels)
            .and_then(|v| v.checked_mul(size))
            .and_then(|v| v.checked_mul(size))
            .ok_or_else(|| shape_err(op, "weight count overflows".to_owned()))?;
        if weights.len() != weight_len {
            return Err(shape_err(
                op,
                format!(
                    "weights have {} elements, expected {weight_len}",
                    weights.len()
                ),
            ));
        }
        if biases.len() != filters {
            return Err(shape_err(
                op,
                format!("biases have {} elements, expected {filters}", biases.len()),
            ));
        }
        let shape = [batch, filters, out_h, out_w];
        let convolved = if dilation == 1 {
            self.conv2d_f32(
                input,
                weights,
                &shape,
                input_channels,
                in_h,
                in_w,
                filters,
                size,
                stride_x,
                stride_y,
                dilation,
                padding,
                1,
                config,
            )?
        } else {
            let output_len = batch
                .checked_mul(filters)
                .and_then(|n| n.checked_mul(out_h))
                .and_then(|n| n.checked_mul(out_w))
                .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
            let input = self.f32v(input)?;
            let weights = self.f32v(weights)?;
            let mut output = self.alloc_uninit(output_len)?;
            // SAFETY: checked operand lengths and explicit output shape bound all taps.
            unsafe {
                self.deform_backward.deform_predictor_forward_dilated(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &input.buf,
                    &weights.buf,
                    &mut output,
                    batch,
                    input_channels,
                    filters,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    size,
                    stride_x,
                    stride_y,
                    padding,
                    dilation,
                )
            }
            .map_err(launch_err)?;
            OxideTensorF32::new(output, &shape)
        };
        self.bias_add_f32(&convolved, biases, filters, out_h * out_w, config)
    }

    /// Elementwise source `sanitize`.
    pub fn deform_sanitize_f32(
        &self,
        raw: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if raw.len() == 0 {
            return Err(shape_err(
                "deform_sanitize_f32",
                "input must not be empty".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as the validated input.
        unsafe {
            self.deform_sample.deform_sanitize_tensor(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Full native `[deform_conv]` **forward** (raw output, before the layer's
    /// own bias/BN/activation epilogue), composed from the tested stages:
    /// sanitize input -> offset predictor -> per-batch stats -> envelope ->
    /// (optional mask predictor + sigmoid) -> deformable accumulation.
    ///
    /// Dilated predictors use an explicit-extent direct CUDA kernel; the same
    /// stages are dispatched by the generic runtime for non-BN variants.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_forward_native(
        &self,
        input: &OxideTensorF32,
        main_weights: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        masks: Option<(&OxideTensorF32, &OxideTensorF32)>,
        channels: usize,
        filters: usize,
        groups: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_conv_forward_native";
        if groups == 0 || channels % groups != 0 || filters % groups != 0 {
            return Err(shape_err(
                op,
                "groups must divide both input channels and filters".to_owned(),
            ));
        }
        let kernel_area = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let plane = out_h * out_w;
        let max_offset = size as f32 * dilation as f32 * 2.0;
        let sanitized = self.deform_sanitize_f32(input, config)?;
        let raw_offsets = self.deform_offset_predict(
            &sanitized,
            offset_weights,
            offset_biases,
            channels,
            in_h,
            in_w,
            2 * kernel_area,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            out_h,
            out_w,
            config,
        )?;
        let (mean, inv_std, normalize) =
            self.deform_offset_stats(&raw_offsets, 2 * kernel_area * plane, max_offset, config)?;
        let offsets = self.deform_offset_envelope(
            &raw_offsets,
            &mean,
            &inv_std,
            &normalize,
            2 * kernel_area,
            plane,
            max_offset,
            config,
        )?;
        let mask_probabilities = match masks {
            Some((weights, biases)) => Some(self.deform_mask_sigmoid(
                &self.deform_offset_predict(
                    &sanitized,
                    weights,
                    biases,
                    channels,
                    in_h,
                    in_w,
                    kernel_area,
                    size,
                    stride_x,
                    stride_y,
                    padding,
                    dilation,
                    out_h,
                    out_w,
                    config,
                )?,
                config,
            )?),
            None => None,
        };
        // The accumulation kernel ignores `masks` when `use_mask == 0`; a
        // one-element buffer keeps the operand list uniform without an empty
        // allocation.
        let empty = self.upload_f32(&[1], &[0.0])?;
        self.deform_conv_accumulate(
            &sanitized,
            &offsets,
            mask_probabilities.as_ref().unwrap_or(&empty),
            main_weights,
            channels,
            filters,
            channels / groups,
            filters / groups,
            in_h,
            in_w,
            out_h,
            out_w,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            mask_probabilities.is_some(),
            config,
        )
    }

    /// Adjoint of [`Self::deform_conv_accumulate`]: returns
    /// `(d_input, d_offsets, d_masks, d_weights)`. Atomic scatter targets, so
    /// results agree with the host reference to accumulation-order tolerance.
    /// The input `sanitize` slope is **not** applied here; the caller does it
    /// once on the summed input gradient.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_accumulate_backward(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "deform_conv_accumulate_backward";
        let kernels = geometry.kernels();
        let plane = geometry.plane();
        let input_len =
            geometry.batch * geometry.channels * geometry.input_height * geometry.input_width;
        let offset_len = geometry.batch * 2 * kernels * plane;
        let mask_len = geometry.batch * kernels * plane;
        let weight_len = geometry.filters * geometry.channels_per_group * kernels;
        if input.len() != input_len
            || offsets.len() != offset_len
            || weights.len() != weight_len
            || output_grad.len() != geometry.batch * geometry.filters * plane
            || (use_mask && masks.len() != mask_len)
        {
            return Err(shape_err(
                op,
                "operand length does not match the geometry".to_owned(),
            ));
        }
        let d_input = self.alloc(input_len)?;
        let d_offsets = self.alloc(offset_len)?;
        let d_masks = self.alloc(if use_mask { mask_len } else { 1 })?;
        let d_weights = self.alloc(weight_len)?;
        // SAFETY: one thread per output element; all four targets are zeroed
        // buffers of the validated extents and are written only by atomic adds.
        unsafe {
            self.deform_backward.deform_conv_accumulate_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &input.buf,
                &offsets.buf,
                &masks.buf,
                &weights.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_offsets.cu_deviceptr() as *mut f32,
                d_masks.cu_deviceptr() as *mut f32,
                d_weights.cu_deviceptr() as *mut f32,
                geometry.channels,
                geometry.filters,
                geometry.channels_per_group,
                geometry.filters_per_group,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(use_mask),
            )
        }
        .map_err(launch_err)?;
        let mask_shape = [if use_mask { mask_len } else { 0 }];
        Ok((
            OxideTensorF32::new(
                d_input,
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
            ),
            OxideTensorF32::new(
                d_offsets,
                &[
                    geometry.batch,
                    2 * kernels,
                    geometry.output_height,
                    geometry.output_width,
                ],
            ),
            OxideTensorF32::new(d_masks, &mask_shape),
            OxideTensorF32::new(d_weights, &[weight_len]),
        ))
    }

    /// Adjoint of [`Self::deform_offset_predict`]: returns
    /// `(d_input, d_weights, d_biases)`. Atomic scatter; the input `sanitize`
    /// slope is the caller's job.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_predictor_backward(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        channels: usize,
        filters: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "deform_predictor_backward";
        let kernel_area = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let column_count = channels * kernel_area;
        let plane = out_h * out_w;
        let weight_len = filters * column_count;
        if input.len() != batch * channels * in_h * in_w
            || weights.len() != weight_len
            || output_grad.len() != batch * filters * plane
        {
            return Err(shape_err(
                op,
                "operand length does not match the geometry".to_owned(),
            ));
        }
        let input_len = batch * channels * in_h * in_w;
        let d_input = self.alloc(input_len)?;
        let d_weights = self.alloc(weight_len)?;
        let d_biases = self.alloc(filters)?;
        // SAFETY: one thread per column element; all three targets are zeroed
        // buffers of the validated extents and are written only by atomic adds.
        unsafe {
            self.deform_backward.deform_predictor_backward(
                &self.stream,
                launch::elementwise(batch * column_count * plane, config),
                &input.buf,
                &weights.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_weights.cu_deviceptr() as *mut f32,
                d_biases.cu_deviceptr() as *mut f32,
                channels,
                filters,
                in_h,
                in_w,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding,
                dilation,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(d_input, &[batch, channels, in_h, in_w]),
            OxideTensorF32::new(d_weights, &[weight_len]),
            OxideTensorF32::new(d_biases, &[filters]),
        ))
    }

    /// Pre-normalization envelope adjoint (lhtan + sanitize slopes).
    #[allow(clippy::too_many_arguments)]
    pub fn deform_envelope_pre_backward(
        &self,
        raw: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        normalize: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_envelope_pre_backward";
        let per_batch = planes
            .checked_mul(spatial)
            .filter(|&v| v > 0)
            .ok_or_else(|| shape_err(op, "planes and spatial must be positive".to_owned()))?;
        if raw.len() == 0 || raw.len() % per_batch != 0 || output_grad.len() != raw.len() {
            return Err(shape_err(op, "raw/gradient length mismatch".to_owned()));
        }
        let batch = raw.len() / per_batch;
        for (name, tensor) in [
            ("mean", mean),
            ("inv_std", inv_std),
            ("normalize", normalize),
        ] {
            if tensor.len() != batch {
                return Err(shape_err(
                    op,
                    format!("{name} has {} elements, expected {batch}", tensor.len()),
                ));
            }
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element; every operand is length-checked above.
        unsafe {
            self.deform_backward.deform_envelope_pre_backward(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mean.buf,
                &inv_std.buf,
                &normalize.buf,
                &output_grad.buf,
                &mut out,
                per_batch,
                max_offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Mask sigmoid adjoint.
    pub fn deform_mask_sigmoid_backward(
        &self,
        logits: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_mask_sigmoid_backward";
        if logits.len() == 0 || logits.len() != output_grad.len() {
            return Err(shape_err(op, "logit/gradient length mismatch".to_owned()));
        }
        let mut out = self.alloc_uninit(logits.len())?;
        // SAFETY: one thread per element, same length as both inputs.
        unsafe {
            self.deform_backward.deform_mask_sigmoid_backward(
                &self.stream,
                launch::elementwise(logits.len(), config),
                &logits.buf,
                &output_grad.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, logits.shape()))
    }

    /// Scale a summed input gradient by the raw input's `sanitize` slope.
    pub fn deform_scale_input_gradient(
        &self,
        raw: &OxideTensorF32,
        grad: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_scale_input_gradient";
        if raw.len() == 0 || raw.len() != grad.len() {
            return Err(shape_err(op, "raw/gradient length mismatch".to_owned()));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as both inputs.
        unsafe {
            self.deform_backward.deform_scale_input_gradient(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &grad.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Composed native `[deform_conv]` backward. Every per-element stage runs on
    /// the device; the **only** host hop is the batch-serial normalization
    /// Jacobian (mean/std coupling), which also applies the stage-1 slope.
    ///
    /// Returns `(d_input, d_main_weights, d_offset_weights, d_offset_biases,
    /// d_mask_weights, d_mask_biases)`.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_backward_native(
        &self,
        input_sanitized: &OxideTensorF32,
        raw_input: &OxideTensorF32,
        main_weights: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        mask_weights: Option<&OxideTensorF32>,
        raw_offsets: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        stddev: &OxideTensorF32,
        normalize: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: Option<&OxideTensorF32>,
        mask_logits: Option<&OxideTensorF32>,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            Option<OxideTensorF32>,
            Option<OxideTensorF32>,
        ),
        OxideError,
    > {
        let op = "deform_conv_backward_native";
        let kernels = geometry.kernels();
        let plane = geometry.plane();
        let per_batch = 2 * kernels * plane;
        let empty = self.upload_f32(&[1], &[0.0])?;
        let masks_operand = masks.unwrap_or(&empty);
        // 1. accumulation adjoint.
        let (d_input_accum, d_offsets, d_masks, d_main_weights) = self
            .deform_conv_accumulate_backward(
                input_sanitized,
                offsets,
                masks_operand,
                main_weights,
                output_grad,
                geometry,
                masks.is_some(),
                config,
            )?;
        // 2. lhtan/sanitize slopes (device).
        let d_pre = self.deform_envelope_pre_backward(
            raw_offsets,
            mean,
            inv_std,
            normalize,
            &d_offsets,
            2 * kernels,
            plane,
            max_offset,
            config,
        )?;
        // 3. the one host hop: batch-coupled normalization Jacobian + stage-1 slope.
        let stage1_host = self.download_f32(raw_offsets)?;
        let stage1_host: Vec<f32> = stage1_host
            .iter()
            .map(|&v| lightnet_cuda::host_ops::deform_conv::sanitize(v))
            .collect();
        let d_pre_host = self.download_f32(&d_pre)?;
        let mean_host = self.download_f32(mean)?;
        let std_host = self.download_f32(stddev)?;
        let normalize_host = self.download_f32(normalize)?;
        let d_stage1 = lightnet_cuda::host_ops::deform_conv::offset_normalization_backward(
            &stage1_host,
            &mean_host,
            &std_host,
            &normalize_host,
            &d_pre_host,
            per_batch,
        )
        .map_err(|msg| shape_err(op, msg))?;
        let raw_host = self.download_f32(raw_offsets)?;
        let d_offsets_predictor: Vec<f32> = d_stage1
            .iter()
            .zip(&raw_host)
            .map(|(&d, &raw)| {
                let slope = if raw.is_finite() && raw.abs() < 10_000.0 {
                    1.0
                } else {
                    0.0
                };
                d * slope
            })
            .collect();
        let d_offsets_predictor = self.upload_f32(raw_offsets.shape(), &d_offsets_predictor)?;
        // 4. predictor + column adjoint (device).
        let (d_input_off, d_offset_weights, d_offset_biases) = self.deform_predictor_backward(
            input_sanitized,
            offset_weights,
            &d_offsets_predictor,
            geometry.channels,
            2 * kernels,
            geometry.input_height,
            geometry.input_width,
            geometry.output_height,
            geometry.output_width,
            geometry.size,
            geometry.stride_x,
            geometry.stride_y,
            geometry.padding,
            geometry.dilation,
            geometry.batch,
            config,
        )?;
        // 5. mask path (device), when present.
        let (d_input_mask, d_mask_weights, d_mask_biases) =
            match (mask_weights, mask_logits, masks.is_some()) {
                (Some(weights), Some(logits), true) => {
                    let d_logits = self.deform_mask_sigmoid_backward(logits, &d_masks, config)?;
                    let (d_in, d_w, d_b) = self.deform_predictor_backward(
                        input_sanitized,
                        weights,
                        &d_logits,
                        geometry.channels,
                        kernels,
                        geometry.input_height,
                        geometry.input_width,
                        geometry.output_height,
                        geometry.output_width,
                        geometry.size,
                        geometry.stride_x,
                        geometry.stride_y,
                        geometry.padding,
                        geometry.dilation,
                        geometry.batch,
                        config,
                    )?;
                    (Some(d_in), Some(d_w), Some(d_b))
                }
                _ => (None, None, None),
            };
        // 6. sum and apply the input slope exactly once.
        let mut sum = self.vector_add_f32(&d_input_accum, &d_input_off, config)?;
        if let Some(d_in) = d_input_mask.as_ref() {
            sum = self.vector_add_f32(&sum, d_in, config)?;
        }
        let d_input = self.deform_scale_input_gradient(raw_input, &sum, config)?;
        Ok((
            d_input,
            d_main_weights,
            d_offset_weights,
            d_offset_biases,
            d_mask_weights,
            d_mask_biases,
        ))
    }

    /// Device DCNv4 aggregation (offsets given). Rejects kernel areas above
    /// `kernels::dcnv4::MAX_POINTS`, which the fixed-size mask buffer cannot hold.
    #[allow(clippy::too_many_arguments)]
    pub fn dcnv4_aggregate_native(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::dcnv4::Dcnv4Geometry,
        offset_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "dcnv4_aggregate_native";
        geometry.checked().map_err(|msg| shape_err(op, msg))?;
        let points = geometry.kernel_points();
        if points > crate::kernels::dcnv4::MAX_POINTS {
            return Err(shape_err(
                op,
                format!("kernel area {points} exceeds the device limit"),
            ));
        }
        let plane = geometry.input_height * geometry.input_width;
        let spatial = geometry.spatial();
        let offset_dim = geometry.offset_dim();
        if input.len() != geometry.batch * geometry.channels * plane
            || offsets.len() != geometry.batch * spatial * offset_dim
        {
            return Err(shape_err(op, "operand length mismatch".to_owned()));
        }
        let out_len = geometry.batch * geometry.channels * spatial;
        // Zeroed: the kernel accumulates into each owned output cell.
        let mut out = self.alloc(out_len)?;
        let threads = geometry.batch * geometry.groups * spatial;
        // SAFETY: one thread per (batch, group, cell); each writes only its own
        // group's channels, and every tap is bounds-checked in the kernel.
        unsafe {
            self.dcnv4.dcnv4_aggregate(
                &self.stream,
                launch::elementwise(threads, config),
                &input.buf,
                &offsets.buf,
                &mut out,
                geometry.channels,
                geometry.groups,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(geometry.softmax),
                u32::from(geometry.remove_center),
                offset_scale,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(
            out,
            &[
                geometry.batch,
                geometry.channels,
                geometry.output_height,
                geometry.output_width,
            ],
        ))
    }

    /// Device adjoint of [`Self::dcnv4_aggregate_native`]: `(d_input, d_offsets)`.
    #[allow(clippy::too_many_arguments)]
    pub fn dcnv4_aggregate_backward_native(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::dcnv4::Dcnv4Geometry,
        offset_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "dcnv4_aggregate_backward_native";
        geometry.checked().map_err(|msg| shape_err(op, msg))?;
        let points = geometry.kernel_points();
        if points > crate::kernels::dcnv4::MAX_POINTS {
            return Err(shape_err(
                op,
                format!("kernel area {points} exceeds the device limit"),
            ));
        }
        let plane = geometry.input_height * geometry.input_width;
        let spatial = geometry.spatial();
        let offset_dim = geometry.offset_dim();
        if input.len() != geometry.batch * geometry.channels * plane
            || offsets.len() != geometry.batch * spatial * offset_dim
            || output_grad.len() != geometry.batch * geometry.channels * spatial
        {
            return Err(shape_err(op, "operand length mismatch".to_owned()));
        }
        let d_input = self.alloc(geometry.batch * geometry.channels * plane)?;
        let d_offsets = self.alloc(geometry.batch * spatial * offset_dim)?;
        let threads = geometry.batch * geometry.groups * spatial;
        // SAFETY: one thread per (batch, group, cell); `d_offsets` slots are
        // owned by that thread, `d_input` is written only by atomic adds into a
        // zeroed buffer of the validated extent.
        unsafe {
            self.dcnv4_backward.dcnv4_aggregate_backward(
                &self.stream,
                launch::elementwise(threads, config),
                &input.buf,
                &offsets.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_offsets.cu_deviceptr() as *mut f32,
                geometry.channels,
                geometry.groups,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(geometry.softmax),
                u32::from(geometry.remove_center),
                offset_scale,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(
                d_input,
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
            ),
            OxideTensorF32::new(
                d_offsets,
                &[geometry.batch, geometry.spatial(), geometry.offset_dim()],
            ),
        ))
    }

    /// Per-batch offset statistics over the sanitized stage-1 offsets. Mirrors
    /// the source f64 accumulation and returns `(mean, inv_std, normalize)`
    /// (the last as 0.0/1.0), each of length `batch`.
    pub fn deform_offset_stats(
        &self,
        raw: &OxideTensorF32,
        per_batch: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "deform_offset_stats";
        if per_batch == 0 || raw.len() == 0 || raw.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "raw has {} elements, not a positive multiple of per_batch {per_batch}",
                    raw.len()
                ),
            ));
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let batch = raw.len() / per_batch;
        let mut mean = self.alloc_uninit(batch)?;
        let mut inv_std = self.alloc_uninit(batch)?;
        let mut normalize = self.alloc_uninit(batch)?;
        // SAFETY: one thread per batch; each thread owns its own output slot and
        // reads only its own contiguous `per_batch` slice.
        unsafe {
            self.deform_sample.deform_offset_stats(
                &self.stream,
                launch::elementwise(batch, config),
                &raw.buf,
                &mut mean,
                &mut inv_std,
                &mut normalize,
                per_batch,
                f64::from(max_offset),
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(mean, &[batch]),
            OxideTensorF32::new(inv_std, &[batch]),
            OxideTensorF32::new(normalize, &[batch]),
        ))
    }

    /// `[deform_conv]` offset envelope: `sanitize -> optional per-batch
    /// normalization -> sanitize -> centered lhtan`. `mean`, `inv_std` and
    /// `normalize` (0.0/1.0) are per-batch tensors produced by the caller; the
    /// stats themselves are not computed here.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_offset_envelope(
        &self,
        raw: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        normalize: &OxideTensorF32,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_offset_envelope";
        let per_batch = planes
            .checked_mul(spatial)
            .filter(|&v| v > 0)
            .ok_or_else(|| shape_err(op, "planes and spatial must be positive".to_owned()))?;
        if raw.len() == 0 || raw.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "raw has {} elements, not a multiple of planes*spatial = {per_batch}",
                    raw.len()
                ),
            ));
        }
        let batch = raw.len() / per_batch;
        for (name, tensor) in [
            ("mean", mean),
            ("inv_std", inv_std),
            ("normalize", normalize),
        ] {
            if tensor.len() != batch {
                return Err(shape_err(
                    op,
                    format!("{name} has {} elements, expected {batch}", tensor.len()),
                ));
            }
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per offset element; all four inputs are validated
        // above and indexed by decode-verified positions.
        unsafe {
            self.deform_sample.deform_offset_envelope(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mean.buf,
                &inv_std.buf,
                &normalize.buf,
                &mut out,
                planes,
                spatial,
                max_offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// `[deform_conv]` mask probabilities: `stable_sigmoid(sanitize(logit))`.
    pub fn deform_mask_sigmoid(
        &self,
        raw: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if raw.len() == 0 {
            return Err(shape_err(
                "deform_mask_sigmoid",
                "logits must not be empty".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as the validated input.
        unsafe {
            self.deform_sample.deform_mask_sigmoid(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }
}

impl lightnet_cuda::DenseLossBackend for OxideBackend {
    fn jev_ce_loss_f32(
        &self,
        logits: &Self::Tensor,
        truth: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        normalize: bool,
        ignore_empty: bool,
        label_smoothing: f32,
        scale: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, Self::Tensor), Self::Error> {
        let logits_f = self.f32v(logits)?;
        let logits = &*logits_f;
        let truth_f = self.f32v(truth)?;
        let truth = &*truth_f;
        const OP: &str = "jev_ce_loss_f32";
        let batch = if shape.len() == 4 { shape[0] } else { 0 };
        if shape.len() != 4
            || batch == 0
            || groups == 0
            || candidates == 0
            || groups
                .checked_mul(candidates)
                .and_then(|per_sample| per_sample.checked_mul(batch))
                != Some(logits.len())
            || logits.len() != truth.len()
            || logits.shape() != shape
            || truth.shape() != shape
        {
            return Err(shape_err(
                OP,
                format!(
                    "incompatible JevCe shape/groups/candidates: {shape:?}, {groups}, {candidates}"
                ),
            ));
        }
        let work = batch * groups;
        let mut delta = self.alloc_uninit(logits.len())?;
        let mut losses = self.alloc(work)?;
        let mut active = self.alloc(work)?;
        // SAFETY: one thread per sample/group owns its candidate entries and
        // its loss/active slot.
        unsafe {
            self.dense_loss.jev_ce_loss(
                &self.stream,
                launch::elementwise(work, config),
                &logits.buf,
                &truth.buf,
                &mut delta,
                &mut losses,
                &mut active,
                batch,
                groups,
                candidates,
                temperature,
                normalize as u32,
                ignore_empty as u32,
                label_smoothing,
                scale,
                channel_major as u32,
            )
        }
        .map_err(launch_err)?;
        let loss_host = losses.to_host_vec(&self.stream)?;
        let active_host = active.to_host_vec(&self.stream)?;
        let (mut sum, mut count) = (0.0f64, 0usize);
        for (loss, active) in loss_host.iter().zip(&active_host) {
            if *active != 0.0 {
                sum += *loss as f64;
                count += 1;
            }
        }
        let mean = if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        };
        Ok((mean, OxideTensorF32::new(delta, shape)))
    }

    fn segmentation_loss_f32(
        &self,
        output: &Self::Tensor,
        truth: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        spatial: usize,
        softmax_loss: bool,
        scale: f32,
        gate_strength: f32,
        gate_t_correct: f32,
        gate_max_compensation: f32,
        max_delta: f32,
        clipping: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, Self::Tensor), Self::Error> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let truth_f = self.f32v(truth)?;
        let truth = &*truth_f;
        const OP: &str = "segmentation_loss_f32";
        let batch = if shape.len() == 4 { shape[0] } else { 0 };
        let expected_spatial = if shape.len() == 4 {
            shape[2] * shape[3]
        } else {
            0
        };
        if shape.len() != 4
            || batch == 0
            || classes == 0
            || spatial == 0
            || spatial != expected_spatial
            || classes
                .checked_mul(spatial)
                .and_then(|per_sample| per_sample.checked_mul(batch))
                != Some(output.len())
            || output.len() != truth.len()
            || output.shape() != shape
            || truth.shape() != shape
        {
            return Err(shape_err(
                OP,
                format!(
                    "incompatible segmentation shape/classes/spatial: {shape:?}, {classes}, {spatial}"
                ),
            ));
        }
        let total = output.len();
        let mut delta = self.alloc_uninit(total)?;
        let mut partial_loss = self.alloc(batch)?;
        let mut suppressed = self.alloc(batch)?;
        // SAFETY: one thread per sample owns its output/truth reads and its
        // per-sample loss/suppressed slots; the delta slots are disjoint.
        unsafe {
            self.dense_loss.segmentation_loss(
                &self.stream,
                launch::elementwise(batch, config),
                &output.buf,
                &truth.buf,
                &mut delta,
                &mut partial_loss,
                &mut suppressed,
                batch,
                classes,
                spatial,
                softmax_loss as u32,
                scale,
                gate_strength,
                gate_t_correct,
            )
        }
        .map_err(launch_err)?;
        let partial_host = partial_loss.to_host_vec(&self.stream)?;
        let suppressed_host = suppressed.to_host_vec(&self.stream)?;
        let mut loss_sum = 0.0f32;
        let mut suppressed_total = 0.0f32;
        for (loss, count) in partial_host.iter().zip(&suppressed_host) {
            loss_sum += *loss;
            suppressed_total += *count;
        }
        let total_pixels = batch * spatial;
        let active = total_pixels as f32 - suppressed_total;
        let mut compensation = if active > 0.0 && active < total_pixels as f32 {
            (total_pixels as f32 / active).min(gate_max_compensation)
        } else {
            1.0
        };
        if compensation <= 1.0 {
            compensation = 1.0;
        }
        // SAFETY: one thread per delta element.
        unsafe {
            self.dense_loss.segmentation_scale_delta(
                &self.stream,
                launch::elementwise(total, config),
                &mut delta,
                compensation,
                max_delta,
                clipping as u32,
            )
        }
        .map_err(launch_err)?;
        let mean = if total_pixels == 0 {
            0.0
        } else {
            loss_sum / total_pixels as f32
        };
        Ok((mean, OxideTensorF32::new(delta, shape)))
    }
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

    fn bias_activation_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        if x.is_half() {
            let biased = self.bias_add_f32(x, bias, channels, channel_stride, config)?;
            return self.activation_f32(&biased, activation, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let bias_f = self.f32v(bias)?;
        let bias = &*bias_f;
        check_channel_layout(
            "bias_activation_f32",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.bias_activation(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
                activation_code(activation),
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

    /// Row-wise LayerNorm, one block (one thread) per row so the reduction needs
    /// no second pass.  `rows * cols` must equal the tensor length and `cols` must
    /// fit [`activations::layernorm::LAYERNORM_MAX_COLS`].
    /// `out[r, n] = sum_k x[r, k] * w[n, k] + b[n]`, routed into the
    /// existing fp32 GEMM with `trans_b` so the `[out_features, in_features]`
    /// weight layout is read as `B^T`.  `rows` becomes the GEMM's `m`, so one
    /// launch covers every token of a sequence.
    fn muon_orthogonalize_f32(
        &self,
        matrix: &Self::Tensor,
        shape: &[usize],
        rows: usize,
        cols: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        if shape.iter().product::<usize>() != rows.saturating_mul(cols) {
            return Err(shape_err(
                "muon_orthogonalize_f32",
                format!("shape {shape:?} does not match {rows}x{cols}"),
            ));
        }
        let input = self.f32v(matrix)?;
        OxideBackend::muon_orthogonalize_f32(self, &input, rows, cols)
    }

    fn muon_momentum_lookahead_f32(
        &self,
        updates: &Self::Tensor,
        momentum: &Self::Tensor,
        shape: &[usize],
        batch: usize,
        loss_scale: f32,
        coefficient: f32,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let updates = self.f32v(updates)?;
        let momentum = self.f32v(momentum)?;
        OxideBackend::muon_momentum_lookahead_f32(
            self,
            &updates,
            &momentum,
            shape,
            batch,
            loss_scale,
            coefficient,
        )
    }

    fn muon_apply_update_f32(
        &self,
        weights: &Self::Tensor,
        direction: &Self::Tensor,
        shape: &[usize],
        learning_rate: f32,
        decay: f32,
        direction_scale: f32,
    ) -> Result<Self::Tensor, Self::Error> {
        let weights = self.f32v(weights)?;
        let direction = self.f32v(direction)?;
        OxideBackend::muon_apply_update_f32(
            self,
            &weights,
            &direction,
            shape,
            learning_rate,
            decay,
            direction_scale,
        )
    }

    fn linear_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        in_features: usize,
        out_features: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        if rows == 0 || in_features == 0 || out_features == 0 {
            return Err(OxideError::Shape(
                "linear_f32: rows, in_features and out_features must be > 0".to_owned(),
            ));
        }
        if shape.iter().product::<usize>() != rows * out_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: shape {shape:?} must hold rows*out_features = {}",
                rows * out_features
            )));
        }
        let x_f = self.f32v(x)?.into_owned();
        let w_f = self.f32v(weights)?.into_owned();
        if x_f.len() < rows * in_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: activation has {} entries, expected {}",
                x_f.len(),
                rows * in_features
            )));
        }
        if w_f.len() < out_features * in_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: weights have {} entries, expected {}",
                w_f.len(),
                out_features * in_features
            )));
        }
        // Fold the bias into a beta-1 accumulator copy of the output, which the
        // GEMM already accumulates into; that avoids a second epilogue pass.
        let mut acc = self.alloc_uninit(rows * out_features)?;
        if let Some(bias) = bias {
            let b = self.f32v(bias)?.into_owned();
            if b.len() < out_features {
                return Err(OxideError::Shape(format!(
                    "linear_f32: bias has {} entries, expected {out_features}",
                    b.len()
                )));
            }
            let host = b.buf.to_host_vec(&self.stream)?;
            let mut values = vec![0.0f32; rows * out_features];
            for r in 0..rows {
                values[r * out_features..(r + 1) * out_features]
                    .copy_from_slice(&host[..out_features]);
            }
            acc.copy_from_host(&self.stream, &values)?;
        } else {
            acc.copy_from_host(&self.stream, &vec![0.0f32; rows * out_features])?;
        }
        // `gemm` takes a `&mut DeviceBuffer`, so drive it on the pooled buffer
        // directly and keep the tensor only for the returned shape.
        let mut acc_raw = acc;
        self.gemm(
            &x_f.buf,
            &w_f.buf,
            &mut acc_raw,
            &GemmArgs {
                m: rows,
                n: out_features,
                k: in_features,
                lda: in_features,
                // `w` is [out_features, in_features], i.e. B^T of the
                // [in_features, out_features] matrix the GEMM wants.
                ldb: in_features,
                ldc: out_features,
                trans_a: false,
                trans_b: true,
                alpha: 1.0,
                beta: 1.0,
                ..GemmArgs::default()
            },
        )?;
        Ok(OxideTensorF32::new(acc_raw, shape))
    }

    /// Fused residual + bias epilogue, one thread per output element:
    /// `out[i] = residual[i] + y[i] + bias[i % cols]`.
    fn residual_bias_f32(
        &self,
        residual: &Self::Tensor,
        y: &Self::Tensor,
        bias: Option<&Self::Tensor>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            OxideError::Shape("residual_bias_f32: rows * cols overflowed".to_owned())
        })?;
        if rows == 0 || cols == 0 || len != shape.iter().product::<usize>() {
            return Err(OxideError::Shape(format!(
                "residual_bias_f32: rows*cols must equal the tensor length ({rows}*{cols} vs {:?})",
                shape
            )));
        }
        let r = self.f32v(residual)?.into_owned();
        let yv = self.f32v(y)?.into_owned();
        // A missing bias is a one-element placeholder the kernel never indexes.
        let bias_tensor = match bias {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1usize]),
        };
        let has_bias = u32::from(bias.is_some());
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: one thread per element, and the kernel only reads `bias` when
        // `has_bias` is set (then it has `cols` entries).
        unsafe {
            self.residual_epilogue.residual_bias(
                &self.stream,
                launch::elementwise(len, config),
                &r.buf,
                &yv.buf,
                &bias_tensor.buf,
                &mut out,
                cols,
                has_bias,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

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
        check_layernorm_shape("layernorm_forward_f32", shape, rows, cols)?;
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        // A missing gamma / beta becomes a one-element placeholder tensor the
        // kernel never reads (the flag gates the access), so there is no `cols`
        // allocation and no extra launch.  Both arms are `OxideTensorF32` so the
        // borrowed buffer below coerces exactly like every other operand.
        let gamma_tensor = match gamma {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let beta_tensor = match beta {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let has_gamma = u32::from(gamma.is_some());
        let has_beta = u32::from(beta.is_some());
        let mut out = self.alloc_uninit(rows * cols)?;
        // SAFETY: one thread per row, each writing only its own `cols` outputs.
        unsafe {
            self.layernorm.layernorm_forward(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &x.buf,
                &gamma_tensor.buf,
                &beta_tensor.buf,
                &mut out,
                rows,
                cols,
                eps,
                has_gamma,
                has_beta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    /// Row-wise LayerNorm input gradient, same one-thread-per-row reduction.
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
        check_layernorm_shape("layernorm_backward_f32", shape, rows, cols)?;
        let dy_f = self.f32v(output_grad)?;
        let x_f = self.f32v(x)?;
        let dy = &*dy_f;
        let x = &*x_f;
        let gamma_tensor = match gamma {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let has_gamma = u32::from(gamma.is_some());
        let mut dx = self.alloc_uninit(rows * cols)?;
        // SAFETY: one thread per row, each writing only its own `cols` outputs.
        unsafe {
            self.layernorm.layernorm_backward(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &dy.buf,
                &x.buf,
                &gamma_tensor.buf,
                &mut dx,
                rows,
                cols,
                eps,
                has_gamma,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
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

    fn gelu_erf_f32(
        &self,
        input: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "gelu_erf_f32";
        let len = lightnet_cuda::host_ops::gelu_erf::checked_len(shape)
            .map_err(|err| shape_err(op, err))?;
        if input.len() != len || input.shape() != shape {
            return Err(shape_err(op, "input shape/storage mismatch".to_owned()));
        }
        let input_f = self.f32v(input)?;
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: checked nonempty storage; each thread owns one output.
        unsafe {
            self.activations.gelu_erf(
                &self.stream,
                launch::elementwise(len, config),
                &input_f.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn gelu_erf_backward_f32(
        &self,
        input: &Self::Tensor,
        seeds: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "gelu_erf_backward_f32";
        let len = lightnet_cuda::host_ops::gelu_erf::checked_len(shape)
            .map_err(|err| shape_err(op, err))?;
        if input.len() != len
            || seeds.len() != len
            || input.shape() != shape
            || seeds.shape() != shape
        {
            return Err(shape_err(
                op,
                "input/seed shape or storage mismatch".to_owned(),
            ));
        }
        let input_f = self.f32v(input)?;
        let seeds_f = self.f32v(seeds)?;
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: all three arrays have the same checked length.
        unsafe {
            self.activations.gelu_erf_backward(
                &self.stream,
                launch::elementwise(len, config),
                &input_f.buf,
                &seeds_f.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
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

    fn normalize_channels_f32(
        &self,
        input: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "normalize_channels_f32";
        let total = channel_normalization_size(op, output_shape, batch, channels, spatial)?;
        let input = self.f32v(input)?;
        let input = &*input;
        if input.len() != total || input.shape() != output_shape {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected {output_shape:?}/{total}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each output element; shape and dimensions
        // have been validated and each thread's channel loop stays in bounds.
        unsafe {
            self.normalize_channels.normalize_channels(
                &self.stream,
                launch::elementwise(total, config),
                &input.buf,
                &mut output,
                channels,
                spatial,
                channel_normalization_code(normalization),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn normalize_channels_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        output: &Self::Tensor,
        delta: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "normalize_channels_backward_f32";
        let total = channel_normalization_size(op, output_shape, batch, channels, spatial)?;
        let activation_input = self.f32v(activation_input)?;
        let activation_input = &*activation_input;
        let output = self.f32v(output)?;
        let output = &*output;
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        if activation_input.len() != total
            || output.len() != total
            || delta.len() != total
            || activation_input.shape() != output_shape
            || output.shape() != output_shape
            || delta.shape() != output_shape
        {
            return Err(shape_err(
                op,
                format!(
                    "input/output/delta shapes and lengths are {:?}/{}, {:?}/{}, {:?}/{}, expected {output_shape:?}/{total}",
                    activation_input.shape(),
                    activation_input.len(),
                    output.shape(),
                    output.len(),
                    delta.shape(),
                    delta.len()
                ),
            ));
        }
        let mut gradient = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each gradient element; all tensor lengths
        // and nonzero geometry were checked above.
        unsafe {
            self.normalize_channels.normalize_channels_backward(
                &self.stream,
                launch::elementwise(total, config),
                &activation_input.buf,
                &output.buf,
                &delta.buf,
                &mut gradient,
                channels,
                spatial,
                channel_normalization_code(normalization),
                cpp_grad as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, output_shape))
    }

    fn eml_convolutional_f32(
        &self,
        x_branch: &Self::Tensor,
        self_branch: &Self::Tensor,
        residual: Option<&Self::Tensor>,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "eml_convolutional_f32";
        let clamp = if clamp > 0.0 { clamp } else { 4.0 };
        let eps = if eps > 0.0 { eps } else { 1.0e-6 };
        let total: usize = output_shape.iter().product();
        if output_shape.is_empty() || total == 0 {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let x = self.f32v(x_branch)?;
        let y = self.f32v(self_branch)?;
        let residual_tensor = residual.map(|tensor| self.f32v(tensor)).transpose()?;
        let has_residual = residual_tensor.is_some();
        let x = &*x;
        let y = &*y;
        let residual = residual_tensor.as_ref().map(|tensor| &**tensor);
        if x.len() != total
            || y.len() != total
            || x.shape() != output_shape
            || y.shape() != output_shape
            || residual
                .is_some_and(|tensor| tensor.len() != total || tensor.shape() != output_shape)
        {
            return Err(shape_err(
                op,
                format!(
                    "branch/residual shapes must match {output_shape:?}/{total}; got {:?}/{}, {:?}/{}, {:?}",
                    x.shape(),
                    x.len(),
                    y.shape(),
                    y.len(),
                    residual.map(|tensor| (tensor.shape(), tensor.len()))
                ),
            ));
        }
        let mut output = self.alloc_uninit(total)?;
        let residual = residual.unwrap_or(x);
        // SAFETY: one thread owns each output index; input extents and shapes
        // match the validated nonempty output extent above.
        unsafe {
            self.eml.eml_convolutional_forward(
                &self.stream,
                launch::elementwise(total, config),
                &x.buf,
                &y.buf,
                &residual.buf,
                &mut output,
                clamp,
                eps,
                scale,
                has_residual as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn eml_convolutional_backward_f32(
        &self,
        x_branch: &Self::Tensor,
        self_branch: &Self::Tensor,
        output_grad: &Self::Tensor,
        output_shape: &[usize],
        clamp: f32,
        eps: f32,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let op = "eml_convolutional_backward_f32";
        let clamp = if clamp > 0.0 { clamp } else { 4.0 };
        let eps = if eps > 0.0 { eps } else { 1.0e-6 };
        let total: usize = output_shape.iter().product();
        if output_shape.is_empty() || total == 0 {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let x = self.f32v(x_branch)?;
        let y = self.f32v(self_branch)?;
        let delta = self.f32v(output_grad)?;
        let x = &*x;
        let y = &*y;
        let delta = &*delta;
        if x.len() != total
            || y.len() != total
            || delta.len() != total
            || x.shape() != output_shape
            || y.shape() != output_shape
            || delta.shape() != output_shape
        {
            return Err(shape_err(
                op,
                format!(
                    "branch/delta shapes must match {output_shape:?}/{total}; got {:?}/{}, {:?}/{}, {:?}/{}",
                    x.shape(),
                    x.len(),
                    y.shape(),
                    y.len(),
                    delta.shape(),
                    delta.len()
                ),
            ));
        }
        let mut x_gradient = self.alloc_uninit(total)?;
        let mut self_gradient = self.alloc_uninit(total)?;
        // SAFETY: each thread writes matching positions in both gradients;
        // the three input extents and shapes have been checked above.
        unsafe {
            self.eml.eml_convolutional_backward(
                &self.stream,
                launch::elementwise(total, config),
                &x.buf,
                &y.buf,
                &delta.buf,
                &mut x_gradient,
                &mut self_gradient,
                clamp,
                eps,
                scale,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(x_gradient, output_shape),
            OxideTensorF32::new(self_gradient, output_shape),
        ))
    }

    fn lstm_gate_backward_f32(
        &self,
        hidden_gradient: &Self::Tensor,
        cell: &Self::Tensor,
        previous_cell: &Self::Tensor,
        forget: &Self::Tensor,
        input_gate: &Self::Tensor,
        candidate: &Self::Tensor,
        output_gate: &Self::Tensor,
        next_cell_gradient: &Self::Tensor,
        elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
            Self::Tensor,
        ),
        Self::Error,
    > {
        let hidden_gradient = self.f32v(hidden_gradient)?;
        let cell = self.f32v(cell)?;
        let previous_cell = self.f32v(previous_cell)?;
        let forget = self.f32v(forget)?;
        let input_gate = self.f32v(input_gate)?;
        let candidate = self.f32v(candidate)?;
        let output_gate = self.f32v(output_gate)?;
        let next_cell_gradient = self.f32v(next_cell_gradient)?;
        let op = "lstm_gate_backward_f32";
        for input in [
            &*cell,
            &*previous_cell,
            &*forget,
            &*input_gate,
            &*candidate,
            &*output_gate,
            &*next_cell_gradient,
        ] {
            same_len(op, &hidden_gradient, input)?;
        }
        if hidden_gradient.len() != elements {
            return Err(shape_err(
                op,
                format!(
                    "expected {elements} elements, got {}",
                    hidden_gradient.len()
                ),
            ));
        }
        let mut forget_gradient = self.alloc_uninit(elements)?;
        let mut input_gate_gradient = self.alloc_uninit(elements)?;
        let mut candidate_gradient = self.alloc_uninit(elements)?;
        let mut output_gate_gradient = self.alloc_uninit(elements)?;
        let mut cell_gradient = self.alloc_uninit(elements)?;
        // SAFETY: one thread owns the same element across five disjoint outputs.
        unsafe {
            self.recurrent.lstm_gate_backward(
                &self.stream,
                launch::elementwise(elements, config),
                &hidden_gradient.buf,
                &cell.buf,
                &previous_cell.buf,
                &forget.buf,
                &input_gate.buf,
                &candidate.buf,
                &output_gate.buf,
                &next_cell_gradient.buf,
                &mut forget_gradient,
                &mut input_gate_gradient,
                &mut candidate_gradient,
                &mut output_gate_gradient,
                &mut cell_gradient,
            )
        }
        .map_err(launch_err)?;
        let shape = hidden_gradient.shape();
        Ok((
            OxideTensorF32::new(forget_gradient, shape),
            OxideTensorF32::new(input_gate_gradient, shape),
            OxideTensorF32::new(candidate_gradient, shape),
            OxideTensorF32::new(output_gate_gradient, shape),
            OxideTensorF32::new(cell_gradient, shape),
        ))
    }

    fn sequence_slice_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        step_index: usize,
        steps: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "sequence_slice_f32";
        if steps == 0 || step_index >= steps {
            return Err(shape_err(
                op,
                "invalid recurrent step index/count".to_owned(),
            ));
        }
        let input_elements = input_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "input shape element count overflow".to_owned()))?;
        if input_elements % steps != 0 {
            return Err(shape_err(
                op,
                format!("input has {input_elements} elements, not divisible by {steps} steps"),
            ));
        }
        let step_elements = input_elements / steps;
        let output_elements = output_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "output shape element count overflow".to_owned()))?;
        if output_elements != step_elements {
            return Err(shape_err(
                op,
                format!("output has {output_elements} elements, expected {step_elements}"),
            ));
        }
        let input = self.f32v(input)?;
        if input.len() != input_elements {
            return Err(shape_err(
                op,
                format!(
                    "input tensor has {} elements, expected {input_elements}",
                    input.len()
                ),
            ));
        }
        let offset = step_index
            .checked_mul(step_elements)
            .ok_or_else(|| shape_err(op, "sequence input offset overflow".to_owned()))?;
        let mut output = self.alloc_uninit(step_elements)?;
        // SAFETY: checked shapes and step index keep this copy within input.
        unsafe {
            self.recurrent.sequence_slice_step(
                &self.stream,
                launch::elementwise(step_elements, config),
                &input.buf,
                &mut output,
                offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn sequence_concat_f32(
        &self,
        steps: &[&Self::Tensor],
        output_shape: &[usize],
        step_elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "sequence_concat_f32";
        if steps.is_empty() {
            return Err(shape_err(
                op,
                "at least one recurrent step is required".to_owned(),
            ));
        }
        let expected = steps
            .len()
            .checked_mul(step_elements)
            .ok_or_else(|| shape_err(op, "output element count overflow".to_owned()))?;
        let shape_elements = output_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "output shape element count overflow".to_owned()))?;
        if shape_elements != expected {
            return Err(shape_err(
                op,
                format!(
                    "shape {output_shape:?} has {shape_elements} elements, expected {expected}"
                ),
            ));
        }
        let mut output = self.alloc_uninit(expected)?;
        for (step_index, input) in steps.iter().enumerate() {
            let input = self.f32v(input)?;
            if input.len() != step_elements {
                return Err(shape_err(
                    op,
                    format!(
                        "step {step_index} has {} elements, expected {step_elements}",
                        input.len()
                    ),
                ));
            }
            let offset = step_index
                .checked_mul(step_elements)
                .ok_or_else(|| shape_err(op, "step output offset overflow".to_owned()))?;
            // SAFETY: each ordered launch writes a distinct output slice.
            unsafe {
                self.recurrent.sequence_concat_step(
                    &self.stream,
                    launch::elementwise(step_elements, config),
                    &input.buf,
                    &mut output,
                    offset,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(output, output_shape))
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

    fn vector_mul_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let y_f = self.f32v(y)?;
        let y = &*y_f;
        same_len("vector_mul_f32", x, y)?;
        let expected: usize = shape.iter().product();
        if expected != x.len() {
            return Err(OxideError::Shape(format!(
                "vector_mul_f32 output shape has {expected} elements, expected {}",
                x.len()
            )));
        }
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.elementwise.vector_mul(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &y.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
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

    fn shortcut_weight_gradient_f32(
        &self,
        delta: &Self::Tensor,
        source: &Self::Tensor,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "shortcut_weight_gradient_f32";
        if batch == 0 || per_sample == 0 || layer_step == 0 || per_sample % layer_step != 0 {
            return Err(shape_err(
                op,
                "invalid batch or shortcut slot geometry".to_owned(),
            ));
        }
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        let source = self.f32v(source)?;
        let source = &*source;
        if delta.len() != batch * per_sample || source.len() != batch * source_per_sample {
            return Err(shape_err(
                op,
                format!(
                    "delta/source lengths are {}/{}, expected {}/{}",
                    delta.len(),
                    source.len(),
                    batch * per_sample,
                    batch * source_per_sample
                ),
            ));
        }
        let mut gradient = self.alloc_uninit(layer_step)?;
        // SAFETY: one thread owns each coefficient slot and reads only the
        // validated delta/source extents.
        unsafe {
            self.shortcut.shortcut_weight_gradient(
                &self.stream,
                launch::elementwise(layer_step, config),
                &delta.buf,
                &source.buf,
                &mut gradient,
                batch,
                per_sample,
                source_per_sample,
                layer_step,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, &[layer_step]))
    }

    fn shortcut_weight_gradient_accumulate_f32(
        &self,
        accumulated: &Self::Tensor,
        delta: &Self::Tensor,
        source: &Self::Tensor,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "shortcut_weight_gradient_accumulate_f32";
        if batch == 0
            || per_sample == 0
            || layer_step == 0
            || per_sample % layer_step != 0
            || num_inputs == 0
            || source_index >= num_inputs
        {
            return Err(shape_err(
                op,
                "invalid shortcut gradient geometry".to_owned(),
            ));
        }
        let total = num_inputs
            .checked_mul(layer_step)
            .ok_or_else(|| shape_err(op, "shortcut weight count overflows usize".to_owned()))?;
        let accumulated = self.f32v(accumulated)?;
        let accumulated = &*accumulated;
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        let source = self.f32v(source)?;
        let source = &*source;
        if accumulated.len() != total
            || delta.len() != batch * per_sample
            || source.len() != batch * source_per_sample
        {
            return Err(shape_err(
                op,
                format!(
                    "tensor lengths are {}/{}/{}, expected {}/{}/{}",
                    accumulated.len(),
                    delta.len(),
                    source.len(),
                    total,
                    batch * per_sample,
                    batch * source_per_sample
                ),
            ));
        }
        let mut out = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each output coefficient; all input extents
        // and source-row bounds are validated above.
        unsafe {
            self.shortcut.shortcut_weight_gradient_accumulate(
                &self.stream,
                launch::elementwise(total, config),
                &accumulated.buf,
                &delta.buf,
                &source.buf,
                &mut out,
                batch,
                per_sample,
                source_per_sample,
                num_inputs,
                layer_step,
                source_index,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[num_inputs, layer_step]))
    }

    fn shortcut_weight_normalized_gradient_f32(
        &self,
        raw_weights: &Self::Tensor,
        products: &Self::Tensor,
        num_inputs: usize,
        layer_step: usize,
        normalization: F32ShortcutWeightNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "shortcut_weight_normalized_gradient_f32";
        if num_inputs == 0 || layer_step == 0 {
            return Err(shape_err(op, "empty shortcut weight geometry".to_owned()));
        }
        let total = num_inputs
            .checked_mul(layer_step)
            .ok_or_else(|| shape_err(op, "shortcut weight count overflows usize".to_owned()))?;
        let raw_weights = self.f32v(raw_weights)?;
        let raw_weights = &*raw_weights;
        let products = self.f32v(products)?;
        let products = &*products;
        if raw_weights.len() != total || products.len() != total {
            return Err(shape_err(
                op,
                format!(
                    "raw/product lengths are {}/{}, expected {total}",
                    raw_weights.len(),
                    products.len()
                ),
            ));
        }
        let normalization_code = match normalization {
            F32ShortcutWeightNormalization::None => 0,
            F32ShortcutWeightNormalization::Relu => 1,
            F32ShortcutWeightNormalization::Softmax => 2,
        };
        let mut gradient = self.alloc_uninit(total)?;
        // SAFETY: one thread owns each result coefficient; input rows and slots
        // are bounded by the validated total length.
        unsafe {
            self.shortcut.shortcut_weight_normalized_gradient(
                &self.stream,
                launch::elementwise(total, config),
                &raw_weights.buf,
                &products.buf,
                &mut gradient,
                num_inputs,
                layer_step,
                normalization_code,
                cpp_grad as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, &[num_inputs, layer_step]))
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
            let halves = inputs
                .iter()
                .map(|t| self.as_half(t))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&OxideTensorF16> = halves.iter().collect();
            let out = self.route_concat_f16(
                &refs,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                output_shape,
                channel_stride,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let op = "route_concat_f32";
        let n = inputs.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
        {
            return Err(shape_err(
                op,
                "needs >= 1 input and equal-length metadata".to_owned(),
            ));
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
                    format!(
                        "input {k} length {} not a multiple of {per_batch}",
                        inputs[k].len()
                    ),
                ));
            }
            let b = inputs[k].len() / per_batch;
            if *batch.get_or_insert(b) != b {
                return Err(shape_err(
                    op,
                    format!("input {k} batch {b} differs from {batch:?}"),
                ));
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
            return self.route_backward_amp(
                output_grad,
                input_shapes,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                channel_stride,
                config,
            );
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
            return Err(shape_err(
                op,
                "needs >= 1 input and equal-length metadata".to_owned(),
            ));
        }
        if channel_stride == 0 {
            return Err(shape_err(op, "channel_stride must be > 0".to_owned()));
        }
        let out_total: usize = input_channels.iter().sum();
        let per_batch = out_total * channel_stride;
        if per_batch == 0 || output_grad.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "gradient length {} not a multiple of {per_batch}",
                    output_grad.len()
                ),
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

    fn deconvolutional_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let weights_f = self.f32v(weights)?;
        self.deconvolutional_forward_native(
            &x_f,
            &weights_f,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride,
            config,
        )
    }

    fn deconvolutional_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let x_f = self.f32v(x)?;
        let weights_f = self.f32v(weights)?;
        let output_grad_f = self.f32v(output_grad)?;
        self.deconvolutional_backward_native(
            &x_f,
            &weights_f,
            &output_grad_f,
            input_shape,
            output_shape,
            filters,
            kernel_size,
            stride,
            config,
        )
    }

    fn local_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err(shape_err(
                "local_f32",
                "input and output shapes must be NCHW".into(),
            ));
        }
        let [batch, channels, height, width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let [out_batch, out_filters, out_h, out_w] = [
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        ];
        if [
            batch,
            channels,
            height,
            width,
            filters,
            out_h,
            out_w,
            kernel_size,
            stride_x,
            stride_y,
        ]
        .contains(&0)
            || out_batch != batch
            || out_filters != filters
        {
            return Err(shape_err("local_f32", "invalid local geometry".into()));
        }
        let locations = out_h
            .checked_mul(out_w)
            .ok_or_else(|| shape_err("local_f32", "output location count overflow".into()))?;
        let kernel = channels
            .checked_mul(kernel_size)
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| shape_err("local_f32", "kernel size overflow".into()))?;
        let expected_x = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(height))
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| shape_err("local_f32", "input size overflow".into()))?;
        let expected_w = locations
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| shape_err("local_f32", "weight size overflow".into()))?;
        let expected_out = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(locations))
            .ok_or_else(|| shape_err("local_f32", "output size overflow".into()))?;
        if x.len() != expected_x
            || weights.len() != expected_w
            || output_shape.iter().product::<usize>() != expected_out
        {
            return Err(shape_err(
                "local_f32",
                format!(
                    "operand lengths input/weights/output-shape mismatch; expected {expected_x}/{expected_w}/{expected_out}"
                ),
            ));
        }
        let expected_biases = locations
            .checked_mul(filters)
            .ok_or_else(|| shape_err("local_f32", "bias size overflow".into()))?;
        if biases.len() != expected_biases {
            return Err(shape_err(
                "local_f32",
                format!("expected {expected_biases} biases, got {}", biases.len()),
            ));
        }
        let x = self.f32v(x)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut output = self.alloc_uninit(expected_out)?;
        // SAFETY: launch geometry and all indexed extents were checked above;
        // each output element has exactly one owning thread.
        unsafe {
            self.local.local_fwd(
                &self.stream,
                launch::elementwise(expected_out, config),
                &x.buf,
                &weights.buf,
                &biases.buf,
                &mut output,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn local_backward_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err(shape_err(
                "local_backward_f32",
                "input and output shapes must be NCHW".into(),
            ));
        }
        let (batch, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_h, out_w) = (output_shape[2], output_shape[3]);
        let locations = out_h.checked_mul(out_w).ok_or_else(|| {
            shape_err(
                "local_backward_f32",
                "output location count overflow".into(),
            )
        })?;
        let kernel = channels
            .checked_mul(kernel_size)
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| shape_err("local_backward_f32", "kernel size overflow".into()))?;
        let x_len = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(height))
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| shape_err("local_backward_f32", "input size overflow".into()))?;
        let output_len = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(locations))
            .ok_or_else(|| shape_err("local_backward_f32", "output size overflow".into()))?;
        let weights_len = locations
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| shape_err("local_backward_f32", "weight size overflow".into()))?;
        if x.len() != x_len
            || weights.len() != weights_len
            || output_grad.len() != output_len
            || output_shape[0] != batch
            || output_shape[1] != filters
            || [
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
            ]
            .contains(&0)
        {
            return Err(shape_err(
                "local_backward_f32",
                "invalid operand lengths or geometry".into(),
            ));
        }
        let x = self.f32v(x)?;
        let weights = self.f32v(weights)?;
        let delta = self.f32v(output_grad)?;
        let mut dx = self.alloc_uninit(x_len)?;
        let mut dw = self.alloc_uninit(weights_len)?;
        let mut db = self.alloc_uninit(locations * filters)?;
        // SAFETY: dgrad gathers overlapping patches; wgrad and bgrad each
        // reduce batch in a single owning thread, so no atomics are needed.
        unsafe {
            self.local.local_dgrad(
                &self.stream,
                launch::elementwise(x_len, config),
                &delta.buf,
                &weights.buf,
                &mut dx,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.local.local_wgrad(
                &self.stream,
                launch::elementwise(weights_len, config),
                &x.buf,
                &delta.buf,
                &mut dw,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.local.local_bgrad(
                &self.stream,
                launch::elementwise(locations * filters, config),
                &delta.buf,
                &mut db,
                batch,
                filters,
                locations,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(dx, input_shape),
            OxideTensorF32::new(
                dw,
                &[locations, filters, channels, kernel_size, kernel_size],
            ),
            OxideTensorF32::new(db, &[locations, filters]),
        ))
    }

    fn graph_conv_aggregate_f32(
        &self,
        input: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_aggregate_forward_native(input, geometry, config)
    }

    fn graph_conv_reference_f32(
        &self,
        input: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_reference_forward_native(input, geometry, config)
    }

    fn graph_conv_reference_backward_f32(
        &self,
        output_gradient: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_reference_backward_native(output_gradient, geometry, config)
    }

    fn graph_conv_learned_aggregate_f32(
        &self,
        input: &Self::Tensor,
        edge_kernel: &Self::Tensor,
        edge_biases: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_learned_aggregate_forward_native(
            input,
            edge_kernel,
            edge_biases,
            geometry,
            groups,
            config,
        )
    }

    fn graph_conv_learned_aggregate_backward_f32(
        &self,
        input: &Self::Tensor,
        edge_kernel: &Self::Tensor,
        edge_biases: &Self::Tensor,
        output_gradient: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.graph_conv_learned_aggregate_backward_native(
            input,
            edge_kernel,
            edge_biases,
            output_gradient,
            geometry,
            groups,
            config,
        )
    }

    fn graph_conv_aggregate_backward_f32(
        &self,
        output_gradient: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_aggregate_backward_native(output_gradient, geometry, config)
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
            return self.conv_forward_amp(
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
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        self.conv_forward(
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
    ) -> Result<Self::Tensor, Self::Error> {
        if self.amp {
            return self.conv_backward_input_amp(
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
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        self.conv_backward_input(
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
    ) -> Result<Self::Tensor, Self::Error> {
        if self.amp {
            return self.conv_backward_weights_amp(
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
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        self.conv_backward_weights(
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
    ) -> Result<Self::Tensor, Self::Error> {
        if x.is_half() {
            let xh = self.as_half(x)?;
            let out = self.maxpool_f16(
                &xh,
                output_shape,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let (_, out_h, out_w) = pool_layout(
            "maxpool_f32",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
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
            return self.maxpool_backward_amp(
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
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let (out_h, out_w) = Self::grad_hw(
            "maxpool_backward_f32",
            output_grad,
            x.len(),
            channels,
            input_height,
            input_width,
        )?;
        pool_layout(
            "maxpool_backward_f32",
            x.len(),
            output_grad.shape(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
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

    fn maxpool_depth_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        if x.shape().len() != 4
            || output_shape.len() != 4
            || input_channels == 0
            || output_shape[1] == 0
            || output_shape[2] == 0
            || output_shape[3] == 0
            || output_shape[1] > input_channels
            || x.shape()[1] != input_channels
            || x.shape()[0] != output_shape[0]
            || x.shape()[2] != output_shape[2]
            || x.shape()[3] != output_shape[3]
        {
            return Err(shape_err(
                "maxpool_depth_f32",
                "incompatible NCHW shapes".to_owned(),
            ));
        }
        let spatial = output_shape[2] * output_shape[3];
        let mut output = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: shape checks establish matching batch/spatial dimensions and
        // valid strided channel groups; each output is written exactly once.
        unsafe {
            self.pool.maxpool_depth_forward(
                &self.stream,
                launch::elementwise(output.len(), config),
                &x.buf,
                &mut output,
                input_channels,
                output_shape[1],
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn maxpool_depth_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        if input_shape.len() != 4
            || x.shape() != input_shape
            || output_grad.shape().len() != 4
            || output_channels == 0
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] == 0
            || input_shape[3] == 0
            || output_channels > input_shape[1]
            || output_grad.shape()[0] != input_shape[0]
            || output_grad.shape()[1] != output_channels
            || output_grad.shape()[2] != input_shape[2]
            || output_grad.shape()[3] != input_shape[3]
        {
            return Err(shape_err(
                "maxpool_depth_backward_f32",
                "incompatible NCHW shapes".to_owned(),
            ));
        }
        let spatial = input_shape[2] * input_shape[3];
        let mut dx = self.alloc_uninit(x.len())?;
        // SAFETY: shape checks establish matching extents; each input gradient
        // has one owner and all entries are written.
        unsafe {
            self.pool.maxpool_depth_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &x.buf,
                &output_grad.buf,
                &mut dx,
                input_shape[1],
                output_channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
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
            "local_avgpool_f32",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
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
        let batch = nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )?;
        let (out_h, out_w) = Self::grad_hw(
            op,
            output_grad,
            batch * channels * input_height * input_width,
            channels,
            input_height,
            input_width,
        )?;
        pool_layout(
            op,
            input_shape.iter().product(),
            output_grad.shape(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
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
                format!(
                    "input has {} elements, expected {batch}*{channels}*{area}",
                    x.len()
                ),
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
        let batch = nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )?;
        let area = input_height * input_width;
        if area == 0 || output_grad.len() != batch * channels {
            return Err(shape_err(
                op,
                format!(
                    "gradient has {} elements, expected {batch}*{channels}",
                    output_grad.len()
                ),
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

    fn moe_router_gates_f32(
        &self,
        input: &Self::Tensor,
        router_weights: &Self::Tensor,
        router_biases: &Self::Tensor,
        batch: usize,
        channels: usize,
        spatial: usize,
        experts: usize,
        top_k: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let router_weights_f = self.f32v(router_weights)?;
        let router_weights = &*router_weights_f;
        let router_biases_f = self.f32v(router_biases)?;
        let router_biases = &*router_biases_f;
        let op = "moe_router_gates_f32";
        if batch == 0 || channels == 0 || spatial == 0 || experts == 0 {
            return Err(shape_err(
                op,
                "batch, channels, spatial, and experts must be nonzero".into(),
            ));
        }
        if input.len() != batch * channels * spatial
            || router_weights.len() != experts * channels
            || router_biases.len() != experts
        {
            return Err(shape_err(
                op,
                format!(
                    "input/weights/biases have lengths {}/{}/{}, expected {}/{}/{}",
                    input.len(),
                    router_weights.len(),
                    router_biases.len(),
                    batch * channels * spatial,
                    experts * channels,
                    experts
                ),
            ));
        }
        let mut logits = self.alloc_uninit(batch * experts)?;
        // SAFETY: one thread per (sample, expert) logit, each uniquely written.
        unsafe {
            self.moe.moe_router_logits(
                &self.stream,
                launch::elementwise(logits.len(), config),
                &input.buf,
                &router_weights.buf,
                &router_biases.buf,
                &mut logits,
                channels,
                spatial,
                experts,
                batch,
                temperature,
            )
        }
        .map_err(launch_err)?;
        let mut gates = self.alloc_uninit(batch * experts)?;
        // SAFETY: one thread per batch row, each uniquely writes `experts` gates.
        unsafe {
            self.moe.moe_router_softmax_topk(
                &self.stream,
                launch::elementwise(batch, config),
                &logits,
                &mut gates,
                batch,
                experts,
                top_k,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gates, &[batch, experts]))
    }

    fn moe_router_pool_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.f32v(input)?;
        let op = "moe_router_pool_f32";
        if batch == 0 || channels == 0 || spatial == 0 {
            return Err(shape_err(
                op,
                "batch, channels, and spatial must be nonzero".into(),
            ));
        }
        if input.len() != batch * channels * spatial {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {}",
                    input.len(),
                    batch * channels * spatial
                ),
            ));
        }
        let mut pooled = self.alloc_uninit(batch * channels)?;
        // SAFETY: one thread uniquely writes one sample/channel descriptor.
        unsafe {
            self.moe.moe_router_pool(
                &self.stream,
                launch::elementwise(batch * channels, config),
                &input.buf,
                &mut pooled,
                channels,
                spatial,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(pooled, &[batch, channels]))
    }

    fn moe_router_pool_backward_f32(
        &self,
        pooled_gradient: &Self::Tensor,
        input_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let pooled_gradient = self.f32v(pooled_gradient)?;
        let op = "moe_router_pool_backward_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] == 0
            || input_shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("input shape must be nonzero NCHW, got {input_shape:?}"),
            ));
        }
        let (batch, channels) = (input_shape[0], input_shape[1]);
        let spatial = input_shape[2] * input_shape[3];
        if pooled_gradient.len() != batch * channels {
            return Err(shape_err(
                op,
                format!(
                    "pooled gradient has {} elements, expected {}",
                    pooled_gradient.len(),
                    batch * channels
                ),
            ));
        }
        let mut input_gradient = self.alloc_uninit(batch * channels * spatial)?;
        // SAFETY: one thread uniquely writes one input-gradient element.
        unsafe {
            self.moe.moe_router_pool_backward(
                &self.stream,
                launch::elementwise(input_gradient.len(), config),
                &pooled_gradient.buf,
                &mut input_gradient,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_gradient, input_shape))
    }

    fn moe_gate_dot_f32(
        &self,
        output_gradient: &Self::Tensor,
        expert_output: &Self::Tensor,
        batch: usize,
        per_image: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_gradient = self.f32v(output_gradient)?;
        let expert_output = self.f32v(expert_output)?;
        let op = "moe_gate_dot_f32";
        let expected = batch * per_image;
        if batch == 0 || per_image == 0 {
            return Err(shape_err(
                op,
                "batch and per-image size must be nonzero".into(),
            ));
        }
        if output_gradient.len() != expected || expert_output.len() != expected {
            return Err(shape_err(
                op,
                format!(
                    "gradient/output lengths are {}/{}, expected {}",
                    output_gradient.len(),
                    expert_output.len(),
                    expected
                ),
            ));
        }
        let mut partial = self.alloc_uninit(batch * 256)?;
        // SAFETY: one thread owns each (sample, reduction-lane) partial.
        unsafe {
            self.moe.moe_gate_dot_partial(
                &self.stream,
                launch::elementwise(batch * 256, config),
                &output_gradient.buf,
                &expert_output.buf,
                &mut partial,
                per_image,
                batch,
            )
        }
        .map_err(launch_err)?;
        let mut dots = self.alloc_uninit(batch)?;
        // SAFETY: one thread uniquely folds each sample's 256 partial sums.
        unsafe {
            self.moe.moe_gate_dot_reduce(
                &self.stream,
                launch::elementwise(batch, config),
                &partial,
                &mut dots,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dots, &[batch]))
    }

    fn moe_gate_column_f32(
        &self,
        gates: &Self::Tensor,
        batch: usize,
        experts: usize,
        expert: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let gates = self.f32v(gates)?;
        let op = "moe_gate_column_f32";
        if batch == 0 || experts == 0 || expert >= experts || gates.len() != batch * experts {
            return Err(shape_err(
                op,
                format!(
                    "invalid gate shape/slot: batch={batch}, experts={experts}, expert={expert}, len={}",
                    gates.len()
                ),
            ));
        }
        let mut column = self.alloc_uninit(batch)?;
        // SAFETY: one thread per sample reads its gate and writes one column element.
        unsafe {
            self.moe.moe_gate_column(
                &self.stream,
                launch::elementwise(batch, config),
                &gates.buf,
                &mut column,
                experts,
                expert,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(column, &[batch]))
    }

    fn moe_router_backward_f32(
        &self,
        gates: &Self::Tensor,
        output_gradient: &Self::Tensor,
        expert_outputs: &[Self::Tensor],
        batch: usize,
        experts: usize,
        per_image: usize,
        load_balance: f32,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let gates = self.f32v(gates)?;
        let output_gradient = self.f32v(output_gradient)?;
        let op = "moe_router_backward_f32";
        if batch == 0 || experts == 0 || per_image == 0 || expert_outputs.len() != experts {
            return Err(shape_err(
                op,
                format!(
                    "invalid dimensions or expert count: {batch}x{experts}x{per_image}, {} outputs",
                    expert_outputs.len()
                ),
            ));
        }
        let gate_count = batch * experts;
        let output_count = batch * per_image;
        if gates.len() != gate_count || output_gradient.len() != output_count {
            return Err(shape_err(
                op,
                format!(
                    "gate/gradient lengths are {}/{}, expected {gate_count}/{output_count}",
                    gates.len(),
                    output_gradient.len()
                ),
            ));
        }
        let mut partial = self.alloc_uninit(batch * 256)?;
        let mut dots = self.alloc_uninit(gate_count)?;
        for (expert, output) in expert_outputs.iter().enumerate() {
            let output = self.f32v(output)?;
            if output.len() != output_count {
                return Err(shape_err(
                    op,
                    format!(
                        "expert {expert} has {} values, expected {output_count}",
                        output.len()
                    ),
                ));
            }
            // SAFETY: one thread owns each (sample, reduction-lane) partial. The
            // workspace is reused on the same stream before folding this expert.
            unsafe {
                self.moe.moe_gate_dot_partial(
                    &self.stream,
                    launch::elementwise(batch * 256, config),
                    &output_gradient.buf,
                    &output.buf,
                    &mut partial,
                    per_image,
                    batch,
                )
            }
            .map_err(launch_err)?;
            // SAFETY: each expert launch writes a disjoint column of the dots matrix.
            unsafe {
                self.moe.moe_gate_dot_reduce_column(
                    &self.stream,
                    launch::elementwise(batch, config),
                    &partial,
                    &mut dots,
                    experts,
                    expert,
                )
            }
            .map_err(launch_err)?;
        }
        let mut dlogit = self.alloc_uninit(gate_count)?;
        // SAFETY: one thread owns one complete batch row of router derivatives.
        unsafe {
            self.moe.moe_router_gate_backward(
                &self.stream,
                launch::elementwise(batch, config),
                &gates.buf,
                &dots,
                &mut dlogit,
                batch,
                experts,
                load_balance,
                temperature,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dlogit, &[batch, experts]))
    }

    fn hyperbolic_foveation_predictor_f32(
        &self,
        input: &Self::Tensor,
        shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let source_f = self.f32v(input)?;
        let source = &*source_f;
        let op = "hyperbolic_foveation_predictor_f32";
        if shape.len() != 4
            || shape.contains(&0)
            || downsample == 0
            || hidden_count == 0
            || param_count < 4
        {
            return Err(shape_err(
                op,
                "invalid source shape or predictor dimensions".to_owned(),
            ));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let input_count = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "source size overflows".to_owned()))?;
        if source.len() != input_count {
            return Err(shape_err(
                op,
                format!("source has {} values, expected {input_count}", source.len()),
            ));
        }
        let pooled_height = (height / downsample).max(1);
        let pooled_width = (width / downsample).max(1);
        let input_dim = channels
            .checked_mul(pooled_height)
            .and_then(|n| n.checked_mul(pooled_width))
            .ok_or_else(|| shape_err(op, "pooled predictor size overflows".to_owned()))?;
        let expected_w1 = hidden_count
            .checked_mul(input_dim)
            .ok_or_else(|| shape_err(op, "first projection size overflows".to_owned()))?;
        let expected_w2 = param_count
            .checked_mul(hidden_count)
            .ok_or_else(|| shape_err(op, "second projection size overflows".to_owned()))?;
        let pooled_count = batch
            .checked_mul(input_dim)
            .ok_or_else(|| shape_err(op, "pooled batch size overflows".to_owned()))?;
        let output_count = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "predictor output size overflows".to_owned()))?;
        if w1.len() != expected_w1
            || b1.len() != hidden_count
            || w2.len() != expected_w2
            || b2.len() != param_count
        {
            return Err(shape_err(
                op,
                "predictor parameter lengths do not match dimensions".to_owned(),
            ));
        }
        let mut pooled = self.alloc_uninit(pooled_count)?;
        // SAFETY: output geometry is bounded by the validated positive source
        // dimensions and each output element owns its tile reduction.
        unsafe {
            self.resample.hyperbolic_predictor_pool(
                &self.stream,
                launch::elementwise(pooled_count, config),
                &source.buf,
                &mut pooled,
                channels,
                height,
                width,
                pooled_height,
                pooled_width,
                downsample,
            )
        }
        .map_err(launch_err)?;
        let pooled = OxideTensorF32::new(pooled, &[batch, input_dim]);
        let w1 = self.upload_f32(&[hidden_count, input_dim], w1)?;
        let b1 = self.upload_f32(&[hidden_count], b1)?;
        let preactivation = self.linear_f32(
            &pooled,
            &w1,
            Some(&b1),
            &[batch, hidden_count],
            batch,
            input_dim,
            hidden_count,
            config,
        )?;
        let hidden = self.activation_f32(&preactivation, F32ActivationKind::Relu, config)?;
        let w2 = self.upload_f32(&[param_count, hidden_count], w2)?;
        let b2 = self.upload_f32(&[param_count], b2)?;
        let raw = self.linear_f32(
            &hidden,
            &w2,
            Some(&b2),
            &[batch, param_count],
            batch,
            hidden_count,
            param_count,
            config,
        )?;
        let raw_f = self.f32v(&raw)?;
        let raw = &*raw_f;
        let mut constrained = self.alloc_uninit(output_count)?;
        // SAFETY: each output element owns one validated raw parameter value.
        unsafe {
            self.resample.hyperbolic_predictor_constrain(
                &self.stream,
                launch::elementwise(output_count, config),
                &raw.buf,
                &mut constrained,
                param_count,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(constrained, &[batch, param_count]))
    }

    fn hyperbolic_foveation_predictor_backward_f32(
        &self,
        pooled: &Self::Tensor,
        hidden_preactivation: &Self::Tensor,
        hidden: &Self::Tensor,
        raw_output: &Self::Tensor,
        constrained_seed: &Self::Tensor,
        w1: &Self::Tensor,
        w2: &Self::Tensor,
        input_shape: &[usize],
        downsample: usize,
        hidden_count: usize,
        param_count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let pooled_f = self.f32v(pooled)?;
        let pooled = &*pooled_f;
        let pre_f = self.f32v(hidden_preactivation)?;
        let pre = &*pre_f;
        let hidden_f = self.f32v(hidden)?;
        let hidden = &*hidden_f;
        let raw_f = self.f32v(raw_output)?;
        let raw = &*raw_f;
        let seed_f = self.f32v(constrained_seed)?;
        let seed = &*seed_f;
        let w1_f = self.f32v(w1)?;
        let w1 = &*w1_f;
        let w2_f = self.f32v(w2)?;
        let w2 = &*w2_f;
        let op = "hyperbolic_foveation_predictor_backward_f32";
        if input_shape.len() != 4
            || input_shape.contains(&0)
            || downsample == 0
            || hidden_count == 0
            || param_count < 4
        {
            return Err(shape_err(op, "expected positive NCHW shape, downsample, hidden count, and at least four parameters".to_owned()));
        }
        let (batch, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let pooled_height = (height / downsample).max(1);
        let pooled_width = (width / downsample).max(1);
        let input_dim = channels
            .checked_mul(pooled_height)
            .and_then(|n| n.checked_mul(pooled_width))
            .ok_or_else(|| shape_err(op, "pooled predictor size overflows".to_owned()))?;
        let input_count = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let hidden_count_batch = batch
            .checked_mul(hidden_count)
            .ok_or_else(|| shape_err(op, "hidden size overflows".to_owned()))?;
        let param_count_batch = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if pooled.len() != batch * input_dim
            || pre.len() != hidden_count_batch
            || hidden.len() != hidden_count_batch
            || raw.len() != param_count_batch
            || seed.len() != param_count_batch
            || w1.len() != hidden_count * input_dim
            || w2.len() != param_count * hidden_count
        {
            return Err(shape_err(
                op,
                "predictor cache/weight lengths do not match dimensions".to_owned(),
            ));
        }
        let source_len = input_count;
        let dw1_len = hidden_count
            .checked_mul(input_dim)
            .and_then(|n| n.checked_add(hidden_count))
            .ok_or_else(|| shape_err(op, "w1 gradient size overflows".to_owned()))?;
        let dw2_len = param_count
            .checked_mul(hidden_count)
            .and_then(|n| n.checked_add(param_count))
            .ok_or_else(|| shape_err(op, "w2 gradient size overflows".to_owned()))?;
        let mut source_grad = self.alloc_uninit(source_len)?;
        let mut w1_grad = self.alloc_uninit(dw1_len)?;
        let mut w2_grad = self.alloc_uninit(dw2_len)?;
        // SAFETY: all caches and weights have validated lengths; each output
        // scalar is written by exactly one thread in each kernel.
        unsafe {
            self.resample.hyperbolic_predictor_backward_source(
                &self.stream,
                launch::elementwise(source_len, config),
                &pre.buf,
                &raw.buf,
                &seed.buf,
                &w1.buf,
                &w2.buf,
                &mut source_grad,
                channels,
                height,
                width,
                downsample,
                pooled_height,
                pooled_width,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_predictor_backward_w1(
                &self.stream,
                launch::elementwise(dw1_len, config),
                &pooled.buf,
                &pre.buf,
                &raw.buf,
                &seed.buf,
                &w2.buf,
                &mut w1_grad,
                batch,
                input_dim,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_predictor_backward_w2(
                &self.stream,
                launch::elementwise(dw2_len, config),
                &hidden.buf,
                &raw.buf,
                &seed.buf,
                &mut w2_grad,
                batch,
                hidden_count,
                param_count,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(source_grad, input_shape),
            OxideTensorF32::new(w1_grad, &[dw1_len]),
            OxideTensorF32::new(w2_grad, &[dw2_len]),
        ))
    }

    fn hyperbolic_newton_grid_f32(
        &self,
        params: &Self::Tensor,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let op = "hyperbolic_newton_grid_f32";
        if batch == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch and spatial dimensions must be positive".to_owned(),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected parameter count 4 or 5, got {param_count}"),
            ));
        }
        let expected_params = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if params.len() != expected_params {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_params} parameters, got {}",
                    params.len()
                ),
            ));
        }
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        let mut grid = self.alloc_uninit(grid_len)?;
        // SAFETY: each output coordinate is owned by one thread and parameter
        // rows are validated to contain four/five values per batch.
        unsafe {
            self.resample.hyperbolic_newton_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &params.buf,
                &mut grid,
                height,
                width,
                param_count,
                alpha,
                p_exp,
                iterations,
                eta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(grid, &[batch, height, width, 2]))
    }

    fn hyperbolic_fixed_point_grid_f32(
        &self,
        params: &Self::Tensor,
        batch: usize,
        height: usize,
        width: usize,
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let op = "hyperbolic_fixed_point_grid_f32";
        if batch == 0 || height == 0 || width == 0 {
            return Err(shape_err(
                op,
                "batch and spatial dimensions must be positive".to_owned(),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected parameter count 4 or 5, got {param_count}"),
            ));
        }
        let expected_params = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        if params.len() != expected_params {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_params} parameters, got {}",
                    params.len()
                ),
            ));
        }
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        let mut grid = self.alloc_uninit(grid_len)?;
        // SAFETY: each thread owns one grid scalar, dimensions/parameter lengths
        // were validated, and every thread reads only its batch parameter row.
        unsafe {
            self.resample.hyperbolic_fixed_point_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &params.buf,
                &mut grid,
                height,
                width,
                param_count,
                alpha,
                p_exp,
                iterations,
                eta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(grid, &[batch, height, width, 2]))
    }

    fn hyperbolic_foveation_warp_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let grid_f = self.f32v(grid)?;
        let grid = &*grid_f;
        let op = "hyperbolic_foveation_warp_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if input.len() != input_len || grid.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "input/grid lengths are {}/{}, expected {input_len}/{grid_len}",
                    input.len(),
                    grid.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(input_len)?;
        // SAFETY: each output owns one element and only reads validated input
        // and the corresponding per-batch/pixel grid coordinates.
        unsafe {
            self.resample.hyperbolic_warp(
                &self.stream,
                launch::elementwise(input_len, config),
                &input.buf,
                &grid.buf,
                &mut output,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn hyperbolic_foveation_warp_backward_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        output_gradient: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let grid_f = self.f32v(grid)?;
        let grid = &*grid_f;
        let dy_f = self.f32v(output_gradient)?;
        let dy = &*dy_f;
        let op = "hyperbolic_foveation_warp_backward_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "input size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if input.len() != input_len || dy.len() != input_len || grid.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "input/gradient/grid lengths are {}/{}/{}, expected {input_len}/{input_len}/{grid_len}",
                    input.len(),
                    dy.len(),
                    grid.len()
                ),
            ));
        }
        let input_gradient = self.alloc(input_len)?;
        let mut grid_gradient = self.alloc_uninit(grid_len)?;
        // SAFETY: dimensions and lengths are validated; input-gradient atomic
        // scatters target a zeroed buffer, and grid-gradient scalars have unique owners.
        unsafe {
            self.resample.hyperbolic_warp_backward_input(
                &self.stream,
                launch::elementwise(input_len, config),
                &grid.buf,
                &dy.buf,
                input_gradient.cu_deviceptr() as *mut f32,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.resample.hyperbolic_warp_backward_grid(
                &self.stream,
                launch::elementwise(grid_len, config),
                &input.buf,
                &grid.buf,
                &dy.buf,
                &mut grid_gradient,
                channels,
                height,
                width,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(input_gradient, shape),
            OxideTensorF32::new(grid_gradient, &[batch, height, width, 2]),
        ))
    }

    fn hyperbolic_foveation_inverse_param_grad_f32(
        &self,
        params: &Self::Tensor,
        grid_gradient: &Self::Tensor,
        shape: &[usize],
        param_count: usize,
        alpha: f32,
        p_exp: f32,
        iterations: usize,
        eta: f32,
        newton: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Option<Self::Tensor>, Self::Error> {
        let params_f = self.f32v(params)?;
        let params = &*params_f;
        let grid_f = self.f32v(grid_gradient)?;
        let grid_gradient = &*grid_f;
        let op = "hyperbolic_foveation_inverse_param_grad_f32";
        if shape.len() != 4 || shape.contains(&0) {
            return Err(shape_err(
                op,
                format!("expected positive NCHW shape, got {shape:?}"),
            ));
        }
        if param_count != 4 && param_count != 5 {
            return Err(shape_err(
                op,
                format!("expected 4 or 5 HF parameters, got {param_count}"),
            ));
        }
        let (batch, height, width) = (shape[0], shape[2], shape[3]);
        let params_len = batch
            .checked_mul(param_count)
            .ok_or_else(|| shape_err(op, "parameter size overflows".to_owned()))?;
        let grid_len = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "grid size overflows".to_owned()))?;
        if params.len() != params_len || grid_gradient.len() != grid_len {
            return Err(shape_err(
                op,
                format!(
                    "parameter/grid lengths are {}/{}, expected {params_len}/{grid_len}",
                    params.len(),
                    grid_gradient.len()
                ),
            ));
        }
        let output_len = batch
            .checked_mul(5)
            .ok_or_else(|| shape_err(op, "output size overflows".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: all tensor lengths and dimensions are checked and each
        // thread owns one output batch/parameter lane.
        if newton {
            unsafe {
                self.resample.hyperbolic_newton_param_grad(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &params.buf,
                    &grid_gradient.buf,
                    &mut output,
                    batch,
                    height,
                    width,
                    param_count,
                    alpha,
                    p_exp,
                    iterations,
                    eta,
                )
            }
            .map_err(launch_err)?;
        } else {
            unsafe {
                self.resample.hyperbolic_fixed_point_param_grad(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &params.buf,
                    &grid_gradient.buf,
                    &mut output,
                    batch,
                    height,
                    width,
                    param_count,
                    alpha,
                    p_exp,
                    iterations,
                    eta,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(Some(OxideTensorF32::new(output, &[batch, 5])))
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
        input_saturations: &Self::Tensor,
        input_exposures: &Self::Tensor,
        apply_hsv: bool,
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
        let input_saturations_f = self.f32v(input_saturations)?;
        let input_saturations = &*input_saturations_f;
        let input_exposures_f = self.f32v(input_exposures)?;
        let input_exposures = &*input_exposures_f;
        let op = "crop_bilinear_f32";
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, h, w]"),
            ));
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
        if input_saturations.len() != batch || input_exposures.len() != batch {
            return Err(shape_err(
                op,
                format!(
                    "expected one saturation/exposure adjustment per batch ({batch}), got {} and {}",
                    input_saturations.len(),
                    input_exposures.len()
                ),
            ));
        }
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}",
                    x.len()
                ),
            ));
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
                &input_saturations.buf,
                &input_exposures.buf,
                apply_hsv,
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
            return Err(shape_err(
                op,
                format!("gradient shape {g:?} is not [batch, {channels}, h, w]"),
            ));
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
        nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )
        .and_then(|b| {
            if b == batch {
                Ok(())
            } else {
                Err(shape_err(
                    op,
                    format!("input batch {b} != gradient batch {batch}"),
                ))
            }
        })?;
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

    fn deform_bilinear_gather_f32(
        &self,
        input: &Self::Tensor,
        sample_y: &Self::Tensor,
        sample_x: &Self::Tensor,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let y_f = self.f32v(sample_y)?;
        let y = &*y_f;
        let x_f = self.f32v(sample_x)?;
        let x = &*x_f;
        let op = "deform_bilinear_gather_f32";
        deform_gather_shape(op, input, y, x, None, geometry)?;
        let mut out = self.alloc_uninit(geometry.output_len())?;
        // SAFETY: one thread per output element; every tap is bounds-checked
        // in the kernel and only reads `input`.
        unsafe {
            self.deform_sample.deform_gather(
                &self.stream,
                launch::elementwise(geometry.output_len(), config),
                &input.buf,
                &y.buf,
                &x.buf,
                &mut out,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
                geometry.samples,
                geometry.output_height,
                geometry.output_width,
            )
        }
        .map_err(launch_err)?;
        let shape = [
            geometry.batch,
            geometry.samples,
            geometry.channels,
            geometry.output_height,
            geometry.output_width,
        ];
        Ok(OxideTensorF32::new(out, &shape))
    }

    fn deform_bilinear_gather_backward_f32(
        &self,
        input: &Self::Tensor,
        sample_y: &Self::Tensor,
        sample_x: &Self::Tensor,
        output_grad: &Self::Tensor,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let y_f = self.f32v(sample_y)?;
        let y = &*y_f;
        let x_f = self.f32v(sample_x)?;
        let x = &*x_f;
        let grad_f = self.f32v(output_grad)?;
        let grad = &*grad_f;
        let op = "deform_bilinear_gather_backward_f32";
        deform_gather_shape(op, input, y, x, Some(grad), geometry)?;
        // `alloc` is zero-filled: the kernel only touches `d_input` through
        // in-image atomic adds, so the zeros are the scatter identity.
        let d_input = self.alloc(geometry.input_len())?;
        let mut d_y = self.alloc_uninit(geometry.sample_len())?;
        let mut d_x = self.alloc_uninit(geometry.sample_len())?;
        // SAFETY: one thread per sample cell owns `d_y`/`d_x`; `d_input` is a
        // live zeroed buffer written only by atomic adds on validated taps.
        unsafe {
            self.deform_sample.deform_gather_backward(
                &self.stream,
                launch::elementwise(geometry.sample_len(), config),
                &input.buf,
                &y.buf,
                &x.buf,
                &grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                &mut d_y,
                &mut d_x,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
                geometry.samples,
                geometry.output_height,
                geometry.output_width,
            )
        }
        .map_err(launch_err)?;
        let input_shape = [
            geometry.batch,
            geometry.channels,
            geometry.input_height,
            geometry.input_width,
        ];
        let sample_shape = [
            geometry.batch,
            geometry.samples,
            geometry.output_height,
            geometry.output_width,
        ];
        Ok((
            OxideTensorF32::new(d_input, &input_shape),
            OxideTensorF32::new(d_y, &sample_shape),
            OxideTensorF32::new(d_x, &sample_shape),
        ))
    }

    fn dcnv4_forward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        desc: lightnet_cuda::host_ops::dcnv4::Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.dcnv4_forward_native(input, weights, biases, desc, config)
    }

    fn dcnv4_backward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        offsets: &Self::Tensor,
        seed: &Self::Tensor,
        desc: lightnet_cuda::host_ops::dcnv4::Dcnv4Composition,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let grad = self.dcnv4_backward_native(input, weights, offsets, seed, desc, config)?;
        Ok((grad.input, grad.offset_weights, grad.offset_biases))
    }

    fn deform_sanitize_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let out = OxideBackend::deform_sanitize_f32(self, x, config)?;
        let _ = shape;
        Ok(out)
    }

    fn deform_offset_stats(
        &self,
        raw: &Self::Tensor,
        per_batch: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::deform_offset_stats(self, raw, per_batch, max_offset, config)
    }

    fn deform_offset_envelope(
        &self,
        raw: &Self::Tensor,
        mean: &Self::Tensor,
        inv_std: &Self::Tensor,
        normalize: &Self::Tensor,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = shape;
        OxideBackend::deform_offset_envelope(
            self, raw, mean, inv_std, normalize, planes, spatial, max_offset, config,
        )
    }

    fn deform_mask_sigmoid(
        &self,
        raw: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = shape;
        OxideBackend::deform_mask_sigmoid(self, raw, config)
    }

    fn dsconv_forward_native_f32(
        &self,
        input: &Self::Tensor,
        offset_weights: &Self::Tensor,
        offset_biases: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::DsConvGeometry,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Option<Result<Self::Tensor, Self::Error>> {
        Some(OxideBackend::dsconv_forward_native_f32(
            self,
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            geometry.batch,
            geometry.channels,
            geometry.filters,
            geometry.input_height,
            geometry.input_width,
            geometry.kernel_size,
            geometry.extend_scope,
            geometry.morph,
            activation,
            config,
        ))
    }

    fn dsconv_backward_native_f32(
        &self,
        input: &Self::Tensor,
        offset_weights: &Self::Tensor,
        offset_biases: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        output_gradient: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::DsConvGeometry,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Option<
        Result<
            (
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
                Self::Tensor,
            ),
            Self::Error,
        >,
    > {
        Some(OxideBackend::dsconv_backward_native_f32(
            self,
            input,
            offset_weights,
            offset_biases,
            weights,
            biases,
            output_gradient,
            geometry.batch,
            geometry.channels,
            geometry.filters,
            geometry.input_height,
            geometry.input_width,
            geometry.kernel_size,
            geometry.extend_scope,
            geometry.morph,
            activation,
            config,
        ))
    }

    fn deform_predictor_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        biases: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        in_h: usize,
        in_w: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.f32v(input)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let per_batch = channels
            .checked_mul(in_h)
            .and_then(|n| n.checked_mul(in_w))
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                shape_err("deform_predictor_f32", "input extent overflows".to_owned())
            })?;
        if input.len() % per_batch != 0 {
            return Err(shape_err(
                "deform_predictor_f32",
                "input length is not a whole batch".to_owned(),
            ));
        }
        let batch = input.len() / per_batch;
        if output_shape != [batch, filters, out_h, out_w] {
            return Err(shape_err(
                "deform_predictor_f32",
                "output shape does not match predictor geometry".to_owned(),
            ));
        }
        OxideBackend::deform_offset_predict(
            self, &input, &weights, &biases, channels, in_h, in_w, filters, size, stride_x,
            stride_y, padding, dilation, out_h, out_w, config,
        )
    }

    fn deform_conv_accumulate_f32(
        &self,
        input: &Self::Tensor,
        offsets: &Self::Tensor,
        masks: &Self::Tensor,
        weights: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let out = OxideBackend::deform_conv_accumulate(
            self,
            input,
            offsets,
            masks,
            weights,
            geometry.channels,
            geometry.filters,
            geometry.channels_per_group,
            geometry.filters_per_group,
            geometry.input_height,
            geometry.input_width,
            geometry.output_height,
            geometry.output_width,
            geometry.size,
            geometry.stride_x,
            geometry.stride_y,
            geometry.padding,
            geometry.dilation,
            use_mask,
            config,
        )?;
        let _ = shape;
        let expected = [
            geometry.batch,
            geometry.filters,
            geometry.output_height,
            geometry.output_width,
        ];
        if out.shape() != expected {
            return Err(shape_err(
                "deform_conv_accumulate_f32",
                format!("output shape {:?} != {expected:?}", out.shape()),
            ));
        }
        Ok(out)
    }

    fn deform_conv_accumulate_backward_f32(
        &self,
        input: &Self::Tensor,
        offsets: &Self::Tensor,
        masks: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::deform_conv_accumulate_backward(
            self,
            input,
            offsets,
            masks,
            weights,
            output_grad,
            geometry,
            use_mask,
            config,
        )
    }

    fn deform_envelope_pre_backward_f32(
        &self,
        raw: &Self::Tensor,
        mean: &Self::Tensor,
        inv_std: &Self::Tensor,
        normalize: &Self::Tensor,
        output_grad: &Self::Tensor,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = shape;
        OxideBackend::deform_envelope_pre_backward(
            self,
            raw,
            mean,
            inv_std,
            normalize,
            output_grad,
            planes,
            spatial,
            max_offset,
            config,
        )
    }

    fn deform_mask_sigmoid_backward_f32(
        &self,
        logits: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = shape;
        OxideBackend::deform_mask_sigmoid_backward(self, logits, output_grad, config)
    }

    fn deform_predictor_backward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        output_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        filters: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::deform_predictor_backward(
            self,
            input,
            weights,
            output_grad,
            channels,
            filters,
            in_h,
            in_w,
            out_h,
            out_w,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            batch,
            config,
        )
    }

    fn deform_scale_input_gradient_f32(
        &self,
        raw: &Self::Tensor,
        grad: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = shape;
        OxideBackend::deform_scale_input_gradient(self, raw, grad, config)
    }

    fn wmhf_haar_dwt_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::wmhf_haar_dwt_f32(self, input, batch, channels, height, width, config)
    }

    fn wmhf_haar_dwt_backward_f32(
        &self,
        ll_grad: &Self::Tensor,
        lh_grad: &Self::Tensor,
        hl_grad: &Self::Tensor,
        hh_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_haar_dwt_backward_f32(
            self, ll_grad, lh_grad, hl_grad, hh_grad, batch, channels, height, width, config,
        )
    }

    fn wmhf_haar_iwt_f32(
        &self,
        ll: &Self::Tensor,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_haar_iwt_f32(
            self, ll, lh, hl, hh, batch, channels, height, width, config,
        )
    }

    fn wmhf_haar_iwt_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::wmhf_haar_iwt_backward_f32(
            self,
            output_grad,
            batch,
            channels,
            height,
            width,
            config,
        )
    }

    fn wmhf_scan4_forward_f32(
        &self,
        input: &Self::Tensor,
        weights: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_scan4_forward_f32(
            self, input, weights, batch, channels, height, width, config,
        )
    }

    fn wmhf_scan4_backward_f32(
        &self,
        input: &Self::Tensor,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::wmhf_scan4_backward_f32(
            self,
            input,
            output_grad,
            weights,
            batch,
            channels,
            height,
            width,
            config,
        )
    }

    fn wmhf_high_frequency_forward_f32(
        &self,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_high_frequency_forward_f32(
            self, lh, hl, hh, batch, channels, height, width, config,
        )
    }

    fn wmhf_high_frequency_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        lh: &Self::Tensor,
        hl: &Self::Tensor,
        hh: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::wmhf_high_frequency_backward_f32(
            self,
            output_grad,
            lh,
            hl,
            hh,
            batch,
            channels,
            height,
            width,
            config,
        )
    }

    fn wmhf_concat_local_f32(
        &self,
        branch3: &Self::Tensor,
        branch5: &Self::Tensor,
        branch7: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_concat_local_f32(
            self, branch3, branch5, branch7, batch, channels, height, width, config,
        )
    }

    fn wmhf_split_channels_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        begin: usize,
        count: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::wmhf_split_channels_f32(
            self, input, batch, channels, height, width, begin, count, config,
        )
    }

    fn vit_patchify_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(usize, usize, Self::Tensor), Self::Error> {
        OxideBackend::vit_patchify_f32(
            self,
            input,
            batch,
            channels,
            height,
            width,
            patch_size,
            patch_stride,
            pad,
            config,
        )
    }

    fn vit_add_pos_embed_f32(
        &self,
        tokens: &Self::Tensor,
        pos: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_add_pos_embed_f32(self, tokens, pos, config)
    }

    fn vit_clamp_feature_f32(
        &self,
        input: &Self::Tensor,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_clamp_feature_f32(self, input, clip, config)
    }

    fn vit_mhc_residual_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_mhc_residual_f32(self, skip, branch, skip_scale, branch_scale, clip, config)
    }

    fn vit_tokens_to_nchw_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_tokens_to_nchw_f32(self, input, batch, out_h, out_w, hidden, config)
    }

    fn vit_nchw_to_tokens_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_nchw_to_tokens_f32(self, input, batch, out_h, out_w, hidden, config)
    }

    fn vit_patch_scatter_f32(
        &self,
        patch_grad: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_patch_scatter_f32(
            self,
            patch_grad,
            batch,
            channels,
            height,
            width,
            patch_size,
            patch_stride,
            pad,
            out_h,
            out_w,
            config,
        )
    }

    fn vit_clamp_backward_f32(
        &self,
        raw: &Self::Tensor,
        grad: &Self::Tensor,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_clamp_backward_f32(self, raw, grad, clip, config)
    }

    fn vit_mhc_residual_backward_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        grad: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        OxideBackend::vit_mhc_residual_backward_f32(
            self,
            skip,
            branch,
            grad,
            skip_scale,
            branch_scale,
            mix,
            post_skip,
            post_branch,
            clip,
            config,
        )
    }

    fn vit_pos_embed_backward_f32(
        &self,
        grad: &Self::Tensor,
        batch: usize,
        table: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        OxideBackend::vit_pos_embed_backward_f32(self, grad, batch, table, config)
    }

    fn clifford_interaction_f32(
        &self, state: &Self::Tensor, context: &Self::Tensor,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.clifford_interaction_native(state, context, batch, channels, spatial, shift, inner, pooled, config)
    }

    fn clifford_interaction_backward_f32(
        &self, state: &Self::Tensor, context: &Self::Tensor, grad: &Self::Tensor,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.clifford_interaction_backward_native(state, context, grad, batch, channels, spatial, shift, inner, pooled, config)
    }

    fn clifford_residual_f32(
        &self, input: &Self::Tensor, mixed: &Self::Tensor, scale: &Self::Tensor,
        mask: &Self::Tensor, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.clifford_residual_native(input, mixed, scale, mask, batch, channels, spatial, config)
    }

    fn clifford_residual_backward_f32(
        &self, mixed: &Self::Tensor, grad: &Self::Tensor, scale: &Self::Tensor,
        mask: &Self::Tensor, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.clifford_residual_backward_native(mixed, grad, scale, mask, batch, channels, spatial, config)
    }

    fn swin_window_partition_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.swin_window_partition_native(
            input,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
            config,
        )
    }

    fn swin_window_unpartition_f32(
        &self,
        windows: &Self::Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.swin_window_unpartition_native(
            windows,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
            clip,
            config,
        )
    }

    fn swin_mhc_residual_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.swin_mhc_residual_native(skip, branch, skip_scale, branch_scale, config)
    }

    fn swin_mhc_residual_backward_f32(
        &self,
        skip: &Self::Tensor,
        branch: &Self::Tensor,
        grad: &Self::Tensor,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.swin_mhc_residual_backward_native(
            skip,
            branch,
            grad,
            skip_scale,
            branch_scale,
            mix,
            post_skip,
            post_branch,
            config,
        )
    }

    fn swin_window_attention_f32(
        &self,
        qkv: &Self::Tensor,
        relative_position_bias: &Self::Tensor,
        mask: &Self::Tensor,
        total_windows: usize,
        spatial_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.swin_window_attention_native(
            qkv,
            relative_position_bias,
            mask,
            total_windows,
            spatial_windows,
            window,
            channels,
            heads,
            config,
        )
    }

    fn swin_window_attention_backward_f32(
        &self,
        qkv: &Self::Tensor,
        probabilities: &Self::Tensor,
        active: &Self::Tensor,
        context_grad: &Self::Tensor,
        total_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.swin_window_attention_backward_native(
            qkv,
            probabilities,
            active,
            context_grad,
            total_windows,
            window,
            channels,
            heads,
            config,
        )
    }

    fn wmhf_apply_gate_f32(
        &self,
        fuse: &Self::Tensor,
        gate: &Self::Tensor,
        projected: &Self::Tensor,
        shortcut: Option<&Self::Tensor>,
        freq_scale: f32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        // The native kernel validates matching operand shapes and keeps `fuse`'s.
        let _ = shape;
        OxideBackend::wmhf_apply_gate_f32(self, fuse, gate, projected, shortcut, freq_scale, config)
    }

    fn wmhf_apply_gate_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        gate: &Self::Tensor,
        projected: &Self::Tensor,
        freq_scale: f32,
        use_shortcut: bool,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let _ = shape;
        OxideBackend::wmhf_apply_gate_backward_f32(
            self,
            output_grad,
            gate,
            projected,
            freq_scale,
            use_shortcut,
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
    ) -> Result<Self::Tensor, Self::Error> {
        if x.is_half() && !reverse {
            let xh = self.as_half(x)?;
            let out = self.upsample_f16(
                &xh,
                output_shape,
                channels,
                input_height,
                input_width,
                stride,
                scale,
                reverse,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "upsample_f32";
        if stride == 0 || channels == 0 || input_height == 0 || input_width == 0 {
            return Err(shape_err(
                op,
                "channels, spatial dims and stride must be > 0".to_owned(),
            ));
        }
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, h, w]"),
            ));
        }
        let (batch, out_h, out_w) = (output_shape[0], output_shape[2], output_shape[3]);
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}",
                    x.len()
                ),
            ));
        }
        if reverse {
            // floor (C++) and ceil (cuTile) sized outputs are both accepted.
            if out_h == 0
                || out_w == 0
                || out_h > input_height.div_ceil(stride)
                || out_w > input_width.div_ceil(stride)
                || out_h < input_height / stride
                || out_w < input_width / stride
            {
                return Err(shape_err(
                    op,
                    format!(
                        "reverse output {out_h}x{out_w} incompatible with input {input_height}x{input_width} / stride {stride}"
                    ),
                ));
            }
        } else if out_h != input_height * stride || out_w != input_width * stride {
            return Err(shape_err(
                op,
                format!(
                    "output {out_h}x{out_w} != input {input_height}x{input_width} * stride {stride}"
                ),
            ));
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
            return self.upsample_backward_amp(
                output_grad,
                input_shape,
                channels,
                output_height,
                output_width,
                stride,
                scale,
                config,
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "upsample_backward_f32";
        if stride == 0 || input_shape.len() != 4 || input_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("bad stride {stride} or input shape {input_shape:?}"),
            ));
        }
        let (batch, in_h, in_w) = (input_shape[0], input_shape[2], input_shape[3]);
        if in_h == 0 || in_w == 0 || output_height == 0 || output_width == 0 {
            return Err(shape_err(op, "spatial dims must be > 0".to_owned()));
        }
        if output_grad.len() != batch * channels * output_height * output_width {
            return Err(shape_err(
                op,
                format!(
                    "gradient has {} elements, expected {batch}*{channels}*{output_height}*{output_width}",
                    output_grad.len()
                ),
            ));
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
            return Err(shape_err(
                op,
                format!(
                    "output {output_height}x{output_width} fits neither expand nor reverse of input {in_h}x{in_w}, stride {stride}"
                ),
            ));
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
        if batch == 0
            || x.len() % batch != 0
            || out_len % batch != 0
            || out_len == 0
            || x.is_empty()
        {
            return Err(shape_err(
                op,
                format!(
                    "batch {batch}, {} source elements and output shape {output_shape:?} are inconsistent",
                    x.len()
                ),
            ));
        }
        let (from_len, to_len) = (x.len() / batch, out_len / batch);
        let mut out = self.alloc_uninit(out_len)?;
        // SAFETY: one thread per output element; reads stay inside `x`
        // (`i < from_len`, `b < batch`).
        unsafe {
            self.shortcut.shortcut_fit(
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

    fn flash_attention_forward_f32(
        &self,
        qkv: &Self::Tensor,
        output_shape: &[usize],
        lse_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let op = "flash_attention_forward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let lse_len = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "LSE length overflows usize".to_owned()))?;
        if output_shape != [batch, tokens, channels] {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} must be [batch, tokens, channels]"),
            ));
        }
        if lse_shape != [batch, heads, tokens] {
            return Err(shape_err(
                op,
                format!("LSE shape {lse_shape:?} must be [batch, heads, tokens]"),
            ));
        }
        let qkv_f = self.f32v(qkv)?;
        let qkv = &*qkv_f;
        if qkv.len() != qkv_len {
            return Err(shape_err(
                op,
                format!("QKV has {} values, expected {qkv_len}", qkv.len()),
            ));
        }
        let head_dim = channels / heads;
        if head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} exceeds the kernel limit 128"),
            ));
        }
        let mut out = self.alloc_uninit(output_len)?;
        let mut lse = self.alloc_uninit(lse_len)?;
        // SAFETY: one query thread writes its unique head-channel interval in
        // `out` and its unique flattened query index in `lse`.
        unsafe {
            self.attention.flash_attention_forward(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv.buf,
                &mut out,
                &mut lse,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(out, output_shape),
            OxideTensorF32::new(lse, lse_shape),
        ))
    }

    fn laya_head_prepare_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        type_embedding: &[f32],
        hidden: usize,
        qtype: i32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let op = "laya_head_prepare_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] != 1
            || input_shape[3] == 0
            || hidden == 0
            || !((0..=2).contains(&qtype) || qtype == -1)
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}, hidden={hidden}, qtype={qtype}"),
            ));
        }
        let (batch, channels, sequence) = (input_shape[0], input_shape[1], input_shape[3]);
        let expected_channels = hidden
            .checked_add(if qtype == -1 { 2 } else { 1 })
            .ok_or_else(|| shape_err(op, "channel count overflows".to_owned()))?;
        if channels != expected_channels || type_embedding.len() != 3 * hidden {
            return Err(shape_err(
                op,
                format!(
                    "expected {expected_channels} input channels and {} type embedding values",
                    3 * hidden
                ),
            ));
        }
        let input = self.f32v(input)?;
        if input.shape() != input_shape {
            return Err(shape_err(
                op,
                format!(
                    "input tensor shape {:?} does not match {input_shape:?}",
                    input.shape()
                ),
            ));
        }
        let input_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(sequence))
            .ok_or_else(|| shape_err(op, "input length overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(sequence)
            .and_then(|n| n.checked_mul(hidden))
            .ok_or_else(|| shape_err(op, "token output length overflows".to_owned()))?;
        let mask_len = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "mask length overflows".to_owned()))?;
        if input.len() != input_len {
            return Err(shape_err(
                op,
                format!("input has {} values, expected {input_len}", input.len()),
            ));
        }
        let type_embedding = self.upload_f32(&[3, hidden], type_embedding)?;
        let mut tokens = self.alloc_uninit(output_len)?;
        let mut mask = self.alloc_uninit(mask_len)?;
        // SAFETY: validated NCHW extents bound each input/type-embedding read;
        // each thread writes a distinct token feature and h=0 writes one mask.
        unsafe {
            self.laya.laya_head_prepare(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &type_embedding.buf,
                &mut tokens,
                &mut mask,
                hidden,
                channels,
                sequence,
                qtype,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(tokens, &[batch * sequence, hidden]),
            OxideTensorF32::new(mask, &[batch, sequence]),
        ))
    }

    fn laya_head_pack_f32(
        &self,
        tokens: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_head_pack_f32";
        let expected_len = batch
            .checked_mul(sequence)
            .and_then(|n| n.checked_mul(hidden))
            .ok_or_else(|| shape_err(op, "tensor length overflows".to_owned()))?;
        if batch == 0
            || sequence == 0
            || hidden == 0
            || output_shape != [batch, hidden, 1, sequence]
        {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let tokens = self.f32v(tokens)?;
        if tokens.shape() != [batch * sequence, hidden] || tokens.len() != expected_len {
            return Err(shape_err(
                op,
                format!(
                    "token tensor has {} values, expected {expected_len}",
                    tokens.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(expected_len)?;
        // SAFETY: checked equal lengths make the channel-major index map valid.
        unsafe {
            self.laya.laya_head_pack(
                &self.stream,
                launch::elementwise(expected_len, config),
                &tokens.buf,
                &mut output,
                sequence,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_scorer_unpack_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        hidden: usize,
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_scorer_unpack_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] != hidden
            || input_shape[2] != 1
            || input_shape[3] != options
            || hidden == 0
            || options == 0
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}"),
            ));
        }
        let batch = input_shape[0];
        let len = batch
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(options))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let rows = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != input_shape || input.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length mismatch: {:?}/{}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: exact tensor extents bound the channel-major gather indices.
        unsafe {
            self.laya.laya_scorer_unpack(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                hidden,
                options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[rows, hidden]))
    }

    fn laya_scorer_pack_f32(
        &self,
        logits: &Self::Tensor,
        batch: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_scorer_pack_f32";
        let len = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if batch == 0 || options == 0 || output_shape != [batch, 1, 1, options] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let logits = self.f32v(logits)?;
        if logits.shape() != [len, 1] || logits.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "logit tensor shape/length mismatch: {:?}/{}",
                    logits.shape(),
                    logits.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: logits/output lengths are equal and every thread owns one slot.
        unsafe {
            self.laya.laya_scorer_pack(
                &self.stream,
                launch::elementwise(len, config),
                &logits.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_encoder_embed_f32(
        &self,
        input: &Self::Tensor,
        embeddings: &Self::Tensor,
        input_shape: &[usize],
        hidden: usize,
        vocab: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        let op = "laya_encoder_embed_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] != 2
            || input_shape[2] != 1
            || input_shape[3] == 0
            || hidden == 0
            || vocab == 0
        {
            return Err(shape_err(
                op,
                format!("invalid input shape {input_shape:?}"),
            ));
        }
        let batch = input_shape[0];
        let sequence = input_shape[3];
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "embedding length overflows usize".to_owned()))?;
        let input_len = batch
            .checked_mul(2)
            .and_then(|n| n.checked_mul(sequence))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let input = self.f32v(input)?;
        let embeddings = self.f32v(embeddings)?;
        if input.shape() != input_shape
            || input.len() != input_len
            || embeddings.shape() != [vocab, hidden]
            || embeddings.len() != vocab * hidden
        {
            return Err(shape_err(
                op,
                "input/embedding extents do not match".to_owned(),
            ));
        }
        let mut tokens = self.alloc_uninit(len)?;
        let mut mask = self.alloc_uninit(rows)?;
        // SAFETY: exact checked lengths bound each clamped id and mask read.
        unsafe {
            self.laya.laya_encoder_embed(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &embeddings.buf,
                &mut tokens,
                &mut mask,
                hidden,
                sequence,
                vocab,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(tokens, &[rows, hidden]),
            OxideTensorF32::new(mask, &[batch, sequence]),
        ))
    }

    fn laya_encoder_attention_f32(
        &self,
        qkv: &Self::Tensor,
        mask: &Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        heads: usize,
        sliding: bool,
        window_radius: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_encoder_attention_f32";
        if batch == 0 || sequence == 0 || hidden == 0 || heads == 0 || hidden % heads != 0 {
            return Err(shape_err(
                op,
                "invalid batch/sequence/hidden/head dimensions".to_owned(),
            ));
        }
        let head_dim = hidden / heads;
        if head_dim % 2 != 0 || head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} must be even and <=128"),
            ));
        }
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if output_shape != [rows, hidden] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let mask_len = rows;
        let table_len = sequence
            .checked_mul(head_dim / 2)
            .ok_or_else(|| shape_err(op, "RoPE table length overflows usize".to_owned()))?;
        let qkv = self.f32v(qkv)?;
        let mask = self.f32v(mask)?;
        let cos = self.f32v(cos)?;
        let sin = self.f32v(sin)?;
        if qkv.shape() != [rows, 3 * hidden]
            || qkv.len() != qkv_len
            || mask.shape() != [batch, sequence]
            || mask.len() != mask_len
            || cos.shape() != [sequence, head_dim / 2]
            || cos.len() != table_len
            || sin.shape() != [sequence, head_dim / 2]
            || sin.len() != table_len
        {
            return Err(shape_err(
                op,
                "QKV/mask/RoPE extents do not match".to_owned(),
            ));
        }
        let query_heads = rows
            .checked_mul(heads)
            .ok_or_else(|| shape_err(op, "query count overflows usize".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: validated dimensions bound every QKV, mask, RoPE and output access.
        unsafe {
            self.laya.laya_encoder_attention(
                &self.stream,
                launch::elementwise(query_heads, config),
                &qkv.buf,
                &mask.buf,
                &cos.buf,
                &sin.buf,
                &mut output,
                sequence,
                hidden,
                heads,
                window_radius,
                sliding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_encoder_geglu_f32(
        &self,
        input: &Self::Tensor,
        rows: usize,
        intermediate: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_encoder_geglu_f32";
        let input_len = rows
            .checked_mul(intermediate)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| shape_err(op, "input length overflows usize".to_owned()))?;
        let output_len = rows
            .checked_mul(intermediate)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if rows == 0 || intermediate == 0 || output_shape != [rows, intermediate] {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let input = self.f32v(input)?;
        if input.shape() != [rows, 2 * intermediate] || input.len() != input_len {
            return Err(shape_err(op, "GeGLU input extent mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: checked 2I input and I output rows bound every chunk access.
        unsafe {
            self.laya.laya_encoder_geglu(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                intermediate,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_encoder_pack_f32(
        &self,
        tokens: &Self::Tensor,
        batch: usize,
        sequence: usize,
        hidden: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_encoder_pack_f32";
        let rows = batch
            .checked_mul(sequence)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let len = rows
            .checked_mul(hidden)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        if batch == 0
            || sequence == 0
            || hidden == 0
            || output_shape != [batch, hidden, 1, sequence]
        {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let tokens = self.f32v(tokens)?;
        if tokens.shape() != [rows, hidden] || tokens.len() != len {
            return Err(shape_err(op, "token tensor extent mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: equal checked extents bound the token/channel-major permutation.
        unsafe {
            self.laya.laya_encoder_pack(
                &self.stream,
                launch::elementwise(len, config),
                &tokens.buf,
                &mut output,
                sequence,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_scorer_seed_rows_f32(
        &self,
        seed: &Self::Tensor,
        batch: usize,
        options: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_scorer_seed_rows_f32";
        let len = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "seed length overflows usize".to_owned()))?;
        if batch == 0 || options == 0 {
            return Err(shape_err(op, "batch/options must be positive".to_owned()));
        }
        let seed = self.f32v(seed)?;
        if seed.shape() != [batch, 1, 1, options] || seed.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "seed tensor shape/length mismatch: {:?}/{}",
                    seed.shape(),
                    seed.len()
                ),
            ));
        }
        Ok(OxideTensorF32 {
            buf: seed.buf.clone(),
            shape: vec![len, 1],
            half: None,
        })
    }

    fn laya_scorer_pack_input_grad_f32(
        &self,
        rows: &Self::Tensor,
        batch: usize,
        hidden: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_scorer_pack_input_grad_f32";
        let len = batch
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(options))
            .ok_or_else(|| shape_err(op, "gradient length overflows usize".to_owned()))?;
        let rows_count = batch
            .checked_mul(options)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        if batch == 0 || hidden == 0 || options == 0 || output_shape != [batch, hidden, 1, options]
        {
            return Err(shape_err(
                op,
                format!("invalid output shape {output_shape:?}"),
            ));
        }
        let rows = self.f32v(rows)?;
        if rows.shape() != [rows_count, hidden] || rows.len() != len {
            return Err(shape_err(
                op,
                format!(
                    "row tensor shape/length mismatch: {:?}/{}",
                    rows.shape(),
                    rows.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: equal extents bound all row/channel/option transpose indices.
        unsafe {
            self.laya.laya_scorer_pack_input_grad(
                &self.stream,
                launch::elementwise(len, config),
                &rows.buf,
                &mut output,
                hidden,
                options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn laya_head_attention_f32(
        &self,
        qkv: &Self::Tensor,
        mask: &Self::Tensor,
        output_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_head_attention_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        if output_shape != [rows, channels] {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} must be [batch*tokens, channels]"),
            ));
        }
        let output_len = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "output length overflows usize".to_owned()))?;
        let qkv_len = output_len
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let qkv_f = self.f32v(qkv)?;
        let mask_f = self.f32v(mask)?;
        if qkv_f.shape() != [rows, 3 * channels]
            || mask_f.shape() != [batch, tokens]
            || qkv_f.len() != qkv_len
            || mask_f.len() != rows
        {
            return Err(shape_err(
                op,
                format!(
                    "QKV/mask lengths are {}/{}, expected {qkv_len}/{rows}",
                    qkv_f.len(),
                    mask_f.len()
                ),
            ));
        }
        let head_dim = channels / heads;
        if head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} exceeds the kernel limit 128"),
            ));
        }
        let query_count = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "query count overflows usize".to_owned()))?;
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: checked dimensions bound every Q/K/V/mask access, and one
        // query/head thread exclusively writes its output channel interval.
        unsafe {
            self.laya.laya_head_attention_forward(
                &self.stream,
                launch::elementwise(query_count, config),
                &qkv_f.buf,
                &mask_f.buf,
                &mut output,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    fn flash_attention_backward_f32(
        &self,
        qkv: &Self::Tensor,
        context: &Self::Tensor,
        output_grad: &Self::Tensor,
        lse: &Self::Tensor,
        qkv_grad_shape: &[usize],
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "flash_attention_backward_f32";
        if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
            return Err(shape_err(
                op,
                "dimensions must be positive and channels divisible by heads".to_owned(),
            ));
        }
        let rows = batch
            .checked_mul(tokens)
            .ok_or_else(|| shape_err(op, "row count overflows usize".to_owned()))?;
        let elements = rows
            .checked_mul(channels)
            .ok_or_else(|| shape_err(op, "attention size overflows usize".to_owned()))?;
        let qkv_len = elements
            .checked_mul(3)
            .ok_or_else(|| shape_err(op, "QKV length overflows usize".to_owned()))?;
        let lse_len = batch
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(tokens))
            .ok_or_else(|| shape_err(op, "LSE length overflows usize".to_owned()))?;
        if qkv_grad_shape != [batch, tokens, 3 * channels] {
            return Err(shape_err(
                op,
                format!("dQKV shape {qkv_grad_shape:?} must be [batch, tokens, 3*channels]"),
            ));
        }
        let qkv_f = self.f32v(qkv)?;
        let context_f = self.f32v(context)?;
        let grad_f = self.f32v(output_grad)?;
        let lse_f = self.f32v(lse)?;
        if qkv_f.len() != qkv_len
            || context_f.len() != elements
            || grad_f.len() != elements
            || lse_f.len() != lse_len
        {
            return Err(shape_err(
                op,
                format!(
                    "QKV/context/gradient/LSE lengths are {}/{}/{}/{}, expected {qkv_len}/{elements}/{elements}/{lse_len}",
                    qkv_f.len(),
                    context_f.len(),
                    grad_f.len(),
                    lse_f.len()
                ),
            ));
        }
        let head_dim = channels / heads;
        if head_dim > 128 {
            return Err(shape_err(
                op,
                format!("head dimension {head_dim} exceeds the kernel limit 128"),
            ));
        }
        let mut dqkv = self.alloc_uninit(qkv_len)?;
        let mut row_dot = self.alloc_uninit(lse_len)?;
        // SAFETY: each query/head thread writes its Q row and D_i scalar.
        unsafe {
            self.attention.flash_attention_backward_dq(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv_f.buf,
                &context_f.buf,
                &grad_f.buf,
                &lse_f.buf,
                &mut dqkv,
                &mut row_dot,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: each key/head thread writes its K and V rows, disjoint from dQ
        // and from every other key/head row in the fused gradient buffer.
        unsafe {
            self.attention.flash_attention_backward_dkdv(
                &self.stream,
                launch::elementwise(lse_len, ElementwiseLaunchConfig::default()),
                &qkv_f.buf,
                &grad_f.buf,
                &lse_f.buf,
                &row_dot,
                &mut dqkv,
                batch,
                tokens,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dqkv, qkv_grad_shape))
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
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", x.len()),
            ));
        }
        if group_size == 0 || x.is_empty() || x.len() % group_size != 0 {
            return Err(shape_err(
                op,
                format!(
                    "length {} is not a multiple of group size {group_size}",
                    x.len()
                ),
            ));
        }
        let n_groups = x.len() / group_size;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.softmax.softmax_forward(
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
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", output.len()),
            ));
        }
        if group_size == 0 || output.is_empty() || output.len() % group_size != 0 {
            return Err(shape_err(
                op,
                format!(
                    "length {} is not a multiple of group size {group_size}",
                    output.len()
                ),
            ));
        }
        let n_groups = output.len() / group_size;
        let mut dx = self.alloc_uninit(output.len())?;
        // SAFETY: one thread per group, each owning its `group_size` outputs.
        unsafe {
            self.softmax.softmax_backward(
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

    fn centernet_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "centernet_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] < classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible CenterNet shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.centernet_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn modern_yolo_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "modern_yolo_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] <= classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible modern-YOLO shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.modern_yolo_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn segmentation_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "segmentation_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] != classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible segmentation shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let pixels = shape[0] * spatial;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions ensure one thread per batch/pixel owns
        // exactly one slot in every class plane.
        unsafe {
            self.dense_loss.segmentation_forward(
                &self.stream,
                launch::elementwise(pixels, config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
                temperature,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn jev_ce_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "jev_ce_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || groups == 0
            || candidates == 0
            || groups
                .checked_mul(candidates)
                .and_then(|per_sample| per_sample.checked_mul(shape[0]))
                != Some(x.len())
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible JevCe shape/groups/candidates: {shape:?}, {groups}, {candidates}"
                ),
            ));
        }
        let work = shape[0] * groups;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: every sample/group owns a disjoint candidate range in the
        // selected storage layout.
        unsafe {
            self.dense_loss.jev_ce_forward(
                &self.stream,
                launch::elementwise(work, config),
                &x.buf,
                &mut output,
                groups,
                candidates,
                temperature,
                channel_major as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn laya_loss_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        max_options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "laya_loss_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || max_options == 0
            || max_options.checked_mul(shape[0]) != Some(x.len())
        {
            return Err(shape_err(
                op,
                format!("incompatible LayaLoss shape/options: {shape:?}, {max_options}"),
            ));
        }
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated batch/options spans are disjoint; one thread owns
        // one complete sample row.
        unsafe {
            self.dense_loss.laya_loss_forward(
                &self.stream,
                launch::elementwise(shape[0], config),
                &x.buf,
                &mut output,
                max_options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn region_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "region_forward_f32";
        if shape.len() != 4
            || shape[1] != anchors.saturating_mul(entries)
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || anchors == 0
            || entries < classes.saturating_add(5)
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible Region shape/parameters: {shape:?}, anchors={anchors}, entries={entries}, classes={classes}"
                ),
            ));
        }
        let spatial = shape[2] * shape[3];
        let groups = shape[0] * spatial * anchors;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: shape and Region layout checks establish valid input/output
        // rows; one thread writes one disjoint anchor/location row.
        unsafe {
            self.dense_loss.region_forward(
                &self.stream,
                launch::elementwise(groups, config),
                &x.buf,
                &mut output,
                spatial,
                anchors,
                entries,
                classes,
                softmax_classes as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn region_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "region_backward_f32";
        if shape.len() != 4
            || shape[1] != anchors.saturating_mul(entries)
            || shape.iter().product::<usize>() != output.len()
            || output.shape() != shape
            || output_grad.shape() != shape
            || output_grad.len() != output.len()
            || anchors == 0
            || entries < classes.saturating_add(5)
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible Region shape/parameters: {shape:?}, anchors={anchors}, entries={entries}, classes={classes}"
                ),
            ));
        }
        let spatial = shape[2] * shape[3];
        let groups = shape[0] * spatial * anchors;
        let mut input_grad = self.alloc_uninit(output.len())?;
        // SAFETY: each anchor/location thread writes unique channel-major input
        // slots, and the output/gradient storage shapes were checked above.
        unsafe {
            self.dense_loss.region_backward(
                &self.stream,
                launch::elementwise(groups, config),
                &output.buf,
                &output_grad.buf,
                &mut input_grad,
                spatial,
                anchors,
                entries,
                classes,
                softmax_classes as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_grad, shape))
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
        self.reorg_launch(
            "reorg_f32",
            x,
            input_shape,
            output_shape,
            stride,
            reverse,
            false,
            config,
        )
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

    fn channel_shuffle_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        if shape.len() != 4
            || groups == 0
            || shape[1] % groups != 0
            || x.len() != shape.iter().product::<usize>()
        {
            return Err(shape_err(
                "channel_shuffle_f32",
                format!(
                    "bad shape {shape:?} / groups {groups} for {} elements",
                    x.len()
                ),
            ));
        }
        let (rows, cols) = if inverse {
            (shape[1] / groups, groups)
        } else {
            (groups, shape[1] / groups)
        };
        let len = shape[2] * shape[3];
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; every source index is < x.len() by the shape check.
        unsafe {
            self.channel_shuffle.channel_permute(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                rows,
                cols,
                len,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    fn laya_act_head_backward_f32(
        &self,
        inputs: &lightnet_cuda::LayaActBackwardInputs<'_, Self::Tensor>,
        layout: &lightnet_cuda::host_ops::laya_act::MlpLayout,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::LayaActHeadGrad<Self::Tensor>, Self::Error> {
        let op = "laya_act_head_backward_f32";
        let d = layout.features();
        if inputs.head.len() != d.head_len
            || inputs.scorer.len() != d.logits_len
            || inputs.markers.len() != d.marker_len
            || inputs.fc1_w.len() != layout.fc1_len()
            || inputs.fc1_b.len() != layout.fc_dim()
            || inputs.fc2_w.len() != layout.fc2_len()
            || inputs.seeds.len() != layout.seed_len()
        {
            return Err(shape_err(
                op,
                "act-head backward operand storage mismatch".to_owned(),
            ));
        }
        if inputs.fc1_w.shape() != [layout.fc_dim(), layout.inputs()]
            || inputs.fc1_b.shape() != [layout.fc_dim()]
            || inputs.fc2_w.shape() != [layout.n_act(), layout.fc_dim()]
            || (inputs.seeds.shape() != [d.batch, layout.n_act()]
                && inputs.seeds.shape() != [d.batch, layout.n_act(), 1, 1])
        {
            return Err(shape_err(
                op,
                "act-head backward matrix/seed metadata mismatch".to_owned(),
            ));
        }
        // Shared composition dispatches back into native Oxide overrides;
        // all sources are checked before any math, and no data is downloaded.
        lightnet_cuda::laya_act::backward_composed(self, inputs, layout, config)
    }

    fn laya_act_cls_scatter_f32(
        &self,
        dcls: &Self::Tensor,
        head_grad: &Self::Tensor,
        batch: usize,
        hidden: usize,
        tokens: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_act_cls_scatter_f32";
        let cls_len = batch
            .checked_mul(hidden)
            .filter(|&len| batch > 0 && hidden > 0 && len <= i32::MAX as usize)
            .ok_or_else(|| shape_err(op, "invalid batch/hidden dimensions".to_owned()))?;
        let full_len = cls_len
            .checked_mul(tokens)
            .filter(|&len| tokens > 0 && len <= i32::MAX as usize)
            .ok_or_else(|| shape_err(op, "invalid token count/storage".to_owned()))?;
        if dcls.len() != cls_len || head_grad.len() != full_len {
            return Err(shape_err(
                op,
                format!(
                    "storage mismatch: expected dcls={cls_len}, head_grad={full_len}; got dcls={}, head_grad={}",
                    dcls.len(),
                    head_grad.len()
                ),
            ));
        }
        if (dcls.shape() != [batch, hidden] && dcls.shape() != [batch, hidden, 1, 1])
            || head_grad.shape() != [batch, hidden, 1, tokens]
        {
            return Err(shape_err(
                op,
                "expected dcls [B,H] and head gradient [B,H,1,T]".to_owned(),
            ));
        }
        let dcls_f = self.f32v(dcls)?;
        let head_grad_f = self.f32v(head_grad)?;
        let mut output = self.alloc_uninit(full_len)?;
        // SAFETY: dimensions/storage are checked, and each thread writes one
        // unique output element while reading only in-bounds source values.
        unsafe {
            self.laya.laya_act_cls_scatter(
                &self.stream,
                launch::elementwise(full_len, config),
                &dcls_f.buf,
                &head_grad_f.buf,
                &mut output,
                tokens,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, hidden, 1, tokens]))
    }

    fn laya_act_features_f32(
        &self,
        head: &Self::Tensor,
        scorer: &Self::Tensor,
        markers: &Self::Tensor,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "laya_act_features_f32";
        let d =
            lightnet_cuda::host_ops::laya_act::Layout::new(head_shape, scorer_shape, marker_shape)
                .map_err(|err| shape_err(op, err))?;
        if head.len() != d.head_len || scorer.len() != d.logits_len || markers.len() != d.marker_len
        {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let head_f = self.f32v(head)?;
        let scorer_f = self.f32v(scorer)?;
        let markers_f = self.f32v(markers)?;
        let mut output = self.alloc_uninit(d.output_len)?;
        // SAFETY: positive checked shapes/products bound all source reads;
        // each sample thread owns an entire disjoint output row.
        unsafe {
            self.laya.laya_act_features(
                &self.stream,
                launch::elementwise(d.batch, config),
                &head_f.buf,
                &scorer_f.buf,
                &markers_f.buf,
                &mut output,
                d.hidden,
                d.options,
                d.marker_channels,
                d.head_width,
                d.marker_width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &d.output_shape()))
    }

    fn marker_gather_f32(
        &self,
        features: &Self::Tensor,
        markers: &Self::Tensor,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "marker_gather_f32";
        let d = lightnet_cuda::host_ops::marker_gather::Layout::new(
            feature_shape,
            marker_shape,
            options,
        )
        .map_err(|err| shape_err(op, err))?;
        if features.len() != d.input_len || markers.len() != d.markers_len {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let features_f = self.f32v(features)?;
        let markers_f = self.f32v(markers)?;
        let mut output = self.alloc_uninit(d.output_len)?;
        // SAFETY: checked dimensions/storage bound every feature and marker index;
        // one thread owns each output element.
        unsafe {
            self.laya.marker_gather(
                &self.stream,
                launch::elementwise(d.output_len, config),
                &features_f.buf,
                &markers_f.buf,
                &mut output,
                d.hidden,
                d.sequence,
                d.marker_channels,
                d.options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &d.output_shape()))
    }

    fn marker_gather_backward_f32(
        &self,
        markers: &Self::Tensor,
        output_grad: &Self::Tensor,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let op = "marker_gather_backward_f32";
        let d = lightnet_cuda::host_ops::marker_gather::Layout::new(
            feature_shape,
            marker_shape,
            options,
        )
        .map_err(|err| shape_err(op, err))?;
        if markers.len() != d.markers_len || output_grad.len() != d.output_len {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let markers_f = self.f32v(markers)?;
        let grad_f = self.f32v(output_grad)?;
        let mut dx = self.alloc_uninit(d.input_len)?;
        // SAFETY: validated layout bounds reads and one thread owns each dx.
        unsafe {
            self.laya.marker_gather_backward(
                &self.stream,
                launch::elementwise(d.input_len, config),
                &markers_f.buf,
                &grad_f.buf,
                &mut dx,
                d.hidden,
                d.sequence,
                d.marker_channels,
                d.options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, feature_shape))
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
            return Err(shape_err(
                op,
                format!("probability {probability} not in [0, 1) or empty input"),
            ));
        }
        if shape.iter().product::<usize>() != x.len() {
            return Err(shape_err(
                op,
                format!("shape {shape:?} does not match {} elements", x.len()),
            ));
        }
        let mut out = self.alloc_uninit(x.len())?;
        let mut mask = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `out` and `mask` slot.
        unsafe {
            self.dropout.dropout_forward(
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
        Ok((
            OxideTensorF32::new(out, shape),
            OxideTensorF32::new(mask, shape),
        ))
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
            return Err(shape_err(
                "dropout_backward_f32",
                format!("shape {shape:?} does not match {} elements", mask.len()),
            ));
        }
        let mut dx = self.alloc_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.dropout.dropout_backward(
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
            return Err(shape_err(
                op,
                format!("shape {shape:?} is not NCHW of {} elements", x.len()),
            ));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if block_size == 0 || block_size > w.min(h) || !(0.0..=1.0).contains(&block_prob) {
            return Err(shape_err(
                op,
                format!(
                    "block_size {block_size} must be in 1..={} and prob {block_prob} in [0, 1]",
                    w.min(h)
                ),
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
            self.dropout.dropblock_plane(
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
            self.dropout.dropblock_scales(
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
            self.dropout.dropblock_apply(
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
        Ok((
            OxideTensorF32::new(out, shape),
            OxideTensorF32::new(mask, shape),
        ))
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
            return Err(shape_err(
                "constrain_f32",
                format!("shape {shape:?} does not match {} elements", x.len()),
            ));
        }
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own slot.
        unsafe {
            self.layers_extra.constrain(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                limit,
            )
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
            return Err(shape_err(
                op,
                format!("filter size {f_size} does not divide {} elements", x.len()),
            ));
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

    fn foveation_film_forward_f32(
        &self,
        input: &Self::Tensor,
        hf_params: &Self::Tensor,
        w_embed: &Self::Tensor,
        b_embed: &Self::Tensor,
        w_gamma: &Self::Tensor,
        b_gamma: &Self::Tensor,
        w_beta: &Self::Tensor,
        b_beta: &Self::Tensor,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_forward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let (batch, channels) = (shape[0], shape[1]);
        if param_count == 0 || embed_dim == 0 {
            return Err(OxideError::Shape(
                "foveation_film_forward_f32: param_count and embed_dim must be positive".to_owned(),
            ));
        }
        let embedding_pre = self.linear_f32(
            hf_params,
            w_embed,
            Some(b_embed),
            &[batch, embed_dim],
            batch,
            param_count,
            embed_dim,
            config,
        )?;
        let embedding = self.activation_f32(&embedding_pre, F32ActivationKind::Relu, config)?;
        let gamma = self.linear_f32(
            &embedding,
            w_gamma,
            Some(b_gamma),
            &[batch, channels],
            batch,
            embed_dim,
            channels,
            config,
        )?;
        let beta = self.linear_f32(
            &embedding,
            w_beta,
            Some(b_beta),
            &[batch, channels],
            batch,
            embed_dim,
            channels,
            config,
        )?;
        let output = self.foveation_film_affine_f32(input, &gamma, &beta, shape, config)?;
        Ok((output, embedding, gamma, beta))
    }

    fn foveation_film_affine_f32(
        &self,
        input: &Self::Tensor,
        gamma: &Self::Tensor,
        beta: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let input = self.f32v(input)?;
        let input = &*input;
        let gamma = self.f32v(gamma)?;
        let gamma = &*gamma;
        let beta = self.f32v(beta)?;
        let beta = &*beta;
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_affine_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: spatial extent overflows".to_owned())
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: channel extent overflows".to_owned())
        })?;
        let expected = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape("foveation_film_affine_f32: tensor extent overflows".to_owned())
        })?;
        if input.len() != expected || gamma.len() != channel_count || beta.len() != channel_count {
            return Err(OxideError::Shape(
                "foveation_film_affine_f32: tensor extents do not match NCHW / [B,C] layouts"
                    .to_owned(),
            ));
        }
        let mut output = self.alloc_uninit(expected)?;
        // SAFETY: host validation proves the input and [B,C] operands cover
        // the full launch; each thread owns exactly one output element.
        unsafe {
            self.foveation_film.film_affine(
                &self.stream,
                launch::elementwise(expected, config),
                &input.buf,
                &gamma.buf,
                &beta.buf,
                &mut output,
                channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    fn foveation_film_backward_f32(
        &self,
        input: &Self::Tensor,
        output_gradient: &Self::Tensor,
        hf_params: &Self::Tensor,
        embedding: &Self::Tensor,
        gamma: &Self::Tensor,
        w_embed: &Self::Tensor,
        w_gamma: &Self::Tensor,
        w_beta: &Self::Tensor,
        shape: &[usize],
        param_count: usize,
        embed_dim: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<FoveationFilmGradients<Self::Tensor>, Self::Error> {
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_backward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        if param_count == 0 || embed_dim == 0 {
            return Err(OxideError::Shape(
                "foveation_film_backward_f32: param_count and embed_dim must be positive"
                    .to_owned(),
            ));
        }
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: spatial extent overflows".to_owned())
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: channel extent overflows".to_owned())
        })?;
        let input_count = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: input extent overflows".to_owned())
        })?;
        let embed_count = batch.checked_mul(embed_dim).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: embedding extent overflows".to_owned())
        })?;
        let hf_count = batch.checked_mul(param_count).ok_or_else(|| {
            OxideError::Shape("foveation_film_backward_f32: HF extent overflows".to_owned())
        })?;
        let embed_weights_count = embed_dim.checked_mul(param_count).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_backward_f32: embedding weight extent overflows".to_owned(),
            )
        })?;
        let channel_weights_count = channels.checked_mul(embed_dim).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_backward_f32: channel weight extent overflows".to_owned(),
            )
        })?;
        let input = self.f32v(input)?;
        let input = &*input;
        let hf_params = self.f32v(hf_params)?;
        let hf_params = &*hf_params;
        let embedding = self.f32v(embedding)?;
        let embedding = &*embedding;
        let w_embed = self.f32v(w_embed)?;
        let w_embed = &*w_embed;
        let w_gamma = self.f32v(w_gamma)?;
        let w_gamma = &*w_gamma;
        let w_beta = self.f32v(w_beta)?;
        let w_beta = &*w_beta;
        if input.len() != input_count
            || hf_params.len() != hf_count
            || embedding.len() != embed_count
            || w_embed.len() != embed_weights_count
            || w_gamma.len() != channel_weights_count
            || w_beta.len() != channel_weights_count
        {
            return Err(OxideError::Shape(
                "foveation_film_backward_f32: operand extents do not match configured layouts"
                    .to_owned(),
            ));
        }
        let (dx, dgamma, dbeta) =
            self.foveation_film_affine_backward_f32(input, output_gradient, gamma, shape, config)?;
        let dgamma_f = self.f32v(&dgamma)?;
        let dgamma_f = &*dgamma_f;
        let dbeta_f = self.f32v(&dbeta)?;
        let dbeta_f = &*dbeta_f;
        let mut d_embedding = self.alloc_uninit(embed_count)?;
        // Every embedding-gradient element is written by one thread.
        unsafe {
            self.foveation_film.film_embedding_backward(
                &self.stream,
                launch::elementwise(embed_count, config),
                &embedding.buf,
                &dgamma_f.buf,
                &dbeta_f.buf,
                &w_gamma.buf,
                &w_beta.buf,
                &mut d_embedding,
                channels,
                embed_dim,
            )
        }
        .map_err(launch_err)?;

        let dw_embed_count = embed_weights_count;
        let dw_channel_count = channel_weights_count;
        let target_count = dw_embed_count
            .max(embed_dim)
            .max(dw_channel_count)
            .max(channels)
            .max(hf_count);
        let mut dw_embed = self.alloc_uninit(dw_embed_count)?;
        let mut db_embed = self.alloc_uninit(embed_dim)?;
        let mut dw_gamma = self.alloc_uninit(dw_channel_count)?;
        let mut db_gamma = self.alloc_uninit(channels)?;
        let mut dw_beta = self.alloc_uninit(dw_channel_count)?;
        let mut db_beta = self.alloc_uninit(channels)?;
        let mut dhf = self.alloc_uninit(hf_count)?;
        // Every output element is owned by a single thread; reductions run in
        // a fixed order over the batch or embedding axes.
        unsafe {
            self.foveation_film.film_projection_backward(
                &self.stream,
                launch::elementwise(target_count, config),
                &hf_params.buf,
                &embedding.buf,
                &dgamma_f.buf,
                &dbeta_f.buf,
                &d_embedding,
                &w_embed.buf,
                &mut dw_embed,
                &mut db_embed,
                &mut dw_gamma,
                &mut db_gamma,
                &mut dw_beta,
                &mut db_beta,
                &mut dhf,
                batch,
                channels,
                param_count,
                embed_dim,
            )
        }
        .map_err(launch_err)?;
        Ok(FoveationFilmGradients {
            input: dx,
            hf_params: OxideTensorF32::new(dhf, &[batch, param_count]),
            w_embed: OxideTensorF32::new(dw_embed, &[embed_dim, param_count]),
            b_embed: OxideTensorF32::new(db_embed, &[embed_dim]),
            w_gamma: OxideTensorF32::new(dw_gamma, &[channels, embed_dim]),
            b_gamma: OxideTensorF32::new(db_gamma, &[channels]),
            w_beta: OxideTensorF32::new(dw_beta, &[channels, embed_dim]),
            b_beta: OxideTensorF32::new(db_beta, &[channels]),
        })
    }

    fn foveation_film_affine_backward_f32(
        &self,
        input: &Self::Tensor,
        output_gradient: &Self::Tensor,
        gamma: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let input = self.f32v(input)?;
        let input = &*input;
        let output_gradient = self.f32v(output_gradient)?;
        let output_gradient = &*output_gradient;
        let gamma = self.f32v(gamma)?;
        let gamma = &*gamma;
        if shape.len() != 4 || shape.iter().any(|&dim| dim == 0) {
            return Err(OxideError::Shape(format!(
                "foveation_film_affine_backward_f32: expected non-empty NCHW shape, got {shape:?}"
            )));
        }
        let [batch, channels, height, width] = [shape[0], shape[1], shape[2], shape[3]];
        let spatial = height.checked_mul(width).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: spatial extent overflows".to_owned(),
            )
        })?;
        let channel_count = batch.checked_mul(channels).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: channel extent overflows".to_owned(),
            )
        })?;
        let expected = channel_count.checked_mul(spatial).ok_or_else(|| {
            OxideError::Shape(
                "foveation_film_affine_backward_f32: tensor extent overflows".to_owned(),
            )
        })?;
        if input.len() != expected
            || output_gradient.len() != expected
            || gamma.len() != channel_count
        {
            return Err(OxideError::Shape(
                "foveation_film_affine_backward_f32: tensor extents do not match NCHW / [B,C] layouts".to_owned(),
            ));
        }
        let mut dx = self.alloc_uninit(expected)?;
        let mut dgamma = self.alloc_uninit(channel_count)?;
        let mut dbeta = self.alloc_uninit(channel_count)?;
        // SAFETY: checked extents; dx is element-owned and parameter gradients
        // are reduced by one thread per sample/channel.
        unsafe {
            self.foveation_film.film_affine_dx(
                &self.stream,
                launch::elementwise(expected, config),
                &output_gradient.buf,
                &gamma.buf,
                &mut dx,
                channels,
                spatial,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.foveation_film.film_affine_param_grads(
                &self.stream,
                launch::elementwise(channel_count, config),
                &input.buf,
                &output_gradient.buf,
                &mut dgamma,
                &mut dbeta,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(dx, shape),
            OxideTensorF32::new(dgamma, &[batch, channels]),
            OxideTensorF32::new(dbeta, &[batch, channels]),
        ))
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
        let (channels, hw) =
            self.scale_channels_layout("scale_channels_f32", scale, source, shape, scale_wh)?;
        let mut out = self.alloc_uninit(source.len())?;
        // SAFETY: one thread per output element; scale index is in range by
        // the layout check.
        unsafe {
            self.scale_channels.scale_channels_forward(
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
            self.scale_channels.scale_channels_dsource(
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
            self.scale_channels.scale_channels_dscale(
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
        Ok((
            OxideTensorF32::new(d_scale, scale.shape()),
            OxideTensorF32::new(d_src, shape),
        ))
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
            return Err(shape_err(
                op,
                format!("expected {outputs} biases, got {}", biases.len()),
            ));
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
            return Err(shape_err(
                op,
                format!(
                    "expected {} delta elements, got {}",
                    batch * outputs,
                    delta.len()
                ),
            ));
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
            return self.batchnorm_train_forward_amp(
                x,
                scales,
                biases,
                rolling_mean,
                rolling_variance,
                shape,
                keep,
                config,
            );
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
        let (channels, stride, row_groups) = Self::batchnorm_layout(
            op,
            shape,
            x.len(),
            &[scales, biases, rolling_mean, rolling_variance],
        )?;
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
        unsafe {
            self.batchnorm
                .bn_mean_finalize(&self.stream, per_channel, &partial, &mut mean, 1.0 / n)
        }
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
            return self.batchnorm_train_backward_amp(
                output_grad,
                x_hat,
                variance,
                scales,
                shape,
                config,
            );
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

    fn sanitized_sgd_update_multi_f32(
        &self,
        items: &[SgdUpdateItem<'_, Self::Tensor>],
        lr: f32,
        momentum: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor)>, Self::Error> {
        let op = "sanitized_sgd_update_multi_f32";
        if items.is_empty() || items.len() > 3 {
            return Err(OxideError::Shape(format!(
                "{op}: 1..=3 tensors per launch, got {}",
                items.len()
            )));
        }
        // Resolve every operand to f32 (AMP tensors are widened by `f32v`).
        let mut w_f = Vec::new();
        let mut u_f = Vec::new();
        let mut g_f = Vec::new();
        for it in items {
            let w = self.f32v(it.weights)?;
            let u = self.f32v(it.updates)?;
            same_len(op, &*w, &*u)?;
            if it.shape.iter().product::<usize>() != w.len() {
                return Err(OxideError::Shape(format!(
                    "{op}: shape {:?} does not match {} elements",
                    it.shape,
                    w.len()
                )));
            }
            let g = match it.grad {
                Some(g) => {
                    let g = self.f32v(g)?;
                    same_len(op, &*w, &*g)?;
                    Some(g)
                }
                None => None,
            };
            w_f.push(w);
            u_f.push(u);
            g_f.push(g);
        }
        let batch = batch.max(1) as f32;
        // Unused segments reuse segment 0's inputs (never read, n == 0) and a
        // one-element dummy output.
        let seg = |k: usize| if k < items.len() { k } else { 0 };
        let n = |k: usize| if k < items.len() { w_f[k].len() } else { 0 };
        let mut w_out = Vec::new();
        let mut u_out = Vec::new();
        for k in 0..3 {
            let len = if k < items.len() { w_f[k].len() } else { 1 };
            w_out.push(self.alloc_uninit(len)?);
            u_out.push(self.alloc_uninit(len)?);
        }
        let gbuf = |k: usize| match &g_f[seg(k)] {
            Some(g) => &g.buf,
            None => &u_f[seg(k)].buf,
        };
        let hg = |k: usize| (k < items.len() && g_f[k].is_some()) as u32;
        let decay = |k: usize| {
            if k < items.len() {
                items[k].decay * batch
            } else {
                0.0
            }
        };
        let total = n(0) + n(1) + n(2);
        let [wo0, wo1, wo2] = &mut *w_out else {
            unreachable!()
        };
        let [uo0, uo1, uo2] = &mut *u_out else {
            unreachable!()
        };
        // SAFETY: one thread per element of the concatenation; every output
        // slot of every used segment is written.
        unsafe {
            self.optim.sanitized_sgd_update3(
                &self.stream,
                launch::elementwise(total, config),
                &w_f[0].buf,
                &u_f[0].buf,
                gbuf(0),
                wo0,
                uo0,
                &w_f[seg(1)].buf,
                &u_f[seg(1)].buf,
                gbuf(1),
                wo1,
                uo1,
                &w_f[seg(2)].buf,
                &u_f[seg(2)].buf,
                gbuf(2),
                wo2,
                uo2,
                n(0),
                n(1),
                n(2),
                hg(0),
                hg(1),
                hg(2),
                items[0].fix_weights as u32,
                lr / batch,
                momentum,
                decay(0),
                decay(1),
                decay(2),
            )
        }
        .map_err(launch_err)?;
        let mut out = Vec::with_capacity(items.len());
        let mut wi = w_out.into_iter();
        let mut ui = u_out.into_iter();
        for it in items {
            out.push((
                OxideTensorF32::new(wi.next().expect("w_out"), it.shape),
                OxideTensorF32::new(ui.next().expect("u_out"), it.shape),
            ));
        }
        Ok(out)
    }

    fn sanitized_adam_update_multi_f32(
        &self,
        items: &[AdamUpdateItem<'_, Self::Tensor>],
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor)>, Self::Error> {
        let op = "sanitized_adam_update_multi_f32";
        if items.is_empty() || items.len() > 3 {
            return Err(OxideError::Shape(format!(
                "{op}: 1..=3 tensors per launch, got {}",
                items.len()
            )));
        }
        let mut w_f = Vec::new();
        let mut u_f = Vec::new();
        let mut g_f = Vec::new();
        let mut m_f = Vec::new();
        let mut v_f = Vec::new();
        for it in items {
            let w = self.f32v(it.weights)?;
            let u = self.f32v(it.updates)?;
            let m = self.f32v(it.m)?;
            let v = self.f32v(it.v)?;
            same_len(op, &*w, &*u)?;
            same_len(op, &*w, &*m)?;
            same_len(op, &*w, &*v)?;
            if it.shape.iter().product::<usize>() != w.len() {
                return Err(OxideError::Shape(format!(
                    "{op}: shape {:?} does not match {} elements",
                    it.shape,
                    w.len()
                )));
            }
            let g = match it.grad {
                Some(g) => {
                    let g = self.f32v(g)?;
                    same_len(op, &*w, &*g)?;
                    Some(g)
                }
                None => None,
            };
            w_f.push(w);
            u_f.push(u);
            g_f.push(g);
            m_f.push(m);
            v_f.push(v);
        }
        let seg = |k: usize| if k < items.len() { k } else { 0 };
        let n = |k: usize| if k < items.len() { w_f[k].len() } else { 0 };
        let mut w_out = Vec::new();
        let mut m_out = Vec::new();
        let mut v_out = Vec::new();
        let mut u_out = Vec::new();
        for k in 0..3 {
            let len = if k < items.len() { w_f[k].len() } else { 1 };
            w_out.push(self.alloc_uninit(len)?);
            m_out.push(self.alloc_uninit(len)?);
            v_out.push(self.alloc_uninit(len)?);
            u_out.push(self.alloc_uninit(len)?);
        }
        let gbuf = |k: usize| match &g_f[seg(k)] {
            Some(g) => &g.buf,
            None => &u_f[seg(k)].buf,
        };
        let hg = |k: usize| (k < items.len() && g_f[k].is_some()) as u32;
        let scalars = |k: usize| {
            if k < items.len() {
                let (c, d, _, _) = items[k].params.kernel_scalars();
                (c, d)
            } else {
                (0.0, 0.0)
            }
        };
        let p0 = items[0].params;
        let (_, _, bc1, bc2) = p0.kernel_scalars();
        let total = n(0) + n(1) + n(2);
        let [wo0, wo1, wo2] = &mut *w_out else {
            unreachable!()
        };
        let [mo0, mo1, mo2] = &mut *m_out else {
            unreachable!()
        };
        let [vo0, vo1, vo2] = &mut *v_out else {
            unreachable!()
        };
        let [uo0, uo1, uo2] = &mut *u_out else {
            unreachable!()
        };
        // SAFETY: as in `sanitized_sgd_update_multi_f32`.
        unsafe {
            self.optim.sanitized_adam_update3(
                &self.stream,
                launch::elementwise(total, config),
                &w_f[0].buf,
                &u_f[0].buf,
                gbuf(0),
                &m_f[0].buf,
                &v_f[0].buf,
                wo0,
                mo0,
                vo0,
                uo0,
                &w_f[seg(1)].buf,
                &u_f[seg(1)].buf,
                gbuf(1),
                &m_f[seg(1)].buf,
                &v_f[seg(1)].buf,
                wo1,
                mo1,
                vo1,
                uo1,
                &w_f[seg(2)].buf,
                &u_f[seg(2)].buf,
                gbuf(2),
                &m_f[seg(2)].buf,
                &v_f[seg(2)].buf,
                wo2,
                mo2,
                vo2,
                uo2,
                n(0),
                n(1),
                n(2),
                hg(0),
                hg(1),
                hg(2),
                items[0].fix_weights as u32,
                p0.lr,
                p0.beta1,
                p0.beta2,
                p0.eps,
                bc1,
                bc2,
                scalars(0).0,
                scalars(0).1,
                scalars(1).0,
                scalars(1).1,
                scalars(2).0,
                scalars(2).1,
            )
        }
        .map_err(launch_err)?;
        let mut out = Vec::with_capacity(items.len());
        let mut wi = w_out.into_iter();
        let mut mi = m_out.into_iter();
        let mut vi = v_out.into_iter();
        let mut ui = u_out.into_iter();
        for it in items {
            out.push((
                OxideTensorF32::new(wi.next().expect("w_out"), it.shape),
                OxideTensorF32::new(mi.next().expect("m_out"), it.shape),
                OxideTensorF32::new(vi.next().expect("v_out"), it.shape),
                OxideTensorF32::new(ui.next().expect("u_out"), it.shape),
            ));
        }
        Ok(out)
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
