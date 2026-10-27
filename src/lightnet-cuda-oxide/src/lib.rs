//! `F32TensorOpsBackend` implemented with kernels written in Rust and
//! compiled by cuda-oxide (`cargo oxide`).  Build this crate only through
//! `cargo oxide`; plain `cargo build` cannot compile the device code.

pub mod conv;
mod conv2d_backend;
mod recurrent_backend;
mod clifford;
mod dcnv4_native;
mod deconvolutional;
mod channel_shuffle_backend;
mod clifford_backend;
mod gelu_erf_backend;
mod softmax_backend;
mod reorg_backend;
mod crop_backend;
mod local_backend;
mod pool_backend;
mod graph_conv_backend;
mod tucker_attention_backend;
mod kimi_kda_backend;
mod mambavision_backend;
mod swin_backend;
mod route_backend;
mod upsample_backend;
mod connected_backend;
mod eml_backend;
mod batchnorm_backend;
mod dense_loss_backend;
mod laya_act_backend;
mod depth_backend;
mod wmhf_backend;
mod deform_conv_backend;
mod foveation_film_backend;
mod hyperbolic_backend;
mod moe_backend;
mod shortcut_backend;
mod flash_attention_backend;
mod laya_encoder_backend;
mod laya_head_backend;
mod laya_scorer_backend;
mod dsconv_backend;
mod graph_conv;
mod swin;
mod mambavision;
mod kimi_kda;
mod tucker_attention;
#[path = "vit.rs"]
mod vit_backend;
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
    mambavision: kernels::mambavision::kernels::LoadedModule,
    kimi_kda: kernels::kimi_kda::kernels::LoadedModule,
    tucker_attention: kernels::tucker_attention::kernels::LoadedModule,
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

pub(crate) fn shape_err(op: &str, msg: String) -> OxideError {
    OxideError::Shape(format!("{op}: {msg}"))
}

/// Checks an NCHW `[batch, channels, h, w]` shape and returns its batch.
pub(crate) fn nchw_batch(
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
pub(crate) fn pool_layout(
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
        let mambavision = kernels::mambavision::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load mambavision module: {err:?}")))?;
        let kimi_kda = kernels::kimi_kda::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load kimi_kda module: {err:?}")))?;
        let tucker_attention = kernels::tucker_attention::kernels::load(&ctx)
            .map_err(|err| OxideError::Shape(format!("load tucker_attention module: {err:?}")))?;
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
            mambavision,
            kimi_kda,
            tucker_attention,
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
    pub(crate) fn reorg_launch(
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
    pub(crate) fn batchnorm_layout(
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
    pub(crate) fn fold_partials(&self, partial: &PooledBuf, channels: usize) -> Result<PooledBuf, OxideError> {
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

    pub(crate) fn connected_layout(
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
pub(crate) fn deform_gather_shape(
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
        self.jev_ce_loss_backend_f32(
            logits, truth, shape, groups, candidates, temperature, normalize, ignore_empty,
            label_smoothing, scale, channel_major, config,
        )
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
        self.segmentation_loss_backend_f32(
            output, truth, shape, classes, spatial, softmax_loss, scale, gate_strength,
            gate_t_correct, gate_max_compensation, max_delta, clipping, config,
        )
    }
}

impl F32TensorOpsBackend for OxideBackend {
    type Error = OxideError;
    type Tensor = OxideTensorF32;

    fn any_non_finite_f32(&self, tensor: &Self::Tensor) -> Result<bool, Self::Error> {
        let input = self.f32v(tensor)?;
        if input.len() == 0 { return Ok(false); }
        let chunks = input.len().div_ceil(256);
        let mut flags = self.alloc_uninit(chunks)?;
        let mut scalar = self.alloc_uninit(1)?;
        let config = ElementwiseLaunchConfig::default();
        // SAFETY: exactly ceil(len/256) chunk owners, then one scalar owner.
        // The launches share a stream and the reduction only sees written flags.
        unsafe { self.optim.nonfinite_partial(&self.stream, launch::elementwise(chunks, config),
            &input.buf, &mut flags) }.map_err(launch_err)?;
        unsafe { self.optim.nonfinite_reduce(&self.stream, launch::elementwise(1, config),
            &flags, &mut scalar) }.map_err(launch_err)?;
        Ok(scalar.to_host_vec(&self.stream)?[0] != 0.0)
    }

    fn depth_tap_clamp_f32(&self, raw: &Self::Tensor, gradient: Option<&Self::Tensor>,
        shape: &[usize], clip: f32, config: ElementwiseLaunchConfig) -> Result<Self::Tensor, Self::Error> {
        self.depth_backend_tap_clamp_f32(raw, gradient, shape, clip, config)
    }

    fn kimi_scan_f32(
        &self, q: &Self::Tensor, k: &Self::Tensor, v: &Self::Tensor,
        beta: &Self::Tensor, decay: &Self::Tensor, g: lightnet_cuda::kimi_kda::KdaGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::kimi_kda::KdaForward<Self::Tensor>, Self::Error> {
        self.kimi_scan_backend_f32(q, k, v, beta, decay, g, config)
    }

    fn kimi_scan_backward_f32(
        &self, q: &Self::Tensor, v: &Self::Tensor, beta: &Self::Tensor, decay: &Self::Tensor,
        forward: &lightnet_cuda::kimi_kda::KdaForward<Self::Tensor>, dy: &Self::Tensor,
        g: lightnet_cuda::kimi_kda::KdaGeometry, config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::kimi_kda::KdaGrad<Self::Tensor>, Self::Error> {
        self.kimi_scan_backend_backward_f32(q, v, beta, decay, forward, dy, g, config)
    }

    fn kimi_router_gates_f32(
        &self, logits: &Self::Tensor, rows: usize, experts: usize, top_k: usize,
        temperature: f32, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.kimi_router_backend_gates_f32(logits, rows, experts, top_k, temperature, config)
    }

    fn mambavision_depthwise_f32(
        &self, input: &Self::Tensor, weights: &Self::Tensor, bias: &Self::Tensor,
        batch: usize, tokens: usize, channels: usize, branch: usize, kernel: usize,
        causal: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.mambavision_backend_depthwise_f32(input, weights, bias, batch, tokens,
            channels, branch, kernel, causal, config)
    }

    fn mambavision_depthwise_backward_f32(
        &self, input: &Self::Tensor, weights: &Self::Tensor, pre: &Self::Tensor,
        grad: &Self::Tensor, batch: usize, tokens: usize, channels: usize,
        branch: usize, kernel: usize, causal: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.mambavision_backend_depthwise_backward_f32(input, weights, pre, grad,
            batch, tokens, channels, branch, kernel, causal, config)
    }

    fn mambavision_scan_f32(
        &self, x: &Self::Tensor, projection: &Self::Tensor, dt_weights: &Self::Tensor,
        dt_bias: &Self::Tensor, a_log: &Self::Tensor, d_skip: &Self::Tensor,
        state_in: &Self::Tensor, g: lightnet_cuda::mambavision::MambaScanGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::mambavision::MambaScanForward<Self::Tensor>, Self::Error> {
        self.mambavision_backend_scan_f32(x, projection, dt_weights, dt_bias, a_log,
            d_skip, state_in, g, config)
    }

    fn mambavision_scan_backward_f32(
        &self, x: &Self::Tensor, projection: &Self::Tensor, dt_weights: &Self::Tensor,
        a_log: &Self::Tensor, d_skip: &Self::Tensor, state_in: &Self::Tensor,
        dt_pre: &Self::Tensor, states: &Self::Tensor, grad: &Self::Tensor,
        state_grad: &Self::Tensor, g: lightnet_cuda::mambavision::MambaScanGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::mambavision::MambaScanGrad<Self::Tensor>, Self::Error> {
        self.mambavision_backend_scan_backward_f32(x, projection, dt_weights, a_log,
            d_skip, state_in, dt_pre, states, grad, state_grad, g, config)
    }

    fn mambavision_msa_f32(
        &self, input: &Self::Tensor, batch: usize, tokens: usize, channels: usize,
        heads: usize, block: usize, topk: usize, config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.mambavision_backend_msa_f32(input, batch, tokens, channels, heads, block, topk, config)
    }

    fn mambavision_msa_backward_f32(
        &self, input: &Self::Tensor, probabilities: &Self::Tensor, active: &Self::Tensor,
        grad: &Self::Tensor, batch: usize, tokens: usize, channels: usize,
        heads: usize, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.mambavision_backend_msa_backward_f32(input, probabilities, active, grad,
            batch, tokens, channels, heads, config)
    }

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
        self.gelu_erf_backend_f32(input, shape, config)
    }

    fn gelu_erf_backward_f32(
        &self,
        input: &Self::Tensor,
        seeds: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.gelu_erf_backward_backend_f32(input, seeds, shape, config)
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
        self.eml_convolutional_backend_f32(
            x_branch, self_branch, residual, output_shape, clamp, eps, scale, config,
        )
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
        self.eml_convolutional_backend_backward_f32(
            x_branch, self_branch, output_grad, output_shape, clamp, eps, scale, config,
        )
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
        self.lstm_gate_backward_f32_backend(
            hidden_gradient, cell, previous_cell, forget, input_gate, candidate, output_gate, next_cell_gradient, elements, config,
        )
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
        self.sequence_slice_f32_backend(
            input, input_shape, step_index, steps, output_shape, config,
        )
    }

    fn sequence_concat_f32(
        &self,
        steps: &[&Self::Tensor],
        output_shape: &[usize],
        step_elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.sequence_concat_f32_backend(
            steps, output_shape, step_elements, config,
        )
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.shortcut_add_backend_f32(x, y, config)
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
        self.shortcut_weight_gradient_backend_f32(delta, source, batch, per_sample, source_per_sample, layer_step, config)
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
        self.shortcut_weight_gradient_accumulate_backend_f32(accumulated, delta, source, batch, per_sample, source_per_sample, num_inputs, layer_step, source_index, config)
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
        self.shortcut_weight_normalized_gradient_backend_f32(raw_weights, products, num_inputs, layer_step, normalization, cpp_grad, config)
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
        self.route_backend_concat_f32(
            inputs, input_channel_offsets, input_channels, input_total_channels,
            output_shape, channel_stride, config,
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
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        self.route_backend_backward_f32(
            output_grad, input_shapes, input_channel_offsets, input_channels,
            input_total_channels, channel_stride, config,
        )
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
        self.deconvolutional_f32(
            x,
            weights,
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
        self.deconvolutional_backward_f32(
            x,
            weights,
            output_grad,
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
        self.local_backend_f32(
            x,
            weights,
            biases,
            input_shape,
            output_shape,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            padding,
            config,
        )
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
        self.local_backend_backward_f32(
            x,
            weights,
            output_grad,
            input_shape,
            output_shape,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn graph_conv_aggregate_f32(
        &self,
        input: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_backend_aggregate_f32(input, geometry, config)
    }

    fn graph_conv_reference_f32(
        &self,
        input: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_backend_reference_f32(input, geometry, config)
    }

    fn graph_conv_reference_backward_f32(
        &self,
        output_gradient: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.graph_conv_backend_reference_backward_f32(output_gradient, geometry, config)
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
        self.graph_conv_backend_learned_aggregate_f32(
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
        self.graph_conv_backend_learned_aggregate_backward_f32(
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
        self.graph_conv_backend_aggregate_backward_f32(output_gradient, geometry, config)
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
        self.conv2d_f32_backend(
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
        self.conv2d_backward_input_f32_backend(
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
        self.conv2d_backward_weights_f32_backend(
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
        self.pool_backend_maxpool_f32(
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.pool_backend_maxpool_backward_f32(
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

    fn maxpool_depth_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.depth_backend_maxpool_depth_f32(x, output_shape, input_channels, config)
    }

    fn maxpool_depth_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        output_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.depth_backend_maxpool_depth_backward_f32(
            x, output_grad, input_shape, output_channels, config,
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.pool_backend_local_avgpool_f32(
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.pool_backend_local_avgpool_backward_f32(
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
    ) -> Result<Self::Tensor, Self::Error> {
        self.pool_backend_avgpool_f32(
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            config,
        )
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
        self.pool_backend_avgpool_backward_f32(
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            config,
        )
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
        self.moe_router_gates_backend_f32(input, router_weights, router_biases, batch, channels, spatial, experts, top_k, temperature, config)
    }

    fn moe_router_pool_f32(
        &self,
        input: &Self::Tensor,
        batch: usize,
        channels: usize,
        spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.moe_router_pool_backend_f32(input, batch, channels, spatial, config)
    }

    fn moe_router_pool_backward_f32(
        &self,
        pooled_gradient: &Self::Tensor,
        input_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.moe_router_pool_backward_backend_f32(pooled_gradient, input_shape, config)
    }

    fn moe_gate_dot_f32(
        &self,
        output_gradient: &Self::Tensor,
        expert_output: &Self::Tensor,
        batch: usize,
        per_image: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.moe_gate_dot_backend_f32(output_gradient, expert_output, batch, per_image, config)
    }

    fn moe_gate_column_f32(
        &self,
        gates: &Self::Tensor,
        batch: usize,
        experts: usize,
        expert: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.moe_gate_column_backend_f32(gates, batch, experts, expert, config)
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
        self.moe_router_backward_backend_f32(gates, output_gradient, expert_outputs, batch, experts, per_image, load_balance, temperature, config)
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
        self.hyperbolic_foveation_predictor_backend_f32(
            input,
            shape,
            downsample,
            hidden_count,
            param_count,
            w1,
            b1,
            w2,
            b2,
            config,
        )
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
        self.hyperbolic_foveation_predictor_backward_backend_f32(
            pooled,
            hidden_preactivation,
            hidden,
            raw_output,
            constrained_seed,
            w1,
            w2,
            input_shape,
            downsample,
            hidden_count,
            param_count,
            config,
        )
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
        self.hyperbolic_newton_grid_backend_f32(
            params,
            batch,
            height,
            width,
            param_count,
            alpha,
            p_exp,
            iterations,
            eta,
            config,
        )
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
        self.hyperbolic_fixed_point_grid_backend_f32(
            params,
            batch,
            height,
            width,
            param_count,
            alpha,
            p_exp,
            iterations,
            eta,
            config,
        )
    }

    fn hyperbolic_foveation_warp_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.hyperbolic_foveation_warp_backend_f32(
            input,
            grid,
            shape,
            config,
        )
    }

    fn hyperbolic_foveation_warp_backward_f32(
        &self,
        input: &Self::Tensor,
        grid: &Self::Tensor,
        output_gradient: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.hyperbolic_foveation_warp_backward_backend_f32(
            input,
            grid,
            output_gradient,
            shape,
            config,
        )
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
        self.hyperbolic_foveation_inverse_param_grad_backend_f32(
            params,
            grid_gradient,
            shape,
            param_count,
            alpha,
            p_exp,
            iterations,
            eta,
            newton,
            config,
        )
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
        self.crop_backend_f32(
            x, output_shape, channels, input_height, input_width, source_x, source_y,
            input_scales, input_offsets, input_saturations, input_exposures, apply_hsv, config,
        )
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
        self.crop_backend_backward_f32(
            output_grad, input_shape, channels, input_height, input_width, source_x, source_y,
            input_scales, config,
        )
    }

    fn deform_bilinear_gather_f32(
        &self,
        input: &Self::Tensor,
        sample_y: &Self::Tensor,
        sample_x: &Self::Tensor,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.deform_bilinear_gather_backend_f32(
            input, sample_y, sample_x, geometry, config,
        )
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
        self.deform_bilinear_gather_backward_backend_f32(
            input, sample_y, sample_x, output_grad, geometry, config,
        )
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
        self.deform_predictor_backend_f32(
            input, weights, biases, output_shape, channels, in_h, in_w, filters, size, stride_x, stride_y, padding, dilation, out_h, out_w, config,
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
        self.deform_conv_accumulate_backend_f32(
            input, offsets, masks, weights, geometry, use_mask, shape, config,
        )
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
        OxideBackend::wmhf_backend_haar_dwt_f32(self, input, batch, channels, height, width, config)
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
        OxideBackend::wmhf_backend_haar_dwt_backward_f32(
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
        OxideBackend::wmhf_backend_haar_iwt_f32(
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
        OxideBackend::wmhf_backend_haar_iwt_backward_f32(
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
        OxideBackend::wmhf_backend_scan4_forward_f32(
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
        OxideBackend::wmhf_backend_scan4_backward_f32(
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
        OxideBackend::wmhf_backend_high_frequency_forward_f32(
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
        OxideBackend::wmhf_backend_high_frequency_backward_f32(
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
        OxideBackend::wmhf_backend_concat_local_f32(
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
        OxideBackend::wmhf_backend_split_channels_f32(
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

    fn vit_resident_mhc_forward_f32(&self, skip: &Self::Tensor, branch: &Self::Tensor,
        params: &Self::Tensor, site: usize, clip: f32, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let skip = self.f32v(skip)?; let branch = self.f32v(branch)?; let params = self.f32v(params)?;
        if skip.len() == 0 || skip.len() != branch.len() || params.len() != 12 || site >= 2
            || !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err("resident ViT mHC", "invalid lengths, site or clip".into()));
        }
        let mut out = self.alloc_uninit(skip.len())?;
        // SAFETY: validated operands and one writer per output element.
        unsafe { self.vit.vit_resident_mhc(&self.stream, launch::elementwise(skip.len(),config),
            &skip.buf, &branch.buf, &skip.buf, &params.buf, &mut out, site, 0, clip) }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, skip.shape()))
    }

    fn vit_resident_mhc_backward_f32(&self, skip: &Self::Tensor, branch: &Self::Tensor,
        grad: &Self::Tensor, params: &Self::Tensor, site: usize, clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        let skip = self.f32v(skip)?; let branch = self.f32v(branch)?;
        let grad = self.f32v(grad)?; let params = self.f32v(params)?;
        let n = skip.len();
        if n == 0 || n != branch.len() || n != grad.len() || params.len() != 12 || site >= 2
            || !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err("resident ViT mHC backward", "invalid lengths, site or clip".into()));
        }
        let mut ds = self.alloc_uninit(n)?; let mut db = self.alloc_uninit(n)?;
        let mut terms = self.alloc_uninit(3*n)?;
        // SAFETY: each launch fully writes a disjoint output of its stated length.
        unsafe {
            self.vit.vit_resident_mhc(&self.stream, launch::elementwise(n,config), &skip.buf,
                &branch.buf, &grad.buf, &params.buf, &mut ds, site, 1, clip).map_err(launch_err)?;
            self.vit.vit_resident_mhc(&self.stream, launch::elementwise(n,config), &skip.buf,
                &branch.buf, &grad.buf, &params.buf, &mut db, site, 2, clip).map_err(launch_err)?;
            self.vit.vit_resident_mhc(&self.stream, launch::elementwise(3*n,config), &skip.buf,
                &branch.buf, &grad.buf, &params.buf, &mut terms, site, 3, clip).map_err(launch_err)?;
        }
        let sums = self.bias_backward_f32(&OxideTensorF32::new(terms,&[3,n]),3,n,config)?;
        let mut dp = self.alloc_uninit(6)?;
        // SAFETY: six logit outputs read the validated twelve parameters and three sums.
        unsafe { self.vit.vit_resident_mhc_gradient(&self.stream,launch::elementwise(6,config),
            &params.buf,&sums.buf,&mut dp,site) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(ds,grad.shape()),OxideTensorF32::new(db,grad.shape()),
            OxideTensorF32::new(dp,&[6])))
    }

    fn vit_prepare_updates_f32(&self, updates: &Self::Tensor, excluded_start: usize,
        excluded_end: usize, limit: f32, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let updates = self.f32v(updates)?;
        if excluded_start > excluded_end || excluded_end > updates.len() || !limit.is_finite() || limit <= 0.0 {
            return Err(shape_err("ViT update clipping", "invalid excluded range or limit".into()));
        }
        let mut scale = self.alloc_uninit(1)?;
        let mut out = self.alloc_uninit(updates.len())?;
        // SAFETY: one scalar writer followed by one writer per update, on the same stream.
        unsafe {
            self.vit.vit_update_norm(&self.stream,launch::elementwise(1,config),&updates.buf,
                &mut scale,excluded_start,excluded_end,limit).map_err(launch_err)?;
            self.vit.vit_update_scale(&self.stream,launch::elementwise(updates.len(),config),
                &updates.buf,&scale,&mut out,excluded_start,excluded_end).map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(out,&[updates.len()]))
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

    fn tucker_attention_f32(
        &self, q: &Self::Tensor, k: &Self::Tensor, v: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::tucker_attention::AttentionGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.tucker_attention_backend_f32(q, k, v, geometry, config)
    }

    fn tucker_attention_backward_f32(
        &self, q: &Self::Tensor, k: &Self::Tensor, v: &Self::Tensor,
        probabilities: &Self::Tensor, active: &Self::Tensor, grad: &Self::Tensor,
        geometry: lightnet_cuda::host_ops::tucker_attention::AttentionGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.tucker_attention_backend_backward_f32(
            q,
            k,
            v,
            probabilities,
            active,
            grad,
            geometry,
            config,
        )
    }

fn clifford_fourier_f32(&self, input: &Self::Tensor, wr: &Self::Tensor, wi: &Self::Tensor,
    batch: usize, channels: usize, height: usize, width: usize, mh: usize, mw: usize,
    config: ElementwiseLaunchConfig) -> Result<Self::Tensor, Self::Error> {
    self.clifford_fourier_native(input, None, wr, wi, batch, channels, height, width, mh, mw, config).map(|v| v.0)
}
fn clifford_fourier_backward_f32(&self, input: &Self::Tensor, grad: &Self::Tensor, wr: &Self::Tensor, wi: &Self::Tensor,
    batch: usize, channels: usize, height: usize, width: usize, mh: usize, mw: usize,
    config: ElementwiseLaunchConfig) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
    let (dx, weights) = self.clifford_fourier_native(input, Some(grad), wr, wi, batch, channels, height, width, mh, mw, config)?;
    let (dw, dv) = weights.expect("backward returns weight gradients"); Ok((dx, dw, dv))
}

    fn clifford_interaction_f32(
        &self, state: &Self::Tensor, context: &Self::Tensor,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.clifford_backend_interaction_f32(state, context, batch, channels, spatial, shift, inner, pooled, config)
    }

    fn clifford_interaction_backward_f32(
        &self, state: &Self::Tensor, context: &Self::Tensor, grad: &Self::Tensor,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.clifford_backend_interaction_backward_f32(state, context, grad, batch, channels, spatial, shift, inner, pooled, config)
    }

    fn clifford_residual_f32(
        &self, input: &Self::Tensor, mixed: &Self::Tensor, scale: &Self::Tensor,
        mask: &Self::Tensor, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.clifford_backend_residual_f32(input, mixed, scale, mask, batch, channels, spatial, config)
    }

    fn clifford_residual_backward_f32(
        &self, mixed: &Self::Tensor, grad: &Self::Tensor, scale: &Self::Tensor,
        mask: &Self::Tensor, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.clifford_backend_residual_backward_f32(mixed, grad, scale, mask, batch, channels, spatial, config)
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
        self.swin_backend_window_partition_f32(
            input, batch, channels, height, width, padded_height, padded_width, window, shift, config,
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
        self.swin_backend_window_unpartition_f32(
            windows, batch, channels, height, width, padded_height, padded_width, window, shift, clip, config,
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
        self.swin_backend_mhc_residual_f32(skip, branch, skip_scale, branch_scale, config)
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
        self.swin_backend_mhc_residual_backward_f32(
            skip, branch, grad, skip_scale, branch_scale, mix, post_skip, post_branch, config,
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
        self.swin_backend_window_attention_f32(
            qkv, relative_position_bias, mask, total_windows, spatial_windows, window, channels, heads, config,
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
        self.swin_backend_window_attention_backward_f32(
            qkv, probabilities, active, context_grad, total_windows, window, channels, heads, config,
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
        OxideBackend::wmhf_backend_apply_gate_f32(self, fuse, gate, projected, shortcut, freq_scale, config)
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
        OxideBackend::wmhf_backend_apply_gate_backward_f32(
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
        self.upsample_backend_f32(
            x, output_shape, channels, input_height, input_width, stride, scale, reverse, config,
        )
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
        self.upsample_backend_backward_f32(
            output_grad, input_shape, channels, output_height, output_width, stride, scale, config,
        )
    }

    fn shortcut_fit_f32(
        &self,
        x: &Self::Tensor,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.shortcut_fit_backend_f32(x, batch, output_shape, config)
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
        self.flash_attention_forward_backend_f32(qkv, output_shape, lse_shape, batch, tokens, channels, heads)
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
        self.laya_head_prepare_backend_f32(input, input_shape, type_embedding, hidden, qtype, config)
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
        self.laya_head_pack_backend_f32(tokens, batch, sequence, hidden, output_shape, config)
    }

    fn laya_scorer_unpack_f32(
        &self,
        input: &Self::Tensor,
        input_shape: &[usize],
        hidden: usize,
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.laya_scorer_unpack_backend_f32(input, input_shape, hidden, options, config)
    }

    fn laya_scorer_pack_f32(
        &self,
        logits: &Self::Tensor,
        batch: usize,
        options: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.laya_scorer_pack_backend_f32(logits, batch, options, output_shape, config)
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
        self.laya_encoder_embed_backend_f32(input, embeddings, input_shape, hidden, vocab, config)
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
        self.laya_encoder_attention_backend_f32(qkv, mask, cos, sin, batch, sequence, hidden, heads, sliding, window_radius, output_shape, config)
    }

    fn laya_encoder_geglu_f32(
        &self,
        input: &Self::Tensor,
        rows: usize,
        intermediate: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.laya_encoder_geglu_backend_f32(input, rows, intermediate, output_shape, config)
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
        self.laya_encoder_pack_backend_f32(tokens, batch, sequence, hidden, output_shape, config)
    }

    fn laya_scorer_seed_rows_f32(
        &self,
        seed: &Self::Tensor,
        batch: usize,
        options: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.laya_scorer_seed_rows_backend_f32(seed, batch, options, _config)
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
        self.laya_scorer_pack_input_grad_backend_f32(rows, batch, hidden, options, output_shape, config)
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
        self.laya_head_attention_backend_f32(qkv, mask, output_shape, batch, tokens, channels, heads, config)
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
        self.flash_attention_backward_backend_f32(qkv, context, output_grad, lse, qkv_grad_shape, batch, tokens, channels, heads)
    }

    fn softmax_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.softmax_backend_f32(x, shape, group_size, config)
    }

    fn softmax_backward_f32(
        &self,
        output: &Self::Tensor,
        output_grad: &Self::Tensor,
        shape: &[usize],
        group_size: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.softmax_backend_backward_f32(output, output_grad, shape, group_size, config)
    }

    fn centernet_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.centernet_forward_backend_f32(x, shape, classes, config)
    }

    fn modern_yolo_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.modern_yolo_forward_backend_f32(x, shape, classes, config)
    }

    fn segmentation_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        classes: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.segmentation_forward_backend_f32(x, shape, classes, temperature, config)
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
        self.jev_ce_forward_backend_f32(x, shape, groups, candidates, temperature, channel_major, config)
    }

    fn laya_loss_forward_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        max_options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.laya_loss_forward_backend_f32(x, shape, max_options, config)
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
        self.region_forward_backend_f32(x, shape, anchors, entries, classes, softmax_classes, config)
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
        self.region_backward_backend_f32(output, output_grad, shape, anchors, entries, classes, softmax_classes, config)
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
        self.reorg_backend_f32(x, input_shape, output_shape, stride, reverse, config)
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
        self.reorg_backend_backward_f32(
            output_grad,
            input_shape,
            output_shape,
            stride,
            reverse,
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
        self.channel_shuffle_backend_f32(x, shape, groups, inverse, config)
    }

    fn laya_act_head_backward_f32(
        &self,
        inputs: &lightnet_cuda::LayaActBackwardInputs<'_, Self::Tensor>,
        layout: &lightnet_cuda::host_ops::laya_act::MlpLayout,
        config: ElementwiseLaunchConfig,
    ) -> Result<lightnet_cuda::LayaActHeadGrad<Self::Tensor>, Self::Error> {
        self.laya_act_head_backward_backend_f32(inputs, layout, config)
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
        self.laya_act_cls_scatter_backend_f32(dcls, head_grad, batch, hidden, tokens, config)
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
        self.laya_act_features_backend_f32(
            head,
            scorer,
            markers,
            head_shape,
            scorer_shape,
            marker_shape,
            config,
        )
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
        self.foveation_film_forward_backend_f32(
            input, hf_params, w_embed, b_embed, w_gamma, b_gamma, w_beta, b_beta, shape, param_count, embed_dim, config,
        )
    }

    fn foveation_film_affine_f32(
        &self,
        input: &Self::Tensor,
        gamma: &Self::Tensor,
        beta: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        self.foveation_film_affine_backend_f32(
            input, gamma, beta, shape, config,
        )
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
        self.foveation_film_backward_backend_f32(
            input, output_gradient, hf_params, embedding, gamma, w_embed, w_gamma, w_beta, shape, param_count, embed_dim, config,
        )
    }

    fn foveation_film_affine_backward_f32(
        &self,
        input: &Self::Tensor,
        output_gradient: &Self::Tensor,
        gamma: &Self::Tensor,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor, Self::Tensor), Self::Error> {
        self.foveation_film_affine_backward_backend_f32(
            input, output_gradient, gamma, shape, config,
        )
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
        self.connected_backend_f32(x, weights, biases, batch, inputs, outputs, config)
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
        self.connected_backend_backward_f32(x, weights, delta, batch, inputs, outputs, config)
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
        self.batchnorm_train_forward_backend_f32(
            x, scales, biases, rolling_mean, rolling_variance, shape, keep, config,
        )
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
        self.batchnorm_train_backward_backend_f32(output_grad, x_hat, variance, scales, shape, config)
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
