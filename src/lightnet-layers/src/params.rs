//! Shared layer parameter payloads and training buffers.

#[doc(hidden)]
pub const BATCH_NORM_EPSILON: f32 = 0.000_01;

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

/// Weighted `[shortcut]` parameters (tasks/rust/04_reference/layers.md Task 6).
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

#[derive(Clone, Debug, PartialEq)]
pub struct EmlConvolutionalParams {
    pub input: ConvolutionalParams,
    pub self_params: ConvolutionalParams,
}

/// `[graph_conv]` projection, optional self projection, and learned edge scorer.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphConvParams {
    pub projection: ConvolutionalParams,
    pub self_weights: Option<Vec<f32>>,
    pub edge_kernel: Option<Vec<f32>>,
    pub edge_biases: Option<Vec<f32>>,
}

/// `[dcnv4]` parameters. `dummy_weights` are compatibility-only; the offset
/// projection is the learned spatial parameter block.
#[derive(Clone, Debug, PartialEq)]
pub struct Dcnv4Params {
    pub biases: Vec<f32>,
    pub batch_norm: Option<BatchNormParams>,
    pub dummy_weights: Vec<f32>,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
}

/// `[deform_conv]` main convolution plus full-channel offset and optional mask predictors.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformConvParams {
    pub main: ConvolutionalParams,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
    pub mask_weights: Option<Vec<f32>>,
    pub mask_biases: Option<Vec<f32>>,
}

/// `[dsconv]` collapse convolution and its shared offset predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct DsConvParams {
    pub collapse: ConvolutionalParams,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
    pub extra_offset_weights: Vec<Vec<f32>>,
}

/// `[wmhf]`: seven independent convolution blocks followed by four scan
/// vectors (`raw_A`, `B`, `C`, `D`).
#[derive(Clone, Debug, PartialEq)]
pub struct WmhfParams {
    pub pre: ConvolutionalParams,
    pub local_dw3: ConvolutionalParams,
    pub local_dw5: ConvolutionalParams,
    pub local_dw7: ConvolutionalParams,
    pub local_mix: ConvolutionalParams,
    pub hf_gate: ConvolutionalParams,
    pub fuse: ConvolutionalParams,
    pub scan_weights: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecursiveBlockParams {
    pub body: Vec<LayerParams>,
    /// Ouroboros controller dense matrix; omitted from Darknet `.weights` by C++.
    pub controller_weights: Vec<f32>,
    /// Controller gamma/gate biases; omitted from Darknet `.weights` by C++.
    pub controller_biases: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicFoveationParams {
    pub hf_params: Vec<f32>,
    pub pred_w1: Vec<f32>,
    pub pred_b1: Vec<f32>,
    pub pred_w2: Vec<f32>,
    pub pred_b2: Vec<f32>,
}

/// `[foveation_film]`'s embedding and per-channel affine projection parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct FoveationFilmParams {
    pub w_embed: Vec<f32>,
    pub b_embed: Vec<f32>,
    pub w_gamma: Vec<f32>,
    pub b_gamma: Vec<f32>,
    pub w_beta: Vec<f32>,
    pub b_beta: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FoveationFilmTrainBuffers {
    pub w_embed_updates: Vec<f32>,
    pub b_embed_updates: Vec<f32>,
    pub w_gamma_updates: Vec<f32>,
    pub b_gamma_updates: Vec<f32>,
    pub w_beta_updates: Vec<f32>,
    pub b_beta_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicFoveationTrainBuffers {
    pub hf_param_updates: Vec<f32>,
    pub pred_w1_updates: Vec<f32>,
    pub pred_b1_updates: Vec<f32>,
    pub pred_w2_updates: Vec<f32>,
    pub pred_b2_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecursiveBlockTrainBuffers {
    pub body: Vec<LayerTrainBuffers>,
    pub controller_weight_updates: Vec<f32>,
    pub controller_bias_updates: Vec<f32>,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum LayerParams {
    None,
    RecursiveBlock(RecursiveBlockParams),
    FoveationFilm(FoveationFilmParams),
    HyperbolicFoveation(HyperbolicFoveationParams),
    BatchNorm(BatchNormParams),
    Convolutional(ConvolutionalParams),
    Connected(ConnectedParams),
    Local(LocalParams),
    Shortcut(ShortcutParams),
    Gru(GruParams),
    Rnn(RnnParams),
    Lstm(LstmParams),
    Crnn(CrnnParams),
    /// `[laya_scorer]`'s six frozen tensors (loaded from its `.weights` blob).
    LayaScorer(crate::laya_scorer::LayaScorerWeights),
    /// `[laya_act_head]`'s four frozen tensors (loaded from its `.weights` blob).
    LayaActHead(crate::laya_act_head::LayaActHeadWeights),
    /// `[laya_encoder]`'s frozen ModernBERT weights (loaded from its `.weights` blob).
    LayaEncoder(crate::laya_encoder::LayaEncoderWeights),
    /// `[laya_head]`'s frozen decision-head weights (loaded from its `.weights` blob).
    LayaHead(crate::laya_head::LayaHeadWeights),
    /// `[depth_anything]` frozen encoder weights in source blob order.
    DepthAnything(crate::depth_anything::DepthAnythingWeights),
    /// `[dpt_head]` trainable 66-slot source-order neck/head arena.
    DptHead(crate::dpt_head::DptHeadWeights),
    /// `[vit]` global transformer parameters in the C++ save order.
    Vit(crate::vit::VitWeights),
    /// `[swin]` parameters in the Rust-defined Swin blob order.
    Swin(crate::swin::SwinWeights),
    /// `[tucker_attention]`: bias followed by flat Q/K/V/O factor weights.
    TuckerAttention(crate::tucker_attention::TuckerAttentionParams),
    /// `[mambavision]`: all trainable tensors in source `.weights` order.
    MambaVision(crate::mambavision::MambaVisionParams),
    /// `[kimi_kda]`: contiguous source-order biases followed by flat weights.
    KimiKda(crate::kimi_kda::KimiKdaParams),
    /// `[moe_convolutional]`: one convolution per expert plus the router's
    /// `[experts, channels]` weights and `[experts]` biases.
    Moe(MoeParams),
    /// `[eml_convolutional]`: the `input_layer` and `self_layer` conv blocks.
    EmlConvolutional(EmlConvolutionalParams),
    Dcnv4(Dcnv4Params),
    GraphConv(GraphConvParams),
    DeformConv(DeformConvParams),
    DsConv(DsConvParams),
    Wmhf(WmhfParams),
    Clifford(crate::clifford_params::CliffordParams),
}

/// The parameters of one `[moe_convolutional]` layer.
///
/// `experts` holds `spec.experts` independent convolutions, so the layer can
/// reuse the ordinary convolution forward/backward/update verbatim; the router
/// is the only genuinely new state.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeParams {
    pub router_weights: Vec<f32>,
    pub router_biases: Vec<f32>,
    pub experts: Vec<ConvolutionalParams>,
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

/// `[moe_convolutional]`'s accumulators: the router's own updates plus one
/// convolution update set per expert.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeTrainBuffers {
    pub router_weight_updates: Vec<f32>,
    pub router_bias_updates: Vec<f32>,
    pub experts: Vec<ConvolutionalTrainBuffers>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct EmlConvolutionalTrainBuffers {
    pub input: ConvolutionalTrainBuffers,
    pub self_params: ConvolutionalTrainBuffers,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphConvTrainBuffers {
    pub projection: ConvolutionalTrainBuffers,
    pub self_weight_updates: Option<Vec<f32>>,
    pub edge_kernel_updates: Option<Vec<f32>>,
    pub edge_bias_updates: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dcnv4TrainBuffers {
    pub bias_updates: Vec<f32>,
    pub batch_norm: Option<BatchNormTrainBuffers>,
    pub offset_weight_updates: Vec<f32>,
    pub offset_bias_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeformConvTrainBuffers {
    pub main: ConvolutionalTrainBuffers,
    pub offset_weight_updates: Vec<f32>,
    pub offset_bias_updates: Vec<f32>,
    pub mask_weight_updates: Option<Vec<f32>>,
    pub mask_bias_updates: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DsConvTrainBuffers {
    pub collapse: ConvolutionalTrainBuffers,
    pub offset_weight_updates: Vec<f32>,
    pub offset_bias_updates: Vec<f32>,
    pub extra_offset_weight_updates: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WmhfTrainBuffers {
    pub pre: ConvolutionalTrainBuffers,
    pub local_dw3: ConvolutionalTrainBuffers,
    pub local_dw5: ConvolutionalTrainBuffers,
    pub local_dw7: ConvolutionalTrainBuffers,
    pub local_mix: ConvolutionalTrainBuffers,
    pub hf_gate: ConvolutionalTrainBuffers,
    pub fuse: ConvolutionalTrainBuffers,
    pub scan_weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitTrainBuffers {
    /// Descending update directions (negative mathematical gradients), in
    /// `VitWeights` block shape/order.
    pub updates: crate::vit::VitWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwinTrainBuffers {
    /// Accumulated descending update directions in the Rust Swin blob order.
    pub updates: crate::swin::SwinWeights,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuckerAttentionTrainBuffers {
    /// Accumulated descending update directions for biases and flat weights.
    pub bias_updates: Vec<f32>,
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionTrainBuffers {
    /// Accumulated update directions in the MambaVision source blob order.
    pub updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KimiKdaTrainBuffers {
    /// Descending update directions; weights and biases retain separate decay behavior.
    pub bias_updates: Vec<f32>,
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DptHeadTrainBuffers {
    /// Momentum/gradient accumulator in the 66 source tensor-slot order.
    pub weight_updates: Vec<f32>,
    /// Whether this update window received a nonzero Rust-mode DPT gradient.
    pub has_gradient: bool,
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum LayerTrainBuffers {
    None,
    RecursiveBlock(RecursiveBlockTrainBuffers),
    FoveationFilm(FoveationFilmTrainBuffers),
    HyperbolicFoveation(HyperbolicFoveationTrainBuffers),
    BatchNorm(StandaloneBatchNormTrainBuffers),
    Convolutional(ConvolutionalTrainBuffers),
    Connected(ConnectedTrainBuffers),
    Local(LocalTrainBuffers),
    Shortcut(ShortcutTrainBuffers),
    Gru(GruTrainBuffers),
    Rnn(RnnTrainBuffers),
    Lstm(LstmTrainBuffers),
    Crnn(CrnnTrainBuffers),
    Moe(MoeTrainBuffers),
    EmlConvolutional(EmlConvolutionalTrainBuffers),
    Dcnv4(Dcnv4TrainBuffers),
    GraphConv(GraphConvTrainBuffers),
    DeformConv(DeformConvTrainBuffers),
    DsConv(DsConvTrainBuffers),
    Wmhf(WmhfTrainBuffers),
    Clifford(crate::clifford_params::CliffordTrainBuffers),
    Vit(VitTrainBuffers),
    Swin(SwinTrainBuffers),
    TuckerAttention(TuckerAttentionTrainBuffers),
    MambaVision(MambaVisionTrainBuffers),
    KimiKda(KimiKdaTrainBuffers),
    DptHead(DptHeadTrainBuffers),
}

