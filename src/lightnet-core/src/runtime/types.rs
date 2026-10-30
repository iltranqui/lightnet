use super::*;

/// Backward-pass policy for the few ops where C++ Darknet's hand-written gradient is not the
/// exact derivative of its forward. `Analytical` (default) is the exact derivative, `Cpp`
/// reproduces the C++ shortcut for training parity. Selected by `LIGHTNET_GRAD_MODE=analytical|cpp`
/// and read at each backward. Affects only: softmax (C++ passes `dy` through; its cost layer
/// folds the Jacobian in), `scale_channels` with `scale_wh=1` (C++ divides the scale gradient
/// by `w*h`), and frozen `[depth_anything]` (whose source backward callback is a no-op). See
/// `tasks/rust/todo.md` "Analytical gradients everywhere" for the full audit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GradMode {
    #[default]
    Analytical,
    Cpp,
}

impl GradMode {
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("LIGHTNET_GRAD_MODE").unwrap_or_default())
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

    pub(super) fn channel_area(&self) -> usize {
        self.shape.width * self.shape.height
    }

    pub(super) fn batch_stride(&self) -> usize {
        self.shape.channels * self.channel_area()
    }

    pub(crate) fn offset(&self, batch: usize, channel: usize, y: usize, x: usize) -> usize {
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
    Clifford(CliffordParams),
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
    Clifford(CliffordTrainBuffers),
    Vit(VitTrainBuffers),
    Swin(SwinTrainBuffers),
    TuckerAttention(TuckerAttentionTrainBuffers),
    MambaVision(MambaVisionTrainBuffers),
    KimiKda(KimiKdaTrainBuffers),
    DptHead(DptHeadTrainBuffers),
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

/// A raw-logit projected cuboid head, never placed in the 2D head vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Yolo3dHeadOutput {
    pub layer_index: usize,
    pub output: Tensor,
}

/// One of the four modern anchor-free heads (`[yolox]`, `[ppyoloe]`,
/// `[yolonas]`, `[centernet]`) captured by a forward pass so `detect.rs` can
/// decode it.  Same shape as [`YoloHeadOutput`]; kept separate so the decode
/// path can dispatch on the layer kind without re-deriving it.
#[derive(Clone, Debug, PartialEq)]
pub struct ModernHeadOutput {
    pub layer_index: usize,
    pub output: Tensor,
}

/// True for the modern anchor-free heads that `detect.rs` can decode.
pub fn is_modern_head(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Yolox(_)
            | LayerKind::Ppyoloe(_)
            | LayerKind::Yolonas(_)
            | LayerKind::CenterNet(_)
    )
}

/// True for the heads whose immediately preceding convolution must stay off the
/// quantised (FP8/FP4) tensor-core convolution path.
///
/// The FP8 policy is "every convolution except the one immediately before a
/// detection or segmentation head", so this predicate is what the conv layer
/// consults, and what the FP8 policy tests pin.
pub fn is_quantized_conv_boundary_head(kind: &LayerKind) -> bool {
    matches!(
        kind,
        LayerKind::Yolo(_)
            | LayerKind::ObbYolo(_)
            | LayerKind::GaussianYolo(_)
            | LayerKind::Region(_)
            | LayerKind::Segmentation(_)
    ) || is_modern_head(kind)
}

/// True for a `[convolutional]` layer whose successor is a detection or
/// segmentation head, i.e. the convolution that must keep f32 conv GEMMs.
pub fn conv_feeds_detection_head(network: &Network, layer_index: usize) -> bool {
    if !matches!(
        network.layers.get(layer_index).map(|layer| &layer.kind),
        Some(LayerKind::Convolutional(_))
    ) {
        return false;
    }
    network
        .layers
        .get(layer_index + 1)
        .is_some_and(|next| is_quantized_conv_boundary_head(&next.kind))
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

/// CPU fallback cache for `[moe_convolutional]` backward. The device path keeps
/// equivalent tensors in [`GpuMoeBackwardCache`] to avoid per-expert
/// host/device round trips during training.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeForwardCache {
    pub gates: Vec<f32>,
    pub expert_outputs: Vec<Tensor>,
    pub expert_activation_inputs: Vec<Option<Tensor>>,
    pub pooled: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EmlForwardCache {
    pub input_branch_output: Tensor,
    pub self_branch_output: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeformConvRuntimeCache {
    pub math: crate::deform_conv::DeformConvForwardCache,
    pub output_clamp_slope: Vec<f32>,
    pub output_nondiff_boundary: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DsConvRuntimeCache {
    pub math: DsConvRuntimeMath,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DsConvRuntimeMath {
    Single(crate::dsconv::DsConvForwardCache),
    Multi(crate::dsconv::DsConvMultiViewCache),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphConvRuntimeCache {
    pub math: crate::graph_conv::GraphConvForwardCache,
    pub training: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WmhfConvRuntimeCache {
    pub input: Tensor,
    pub raw_output: Tensor,
    pub activation_input: Tensor,
    pub output: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WmhfRuntimeCache {
    pub pre: WmhfConvRuntimeCache,
    pub local_dw3: WmhfConvRuntimeCache,
    pub local_dw5: WmhfConvRuntimeCache,
    pub local_dw7: WmhfConvRuntimeCache,
    pub local_mix: WmhfConvRuntimeCache,
    pub hf_gate: WmhfConvRuntimeCache,
    pub fuse: WmhfConvRuntimeCache,
    pub dwt: crate::wmhf::HaarCoefficients,
    pub training: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecursiveBlockForwardCache {
    /// Body-network tapes, one per shared-weight recurrence pass.
    pub body_passes: Vec<ForwardResult>,
    /// Hidden inputs h_0..h_{N-1}; h_0 is the outer layer input.
    pub hidden_inputs: Vec<Tensor>,
    /// Controller input/output vectors per pass; empty for static recurrence.
    pub controller_inputs: Vec<Vec<f32>>,
    pub controller_outputs: Vec<Vec<f32>>,
    /// Raw recurrence outputs before between-pass lattice projection.
    pub lattice_inputs: Vec<Option<Tensor>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HyperbolicFoveationForwardCache {
    pub grid: Vec<f32>,
    /// Raw global or constrained predictor values per batch item.
    pub params_per_batch: Vec<f32>,
    pub predictor: Option<crate::hyperbolic_foveation::PredictorForwardCache>,
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
    /// `[moe_convolutional]` routers: the per-image gates and each expert's
    /// output, which the backward pass needs for the gate dot products and for
    /// the scaled expert gradients.
    pub moe_caches: Vec<Option<MoeForwardCache>>,
    pub eml_caches: Vec<Option<EmlForwardCache>>,
    pub(crate) dcnv4_caches: Vec<Option<crate::dcnv4::Dcnv4ForwardCache>>,
    pub(crate) deform_conv_caches: Vec<Option<DeformConvRuntimeCache>>,
    pub(crate) dsconv_caches: Vec<Option<DsConvRuntimeCache>>,
    pub(crate) graph_conv_caches: Vec<Option<GraphConvRuntimeCache>>,
    pub(crate) wmhf_caches: Vec<Option<WmhfRuntimeCache>>,
    pub(crate) clifford_caches: Vec<Option<CliffordRuntimeCache>>,
    pub(crate) recursive_block_caches: Vec<Option<RecursiveBlockForwardCache>>,
    pub(crate) hyperbolic_foveation_caches: Vec<Option<HyperbolicFoveationForwardCache>>,
    pub(crate) foveation_film_caches: Vec<Option<crate::foveation_film::FoveationFilmForwardCache>>,
    pub vit_caches: Vec<Option<crate::vit::VitForwardCache>>,
    pub(crate) depth_anything_caches: Vec<Option<crate::depth_anything::DepthAnythingForwardCache>>,
    pub(crate) dpt_head_caches: Vec<Option<crate::dpt_head::DptHeadForwardCache>>,
    pub swin_caches: Vec<Option<crate::swin_block::SwinForwardCache>>,
    pub tucker_attention_caches: Vec<Option<crate::tucker_attention::TuckerAttentionForwardCache>>,
    /// Forward tapes include the explicit recurrent scan state for MambaVision.
    pub mambavision_caches: Vec<Option<crate::mambavision::MambaVisionForward>>,
    /// Kimi KDA per-layer forward tape; its cache also exposes the auxiliary router cost.
    pub kimi_kda_caches: Vec<Option<crate::kimi_kda::KimiKdaForwardCache>>,
    /// Valid proposal metadata and Stage-2 raw outputs for each `[obb_refine]` layer.
    pub obb_refine_outputs: Vec<crate::carpn::ObbRefineOutput>,
    pub yolo_heads: Vec<YoloHeadOutput>,
    pub yolo_3d_heads: Vec<Yolo3dHeadOutput>,
    /// The modern anchor-free heads, in layer order.
    pub modern_heads: Vec<ModernHeadOutput>,
}

impl ForwardResult {
    /// Build an inference result containing only detector heads and no per-layer caches.
    pub fn from_heads(
        yolo_heads: Vec<YoloHeadOutput>,
        modern_heads: Vec<ModernHeadOutput>,
    ) -> Self {
        Self {
            layer_outputs: Vec::new(),
            layer_activation_inputs: Vec::new(),
            dropout_masks: Vec::new(),
            crop_caches: Vec::new(),
            recurrent_caches: Vec::new(),
            moe_caches: Vec::new(),
            eml_caches: Vec::new(),
            dcnv4_caches: Vec::new(),
            deform_conv_caches: Vec::new(),
            dsconv_caches: Vec::new(),
            graph_conv_caches: Vec::new(),
            wmhf_caches: Vec::new(),
            clifford_caches: Vec::new(),
            recursive_block_caches: Vec::new(),
            hyperbolic_foveation_caches: Vec::new(),
            foveation_film_caches: Vec::new(),
            vit_caches: Vec::new(),
            depth_anything_caches: Vec::new(),
            dpt_head_caches: Vec::new(),
            swin_caches: Vec::new(),
            tucker_attention_caches: Vec::new(),
            mambavision_caches: Vec::new(),
            kimi_kda_caches: Vec::new(),
            obb_refine_outputs: Vec::new(),
            yolo_heads,
            yolo_3d_heads: Vec::new(),
            modern_heads,
        }
    }

    pub fn final_output(&self) -> Option<&Tensor> {
        self.layer_outputs.last()
    }
}

#[cfg(test)]
mod forward_result_from_heads_tests {
    use super::ForwardResult;

    #[test]
    fn forward_result_from_heads_has_empty_caches() {
        let result = ForwardResult::from_heads(Vec::new(), Vec::new());
        assert!(result.layer_outputs.is_empty());
        assert!(result.layer_activation_inputs.is_empty());
        assert!(result.dropout_masks.is_empty());
        assert!(result.crop_caches.is_empty());
        assert!(result.recurrent_caches.is_empty());
        assert!(result.moe_caches.is_empty());
        assert!(result.eml_caches.is_empty());
        assert!(result.dcnv4_caches.is_empty());
        assert!(result.deform_conv_caches.is_empty());
        assert!(result.dsconv_caches.is_empty());
        assert!(result.graph_conv_caches.is_empty());
        assert!(result.wmhf_caches.is_empty());
        assert!(result.hyperbolic_foveation_caches.is_empty());
        assert!(result.foveation_film_caches.is_empty());
        assert!(result.vit_caches.is_empty());
        assert!(result.depth_anything_caches.is_empty());
        assert!(result.dpt_head_caches.is_empty());
        assert!(result.swin_caches.is_empty());
        assert!(result.tucker_attention_caches.is_empty());
        assert!(result.mambavision_caches.is_empty());
        assert!(result.kimi_kda_caches.is_empty());
        assert!(result.obb_refine_outputs.is_empty());
        assert!(result.yolo_heads.is_empty());
        assert!(result.modern_heads.is_empty());
        assert!(result.final_output().is_none());
    }
}
