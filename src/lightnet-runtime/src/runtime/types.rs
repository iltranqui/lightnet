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

pub use crate::model::Tensor;

impl From<PlanarRgbImage> for Tensor {
    fn from(value: PlanarRgbImage) -> Self {
        Self {
            shape: value.shape,
            data: value.data,
        }
    }
}

pub use lightnet_layers::params::*;

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
#[doc(hidden)] pub struct DeformConvRuntimeCache {
    pub math: crate::deform_conv::DeformConvForwardCache,
    pub output_clamp_slope: Vec<f32>,
    pub output_nondiff_boundary: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)] pub struct DsConvRuntimeCache {
    pub math: DsConvRuntimeMath,
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)] pub enum DsConvRuntimeMath {
    Single(crate::dsconv::DsConvForwardCache),
    Multi(crate::dsconv::DsConvMultiViewCache),
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)] pub struct GraphConvRuntimeCache {
    pub math: crate::graph_conv::GraphConvForwardCache,
    pub training: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)] pub struct WmhfConvRuntimeCache {
    pub input: Tensor,
    pub raw_output: Tensor,
    pub activation_input: Tensor,
    pub output: Tensor,
}

#[derive(Clone, Debug, PartialEq)]
#[doc(hidden)] pub struct WmhfRuntimeCache {
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
#[doc(hidden)]
pub struct HyperbolicFoveationForwardCache {
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
    #[doc(hidden)] pub dcnv4_caches: Vec<Option<crate::dcnv4::Dcnv4ForwardCache>>,
    #[doc(hidden)] pub deform_conv_caches: Vec<Option<DeformConvRuntimeCache>>,
    #[doc(hidden)] pub dsconv_caches: Vec<Option<DsConvRuntimeCache>>,
    #[doc(hidden)] pub graph_conv_caches: Vec<Option<GraphConvRuntimeCache>>,
    #[doc(hidden)] pub wmhf_caches: Vec<Option<WmhfRuntimeCache>>,
    #[doc(hidden)] pub clifford_caches: Vec<Option<CliffordRuntimeCache>>,
    #[doc(hidden)] pub recursive_block_caches: Vec<Option<RecursiveBlockForwardCache>>,
    #[doc(hidden)]
    pub hyperbolic_foveation_caches: Vec<Option<HyperbolicFoveationForwardCache>>,
    #[doc(hidden)] pub foveation_film_caches: Vec<Option<crate::foveation_film::FoveationFilmForwardCache>>,
    pub vit_caches: Vec<Option<crate::vit::VitForwardCache>>,
    #[doc(hidden)] pub depth_anything_caches: Vec<Option<crate::depth_anything::DepthAnythingForwardCache>>,
    #[doc(hidden)] pub dpt_head_caches: Vec<Option<crate::dpt_head::DptHeadForwardCache>>,
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
