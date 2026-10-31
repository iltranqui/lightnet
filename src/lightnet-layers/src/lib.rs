#![feature(float_erf)]
//! CPU reference layers and their parameter data.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub use lightnet_model::{activation_math, cfg, detection_geometry, error, model, names, training_types};
#[cfg(test)]
pub use lightnet_model::axis_fixture;
pub use lightnet_model::cfg::ConfigFile;
pub use lightnet_model::model::{Network, TensorShape};
pub mod params;
#[doc(hidden)]
pub mod clifford_params;
pub mod carpn;
pub mod centernet;
pub mod channel_shuffle;
pub mod channel_slice;
pub mod contrastive;
pub mod dcnv4;
pub mod deconvolutional;
pub mod deform_conv;
pub mod depth_anything;
pub mod dpt_head;
pub mod dsconv;
pub mod eml_conv_math;
pub mod foveation_film;
pub mod graph_conv;
pub mod gru;
pub mod hyperbolic_foveation;
pub mod jev_ce;
pub mod kimi_kda;
pub mod laya_act_head;
pub mod laya_encoder;
pub mod laya_head;
pub mod laya_loss;
pub mod laya_marker_gather;
pub mod laya_scorer;
pub mod mambavision;
pub mod maxpool_depth;
pub mod modern_yolo;
pub mod moe_conv;
pub mod omd_loss;
pub mod segmentation;
pub mod shortcut_weights;
pub mod swin;
pub mod tucker_attention;
pub mod vc_attention;
pub mod vit;
pub mod wmhf;
