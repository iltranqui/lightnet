//! Shared model description and geometry for the Lightnet runtime.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

#[doc(hidden)]
pub mod activation_math;
pub mod cfg;
pub mod error;
#[doc(hidden)]
pub mod layers;
pub mod model;
pub mod names;
pub mod training_types;
#[doc(hidden)]
pub use layers::{clifford, depth_anything, dpt_head, kimi_kda, laya_act_head, laya_scorer, mambavision, swin, tucker_attention, vit};
#[doc(hidden)]
pub mod axis_fixture;
#[doc(hidden)]
pub mod detection_geometry;
