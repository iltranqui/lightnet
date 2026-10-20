//! Optional `cutile-rs` device backend for the Rust Darknet executor.
//!
//! The bridge owns CUDA context/stream creation and translates the generic
//! `darknet-cuda` operations into cuTile kernels.  It remains outside the main
//! workspace because CUDA headers, libclang, and a compatible driver are not
//! required for CPU-only development.

#![allow(clippy::too_many_arguments)]

mod aot;
mod backend;
mod error;
mod kernels;
mod ops_conv;
mod ops_crop;
mod ops_elementwise;
mod ops_optim;
mod ops_pool;
mod ops_route;
mod ops_upsample;
mod trait_impl;
mod validate;

#[cfg(test)]
mod tests;

pub use aot::parse_aot_target_sms;
pub use backend::{CutileBackend, CutileTensorF32};
pub use darknet_cuda::{CUTILE_RS_REFERENCE, DEFAULT_TILE_1D, default_cutile_plan};
pub use error::CutileBackendError;
