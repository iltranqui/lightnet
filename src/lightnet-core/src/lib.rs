//! CPU-first Rust implementation of the Darknet/YOLO runtime.
//!
//! This crate is a facade. The code lives in four crates, each depending only
//! on the ones before it:
//!
//! 1. `lightnet-model`: `.cfg` parsing, the network description, shared types.
//! 2. `lightnet-layers`: CPU reference math for each layer family.
//! 3. `lightnet-runtime`: CPU and backend executors, per-layer GPU glue,
//!    `.weights` I/O, initialisation, Muon state, ONNX export.
//! 4. `lightnet-train`: losses, augmentation, detection decode, evaluation.
//!
//! `pub use lightnet_train::*` re-exports the whole stack under the old
//! `lightnet_core::...` paths, so the CLI, the examples and the Oxide crate
//! need no source change. Unit tests that span several of these crates live
//! here, next to the facade.

pub use lightnet_train::*;

#[cfg(test)]
mod layer_tests;
#[cfg(test)]
mod model_tests;
#[cfg(test)]
mod runtime_tests;
