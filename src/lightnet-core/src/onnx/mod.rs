//! ONNX export of a Darknet network (see tasks/rust/03_plans/misc/onnx.md).
mod export;
pub(crate) mod proto;

pub use export::{OnnxOptions, export_onnx};
