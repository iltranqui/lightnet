//! Pure-Rust Darknet YOLO: cfg/weights loading, BN folding, a CPU reference forward, and decode + NMS.
//!
//! For a GPU executor: build a [`Model`], then for every [`LayerKind::Conv`] use the folded
//! `weights` (contiguous `[out][in * kh * kw]`) and `bias`; per-layer shapes are in
//! [`Layer::in_shape`] / [`Layer::out_shape`]. [`reference::forward`] gives every layer's output as
//! ground truth; feed the `[yolo]` layer inputs to [`yolo::detect_heads`].

pub mod augment;
pub mod cfg;
pub mod data;
pub mod draw;
mod error;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod image;
pub mod metrics;
pub mod model;
pub mod reference;
pub mod train;
pub mod weights;
pub mod yolo;

pub use cfg::{Activation, Cfg};
pub use error::{Error, Result};
pub use model::{Conv, Layer, LayerKind, Model, Shape};
pub use yolo::{DetectOptions, Detection};

/// Read a Darknet `.names` file (one class per line).
pub fn load_names(path: impl AsRef<std::path::Path>) -> Result<Vec<String>> {
    let p = path.as_ref();
    let s = std::fs::read_to_string(p).map_err(|source| Error::Io { path: p.display().to_string(), source })?;
    Ok(s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
}

/// CPU pipeline for one image: resize to the net size, forward, decode, NMS.
/// Returned boxes are normalized to the image.
pub fn detect_cpu(model: &Model, img: &image::Image, opts: &DetectOptions) -> Result<Vec<Detection>> {
    let sized = img.resize(model.net.width, model.net.height);
    let outs = reference::forward(model, &sized.data)?;
    Ok(yolo::detect(model, &outs, opts))
}

/// GPU pipeline for one image: same as [`detect_cpu`] with the forward on the GPU.
#[cfg(feature = "gpu")]
pub fn detect_gpu(model: &Model, gpu: &gpu::GpuModel, img: &image::Image, opts: &DetectOptions) -> Result<Vec<Detection>> {
    let sized = img.resize(model.net.width, model.net.height);
    let heads = gpu.forward(&sized.data)?;
    let refs: Vec<&[f32]> = heads.iter().map(Vec::as_slice).collect();
    Ok(yolo::detect_heads(model, &refs, opts))
}
