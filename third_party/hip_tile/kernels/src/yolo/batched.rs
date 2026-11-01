//! Batched forward launches: one launch per image over offset views.
//!
//! The single-image kernels are untouched (the batch-1 inference path stays byte-identical).
//! Every view below describes **one** image; image `b` sits at `+ b * rows * ld`, the same
//! convention the backward kernels stride over. Launches queue on one stream, so a batch costs
//! `batch` launches per layer and no extra synchronisation.

use hip_rt::{Module, Result, Stream};
use hip_tile::{Tensor, TensorMut};

use super::concat::launch_concat;
use super::conv::{ConvParams, launch_conv};
use super::maxpool::{PoolParams, launch_maxpool};
use super::upsample::launch_upsample;

/// Image `b` of a batch laid out back to back behind `t`.
pub(crate) fn image<T>(t: Tensor<T>, b: u32) -> Tensor<T> {
    let off = b as usize * t.rows() as usize * t.ld() as usize;
    Tensor::new(unsafe { t.ptr().add(off) }, t.rows(), t.cols(), t.ld())
}

pub(crate) fn image_mut<T>(t: TensorMut<T>, b: u32) -> TensorMut<T> {
    let off = b as usize * t.rows() as usize * t.ld() as usize;
    TensorMut::new(unsafe { t.ptr().add(off) }, t.rows(), t.cols(), t.ld())
}

/// `launch_conv` for `batch` images. `w` and `bias` are shared by all images.
#[allow(clippy::too_many_arguments)]
pub fn launch_conv_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    x: Tensor<f32>,
    w: Tensor<f32>,
    bias: Tensor<f32>,
    y: TensorMut<f32>,
    p: &ConvParams,
) -> Result<()> {
    for b in 0..batch {
        launch_conv(module, stream, image(x, b), w, bias, image_mut(y, b), p)?;
    }
    Ok(())
}

/// `launch_maxpool` for `batch` images.
pub fn launch_maxpool_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    x: Tensor<f32>,
    y: TensorMut<f32>,
    p: &PoolParams,
) -> Result<()> {
    for b in 0..batch {
        launch_maxpool(module, stream, image(x, b), image_mut(y, b), p)?;
    }
    Ok(())
}

/// `launch_upsample` for `batch` images.
pub fn launch_upsample_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    x: Tensor<f32>,
    y: TensorMut<f32>,
    w: u32,
    stride: u32,
) -> Result<()> {
    for b in 0..batch {
        launch_upsample(module, stream, image(x, b), image_mut(y, b), w, stride)?;
    }
    Ok(())
}

/// `launch_concat` for `batch` images: `x` and `y` are channel-slice views of image 0 of the
/// source and the destination. The per-image strides come from the full buffers (`Csrc * ld`,
/// `Cdst * ld`), because a route may read a group or write a slice of a larger tensor.
pub fn launch_concat_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    x: Tensor<f32>,
    x_image_stride: usize,
    y: TensorMut<f32>,
    y_image_stride: usize,
) -> Result<()> {
    for b in 0..batch {
        let xb = Tensor::new(unsafe { x.ptr().add(b as usize * x_image_stride) }, x.rows(), x.cols(), x.ld());
        let yb = TensorMut::new(
            unsafe { y.ptr().add(b as usize * y_image_stride) },
            y.rows(),
            y.cols(),
            y.ld(),
        );
        launch_concat(module, stream, xb, yb)?;
    }
    Ok(())
}
