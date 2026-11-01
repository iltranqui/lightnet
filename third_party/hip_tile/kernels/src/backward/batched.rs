//! Batched backward launches for the gather/accumulate kernels: one launch per image over offset
//! views, the same convention as [`crate::yolo::batched`]. Every view describes **one** image of
//! a buffer whose images follow each other; the explicit strides cover views that are slices of a
//! larger tensor (route groups).

use hip_rt::{Module, Result, Stream};
use hip_tile::{Tensor, TensorMut};

use super::maxpool_bwd::launch_maxpool_bwd;
use super::route_bwd::launch_route_bwd;
use super::upsample_bwd::launch_upsample_bwd;
use crate::yolo::batched::{image, image_mut};
use crate::yolo::maxpool::PoolParams;

/// `launch_maxpool_bwd` for `batch` images (`dx` is accumulated into).
pub fn launch_maxpool_bwd_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    x: Tensor<f32>,
    dy: Tensor<f32>,
    dx: TensorMut<f32>,
    p: &PoolParams,
) -> Result<()> {
    for b in 0..batch {
        launch_maxpool_bwd(module, stream, image(x, b), image(dy, b), image_mut(dx, b), p)?;
    }
    Ok(())
}

/// `launch_upsample_bwd` for `batch` images (`dx` is accumulated into).
pub fn launch_upsample_bwd_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    dy: Tensor<f32>,
    dx: TensorMut<f32>,
    w: u32,
    stride: u32,
) -> Result<()> {
    for b in 0..batch {
        launch_upsample_bwd(module, stream, image(dy, b), image_mut(dx, b), w, stride)?;
    }
    Ok(())
}

/// `launch_route_bwd` for `batch` images: `src` is a channel block of image 0 of the route's
/// gradient, `dst` the matching block of image 0 of the source's gradient. Strides are the full
/// per-image sizes of those two buffers.
pub fn launch_route_bwd_batched(
    module: &Module,
    stream: Option<&Stream>,
    batch: u32,
    src: Tensor<f32>,
    src_image_stride: usize,
    dst: TensorMut<f32>,
    dst_image_stride: usize,
) -> Result<()> {
    for b in 0..batch {
        let s = Tensor::new(unsafe { src.ptr().add(b as usize * src_image_stride) }, src.rows(), src.cols(), src.ld());
        let d = TensorMut::new(unsafe { dst.ptr().add(b as usize * dst_image_stride) }, dst.rows(), dst.cols(), dst.ld());
        launch_route_bwd(module, stream, s, d)?;
    }
    Ok(())
}
