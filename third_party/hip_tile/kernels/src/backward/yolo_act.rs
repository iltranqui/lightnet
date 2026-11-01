//! Forward activation of a `[yolo]` layer (`yolo_layer.cpp:forward_yolo_layer`), elementwise:
//! `x`,`y` entries get `sigmoid(raw)·scale_x_y − 0.5·(scale_x_y − 1)`, `w`,`h` stay raw, and
//! objectness and class entries get `sigmoid(raw)`; a non-finite result becomes 0.
//!
//! The head is `[batch x n_mask x (classes+5) x plane]`; the entry of flat index `i` is
//! `((i % plain) / plane) % (classes+5)`.

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

#[cfg(target_arch = "amdgpu")]
use super::yolo_delta::expf;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct YoloActParams {
    /// Elements of one image's head (`n_mask * (classes+5) * plane`).
    pub plain: u32,
    pub plane: u32,
    /// `classes + 5`.
    pub entries: u32,
    pub scale_x_y: f32,
}

// SAFETY: `repr(C)` struct of `u32`/`f32`.
unsafe impl KernelArg for YoloActParams {}

#[kernel]
fn yolo_act(x: Tensor<f32>, y: TensorMut<f32>, p: YoloActParams) {
    let i = (block_id().1 * BLOCK_THREADS + thread_id()) as usize;
    if i < x.cols() as usize {
        let e = ((i % p.plain as usize) / p.plane as usize) % p.entries as usize;
        let raw = unsafe { *x.ptr().add(i) };
        let sig = 1.0 / (1.0 + expf(-raw));
        let val = if e < 2 {
            sig * p.scale_x_y - 0.5 * (p.scale_x_y - 1.0)
        } else if e < 4 {
            raw
        } else {
            sig
        };
        unsafe { *y.ptr().add(i) = if val.is_finite() { val } else { 0.0 } };
    }
}

/// `x`, `y`: flat `[batch*plain]` vectors.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_yolo_act(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    p: &YoloActParams,
) -> hip_rt::Result<()> {
    assert_eq!(x.cols(), y.cols(), "yolo_act: length mismatch");
    assert_eq!(x.cols() % p.plain, 0, "yolo_act: length is not batch*plain");
    let grid = (x.cols().div_ceil(BLOCK_THREADS), 1);
    yolo_act::launch(module, stream, grid, x, y, *p)
}
