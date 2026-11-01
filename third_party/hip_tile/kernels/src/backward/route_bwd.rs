//! `[route]` backward: add a channel block of the downstream gradient into a source's gradient.
//!
//! Darknet does `axpy` per source (`route_layer.cpp:140-159`); a group is a contiguous block of
//! channels, so the destination is a view at `group_id * c_per_group` with the same `ld`.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 64;
pub const TILE_N: usize = 128;

/// `dst += src` for whole channel blocks (route / concat fan-in).
#[kernel]
fn route_bwd(src: Tensor<f32>, dst: TensorMut<f32>) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let t = src.load_tile::<TILE_M, TILE_N>(r0, c0);
    dst.store_tile_add(r0, c0, &t);
}

/// `src: [Cg x HW]` (a view into the route's own gradient), `dst: [Cg x HW]` (a view into the
/// source layer's gradient) — accumulated into, never overwritten.
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_route_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    src: hip_tile::Tensor<f32>,
    dst: hip_tile::TensorMut<f32>,
) -> hip_rt::Result<()> {
    assert_eq!((src.rows(), src.cols()), (dst.rows(), dst.cols()), "route_bwd: shape mismatch");
    let grid = (dst.cols().div_ceil(TILE_N as u32), dst.rows().div_ceil(TILE_M as u32));
    route_bwd::launch(module, stream, grid, src, dst)
}
