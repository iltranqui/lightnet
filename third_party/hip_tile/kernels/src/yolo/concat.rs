//! Route/concat: copy a `[C x HW]` tensor into a channel slice of a bigger buffer.
//!
//! Pass `y` as a view at the slice's first channel (`ptr.add(ch0 * ld)`, `rows = C`,
//! same `ld` as the full buffer). Launch with `grid = (ceil(cols / TILE_N), ceil(rows / TILE_M))`.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;

#[kernel]
fn concat(x: Tensor<f32>, y: TensorMut<f32>) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * TILE_M as u32, bj * TILE_N as u32);
    let t = x.load_tile::<TILE_M, TILE_N>(r0, c0);
    y.store_tile(r0, c0, &t);
}

/// Copy `x` into `y` (same `rows x cols`).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_concat(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
) -> hip_rt::Result<()> {
    assert!(
        x.rows() == y.rows() && x.cols() == y.cols(),
        "concat: shape mismatch"
    );
    let grid = (
        x.cols().div_ceil(TILE_N as u32),
        x.rows().div_ceil(TILE_M as u32),
    );
    concat::launch(module, stream, grid, x, y)
}
