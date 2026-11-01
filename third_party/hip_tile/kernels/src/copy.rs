//! `y = x`, one 32x32 tile per block.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

#[cfg_attr(not(target_arch = "amdgpu"), allow(dead_code))]
const TILE: usize = 32;

#[kernel]
fn copy(x: Tensor<f32>, y: TensorMut<f32>) {
    let (bi, bj) = block_id();
    let t = x.load_tile::<TILE, TILE>(bi * TILE as u32, bj * TILE as u32);
    y.store_tile(bi * TILE as u32, bj * TILE as u32, &t);
}
