//! {One line: what the kernel computes.} Launch with `grid = ({cols blocks}, {rows blocks})`.
//
// Template: copy to kernels/src/<name>.rs, add `pub mod <name>;` to kernels/src/lib.rs.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const BM: usize = 64;
pub const BN: usize = 64;

#[kernel(lds = 0)]
fn name(x: Tensor<f32>, y: TensorMut<f32>) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * BM as u32, bj * BN as u32);
    let t = x.load_tile::<BM, BN>(r0, c0);
    y.store_tile(r0, c0, &t.map(|v| v));
}
