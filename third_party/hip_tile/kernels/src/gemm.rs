//! `c = a * b` (row-major, f32). One 128x128 tile of `c` per block, K swept in steps of 16.
//!
//! Launch with `grid = (ceil(n / BN), ceil(m / BM))`, see `BN` and `BM`.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const BM: usize = 128;
pub const BN: usize = 128;
pub const BK: usize = 16;

#[kernel(lds = mma_lds_bytes::<f32>(BM, BN, BK))]
fn gemm(a: Tensor<f32>, b: Tensor<f32>, c: TensorMut<f32>) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * BM as u32, bj * BN as u32);
    let mut acc = Tile::<f32, BM, BN>::zeros();
    let mut k = 0;
    while k < a.cols() {
        let ta = a.load_tile::<BM, BK>(r0, k);
        let tb = b.load_tile::<BK, BN>(k, c0);
        acc = mma(&ta, &tb, acc);
        k += BK as u32;
    }
    c.store_tile(r0, c0, &acc);
}
