//! `y = a * x + y`, one element per thread.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

#[kernel]
fn saxpy(a: f32, x: Tensor<f32>, y: TensorMut<f32>) {
    let i = block_id().1 * BLOCK_THREADS + thread_id();
    if i < x.cols() {
        let (xp, yp) = (x.ptr(), y.ptr());
        *yp.add(i as usize) = a * *xp.add(i as usize) + *yp.add(i as usize);
    }
}
