//! Channel permutation kernel for channel_shuffle.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[b, j*rows + i, s] = x[b, i*cols + j, s]`: one thread per output element.
    #[kernel]
    pub fn channel_permute(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        len: usize,
    ) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let per = rows * cols * len;
            let b = g / per;
            let r = g % per;
            let p = r / len; // output channel j*rows + i
            let s = r % len;
            let j = p / rows;
            let i = p % rows;
            *o = x[b * per + (i * cols + j) * len + s];
        }
    }
}
