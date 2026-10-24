//! Reorg kernels (space-to-depth and back).

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Big `[b, cb, hb, wb]` -> small `[b, cb*s*s, hs, ws]`: one thread per
    /// small element gathers its big source.
    #[kernel]
    pub fn reorg_to_small(
        big: &[f32],
        mut small: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= small.len() {
            return;
        }
        let cs = cb * stride * stride;
        let plane = hs * ws;
        let bk = i / plane;
        let r = i % plane;
        let b = bk / cs;
        let k = bk % cs;
        let c2 = k % cb;
        let off = k / cb;
        let h2 = (r / ws) * stride + off / stride;
        let w2 = (r % ws) * stride + off % stride;
        if let Some(o) = small.get_mut(idx) {
            *o = big[((b * cb + c2) * hb + h2) * wb + w2];
        }
    }

    /// Small -> big: one thread per big element reads its small source, or
    /// writes 0 on the border not covered when `hb % stride != 0`.
    #[kernel]
    pub fn reorg_to_big(
        small: &[f32],
        mut big: DisjointSlice<f32>,
        cb: usize,
        hb: usize,
        wb: usize,
        hs: usize,
        ws: usize,
        stride: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= big.len() {
            return;
        }
        let plane = hb * wb;
        let bc = i / plane;
        let r = i % plane;
        let b = bc / cb;
        let c2 = bc % cb;
        let h2 = r / wb;
        let w2 = r % wb;
        let j = h2 / stride;
        let x = w2 / stride;
        let value = if j >= hs || x >= ws {
            0.0
        } else {
            let off = (h2 % stride) * stride + w2 % stride;
            let k = off * cb + c2;
            small[((b * cb * stride * stride + k) * hs + j) * ws + x]
        };
        if let Some(o) = big.get_mut(idx) {
            *o = value;
        }
    }
}
