//! Dropout and dropblock kernels.

use super::layers_extra::{dropout_unit};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `mask[i] = 0` when `dropout_unit(seed, i) < probability` else `scale`;
    /// `out[i] = x[i] * mask[i]`.
    #[kernel]
    pub fn dropout_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        probability: f32,
        scale: f32,
        seed: u64,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = if dropout_unit(seed, i as u64) < probability {
            0.0
        } else {
            scale
        };
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }

    /// `dx[i] = dy[i] * mask[i]`.
    #[kernel]
    pub fn dropout_backward(dy: &[f32], mask: &[f32], mut dx: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = dx.get_mut(idx) {
            *o = dy[i] * mask[i];
        }
    }

    /// DropBlock stage 1, one thread per `(sample, channel)` plane: picks the
    /// cell with the lowest quantised random value below `prob`
    /// (`lightnet_cuda::host_ops::dropblock_corner`), writes the plane of
    /// `keep` (`1`, `0` inside the block) and `dropped[plane]` (`bs^2` or 0).
    #[kernel]
    pub fn dropblock_plane(
        mut keep: DisjointSlice<f32>,
        mut dropped: DisjointSlice<f32>,
        seed: u64,
        w: usize,
        h: usize,
        prob: f32,
        bs: usize,
    ) {
        let idx = thread::index_1d();
        let plane = idx.get();
        let spatial = w * h;
        if plane >= dropped.len() {
            return;
        }
        let mut best = 1_000_000i32;
        let mut cell = usize::MAX;
        let mut i = 0usize;
        while i < spatial {
            let u = dropout_unit(seed, (plane * spatial + i) as u64);
            if u < prob {
                let q = (u * 1_000_000.0) as i32;
                if q < best {
                    best = q;
                    cell = i;
                }
            }
            i += 1;
        }
        let mut bx = 0usize;
        let mut by = 0usize;
        if cell != usize::MAX {
            bx = cell % w;
            by = cell / w;
            if bx > w - bs {
                bx -= w - bs;
            }
            if by > h - bs {
                by -= h - bs;
            }
            if bx > w - bs {
                bx = w - bs;
            }
            if by > h - bs {
                by = h - bs;
            }
        }
        // SAFETY: this thread owns plane `plane` of `keep` (`spatial` cells)
        // and slot `plane` of `dropped`; the host allocates both exactly.
        unsafe {
            let base = keep.as_mut_ptr().add(plane * spatial);
            let mut j = 0usize;
            while j < spatial {
                let y = j / w;
                let x = j % w;
                let inside = cell != usize::MAX && x >= bx && x < bx + bs && y >= by && y < by + bs;
                *base.add(j) = if inside { 0.0 } else { 1.0 };
                j += 1;
            }
            *dropped.as_mut_ptr().add(plane) = if cell != usize::MAX {
                (bs * bs) as f32
            } else {
                0.0
            };
        }
    }

    /// DropBlock stage 2, one thread per sample: sums `dropped` over the
    /// sample's `channels` planes in order and writes
    /// `host_ops::dropblock_scale(sum, outputs)`.
    #[kernel]
    pub fn dropblock_scales(
        dropped: &[f32],
        mut scales: DisjointSlice<f32>,
        channels: usize,
        outputs: usize,
    ) {
        let idx = thread::index_1d();
        let b = idx.get();
        let mut sum = 0.0f32;
        let mut c = 0usize;
        while c < channels && b * channels + c < dropped.len() {
            sum += dropped[b * channels + c];
            c += 1;
        }
        let keep = 1.0 - sum / outputs as f32;
        if let Some(o) = scales.get_mut(idx) {
            *o = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        }
    }

    /// DropBlock stage 3, one thread per element: `mask = keep * scale[b]`,
    /// `out = x * mask`; `per_batch = channels * h * w`.
    #[kernel]
    pub fn dropblock_apply(
        x: &[f32],
        keep: &[f32],
        scales: &[f32],
        mut out: DisjointSlice<f32>,
        mut mask: DisjointSlice<f32>,
        per_batch: usize,
    ) {
        let i = thread::index_1d().get();
        if i >= out.len() {
            return;
        }
        let m = keep[i] * scales[i / per_batch];
        // SAFETY: each thread writes its own slot `i` of both buffers; the
        // host allocates `out` and `mask` with the same length as `x`.
        unsafe {
            *out.as_mut_ptr().add(i) = x[i] * m;
            *mask.as_mut_ptr().add(i) = m;
        }
    }
}
