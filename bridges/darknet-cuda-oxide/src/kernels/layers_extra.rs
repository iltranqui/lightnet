//! Softmax, reorg, dropout and scale_channels (the connected layer goes
//! through the GEMM and bias kernels instead).
//!
//! Every kernel owns its output elements exclusively (no atomics), so results
//! are deterministic.  Semantics follow `src-lib/{softmax,reorg,dropout,
//! scale_channels}_layer.*` and the `darknet-core` CPU runtime; the host
//! references live in `darknet_cuda::host_ops`.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Device twin of `darknet_cuda::dropout_unit`: must stay bit-identical.
#[device]
pub fn dropout_unit(seed: u64, index: u64) -> f32 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `out[b * to_len + i] = i < min(from_len, to_len) ? x[b * from_len + i] : 0`
    /// (flat per-sample crop / zero-pad, the Darknet mismatched shortcut).
    #[kernel]
    pub fn shortcut_fit(x: &[f32], mut out: DisjointSlice<f32>, from_len: usize, to_len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let b = g / to_len;
            let i = g - b * to_len;
            *o = if i < from_len { x[b * from_len + i] } else { 0.0 };
        }
    }

    /// One thread per softmax group of `group_size` contiguous elements
    /// (`n_groups = batch * groups`): max, exp + sum, normalise.
    #[kernel]
    pub fn softmax_forward(x: &[f32], mut out: DisjointSlice<f32>, group_size: usize, n_groups: usize) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut max = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < group_size {
            let v = x[base + i];
            if v > max {
                max = v;
            }
            i += 1;
        }
        let mut sum = 0.0f32;
        i = 0;
        while i < group_size {
            let e = (x[base + i] - max).exp();
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`;
            // the host checks `out` covers `n_groups * group_size`.
            unsafe {
                *out.as_mut_ptr().add(base + i) = e;
            }
            sum += e;
            i += 1;
        }
        if sum > 0.0 {
            i = 0;
            while i < group_size {
                // SAFETY: as above; this thread already wrote the element.
                unsafe {
                    *out.as_mut_ptr().add(base + i) /= sum;
                }
                i += 1;
            }
        }
    }

    /// `dx = y * (dy - sum(y * dy))`, one thread per group.
    #[kernel]
    pub fn softmax_backward(
        y: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut dot = 0.0f32;
        let mut i = 0usize;
        while i < group_size {
            dot += y[base + i] * dy[base + i];
            i += 1;
        }
        i = 0;
        while i < group_size {
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`.
            unsafe {
                *dx.as_mut_ptr().add(base + i) = y[base + i] * (dy[base + i] - dot);
            }
            i += 1;
        }
    }

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
        let m = if dropout_unit(seed, i as u64) < probability { 0.0 } else { scale };
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
    /// (`darknet_cuda::host_ops::dropblock_corner`), writes the plane of
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
            *dropped.as_mut_ptr().add(plane) = if cell != usize::MAX { (bs * bs) as f32 } else { 0.0 };
        }
    }

    /// DropBlock stage 2, one thread per sample: sums `dropped` over the
    /// sample's `channels` planes in order and writes
    /// `host_ops::dropblock_scale(sum, outputs)`.
    #[kernel]
    pub fn dropblock_scales(dropped: &[f32], mut scales: DisjointSlice<f32>, channels: usize, outputs: usize) {
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

    /// Darknet `constrain_ongpu`: `out = min(limit, max(-limit, x))`, NaN -> `-limit`.
    #[kernel]
    pub fn constrain(x: &[f32], mut out: DisjointSlice<f32>, limit: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            let mut v = x[i];
            if !(v > -limit) {
                v = -limit;
            }
            if v > limit {
                v = limit;
            }
            if let Some(o) = out.get_mut(idx) {
                *o = v;
            }
        }
    }

    /// Darknet `gradient_centralization_gpu`, one thread per filter of
    /// `f_size` contiguous values (sequential sum, = the host reference); a
    /// plain copy when `enabled == 0`.
    #[kernel]
    pub fn gradient_centralize(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        f_size: usize,
        filters: usize,
        enabled: u32,
    ) {
        let f = thread::index_1d().get();
        if f >= filters {
            return;
        }
        let base = f * f_size;
        let mut sum = 0.0f32;
        let mut i = 0usize;
        while i < f_size {
            sum += x[base + i];
            i += 1;
        }
        let mean = if enabled != 0 { sum / f_size as f32 } else { 0.0 };
        // SAFETY: this thread owns the `f_size` values of filter `f`.
        unsafe {
            let dst = out.as_mut_ptr().add(base);
            let mut i = 0usize;
            while i < f_size {
                *dst.add(i) = x[base + i] - mean;
                i += 1;
            }
        }
    }

    /// `out[i] = scale[si(i)] * src[i]`, `si` = `[batch, hw]` (`scale_wh != 0`)
    /// or `[batch, channels]` index of element `i`.
    #[kernel]
    pub fn scale_channels_forward(
        scale: &[f32],
        src: &[f32],
        mut out: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 { b * hw + r % hw } else { b * channels + r / hw };
        if let Some(o) = out.get_mut(idx) {
            *o = scale[si] * src[i];
        }
    }

    /// `d_source[i] = delta[i] * scale[si(i)]`.
    #[kernel]
    pub fn scale_channels_dsource(
        delta: &[f32],
        scale: &[f32],
        mut d_src: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let per_batch = channels * hw;
        let b = i / per_batch;
        let r = i % per_batch;
        let si = if scale_wh != 0 { b * hw + r % hw } else { b * channels + r / hw };
        if let Some(o) = d_src.get_mut(idx) {
            *o = delta[i] * scale[si];
        }
    }

    /// `d_scale[j] = sum delta * src` over the elements sharing scale `j`;
    /// one thread per scale element, fixed summation order.
    #[kernel]
    pub fn scale_channels_dscale(
        delta: &[f32],
        src: &[f32],
        mut d_scale: DisjointSlice<f32>,
        channels: usize,
        hw: usize,
        scale_wh: u32,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        if j >= d_scale.len() {
            return;
        }
        let mut sum = 0.0f32;
        if scale_wh != 0 {
            let b = j / hw;
            let s = j % hw;
            let mut c = 0usize;
            while c < channels {
                let i = (b * channels + c) * hw + s;
                sum += delta[i] * src[i];
                c += 1;
            }
        } else {
            let base = j * hw;
            let mut s = 0usize;
            while s < hw {
                sum += delta[base + s] * src[base + s];
                s += 1;
            }
        }
        if let Some(o) = d_scale.get_mut(idx) {
            *o = sum;
        }
    }
}
