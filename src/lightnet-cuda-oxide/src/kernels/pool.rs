//! Max / local-average / global-average pooling, forward and backward.
//!
//! Window rule (shared by the C++ kernels and the `lightnet-core` reference):
//! output `(oy, ox)` covers input rows `oy*stride_y + ky - off` for
//! `ky < size` (same for columns) with `off = padding / 2`; taps outside the
//! image are skipped, not treated as zero.
//!
//! Every backward kernel is a gather (one thread per input element), so there
//! are no atomics and results are bit-for-bit deterministic.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// Flat index (inside `x`) of the first maximum of the window of output
/// `(oy, ox)`, scanning `ky` then `kx` and replacing only on strict `>`;
/// `usize::MAX` when the window has no valid tap (or only NaN / -inf).
/// `plane` is the flat offset of this (batch, channel) plane.
#[device]
pub fn window_argmax(
    x: &[f32],
    plane: usize,
    h: usize,
    w: usize,
    oy: usize,
    ox: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    off: usize,
) -> usize {
    let y0 = (oy * stride_y) as i64 - off as i64;
    let x0 = (ox * stride_x) as i64 - off as i64;
    let mut best = f32::NEG_INFINITY;
    let mut best_i = usize::MAX;
    let mut ky = 0usize;
    while ky < size {
        let iy = y0 + ky as i64;
        if iy >= 0 && iy < h as i64 {
            let mut kx = 0usize;
            while kx < size {
                let ix = x0 + kx as i64;
                if ix >= 0 && ix < w as i64 {
                    let idx = plane + iy as usize * w + ix as usize;
                    let v = x[idx];
                    if v > best {
                        best = v;
                        best_i = idx;
                    }
                }
                kx += 1;
            }
        }
        ky += 1;
    }
    best_i
}

/// Number of in-image taps of a 1-D window starting at `o * stride - off`.
#[device]
pub fn window_count(o: usize, stride: usize, size: usize, off: usize, limit: usize) -> usize {
    let start = (o * stride) as i64 - off as i64;
    let lo = if start > 0 { start } else { 0 };
    let end = start + size as i64;
    let hi = if end < limit as i64 {
        end
    } else {
        limit as i64
    };
    if hi > lo { (hi - lo) as usize } else { 0 }
}

/// Smallest output index `o >= 0` whose window reaches input coordinate `i`,
/// i.e. `o * stride + size - 1 - off >= i`.
#[device]
pub fn first_window(i: usize, stride: usize, size: usize, off: usize) -> i64 {
    let need = i as i64 + off as i64 - (size as i64 - 1);
    if need <= 0 {
        0
    } else {
        (need + stride as i64 - 1) / stride as i64
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Depth pooling: one thread per output element, strict `>` preserves the
    /// first channel in each strided channel group on ties.
    #[kernel]
    pub fn maxpool_depth_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        input_channels: usize,
        output_channels: usize,
        spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let group = i / spatial % output_channels;
        let batch = i / (output_channels * spatial);
        let position = i % spatial;
        let first = (batch * input_channels + group) * spatial + position;
        let mut best = x[first];
        let mut channel = group + output_channels;
        while channel < input_channels {
            let candidate = x[(batch * input_channels + channel) * spatial + position];
            if candidate > best {
                best = candidate;
            }
            channel += output_channels;
        }
        if let Some(value) = out.get_mut(idx) {
            *value = best;
        }
    }

    /// One thread per input element; each channel belongs to one output group.
    #[kernel]
    pub fn maxpool_depth_backward(
        x: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        input_channels: usize,
        output_channels: usize,
        spatial: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        let channel = i / spatial % input_channels;
        let batch = i / (input_channels * spatial);
        let position = i % spatial;
        let group = channel % output_channels;
        let first = (batch * input_channels + group) * spatial + position;
        let mut winner = group;
        let mut best = x[first];
        let mut candidate_channel = group + output_channels;
        while candidate_channel < input_channels {
            let candidate = x[(batch * input_channels + candidate_channel) * spatial + position];
            if candidate > best {
                best = candidate;
                winner = candidate_channel;
            }
            candidate_channel += output_channels;
        }
        if let Some(value) = dx.get_mut(idx) {
            *value = if channel == winner {
                dy[(batch * output_channels + group) * spatial + position]
            } else {
                0.0
            };
        }
    }

    /// One thread per output element.
    #[kernel]
    pub fn maxpool_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        off: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let bc = i / oplane;
        let r = i % oplane;
        let best = window_argmax(
            x,
            bc * h * w,
            h,
            w,
            r / out_w,
            r % out_w,
            size,
            stride_x,
            stride_y,
            off,
        );
        if let Some(o) = out.get_mut(idx) {
            *o = if best == usize::MAX {
                f32::NEG_INFINITY
            } else {
                x[best]
            };
        }
    }

    /// One thread per input element: sums `dy` of every output window whose
    /// recomputed argmax is this element.  Ties resolve exactly as in the
    /// forward pass (first tap wins), so the gradient goes to one element.
    #[kernel]
    pub fn maxpool_backward(
        x: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        off: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        let plane = h * w;
        let bc = i / plane;
        let r = i % plane;
        let iy = r / w;
        let ix = r % w;
        let oy_lo = first_window(iy, stride_y, size, off);
        let ox_lo = first_window(ix, stride_x, size, off);
        let mut oy_hi = (iy + off) as i64 / stride_y as i64;
        if oy_hi > out_h as i64 - 1 {
            oy_hi = out_h as i64 - 1;
        }
        let mut ox_hi = (ix + off) as i64 / stride_x as i64;
        if ox_hi > out_w as i64 - 1 {
            ox_hi = out_w as i64 - 1;
        }
        let mut sum = 0.0f32;
        let mut oy = oy_lo;
        while oy <= oy_hi {
            let mut ox = ox_lo;
            while ox <= ox_hi {
                let best = window_argmax(
                    x,
                    bc * plane,
                    h,
                    w,
                    oy as usize,
                    ox as usize,
                    size,
                    stride_x,
                    stride_y,
                    off,
                );
                if best == i {
                    sum += dy[(bc * out_h + oy as usize) * out_w + ox as usize];
                }
                ox += 1;
            }
            oy += 1;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = sum;
        }
    }

    /// Mean over the in-image taps of each window; one thread per output.
    #[kernel]
    pub fn local_avgpool_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        off: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let bc = i / oplane;
        let r = i % oplane;
        let oy = r / out_w;
        let ox = r % out_w;
        let y0 = (oy * stride_y) as i64 - off as i64;
        let x0 = (ox * stride_x) as i64 - off as i64;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut ky = 0usize;
        while ky < size {
            let iy = y0 + ky as i64;
            if iy >= 0 && iy < h as i64 {
                let mut kx = 0usize;
                while kx < size {
                    let ix = x0 + kx as i64;
                    if ix >= 0 && ix < w as i64 {
                        sum += x[(bc * h + iy as usize) * w + ix as usize];
                        count += 1;
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    /// Gather: each output window spreads `dy / count` over its in-image taps.
    #[kernel]
    pub fn local_avgpool_backward(
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        h: usize,
        w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        off: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        let plane = h * w;
        let bc = i / plane;
        let r = i % plane;
        let iy = r / w;
        let ix = r % w;
        let oy_lo = first_window(iy, stride_y, size, off);
        let ox_lo = first_window(ix, stride_x, size, off);
        let mut oy_hi = (iy + off) as i64 / stride_y as i64;
        if oy_hi > out_h as i64 - 1 {
            oy_hi = out_h as i64 - 1;
        }
        let mut ox_hi = (ix + off) as i64 / stride_x as i64;
        if ox_hi > out_w as i64 - 1 {
            ox_hi = out_w as i64 - 1;
        }
        let mut sum = 0.0f32;
        let mut oy = oy_lo;
        while oy <= oy_hi {
            let cy = window_count(oy as usize, stride_y, size, off, h);
            let mut ox = ox_lo;
            while ox <= ox_hi {
                let cx = window_count(ox as usize, stride_x, size, off, w);
                let count = cy * cx;
                if count > 0 {
                    sum += dy[(bc * out_h + oy as usize) * out_w + ox as usize] / count as f32;
                }
                ox += 1;
            }
            oy += 1;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = sum;
        }
    }

    /// Global average: one thread per (batch, channel) plane, sequential sum
    /// in memory order like the C++ kernel.
    #[kernel]
    pub fn avgpool_forward(x: &[f32], mut out: DisjointSlice<f32>, area: usize) {
        let idx = thread::index_1d();
        let bc = idx.get();
        if bc >= out.len() {
            return;
        }
        let mut sum = 0.0f32;
        let mut k = 0usize;
        while k < area {
            sum += x[bc * area + k];
            k += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = sum / area as f32;
        }
    }

    /// `dx[bc, k] = dy[bc] / area`; one thread per input element.
    #[kernel]
    pub fn avgpool_backward(dy: &[f32], mut dx: DisjointSlice<f32>, area: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dx.len() {
            return;
        }
        if let Some(o) = dx.get_mut(idx) {
            *o = dy[i / area] / area as f32;
        }
    }
}
