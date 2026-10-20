//! f16-storage inference ops (P6): im2col with padded GEMM layout, maxpool,
//! route and nearest upsample on raw `u16` half bit patterns.  Compare /
//! arithmetic happens in f32 after an exact widening, results are rounded back
//! to nearest even (data movement ops are bit copies).

use super::activations::activate;
use super::convert::{f16_bits_to_f32, f32_to_f16_bits};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// im2col of one batch element into the GEMM operand layout used by the
    /// f16 conv: group `g` owns a `kpad x npad` block
    /// `col[(g * kpad + rr) * npad + p]` where `rr < kg` is the row inside the
    /// group (`r = g * kg + rr = c * k * k + ky * k + kx`) and `p < oh * ow`;
    /// the K / N padding is written as zero.
    #[kernel]
    pub fn im2col_f16(
        x: &[u16],
        mut col: DisjointSlice<u16>,
        x_base: usize,
        h: usize,
        w: usize,
        ksize: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        out_h: usize,
        out_w: usize,
        kg: usize,
        kpad: usize,
        npad: usize,
        groups: usize,
        images: usize,
        img_stride: usize,
    ) {
        let idx = thread::index_1d();
        let gi = idx.get();
        let blk = kpad * npad;
        if gi >= images * groups * blk {
            return;
        }
        // Column buffer layout `[image][group][kpad][npad]`; image `m` reads
        // `x` from `x_base + m * img_stride`.
        let img = gi / (groups * blk);
        let i = gi % (groups * blk);
        let x_base = x_base + img * img_stride;
        let grp = i / blk;
        let rem = i % blk;
        let rr = rem / npad;
        let p = rem % npad;
        let ohw = out_h * out_w;
        let mut v = 0u16;
        if rr < kg && p < ohw {
            let r = grp * kg + rr;
            let kk = ksize * ksize;
            let c = r / kk;
            let kr = r % kk;
            let ky = kr / ksize;
            let kx = kr % ksize;
            let oy = p / out_w;
            let ox = p % out_w;
            let iy = (oy * stride_y + ky * dilation) as i64 - padding as i64;
            let ix = (ox * stride_x + kx * dilation) as i64 - padding as i64;
            if iy >= 0 && iy < h as i64 && ix >= 0 && ix < w as i64 {
                v = x[x_base + (c * h + iy as usize) * w + ix as usize];
            }
        }
        if let Some(o) = col.get_mut(idx) {
            *o = v;
        }
    }

    /// Same window rule as the f32 `maxpool_forward` (taps outside the image
    /// skipped, strict `>`, empty window -> -inf).
    #[kernel]
    pub fn maxpool_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
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
        let plane = (i / oplane) * h * w;
        let r = i % oplane;
        let y0 = ((r / out_w) * stride_y) as i64 - off as i64;
        let x0 = ((r % out_w) * stride_x) as i64 - off as i64;
        let mut best = f32::NEG_INFINITY;
        let mut best_bits = 0xFC00u16;
        let mut ky = 0usize;
        while ky < size {
            let iy = y0 + ky as i64;
            if iy >= 0 && iy < h as i64 {
                let mut kx = 0usize;
                while kx < size {
                    let ix = x0 + kx as i64;
                    if ix >= 0 && ix < w as i64 {
                        let bits = x[plane + iy as usize * w + ix as usize];
                        let v = f16_bits_to_f32(bits);
                        if v > best {
                            best = v;
                            best_bits = bits;
                        }
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = best_bits;
        }
    }

    /// One thread per output element; first channel wins on ties, matching
    /// f32 `maxpool_depth_forward` and the CPU reference.
    #[kernel]
    pub fn maxpool_depth_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
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
        let mut best_bits = x[first];
        let mut best = f16_bits_to_f32(best_bits);
        let mut channel = group + output_channels;
        while channel < input_channels {
            let bits = x[(batch * input_channels + channel) * spatial + position];
            let candidate = f16_bits_to_f32(bits);
            if candidate > best {
                best = candidate;
                best_bits = bits;
            }
            channel += output_channels;
        }
        if let Some(value) = out.get_mut(idx) {
            *value = best_bits;
        }
    }

    /// Nearest upsample; `scale != 1` multiplies in f32 and rounds back.
    #[kernel]
    pub fn upsample_nearest_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let oplane = out_h * out_w;
        let bc = i / oplane;
        let r = i % oplane;
        let iy = (r / out_w) / stride;
        let ix = (r % out_w) / stride;
        let bits = x[bc * in_h * in_w + iy * in_w + ix];
        if let Some(o) = out.get_mut(idx) {
            *o = if scale == 1.0 { bits } else { f32_to_f16_bits(scale * f16_bits_to_f32(bits), 0) };
        }
    }

    /// Region forward on half-storage values: channel-major NCHW to spatial-
    /// major entry rows, with f32 sigmoid/optional softmax and half stores.
    #[kernel]
    pub fn region_forward_f16(
        input: &[u16],
        mut output: DisjointSlice<u16>,
        spatial: usize,
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: u32,
    ) {
        let group = thread::index_1d().get();
        if group >= output.len() / entries {
            return;
        }
        let groups_per_batch = spatial * anchors;
        let batch = group / groups_per_batch;
        let rem = group % groups_per_batch;
        let location = rem / anchors;
        let anchor = rem % anchors;
        let channels = anchors * entries;
        let source_index = |entry: usize| {
            (batch * channels + anchor * entries + entry) * spatial + location
        };
        let destination = (batch * spatial + location) * channels + anchor * entries;
        let mut entry = 0usize;
        while entry < entries {
            let bits = input[source_index(entry)];
            // SAFETY: each group exclusively owns its entries-wide output row.
            unsafe { *output.as_mut_ptr().add(destination + entry) = bits; }
            entry += 1;
        }
        let objectness = f16_bits_to_f32(input[source_index(4)]);
        let sigmoid = 1.0 / (1.0 + (-objectness).exp());
        unsafe { *output.as_mut_ptr().add(destination + 4) = f32_to_f16_bits(sigmoid, 0); }
        if softmax_classes != 0 {
            let mut max_value = f32::NEG_INFINITY;
            entry = 0;
            while entry < classes {
                let value = f16_bits_to_f32(input[source_index(5 + entry)]);
                if value > max_value { max_value = value; }
                entry += 1;
            }
            let mut sum = 0.0f32;
            entry = 0;
            while entry < classes {
                let value = (f16_bits_to_f32(input[source_index(5 + entry)]) - max_value).exp();
                unsafe { *output.as_mut_ptr().add(destination + 5 + entry) = f32_to_f16_bits(value, 0); }
                sum += value;
                entry += 1;
            }
            if sum > 0.0 {
                entry = 0;
                while entry < classes {
                    let index = destination + 5 + entry;
                    let value = f16_bits_to_f32(unsafe { *output.as_mut_ptr().add(index) }) / sum;
                    unsafe { *output.as_mut_ptr().add(index) = f32_to_f16_bits(value, 0); }
                    entry += 1;
                }
            }
        }
    }

    /// Stable softmax over each contiguous group; one thread owns one group,
    /// reductions use f32 and each final probability rounds once to half.
    #[kernel]
    pub fn softmax_forward_f16(
        input: &[u16],
        mut output: DisjointSlice<u16>,
        group_size: usize,
        n_groups: usize,
    ) {
        let group = thread::index_1d().get();
        if group >= n_groups {
            return;
        }
        let base = group * group_size;
        let mut max_value = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < group_size {
            let value = f16_bits_to_f32(input[base + i]);
            if value > max_value { max_value = value; }
            i += 1;
        }
        let mut sum = 0.0f32;
        i = 0;
        while i < group_size {
            sum += (f16_bits_to_f32(input[base + i]) - max_value).exp();
            i += 1;
        }
        i = 0;
        while i < group_size {
            let exponential = (f16_bits_to_f32(input[base + i]) - max_value).exp();
            let probability = if sum > 0.0 { exponential / sum } else { exponential };
            unsafe {
                *output.as_mut_ptr().add(base + i) = f32_to_f16_bits(probability, 0);
            }
            i += 1;
        }
    }

    /// Big `[b,cb,hb,wb]` to small `[b,cb*s*s,hs,ws]` reorg.
    #[kernel]
    pub fn reorg_to_small_f16(
        big: &[u16],
        mut small: DisjointSlice<u16>,
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

    /// Small to big reorg, zero-filling uncovered bottom/right borders.
    #[kernel]
    pub fn reorg_to_big_f16(
        small: &[u16],
        mut big: DisjointSlice<u16>,
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
            0u16
        } else {
            let off = (h2 % stride) * stride + w2 % stride;
            let k = off * cb + c2;
            small[((b * cb * stride * stride + k) * hs + j) * ws + x]
        };
        if let Some(o) = big.get_mut(idx) {
            *o = value;
        }
    }

    /// Same strided channel-slice copy as the f32 `route_copy`.
    #[kernel]
    pub fn route_copy_f16(
        src: &[u16],
        mut dst: DisjointSlice<u16>,
        batch: usize,
        channels: usize,
        src_total: usize,
        src_offset: usize,
        dst_total: usize,
        dst_offset: usize,
        stride: usize,
    ) {
        let i = thread::index_1d().get();
        let per_batch = channels * stride;
        if i >= batch * per_batch {
            return;
        }
        let b = i / per_batch;
        let r = i % per_batch;
        let c = r / stride;
        let s = r % stride;
        let from = (b * src_total + src_offset + c) * stride + s;
        let to = (b * dst_total + dst_offset + c) * stride + s;
        // SAFETY: each thread writes a distinct `to`; the host checks that
        // `dst` covers `batch * dst_total * stride` elements.
        unsafe {
            *dst.as_mut_ptr().add(to) = src[from];
        }
    }

    /// `out = f16(act(a + b))` (shortcut with a same-shaped source; the sum is
    /// formed in f32 and rounded once).
    #[kernel]
    pub fn shortcut_add_act_f16(a: &[u16], b: &[u16], mut out: DisjointSlice<u16>, kind: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_f16_bits(activate(kind, f16_bits_to_f32(a[i]) + f16_bits_to_f32(b[i])), 0);
        }
    }

    /// `out = f16(act(input * source))` for same-shaped SAM tensors.
    #[kernel]
    pub fn sam_f16(input: &[u16], source: &[u16], mut out: DisjointSlice<u16>, kind: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(value) = out.get_mut(idx) {
            let product = f16_bits_to_f32(input[i]) * f16_bits_to_f32(source[i]);
            *value = f32_to_f16_bits(activate(kind, product), 0);
        }
    }

    /// NCHW channel shuffle: out channel `j*groups+i` reads input `i*cols+j`.
    #[kernel]
    pub fn channel_shuffle_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
        channels: usize,
        groups: usize,
        spatial: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        if linear >= out.len() {
            return;
        }
        let plane = channels * spatial;
        let batch = linear / plane;
        let output_channel = linear / spatial % channels;
        let position = linear % spatial;
        let rows = groups;
        let cols = channels / groups;
        let row = output_channel % rows;
        let col = output_channel / rows;
        let input_channel = row * cols + col;
        let value = x[batch * plane + input_channel * spatial + position];
        if let Some(output) = out.get_mut(idx) {
            *output = value;
        }
    }

    /// `out = f16(act(x * scale[c] + bias[c]))`, `c = (i / stride) % channels`
    /// (standalone bias + activation and the folded inference batch norm).
    /// `has_scale == 0` ignores `scale` (any buffer may be passed).
    #[kernel]
    pub fn channel_affine_act_f16(
        x: &[u16],
        scale: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<u16>,
        channels: usize,
        stride: usize,
        has_scale: u32,
        kind: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let c = (i / stride) % channels;
            let mut v = f16_bits_to_f32(x[i]);
            if has_scale != 0 {
                v *= scale[c];
            }
            *o = f32_to_f16_bits(activate(kind, v + bias[c]), 0);
        }
    }

    /// Mean over the in-image taps of each window (f32 sum, one rounding).
    #[kernel]
    pub fn local_avgpool_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
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
        let y0 = ((r / out_w) * stride_y) as i64 - off as i64;
        let x0 = ((r % out_w) * stride_x) as i64 - off as i64;
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
                        sum += f16_bits_to_f32(x[(bc * h + iy as usize) * w + ix as usize]);
                        count += 1;
                    }
                    kx += 1;
                }
            }
            ky += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_f16_bits(if count > 0 { sum / count as f32 } else { 0.0 }, 0);
        }
    }

    /// Global average, one thread per (batch, channel) plane, f32 sequential sum.
    #[kernel]
    pub fn avgpool_f16(x: &[u16], mut out: DisjointSlice<u16>, area: usize) {
        let idx = thread::index_1d();
        let bc = idx.get();
        if bc >= out.len() {
            return;
        }
        let mut sum = 0.0f32;
        let mut k = 0usize;
        while k < area {
            sum += f16_bits_to_f32(x[bc * area + k]);
            k += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_f16_bits(sum / area as f32, 0);
        }
    }

    /// `reverse` upsample: each small output pixel sums the `stride x stride`
    /// block of the big input it covers (taps past the edge skipped).
    #[kernel]
    pub fn upsample_reverse_f16(
        x: &[u16],
        mut out: DisjointSlice<u16>,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        stride: usize,
        scale: f32,
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
        let mut sum = 0.0f32;
        let mut dy = 0usize;
        while dy < stride {
            let iy = oy * stride + dy;
            if iy < in_h {
                let mut dx = 0usize;
                while dx < stride {
                    let ix = ox * stride + dx;
                    if ix < in_w {
                        sum += f16_bits_to_f32(x[bc * in_h * in_w + iy * in_w + ix]);
                    }
                    dx += 1;
                }
            }
            dy += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_f16_bits(scale * sum, 0);
        }
    }

    /// Flat per-sample crop / zero-pad of a mismatched shortcut source (bit copy).
    #[kernel]
    pub fn shortcut_fit_f16(x: &[u16], mut out: DisjointSlice<u16>, from_len: usize, to_len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let b = g / to_len;
            let i = g - b * to_len;
            *o = if i < from_len { x[b * from_len + i] } else { 0 };
        }
    }
}
