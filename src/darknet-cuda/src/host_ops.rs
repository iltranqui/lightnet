//! Host reference implementations behind the default bodies of the layer ops
//! in [`crate::F32TensorOpsBackend`] (softmax, reorg, dropout, scale_channels,
//! connected, batch-norm training, Adam).  A backend without a device kernel for one of these ops
//! round-trips through the host and still produces the exact result a device
//! kernel must match.

/// Deterministic counter-hash RNG for dropout: a splitmix64 finaliser over
/// `seed + (index + 1) * golden`, reduced to a uniform f32 in `[0, 1)` with 24
/// exact bits.  Device kernels reproduce this bit for bit, so a mask depends
/// only on `(seed, index)`, never on launch geometry.
pub fn dropout_unit(seed: u64, index: u64) -> f32 {
    let mut z = seed.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / 16_777_216.0
}

/// Softmax over every contiguous `group_size` run of `x`.
pub fn softmax(x: &[f32], group_size: usize) -> Vec<f32> {
    assert!(group_size > 0 && x.len() % group_size == 0, "softmax: bad group size");
    let mut out = vec![0.0f32; x.len()];
    for (src, dst) in x.chunks(group_size).zip(out.chunks_mut(group_size)) {
        let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (d, v) in dst.iter_mut().zip(src) {
            *d = (*v - max).exp();
            sum += *d;
        }
        if sum > 0.0 {
            for d in dst.iter_mut() {
                *d /= sum;
            }
        }
    }
    out
}

/// Darknet's mismatched-shape shortcut layout (`shortcut_multilayer_*`): per
/// batch sample the flat vector of `x.len() / batch` elements is cropped or
/// zero-padded to `to_len` elements (`src_i < min(from, to)` is copied).
/// Adding the result to the layer input is the forward shortcut; applying it
/// to the layer gradient gives the gradient of the (cropped/padded) source.
pub fn shortcut_fit(x: &[f32], batch: usize, to_len: usize) -> Vec<f32> {
    assert!(batch > 0 && x.len() % batch == 0, "shortcut_fit: bad batch");
    let from_len = x.len() / batch;
    let copied = from_len.min(to_len);
    let mut out = vec![0.0f32; batch * to_len];
    for b in 0..batch {
        out[b * to_len..b * to_len + copied].copy_from_slice(&x[b * from_len..b * from_len + copied]);
    }
    out
}

/// Softmax Jacobian-vector product: `dx = y * (dy - sum(y * dy))` per group.
///
/// Reference for [`crate::F32TensorOpsBackend::softmax_backward_f32`]; `y` is the softmax
/// **output**, so the caller passes the forward's result rather than its logits.
pub fn softmax_backward(y: &[f32], dy: &[f32], group_size: usize) -> Vec<f32> {
    assert!(
        group_size > 0 && y.len() == dy.len() && y.len() % group_size == 0,
        "softmax backward: bad group size"
    );
    let mut dx = vec![0.0f32; y.len()];
    for ((yg, dyg), dxg) in y
        .chunks(group_size)
        .zip(dy.chunks(group_size))
        .zip(dx.chunks_mut(group_size))
    {
        let dot: f32 = yg.iter().zip(dyg).map(|(a, b)| a * b).sum();
        for ((d, a), b) in dxg.iter_mut().zip(yg).zip(dyg) {
            *d = a * (b - dot);
        }
    }
    dx
}

/// Geometry of the `[batch, cb, hb, wb]` <-> `[batch, cb*s*s, hs, ws]` reorg
/// bijection (`hs = hb / s`, `ws = wb / s`, extra border rows/cols unused).
#[derive(Clone, Copy, Debug)]
pub struct ReorgDims {
    pub batch: usize,
    pub cb: usize,
    pub hb: usize,
    pub wb: usize,
    pub hs: usize,
    pub ws: usize,
    pub stride: usize,
}

impl ReorgDims {
    pub fn big_len(&self) -> usize {
        self.batch * self.cb * self.hb * self.wb
    }
    pub fn small_len(&self) -> usize {
        self.batch * self.cb * self.stride * self.stride * self.hs * self.ws
    }
}

/// Validates a reorg layer's `[batch, c, h, w]` input/output shapes and returns
/// the shared bijection geometry.  Non-reverse: input is the big tensor, output
/// the small one; reverse swaps the roles.
pub fn reorg_dims(
    input_shape: &[usize],
    output_shape: &[usize],
    stride: usize,
    reverse: bool,
) -> Result<ReorgDims, String> {
    if stride == 0 || input_shape.len() != 4 || output_shape.len() != 4 {
        return Err(format!(
            "reorg: bad stride {stride} or shapes {input_shape:?} -> {output_shape:?}"
        ));
    }
    let (big, small) = if reverse { (output_shape, input_shape) } else { (input_shape, output_shape) };
    let d = ReorgDims {
        batch: big[0],
        cb: big[1],
        hb: big[2],
        wb: big[3],
        hs: small[2],
        ws: small[3],
        stride,
    };
    if small[0] != d.batch
        || d.cb == 0
        || small[1] != d.cb * stride * stride
        || d.hs != d.hb / stride
        || d.ws != d.wb / stride
        || d.hs == 0
        || d.ws == 0
    {
        return Err(format!(
            "reorg: shapes {input_shape:?} -> {output_shape:?} do not match stride {stride} (reverse={reverse})"
        ));
    }
    Ok(d)
}

/// Index into the big tensor feeding small element `(b, k, j, i)`.
fn reorg_big_index(d: &ReorgDims, b: usize, k: usize, j: usize, i: usize) -> usize {
    let c2 = k % d.cb;
    let off = k / d.cb;
    let h2 = j * d.stride + off / d.stride;
    let w2 = i * d.stride + off % d.stride;
    ((b * d.cb + c2) * d.hb + h2) * d.wb + w2
}

/// Big -> small (non-reverse reorg forward, reverse reorg backward).
pub fn reorg_to_small(big: &[f32], d: &ReorgDims) -> Vec<f32> {
    assert_eq!(big.len(), d.big_len(), "reorg: big tensor length");
    let cs = d.cb * d.stride * d.stride;
    let mut small = vec![0.0f32; d.small_len()];
    for b in 0..d.batch {
        for k in 0..cs {
            for j in 0..d.hs {
                for i in 0..d.ws {
                    small[((b * cs + k) * d.hs + j) * d.ws + i] = big[reorg_big_index(d, b, k, j, i)];
                }
            }
        }
    }
    small
}

/// Small -> big (reverse reorg forward, non-reverse reorg backward).  Border
/// elements outside the `hs*s x ws*s` covered area are zero.
pub fn reorg_to_big(small: &[f32], d: &ReorgDims) -> Vec<f32> {
    assert_eq!(small.len(), d.small_len(), "reorg: small tensor length");
    let cs = d.cb * d.stride * d.stride;
    let mut big = vec![0.0f32; d.big_len()];
    for b in 0..d.batch {
        for k in 0..cs {
            for j in 0..d.hs {
                for i in 0..d.ws {
                    big[reorg_big_index(d, b, k, j, i)] = small[((b * cs + k) * d.hs + j) * d.ws + i];
                }
            }
        }
    }
    big
}

/// Element-wise `x * mask` plus the mask itself (`0` below `probability`,
/// `1 / (1 - probability)` otherwise).
pub fn dropout(x: &[f32], probability: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    assert!((0.0..1.0).contains(&probability), "dropout: probability must be in [0, 1)");
    let scale = 1.0 / (1.0 - probability);
    let mask: Vec<f32> = (0..x.len())
        .map(|i| if dropout_unit(seed, i as u64) < probability { 0.0 } else { scale })
        .collect();
    let out = x.iter().zip(&mask).map(|(v, m)| v * m).collect();
    (out, mask)
}

/// DropBlock cell (top-left corner) chosen for one `(sample, channel)` plane,
/// or `None` when no cell of the plane is below `block_prob`.  C++
/// `dropblock_fast_kernel`: the cell with the lowest random value (quantised
/// to 1e-6 like the C++ `atomicMin` int; ties go to the lowest index) is moved
/// inside the image and expanded to a `block_size` square.  `plane` indexes
/// `sample * channels + channel`; the random value of cell `i` is
/// `dropout_unit(seed, plane * h * w + i)`.
pub fn dropblock_corner(
    seed: u64,
    plane: usize,
    w: usize,
    h: usize,
    block_prob: f32,
    block_size: usize,
) -> Option<(usize, usize)> {
    let spatial = w * h;
    let mut best = 1_000_000i32;
    let mut cell = None;
    for i in 0..spatial {
        let u = dropout_unit(seed, (plane * spatial + i) as u64);
        if u < block_prob {
            let q = (u * 1_000_000.0) as i32;
            if q < best {
                best = q;
                cell = Some(i);
            }
        }
    }
    cell.map(|i| {
        let (mut bx, mut by) = (i % w, i / w);
        if bx > w - block_size {
            bx -= w - block_size;
        }
        if by > h - block_size {
            by -= h - block_size;
        }
        (bx.min(w - block_size), by.min(h - block_size))
    })
}

/// Per-sample DropBlock rescale `1 / (1 - dropped / outputs)` where `dropped`
/// is `block_size^2` per plane that received a block (C++
/// `set_scales_dropblock_kernel`).  `0` when everything would be dropped.
pub fn dropblock_scale(dropped_cells: f32, outputs: usize) -> f32 {
    let keep = 1.0 - dropped_cells / outputs as f32;
    if keep > 0.0 { 1.0 / keep } else { 0.0 }
}

/// DropBlock forward over an NCHW `shape`: one `block_size` square is zeroed
/// in every `(sample, channel)` plane with probability-driven selection (see
/// [`dropblock_corner`]) and each sample is rescaled by [`dropblock_scale`].
/// Returns `(x * mask, mask)`; `mask` holds the per-element multiplier (`0`
/// inside blocks, the sample scale elsewhere), so the exact backward is
/// `dx = dy * mask` like C++ `backward_dropblock_kernel` + scale.
///
/// Deviation from C++: the C++ kernel indexes the random values with
/// `b*spatial*f + f*spatial + i` instead of `(b*filters + f)*spatial + i`
/// (the planes of later samples share random values).  This reference draws
/// an independent value per cell.
pub fn dropblock(
    x: &[f32],
    shape: &[usize],
    block_prob: f32,
    block_size: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    assert!(shape.len() == 4, "dropblock: shape must be NCHW");
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    assert!(x.len() == n * c * h * w, "dropblock: shape/data mismatch");
    assert!(
        block_size >= 1 && block_size <= w.min(h),
        "dropblock: block_size must be in 1..=min(w, h)"
    );
    let spatial = h * w;
    let mut mask = vec![1.0f32; x.len()];
    for b in 0..n {
        let mut dropped = 0.0f32;
        for ch in 0..c {
            let plane = b * c + ch;
            if let Some((bx, by)) = dropblock_corner(seed, plane, w, h, block_prob, block_size) {
                for y in by..by + block_size {
                    for xx in bx..bx + block_size {
                        mask[plane * spatial + y * w + xx] = 0.0;
                    }
                }
                dropped += (block_size * block_size) as f32;
            }
        }
        let scale = dropblock_scale(dropped, c * spatial);
        for m in &mut mask[b * c * spatial..(b + 1) * c * spatial] {
            *m *= scale;
        }
    }
    let out = x.iter().zip(&mask).map(|(v, m)| v * m).collect();
    (out, mask)
}

/// Darknet `constrain_ongpu`: `min(limit, max(-limit, v))` with C's `fmax`
/// NaN handling (NaN becomes `-limit`).
pub fn constrain(x: &[f32], limit: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let mut v = v;
            if !(v > -limit) {
                v = -limit;
            }
            if v > limit {
                v = limit;
            }
            v
        })
        .collect()
}

/// Darknet `gradient_centralization_gpu`: subtract the mean of every
/// contiguous filter of `f_size` values.  Like C++ it only runs when
/// `f_size % 32 == 0`, otherwise the input is returned unchanged.
pub fn gradient_centralization(x: &[f32], f_size: usize) -> Vec<f32> {
    assert!(f_size > 0 && x.len() % f_size == 0, "gradient_centralization: bad filter size");
    let mut out = x.to_vec();
    if f_size % 32 != 0 {
        return out;
    }
    for filter in out.chunks_mut(f_size) {
        let mut sum = 0.0f32;
        for v in filter.iter() {
            sum += *v;
        }
        let mean = sum / f_size as f32;
        for v in filter.iter_mut() {
            *v -= mean;
        }
    }
    out
}

/// Exact weighted-shortcut normalization-chain derivative from `raw` and
/// `z = dL/d(effective_weights)`, stored `[input][slot]`.
pub fn shortcut_weight_normalized_gradient(
    raw: &[f32],
    z: &[f32],
    layer_step: usize,
    normalization: crate::F32ShortcutWeightNormalization,
    cpp_grad: bool,
) -> Vec<f32> {
    assert!(layer_step > 0 && raw.len().is_multiple_of(layer_step));
    assert_eq!(raw.len(), z.len());
    let n_inputs = raw.len() / layer_step;
    let mut gradient = vec![0.0; raw.len()];
    for slot in 0..layer_step {
        let at = |input: usize| input * layer_step + slot;
        match (normalization, cpp_grad) {
            (crate::F32ShortcutWeightNormalization::None, _) | (_, true) => {
                for input in 0..n_inputs {
                    gradient[at(input)] = z[at(input)];
                }
            }
            (crate::F32ShortcutWeightNormalization::Relu, false) => {
                let mut positive_sum = 0.0f32;
                let mut dot = 0.0f32;
                for input in 0..n_inputs {
                    let positive = raw[at(input)].max(0.0);
                    positive_sum += positive;
                    dot += z[at(input)] * positive;
                }
                let sum = 0.0001f32 + positive_sum;
                for input in 0..n_inputs {
                    let raw_value = raw[at(input)];
                    gradient[at(input)] = if raw_value > 0.0 {
                        z[at(input)] / sum - dot / (sum * sum)
                    } else {
                        0.0
                    };
                }
            }
            (crate::F32ShortcutWeightNormalization::Softmax, false) => {
                let mut max = f32::NEG_INFINITY;
                for input in 0..n_inputs {
                    max = max.max(raw[at(input)]);
                }
                let mut exp_sum = 0.0f32;
                for input in 0..n_inputs {
                    exp_sum += (raw[at(input)] - max).exp();
                }
                let sum = 0.0001f32 + exp_sum;
                let mut dot = 0.0f32;
                for input in 0..n_inputs {
                    let effective = (raw[at(input)] - max).exp() / sum;
                    dot += z[at(input)] * effective;
                }
                for input in 0..n_inputs {
                    let effective = (raw[at(input)] - max).exp() / sum;
                    gradient[at(input)] = effective * (z[at(input)] - dot);
                }
            }
        }
    }
    gradient
}

/// Host reference for Darknet's channel-wise convolution activations. The
/// input/output are contiguous NCHW; normalization is independent per batch
/// and spatial location.
pub fn normalize_channels(
    input: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    normalization: crate::F32ChannelNormalization,
) -> Vec<f32> {
    let total = channel_normalization_len(batch, channels, spatial);
    assert_eq!(input.len(), total);
    let mut output = vec![0.0f32; total];
    for b in 0..batch {
        for position in 0..spatial {
            let base = b * channels * spatial + position;
            match normalization {
                crate::F32ChannelNormalization::Relu => {
                    let mut sum = 0.0001f32;
                    for channel in 0..channels {
                        let value = input[base + channel * spatial];
                        if value > 0.0 {
                            sum += value;
                        }
                    }
                    for channel in 0..channels {
                        let index = base + channel * spatial;
                        let value = input[index];
                        output[index] = if value > 0.0 { value / sum } else { 0.0 };
                    }
                }
                crate::F32ChannelNormalization::Softmax
                | crate::F32ChannelNormalization::SoftmaxMaxval => {
                    let maximum = if normalization == crate::F32ChannelNormalization::SoftmaxMaxval
                    {
                        let mut maximum = -f32::MAX;
                        for channel in 0..channels {
                            let value = input[base + channel * spatial];
                            if value > maximum || channel == 0 {
                                maximum = value;
                            }
                        }
                        maximum
                    } else {
                        0.0
                    };
                    let mut sum = 0.0001f32;
                    for channel in 0..channels {
                        sum += (input[base + channel * spatial] - maximum).exp();
                    }
                    for channel in 0..channels {
                        let index = base + channel * spatial;
                        output[index] = (input[index] - maximum).exp() / sum;
                    }
                }
            }
        }
    }
    output
}

/// Exact normalization-chain derivative, with an opt-in replica of the C++
/// CUDA kernels' legacy diagonal shortcut (`cpp_grad`).
pub fn normalize_channels_backward(
    activation_input: &[f32],
    output: &[f32],
    delta: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    normalization: crate::F32ChannelNormalization,
    cpp_grad: bool,
) -> Vec<f32> {
    let total = channel_normalization_len(batch, channels, spatial);
    assert_eq!(activation_input.len(), total);
    assert_eq!(output.len(), total);
    assert_eq!(delta.len(), total);
    let mut gradient = vec![0.0f32; total];
    for b in 0..batch {
        for position in 0..spatial {
            let base = b * channels * spatial + position;
            let mut dot = 0.0f32;
            for channel in 0..channels {
                let index = base + channel * spatial;
                dot += output[index] * delta[index];
            }
            if cpp_grad {
                for channel in 0..channels {
                    let index = base + channel * spatial;
                    let y = output[index];
                    gradient[index] = match normalization {
                        crate::F32ChannelNormalization::Relu if y > 0.0 => delta[index] * y,
                        crate::F32ChannelNormalization::Relu => delta[index],
                        crate::F32ChannelNormalization::Softmax
                        | crate::F32ChannelNormalization::SoftmaxMaxval => {
                            let value = delta[index] * y * (1.0 - y);
                            if value.is_finite() { value } else { 0.0 }
                        }
                    };
                }
            } else {
                match normalization {
                    crate::F32ChannelNormalization::Relu => {
                        let mut sum = 0.0001f32;
                        for channel in 0..channels {
                            let value = activation_input[base + channel * spatial];
                            if value > 0.0 {
                                sum += value;
                            }
                        }
                        for channel in 0..channels {
                            let index = base + channel * spatial;
                            gradient[index] = if activation_input[index] > 0.0 {
                                (delta[index] - dot) / sum
                            } else {
                                0.0
                            };
                        }
                    }
                    crate::F32ChannelNormalization::Softmax
                    | crate::F32ChannelNormalization::SoftmaxMaxval => {
                        for channel in 0..channels {
                            let index = base + channel * spatial;
                            gradient[index] = output[index] * (delta[index] - dot);
                        }
                    }
                }
            }
        }
    }
    gradient
}

/// Stable softplus used by the source-derived EML convolution branch.
fn eml_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn eml_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Host reference for the EML convolution's two-branch elementwise combine.
pub fn eml_convolutional_forward(
    x_branch: &[f32],
    self_branch: &[f32],
    residual: Option<&[f32]>,
    clamp: f32,
    eps: f32,
    scale: f32,
) -> Vec<f32> {
    let clamp = if clamp > 0.0 { clamp } else { 4.0 };
    let eps = if eps > 0.0 { eps } else { 1.0e-6 };
    assert_eq!(x_branch.len(), self_branch.len(), "EML branch lengths differ");
    if let Some(residual) = residual {
        assert_eq!(residual.len(), x_branch.len(), "EML residual length differs");
    }
    x_branch
        .iter()
        .zip(self_branch)
        .enumerate()
        .map(|(i, (&x, &y))| {
            let clamped = x.max(-clamp).min(clamp);
            let value = scale * (clamped.exp() - (eml_softplus(y) + eps).ln());
            value + residual.map_or(0.0, |values| values[i])
        })
        .collect()
}

/// Exact EML branch adjoints. The residual path, when present, is the identity
/// and therefore receives `delta` unchanged.
pub fn eml_convolutional_backward(
    x_branch: &[f32],
    self_branch: &[f32],
    delta: &[f32],
    clamp: f32,
    eps: f32,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let clamp = if clamp > 0.0 { clamp } else { 4.0 };
    let eps = if eps > 0.0 { eps } else { 1.0e-6 };
    assert_eq!(x_branch.len(), self_branch.len(), "EML branch lengths differ");
    assert_eq!(x_branch.len(), delta.len(), "EML delta length differs");
    let mut x_gradient = Vec::with_capacity(delta.len());
    let mut self_gradient = Vec::with_capacity(delta.len());
    for ((&x, &y), &dout) in x_branch.iter().zip(self_branch).zip(delta) {
        let dx = if x > -clamp && x < clamp {
            x.max(-clamp).min(clamp).exp()
        } else {
            0.0
        };
        let dy = -eml_sigmoid(y) / (eml_softplus(y) + eps);
        let scaled_delta = dout * scale;
        x_gradient.push(scaled_delta * dx);
        self_gradient.push(scaled_delta * dy);
    }
    (x_gradient, self_gradient)
}

fn channel_normalization_len(batch: usize, channels: usize, spatial: usize) -> usize {
    assert!(batch > 0 && channels > 0 && spatial > 0);
    batch
        .checked_mul(channels)
        .and_then(|elements| elements.checked_mul(spatial))
        .expect("channel-normalization tensor size overflows usize")
}

/// Index of the scale value multiplying element `i` of a `[batch, c, h*w]`
/// tensor: `[batch, h*w]` when `scale_wh`, else `[batch, c]`.
fn scale_index(i: usize, channels: usize, hw: usize, scale_wh: bool) -> usize {
    let b = i / (channels * hw);
    let r = i % (channels * hw);
    if scale_wh { b * hw + r % hw } else { b * channels + r / hw }
}

/// `out = scale (broadcast) * source`.
pub fn scale_channels(scale: &[f32], source: &[f32], channels: usize, hw: usize, scale_wh: bool) -> Vec<f32> {
    source
        .iter()
        .enumerate()
        .map(|(i, s)| scale[scale_index(i, channels, hw, scale_wh)] * s)
        .collect()
}

/// `(d_scale, d_source)` of [`scale_channels`].
pub fn scale_channels_backward(
    delta: &[f32],
    scale: &[f32],
    source: &[f32],
    channels: usize,
    hw: usize,
    scale_wh: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut d_scale = vec![0.0f32; scale.len()];
    let mut d_source = vec![0.0f32; source.len()];
    for (i, (g, s)) in delta.iter().zip(source).enumerate() {
        let si = scale_index(i, channels, hw, scale_wh);
        d_source[i] = g * scale[si];
        d_scale[si] += g * s;
    }
    (d_scale, d_source)
}

/// `out[b, o] = sum_i x[b, i] * w[o, i] + bias[o]` (weights are `[outputs][inputs]`).
pub fn connected(x: &[f32], w: &[f32], bias: &[f32], batch: usize, inputs: usize, outputs: usize) -> Vec<f32> {
    assert!(x.len() == batch * inputs && w.len() == outputs * inputs && bias.len() == outputs);
    let mut out = vec![0.0f32; batch * outputs];
    for b in 0..batch {
        for o in 0..outputs {
            let mut acc = 0.0f32;
            for i in 0..inputs {
                acc += x[b * inputs + i] * w[o * inputs + i];
            }
            out[b * outputs + o] = acc + bias[o];
        }
    }
    out
}

/// `(dx, dw, db)` of [`connected`] for upstream gradient `delta` (`[batch, outputs]`).
pub fn connected_backward(
    x: &[f32],
    w: &[f32],
    delta: &[f32],
    batch: usize,
    inputs: usize,
    outputs: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(x.len() == batch * inputs && w.len() == outputs * inputs && delta.len() == batch * outputs);
    let mut dx = vec![0.0f32; batch * inputs];
    let mut dw = vec![0.0f32; outputs * inputs];
    let mut db = vec![0.0f32; outputs];
    for b in 0..batch {
        for o in 0..outputs {
            let g = delta[b * outputs + o];
            db[o] += g;
            for i in 0..inputs {
                dw[o * inputs + i] += g * x[b * inputs + i];
                dx[b * inputs + i] += g * w[o * inputs + i];
            }
        }
    }
    (dx, dw, db)
}

/// Variance epsilon shared with the Rust runtime (`BATCH_NORM_EPSILON`).
pub const BATCH_NORM_EPSILON: f32 = 0.000_01;

/// Host result of [`batchnorm_train_forward`] (all NCHW-flat / per-channel).
#[derive(Clone, Debug)]
pub struct BatchNormTrainHost {
    pub output: Vec<f32>,
    pub x_hat: Vec<f32>,
    pub mean: Vec<f32>,
    /// Biased batch variance (`sum_sq / n`), the one used to normalise.
    pub variance: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_variance: Vec<f32>,
}

fn per_channel_sums(x: &[f32], channels: usize, spatial: usize, f: impl Fn(usize, f32) -> f32) -> Vec<f32> {
    let mut sums = vec![0.0f32; channels];
    for (i, v) in x.iter().enumerate() {
        let c = (i / spatial) % channels;
        sums[c] += f(c, *v);
    }
    sums
}

/// Batch-norm training forward over `[batch, channels, spatial]`: batch mean /
/// biased variance, `x_hat = (x - mean) / sqrt(var + eps)`, `out = x_hat *
/// scale + bias`, and the rolling EMA `new = keep * old + (1 - keep) * batch`
/// (the rolling variance takes the unbiased `sum_sq / (n - 1)`, as the CPU
/// runtime's `channel_mean_variance` and cuDNN do).
pub fn batchnorm_train_forward(
    x: &[f32],
    scales: &[f32],
    biases: &[f32],
    rolling_mean: &[f32],
    rolling_variance: &[f32],
    channels: usize,
    spatial: usize,
    keep: f32,
) -> BatchNormTrainHost {
    assert!(channels > 0 && spatial > 0 && x.len() % (channels * spatial) == 0, "batchnorm: bad layout");
    assert!(
        scales.len() == channels
            && biases.len() == channels
            && rolling_mean.len() == channels
            && rolling_variance.len() == channels,
        "batchnorm: per-channel length mismatch"
    );
    let n = (x.len() / channels) as f32;
    let mean: Vec<f32> = per_channel_sums(x, channels, spatial, |_, v| v)
        .into_iter()
        .map(|s| s / n)
        .collect();
    let sum_sq = per_channel_sums(x, channels, spatial, |c, v| (v - mean[c]) * (v - mean[c]));
    let variance: Vec<f32> = sum_sq.iter().map(|s| s / n).collect();
    let unbiased = 1.0 / (n - 1.0).max(1.0);
    let update = 1.0 - keep;
    let mut x_hat = vec![0.0f32; x.len()];
    let mut output = vec![0.0f32; x.len()];
    for (i, v) in x.iter().enumerate() {
        let c = (i / spatial) % channels;
        x_hat[i] = (v - mean[c]) / (variance[c] + BATCH_NORM_EPSILON).sqrt();
        output[i] = x_hat[i] * scales[c] + biases[c];
    }
    BatchNormTrainHost {
        output,
        x_hat,
        rolling_mean: (0..channels).map(|c| keep * rolling_mean[c] + update * mean[c]).collect(),
        rolling_variance: (0..channels)
            .map(|c| keep * rolling_variance[c] + update * (sum_sq[c] * unbiased))
            .collect(),
        mean,
        variance,
    }
}

/// `(dx, dscale, dbias)` of [`batchnorm_train_forward`]:
/// `dx = scale / sqrt(var + eps) / n * (n * dy - sum(dy) - x_hat * sum(dy * x_hat))`.
pub fn batchnorm_train_backward(
    dy: &[f32],
    x_hat: &[f32],
    variance: &[f32],
    scales: &[f32],
    channels: usize,
    spatial: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(
        channels > 0
            && spatial > 0
            && dy.len() == x_hat.len()
            && dy.len() % (channels * spatial) == 0
            && variance.len() == channels
            && scales.len() == channels,
        "batchnorm backward: bad layout"
    );
    let n = (dy.len() / channels) as f32;
    let dbias = per_channel_sums(dy, channels, spatial, |_, v| v);
    let mut dscale = vec![0.0f32; channels];
    for (i, (g, h)) in dy.iter().zip(x_hat).enumerate() {
        dscale[(i / spatial) % channels] += g * h;
    }
    let dx = dy
        .iter()
        .zip(x_hat)
        .enumerate()
        .map(|(i, (g, h))| {
            let c = (i / spatial) % channels;
            let inv_std = 1.0 / (variance[c] + BATCH_NORM_EPSILON).sqrt();
            scales[c] * inv_std / n * (n * g - dbias[c] - h * dscale[c])
        })
        .collect();
    (dx, dscale, dbias)
}

/// Hyper-parameters of [`adam_update`] (Darknet `adam_update_gpu`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub decay: f32,
    pub batch: usize,
    /// 1-based step count used for the bias corrections.
    pub step: u32,
    /// `false`: C++ semantics, `decay * batch * w` is folded into the update
    /// before the moments (L2).  `true`: AdamW, `w -= lr * decay * w` outside
    /// the moments.
    pub decoupled_decay: bool,
}

impl AdamParams {
    /// `(coupled_decay, decoupled_decay, bias_correction1, bias_correction2)`
    /// kernel scalars; the bias corrections use `powf` like the C++ kernel.
    pub fn kernel_scalars(&self) -> (f32, f32, f32, f32) {
        let (coupled, decoupled) = if self.decoupled_decay {
            (0.0, self.lr * self.decay)
        } else {
            (self.decay * self.batch.max(1) as f32, 0.0)
        };
        let t = self.step.max(1) as f32;
        (coupled, decoupled, 1.0 - self.beta1.powf(t), 1.0 - self.beta2.powf(t))
    }
}

/// Darknet Adam step (updates `u` point downhill, so `w += lr * mhat / (sqrt(vhat) + eps)`).
/// Returns `(w', m', v')`; the caller resets `u` to zero like C++.
pub fn adam_update(w: &[f32], u: &[f32], m: &[f32], v: &[f32], p: &AdamParams) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(w.len() == u.len() && w.len() == m.len() && w.len() == v.len(), "adam: length mismatch");
    let (coupled, decoupled, bc1, bc2) = p.kernel_scalars();
    let mut w_out = vec![0.0f32; w.len()];
    let mut m_out = vec![0.0f32; w.len()];
    let mut v_out = vec![0.0f32; w.len()];
    for i in 0..w.len() {
        let g = u[i] - coupled * w[i];
        m_out[i] = p.beta1 * m[i] + (1.0 - p.beta1) * g;
        v_out[i] = p.beta2 * v[i] + (1.0 - p.beta2) * (g * g);
        let (mhat, vhat) = (m_out[i] / bc1, v_out[i] / bc2);
        w_out[i] = w[i] + p.lr * mhat / (vhat.sqrt() + p.eps) - decoupled * w[i];
    }
    (w_out, m_out, v_out)
}

/// Darknet's two device sanitizers (`blas_kernels.cu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanInfMode {
    /// `reset_nan_and_inf`: non-finite -> 0 (used on accumulated updates).
    Reset,
    /// `fix_nan_and_inf`: non-finite -> `1 / (|index| + 1)` (used on weights).
    Fix,
}

/// Host reference of Darknet `reset_nan_and_inf` / `fix_nan_and_inf`.
pub fn sanitize_nan_inf(x: &[f32], mode: NanInfMode) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            if v.is_finite() {
                v
            } else {
                match mode {
                    NanInfMode::Reset => 0.0,
                    NanInfMode::Fix => 1.0 / (i as f32 + 1.0),
                }
            }
        })
        .collect()
}

/// Channel permutation of `[batch][rows*cols][len]`: output channel `j*rows + i` is input
/// channel `i*cols + j` (C++ `channel_shuffle_op`). With `rows = groups, cols = c/groups` this is
/// `[channel_shuffle]`; swapping `rows`/`cols` gives the exact inverse (the backward pass).
pub fn channel_permute(x: &[f32], batch: usize, rows: usize, cols: usize, len: usize) -> Vec<f32> {
    let per = rows * cols * len;
    assert!(per > 0 && x.len() == batch * per, "channel_permute: bad shape");
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                let src = b * per + (i * cols + j) * len;
                let dst = b * per + (j * rows + i) * len;
                out[dst..dst + len].copy_from_slice(&x[src..src + len]);
            }
        }
    }
    out
}

#[cfg(test)]
mod channel_permute_tests {
    use super::*;

    #[test]
    fn channel_permute_is_the_group_transpose_and_its_inverse() {
        // 6 channels, groups = 2 -> rows 2, cols 3: out[j*2+i] = in[i*3+j]
        let x: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let shuffled = channel_permute(&x, 1, 2, 3, 1);
        assert_eq!(shuffled, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(channel_permute(&shuffled, 1, 3, 2, 1), x);
        // two samples, len 2: whole pixel runs move together
        let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let y2 = channel_permute(&x2, 2, 2, 3, 2);
        assert_eq!(&y2[0..4], &[0.0, 1.0, 6.0, 7.0]); // channels 0 and 3 of sample 0
        assert_eq!(&y2[12..16], &[12.0, 13.0, 18.0, 19.0]);
    }
}

/// Row-wise LayerNorm over a flat `rows x cols` buffer (reference for
/// [`crate::F32TensorOpsBackend::layernorm_forward_f32`]).
///
/// `y = gamma * (x - mean) * inv_std + beta`, with `inv_std = 1/sqrt(var + eps)`
/// and `var` the *population* variance (biased, matching torch and the C++ LAYA
/// and ViT layers).  Non-finite inputs count as 0, which is what the LAYA
/// scorer's sanitiser does before normalising.
///
/// `gamma` and `beta` may be `None` for the bias-free variant (ModernBERT
/// `attn_norm` / `mlp_norm`, and `[laya_encoder]`, are all bias-free).
pub fn layernorm(
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    assert!(rows > 0 && cols > 0 && x.len() >= rows * cols, "layernorm: bad shape");
    let mut out = vec![0.0f32; rows * cols];
    let inv_c = 1.0 / cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for v in row {
            mean += if v.is_finite() { *v } else { 0.0 };
        }
        mean *= inv_c;
        let mut var = 0.0f32;
        for v in row {
            let d = (if v.is_finite() { *v } else { 0.0 }) - mean;
            var += d * d;
        }
        var *= inv_c;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter().enumerate() {
            let normalised = ((if v.is_finite() { *v } else { 0.0 }) - mean) * inv_std;
            let g = gamma.map_or(1.0, |g| g[i]);
            let b = beta.map_or(0.0, |b| b[i]);
            out[r * cols + i] = g * normalised + b;
        }
    }
    out
}

/// Row-wise LayerNorm backward over a flat `rows x cols` buffer: the input
/// gradient only (reference for
/// [`crate::F32TensorOpsBackend::layernorm_backward_f32`]).
///
/// With `xhat = (x - mean) * inv_std`, `g = dy * gamma * inv_std` and
/// `m = mean(xhat)`:
///
/// ```text
/// dx = g - mean(g) - xhat * mean(g * xhat)
/// ```
///
/// `mean` and `inv_std` are recomputed from `x` rather than cached, so the
/// backward needs no saved state; that is one extra reduction pass over a row the
/// kernel is already reading, and it keeps the op's signature to tensors only.
pub fn layernorm_backward(
    dy: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    gamma: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    assert!(rows > 0 && cols > 0, "layernorm_backward: bad shape");
    assert!(dy.len() >= rows * cols && x.len() >= rows * cols, "layernorm_backward: bad shape");
    let mut dx = vec![0.0f32; rows * cols];
    let inv_c = 1.0 / cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let grad = &dy[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for v in row {
            mean += if v.is_finite() { *v } else { 0.0 };
        }
        mean *= inv_c;
        let mut var = 0.0f32;
        for v in row {
            let d = (if v.is_finite() { *v } else { 0.0 }) - mean;
            var += d * d;
        }
        var *= inv_c;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut xhat = vec![0.0f32; cols];
        let mut g = vec![0.0f32; cols];
        let mut mean_g = 0.0f32;
        let mut mean_gx = 0.0f32;
        for i in 0..cols {
            xhat[i] = ((if row[i].is_finite() { row[i] } else { 0.0 }) - mean) * inv_std;
            g[i] = grad[i] * gamma.map_or(1.0, |g| g[i]) * inv_std;
            mean_g += g[i];
            mean_gx += g[i] * xhat[i];
        }
        mean_g *= inv_c;
        mean_gx *= inv_c;
        for i in 0..cols {
            dx[r * cols + i] = g[i] - mean_g - xhat[i] * mean_gx;
        }
    }
    dx
}

#[cfg(test)]
mod layernorm_tests {
    use super::*;

    #[test]
    fn a_constant_row_normalises_to_the_bias() {
        let x = [2.0f32, 2.0, 2.0, 2.0];
        let beta = [0.5f32, -0.5, 0.25, 0.0];
        assert_eq!(layernorm(&x, 1, 4, None, Some(&beta), 1e-5), beta);
        // Without bias the whole row collapses to zero.
        assert_eq!(layernorm(&x, 1, 4, None, None, 1e-5), vec![0.0; 4]);
    }

    #[test]
    fn rows_are_normalised_independently_and_gamma_scales_them() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = [2.0f32; 4];
        let out = layernorm(&x, 2, 4, Some(&gamma), None, 1e-5);
        let expect = |v: f32, mean: f32, var: f32| 2.0 * (v - mean) / (var + 1e-5).sqrt();
        assert!((out[0] - expect(1.0, 2.5, 1.25)).abs() < 1e-5, "{}", out[0]);
        assert!((out[7] - expect(40.0, 25.0, 125.0)).abs() < 1e-5, "{}", out[7]);
        // Non-finite inputs are treated as zero.
        let dirty = [f32::NAN, 1.0, f32::INFINITY, 3.0];
        let out = layernorm(&dirty, 1, 4, None, None, 1e-5);
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }

    #[test]
    fn the_backward_is_the_exact_derivative() {
        // Central finite differences of sum(g * layernorm(x)) against
        // layernorm_backward, with a non-trivial gamma.
        let x: Vec<f32> = (0..12).map(|i| ((i * 5 % 7) as f32 - 3.0) / 2.0).collect();
        let gamma: Vec<f32> = (0..4).map(|i| 0.5 + 0.2 * i as f32).collect();
        let beta: Vec<f32> = (0..4).map(|i| -0.1 * i as f32).collect();
        let g: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37 - 0.5).sin()).collect();
        let (rows, cols) = (3usize, 4usize);
        let dx = layernorm_backward(&g, &x, rows, cols, Some(&gamma), 1e-5);
        let eps = 1e-3f32;
        for i in 0..x.len() {
            let mut hi = x.clone();
            hi[i] += eps;
            let mut lo = x.clone();
            lo[i] -= eps;
            let f = |v: &[f32]| -> f32 {
                let y = layernorm(v, rows, cols, Some(&gamma), Some(&beta), 1e-5);
                g.iter().zip(&y).map(|(a, b)| a * b).sum()
            };
            let numeric = (f(&hi) - f(&lo)) / (2.0 * eps);
            let slack = 2e-2 * dx[i].abs().max(1e-2);
            assert!((numeric - dx[i]).abs() <= slack, "i={i}: numeric {numeric} analytic {}", dx[i]);
        }
    }

    #[test]
    fn the_backward_kills_the_all_ones_direction() {
        // A shift of every element in a row leaves the normalised output
        // unchanged, so the sum of that row's input gradient must be zero.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let ones = [1.0f32; 4];
        let dx = layernorm_backward(&ones, &x, 1, 4, None, 1e-5);
        assert!(dx.iter().sum::<f32>().abs() < 1e-5, "{dx:?}");
    }

    #[test]
    fn the_softmax_backward_matches_the_full_jacobian() {
        // For a uniform group y = 1/n, dy = ones: dx = y - (1/n)*sum(y) ...
        let y = [0.25f32; 4];
        let dy = [1.0f32; 4];
        let dx = softmax_backward(&y, &dy, 4);
        // dot = sum(y*dy) = 1, so dx = 0.25 * (1 - 1) = 0.
        assert!(dx.iter().all(|v| v.abs() < 1e-6), "{dx:?}");
        // With dy = y the gradient is y * (y - sum(y^2)).
        let dy2 = y.to_vec();
        let dx2 = softmax_backward(&y, &dy2, 4);
        let want = 0.25 * (0.25 - 0.25f32);
        assert!((dx2[0] - want).abs() < 1e-6, "{} vs {want}", dx2[0]);
        // Multiple groups are independent (a group of 2 plus a group of 1).
        let y3 = [0.5f32, 0.5, 1.0, 1.0];
        let dy3 = [1.0f32, 3.0, 2.0, 5.0];
        let dx3 = softmax_backward(&y3, &dy3, 2);
        assert!((dx3[0] - 0.5 * (1.0 - 2.0)).abs() < 1e-6, "{}", dx3[0]);
        // Second group: y = 1, dy = (2, 5) -> dot 7 -> dx = -5 each.
        assert!((dx3[2] - (2.0 - 7.0)).abs() < 1e-6, "{}", dx3[2]);
        assert!((dx3[3] - (5.0 - 7.0)).abs() < 1e-6, "{}", dx3[3]);
    }
}

/// Fused residual + bias epilogue for a transformer block: over a flat
/// `rows x cols` row-major slab,
/// `out[i] = residual[i] + y[i] + bias[i % cols]`.
///
/// The bias is per **column** (the last dimension), which is how every ported
/// attention / FFN output projection is biased - the row index is the token and
/// must not affect the offset.  `bias` may be `None` for the bias-free
/// projections.  Reference for
/// [`crate::F32TensorOpsBackend::residual_bias_f32`].
///
/// Doing this in one pass matters: the unfused form
/// (`bias_add` then `vector_add`) materialises a full extra temporary per
/// residual connection, and a transformer block has two of them.
pub fn residual_bias(
    residual: &[f32],
    y: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let len = rows * cols;
    assert!(residual.len() >= len && y.len() >= len, "residual_bias: bad shape");
    let mut out = vec![0.0f32; len];
    for i in 0..len {
        let b = match bias {
            Some(bias) => {
                assert!(bias.len() >= cols, "residual_bias: bias must have cols entries");
                bias[i % cols]
            }
            None => 0.0,
        };
        out[i] = residual[i] + y[i] + b;
    }
    out
}

#[cfg(test)]
mod residual_bias_tests {
    use super::*;

    #[test]
    fn the_bias_is_per_column_and_not_per_row() {
        let residual = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y = [0.0f32; 6];
        let bias = [10.0f32, 20.0];
        // rows = 3, cols = 2 -> the pattern repeats with period `cols`.
        assert_eq!(residual_bias(&residual, &y, Some(&bias), 3, 2), vec![11.0, 21.0, 11.0, 21.0, 11.0, 21.0]);
        // No bias is a plain residual add.
        assert_eq!(residual_bias(&residual, &y, None, 3, 2), residual);
    }

    #[test]
    #[should_panic(expected = "residual_bias: bad shape")]
    fn a_short_buffer_is_rejected() {
        // rows * cols = 6 but only 4 elements are supplied.
        let _ = residual_bias(&[0.0; 4], &[0.0; 4], None, 3, 2);
    }

    #[test]
    #[should_panic(expected = "residual_bias: bias must have cols entries")]
    fn a_short_bias_is_rejected() {
        let _ = residual_bias(&[0.0; 6], &[0.0; 6], Some(&[1.0]), 3, 2);
    }
}

/// The batched-GEMM wrapper every transformer layer's projections need:
/// `out[r, n] = sum_k x[r, k] * w[n, k] + b[n]`, i.e. `x @ w^T + b` with the
/// **torch `nn.Linear` weight layout** (`w` is `[out_features, in_features]`,
/// row-major) over a flat `rows x in_features` activation.
///
/// `rows` is the batch being folded into the GEMM's `m` dimension, which is what
/// makes one call cover every token of a sequence.  `bias` may be `None` for the
/// bias-free projections (`[laya_encoder]`'s `Wqkv` / `Wo`, for instance).
/// Reference for [`crate::F32TensorOpsBackend::linear_f32`].
pub fn linear(
    x: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    weights: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    assert!(rows > 0 && in_features > 0 && out_features > 0, "linear: bad shape");
    assert!(x.len() >= rows * in_features, "linear: activation too short");
    assert!(weights.len() >= out_features * in_features, "linear: weights too short");
    let mut out = vec![0.0f32; rows * out_features];
    for r in 0..rows {
        let x_row = &x[r * in_features..(r + 1) * in_features];
        let o_row = &mut out[r * out_features..(r + 1) * out_features];
        for (n, dst) in o_row.iter_mut().enumerate() {
            let w_row = &weights[n * in_features..(n + 1) * in_features];
            let mut acc = 0.0f32;
            for k in 0..in_features {
                acc += x_row[k] * w_row[k];
            }
            *dst = acc + bias.map_or(0.0, |b| b[n]);
        }
    }
    out
}

#[cfg(test)]
mod linear_tests {
    use super::*;

    #[test]
    fn it_is_the_hand_computed_row_dot_product() {
        // rows = 2, in = 3, out = 2.  w = [[1,2,3],[4,5,6]], b = [10, 20].
        let x = [1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5];
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Row 0: [1,0,-1] . [1,2,3] = -2 and . [4,5,6] = -2; with the bias
        // those become 8 and 18.
        // Row 1: [0.5,0.5,0.5] . [1,2,3] = 3 and . [4,5,6] = 7.5; biased ->
        // 13 and 27.5.
        let out = linear(&x, 2, 3, 2, &w, Some(&[10.0, 20.0]));
        assert_eq!(out, vec![8.0, 18.0, 13.0, 27.5]);
        let out = linear(&x, 2, 3, 2, &w, None);
        assert_eq!(out, vec![-2.0, -2.0, 3.0, 7.5]);
    }

    #[test]
    fn every_token_gets_its_own_output_row() {
        // rows = 4, in = 1, out = 1: the result is the input scaled, one row each.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let out = linear(&x, 4, 1, 1, &[0.5], Some(&[1.0]));
        assert_eq!(out, vec![1.5, 2.0, 2.5, 3.0]);
    }
}

#[cfg(test)]
mod eml_convolutional_tests {
    use super::*;

    #[test]
    fn branch_forward_and_backward_match_finite_differences() {
        let (clamp, eps, scale) = (2.5f32, 1e-4f32, 0.7f32);
        let x = [-3.2f32, -0.8, 0.4, 3.1];
        let y = [-25.0f32, -1.3, 0.7, 22.0];
        let residual = [0.2f32, -0.1, 0.3, 0.5];
        let delta = [0.4f32, -0.3, 0.8, -0.5];
        let output = eml_convolutional_forward(&x, &y, Some(&residual), clamp, eps, scale);
        let no_residual = eml_convolutional_forward(&x, &y, None, clamp, eps, scale);
        assert_eq!(
            eml_convolutional_forward(&x, &y, None, 0.0, -1.0, scale),
            eml_convolutional_forward(&x, &y, None, 4.0, 1e-6, scale),
            "C++ nonpositive clamp/epsilon fallbacks"
        );
        for i in 0..output.len() {
            assert!((output[i] - no_residual[i] - residual[i]).abs() < 3e-7);
        }
        let (dx, dy) = eml_convolutional_backward(&x, &y, &delta, clamp, eps, scale);
        let loss = |xv: &[f32], yv: &[f32]| {
            eml_convolutional_forward(xv, yv, Some(&residual), clamp, eps, scale)
                .iter()
                .zip(delta)
                .map(|(value, d)| value * d)
                .sum::<f32>()
        };
        let mut xv = x;
        let mut yv = y;
        let h = 1e-3;
        for i in 0..x.len() {
            let original = xv[i];
            xv[i] = original + h;
            let plus = loss(&xv, &yv);
            xv[i] = original - h;
            let minus = loss(&xv, &yv);
            xv[i] = original;
            let fd_x = (plus - minus) / (2.0 * h);
            assert!((dx[i] - fd_x).abs() < 3e-3, "dx[{i}]: {} vs {fd_x}", dx[i]);

            let original = yv[i];
            yv[i] = original + h;
            let plus = loss(&xv, &yv);
            yv[i] = original - h;
            let minus = loss(&xv, &yv);
            yv[i] = original;
            let fd_y = (plus - minus) / (2.0 * h);
            assert!((dy[i] - fd_y).abs() < 3e-3, "dy[{i}]: {} vs {fd_y}", dy[i]);
        }
    }
}
