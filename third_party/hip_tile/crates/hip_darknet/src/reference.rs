//! Straightforward f32 CPU forward pass. Ground truth for the GPU executor; returns every layer's output.

use crate::error::{Error, Result};
use crate::model::{Conv, LayerKind, Model, Shape};
use crate::cfg::MaxpoolCfg;

/// Run the network on one CHW f32 image (`model.input_shape()`), returning each layer's output (CHW).
/// `[route]` outputs are copies; `[yolo]` outputs are the raw (pre-sigmoid) input, see `yolo::decode`.
pub fn forward(model: &Model, input: &[f32]) -> Result<Vec<Vec<f32>>> {
    let in_shape = model.input_shape();
    if input.len() != in_shape.len() {
        return Err(Error::Model(format!("input has {} floats, expected {}", input.len(), in_shape.len())));
    }
    let mut outs: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    for (i, layer) in model.layers.iter().enumerate() {
        let prev: &[f32] = if i == 0 { input } else { &outs[i - 1] };
        let out = match &layer.kind {
            LayerKind::Conv(c) => {
                if c.weights.len() != c.n_weights() {
                    return Err(Error::Model(format!("layer {i}: weights not loaded")));
                }
                conv2d(prev, layer.in_shape, c, layer.out_shape)
            }
            LayerKind::Maxpool(m) => maxpool(prev, layer.in_shape, m, layer.out_shape),
            LayerKind::Route { sources, groups, group_id } => {
                let parts: Vec<&[f32]> = sources.iter().map(|&s| outs[s].as_slice()).collect();
                route(&parts, *groups, *group_id)
            }
            LayerKind::Upsample { stride } => upsample(prev, layer.in_shape, *stride),
            LayerKind::Yolo(_) => prev.to_vec(),
        };
        outs.push(out);
    }
    Ok(outs)
}

/// Convolution with bias and activation. `weights` is `[out][in][ky][kx]`.
pub fn conv2d(input: &[f32], ins: Shape, conv: &Conv, outs: Shape) -> Vec<f32> {
    let (k, s, p) = (conv.cfg.size, conv.cfg.stride as isize, conv.cfg.padding as isize);
    let (iw, ih, ow, oh) = (ins.w as isize, ins.h as isize, outs.w, outs.h);
    let mut out = vec![0f32; outs.len()];
    for oc in 0..outs.c {
        let plane = &mut out[oc * oh * ow..(oc + 1) * oh * ow];
        plane.fill(conv.bias[oc]);
        let wbase = oc * conv.k();
        for ic in 0..ins.c {
            let inp = &input[ic * ins.h * ins.w..(ic + 1) * ins.h * ins.w];
            for ky in 0..k {
                for kx in 0..k {
                    let wv = conv.weights[wbase + (ic * k + ky) * k + kx];
                    // ox range where 0 <= ox*s + kx - p < iw.
                    let off = kx as isize - p;
                    let lo = if off >= 0 { 0 } else { (-off + s - 1) / s };
                    let hi_n = iw - 1 - off; // ox*s <= hi_n
                    if hi_n < 0 {
                        continue;
                    }
                    let hi = (hi_n / s).min(ow as isize - 1);
                    if lo > hi {
                        continue;
                    }
                    for oy in 0..oh {
                        let iy = oy as isize * s + ky as isize - p;
                        if iy < 0 || iy >= ih {
                            continue;
                        }
                        let dst = &mut plane[oy * ow + lo as usize..oy * ow + hi as usize + 1];
                        let first = (lo * s + off) as usize;
                        let row = &inp[iy as usize * ins.w..(iy as usize + 1) * ins.w];
                        if s == 1 {
                            let n = dst.len();
                            for (d, x) in dst.iter_mut().zip(&row[first..first + n]) {
                                *d += wv * x;
                            }
                        } else {
                            for (j, d) in dst.iter_mut().enumerate() {
                                *d += wv * row[first + j * s as usize];
                            }
                        }
                    }
                }
            }
        }
        for v in plane.iter_mut() {
            *v = conv.cfg.activation.apply(*v);
        }
    }
    out
}

/// Darknet maxpool: window origin `-padding / 2` (integer division), out-of-image taps ignored.
pub fn maxpool(input: &[f32], ins: Shape, m: &MaxpoolCfg, outs: Shape) -> Vec<f32> {
    let off = -((m.padding / 2) as isize);
    let mut out = vec![0f32; outs.len()];
    for c in 0..outs.c {
        for oy in 0..outs.h {
            for ox in 0..outs.w {
                let mut best = -f32::MAX;
                for ky in 0..m.size {
                    for kx in 0..m.size {
                        let iy = off + (oy * m.stride + ky) as isize;
                        let ix = off + (ox * m.stride + kx) as isize;
                        if iy >= 0 && iy < ins.h as isize && ix >= 0 && ix < ins.w as isize {
                            let v = input[(c * ins.h + iy as usize) * ins.w + ix as usize];
                            if v > best {
                                best = v;
                            }
                        }
                    }
                }
                out[(c * outs.h + oy) * outs.w + ox] = best;
            }
        }
    }
    out
}

/// Nearest-neighbour upsample.
pub fn upsample(input: &[f32], ins: Shape, stride: usize) -> Vec<f32> {
    let (oh, ow) = (ins.h * stride, ins.w * stride);
    let mut out = vec![0f32; ins.c * oh * ow];
    for c in 0..ins.c {
        for y in 0..oh {
            for x in 0..ow {
                out[(c * oh + y) * ow + x] = input[(c * ins.h + y / stride) * ins.w + x / stride];
            }
        }
    }
    out
}

/// Concatenate channel group `group_id` (of `groups` equal channel blocks) of every source.
/// CHW layout makes each group a contiguous slice.
pub fn route(parts: &[&[f32]], groups: usize, group_id: usize) -> Vec<f32> {
    let mut out = vec![];
    for p in parts {
        let n = p.len() / groups;
        out.extend_from_slice(&p[n * group_id..n * (group_id + 1)]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::{Activation, ConvCfg};

    #[allow(clippy::too_many_arguments)]
    fn conv(c_in: usize, filters: usize, size: usize, stride: usize, padding: usize, act: Activation, w: Vec<f32>, b: Vec<f32>) -> Conv {
        Conv {
            cfg: ConvCfg { filters, size, stride, padding, batch_normalize: false, activation: act },
            c_in,
            weights: w,
            bias: b,
        }
    }

    #[test]
    fn conv_3x3_pad1_hand_computed() {
        // 1ch 3x3 input 1..9, all-ones kernel: output = sum of the 3x3 neighbourhood (zero padded).
        let inp: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let c = conv(1, 1, 3, 1, 1, Activation::Linear, vec![1.0; 9], vec![0.5]);
        let s = Shape { c: 1, h: 3, w: 3 };
        let out = conv2d(&inp, s, &c, s);
        assert_eq!(out, vec![12.5, 21.5, 16.5, 27.5, 45.5, 33.5, 24.5, 39.5, 28.5]);
    }

    #[test]
    fn conv_stride2_and_channels() {
        // 2ch 4x4, 1x1 kernel stride 2 picks every other pixel: w = [1, 10] mixes channels.
        let mut inp = vec![];
        inp.extend((0..16).map(|v| v as f32));
        inp.extend((0..16).map(|v| 100.0 + v as f32));
        let c = conv(2, 1, 1, 2, 0, Activation::Linear, vec![1.0, 10.0], vec![0.0]);
        let out = conv2d(&inp, Shape { c: 2, h: 4, w: 4 }, &c, Shape { c: 1, h: 2, w: 2 });
        // p = flat index of the sampled pixel (0, 2, 8, 10): p + 10 * (100 + p).
        assert_eq!(out, vec![1000.0, 1022.0, 1088.0, 1110.0]);
    }

    #[test]
    fn conv_stride2_3x3_pad1_edges() {
        // 4x4 input of ones, 3x3 ones kernel, stride 2, pad 1 -> out 2x2.
        // out(0,0) covers rows/cols -1..1 -> 2x2 valid = 4; out(0,1) covers cols 1..3 -> 2x3 = 6.
        let c = conv(1, 1, 3, 2, 1, Activation::Linear, vec![1.0; 9], vec![0.0]);
        let out = conv2d(&[1.0; 16], Shape { c: 1, h: 4, w: 4 }, &c, Shape { c: 1, h: 2, w: 2 });
        assert_eq!(out, vec![4.0, 6.0, 6.0, 9.0]);
    }

    #[test]
    fn leaky_and_logistic() {
        assert_eq!(Activation::Leaky.apply(-2.0), -0.2);
        assert_eq!(Activation::Leaky.apply(3.0), 3.0);
        assert_eq!(Activation::Leaky.apply(0.0), 0.0);
        assert!((Activation::Logistic.apply(0.0) - 0.5).abs() < 1e-7);
        let c = conv(1, 1, 1, 1, 0, Activation::Leaky, vec![1.0], vec![0.0]);
        let s = Shape { c: 1, h: 1, w: 2 };
        assert_eq!(conv2d(&[-10.0, 4.0], s, &c, s), vec![-1.0, 4.0]);
    }

    #[test]
    fn maxpool_odd_size_uses_padding() {
        // 3x3 in, size 2 stride 2 padding 1 (default): out = (3+1-2)/2+1 = 2, offset 0.
        // Window (1,1) covers rows/cols 2..3 with only (2,2) valid.
        let inp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let m = MaxpoolCfg { size: 2, stride: 2, padding: 1 };
        let out = maxpool(&inp, Shape { c: 1, h: 3, w: 3 }, &m, Shape { c: 1, h: 2, w: 2 });
        assert_eq!(out, vec![5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn maxpool_negative_values() {
        let m = MaxpoolCfg { size: 2, stride: 2, padding: 1 };
        let out = maxpool(&[-3.0, -1.0, -2.0, -4.0], Shape { c: 1, h: 2, w: 2 }, &m, Shape { c: 1, h: 1, w: 1 });
        assert_eq!(out, vec![-1.0]);
    }

    #[test]
    fn upsample_nearest() {
        let out = upsample(&[1.0, 2.0, 3.0, 4.0], Shape { c: 1, h: 2, w: 2 }, 2);
        assert_eq!(out, vec![1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]);
    }

    #[test]
    fn route_group_slices() {
        let a = [1., 2., 3., 4.]; // 4 channels of 1 pixel
        let b = [5., 6.];
        assert_eq!(route(&[&a], 2, 1), vec![3., 4.]);
        assert_eq!(route(&[&a, &b], 1, 0), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(route(&[&a, &b], 2, 0), vec![1., 2., 5.]);
    }
}
