//! A training-mode network: f32 forward that keeps BN unfolded, batch statistics, Darknet's
//! `[yolo]` deltas, the full backward pass, and the SGD step.
//!
//! Layout is NCHW with a batch: element `(b, c, y, x)` is at `((b*C + c)*H + y)*W + x`.
//! Semantics: `_shared/darknet-semantics.md` (Training loss / Training deltas / Batch norm,
//! training path / SGD and learning rate / Gradient routing).

use crate::augment::Rng;
use crate::cfg::{Activation, Cfg, ConvCfg, LrPolicy, MaxpoolCfg, TrainCfg, YoloCfg};
use crate::data::BoxLabel;
use crate::error::{Error, Result};
use crate::model::{LayerKind, Model, Shape};
use crate::train::deltas::{self, HeadStats};

/// Batch-norm epsilon of the training path (`batchnorm_layer.cpp:106-133`).
pub const BN_EPS: f32 = 1e-5;
/// Momentum of the rolling statistics (`0.9*old + 0.1*new`).
pub const ROLLING_MOMENTUM: f32 = 0.9;

/// One conv layer's trainable parameters and momentum buffers.
#[derive(Debug, Clone)]
pub struct ConvState {
    /// `[out][in * kh * kw]`, Darknet order.
    pub weights: Vec<f32>,
    /// Conv bias; for a batch-normalised layer this is BN's shift, which Darknet never trains
    /// (`update_convolutional_layer` applies `bias_updates` that the BN path never fills).
    pub bias: Vec<f32>,
    pub scales: Vec<f32>,
    /// Per-channel batch mean / (unbiased) batch variance of the last forward.
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_var: Vec<f32>,
    pub weight_updates: Vec<f32>,
    pub bias_updates: Vec<f32>,
    pub scale_updates: Vec<f32>,
}

/// A network that can be trained. Mirrors [`Model`]'s layer list, one buffer per layer.
pub struct TrainNet {
    pub cfg: Cfg,
    pub model: Model,
    pub batch: usize,
    /// Aligned with `model.layers`; `Some` only for conv layers.
    pub convs: Vec<Option<ConvState>>,
    /// Per layer activation, NCHW with the batch.
    pub outs: Vec<Vec<f32>>,
    /// Per layer gradient w.r.t. `outs[i]`, NCHW with the batch.
    pub grads: Vec<Vec<f32>>,
    /// Per conv layer: the conv sum before normalisation.
    pub conv_sum: Vec<Option<Vec<f32>>>,
    /// Per conv layer: the normalised (pre-scale) value, Darknet's `x_norm`.
    pub conv_norm: Vec<Option<Vec<f32>>>,
    /// Per maxpool layer: argmax offset inside a channel plane.
    pub pool_idx: Vec<Option<Vec<u32>>>,
    /// The batch input, NCHW.
    pub input: Vec<f32>,
    /// Gradient w.r.t. the input (layer 0's incoming gradient).
    pub input_grad: Vec<f32>,
    /// Last computed network loss.
    pub loss: f32,
}

impl TrainNet {
    /// Build from a cfg; weights are initialised like Darknet (`convolutional_layer.cpp:1804-1814`),
    /// BN shift at zero, rolling stats `0/1`.
    pub fn new(cfg: &Cfg, batch: usize, seed: u64) -> Result<TrainNet> {
        let model = Model::from_cfg(cfg)?;
        let mut rng = Rng::new(seed);
        let mut convs = Vec::with_capacity(model.layers.len());
        let mut outs = Vec::with_capacity(model.layers.len());
        let mut grads = Vec::with_capacity(model.layers.len());
        let mut conv_sum = Vec::with_capacity(model.layers.len());
        let mut conv_norm = Vec::with_capacity(model.layers.len());
        let mut pool_idx = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let n = batch * layer.out_shape.len();
            outs.push(vec![0.0; n]);
            grads.push(vec![0.0; n]);
            match &layer.kind {
                LayerKind::Conv(c) => {
                    let filters = c.cfg.filters;
                    let k = c.k();
                    let scale = (2.0 / k as f32).sqrt();
                    let weights = (0..filters * k).map(|_| rng.uniform(-1.0, 1.0) * scale).collect();
                    convs.push(Some(ConvState {
                        weights,
                        bias: vec![0.0; filters],
                        scales: vec![1.0; filters],
                        mean: vec![0.0; filters],
                        var: vec![1.0; filters],
                        rolling_mean: vec![0.0; filters],
                        rolling_var: vec![1.0; filters],
                        weight_updates: vec![0.0; filters * k],
                        bias_updates: vec![0.0; filters],
                        scale_updates: vec![0.0; filters],
                    }));
                    conv_sum.push(Some(vec![0.0; n]));
                    conv_norm.push(Some(vec![0.0; n]));
                    pool_idx.push(None);
                }
                LayerKind::Maxpool(_) => {
                    convs.push(None);
                    conv_sum.push(None);
                    conv_norm.push(None);
                    pool_idx.push(Some(vec![0; n]));
                }
                _ => {
                    convs.push(None);
                    conv_sum.push(None);
                    conv_norm.push(None);
                    pool_idx.push(None);
                }
            }
        }
        let input_len = batch * model.input_shape().len();
        Ok(TrainNet {
            cfg: cfg.clone(),
            model,
            batch,
            convs,
            outs,
            grads,
            conv_sum,
            conv_norm,
            pool_idx,
            input: vec![0.0; input_len],
            input_grad: vec![0.0; input_len],
            loss: 0.0,
        })
    }

    pub fn net_w(&self) -> usize {
        self.model.net.width
    }
    pub fn net_h(&self) -> usize {
        self.model.net.height
    }

    /// Copy one batch of CHW images into the input buffer.
    pub fn set_input(&mut self, chw: &[f32]) -> Result<()> {
        if chw.len() != self.input.len() {
            return Err(Error::Model(format!("input has {} floats, expected {}", chw.len(), self.input.len())));
        }
        self.input.copy_from_slice(chw);
        Ok(())
    }

    /// Training-mode forward over the whole batch.
    pub fn forward(&mut self) -> Result<()> {
        let model = &self.model;
        let batch = self.batch;
        for i in 0..model.layers.len() {
            let layer = &model.layers[i];
            let (ins, shape) = (layer.in_shape, layer.out_shape);
            // Take the destination out so the sources can be read while it is filled.
            let mut dst = std::mem::take(&mut self.outs[i]);
            match &layer.kind {
                LayerKind::Conv(c) => {
                    let prev: &[f32] = if i == 0 { &self.input } else { &self.outs[i - 1] };
                    let conv = self.convs[i].as_mut().expect("conv state");
                    let sum = self.conv_sum[i].as_mut().expect("conv sum");
                    conv_forward(sum, prev, &conv.weights, &conv.bias, ins, shape, &c.cfg, batch);
                    if c.cfg.batch_normalize {
                        bn_forward(
                            sum,
                            &mut dst,
                            self.conv_norm[i].as_mut().expect("norm"),
                            &mut conv.mean,
                            &mut conv.var,
                            &mut conv.rolling_mean,
                            &mut conv.rolling_var,
                            &conv.scales,
                            &conv.bias,
                            batch,
                            shape,
                        );
                    } else {
                        dst.copy_from_slice(sum);
                    }
                    apply_activation(&mut dst, c.cfg.activation);
                }
                LayerKind::Maxpool(m) => {
                    let prev: &[f32] = if i == 0 { &self.input } else { &self.outs[i - 1] };
                    let idx = self.pool_idx[i].as_mut().expect("pool idx");
                    maxpool_forward(prev, &mut dst, idx, ins, shape, m, batch);
                }
                LayerKind::Route { sources, groups, group_id } => {
                    route_forward(&self.outs, &mut dst, sources, *groups, *group_id, model, batch, shape);
                }
                LayerKind::Upsample { stride } => {
                    let prev: &[f32] = if i == 0 { &self.input } else { &self.outs[i - 1] };
                    upsample_forward(prev, &mut dst, ins, shape, *stride, batch);
                }
                LayerKind::Yolo(y) => {
                    let prev: &[f32] = if i == 0 { &self.input } else { &self.outs[i - 1] };
                    yolo_forward(prev, &mut dst, shape, y, batch);
                }
            }
            self.outs[i] = dst;
        }
        Ok(())
    }

    /// Compute the `[yolo]` deltas for one batch and return the network loss
    /// (the mean of the per-head costs, `darknet_network.cpp:get_network_cost`).
    pub fn loss_and_deltas(&mut self, truths: &[Vec<BoxLabel>]) -> Result<f32> {
        if truths.len() != self.batch {
            return Err(Error::Model(format!("{} truth sets for a batch of {}", truths.len(), self.batch)));
        }
        let heads = self.model.yolo_layers();
        if heads.is_empty() {
            return Err(Error::Model("cfg has no [yolo] layer".into()));
        }
        let (net_w, net_h) = (self.net_w(), self.net_h());
        let mut total = 0.0f32;
        for &i in &heads {
            let LayerKind::Yolo(y) = &self.model.layers[i].kind else { unreachable!() };
            let shape = self.model.layers[i].out_shape;
            let plain = shape.len();
            let mut stats = HeadStats::default();
            let mut sq_all = 0.0f32;
            let mut sq_nobox = 0.0f32;
            let prev = i - 1;
            for b in 0..self.batch {
                let out = &self.outs[i][b * plain..(b + 1) * plain];
                let mut delta = vec![0.0f32; plain];
                let s = deltas::yolo_delta(out, shape, y, &truths[b], net_w, net_h, &mut delta);
                stats.count += s.count;
                stats.tot_iou += s.tot_iou;
                stats.tot_iou_loss += s.tot_iou_loss;
                stats.tot_giou_loss += s.tot_giou_loss;
                sq_all += deltas::delta_sq(&delta, shape, y, false);
                sq_nobox += deltas::delta_sq(&delta, shape, y, true);
                // Darknet adds the head's delta into the previous layer's delta buffer.
                let g = &mut self.grads[prev][b * plain..(b + 1) * plain];
                for (gi, &d) in g.iter_mut().zip(&delta) {
                    *gi += d;
                }
            }
            let cost = deltas::head_cost_from_sums(y, &stats, sq_all, sq_nobox);
            total += cost;
        }
        self.loss = total / heads.len() as f32;
        Ok(self.loss)
    }

    /// Backward through every layer, accumulating parameter updates and input gradients.
    pub fn backward(&mut self) {
        let model = &self.model;
        let batch = self.batch;
        for i in (0..model.layers.len()).rev() {
            let layer = &model.layers[i];
            let (ins, shape) = (layer.in_shape, layer.out_shape);
            if matches!(layer.kind, LayerKind::Yolo(_)) {
                continue;
            }
            let mut delta = std::mem::take(&mut self.grads[i]);
            match &layer.kind {
                LayerKind::Conv(c) => {
                    let prev_out: &[f32] = if i == 0 { &self.input } else { &self.outs[i - 1] };
                    let post = &self.outs[i];
                    let conv = self.convs[i].as_mut().expect("conv state");
                    if i == 0 {
                        conv_backward(
                            &mut delta,
                            &mut self.input_grad,
                            post,
                            prev_out,
                            shape,
                            ins,
                            &c.cfg,
                            conv,
                            self.conv_sum[i].as_deref(),
                            self.conv_norm[i].as_deref(),
                            batch,
                        );
                    } else {
                        let prev_grad = &mut self.grads[i - 1];
                        conv_backward(
                            &mut delta,
                            prev_grad,
                            post,
                            prev_out,
                            shape,
                            ins,
                            &c.cfg,
                            conv,
                            self.conv_sum[i].as_deref(),
                            self.conv_norm[i].as_deref(),
                            batch,
                        );
                    }
                }
                LayerKind::Maxpool(m) => {
                    let idx = self.pool_idx[i].as_deref().expect("pool idx");
                    if i == 0 {
                        maxpool_backward(&delta, &mut self.input_grad, idx, ins, m, batch);
                    } else {
                        maxpool_backward(&delta, &mut self.grads[i - 1], idx, ins, m, batch);
                    }
                }
                LayerKind::Route { sources, groups, group_id } => {
                    route_backward(&delta, &mut self.grads, sources, *groups, *group_id, model, batch, shape);
                }
                LayerKind::Upsample { stride } => {
                    if i == 0 {
                        upsample_backward(&delta, &mut self.input_grad, ins, *stride, batch);
                    } else {
                        upsample_backward(&delta, &mut self.grads[i - 1], ins, *stride, batch);
                    }
                }
                LayerKind::Yolo(_) => unreachable!(),
            }
            self.grads[i] = delta;
        }
    }

    /// Apply Darknet's SGD step (`convolutional_layer.cpp:2983-3005`) to every conv layer.
    pub fn update(&mut self, lr: f32) {
        let t = &self.cfg.train;
        let batch = t.effective_batch() as f32;
        for conv in self.convs.iter_mut().flatten() {
            let step = lr / batch;
            for j in 0..conv.weights.len() {
                conv.weight_updates[j] += -t.decay * batch * conv.weights[j];
                conv.weights[j] += step * conv.weight_updates[j];
                conv.weight_updates[j] *= t.momentum;
            }
            // Bias and BN scales: no weight decay.
            for j in 0..conv.bias.len() {
                conv.bias[j] += step * conv.bias_updates[j];
                conv.bias_updates[j] *= t.momentum;
                conv.scales[j] += step * conv.scale_updates[j];
                conv.scale_updates[j] *= t.momentum;
            }
        }
    }

    pub fn zero_grads(&mut self) {
        for g in &mut self.grads {
            g.fill(0.0);
        }
        self.input_grad.fill(0.0);
    }

    /// Fold the rolling BN statistics into a [`Model`] for inference and evaluation
    /// (`darknet_network.cpp:2905-2940`, arithmetic in f64 like Darknet).
    pub fn to_model(&self) -> Result<Model> {
        let mut model = Model::from_cfg(&self.cfg)?;
        for (i, layer) in model.layers.iter_mut().enumerate() {
            let LayerKind::Conv(c) = &mut layer.kind else { continue };
            let state = self.convs[i].as_ref().expect("conv state");
            c.weights = state.weights.clone();
            c.bias = state.bias.clone();
            if c.cfg.batch_normalize {
                let bn = crate::model::Bn {
                    scales: state.scales.clone(),
                    mean: state.rolling_mean.clone(),
                    var: state.rolling_var.clone(),
                };
                let k = c.k();
                crate::weights::fold_bn(&mut c.weights, &mut c.bias, &bn, k);
            }
        }
        model.seen = 0;
        Ok(model)
    }

    /// Total number of conv parameters (weights + bias + BN scales).
    pub fn n_params(&self) -> usize {
        self.convs.iter().flatten().map(|c| c.weights.len() + c.bias.len() + c.scales.len()).sum()
    }
}

/// `darknet_network.cpp:589-596` + `711-738`: the learning rate for a 1-based batch number.
pub fn learning_rate(cfg: &TrainCfg, batch_num: usize) -> f32 {
    if cfg.burn_in > 0 && batch_num < cfg.burn_in {
        return cfg.learning_rate * (batch_num as f32 / cfg.burn_in as f32).powf(cfg.power);
    }
    match &cfg.policy {
        LrPolicy::Constant => cfg.learning_rate,
        LrPolicy::Steps { steps, scales } => {
            let mut rate = cfg.learning_rate;
            for (step, scale) in steps.iter().zip(scales) {
                if *step > batch_num {
                    return rate;
                }
                rate *= *scale;
            }
            rate
        }
    }
}

fn apply_activation(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Linear => {}
        Activation::Leaky => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= 0.1;
                }
            }
        }
        Activation::Logistic => {
            for v in buf.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
    }
}

fn activation_backward(buf: &mut [f32], post: &[f32], act: Activation) {
    match act {
        Activation::Linear => {}
        Activation::Leaky => {
            for (d, &o) in buf.iter_mut().zip(post) {
                if o < 0.0 {
                    *d *= 0.1;
                }
            }
        }
        Activation::Logistic => {
            for (d, &o) in buf.iter_mut().zip(post) {
                *d *= o * (1.0 - o);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn conv_forward(sum: &mut [f32], input: &[f32], w: &[f32], bias: &[f32], ins: Shape, outs: Shape, cfg: &ConvCfg, batch: usize) {
    let (k, s, p) = (cfg.size, cfg.stride as isize, cfg.padding as isize);
    let (iw, ih, ow, oh) = (ins.w as isize, ins.h as isize, outs.w, outs.h);
    let ins_plane = ins.h * ins.w;
    for b in 0..batch {
        let xb = &input[b * ins.len()..(b + 1) * ins.len()];
        let sb = &mut sum[b * outs.len()..(b + 1) * outs.len()];
        for oc in 0..outs.c {
            let plane = &mut sb[oc * oh * ow..(oc + 1) * oh * ow];
            plane.fill(if cfg.batch_normalize { 0.0 } else { bias[oc] });
            let wbase = oc * ins.c * k * k;
            for ic in 0..ins.c {
                let inp = &xb[ic * ins_plane..(ic + 1) * ins_plane];
                for ky in 0..k {
                    for kx in 0..k {
                        let wv = w[wbase + (ic * k + ky) * k + kx];
                        for oy in 0..oh {
                            let iy = oy as isize * s + ky as isize - p;
                            if iy < 0 || iy >= ih {
                                continue;
                            }
                            for ox in 0..ow {
                                let ix = ox as isize * s + kx as isize - p;
                                if ix < 0 || ix >= iw {
                                    continue;
                                }
                                plane[oy * ow + ox] += wv * inp[iy as usize * ins.w + ix as usize];
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn bn_forward(
    sum: &[f32],
    out: &mut [f32],
    norm: &mut [f32],
    mean: &mut [f32],
    var: &mut [f32],
    rolling_mean: &mut [f32],
    rolling_var: &mut [f32],
    scales: &[f32],
    bias: &[f32],
    batch: usize,
    shape: Shape,
) {
    let spatial = shape.h * shape.w;
    let n = batch * spatial;
    let scale_n = 1.0 / n as f32;
    let var_n = if n > 1 { 1.0 / (n - 1) as f32 } else { 0.0 };
    for c in 0..shape.c {
        let mut m = 0.0f32;
        for b in 0..batch {
            let base = (b * shape.c + c) * spatial;
            m += sum[base..base + spatial].iter().sum::<f32>();
        }
        m *= scale_n;
        let mut v = 0.0f32;
        for b in 0..batch {
            let base = (b * shape.c + c) * spatial;
            for &x in &sum[base..base + spatial] {
                v += (x - m) * (x - m);
            }
        }
        v *= var_n;
        mean[c] = m;
        var[c] = v;
        rolling_mean[c] = ROLLING_MOMENTUM * rolling_mean[c] + (1.0 - ROLLING_MOMENTUM) * m;
        rolling_var[c] = ROLLING_MOMENTUM * rolling_var[c] + (1.0 - ROLLING_MOMENTUM) * v;
        let inv = 1.0 / (v + BN_EPS).sqrt();
        for b in 0..batch {
            let base = (b * shape.c + c) * spatial;
            for j in 0..spatial {
                let xhat = (sum[base + j] - m) * inv;
                norm[base + j] = xhat;
                out[base + j] = xhat * scales[c] + bias[c];
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn conv_backward(
    delta: &mut [f32],
    prev_grad: &mut [f32],
    post: &[f32],
    prev_out: &[f32],
    outs: Shape,
    ins: Shape,
    cfg: &ConvCfg,
    conv: &mut ConvState,
    conv_sum: Option<&[f32]>,
    conv_norm: Option<&[f32]>,
    batch: usize,
) {
    let (k, s, p) = (cfg.size, cfg.stride as isize, cfg.padding as isize);
    let (iw, ih) = (ins.w as isize, ins.h as isize);
    let spatial = outs.h * outs.w;
    let ins_plane = ins.h * ins.w;

    // 1. Activation gradient (Darknet `gradient_array`, applied before BN).
    activation_backward(delta, post, cfg.activation);

    // 2. Batch-norm backward, or the bias gradient for a plain conv.
    if cfg.batch_normalize {
        let norm = conv_norm.expect("x_norm");
        let x = conv_sum.expect("conv sum");
        for c in 0..outs.c {
            let mut sdx = 0.0f32;
            for b in 0..batch {
                let base = (b * outs.c + c) * spatial;
                for j in 0..spatial {
                    sdx += norm[base + j] * delta[base + j];
                }
            }
            conv.scale_updates[c] += sdx;
        }
        for c in 0..outs.c {
            let mean = conv.mean[c];
            let var = conv.var[c];
            let mut sd = 0.0f32;
            for b in 0..batch {
                let base = (b * outs.c + c) * spatial;
                for j in 0..spatial {
                    delta[base + j] *= conv.scales[c];
                    sd += delta[base + j];
                }
            }
            let mean_delta = -sd / (var + BN_EPS).sqrt();
            let mut sum_dx = 0.0f32;
            for b in 0..batch {
                let base = (b * outs.c + c) * spatial;
                for j in 0..spatial {
                    sum_dx += delta[base + j] * (x[base + j] - mean);
                }
            }
            let variance_delta = -0.5 * sum_dx / (var + BN_EPS).powf(1.5);
            // Darknet divides by `sqrt(var) + eps`, not `sqrt(var + eps)` (`batchnorm_layer.cpp:62`).
            let inv_std = 1.0 / (var.sqrt() + BN_EPS);
            let n = (batch * spatial) as f32;
            for b in 0..batch {
                let base = (b * outs.c + c) * spatial;
                for j in 0..spatial {
                    delta[base + j] = delta[base + j] * inv_std
                        + variance_delta * 2.0 * (x[base + j] - mean) / n
                        + mean_delta / n;
                }
            }
        }
    } else {
        for c in 0..outs.c {
            let mut sd = 0.0f32;
            for b in 0..batch {
                let base = (b * outs.c + c) * spatial;
                sd += delta[base..base + spatial].iter().sum::<f32>();
            }
            conv.bias_updates[c] += sd;
        }
    }

    // 3. dW, accumulated over the batch (Darknet's `beta = 1` gemm).
    for b in 0..batch {
        let d = &delta[b * outs.len()..(b + 1) * outs.len()];
        let xb = &prev_out[b * ins.len()..(b + 1) * ins.len()];
        for oc in 0..outs.c {
            let dplane = &d[oc * spatial..(oc + 1) * spatial];
            let wbase = oc * ins.c * k * k;
            for ic in 0..ins.c {
                let inp = &xb[ic * ins_plane..(ic + 1) * ins_plane];
                for ky in 0..k {
                    for kx in 0..k {
                        let mut acc = 0.0f32;
                        for oy in 0..outs.h {
                            let iy = oy as isize * s + ky as isize - p;
                            if iy < 0 || iy >= ih {
                                continue;
                            }
                            for ox in 0..outs.w {
                                let ix = ox as isize * s + kx as isize - p;
                                if ix < 0 || ix >= iw {
                                    continue;
                                }
                                acc += dplane[oy * outs.w + ox] * inp[iy as usize * ins.w + ix as usize];
                            }
                        }
                        conv.weight_updates[wbase + (ic * k + ky) * k + kx] += acc;
                    }
                }
            }
        }
    }

    // 4. dX, accumulated into the input's gradient buffer.
    for b in 0..batch {
        let d = &delta[b * outs.len()..(b + 1) * outs.len()];
        let g = &mut prev_grad[b * ins.len()..(b + 1) * ins.len()];
        for oc in 0..outs.c {
            let dplane = &d[oc * spatial..(oc + 1) * spatial];
            let wbase = oc * ins.c * k * k;
            for ic in 0..ins.c {
                let gplane = &mut g[ic * ins_plane..(ic + 1) * ins_plane];
                for ky in 0..k {
                    for kx in 0..k {
                        let wv = conv.weights[wbase + (ic * k + ky) * k + kx];
                        for oy in 0..outs.h {
                            let iy = oy as isize * s + ky as isize - p;
                            if iy < 0 || iy >= ih {
                                continue;
                            }
                            for ox in 0..outs.w {
                                let ix = ox as isize * s + kx as isize - p;
                                if ix < 0 || ix >= iw {
                                    continue;
                                }
                                gplane[iy as usize * ins.w + ix as usize] += wv * dplane[oy * outs.w + ox];
                            }
                        }
                    }
                }
            }
        }
    }
}

fn maxpool_forward(input: &[f32], out: &mut [f32], idx: &mut [u32], ins: Shape, outs: Shape, m: &MaxpoolCfg, batch: usize) {
    let off = -((m.padding / 2) as isize);
    for b in 0..batch {
        let (ib, ob) = (b * ins.len(), b * outs.len());
        for c in 0..outs.c {
            for oy in 0..outs.h {
                for ox in 0..outs.w {
                    let mut best = -f32::MAX;
                    let mut best_i = 0u32;
                    for ky in 0..m.size {
                        for kx in 0..m.size {
                            let iy = off + (oy * m.stride + ky) as isize;
                            let ix = off + (ox * m.stride + kx) as isize;
                            if iy >= 0 && iy < ins.h as isize && ix >= 0 && ix < ins.w as isize {
                                let p = iy as usize * ins.w + ix as usize;
                                let v = input[ib + (c * ins.h + iy as usize) * ins.w + ix as usize];
                                if v > best {
                                    best = v;
                                    best_i = p as u32;
                                }
                            }
                        }
                    }
                    out[ob + (c * outs.h + oy) * outs.w + ox] = if best == -f32::MAX { 0.0 } else { best };
                    idx[ob + (c * outs.h + oy) * outs.w + ox] = best_i;
                }
            }
        }
    }
}

fn maxpool_backward(delta: &[f32], prev_grad: &mut [f32], idx: &[u32], ins: Shape, m: &MaxpoolCfg, batch: usize) {
    let outs = Shape {
        c: ins.c,
        h: (ins.h + m.padding - m.size) / m.stride + 1,
        w: (ins.w + m.padding - m.size) / m.stride + 1,
    };
    for b in 0..batch {
        let (ib, ob) = (b * ins.len(), b * outs.len());
        for c in 0..outs.c {
            for oy in 0..outs.h {
                for ox in 0..outs.w {
                    let d = delta[ob + (c * outs.h + oy) * outs.w + ox];
                    if d == 0.0 {
                        continue;
                    }
                    let p = idx[ob + (c * outs.h + oy) * outs.w + ox] as usize;
                    prev_grad[ib + c * ins.h * ins.w + p] += d;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn route_forward(
    outs: &[Vec<f32>],
    dst: &mut [f32],
    sources: &[usize],
    groups: usize,
    group_id: usize,
    model: &Model,
    batch: usize,
    shape: Shape,
) {
    let spatial = shape.h * shape.w;
    let mut dst_c = 0usize;
    for &s in sources {
        let src_shape = model.layers[s].out_shape;
        let block = src_shape.c / groups;
        for b in 0..batch {
            for c in 0..block {
                let src_off = (b * src_shape.c + group_id * block + c) * spatial;
                let dst_off = (b * shape.c + dst_c + c) * spatial;
                dst[dst_off..dst_off + spatial].copy_from_slice(&outs[s][src_off..src_off + spatial]);
            }
        }
        dst_c += block;
    }
}

#[allow(clippy::too_many_arguments)]
fn route_backward(
    delta: &[f32],
    grads: &mut [Vec<f32>],
    sources: &[usize],
    groups: usize,
    group_id: usize,
    model: &Model,
    batch: usize,
    shape: Shape,
) {
    let spatial = shape.h * shape.w;
    let mut src_off_c = 0usize;
    for &s in sources {
        let src_shape = model.layers[s].out_shape;
        let block = src_shape.c / groups;
        for b in 0..batch {
            for c in 0..block {
                let src_off = (b * src_shape.c + group_id * block + c) * spatial;
                let dst_off = (b * shape.c + src_off_c + c) * spatial;
                for j in 0..spatial {
                    grads[s][src_off + j] += delta[dst_off + j];
                }
            }
        }
        src_off_c += block;
    }
}

fn upsample_forward(input: &[f32], out: &mut [f32], ins: Shape, outs: Shape, stride: usize, batch: usize) {
    let ins_plane = ins.h * ins.w;
    let outs_plane = outs.h * outs.w;
    for b in 0..batch {
        for c in 0..ins.c {
            for y in 0..outs.h {
                for x in 0..outs.w {
                    out[(b * outs.c + c) * outs_plane + y * outs.w + x] =
                        input[(b * ins.c + c) * ins_plane + (y / stride) * ins.w + x / stride];
                }
            }
        }
    }
}

fn upsample_backward(delta: &[f32], prev_grad: &mut [f32], ins: Shape, stride: usize, batch: usize) {
    let outs = Shape { c: ins.c, h: ins.h * stride, w: ins.w * stride };
    let ins_plane = ins.h * ins.w;
    let outs_plane = outs.h * outs.w;
    for b in 0..batch {
        for c in 0..ins.c {
            for y in 0..outs.h {
                for x in 0..outs.w {
                    prev_grad[(b * ins.c + c) * ins_plane + (y / stride) * ins.w + x / stride] +=
                        delta[(b * outs.c + c) * outs_plane + y * outs.w + x];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn yolo_forward(input: &[f32], out: &mut [f32], shape: Shape, y: &YoloCfg, batch: usize) {
    let plane = shape.w * shape.h;
    let s = y.scale_x_y;
    let plain = shape.len();
    for b in 0..batch {
        for n in 0..y.mask.len() {
            let entry = n * (y.classes + 5) * plane;
            for e in 0..(y.classes + 5) {
                for v in 0..plane {
                    let i = b * plain + entry + e * plane + v;
                    let raw = input[i];
                    let val = if e < 2 {
                        let sig = 1.0 / (1.0 + (-raw).exp());
                        sig * s - 0.5 * (s - 1.0)
                    } else if e < 4 {
                        raw
                    } else {
                        1.0 / (1.0 + (-raw).exp())
                    };
                    out[i] = if val.is_finite() { val } else { 0.0 };
                }
            }
        }
    }
}
