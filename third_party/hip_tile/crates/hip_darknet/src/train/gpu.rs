//! The GPU training network: device-resident activations, weights and batch-norm state, with the
//! training-mode forward built from the stage-03 kernels. The CPU [`TrainNet`] is the reference:
//! a [`GpuTrainNet`] is created *from* one, so both start from identical weights and the per-layer
//! outputs can be compared one for one.
//!
//! Layout is NCHW with a batch: image `b` of a layer sits at `b * C*H*W` (`ROADMAP.md` decisions:
//! f32 end to end). Batch handling is per-image launches over offset views
//! (`kernels::yolo::batched`), so the batch-1 inference kernels are untouched.
//!
//! A batch-normalised conv is conv (linear, zero bias) → `bn_stats` → rolling update → `bn_fwd`
//! (normalise, scale/shift, leaky); a plain conv is the fused bias+activation conv.

use hip_rt::{Device, DeviceBuffer, Module, Stream};
use hip_tile::{Tensor, TensorMut};
use kernels::backward::act_bwd::launch_leaky_bwd;
use kernels::backward::batched::{launch_maxpool_bwd_batched, launch_route_bwd_batched, launch_upsample_bwd_batched};
use kernels::backward::bias_bwd::{BiasBwdParams, launch_bias_bwd};
use kernels::backward::bn_bwd::{BnBwdParams, launch_bn_bwd};
use kernels::backward::bn_fwd::{BnFwdParams, launch_bn_fwd};
use kernels::backward::conv_bwd_dw::launch_conv_bwd_dw;
use kernels::backward::conv_bwd_dx::launch_conv_bwd_dx;
use kernels::backward::ConvBwdParams;
use kernels::backward::bn_stats::{BnStatsParams, launch_bn_stats};
use kernels::backward::util::{SgdParams, launch_acc, launch_running_update, launch_sgd_update, launch_zero};
use kernels::backward::yolo_act::{YoloActParams, launch_yolo_act};
use kernels::backward::yolo_delta::{
    IOU_LOSS_CIOU, IOU_LOSS_DIOU, IOU_LOSS_GIOU, IOU_LOSS_IOU, MAX_MASK, YoloDeltaParams, launch_yolo_delta,
};
use kernels::yolo::batched::{launch_concat_batched, launch_conv_batched, launch_maxpool_batched, launch_upsample_batched};
use kernels::yolo::conv::ConvParams;
use kernels::yolo::maxpool::PoolParams;

use crate::cfg::{Activation, IouLoss, TrainCfg};
use crate::data::BoxLabel;
use crate::error::{Error, Result};
use crate::gpu::{KERNELS_ELF, hip, upload};
use crate::model::{LayerKind, Model};
use crate::train::deltas::{HeadStats, delta_sq, head_cost_from_sums};
use crate::train::net::{ROLLING_MOMENTUM, TrainNet};

/// Ground-truth boxes per image the device delta can take (Darknet's default `max_boxes`).
pub const MAX_TRUTHS: usize = 200;

/// One conv layer's device state.
struct GpuConv {
    weights: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    /// All zeros: the bias of the pre-normalisation conv of a BN layer.
    zero_bias: DeviceBuffer<f32>,
    scales: DeviceBuffer<f32>,
    mean: DeviceBuffer<f32>,
    var: DeviceBuffer<f32>,
    rolling_mean: DeviceBuffer<f32>,
    rolling_var: DeviceBuffer<f32>,
    /// The conv sum before normalisation, `[batch x C x H x W]` (BN layers only).
    sum: Option<DeviceBuffer<f32>>,
    /// Darknet's `x_norm`, kept for the backward pass (BN layers only).
    xhat: Option<DeviceBuffer<f32>>,
    /// Gradient accumulators that double as the momentum buffers (Darknet's `*_updates`).
    weight_updates: DeviceBuffer<f32>,
    bias_updates: DeviceBuffer<f32>,
    scale_updates: DeviceBuffer<f32>,
}

/// One `[yolo]` head's device state for the loss.
struct GpuHead {
    anchors: DeviceBuffer<f32>,
    /// This head's delta, `[batch x plain]`, rebuilt every iteration.
    delta: DeviceBuffer<f32>,
    /// `[batch*MAX_TRUTHS x 6]` as `(class, x, y, w, h, 0)`.
    truths: DeviceBuffer<f32>,
    counts: DeviceBuffer<f32>,
    /// `[batch*MAX_TRUTHS x 4]` per-truth statistics written by the delta kernel.
    stats: DeviceBuffer<f32>,
}

pub struct GpuTrainNet {
    model: Model,
    batch: usize,
    _device: Device,
    module: Module,
    stream: Stream,
    input: DeviceBuffer<f32>,
    /// Per layer activation; a `[yolo]` layer owns its (activated) output.
    outs: Vec<DeviceBuffer<f32>>,
    convs: Vec<Option<GpuConv>>,
    train: TrainCfg,
    /// Per layer gradient w.r.t. `outs[i]`.
    grads: Vec<DeviceBuffer<f32>>,
    /// Aligned with `model.layers`; `Some` only for `[yolo]` layers.
    heads: Vec<Option<GpuHead>>,
    /// Last loss returned by [`GpuTrainNet::loss_and_deltas`].
    pub loss: f32,
}

impl GpuTrainNet {
    /// Copy `net`'s weights and batch-norm state to the device and allocate the activations.
    pub fn from_net(net: &TrainNet) -> Result<Self> {
        let model = net.model.clone();
        let batch = net.batch;
        let device = hip(Device::init())?;
        let module = hip(Module::from_elf(KERNELS_ELF))?;
        let stream = hip(Stream::new())?;
        let input = hip(DeviceBuffer::zeroed(batch * model.input_shape().len()))?;
        let mut outs = Vec::with_capacity(model.layers.len());
        let mut convs = Vec::with_capacity(model.layers.len());
        let mut grads = Vec::with_capacity(model.layers.len());
        let mut heads = Vec::with_capacity(model.layers.len());
        for (i, layer) in model.layers.iter().enumerate() {
            outs.push(hip(DeviceBuffer::zeroed(batch * layer.out_shape.len()))?);
            grads.push(hip(DeviceBuffer::zeroed(batch * layer.out_shape.len()))?);
            heads.push(match &layer.kind {
                LayerKind::Yolo(y) => {
                    if y.train.iou_loss == IouLoss::Mse {
                        // The device MSE branch disagrees with the host on a few entries (stage 03).
                        return Err(Error::Unsupported(format!("layer {i}: GPU [yolo] iou_loss=mse")));
                    }
                    if y.mask.len() > MAX_MASK {
                        return Err(Error::Unsupported(format!("layer {i}: mask longer than {MAX_MASK}")));
                    }
                    Some(GpuHead {
                        anchors: hip(DeviceBuffer::from_slice(&y.anchors))?,
                        delta: hip(DeviceBuffer::zeroed(batch * layer.out_shape.len()))?,
                        truths: hip(DeviceBuffer::zeroed(batch * MAX_TRUTHS * 6))?,
                        counts: hip(DeviceBuffer::zeroed(batch))?,
                        stats: hip(DeviceBuffer::zeroed(batch * MAX_TRUTHS * 4))?,
                    })
                }
                _ => None,
            });
            match (&layer.kind, &net.convs[i]) {
                (LayerKind::Conv(c), Some(st)) => {
                    if !matches!(c.cfg.activation, Activation::Linear | Activation::Leaky) {
                        return Err(Error::Unsupported(format!("layer {i}: GPU conv activation {:?}", c.cfg.activation)));
                    }
                    let n = batch * layer.out_shape.len();
                    let bn = c.cfg.batch_normalize;
                    convs.push(Some(GpuConv {
                        weights: hip(DeviceBuffer::from_slice(&st.weights))?,
                        bias: hip(DeviceBuffer::from_slice(&st.bias))?,
                        zero_bias: hip(DeviceBuffer::zeroed(st.bias.len()))?,
                        scales: hip(DeviceBuffer::from_slice(&st.scales))?,
                        mean: hip(DeviceBuffer::from_slice(&st.mean))?,
                        var: hip(DeviceBuffer::from_slice(&st.var))?,
                        rolling_mean: hip(DeviceBuffer::from_slice(&st.rolling_mean))?,
                        rolling_var: hip(DeviceBuffer::from_slice(&st.rolling_var))?,
                        sum: if bn { Some(hip(DeviceBuffer::zeroed(n))?) } else { None },
                        xhat: if bn { Some(hip(DeviceBuffer::zeroed(n))?) } else { None },
                        weight_updates: hip(DeviceBuffer::from_slice(&st.weight_updates))?,
                        bias_updates: hip(DeviceBuffer::from_slice(&st.bias_updates))?,
                        scale_updates: hip(DeviceBuffer::from_slice(&st.scale_updates))?,
                    }));
                }
                _ => convs.push(None),
            }
        }
        Ok(GpuTrainNet { model, batch, _device: device, module, stream, input, outs, convs, train: net.cfg.train.clone(), grads, heads, loss: 0.0 })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Output pointer of layer `i`'s source: the previous layer, or the network input.
    fn prev_ptr(&self, i: usize) -> *const f32 {
        if i == 0 { self.input.as_ptr() } else { self.outs[i - 1].as_ptr() }
    }

    /// Training-mode forward over the whole batch: upload `chw` (NCHW with the batch), queue every
    /// layer on the stream, synchronise once.
    pub fn forward(&mut self, chw: &[f32]) -> Result<()> {
        if chw.len() != self.input.len() {
            return Err(Error::Model(format!("input has {} floats, expected {}", chw.len(), self.input.len())));
        }
        hip(upload(&self.input, chw))?;
        let (md, st, batch) = (&self.module, Some(&self.stream), self.batch as u32);
        for (i, l) in self.model.layers.iter().enumerate() {
            let prev = self.prev_ptr(i);
            let (ins, outs) = (l.in_shape, l.out_shape);
            let (ohw, ihw) = ((outs.h * outs.w) as u32, (ins.h * ins.w) as u32);
            let out_ptr = self.outs[i].as_ptr().cast_mut();
            let y = TensorMut::new(out_ptr, outs.c as u32, ohw, ohw);
            match &l.kind {
                LayerKind::Conv(c) => {
                    let g = self.convs[i].as_ref().expect("conv state");
                    let k = c.k() as u32;
                    let w = Tensor::new(g.weights.as_ptr(), c.cfg.filters as u32, k, k);
                    let x = Tensor::new(prev, ins.c as u32, ihw, ihw);
                    let leaky = c.cfg.activation == Activation::Leaky;
                    let cp = |act| {
                        ConvParams::new(ins.c as u32, ins.h as u32, ins.w as u32, c.cfg.size as u32, c.cfg.stride as u32, c.cfg.padding as u32, act)
                    };
                    if c.cfg.batch_normalize {
                        let (sum, xhat) = (g.sum.as_ref().expect("bn sum"), g.xhat.as_ref().expect("bn xhat"));
                        let ch = outs.c as u32;
                        let zb = Tensor::vector(g.zero_bias.as_ptr(), ch);
                        let sum_y = TensorMut::new(sum.as_ptr().cast_mut(), ch, ohw, ohw);
                        hip(launch_conv_batched(md, st, batch, x, w, zb, sum_y, &cp(ConvParams::LINEAR)))?;
                        // Channel `c` of every image: `[C x spatial]` rows, image stride `C*spatial`.
                        let sv = Tensor::new(sum.as_ptr(), ch, ohw, ch * ohw);
                        let (mean, var) = (g.mean.as_ptr().cast_mut(), g.var.as_ptr().cast_mut());
                        hip(launch_bn_stats(md, st, sv, TensorMut::new(mean, 1, ch, ch), TensorMut::new(var, 1, ch, ch), &BnStatsParams { batch, spatial: ohw }))?;
                        hip(launch_running_update(md, st, TensorMut::new(g.rolling_mean.as_ptr().cast_mut(), 1, ch, ch), Tensor::new(g.mean.as_ptr(), 1, ch, ch), ROLLING_MOMENTUM))?;
                        hip(launch_running_update(md, st, TensorMut::new(g.rolling_var.as_ptr().cast_mut(), 1, ch, ch), Tensor::new(g.var.as_ptr(), 1, ch, ch), ROLLING_MOMENTUM))?;
                        let flat = |p: *const f32| Tensor::vector(p, batch * ch * ohw);
                        let vec_c = |b: &DeviceBuffer<f32>| Tensor::vector(b.as_ptr(), ch);
                        let act = if leaky { ConvParams::LEAKY } else { ConvParams::LINEAR };
                        hip(launch_bn_fwd(
                            md, st, flat(sum.as_ptr()), vec_c(&g.mean), vec_c(&g.var), vec_c(&g.scales), vec_c(&g.bias),
                            TensorMut::vector(xhat.as_ptr().cast_mut(), batch * ch * ohw),
                            TensorMut::vector(out_ptr, batch * ch * ohw),
                            &BnFwdParams { channels: ch, spatial: ohw, act },
                        ))?;
                    } else {
                        let bias = Tensor::vector(g.bias.as_ptr(), c.cfg.filters as u32);
                        let act = if leaky { ConvParams::LEAKY } else { ConvParams::LINEAR };
                        hip(launch_conv_batched(md, st, batch, x, w, bias, y, &cp(act)))?;
                    }
                }
                LayerKind::Maxpool(m) => {
                    let p = PoolParams::new(ins.h as u32, ins.w as u32, m.size as u32, m.stride as u32, Some(m.padding as u32));
                    hip(launch_maxpool_batched(md, st, batch, Tensor::new(prev, ins.c as u32, ihw, ihw), y, &p))?;
                }
                LayerKind::Upsample { stride } => {
                    hip(launch_upsample_batched(md, st, batch, Tensor::new(prev, ins.c as u32, ihw, ihw), y, ins.w as u32, *stride as u32))?;
                }
                LayerKind::Route { sources, groups, group_id } => {
                    let mut ch0 = 0usize;
                    for &s in sources {
                        let src_c = self.model.layers[s].out_shape.c;
                        let c = src_c / groups;
                        let src = unsafe { self.outs[s].as_ptr().add(group_id * c * ohw as usize) };
                        let dst = TensorMut::new(unsafe { out_ptr.add(ch0 * ohw as usize) }, c as u32, ohw, ohw);
                        hip(launch_concat_batched(md, st, batch, Tensor::new(src, c as u32, ohw, ohw), src_c * ohw as usize, dst, outs.c * ohw as usize))?;
                        ch0 += c;
                    }
                }
                LayerKind::Yolo(y_cfg) => {
                    let plain = outs.len() as u32;
                    let p = YoloActParams {
                        plain,
                        plane: ohw,
                        entries: (y_cfg.classes + 5) as u32,
                        scale_x_y: y_cfg.scale_x_y,
                    };
                    hip(launch_yolo_act(md, st, Tensor::vector(prev, batch * plain), TensorMut::vector(out_ptr, batch * plain), &p))?;
                }
            }
        }
        hip(self.stream.synchronize())
    }

    /// Zero every layer gradient; call once per iteration before the loss.
    pub fn zero_grads(&mut self) -> Result<()> {
        for g in &self.grads {
            let n = g.len() as u32;
            hip(launch_zero(&self.module, Some(&self.stream), TensorMut::vector(g.as_ptr().cast_mut(), n)))?;
        }
        hip(self.stream.synchronize())
    }

    /// The `[yolo]` deltas for one batch, added into the previous layers' gradients, and the
    /// network loss (the mean of the per-head costs, as [`TrainNet::loss_and_deltas`]). The
    /// per-head delta and per-truth statistics come back to the host to form the reported cost;
    /// the gradient itself never leaves the device.
    pub fn loss_and_deltas(&mut self, truths: &[Vec<BoxLabel>]) -> Result<f32> {
        if truths.len() != self.batch {
            return Err(Error::Model(format!("{} truth sets for a batch of {}", truths.len(), self.batch)));
        }
        if let Some(n) = truths.iter().map(Vec::len).find(|&n| n > MAX_TRUTHS) {
            return Err(Error::Model(format!("{n} ground-truth boxes in one image, the device takes {MAX_TRUTHS}")));
        }
        let heads = self.model.yolo_layers();
        if heads.is_empty() {
            return Err(Error::Model("cfg has no [yolo] layer".into()));
        }
        let mut rows = vec![0f32; self.batch * MAX_TRUTHS * 6];
        let mut counts = vec![0f32; self.batch];
        for (b, ls) in truths.iter().enumerate() {
            counts[b] = ls.len() as f32;
            for (t, l) in ls.iter().enumerate() {
                rows[(b * MAX_TRUTHS + t) * 6..][..5].copy_from_slice(&[l.class as f32, l.x, l.y, l.w, l.h]);
            }
        }
        let (md, st) = (&self.module, Some(&self.stream));
        let (net_w, net_h) = (self.model.net.width as u32, self.model.net.height as u32);
        let mut total = 0.0f32;
        for &i in &heads {
            let LayerKind::Yolo(y) = &self.model.layers[i].kind else { unreachable!() };
            let shape = self.model.layers[i].out_shape;
            let plain = shape.len();
            let head = self.heads[i].as_ref().expect("yolo head");
            hip(upload(&head.truths, &rows))?;
            hip(upload(&head.counts, &counts))?;
            let t = &y.train;
            let mut p = YoloDeltaParams {
                classes: y.classes as u32,
                num: (y.anchors.len() / 2) as u32,
                n_mask: y.mask.len() as u32,
                mask: [0; MAX_MASK],
                scale_x_y: y.scale_x_y,
                iou_normalizer: t.iou_normalizer,
                obj_normalizer: t.obj_normalizer,
                cls_normalizer: t.cls_normalizer,
                ignore_thresh: t.ignore_thresh,
                truth_thresh: t.truth_thresh,
                max_delta: t.max_delta,
                iou_loss: match t.iou_loss {
                    IouLoss::Iou => IOU_LOSS_IOU,
                    IouLoss::Giou => IOU_LOSS_GIOU,
                    IouLoss::Diou => IOU_LOSS_DIOU,
                    IouLoss::Ciou => IOU_LOSS_CIOU,
                    IouLoss::Mse => unreachable!("rejected in from_net"),
                },
                lw: shape.w as u32,
                lh: shape.h as u32,
                net_w,
                net_h,
                batch: self.batch as u32,
                max_truths: MAX_TRUTHS as u32,
            };
            for (k, m) in y.mask.iter().enumerate() {
                p.mask[k] = *m as u32;
            }
            let planes = (y.mask.len() * (y.classes + 5)) as u32;
            let plane = (shape.w * shape.h) as u32;
            let n = (self.batch * plain) as u32;
            hip(launch_zero(md, st, TensorMut::vector(head.delta.as_ptr().cast_mut(), n)))?;
            hip(launch_yolo_delta(
                md,
                st,
                Tensor::new(self.outs[i].as_ptr(), planes, plane, plane),
                Tensor::vector(head.anchors.as_ptr(), y.anchors.len() as u32),
                Tensor::new(head.truths.as_ptr(), (self.batch * MAX_TRUTHS) as u32, 6, 6),
                Tensor::vector(head.counts.as_ptr(), self.batch as u32),
                TensorMut::new(head.delta.as_ptr().cast_mut(), planes, plane, plane),
                TensorMut::new(head.stats.as_ptr().cast_mut(), (self.batch * MAX_TRUTHS) as u32, 4, 4),
                &p,
            ))?;
            // Darknet adds the head's delta into the previous layer's gradient.
            hip(launch_acc(md, st, Tensor::vector(head.delta.as_ptr(), n), TensorMut::vector(self.grads[i - 1].as_ptr().cast_mut(), n)))?;
            hip(self.stream.synchronize())?;

            let delta = hip(head.delta.copy_to_vec())?;
            let per_truth = hip(head.stats.copy_to_vec())?;
            let mut stats = HeadStats::default();
            for r in per_truth.as_chunks::<4>().0 {
                stats.count += r[0] as usize;
                stats.tot_iou += r[1];
                stats.tot_iou_loss += r[2];
                stats.tot_giou_loss += r[3];
            }
            let (mut sq_all, mut sq_nobox) = (0.0f32, 0.0f32);
            for img in delta.chunks_exact(plain) {
                sq_all += delta_sq(img, shape, y, false);
                sq_nobox += delta_sq(img, shape, y, true);
            }
            total += head_cost_from_sums(y, &stats, sq_all, sq_nobox);
        }
        self.loss = total / heads.len() as f32;
        Ok(self.loss)
    }

    /// Overwrite the device forward state (input, every activation, the BN sums, `x_norm` and
    /// batch statistics) with `net`'s after its [`TrainNet::forward`]. The backward pass then runs
    /// on exactly the CPU's forward values, which isolates it in parity tests: two f32 forwards
    /// differ by rounding, and a rounding-sized difference at a leaky-ReLU kink flips a gradient
    /// factor between 1 and 0.1.
    pub fn load_activations(&mut self, net: &TrainNet) -> Result<()> {
        hip(upload(&self.input, &net.input))?;
        for (i, out) in self.outs.iter().enumerate() {
            hip(upload(out, &net.outs[i]))?;
        }
        for (i, g) in self.convs.iter().enumerate() {
            let (Some(g), Some(st)) = (g, &net.convs[i]) else { continue };
            hip(upload(&g.mean, &st.mean))?;
            hip(upload(&g.var, &st.var))?;
            if let (Some(sum), Some(xhat)) = (&g.sum, &g.xhat) {
                hip(upload(sum, net.conv_sum[i].as_deref().expect("conv sum")))?;
                hip(upload(xhat, net.conv_norm[i].as_deref().expect("x_norm")))?;
            }
        }
        Ok(())
    }

    /// Backward through every layer, accumulating parameter updates (`*_updates`) and the
    /// gradients of every layer's output. Call after [`GpuTrainNet::loss_and_deltas`]; the `[yolo]`
    /// layers are skipped because their delta already sits in the previous layer's gradient. The
    /// network input's gradient is not computed.
    pub fn backward(&mut self) -> Result<()> {
        self.backward_until(0)
    }

    /// [`GpuTrainNet::backward`] stopping after layer `first` (layers `first..` are processed),
    /// to inspect the gradients mid-network.
    pub fn backward_until(&mut self, first: usize) -> Result<()> {
        let (md, st, batch) = (&self.module, Some(&self.stream), self.batch as u32);
        for i in (first..self.model.layers.len()).rev() {
            let l = &self.model.layers[i];
            let (ins, outs) = (l.in_shape, l.out_shape);
            let (ohw, ihw) = ((outs.h * outs.w) as u32, (ins.h * ins.w) as u32);
            let prev: *const f32 = if i == 0 { self.input.as_ptr() } else { self.outs[i - 1].as_ptr() };
            let delta = self.grads[i].as_ptr().cast_mut();
            match &l.kind {
                LayerKind::Yolo(_) => {}
                LayerKind::Conv(c) => {
                    let g = self.convs[i].as_ref().expect("conv state");
                    let (ch, kdim) = (outs.c as u32, c.k() as u32);
                    let flat = batch * ch * ohw;
                    if c.cfg.activation == Activation::Leaky {
                        hip(launch_leaky_bwd(md, st, TensorMut::vector(delta, flat), Tensor::vector(self.outs[i].as_ptr(), flat)))?;
                    }
                    // The channel view of the whole batch: `[C x spatial]` rows, image stride `C*spatial`.
                    let dview = TensorMut::new(delta, ch, ohw, ch * ohw);
                    if c.cfg.batch_normalize {
                        let (sum, xhat) = (g.sum.as_ref().expect("bn sum"), g.xhat.as_ref().expect("bn xhat"));
                        let vec_c = |b: &DeviceBuffer<f32>| Tensor::vector(b.as_ptr(), ch);
                        hip(launch_bn_bwd(
                            md, st,
                            Tensor::new(sum.as_ptr(), ch, ohw, ch * ohw),
                            Tensor::new(xhat.as_ptr(), ch, ohw, ch * ohw),
                            dview, vec_c(&g.mean), vec_c(&g.var), vec_c(&g.scales),
                            TensorMut::vector(g.scale_updates.as_ptr().cast_mut(), ch),
                            &BnBwdParams { batch, spatial: ohw },
                        ))?;
                    } else {
                        hip(launch_bias_bwd(
                            md, st,
                            Tensor::new(delta, ch, ohw, ch * ohw),
                            TensorMut::vector(g.bias_updates.as_ptr().cast_mut(), ch),
                            &BiasBwdParams { batch, spatial: ohw },
                        ))?;
                    }
                    let conv = ConvParams::with_cout(
                        ins.c as u32, ch, ins.h as u32, ins.w as u32, c.cfg.size as u32, c.cfg.stride as u32, c.cfg.padding as u32,
                        ConvParams::LINEAR,
                    );
                    let p = ConvBwdParams::new(conv, batch);
                    let d_img = Tensor::new(delta, ch, ohw, ohw);
                    hip(launch_conv_bwd_dw(
                        md, st, d_img, Tensor::new(prev, ins.c as u32, ihw, ihw),
                        TensorMut::new(g.weight_updates.as_ptr().cast_mut(), ch, kdim, kdim), &p,
                    ))?;
                    if i > 0 {
                        hip(launch_conv_bwd_dx(
                            md, st, d_img, Tensor::new(g.weights.as_ptr(), ch, kdim, kdim),
                            TensorMut::new(self.grads[i - 1].as_ptr().cast_mut(), ins.c as u32, ihw, ihw), &p,
                        ))?;
                    }
                }
                LayerKind::Maxpool(m) => {
                    if i > 0 {
                        let p = PoolParams::new(ins.h as u32, ins.w as u32, m.size as u32, m.stride as u32, Some(m.padding as u32));
                        hip(launch_maxpool_bwd_batched(
                            md, st, batch,
                            Tensor::new(prev, ins.c as u32, ihw, ihw),
                            Tensor::new(delta, outs.c as u32, ohw, ohw),
                            TensorMut::new(self.grads[i - 1].as_ptr().cast_mut(), ins.c as u32, ihw, ihw),
                            &p,
                        ))?;
                    }
                }
                LayerKind::Upsample { stride } => {
                    if i > 0 {
                        hip(launch_upsample_bwd_batched(
                            md, st, batch,
                            Tensor::new(delta, outs.c as u32, ohw, ohw),
                            TensorMut::new(self.grads[i - 1].as_ptr().cast_mut(), ins.c as u32, ihw, ihw),
                            ins.w as u32, *stride as u32,
                        ))?;
                    }
                }
                LayerKind::Route { sources, groups, group_id } => {
                    let mut ch0 = 0usize;
                    for &s in sources {
                        let src_c = self.model.layers[s].out_shape.c;
                        let c = src_c / groups;
                        let src = unsafe { delta.add(ch0 * ohw as usize) };
                        let dst = unsafe { self.grads[s].as_ptr().cast_mut().add(group_id * c * ohw as usize) };
                        hip(launch_route_bwd_batched(
                            md, st, batch,
                            Tensor::new(src, c as u32, ohw, ohw), outs.c * ohw as usize,
                            TensorMut::new(dst, c as u32, ohw, ohw), src_c * ohw as usize,
                        ))?;
                        ch0 += c;
                    }
                }
            }
        }
        hip(self.stream.synchronize())
    }

    /// Darknet's SGD step on every conv layer (`convolutional_layer.cpp:2983-3005`), on the device.
    pub fn update(&mut self, lr: f32) -> Result<()> {
        let t = &self.train;
        let batch = t.effective_batch() as f32;
        let (md, st) = (&self.module, Some(&self.stream));
        let with_decay = SgdParams { learning_rate: lr, momentum: t.momentum, decay: t.decay, batch };
        // Bias and BN scales are never decayed.
        let no_decay = SgdParams { decay: 0.0, ..with_decay };
        for g in self.convs.iter().flatten() {
            let vec = |b: &DeviceBuffer<f32>| TensorMut::vector(b.as_ptr().cast_mut(), b.len() as u32);
            hip(launch_sgd_update(md, st, vec(&g.weights), vec(&g.weight_updates), &with_decay))?;
            hip(launch_sgd_update(md, st, vec(&g.bias), vec(&g.bias_updates), &no_decay))?;
            hip(launch_sgd_update(md, st, vec(&g.scales), vec(&g.scale_updates), &no_decay))?;
        }
        hip(self.stream.synchronize())
    }

    /// Copy the device parameters, momentum buffers and batch-norm state back into `net`
    /// (parity checks, and later checkpoints).
    pub fn download_into(&self, net: &mut TrainNet) -> Result<()> {
        for (g, st) in self.convs.iter().zip(net.convs.iter_mut()) {
            let (Some(g), Some(st)) = (g, st) else { continue };
            st.weights = hip(g.weights.copy_to_vec())?;
            st.bias = hip(g.bias.copy_to_vec())?;
            st.scales = hip(g.scales.copy_to_vec())?;
            st.mean = hip(g.mean.copy_to_vec())?;
            st.var = hip(g.var.copy_to_vec())?;
            st.rolling_mean = hip(g.rolling_mean.copy_to_vec())?;
            st.rolling_var = hip(g.rolling_var.copy_to_vec())?;
            st.weight_updates = hip(g.weight_updates.copy_to_vec())?;
            st.bias_updates = hip(g.bias_updates.copy_to_vec())?;
            st.scale_updates = hip(g.scale_updates.copy_to_vec())?;
        }
        Ok(())
    }

    /// Download layer `i`'s gradient w.r.t. its output (`[batch x C x H x W]`), for parity checks.
    pub fn layer_grad(&self, i: usize) -> Result<Vec<f32>> {
        hip(self.grads[i].copy_to_vec())
    }

    /// Download layer `i`'s output (`[batch x C x H x W]`), for parity checks.
    pub fn layer_out(&self, i: usize) -> Result<Vec<f32>> {
        hip(self.outs[i].copy_to_vec())
    }

    /// Download a conv layer's batch statistics and rolling statistics:
    /// `(mean, var, rolling_mean, rolling_var)`.
    pub fn bn_state(&self, i: usize) -> Result<Option<[Vec<f32>; 4]>> {
        let Some(g) = &self.convs[i] else { return Ok(None) };
        Ok(Some([
            hip(g.mean.copy_to_vec())?,
            hip(g.var.copy_to_vec())?,
            hip(g.rolling_mean.copy_to_vec())?,
            hip(g.rolling_var.copy_to_vec())?,
        ]))
    }
}
