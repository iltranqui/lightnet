//! Host loss for dense-truth heads (tasks/rust/layers.md Task 10).

use crate::error::{Error, Result};
use crate::model::{LayerKind, Network};
use crate::runtime::{ForwardResult, Tensor};

/// C++ `SECRET_NUM`: a truth value that masks the element out of a `masked` cost layer.
pub const SECRET_NUM: f32 = -1234.0;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseTrainingLoss {
    /// Mean of the per-layer `cost` over the loss layers that had truth (C++ `get_network_cost`).
    pub loss: f32,
    /// Descent-direction seeds, `Some` for every loss layer that had truth. Feed them to
    /// `backward_accumulate_seeded`; the loss layers' backward arms pass them through.
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
}

pub fn compute_dense_training_loss(
    network: &Network,
    forward: &ForwardResult,
    input: &Tensor,
    truth: &[Option<Tensor>],
) -> Result<DenseTrainingLoss> {
    compute_dense_training_loss_at(network, forward, input, truth, 0)
}

/// [`compute_dense_training_loss`] with the training iteration, which the
/// `[segmentation]` correct-pixel gate needs (`gate_strength` is a raised cosine
/// over `gate_ramp_iters` after `gate_warmup_iters`).  Iteration 0 means "before
/// the warmup", i.e. the gate contributes nothing.
pub fn compute_dense_training_loss_at(
    network: &Network,
    forward: &ForwardResult,
    input: &Tensor,
    truth: &[Option<Tensor>],
    iteration: usize,
) -> Result<DenseTrainingLoss> {
    if truth.len() != network.layers.len() {
        return Err(Error::InvalidConfig(format!(
            "dense truth count mismatch: expected {}, got {}",
            network.layers.len(),
            truth.len()
        )));
    }
    let mut seeds = vec![None; network.layers.len()];
    let (mut total, mut count) = (0.0f32, 0usize);
    for layer in &network.layers {
        let Some(t) = truth[layer.index].as_ref() else { continue };
        // `[omd_loss]` is the one dense-truth head whose truth is *not* its own
        // output: C++ reads a box table (`batch * max_boxes * truth_size`) or a
        // per-image class vector, so its shape check lives in the module (which
        // falls back to "no samples" when the truth is shorter than it needs).
        // `[contrastive]` (truth `batch * classes`), `[yolox]` (truth
        // `batch * max * truth_size`) and `[omd_loss]` (truth
        // `batch * max_boxes * truth_size`) are the heads whose truth is not
        // their own output; their length checks live in their modules.
        if !matches!(
            layer.kind,
            LayerKind::OmdLoss(_)
                | LayerKind::Contrastive(_)
                | LayerKind::Yolox(_)
                | LayerKind::Ppyoloe(_)
                | LayerKind::Yolonas(_)
                | LayerKind::CenterNet(_)
        ) && t.shape != layer.output_shape
        {
            return Err(Error::InvalidConfig(format!(
                "layer #{} ([{}]) truth shape {} != output shape {}",
                layer.index,
                layer.kind.name(),
                t.shape,
                layer.output_shape
            )));
        }
        let prev = if layer.index == 0 { input } else { &forward.layer_outputs[layer.index - 1] };
        let (delta, cost) = match &layer.kind {
            LayerKind::Cost { kind, scale } => elementwise_cost(prev, t, kind, *scale),
            LayerKind::Regression(spec) => elementwise_cost(prev, t, "sse", spec.scale),
            LayerKind::Logistic => logistic_cost(&forward.layer_outputs[layer.index], t),
            LayerKind::CenterNet(spec) => {
                // `[centernet]`'s truth is `batch * max * truth_size`, and the
                // head's pixel-space gates need the network size.
                let layout = crate::centernet::CenterNetLayout::for_spec(
                    layer.output_shape.batch,
                    layer.output_shape.width,
                    layer.output_shape.height,
                    spec,
                );
                if forward.layer_outputs[layer.index].data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ([centernet]) needs {} channels per cell but the output has {} entries",
                        layer.index,
                        layout.entries(),
                        forward.layer_outputs[layer.index].data.len()
                    )));
                }
                // `NetConfig` carries no class-multiplier or class-map fields
                // yet, so those keep their `None` (C++ default) here.
                let settings = crate::centernet::CenterNetSettings {
                    truth_size: 6,
                    net_width: network.net.width,
                    net_height: network.net.height,
                    ..crate::centernet::CenterNetSettings::default()
                };
                let (delta, cost, _positives) = crate::centernet::centernet_training_delta(
                    &forward.layer_outputs[layer.index].data,
                    &t.data,
                    &layout,
                    spec,
                    &settings,
                )
                .map_err(|err| Error::InvalidConfig(err.0))?;
                (Tensor::from_data(layer.output_shape, delta)?, cost)
            }
            LayerKind::Yolox(spec) | LayerKind::Ppyoloe(spec) | LayerKind::Yolonas(spec) => {
                // C++ reads the box truth `max_boxes * truth_size` per image.
                let layout = crate::modern_yolo::ModernLayout::for_spec(
                    layer.output_shape.batch,
                    layer.output_shape.width,
                    layer.output_shape.height,
                    spec,
                );
                if forward.layer_outputs[layer.index].data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ([yolox]) needs {} channels per cell but the output has {} entries",
                        layer.index,
                        layout.outputs(),
                        forward.layer_outputs[layer.index].data.len()
                    )));
                }
                // C++ `parse_yolox_section` reads only `classes`, `max`,
                // `jitter` and `resize`, so every training knob stays at the
                // value `set_default_training_params` gives it: normalisers 1,
                // `ignore_thresh` 0.7, `max_delta` FLT_MAX and no smoothing or
                // class map.  They are exposed on `ModernYoloSettings` so a
                // caller can override them once the net-level keys exist.
                let settings = crate::modern_yolo::ModernYoloSettings::default();
                let (delta, cost, _positives) =
                    crate::modern_yolo::yolox_training_delta(
                        &forward.layer_outputs[layer.index].data,
                        &t.data,
                        &layout,
                        spec,
                        &settings,
                    )
                    .map_err(|err| Error::InvalidConfig(err.0))?;
                (Tensor::from_data(layer.output_shape, delta)?, cost)
            }
            LayerKind::Contrastive(spec) => {
                // The classifier branch wants one embedding per image.
                let spatial = prev.shape.width * prev.shape.height;
                if spatial != 1 || prev.shape.channels == 0 {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ([contrastive]) needs one embedding per image, got {}x{}x{}",
                        layer.index,
                        prev.shape.width,
                        prev.shape.height,
                        prev.shape.channels
                    )));
                }
                let result = crate::contrastive::contrastive_forward(
                    &prev.data,
                    Some(&t.data),
                    spec,
                    prev.shape.batch,
                    prev.shape.channels,
                    network.net.time_steps,
                    network.net.adversarial,
                    network.net.label_smooth_eps,
                    true,
                )
                .map_err(Error::InvalidConfig)?;
                (Tensor::from_data(prev.shape, result.delta)?, result.cost)
            }
            LayerKind::OmdLoss(spec) => {
                // The layer is a pass-through, so the seed it produces is the
                // metric delta of the *previous* layer's features - which is
                // exactly what the pass-through backward then forwards.
                let (output, result) = crate::omd_loss::omd_loss_forward(
                    &prev.data,
                    Some(&t.data),
                    spec,
                    &prev.shape,
                    network.net.width,
                    network.net.height,
                    true,
                );
                debug_assert_eq!(output, prev.data, "[omd_loss] must be a pass-through");
                (Tensor::from_data(prev.shape, result.delta)?, result.cost)
            }
            LayerKind::Segmentation(spec) => {
                let shape = layer.output_shape;
                let result = crate::segmentation::segmentation_loss(
                    &forward.layer_outputs[layer.index].data,
                    &t.data,
                    spec,
                    shape.batch,
                    shape.width * shape.height,
                    crate::segmentation::gate_strength(spec, iteration),
                );
                let delta = Tensor::from_data(shape, result.delta.unwrap_or_default())?;
                (delta, result.loss)
            }
            LayerKind::JevCe(spec) => {
                let result = crate::jev_ce::jev_ce_loss(&prev.data, &t.data, spec, prev.shape.batch);
                let mut delta = Tensor::from_data(prev.shape, result.delta)?;
                for v in &mut delta.data {
                    *v *= spec.scale; // C++ backward: axpy(scale, delta)
                }
                (delta, result.mean_ce)
            }
            other => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([{}]) is not a dense-truth loss layer",
                    layer.index,
                    other.name()
                )));
            }
        };
        seeds[layer.index] = Some(delta);
        total += cost;
        count += 1;
    }
    Ok(DenseTrainingLoss {
        loss: if count == 0 { 0.0 } else { total / count as f32 },
        layer_output_gradient_seeds: seeds,
    })
}

/// C++ `l2_cpu` / `smooth_l1_cpu` (+ `masked`): `delta = scale * (truth - pred)`, `cost = sum error`.
fn elementwise_cost(pred: &Tensor, truth: &Tensor, kind: &str, scale: f32) -> (Tensor, f32) {
    let (masked, smooth) = (kind == "masked", kind == "smooth");
    let mut delta = Tensor::zeros(pred.shape);
    let mut cost = 0.0f32;
    for (i, (p, t)) in pred.data.iter().zip(&truth.data).enumerate() {
        // C++ overwrites the prediction with SECRET_NUM where the truth is SECRET_NUM => diff 0.
        let p = if masked && *t == SECRET_NUM { SECRET_NUM } else { *p };
        let diff = t - p;
        let (error, d) = if smooth && diff.abs() >= 1.0 {
            (2.0 * diff.abs() - 1.0, if diff > 0.0 { 1.0 } else { -1.0 })
        } else {
            (diff * diff, diff)
        };
        cost += error;
        delta.data[i] = d * scale;
    }
    (delta, cost)
}

/// C++ `logistic_x_ent_cpu` on the sigmoid output: `delta = t - p` (logit space), cross-entropy cost.
/// `ln` arguments are clamped to `[1e-12, 1 - 1e-12]` for the reported cost only (C++ yields inf).
fn logistic_cost(pred: &Tensor, truth: &Tensor) -> (Tensor, f32) {
    let mut delta = Tensor::zeros(pred.shape);
    let mut cost = 0.0f32;
    for (i, (p, t)) in pred.data.iter().zip(&truth.data).enumerate() {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        cost += -t * pc.ln() - (1.0 - t) * (1.0 - pc).ln();
        delta.data[i] = t - p;
    }
    (delta, cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigFile, ExecutableNetwork, LayerParams, Network, TensorShape};

    fn run(cfg: &str, set: impl FnOnce(&mut ExecutableNetwork), truth: Vec<f32>, head: usize) -> DenseTrainingLoss {
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", cfg).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        set(&mut exec);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let shape = exec.network.layers[head].output_shape;
        let mut t = vec![None; exec.network.layers.len()];
        t[head] = Some(Tensor::from_data(shape, truth).unwrap());
        compute_dense_training_loss(&exec.network, &forward, &input, &t).unwrap()
    }

    fn set_bias(exec: &mut ExecutableNetwork, bias: Vec<f32>) {
        let LayerParams::Connected(p) = &mut exec.params[0] else { panic!() };
        p.biases = bias;
    }

    #[test]
    fn cost_sse_smooth_masked_match_c_formulas() {
        let cfg = |kind: &str| format!("[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\nactivation=linear\n[cost]\ntype={kind}\nscale=0.5\n");
        // prediction = bias = [1, 2]; truth [3, 0]: diff [2, -2]
        let sse = run(&cfg("sse"), |e| set_bias(e, vec![1.0, 2.0]), vec![3.0, 0.0], 1);
        assert_eq!(sse.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![1.0, -1.0]); // scale * diff
        assert_eq!(sse.loss, 8.0); // 4 + 4
        // truth [4, 2.5]: diff [3, 0.5] -> |3| >= 1: 2*3-1 = 5, 0.25; deltas 1, 0.5 (sign for |diff|>=1) * scale
        let smooth = run(&cfg("smooth"), |e| set_bias(e, vec![1.0, 2.0]), vec![4.0, 2.5], 1);
        assert_eq!(smooth.loss, 5.25);
        assert_eq!(smooth.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![0.5, 0.25]);
        // masked: first truth is SECRET_NUM -> prediction masked, no loss/gradient there
        let masked = run(&cfg("masked"), |e| set_bias(e, vec![1.0, 2.0]), vec![SECRET_NUM, 0.0], 1);
        assert_eq!(masked.loss, 4.0);
        assert_eq!(masked.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![0.0, -1.0]);
    }

    #[test]
    fn logistic_head_uses_sigmoid_output_and_logit_space_seed() {
        // zero connected params -> logit 0 -> p = 0.5 for every element, batch 2
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=3\nbatch=2\n[connected]\noutput=3\nactivation=linear\n[logistic]\n";
        let loss = run(cfg, |_| {}, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed.data, vec![0.5, -0.5, 0.5, -0.5, -0.5, 0.5]); // t - p
        assert!((loss.loss - 6.0 * std::f32::consts::LN_2).abs() < 1e-5, "{}", loss.loss);
    }

    #[test]
    fn regression_head_is_l2_with_scale_and_passes_the_prediction_through() {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\nactivation=linear\n[regression]\nscale=2\n";
        let loss = run(cfg, |e| set_bias(e, vec![1.0, 2.0]), vec![3.0, 1.0], 1);
        assert_eq!(loss.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![4.0, -2.0]); // 2 * (t - p)
        assert_eq!(loss.loss, 4.0 + 1.0);
    }

    #[test]
    fn segmentation_binary_head_is_logistic_ce_over_the_sigmoid_output() {
        // Zero connected params -> logit 0 -> p = 0.5 per pixel, batch 2, 3 pixels.
        let cfg = "[net]\nwidth=3\nheight=1\nchannels=1\nbatch=2\n\
                   [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
                   [segmentation]\nclasses=1\n";
        let truth = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let loss = run(cfg, |_| {}, truth, 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        // delta = t - p = +-0.5
        assert_eq!(seed.data, vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5]);
        // cost is per pixel: every pixel costs -t ln p - (1-t) ln(1-p) = ln 2.
        assert!((loss.loss - std::f32::consts::LN_2).abs() < 1e-6, "{}", loss.loss);
    }

    #[test]
    fn segmentation_multiclass_head_is_softmax_ce_over_per_pixel_channels() {
        // Zero connected params -> every logit 0 -> p = 1/3 per class.
        let cfg = "[net]\nwidth=2\nheight=1\nchannels=3\n\
                   [convolutional]\nfilters=3\nsize=1\nactivation=linear\n\
                   [segmentation]\nclasses=3\n";
        // NCHW with two pixels: pixel 0 is class 2 and pixel 1 is class 1, so
        // the one-hot truth is [0,0,1, 0,1,0] (channel-major, spatial stride 2).
        let loss = run(cfg, |_| {}, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        let third = 1.0f32 / 3.0;
        // delta = t - p, so the two truth channels get 2/3 and the other four
        // get -1/3.
        for (i, want) in [0usize, 1, 3, 5].iter().map(|i| (*i, -third)).chain([2usize, 4].iter().map(|i| (*i, 1.0 - third))) {
            assert!((seed.data[i] - want).abs() < 1e-6, "[{i}]: {} vs {want}", seed.data[i]);
        }
        // Two pixels, each costing ln 3, so the per-pixel mean is ln 3.
        assert!((loss.loss - 3f32.ln()).abs() < 1e-6, "{}", loss.loss);
    }

    #[test]
    fn segmentation_head_applies_ce_scale_and_the_gate_needs_the_iteration() {
        let cfg = |gate: &str| {
            format!(
                "[net]\nwidth=2\nheight=1\nchannels=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [segmentation]\nclasses=2\ncls_normalizer=2\nlambda_ce=0.5\n{gate}"
            )
        };
        // NCHW with two pixels and two classes: pixel 0 -> class 0, pixel 1 -> class 0.
        let truth = vec![1.0, 1.0, 0.0, 0.0];
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", &cfg("")).unwrap()).unwrap();
        let exec = ExecutableNetwork::new(network);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let shape = exec.network.layers[1].output_shape;
        let mut t = vec![None; exec.network.layers.len()];
        t[1] = Some(Tensor::from_data(shape, truth.clone()).unwrap());
        let plain = compute_dense_training_loss(&exec.network, &forward, &input, &t).unwrap();
        let scaled = compute_dense_training_loss_at(&exec.network, &forward, &input, &t, 0).unwrap();
        assert_eq!(plain.loss, scaled.loss, "iteration 0 is before the gate warmup");
        // ce_scale = lambda_ce * ce_normalizer * cls_normalizer = 0.5 * 1 * 2 = 1.
        let reference = crate::segmentation::segmentation_loss(
            &forward.layer_outputs[1].data,
            &truth,
            match &exec.network.layers[1].kind {
                LayerKind::Segmentation(spec) => spec,
                other => panic!("not segmentation: {}", other.name()),
            },
            shape.batch,
            shape.width * shape.height,
            0.0,
        );
        assert!((plain.loss - reference.loss).abs() < 1e-6, "{} vs {}", plain.loss, reference.loss);
        assert_eq!(
            plain.layer_output_gradient_seeds[1].as_ref().unwrap().data,
            reference.delta.unwrap()
        );

        // With the gate on and the iteration past the warmup + ramp, the
        // strength saturates at 1, every already-correct confident pixel is
        // suppressed, and the survivors are compensated by total/active.
        let gated_cfg = cfg("gate_enabled=1\ngate_t_correct=0.3\ngate_warmup_iters=0\ngate_ramp_iters=1\n");
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", &gated_cfg).unwrap()).unwrap();
        let exec = ExecutableNetwork::new(network);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let shape = exec.network.layers[1].output_shape;
        let mut t = vec![None; exec.network.layers.len()];
        t[1] = Some(Tensor::from_data(shape, truth).unwrap());
        let at_peak = compute_dense_training_loss_at(&exec.network, &forward, &input, &t, 1).unwrap();
        let before = compute_dense_training_loss_at(&exec.network, &forward, &input, &t, 0).unwrap();
        let seed_peak = at_peak.layer_output_gradient_seeds[1].as_ref().unwrap();
        // The uniform prediction makes class 0 the argmax for both pixels, and
        // the truth picks class 0, so both pixels are correct with p = 1/2 >=
        // t_correct.  At strength 1 the gate scales their deltas by 0, no pixel
        // is left active, so the compensation is skipped and the seeds vanish.
        assert!(
            seed_peak.data.iter().all(|v| v.abs() < 1e-6),
            "fully suppressed pixels have no seed: {:?}",
            seed_peak.data
        );
        let seed_before = before.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed_before.data, vec![0.5, 0.5, -0.5, -0.5], "before the ramp the gate is off");
    }

    /// `[omd_loss]` with box truth: a 1x1 convolution copies its input, so each
    /// image's box sample sees that image's own two features.  The two are
    /// labelled differently, so their pair is negative.
    fn omd_detection_loss(input: &[f32]) -> DenseTrainingLoss {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=2\nbatch=2\n\
                   [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                   [omd_loss]\nclasses=2\nmax_boxes=2\ntruth_size=6\n\
                   lambda=1\nmargin=0.5\n";
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", cfg).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        {
            // Identity 1x1 convolution: weights [f0c0, f0c1, f1c0, f1c1], no bias.
            let LayerParams::Convolutional(p) = &mut exec.params[0] else { panic!() };
            p.weights = vec![1.0, 0.0, 0.0, 1.0];
            p.biases = vec![0.0, 0.0];
        }
        let shape = exec.network.net.input_shape();
        let data = input.to_vec();
        assert_eq!(data.len(), shape.batch * shape.element_count());
        let input = Tensor::from_data(shape, data).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        // Box truth: one box per image, class 0 and class 1.  The layer's truth
        // is `batch * max_boxes * truth_size`, not its output shape.
        let mut truth = vec![0.0f32; 2 * 2 * 6];
        for (b, cls) in [(0usize, 0.0f32), (1, 1.0)] {
            let base = b * 2 * 6;
            truth[base] = 0.5;
            truth[base + 1] = 0.5;
            truth[base + 2] = 0.5;
            truth[base + 3] = 0.5;
            truth[base + 4] = cls;
        }
        let mut t = vec![None; exec.network.layers.len()];
        t[1] = Some(Tensor::from_data(TensorShape::new(12, 1, 1, 2), truth).unwrap());
        compute_dense_training_loss(&exec.network, &forward, &input, &t).unwrap()
    }

    /// `[yolox]` reads `classes + 5` channels per cell, so a cfg that hands it
    /// fewer is a config error rather than a silent reinterpretation.
    #[test]
    fn yolox_head_checks_its_channel_count_and_produces_a_head_seed() {
        fn run_yolox(classes: usize, filters: usize) -> crate::error::Result<DenseTrainingLoss> {
            let cfg = format!(
                "[net]\nwidth=2\nheight=2\nchannels={filters}\nbatch=2\n\
                 [convolutional]\nfilters={filters}\nsize=1\n\
                 [yolox]\nclasses={classes}\nmax=2\n"
            );
            let network = Network::from_config(ConfigFile::parse_str("d.cfg", &cfg).unwrap()).unwrap();
            let exec = ExecutableNetwork::new(network);
            let shape = exec.network.net.input_shape();
            let input = Tensor::zeros(shape);
            let forward = exec.forward_training(&input).unwrap();
            // Two boxes per image, `truth_size = 6`.
            let mut truth = vec![0.0f32; 2 * 2 * 6];
            truth[0] = 0.5;
            truth[1] = 0.5;
            truth[2] = 1.0;
            truth[3] = 1.0;
            truth[4] = 0.0;
            let mut t = vec![None; exec.network.layers.len()];
            t[1] = Some(Tensor::from_data(TensorShape::new(2 * 6, 1, 1, 2), truth).unwrap());
            compute_dense_training_loss(&exec.network, &forward, &input, &t)
        }
        // 2 classes needs 2 + 5 = 7 channels: the head produces a seed.
        let loss = run_yolox(2, 7).expect("a 7-channel head carries 2 classes");
        assert!(loss.loss.is_finite());
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed.data.len(), 2 * 7 * 4, "one entry per output element");
        // 9 classes would need 14 channels, so 7 is not enough.
        let err = run_yolox(9, 7).expect_err("a 7-channel head cannot carry 9 classes");
        assert!(format!("{err}").contains("channels per cell"), "{err}");
    }

    #[test]
    fn contrastive_head_uses_the_embedding_delta_not_its_own_output() {
        // A global-average `[convolutional]` hands `[contrastive]` one embedding
        // vector per image; three images of the same class with spread-out
        // embeddings must produce a non-zero seed on the *convolution*.
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=2\nbatch=3\n\
                   [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                   [contrastive]\nclasses=1\ncls_normalizer=1\n";
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", cfg).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        {
            // Identity 1x1 convolution so the embeddings are the inputs.
            let LayerParams::Convolutional(p) = &mut exec.params[0] else { panic!() };
            p.weights = vec![1.0, 0.0, 0.0, 1.0];
            p.biases = vec![0.0, 0.0];
        }
        let shape = exec.network.net.input_shape();
        let input = Tensor::from_data(shape, vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        let truth = vec![1.0f32; 3]; // every image is class 0
        let mut t = vec![None; exec.network.layers.len()];
        t[1] = Some(Tensor::from_data(TensorShape::new(1, 1, 1, 3), truth.clone()).unwrap());
        let loss = compute_dense_training_loss(&exec.network, &forward, &input, &t).unwrap();
        assert!(loss.loss > 0.0, "cost {}", loss.loss);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed.data.len(), 6, "one entry per previous-layer element");
        assert!(seed.data.iter().any(|d| d.abs() > 1e-6), "{:?}", seed.data);
    }

    #[test]
    fn contrastive_detection_branch_reports_itself_as_unported() {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=2\n\
                   [convolutional]\nfilters=2\nsize=1\n[contrastive]\nclasses=1\nyolo_layer=-1\n";
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", cfg).unwrap()).unwrap();
        let exec = ExecutableNetwork::new(network);
        let shape = exec.network.net.input_shape();
        let input = Tensor::zeros(shape);
        let forward = exec.forward_training(&input).unwrap();
        let mut t = vec![None; exec.network.layers.len()];
        t[1] = Some(Tensor::zeros(TensorShape::new(1, 1, 1, shape.batch)));
        let err = compute_dense_training_loss(&exec.network, &forward, &input, &t)
            .expect_err("the detection branch must not pretend to work");
        assert!(format!("{err}").contains("detection branch"), "{err}");
    }

    #[test]
    fn omd_loss_head_is_a_pass_through_that_seeds_the_metric_delta() {
        // Near-identical images with different labels: the negative pair fires.
        let loss = omd_detection_loss(&[1.0, 0.0, 0.9, 0.1]);
        assert!(loss.loss > 0.0, "cost {}", loss.loss);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed.data.len(), 4, "one entry per previous-layer element");
        assert!(seed.data.iter().any(|d| d.abs() > 1e-5), "something must be pulled: {:?}", seed.data);
        // Orthogonal images with different labels: distance 1, violation
        // 0.5 - 1 < 0, so the pair is already inside the margin and nothing moves.
        let relaxed = omd_detection_loss(&[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(relaxed.loss, 0.0);
        assert!(relaxed.layer_output_gradient_seeds[1].as_ref().unwrap().data.iter().all(|d| d.abs() < 1e-6));
    }

    #[test]
    fn jev_ce_head_seed_is_scale_times_delta_and_cost_is_mean_ce() {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=3\n[connected]\noutput=3\nactivation=linear\n[jev_ce]\ngroups=1\nscale=2\n";
        let loss = run(cfg, |_| {}, vec![1.0, 0.0, 0.0], 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        let third = 1.0f32 / 3.0;
        assert!((seed.data[0] - 2.0 * (1.0 - third)).abs() < 1e-6);
        assert!((loss.loss - 3f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn layers_without_truth_are_skipped_and_wrong_shapes_rejected() {
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\n[cost]\n").unwrap()).unwrap();
        let exec = ExecutableNetwork::new(network);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let none = compute_dense_training_loss(&exec.network, &forward, &input, &[None, None]).unwrap();
        assert_eq!(none.loss, 0.0);
        assert!(none.layer_output_gradient_seeds.iter().all(Option::is_none));
        let bad = Tensor::zeros(TensorShape::new(1, 1, 3, 1));
        assert!(compute_dense_training_loss(&exec.network, &forward, &input, &[None, Some(bad)]).is_err());
    }
}
