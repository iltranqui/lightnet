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
        if t.shape != layer.output_shape {
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
