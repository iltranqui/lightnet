//! The GPU training forward against the CPU reference [`TrainNet`], layer by layer, on the real
//! LegoGears cfg with a batch of images. Skips when the DarkSuite dataset is not present.

#![cfg(feature = "gpu")]

use hip_darknet::cfg::Cfg;
use hip_darknet::data::BoxLabel;
use hip_darknet::train::{GpuTrainNet, TrainNet};

const DIR: &str = "/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2";

/// Largest absolute difference as a fraction of the tensor's largest magnitude. (A per-element
/// relative error with a floor would inflate the figure ~100x on near-zero entries; this is the
/// honest f32 number.)
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-9);
    got.iter().zip(want).map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs() / scale).fold(0.0, f64::max)
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The LegoGears net at batch 2 with non-trivial BN scale/shift and a random input batch.
fn setup() -> Option<(TrainNet, Vec<f32>)> {
    let path = std::path::Path::new(DIR).join("LegoGears_v2.cfg");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return None;
    }
    let cfg = Cfg::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut net = TrainNet::new(&cfg, 2, 7).unwrap();
    // Non-trivial BN scale/shift so the affine part of the layer is actually tested.
    let mut next = lcg(11);
    for conv in net.convs.iter_mut().flatten() {
        for s in conv.scales.iter_mut() {
            *s = 0.5 + next();
        }
        for b in conv.bias.iter_mut() {
            *b = next() - 0.5;
        }
    }
    let mut next = lcg(3);
    let input = (0..net.input.len()).map(|_| next()).collect();
    Some((net, input))
}

#[test]
fn gpu_train_forward_matches_cpu_per_layer() {
    let Some((mut net, input)) = setup() else { return };

    let mut gpu = GpuTrainNet::from_net(&net).unwrap();
    net.set_input(&input).unwrap();
    net.forward().unwrap();
    gpu.forward(&input).unwrap();

    let mut worst = 0f64;
    for i in 0..net.model.layers.len() {
        let got = gpu.layer_out(i).unwrap();
        let e = rel_err(&got, &net.outs[i]);
        println!("layer {i:2} {:?}: err/scale {e:.3e}", net.model.layers[i].out_shape);
        assert!(e < 1e-4, "layer {i}: err {e:e}");
        worst = worst.max(e);
        if let Some([mean, var, rmean, rvar]) = gpu.bn_state(i).unwrap() {
            let conv = net.convs[i].as_ref().unwrap();
            for (name, g, w) in [("mean", &mean, &conv.mean), ("var", &var, &conv.var), ("rolling_mean", &rmean, &conv.rolling_mean), ("rolling_var", &rvar, &conv.rolling_var)] {
                // A plain (non-BN) conv leaves its statistics at their initial values on both sides.
                let e = rel_err(g, w);
                assert!(e < 1e-4, "layer {i} {name}: err {e:e}");
            }
        }
    }
    println!("worst layer err/scale {worst:.3e}");
}

/// A few random ground-truth boxes per image, 5 classes.
fn labels(batch: usize) -> Vec<Vec<BoxLabel>> {
    let mut next = lcg(5);
    (0..batch)
        .map(|_| {
            (0..1 + (next() * 3.0) as usize)
                .map(|_| BoxLabel {
                    class: (next() * 5.0) as usize % 5,
                    x: 0.15 + 0.7 * next(),
                    y: 0.15 + 0.7 * next(),
                    w: 0.05 + 0.3 * next(),
                    h: 0.05 + 0.3 * next(),
                })
                .collect()
        })
        .collect()
}

/// The `[yolo]` deltas and the network loss on the device against the CPU trainer, for a fixed
/// seeded batch with ground-truth boxes.
#[test]
fn gpu_loss_and_deltas_match_cpu() {
    let Some((mut net, input)) = setup() else { return };
    let labels = labels(net.batch);

    let mut gpu = GpuTrainNet::from_net(&net).unwrap();
    net.set_input(&input).unwrap();
    net.zero_grads();
    net.forward().unwrap();
    let cpu_loss = net.loss_and_deltas(&labels).unwrap();
    gpu.zero_grads().unwrap();
    gpu.forward(&input).unwrap();
    let gpu_loss = gpu.loss_and_deltas(&labels).unwrap();
    println!("loss: cpu {cpu_loss:.6} gpu {gpu_loss:.6}");
    assert!(cpu_loss.is_finite() && cpu_loss > 0.0, "cpu loss {cpu_loss}");
    let rel = f64::from((gpu_loss - cpu_loss).abs()) / f64::from(cpu_loss.abs());
    assert!(rel < 1e-3, "loss: cpu {cpu_loss} gpu {gpu_loss} (rel {rel:e})");

    // The delta of each head lands in the previous layer's gradient.
    for &y in &net.model.yolo_layers() {
        let e = rel_err(&gpu.layer_grad(y - 1).unwrap(), &net.grads[y - 1]);
        println!("head layer {y}: grad err/scale {e:.3e}");
        assert!(e < 1e-3, "head {y} gradient: err {e:e}");
    }
}

/// One whole training step: the per-layer gradients, every parameter gradient, and the weights
/// after the SGD update, GPU against the CPU reference.
#[test]
fn gpu_backward_and_update_match_cpu() {
    let Some((mut net, input)) = setup() else { return };
    let labels = labels(net.batch);
    let mut gpu = GpuTrainNet::from_net(&net).unwrap();

    net.set_input(&input).unwrap();
    net.zero_grads();
    net.forward().unwrap();
    net.loss_and_deltas(&labels).unwrap();
    net.backward();
    // The backward pass runs on the CPU's own forward values: two f32 forwards differ by rounding,
    // and a value within that rounding of a leaky-ReLU kink flips its gradient factor (1 vs 0.1),
    // which would show up as a spurious gradient mismatch. The forward is compared on its own above.
    gpu.load_activations(&net).unwrap();
    gpu.zero_grads().unwrap();
    gpu.loss_and_deltas(&labels).unwrap();
    gpu.backward().unwrap();

    // Gradients w.r.t. every layer's output. The ones the CPU skips (a yolo layer) are untouched.
    let mut worst_layer = 0f64;
    for i in 0..net.model.layers.len() {
        let e = rel_err(&gpu.layer_grad(i).unwrap(), &net.grads[i]);
        println!("layer {i:2} {:?} grad err/scale {e:.3e}", net.model.layers[i].out_shape);
        worst_layer = worst_layer.max(e);
    }
    assert!(worst_layer < 1e-4, "layer gradient: worst err {worst_layer:e}");

    // Parameter gradients (the momentum buffers start at zero, so they hold exactly the gradient).
    let mut got = TrainNet::new(&net.cfg.clone(), net.batch, 0).unwrap();
    gpu.download_into(&mut got).unwrap();
    let mut worst_param = 0f64;
    for (i, (g, c)) in got.convs.iter().zip(&net.convs).enumerate() {
        let (Some(g), Some(c)) = (g, c) else { continue };
        for (name, a, b) in [("weight", &g.weight_updates, &c.weight_updates), ("bias", &g.bias_updates, &c.bias_updates), ("scale", &g.scale_updates, &c.scale_updates)] {
            let e = rel_err(a, b);
            println!("layer {i:2} {name} gradient err/scale {e:.3e}");
            assert!(e < 1e-4, "layer {i} {name} gradient: err {e:e}");
            worst_param = worst_param.max(e);
        }
    }

    // The SGD step.
    let lr = 1e-3;
    net.update(lr);
    gpu.update(lr).unwrap();
    gpu.download_into(&mut got).unwrap();
    let mut worst_w = 0f64;
    for (i, (g, c)) in got.convs.iter().zip(&net.convs).enumerate() {
        let (Some(g), Some(c)) = (g, c) else { continue };
        for (name, a, b) in [("weights", &g.weights, &c.weights), ("bias", &g.bias, &c.bias), ("scales", &g.scales, &c.scales), ("momentum", &g.weight_updates, &c.weight_updates)] {
            let e = rel_err(a, b);
            assert!(e < 1e-4, "layer {i} {name} after update: err {e:e}");
            worst_w = worst_w.max(e);
        }
    }
    println!("worst: layer grad {worst_layer:.3e}, param grad {worst_param:.3e}, after update {worst_w:.3e}");
}
