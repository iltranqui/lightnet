//! Stage-02 gate: every backward pass is checked numerically, in f32, without the Darknet binary.
//!
//! Method: with a fixed upstream gradient `g`, `L(x) = <g, out(x)>`. The analytic gradient from
//! [`TrainNet::backward`] must equal central differences of `L` w.r.t. the input (and, for a conv,
//! w.r.t. its weights). This is the only external check available (`_shared/rules.md`), and it is
//! strict: an error here means the GPU port would be learning from the wrong gradient.

use hip_darknet::cfg::Cfg;
use hip_darknet::train::net::{learning_rate, TrainNet};
use hip_darknet::cfg::IouLoss;

fn build(cfg: &str, batch: usize, seed: u64) -> TrainNet {
    let cfg = Cfg::parse(cfg).unwrap();
    TrainNet::new(&cfg, batch, seed).unwrap()
}

fn fill_pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32;
            (v - 0.5) * (1.0 + (i % 3) as f32)
        })
        .collect()
}

/// What `TrainNet::backward` reports for the input gradient, given a fixed upstream gradient at
/// `layer`.
fn analytic_input_grad(net: &mut TrainNet, input: &[f32], layer: usize, g: &[f32]) -> Vec<f32> {
    net.set_input(input).unwrap();
    net.zero_grads();
    net.forward().unwrap();
    net.grads[layer].copy_from_slice(g);
    net.backward();
    net.input_grad.clone()
}

/// `L(x) = <g, out_layer(x)>`, recomputed by a full forward.
fn loss_at(net: &mut TrainNet, input: &[f32], layer: usize, g: &[f32]) -> f32 {
    net.set_input(input).unwrap();
    net.forward().unwrap();
    net.outs[layer].iter().zip(g).map(|(a, b)| a * b).sum()
}

/// Returns the numerical gradient and the f32 cancellation floor of the central differences
/// (`~eps * |L| / h`), which is what a component below that size cannot be checked against.
fn fd_input_grad(net: &mut TrainNet, x: &[f32], layer: usize, g: &[f32], h: f32) -> (Vec<f32>, f32) {
    let base = loss_at(net, x, layer, g);
    let noise = 8.0 * f32::EPSILON * base.abs() / h;
    let mut grad = vec![0.0f32; x.len()];
    let mut probe = x.to_vec();
    for i in 0..x.len() {
        let orig = probe[i];
        probe[i] = orig + h;
        let plus = loss_at(net, &probe, layer, g);
        probe[i] = orig - h;
        let minus = loss_at(net, &probe, layer, g);
        probe[i] = orig;
        grad[i] = (plus - minus) / (2.0 * h);
    }
    (grad, noise)
}

fn compare_with_noise(name: &str, analytic: &[f32], numeric: &[f32], rel_tol: f32, noise: f32) {
    // f32 central differences lose ~7 digits to cancellation, so a component whose magnitude is
    // far below the largest component is compared against a fraction of that largest one.
    let max_abs = analytic.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let mut worst = 0.0f32;
    let mut worst_i = 0;
    for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
        let denom = a.abs().max(n.abs()).max(1e-2 * max_abs).max(noise);
        let rel = (a - n).abs() / denom;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    assert!(
        worst <= rel_tol,
        "{name}: worst relative error {worst} at index {worst_i} (analytic {}, numeric {}) — tolerance {rel_tol}, max|analytic| {max_abs}, fd noise {noise}",
        analytic[worst_i],
        numeric[worst_i]
    );
}

#[test]
fn conv_dx_matches_numerics() {
    // 2 channels 6x5 -> 3 channels 4x3, 3x3 kernel stride 2, linear.
    let cfg = "[net]\nwidth=6\nheight=5\nchannels=2\nbatch=1\n\
        [convolutional]\nfilters=3\nsize=3\nstride=2\npad=0\nactivation=linear\n";
    let mut net = build(cfg, 1, 11);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 3);
    let g = fill_pattern(net.outs[last].len(), 5);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("conv dX", &analytic, &numeric, 2e-2, noise);
}

#[test]
fn conv_dw_matches_numerics() {
    let cfg = "[net]\nwidth=5\nheight=4\nchannels=2\nbatch=1\n\
        [convolutional]\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n";
    let mut net = build(cfg, 1, 23);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 7);
    let g = fill_pattern(net.outs[last].len(), 9);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let updates: Vec<f32> = net.convs[last].as_ref().unwrap().weight_updates.clone();
    assert_eq!(updates.len(), 2 * 2 * 3 * 3);
    // Checks the input gradient and the weight gradient independently.
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("conv dX (pad 1)", &analytic, &numeric, 2e-2, noise);

    // Now weights: perturb one weight, recompute L, compare with `weight_updates`.
    let mut worst = 0.0f32;
    #[allow(clippy::needless_range_loop)]
    for w in 0..updates.len() {
        let orig = net.convs[last].as_ref().unwrap().weights[w];
        let step = 1e-3;
        net.convs[last].as_mut().unwrap().weights[w] = orig + step;
        let plus = loss_at(&mut net, &x, last, &g);
        net.convs[last].as_mut().unwrap().weights[w] = orig - step;
        let minus = loss_at(&mut net, &x, last, &g);
        net.convs[last].as_mut().unwrap().weights[w] = orig;
        let numeric = (plus - minus) / (2.0 * step);
        let denom = updates[w].abs().max(numeric.abs()).max(1e-3);
        worst = worst.max((updates[w] - numeric).abs() / denom);
    }
    assert!(worst <= 2e-2, "conv dW worst relative error {worst}");
}

#[test]
fn bn_conv_dx_matches_numerics() {
    // Batch 2 so the unbiased variance is defined; BN + leaky, then FD through the batch statistics.
    // Darknet's BN backward uses `n` where the exact derivative of its unbiased variance needs
    // `n-1` (documented in `_shared/darknet-semantics.md`), so n must be large enough for that
    // known ~1/n difference to sit inside the tolerance: 2 * 16 * 16 = 512 -> 0.2 %.
    let cfg = "[net]\nwidth=16\nheight=16\nchannels=1\nbatch=2\n\
        [convolutional]\nbatch_normalize=1\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n";
    let mut net = build(cfg, 2, 31);
    // Non-unit scales make the BN backward non-trivial.
    for (i, s) in net.convs[0].as_mut().unwrap().scales.iter_mut().enumerate() {
        *s = 1.0 + 0.25 * i as f32;
    }
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 13);
    let g = fill_pattern(net.outs[last].len(), 17);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    // Big probe step: BN output is nearly scale invariant, and the f32 cancellation floor of the
    // central difference (~eps*|L|/h) otherwise swamps the comparison.
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 2e-2);
    compare_with_noise("bn conv dX", &analytic, &numeric, 4e-2, noise);
}

#[test]
fn bn_scale_updates_match_numerics() {
    let cfg = "[net]\nwidth=5\nheight=4\nchannels=2\nbatch=2\n\
        [convolutional]\nbatch_normalize=1\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n";
    let mut net = build(cfg, 2, 37);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 19);
    let g = fill_pattern(net.outs[last].len(), 29);
    net.set_input(&x).unwrap();
    net.zero_grads();
    net.forward().unwrap();
    net.grads[last].copy_from_slice(&g);
    net.backward();
    let analytic = net.convs[0].as_ref().unwrap().scale_updates.clone();
    assert_eq!(analytic.len(), 2);
    let step = 1e-3;
    #[allow(clippy::needless_range_loop)]
    for c in 0..analytic.len() {
        let orig = net.convs[0].as_ref().unwrap().scales[c];
        net.convs[0].as_mut().unwrap().scales[c] = orig + step;
        let plus = loss_at(&mut net, &x, last, &g);
        net.convs[0].as_mut().unwrap().scales[c] = orig - step;
        let minus = loss_at(&mut net, &x, last, &g);
        net.convs[0].as_mut().unwrap().scales[c] = orig;
        let numeric = (plus - minus) / (2.0 * step);
        let denom = analytic[c].abs().max(numeric.abs()).max(1e-2);
        let rel = (analytic[c] - numeric).abs() / denom;
        assert!(rel <= 3e-2, "bn dscale[{c}]: analytic {} numeric {} (rel {rel})", analytic[c], numeric);
    }
}

#[test]
fn maxpool_dx_matches_numerics() {
    let cfg = "[net]\nwidth=6\nheight=5\nchannels=2\nbatch=1\n[maxpool]\nsize=2\nstride=2\n";
    let mut net = build(cfg, 1, 41);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 43);
    let g = fill_pattern(net.outs[last].len(), 47);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("maxpool dX", &analytic, &numeric, 5e-3, noise);
}

#[test]
fn upsample_dx_matches_numerics() {
    let cfg = "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[upsample]\nstride=2\n";
    let mut net = build(cfg, 2, 53);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 59);
    let g = fill_pattern(net.outs[last].len(), 61);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("upsample dX", &analytic, &numeric, 5e-3, noise);
}

#[test]
fn route_groups_dx_matches_numerics() {
    // Two conv branches then a grouped route: both sources must receive gradient.
    let cfg = "[net]\nwidth=5\nheight=4\nchannels=2\nbatch=1\n\
        [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=linear\n\
        [convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\nactivation=linear\n\
        [route]\nlayers=-1,-2\ngroups=2\ngroup_id=1\n";
    let mut net = build(cfg, 1, 67);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 71);
    let g = fill_pattern(net.outs[last].len(), 73);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("route dX", &analytic, &numeric, 2e-2, noise);
}

#[test]
fn concat_route_dx_matches_numerics() {
    // A full (ungrouped) route of two different layers: the sum of both branches' gradients.
    let cfg = "[net]\nwidth=5\nheight=4\nchannels=2\nbatch=1\n\
        [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nactivation=linear\n\
        [convolutional]\nfilters=3\nsize=1\nstride=1\npad=0\nactivation=linear\n\
        [route]\nlayers=-1,-2\n";
    let mut net = build(cfg, 1, 79);
    let last = net.model.layers.len() - 1;
    let x = fill_pattern(net.input.len(), 83);
    let g = fill_pattern(net.outs[last].len(), 89);
    let analytic = analytic_input_grad(&mut net, &x, last, &g);
    let (numeric, noise) = fd_input_grad(&mut net, &x, last, &g, 3e-3);
    compare_with_noise("concat route dX", &analytic, &numeric, 3e-2, noise);
}

/// The leaky slope is applied from the sign of the stored output — a unit-level check that does
/// not rely on finite differences (which cannot straddle the kink).
#[test]
fn leaky_gradient_uses_the_slope() {
    let cfg = "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
        [convolutional]\nfilters=1\nsize=1\nstride=1\npad=0\nactivation=leaky\n";
    let mut net = build(cfg, 1, 211);
    net.convs[0].as_mut().unwrap().weights[0] = 1.0;
    let x: Vec<f32> = vec![-2.0, -0.5, 0.5, 2.0, 1.0, -1.0, 3.0, -3.0, 0.25, -0.25, 0.75, -0.75, 1.5, -1.5, 2.5, -2.5];
    net.set_input(&x).unwrap();
    net.zero_grads();
    net.forward().unwrap();
    let n = net.outs[0].len();
    net.grads[0].fill(1.0);
    net.backward();
    for (i, &xv) in x.iter().enumerate() {
        let expected = if xv > 0.0 { 1.0 } else { 0.1 };
        assert!((net.input_grad[i] - expected).abs() < 1e-6, "x {xv}: {} vs {expected}", net.input_grad[i]);
    }
    assert_eq!(net.outs[0].len(), n);
    let _ = IouLoss::Ciou;
}

#[test]
fn sgd_step_matches_darknet_formula() {
    let cfg = "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=2\nmomentum=0.9\ndecay=0.0005\n\
        [convolutional]\nfilters=1\nsize=1\nstride=1\npad=0\nactivation=linear\n";
    let mut net = build(cfg, 2, 101);
    let conv = net.convs[0].as_mut().unwrap();
    conv.weights[0] = 2.0;
    conv.weight_updates[0] = 3.0;
    conv.bias[0] = 0.5;
    conv.bias_updates[0] = -1.0;
    conv.scales[0] = 1.5;
    conv.scale_updates[0] = 0.5;
    let lr = 0.1f32;
    net.update(lr);
    let conv = net.convs[0].as_ref().unwrap();
    let batch = 2.0f32;
    // w += (lr/batch) * (update - decay*batch*w); update *= momentum
    let expect_w = 2.0 + (lr / batch) * (3.0 - 0.0005 * batch * 2.0);
    assert!((conv.weights[0] - expect_w).abs() < 1e-6, "{} vs {expect_w}", conv.weights[0]);
    // The decay term goes into the update before it is applied and decayed.
    let expect_u = (3.0 - 0.0005 * batch * 2.0) * 0.9;
    assert!((conv.weight_updates[0] - expect_u).abs() < 1e-6, "{} vs {expect_u}", conv.weight_updates[0]);
    // bias and scales get no weight decay.
    assert!((conv.bias[0] - (0.5 - lr / batch)).abs() < 1e-6);
    assert!((conv.bias_updates[0] - -0.9).abs() < 1e-6);
    assert!((conv.scales[0] - (1.5 + (lr / batch) * 0.5)).abs() < 1e-6);
    assert!((conv.scale_updates[0] - 0.45).abs() < 1e-6);
}

#[test]
fn lr_schedule_burn_in_and_steps() {
    let mut cfg = Cfg::parse("[net]\nwidth=4\nheight=4\nchannels=1\nlearning_rate=0.00261\nburn_in=1000\n\
        policy=steps\nsteps=2400,2700\nscales=0.2,0.1\n")
    .unwrap()
    .train;
    // burn_in with power 4: (i/1000)^4. The argument is Darknet's `batch_num`, i.e. the number of
    // completed updates, which the trainer passes as `iteration - 1` (that is why the reference
    // csv prints lr = 0 for its first iterations).
    assert_eq!(learning_rate(&cfg, 0), 0.0);
    assert!((learning_rate(&cfg, 500) - 0.00261 * 0.5f32.powi(4)).abs() < 1e-9);
    assert!((learning_rate(&cfg, 1000) - 0.00261).abs() < 1e-9);
    assert!((learning_rate(&cfg, 2399) - 0.00261).abs() < 1e-9);
    assert!((learning_rate(&cfg, 2400) - 0.00261 * 0.2).abs() < 1e-9);
    assert!((learning_rate(&cfg, 2699) - 0.00261 * 0.2).abs() < 1e-9);
    assert!((learning_rate(&cfg, 2700) - 0.00261 * 0.2 * 0.1).abs() < 1e-9);
    cfg.burn_in = 0;
    cfg.policy = hip_darknet::cfg::LrPolicy::Constant;
    assert_eq!(learning_rate(&cfg, 7), 0.00261);
}

/// Regression: maxpool forward/backward once ignored the batch and only handled image 0, which no
/// `batch = 1` test could see. A batch must equal its images run one by one.
#[test]
fn maxpool_batched_equals_per_image() {
    let cfg = "[net]\nwidth=6\nheight=5\nchannels=2\nbatch=1\n[maxpool]\nsize=2\nstride=2\n";
    let (batch, mut batched) = (3, build(cfg, 3, 41));
    let per = batched.input.len() / batch;
    let x = fill_pattern(batched.input.len(), 43);
    let last = batched.model.layers.len() - 1;
    let out_per = batched.outs[last].len() / batch;
    let g = fill_pattern(batched.outs[last].len(), 47);
    let got_dx = analytic_input_grad(&mut batched, &x, last, &g);
    let got_out = batched.outs[last].clone();
    for b in 0..batch {
        let mut single = build(cfg, 1, 41);
        let dx = analytic_input_grad(&mut single, &x[b * per..(b + 1) * per], last, &g[b * out_per..(b + 1) * out_per]);
        assert_eq!(&got_out[b * out_per..(b + 1) * out_per], &single.outs[last][..], "forward, image {b}");
        assert_eq!(&got_dx[b * per..(b + 1) * per], &dx[..], "backward, image {b}");
    }
}
