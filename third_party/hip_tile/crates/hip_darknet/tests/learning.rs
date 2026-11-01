//! End-to-end sanity of the training math: a tiny net trained on a fixed synthetic sample must
//! lose loss monotonically-ish and drop it by a wide margin. This exercises forward, Darknet
//! deltas, the whole backward pass and the SGD step together — the pieces stage 02 owns.

use hip_darknet::cfg::Cfg;
use hip_darknet::data::BoxLabel;
use hip_darknet::train::TrainNet;

const CFG: &str = "[net]\nwidth=16\nheight=16\nchannels=3\nbatch=3\nmomentum=0.9\ndecay=0.0\n\
    learning_rate=0.01\nburn_in=0\nmax_batches=400\npolicy=constant\n\
    [convolutional]\nbatch_normalize=1\nfilters=8\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
    [convolutional]\nfilters=6\nsize=1\nstride=1\npad=0\nactivation=linear\n\
    [yolo]\nmask=0\nanchors=6,6\nclasses=1\nnum=1\nscale_x_y=1.0\niou_loss=ciou\niou_normalizer=0.07\n\
    ignore_thresh=0.7\ntruth_thresh=1.0\n";

/// `n` noisy images, each with a bright square at the same place.
/// `n > 1` matters: with a single image the batch variance of a BN layer is zero.
fn sample(n: usize) -> (Vec<f32>, Vec<Vec<BoxLabel>>) {
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0.0f32; n * 3 * w * h];
    let mut labels = vec![];
    let mut seed = 12345u64;
    for b in 0..n {
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((seed >> 33) as f32 / (1u64 << 31) as f32) * 0.05;
                    let bright = (3..7).contains(&x) && (3..7).contains(&y);
                    data[(b * 3 + c) * w * h + y * w + x] = if bright { 0.9 } else { 0.05 + noise };
                }
            }
        }
        labels.push(vec![BoxLabel { class: 0, x: 5.0 / 16.0, y: 5.0 / 16.0, w: 4.0 / 16.0, h: 4.0 / 16.0 }]);
    }
    (data, labels)
}

#[test]
fn tiny_net_learns_one_box() {
    let cfg = Cfg::parse(CFG).unwrap();
    let batch = 3;
    let mut net = TrainNet::new(&cfg, batch, 4).unwrap();
    let (input, labels) = sample(batch);
    net.set_input(&input).unwrap();
    let mut losses = vec![];
    for iteration in 1..=1000 {
        net.zero_grads();
        net.forward().unwrap();
        let loss = net.loss_and_deltas(&labels).unwrap();
        net.backward();
        let lr = hip_darknet::train::learning_rate(&cfg.train, iteration - 1);
        net.update(lr);
        losses.push(loss);
        assert!(loss.is_finite(), "iteration {iteration}: loss {loss}");
    }
    let first = losses[..10].iter().sum::<f32>() / 10.0;
    let last = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
    println!("loss {first:.4} -> {last:.4}");
    assert!(last < first * 0.2, "loss should fall by at least 5x: {first} -> {last}");
    // The class head must have learned the matched anchor (an objectness assertion does not work
    // here: the toy net shares one filter across all 256 cells with a single positive, so the
    // objectness gradient is dominated by the 255 negatives — Darknet behaves the same way).
    let yolo = net.model.yolo_layers()[0];
    let shape = net.model.layers[yolo].out_shape;
    let plane = shape.w * shape.h;
    let cell = 5 * shape.w + 5;
    net.forward().unwrap();
    let cls = net.outs[yolo][5 * plane + cell];
    println!("responsible anchor class probability after training: {cls:.4}");
    assert!(cls > 0.9, "the matched anchor's class never learned: {cls}");
}

#[test]
fn training_is_deterministic_for_a_seed() {
    let cfg = Cfg::parse(CFG).unwrap();
    let (input, labels) = sample(3);
    let run = |seed: u64| {
        let mut net = TrainNet::new(&cfg, 3, seed).unwrap();
        net.set_input(&input).unwrap();
        let mut losses = vec![];
        for _ in 0..5 {
            net.zero_grads();
            net.forward().unwrap();
            losses.push(net.loss_and_deltas(&labels).unwrap());
            net.backward();
            net.update(0.01);
        }
        losses
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}
