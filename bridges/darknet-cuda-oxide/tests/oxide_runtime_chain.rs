//! Runtime-level checks: `GpuExecutableNetwork<OxideBackend>` (real Rust GPU
//! kernels) against the Rust CPU `ExecutableNetwork` on tiny nets.
//!
//! * `chained_*`: training forward -> seeded backward on conv + activation
//!   stacks (leaky / relu6 / ramp / mish / swish, a mismatched-shape shortcut):
//!   every layer output and every weight / bias gradient matches the CPU
//!   backward.
//! * `few_sgd_steps_*`: a few SGD steps (momentum + decay) on the same weights
//!   and the same batch with an L2 loss: loss trajectory and final weights
//!   match the CPU run (no BN: the GPU path uses batch statistics, the CPU
//!   path rolling ones, by design).

use darknet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor, TensorShape};
use darknet_cuda_oxide::OxideBackend;

fn network(cfg: &str) -> Network {
    Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).expect("cfg")).expect("network")
}

fn fill_params(params: &mut [LayerParams]) {
    for (layer, p) in params.iter_mut().enumerate() {
        if let LayerParams::Convolutional(conv) = p {
            for (i, w) in conv.weights.iter_mut().enumerate() {
                *w = ((i + 3 * layer) as f32 * 0.7).sin() * 0.6;
            }
            for (i, b) in conv.biases.iter_mut().enumerate() {
                *b = ((i + layer) as f32 * 1.3).cos() * 0.2;
            }
        }
    }
}

fn patterned(shape: TensorShape) -> Tensor {
    let data = (0..shape.batch * shape.element_count())
        .map(|i| (i as f32 * 0.37).sin() + 0.1 * (i % 5) as f32)
        .collect();
    Tensor::from_data(shape, data).expect("tensor")
}

fn close(name: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length");
    let scale = a.iter().chain(b).fold(1.0f32, |m, v| m.max(v.abs()));
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol * scale, "{name}[{i}]: cpu {x} vs oxide {y} (scale {scale})");
    }
}

fn conv_stack(act: &str) -> String {
    format!(
        "[net]\nwidth=6\nheight=6\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation={act}\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation={act}\n\
         [convolutional]\nfilters=3\nsize=3\nstride=2\npad=1\nactivation=linear\n"
    )
}

/// Same weights, same input, same seed on the last layer: layer outputs and
/// weight / bias gradients agree.
fn assert_chain(cfg: &str, seed_layer: usize, what: &str) {
    let net = network(cfg);
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");

    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("cpu forward");
    let gpu_fwd = gpu.forward_training_with_state(&input).expect("oxide forward");
    for (i, (a, b)) in cpu_fwd.layer_outputs.iter().zip(&gpu_fwd.cpu_forward.layer_outputs).enumerate() {
        close(&format!("{what} layer {i} output"), &a.data, &b.data, 1e-4);
    }

    let shape = cpu.network.layers[seed_layer].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count()).map(|i| (i as f32 * 0.91).cos()).collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[seed_layer] = Some(seed);
    cpu.zero_train_buffers();
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds).expect("cpu backward");
    gpu.zero_train_buffers_gpu().expect("zero");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds).expect("oxide backward");

    // Gradients are read through one SGD step with lr 1, no momentum / decay:
    // w_new = w_old + updates (Darknet adds the accumulated delta products).
    let before = cpu.params.clone();
    cpu.update_weights(1.0, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1.0, 0.0, 0.0, 1).expect("oxide update");
    let mut checked = 0;
    for (i, (b, (c, g))) in before.iter().zip(cpu.params.iter().zip(&gpu.params)).enumerate() {
        let (LayerParams::Convolutional(b), LayerParams::Convolutional(c), LayerParams::Convolutional(g)) = (b, c, g)
        else {
            continue;
        };
        let delta = |a: &[f32], o: &[f32]| a.iter().zip(o).map(|(x, y)| x - y).collect::<Vec<f32>>();
        let (cw, gw) = (delta(&c.weights, &b.weights), delta(&g.weights, &b.weights));
        assert!(cw.iter().any(|v| v.abs() > 1e-6), "{what} layer {i}: CPU weight gradient is all zero");
        close(&format!("{what} layer {i} weight gradient"), &cw, &gw, 1e-3);
        close(&format!("{what} layer {i} bias gradient"), &delta(&c.biases, &b.biases), &delta(&g.biases, &b.biases), 1e-3);
        checked += 1;
    }
    assert!(checked >= 2, "{what}: expected several convolution layers");
}

#[test]
fn chained_conv_activation_stacks_match_cpu_backward() {
    for act in ["leaky", "relu6", "ramp", "mish", "swish", "logistic", "relu"] {
        assert_chain(&conv_stack(act), 2, act);
    }
}

#[test]
fn chained_mismatched_shortcuts_match_cpu_backward() {
    // source (4 channels) larger than the layer (2 channels): cropped
    assert_chain(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=leaky\n\
         [convolutional]\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n\
         [shortcut]\nfrom=-2\nactivation=relu6\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        3,
        "shortcut crop",
    );
    // source (2 channels) smaller than the layer (4 channels): zero-padded
    assert_chain(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
         [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [shortcut]\nfrom=-2\nactivation=ramp\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        3,
        "shortcut pad",
    );
}

#[test]
fn few_sgd_steps_match_cpu_loss_trajectory_and_weights() {
    let cfg = "[net]\nwidth=6\nheight=6\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=relu6\n\
         [shortcut]\nfrom=-2\nactivation=ramp\n\
         [convolutional]\nfilters=3\nsize=3\nstride=2\npad=1\nactivation=linear\n";
    let net = network(cfg);
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");

    let input = patterned(cpu.network.net.input_shape());
    let out_shape = cpu.network.layers[3].output_shape;
    let target: Vec<f32> = (0..out_shape.batch * out_shape.element_count()).map(|i| 0.03 * i as f32 - 0.5).collect();
    let (lr, momentum, decay, batch) = (2.0e-4f32, 0.9f32, 5.0e-4f32, 2usize);
    let seed_and_loss = |out: &Tensor| -> (Tensor, f32) {
        let diff: Vec<f32> = target.iter().zip(&out.data).map(|(t, o)| t - o).collect();
        let loss = 0.5 * diff.iter().map(|d| d * d).sum::<f32>();
        (Tensor::from_data(out.shape, diff).expect("seed"), loss)
    };
    let (mut cpu_loss, mut gpu_loss) = (vec![], vec![]);
    for _ in 0..8 {
        cpu.zero_train_buffers();
        let fwd = cpu.forward_training(&input).expect("cpu forward");
        let (seed, loss) = seed_and_loss(fwd.final_output().expect("output"));
        cpu_loss.push(loss);
        let mut seeds = vec![None; 4];
        seeds[3] = Some(seed);
        cpu.backward_accumulate_seeded(&input, &fwd, &seeds).expect("cpu backward");
        cpu.update_weights(lr, momentum, decay, batch);

        gpu.zero_train_buffers_gpu().expect("zero");
        let fwd = gpu.forward_training_with_state(&input).expect("oxide forward");
        let (seed, loss) = seed_and_loss(fwd.cpu_forward.final_output().expect("output"));
        gpu_loss.push(loss);
        let mut seeds = vec![None; 4];
        seeds[3] = Some(seed);
        gpu.backward_accumulate_seeded_gpu(&fwd, &seeds).expect("oxide backward");
        gpu.update_weights_gpu(lr, momentum, decay, batch).expect("oxide update");
    }
    println!("loss trajectory cpu  {cpu_loss:?}\nloss trajectory oxide {gpu_loss:?}");
    for (step, (c, g)) in cpu_loss.iter().zip(&gpu_loss).enumerate() {
        assert!((c - g).abs() <= 1e-3 * c.abs().max(1.0), "step {step}: cpu loss {c} vs oxide {g}");
    }
    assert!(cpu_loss.last() < cpu_loss.first(), "loss must go down: {cpu_loss:?}");
    for (i, (c, g)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(c), LayerParams::Convolutional(g)) = (c, g) {
            close(&format!("layer {i} weights"), &c.weights, &g.weights, 1e-3);
            close(&format!("layer {i} biases"), &c.biases, &g.biases, 1e-3);
        }
    }
}
