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

use darknet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, LayerTrainBuffers, Network,
    Tensor, TensorShape,
};
use darknet_cuda::F32TensorOpsBackend;
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
    for act in [
        "leaky",
        "loggy",
        "hardtan",
        "relie",
        "lhtan",
        "plse",
        "stair",
        "relu6",
        "ramp",
        "mish",
        "swish",
        "logistic",
        "relu",
        "elu",
        "selu",
        "hard_mish",
    ] {
        assert_chain(&conv_stack(act), 2, act);
    }
}

#[test]
fn maxpool_zero_nonmax_forward_backward_and_sgd_match_cpu() {
    let net = network(
        "[net]\nwidth=3\nheight=3\nchannels=1\nbatch=1\n\
         [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
         [maxpool]\nsize=2\nstride=1\npadding=0\nmaxpool_zero_nonmax=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(cpu_conv) = &mut cpu.params[0] else {
        panic!("convolution parameters");
    };
    cpu_conv.weights[0] = 1.0;
    cpu_conv.biases[0] = 0.0;
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("device train buffers");

    let input = Tensor::from_data(
        TensorShape::new(3, 3, 1, 1),
        vec![1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
    )
    .expect("input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide forward");
    for (index, (cpu_output, gpu_output)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&gpu_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(
            &format!("maxpool layer {index} output"),
            &cpu_output.data,
            &gpu_output.data,
            1e-6,
        );
    }
    assert_eq!(cpu_forward.layer_outputs[1].data, vec![3.0, 3.0, 3.0, 3.0]);

    let seed = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0])
        .expect("maxpool seed");
    let mut seeds = vec![None; 2];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero device buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide backward");
    cpu.update_weights(1.0, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1.0, 0.0, 0.0, 1)
        .expect("Oxide SGD");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("updated convolution parameters");
    };
    close("maxpool zero_nonmax conv weights", &cpu_conv.weights, &gpu_conv.weights, 1e-6);
    close("maxpool zero_nonmax conv biases", &cpu_conv.biases, &gpu_conv.biases, 1e-6);
    assert_eq!(cpu_conv.weights, vec![31.0]);
    assert_eq!(cpu_conv.biases, vec![10.0]);
}

#[test]
fn weighted_shortcut_runtime_matches_cpu_forward_backward_and_sgd() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
         [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=relu\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let LayerParams::Shortcut(weights) = &mut cpu.params[2] else {
        panic!("weighted shortcut parameters expected");
    };
    weights.weights = vec![0.8, 0.5, 0.6, 0.9];
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");

    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("cpu forward");
    let gpu_fwd = gpu.forward_training_device(&input).expect("oxide forward");
    let oxide_output = gpu
        .backend
        .download_f32(&gpu_fwd.gpu_layer_outputs[2])
        .expect("download output");
    close(
        "weighted shortcut forward",
        &cpu_fwd.layer_outputs[2].data,
        &oxide_output,
        1e-4,
    );

    let shape = cpu.network.layers[2].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.53).cos())
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[2] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero gradients");
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("cpu backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("oxide device backward");
    cpu.update_weights(0.01, 0.0, 0.0, 1);
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("update weights");

    let LayerParams::Shortcut(cpu_weights) = &cpu.params[2] else {
        unreachable!()
    };
    let LayerParams::Shortcut(gpu_weights) = &gpu.params[2] else {
        unreachable!()
    };
    close(
        "weighted shortcut SGD",
        &cpu_weights.weights,
        &gpu_weights.weights,
        1e-6,
    );
    for (index, (cpu_param, gpu_param)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (cpu_param, gpu_param)
        {
            close(
                &format!("conv {index} SGD weights"),
                &cpu_conv.weights,
                &gpu_conv.weights,
                1e-3,
            );
            close(
                &format!("conv {index} SGD biases"),
                &cpu_conv.biases,
                &gpu_conv.biases,
                1e-3,
            );
        }
    }
}

fn expected_first_adam_step(weights: &[f32], updates: &[f32], lr: f32, eps: f32) -> Vec<f32> {
    weights
        .iter()
        .zip(updates)
        .map(|(weight, update)| weight + lr * update / (update.abs() + eps))
        .collect()
}

#[test]
fn moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients() {
    let net = network(
        "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=2\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n\
         [moe_convolutional]\nfilters=1\nsize=1\nbatch_normalize=0\nexperts=2\n\
         top_k=0\nload_balance=0\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    let LayerParams::Moe(moe) = &mut gpu.params[0] else {
        panic!("MoE parameters")
    };
    moe.router_weights.copy_from_slice(&[0.15, -0.25]);
    moe.router_biases.copy_from_slice(&[0.03, -0.02]);
    moe.experts[0].weights[0] = 0.25;
    moe.experts[0].biases[0] = 0.02;
    moe.experts[1].weights[0] = -0.35;
    moe.experts[1].biases[0] = -0.04;
    cpu.params = gpu.params.clone();
    gpu.init_train_buffers_gpu().expect("device train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("CPU MoE forward");
    let gpu_fwd = gpu
        .forward_training_device(&input)
        .expect("Oxide MoE forward");
    close(
        "MoE Adam forward",
        &cpu_fwd.layer_outputs[0].data,
        &gpu.backend
            .download_f32(&gpu_fwd.gpu_layer_outputs[0])
            .expect("download MoE output"),
        1e-4,
    );
    let shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.41).sin() - 0.15)
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; 1];
    seeds[0] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("zero device train buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("CPU MoE backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("Oxide MoE backward");

    let LayerTrainBuffers::Moe(cpu_updates) = &cpu.train_buffers[0] else {
        panic!("CPU MoE train buffers");
    };
    let LayerParams::Moe(cpu_params) = &cpu.params[0] else {
        panic!("CPU MoE parameters");
    };
    let lr = 0.01;
    let eps = cpu.network.net.adam_eps;
    let expected_router_weights = expected_first_adam_step(
        &cpu_params.router_weights,
        &cpu_updates.router_weight_updates,
        lr,
        eps,
    );
    let expected_router_biases = expected_first_adam_step(
        &cpu_params.router_biases,
        &cpu_updates.router_bias_updates,
        lr,
        eps,
    );
    let expected_experts: Vec<_> = cpu_params
        .experts
        .iter()
        .zip(&cpu_updates.experts)
        .map(|(params, updates)| {
            (
                expected_first_adam_step(&params.weights, &updates.weight_updates, lr, eps),
                expected_first_adam_step(&params.biases, &updates.bias_updates, lr, eps),
            )
        })
        .collect();
    gpu.update_weights_gpu_deferred(lr, 0.0, 0.0, 2)
        .expect("deferred Oxide device MoE Adam");
    gpu.sync_host_weights_from_device()
        .expect("sync deferred Oxide MoE Adam");
    let LayerParams::Moe(gpu_params) = &gpu.params[0] else {
        panic!("updated GPU MoE parameters");
    };
    close(
        "MoE Adam router weights",
        &expected_router_weights,
        &gpu_params.router_weights,
        1e-4,
    );
    close(
        "MoE Adam router biases",
        &expected_router_biases,
        &gpu_params.router_biases,
        1e-4,
    );
    for (index, (expert, (weights, biases))) in
        gpu_params.experts.iter().zip(&expected_experts).enumerate()
    {
        close(
            &format!("MoE Adam expert {index} weights"),
            weights,
            &expert.weights,
            1e-4,
        );
        close(
            &format!("MoE Adam expert {index} biases"),
            biases,
            &expert.biases,
            1e-4,
        );
    }
}

#[test]
fn moe_batchnorm_expert_device_backward_matches_cpu_on_oxide() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n\
         [moe_convolutional]\nfilters=2\nsize=1\nexperts=2\n\
         top_k=0\nload_balance=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    let LayerParams::Convolutional(conv) = &mut gpu.params[0] else {
        panic!("conv parameters")
    };
    conv.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    conv.biases.copy_from_slice(&[0.0, 0.0]);
    {
        let LayerParams::Moe(moe) = &mut gpu.params[1] else {
            panic!("MoE parameters")
        };
        moe.router_weights.copy_from_slice(&[0.15, -0.25, -0.05, 0.2]);
        moe.router_biases.copy_from_slice(&[0.03, -0.02]);
        for (slot, expert) in moe.experts.iter_mut().enumerate() {
            expert.weights.copy_from_slice(&[
                0.25 + slot as f32 * 0.1,
                -0.1,
                0.05,
                0.35 - slot as f32 * 0.08,
            ]);
            expert.biases.copy_from_slice(&[0.02, -0.04]);
            let bn = expert.batch_norm.as_mut().expect("MoE BN parameters");
            bn.scales.copy_from_slice(&[0.8, 1.1]);
            bn.rolling_mean.copy_from_slice(&[0.1, -0.2]);
            bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);
        }
    }
    cpu.params = gpu.params.clone();
    gpu.init_train_buffers_gpu().expect("device train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("CPU MoE forward");
    let gpu_fwd = gpu
        .forward_training_device(&input)
        .expect("Oxide MoE forward");
    for layer_index in 0..2 {
        close(
            &format!("MoE BN forward layer {layer_index}"),
            &cpu_fwd.layer_outputs[layer_index].data,
            &gpu.backend
                .download_f32(&gpu_fwd.gpu_layer_outputs[layer_index])
                .expect("download MoE BN forward"),
            1e-4,
        );
    }
    let shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; 2];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero train buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("CPU MoE BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("Oxide MoE BN backward");
    cpu.update_weights(0.01, 0.0, 0.0, 2);
    gpu.update_weights_gpu_deferred(0.01, 0.0, 0.0, 2)
        .expect("deferred Oxide MoE BN SGD");
    gpu.sync_host_weights_from_device()
        .expect("sync deferred Oxide MoE BN SGD");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("conv parameters after BN update")
    };
    close("MoE BN preceding conv weights", &cpu_conv.weights, &gpu_conv.weights, 1e-4);
    close("MoE BN preceding conv biases", &cpu_conv.biases, &gpu_conv.biases, 1e-4);
    let (LayerParams::Moe(cpu_moe), LayerParams::Moe(gpu_moe)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("MoE parameters after BN update")
    };
    close("MoE BN router weights", &cpu_moe.router_weights, &gpu_moe.router_weights, 1e-4);
    close("MoE BN router biases", &cpu_moe.router_biases, &gpu_moe.router_biases, 1e-4);
    for (index, (cpu_expert, gpu_expert)) in cpu_moe.experts.iter().zip(&gpu_moe.experts).enumerate() {
        close(
            &format!("MoE BN expert {index} weights"),
            &cpu_expert.weights,
            &gpu_expert.weights,
            1e-4,
        );
        close(
            &format!("MoE BN expert {index} biases"),
            &cpu_expert.biases,
            &gpu_expert.biases,
            1e-4,
        );
        close(
            &format!("MoE BN expert {index} scales"),
            &cpu_expert.batch_norm.as_ref().unwrap().scales,
            &gpu_expert.batch_norm.as_ref().unwrap().scales,
            1e-4,
        );
    }
}

#[test]
fn moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n[moe_convolutional]\nfilters=2\nsize=1\nexperts=2\ntop_k=0\nload_balance=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    let LayerParams::Convolutional(conv) = &mut gpu.params[0] else {
        panic!("conv parameters")
    };
    conv.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    conv.biases.copy_from_slice(&[0.0, 0.0]);
    {
        let LayerParams::Moe(moe) = &mut gpu.params[1] else {
            panic!("MoE parameters")
        };
        moe.router_weights.copy_from_slice(&[0.15, -0.25, -0.05, 0.2]);
        moe.router_biases.copy_from_slice(&[0.03, -0.02]);
        for (slot, expert) in moe.experts.iter_mut().enumerate() {
            expert.weights.copy_from_slice(&[
                0.25 + slot as f32 * 0.1,
                -0.1,
                0.05,
                0.35 - slot as f32 * 0.08,
            ]);
            expert.biases.copy_from_slice(&[0.02, -0.04]);
            let bn = expert.batch_norm.as_mut().expect("MoE BN parameters");
            bn.scales.copy_from_slice(&[0.8, 1.1]);
            bn.rolling_mean.copy_from_slice(&[0.1, -0.2]);
            bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);
        }
    }
    cpu.params.clone_from(&gpu.params);
    gpu.init_train_buffers_gpu().expect("device train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("CPU forward");
    let gpu_fwd = gpu
        .forward_training_device(&input)
        .expect("Oxide forward");
    for layer_index in 0..2 {
        close(
            &format!("MoE BN Adam forward {layer_index}"),
            &cpu_fwd.layer_outputs[layer_index].data,
            &gpu.backend
                .download_f32(&gpu_fwd.gpu_layer_outputs[layer_index])
                .expect("download MoE BN Adam output"),
            1e-4,
        );
    }
    let shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.41).sin() - 0.15)
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; 2];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero train buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("CPU MoE BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("Oxide MoE BN backward");

    let lr = 0.01;
    let eps = cpu.network.net.adam_eps;
    let LayerTrainBuffers::Convolutional(cpu_conv_updates) = &cpu.train_buffers[0] else {
        panic!("CPU conv gradients")
    };
    let LayerParams::Convolutional(cpu_conv_params) = &cpu.params[0] else {
        panic!("CPU conv params")
    };
    let expected_conv_weights = expected_first_adam_step(
        &cpu_conv_params.weights,
        &cpu_conv_updates.weight_updates,
        lr,
        eps,
    );
    let expected_conv_biases = expected_first_adam_step(
        &cpu_conv_params.biases,
        &cpu_conv_updates.bias_updates,
        lr,
        eps,
    );
    let (LayerParams::Moe(cpu_moe_params), LayerTrainBuffers::Moe(cpu_moe_updates)) =
        (&cpu.params[1], &cpu.train_buffers[1])
    else {
        panic!("CPU MoE params/gradients")
    };
    let expected_router_weights = expected_first_adam_step(
        &cpu_moe_params.router_weights,
        &cpu_moe_updates.router_weight_updates,
        lr,
        eps,
    );
    let expected_router_biases = expected_first_adam_step(
        &cpu_moe_params.router_biases,
        &cpu_moe_updates.router_bias_updates,
        lr,
        eps,
    );
    let expected_experts: Vec<_> = cpu_moe_params
        .experts
        .iter()
        .zip(&cpu_moe_updates.experts)
        .map(|(params, updates)| {
            (
                expected_first_adam_step(&params.weights, &updates.weight_updates, lr, eps),
                expected_first_adam_step(&params.biases, &updates.bias_updates, lr, eps),
                expected_first_adam_step(
                    &params.batch_norm.as_ref().unwrap().scales,
                    &updates.batch_norm.as_ref().unwrap().scale_updates,
                    lr,
                    eps,
                ),
            )
        })
        .collect();
    gpu.update_weights_gpu_deferred(lr, 0.0, 0.0, 2)
        .expect("deferred Oxide BN MoE Adam");
    gpu.sync_host_weights_from_device()
        .expect("sync deferred BN MoE Adam parameters");

    let (LayerParams::Convolutional(gpu_conv), LayerParams::Moe(gpu_moe)) =
        (&gpu.params[0], &gpu.params[1])
    else {
        panic!("updated GPU parameters")
    };
    close("BN Adam preceding conv weights", &expected_conv_weights, &gpu_conv.weights, 1e-4);
    close("BN Adam preceding conv biases", &expected_conv_biases, &gpu_conv.biases, 1e-4);
    close("BN Adam router weights", &expected_router_weights, &gpu_moe.router_weights, 1e-4);
    close("BN Adam router biases", &expected_router_biases, &gpu_moe.router_biases, 1e-4);
    for (index, (expert, (weights, biases, scales))) in
        gpu_moe.experts.iter().zip(&expected_experts).enumerate()
    {
        close(&format!("BN Adam expert {index} weights"), weights, &expert.weights, 1e-4);
        close(&format!("BN Adam expert {index} biases"), biases, &expert.biases, 1e-4);
        close(
            &format!("BN Adam expert {index} scales"),
            scales,
            &expert.batch_norm.as_ref().unwrap().scales,
            1e-4,
        );
    }
}

#[test]
fn moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n\
         [moe_convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nexperts=3\n\
         top_k=0\nload_balance=0.05\ntemperature=1.4\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    let LayerParams::Convolutional(conv) = &mut gpu.params[0] else { panic!("conv parameters") };
    conv.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    conv.biases.copy_from_slice(&[0.0, 0.0]);
    {
        let LayerParams::Moe(moe) = &mut gpu.params[1] else { panic!("MoE parameters") };
        for (i, weight) in moe.router_weights.iter_mut().enumerate() {
            *weight = (i as f32 - 2.0) * 0.07;
        }
        moe.router_biases.copy_from_slice(&[0.2, -0.1, 0.05]);
        for (slot, expert) in moe.experts.iter_mut().enumerate() {
            for (i, weight) in expert.weights.iter_mut().enumerate() {
                *weight = 0.08 * (1 + slot * 4 + i) as f32;
            }
            expert.biases.copy_from_slice(&[slot as f32 * 0.03, -0.02]);
        }
    }
    cpu.params = gpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("CPU MoE forward");
    let gpu_fwd = gpu.forward_training_device(&input).expect("Oxide MoE forward");
    let output = gpu.backend.download_f32(&gpu_fwd.gpu_layer_outputs[1]).expect("download MoE output");
    close("MoE expert blend", &cpu_fwd.layer_outputs[1].data, &output, 1e-4);
    assert!(gpu_fwd.cpu_forward.moe_caches[1].is_none());

    let shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; 2];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero train buffers");
    for _ in 0..2 {
        cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
            .expect("CPU MoE backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
            .expect("Oxide MoE backward");
    }
    cpu.update_weights(0.01, 0.9, 0.001, 4);
    gpu.update_weights_gpu_deferred(0.01, 0.9, 0.001, 4)
        .expect("MoE SGD");

    let cpu_fwd = cpu
        .forward_training(&input)
        .expect("second CPU MoE forward");
    let gpu_fwd = gpu
        .forward_training_device(&input)
        .expect("second Oxide MoE forward");
    let second_output = gpu
        .backend
        .download_f32(&gpu_fwd.gpu_layer_outputs[1])
        .expect("download second MoE output");
    close(
        "second forward with stale host params",
        &cpu_fwd.layer_outputs[1].data,
        &second_output,
        1e-4,
    );
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("second CPU MoE backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("second Oxide MoE backward");
    cpu.update_weights(0.01, 0.9, 0.001, 2);
    gpu.update_weights_gpu_deferred(0.01, 0.9, 0.001, 2)
        .expect("second MoE SGD");
    gpu.sync_host_weights_from_device()
        .expect("sync second deferred MoE weights");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("conv parameters after second MoE backward");
    };
    close(
        "second-step conv weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        1e-4,
    );
    close(
        "second-step conv biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        1e-4,
    );
    let LayerParams::Moe(cpu_moe) = &cpu.params[1] else {
        unreachable!()
    };
    let LayerParams::Moe(gpu_moe) = &gpu.params[1] else {
        unreachable!()
    };
    close(
        "second-step router weights",
        &cpu_moe.router_weights,
        &gpu_moe.router_weights,
        1e-5,
    );
    close(
        "second-step router biases",
        &cpu_moe.router_biases,
        &gpu_moe.router_biases,
        1e-5,
    );
    for (cpu_expert, gpu_expert) in cpu_moe.experts.iter().zip(&gpu_moe.experts) {
        close(
            "second-step expert weights",
            &cpu_expert.weights,
            &gpu_expert.weights,
            1e-5,
        );
        close(
            "second-step expert biases",
            &cpu_expert.biases,
            &gpu_expert.biases,
            1e-5,
        );
    }
}

#[test]
fn amp_dropout_and_dropblock_runtime_chain_matches_f32_reference() {
    let net = network(
        "[net]\nwidth=8\nheight=8\nchannels=2\nbatch=2\nmax_batches=10\n\
         [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [dropout]\nprobability=0.2\n\
         [dropout]\nprobability=0.2\ndropblock=1\ndropblock_size_abs=3\n\
         [convolutional]\nfilters=4\nsize=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(
        net,
        OxideBackend::new().expect("oxide backend").with_amp(true),
    );
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_fwd = cpu.forward_training(&input).expect("f32 training forward");
    let gpu_fwd = gpu.forward_training_with_state(&input).expect("AMP training forward");

    assert_eq!(cpu_fwd.dropout_masks, gpu_fwd.cpu_forward.dropout_masks);
    for layer_index in [0, 1, 2, 3] {
        let oxide = gpu.backend.download_f32(&gpu_fwd.gpu_layer_outputs[layer_index]).expect("download AMP layer");
        close(&format!("AMP dropout chain layer {layer_index}"), &cpu_fwd.layer_outputs[layer_index].data, &oxide, 8e-3);
    }
    let shape = cpu.network.layers[3].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count()).map(|i| (i as f32 * 0.71).cos()).collect(),
    ).expect("seed");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[3] = Some(seed);
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds).expect("f32 backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds).expect("AMP backward");
    cpu.update_weights(1e-4, 0.0, 0.0, 2);
    gpu.update_weights_gpu(1e-4, 0.0, 0.0, 2).expect("AMP SGD update");
    for (layer_index, (cpu_param, gpu_param)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) = (cpu_param, gpu_param) {
            close(&format!("AMP dropout chain layer {layer_index} weights"), &cpu_conv.weights, &gpu_conv.weights, 2e-2);
            close(&format!("AMP dropout chain layer {layer_index} biases"), &cpu_conv.biases, &gpu_conv.biases, 2e-2);
        }
    }
}

#[test]
fn subdivisions_deferred_gradient_flush_matches_cpu() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=1\nsubdivisions=2\n\
         [convolutional]\nfilters=3\nsize=1\nactivation=leaky\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("train buffers");

    let lr = 1e-4;
    for iteration in 0..2 {
        for subdivision in 0..2 {
            let input = Tensor::from_data(
                cpu.network.net.input_shape(),
                (0..cpu.network.net.input_shape().element_count())
                    .map(|i| ((i + iteration * 17 + subdivision * 7) as f32 * 0.23).sin())
                    .collect(),
            ).expect("input");
            let cpu_fwd = cpu.forward_training(&input).expect("cpu forward");
            let gpu_fwd = gpu.forward_training_with_state(&input).expect("oxide forward");
            let output = gpu.backend.download_f32(&gpu_fwd.gpu_layer_outputs[1]).expect("download output");
            close("subdivision forward", &cpu_fwd.layer_outputs[1].data, &output, 2e-4);
            let shape = cpu.network.layers[1].output_shape;
            let seed = Tensor::from_data(
                shape,
                (0..shape.element_count())
                    .map(|i| ((i + iteration * 13 + subdivision * 5) as f32 * 0.31).cos())
                    .collect(),
            ).expect("seed");
            let mut seeds = vec![None; cpu.network.layers.len()];
            seeds[1] = Some(seed);
            cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds).expect("cpu backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds).expect("oxide backward");
        }
        cpu.update_weights(lr, 0.9, 5e-4, 2);
        gpu.update_weights_gpu_deferred(lr, 0.9, 5e-4, 2).expect("deferred update");
    }
    gpu.sync_host_weights_from_device().expect("sync final weights");
    for (index, (cpu_param, gpu_param)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) = (cpu_param, gpu_param) {
            close(&format!("subdivision layer {index} weights"), &cpu_conv.weights, &gpu_conv.weights, 2e-3);
            close(&format!("subdivision layer {index} biases"), &cpu_conv.biases, &gpu_conv.biases, 2e-3);
        }
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
fn batch_normalized_rnn_device_backward_matches_cpu_on_oxide() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n\
         [connected]\noutput=2\nactivation=linear\n\
         [rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Connected(input) = &mut cpu.params[0] else {
        panic!("upstream connected params");
    };
    input.weights.copy_from_slice(&[0.4, -0.2, 0.1, 0.35]);
    input.biases.copy_from_slice(&[0.03, -0.04]);
    let LayerParams::Rnn(rnn) = &mut cpu.params[1] else {
        panic!("RNN params");
    };
    for (slot, projection) in [&mut rnn.input, &mut rnn.self_params, &mut rnn.output]
        .into_iter()
        .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.04 + (slot * 5 + index) as f32 * 0.009;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 1.0) * 0.03 + index as f32 * 0.02;
        }
        let bn = projection.batch_norm.as_mut().expect("RNN BN params");
        for feature in 0..bn.scales.len() {
            bn.scales[feature] = 0.75 + feature as f32 * 0.16;
            bn.rolling_mean[feature] = -0.12 + feature as f32 * 0.07;
            bn.rolling_variance[feature] = 0.55 + feature as f32 * 0.19;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("GPU BN RNN train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU BN RNN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("Oxide BN RNN forward");
    for index in 0..2 {
        close(
            &format!("BN RNN layer {index} output"),
            &cpu_forward.layer_outputs[index].data,
            &gpu_forward.cpu_forward.layer_outputs[index].data,
            1e-4,
        );
    }

    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.43).cos() * 0.6)
            .collect(),
    )
    .expect("RNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero GPU BN RNN buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU BN RNN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide BN RNN backward");
    cpu.update_weights(0.025, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.025, 0.0, 0.0, 2)
        .expect("Oxide BN RNN optimizer step");

    let (LayerParams::Connected(cpu_input), LayerParams::Connected(gpu_input)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("updated upstream connected params");
    };
    close("BN RNN upstream weights", &cpu_input.weights, &gpu_input.weights, 3e-4);
    close("BN RNN upstream biases", &cpu_input.biases, &gpu_input.biases, 3e-4);
    let (LayerParams::Rnn(cpu_rnn), LayerParams::Rnn(gpu_rnn)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("updated RNN params");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("input", &cpu_rnn.input, &gpu_rnn.input),
        ("recurrent", &cpu_rnn.self_params, &gpu_rnn.self_params),
        ("output", &cpu_rnn.output, &gpu_rnn.output),
    ] {
        close(
            &format!("BN RNN {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            3e-4,
        );
        close(
            &format!("BN RNN {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            3e-4,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        close(
            &format!("BN RNN {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            3e-4,
        );
        close(
            &format!("BN RNN {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            3e-4,
        );
        close(
            &format!("BN RNN {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            3e-4,
        );
    }
}

#[test]
fn rnn_recurrent_projections_and_training_backward_match_cpu() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n\
         [rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    if let LayerParams::Rnn(params) = &mut cpu.params[0] {
        for (i, value) in params.input.weights.iter_mut().enumerate() {
            *value = 0.05 + i as f32 * 0.01;
        }
        for (i, value) in params.input.biases.iter_mut().enumerate() {
            *value = 0.01 * (i + 1) as f32;
        }
        for (i, value) in params.self_params.weights.iter_mut().enumerate() {
            *value = 0.03 + i as f32 * 0.005;
        }
        for (i, value) in params.self_params.biases.iter_mut().enumerate() {
            *value = -0.01 * (i + 1) as f32;
        }
        for (i, value) in params.output.weights.iter_mut().enumerate() {
            *value = 0.02 + i as f32 * 0.004;
        }
        for (i, value) in params.output.biases.iter_mut().enumerate() {
            *value = 0.015 * (i + 1) as f32;
        }
    } else {
        panic!("expected RNN parameters");
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward(&input).expect("CPU RNN forward");
    let gpu_forward = gpu.forward_with_state(&input).expect("Oxide RNN forward");
    close(
        "RNN output",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("Oxide output")
            .data,
        2e-4,
    );

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu().expect("RNN train buffers");
    gpu.zero_train_buffers_gpu()
        .expect("clear RNN train buffers");
    let cpu_training = cpu
        .forward_training(&input)
        .expect("CPU RNN training forward");
    let gpu_training = gpu
        .forward_training_device(&input)
        .expect("Oxide RNN training forward");
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|i| (i as f32 * 0.41).cos() * 0.7)
            .collect(),
    )
    .expect("RNN output gradient");
    cpu.backward_accumulate_seeded(&input, &cpu_training, &[Some(seed.clone())])
        .expect("CPU RNN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_training, &[Some(seed)])
        .expect("Oxide RNN backward");
    cpu.update_weights(0.05, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.05, 0.0, 0.0, 2)
        .expect("Oxide RNN SGD step");
    let (LayerParams::Rnn(cpu_params), LayerParams::Rnn(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("expected RNN parameters after training");
    };
    for (name, cpu_values, gpu_values) in [
        (
            "input weights",
            &cpu_params.input.weights,
            &gpu_params.input.weights,
        ),
        (
            "input biases",
            &cpu_params.input.biases,
            &gpu_params.input.biases,
        ),
        (
            "self weights",
            &cpu_params.self_params.weights,
            &gpu_params.self_params.weights,
        ),
        (
            "self biases",
            &cpu_params.self_params.biases,
            &gpu_params.self_params.biases,
        ),
        (
            "output weights",
            &cpu_params.output.weights,
            &gpu_params.output.weights,
        ),
        (
            "output biases",
            &cpu_params.output.biases,
            &gpu_params.output.biases,
        ),
    ] {
        close(&format!("RNN {name}"), cpu_values, gpu_values, 4e-4);
    }
}

#[test]
fn batch_normalized_lstm_device_backward_matches_cpu_on_oxide() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
         [lstm]\noutput=3\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(input_conv) = &mut cpu.params[0] else {
        panic!("upstream convolution parameters");
    };
    for (index, weight) in input_conv.weights.iter_mut().enumerate() {
        *weight = 0.04 + index as f32 * 0.013;
    }
    input_conv.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Lstm(lstm) = &mut cpu.params[1] else {
        panic!("LSTM parameters");
    };
    for (slot, projection) in [
        &mut lstm.wf,
        &mut lstm.wi,
        &mut lstm.wg,
        &mut lstm.wo,
        &mut lstm.uf,
        &mut lstm.ui,
        &mut lstm.ug,
        &mut lstm.uo,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.015 + (slot * 3 + index) as f32 * 0.004;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 3.5) * 0.01 + index as f32 * 0.006;
        }
        let bn = projection.batch_norm.as_mut().expect("LSTM BN params");
        for feature in 0..bn.scales.len() {
            bn.scales[feature] = 0.72 + feature as f32 * 0.13;
            bn.rolling_mean[feature] = -0.08 + feature as f32 * 0.05;
            bn.rolling_variance[feature] = 0.6 + feature as f32 * 0.17;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("LSTM BN train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU LSTM BN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("Oxide LSTM BN forward");
    close(
        "BN LSTM output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        1e-4,
    );
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(darknet_core::runtime::RecurrentForwardCache::Lstm(_)))
    ));

    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.41).cos() * 0.55)
            .collect(),
    )
    .expect("LSTM output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero LSTM BN buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU LSTM BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide LSTM BN backward");
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("Oxide LSTM BN optimizer step");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("upstream convolution parameters after update");
    };
    close("BN LSTM upstream weights", &cpu_conv.weights, &gpu_conv.weights, 3e-4);
    close("BN LSTM upstream biases", &cpu_conv.biases, &gpu_conv.biases, 3e-4);
    let (LayerParams::Lstm(cpu_lstm), LayerParams::Lstm(gpu_lstm)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("LSTM parameters after update");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("wf", &cpu_lstm.wf, &gpu_lstm.wf),
        ("wi", &cpu_lstm.wi, &gpu_lstm.wi),
        ("wg", &cpu_lstm.wg, &gpu_lstm.wg),
        ("wo", &cpu_lstm.wo, &gpu_lstm.wo),
        ("uf", &cpu_lstm.uf, &gpu_lstm.uf),
        ("ui", &cpu_lstm.ui, &gpu_lstm.ui),
        ("ug", &cpu_lstm.ug, &gpu_lstm.ug),
        ("uo", &cpu_lstm.uo, &gpu_lstm.uo),
    ] {
        close(
            &format!("BN LSTM {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            3e-4,
        );
        close(
            &format!("BN LSTM {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            3e-4,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        close(
            &format!("BN LSTM {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            3e-4,
        );
        close(
            &format!("BN LSTM {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            3e-4,
        );
        close(
            &format!("BN LSTM {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            3e-4,
        );
    }
}

#[test]
fn lstm_no_batch_norm_forward_and_device_backward_match_cpu() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
         [lstm]\noutput=3\nbatch_normalize=0\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(input_conv) = &mut cpu.params[0] else {
        panic!("expected input convolution parameters");
    };
    for (index, value) in input_conv.weights.iter_mut().enumerate() {
        *value = 0.04 + index as f32 * 0.013;
    }
    input_conv.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Lstm(params) = &mut cpu.params[1] else {
        panic!("expected LSTM parameters");
    };
    for (slot, projection) in [
        &mut params.wf,
        &mut params.wi,
        &mut params.wg,
        &mut params.wo,
        &mut params.uf,
        &mut params.ui,
        &mut params.ug,
        &mut params.uo,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.015 + (slot * 3 + index) as f32 * 0.004;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 3.5) * 0.01 + index as f32 * 0.006;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward(&input).expect("CPU LSTM forward");
    let gpu_forward = gpu.forward_with_state(&input).expect("Oxide LSTM forward");
    close(
        "LSTM output",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("Oxide output")
            .data,
        3e-4,
    );

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu().expect("LSTM train buffers");
    gpu.zero_train_buffers_gpu()
        .expect("clear LSTM train buffers");
    let cpu_training = cpu.forward_training(&input).expect("CPU LSTM training");
    let gpu_training = gpu
        .forward_training_device(&input)
        .expect("Oxide LSTM training");
    let training_output = gpu
        .backend
        .download_f32(&gpu_training.gpu_layer_outputs[1])
        .expect("download LSTM device-training output");
    close(
        "LSTM device-training output",
        &cpu_training.layer_outputs[1].data,
        &training_output,
        3e-4,
    );
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.43).cos() * 0.6)
            .collect(),
    )
    .expect("LSTM output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_training, &seeds)
        .expect("CPU LSTM backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_training, &seeds)
        .expect("LSTM device backward");
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("LSTM optimizer step");
    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("expected input convolution parameters after training");
    };
    close(
        "LSTM upstream convolution weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        3e-4,
    );
    close(
        "LSTM upstream convolution biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        3e-4,
    );
    let (LayerParams::Lstm(cpu_params), LayerParams::Lstm(gpu_params)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("expected LSTM parameters after training");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("wf", &cpu_params.wf, &gpu_params.wf),
        ("wi", &cpu_params.wi, &gpu_params.wi),
        ("wg", &cpu_params.wg, &gpu_params.wg),
        ("wo", &cpu_params.wo, &gpu_params.wo),
        ("uf", &cpu_params.uf, &gpu_params.uf),
        ("ui", &cpu_params.ui, &gpu_params.ui),
        ("ug", &cpu_params.ug, &gpu_params.ug),
        ("uo", &cpu_params.uo, &gpu_params.uo),
    ] {
        close(
            &format!("LSTM {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            3e-4,
        );
        close(
            &format!("LSTM {name} biases"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            3e-4,
        );
    }
}

#[test]
fn batch_normalized_crnn_device_backward_matches_cpu_on_oxide() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\ntime_steps=2\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
         [crnn]\nhidden=3\noutput=2\nsize=3\nstride=1\npad=1\nbatch_normalize=1\nactivation=tanh\nshortcut=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(input_conv) = &mut cpu.params[0] else {
        panic!("upstream convolution parameters");
    };
    for (index, weight) in input_conv.weights.iter_mut().enumerate() {
        *weight = 0.04 + index as f32 * 0.013;
    }
    input_conv.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Crnn(crnn) = &mut cpu.params[1] else {
        panic!("CRNN parameters");
    };
    for (slot, projection) in [&mut crnn.input, &mut crnn.self_params, &mut crnn.output]
        .into_iter()
        .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.03 + (slot * 7 + index) as f32 * 0.005;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 1.0) * 0.025 + index as f32 * 0.01;
        }
        let bn = projection.batch_norm.as_mut().expect("CRNN BN params");
        for feature in 0..bn.scales.len() {
            bn.scales[feature] = 0.7 + feature as f32 * 0.14;
            bn.rolling_mean[feature] = -0.1 + feature as f32 * 0.06;
            bn.rolling_variance[feature] = 0.58 + feature as f32 * 0.16;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("CRNN BN train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU CRNN BN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("Oxide CRNN BN forward");
    close(
        "BN CRNN output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        2e-4,
    );
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(darknet_core::runtime::RecurrentForwardCache::Crnn(_)))
    ));

    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.37).cos() * 0.6)
            .collect(),
    )
    .expect("CRNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero CRNN BN buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU CRNN BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide CRNN BN backward");
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("Oxide CRNN BN optimizer step");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("upstream convolution parameters after update");
    };
    close(
        "BN CRNN upstream weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        8e-4,
    );
    close(
        "BN CRNN upstream biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        8e-4,
    );
    let (LayerParams::Crnn(cpu_crnn), LayerParams::Crnn(gpu_crnn)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("CRNN parameters after update");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("input", &cpu_crnn.input, &gpu_crnn.input),
        ("recurrent", &cpu_crnn.self_params, &gpu_crnn.self_params),
        ("output", &cpu_crnn.output, &gpu_crnn.output),
    ] {
        close(
            &format!("BN CRNN {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            8e-4,
        );
        close(
            &format!("BN CRNN {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            8e-4,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        close(
            &format!("BN CRNN {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            8e-4,
        );
        close(
            &format!("BN CRNN {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            8e-4,
        );
        close(
            &format!("BN CRNN {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            8e-4,
        );
    }
}

#[test]
fn crnn_recurrent_convolutions_and_training_backward_match_cpu() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\ntime_steps=2\n\
         [convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
         [crnn]\nhidden=3\noutput=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=tanh\nshortcut=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(input_conv) = &mut cpu.params[0] else {
        panic!("expected input convolution parameters");
    };
    for (index, value) in input_conv.weights.iter_mut().enumerate() {
        *value = 0.04 + index as f32 * 0.013;
    }
    input_conv.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Crnn(params) = &mut cpu.params[1] else {
        panic!("expected CRNN parameters");
    };
    for (slot, nested) in [
        &mut params.input,
        &mut params.self_params,
        &mut params.output,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, value) in nested.weights.iter_mut().enumerate() {
            *value = 0.025 + (slot * 3 + index) as f32 * 0.006;
        }
        for (index, value) in nested.biases.iter_mut().enumerate() {
            *value = (slot as f32 - 1.0) * 0.02 + index as f32 * 0.007;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward(&input).expect("CPU CRNN forward");
    let gpu_forward = gpu.forward_with_state(&input).expect("Oxide CRNN forward");
    close(
        "CRNN output",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("Oxide output")
            .data,
        2e-4,
    );

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu().expect("CRNN train buffers");
    gpu.zero_train_buffers_gpu()
        .expect("clear CRNN train buffers");
    let cpu_training = cpu.forward_training(&input).expect("CPU CRNN training");
    let gpu_training = gpu
        .forward_training_device(&input)
        .expect("Oxide CRNN training");
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.37).cos() * 0.8)
            .collect(),
    )
    .expect("CRNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_training, &seeds)
        .expect("CPU CRNN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_training, &seeds)
        .expect("Oxide CRNN backward");
    cpu.update_weights(0.03, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.03, 0.0, 0.0, 2)
        .expect("Oxide CRNN SGD");
    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("expected input convolution parameters after training");
    };
    close(
        "CRNN upstream conv weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        8e-4,
    );
    close(
        "CRNN upstream conv biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        8e-4,
    );
    let (LayerParams::Crnn(cpu_params), LayerParams::Crnn(gpu_params)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("expected CRNN parameters after training");
    };
    for (name, cpu_conv, gpu_conv) in [
        ("input", &cpu_params.input, &gpu_params.input),
        (
            "recurrent",
            &cpu_params.self_params,
            &gpu_params.self_params,
        ),
        ("output", &cpu_params.output, &gpu_params.output),
    ] {
        close(
            &format!("CRNN {name} weights"),
            &cpu_conv.weights,
            &gpu_conv.weights,
            8e-4,
        );
        close(
            &format!("CRNN {name} biases"),
            &cpu_conv.biases,
            &gpu_conv.biases,
            8e-4,
        );
    }
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
