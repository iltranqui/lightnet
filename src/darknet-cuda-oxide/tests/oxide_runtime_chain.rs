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
        assert!(
            (x - y).abs() <= tol * scale,
            "{name}[{i}]: cpu {x} vs oxide {y} (scale {scale})"
        );
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
    let gpu_fwd = gpu
        .forward_training_with_state(&input)
        .expect("oxide forward");
    let compact = gpu.forward_compact(&input).expect("oxide compact forward");
    close(
        &format!("{what} compact final output"),
        &cpu_fwd.final_output().expect("CPU output").data,
        &compact.final_output.expect("compact output").data,
        1e-4,
    );
    for (i, (a, b)) in cpu_fwd
        .layer_outputs
        .iter()
        .zip(&gpu_fwd.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(&format!("{what} layer {i} output"), &a.data, &b.data, 1e-4);
    }

    let shape = cpu.network.layers[seed_layer].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.91).cos())
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[seed_layer] = Some(seed);
    cpu.zero_train_buffers();
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("cpu backward");
    gpu.zero_train_buffers_gpu().expect("zero");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("oxide backward");

    // Gradients are read through one SGD step with lr 1, no momentum / decay:
    // w_new = w_old + updates (Darknet adds the accumulated delta products).
    let before = cpu.params.clone();
    cpu.update_weights(1.0, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1.0, 0.0, 0.0, 1)
        .expect("oxide update");
    let mut checked = 0;
    for (i, (b, (c, g))) in before
        .iter()
        .zip(cpu.params.iter().zip(&gpu.params))
        .enumerate()
    {
        let (
            LayerParams::Convolutional(b),
            LayerParams::Convolutional(c),
            LayerParams::Convolutional(g),
        ) = (b, c, g)
        else {
            continue;
        };
        let delta =
            |a: &[f32], o: &[f32]| a.iter().zip(o).map(|(x, y)| x - y).collect::<Vec<f32>>();
        let (cw, gw) = (delta(&c.weights, &b.weights), delta(&g.weights, &b.weights));
        assert!(
            cw.iter().any(|v| v.abs() > 1e-6),
            "{what} layer {i}: CPU weight gradient is all zero"
        );
        close(&format!("{what} layer {i} weight gradient"), &cw, &gw, 1e-3);
        close(
            &format!("{what} layer {i} bias gradient"),
            &delta(&c.biases, &b.biases),
            &delta(&g.biases, &b.biases),
            1e-3,
        );
        checked += 1;
    }
    assert!(checked >= 2, "{what}: expected several convolution layers");
}

#[test]
fn marker_gather_native_full_compact_backward_and_upstream_sgd_match_cpu() {
    assert_chain(
        "[net]\nwidth=5\nheight=1\nchannels=8\nbatch=2\n\
         [convolutional]\nfilters=8\nsize=1\nactivation=linear\n\
         [convolutional]\nfilters=8\nsize=1\nactivation=linear\n\
         [laya_marker_gather]\nlayers=1,0\nhidden=8\nmax_options=3\n",
        2,
        "native marker gather",
    );
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
        "normalize_channels",
        "normalize_channels_softmax",
        "normalize_channels_softmax_maxval",
    ] {
        assert_chain(&conv_stack(act), 2, act);
    }
}

#[test]
fn vit_flash_attention_runtime_uses_native_forward_and_backward_and_matches_cpu() {
    let net = network(
        r#"
            [net]
            width=4
            height=4
            channels=1
            batch=1

            [convolutional]
            filters=2
            size=1
            activation=linear

            [vit]
            dim=4
            heads=2
            patch_size=2
            patch_stride=2
            mlp_dim=8
            flash_attention=1
        "#,
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 17);
    let input = patterned(net.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU ViT forward");

    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("ViT train buffers");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide ViT forward");
    for (index, (cpu_output, oxide_output)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&gpu_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(
            &format!("ViT flash chain layer {index}"),
            &cpu_output.data,
            &oxide_output.data,
            2e-4,
        );
    }

    let output = cpu_forward.final_output().expect("ViT output");
    let seed = Tensor::filled(output.shape, 0.1);
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("zero ViT train buffers");
    // The rest of the transformer backward remains on the host fallback; this
    // flash-attention dQ/dK/dV derivative dispatches through Oxide's native kernels.
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU ViT backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide ViT backward fallback");
    cpu.update_weights(1e-4, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1e-4, 0.0, 0.0, 1)
        .expect("Oxide fallback update");

    let (LayerParams::Vit(cpu_vit), LayerParams::Vit(oxide_vit)) = (&cpu.params[1], &gpu.params[1])
    else {
        panic!("ViT parameter layouts");
    };
    close(
        "ViT flash patch gradient/update",
        &cpu_vit.patch_embed,
        &oxide_vit.patch_embed,
        2e-4,
    );
    close(
        "ViT flash QKV gradient/update",
        &cpu_vit.qkv_weight,
        &oxide_vit.qkv_weight,
        2e-4,
    );
    close(
        "ViT flash attention projection update",
        &cpu_vit.wo,
        &oxide_vit.wo,
        2e-4,
    );
}

#[test]
fn vit_flash_attention_unsupported_head_dim_uses_host_reference() {
    let net = network(
        r#"
            [net]
            width=2
            height=2
            channels=1
            batch=1

            [vit]
            dim=129
            heads=1
            patch_size=1
            patch_stride=1
            mlp_dim=129
            flash_attention=1
        "#,
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 23);
    let input = patterned(net.net.input_shape());
    let cpu_forward = cpu.forward(&input).expect("CPU ViT forward");
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    let oxide_forward = gpu
        .forward_with_state(&input)
        .expect("host fallback forward");
    close(
        "ViT flash unsupported head-dim fallback",
        &cpu_forward.final_output().expect("CPU output").data,
        &oxide_forward
            .cpu_forward
            .final_output()
            .expect("Oxide output")
            .data,
        2e-4,
    );
}

#[test]
fn cpp_channel_normalization_shortcut_matches_cpu_runtime() {
    assert_chain(
        r#"
            [net]
            width=2
            height=1
            channels=2
            batch=1

            [convolutional]
            filters=3
            size=1
            activation=normalize_channels_softmax

            [convolutional]
            filters=2
            size=1
            activation=normalize_channels
        "#,
        1,
        "cpp NORM_CHAN",
    );
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
    gpu.update_weights_gpu(1.0, 0.0, 0.0, 1).expect("Oxide SGD");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("updated convolution parameters");
    };
    close(
        "maxpool zero_nonmax conv weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        1e-6,
    );
    close(
        "maxpool zero_nonmax conv biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        1e-6,
    );
    assert_eq!(cpu_conv.weights, vec![31.0]);
    assert_eq!(cpu_conv.biases, vec![10.0]);
}

#[test]
fn maxpool_depth_native_forward_backward_matches_cpu_on_oxide() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=4\nbatch=1\n\
         [maxpool]\nmaxpool_depth=1\nout_channels=2\n",
    );
    let cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.init_train_buffers_gpu().expect("depth-pool train buffers");
    let input = Tensor::from_data(
        TensorShape::new(2, 1, 4, 1),
        vec![1.0, 5.0, 2.0, 5.0, 3.0, 4.0, 2.0, 9.0],
    )
    .expect("depth-pool input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU depth-pool forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide depth-pool forward");
    let backend = OxideBackend::new().expect("Oxide backend");
    let gpu_output = backend
        .download_f32(&gpu_forward.gpu_layer_outputs[0])
        .expect("download depth-pool output");
    close(
        "depth-pool output",
        &gpu_output,
        &cpu_forward.layer_outputs[0].data,
        0.0,
    );
    assert_eq!(cpu_forward.layer_outputs[0].data, vec![3.0, 5.0, 2.0, 9.0]);
    let seed = Tensor::from_data(
        cpu.network.layers[0].output_shape,
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("depth-pool seed");
    let seeds = [Some(seed)];
    let cpu_backward = cpu
        .backward(&input, &cpu_forward, seeds[0].as_ref().unwrap())
        .expect("CPU depth-pool backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide depth-pool runtime backward");
    let input_gpu = backend
        .upload_f32(&[1, 4, 1, 2], &input.data)
        .expect("upload depth-pool input");
    let seed_gpu = backend
        .upload_f32(&[1, 2, 1, 2], &seeds[0].as_ref().unwrap().data)
        .expect("upload depth-pool seed");
    let dx_gpu = backend
        .maxpool_depth_backward_f32(
            &input_gpu,
            &seed_gpu,
            &[1, 4, 1, 2],
            2,
            darknet_cuda::ElementwiseLaunchConfig::default(),
        )
        .expect("Oxide depth-pool input gradient");
    let dx = backend
        .download_f32(&dx_gpu)
        .expect("download depth-pool dx");
    close(
        "depth-pool input gradient",
        &dx,
        &cpu_backward.input_gradient.data,
        0.0,
    );
    assert_eq!(
        cpu_backward.input_gradient.data,
        vec![0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0]
    );
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
        moe.router_weights
            .copy_from_slice(&[0.15, -0.25, -0.05, 0.2]);
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
    close(
        "MoE BN preceding conv weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        1e-4,
    );
    close(
        "MoE BN preceding conv biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        1e-4,
    );
    let (LayerParams::Moe(cpu_moe), LayerParams::Moe(gpu_moe)) = (&cpu.params[1], &gpu.params[1])
    else {
        panic!("MoE parameters after BN update")
    };
    close(
        "MoE BN router weights",
        &cpu_moe.router_weights,
        &gpu_moe.router_weights,
        1e-4,
    );
    close(
        "MoE BN router biases",
        &cpu_moe.router_biases,
        &gpu_moe.router_biases,
        1e-4,
    );
    for (index, (cpu_expert, gpu_expert)) in
        cpu_moe.experts.iter().zip(&gpu_moe.experts).enumerate()
    {
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
        moe.router_weights
            .copy_from_slice(&[0.15, -0.25, -0.05, 0.2]);
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
    let gpu_fwd = gpu.forward_training_device(&input).expect("Oxide forward");
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
    close(
        "BN Adam preceding conv weights",
        &expected_conv_weights,
        &gpu_conv.weights,
        1e-4,
    );
    close(
        "BN Adam preceding conv biases",
        &expected_conv_biases,
        &gpu_conv.biases,
        1e-4,
    );
    close(
        "BN Adam router weights",
        &expected_router_weights,
        &gpu_moe.router_weights,
        1e-4,
    );
    close(
        "BN Adam router biases",
        &expected_router_biases,
        &gpu_moe.router_biases,
        1e-4,
    );
    for (index, (expert, (weights, biases, scales))) in
        gpu_moe.experts.iter().zip(&expected_experts).enumerate()
    {
        close(
            &format!("BN Adam expert {index} weights"),
            weights,
            &expert.weights,
            1e-4,
        );
        close(
            &format!("BN Adam expert {index} biases"),
            biases,
            &expert.biases,
            1e-4,
        );
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
    let LayerParams::Convolutional(conv) = &mut gpu.params[0] else {
        panic!("conv parameters")
    };
    conv.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    conv.biases.copy_from_slice(&[0.0, 0.0]);
    {
        let LayerParams::Moe(moe) = &mut gpu.params[1] else {
            panic!("MoE parameters")
        };
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
    let gpu_fwd = gpu
        .forward_training_device(&input)
        .expect("Oxide MoE forward");
    let output = gpu
        .backend
        .download_f32(&gpu_fwd.gpu_layer_outputs[1])
        .expect("download MoE output");
    close(
        "MoE expert blend",
        &cpu_fwd.layer_outputs[1].data,
        &output,
        1e-4,
    );
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
    let gpu_fwd = gpu
        .forward_training_with_state(&input)
        .expect("AMP training forward");

    assert_eq!(cpu_fwd.dropout_masks, gpu_fwd.cpu_forward.dropout_masks);
    for layer_index in [0, 1, 2, 3] {
        let oxide = gpu
            .backend
            .download_f32(&gpu_fwd.gpu_layer_outputs[layer_index])
            .expect("download AMP layer");
        close(
            &format!("AMP dropout chain layer {layer_index}"),
            &cpu_fwd.layer_outputs[layer_index].data,
            &oxide,
            8e-3,
        );
    }
    let shape = cpu.network.layers[3].output_shape;
    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i as f32 * 0.71).cos())
            .collect(),
    )
    .expect("seed");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[3] = Some(seed);
    cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
        .expect("f32 backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
        .expect("AMP backward");
    cpu.update_weights(1e-4, 0.0, 0.0, 2);
    gpu.update_weights_gpu(1e-4, 0.0, 0.0, 2)
        .expect("AMP SGD update");
    for (layer_index, (cpu_param, gpu_param)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (cpu_param, gpu_param)
        {
            close(
                &format!("AMP dropout chain layer {layer_index} weights"),
                &cpu_conv.weights,
                &gpu_conv.weights,
                2e-2,
            );
            close(
                &format!("AMP dropout chain layer {layer_index} biases"),
                &cpu_conv.biases,
                &gpu_conv.biases,
                2e-2,
            );
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
            )
            .expect("input");
            let cpu_fwd = cpu.forward_training(&input).expect("cpu forward");
            let gpu_fwd = gpu
                .forward_training_with_state(&input)
                .expect("oxide forward");
            let output = gpu
                .backend
                .download_f32(&gpu_fwd.gpu_layer_outputs[1])
                .expect("download output");
            close(
                "subdivision forward",
                &cpu_fwd.layer_outputs[1].data,
                &output,
                2e-4,
            );
            let shape = cpu.network.layers[1].output_shape;
            let seed = Tensor::from_data(
                shape,
                (0..shape.element_count())
                    .map(|i| ((i + iteration * 13 + subdivision * 5) as f32 * 0.31).cos())
                    .collect(),
            )
            .expect("seed");
            let mut seeds = vec![None; cpu.network.layers.len()];
            seeds[1] = Some(seed);
            cpu.backward_accumulate_seeded(&input, &cpu_fwd, &seeds)
                .expect("cpu backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_fwd, &seeds)
                .expect("oxide backward");
        }
        cpu.update_weights(lr, 0.9, 5e-4, 2);
        gpu.update_weights_gpu_deferred(lr, 0.9, 5e-4, 2)
            .expect("deferred update");
    }
    gpu.sync_host_weights_from_device()
        .expect("sync final weights");
    for (index, (cpu_param, gpu_param)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (cpu_param, gpu_param)
        {
            close(
                &format!("subdivision layer {index} weights"),
                &cpu_conv.weights,
                &gpu_conv.weights,
                2e-3,
            );
            close(
                &format!("subdivision layer {index} biases"),
                &cpu_conv.biases,
                &gpu_conv.biases,
                2e-3,
            );
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
fn rnn_native_inference_matches_cpu_with_rolling_batch_norm() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Rnn(params) = &mut cpu.params[0] else { panic!("expected RNN parameters"); };
    for (block_index, block) in [
        &mut params.input, &mut params.self_params, &mut params.output,
    ].into_iter().enumerate() {
        for (index, value) in block.weights.iter_mut().enumerate() {
            *value = 0.01 * (1 + block_index + index) as f32;
        }
        for (index, value) in block.biases.iter_mut().enumerate() {
            *value = -0.02 * (1 + block_index + index) as f32;
        }
        let bn = block.batch_norm.as_mut().expect("RNN BN state");
        for channel in 0..bn.biases.len() {
            bn.biases[channel] = 0.03 * (1 + channel) as f32;
            bn.scales[channel] = 0.7 + 0.05 * channel as f32;
            bn.rolling_mean[channel] = -0.04 * channel as f32;
            bn.rolling_variance[channel] = 0.6 + 0.1 * channel as f32;
        }
    }
    let input = patterned(cpu.network.net.input_shape());
    let expected = cpu.forward(&input).expect("CPU RNN forward").final_output().expect("RNN output").clone();
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).expect("native GPU RNN inference with BN");
    let actual = gpu_forward.cpu_forward.final_output().expect("GPU RNN output");
    close("native RNN BN inference", &expected.data, &actual.data, 4e-4);
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
    gpu.init_train_buffers_gpu()
        .expect("GPU BN RNN train buffers");
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
    gpu.zero_train_buffers_gpu()
        .expect("zero GPU BN RNN buffers");
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
    close(
        "BN RNN upstream weights",
        &cpu_input.weights,
        &gpu_input.weights,
        3e-4,
    );
    close(
        "BN RNN upstream biases",
        &cpu_input.biases,
        &gpu_input.biases,
        3e-4,
    );
    let (LayerParams::Rnn(cpu_rnn), LayerParams::Rnn(gpu_rnn)) = (&cpu.params[1], &gpu.params[1])
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
fn gru_native_inference_matches_cpu_for_multiple_steps() {
    let net = network("[net]\nwidth=2\nheight=1\nchannels=1\nbatch=4\n[gru]\noutput=3\nsteps=2\n");
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Gru(params) = &mut cpu.params[0] else { panic!("expected GRU parameters"); };
    for (block_index, block) in [
        &mut params.input_z, &mut params.state_z, &mut params.input_r,
        &mut params.state_r, &mut params.input_h, &mut params.state_h,
    ].into_iter().enumerate() {
        for (index, value) in block.weights.iter_mut().enumerate() {
            *value = 0.01 * (1 + block_index + index) as f32;
        }
        for (index, value) in block.biases.iter_mut().enumerate() {
            *value = -0.02 * (1 + block_index + index) as f32;
        }
    }
    let input = patterned(cpu.network.net.input_shape());
    let expected = cpu.forward(&input).expect("CPU GRU forward").final_output().expect("GRU output").clone();
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).expect("native GPU GRU inference");
    let actual = gpu_forward.cpu_forward.final_output().expect("GPU GRU output");
    close("native GRU inference", &expected.data, &actual.data, 2e-4);
}

#[test]
fn gru_native_inference_matches_cpu_with_rolling_batch_norm() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=4\n[gru]\noutput=3\nsteps=2\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Gru(params) = &mut cpu.params[0] else { panic!("expected GRU parameters"); };
    for (block_index, block) in [
        &mut params.input_z, &mut params.state_z, &mut params.input_r,
        &mut params.state_r, &mut params.input_h, &mut params.state_h,
    ].into_iter().enumerate() {
        for (index, value) in block.weights.iter_mut().enumerate() {
            *value = 0.01 * (1 + block_index + index) as f32;
        }
        for (index, value) in block.biases.iter_mut().enumerate() {
            *value = -0.02 * (1 + block_index + index) as f32;
        }
        let bn = block.batch_norm.as_mut().expect("GRU BN state");
        for channel in 0..bn.biases.len() {
            bn.biases[channel] = 0.03 * (1 + channel) as f32;
            bn.scales[channel] = 0.7 + 0.05 * channel as f32;
            bn.rolling_mean[channel] = -0.04 * channel as f32;
            bn.rolling_variance[channel] = 0.6 + 0.1 * channel as f32;
        }
    }
    let input = patterned(cpu.network.net.input_shape());
    let expected = cpu.forward(&input).expect("CPU GRU forward").final_output().expect("GRU output").clone();
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).expect("native GPU GRU inference with BN");
    let actual = gpu_forward.cpu_forward.final_output().expect("GPU GRU output");
    close("native GRU BN inference", &expected.data, &actual.data, 3e-4);
}

#[test]
fn gru_gpu_training_path_is_rejected_as_unsupported_not_misconfigured() {
    let net = network("[net]\nwidth=2\nheight=1\nchannels=1\nbatch=1\n[gru]\noutput=2\nsteps=2\n");
    let input = patterned(net.net.input_shape());
    let gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    let error = match gpu.forward_training_with_state(&input) {
        Err(error) => error,
        Ok(_) => panic!("GRU GPU training remains unsupported without native backward support"),
    };
    assert!(
        format!("{error:?}").contains("UnsupportedLayer"),
        "unexpected GRU GPU error: {error:?}"
    );
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
fn lstm_native_inference_matches_cpu_with_rolling_batch_norm() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[lstm]\noutput=3\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Lstm(params) = &mut cpu.params[0] else { panic!("expected LSTM parameters"); };
    for (block_index, block) in [
        &mut params.wf, &mut params.wi, &mut params.wg, &mut params.wo,
        &mut params.uf, &mut params.ui, &mut params.ug, &mut params.uo,
    ].into_iter().enumerate() {
        for (index, value) in block.weights.iter_mut().enumerate() {
            *value = 0.008 * (1 + block_index + index) as f32;
        }
        for (index, value) in block.biases.iter_mut().enumerate() {
            *value = -0.01 * (1 + block_index + index) as f32;
        }
        let bn = block.batch_norm.as_mut().expect("LSTM BN state");
        for channel in 0..bn.biases.len() {
            bn.biases[channel] = 0.02 * (1 + channel) as f32;
            bn.scales[channel] = 0.75 + 0.03 * channel as f32;
            bn.rolling_mean[channel] = -0.025 * channel as f32;
            bn.rolling_variance[channel] = 0.7 + 0.08 * channel as f32;
        }
    }
    let input = patterned(cpu.network.net.input_shape());
    let expected = cpu.forward(&input).expect("CPU LSTM forward").final_output().expect("LSTM output").clone();
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).expect("native GPU LSTM inference with BN");
    let actual = gpu_forward.cpu_forward.final_output().expect("GPU LSTM output");
    close("native LSTM BN inference", &expected.data, &actual.data, 4e-4);
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
    close(
        "BN LSTM upstream weights",
        &cpu_conv.weights,
        &gpu_conv.weights,
        3e-4,
    );
    close(
        "BN LSTM upstream biases",
        &cpu_conv.biases,
        &gpu_conv.biases,
        3e-4,
    );
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
    let target: Vec<f32> = (0..out_shape.batch * out_shape.element_count())
        .map(|i| 0.03 * i as f32 - 0.5)
        .collect();
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
        cpu.backward_accumulate_seeded(&input, &fwd, &seeds)
            .expect("cpu backward");
        cpu.update_weights(lr, momentum, decay, batch);

        gpu.zero_train_buffers_gpu().expect("zero");
        let fwd = gpu
            .forward_training_with_state(&input)
            .expect("oxide forward");
        let (seed, loss) = seed_and_loss(fwd.cpu_forward.final_output().expect("output"));
        gpu_loss.push(loss);
        let mut seeds = vec![None; 4];
        seeds[3] = Some(seed);
        gpu.backward_accumulate_seeded_gpu(&fwd, &seeds)
            .expect("oxide backward");
        gpu.update_weights_gpu(lr, momentum, decay, batch)
            .expect("oxide update");
    }
    println!("loss trajectory cpu  {cpu_loss:?}\nloss trajectory oxide {gpu_loss:?}");
    for (step, (c, g)) in cpu_loss.iter().zip(&gpu_loss).enumerate() {
        assert!(
            (c - g).abs() <= 1e-3 * c.abs().max(1.0),
            "step {step}: cpu loss {c} vs oxide {g}"
        );
    }
    assert!(
        cpu_loss.last() < cpu_loss.first(),
        "loss must go down: {cpu_loss:?}"
    );
    for (i, (c, g)) in cpu.params.iter().zip(&gpu.params).enumerate() {
        if let (LayerParams::Convolutional(c), LayerParams::Convolutional(g)) = (c, g) {
            close(&format!("layer {i} weights"), &c.weights, &g.weights, 1e-3);
            close(&format!("layer {i} biases"), &c.biases, &g.biases, 1e-3);
        }
    }
}

#[test]
fn eml_two_branch_forward_backward_on_oxide() {
    for batch_normalize in [false, true] {
        let cfg = format!(
            "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[eml_convolutional]\nfilters=2\nsize=1\nresidual=1\nbatch_normalize={}\n",
            if batch_normalize { 1 } else { 0 }
        );
        let net = network(&cfg);
        let mut cpu = ExecutableNetwork::new(net.clone());
        let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
        let LayerParams::EmlConvolutional(params) = &mut gpu.params[0] else {
            panic!("EML parameters");
        };
        params.input.weights.copy_from_slice(&[0.4, -0.2, 0.1, 0.3]);
        params.input.biases.copy_from_slice(&[0.05, -0.1]);
        params
            .self_params
            .weights
            .copy_from_slice(&[-0.3, 0.25, 0.2, 0.15]);
        params.self_params.biases.copy_from_slice(&[0.12, 0.03]);
        for (index, conv) in [&mut params.input, &mut params.self_params]
            .into_iter()
            .enumerate()
        {
            if let Some(batch_norm) = conv.batch_norm.as_mut() {
                batch_norm
                    .scales
                    .copy_from_slice(&[0.8 + index as f32 * 0.2, 1.1]);
                batch_norm.rolling_mean.copy_from_slice(&[-0.1, 0.07]);
                batch_norm.rolling_variance.copy_from_slice(&[0.9, 1.3]);
            }
        }
        cpu.params = gpu.params.clone();

        let input = patterned(cpu.network.net.input_shape());
        let cpu_inference = cpu.forward(&input).expect("CPU EML inference");
        let gpu_inference = gpu.forward_with_state(&input).expect("Oxide EML inference");
        let oxide_inference = gpu
            .backend
            .download_f32(&gpu_inference.gpu_layer_outputs[0])
            .expect("download EML inference");
        close(
            "EML rolling-stat inference output",
            &cpu_inference.layer_outputs[0].data,
            &oxide_inference,
            2e-4,
        );

        let cpu_forward = cpu.forward_training(&input).expect("CPU EML forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide EML training forward");
        let oxide_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download EML training output");
        if batch_normalize {
            assert!(oxide_output.iter().all(|value| value.is_finite()));
        } else {
            close(
                "EML no-BN training output",
                &cpu_forward.layer_outputs[0].data,
                &oxide_output,
                2e-4,
            );
        }
        assert!(
            gpu_forward.cpu_forward.eml_caches[0].is_none(),
            "native EML execution does not retain host branch caches"
        );

        let initial_batch_norm = if batch_normalize {
            let LayerParams::EmlConvolutional(params) = &gpu.params[0] else {
                unreachable!();
            };
            Some([
                (
                    params
                        .input
                        .batch_norm
                        .as_ref()
                        .unwrap()
                        .rolling_mean
                        .clone(),
                    params.input.batch_norm.as_ref().unwrap().scales.clone(),
                    params.input.weights.clone(),
                ),
                (
                    params
                        .self_params
                        .batch_norm
                        .as_ref()
                        .unwrap()
                        .rolling_mean
                        .clone(),
                    params
                        .self_params
                        .batch_norm
                        .as_ref()
                        .unwrap()
                        .scales
                        .clone(),
                    params.self_params.weights.clone(),
                ),
            ])
        } else {
            None
        };
        let shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::from_data(
            shape,
            (0..shape.batch * shape.element_count())
                .map(|i| (i as f32 * 0.41).cos())
                .collect(),
        )
        .expect("EML seed");
        let seeds = vec![Some(seed)];
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero EML train buffers");
        for _ in 0..2 {
            if !batch_normalize {
                cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                    .expect("CPU EML backward");
            }
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide EML backward");
        }
        assert!(!gpu.gradients_non_finite().expect("EML gradient finiteness"));
        let params_before_optimizer = gpu.params.clone();
        if let Some(initial) = initial_batch_norm.as_ref() {
            let LayerParams::EmlConvolutional(params) = &gpu.params[0] else {
                unreachable!();
            };
            assert_ne!(
                params.input.batch_norm.as_ref().unwrap().rolling_mean,
                initial[0].0,
                "input-branch rolling mean updated"
            );
            assert_ne!(
                params.self_params.batch_norm.as_ref().unwrap().rolling_mean,
                initial[1].0,
                "self-branch rolling mean updated"
            );
        } else {
            cpu.update_weights(0.01, 0.8, 0.0, 2);
        }
        if batch_normalize {
            gpu.update_weights_gpu(0.01, 0.8, 0.0, 2)
                .expect("EML BN device SGD update");
        } else {
            gpu.update_weights_gpu_deferred(0.01, 0.8, 0.0, 2)
                .expect("deferred EML device SGD update");
            assert_eq!(
                gpu.params, params_before_optimizer,
                "deferred EML host sync"
            );
            gpu.sync_host_weights_from_device()
                .expect("sync deferred EML weights");
            assert_eq!(gpu.params, cpu.params, "no-BN EML SGD parity");

            let cpu_next = cpu.forward_training(&input).expect("next CPU EML forward");
            let gpu_next = gpu
                .forward_training_device(&input)
                .expect("next Oxide EML forward");
            cpu.backward_accumulate_seeded(&input, &cpu_next, &seeds)
                .expect("next CPU EML backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_next, &seeds)
                .expect("next Oxide EML backward");
            cpu.update_weights(0.01, 0.8, 0.0, 2);
            gpu.update_weights_gpu_deferred(0.01, 0.8, 0.0, 2)
                .expect("second deferred EML SGD update");
            gpu.sync_host_weights_from_device()
                .expect("sync second EML SGD update");
            let (
                LayerParams::EmlConvolutional(gpu_params),
                LayerParams::EmlConvolutional(cpu_params),
            ) = (&gpu.params[0], &cpu.params[0])
            else {
                panic!("EML params after momentum update");
            };
            for (name, gpu_branch, cpu_branch) in [
                ("input", &gpu_params.input, &cpu_params.input),
                ("self", &gpu_params.self_params, &cpu_params.self_params),
            ] {
                close(
                    &format!("EML {name} SGD momentum weights"),
                    &gpu_branch.weights,
                    &cpu_branch.weights,
                    1e-6,
                );
                close(
                    &format!("EML {name} SGD momentum biases"),
                    &gpu_branch.biases,
                    &cpu_branch.biases,
                    1e-6,
                );
            }
        }
        if let Some(initial) = initial_batch_norm.as_ref() {
            let LayerParams::EmlConvolutional(params) = &gpu.params[0] else {
                unreachable!();
            };
            for (branch, initial) in [
                (&params.input, &initial[0]),
                (&params.self_params, &initial[1]),
            ] {
                let batch_norm = branch.batch_norm.as_ref().unwrap();
                assert_ne!(
                    batch_norm.scales, initial.1,
                    "EML BN scale optimizer update"
                );
                assert_ne!(
                    branch.weights, initial.2,
                    "EML BN convolution optimizer update"
                );
            }
            cpu.params = gpu.params.clone();
        }

        let cpu_after = cpu.forward(&input).expect("CPU EML after update");
        let gpu_after = gpu.forward(&input).expect("Oxide EML after update");
        close(
            "EML forward after host update invalidates device branch cache",
            &cpu_after.layer_outputs[0].data,
            &gpu_after.layer_outputs[0].data,
            2e-4,
        );
    }
}

#[test]
fn eml_adam_updates_both_device_branches_with_deferred_sync() {
    for batch_normalize in [false, true] {
        let cfg = format!(
            "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=2\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n\
             [eml_convolutional]\nfilters=2\nsize=1\nbatch_normalize={}\nresidual=0\n",
            u8::from(batch_normalize),
        );
        let mut gpu =
            GpuExecutableNetwork::new(network(&cfg), OxideBackend::new().expect("Oxide backend"));
        let LayerParams::EmlConvolutional(params) = &mut gpu.params[0] else {
            panic!("EML parameters");
        };
        for (index, branch) in [&mut params.input, &mut params.self_params]
            .into_iter()
            .enumerate()
        {
            for (i, weight) in branch.weights.iter_mut().enumerate() {
                *weight = ((i + index) as f32 * 0.7).sin() * 0.4;
            }
            for (i, bias) in branch.biases.iter_mut().enumerate() {
                *bias = ((i + index) as f32 * 0.3).cos() * 0.1;
            }
        }
        let input_shape = gpu.network.net.input_shape();
        let input = Tensor::from_data(
            input_shape,
            (0..input_shape.batch * input_shape.element_count())
                .map(|i| 0.25 + 0.1 * i as f32)
                .collect(),
        )
        .expect("EML Adam input");
        gpu.zero_train_buffers_gpu().expect("zero EML Adam buffers");
        let forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide EML Adam forward");
        let shape = gpu.network.layers[0].output_shape;
        let seed = Tensor::filled(shape, 1.0);
        gpu.backward_accumulate_seeded_gpu(&forward, &[Some(seed.clone())])
            .expect("Oxide EML Adam backward");
        let cpu_adam_reference = if !batch_normalize {
            let mut cpu = ExecutableNetwork::new(gpu.network.clone());
            cpu.params = gpu.params.clone();
            let cpu_forward = cpu.forward_training(&input).expect("CPU EML Adam forward");
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &[Some(seed)])
                .expect("CPU EML Adam backward");
            let LayerTrainBuffers::EmlConvolutional(train) = &cpu.train_buffers[0] else {
                unreachable!()
            };
            let LayerParams::EmlConvolutional(params) = &cpu.params[0] else {
                unreachable!()
            };
            assert!(train.input.weight_updates.iter().any(|value| *value != 0.0));
            let lr = 0.01;
            let eps = cpu.network.net.adam_eps;
            Some((
                expected_first_adam_step(
                    &params.input.weights,
                    &train.input.weight_updates,
                    lr,
                    eps,
                ),
                expected_first_adam_step(&params.input.biases, &train.input.bias_updates, lr, eps),
                expected_first_adam_step(
                    &params.self_params.weights,
                    &train.self_params.weight_updates,
                    lr,
                    eps,
                ),
                expected_first_adam_step(
                    &params.self_params.biases,
                    &train.self_params.bias_updates,
                    lr,
                    eps,
                ),
            ))
        } else {
            None
        };
        assert!(!gpu.gradients_non_finite().expect("finite EML Adam grads"));
        let before = gpu.params.clone();
        gpu.update_weights_gpu_deferred(0.01, 0.0, 0.0, 2)
            .expect("deferred Oxide EML Adam update");
        assert_eq!(gpu.params, before, "deferred EML Adam host sync");
        gpu.sync_host_weights_from_device()
            .expect("sync deferred EML Adam params");
        let sidecar =
            std::env::temp_dir().join(format!("oxide-eml-adam-{}.state", std::process::id()));
        gpu.save_optimizer_state(&sidecar)
            .expect("save EML Adam state");
        assert_eq!(&std::fs::read(&sidecar).unwrap()[..8], b"DNOPT5\0\0");
        std::fs::remove_file(sidecar).ok();
        let (LayerParams::EmlConvolutional(before), LayerParams::EmlConvolutional(after)) =
            (&before[0], &gpu.params[0])
        else {
            unreachable!()
        };
        for (name, old, new) in [
            ("input", &before.input, &after.input),
            ("self", &before.self_params, &after.self_params),
        ] {
            assert!(new.weights.iter().all(|value| value.is_finite()));
            assert!(
                old.weights != new.weights || old.biases != new.biases,
                "EML Adam {name} branch was unchanged"
            );
            if batch_normalize {
                let old_bn = old.batch_norm.as_ref().unwrap();
                let new_bn = new.batch_norm.as_ref().unwrap();
                assert!(new_bn.scales.iter().all(|value| value.is_finite()));
                assert_ne!(old_bn.scales, new_bn.scales, "EML Adam {name} BN scales");
            }
        }
        if let Some((input_weights, input_biases, self_weights, self_biases)) = cpu_adam_reference {
            close(
                "EML Adam input weights",
                &input_weights,
                &after.input.weights,
                3e-4,
            );
            close(
                "EML Adam input biases",
                &input_biases,
                &after.input.biases,
                3e-4,
            );
            close(
                "EML Adam self weights",
                &self_weights,
                &after.self_params.weights,
                3e-4,
            );
            close(
                "EML Adam self biases",
                &self_biases,
                &after.self_params.biases,
                3e-4,
            );
        }
    }
}

#[test]
fn dcnv4_bn_native_f32_matches_cpu_chain_and_updates() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
         [dcnv4]\nsize=1\npadding=0\nd_stride=2\nsoftmax=0\
         \noffset_scale=0.6\nbatch_normalize=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 73);
    fill_params(&mut cpu.params);
    let LayerParams::Dcnv4(params) = &mut cpu.params[1] else {
        panic!("DCNv4 parameters")
    };
    for (index, weight) in params.offset_weights.iter_mut().enumerate() {
        *weight = ((index as f32 * 0.37).sin()) * 0.001;
    }
    params.offset_biases[0] = 0.13;
    params.offset_biases[1] = 0.19;
    params.offset_biases[2] = 0.8;
    params.biases.fill(7.0);
    let bn = params.batch_norm.as_mut().unwrap();
    bn.biases.fill(9.0);
    bn.scales.copy_from_slice(&[1.1, -0.7]);
    bn.rolling_mean.copy_from_slice(&[0.2, -0.1]);
    bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);

    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("DCNv4 train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU DCNv4 forward");
    let oxide_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide DCNv4 native BN forward");
    for (index, (cpu_output, oxide_output)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&oxide_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(
            &format!("DCNv4 native BN layer {index}"),
            &cpu_output.data,
            &oxide_output.data,
            2e-4,
        );
    }

    let seed = Tensor::filled(cpu.network.layers[1].output_shape, 0.17);
    assert!(
        cpu.backward(&input, &oxide_forward.cpu_forward, &seed)
            .unwrap_err()
            .to_string()
            .contains("missing [dcnv4] forward cache"),
        "BN must use native math now"
    );
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("zero DCNv4 train buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU DCNv4 backward");
    gpu.backward_accumulate_seeded_gpu(&oxide_forward, &seeds)
        .expect("Oxide DCNv4 native BN backward");
    let before = cpu.params.clone();
    cpu.update_weights(1e-3, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1e-3, 0.0, 0.0, 1)
        .expect("Oxide DCNv4 native BN update");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(oxide_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("preceding convolution parameters")
    };
    close(
        "DCNv4 chain convolution weights",
        &cpu_conv.weights,
        &oxide_conv.weights,
        5e-4,
    );
    close(
        "DCNv4 chain convolution biases",
        &cpu_conv.biases,
        &oxide_conv.biases,
        5e-4,
    );
    let (LayerParams::Dcnv4(old), LayerParams::Dcnv4(cpu_dcn), LayerParams::Dcnv4(oxide_dcn)) =
        (&before[1], &cpu.params[1], &gpu.params[1])
    else {
        panic!("DCNv4 parameters")
    };
    close(
        "DCNv4 offset weights",
        &cpu_dcn.offset_weights,
        &oxide_dcn.offset_weights,
        5e-4,
    );
    close(
        "DCNv4 offset biases",
        &cpu_dcn.offset_biases,
        &oxide_dcn.offset_biases,
        5e-4,
    );
    assert_ne!(old.offset_weights, cpu_dcn.offset_weights);
    let cb = cpu_dcn.batch_norm.as_ref().unwrap();
    let gb = oxide_dcn.batch_norm.as_ref().unwrap();
    close("DCNv4 BN gamma", &cb.scales, &gb.scales, 3e-5);
    close(
        "DCNv4 BN rolling mean",
        &cb.rolling_mean,
        &gb.rolling_mean,
        3e-5,
    );
    close(
        "DCNv4 BN rolling variance",
        &cb.rolling_variance,
        &gb.rolling_variance,
        3e-5,
    );
    assert_eq!(old.biases, cpu_dcn.biases);
    assert_eq!(old.biases, oxide_dcn.biases);
    assert_eq!(cb.biases, vec![9.0; 2]);
    assert_eq!(gb.biases, cb.biases);
    assert_eq!(old.dummy_weights, cpu_dcn.dummy_weights);
    assert_eq!(cpu_dcn.dummy_weights, oxide_dcn.dummy_weights);
}

#[test]
fn dcnv4_mixed_network_cpu_and_gpu_backward_succeed() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n\
        [dcnv4]\nsize=1\npadding=0\nd_stride=2\nsoftmax=0\nbatch_normalize=1\nactivation=linear\n\
        [deconvolutional]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
        unreachable!()
    };
    p.offset_biases[..3].copy_from_slice(&[0.43, 0.57, 0.8]);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cf = cpu.forward_training(&input).unwrap();
    let gf = gpu.forward_training_with_state(&input).unwrap();
    let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.3);
    let seeds = [Some(seed), None];
    // The GPU forward keeps its DCNv4 cache in GPU state, not the host
    // snapshot. Consume each forward state with its matching backward path.
    cpu.backward_accumulate_seeded(&input, &cf, &seeds).unwrap();
    gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
}

#[test]
fn dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu() {
    for batch_normalize in [false, true] {
    for adam in [false, true] {
        for (softmax, remove_center, activation, size, padding, dilation) in [
            (1, 1, "swish", 3, 2, 2),
            (0, 0, "leaky", 3, 2, 2),
            (0, 0, "revleaky", 3, 2, 2),
            (0, 0, "leaky", 7, 3, 1),
            (0, 0, "leaky", 9, 4, 1),
            (0, 0, "leaky", 11, 5, 1),
            (0, 0, "leaky", 13, 6, 1),
            (0, 0, "leaky", 15, 7, 1),
            (0, 0, "leaky", 17, 8, 1),
            (0, 0, "leaky", 19, 9, 1),
            (0, 0, "leaky", 21, 10, 1),
            (0, 0, "leaky", 23, 11, 1),
            (0, 0, "leaky", 25, 12, 1),
            (0, 0, "leaky", 27, 13, 1),
        ] {
            // Keep large-kernel runtime optimizer parity on SGD for this slice;
            // Adam's CPU/GPU delta check remains covered through 15x15.
            if size >= 17 && adam {
                continue;
            }
            let net = network(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\ntry_fix_nan=0\nadam={}\ndecay=0.01\n\
                 [convolutional]\nfilters=4\nsize=1\nactivation=linear\n\
                 [dcnv4]\nsize={size}\npadding={padding}\ndilation={dilation}\ngroups=2\nd_stride=2\n\
                 softmax={softmax}\nremove_center={remove_center}\noffset_scale=0.75\nactivation={activation}\nbatch_normalize={}\n",
                u8::from(adam), u8::from(batch_normalize)
            ));
            let mut cpu = ExecutableNetwork::new(net.clone());
            fill_params(&mut cpu.params);
            let LayerParams::Dcnv4(p) = &mut cpu.params[1] else {
                unreachable!()
            };
            for (i, w) in p.offset_weights.iter_mut().enumerate() {
                *w = ((i * 7 % 17) as f32 - 8.0) * 0.002;
            }
            for (i, b) in p.offset_biases.iter_mut().enumerate() {
                *b = 0.43 + (i % 5) as f32 * 0.03;
            }
            p.biases.copy_from_slice(&[0.03, -0.02, 0.05, -0.01]);
            if let Some(bn) = &mut p.batch_norm {
                p.biases.fill(7.0);
                bn.biases.fill(9.0);
                bn.scales.copy_from_slice(&[0.0,0.9,-1.2,0.6]);
                bn.rolling_mean.copy_from_slice(&[0.2,-0.1,0.3,-0.2]);
                bn.rolling_variance.copy_from_slice(&[0.7,1.3,0.6,1.2]);
            }
            let initial = cpu.params.clone();
            let mut gpu = GpuExecutableNetwork::new(
                net,
                OxideBackend::new()
                    .unwrap()
                    .with_gemm_mode(darknet_cuda_oxide::GemmMode::Fp32),
            );
            gpu.params = cpu.params.clone();
            gpu.init_train_buffers_gpu().unwrap();
            for step in 0..2 {
                let mut input = patterned(cpu.network.net.input_shape());
                for v in &mut input.data {
                    *v *= 0.8 + 0.1 * step as f32;
                }
                let cf = cpu.forward_training(&input).unwrap();
                let gf = gpu.forward_training_with_state(&input).unwrap();
                for (a, b) in cf.layer_outputs.iter().zip(&gf.cpu_forward.layer_outputs) {
                    close("DCNv4 native runtime output", &a.data, &b.data, 3e-4);
                }
                let shape = cpu.network.layers[1].output_shape;
                let seed = Tensor::from_data(
                    shape,
                    (0..shape.batch * shape.element_count())
                        .map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.17)
                        .collect(),
                )
                .unwrap();
                // Eligible GPU forward deliberately does not produce a CPU
                // DCNv4 cache. CPU fallback here would fail, so successful GPU
                // backward below cannot be a hidden whole-network CPU pass.
                assert!(
                    cpu.backward(&input, &gf.cpu_forward, &seed)
                        .unwrap_err()
                        .to_string()
                        .contains("missing [dcnv4] forward cache")
                );
                let seeds = [None, Some(seed)];
                cpu.backward_accumulate_seeded(&input, &cf, &seeds).unwrap();
                gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
                let before = cpu.params.clone();
                let before_gpu = gpu.params.clone();
                if adam {
                    cpu.update_weights_adam(0.005, 2).unwrap();
                } else {
                    cpu.update_weights(0.005, 0.3, 0.01, 2);
                }
                gpu.update_weights_gpu(0.005, 0.3, 0.01, 2).unwrap();
                let compare = |name: &str, c: &[f32], g: &[f32], bc: &[f32], bg: &[f32]| {
                    close(name, c, g, 2e-5);
                    for i in 0..c.len() {
                        let dc = c[i] - bc[i];
                        let dg = g[i] - bg[i];
                        assert!(
                            (dc - dg).abs() <= 5e-7 + 1e-3 * dc.abs(),
                            "{name} size={size} bn={batch_normalize} step={step} adam={adam} delta[{i}]: {dc} vs {dg}"
                        );
                    }
                    assert_ne!(bg, g, "{name}: GPU block did not move");
                };
                let (
                    LayerParams::Convolutional(c),
                    LayerParams::Convolutional(g),
                    LayerParams::Convolutional(bc),
                    LayerParams::Convolutional(bg),
                ) = (&cpu.params[0], &gpu.params[0], &before[0], &before_gpu[0])
                else {
                    unreachable!()
                };
                compare(
                    "upstream conv weights",
                    &c.weights,
                    &g.weights,
                    &bc.weights,
                    &bg.weights,
                );
                compare(
                    "upstream conv biases",
                    &c.biases,
                    &g.biases,
                    &bc.biases,
                    &bg.biases,
                );
                let (
                    LayerParams::Dcnv4(c),
                    LayerParams::Dcnv4(g),
                    LayerParams::Dcnv4(bc),
                    LayerParams::Dcnv4(bg),
                ) = (&cpu.params[1], &gpu.params[1], &before[1], &before_gpu[1])
                else {
                    unreachable!()
                };
                if let (Some(cbn),Some(gbn),Some(bcbn),Some(bgbn)) =
                    (&c.batch_norm,&g.batch_norm,&bc.batch_norm,&bg.batch_norm) {
                    compare("DCNv4 gamma",&cbn.scales,&gbn.scales,&bcbn.scales,&bgbn.scales);
                    close("DCNv4 rolling mean",&cbn.rolling_mean,&gbn.rolling_mean,3e-5);
                    close("DCNv4 rolling variance",&cbn.rolling_variance,&gbn.rolling_variance,3e-5);
                    assert_eq!(cbn.biases,gbn.biases);
                    assert_eq!(cbn.biases,vec![9.0;4]);
                    assert_eq!(c.biases,bc.biases); assert_eq!(g.biases,bg.biases);
                } else {
                    compare("DCNv4 output biases",&c.biases,&g.biases,&bc.biases,&bg.biases);
                }
                compare(
                    "DCNv4 offset weights",
                    &c.offset_weights,
                    &g.offset_weights,
                    &bc.offset_weights,
                    &bg.offset_weights,
                );
                compare(
                    "DCNv4 offset biases",
                    &c.offset_biases,
                    &g.offset_biases,
                    &bc.offset_biases,
                    &bg.offset_biases,
                );
                let LayerParams::Dcnv4(old) = &initial[1] else {
                    unreachable!()
                };
                assert_eq!(c.dummy_weights, old.dummy_weights);
                assert_eq!(g.dummy_weights, old.dummy_weights);
            }
        }
    }
    }
}

#[test]
fn deform_conv_f32_host_fallback_matches_cpu_chain_and_updates() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=1\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
         [deform_conv]\nfilters=2\ngroups=2\nsize=1\nmask=1\n\
         batch_normalize=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let LayerParams::DeformConv(params) = &mut cpu.params[1] else {
        panic!("deform_conv parameters")
    };
    for (index, weight) in params.main.weights.iter_mut().enumerate() {
        *weight = 0.8 + index as f32 * 0.17;
    }
    params.main.biases.copy_from_slice(&[0.05, -0.03]);
    let batch_norm = params.main.batch_norm.as_mut().unwrap();
    batch_norm.scales.copy_from_slice(&[1.1, 0.9]);
    batch_norm.rolling_mean.copy_from_slice(&[0.02, -0.04]);
    batch_norm.rolling_variance.copy_from_slice(&[0.8, 1.2]);
    params
        .offset_weights
        .copy_from_slice(&[0.0008, -0.0004, 0.0005, 0.0003]);
    params.offset_biases.copy_from_slice(&[0.12, -0.09]);
    params
        .mask_weights
        .as_mut()
        .unwrap()
        .copy_from_slice(&[0.02, -0.015]);
    params.mask_biases.as_mut().unwrap().copy_from_slice(&[0.1]);

    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("deform_conv train buffers");
    let input = patterned(cpu.network.net.input_shape());
    assert!(gpu.forward_compact(&input).is_err());
    let cpu_forward = cpu
        .forward_training(&input)
        .expect("CPU deform_conv forward");
    let oxide_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide deform_conv host-fallback forward");
    for (index, (cpu_output, oxide_output)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&oxide_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(
            &format!("deform_conv host fallback layer {index}"),
            &cpu_output.data,
            &oxide_output.data,
            2e-4,
        );
    }

    let seed = Tensor::filled(cpu.network.layers[1].output_shape, 0.13);
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("zero deform_conv buffers");
    let cpu_backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU deform_conv backward");
    gpu.backward_accumulate_seeded_gpu(&oxide_forward, &seeds)
        .expect("Oxide deform_conv host-fallback backward");
    let before = cpu.params.clone();
    cpu.update_weights(1e-3, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1e-3, 0.0, 0.0, 1)
        .expect("Oxide deform_conv host-fallback update");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(oxide_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("preceding convolution parameters")
    };
    close(
        "deform_conv chain convolution weights",
        &cpu_conv.weights,
        &oxide_conv.weights,
        5e-4,
    );
    close(
        "deform_conv chain convolution biases",
        &cpu_conv.biases,
        &oxide_conv.biases,
        5e-4,
    );
    let (
        LayerParams::DeformConv(old),
        LayerParams::DeformConv(cpu_deform),
        LayerParams::DeformConv(oxide_deform),
    ) = (&before[1], &cpu.params[1], &gpu.params[1])
    else {
        panic!("deform_conv parameters")
    };
    close(
        "deform_conv main weights",
        &cpu_deform.main.weights,
        &oxide_deform.main.weights,
        5e-4,
    );
    close(
        "deform_conv offset weights",
        &cpu_deform.offset_weights,
        &oxide_deform.offset_weights,
        5e-4,
    );
    close(
        "deform_conv mask weights",
        cpu_deform.mask_weights.as_ref().unwrap(),
        oxide_deform.mask_weights.as_ref().unwrap(),
        5e-4,
    );
    assert_ne!(old.main.weights, cpu_deform.main.weights);
    assert_ne!(old.offset_weights, cpu_deform.offset_weights);
    assert_ne!(old.mask_weights, cpu_deform.mask_weights);
    assert!(
        cpu_backward
            .input_gradient
            .data
            .iter()
            .any(|value| *value != 0.0)
    );
}

/// Native device path (no batch norm, dilation 1) end to end: forward outputs,
/// backward gradients (via one SGD step) and the parameter values must match the
/// CPU reference. Also asserts the host forward cache is *absent*, which is what
/// distinguishes this from the host-fallback test above.
#[test]
fn deform_conv_device_path_matches_cpu_chain_and_updates() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=1\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
         [deform_conv]\nfilters=2\ngroups=2\nsize=3\nmask=1\n\
         batch_normalize=0\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let LayerParams::DeformConv(params) = &mut cpu.params[1] else {
        panic!("deform_conv parameters")
    };
    for (index, weight) in params.main.weights.iter_mut().enumerate() {
        *weight = 0.06 + index as f32 * 0.011;
    }
    for (index, bias) in params.main.biases.iter_mut().enumerate() {
        *bias = 0.05 - index as f32 * 0.08;
    }
    for (index, weight) in params.offset_weights.iter_mut().enumerate() {
        *weight = 0.0008 - index as f32 * 0.00003;
    }
    for (index, bias) in params.offset_biases.iter_mut().enumerate() {
        *bias = 0.12 - index as f32 * 0.003;
    }
    params
        .mask_weights
        .as_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
        .for_each(|(i, w)| *w = 0.02 - i as f32 * 0.0005);
    for (index, bias) in params.mask_biases.as_mut().unwrap().iter_mut().enumerate() {
        *bias = 0.1 - index as f32 * 0.004;
    }

    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("deform_conv buffers");
    // `requires_cpu_backward_fallback` and the host cache are crate-private, so
    // the device path is established behaviourally: this network is deform-only
    // with `batch_normalize=0` and `dilation=1`, and the backward below fails
    // outright if the device forward cache is missing.
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU deform forward");
    let oxide_forward = gpu
        .forward_training_with_state(&input)
        .expect("device deform forward");
    for (index, (cpu_output, oxide_output)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&oxide_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        close(
            &format!("deform device layer {index}"),
            &cpu_output.data,
            &oxide_output.data,
            5e-4,
        );
    }

    let seed = Tensor::filled(cpu.network.layers[1].output_shape, 0.13);
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(seed);
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero deform buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU deform backward");
    gpu.backward_accumulate_seeded_gpu(&oxide_forward, &seeds)
        .expect("device deform backward");
    let before = cpu.params.clone();
    cpu.update_weights(1e-3, 0.0, 0.0, 1);
    gpu.update_weights_gpu(1e-3, 0.0, 0.0, 1)
        .expect("device deform update");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(oxide_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("preceding convolution parameters")
    };
    close("deform device conv weights", &cpu_conv.weights, &oxide_conv.weights, 5e-4);
    close("deform device conv biases", &cpu_conv.biases, &oxide_conv.biases, 5e-4);
    let (
        LayerParams::DeformConv(old),
        LayerParams::DeformConv(cpu_deform),
        LayerParams::DeformConv(oxide_deform),
    ) = (&before[1], &cpu.params[1], &gpu.params[1])
    else {
        panic!("deform_conv parameters")
    };
    close("deform device main weights", &cpu_deform.main.weights, &oxide_deform.main.weights, 5e-4);
    close("deform device main biases", &cpu_deform.main.biases, &oxide_deform.main.biases, 5e-4);
    close(
        "deform device offset weights",
        &cpu_deform.offset_weights,
        &oxide_deform.offset_weights,
        5e-4,
    );
    close(
        "deform device offset biases",
        &cpu_deform.offset_biases,
        &oxide_deform.offset_biases,
        5e-4,
    );
    close(
        "deform device mask weights",
        cpu_deform.mask_weights.as_ref().unwrap(),
        oxide_deform.mask_weights.as_ref().unwrap(),
        5e-4,
    );
    // The step must have actually moved every block away from its initial value.
    let moved = |a: &[f32], b: &[f32]| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-7);
    assert!(moved(&old.main.weights, &oxide_deform.main.weights), "main weights unchanged");
    assert!(
        moved(&old.offset_weights, &oxide_deform.offset_weights),
        "offset weights unchanged"
    );
    assert!(
        moved(
            old.mask_weights.as_ref().unwrap(),
            oxide_deform.mask_weights.as_ref().unwrap()
        ),
        "mask weights unchanged"
    );
}

#[test]
fn gpu_muon_deform_conv_native_gradients_match_cpu_across_steps() {
    let net = network(
        r#"[net]
width=4
height=4
channels=2
batch=1
muon=1
decoupled_decay=1
decay=0.0001
[convolutional]
filters=2
size=1
activation=linear
[deform_conv]
filters=2
groups=2
size=3
mask=1
batch_normalize=0
activation=linear
"#,
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    fill_params(&mut cpu.params);
    let LayerParams::Convolutional(conv) = &mut cpu.params[0] else {
        panic!("preceding convolution parameters")
    };
    conv.weights.copy_from_slice(&[0.8, -0.2, 0.1, 0.7]);
    conv.biases.copy_from_slice(&[0.03, -0.02]);
    let LayerParams::DeformConv(params) = &mut cpu.params[1] else {
        panic!("deform_conv parameters")
    };
    for (index, weight) in params.main.weights.iter_mut().enumerate() {
        *weight = 0.06 + index as f32 * 0.011;
    }
    for (index, bias) in params.main.biases.iter_mut().enumerate() {
        *bias = 0.05 - index as f32 * 0.08;
    }
    for (index, weight) in params.offset_weights.iter_mut().enumerate() {
        *weight = 0.0008 - index as f32 * 0.00003;
    }
    for (index, bias) in params.offset_biases.iter_mut().enumerate() {
        *bias = 0.12 - index as f32 * 0.003;
    }
    for (index, weight) in params.mask_weights.as_mut().unwrap().iter_mut().enumerate() {
        *weight = 0.02 - index as f32 * 0.0005;
    }
    for (index, bias) in params.mask_biases.as_mut().unwrap().iter_mut().enumerate() {
        *bias = 0.1 - index as f32 * 0.004;
    }

    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("deform Muon buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero deform Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU deform forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide deform forward");
        let output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[1])
            .expect("download deform output");
        close(
            &format!("deform Muon output step {step}"),
            &output,
            &cpu_forward.layer_outputs[1].data,
            5e-4,
        );
        for microbatch in 0..2 {
            let seed = Tensor::filled(
                cpu.network.layers[1].output_shape,
                0.08 + step as f32 * 0.01 + microbatch as f32 * 0.005,
            );
            let mut seeds = vec![None; 2];
            seeds[1] = Some(seed);
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU deform Muon backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide deform Muon backward");
        }
        cpu.update_weights_muon(0.01, 2)
            .expect("CPU deform Muon update");
        gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 2)
            .expect("Oxide deform Muon update");

        let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("preceding convolution parameter layout")
        };
        close(
            &format!("deform Muon upstream weights step {step}"),
            &gpu_conv.weights,
            &cpu_conv.weights,
            1e-3,
        );
        close(
            &format!("deform Muon upstream biases step {step}"),
            &gpu_conv.biases,
            &cpu_conv.biases,
            1e-3,
        );
        let (LayerParams::DeformConv(cpu_params), LayerParams::DeformConv(gpu_params)) =
            (&cpu.params[1], &gpu.params[1])
        else {
            panic!("deform_conv parameter layout")
        };
        for (name, cpu_values, gpu_values) in [
            ("main weights", &cpu_params.main.weights, &gpu_params.main.weights),
            ("main biases", &cpu_params.main.biases, &gpu_params.main.biases),
            ("offset weights", &cpu_params.offset_weights, &gpu_params.offset_weights),
            ("offset biases", &cpu_params.offset_biases, &gpu_params.offset_biases),
            (
                "mask weights",
                cpu_params.mask_weights.as_ref().unwrap(),
                gpu_params.mask_weights.as_ref().unwrap(),
            ),
            (
                "mask biases",
                cpu_params.mask_biases.as_ref().unwrap(),
                gpu_params.mask_biases.as_ref().unwrap(),
            ),
        ] {
            close(
                &format!("deform Muon {name} step {step}"),
                gpu_values,
                cpu_values,
                1e-3,
            );
        }
    }
}

#[test]
fn sam_native_gpu_forward_backward_fans_out_source_gradient_matches_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n\
         [convolutional]\nfilters=1\nsize=1\nstride=1\npadding=0\nactivation=linear\n\
         [convolutional]\nfilters=1\nsize=1\nstride=1\npadding=0\nactivation=linear\n\
         [sam]\nfrom=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    for (index, params) in cpu.params.iter_mut().enumerate() {
        let LayerParams::Convolutional(conv) = params else {
            continue;
        };
        conv.weights.fill(if index == 0 { 0.5 } else { 0.8 });
        conv.biases.fill(if index == 0 { 0.1 } else { 0.2 });
    }
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("SAM train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero SAM train buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU SAM forward");
        let gpu_summary = gpu
            .forward_training_device(&input)
            .expect("Oxide SAM training summary");
        let gpu_final = gpu
            .backend
            .download_f32(&gpu_summary.gpu_layer_outputs[2])
            .expect("download SAM device-summary output");
        close(
            &format!("SAM device-summary final output step {step}"),
            &cpu_forward.final_output().expect("CPU output").data,
            &gpu_final,
            2e-4,
        );
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide SAM forward");
        for (index, (cpu_output, gpu_output)) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
            .enumerate()
        {
            close(
                &format!("SAM layer {index} forward step {step}"),
                &cpu_output.data,
                &gpu_output.data,
                2e-4,
            );
        }
        let output_shape = cpu.network.layers[2].output_shape;
        let seed = Tensor::filled(output_shape, 0.07 + step as f32 * 0.01);
        let seeds = [None, None, Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU SAM backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide SAM backward");
        cpu.update_weights(0.01, 0.0, 0.0001, 1);
        gpu.update_weights_gpu(0.01, 0.0, 0.0001, 1)
            .expect("Oxide SAM SGD update");
        for index in 0..2 {
            let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
                (&cpu.params[index], &gpu.params[index])
            else {
                panic!("convolutional parameter layout expected");
            };
            close(
                &format!("SAM conv {index} weights step {step}"),
                &gpu_conv.weights,
                &cpu_conv.weights,
                3e-4,
            );
            close(
                &format!("SAM conv {index} biases step {step}"),
                &gpu_conv.biases,
                &cpu_conv.biases,
                3e-4,
            );
        }
    }
}

#[test]
fn local_native_gpu_training_matches_cpu_across_two_updates() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
         [local]\nfilters=3\nsize=2\nstride=1\npadding=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 61);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("local train buffers");
    assert_eq!(
        gpu.params, cpu.params,
        "GPU initialization changed local params"
    );

    for step in 0..2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero GPU local buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU local forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide local forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close("local forward", &cpu_output.data, &gpu_output.data, 2e-4);
        }
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::filled(output_shape, 0.03 + step as f32 * 0.01);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU local backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide local backward");
        cpu.update_weights(0.01, 0.0, 0.0, 2);
        gpu.update_weights_gpu(0.01, 0.0, 0.0, 2)
            .expect("GPU local update");
        let (LayerParams::Local(cpu_local), LayerParams::Local(gpu_local)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("local parameter layout expected");
        };
        close(
            "local weights",
            &gpu_local.weights,
            &cpu_local.weights,
            5e-4,
        );
        close("local biases", &gpu_local.biases, &cpu_local.biases, 5e-4);
    }
}

#[test]
fn gpu_muon_convolution_device_gradients_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [convolutional]\nfilters=2\nsize=3\nstride=1\npadding=1\ngroups=2\nbatch_normalize=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 79);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("convolution Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero convolution Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU convolution Muon forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide convolution Muon forward");
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download convolution Muon output");
        close(
            &format!("convolution Muon output step {step}"),
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            2e-4,
        );
        for microbatch in 0..2 {
            let seed = Tensor::filled(
                cpu.network.layers[0].output_shape,
                0.03 + step as f32 * 0.01 + microbatch as f32 * 0.005,
            );
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU convolution Muon backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide convolution Muon backward");
        }
        cpu.update_weights_muon(0.01, 2)
            .expect("CPU convolution Muon update");
        gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 2)
            .expect("Oxide convolution Muon update");
        let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("convolution parameter layout expected");
        };
        close(
            &format!("convolution Muon weights step {step}"),
            &gpu_conv.weights,
            &cpu_conv.weights,
            8e-4,
        );
        close(
            &format!("convolution Muon biases step {step}"),
            &gpu_conv.biases,
            &cpu_conv.biases,
            8e-4,
        );
    }
}

#[test]
fn gpu_muon_eml_device_gradients_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [eml_convolutional]\nfilters=2\nsize=1\nresidual=1\nbatch_normalize=0\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 89);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("EML Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero EML Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU EML Muon forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide EML Muon forward");
        assert!(gpu_forward.cpu_forward.eml_caches[0].is_none());
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download EML Muon output");
        close(
            &format!("EML Muon output step {step}"),
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            2e-4,
        );
        for microbatch in 0..2 {
            let seed = Tensor::filled(
                cpu.network.layers[0].output_shape,
                0.03 + step as f32 * 0.01 + microbatch as f32 * 0.005,
            );
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU EML Muon backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide EML Muon backward");
        }
        cpu.update_weights_muon(0.01, 2)
            .expect("CPU EML Muon update");
        gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 2)
            .expect("Oxide EML Muon update");
        let (LayerParams::EmlConvolutional(cpu_eml), LayerParams::EmlConvolutional(gpu_eml)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("EML parameter layout expected");
        };
        for (name, cpu_branch, gpu_branch) in [
            ("input", &cpu_eml.input, &gpu_eml.input),
            ("self", &cpu_eml.self_params, &gpu_eml.self_params),
        ] {
            close(
                &format!("EML {name} Muon weights step {step}"),
                &gpu_branch.weights,
                &cpu_branch.weights,
                8e-4,
            );
            close(
                &format!("EML {name} Muon biases step {step}"),
                &gpu_branch.biases,
                &cpu_branch.biases,
                8e-4,
            );
        }
    }
}

#[test]
fn gpu_muon_moe_device_gradients_match_cpu_across_steps() {
    let net = network(
        r#"[net]
width=3
height=2
channels=2
batch=2
muon=1
decoupled_decay=1
decay=0.0001
[moe_convolutional]
filters=2
size=1
experts=2
top_k=0
load_balance=0.05
temperature=1.4
batch_normalize=0
activation=leaky
"#,
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 97);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("MoE Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero MoE Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU MoE Muon forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide MoE Muon forward");
        assert!(gpu_forward.cpu_forward.moe_caches[0].is_none());
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download MoE Muon output");
        close(
            &format!("MoE Muon output step {step}"),
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            3e-4,
        );
        for microbatch in 0..2 {
            let seed = Tensor::filled(
                cpu.network.layers[0].output_shape,
                0.025 + step as f32 * 0.008 + microbatch as f32 * 0.004,
            );
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU MoE Muon backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide MoE Muon backward");
        }
        cpu.update_weights_muon(0.01, 2)
            .expect("CPU MoE Muon update");
        gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 2)
            .expect("Oxide MoE Muon update");
        let (LayerParams::Moe(cpu_moe), LayerParams::Moe(gpu_moe)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("MoE parameter layout expected");
        };
        close(
            &format!("MoE router weights step {step}"),
            &gpu_moe.router_weights,
            &cpu_moe.router_weights,
            8e-4,
        );
        close(
            &format!("MoE router biases step {step}"),
            &gpu_moe.router_biases,
            &cpu_moe.router_biases,
            8e-4,
        );
        for (index, (cpu_expert, gpu_expert)) in
            cpu_moe.experts.iter().zip(&gpu_moe.experts).enumerate()
        {
            close(
                &format!("MoE expert {index} weights step {step}"),
                &gpu_expert.weights,
                &cpu_expert.weights,
                8e-4,
            );
            close(
                &format!("MoE expert {index} biases step {step}"),
                &gpu_expert.biases,
                &cpu_expert.biases,
                8e-4,
            );
        }
    }
}

#[test]
fn gpu_muon_dcnv4_offset_projection_matches_cpu_across_steps() {
    let net = network(
        r#"[net]
width=3
height=2
channels=2
batch=2
muon=1
decoupled_decay=1
decay=0.0001
[dcnv4]
filters=2
size=1
groups=2
d_stride=1
batch_normalize=0
activation=leaky
"#,
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Dcnv4(params) = &mut cpu.params[0] else {
        panic!("DCNv4 parameters")
    };
    for (index, weight) in params.offset_weights.iter_mut().enumerate() {
        *weight = (index as f32 - 3.0) * 0.004;
    }
    for (index, bias) in params.offset_biases.iter_mut().enumerate() {
        *bias = (index as f32 - 2.0) * 0.03;
    }
    params.biases.copy_from_slice(&[0.02, -0.015]);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("DCNv4 Muon buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero DCNv4 Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU DCNv4 forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide DCNv4 forward");
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download DCNv4 output");
        close(
            &format!("DCNv4 Muon output step {step}"),
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            3e-4,
        );
        for microbatch in 0..2 {
            let seed = Tensor::filled(
                cpu.network.layers[0].output_shape,
                0.035 + step as f32 * 0.006 + microbatch as f32 * 0.003,
            );
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU DCNv4 Muon backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("Oxide DCNv4 Muon backward");
        }
        cpu.update_weights_muon(0.01, 2)
            .expect("CPU DCNv4 Muon update");
        gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 2)
            .expect("Oxide DCNv4 Muon update");
        let (LayerParams::Dcnv4(cpu_params), LayerParams::Dcnv4(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("DCNv4 parameter layout expected");
        };
        close(
            &format!("DCNv4 Muon offset weights step {step}"),
            &gpu_params.offset_weights,
            &cpu_params.offset_weights,
            8e-4,
        );
        close(
            &format!("DCNv4 Muon offset biases step {step}"),
            &gpu_params.offset_biases,
            &cpu_params.offset_biases,
            8e-4,
        );
        close(
            &format!("DCNv4 Muon output biases step {step}"),
            &gpu_params.biases,
            &cpu_params.biases,
            8e-4,
        );
    }
}

#[test]
fn gpu_muon_local_location_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [local]\nfilters=3\nsize=2\nstride=1\npadding=0\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 67);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("local Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero local Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU local Muon forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide local Muon forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("local Muon step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                2e-4,
            );
        }
        let seed = Tensor::filled(
            cpu.network.layers[0].output_shape,
            0.02 + step as f32 * 0.005,
        );
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU local Muon backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide local Muon backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU local Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU local Muon update");
        let (LayerParams::Local(cpu_local), LayerParams::Local(gpu_local)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("local parameter layout expected");
        };
        close(
            &format!("local Muon weights step {step}"),
            &gpu_local.weights,
            &cpu_local.weights,
            1e-3,
        );
        close(
            &format!("local Muon biases step {step}"),
            &gpu_local.biases,
            &cpu_local.biases,
            1e-3,
        );
    }
}

#[test]
fn gpu_muon_crnn_device_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\ntime_steps=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [crnn]\nhidden=3\noutput=2\nsize=1\nstride=1\nactivation=tanh\nshortcut=1\nbatch_normalize=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 79);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("CRNN Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero CRNN Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU CRNN Muon forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide CRNN Muon forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("CRNN Muon step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::filled(output_shape, 0.014 + step as f32 * 0.003);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU CRNN Muon backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide CRNN Muon backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU CRNN Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU CRNN Muon update");
        let (LayerParams::Crnn(cpu_crnn), LayerParams::Crnn(gpu_crnn)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("CRNN parameter layout expected");
        };
        for (name, cpu_projection, gpu_projection) in [
            ("input", &cpu_crnn.input, &gpu_crnn.input),
            ("self", &cpu_crnn.self_params, &gpu_crnn.self_params),
            ("output", &cpu_crnn.output, &gpu_crnn.output),
        ] {
            close(
                &format!("CRNN Muon {name} weights step {step}"),
                &gpu_projection.weights,
                &cpu_projection.weights,
                1e-3,
            );
            close(
                &format!("CRNN Muon {name} biases step {step}"),
                &gpu_projection.biases,
                &cpu_projection.biases,
                1e-3,
            );
            let cpu_bn = cpu_projection.batch_norm.as_ref().expect("CPU CRNN BN");
            let gpu_bn = gpu_projection.batch_norm.as_ref().expect("GPU CRNN BN");
            close(
                &format!("CRNN Muon {name} BN scales step {step}"),
                &gpu_bn.scales,
                &cpu_bn.scales,
                1e-3,
            );
        }
    }
}

#[test]
fn gpu_muon_rnn_device_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 71);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("RNN Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero RNN Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU RNN Muon forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide RNN Muon forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("RNN Muon step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::filled(output_shape, 0.015 + step as f32 * 0.005);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU RNN Muon backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide RNN Muon backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU RNN Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU RNN Muon update");
        let (LayerParams::Rnn(cpu_rnn), LayerParams::Rnn(gpu_rnn)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("RNN parameter layout expected");
        };
        for (name, cpu_group, gpu_group) in [
            ("input", &cpu_rnn.input, &gpu_rnn.input),
            ("self", &cpu_rnn.self_params, &gpu_rnn.self_params),
            ("output", &cpu_rnn.output, &gpu_rnn.output),
        ] {
            close(
                &format!("RNN Muon {name} weights step {step}"),
                &gpu_group.weights,
                &cpu_group.weights,
                1e-3,
            );
            close(
                &format!("RNN Muon {name} biases step {step}"),
                &gpu_group.biases,
                &cpu_group.biases,
                1e-3,
            );
        }
    }
}

#[test]
fn gpu_muon_lstm_device_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\ninputs=2\nbatch=2\ntime_steps=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [lstm]\noutput=3\nbatch_normalize=0\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 73);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("LSTM Muon train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero LSTM Muon buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU LSTM Muon forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide LSTM Muon forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("LSTM Muon step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::filled(output_shape, 0.012 + step as f32 * 0.004);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU LSTM Muon backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide LSTM Muon backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU LSTM Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU LSTM Muon update");
        let (LayerParams::Lstm(cpu_lstm), LayerParams::Lstm(gpu_lstm)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("LSTM parameter layout expected");
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
                &format!("LSTM Muon {name} weights step {step}"),
                &gpu_projection.weights,
                &cpu_projection.weights,
                1e-3,
            );
            close(
                &format!("LSTM Muon {name} biases step {step}"),
                &gpu_projection.biases,
                &cpu_projection.biases,
                1e-3,
            );
        }
    }
}

#[test]
fn gpu_muon_kimi_kda_host_gradient_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [kimi_kda]\nfilters=2\nrank=1\nexperts=2\ntop_k=1\nhidden=2\ntemperature=1\nload_balance=0\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 59);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("Kimi KDA train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero Kimi KDA buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU Kimi KDA forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide Kimi KDA forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("Kimi KDA step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.02 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU Kimi KDA backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide Kimi KDA backward");
        cpu.update_weights_muon(0.02, 1)
            .expect("CPU Kimi KDA Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU Kimi KDA Muon update");
        let (LayerParams::KimiKda(cpu_kimi), LayerParams::KimiKda(gpu_kimi)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("Kimi KDA parameter layout expected");
        };
        close(
            &format!("Kimi KDA weights step {step}"),
            &gpu_kimi.weights,
            &cpu_kimi.weights,
            1e-3,
        );
        close(
            &format!("Kimi KDA biases step {step}"),
            &gpu_kimi.biases,
            &cpu_kimi.biases,
            1e-3,
        );
    }
}

#[test]
fn gpu_muon_mambavision_host_gradient_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [mambavision]\nfilters=2\nstate=2\nconv_size=1\ndt_rank=1\nffn_ratio=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 53);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("MambaVision train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu()
            .expect("zero MambaVision buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu
            .forward_training(&input)
            .expect("CPU MambaVision forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide MambaVision forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("MambaVision step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.025 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU MambaVision backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide MambaVision backward");
        cpu.update_weights_muon(0.02, 1)
            .expect("CPU MambaVision Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU MambaVision Muon update");
        let (LayerParams::MambaVision(cpu_mamba), LayerParams::MambaVision(gpu_mamba)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("MambaVision parameter layout expected");
        };
        close(
            &format!("MambaVision weights step {step}"),
            &gpu_mamba.values,
            &cpu_mamba.values,
            1e-3,
        );
    }
}

#[test]
fn gpu_muon_tucker_host_gradient_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [tucker_attention]\nfilters=2\nsize=2\nheads=1\nrank_q=1\nrank_k=1\nrank_v=1\nrank_o=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 47);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("Tucker train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero Tucker buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU Tucker forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide Tucker forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("Tucker step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.025 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU Tucker backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide Tucker backward");
        cpu.update_weights_muon(0.02, 1)
            .expect("CPU Tucker Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU Tucker Muon update");
        let (LayerParams::TuckerAttention(cpu_tucker), LayerParams::TuckerAttention(gpu_tucker)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("Tucker parameter layout expected");
        };
        close(
            &format!("Tucker weights step {step}"),
            &gpu_tucker.weights,
            &cpu_tucker.weights,
            1e-3,
        );
        close(
            &format!("Tucker biases step {step}"),
            &gpu_tucker.biases,
            &cpu_tucker.biases,
            1e-3,
        );
    }
}

#[test]
fn gpu_muon_swin_host_gradient_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [swin]\nfilters=2\nsize=2\nheads=1\nshift=0\nffn_ratio=1\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 43);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("Swin train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero Swin buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU Swin forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide Swin forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("Swin step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.035 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU Swin backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide Swin backward");
        cpu.update_weights_muon(0.02, 1)
            .expect("CPU Swin Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU Swin Muon update");
        let (LayerParams::Swin(cpu_swin), LayerParams::Swin(gpu_swin)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("Swin parameter layout expected");
        };
        for (name, cpu_values, gpu_values) in [
            ("QKV weights", &cpu_swin.qkv_weight, &gpu_swin.qkv_weight),
            ("attention projection", &cpu_swin.wo, &gpu_swin.wo),
            ("FFN first weights", &cpu_swin.ffn_w1, &gpu_swin.ffn_w1),
            ("FFN second weights", &cpu_swin.ffn_w2, &gpu_swin.ffn_w2),
            ("QKV bias", &cpu_swin.qkv_bias, &gpu_swin.qkv_bias),
            ("attention bias", &cpu_swin.wo_bias, &gpu_swin.wo_bias),
            ("FFN first bias", &cpu_swin.ffn_b1, &gpu_swin.ffn_b1),
            ("FFN second bias", &cpu_swin.ffn_b2, &gpu_swin.ffn_b2),
            ("LN1 gamma", &cpu_swin.ln1_gamma, &gpu_swin.ln1_gamma),
            ("LN1 beta", &cpu_swin.ln1_beta, &gpu_swin.ln1_beta),
            ("LN2 gamma", &cpu_swin.ln2_gamma, &gpu_swin.ln2_gamma),
            ("LN2 beta", &cpu_swin.ln2_beta, &gpu_swin.ln2_beta),
            (
                "relative position bias",
                &cpu_swin.relative_position_bias,
                &gpu_swin.relative_position_bias,
            ),
        ] {
            close(
                &format!("Swin {name} step {step}"),
                gpu_values,
                cpu_values,
                1e-3,
            );
        }
        assert_eq!(gpu_swin.res_proj, cpu_swin.res_proj);
        assert_eq!(gpu_swin.mhc, cpu_swin.mhc);
    }
}

#[test]
fn gpu_muon_vit_host_gradient_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0001\n\
         [vit]\ndim=2\nheads=1\npatch_size=1\npatch_stride=1\nmlp_dim=2\nflash_attention=0\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    darknet_core::initialize_training_params(&net, &mut cpu.params, 41);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("ViT train buffers");

    for step in 1..=2 {
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero ViT buffers");
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU ViT forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide ViT forward");
        for (cpu_output, gpu_output) in cpu_forward
            .layer_outputs
            .iter()
            .zip(&gpu_forward.cpu_forward.layer_outputs)
        {
            close(
                &format!("ViT step {step} forward"),
                &cpu_output.data,
                &gpu_output.data,
                3e-4,
            );
        }
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.04 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU ViT backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide ViT backward");
        cpu.update_weights_muon(0.02, 1)
            .expect("CPU ViT Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU ViT Muon update");
        let (LayerParams::Vit(cpu_vit), LayerParams::Vit(gpu_vit)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("ViT parameter layout expected");
        };
        for (name, cpu_values, gpu_values) in [
            (
                "patch embedding",
                &cpu_vit.patch_embed,
                &gpu_vit.patch_embed,
            ),
            ("QKV weights", &cpu_vit.qkv_weight, &gpu_vit.qkv_weight),
            ("attention projection", &cpu_vit.wo, &gpu_vit.wo),
            ("FFN first weights", &cpu_vit.ffn_w1, &gpu_vit.ffn_w1),
            ("FFN second weights", &cpu_vit.ffn_w2, &gpu_vit.ffn_w2),
            ("patch bias", &cpu_vit.patch_bias, &gpu_vit.patch_bias),
            ("QKV bias", &cpu_vit.qkv_bias, &gpu_vit.qkv_bias),
            ("attention bias", &cpu_vit.wo_bias, &gpu_vit.wo_bias),
            ("FFN first bias", &cpu_vit.ffn_b1, &gpu_vit.ffn_b1),
            ("FFN second bias", &cpu_vit.ffn_b2, &gpu_vit.ffn_b2),
            ("LN1 gamma", &cpu_vit.ln1_gamma, &gpu_vit.ln1_gamma),
            ("LN1 beta", &cpu_vit.ln1_beta, &gpu_vit.ln1_beta),
            ("LN2 gamma", &cpu_vit.ln2_gamma, &gpu_vit.ln2_gamma),
            ("LN2 beta", &cpu_vit.ln2_beta, &gpu_vit.ln2_beta),
        ] {
            close(
                &format!("ViT {name} step {step}"),
                gpu_values,
                cpu_values,
                1e-3,
            );
        }
        assert_eq!(gpu_vit.pos_embed, cpu_vit.pos_embed);
        assert_eq!(gpu_vit.mhc, cpu_vit.mhc);
    }
}

#[test]
fn gpu_muon_graph_conv_matrices_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\n\
         [graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU GraphConv forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide GraphConv forward");
        close(
            "GPU Muon GraphConv output",
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            &cpu_forward.layer_outputs[0].data,
            6e-5,
        );
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::from_data(
            output_shape,
            (0..output_shape.element_count() * output_shape.batch)
                .map(|index| (index as f32 * 0.37 + step as f32 * 0.19).sin() * 0.3)
                .collect(),
        )
        .unwrap();
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU GraphConv backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide GraphConv backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU GraphConv Muon");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU GraphConv Muon");

        let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("GraphConv parameters expected after Muon update");
        };
        close(
            "GPU Muon GraphConv projection weights",
            &gpu_params.projection.weights,
            &cpu_params.projection.weights,
            8e-4,
        );
        close(
            "GPU Muon GraphConv projection biases",
            &gpu_params.projection.biases,
            &cpu_params.projection.biases,
            8e-4,
        );
        close(
            "GPU Muon GraphConv self weights",
            gpu_params.self_weights.as_ref().unwrap(),
            cpu_params.self_weights.as_ref().unwrap(),
            8e-4,
        );
        close(
            "GPU Muon GraphConv edge kernel",
            gpu_params.edge_kernel.as_ref().unwrap(),
            cpu_params.edge_kernel.as_ref().unwrap(),
            8e-4,
        );
        close(
            "GPU Muon GraphConv edge biases",
            gpu_params.edge_biases.as_ref().unwrap(),
            cpu_params.edge_biases.as_ref().unwrap(),
            8e-4,
        );
    }

    let dir = std::env::temp_dir().join(format!(
        "oxide-runtime-graph-conv-muon-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let cpu_state = dir.join("cpu.state");
    let gpu_state = dir.join("gpu.state");
    cpu.save_optimizer_state(&cpu_state)
        .expect("save CPU GraphConv Muon state");
    gpu.save_optimizer_state(&gpu_state)
        .expect("save GPU GraphConv Muon state");

    let mut resumed_cpu = ExecutableNetwork::new(net.clone());
    resumed_cpu.params = cpu.params.clone();
    resumed_cpu
        .load_optimizer_state(&cpu_state)
        .expect("load CPU GraphConv Muon state");
    let mut resumed_gpu =
        GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide resume"));
    resumed_gpu.params = gpu.params.clone();
    resumed_gpu
        .load_optimizer_state(&gpu_state)
        .expect("load GPU GraphConv Muon state");

    let input = patterned(resumed_cpu.network.net.input_shape());
    let cpu_forward = resumed_cpu
        .forward_training(&input)
        .expect("resumed CPU GraphConv forward");
    let gpu_forward = resumed_gpu
        .forward_training_with_state(&input)
        .expect("resumed Oxide GraphConv forward");
    close(
        "resumed GPU Muon GraphConv output",
        &gpu_forward.cpu_forward.layer_outputs[0].data,
        &cpu_forward.layer_outputs[0].data,
        6e-5,
    );
    let output_shape = resumed_cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.element_count() * output_shape.batch)
            .map(|index| (index as f32 * 0.23 - 0.41).cos() * 0.25)
            .collect(),
    )
    .unwrap();
    let seeds = [Some(seed)];
    resumed_cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("resumed CPU GraphConv backward");
    resumed_gpu
        .backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("resumed Oxide GraphConv backward");
    resumed_cpu
        .update_weights_muon(0.02, 2)
        .expect("resumed CPU GraphConv Muon update");
    resumed_gpu
        .update_weights_gpu(0.02, 0.9, net.net.decay, 2)
        .expect("resumed GPU GraphConv Muon update");
    let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
        (&resumed_cpu.params[0], &resumed_gpu.params[0])
    else {
        panic!("GraphConv parameters expected after resumed Muon update");
    };
    close(
        "resumed GPU Muon GraphConv projection weights",
        &gpu_params.projection.weights,
        &cpu_params.projection.weights,
        8e-4,
    );
    close(
        "resumed GPU Muon GraphConv projection biases",
        &gpu_params.projection.biases,
        &cpu_params.projection.biases,
        8e-4,
    );
    close(
        "resumed GPU Muon GraphConv self weights",
        gpu_params.self_weights.as_ref().unwrap(),
        cpu_params.self_weights.as_ref().unwrap(),
        8e-4,
    );
    close(
        "resumed GPU Muon GraphConv edge kernel",
        gpu_params.edge_kernel.as_ref().unwrap(),
        cpu_params.edge_kernel.as_ref().unwrap(),
        8e-4,
    );
    close(
        "resumed GPU Muon GraphConv edge biases",
        gpu_params.edge_biases.as_ref().unwrap(),
        cpu_params.edge_biases.as_ref().unwrap(),
        8e-4,
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn gpu_muon_graph_conv_batch_norm_native_training_matches_cpu() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\n\
         [graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nbatch_normalize=1\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU GraphConv BN forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide GraphConv BN native forward");
        assert!(gpu_forward.gpu_batch_norm_caches[0].is_some());
        close(
            "GPU Muon GraphConv BN native output",
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            &cpu_forward.layer_outputs[0].data,
            3e-5,
        );
        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::from_data(
            output_shape,
            (0..output_shape.element_count() * output_shape.batch)
                .map(|index| (index as f32 * 0.33 + step as f32 * 0.21).cos() * 0.2)
                .collect(),
        )
        .unwrap();
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU GraphConv BN backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide GraphConv BN native backward");
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU GraphConv BN Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU GraphConv BN Muon update");

        let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("GraphConv parameters expected after BN Muon update");
        };
        close(
            "GPU Muon GraphConv BN projection weights",
            &gpu_params.projection.weights,
            &cpu_params.projection.weights,
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN projection biases",
            &gpu_params.projection.biases,
            &cpu_params.projection.biases,
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN self weights",
            gpu_params.self_weights.as_ref().unwrap(),
            cpu_params.self_weights.as_ref().unwrap(),
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN edge kernel",
            gpu_params.edge_kernel.as_ref().unwrap(),
            cpu_params.edge_kernel.as_ref().unwrap(),
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN edge biases",
            gpu_params.edge_biases.as_ref().unwrap(),
            cpu_params.edge_biases.as_ref().unwrap(),
            8e-4,
        );
        let cpu_bn = cpu_params.projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_params.projection.batch_norm.as_ref().unwrap();
        close(
            "GPU Muon GraphConv BN gamma",
            &gpu_bn.scales,
            &cpu_bn.scales,
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN beta",
            &gpu_bn.biases,
            &cpu_bn.biases,
            8e-4,
        );
        close(
            "GPU Muon GraphConv BN rolling mean",
            &gpu_bn.rolling_mean,
            &cpu_bn.rolling_mean,
            1e-6,
        );
        close(
            "GPU Muon GraphConv BN rolling variance",
            &gpu_bn.rolling_variance,
            &cpu_bn.rolling_variance,
            1e-6,
        );
    }
}

#[test]
fn gpu_muon_graph_conv_dont_update_gate_preserves_parameters() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\n\
         [graph_conv]\nfilters=2\ngroups=1\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nactivation=leaky\ndont_update=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let initial = cpu.params.clone();
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = initial.clone();

    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU GraphConv forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("Oxide GraphConv forward");
    close(
        "gated GPU Muon GraphConv output",
        &gpu_forward.cpu_forward.layer_outputs[0].data,
        &cpu_forward.layer_outputs[0].data,
        6e-5,
    );
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.element_count() * output_shape.batch)
            .map(|index| (index as f32 * 0.29 + 0.4).sin() * 0.3)
            .collect(),
    )
    .unwrap();
    let seeds = [Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU GraphConv backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("Oxide GraphConv backward");
    cpu.update_weights_muon(0.02, 2)
        .expect("CPU gated GraphConv Muon update");
    gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
        .expect("GPU gated GraphConv Muon update");
    assert_eq!(cpu.params, initial);
    assert_eq!(gpu.params, initial);
    assert_eq!(cpu.adam_step(), 1);
    assert_eq!(gpu.adam_step(), 1);
}

#[test]
fn gpu_muon_connected_two_step_update_matches_cpu() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\nloss_scale=8\n\
         [connected]\noutput=3\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    if let LayerParams::Connected(params) = &mut cpu.params[0] {
        params.weights = vec![0.2, -0.1, 0.3, 0.4, -0.25, 0.15];
        params.biases = vec![0.03, -0.02, 0.01];
    } else {
        panic!("connected parameters expected");
    }
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide forward");
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.07 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide backward");
        cpu.update_weights_muon(0.02, 2).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU Muon update");
        let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("connected parameters expected");
        };
        close("GPU Muon connected weights", &gpu_params.weights, &cpu_params.weights, 8e-4);
        close("GPU Muon connected biases", &gpu_params.biases, &cpu_params.biases, 8e-4);
    }

    let dir = std::env::temp_dir().join(format!("oxide-runtime-muon-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cpu_state = dir.join("cpu.state");
    let gpu_state = dir.join("gpu.state");
    cpu.save_optimizer_state(&cpu_state).expect("save CPU state");
    gpu.save_optimizer_state(&gpu_state).expect("save GPU state");
    let mut resumed_cpu = ExecutableNetwork::new(net.clone());
    resumed_cpu.params = cpu.params.clone();
    resumed_cpu
        .load_optimizer_state(&cpu_state)
        .expect("load CPU state");
    let mut resumed_gpu =
        GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide resume"));
    resumed_gpu.params = gpu.params.clone();
    resumed_gpu
        .load_optimizer_state(&gpu_state)
        .expect("load GPU state");
    let input = patterned(resumed_cpu.network.net.input_shape());
    let cpu_forward = resumed_cpu.forward_training(&input).expect("resumed CPU forward");
    let gpu_forward = resumed_gpu
        .forward_training_with_state(&input)
        .expect("resumed GPU forward");
    let seed = Tensor::filled(resumed_cpu.network.layers[0].output_shape, -0.04);
    let seeds = [Some(seed)];
    resumed_cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("resumed CPU backward");
    resumed_gpu
        .backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("resumed GPU backward");
    resumed_cpu.update_weights_muon(0.02, 2).expect("resumed CPU update");
    resumed_gpu
        .update_weights_gpu(0.02, 0.9, net.net.decay, 2)
        .expect("resumed GPU update");
    let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
        (&resumed_cpu.params[0], &resumed_gpu.params[0])
    else {
        panic!("connected parameters expected after resume");
    };
    close("resumed GPU Muon weights", &gpu_params.weights, &cpu_params.weights, 8e-4);
    close("resumed GPU Muon biases", &gpu_params.biases, &cpu_params.biases, 8e-4);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn gpu_muon_connected_batch_norm_matches_cpu_across_steps() {
    connected_bn_muon_matches_cpu("linear");
    connected_bn_muon_matches_cpu("leaky");
}

fn connected_bn_muon_matches_cpu(activation: &str) {
    let cfg = format!(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0002\nloss_scale=1\n\
         [connected]\noutput=3\nbatch_normalize=1\nactivation={activation}\n"
    );
    let net = network(&cfg);
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Connected(params) = &mut cpu.params[0] else {
        panic!("connected parameters expected");
    };
    params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4, -0.25, 0.15]);
    params.biases.copy_from_slice(&[0.03, -0.02, 0.01]);
    let bn = params.batch_norm.as_mut().expect("connected BN parameters");
    bn.scales.copy_from_slice(&[0.7, 1.1, 0.85]);
    bn.biases.copy_from_slice(&[-0.04, 0.02, 0.06]);
    bn.rolling_mean.copy_from_slice(&[0.1, -0.05, 0.08]);
    bn.rolling_variance.copy_from_slice(&[0.8, 1.2, 0.65]);

    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide connected BN forward");
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download connected BN output");
        close(
            &format!("connected BN forward step {step}"),
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            3e-5,
        );
        let seed = Tensor::filled(
            cpu.network.layers[0].output_shape,
            0.05 * step as f32,
        );
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU connected BN backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("GPU Muon connected BN backward");
        let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("connected parameters expected after backward");
        };
        close(
            &format!("connected BN rolling mean before update step {step}"),
            &gpu_params.batch_norm.as_ref().unwrap().rolling_mean,
            &cpu_params.batch_norm.as_ref().unwrap().rolling_mean,
            1e-6,
        );
        cpu.update_weights_muon(0.02, 2)
            .expect("CPU connected BN Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU connected BN Muon update");
        let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("connected parameters expected after update");
        };
        close(
            &format!("connected BN Muon weights step {step}"),
            &gpu_params.weights,
            &cpu_params.weights,
            8e-4,
        );
        close(
            &format!("connected BN Muon biases step {step}"),
            &gpu_params.biases,
            &cpu_params.biases,
            8e-4,
        );
        let cpu_bn = cpu_params.batch_norm.as_ref().expect("CPU BN state");
        let gpu_bn = gpu_params.batch_norm.as_ref().expect("GPU BN state");
        close(
            &format!("connected BN Muon scales step {step}"),
            &gpu_bn.scales,
            &cpu_bn.scales,
            8e-4,
        );
        close(
            &format!("connected BN Muon rolling mean step {step}"),
            &gpu_bn.rolling_mean,
            &cpu_bn.rolling_mean,
            1e-6,
        );
        close(
            &format!("connected BN Muon rolling variance step {step}"),
            &gpu_bn.rolling_variance,
            &cpu_bn.rolling_variance,
            1e-6,
        );
    }
}

#[test]
fn gpu_connected_bn_backward_matches_cpu_sgd_and_adam() {
    for adam in [false, true] {
        let cfg = format!(
            "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nadam={}\ndecay=0\nloss_scale=1\n\
             [connected]\noutput=3\nbatch_normalize=1\nactivation=leaky\n",
            usize::from(adam)
        );
        let net = network(&cfg);
        let mut cpu = ExecutableNetwork::new(net.clone());
        let LayerParams::Connected(params) = &mut cpu.params[0] else {
            panic!("connected parameters expected");
        };
        params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4, -0.25, 0.15]);
        params.biases.copy_from_slice(&[0.03, -0.02, 0.01]);
        let bn = params.batch_norm.as_mut().expect("connected BN parameters");
        bn.scales.copy_from_slice(&[0.7, 1.1, 0.85]);
        bn.biases.copy_from_slice(&[-0.04, 0.02, 0.06]);
        bn.rolling_mean.copy_from_slice(&[0.1, -0.05, 0.08]);
        bn.rolling_variance.copy_from_slice(&[0.8, 1.2, 0.65]);
        let mut gpu = GpuExecutableNetwork::new(
            net.clone(),
            OxideBackend::new().expect("Oxide"),
        );
        gpu.params = cpu.params.clone();

        for step in 1..=2 {
            let input = patterned(cpu.network.net.input_shape());
            let cpu_forward = cpu.forward_training(&input).expect("CPU connected BN forward");
            let gpu_forward = gpu
                .forward_training_device(&input)
                .expect("Oxide connected BN forward");
            let gpu_output = gpu
                .backend
                .download_f32(&gpu_forward.gpu_layer_outputs[0])
                .expect("download connected BN output");
            close(
                &format!("connected BN optimizer forward adam={adam} step={step}"),
                &gpu_output,
                &cpu_forward.layer_outputs[0].data,
                3e-5,
            );
            let seed =
                Tensor::filled(cpu.network.layers[0].output_shape, 0.05 * step as f32);
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU connected BN backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("GPU connected BN backward");
            if adam {
                cpu.update_weights_adam(0.003, 2)
                    .expect("CPU connected BN Adam update");
            } else {
                cpu.update_weights(0.005, 0.3, 0.0, 2);
            }
            gpu.update_weights_gpu(if adam { 0.003 } else { 0.005 }, 0.3, 0.0, 2)
            .expect("GPU connected BN optimizer update");
            let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
                (&cpu.params[0], &gpu.params[0])
            else {
                unreachable!();
            };
            let label = format!("connected BN adam={adam} step={step}");
            close(
                &format!("{label} weights"),
                &gpu_params.weights,
                &cpu_params.weights,
                8e-4,
            );
            close(
                &format!("{label} biases"),
                &gpu_params.biases,
                &cpu_params.biases,
                8e-4,
            );
            let cpu_bn = cpu_params.batch_norm.as_ref().expect("CPU BN state");
            let gpu_bn = gpu_params.batch_norm.as_ref().expect("GPU BN state");
            close(
                &format!("{label} scales"),
                &gpu_bn.scales,
                &cpu_bn.scales,
                8e-4,
            );
            close(
                &format!("{label} rolling mean"),
                &gpu_bn.rolling_mean,
                &cpu_bn.rolling_mean,
                1e-6,
            );
            close(
                &format!("{label} rolling variance"),
                &gpu_bn.rolling_variance,
                &cpu_bn.rolling_variance,
                1e-6,
            );
        }
    }
}

#[test]
fn gpu_muon_multiple_connected_layers_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\nloss_scale=8\n\
         [connected]\noutput=3\nactivation=linear\n\
         [connected]\noutput=2\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    for (layer, params) in cpu.params.iter_mut().enumerate() {
        let LayerParams::Connected(params) = params else {
            panic!("connected parameters expected");
        };
        for (i, weight) in params.weights.iter_mut().enumerate() {
            *weight = ((i + layer * 7) as f32 * 0.23).sin() * 0.3;
        }
        for (i, bias) in params.biases.iter_mut().enumerate() {
            *bias = ((i + layer) as f32 * 0.41).cos() * 0.1;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide forward");
        let seed = Tensor::filled(cpu.network.layers[1].output_shape, 0.03 * step as f32);
        let seeds = [None, Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide backward");
        cpu.update_weights_muon(0.02, 2).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU Muon update");
        for layer in 0..2 {
            let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
                (&cpu.params[layer], &gpu.params[layer])
            else {
                panic!("connected parameters expected after update");
            };
            close(
                &format!("GPU Muon layer {layer} weights"),
                &gpu_params.weights,
                &cpu_params.weights,
                8e-4,
            );
            close(
                &format!("GPU Muon layer {layer} biases"),
                &gpu_params.biases,
                &cpu_params.biases,
                8e-4,
            );
        }
    }
}

#[test]
fn gpu_muon_connected_update_gate_matches_cpu_on_device() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\nloss_scale=8\n\
         [connected]\noutput=3\nactivation=linear\ndont_update=1\n\
         [connected]\noutput=2\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    for (layer, params) in cpu.params.iter_mut().enumerate() {
        let LayerParams::Connected(params) = params else {
            panic!("connected parameters expected");
        };
        for (i, weight) in params.weights.iter_mut().enumerate() {
            *weight = ((i + layer * 5) as f32 * 0.31).sin() * 0.25;
        }
        for (i, bias) in params.biases.iter_mut().enumerate() {
            *bias = ((i + layer) as f32 * 0.27).cos() * 0.08;
        }
    }
    let first_layer_initial = cpu.params[0].clone();
    let second_layer_initial = cpu.params[1].clone();
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide forward");
        let seed = Tensor::filled(cpu.network.layers[1].output_shape, 0.04 * step as f32);
        let seeds = [None, Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide backward");
        cpu.update_weights_muon(0.02, 2).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU Muon update");
        for layer in 0..2 {
            let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
                (&cpu.params[layer], &gpu.params[layer])
            else {
                panic!("connected parameters expected after update");
            };
            close(
                &format!("gated GPU Muon layer {layer} weights"),
                &gpu_params.weights,
                &cpu_params.weights,
                8e-4,
            );
            close(
                &format!("gated GPU Muon layer {layer} biases"),
                &gpu_params.biases,
                &cpu_params.biases,
                8e-4,
            );
        }
        if step == 1 {
            assert_eq!(cpu.params[0], first_layer_initial);
            assert_eq!(gpu.params[0], first_layer_initial);
            assert_ne!(cpu.params[1], second_layer_initial);
            assert_ne!(gpu.params[1], second_layer_initial);
            cpu.set_train_iteration(1);
            gpu.set_train_iteration(1);
        }
    }
}

#[test]
fn gpu_muon_connected_train_only_bn_gate_matches_cpu_on_device() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0002\n\
         [connected]\noutput=3\nactivation=linear\ntrain_only_bn=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Connected(params) = &mut cpu.params[0] else {
        panic!("connected parameters expected");
    };
    params
        .weights
        .copy_from_slice(&[0.2, -0.1, 0.3, 0.4, -0.25, 0.15]);
    params.biases.copy_from_slice(&[0.03, -0.02, 0.01]);
    let initial = cpu.params.clone();
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("Oxide connected forward");
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.03 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide backward");
        cpu.update_weights_muon(0.02, 2).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU Muon update");
        assert_eq!(
            cpu.params, initial,
            "CPU train_only_bn gate should skip weights"
        );
        assert_eq!(
            gpu.params, initial,
            "GPU train_only_bn gate should skip weights"
        );
    }
}

#[test]
fn gpu_muon_connected_burnin_gate_matches_cpu_on_device() {
    let net = network(
        "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nburn_in=2\nmuon=1\ndecoupled_decay=1\ndecay=0.0005\n\
         [connected]\noutput=2\nactivation=linear\nburnin_update=1\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Connected(params) = &mut cpu.params[0] else {
        panic!("connected parameters expected");
    };
    params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4]);
    params.biases.copy_from_slice(&[0.03, -0.02]);
    let initial = cpu.params.clone();
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for (iteration, seed_value) in [(0, 0.03), (2, -0.05)] {
        cpu.set_train_iteration(iteration);
        gpu.set_train_iteration(iteration);
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("Oxide forward");
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, seed_value);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide backward");
        cpu.update_weights_muon(0.02, 2).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 2)
            .expect("GPU Muon update");
        if iteration == 0 {
            assert_eq!(cpu.params, initial, "CPU burn-in gate must hold");
            assert_eq!(gpu.params, initial, "GPU burn-in gate must hold");
            continue;
        }
        let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("connected parameters expected after update");
        };
        close(
            "burn-in gated GPU Muon weights",
            &gpu_params.weights,
            &cpu_params.weights,
            8e-4,
        );
        close(
            "burn-in gated GPU Muon biases",
            &gpu_params.biases,
            &cpu_params.biases,
            8e-4,
        );
        assert_ne!(gpu.params, initial, "eligible layer should update after burn-in");
    }
}

#[test]
fn gpu_deconvolution_native_sgd_and_adam_updates_match_cpu_across_steps() {
    for (name, optimizer_cfg) in [
        ("SGD", "adam=0"),
        ("Adam", "adam=1\nb1=0.8\nb2=0.95\neps=0.001"),
    ] {
        let cfg = format!(
            "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nlearning_rate=0.01\n{optimizer_cfg}\n\
             [deconvolutional]\nfilters=1\nsize=2\nstride=2\nactivation=linear\n"
        );
        let net = network(&cfg);
        let mut cpu = ExecutableNetwork::new(net.clone());
        let LayerParams::Convolutional(params) = &mut cpu.params[0] else {
            panic!("deconvolution parameters expected");
        };
        params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4]);
        params.biases.copy_from_slice(&[0.05]);
        let mut gpu =
            GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide backend"));
        gpu.params = cpu.params.clone();

        for step in 1..=2 {
            cpu.zero_train_buffers();
            gpu.zero_train_buffers_gpu().expect("zero Oxide buffers");
            let input = patterned(cpu.network.net.input_shape());
            let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
            let gpu_forward = gpu
                .forward_training_device(&input)
                .expect("native Oxide deconvolution forward");
            let gpu_output = gpu
                .backend
                .download_f32(&gpu_forward.gpu_layer_outputs[0])
                .expect("download native output for parity");
            close(
                &format!("{name} step {step} forward"),
                &gpu_output,
                &cpu_forward.layer_outputs[0].data,
                3e-5,
            );
            let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.04 * step as f32);
            let seeds = [Some(seed)];
            cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
                .expect("CPU backward");
            gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
                .expect("native Oxide deconvolution backward");
            if net.net.adam {
                cpu.update_weights_adam(0.01, 1)
                    .expect("CPU deconvolution Adam update");
            } else {
                cpu.update_weights(0.01, 0.9, net.net.decay, 1);
            }
            gpu.update_weights_gpu(0.01, 0.9, net.net.decay, 1)
                .expect("host-staged deconvolution optimizer update");
            let (LayerParams::Convolutional(cpu_params), LayerParams::Convolutional(gpu_params)) =
                (&cpu.params[0], &gpu.params[0])
            else {
                panic!("deconvolution parameters expected after update");
            };
            close(
                &format!("{name} step {step} weights"),
                &gpu_params.weights,
                &cpu_params.weights,
                8e-4,
            );
            close(
                &format!("{name} step {step} biases"),
                &gpu_params.biases,
                &cpu_params.biases,
                8e-4,
            );
        }
    }
}

#[test]
fn gpu_deconvolution_native_forward_backward_and_muon_match_cpu_across_steps() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.0002\nloss_scale=8\n\
         [deconvolutional]\nfilters=1\nsize=2\nstride=2\nactivation=linear\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(params) = &mut cpu.params[0] else {
        panic!("deconvolution parameters expected");
    };
    params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4]);
    params.biases.copy_from_slice(&[0.05]);
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();

    for step in 1..=2 {
        let input = patterned(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
        let gpu_forward = gpu
            .forward_training_device(&input)
            .expect("native Oxide deconvolution forward");
        assert!(gpu_forward.cpu_forward.layer_outputs[0].data.is_empty());
        let gpu_output = gpu
            .backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download native output for parity assertion");
        close(
            "native deconvolution output",
            &gpu_output,
            &cpu_forward.layer_outputs[0].data,
            3e-5,
        );
        let seed = Tensor::filled(cpu.network.layers[0].output_shape, 0.025 * step as f32);
        let seeds = [Some(seed)];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("native Oxide deconvolution backward");
        cpu.update_weights_muon(0.02, 1).expect("CPU Muon update");
        gpu.update_weights_gpu(0.02, 0.9, net.net.decay, 1)
            .expect("GPU Muon update");
        let (LayerParams::Convolutional(cpu_params), LayerParams::Convolutional(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("deconvolution parameters expected after update");
        };
        close(
            "GPU Muon deconvolution weights",
            &gpu_params.weights,
            &cpu_params.weights,
            8e-4,
        );
        close(
            "GPU Muon deconvolution biases",
            &gpu_params.biases,
            &cpu_params.biases,
            8e-4,
        );
    }
}

#[test]
fn graph_conv_uniform_runtime_forward_backward_and_sgd_match_cpu() {
    let net = network("[net]\nwidth=4\nheight=3\nchannels=2\nbatch=2\n[convolutional]\nfilters=2\nsize=1\nactivation=leaky\n[graph_conv]\nfilters=4\ngroups=2\nsize=3\npad=1\nedge_mode=0\ngraph_use_self=0\nactivation=leaky\n");
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::Convolutional(conv) = &mut cpu.params[0] else {
        panic!("preceding convolution parameters expected");
    };
    conv.weights.copy_from_slice(&[0.2, -0.3, 0.4, 0.1]);
    conv.biases.copy_from_slice(&[0.05, -0.02]);
    let LayerParams::GraphConv(graph) = &mut cpu.params[1] else {
        panic!("GraphConv parameters expected");
    };
    graph.projection.weights.copy_from_slice(&[0.3, -0.2, 0.1, 0.4]);
    graph.projection.biases.copy_from_slice(&[0.01, -0.03, 0.02, 0.04]);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("GPU train buffers");

    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu.forward_training_with_state(&input).expect("native GraphConv GPU forward");
    for (index, (expected, actual)) in cpu_forward.layer_outputs.iter()
        .zip(&gpu_forward.cpu_forward.layer_outputs).enumerate()
    {
        close(&format!("GraphConv chain output layer {index}"), &actual.data, &expected.data, 4e-5);
    }

    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.29).cos() * 0.2)
            .collect(),
    ).expect("GraphConv seed");
    let seeds = [None, Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero GPU buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).expect("CPU backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds).expect("native GraphConv GPU backward");
    cpu.update_weights(0.01, 0.0, 0.0, 1);
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1).expect("GraphConv host-staged optimizer update");

    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else { panic!("preceding convolution parameters expected after update"); };
    close("GraphConv upstream input-gradient weight", &gpu_conv.weights, &cpu_conv.weights, 5e-5);
    close("GraphConv upstream input-gradient bias", &gpu_conv.biases, &cpu_conv.biases, 5e-5);
    let (LayerParams::GraphConv(cpu_graph), LayerParams::GraphConv(gpu_graph)) =
        (&cpu.params[1], &gpu.params[1])
    else { panic!("GraphConv parameters expected after update"); };
    close("GraphConv projection weight gradient", &gpu_graph.projection.weights, &cpu_graph.projection.weights, 5e-5);
    close("GraphConv projection bias gradient", &gpu_graph.projection.biases, &cpu_graph.projection.biases, 5e-5);

    cpu.network.net.adam = true;
    gpu.network.net.adam = true;
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero GraphConv Adam buffers");
    let cpu_forward = cpu.forward_training(&input).expect("CPU Adam forward");
    let gpu_forward = gpu.forward_training_with_state(&input).expect("Oxide Adam forward");
    let seeds = [None, Some(Tensor::filled(output_shape, 0.07))];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).expect("CPU Adam backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds).expect("Oxide Adam backward");
    cpu.update_weights_adam(0.01, 1).expect("CPU GraphConv Adam update");
    gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).expect("host-staged GraphConv Adam update");
    let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
        (&cpu.params[0], &gpu.params[0])
    else { panic!("preceding convolution parameters expected after Adam"); };
    close("GraphConv Adam upstream weights", &gpu_conv.weights, &cpu_conv.weights, 5e-5);
    close("GraphConv Adam upstream biases", &gpu_conv.biases, &cpu_conv.biases, 5e-5);
    let (LayerParams::GraphConv(cpu_graph), LayerParams::GraphConv(gpu_graph)) =
        (&cpu.params[1], &gpu.params[1])
    else { panic!("GraphConv parameters expected after Adam"); };
    close("GraphConv Adam projection weights", &gpu_graph.projection.weights, &cpu_graph.projection.weights, 5e-5);
    close("GraphConv Adam projection biases", &gpu_graph.projection.biases, &cpu_graph.projection.biases, 5e-5);
}

#[test]
fn graph_conv_learned_runtime_forward_backward_and_updates_match_cpu() {
    let net = network("[net]\nwidth=4\nheight=3\nchannels=2\nbatch=1\n[graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=0\nactivation=linear\n");
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("GraphConv parameters expected");
    };
    params.projection.weights.copy_from_slice(&[0.3, -0.2]);
    params.projection.biases.copy_from_slice(&[0.01, -0.03]);
    params.edge_kernel.as_mut().unwrap().copy_from_slice(
        &(0..36).map(|i| (i as f32 * 0.17).sin() * 0.08).collect::<Vec<_>>(),
    );
    params.edge_biases.as_mut().unwrap().copy_from_slice(
        &(0..18).map(|i| (i as f32 * 0.13).cos() * 0.03).collect::<Vec<_>>(),
    );
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("initialize learned GraphConv train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.31).cos() * 0.2)
            .collect(),
    ).expect("learned GraphConv seed");
    let seeds = [Some(seed)];

    for use_adam in [false, true] {
        cpu.network.net.adam = use_adam;
        gpu.network.net.adam = use_adam;
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero learned GraphConv gradients");
        let cpu_forward = cpu.forward_training(&input).expect("CPU learned GraphConv forward");
        let gpu_forward = gpu.forward_training_with_state(&input).expect("Oxide learned GraphConv forward");
        close(
            "learned GraphConv runtime forward",
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            &cpu_forward.layer_outputs[0].data,
            5e-5,
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU learned GraphConv backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide learned GraphConv backward");
        if use_adam {
            cpu.update_weights_adam(0.01, 1).expect("CPU learned GraphConv Adam");
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).expect("Oxide learned GraphConv Adam");
        } else {
            cpu.update_weights(0.01, 0.0, 0.0, 1);
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).expect("Oxide learned GraphConv SGD");
        }
        let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("GraphConv parameters expected after update");
        };
        close("learned GraphConv projection weights", &gpu_params.projection.weights, &cpu_params.projection.weights, 6e-5);
        close("learned GraphConv projection biases", &gpu_params.projection.biases, &cpu_params.projection.biases, 6e-5);
        close("learned GraphConv edge kernel", gpu_params.edge_kernel.as_ref().unwrap(), cpu_params.edge_kernel.as_ref().unwrap(), 6e-5);
        close("learned GraphConv edge biases", gpu_params.edge_biases.as_ref().unwrap(), cpu_params.edge_biases.as_ref().unwrap(), 6e-5);
    }
}

#[test]
fn graph_conv_self_projection_runtime_forward_backward_and_sgd_adam_match_cpu() {
    let net = network("[net]\nwidth=4\nheight=3\nchannels=2\nbatch=1\n[graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=0\ngraph_use_self=1\nactivation=linear\n");
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("GraphConv parameters expected");
    };
    params.projection.weights.copy_from_slice(&[0.3, -0.2]);
    params.projection.biases.copy_from_slice(&[0.01, -0.03]);
    params.self_weights.as_mut().unwrap().copy_from_slice(&[-0.15, 0.25]);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("initialize self GraphConv train buffers");
    let input = patterned(cpu.network.net.input_shape());
    let output_shape = cpu.network.layers[0].output_shape;
    let seeds = [Some(Tensor::filled(output_shape, 0.07))];

    for use_adam in [false, true] {
        cpu.network.net.adam = use_adam;
        gpu.network.net.adam = use_adam;
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero self GraphConv gradients");
        let cpu_forward = cpu.forward_training(&input).expect("CPU self GraphConv forward");
        let gpu_forward = gpu.forward_training_with_state(&input).expect("Oxide self GraphConv forward");
        close(
            "GraphConv self-projection runtime forward",
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            &cpu_forward.layer_outputs[0].data,
            5e-5,
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU self GraphConv backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("Oxide self GraphConv backward");
        if use_adam {
            cpu.update_weights_adam(0.01, 1).expect("CPU self GraphConv Adam");
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).expect("Oxide self GraphConv Adam");
        } else {
            cpu.update_weights(0.01, 0.0, 0.0, 1);
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).expect("Oxide self GraphConv SGD");
        }
        let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("GraphConv parameters expected after self update");
        };
        close("GraphConv self projection weights", &gpu_params.projection.weights, &cpu_params.projection.weights, 6e-5);
        close("GraphConv self projection biases", &gpu_params.projection.biases, &cpu_params.projection.biases, 6e-5);
        close("GraphConv self weights", gpu_params.self_weights.as_ref().unwrap(), cpu_params.self_weights.as_ref().unwrap(), 6e-5);
    }
}

#[test]
fn region_native_gpu_forward_backward_matches_cpu() {
    for softmax in [false, true] {
        let cfg = format!(
            "[net]\nwidth=3\nheight=2\nchannels=7\nbatch=2\n[region]\nclasses=2\ncoords=4\nnum=1\nsoftmax={}\n",
            usize::from(softmax),
        );
        let net = network(&cfg);
        let cpu = ExecutableNetwork::new(net.clone());
        let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
        let input = patterned(net.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).expect("CPU Region forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("native Region GPU forward");
        let backend = OxideBackend::new().expect("Oxide backend");
        let gpu_output = backend
            .download_f32(&gpu_forward.gpu_layer_outputs[0])
            .expect("download Region output");
        close(
            &format!("Region softmax={softmax} output"),
            &cpu_forward.layer_outputs[0].data,
            &gpu_output,
            2e-6,
        );
        let compact = gpu.forward_compact(&input).expect("compact Region forward");
        close(
            &format!("Region softmax={softmax} compact output"),
            &cpu_forward.layer_outputs[0].data,
            &compact.final_output.expect("compact final output").data,
            2e-6,
        );

        let output_shape = cpu.network.layers[0].output_shape;
        let seed = Tensor::from_data(
            output_shape,
            (0..output_shape.batch * output_shape.element_count())
                .map(|i| ((i * 7 % 19) as f32 - 8.0) * 0.03)
                .collect(),
        )
        .expect("Region seed");
        let cpu_backward = cpu
            .backward(&input, &cpu_forward, &seed)
            .expect("CPU Region backward");
        let seeds = [Some(seed.clone())];
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("native Region GPU backward");
        let seed_gpu = backend
            .upload_f32(&[2, 7, 2, 3], &seed.data)
            .expect("upload Region seed");
        let input_grad_gpu = backend
            .region_backward_f32(
                &gpu_forward.gpu_layer_outputs[0],
                &seed_gpu,
                &[2, 7, 2, 3],
                1,
                7,
                2,
                softmax,
                darknet_cuda::ElementwiseLaunchConfig::default(),
            )
            .expect("Region input-gradient primitive");
        let input_grad = backend
            .download_f32(&input_grad_gpu)
            .expect("download Region input gradient");
        close(
            &format!("Region softmax={softmax} input gradient"),
            &cpu_backward.input_gradient.data,
            &input_grad,
            3e-6,
        );
    }
}

#[test]
fn crop_native_training_augmentation_and_backward_match_cpu() {
    let net = network(
        "[net]\nwidth=4\nheight=3\nchannels=1\nbatch=2\nadam=0\n\
         [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
         [crop]\ncrop_width=3\ncrop_height=2\nflip=1\nangle=17\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let mut gpu = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().expect("Oxide"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let output_shape = cpu.network.layers[1].output_shape;

    for use_adam in [false, true] {
        cpu.network.net.adam = use_adam;
        gpu.network.net.adam = use_adam;
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero crop gradients");
        let cpu_forward = cpu.forward_training(&input).expect("CPU augmented Crop forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("native augmented Crop forward");
        close(
            "Crop training output",
            &gpu_forward.cpu_forward.layer_outputs[1].data,
            &cpu_forward.layer_outputs[1].data,
            5e-5,
        );
        let seeds = [None, Some(Tensor::filled(output_shape, 0.07))];
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU augmented Crop backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("native augmented Crop backward");
        if use_adam {
            cpu.update_weights_adam(0.01, 1).expect("CPU Crop-chain Adam");
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
                .expect("Oxide Crop-chain Adam");
        } else {
            cpu.update_weights(0.01, 0.0, 0.0, 1);
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
                .expect("Oxide Crop-chain SGD");
        }
        let (LayerParams::Convolutional(cpu_conv), LayerParams::Convolutional(gpu_conv)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("convolutional params expected before Crop");
        };
        close("Crop-chain convolution weights", &gpu_conv.weights, &cpu_conv.weights, 1e-4);
        close("Crop-chain convolution biases", &gpu_conv.biases, &cpu_conv.biases, 1e-4);
    }

    let unsupported_net = network(
        "[net]\nwidth=4\nheight=3\nchannels=1\nbatch=1\n\
         [crop]\ncrop_width=3\ncrop_height=2\nsaturation=1.2\n",
    );
    let unsupported_gpu = GpuExecutableNetwork::new(
        unsupported_net.clone(),
        OxideBackend::new().expect("Oxide"),
    );
    let unsupported_input = patterned(unsupported_net.net.input_shape());
    let error = match unsupported_gpu.forward_training_with_state(&unsupported_input) {
        Err(error) => error,
        Ok(_) => panic!("training color adjustment must be explicitly unsupported"),
    };
    assert!(error.to_string().contains("crop_gpu_training_color_augmentation"));
}

#[test]
fn standalone_batch_norm_native_training_matches_cpu_adam() {
    let net = network(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[batchnorm]\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::BatchNorm(params) = &mut cpu.params[0] else {
        panic!("standalone BatchNorm parameters expected");
    };
    params.scales.copy_from_slice(&[1.2, 0.85]);
    params.biases.copy_from_slice(&[0.2, -0.1]);
    params.rolling_mean.copy_from_slice(&[0.1, -0.2]);
    params.rolling_variance.copy_from_slice(&[0.8, 1.1]);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU BatchNorm forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("native Oxide BatchNorm forward");
    assert!(gpu_forward.gpu_batch_norm_caches[0].is_some());
    close(
        "standalone BatchNorm training output",
        &gpu_forward.cpu_forward.layer_outputs[0].data,
        &cpu_forward.layer_outputs[0].data,
        3e-5,
    );
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.element_count() * output_shape.batch)
            .map(|index| (index as f32 * 0.41).sin() * 0.3)
            .collect(),
    )
    .unwrap();
    let seeds = [Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU BatchNorm backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("native Oxide BatchNorm backward");
    cpu.update_weights_adam(0.01, 1).expect("CPU BatchNorm Adam");
    gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
        .expect("Oxide BatchNorm Adam fallback");
    let (LayerParams::BatchNorm(cpu_params), LayerParams::BatchNorm(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("standalone BatchNorm parameters expected after update");
    };
    close("standalone BatchNorm scales", &gpu_params.scales, &cpu_params.scales, 2e-6);
    close("standalone BatchNorm biases", &gpu_params.biases, &cpu_params.biases, 2e-6);
    close("standalone BatchNorm rolling mean", &gpu_params.rolling_mean, &cpu_params.rolling_mean, 1e-6);
    close("standalone BatchNorm rolling variance", &gpu_params.rolling_variance, &cpu_params.rolling_variance, 1e-3);
}

#[test]
fn graph_conv_batch_norm_native_training_forward_backward_sgd_adam_matches_cpu() {
    let net = network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nbatch_normalize=1\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("GraphConv parameters expected");
    };
    params.projection.weights.copy_from_slice(&[0.3, -0.2]);
    params.projection.biases.copy_from_slice(&[0.01, -0.03]);
    params.self_weights.as_mut().unwrap().copy_from_slice(&[-0.15, 0.25]);
    params.edge_kernel.as_mut().unwrap().copy_from_slice(
        &(0..36).map(|i| (i as f32 * 0.17).sin() * 0.08).collect::<Vec<_>>(),
    );
    params.edge_biases.as_mut().unwrap().copy_from_slice(
        &(0..18).map(|i| (i as f32 * 0.13).cos() * 0.03).collect::<Vec<_>>(),
    );
    let batch_norm = params.projection.batch_norm.as_mut().unwrap();
    batch_norm.scales.copy_from_slice(&[1.2, 0.85]);
    batch_norm.biases.copy_from_slice(&[0.01, -0.03]);
    batch_norm.rolling_mean.copy_from_slice(&[0.15, -0.1]);
    batch_norm.rolling_variance.copy_from_slice(&[0.8, 1.1]);

    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let output_shape = cpu.network.layers[0].output_shape;
    let seeds = [Some(Tensor::filled(output_shape, 0.07))];

    for use_adam in [false, true] {
        cpu.network.net.adam = use_adam;
        gpu.network.net.adam = use_adam;
        cpu.zero_train_buffers();
        gpu.zero_train_buffers_gpu().expect("zero GraphConv BN gradients");
        let cpu_forward = cpu.forward_training(&input).expect("CPU GraphConv BN forward");
        let gpu_forward = gpu
            .forward_training_with_state(&input)
            .expect("native Oxide GraphConv BN forward");
        assert!(gpu_forward.gpu_batch_norm_caches[0].is_some());
        close(
            "GraphConv BN native training output",
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            &cpu_forward.layer_outputs[0].data,
            6e-5,
        );
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU GraphConv BN backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("native Oxide GraphConv BN backward");
        if use_adam {
            cpu.update_weights_adam(0.01, 1).expect("CPU GraphConv BN Adam");
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
                .expect("Oxide GraphConv BN Adam");
        } else {
            cpu.update_weights(0.01, 0.0, 0.0, 1);
            gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
                .expect("Oxide GraphConv BN SGD");
        }
        let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
            (&cpu.params[0], &gpu.params[0])
        else {
            panic!("GraphConv parameters expected after BN update");
        };
        close("GraphConv BN projection weights", &gpu_params.projection.weights, &cpu_params.projection.weights, 8e-4);
        close("GraphConv BN projection biases", &gpu_params.projection.biases, &cpu_params.projection.biases, 8e-4);
        close("GraphConv BN self weights", gpu_params.self_weights.as_ref().unwrap(), cpu_params.self_weights.as_ref().unwrap(), 8e-4);
        close("GraphConv BN edge kernel", gpu_params.edge_kernel.as_ref().unwrap(), cpu_params.edge_kernel.as_ref().unwrap(), 8e-4);
        close("GraphConv BN edge biases", gpu_params.edge_biases.as_ref().unwrap(), cpu_params.edge_biases.as_ref().unwrap(), 8e-4);
        let cpu_bn = cpu_params.projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_params.projection.batch_norm.as_ref().unwrap();
        close("GraphConv BN gamma", &gpu_bn.scales, &cpu_bn.scales, 8e-4);
        close("GraphConv BN rolling mean", &gpu_bn.rolling_mean, &cpu_bn.rolling_mean, 1e-6);
        close("GraphConv BN rolling variance", &gpu_bn.rolling_variance, &cpu_bn.rolling_variance, 1e-3);
    }
}

#[test]
fn graph_conv_compact_inference_matches_cpu_with_learned_edges_and_self() {
    let net = network("[net]\nwidth=4\nheight=3\nchannels=2\nbatch=1\n[graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nactivation=leaky\n");
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("GraphConv parameters expected");
    };
    params.projection.weights.copy_from_slice(&[0.3, -0.2]);
    params.projection.biases.copy_from_slice(&[0.01, -0.03]);
    params.self_weights.as_mut().unwrap().copy_from_slice(&[-0.15, 0.25]);
    params.edge_kernel.as_mut().unwrap().copy_from_slice(
        &(0..36).map(|i| (i as f32 * 0.17).sin() * 0.08).collect::<Vec<_>>(),
    );
    params.edge_biases.as_mut().unwrap().copy_from_slice(
        &(0..18).map(|i| (i as f32 * 0.13).cos() * 0.03).collect::<Vec<_>>(),
    );
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_output = cpu.forward(&input).expect("CPU compact GraphConv reference");
    let compact = gpu.forward_compact(&input).expect("native compact GraphConv");
    close(
        "GraphConv compact inference",
        &compact.final_output.expect("GraphConv final output").data,
        &cpu_output.final_output().expect("CPU final output").data,
        6e-5,
    );
}

#[test]
fn graph_conv_compact_inference_matches_cpu_with_batch_norm() {
    let net = network(
        "[net]\nwidth=4\nheight=3\nchannels=2\nbatch=1\n[graph_conv]\nfilters=2\ngroups=2\nsize=3\npad=1\nedge_mode=1\ngraph_use_self=1\nbatch_normalize=1\nactivation=leaky\n",
    );
    let mut cpu = ExecutableNetwork::new(net.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("GraphConv parameters expected");
    };
    params.projection.weights.copy_from_slice(&[0.3, -0.2]);
    params.projection.biases.copy_from_slice(&[0.01, -0.03]);
    params
        .self_weights
        .as_mut()
        .unwrap()
        .copy_from_slice(&[-0.15, 0.25]);
    params.edge_kernel.as_mut().unwrap().copy_from_slice(
        &(0..36)
            .map(|i| (i as f32 * 0.17).sin() * 0.08)
            .collect::<Vec<_>>(),
    );
    params.edge_biases.as_mut().unwrap().copy_from_slice(
        &(0..18)
            .map(|i| (i as f32 * 0.13).cos() * 0.03)
            .collect::<Vec<_>>(),
    );
    let batch_norm = params.projection.batch_norm.as_mut().unwrap();
    batch_norm.scales.copy_from_slice(&[1.2, 0.85]);
    batch_norm.rolling_mean.copy_from_slice(&[0.15, -0.1]);
    batch_norm.rolling_variance.copy_from_slice(&[0.8, 1.1]);
    let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().expect("Oxide backend"));
    gpu.params = cpu.params.clone();
    let input = patterned(cpu.network.net.input_shape());
    let cpu_output = cpu
        .forward(&input)
        .expect("CPU GraphConv BN reference");
    let full = gpu.forward(&input).expect("native full GraphConv BN inference");
    close(
        "GraphConv full BN inference",
        &full.final_output().expect("full GraphConv output").data,
        &cpu_output.final_output().expect("CPU GraphConv output").data,
        6e-5,
    );
    let compact = gpu
        .forward_compact(&input)
        .expect("native compact GraphConv BN");
    close(
        "GraphConv compact BN inference",
        &compact.final_output.expect("GraphConv final output").data,
        &cpu_output.final_output().expect("CPU final output").data,
        6e-5,
    );
}
