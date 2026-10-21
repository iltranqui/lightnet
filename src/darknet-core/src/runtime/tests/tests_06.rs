use super::*;

#[test]
fn relu6_and_ramp_follow_cpp_activation_and_gradient() {
    let relu6 = |x: f32| super::activate(Activation::Relu6, x);
    assert_eq!(
        [relu6(-1.0), relu6(0.0), relu6(3.0), relu6(6.0), relu6(9.0)],
        [0.0, 0.0, 3.0, 6.0, 6.0]
    );
    let ramp = |x: f32| super::activate(Activation::Ramp, x);
    assert_eq!(ramp(-2.0), -0.2);
    assert_eq!(ramp(2.0), 2.0 + 0.2);
    let grad = |a: &Activation, out: f32| {
        super::activation_gradient_from_output(a, out, None, "t").expect("gradient")
    };
    // output form == C++ input form: (x>0 && x<6) and (x>0)+0.1
    for x in [-3.0f32, 0.0, 2.0, 5.99, 6.0, 8.0] {
        assert_eq!(
            grad(&Activation::Relu6, relu6(x)),
            (x > 0.0 && x < 6.0) as i32 as f32,
            "relu6 x={x}"
        );
        assert_eq!(
            grad(&Activation::Ramp, ramp(x)),
            (x > 0.0) as i32 as f32 + 0.1,
            "ramp x={x}"
        );
    }
    for (name, expected) in [("relu6", Activation::Relu6), ("ramp", Activation::Ramp)] {
        let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={name}\n"),
                )
                .expect("cfg"),
            )
            .expect("network");
        let LayerKind::Convolutional(spec) = &network.layers[0].kind else {
            panic!("expected convolution");
        };
        assert_eq!(spec.activation, expected);
        assert!(spec.activation.to_cuda_f32_kind().is_some());
    }
}

#[test]
fn extra_activations_parse_dispatch_and_use_the_exact_input_gradient() {
    let all = [
        ("gelu", Activation::Gelu),
        ("elu", Activation::Elu),
        ("selu", Activation::Selu),
        ("relie", Activation::Relie),
        ("plse", Activation::Plse),
        ("hardtan", Activation::Hardtan),
        ("lhtan", Activation::Lhtan),
        ("stair", Activation::Stair),
        ("hard_mish", Activation::HardMish),
        ("revleaky", Activation::Revleaky),
    ];
    for (name, expected) in &all {
        let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={name}\n"),
                )
                .expect("cfg"),
            )
            .expect("network");
        let LayerKind::Convolutional(spec) = &network.layers[0].kind else {
            panic!("expected convolution");
        };
        assert_eq!(&spec.activation, expected, "{name}");
        // dispatch -> central difference of `activate` vs the runtime gradient
        for x in [-1.7f32, -0.6, 0.4, 0.9, 2.3] {
            let h = 1e-2;
            let f = |v: f32| super::activate(expected.clone(), v);
            let fd = (f(x + h) - f(x - h)) / (2.0 * h);
            let grad = super::activation_gradient_from_output(expected, f(x), Some(x), "t")
                .expect("gradient");
            assert!(
                (fd - grad).abs() <= 3e-3 + 3e-3 * fd.abs(),
                "{name} x={x}: fd {fd}, grad {grad}"
            );
        }
    }
    // input-form gradients must refuse to run without the cached pre-activation
    assert!(super::activation_gradient_from_output(&Activation::Gelu, 0.5, None, "t").is_err());
}

#[test]
fn gpu_jev_ce_head_matches_cpu_forward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=1\nheight=1\nchannels=4\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\nstride=1\nactivation=linear\n\
             [jev_ce]\ngroups=2\ntemperature=1.5\n",
        1,
    );
}

#[test]
fn gpu_logistic_head_matches_cpu_forward_and_passes_the_seed_through() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [logistic]\n",
        1,
    );
}

#[test]
fn deconvolutional_layer_trains_on_cpu_and_gpu_native_ops_identically() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "inline.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation=leaky\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    fill_conv_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    assert!(
        !gpu.requires_cpu_backward_fallback(),
        "supported deconvolutional activation uses the GPU backward path"
    );

    let input = patterned_input(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    for (i, (a, b)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&gpu_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        assert_slice_close(&format!("layer {i}"), &a.data, &b.data, 1e-5);
    }
    let shape = cpu.network.layers[1].output_shape;
    let mut seeds = vec![None; 2];
    seeds[1] = Some(
        Tensor::from_data(
            shape,
            (0..2 * shape.element_count())
                .map(|i| (i as f32 * 0.91).cos())
                .collect(),
        )
        .unwrap(),
    );
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    let LayerTrainBuffers::Convolutional(cpu_train) = &cpu.train_buffers[1] else {
        panic!()
    };
    let fallback = gpu.cpu_fallback_train_buffers.borrow();
    let LayerTrainBuffers::Convolutional(gpu_train) = &fallback[1] else {
        panic!()
    };
    assert!(cpu_train.weight_updates.iter().any(|g| g.abs() > 1e-6));
    assert_slice_close(
        "weight_updates",
        &cpu_train.weight_updates,
        &gpu_train.weight_updates,
        1e-4,
    );
    assert_slice_close(
        "bias_updates",
        &cpu_train.bias_updates,
        &gpu_train.bias_updates,
        1e-4,
    );
    drop(fallback);

    // one SGD step moves the host-resident deconvolution weights on both paths alike
    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    let (LayerParams::Convolutional(a), LayerParams::Convolutional(b)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!()
    };
    assert_slice_close("updated weights", &a.weights, &b.weights, 1e-5);
}

#[test]
fn gpu_channel_shuffle_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\nstride=1\nactivation=linear\n\
             [channel_shuffle]\ngroups=3\n",
        1,
    );
}

#[test]
fn gpu_channel_slice_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [channel_slice]\nfrom=-1\nstart=1\nend=3\n",
        1,
    );
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [maxpool]\nsize=2\nstride=2\n\
             [channel_slice]\nfrom=-2\nstart=0\nend=2\n",
        2,
    );
}

/// `[segmentation]` is a pass-through loss head: the GPU forward must match
/// the CPU one exactly (the softmax is a host round trip today) and its seed
/// must reach the convolution before it unchanged.
#[test]
fn gpu_segmentation_head_matches_cpu_forward_and_backward() {
    for classes in [1usize, 3] {
        assert_gpu_layer_chain_matches_cpu(
            &format!(
                "[net]\nwidth=2\nheight=2\nchannels={classes}\nbatch=2\n\
                     [convolutional]\nfilters={classes}\nsize=1\nstride=1\nactivation=linear\n\
                     [segmentation]\nclasses={classes}\n"
            ),
            1,
        );
    }
}

/// End-to-end CPU: the layer's output, the loss seed and the convolution
/// gradient must agree with the module's own numbers.
#[test]
fn segmentation_layer_forward_and_backward_match_the_module() {
    let cfg = "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                   [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
                   [segmentation]\nclasses=2\nlambda_ce=0.5\ncls_normalizer=2\n";
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
    let mut exec = ExecutableNetwork::new(network);
    fill_conv_params(&mut exec.params);
    let input = patterned_input(exec.network.net.input_shape());
    let forward = exec.forward_training(&input).unwrap();

    let spec = match &exec.network.layers[1].kind {
        LayerKind::Segmentation(spec) => spec.clone(),
        other => panic!("expected segmentation, got {}", other.name()),
    };
    let shape = exec.network.layers[1].output_shape;
    let spatial = shape.width * shape.height;
    let expected = crate::segmentation::segmentation_forward(
        &forward.layer_outputs[0].data,
        &spec,
        shape.batch,
        spatial,
    );
    assert_slice_close(
        "segmentation output",
        &forward.layer_outputs[1].data,
        &expected.output,
        1e-6,
    );

    // A seed of all ones on the head must arrive at the previous layer
    // unchanged: the layer is a pass-through for the backward.
    let mut seeds = vec![None; exec.network.layers.len()];
    seeds[1] =
        Some(Tensor::from_data(shape, vec![1.0; shape.batch * shape.element_count()]).unwrap());
    exec.backward_accumulate_seeded(&input, &forward, &seeds)
        .unwrap();
    let grads = &exec.train_buffers[0];
    let LayerTrainBuffers::Convolutional(train) = grads else {
        panic!("expected conv buffers")
    };
    assert_eq!(train.bias_updates.len(), shape.channels);
    // A pure pass-through cannot drive a bias gradient: sum of weights = 1.
    // A convolution's bias gradient is the plain sum of its output
    // gradient, so an all-ones seed on the head must reach it untouched:
    // every bias gets exactly batch * H * W.
    let expected = (shape.batch * shape.width * shape.height) as f32;
    for (i, g) in train.bias_updates.iter().enumerate() {
        assert!(
            (g - expected).abs() < 1e-4,
            "bias {i} gradient {g}, expected {expected}"
        );
    }
}

/// `[omd_loss]` is a forward pass-through and a backward pass-through, so
/// the GPU chain must match the CPU one exactly for any seed.
#[test]
fn gpu_omd_loss_head_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=2\nheight=2\nchannels=4\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [omd_loss]\nclasses=2\nmax_boxes=4\nlambda=0.5\n",
        1,
    );
}

/// End-to-end CPU: the layer's output is the previous layer's output, and a
/// seed on the head reaches the convolution untouched.
#[test]
fn omd_loss_layer_forward_and_backward_are_pass_through() {
    let cfg = "[net]\nwidth=2\nheight=2\nchannels=3\nbatch=2\n\
                   [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
                   [omd_loss]\nclasses=2\nmax_boxes=4\n";
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
    let mut exec = ExecutableNetwork::new(network);
    fill_conv_params(&mut exec.params);
    let input = patterned_input(exec.network.net.input_shape());
    let forward = exec.forward_training(&input).unwrap();
    assert_eq!(
        forward.layer_outputs[1].data, forward.layer_outputs[0].data,
        "[omd_loss] must be a forward pass-through"
    );

    let shape = exec.network.layers[1].output_shape;
    let seed = Tensor::from_data(shape, vec![0.5f32; shape.batch * shape.element_count()]).unwrap();
    let mut seeds = vec![None; exec.network.layers.len()];
    seeds[1] = Some(seed);
    exec.backward_accumulate_seeded(&input, &forward, &seeds)
        .unwrap();
    let LayerTrainBuffers::Convolutional(train) = &exec.train_buffers[0] else {
        panic!()
    };
    // The 1x1 convolution's bias gradient is the plain sum of the seed.
    let expected = 0.5 * (shape.batch * shape.width * shape.height) as f32;
    for (i, g) in train.bias_updates.iter().enumerate() {
        assert!(
            (g - expected).abs() < 1e-4,
            "bias {i} gradient {g}, expected {expected}"
        );
    }
}

/// `[contrastive]` is a forward pass-through and a backward pass-through, so
/// the GPU chain must match the CPU one for any seed.
/// `[yolox]` transforms its input (the trailing `1 + classes` entries of
/// every cell go through a sigmoid) and its seed reaches the previous layer.
/// `[laya_scorer]` is LayerNorm + two dense layers; this FakeBackend test
/// exercises the backend trait's CPU defaults, while Oxide has a separate
/// actual-device forward test. Its input gradient remains host-computed.
/// (The generic `assert_gpu_layer_chain_matches_cpu`
/// helper is not used here: it also checks the preceding convolution's
/// weight gradient, which cancels out for a 1x1 convolution with this
/// narrow a seed.)
#[test]
fn gpu_laya_scorer_matches_cpu_forward_and_backward() {
    let cfg = "[net]\nwidth=4\nheight=1\nchannels=4\nbatch=2\n\
                   [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
                   [laya_scorer]\nhidden=4\nmax_options=4\n";
    let network =
        Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).expect("cfg parses"))
            .expect("network builds");
    let mut cpu = ExecutableNetwork::new(network.clone());
    fill_conv_params(&mut cpu.params);
    {
        // Give the scorer a trained-like gamma and dense fc1 so the seed
        // actually reaches the convolution.
        let LayerParams::LayaScorer(weights) = &mut cpu.params[1] else {
            panic!()
        };
        weights.ln_w.fill(1.0);
        for (i, v) in weights.fc1_w.iter_mut().enumerate() {
            *v = ((i * 7 % 9) as f32 - 4.0) / 6.0;
        }
        for (i, v) in weights.fc2_w.iter_mut().enumerate() {
            *v = (i as f32 % 3.0) * 0.5 - 0.25;
        }
        weights.fc2_b = 0.1;
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().expect("GPU train buffers");

    let input = patterned_input(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("GPU forward");
    assert_slice_close(
        "laya_scorer output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        1e-6,
    );

    let mut seeds = vec![None; cpu.network.layers.len()];
    let shape = cpu.network.layers[1].output_shape;
    seeds[1] = Some(
        Tensor::from_data(
            shape,
            (0..shape.batch * shape.element_count())
                .map(|i| (i as f32 * 0.91).cos())
                .collect(),
        )
        .expect("seed tensor"),
    );
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU backward");
    // The seed's effect on the preceding convolution shows up in its bias
    // gradient (the plain sum of its output gradient), which is non-zero and
    // does not cancel for this seed.
    let LayerTrainBuffers::Convolutional(cpu_train) = &cpu.train_buffers[0] else {
        panic!("expected convolutional buffers")
    };
    let gpu_buffers = gpu.device_train_buffers.borrow();
    let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers");

    assert!(
        cpu_train.bias_updates.iter().any(|g| g.abs() > 1e-6),
        "the seed must reach the convolution: {:?}",
        cpu_train.bias_updates
    );
    assert_slice_close(
        "convolution bias gradient through laya_scorer",
        &cpu_train.bias_updates,
        &gpu_train.bias_updates.data,
        1e-5,
    );
}

#[test]
fn gpu_yolox_head_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=2\nheight=2\nchannels=7\nbatch=2\n\
             [convolutional]\nfilters=7\nsize=1\nstride=1\nactivation=linear\n\
             [yolox]\nclasses=2\nmax=4\n",
        1,
    );
}

#[test]
fn gpu_contrastive_head_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=1\nheight=1\nchannels=4\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [contrastive]\nclasses=2\ncls_normalizer=0.5\n",
        1,
    );
}

#[test]
fn gpu_gelu_convolution_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=gelu\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        1,
    );
}

#[test]
fn weighted_shortcut_layer_backward_matches_finite_differences_on_its_weights() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=relu\nactivation=leaky\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut net = ExecutableNetwork::new(network);
    fill_conv_params(&mut net.params);
    let LayerParams::Shortcut(p) = &mut net.params[2] else {
        panic!()
    };
    p.weights = vec![0.9, 0.4, 0.7, 0.6]; // (n+1)=2 inputs x 2 channels
    let input = patterned_input(net.network.net.input_shape());
    let seed = Tensor::from_data(
        net.network.layers[2].output_shape,
        (0..16).map(|i| (i as f32 * 0.61).cos()).collect(),
    )
    .unwrap();
    let loss = |net: &ExecutableNetwork| -> f32 {
        let out = net.forward_training(&input).unwrap();
        out.layer_outputs[2]
            .data
            .iter()
            .zip(&seed.data)
            .map(|(o, s)| o * s)
            .sum()
    };
    let forward = net.forward_training(&input).unwrap();
    let mut seeds = vec![None; 3];
    seeds[2] = Some(seed.clone());
    let backward = net
        .backward_accumulate_seeded(&input, &forward, &seeds)
        .unwrap();
    let LayerParams::Shortcut(g) = &backward.param_gradients[2] else {
        panic!()
    };
    for k in 0..4 {
        let h = 1e-2f32;
        let (mut np, mut nm) = (net.clone(), net.clone());
        if let LayerParams::Shortcut(p) = &mut np.params[2] {
            p.weights[k] += h;
        }
        if let LayerParams::Shortcut(p) = &mut nm.params[2] {
            p.weights[k] -= h;
        }
        let fd = (loss(&np) - loss(&nm)) / (2.0 * h);
        assert!(
            (fd - g.weights[k]).abs() <= 5e-3 + 5e-3 * fd.abs(),
            "w[{k}]: fd {fd}, analytic {}",
            g.weights[k]
        );
    }
}

#[test]
fn gpu_weighted_shortcut_device_backward_matches_cpu() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=relu\nactivation=leaky\n",
            ).unwrap(),
        ).unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    fill_conv_params(&mut cpu.params);
    if let LayerParams::Shortcut(params) = &mut cpu.params[2] {
        params.weights = vec![0.8, 0.5, 0.6, 0.9];
    } else {
        panic!("weighted shortcut parameters expected");
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());

    let input = patterned_input(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu.forward_training_device(&input).expect("GPU forward");
    let device_output = gpu
        .backend
        .download_f32(&gpu_forward.gpu_layer_outputs[2])
        .unwrap();
    assert_slice_close(
        "weighted shortcut oxide/fake forward",
        &cpu_forward.layer_outputs[2].data,
        &device_output,
        1e-5,
    );

    let seed_shape = cpu.network.layers[2].output_shape;
    let seed = Tensor::from_data(
        seed_shape,
        (0..seed_shape.batch * seed_shape.element_count())
            .map(|i| (i as f32 * 0.53).cos())
            .collect(),
    )
    .unwrap();
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[2] = Some(seed);
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    let cpu_shortcut = match &cpu.train_buffers[2] {
        LayerTrainBuffers::Shortcut(buffers) => &buffers.weight_updates,
        other => panic!("unexpected CPU train buffer {other:?}"),
    };
    let host_buffers = gpu.cpu_fallback_train_buffers.borrow();
    let gpu_shortcut = match &host_buffers[2] {
        LayerTrainBuffers::Shortcut(buffers) => &buffers.weight_updates,
        other => panic!("unexpected host train buffer {other:?}"),
    };
    assert_slice_close(
        "weighted shortcut host weight updates",
        cpu_shortcut,
        gpu_shortcut,
        1e-6,
    );
}

#[test]
fn gpu_connected_forward_matches_cpu_with_and_without_batch_norm() {
    for batch_normalize in [0, 1] {
        let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!(
                        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=3\n\
                         [connected]\noutput=5\nbatch_normalize={batch_normalize}\nactivation=leaky\n"
                    ),
                )
                .expect("cfg should parse"),
            )
            .expect("network should build");
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::Connected(params) = &mut cpu.params[0] else {
            panic!("expected connected parameters");
        };
        for (i, w) in params.weights.iter_mut().enumerate() {
            *w = (i as f32 * 0.43).sin() * 0.5;
        }
        for (i, b) in params.biases.iter_mut().enumerate() {
            *b = 0.1 * i as f32 - 0.2;
        }
        if let Some(bn) = params.batch_norm.as_mut() {
            bn.scales = vec![1.5, 0.7, -0.4, 1.0, 2.0];
            bn.rolling_mean = vec![0.1, -0.2, 0.3, 0.0, 0.5];
            bn.rolling_variance = vec![0.5, 1.5, 2.0, 0.8, 1.2];
        }
        assert_eq!(params.batch_norm.is_some(), batch_normalize == 1);
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        gpu.params = cpu.params.clone();
        let input = patterned_input(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward(&input).expect("CPU forward");
        let gpu_forward = gpu.forward_with_state(&input).expect("GPU forward");
        assert_slice_close(
            "connected output",
            &cpu_forward.layer_outputs[0].data,
            &gpu_forward.cpu_forward.layer_outputs[0].data,
            1e-5,
        );
        assert_slice_close(
            "connected activation input",
            &cpu_forward.layer_activation_inputs[0]
                .as_ref()
                .unwrap()
                .data,
            &gpu_forward.cpu_forward.layer_activation_inputs[0]
                .as_ref()
                .unwrap()
                .data,
            1e-5,
        );
    }
}

// ---- P8 optimizer wiring: Adam/AdamW, loss scale, NaN/Inf, accumulation ----

pub(super) fn optimizer_net(net_extra: &str) -> GpuExecutableNetwork<FakeBackend> {
    let cfg = format!(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n{net_extra}\n\
             [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nbatch_normalize=1\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n"
    );
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg"))
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    fill_conv_params(&mut gpu.params);
    if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
        let bn = p.batch_norm.as_mut().expect("bn");
        bn.scales = vec![1.2, 0.8, -0.6];
    }
    gpu.init_train_buffers_gpu().expect("train buffers");
    gpu
}

pub(super) fn optimizer_seeds(
    gpu: &GpuExecutableNetwork<FakeBackend>,
    step: usize,
) -> Vec<Option<Tensor>> {
    let mut seeds = vec![None; gpu.network.layers.len()];
    let shape = gpu.network.layers[1].output_shape;
    seeds[1] = Some(
        Tensor::from_data(
            shape,
            (0..shape.batch * shape.element_count())
                .map(|i| ((i + 7 * step) as f32 * 0.91).cos() * 0.5)
                .collect(),
        )
        .expect("seed"),
    );
    seeds
}

/// (weights, biases, scales) of the two conv layers as host vectors plus
/// the accumulated updates, from the device state.
pub(super) fn optimizer_snapshot(
    gpu: &GpuExecutableNetwork<FakeBackend>,
) -> Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>)> {
    (0..2)
        .map(|i| {
            let train = gpu.device_train_buffers.borrow()[i].clone().expect("train");
            (
                train.weight_updates.data,
                train.bias_updates.data,
                train.scale_updates.map(|t| t.data),
            )
        })
        .collect()
}

pub(super) fn host_params(
    gpu: &GpuExecutableNetwork<FakeBackend>,
    layer: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let LayerParams::Convolutional(p) = &gpu.params[layer] else {
        panic!("conv params")
    };
    (
        p.weights.clone(),
        p.biases.clone(),
        p.batch_norm
            .as_ref()
            .map(|b| b.scales.clone())
            .unwrap_or_default(),
    )
}

/// f64 Adam / AdamW reference of one tensor (C++ `adam_update_gpu`).
#[allow(clippy::too_many_arguments)]
pub(super) fn ref_adam_tensor(
    w: &mut [f64],
    m: &mut [f64],
    v: &mut [f64],
    u: &[f32],
    lr: f64,
    (b1, b2, eps): (f64, f64, f64),
    decay_batch: f64,
    decoupled: f64,
    t: i32,
) {
    for i in 0..w.len() {
        let g = u[i] as f64 - decay_batch * w[i];
        m[i] = b1 * m[i] + (1.0 - b1) * g;
        v[i] = b2 * v[i] + (1.0 - b2) * g * g;
        let mhat = m[i] / (1.0 - b1.powi(t));
        let vhat = v[i] / (1.0 - b2.powi(t));
        w[i] += lr * mhat / (vhat.sqrt() + eps) - decoupled * w[i];
    }
}

#[test]
fn gpu_adam_and_adamw_match_f64_reference_over_steps() {
    for decoupled in [false, true] {
        let mut gpu = optimizer_net(&format!(
            "adam=1\nB1=0.8\nB2=0.95\neps=0.00001\ndecoupled_decay={}",
            decoupled as u8
        ));
        assert!(gpu.network.net.adam && gpu.network.net.decoupled_decay == decoupled);
        assert_eq!(
            (gpu.network.net.adam_b1, gpu.network.net.adam_b2),
            (0.8, 0.95)
        );
        // reference state: (w, b, s) x (m, v) per layer
        let ws: Vec<_> = (0..2).map(|l| host_params(&gpu, l)).collect();
        let mut refw: Vec<[Vec<f64>; 3]> = ws
            .iter()
            .map(|(w, b, s)| [w, b, s].map(|x| x.iter().map(|&f| f as f64).collect()))
            .collect();
        let mut refm: Vec<[Vec<f64>; 3]> = refw
            .iter()
            .map(|a| a.clone().map(|x| vec![0.0; x.len()]))
            .collect();
        let mut refv = refm.clone();
        let (lr, decay, batch) = (0.01f32, 0.05f32, 2usize);
        for step in 1..=4 {
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            let seeds = optimizer_seeds(&gpu, step);
            gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
                .expect("backward");
            let updates = optimizer_snapshot(&gpu);
            gpu.update_weights_gpu(lr, 0.9, decay, batch)
                .expect("adam update");
            assert_eq!(gpu.adam_step() as usize, step);
            let (coupled, dec) = if decoupled {
                (0.0, lr as f64 * decay as f64)
            } else {
                (decay as f64 * batch as f64, 0.0)
            };
            for layer in 0..2 {
                let (uw, ub, us) = &updates[layer];
                for (k, u) in [Some(uw), Some(ub), us.as_ref()].into_iter().enumerate() {
                    let Some(u) = u else { continue };
                    // C++ decays bias/scale only for coupled Adam.
                    let (c, d) = if k == 0 {
                        (coupled, dec)
                    } else if decoupled {
                        (0.0, 0.0)
                    } else {
                        (coupled, 0.0)
                    };
                    ref_adam_tensor(
                        &mut refw[layer][k],
                        &mut refm[layer][k],
                        &mut refv[layer][k],
                        u,
                        lr as f64,
                        (0.8, 0.95, 1e-5),
                        c,
                        d,
                        step as i32,
                    );
                }
                let (w, b, s) = host_params(&gpu, layer);
                for (k, got) in [w, b, s].iter().enumerate() {
                    for (i, g) in got.iter().enumerate() {
                        let want = refw[layer][k][i];
                        assert!(
                            (*g as f64 - want).abs() < 2e-5,
                            "decoupled={decoupled} step {step} layer {layer} tensor {k}[{i}]: {g} vs {want}"
                        );
                    }
                }
            }
            // Adam zeroes the accumulated updates (C++ adam_update_gpu).
            for (uw, ub, us) in optimizer_snapshot(&gpu) {
                assert!(
                    uw.iter()
                        .chain(&ub)
                        .chain(us.iter().flatten())
                        .all(|x| *x == 0.0)
                );
            }
        }
    }
}

#[test]
fn deferred_gpu_update_matches_eager_after_sync_and_refolds_lazily() {
    let mut eager = optimizer_net("");
    let mut lazy = optimizer_net("");
    let input = Tensor::from_data(
        eager.network.net.input_shape(),
        (0..eager.network.net.input_shape().element_count() * 2)
            .map(|i| (i as f32 * 0.37).sin())
            .collect(),
    )
    .expect("input");
    for step in 0..3 {
        for (gpu, deferred) in [(&mut eager, false), (&mut lazy, true)] {
            let forward = gpu.forward_training_device(&input).expect("forward");
            let seeds = optimizer_seeds(gpu, step);
            gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
                .expect("backward");
            if deferred {
                gpu.update_weights_gpu_deferred(0.05, 0.9, 0.001, 2)
                    .expect("update");
            } else {
                gpu.update_weights_gpu(0.05, 0.9, 0.001, 2).expect("update");
            }
        }
    }
    lazy.sync_host_weights_from_device().expect("sync");
    for layer in 0..2 {
        assert_eq!(
            host_params(&eager, layer),
            host_params(&lazy, layer),
            "layer {layer}"
        );
    }
    // Inference forward (BN-folded weights) agrees after the lazy refold.
    let want = eager.forward(&input).expect("eager inference");
    let got = lazy.forward(&input).expect("lazy inference");
    assert_eq!(want.layer_outputs.len(), got.layer_outputs.len());
    for (w, g) in want.layer_outputs.iter().zip(&got.layer_outputs) {
        assert_slice_close("inference", &g.data, &w.data, 1e-5);
    }
}

#[test]
fn gpu_adam_updates_all_gru_parameter_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[gru]\noutput=1\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let mut expected = Vec::new();
    {
        let LayerParams::Gru(params) = &mut gpu.params[0] else {
            panic!("GRU params")
        };
        let LayerTrainBuffers::Gru(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("GRU train buffers");
        };
        let blocks = [
            (&mut params.input_z, &mut train.input_z),
            (&mut params.state_z, &mut train.state_z),
            (&mut params.input_r, &mut train.input_r),
            (&mut params.state_r, &mut train.state_r),
            (&mut params.input_h, &mut train.input_h),
            (&mut params.state_h, &mut train.state_h),
        ];
        for (index, (params, train)) in blocks.into_iter().enumerate() {
            params.weights[0] = 0.1 * (index as f32 + 1.0);
            params.biases[0] = -0.05 * (index as f32 + 1.0);
            train.weight_updates[0] = 0.03 * (index as f32 + 1.0);
            train.bias_updates[0] = -0.02 * (index as f32 + 1.0);
            let zeros = [0.0f32];
            let weights = darknet_cuda::host_ops::adam_update(
                &params.weights,
                &train.weight_updates,
                &zeros,
                &zeros,
                &adam,
            )
            .0;
            let biases = darknet_cuda::host_ops::adam_update(
                &params.biases,
                &train.bias_updates,
                &zeros,
                &zeros,
                &adam,
            )
            .0;
            expected.push((weights, biases));
        }
    }
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("GRU Adam update");
    let LayerParams::Gru(params) = &gpu.params[0] else {
        unreachable!()
    };
    let blocks = [
        &params.input_z,
        &params.state_z,
        &params.input_r,
        &params.state_r,
        &params.input_h,
        &params.state_h,
    ];
    for (index, (weights, biases)) in expected.iter().enumerate() {
        assert_slice_close(
            "GRU block weights Adam",
            weights,
            &blocks[index].weights,
            1e-7,
        );
        assert_slice_close("GRU block biases Adam", biases, &blocks[index].biases, 1e-7);
    }
}

#[test]
fn gpu_adam_supports_connected_layers() {
    // Connected parameters stay on the host but now have Adam moments, so a
    // connected layer trains with `adam=1` on the GPU path.
    let network = Network::from_config(
        ConfigFile::parse_str(
            "inline.cfg",
            "[net]\ninputs=2\nbatch=1\nadam=1\n[connected]\noutput=1\nactivation=linear\n",
        )
        .expect("cfg"),
    )
    .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::Connected(params) = &mut gpu.params[0] else {
        panic!("expected connected parameters");
    };
    params.weights = vec![2.0, 3.0];
    params.biases = vec![0.5];
    let input = Tensor::from_data(gpu.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
    let forward = gpu.forward_with_state(&input).expect("forward");
    let seed = vec![Some(Tensor::filled(
        gpu.network.layers[0].output_shape,
        1.0,
    ))];
    gpu.backward_accumulate_seeded_gpu(&forward, &seed)
        .expect("backward");
    gpu.update_weights_gpu(0.1, 0.9, 0.0, 1)
        .expect("adam update");
    let LayerParams::Connected(params) = &gpu.params[0] else {
        panic!("expected connected parameters");
    };
    // Adam steps by ~lr regardless of gradient magnitude.
    assert!(
        params.weights[0] > 2.0 && params.weights[1] > 3.0,
        "weights should move: {:?}",
        params.weights
    );
}

#[test]
fn gpu_adam_updates_standalone_batchnorm_host_parameters() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[batchnorm]\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let (initial_scales, initial_biases) = {
        let LayerParams::BatchNorm(params) = &mut gpu.params[0] else {
            panic!("batchnorm params");
        };
        params.scales.copy_from_slice(&[1.2, 0.8]);
        params.biases.copy_from_slice(&[0.2, -0.1]);
        (params.scales.clone(), params.biases.clone())
    };
    let (scale_updates, bias_updates) = {
        let LayerTrainBuffers::BatchNorm(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0]
        else {
            panic!("batchnorm train buffers");
        };
        train.scale_updates.copy_from_slice(&[0.2, -0.3]);
        train.bias_updates.copy_from_slice(&[-0.1, 0.25]);
        (train.scale_updates.clone(), train.bias_updates.clone())
    };
    let step = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let zeros = [0.0f32; 2];
    let (expected_scales, scales_m, scales_v) =
        darknet_cuda::host_ops::adam_update(&initial_scales, &scale_updates, &zeros, &zeros, &step);
    let (expected_biases, biases_m, biases_v) =
        darknet_cuda::host_ops::adam_update(&initial_biases, &bias_updates, &zeros, &zeros, &step);

    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("standalone batchnorm Adam update");
    let LayerParams::BatchNorm(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "batchnorm scales Adam",
        &expected_scales,
        &params.scales,
        1e-7,
    );
    assert_slice_close(
        "batchnorm biases Adam",
        &expected_biases,
        &params.biases,
        1e-7,
    );
    {
        let fallback = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::BatchNorm(train) = &fallback[0] else {
            panic!("batchnorm train buffers");
        };
        assert!(
            train
                .scale_updates
                .iter()
                .chain(&train.bias_updates)
                .all(|u| *u == 0.0)
        );
    }

    let (scale_updates, bias_updates) = {
        let LayerTrainBuffers::BatchNorm(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0]
        else {
            panic!("batchnorm train buffers");
        };
        train.scale_updates.copy_from_slice(&[-0.15, 0.12]);
        train.bias_updates.copy_from_slice(&[0.3, -0.2]);
        (train.scale_updates.clone(), train.bias_updates.clone())
    };
    let step2 = super::AdamParams { step: 2, ..step };
    let expected_scales = darknet_cuda::host_ops::adam_update(
        &expected_scales,
        &scale_updates,
        &scales_m,
        &scales_v,
        &step2,
    )
    .0;
    let expected_biases = darknet_cuda::host_ops::adam_update(
        &expected_biases,
        &bias_updates,
        &biases_m,
        &biases_v,
        &step2,
    )
    .0;
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("second standalone batchnorm Adam update");
    let LayerParams::BatchNorm(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "batchnorm scales Adam step 2",
        &expected_scales,
        &params.scales,
        1e-7,
    );
    assert_slice_close(
        "batchnorm biases Adam step 2",
        &expected_biases,
        &params.biases,
        1e-7,
    );
}

#[test]
fn gpu_adam_updates_all_rnn_parameter_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\ndecoupled_decay=1\n[rnn]\nhidden=1\noutput=1\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.2,
        batch: 1,
        step: 1,
        decoupled_decay: true,
    };
    let no_decay = super::AdamParams { decay: 0.0, ..adam };
    let mut expected = Vec::new();
    {
        let LayerParams::Rnn(params) = &mut gpu.params[0] else {
            panic!("RNN params")
        };
        let LayerTrainBuffers::Rnn(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("RNN train buffers");
        };
        let blocks = [
            (&mut params.input, &mut train.input, 0.2, -0.1),
            (&mut params.self_params, &mut train.self_params, 0.4, 0.3),
            (&mut params.output, &mut train.output, -0.5, 0.6),
        ];
        for (block_index, (params, train, weight, bias)) in blocks.into_iter().enumerate() {
            params.weights[0] = weight;
            params.biases[0] = bias;
            train.weight_updates[0] = 0.1 * (block_index as f32 + 1.0);
            train.bias_updates[0] = -0.07 * (block_index as f32 + 1.0);
            let zeros = [0.0f32];
            let ew = darknet_cuda::host_ops::adam_update(
                &params.weights,
                &train.weight_updates,
                &zeros,
                &zeros,
                &adam,
            )
            .0;
            let eb = darknet_cuda::host_ops::adam_update(
                &params.biases,
                &train.bias_updates,
                &zeros,
                &zeros,
                &no_decay,
            )
            .0;
            expected.push((ew, eb));
        }
    }
    gpu.update_weights_gpu(0.01, 0.0, 0.2, 1)
        .expect("RNN Adam update");
    let LayerParams::Rnn(params) = &gpu.params[0] else {
        unreachable!()
    };
    for (block_index, (weights, biases)) in expected.iter().enumerate() {
        let block = match block_index {
            0 => &params.input,
            1 => &params.self_params,
            _ => &params.output,
        };
        assert_slice_close("RNN block weights Adam", weights, &block.weights, 1e-7);
        assert_slice_close("RNN block biases Adam", biases, &block.biases, 1e-7);
    }
}

#[test]
fn gpu_adam_updates_moe_router_and_expert_host_parameters() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[moe_convolutional]\nfilters=1\nsize=1\nexperts=2\ntop_k=0\nload_balance=0\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let mut expected_experts = Vec::new();
    let (expected_router_weights, expected_router_biases) = {
        let LayerParams::Moe(params) = &mut gpu.params[0] else {
            panic!("MoE params")
        };
        let LayerTrainBuffers::Moe(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("MoE train buffers");
        };
        params.router_weights.copy_from_slice(&[0.2, -0.4]);
        params.router_biases.copy_from_slice(&[0.1, -0.2]);
        train.router_weight_updates.copy_from_slice(&[0.03, -0.02]);
        train.router_bias_updates.copy_from_slice(&[-0.01, 0.04]);
        let zero_router = [0.0f32; 2];
        let zero_expert = [0.0f32; 1];
        let no_decay = super::AdamParams { decay: 0.0, ..adam };
        let router_weights = darknet_cuda::host_ops::adam_update(
            &params.router_weights,
            &train.router_weight_updates,
            &zero_router,
            &zero_router,
            &no_decay,
        )
        .0;
        let router_biases = darknet_cuda::host_ops::adam_update(
            &params.router_biases,
            &train.router_bias_updates,
            &zero_router,
            &zero_router,
            &no_decay,
        )
        .0;
        for (index, (expert, updates)) in params
            .experts
            .iter_mut()
            .zip(&mut train.experts)
            .enumerate()
        {
            expert.weights[0] = 0.15 * (index as f32 + 1.0);
            expert.biases[0] = -0.05 * (index as f32 + 1.0);
            updates.weight_updates[0] = 0.02 * (index as f32 + 1.0);
            updates.bias_updates[0] = -0.03 * (index as f32 + 1.0);
            let weights = darknet_cuda::host_ops::adam_update(
                &expert.weights,
                &updates.weight_updates,
                &zero_expert,
                &zero_expert,
                &adam,
            )
            .0;
            let biases = darknet_cuda::host_ops::adam_update(
                &expert.biases,
                &updates.bias_updates,
                &zero_expert,
                &zero_expert,
                &adam,
            )
            .0;
            expected_experts.push((weights, biases));
        }
        (router_weights, router_biases)
    };
    gpu.warm_device_conv_params().expect("warm experts");
    assert_eq!(gpu.device_conv_params.borrow()[0].len(), 2);
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("MoE Adam update");
    let LayerParams::Moe(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "MoE router weights Adam",
        &expected_router_weights,
        &params.router_weights,
        1e-7,
    );
    assert_slice_close(
        "MoE router biases Adam",
        &expected_router_biases,
        &params.router_biases,
        1e-7,
    );
    for (index, (expected_weights, expected_biases)) in expected_experts.iter().enumerate() {
        assert_slice_close(
            "MoE expert weights Adam",
            expected_weights,
            &params.experts[index].weights,
            1e-7,
        );
        assert_slice_close(
            "MoE expert biases Adam",
            expected_biases,
            &params.experts[index].biases,
            1e-7,
        );
    }
    assert!(
        gpu.device_conv_params.borrow()[0].is_empty(),
        "stale expert cache must be invalidated"
    );
    gpu.warm_device_conv_params()
        .expect("rewarm updated experts");
    let cache = gpu.device_conv_params.borrow();
    assert_eq!(cache[0].len(), 2);
    for (index, (expected_weights, _)) in expected_experts.iter().enumerate() {
        let cached = cache[0][index].as_ref().expect("expert cache");
        let weights = gpu
            .backend
            .download_f32(&cached.raw_weights)
            .expect("download expert weights");
        assert_slice_close(
            "reuploaded MoE expert weights",
            expected_weights,
            &weights,
            0.0,
        );
    }
}

#[test]
fn gpu_adam_updates_batchnorm_moe_on_device() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[moe_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\nexperts=2\ntop_k=0\nload_balance=0\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    {
        let LayerParams::Moe(params) = &mut gpu.params[0] else {
            panic!("MoE parameters");
        };
        params.router_weights.copy_from_slice(&[0.2, -0.4]);
        params.router_biases.copy_from_slice(&[0.1, -0.2]);
        for (index, expert) in params.experts.iter_mut().enumerate() {
            expert.weights[0] = 0.15 * (index as f32 + 1.0);
            expert.biases[0] = -0.05 * (index as f32 + 1.0);
            expert
                .batch_norm
                .as_mut()
                .expect("MoE BN parameters")
                .scales[0] = 0.7 + index as f32 * 0.2;
        }
    }
    gpu.zero_train_buffers_gpu().expect("device train buffers");
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let zero_router = [0.0f32; 2];
    let zero_expert = [0.0f32; 1];
    let expected = {
        let LayerParams::Moe(params) = &gpu.params[0] else {
            unreachable!();
        };
        let router_weights = darknet_cuda::host_ops::adam_update(
            &params.router_weights,
            &[0.03, -0.02],
            &zero_router,
            &zero_router,
            &adam,
        )
        .0;
        let router_biases = darknet_cuda::host_ops::adam_update(
            &params.router_biases,
            &[-0.01, 0.04],
            &zero_router,
            &zero_router,
            &adam,
        )
        .0;
        let experts = params
            .experts
            .iter()
            .enumerate()
            .map(|(index, expert)| {
                let weight_gradient = [0.02 * (index as f32 + 1.0)];
                let bias_gradient = [-0.03 * (index as f32 + 1.0)];
                let scale_gradient = [0.04 * (index as f32 + 1.0)];
                let weights = darknet_cuda::host_ops::adam_update(
                    &expert.weights,
                    &weight_gradient,
                    &zero_expert,
                    &zero_expert,
                    &adam,
                )
                .0;
                let biases = darknet_cuda::host_ops::adam_update(
                    &expert.biases,
                    &bias_gradient,
                    &zero_expert,
                    &zero_expert,
                    &adam,
                )
                .0;
                let scales = darknet_cuda::host_ops::adam_update(
                    &expert.batch_norm.as_ref().unwrap().scales,
                    &scale_gradient,
                    &zero_expert,
                    &zero_expert,
                    &adam,
                )
                .0;
                (weights, biases, scales)
            })
            .collect::<Vec<_>>();
        (router_weights, router_biases, experts)
    };
    {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("device MoE buffers");
        train
            .router_weight_updates
            .data
            .copy_from_slice(&[0.03, -0.02]);
        train
            .router_bias_updates
            .data
            .copy_from_slice(&[-0.01, 0.04]);
        for index in 0..2 {
            train.expert_weight_updates[index]
                .data
                .copy_from_slice(&[0.02 * (index as f32 + 1.0)]);
            train.expert_bias_updates[index]
                .data
                .copy_from_slice(&[-0.03 * (index as f32 + 1.0)]);
            train.expert_scale_updates[index]
                .as_mut()
                .expect("device MoE scale buffer")
                .data
                .copy_from_slice(&[0.04 * (index as f32 + 1.0)]);
        }
        train.dirty = true;
    }
    gpu.update_weights_gpu_deferred(0.01, 0.0, 0.0, 1)
        .expect("deferred BN MoE Adam");
    assert!(gpu.host_weights_stale);
    gpu.sync_host_weights_from_device()
        .expect("sync deferred BN MoE Adam");
    assert!(!gpu.host_weights_stale);
    let LayerParams::Moe(params) = &gpu.params[0] else {
        unreachable!();
    };
    assert_slice_close(
        "BN MoE Adam router weights",
        &expected.0,
        &params.router_weights,
        1e-7,
    );
    assert_slice_close(
        "BN MoE Adam router biases",
        &expected.1,
        &params.router_biases,
        1e-7,
    );
    for (expert, (weights, biases, scales)) in params.experts.iter().zip(&expected.2) {
        assert_slice_close("BN MoE Adam expert weights", weights, &expert.weights, 1e-7);
        assert_slice_close("BN MoE Adam expert biases", biases, &expert.biases, 1e-7);
        assert_slice_close(
            "BN MoE Adam expert scales",
            scales,
            &expert.batch_norm.as_ref().unwrap().scales,
            1e-7,
        );
    }
}

#[test]
fn gpu_sgd_updates_batchnorm_moe_on_device_with_momentum() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\n[moe_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\nexperts=2\ntop_k=0\nload_balance=0\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    {
        let LayerParams::Moe(params) = &mut gpu.params[0] else {
            panic!("MoE parameters");
        };
        params.router_weights.copy_from_slice(&[0.2, -0.4]);
        params.router_biases.copy_from_slice(&[0.1, -0.2]);
        for (index, expert) in params.experts.iter_mut().enumerate() {
            expert.weights[0] = 0.15 * (index as f32 + 1.0);
            expert.biases[0] = -0.05 * (index as f32 + 1.0);
            expert
                .batch_norm
                .as_mut()
                .expect("MoE BN parameters")
                .scales[0] = 0.7 + index as f32 * 0.2;
        }
    }
    gpu.zero_train_buffers_gpu().expect("device train buffers");
    let lr = 0.01;
    let momentum = 0.8;
    let router_weight_gradient = [0.03, -0.02];
    let router_bias_gradient = [-0.01, 0.04];
    let expert_weight_gradients = [[0.02], [-0.04]];
    let expert_bias_gradients = [[-0.03], [0.05]];
    let expert_scale_gradients = [[0.04], [-0.06]];
    let (expected_router_weights, expected_router_biases, expected_experts) = {
        let LayerParams::Moe(params) = &gpu.params[0] else {
            unreachable!();
        };
        let mut router_weights = params.router_weights.clone();
        let mut router_biases = params.router_biases.clone();
        let mut router_weight_updates = router_weight_gradient;
        let mut router_bias_updates = router_bias_gradient;
        super::sgd_update(
            &mut router_weights,
            &mut router_weight_updates,
            lr,
            momentum,
            0.0,
            1,
        );
        super::sgd_update(
            &mut router_biases,
            &mut router_bias_updates,
            lr,
            momentum,
            0.0,
            1,
        );
        let experts = params
            .experts
            .iter()
            .enumerate()
            .map(|(index, expert)| {
                let mut weights = expert.weights.clone();
                let mut biases = expert.biases.clone();
                let mut scales = expert.batch_norm.as_ref().unwrap().scales.clone();
                let mut weight_updates = expert_weight_gradients[index];
                let mut bias_updates = expert_bias_gradients[index];
                let mut scale_updates = expert_scale_gradients[index];
                super::sgd_update(&mut weights, &mut weight_updates, lr, momentum, 0.0, 1);
                super::sgd_update(&mut biases, &mut bias_updates, lr, momentum, 0.0, 1);
                super::sgd_update(&mut scales, &mut scale_updates, lr, momentum, 0.0, 1);
                (
                    weights,
                    biases,
                    scales,
                    weight_updates,
                    bias_updates,
                    scale_updates,
                )
            })
            .collect::<Vec<_>>();
        (router_weights, router_biases, experts)
    };
    {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("device MoE buffers");
        train
            .router_weight_updates
            .data
            .copy_from_slice(&router_weight_gradient);
        train
            .router_bias_updates
            .data
            .copy_from_slice(&router_bias_gradient);
        for index in 0..2 {
            train.expert_weight_updates[index]
                .data
                .copy_from_slice(&expert_weight_gradients[index]);
            train.expert_bias_updates[index]
                .data
                .copy_from_slice(&expert_bias_gradients[index]);
            train.expert_scale_updates[index]
                .as_mut()
                .expect("device MoE scale buffer")
                .data
                .copy_from_slice(&expert_scale_gradients[index]);
        }
        train.dirty = true;
    }
    gpu.update_weights_gpu_deferred(lr, momentum, 0.0, 1)
        .expect("deferred BN MoE SGD");
    assert!(gpu.host_weights_stale);
    gpu.sync_host_weights_from_device()
        .expect("sync deferred BN MoE SGD");
    let LayerParams::Moe(params) = &gpu.params[0] else {
        unreachable!();
    };
    assert_slice_close(
        "BN MoE SGD router weights",
        &expected_router_weights,
        &params.router_weights,
        1e-7,
    );
    assert_slice_close(
        "BN MoE SGD router biases",
        &expected_router_biases,
        &params.router_biases,
        1e-7,
    );
    for (index, (expert, expected)) in params.experts.iter().zip(&expected_experts).enumerate() {
        assert_slice_close(
            &format!("BN MoE SGD expert {index} weights"),
            &expected.0,
            &expert.weights,
            1e-7,
        );
        assert_slice_close(
            &format!("BN MoE SGD expert {index} biases"),
            &expected.1,
            &expert.biases,
            1e-7,
        );
        assert_slice_close(
            &format!("BN MoE SGD expert {index} scales"),
            &expected.2,
            &expert.batch_norm.as_ref().unwrap().scales,
            1e-7,
        );
        let train = gpu.device_moe_train_buffers.borrow();
        let momentum_buffers = train[0].as_ref().unwrap();
        assert_slice_close(
            &format!("BN MoE SGD expert {index} weight momentum"),
            &expected.3,
            &momentum_buffers.expert_weight_updates[index].data,
            1e-7,
        );
        assert_slice_close(
            &format!("BN MoE SGD expert {index} bias momentum"),
            &expected.4,
            &momentum_buffers.expert_bias_updates[index].data,
            1e-7,
        );
        assert_slice_close(
            &format!("BN MoE SGD expert {index} scale momentum"),
            &expected.5,
            &momentum_buffers.expert_scale_updates[index]
                .as_ref()
                .unwrap()
                .data,
            1e-7,
        );
    }
}

#[test]
fn gpu_adam_updates_deconvolutional_host_parameters() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[deconvolutional]\nfilters=1\nsize=2\nstride=2\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::Convolutional(params) = &mut gpu.params[0] else {
        panic!("deconvolutional parameters");
    };
    params.weights.copy_from_slice(&[0.2, -0.3, 0.4, -0.5]);
    params.biases[0] = 0.1;
    let initial_weights = params.weights.clone();
    let initial_biases = params.biases.clone();
    let LayerTrainBuffers::Convolutional(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0]
    else {
        panic!("deconvolutional train buffers");
    };
    train
        .weight_updates
        .copy_from_slice(&[0.03, -0.02, 0.01, -0.04]);
    train.bias_updates[0] = -0.05;
    let weight_updates = train.weight_updates.clone();
    let bias_updates = train.bias_updates.clone();
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let zeros_weights = [0.0f32; 4];
    let zeros_bias = [0.0f32; 1];
    let expected_weights = darknet_cuda::host_ops::adam_update(
        &initial_weights,
        &weight_updates,
        &zeros_weights,
        &zeros_weights,
        &adam,
    )
    .0;
    let expected_bias = darknet_cuda::host_ops::adam_update(
        &initial_biases,
        &bias_updates,
        &zeros_bias,
        &zeros_bias,
        &adam,
    )
    .0;
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("deconvolutional Adam update");
    let LayerParams::Convolutional(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "deconvolutional weights Adam",
        &expected_weights,
        &params.weights,
        1e-7,
    );
    assert_slice_close(
        "deconvolutional biases Adam",
        &expected_bias,
        &params.biases,
        1e-7,
    );
}

#[test]
fn gpu_adam_updates_vit_parameters_in_cpp_weight_order() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\ndecoupled_decay=1\n[vit]\ndim=2\nheads=1\npatch_size=1\npatch_stride=1\nmlp_dim=2\npos_embed=learned\nlearning_rate=0.5\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let field_lengths = {
        let LayerParams::Vit(params) = &mut gpu.params[0] else {
            panic!("ViT parameters");
        };
        let mut value = 0.05;
        for part in [
            &mut params.patch_embed,
            &mut params.patch_bias,
            &mut params.qkv_bias,
            &mut params.ln1_gamma,
            &mut params.ln1_beta,
            &mut params.qkv_weight,
            &mut params.wo,
            &mut params.wo_bias,
            &mut params.ln2_gamma,
            &mut params.ln2_beta,
            &mut params.ffn_w1,
            &mut params.ffn_b1,
            &mut params.ffn_w2,
            &mut params.ffn_b2,
            &mut params.pos_embed,
            &mut params.mhc,
        ] {
            for weight in part {
                *weight = value;
                value += 0.01;
            }
        }
        [
            params.patch_embed.len(),
            params.patch_bias.len(),
            params.qkv_bias.len(),
            params.ln1_gamma.len(),
            params.ln1_beta.len(),
            params.qkv_weight.len(),
            params.wo.len(),
            params.wo_bias.len(),
            params.ln2_gamma.len(),
            params.ln2_beta.len(),
            params.ffn_w1.len(),
            params.ffn_b1.len(),
            params.ffn_w2.len(),
            params.ffn_b2.len(),
            params.pos_embed.len(),
            params.mhc.len(),
        ]
    };
    {
        let LayerTrainBuffers::Vit(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("ViT train buffers");
        };
        let mut value = 1.0;
        for part in [
            &mut train.updates.patch_embed,
            &mut train.updates.patch_bias,
            &mut train.updates.qkv_bias,
            &mut train.updates.ln1_gamma,
            &mut train.updates.ln1_beta,
            &mut train.updates.qkv_weight,
            &mut train.updates.wo,
            &mut train.updates.wo_bias,
            &mut train.updates.ln2_gamma,
            &mut train.updates.ln2_beta,
            &mut train.updates.ffn_w1,
            &mut train.updates.ffn_b1,
            &mut train.updates.ffn_w2,
            &mut train.updates.ffn_b2,
            &mut train.updates.pos_embed,
            &mut train.updates.mhc,
        ] {
            for update in part {
                *update = value;
                value += 0.1;
            }
        }
    }
    let (initial_blob, gradient_blob) = {
        let LayerParams::Vit(params) = &gpu.params[0] else {
            unreachable!()
        };
        let buffers = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::Vit(train) = &buffers[0] else {
            unreachable!()
        };
        let mut updates = train.updates.clone();
        super::sanitize_vit_updates(&mut updates, true);
        let clip = super::vit_update_clip_scale(&updates, 5.0, true);
        assert!(clip < 1.0, "test gradients must exercise ViT clipping");
        super::scale_vit_updates(&mut updates, clip);
        (params.to_blob(), updates.to_blob())
    };
    let adam = super::AdamParams {
        lr: 0.005,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.1,
        batch: 1,
        step: 1,
        decoupled_decay: true,
    };
    let decay_mask = [
        true, false, false, false, false, true, true, false, false, false, true, false, true,
        false, true, false,
    ];
    let mut expected = initial_blob;
    let mut offset = 0;
    for (index, (&len, &weight_decay)) in field_lengths.iter().zip(&decay_mask).enumerate() {
        let update_adam = super::AdamParams {
            lr: if index == 15 { adam.lr * 0.1 } else { adam.lr },
            decay: if weight_decay { adam.decay } else { 0.0 },
            ..adam
        };
        let zeros = vec![0.0; len];
        let updated = darknet_cuda::host_ops::adam_update(
            &expected[offset..offset + len],
            &gradient_blob[offset..offset + len],
            &zeros,
            &zeros,
            &update_adam,
        )
        .0;
        expected[offset..offset + len].copy_from_slice(&updated);
        offset += len;
    }
    gpu.update_weights_gpu(0.01, 0.0, 0.1, 1)
        .expect("ViT Adam update");
    let LayerParams::Vit(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close("ViT Adam C++ order", &expected, &params.to_blob(), 1e-7);
}

#[test]
fn gpu_adam_updates_weighted_shortcut_parameters() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\ninputs=2\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[connected]\noutput=2\nactivation=linear\n[shortcut]\nfrom=-1\nweights_type=per_feature\nweights_normalization=none\nlearning_rate=0.5\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::Shortcut(params) = &mut gpu.params[1] else {
        panic!("weighted shortcut parameters");
    };
    params.weights.copy_from_slice(&[0.4, -0.2]);
    let initial = params.weights.clone();
    let LayerTrainBuffers::Shortcut(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[1]
    else {
        panic!("weighted shortcut train buffers");
    };
    train.weight_updates.copy_from_slice(&[0.3, -0.1]);
    let updates = train.weight_updates.clone();
    let step = super::AdamParams {
        lr: 0.005,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let zeros = [0.0f32; 2];
    let expected = darknet_cuda::host_ops::adam_update(&initial, &updates, &zeros, &zeros, &step).0;
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("weighted shortcut Adam update");
    let LayerParams::Shortcut(params) = &gpu.params[1] else {
        unreachable!()
    };
    assert_slice_close("weighted shortcut Adam", &expected, &params.weights, 1e-7);
}

#[test]
fn gpu_adam_updates_local_host_parameters() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[local]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let (initial_weights, initial_biases) = {
        let LayerParams::Local(params) = &mut gpu.params[0] else {
            panic!("local params")
        };
        params.weights.copy_from_slice(&[0.4, -0.2, 0.7, 0.1]);
        params.biases.copy_from_slice(&[0.3, -0.5, 0.2, 0.8]);
        (params.weights.clone(), params.biases.clone())
    };
    let (weight_updates, bias_updates) = {
        let LayerTrainBuffers::Local(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0]
        else {
            panic!("local train buffers");
        };
        train
            .weight_updates
            .copy_from_slice(&[0.2, -0.1, 0.3, -0.4]);
        train.bias_updates.copy_from_slice(&[-0.3, 0.1, 0.2, -0.25]);
        (train.weight_updates.clone(), train.bias_updates.clone())
    };
    let adam = super::AdamParams {
        lr: 0.01,
        beta1: 0.8,
        beta2: 0.95,
        eps: 0.001,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: false,
    };
    let zeros = [0.0f32; 4];
    let expected_weights = darknet_cuda::host_ops::adam_update(
        &initial_weights,
        &weight_updates,
        &zeros,
        &zeros,
        &adam,
    )
    .0;
    let expected_biases =
        darknet_cuda::host_ops::adam_update(&initial_biases, &bias_updates, &zeros, &zeros, &adam)
            .0;
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("local Adam update");
    let LayerParams::Local(params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "local weights Adam",
        &expected_weights,
        &params.weights,
        1e-7,
    );
    assert_slice_close("local biases Adam", &expected_biases, &params.biases, 1e-7);
}

#[test]
fn gpu_sgd_is_default_and_loss_scale_is_neutral() {
    // loss_scale scales the seeds and divides the lr back out: SGD and Adam
    // results are independent of it (up to rounding).
    for adam in [0, 1] {
        let run = |loss_scale: f32| {
            let mut gpu = optimizer_net(&format!("adam={adam}\nloss_scale={loss_scale}"));
            assert_eq!(gpu.network.net.loss_scale, loss_scale);
            for step in 1..=3 {
                let input = patterned_input(gpu.network.net.input_shape());
                let forward = gpu.forward_training_with_state(&input).expect("forward");
                let seeds = optimizer_seeds(&gpu, step);
                gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
                    .expect("backward");
                gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2)
                    .expect("update");
            }
            (host_params(&gpu, 0), host_params(&gpu, 1))
        };
        let (a, b) = (run(1.0), run(256.0));
        for (x, y) in [(a.0, b.0), (a.1, b.1)] {
            assert_slice_close("weights", &x.0, &y.0, 2e-5);
            assert_slice_close("biases", &x.1, &y.1, 2e-5);
            assert_slice_close("scales", &x.2, &y.2, 2e-5);
        }
    }
    let gpu = optimizer_net("");
    assert!(
        !gpu.network.net.adam
            && gpu.network.net.loss_scale == 1.0
            && !gpu.network.net.decoupled_decay
    );
}

#[test]
fn gpu_update_resets_nan_inf_updates_and_fixes_nan_inf_weights() {
    let mut gpu = optimizer_net("");
    {
        let mut cached = gpu.device_conv_params.borrow_mut();
        let c = cached[1][0].as_mut().expect("cached params");
        c.raw_weights.data[1] = f32::INFINITY;
        c.raw_weights.data[2] = f32::NAN;
        c.raw_weights.data[0] = 0.25;
        let mut train = gpu.device_train_buffers.borrow_mut();
        let t = train[1].as_mut().expect("train");
        t.weight_updates.data[0] = f32::NAN;
        t.weight_updates.data[3] = f32::NEG_INFINITY;
        t.bias_updates.data[0] = f32::NAN;
    }
    // lr = 0: weights only change through the sanitizer.
    gpu.update_weights_gpu(0.0, 0.9, 0.0, 1).expect("update");
    let (w, b, _) = host_params(&gpu, 1);
    assert_eq!(w[0], 0.25, "finite weight untouched");
    assert_eq!(w[1], 1.0 / 2.0, "fix_nan_and_inf: 1/(index+1)");
    assert_eq!(w[2], 1.0 / 3.0);
    assert!(w.iter().chain(&b).all(|x| x.is_finite()));
    let train = gpu.device_train_buffers.borrow()[1].clone().expect("train");
    assert!(
        train
            .weight_updates
            .data
            .iter()
            .chain(&train.bias_updates.data)
            .all(|x| x.is_finite())
    );
}

#[test]
fn loss_scale_helpers_and_non_finite_detection() {
    let mut gpu = optimizer_net("adam=1\nloss_scale=8\n");
    assert_eq!(gpu.loss_scale(), 8.0);
    gpu.set_loss_scale(4.0);
    assert_eq!(gpu.loss_scale(), 4.0);
    let input = patterned_input(gpu.network.net.input_shape());
    let forward = gpu.forward_training_with_state(&input).expect("forward");
    let seeds = optimizer_seeds(&gpu, 1);
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("backward");
    assert!(!gpu.gradients_non_finite().expect("finite check"));
    {
        let mut buffers = gpu.device_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("train buffer");
        let mut data = gpu
            .backend
            .download_f32(&train.weight_updates)
            .expect("download");
        data[0] = f32::INFINITY;
        train.weight_updates = gpu
            .backend
            .upload_f32(&[data.len()], &data)
            .expect("upload");
    }
    assert!(gpu.gradients_non_finite().expect("finite check"));
    gpu.set_loss_scale(0.0);
    assert_eq!(gpu.loss_scale(), 4.0);
}

fn optimizer_net_with_gradients() -> GpuExecutableNetwork<FakeBackend> {
    let mut gpu = optimizer_net("");
    let input = patterned_input(gpu.network.net.input_shape());
    let forward = gpu.forward_training_with_state(&input).expect("forward");
    let seeds = optimizer_seeds(&gpu, 1);
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("backward");
    gpu
}

#[test]
fn gradients_non_finite_sees_bn_scale_gradient() {
    let gpu = optimizer_net_with_gradients();
    let mut buffers = gpu.device_train_buffers.borrow_mut();
    let train = buffers[0].as_mut().expect("BN convolution train buffer");
    train
        .scale_updates
        .as_mut()
        .expect("BN scale updates")
        .data[0] = f32::INFINITY;
    drop(buffers);
    assert!(gpu.gradients_non_finite().expect("finite check"));
}

#[test]
fn gradients_non_finite_sees_conv_weight_gradient() {
    let gpu = optimizer_net_with_gradients();
    gpu.device_train_buffers.borrow_mut()[0]
        .as_mut()
        .expect("convolution train buffer")
        .weight_updates
        .data[0] = f32::INFINITY;
    assert!(gpu.gradients_non_finite().expect("finite check"));
}

#[test]
fn gradients_non_finite_sees_bn_bias_gradient() {
    let gpu = optimizer_net_with_gradients();
    gpu.device_train_buffers.borrow_mut()[0]
        .as_mut()
        .expect("BN convolution train buffer")
        .bias_updates
        .data[0] = f32::INFINITY;
    assert!(gpu.gradients_non_finite().expect("finite check"));
}

#[test]
fn cpu_muon_hybrid_uses_adamw_fallback_and_round_trips_dnopt6() {
    fn build() -> ExecutableNetwork {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "muon.cfg",
                r#"[net]
width=1
height=1
channels=1
batch=1
muon=1
loss_scale=2
decay=0.1
[connected]
output=1
batch_normalize=1
activation=linear
"#,
            )
            .expect("Muon cfg"),
        )
        .expect("Muon network");
        ExecutableNetwork::new(network)
    }
    fn set_updates(executable: &mut ExecutableNetwork, values: [f32; 3]) {
        let LayerTrainBuffers::Connected(train) = &mut executable.train_buffers[0] else {
            panic!("connected train buffers");
        };
        train.weight_updates[0] = values[0];
        train.bias_updates[0] = values[1];
        let batch_norm = train.batch_norm.as_mut().expect("connected BN train");
        batch_norm.scale_updates[0] = values[2];
    }

    let mut original = build();
    let LayerParams::Connected(params) = &mut original.params[0] else {
        panic!("connected parameters");
    };
    params.weights[0] = 2.0;
    params.biases[0] = 3.0;
    let batch_norm = params.batch_norm.as_mut().expect("connected BN params");
    batch_norm.scales[0] = 1.0;
    batch_norm.biases[0] = -1.0;
    set_updates(&mut original, [4.0, 2.0, 3.0]);
    original.update_weights_muon(0.1, 2).expect("Muon update");

    let LayerParams::Connected(params) = &original.params[0] else {
        unreachable!()
    };
    assert!((params.weights[0] - 1.993_928_7).abs() < 2e-6);
    assert!((params.biases[0] - 3.1).abs() < 2e-6);
    let batch_norm = params.batch_norm.as_ref().unwrap();
    assert!((batch_norm.scales[0] - 1.1).abs() < 2e-6);
    assert_eq!(batch_norm.biases[0], -1.0);
    assert_eq!(original.adam_step(), 1);
    assert_eq!(original.muon_state[0].as_ref().unwrap().matrices.len(), 1);
    assert_eq!(
        original.muon_state[0].as_ref().unwrap().matrices[0].momentum,
        [1.0]
    );

    let dir = std::env::temp_dir().join(format!("darknet-muon-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("muon.state");
    original.save_optimizer_state(&path).expect("save DNOPT6");
    assert_eq!(&std::fs::read(&path).unwrap()[..8], b"DNOPT6\0\0");

    let mut resumed = build();
    resumed.params = original.params.clone();
    resumed.load_optimizer_state(&path).expect("load DNOPT6");
    assert_eq!(resumed.adam_step(), original.adam_step());
    assert_eq!(resumed.muon_state, original.muon_state);
    assert_eq!(resumed.host_adam_state, original.host_adam_state);

    original.zero_train_buffers();
    resumed.zero_train_buffers();
    set_updates(&mut original, [-2.0, -1.0, -3.0]);
    set_updates(&mut resumed, [-2.0, -1.0, -3.0]);
    original
        .update_weights_muon(0.1, 2)
        .expect("original next step");
    resumed
        .update_weights_muon(0.1, 2)
        .expect("resumed next step");
    assert_eq!(resumed.params, original.params);
    assert_eq!(resumed.muon_state, original.muon_state);
    assert_eq!(resumed.host_adam_state, original.host_adam_state);

    let legacy_cfg = r#"[net]
width=1
height=1
channels=1
batch=1
adam=1
decoupled_decay=1
loss_scale=2
decay=0.1
[connected]
output=1
batch_normalize=1
activation=linear
"#;
    let legacy_network = Network::from_config(
        ConfigFile::parse_str("legacy-adam.cfg", legacy_cfg).expect("legacy cfg"),
    )
    .expect("legacy network");
    let mut legacy = ExecutableNetwork::new(legacy_network);
    set_updates(&mut legacy, [1.0, 2.0, 3.0]);
    legacy
        .update_weights_adam(0.1, 2)
        .expect("legacy Adam update");
    let legacy_path = dir.join("legacy-v3.state");
    legacy
        .save_optimizer_state(&legacy_path)
        .expect("save DNOPT3");
    assert_eq!(&std::fs::read(&legacy_path).unwrap()[..8], b"DNOPT3\0\0");
    let mut migrated = build();
    migrated
        .load_optimizer_state(&legacy_path)
        .expect("Muon reads DNOPT3");
    assert_eq!(migrated.adam_step(), 0);
    assert!(migrated.host_adam_state.iter().all(Option::is_none));
    assert!(migrated.muon_state.iter().all(Option::is_none));

    let mut malformed = Vec::new();
    malformed.extend_from_slice(b"DNOPT6\0\0");
    malformed.extend_from_slice(&1u32.to_le_bytes());
    for count in [0u64, 0, 0, 0, 1] {
        malformed.extend_from_slice(&count.to_le_bytes());
    }
    malformed.extend_from_slice(&0u64.to_le_bytes());
    write_muon_layer_state(
        &mut malformed,
        &MuonLayerState {
            matrices: vec![MuonMatrixState {
                rows: 2,
                cols: 1,
                momentum: vec![0.0; 2],
            }],
        },
    )
    .expect("encode malformed-shape state");
    let malformed_path = dir.join("malformed-v6.state");
    std::fs::write(&malformed_path, malformed).unwrap();
    let mut invalid = build();
    let error = invalid
        .load_optimizer_state(&malformed_path)
        .expect_err("wrong Muon matrix shape must reject");
    assert!(error.to_string().contains("matrix shapes"));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn gpu_muon_connected_update_matches_cpu_and_round_trips_sidecar() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\nmuon=1\n[connected]\noutput=2\n",
        )
        .expect("Muon cfg"),
    )
    .expect("Muon network");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let gradients = (vec![0.1, -0.2, 0.3, -0.4], vec![0.05, -0.07]);
    let (LayerTrainBuffers::Connected(cpu_train), LayerTrainBuffers::Connected(gpu_train)) = (
        &mut cpu.train_buffers[0],
        &mut gpu.cpu_fallback_train_buffers.get_mut()[0],
    ) else {
        panic!("connected buffers expected");
    };
    cpu_train.weight_updates.clone_from(&gradients.0);
    cpu_train.bias_updates.clone_from(&gradients.1);
    gpu_train.weight_updates.clone_from(&gradients.0);
    gpu_train.bias_updates.clone_from(&gradients.1);

    cpu.update_weights_muon(0.03, 2).expect("CPU Muon");
    gpu.update_weights_gpu(0.03, 0.9, network.net.decay, 2)
        .expect("GPU Muon");
    let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("connected parameters expected");
    };
    assert_slice_close(
        "GPU Muon weights",
        &gpu_params.weights,
        &cpu_params.weights,
        3e-5,
    );
    assert_slice_close(
        "GPU Muon biases",
        &gpu_params.biases,
        &cpu_params.biases,
        3e-5,
    );
    assert_eq!(gpu.muon_state, cpu.muon_state);
    assert_eq!(gpu.adam_step, cpu.adam_step);

    let dir = std::env::temp_dir().join(format!("darknet-gpu-muon-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let gpu_path = dir.join("gpu.state");
    let cpu_path = dir.join("cpu.state");
    gpu.save_optimizer_state(&gpu_path)
        .expect("save GPU Muon state");
    cpu.save_optimizer_state(&cpu_path)
        .expect("save CPU Muon state");
    assert_eq!(
        std::fs::read(&gpu_path).unwrap(),
        std::fs::read(&cpu_path).unwrap(),
        "host-only GPU Muon sidecar retains the shared DNOPT6 format",
    );
    let mut restored = GpuExecutableNetwork::new(network, FakeBackend);
    restored
        .load_optimizer_state(&gpu_path)
        .expect("load GPU Muon state");
    assert_eq!(restored.muon_state, gpu.muon_state);
    assert_eq!(restored.host_adam_state, gpu.host_adam_state);
    assert_eq!(restored.adam_step, gpu.adam_step);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn gpu_muon_standalone_batchnorm_matches_cpu_adamw_fallback() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon-bn.cfg",
            concat!(
                "[net]\n",
                "width=2\nheight=2\nchannels=2\nbatch=1\n",
                "muon=1\ndecoupled_decay=1\ndecay=0.0005\n",
                "[batchnorm]\n",
            ),
        )
        .expect("BatchNorm Muon cfg"),
    )
    .expect("BatchNorm Muon network");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let (LayerParams::BatchNorm(cpu_params), LayerParams::BatchNorm(gpu_params)) =
        (&mut cpu.params[0], &mut gpu.params[0])
    else {
        panic!("standalone BatchNorm parameters expected");
    };
    cpu_params.scales.copy_from_slice(&[1.2, 0.8]);
    cpu_params.biases.copy_from_slice(&[0.2, -0.1]);
    gpu_params.clone_from(cpu_params);
    let (LayerTrainBuffers::BatchNorm(cpu_train), LayerTrainBuffers::BatchNorm(gpu_train)) = (
        &mut cpu.train_buffers[0],
        &mut gpu.cpu_fallback_train_buffers.get_mut()[0],
    ) else {
        panic!("standalone BatchNorm train buffers expected");
    };
    cpu_train.scale_updates.copy_from_slice(&[0.2, -0.3]);
    cpu_train.bias_updates.copy_from_slice(&[-0.1, 0.25]);
    gpu_train.clone_from(cpu_train);

    cpu.update_weights_muon(0.01, 1).expect("CPU Muon fallback");
    gpu.update_weights_gpu(0.01, 0.9, network.net.decay, 1)
        .expect("GPU Muon fallback");
    let (LayerParams::BatchNorm(cpu_params), LayerParams::BatchNorm(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("standalone BatchNorm parameters expected after update");
    };
    assert_slice_close(
        "GPU Muon BN scales",
        &cpu_params.scales,
        &gpu_params.scales,
        1e-7,
    );
    assert_slice_close(
        "GPU Muon BN biases",
        &cpu_params.biases,
        &gpu_params.biases,
        1e-7,
    );
    assert_eq!(*gpu.host_adam_state.borrow(), cpu.host_adam_state);
    assert_eq!(gpu.muon_state, cpu.muon_state);
}

#[test]
fn gpu_muon_respects_layer_update_gate_without_advancing_layer_state() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon-gate.cfg",
            concat!(
                "[net]\n",
                "width=2\nheight=1\nchannels=1\nbatch=1\nmuon=1\ndecoupled_decay=1\n",
                "[connected]\noutput=2\ndont_update=1\n",
            ),
        )
        .expect("Muon gate cfg"),
    )
    .expect("Muon gate network");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    if let (LayerParams::Connected(cpu_params), LayerParams::Connected(gpu_params)) =
        (&mut cpu.params[0], &mut gpu.params[0])
    {
        cpu_params.weights.copy_from_slice(&[0.2, -0.1, 0.3, 0.4]);
        cpu_params.biases.copy_from_slice(&[0.03, -0.02]);
        gpu_params.clone_from(cpu_params);
    } else {
        panic!("connected parameters expected");
    }
    if let (LayerTrainBuffers::Connected(cpu_train), LayerTrainBuffers::Connected(gpu_train)) = (
        &mut cpu.train_buffers[0],
        &mut gpu.cpu_fallback_train_buffers.get_mut()[0],
    ) {
        cpu_train.weight_updates.fill(0.5);
        cpu_train.bias_updates.fill(-0.25);
        gpu_train.clone_from(cpu_train);
    } else {
        panic!("connected train buffers expected");
    }
    let before = cpu.params.clone();
    cpu.update_weights_muon(0.01, 1).expect("CPU gated Muon");
    gpu.update_weights_gpu(0.01, 0.9, network.net.decay, 1)
        .expect("GPU gated Muon");
    assert_eq!(cpu.params, before);
    assert_eq!(gpu.params, before);
    assert_eq!(cpu.adam_step(), 1);
    assert_eq!(gpu.adam_step(), 1);
    assert!(cpu.muon_state.iter().all(Option::is_none));
    assert!(gpu.muon_state.iter().all(Option::is_none));
    assert!(cpu.host_adam_state.iter().all(Option::is_none));
    assert!(gpu.host_adam_state.borrow().iter().all(Option::is_none));
}

#[test]
fn gpu_muon_rejects_batch_normalized_convolution_eml_moe_dcnv4_and_deform_before_updating() {
    let mut gpu = optimizer_net("muon=1\ndecoupled_decay=1");
    let before = gpu.params.clone();
    let error = gpu
        .update_weights_gpu(0.01, 0.9, 0.0005, 2)
        .expect_err("batch-normalized convolution Muon staging is not integrated yet");
    assert!(
        error
            .to_string()
            .contains("other parameter families remain explicitly unsupported")
    );
    assert_eq!(gpu.params, before);
    assert_eq!(gpu.adam_step, 0);

    let network = Network::from_config(
            ConfigFile::parse_str(
                "muon-eml-bn.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\nmuon=1\ndecoupled_decay=1\n[eml_convolutional]\nfilters=2\nsize=1\nbatch_normalize=1\n",
            )
            .expect("EML Muon cfg should parse"),
        )
        .expect("EML Muon network should build");
    let mut eml = GpuExecutableNetwork::new(network, FakeBackend);
    eml.init_train_buffers_gpu()
        .expect("EML Muon buffers should initialize");
    let before = eml.params.clone();
    let error = eml
        .update_weights_gpu(0.01, 0.9, 0.0005, 1)
        .expect_err("batch-normalized EML Muon staging is not integrated yet");
    assert!(
        error
            .to_string()
            .contains("other parameter families remain explicitly unsupported")
    );
    assert_eq!(eml.params, before);
    assert_eq!(eml.adam_step, 0);

    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon-moe-bn.cfg",
            r#"[net]
width=2
height=2
channels=2
batch=1
muon=1
decoupled_decay=1
[moe_convolutional]
filters=2
size=1
experts=2
top_k=0
batch_normalize=1
"#,
        )
        .expect("MoE Muon cfg should parse"),
    )
    .expect("MoE Muon network should build");
    let mut moe = GpuExecutableNetwork::new(network, FakeBackend);
    moe.init_train_buffers_gpu()
        .expect("MoE Muon buffers should initialize");
    let before = moe.params.clone();
    let error = moe
        .update_weights_gpu(0.01, 0.9, 0.0005, 1)
        .expect_err("batch-normalized MoE Muon staging is not integrated yet");
    assert!(
        error
            .to_string()
            .contains("other parameter families remain explicitly unsupported")
    );
    assert_eq!(moe.params, before);
    assert_eq!(moe.adam_step, 0);

    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon-dcnv4-bn.cfg",
            r#"[net]
width=2
height=2
channels=2
batch=1
muon=1
decoupled_decay=1
[dcnv4]
filters=2
size=1
groups=2
d_stride=1
batch_normalize=1
"#,
        )
        .expect("DCNv4 Muon cfg should parse"),
    )
    .expect("DCNv4 Muon network should build");
    let mut dcnv4 = GpuExecutableNetwork::new(network, FakeBackend);
    dcnv4
        .init_train_buffers_gpu()
        .expect("DCNv4 Muon buffers should initialize");
    let before = dcnv4.params.clone();
    let error = dcnv4
        .update_weights_gpu(0.01, 0.9, 0.0005, 1)
        .expect_err("batch-normalized DCNv4 Muon staging is not integrated yet");
    assert!(
        error
            .to_string()
            .contains("other parameter families remain explicitly unsupported")
    );
    assert_eq!(dcnv4.params, before);
    assert_eq!(dcnv4.adam_step, 0);

    let network = Network::from_config(
        ConfigFile::parse_str(
            "muon-deform-bn.cfg",
            r#"[net]
width=2
height=2
channels=2
batch=1
muon=1
decoupled_decay=1
[deform_conv]
filters=2
size=1
groups=2
mask=1
batch_normalize=1
"#,
        )
        .expect("deform_conv Muon cfg should parse"),
    )
    .expect("deform_conv Muon network should build");
    let mut deform = GpuExecutableNetwork::new(network, FakeBackend);
    deform
        .init_train_buffers_gpu()
        .expect("deform_conv Muon buffers should initialize");
    let before = deform.params.clone();
    let error = deform
        .update_weights_gpu(0.01, 0.9, 0.0005, 1)
        .expect_err("batch-normalized deform_conv Muon staging is not integrated yet");
    assert!(
        error
            .to_string()
            .contains("other parameter families remain explicitly unsupported")
    );
    assert_eq!(deform.params, before);
    assert_eq!(deform.adam_step, 0);
}
