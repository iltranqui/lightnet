use super::*;

#[test]
fn dcnv4_host_adam_and_optimizer_state_round_trip() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "dcnv4-adam.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=1\nadam=1\ndecay=0.05\n\
                 [dcnv4]\nsize=1\npadding=0\nsoftmax=0\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(params) = &mut cpu.params[0] else {
        panic!("DCNv4 params")
    };
    params.offset_weights.fill(0.25);
    params.offset_biases.fill(0.5);
    params.biases.fill(0.75);
    let LayerTrainBuffers::Dcnv4(train) = &mut cpu.train_buffers[0] else {
        panic!("DCNv4 train buffers")
    };
    train.offset_weight_updates.fill(-0.5);
    train.offset_bias_updates.fill(-0.25);
    train.bias_updates.fill(-0.75);

    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    *gpu.cpu_fallback_train_buffers.borrow_mut() = cpu.train_buffers.clone();
    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.05, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    assert_eq!(
        gpu.cpu_fallback_train_buffers.borrow().clone(),
        cpu.train_buffers
    );

    let path =
        std::env::temp_dir().join(format!("darknet-dcnv4-adam-{}.state", std::process::id()));
    cpu.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.host_adam_state, cpu.host_adam_state);
    assert_eq!(restored.adam_step, cpu.adam_step);
    let _ = fs::remove_file(path);
}

#[test]
fn kimi_kda_runtime_and_fake_backend_host_fallback_match() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "kimi-runtime.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=2\nbatch=1\n\
                 [kimi_kda]\nrank=1\nexperts=2\ntop_k=0\nhidden=2\nload_balance=0.03\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 137);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(gpu.requires_cpu_backward_fallback());

    let input =
        Tensor::from_data(TensorShape::new(2, 1, 2, 1), vec![0.21, -0.31, 0.47, 0.16]).unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    let cache = cpu_forward.kimi_kda_caches[0].as_ref().unwrap();
    assert!(cache.load_balance_cost.is_finite());
    assert!(cache.load_balance_cost >= 0.0);
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    assert_eq!(
        gpu_forward.cpu_forward.kimi_kda_caches,
        cpu_forward.kimi_kda_caches
    );

    let seed = Tensor::filled(cpu_forward.final_output().unwrap().shape, 0.25);
    let seeds = [Some(seed)];
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let LayerParams::KimiKda(gradient) = &backward.param_gradients[0] else {
        panic!("Kimi KDA gradients")
    };
    assert!(gradient.weights.iter().any(|value| *value != 0.0));
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    assert_eq!(
        cpu.train_buffers,
        gpu.cpu_fallback_train_buffers.borrow().clone()
    );

    cpu.update_weights(0.01, 0.2, 0.1, 1);
    gpu.update_weights_gpu(0.01, 0.2, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
}

#[test]
fn kimi_kda_host_adam_updates_match_fake_backend_and_persist_state() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "kimi-adam.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\nadam=1\ndecay=0.1\n\
                 [kimi_kda]\nrank=1\nexperts=2\nhidden=2\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    if let LayerParams::KimiKda(params) = &mut cpu.params[0] {
        params.weights.fill(0.25);
        params.biases.fill(0.5);
    } else {
        panic!("Kimi KDA params")
    }
    if let LayerTrainBuffers::KimiKda(train) = &mut cpu.train_buffers[0] {
        train.weight_updates.fill(-0.5);
        train.bias_updates.fill(-0.5);
    } else {
        panic!("Kimi KDA train buffers")
    }
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    *gpu.cpu_fallback_train_buffers.borrow_mut() = cpu.train_buffers.clone();
    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    assert_eq!(
        gpu.cpu_fallback_train_buffers.borrow().clone(),
        cpu.train_buffers
    );

    let path = std::env::temp_dir().join(format!("darknet-kimi-adam-{}.state", std::process::id()));
    cpu.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step(), cpu.adam_step());
    let round_trip = path.with_extension("round-trip.state");
    restored.save_optimizer_state(&round_trip).unwrap();
    assert_eq!(fs::read(&path).unwrap(), fs::read(&round_trip).unwrap());
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(round_trip);
}

#[test]
fn mambavision_sgd_decay_applies_only_to_source_weight_blocks() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "mambavision-sgd.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\n\
                 [mambavision]\nfilters=4\nstate=1\nconv_size=1\ndt_rank=1\nffn_ratio=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::MambaVision(params) = &mut executable.params[0] else {
        panic!("MambaVision params")
    };
    params.values.fill(1.0);
    let LayerTrainBuffers::MambaVision(train) = &mut executable.train_buffers[0] else {
        panic!("MambaVision train buffers")
    };
    train.updates.fill(0.0);
    executable.update_weights(0.1, 0.0, 0.2, 1);
    let LayerParams::MambaVision(params) = &executable.params[0] else {
        panic!("MambaVision params")
    };
    let blocks = crate::mambavision::mambavision_param_blocks(params).unwrap();
    for block in blocks {
        let expected = if block.decay { 0.98 } else { 1.0 };
        assert!(
            params.values[block.range.clone()]
                .iter()
                .all(|value| (*value - expected).abs() < 1.0e-6),
            "unexpected decay behavior for {}",
            block.name
        );
    }
}

#[test]
fn mambavision_host_adam_updates_match_fake_backend() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "mambavision-adam.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\nadam=1\ndecay=0.1\n\
                 [mambavision]\nstate=1\nconv_size=1\ndt_rank=1\nffn_ratio=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    if let LayerParams::MambaVision(params) = &mut cpu.params[0] {
        params.values.fill(0.25);
    } else {
        panic!("MambaVision params")
    }
    if let LayerTrainBuffers::MambaVision(train) = &mut cpu.train_buffers[0] {
        train.updates.fill(-0.5);
    } else {
        panic!("MambaVision train buffers")
    }
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    *gpu.cpu_fallback_train_buffers.borrow_mut() = cpu.train_buffers.clone();
    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    assert_eq!(
        gpu.cpu_fallback_train_buffers.borrow().clone(),
        cpu.train_buffers
    );
    let path = std::env::temp_dir().join(format!(
        "darknet-mambavision-adam-{}.state",
        std::process::id()
    ));
    cpu.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step(), cpu.adam_step());
    let round_trip = path.with_extension("round-trip.state");
    restored.save_optimizer_state(&round_trip).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&round_trip).unwrap()
    );
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(round_trip);
}

#[test]
fn mambavision_optimizer_accumulator_is_negative_mathematical_gradient() {
    let network=Network::from_config(ConfigFile::parse_str("mamba-descent.cfg",
        "[net]\nwidth=1\nheight=1\nchannels=2\nbatch=1\n[mambavision]\nfilters=2\nstate=1\nconv_size=1\ndt_rank=1\nffn_ratio=1\nactivation=linear\n").unwrap()).unwrap();
    let mut cpu=ExecutableNetwork::new(network);
    let input=Tensor::from_data(cpu.network.net.input_shape(),vec![0.2,-0.4]).unwrap();
    let forward=cpu.forward_training(&input).unwrap();
    let output=forward.final_output().unwrap();
    let seed=Tensor::from_data(output.shape,vec![0.3,-0.7]).unwrap();
    let backward=cpu.backward_accumulate_seeded(&input,&forward,&[Some(seed)]).unwrap();
    let LayerParams::MambaVision(grad)=&backward.param_gradients[0]else{panic!("Mamba gradient")};
    let LayerTrainBuffers::MambaVision(train)=&cpu.train_buffers[0]else{panic!("Mamba buffers")};
    assert!(grad.values.iter().any(|x|x.abs()>1e-4));
    for(i,(&direction,&derivative))in train.updates.iter().zip(&grad.values).enumerate(){
        assert!((direction+derivative).abs()<1e-6,"parameter {i}: direction {direction}, derivative {derivative}");
    }
}

#[test]
fn tucker_attention_host_adam_and_fake_backend_updates_match() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "tucker-adam.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\nadam=1\ndecay=0.1\n\
                 [tucker]\nsize=1\nheads=1\nrank=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    if let LayerParams::TuckerAttention(params) = &mut cpu.params[0] {
        params.weights.fill(0.25);
        params.biases.fill(0.5);
    } else {
        panic!("Tucker attention params")
    }
    if let LayerTrainBuffers::TuckerAttention(train) = &mut cpu.train_buffers[0] {
        train.weight_updates.fill(-0.5);
        train.bias_updates.fill(-0.5);
    } else {
        panic!("Tucker attention train buffers")
    }
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    *gpu.cpu_fallback_train_buffers.borrow_mut() = cpu.train_buffers.clone();

    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    assert_eq!(
        gpu.cpu_fallback_train_buffers.borrow().clone(),
        cpu.train_buffers
    );

    let dir = std::env::temp_dir().join(format!("darknet-tucker-adam-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("optimizer.state");
    cpu.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step(), cpu.adam_step());
    let round_trip = dir.join("optimizer-round-trip.state");
    restored.save_optimizer_state(&round_trip).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&round_trip).unwrap()
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn swin_cpu_runtime_backward_accumulates_and_updates_parameters() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "swin-runtime-chain.cfg",
            "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=1\n\
             [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
             [swin]\nfilters=3\nsize=2\nheads=1\nshift=1\nffn_ratio=2\nactivation=gelu\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut executable.params, 91);
    let LayerParams::Convolutional(conv) = &mut executable.params[0] else {
        panic!("preceding convolution params")
    };
    conv.weights.copy_from_slice(&[0.7, -0.25, 0.2, 0.6]);
    conv.biases.copy_from_slice(&[0.1, -0.05]);
    let LayerParams::Swin(swin) = &mut executable.params[1] else {
        panic!("Swin params")
    };
    swin.mhc.copy_from_slice(&[
        0.3, -0.2, 0.1, 0.4, 0.2, -0.1, -0.3, 0.1, 0.25, 0.2, -0.15, 0.3,
    ]);
    let input = Tensor::from_data(
        TensorShape::new(3, 2, 2, 1),
        (0..12).map(|index| (index as f32 - 5.0) * 0.07).collect(),
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    let output = forward.final_output().unwrap();
    assert_eq!(output.shape, TensorShape::new(3, 2, 3, 1));
    let seed = Tensor::from_data(
        output.shape,
        (0..output.len())
            .map(|index| (index as f32 * 0.31 + 0.2).sin() * 0.2)
            .collect(),
    )
    .unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    let LayerParams::Swin(swin_gradient) = &backward.param_gradients[1] else {
        panic!("Swin parameter gradient")
    };
    let parameter_index = 0;
    let epsilon = 1.0e-3;
    let objective = |model: &ExecutableNetwork, x: &Tensor| {
        model
            .forward(x)
            .unwrap()
            .final_output()
            .unwrap()
            .data
            .iter()
            .zip(&seed.data)
            .map(|(value, gradient)| value * gradient)
            .sum::<f32>()
    };
    let mut plus = executable.clone();
    let mut minus = executable.clone();
    let LayerParams::Swin(weights) = &mut plus.params[1] else {
        unreachable!()
    };
    weights.ffn_w2[parameter_index] += epsilon;
    let LayerParams::Swin(weights) = &mut minus.params[1] else {
        unreachable!()
    };
    weights.ffn_w2[parameter_index] -= epsilon;
    let numerical = (objective(&plus, &input) - objective(&minus, &input)) / (2.0 * epsilon);
    let analytic = swin_gradient.ffn_w2[parameter_index];
    assert!(
        (analytic - numerical).abs() < 3.0e-3 + 3.0e-2 * analytic.abs().max(numerical.abs()),
        "runtime ffn_w2 gradient: analytic={analytic} numeric={numerical}"
    );

    let before = match &executable.params[1] {
        LayerParams::Swin(weights) => weights.clone(),
        _ => unreachable!(),
    };
    executable.zero_train_buffers();
    executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    let LayerTrainBuffers::Swin(train) = &executable.train_buffers[1] else {
        panic!("Swin train buffers")
    };
    assert!(
        train
            .updates
            .to_blob()
            .unwrap()
            .iter()
            .any(|value| *value != 0.0)
    );
    executable.update_weights(0.005, 0.4, 0.0, 1);
    let LayerParams::Swin(after) = &executable.params[1] else {
        panic!("Swin params")
    };
    assert_ne!(*after, before);
}

#[test]
fn swin_sgd_update_clips_then_decays_only_matrix_parameters() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "swin-update.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=2\n\
             [swin]\nfilters=3\nsize=1\nheads=1\nffn_ratio=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Swin(params) = &mut executable.params[0] else {
        panic!("Swin params")
    };
    params.qkv_weight[0] = 1.0;
    params.mhc[0] = 7.99;
    let LayerTrainBuffers::Swin(train) = &mut executable.train_buffers[0] else {
        panic!("Swin train buffers")
    };
    train.updates.qkv_weight[0] = -0.5;
    train.updates.qkv_bias[0] = -0.5;
    train.updates.mhc[0] = 1.0;
    executable.update_weights(0.1, 0.0, 0.2, 1);
    let LayerParams::Swin(params) = &executable.params[0] else {
        panic!("Swin params")
    };
    assert!((params.qkv_weight[0] - 0.93).abs() < 1.0e-6);
    assert!((params.qkv_bias[0] + 0.05).abs() < 1.0e-6);
    assert_eq!(params.mhc[0], 8.0);
}

#[test]
fn swin_adam_updates_and_optimizer_state_round_trip() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "swin-adam.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=2\nadam=1\n\
             [swin]\nfilters=2\nsize=1\nheads=1\nffn_ratio=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::Swin(params) = &mut executable.params[0] else {
        panic!("Swin params")
    };
    params.qkv_weight[0] = 0.25;
    params.relative_position_bias[0] = 0.1;
    let LayerTrainBuffers::Swin(train) = &mut executable.train_buffers[0] else {
        panic!("Swin train buffers")
    };
    train.updates.qkv_weight[0] = -0.2;
    train.updates.qkv_bias[0] = -0.1;
    train.updates.relative_position_bias[0] = -0.15;
    train.updates.mhc[0] = -0.05;
    executable.update_weights_adam(0.01, 1).unwrap();
    assert_eq!(executable.adam_step(), 1);
    let LayerParams::Swin(updated) = &executable.params[0] else {
        unreachable!()
    };
    assert_ne!(updated.qkv_weight[0], 0.25);
    assert_ne!(updated.relative_position_bias[0], 0.1);

    let path = std::env::temp_dir().join("darknet-swin-adam-roundtrip.dnopt");
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step(), 1);
    assert_eq!(restored.host_adam_state, executable.host_adam_state);
    let _ = fs::remove_file(path);
}

#[test]
fn swin_gpu_host_round_trip_matches_cpu_chain_and_backward_updates() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "swin-fake-backend-chain.cfg",
            // `revleaky` has no device activation, so this Swin keeps the host
            // round trip and the CPU backward fallback that this test covers.
            "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=1\n\
             [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
             [swin]\nfilters=3\nsize=2\nheads=1\nshift=1\nffn_ratio=2\nactivation=revleaky\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 107);
    let input = Tensor::from_data(
        TensorShape::new(3, 2, 2, 1),
        (0..12).map(|index| (index as f32 - 4.0) * 0.06).collect(),
    )
    .unwrap();
    let cpu_forward = cpu.forward(&input).unwrap();
    let output = cpu_forward.final_output().unwrap();
    let seeds = vec![
        None,
        Some(Tensor::from_data(output.shape, vec![0.2; output.len()]).unwrap()),
    ];
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    assert!(gpu.requires_cpu_backward_fallback());
    let cpu_backward = cpu.backward_seeded(&input, &cpu_forward, &seeds).unwrap();
    let fake_backend_backward = cpu
        .backward_seeded(&input, &gpu_forward.cpu_forward, &seeds)
        .unwrap();
    assert_eq!(
        fake_backend_backward.input_gradient,
        cpu_backward.input_gradient
    );
    assert_eq!(
        fake_backend_backward.param_gradients,
        cpu_backward.param_gradients
    );
    cpu.zero_train_buffers();
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    gpu.zero_train_buffers_gpu().unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    assert_eq!(
        cpu.train_buffers[1],
        gpu.cpu_fallback_train_buffers.borrow()[1]
    );
    cpu.update_weights(0.003, 0.4, 0.0, 1);
    gpu.update_weights_gpu(0.003, 0.4, 0.0, 1).unwrap();
    let LayerParams::Swin(cpu_swin) = &cpu.params[1] else {
        unreachable!()
    };
    let LayerParams::Swin(gpu_swin) = &gpu.params[1] else {
        unreachable!()
    };
    assert_eq!(gpu_swin, cpu_swin);
}

#[test]
fn swin_gpu_native_path_matches_cpu_forward_backward_and_staged_gradients() {
    // Shifted windows, bottom/right padding (3x5 input, window 2), two heads,
    // a C->N residual projection and batch 2 (per-sample mask reuse).
    let network = Network::from_config(
        ConfigFile::parse_str(
            "swin-native-fake-backend.cfg",
            "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n\
             [swin]\nfilters=4\nsize=2\nheads=2\nshift=1\nffn_ratio=2\nactivation=gelu\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 211);
    let LayerParams::Swin(weights) = &mut cpu.params[0] else {
        panic!("Swin params")
    };
    for (index, value) in weights.relative_position_bias.iter_mut().enumerate() {
        *value = ((index % 5) as f32 - 2.0) * 0.03;
    }
    weights.mhc.copy_from_slice(&[
        0.3, -0.2, 0.1, 0.4, 0.2, -0.1, -0.3, 0.1, 0.25, 0.2, -0.15, 0.3,
    ]);
    let input = Tensor::from_data(
        TensorShape::new(5, 3, 2, 2),
        (0..60).map(|index| ((index as f32) * 0.37).sin() * 0.8).collect(),
    )
    .unwrap();
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    let close = |name: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (index, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{name}[{index}]: gpu={g} cpu={w}"
            );
        }
    };

    let cpu_output = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let gpu_output = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    assert_eq!(gpu_output.shape, cpu_output.shape);
    close("inference output", &gpu_output.data, &cpu_output.data);

    let seed = Tensor::from_data(
        cpu_output.shape,
        (0..cpu_output.len())
            .map(|index| ((index as f32) * 0.41 + 0.2).cos() * 0.3)
            .collect(),
    )
    .unwrap();
    let seeds = [Some(seed)];
    gpu.init_train_buffers_gpu().unwrap();
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    close(
        "training output",
        &gpu_state.cpu_forward.final_output().unwrap().data,
        &cpu_forward.final_output().unwrap().data,
    );
    // The native training forward keeps no host cache.
    assert!(gpu_state.cpu_forward.swin_caches[0].is_none());
    let cpu_backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &seeds)
        .unwrap()
        .expect("captured Swin input gradient");
    close(
        "input gradient",
        &gpu_input_gradient.data,
        &cpu_backward.input_gradient.data,
    );
    let LayerTrainBuffers::Swin(cpu_train) = &cpu.train_buffers[0] else {
        panic!("Swin CPU train buffers")
    };
    let buffers = gpu.host_fallback_train_buffers();
    let LayerTrainBuffers::Swin(gpu_train) = &buffers[0] else {
        panic!("Swin native-staged updates")
    };
    close(
        "staged updates",
        &gpu_train.updates.to_blob().unwrap(),
        &cpu_train.updates.to_blob().unwrap(),
    );
}

#[test]
fn vit_gpu_host_fallback_matches_a_preceding_convolution_and_backward() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "vit-chain.cfg",
            // `revleaky` has no device activation, so this ViT keeps the host
            // fallback that this test covers.
            "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
             [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
             [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\nflash_attention=1\nactivation=revleaky\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 41);
    let LayerParams::Convolutional(conv) = &mut cpu.params[0] else {
        panic!("conv params")
    };
    conv.weights[0] = 0.7;
    conv.biases[0] = -0.1;
    let input = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        (0..16).map(|i| (i as f32 - 7.0) * 0.04).collect(),
    )
    .unwrap();
    let cpu_forward = cpu.forward(&input).unwrap();

    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    let output = gpu_forward.cpu_forward.final_output().unwrap();
    let seed = Tensor::from_data(output.shape, vec![0.2; output.len()]).unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[None, Some(seed)])
        .unwrap();
    let fallback = gpu.cpu_fallback_train_buffers.borrow();
    let LayerTrainBuffers::Vit(vit) = &fallback[1] else {
        panic!("ViT fallback updates")
    };
    assert!(vit.updates.to_blob().iter().any(|value| *value != 0.0));
}

#[test]
fn vit_vc_gpu_host_fallback_matches_cpu_forward_and_backward() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "vit-vc-chain.cfg",
                "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
             [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
             [vit]\ndim=16\nheads=1\npatch_size=1\npatch_stride=1\nmlp_dim=16\nflash_attention=1\nvc_attention=1\nvc_expcast=1\nvc_perm=spatial8\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 71);
    let LayerParams::Convolutional(conv) = &mut cpu.params[0] else {
        panic!("convolution params")
    };
    conv.weights[0] = 0.7;
    conv.biases[0] = -0.1;
    let input = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        (0..16).map(|index| (index as f32 - 6.0) * 0.07).collect(),
    )
    .unwrap();
    let cpu_forward = cpu.forward(&input).unwrap();

    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    let output = gpu_forward.cpu_forward.final_output().unwrap();
    let seed = Tensor::from_data(output.shape, vec![0.2; output.len()]).unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[None, Some(seed)])
        .unwrap();
    let fallback = gpu.cpu_fallback_train_buffers.borrow();
    let LayerTrainBuffers::Vit(vit) = &fallback[1] else {
        panic!("ViT fallback updates")
    };
    assert!(vit.updates.to_blob().iter().any(|value| *value != 0.0));
}

#[test]
fn legacy_detection_layer_preserves_forward_and_backward_tensors() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "detection.cfg",
            r#"
                    [net]
                    width=1
                    height=1
                    channels=7
                    batch=1

                    [detection]
                    classes=2
                    coords=4
                    side=1
                    num=1
                "#,
        )
        .expect("cfg parses"),
    )
    .expect("network constructs");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0; 7])
        .expect("input tensor");
    let forward = executable
        .forward_training(&input)
        .expect("forward succeeds");
    assert_eq!(forward.final_output(), Some(&input));

    let gradient = Tensor::filled(input.shape, 0.25);
    let backward = executable
        .backward(&input, &forward, &gradient)
        .expect("backward succeeds");
    assert_eq!(backward.input_gradient, gradient);
}

#[test]
fn crop_layer_center_samples_and_scatters_bilinear_gradients() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "crop.cfg",
            r#"
                    [net]
                    width=4
                    height=3
                    channels=1
                    batch=1

                    [crop]
                    crop_width=2
                    crop_height=2
                    noadjust=1
                "#,
        )
        .expect("cfg parses"),
    )
    .expect("network constructs");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        (0..12).map(|value| value as f32).collect(),
    )
    .expect("input tensor");
    let forward = executable.forward(&input).expect("crop forward succeeds");
    assert_eq!(
        forward.final_output().expect("crop output").data,
        vec![3.0, 4.0, 7.0, 8.0]
    );

    let gradient = Tensor::filled(forward.final_output().expect("output").shape, 1.0);
    let backward = executable
        .backward(&input, &forward, &gradient)
        .expect("crop backward succeeds");
    assert_eq!(
        backward.input_gradient.data,
        vec![0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0,]
    );
}

#[test]
fn crop_axis_layout_maps_batches_channels_and_center_window() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "crop-axis.cfg",
            r#"
                    [net]
                    width=5
                    height=3
                    channels=2
                    batch=2

                    [crop]
                    crop_width=3
                    crop_height=1
                    noadjust=1
                "#,
        )
        .expect("cfg parses"),
    )
    .expect("network constructs");
    let executable = ExecutableNetwork::new(network);
    let shape = executable.network.net.input_shape();
    let mut data = vec![0.0; shape.batch * shape.element_count()];
    for b in 0..shape.batch {
        for c in 0..shape.channels {
            for y in 0..shape.height {
                for x in 0..shape.width {
                    data[((b * shape.channels + c) * shape.height + y) * shape.width + x] =
                        (b * 1000 + c * 100 + y * 10 + x) as f32;
                }
            }
        }
    }
    let input = Tensor::from_data(shape, data).expect("input tensor");
    let forward = executable.forward(&input).expect("crop forward succeeds");
    assert_eq!(
        forward.final_output().expect("crop output").data,
        vec![
            11.0, 12.0, 13.0, 111.0, 112.0, 113.0, 1011.0, 1012.0, 1013.0, 1111.0, 1112.0, 1113.0
        ]
    );
    let cache = forward.crop_caches[0]
        .as_ref()
        .expect("crop coordinate cache");
    assert_eq!(cache.source_x, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(cache.source_y, vec![1.0; 6]);
}

#[test]
fn crop_layer_applies_darknet_level_adjustment_before_sampling() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "crop-levels.cfg",
            r#"
                    [net]
                    width=1
                    height=1
                    channels=3
                    batch=1

                    [crop]
                    crop_width=1
                    crop_height=1
                    saturation=1
                    exposure=1
                    shift=0
                "#,
        )
        .expect("cfg parses"),
    )
    .expect("network constructs");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![0.25, 0.5, 0.75])
        .expect("input tensor");
    let output = executable
        .forward(&input)
        .expect("crop forward succeeds")
        .final_output()
        .expect("crop output")
        .clone();
    assert_eq!(output.data, vec![-0.5, 0.0, 0.5]);
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FakeTensor {
    pub(super) shape: Vec<usize>,
    pub(super) data: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct FakeBackend;

impl F32TensorOpsBackend for FakeBackend {
    type Error = String;
    type Tensor = FakeTensor;

    fn upload_f32(
        &self,
        shape: &[usize],
        host: &[f32],
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let expected = shape.iter().product::<usize>();
        if host.len() != expected {
            return Err(format!(
                "upload_f32 expected {expected} elements for shape {:?}, got {}",
                shape,
                host.len()
            ));
        }
        Ok(FakeTensor {
            shape: shape.to_vec(),
            data: host.to_vec(),
        })
    }

    fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
        Ok(FakeTensor {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        })
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> std::result::Result<Vec<f32>, Self::Error> {
        Ok(tensor.data.clone())
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if bias.data.len() != channels {
            return Err("bias_add_f32 bias length mismatch".to_owned());
        }
        let batch = x.data.len() / (channels * channel_stride);
        let mut out = x.data.clone();
        for b in 0..batch {
            for c in 0..channels {
                let base = (b * channels + c) * channel_stride;
                for idx in 0..channel_stride {
                    out[base + idx] += bias.data[c];
                }
            }
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: out,
        })
    }

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if scale.data.len() != channels {
            return Err("channel_scale_f32 scale length mismatch".to_owned());
        }
        let batch = x.data.len() / (channels * channel_stride);
        let mut out = x.data.clone();
        for b in 0..batch {
            for c in 0..channels {
                let base = (b * channels + c) * channel_stride;
                for idx in 0..channel_stride {
                    out[base + idx] *= scale.data[c];
                }
            }
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: out,
        })
    }

    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = output_grad.data.len() / (channels * channel_stride);
        let mut bias_grad = vec![0.0; channels];
        for b in 0..batch {
            for (c, slot) in bias_grad.iter_mut().enumerate().take(channels) {
                let base = (b * channels + c) * channel_stride;
                *slot += output_grad.data[base..base + channel_stride]
                    .iter()
                    .sum::<f32>();
            }
        }
        Ok(FakeTensor {
            shape: vec![channels],
            data: bias_grad,
        })
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: x
                .data
                .iter()
                .copied()
                .map(|value| apply_fake_activation(activation, value))
                .collect(),
        })
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if activation_input.data.len() != delta.data.len() {
            return Err("activation_backward_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: activation_input.shape.clone(),
            data: activation_input
                .data
                .iter()
                .zip(&delta.data)
                .map(|(value, grad)| grad * fake_activation_gradient(activation, *value))
                .collect(),
        })
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("shortcut_add_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: x
                .data
                .iter()
                .zip(&y.data)
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        })
    }

    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if inputs.len() != input_channel_offsets.len()
            || inputs.len() != input_channels.len()
            || inputs.len() != input_total_channels.len()
        {
            return Err("route_concat_f32 metadata length mismatch".to_owned());
        }
        let batch = inputs
            .first()
            .and_then(|tensor| tensor.shape.first().copied())
            .ok_or("route_concat_f32 requires at least one input")?;
        let total_channels = input_channels.iter().sum::<usize>();
        let mut out = vec![0.0; output_shape.iter().product()];
        let mut channel_offset = 0usize;

        for (((input, source_channel_offset), src_channels), source_total_channels) in inputs
            .iter()
            .zip(input_channel_offsets.iter().copied())
            .zip(input_channels.iter().copied())
            .zip(input_total_channels.iter().copied())
        {
            for b in 0..batch {
                for c in 0..src_channels {
                    let src_base =
                        (b * source_total_channels + source_channel_offset + c) * channel_stride;
                    let dst_base = (b * total_channels + channel_offset + c) * channel_stride;
                    out[dst_base..dst_base + channel_stride]
                        .copy_from_slice(&input.data[src_base..src_base + channel_stride]);
                }
            }
            channel_offset += src_channels;
        }

        Ok(FakeTensor {
            shape: output_shape.to_vec(),
            data: out,
        })
    }

    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
        if input_shapes.len() != input_channel_offsets.len()
            || input_shapes.len() != input_channels.len()
            || input_shapes.len() != input_total_channels.len()
        {
            return Err("route_backward_f32 metadata length mismatch".to_owned());
        }
        let batch = output_grad.shape[0];
        let mut output_channel_offset = 0usize;
        let mut out = Vec::with_capacity(input_shapes.len());

        for (((input_shape, source_channel_offset), src_channels), source_total_channels) in
            input_shapes
                .iter()
                .zip(input_channel_offsets.iter().copied())
                .zip(input_channels.iter().copied())
                .zip(input_total_channels.iter().copied())
        {
            let mut input_grad = vec![0.0; input_shape.iter().product()];
            for b in 0..batch {
                for c in 0..src_channels {
                    let src_base =
                        (b * output_grad.shape[1] + output_channel_offset + c) * channel_stride;
                    let dst_base =
                        (b * source_total_channels + source_channel_offset + c) * channel_stride;
                    input_grad[dst_base..dst_base + channel_stride]
                        .copy_from_slice(&output_grad.data[src_base..src_base + channel_stride]);
                }
            }
            out.push(FakeTensor {
                shape: input_shape.clone(),
                data: input_grad,
            });
            output_channel_offset += src_channels;
        }

        Ok(out)
    }

    fn conv2d_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let input = Tensor::from_data(
            TensorShape::new(input_width, input_height, input_channels, x.shape[0]),
            x.data.clone(),
        )
        .map_err(|err| err.to_string())?;
        let output_shape = TensorShape::new(
            output_shape[3],
            output_shape[2],
            output_shape[1],
            output_shape[0],
        );
        let spec = ConvolutionalSpec {
            filters,
            size: kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            share_index: None,
            batch_normalize: false,
            activation: Activation::Linear,
        };
        let params = ConvolutionalParams {
            biases: vec![0.0; filters],
            batch_norm: None,
            weights: weights.data.clone(),
        };
        let output = super::forward_convolutional(&spec, &params, &input, output_shape);
        Ok(FakeTensor {
            shape: vec![
                output.shape.batch,
                output.shape.channels,
                output.shape.height,
                output.shape.width,
            ],
            data: output.data,
        })
    }

    fn conv2d_backward_input_f32(
        &self,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        input_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = input_shape[0];
        let channels_per_group = input_channels / groups;
        let filters_per_group = filters / groups;
        let output_height = output_grad.shape[2];
        let output_width = output_grad.shape[3];
        let mut input_grad = vec![0.0; input_shape.iter().product()];

        for b in 0..batch {
            for channel in 0..input_channels {
                let group = channel / channels_per_group;
                let input_channel_in_group = channel % channels_per_group;
                for iy in 0..input_height {
                    for ix in 0..input_width {
                        let mut acc = 0.0;
                        for filter_in_group in 0..filters_per_group {
                            let filter = group * filters_per_group + filter_in_group;
                            for ky in 0..kernel_size {
                                let oy_num =
                                    iy as isize + padding as isize - (ky * dilation) as isize;
                                if oy_num < 0 || oy_num % stride_y as isize != 0 {
                                    continue;
                                }
                                let oy = (oy_num / stride_y as isize) as usize;
                                if oy >= output_height {
                                    continue;
                                }
                                for kx in 0..kernel_size {
                                    let ox_num =
                                        ix as isize + padding as isize - (kx * dilation) as isize;
                                    if ox_num < 0 || ox_num % stride_x as isize != 0 {
                                        continue;
                                    }
                                    let ox = (ox_num / stride_x as isize) as usize;
                                    if ox >= output_width {
                                        continue;
                                    }

                                    let grad_index = ((((b * filters) + filter) * output_height)
                                        + oy)
                                        * output_width
                                        + ox;
                                    let weight_index = ((filter * channels_per_group
                                        + input_channel_in_group)
                                        * kernel_size
                                        + ky)
                                        * kernel_size
                                        + kx;
                                    acc +=
                                        output_grad.data[grad_index] * weights.data[weight_index];
                                }
                            }
                        }
                        let input_index = ((((b * input_channels) + channel) * input_height) + iy)
                            * input_width
                            + ix;
                        input_grad[input_index] = acc;
                    }
                }
            }
        }

        Ok(FakeTensor {
            shape: input_shape.to_vec(),
            data: input_grad,
        })
    }

    fn conv2d_backward_weights_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        weight_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = x.shape[0];
        let channels_per_group = input_channels / groups;
        let filters_per_group = filters / groups;
        let output_height = output_grad.shape[2];
        let output_width = output_grad.shape[3];
        let mut weight_grad = vec![0.0; weight_shape.iter().product()];

        for filter in 0..filters {
            let group = filter / filters_per_group;
            for input_channel_in_group in 0..channels_per_group {
                let input_channel = group * channels_per_group + input_channel_in_group;
                for ky in 0..kernel_size {
                    for kx in 0..kernel_size {
                        let mut acc = 0.0;
                        for b in 0..batch {
                            for oy in 0..output_height {
                                let in_y = oy * stride_y + ky * dilation;
                                let src_y = in_y as isize - padding as isize;
                                if src_y < 0 || src_y >= input_height as isize {
                                    continue;
                                }
                                for ox in 0..output_width {
                                    let in_x = ox * stride_x + kx * dilation;
                                    let src_x = in_x as isize - padding as isize;
                                    if src_x < 0 || src_x >= input_width as isize {
                                        continue;
                                    }
                                    let input_index = ((((b * input_channels) + input_channel)
                                        * input_height)
                                        + src_y as usize)
                                        * input_width
                                        + src_x as usize;
                                    let grad_index = ((((b * filters) + filter) * output_height)
                                        + oy)
                                        * output_width
                                        + ox;
                                    acc += x.data[input_index] * output_grad.data[grad_index];
                                }
                            }
                        }
                        let weight_index = ((filter * channels_per_group + input_channel_in_group)
                            * kernel_size
                            + ky)
                            * kernel_size
                            + kx;
                        weight_grad[weight_index] = acc;
                    }
                }
            }
        }

        Ok(FakeTensor {
            shape: weight_shape.to_vec(),
            data: weight_grad,
        })
    }

    fn upsample_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = x.shape[0];
        let output_height = output_shape[2];
        let output_width = output_shape[3];
        let mut out = vec![0.0; output_shape.iter().product()];

        if reverse {
            for b in 0..batch {
                for c in 0..channels {
                    for y in 0..input_height {
                        for x_idx in 0..input_width {
                            let src =
                                ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                            let dst_y = y / stride;
                            let dst_x = x_idx / stride;
                            let dst = ((((b * channels) + c) * output_height) + dst_y)
                                * output_width
                                + dst_x;
                            out[dst] += scale * x.data[src];
                        }
                    }
                }
            }
        } else {
            for b in 0..batch {
                for c in 0..channels {
                    for y in 0..output_height {
                        for x_idx in 0..output_width {
                            let src_y = y / stride;
                            let src_x = x_idx / stride;
                            let src = ((((b * channels) + c) * input_height) + src_y) * input_width
                                + src_x;
                            let dst =
                                ((((b * channels) + c) * output_height) + y) * output_width + x_idx;
                            out[dst] = scale * x.data[src];
                        }
                    }
                }
            }
        }

        Ok(FakeTensor {
            shape: output_shape.to_vec(),
            data: out,
        })
    }

    fn upsample_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.upsample_f32(
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            true,
            config,
        )
    }

    fn maxpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let input = Tensor::from_data(
            TensorShape::new(input_width, input_height, channels, x.shape[0]),
            x.data.clone(),
        )
        .map_err(|err| err.to_string())?;
        let output_shape = TensorShape::new(
            output_shape[3],
            output_shape[2],
            output_shape[1],
            output_shape[0],
        );
        let output = super::forward_maxpool(
            size,
            stride_x,
            stride_y,
            padding,
            &input,
            output_shape,
            false,
        )
        .map_err(|err| err.to_string())?;
        Ok(FakeTensor {
            shape: vec![
                output.shape.batch,
                output.shape.channels,
                output.shape.height,
                output.shape.width,
            ],
            data: output.data,
        })
    }

    fn maxpool_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = x.shape[0];
        let output_height = output_grad.shape[2];
        let output_width = output_grad.shape[3];
        let pad_offset = (padding / 2) as isize;
        let mut input_grad = vec![0.0; x.data.len()];

        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..output_height {
                    for ox in 0..output_width {
                        let mut best_index = None;
                        let mut best_value = f32::NEG_INFINITY;
                        for ky in 0..size {
                            let in_y = oy as isize * stride_y as isize + ky as isize - pad_offset;
                            if !(0..input_height as isize).contains(&in_y) {
                                continue;
                            }
                            for kx in 0..size {
                                let in_x =
                                    ox as isize * stride_x as isize + kx as isize - pad_offset;
                                if !(0..input_width as isize).contains(&in_x) {
                                    continue;
                                }
                                let input_index = ((((b * channels) + c) * input_height)
                                    + in_y as usize)
                                    * input_width
                                    + in_x as usize;
                                let value = x.data[input_index];
                                if value > best_value {
                                    best_value = value;
                                    best_index = Some(input_index);
                                }
                            }
                        }

                        if let Some(input_index) = best_index {
                            let output_index =
                                ((((b * channels) + c) * output_height) + oy) * output_width + ox;
                            input_grad[input_index] += output_grad.data[output_index];
                        }
                    }
                }
            }
        }

        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: input_grad,
        })
    }

    fn local_avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let input = Tensor::from_data(
            TensorShape::new(input_width, input_height, channels, x.shape[0]),
            x.data.clone(),
        )
        .map_err(|err| err.to_string())?;
        let output_shape = TensorShape::new(
            output_shape[3],
            output_shape[2],
            output_shape[1],
            output_shape[0],
        );
        let output = super::forward_maxpool(
            size,
            stride_x,
            stride_y,
            padding,
            &input,
            output_shape,
            true,
        )
        .map_err(|err| err.to_string())?;
        Ok(FakeTensor {
            shape: vec![
                output.shape.batch,
                output.shape.channels,
                output.shape.height,
                output.shape.width,
            ],
            data: output.data,
        })
    }

    fn local_avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let input = Tensor::zeros(TensorShape::new(
            input_width,
            input_height,
            channels,
            input_shape[0],
        ));
        let output_grad = Tensor::from_data(
            TensorShape::new(
                output_grad.shape[3],
                output_grad.shape[2],
                output_grad.shape[1],
                output_grad.shape[0],
            ),
            output_grad.data.clone(),
        )
        .map_err(|err| err.to_string())?;
        let input_grad = super::backward_maxpool(
            size,
            stride_x,
            stride_y,
            padding,
            &input,
            &output_grad,
            true,
            false,
        )
        .map_err(|err| err.to_string())?;
        Ok(FakeTensor {
            shape: input_shape.to_vec(),
            data: input_grad.data,
        })
    }

    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err("avgpool_f32 output shape mismatch".to_owned());
        }
        let batch = x.shape[0];
        let area = (input_height * input_width) as f32;
        let mut out = vec![0.0; output_shape.iter().product()];
        for b in 0..batch {
            for c in 0..channels {
                let mut acc = 0.0;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let src = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        acc += x.data[src];
                    }
                }
                out[b * channels + c] = acc / area;
            }
        }
        Ok(FakeTensor {
            shape: output_shape.to_vec(),
            data: out,
        })
    }

    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let batch = input_shape[0];
        let area = (input_height * input_width) as f32;
        let mut input_grad = vec![0.0; input_shape.iter().product()];
        for b in 0..batch {
            for c in 0..channels {
                let grad = output_grad.data[b * channels + c] / area;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let dst = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        input_grad[dst] = grad;
                    }
                }
            }
        }
        Ok(FakeTensor {
            shape: input_shape.to_vec(),
            data: input_grad,
        })
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("vector_add_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: x
                .data
                .iter()
                .zip(&y.data)
                .map(|(lhs, rhs)| lhs + rhs)
                .collect(),
        })
    }

    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("saxpy_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: x
                .data
                .iter()
                .zip(&y.data)
                .map(|(lhs, rhs)| alpha * lhs + rhs)
                .collect(),
        })
    }

    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
        if weights.shape != updates.shape {
            return Err("sgd_update_f32 shape mismatch".to_owned());
        }
        let mut new_weights = weights.data.clone();
        let mut new_updates = updates.data.clone();
        super::sgd_update(
            &mut new_weights,
            &mut new_updates,
            lr,
            momentum,
            decay,
            batch,
        );
        Ok((
            FakeTensor {
                shape: weights.shape.clone(),
                data: new_weights,
            },
            FakeTensor {
                shape: updates.shape.clone(),
                data: new_updates,
            },
        ))
    }
}

pub(super) fn apply_fake_activation(activation: F32ActivationKind, value: f32) -> f32 {
    let activation = match activation {
        F32ActivationKind::Linear => Activation::Linear,
        F32ActivationKind::Loggy => Activation::Loggy,
        F32ActivationKind::Hardtan => Activation::Hardtan,
        F32ActivationKind::Relie => Activation::Relie,
        F32ActivationKind::Lhtan => Activation::Lhtan,
        F32ActivationKind::Plse => Activation::Plse,
        F32ActivationKind::Stair => Activation::Stair,
        F32ActivationKind::Logistic => Activation::Logistic,
        F32ActivationKind::Relu => Activation::Relu,
        F32ActivationKind::Leaky => Activation::Leaky,
        F32ActivationKind::Mish => Activation::Mish,
        F32ActivationKind::Swish => Activation::Swish,
        F32ActivationKind::Tanh => Activation::Tanh,
        F32ActivationKind::Relu6 => Activation::Relu6,
        F32ActivationKind::Ramp => Activation::Ramp,
        F32ActivationKind::Gelu => Activation::Gelu,
        F32ActivationKind::Elu => Activation::Elu,
        F32ActivationKind::Selu => Activation::Selu,
        F32ActivationKind::HardMish => Activation::HardMish,
    };
    super::activate(activation, value)
}

pub(super) fn fake_activation_gradient(activation: F32ActivationKind, value: f32) -> f32 {
    match activation {
        F32ActivationKind::Linear => 1.0,
        F32ActivationKind::Loggy => crate::activation_math::loggy_grad(value),
        F32ActivationKind::Hardtan => crate::activation_math::hardtan_grad(value),
        F32ActivationKind::Relie => {
            if value > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        F32ActivationKind::Lhtan => crate::activation_math::lhtan_grad(value),
        F32ActivationKind::Plse => crate::activation_math::plse_grad(value),
        F32ActivationKind::Stair => crate::activation_math::stair_grad(value),
        F32ActivationKind::Logistic => {
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            sigmoid * (1.0 - sigmoid)
        }
        F32ActivationKind::Relu => (value > 0.0) as i32 as f32,
        F32ActivationKind::Leaky => {
            if value > 0.0 {
                1.0
            } else {
                0.1
            }
        }
        F32ActivationKind::Mish => {
            const MISH_THRESHOLD: f32 = 20.0;
            let softplus = if value > MISH_THRESHOLD {
                value
            } else if value < -MISH_THRESHOLD {
                value.exp()
            } else {
                (1.0 + value.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tsp = softplus.tanh();
            let grad_tsp = (1.0 - tsp * tsp) * grad_sp;
            value * grad_tsp + tsp
        }
        F32ActivationKind::Swish => {
            let sigmoid = 1.0 / (1.0 + (-value).exp());
            sigmoid + value * sigmoid * (1.0 - sigmoid)
        }
        F32ActivationKind::Tanh => {
            let tanh = value.tanh();
            1.0 - tanh * tanh
        }
        F32ActivationKind::Relu6 => (value > 0.0 && value < 6.0) as i32 as f32,
        F32ActivationKind::Ramp => (value > 0.0) as i32 as f32 + 0.1,
        F32ActivationKind::Gelu => crate::activation_math::gelu_grad(value),
        F32ActivationKind::Elu => crate::activation_math::elu_grad(value),
        F32ActivationKind::Selu => crate::activation_math::selu_grad(value),
        F32ActivationKind::HardMish => crate::activation_math::hard_mish_grad(value),
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct CountingBackend {
    pub(super) uploads: Arc<AtomicUsize>,
    pub(super) inner: FakeBackend,
}

impl CountingBackend {
    pub(super) fn upload_count(&self) -> usize {
        self.uploads.load(Ordering::Relaxed)
    }
}

impl F32TensorOpsBackend for CountingBackend {
    type Error = String;
    type Tensor = FakeTensor;

    fn upload_f32(
        &self,
        shape: &[usize],
        host: &[f32],
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        self.inner.upload_f32(shape, host)
    }

    fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.zeros_f32(shape)
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> std::result::Result<Vec<f32>, Self::Error> {
        self.inner.download_f32(tensor)
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner
            .bias_add_f32(x, bias, channels, channel_stride, config)
    }

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner
            .channel_scale_f32(x, scale, channels, channel_stride, config)
    }

    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner
            .bias_backward_f32(output_grad, channels, channel_stride, config)
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.activation_f32(x, activation, config)
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner
            .activation_backward_f32(activation_input, delta, activation, config)
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.shortcut_add_f32(x, y, config)
    }

    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.route_concat_f32(
            inputs,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
            config,
        )
    }

    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
        self.inner.route_backward_f32(
            output_grad,
            input_shapes,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            channel_stride,
            config,
        )
    }

    fn conv2d_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.conv2d_f32(
            x,
            weights,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

    fn conv2d_backward_input_f32(
        &self,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        input_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.conv2d_backward_input_f32(
            output_grad,
            weights,
            input_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

    fn conv2d_backward_weights_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        weight_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.conv2d_backward_weights_f32(
            x,
            output_grad,
            weight_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

    fn maxpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.maxpool_f32(
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn maxpool_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.maxpool_backward_f32(
            x,
            output_grad,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn local_avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.local_avgpool_f32(
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn local_avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.local_avgpool_backward_f32(
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner
            .avgpool_f32(x, output_shape, channels, input_height, input_width, config)
    }

    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.avgpool_backward_f32(
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            config,
        )
    }

    fn upsample_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.upsample_f32(
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            scale,
            reverse,
            config,
        )
    }

    fn upsample_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.upsample_backward_f32(
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            config,
        )
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.vector_add_f32(x, y, config)
    }

    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.inner.saxpy_f32(alpha, x, y, config)
    }

    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
        self.inner
            .sgd_update_f32(weights, updates, lr, momentum, decay, batch, config)
    }
}

pub(super) fn assert_same_forward(cpu: &ForwardResult, gpu: &ForwardResult) {
    assert_eq!(cpu.layer_outputs.len(), gpu.layer_outputs.len());
    assert_eq!(cpu.yolo_heads.len(), gpu.yolo_heads.len());
    for (lhs, rhs) in cpu.layer_outputs.iter().zip(&gpu.layer_outputs) {
        assert_eq!(lhs.shape, rhs.shape);
        assert_eq!(lhs.data, rhs.data);
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ShapeOnlyBackend;

impl F32TensorOpsBackend for ShapeOnlyBackend {
    type Error = String;
    type Tensor = FakeTensor;

    fn upload_f32(
        &self,
        shape: &[usize],
        host: &[f32],
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        let expected = shape.iter().product::<usize>();
        if host.len() != expected {
            return Err(format!(
                "upload_f32 expected {expected} elements for shape {:?}, got {}",
                shape,
                host.len()
            ));
        }
        Ok(FakeTensor {
            shape: shape.to_vec(),
            data: host.to_vec(),
        })
    }

    fn zeros_f32(&self, shape: &[usize]) -> std::result::Result<Self::Tensor, Self::Error> {
        Ok(FakeTensor {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        })
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> std::result::Result<Vec<f32>, Self::Error> {
        Ok(tensor.data.clone())
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        _bias: &Self::Tensor,
        _channels: usize,
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        Ok(x.clone())
    }

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        _channels: usize,
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if scale.shape.len() != 1 {
            return Err("channel_scale_f32 expects flat scale tensor".to_owned());
        }
        Ok(x.clone())
    }

    fn bias_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        channels: usize,
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(&[channels])
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        _activation: F32ActivationKind,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        Ok(x.clone())
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        _activation: F32ActivationKind,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if activation_input.data.len() != delta.data.len() {
            return Err("activation_backward_f32 shape mismatch".to_owned());
        }
        Ok(delta.clone())
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("shortcut_add_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: vec![0.0; x.data.len()],
        })
    }

    fn route_concat_f32(
        &self,
        _inputs: &[&Self::Tensor],
        _input_channel_offsets: &[usize],
        _input_channels: &[usize],
        _input_total_channels: &[usize],
        output_shape: &[usize],
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn route_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        _input_channel_offsets: &[usize],
        _input_channels: &[usize],
        _input_total_channels: &[usize],
        _channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Vec<Self::Tensor>, Self::Error> {
        input_shapes
            .iter()
            .map(|shape| self.zeros_f32(shape))
            .collect()
    }

    fn conv2d_f32(
        &self,
        _x: &Self::Tensor,
        _weights: &Self::Tensor,
        output_shape: &[usize],
        _input_channels: usize,
        _input_height: usize,
        _input_width: usize,
        _filters: usize,
        _kernel_size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _dilation: usize,
        _padding: usize,
        _groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn conv2d_backward_input_f32(
        &self,
        _output_grad: &Self::Tensor,
        _weights: &Self::Tensor,
        input_shape: &[usize],
        _input_channels: usize,
        _input_height: usize,
        _input_width: usize,
        _filters: usize,
        _kernel_size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _dilation: usize,
        _padding: usize,
        _groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(input_shape)
    }

    fn conv2d_backward_weights_f32(
        &self,
        _x: &Self::Tensor,
        _output_grad: &Self::Tensor,
        weight_shape: &[usize],
        _input_channels: usize,
        _input_height: usize,
        _input_width: usize,
        _filters: usize,
        _kernel_size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _dilation: usize,
        _padding: usize,
        _groups: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(weight_shape)
    }

    fn maxpool_f32(
        &self,
        _x: &Self::Tensor,
        output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn maxpool_backward_f32(
        &self,
        x: &Self::Tensor,
        _output_grad: &Self::Tensor,
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(&x.shape)
    }

    fn local_avgpool_f32(
        &self,
        _x: &Self::Tensor,
        output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn local_avgpool_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        input_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _size: usize,
        _stride_x: usize,
        _stride_y: usize,
        _padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(input_shape)
    }

    fn avgpool_f32(
        &self,
        _x: &Self::Tensor,
        output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn avgpool_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        input_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(input_shape)
    }

    fn upsample_f32(
        &self,
        _x: &Self::Tensor,
        output_shape: &[usize],
        _channels: usize,
        _input_height: usize,
        _input_width: usize,
        _stride: usize,
        _scale: f32,
        _reverse: bool,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(output_shape)
    }

    fn upsample_backward_f32(
        &self,
        _output_grad: &Self::Tensor,
        input_shape: &[usize],
        _channels: usize,
        _output_height: usize,
        _output_width: usize,
        _stride: usize,
        _scale: f32,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        self.zeros_f32(input_shape)
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("vector_add_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: vec![0.0; x.data.len()],
        })
    }

    fn saxpy_f32(
        &self,
        _alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<Self::Tensor, Self::Error> {
        if x.shape != y.shape {
            return Err("saxpy_f32 shape mismatch".to_owned());
        }
        Ok(FakeTensor {
            shape: x.shape.clone(),
            data: vec![0.0; x.data.len()],
        })
    }

    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        _lr: f32,
        _momentum: f32,
        _decay: f32,
        _batch: usize,
        _config: ElementwiseLaunchConfig,
    ) -> std::result::Result<(Self::Tensor, Self::Tensor), Self::Error> {
        if weights.shape != updates.shape {
            return Err("sgd_update_f32 shape mismatch".to_owned());
        }
        Ok((weights.clone(), updates.clone()))
    }
}

pub(super) fn squared_error_loss(
    executable: &ExecutableNetwork,
    input: &Tensor,
    target: &[f32],
) -> f32 {
    let forward = executable
        .forward(input)
        .expect("loss evaluation forward should succeed");
    let output = forward
        .final_output()
        .expect("loss evaluation needs a final output");
    assert_eq!(output.data.len(), target.len());
    output
        .data
        .iter()
        .zip(target)
        .map(|(prediction, target)| {
            let error = prediction - target;
            0.5 * error * error
        })
        .sum()
}

pub(super) fn assert_slice_close(name: &str, expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), actual.len(), "{name} length mismatch");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            (expected - actual).abs() <= tolerance,
            "{name}[{index}] differs: expected {expected:.8e}, actual {actual:.8e}, tolerance {tolerance:.8e}"
        );
    }
}

/// Independent analytical reference for a Darknet NCHW cross-correlation.
///
/// The backward equations used here are closed-form chain-rule equations:
///
/// ```text
/// dL/dw[f,c,ky,kx] += x[b,c,iy,ix] * dL/dy[b,f,oy,ox]
/// dL/db[f]          += dL/dy[b,f,oy,ox]
/// dL/dx[b,c,iy,ix]  += w[f,c,ky,kx] * dL/dy[b,f,oy,ox]
/// ```
///
/// Keeping this oracle separate from the runtime implementation makes the
/// test catch indexing, padding, grouping, and gradient-sign mistakes.
pub(super) struct AnalyticalConvCase<'a> {
    pub(super) batch: usize,
    pub(super) channels: usize,
    pub(super) height: usize,
    pub(super) width: usize,
    pub(super) filters: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) groups: usize,
    pub(super) input: &'a [f32],
    pub(super) weights: &'a [f32],
    pub(super) biases: &'a [f32],
}

pub(super) fn analytical_convolution_output(case: &AnalyticalConvCase<'_>) -> Vec<f32> {
    let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
    let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
    let channels_per_group = case.channels / case.groups;
    let filters_per_group = case.filters / case.groups;
    let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
        (((batch * case.channels + channel) * case.height) + y) * case.width + x
    };
    let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
        (((batch * case.filters + filter) * output_height) + y) * output_width + x
    };
    let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
        (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
    };
    let mut output = vec![0.0; case.batch * case.filters * output_height * output_width];

    for batch in 0..case.batch {
        for filter in 0..case.filters {
            let group = filter / filters_per_group;
            for y in 0..output_height {
                for x in 0..output_width {
                    let mut value = case.biases[filter];
                    for channel_in_group in 0..channels_per_group {
                        let channel = group * channels_per_group + channel_in_group;
                        for ky in 0..case.kernel {
                            for kx in 0..case.kernel {
                                let source_y = y * case.stride + ky;
                                let source_x = x * case.stride + kx;
                                if source_y < case.padding
                                    || source_x < case.padding
                                    || source_y >= case.height + case.padding
                                    || source_x >= case.width + case.padding
                                {
                                    continue;
                                }
                                let source_y = source_y - case.padding;
                                let source_x = source_x - case.padding;
                                value += case.input
                                    [input_index(batch, channel, source_y, source_x)]
                                    * case.weights[weight_index(filter, channel_in_group, ky, kx)];
                            }
                        }
                    }
                    output[output_index(batch, filter, y, x)] = value;
                }
            }
        }
    }
    output
}

pub(super) fn analytical_convolution_gradients(
    case: &AnalyticalConvCase<'_>,
    output_gradient: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
    let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
    let channels_per_group = case.channels / case.groups;
    let filters_per_group = case.filters / case.groups;
    let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
        (((batch * case.channels + channel) * case.height) + y) * case.width + x
    };
    let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
        (((batch * case.filters + filter) * output_height) + y) * output_width + x
    };
    let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
        (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
    };
    let mut input_gradient = vec![0.0; case.input.len()];
    let mut weight_gradient = vec![0.0; case.weights.len()];
    let mut bias_gradient = vec![0.0; case.biases.len()];

    assert_eq!(
        output_gradient.len(),
        case.batch * case.filters * output_height * output_width
    );
    for batch in 0..case.batch {
        for filter in 0..case.filters {
            let group = filter / filters_per_group;
            for y in 0..output_height {
                for x in 0..output_width {
                    let gradient = output_gradient[output_index(batch, filter, y, x)];
                    bias_gradient[filter] += gradient;
                    for channel_in_group in 0..channels_per_group {
                        let channel = group * channels_per_group + channel_in_group;
                        for ky in 0..case.kernel {
                            for kx in 0..case.kernel {
                                let source_y = y * case.stride + ky;
                                let source_x = x * case.stride + kx;
                                if source_y < case.padding
                                    || source_x < case.padding
                                    || source_y >= case.height + case.padding
                                    || source_x >= case.width + case.padding
                                {
                                    continue;
                                }
                                let source_y = source_y - case.padding;
                                let source_x = source_x - case.padding;
                                let input_offset = input_index(batch, channel, source_y, source_x);
                                let weight_offset = weight_index(filter, channel_in_group, ky, kx);
                                weight_gradient[weight_offset] +=
                                    case.input[input_offset] * gradient;
                                input_gradient[input_offset] +=
                                    case.weights[weight_offset] * gradient;
                            }
                        }
                    }
                }
            }
        }
    }
    (input_gradient, weight_gradient, bias_gradient)
}
