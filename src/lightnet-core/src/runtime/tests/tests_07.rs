use super::*;

#[test]
fn optimizer_state_round_trips_adam_moments() {
    let dir = std::env::temp_dir().join(format!("darknet-opt-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("opt.state");

    let step = |gpu: &mut GpuExecutableNetwork<FakeBackend>, s: usize| {
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu.forward_training_with_state(&input).expect("forward");
        let seeds = optimizer_seeds(gpu, s);
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("backward");
        gpu.update_weights_gpu(0.01, 0.9, 0.0, 2).expect("update");
    };

    let mut original = optimizer_net("adam=1\nloss_scale=1\n");
    step(&mut original, 1);
    step(&mut original, 2);
    let weights_path = dir.join("model.weights");
    crate::weights::save_weights(&original.network, &original.params, &weights_path, 2)
        .expect("save weights");
    original.save_optimizer_state(&path).expect("save");
    assert_eq!(original.adam_step(), 2);

    // Resume into a fresh network: restore parameters from the weights file
    // and moments/step from the sidecar.
    let mut resumed = optimizer_net("adam=1\nloss_scale=1\n");
    resumed
        .load_weights_file(&weights_path)
        .expect("load weights");
    resumed.load_optimizer_state(&path).expect("load");
    assert_eq!(resumed.adam_step(), 2);
    let path2 = dir.join("opt2.state");
    resumed.save_optimizer_state(&path2).expect("resave");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&path2).unwrap(),
        "optimizer state round-trip changed bytes"
    );

    // The next step must match, which only holds if the moments were restored.
    step(&mut original, 3);
    step(&mut resumed, 3);
    for layer in 0..2 {
        let (w0, b0, s0) = host_params(&original, layer);
        let (w1, b1, s1) = host_params(&resumed, layer);
        assert_slice_close("weights", &w0, &w1, 1e-6);
        assert_slice_close("biases", &b0, &b1, 1e-6);
        assert_slice_close("scales", &s0, &s1, 1e-6);
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn eml_adam_migrates_restored_host_moments_from_dnopt3() {
    let dir = std::env::temp_dir().join(format!("darknet-eml-migrate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("legacy-eml-opt.state");
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=1\nadam=1\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\n";
    let build = || {
        let network =
            Network::from_config(ConfigFile::parse_str("eml-migrate.cfg", cfg).expect("EML cfg"))
                .expect("EML network");
        GpuExecutableNetwork::new(network, FakeBackend)
    };
    let host_branch = |weight_m: f32, bias_m: f32| HostAdamLayerState {
        weights: Some((vec![weight_m], vec![weight_m + 1.0])),
        biases: Some((vec![bias_m], vec![bias_m + 1.0])),
        scales: Some((vec![weight_m + 2.0], vec![weight_m + 3.0])),
        ..HostAdamLayerState::default()
    };
    let original = build();
    original.host_adam_state.borrow_mut()[0] = Some(HostAdamLayerState {
        children: vec![host_branch(0.1, 0.2), host_branch(0.3, 0.4)],
        ..HostAdamLayerState::default()
    });
    original
        .save_optimizer_state(&path)
        .expect("save legacy sidecar");
    assert_eq!(&std::fs::read(&path).unwrap()[..8], b"DNOPT3\0\0");

    let mut restored = build();
    restored
        .load_optimizer_state(&path)
        .expect("load legacy EML moments");
    let state = restored
        .eml_adam_state_for_update(0)
        .expect("migrate host EML moments");
    for (branch, expected_weight_m, expected_bias_m) in
        [(&state.input, 0.1, 0.2), (&state.self_params, 0.3, 0.4)]
    {
        let (weight_m, weight_v) = branch.weights.as_ref().expect("weight moments");
        assert_eq!(
            restored.backend.download_f32(weight_m).unwrap(),
            [expected_weight_m]
        );
        assert_eq!(
            restored.backend.download_f32(weight_v).unwrap(),
            [expected_weight_m + 1.0]
        );
        let (bias_m, bias_v) = branch.biases.as_ref().expect("bias moments");
        assert_eq!(
            restored.backend.download_f32(bias_m).unwrap(),
            [expected_bias_m]
        );
        assert_eq!(
            restored.backend.download_f32(bias_v).unwrap(),
            [expected_bias_m + 1.0]
        );
        let (scale_m, scale_v) = branch.scales.as_ref().expect("scale moments");
        assert_eq!(
            restored.backend.download_f32(scale_m).unwrap(),
            [expected_weight_m + 2.0]
        );
        assert_eq!(
            restored.backend.download_f32(scale_v).unwrap(),
            [expected_weight_m + 3.0]
        );
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn optimizer_state_round_trips_eml_adam_moments_dnopt5() {
    let dir = std::env::temp_dir().join(format!("darknet-eml-opt-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("eml-opt.state");
    let path2 = dir.join("eml-opt2.state");
    let cfg = "[net]\nchannels=1\nbatch=2\nheight=1\nwidth=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\nresidual=0\n";
    let build = || {
        let network =
            Network::from_config(ConfigFile::parse_str("eml-opt-state.cfg", cfg).expect("EML cfg"))
                .expect("EML network");
        GpuExecutableNetwork::new(network, FakeBackend)
    };
    let step = |gpu: &mut GpuExecutableNetwork<FakeBackend>| {
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu
            .forward_training_with_state(&input)
            .expect("EML Adam training forward");
        let output = forward.cpu_forward.final_output().unwrap();
        let seeds = vec![Some(Tensor::filled(output.shape, 0.7))];
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("EML Adam backward");
        gpu.update_weights_gpu(0.01, 0.0, 0.0, 2)
            .expect("EML Adam update");
    };

    let mut original = build();
    step(&mut original);
    assert!(
        original.eml_adam_state[0]
            .as_ref()
            .unwrap()
            .input
            .scales
            .is_some()
    );
    assert!(
        original.eml_adam_state[0]
            .as_ref()
            .unwrap()
            .self_params
            .scales
            .is_some()
    );
    original.save_optimizer_state(&path).expect("save DNOPT5");
    assert_eq!(&std::fs::read(&path).unwrap()[..8], b"DNOPT5\0\0");

    let mut resumed = build();
    resumed.params = original.params.clone();
    resumed.load_optimizer_state(&path).expect("load DNOPT5");
    assert_eq!(resumed.adam_step(), 1);
    resumed.save_optimizer_state(&path2).expect("resave DNOPT5");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&path2).unwrap(),
        "EML optimizer sidecar round-trip changed bytes"
    );

    step(&mut original);
    step(&mut resumed);
    for (expected, actual) in original.params.iter().zip(&resumed.params) {
        let (LayerParams::EmlConvolutional(expected), LayerParams::EmlConvolutional(actual)) =
            (expected, actual)
        else {
            unreachable!()
        };
        for (expected, actual) in [
            (&expected.input, &actual.input),
            (&expected.self_params, &actual.self_params),
        ] {
            assert_slice_close(
                "EML resumed Adam weights",
                &expected.weights,
                &actual.weights,
                1e-6,
            );
            assert_slice_close(
                "EML resumed Adam biases",
                &expected.biases,
                &actual.biases,
                1e-6,
            );
            assert_slice_close(
                "EML resumed Adam BN scales",
                &expected.batch_norm.as_ref().unwrap().scales,
                &actual.batch_norm.as_ref().unwrap().scales,
                1e-6,
            );
        }
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn optimizer_state_round_trips_host_adam_moments_and_reads_v1() {
    let dir = std::env::temp_dir().join(format!("darknet-host-opt-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("host-opt.state");
    let cfg = "[net]\ninputs=2\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[connected]\noutput=2\nactivation=linear\n";
    let build = || {
        let network =
            Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
        GpuExecutableNetwork::new(network, FakeBackend)
    };
    let set_grads = |gpu: &mut GpuExecutableNetwork<FakeBackend>,
                     weights: [f32; 4],
                     biases: [f32; 2]| {
        let LayerTrainBuffers::Connected(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0]
        else {
            panic!("connected train buffers");
        };
        train.weight_updates.copy_from_slice(&weights);
        train.bias_updates.copy_from_slice(&biases);
    };
    let mut original = build();
    set_grads(&mut original, [0.1, -0.2, 0.05, -0.1], [0.05, -0.1]);
    original
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("first Adam step");
    original
        .save_optimizer_state(&path)
        .expect("save host optimizer state");
    let sidecar = std::fs::read(&path).unwrap();
    assert_eq!(&sidecar[..8], b"DNOPT3\0\0");
    let legacy_v2_path = dir.join("legacy-v2.state");
    let mut legacy_v2 = sidecar.clone();
    legacy_v2[..8].copy_from_slice(b"DNOPT2\0\0");
    legacy_v2.truncate(legacy_v2.len() - 8);
    std::fs::write(&legacy_v2_path, legacy_v2).unwrap();
    let mut resumed_v2 = build();
    resumed_v2.params.clone_from(&original.params);
    resumed_v2
        .load_optimizer_state(&legacy_v2_path)
        .expect("restore DNOPT2 host moments");
    let v2_round_trip_path = dir.join("v2-round-trip.state");
    resumed_v2
        .save_optimizer_state(&v2_round_trip_path)
        .expect("resave DNOPT2 state as DNOPT3");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&v2_round_trip_path).unwrap(),
        "DNOPT2 state changed during DNOPT3 migration"
    );

    let mut resumed = build();
    resumed.params.clone_from(&original.params);
    resumed
        .load_optimizer_state(&path)
        .expect("restore host moments");
    assert_eq!(resumed.adam_step(), 1);
    set_grads(&mut original, [0.3, 0.05, -0.2, 0.4], [-0.2, 0.4]);
    set_grads(&mut resumed, [0.3, 0.05, -0.2, 0.4], [-0.2, 0.4]);
    original
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("continued Adam step");
    resumed
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("resumed Adam step");
    let (LayerParams::Connected(original_params), LayerParams::Connected(resumed_params)) =
        (&original.params[0], &resumed.params[0])
    else {
        unreachable!()
    };
    assert_slice_close(
        "resumed host Adam weights",
        &original_params.weights,
        &resumed_params.weights,
        1e-7,
    );
    assert_slice_close(
        "resumed host Adam biases",
        &original_params.biases,
        &resumed_params.biases,
        1e-7,
    );

    let legacy_path = dir.join("legacy-v1.state");
    let mut legacy = b"DNOPT1\0\0".to_vec();
    legacy.extend_from_slice(&7u32.to_le_bytes());
    legacy.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(&legacy_path, legacy).unwrap();
    let mut legacy_network = build();
    legacy_network
        .load_optimizer_state(&legacy_path)
        .expect("load legacy v1 state");
    assert_eq!(legacy_network.adam_step(), 7);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn optimizer_state_migrates_and_round_trips_batchnorm_moe_adam_moments() {
    let dir = std::env::temp_dir().join(format!("darknet-moe-bn-opt-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let host_path = dir.join("host-v3.state");
    let device_path = dir.join("device-v4.state");
    let resumed_path = dir.join("resumed-v4.state");
    let cfg = "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[moe_convolutional]\nfilters=1\nsize=1\nexperts=2\ntop_k=0\nload_balance=0\nactivation=linear\n";
    let build = || {
        let network =
            Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        let LayerParams::Moe(params) = &mut gpu.params[0] else {
            panic!("MoE parameters");
        };
        params.router_weights.copy_from_slice(&[0.2, -0.4]);
        params.router_biases.copy_from_slice(&[0.1, -0.2]);
        for (index, expert) in params.experts.iter_mut().enumerate() {
            expert.weights[0] = 0.15 * (index as f32 + 1.0);
            expert.biases[0] = -0.05 * (index as f32 + 1.0);
            expert.batch_norm.as_mut().expect("MoE BN params").scales[0] = 0.7 + index as f32 * 0.2;
        }
        gpu
    };
    let host_grads = |gpu: &mut GpuExecutableNetwork<FakeBackend>, step: f32| {
        let LayerTrainBuffers::Moe(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("host MoE train buffers");
        };
        train
            .router_weight_updates
            .copy_from_slice(&[0.03 * step, -0.02 * step]);
        train
            .router_bias_updates
            .copy_from_slice(&[-0.01 * step, 0.04 * step]);
        for (index, expert) in train.experts.iter_mut().enumerate() {
            expert.weight_updates[0] = 0.02 * step * (index as f32 + 1.0);
            expert.bias_updates[0] = -0.03 * step * (index as f32 + 1.0);
            expert
                .batch_norm
                .as_mut()
                .expect("host MoE BN train buffers")
                .scale_updates[0] = 0.04 * step * (index as f32 + 1.0);
        }
    };
    let device_grads = |gpu: &mut GpuExecutableNetwork<FakeBackend>, step: f32| {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("device MoE train buffers");
        train
            .router_weight_updates
            .data
            .copy_from_slice(&[0.03 * step, -0.02 * step]);
        train
            .router_bias_updates
            .data
            .copy_from_slice(&[-0.01 * step, 0.04 * step]);
        for index in 0..2 {
            train.expert_weight_updates[index].data[0] = 0.02 * step * (index as f32 + 1.0);
            train.expert_bias_updates[index].data[0] = -0.03 * step * (index as f32 + 1.0);
            train.expert_scale_updates[index]
                .as_mut()
                .expect("device MoE scale updates")
                .data[0] = 0.04 * step * (index as f32 + 1.0);
        }
        train.dirty = true;
    };
    let assert_params_close =
        |a: &GpuExecutableNetwork<FakeBackend>, b: &GpuExecutableNetwork<FakeBackend>| {
            let (LayerParams::Moe(a), LayerParams::Moe(b)) = (&a.params[0], &b.params[0]) else {
                panic!("MoE parameters");
            };
            assert_slice_close(
                "MoE router weights",
                &a.router_weights,
                &b.router_weights,
                2e-6,
            );
            assert_slice_close(
                "MoE router biases",
                &a.router_biases,
                &b.router_biases,
                2e-6,
            );
            for (a, b) in a.experts.iter().zip(&b.experts) {
                assert_slice_close("MoE expert weights", &a.weights, &b.weights, 2e-6);
                assert_slice_close("MoE expert biases", &a.biases, &b.biases, 2e-6);
                assert_slice_close(
                    "MoE BN scales",
                    &a.batch_norm.as_ref().unwrap().scales,
                    &b.batch_norm.as_ref().unwrap().scales,
                    2e-6,
                );
            }
        };

    // Save legacy-format nested host moments, including each BN scale slot.
    let mut host = build();
    host_grads(&mut host, 1.0);
    host.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("first host BN MoE Adam step");
    host.save_optimizer_state(&host_path)
        .expect("save host BN MoE moments");
    assert_eq!(&std::fs::read(&host_path).unwrap()[..8], b"DNOPT3\0\0");

    let mut host_reference = build();
    host_reference.params.clone_from(&host.params);
    host_reference
        .load_optimizer_state(&host_path)
        .expect("restore host BN MoE moments");
    host_grads(&mut host_reference, 2.0);
    host_reference
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("second host BN MoE Adam step");

    let mut device = build();
    device.params.clone_from(&host.params);
    device.init_train_buffers_gpu().expect("device BN buffers");
    device
        .load_optimizer_state(&host_path)
        .expect("load legacy host moments for BN device migration");
    device_grads(&mut device, 2.0);
    device
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("BN MoE device Adam after migration");
    assert_params_close(&device, &host_reference);
    assert!(device.host_adam_state.borrow()[0].is_none());
    assert!(
        device.moe_adam_state[0]
            .as_ref()
            .unwrap()
            .experts
            .iter()
            .all(|expert| expert.scales.is_some())
    );
    device
        .save_optimizer_state(&device_path)
        .expect("save DNOPT4 BN moments");
    let saved = std::fs::read(&device_path).unwrap();
    assert_eq!(&saved[..8], b"DNOPT4\0\0");

    let mut resumed = build();
    resumed.params.clone_from(&device.params);
    resumed
        .init_train_buffers_gpu()
        .expect("resumed device buffers");
    resumed
        .load_optimizer_state(&device_path)
        .expect("restore DNOPT4 scale moments");
    resumed
        .save_optimizer_state(&resumed_path)
        .expect("resave DNOPT4 moments");
    assert_eq!(saved, std::fs::read(&resumed_path).unwrap());
    device_grads(&mut device, 3.0);
    device_grads(&mut resumed, 3.0);
    device
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("continued BN MoE device Adam");
    resumed
        .update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("resumed BN MoE device Adam");
    assert_params_close(&device, &resumed);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn optimizer_state_round_trips_nested_moe_moments() {
    let dir = std::env::temp_dir().join(format!("darknet-moe-opt-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("moe-opt.state");
    let cfg = "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n[moe_convolutional]\nfilters=1\nsize=1\nexperts=2\ntop_k=0\nload_balance=0\nbatch_normalize=0\nactivation=linear\n";
    let build = || {
        let network =
            Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
        GpuExecutableNetwork::new(network, FakeBackend)
    };
    let set_grads = |gpu: &mut GpuExecutableNetwork<FakeBackend>, step: usize| {
        let values = if step == 1 {
            ([0.1, -0.2], [-0.3, 0.4], [0.05, -0.08])
        } else {
            ([-0.25, 0.15], [0.2, -0.1], [-0.12, 0.07])
        };
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("GPU MoE device train buffers");
        train.router_weight_updates.data.copy_from_slice(&values.0);
        train.router_bias_updates.data.copy_from_slice(&values.1);
        for (index, (weights, biases)) in train
            .expert_weight_updates
            .iter_mut()
            .zip(&mut train.expert_bias_updates)
            .enumerate()
        {
            weights.data[0] = values.2[0] * (index as f32 + 1.0);
            biases.data[0] = values.2[1] * (index as f32 + 1.0);
        }
        train.dirty = true;
    };
    let set_host_grads = |gpu: &mut GpuExecutableNetwork<FakeBackend>, step: usize| {
        let values = if step == 1 {
            ([0.1, -0.2], [-0.3, 0.4], [0.05, -0.08])
        } else {
            ([-0.25, 0.15], [0.2, -0.1], [-0.12, 0.07])
        };
        let LayerTrainBuffers::Moe(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("host-fallback MoE train buffers");
        };
        train.router_weight_updates.copy_from_slice(&values.0);
        train.router_bias_updates.copy_from_slice(&values.1);
        for (index, expert) in train.experts.iter_mut().enumerate() {
            expert.weight_updates[0] = values.2[0] * (index as f32 + 1.0);
            expert.bias_updates[0] = values.2[1] * (index as f32 + 1.0);
        }
    };
    let assert_moe_params_close = |a: &LayerParams, b: &LayerParams, tolerance: f32| {
        let (LayerParams::Moe(a), LayerParams::Moe(b)) = (a, b) else {
            panic!("MoE params expected for Adam parity");
        };
        assert_slice_close(
            "MoE Adam router weights",
            &a.router_weights,
            &b.router_weights,
            tolerance,
        );
        assert_slice_close(
            "MoE Adam router biases",
            &a.router_biases,
            &b.router_biases,
            tolerance,
        );
        for (expert_a, expert_b) in a.experts.iter().zip(&b.experts) {
            assert_slice_close(
                "MoE Adam expert weights",
                &expert_a.weights,
                &expert_b.weights,
                tolerance,
            );
            assert_slice_close(
                "MoE Adam expert biases",
                &expert_a.biases,
                &expert_b.biases,
                tolerance,
            );
        }
    };
    let mut original = build();
    let LayerParams::Moe(params) = &mut original.params[0] else {
        unreachable!()
    };
    params.router_weights.copy_from_slice(&[0.2, -0.4]);
    params.router_biases.copy_from_slice(&[0.1, -0.2]);
    for (index, expert) in params.experts.iter_mut().enumerate() {
        expert.weights[0] = 0.15 * (index as f32 + 1.0);
        expert.biases[0] = -0.05 * (index as f32 + 1.0);
    }
    original.init_train_buffers_gpu().expect("device buffers");
    assert!(original.device_moe_train_buffers.borrow()[0].is_some());
    let mut host_reference = build();
    host_reference.params.clone_from(&original.params);
    set_grads(&mut original, 1);
    set_host_grads(&mut host_reference, 1);
    original
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("first device MoE Adam step");
    host_reference
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("first host MoE Adam reference step");
    assert_moe_params_close(&original.params[0], &host_reference.params[0], 2e-6);
    let device_state = original.moe_adam_state[0]
        .as_ref()
        .expect("device MoE Adam state");
    let host_states = host_reference.host_adam_state.borrow();
    let host_state = host_states[0].as_ref().expect("host MoE Adam state");
    let compare_moments = |name: &str,
                           device: &Option<(FakeTensor, FakeTensor)>,
                           host: &Option<(Vec<f32>, Vec<f32>)>| {
        let (Some((device_first, device_second)), Some((host_first, host_second))) =
            (device.as_ref(), host.as_ref())
        else {
            panic!("{name} Adam moments missing on a path");
        };
        assert_slice_close(name, &device_first.data, host_first, 2e-6);
        assert_slice_close(name, &device_second.data, host_second, 2e-6);
    };
    compare_moments(
        "router weight moment",
        &device_state.router.weights,
        &host_state.weights,
    );
    compare_moments(
        "router bias moment",
        &device_state.router.biases,
        &host_state.biases,
    );
    for (index, device_expert) in device_state.experts.iter().enumerate() {
        let host_expert = &host_state.children[index];
        compare_moments(
            "expert weight moment",
            &device_expert.weights,
            &host_expert.weights,
        );
        compare_moments(
            "expert bias moment",
            &device_expert.biases,
            &host_expert.biases,
        );
    }
    drop(host_states);
    let legacy_v3_path = dir.join("moe-opt-host-v3.state");
    host_reference
        .save_optimizer_state(&legacy_v3_path)
        .expect("save legacy host-only MoE moments");
    let mut legacy_bytes = std::fs::read(&legacy_v3_path).unwrap();
    assert_eq!(&legacy_bytes[..8], b"DNOPT3\0\0");
    assert_eq!(&legacy_bytes[legacy_bytes.len() - 8..], &[0; 8]);
    legacy_bytes.truncate(legacy_bytes.len() - 8);
    legacy_bytes[..8].copy_from_slice(b"DNOPT2\0\0");
    let legacy_path = dir.join("moe-opt-legacy-v2.state");
    std::fs::write(&legacy_path, legacy_bytes).unwrap();

    original
        .save_optimizer_state(&path)
        .expect("save nested moments");
    assert_eq!(&std::fs::read(&path).unwrap()[..8], b"DNOPT3\0\0");

    let mut migrated = build();
    migrated.params.clone_from(&host_reference.params);
    migrated
        .init_train_buffers_gpu()
        .expect("legacy migrated device buffers");
    migrated
        .load_optimizer_state(&legacy_path)
        .expect("load legacy host MoE moments for device migration");
    let mut resumed = build();
    resumed.params.clone_from(&original.params);
    resumed
        .init_train_buffers_gpu()
        .expect("resumed device buffers");
    assert!(resumed.device_moe_train_buffers.borrow()[0].is_some());
    resumed
        .load_optimizer_state(&path)
        .expect("restore nested moments");
    let path2 = dir.join("moe-opt-resaved.state");
    resumed
        .save_optimizer_state(&path2)
        .expect("resave nested moments");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&path2).unwrap(),
        "device MoE Adam moments changed during sidecar round-trip"
    );
    set_grads(&mut original, 2);
    set_grads(&mut resumed, 2);
    set_grads(&mut migrated, 2);
    set_host_grads(&mut host_reference, 2);
    let device_buffers = original.device_moe_train_buffers.borrow();
    let device_train = device_buffers[0]
        .as_ref()
        .expect("second-step device buffers");
    let host_buffers = host_reference.cpu_fallback_train_buffers.borrow();
    let LayerTrainBuffers::Moe(host_train) = &host_buffers[0] else {
        panic!("second-step host fallback buffers");
    };
    assert_slice_close(
        "second-step expert gradients",
        &device_train.expert_weight_updates[1].data,
        &host_train.experts[1].weight_updates,
        1e-7,
    );
    let device_state_before = original.moe_adam_state[0]
        .as_ref()
        .expect("device MoE state before second step");
    let host_state = host_reference.host_adam_state.borrow();
    let host_moment = host_state[0]
        .as_ref()
        .expect("host Adam state before second step")
        .children[1]
        .weights
        .as_ref()
        .expect("host expert moment before second step");
    assert_slice_close(
        "host moment retained before second step",
        &device_state_before.experts[1]
            .weights
            .as_ref()
            .expect("device moment before second step")
            .0
            .data,
        &host_moment.0,
        2e-6,
    );
    drop(host_state);
    drop(host_buffers);
    drop(device_buffers);
    original
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("continued device MoE Adam step");
    resumed
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("resumed device MoE Adam step");
    migrated
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("device MoE Adam step after legacy moment migration");
    host_reference
        .update_weights_gpu(0.01, 0.0, 0.001, 1)
        .expect("continued host MoE Adam reference step");
    assert_eq!(original.adam_step, host_reference.adam_step);
    assert_eq!(migrated.adam_step, host_reference.adam_step);
    let device_state = original.moe_adam_state[0]
        .as_ref()
        .expect("continued device MoE Adam state");
    let host_states = host_reference.host_adam_state.borrow();
    let host_state = host_states[0]
        .as_ref()
        .expect("continued host MoE Adam state");
    compare_moments(
        "continued router weight moment",
        &device_state.router.weights,
        &host_state.weights,
    );
    compare_moments(
        "continued router bias moment",
        &device_state.router.biases,
        &host_state.biases,
    );
    for (index, device_expert) in device_state.experts.iter().enumerate() {
        let host_expert = &host_state.children[index];
        compare_moments(
            "continued expert weight moment",
            &device_expert.weights,
            &host_expert.weights,
        );
        compare_moments(
            "continued expert bias moment",
            &device_expert.biases,
            &host_expert.biases,
        );
    }
    let migrated_state = migrated.moe_adam_state[0]
        .as_ref()
        .expect("migrated device MoE Adam state");
    compare_moments(
        "migrated router weight moment",
        &migrated_state.router.weights,
        &host_state.weights,
    );
    compare_moments(
        "migrated router bias moment",
        &migrated_state.router.biases,
        &host_state.biases,
    );
    for (index, device_expert) in migrated_state.experts.iter().enumerate() {
        let host_expert = &host_state.children[index];
        compare_moments(
            "migrated expert weight moment",
            &device_expert.weights,
            &host_expert.weights,
        );
        compare_moments(
            "migrated expert bias moment",
            &device_expert.biases,
            &host_expert.biases,
        );
    }
    assert!(migrated.host_adam_state.borrow()[0].is_none());
    drop(host_states);
    assert_moe_params_close(&original.params[0], &host_reference.params[0], 2e-6);
    assert_moe_params_close(&migrated.params[0], &host_reference.params[0], 2e-6);
    let (LayerParams::Moe(original), LayerParams::Moe(resumed)) =
        (&original.params[0], &resumed.params[0])
    else {
        unreachable!()
    };
    assert_slice_close(
        "resumed MoE router weights",
        &original.router_weights,
        &resumed.router_weights,
        1e-7,
    );
    assert_slice_close(
        "resumed MoE router biases",
        &original.router_biases,
        &resumed.router_biases,
        1e-7,
    );
    for (a, b) in original.experts.iter().zip(&resumed.experts) {
        assert_slice_close("resumed MoE expert weights", &a.weights, &b.weights, 1e-7);
        assert_slice_close("resumed MoE expert biases", &a.biases, &b.biases, 1e-7);
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn gpu_gradients_accumulate_across_subdivisions() {
    // Two backward passes into the same buffers == the sum of two fresh passes.
    let grads = |steps: &[usize]| {
        let mut gpu = optimizer_net("subdivisions=1");
        for &step in steps {
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            let seeds = optimizer_seeds(&gpu, step);
            gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
                .expect("backward");
        }
        optimizer_snapshot(&gpu)
    };
    let both = grads(&[1, 2]);
    let (first, second) = (grads(&[1]), grads(&[2]));
    for layer in 0..2 {
        let sum = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<_>>();
        assert_slice_close(
            "weight updates",
            &sum(&first[layer].0, &second[layer].0),
            &both[layer].0,
            1e-5,
        );
        assert_slice_close(
            "bias updates",
            &sum(&first[layer].1, &second[layer].1),
            &both[layer].1,
            1e-5,
        );
    }
}

/// Finite-difference check of the GPU-path backward of dropout (fixed
/// mask) and scale_channels (both scale layouts) through a conv.
#[test]
fn gpu_dropout_and_scale_channels_backward_match_finite_difference() {
    for cfg_tail in [
        "[dropout]\nprobability=0.4\n",
        "[avgpool]\n[scale_channels]\nfrom=-2\nactivation=leaky\n",
        "[convolutional]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n[scale_channels]\nfrom=-2\nscale_wh=1\nactivation=linear\n",
    ] {
        let cfg = format!(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n{cfg_tail}"
        );
        let build = |weights: &[f32]| {
            let network =
                Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg"))
                    .expect("network");
            let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
            fill_conv_params(&mut gpu.params);
            if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
                p.weights = weights.to_vec();
            }
            gpu
        };
        let base: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).sin() * 0.6).collect();
        let last = build(&base).network.layers.len() - 1;
        let loss = |weights: &[f32]| -> f64 {
            let gpu = build(weights);
            let input = patterned_input(gpu.network.net.input_shape());
            let forward = gpu.forward_training_with_state(&input).expect("forward");
            let out = &forward.cpu_forward.layer_outputs[last];
            out.data
                .iter()
                .enumerate()
                .map(|(i, o)| *o as f64 * (i as f64 * 0.91).cos())
                .sum()
        };
        let mut gpu = build(&base);
        gpu.init_train_buffers_gpu().expect("buffers");
        assert!(
            !gpu.requires_cpu_backward_fallback(),
            "GPU backward path expected"
        );
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu.forward_training_with_state(&input).expect("forward");
        let shape = gpu.network.layers[last].output_shape;
        let mut seeds = vec![None; gpu.network.layers.len()];
        seeds[last] = Some(
            Tensor::from_data(
                shape,
                (0..shape.batch * shape.element_count())
                    .map(|i| (i as f32 * 0.91).cos())
                    .collect(),
            )
            .expect("seed"),
        );
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("backward");
        let analytic = gpu.device_train_buffers.borrow()[0]
            .clone()
            .expect("train")
            .weight_updates
            .data;
        let h = 1e-3f32;
        for (i, a) in analytic.iter().enumerate() {
            let (mut wp, mut wm) = (base.clone(), base.clone());
            wp[i] += h;
            wm[i] -= h;
            let fd = (loss(&wp) - loss(&wm)) / (2.0 * h as f64);
            assert!(
                (*a as f64 - fd).abs() < 2e-3 + 2e-3 * fd.abs(),
                "[{cfg_tail:?}] dW[{i}]: analytic {a} vs finite difference {fd}"
            );
        }
    }
}

#[test]
fn cpu_reorg_backward_is_the_adjoint_of_forward() {
    // <reorg(x), y> == <x, reorg_backward(y)> for forward and reverse layers
    for reverse in [false, true] {
        let spec = crate::model::ReorgSpec { stride: 2, reverse };
        let (small, big) = (TensorShape::new(3, 2, 8, 2), TensorShape::new(6, 4, 2, 2));
        let (input_shape, output_shape) = if reverse { (small, big) } else { (big, small) };
        let x = patterned_input(input_shape);
        let y = patterned_input(output_shape);
        let fx = super::forward_reorg(spec.stride, spec.reverse, &x, output_shape).unwrap();
        let bty = super::backward_reorg(&spec, &y, input_shape).unwrap();
        let lhs: f32 = fx.data.iter().zip(&y.data).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.data.iter().zip(&bty.data).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() < 1e-3,
            "reverse={reverse}: {lhs} vs {rhs}"
        );
    }
}

// ---- DropBlock ----

pub(super) const DROPBLOCK_CFG: &str = "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
         [dropout]\nprobability=0.9\ndropblock=1\ndropblock_size_abs=2\n";

#[test]
fn dropblock_cfg_parses_like_cpp() {
    let spec = |tail: &str| {
        let cfg = format!(
            "[net]\nwidth=8\nheight=6\nchannels=1\nbatch=1\n[dropout]\nprobability=0.3\n{tail}"
        );
        let network =
            Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).unwrap()).unwrap();
        match &network.layers[0].kind {
            LayerKind::Dropout {
                probability,
                dropblock,
            } => {
                assert_eq!(*probability, Some(0.3));
                *dropblock
            }
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(spec(""), None);
    assert_eq!(spec("dropblock=0\ndropblock_size_abs=3\n"), None);
    // neither key set: C++ defaults to 7 (the cap only applies to an explicit value).
    assert_eq!(
        spec("dropblock=1\n"),
        Some(crate::model::DropblockSpec {
            size_rel: 0.0,
            size_abs: 7
        })
    );
    assert_eq!(
        spec("dropblock=1\ndropblock_size_abs=20\n"),
        Some(crate::model::DropblockSpec {
            size_rel: 0.0,
            size_abs: 6
        })
    );
    assert_eq!(
        spec("dropblock=1\ndropblock_size_rel=0.5\n"),
        Some(crate::model::DropblockSpec {
            size_rel: 0.5,
            size_abs: 0
        })
    );
    // both set: the absolute size wins
    assert_eq!(
        spec("dropblock=1\ndropblock_size_rel=0.5\ndropblock_size_abs=3\n"),
        Some(crate::model::DropblockSpec {
            size_rel: 0.0,
            size_abs: 3
        })
    );
}

#[test]
fn dropblock_plan_follows_the_cpp_ramp() {
    let shape = TensorShape::new(10, 8, 4, 1);
    let abs = crate::model::DropblockSpec {
        size_rel: 0.0,
        size_abs: 7,
    };
    // max_batches = 0: no ramp.
    assert_eq!(
        super::dropblock_plan(&abs, 0.5, shape, 0, 0),
        (0.5 / 49.0, 7)
    );
    // ramp over the first 85% of 100 batches: iteration 0 -> nothing.
    assert_eq!(super::dropblock_plan(&abs, 0.5, shape, 0, 100), (0.0, 1));
    // halfway through the ramp: multiplier 0.5 -> int(3.5) = 3.
    let (p, size) = super::dropblock_plan(&abs, 0.5, shape, 42, 100);
    assert_eq!(size, 3);
    assert!((p - 0.5 * (42.0 / 85.0) / 9.0).abs() < 1e-7);
    // after the ramp, the size is capped by the smaller side (height 8).
    assert_eq!(
        super::dropblock_plan(&abs, 0.5, shape, 90, 100),
        (0.5 / 49.0, 7)
    );
    let big = crate::model::DropblockSpec {
        size_rel: 0.0,
        size_abs: 9,
    };
    assert_eq!(super::dropblock_plan(&big, 0.5, shape, 90, 100).1, 8);
    // relative size uses width/height separately, the square is the smaller one.
    let rel = crate::model::DropblockSpec {
        size_rel: 0.5,
        size_abs: 0,
    };
    assert_eq!(super::dropblock_plan(&rel, 0.5, shape, 0, 0).1, 4);
}

#[test]
fn dropout_element_mask_and_adjoint_preserve_nchw_slots() {
    use crate::axis_fixture::{axis_tensor, nchw_index};

    let shape = TensorShape::new(3, 2, 2, 2);
    let input = Tensor::from_data(shape, axis_tensor(shape)).expect("dropout axis input");
    let mut rng_state = 0xD09B_10C5_u64;
    let (output, mask) =
        super::forward_dropout(&input, 0.25, true, &mut rng_state).expect("training dropout");
    let mask = mask.expect("training dropout mask");
    assert_eq!(mask.shape, shape);
    for n in 0..shape.batch {
        for c in 0..shape.channels {
            for h in 0..shape.height {
                for w in 0..shape.width {
                    let index = nchw_index(shape, n, c, h, w);
                    assert_eq!(output.data[index], input.data[index] * mask.data[index]);
                }
            }
        }
    }
    let per_sample = shape.element_count();
    assert_ne!(
        &mask.data[..per_sample],
        &mask.data[per_sample..],
        "C++ draws one mask value per element across the full batch, not one mask per sample"
    );
    let seed = Tensor::from_data(shape, vec![1.0; shape.batch * per_sample])
        .expect("dropout backward seed");
    let backward = super::backward_dropout(&seed, Some(&mask)).expect("dropout backward");
    for index in 0..backward.data.len() {
        assert_eq!(backward.data[index], mask.data[index]);
    }
    let (inference, inference_mask) =
        super::forward_dropout(&input, 0.25, false, &mut rng_state).expect("inference dropout");
    assert_eq!(inference.data, input.data);
    assert!(inference_mask.is_none());
}

#[test]
fn dropblock_reference_zeroes_one_square_per_plane_and_rescales_per_sample() {
    let shape = [2usize, 3, 5, 6];
    let x: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| 1.0 + (i % 7) as f32)
        .collect();
    let bs = 2;
    // block_prob 1 selects the lowest-random cell of every plane.
    let (out, mask) = lightnet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 99);
    let (c, h, w) = (shape[1], shape[2], shape[3]);
    for b in 0..shape[0] {
        let dropped: usize = mask[b * c * h * w..(b + 1) * c * h * w]
            .iter()
            .filter(|m| **m == 0.0)
            .count();
        assert_eq!(dropped, c * bs * bs, "one {bs}x{bs} block per channel");
        let scale = 1.0 / (1.0 - dropped as f32 / (c * h * w) as f32);
        for ch in 0..c {
            let plane = &mask[(b * c + ch) * h * w..(b * c + ch + 1) * h * w];
            let zeros: Vec<(usize, usize)> = (0..h * w)
                .filter(|i| plane[*i] == 0.0)
                .map(|i| (i / w, i % w))
                .collect();
            let (y0, x0) = (
                zeros.iter().map(|p| p.0).min().unwrap(),
                zeros.iter().map(|p| p.1).min().unwrap(),
            );
            let expected: Vec<(usize, usize)> = (0..bs)
                .flat_map(|dy| (0..bs).map(move |dx| (y0 + dy, x0 + dx)))
                .collect();
            assert_eq!(zeros, expected, "contiguous square inside the plane");
            for i in 0..h * w {
                if plane[i] != 0.0 {
                    assert_eq!(plane[i], scale);
                }
            }
        }
    }
    for ((o, m), v) in out.iter().zip(&mask).zip(&x) {
        assert_eq!(*o, v * m);
    }
    // probability 0 keeps everything (scale 1)
    let (out0, mask0) = lightnet_cuda::host_ops::dropblock(&x, &shape, 0.0, bs, 99);
    assert_eq!(out0, x);
    assert!(mask0.iter().all(|m| *m == 1.0));
    // deterministic for a seed, different for another
    assert_eq!(
        lightnet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 99).1,
        mask
    );
    assert_ne!(
        lightnet_cuda::host_ops::dropblock(&x, &shape, 1.0, bs, 100).1,
        mask
    );
}

#[test]
fn gpu_dropblock_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(DROPBLOCK_CFG, 1);
    // the mask is not vacuous
    let network =
        Network::from_config(ConfigFile::parse_str("inline.cfg", DROPBLOCK_CFG).unwrap()).unwrap();
    let cpu = ExecutableNetwork::new(network);
    let forward = cpu
        .forward_training(&patterned_input(cpu.network.net.input_shape()))
        .unwrap();
    let mask = forward.dropout_masks[1].as_ref().expect("dropblock mask");
    assert!(mask.data.iter().any(|m| *m == 0.0) && mask.data.iter().any(|m| *m > 1.0));
    // inference leaves the input alone
    let infer = cpu
        .forward(&patterned_input(cpu.network.net.input_shape()))
        .unwrap();
    assert!(infer.dropout_masks[1].is_none());
    assert_eq!(infer.layer_outputs[1], infer.layer_outputs[0]);
}

/// The training iteration changes the DropBlock mask (ramp + seed mix);
/// iteration 0 of a ramped schedule drops nothing.
#[test]
fn dropblock_mask_follows_the_train_iteration() {
    let cfg = DROPBLOCK_CFG.replace("batch=2\n", "batch=2\nmax_batches=100\n");
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).unwrap()).unwrap();
    let mut cpu = ExecutableNetwork::new(network);
    let input = patterned_input(cpu.network.net.input_shape());
    cpu.set_train_iteration(0);
    assert!(cpu.forward_training(&input).unwrap().dropout_masks[1].is_none());
    cpu.set_train_iteration(60);
    let a = cpu.forward_training(&input).unwrap().dropout_masks[1]
        .clone()
        .expect("mask");
    cpu.set_train_iteration(61);
    let b = cpu.forward_training(&input).unwrap().dropout_masks[1]
        .clone()
        .expect("mask");
    assert_ne!(a, b);
}

#[test]
fn gpu_dropblock_backward_matches_finite_difference() {
    let cfg = format!(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [dropout]\nprobability=0.9\ndropblock=1\ndropblock_size_abs=2\n"
    );
    let build = |weights: &[f32]| {
        let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).expect("cfg"))
            .expect("network");
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        fill_conv_params(&mut gpu.params);
        if let LayerParams::Convolutional(p) = &mut gpu.params[0] {
            p.weights = weights.to_vec();
        }
        gpu
    };
    let base: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).sin() * 0.6).collect();
    let last = 1;
    let loss = |weights: &[f32]| -> f64 {
        let gpu = build(weights);
        let input = patterned_input(gpu.network.net.input_shape());
        let forward = gpu.forward_training_with_state(&input).expect("forward");
        forward.cpu_forward.layer_outputs[last]
            .data
            .iter()
            .enumerate()
            .map(|(i, o)| *o as f64 * (i as f64 * 0.91).cos())
            .sum()
    };
    let mut gpu = build(&base);
    gpu.init_train_buffers_gpu().expect("buffers");
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = patterned_input(gpu.network.net.input_shape());
    let forward = gpu.forward_training_with_state(&input).expect("forward");
    assert!(
        forward.cpu_forward.dropout_masks[last]
            .as_ref()
            .expect("mask")
            .data
            .contains(&0.0)
    );
    let shape = gpu.network.layers[last].output_shape;
    let mut seeds = vec![None; gpu.network.layers.len()];
    seeds[last] = Some(
        Tensor::from_data(
            shape,
            (0..shape.batch * shape.element_count())
                .map(|i| (i as f32 * 0.91).cos())
                .collect(),
        )
        .unwrap(),
    );
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("backward");
    let analytic = gpu.device_train_buffers.borrow()[0]
        .clone()
        .unwrap()
        .weight_updates
        .data;
    let h = 1e-3f32;
    for (i, a) in analytic.iter().enumerate() {
        let (mut wp, mut wm) = (base.clone(), base.clone());
        wp[i] += h;
        wm[i] -= h;
        let fd = (loss(&wp) - loss(&wm)) / (2.0 * h as f64);
        assert!(
            (*a as f64 - fd).abs() < 2e-3 + 2e-3 * fd.abs(),
            "dW[{i}]: analytic {a} vs finite difference {fd}"
        );
    }
}

// ---- per-layer update keys and try_fix_nan ----

/// Two convs (layer 0 with BN, 32-channel input so `grad_centr` applies,
/// layer 1 without BN); `l0`/`l1` add keys to the layer sections.
pub(super) fn keyed_cfg(net_extra: &str, l0: &str, l1: &str) -> String {
    format!(
        "[net]\nwidth=2\nheight=2\nchannels=32\nbatch=1\n{net_extra}\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nbatch_normalize=1\nactivation=linear\n{l0}\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n{l1}\n"
    )
}

pub(super) fn keyed_cpu(cfg: &str) -> ExecutableNetwork {
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).unwrap()).unwrap();
    let mut cpu = ExecutableNetwork::new(network);
    fill_conv_params(&mut cpu.params);
    for (layer, train) in cpu.train_buffers.iter_mut().enumerate() {
        if let LayerTrainBuffers::Convolutional(t) = train {
            for (i, u) in t.weight_updates.iter_mut().enumerate() {
                *u = ((i * 3 + layer) % 11) as f32 * 0.25 - 1.0;
            }
            for (i, u) in t.bias_updates.iter_mut().enumerate() {
                *u = i as f32 * 0.5 - 0.3;
            }
            if let Some(bn) = t.batch_norm.as_mut() {
                bn.scale_updates = vec![0.4, -0.2, 0.6];
            }
        }
    }
    cpu
}

pub(super) fn keyed_gpu(cfg: &str) -> GpuExecutableNetwork<FakeBackend> {
    let cpu = keyed_cpu(cfg);
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    {
        let mut train = gpu.device_train_buffers.borrow_mut();
        for (layer, t) in train.iter_mut().enumerate() {
            let LayerTrainBuffers::Convolutional(c) = &cpu.train_buffers[layer] else {
                continue;
            };
            let t = t.as_mut().unwrap();
            t.weight_updates.data = c.weight_updates.clone();
            t.bias_updates.data = c.bias_updates.clone();
            if let (Some(s), Some(bn)) = (t.scale_updates.as_mut(), c.batch_norm.as_ref()) {
                s.data = bn.scale_updates.clone();
            }
        }
    }
    gpu
}

/// Weights of both layers after one plain step (lr 0.1, momentum 0, no
/// decay, batch 1) from the buffers of `keyed_cpu`, on CPU and GPU.
pub(super) fn keyed_step(cfg: &str, iteration: usize) -> [(Vec<f32>, Vec<f32>); 2] {
    let mut cpu = keyed_cpu(cfg);
    cpu.set_train_iteration(iteration);
    cpu.update_weights(0.1, 0.0, 0.0, 1);
    let mut gpu = keyed_gpu(cfg);
    gpu.set_train_iteration(iteration);
    gpu.update_weights_gpu(0.1, 0.0, 0.0, 1).unwrap();
    let pick = |params: &[LayerParams], i: usize| match &params[i] {
        LayerParams::Convolutional(p) => (p.weights.clone(), p.biases.clone()),
        _ => unreachable!(),
    };
    for i in 0..2 {
        let (c, g) = (pick(&cpu.params, i), pick(&gpu.params, i));
        assert_slice_close("cpu/gpu weights", &c.0, &g.0, 1e-6);
        assert_slice_close("cpu/gpu biases", &c.1, &g.1, 1e-6);
    }
    [pick(&cpu.params, 0), pick(&cpu.params, 1)]
}

#[test]
fn layer_train_keys_parse_and_train_only_bn_propagates_backwards() {
    let net = |l0: &str, l1: &str| {
        Network::from_config(ConfigFile::parse_str("inline.cfg", &keyed_cfg("", l0, l1)).unwrap())
            .unwrap()
    };
    let n = net(
        "learning_rate=0.5\nclip=2\ngrad_centr=1\nburnin_update=2\n",
        "dont_update=7\n",
    );
    assert_eq!(n.layers[0].train.learning_rate_scale, 0.5);
    assert_eq!(
        (
            n.layers[0].train.clip,
            n.layers[0].train.grad_centr,
            n.layers[0].train.burnin_update
        ),
        (2.0, true, 2)
    );
    assert_eq!(
        (
            n.layers[1].train.dont_update,
            n.layers[1].train.learning_rate_scale
        ),
        (7, 1.0)
    );
    assert!(!n.layers[0].train.train_only_bn && !n.layers[1].train.train_only_bn);
    // C++ set_train_only_bn: a later layer's flag covers every earlier layer
    let n = net("", "train_only_bn=1\n");
    assert!(n.layers[0].train.train_only_bn && n.layers[1].train.train_only_bn);
    let n = net("train_only_bn=1\n", "");
    assert!(n.layers[0].train.train_only_bn && !n.layers[1].train.train_only_bn);
    assert!(
        !Network::from_config(ConfigFile::parse_str("i.cfg", &keyed_cfg("", "", "")).unwrap())
            .unwrap()
            .net
            .try_fix_nan
    );
    assert!(
        Network::from_config(
            ConfigFile::parse_str("i.cfg", &keyed_cfg("try_fix_nan=1", "", "")).unwrap()
        )
        .unwrap()
        .net
        .try_fix_nan
    );
}

#[test]
fn per_layer_learning_rate_scales_only_that_layer_cpu_and_gpu() {
    let plain = keyed_step(&keyed_cfg("", "", ""), 0);
    let scaled = keyed_step(&keyed_cfg("", "learning_rate=0.5\n", ""), 0);
    let base = keyed_cpu(&keyed_cfg("", "", ""));
    let LayerParams::Convolutional(p0) = &base.params[0] else {
        unreachable!()
    };
    for i in 0..p0.weights.len() {
        let d_plain = plain[0].0[i] - p0.weights[i];
        let d_scaled = scaled[0].0[i] - p0.weights[i];
        assert!(
            (d_scaled - 0.5 * d_plain).abs() < 1e-6,
            "w[{i}]: {d_scaled} vs half of {d_plain}"
        );
    }
    assert!(plain[0].0 != p0.weights);
    assert_eq!(plain[1], scaled[1], "other layers keep the global rate");
}

#[test]
fn dont_update_burnin_update_and_train_only_bn_gate_the_step() {
    let base = keyed_cpu(&keyed_cfg("", "", ""));
    let init = |i: usize| match &base.params[i] {
        LayerParams::Convolutional(p) => (p.weights.clone(), p.biases.clone()),
        _ => unreachable!(),
    };
    // dont_update=3: C++ updates once iteration_num (= iteration + 1) > 3.
    for (iteration, updated) in [(0, false), (2, false), (3, true)] {
        let r = keyed_step(&keyed_cfg("", "", "dont_update=3\n"), iteration);
        assert_eq!(
            r[1] != init(1),
            updated,
            "dont_update at iteration {iteration}"
        );
        assert!(r[0] != init(0), "layer 0 is not gated");
    }
    // burnin_update=2, burn_in=10: skipped while 2*10 > iteration_num.
    for (iteration, updated) in [(0, false), (18, false), (19, true)] {
        let r = keyed_step(&keyed_cfg("burn_in=10", "burnin_update=2\n", ""), iteration);
        assert_eq!(
            r[0] != init(0),
            updated,
            "burnin_update at iteration {iteration}"
        );
        assert!(r[1] != init(1));
    }
    // train_only_bn on layer 1 freezes both layers (C++ skips the update
    // of every layer up to the flagged one).
    let r = keyed_step(&keyed_cfg("", "", "train_only_bn=1\n"), 5);
    assert_eq!(r[0], init(0));
    assert_eq!(r[1], init(1));

    let mut muon = keyed_cpu(&keyed_cfg("muon=1\n", "", "train_only_bn=1\n"));
    let before = muon.params.clone();
    muon.set_train_iteration(5);
    muon.update_weights_muon(0.1, 1).unwrap();
    assert_eq!(
        muon.params, before,
        "train_only_bn freezes Muon and fallback"
    );
    assert!(muon.muon_state.iter().all(Option::is_none));
    assert!(muon.host_adam_state.iter().all(Option::is_none));
    assert_eq!(muon.adam_step(), 1, "the global update step still advances");
}

#[test]
fn clip_clamps_weights_after_the_step() {
    let free = keyed_step(&keyed_cfg("", "", ""), 0);
    let clipped = keyed_step(&keyed_cfg("", "clip=0.3\n", ""), 0);
    assert!(
        free[0].0.iter().any(|w| w.abs() > 0.3),
        "test must exceed the clip"
    );
    let expected: Vec<f32> = free[0].0.iter().map(|w| w.clamp(-0.3, 0.3)).collect();
    assert_slice_close("clipped weights", &clipped[0].0, &expected, 1e-6);
    assert_eq!(clipped[0].1, free[0].1, "biases are not clipped");
    assert_eq!(clipped[1], free[1]);
}

/// `keyed_step` compares CPU and FakeBackend GPU updates: the BN filter
/// with 32 weights is centered, while the no-BN layer remains unchanged.
#[test]
fn gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate() {
    let base = keyed_cpu(&keyed_cfg("", "", ""));
    let LayerParams::Convolutional(p0) = &base.params[0] else {
        unreachable!()
    };
    let LayerTrainBuffers::Convolutional(t0) = &base.train_buffers[0] else {
        unreachable!()
    };
    let free = keyed_step(&keyed_cfg("", "", ""), 0);
    let centred = keyed_step(&keyed_cfg("", "grad_centr=1\n", ""), 0);
    let f_size = 32;
    for f in 0..3 {
        let upd = &t0.weight_updates[f * f_size..(f + 1) * f_size];
        let mean = upd.iter().sum::<f32>() / f_size as f32;
        assert!(mean.abs() > 1e-3, "filter {f} needs a non-zero mean");
        for i in 0..f_size {
            let idx = f * f_size + i;
            let expect = p0.weights[idx] + 0.1 * (upd[i] - mean);
            assert!((centred[0].0[idx] - expect).abs() < 1e-5, "w[{idx}]");
        }
    }
    assert!(free[0].0 != centred[0].0);
    assert_eq!(free[0].1, centred[0].1, "biases untouched");
    assert_eq!(free[1], centred[1], "no BN -> no centralisation");
    // grad_centr without batch norm does nothing (C++ condition)
    let cfg = keyed_cfg("", "", "grad_centr=1\n");
    assert_eq!(keyed_step(&cfg, 0)[1], free[1]);
}

#[test]
fn try_fix_nan_fixes_conv_outputs_and_clamps_deltas_cpu_and_gpu() {
    let cfg = |fix: u8| keyed_cfg(&format!("try_fix_nan={fix}"), "", "");
    let input_of = |shape| patterned_input(shape);
    // forward: a NaN weight poisons the output unless try_fix_nan is set
    for fix in [0u8, 1] {
        let mut cpu = keyed_cpu(&cfg(fix));
        if let LayerParams::Convolutional(p) = &mut cpu.params[1] {
            p.weights[0] = f32::NAN;
        }
        let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
        gpu.params = cpu.params.clone();
        let input = input_of(cpu.network.net.input_shape());
        let cpu_out = cpu.forward_training(&input).unwrap().layer_outputs[1].clone();
        let gpu_out = gpu
            .forward_training_with_state(&input)
            .unwrap()
            .cpu_forward
            .layer_outputs[1]
            .clone();
        for out in [&cpu_out, &gpu_out] {
            assert_eq!(
                out.data.iter().all(|v| v.is_finite()),
                fix == 1,
                "try_fix_nan={fix}"
            );
        }
        if fix == 1 {
            assert_slice_close("fixed output", &cpu_out.data, &gpu_out.data, 1e-6);
        }
    }
    // backward: output deltas are clamped to [-1, 1] (last layer is linear)
    let big: Vec<f32> = (0..8)
        .map(|i| [5.0, -7.0, 0.25, -0.5, 3.0, -0.1, 9.0, 0.9][i])
        .collect();
    let clamped: Vec<f32> = big.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
    let run = |fix: u8, seed: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut cpu = keyed_cpu(&cfg(fix));
        for t in &mut cpu.train_buffers {
            if let LayerTrainBuffers::Convolutional(t) = t {
                t.weight_updates.fill(0.0);
                t.bias_updates.fill(0.0);
            }
        }
        let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        let input = input_of(cpu.network.net.input_shape());
        let shape = cpu.network.layers[1].output_shape;
        let mut seeds = vec![None; 2];
        seeds[1] = Some(Tensor::from_data(shape, seed.to_vec()).unwrap());
        let f = cpu.forward_training(&input).unwrap();
        cpu.backward_accumulate_seeded(&input, &f, &seeds).unwrap();
        let gf = gpu.forward_training_with_state(&input).unwrap();
        gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
        let LayerTrainBuffers::Convolutional(c) = &cpu.train_buffers[1] else {
            unreachable!()
        };
        let g = gpu.device_train_buffers.borrow()[1].clone().unwrap();
        assert_slice_close(
            "cpu/gpu layer-1 gradient",
            &c.weight_updates,
            &g.weight_updates.data,
            1e-4,
        );
        (c.weight_updates.clone(), c.bias_updates.clone())
    };
    let with_fix = run(1, &big);
    let reference = run(0, &clamped);
    let unclamped = run(0, &big);
    assert_slice_close("clamped delta gradient", &with_fix.0, &reference.0, 1e-5);
    assert_slice_close("clamped bias gradient", &with_fix.1, &reference.1, 1e-5);
    assert!(
        with_fix.1 != unclamped.1,
        "without try_fix_nan the large deltas pass through"
    );
}

pub(super) fn recursive_test_network(
    channels: usize,
    body_count: usize,
    parent_options: &str,
    body_sections: &str,
    net_options: &str,
) -> Network {
    let cfg = format!(
        "[net]\nwidth=2\nheight=1\nchannels={channels}\nbatch=1\n{net_options}\n\
             [recursive_block]\nbody={body_count}\n{parent_options}\n{body_sections}"
    );
    Network::from_config(ConfigFile::parse_str("recursive-block-test.cfg", &cfg).unwrap()).unwrap()
}

pub(super) fn recursive_objective(
    executable: &ExecutableNetwork,
    input: &Tensor,
    seed: &Tensor,
) -> f32 {
    executable
        .forward_training(input)
        .unwrap()
        .final_output()
        .unwrap()
        .data
        .iter()
        .zip(&seed.data)
        .map(|(output, upstream)| output * upstream)
        .sum()
}

pub(super) fn check_recursive_scalar_finite_difference(
    executable: &mut ExecutableNetwork,
    input: &Tensor,
    seed: &Tensor,
    name: &str,
    original: f32,
    analytical: f32,
    mut set: impl FnMut(&mut ExecutableNetwork, f32),
) {
    let eps = 5.0e-4;
    set(executable, original + eps);
    let plus = recursive_objective(executable, input, seed);
    set(executable, original - eps);
    let minus = recursive_objective(executable, input, seed);
    set(executable, original);
    let numerical = (plus - minus) / (2.0 * eps);
    let tolerance = 2.0e-3 + 2.0e-2 * analytical.abs().max(numerical.abs());
    assert!(
        (analytical - numerical).abs() <= tolerance,
        "[recursive_block] {name}: analytic {analytical}, numerical {numerical}, tolerance {tolerance}"
    );
}

pub(super) fn check_recursive_input_gradients(
    executable: &ExecutableNetwork,
    input: &Tensor,
    seed: &Tensor,
    analytic: &Tensor,
) {
    for index in 0..input.data.len() {
        let eps = 5.0e-4;
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus.data[index] += eps;
        minus.data[index] -= eps;
        let numerical = (recursive_objective(executable, &plus, seed)
            - recursive_objective(executable, &minus, seed))
            / (2.0 * eps);
        let tolerance = 2.0e-3 + 2.0e-2 * analytic.data[index].abs().max(numerical.abs());
        assert!(
            (analytic.data[index] - numerical).abs() <= tolerance,
            "[recursive_block] input[{index}]: analytic {}, numerical {numerical}, tolerance {tolerance}",
            analytic.data[index]
        );
    }
}

#[test]
fn recursive_block_consumes_body_sections_without_shifting_top_level_indices() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "recursive-index.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=1\n\
                 [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
                 [recurrent_block]\nloops=2\nbody=1\n\
                 [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(network.layers.len(), 3);
    assert_eq!(
        network
            .layers
            .iter()
            .map(|layer| layer.index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let LayerKind::RecursiveBlock(spec) = &network.layers[1].kind else {
        panic!("recursive block layer")
    };
    assert_eq!(spec.body.len(), 1);
    assert_eq!(spec.body[0].index, 0);
    assert_eq!(spec.loops, 2);
    assert_eq!(
        network.layers[1].output_shape,
        network.layers[1].input_shape
    );
}

#[test]
fn recursive_block_rejects_unsupported_and_malformed_options() {
    let valid_body = "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n";
    let invalid = [
        ("loops=0\n", valid_body, "loops must be >= 1"),
        ("ouroboros=2\n", valid_body, "ouroboros must be 0 or 1"),
        ("lora_rank=0\n", valid_body, "Stage-2 Conv-LoRA"),
        ("mystery=1\n", valid_body, "unknown or unsupported option"),
        ("", "[route]\nlayers=-1\n", "unsupported layer kind"),
        (
            "",
            "[convolutional]\nfilters=1\nsize=3\nactivation=linear\n",
            "body changes dimensions",
        ),
        (
            "",
            "[convolutional]\nfilters=1\nsize=1\nactivation=linear\nmystery=1\n",
            "unknown or unsupported option",
        ),
    ];
    for (options, body, expected) in invalid {
        let cfg = format!(
            "[net]\nwidth=2\nheight=1\nchannels=1\n[recursive_block]\nbody=1\n{options}{body}"
        );
        let error =
            Network::from_config(ConfigFile::parse_str("invalid-recursive.cfg", &cfg).unwrap())
                .unwrap_err()
                .to_string();
        assert!(
            error.contains(expected),
            "{error} did not contain `{expected}`"
        );
    }
    let valid = recursive_test_network(1, 1, "loops=1\n", valid_body, "");
    assert_eq!(valid.layers.len(), 1);
}

#[test]
fn recursive_block_dropout_masks_are_independent_and_reproducible_per_pass() {
    let default_network = Network::from_config(
        ConfigFile::parse_str(
            "recursive-dropout-default.cfg",
            r#"[net]
width=2
height=1
channels=1
[recursive_block]
body=1
[dropout]
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let LayerKind::RecursiveBlock(spec) = &default_network.layers[0].kind else {
        panic!("recursive-block layer")
    };
    assert!(matches!(
        spec.body[0].kind,
        LayerKind::Dropout {
            probability: None,
            ..
        }
    ));

    let network = Network::from_config(
        ConfigFile::parse_str(
            "recursive-dropout.cfg",
            r#"[net]
width=64
height=1
channels=1
batch=1
[recursive_block]
loops=4
injection=0
body=1
[dropout]
probability=0.5
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let executable = ExecutableNetwork::new(network.clone());
    let input = Tensor::from_data(network.net.input_shape(), vec![1.0; 64]).unwrap();
    let first = executable.forward_training(&input).unwrap();
    let second = executable.forward_training(&input).unwrap();
    let first_cache = first.recursive_block_caches[0].as_ref().unwrap();
    let second_cache = second.recursive_block_caches[0].as_ref().unwrap();
    let masks = first_cache
        .body_passes
        .iter()
        .map(|pass| pass.dropout_masks[0].as_ref().unwrap().data.clone())
        .collect::<Vec<_>>();
    let repeated_masks = second_cache
        .body_passes
        .iter()
        .map(|pass| pass.dropout_masks[0].as_ref().unwrap().data.clone())
        .collect::<Vec<_>>();
    assert_eq!(masks, repeated_masks);
    assert!(masks.windows(2).any(|pair| pair[0] != pair[1]));
}

#[test]
fn recursive_block_static_full_bptt_matches_input_and_parameter_differences() {
    let network = recursive_test_network(
        1,
        1,
        "loops=3\ninjection=1\nbody_scale=0.3\nresidual_scale=0.4\ninjection_scale=0.2\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        "",
    );
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::RecursiveBlock(params) = &mut executable.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(body) = &mut params.body[0] else {
        panic!("body convolution parameters")
    };
    body.weights[0] = 0.7;
    body.biases[0] = 0.1;
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![0.3, 0.8]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let output = forward.final_output().unwrap();
    for (got, source) in output.data.iter().zip(&input.data) {
        let mut expected = *source;
        for _ in 0..3 {
            expected = 0.3 * (0.7 * expected + 0.1) + 0.4 * expected + 0.2 * source;
        }
        assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
    }
    let seed = Tensor::from_data(output.shape, vec![0.7, -0.4]).unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    check_recursive_input_gradients(&executable, &input, &seed, &backward.input_gradient);
    let LayerParams::RecursiveBlock(gradient) = &backward.param_gradients[0] else {
        panic!("recursive-block parameter gradients")
    };
    let LayerParams::Convolutional(body_gradient) = &gradient.body[0] else {
        panic!("body convolution gradients")
    };
    let weight = match &executable.params[0] {
        LayerParams::RecursiveBlock(params) => match &params.body[0] {
            LayerParams::Convolutional(body) => body.weights[0],
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    check_recursive_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "body weight",
        weight,
        body_gradient.weights[0],
        |exe, value| {
            let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                unreachable!()
            };
            let LayerParams::Convolutional(body) = &mut params.body[0] else {
                unreachable!()
            };
            body.weights[0] = value;
        },
    );
    let bias = match &executable.params[0] {
        LayerParams::RecursiveBlock(params) => match &params.body[0] {
            LayerParams::Convolutional(body) => body.biases[0],
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    check_recursive_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "body bias",
        bias,
        body_gradient.biases[0],
        |exe, value| {
            let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                unreachable!()
            };
            let LayerParams::Convolutional(body) = &mut params.body[0] else {
                unreachable!()
            };
            body.biases[0] = value;
        },
    );
}

#[test]
fn recursive_block_ouroboros_controller_and_shared_body_match_finite_differences() {
    let network = recursive_test_network(
        1,
        1,
        "loops=3\ninjection=1\ninjection_scale=0.2\nbody_scale=0.8\n\
             ouroboros=1\ngate_bias=-0.2\ngamma_clip=2\ngate_clip=2\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        "",
    );
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::RecursiveBlock(params) = &mut executable.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(body) = &mut params.body[0] else {
        panic!("body convolution parameters")
    };
    body.weights[0] = 0.6;
    body.biases[0] = 0.08;
    params
        .controller_weights
        .copy_from_slice(&[0.05, -0.03, -0.04, 0.02]);
    params.controller_biases.copy_from_slice(&[0.1, -0.2]);
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![0.2, 0.7]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let seed = Tensor::from_data(forward.final_output().unwrap().shape, vec![0.7, -0.4]).unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    check_recursive_input_gradients(&executable, &input, &seed, &backward.input_gradient);
    let LayerParams::RecursiveBlock(gradient) = &backward.param_gradients[0] else {
        panic!("recursive-block parameter gradients")
    };
    let LayerParams::Convolutional(body_gradient) = &gradient.body[0] else {
        panic!("body convolution gradients")
    };
    for (name, index, original, analytic) in [
        (
            "controller weight",
            0,
            match &executable.params[0] {
                LayerParams::RecursiveBlock(params) => params.controller_weights[0],
                _ => unreachable!(),
            },
            gradient.controller_weights[0],
        ),
        (
            "controller gamma bias",
            0,
            match &executable.params[0] {
                LayerParams::RecursiveBlock(params) => params.controller_biases[0],
                _ => unreachable!(),
            },
            gradient.controller_biases[0],
        ),
        (
            "controller gate bias",
            1,
            match &executable.params[0] {
                LayerParams::RecursiveBlock(params) => params.controller_biases[1],
                _ => unreachable!(),
            },
            gradient.controller_biases[1],
        ),
    ] {
        check_recursive_scalar_finite_difference(
            &mut executable,
            &input,
            &seed,
            name,
            original,
            analytic,
            move |exe, value| {
                let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                    unreachable!()
                };
                if name == "controller weight" {
                    params.controller_weights[index] = value;
                } else {
                    params.controller_biases[index] = value;
                }
            },
        );
    }
    for index in 1..3 {
        let original = match &executable.params[0] {
            LayerParams::RecursiveBlock(params) => params.controller_weights[index],
            _ => unreachable!(),
        };
        check_recursive_scalar_finite_difference(
            &mut executable,
            &input,
            &seed,
            "controller weight",
            original,
            gradient.controller_weights[index],
            move |exe, value| {
                let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                    unreachable!()
                };
                params.controller_weights[index] = value;
            },
        );
    }
    let original_body_weight = match &executable.params[0] {
        LayerParams::RecursiveBlock(params) => match &params.body[0] {
            LayerParams::Convolutional(body) => body.weights[0],
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    check_recursive_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "shared body weight",
        original_body_weight,
        body_gradient.weights[0],
        |exe, value| {
            let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                unreachable!()
            };
            let LayerParams::Convolutional(body) = &mut params.body[0] else {
                unreachable!()
            };
            body.weights[0] = value;
        },
    );
    let original_body_bias = match &executable.params[0] {
        LayerParams::RecursiveBlock(params) => match &params.body[0] {
            LayerParams::Convolutional(body) => body.biases[0],
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };
    check_recursive_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "shared body bias",
        original_body_bias,
        body_gradient.biases[0],
        |exe, value| {
            let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                unreachable!()
            };
            let LayerParams::Convolutional(body) = &mut params.body[0] else {
                unreachable!()
            };
            body.biases[0] = value;
        },
    );
    let original_controller_weight = match &executable.params[0] {
        LayerParams::RecursiveBlock(params) => params.controller_weights[3],
        _ => unreachable!(),
    };
    check_recursive_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "controller step-input weight",
        original_controller_weight,
        gradient.controller_weights[3],
        |exe, value| {
            let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                unreachable!()
            };
            params.controller_weights[3] = value;
        },
    );
}

#[test]
fn recursive_block_lattice_projection_backward_matches_finite_differences() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "recursive-lattice.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\nbatch=1\n\
                 [recursive_block]\nloops=2\ninjection=0\nbody_scale=1\n\
                 residual_scale=0\nlattice_group=2\nlattice_threshold=0.1\n\
                 lattice_temperature=1\nlattice_mix=0.3\nbody=1\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::RecursiveBlock(params) = &mut executable.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(body) = &mut params.body[0] else {
        panic!("body convolution parameters")
    };
    body.weights.copy_from_slice(&[1.2, 0.1, -0.02, 1.375]);
    body.biases.copy_from_slice(&[0.0, 0.0]);
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![0.35, 0.8]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let seed = Tensor::from_data(forward.final_output().unwrap().shape, vec![0.7, -0.4]).unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    check_recursive_input_gradients(&executable, &input, &seed, &backward.input_gradient);
    let LayerParams::RecursiveBlock(gradient) = &backward.param_gradients[0] else {
        panic!("recursive-block parameter gradients")
    };
    let LayerParams::Convolutional(body_gradient) = &gradient.body[0] else {
        panic!("body convolution gradients")
    };
    for index in 0..4 {
        let original = match &executable.params[0] {
            LayerParams::RecursiveBlock(params) => match &params.body[0] {
                LayerParams::Convolutional(body) => body.weights[index],
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        check_recursive_scalar_finite_difference(
            &mut executable,
            &input,
            &seed,
            "lattice body weight",
            original,
            body_gradient.weights[index],
            move |exe, value| {
                let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                    unreachable!()
                };
                let LayerParams::Convolutional(body) = &mut params.body[0] else {
                    unreachable!()
                };
                body.weights[index] = value;
            },
        );
    }
    for index in 0..2 {
        let original = match &executable.params[0] {
            LayerParams::RecursiveBlock(params) => match &params.body[0] {
                LayerParams::Convolutional(body) => body.biases[index],
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        check_recursive_scalar_finite_difference(
            &mut executable,
            &input,
            &seed,
            "lattice body bias",
            original,
            body_gradient.biases[index],
            move |exe, value| {
                let LayerParams::RecursiveBlock(params) = &mut exe.params[0] else {
                    unreachable!()
                };
                let LayerParams::Convolutional(body) = &mut params.body[0] else {
                    unreachable!()
                };
                body.biases[index] = value;
            },
        );
    }
}

#[test]
fn recursive_block_initialization_and_weights_follow_body_only_source_order() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "recursive-weights.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=1\nbatch=1\n\
                 [recursive_block]\nloops=2\nouroboros=1\nbody=2\n\
                 [convolutional]\nbatch_normalize=1\nfilters=1\nsize=1\nactivation=linear\n\
                 [connected]\noutput=1\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut first = ExecutableNetwork::new(network.clone());
    let mut second = ExecutableNetwork::new(network.clone());
    crate::initialize::initialize_training_params(&network, &mut first.params, 77);
    crate::initialize::initialize_training_params(&network, &mut second.params, 77);
    assert_eq!(first.params, second.params);
    let LayerParams::RecursiveBlock(params) = &first.params[0] else {
        panic!("recursive-block parameters")
    };
    assert!(params.controller_weights.iter().all(|value| *value == 0.0));
    assert_eq!(params.controller_biases, [0.0, -2.0]);
    assert_eq!(network.layers[0].weights().unwrap().parameter_count(), 7);

    let LayerParams::RecursiveBlock(params) = &mut first.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(conv) = &mut params.body[0] else {
        panic!("body convolution parameters")
    };
    conv.biases[0] = 1.0;
    conv.weights[0] = 5.0;
    let batch_norm = conv.batch_norm.as_mut().unwrap();
    batch_norm.scales[0] = 2.0;
    batch_norm.rolling_mean[0] = 3.0;
    batch_norm.rolling_variance[0] = 4.0;
    let LayerParams::Connected(connected) = &mut params.body[1] else {
        panic!("body connected parameters")
    };
    connected.biases[0] = 6.0;
    connected.weights[0] = 7.0;
    params.controller_weights.fill(9.0);
    params.controller_biases.copy_from_slice(&[8.0, 7.0]);
    let path = std::env::temp_dir().join(format!(
        "darknet-recursive-block-{}.weights",
        std::process::id()
    ));
    crate::weights::save_weights(&network, &first.params, &path, 0).unwrap();
    let bytes = fs::read(&path).unwrap();
    let expected_payload = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(&bytes[20..], expected_payload);
    assert_eq!(bytes.len(), 20 + 7 * 4);

    let mut restored = ExecutableNetwork::new(network.clone());
    if let LayerParams::RecursiveBlock(params) = &mut restored.params[0] {
        params.controller_weights.fill(5.0);
    }
    restored.load_weights_file(&path).unwrap();
    let _ = fs::remove_file(&path);
    let LayerParams::RecursiveBlock(params) = &restored.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(conv) = &params.body[0] else {
        panic!("body convolution parameters")
    };
    assert_eq!(conv.biases, [1.0]);
    assert_eq!(conv.weights, [5.0]);
    let batch_norm = conv.batch_norm.as_ref().unwrap();
    assert_eq!(batch_norm.scales, [2.0]);
    assert_eq!(batch_norm.rolling_mean, [3.0]);
    assert_eq!(batch_norm.rolling_variance, [4.0]);
    let LayerParams::Connected(connected) = &params.body[1] else {
        panic!("body connected parameters")
    };
    assert_eq!(connected.biases, [6.0]);
    assert_eq!(connected.weights, [7.0]);
    assert_eq!(params.controller_weights, [5.0; 4]);
    assert_eq!(params.controller_biases, [0.0, -2.0]);
}

#[test]
fn recursive_block_training_updates_nested_batchnorm_statistics_per_pass() {
    let network = recursive_test_network(
        1,
        1,
        "loops=2\ninjection=0\nbody_scale=1\nresidual_scale=0\n",
        "[convolutional]\nbatch_normalize=1\nfilters=1\nsize=1\nactivation=linear\n",
        "",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::RecursiveBlock(params) = &mut executable.params[0] else {
        panic!("recursive-block parameters")
    };
    let LayerParams::Convolutional(body) = &mut params.body[0] else {
        panic!("body convolution parameters")
    };
    body.weights[0] = 1.0;
    body.batch_norm.as_mut().unwrap().scales[0] = 1.0;
    body.batch_norm.as_mut().unwrap().rolling_variance[0] = 1.0;
    let input = Tensor::from_data(network.net.input_shape(), vec![0.2, 0.6]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let seed = Tensor::from_data(forward.final_output().unwrap().shape, vec![0.3, -0.2]).unwrap();
    executable
        .backward_accumulate_seeded(&input, &forward, &[Some(seed)])
        .unwrap();
    let LayerParams::RecursiveBlock(params) = &executable.params[0] else {
        unreachable!()
    };
    let LayerParams::Convolutional(body) = &params.body[0] else {
        unreachable!()
    };
    let batch_norm = body.batch_norm.as_ref().unwrap();
    assert!(batch_norm.rolling_mean[0] > 0.0);
    assert!(batch_norm.rolling_variance[0] < 1.0);
}

#[test]
fn recursive_block_sgd_updates_controller_and_shared_body_with_parent_rate() {
    let network = recursive_test_network(
        1,
        1,
        "loops=2\nouroboros=1\nlearning_rate=0.5\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        "",
    );
    let mut executable = ExecutableNetwork::new(network);
    if let LayerParams::RecursiveBlock(params) = &mut executable.params[0] {
        params.controller_weights.fill(0.05);
        params.controller_biases.fill(0.1);
        let LayerParams::Convolutional(body) = &mut params.body[0] else {
            panic!("body convolution parameters")
        };
        body.weights[0] = 0.5;
        body.biases[0] = 0.1;
    }
    if let LayerTrainBuffers::RecursiveBlock(train) = &mut executable.train_buffers[0] {
        train.controller_weight_updates.fill(0.1);
        train.controller_bias_updates.fill(0.2);
        let LayerTrainBuffers::Convolutional(body) = &mut train.body[0] else {
            panic!("body convolution train buffers")
        };
        body.weight_updates[0] = 0.3;
        body.bias_updates[0] = 0.4;
    }
    executable.update_weights(0.2, 0.0, 0.0, 1);
    let LayerParams::RecursiveBlock(params) = &executable.params[0] else {
        unreachable!()
    };
    let LayerParams::Convolutional(body) = &params.body[0] else {
        unreachable!()
    };
    assert!((body.weights[0] - 0.53).abs() < 1e-6);
    assert!((body.biases[0] - 0.14).abs() < 1e-6);
    assert!((params.controller_weights[0] - 0.06).abs() < 1e-6);
    assert!((params.controller_biases[0] - 0.12).abs() < 1e-6);
}

#[test]
fn recursive_block_adam_state_round_trips_and_rejects_wrong_body_shapes() {
    let network = recursive_test_network(
        1,
        1,
        "loops=2\nouroboros=1\n",
        "[convolutional]\nbatch_normalize=1\nfilters=1\nsize=1\nactivation=linear\n",
        "adam=1\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    if let LayerParams::RecursiveBlock(params) = &mut executable.params[0] {
        params.controller_weights.fill(0.05);
        params.controller_biases.copy_from_slice(&[0.1, -0.2]);
        let LayerParams::Convolutional(body) = &mut params.body[0] else {
            panic!("body convolution parameters")
        };
        body.weights[0] = 0.5;
        body.biases[0] = 0.1;
        body.batch_norm.as_mut().unwrap().scales[0] = 1.0;
    }
    if let LayerTrainBuffers::RecursiveBlock(train) = &mut executable.train_buffers[0] {
        train.controller_weight_updates.fill(0.2);
        train.controller_bias_updates.fill(0.1);
        let LayerTrainBuffers::Convolutional(body) = &mut train.body[0] else {
            panic!("body convolution train buffers")
        };
        body.weight_updates[0] = 0.3;
        body.bias_updates[0] = 0.4;
        body.batch_norm.as_mut().unwrap().scale_updates[0] = 0.5;
    }
    executable.update_weights_adam(0.01, 1).unwrap();
    assert!(executable.host_adam_state[0].is_some());

    let path =
        std::env::temp_dir().join(format!("darknet-recursive-adam-{}.opt", std::process::id()));
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network.clone());
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.host_adam_state, executable.host_adam_state);

    let mut malformed = executable.host_adam_state[0].clone().unwrap();
    malformed.children[0].weights = Some((Vec::new(), Vec::new()));
    executable.host_adam_state[0] = Some(malformed);
    let malformed_path = std::env::temp_dir().join(format!(
        "darknet-recursive-adam-bad-{}.opt",
        std::process::id()
    ));
    executable.save_optimizer_state(&malformed_path).unwrap();
    let error = restored
        .load_optimizer_state(&malformed_path)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("recursive-block Adam body convolution weight"),
        "{error}"
    );
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(malformed_path);
}

#[test]
fn recursive_block_muon_maps_controller_and_body_matrices_and_rejects_gpu() {
    let network = recursive_test_network(
        1,
        1,
        "loops=2\nouroboros=1\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        "muon=1\ndecoupled_decay=1\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    if let LayerTrainBuffers::RecursiveBlock(train) = &mut executable.train_buffers[0] {
        train.controller_weight_updates.fill(0.1);
        train.controller_bias_updates.fill(0.1);
        let LayerTrainBuffers::Convolutional(body) = &mut train.body[0] else {
            panic!("body convolution train buffers")
        };
        body.weight_updates[0] = 0.2;
        body.bias_updates[0] = 0.1;
    }
    executable.update_weights_muon(0.01, 1).unwrap();
    let matrices = &executable.muon_state[0].as_ref().unwrap().matrices;
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols))
            .collect::<Vec<_>>(),
        [(2, 2), (1, 1)]
    );

    let path =
        std::env::temp_dir().join(format!("darknet-recursive-muon-{}.opt", std::process::id()));
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network.clone());
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.muon_state, executable.muon_state);
    let _ = fs::remove_file(path);

    let gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let input = Tensor::from_data(network.net.input_shape(), vec![0.1; 2]).unwrap();
    assert!(
        gpu.forward(&input)
            .unwrap_err()
            .to_string()
            .contains("[recursive_block] GPU execution is not implemented in the Stage-1 CPU port")
    );
    assert!(
        gpu.forward_compact(&input)
            .unwrap_err()
            .to_string()
            .contains(
                "[recursive_block] compact GPU execution is not implemented in the Stage-1 CPU port"
            )
    );
    let gpu_update_network = recursive_test_network(
        1,
        1,
        "loops=2\nouroboros=1\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        "",
    );
    let mut gpu = GpuExecutableNetwork::new(gpu_update_network, FakeBackend);
    assert!(
        gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
            .unwrap_err()
            .to_string()
            .contains("[recursive_block] GPU updates are not implemented in the Stage-1 CPU port")
    );
}
