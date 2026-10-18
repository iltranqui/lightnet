use super::*;

fn dcnv4_network(extra: &str, tail: &str) -> Network {
    Network::from_config(
        ConfigFile::parse_str(
            "dcn-gpu.cfg",
            &format!(
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\ntry_fix_nan=0\n\
         [dcnv4]\nsize=3\npadding=1\ngroups=2\nd_stride=2\nsoftmax=1\n\
         remove_center=1\nactivation=linear\n{extra}\n{tail}"
            ),
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn dcnv4_gpu_selects_native_math_without_host_cache() {
    let network = dcnv4_network("", "");
    let gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = Tensor::filled(network.net.input_shape(), 0.2);
    let forward = gpu.forward_training_with_state(&input).unwrap();
    assert!(forward.cpu_forward.dcnv4_caches[0].is_none());
}

#[test]
fn dcnv4_gpu_rejects_parameter_storage_before_backend_math() {
    let network = dcnv4_network("", "");
    for block in 0..3 {
        let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        let LayerParams::Dcnv4(params) = &mut gpu.params[0] else {
            unreachable!()
        };
        match block {
            0 => {
                params.offset_weights.pop();
            }
            1 => {
                params.offset_biases.pop();
            }
            _ => {
                params.biases.pop();
            }
        }
        let input = Tensor::filled(network.net.input_shape(), 0.2);
        assert!(
            gpu.forward_training_with_state(&input)
                .unwrap_err()
                .to_string()
                .contains("DCNv4 parameter storage mismatch")
        );
    }
}

#[test]
fn dcnv4_gpu_multiple_forward_states_and_microbatch_accumulation_match_cpu() {
    let network = dcnv4_network("", "");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
        unreachable!()
    };
    for (i, w) in p.offset_weights.iter_mut().enumerate() {
        *w = ((i % 9) as f32 - 4.0) * 0.001;
    }
    p.offset_biases.fill(0.47);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    let a = Tensor::filled(network.net.input_shape(), 0.2);
    let b = Tensor::from_data(
        a.shape,
        a.data
            .iter()
            .enumerate()
            .map(|(i, _)| (i % 13) as f32 * 0.03 - 0.15)
            .collect(),
    )
    .unwrap();
    let ca = cpu.forward_training(&a).unwrap();
    let cb = cpu.forward_training(&b).unwrap();
    let ga = gpu.forward_training_with_state(&a).unwrap();
    let gb = gpu.forward_training_with_state(&b).unwrap();
    let seeds = [Some(
        Tensor::from_data(
            network.layers[0].output_shape,
            (0..a.data.len())
                .map(|i| (i % 7) as f32 * 0.2 - 0.6)
                .collect(),
        )
        .unwrap(),
    )];
    for (input, cf, gf) in [(&a, &ca, &ga), (&b, &cb, &gb)] {
        cpu.backward_accumulate_seeded(input, cf, &seeds).unwrap();
        gpu.backward_accumulate_seeded_gpu(gf, &seeds).unwrap();
        let LayerTrainBuffers::Dcnv4(want) = &cpu.train_buffers[0] else {
            unreachable!()
        };
        let buffers = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::Dcnv4(got) = &buffers[0] else {
            unreachable!()
        };
        for (a, b) in [
            (&want.bias_updates, &got.bias_updates),
            (&want.offset_weight_updates, &got.offset_weight_updates),
            (&want.offset_bias_updates, &got.offset_bias_updates),
        ] {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 1e-5, "{x} vs {y}");
            }
        }
    }
}

#[test]
fn dcnv4_gpu_and_host_only_deform_choose_the_same_whole_network_path() {
    let network = dcnv4_network(
        "",
        "[deform_conv]\nfilters=2\nsize=1\nbatch_normalize=1\nactivation=linear\n",
    );
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    assert!(super::super::has_host_only_deform(&network));
    let input = Tensor::filled(network.net.input_shape(), 0.2);
    let forward = gpu.forward_training_with_state(&input).unwrap();
    assert!(forward.cpu_forward.dcnv4_caches[0].is_some());
    let seeds = [
        Some(Tensor::filled(network.layers[0].output_shape, 0.5)),
        None,
    ];
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .unwrap();
}

#[test]
fn dcnv4_gpu_retains_host_cache_for_bn_and_mixed_fallback() {
    let mut large = dcnv4_network("", "");
    let LayerKind::Dcnv4(spec) = &mut large.layers[0].kind else {
        unreachable!()
    };
    spec.size = 7;
    spec.padding = 3;
    let mut normalized = dcnv4_network("", "");
    let LayerKind::Dcnv4(spec) = &mut normalized.layers[0].kind else {
        unreachable!()
    };
    spec.activation = Activation::NormalizeChannels;
    for network in [
        large,
        normalized,
        dcnv4_network("batch_normalize=1", ""),
        dcnv4_network(
            "",
            "[deconvolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        ),
    ] {
        let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        assert!(gpu.requires_cpu_backward_fallback());
        let input = Tensor::filled(network.net.input_shape(), 0.2);
        let forward = gpu.forward_training_with_state(&input).unwrap();
        assert!(forward.cpu_forward.dcnv4_caches[0].is_some());
        let mut seeds = vec![None; network.layers.len()];
        seeds[0] = Some(Tensor::filled(network.layers[0].output_shape, 0.4));
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .unwrap();
    }
}
