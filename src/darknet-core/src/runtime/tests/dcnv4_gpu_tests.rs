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
fn dcnv4_bn_gpu_uses_native_path_before_stats_update() {
    let network = dcnv4_network("batch_normalize=1", "");
    let gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    assert!(!gpu.requires_cpu_backward_fallback());
    let before = gpu.params.clone();
    let input = Tensor::filled(network.net.input_shape(), 0.2);
    let forward = gpu.forward_training_with_state(&input).unwrap();
    assert!(forward.cpu_forward.dcnv4_caches[0].is_none());
    assert_eq!(gpu.params, before, "forward must not advance rolling stats");
}

#[test]
fn dcnv4_bn_native_gradients_match_cpu_including_zero_gamma() {
    use darknet_cuda::F32TensorOpsBackend;
    let network = dcnv4_network("batch_normalize=1", "");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
        unreachable!()
    };
    p.biases.fill(7.0);
    p.offset_biases.fill(0.47);
    for (i, w) in p.offset_weights.iter_mut().enumerate() {
        *w = ((i % 9) as f32 - 4.0) * 0.001;
    }
    let bn = p.batch_norm.as_mut().unwrap();
    bn.biases.fill(9.0); // beta is deliberately unused by CPU DCNv4.
    bn.scales.copy_from_slice(&[0.0, -1.2]);
    bn.rolling_mean.copy_from_slice(&[0.2, -0.1]);
    bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    let shape = network.net.input_shape();
    let input = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i % 13) as f32 * 0.03 - 0.15)
            .collect(),
    )
    .unwrap();
    let cf = cpu.forward_training(&input).unwrap();
    let gf = gpu.forward_training_with_state(&input).unwrap();
    let seed = Tensor::from_data(
        network.layers[0].output_shape,
        (0..input.data.len())
            .map(|i| (i % 7) as f32 * 0.2 - 0.6)
            .collect(),
    )
    .unwrap();
    let want = cpu.backward(&input, &cf, &seed).unwrap();
    let LayerParams::Dcnv4(params) = &gpu.params[0] else {
        unreachable!()
    };
    let LayerKind::Dcnv4(spec) = &network.layers[0].kind else {
        unreachable!()
    };
    let device_seed = gpu
        .backend
        .upload_f32(
            &[
                seed.shape.batch,
                seed.shape.channels,
                seed.shape.height,
                seed.shape.width,
            ],
            &seed.data,
        )
        .unwrap();
    let got = super::super::dcnv4_gpu::backward(
        &gpu.backend,
        &gf.input_gpu,
        gf.gpu_layer_activation_inputs[0].as_ref().unwrap(),
        &device_seed,
        &network.layers[0],
        spec,
        params,
    )
    .unwrap();
    let LayerParams::Dcnv4(wp) = &want.param_gradients[0] else {
        unreachable!()
    };
    let dx = gpu.backend.download_f32(&got.input).unwrap();
    for (a, b) in [
        (&dx, &want.input_gradient.data),
        (&got.weights, &wp.offset_weights),
        (&got.offset_biases, &wp.offset_biases),
        (
            got.scales.as_ref().unwrap(),
            &wp.batch_norm.as_ref().unwrap().scales,
        ),
    ] {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 3e-5, "{x} vs {y}");
        }
    }
    assert_eq!(got.biases, vec![0.0; 2]);
    assert!(
        got.scales.as_ref().unwrap()[0].abs() > 1e-5,
        "gamma=0 still needs its derivative"
    );
    let before = gpu.params.clone();
    let seeds = [Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cf, &seeds).unwrap();
    gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
    let LayerParams::Dcnv4(want) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Dcnv4(got) = &gpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Dcnv4(old) = &before[0] else {
        unreachable!()
    };
    for (a, b) in [
        (
            &want.batch_norm.as_ref().unwrap().rolling_mean,
            &got.batch_norm.as_ref().unwrap().rolling_mean,
        ),
        (
            &want.batch_norm.as_ref().unwrap().rolling_variance,
            &got.batch_norm.as_ref().unwrap().rolling_variance,
        ),
    ] {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-6);
        }
    }
    assert_ne!(
        got.batch_norm.as_ref().unwrap().rolling_mean,
        old.batch_norm.as_ref().unwrap().rolling_mean
    );
    assert_eq!(got.biases, old.biases);
    assert_eq!(
        got.batch_norm.as_ref().unwrap().biases,
        old.batch_norm.as_ref().unwrap().biases
    );
}

#[test]
fn dcnv4_bn_native_updates_unseeded_stats_and_rejects_bad_bn_storage() {
    let network = dcnv4_network("batch_normalize=1", "");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
        unreachable!()
    };
    p.offset_biases.fill(0.47);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    let input = Tensor::filled(network.net.input_shape(), 0.2);
    let cf = cpu.forward_training(&input).unwrap();
    let gf = gpu.forward_training_with_state(&input).unwrap();
    cpu.backward_accumulate_seeded(&input, &cf, &[None])
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&gf, &[None]).unwrap();
    let LayerParams::Dcnv4(want) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Dcnv4(got) = &gpu.params[0] else {
        unreachable!()
    };
    for (x, y) in want
        .batch_norm
        .as_ref()
        .unwrap()
        .rolling_mean
        .iter()
        .zip(&got.batch_norm.as_ref().unwrap().rolling_mean)
    {
        assert!((x - y).abs() < 1e-6);
    }
    for slot in 0..5 {
        let mut bad = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        let LayerParams::Dcnv4(p) = &mut bad.params[0] else {
            unreachable!()
        };
        let bn = p.batch_norm.as_mut().unwrap();
        match slot {
            0 => {
                bn.biases.pop();
            }
            1 => {
                bn.scales.pop();
            }
            2 => {
                bn.rolling_mean.pop();
            }
            3 => {
                bn.rolling_variance.pop();
            }
            _ => {
                p.batch_norm = None;
            }
        }
        assert!(
            bad.forward_training_with_state(&input)
                .unwrap_err()
                .to_string()
                .contains("parameter storage mismatch")
        );
    }
}

#[test]
fn dcnv4_bn_native_preserves_optimizer_gates_but_advances_stats() {
    for key in ["dont_update=1", "train_only_bn=1", "burnin_update=2"] {
        for adam in [false, true] {
            let mut network = dcnv4_network(&format!("batch_normalize=1\n{key}"), "");
            network.net.adam = adam;
            network.net.burn_in = 10;
            let mut cpu = ExecutableNetwork::new(network.clone());
            let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
                unreachable!()
            };
            p.offset_biases.fill(0.47);
            let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
            gpu.params = cpu.params.clone();
            let original = gpu.params.clone();
            let input = Tensor::filled(network.net.input_shape(), 0.2);
            let cf = cpu.forward_training(&input).unwrap();
            let gf = gpu.forward_training_with_state(&input).unwrap();
            let seeds = [Some(Tensor::filled(network.layers[0].output_shape, 0.3))];
            cpu.backward_accumulate_seeded(&input, &cf, &seeds).unwrap();
            gpu.backward_accumulate_seeded_gpu(&gf, &seeds).unwrap();
            let bc = cpu.params.clone();
            let bg = gpu.params.clone();
            if adam {
                cpu.update_weights_adam(0.01, 2).unwrap();
            } else {
                cpu.update_weights(0.01, 0.3, 0.1, 2);
            }
            gpu.update_weights_gpu(0.01, 0.3, 0.1, 2).unwrap();
            assert_eq!(cpu.params, bc, "CPU gate {key}");
            assert_eq!(gpu.params, bg, "GPU gate {key}");
            let LayerParams::Dcnv4(old) = &original[0] else {
                unreachable!()
            };
            let LayerParams::Dcnv4(now) = &gpu.params[0] else {
                unreachable!()
            };
            assert_ne!(
                old.batch_norm.as_ref().unwrap().rolling_mean,
                now.batch_norm.as_ref().unwrap().rolling_mean
            );
        }
    }
}

#[test]
fn dcnv4_channel_normalization_selects_native_math() {
    let mut network = dcnv4_network("", "");
    let LayerKind::Dcnv4(spec) = &mut network.layers[0].kind else { unreachable!() };
    spec.activation=Activation::NormalizeChannelsSoftmaxMaxval;
    let gpu=GpuExecutableNetwork::new(network.clone(),FakeBackend);
    assert!(!gpu.requires_cpu_backward_fallback());
    let input=Tensor::filled(network.net.input_shape(),0.2);
    let forward=gpu.forward_training_with_state(&input).unwrap();
    assert!(forward.cpu_forward.dcnv4_caches[0].is_none());
}

#[test]
fn dcnv4_channel_normalization_modes_and_edge_columns_match_reference() {
    use super::super::{GradMode, dcnv4_gpu};
    for activation in [Activation::NormalizeChannels,Activation::NormalizeChannelsSoftmax,Activation::NormalizeChannelsSoftmaxMaxval] {
        let mut network=dcnv4_network("", "");
        let LayerKind::Dcnv4(spec)=&mut network.layers[0].kind else { unreachable!() };
        spec.activation=activation;
        let LayerKind::Dcnv4(spec)=&network.layers[0].kind else { unreachable!() };
        let norm=spec.activation.to_channel_normalization().unwrap();
        let gpu=GpuExecutableNetwork::new(network.clone(),FakeBackend);
        let shape=network.layers[0].output_shape;
        let spatial=shape.height*shape.width;
        let mut values=vec![0.0;shape.batch*shape.element_count()];
        for b in 0..shape.batch { for c in 0..shape.channels { for q in 0..spatial {
            values[(b*shape.channels+c)*spatial+q]=match q%4 {0=>0.0,1=>-1.0,2=>0.5,_=>if c==0 {-0.7} else {1.3}};
        }}}
        let seed:Vec<_>=(0..values.len()).map(|i| (i%7) as f32*0.3-0.9).collect();
        let sh=[shape.batch,shape.channels,shape.height,shape.width];
        let pre=gpu.backend.upload_f32(&sh,&values).unwrap();
        let s=gpu.backend.upload_f32(&sh,&seed).unwrap();
        let output=dcnv4_gpu::activation_output(&gpu.backend,&pre,&network.layers[0],spec).unwrap();
        let want=darknet_cuda::host_ops::normalize_channels(&values,shape.batch,shape.channels,spatial,norm);
        assert_eq!(gpu.backend.download_f32(&output).unwrap(),want);
        for mode in [GradMode::Analytical,GradMode::Cpp] {
            let g=dcnv4_gpu::activation_delta(&gpu.backend,&pre,&s,&network.layers[0],spec,mode).unwrap();
            let want=darknet_cuda::host_ops::normalize_channels_backward(&values,&want,&seed,
                shape.batch,shape.channels,spatial,norm,mode==GradMode::Cpp);
            assert_eq!(gpu.backend.download_f32(&g).unwrap(),want);
        }
    }
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
fn dcnv4_gpu_retains_host_cache_for_excluded_and_mixed_variants() {
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
    spec.activation = Activation::Revleaky;
    for network in [
        large,
        normalized,
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
