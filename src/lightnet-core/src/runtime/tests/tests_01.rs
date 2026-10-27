use super::*;

pub(super) fn fill_wmhf_test_params(executable: &mut ExecutableNetwork) {
    let LayerParams::Wmhf(params) = &mut executable.params[0] else {
        panic!("wmhf params")
    };
    for (block, conv) in [
        &mut params.pre,
        &mut params.local_dw3,
        &mut params.local_dw5,
        &mut params.local_dw7,
        &mut params.local_mix,
        &mut params.hf_gate,
        &mut params.fuse,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, value) in conv.weights.iter_mut().enumerate() {
            *value = 0.025 + block as f32 * 0.004 + (index % 9) as f32 * 0.001;
        }
        for (index, value) in conv.biases.iter_mut().enumerate() {
            *value = 0.03 + block as f32 * 0.006 + index as f32 * 0.002;
        }
        if let Some(batch_norm) = &mut conv.batch_norm {
            batch_norm.scales.fill(0.8 + block as f32 * 0.015);
            batch_norm.rolling_mean.fill(0.23);
            batch_norm.rolling_variance.fill(1.3);
        }
    }
    let global = params.scan_weights.len() / 4;
    params.scan_weights[..global].fill(0.17);
    params.scan_weights[global..2 * global].fill(0.4);
    params.scan_weights[2 * global..3 * global].fill(0.7);
    params.scan_weights[3 * global..].fill(-0.2);
}

pub(super) fn wmhf_test_input(batch: usize) -> Tensor {
    Tensor::from_data(
        TensorShape::new(3, 3, 2, batch),
        (0..18 * batch)
            .map(|index| {
                0.17 + ((index * 7 + index / 5) % 19) as f32 * 0.071 + (index / 9) as f32 * 0.013
            })
            .collect(),
    )
    .unwrap()
}

pub(super) fn wmhf_test_seed(shape: TensorShape) -> Tensor {
    Tensor::from_data(
        shape,
        (0..shape.element_count() * shape.batch)
            .map(|index| ((index % 7) as f32 - 2.5) * 0.11 + 0.07)
            .collect(),
    )
    .unwrap()
}

pub(super) fn wmhf_objective(executable: &ExecutableNetwork, input: &Tensor, seed: &Tensor) -> f32 {
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

pub(super) fn check_wmhf_scalar_finite_difference(
    executable: &mut ExecutableNetwork,
    input: &Tensor,
    seed: &Tensor,
    name: &str,
    original: f32,
    analytical: f32,
    mut set: impl FnMut(&mut ExecutableNetwork, f32),
) {
    let eps = 1.5e-3;
    set(executable, original + eps);
    let plus = wmhf_objective(executable, input, seed);
    set(executable, original - eps);
    let minus = wmhf_objective(executable, input, seed);
    set(executable, original);
    let numerical = (plus - minus) / (2.0 * eps);
    let tolerance = 4.0e-3 + 2.5e-2 * numerical.abs().max(analytical.abs());
    assert!(
        (analytical - numerical).abs() <= tolerance,
        "[wmhf] {name}: analytical {analytical}, numerical {numerical}, tolerance {tolerance}"
    );
}

pub(super) fn repo_cfg(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../cfg")
        .join(name)
}

#[test]
fn vit_runtime_forward_backward_matches_input_and_parameter_finite_differences() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "vit-runtime.cfg",
            "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
             [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\nflash_attention=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Vit(weights) = &mut executable.params[0] else {
        panic!("Vit params")
    };
    let fill = |v: &mut [f32], scale: f32| {
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i % 7) as f32 - 3.0) * scale;
        }
    };
    fill(&mut weights.patch_embed, 0.025);
    fill(&mut weights.patch_bias, 0.01);
    fill(&mut weights.qkv_weight, 0.02);
    fill(&mut weights.qkv_bias, 0.01);
    weights.ln1_gamma.fill(1.0);
    fill(&mut weights.ln1_beta, 0.005);
    fill(&mut weights.wo, 0.02);
    fill(&mut weights.wo_bias, 0.01);
    weights.ln2_gamma.fill(1.0);
    fill(&mut weights.ln2_beta, 0.005);
    fill(&mut weights.ffn_w1, 0.02);
    fill(&mut weights.ffn_b1, 0.01);
    fill(&mut weights.ffn_w2, 0.02);
    fill(&mut weights.ffn_b2, 0.01);
    fill(&mut weights.pos_embed, 0.005);
    weights.mhc = crate::vit::MHC_DEFAULTS.to_vec();

    let input = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        (0..16).map(|i| (i as f32 - 7.0) * 0.07).collect(),
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    let output = forward.final_output().unwrap();
    let seed = Tensor::from_data(
        output.shape,
        (0..output.len()).map(|i| (i as f32 + 1.0) * 0.13).collect(),
    )
    .unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    let objective = |out: &Tensor| {
        out.data
            .iter()
            .zip(&seed.data)
            .map(|(a, b)| a * b)
            .sum::<f32>()
    };
    let eps = 1e-3;

    let mut plus_input = input.clone();
    plus_input.data[5] += eps;
    let mut minus_input = input.clone();
    minus_input.data[5] -= eps;
    let numerical_input = (objective(
        executable
            .forward(&plus_input)
            .unwrap()
            .final_output()
            .unwrap(),
    ) - objective(
        executable
            .forward(&minus_input)
            .unwrap()
            .final_output()
            .unwrap(),
    )) / (2.0 * eps);
    assert!(
        (backward.input_gradient.data[5] - numerical_input).abs() < 2e-3,
        "input gradient: analytic={} numeric={numerical_input}",
        backward.input_gradient.data[5]
    );

    let LayerParams::Vit(analytic) = &backward.param_gradients[0] else {
        panic!("Vit gradients")
    };
    let numerical = |mut x: ExecutableNetwork, index: usize, ffn: bool| {
        {
            let LayerParams::Vit(w) = &mut x.params[0] else {
                panic!("Vit params")
            };
            let values = if ffn {
                &mut w.ffn_w1
            } else {
                &mut w.patch_embed
            };
            values[index] += eps;
        }
        let p = objective(x.forward(&input).unwrap().final_output().unwrap());
        {
            let LayerParams::Vit(w) = &mut x.params[0] else {
                panic!("Vit params")
            };
            let values = if ffn {
                &mut w.ffn_w1
            } else {
                &mut w.patch_embed
            };
            values[index] -= 2.0 * eps;
        }
        let m = objective(x.forward(&input).unwrap().final_output().unwrap());
        (p - m) / (2.0 * eps)
    };
    for (analytic, numeric, name) in [
        (
            analytic.patch_embed[3],
            numerical(executable.clone(), 3, false),
            "patch_embed",
        ),
        (
            analytic.ffn_w1[2],
            numerical(executable.clone(), 2, true),
            "ffn_w1",
        ),
    ] {
        assert!(
            (analytic - numeric).abs() < 2e-3,
            "{name} gradient: analytic={analytic} numeric={numeric}"
        );
    }

    let pre_update = match &executable.params[0] {
        LayerParams::Vit(w) => w.patch_embed[3],
        _ => unreachable!(),
    };
    executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    let LayerTrainBuffers::Vit(train) = &executable.train_buffers[0] else {
        panic!("Vit updates")
    };
    let clip_scale = super::vit_update_clip_scale(&train.updates, 5.0, true);
    assert!((train.updates.patch_embed[3] + analytic.patch_embed[3]).abs() < 1e-6);
    executable.update_weights(0.01, 0.0, 0.0, 1);
    let LayerParams::Vit(updated) = &executable.params[0] else {
        panic!("Vit params")
    };
    assert!(
        (updated.patch_embed[3] - (pre_update - 0.01 * analytic.patch_embed[3] * clip_scale)).abs()
            < 1e-6
    );

    let mut gpu = GpuExecutableNetwork::new(executable.network.clone(), FakeBackend);
    gpu.params = executable.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).unwrap();
    let gpu_output = gpu_forward.cpu_forward.final_output().unwrap();
    let cpu_output = executable.forward(&input).unwrap();
    // Inference takes the native device path, whose reductions order floats
    // differently from the host reference.
    let cpu_output = cpu_output.final_output().unwrap();
    assert_eq!(gpu_output.shape, cpu_output.shape);
    for (g, c) in gpu_output.data.iter().zip(&cpu_output.data) {
        assert!((g - c).abs() <= 1e-5 * c.abs().max(1.0), "gpu={g} cpu={c}");
    }
    // Flash `[vit]` with a mapped activation takes the native device backward:
    // the training forward keeps a device cache and the backward stages the
    // same parameter gradients the CPU reference returns (here through the
    // fake backend's host reference ops).
    assert!(!gpu.requires_cpu_backward_fallback());
    let cpu_forward = executable.forward(&input).unwrap();
    let cpu_backward = executable.backward(&input, &cpu_forward, &seed).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &[Some(seed.clone())])
        .unwrap()
        .expect("captured ViT input gradient");
    for (g, c) in gpu_input_gradient
        .data
        .iter()
        .zip(&cpu_backward.input_gradient.data)
    {
        assert!((g - c).abs() <= 1e-4 * c.abs().max(1.0), "input grad gpu={g} cpu={c}");
    }
    let LayerParams::Vit(cpu_grad) = &cpu_backward.param_gradients[0] else {
        panic!("Vit gradients")
    };
    gpu.synchronize_resident_parameter_gradients().unwrap();
    let buffers = gpu.host_fallback_train_buffers();
    let LayerTrainBuffers::Vit(train) = &buffers[0] else {
        panic!("Vit native-staged updates")
    };
    for (i, (u, c)) in train
        .updates
        .to_blob()
        .iter()
        .zip(cpu_grad.to_blob())
        .enumerate()
    {
        assert!(
            (u + c).abs() <= 1e-4 * c.abs().max(1.0),
            "param blob[{i}]: staged update {u}, CPU gradient {c}"
        );
    }
}

#[test]
fn vit_sgd_update_applies_decay_and_reduced_mhc_rate() {
    let network = Network::from_config(ConfigFile::parse_str(
            "vit-update.cfg",
            "[net]\nwidth=4\nheight=4\nchannels=1\n[vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\n",
        ).unwrap()).unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Vit(params) = &mut executable.params[0] else {
        panic!("Vit params")
    };
    params.patch_embed[0] = 1.0;
    params.pos_embed[0] = 1.0;
    let mhc_before = params.mhc[0];
    let LayerTrainBuffers::Vit(train) = &mut executable.train_buffers[0] else {
        panic!("Vit updates")
    };
    train.updates.patch_embed[0] = -0.5;
    train.updates.patch_bias[0] = -0.5;
    train.updates.mhc[0] = -0.5;
    executable.update_weights(0.1, 0.0, 0.2, 1);
    let LayerParams::Vit(params) = &executable.params[0] else {
        panic!("Vit params")
    };
    assert!((params.patch_embed[0] - 0.93).abs() < 1e-6); // matrix decay
    assert!((params.patch_bias[0] + 0.05).abs() < 1e-6); // no bias decay
    assert!((params.pos_embed[0] - 0.98).abs() < 1e-6); // learned positions decay
    assert!((params.mhc[0] - (mhc_before - 0.005)).abs() < 1e-6); // 0.1x LR
}

#[test]
fn tucker_attention_cpu_and_fake_backend_forward_backward_and_sgd_match() {
    // Revleaky deliberately selects the documented unmapped-activation fallback.
    let network = Network::from_config(
        ConfigFile::parse_str(
            "tucker-runtime.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\n\
                 [tucker_attention]\nsize=2\nheads=1\nrank=1\nrank_k=2\nrank_o=2\n\
                 activation=revleaky\nmsa=1\nmsa_block=1\nmsa_topk=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::TuckerAttention(params) = &mut cpu.params[0] else {
        panic!("Tucker attention params")
    };
    for (index, weight) in params.weights.iter_mut().enumerate() {
        *weight = (index as f32 * 0.071).sin() * 0.4 + 0.2;
    }
    params.biases.copy_from_slice(&[0.03, -0.02]);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(gpu.requires_cpu_backward_fallback());

    let input = Tensor::from_data(
        TensorShape::new(2, 2, 2, 1),
        vec![-0.31, 0.22, 0.48, -0.17, 0.15, 0.33, -0.42, 0.27],
    )
    .unwrap();
    let cpu_forward = cpu.forward(&input).unwrap();
    assert!(cpu_forward.tucker_attention_caches[0].is_some());
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    let output = cpu_forward.final_output().unwrap();
    let seed = Tensor::filled(output.shape, 0.25);
    let seeds = [Some(seed)];
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let LayerParams::TuckerAttention(gradient) = &backward.param_gradients[0] else {
        panic!("Tucker attention gradients")
    };
    assert!(gradient.weights.iter().any(|gradient| *gradient != 0.0));
    assert!(gradient.biases.iter().any(|gradient| *gradient != 0.0));
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    assert_eq!(
        cpu.train_buffers,
        gpu.cpu_fallback_train_buffers.borrow().clone()
    );

    let before = cpu.params[0].clone();
    cpu.update_weights(0.01, 0.2, 0.1, 1);
    gpu.update_weights_gpu(0.01, 0.2, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    let LayerParams::TuckerAttention(before) = before else {
        panic!("Tucker attention params")
    };
    let LayerParams::TuckerAttention(after) = &cpu.params[0] else {
        panic!("Tucker attention params")
    };
    assert_ne!(before.weights, after.weights);
    assert_ne!(before.biases, after.biases);
}

#[test]
fn mambavision_runtime_and_fake_backend_host_fallback_match() {
    // Keep this host-cache/state test on an explicitly unsupported device activation.
    let network = Network::from_config(
        ConfigFile::parse_str(
            "mambavision-runtime.cfg",
            "[net]\nwidth=2\nheight=1\nchannels=2\nbatch=1\n\
                 [mambavision]\nstate=1\nconv_size=1\ndt_rank=1\nffn_ratio=1\nactivation=revleaky\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    crate::initialize_training_params(&network, &mut cpu.params, 107);
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(gpu.requires_cpu_backward_fallback());

    let input =
        Tensor::from_data(TensorShape::new(2, 1, 2, 1), vec![0.21, -0.31, 0.47, 0.16]).unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    assert!(cpu_forward.mambavision_caches[0].is_some());
    let incoming_state = crate::MambaVisionState { values: vec![0.19] };
    let state_list = [Some(incoming_state.clone())];
    let stateful = cpu
        .forward_with_mambavision_states(&input, &state_list)
        .unwrap();
    assert_eq!(
        stateful.mambavision_caches[0]
            .as_ref()
            .unwrap()
            .state_out
            .values
            .len(),
        1
    );
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    let seed = Tensor::filled(cpu_forward.final_output().unwrap().shape, 0.25);
    let seeds = [Some(seed)];
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    assert!(backward.mambavision_state_gradients[0].is_some());
    let LayerParams::MambaVision(gradient) = &backward.param_gradients[0] else {
        panic!("MambaVision gradients")
    };
    assert!(gradient.values.iter().any(|value| *value != 0.0));
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
fn dcnv4_runtime_backward_and_fake_backend_native_defaults_match() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "dcnv4-runtime.cfg",
            "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n\
                 [dcnv4]\nsize=1\npadding=0\nd_stride=2\nsoftmax=0\
                 \noffset_scale=0.6\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(params) = &mut cpu.params[0] else {
        panic!("DCNv4 params")
    };
    params.offset_weights[0] = 0.002;
    params.offset_weights[1] = 0.003;
    params.offset_weights[2] = 0.001;
    params.offset_biases[0] = 0.13;
    params.offset_biases[1] = 0.19;
    params.offset_biases[2] = 0.8;
    params.biases[0] = 0.07;
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());

    let input = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        (0..16).map(|index| 0.3 + index as f32 * 0.037).collect(),
    )
    .unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    assert!(cpu_forward.dcnv4_caches[0].is_some());
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    assert!(gpu_forward.cpu_forward.dcnv4_caches[0].is_none());

    let seed = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        (0..16)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.21)
            .collect(),
    )
    .unwrap();
    let seeds = [Some(seed.clone())];
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let LayerParams::Dcnv4(gradient) = &backward.param_gradients[0] else {
        panic!("DCNv4 gradients")
    };
    let objective = |input: &Tensor, params: &[LayerParams]| {
        let mut executable = ExecutableNetwork::new(network.clone());
        executable.params = params.to_vec();
        let output = executable
            .forward(input)
            .unwrap()
            .final_output()
            .unwrap()
            .clone();
        output
            .data
            .iter()
            .zip(&seed.data)
            .map(|(value, grad)| value * grad)
            .sum::<f32>()
    };
    let epsilon = 1.0e-3;
    // Check input, offset projection, offset bias, and ordinary channel bias.
    let mut plus_input = input.clone();
    plus_input.data[5] += epsilon;
    let mut minus_input = input.clone();
    minus_input.data[5] -= epsilon;
    let numerical_input = (objective(&plus_input, &cpu.params)
        - objective(&minus_input, &cpu.params))
        / (2.0 * epsilon);
    assert!((backward.input_gradient.data[5] - numerical_input).abs() < 3.0e-3);
    for (actual, block, index) in [
        (gradient.offset_weights[0], 0, 0),
        (gradient.offset_biases[0], 1, 0),
        (gradient.biases[0], 2, 0),
    ] {
        let mut plus = cpu.params.clone();
        let mut minus = cpu.params.clone();
        let perturb = |params: &mut [LayerParams], delta: f32| {
            let LayerParams::Dcnv4(dcnv4) = &mut params[0] else {
                panic!("DCNv4 params")
            };
            match block {
                0 => dcnv4.offset_weights[index] += delta,
                1 => dcnv4.offset_biases[index] += delta,
                _ => dcnv4.biases[index] += delta,
            }
        };
        perturb(&mut plus, epsilon);
        perturb(&mut minus, -epsilon);
        let fd = (objective(&input, &plus) - objective(&input, &minus)) / (2.0 * epsilon);
        assert!(
            (actual - fd).abs() < 3.0e-3,
            "gradient mismatch: {actual} vs {fd}"
        );
    }

    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    let LayerTrainBuffers::Dcnv4(cpu_train) = &cpu.train_buffers[0] else {
        panic!("DCNv4 train buffers")
    };
    {
        let device_train = gpu.device_dcnv4_train_buffers.borrow();
        let device_train = device_train[0].as_ref().unwrap();
        assert_eq!(
            gpu.backend
                .download_f32(&device_train.offset_weight_updates)
                .unwrap(),
            cpu_train.offset_weight_updates
        );
        assert_eq!(
            gpu.backend
                .download_f32(&device_train.offset_bias_updates)
                .unwrap(),
            cpu_train.offset_bias_updates
        );
        assert_eq!(
            gpu.backend
                .download_f32(&device_train.bias_updates)
                .unwrap(),
            cpu_train.bias_updates
        );
    }
    cpu.update_weights(0.01, 0.2, 0.1, 1);
    gpu.update_weights_gpu(0.01, 0.2, 0.1, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    let LayerParams::Dcnv4(after) = &cpu.params[0] else {
        panic!("DCNv4 params")
    };
    assert!(after.dummy_weights.iter().all(|value| *value == 0.0));
}

#[test]
fn region_axis_layout_matches_cpp_flatten_objectness_and_classes() {
    use crate::axis_fixture::axis_tensor;

    // C++ `forward_region_layer` copies the NCHW source, applies
    // `flatten(w*h, num*(coords+classes+1), batch, 1)`, activates each
    // objectness at entry + 4, and optionally softmaxes classes from +5.
    // The non-square, two-sample fixture makes both the permutation and
    // independent sample rows observable.
    let shape = TensorShape::new(3, 2, 7, 2);
    let network = Network::from_config(
        ConfigFile::parse_str(
            "region-axis.cfg",
            "[net]\nwidth=3\nheight=2\nchannels=7\nbatch=2\n\
                 [region]\nclasses=2\ncoords=4\nnum=1\nsoftmax=1\n",
        )
        .unwrap(),
    )
    .unwrap();
    let cpu = ExecutableNetwork::new(network);
    let input_data: Vec<f32> = axis_tensor(shape)
        .into_iter()
        .map(|value| value / 1000.0)
        .collect();
    let input = Tensor::from_data(shape, input_data.clone()).unwrap();
    let forward = cpu.forward(&input).unwrap();
    let output = &forward.layer_outputs[0];
    let spatial = shape.width * shape.height;
    let channels = shape.channels;
    let plane = spatial * channels;

    for batch in 0..shape.batch {
        for location in 0..spatial {
            let source = |channel: usize| input_data[batch * plane + channel * spatial + location];
            let maximum = source(5).max(source(6));
            let p5 = (source(5) - maximum).exp();
            let p6 = (source(6) - maximum).exp();
            let class_sum = p5 + p6;
            let probabilities = [p5 / class_sum, p6 / class_sum];
            for channel in 0..channels {
                let expected = match channel {
                    4 => 1.0 / (1.0 + (-source(4)).exp()),
                    5 => probabilities[0],
                    6 => probabilities[1],
                    _ => source(channel),
                };
                let index = batch * plane + location * channels + channel;
                assert!(
                    (output.data[index] - expected).abs() < 1e-6,
                    "Region C++ flatten must map sample {batch}, spatial {location}, channel {channel}; got {}, expected {expected}",
                    output.data[index]
                );
            }
        }
    }

    // The supplied-seed backward is the exact adjoint of the Rust/C++
    // forward transform (the truth-driven Region loss remains separate).
    let seed_data: Vec<f32> = (0..plane * shape.batch)
        .map(|index| 0.11 + index as f32 * 0.007)
        .collect();
    let seed = Tensor::from_data(shape, seed_data.clone()).unwrap();
    let backward = cpu.backward(&input, &forward, &seed).unwrap();
    for batch in 0..shape.batch {
        for location in 0..spatial {
            let output_base = batch * plane + location * channels;
            let objectness = output.data[output_base + 4];
            let class_dot = output.data[output_base + 5] * seed_data[output_base + 5]
                + output.data[output_base + 6] * seed_data[output_base + 6];
            for channel in 0..channels {
                let output_index = output_base + channel;
                let input_index = batch * plane + channel * spatial + location;
                let expected = match channel {
                    4 => seed_data[output_index] * objectness * (1.0 - objectness),
                    5 | 6 => {
                        let probability = output.data[output_index];
                        probability * (seed_data[output_index] - class_dot)
                    }
                    _ => seed_data[output_index],
                };
                assert!(
                    (backward.input_gradient.data[input_index] - expected).abs() < 1e-6,
                    "Region adjoint must unflatten sample {batch}, spatial {location}, channel {channel}"
                );
            }
        }
    }
}

#[test]
fn convolutional_axis_layout_keeps_nchw_and_group_partitions() {
    use crate::axis_fixture::{axis_tensor, axis_value, nchw_index};

    // Two-channel 1x1 convolution whose weight matrix swaps the channels:
    // `[out][in][kh][kw]` = [[0,1],[1,0]]. Output channel 0 must then carry
    // input channel 1 at the same `(n,h,w)`, which pins both the NCHW order
    // and the weight axis.
    let network = Network::from_config(
            ConfigFile::parse_str(
                "conv-axis.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network);
    let LayerParams::Convolutional(params) = &mut cpu.params[0] else {
        panic!("convolutional params")
    };
    params.weights = vec![0.0, 1.0, 1.0, 0.0];
    params.biases = vec![0.0, 0.0];

    let shape = TensorShape::new(3, 2, 2, 2);
    let input = Tensor::from_data(shape, axis_tensor(shape)).unwrap();
    let output = cpu.forward(&input).unwrap().layer_outputs[0].clone();
    for n in 0..shape.batch {
        for h in 0..shape.height {
            for w in 0..shape.width {
                assert_eq!(
                    output.data[nchw_index(shape, n, 0, h, w)],
                    axis_value(n, 1, h, w),
                    "output channel 0 at ({n},{h},{w}) must read input channel 1"
                );
                assert_eq!(
                    output.data[nchw_index(shape, n, 1, h, w)],
                    axis_value(n, 0, h, w),
                    "output channel 1 at ({n},{h},{w}) must read input channel 0"
                );
            }
        }
    }

    // `groups=2` partitions channels instead of mixing them: output 0 sees
    // only input 0 and output 1 only input 1, per sample.
    let grouped = Network::from_config(
            ConfigFile::parse_str(
                "conv-groups-axis.cfg",
                "[net]\nwidth=2\nheight=1\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\ngroups=2\nbatch_normalize=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut grouped_cpu = ExecutableNetwork::new(grouped);
    let LayerParams::Convolutional(params) = &mut grouped_cpu.params[0] else {
        panic!("grouped convolutional params")
    };
    params.weights = vec![2.0, 3.0];
    params.biases = vec![0.0, 0.0];
    let shape = TensorShape::new(2, 1, 2, 2);
    let input = Tensor::from_data(shape, axis_tensor(shape)).unwrap();
    let output = grouped_cpu.forward(&input).unwrap().layer_outputs[0].clone();
    for n in 0..shape.batch {
        for w in 0..shape.width {
            assert_eq!(
                output.data[nchw_index(shape, n, 0, 0, w)],
                2.0 * axis_value(n, 0, 0, w),
                "group 0 must only see input channel 0 at ({n},{w})"
            );
            assert_eq!(
                output.data[nchw_index(shape, n, 1, 0, w)],
                3.0 * axis_value(n, 1, 0, w),
                "group 1 must only see input channel 1 at ({n},{w})"
            );
        }
    }
}

#[test]
fn deform_conv_backward_rejects_epilogue_clamp_boundary() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-clamp.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::DeformConv(params) = &mut executable.params[0] else {
        panic!("deform_conv params")
    };
    params.main.biases[0] = 10_000.0;
    let input = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![0.0]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    assert_eq!(forward.final_output().unwrap().data, vec![10_000.0]);
    let seeds = [Some(Tensor::filled(input.shape, 1.0))];
    assert!(
        executable
            .backward_accumulate_seeded(&input, &forward, &seeds)
            .is_err()
    );
}

#[test]
fn deform_conv_train_only_bn_gates_all_optimizer_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-train-only-bn.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=1\nbatch_normalize=1\ntrain_only_bn=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    assert!(network.layers[0].train.train_only_bn);
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::DeformConv(params) = &mut executable.params[0] else {
        panic!("deform_conv params")
    };
    params.main.weights[0] = 0.25;
    let before = executable.params.clone();
    let LayerTrainBuffers::DeformConv(train) = &mut executable.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    train.main.weight_updates.fill(1.0);
    train.main.bias_updates.fill(1.0);
    train.offset_weight_updates.fill(1.0);
    train.offset_bias_updates.fill(1.0);
    train.mask_weight_updates.as_mut().unwrap().fill(1.0);
    train.mask_bias_updates.as_mut().unwrap().fill(1.0);

    executable.update_weights(0.1, 0.9, 0.1, 1);
    assert_eq!(executable.params, before);
    let LayerTrainBuffers::DeformConv(train) = &executable.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    assert!(train.main.weight_updates.iter().all(|value| *value == 1.0));
    assert!(
        train
            .offset_weight_updates
            .iter()
            .all(|value| *value == 1.0)
    );
    assert!(
        train
            .mask_weight_updates
            .as_ref()
            .unwrap()
            .iter()
            .all(|value| *value == 1.0)
    );
}

#[test]
fn deform_conv_backward_accumulates_across_microbatches() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-accumulation.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::DeformConv(params) = &mut executable.params[0] else {
        panic!("deform_conv params")
    };
    params.main.weights[0] = 0.5;
    let input = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![2.0]).unwrap();
    let seed = [Some(Tensor::from_data(input.shape, vec![0.75]).unwrap())];

    let first_forward = executable.forward_training(&input).unwrap();
    executable
        .backward_accumulate_seeded(&input, &first_forward, &seed)
        .unwrap();
    let LayerTrainBuffers::DeformConv(train) = &executable.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    let first_weight_update = train.main.weight_updates[0];
    let first_bias_update = train.main.bias_updates[0];

    let second_forward = executable.forward_training(&input).unwrap();
    executable
        .backward_accumulate_seeded(&input, &second_forward, &seed)
        .unwrap();
    let LayerTrainBuffers::DeformConv(train) = &executable.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    assert_eq!(train.main.weight_updates[0], 2.0 * first_weight_update);
    assert_eq!(train.main.bias_updates[0], 2.0 * first_bias_update);
}

#[test]
fn deform_conv_compact_gpu_path_rejects_explicitly() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-compact.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![1.0]).unwrap();
    let error = gpu
        .forward_compact(&input)
        .expect_err("compact execution must reject deform_conv");
    assert!(
        error
            .to_string()
            .contains("[deform_conv] compact GPU execution is unsupported")
    );
}

#[test]
fn deform_conv_loss_scale_is_neutral_for_gpu_host_fallback_updates() {
    let run = |loss_scale: usize| {
        let network = Network::from_config(
                ConfigFile::parse_str(
                    "deform-conv-loss-scale.cfg",
                    &format!(
                        "[net]\nwidth=2\nheight=2\nchannels=1\nloss_scale={loss_scale}\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=0\nactivation=linear\n"
                    ),
                )
                .unwrap(),
            )
            .unwrap();
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        let LayerParams::DeformConv(params) = &mut gpu.params[0] else {
            panic!("deform_conv params")
        };
        params.main.weights[0] = 0.4;
        params.main.biases[0] = 0.05;
        params.offset_weights.copy_from_slice(&[0.02, -0.03]);
        params.offset_biases.copy_from_slice(&[0.07, -0.04]);
        let input =
            Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![0.1, 0.3, 0.5, 0.7]).unwrap();
        let forward = gpu.forward_training_with_state(&input).unwrap();
        let seeds = [Some(
            Tensor::from_data(input.shape, vec![0.2, -0.3, 0.4, 0.1]).unwrap(),
        )];
        gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
            .unwrap();
        gpu.update_weights_gpu(0.02, 0.0, 0.01, 1).unwrap();
        gpu.params
    };

    assert_eq!(run(1), run(8));
}

#[test]
fn deform_conv_sgd_uses_source_rate_groups_and_moves_downhill() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-sgd.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[deform_conv]\nfilters=1\nsize=1\nuse_mask=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::DeformConv(params) = &mut executable.params[0] else {
        panic!("deform_conv params")
    };
    params.main.weights[0] = 0.5;
    let input = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![1.0]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let initial = forward.final_output().unwrap().data[0];
    assert!((initial - 0.25).abs() < 1.0e-6);
    let seeds = [Some(Tensor::filled(input.shape, -initial))];
    executable
        .backward_accumulate_seeded(&input, &forward, &seeds)
        .unwrap();
    let LayerTrainBuffers::DeformConv(train) = &mut executable.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    assert!((train.main.weight_updates[0] + 0.125).abs() < 1.0e-6);
    train.offset_weight_updates[0] = 1.0;
    train.offset_bias_updates[0] = 1.0;
    train.mask_weight_updates.as_mut().unwrap()[0] = 1.0;
    train.mask_bias_updates.as_mut().unwrap()[0] = 1.0;

    executable.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::DeformConv(params) = &executable.params[0] else {
        panic!("deform_conv params")
    };
    assert!((params.main.weights[0] - 0.4875).abs() < 1.0e-6);
    assert!((params.main.biases[0] + 0.025).abs() < 1.0e-6);
    assert!((params.offset_weights[0] - 0.001).abs() < 1.0e-7);
    assert!((params.offset_biases[0] - 0.001).abs() < 1.0e-7);
    assert!((params.mask_weights.as_ref().unwrap()[0] - 0.001).abs() < 1.0e-7);
    assert!((params.mask_biases.as_ref().unwrap()[0] - 0.001).abs() < 1.0e-7);
    let updated = executable
        .forward(&input)
        .unwrap()
        .final_output()
        .unwrap()
        .data[0];
    assert!(
        0.5 * updated * updated < 0.5 * initial * initial,
        "Darknet update should descend: output {initial} -> {updated}"
    );
}

#[test]
fn graph_conv_runtime_nchw_backward_and_gpu_host_fallback_updates_match_cpu() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-runtime.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\n[graph_conv]\nfilters=1\nsize=3\npad=1\nedge_mode=0\ngraph_use_self=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("graph_conv params");
    };
    params.projection.weights[0] = 0.5;
    params.projection.biases[0] = 0.25;
    params.self_weights.as_mut().unwrap()[0] = 0.1;
    let input = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let forward = cpu.forward(&input).unwrap();
    assert_eq!(forward.graph_conv_caches.len(), 1);
    assert!(forward.graph_conv_caches[0].is_some());
    assert_eq!(forward.layer_outputs[0].data, vec![1.6, 1.7, 1.8, 1.9]);
    let seed = Tensor::from_data(forward.layer_outputs[0].shape, vec![1.0; 4]).unwrap();
    let bias_objective = |network: &mut ExecutableNetwork, bias: f32| {
        let LayerParams::GraphConv(params) = &mut network.params[0] else {
            panic!("graph_conv params");
        };
        params.projection.biases[0] = bias;
        network.forward(&input).unwrap().layer_outputs[0]
            .data
            .iter()
            .zip(&seed.data)
            .map(|(value, gradient)| value * gradient)
            .sum::<f32>()
    };
    let epsilon = 1.0e-3;
    let bias_plus = bias_objective(&mut cpu, 0.25 + epsilon);
    let bias_minus = bias_objective(&mut cpu, 0.25 - epsilon);
    let numeric_bias_gradient = (bias_plus - bias_minus) / (2.0 * epsilon);
    let _ = bias_objective(&mut cpu, 0.25);
    assert!((numeric_bias_gradient - 4.0).abs() < 1.0e-3);

    let backward = cpu.backward(&input, &forward, &seed).unwrap();
    assert_eq!(backward.input_gradient.data, vec![0.6; 4]);
    let LayerParams::GraphConv(gradient) = &backward.param_gradients[0] else {
        panic!("graph_conv gradient");
    };
    assert_eq!(gradient.projection.biases, vec![4.0]);
    assert_eq!(gradient.projection.weights, vec![10.0]);
    assert_eq!(gradient.self_weights.as_deref(), Some(&[10.0][..]));

    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.layer_outputs[0].data,
        forward.layer_outputs[0].data
    );
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let seeds = [Some(seed.clone())];
    cpu.backward_accumulate_seeded(&input, &forward, &seeds)
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    cpu.update_weights(0.01, 0.9, 0.0, 1);
    gpu.update_weights_gpu(0.01, 0.9, 0.0, 1).unwrap();
    let (LayerParams::GraphConv(cpu_params), LayerParams::GraphConv(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("GraphConv parameters expected after GPU update");
    };
    assert_eq!(cpu_params, gpu_params);
}

#[test]
fn graph_conv_backward_accumulates_into_preceding_convolution() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-chain.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[graph_conv]\nfilters=1\nsize=1\nedge_mode=0\ngraph_use_self=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Convolutional(convolution) = &mut executable.params[0] else {
        panic!("preceding convolution params");
    };
    convolution.weights[0] = 2.0;
    convolution.biases[0] = 0.1;
    let LayerParams::GraphConv(graph) = &mut executable.params[1] else {
        panic!("graph_conv params");
    };
    graph.projection.weights[0] = 3.0;
    graph.projection.biases[0] = 0.2;

    let input = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let forward = executable.forward(&input).unwrap();
    for (actual, expected) in forward.layer_outputs[1]
        .data
        .iter()
        .zip([6.5, 12.5, 18.5, 24.5])
    {
        assert!((actual - expected).abs() < 3.0e-6);
    }
    let seed =
        Tensor::from_data(forward.layer_outputs[1].shape, vec![0.5, 1.0, -0.5, 2.0]).unwrap();
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    assert_eq!(backward.input_gradient.data, vec![3.0, 6.0, -3.0, 12.0]);
    let LayerParams::GraphConv(graph) = &backward.param_gradients[1] else {
        panic!("graph_conv gradient");
    };
    assert!((graph.projection.weights[0] - 18.3).abs() < 1.0e-5);
    assert_eq!(graph.projection.biases, vec![3.0]);
    let LayerParams::Convolutional(convolution) = &backward.param_gradients[0] else {
        panic!("preceding convolution gradient");
    };
    assert!((convolution.weights[0] - 27.0).abs() < 1.0e-5);
    assert_eq!(convolution.biases, vec![9.0]);
}

#[test]
fn graph_conv_training_batchnorm_uses_batch_stats_and_updates_rolling_stats() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-bn.cfg",
                "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\n[graph_conv]\nfilters=1\nsize=1\nedge_mode=0\ngraph_use_self=0\nbatch_normalize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::GraphConv(params) = &mut executable.params[0] else {
        panic!("graph_conv params");
    };
    params.projection.weights[0] = 1.0;
    params.projection.biases[0] = 0.1;
    params.projection.batch_norm.as_mut().unwrap().scales[0] = 2.0;
    let input = Tensor::from_data(TensorShape::new(2, 1, 1, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let output = &forward.layer_outputs[0];
    let expected = [
        2.0 * (-1.5f32 / 1.25001f32.sqrt()) + 0.1,
        2.0 * (-0.5f32 / 1.25001f32.sqrt()) + 0.1,
        2.0 * (0.5f32 / 1.25001f32.sqrt()) + 0.1,
        2.0 * (1.5f32 / 1.25001f32.sqrt()) + 0.1,
    ];
    for (actual, expected) in output.data.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
    let seed = Tensor::from_data(output.shape, vec![1.0; output.len()]).unwrap();
    let backward = executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .all(|value| value.abs() < 1.0e-5)
    );
    let LayerParams::GraphConv(params) = &executable.params[0] else {
        panic!("graph_conv params");
    };
    let batch_norm = params.projection.batch_norm.as_ref().unwrap();
    assert!((batch_norm.rolling_mean[0] - 0.25).abs() < 1.0e-6);
    assert!((batch_norm.rolling_variance[0] - 0.125).abs() < 1.0e-6);
    let LayerTrainBuffers::GraphConv(train) = &executable.train_buffers[0] else {
        panic!("graph_conv train buffers");
    };
    assert_eq!(train.projection.bias_updates, vec![4.0]);
    assert!(train.projection.batch_norm.as_ref().unwrap().scale_updates[0].abs() < 1.0e-5);
}

#[test]
fn graph_conv_batchnorm_inference_uses_gpu_backend_without_cpu_cache() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-bn-inference.cfg",
                "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=2\n[graph_conv]\nfilters=1\nsize=1\nedge_mode=0\ngraph_use_self=0\nbatch_normalize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::GraphConv(params) = &mut cpu.params[0] else {
        panic!("graph_conv params");
    };
    params.projection.weights[0] = 0.7;
    params.projection.biases[0] = -0.15;
    let batch_norm = params.projection.batch_norm.as_mut().unwrap();
    batch_norm.scales[0] = 1.4;
    batch_norm.rolling_mean[0] = 0.25;
    batch_norm.rolling_variance[0] = 0.6;
    let input = Tensor::from_data(TensorShape::new(2, 1, 1, 2), vec![0.2, -0.4, 0.8, 0.5]).unwrap();
    let cpu_forward = cpu.forward(&input).unwrap();
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).unwrap();
    assert!(gpu_forward.cpu_forward.graph_conv_caches[0].is_none());
    let actual = &gpu_forward.cpu_forward.layer_outputs[0].data;
    let expected = &cpu_forward.layer_outputs[0].data;
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "GraphConv full BN inference[{index}]: {actual} vs {expected}"
        );
    }
}

#[test]
fn graph_conv_sgd_updates_source_weight_groups_and_momentum() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-sgd.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\n[graph_conv]\nsize=1\nedge_mode=1\ngraph_use_self=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::GraphConv(params) = &mut executable.params[0] else {
        panic!("graph_conv params");
    };
    params.projection.weights[0] = 1.0;
    params.projection.biases[0] = 2.0;
    params.self_weights.as_mut().unwrap()[0] = 1.0;
    params.edge_kernel.as_mut().unwrap().fill(1.0);
    params.edge_biases.as_mut().unwrap()[0] = 0.0;
    let LayerTrainBuffers::GraphConv(train) = &mut executable.train_buffers[0] else {
        panic!("graph_conv train buffers");
    };
    train.projection.weight_updates[0] = 0.5;
    train.projection.bias_updates[0] = 1.0;
    train.self_weight_updates.as_mut().unwrap()[0] = 2.0;
    train.edge_kernel_updates.as_mut().unwrap().fill(3.0);
    train.edge_bias_updates.as_mut().unwrap()[0] = 4.0;

    executable.update_weights(0.1, 0.5, 0.2, 2);
    let LayerParams::GraphConv(params) = &executable.params[0] else {
        panic!("graph_conv params");
    };
    let LayerTrainBuffers::GraphConv(train) = &executable.train_buffers[0] else {
        panic!("graph_conv train buffers");
    };
    assert!((params.projection.weights[0] - 1.005).abs() < 1.0e-6);
    assert!((params.projection.biases[0] - 2.05).abs() < 1.0e-6);
    assert!((params.self_weights.as_ref().unwrap()[0] - 1.08).abs() < 1.0e-6);
    assert!((params.edge_kernel.as_ref().unwrap()[0] - 1.13).abs() < 1.0e-6);
    assert!((params.edge_biases.as_ref().unwrap()[0] - 0.2).abs() < 1.0e-6);
    assert!((train.projection.weight_updates[0] - 0.05).abs() < 1.0e-6);
    assert!((train.projection.bias_updates[0] - 0.5).abs() < 1.0e-6);
    assert!((train.self_weight_updates.as_ref().unwrap()[0] - 0.8).abs() < 1.0e-6);
    assert!((train.edge_kernel_updates.as_ref().unwrap()[0] - 1.3).abs() < 1.0e-6);
    assert!((train.edge_bias_updates.as_ref().unwrap()[0] - 2.0).abs() < 1.0e-6);
}

#[test]
fn graph_conv_adam_updates_and_persists_all_moment_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-adam.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nadam=1\n[graph_conv]\nsize=1\nedge_mode=1\ngraph_use_self=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::GraphConv(params) = &mut executable.params[0] else {
        panic!("graph_conv params");
    };
    params.projection.weights[0] = 1.0;
    params.projection.biases[0] = 1.0;
    params.self_weights.as_mut().unwrap()[0] = 1.0;
    params.edge_kernel.as_mut().unwrap().fill(1.0);
    params.edge_biases.as_mut().unwrap()[0] = 1.0;
    let LayerTrainBuffers::GraphConv(train) = &mut executable.train_buffers[0] else {
        panic!("graph_conv train buffers");
    };
    train.projection.weight_updates[0] = 0.2;
    train.projection.bias_updates[0] = 0.3;
    train.self_weight_updates.as_mut().unwrap()[0] = 0.4;
    train.edge_kernel_updates.as_mut().unwrap().fill(0.5);
    train.edge_bias_updates.as_mut().unwrap()[0] = 0.6;
    executable.update_weights_adam(0.01, 1).unwrap();
    assert_eq!(executable.adam_step(), 1);
    assert!(
        executable
            .train_buffers
            .iter()
            .all(|buffers| match buffers {
                LayerTrainBuffers::GraphConv(train) => {
                    train
                        .projection
                        .weight_updates
                        .iter()
                        .all(|value| *value == 0.0)
                        && train
                            .projection
                            .bias_updates
                            .iter()
                            .all(|value| *value == 0.0)
                        && train
                            .self_weight_updates
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|value| *value == 0.0)
                        && train
                            .edge_kernel_updates
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|value| *value == 0.0)
                        && train
                            .edge_bias_updates
                            .as_ref()
                            .unwrap()
                            .iter()
                            .all(|value| *value == 0.0)
                }
                LayerTrainBuffers::None => true,
                _ => false,
            })
    );

    let path = std::env::temp_dir().join(format!(
        "darknet-graph-conv-adam-{}.state",
        std::process::id()
    ));
    let round_trip = path.with_extension("round-trip.state");
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step(), executable.adam_step());
    restored.save_optimizer_state(&round_trip).unwrap();
    assert_eq!(fs::read(&path).unwrap(), fs::read(&round_trip).unwrap());
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(round_trip);
}

#[test]
fn dsconv_runtime_forward_backward_and_native_backend_gate() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-runtime.cfg",
                "[net]\nwidth=5\nheight=4\nchannels=1\nbatch=1\n[dsconv]\nfilters=3\nsize=3\nmorph=2\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::DsConv(params) = &mut cpu.params[0] else {
        panic!("dsconv params")
    };
    for (index, weight) in params.collapse.weights.iter_mut().enumerate() {
        *weight = 0.04 + index as f32 * 0.013;
    }
    params.collapse.biases.fill(0.02);
    params.offset_weights.fill(0.0);
    params.offset_biases[0] = -0.12;
    params.offset_biases[2] = 0.14;
    params.offset_biases[3] = -0.11;
    params.offset_biases[5] = 0.13;

    let input = Tensor::from_data(
        TensorShape::new(5, 4, 1, 1),
        (0..20)
            .map(|index| (index * index) as f32 * 0.003 - 0.2)
            .collect(),
    )
    .unwrap();
    let forward = cpu.forward_training(&input).unwrap();
    assert!(forward.dsconv_caches[0].is_some());
    let seed = Tensor::filled(forward.layer_outputs[0].shape, 0.25);
    let backward = cpu.backward(&input, &forward, &seed).unwrap();
    let LayerParams::DsConv(gradient) = &backward.param_gradients[0] else {
        panic!("dsconv gradients")
    };
    assert!(gradient.collapse.weights.iter().any(|value| *value != 0.0));
    assert!(gradient.collapse.biases.iter().any(|value| *value != 0.0));
    assert!(gradient.offset_weights.iter().any(|value| *value != 0.0));
    assert!(gradient.offset_biases.iter().any(|value| *value != 0.0));
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .any(|value| *value != 0.0)
    );

    let gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let error = gpu
        .forward(&input)
        .expect_err("FakeBackend does not implement native DSConv");
    assert!(
        error
            .to_string()
            .contains("backend does not provide native f32 [dsconv]")
    );
}

#[test]
fn dsconv_multi_template_runtime_accepts_cpu_and_rejects_gpu() {
    let network = Network::from_config(ConfigFile::parse_str(
        "dsconv-multi.cfg",
        "[net]\nwidth=2\nheight=2\nchannels=1\n[dsconv]\nsize=3\ntemplates=3\n",
    ).unwrap()).unwrap();
    let input = Tensor::filled(TensorShape::new(2, 2, 1, 1), 1.0);
    let cpu = ExecutableNetwork::new(network.clone());
    assert!(cpu.forward_training(&input).is_ok());
    let gpu = GpuExecutableNetwork::new(network, FakeBackend);
    assert!(gpu.forward(&input).unwrap_err().to_string().contains("templates>1"));
}

#[test]
fn dsconv_sgd_uses_source_rate_groups_and_update_gates() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-sgd.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\ndecay=0.1\n[dsconv]\nfilters=1\nsize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DsConv(params) = &mut executable.params[0] else {
        panic!("dsconv params")
    };
    params.collapse.weights[0] = 2.0;
    params.collapse.biases[0] = 3.0;
    params.offset_weights[0] = 4.0;
    params.offset_biases[0] = 5.0;
    let LayerTrainBuffers::DsConv(train) = &mut executable.train_buffers[0] else {
        panic!("dsconv train buffers")
    };
    train.collapse.weight_updates[0] = 1.0;
    train.collapse.bias_updates[0] = 1.0;
    train.offset_weight_updates[0] = 1.0;
    train.offset_bias_updates[0] = 1.0;
    executable.update_weights(0.2, 0.0, 0.1, 2);
    let LayerParams::DsConv(params) = &executable.params[0] else {
        panic!("dsconv params")
    };
    assert!((params.collapse.weights[0] - 2.06).abs() < 1.0e-6);
    assert!((params.collapse.biases[0] - 3.1).abs() < 1.0e-6);
    assert!((params.offset_weights[0] - 4.0002).abs() < 1.0e-6);
    assert!((params.offset_biases[0] - 5.001).abs() < 1.0e-6);

    let frozen_network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-frozen.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\ndecay=0.1\n[dsconv]\nfilters=1\nsize=1\ndont_update=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut frozen = ExecutableNetwork::new(frozen_network);
    let LayerParams::DsConv(params) = &mut frozen.params[0] else {
        panic!("dsconv params")
    };
    params.collapse.weights[0] = 2.0;
    let LayerTrainBuffers::DsConv(train) = &mut frozen.train_buffers[0] else {
        panic!("dsconv train buffers")
    };
    train.collapse.weight_updates[0] = 1.0;
    let before = frozen.params.clone();
    frozen.update_weights(0.2, 0.0, 0.1, 1);
    assert_eq!(frozen.params, before);
}

#[test]
fn dsconv_adam_state_round_trips_all_parameter_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-adam.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=1\nadam=1\ndecay=0.05\n[dsconv]\nfilters=1\nsize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DsConv(params) = &mut executable.params[0] else {
        panic!("dsconv params")
    };
    params.collapse.weights.fill(0.25);
    params.collapse.biases.fill(0.5);
    params.offset_weights.fill(0.75);
    params.offset_biases.fill(1.0);
    let LayerTrainBuffers::DsConv(train) = &mut executable.train_buffers[0] else {
        panic!("dsconv train buffers")
    };
    train.collapse.weight_updates.fill(-0.2);
    train.collapse.bias_updates.fill(-0.3);
    train.offset_weight_updates.fill(-0.4);
    train.offset_bias_updates.fill(-0.5);
    executable.update_weights_adam(0.01, 1).unwrap();
    assert!(executable.host_adam_state[0].is_some());
    let path =
        std::env::temp_dir().join(format!("darknet-dsconv-adam-{}.state", std::process::id()));
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.host_adam_state, executable.host_adam_state);
    assert_eq!(restored.adam_step, executable.adam_step);
    let _ = fs::remove_file(path);
}

#[test]
fn wmhf_cpu_full_forward_matches_hand_computation() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-hand-forward.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\n[wmhf]\nfilters=4\nbatch_normalize=0\nfreq_scale=0.25\nshortcut=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Wmhf(params) = &mut executable.params[0] else {
        panic!("wmhf params")
    };
    params
        .pre
        .weights
        .copy_from_slice(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]);
    params.pre.biases.fill(0.0);
    for conv in [
        &mut params.local_dw3,
        &mut params.local_dw5,
        &mut params.local_dw7,
        &mut params.local_mix,
        &mut params.hf_gate,
    ] {
        conv.weights.fill(0.0);
        conv.biases.fill(0.0);
    }
    params.fuse.weights.fill(0.0);
    params.fuse.biases.fill(0.0);
    for channel in 0..4 {
        params.fuse.weights[channel * 4 + channel] = 1.0;
    }
    params.scan_weights.fill(0.0);
    let input = Tensor::from_data(
        TensorShape::new(2, 2, 2, 1),
        vec![1.0, 2.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    assert_eq!(
        forward.final_output().unwrap().data,
        vec![
            1.125, 2.25, 3.375, 5.625, 0.25, 0.5, 0.75, 1.25, 0.375, 0.75, 1.125, 1.875, -6.5,
            -2.0, 2.5, 11.5,
        ]
    );
}

#[test]
fn wmhf_cpu_backward_matches_finite_differences_across_all_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-finite-difference.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=1\n[wmhf]\nfilters=4\nbatch_normalize=0\nfreq_scale=0.31\nshortcut=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    fill_wmhf_test_params(&mut executable);
    let input = wmhf_test_input(1);
    let forward = executable.forward_training(&input).unwrap();
    assert!(forward.wmhf_caches[0].is_some());
    let seed = wmhf_test_seed(forward.layer_outputs[0].shape);
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    let zero_seed = Tensor::filled(seed.shape, 0.0);
    let zero_backward = executable.backward(&input, &forward, &zero_seed).unwrap();
    assert!(
        zero_backward
            .input_gradient
            .data
            .iter()
            .all(|value| *value == 0.0)
    );
    let LayerParams::Wmhf(zero_gradient) = &zero_backward.param_gradients[0] else {
        panic!("wmhf zero gradient")
    };
    assert!(zero_gradient.scan_weights.iter().all(|value| *value == 0.0));
    let LayerParams::Wmhf(gradient) = &backward.param_gradients[0] else {
        panic!("wmhf gradient")
    };
    macro_rules! check_conv {
        ($field:ident, $weight_index:expr, $bias_index:expr) => {{
            let (weight, weight_grad, bias, bias_grad) = {
                let LayerParams::Wmhf(params) = &executable.params[0] else {
                    panic!("wmhf params")
                };
                (
                    params.$field.weights[$weight_index],
                    gradient.$field.weights[$weight_index],
                    params.$field.biases[$bias_index],
                    gradient.$field.biases[$bias_index],
                )
            };
            check_wmhf_scalar_finite_difference(
                &mut executable,
                &input,
                &seed,
                stringify!($field),
                weight,
                weight_grad,
                |exe, value| {
                    let LayerParams::Wmhf(params) = &mut exe.params[0] else {
                        panic!("wmhf params")
                    };
                    params.$field.weights[$weight_index] = value;
                },
            );
            check_wmhf_scalar_finite_difference(
                &mut executable,
                &input,
                &seed,
                stringify!($field),
                bias,
                bias_grad,
                |exe, value| {
                    let LayerParams::Wmhf(params) = &mut exe.params[0] else {
                        panic!("wmhf params")
                    };
                    params.$field.biases[$bias_index] = value;
                },
            );
        }};
    }
    check_conv!(pre, 1, 0);
    check_conv!(local_dw3, 4, 1);
    check_conv!(local_dw5, 23, 0);
    check_conv!(local_dw7, 61, 1);
    check_conv!(local_mix, 5, 0);
    check_conv!(hf_gate, 0, 2);
    check_conv!(fuse, 7, 1);
    for index in 0..4 {
        let (weight, analytical) = {
            let LayerParams::Wmhf(params) = &executable.params[0] else {
                panic!("wmhf params")
            };
            (params.scan_weights[index], gradient.scan_weights[index])
        };
        check_wmhf_scalar_finite_difference(
            &mut executable,
            &input,
            &seed,
            "scan_weights",
            weight,
            analytical,
            |exe, value| {
                let LayerParams::Wmhf(params) = &mut exe.params[0] else {
                    panic!("wmhf params")
                };
                params.scan_weights[index] = value;
            },
        );
    }
    let index = 5;
    let eps = 1.5e-3;
    let mut plus = input.clone();
    plus.data[index] += eps;
    let mut minus = input.clone();
    minus.data[index] -= eps;
    let numerical = (wmhf_objective(&executable, &plus, &seed)
        - wmhf_objective(&executable, &minus, &seed))
        / (2.0 * eps);
    let analytical = backward.input_gradient.data[index];
    let tolerance = 4.0e-3 + 2.5e-2 * numerical.abs().max(analytical.abs());
    assert!(
        (analytical - numerical).abs() <= tolerance,
        "[wmhf] input gradient: analytical {analytical}, numerical {numerical}, tolerance {tolerance}"
    );
}

#[test]
fn wmhf_backward_accumulates_into_preceding_convolution() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-preceding-conv.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=1\n[convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n[wmhf]\nfilters=4\nbatch_normalize=0\nshortcut=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Convolutional(conv) = &mut executable.params[0] else {
        panic!("preceding convolution params")
    };
    conv.weights.copy_from_slice(&[0.7, -0.2, 0.1, 0.8]);
    conv.biases.copy_from_slice(&[0.03, -0.02]);
    let LayerParams::Wmhf(params) = &mut executable.params[1] else {
        panic!("wmhf params")
    };
    for conv in [
        &mut params.pre,
        &mut params.local_dw3,
        &mut params.local_dw5,
        &mut params.local_dw7,
        &mut params.local_mix,
        &mut params.hf_gate,
        &mut params.fuse,
    ] {
        conv.weights.fill(0.04);
        conv.biases.fill(0.02);
    }
    params.scan_weights.fill(0.2);
    params.scan_weights[1] = 0.5;
    params.scan_weights[2] = 0.7;
    let input = wmhf_test_input(1);
    let forward = executable.forward_training(&input).unwrap();
    assert!(forward.wmhf_caches[1].is_some());
    let seed = wmhf_test_seed(forward.layer_outputs[1].shape);
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    let LayerParams::Convolutional(gradient) = &backward.param_gradients[0] else {
        panic!("preceding convolution gradient")
    };
    assert!(gradient.weights.iter().any(|value| *value != 0.0));
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .any(|value| *value != 0.0)
    );
    let value = match &executable.params[0] {
        LayerParams::Convolutional(params) => params.weights[0],
        _ => unreachable!(),
    };
    check_wmhf_scalar_finite_difference(
        &mut executable,
        &input,
        &seed,
        "preceding convolution weight",
        value,
        gradient.weights[0],
        |exe, value| {
            let LayerParams::Convolutional(params) = &mut exe.params[0] else {
                panic!("preceding convolution params")
            };
            params.weights[0] = value;
        },
    );
}

#[test]
fn wmhf_training_batchnorm_backward_and_rolling_stats_match_forward() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-batchnorm.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n[wmhf]\nfilters=4\nbatch_normalize=1\nshortcut=0\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network);
    fill_wmhf_test_params(&mut executable);
    let input = wmhf_test_input(2);
    let forward = executable.forward_training(&input).unwrap();
    assert!(forward.wmhf_caches[0].as_ref().unwrap().training);
    let inference = executable.forward(&input).unwrap();
    assert_ne!(
        forward.layer_outputs[0].data,
        inference.layer_outputs[0].data
    );
    let seed = wmhf_test_seed(forward.layer_outputs[0].shape);
    let backward = executable.backward(&input, &forward, &seed).unwrap();
    let LayerParams::Wmhf(gradient) = &backward.param_gradients[0] else {
        panic!("wmhf gradient")
    };
    macro_rules! check_bn {
        ($field:ident) => {{
            let (beta, beta_grad, scale, scale_grad) = {
                let LayerParams::Wmhf(params) = &executable.params[0] else {
                    panic!("wmhf params")
                };
                (
                    params.$field.biases[0],
                    gradient.$field.biases[0],
                    params.$field.batch_norm.as_ref().unwrap().scales[0],
                    gradient.$field.batch_norm.as_ref().unwrap().scales[0],
                )
            };
            check_wmhf_scalar_finite_difference(
                &mut executable,
                &input,
                &seed,
                stringify!($field),
                beta,
                beta_grad,
                |exe, value| {
                    let LayerParams::Wmhf(params) = &mut exe.params[0] else {
                        panic!("wmhf params")
                    };
                    params.$field.biases[0] = value;
                },
            );
            check_wmhf_scalar_finite_difference(
                &mut executable,
                &input,
                &seed,
                stringify!($field),
                scale,
                scale_grad,
                |exe, value| {
                    let LayerParams::Wmhf(params) = &mut exe.params[0] else {
                        panic!("wmhf params")
                    };
                    params.$field.batch_norm.as_mut().unwrap().scales[0] = value;
                },
            );
        }};
    }
    check_bn!(pre);
    check_bn!(local_mix);
    check_bn!(fuse);

    let cache = forward.wmhf_caches[0].as_ref().unwrap();
    let expected_stats = [
        super::batchnorm_channel_stats(&cache.pre.raw_output),
        super::batchnorm_channel_stats(&cache.local_mix.raw_output),
        super::batchnorm_channel_stats(&cache.fuse.raw_output),
    ];
    let initial_stats = {
        let LayerParams::Wmhf(params) = &executable.params[0] else {
            panic!("wmhf params")
        };
        [&params.pre, &params.local_mix, &params.fuse].map(|conv| {
            let bn = conv.batch_norm.as_ref().unwrap();
            (bn.rolling_mean.clone(), bn.rolling_variance.clone())
        })
    };
    executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    let LayerParams::Wmhf(params) = &executable.params[0] else {
        panic!("wmhf params")
    };
    for (index, conv) in [&params.pre, &params.local_mix, &params.fuse]
        .into_iter()
        .enumerate()
    {
        let bn = conv.batch_norm.as_ref().unwrap();
        for channel in 0..bn.scales.len() {
            let mean =
                0.9 * initial_stats[index].0[channel] + 0.1 * expected_stats[index].0[channel];
            let variance =
                0.9 * initial_stats[index].1[channel] + 0.1 * expected_stats[index].1[channel];
            assert!((bn.rolling_mean[channel] - mean).abs() < 1.0e-5);
            assert!((bn.rolling_variance[channel] - variance).abs() < 1.0e-5);
        }
    }
    let LayerTrainBuffers::Wmhf(train) = &executable.train_buffers[0] else {
        panic!("wmhf train buffers")
    };
    assert!(train.scan_weight_updates.iter().any(|value| *value != 0.0));
    for conv in [
        &train.pre,
        &train.local_dw3,
        &train.local_dw5,
        &train.local_dw7,
        &train.local_mix,
        &train.hf_gate,
        &train.fuse,
    ] {
        assert!(conv.weight_updates.iter().any(|value| *value != 0.0));
        assert!(conv.bias_updates.iter().any(|value| *value != 0.0));
    }
}

pub(super) fn clifford_test_network(optimizer: &str) -> Network {
    Network::from_config(
        ConfigFile::parse_str(
            "clifford-runtime.cfg",
            &format!(
                r#"[net]
width=2
height=1
channels=2
batch=2
{optimizer}
[clifford]
filters=2
shifts=1
ctx_mode=diff
gffn_mode=hybrid
global_op=avgpool
dwconv_size=1
num_dwconv=1
activation=swish
"#
            ),
        )
        .unwrap(),
    )
    .unwrap()
}

pub(super) fn run_clifford_backward(executable: &mut ExecutableNetwork) -> ForwardResult {
    let input = Tensor::from_data(
        TensorShape::new(2, 1, 2, 2),
        vec![0.13, -0.21, 0.31, 0.07, -0.17, 0.23, 0.09, -0.29],
    )
    .unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let seed = Tensor::from_data(
        input.shape,
        vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.15, -0.35],
    )
    .unwrap();
    executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    forward
}

#[test]
fn clifford_runtime_sgd_adam_and_muon_update_cpu_parameters() {
    let cases = [("", 0), ("adam=1", 1), ("muon=1", 2)];
    for (optimizer, mode) in cases {
        let network = clifford_test_network(optimizer);
        let mut executable = ExecutableNetwork::new(network.clone());
        crate::initialize::initialize_training_params(&network, &mut executable.params, 41);
        let initial_params = executable.params.clone();
        let forward = run_clifford_backward(&mut executable);
        assert!(forward.clifford_caches[0].is_some());
        let LayerTrainBuffers::Clifford(train) = &executable.train_buffers[0] else {
            panic!("Clifford train buffers")
        };
        assert!(
            train
                .determinant_weight_updates
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            train
                .local
                .projection_weight_updates
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            train
                .local
                .gate_weight_updates
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            train.depthwise[0]
                .weight_updates
                .iter()
                .any(|value| *value != 0.0)
        );
        assert!(
            train
                .global
                .as_ref()
                .unwrap()
                .gate_weight_updates
                .iter()
                .any(|value| *value != 0.0)
        );

        let cache = forward.clifford_caches[0].as_ref().unwrap();
        let expected_stats =
            super::batchnorm_channel_stats(&cache.local.as_ref().unwrap().depthwise[0].raw_output);
        let (initial_mean, initial_variance) = {
            let LayerParams::Clifford(params) = &initial_params[0] else {
                unreachable!()
            };
            let bn = params.depthwise[0].batch_norm.as_ref().unwrap();
            (bn.rolling_mean.clone(), bn.rolling_variance.clone())
        };
        // The backward-accumulate call above performs the existing running
        // statistic update as part of the CPU training path.
        let LayerParams::Clifford(after_backward) = &executable.params[0] else {
            unreachable!()
        };
        let bn = after_backward.depthwise[0].batch_norm.as_ref().unwrap();
        for channel in 0..2 {
            assert!(
                (bn.rolling_mean[channel]
                    - (0.9 * initial_mean[channel] + 0.1 * expected_stats.0[channel]))
                    .abs()
                    < 1.0e-5
            );
            assert!(
                (bn.rolling_variance[channel]
                    - (0.9 * initial_variance[channel] + 0.1 * expected_stats.1[channel]))
                    .abs()
                    < 1.0e-5
            );
        }

        let params_before_update = executable.params.clone();
        match mode {
            0 => executable.update_weights(0.01, 0.2, 0.01, 2),
            1 => executable.update_weights_adam(0.01, 2).unwrap(),
            2 => executable.update_weights_muon(0.01, 2).unwrap(),
            _ => unreachable!(),
        }
        assert_ne!(
            executable.params, params_before_update,
            "{optimizer} update changed no parameters"
        );
        if mode == 1 {
            let path = std::env::temp_dir().join(format!(
                "darknet-clifford-adam-{}.state",
                std::process::id()
            ));
            executable.save_optimizer_state(&path).unwrap();
            let mut restored = ExecutableNetwork::new(network.clone());
            restored.load_optimizer_state(&path).unwrap();
            assert_eq!(restored.host_adam_state, executable.host_adam_state);
            assert_eq!(restored.adam_step(), executable.adam_step());
            let _ = fs::remove_file(path);
        }
        if mode == 2 {
            let state = executable.muon_state[0].as_ref().unwrap();
            assert_eq!(state.matrices.len(), 7);
            assert_eq!(
                crate::muon_runtime::matrix_shapes(
                    &network.layers[0].kind,
                    network.layers[0].input_shape,
                    network.layers[0].output_shape,
                    &mut executable.params[0].clone(),
                    &mut executable.train_buffers[0].clone(),
                )
                .unwrap(),
                vec![(2, 2), (1, 1), (1, 1), (2, 4), (2, 4), (2, 4), (2, 4)]
            );
            let path = std::env::temp_dir().join(format!(
                "darknet-clifford-muon-{}.state",
                std::process::id()
            ));
            executable.save_optimizer_state(&path).unwrap();
            let mut restored = ExecutableNetwork::new(network.clone());
            restored.load_optimizer_state(&path).unwrap();
            assert_eq!(restored.muon_state, executable.muon_state);
            let _ = fs::remove_file(path);
        }
    }
}

#[test]
fn clifford_global_only_skips_unused_local_training_and_decay() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "clifford-global-only.cfg",
            r#"[net]
width=2
height=1
channels=2
batch=2
adam=1
decay=0.5
decoupled_decay=1
[clifford]
filters=2
shifts=1
gffn_mode=global
global_op=avgpool
dwconv_size=1
num_dwconv=1
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    crate::initialize::initialize_training_params(&network, &mut executable.params, 19);
    let (local_before, depthwise_before, determinant_before) = {
        let LayerParams::Clifford(params) = &executable.params[0] else {
            panic!("Clifford parameters")
        };
        (
            params.local.clone(),
            params.depthwise.clone(),
            params.determinant_weights.clone(),
        )
    };
    let input = Tensor::from_data(
        TensorShape::new(2, 1, 2, 2),
        vec![0.13, -0.21, 0.31, 0.07, -0.17, 0.23, 0.09, -0.29],
    )
    .unwrap();
    let forward = executable.forward_training(&input).unwrap();
    assert!(forward.clifford_caches[0].as_ref().unwrap().local.is_none());
    let seed = Tensor::from_data(input.shape, vec![0.3; input.data.len()]).unwrap();
    executable
        .backward_accumulate(&input, &forward, &seed)
        .unwrap();
    let LayerTrainBuffers::Clifford(train) = &executable.train_buffers[0] else {
        unreachable!()
    };
    assert!(
        train.depthwise[0]
            .weight_updates
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        train
            .local
            .projection_weight_updates
            .iter()
            .all(|value| *value == 0.0)
    );
    executable.update_weights_adam(0.01, 2).unwrap();
    let LayerParams::Clifford(params) = &executable.params[0] else {
        unreachable!()
    };
    assert_eq!(params.local, local_before);
    assert_eq!(params.depthwise, depthwise_before);
    assert_eq!(params.determinant_weights, determinant_before);
}

#[test]
fn clifford_adam_updates_fourier_multipliers() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "clifford-fourier-adam.cfg",
            r#"[net]
width=2
height=1
channels=2
batch=2
adam=1
decay=0.1
[clifford]
filters=2
shifts=1
gffn_mode=hybrid
global_op=fourier
fno_modes_h=1
fno_modes_w=2
dwconv_size=1
num_dwconv=1
"#,
        )
        .unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    crate::initialize::initialize_training_params(&network, &mut executable.params, 73);
    let before = {
        let LayerParams::Clifford(params) = &executable.params[0] else {
            panic!("Clifford parameters")
        };
        params.fourier.clone().unwrap()
    };
    run_clifford_backward(&mut executable);
    executable.update_weights_adam(0.01, 2).unwrap();
    let LayerParams::Clifford(params) = &executable.params[0] else {
        unreachable!()
    };
    let after = params.fourier.as_ref().unwrap();
    assert_ne!(after.real, before.real);
    assert_ne!(after.imaginary, before.imaginary);
}

#[test]
fn clifford_gpu_compact_f32_matches_full_forward() {
    let network = clifford_test_network("");
    let gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(TensorShape::new(2, 1, 2, 2), vec![0.1; 8]).unwrap();
    let full = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
    assert_eq!(compact.shape, full.shape);
    for (a,b) in compact.data.iter().zip(&full.data) { assert!((a-b).abs()<1e-5); }
}

#[test]
fn wmhf_sgd_updates_every_block_and_gpu_forward_matches_cpu() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-sgd.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=2\ndecay=0.1\n[wmhf]\nfilters=4\nbatch_normalize=1\nlearning_rate=0.25\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::Wmhf(params) = &mut executable.params[0] else {
        panic!("wmhf params")
    };
    for conv in [
        &mut params.pre,
        &mut params.local_dw3,
        &mut params.local_dw5,
        &mut params.local_dw7,
        &mut params.local_mix,
        &mut params.hf_gate,
        &mut params.fuse,
    ] {
        conv.weights.fill(2.0);
        conv.biases.fill(3.0);
        if let Some(bn) = &mut conv.batch_norm {
            bn.scales.fill(1.0);
        }
    }
    params.scan_weights.fill(2.0);
    let LayerTrainBuffers::Wmhf(train) = &mut executable.train_buffers[0] else {
        panic!("wmhf train buffers")
    };
    for conv in [
        &mut train.pre,
        &mut train.local_dw3,
        &mut train.local_dw5,
        &mut train.local_dw7,
        &mut train.local_mix,
        &mut train.hf_gate,
        &mut train.fuse,
    ] {
        conv.weight_updates.fill(1.0);
        conv.bias_updates.fill(1.0);
        if let Some(bn) = &mut conv.batch_norm {
            bn.scale_updates.fill(1.0);
        }
    }
    train.scan_weight_updates.fill(1.0);
    executable.update_weights(0.2, 0.5, 0.1, 2);
    let LayerParams::Wmhf(params) = &executable.params[0] else {
        panic!("wmhf params")
    };
    for conv in [
        &params.pre,
        &params.local_dw3,
        &params.local_dw5,
        &params.local_dw7,
        &params.local_mix,
        &params.hf_gate,
        &params.fuse,
    ] {
        assert!((conv.weights[0] - 2.06).abs() < 1.0e-6);
        assert!((conv.biases[0] - 3.1).abs() < 1.0e-6);
        if let Some(bn) = &conv.batch_norm {
            assert!((bn.scales[0] - 1.1).abs() < 1.0e-6);
        }
    }
    assert!((params.scan_weights[0] - 2.015).abs() < 1.0e-6);

    // The full GPU forward is native (FakeBackend runs the trait host
    // defaults for the WMHF primitives). Compact f32 reuses that path;
    // compact half-storage support remains a separate contract.
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = executable.params.clone();
    let input = Tensor::from_data(TensorShape::new(1, 1, 2, 1), vec![0.1, 0.2]).unwrap();
    let expected = executable.forward(&input).unwrap().final_output().unwrap().clone();
    let actual = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    assert_eq!(actual.shape, expected.shape);
    assert_wmhf_close("GPU inference forward", &actual.data, &expected.data, 1.0e-5);
    let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
    assert_wmhf_close("GPU compact inference", &compact.data, &expected.data, 1e-5);
}

fn wmhf_buffer_blob(buffers: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::Wmhf(train) = buffers else {
        panic!("wmhf train buffers")
    };
    let mut blob = Vec::new();
    for conv in [
        &train.pre,
        &train.local_dw3,
        &train.local_dw5,
        &train.local_dw7,
        &train.local_mix,
        &train.hf_gate,
        &train.fuse,
    ] {
        blob.extend_from_slice(&conv.weight_updates);
        blob.extend_from_slice(&conv.bias_updates);
        if let Some(batch_norm) = &conv.batch_norm {
            blob.extend_from_slice(&batch_norm.scale_updates);
        }
    }
    blob.extend_from_slice(&train.scan_weight_updates);
    blob
}

fn wmhf_param_blob(params: &LayerParams) -> Vec<f32> {
    let LayerParams::Wmhf(params) = params else {
        panic!("wmhf params")
    };
    let mut blob = Vec::new();
    for conv in [
        &params.pre,
        &params.local_dw3,
        &params.local_dw5,
        &params.local_dw7,
        &params.local_mix,
        &params.hf_gate,
        &params.fuse,
    ] {
        blob.extend_from_slice(&conv.weights);
        blob.extend_from_slice(&conv.biases);
        if let Some(batch_norm) = &conv.batch_norm {
            blob.extend_from_slice(&batch_norm.scales);
            blob.extend_from_slice(&batch_norm.rolling_mean);
            blob.extend_from_slice(&batch_norm.rolling_variance);
        }
    }
    blob.extend_from_slice(&params.scan_weights);
    blob
}

fn assert_wmhf_close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "[wmhf] {name}: length");
    let scale = want.iter().fold(1.0f32, |max, value| max.max(value.abs()));
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= tolerance * scale,
            "[wmhf] {name}[{index}]: got {got}, want {want}"
        );
    }
}

/// GPU-runtime composition (native forward, backward, staged gradients,
/// rolling statistics and the host-staged SGD step) against the CPU reference
/// on the FakeBackend host defaults; the Oxide suite repeats it on the device.
fn check_wmhf_fake_gpu_training(cfg: &str) {
    let network =
        Network::from_config(ConfigFile::parse_str("wmhf-fake-gpu.cfg", cfg).unwrap()).unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    fill_wmhf_test_params(&mut cpu);
    let shape = network.net.input_shape();
    let input = Tensor::from_data(
        shape,
        (0..shape.element_count() * shape.batch)
            .map(|index| ((index * 7 + index / 5) % 19) as f32 * 0.071 - 0.55)
            .collect(),
    )
    .unwrap();
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();

    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    let expected = cpu_forward.final_output().unwrap().clone();
    let actual = gpu_state.cpu_forward.final_output().unwrap().clone();
    assert_wmhf_close("forward", &actual.data, &expected.data, 1.0e-5);

    let seeds = [Some(wmhf_test_seed(expected.shape))];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let cpu_backward = cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &seeds)
        .unwrap()
        .expect("native [wmhf] input gradient");
    assert_wmhf_close(
        "input gradient",
        &gpu_input_gradient.data,
        &cpu_backward.input_gradient.data,
        1.0e-4,
    );
    assert_wmhf_close(
        "staged gradients",
        &wmhf_buffer_blob(&gpu.host_fallback_train_buffers()[0]),
        &wmhf_buffer_blob(&cpu.train_buffers[0]),
        1.0e-4,
    );
    assert_wmhf_close(
        "rolling statistics",
        &wmhf_param_blob(&gpu.params[0]),
        &wmhf_param_blob(&cpu.params[0]),
        1.0e-5,
    );

    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    assert_wmhf_close(
        "updated parameters",
        &wmhf_param_blob(&gpu.params[0]),
        &wmhf_param_blob(&cpu.params[0]),
        1.0e-4,
    );
}

#[test]
fn wmhf_gpu_runtime_training_matches_cpu_on_host_defaults() {
    // Batch-statistics BN, identity channels, no shortcut, leaky output.
    check_wmhf_fake_gpu_training(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n[wmhf]\nfilters=4\nbatch_normalize=1\nshortcut=0\nactivation=leaky\n",
    );
    // No BN, no identity channels, shortcut, odd non-square spatial shape.
    check_wmhf_fake_gpu_training(
        "[net]\nwidth=5\nheight=3\nchannels=4\nbatch=2\n[wmhf]\nfilters=4\nidentity_ratio=0\nbatch_normalize=0\nshortcut=1\nfreq_scale=0.31\nactivation=linear\n",
    );
}

#[test]
fn wmhf_adam_state_round_trips_all_parameter_blocks() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-adam.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=2\nadam=1\ndecay=0.05\n[wmhf]\nfilters=4\nbatch_normalize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    fill_wmhf_test_params(&mut executable);
    let LayerTrainBuffers::Wmhf(train) = &mut executable.train_buffers[0] else {
        panic!("wmhf train buffers")
    };
    for conv in [
        &mut train.pre,
        &mut train.local_dw3,
        &mut train.local_dw5,
        &mut train.local_dw7,
        &mut train.local_mix,
        &mut train.hf_gate,
        &mut train.fuse,
    ] {
        conv.weight_updates.fill(0.2);
        conv.bias_updates.fill(0.3);
        if let Some(bn) = &mut conv.batch_norm {
            bn.scale_updates.fill(0.4);
        }
    }
    train.scan_weight_updates.fill(0.5);
    executable.update_weights_adam(0.01, 1).unwrap();
    assert_eq!(executable.adam_step(), 1);
    let path = std::env::temp_dir().join(format!("darknet-wmhf-adam-{}.state", std::process::id()));
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.host_adam_state, executable.host_adam_state);
    assert_eq!(restored.adam_step(), executable.adam_step());
    let _ = fs::remove_file(path);
}

#[test]
fn deform_conv_runtime_backward_and_fake_backend_host_fallback_match() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-runtime.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=1\n[convolutional]\nfilters=2\nsize=1\nactivation=linear\n[deform_conv]\nfilters=2\ngroups=2\nsize=1\nuse_mask=1\nbatch_normalize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Convolutional(conv) = &mut cpu.params[0] else {
        panic!("preceding convolution params")
    };
    conv.weights.copy_from_slice(&[1.1, -0.7, 0.8, 1.3]);
    conv.biases.copy_from_slice(&[0.04, -0.03]);
    let LayerParams::DeformConv(params) = &mut cpu.params[1] else {
        panic!("deform_conv params")
    };
    params.main.weights.copy_from_slice(&[0.9, 1.2]);
    params.main.biases.copy_from_slice(&[0.07, -0.05]);
    let batch_norm = params.main.batch_norm.as_mut().unwrap();
    batch_norm.scales.copy_from_slice(&[1.1, 0.85]);
    batch_norm.rolling_mean.copy_from_slice(&[0.03, -0.08]);
    batch_norm.rolling_variance.copy_from_slice(&[0.7, 1.2]);
    params
        .offset_weights
        .copy_from_slice(&[0.001, -0.0007, 0.0008, 0.0004]);
    params.offset_biases.copy_from_slice(&[0.11, -0.13]);
    params
        .mask_weights
        .as_mut()
        .unwrap()
        .copy_from_slice(&[0.03, -0.02]);
    params
        .mask_biases
        .as_mut()
        .unwrap()
        .copy_from_slice(&[0.12]);

    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    // The native device backward covers this layer, so the network no longer
    // forces the host fallback; the comparison below still proves FakeBackend
    // (host defaults) and CPU agree.
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = Tensor::from_data(
        TensorShape::new(3, 3, 2, 1),
        (0..18)
            .map(|index| 0.18 + index as f32 * 0.027 - (index % 4) as f32 * 0.011)
            .collect(),
    )
    .unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    assert!(cpu_forward.deform_conv_caches[1].is_some());
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.layer_outputs,
        cpu_forward.layer_outputs
    );
    assert_eq!(
        gpu_forward.cpu_forward.deform_conv_caches,
        cpu_forward.deform_conv_caches
    );

    let seed = Tensor::from_data(
        TensorShape::new(3, 3, 2, 1),
        (0..18)
            .map(|index| 0.14 * ((index % 5) as f32 - 1.8))
            .collect(),
    )
    .unwrap();
    let seeds = [None, Some(seed.clone())];
    let base_params = cpu.params.clone();
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let LayerParams::DeformConv(updated_params) = &cpu.params[1] else {
        panic!("deform_conv params")
    };
    let updated_bn = updated_params.main.batch_norm.as_ref().unwrap();
    let LayerParams::DeformConv(original_params) = &base_params[1] else {
        panic!("original deform_conv params")
    };
    let original_bn = original_params.main.batch_norm.as_ref().unwrap();
    assert_ne!(updated_bn.rolling_mean, original_bn.rolling_mean);
    assert_ne!(updated_bn.rolling_variance, original_bn.rolling_variance);
    let LayerParams::DeformConv(gradient) = &backward.param_gradients[1] else {
        panic!("deform_conv gradients")
    };
    let objective = |scale: f32| {
        let mut executable = ExecutableNetwork::new(network.clone());
        executable.params = base_params.clone();
        let LayerParams::DeformConv(params) = &mut executable.params[1] else {
            panic!("deform_conv params")
        };
        params.main.batch_norm.as_mut().unwrap().scales[0] = scale;
        executable
            .forward(&input)
            .unwrap()
            .final_output()
            .unwrap()
            .data
            .iter()
            .zip(&seed.data)
            .map(|(value, grad)| value * grad)
            .sum::<f32>()
    };
    let epsilon = 1.0e-3;
    let numerical_scale = (objective(1.1 + epsilon) - objective(1.1 - epsilon)) / (2.0 * epsilon);
    assert!(
        (gradient.main.batch_norm.as_ref().unwrap().scales[0] - numerical_scale).abs() < 2.0e-3,
        "BN scale gradient {} differs from finite difference {numerical_scale}",
        gradient.main.batch_norm.as_ref().unwrap().scales[0]
    );

    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    let device_buffers = gpu.device_train_buffers.borrow();
    let conv_train = device_buffers[0].as_ref().expect("device conv gradients");
    let LayerTrainBuffers::Convolutional(cpu_conv_train) = &cpu.train_buffers[0] else {
        panic!("preceding convolution train buffers")
    };
    assert_eq!(
        gpu.backend
            .download_f32(&conv_train.weight_updates)
            .unwrap(),
        cpu_conv_train.weight_updates
    );
    assert_eq!(
        gpu.backend.download_f32(&conv_train.bias_updates).unwrap(),
        cpu_conv_train.bias_updates
    );
    drop(device_buffers);
    assert_eq!(
        cpu.train_buffers[1],
        gpu.cpu_fallback_train_buffers.borrow()[1]
    );
    assert_eq!(cpu.params, gpu.params);
    let LayerTrainBuffers::DeformConv(deform_train) = &cpu.train_buffers[1] else {
        panic!("deform_conv train buffers")
    };
    assert!(
        deform_train
            .main
            .weight_updates
            .iter()
            .any(|value| *value != 0.0)
    );
    assert!(
        deform_train
            .offset_weight_updates
            .iter()
            .any(|value| *value != 0.0)
    );
    assert!(
        deform_train
            .mask_weight_updates
            .as_ref()
            .unwrap()
            .iter()
            .any(|value| *value != 0.0)
    );

    cpu.update_weights(0.01, 0.2, 0.01, 1);
    gpu.update_weights_gpu(0.01, 0.2, 0.01, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
}

#[test]
fn deform_conv_host_adam_and_optimizer_state_round_trip() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-adam.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nadam=1\ndecay=0.05\n[deform_conv]\nfilters=2\ngroups=2\nsize=1\nuse_mask=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::DeformConv(params) = &mut cpu.params[0] else {
        panic!("deform_conv params")
    };
    params.main.weights.fill(0.25);
    params.main.biases.fill(0.5);
    params.offset_weights.fill(0.75);
    params.offset_biases.fill(1.0);
    params.mask_weights.as_mut().unwrap().fill(1.25);
    params.mask_biases.as_mut().unwrap().fill(1.5);
    let LayerTrainBuffers::DeformConv(train) = &mut cpu.train_buffers[0] else {
        panic!("deform_conv train buffers")
    };
    train.main.weight_updates.fill(-0.2);
    train.main.bias_updates.fill(-0.3);
    train.offset_weight_updates.fill(-0.4);
    train.offset_bias_updates.fill(-0.5);
    train.mask_weight_updates.as_mut().unwrap().fill(-0.6);
    train.mask_bias_updates.as_mut().unwrap().fill(-0.7);

    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    *gpu.cpu_fallback_train_buffers.borrow_mut() = cpu.train_buffers.clone();
    cpu.network.layers[0].train.dont_update = 1;
    gpu.network.layers[0].train.dont_update = 1;
    let frozen = cpu.params.clone();
    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.05, 1).unwrap();
    assert_eq!(cpu.params, frozen);
    assert_eq!(gpu.params, frozen);
    cpu.network.layers[0].train.dont_update = 0;
    gpu.network.layers[0].train.dont_update = 0;
    cpu.update_weights_adam(0.01, 1).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.05, 1).unwrap();
    assert_eq!(gpu.params, cpu.params);
    assert_ne!(gpu.params, frozen);

    let path = std::env::temp_dir().join(format!(
        "darknet-deform-conv-adam-{}.state",
        std::process::id()
    ));
    cpu.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(network);
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.host_adam_state, cpu.host_adam_state);
    assert_eq!(restored.adam_step, cpu.adam_step);
    let _ = fs::remove_file(path);
}

#[test]
fn dcnv4_batchnorm_scale_derivative_and_rolling_stats_use_native_defaults() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "dcnv4-bn-runtime.cfg",
            "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\n\
                 [dcnv4]\nsize=1\npadding=0\nbatch_normalize=1\
                 \nd_stride=1\nsoftmax=0\nactivation=linear\n",
        )
        .unwrap(),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Dcnv4(params) = &mut cpu.params[0] else {
        panic!("DCNv4 params")
    };
    params.offset_biases[2] = 1.0;
    params.biases[0] = 7.0; // C++ DCNv4's BN path does not add ordinary biases.
    let batch_norm = params.batch_norm.as_mut().unwrap();
    batch_norm.scales[0] = 1.3;
    batch_norm.rolling_mean[0] = 0.2;
    batch_norm.rolling_variance[0] = 0.7;

    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let cpu_forward = cpu.forward_training(&input).unwrap();
    let expected = input
        .data
        .iter()
        .map(|value| (value - 0.2) / (0.7 + super::BATCH_NORM_EPSILON).sqrt() * 1.3)
        .collect::<Vec<_>>();
    for (actual, expected) in cpu_forward
        .final_output()
        .unwrap()
        .data
        .iter()
        .zip(expected)
    {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        cpu_forward.final_output()
    );
    let seed = Tensor::filled(TensorShape::new(2, 2, 1, 1), 0.5);
    let seeds = [Some(seed)];
    let backward = cpu
        .backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    let LayerParams::Dcnv4(gradient) = &backward.param_gradients[0] else {
        panic!("DCNv4 gradients")
    };
    let expected_scale_gradient = input
        .data
        .iter()
        .map(|value| 0.5 * (value - 0.2) / (0.7 + super::BATCH_NORM_EPSILON).sqrt())
        .sum::<f32>();
    assert!(
        (gradient.batch_norm.as_ref().unwrap().scales[0] - expected_scale_gradient).abs() < 1.0e-5,
        "actual scale gradient {}, expected {expected_scale_gradient}",
        gradient.batch_norm.as_ref().unwrap().scales[0]
    );
    assert_eq!(gradient.biases, vec![0.0]);
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    assert_eq!(cpu.params, gpu.params);
    let LayerTrainBuffers::Dcnv4(cpu_train) = &cpu.train_buffers[0] else {
        panic!("DCNv4 train buffers")
    };
    let device_train = gpu.device_dcnv4_train_buffers.borrow();
    let device_train = device_train[0].as_ref().unwrap();
    assert_eq!(
        gpu.backend
            .download_f32(&device_train.offset_weight_updates)
            .unwrap(),
        cpu_train.offset_weight_updates
    );
    assert_eq!(
        gpu.backend
            .download_f32(&device_train.offset_bias_updates)
            .unwrap(),
        cpu_train.offset_bias_updates
    );
    assert_eq!(
        gpu.backend
            .download_f32(&device_train.bias_updates)
            .unwrap(),
        cpu_train.bias_updates
    );
    assert_eq!(
        gpu.backend
            .download_f32(device_train.scale_updates.as_ref().unwrap())
            .unwrap(),
        cpu_train.batch_norm.as_ref().unwrap().scale_updates
    );
    let LayerParams::Dcnv4(params) = &cpu.params[0] else {
        panic!("DCNv4 params")
    };
    let batch_norm = params.batch_norm.as_ref().unwrap();
    assert!((batch_norm.rolling_mean[0] - 0.43).abs() < 1.0e-6);
    assert!((batch_norm.rolling_variance[0] - 0.796_666_7).abs() < 1.0e-5);
}
