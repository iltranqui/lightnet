use super::*;

#[test]
fn laya_act_head_gpu_forward_matches_hand_erf_and_retains_cpu_backward() {
    let mut cpu = laya_act_head_network();
    let input = laya_act_head_input(&cpu);
    let LayerParams::LayaActHead(p) = &mut cpu.params[3] else {
        panic!("act")
    };
    p.fc1_w.fill(0.0);
    p.fc1_b.fill(0.0);
    p.fc2_w.fill(0.0);
    p.fc2_b.fill(0.0);
    p.fc1_w[0] = 1.0;
    p.fc1_b[0] = -1.5; // CLS .5 -> -1
    p.fc1_w[11] = 1.0;
    p.fc1_b[1] = 0.9; // CLS channel 1 .1 -> +1
    p.fc2_w[0] = 1.0;
    p.fc2_w[4] = 1.0;
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(
        gpu.requires_cpu_backward_fallback(),
        "forward-only change must retain CPU backward"
    );
    let calls = crate::laya_act_head::cpu_forward_call_count();
    for result in [
        gpu.forward(&input).unwrap().final_output().unwrap().clone(),
        gpu.forward_compact(&input).unwrap().final_output.unwrap(),
    ] {
        for (&a, b) in result.data.iter().zip([-0.15865525, 0.84134475]) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    assert_eq!(
        crate::laya_act_head::cpu_forward_call_count(),
        calls,
        "GPU forwards must not execute the CPU act-head"
    );
    // Existing CPU backward must still work after native forward, and this
    // increment must not silently start updating the frozen act parameters.
    let fixed = gpu.params[3].clone();
    let previous_conv = gpu.params[0].clone();
    let state = gpu.forward_training_with_state(&input).unwrap();
    let cpu_state = cpu.forward_training(&input).unwrap();
    let mut seeds = vec![None; 4];
    seeds[3] = Some(Tensor::filled(cpu.network.layers[3].output_shape, 0.25));
    cpu.backward_accumulate_seeded(&input, &cpu_state, &seeds)
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&state, &seeds).unwrap();
    cpu.update_weights(0.01, 0.0, 0.0, 1);
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1).unwrap();
    assert_eq!(gpu.params[3], fixed);
    assert_ne!(gpu.params[0], previous_conv);
    let LayerParams::Convolutional(c) = &cpu.params[0] else {
        panic!("conv")
    };
    let LayerParams::Convolutional(g) = &gpu.params[0] else {
        panic!("conv")
    };
    for (a, b) in c
        .weights
        .iter()
        .chain(&c.biases)
        .zip(g.weights.iter().chain(&g.biases))
    {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn laya_act_head_gpu_forward_rejects_all_parameter_blocks_and_bad_dimensions() {
    let cpu = laya_act_head_network();
    let input = laya_act_head_input(&cpu);
    for block in 0..4 {
        let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
        gpu.params = cpu.params.clone();
        let LayerParams::LayaActHead(p) = &mut gpu.params[3] else {
            panic!("act")
        };
        match block {
            0 => {
                p.fc1_w.pop();
            }
            1 => {
                p.fc1_b.pop();
            }
            2 => {
                p.fc2_w.pop();
            }
            _ => {
                p.fc2_b.pop();
            }
        }
        for error in [
            gpu.forward(&input).unwrap_err(),
            gpu.forward_compact(&input).unwrap_err(),
        ] {
            assert!(error.to_string().contains("act-head parameter"), "{error}");
        }
    }
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    let LayerKind::LayaActHead(spec) = &mut gpu.network.layers[3].kind else {
        panic!("act")
    };
    spec.fc_dim = usize::MAX;
    assert!(
        gpu.forward(&input)
            .unwrap_err()
            .to_string()
            .contains("act-head")
    );
}
