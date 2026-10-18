use super::*;

#[test]
fn marker_gather_gpu_uses_device_backward_and_accumulates_only_feature_source() {
    let mut cpu = laya_marker_gather_network();
    let input = laya_marker_gather_input(&cpu);
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    assert!(
        !gpu.requires_cpu_backward_fallback(),
        "marker gather must not force whole-net CPU backward"
    );
    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
    assert_eq!(
        cpu_forward.final_output(),
        gpu_forward.cpu_forward.final_output()
    );
    assert_eq!(
        gpu.forward_compact(&input).unwrap().final_output.as_ref(),
        cpu_forward.final_output()
    );
    // Both source references alias layer zero; B is metadata and must not
    // receive the gather derivative a second time. Also retain its direct seed.
    let seeds = vec![
        Some(Tensor::filled(cpu.network.layers[0].output_shape, 0.125)),
        Some(Tensor::filled(cpu.network.layers[1].output_shape, 0.25)),
    ];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .unwrap();
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .unwrap();
    cpu.update_weights(0.01, 0.0, 0.0, 1);
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1).unwrap();
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
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
