//! Unrun native DPT output/input/parameter/update parity fixtures.
//! Reference: Rust CPU forward/backward, no compiled-C++ golden.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams,
    LayerTrainBuffers, Network, Tensor};
use lightnet_cuda::F32TensorOpsBackend;
use lightnet_cuda_oxide::OxideBackend;

fn close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}");
    let scale = want.iter().fold(1.0_f32, |s, x| s.max(x.abs()));
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= tolerance*scale,
            "{name}[{i}]: {a} vs {b}");
    }
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.025 + (i as f32 * 0.17).sin() * 0.01).collect()
}
fn updates(buffer: &LayerTrainBuffers) -> &[f32] {
    let LayerTrainBuffers::DptHead(t) = buffer else { panic!("DPT buffers") };
    &t.weight_updates
}
fn weights(p: &LayerParams) -> &[f32] {
    let LayerParams::DptHead(p) = p else { panic!("DPT params") };
    &p.values
}

#[test]
fn dpt_native_four_tap_forward_backward_accumulation_and_updates_match_cpu() {
    // Odd and non-square grids exercise ceil(stride=2), align-corners resize
    // and all four fusion branches. Two batches pin tap channel routing.
    for (width, height, optimizer) in [(3, 2, ""), (3, 3, "adam=1\n"), (2, 3, "muon=1\ndecoupled_decay=1\n")] {
        let cfg = format!("[net]\nwidth={width}\nheight={height}\nchannels=4\nbatch=2\n{optimizer}\n[dpt_head]\nembed_dim=1\nfusion=2\nhead_hidden=2\npatch_size=1\ntaps=4\nneck=1,1,1,1\nreassemble=4,2,1,0\n");
        let network = Network::from_config(ConfigFile::parse_str("native-dpt.cfg", &cfg).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::DptHead(p) = &mut cpu.params[0] else { panic!("DPT params") };
        p.values = values(p.values.len());
        let mut gpu = GpuExecutableNetwork::new(network, OxideBackend::new().unwrap());
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(shape, values(shape.batch * shape.element_count())).unwrap();
        for _ in 0..2 {
            cpu.zero_train_buffers(); gpu.zero_train_buffers_gpu().unwrap();
            let host = cpu.forward_training(&input).unwrap();
            let device = gpu.forward_training_device(&input).unwrap();
            assert!(device.cpu_forward.layer_outputs[0].data.is_empty());
            let expected = host.final_output().unwrap();
            close("output", &gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(), &expected.data, 3e-4);
            let seeds = [Some(Tensor::from_data(expected.shape, values(expected.len())).unwrap())];
            for _ in 0..2 {
                let expected = cpu.backward_accumulate_seeded(&input, &host, &seeds).unwrap();
                let actual = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device, &seeds).unwrap().unwrap();
                close("input gradient", &actual.data, &expected.input_gradient.data, 1e-3);
            }
            close("parameter accumulation", updates(&gpu.host_fallback_train_buffers()[0]), updates(&cpu.train_buffers[0]), 1e-3);
            cpu.update_weights(0.003, 0.9, 0.0005, 2);
            gpu.update_weights_gpu(0.003, 0.9, 0.0005, 2).unwrap();
            close("updated weights", weights(&gpu.params[0]), weights(&cpu.params[0]), 1e-3);
        }
    }
}
