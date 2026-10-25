//! Unrun source parity fixtures for native Kimi KDA, against the Rust CPU
//! reference. No compiled-C++ golden and no performance claim.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams,
    LayerTrainBuffers, Network, Tensor, TensorShape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::kimi_kda::{self, KdaGeometry};
use lightnet_cuda_oxide::OxideBackend;

fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.29).cos() * 0.2).collect()
}
fn close(label: &str, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len(), "{label}");
    let scale = expected.iter().fold(1.0_f32, |m, x| m.max(x.abs()));
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite() && (a-b).abs() <= tolerance * scale,
            "{label}[{i}]: {a} vs {b}");
    }
}

#[test]
fn scan_forward_and_all_projection_adjoints_match_host_reference() {
    let be = OxideBackend::new().unwrap();
    let cfg = ElementwiseLaunchConfig::default();
    for rank in [1, 2, 3] {
        let g = KdaGeometry { batch: 2, tokens: 5, channels: 3, rank };
        let (q, k, v, beta, decay, dy) = (values(10*rank, 1.0), values(10*rank, 2.0),
            values(30, 3.0), values(10*rank, 4.0), values(10*rank, 5.0), values(30, 6.0));
        let up = |x: &[f32]| be.upload_f32(&[x.len()], x).unwrap();
        let (qt, kt, vt, bt, dt, dyt) = (up(&q), up(&k), up(&v), up(&beta), up(&decay), up(&dy));
        let host = kimi_kda::scan(&q, &k, &v, &beta, &decay, g);
        let device = be.kimi_scan_f32(&qt, &kt, &vt, &bt, &dt, g, cfg).unwrap();
        close("scan", &be.download_f32(&device.output).unwrap(), &host.output, 2e-5);
        close("states", &be.download_f32(&device.states).unwrap(), &host.states, 2e-5);
        let expected = kimi_kda::backward(&q, &v, &beta, &decay, &host, &dy, g);
        let got = be.kimi_scan_backward_f32(&qt, &vt, &bt, &dt, &device, &dyt, g, cfg).unwrap();
        for (label, a, b) in [("q", &got.q, &expected.q), ("k", &got.k, &expected.k),
            ("v", &got.v, &expected.v), ("beta", &got.beta, &expected.beta),
            ("decay", &got.decay, &expected.decay)] {
            close(label, &be.download_f32(a).unwrap(), b, 8e-5);
        }
    }
}

#[test]
fn topk_ties_select_exactly_k_by_index() {
    let be = OxideBackend::new().unwrap();
    let logits = [0.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let x = be.upload_f32(&[2, 3], &logits).unwrap();
    for topk in [0, 1, 2, 3] {
        let got = be.kimi_router_gates_f32(&x, 2, 3, topk, 0.7, ElementwiseLaunchConfig::default()).unwrap();
        let got = be.download_f32(&got).unwrap();
        close("router", &got, &kimi_kda::gates(&logits, 2, 3, topk, 0.7), 1e-6);
        for row in got.chunks_exact(3) {
            assert_eq!(row.iter().filter(|&&x| x > 0.0).count(), if topk == 0 { 3 } else { topk });
        }
    }
}

fn updates(buffer: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::KimiKda(t) = buffer else { panic!("Kimi buffers") };
    [t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()
}
fn blob(p: &LayerParams) -> Vec<f32> {
    let LayerParams::KimiKda(p) = p else { panic!("Kimi params") };
    p.to_blob().unwrap()
}

#[test]
fn kimi_native_runtime_two_step_outputs_gradients_and_updates_match_cpu() {
    for (topk, optimizer) in [(0, ""), (1, "adam=1\n"), (2, "muon=1\ndecoupled_decay=1\n")] {
        let text = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=3\n{optimizer}\n[kimi_kda]\nfilters=3\nrank=2\nexperts=3\ntop_k={topk}\nhidden=4\ntemperature=0.7\nload_balance=0.03\nactivation=gelu\n");
        let network = Network::from_config(ConfigFile::parse_str("native-kimi.cfg", &text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::KimiKda(p) = &mut cpu.params[0] else { panic!("Kimi params") };
        p.weights = values(p.weights.len(), 3.0);
        p.biases = values(p.biases.len(), 4.0);
        let mut gpu = GpuExecutableNetwork::new(network, OxideBackend::new().unwrap());
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        let shape = TensorShape::new(3, 2, 3, 2);
        let input = Tensor::from_data(shape, values(36, 5.0)).unwrap();
        for _ in 0..2 {
            cpu.zero_train_buffers(); gpu.zero_train_buffers_gpu().unwrap();
            let host = cpu.forward_training(&input).unwrap();
            let device = gpu.forward_training_device(&input).unwrap();
            assert!(device.cpu_forward.layer_outputs[0].data.is_empty());
            let expected = host.final_output().unwrap();
            close("output", &gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(), &expected.data, 3e-4);
            let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
            close("compact forward", &compact.data, &expected.data, 3e-4);
            let seeds = [Some(Tensor::from_data(expected.shape, values(expected.len(), 9.0)).unwrap())];
            for _ in 0..2 {
                let expected = cpu.backward_accumulate_seeded(&input, &host, &seeds).unwrap();
                let got = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device, &seeds).unwrap().unwrap();
                close("input gradient", &got.data, &expected.input_gradient.data, 1e-3);
            }
            close("parameter accumulation", &updates(&gpu.host_fallback_train_buffers()[0]), &updates(&cpu.train_buffers[0]), 1e-3);
            cpu.update_weights(0.003, 0.9, 0.0005, 2);
            gpu.update_weights_gpu(0.003, 0.9, 0.0005, 2).unwrap();
            close("parameters", &blob(&gpu.params[0]), &blob(&cpu.params[0]), 1e-3);
        }
    }
}
