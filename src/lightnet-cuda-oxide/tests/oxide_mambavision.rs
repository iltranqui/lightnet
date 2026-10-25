//! Source-only parity fixtures; these have not been executed.
//! Rust CPU reference, no compiled-C++ golden. Relative tolerances use
//! max(1, max(abs(reference))): 2e-4 output, 8e-4 gradients/updates.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams,
    LayerTrainBuffers, Network, Tensor, TensorShape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::mambavision::{self, MambaScanGeometry};
use lightnet_cuda_oxide::OxideBackend;

fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + phase) * 0.37).sin() * 0.15).collect()
}

fn close(label: &str, got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len(), "{label}");
    let scale = expected.iter().fold(1.0_f32, |a, x| a.max(x.abs()));
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(g.is_finite() && e.is_finite() && (g - e).abs() <= tolerance * scale,
            "{label}[{i}]: {g} vs {e}");
    }
}

#[test]
fn selective_scan_nonzero_state_and_final_state_adjoint_match_reference() {
    let be = OxideBackend::new().unwrap();
    let cfg = ElementwiseLaunchConfig::default();
    let g = MambaScanGeometry { batch: 2, tokens: 5, channels: 3, state: 2, rank: 2 };
    let x = values(30, 1.0);
    let projection = values(10 * g.width(), 2.0);
    let w = values(6, 3.0);
    let bias = values(3, 4.0);
    let a = values(6, 5.0);
    let skip = values(3, 6.0);
    let state = values(12, 7.0);
    let dy = values(30, 8.0);
    let ds = values(12, 9.0);
    let up = |v: &[f32]| be.upload_f32(&[v.len()], v).unwrap();
    let (xt, pt, wt, bt, at, skt, st, dyt, dst) =
        (up(&x), up(&projection), up(&w), up(&bias), up(&a), up(&skip), up(&state), up(&dy), up(&ds));
    let reference = mambavision::scan(&x, &projection, &w, &bias, &a, &skip, &state, g);
    let forward = be.mambavision_scan_f32(&xt, &pt, &wt, &bt, &at, &skt, &st, g, cfg).unwrap();
    for (label, actual, expected) in [
        ("output", &forward.output, &reference.output),
        ("states", &forward.states, &reference.states),
        ("final state", &forward.state_out, &reference.state_out),
    ] { close(label, &be.download_f32(actual).unwrap(), expected, 2e-5); }
    let expected = mambavision::scan_backward(&x, &projection, &w, &a, &skip, &state,
        &reference.dt_pre, &reference.states, &dy, &ds, g);
    let actual = be.mambavision_scan_backward_f32(&xt, &pt, &wt, &at, &skt, &st,
        &forward.dt_pre, &forward.states, &dyt, &dst, g, cfg).unwrap();
    for (label, actual, expected) in [
        ("input", &actual.input, &expected.input),
        ("selective", &actual.projection, &expected.projection),
        ("dt weights", &actual.dt_weights, &expected.dt_weights),
        ("dt bias", &actual.dt_bias, &expected.dt_bias),
        ("A log", &actual.a_log, &expected.a_log),
        ("skip", &actual.d_skip, &expected.d_skip),
        ("initial state", &actual.state_in, &expected.state_in),
    ] { close(label, &be.download_f32(actual).unwrap(), expected, 8e-5); }
}

fn updates(buffers: &LayerTrainBuffers) -> &[f32] {
    let LayerTrainBuffers::MambaVision(t) = buffers else { panic!("Mamba buffers") };
    &t.updates
}

fn blob(params: &LayerParams) -> &[f32] {
    let LayerParams::MambaVision(p) = params else { panic!("Mamba params") };
    &p.values
}

#[test]
fn mambavision_native_runtime_accumulation_and_two_updates_match_cpu() {
    // Batch > 1 and non-square spatial geometry pin the flat-BDT projection.
    // Cover both residual projection shapes, centered/causal conv and MSA.
    for (channels, filters, causal, msa, optimizer) in [
        (3, 4, 0, 0, ""), (4, 4, 1, 0, "adam=1\n"),
        (3, 4, 1, 1, ""), (4, 4, 0, 1, "muon=1\ndecoupled_decay=1\n"),
    ] {
        let text = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels={channels}\n{optimizer}\n[mambavision]\nfilters={filters}\nstate=2\nconv_size=3\ndt_rank=2\nffn_ratio=2\nactivation=gelu\ncausal={causal}\nmsa={msa}\nmsa_heads=2\nmsa_block_size=2\nmsa_topk=2\n");
        let network = Network::from_config(ConfigFile::parse_str("mamba-native.cfg", &text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        if let LayerParams::MambaVision(p) = &mut cpu.params[0] {
            p.values = values(p.values.len(), 3.0);
        } else { panic!("Mamba params"); }
        let mut gpu = GpuExecutableNetwork::new(network, OxideBackend::new().unwrap());
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        let shape = TensorShape::new(3, 2, channels, 2);
        let input = Tensor::from_data(shape, values(shape.batch * shape.element_count(), 11.0)).unwrap();
        for _ in 0..2 {
            cpu.zero_train_buffers();
            gpu.zero_train_buffers_gpu().unwrap();
            let host = cpu.forward_training(&input).unwrap();
            // Empty host outputs demonstrate this did not select CPU fallback.
            let device = gpu.forward_training_device(&input).unwrap();
            assert!(device.cpu_forward.layer_outputs[0].data.is_empty());
            let expected = host.final_output().unwrap();
            let actual = gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap();
            close("forward", &actual, &expected.data, 2e-4);
            let seed = Tensor::from_data(expected.shape, values(expected.len(), 21.0)).unwrap();
            let seeds = [Some(seed)];
            for _ in 0..2 {
                let reference = cpu.backward_accumulate_seeded(&input, &host, &seeds).unwrap();
                let actual = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device, &seeds)
                    .unwrap().expect("native Mamba input gradient");
                close("input gradient", &actual.data, &reference.input_gradient.data, 8e-4);
            }
            close("accumulated parameter gradients", updates(&gpu.host_fallback_train_buffers()[0]),
                updates(&cpu.train_buffers[0]), 8e-4);
            cpu.update_weights(0.003, 0.9, 0.0005, 2);
            gpu.update_weights_gpu(0.003, 0.9, 0.0005, 2).unwrap();
            close("updated parameters", blob(&gpu.params[0]), blob(&cpu.params[0]), 8e-4);
        }
    }
}
