//! Native f32 act-head forward composition; backward/update remain unchanged.
use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor,
};
use lightnet_cuda_oxide::{GemmMode, OxideBackend};

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-4 * (1.0 + b.abs()), "{i}: {a} vs {b}");
    }
}

#[test]
fn laya_act_head_native_full_compact_observe_edits_blob_replacement_and_sources() {
    for (all_masked, aliased, refs) in [
        (false, false, "0,1,2"),
        (true, false, "-3,-2,-1"),
        (false, true, "0,1,0"),
    ] {
        let h = if aliased { 8 } else { 6 };
        let cfg = ConfigFile::parse_str(
            "native-act.cfg",
            &format!(
                "[net]\nwidth=4\nheight=1\nchannels={h}\nbatch=2\n\
             [convolutional]\nfilters={h}\nsize=1\nactivation=linear\n\
             [convolutional]\nfilters=1\nsize=1\nactivation=linear\n\
             [convolutional]\nfilters=8\nsize=1\nactivation=linear\n\
             [laya_act_head]\nlayers={refs}\nhidden={h}\nmax_options=4\nfc_dim=5\nn_act=3\n"
            ),
        )
        .unwrap();
        let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
        for (i, p) in cpu.params.iter_mut().enumerate() {
            match p {
                LayerParams::Convolutional(p) => {
                    p.weights.fill(0.0);
                    p.biases.fill(0.0);
                    if i == 0 {
                        for c in 0..h {
                            p.weights[c * h + c] = 1.0;
                        }
                    } else if i == 1 {
                        p.weights[0] = 1.0;
                    } else if !all_masked {
                        p.biases.fill(1.0);
                    }
                }
                LayerParams::LayaActHead(p) => {
                    for (i, w) in p.fc1_w.iter_mut().enumerate() {
                        *w = ((i % 11) as f32 - 5.0) * 0.04;
                    }
                    p.fc1_b.copy_from_slice(&[-1.0, 0.0, 1.0, -0.5, 0.5]);
                    for (i, w) in p.fc2_w.iter_mut().enumerate() {
                        *w = ((i % 7) as f32 - 3.0) * 0.07;
                    }
                    p.fc2_b.copy_from_slice(&[0.1, -0.2, 0.3]);
                }
                _ => {}
            }
        }
        let mut gpu = GpuExecutableNetwork::new(
            cpu.network.clone(),
            OxideBackend::new().unwrap().with_gemm_mode(GemmMode::Fp32),
        );
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(
            shape,
            (0..2 * h * 4)
                .map(|i| (i % 17) as f32 * 0.125 - 0.75)
                .collect(),
        )
        .unwrap();
        let mut previous = Vec::new();
        for stage in 0..3 {
            if stage == 1 {
                let LayerParams::LayaActHead(p) = &mut cpu.params[3] else {
                    panic!("act")
                };
                p.fc1_w[0] += 0.75;
                p.fc1_b[0] -= 0.25;
                p.fc2_w[0] -= 0.6;
                p.fc2_b[0] += 0.5;
            } else if stage == 2 {
                let LayerParams::LayaActHead(p) = &cpu.params[3] else {
                    panic!("act")
                };
                let blob: Vec<_> = p.to_blob().into_iter().map(|v| v * 0.3 + 0.01).collect();
                cpu.params[3] = LayerParams::LayaActHead(
                    lightnet_core::laya_act_head::LayaActHeadWeights::from_blob(&blob, h, 5, 3)
                        .unwrap(),
                );
            }
            gpu.params = cpu.params.clone();
            let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
            if stage > 0 {
                assert_ne!(
                    expected.data, previous,
                    "test must exercise changed outputs"
                );
            }
            previous = expected.data.clone();
            let full = gpu.forward(&input).unwrap().final_output().unwrap().clone();
            let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
            assert_eq!(full.shape, expected.shape);
            assert_eq!(compact.shape, expected.shape);
            close(&full.data, &expected.data);
            close(&compact.data, &expected.data);
        }
    }
}

#[test]
fn laya_act_head_native_invalid_parameters_error_cleanly() {
    let cfg=ConfigFile::parse_str("bad-act.cfg","[net]\nwidth=2\nheight=1\nchannels=4\n[convolutional]\nfilters=4\nsize=1\nactivation=linear\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[laya_act_head]\nlayers=0,1,0\nhidden=4\nmax_options=2\nfc_dim=3\nn_act=2\n").unwrap();
    let mut gpu = GpuExecutableNetwork::new(
        Network::from_config(cfg).unwrap(),
        OxideBackend::new().unwrap(),
    );
    let LayerParams::LayaActHead(p) = &mut gpu.params[2] else {
        panic!("act")
    };
    p.fc1_w.pop();
    let input = Tensor::zeros(gpu.network.net.input_shape());
    for err in [
        gpu.forward(&input).unwrap_err(),
        gpu.forward_compact(&input).unwrap_err(),
    ] {
        assert!(
            err.to_string().contains("act-head parameter fc1_w"),
            "{err}"
        );
    }
    // Compact f16 validates the same frozen head tensors before dispatch.
    let err = gpu.forward_compact_f16(&input).unwrap_err().to_string();
    assert!(err.contains("act-head parameter fc1_w"), "{err}");
}
