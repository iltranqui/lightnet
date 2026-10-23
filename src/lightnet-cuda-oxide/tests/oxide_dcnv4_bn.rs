//! Real-device rolling-stat BN derivatives via DCNv4 runtime, not batch BN.
//! Supplied-seed SGD with lr=1 reads positive derivatives from weight motion;
//! FD forward calls hold rolling statistics fixed. No dataset training/golden.

use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor,
};
use lightnet_cuda_oxide::{GemmMode, OxideBackend};

fn backend() -> OxideBackend {
    OxideBackend::new().unwrap().with_gemm_mode(GemmMode::Fp32)
}
fn network(prefix: &str, dims: &str, body: &str) -> Network {
    Network::from_config(ConfigFile::parse_str("bn.cfg", &format!(
        "[net]\n{dims}\ntry_fix_nan=0\n{prefix}\n[dcnv4]\n{body}\nbatch_normalize=1\nactivation=linear\n"
    )).unwrap()).unwrap()
}
fn close(name: &str, got: f32, want: f32) {
    assert!(
        (got - want).abs() < 3e-3 + 1e-3 * want.abs(),
        "{name}: {got} vs {want}"
    );
}

#[test]
fn native_bn_runtime_forward_derivatives_hold_rolling_stats_fixed() {
    let dims = "width=4\nheight=3\nchannels=2\nbatch=2";
    let body =
        "size=3\npadding=1\ngroups=2\nd_stride=2\nsoftmax=1\nremove_center=1\noffset_scale=1.0";
    let net = network(
        "[convolutional]\nfilters=2\nsize=1\nactivation=linear",
        dims,
        body,
    );
    let mut gpu = GpuExecutableNetwork::new(net, backend());
    let LayerParams::Convolutional(c) = &mut gpu.params[0] else {
        unreachable!()
    };
    c.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    c.biases.fill(0.0);
    let LayerParams::Dcnv4(p) = &mut gpu.params[1] else {
        unreachable!()
    };
    for (i, w) in p.offset_weights.iter_mut().enumerate() {
        *w = (i % 3) as f32 * 0.0005 - 0.0005;
    }
    p.offset_biases.fill(0.5);
    p.biases.fill(7.0);
    let bn = p.batch_norm.as_mut().unwrap();
    bn.biases.fill(9.0);
    bn.scales.copy_from_slice(&[0.0, -1.3]);
    bn.rolling_mean.copy_from_slice(&[0.2, -0.1]);
    bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);
    let original = gpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    let shape = gpu.network.net.input_shape();
    let input = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| (i % 7) as f32 * 0.15 - 0.45)
            .collect(),
    )
    .unwrap();
    let forward = gpu.forward_training_with_state(&input).unwrap();
    let middle = forward.cpu_forward.layer_outputs[0].clone();
    let seed = Tensor::from_data(
        gpu.network.layers[1].output_shape,
        (0..input.data.len())
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.13)
            .collect(),
    )
    .unwrap();
    let seeds = [None, Some(seed.clone())];
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .unwrap();
    gpu.update_weights_gpu(1.0, 0.0, 0.0, 2).unwrap();
    let LayerParams::Convolutional(c) = &gpu.params[0] else {
        unreachable!()
    };
    let conv_grad = (c.weights[3] - 1.0) * 2.0;
    let LayerParams::Dcnv4(p) = &gpu.params[1] else {
        unreachable!()
    };
    let LayerParams::Dcnv4(old) = &original[1] else {
        unreachable!()
    };
    let dw: Vec<_> = p
        .offset_weights
        .iter()
        .zip(&old.offset_weights)
        .map(|(n, o)| (n - o) * 2.0 / 0.01)
        .collect();
    let db: Vec<_> = p
        .offset_biases
        .iter()
        .zip(&old.offset_biases)
        .map(|(n, o)| (n - o) * 2.0 / 0.01)
        .collect();
    let ds: Vec<_> = p
        .batch_norm
        .as_ref()
        .unwrap()
        .scales
        .iter()
        .zip(&old.batch_norm.as_ref().unwrap().scales)
        .map(|(n, o)| (n - o) * 2.0)
        .collect();
    assert!(ds[0].abs() > 1e-4, "gamma=0 must retain its derivative");
    assert_eq!(p.biases, old.biases);
    assert_eq!(
        p.batch_norm.as_ref().unwrap().biases,
        old.batch_norm.as_ref().unwrap().biases
    );
    let net = network("", dims, body);
    let mut fd = GpuExecutableNetwork::new(net, backend());
    let objective =
        |fd: &mut GpuExecutableNetwork<OxideBackend>, data: &Tensor, params: &LayerParams| -> f32 {
            fd.params[0] = params.clone(); // DCNv4 uploads each call; no cached conv here.
            let before = fd.params.clone();
            let f = fd.forward_training_with_state(data).unwrap();
            assert_eq!(fd.params, before, "forward-only FD must not change EMA");
            f.cpu_forward
                .final_output()
                .unwrap()
                .data
                .iter()
                .zip(&seed.data)
                .map(|(v, s)| v * s)
                .sum()
        };
    let eps = 0.002;
    // All sampling coordinates are integral+0.5 +/-0.01 from tiny projection
    // weights. Perturbations below cannot cross a bilinear floor-cell seam.
    let mut hi = middle.clone();
    let mut lo = middle.clone();
    for b in 0..2 {
        for q in 0..12 {
            let i = (b * 2 + 1) * 12 + q;
            hi.data[i] += eps * input.data[i];
            lo.data[i] -= eps * input.data[i];
        }
    }
    let numerical = (objective(&mut fd, &hi, &original[1]) - objective(&mut fd, &lo, &original[1]))
        / (2.0 * eps);
    close(
        "input-adjoint direction via upstream identity conv",
        conv_grad,
        numerical,
    );
    for (slot, indices) in [
        (0, vec![0, 1, 24 * 18, 26 * 18 + 4, 40 * 18 + 9]),
        (1, vec![0, 24, 26, 40]),
        (2, vec![0, 1]),
    ] {
        for i in indices {
            let mut hi = original[1].clone();
            let mut lo = original[1].clone();
            let (LayerParams::Dcnv4(h), LayerParams::Dcnv4(l)) = (&mut hi, &mut lo) else {
                unreachable!()
            };
            let actual = match slot {
                0 => {
                    h.offset_weights[i] += eps;
                    l.offset_weights[i] -= eps;
                    dw[i]
                }
                1 => {
                    h.offset_biases[i] += eps;
                    l.offset_biases[i] -= eps;
                    db[i]
                }
                _ => {
                    h.batch_norm.as_mut().unwrap().scales[i] += eps;
                    l.batch_norm.as_mut().unwrap().scales[i] -= eps;
                    ds[i]
                }
            };
            let numerical =
                (objective(&mut fd, &middle, &hi) - objective(&mut fd, &middle, &lo)) / (2.0 * eps);
            close(&format!("slot={slot} i={i}"), actual, numerical);
        }
    }
}

#[test]
fn native_bn_unseeded_and_single_sample_stats_match_cpu() {
    for (width, height, batch) in [(4, 3, 2), (1, 1, 1)] {
        let net = network(
            "",
            &format!("width={width}\nheight={height}\nchannels=2\nbatch={batch}"),
            "size=1\npadding=0\nsoftmax=0\nd_stride=2\noffset_scale=1.0",
        );
        let mut cpu = ExecutableNetwork::new(net.clone());
        let LayerParams::Dcnv4(p) = &mut cpu.params[0] else {
            unreachable!()
        };
        p.offset_biases[..3].copy_from_slice(&[0.47, 0.53, 0.8]);
        let bn = p.batch_norm.as_mut().unwrap();
        bn.scales.copy_from_slice(&[0.0, -1.3]);
        bn.rolling_mean.copy_from_slice(&[0.2, -0.1]);
        bn.rolling_variance.copy_from_slice(&[0.7, 1.3]);
        let mut gpu = GpuExecutableNetwork::new(net, backend());
        gpu.params = cpu.params.clone();
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(
            shape,
            (0..batch * shape.element_count())
                .map(|i| (i % 11) as f32 * 0.13 - 0.5)
                .collect(),
        )
        .unwrap();
        let cf = cpu.forward_training(&input).unwrap();
        let gf = gpu.forward_training_with_state(&input).unwrap();
        cpu.backward_accumulate_seeded(&input, &cf, &[None])
            .unwrap();
        gpu.backward_accumulate_seeded_gpu(&gf, &[None]).unwrap();
        let (LayerParams::Dcnv4(c), LayerParams::Dcnv4(g)) = (&cpu.params[0], &gpu.params[0])
        else {
            unreachable!()
        };
        let (c, g) = (
            c.batch_norm.as_ref().unwrap(),
            g.batch_norm.as_ref().unwrap(),
        );
        for (name, a, b) in [
            ("mean", &c.rolling_mean, &g.rolling_mean),
            ("var", &c.rolling_variance, &g.rolling_variance),
        ] {
            for (x, y) in a.iter().zip(b) {
                assert!((x - y).abs() < 2e-6, "{name}: {x} vs {y}");
            }
        }
        assert_eq!(c.scales, g.scales);
    }
}
