use super::*;

#[test]
fn laya_act_head_rejects_shape_mismatches() {
    let net = "[net]\nchannels=6\nbatch=1\nheight=1\nwidth=2\n\
                   [convolutional]\nfilters=6\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [convolutional]\nfilters=1\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [convolutional]\nfilters=4\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n";
    let build = |section: &str| {
        ConfigFile::parse_str("laya-ah.cfg", &format!("{net}{section}"))
            .map_err(|err| err.to_string())
            .and_then(|cfg| {
                Network::from_config(cfg)
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            })
    };
    // head source must have `hidden` channels.
    assert!(build("[laya_act_head]\nlayers=0,1,2\nhidden=5\nmax_options=2\n").is_err());
    // scorer source must be 1 x max_options.
    assert!(build("[laya_act_head]\nlayers=0,1,2\nhidden=6\nmax_options=3\n").is_err());
    // marker source must have 2*max_options channels.
    assert!(build("[laya_act_head]\nlayers=0,1,2\nhidden=6\nmax_options=1\n").is_err());
    // exactly three sources.
    assert!(build("[laya_act_head]\nlayers=0,1\nhidden=6\nmax_options=2\n").is_err());
    // the valid shape builds.
    assert!(build("[laya_act_head]\nlayers=0,1,2\nhidden=6\nmax_options=2\n").is_ok());
}

pub(super) fn laya_loss_network(mode: &str) -> ExecutableNetwork {
    let cfg = format!(
        "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=2\n\
             [convolutional]\nfilters=1\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
             [laya_loss]\nmax_options=2\nmode={mode}\n"
    );
    let mut executable = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("laya-loss.cfg", &cfg).expect("cfg"))
            .expect("network"),
    );
    let LayerParams::Convolutional(p) = &mut executable.params[0] else {
        panic!("expected conv params")
    };
    p.weights = vec![1.0];
    p.biases = vec![0.0];
    executable
}

pub(super) fn laya_loss_truth() -> Tensor {
    // [target K=2 | valid K=2 | qtype | ep_group | ep_step] -> 7 values.
    Tensor::from_data(
        TensorShape::new(7, 1, 1, 1),
        vec![1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0],
    )
    .expect("truth")
}

#[test]
fn cpu_laya_loss_ce_forward_and_seed() {
    let executable = laya_loss_network("ce");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![0.5, 2.0]).expect("input");
    let forward = executable.forward(&input).expect("forward");
    let probs = &forward.layer_outputs[1];
    let e1 = (2.0f32 - 0.5).exp();
    let p = [1.0 / (1.0 + e1), e1 / (1.0 + e1)];
    assert!((probs.data[0] - p[0]).abs() < 1e-6, "{:?}", probs.data);
    assert!((probs.data[1] - p[1]).abs() < 1e-6);

    let mut truths = vec![None; executable.network.layers.len()];
    truths[1] = Some(laya_loss_truth());
    let loss = crate::dense_loss::compute_dense_training_loss(
        &executable.network,
        &forward,
        &input,
        &truths,
    )
    .expect("dense loss");
    let seed = loss.layer_output_gradient_seeds[1]
        .as_ref()
        .expect("laya_loss seed");
    // CE delta = y - p.
    assert!(
        (seed.data[0] - (1.0 - p[0])).abs() < 1e-6,
        "{:?}",
        seed.data
    );
    assert!((seed.data[1] - (0.0 - p[1])).abs() < 1e-6);
}

#[test]
fn cpu_laya_loss_rlcd_runs_and_produces_a_finite_seed() {
    let executable = laya_loss_network("rlcd");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![0.5, 2.0]).expect("input");
    let forward = executable.forward(&input).expect("forward");
    let mut truths = vec![None; executable.network.layers.len()];
    truths[1] = Some(laya_loss_truth());
    let loss = crate::dense_loss::compute_dense_training_loss_at(
        &executable.network,
        &forward,
        &input,
        &truths,
        0,
    )
    .expect("dense loss");
    let seed = loss.layer_output_gradient_seeds[1]
        .as_ref()
        .expect("laya_loss seed");
    assert!(seed.data.iter().all(|v| v.is_finite()), "{:?}", seed.data);
    assert!(loss.loss.is_finite());
}

#[test]
fn laya_loss_rejects_bad_configs() {
    let net = "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=2\n\
                   [convolutional]\nfilters=1\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n";
    let build = |section: &str| {
        ConfigFile::parse_str("laya-loss.cfg", &format!("{net}{section}"))
            .map_err(|err| err.to_string())
            .and_then(|cfg| {
                Network::from_config(cfg)
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            })
    };
    assert!(build("[laya_loss]\nmax_options=2\nmode=bad\n").is_err());
    assert!(build("[laya_loss]\nmax_options=3\nmode=ce\n").is_err());
    assert!(build("[laya_loss]\nmax_options=2\nmode=ce\n").is_ok());
}

pub(super) fn laya_encoder_network() -> ExecutableNetwork {
    let cfg = "[net]\nchannels=2\nbatch=1\nheight=1\nwidth=3\n\
                   [laya_encoder]\nhidden=4\nheads=2\nlayers=2\nintermediate=8\nvocab=10\n";
    let mut executable = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("laya-enc.cfg", cfg).expect("cfg"))
            .expect("network"),
    );
    let LayerParams::LayaEncoder(w) = &mut executable.params[0] else {
        panic!("expected laya_encoder params")
    };
    let mut state = 0x1234_5678u64;
    let mut fill = |len: usize| {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect::<Vec<f32>>()
    };
    *w = crate::laya_encoder::LayaEncoderWeights {
        tok_embed: fill(10 * 4),
        embed_norm: fill(4),
        attn_norm: fill(1 * 4),
        wqkv: fill(2 * 3 * 4 * 4),
        wo: fill(2 * 4 * 4),
        mlp_norm: fill(2 * 4),
        wi: fill(2 * 2 * 8 * 4),
        wo_mlp: fill(2 * 4 * 8),
        final_norm: fill(4),
    };
    executable
}

#[test]
fn cpu_laya_encoder_forward_matches_the_module() {
    let executable = laya_encoder_network();
    // token ids [0,1,2], mask [1,1,0].
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![0.0, 1.0, 2.0, 1.0, 1.0, 0.0],
    )
    .expect("input");
    let forward = executable.forward(&input).expect("forward");
    let LayerParams::LayaEncoder(weights) = &executable.params[0] else {
        panic!("expected laya_encoder params")
    };
    let geometry = crate::laya_encoder::LayaEncoderGeometry {
        batch: 1,
        seq_len: 3,
        hidden: 4,
        heads: 2,
        n_layers: 2,
        intermediate: 8,
        vocab: 10,
        local_window: crate::model::LAYA_ENCODER_LOCAL_WINDOW,
        rope_theta_full: crate::model::LAYA_ENCODER_ROPE_THETA_FULL,
        rope_theta_local: crate::model::LAYA_ENCODER_ROPE_THETA_LOCAL,
        ln_eps: crate::model::LAYA_LAYERNORM_EPS,
    };
    let want = crate::laya_encoder::encoder_forward(&input.data, weights, &geometry)
        .expect("module forward");
    assert_eq!(forward.layer_outputs[0].data, want);
    assert_eq!(forward.layer_outputs[0].data.len(), 4 * 3);
}

#[test]
fn laya_encoder_rejects_bad_configs() {
    let build = |channels: usize, section: &str| {
        let net = format!("[net]\nchannels={channels}\nbatch=1\nheight=1\nwidth=3\n");
        ConfigFile::parse_str("laya-enc.cfg", &format!("{net}{section}"))
            .map_err(|err| err.to_string())
            .and_then(|cfg| {
                Network::from_config(cfg)
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            })
    };
    // channels must be 2 (token id + mask).
    assert!(
        build(
            1,
            "[laya_encoder]\nhidden=2\nheads=1\nlayers=1\nintermediate=2\nvocab=3\n"
        )
        .is_err()
    );
    // hidden must be divisible by heads.
    assert!(
        build(
            2,
            "[laya_encoder]\nhidden=3\nheads=2\nlayers=1\nintermediate=2\nvocab=3\n"
        )
        .is_err()
    );
    // valid.
    assert!(
        build(
            2,
            "[laya_encoder]\nhidden=2\nheads=1\nlayers=1\nintermediate=2\nvocab=3\n"
        )
        .is_ok()
    );
}

pub(super) fn laya_head_network() -> ExecutableNetwork {
    let cfg = "[net]\nchannels=5\nbatch=1\nheight=1\nwidth=3\n\
                   [laya_head]\nhidden=4\nheads=2\nlayers=2\nffn_dim=4\nqtype=0\n";
    let mut executable = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("laya-head.cfg", cfg).expect("cfg"))
            .expect("network"),
    );
    let LayerParams::LayaHead(w) = &mut executable.params[0] else {
        panic!("expected laya_head params")
    };
    let mut state = 0x9E37_79B9u64;
    let mut fill = |len: usize| {
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect::<Vec<f32>>()
    };
    *w = crate::laya_head::LayaHeadWeights {
        norm1_w: fill(2 * 4),
        norm1_b: fill(2 * 4),
        in_proj_w: fill(2 * 3 * 4 * 4),
        in_proj_b: fill(2 * 3 * 4),
        out_proj_w: fill(2 * 4 * 4),
        out_proj_b: fill(2 * 4),
        norm2_w: fill(2 * 4),
        norm2_b: fill(2 * 4),
        linear1_w: fill(2 * 4 * 4),
        linear1_b: fill(2 * 4),
        linear2_w: fill(2 * 4 * 4),
        linear2_b: fill(2 * 4),
        type_emb: fill(3 * 4),
    };
    executable
}

#[test]
fn cpu_laya_head_forward_matches_the_module() {
    let executable = laya_head_network();
    // [batch, hidden+mask=5, 1, T=3]: hidden states + mask channel.
    let mut input_data = vec![0.0f32; 5 * 3];
    for (i, v) in input_data.iter_mut().enumerate() {
        *v = (i as f32 * 0.17).sin();
    }
    for t in 0..3 {
        input_data[4 * 3 + t] = 1.0; // mask all valid
    }
    let input =
        Tensor::from_data(executable.network.net.input_shape(), input_data.clone()).expect("input");
    let forward = executable.forward(&input).expect("forward");
    let LayerParams::LayaHead(weights) = &executable.params[0] else {
        panic!("expected laya_head params")
    };
    let geometry = crate::laya_head::LayaHeadGeometry {
        batch: 1,
        seq_len: 3,
        hidden: 4,
        heads: 2,
        n_layers: 2,
        ffn_dim: 4,
        ln_eps: crate::model::LAYA_LAYERNORM_EPS,
        qtype: 0,
    };
    let want =
        crate::laya_head::head_forward(&input_data, weights, &geometry).expect("module forward");
    assert_eq!(forward.layer_outputs[0].data, want);
    assert_eq!(forward.layer_outputs[0].data.len(), 4 * 3);
}

#[test]
fn laya_head_rejects_bad_configs() {
    let build = |channels: usize, section: &str| {
        let net = format!("[net]\nchannels={channels}\nbatch=1\nheight=1\nwidth=3\n");
        ConfigFile::parse_str("laya-head.cfg", &format!("{net}{section}"))
            .map_err(|err| err.to_string())
            .and_then(|cfg| {
                Network::from_config(cfg)
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            })
    };
    // channels must be hidden + mask = 5 for qtype=0.
    assert!(
        build(
            4,
            "[laya_head]\nhidden=4\nheads=2\nlayers=2\nffn_dim=4\nqtype=0\n"
        )
        .is_err()
    );
    // qtype=-1 adds a channel (6).
    assert!(
        build(
            6,
            "[laya_head]\nhidden=4\nheads=2\nlayers=2\nffn_dim=4\nqtype=-1\n"
        )
        .is_ok()
    );
    // qtype out of range.
    assert!(
        build(
            5,
            "[laya_head]\nhidden=4\nheads=2\nlayers=2\nffn_dim=4\nqtype=3\n"
        )
        .is_err()
    );
}

#[test]
fn eml_convolutional_axis_layout_preserves_batch_channel_and_spatial_order() {
    let cfg = "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=3\n\
                   [eml_convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\
                   \nresidual=0\neml_scale=1\neml_clamp=4\neml_eps=0.000001\n";
    let network =
        Network::from_config(ConfigFile::parse_str("eml-axis.cfg", cfg).expect("EML cfg parses"))
            .expect("EML network builds");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::EmlConvolutional(params) = &mut executable.params[0] else {
        panic!("expected EML parameters");
    };
    // [out_channel, in_channel, ky, kx]: one branch preserves channels,
    // the other swaps them, making channel-axis transposition observable.
    params.input.weights.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);
    params.input.biases.copy_from_slice(&[0.1, -0.2]);
    params
        .self_params
        .weights
        .copy_from_slice(&[0.0, 1.0, 1.0, 0.0]);
    params.self_params.biases.copy_from_slice(&[0.3, -0.4]);

    // B=2,C=2,H=2,W=3, NCHW with distinct batch/channel/spatial values.
    let input_values = vec![
        0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1.0, -2.0, 0.5, 0.0, -1.0, 2.0, -1.5, 0.25, 1.5, -0.25,
        2.5, -2.5, 0.75, -0.75, 1.25, -1.25, 3.0, -3.0,
    ];
    let shape = executable.network.net.input_shape();
    let input = Tensor::from_data(shape, input_values.clone()).expect("EML input shape");
    let forward = executable.forward(&input).expect("EML forward");
    let output = forward.layer_outputs.get(0).expect("EML output exists");
    assert_eq!(output.shape, shape);

    let spatial = 6;
    let channels = 2;
    let eps = 1.0e-6;
    let mut expected = Vec::with_capacity(input_values.len());
    for b in 0..2 {
        for c in 0..channels {
            for p in 0..spatial {
                let x = input_values[b * channels * spatial + c * spatial + p]
                    + if c == 0 { 0.1 } else { -0.2 };
                let y = input_values[b * channels * spatial + (1 - c) * spatial + p]
                    + if c == 0 { 0.3 } else { -0.4 };
                let softplus_y = y.exp().ln_1p();
                expected.push(x.exp() - (softplus_y + eps).ln());
            }
        }
    }
    for (i, (got, want)) in output.data.iter().zip(expected).enumerate() {
        assert!((got - want).abs() < 2e-5, "output[{i}]={got}, want {want}");
    }
}

#[test]
fn eml_convolutional_cpu_math_gradients_and_gpu_device_sgd_are_integrated() {
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=2\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\nresidual=1\n";
    let build = || {
        let network = Network::from_config(
            ConfigFile::parse_str("eml-runtime.cfg", cfg).expect("EML cfg parses"),
        )
        .expect("EML network builds");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::EmlConvolutional(params) = &mut executable.params[0] else {
            panic!("expected EML parameters");
        };
        params.input.weights[0] = 0.4;
        params.input.biases[0] = 0.1;
        params.self_params.weights[0] = -0.3;
        params.self_params.biases[0] = 0.2;
        let input_bn = params.input.batch_norm.as_mut().expect("input branch BN");
        input_bn.scales[0] = 0.8;
        input_bn.rolling_mean[0] = -0.1;
        input_bn.rolling_variance[0] = 0.9;
        let self_bn = params
            .self_params
            .batch_norm
            .as_mut()
            .expect("self branch BN");
        self_bn.scales[0] = 1.2;
        self_bn.rolling_mean[0] = 0.05;
        self_bn.rolling_variance[0] = 1.3;
        executable
    };
    fn perturb(executable: &mut ExecutableNetwork, branch: usize, weight: bool, value: f32) {
        let LayerParams::EmlConvolutional(params) = &mut executable.params[0] else {
            unreachable!()
        };
        let conv = if branch == 0 {
            &mut params.input
        } else {
            &mut params.self_params
        };
        if weight {
            conv.weights[0] += value;
        } else {
            conv.biases[0] += value;
        }
    }

    let input =
        Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![0.25, -0.4]).expect("input shape");
    let seed =
        Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![0.7, -0.35]).expect("seed shape");
    let executable = build();
    let forward = executable.forward(&input).expect("EML CPU forward");
    let objective = |executable: &ExecutableNetwork| {
        executable
            .forward(&input)
            .expect("EML perturbed forward")
            .final_output()
            .unwrap()
            .data
            .iter()
            .zip(&seed.data)
            .map(|(out, delta)| out * delta)
            .sum::<f32>()
    };
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("EML CPU backward");
    let LayerParams::EmlConvolutional(analytic) = &backward.param_gradients[0] else {
        panic!("expected EML parameter gradients");
    };
    let eps = 1e-3;
    for branch in 0..2 {
        for weight in [true, false] {
            let mut plus = build();
            perturb(&mut plus, branch, weight, eps);
            let mut minus = build();
            perturb(&mut minus, branch, weight, -eps);
            let numeric = (objective(&plus) - objective(&minus)) / (2.0 * eps);
            let conv = if branch == 0 {
                &analytic.input
            } else {
                &analytic.self_params
            };
            let got = if weight {
                conv.weights[0]
            } else {
                conv.biases[0]
            };
            assert!(
                (got - numeric).abs() <= 2e-3 * numeric.abs().max(1.0),
                "branch={branch} weight={weight}: analytic={got}, numeric={numeric}"
            );
        }
    }

    let mut cpu_optimizer = build();
    let cpu_forward = cpu_optimizer
        .forward(&input)
        .expect("EML optimizer forward");
    cpu_optimizer
        .backward_accumulate_seeded(&input, &cpu_forward, &[Some(seed.clone())])
        .expect("EML CPU accumulated backward");
    cpu_optimizer.update_weights(0.01, 0.0, 0.0, 1);

    let mut gpu = GpuExecutableNetwork::new(executable.network.clone(), FakeBackend);
    gpu.params = executable.params.clone();
    let gpu_forward = gpu.forward_with_state(&input).expect("EML device forward");
    assert_eq!(
        gpu_forward.cpu_forward.final_output(),
        forward.final_output()
    );
    assert!(!gpu.requires_cpu_backward_fallback());
    gpu.zero_train_buffers_gpu().expect("clear GPU buffers");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[Some(seed.clone())])
        .expect("EML device backward");
    let device = gpu.device_eml_train_buffers.borrow();
    let train = device[0].as_ref().expect("EML device train buffers");
    assert!(
        train
            .input
            .weight_updates
            .data
            .iter()
            .any(|value| *value != 0.0)
    );
    assert!(
        train
            .self_params
            .weight_updates
            .data
            .iter()
            .any(|value| *value != 0.0)
    );
    drop(device);
    let before = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => params.input.weights[0],
        _ => unreachable!(),
    };
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("EML device SGD update");
    let after = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => params.input.weights[0],
        _ => unreachable!(),
    };
    assert_ne!(before, after, "device optimizer must update EML weights");
    let LayerParams::EmlConvolutional(cpu_params) = &cpu_optimizer.params[0] else {
        unreachable!()
    };
    assert!((after - cpu_params.input.weights[0]).abs() < 1e-6);
    assert_eq!(&gpu.params[0], &cpu_optimizer.params[0]);
}

#[test]
fn eml_sgd_uses_host_optimizer_when_another_layer_requires_cpu_fallback() {
    // `revleaky` has no device activation, so the ViT forces the CPU fallback.
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=4\nwidth=4\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=0\nresidual=0\n\
                   [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\nactivation=revleaky\n";
    let network = Network::from_config(
        ConfigFile::parse_str("eml-fallback.cfg", cfg).expect("fallback cfg parses"),
    )
    .expect("fallback network builds");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.init_train_buffers_gpu()
        .expect("fallback train buffers");
    assert!(gpu.requires_cpu_backward_fallback());

    let before = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => {
            (params.input.weights[0], params.self_params.biases[0])
        }
        _ => unreachable!(),
    };
    {
        let mut buffers = gpu.cpu_fallback_train_buffers.borrow_mut();
        let LayerTrainBuffers::EmlConvolutional(train) = &mut buffers[0] else {
            panic!("expected EML CPU-fallback buffers");
        };
        train.input.weight_updates[0] = 1.0;
        train.self_params.bias_updates[0] = 2.0;
    }
    gpu.update_weights_gpu(0.1, 0.0, 0.0, 1)
        .expect("host fallback EML SGD update");
    let after = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => {
            (params.input.weights[0], params.self_params.biases[0])
        }
        _ => unreachable!(),
    };
    assert_ne!(before, after, "CPU fallback must update both EML branches");
    assert!(
        gpu.device_eml_train_buffers.borrow()[0]
            .as_ref()
            .unwrap()
            .input
            .weight_updates
            .data
            .iter()
            .all(|value| *value == 0.0)
    );
}

#[test]
fn eml_device_gradients_participate_in_loss_scale_finiteness_check() {
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=1\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=1\n";
    let network = Network::from_config(
        ConfigFile::parse_str("eml-finite.cfg", cfg).expect("EML finite cfg parses"),
    )
    .expect("EML finite network builds");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.zero_train_buffers_gpu().expect("EML device buffers");
    gpu.device_eml_train_buffers.borrow_mut()[0]
        .as_mut()
        .expect("EML train buffers")
        .input
        .weight_updates
        .data[0] = f32::NAN;
    assert!(
        gpu.gradients_non_finite()
            .expect("check EML weight gradient")
    );

    gpu.zero_train_buffers_gpu()
        .expect("reset EML device buffers");
    gpu.device_eml_train_buffers.borrow_mut()[0]
        .as_mut()
        .expect("EML train buffers")
        .self_params
        .scale_updates
        .as_mut()
        .expect("EML BN scale updates")
        .data[0] = f32::NAN;
    assert!(
        gpu.gradients_non_finite()
            .expect("check EML BN scale gradient")
    );
}

#[test]
fn eml_convolutional_gpu_adam_updates_device_parameters() {
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=1\nwidth=1\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=0\nresidual=0\n";
    let network = Network::from_config(
        ConfigFile::parse_str("eml-adam.cfg", cfg).expect("EML Adam cfg parses"),
    )
    .expect("EML Adam network builds");
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let LayerParams::EmlConvolutional(params) = &mut gpu.params[0] else {
        panic!("expected EML params");
    };
    params.input.weights[0] = 0.4;
    params.input.biases[0] = 0.1;
    params.self_params.weights[0] = -0.3;
    params.self_params.biases[0] = 0.2;
    let input = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![0.25]).unwrap();
    let mut cpu = ExecutableNetwork::new(network);
    cpu.params = gpu.params.clone();
    let cpu_forward = cpu.forward_training(&input).expect("CPU EML Adam forward");
    let forward = gpu.forward_with_state(&input).expect("EML Adam forward");
    let seed = Tensor::filled(cpu_forward.final_output().unwrap().shape, 1.0);
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &[Some(seed.clone())])
        .expect("CPU EML Adam backward");
    gpu.backward_accumulate_seeded_gpu(&forward, &[Some(seed)])
        .expect("EML Adam backward");
    let LayerTrainBuffers::EmlConvolutional(cpu_train) = &cpu.train_buffers[0] else {
        unreachable!()
    };
    let device_train = gpu.device_eml_train_buffers.borrow();
    let gpu_train = device_train[0].as_ref().unwrap();
    assert_slice_close(
        "EML Adam input weight gradient",
        &cpu_train.input.weight_updates,
        &gpu_train.input.weight_updates.data,
        1e-5,
    );
    assert_slice_close(
        "EML Adam self weight gradient",
        &cpu_train.self_params.weight_updates,
        &gpu_train.self_params.weight_updates.data,
        1e-5,
    );
    drop(device_train);
    let adam = lightnet_cuda::host_ops::AdamParams {
        lr: 0.01,
        beta1: gpu.network.net.adam_b1,
        beta2: gpu.network.net.adam_b2,
        eps: gpu.network.net.adam_eps,
        decay: 0.0,
        batch: 1,
        step: 1,
        decoupled_decay: gpu.network.net.decoupled_decay,
    };
    let LayerParams::EmlConvolutional(before) = &gpu.params[0] else {
        unreachable!()
    };
    let expected = |weights: &[f32], updates: &[f32]| {
        lightnet_cuda::host_ops::adam_update(
            weights,
            updates,
            &vec![0.0; weights.len()],
            &vec![0.0; weights.len()],
            &adam,
        )
        .0
    };
    let expected_input_weights = expected(&before.input.weights, &cpu_train.input.weight_updates);
    let expected_input_biases = expected(&before.input.biases, &cpu_train.input.bias_updates);
    let expected_self_weights = expected(
        &before.self_params.weights,
        &cpu_train.self_params.weight_updates,
    );
    let expected_self_biases = expected(
        &before.self_params.biases,
        &cpu_train.self_params.bias_updates,
    );
    let before = gpu.params.clone();
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("EML device Adam update");
    let LayerParams::EmlConvolutional(before) = &before[0] else {
        unreachable!()
    };
    let LayerParams::EmlConvolutional(updated) = &gpu.params[0] else {
        unreachable!()
    };
    assert_ne!(before.input.weights, updated.input.weights);
    assert_ne!(before.input.biases, updated.input.biases);
    assert_ne!(before.self_params.weights, updated.self_params.weights);
    assert_ne!(before.self_params.biases, updated.self_params.biases);
    assert_slice_close(
        "EML Adam updated input weights",
        &expected_input_weights,
        &updated.input.weights,
        1e-5,
    );
    assert_slice_close(
        "EML Adam updated input biases",
        &expected_input_biases,
        &updated.input.biases,
        1e-5,
    );
    assert_slice_close(
        "EML Adam updated self weights",
        &expected_self_weights,
        &updated.self_params.weights,
        1e-5,
    );
    assert_slice_close(
        "EML Adam updated self biases",
        &expected_self_biases,
        &updated.self_params.biases,
        1e-5,
    );
    let moments = gpu.eml_adam_state[0]
        .as_ref()
        .expect("device EML Adam state");
    assert!(moments.input.weights.is_some());
    assert!(moments.input.biases.is_some());
    assert!(moments.self_params.weights.is_some());
    assert!(moments.self_params.biases.is_some());
    assert!(gpu.host_adam_state.borrow()[0].is_none());
}

#[test]
fn eml_adam_stays_host_side_when_network_requires_cpu_backward_fallback() {
    // `revleaky` has no device activation, so the ViT forces the CPU fallback.
    let cfg = "[net]\nchannels=1\nbatch=1\nheight=4\nwidth=4\nadam=1\nb1=0.8\nb2=0.95\neps=0.001\n\
                   [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=0\nresidual=0\n\
                   [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\nactivation=revleaky\n";
    let network = Network::from_config(
        ConfigFile::parse_str("eml-adam-fallback.cfg", cfg).expect("EML Adam fallback cfg parses"),
    )
    .expect("EML Adam fallback network builds");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.init_train_buffers_gpu()
        .expect("fallback train buffers");
    assert!(gpu.requires_cpu_backward_fallback());
    {
        let mut buffers = gpu.cpu_fallback_train_buffers.borrow_mut();
        let LayerTrainBuffers::EmlConvolutional(train) = &mut buffers[0] else {
            panic!("expected host EML train buffers");
        };
        train.input.weight_updates[0] = 1.0;
        train.self_params.bias_updates[0] = 1.0;
    }
    let before = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => {
            (params.input.weights[0], params.self_params.biases[0])
        }
        _ => unreachable!(),
    };
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
        .expect("host fallback Adam update");
    let after = match &gpu.params[0] {
        LayerParams::EmlConvolutional(params) => {
            (params.input.weights[0], params.self_params.biases[0])
        }
        _ => unreachable!(),
    };
    assert_ne!(before, after);
    assert!(gpu.host_adam_state.borrow()[0].is_some());
    assert!(gpu.eml_adam_state[0].is_none());
}

/// C++ `parse_moe_convolutional_section` raises two hard errors, and both
/// must stay errors here: clamping `experts` or `top_k` would silently
/// change the parameter count the `.weights` file is read with.
#[test]
fn moe_convolutional_rejects_the_configs_cxx_rejects() {
    let net = "[net]\nchannels=2\nbatch=1\nheight=2\nwidth=2\n";
    // `experts` must be >= 1.
    let cfg = format!("{net}[moe_convolutional]\nfilters=2\nsize=1\nexperts=0\n");
    let error = ConfigFile::parse_str("moe.cfg", &cfg)
        .err()
        .or_else(|| Network::from_config(ConfigFile::parse_str("moe.cfg", &cfg).unwrap()).err());
    assert!(
        matches!(error, Some(crate::Error::InvalidConfig(ref message)) if message.contains("experts must be >= 1")),
        "{error:?}"
    );

    // `top_k` cannot exceed `experts`.
    let cfg = format!("{net}[moe_convolutional]\nfilters=2\nsize=1\nexperts=2\ntop_k=3\n");
    let error = ConfigFile::parse_str("moe.cfg", &cfg)
        .err()
        .or_else(|| Network::from_config(ConfigFile::parse_str("moe.cfg", &cfg).unwrap()).err());
    assert!(
        matches!(error, Some(crate::Error::InvalidConfig(ref message)) if message.contains("top_k=3")),
        "{error:?}"
    );

    // `top_k = experts` is legal and means "no mask", as in C++.
    let cfg = format!("{net}[moe_convolutional]\nfilters=2\nsize=1\nexperts=2\ntop_k=2\n");
    assert!(
        Network::from_config(ConfigFile::parse_str("moe.cfg", &cfg).unwrap()).is_ok(),
        "top_k=experts must parse"
    );
}

/// A hand-computed `[moe_convolutional]` forward: one image, two 1x1
/// experts, so the blend is exactly the router-weighted sum of the two
/// convolutions and can be checked by hand rather than by re-running the
/// implementation.
#[test]
fn cpu_moe_convolutional_blend_matches_a_hand_computation() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "moe-blend.cfg",
            "[net]\nchannels=2\nbatch=1\nheight=2\nwidth=2\n\
             [moe_convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\n\
             activation=linear\nexperts=2\ntop_k=0\nload_balance=0\ntemperature=1\n",
        )
        .expect("moe cfg should parse"),
    )
    .expect("moe network should build");

    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Moe(params) = &mut executable.params[0] else {
        panic!("expected moe parameters");
    };
    // Router picks expert 0 strongly and expert 1 weakly: the pooled input
    // is 1 in channel 0, so the bias alone decides.
    params.router_biases = vec![2.0, -1.0];
    params.router_weights = vec![0.0, 0.0, 0.0, 0.0];
    // Expert 0 is the identity, expert 1 negates.
    params.experts[0].weights = vec![1.0, 0.0, 0.0, 1.0];
    params.experts[0].biases = vec![0.0, 0.0];
    params.experts[1].weights = vec![-1.0, 0.0, 0.0, -1.0];
    params.experts[1].biases = vec![0.0, 0.0];

    // Layout [b][c][h][w]: channel 0 carries the signal, channel 1 is zero.
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("moe input should have the expected shape");

    let forward = executable
        .forward(&input)
        .expect("moe forward should succeed");
    let output = forward
        .layer_outputs
        .get(0)
        .expect("moe output should exist");

    // softmax([2, -1]) = [e^3/(e^3+1), 1/(e^3+1)].
    let e3 = 3.0f32.exp();
    let gate0 = e3 / (e3 + 1.0);
    let gate1 = 1.0 / (e3 + 1.0);
    // Expert 0 is the identity and expert 1 is its negation, so the blend
    // is (gate0 - gate1) times the input, element by element.
    let scale = gate0 - gate1;
    let source = input.data.clone();
    for (index, value) in output.data.iter().enumerate() {
        let want = scale * source[index];
        assert!(
            (value - want).abs() < 1e-5,
            "element {index}: {value} vs {want} (gates {gate0}, {gate1})"
        );
    }

    // The cached gates must be the distribution the blend used.
    let cache = forward
        .moe_caches
        .get(0)
        .and_then(|slot| slot.as_ref())
        .expect("moe forward should cache its router");
    assert!((cache.gates[0] - gate0).abs() < 1e-6, "{:?}", cache.gates);
    assert!((cache.gates[0] + cache.gates[1] - 1.0).abs() < 1e-6);
    assert_eq!(cache.expert_outputs.len(), 2);
}

#[test]
fn route_axis_layout_pins_named_source_order_group_slice_and_delta_scatter() {
    use crate::axis_fixture::{axis_tensor, axis_value, nchw_index};

    let network = Network::from_config(
        ConfigFile::parse_str(
            "route-axis.cfg",
            "[net]\nchannels=4\nbatch=2\nheight=2\nwidth=3\n\
                 [maxpool]\nsize=1\nstride=1\n\
                 [convolutional]\nfilters=4\nsize=1\nstride=1\npad=0\
                 \nbatch_normalize=0\nactivation=linear\n\
                 [route]\nlayers=-1,-2\ngroups=2\ngroup_id=1\n",
        )
        .expect("route axis cfg parses"),
    )
    .expect("route axis network builds");
    let route_index = 2;
    let LayerKind::Route(spec) = &network.layers[route_index].kind else {
        panic!("expected route layer");
    };
    let source_shape = TensorShape::new(3, 2, 4, 2);
    let source0 =
        Tensor::from_data(source_shape, axis_tensor(source_shape)).expect("first axis source");
    let source1 = Tensor::from_data(
        source_shape,
        source0
            .data
            .iter()
            .map(|value| -2.0 * value + 0.25)
            .collect(),
    )
    .expect("second axis source");
    let sources = vec![source0.clone(), source1.clone()];
    let output_shape = network.layers[route_index].output_shape;
    let output =
        forward_route(spec, route_index, &sources, output_shape).expect("route source selection");

    // C++ route forward visits configured layers in order and, for each
    // source, copies its selected group into the next output channels.
    // Thus `layers=-1,-2`, `groups=2`, `group_id=1` yields source1 C=2,3
    // followed by source0 C=2,3 (not the reverse and not group zero).
    assert_eq!(output_shape, TensorShape::new(3, 2, 4, 2));
    for n in 0..2 {
        for h in 0..2 {
            for w in 0..3 {
                let expected = [
                    -2.0 * axis_value(n, 2, h, w) + 0.25,
                    -2.0 * axis_value(n, 3, h, w) + 0.25,
                    axis_value(n, 2, h, w),
                    axis_value(n, 3, h, w),
                ];
                for (c, want) in expected.into_iter().enumerate() {
                    assert_eq!(
                        output.data[nchw_index(output_shape, n, c, h, w)],
                        want,
                        "wrong route source/group at (n={n}, c={c}, h={h}, w={w})"
                    );
                }
            }
        }
    }

    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| index as f32 + 0.5)
            .collect(),
    )
    .expect("route seed");
    let mut source_grads = vec![Tensor::zeros(source_shape), Tensor::zeros(source_shape)];
    backward_route(spec, route_index, &seed, &mut source_grads, &network.layers)
        .expect("route gradient scatter");
    for n in 0..2 {
        for c in 0..4 {
            for h in 0..2 {
                for w in 0..3 {
                    let index = nchw_index(source_shape, n, c, h, w);
                    let expected_source1 = if c == 2 {
                        seed.get(n, 0, h, w)
                    } else if c == 3 {
                        seed.get(n, 1, h, w)
                    } else {
                        0.0
                    };
                    let expected_source0 = if c == 2 {
                        seed.get(n, 2, h, w)
                    } else if c == 3 {
                        seed.get(n, 3, h, w)
                    } else {
                        0.0
                    };
                    assert_eq!(source_grads[1].data[index], expected_source1);
                    assert_eq!(source_grads[0].data[index], expected_source0);
                }
            }
        }
    }
}

#[test]
fn upsample_axis_layout_pins_forward_repeat_reverse_sum_and_adjoint() {
    use crate::axis_fixture::{axis_tensor, axis_value, nchw_index};

    let forward_network = Network::from_config(
        ConfigFile::parse_str(
            "upsample-axis-forward.cfg",
            "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=3\n\
                 [upsample]\nstride=2\nscale=0.5\n",
        )
        .expect("forward upsample cfg parses"),
    )
    .expect("forward upsample network builds");
    let LayerKind::Upsample(forward_spec) = &forward_network.layers[0].kind else {
        panic!("expected forward upsample layer");
    };
    let low_shape = forward_network.net.input_shape();
    let high_shape = forward_network.layers[0].output_shape;
    let low_input = Tensor::from_data(low_shape, axis_tensor(low_shape)).expect("low input");
    let high_output =
        forward_upsample(forward_spec, &low_input, high_shape).expect("upsample forward");
    assert_eq!(high_shape, TensorShape::new(6, 4, 2, 2));
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..4 {
                for x in 0..6 {
                    let want = 0.5 * axis_value(n, c, y / 2, x / 2);
                    assert_eq!(
                        high_output.data[nchw_index(high_shape, n, c, y, x)],
                        want,
                        "upsample repeat/scale mismatch at (n={n}, c={c}, y={y}, x={x})"
                    );
                }
            }
        }
    }
    let high_seed = Tensor::from_data(high_shape, axis_tensor(high_shape)).expect("high seed");
    let low_gradient =
        backward_upsample(forward_spec, &high_seed, low_shape).expect("upsample backward");
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..2 {
                for x in 0..3 {
                    let mut want = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            want += high_seed.get(n, c, y * 2 + dy, x * 2 + dx);
                        }
                    }
                    want *= 0.5;
                    assert_eq!(
                        low_gradient.get(n, c, y, x),
                        want,
                        "upsample adjoint mismatch at (n={n}, c={c}, y={y}, x={x})"
                    );
                }
            }
        }
    }

    // Darknet's negative-stride reverse mode sums each complete stride
    // block and drops incomplete trailing rows/columns from integer division.
    let reverse_network = Network::from_config(
        ConfigFile::parse_str(
            "upsample-axis-reverse.cfg",
            "[net]\nchannels=2\nbatch=2\nheight=3\nwidth=5\n\
                 [upsample]\nstride=-2\nscale=0.25\n",
        )
        .expect("reverse upsample cfg parses"),
    )
    .expect("reverse upsample network builds");
    let LayerKind::Upsample(reverse_spec) = &reverse_network.layers[0].kind else {
        panic!("expected reverse upsample layer");
    };
    let high_shape = reverse_network.net.input_shape();
    let low_shape = reverse_network.layers[0].output_shape;
    let high_input = Tensor::from_data(high_shape, axis_tensor(high_shape)).expect("high input");
    let low_output =
        forward_upsample(reverse_spec, &high_input, low_shape).expect("reverse forward");
    assert_eq!(low_shape, TensorShape::new(2, 1, 2, 2));
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..1 {
                for x in 0..2 {
                    let mut want = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            want += high_input.get(n, c, y * 2 + dy, x * 2 + dx);
                        }
                    }
                    want *= 0.25;
                    assert_eq!(low_output.get(n, c, y, x), want);
                }
            }
        }
    }
    let low_seed = Tensor::from_data(low_shape, axis_tensor(low_shape)).expect("low seed");
    let high_gradient =
        backward_upsample(reverse_spec, &low_seed, high_shape).expect("reverse adjoint");
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..3 {
                for x in 0..5 {
                    let want = if y < 2 && x < 4 {
                        0.25 * low_seed.get(n, c, y / 2, x / 2)
                    } else {
                        0.0
                    };
                    assert_eq!(
                        high_gradient.get(n, c, y, x),
                        want,
                        "reverse adjoint extent mismatch at (n={n}, c={c}, y={y}, x={x})"
                    );
                }
            }
        }
    }
}

#[test]
fn reorg_axis_layout_pins_batch_channel_offset_and_adjoint() {
    use crate::axis_fixture::{axis_tensor, axis_value, nchw_index};

    let regular_network = Network::from_config(
        ConfigFile::parse_str(
            "reorg-axis.cfg",
            "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=4\n\
                 [reorg]\nstride=2\n",
        )
        .expect("regular Reorg cfg parses"),
    )
    .expect("regular Reorg network builds");
    let LayerKind::Reorg(regular_spec) = &regular_network.layers[0].kind else {
        panic!("expected regular Reorg layer");
    };
    let input_shape = regular_network.net.input_shape();
    let packed_shape = regular_network.layers[0].output_shape;
    assert_eq!(packed_shape, TensorShape::new(2, 1, 8, 2));
    let input = Tensor::from_data(input_shape, axis_tensor(input_shape)).expect("Reorg input");
    let packed = forward_reorg(2, false, &input, packed_shape).expect("regular Reorg forward");

    // C++ reorg_cpu packs each 2x2 spatial offset into a channel block:
    // k = input_channel + offset_channel * C, offset=(dy,dx), with dx fastest.
    for n in 0..2 {
        for c in 0..8 {
            let in_channel = c % 2;
            let offset = c / 2;
            let dy = offset / 2;
            let dx = offset % 2;
            for y in 0..1 {
                for x in 0..2 {
                    let want = axis_value(n, in_channel, y * 2 + dy, x * 2 + dx);
                    assert_eq!(
                        packed.data[nchw_index(packed_shape, n, c, y, x)],
                        want,
                        "wrong Reorg offset/channel at (n={n}, c={c}, y={y}, x={x})"
                    );
                }
            }
        }
    }
    let packed_seed =
        Tensor::from_data(packed_shape, axis_tensor(packed_shape)).expect("packed seed");
    let input_gradient =
        backward_reorg(regular_spec, &packed_seed, input_shape).expect("regular Reorg adjoint");
    for n in 0..2 {
        for c in 0..2 {
            for y in 0..2 {
                for x in 0..4 {
                    let packed_channel = c + ((y % 2) * 2 + (x % 2)) * 2;
                    let want = packed_seed.get(n, packed_channel, y / 2, x / 2);
                    assert_eq!(input_gradient.get(n, c, y, x), want);
                }
            }
        }
    }

    // The C++ reverse path has a source-level divisor defect: reorg_cpu
    // computes in_c = out_c / stride^2 although reverse's out_c is already
    // the reduced channel count. Keep this pin scoped to the valid forward
    // branch until a compiled-C++ oracle resolves that reverse contract.
}

#[test]
fn sam_axis_layout_pins_elementwise_nchw_source_and_adjoint() {
    use crate::axis_fixture::{axis_tensor, nchw_index};
    use crate::model::{Activation, SamSpec};

    let shape = TensorShape::new(3, 2, 2, 2);
    let input = Tensor::from_data(shape, axis_tensor(shape)).expect("SAM input");
    let source = Tensor::from_data(
        shape,
        axis_tensor(shape)
            .into_iter()
            .map(|value| value + 0.5)
            .collect(),
    )
    .expect("named SAM source");
    let seed = Tensor::from_data(
        shape,
        axis_tensor(shape)
            .into_iter()
            .map(|value| value + 0.25)
            .collect(),
    )
    .expect("SAM adjoint seed");
    let spec = SamSpec {
        from: 0,
        activation: Activation::Linear,
    };
    let (output, activation_input) =
        super::forward_sam(&spec, &input, &source, shape).expect("SAM forward");
    let (input_gradient, source_gradient) = super::backward_sam(
        &spec,
        &input,
        &source,
        &output,
        &seed,
        activation_input.as_ref(),
    )
    .expect("SAM backward");

    for n in 0..shape.batch {
        for c in 0..shape.channels {
            for h in 0..shape.height {
                for w in 0..shape.width {
                    let index = nchw_index(shape, n, c, h, w);
                    assert_eq!(output.data[index], input.data[index] * source.data[index]);
                    assert_eq!(
                        input_gradient.data[index],
                        seed.data[index] * source.data[index]
                    );
                    assert_eq!(
                        source_gradient.data[index],
                        seed.data[index] * input.data[index]
                    );
                }
            }
        }
    }
}

#[test]
fn scale_channels_axis_layout_pins_named_source_and_channel_or_spatial_broadcast() {
    use crate::axis_fixture::{axis_tensor, nchw_index};
    use crate::model::{Activation, ScaleChannelsSpec};

    let source_shape = TensorShape::new(3, 2, 2, 2);
    let source =
        Tensor::from_data(source_shape, axis_tensor(source_shape)).expect("named source output");
    for scale_wh in [false, true] {
        let input_shape = if scale_wh {
            TensorShape::new(3, 2, 1, 2)
        } else {
            TensorShape::new(1, 1, 2, 2)
        };
        let input = Tensor::from_data(input_shape, axis_tensor(input_shape)).expect("scale input");
        let spec = ScaleChannelsSpec {
            from: 0,
            scale_wh,
            activation: Activation::Linear,
        };
        let (output, activation_input) =
            super::forward_scale_channels(&spec, &input, &source, source_shape)
                .expect("scale_channels forward");
        let seed = Tensor::from_data(source_shape, axis_tensor(source_shape))
            .expect("scale_channels seed");
        let (input_gradient, source_gradient) = super::backward_scale_channels(
            &spec,
            &input,
            &source,
            &output,
            &seed,
            activation_input.as_ref(),
        )
        .expect("scale_channels backward");
        let mut expected_input_gradient =
            vec![0.0; input_shape.element_count() * input_shape.batch];
        for n in 0..source_shape.batch {
            for c in 0..source_shape.channels {
                for h in 0..source_shape.height {
                    for w in 0..source_shape.width {
                        let output_index = nchw_index(source_shape, n, c, h, w);
                        let input_index = if scale_wh {
                            nchw_index(input_shape, n, 0, h, w)
                        } else {
                            nchw_index(input_shape, n, c, 0, 0)
                        };
                        let scale = input.data[input_index];
                        assert_eq!(
                            output.data[output_index],
                            scale * source.data[output_index],
                            "from=0 output slot mismatch at (n={n},c={c},h={h},w={w}), scale_wh={scale_wh}"
                        );
                        assert_eq!(
                            source_gradient.data[output_index],
                            seed.data[output_index] * scale,
                            "named-source adjoint mismatch at (n={n},c={c},h={h},w={w}), scale_wh={scale_wh}"
                        );
                        expected_input_gradient[input_index] +=
                            seed.data[output_index] * source.data[output_index];
                    }
                }
            }
        }
        assert_eq!(input_gradient.data, expected_input_gradient);
    }
}

#[test]
fn dcnv4_native_gate_accepts_large_kernels() {
    for (size, expected_host) in [(31, false), (33, false)] {
        let padding = size / 2;
        let cfg = format!(
            "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=1\n\
                 [dcnv4]\nfilters=2\nsize={size}\npadding={padding}\n\
                 groups=1\nd_stride=1\nactivation=linear\n"
        );
        let network = Network::from_config(
            ConfigFile::parse_str("inline.cfg", &cfg).expect("DCNv4 gate cfg"),
        )
        .expect("DCNv4 gate network");
        let LayerKind::Dcnv4(spec) = &network.layers[0].kind else {
            panic!("expected DCNv4 layer");
        };
        assert_eq!(
            super::dcnv4_gpu::needs_host(spec),
            expected_host,
            "unexpected native gate for size={size}"
        );
    }
}

#[test]
fn softmax_axis_layout_pins_per_batch_contiguous_groups_and_adjoint() {
    use crate::axis_fixture::{axis_tensor, decode_axis_value};

    let shape = TensorShape::new(3, 2, 4, 2);
    let input = Tensor::from_data(shape, axis_tensor(shape)).expect("softmax axis input");
    let groups = 3;
    let per_batch = shape.element_count();
    let group_size = per_batch / groups;
    let output = forward_softmax(&input, groups, false).expect("grouped softmax forward");

    // C++ passes n=inputs/groups, groups, group_offset=n, and batch_offset=inputs:
    // every sample is split into contiguous flat groups, not per-channel groups.
    for batch in 0..shape.batch {
        for group in 0..groups {
            let start = batch * per_batch + group * group_size;
            let end = start + group_size;
            let max_value = input.data[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let exponentials = input.data[start..end]
                .iter()
                .map(|value| (value - max_value).exp())
                .collect::<Vec<_>>();
            let denominator = exponentials.iter().sum::<f32>();
            for (offset, exponential) in exponentials.into_iter().enumerate() {
                let index = start + offset;
                let (n, c, h, w) = decode_axis_value(input.data[index]);
                assert_eq!(n, batch);
                assert_eq!(
                    output.data[index],
                    exponential / denominator,
                    "wrong softmax group axis at (n={n}, c={c}, h={h}, w={w})"
                );
            }
            let sum = output.data[start..end].iter().sum::<f32>();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "batch={batch} group={group} sum={sum}"
            );
        }
    }

    let seed = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|index| 0.25 + index as f32 * 0.1)
            .collect(),
    )
    .expect("softmax axis seed");
    let input_gradient = backward_softmax(&output, &seed, groups, false).expect("softmax adjoint");
    for batch in 0..shape.batch {
        for group in 0..groups {
            let start = batch * per_batch + group * group_size;
            let end = start + group_size;
            let dot = output.data[start..end]
                .iter()
                .zip(&seed.data[start..end])
                .map(|(probability, gradient)| probability * gradient)
                .sum::<f32>();
            for index in start..end {
                let want = output.data[index] * (seed.data[index] - dot);
                assert_eq!(
                    input_gradient.data[index], want,
                    "softmax adjoint escaped batch/group at flat index {index}"
                );
            }
        }
    }
}

#[test]
fn spatial_softmax_normalizes_channels_per_pixel_and_has_correct_adjoint() {
    let cfg = ConfigFile::parse_str("spatial.cfg", "[net]\nwidth=2\nheight=1\nchannels=2\nbatch=1\n[softmax]\nspatial=1\n").unwrap();
    let network = Network::from_config(cfg).unwrap();
    assert!(matches!(network.layers[0].kind, LayerKind::Softmax { spatial: true, .. }));
    let shape = TensorShape::new(2, 1, 2, 1);
    let input = Tensor::from_data(shape, vec![0.0, 2.0, 0.0, 0.0]).unwrap();
    let output = forward_softmax(&input, 1, true).unwrap();
    assert!((output.data[0] - 0.5).abs() < 1e-6);
    assert!((output.data[2] - 0.5).abs() < 1e-6);
    assert!((output.data[1] - 0.8807971).abs() < 1e-6);
    assert!((output.data[3] - 0.1192029).abs() < 1e-6);
    let seed = Tensor::from_data(shape, vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    let grad = backward_softmax(&output, &seed, 1, true).unwrap();
    assert!((grad.data[0] - 0.25).abs() < 1e-6);
    assert!((grad.data[2] + 0.25).abs() < 1e-6);
    assert!((grad.data[1] - 0.1049936).abs() < 1e-6);
    assert!((grad.data[3] + 0.1049936).abs() < 1e-6);
}

#[test]
fn cpu_moe_convolutional_axis_layout_keeps_router_and_blend_per_batch() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "moe-axis.cfg",
            "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=3\n\
                 [moe_convolutional]\nfilters=2\nsize=1\nstride=1\
                 \nbatch_normalize=0\nactivation=linear\nexperts=2\
                 \ntop_k=0\nload_balance=0\ntemperature=1\n",
        )
        .expect("MoE cfg parses"),
    )
    .expect("MoE network builds");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Moe(params) = &mut executable.params[0] else {
        panic!("expected MoE parameters");
    };
    params.router_biases = vec![0.0, 0.0];
    params.router_weights = vec![1.0, 0.0, 0.0, 1.0];
    params.experts[0].weights = vec![1.0, 0.0, 0.0, 1.0];
    params.experts[0].biases = vec![0.0, 0.0];
    params.experts[1].weights = vec![-1.0, 0.0, 0.0, -1.0];
    params.experts[1].biases = vec![0.0, 0.0];

    // The two samples have distinct global channel means on a non-square grid.
    let input_values = vec![
        1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 6.0, 3.0, 3.0, 6.0, 0.0,
    ];
    let shape = executable.network.net.input_shape();
    let input = Tensor::from_data(shape, input_values.clone()).expect("MoE input shape");
    let forward = executable.forward(&input).expect("MoE forward");
    let output = forward.layer_outputs.get(0).expect("MoE output exists");
    let cache = forward.moe_caches[0]
        .as_ref()
        .expect("router gates are cached per batch row");

    let e = 1.0f32.exp();
    let e3 = 3.0f32.exp();
    let expected_gates = [
        e / (e + 1.0),
        1.0 / (e + 1.0),
        1.0 / (e3 + 1.0),
        e3 / (e3 + 1.0),
    ];
    for (i, (got, want)) in cache.gates.iter().zip(expected_gates).enumerate() {
        assert!((got - want).abs() < 1e-6, "gate[{i}]={got}, want {want}");
    }
    for b in 0..2 {
        let scale = cache.gates[b * 2] - cache.gates[b * 2 + 1];
        for i in 0..12 {
            let index = b * 12 + i;
            let want = scale * input_values[index];
            assert!(
                (output.data[index] - want).abs() < 1e-5,
                "output[{index}]={} vs {want}",
                output.data[index]
            );
        }
    }
}

/// Every parameter of `[moe_convolutional]` - router weights, router biases
/// and both experts - must have an analytic gradient equal to a central
/// finite difference of the blended output.  The experts reuse the ordinary
/// convolution backward, so this also pins the gate-scaling and the router
/// chain that are new here.
#[test]
fn cpu_moe_convolutional_gradients_match_finite_differences() {
    let cfg = "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=2\n\
                   [moe_convolutional]\nfilters=2\nsize=3\nstride=1\npad=1\nbatch_normalize=0\n\
                   activation=leaky\nexperts=3\ntop_k=0\nload_balance=0.05\ntemperature=1.7\n";
    let build = || {
        let mut executable = ExecutableNetwork::new(
            Network::from_config(
                ConfigFile::parse_str("moe-grad.cfg", cfg).expect("moe cfg should parse"),
            )
            .expect("moe network should build"),
        );
        let LayerParams::Moe(params) = &mut executable.params[0] else {
            panic!("expected moe parameters");
        };
        // Distinct values everywhere so a transposed index cannot pass.
        let mut next = 0.1f32;
        let mut fill = |len: usize| {
            let values: Vec<f32> = (0..len)
                .map(|_| {
                    next += 0.037;
                    next * 0.5
                })
                .collect();
            values
        };
        params.router_weights = fill(params.router_weights.len());
        params.router_biases = fill(params.router_biases.len());
        for expert in &mut params.experts {
            expert.weights = fill(expert.weights.len());
            expert.biases = fill(expert.biases.len());
        }
        executable
    };

    let executable = build();
    // A non-uniform input: an all-equal input would make the global average
    // pool lose the spatial structure and hide most gradient mistakes.
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![
            0.3, -0.7, 1.1, 0.9, -0.2, 0.5, 0.8, -1.3, 0.6, 0.4, -0.9, 1.2, 0.7, -0.1, 0.2, 0.9,
        ],
    )
    .expect("moe input should have the expected shape");

    let loss_of = |executable: &ExecutableNetwork| -> f32 {
        let forward = executable
            .forward(&input)
            .expect("moe forward should succeed");
        let output = forward.layer_outputs[0].data.clone();
        // A fixed, non-uniform weighting so the gradient is not degenerate.
        output
            .iter()
            .enumerate()
            .map(|(index, value)| value * (0.1 + 0.03 * (index % 5) as f32))
            .sum::<f32>()
    };

    let forward = executable
        .forward(&input)
        .expect("moe forward should succeed");
    let seed = Tensor::from_data(
        forward.layer_outputs[0].shape,
        (0..forward.layer_outputs[0].data.len())
            .map(|index| 0.1 + 0.03 * (index % 5) as f32)
            .collect(),
    )
    .expect("seed should have the output shape");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("moe backward should succeed");
    let LayerParams::Moe(gradients) = &backward.param_gradients[0] else {
        panic!("expected moe parameter gradients");
    };

    // f32 central differences are only good to about |f| * 1e-7 / eps, so
    // eps is chosen large enough to keep that noise under 1e-4 and the
    // tolerance mixes a relative and an absolute term.
    let eps = 1e-2f32;
    let check = |name: String, analytic: &[f32], numeric: &[f32]| {
        assert_eq!(
            analytic.len(),
            numeric.len(),
            "{name}: gradient length mismatch"
        );
        for (index, (analytic, value)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert!(
                (analytic - value).abs() <= 5e-2 * value.abs().max(2e-4),
                "{name}[{index}]: analytic {analytic} numeric {value}"
            );
        }
    };

    // Router biases.
    let mut numeric_router_biases = Vec::new();
    for index in 0..2 + 1 {
        let mut hi = build();
        let LayerParams::Moe(params) = &mut hi.params[0] else {
            unreachable!()
        };
        params.router_biases[index] += eps;
        let mut lo = build();
        let LayerParams::Moe(params) = &mut lo.params[0] else {
            unreachable!()
        };
        params.router_biases[index] -= eps;
        numeric_router_biases.push((loss_of(&hi) - loss_of(&lo)) / (2.0 * eps));
    }
    check(
        "router_bias".to_owned(),
        &gradients.router_biases,
        &numeric_router_biases,
    );

    // Router weights (6 = 3 experts x 2 channels).
    let router_len = gradients.router_weights.len();
    let mut numeric_router_weights = Vec::new();
    for index in 0..router_len {
        let mut hi = build();
        let LayerParams::Moe(params) = &mut hi.params[0] else {
            unreachable!()
        };
        params.router_weights[index] += eps;
        let mut lo = build();
        let LayerParams::Moe(params) = &mut lo.params[0] else {
            unreachable!()
        };
        params.router_weights[index] -= eps;
        numeric_router_weights.push((loss_of(&hi) - loss_of(&lo)) / (2.0 * eps));
    }
    check(
        "router_weight".to_owned(),
        &gradients.router_weights,
        &numeric_router_weights,
    );

    // Every expert's weights and biases.
    for e in 0..gradients.experts.len() {
        for field in ["weights", "biases"] {
            let (analytic, len) = if field == "weights" {
                (
                    gradients.experts[e].weights.clone(),
                    gradients.experts[e].weights.len(),
                )
            } else {
                (
                    gradients.experts[e].biases.clone(),
                    gradients.experts[e].biases.len(),
                )
            };
            let mut numeric = Vec::new();
            for index in 0..len {
                let mut hi = build();
                let LayerParams::Moe(params) = &mut hi.params[0] else {
                    unreachable!()
                };
                if field == "weights" {
                    params.experts[e].weights[index] += eps;
                } else {
                    params.experts[e].biases[index] += eps;
                }
                let mut lo = build();
                let LayerParams::Moe(params) = &mut lo.params[0] else {
                    unreachable!()
                };
                if field == "weights" {
                    params.experts[e].weights[index] -= eps;
                } else {
                    params.experts[e].biases[index] -= eps;
                }
                numeric.push((loss_of(&hi) - loss_of(&lo)) / (2.0 * eps));
            }
            check(format!("expert{e}.{field}"), &analytic, &numeric);
        }
    }
}

#[test]
fn cpu_forward_and_backward_pass_match_connected_reference() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "cpu-forward-backward.cfg",
            "[net]\ninputs=2\nbatch=1\n[connected]\noutput=1\nactivation=linear\n",
        )
        .expect("connected cfg should parse"),
    )
    .expect("connected network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Connected(params) = &mut executable.params[0] else {
        panic!("expected connected parameters");
    };
    params.weights = vec![2.0, -1.0];
    params.biases = vec![0.25];

    let input = Tensor::from_data(executable.network.net.input_shape(), vec![3.0, 4.0])
        .expect("connected input should have the expected shape");

    let inference = executable
        .forward(&input)
        .expect("CPU inference forward should succeed");
    assert_eq!(
        inference.final_output().expect("forward output").data,
        vec![2.25]
    );

    let training = executable
        .forward_training(&input)
        .expect("CPU training forward should succeed");
    let seed = Tensor::filled(training.final_output().expect("training output").shape, 1.0);
    let backward = executable
        .backward(&input, &training, &seed)
        .expect("CPU backward should succeed");

    assert_eq!(backward.input_gradient.data, vec![2.0, -1.0]);
    match &backward.param_gradients[0] {
        LayerParams::Connected(gradient) => {
            assert_eq!(gradient.weights, vec![3.0, 4.0]);
            assert_eq!(gradient.biases, vec![1.0]);
        }
        other => panic!("unexpected connected gradients: {other:?}"),
    }

    executable.zero_train_buffers();
    executable
        .backward_accumulate(&input, &training, &seed)
        .expect("CPU training backward should accumulate");
    let LayerTrainBuffers::Connected(train) = &executable.train_buffers[0] else {
        panic!("expected connected training buffers");
    };
    assert_eq!(train.weight_updates, vec![3.0, 4.0]);
    assert_eq!(train.bias_updates, vec![1.0]);
}

#[test]
fn cpu_crop_center_forward_and_backward_scatter() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "cpu-crop.cfg",
                "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n[crop]\ncrop_width=2\ncrop_height=2\nflip=0\nangle=0\nnoadjust=1\n",
            )
            .expect("crop cfg should parse"),
        )
        .expect("crop network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        (1..=16).map(|value| value as f32).collect(),
    )
    .expect("crop input should have the expected shape");

    let forward = executable
        .forward(&input)
        .expect("CPU crop forward should succeed");
    assert_eq!(
        forward.final_output().expect("crop output").data,
        vec![6.0, 7.0, 10.0, 11.0]
    );

    let seed = Tensor::from_data(
        forward.final_output().expect("crop output").shape,
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("crop gradient should have the expected shape");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("CPU crop backward should succeed");
    assert_eq!(
        backward.input_gradient.data,
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]
    );
}

#[test]
fn container_convolution_cache_is_keyed_by_owner_and_expert_slot() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "gpu-moe-cache.cfg",
                "[net]\nchannels=2\nbatch=1\nheight=2\nwidth=2\n[moe_convolutional]\nfilters=2\nsize=1\nexperts=3\ntop_k=1\nactivation=linear\n",
            )
            .expect("moe cfg"),
        )
        .expect("moe network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::Moe(moe) = &mut gpu.params[0] else {
        panic!("expected moe parameters");
    };
    for (slot, expert) in moe.experts.iter_mut().enumerate() {
        expert.weights.fill(slot as f32 + 1.0);
    }

    gpu.warm_device_conv_params().expect("warm experts");
    let cache = gpu.device_conv_params.borrow();
    assert_eq!(cache[0].len(), 3);
    for (slot, params) in cache[0].iter().enumerate() {
        let params = params.as_ref().expect("cached expert slot");
        assert_eq!(params.raw_weights.shape, vec![2, 2, 1, 1]);
        assert!(
            params
                .raw_weights
                .data
                .iter()
                .all(|w| *w == slot as f32 + 1.0)
        );
    }
    drop(cache);

    gpu.update_cpu_fallback_weights(0.1, 0.0, 0.0, 1, &[], &[], &[]).unwrap();
    assert!(gpu.device_conv_params.borrow()[0].is_empty());
    gpu.warm_device_conv_params()
        .expect("rewarm updated experts");
    assert_eq!(gpu.device_conv_params.borrow()[0].len(), 3);
}

#[test]
fn gpu_moe_batchnorm_expert_device_backward_matches_cpu() {
    let cfg = "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=2\n[moe_convolutional]\nfilters=2\nsize=1\nexperts=3\ntop_k=0\nload_balance=0\ntemperature=1.4\nactivation=leaky\n";
    let network = Network::from_config(ConfigFile::parse_str("gpu-moe.cfg", cfg).unwrap())
        .expect("moe network");
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let LayerParams::Moe(moe) = &mut gpu.params[0] else {
        panic!("expected moe parameters");
    };
    for (i, weight) in moe.router_weights.iter_mut().enumerate() {
        *weight = (i as f32 - 2.0) * 0.07;
    }
    moe.router_biases.copy_from_slice(&[0.2, -0.1, 0.05]);
    for (slot, expert) in moe.experts.iter_mut().enumerate() {
        for (i, weight) in expert.weights.iter_mut().enumerate() {
            *weight = 0.08 * (1 + slot * 4 + i) as f32;
        }
        expert.biases.copy_from_slice(&[slot as f32 * 0.03, -0.02]);
    }
    let mut cpu = ExecutableNetwork::new(network);
    cpu.params = gpu.params.clone();
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..16).map(|i| (i as f32 * 0.37).sin()).collect(),
    )
    .expect("moe input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU MoE forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("GPU MoE forward");
    assert!(!gpu.requires_cpu_backward_fallback());
    assert_slice_close(
        "MoE forward",
        &cpu_forward.layer_outputs[0].data,
        &gpu_forward.cpu_forward.layer_outputs[0].data,
        2e-5,
    );
    assert!(gpu_forward.cpu_forward.moe_caches[0].is_some());

    let seed = Tensor::from_data(
        cpu_forward.layer_outputs[0].shape,
        (0..cpu_forward.layer_outputs[0].data.len())
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    )
    .expect("MoE seed");
    let seeds = vec![Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero GPU buffers");
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU MoE backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU MoE batch-normalized backward");
    {
        let LayerTrainBuffers::Moe(cpu_train) = &cpu.train_buffers[0] else {
            panic!("CPU MoE train buffers");
        };
        let device_buffers = gpu.device_moe_train_buffers.borrow();
        let gpu_train = device_buffers[0].as_ref().expect("GPU MoE train buffers");
        assert_slice_close(
            "router weight grads",
            &cpu_train.router_weight_updates,
            &gpu_train.router_weight_updates.data,
            2e-5,
        );
        assert_slice_close(
            "router bias grads",
            &cpu_train.router_bias_updates,
            &gpu_train.router_bias_updates.data,
            2e-5,
        );
        for (cpu_expert, ((gpu_weights, gpu_biases), gpu_scales)) in cpu_train.experts.iter().zip(
            gpu_train
                .expert_weight_updates
                .iter()
                .zip(&gpu_train.expert_bias_updates)
                .zip(&gpu_train.expert_scale_updates),
        ) {
            assert_slice_close(
                "expert weight grads",
                &cpu_expert.weight_updates,
                &gpu_weights.data,
                2e-5,
            );
            assert_slice_close(
                "expert BN bias grads",
                &cpu_expert.bias_updates,
                &gpu_biases.data,
                2e-5,
            );
            let cpu_scales = &cpu_expert
                .batch_norm
                .as_ref()
                .expect("CPU BN buffers")
                .scale_updates;
            assert_slice_close(
                "expert BN scale grads",
                cpu_scales,
                &gpu_scales.as_ref().expect("GPU BN scale buffer").data,
                2e-5,
            );
        }
    }
    cpu.update_weights(0.01, 0.9, 0.001, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.001, 2)
        .expect("GPU MoE BN host optimizer");
    let LayerParams::Moe(cpu_params) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Moe(gpu_params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "router weights",
        &cpu_params.router_weights,
        &gpu_params.router_weights,
        2e-5,
    );
    assert_slice_close(
        "router biases",
        &cpu_params.router_biases,
        &gpu_params.router_biases,
        2e-5,
    );
    for (cpu_expert, gpu_expert) in cpu_params.experts.iter().zip(&gpu_params.experts) {
        assert_slice_close(
            "expert weights",
            &cpu_expert.weights,
            &gpu_expert.weights,
            2e-5,
        );
        assert_slice_close(
            "expert biases",
            &cpu_expert.biases,
            &gpu_expert.biases,
            2e-5,
        );
        let cpu_bn = cpu_expert.batch_norm.as_ref().expect("CPU BN parameters");
        let gpu_bn = gpu_expert.batch_norm.as_ref().expect("GPU BN parameters");
        assert_slice_close("expert BN scales", &cpu_bn.scales, &gpu_bn.scales, 2e-5);
    }
}

#[test]
fn gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check() {
    let cfg = "[net]\nchannels=2\nbatch=1\nheight=1\nwidth=1\n[moe_convolutional]\nfilters=2\nsize=1\nexperts=2\ntop_k=0\n";
    let network = Network::from_config(ConfigFile::parse_str("gpu-moe-overflow.cfg", cfg).unwrap())
        .expect("MoE network");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.zero_train_buffers_gpu()
        .expect("initialize GPU MoE buffers");
    assert!(!gpu.gradients_non_finite().expect("finite zero buffers"));
    {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("GPU MoE train buffers");
        train.dirty = true;
        train.router_weight_updates.data[0] = f32::NAN;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("check non-finite MoE router gradient")
    );
    {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("GPU MoE train buffers");
        train.router_weight_updates.data[0] = 0.0;
        train.expert_bias_updates[0].data[0] = f32::INFINITY;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("check non-finite MoE expert gradient")
    );
    {
        let mut buffers = gpu.device_moe_train_buffers.borrow_mut();
        let train = buffers[0].as_mut().expect("GPU MoE train buffers");
        train.expert_bias_updates[0].data[0] = 0.0;
        train.expert_scale_updates[0]
            .as_mut()
            .expect("GPU MoE BN scale buffer")
            .data[0] = f32::NAN;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("check non-finite MoE BN scale gradient")
    );
}

#[test]
fn gpu_moe_expert_device_backward_matches_cpu() {
    let cfg = "[net]\nchannels=2\nbatch=2\nheight=2\nwidth=2\n[moe_convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nexperts=3\ntop_k=0\nload_balance=0.05\ntemperature=1.4\nactivation=leaky\n";
    let network = Network::from_config(ConfigFile::parse_str("gpu-moe.cfg", cfg).unwrap())
        .expect("moe network");
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let LayerParams::Moe(moe) = &mut gpu.params[0] else {
        panic!("expected moe parameters");
    };
    for (i, weight) in moe.router_weights.iter_mut().enumerate() {
        *weight = (i as f32 - 2.0) * 0.07;
    }
    moe.router_biases.copy_from_slice(&[0.2, -0.1, 0.05]);
    for (slot, expert) in moe.experts.iter_mut().enumerate() {
        for (i, weight) in expert.weights.iter_mut().enumerate() {
            *weight = 0.08 * (1 + slot * 4 + i) as f32;
        }
        expert.biases.copy_from_slice(&[slot as f32 * 0.03, -0.02]);
    }
    let mut cpu = ExecutableNetwork::new(network);
    cpu.params = gpu.params.clone();
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..16).map(|i| (i as f32 * 0.37).sin()).collect(),
    )
    .expect("moe input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU MoE forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("GPU MoE forward");
    assert!(!gpu.requires_cpu_backward_fallback());
    let gpu_output = gpu
        .backend
        .download_f32(&gpu_forward.gpu_layer_outputs[0])
        .expect("download GPU MoE output");
    assert_slice_close(
        "MoE forward",
        &cpu_forward.layer_outputs[0].data,
        &gpu_output,
        2e-5,
    );
    assert!(gpu_forward.cpu_forward.moe_caches[0].is_none());
    assert!(gpu_forward.gpu_moe_caches[0].is_some());
    assert!(
        gpu.device_conv_params.borrow()[0]
            .iter()
            .take(3)
            .all(Option::is_some)
    );

    let seed = Tensor::from_data(
        cpu_forward.layer_outputs[0].shape,
        (0..cpu_forward.layer_outputs[0].data.len())
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    )
    .expect("MoE seed");
    let seeds = vec![Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().expect("zero GPU buffers");
    for _ in 0..2 {
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
            .expect("CPU MoE backward");
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
            .expect("GPU MoE backward");
    }
    {
        let LayerTrainBuffers::Moe(cpu_train) = &cpu.train_buffers[0] else {
            panic!("CPU MoE train buffers");
        };
        let device_buffers = gpu.device_moe_train_buffers.borrow();
        let gpu_train = device_buffers[0]
            .as_ref()
            .expect("GPU MoE device train buffers");
        assert!(gpu_train.dirty);
        assert_slice_close(
            "router weight grads",
            &cpu_train.router_weight_updates,
            &gpu_train.router_weight_updates.data,
            2e-5,
        );
        assert_slice_close(
            "router bias grads",
            &cpu_train.router_bias_updates,
            &gpu_train.router_bias_updates.data,
            2e-5,
        );
        for ((cpu_expert, gpu_weights), gpu_biases) in cpu_train
            .experts
            .iter()
            .zip(&gpu_train.expert_weight_updates)
            .zip(&gpu_train.expert_bias_updates)
        {
            assert_slice_close(
                "expert weight grads",
                &cpu_expert.weight_updates,
                &gpu_weights.data,
                2e-5,
            );
            assert_slice_close(
                "expert bias grads",
                &cpu_expert.bias_updates,
                &gpu_biases.data,
                2e-5,
            );
        }
    }
    {
        let fallback = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::Moe(host_train) = &fallback[0] else {
            panic!("GPU MoE host train buffers");
        };
        assert!(
            host_train
                .router_weight_updates
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            host_train
                .router_bias_updates
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(host_train.experts.iter().all(|expert| {
            expert.weight_updates.iter().all(|value| *value == 0.0)
                && expert.bias_updates.iter().all(|value| *value == 0.0)
        }));
    }
    let LayerParams::Moe(cpu_params) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Moe(gpu_params) = &gpu.params[0] else {
        unreachable!()
    };
    let cpu_router_before = cpu_params.router_weights.clone();
    let gpu_router_before = gpu_params.router_weights.clone();
    cpu.network.layers[0].train.dont_update = 2;
    gpu.network.layers[0].train.dont_update = 2;
    cpu.update_weights(0.01, 0.9, 0.001, 4);
    gpu.update_weights_gpu(0.01, 0.9, 0.001, 4)
        .expect("skip MoE SGD before dont_update expires");
    let LayerParams::Moe(cpu_params) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Moe(gpu_params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_eq!(cpu_params.router_weights, cpu_router_before);
    assert_eq!(gpu_params.router_weights, gpu_router_before);
    assert!(
        gpu.device_moe_train_buffers.borrow()[0]
            .as_ref()
            .expect("GPU MoE device train buffers")
            .dirty
    );
    cpu.network.layers[0].train.dont_update = 0;
    gpu.network.layers[0].train.dont_update = 0;
    cpu.update_weights(0.01, 0.9, 0.001, 4);
    gpu.update_weights_gpu(0.01, 0.9, 0.001, 4)
        .expect("GPU MoE SGD");
    assert!(!gpu.host_weights_stale);
    assert!(
        !gpu.device_moe_train_buffers.borrow()[0]
            .as_ref()
            .expect("GPU MoE device train buffers")
            .dirty
    );
    let LayerTrainBuffers::Moe(cpu_train) = &cpu.train_buffers[0] else {
        panic!("CPU MoE train buffers after SGD");
    };
    let device_train_buffers = gpu.device_moe_train_buffers.borrow();
    let gpu_train = device_train_buffers[0]
        .as_ref()
        .expect("GPU MoE device train buffers after SGD");
    assert_slice_close(
        "router weight momentum",
        &cpu_train.router_weight_updates,
        &gpu_train.router_weight_updates.data,
        2e-5,
    );
    assert_slice_close(
        "router bias momentum",
        &cpu_train.router_bias_updates,
        &gpu_train.router_bias_updates.data,
        2e-5,
    );
    let host_fallback = gpu.cpu_fallback_train_buffers.borrow();
    let LayerTrainBuffers::Moe(host_train) = &host_fallback[0] else {
        panic!("GPU host-fallback MoE buffers");
    };
    assert!(
        host_train
            .router_weight_updates
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(
        host_train
            .router_bias_updates
            .iter()
            .all(|value| *value == 0.0)
    );
    drop(host_fallback);
    drop(device_train_buffers);
    let LayerParams::Moe(cpu_params) = &cpu.params[0] else {
        unreachable!()
    };
    let LayerParams::Moe(gpu_params) = &gpu.params[0] else {
        unreachable!()
    };
    assert_slice_close(
        "router weights",
        &cpu_params.router_weights,
        &gpu_params.router_weights,
        2e-5,
    );
    assert_slice_close(
        "router biases",
        &cpu_params.router_biases,
        &gpu_params.router_biases,
        2e-5,
    );
    for (cpu_expert, gpu_expert) in cpu_params.experts.iter().zip(&gpu_params.experts) {
        assert_slice_close(
            "expert weights",
            &cpu_expert.weights,
            &gpu_expert.weights,
            2e-5,
        );
        assert_slice_close(
            "expert biases",
            &cpu_expert.biases,
            &gpu_expert.biases,
            2e-5,
        );
    }
}

#[test]
fn gpu_connected_layers_train_on_the_gpu_path() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "gpu-connected.cfg",
            "[net]\ninputs=2\nbatch=1\n[connected]\noutput=1\nactivation=linear\n",
        )
        .expect("connected cfg should parse"),
    )
    .expect("connected network should build");
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::Connected(params) = &mut gpu.params[0] else {
        panic!("expected connected parameters");
    };
    params.weights = vec![2.0, 3.0];
    params.biases = vec![0.5];
    let input = Tensor::from_data(gpu.network.net.input_shape(), vec![1.0, 2.0])
        .expect("connected input should have the expected shape");
    assert!(
        !gpu.requires_cpu_backward_fallback(),
        "connected layers must use the GPU backward path"
    );
    let forward = gpu
        .forward_with_state(&input)
        .expect("GPU connected forward should succeed");
    assert_eq!(forward.cpu_forward.final_output().unwrap().data, vec![8.5]);
    let seed = vec![Some(Tensor::filled(
        gpu.network.layers[0].output_shape,
        1.0,
    ))];
    gpu.backward_accumulate_seeded_gpu(&forward, &seed)
        .expect("GPU connected backward should succeed");
    gpu.update_weights_gpu(0.1, 0.0, 0.0, 1)
        .expect("GPU connected update should succeed");
    let LayerParams::Connected(params) = &gpu.params[0] else {
        panic!("expected connected parameters after update");
    };
    assert_eq!(params.weights, vec![2.1, 3.2]);
}

#[test]
fn local_layer_uses_position_specific_kernels_in_forward_and_backward() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "local-runtime.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n[local]\nfilters=1\nsize=2\nstride=1\npad=0\nactivation=linear\n",
            )
            .expect("local cfg should parse"),
        )
        .expect("local network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Local(params) = &mut executable.params[0] else {
        panic!("expected local parameters");
    };
    // location 0 owns [1, 2, 3, 4], location 1 owns [5, 6, 7, 8].
    params.biases = vec![0.5, -0.5];
    params.weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .expect("local input should have the expected shape");

    let forward = executable
        .forward(&input)
        .expect("local forward should succeed");
    assert_eq!(
        forward.final_output().expect("local output").data,
        vec![37.5, 110.5]
    );
    let seed = Tensor::from_data(executable.network.layers[0].output_shape, vec![1.0, 2.0])
        .expect("local gradient should have the expected shape");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("local backward should succeed");
    assert_eq!(
        backward.input_gradient.data,
        vec![1.0, 12.0, 12.0, 3.0, 18.0, 16.0]
    );
    let LayerParams::Local(gradient) = &backward.param_gradients[0] else {
        panic!("expected local gradients");
    };
    assert_eq!(gradient.biases, vec![1.0, 2.0]);
    assert_eq!(
        gradient.weights,
        vec![1.0, 2.0, 4.0, 5.0, 4.0, 6.0, 10.0, 12.0]
    );
}

#[test]
fn gru_axis_layout_keeps_samples_independent_and_gate_blocks_ordered() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "gru-axis.cfg",
            "[net]\ninputs=1\nbatch=2\ntime_steps=2\n[gru]\noutput=1\n",
        )
        .expect("GRU cfg should parse"),
    )
    .expect("GRU network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Gru(params) = &mut executable.params[0] else {
        panic!("expected GRU parameters");
    };
    for (projection, weight, bias) in [
        (&mut params.input_z, 1.0, -0.2),
        (&mut params.state_z, 0.5, 0.1),
        (&mut params.input_r, -0.7, 0.3),
        (&mut params.state_r, 0.4, -0.1),
        (&mut params.input_h, 0.9, 0.05),
        (&mut params.state_h, -0.6, 0.2),
    ] {
        projection.weights[0] = weight;
        projection.biases[0] = bias;
    }
    let input_values = [0.2_f32, -0.3, 0.7, 0.4];
    let input = Tensor::from_data(executable.network.net.input_shape(), input_values.to_vec())
        .expect("GRU input should have the expected shape");
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
    let mut expected = Vec::with_capacity(4);
    let mut previous = [0.0_f32; 2];
    for step in 0..2 {
        for sample in 0..2 {
            let x = input_values[step * 2 + sample];
            let p = previous[sample];
            let z = sigmoid(x * 1.0 - 0.2 + p * 0.5 + 0.1);
            let r = sigmoid(x * -0.7 + 0.3 + p * 0.4 - 0.1);
            let candidate = sigmoid(x * 0.9 + 0.05 + (r * p) * -0.6 + 0.2);
            let state = z * p + (1.0 - z) * candidate;
            expected.push(state);
            previous[sample] = state;
        }
    }
    let forward = executable
        .forward(&input)
        .expect("GRU forward should succeed");
    let actual = forward.final_output().expect("GRU output");
    assert_slice_close("GRU sample/gate axis layout", &actual.data, &expected, 1e-7);
}

#[test]
fn gru_layer_applies_darknet_update_gate_over_sequence_steps() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "gru-runtime.cfg",
            "[net]\ninputs=1\nbatch=2\ntime_steps=2\n[gru]\noutput=1\n",
        )
        .expect("GRU cfg should parse"),
    )
    .expect("GRU network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![0.0, 0.0, 0.0, 0.0],
    )
    .expect("GRU input should have the expected shape");
    let forward = executable
        .forward(&input)
        .expect("GRU forward should succeed");
    assert_eq!(
        forward.final_output().expect("GRU output").data,
        vec![0.25, 0.25, 0.375, 0.375]
    );
}

pub(super) fn fill_conv_params(params: &mut [LayerParams]) {
    for (layer, params) in params.iter_mut().enumerate() {
        if let LayerParams::Convolutional(conv) = params {
            for (i, w) in conv.weights.iter_mut().enumerate() {
                *w = ((i + 3 * layer) as f32 * 0.7).sin() * 0.6;
            }
            for (i, b) in conv.biases.iter_mut().enumerate() {
                *b = ((i + layer) as f32 * 1.3).cos() * 0.2;
            }
        }
    }
}

pub(super) fn patterned_input(shape: TensorShape) -> Tensor {
    let data = (0..shape.batch * shape.element_count())
        .map(|i| (i as f32 * 0.37).sin() + 0.1 * (i % 5) as f32)
        .collect();
    Tensor::from_data(shape, data).expect("input tensor")
}

/// Runs the same training forward + seeded backward on the CPU network and
/// on the GPU network (fake backend, default host layer ops) and compares
/// every layer output, dropout mask and convolution gradient.
pub(super) fn assert_gpu_layer_chain_matches_cpu(cfg_text: &str, seed_layer: usize) {
    let network = Network::from_config(
        ConfigFile::parse_str("inline.cfg", cfg_text).expect("cfg should parse"),
    )
    .expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    fill_conv_params(&mut cpu.params);
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");
    assert!(
        !gpu.requires_cpu_backward_fallback(),
        "chain must use the GPU backward path"
    );

    let input = patterned_input(cpu.network.net.input_shape());
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu
        .forward_training_with_state(&input)
        .expect("GPU forward");
    for (index, (lhs, rhs)) in cpu_forward
        .layer_outputs
        .iter()
        .zip(&gpu_forward.cpu_forward.layer_outputs)
        .enumerate()
    {
        assert_eq!(lhs.shape, rhs.shape);
        assert_slice_close(&format!("layer {index} output"), &lhs.data, &rhs.data, 1e-5);
    }
    assert_eq!(
        cpu_forward.dropout_masks,
        gpu_forward.cpu_forward.dropout_masks
    );

    let mut seeds = vec![None; cpu.network.layers.len()];
    let seed_shape = cpu.network.layers[seed_layer].output_shape;
    seeds[seed_layer] = Some(
        Tensor::from_data(
            seed_shape,
            (0..seed_shape.batch * seed_shape.element_count())
                .map(|i| (i as f32 * 0.91).cos())
                .collect(),
        )
        .expect("seed tensor"),
    );
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU backward");

    let gpu_buffers = gpu.device_train_buffers.borrow();
    let mut checked = 0;
    for (index, train) in cpu.train_buffers.iter().enumerate() {
        let LayerTrainBuffers::Convolutional(cpu_train) = train else {
            continue;
        };
        let gpu_train = gpu_buffers[index].as_ref().expect("GPU train buffers");
        assert_slice_close(
            &format!("layer {index} weight gradient"),
            &cpu_train.weight_updates,
            &gpu_train.weight_updates.data,
            1e-4,
        );
        assert_slice_close(
            &format!("layer {index} bias gradient"),
            &cpu_train.bias_updates,
            &gpu_train.bias_updates.data,
            1e-4,
        );
        assert!(cpu_train.weight_updates.iter().any(|g| g.abs() > 1e-6));
        checked += 1;
    }
    assert!(checked > 0);
}

#[test]
fn grad_mode_parses_analytical_default_and_cpp() {
    assert_eq!(
        crate::runtime::GradMode::parse(""),
        crate::runtime::GradMode::Analytical
    );
    assert_eq!(
        crate::runtime::GradMode::parse("analytical"),
        crate::runtime::GradMode::Analytical
    );
    assert_eq!(
        crate::runtime::GradMode::parse("garbage"),
        crate::runtime::GradMode::Analytical
    );
    assert_eq!(
        crate::runtime::GradMode::parse("CPP"),
        crate::runtime::GradMode::Cpp
    );
    assert_eq!(
        crate::runtime::GradMode::parse("c++"),
        crate::runtime::GradMode::Cpp
    );
}

#[test]
fn gpu_dropout_reorg_softmax_match_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [dropout]\nprobability=0.4\n\
             [reorg]\nstride=2\n\
             [softmax]\ngroups=2\n",
        3,
    );
}

#[test]
fn gpu_reverse_reorg_matches_cpu_forward_and_backward() {
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [reorg]\nstride=2\n\
             [reorg]\nstride=2\nreverse=1\n",
        2,
    );
}

#[test]
fn gpu_scale_channels_matches_cpu_forward_and_backward() {
    // per-channel scale (avgpool output) and per-pixel scale (1-filter conv)
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [avgpool]\n\
             [scale_channels]\nfrom=-2\nactivation=leaky\n",
        2,
    );
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [convolutional]\nfilters=1\nsize=1\nstride=1\nactivation=linear\n\
             [scale_channels]\nfrom=-2\nscale_wh=1\nactivation=linear\n",
        2,
    );
}

/// Darknet resizes a mismatched shortcut source flat per sample (zero-pad
/// when the source is smaller, crop when larger); the GPU path used to fall
/// back to the CPU for the whole network.
#[test]
fn gpu_mismatched_shortcut_matches_cpu_forward_and_backward() {
    // source smaller than the layer: 2-channel source into a 4-channel input.
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
             [shortcut]\nfrom=-2\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        3,
    );
    // source larger than the layer: the 4-channel source is cropped.
    assert_gpu_layer_chain_matches_cpu(
        "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=leaky\n\
             [convolutional]\nfilters=2\nsize=3\nstride=1\npad=1\nactivation=linear\n\
             [shortcut]\nfrom=-2\nactivation=linear\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
        3,
    );
}

/// Chained conv + activation stacks (incl. relu6 and ramp): GPU-runtime
/// weight/bias gradients equal the CPU `ExecutableNetwork` backward.
#[test]
fn gpu_activation_chains_match_cpu_forward_and_backward() {
    for act in [
        "leaky",
        "loggy",
        "hardtan",
        "relie",
        "lhtan",
        "plse",
        "stair",
        "relu6",
        "ramp",
        "mish",
        "swish",
        "logistic",
        "relu",
        "elu",
        "selu",
        "hard_mish",
    ] {
        assert_gpu_layer_chain_matches_cpu(
            &format!(
                "[net]\nwidth=4\nheight=4\nchannels=2\nbatch=2\n\
                     [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nactivation={act}\n\
                     [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation={act}\n"
            ),
            1,
        );
    }
}
