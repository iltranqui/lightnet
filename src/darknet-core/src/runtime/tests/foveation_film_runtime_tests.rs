use super::*;
use crate::cfg::ConfigFile;
use crate::weights::inspect_weights_file;
use std::fs;

fn build_film_network(net: &str, layers: &str) -> Network {
    Network::from_config(
        ConfigFile::parse_str("foveation-film-runtime.cfg", &format!("{net}\n{layers}"))
            .expect("foveation-film cfg parses"),
    )
    .expect("foveation-film network builds")
}

fn learned_film_network(backprop_to_hf: bool, net_options: &str) -> Network {
    build_film_network(
        &format!("[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n{net_options}"),
        &format!(
            "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[hf]\nparams=learned\npred_hidden=2\ndownsample=1\npred_source_layer=0\ninit_cx=0.05\ninit_cy=-0.04\ninit_radius=0.8\ninit_strength=0.3\n[foveation_film]\nembed_dim=2\nfilm_init_scale=0.25\nbackprop_to_hf={}\n",
            usize::from(backprop_to_hf)
        ),
    )
}

fn configure_learned_film(executable: &mut ExecutableNetwork) {
    let LayerParams::Convolutional(conv) = &mut executable.params[0] else {
        panic!("expected source convolution parameters")
    };
    conv.weights.fill(0.0);
    conv.biases.fill(0.0);

    let LayerParams::HyperbolicFoveation(hf) = &mut executable.params[1] else {
        panic!("expected learned HF parameters")
    };
    hf.pred_w1.fill(0.025);
    hf.pred_b1.copy_from_slice(&[0.6, 0.7]);
    hf.pred_w2
        .copy_from_slice(&[0.08, -0.03, 0.04, 0.07, -0.05, 0.06, 0.03, 0.09]);
    hf.pred_b2.copy_from_slice(&[0.1, -0.08, 0.3, 0.2]);

    let LayerParams::FoveationFilm(film) = &mut executable.params[2] else {
        panic!("expected FiLM parameters")
    };
    film.w_embed
        .copy_from_slice(&[0.2, 0.1, -0.1, 0.2, 0.15, 0.05, 0.1, -0.08]);
    film.b_embed.copy_from_slice(&[0.5, 0.7]);
    film.w_gamma.copy_from_slice(&[0.1, -0.2]);
    film.b_gamma[0] = 0.05;
    film.w_beta.copy_from_slice(&[0.3, 0.2]);
    film.b_beta[0] = 0.1;
}

fn scalar_loss(executable: &ExecutableNetwork, input: &Tensor, seed: &Tensor) -> f32 {
    let output = executable.forward(input).unwrap().layer_outputs[2].clone();
    output
        .data
        .iter()
        .zip(&seed.data)
        .map(|(value, gradient)| value * gradient)
        .sum()
}

#[test]
fn foveation_film_initialization_is_identity_and_seeded() {
    let network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n",
        "[hf]\nparams=fixed\ninit_cx=0\ninit_cy=0\ninit_radius=1\ninit_strength=0\n[foveation_film]\nembed_dim=3\n",
    );
    let mut first = ExecutableNetwork::new(network.clone());
    let mut second = ExecutableNetwork::new(network.clone());
    crate::initialize::initialize_training_params(&network, &mut first.params, 73);
    crate::initialize::initialize_training_params(&network, &mut second.params, 73);
    assert_eq!(first.params, second.params);
    let input = Tensor::from_data(
        network.net.input_shape(),
        (0..network.net.input_shape().batch * network.net.input_shape().element_count())
            .map(|index| (index as f32 - 5.0) * 0.1)
            .collect(),
    )
    .unwrap();
    let output = first.forward(&input).unwrap();
    assert_eq!(output.layer_outputs[1].data, output.layer_outputs[0].data);
}

#[test]
fn foveation_film_backward_routes_additive_gradients_through_hf_predictor_and_source() {
    let network = learned_film_network(true, "");
    let mut executable = ExecutableNetwork::new(network.clone());
    configure_learned_film(&mut executable);
    let input = Tensor::zeros(network.net.input_shape());
    let forward = executable.forward(&input).unwrap();
    let seed = Tensor::from_data(network.layers[2].output_shape, vec![1.0; 6]).unwrap();
    let mut seeds = vec![None; network.layers.len()];
    seeds[2] = Some(seed.clone());
    let backward = executable
        .backward_seeded(&input, &forward, &seeds)
        .unwrap();

    let LayerParams::FoveationFilm(film_gradient) = &backward.param_gradients[2] else {
        panic!("expected FiLM parameter gradients")
    };
    assert!(film_gradient.b_beta[0] > 0.0);
    assert!(
        film_gradient
            .w_beta
            .iter()
            .any(|value| value.abs() > 1.0e-6)
    );
    let LayerParams::HyperbolicFoveation(hf_gradient) = &backward.param_gradients[1] else {
        panic!("expected HF predictor gradients")
    };
    assert!(hf_gradient.pred_b2.iter().any(|value| value.abs() > 1.0e-6));
    assert!(hf_gradient.pred_w2.iter().any(|value| value.abs() > 1.0e-6));
    let LayerParams::Convolutional(conv_gradient) = &backward.param_gradients[0] else {
        panic!("expected predictor source convolution gradients")
    };
    assert!(conv_gradient.biases[0].abs() > 1.0e-6);
    assert!(
        backward.layer_output_gradients[0]
            .data
            .iter()
            .any(|value| value.abs() > 1.0e-6)
    );

    let mut accumulating = ExecutableNetwork::new(network.clone());
    configure_learned_film(&mut accumulating);
    let accumulated_forward = accumulating.forward(&input).unwrap();
    let accumulated = accumulating
        .backward_accumulate_seeded(&input, &accumulated_forward, &seeds)
        .unwrap();
    let LayerTrainBuffers::FoveationFilm(train) = &accumulating.train_buffers[2] else {
        unreachable!()
    };
    let LayerParams::FoveationFilm(gradient) = &accumulated.param_gradients[2] else {
        unreachable!()
    };
    assert_eq!(train.w_embed_updates, gradient.w_embed);
    assert_eq!(train.b_beta_updates, gradient.b_beta);

    let epsilon = 1.0e-3;
    let base_film = match &executable.params[2] {
        LayerParams::FoveationFilm(params) => params.clone(),
        _ => unreachable!(),
    };
    for (index, analytic) in [(0, film_gradient.w_beta[0])] {
        let mut plus = base_film.clone();
        let mut minus = base_film.clone();
        plus.w_beta[index] += epsilon;
        minus.w_beta[index] -= epsilon;
        executable.params[2] = LayerParams::FoveationFilm(plus);
        let loss_plus = scalar_loss(&executable, &input, &seed);
        executable.params[2] = LayerParams::FoveationFilm(minus);
        let loss_minus = scalar_loss(&executable, &input, &seed);
        let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert!(
            (numeric - analytic).abs() < 2.0e-3,
            "film: {numeric} != {analytic}"
        );
    }
    executable.params[2] = LayerParams::FoveationFilm(base_film);

    let base_hf = match &executable.params[1] {
        LayerParams::HyperbolicFoveation(params) => params.clone(),
        _ => unreachable!(),
    };
    let mut plus = base_hf.clone();
    let mut minus = base_hf.clone();
    plus.pred_b2[0] += epsilon;
    minus.pred_b2[0] -= epsilon;
    executable.params[1] = LayerParams::HyperbolicFoveation(plus);
    let loss_plus = scalar_loss(&executable, &input, &seed);
    executable.params[1] = LayerParams::HyperbolicFoveation(minus);
    let loss_minus = scalar_loss(&executable, &input, &seed);
    let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert!(
        (numeric - hf_gradient.pred_b2[0]).abs() < 2.0e-3,
        "HF predictor bias: {numeric} != {}",
        hf_gradient.pred_b2[0]
    );
    executable.params[1] = LayerParams::HyperbolicFoveation(base_hf);

    let LayerParams::Convolutional(conv) = &executable.params[0] else {
        unreachable!()
    };
    let original_bias = conv.biases[0];
    if let LayerParams::Convolutional(conv) = &mut executable.params[0] {
        conv.biases[0] = original_bias + epsilon;
    }
    let loss_plus = scalar_loss(&executable, &input, &seed);
    if let LayerParams::Convolutional(conv) = &mut executable.params[0] {
        conv.biases[0] = original_bias - epsilon;
    }
    let loss_minus = scalar_loss(&executable, &input, &seed);
    let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert!(
        (numeric - conv_gradient.biases[0]).abs() < 3.0e-3,
        "HF predictor source: {numeric} != {}",
        conv_gradient.biases[0]
    );

    let detached_network = learned_film_network(false, "");
    let mut detached = ExecutableNetwork::new(detached_network.clone());
    configure_learned_film(&mut detached);
    let detached_forward = detached.forward(&input).unwrap();
    let detached_backward = detached
        .backward_seeded(&input, &detached_forward, &seeds)
        .unwrap();
    let LayerParams::HyperbolicFoveation(detached_hf) = &detached_backward.param_gradients[1]
    else {
        unreachable!()
    };
    assert!(detached_hf.pred_w1.iter().all(|value| value.abs() < 1.0e-7));
    assert!(detached_hf.pred_b1.iter().all(|value| value.abs() < 1.0e-7));
    assert!(detached_hf.pred_w2.iter().all(|value| value.abs() < 1.0e-7));
    assert!(detached_hf.pred_b2.iter().all(|value| value.abs() < 1.0e-7));
}

#[test]
fn foveation_film_sources_add_to_each_other_and_existing_hf_auxiliary_gradients() {
    let network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n",
        "[hf]\nparams=fixed\ninit_cx=0.1\ninit_cy=-0.1\ninit_radius=0.8\ninit_strength=0.4\n[foveation_film]\nembed_dim=2\nfilm_init_scale=0.2\n[foveation_film]\nembed_dim=2\nfilm_init_scale=0.2\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    if let LayerParams::HyperbolicFoveation(hf) = &mut executable.params[0] {
        hf.hf_params.copy_from_slice(&[0.1, -0.1, 0.8, 0.4]);
    }
    for index in [1, 2] {
        let LayerParams::FoveationFilm(film) = &mut executable.params[index] else {
            unreachable!()
        };
        film.w_embed.fill(0.08 * index as f32);
        film.b_embed.fill(0.6);
        film.w_gamma.fill(0.03 * index as f32);
        film.b_gamma.fill(0.01);
        film.w_beta.fill(0.04 * index as f32);
        film.b_beta.fill(0.02);
    }
    let input = Tensor::from_data(
        network.net.input_shape(),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    let mut seeds = vec![None; network.layers.len()];
    seeds[2] = Some(Tensor::from_data(network.layers[2].output_shape, vec![1.0; 6]).unwrap());
    let without_auxiliary = executable
        .backward_seeded(&input, &forward, &seeds)
        .unwrap();
    let mut auxiliary = vec![None; network.layers.len()];
    auxiliary[0] = Some(vec![0.13, -0.27, 0.31, -0.09]);
    let mut workspace = vec![0.0; crate::runtime::compute_cpu_workspace_size(&network)];
    let mut attention_backward =
        |d_context: &[f32], _: &[f32], attention: &crate::vit::AttentionCache| {
            crate::vit::attention_backward(d_context, attention)
        };
    let with_auxiliary = executable
        .backward_seeded_core(
            &mut workspace,
            &input,
            &forward,
            &seeds,
            Some(&auxiliary),
            &mut attention_backward,
        )
        .unwrap();
    let LayerParams::HyperbolicFoveation(plain) = &without_auxiliary.param_gradients[0] else {
        unreachable!()
    };
    let LayerParams::HyperbolicFoveation(combined) = &with_auxiliary.param_gradients[0] else {
        unreachable!()
    };
    for (index, seed) in auxiliary[0].as_ref().unwrap().iter().enumerate() {
        assert!((combined.hf_params[index] - plain.hf_params[index] - seed).abs() < 1.0e-6);
    }
}

#[test]
fn foveation_film_sgd_honors_dont_update_burnin_and_train_only_bn_gates() {
    let mut dont_update = ExecutableNetwork::new(build_film_network(
        "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=1\ndont_update=1\n",
    ));
    let LayerTrainBuffers::FoveationFilm(train) = &mut dont_update.train_buffers[1] else {
        unreachable!()
    };
    train.w_embed_updates[0] = 1.0;
    dont_update.set_train_iteration(0);
    dont_update.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &dont_update.params[1] else {
        unreachable!()
    };
    assert_eq!(film.w_embed[0], 0.0);
    dont_update.set_train_iteration(1);
    dont_update.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &dont_update.params[1] else {
        unreachable!()
    };
    assert!((film.w_embed[0] - 0.1).abs() < 1.0e-6);

    let mut burnin = ExecutableNetwork::new(build_film_network(
        "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\nburn_in=2\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=1\nburnin_update=1\n",
    ));
    let LayerTrainBuffers::FoveationFilm(train) = &mut burnin.train_buffers[1] else {
        unreachable!()
    };
    train.w_embed_updates[0] = 1.0;
    burnin.set_train_iteration(0);
    burnin.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &burnin.params[1] else {
        unreachable!()
    };
    assert_eq!(film.w_embed[0], 0.0);
    burnin.set_train_iteration(1);
    burnin.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &burnin.params[1] else {
        unreachable!()
    };
    assert!((film.w_embed[0] - 0.1).abs() < 1.0e-6);

    let mut train_only_bn = ExecutableNetwork::new(build_film_network(
        "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=1\ntrain_only_bn=1\n",
    ));
    let LayerTrainBuffers::FoveationFilm(train) = &mut train_only_bn.train_buffers[1] else {
        unreachable!()
    };
    train.w_embed_updates[0] = 1.0;
    train_only_bn.set_train_iteration(1);
    train_only_bn.update_weights(0.1, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &train_only_bn.params[1] else {
        unreachable!()
    };
    assert_eq!(film.w_embed[0], 0.0);
}

#[test]
fn foveation_film_weights_round_trip_exact_source_order() {
    let network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=3\nfilm_init_scale=0.2\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    if let LayerParams::HyperbolicFoveation(hf) = &mut executable.params[0] {
        hf.hf_params.copy_from_slice(&[0.11, -0.12, 0.81, 0.37]);
    }
    let expected = [
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        vec![13.0, 14.0, 15.0],
        vec![16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
        vec![22.0, 23.0],
        vec![24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
        vec![30.0, 31.0],
    ];
    if let LayerParams::FoveationFilm(film) = &mut executable.params[1] {
        film.w_embed.clone_from(&expected[0]);
        film.b_embed.clone_from(&expected[1]);
        film.w_gamma.clone_from(&expected[2]);
        film.b_gamma.clone_from(&expected[3]);
        film.w_beta.clone_from(&expected[4]);
        film.b_beta.clone_from(&expected[5]);
    }
    let original = executable.params.clone();
    let path = std::env::temp_dir().join(format!(
        "darknet-foveation-film-weights-{}.weights",
        std::process::id()
    ));
    crate::weights::save_weights(&network, &executable.params, &path, 19).unwrap();
    let inspection = inspect_weights_file(&network, &path).unwrap();
    let arrays = &inspection.layers[1].arrays;
    assert_eq!(
        arrays
            .iter()
            .map(|array| array.name.as_str())
            .collect::<Vec<_>>(),
        [
            "w_embed", "b_embed", "w_gamma", "b_gamma", "w_beta", "b_beta"
        ]
    );
    for (array, values) in arrays.iter().zip(&expected) {
        assert_eq!(array.len, values.len());
        for (index, expected_value) in values.iter().enumerate() {
            let offset = array.byte_offset as usize + index * 4;
            let bytes: [u8; 4] = fs::read(&path).unwrap()[offset..offset + 4]
                .try_into()
                .unwrap();
            assert_eq!(f32::from_le_bytes(bytes), *expected_value);
        }
    }
    let expected_count = 4 + expected.iter().map(Vec::len).sum::<usize>();
    assert_eq!(inspection.file_size, 20 + (expected_count * 4) as u64);
    assert_eq!(inspection.trailing_bytes, 0);
    let mut restored = ExecutableNetwork::new(network.clone());
    restored.load_weights_file(&path).unwrap();
    assert_eq!(restored.params, original);
    let _ = fs::remove_file(path);
}

#[test]
fn foveation_film_sgd_adamw_muon_and_gpu_boundary() {
    let network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=2\n",
    );
    let mut sgd = ExecutableNetwork::new(network.clone());
    let LayerTrainBuffers::FoveationFilm(train) = &mut sgd.train_buffers[1] else {
        panic!("expected FiLM train buffers")
    };
    train.w_embed_updates[0] = 1.0;
    train.b_beta_updates[0] = 0.5;
    let before = match &sgd.params[1] {
        LayerParams::FoveationFilm(film) => (film.w_embed[0], film.b_beta[0]),
        _ => unreachable!(),
    };
    sgd.update_weights(0.2, 0.0, 0.0, 1);
    let LayerParams::FoveationFilm(film) = &sgd.params[1] else {
        unreachable!()
    };
    assert!((film.w_embed[0] - (before.0 + 0.2)).abs() < 1.0e-6);
    assert!((film.b_beta[0] - (before.1 + 0.1)).abs() < 1.0e-6);

    let adam_network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nadam=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=2\n",
    );
    let mut adam = ExecutableNetwork::new(adam_network.clone());
    let LayerTrainBuffers::FoveationFilm(train) = &mut adam.train_buffers[1] else {
        unreachable!()
    };
    train.w_embed_updates.fill(0.2);
    train.b_embed_updates.fill(0.1);
    train.w_gamma_updates.fill(-0.3);
    train.b_gamma_updates.fill(0.15);
    train.w_beta_updates.fill(0.25);
    train.b_beta_updates.fill(-0.05);
    adam.update_weights_adam(0.01, 1).unwrap();
    let adam_path = std::env::temp_dir().join(format!(
        "darknet-foveation-film-adam-{}.opt",
        std::process::id()
    ));
    adam.save_optimizer_state(&adam_path).unwrap();
    let mut adam_restored = ExecutableNetwork::new(adam_network);
    adam_restored.load_optimizer_state(&adam_path).unwrap();
    assert_eq!(adam_restored.host_adam_state, adam.host_adam_state);
    let mismatched_network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nadam=1\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=3\n",
    );
    let mut mismatched = ExecutableNetwork::new(mismatched_network);
    assert!(mismatched.load_optimizer_state(&adam_path).is_err());
    let _ = fs::remove_file(adam_path);

    let muon_network = build_film_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.001\n",
        "[hf]\nparams=fixed\n[foveation_film]\nembed_dim=2\n",
    );
    let mut muon = ExecutableNetwork::new(muon_network.clone());
    let LayerTrainBuffers::FoveationFilm(train) = &mut muon.train_buffers[1] else {
        unreachable!()
    };
    train.w_embed_updates.fill(0.2);
    train.w_gamma_updates.fill(-0.1);
    train.w_beta_updates.fill(0.15);
    train.b_embed_updates.fill(0.05);
    train.b_gamma_updates.fill(0.03);
    train.b_beta_updates.fill(-0.02);
    muon.update_weights_muon(0.01, 1).unwrap();
    let matrices = &muon.muon_state[1].as_ref().unwrap().matrices;
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols))
            .collect::<Vec<_>>(),
        [(2, 4), (1, 2), (1, 2)]
    );
    let muon_path = std::env::temp_dir().join(format!(
        "darknet-foveation-film-muon-{}.opt",
        std::process::id()
    ));
    muon.save_optimizer_state(&muon_path).unwrap();
    let mut muon_restored = ExecutableNetwork::new(muon_network);
    muon_restored.load_optimizer_state(&muon_path).unwrap();
    assert_eq!(muon_restored.muon_state[1], muon.muon_state[1]);
    let _ = fs::remove_file(muon_path);

    let f16_rejection = crate::runtime::reject_gpu_hyperbolic_foveation(&network, "f16 inference")
        .unwrap_err()
        .to_string();
    assert!(f16_rejection.contains("GPU f16 inference is not implemented"));
    let gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let input = Tensor::zeros(network.net.input_shape());
    assert!(
        gpu.forward(&input)
            .unwrap_err()
            .to_string()
            .contains("GPU execution is not implemented in Stage-1 CPU support")
    );
    assert!(
        gpu.forward_compact(&input)
            .unwrap_err()
            .to_string()
            .contains("GPU compact inference is not implemented in Stage-1 CPU support")
    );
    let dummy_state = crate::runtime::GpuForwardState {
        input_gpu: FakeTensor {
            shape: vec![1, 1, 2, 3],
            data: vec![0.0; 6],
        },
        gpu_layer_outputs: Vec::new(),
        gpu_layer_activation_inputs: Vec::new(),
        gpu_batch_norm_caches: Vec::new(),
        gpu_rnn_caches: Vec::new(),
        gpu_crnn_caches: Vec::new(),
        gpu_lstm_caches: Vec::new(),
        gpu_moe_caches: Vec::new(),
        gpu_eml_caches: Vec::new(),
        cpu_forward: ForwardResult::from_heads(Vec::new(), Vec::new()),
    };
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    assert!(
        gpu.backward_accumulate_seeded_gpu(&dummy_state, &[])
            .unwrap_err()
            .to_string()
            .contains("GPU backward and updates is not implemented in Stage-1 CPU support")
    );
    assert!(
        gpu.update_weights_gpu(0.01, 0.0, 0.0, 1)
            .unwrap_err()
            .to_string()
            .contains("GPU optimizer updates is not implemented in Stage-1 CPU support")
    );
}
