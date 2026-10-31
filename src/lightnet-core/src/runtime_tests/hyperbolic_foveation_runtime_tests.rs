use super::*;
use crate::cfg::ConfigFile;
use crate::weights::inspect_weights_file;
use std::fs;

fn build_hf_network(net: &str, layers: &str) -> Network {
    Network::from_config(
        ConfigFile::parse_str(
            "hyperbolic-foveation-runtime.cfg",
            &format!("{net}\n{layers}"),
        )
        .expect("hyperbolic-foveation cfg parses"),
    )
    .expect("hyperbolic-foveation network builds")
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn set_hf_params(executable: &mut ExecutableNetwork, layer_index: usize, values: &[f32]) {
    let LayerParams::HyperbolicFoveation(params) = &mut executable.params[layer_index] else {
        panic!("expected hyperbolic-foveation params")
    };
    params.hf_params.copy_from_slice(values);
}

fn hf_loss(executable: &ExecutableNetwork, input: &Tensor, seed: &Tensor) -> f32 {
    let forward = executable
        .forward(input)
        .expect("hyperbolic-foveation forward");
    dot(&forward.layer_outputs.last().unwrap().data, &seed.data)
}

fn hf_cfg(mode: &str) -> String {
    format!(
        "[hyperbolic_foveation]\ninit_cx=0.08\ninit_cy=-0.06\ninit_radius=0.95\ninit_radius_x=0.82\ninit_radius_y=1.08\ninit_strength=0.34\nalpha=1.7\np_exp=2.2\nradius_mode={mode}\nnr_iters=4\nnr_eta=0.7\n"
    )
}

fn seeded_hf_network(mode: &str) -> Network {
    build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=2\nbatch=2\n",
        &hf_cfg(mode),
    )
}

fn sample_input(shape: TensorShape) -> Tensor {
    Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|index| 0.2 + (index as f32 * 0.037).sin() * 0.4 + index as f32 * 0.002)
            .collect(),
    )
    .unwrap()
}

fn central_seed(shape: TensorShape) -> Tensor {
    let mut seed = vec![0.0; shape.batch * shape.element_count()];
    for n in 0..shape.batch {
        for c in 0..shape.channels {
            for y in 1..shape.height.saturating_sub(1) {
                for x in 1..shape.width.saturating_sub(1) {
                    let index = ((n * shape.channels + c) * shape.height + y) * shape.width + x;
                    seed[index] = 0.1 + index as f32 * 0.003;
                }
            }
        }
    }
    Tensor::from_data(shape, seed).unwrap()
}

#[test]
fn global_isotropic_and_anisotropic_forward_backward_match_finite_differences() {
    for (mode, parameter_count) in [("isotropic", 4), ("anisotropic", 5)] {
        let network = seeded_hf_network(mode);
        let mut executable = ExecutableNetwork::new(network.clone());
        let input = sample_input(network.net.input_shape());
        let output_shape = network.layers[0].output_shape;
        let seed = central_seed(output_shape);
        let forward = executable.forward(&input).unwrap();
        let backward = executable
            .backward_seeded(&input, &forward, &[Some(seed.clone())])
            .unwrap();
        let LayerParams::HyperbolicFoveation(gradient) = &backward.param_gradients[0] else {
            panic!("expected hyperbolic-foveation parameter gradients")
        };
        assert_eq!(gradient.hf_params.len(), parameter_count);

        let original = match &executable.params[0] {
            LayerParams::HyperbolicFoveation(params) => params.hf_params.clone(),
            _ => unreachable!(),
        };
        let epsilon = 1.0e-3;
        for (index, analytic) in gradient.hf_params.iter().copied().enumerate() {
            let mut plus = original.clone();
            let mut minus = original.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            set_hf_params(&mut executable, 0, &plus);
            let loss_plus = hf_loss(&executable, &input, &seed);
            set_hf_params(&mut executable, 0, &minus);
            let loss_minus = hf_loss(&executable, &input, &seed);
            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!(
                (numeric - analytic).abs() < 0.025,
                "mode={mode}, parameter={index}, numeric={numeric}, analytic={analytic}"
            );
        }
        set_hf_params(&mut executable, 0, &original);

        let input_index =
            ((1 * input.shape.channels) * input.shape.height + 1) * input.shape.width + 2;
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus.data[input_index] += epsilon;
        minus.data[input_index] -= epsilon;
        let numeric = (hf_loss(&executable, &plus, &seed) - hf_loss(&executable, &minus, &seed))
            / (2.0 * epsilon);
        let analytic = backward.input_gradient.data[input_index];
        assert!(
            (numeric - analytic).abs() < 0.02,
            "mode={mode}, input index={input_index}, numeric={numeric}, analytic={analytic}"
        );
    }
}

#[test]
fn learned_predictor_fans_gradients_into_selected_source_and_network_input() {
    let network = build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=1\nbatch=2\n",
        "[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[hyperbolic_foveation]\npred_hidden=2\ndownsample=2\npred_source_layer=0\nwarp_source=network_input\ninit_cx=0.03\ninit_cy=-0.04\ninit_radius=0.9\ninit_strength=0.38\nnr_iters=3\nnr_eta=0.72\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::Convolutional(conv) = &mut executable.params[0] else {
        panic!("expected convolution params")
    };
    conv.weights.fill(0.8);
    conv.biases.fill(0.1);
    let LayerParams::HyperbolicFoveation(hf) = &mut executable.params[1] else {
        panic!("expected hyperbolic-foveation params")
    };
    hf.pred_w1
        .copy_from_slice(&[0.08, 0.05, 0.07, 0.04, 0.06, 0.09, 0.03, 0.08]);
    hf.pred_b1.copy_from_slice(&[0.45, 0.55]);
    hf.pred_w2
        .copy_from_slice(&[0.03, -0.02, -0.01, 0.04, 0.02, 0.01, -0.03, 0.05]);

    let input = sample_input(network.net.input_shape());
    let seed = central_seed(network.layers[1].output_shape);
    let forward = executable.forward(&input).unwrap();
    let backward = executable
        .backward_seeded(&input, &forward, &[None, Some(seed.clone())])
        .unwrap();
    let LayerParams::HyperbolicFoveation(gradient) = &backward.param_gradients[1] else {
        panic!("expected predictor gradients")
    };
    let LayerParams::Convolutional(conv_gradient) = &backward.param_gradients[0] else {
        panic!("expected source-layer convolution gradients")
    };
    assert!(conv_gradient.weights[0].abs() > 1.0e-5);
    assert!(
        backward.layer_output_gradients[0]
            .data
            .iter()
            .any(|value| *value != 0.0)
    );

    let epsilon = 1.0e-3;
    let base_hf = match &executable.params[1] {
        LayerParams::HyperbolicFoveation(params) => params.clone(),
        _ => unreachable!(),
    };
    for (block, index, analytic) in [
        ("w1", 0, gradient.pred_w1[0]),
        ("b1", 0, gradient.pred_b1[0]),
        ("w2", 5, gradient.pred_w2[5]),
        ("b2", 0, gradient.pred_b2[0]),
    ] {
        let mut plus = base_hf.clone();
        let mut minus = base_hf.clone();
        match block {
            "w1" => {
                plus.pred_w1[index] += epsilon;
                minus.pred_w1[index] -= epsilon;
            }
            "b1" => {
                plus.pred_b1[index] += epsilon;
                minus.pred_b1[index] -= epsilon;
            }
            "w2" => {
                plus.pred_w2[index] += epsilon;
                minus.pred_w2[index] -= epsilon;
            }
            "b2" => {
                plus.pred_b2[index] += epsilon;
                minus.pred_b2[index] -= epsilon;
            }
            _ => unreachable!(),
        }
        executable.params[1] = LayerParams::HyperbolicFoveation(plus);
        let loss_plus = hf_loss(&executable, &input, &seed);
        executable.params[1] = LayerParams::HyperbolicFoveation(minus);
        let loss_minus = hf_loss(&executable, &input, &seed);
        let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert!(
            (numeric - analytic).abs() < 0.03,
            "predictor {block}[{index}]: numeric={numeric}, analytic={analytic}"
        );
    }
    executable.params[1] = LayerParams::HyperbolicFoveation(base_hf);

    let mut plus = input.clone();
    let mut minus = input.clone();
    let input_index = 1 * input.shape.height * input.shape.width + 2;
    plus.data[input_index] += epsilon;
    minus.data[input_index] -= epsilon;
    let numeric = (hf_loss(&executable, &plus, &seed) - hf_loss(&executable, &minus, &seed))
        / (2.0 * epsilon);
    let analytic = backward.input_gradient.data[input_index];
    assert!(
        (numeric - analytic).abs() < 0.025,
        "fan-in input derivative: numeric={numeric}, analytic={analytic}"
    );

    let LayerParams::Convolutional(source_params) = &executable.params[0] else {
        unreachable!()
    };
    let original_weight = source_params.weights[0];
    if let LayerParams::Convolutional(params) = &mut executable.params[0] {
        params.weights[0] = original_weight + epsilon;
    }
    let loss_plus = hf_loss(&executable, &input, &seed);
    if let LayerParams::Convolutional(params) = &mut executable.params[0] {
        params.weights[0] = original_weight - epsilon;
    }
    let loss_minus = hf_loss(&executable, &input, &seed);
    let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert!(
        (numeric - conv_gradient.weights[0]).abs() < 0.03,
        "predictor-source convolution derivative: numeric={numeric}, analytic={}",
        conv_gradient.weights[0]
    );
}

#[test]
fn training_initialization_and_sgd_honor_hf_defaults_and_learning_rate() {
    let network = build_hf_network(
        "[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\n",
        "[hyperbolic_foveation]\nparams=learned\ninit_cx=0.1\ninit_cy=-0.2\ninit_radius=0.8\ninit_strength=0.25\nhf_lr_mult=0.25\nlearning_rate=0.5\ndownsample=2\n",
    );
    assert_eq!(network.layers[0].kind.name(), "hyperbolic_foveation");
    let mut first = ExecutableNetwork::new(network.clone());
    let mut second = ExecutableNetwork::new(network.clone());
    crate::initialize::initialize_training_params(&network, &mut first.params, 42);
    crate::initialize::initialize_training_params(&network, &mut second.params, 42);
    assert_eq!(first.params, second.params);
    let LayerParams::HyperbolicFoveation(params) = &first.params[0] else {
        panic!("expected hyperbolic-foveation parameters")
    };
    assert_eq!(params.pred_b1.len(), 64);
    assert_eq!(params.pred_w1.len(), 64 * 4);
    let expected_biases = crate::runtime::hyperbolic_foveation_predictor_initial_biases(
        match &network.layers[0].kind {
            LayerKind::HyperbolicFoveation(spec) => spec,
            _ => unreachable!(),
        },
    );
    assert_eq!(params.pred_b2, expected_biases);
}

#[test]
fn hf_weights_round_trip_in_source_array_order() {
    for (radius_mode, hidden, expected_lengths, expected_file_size) in [
        ("isotropic", 0, &[4][..], 36),
        ("anisotropic", 0, &[5][..], 40),
        ("isotropic", 2, &[4, 8, 2, 8, 4][..], 124),
        ("anisotropic", 2, &[5, 8, 2, 10, 5][..], 140),
    ] {
        let network = build_hf_network(
            "[net]\nwidth=4\nheight=3\nchannels=2\nbatch=1\n",
            &format!("[hf]\nradius_mode={radius_mode}\npred_hidden={hidden}\ndownsample=2\n"),
        );
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::HyperbolicFoveation(params) = &mut executable.params[0] else {
            panic!("expected hyperbolic-foveation parameters")
        };
        for (array, values) in [
            (&mut params.hf_params, 1.0),
            (&mut params.pred_w1, 10.0),
            (&mut params.pred_b1, 20.0),
            (&mut params.pred_w2, 30.0),
            (&mut params.pred_b2, 40.0),
        ] {
            for (index, value) in array.iter_mut().enumerate() {
                *value = values + index as f32;
            }
        }
        let original = executable.params.clone();
        let path = std::env::temp_dir().join(format!(
            "darknet-hf-weights-{}-{radius_mode}-{hidden}.weights",
            std::process::id()
        ));
        crate::weights::save_weights(&network, &executable.params, &path, 17).unwrap();
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|array| array.len)
                .collect::<Vec<_>>(),
            expected_lengths
        );
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(fs::metadata(&path).unwrap().len(), expected_file_size);
        let mut restored = ExecutableNetwork::new(network);
        restored.load_weights_file(&path).unwrap();
        assert_eq!(restored.params, original);
        let _ = fs::remove_file(path);
    }
}

#[test]
fn hf_sgd_adam_and_muon_update_and_persist_state() {
    let sgd_network = build_hf_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n",
        "[hf]\nhf_lr_mult=0.25\nlearning_rate=0.5\n",
    );
    let mut sgd = ExecutableNetwork::new(sgd_network);
    let LayerTrainBuffers::HyperbolicFoveation(train) = &mut sgd.train_buffers[0] else {
        panic!("expected hyperbolic-foveation train buffers")
    };
    train.hf_param_updates[0] = 1.0;
    let before = match &sgd.params[0] {
        LayerParams::HyperbolicFoveation(params) => params.hf_params[0],
        _ => unreachable!(),
    };
    sgd.update_weights(0.2, 0.0, 0.0, 1);
    let LayerParams::HyperbolicFoveation(params) = &sgd.params[0] else {
        unreachable!()
    };
    assert!((params.hf_params[0] - (before + 0.025)).abs() < 1.0e-6);

    let adam_network = build_hf_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nadam=1\n",
        "[hf]\npred_hidden=2\ndownsample=2\n",
    );
    let mut adam = ExecutableNetwork::new(adam_network.clone());
    let LayerTrainBuffers::HyperbolicFoveation(train) = &mut adam.train_buffers[0] else {
        panic!("expected hyperbolic-foveation train buffers")
    };
    train.pred_w1_updates.fill(0.2);
    train.pred_b1_updates.fill(0.1);
    train.pred_w2_updates.fill(-0.3);
    train.pred_b2_updates.fill(0.15);
    adam.update_weights_adam(0.01, 1).unwrap();
    assert!(adam.host_adam_state[0].is_some());
    let adam_path =
        std::env::temp_dir().join(format!("darknet-hf-adam-{}.opt", std::process::id()));
    adam.save_optimizer_state(&adam_path).unwrap();
    let mut adam_restored = ExecutableNetwork::new(adam_network);
    adam_restored.load_optimizer_state(&adam_path).unwrap();
    assert_eq!(adam_restored.host_adam_state, adam.host_adam_state);
    let mismatched_adam_network = build_hf_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nadam=1\n",
        "[hf]\npred_hidden=3\ndownsample=2\n",
    );
    let mut mismatched_adam = ExecutableNetwork::new(mismatched_adam_network);
    assert!(mismatched_adam.load_optimizer_state(&adam_path).is_err());
    let _ = fs::remove_file(adam_path);

    let muon_network = build_hf_network(
        "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\nmuon=1\ndecoupled_decay=1\ndecay=0.001\n",
        "[hf]\npred_hidden=2\ndownsample=2\n",
    );
    let mut muon = ExecutableNetwork::new(muon_network.clone());
    let LayerTrainBuffers::HyperbolicFoveation(train) = &mut muon.train_buffers[0] else {
        panic!("expected hyperbolic-foveation train buffers")
    };
    train.pred_w1_updates.fill(0.2);
    train.pred_w2_updates.fill(-0.15);
    train.pred_b1_updates.fill(0.1);
    train.pred_b2_updates.fill(0.05);
    muon.update_weights_muon(0.01, 1).unwrap();
    let matrices = &muon.muon_state[0].as_ref().unwrap().matrices;
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols))
            .collect::<Vec<_>>(),
        [(2, 1), (4, 2)]
    );
    let muon_path =
        std::env::temp_dir().join(format!("darknet-hf-muon-{}.opt", std::process::id()));
    muon.save_optimizer_state(&muon_path).unwrap();
    let mut muon_restored = ExecutableNetwork::new(muon_network);
    muon_restored.load_optimizer_state(&muon_path).unwrap();
    assert_eq!(muon_restored.muon_state, muon.muon_state);
    assert_eq!(muon_restored.host_adam_state, muon.host_adam_state);
    let _ = fs::remove_file(muon_path);
}

#[test]
fn tiny_box_and_strength_auxiliary_gradients_match_the_source_regularizer() {
    let network = build_hf_network(
        "[net]\nwidth=32\nheight=24\nchannels=1\nbatch=2\n",
        "[convolutional]\nfilters=6\nsize=1\nactivation=linear\n[hyperbolic_foveation]\nparams=fixed\ninit_cx=0.2\ninit_cy=-0.1\ninit_radius=0.7\ninit_strength=0.3\nsmall_box_px=10\nsmall_box_strength=0.85\ntiny_center_loss=2\ntiny_radius_loss=0.5\ntiny_strength_loss=3\nstrength_zero_loss=0.4\n[yolo]\nmask=0\nanchors=1,1\nclasses=1\nnum=1\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    let input = Tensor::from_data(network.net.input_shape(), vec![0.0; 32 * 24 * 2]).unwrap();
    let forward = executable.forward_training(&input).unwrap();
    let labels = vec![
        vec![
            crate::train::YoloLabel {
                class_id: 0,
                x: 0.25,
                y: 0.4,
                w: 0.1,
                h: 0.125,
            },
            crate::train::YoloLabel {
                class_id: 0,
                x: 0.75,
                y: 0.6,
                w: 0.15,
                h: 0.1,
            },
        ],
        Vec::new(),
    ];
    let original_labels = labels.clone();
    let seeds = vec![None; network.layers.len()];
    let backward = executable
        .backward_accumulate_seeded_with_hf_auxiliary_gradients(&input, &forward, &seeds, &labels)
        .unwrap();
    assert_eq!(labels, original_labels);
    let LayerParams::HyperbolicFoveation(gradient) = &backward.param_gradients[1] else {
        panic!("expected hyperbolic-foveation gradients")
    };

    let tiny = &labels[0];
    let weights = tiny
        .iter()
        .map(|label| 1.0 / (label.w * 32.0 * label.h * 24.0).max(1.0))
        .collect::<Vec<_>>();
    let weight_sum = weights.iter().sum::<f32>();
    let target_x = tiny
        .iter()
        .zip(&weights)
        .map(|(label, weight)| weight * (2.0 * label.x - 1.0))
        .sum::<f32>()
        / weight_sum;
    let target_y = tiny
        .iter()
        .zip(&weights)
        .map(|(label, weight)| weight * (2.0 * label.y - 1.0))
        .sum::<f32>()
        / weight_sum;
    let target_radius = tiny
        .iter()
        .map(|label| {
            let dx = 2.0 * label.x - 1.0 - target_x;
            let dy = 2.0 * label.y - 1.0 - target_y;
            (dx * dx + dy * dy).sqrt() + 2.0 * label.w.max(label.h) + 0.15
        })
        .fold(0.25f32, f32::max)
        .clamp(0.1, 2.0);
    assert!((gradient.hf_params[0] - 2.0 * (0.2 - target_x)).abs() < 1.0e-5);
    assert!((gradient.hf_params[1] - 2.0 * (-0.1 - target_y)).abs() < 1.0e-5);
    assert!((gradient.hf_params[2] - 0.5 * (0.7 - target_radius)).abs() < 1.0e-5);
    let expected_strength = 3.0 * (0.3 - 0.85) + 2.0 * 0.4;
    assert!((gradient.hf_params[3] - expected_strength).abs() < 1.0e-5);
    let LayerTrainBuffers::HyperbolicFoveation(train) = &executable.train_buffers[1] else {
        panic!("expected hyperbolic-foveation train buffers")
    };
    assert_eq!(train.hf_param_updates, gradient.hf_params);
}

#[test]
fn tiny_box_auxiliary_gradients_backpropagate_through_the_learned_predictor() {
    let network = build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=1\nbatch=1\n",
        "[convolutional]\nfilters=6\nsize=1\nactivation=linear\n[hyperbolic_foveation]\npred_hidden=2\ndownsample=2\nsmall_box_px=10\nsmall_box_strength=0.85\ntiny_center_loss=1.7\ntiny_radius_loss=0.3\ntiny_strength_loss=0.8\n[yolo]\nmask=0\nanchors=1,1\nclasses=1\nnum=1\n",
    );
    let mut executable = ExecutableNetwork::new(network.clone());
    if let LayerParams::Convolutional(params) = &mut executable.params[0] {
        params.weights.fill(0.12);
        params.biases.fill(0.2);
    }
    if let LayerParams::HyperbolicFoveation(params) = &mut executable.params[1] {
        params.pred_w1.fill(0.02);
        params.pred_b1.copy_from_slice(&[0.4, 0.6]);
        params
            .pred_w2
            .copy_from_slice(&[0.03, -0.02, 0.01, 0.04, 0.02, 0.01, -0.03, 0.05]);
        params.pred_b2.copy_from_slice(&[0.1, -0.05, 1.0, -0.6]);
    }
    let input = sample_input(network.net.input_shape());
    let label = crate::train::YoloLabel {
        class_id: 0,
        x: 0.3,
        y: 0.4,
        w: 0.2,
        h: 0.15,
    };
    let labels = vec![vec![label.clone()]];
    let forward = executable.forward_training(&input).unwrap();
    let mut seeds = vec![None; network.layers.len()];
    seeds[2] = Some(Tensor::zeros(network.layers[2].output_shape));
    let backward = executable
        .backward_accumulate_seeded_with_hf_auxiliary_gradients(&input, &forward, &seeds, &labels)
        .unwrap();
    let LayerParams::HyperbolicFoveation(gradient) = &backward.param_gradients[1] else {
        panic!("expected learned predictor gradients")
    };
    let LayerKind::HyperbolicFoveation(spec) = &network.layers[1].kind else {
        unreachable!()
    };
    let target_x = 2.0 * label.x - 1.0;
    let target_y = 2.0 * label.y - 1.0;
    let target_radius = (0.25 + 2.0 * label.w.max(label.h) + 0.15).clamp(0.1, 2.0);
    let auxiliary_loss = |executable: &ExecutableNetwork| {
        let forward = executable.forward(&input).unwrap();
        let params = forward.hyperbolic_foveation_caches[1]
            .as_ref()
            .unwrap()
            .params_per_batch
            .as_slice();
        0.5 * spec.tiny_center_loss
            * ((params[0] - target_x).powi(2) + (params[1] - target_y).powi(2))
            + 0.5 * spec.tiny_radius_loss * (params[2] - target_radius).powi(2)
            + 0.5 * spec.tiny_strength_loss * (params[3] - spec.small_box_strength).powi(2)
    };
    let base = match &executable.params[1] {
        LayerParams::HyperbolicFoveation(params) => params.clone(),
        _ => unreachable!(),
    };
    let epsilon = 1.0e-3;
    for (block, index, analytic) in [
        ("pred_w2", 0, gradient.pred_w2[0]),
        ("pred_b2", 0, gradient.pred_b2[0]),
    ] {
        let mut plus = base.clone();
        let mut minus = base.clone();
        match block {
            "pred_w2" => {
                plus.pred_w2[index] += epsilon;
                minus.pred_w2[index] -= epsilon;
            }
            "pred_b2" => {
                plus.pred_b2[index] += epsilon;
                minus.pred_b2[index] -= epsilon;
            }
            _ => unreachable!(),
        }
        executable.params[1] = LayerParams::HyperbolicFoveation(plus);
        let loss_plus = auxiliary_loss(&executable);
        executable.params[1] = LayerParams::HyperbolicFoveation(minus);
        let loss_minus = auxiliary_loss(&executable);
        let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert!(
            (numeric - analytic).abs() < 0.02,
            "{block}[{index}]: numeric={numeric}, analytic={analytic}"
        );
    }
}

#[test]
fn detection_decoding_inverts_hf_boxes_using_the_selected_batch_parameters() {
    let network = build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=6\nbatch=2\n",
        "[hyperbolic_foveation]\nparams=fixed\ninit_cx=0\ninit_cy=0\ninit_radius=1\ninit_strength=0.4\n[yolo]\nmask=0\nanchors=1,1\nclasses=1\nnum=1\n",
    );
    let executable = ExecutableNetwork::new(network.clone());
    let input = Tensor::from_data(network.net.input_shape(), vec![0.0; 5 * 4 * 6 * 2]).unwrap();
    let mut forward = executable.forward(&input).unwrap();
    let head = &mut forward.yolo_heads[0];
    let cells = head.output.shape.width * head.output.shape.height;
    let sample_len = head.output.shape.element_count();
    head.output.data.fill(0.0);
    for batch in 0..2 {
        let base = batch * sample_len;
        let cell = 1 * head.output.shape.width + 2;
        head.output.data[base + 4 * cells + cell] = 10.0;
        head.output.data[base + 5 * cells + cell] = 10.0;
    }
    let cache = forward.hyperbolic_foveation_caches[0]
        .as_mut()
        .expect("foveation cache exists");
    cache
        .params_per_batch
        .copy_from_slice(&[-0.25, 0.10, 0.80, 0.70, 0.25, -0.10, 1.10, 0.60]);
    let options = crate::detect::DetectionOptions {
        confidence_threshold: 0.9,
        nms_threshold: 0.45,
        letterbox: false,
    };
    let LayerKind::HyperbolicFoveation(spec) = &network.layers[0].kind else {
        unreachable!()
    };
    let params = [
        crate::hyperbolic_foveation::HfParams {
            center_x: -0.25,
            center_y: 0.10,
            radius_x: 0.80,
            radius_y: 0.80,
            strength: 0.70,
            alpha: spec.alpha,
            p_exp: spec.p_exp,
            radius_mode: spec.radius_mode,
        },
        crate::hyperbolic_foveation::HfParams {
            center_x: 0.25,
            center_y: -0.10,
            radius_x: 1.10,
            radius_y: 1.10,
            strength: 0.60,
            alpha: spec.alpha,
            p_exp: spec.p_exp,
            radius_mode: spec.radius_mode,
        },
    ];
    for batch in 0..2 {
        let detections =
            crate::detect::decode_network_detections(&network, &forward, batch, 100, 80, options)
                .unwrap();
        assert_eq!(detections.len(), 1);
        let expected = crate::hyperbolic_foveation::transform_box_with_jacobian(
            [0.5, 0.375, 0.2, 0.25],
            params[batch],
            true,
            spec.inverse_method,
            spec.nr_iters,
            spec.nr_eta,
        );
        let actual = &detections[0].bbox;
        for (got, want) in [
            (actual.x / 100.0, expected[0]),
            (actual.y / 80.0, expected[1]),
            (actual.w / 100.0, expected[2]),
            (actual.h / 80.0, expected[3]),
        ] {
            assert!((got - want).abs() < 1.0e-5, "got={got}, expected={want}");
        }
    }
}

#[test]
fn obb_network_decoding_inverts_boxes_and_front_points_with_hf_parameters() {
    let network = build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=6\nbatch=1\nobb=1\n",
        "[convolutional]\nfilters=8\nsize=1\nactivation=linear\n[hyperbolic_foveation]\nparams=fixed\ninit_cx=0.2\ninit_cy=-0.15\ninit_radius=0.85\ninit_strength=0.65\n[obb_yolo]\nmask=0\nanchors=1,1\nclasses=1\nnum=1\nnew_coords=1\n",
    );
    let executable = ExecutableNetwork::new(network.clone());
    let input = Tensor::from_data(network.net.input_shape(), vec![0.0; 5 * 4 * 6]).unwrap();
    let mut forward = executable.forward(&input).unwrap();
    let head = &mut forward.yolo_heads[0];
    let cells = head.output.shape.width * head.output.shape.height;
    let cell = head.output.shape.width + 2;
    head.output.data.fill(0.0);
    head.output.data[cell] = 0.5;
    head.output.data[cells + cell] = 0.5;
    head.output.data[2 * cells + cell] = 0.5;
    head.output.data[3 * cells + cell] = 0.5;
    head.output.data[4 * cells + cell] = 0.25;
    head.output.data[5 * cells + cell] = 0.2;
    head.output.data[6 * cells + cell] = 10.0;
    head.output.data[7 * cells + cell] = 10.0;

    let options = crate::detect::DetectionOptions {
        confidence_threshold: 0.9,
        nms_threshold: 0.45,
        letterbox: false,
    };
    let detections =
        crate::detect::decode_obb_network_detections(&network, &forward, 100, 80, options).unwrap();
    assert_eq!(detections.len(), 1);
    let LayerKind::HyperbolicFoveation(spec) = &network.layers[1].kind else {
        unreachable!()
    };
    let params = crate::hyperbolic_foveation::HfParams {
        center_x: 0.2,
        center_y: -0.15,
        radius_x: 0.85,
        radius_y: 0.85,
        strength: 0.65,
        alpha: spec.alpha,
        p_exp: spec.p_exp,
        radius_mode: spec.radius_mode,
    };
    let expected_box = crate::hyperbolic_foveation::transform_box_with_jacobian(
        [0.5, 0.375, 0.2, 0.25],
        params,
        true,
        spec.inverse_method,
        spec.nr_iters,
        spec.nr_eta,
    );
    let bbox = detections[0].detection.bbox;
    for (actual, expected) in [
        (bbox.x / 100.0, expected_box[0]),
        (bbox.y / 80.0, expected_box[1]),
        (bbox.w / 100.0, expected_box[2]),
        (bbox.h / 80.0, expected_box[3]),
    ] {
        assert!((actual - expected).abs() < 1.0e-5);
    }
    let front = match detections[0].geometry {
        crate::ObbDetectionGeometry::FrontPoint { x, y } => [x, y],
        crate::ObbDetectionGeometry::CarpnQuad(_) => panic!("expected front-point OBB geometry"),
    };
    let front_in = [
        2.0 * (2.0 + 0.25) / 5.0 - 1.0,
        2.0 * (1.0 + 0.2) / 4.0 - 1.0,
    ];
    let front_expected = crate::hyperbolic_foveation::inverse_point(
        front_in,
        params,
        spec.inverse_method,
        spec.nr_iters,
        spec.nr_eta,
    )
    .point;
    assert!((front[0] / 100.0 - 0.5 * (front_expected[0] + 1.0)).abs() < 1.0e-5);
    assert!((front[1] / 80.0 - 0.5 * (front_expected[1] + 1.0)).abs() < 1.0e-5);
}

#[test]
fn map_validation_uses_inverse_foveated_boxes_in_normalized_coordinates() {
    let network = build_hf_network(
        "[net]\nwidth=5\nheight=4\nchannels=3\nbatch=1\n",
        "[convolutional]\nfilters=6\nsize=1\nactivation=linear\n[hyperbolic_foveation]\nparams=fixed\ninit_cx=0\ninit_cy=0\ninit_radius=1\ninit_strength=0.4\n[yolo]\nmask=0\nanchors=1,1\nclasses=1\nnum=1\n",
    );
    let executable = ExecutableNetwork::new(network.clone());
    let LayerKind::HyperbolicFoveation(spec) = &network.layers[1].kind else {
        unreachable!()
    };
    let params = crate::hyperbolic_foveation::HfParams {
        center_x: 0.2,
        center_y: -0.15,
        radius_x: 0.85,
        radius_y: 0.85,
        strength: 0.65,
        alpha: spec.alpha,
        p_exp: spec.p_exp,
        radius_mode: spec.radius_mode,
    };
    let expected = crate::hyperbolic_foveation::transform_box_with_jacobian(
        [0.5, 0.375, 0.2, 0.25],
        params,
        true,
        spec.inverse_method,
        spec.nr_iters,
        spec.nr_eta,
    );
    let image_path =
        std::env::temp_dir().join(format!("darknet-hf-map-{}.png", std::process::id()));
    image::RgbImage::from_pixel(100, 80, image::Rgb([0, 0, 0]))
        .save(&image_path)
        .unwrap();
    let dataset = crate::evaluate::ValidationDataset {
        classes: 1,
        images: vec![(
            image_path.clone(),
            vec![crate::train::YoloLabel {
                class_id: 0,
                x: expected[0],
                y: expected[1],
                w: expected[2],
                h: expected[3],
            }],
        )],
    };
    let report = dataset
        .evaluate_with_forward(
            &network,
            crate::detect::DetectionOptions {
                confidence_threshold: 0.9,
                nms_threshold: 0.45,
                letterbox: true,
            },
            0.5,
            |input| {
                let mut forward = executable.forward(input)?;
                let head = &mut forward.yolo_heads[0];
                let cells = head.output.shape.width * head.output.shape.height;
                head.output.data.fill(0.0);
                let cell = head.output.shape.width + 2;
                head.output.data[4 * cells + cell] = 10.0;
                head.output.data[5 * cells + cell] = 10.0;
                forward.hyperbolic_foveation_caches[1]
                    .as_mut()
                    .unwrap()
                    .params_per_batch
                    .copy_from_slice(&[0.2, -0.15, 0.85, 0.65]);
                Ok(forward)
            },
        )
        .unwrap();
    assert_eq!(report.mean_average_precision, 1.0);
    assert_eq!(report.detection_counts, [1]);
    let _ = fs::remove_file(image_path);
}

#[test]
fn hf_sgd_update_obeys_dont_update_burnin_and_train_only_bn_gates() {
    for (net_options, layer_options, iteration, should_update) in [
        ("", "dont_update=1\n", 0, false),
        ("burn_in=10\n", "burnin_update=2\n", 0, false),
        ("", "train_only_bn=1\n", 2, false),
        ("", "dont_update=1\n", 1, true),
    ] {
        let network = build_hf_network(
            &format!("[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n{net_options}"),
            &format!("[hf]\ninit_strength=0.3\n{layer_options}"),
        );
        let mut executable = ExecutableNetwork::new(network);
        if let LayerTrainBuffers::HyperbolicFoveation(train) = &mut executable.train_buffers[0] {
            train.hf_param_updates[0] = 1.0;
        } else {
            panic!("expected hyperbolic-foveation train buffers");
        }
        executable.set_train_iteration(iteration);
        let before = match &executable.params[0] {
            LayerParams::HyperbolicFoveation(params) => params.hf_params[0],
            _ => unreachable!(),
        };
        executable.update_weights(0.1, 0.0, 0.0, 1);
        let after = match &executable.params[0] {
            LayerParams::HyperbolicFoveation(params) => params.hf_params[0],
            _ => unreachable!(),
        };
        assert_eq!(after != before, should_update);
    }
}

#[test]
fn gpu_forward_is_supported_while_compact_and_updates_reject_hf() {
    let network = seeded_hf_network("isotropic");
    let input = sample_input(network.net.input_shape());
    let expected = ExecutableNetwork::new(network.clone())
        .forward(&input)
        .unwrap();
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let actual = gpu.forward(&input).expect("HF GPU-reference forward");
    assert_eq!(actual.layer_outputs, expected.layer_outputs);
    assert!(
        gpu.forward_compact(&input)
            .unwrap_err()
            .to_string()
            .contains("[hyperbolic_foveation] GPU compact inference is not implemented")
    );
    let training_forward = gpu.forward_training_with_state(&input).unwrap();
    let seed = Tensor::zeros(gpu.network.layers[0].output_shape);
    gpu.zero_train_buffers_gpu().unwrap();
    gpu.backward_accumulate_seeded_gpu(&training_forward, &[Some(seed)])
        .expect("HF host-fallback backward with native warp adjoints");
    gpu.update_weights_gpu(0.01, 0.9, 0.0, 1)
        .expect("HF host-fallback optimizer update");
}
