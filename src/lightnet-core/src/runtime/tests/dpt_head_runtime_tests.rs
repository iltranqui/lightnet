use super::*;
use crate::depth_anything::DepthAnythingWeights;
use crate::runtime::{GradMode, reject_gpu_hyperbolic_foveation};
use crate::weights::{WeightsArrayInspection, inspect_weights_file, save_weights};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn dpt_config(section: &str, batch: usize, net_options: &str, layer_options: &str) -> String {
    format!(
        "[net]\nwidth=3\nheight=2\nchannels=4\nbatch={batch}\n{net_options}\n\
         [{section}]\nembed_dim=1\nfusion=2\nhead_hidden=1\npatch_size=1\ntaps=4\n\
         neck=1,1,1,1\nreassemble=4,2,1,0\n{layer_options}\n"
    )
}

fn tiny_dpt_config(section: &str) -> String {
    dpt_config(section, 2, "", "")
}

fn make_executable(
    section: &str,
    batch: usize,
    net_options: &str,
    layer_options: &str,
) -> ExecutableNetwork {
    let config = ConfigFile::parse_str(
        "dpt-head-runtime.cfg",
        &dpt_config(section, batch, net_options, layer_options),
    )
    .expect("DPT cfg parses");
    ExecutableNetwork::new(Network::from_config(config).expect("DPT network builds"))
}

fn fill_dpt_weights(executable: &mut ExecutableNetwork) {
    let weights = executable
        .params
        .iter_mut()
        .find_map(|params| match params {
            LayerParams::DptHead(weights) => Some(weights),
            _ => None,
        })
        .expect("expected DPT parameters");
    for (index, value) in weights.values.iter_mut().enumerate() {
        *value = 0.25 + (index % 13) as f32 * 0.01;
    }
}

fn dpt_input(network: &Network) -> Tensor {
    let shape = network.net.input_shape();
    let data = (0..shape.element_count() * shape.batch)
        .map(|index| 0.1 + index as f32 * 0.01)
        .collect();
    Tensor::from_data(shape, data).expect("input tensor")
}

fn dpt_output_seed(network: &Network) -> Tensor {
    let shape = network.layers.last().expect("DPT layer").output_shape;
    Tensor::from_data(shape, vec![0.7; shape.element_count() * shape.batch]).expect("output seed")
}

fn temp_weights_path(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "darknet-dpt-{label}-{}-{nonce}.weights",
        std::process::id()
    ))
}

#[test]
fn dpt_head_config_aliases_shape_and_weight_metadata() {
    for alias in ["dpt", "dpt_head", "depth_anything_head", "neck"] {
        let config = ConfigFile::parse_str("dpt-head-test.cfg", &tiny_dpt_config(alias))
            .expect("DPT cfg parses");
        let network = Network::from_config(config).expect("DPT network builds");
        assert_eq!(network.layers.len(), 1, "alias [{alias}]");
        assert_eq!(network.layers[0].kind.name(), "dpt_head");
        assert_eq!(network.layers[0].input_shape, TensorShape::new(3, 2, 4, 2));
        assert_eq!(network.layers[0].output_shape, TensorShape::new(3, 2, 1, 2));
        let weights = network.layers[0].weights().expect("DPT has weights");
        assert_eq!(weights.entries.len(), 66, "alias [{alias}]");
        assert_eq!(
            weights.entries.iter().map(|entry| entry.len).sum::<usize>(),
            775
        );
        let expected_lengths = vec![1; 8]
            .into_iter()
            .chain([16, 1, 4, 1, 0, 0, 9, 1])
            .chain(vec![18; 4])
            .chain([4, 2, 4, 2, 4, 2, 4, 2])
            .chain((0..32).map(|index| if index % 2 == 0 { 36 } else { 2 }))
            .chain([18, 1, 9, 1, 1, 1])
            .collect::<Vec<_>>();
        assert_eq!(
            weights
                .entries
                .iter()
                .map(|entry| entry.len)
                .collect::<Vec<_>>(),
            expected_lengths
        );
        assert!(
            weights
                .entries
                .iter()
                .enumerate()
                .all(|(index, entry)| entry.array.name() == format!("dpt_tensor_{index:02}"))
        );
        assert_eq!(weights.entries[12].len, 0);
        assert_eq!(weights.entries[13].len, 0);
    }

    let defaults = Network::from_config(
        ConfigFile::parse_str(
            "dpt-head-defaults.cfg",
            "[net]\nwidth=2\nheight=3\nchannels=1536\nbatch=1\n[dpt_head]\n",
        )
        .unwrap(),
    )
    .unwrap();
    let LayerKind::DptHead(spec) = &defaults.layers[0].kind else {
        panic!("DPT defaults layer");
    };
    assert_eq!(
        (
            spec.embed_dim,
            spec.fusion,
            spec.head_hidden,
            spec.patch_size,
            spec.taps
        ),
        (384, 64, 32, 14, 4)
    );
    assert_eq!(spec.neck, [48, 96, 192, 384]);
    assert_eq!(spec.reassemble, [4, 2, 1, 0]);
    assert_eq!(
        defaults.layers[0].output_shape,
        TensorShape::new(28, 42, 1, 1)
    );
    assert_eq!(
        defaults.layers[0]
            .weights()
            .unwrap()
            .entries
            .iter()
            .map(|entry| entry.len)
            .sum::<usize>(),
        2_728_513
    );
}

#[test]
fn dpt_head_construction_and_seeded_training_initialization_are_zeroed() {
    let network = Network::from_config(
        ConfigFile::parse_str("dpt-head-init.cfg", &tiny_dpt_config("dpt_head")).unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        panic!("expected DPT weights");
    };
    assert!(weights.values.iter().all(|value| *value == 0.0));
    crate::initialize::initialize_training_params(&network, &mut executable.params, 41);
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        unreachable!()
    };
    assert!(weights.values.iter().all(|value| *value == 0.0));
}

#[test]
fn dpt_head_rejects_unknown_invalid_and_gpu_stream_config() {
    for (options, reason) in [
        ("mystery=1", "unknown key"),
        ("stream=0", "stream"),
        ("wait_stream=1", "wait_stream"),
        ("taps=3", "four taps"),
        ("fusion=3", "even"),
        ("neck=1,1,1", "four values"),
        ("reassemble=4,2,2,0", "reassemble"),
        ("embed_dim=0", "positive"),
    ] {
        let config = ConfigFile::parse_str(
            "dpt-head-invalid.cfg",
            &dpt_config("dpt_head", 2, "", options),
        )
        .expect("cfg parser accepts raw key/value syntax");
        let error = Network::from_config(config).expect_err(reason).to_string();
        assert!(error.contains("[dpt_head]"), "{reason}: {error}");
    }
}

#[test]
fn dpt_head_weights_round_trip_exact_66_slot_order_and_file_length() {
    let network = Network::from_config(
        ConfigFile::parse_str("dpt-head-weights.cfg", &tiny_dpt_config("dpt_head")).unwrap(),
    )
    .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DptHead(weights) = &mut executable.params[0] else {
        panic!("expected DPT parameters");
    };
    for (index, value) in weights.values.iter_mut().enumerate() {
        *value = index as f32 + 0.25;
    }
    let parameter_count = weights.values.len();
    assert_eq!(parameter_count, 775);

    let path = temp_weights_path("source-order");
    save_weights(&network, &executable.params, &path, 23).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        20 + 4 * parameter_count as u64
    );
    let inspection = inspect_weights_file(&network, &path).unwrap();
    assert_eq!(inspection.trailing_bytes, 0);
    let arrays = &inspection.layers[0].arrays;
    assert_eq!(arrays.len(), 66);
    let mut offset = 20u64;
    let bytes = fs::read(&path).unwrap();
    for (slot, array) in arrays.iter().enumerate() {
        assert_eq!(array.name, format!("dpt_tensor_{slot:02}"));
        assert_eq!(array.byte_offset, offset);
        assert_eq!(array.byte_len, array.len as u64 * 4);
        if array.len != 0 {
            let first = f32::from_le_bytes(
                bytes[array.byte_offset as usize..array.byte_offset as usize + 4]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(first, offset_to_first_value(arrays, slot));
        }
        offset += array.byte_len;
    }
    let mut restored = ExecutableNetwork::new(network);
    restored.load_weights_file(&path).unwrap();
    assert_eq!(restored.params, executable.params);
    let _ = fs::remove_file(path);
}

fn offset_to_first_value(arrays: &[WeightsArrayInspection], slot: usize) -> f32 {
    let first_nonempty = arrays[..slot].iter().map(|array| array.len).sum::<usize>();
    first_nonempty as f32 + 0.25
}

#[test]
fn dpt_head_constant_bias_matches_hand_computed_nchw_reference() {
    let mut executable = make_executable("dpt_head", 2, "", "");
    let LayerParams::DptHead(weights) = &mut executable.params[0] else {
        panic!("expected DPT weights");
    };
    weights.values.fill(0.0);
    let layout = match &executable.network.layers[0].kind {
        LayerKind::DptHead(spec) => crate::dpt_head::dpt_head_weight_layout(
            spec.geometry(executable.network.layers[0].input_shape),
        )
        .unwrap(),
        _ => unreachable!(),
    };
    let bias = layout.get(65);
    weights.values[bias.start] = 1.25;
    let output = executable
        .forward(&dpt_input(&executable.network))
        .unwrap()
        .final_output()
        .unwrap()
        .clone();
    assert_eq!(output.shape, TensorShape::new(3, 2, 1, 2));
    assert_eq!(output.data, vec![1.25; 12]);
}

#[test]
fn dpt_head_axis_layout_batched_nchw_forward_matches_independent_non_square_samples() {
    let axis_config = |batch| {
        dpt_config("dpt_head", batch, "", "")
            .replace("channels=4", "channels=8")
            .replace("embed_dim=1", "embed_dim=2")
    };
    let mut batched = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("dpt-head-axis.cfg", &axis_config(2)).unwrap())
            .unwrap(),
    );
    fill_dpt_weights(&mut batched);
    let input = dpt_input(&batched.network);
    let batched_output = batched
        .forward(&input)
        .unwrap()
        .final_output()
        .unwrap()
        .clone();
    assert_eq!(batched_output.shape, TensorShape::new(3, 2, 1, 2));
    assert!(batched_output.data.iter().all(|value| value.is_finite()));

    let mut single = ExecutableNetwork::new(
        Network::from_config(
            ConfigFile::parse_str("dpt-head-axis-single.cfg", &axis_config(1)).unwrap(),
        )
        .unwrap(),
    );
    single.params[0] = batched.params[0].clone();
    let sample_len = input.data.len() / 2;
    for batch in 0..2 {
        let sample = Tensor::from_data(
            single.network.net.input_shape(),
            input.data[batch * sample_len..(batch + 1) * sample_len].to_vec(),
        )
        .unwrap();
        let output = single
            .forward(&sample)
            .unwrap()
            .final_output()
            .unwrap()
            .clone();
        let output_len = output.data.len();
        assert_eq!(
            &batched_output.data[batch * output_len..(batch + 1) * output_len],
            output.data.as_slice(),
            "batch {batch} must be independent and NCHW contiguous"
        );
    }
    assert_ne!(&batched_output.data[..6], &batched_output.data[6..]);
}

#[test]
fn dpt_head_full_runtime_backward_accumulates_negative_updates_and_sgd_applies_them() {
    let mut executable = make_executable("dpt_head", 2, "", "");
    fill_dpt_weights(&mut executable);
    let input = dpt_input(&executable.network);
    let forward = executable.forward(&input).unwrap();
    let seed = dpt_output_seed(&executable.network);
    let backward = executable
        .backward_seeded_with_grad_mode(&input, &forward, &[Some(seed)], GradMode::Analytical)
        .unwrap();
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .any(|value| value.abs() > 1.0e-9)
    );
    let LayerParams::DptHead(gradients) = &backward.param_gradients[0] else {
        panic!("expected DPT gradients");
    };
    assert!(gradients.values.iter().any(|value| value.abs() > 1.0e-9));
    let before = match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => panic!("expected DPT weights"),
    };
    executable
        .accumulate_train_buffers(&backward.param_gradients)
        .unwrap();
    let LayerTrainBuffers::DptHead(train) = &executable.train_buffers[0] else {
        panic!("expected DPT train buffers");
    };
    assert!(train.has_gradient);
    assert!(train.weight_updates.iter().any(|value| *value < 0.0));
    executable.update_weights(0.01, 0.0, 0.0, 1);
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        panic!("expected DPT weights");
    };
    assert!(
        weights
            .values
            .iter()
            .zip(&before)
            .any(|(after, before)| after != before)
    );
    let layout =
        crate::dpt_head::dpt_head_weight_layout(match &executable.network.layers[0].kind {
            LayerKind::DptHead(spec) => spec.geometry(executable.network.layers[0].input_shape),
            _ => unreachable!(),
        })
        .unwrap();
    for slot in [28, 29, 36, 37] {
        let range = layout.get(slot);
        assert_eq!(
            &weights.values[range.start..range.start + range.len],
            &before[range.start..range.start + range.len],
            "dormant residual slot {slot} must not be updated"
        );
    }
}

#[test]
fn dpt_head_adamw_update_and_optimizer_state_round_trip() {
    let net_options = "adam=1\ndecoupled_decay=1\ndecay=0.02";
    let mut executable = make_executable("dpt_head", 2, net_options, "");
    fill_dpt_weights(&mut executable);
    let input = dpt_input(&executable.network);
    let forward = executable.forward(&input).unwrap();
    let backward = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &[Some(dpt_output_seed(&executable.network))],
            GradMode::Analytical,
        )
        .unwrap();
    let before = match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => unreachable!(),
    };
    executable
        .accumulate_train_buffers(&backward.param_gradients)
        .unwrap();
    executable.update_weights_adam(0.01, 1).unwrap();
    assert!(match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.iter().zip(&before).any(|(a, b)| a != b),
        _ => false,
    });

    let path = temp_weights_path("adam-state");
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(executable.network.clone());
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.adam_step, executable.adam_step);
    assert_eq!(restored.host_adam_state, executable.host_adam_state);
    let _ = fs::remove_file(path);
}

#[test]
fn dpt_head_muon_uses_stable_slot_views_and_round_trips_optimizer_state() {
    let mut executable =
        make_executable("dpt_head", 1, "muon=1\ndecoupled_decay=1\ndecay=0.01", "");
    fill_dpt_weights(&mut executable);
    let layer = executable.network.layers[0].clone();
    let shapes = crate::muon_runtime::matrix_shapes(
        &layer.kind,
        layer.input_shape,
        layer.output_shape,
        &mut executable.params[0],
        &mut executable.train_buffers[0],
    )
    .unwrap();
    let expected = vec![(1, 1); 4]
        .into_iter()
        .chain([(1, 16), (1, 4), (1, 9)])
        .chain(vec![(2, 9); 4])
        .chain(vec![(2, 2); 4])
        .chain(vec![(2, 18); 14])
        .chain([(1, 18), (1, 9), (1, 1)])
        .collect::<Vec<_>>();
    assert_eq!(shapes, expected);

    let LayerParams::DptHead(weights) = &mut executable.params[0] else {
        panic!("expected DPT weights");
    };
    let layout = crate::dpt_head::dpt_head_weight_layout(match &layer.kind {
        LayerKind::DptHead(spec) => spec.geometry(layer.input_shape),
        _ => unreachable!(),
    })
    .unwrap();
    for &slot in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
        let range = layout.get(slot);
        weights.values[range.start..range.start + range.len].fill(slot as f32 + 1.0);
    }
    let mut visit_order = Vec::new();
    crate::muon_runtime::visit_muon_matrices(
        &layer.kind,
        layer.input_shape,
        layer.output_shape,
        &mut executable.params[0],
        &mut executable.train_buffers[0],
        &mut |values, _, _, _, _| {
            visit_order.push(values[0]);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        visit_order,
        crate::dpt_head::DPT_MUON_WEIGHT_SLOTS
            .iter()
            .map(|slot| *slot as f32 + 1.0)
            .collect::<Vec<_>>()
    );

    fill_dpt_weights(&mut executable);
    let input = dpt_input(&executable.network);
    let forward = executable.forward(&input).unwrap();
    let backward = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &[Some(dpt_output_seed(&executable.network))],
            GradMode::Analytical,
        )
        .unwrap();
    let before = match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => unreachable!(),
    };
    executable
        .accumulate_train_buffers(&backward.param_gradients)
        .unwrap();
    executable.update_weights_muon(0.001, 1).unwrap();
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        unreachable!()
    };
    assert!(
        weights
            .values
            .iter()
            .zip(&before)
            .any(|(after, before)| after != before)
    );
    let LayerTrainBuffers::DptHead(train) = &executable.train_buffers[0] else {
        unreachable!()
    };
    assert!(!train.has_gradient);

    let path = temp_weights_path("muon-state");
    executable.save_optimizer_state(&path).unwrap();
    let mut restored = ExecutableNetwork::new(executable.network.clone());
    restored.load_optimizer_state(&path).unwrap();
    assert_eq!(restored.muon_state, executable.muon_state);
    assert_eq!(restored.host_adam_state, executable.host_adam_state);
    let _ = fs::remove_file(path);
}

#[test]
fn dpt_head_update_gate_preserves_weights_and_does_not_advance_muon_state() {
    let mut executable = make_executable(
        "dpt_head",
        1,
        "muon=1\ndecoupled_decay=1\ndecay=0.01",
        "dont_update=1",
    );
    fill_dpt_weights(&mut executable);
    let input = dpt_input(&executable.network);
    let forward = executable.forward(&input).unwrap();
    let backward = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &[Some(dpt_output_seed(&executable.network))],
            GradMode::Analytical,
        )
        .unwrap();
    executable
        .accumulate_train_buffers(&backward.param_gradients)
        .unwrap();
    let before = match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => unreachable!(),
    };
    assert!(!executable.network.layers[0].train.updates_at(1, 0));
    executable.update_weights_muon(0.001, 1).unwrap();
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        unreachable!()
    };
    assert_eq!(weights.values, before);
    assert!(executable.muon_state[0].is_none());
    assert!(executable.host_adam_state[0].is_none());
}

#[test]
fn dpt_head_muon_applies_adamw_decay_without_matrix_adam_moments() {
    let learning_rate = 0.01;
    let decay = 0.1;
    let mut executable = make_executable("dpt_head", 1, "muon=1\ndecoupled_decay=1\ndecay=0.1", "");
    let LayerParams::DptHead(weights) = &mut executable.params[0] else {
        unreachable!()
    };
    weights.values.fill(0.5);
    let mut gradients = executable.params.clone();
    let LayerParams::DptHead(gradient) = &mut gradients[0] else {
        unreachable!()
    };
    gradient.values.fill(0.0);
    let layout =
        crate::dpt_head::dpt_head_weight_layout(match &executable.network.layers[0].kind {
            LayerKind::DptHead(spec) => spec.geometry(executable.network.layers[0].input_shape),
            _ => unreachable!(),
        })
        .unwrap();
    gradient.values[layout.get(65).start] = 1.0;
    executable.accumulate_train_buffers(&gradients).unwrap();
    executable.update_weights_muon(learning_rate, 1).unwrap();

    let expected_weight = 0.5 * (1.0 - learning_rate * decay);
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        unreachable!()
    };
    for &slot in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
        let range = layout.get(slot);
        for weight in &weights.values[range.start..range.start + range.len] {
            assert!(
                (weight - expected_weight).abs() < 1.0e-6,
                "matrix slot {slot}"
            );
        }
    }
    let host_state = executable.host_adam_state[0].as_ref().unwrap();
    for &slot in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
        assert!(host_state.children[slot].weights.is_none());
    }
    assert!(host_state.children[65].biases.is_some());
}

#[test]
fn dpt_head_cpp_mode_is_a_noop_including_adamw_decay() {
    let mut executable = make_executable("dpt_head", 2, "adam=1\ndecoupled_decay=1\ndecay=0.5", "");
    fill_dpt_weights(&mut executable);
    let input = dpt_input(&executable.network);
    let forward = executable.forward(&input).unwrap();
    let backward = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &[Some(dpt_output_seed(&executable.network))],
            GradMode::Cpp,
        )
        .unwrap();
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .all(|value| *value == 0.0)
    );
    let LayerParams::DptHead(gradients) = &backward.param_gradients[0] else {
        panic!("expected DPT gradient slot");
    };
    assert!(gradients.values.iter().all(|value| *value == 0.0));
    let before = match &executable.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => unreachable!(),
    };
    executable
        .accumulate_train_buffers(&backward.param_gradients)
        .unwrap();
    let LayerTrainBuffers::DptHead(train) = &executable.train_buffers[0] else {
        unreachable!()
    };
    assert!(!train.has_gradient);
    executable.update_weights_adam(0.1, 1).unwrap();
    let LayerParams::DptHead(weights) = &executable.params[0] else {
        unreachable!()
    };
    assert_eq!(weights.values, before);
    assert!(executable.host_adam_state[0].is_none());

    let mut sgd = make_executable("dpt_head", 2, "", "");
    fill_dpt_weights(&mut sgd);
    let sgd_input = dpt_input(&sgd.network);
    let sgd_forward = sgd.forward(&sgd_input).unwrap();
    let sgd_backward = sgd
        .backward_seeded_with_grad_mode(
            &sgd_input,
            &sgd_forward,
            &[Some(dpt_output_seed(&sgd.network))],
            GradMode::Cpp,
        )
        .unwrap();
    let sgd_before = match &sgd.params[0] {
        LayerParams::DptHead(weights) => weights.values.clone(),
        _ => unreachable!(),
    };
    sgd.accumulate_train_buffers(&sgd_backward.param_gradients)
        .unwrap();
    sgd.update_weights(0.1, 0.0, 0.5, 1);
    let LayerParams::DptHead(weights) = &sgd.params[0] else {
        unreachable!()
    };
    assert_eq!(weights.values, sgd_before);
}

#[test]
fn dpt_head_to_depth_anything_chain_receives_analytical_input_gradient() {
    let config = "[net]\nwidth=3\nheight=2\nchannels=3\nbatch=1\n\
                  [depth_anything]\npatch_size=1\ndim=1\ndepth=1\nheads=1\nmlp_dim=2\n\
                  pos_grid_h=2\npos_grid_w=3\ntaps=0,0,0,0\napply_imagenet_norm=0\nfreeze=1\n\
                  [dpt_head]\nembed_dim=1\nfusion=2\nhead_hidden=1\npatch_size=1\n\
                  neck=1,1,1,1\nreassemble=4,2,1,0\n";
    let network =
        Network::from_config(ConfigFile::parse_str("dpt-da-chain.cfg", config).unwrap()).unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DepthAnything(weights) = &mut executable.params[0] else {
        panic!("expected frozen encoder weights");
    };
    let geometry = match &network.layers[0].kind {
        LayerKind::DepthAnything(spec) => spec.geometry(network.layers[0].input_shape),
        _ => unreachable!(),
    };
    let mut blob = weights.to_blob();
    for (index, value) in blob.iter_mut().enumerate() {
        *value = ((index % 17) as f32 * 0.23).sin() * 0.1;
    }
    *weights = DepthAnythingWeights::from_blob(&blob, &geometry).unwrap();
    fill_dpt_weights(&mut executable);

    let input = Tensor::from_data(
        network.net.input_shape(),
        (0..network.net.input_shape().element_count())
            .map(|index| 0.2 + index as f32 * 0.03)
            .collect(),
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    let seed = Tensor::from_data(
        network.layers[1].output_shape,
        vec![0.5; network.layers[1].output_shape.element_count()],
    )
    .unwrap();
    let backward = executable
        .backward_seeded_with_grad_mode(&input, &forward, &[None, Some(seed)], GradMode::Analytical)
        .unwrap();
    assert!(
        backward.layer_output_gradients[0]
            .data
            .iter()
            .any(|value| value.abs() > 1.0e-9)
    );
}

#[test]
fn dpt_head_native_f32_dispatch_and_unsupported_half_boundary() {
    let network = Network::from_config(
        ConfigFile::parse_str("dpt-head-gpu.cfg", &tiny_dpt_config("dpt_head")).unwrap(),
    )
    .unwrap();
    let input = Tensor::zeros(network.net.input_shape());
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    assert!(!gpu.requires_cpu_backward_fallback());
    gpu.init_train_buffers_gpu().unwrap();
    let state = gpu.forward_training_device(&input).unwrap();
    let shape = network.layers[0].output_shape;
    let seeds = [Some(Tensor::from_data(shape, vec![1.0;shape.batch*shape.element_count()]).unwrap())];
    gpu.backward_accumulate_seeded_gpu(&state, &seeds).unwrap();
    gpu.update_weights_gpu(0.01, 0.0, 0.0, 1).unwrap();
    assert!(gpu.forward_compact(&input).unwrap().final_output.is_some());
    let error = reject_gpu_hyperbolic_foveation(&network, "f16 inference")
        .expect_err("DPT f16 GPU execution must fail");
    assert!(error.to_string().contains("Stage-1 CPU support"));

}

#[test]
fn dpt_head_resident_optimizers_preserve_inactive_slots_and_moments() {
    let inactive = [28usize, 29, 36, 37];
    for (optimizer, use_adam, use_muon) in [
        ("", false, false), ("adam=1\n", true, false),
        ("muon=1\ndecoupled_decay=1\n", false, true),
    ] {
        let network = Network::from_config(ConfigFile::parse_str(
            "dpt-resident-slots.cfg", &dpt_config("dpt_head", 2, optimizer, ""),
        ).unwrap()).unwrap();
        let input = dpt_input(&network);
        let geometry = match &network.layers[0].kind {
            LayerKind::DptHead(spec) => spec.geometry(network.layers[0].input_shape),
            _ => unreachable!(),
        };
        let layout = crate::dpt_head::dpt_head_weight_layout(geometry).unwrap();
        let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
        gpu.init_train_buffers_gpu().unwrap();
        let inactive_values = {
            let LayerParams::DptHead(params) = &mut gpu.params[0] else { panic!("DPT params") };
            for (index, value) in params.values.iter_mut().enumerate() {
                *value = 0.25 + (index % 13) as f32 * 0.01;
            }
            inactive.iter().map(|&slot| {
                let range = layout.get(slot);
                (range.start..range.start+range.len).map(|i| params.values[i]).collect::<Vec<_>>()
            }).collect::<Vec<_>>()
        };
        let mut seeded = vec![0.0; layout.total];
        for (n, &slot) in inactive.iter().enumerate() {
            let range = layout.get(slot);
            for value in &mut seeded[range.start..range.start+range.len] { *value = 0.03 + n as f32 * 0.01; }
        }
        let LayerTrainBuffers::DptHead(train) = &mut gpu.cpu_fallback_train_buffers.get_mut()[0] else {
            panic!("DPT train buffers")
        };
        train.weight_updates.copy_from_slice(&seeded);
        let forward = gpu.forward_training_device(&input).unwrap();
        gpu.update_weights_gpu(0.01, 0.9, 0.02, 2).unwrap();
        let LayerParams::DptHead(params) = &gpu.params[0] else { panic!("DPT params") };
        let host_buffers = gpu.host_fallback_train_buffers();
        let LayerTrainBuffers::DptHead(train) = &host_buffers[0] else {
            panic!("DPT train buffers")
        };
        for (n, &slot) in inactive.iter().enumerate() {
            let range = layout.get(slot);
            assert_eq!(&params.values[range.start..range.start+range.len], inactive_values[n]);
            assert_eq!(&train.weight_updates[range.start..range.start+range.len],
                &seeded[range.start..range.start+range.len]);
        }
        if use_adam || use_muon {
            let host_state = gpu.host_adam_state.borrow();
            let state = host_state[0].as_ref().expect("DPT Adam state");
            for &slot in &inactive {
                let child = &state.children[slot];
                assert!(child.weights.is_none() && child.biases.is_none(),
                    "inactive DPT slot {slot} must not acquire Adam moments");
            }
            assert_eq!(state.children[0].weights.is_some(), use_adam,
                "DPT matrix Adam moments must follow the selected optimizer policy");
            assert!(state.children[1].biases.is_some());
            let expected = state.clone();
            drop(host_state);
            if use_adam || use_muon {
                let path = temp_weights_path("resident-adam-muon");
                gpu.save_optimizer_state(&path).unwrap();
                let mut restored = GpuExecutableNetwork::new(network.clone(), FakeBackend);
                restored.params = gpu.params.clone();
                restored.init_train_buffers_gpu().unwrap();
                restored.load_optimizer_state(&path).unwrap();
                let restored_state = restored.host_adam_state.borrow();
                assert_eq!(restored_state[0].as_ref(), Some(&expected));
                drop(restored_state);
                fs::remove_file(path).unwrap();
            }
        }
        drop(forward);
    }
}
