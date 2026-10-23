use super::*;
use crate::depth_anything::{DepthAnythingGeometry, DepthAnythingWeights};
use crate::model::TensorShape;
use crate::weights::{inspect_weights_file, save_weights};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn depth_layer_options(extra: &str) -> String {
    format!(
        "patch_size=1\ndim=2\ndepth=1\nheads=1\nmlp_dim=3\npos_grid=1\ntaps=0\napply_imagenet_norm=0\nfreeze=1\n{extra}"
    )
}

fn depth_network(net_options: &str, layer_options: &str) -> Network {
    let config = format!(
        "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\n{net_options}\n[depth_anything]\n{}",
        depth_layer_options(layer_options)
    );
    Network::from_config(
        ConfigFile::parse_str("depth-anything-runtime.cfg", &config).expect("depth cfg parses"),
    )
    .expect("depth network builds")
}

fn small_depth_geometry() -> DepthAnythingGeometry {
    DepthAnythingGeometry {
        batch: 1,
        height: 1,
        width: 2,
        patch_size: 1,
        dim: 2,
        depth: 1,
        heads: 1,
        mlp_dim: 3,
        pos_grid_h: 1,
        pos_grid_w: 1,
        ln_eps: 1.0e-6,
        apply_imagenet_norm: false,
        imagenet_mean: [0.485, 0.456, 0.406],
        imagenet_std: [0.229, 0.224, 0.225],
        taps: vec![0],
        output_clamp: 0.0,
    }
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "darknet-depth-anything-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("create temp test directory");
    path
}

fn write_darknet_blob(path: &std::path::Path, values: &[f32]) {
    let mut bytes = Vec::with_capacity(20 + values.len() * 4);
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&17u64.to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes).expect("write test weights");
}

fn configure_depth_weights(executable: &mut ExecutableNetwork) {
    let layer_index = executable
        .network
        .layers
        .iter()
        .position(|layer| matches!(layer.kind, LayerKind::DepthAnything(_)))
        .expect("depth_anything layer");
    let LayerKind::DepthAnything(spec) = &executable.network.layers[layer_index].kind else {
        unreachable!()
    };
    let geometry = spec.geometry(executable.network.layers[layer_index].input_shape);
    let LayerParams::DepthAnything(weights) = &mut executable.params[layer_index] else {
        panic!("expected depth_anything parameters")
    };
    let mut blob = weights.to_blob();
    for (index, value) in blob.iter_mut().enumerate() {
        *value = ((index % 23) as f32 * 0.29).sin() * 0.1;
    }
    *weights = DepthAnythingWeights::from_blob(&blob, &geometry).unwrap();
}

#[test]
fn depth_anything_config_aliases_defaults_shape_and_weight_metadata() {
    let aliases = ["depth_anything", "depth_anything_v2", "dinov2", "dino"];
    for alias in aliases {
        let config = format!(
            "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\n[{alias}]\n{}",
            depth_layer_options("")
        );
        let network =
            Network::from_config(ConfigFile::parse_str("depth-alias.cfg", &config).unwrap())
                .unwrap();
        assert_eq!(network.layers[0].kind.name(), "depth_anything");
        assert_eq!(network.layers[0].output_shape, TensorShape::new(2, 1, 2, 1));
    }

    let defaults = Network::from_config(
        ConfigFile::parse_str(
            "depth-defaults.cfg",
            "[net]\nwidth=28\nheight=42\nchannels=3\nbatch=2\n[depth_anything]\n",
        )
        .unwrap(),
    )
    .unwrap();
    let LayerKind::DepthAnything(spec) = &defaults.layers[0].kind else {
        panic!("depth_anything kind")
    };
    assert_eq!(
        (spec.patch_size, spec.dim, spec.depth, spec.heads),
        (14, 384, 12, 6)
    );
    assert_eq!(spec.mlp_dim, 1536);
    assert_eq!((spec.pos_grid_h, spec.pos_grid_w), (37, 37));
    assert_eq!(spec.taps, [2, 5, 8, 11]);
    assert!(spec.apply_imagenet_norm);
    assert_eq!(
        defaults.layers[0].output_shape,
        TensorShape::new(2, 3, 1536, 2)
    );
    let arrays = defaults.layers[0].weights().unwrap().entries;
    assert_eq!(arrays.first().unwrap().array.name(), "patch_w");
    assert_eq!(arrays.last().unwrap().array.name(), "ls2");
    assert_eq!(
        arrays.iter().map(|entry| entry.len).sum::<usize>(),
        spec.geometry(defaults.layers[0].input_shape)
            .weight_count()
            .unwrap()
    );

    for (net, layer, reason) in [
        ("width=2\nheight=1\nchannels=1", "", "RGB input"),
        ("width=2\nheight=1\nchannels=3", "variant=base", "variant"),
        (
            "width=2\nheight=1\nchannels=3",
            "heads=0",
            "positive dimensions",
        ),
        (
            "width=2\nheight=1\nchannels=3",
            "taps=1",
            "tap outside depth",
        ),
        (
            "width=2\nheight=1\nchannels=3",
            "mean=0.1,0.2",
            "mean length",
        ),
        (
            "width=2\nheight=1\nchannels=3",
            "std=0.2,0,-0.2",
            "positive std",
        ),
        ("width=2\nheight=1\nchannels=3", "surprise=1", "unknown key"),
        (
            "width=2\nheight=1\nchannels=3",
            "stream=0",
            "unsupported GPU stream",
        ),
        (
            "width=2\nheight=1\nchannels=3",
            "freeze=2",
            "invalid freeze",
        ),
    ] {
        let config = format!(
            "[net]\n{net}\nbatch=1\n[depth_anything]\n{}",
            depth_layer_options(layer)
        );
        let error =
            Network::from_config(ConfigFile::parse_str("depth-invalid.cfg", &config).unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("depth_anything"), "{reason}: {error}");
    }
}

#[test]
fn depth_anything_runtime_forward_backward_and_compact_cpu_fallback() {
    let network = depth_network("", "");
    let mut executable = ExecutableNetwork::new(network.clone());
    configure_depth_weights(&mut executable);
    assert_eq!(executable.train_buffers[0], LayerTrainBuffers::None);

    let input = Tensor::from_data(
        network.net.input_shape(),
        vec![0.15, 0.37, -0.2, 0.6, 0.8, -0.4],
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    assert_eq!(forward.layer_outputs[0].shape, TensorShape::new(2, 1, 2, 1));
    assert!(forward.depth_anything_caches[0].is_some());
    let compact = executable.forward_inference(&input).unwrap();
    assert_eq!(compact, forward.layer_outputs[0]);

    let seed =
        Tensor::from_data(network.layers[0].output_shape, vec![0.3, -0.7, 0.5, 0.2]).unwrap();
    let seeds = [Some(seed.clone())];
    let cpp_result = executable
        .backward_seeded_with_grad_mode(&input, &forward, &seeds, super::super::GradMode::Cpp)
        .unwrap();
    assert!(
        cpp_result
            .input_gradient
            .data
            .iter()
            .all(|value| *value == 0.0)
    );
    let result = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &seeds,
            super::super::GradMode::Analytical,
        )
        .unwrap();
    let LayerParams::DepthAnything(gradient) = &result.param_gradients[0] else {
        panic!("expected frozen depth gradient slot")
    };
    assert!(gradient.as_slice().iter().all(|value| *value == 0.0));
    assert!(
        result
            .input_gradient
            .data
            .iter()
            .any(|value| value.abs() > 1.0e-8)
    );

    let LayerKind::DepthAnything(spec) = &network.layers[0].kind else {
        panic!("depth_anything spec")
    };
    let LayerParams::DepthAnything(weights) = &executable.params[0] else {
        panic!("depth_anything weights")
    };
    let cache = forward.depth_anything_caches[0].as_ref().unwrap();
    assert!(
        super::super::depth_anything_input_gradient(
            super::super::GradMode::Cpp,
            spec,
            input.shape,
            &seed.data,
            weights,
            cache,
        )
        .unwrap()
        .is_none()
    );
    let analytical = super::super::depth_anything_input_gradient(
        super::super::GradMode::Analytical,
        spec,
        input.shape,
        &seed.data,
        weights,
        cache,
    )
    .unwrap()
    .unwrap();
    assert_eq!(analytical, result.input_gradient);
}

#[test]
fn depth_anything_input_gradient_fans_into_preceding_convolution_and_cpp_stops() {
    let config = format!(
        "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\n[convolutional]\nfilters=3\nsize=1\nactivation=linear\n[depth_anything]\n{}",
        depth_layer_options("")
    );
    let network =
        Network::from_config(ConfigFile::parse_str("depth-anything-fanin.cfg", &config).unwrap())
            .unwrap();
    let mut executable = ExecutableNetwork::new(network.clone());
    configure_depth_weights(&mut executable);
    let LayerParams::Convolutional(conv) = &mut executable.params[0] else {
        panic!("preceding convolution params")
    };
    conv.biases.fill(0.0);
    conv.weights.fill(0.0);
    for channel in 0..3 {
        conv.weights[channel * 3 + channel] = 1.0;
    }

    let input = Tensor::from_data(
        network.net.input_shape(),
        vec![0.15, 0.37, -0.2, 0.6, 0.8, -0.4],
    )
    .unwrap();
    let forward = executable.forward(&input).unwrap();
    let seed =
        Tensor::from_data(network.layers[1].output_shape, vec![0.3, -0.7, 0.5, 0.2]).unwrap();
    let seeds = [None, Some(seed)];
    let analytical = executable
        .backward_seeded_with_grad_mode(
            &input,
            &forward,
            &seeds,
            super::super::GradMode::Analytical,
        )
        .unwrap();
    assert!(
        analytical
            .input_gradient
            .data
            .iter()
            .any(|value| value.abs() > 1.0e-8)
    );
    assert_eq!(
        analytical.layer_output_gradients[0].data,
        analytical.input_gradient.data
    );
    let LayerParams::Convolutional(conv_gradient) = &analytical.param_gradients[0] else {
        panic!("preceding convolution gradient")
    };
    assert!(
        conv_gradient
            .weights
            .iter()
            .any(|value| value.abs() > 1.0e-8)
    );

    let cpp = executable
        .backward_seeded_with_grad_mode(&input, &forward, &seeds, super::super::GradMode::Cpp)
        .unwrap();
    assert!(cpp.input_gradient.data.iter().all(|value| *value == 0.0));
    let LayerParams::Convolutional(conv_gradient) = &cpp.param_gradients[0] else {
        panic!("preceding convolution gradient")
    };
    assert!(conv_gradient.weights.iter().all(|value| *value == 0.0));
}

#[test]
fn depth_anything_frozen_values_survive_sgd_adamw_and_muon_updates() {
    let baseline = depth_network("", "");
    let mut sgd = ExecutableNetwork::new(baseline);
    configure_depth_weights(&mut sgd);
    let original = sgd.params[0].clone();
    sgd.update_weights(0.1, 0.9, 0.01, 1);
    assert_eq!(sgd.params[0], original);

    let adam_network = depth_network("adam=1\ndecoupled_decay=1", "");
    let mut adam = ExecutableNetwork::new(adam_network);
    configure_depth_weights(&mut adam);
    let original = adam.params[0].clone();
    adam.update_weights_adam(0.1, 1).unwrap();
    assert_eq!(adam.params[0], original);

    let muon_network = depth_network("muon=1\ndecoupled_decay=1", "");
    let mut muon = ExecutableNetwork::new(muon_network);
    configure_depth_weights(&mut muon);
    let original = muon.params[0].clone();
    muon.update_weights_muon(0.1, 1).unwrap();
    assert_eq!(muon.params[0], original);
}

#[test]
fn depth_anything_external_weights_resolve_and_are_excluded_from_network_stream() {
    let dir = temp_dir("external");
    let cfg_path = dir.join("network.cfg");
    let weights_path = dir.join("encoder.weights");
    let geometry = small_depth_geometry();
    let blob = (0..geometry.weight_count().unwrap())
        .map(|index| (index as f32 + 1.0) * 0.01)
        .collect::<Vec<_>>();
    write_darknet_blob(&weights_path, &blob);
    let config = format!(
        "[net]\nwidth=2\nheight=1\nchannels=3\nbatch=1\n[depth_anything]\n{}weights=encoder.weights\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n",
        depth_layer_options("")
    );
    let network = Network::from_config(ConfigFile::parse_str(&cfg_path, &config).unwrap()).unwrap();
    let LayerKind::DepthAnything(spec) = &network.layers[0].kind else {
        panic!("depth_anything spec")
    };
    assert_eq!(spec.weights_path.as_deref(), Some(weights_path.as_path()));
    assert_eq!(spec.external_weights.as_ref().unwrap().as_slice(), blob);
    assert!(network.layers[0].weights().is_none());
    assert!(network.layers[1].weights().is_some());

    let mut executable = ExecutableNetwork::new(network.clone());
    let LayerParams::DepthAnything(params) = &executable.params[0] else {
        panic!("depth_anything params")
    };
    assert_eq!(params.as_slice(), blob);
    let LayerParams::Convolutional(conv) = &mut executable.params[1] else {
        panic!("neighbor convolution params")
    };
    conv.biases[0] = 0.25;
    conv.weights.copy_from_slice(&[0.5, -0.75]);

    let checkpoint = dir.join("trainable.weights");
    save_weights(&network, &executable.params, &checkpoint, 9).unwrap();
    let inspection = inspect_weights_file(&network, &checkpoint).unwrap();
    assert_eq!(inspection.trailing_bytes, 0);
    assert_eq!(inspection.layers.len(), 1);
    assert_eq!(inspection.layers[0].layer_index, 1);
    assert_eq!(fs::metadata(&checkpoint).unwrap().len(), 20 + 3 * 4);

    let mut restored = ExecutableNetwork::new(network.clone());
    restored.load_weights_file(&checkpoint).unwrap();
    assert_eq!(restored.params[0], executable.params[0]);
    assert_eq!(restored.params[1], executable.params[1]);

    let bad_path = dir.join("extra.weights");
    let mut bad_file = fs::read(&weights_path).unwrap();
    bad_file.push(0);
    fs::write(&bad_path, bad_file).unwrap();
    let bad_cfg = config.replace("encoder.weights", "extra.weights");
    assert!(Network::from_config(ConfigFile::parse_str(&cfg_path, &bad_cfg).unwrap()).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn depth_anything_sequential_network_weights_round_trip_type_major() {
    let network = depth_network("", "");
    let mut original = ExecutableNetwork::new(network.clone());
    configure_depth_weights(&mut original);
    let params_before = original.params[0].clone();
    let dir = temp_dir("sequential");
    let path = dir.join("network.weights");
    save_weights(&network, &original.params, &path, 23).unwrap();

    let inspection = inspect_weights_file(&network, &path).unwrap();
    assert_eq!(inspection.header.seen, 23);
    assert_eq!(inspection.trailing_bytes, 0);
    assert_eq!(inspection.layers.len(), 1);
    let LayerKind::DepthAnything(spec) = &network.layers[0].kind else {
        panic!("depth_anything spec")
    };
    assert_eq!(
        inspection.layers[0]
            .arrays
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        crate::depth_anything::weight_array_sizes(&spec.geometry(network.layers[0].input_shape))
            .unwrap()
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        20 + spec
            .geometry(network.layers[0].input_shape)
            .weight_count()
            .unwrap() as u64
            * 4
    );

    let mut restored = ExecutableNetwork::new(network);
    restored.load_weights_file(&path).unwrap();
    assert_eq!(restored.params[0], params_before);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn depth_anything_gpu_execution_backward_and_updates_reject_explicitly() {
    let network = depth_network("", "");
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    let input = Tensor::from_data(network.net.input_shape(), vec![0.0; 6]).unwrap();
    for error in [
        gpu.forward(&input).unwrap_err().to_string(),
        gpu.forward_with_state(&input).unwrap_err().to_string(),
        gpu.forward_compact(&input).unwrap_err().to_string(),
        gpu.update_weights_gpu(0.1, 0.0, 0.0, 1)
            .unwrap_err()
            .to_string(),
        gpu.update_weights_gpu_deferred(0.1, 0.0, 0.0, 1)
            .unwrap_err()
            .to_string(),
    ] {
        assert!(error.contains("[depth_anything] GPU"), "{error}");
        assert!(error.contains("Stage-1 CPU support"), "{error}");
    }

    let dummy_forward = super::super::GpuForwardState {
        input_gpu: FakeTensor {
            shape: Vec::new(),
            data: Vec::new(),
        },
        gpu_layer_outputs: Vec::new(),
        gpu_layer_activation_inputs: Vec::new(),
        gpu_batch_norm_caches: Vec::new(),
        gpu_rnn_caches: Vec::new(),
        gpu_crnn_caches: Vec::new(),
        gpu_lstm_caches: Vec::new(),
        gpu_moe_caches: Vec::new(),
        gpu_eml_caches: Vec::new(),
        gpu_dcnv4_caches: Vec::new(),
        cpu_forward: ForwardResult::from_heads(Vec::new(), Vec::new()),
    };
    let backward_error = gpu
        .backward_accumulate_seeded_gpu(&dummy_forward, &[])
        .unwrap_err()
        .to_string();
    assert!(backward_error.contains("[depth_anything] GPU"));
    assert!(backward_error.contains("Stage-1 CPU support"));
}
