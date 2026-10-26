use super::*;
use crate::detect::{self, DetectionOptions};
use crate::error::Result;
use crate::evaluate::{ObbValidationDataset, ValidationDataset};
use crate::runtime::{GpuForwardState, reject_gpu_hyperbolic_foveation};
use crate::train::{self, DataConfig};
use crate::weights::{inspect_weights_file, save_weights};

fn network(options: &str) -> Network {
    Network::from_config(ConfigFile::parse_str("yolo3d-runtime.cfg", &format!(
        "[net]\nwidth=3\nheight=5\nchannels=2\nbatch=2\n{options}\n[convolutional]\nfilters=30\nsize=1\nactivation=linear\n[yolo_3d]\n"
    )).unwrap()).unwrap()
}

fn executable(options: &str) -> ExecutableNetwork {
    let mut e = ExecutableNetwork::new(network(options));
    let LayerParams::Convolutional(p) = &mut e.params[0] else {
        panic!("conv")
    };
    p.weights.fill(0.1);
    for (c, bias) in p.biases.iter_mut().enumerate() {
        *bias = c as f32 * 0.03 - 0.4;
    }
    e
}

fn input(e: &ExecutableNetwork) -> Tensor {
    let shape = e.network.net.input_shape();
    Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| 0.1 + (i % 17) as f32 * 0.02)
            .collect(),
    )
    .unwrap()
}

#[test]
fn yolo_3d_forward_is_raw_pass_through_and_records_separate_head() {
    let cfg = ConfigFile::parse_str(
        "raw.cfg",
        "[net]\nwidth=3\nheight=5\nchannels=62\nbatch=2\n[cuboid]\nclasses=2\nnum=2\nmask=1,0\n",
    )
    .unwrap();
    let e = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let shape = e.network.net.input_shape();
    let tensor = Tensor::from_data(
        shape,
        (0..shape.batch * shape.element_count())
            .map(|i| i as f32 - 900.0)
            .collect(),
    )
    .unwrap();
    for forward in [
        e.forward(&tensor).unwrap(),
        e.forward_training(&tensor).unwrap(),
    ] {
        assert_eq!(forward.final_output(), Some(&tensor));
        assert!(forward.yolo_heads.is_empty() && forward.modern_heads.is_empty());
        assert_eq!(forward.yolo_3d_heads.len(), 1);
        assert_eq!(forward.yolo_3d_heads[0].layer_index, 0);
        assert_eq!(forward.yolo_3d_heads[0].output, tensor);
    }
    assert_eq!(e.forward_inference(&tensor).unwrap(), tensor);
    assert!(matches!(e.params[0], LayerParams::None));
    assert!(matches!(e.train_buffers[0], LayerTrainBuffers::None));
}

#[test]
fn yolo_3d_backward_is_identity_and_reaches_upstream_convolution() {
    let e = executable("");
    let x = input(&e);
    let f = e.forward(&x).unwrap();
    let output = f.final_output().unwrap();
    let seed = Tensor::from_data(
        output.shape,
        (0..output.len())
            .map(|i| (i % 11) as f32 * 0.01 - 0.03)
            .collect(),
    )
    .unwrap();
    let backward = e.backward(&x, &f, &seed).unwrap();
    assert_eq!(backward.layer_output_gradients[0], seed);
    assert!(matches!(backward.param_gradients[1], LayerParams::None));
    let LayerParams::Convolutional(gradient) = &backward.param_gradients[0] else {
        panic!("conv gradient")
    };
    // Raw-output dot-product cotangent, not a cuboid loss or source delta.
    for index in [0, 17, 59] {
        let score = |offset: f32| {
            let mut shifted = e.clone();
            let LayerParams::Convolutional(p) = &mut shifted.params[0] else {
                unreachable!()
            };
            p.weights[index] += offset;
            let f = shifted.forward(&x).unwrap();
            f.final_output()
                .unwrap()
                .data
                .iter()
                .zip(&seed.data)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
        };
        let fd = (score(0.001) - score(-0.001)) / 0.002;
        assert!(
            (fd - f64::from(gradient.weights[index])).abs() < 0.001,
            "index {index}: {fd} vs {}",
            gradient.weights[index]
        );
    }
}

#[test]
fn yolo_3d_mixed_forward_keeps_the_two_head_collections_disjoint() {
    let cfg = ConfigFile::parse_str("mixed.cfg", "[net]\nwidth=3\nheight=5\nchannels=6\n[yolo]\nclasses=1\nnum=1\n[convolutional]\nsize=1\nfilters=30\n[yolo_3d]\n").unwrap();
    let e = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let f = e
        .forward(&Tensor::zeros(e.network.net.input_shape()))
        .unwrap();
    assert_eq!(f.yolo_heads.len(), 1);
    assert_eq!(f.yolo_heads[0].layer_index, 0);
    assert_eq!(f.yolo_3d_heads.len(), 1);
    assert_eq!(f.yolo_3d_heads[0].layer_index, 2);
}

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("yolo3d-{label}-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn yolo_3d_weights_round_trip_exact_length_with_upstream_conv() {
    let e = executable("");
    let dir = TempDir::new("weights");
    let path = dir.0.join("model.weights");
    save_weights(&e.network, &e.params, &path, 7).unwrap();
    // 30 biases + [30 out][2 in][1][1] weights; cuboid head adds zero bytes.
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 20 + 4 * 90);
    let inspected = inspect_weights_file(&e.network, &path).unwrap();
    assert_eq!(inspected.trailing_bytes, 0);
    assert_eq!(inspected.layers.len(), 1);
    let mut restored = ExecutableNetwork::new(e.network.clone());
    restored.load_weights_file(&path).unwrap();
    assert_eq!(restored.params, e.params);
}

#[test]
fn yolo_3d_seeded_sgd_adam_and_muon_update_only_upstream_weights() {
    for optimizer in ["sgd", "adam", "muon"] {
        let options = match optimizer {
            "adam" => "adam=1",
            "muon" => "muon=1",
            _ => "",
        };
        let mut e = executable(options);
        let before = e.params.clone();
        let x = input(&e);
        let f = e.forward(&x).unwrap();
        let seed = Tensor::filled(f.final_output().unwrap().shape, 0.03);
        e.backward_accumulate(&x, &f, &seed).unwrap();
        match optimizer {
            "adam" => e.update_weights_adam(0.01, 2).unwrap(),
            "muon" => e.update_weights_muon(0.01, 2).unwrap(),
            _ => e.update_weights(0.01, 0.0, 0.0, 2),
        }
        assert_ne!(e.params[0], before[0], "{optimizer}");
        assert!(matches!(e.params[1], LayerParams::None));
        assert!(matches!(e.train_buffers[1], LayerTrainBuffers::None));
        assert!(e.muon_state[1].is_none());
        assert!(e.host_adam_state[1].is_none());
        let dir = TempDir::new(optimizer);
        let state = dir.0.join("optimizer.opt");
        e.save_optimizer_state(&state).unwrap();
        let mut restored = ExecutableNetwork::new(e.network.clone());
        restored.load_optimizer_state(&state).unwrap();
        assert_eq!(restored.train_buffers, e.train_buffers);
        assert_eq!(restored.muon_state, e.muon_state);
        assert!(restored.muon_state[1].is_none());
        assert!(restored.host_adam_state[1].is_none());
    }
}

#[test]
fn yolo_3d_existing_2d_and_obb_apis_reject_even_empty_head_lists() {
    let e = executable("");
    let f = e.forward(&input(&e)).unwrap();
    let options = DetectionOptions::default();
    let check = |result: Result<()>| {
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("[yolo_3d]") && error.contains("not implemented"),
            "{error}"
        );
    };
    check(train::compute_yolo_training_loss(&e.network, &f, &[vec![], vec![]]).map(|_| ()));
    check(train::compute_obb_yolo_training_deltas(&e.network, &f, &[vec![], vec![]]).map(|_| ()));
    check(detect::decode_yolo_detections(&e.network, &[], 3, 5, options).map(|_| ()));
    check(detect::decode_network_detections(&e.network, &f, 0, 3, 5, options).map(|_| ()));
    check(detect::decode_obb_yolo_detections(&e.network, &[], 3, 5, options).map(|_| ()));
    check(detect::decode_obb_network_detections(&e.network, &f, 3, 5, options).map(|_| ()));
    check(detect::decode_modern_detections(&e.network, &[], 3, 5, options).map(|_| ()));
    let data = DataConfig {
        source: "missing.data".into(),
        classes: None,
        train: None,
        valid: None,
        names: None,
        backup: None,
        options: Default::default(),
    };
    check(ValidationDataset::from_data(&e.network, &data).map(|_| ()));
    check(ObbValidationDataset::from_data(&e.network, &data).map(|_| ()));
}

#[test]
fn yolo_3d_rejects_gpu_execution_backward_and_updates() {
    let net = network("");
    let input = Tensor::zeros(net.net.input_shape());
    let mut gpu = GpuExecutableNetwork::new(net.clone(), FakeBackend);
    for result in [
        gpu.forward(&input).map(|_| ()),
        gpu.forward_compact(&input).map(|_| ()),
        gpu.update_weights_gpu(0.01, 0.0, 0.0, 2),
        gpu.update_weights_gpu_deferred(0.01, 0.0, 0.0, 2),
        reject_gpu_hyperbolic_foveation(&net, "f16 inference"),
    ] {
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("yolo_3d") && error.contains("Stage-1 CPU support"),
            "{error}"
        );
    }
    let state = GpuForwardState {
        input_gpu: FakeTensor {
            shape: vec![],
            data: vec![],
        },
        gpu_layer_outputs: vec![],
        gpu_layer_activation_inputs: vec![],
        gpu_batch_norm_caches: vec![],
        gpu_rnn_caches: vec![],
        gpu_crnn_caches: vec![],
        gpu_lstm_caches: vec![],
        gpu_gru_caches: vec![],
        tucker_attention_device_caches: vec![],
        mambavision_device_caches: vec![],
        kimi_kda_device_caches: vec![],
        deform_device_caches: vec![],
        clifford_device_caches: vec![],
        vit_device_caches: vec![],
        wmhf_device_caches: vec![],
        swin_device_caches: vec![],
        dpt_head_device_caches: vec![],
        depth_anything_device_caches: vec![],
        recursive_block_device_caches: vec![],
        gpu_moe_caches: vec![],
        gpu_eml_caches: vec![],
        gpu_dcnv4_caches: vec![],
        cpu_forward: ForwardResult::from_heads(vec![], vec![]),
    };
    let error = gpu
        .backward_accumulate_seeded_gpu(&state, &[None, None])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("yolo_3d") && error.contains("Stage-1 CPU support"),
        "{error}"
    );
}

#[test]
fn yolo_3d_from_heads_constructor_has_no_cuboid_outputs() {
    assert!(
        ForwardResult::from_heads(vec![], vec![])
            .yolo_3d_heads
            .is_empty()
    );
}

#[test]
fn yolo_3d_onnx_export_still_rejects() {
    let e = executable("");
    let error = crate::export_onnx(&e.network, &e.params, &crate::OnnxOptions::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("yolo_3d"), "{error}");
}
