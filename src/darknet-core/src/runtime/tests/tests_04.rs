use super::*;

#[test]
fn gpu_batchnorm_conv_updates_running_stats() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut executable.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 1.0;
            let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
            batch_norm.rolling_mean[0] = 10.0;
            batch_norm.rolling_variance[0] = 20.0;
        }
        other => panic!("unexpected params: {other:?}"),
    }
    executable
        .init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
        .expect("input tensor");
    let seeds = vec![Some(Tensor::filled(
        executable.network.layers[0].output_shape,
        0.0,
    ))];
    let forward = executable
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    executable
        .backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("GPU backward should update stats");

    let params = match &executable.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
    assert!((batch_norm.rolling_mean[0] - 9.93).abs() < 1.0e-6);
    assert!((batch_norm.rolling_variance[0] - 19.82).abs() < 1.0e-6);
}

#[test]
fn gpu_batchnorm_running_stats_update_without_gradient_seed() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut executable.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 1.0;
            let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
            batch_norm.rolling_mean[0] = 10.0;
            batch_norm.rolling_variance[0] = 20.0;
        }
        other => panic!("unexpected params: {other:?}"),
    }
    executable
        .init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
        .expect("input tensor");
    let seeds = vec![None; executable.network.layers.len()];
    let forward = executable
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    executable
        .backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("GPU backward should update stats");

    let params = match &executable.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
    assert!((batch_norm.rolling_mean[0] - 9.93).abs() < 1.0e-6);
    assert!((batch_norm.rolling_variance[0] - 19.82).abs() < 1.0e-6);

    let train_buffers = executable.device_train_buffers.borrow();
    let train = train_buffers[0].as_ref().expect("train buffers exist");
    assert_eq!(train.weight_updates.data, vec![0.0]);
    assert_eq!(train.bias_updates.data, vec![0.0]);
}

pub(super) fn gpu_training_bn_network() -> Network {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                batch=2
                width=3
                height=2
                channels=2

                [convolutional]
                filters=3
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    Network::from_config(cfg).expect("network should build")
}

pub(super) fn gpu_training_bn_executable(
    weights: &[f32],
    scales: &[f32],
    biases: &[f32],
) -> GpuExecutableNetwork<FakeBackend> {
    let mut gpu = GpuExecutableNetwork::new(gpu_training_bn_network(), FakeBackend);
    match &mut gpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights = weights.to_vec();
            params.biases = biases.to_vec();
            let bn = params.batch_norm.as_mut().expect("batch norm params");
            bn.scales = scales.to_vec();
            bn.rolling_mean = vec![0.3, -0.2, 0.5];
            bn.rolling_variance = vec![2.0, 0.5, 3.0];
        }
        other => panic!("unexpected params: {other:?}"),
    }
    gpu
}

pub(super) const BN_TRAIN_WEIGHTS: [f32; 6] = [0.5, -1.0, 0.25, 0.75, -0.5, 1.5];
pub(super) const BN_TRAIN_SCALES: [f32; 3] = [1.5, 0.5, -2.0];
pub(super) const BN_TRAIN_BIASES: [f32; 3] = [0.1, -0.2, 0.3];

pub(super) fn bn_train_input() -> Vec<f32> {
    (0..24)
        .map(|i| ((i * 7 % 11) as f32) * 0.37 - 1.5)
        .collect()
}

/// Pins the rolling-stat EMA factors to C++: GPU `.99`/`.01` (batchnorm_layer.cpp
/// scal_ongpu/axpy_ongpu, cuDNN factor .01), host `.9`/`.1` (scal_cpu/axpy_cpu).
#[test]
fn batch_norm_rolling_keep_matches_cpp() {
    assert_eq!(super::BATCH_NORM_ROLLING_KEEP, 0.99);
    assert_eq!(super::BATCH_NORM_ROLLING_KEEP_CPU, 0.9);
    let mut bn = super::BatchNormParams {
        scales: vec![1.0],
        biases: vec![0.0],
        rolling_mean: vec![10.0],
        rolling_variance: vec![20.0],
    };
    super::update_batch_norm_running_stats(&mut bn, &[3.0], &[2.0], super::BATCH_NORM_ROLLING_KEEP)
        .unwrap();
    // .99 * 10 + .01 * 3, .99 * 20 + .01 * 2
    assert!((bn.rolling_mean[0] - 9.93).abs() < 1.0e-5);
    assert!((bn.rolling_variance[0] - 19.82).abs() < 1.0e-5);
    super::update_batch_norm_running_stats(
        &mut bn,
        &[3.0],
        &[2.0],
        super::BATCH_NORM_ROLLING_KEEP_CPU,
    )
    .unwrap();
    assert!((bn.rolling_mean[0] - (0.9 * 9.93 + 0.3)).abs() < 1.0e-5);
}

#[test]
fn gpu_training_forward_uses_batch_statistics_and_updates_rolling_stats() {
    let mut gpu = gpu_training_bn_executable(&BN_TRAIN_WEIGHTS, &BN_TRAIN_SCALES, &BN_TRAIN_BIASES);
    gpu.init_train_buffers_gpu().expect("train buffers");
    let input = Tensor::from_data(gpu.network.net.input_shape(), bn_train_input()).expect("input");
    let forward = gpu
        .forward_training_with_state(&input)
        .expect("training forward");

    // f64 reference: 1x1 conv, batch statistics over (batch, h, w).
    let (batch, spatial, in_c, out_c) = (2usize, 6usize, 2usize, 3usize);
    let mut raw = vec![0.0f64; batch * out_c * spatial];
    for b in 0..batch {
        for o in 0..out_c {
            for s in 0..spatial {
                raw[(b * out_c + o) * spatial + s] = (0..in_c)
                    .map(|c| {
                        BN_TRAIN_WEIGHTS[o * in_c + c] as f64
                            * input.data[(b * in_c + c) * spatial + s] as f64
                    })
                    .sum();
            }
        }
    }
    let n = (batch * spatial) as f64;
    let mut want = vec![0.0f64; raw.len()];
    let mut want_rolling = [(0.0f64, 0.0f64); 3];
    for o in 0..out_c {
        let at = |b: usize, s: usize| raw[(b * out_c + o) * spatial + s];
        let mean: f64 = (0..batch)
            .flat_map(|b| (0..spatial).map(move |s| (b, s)))
            .map(|(b, s)| at(b, s))
            .sum::<f64>()
            / n;
        let sq: f64 = (0..batch)
            .flat_map(|b| (0..spatial).map(move |s| (b, s)))
            .map(|(b, s)| (at(b, s) - mean).powi(2))
            .sum();
        let var = sq / n;
        for b in 0..batch {
            for s in 0..spatial {
                want[(b * out_c + o) * spatial + s] = (at(b, s) - mean) / (var + 1.0e-5).sqrt()
                    * BN_TRAIN_SCALES[o] as f64
                    + BN_TRAIN_BIASES[o] as f64;
            }
        }
        let (rm, rv) = ([0.3, -0.2, 0.5][o], [2.0, 0.5, 3.0][o]);
        want_rolling[o] = (0.99 * rm + 0.01 * mean, 0.99 * rv + 0.01 * sq / (n - 1.0));
    }
    for (g, w) in forward.cpu_forward.layer_outputs[0].data.iter().zip(&want) {
        assert!((*g as f64 - w).abs() < 1.0e-4, "output {g} vs {w}");
    }
    // Inference forward still uses the rolling statistics.
    let inference = gpu.forward(&input).expect("inference forward");
    assert!(
        inference.layer_outputs[0]
            .data
            .iter()
            .zip(&want)
            .any(|(g, w)| (*g as f64 - w).abs() > 1.0e-2),
        "inference must not use batch statistics"
    );

    let seeds = vec![None; gpu.network.layers.len()];
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("backward");
    let Some(LayerParams::Convolutional(params)) = gpu.params.first() else {
        panic!("conv params expected");
    };
    let bn = params.batch_norm.as_ref().expect("batch norm params");
    for o in 0..out_c {
        assert!((bn.rolling_mean[o] as f64 - want_rolling[o].0).abs() < 1.0e-5);
        assert!((bn.rolling_variance[o] as f64 - want_rolling[o].1).abs() < 1.0e-5);
    }
}

#[test]
fn gpu_training_batch_norm_backward_matches_finite_difference() {
    let input_data = bn_train_input();
    let seed_data: Vec<f32> = (0..36).map(|i| ((i * 5 % 7) as f32) * 0.3 - 0.8).collect();
    // L = sum(seed * output) through the batch-statistics forward.
    let loss = |weights: &[f32], scales: &[f32], biases: &[f32]| -> f64 {
        let gpu = gpu_training_bn_executable(weights, scales, biases);
        let input =
            Tensor::from_data(gpu.network.net.input_shape(), input_data.clone()).expect("input");
        let forward = gpu
            .forward_training_with_state(&input)
            .expect("training forward");
        forward.cpu_forward.layer_outputs[0]
            .data
            .iter()
            .zip(&seed_data)
            .map(|(o, s)| *o as f64 * *s as f64)
            .sum()
    };

    let mut gpu = gpu_training_bn_executable(&BN_TRAIN_WEIGHTS, &BN_TRAIN_SCALES, &BN_TRAIN_BIASES);
    gpu.init_train_buffers_gpu().expect("train buffers");
    let input =
        Tensor::from_data(gpu.network.net.input_shape(), input_data.clone()).expect("input");
    let forward = gpu
        .forward_training_with_state(&input)
        .expect("training forward");
    let seeds = vec![Some(
        Tensor::from_data(gpu.network.layers[0].output_shape, seed_data.clone()).expect("seed"),
    )];
    gpu.backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("backward");
    let (weight_updates, bias_updates, scale_updates) = {
        let buffers = gpu.device_train_buffers.borrow();
        let train = buffers[0].as_ref().expect("train buffers");
        (
            train.weight_updates.data.clone(),
            train.bias_updates.data.clone(),
            train
                .scale_updates
                .as_ref()
                .expect("scale updates")
                .data
                .clone(),
        )
    };

    let eps = 1.0e-2f32;
    let fd = |which: usize, index: usize| -> f64 {
        let (mut w, mut s, mut b) = (
            BN_TRAIN_WEIGHTS.to_vec(),
            BN_TRAIN_SCALES.to_vec(),
            BN_TRAIN_BIASES.to_vec(),
        );
        let target = match which {
            0 => &mut w,
            1 => &mut s,
            _ => &mut b,
        };
        target[index] += eps;
        let lp = loss(&w, &s, &b);
        let target = match which {
            0 => &mut w,
            1 => &mut s,
            _ => &mut b,
        };
        target[index] -= 2.0 * eps;
        let lm = loss(&w, &s, &b);
        (lp - lm) / (2.0 * eps as f64)
    };
    for (which, name, got) in [
        (0, "weight", &weight_updates),
        (1, "scale", &scale_updates),
        (2, "bias", &bias_updates),
    ] {
        for (i, g) in got.iter().enumerate() {
            let want = fd(which, i);
            assert!(
                (*g as f64 - want).abs() < 2.0e-2 + 2.0e-2 * want.abs(),
                "{name} update[{i}]: backward {g} vs finite difference {want}"
            );
        }
    }
}

#[test]
fn gpu_yolo_head_matches_cpu_and_cpp_activated_decode() {
    // Documents P3b: the Rust runtime keeps YOLO head outputs as raw
    // (pre-activation) logits on both CPU and GPU paths and applies
    // logistic + scale_x_y on the host inside decode/loss.  C++ applies
    // them on the device in forward_yolo_layer_gpu before decoding.  For
    // new_coords=0 (LegoGears_v2) both must yield identical detections.
    use crate::detect::{DetectionOptions, decode_yolo_detections};

    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                batch=1
                width=8
                height=8
                channels=1

                [convolutional]
                filters=7
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=9,12
                classes=2
                num=1
                scale_x_y=1.05
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    match &mut cpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights = vec![1.3, -0.9, 0.7, 1.1, 2.5, 1.7, -1.4];
            params.biases = vec![0.1, -0.2, 0.3, -0.1, -1.5, -0.8, -1.2];
        }
        other => panic!("unexpected params: {other:?}"),
    }
    gpu.params = cpu.params.clone();
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..64).map(|idx| (idx as f32 - 20.0) / 16.0).collect(),
    )
    .expect("input tensor");

    let cpu_forward = cpu.forward(&input).expect("CPU forward");
    let gpu_full = gpu.forward_with_state(&input).expect("GPU forward");
    let gpu_compact = gpu.forward_compact(&input).expect("GPU compact forward");
    assert_eq!(cpu_forward.yolo_heads, gpu_full.cpu_forward.yolo_heads);
    assert_eq!(cpu_forward.yolo_heads, gpu_compact.yolo_heads);

    let options = DetectionOptions {
        confidence_threshold: 0.1,
        nms_threshold: 0.45,
        letterbox: false,
    };
    let decode = |heads: &[super::YoloHeadOutput]| {
        let mut dets =
            crate::detect::decode_yolo_detections(&network, heads, 8, 8, options).expect("decode");
        dets.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
        dets
    };
    let cpu_dets = decode(&cpu_forward.yolo_heads);
    let gpu_dets = decode(&gpu_compact.yolo_heads);
    assert_eq!(cpu_dets, gpu_dets);

    // C++ reference: forward_yolo_layer_gpu activation, then
    // get_yolo_detections on the activated tensor (before NMS).
    let spec = match &network.layers[1].kind {
        LayerKind::Yolo(spec) => spec.clone(),
        other => panic!("unexpected layer {other:?}"),
    };
    let raw = &cpu_forward.yolo_heads[0].output;
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let (w, h, entries) = (8usize, 8usize, spec.classes + 5);
    let mut reference = Vec::new();
    for row in 0..h {
        for col in 0..w {
            // activated output, C++ layout: x,y logistic then scal_add; obj+cls logistic
            let x = spec.scale_x_y * sig(raw.get(0, 0, row, col)) - 0.5 * (spec.scale_x_y - 1.0);
            let y = spec.scale_x_y * sig(raw.get(0, 1, row, col)) - 0.5 * (spec.scale_x_y - 1.0);
            let obj = sig(raw.get(0, 4, row, col));
            if obj <= options.confidence_threshold {
                continue;
            }
            let (bw, bh) = (
                raw.get(0, 2, row, col).exp() * 9.0 / 8.0,
                raw.get(0, 3, row, col).exp() * 12.0 / 8.0,
            );
            for c in 0..spec.classes {
                let conf = obj * sig(raw.get(0, 5 + c, row, col));
                if conf > options.confidence_threshold {
                    reference.push((
                        conf,
                        (col as f32 + x) / w as f32 * 8.0,
                        (row as f32 + y) / h as f32 * 8.0,
                        bw * 8.0,
                        bh * 8.0,
                    ));
                }
            }
        }
    }
    assert_eq!(entries, 7);
    assert!(!reference.is_empty(), "test must exercise detections");
    reference.sort_by(|a, b| a.0.total_cmp(&b.0));
    // decode applies NMS; compare the pre-NMS reference against a run with
    // NMS disabled via an overlap threshold of 1.0.
    let no_nms = DetectionOptions {
        nms_threshold: 1.0,
        ..options
    };
    let mut dets =
        decode_yolo_detections(&network, &cpu_forward.yolo_heads, 8, 8, no_nms).expect("decode");
    dets.sort_by(|a, b| a.confidence.total_cmp(&b.confidence));
    assert_eq!(dets.len(), reference.len());
    for (d, r) in dets.iter().zip(&reference) {
        assert!((d.confidence - r.0).abs() < 1e-6);
        assert!((d.bbox.x - r.1).abs() < 1e-4);
        assert!((d.bbox.y - r.2).abs() < 1e-4);
        assert!((d.bbox.w - r.3).abs() < 1e-4);
        assert!((d.bbox.h - r.4).abs() < 1e-4);
    }
}

#[test]
fn gpu_yolo_loss_backward_matches_cpu_train_buffers() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                batch=1
                width=8
                height=8
                channels=1

                [convolutional]
                filters=6
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut cpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights = vec![0.03, -0.02, 0.05, 0.07, -0.04, 0.01];
            params.biases = vec![0.2, -0.1, 0.05, -0.15, 0.3, -0.25];
        }
        other => panic!("unexpected params: {other:?}"),
    }
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..64).map(|idx| (idx as f32 + 1.0) / 64.0).collect(),
    )
    .expect("input tensor");
    let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

    let labels = vec![vec![YoloLabel {
        class_id: 0,
        x: 0.5,
        y: 0.5,
        w: 0.25,
        h: 0.25,
    }]];
    let training =
        compute_yolo_training_loss(&cpu.network, &cpu_forward, &labels).expect("loss computes");
    assert!(training.loss.is_finite());
    assert!(training.positive_matches > 0);

    cpu.backward_accumulate_seeded(&input, &cpu_forward, &training.layer_output_gradient_seeds)
        .expect("CPU backward should accumulate");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &training.layer_output_gradient_seeds)
        .expect("GPU backward should accumulate");

    let cpu_train = match &cpu.train_buffers[0] {
        LayerTrainBuffers::Convolutional(train) => train,
        other => panic!("unexpected CPU train buffers: {other:?}"),
    };
    let gpu_buffers = gpu.device_train_buffers.borrow();
    let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");

    assert_eq!(
        cpu_train.bias_updates.len(),
        gpu_train.bias_updates.data.len()
    );
    for (cpu_value, gpu_value) in cpu_train
        .bias_updates
        .iter()
        .zip(&gpu_train.bias_updates.data)
    {
        assert!((cpu_value - gpu_value).abs() < 1.0e-5);
    }
    assert_eq!(
        cpu_train.weight_updates.len(),
        gpu_train.weight_updates.data.len()
    );
    for (cpu_value, gpu_value) in cpu_train
        .weight_updates
        .iter()
        .zip(&gpu_train.weight_updates.data)
    {
        assert!((cpu_value - gpu_value).abs() < 1.0e-5);
    }
}

#[test]
fn gpu_avgpool_backward_feeds_previous_convolution() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [avgpool]
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut cpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 1.0;
            params.biases[0] = 0.0;
        }
        other => panic!("unexpected params: {other:?}"),
    }
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input = Tensor::from_data(cpu.network.net.input_shape(), vec![1.0, 2.0, 3.0, 4.0])
        .expect("input tensor");
    let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(
        Tensor::from_data(cpu.network.layers[1].output_shape, vec![8.0]).expect("seed tensor"),
    );
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU backward should accumulate");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU backward should accumulate");

    let cpu_train = match &cpu.train_buffers[0] {
        LayerTrainBuffers::Convolutional(train) => train,
        other => panic!("unexpected CPU train buffers: {other:?}"),
    };
    let gpu_buffers = gpu.device_train_buffers.borrow();
    let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");
    assert_eq!(cpu_train.bias_updates, vec![8.0]);
    assert_eq!(cpu_train.weight_updates, vec![20.0]);
    assert_eq!(gpu_train.bias_updates.data, cpu_train.bias_updates);
    assert_eq!(gpu_train.weight_updates.data, cpu_train.weight_updates);
}

#[test]
fn gpu_local_avgpool_backward_feeds_previous_convolution() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [local_avgpool]
                size=2
                stride=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut cpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 1.0;
            params.biases[0] = 0.0;
        }
        other => panic!("unexpected params: {other:?}"),
    }
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input = Tensor::from_data(cpu.network.net.input_shape(), vec![1.0, 2.0, 3.0, 4.0])
        .expect("input tensor");
    let cpu_forward = cpu.forward(&input).expect("CPU forward should succeed");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    assert_same_forward(&cpu_forward, &gpu_forward.cpu_forward);

    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[1] = Some(
        Tensor::from_data(
            cpu.network.layers[1].output_shape,
            vec![4.0, 6.0, 8.0, 10.0],
        )
        .expect("seed tensor"),
    );
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU backward should accumulate");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU backward should accumulate");

    let cpu_train = match &cpu.train_buffers[0] {
        LayerTrainBuffers::Convolutional(train) => train,
        other => panic!("unexpected CPU train buffers: {other:?}"),
    };
    let gpu_buffers = gpu.device_train_buffers.borrow();
    let gpu_train = gpu_buffers[0].as_ref().expect("GPU train buffers exist");
    assert_eq!(cpu_train.bias_updates, vec![28.0]);
    assert_eq!(cpu_train.weight_updates, vec![96.0]);
    assert_eq!(gpu_train.bias_updates.data, cpu_train.bias_updates);
    assert_eq!(gpu_train.weight_updates.data, cpu_train.weight_updates);
}

#[test]
fn gpu_runtime_matches_cpu_for_conv_maxpool_slice() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=4
                height=4
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [maxpool]
                size=2
                stride=2

                [upsample]
                stride=2
                scale=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);

    let path = std::env::temp_dir().join("darknet-runtime-parity.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.25f32.to_le_bytes());
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    cpu.load_weights_file(&path)
        .expect("cpu weights should load");
    gpu.load_weights_file(&path)
        .expect("gpu weights should load");

    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0, 5.0, 7.0, 0.0, 2.0, 4.0, 6.0,
        ],
    )
    .expect("tensor");
    let cpu_result = cpu.forward(&input).expect("cpu forward should succeed");
    let gpu_result = gpu.forward(&input).expect("gpu forward should succeed");

    assert_same_forward(&cpu_result, &gpu_result);

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_forward_compact_matches_full_forward_on_small_network() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [maxpool]
                size=2
                stride=2

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);

    let path = std::env::temp_dir().join("darknet-runtime-compact.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");

    let full = executable
        .forward(&input)
        .expect("full forward should succeed");
    let compact = executable
        .forward_compact(&input)
        .expect("compact forward should succeed");

    assert_eq!(compact.final_output, full.final_output().cloned());
    assert_eq!(compact.yolo_heads, full.yolo_heads);

    let _ = fs::remove_file(path);
}

#[test]
fn cpu_inference_compact_releases_route_and_shortcut_sources() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [upsample]
                stride=2
                scale=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1,-2

                [shortcut]
                from=-3
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0, -2.0])
        .expect("input tensor");

    let full = executable
        .forward(&input)
        .expect("full inference should succeed")
        .final_output()
        .cloned()
        .expect("full output");
    let compact = executable
        .forward_inference(&input)
        .expect("compact inference should succeed");
    assert_eq!(compact, full);
}

#[test]
fn gpu_runtime_runs_primary_yolov4_tiny_targets() {
    for cfg_name in ["yolov4-tiny.cfg", "yolov4-tiny-3l.cfg"] {
        let network = Network::from_path(repo_cfg(cfg_name)).expect("cfg should parse");
        let unsupported = network
            .layers
            .iter()
            .find(|layer| {
                !matches!(
                    layer.kind,
                    LayerKind::Convolutional(_)
                        | LayerKind::MaxPool(_)
                        | LayerKind::Route(_)
                        | LayerKind::Shortcut(_)
                        | LayerKind::Upsample(_)
                        | LayerKind::Yolo(_)
                        | LayerKind::ObbYolo(_)
                        | LayerKind::GaussianYolo(_)
                        | LayerKind::Region(_)
                        | LayerKind::Detection(_)
                        | LayerKind::Softmax { .. }
                        | LayerKind::Cost { .. }
                        | LayerKind::Dropout { .. }
                        | LayerKind::ScaleChannels(_)
                        | LayerKind::Sam(_)
                        | LayerKind::Contrastive(_)
                )
            })
            .map(|layer| (layer.index, layer.kind.name().to_owned()));

        assert_eq!(
            unsupported, None,
            "{cfg_name} contains an unsupported GPU layer"
        );
    }
}

#[test]
fn gpu_forward_compact_runs_primary_yolov4_tiny_targets() {
    for (cfg_name, expected_heads) in [("yolov4-tiny.cfg", 2usize), ("yolov4-tiny-3l.cfg", 3usize)]
    {
        let network = Network::from_path(repo_cfg(cfg_name)).expect("cfg should parse");
        let executable = GpuExecutableNetwork::new(network, ShapeOnlyBackend);
        let input = executable.zero_input();
        let summary = executable
            .forward_compact(&input)
            .expect("compact gpu forward should succeed");

        assert_eq!(summary.yolo_heads.len(), expected_heads);
        assert!(summary.final_output.is_some());
    }
}

#[test]
fn rnn_axis_layout_pins_cpp_step_batch_and_input_channels() {
    // C++ `forward_rnn_layer` advances `state.input` by `l.inputs*l.batch`
    // each step, keeps state rows independent per sample, and writes each
    // step to its output-layer slice (`rnn_layer.cpp:92-133`).
    let network = Network::from_config(
            ConfigFile::parse_str(
                "rnn-axis-layout.cfg",
                "[net]\nwidth=3\nheight=1\nchannels=1\nbatch=2\ntime_steps=2\n[rnn]\nhidden=2\noutput=3\nactivation=linear\n",
            )
            .expect("RNN axis config should parse"),
        )
        .expect("RNN axis network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Rnn(params) = &mut executable.params[0] else {
        panic!("expected RNN parameters");
    };
    params
        .input
        .weights
        .copy_from_slice(&[1.0, 2.0, 4.0, 3.0, 5.0, 7.0]);
    params
        .self_params
        .weights
        .copy_from_slice(&[0.5, 0.25, 0.2, 0.6]);
    params
        .output
        .weights
        .copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 2.0, -1.0]);
    params.input.biases.fill(0.0);
    params.self_params.biases.fill(0.0);
    params.output.biases.fill(0.0);

    let sequence_batch = 2;
    let steps = 2;
    let inputs = 3;
    let mut input_values = Vec::with_capacity(steps * sequence_batch * inputs);
    for step in 0..steps {
        for sample in 0..sequence_batch {
            for channel in 0..inputs {
                input_values.push(crate::axis_fixture::axis_value(step, sample, 0, channel) + 1.0);
            }
        }
    }
    let input = Tensor::from_data(executable.network.net.input_shape(), input_values.clone())
        .expect("RNN axis input should have the expected shape");
    let forward = executable
        .forward(&input)
        .expect("RNN forward should succeed");
    let output = forward.final_output().expect("RNN output should exist");

    // Independent source-order calculation: weights are [out][in], the
    // sequence input is [step][sample][input], and output is
    // [step][sample][output].
    let mut expected = vec![0.0; steps * sequence_batch * 3];
    let mut previous = [[0.0f32; 2]; 2];
    for step in 0..steps {
        for sample in 0..sequence_batch {
            let src = (step * sequence_batch + sample) * inputs;
            let x = &input_values[src..src + inputs];
            let mut state = [
                x[0] + 2.0 * x[1] + 4.0 * x[2],
                3.0 * x[0] + 5.0 * x[1] + 7.0 * x[2],
            ];
            state[0] += 0.5 * previous[sample][0] + 0.25 * previous[sample][1];
            state[1] += 0.2 * previous[sample][0] + 0.6 * previous[sample][1];
            let dst = (step * sequence_batch + sample) * 3;
            expected[dst] = state[0];
            expected[dst + 1] = state[1];
            expected[dst + 2] = 2.0 * state[0] - state[1];
            previous[sample] = state;
        }
    }
    assert_eq!(output.shape.batch, 4, "time-major flattened batch");
    assert_eq!(output.shape.channels, 3);
    for (index, (&actual, &want)) in output.data.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() <= 2.0e-5 * (1.0 + want.abs()),
            "RNN step/sample/output axis at flat index {index}: {actual} != {want}"
        );
    }
}

#[test]
fn rnn_forward_backward_and_weight_updates_are_wired() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "rnn.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[rnn]\nhidden=3\noutput=2\nactivation=linear\n",
            )
            .expect("RNN cfg should parse"),
        )
        .expect("RNN network should build");
    assert_eq!(network.net.input_shape().batch, 4);
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Rnn(params) = &mut executable.params[0] else {
        panic!("expected RNN parameters");
    };
    for (index, value) in params.input.weights.iter_mut().enumerate() {
        *value = 0.05 + index as f32 * 0.01;
    }
    for (index, value) in params.self_params.weights.iter_mut().enumerate() {
        *value = 0.03 + index as f32 * 0.005;
    }
    for (index, value) in params.output.weights.iter_mut().enumerate() {
        *value = 0.02 + index as f32 * 0.004;
    }
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        (0..8).map(|index| index as f32 * 0.1 - 0.2).collect(),
    )
    .expect("RNN input should have the expected shape");
    let forward = executable
        .forward(&input)
        .expect("RNN forward should succeed");
    let output = forward.final_output().expect("RNN output should exist");
    assert_eq!(output.shape, executable.network.layers[0].output_shape);
    let seed = Tensor::filled(output.shape, 1.0);
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("RNN backward should succeed");
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .all(|value| value.is_finite())
    );

    executable.zero_train_buffers();
    let forward = executable
        .forward_training(&input)
        .expect("RNN training forward should succeed");
    executable
        .backward_accumulate(&input, &forward, &seed)
        .expect("RNN train backward should succeed");
    assert!(matches!(
        executable.train_buffers[0],
        LayerTrainBuffers::Rnn(_)
    ));
}

#[test]
fn gpu_batch_normalized_rnn_backward_matches_cpu_and_updates_rolling_stats() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "rnn-bn-gpu.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[connected]\noutput=2\nactivation=linear\n[rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\nbatch_normalize=1\n",
            )
            .expect("RNN cfg should parse"),
        )
        .expect("RNN network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Connected(input_params) = &mut cpu.params[0] else {
        panic!("expected upstream connected parameters");
    };
    input_params
        .weights
        .copy_from_slice(&[0.4, -0.2, 0.1, 0.35]);
    input_params.biases.copy_from_slice(&[0.03, -0.04]);
    let LayerParams::Rnn(rnn) = &mut cpu.params[1] else {
        panic!("expected RNN parameters");
    };
    for (slot, projection) in [&mut rnn.input, &mut rnn.self_params, &mut rnn.output]
        .into_iter()
        .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.04 + (slot * 5 + index) as f32 * 0.009;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 1.0) * 0.03 + index as f32 * 0.02;
        }
        let batch_norm = projection.batch_norm.as_mut().expect("RNN BN projection");
        for feature in 0..batch_norm.scales.len() {
            batch_norm.scales[feature] = 0.75 + feature as f32 * 0.16;
            batch_norm.rolling_mean[feature] = -0.12 + feature as f32 * 0.07;
            batch_norm.rolling_variance[feature] = 0.55 + feature as f32 * 0.19;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    gpu.init_train_buffers_gpu()
        .expect("GPU RNN BN train buffers");
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..8)
            .map(|index| (index as f32 * 0.31).sin() * 0.7)
            .collect(),
    )
    .expect("RNN input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU RNN BN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("GPU RNN BN host-reference forward");
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(super::RecurrentForwardCache::Rnn(_)))
    ));
    assert!(gpu_forward.gpu_rnn_caches[1].is_none());
    assert_slice_close(
        "BN RNN output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        1e-6,
    );

    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("clear GPU BN RNN train buffers");
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.43).cos() * 0.6)
            .collect(),
    )
    .expect("RNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU BN RNN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU BN RNN backward");
    cpu.update_weights(0.025, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.025, 0.0, 0.0, 2)
        .expect("GPU BN RNN host optimizer step");

    let (LayerParams::Connected(cpu_input), LayerParams::Connected(gpu_input)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("upstream connected parameters after update");
    };
    assert_slice_close(
        "BN RNN upstream weights",
        &cpu_input.weights,
        &gpu_input.weights,
        3e-5,
    );
    assert_slice_close(
        "BN RNN upstream biases",
        &cpu_input.biases,
        &gpu_input.biases,
        3e-5,
    );
    let (LayerParams::Rnn(cpu_rnn), LayerParams::Rnn(gpu_rnn)) = (&cpu.params[1], &gpu.params[1])
    else {
        panic!("RNN parameters after update");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("input", &cpu_rnn.input, &gpu_rnn.input),
        ("recurrent", &cpu_rnn.self_params, &gpu_rnn.self_params),
        ("output", &cpu_rnn.output, &gpu_rnn.output),
    ] {
        assert_slice_close(
            &format!("BN RNN {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            3e-5,
        );
        assert_slice_close(
            &format!("BN RNN {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            3e-5,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        assert_slice_close(
            &format!("BN RNN {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            3e-5,
        );
        assert_slice_close(
            &format!("BN RNN {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            3e-5,
        );
        assert_slice_close(
            &format!("BN RNN {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            3e-5,
        );
    }
    assert!(!gpu.gradients_non_finite().expect("finite BN RNN gradients"));
    {
        let mut buffers = gpu.cpu_fallback_train_buffers.borrow_mut();
        let LayerTrainBuffers::Rnn(train) = &mut buffers[1] else {
            panic!("RNN host accumulators");
        };
        train.input.batch_norm.as_mut().unwrap().scale_updates[0] = f32::INFINITY;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("BN RNN host scale overflow check")
    );
}

#[test]
fn gpu_rnn_forward_projects_and_activates_on_backend_with_host_cache_parity() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "rnn-gpu.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[rnn]\nhidden=3\noutput=2\nactivation=tanh\nlogistic=1\nshortcut=1\n",
            )
            .expect("RNN cfg should parse"),
        )
        .expect("RNN network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    for params in [&mut cpu.params[0], &mut gpu.params[0]] {
        let LayerParams::Rnn(params) = params else {
            panic!("expected RNN parameters");
        };
        for (index, value) in params.input.weights.iter_mut().enumerate() {
            *value = 0.05 + index as f32 * 0.01;
        }
        for (index, value) in params.self_params.weights.iter_mut().enumerate() {
            *value = 0.03 + index as f32 * 0.005;
        }
        for (index, value) in params.output.weights.iter_mut().enumerate() {
            *value = 0.02 + index as f32 * 0.004;
        }
    }
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..8).map(|index| index as f32 * 0.1 - 0.2).collect(),
    )
    .expect("RNN input should have the expected shape");
    let cpu_forward = cpu.forward(&input).expect("CPU RNN forward should succeed");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("backend RNN forward should succeed");
    let expected = cpu_forward
        .final_output()
        .expect("CPU RNN output should exist");
    let actual = gpu_forward
        .cpu_forward
        .final_output()
        .expect("backend RNN output should exist");
    assert_eq!(actual.shape, expected.shape);
    for (index, (actual, expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "RNN output {index}: backend {actual}, CPU {expected}"
        );
    }
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.first(),
        Some(Some(super::RecurrentForwardCache::Rnn(_)))
    ));

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu()
        .expect("GPU RNN train buffers should initialize");
    gpu.zero_train_buffers_gpu()
        .expect("GPU RNN train buffers should clear");
    let cpu_training_forward = cpu
        .forward_training(&input)
        .expect("CPU RNN training forward should succeed");
    let gpu_training_forward = gpu
        .forward_training_device(&input)
        .expect("GPU RNN training forward should succeed");
    assert!(gpu_training_forward.gpu_rnn_caches[0].is_some());
    assert!(gpu_training_forward.cpu_forward.recurrent_caches[0].is_none());
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::filled(output_shape, 1.0);
    cpu.backward_accumulate_seeded(&input, &cpu_training_forward, &[Some(seed.clone())])
        .expect("CPU RNN training backward should succeed");
    gpu.backward_accumulate_seeded_gpu(&gpu_training_forward, &[Some(seed)])
        .expect("GPU RNN training backward should succeed");
    cpu.update_weights(0.05, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.05, 0.0, 0.0, 2)
        .expect("GPU RNN optimizer step should succeed");
    let (LayerParams::Rnn(cpu_params), LayerParams::Rnn(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("expected RNN parameters after training");
    };
    for (name, actual, expected) in [
        (
            "input weights",
            &gpu_params.input.weights,
            &cpu_params.input.weights,
        ),
        (
            "input biases",
            &gpu_params.input.biases,
            &cpu_params.input.biases,
        ),
        (
            "self weights",
            &gpu_params.self_params.weights,
            &cpu_params.self_params.weights,
        ),
        (
            "self biases",
            &gpu_params.self_params.biases,
            &cpu_params.self_params.biases,
        ),
        (
            "output weights",
            &gpu_params.output.weights,
            &cpu_params.output.weights,
        ),
        (
            "output biases",
            &gpu_params.output.biases,
            &cpu_params.output.biases,
        ),
    ] {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-5,
                "RNN {name} {index}: GPU {actual}, CPU {expected}"
            );
        }
    }
}

#[test]
fn gpu_crnn_device_forward_and_backward_match_cpu_updates() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "crnn-gpu.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\ntime_steps=2\n[convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n[crnn]\nhidden=3\noutput=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=tanh\nshortcut=1\n",
            )
            .expect("CRNN cfg should parse"),
        )
        .expect("CRNN network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Convolutional(input_conv) = &mut cpu.params[0] else {
        panic!("expected input convolution parameters");
    };
    for (index, value) in input_conv.weights.iter_mut().enumerate() {
        *value = 0.04 + index as f32 * 0.013;
    }
    input_conv.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Crnn(params) = &mut cpu.params[1] else {
        panic!("expected CRNN parameters");
    };
    for (slot, nested) in [
        &mut params.input,
        &mut params.self_params,
        &mut params.output,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, value) in nested.weights.iter_mut().enumerate() {
            *value = 0.025 + (slot * 3 + index) as f32 * 0.006;
        }
        for (index, value) in nested.biases.iter_mut().enumerate() {
            *value = (slot as f32 - 1.0) * 0.02 + index as f32 * 0.007;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..cpu.network.net.input_shape().batch * cpu.network.net.input_shape().element_count())
            .map(|index| (index as f32 * 0.23).sin() * 0.6)
            .collect(),
    )
    .expect("CRNN input should have the expected shape");
    let cpu_forward = cpu.forward(&input).expect("CPU CRNN forward");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("backend CRNN forward");
    assert_slice_close(
        "CRNN output",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("backend output")
            .data,
        2e-5,
    );
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(super::RecurrentForwardCache::Crnn(_)))
    ));

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu()
        .expect("GPU CRNN train buffers");
    gpu.zero_train_buffers_gpu()
        .expect("clear GPU CRNN train buffers");
    let cpu_training = cpu
        .forward_training(&input)
        .expect("CPU CRNN training forward");
    let gpu_training = gpu
        .forward_training_device(&input)
        .expect("backend CRNN training forward");
    assert!(gpu_training.gpu_crnn_caches[1].is_some());
    assert!(gpu_training.cpu_forward.recurrent_caches[1].is_none());
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.37).cos() * 0.8)
            .collect(),
    )
    .expect("CRNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_training, &seeds)
        .expect("CPU CRNN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_training, &seeds)
        .expect("backend CRNN backward");
    cpu.update_weights(0.03, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.03, 0.0, 0.0, 2)
        .expect("backend CRNN optimizer step");
    let LayerParams::Convolutional(cpu_input) = &cpu.params[0] else {
        unreachable!();
    };
    let LayerParams::Convolutional(gpu_input) = &gpu.params[0] else {
        unreachable!();
    };
    assert_slice_close(
        "CRNN upstream conv weights",
        &cpu_input.weights,
        &gpu_input.weights,
        4e-4,
    );
    assert_slice_close(
        "CRNN upstream conv biases",
        &cpu_input.biases,
        &gpu_input.biases,
        4e-4,
    );
    let (LayerParams::Crnn(cpu_params), LayerParams::Crnn(gpu_params)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        unreachable!();
    };
    for (name, cpu_conv, gpu_conv) in [
        ("input", &cpu_params.input, &gpu_params.input),
        (
            "recurrent",
            &cpu_params.self_params,
            &gpu_params.self_params,
        ),
        ("output", &cpu_params.output, &gpu_params.output),
    ] {
        assert_slice_close(
            &format!("CRNN {name} weights"),
            &cpu_conv.weights,
            &gpu_conv.weights,
            4e-4,
        );
        assert_slice_close(
            &format!("CRNN {name} biases"),
            &cpu_conv.biases,
            &gpu_conv.biases,
            4e-4,
        );
    }
}

#[test]
fn lstm_axis_layout_pins_cpp_gate_blocks_steps_and_samples() {
    // C++ constructs and evaluates separate `uf/ui/ug/uo` and
    // `wf/wi/wg/wo` projections, then forms the cell in f,i,g,o order and
    // writes time-major output slices (`lstm_layer.cpp:46-83,177-260`).
    let network = Network::from_config(
        ConfigFile::parse_str(
            "lstm-axis-layout.cfg",
            "[net]\nwidth=3\nheight=1\nchannels=1\nbatch=2\ntime_steps=2\n[lstm]\noutput=2\n",
        )
        .expect("LSTM axis config should parse"),
    )
    .expect("LSTM axis network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Lstm(params) = &mut executable.params[0] else {
        panic!("expected LSTM parameters");
    };
    for recurrent in [
        &mut params.wf,
        &mut params.wi,
        &mut params.wg,
        &mut params.wo,
    ] {
        recurrent.weights.fill(0.0);
        recurrent.biases.fill(0.0);
    }
    params
        .uf
        .weights
        .copy_from_slice(&[0.11, -0.07, 0.03, -0.05, 0.09, 0.02]);
    params
        .ui
        .weights
        .copy_from_slice(&[-0.03, 0.08, 0.06, 0.12, -0.04, 0.07]);
    params
        .ug
        .weights
        .copy_from_slice(&[0.09, 0.02, -0.06, -0.08, 0.05, 0.1]);
    params
        .uo
        .weights
        .copy_from_slice(&[-0.04, 0.1, 0.03, 0.06, -0.09, 0.02]);
    params.uf.biases.copy_from_slice(&[0.12, -0.08]);
    params.ui.biases.copy_from_slice(&[-0.1, 0.14]);
    params.ug.biases.copy_from_slice(&[0.05, -0.07]);
    params.uo.biases.copy_from_slice(&[0.03, 0.09]);
    let uf_weights = params.uf.weights.clone();
    let ui_weights = params.ui.weights.clone();
    let ug_weights = params.ug.weights.clone();
    let uo_weights = params.uo.weights.clone();
    let uf_biases = params.uf.biases.clone();
    let ui_biases = params.ui.biases.clone();
    let ug_biases = params.ug.biases.clone();
    let uo_biases = params.uo.biases.clone();

    let (steps, samples, inputs, hidden) = (2usize, 2usize, 3usize, 2usize);
    let mut input_values = Vec::with_capacity(steps * samples * inputs);
    for step in 0..steps {
        for sample in 0..samples {
            for feature in 0..inputs {
                input_values
                    .push(crate::axis_fixture::axis_value(step, sample, 0, feature) * 0.001 + 0.1);
            }
        }
    }
    let input = Tensor::from_data(executable.network.net.input_shape(), input_values.clone())
        .expect("LSTM axis input should have expected shape");
    let forward = executable
        .forward(&input)
        .expect("LSTM forward should succeed");
    let output = forward.final_output().expect("LSTM output should exist");

    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let project = |weights: &[f32], bias: &[f32], feature: usize, input: &[f32]| {
        bias[feature]
            + weights[feature * inputs..(feature + 1) * inputs]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
    };
    let mut expected = vec![0.0; steps * samples * hidden];
    let mut previous_cell = [[0.0f32; 2]; 2];
    for step in 0..steps {
        for sample in 0..samples {
            let src = (step * samples + sample) * inputs;
            let x = &input_values[src..src + inputs];
            for feature in 0..hidden {
                let f = sigmoid(project(&uf_weights, &uf_biases, feature, x));
                let i = sigmoid(project(&ui_weights, &ui_biases, feature, x));
                let g = project(&ug_weights, &ug_biases, feature, x).tanh();
                let o = sigmoid(project(&uo_weights, &uo_biases, feature, x));
                let cell = f * previous_cell[sample][feature] + i * g;
                let state = o * cell.tanh();
                expected[(step * samples + sample) * hidden + feature] = state;
                previous_cell[sample][feature] = cell;
            }
        }
    }
    assert_eq!(
        output.shape.batch,
        steps * samples,
        "time-major output batch"
    );
    assert_eq!(output.shape.channels, hidden);
    for (index, (&actual, &want)) in output.data.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
            "LSTM gate/step/sample axis at flat index {index}: {actual} != {want}"
        );
    }
}

#[test]
fn lstm_forward_backward_and_weight_updates_are_wired() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "lstm.cfg",
            "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[lstm]\noutput=3\n",
        )
        .expect("LSTM cfg should parse"),
    )
    .expect("LSTM network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Lstm(params) = &mut executable.params[0] else {
        panic!("expected LSTM parameters");
    };
    for nested in [
        &mut params.wf,
        &mut params.wi,
        &mut params.wg,
        &mut params.wo,
        &mut params.uf,
        &mut params.ui,
        &mut params.ug,
        &mut params.uo,
    ] {
        for value in &mut nested.weights {
            *value = 0.04;
        }
        for value in &mut nested.biases {
            *value = 0.01;
        }
    }
    let input = Tensor::filled(executable.network.net.input_shape(), 0.2);
    let forward = executable
        .forward(&input)
        .expect("LSTM forward should succeed");
    let output = forward.final_output().expect("LSTM output should exist");
    let seed = Tensor::filled(output.shape, 1.0);
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("LSTM backward should succeed");
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .all(|value| value.is_finite())
    );
    executable
        .backward_accumulate(&input, &forward, &seed)
        .expect("LSTM train backward should succeed");
    assert!(matches!(
        executable.train_buffers[0],
        LayerTrainBuffers::Lstm(_)
    ));
}

#[test]
fn gpu_lstm_no_batch_norm_forward_and_device_backward_match_cpu() {
    let network = Network::from_config(
        ConfigFile::parse_str(
            "lstm-gpu.cfg",
            "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[lstm]\noutput=3\nbatch_normalize=0\n",
        )
        .expect("LSTM cfg should parse"),
    )
    .expect("LSTM network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Lstm(params) = &mut cpu.params[0] else {
        panic!("expected LSTM parameters");
    };
    for (slot, projection) in [
        &mut params.wf,
        &mut params.wi,
        &mut params.wg,
        &mut params.wo,
        &mut params.uf,
        &mut params.ui,
        &mut params.ug,
        &mut params.uo,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.015 + (slot * 3 + index) as f32 * 0.004;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 3.5) * 0.01 + index as f32 * 0.006;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..cpu.network.net.input_shape().batch * cpu.network.net.input_shape().element_count())
            .map(|index| (index as f32 * 0.31).sin() * 0.7)
            .collect(),
    )
    .expect("LSTM input should have the expected shape");
    let cpu_forward = cpu.forward(&input).expect("CPU LSTM forward");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("backend LSTM forward");
    assert_slice_close(
        "LSTM output",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("backend output")
            .data,
        3e-5,
    );
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.first(),
        Some(Some(super::RecurrentForwardCache::Lstm(_)))
    ));

    cpu.zero_train_buffers();
    gpu.init_train_buffers_gpu().expect("LSTM train buffers");
    gpu.zero_train_buffers_gpu()
        .expect("clear LSTM train buffers");
    let cpu_training = cpu
        .forward_training(&input)
        .expect("CPU LSTM training forward");
    let gpu_training = gpu
        .forward_training_device(&input)
        .expect("backend LSTM training forward");
    assert_slice_close(
        "LSTM device-training output",
        &cpu_training.layer_outputs[0].data,
        &gpu.backend
            .download_f32(&gpu_training.gpu_layer_outputs[0])
            .expect("download device-training output"),
        3e-5,
    );
    assert!(gpu_training.gpu_lstm_caches[0].is_some());
    assert!(gpu_training.cpu_forward.recurrent_caches[0].is_none());
    let output_shape = cpu.network.layers[0].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.43).cos() * 0.6)
            .collect(),
    )
    .expect("LSTM output gradient");
    let seeds = vec![Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_training, &seeds)
        .expect("CPU LSTM backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_training, &seeds)
        .expect("LSTM device backward");
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("LSTM optimizer step");
    let (LayerParams::Lstm(cpu_params), LayerParams::Lstm(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        unreachable!();
    };
    for (name, cpu_projection, gpu_projection) in [
        ("wf", &cpu_params.wf, &gpu_params.wf),
        ("wi", &cpu_params.wi, &gpu_params.wi),
        ("wg", &cpu_params.wg, &gpu_params.wg),
        ("wo", &cpu_params.wo, &gpu_params.wo),
        ("uf", &cpu_params.uf, &gpu_params.uf),
        ("ui", &cpu_params.ui, &gpu_params.ui),
        ("ug", &cpu_params.ug, &gpu_params.ug),
        ("uo", &cpu_params.uo, &gpu_params.uo),
    ] {
        assert_slice_close(
            &format!("LSTM {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            2e-5,
        );
        assert_slice_close(
            &format!("LSTM {name} biases"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            2e-5,
        );
    }
}

#[test]
fn gpu_batch_normalized_lstm_backward_matches_cpu_and_updates_rolling_stats() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "lstm-bn-gpu.cfg",
                "[net]\ninputs=2\nbatch=2\ntime_steps=2\n[connected]\noutput=2\nactivation=linear\n[lstm]\noutput=3\nbatch_normalize=1\n",
            )
            .expect("LSTM BN cfg should parse"),
        )
        .expect("LSTM BN network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Connected(upstream) = &mut cpu.params[0] else {
        panic!("expected upstream connected parameters");
    };
    upstream.weights.copy_from_slice(&[0.31, -0.18, 0.11, 0.27]);
    upstream.biases.copy_from_slice(&[0.02, -0.035]);
    let LayerParams::Lstm(params) = &mut cpu.params[1] else {
        panic!("expected LSTM parameters");
    };
    for (slot, projection) in [
        &mut params.wf,
        &mut params.wi,
        &mut params.wg,
        &mut params.wo,
        &mut params.uf,
        &mut params.ui,
        &mut params.ug,
        &mut params.uo,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.015 + (slot * 3 + index) as f32 * 0.004;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 3.5) * 0.01 + index as f32 * 0.006;
        }
        let batch_norm = projection.batch_norm.as_mut().expect("LSTM BN params");
        for feature in 0..batch_norm.scales.len() {
            batch_norm.scales[feature] = 0.72 + feature as f32 * 0.13;
            batch_norm.rolling_mean[feature] = -0.08 + feature as f32 * 0.05;
            batch_norm.rolling_variance[feature] = 0.6 + feature as f32 * 0.17;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    gpu.init_train_buffers_gpu()
        .expect("GPU LSTM BN train buffers");
    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..8)
            .map(|index| (index as f32 * 0.29).sin() * 0.8)
            .collect(),
    )
    .expect("LSTM input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU LSTM BN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("GPU LSTM BN host-reference forward");
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(super::RecurrentForwardCache::Lstm(_)))
    ));
    assert!(gpu_forward.gpu_lstm_caches[1].is_none());
    assert_slice_close(
        "BN LSTM output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        1e-6,
    );

    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("clear GPU LSTM BN train buffers");
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.41).cos() * 0.55)
            .collect(),
    )
    .expect("LSTM output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU LSTM BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU LSTM BN backward");
    assert!(
        !gpu.gradients_non_finite()
            .expect("finite LSTM BN gradients")
    );
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("GPU LSTM BN host optimizer step");

    let (LayerParams::Connected(cpu_input), LayerParams::Connected(gpu_input)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("upstream connected parameters after update");
    };
    assert_slice_close(
        "BN LSTM upstream weights",
        &cpu_input.weights,
        &gpu_input.weights,
        3e-5,
    );
    assert_slice_close(
        "BN LSTM upstream biases",
        &cpu_input.biases,
        &gpu_input.biases,
        3e-5,
    );
    let (LayerParams::Lstm(cpu_lstm), LayerParams::Lstm(gpu_lstm)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("LSTM parameters after update");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("wf", &cpu_lstm.wf, &gpu_lstm.wf),
        ("wi", &cpu_lstm.wi, &gpu_lstm.wi),
        ("wg", &cpu_lstm.wg, &gpu_lstm.wg),
        ("wo", &cpu_lstm.wo, &gpu_lstm.wo),
        ("uf", &cpu_lstm.uf, &gpu_lstm.uf),
        ("ui", &cpu_lstm.ui, &gpu_lstm.ui),
        ("ug", &cpu_lstm.ug, &gpu_lstm.ug),
        ("uo", &cpu_lstm.uo, &gpu_lstm.uo),
    ] {
        assert_slice_close(
            &format!("BN LSTM {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            3e-5,
        );
        assert_slice_close(
            &format!("BN LSTM {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            3e-5,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        assert_slice_close(
            &format!("BN LSTM {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            3e-5,
        );
        assert_slice_close(
            &format!("BN LSTM {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            3e-5,
        );
        assert_slice_close(
            &format!("BN LSTM {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            3e-5,
        );
    }
    {
        let mut buffers = gpu.cpu_fallback_train_buffers.borrow_mut();
        let LayerTrainBuffers::Lstm(train) = &mut buffers[1] else {
            panic!("LSTM host accumulators");
        };
        train.wf.batch_norm.as_mut().unwrap().scale_updates[0] = f32::INFINITY;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("LSTM BN host scale overflow check")
    );
}

#[test]
fn gpu_batch_normalized_crnn_backward_matches_cpu_and_updates_rolling_stats() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "crnn-bn-gpu.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\ntime_steps=2\n[convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n[crnn]\nhidden=3\noutput=2\nsize=3\nstride=1\npad=1\nbatch_normalize=1\nactivation=tanh\nshortcut=1\n",
            )
            .expect("CRNN BN cfg should parse"),
        )
        .expect("CRNN BN network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let LayerParams::Convolutional(upstream) = &mut cpu.params[0] else {
        panic!("expected upstream convolution parameters");
    };
    for (index, weight) in upstream.weights.iter_mut().enumerate() {
        *weight = 0.04 + index as f32 * 0.013;
    }
    upstream.biases.copy_from_slice(&[0.015, -0.025]);
    let LayerParams::Crnn(params) = &mut cpu.params[1] else {
        panic!("expected CRNN parameters");
    };
    for (slot, projection) in [
        &mut params.input,
        &mut params.self_params,
        &mut params.output,
    ]
    .into_iter()
    .enumerate()
    {
        for (index, weight) in projection.weights.iter_mut().enumerate() {
            *weight = 0.03 + (slot * 7 + index) as f32 * 0.005;
        }
        for (index, bias) in projection.biases.iter_mut().enumerate() {
            *bias = (slot as f32 - 1.0) * 0.025 + index as f32 * 0.01;
        }
        let batch_norm = projection.batch_norm.as_mut().expect("CRNN BN params");
        for feature in 0..batch_norm.scales.len() {
            batch_norm.scales[feature] = 0.7 + feature as f32 * 0.14;
            batch_norm.rolling_mean[feature] = -0.1 + feature as f32 * 0.06;
            batch_norm.rolling_variance[feature] = 0.58 + feature as f32 * 0.16;
        }
    }
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    gpu.params = cpu.params.clone();
    assert!(!gpu.requires_cpu_backward_fallback());
    gpu.init_train_buffers_gpu()
        .expect("GPU CRNN BN train buffers");
    let input_shape = cpu.network.net.input_shape();
    let input = Tensor::from_data(
        input_shape,
        (0..input_shape.batch * input_shape.element_count())
            .map(|index| (index as f32 * 0.23).sin() * 0.75)
            .collect(),
    )
    .expect("CRNN input");
    let cpu_forward = cpu.forward_training(&input).expect("CPU CRNN BN forward");
    let gpu_forward = gpu
        .forward_training_device(&input)
        .expect("GPU CRNN BN native recurrent forward");
    assert!(matches!(
        gpu_forward.cpu_forward.recurrent_caches.get(1),
        Some(Some(super::RecurrentForwardCache::Crnn(_)))
    ));
    assert!(gpu_forward.gpu_crnn_caches[1].is_some());
    assert_slice_close(
        "BN CRNN output",
        &cpu_forward.layer_outputs[1].data,
        &gpu_forward.cpu_forward.layer_outputs[1].data,
        1e-6,
    );

    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu()
        .expect("clear GPU CRNN BN train buffers");
    let output_shape = cpu.network.layers[1].output_shape;
    let seed = Tensor::from_data(
        output_shape,
        (0..output_shape.batch * output_shape.element_count())
            .map(|index| (index as f32 * 0.37).cos() * 0.6)
            .collect(),
    )
    .expect("CRNN output gradient");
    let seeds = vec![None, Some(seed)];
    cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds)
        .expect("CPU CRNN BN backward");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds)
        .expect("GPU CRNN BN backward");
    assert!(
        !gpu.gradients_non_finite()
            .expect("finite CRNN BN gradients")
    );
    cpu.update_weights(0.02, 0.0, 0.0, 2);
    gpu.update_weights_gpu(0.02, 0.0, 0.0, 2)
        .expect("GPU CRNN BN host optimizer step");

    let (LayerParams::Convolutional(cpu_upstream), LayerParams::Convolutional(gpu_upstream)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("upstream convolution parameters after update");
    };
    assert_slice_close(
        "BN CRNN upstream weights",
        &cpu_upstream.weights,
        &gpu_upstream.weights,
        4e-5,
    );
    assert_slice_close(
        "BN CRNN upstream biases",
        &cpu_upstream.biases,
        &gpu_upstream.biases,
        4e-5,
    );
    let (LayerParams::Crnn(cpu_crnn), LayerParams::Crnn(gpu_crnn)) =
        (&cpu.params[1], &gpu.params[1])
    else {
        panic!("CRNN parameters after update");
    };
    for (name, cpu_projection, gpu_projection) in [
        ("input", &cpu_crnn.input, &gpu_crnn.input),
        ("recurrent", &cpu_crnn.self_params, &gpu_crnn.self_params),
        ("output", &cpu_crnn.output, &gpu_crnn.output),
    ] {
        assert_slice_close(
            &format!("BN CRNN {name} weights"),
            &cpu_projection.weights,
            &gpu_projection.weights,
            4e-5,
        );
        assert_slice_close(
            &format!("BN CRNN {name} beta"),
            &cpu_projection.biases,
            &gpu_projection.biases,
            4e-5,
        );
        let cpu_bn = cpu_projection.batch_norm.as_ref().unwrap();
        let gpu_bn = gpu_projection.batch_norm.as_ref().unwrap();
        assert_slice_close(
            &format!("BN CRNN {name} scales"),
            &cpu_bn.scales,
            &gpu_bn.scales,
            4e-5,
        );
        assert_slice_close(
            &format!("BN CRNN {name} rolling mean"),
            &cpu_bn.rolling_mean,
            &gpu_bn.rolling_mean,
            4e-5,
        );
        assert_slice_close(
            &format!("BN CRNN {name} rolling variance"),
            &cpu_bn.rolling_variance,
            &gpu_bn.rolling_variance,
            4e-5,
        );
    }
    {
        let mut buffers = gpu.cpu_fallback_train_buffers.borrow_mut();
        let LayerTrainBuffers::Crnn(train) = &mut buffers[1] else {
            panic!("CRNN host accumulators");
        };
        train.input.batch_norm.as_mut().unwrap().scale_updates[0] = f32::INFINITY;
    }
    assert!(
        gpu.gradients_non_finite()
            .expect("CRNN BN host scale overflow check")
    );
}

#[test]
fn crnn_axis_layout_pins_cpp_spatial_step_and_sample_order() {
    // C++ runs input/self/output convolutions per time step, adds the
    // per-spatial hidden state, advances the input slice, and appends an
    // output convolution slice (`crnn_layer.cpp:165-218`).
    let network = Network::from_config(
            ConfigFile::parse_str(
                "crnn-axis-layout.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=1\nbatch=2\ntime_steps=2\n[crnn]\nhidden=2\noutput=2\nsize=1\nstride=1\npad=0\nactivation=linear\nshortcut=0\n",
            )
            .expect("CRNN axis config should parse"),
        )
        .expect("CRNN axis network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Crnn(params) = &mut executable.params[0] else {
        panic!("expected CRNN parameters");
    };
    params.input.weights.copy_from_slice(&[0.5, -0.25]);
    params.input.biases.copy_from_slice(&[0.1, -0.05]);
    params
        .self_params
        .weights
        .copy_from_slice(&[0.3, 0.1, -0.2, 0.4]);
    params.self_params.biases.copy_from_slice(&[0.02, -0.03]);
    params
        .output
        .weights
        .copy_from_slice(&[1.0, -0.5, 0.25, 0.75]);
    params.output.biases.copy_from_slice(&[-0.01, 0.04]);
    let input_weights = params.input.weights.clone();
    let input_biases = params.input.biases.clone();
    let self_weights = params.self_params.weights.clone();
    let self_biases = params.self_params.biases.clone();
    let output_weights = params.output.weights.clone();
    let output_biases = params.output.biases.clone();

    let (steps, samples, height, width) = (2usize, 2usize, 2usize, 3usize);
    let area = height * width;
    let mut input_values = Vec::with_capacity(steps * samples * area);
    for step in 0..steps {
        for sample in 0..samples {
            for y in 0..height {
                for x in 0..width {
                    input_values
                        .push(crate::axis_fixture::axis_value(step, sample, y, x) * 0.001 + 0.01);
                }
            }
        }
    }
    let input = Tensor::from_data(executable.network.net.input_shape(), input_values.clone())
        .expect("CRNN axis input should have expected shape");
    let forward = executable
        .forward(&input)
        .expect("CRNN forward should succeed");
    let output = forward.final_output().expect("CRNN output should exist");

    let mut expected = vec![0.0; steps * samples * 2 * area];
    let mut previous = vec![[0.0f32; 2]; samples * area];
    for step in 0..steps {
        for sample in 0..samples {
            for spatial in 0..area {
                let input_index = (step * samples + sample) * area + spatial;
                let value = input_values[input_index];
                let old = previous[sample * area + spatial];
                let state = [
                    input_weights[0] * value
                        + input_biases[0]
                        + self_weights[0] * old[0]
                        + self_weights[1] * old[1]
                        + self_biases[0],
                    input_weights[1] * value
                        + input_biases[1]
                        + self_weights[2] * old[0]
                        + self_weights[3] * old[1]
                        + self_biases[1],
                ];
                let dst = (step * samples + sample) * 2 * area + spatial;
                expected[dst] =
                    output_weights[0] * state[0] + output_weights[1] * state[1] + output_biases[0];
                expected[dst + area] =
                    output_weights[2] * state[0] + output_weights[3] * state[1] + output_biases[1];
                previous[sample * area + spatial] = state;
            }
        }
    }
    assert_eq!(
        output.shape.batch,
        steps * samples,
        "time-major output batch"
    );
    assert_eq!(
        (
            output.shape.channels,
            output.shape.height,
            output.shape.width
        ),
        (2, height, width)
    );
    for (index, (&actual, &want)) in output.data.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
            "CRNN time/sample/channel/spatial axis at flat index {index}: {actual} != {want}"
        );
    }
}

#[test]
fn crnn_forward_backward_and_weight_updates_are_wired() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "crnn.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=1\nbatch=2\ntime_steps=2\n[crnn]\nsize=1\nhidden=2\noutput=1\nactivation=linear\n",
            )
            .expect("CRNN cfg should parse"),
        )
        .expect("CRNN network should build");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Crnn(params) = &mut executable.params[0] else {
        panic!("expected CRNN parameters");
    };
    for nested in [
        &mut params.input,
        &mut params.self_params,
        &mut params.output,
    ] {
        for value in &mut nested.weights {
            *value = 0.1;
        }
        for value in &mut nested.biases {
            *value = 0.01;
        }
    }
    let input = Tensor::filled(executable.network.net.input_shape(), 0.2);
    let forward = executable
        .forward(&input)
        .expect("CRNN forward should succeed");
    let output = forward.final_output().expect("CRNN output should exist");
    let seed = Tensor::filled(output.shape, 1.0);
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("CRNN backward should succeed");
    assert!(
        backward
            .input_gradient
            .data
            .iter()
            .all(|value| value.is_finite())
    );
    executable
        .backward_accumulate(&input, &forward, &seed)
        .expect("CRNN train backward should succeed");
    assert!(matches!(
        executable.train_buffers[0],
        LayerTrainBuffers::Crnn(_)
    ));
}

/// `[laya_marker_gather]` on the CPU.  Layer 0 is an identity 1x1 conv, so
/// input A is exactly the network input; layer 1 gathers one hidden column
/// per option at the position read from input channel 0, column 0.
pub(super) fn laya_marker_gather_network() -> ExecutableNetwork {
    let cfg = "[net]\nchannels=6\nbatch=1\nheight=1\nwidth=3\n\
                   [convolutional]\nfilters=6\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [laya_marker_gather]\nlayers=0,0\nhidden=6\nmax_options=1\n";
    let mut executable = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("laya-mg.cfg", cfg).expect("cfg"))
            .expect("network"),
    );
    let LayerParams::Convolutional(params) = &mut executable.params[0] else {
        panic!("expected conv params");
    };
    let mut weights = vec![0.0f32; 36];
    for f in 0..6 {
        weights[f * 6 + f] = 1.0;
    }
    params.weights = weights;
    params.biases = vec![0.0; 6];
    executable
}

/// `[b][c][h][w]` = 6 channels x 1 x 3.  Channel 0 column 0 is the marker
/// position (2.0 -> column 2); every channel carries a distinct payload.
pub(super) fn laya_marker_gather_input(executable: &ExecutableNetwork) -> Tensor {
    let channels = [
        [2.0f32, 0.0, 0.0],
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
        [40.0, 41.0, 42.0],
        [50.0, 51.0, 52.0],
    ];
    let mut data = vec![0.0f32; 18];
    for (c, row) in channels.iter().enumerate() {
        for (t, value) in row.iter().enumerate() {
            data[c * 3 + t] = *value;
        }
    }
    Tensor::from_data(executable.network.net.input_shape(), data).expect("input shape")
}

#[test]
fn cpu_laya_marker_gather_selects_the_marker_column() {
    let executable = laya_marker_gather_network();
    let input = laya_marker_gather_input(&executable);
    let forward = executable.forward(&input).expect("forward");
    let output = &forward.layer_outputs[1];
    // Column 2 of every channel, in channel order.
    assert_eq!(output.data, vec![0.0, 12.0, 22.0, 32.0, 42.0, 52.0]);
}

#[test]
fn cpu_laya_marker_gather_backward_scatters_into_input_a() {
    let executable = laya_marker_gather_network();
    let input = laya_marker_gather_input(&executable);
    let forward = executable.forward(&input).expect("forward");
    let seed = Tensor::from_data(
        forward.layer_outputs[1].shape,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .expect("seed");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("backward");
    // delta_A[c, 2] = seed[c]; every other column stays zero.
    let mut want = vec![0.0f32; 18];
    for c in 0..6 {
        want[c * 3 + 2] = (c as f32) + 1.0;
    }
    assert_eq!(backward.layer_output_gradients[0].data, want);
}

#[test]
fn laya_marker_gather_rejects_shape_mismatches() {
    let net = "[net]\nchannels=6\nbatch=1\nheight=1\nwidth=3\n\
                   [convolutional]\nfilters=6\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n";
    let build = |section: &str| {
        ConfigFile::parse_str("laya-mg.cfg", &format!("{net}{section}"))
            .map_err(|err| err.to_string())
            .and_then(|cfg| {
                Network::from_config(cfg)
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            })
    };
    // hidden must equal input A's channels (6).
    assert!(build("[laya_marker_gather]\nlayers=0,0\nhidden=5\nmax_options=1\n").is_err());
    // input B (6 channels) must have >= 2*max_options channels.
    assert!(build("[laya_marker_gather]\nlayers=0,0\nhidden=6\nmax_options=4\n").is_err());
    // exactly two sources.
    assert!(build("[laya_marker_gather]\nlayers=0\nhidden=6\nmax_options=1\n").is_err());
    // the valid shape builds.
    assert!(build("[laya_marker_gather]\nlayers=0,0\nhidden=6\nmax_options=1\n").is_ok());
}

/// `[laya_act_head]` on the CPU.  Three 1x1 convs produce the head (identity),
/// marker (all-ones) and scorer (input channel 0) sources, then the act head.
pub(super) fn laya_act_head_network() -> ExecutableNetwork {
    let cfg = "[net]\nchannels=6\nbatch=1\nheight=1\nwidth=2\n\
                   [convolutional]\nfilters=6\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [convolutional]\nfilters=1\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [convolutional]\nfilters=4\nsize=1\nstride=1\nbatch_normalize=0\nactivation=linear\n\
                   [laya_act_head]\nlayers=0,1,2\nhidden=6\nmax_options=2\nfc_dim=3\nn_act=2\n";
    let mut executable = ExecutableNetwork::new(
        Network::from_config(ConfigFile::parse_str("laya-ah.cfg", cfg).expect("cfg"))
            .expect("network"),
    );
    // Head source: identity 6x6 1x1 conv.
    let LayerParams::Convolutional(p) = &mut executable.params[0] else {
        panic!("expected conv params")
    };
    let mut w = vec![0.0f32; 36];
    for f in 0..6 {
        w[f * 6 + f] = 1.0;
    }
    p.weights = w;
    p.biases = vec![0.0; 6];
    // Scorer source: read head channel 0 at each column.
    let LayerParams::Convolutional(p) = &mut executable.params[1] else {
        panic!("expected conv params")
    };
    let mut w = vec![0.0f32; 6];
    w[0] = 1.0;
    p.weights = w;
    p.biases = vec![0.0; 1];
    // Marker source: zero weights, bias 1 -> every mask > 0.5 (all valid).
    let LayerParams::Convolutional(p) = &mut executable.params[2] else {
        panic!("expected conv params")
    };
    p.weights = vec![0.0; 4];
    p.biases = vec![1.0; 4];
    // Deterministic act-head weights.
    let LayerParams::LayaActHead(p) = &mut executable.params[3] else {
        panic!("expected laya_act_head params")
    };
    let input = 6 + crate::laya_act_head::LAYA_ACT_HEAD_N_FEATS;
    let mut next = 0.0f32;
    let mut fill = |len: usize| {
        (0..len)
            .map(|_| {
                next += 0.013;
                (next * 0.7).sin() * 0.4
            })
            .collect::<Vec<f32>>()
    };
    p.fc1_w = fill(3 * input);
    p.fc1_b = fill(3);
    p.fc2_w = fill(2 * 3);
    p.fc2_b = fill(2);
    executable
}

pub(super) fn laya_act_head_input(executable: &ExecutableNetwork) -> Tensor {
    // [b][c][h][w]: channels=6, w=2. Channel 0 carries the scorer logits
    // [0.5, 2.0]; the rest are the CLS payload.
    let mut data = vec![0.0f32; 12];
    data[0] = 0.5;
    data[1] = 2.0;
    for c in 1..6 {
        data[c * 2] = c as f32 * 0.1;
        data[c * 2 + 1] = -(c as f32) * 0.2;
    }
    Tensor::from_data(executable.network.net.input_shape(), data).expect("input shape")
}

#[test]
fn cpu_laya_act_head_forward_matches_the_module() {
    let executable = laya_act_head_network();
    let input = laya_act_head_input(&executable);
    let forward = executable.forward(&input).expect("forward");
    let LayerParams::LayaActHead(weights) = &executable.params[3] else {
        panic!("expected act_head params")
    };
    let head = &forward.layer_outputs[0];
    let scorer = &forward.layer_outputs[1];
    let marker = &forward.layer_outputs[2];
    let want = crate::laya_act_head::act_head_forward(
        &head.data,
        &scorer.data,
        &marker.data,
        weights,
        1,
        6,
        2,
        marker.shape.channels,
        head.shape.width,
        marker.shape.width,
        3,
        2,
    )
    .expect("module forward");
    assert_eq!(forward.layer_outputs[3].data, want);
}

#[test]
fn cpu_laya_act_head_backward_scatters_dcls_into_the_head_source() {
    let executable = laya_act_head_network();
    let input = laya_act_head_input(&executable);
    let forward = executable.forward(&input).expect("forward");
    let seed = Tensor::from_data(forward.layer_outputs[3].shape, vec![1.0, -0.5]).expect("seed");
    let LayerParams::LayaActHead(weights) = &executable.params[3] else {
        panic!("expected act_head params")
    };
    let head = &forward.layer_outputs[0];
    let scorer = &forward.layer_outputs[1];
    let marker = &forward.layer_outputs[2];
    let grad = crate::laya_act_head::act_head_backward(
        &head.data,
        &scorer.data,
        &marker.data,
        weights,
        &seed.data,
        1,
        6,
        2,
        marker.shape.channels,
        head.shape.width,
        marker.shape.width,
        3,
        2,
    )
    .expect("module backward");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("backward");
    let head_grad = &backward.layer_output_gradients[0];
    // Column 0 gets the pooled-CLS gradient; column 1 stays zero.
    for e in 0..6 {
        assert!(
            (head_grad.data[e * 2] - grad.dcls[e]).abs() < 1e-6,
            "channel {e}: {} vs {}",
            head_grad.data[e * 2],
            grad.dcls[e]
        );
        assert!(
            head_grad.data[e * 2 + 1].abs() < 1e-6,
            "column 1 must stay zero"
        );
    }
}
