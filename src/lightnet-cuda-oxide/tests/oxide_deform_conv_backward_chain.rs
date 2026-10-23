//! `[deform_conv]` composed backward vs the **real CPU backward** oracle.
//!
//! Builds an `ExecutableNetwork` from cfg, runs the CPU training forward and
//! `backward_accumulate_seeded`, then compares the layer's parameter gradients
//! with the four-stage reference in `lightnet_cuda::host_ops::deform_conv`.
//! With `batch_normalize=0` and `activation=linear`, the layer's output gradient
//! is the seed itself.
//!
//! No compiled-C++ golden exists: this is the Rust CPU backward vs the
//! decomposed reference.

use lightnet_core::cfg::ConfigFile;
use lightnet_core::{
    ExecutableNetwork, LayerParams, LayerTrainBuffers, Network, Tensor,
};
use lightnet_cuda::host_ops::deform_conv as host;

const TOL: f32 = 2e-4;

const UNMASKED: &str = "\
[net]
width=6
height=4
channels=1
batch=1

[deform_conv]
filters=2
size=3
stride=1
pad=1
groups=1
batch_normalize=0
activation=linear
mask=0
";

const MASKED: &str = "\
[net]
width=5
height=3
channels=2
batch=1

[deform_conv]
filters=3
size=3
stride=1
pad=1
groups=1
batch_normalize=0
activation=linear
mask=1
";

fn build(cfg_text: &str) -> ExecutableNetwork {
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg_text).unwrap())
        .expect("parse cfg");
    ExecutableNetwork::new(network)
}

fn assert_close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        assert!(
            err <= TOL + TOL * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

fn check(cfg_text: &str, label: &str) {
    let mut cpu = build(cfg_text);
    let input_shape = cpu.network.net.input_shape();
    let data: Vec<f32> = (0..input_shape.element_count())
        .map(|i| ((i * 7 % 17) as f32) * 0.125 - 1.0)
        .collect();
    let input = Tensor::from_data(input_shape, data).unwrap();

    let layer_index = cpu.network.layers.len() - 1;
    let out_shape = cpu.network.layers[layer_index].output_shape;
    let in_shape = cpu.network.layers[layer_index].input_shape;
    let spec = match &cpu.network.layers[layer_index].kind {
        lightnet_core::LayerKind::DeformConv(spec) => spec.clone(),
        _ => panic!("{label}: last layer is not [deform_conv]"),
    };
    let seed_data: Vec<f32> = (0..out_shape.element_count())
        .map(|i| 0.25 + ((i % 5) as f32) * 0.125)
        .collect();
    let seed = Tensor::from_data(out_shape, seed_data.clone()).unwrap();

    let forward = cpu.forward_training(&input).expect("cpu forward");
    let mut seeds = vec![None; cpu.network.layers.len()];
    seeds[layer_index] = Some(seed);
    cpu.backward_accumulate_seeded(&input, &forward, &seeds)
        .expect("cpu backward");

    let LayerParams::DeformConv(params) = &cpu.params[layer_index] else {
        panic!("{label}: not a deform layer");
    };
    let geometry = host::DeformGeometry {
        batch: in_shape.batch,
        channels: in_shape.channels,
        filters: spec.filters,
        channels_per_group: in_shape.channels / spec.groups,
        filters_per_group: spec.filters / spec.groups,
        input_height: in_shape.height,
        input_width: in_shape.width,
        output_height: out_shape.height,
        output_width: out_shape.width,
        size: spec.size,
        stride_x: spec.stride_x,
        stride_y: spec.stride_y,
        padding: spec.padding,
        dilation: spec.dilation,
    };
    // The gate threshold the layer itself uses.
    let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
    let fwd = host::forward(
        &input.data,
        &params.main.weights,
        &params.offset_weights,
        &params.offset_biases,
        params.mask_weights.as_deref(),
        params.mask_biases.as_deref(),
        &geometry,
        max_offset,
    )
    .expect("host forward");
    let grads = host::backward(
        &input.data,
        &fwd,
        &params.main.weights,
        &params.offset_weights,
        params.mask_weights.as_deref(),
        &seed_data,
        &geometry,
        max_offset,
    )
    .expect("host backward");

    let LayerTrainBuffers::DeformConv(train) = &cpu.train_buffers[layer_index] else {
        panic!("{label}: not deform train buffers");
    };
    assert_close(
        &format!("{label}/main_w"),
        &train.main.weight_updates,
        &grads.d_main_weights,
    );
    assert_close(
        &format!("{label}/off_w"),
        &train.offset_weight_updates,
        &grads.d_offset_weights,
    );
    assert_close(
        &format!("{label}/off_b"),
        &train.offset_bias_updates,
        &grads.d_offset_biases,
    );
    match (&train.mask_weight_updates, &train.mask_bias_updates) {
        (Some(weights), Some(biases)) => {
            assert_close(&format!("{label}/mask_w"), weights, &grads.d_mask_weights);
            assert_close(&format!("{label}/mask_b"), biases, &grads.d_mask_biases);
        }
        (None, None) => assert!(grads.d_mask_weights.is_empty()),
        _ => panic!("{label}: inconsistent mask train buffers"),
    }
}

#[test]
fn composed_backward_matches_the_cpu_runtime_without_masks() {
    check(UNMASKED, "unmasked");
}

#[test]
fn composed_backward_matches_the_cpu_runtime_with_masks() {
    check(MASKED, "masked");
}

// ---------------------------------------------------------------------------
// Native backward composition vs the CPU-gradient oracle
// ---------------------------------------------------------------------------

/// Runs the native device backward on the layer's real parameters and compares
/// every gradient with the CPU backward captured by the test above.
#[test]
fn native_backward_composition_matches_the_cpu_gradients() {
    use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
    use lightnet_cuda_oxide::OxideBackend;

    for (cfg_text, label) in [(UNMASKED, "unmasked"), (MASKED, "masked")] {
        let mut cpu = build(cfg_text);
        let input_shape = cpu.network.net.input_shape();
        let data: Vec<f32> = (0..input_shape.element_count())
            .map(|i| ((i * 7 % 17) as f32) * 0.125 - 1.0)
            .collect();
        let input = Tensor::from_data(input_shape, data).unwrap();
        let layer_index = cpu.network.layers.len() - 1;
        let out_shape = cpu.network.layers[layer_index].output_shape;
        let in_shape = cpu.network.layers[layer_index].input_shape;
        let spec = match &cpu.network.layers[layer_index].kind {
            lightnet_core::LayerKind::DeformConv(spec) => spec.clone(),
            _ => panic!("{label}: not a deform layer"),
        };
        let seed_data: Vec<f32> = (0..out_shape.element_count())
            .map(|i| 0.25 + ((i % 5) as f32) * 0.125)
            .collect();
        let seed = Tensor::from_data(out_shape, seed_data.clone()).unwrap();
        let forward = cpu.forward_training(&input).expect("cpu forward");
        let mut seeds = vec![None; cpu.network.layers.len()];
        seeds[layer_index] = Some(seed);
        cpu.backward_accumulate_seeded(&input, &forward, &seeds)
            .expect("cpu backward");

        let LayerParams::DeformConv(params) = &cpu.params[layer_index] else {
            panic!("{label}: params");
        };
        let geometry = host::DeformGeometry {
            batch: in_shape.batch,
            channels: in_shape.channels,
            filters: spec.filters,
            channels_per_group: in_shape.channels / spec.groups,
            filters_per_group: spec.filters / spec.groups,
            input_height: in_shape.height,
            input_width: in_shape.width,
            output_height: out_shape.height,
            output_width: out_shape.width,
            size: spec.size,
            stride_x: spec.stride_x,
            stride_y: spec.stride_y,
            padding: spec.padding,
            dilation: spec.dilation,
        };
        let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
        let fwd = host::forward(
            &input.data,
            &params.main.weights,
            &params.offset_weights,
            &params.offset_biases,
            params.mask_weights.as_deref(),
            params.mask_biases.as_deref(),
            &geometry,
            max_offset,
        )
        .expect("host forward");

        let backend = OxideBackend::new().expect("oxide backend");
        let cfg = ElementwiseLaunchConfig::default();
        let upload = |values: &[f32]| backend.upload_f32(&[values.len()], values).unwrap();
        let t_input_s = backend
            .upload_f32(&[fwd.input_sanitized.len()], &fwd.input_sanitized)
            .unwrap();
        let t_raw_input = upload(&input.data);
        let t_main = upload(&params.main.weights);
        let t_off_w = upload(&params.offset_weights);
        let t_raw_offsets = upload(&fwd.raw_offsets);
        let t_mean = upload(&fwd.stats.mean);
        let t_inv = upload(&fwd.stats.inv_std);
        let t_std = upload(&fwd.stats.stddev);
        let t_norm = upload(&fwd.stats.normalize);
        let t_offsets = upload(&fwd.offsets);
        let t_seed = upload(&seed_data);
        let t_masks = fwd.masks.as_ref().map(|m| upload(m));
        let t_mask_w = params.mask_weights.as_ref().map(|w| upload(w));
        let t_mask_logits = fwd.mask_logits.as_ref().map(|l| upload(l));

        let (d_input, d_main, d_off_w, d_off_b, d_mask_w, d_mask_b) = backend
            .deform_conv_backward_native(
                &t_input_s,
                &t_raw_input,
                &t_main,
                &t_off_w,
                t_mask_w.as_ref(),
                &t_raw_offsets,
                &t_mean,
                &t_inv,
                &t_std,
                &t_norm,
                &t_offsets,
                t_masks.as_ref(),
                t_mask_logits.as_ref(),
                &t_seed,
                geometry,
                max_offset,
                cfg,
            )
            .expect("native backward");

        let LayerTrainBuffers::DeformConv(train) = &cpu.train_buffers[layer_index] else {
            panic!("{label}: train buffers");
        };
        let tol = 2e-4;
        let close = |name: &str, got: Vec<f32>, want: &[f32]| {
            assert_eq!(got.len(), want.len(), "{label}/{name}: length");
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                let err = (g - w).abs();
                assert!(
                    err <= tol + tol * w.abs(),
                    "{label}/{name}[{i}]: got {g}, want {w}, err {err}"
                );
            }
        };
        close(
            "main_w",
            backend.download_f32(&d_main).unwrap(),
            &train.main.weight_updates,
        );
        close(
            "off_w",
            backend.download_f32(&d_off_w).unwrap(),
            &train.offset_weight_updates,
        );
        close(
            "off_b",
            backend.download_f32(&d_off_b).unwrap(),
            &train.offset_bias_updates,
        );
        close(
            "d_input",
            backend.download_f32(&d_input).unwrap(),
            // d_input is not in the CPU train buffers; compare with the FD-exact
            // host composition instead.
            &host::backward(
                &input.data,
                &fwd,
                &params.main.weights,
                &params.offset_weights,
                params.mask_weights.as_deref(),
                &seed_data,
                &geometry,
                max_offset,
            )
            .unwrap()
            .d_input,
        );
        if let (Some(w), Some(b)) = (d_mask_w, d_mask_b) {
            close(
                "mask_w",
                backend.download_f32(&w).unwrap(),
                train.mask_weight_updates.as_deref().unwrap(),
            );
            close(
                "mask_b",
                backend.download_f32(&b).unwrap(),
                train.mask_bias_updates.as_deref().unwrap(),
            );
        }
    }
}
