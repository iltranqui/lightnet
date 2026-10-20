//! `[deform_conv]` composed native forward vs the **real CPU runtime** oracle.
//!
//! Unlike the stage-by-stage fixtures in `oxide_deform_envelope`, this builds an
//! actual `ExecutableNetwork` from a cfg, runs its CPU forward, and compares the
//! deform layer's output (after the linear epilogue) with
//! `OxideBackend::deform_conv_forward_native`. Weights come from the runtime's
//! own deterministic initialization, so the comparison covers real parameter
//! layout as well as the math.
//!
//! No compiled-C++ golden exists: this is device vs the Rust CPU reference.

use darknet_core::cfg::ConfigFile;
use darknet_core::{ExecutableNetwork, LayerParams, Network, Tensor};
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

/// The composed path runs the predictor through a GEMM whose summation order
/// differs from the CPU loop, so allow a few ulp of headroom.
const TOL: f32 = 1e-4;

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn build(cfg_text: &str) -> ExecutableNetwork {
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", cfg_text).unwrap())
        .expect("parse cfg");
    ExecutableNetwork::new(network)
}

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

fn cpu_layer_output(cpu: &ExecutableNetwork, input: &Tensor, label: &str) -> Vec<f32> {
    let forward = cpu.forward(input).expect("cpu forward");
    let output = forward.layer_outputs.last().expect("layer output");
    assert_eq!(
        output.shape,
        cpu.network.layers.last().unwrap().output_shape,
        "{label}: output shape"
    );
    output.data.clone()
}

fn native_raw(cpu: &ExecutableNetwork, input: &Tensor, label: &str) -> Vec<f32> {
    let backend = backend();
    let layer = cpu.network.layers.last().expect("deform layer");
    let darknet_core::LayerKind::DeformConv(spec) = &layer.kind else {
        panic!("{label}: last layer is not [deform_conv]");
    };
    let LayerParams::DeformConv(params) = &cpu.params[layer.index] else {
        panic!("{label}: layer params are not DeformConv");
    };
    let shape = layer.input_shape;
    let out = layer.output_shape;
    let t_in = backend
        .upload_f32(
            &[shape.batch, shape.channels, shape.height, shape.width],
            &input.data,
        )
        .unwrap();
    let t_main = backend
        .upload_f32(&[params.main.weights.len()], &params.main.weights)
        .unwrap();
    let t_off_w = backend
        .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
        .unwrap();
    let t_off_b = backend
        .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
        .unwrap();
    let mask_w = params
        .mask_weights
        .as_ref()
        .map(|w| backend.upload_f32(&[w.len()], w).unwrap());
    let mask_b = params
        .mask_biases
        .as_ref()
        .map(|b| backend.upload_f32(&[b.len()], b).unwrap());
    let masks = match (mask_w.as_ref(), mask_b.as_ref()) {
        (Some(w), Some(b)) => Some((w, b)),
        _ => None,
    };
    let raw = backend
        .deform_conv_forward_native(
            &t_in,
            &t_main,
            &t_off_w,
            &t_off_b,
            masks,
            shape.channels,
            spec.filters,
            spec.groups,
            shape.height,
            shape.width,
            out.height,
            out.width,
            spec.size,
            spec.stride_x,
            spec.stride_y,
            spec.padding,
            spec.dilation,
            cfg(),
        )
        .unwrap();
    // The layer's own epilogue is bias + BN + activation; these cfgs pin it to
    // bias + linear, so add the bias here and nothing else.
    assert!(
        params.main.batch_norm.is_none(),
        "{label}: fixture must not use batch norm"
    );
    let t_bias = backend
        .upload_f32(&[params.main.biases.len()], &params.main.biases)
        .unwrap();
    let biased = backend
        .bias_add_f32(&raw, &t_bias, spec.filters, out.height * out.width, cfg())
        .unwrap();
    backend.download_f32(&biased).unwrap()
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

#[test]
fn native_forward_matches_the_cpu_runtime_without_masks() {
    let cpu = build(UNMASKED);
    let shape = cpu.network.net.input_shape();
    let data: Vec<f32> = (0..shape.element_count())
        .map(|i| ((i * 7 % 17) as f32) * 0.125 - 1.0)
        .collect();
    let input = Tensor::from_data(shape, data).unwrap();
    let want = cpu_layer_output(&cpu, &input, "unmasked");
    let got = native_raw(&cpu, &input, "unmasked");
    assert_close("unmasked", &got, &want);
}

#[test]
fn native_forward_matches_the_cpu_runtime_with_masks() {
    let cpu = build(MASKED);
    let shape = cpu.network.net.input_shape();
    let data: Vec<f32> = (0..shape.element_count())
        .map(|i| ((i * 5 % 13) as f32) * 0.25 - 1.5)
        .collect();
    let input = Tensor::from_data(shape, data).unwrap();
    let want = cpu_layer_output(&cpu, &input, "masked");
    let got = native_raw(&cpu, &input, "masked");
    assert_close("masked", &got, &want);
}

#[test]
fn dilated_layer_native_forward_matches_cpu_runtime() {
    let cfg_text = UNMASKED.replace("stride=1", "stride=1\ndilation=2");
    let cpu = build(&cfg_text);
    let layer = cpu.network.layers.last().unwrap();
    let darknet_core::LayerKind::DeformConv(spec) = &layer.kind else {
        panic!("not a deform layer");
    };
    assert_eq!(spec.dilation, 2, "cfg must carry the dilation");
    let shape = cpu.network.net.input_shape();
    let data: Vec<f32> = (0..shape.element_count())
        .map(|i| ((i * 7 % 17) as f32) * 0.125 - 1.0)
        .collect();
    let input = Tensor::from_data(shape, data).unwrap();
    let want = cpu_layer_output(&cpu, &input, "dilated");
    let got = native_raw(&cpu, &input, "dilated");
    assert_close("dilated", &got, &want);
}
