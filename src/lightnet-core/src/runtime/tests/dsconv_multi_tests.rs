use super::*;

fn fixture(templates: usize, drop: f32) -> (ExecutableNetwork, Tensor) {
    let cfg = format!("[net]\nwidth=4\nheight=4\nchannels=1\nbatch=1\nadam=1\n[dsconv]\nfilters=1\nsize=3\nmorph=0\nactivation=linear\ndsc_templates={templates}\ndsc_drop_prob={drop}\n");
    let network = Network::from_config(ConfigFile::parse_str("dsconv-multi-test.cfg", &cfg).unwrap()).unwrap();
    let mut net = ExecutableNetwork::new(network);
    let LayerParams::DsConv(p) = &mut net.params[0] else { panic!() };
    p.collapse.weights.copy_from_slice(&[0.31, -0.22, 0.47]);
    p.collapse.biases.fill(0.17);
    p.offset_biases.copy_from_slice(&[0.21, 0.04, -0.18, -0.16, 0.03, 0.19]);
    for (i, v) in p.offset_weights.iter_mut().enumerate() { *v = 0.0001 * ((i % 7) as f32 - 3.0); }
    for (j, block) in p.extra_offset_weights.iter_mut().enumerate() {
        for (i, v) in block.iter_mut().enumerate() { *v = 0.0001 * ((i % 7) as f32 - 3.0) + 0.0002 * (j + 1) as f32; }
    }
    let input = Tensor::from_data(TensorShape::new(4, 4, 1, 1), (0..16).map(|i| 0.2 + (i * i + 3 * i) as f32 * 0.017).collect()).unwrap();
    (net, input)
}

fn mask_and_oracle(net: &ExecutableNetwork, input: &Tensor, f: &ForwardResult) -> Vec<bool> {
    let LayerKind::DsConv(spec) = &net.network.layers[0].kind else { panic!() };
    let LayerParams::DsConv(p) = &net.params[0] else { panic!() };
    let DsConvRuntimeMath::Multi(cache) = &f.dsconv_caches[0].as_ref().unwrap().math else { panic!() };
    let count = 2 * spec.templates;
    let shape = f.layer_outputs[0].shape;
    for bits in 1..(1usize << count) {
        let mask: Vec<bool> = (0..count).map(|i| bits & (1 << i) != 0).collect();
        let (_, candidate) = crate::dsconv::dsconv_multi_view_forward(input, shape, spec, &p.collapse.weights, &p.collapse.biases, &p.offset_weights, &p.offset_biases, &p.extra_offset_weights, &mask).unwrap();
        if &candidate == cache { return mask; }
    }
    panic!("forward cache does not match any view mask")
}

fn explicit_sum(net: &ExecutableNetwork, input: &Tensor, mask: &[bool]) -> Vec<f32> {
    let LayerKind::DsConv(spec) = &net.network.layers[0].kind else { panic!() };
    let LayerParams::DsConv(p) = &net.params[0] else { panic!() };
    let mut single = spec.clone();
    single.templates = 1;
    let shape = net.network.layers[0].output_shape;
    let mut expected = vec![p.collapse.biases[0]; shape.element_count() * shape.batch];
    let scale = 1.0 / mask.iter().filter(|&&a| a).count() as f32;
    for (slot, &active) in mask.iter().enumerate() {
        if !active { continue; }
        single.morph = if slot % 2 == 0 { spec.morph } else { 1 - spec.morph };
        let block = if slot < 2 { &p.offset_weights } else { &p.extra_offset_weights[slot - 2] };
        let (view, _) = crate::dsconv::dsconv_forward(input, shape, &single, &p.collapse.weights, &[0.0], block, &p.offset_biases).unwrap();
        for (a, b) in expected.iter_mut().zip(view.data) { *a += scale * b; }
    }
    expected
}

#[test]
fn dsconv_multi_template_forward_matches_single_view_sum_and_inference_uses_all_views() {
    for templates in [2, 3] {
        let (net, input) = fixture(templates, 0.65);
        for training in [false, true] {
            let f = if training { net.forward_training_with_rng_seed(&input, 987).unwrap() } else { net.forward(&input).unwrap() };
            let mask = mask_and_oracle(&net, &input, &f);
            if !training { assert!(mask.iter().all(|&v| v)); }
            let expected = explicit_sum(&net, &input, &mask);
            for (a, b) in f.layer_outputs[0].data.iter().zip(expected) { assert!((a - b).abs() < 1e-6, "{a} != {b}"); }
        }
    }
}

#[test]
fn dsconv_multi_template_mask_reuse_and_seeded_fallback() {
    let (net, input) = fixture(3, 0.6);
    let seed = (1..1000).find(|&s| { let f = net.forward_training_with_rng_seed(&input, s).unwrap(); let m = mask_and_oracle(&net, &input, &f); m[2..].iter().any(|&v| !v) && m[2..].iter().any(|&v| v) }).unwrap();
    let f = net.forward_training_with_rng_seed(&input, seed).unwrap();
    let again = net.forward_training_with_rng_seed(&input, seed).unwrap();
    assert_eq!(f.dsconv_caches[0], again.dsconv_caches[0]);
    let mask = mask_and_oracle(&net, &input, &f);
    let upstream = Tensor::filled(f.layer_outputs[0].shape, 1.0);
    let g = net.backward(&input, &f, &upstream).unwrap();
    let _other = net.forward_training_with_rng_seed(&input, seed + 1).unwrap();
    assert_eq!(g.param_gradients, net.backward(&input, &f, &upstream).unwrap().param_gradients);
    let LayerParams::DsConv(g) = &g.param_gradients[0] else { panic!() };
    for (i, block) in g.extra_offset_weights.iter().enumerate() {
        if !mask[i + 2] { assert!(block.iter().all(|&v| v == 0.0)); }
        else { assert!(block.iter().any(|&v| v != 0.0)); }
    }
    let (net, input) = fixture(3, 0.999999);
    let mut rng_state = 123;
    let draw_seed = advance_rng(&mut rng_state);
    assert!((0..6).all(|i| dropout_unit(draw_seed, i) < 0.999999));
    let f = net.forward_training_with_rng_seed(&input, 123).unwrap();
    assert_eq!(mask_and_oracle(&net, &input, &f), vec![true, false, false, false, false, false]);
}

fn step(net: &mut ExecutableNetwork, input: &Tensor, adam: bool) {
    let f = net.forward_training(input).unwrap();
    let seed = Tensor::filled(f.layer_outputs[0].shape, 1.0);
    net.backward_accumulate(input, &f, &seed).unwrap();
    if adam { net.update_weights_adam(0.01, 1).unwrap(); } else { net.update_weights(0.1, 0.0, 0.0, 1); }
}

#[test]
fn dsconv_multi_template_end_to_end_sgd_and_adam_update_every_extra_array() {
    for adam in [false, true] {
        let (mut net, input) = fixture(2, 0.0);
        let LayerParams::DsConv(before) = &net.params[0] else { panic!() };
        let before = before.extra_offset_weights.clone();
        step(&mut net, &input, adam);
        let LayerParams::DsConv(after) = &net.params[0] else { panic!() };
        for (a, b) in before.iter().zip(&after.extra_offset_weights) { assert_ne!(a, b); }
    }
}

#[test]
fn dsconv_multi_template_adam_sidecar_resume_matches_uninterrupted_update() {
    let (mut net, input) = fixture(3, 0.0);
    step(&mut net, &input, true);
    let path = std::env::temp_dir().join(format!("dsconv-multi-{}.dnopt", std::process::id()));
    net.save_optimizer_state(&path).unwrap();
    let (mut resumed, _) = fixture(3, 0.0);
    resumed.params = net.params.clone();
    resumed.load_optimizer_state(&path).unwrap();
    assert_eq!(net.host_adam_state, resumed.host_adam_state);
    step(&mut net, &input, true);
    step(&mut resumed, &input, true);
    assert_eq!(net.params, resumed.params);
    assert_eq!(net.host_adam_state, resumed.host_adam_state);
    std::fs::remove_file(path).unwrap();
}

fn parameter(p: &mut DsConvParams, group: usize) -> &mut Vec<f32> {
    match group { 0 => &mut p.collapse.weights, 1 => &mut p.collapse.biases, 2 => &mut p.offset_biases, 3 => &mut p.offset_weights, _ => &mut p.extra_offset_weights[group - 4] }
}

fn objective(net: &ExecutableNetwork, input: &Tensor, seed: u64, upstream: &Tensor) -> f32 {
    net.forward_training_with_rng_seed(input, seed).unwrap().layer_outputs[0].data.iter().zip(&upstream.data).map(|(a,b)| a*b).sum()
}

#[test]
fn dsconv_multi_template_runtime_finite_differences_all_parameters_and_input() {
    for drop in [0.0, 0.45] {
        let (mut net, mut input) = fixture(3, drop);
        let rng = if drop == 0.0 { 57 } else { (1..1000).find(|&s| { let f = net.forward_training_with_rng_seed(&input, s).unwrap(); let m = mask_and_oracle(&net, &input, &f); m.iter().any(|&v| v) && m[2..].iter().any(|&v| v) && m[2..].iter().any(|&v| !v) }).unwrap() };
        let f = net.forward_training_with_rng_seed(&input, rng).unwrap();
        let mask = mask_and_oracle(&net, &input, &f);
        let upstream = Tensor::from_data(f.layer_outputs[0].shape, (0..16).map(|i| 0.13 + i as f32 * 0.023).collect()).unwrap();
        let backward = net.backward(&input, &f, &upstream).unwrap();
        let LayerParams::DsConv(mut grads) = backward.param_gradients[0].clone() else { panic!() };
        let epsilon = 0.002;
        for i in 0..input.data.len() {
            let original = input.data[i];
            input.data[i] = original + epsilon; let plus = objective(&net, &input, rng, &upstream);
            input.data[i] = original - epsilon; let minus = objective(&net, &input, rng, &upstream);
            input.data[i] = original;
            let numeric = (plus - minus) / (2.0 * epsilon);
            let actual = backward.input_gradient.data[i];
            assert!((actual - numeric).abs() < 0.003 + 0.02 * actual.abs(), "input {i}: {actual} != {numeric}; mask {mask:?}");
        }
        for group in 0..8 {
            let n = parameter(&mut grads, group).len();
            for i in 0..n {
                let LayerParams::DsConv(p) = &mut net.params[0] else { panic!() };
                let original = parameter(p, group)[i];
                parameter(p, group)[i] = original + epsilon;
                let plus = objective(&net, &input, rng, &upstream);
                let LayerParams::DsConv(p) = &mut net.params[0] else { panic!() };
                parameter(p, group)[i] = original - epsilon;
                let minus = objective(&net, &input, rng, &upstream);
                let LayerParams::DsConv(p) = &mut net.params[0] else { panic!() };
                parameter(p, group)[i] = original;
                let actual = parameter(&mut grads, group)[i];
                let numeric = (plus - minus) / (2.0 * epsilon);
                assert!((actual - numeric).abs() < 0.003 + 0.02 * actual.abs(), "group {group} index {i}: {actual} != {numeric}; mask {mask:?}");
            }
        }
    }
}

#[test]
fn dsconv_multi_template_axis_layout_pins_view_order_and_nchw() {
    use crate::axis_fixture::{axis_tensor, probe_shape};
    let shape = probe_shape();
    let cfg = "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[dsconv]\nfilters=3\nsize=3\nmorph=1\nactivation=linear\ndsc_templates=3\n";
    let network = Network::from_config(ConfigFile::parse_str("dsconv-axis.cfg", cfg).unwrap()).unwrap();
    let mut net = ExecutableNetwork::new(network);
    let LayerParams::DsConv(p) = &mut net.params[0] else { panic!() };
    for (i, v) in p.collapse.weights.iter_mut().enumerate() { *v = (i + 1) as f32 * 0.03; }
    p.collapse.biases.copy_from_slice(&[1.0, 2.0, 3.0]);
    for (i, v) in p.offset_weights.iter_mut().enumerate() { *v = ((i % 7) as f32 - 3.0) * 0.0001; }
    for (slot, block) in p.extra_offset_weights.iter_mut().enumerate() {
        for (i, v) in block.iter_mut().enumerate() { *v = ((i % 9) as f32 - 4.0) * 0.0001 * (slot + 1) as f32; }
    }
    p.offset_biases.copy_from_slice(&[0.2, 0.0, -0.2, -0.2, 0.0, 0.2]);
    let input = Tensor::from_data(shape, axis_tensor(shape)).unwrap();
    let f = net.forward(&input).unwrap();
    let LayerKind::DsConv(spec) = &net.network.layers[0].kind else { panic!() };
    let LayerParams::DsConv(p) = &net.params[0] else { panic!() };
    let out_shape = net.network.layers[0].output_shape;
    let mut expected = vec![0.0; f.layer_outputs[0].data.len()];
    let mut single = spec.clone(); single.templates = 1;
    for slot in 0..6 {
        single.morph = if slot % 2 == 0 { 1 } else { 0 };
        let block = if slot < 2 { &p.offset_weights } else { &p.extra_offset_weights[slot - 2] };
        let (view, _) = crate::dsconv::dsconv_forward(&input, out_shape, &single, &p.collapse.weights, &p.collapse.biases, block, &p.offset_biases).unwrap();
        for (dst, val) in expected.iter_mut().zip(view.data) { *dst += val / 6.0; }
    }
    for (index, (actual, expected)) in f.layer_outputs[0].data.iter().zip(expected).enumerate() {
        assert!((actual - expected).abs() < 0.002, "NCHW index {index}: {actual} != {expected}");
    }
}
