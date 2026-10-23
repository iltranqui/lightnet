//! Native LayaHead f32 runtime composition against the Rust CPU reference.
use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor,
};
use lightnet_cuda_oxide::OxideBackend;

fn make_network(qtype: i32) -> ExecutableNetwork {
    let hidden = 4;
    let channels = hidden + if qtype == -1 { 2 } else { 1 };
    let cfg = ConfigFile::parse_str(
        "native-laya-head.cfg",
        &format!(
            "[net]\nwidth=3\nheight=1\nchannels={channels}\nbatch=2\n[laya_head]\nhidden=4\nheads=2\nlayers=2\nffn_dim=5\nqtype={qtype}\n"
        ),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::LayaHead(weights) = &mut cpu.params[0] else {
        panic!("expected LayaHead weights")
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
    *weights = lightnet_core::laya_head::LayaHeadWeights {
        norm1_w: fill(2 * hidden),
        norm1_b: fill(2 * hidden),
        in_proj_w: fill(2 * 3 * hidden * hidden),
        in_proj_b: fill(2 * 3 * hidden),
        out_proj_w: fill(2 * hidden * hidden),
        out_proj_b: fill(2 * hidden),
        norm2_w: fill(2 * hidden),
        norm2_b: fill(2 * hidden),
        linear1_w: fill(2 * 5 * hidden),
        linear1_b: fill(2 * 5),
        linear2_w: fill(2 * hidden * 5),
        linear2_b: fill(2 * hidden),
        type_emb: fill(3 * hidden),
    };
    cpu
}

fn check(qtype: i32) {
    let cpu = make_network(qtype);
    let mut input_data = vec![0.0f32; cpu.network.net.input_shape().element_count() * 2];
    let shape = cpu.network.net.input_shape();
    for (i, value) in input_data.iter_mut().enumerate() {
        *value = ((i as f32) * 0.23).sin() * 0.7;
    }
    // The mask is channel-major at c=hidden; include a mixed row and an empty
    // key set to test the C++/Rust all-masked attention behavior.
    for t in 0..shape.width {
        input_data[(shape.channels - if qtype == -1 { 2 } else { 1 }) * shape.width + t] =
            if t == 1 { 0.0 } else { 1.0 };
        input_data
            [(shape.channels - if qtype == -1 { 2 } else { 1 }) * shape.width + shape.width + t] =
            0.0;
    }
    if qtype == -1 {
        let qtype_channel = shape.channels - 1;
        input_data[qtype_channel * shape.width] = 2.0;
        input_data[(shape.channels + qtype_channel) * shape.width] = 1.0;
    }
    let input = Tensor::from_data(shape, input_data).unwrap();
    let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    for (label, result) in [
        (
            "full",
            gpu.forward(&input).unwrap().final_output().unwrap().clone(),
        ),
        (
            "compact",
            gpu.forward_compact(&input).unwrap().final_output.unwrap(),
        ),
    ] {
        assert_eq!(result.shape, expected.shape, "qtype={qtype} {label}");
        let scale = expected
            .data
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-3);
        let max_diff = result
            .data
            .iter()
            .zip(&expected.data)
            .fold(0.0f32, |m, (got, want)| m.max((got - want).abs()));
        assert!(
            max_diff <= 3e-4 * scale,
            "qtype={qtype} {label}: max diff {max_diff} > 3e-4 * {scale}"
        );
    }
}

#[test]
fn laya_head_native_full_and_compact_match_cpu_for_fixed_qtype() {
    check(1);
}

#[test]
fn laya_head_native_full_and_compact_match_cpu_for_input_qtype() {
    check(-1);
}

#[test]
fn laya_head_native_rejects_malformed_weight_extents() {
    let cpu = make_network(0);
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let LayerParams::LayaHead(weights) = &mut gpu.params[0] else {
        panic!("expected LayaHead weights")
    };
    weights.type_emb.pop();
    let input = Tensor::zeros(gpu.network.net.input_shape());
    for error in [
        gpu.forward(&input).unwrap_err(),
        gpu.forward_compact(&input).unwrap_err(),
    ] {
        assert!(
            error.to_string().contains("weight tensors do not match"),
            "{error}"
        );
    }
}
