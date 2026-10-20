//! Native LayaScorer f32 runtime composition against the Rust CPU reference.
use darknet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor,
};
use darknet_cuda_oxide::OxideBackend;

fn make_network() -> ExecutableNetwork {
    let hidden = 4;
    let options = 3;
    let cfg = ConfigFile::parse_str(
        "native-laya-scorer.cfg",
        &format!(
            "[net]\nwidth={options}\nheight=1\nchannels={hidden}\nbatch=2\n[laya_scorer]\nhidden={hidden}\nmax_options={options}\nln_eps=0.00002\n"
        ),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::LayaScorer(weights) = &mut cpu.params[0] else {
        panic!("expected LayaScorer weights")
    };
    let mut state = 0xA076_1D64_78BD_642Fu64;
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
    *weights = darknet_core::laya_scorer::LayaScorerWeights {
        ln_w: fill(hidden),
        ln_b: fill(hidden),
        fc1_w: fill(hidden * hidden),
        fc1_b: fill(hidden),
        fc2_w: fill(hidden),
        fc2_b: 0.137,
    };
    cpu
}

fn check_full_and_compact() {
    let cpu = make_network();
    let shape = cpu.network.net.input_shape();
    let data = (0..shape.element_count() * shape.batch)
        .map(|i| ((i as f32) * 0.31).sin() * 0.9)
        .collect();
    let input = Tensor::from_data(shape, data).unwrap();
    let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    for (label, actual) in [
        (
            "full",
            gpu.forward(&input).unwrap().final_output().unwrap().clone(),
        ),
        (
            "compact",
            gpu.forward_compact(&input).unwrap().final_output.unwrap(),
        ),
    ] {
        assert_eq!(actual.shape, expected.shape, "{label}");
        let scale = expected
            .data
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-3);
        let max_diff = actual
            .data
            .iter()
            .zip(&expected.data)
            .fold(0.0f32, |m, (got, want)| m.max((got - want).abs()));
        assert!(
            max_diff <= 2e-4 * scale,
            "{label}: max diff {max_diff}, scale {scale}"
        );
    }
}

#[test]
fn laya_scorer_native_full_and_compact_match_cpu() {
    check_full_and_compact();
}

#[test]
fn laya_scorer_native_rejects_malformed_weights() {
    let cpu = make_network();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let LayerParams::LayaScorer(weights) = &mut gpu.params[0] else {
        panic!("expected LayaScorer weights")
    };
    weights.fc1_b.pop();
    let input = Tensor::zeros(gpu.network.net.input_shape());
    for error in [
        gpu.forward(&input).unwrap_err(),
        gpu.forward_compact(&input).unwrap_err(),
    ] {
        assert!(error.to_string().contains("weight fc1_b"), "{error}");
    }
}
