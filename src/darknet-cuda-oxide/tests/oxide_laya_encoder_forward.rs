//! Native LayaEncoder f32 runtime inference against the Rust CPU reference.
use darknet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor,
};
use darknet_cuda_oxide::OxideBackend;

fn make_network() -> ExecutableNetwork {
    let hidden = 4;
    let heads = 2;
    let layers = 3;
    let intermediate = 5;
    let vocab = 7;
    let cfg = ConfigFile::parse_str(
        "native-laya-encoder.cfg",
        &format!(
            "[net]\nwidth=4\nheight=1\nchannels=2\nbatch=2\n[laya_encoder]\nhidden={hidden}\nheads={heads}\nlayers={layers}\nintermediate={intermediate}\nvocab={vocab}\nlocal_window=2\nrope_theta_full=1234\nrope_theta_local=2468\n"
        ),
    )
    .unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::LayaEncoder(weights) = &mut cpu.params[0] else {
        panic!("expected LayaEncoder weights")
    };
    let mut state = 0xE703_7ED1_A0B4_28DBu64;
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
    *weights = darknet_core::laya_encoder::LayaEncoderWeights {
        tok_embed: fill(vocab * hidden),
        embed_norm: fill(hidden),
        attn_norm: fill((layers - 1) * hidden),
        wqkv: fill(layers * 3 * hidden * hidden),
        wo: fill(layers * hidden * hidden),
        mlp_norm: fill(layers * hidden),
        wi: fill(layers * 2 * intermediate * hidden),
        wo_mlp: fill(layers * hidden * intermediate),
        final_norm: fill(hidden),
    };
    cpu
}

fn input() -> Tensor {
    // Channel-major [B,2,1,T]: batch 0 has padding and a local-window edge;
    // batch 1 is fully masked. IDs outside vocabulary exercise clamp behavior.
    Tensor::from_data(
        darknet_core::TensorShape::new(4, 1, 2, 2),
        vec![
            0.0, 2.0, 6.0, 3.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 8.0, -2.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap()
}

#[test]
fn laya_encoder_native_full_and_compact_match_cpu() {
    let cpu = make_network();
    let input = input();
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
            max_diff <= 4e-4 * scale,
            "{label}: max diff {max_diff}, scale {scale}"
        );
    }
}

#[test]
fn laya_encoder_native_rejects_malformed_embedding_extent() {
    let cpu = make_network();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let LayerParams::LayaEncoder(weights) = &mut gpu.params[0] else {
        panic!("expected LayaEncoder weights")
    };
    weights.tok_embed.pop();
    let input = input();
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
