//! `GpuExecutableNetwork::forward_compact_f16` (f16 storage inference, real
//! Rust GPU kernels) against `forward_compact` (f32) on the same weights.
//!
//! Tolerance model: weights, activations and every layer output are rounded to
//! binary16 (u = 2^-11 per rounding, accumulation / bias / activation in f32),
//! so the f16 forward differs from the f32 one by a few u per layer, growing
//! with depth (random weights, no training, so there is no error contraction).
//! The check is scale-normalised: max |f16 - f32| <= tol * max |f32| per
//! output tensor, with tol = 4e-3 for the tiny nets (<= 6 layers) and 5e-3 for
//! the 25 - 100 layer real cfgs (measured values are printed with
//! `--nocapture`: worst 1.3e-3 relative; the tolerances leave >= 3.8x headroom).

use darknet_core::{ConfigFile, Error, GpuExecutableNetwork, GpuForwardSummary, LayerParams, Network, Tensor};
use darknet_cuda_oxide::OxideBackend;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

fn network(cfg: &str) -> Network {
    Network::from_config(ConfigFile::parse_str("inline.cfg", cfg).expect("cfg")).expect("network")
}

/// Deterministic He-style weights (`gain * sqrt(2 / fan_in)`), small biases,
/// plausible BN statistics.
fn fill(net: &Network, params: &mut [LayerParams], seed: u64, gain: f32) {
    let mut rng = Rng(seed);
    for (layer, p) in net.layers.iter().zip(params.iter_mut()) {
        match p {
            LayerParams::Convolutional(conv) => {
                let fan_in = (conv.weights.len() / conv.biases.len().max(1)).max(1);
                let a = gain * (6.0 / fan_in as f32).sqrt();
                for w in &mut conv.weights {
                    *w = rng.range(-a, a);
                }
                for b in &mut conv.biases {
                    *b = rng.range(-0.1, 0.1);
                }
                if let Some(bn) = &mut conv.batch_norm {
                    for v in &mut bn.scales {
                        *v = rng.range(0.8, 1.2);
                    }
                    for v in &mut bn.biases {
                        *v = rng.range(-0.1, 0.1);
                    }
                    for v in &mut bn.rolling_mean {
                        *v = rng.range(-0.2, 0.2);
                    }
                    for v in &mut bn.rolling_variance {
                        *v = rng.range(0.5, 1.5);
                    }
                }
            }
            LayerParams::BatchNorm(bn) => {
                for v in &mut bn.scales {
                    *v = rng.range(0.5, 1.5);
                }
                for v in &mut bn.biases {
                    *v = rng.range(-0.5, 0.5);
                }
                for v in &mut bn.rolling_mean {
                    *v = rng.range(-0.3, 0.3);
                }
                for v in &mut bn.rolling_variance {
                    *v = rng.range(0.5, 2.0);
                }
            }
            _ => {}
        }
        let _ = layer;
    }
}

fn input_for(net: &Network, seed: u64) -> Tensor {
    let shape = net.net.input_shape();
    let mut rng = Rng(seed);
    Tensor::from_data(shape, (0..shape.batch * shape.element_count()).map(|_| rng.next()).collect()).expect("input")
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn compare(what: &str, f32s: &GpuForwardSummary, f16s: &GpuForwardSummary, tol: f32) -> f32 {
    let mut pairs: Vec<(String, &Tensor, &Tensor)> = Vec::new();
    for (a, b) in f32s.yolo_heads.iter().zip(&f16s.yolo_heads) {
        assert_eq!(a.layer_index, b.layer_index);
        pairs.push((format!("yolo head #{}", a.layer_index), &a.output, &b.output));
    }
    assert_eq!(f32s.yolo_heads.len(), f16s.yolo_heads.len(), "{what}: yolo head count");
    pairs.push(("final".into(), f32s.final_output.as_ref().unwrap(), f16s.final_output.as_ref().unwrap()));
    let mut worst = 0.0f32;
    for (name, a, b) in pairs {
        assert_eq!(a.shape, b.shape, "{what} {name}: shape");
        assert_eq!(a.data.len(), b.data.len());
        let scale = max_abs(&a.data);
        assert!(scale > 1e-3, "{what} {name}: degenerate reference (max {scale})");
        let diff = a.data.iter().zip(&b.data).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(b.data.iter().all(|v| v.is_finite()), "{what} {name}: non-finite f16 output");
        println!("{what} {name}: max|d| = {diff:.3e}, max|ref| = {scale:.3e}, rel = {:.3e} (tol {tol:.0e})", diff / scale);
        assert!(diff <= tol * scale, "{what} {name}: max diff {diff} > {tol} * {scale}");
        worst = worst.max(diff / scale);
    }
    worst
}

fn run_pair(what: &str, net: Network, seed: u64, gain: f32, tol: f32) -> f32 {
    let mut exec = GpuExecutableNetwork::new(net, OxideBackend::new().expect("oxide backend"));
    let net = exec.network.clone();
    fill(&net, &mut exec.params, seed, gain);
    let input = input_for(&net, seed ^ 0x55);
    let want = exec.forward_compact(&input).expect("f32 forward");
    let got = exec.forward_compact_f16(&input).expect("f16 forward");
    let worst = compare(what, &want, &got, tol);
    // Second call reuses the cached f16 weights: bit-identical to the first.
    let again = exec.forward_compact_f16(&input).expect("f16 forward 2");
    assert_eq!(got.final_output.unwrap().data, again.final_output.unwrap().data, "{what}: cached-weights rerun differs");
    worst
}

#[test]
fn maxpool_depth_f16_inference_matches_f32() {
    let cfg = "[net]\nwidth=4\nheight=3\nchannels=5\nbatch=2\n\
        [maxpool]\nmaxpool_depth=1\nout_channels=2\n";
    run_pair(
        "maxpool_depth f16",
        network(cfg),
        31,
        1.0,
        4e-3,
    );
}

#[test]
fn channel_shuffle_f16_inference_matches_f32() {
    let cfg = "[net]\nwidth=3\nheight=2\nchannels=6\nbatch=2\n\
        [channel_shuffle]\ngroups=2\n";
    run_pair("channel_shuffle f16", network(cfg), 37, 1.0, 2e-3);
}

#[test]
fn reorg_f16_forward_and_reverse_non_square_inference_match_f32() {
    let cfg = "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n\
        [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
        [reorg]\nstride=2\n\
        [reorg]\nstride=2\nreverse=1\n";
    run_pair("reorg f16 forward+reverse", network(cfg), 43, 0.8, 4e-3);
}

#[test]
fn sam_f16_inference_matches_f32() {
    let cfg = "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n\
        [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
        [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
        [sam]\nfrom=0\nactivation=leaky\n";
    run_pair("sam f16", network(cfg), 41, 0.75, 4e-3);
}

#[test]
fn tiny_net_with_shortcut_maxpool_route_upsample_matches_f32() {
    let cfg = "[net]\nwidth=16\nheight=16\nchannels=3\nbatch=2\n\
        [convolutional]\nbatch_normalize=1\nfilters=8\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
        [convolutional]\nbatch_normalize=1\nfilters=8\nsize=3\nstride=1\npad=1\nactivation=mish\n\
        [shortcut]\nfrom=-2\nactivation=leaky\n\
        [maxpool]\nsize=2\nstride=2\n\
        [convolutional]\nfilters=8\nsize=1\nstride=1\npad=1\nactivation=relu6\n\
        [shortcut]\nfrom=-2\nactivation=linear\n\
        [route]\nlayers=-1,-3\n\
        [convolutional]\nbatch_normalize=1\nfilters=6\nsize=3\nstride=1\npad=1\nactivation=swish\n\
        [upsample]\nstride=2\n\
        [route]\nlayers=-1,2\n\
        [convolutional]\nfilters=5\nsize=1\nstride=1\npad=1\nactivation=linear\n";
    run_pair("tiny shortcut/route/upsample", network(cfg), 3, 1.0, 4e-3);
}

#[test]
fn tiny_net_with_avgpool_local_avgpool_batchnorm_reverse_upsample_matches_f32() {
    let cfg = "[net]\nwidth=16\nheight=16\nchannels=3\nbatch=2\n\
        [convolutional]\nfilters=8\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
        [batchnorm]\n\
        [avgpool]\nsize=2\nstride=2\n";
    // standalone BN layer + global avgpool (resnet tail)
    let net = network(cfg);
    assert!(net.layers.iter().any(|l| matches!(l.kind, darknet_core::LayerKind::AvgPool)));
    run_pair("batchnorm+avgpool", net, 5, 1.0, 4e-3);

    let cfg = "[net]\nwidth=16\nheight=16\nchannels=3\nbatch=2\n\
        [convolutional]\nfilters=6\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
        [local_avgpool]\nsize=3\nstride=2\npadding=2\n\
        [convolutional]\nfilters=6\nsize=3\nstride=1\npad=1\nactivation=logistic\n\
        [upsample]\nstride=2\nreverse=1\n";
    let net = network(cfg);
    assert!(net.layers.iter().any(|l| matches!(l.kind, darknet_core::LayerKind::LocalAvgPool(_))));
    run_pair("local_avgpool+reverse upsample", net, 6, 1.0, 4e-3);
}

#[test]
fn unsupported_layers_still_error_cleanly() {
    // connected layer
    let net = network(
        "[net]\nwidth=8\nheight=8\nchannels=2\nbatch=1\n\
         [convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [connected]\noutput=5\nactivation=linear\n",
    );
    let exec = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().unwrap());
    let input = input_for(&net, 1);
    assert!(matches!(exec.forward_compact_f16(&input), Err(Error::UnsupportedLayer { .. })));
}

#[test]
fn channel_normalization_f16_inference_is_explicitly_rejected() {
    let net = network(
        r#"
            [net]
            width=2
            height=1
            channels=2
            batch=1

            [convolutional]
            filters=3
            size=1
            activation=normalize_channels
        "#,
    );
    let exec = GpuExecutableNetwork::new(net.clone(), OxideBackend::new().unwrap());
    let error = exec
        .forward_compact_f16(&input_for(&net, 23))
        .expect_err("f16 channel normalization is not implemented");
    assert!(matches!(
        error,
        Error::UnsupportedLayer { layer_index: 0, name } if name.contains("NORM_CHAN")
    ));
}

#[test]
fn tiny_net_with_mismatched_shortcut_sources_matches_f32() {
    // source with more channels (cropped) and with fewer (zero padded)
    let cfg = "[net]\nwidth=8\nheight=8\nchannels=2\nbatch=2\n\
         [convolutional]\nfilters=6\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [convolutional]\nfilters=3\nsize=3\nstride=1\npad=1\nactivation=leaky\n\
         [shortcut]\nfrom=-2\nactivation=linear\n\
         [convolutional]\nfilters=8\nsize=1\nstride=1\npad=1\nactivation=leaky\n\
         [shortcut]\nfrom=-2\nactivation=leaky\n";
    run_pair("mismatched shortcut", network(cfg), 8, 1.0, 4e-3);
}

/// Real cfg from `cfg/` with a smaller input and optional truncation, so the
/// layer mix (not the size) is what is exercised.
fn real_cfg(file: &str, size: usize, cut_at: Option<&str>) -> Network {
    let path = format!("{}/../../cfg/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = String::new();
    for line in text.lines() {
        let t = line.trim();
        if cut_at.is_some_and(|c| t == c) {
            out.push_str(&format!("{t}\n"));
            break;
        }
        let l = if t.starts_with("width=") {
            format!("width={size}")
        } else if t.starts_with("height=") {
            format!("height={size}")
        } else if t.starts_with("batch=") {
            "batch=2".to_owned()
        } else if t.starts_with("subdivisions=") {
            "subdivisions=1".to_owned()
        } else {
            line.to_owned()
        };
        out.push_str(&l);
        out.push('\n');
    }
    network(&out)
}

#[test]
fn real_cfgs_f16_matches_f32() {
    // yolov3-tiny: maxpool (incl. size 2 stride 1), route, upsample, 2 yolo heads.
    run_pair("yolov3-tiny", real_cfg("yolov3-tiny.cfg", 128, None), 11, 1.0, 5e-3);
    // yolov4-tiny: grouped route (group_id), mish/leaky.
    run_pair("yolov4-tiny", real_cfg("yolov4-tiny.cfg", 128, None), 12, 1.0, 5e-3);
    // yolov3: 23 shortcuts, upsample, 3 heads, 75 convolutions.
    run_pair("yolov3", real_cfg("yolov3.cfg", 128, None), 13, 0.5, 5e-3);
    // resnet101 up to and including the global avgpool: 33 shortcuts, 4 of them
    // with a resized (zero-padded) source, BN everywhere.
    run_pair("resnet101 (to avgpool)", real_cfg("resnet101.cfg", 64, Some("[avgpool]")), 14, 0.6, 5e-3);
}
