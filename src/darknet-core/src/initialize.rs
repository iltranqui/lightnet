//! Seeded training initialization; ordinary cfg construction remains zero-filled.
use crate::runtime::{BatchNormParams, ConnectedParams, ConvolutionalParams};
use crate::{LayerParams, Network};

/// Initialize fresh training parameters with seeded Rust RNG values, Darknet's
/// per-layer fan-in/Xavier scales, and unit normalization scales. The ViT arm
/// mirrors its C++ drop-in patch, QKV, projection, FFN, and position initializers.
/// The seed is reproducible in Rust, not bit-identical to C++'s RNG.
/// Call before loading weights and before creating device parameter caches.
pub fn initialize_training_params(network: &Network, params: &mut [LayerParams], seed: u64) {
    let mut rng = Rng(seed);
    for (layer, params) in network.layers.iter().zip(params) {
        match params {
            LayerParams::None => {}
            LayerParams::BatchNorm(p) => bn(p),
            LayerParams::Convolutional(p) => conv(p, &mut rng),
            LayerParams::Connected(p) => connected(p, &mut rng),
            LayerParams::Local(p) => {
                let fan = match &layer.kind {
                    crate::LayerKind::Local(spec) => {
                        layer.input_shape.channels * spec.size * spec.size
                    }
                    _ => unreachable!("local parameters require a local layer"),
                };
                rng.weights(&mut p.weights, fan);
                p.biases.fill(0.0);
            }
            // C++ initialises weighted-shortcut weights to 1.0.
            LayerParams::Shortcut(p) => p.weights.fill(1.0),
            // C++ zero-initialises the scorer and expects
            // `load_laya_scorer_weights` to populate it from the shared blob, so
            // re-randomising it here would be wrong.
            LayerParams::LayaScorer(_) => {}
            // Same for `[laya_act_head]`.
            LayerParams::LayaActHead(_) => {}
            // The encoder is frozen with a unit-gamma default; its weights come
            // from the shared blob, not from random init.
            LayerParams::LayaEncoder(_) => {}
            LayerParams::LayaHead(_) => {}
            LayerParams::Swin(p) => {
                let input = layer.input_shape.channels;
                let output = layer.output_shape.channels;
                let hidden = match &layer.kind {
                    crate::LayerKind::Swin(spec) => output * spec.ffn_ratio,
                    _ => unreachable!("Swin parameters require a Swin layer"),
                };
                let xavier_qkv = (6.0 / (input + 3 * input) as f32).sqrt();
                rng.weights_with_scale(&mut p.qkv_weight, xavier_qkv);
                let xavier_projection = (6.0 / (input + output) as f32).sqrt();
                rng.weights_with_scale(&mut p.wo, xavier_projection);
                rng.weights_with_scale(&mut p.ffn_w1, (6.0 / output as f32).sqrt());
                rng.weights_with_scale(&mut p.ffn_w2, (6.0 / hidden as f32).sqrt());
                if let Some(res_proj) = &mut p.res_proj {
                    rng.weights_with_scale(res_proj, xavier_projection);
                }
                rng.weights_with_scale(&mut p.relative_position_bias, 0.02);
                p.qkv_bias.fill(0.0);
                p.wo_bias.fill(0.0);
                p.ln1_gamma.fill(1.0);
                p.ln1_beta.fill(0.0);
                p.ln2_gamma.fill(1.0);
                p.ln2_beta.fill(0.0);
                p.ffn_b1.fill(0.0);
                p.ffn_b2.fill(0.0);
                p.mhc.copy_from_slice(&crate::swin::SWIN_MHC_DEFAULTS);
            }
            LayerParams::Vit(p) => {
                let patch_fan = layer.input_shape.channels
                    * match &layer.kind {
                        crate::LayerKind::Vit(spec) => spec.patch_size * spec.patch_size,
                        _ => unreachable!("ViT parameters require a ViT layer"),
                    };
                let hidden = match &layer.kind {
                    crate::LayerKind::Vit(spec) => spec.dim,
                    _ => unreachable!("ViT parameters require a ViT layer"),
                };
                let mlp = match &layer.kind {
                    crate::LayerKind::Vit(spec) => spec.mlp_dim,
                    _ => unreachable!("ViT parameters require a ViT layer"),
                };
                let spec = match &layer.kind {
                    crate::LayerKind::Vit(spec) => spec,
                    _ => unreachable!("ViT parameters require a ViT layer"),
                };
                if hidden == layer.input_shape.channels {
                    p.patch_embed.fill(0.0);
                    let same_res_overlap = spec.patch_stride == 1
                        && spec.patch_size % 2 == 1
                        && spec.patch_pad == (spec.patch_size - 1) / 2;
                    if same_res_overlap {
                        let center = (spec.patch_size / 2) * spec.patch_size + spec.patch_size / 2;
                        for c in 0..layer.input_shape.channels {
                            p.patch_embed
                                [c * patch_fan + center * layer.input_shape.channels + c] = 1.0;
                        }
                    } else {
                        let scale = 1.0 / (spec.patch_size * spec.patch_size) as f32;
                        for c in 0..layer.input_shape.channels {
                            for y in 0..spec.patch_size {
                                for x in 0..spec.patch_size {
                                    let index = c * patch_fan
                                        + (y * spec.patch_size + x) * layer.input_shape.channels
                                        + c;
                                    p.patch_embed[index] = scale;
                                }
                            }
                        }
                    }
                } else {
                    rng.weights_with_scale(
                        &mut p.patch_embed,
                        (2.0 / (patch_fan + hidden) as f32).sqrt(),
                    );
                }
                rng.weights_with_scale(&mut p.qkv_weight, (2.0 / (4 * hidden) as f32).sqrt());
                rng.weights_with_scale(&mut p.wo, (1.0 / hidden as f32).sqrt());
                rng.weights(&mut p.ffn_w1, hidden);
                rng.weights(&mut p.ffn_w2, mlp);
                p.patch_bias.fill(0.0);
                p.qkv_bias.fill(0.0);
                p.wo_bias.fill(0.0);
                p.ln1_gamma.fill(1.0);
                p.ln1_beta.fill(0.0);
                p.ln2_gamma.fill(1.0);
                p.ln2_beta.fill(0.0);
                p.ffn_b1.fill(0.0);
                p.ffn_b2.fill(0.0);
                if spec.learned_pos_embed {
                    if spec.pos_init_zero {
                        p.pos_embed.fill(0.0);
                    } else {
                        rng.weights_with_scale(&mut p.pos_embed, 0.02);
                    }
                }
            }
            // Every expert is an ordinary convolution, so they get the ordinary
            // convolution initialisation.  The router's weights use the same
            // uniform rule over its `channels`-wide input; a zero router would
            // give every image the same uniform gates and no routing at all.
            LayerParams::Moe(p) => {
                let fan_in = p.router_weights.len() / p.router_biases.len().max(1);
                rng.weights(&mut p.router_weights, fan_in);
                p.router_biases.fill(0.0);
                for expert in &mut p.experts {
                    conv(expert, &mut rng);
                }
            }
            LayerParams::EmlConvolutional(p) => {
                conv(&mut p.input, &mut rng);
                conv(&mut p.self_params, &mut rng);
            }
            LayerParams::Rnn(p) => {
                for p in [&mut p.input, &mut p.self_params, &mut p.output] {
                    connected(p, &mut rng);
                }
            }
            LayerParams::Gru(p) => {
                for p in [
                    &mut p.input_z,
                    &mut p.state_z,
                    &mut p.input_r,
                    &mut p.state_r,
                    &mut p.input_h,
                    &mut p.state_h,
                ] {
                    connected(p, &mut rng);
                }
            }
            LayerParams::Lstm(p) => {
                for p in [
                    &mut p.wf, &mut p.wi, &mut p.wg, &mut p.wo, &mut p.uf, &mut p.ui, &mut p.ug,
                    &mut p.uo,
                ] {
                    connected(p, &mut rng);
                }
            }
            LayerParams::Crnn(p) => {
                for p in [&mut p.input, &mut p.self_params, &mut p.output] {
                    conv(p, &mut rng);
                }
            }
        }
    }
}

fn bn(p: &mut BatchNormParams) {
    p.biases.fill(0.0);
    p.scales.fill(1.0);
    p.rolling_mean.fill(0.0);
    p.rolling_variance.fill(0.0);
}
fn conv(p: &mut ConvolutionalParams, rng: &mut Rng) {
    let fan = p.weights.len() / p.biases.len().max(1);
    rng.weights(&mut p.weights, fan);
    p.biases.fill(0.0);
    if let Some(p) = &mut p.batch_norm {
        bn(p);
    }
}
fn connected(p: &mut ConnectedParams, rng: &mut Rng) {
    let fan = p.weights.len() / p.biases.len().max(1);
    rng.weights(&mut p.weights, fan);
    p.biases.fill(0.0);
    if let Some(p) = &mut p.batch_norm {
        bn(p);
    }
}
struct Rng(u64);
impl Rng {
    fn weights(&mut self, weights: &mut [f32], fan_in: usize) {
        self.weights_with_scale(weights, (2.0 / fan_in.max(1) as f32).sqrt());
    }

    fn weights_with_scale(&mut self, weights: &mut [f32], scale: f32) {
        for weight in weights {
            // SplitMix64: all seeds, including zero, produce a nonconstant stream.
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let unit = (z >> 40) as f32 / 16777216.0;
            *weight = (2.0 * unit - 1.0) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigFile, ExecutableNetwork};
    #[test]
    fn loading_weights_overrides_initialization_and_preserves_seen() {
        let cfg = ConfigFile::parse_str("init.cfg", "[net]\nwidth=1\nheight=1\nchannels=4\n[connected]\noutput=3\nbatch_normalize=1\nactivation=linear\n").unwrap();
        let network = Network::from_config(cfg).unwrap();
        let mut source = ExecutableNetwork::new(network.clone());
        let mut destination = ExecutableNetwork::new(network.clone());
        if let LayerParams::Connected(p) = &source.params[0] {
            assert!(p.weights.iter().all(|w| *w == 0.0));
        }
        initialize_training_params(&network, &mut source.params, 123);
        initialize_training_params(&network, &mut destination.params, 456);
        assert_ne!(source.params, destination.params);
        let path =
            std::env::temp_dir().join(format!("darknet-init-{}.weights", std::process::id()));
        crate::save_weights(&network, &source.params, &path, 512).unwrap();
        let header = destination.load_weights_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(header.seen, 512);
        assert_eq!(source.params, destination.params);
        if let LayerParams::Connected(p) = &source.params[0] {
            let bound = (2.0f32 / 4.0).sqrt();
            assert!(p.weights.iter().all(|w| w.abs() <= bound));
            assert_eq!(p.batch_norm.as_ref().unwrap().scales, vec![1.0; 3]);
        }
    }

    #[test]
    fn swin_training_initialization_is_seeded_and_uses_audited_parameter_bounds() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "swin-init.cfg",
                "[net]\nwidth=4\nheight=3\nchannels=4\n\
                 [swin]\nfilters=6\nsize=2\nheads=2\nffn_ratio=3\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = crate::ExecutableNetwork::new(network.clone());
        let mut second = crate::ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 29);
        initialize_training_params(&network, &mut second.params, 29);
        assert_eq!(first.params, second.params);
        let LayerParams::Swin(weights) = &first.params[0] else {
            panic!("Swin params")
        };
        let within = |values: &[f32], bound: f32| values.iter().all(|value| value.abs() <= bound);
        assert!(within(&weights.qkv_weight, (6.0f32 / 16.0).sqrt()));
        assert!(within(&weights.wo, (6.0f32 / 10.0).sqrt()));
        assert!(within(&weights.ffn_w1, (6.0f32 / 6.0).sqrt()));
        assert!(within(&weights.ffn_w2, (6.0f32 / 18.0).sqrt()));
        assert!(within(
            weights.res_proj.as_ref().unwrap(),
            (6.0f32 / 10.0).sqrt()
        ));
        assert!(within(&weights.relative_position_bias, 0.02));
        assert!(weights.qkv_bias.iter().all(|value| *value == 0.0));
        assert!(weights.ln1_gamma.iter().all(|value| *value == 1.0));
        assert!(weights.ln2_gamma.iter().all(|value| *value == 1.0));
        assert_eq!(weights.mhc, crate::swin::SWIN_MHC_DEFAULTS);
    }

    #[test]
    fn vit_training_initialization_is_seeded_and_respects_position_mode() {
        let make = |pos_init: &str| {
            let cfg = ConfigFile::parse_str(
                "vit-init.cfg",
                &format!("[net]\nwidth=4\nheight=4\nchannels=1\n[vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=4\npos_init={pos_init}\n"),
            ).unwrap();
            Network::from_config(cfg).unwrap()
        };
        let network = make("random");
        let mut a = ExecutableNetwork::new(network.clone());
        let mut b = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut a.params, 17);
        initialize_training_params(&network, &mut b.params, 17);
        assert_eq!(a.params, b.params);
        let LayerParams::Vit(w) = &a.params[0] else {
            panic!("Vit params")
        };
        assert!(w.patch_embed.iter().any(|x| *x != 0.0));
        assert!(w.pos_embed.iter().any(|x| *x != 0.0));
        assert!(w.pos_embed.iter().all(|x| x.abs() <= 0.02));
        assert_eq!(w.ln1_gamma, vec![1.0; 4]);
        assert_eq!(w.ln2_gamma, vec![1.0; 4]);

        let network = make("zeros");
        let mut zero = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut zero.params, 17);
        let LayerParams::Vit(w) = &zero.params[0] else {
            panic!("Vit params")
        };
        assert_eq!(w.pos_embed, vec![0.0; 16]);

        let cfg = ConfigFile::parse_str(
            "vit-dropin.cfg",
            "[net]\nwidth=4\nheight=4\nchannels=1\n[vit]\ndim=1\nheads=1\npatch_size=2\npatch_stride=2\nmlp_dim=2\n",
        ).unwrap();
        let network = Network::from_config(cfg).unwrap();
        let mut dropin = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut dropin.params, 5);
        let LayerParams::Vit(w) = &dropin.params[0] else {
            panic!("Vit params")
        };
        assert_eq!(w.patch_embed, vec![0.25; 4]);
    }

    #[test]
    fn seeded_initialization_is_nonzero_and_keeps_bn_identity() {
        let cfg = ConfigFile::parse_str("init.cfg", "[net]\nwidth=2\nheight=2\nchannels=4\n[convolutional]\nfilters=8\nsize=1\ngroups=2\nbatch_normalize=1\nactivation=leaky\n").unwrap();
        let network = Network::from_config(cfg).unwrap();
        let mut a = ExecutableNetwork::new(network.clone());
        let mut b = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut a.params, 0);
        initialize_training_params(&network, &mut b.params, 0);
        assert_eq!(a.params, b.params);
        if let LayerParams::Convolutional(p) = &a.params[0] {
            assert!(p.weights.iter().all(|w| w.is_finite() && w.abs() <= 1.0));
            assert!(p.weights.iter().any(|w| *w != 0.0));
            assert!(p.weights.windows(2).any(|w| w[0] != w[1]));
            assert_eq!(p.batch_norm.as_ref().unwrap().scales, vec![1.0; 8]);
        } else {
            panic!("expected convolution");
        }
        initialize_training_params(&network, &mut b.params, 1);
        assert_ne!(a.params, b.params);
    }
}
