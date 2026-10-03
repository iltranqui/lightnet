//! Seeded training initialization; ordinary cfg construction remains zero-filled.
use crate::runtime::{BatchNormParams, ConnectedParams, ConvolutionalParams};
use crate::{LayerParams, Network};

/// Initialize fresh training parameters using Darknet's uniform
/// `[-sqrt(2/fan_in), sqrt(2/fan_in)]` weights and unit BN scales.
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
        let scale = (2.0 / fan_in.max(1) as f32).sqrt();
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
        let path = std::env::temp_dir().join(format!("darknet-init-{}.weights", std::process::id()));
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
