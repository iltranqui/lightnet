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
            LayerParams::RecursiveBlock(p) => {
                let crate::LayerKind::RecursiveBlock(spec) = &layer.kind else {
                    unreachable!("recursive-block parameters require a recursive-block layer");
                };
                let mut body_net = Network {
                    source: network.source.clone(),
                    net: network.net.clone(),
                    layers: spec.body.clone(),
                };
                body_net.net.width = layer.input_shape.width;
                body_net.net.height = layer.input_shape.height;
                body_net.net.channels = layer.input_shape.channels;
                body_net.net.batch = layer.input_shape.batch;
                let body_seed = u64::from(rng.next_unit().to_bits());
                initialize_training_params(&body_net, &mut p.body, body_seed);
                p.controller_weights.fill(0.0);
                p.controller_biases.fill(0.0);
                if spec.ouroboros == 1 {
                    let channels = layer.input_shape.channels;
                    for gate in &mut p.controller_biases[channels..2 * channels] {
                        *gate = spec.gate_bias;
                    }
                }
            }
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
            LayerParams::TuckerAttention(p) => {
                let channels = layer.input_shape.channels;
                let bound = (6.0 / (channels as f32 + channels as f32)).sqrt();
                rng.weights_with_scale(&mut p.weights, bound);
                p.biases.fill(0.0);
            }
            LayerParams::KimiKda(p) => {
                crate::kimi_kda::initialize_kimi_kda_params(p, |values, bound| {
                    rng.weights_with_scale(values, bound)
                })
                .expect("validated Kimi KDA parameter geometry");
            }
            LayerParams::MambaVision(p) => {
                let crate::LayerKind::MambaVision(spec) = &layer.kind else {
                    unreachable!("MambaVision parameters require a MambaVision layer");
                };
                let channels = layer.input_shape.channels;
                let filters = spec.filters;
                let d = filters / 2;
                let rank = spec.dt_rank;
                let state = spec.state;
                let hidden = filters * spec.ffn_ratio;
                p.values.fill(0.0);
                let blocks = crate::mambavision::mambavision_param_blocks(p)
                    .expect("validated MambaVision parameter geometry");
                let range = |name: &str| {
                    blocks
                        .iter()
                        .find(|block| block.name == name)
                        .expect("MambaVision parameter block exists")
                        .range
                        .clone()
                };
                let input = range("mambavision_in_proj_weights");
                rng.weights_with_scale(
                    &mut p.values[input],
                    (6.0 / (channels + filters) as f32).sqrt(),
                );
                let x_conv = range("mambavision_conv_x");
                rng.weights_with_scale(&mut p.values[x_conv], (2.0 / spec.conv_size as f32).sqrt());
                let z_conv = range("mambavision_conv_z");
                rng.weights_with_scale(&mut p.values[z_conv], (2.0 / spec.conv_size as f32).sqrt());
                let x_proj = range("mambavision_x_proj");
                let projected = rank + 2 * state;
                rng.weights_with_scale(
                    &mut p.values[x_proj],
                    (6.0 / (d + projected) as f32).sqrt(),
                );
                let dt_proj = range("mambavision_dt_proj");
                rng.weights_with_scale(&mut p.values[dt_proj], (rank as f32).powf(-0.5));
                let dt_bias = range("mambavision_dt_bias");
                for index in 0..d {
                    let log_dt = 0.001f32.ln() + rng.next_unit() * (0.1f32.ln() - 0.001f32.ln());
                    let dt = log_dt.exp();
                    p.values[dt_bias.start + index] = dt + (-(-dt).exp_m1()).ln();
                }
                let a_log = range("mambavision_A_log");
                for channel in 0..d {
                    for state_index in 0..state {
                        p.values[a_log.start + channel * state + state_index] =
                            ((state_index + 1) as f32).ln();
                    }
                }
                p.values[range("mambavision_D")].fill(1.0);
                let out_proj = range("mambavision_out_proj");
                rng.weights_with_scale(
                    &mut p.values[out_proj],
                    (6.0 / (filters + filters) as f32).sqrt(),
                );
                if let Some(block) = blocks
                    .iter()
                    .find(|block| block.name == "mambavision_res_proj")
                {
                    let res_proj = block.range.clone();
                    rng.weights_with_scale(
                        &mut p.values[res_proj],
                        (6.0 / (channels + filters) as f32).sqrt(),
                    );
                }
                p.values[range("mambavision_ln1_gamma")].fill(1.0);
                p.values[range("mambavision_ln2_gamma")].fill(1.0);
                let ffn1 = range("mambavision_ffn_w1");
                rng.weights_with_scale(&mut p.values[ffn1], (6.0 / filters as f32).sqrt());
                let ffn2 = range("mambavision_ffn_w2");
                rng.weights_with_scale(&mut p.values[ffn2], (6.0 / hidden as f32).sqrt());
            }
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
            LayerParams::GraphConv(p) => {
                let crate::LayerKind::GraphConv(spec) = &layer.kind else {
                    unreachable!("graph-conv parameters require a graph-conv layer");
                };
                let channels_per_group = layer.input_shape.channels / spec.groups;
                rng.weights(&mut p.projection.weights, channels_per_group);
                p.projection.biases.fill(0.0);
                if let Some(batch_norm) = &mut p.projection.batch_norm {
                    bn(batch_norm);
                }
                if let Some(weights) = &mut p.self_weights {
                    rng.weights(weights, channels_per_group);
                }
                if let Some(kernel) = &mut p.edge_kernel {
                    rng.weights_with_scale(kernel, 0.001);
                }
                if let Some(biases) = &mut p.edge_biases {
                    biases.fill(0.0);
                }
            }
            LayerParams::Dcnv4(p) => {
                let crate::LayerKind::Dcnv4(spec) = &layer.kind else {
                    unreachable!("DCNv4 parameters require a DCNv4 layer");
                };
                p.biases.fill(0.0);
                p.dummy_weights.fill(0.0);
                p.offset_weights.fill(0.0);
                p.offset_biases.fill(0.0);
                let points = spec.size * spec.size - usize::from(spec.remove_center);
                for group in 0..spec.groups {
                    for point in 0..points {
                        p.offset_biases[group * points * 3 + 2 * points + point] =
                            if spec.softmax { 0.0 } else { 1.0 };
                    }
                }
                if let Some(batch_norm) = &mut p.batch_norm {
                    bn(batch_norm);
                }
            }
            LayerParams::DsConv(p) => {
                let crate::LayerKind::DsConv(spec) = &layer.kind else {
                    unreachable!("dsconv parameters require a dsconv layer");
                };
                let fan_in = layer.input_shape.channels * spec.size;
                rng.weights_with_scale(&mut p.collapse.weights, (1.0 / fan_in as f32).sqrt());
                p.collapse.biases.fill(0.0);
                p.offset_weights.fill(0.0);
                p.offset_biases.fill(0.0);
            }
            LayerParams::Wmhf(p) => {
                let crate::LayerKind::Wmhf(spec) = &layer.kind else {
                    unreachable!("WMHF parameters require a WMHF layer");
                };
                for conv_params in [
                    &mut p.pre,
                    &mut p.local_dw3,
                    &mut p.local_dw5,
                    &mut p.local_dw7,
                    &mut p.local_mix,
                    &mut p.hf_gate,
                    &mut p.fuse,
                ] {
                    conv(conv_params, &mut rng);
                }
                p.scan_weights.fill(0.0);
                let channels = spec.global_channels;
                p.scan_weights[channels..2 * channels].fill(1.0);
                p.scan_weights[2 * channels..3 * channels].fill(1.0);
            }
            LayerParams::Clifford(p) => {
                let crate::LayerKind::Clifford(spec) = &layer.kind else {
                    unreachable!("Clifford parameters require a Clifford layer");
                };
                let channels = layer.input_shape.channels;
                p.layer_norm_gamma.fill(1.0);
                p.layer_norm_beta.fill(0.0);
                p.determinant_biases.fill(0.0);
                rng.weights_with_scale(
                    &mut p.determinant_weights,
                    (2.0 / (2 * channels).max(1) as f32).sqrt(),
                );
                for conv_params in &mut p.depthwise {
                    conv(conv_params, &mut rng);
                }
                initialize_clifford_branch(&mut p.local, channels, spec, &mut rng);
                p.layer_scale.fill(spec.layerscale_init);
                if let Some(global) = &mut p.global {
                    initialize_clifford_branch(global, channels, spec, &mut rng);
                }
                if let Some(fourier) = &mut p.fourier {
                    let modes = spec.fno_modes_h * spec.fno_modes_w;
                    let scale = (2.0 / (modes + channels).max(1) as f32).sqrt();
                    rng.weights_with_scale(&mut fourier.real, scale);
                    rng.weights_with_scale(&mut fourier.imaginary, scale);
                }
            }
            LayerParams::DeformConv(p) => {
                let crate::LayerKind::DeformConv(_) = &layer.kind else {
                    unreachable!("deform_conv parameters require a deform_conv layer");
                };
                conv(&mut p.main, &mut rng);
                rng.weights_with_scale(&mut p.offset_weights, 0.001);
                p.offset_biases.fill(0.0);
                if let Some(weights) = &mut p.mask_weights {
                    rng.weights_with_scale(weights, 0.001);
                }
                if let Some(biases) = &mut p.mask_biases {
                    biases.fill(0.0);
                }
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
fn initialize_clifford_branch(
    branch: &mut crate::clifford::CliffordBranchParams,
    channels: usize,
    spec: &crate::model::CliffordSpec,
    rng: &mut Rng,
) {
    let raw_channels = crate::clifford::raw_channels(spec, channels);
    let projection_scale = (2.0 / (raw_channels + channels).max(1) as f32).sqrt();
    rng.weights_with_scale(&mut branch.projection_weights, projection_scale);
    branch.projection_biases.fill(0.0);
    let gate_scale = (2.0 / (3 * channels).max(1) as f32).sqrt();
    rng.weights_with_scale(&mut branch.gate_weights, gate_scale);
    branch.gate_biases.fill(0.0);
}
struct Rng(u64);
impl Rng {
    fn weights(&mut self, weights: &mut [f32], fan_in: usize) {
        self.weights_with_scale(weights, (2.0 / fan_in.max(1) as f32).sqrt());
    }

    fn next_unit(&mut self) -> f32 {
        // SplitMix64: all seeds, including zero, produce a nonconstant stream.
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 40) as f32 / 16777216.0
    }

    fn weights_with_scale(&mut self, weights: &mut [f32], scale: f32) {
        for weight in weights {
            *weight = (2.0 * self.next_unit() - 1.0) * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigFile, ExecutableNetwork, LayerParams, Network};

    #[test]
    fn clifford_initialization_is_seeded_and_uses_source_scales() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "clifford-init.cfg",
                r#"[net]
width=2
height=2
channels=3
[clifford]
filters=3
shifts=1,2
cli_mode=full
gffn_mode=hybrid
global_op=fourier
fno_modes_h=2
fno_modes_w=2
dwconv_size=1
num_dwconv=1
layerscale=0.01
"#,
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 29);
        initialize_training_params(&network, &mut second.params, 29);
        assert_eq!(first.params, second.params);
        let LayerParams::Clifford(params) = &first.params[0] else {
            panic!("Clifford parameters")
        };
        initialize_training_params(&network, &mut second.params, 30);
        let LayerParams::Clifford(other_seed) = &second.params[0] else {
            unreachable!()
        };
        assert_ne!(params.determinant_weights, other_seed.determinant_weights);
        assert_eq!(params.layer_norm_gamma, vec![1.0; 3]);
        assert_eq!(params.layer_norm_beta, vec![0.0; 3]);
        assert_eq!(params.layer_scale, vec![0.01; 3]);
        let spec = match &network.layers[0].kind {
            crate::LayerKind::Clifford(spec) => spec,
            _ => unreachable!(),
        };
        let raw_channels = crate::clifford::raw_channels(spec, 3);
        let det_scale = (2.0f32 / 6.0).sqrt();
        let projection_scale = (2.0f32 / (raw_channels + 3) as f32).sqrt();
        let gate_scale = (2.0f32 / 9.0).sqrt();
        let fourier_scale = (2.0f32 / 7.0).sqrt();
        for (weights, scale) in [
            (&params.determinant_weights, det_scale),
            (&params.local.projection_weights, projection_scale),
            (&params.local.gate_weights, gate_scale),
            (
                &params.global.as_ref().unwrap().projection_weights,
                projection_scale,
            ),
            (&params.global.as_ref().unwrap().gate_weights, gate_scale),
            (&params.fourier.as_ref().unwrap().real, fourier_scale),
            (&params.fourier.as_ref().unwrap().imaginary, fourier_scale),
        ] {
            assert!(weights.iter().all(|weight| weight.abs() <= scale));
            assert!(weights.iter().any(|weight| *weight != 0.0));
        }
        let conv = &params.depthwise[0];
        assert!(conv.biases.iter().all(|value| *value == 0.0));
        assert!(conv.weights.iter().any(|value| *value != 0.0));
        let bn = conv.batch_norm.as_ref().unwrap();
        assert_eq!(bn.scales, vec![1.0; 3]);
        assert_eq!(bn.rolling_mean, vec![0.0; 3]);
        assert_eq!(bn.rolling_variance, vec![0.0; 3]);
    }

    #[test]
    fn dcnv4_initialization_zeros_projection_and_sets_mask_prior() {
        for (softmax, expected_mask) in [(false, 1.0), (true, 0.0)] {
            let cfg = ConfigFile::parse_str(
                "dcnv4-init.cfg",
                &format!(
                    "[net]\nwidth=1\nheight=1\nchannels=2\n\
                     [dcnv4]\nsize=1\nbatch_normalize=1\nsoftmax={}\n",
                    usize::from(softmax)
                ),
            )
            .unwrap();
            let network = Network::from_config(cfg).unwrap();
            let mut executable = ExecutableNetwork::new(network.clone());
            initialize_training_params(&network, &mut executable.params, 9);
            let LayerParams::Dcnv4(params) = &executable.params[0] else {
                panic!("DCNv4 params")
            };
            assert!(params.dummy_weights.iter().all(|value| *value == 0.0));
            assert!(params.offset_weights.iter().all(|value| *value == 0.0));
            assert_eq!(params.offset_biases[2], expected_mask);
            assert_eq!(params.biases, vec![0.0, 0.0]);
            let batch_norm = params.batch_norm.as_ref().unwrap();
            assert_eq!(batch_norm.scales, vec![1.0, 1.0]);
            assert_eq!(batch_norm.rolling_mean, vec![0.0, 0.0]);
            assert_eq!(batch_norm.rolling_variance, vec![0.0, 0.0]);
        }
    }

    #[test]
    fn deform_conv_initialization_is_seeded_and_uses_source_scales() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "deform-conv-init.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=4\n[deform_conv]\nfilters=6\ngroups=2\nsize=3\nbatch_normalize=1\nmask=1\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 123);
        initialize_training_params(&network, &mut second.params, 123);
        assert_eq!(first.params, second.params);
        let LayerParams::DeformConv(params) = &first.params[0] else {
            panic!("deform_conv params")
        };
        let fan_in = (4 / 2) * 3 * 3;
        let main_bound = (2.0f32 / fan_in as f32).sqrt();
        assert!(
            params
                .main
                .weights
                .iter()
                .all(|value| value.abs() <= main_bound)
        );
        assert!(params.main.biases.iter().all(|value| *value == 0.0));
        assert!(
            params
                .offset_weights
                .iter()
                .all(|value| value.abs() <= 0.001)
        );
        assert!(params.offset_biases.iter().all(|value| *value == 0.0));
        assert!(
            params
                .mask_weights
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| value.abs() <= 0.001)
        );
        assert!(
            params
                .mask_biases
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
        let batch_norm = params.main.batch_norm.as_ref().unwrap();
        assert!(batch_norm.scales.iter().all(|value| *value == 1.0));
        assert!(batch_norm.rolling_mean.iter().all(|value| *value == 0.0));
        assert!(
            batch_norm
                .rolling_variance
                .iter()
                .all(|value| *value == 0.0)
        );
    }

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
    fn tucker_attention_training_initialization_uses_source_uniform_bound() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "tucker-init.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=8\n[tucker]\nheads=4\nrank=2\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 53);
        initialize_training_params(&network, &mut second.params, 53);
        assert_eq!(first.params, second.params);
        let LayerParams::TuckerAttention(params) = &first.params[0] else {
            panic!("Tucker attention params")
        };
        let bound = (6.0f32 / 16.0).sqrt();
        assert!(params.weights.iter().any(|weight| *weight != 0.0));
        assert!(params.weights.iter().all(|weight| weight.abs() <= bound));
        assert_eq!(params.biases, vec![0.0; 8]);
    }

    #[test]
    fn mambavision_training_initialization_is_seeded_and_uses_source_bounds() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "mambavision-init.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=4\n\
                 [mambavision]\nfilters=6\nstate=3\nconv_size=3\ndt_rank=2\nffn_ratio=2\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 107);
        initialize_training_params(&network, &mut second.params, 107);
        assert_eq!(first.params, second.params);
        let LayerParams::MambaVision(params) = &first.params[0] else {
            panic!("MambaVision params")
        };
        let blocks = crate::mambavision::mambavision_param_blocks(params).unwrap();
        let within = |name: &str, bound: f32| {
            let block = blocks.iter().find(|block| block.name == name).unwrap();
            params.values[block.range.clone()]
                .iter()
                .all(|value| value.abs() <= bound)
        };
        assert!(within(
            "mambavision_in_proj_weights",
            (6.0f32 / 10.0).sqrt()
        ));
        assert!(within("mambavision_conv_x", (2.0f32 / 3.0).sqrt()));
        assert!(within("mambavision_conv_z", (2.0f32 / 3.0).sqrt()));
        assert!(within("mambavision_x_proj", (6.0f32 / 11.0).sqrt()));
        assert!(within("mambavision_dt_proj", (0.5f32).sqrt()));
        assert!(within("mambavision_out_proj", (6.0f32 / 12.0).sqrt()));
        assert!(within("mambavision_res_proj", (6.0f32 / 10.0).sqrt()));
        assert!(within("mambavision_ffn_w1", 1.0));
        assert!(within("mambavision_ffn_w2", (6.0f32 / 12.0).sqrt()));
        let a_log = blocks
            .iter()
            .find(|block| block.name == "mambavision_A_log")
            .unwrap();
        assert_eq!(
            &params.values[a_log.range.clone()],
            &[
                0.0, 0.6931472, 1.0986123, 0.0, 0.6931472, 1.0986123, 0.0, 0.6931472, 1.0986123,
            ]
        );
        let skip = blocks
            .iter()
            .find(|block| block.name == "mambavision_D")
            .unwrap();
        assert_eq!(&params.values[skip.range.clone()], &[1.0; 3]);
        let gamma = blocks
            .iter()
            .find(|block| block.name == "mambavision_ln1_gamma")
            .unwrap();
        assert_eq!(&params.values[gamma.range.clone()], &[1.0; 4]);
        let dt = blocks
            .iter()
            .find(|block| block.name == "mambavision_dt_bias")
            .unwrap();
        assert!(
            params.values[dt.range.clone()]
                .iter()
                .all(|value| (-7.0..-2.2).contains(value))
        );
    }

    #[test]
    fn kimi_kda_training_initialization_is_seeded_and_uses_source_bounds() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "kimi-init.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=8\n\
                 [kimi_kda]\nrank=2\nexperts=2\ntop_k=1\nhidden=4\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 131);
        initialize_training_params(&network, &mut second.params, 131);
        assert_eq!(first.params, second.params);
        let LayerParams::KimiKda(params) = &first.params[0] else {
            panic!("Kimi KDA params")
        };
        let blocks = crate::kimi_kda::kimi_kda_weight_blocks(8, 2, 2, 4).unwrap();
        let within = |name: &str, bound: f32| {
            let block = blocks.iter().find(|block| block.name == name).unwrap();
            params.weights[block.range.clone()]
                .iter()
                .all(|value| value.abs() <= bound)
        };
        let qk_bound = (6.0f32 / 10.0).sqrt();
        assert!(within("q", qk_bound));
        assert!(within("k", qk_bound));
        assert!(within("beta", qk_bound));
        assert!(within("decay", qk_bound));
        assert!(within("v", (6.0f32 / 16.0).sqrt()));
        assert!(within("router", qk_bound * 0.05));
        assert!(within("expert_w1", (6.0f32 / 12.0).sqrt()));
        for name in ["o", "expert_w2"] {
            let block = blocks.iter().find(|block| block.name == name).unwrap();
            assert!(
                params.weights[block.range.clone()]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
        assert_eq!(&params.biases[..8], &[1.0; 8]);
        assert_eq!(&params.biases[16..24], &[1.0; 8]);
        assert_eq!(&params.biases[44..46], &[-2.0; 2]);
        assert_eq!(&params.biases[46..48], &[2.0; 2]);
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
    fn graph_conv_initialization_is_seeded_and_uses_source_bounds() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-init.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=4\n[graph_conv]\nfilters=6\ngroups=2\nsize=3\npad=1\nbatch_normalize=1\nedge_mode=1\ngraph_use_self=1\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 83);
        initialize_training_params(&network, &mut second.params, 83);
        assert_eq!(first.params, second.params);
        let LayerParams::GraphConv(params) = &first.params[0] else {
            panic!("graph_conv params");
        };
        let projection_bound = (2.0f32 / 2.0).sqrt();
        assert!(
            params
                .projection
                .weights
                .iter()
                .all(|value| value.abs() <= projection_bound)
        );
        assert!(
            params
                .self_weights
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| value.abs() <= projection_bound)
        );
        assert!(
            params
                .edge_kernel
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| value.abs() <= 0.001)
        );
        assert!(params.projection.biases.iter().all(|value| *value == 0.0));
        assert!(
            params
                .edge_biases
                .as_ref()
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
        let batch_norm = params.projection.batch_norm.as_ref().unwrap();
        assert_eq!(batch_norm.scales, vec![1.0; 6]);
        assert_eq!(batch_norm.rolling_mean, vec![0.0; 6]);
        assert_eq!(batch_norm.rolling_variance, vec![0.0; 6]);
    }

    #[test]
    fn dsconv_initialization_is_seeded_and_uses_source_bound() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-init.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=2\n[dsconv]\nfilters=4\nsize=3\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 83);
        initialize_training_params(&network, &mut second.params, 83);
        assert_eq!(first.params, second.params);
        let LayerParams::DsConv(params) = &first.params[0] else {
            panic!("dsconv params")
        };
        let bound = (1.0f32 / (2.0 * 3.0)).sqrt();
        assert!(
            params
                .collapse
                .weights
                .iter()
                .all(|value| value.abs() <= bound)
        );
        assert!(params.collapse.weights.iter().any(|value| *value != 0.0));
        assert_eq!(params.collapse.biases, vec![0.0; 4]);
        assert!(params.offset_weights.iter().all(|value| *value == 0.0));
        assert!(params.offset_biases.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn wmhf_initialization_is_seeded_and_sets_scan_priors() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-init.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\n[wmhf]\nfilters=4\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut first = ExecutableNetwork::new(network.clone());
        let mut second = ExecutableNetwork::new(network.clone());
        initialize_training_params(&network, &mut first.params, 83);
        initialize_training_params(&network, &mut second.params, 83);
        assert_eq!(first.params, second.params);
        let LayerParams::Wmhf(params) = &first.params[0] else {
            panic!("wmhf params")
        };
        assert!(params.pre.weights.iter().any(|value| *value != 0.0));
        assert!(params.local_dw3.weights.iter().any(|value| *value != 0.0));
        assert!(params.local_dw5.weights.iter().any(|value| *value != 0.0));
        assert!(params.local_dw7.weights.iter().any(|value| *value != 0.0));
        assert!(params.local_mix.weights.iter().any(|value| *value != 0.0));
        assert!(params.hf_gate.weights.iter().any(|value| *value != 0.0));
        assert!(params.fuse.weights.iter().any(|value| *value != 0.0));
        assert_eq!(params.pre.batch_norm.as_ref().unwrap().scales, vec![1.0; 4]);
        assert!(params.local_dw3.batch_norm.is_none());
        assert!(params.local_dw5.batch_norm.is_none());
        assert!(params.local_dw7.batch_norm.is_none());
        assert_eq!(
            params.local_mix.batch_norm.as_ref().unwrap().scales.len(),
            2
        );
        assert!(params.hf_gate.batch_norm.is_none());
        assert_eq!(params.fuse.batch_norm.as_ref().unwrap().scales.len(), 4);
        let channels = network.layers[0]
            .weights()
            .unwrap()
            .entries
            .last()
            .unwrap()
            .len
            / 4;
        assert_eq!(&params.scan_weights[..channels], vec![0.0; channels]);
        assert_eq!(
            &params.scan_weights[channels..2 * channels],
            vec![1.0; channels]
        );
        assert_eq!(
            &params.scan_weights[2 * channels..3 * channels],
            vec![1.0; channels]
        );
        assert_eq!(&params.scan_weights[3 * channels..], vec![0.0; channels]);
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
