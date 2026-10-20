use super::*;

pub(super) fn conv_weight_shape(layer: &crate::model::Layer) -> Vec<usize> {
    match &layer.kind {
        LayerKind::Convolutional(spec) => {
            let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
            vec![spec.filters, grouped_channels, spec.size, spec.size]
        }
        _ => Vec::new(),
    }
}

pub(super) fn vit_geometry(
    layer: &crate::model::Layer,
    spec: &crate::model::VitSpec,
) -> crate::vit::VitGeometry {
    crate::vit::VitGeometry {
        batch: layer.input_shape.batch,
        channels: layer.input_shape.channels,
        height: layer.input_shape.height,
        width: layer.input_shape.width,
        patch_size: spec.patch_size,
        patch_stride: spec.patch_stride,
        patch_pad: spec.patch_pad,
        hidden: spec.dim,
        heads: spec.heads,
        mlp_dim: spec.mlp_dim,
        ln_eps: 1.0e-5,
        activation: spec.activation.clone(),
        learned_pos_embed: spec.learned_pos_embed,
        pos_init_zero: spec.pos_init_zero,
        msa: spec
            .msa_enabled
            .then_some((spec.msa_block_size, spec.msa_topk)),
        flash_attention: spec.flash_attention,
        vc_attention: spec.vc_attention,
        vc_expcast: spec.vc_expcast,
        vc_perm: spec.vc_perm,
    }
}

pub(super) fn zero_vit_weights(weights: &mut crate::vit::VitWeights) {
    for values in [
        &mut weights.patch_embed,
        &mut weights.patch_bias,
        &mut weights.qkv_weight,
        &mut weights.qkv_bias,
        &mut weights.ln1_gamma,
        &mut weights.ln1_beta,
        &mut weights.wo,
        &mut weights.wo_bias,
        &mut weights.ln2_gamma,
        &mut weights.ln2_beta,
        &mut weights.ffn_w1,
        &mut weights.ffn_b1,
        &mut weights.ffn_w2,
        &mut weights.ffn_b2,
        &mut weights.pos_embed,
        &mut weights.mhc,
    ] {
        values.fill(0.0);
    }
}

pub(super) fn default_layer_params(layer: &crate::model::Layer) -> LayerParams {
    match &layer.kind {
        LayerKind::FoveationFilm(spec) => LayerParams::FoveationFilm(FoveationFilmParams {
            w_embed: vec![0.0; spec.embed_dim * spec.param_count],
            b_embed: vec![0.0; spec.embed_dim],
            w_gamma: vec![0.0; layer.input_shape.channels * spec.embed_dim],
            b_gamma: vec![0.0; layer.input_shape.channels],
            w_beta: vec![0.0; layer.input_shape.channels * spec.embed_dim],
            b_beta: vec![0.0; layer.input_shape.channels],
        }),
        LayerKind::RecursiveBlock(spec) => {
            let controller_outputs = if spec.ouroboros == 1 {
                layer.input_shape.channels * 2
            } else {
                0
            };
            let controller_inputs = if spec.ouroboros == 1 {
                layer.input_shape.channels + 1
            } else {
                0
            };
            LayerParams::RecursiveBlock(RecursiveBlockParams {
                body: spec.body.iter().map(default_layer_params).collect(),
                controller_weights: vec![0.0; controller_outputs.saturating_mul(controller_inputs)],
                controller_biases: vec![0.0; controller_outputs],
            })
        }
        LayerKind::HyperbolicFoveation(spec) => {
            let param_count = spec.param_count();
            let mut hf_params = vec![0.0; param_count];
            hf_params[0] = spec.init_cx;
            hf_params[1] = spec.init_cy;
            match spec.radius_mode {
                crate::model::HfRadiusMode::Isotropic => hf_params[2] = spec.init_radius,
                crate::model::HfRadiusMode::Anisotropic => {
                    hf_params[2] = spec.init_radius_x;
                    hf_params[3] = spec.init_radius_y;
                }
            }
            hf_params[param_count - 1] = spec.init_strength;
            let input_dim = spec
                .predictor_input_dim()
                .expect("validated hyperbolic-foveation predictor dimensions");
            LayerParams::HyperbolicFoveation(HyperbolicFoveationParams {
                hf_params,
                pred_w1: vec![0.0; spec.pred_hidden * input_dim],
                pred_b1: vec![0.0; spec.pred_hidden],
                pred_w2: vec![0.0; param_count * spec.pred_hidden],
                pred_b2: hyperbolic_foveation_predictor_initial_biases(spec),
            })
        }
        LayerKind::BatchNorm(_) => {
            LayerParams::BatchNorm(default_batchnorm_params(layer.input_shape.channels))
        }
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerParams::None;
            }
            LayerParams::Convolutional(default_convolutional_params(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ))
        }
        LayerKind::Deconvolutional(spec) => {
            LayerParams::Convolutional(default_convolutional_params(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                1,
                false,
            ))
        }
        LayerKind::Connected(spec) => LayerParams::Connected(default_connected_params(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::ObbRefine(spec) => {
            LayerParams::Connected(default_connected_params(spec.inputs, 5, false))
        }
        LayerKind::Local(spec) => LayerParams::Local(default_local_params(
            layer.input_shape.channels,
            layer.output_shape.width * layer.output_shape.height,
            spec.filters,
            spec.size,
        )),
        LayerKind::LayaScorer(spec) => {
            LayerParams::LayaScorer(crate::laya_scorer::LayaScorerWeights::zeroed(spec.hidden))
        }
        LayerKind::LayaActHead(spec) => LayerParams::LayaActHead(
            crate::laya_act_head::LayaActHeadWeights::zeroed(spec.hidden, spec.fc_dim, spec.n_act),
        ),
        LayerKind::LayaEncoder(spec) => {
            LayerParams::LayaEncoder(crate::laya_encoder::LayaEncoderWeights::zeroed(
                spec.hidden,
                spec.n_layers,
                spec.intermediate,
                spec.vocab,
            ))
        }
        LayerKind::LayaHead(spec) => LayerParams::LayaHead(
            crate::laya_head::LayaHeadWeights::zeroed(spec.hidden, spec.n_layers, spec.ffn_dim),
        ),
        LayerKind::DepthAnything(spec) => {
            let geometry = spec.geometry(layer.input_shape);
            LayerParams::DepthAnything(spec.external_weights.clone().unwrap_or_else(|| {
                crate::depth_anything::DepthAnythingWeights::zeroed(&geometry)
                    .expect("validated depth_anything layer geometry")
            }))
        }
        LayerKind::DptHead(spec) => LayerParams::DptHead(
            crate::dpt_head::DptHeadWeights::zeroed(spec.geometry(layer.input_shape))
                .expect("validated dpt_head layer geometry"),
        ),
        LayerKind::Vit(spec) => LayerParams::Vit(
            crate::vit::VitWeights::zeroed(
                &vit_geometry(layer, spec),
                layer.output_shape.height,
                layer.output_shape.width,
            )
            .expect("validated ViT layer geometry"),
        ),
        LayerKind::Swin(spec) => LayerParams::Swin(
            crate::swin::SwinWeights::zeroed(
                layer.input_shape.channels,
                spec.filters,
                spec.heads,
                spec.size,
                spec.ffn_ratio,
            )
            .expect("validated Swin layer geometry"),
        ),
        LayerKind::TuckerAttention(spec) => LayerParams::TuckerAttention(
            crate::tucker_attention::TuckerAttentionParams::zeroed(
                layer.input_shape.channels,
                spec.heads,
                spec.rank_q,
                spec.rank_k,
                spec.rank_v,
                spec.rank_o,
            )
            .expect("validated Tucker attention layer geometry"),
        ),
        LayerKind::MambaVision(spec) => LayerParams::MambaVision(
            crate::mambavision::MambaVisionParams::zeroed(
                layer.input_shape.channels,
                spec.filters,
                spec.state,
                spec.conv_size,
                spec.dt_rank,
                spec.ffn_ratio,
            )
            .expect("validated MambaVision layer geometry"),
        ),
        LayerKind::KimiKda(spec) => LayerParams::KimiKda(
            crate::kimi_kda::KimiKdaParams::zeroed(
                layer.input_shape.channels,
                spec.rank,
                spec.experts,
                spec.hidden,
            )
            .expect("validated Kimi KDA layer geometry"),
        ),
        LayerKind::Moe(spec) => {
            LayerParams::Moe(default_moe_params(layer.input_shape.channels, spec))
        }
        LayerKind::EmlConvolutional(spec) => LayerParams::EmlConvolutional(
            default_eml_convolutional_params(layer.input_shape.channels, spec),
        ),
        LayerKind::Dcnv4(spec) => {
            LayerParams::Dcnv4(default_dcnv4_params(layer.input_shape.channels, spec))
        }
        LayerKind::GraphConv(spec) => {
            LayerParams::GraphConv(default_graph_conv_params(layer.input_shape.channels, spec))
        }
        LayerKind::DeformConv(spec) => {
            LayerParams::DeformConv(default_deform_conv_params(layer.input_shape.channels, spec))
        }
        LayerKind::DsConv(spec) => {
            LayerParams::DsConv(default_dsconv_params(layer.input_shape.channels, spec))
        }
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerParams::None,
            len => LayerParams::Shortcut(ShortcutParams {
                weights: vec![0.0; len],
            }),
        },
        LayerKind::Gru(spec) => LayerParams::Gru(default_gru_params(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::Rnn(spec) => LayerParams::Rnn(RnnParams {
            input: default_connected_params(
                layer.input_shape.element_count(),
                spec.hidden,
                spec.batch_normalize,
            ),
            self_params: default_connected_params(spec.hidden, spec.hidden, spec.batch_normalize),
            output: default_connected_params(spec.hidden, spec.output, spec.batch_normalize),
        }),
        LayerKind::Lstm(spec) => LayerParams::Lstm(LstmParams {
            wf: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wi: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wg: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            wo: default_connected_params(spec.output, spec.output, spec.batch_normalize),
            uf: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ui: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ug: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            uo: default_connected_params(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Crnn(spec) => LayerParams::Crnn(CrnnParams {
            input: default_convolutional_params(
                layer.input_shape.channels,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            self_params: default_convolutional_params(
                spec.hidden,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            output: default_convolutional_params(
                spec.hidden,
                spec.output,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Wmhf(spec) => {
            LayerParams::Wmhf(default_wmhf_params(layer.input_shape.channels, spec))
        }
        LayerKind::Clifford(spec) => {
            LayerParams::Clifford(CliffordParams::zeroed(layer.input_shape.channels, spec))
        }
        _ => LayerParams::None,
    }
}

pub(super) fn default_layer_train_buffers(layer: &crate::model::Layer) -> LayerTrainBuffers {
    match &layer.kind {
        LayerKind::FoveationFilm(spec) => {
            LayerTrainBuffers::FoveationFilm(FoveationFilmTrainBuffers {
                w_embed_updates: vec![0.0; spec.embed_dim * spec.param_count],
                b_embed_updates: vec![0.0; spec.embed_dim],
                w_gamma_updates: vec![0.0; layer.input_shape.channels * spec.embed_dim],
                b_gamma_updates: vec![0.0; layer.input_shape.channels],
                w_beta_updates: vec![0.0; layer.input_shape.channels * spec.embed_dim],
                b_beta_updates: vec![0.0; layer.input_shape.channels],
            })
        }
        LayerKind::RecursiveBlock(spec) => {
            let controller_outputs = if spec.ouroboros == 1 {
                layer.input_shape.channels * 2
            } else {
                0
            };
            let controller_inputs = if spec.ouroboros == 1 {
                layer.input_shape.channels + 1
            } else {
                0
            };
            LayerTrainBuffers::RecursiveBlock(RecursiveBlockTrainBuffers {
                body: spec.body.iter().map(default_layer_train_buffers).collect(),
                controller_weight_updates: vec![
                    0.0;
                    controller_outputs.saturating_mul(controller_inputs)
                ],
                controller_bias_updates: vec![0.0; controller_outputs],
            })
        }
        LayerKind::HyperbolicFoveation(spec) => {
            let param_count = spec.param_count();
            let input_dim = spec
                .predictor_input_dim()
                .expect("validated hyperbolic-foveation predictor dimensions");
            LayerTrainBuffers::HyperbolicFoveation(HyperbolicFoveationTrainBuffers {
                hf_param_updates: vec![0.0; param_count],
                pred_w1_updates: vec![0.0; spec.pred_hidden * input_dim],
                pred_b1_updates: vec![0.0; spec.pred_hidden],
                pred_w2_updates: vec![0.0; param_count * spec.pred_hidden],
                pred_b2_updates: vec![0.0; if spec.pred_hidden > 0 { param_count } else { 0 }],
            })
        }
        LayerKind::BatchNorm(_) => LayerTrainBuffers::BatchNorm(default_batchnorm_train_buffers(
            layer.input_shape.channels,
        )),
        LayerKind::Convolutional(spec) => {
            if spec.share_index.is_some() {
                return LayerTrainBuffers::None;
            }
            LayerTrainBuffers::Convolutional(default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ))
        }
        LayerKind::Deconvolutional(spec) => {
            LayerTrainBuffers::Convolutional(default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.filters,
                spec.size,
                1,
                false,
            ))
        }
        LayerKind::Connected(spec) => {
            LayerTrainBuffers::Connected(default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ))
        }
        LayerKind::Local(spec) => LayerTrainBuffers::Local(default_local_train_buffers(
            layer.input_shape.channels,
            layer.output_shape.width * layer.output_shape.height,
            spec.filters,
            spec.size,
        )),
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerTrainBuffers::None,
            len => LayerTrainBuffers::Shortcut(ShortcutTrainBuffers {
                weight_updates: vec![0.0; len],
            }),
        },
        LayerKind::Gru(spec) => LayerTrainBuffers::Gru(default_gru_train_buffers(
            layer.input_shape.element_count(),
            spec.output,
            spec.batch_normalize,
        )),
        LayerKind::Rnn(spec) => LayerTrainBuffers::Rnn(RnnTrainBuffers {
            input: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.hidden,
                spec.batch_normalize,
            ),
            self_params: default_connected_train_buffers(
                spec.hidden,
                spec.hidden,
                spec.batch_normalize,
            ),
            output: default_connected_train_buffers(spec.hidden, spec.output, spec.batch_normalize),
        }),
        LayerKind::Lstm(spec) => LayerTrainBuffers::Lstm(LstmTrainBuffers {
            wf: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wi: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wg: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            wo: default_connected_train_buffers(spec.output, spec.output, spec.batch_normalize),
            uf: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ui: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            ug: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
            uo: default_connected_train_buffers(
                layer.input_shape.element_count(),
                spec.output,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Crnn(spec) => LayerTrainBuffers::Crnn(CrnnTrainBuffers {
            input: default_convolutional_train_buffers(
                layer.input_shape.channels,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            self_params: default_convolutional_train_buffers(
                spec.hidden,
                spec.hidden,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
            output: default_convolutional_train_buffers(
                spec.hidden,
                spec.output,
                spec.size,
                spec.groups,
                spec.batch_normalize,
            ),
        }),
        LayerKind::Moe(spec) => {
            LayerTrainBuffers::Moe(default_moe_train_buffers(layer.input_shape.channels, spec))
        }
        LayerKind::EmlConvolutional(spec) => LayerTrainBuffers::EmlConvolutional(
            default_eml_convolutional_train_buffers(layer.input_shape.channels, spec),
        ),
        LayerKind::Dcnv4(spec) => LayerTrainBuffers::Dcnv4(default_dcnv4_train_buffers(
            layer.input_shape.channels,
            spec,
        )),
        LayerKind::GraphConv(spec) => LayerTrainBuffers::GraphConv(
            default_graph_conv_train_buffers(layer.input_shape.channels, spec),
        ),
        LayerKind::DeformConv(spec) => LayerTrainBuffers::DeformConv(
            default_deform_conv_train_buffers(layer.input_shape.channels, spec),
        ),
        LayerKind::DsConv(spec) => LayerTrainBuffers::DsConv(default_dsconv_train_buffers(
            layer.input_shape.channels,
            spec,
        )),
        LayerKind::Wmhf(spec) => {
            LayerTrainBuffers::Wmhf(default_wmhf_train_buffers(layer.input_shape.channels, spec))
        }
        LayerKind::Clifford(spec) => LayerTrainBuffers::Clifford(CliffordTrainBuffers::zeroed(
            layer.input_shape.channels,
            spec,
        )),
        LayerKind::TuckerAttention(spec) => {
            let params = crate::tucker_attention::TuckerAttentionParams::zeroed(
                layer.input_shape.channels,
                spec.heads,
                spec.rank_q,
                spec.rank_k,
                spec.rank_v,
                spec.rank_o,
            )
            .expect("validated Tucker attention layer geometry");
            LayerTrainBuffers::TuckerAttention(TuckerAttentionTrainBuffers {
                bias_updates: vec![0.0; params.biases.len()],
                weight_updates: vec![0.0; params.weights.len()],
            })
        }
        LayerKind::MambaVision(spec) => {
            let params = crate::mambavision::MambaVisionParams::zeroed(
                layer.input_shape.channels,
                spec.filters,
                spec.state,
                spec.conv_size,
                spec.dt_rank,
                spec.ffn_ratio,
            )
            .expect("validated MambaVision layer geometry");
            LayerTrainBuffers::MambaVision(MambaVisionTrainBuffers {
                updates: vec![0.0; params.parameter_count()],
            })
        }
        LayerKind::KimiKda(spec) => {
            let params = crate::kimi_kda::KimiKdaParams::zeroed(
                layer.input_shape.channels,
                spec.rank,
                spec.experts,
                spec.hidden,
            )
            .expect("validated Kimi KDA layer geometry");
            LayerTrainBuffers::KimiKda(KimiKdaTrainBuffers {
                bias_updates: vec![0.0; params.biases.len()],
                weight_updates: vec![0.0; params.weights.len()],
            })
        }
        LayerKind::DptHead(spec) => {
            let total = crate::dpt_head::dpt_head_weight_layout(spec.geometry(layer.input_shape))
                .expect("validated dpt_head weight geometry")
                .total;
            LayerTrainBuffers::DptHead(DptHeadTrainBuffers {
                weight_updates: vec![0.0; total],
                has_gradient: false,
            })
        }
        LayerKind::Swin(spec) => {
            let updates = crate::swin::SwinWeights::zeroed_gradients(
                layer.input_shape.channels,
                spec.filters,
                spec.heads,
                spec.size,
                spec.ffn_ratio,
            )
            .expect("validated Swin layer geometry");
            LayerTrainBuffers::Swin(SwinTrainBuffers { updates })
        }
        LayerKind::Vit(spec) => {
            let mut updates = crate::vit::VitWeights::zeroed(
                &vit_geometry(layer, spec),
                layer.output_shape.height,
                layer.output_shape.width,
            )
            .expect("validated ViT layer geometry");
            zero_vit_weights(&mut updates);
            LayerTrainBuffers::Vit(VitTrainBuffers { updates })
        }
        _ => LayerTrainBuffers::None,
    }
}

/// `[moe_convolutional]`'s parameters: one ordinary convolution per expert plus
/// the router's `[experts, channels]` weights and `[experts]` biases.
pub(super) fn default_moe_params(input_channels: usize, spec: &crate::model::MoeSpec) -> MoeParams {
    let expert = &spec.expert;
    MoeParams {
        router_weights: vec![0.0; spec.experts * input_channels],
        router_biases: vec![0.0; spec.experts],
        experts: (0..spec.experts)
            .map(|_| {
                default_convolutional_params(
                    input_channels,
                    expert.filters,
                    expert.size,
                    expert.groups,
                    expert.batch_normalize,
                )
            })
            .collect(),
    }
}

pub(super) fn default_dcnv4_params(input_channels: usize, spec: &Dcnv4Spec) -> Dcnv4Params {
    let (dummy_count, offset_weight_count, offset_dim) =
        crate::model::dcnv4_parameter_layout(input_channels, spec)
            .expect("validated DCNv4 parameter geometry");
    Dcnv4Params {
        biases: vec![0.0; spec.filters],
        batch_norm: spec
            .batch_normalize
            .then(|| default_batchnorm_params(spec.filters)),
        dummy_weights: vec![0.0; dummy_count],
        offset_weights: vec![0.0; offset_weight_count],
        offset_biases: vec![0.0; offset_dim],
    }
}

pub(super) fn default_eml_convolutional_params(
    input_channels: usize,
    spec: &EmlConvolutionalSpec,
) -> EmlConvolutionalParams {
    let branch = &spec.branch;
    let make = || {
        default_convolutional_params(
            input_channels,
            branch.filters,
            branch.size,
            branch.groups,
            branch.batch_normalize,
        )
    };
    EmlConvolutionalParams {
        input: make(),
        self_params: make(),
    }
}

pub(super) fn default_moe_train_buffers(
    input_channels: usize,
    spec: &crate::model::MoeSpec,
) -> MoeTrainBuffers {
    let expert = &spec.expert;
    MoeTrainBuffers {
        router_weight_updates: vec![0.0; spec.experts * input_channels],
        router_bias_updates: vec![0.0; spec.experts],
        experts: (0..spec.experts)
            .map(|_| {
                default_convolutional_train_buffers(
                    input_channels,
                    expert.filters,
                    expert.size,
                    expert.groups,
                    expert.batch_normalize,
                )
            })
            .collect(),
    }
}

pub(super) fn default_wmhf_params(input_channels: usize, spec: &crate::model::WmhfSpec) -> WmhfParams {
    let local = spec.local_channels;
    let global = spec.global_channels;
    WmhfParams {
        pre: default_convolutional_params(input_channels, spec.filters, 1, 1, spec.batch_normalize),
        local_dw3: default_convolutional_params(local, local, 3, local, false),
        local_dw5: default_convolutional_params(local, local, 5, local, false),
        local_dw7: default_convolutional_params(local, local, 7, local, false),
        local_mix: default_convolutional_params(3 * local, local, 1, 1, spec.batch_normalize),
        hf_gate: default_convolutional_params(global, spec.filters, 1, 1, false),
        fuse: default_convolutional_params(spec.filters, spec.filters, 1, 1, spec.batch_normalize),
        scan_weights: vec![0.0; 4 * global],
    }
}

pub(super) fn default_wmhf_train_buffers(
    input_channels: usize,
    spec: &crate::model::WmhfSpec,
) -> WmhfTrainBuffers {
    let local = spec.local_channels;
    let global = spec.global_channels;
    WmhfTrainBuffers {
        pre: default_convolutional_train_buffers(
            input_channels,
            spec.filters,
            1,
            1,
            spec.batch_normalize,
        ),
        local_dw3: default_convolutional_train_buffers(local, local, 3, local, false),
        local_dw5: default_convolutional_train_buffers(local, local, 5, local, false),
        local_dw7: default_convolutional_train_buffers(local, local, 7, local, false),
        local_mix: default_convolutional_train_buffers(
            3 * local,
            local,
            1,
            1,
            spec.batch_normalize,
        ),
        hf_gate: default_convolutional_train_buffers(global, spec.filters, 1, 1, false),
        fuse: default_convolutional_train_buffers(
            spec.filters,
            spec.filters,
            1,
            1,
            spec.batch_normalize,
        ),
        scan_weight_updates: vec![0.0; 4 * global],
    }
}

pub(super) fn default_graph_conv_params(input_channels: usize, spec: &GraphConvSpec) -> GraphConvParams {
    let channels_per_group = input_channels / spec.groups;
    let projection_size = spec.filters * channels_per_group;
    let kernel_area = spec.size * spec.size;
    GraphConvParams {
        projection: default_convolutional_params(
            input_channels,
            spec.filters,
            1,
            spec.groups,
            spec.batch_normalize,
        ),
        self_weights: spec.use_self.then(|| vec![0.0; projection_size]),
        edge_kernel: (spec.edge_mode == 1)
            .then(|| vec![0.0; spec.groups * kernel_area * 2 * channels_per_group]),
        edge_biases: (spec.edge_mode == 1).then(|| vec![0.0; spec.groups * kernel_area]),
    }
}

pub(super) fn default_graph_conv_train_buffers(
    input_channels: usize,
    spec: &GraphConvSpec,
) -> GraphConvTrainBuffers {
    let channels_per_group = input_channels / spec.groups;
    let projection_size = spec.filters * channels_per_group;
    let kernel_area = spec.size * spec.size;
    GraphConvTrainBuffers {
        projection: default_convolutional_train_buffers(
            input_channels,
            spec.filters,
            1,
            spec.groups,
            spec.batch_normalize,
        ),
        self_weight_updates: spec.use_self.then(|| vec![0.0; projection_size]),
        edge_kernel_updates: (spec.edge_mode == 1)
            .then(|| vec![0.0; spec.groups * kernel_area * 2 * channels_per_group]),
        edge_bias_updates: (spec.edge_mode == 1).then(|| vec![0.0; spec.groups * kernel_area]),
    }
}

pub(super) fn default_dcnv4_train_buffers(input_channels: usize, spec: &Dcnv4Spec) -> Dcnv4TrainBuffers {
    let (_, offset_weight_count, offset_dim) =
        crate::model::dcnv4_parameter_layout(input_channels, spec)
            .expect("validated DCNv4 parameter geometry");
    Dcnv4TrainBuffers {
        bias_updates: vec![0.0; spec.filters],
        batch_norm: spec.batch_normalize.then(|| BatchNormTrainBuffers {
            scale_updates: vec![0.0; spec.filters],
            mean_delta: vec![0.0; spec.filters],
            variance_delta: vec![0.0; spec.filters],
        }),
        offset_weight_updates: vec![0.0; offset_weight_count],
        offset_bias_updates: vec![0.0; offset_dim],
    }
}

pub(super) fn deform_conv_layout_counts(
    input_channels: usize,
    spec: &DeformConvSpec,
) -> (usize, usize, usize, usize) {
    let kernel_area = spec.size * spec.size;
    let offset_filters = 2 * kernel_area;
    let column_channels = input_channels * kernel_area;
    let offset_weights = offset_filters * column_channels;
    let mask_weights = if spec.use_mask {
        kernel_area * column_channels
    } else {
        0
    };
    (kernel_area, offset_filters, offset_weights, mask_weights)
}

pub(super) fn dsconv_layout_counts(input_channels: usize, spec: &crate::model::DsConvSpec) -> (usize, usize) {
    (
        spec.filters * input_channels * spec.size,
        2 * spec.size * input_channels * 9,
    )
}

pub(super) fn default_dsconv_params(input_channels: usize, spec: &crate::model::DsConvSpec) -> DsConvParams {
    let (weights, offset_weights) = dsconv_layout_counts(input_channels, spec);
    DsConvParams {
        collapse: ConvolutionalParams {
            biases: vec![0.0; spec.filters],
            batch_norm: None,
            weights: vec![0.0; weights],
        },
        offset_weights: vec![0.0; offset_weights],
        offset_biases: vec![0.0; 2 * spec.size],
    }
}

pub(super) fn default_dsconv_train_buffers(
    input_channels: usize,
    spec: &crate::model::DsConvSpec,
) -> DsConvTrainBuffers {
    let (weights, offset_weights) = dsconv_layout_counts(input_channels, spec);
    DsConvTrainBuffers {
        collapse: ConvolutionalTrainBuffers {
            bias_updates: vec![0.0; spec.filters],
            batch_norm: None,
            weight_updates: vec![0.0; weights],
        },
        offset_weight_updates: vec![0.0; offset_weights],
        offset_bias_updates: vec![0.0; 2 * spec.size],
    }
}

pub(super) fn default_deform_conv_params(input_channels: usize, spec: &DeformConvSpec) -> DeformConvParams {
    let (kernel_area, offset_filters, offset_weights, mask_weights) =
        deform_conv_layout_counts(input_channels, spec);
    DeformConvParams {
        main: default_convolutional_params(
            input_channels,
            spec.filters,
            spec.size,
            spec.groups,
            spec.batch_normalize,
        ),
        offset_weights: vec![0.0; offset_weights],
        offset_biases: vec![0.0; offset_filters],
        mask_weights: spec.use_mask.then(|| vec![0.0; mask_weights]),
        mask_biases: spec.use_mask.then(|| vec![0.0; kernel_area]),
    }
}

pub(super) fn default_deform_conv_train_buffers(
    input_channels: usize,
    spec: &DeformConvSpec,
) -> DeformConvTrainBuffers {
    let (kernel_area, offset_filters, offset_weights, mask_weights) =
        deform_conv_layout_counts(input_channels, spec);
    DeformConvTrainBuffers {
        main: default_convolutional_train_buffers(
            input_channels,
            spec.filters,
            spec.size,
            spec.groups,
            spec.batch_normalize,
        ),
        offset_weight_updates: vec![0.0; offset_weights],
        offset_bias_updates: vec![0.0; offset_filters],
        mask_weight_updates: spec.use_mask.then(|| vec![0.0; mask_weights]),
        mask_bias_updates: spec.use_mask.then(|| vec![0.0; kernel_area]),
    }
}

pub(super) fn default_eml_convolutional_train_buffers(
    input_channels: usize,
    spec: &EmlConvolutionalSpec,
) -> EmlConvolutionalTrainBuffers {
    let branch = &spec.branch;
    let make = || {
        default_convolutional_train_buffers(
            input_channels,
            branch.filters,
            branch.size,
            branch.groups,
            branch.batch_normalize,
        )
    };
    EmlConvolutionalTrainBuffers {
        input: make(),
        self_params: make(),
    }
}

pub(super) fn default_batchnorm_params(channels: usize) -> BatchNormParams {
    BatchNormParams {
        biases: vec![0.0; channels],
        scales: vec![1.0; channels],
        rolling_mean: vec![0.0; channels],
        rolling_variance: vec![1.0; channels],
    }
}

pub(super) fn default_batchnorm_train_buffers(channels: usize) -> StandaloneBatchNormTrainBuffers {
    StandaloneBatchNormTrainBuffers {
        bias_updates: vec![0.0; channels],
        scale_updates: vec![0.0; channels],
    }
}

pub(super) fn default_connected_params(
    inputs: usize,
    outputs: usize,
    batch_normalize: bool,
) -> ConnectedParams {
    ConnectedParams {
        biases: vec![0.0; outputs],
        batch_norm: batch_normalize.then(|| BatchNormParams {
            biases: vec![0.0; outputs],
            scales: vec![0.0; outputs],
            rolling_mean: vec![0.0; outputs],
            rolling_variance: vec![0.0; outputs],
        }),
        weights: vec![0.0; outputs * inputs],
    }
}

pub(super) fn default_gru_params(inputs: usize, output: usize, batch_normalize: bool) -> GruParams {
    GruParams {
        input_z: default_connected_params(inputs, output, batch_normalize),
        state_z: default_connected_params(output, output, batch_normalize),
        input_r: default_connected_params(inputs, output, batch_normalize),
        state_r: default_connected_params(output, output, batch_normalize),
        input_h: default_connected_params(inputs, output, batch_normalize),
        state_h: default_connected_params(output, output, batch_normalize),
    }
}

pub(super) fn default_convolutional_params(
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
) -> ConvolutionalParams {
    let groups = groups.max(1);
    ConvolutionalParams {
        biases: vec![0.0; filters],
        batch_norm: batch_normalize.then(|| BatchNormParams {
            biases: vec![0.0; filters],
            scales: vec![0.0; filters],
            rolling_mean: vec![0.0; filters],
            rolling_variance: vec![0.0; filters],
        }),
        weights: vec![0.0; filters * (input_channels / groups) * size * size],
    }
}

pub(super) fn default_local_params(
    input_channels: usize,
    locations: usize,
    filters: usize,
    size: usize,
) -> LocalParams {
    LocalParams {
        biases: vec![0.0; locations * filters],
        weights: vec![0.0; locations * filters * input_channels * size * size],
    }
}

pub(super) fn default_connected_train_buffers(
    inputs: usize,
    outputs: usize,
    batch_normalize: bool,
) -> ConnectedTrainBuffers {
    ConnectedTrainBuffers {
        bias_updates: vec![0.0; outputs],
        batch_norm: batch_normalize.then(|| BatchNormTrainBuffers {
            scale_updates: vec![0.0; outputs],
            mean_delta: vec![0.0; outputs],
            variance_delta: vec![0.0; outputs],
        }),
        weight_updates: vec![0.0; outputs * inputs],
    }
}

pub(super) fn default_gru_train_buffers(
    inputs: usize,
    output: usize,
    batch_normalize: bool,
) -> GruTrainBuffers {
    GruTrainBuffers {
        input_z: default_connected_train_buffers(inputs, output, batch_normalize),
        state_z: default_connected_train_buffers(output, output, batch_normalize),
        input_r: default_connected_train_buffers(inputs, output, batch_normalize),
        state_r: default_connected_train_buffers(output, output, batch_normalize),
        input_h: default_connected_train_buffers(inputs, output, batch_normalize),
        state_h: default_connected_train_buffers(output, output, batch_normalize),
    }
}

pub(super) fn default_convolutional_train_buffers(
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
) -> ConvolutionalTrainBuffers {
    let groups = groups.max(1);
    ConvolutionalTrainBuffers {
        bias_updates: vec![0.0; filters],
        batch_norm: batch_normalize.then(|| BatchNormTrainBuffers {
            scale_updates: vec![0.0; filters],
            mean_delta: vec![0.0; filters],
            variance_delta: vec![0.0; filters],
        }),
        weight_updates: vec![0.0; filters * (input_channels / groups) * size * size],
    }
}

pub(super) fn default_local_train_buffers(
    input_channels: usize,
    locations: usize,
    filters: usize,
    size: usize,
) -> LocalTrainBuffers {
    LocalTrainBuffers {
        bias_updates: vec![0.0; locations * filters],
        weight_updates: vec![0.0; locations * filters * input_channels * size * size],
    }
}
