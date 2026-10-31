use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Accumulate seeded GPU backward and return the network-input gradient.
    /// This opt-in diagnostic path may download the input gradient; ordinary
    /// training calls do not retain or transfer it.
    pub fn backward_accumulate_seeded_gpu_with_input_gradient(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<Option<Tensor>> {
        *self.captured_input_gradient.borrow_mut() = None;
        self.capture_input_gradient.set(true);
        let result = self.backward_accumulate_seeded_gpu(forward, layer_output_gradient_seeds);
        self.capture_input_gradient.set(false);
        result?;
        Ok(self.captured_input_gradient.borrow_mut().take())
    }

    /// Accumulates the layer gradients into the device train buffers.  With
    /// `loss_scale != 1` every seed is multiplied by it first (the matching
    /// division happens in `update_weights_gpu`).
    pub fn backward_accumulate_seeded_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        let loss_scale = self.network.net.loss_scale;
        if loss_scale == 1.0 || !(loss_scale > 0.0 && loss_scale.is_finite()) {
            return self.backward_accumulate_gpu_unscaled(forward, layer_output_gradient_seeds);
        }
        let scaled: Vec<Option<Tensor>> = layer_output_gradient_seeds
            .iter()
            .map(|seed| {
                seed.as_ref().map(|seed| Tensor {
                    shape: seed.shape,
                    data: seed.data.iter().map(|g| g * loss_scale).collect(),
                })
            })
            .collect();
        self.backward_accumulate_gpu_unscaled(forward, &scaled)
    }

    pub(super) fn backward_accumulate_gpu_unscaled(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        reject_gpu_stage1_cpu_layers(&self.network, "backward")?;
        if layer_output_gradient_seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "GPU backward seed count mismatch: expected {}, got {}",
                self.network.layers.len(),
                layer_output_gradient_seeds.len()
            )));
        }
        if self.requires_cpu_backward_fallback() || has_host_only_deform(&self.network) {
            return self
                .backward_accumulate_seeded_gpu_cpu_fallback(forward, layer_output_gradient_seeds);
        }
        let mut device_seeds = Vec::with_capacity(layer_output_gradient_seeds.len());
        for seed in layer_output_gradient_seeds {
            device_seeds.push(match seed {
                Some(seed) => Some(self.backend.upload_f32(&tensor_backend_shape(seed.shape), &seed.data)
                    .map_err(|err| backend_error("upload GPU backward seed", err))?),
                None => None,
            });
        }
        self.backward_accumulate_device_unscaled(forward, &device_seeds)
    }

    /// Accumulate gradients from seeds already resident on the device.
    /// A seed must have the shape of its corresponding layer output.
    pub fn backward_accumulate_device_seeded_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        seeds: &[Option<B::Tensor>],
    ) -> Result<()> {
        if seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!("GPU backward seed count mismatch: expected {}, got {}", self.network.layers.len(), seeds.len())));
        }
        if self.requires_cpu_backward_fallback() || has_host_only_deform(&self.network) {
            return Err(Error::InvalidConfig("device-seeded backward does not support host-only layers".into()));
        }
        for (index, seed) in seeds.iter().enumerate() {
            let Some(shape) = seed.as_ref().and_then(|seed| self.backend.tensor_shape(seed)) else {
                continue;
            };
            let expected = tensor_backend_shape(self.network.layers[index].output_shape);
            if shape != expected {
                return Err(Error::InvalidConfig(format!(
                    "GPU backward seed shape mismatch at layer {index}: got {shape:?}, expected {expected:?}"
                )));
            }
        }
        let scale = self.network.net.loss_scale;
        if scale == 1.0 || !(scale > 0.0 && scale.is_finite()) {
            return self.backward_accumulate_device_unscaled(forward, seeds);
        }
        let config = ElementwiseLaunchConfig::default();
        let mut scaled = Vec::with_capacity(seeds.len());
        for (index, seed) in seeds.iter().enumerate() {
            scaled.push(match seed {
                Some(seed) => {
                    let zero = self.backend.zeros_f32(&tensor_backend_shape(self.network.layers[index].output_shape))
                        .map_err(|err| backend_error("allocate scaled GPU seed", err))?;
                    Some(self.backend.saxpy_f32(scale, seed, &zero, config)
                        .map_err(|err| backend_error("scale GPU backward seed", err))?)
                }
                None => None,
            });
        }
        self.backward_accumulate_device_unscaled(forward, &scaled)
    }

    fn backward_accumulate_device_unscaled(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        device_seeds: &[Option<B::Tensor>],
    ) -> Result<()> {
        reject_gpu_stage1_cpu_layers(&self.network, "backward")?;
        let train_buffers_uninitialized = {
            let conv = self.device_train_buffers.borrow();
            let moe = self.device_moe_train_buffers.borrow();
            let eml = self.device_eml_train_buffers.borrow();
            let dcnv4 = self.device_dcnv4_train_buffers.borrow();
            !self.train_buffers_initialized || conv.len() != self.network.layers.len()
                || moe.len() != self.network.layers.len()
                || eml.len() != self.network.layers.len()
                || dcnv4.len() != self.network.layers.len()
                // Constructor vectors have the right length but empty slots.
                // The explicit flag also handles transformer-only networks
                // that legitimately never own convolution buffers.
                || self.cpu_fallback_train_buffers.borrow().len() != self.network.layers.len()
        };
        if train_buffers_uninitialized {
            self.init_train_buffers_gpu()?;
        }

        let config = ElementwiseLaunchConfig::default();
        let mut layer_output_gradients = device_seeds.to_vec();
        let mut foveation_hf_auxiliary_gradients: Vec<Option<Vec<f32>>> =
            vec![None; self.network.layers.len()];

        for layer in self.network.layers.iter().rev() {
            let Some(grad) = layer_output_gradients[layer.index].take() else {
                continue;
            };

            match (&layer.kind, self.params.get(layer.index)) {
                (LayerKind::RecursiveBlock(spec), Some(LayerParams::RecursiveBlock(_))) => {
                    self.backward_recursive_block_gpu(layer, spec, forward, grad, config, &mut layer_output_gradients)?;
                }
                (LayerKind::FoveationFilm(spec), Some(LayerParams::FoveationFilm(params))) => {
                    self.backward_foveation_film_gpu(
                        layer,
                        spec,
                        params,
                        &mut foveation_hf_auxiliary_gradients,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::HyperbolicFoveation(spec),
                    Some(LayerParams::HyperbolicFoveation(params)),
                ) => {
                    self.backward_hyperbolic_foveation_gpu(
                        layer,
                        spec,
                        params,
                        &foveation_hf_auxiliary_gradients,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Convolutional(spec),
                    Some(LayerParams::Convolutional(_) | LayerParams::None),
                ) => {
                    self.backward_convolutional_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::BatchNorm(_), Some(LayerParams::BatchNorm(_))) => {
                    self.backward_batch_norm_gpu(
                        layer,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::GraphConv(spec), Some(LayerParams::GraphConv(params))) => {
                    self.backward_graph_conv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), Some(LayerParams::None)) => {
                    self.backward_maxpool_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LocalAvgPool(spec), Some(LayerParams::None)) => {
                    self.backward_local_avgpool_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::AvgPool, Some(LayerParams::None)) => {
                    self.backward_avgpool_gpu(
                        layer,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crop(_), Some(LayerParams::None)) => {
                    self.backward_crop_gpu(
                        layer,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Route(spec), Some(LayerParams::None)) => {
                    self.backward_route_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LayaMarkerGather(spec), Some(LayerParams::None)) => {
                    self.backward_laya_marker_gather_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LayaActHead(_), Some(LayerParams::LayaActHead(_))) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: "laya_act_head GPU backward".to_owned(),
                    });
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::Shortcut(params))) => {
                    self.backward_weighted_shortcut_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::None)) => {
                    self.backward_shortcut_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Upsample(spec), Some(LayerParams::None)) => {
                    self.backward_upsample_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Region(spec), Some(LayerParams::None)) => {
                    self.backward_region_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Softmax { groups, spatial }, Some(LayerParams::None)) => {
                    self.backward_softmax_gpu(
                        layer,
                        groups,
                        *spatial,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelSlice(spec), Some(LayerParams::None)) => {
                    self.backward_channel_slice_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelShuffle(spec), Some(LayerParams::None)) => {
                    self.backward_channel_shuffle_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Reorg(spec), Some(LayerParams::None)) => {
                    self.backward_reorg_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dropout { .. }, Some(LayerParams::None)) => {
                    self.backward_dropout_gpu(
                        layer,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Sam(spec), Some(LayerParams::None)) => {
                    self.backward_sam_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ScaleChannels(spec), Some(LayerParams::None)) => {
                    self.backward_scale_channels_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Local(spec),
                    Some(LayerParams::Local(params)),
                ) if spec.activation.to_cuda_f32_kind().is_some() => {
                    self.backward_local_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Deconvolutional(spec),
                    Some(LayerParams::Convolutional(params)),
                ) if deconvolutional_spec_is_native(spec) => {
                    self.backward_deconvolutional_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Connected(spec),
                    Some(LayerParams::Connected(connected)),
                ) if spec.activation.to_cuda_f32_kind().is_some() => {
                    self.backward_connected_gpu(
                        layer,
                        spec,
                        connected,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::EmlConvolutional(spec),
                    Some(LayerParams::EmlConvolutional(params)),
                ) => {
                    self.backward_eml_convolutional_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Moe(spec), Some(LayerParams::Moe(params))) => {
                    self.backward_moe_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Gru(spec), Some(LayerParams::Gru(params))) => {
                    self.backward_gru_gpu(layer,spec,params,forward,grad,config,&mut layer_output_gradients)?;
                }
                (LayerKind::Rnn(spec), Some(LayerParams::Rnn(params))) => {
                    self.backward_rnn_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Lstm(spec), Some(LayerParams::Lstm(params))) => {
                    self.backward_lstm_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crnn(spec), Some(LayerParams::Crnn(params))) => {
                    self.backward_crnn_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::LayaScorer(spec),
                    Some(LayerParams::LayaScorer(weights)),
                ) => {
                    let current = forward.gpu_layer_outputs.get(layer.index - 1).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: -1,
                        },
                    )?;
                    let input_gradient = laya_scorer_gpu::backward_input(
                        &self.backend,
                        spec,
                        weights,
                        current,
                        &grad,
                        layer.input_shape,
                        layer.output_shape,
                        config,
                    )?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), Some(LayerParams::None))
                | (LayerKind::ObbYolo(_), Some(LayerParams::None))
                | (LayerKind::GaussianYolo(_), Some(LayerParams::None))
                | (LayerKind::Cost { .. }, Some(LayerParams::None))
                | (LayerKind::Logistic, Some(LayerParams::None))
                | (LayerKind::Segmentation(_), Some(LayerParams::None))
                | (LayerKind::OmdLoss(_), Some(LayerParams::None))
                | (LayerKind::Contrastive(_), Some(LayerParams::None))
                | (LayerKind::Regression(_), Some(LayerParams::None))
                | (LayerKind::JevCe(_), Some(LayerParams::None))
                | (LayerKind::LayaLoss(_), Some(LayerParams::None))
                | (LayerKind::LayaEncoder(_), Some(LayerParams::LayaEncoder(_)))
                | (LayerKind::LayaHead(_), Some(LayerParams::LayaHead(_)))
                // Modern YOLO head loss/backward is host-owned; its forward
                // activation now runs natively on the device.
                | (LayerKind::Yolox(_), Some(LayerParams::None))
                | (LayerKind::Ppyoloe(_), Some(LayerParams::None))
                | (LayerKind::Yolonas(_), Some(LayerParams::None))
                | (LayerKind::CenterNet(_), Some(LayerParams::None)) => {
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dcnv4(spec), Some(LayerParams::Dcnv4(params))) => {
                    self.backward_dcnv4_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DsConv(spec), Some(LayerParams::DsConv(params))) => {
                    self.backward_dsconv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DeformConv(spec), Some(LayerParams::DeformConv(params))) => {
                    self.backward_deform_conv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Wmhf(spec), Some(LayerParams::Wmhf(params))) => {
                    self.backward_wmhf_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Clifford(spec), Some(LayerParams::Clifford(params))) => {
                    self.backward_clifford_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Vit(spec), Some(LayerParams::Vit(weights))) => {
                    self.backward_vit_gpu(
                        layer,
                        &vit_geometry(layer, spec),
                        weights,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Swin(spec), Some(LayerParams::Swin(weights))) => {
                    self.backward_swin_gpu(
                        layer,
                        &crate::layers::swin::gpu::geometry(layer, spec),
                        weights,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::TuckerAttention(spec), Some(LayerParams::TuckerAttention(_))) => {
                    self.backward_tucker_attention_gpu(
                        layer, spec, forward, grad, config, &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MambaVision(spec), Some(LayerParams::MambaVision(_))) => {
                    self.backward_mambavision_gpu(
                        layer, spec, forward, grad, config, &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::KimiKda(spec), Some(LayerParams::KimiKda(_))) => {
                    self.backward_kimi_kda_gpu(
                        layer, spec, forward, grad, config, &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DptHead(spec), Some(LayerParams::DptHead(_))) => {
                    self.backward_dpt_head_gpu(
                        layer, spec, forward, grad, config, &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DepthAnything(spec), Some(LayerParams::DepthAnything(_))) => {
                    self.backward_depth_anything_gpu(
                        layer, spec, forward, grad, config, &mut layer_output_gradients,
                    )?;
                }
                _ => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{}_gpu_backward", layer.kind.name()),
                    });
                }
            }
        }

        self.update_gpu_batch_norm_running_stats_from_forward(forward)?;
        self.update_dcnv4_rolling_stats_gpu(forward)?;
        self.update_clifford_rolling_stats_gpu(forward)?;
        self.update_wmhf_rolling_stats_gpu(forward)?;
        if self.network.layers.iter().any(|layer| {
            matches!(&layer.kind, LayerKind::Rnn(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Lstm(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Crnn(spec) if spec.batch_normalize)
        }) {
            let input = self
                .download_tensor(&forward.input_gpu, self.network.net.input_shape())
                .map_err(|err| {
                    backend_error("download BN recurrent input for rolling stats", err)
                })?;
            update_recurrent_batch_norm_running_stats(
                &self.network,
                &mut self.params,
                &input,
                &forward.cpu_forward,
            )?;
        }
        Ok(())
    }

}
