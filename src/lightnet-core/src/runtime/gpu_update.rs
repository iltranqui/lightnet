use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(super) fn allocate_device_conv_train_buffer(
        &self,
        spec: &ConvolutionalSpec,
        input_channels: usize,
    ) -> Result<GpuConvolutionTrainBuffers<B::Tensor>> {
        let weight_updates = self
            .backend
            .zeros_f32(&[
                spec.filters,
                input_channels / spec.groups.max(1),
                spec.size,
                spec.size,
            ])
            .map_err(|err| backend_error("allocate GPU weight updates", err))?;
        let bias_updates = self
            .backend
            .zeros_f32(&[spec.filters])
            .map_err(|err| backend_error("allocate GPU bias updates", err))?;
        let scale_updates = spec
            .batch_normalize
            .then(|| {
                self.backend
                    .zeros_f32(&[spec.filters])
                    .map_err(|err| backend_error("allocate GPU scale updates", err))
            })
            .transpose()?;
        Ok(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
            pending: None,
        })
    }

    pub fn init_train_buffers_gpu(&mut self) -> Result<()> {
        self.warm_device_conv_params()?;
        let mut train_buffers = Vec::with_capacity(self.network.layers.len());
        for layer in &self.network.layers {
            match &layer.kind {
                LayerKind::Convolutional(spec) if spec.share_index.is_none() => {
                    let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
                    let weight_updates = self
                        .backend
                        .zeros_f32(&[spec.filters, grouped_channels, spec.size, spec.size])
                        .map_err(|err| backend_error("allocate GPU weight updates", err))?;
                    let bias_updates = self
                        .backend
                        .zeros_f32(&[spec.filters])
                        .map_err(|err| backend_error("allocate GPU bias updates", err))?;
                    let scale_updates = spec
                        .batch_normalize
                        .then(|| {
                            self.backend
                                .zeros_f32(&[spec.filters])
                                .map_err(|err| backend_error("allocate GPU scale updates", err))
                        })
                        .transpose()?;
                    train_buffers.push(Some(GpuConvolutionTrainBuffers {
                        weight_updates,
                        bias_updates,
                        scale_updates,
                        pending: None,
                    }));
                }
                _ => train_buffers.push(None),
            }
        }
        *self.device_train_buffers.get_mut() = train_buffers;

        let mut dcnv4_train_buffers = Vec::with_capacity(self.network.layers.len());
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let buffers = match (&layer.kind, params) {
                (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params))
                    if !dcnv4_gpu::needs_host(spec) =>
                {
                    let desc = dcnv4_gpu::descriptor(layer, spec, params)?;
                    let bias_shape = [spec.filters];
                    let offset_weight_shape = [
                        desc.geometry.offset_dim(),
                        desc.geometry.channels,
                        spec.size,
                        spec.size,
                    ];
                    let offset_bias_shape = [desc.geometry.offset_dim()];
                    let bias_updates = self
                        .backend
                        .zeros_f32(&bias_shape)
                        .map_err(|err| backend_error("allocate DCNv4 GPU bias updates", err))?;
                    let offset_weight_updates = self
                        .backend
                        .zeros_f32(&offset_weight_shape)
                        .map_err(|err| {
                            backend_error("allocate DCNv4 GPU offset weight updates", err)
                        })?;
                    let offset_bias_updates =
                        self.backend.zeros_f32(&offset_bias_shape).map_err(|err| {
                            backend_error("allocate DCNv4 GPU offset bias updates", err)
                        })?;
                    let scale_updates = params
                        .batch_norm
                        .as_ref()
                        .map(|_| {
                            self.backend.zeros_f32(&bias_shape).map_err(|err| {
                                backend_error("allocate DCNv4 GPU scale updates", err)
                            })
                        })
                        .transpose()?;
                    let momentum_bias = self
                        .backend
                        .zeros_f32(&bias_shape)
                        .map_err(|err| backend_error("allocate DCNv4 GPU bias momentum", err))?;
                    let momentum_offset_weights = self
                        .backend
                        .zeros_f32(&offset_weight_shape)
                        .map_err(|err| {
                            backend_error("allocate DCNv4 GPU offset weight momentum", err)
                        })?;
                    let momentum_offset_biases =
                        self.backend.zeros_f32(&offset_bias_shape).map_err(|err| {
                            backend_error("allocate DCNv4 GPU offset bias momentum", err)
                        })?;
                    let momentum_scale = params
                        .batch_norm
                        .as_ref()
                        .map(|_| {
                            self.backend.zeros_f32(&bias_shape).map_err(|err| {
                                backend_error("allocate DCNv4 GPU scale momentum", err)
                            })
                        })
                        .transpose()?;
                    Some(GpuDcnv4TrainBuffers {
                        bias_updates,
                        offset_weight_updates,
                        offset_bias_updates,
                        scale_updates,
                        momentum_bias,
                        momentum_offset_weights,
                        momentum_offset_biases,
                        momentum_scale,
                        dirty: false,
                    })
                }
                (LayerKind::Dcnv4(_), LayerParams::Dcnv4(_)) => None,
                (LayerKind::Dcnv4(_), _) => {
                    return Err(Error::InvalidConfig(format!(
                        "DCNv4 layer #{} has inconsistent parameters while allocating GPU train buffers",
                        layer.index
                    )));
                }
                _ => None,
            };
            dcnv4_train_buffers.push(buffers);
        }
        *self.device_dcnv4_train_buffers.get_mut() = dcnv4_train_buffers;

        let mut moe_train_buffers = Vec::with_capacity(self.network.layers.len());
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let buffers = match (&layer.kind, params) {
                (LayerKind::Moe(spec), LayerParams::Moe(moe)) => {
                    if moe.experts.len() != spec.experts
                        || moe.experts.iter().any(|expert| {
                            expert.batch_norm.is_some() != spec.expert.batch_normalize
                        })
                    {
                        return Err(Error::InvalidConfig(format!(
                            "MoE expert parameter mismatch while allocating GPU train buffers at layer #{}",
                            layer.index
                        )));
                    }
                    let router_weight_updates = self
                        .backend
                        .zeros_f32(&[spec.experts, layer.input_shape.channels])
                        .map_err(|err| backend_error("allocate MoE router weight updates", err))?;
                    let router_bias_updates = self
                        .backend
                        .zeros_f32(&[spec.experts])
                        .map_err(|err| backend_error("allocate MoE router bias updates", err))?;
                    let grouped_channels = layer.input_shape.channels / spec.expert.groups.max(1);
                    let mut expert_weight_updates = Vec::with_capacity(spec.experts);
                    let mut expert_bias_updates = Vec::with_capacity(spec.experts);
                    let mut expert_scale_updates = Vec::with_capacity(spec.experts);
                    for expert in &moe.experts {
                        expert_weight_updates.push(
                            self.backend
                                .zeros_f32(&[
                                    spec.expert.filters,
                                    grouped_channels,
                                    spec.expert.size,
                                    spec.expert.size,
                                ])
                                .map_err(|err| {
                                    backend_error("allocate MoE expert weight updates", err)
                                })?,
                        );
                        expert_bias_updates.push(
                            self.backend
                                .zeros_f32(&[spec.expert.filters])
                                .map_err(|err| {
                                    backend_error("allocate MoE expert bias updates", err)
                                })?,
                        );
                        expert_scale_updates.push(
                            expert
                                .batch_norm
                                .as_ref()
                                .map(|_| {
                                    self.backend
                                        .zeros_f32(&[spec.expert.filters])
                                        .map_err(|err| {
                                            backend_error("allocate MoE expert scale updates", err)
                                        })
                                })
                                .transpose()?,
                        );
                    }
                    Some(GpuMoeTrainBuffers {
                        router_weight_updates,
                        router_bias_updates,
                        expert_weight_updates,
                        expert_bias_updates,
                        expert_scale_updates,
                        dirty: false,
                    })
                }
                (LayerKind::Moe(_), _) => {
                    return Err(Error::InvalidConfig(format!(
                        "MoE layer #{} has inconsistent parameters while allocating GPU train buffers",
                        layer.index
                    )));
                }
                _ => None,
            };
            moe_train_buffers.push(buffers);
        }
        *self.device_moe_train_buffers.get_mut() = moe_train_buffers;

        let mut eml_train_buffers = Vec::with_capacity(self.network.layers.len());
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let buffers = match (&layer.kind, params) {
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(eml)) => {
                    if eml.input.batch_norm.is_some() != spec.branch.batch_normalize
                        || eml.self_params.batch_norm.is_some() != spec.branch.batch_normalize
                    {
                        return Err(Error::InvalidConfig(format!(
                            "EML branch parameter mismatch while allocating GPU train buffers at layer #{}",
                            layer.index
                        )));
                    }
                    Some(GpuEmlTrainBuffers {
                        input: self.allocate_device_conv_train_buffer(
                            &spec.branch,
                            layer.input_shape.channels,
                        )?,
                        self_params: self.allocate_device_conv_train_buffer(
                            &spec.branch,
                            layer.input_shape.channels,
                        )?,
                    })
                }
                (LayerKind::EmlConvolutional(_), _) => {
                    return Err(Error::InvalidConfig(format!(
                        "EML layer #{} has inconsistent parameters while allocating GPU train buffers",
                        layer.index
                    )));
                }
                _ => None,
            };
            eml_train_buffers.push(buffers);
        }
        *self.device_eml_train_buffers.get_mut() = eml_train_buffers;
        Ok(())
    }

    pub fn zero_train_buffers_gpu(&mut self) -> Result<()> {
        *self.cpu_fallback_train_buffers.get_mut() = self
            .network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect();
        self.init_train_buffers_gpu()
    }

    /// One optimizer step over device convolutions, EML SGD, and MoE SGD/Adam,
    /// including BN scales. EML Adam, HyperbolicFoveation, and FoveationFilm
    /// parameters are updated through their host fallback buffers.
    /// The optimizer follows `[net]`: SGD with
    /// momentum by default, Adam with `adam=1` (`B1`/`B2`/`eps`, coupled L2
    /// decay like C++, or AdamW with `decoupled_decay=1`; `lr`/`decay`/`batch`
    /// are the same arguments).  `loss_scale` makes the step independent of the
    /// gradient scale applied in `backward_accumulate_seeded_gpu`: SGD divides
    /// lr and multiplies decay, Adam scales eps and the coupled decay.  Updates are reset and weights fixed for
    /// NaN/Inf first (C++ `reset_nan_and_inf` / `fix_nan_and_inf`).
    pub fn update_weights_gpu(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    ) -> Result<()> {
        self.update_weights_gpu_impl(lr, momentum, decay, batch, true)
    }

    /// Like [`Self::update_weights_gpu`] but device convolution params, EML SGD
    /// branches, and supported MoE expert params (SGD/Adam, including BN) can
    /// stay on-device. MoE router parameters synchronize immediately because
    /// the current router forward consumes host parameters; EML branch and MoE
    /// expert params remain stale until [`Self::sync_host_weights_from_device`]
    /// runs (call it before reading or saving `params`). Falls back to eager
    /// sync when CPU backward reads host weights.
    pub fn update_weights_gpu_deferred(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
    ) -> Result<()> {
        let eager = self.requires_cpu_backward_fallback();
        self.update_weights_gpu_impl(lr, momentum, decay, batch, eager)
    }

    /// Refreshes host convolution, EML branch, and MoE expert weights/biases/BN
    /// stats after deferred device updates.
    pub fn sync_host_weights_from_device(&mut self) -> Result<()> {
        if !self.host_weights_stale {
            return Ok(());
        }
        for layer_index in 0..self.network.layers.len() {
            match &self.network.layers[layer_index].kind {
                LayerKind::Convolutional(spec) => {
                    if spec.share_index.is_some() {
                        continue;
                    }
                    let cached = self.device_conv_params_at(layer_index, 0);
                    if let Some(cached) = cached {
                        self.sync_convolution_params_from_device(layer_index, &cached)?;
                    }
                }
                LayerKind::Moe(_) => {
                    let expert_count = match self.params.get(layer_index) {
                        Some(LayerParams::Moe(params)) => params.experts.len(),
                        _ => continue,
                    };
                    for expert_index in 0..expert_count {
                        if let Some(cached) = self.device_conv_params_at(layer_index, expert_index)
                        {
                            self.sync_moe_expert_params_from_device(
                                layer_index,
                                expert_index,
                                &cached,
                            )?;
                        }
                    }
                }
                LayerKind::EmlConvolutional(_) => {
                    for branch_slot in 0..2 {
                        if let Some(cached) = self.device_conv_params_at(layer_index, branch_slot) {
                            self.sync_eml_branch_params_from_device(
                                layer_index,
                                branch_slot,
                                &cached,
                            )?;
                        }
                    }
                }
                _ => continue,
            }
        }
        self.host_weights_stale = false;
        Ok(())
    }

    pub(super) fn update_weights_gpu_muon(&mut self, lr: f32, decay: f32, batch: usize) -> Result<()> {
        let net = self.network.net.clone();
        if !net.muon || net.adam || !net.decoupled_decay {
            return Err(Error::InvalidConfig(
                "GPU Muon requires [net] muon=1, adam=0, and AdamW fallback".to_owned(),
            ));
        }
        if !lr.is_finite() || lr < 0.0 || !decay.is_finite() || decay < 0.0 {
            return Err(Error::InvalidConfig(
                "GPU Muon learning rate and decay must be finite and non-negative".to_owned(),
            ));
        }
        if !net.muon_momentum.is_finite() || !(0.0..1.0).contains(&net.muon_momentum) {
            return Err(Error::InvalidConfig(
                "GPU Muon momentum must be finite and in [0, 1)".to_owned(),
            ));
        }
        if let Some((index, _)) = self.params.iter().enumerate().find(|(index, params)| {
            let kind = &self.network.layers[*index].kind;
            match (kind, *params) {
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(_)) => {
                    spec.batch_normalize
                }
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(_)) => {
                    spec.branch.batch_normalize
                }
                (LayerKind::Moe(spec), LayerParams::Moe(_)) => spec.expert.batch_normalize,
                (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(_)) => spec.batch_normalize,
                (LayerKind::DeformConv(spec), LayerParams::DeformConv(_)) => spec.batch_normalize,
                _ => !matches!(
                    (kind, *params),
                    (_, LayerParams::None)
                        | (LayerKind::BatchNorm(_), LayerParams::BatchNorm(_))
                        | (LayerKind::Connected(_), LayerParams::Connected(_))
                        | (
                            LayerKind::HyperbolicFoveation(_),
                            LayerParams::HyperbolicFoveation(_)
                        )
                        | (LayerKind::FoveationFilm(_), LayerParams::FoveationFilm(_))
                        | (
                            LayerKind::EmlConvolutional(_),
                            LayerParams::EmlConvolutional(_)
                        )
                        | (LayerKind::Moe(_), LayerParams::Moe(_))
                        | (LayerKind::Local(_), LayerParams::Local(_))
                        | (LayerKind::Rnn(_), LayerParams::Rnn(_))
                        | (LayerKind::Lstm(_), LayerParams::Lstm(_))
                        | (LayerKind::Crnn(_), LayerParams::Crnn(_))
                        | (LayerKind::Vit(_), LayerParams::Vit(_))
                        | (LayerKind::Swin(_), LayerParams::Swin(_))
                        | (
                            LayerKind::TuckerAttention(_),
                            LayerParams::TuckerAttention(_)
                        )
                        | (LayerKind::MambaVision(_), LayerParams::MambaVision(_))
                        | (LayerKind::KimiKda(_), LayerParams::KimiKda(_))
                        | (LayerKind::DptHead(_), LayerParams::DptHead(_))
                        | (LayerKind::DepthAnything(_), LayerParams::DepthAnything(_))
                        | (LayerKind::RecursiveBlock(_), LayerParams::RecursiveBlock(_))
                        | (LayerKind::GraphConv(_), LayerParams::GraphConv(_))
                        | (LayerKind::Dcnv4(_), LayerParams::Dcnv4(_))
                        | (LayerKind::DeformConv(_), LayerParams::DeformConv(_))
                        | (LayerKind::DsConv(_), LayerParams::DsConv(_))
                        | (LayerKind::Deconvolutional(_), LayerParams::Convolutional(_))
                ),
            }
        }) {
            return Err(Error::UnsupportedLayer {
                layer_index: index,
                name: "GPU Muon supports standard convolution, EML, MoE, DCNv4, deform_conv, and single-template DSConv matrices without BN via host-staged gradients, GraphConv matrices including BN through the existing CPU fallback, plus host-gradient local/RNN/LSTM/CRNN/Kimi/MambaVision/Tucker/Swin/ViT/deconvolutional/connected matrices and standalone batch-norm parameters; other parameter families remain explicitly unsupported".to_owned(),
            });
        }
        let step = self
            .adam_step
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConfig("GPU Muon step counter overflowed".to_owned()))?;
        let loss_scale = if net.loss_scale > 0.0 && net.loss_scale.is_finite() {
            net.loss_scale
        } else {
            1.0
        };
        let adam = AdamParams {
            lr,
            beta1: net.adam_b1,
            beta2: net.adam_b2,
            eps: net.adam_eps * loss_scale,
            decay,
            batch,
            step,
            decoupled_decay: true,
        };
        let layer_count = self.network.layers.len();
        if self.muon_state.len() != layer_count {
            self.muon_state = (0..layer_count).map(|_| None).collect();
            *self.device_muon_momentum.borrow_mut() = vec![None; layer_count];
        }
        {
            let host_state = self.host_adam_state.get_mut();
            if host_state.len() != layer_count {
                *host_state = (0..layer_count).map(|_| None).collect();
            }
        }
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = net.burn_in;
        let dcnv4_muon_layers: Vec<usize> = self
            .network
            .layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                (layer.train.updates_at(iteration_num, burn_in)
                    && matches!(
                        (&layer.kind, self.params.get(index)),
                        (LayerKind::Dcnv4(spec), Some(LayerParams::Dcnv4(_)))
                            if !spec.batch_normalize
                    ))
                .then_some(index)
            })
            .collect();
        for index in dcnv4_muon_layers {
            self.flush_device_dcnv4_gradients_to_host(Some(index))?;
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        for index in 0..layer_count {
            let layer = self.network.layers[index].clone();
            if !layer.train.updates_at(iteration_num, burn_in)
                || matches!(self.params[index], LayerParams::None)
                || matches!(layer.kind, LayerKind::DepthAnything(_))
            {
                continue;
            }
            let options = layer.train;
            if let LayerKind::Convolutional(spec) = &layer.kind
                && !spec.batch_normalize
            {
                let device_gradients = {
                    let device_buffers = self.device_train_buffers.borrow();
                    let train = device_buffers
                        .get(index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing device convolution gradients for Muon layer #{index}"
                            ))
                        })?;
                    if train.pending.is_some() || train.scale_updates.is_some() {
                        return Err(Error::InvalidConfig(format!(
                            "GPU Muon cannot stage deferred or batch-normalization gradients for convolution layer #{index}"
                        )));
                    }
                    (
                        self.backend
                            .download_f32(&train.weight_updates)
                            .map_err(|err| {
                                backend_error("download convolution Muon gradients", err)
                            })?,
                        self.backend
                            .download_f32(&train.bias_updates)
                            .map_err(|err| {
                                backend_error("download convolution bias gradients", err)
                            })?,
                    )
                };
                let LayerTrainBuffers::Convolutional(train) = &mut buffers[index] else {
                    return Err(Error::InvalidConfig(format!(
                        "convolution Muon layer #{index} has mismatched host train buffers"
                    )));
                };
                train.weight_updates = device_gradients.0;
                train.bias_updates = device_gradients.1;
            }
            if let LayerKind::EmlConvolutional(spec) = &layer.kind
                && !spec.branch.batch_normalize
            {
                let device_gradients = {
                    let device_buffers = self.device_eml_train_buffers.borrow();
                    let train = device_buffers
                        .get(index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing device EML gradients for Muon layer #{index}"
                            ))
                        })?;
                    for branch in [&train.input, &train.self_params] {
                        if branch.pending.is_some() || branch.scale_updates.is_some() {
                            return Err(Error::InvalidConfig(format!(
                                "GPU Muon cannot stage deferred or batch-normalization gradients for EML layer #{index}"
                            )));
                        }
                    }
                    (
                        self.backend
                            .download_f32(&train.input.weight_updates)
                            .map_err(|err| {
                                backend_error("download EML input Muon gradients", err)
                            })?,
                        self.backend
                            .download_f32(&train.input.bias_updates)
                            .map_err(|err| {
                                backend_error("download EML input bias gradients", err)
                            })?,
                        self.backend
                            .download_f32(&train.self_params.weight_updates)
                            .map_err(|err| {
                                backend_error("download EML self Muon gradients", err)
                            })?,
                        self.backend
                            .download_f32(&train.self_params.bias_updates)
                            .map_err(|err| {
                                backend_error("download EML self bias gradients", err)
                            })?,
                    )
                };
                let LayerTrainBuffers::EmlConvolutional(train) = &mut buffers[index] else {
                    return Err(Error::InvalidConfig(format!(
                        "EML Muon layer #{index} has mismatched host train buffers"
                    )));
                };
                train.input.weight_updates = device_gradients.0;
                train.input.bias_updates = device_gradients.1;
                train.self_params.weight_updates = device_gradients.2;
                train.self_params.bias_updates = device_gradients.3;
            }
            if let LayerKind::Moe(spec) = &layer.kind
                && !spec.expert.batch_normalize
            {
                let device_gradients = {
                    let device_buffers = self.device_moe_train_buffers.borrow();
                    let train = device_buffers
                        .get(index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing device MoE gradients for Muon layer #{index}"
                            ))
                        })?;
                    if train.expert_weight_updates.len() != spec.experts
                        || train.expert_bias_updates.len() != spec.experts
                        || train.expert_scale_updates.len() != spec.experts
                        || train.expert_scale_updates.iter().any(Option::is_some)
                    {
                        return Err(Error::InvalidConfig(format!(
                            "GPU Muon MoE gradient layout mismatch at layer #{index}"
                        )));
                    }
                    let experts = train
                        .expert_weight_updates
                        .iter()
                        .zip(&train.expert_bias_updates)
                        .map(|(weights, biases)| {
                            Ok((
                                self.backend.download_f32(weights).map_err(|err| {
                                    backend_error("download MoE expert Muon gradients", err)
                                })?,
                                self.backend.download_f32(biases).map_err(|err| {
                                    backend_error("download MoE expert bias gradients", err)
                                })?,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    (
                        self.backend
                            .download_f32(&train.router_weight_updates)
                            .map_err(|err| {
                                backend_error("download MoE router Muon gradients", err)
                            })?,
                        self.backend
                            .download_f32(&train.router_bias_updates)
                            .map_err(|err| {
                                backend_error("download MoE router bias gradients", err)
                            })?,
                        experts,
                    )
                };
                let LayerTrainBuffers::Moe(train) = &mut buffers[index] else {
                    return Err(Error::InvalidConfig(format!(
                        "MoE Muon layer #{index} has mismatched host train buffers"
                    )));
                };
                train.router_weight_updates = device_gradients.0;
                train.router_bias_updates = device_gradients.1;
                if train.experts.len() != device_gradients.2.len() {
                    return Err(Error::InvalidConfig(format!(
                        "MoE Muon host expert count mismatch at layer #{index}"
                    )));
                }
                for (expert, (weights, biases)) in train.experts.iter_mut().zip(device_gradients.2)
                {
                    expert.weight_updates = weights;
                    expert.bias_updates = biases;
                }
            }
            prepare_muon_layer_gradients(
                &layer.kind,
                options,
                &mut self.params[index],
                &mut buffers[index],
            )?;
            let layer_adam = AdamParams {
                lr: lr * options.learning_rate_scale,
                ..adam
            };
            if let (
                LayerKind::Connected(spec),
                LayerParams::Connected(params),
                LayerTrainBuffers::Connected(train),
            ) = (&layer.kind, &mut self.params[index], &mut buffers[index])
            {
                let rows = spec.output;
                let cols = layer.input_shape.element_count();
                let shape = [rows, cols];
                let state = self.muon_state[index].get_or_insert_with(MuonLayerState::default);
                if state.matrices.len() != 1
                    || state.matrices[0].rows != rows
                    || state.matrices[0].cols != cols
                {
                    state.matrices = vec![MuonMatrixState::zeroed(rows, cols)?];
                    let mut device_state = self.device_muon_momentum.borrow_mut();
                    if device_state.len() != layer_count {
                        *device_state = vec![None; layer_count];
                    }
                    device_state[index] = None;
                }
                let prior_momentum = state.matrices[0].momentum.clone();
                if train
                    .weight_updates
                    .iter()
                    .chain(&prior_momentum)
                    .chain(&params.weights)
                    .any(|value| !value.is_finite())
                {
                    return Err(Error::InvalidConfig(
                        "GPU Muon weights, updates, and momentum must be finite".to_owned(),
                    ));
                }
                let momentum_tensor = {
                    let mut device_state = self.device_muon_momentum.borrow_mut();
                    if device_state.len() != layer_count {
                        *device_state = vec![None; layer_count];
                    }
                    if device_state[index].is_none() {
                        device_state[index] = Some(
                            self.backend
                                .upload_f32(&shape, &prior_momentum)
                                .map_err(|err| backend_error("GPU Muon momentum upload", err))?,
                        );
                    }
                    device_state[index]
                        .as_ref()
                        .expect("initialized GPU Muon momentum")
                        .clone()
                };
                let updates = self
                    .backend
                    .upload_f32(&shape, &train.weight_updates)
                    .map_err(|err| backend_error("GPU Muon update upload", err))?;
                let (next_momentum, lookahead) = self
                    .backend
                    .muon_momentum_lookahead_f32(
                        &updates,
                        &momentum_tensor,
                        &shape,
                        batch,
                        loss_scale,
                        net.muon_momentum,
                    )
                    .map_err(|err| backend_error("GPU Muon momentum/lookahead", err))?;
                if self
                    .backend
                    .any_non_finite_f32(&next_momentum)
                    .map_err(|err| backend_error("GPU Muon momentum finiteness check", err))?
                    || self
                        .backend
                        .any_non_finite_f32(&lookahead)
                        .map_err(|err| backend_error("GPU Muon lookahead finiteness check", err))?
                {
                    return Err(Error::InvalidConfig(
                        "GPU Muon momentum update became non-finite".to_owned(),
                    ));
                }
                let orthogonal = self
                    .backend
                    .muon_orthogonalize_f32(&lookahead, &shape, rows, cols)
                    .map_err(|err| backend_error("GPU Muon Newton-Schulz", err))?;
                if self
                    .backend
                    .any_non_finite_f32(&orthogonal)
                    .map_err(|err| backend_error("GPU Muon direction finiteness check", err))?
                {
                    return Err(Error::InvalidConfig(
                        "GPU Muon Newton-Schulz direction became non-finite".to_owned(),
                    ));
                }
                let weights = self
                    .backend
                    .upload_f32(&shape, &params.weights)
                    .map_err(|err| backend_error("GPU Muon weights upload", err))?;
                let updated_weights = self
                    .backend
                    .muon_apply_update_f32(
                        &weights,
                        &orthogonal,
                        &shape,
                        layer_adam.lr,
                        layer_adam.decay,
                        crate::muon::MUON_SCALE * (rows.max(cols) as f32).sqrt(),
                    )
                    .map_err(|err| backend_error("GPU Muon parameter update", err))?;
                let next_weights = self
                    .backend
                    .download_f32(&updated_weights)
                    .map_err(|err| backend_error("GPU Muon weights download", err))?;
                let next_momentum_host = self
                    .backend
                    .download_f32(&next_momentum)
                    .map_err(|err| backend_error("GPU Muon momentum mirror download", err))?;
                if next_weights
                    .iter()
                    .chain(&next_momentum_host)
                    .any(|value| !value.is_finite())
                {
                    return Err(Error::InvalidConfig(
                        "GPU Muon parameter update became non-finite".to_owned(),
                    ));
                }
                params.weights = next_weights;
                state.matrices[0].momentum = next_momentum_host;
                train.weight_updates.fill(0.0);
                let host_adam = AdamParams {
                    decay: 0.0,
                    ..layer_adam
                };
                let host_state = self.host_adam_state.get_mut();
                let adam_state = host_state[index].get_or_insert_with(HostAdamLayerState::default);
                apply_host_adam_layer(
                    &layer.kind,
                    layer.input_shape,
                    &mut self.params[index],
                    &mut buffers[index],
                    adam_state,
                    &host_adam,
                )?;
                self.device_muon_momentum.borrow_mut()[index] = Some(next_momentum);
                continue;
            }
            let staged = {
                let state = self.muon_state[index].get_or_insert_with(MuonLayerState::default);
                let backend = &self.backend;
                stage_layer_with_orthogonalizer(
                    &layer.kind,
                    layer.input_shape,
                    layer.output_shape,
                    &mut self.params[index],
                    &mut buffers[index],
                    state,
                    layer_adam.lr,
                    batch,
                    loss_scale,
                    net.muon_momentum,
                    |lookahead, rows, cols| {
                        let shape = [rows, cols];
                        let input = backend
                            .upload_f32(&shape, lookahead)
                            .map_err(|err| backend_error("GPU Muon lookahead upload", err))?;
                        let orthogonal = backend
                            .muon_orthogonalize_f32(&input, &shape, rows, cols)
                            .map_err(|err| backend_error("GPU Muon Newton-Schulz", err))?;
                        backend
                            .download_f32(&orthogonal)
                            .map_err(|err| backend_error("GPU Muon direction download", err))
                    },
                )?
            };
            let host_state = self.host_adam_state.get_mut();
            let adam_state = host_state[index].get_or_insert_with(HostAdamLayerState::default);
            apply_host_adam_layer(
                &layer.kind,
                layer.input_shape,
                &mut self.params[index],
                &mut buffers[index],
                adam_state,
                &layer_adam,
            )?;
            let muon_state = self.muon_state[index].get_or_insert_with(MuonLayerState::default);
            apply_staged_layer(
                &layer.kind,
                layer.input_shape,
                layer.output_shape,
                &mut self.params[index],
                &mut buffers[index],
                muon_state,
                staged,
            )?;
            if let (LayerKind::Moe(spec), LayerParams::Moe(params)) =
                (&layer.kind, &self.params[index])
                && !spec.expert.batch_normalize
            {
                for (slot, expert) in params.experts.iter().enumerate() {
                    let refreshed = self.cached_convolution_upload(
                        &spec.expert,
                        layer.input_shape.channels,
                        expert,
                        None,
                    )?;
                    self.set_device_conv_params(index, slot, refreshed)?;
                }
                let mut device_buffers = self.device_moe_train_buffers.borrow_mut();
                let train = device_buffers[index].as_mut().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing device MoE buffers after Muon layer #{index} update"
                    ))
                })?;
                train.router_weight_updates = self
                    .backend
                    .zeros_f32(&[spec.experts, layer.input_shape.channels])
                    .map_err(|err| backend_error("zero MoE router Muon gradients", err))?;
                train.router_bias_updates = self
                    .backend
                    .zeros_f32(&[spec.experts])
                    .map_err(|err| backend_error("zero MoE router bias gradients", err))?;
                train.expert_weight_updates = (0..spec.experts)
                    .map(|_| {
                        self.backend
                            .zeros_f32(&[
                                spec.expert.filters,
                                layer.input_shape.channels / spec.expert.groups.max(1),
                                spec.expert.size,
                                spec.expert.size,
                            ])
                            .map_err(|err| backend_error("zero MoE expert Muon gradients", err))
                    })
                    .collect::<Result<Vec<_>>>()?;
                train.expert_bias_updates = (0..spec.experts)
                    .map(|_| {
                        self.backend
                            .zeros_f32(&[spec.expert.filters])
                            .map_err(|err| backend_error("zero MoE expert bias gradients", err))
                    })
                    .collect::<Result<Vec<_>>>()?;
                train.dirty = false;
            }
            if let (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(params)) =
                (&layer.kind, &self.params[index])
                && !spec.branch.batch_normalize
            {
                for (slot, branch) in [&params.input, &params.self_params].into_iter().enumerate() {
                    let refreshed = self.cached_convolution_upload(
                        &spec.branch,
                        layer.input_shape.channels,
                        branch,
                        None,
                    )?;
                    self.set_device_conv_params(index, slot, refreshed)?;
                }
                let mut device_buffers = self.device_eml_train_buffers.borrow_mut();
                let train = device_buffers[index].as_mut().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing device EML buffers after Muon layer #{index} update"
                    ))
                })?;
                for branch in [&mut train.input, &mut train.self_params] {
                    branch.weight_updates = self
                        .backend
                        .zeros_f32(&[
                            spec.branch.filters,
                            layer.input_shape.channels / spec.branch.groups.max(1),
                            spec.branch.size,
                            spec.branch.size,
                        ])
                        .map_err(|err| backend_error("zero EML Muon gradients", err))?;
                    branch.bias_updates = self
                        .backend
                        .zeros_f32(&[spec.branch.filters])
                        .map_err(|err| backend_error("zero EML Muon bias gradients", err))?;
                    branch.pending = None;
                }
            }
            if let (LayerKind::Convolutional(spec), LayerParams::Convolutional(params)) =
                (&layer.kind, &self.params[index])
                && !spec.batch_normalize
            {
                let refreshed =
                    self.cached_convolution_upload(spec, layer.input_shape.channels, params, None)?;
                self.set_device_conv_params(index, 0, refreshed)?;
                let mut device_buffers = self.device_train_buffers.borrow_mut();
                let train = device_buffers[index].as_mut().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing device convolution buffers after Muon layer #{index} update"
                    ))
                })?;
                train.weight_updates = self
                    .backend
                    .zeros_f32(&[
                        spec.filters,
                        layer.input_shape.channels / spec.groups.max(1),
                        spec.size,
                        spec.size,
                    ])
                    .map_err(|err| backend_error("zero convolution Muon gradients", err))?;
                train.bias_updates = self
                    .backend
                    .zeros_f32(&[spec.filters])
                    .map_err(|err| backend_error("zero convolution Muon bias gradients", err))?;
                train.pending = None;
            }
        }
        self.adam_step = step;
        self.host_weights_stale = false;
        Ok(())
    }

    pub(super) fn update_weights_gpu_impl(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        sync_host_weights: bool,
    ) -> Result<()> {
        reject_gpu_stage1_cpu_layers(&self.network, "optimizer updates")?;
        if self.network.net.muon {
            if !sync_host_weights {
                return Err(Error::InvalidConfig(
                    "GPU Muon does not support deferred host synchronization".to_owned(),
                ));
            }
            return self.update_weights_gpu_muon(lr, decay, batch);
        }
        let config = ElementwiseLaunchConfig::default();
        self.defer_host_sync = !sync_host_weights;
        let net = self.network.net.clone();
        let loss_scale = if net.loss_scale > 0.0 && net.loss_scale.is_finite() {
            net.loss_scale
        } else {
            1.0
        };
        let step = if net.adam {
            if self.params.iter().any(|params| {
                !matches!(
                    params,
                    LayerParams::None
                        | LayerParams::BatchNorm(_)
                        | LayerParams::Local(_)
                        | LayerParams::Shortcut(_)
                        | LayerParams::Convolutional(_)
                        | LayerParams::Connected(_)
                        | LayerParams::Moe(_)
                        | LayerParams::Gru(_)
                        | LayerParams::Vit(_)
                        | LayerParams::Swin(_)
                        | LayerParams::Clifford(_)
                        | LayerParams::TuckerAttention(_)
                        | LayerParams::MambaVision(_)
                        | LayerParams::KimiKda(_)
                        | LayerParams::DptHead(_)
                        | LayerParams::DepthAnything(_)
                        | LayerParams::RecursiveBlock(_)
                        | LayerParams::Rnn(_)
                        | LayerParams::Lstm(_)
                        | LayerParams::Crnn(_)
                        | LayerParams::HyperbolicFoveation(_)
                        | LayerParams::FoveationFilm(_)
                        | LayerParams::EmlConvolutional(_)
                        | LayerParams::Dcnv4(_)
                        | LayerParams::DeformConv(_)
                        | LayerParams::DsConv(_)
                        | LayerParams::GraphConv(_)
                        | LayerParams::Wmhf(_)
                )
            }) {
                return Err(Error::InvalidConfig(
                    "adam=1 on the GPU path supports convolutional, connected, local, shortcut, deform_conv, DSConv, DCNv4, MoE, standalone batchnorm, ViT, Swin, Tucker attention, MambaVision, Kimi KDA, GraphConv, GRU/RNN/LSTM/CRNN, and host-updated HF/FiLM/WMHF layers"
                        .to_string(),
                ));
            }
            self.adam_step += 1;
            GpuOptimizerStep::Adam(AdamParams {
                // Adam is invariant to the gradient scale, so lr stays as is
                // (C++ divides it by loss_scale anyway, which shrinks the
                // step); eps and the coupled decay scale with the gradients.
                lr,
                beta1: net.adam_b1,
                beta2: net.adam_b2,
                eps: net.adam_eps * loss_scale,
                decay: if net.decoupled_decay {
                    decay
                } else {
                    decay * loss_scale
                },
                batch,
                step: self.adam_step,
                decoupled_decay: net.decoupled_decay,
            })
        } else {
            GpuOptimizerStep::Sgd {
                lr: lr / loss_scale,
                momentum,
                decay: decay * loss_scale,
                batch,
            }
        };
        if self.adam_state.len() != self.network.layers.len() {
            self.adam_state = (0..self.network.layers.len()).map(|_| None).collect();
        }
        if self.eml_adam_state.len() != self.network.layers.len() {
            self.eml_adam_state = (0..self.network.layers.len()).map(|_| None).collect();
        }
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        for layer_index in 0..self.network.layers.len() {
            let opts = self.network.layers[layer_index].train;
            // C++ `update_network_gpu` gating (`dont_update`, `burnin_update`,
            // `train_only_bn`): the whole layer is skipped, buffers untouched.
            if !opts.updates_at(iteration_num, burn_in) {
                continue;
            }
            let (owner, filters, weight_shape) = match &self.network.layers[layer_index].kind {
                LayerKind::Convolutional(spec) => {
                    let owner = spec.share_index.unwrap_or(layer_index);
                    if owner != layer_index {
                        continue;
                    }
                    (
                        owner,
                        spec.filters,
                        [
                            spec.filters,
                            self.network.layers[layer_index].input_shape.channels
                                / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                    )
                }
                _ => continue,
            };

            let cached = self.device_conv_params_at(owner, 0).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached GPU convolution params for layer #{owner}"
                ))
            })?;
            let train = self
                .device_train_buffers
                .borrow()
                .get(owner)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing cached GPU train buffers for layer #{owner}"
                    ))
                })?;

            let layer_step = step.with_lr_scale(opts.learning_rate_scale);
            // `grad_centr` (with BN) acts on the sanitised, accumulated weight updates.
            let mut weight_grad = train.weight_updates.clone();
            let pending = train.pending.clone();
            if opts.grad_centr && train.scale_updates.is_some() {
                let f_size = weight_shape[1] * weight_shape[2] * weight_shape[3];
                if let Some(p) = pending.as_ref() {
                    weight_grad = self
                        .backend
                        .saxpy_f32(1.0, &p.weights, &weight_grad, config)
                        .map_err(|err| backend_error("GPU gradient accumulation", err))?;
                }
                weight_grad = self
                    .backend
                    .sanitize_nan_inf_f32(&weight_grad, &weight_shape, NanInfMode::Reset, config)
                    .map_err(|err| backend_error("GPU update reset_nan_and_inf", err))?;
                weight_grad = self
                    .backend
                    .gradient_centralization_f32(&weight_grad, &weight_shape, f_size, config)
                    .map_err(|err| backend_error("GPU gradient centralization", err))?;
            }
            let weights_pending = if opts.grad_centr && train.scale_updates.is_some() {
                None
            } else {
                pending.as_ref().map(|p| &p.weights)
            };
            let mut moments = self.adam_state[owner].take().unwrap_or_default();
            let scales_input = match (cached.bn.as_ref(), train.scale_updates.as_ref()) {
                (Some(bn), Some(updates)) => Some((&bn.scales, updates)),
                (None, Some(_)) => {
                    return Err(Error::InvalidConfig(
                        "GPU batch-norm scale updates exist for a layer without device BN state"
                            .to_owned(),
                    ));
                }
                _ => None,
            };
            let filters_shape = [filters];
            let mut targets = vec![
                GpuOptimTarget {
                    is_weight: true,
                    w: &cached.raw_weights,
                    u: &weight_grad,
                    grad: weights_pending,
                    moments: &mut moments.weights,
                    shape: &weight_shape,
                },
                GpuOptimTarget {
                    is_weight: false,
                    w: &cached.raw_biases,
                    u: &train.bias_updates,
                    grad: pending.as_ref().map(|p| &p.biases),
                    moments: &mut moments.biases,
                    shape: &filters_shape,
                },
            ];
            if let Some((scales, updates)) = scales_input {
                targets.push(GpuOptimTarget {
                    is_weight: false,
                    w: scales,
                    u: updates,
                    grad: pending.as_ref().and_then(|p| p.scales.as_ref()),
                    moments: &mut moments.scales,
                    shape: &filters_shape,
                });
            }
            let mut stepped =
                gpu_optimizer_multi_step(&self.backend, &layer_step, targets, config)?.into_iter();
            let (mut raw_weights, weight_updates) = stepped.next().expect("weights result");
            if opts.clip > 0.0 {
                raw_weights = self
                    .backend
                    .constrain_f32(&raw_weights, &weight_shape, opts.clip, config)
                    .map_err(|err| backend_error("GPU weight clip", err))?;
            }
            let (raw_biases, bias_updates) = stepped.next().expect("biases result");
            let (new_scales, scale_updates) = match stepped.next() {
                Some((w, u)) => (Some(w), Some(u)),
                None => (None, None),
            };
            self.adam_state[owner] = Some(moments);

            let refreshed = self.refresh_device_convolution_params(
                owner,
                raw_weights,
                raw_biases,
                new_scales,
                weight_updates,
                bias_updates,
                scale_updates,
            )?;
            if sync_host_weights {
                self.sync_convolution_params_from_device(owner, &refreshed)?;
            } else {
                self.host_weights_stale = true;
            }
        }
        let device_dcnv4_optimizer_path = !self.requires_cpu_backward_fallback();
        let mut preserve_device_dcnv4_params = vec![false; self.network.layers.len()];
        if device_dcnv4_optimizer_path {
            for layer_index in 0..self.network.layers.len() {
                let layer = &self.network.layers[layer_index];
                if !matches!(layer.kind, LayerKind::Dcnv4(_))
                    || !layer.train.updates_at(iteration_num, burn_in)
                {
                    continue;
                }
                preserve_device_dcnv4_params[layer_index] = match step {
                    GpuOptimizerStep::Sgd { .. } => {
                        self.update_device_dcnv4_sgd(layer_index, &step, config)?
                    }
                    GpuOptimizerStep::Adam(_) => {
                        self.update_device_dcnv4_adam(layer_index, &step, config)?
                    }
                };
            }
        }

        let device_eml_path = !self.requires_cpu_backward_fallback();
        let mut preserve_device_eml_params = vec![false; self.network.layers.len()];
        if device_eml_path {
            for layer_index in 0..self.network.layers.len() {
                let layer = &self.network.layers[layer_index];
                if !matches!(layer.kind, LayerKind::EmlConvolutional(_))
                    || !layer.train.updates_at(iteration_num, burn_in)
                {
                    continue;
                }
                preserve_device_eml_params[layer_index] = self.update_device_eml_optimizer(
                    layer_index,
                    &step,
                    sync_host_weights,
                    config,
                )?;
            }
        }

        let device_moe_path = !self.requires_cpu_backward_fallback();
        let mut preserve_device_moe_params = vec![false; self.network.layers.len()];
        if device_moe_path {
            let buffers = self.device_moe_train_buffers.borrow();
            for (layer_index, layer) in self.network.layers.iter().enumerate() {
                let supports_device_moe_optimizer =
                    match (&layer.kind, self.params.get(layer_index)) {
                        (LayerKind::Moe(spec), Some(LayerParams::Moe(moe))) => {
                            let bn_matches = moe.experts.len() == spec.experts
                                && moe.experts.iter().all(|expert| {
                                    expert.batch_norm.is_some() == spec.expert.batch_normalize
                                });
                            bn_matches
                        }
                        _ => false,
                    };
                if supports_device_moe_optimizer
                    && buffers.get(layer_index).is_some_and(Option::is_some)
                {
                    preserve_device_moe_params[layer_index] = true;
                }
            }
        }
        if device_moe_path {
            for layer_index in 0..self.network.layers.len() {
                let should_update = {
                    let layer = &self.network.layers[layer_index];
                    preserve_device_moe_params[layer_index]
                        && matches!(layer.kind, LayerKind::Moe(_))
                        && layer.train.updates_at(iteration_num, burn_in)
                };
                if !should_update {
                    continue;
                }
                preserve_device_moe_params[layer_index] |= self.update_device_moe_optimizer(
                    layer_index,
                    &step,
                    sync_host_weights,
                    config,
                )?;
            }
        }
        if sync_host_weights {
            self.host_weights_stale = false;
        }
        self.flush_device_moe_gradients_to_host(&preserve_device_moe_params)?;
        if !device_dcnv4_optimizer_path {
            self.flush_device_dcnv4_gradients_to_host(None)?;
        }
        if let GpuOptimizerStep::Adam(adam) = step {
            self.update_cpu_fallback_adam(
                &adam,
                &preserve_device_moe_params,
                &preserve_device_eml_params,
                &preserve_device_dcnv4_params,
            )?;
        } else {
            self.update_cpu_fallback_weights(
                lr / loss_scale,
                momentum,
                decay * loss_scale,
                batch,
                &preserve_device_moe_params,
                &preserve_device_eml_params,
                &preserve_device_dcnv4_params,
            )?;
        }
        Ok(())
    }

    /// Adam step for host-resident parameters, including recurrent, MoE, and
    /// HF/FiLM parameter blocks. Device convolutions are handled by the main loop.
    pub(super) fn update_cpu_fallback_adam(
        &mut self,
        adam: &AdamParams,
        device_moe_layers: &[bool],
        device_eml_layers: &[bool],
        device_dcnv4_layers: &[bool],
    ) -> Result<()> {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        let mut moments_all = self.host_adam_state.borrow_mut();
        if moments_all.len() != self.network.layers.len() {
            *moments_all = (0..self.network.layers.len()).map(|_| None).collect();
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        for (index, layer) in self.network.layers.iter().enumerate() {
            if matches!(layer.kind, LayerKind::Convolutional(_))
                || (matches!(layer.kind, LayerKind::Moe(_))
                    && device_moe_layers.get(index).copied().unwrap_or(false))
                || (matches!(layer.kind, LayerKind::EmlConvolutional(_))
                    && device_eml_layers.get(index).copied().unwrap_or(false))
                || (matches!(layer.kind, LayerKind::Dcnv4(_))
                    && device_dcnv4_layers.get(index).copied().unwrap_or(false))
            {
                continue;
            }
            if !layer.train.updates_at(iteration_num, burn_in) {
                continue;
            }
            let layer_adam = AdamParams {
                lr: adam.lr * layer.train.learning_rate_scale,
                ..*adam
            };
            let adam = &layer_adam;
            match (&mut self.params[index], &mut buffers[index]) {
                (
                    params @ LayerParams::HyperbolicFoveation(_),
                    train @ LayerTrainBuffers::HyperbolicFoveation(_),
                )
                | (
                    params @ LayerParams::FoveationFilm(_),
                    train @ LayerTrainBuffers::FoveationFilm(_),
                )
                | (
                    params @ LayerParams::Clifford(_),
                    train @ LayerTrainBuffers::Clifford(_),
                )
                // `[wmhf]` gradients are staged by its native backward; the
                // update stays host-side with the CPU Adam arm.
                | (params @ LayerParams::Wmhf(_), train @ LayerTrainBuffers::Wmhf(_)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_layer(
                        &layer.kind,
                        layer.input_shape,
                        params,
                        train,
                        moments,
                        adam,
                    )?;
                }
                (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_non_weight(
                        &mut params.scales,
                        &mut train.scale_updates,
                        &mut moments.scales,
                        adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.biases,
                        &mut train.bias_updates,
                        &mut moments.biases,
                        adam,
                    )?;
                }
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_connected(params, train, moments, adam)?;
                }
                (LayerParams::Vit(params), LayerTrainBuffers::Vit(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    let learned_pos_embed =
                        matches!(layer.kind, LayerKind::Vit(ref spec) if spec.learned_pos_embed);
                    apply_host_adam_vit(params, train, moments, adam, learned_pos_embed)?;
                }
                (LayerParams::Swin(params), LayerTrainBuffers::Swin(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_swin(params, train, moments, adam)?;
                }
                (
                    LayerParams::TuckerAttention(params),
                    LayerTrainBuffers::TuckerAttention(train),
                ) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam(
                        &mut params.weights,
                        &mut train.weight_updates,
                        &mut moments.weights,
                        adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.biases,
                        &mut train.bias_updates,
                        &mut moments.biases,
                        adam,
                    )?;
                }
                (LayerParams::DptHead(params), LayerTrainBuffers::DptHead(train)) => {
                    let LayerKind::DptHead(spec) = &layer.kind else {
                        return Err(Error::InvalidConfig("DPT optimizer layer kind mismatch".into()));
                    };
                    let moments = moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_dpt(spec, layer.input_shape, params, train, moments, adam, true)?;
                }
                (LayerParams::MambaVision(params), LayerTrainBuffers::MambaVision(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_mambavision(params, train, moments, adam)?;
                }
                (LayerParams::KimiKda(params), LayerTrainBuffers::KimiKda(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam(
                        &mut params.weights,
                        &mut train.weight_updates,
                        &mut moments.weights,
                        adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.biases,
                        &mut train.bias_updates,
                        &mut moments.biases,
                        adam,
                    )?;
                }
                (LayerParams::GraphConv(params), LayerTrainBuffers::GraphConv(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(
                        &mut params.projection,
                        &mut train.projection,
                        moments,
                        adam,
                    )?;
                    match (&mut params.self_weights, &mut train.self_weight_updates) {
                        (Some(weights), Some(updates)) => {
                            apply_host_adam(
                                weights,
                                updates,
                                &mut host_adam_child(moments, 0).weights,
                                adam,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "GraphConv Adam self-weight mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut params.edge_kernel, &mut train.edge_kernel_updates) {
                        (Some(weights), Some(updates)) => {
                            apply_host_adam(
                                weights,
                                updates,
                                &mut host_adam_child(moments, 1).weights,
                                adam,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "GraphConv Adam edge-kernel mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut params.edge_biases, &mut train.edge_bias_updates) {
                        (Some(biases), Some(updates)) => {
                            apply_host_adam_non_weight(
                                biases,
                                updates,
                                &mut host_adam_child(moments, 1).biases,
                                adam,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "GraphConv Adam edge-bias mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerParams::DeformConv(params), LayerTrainBuffers::DeformConv(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(
                        &mut params.main,
                        &mut train.main,
                        moments,
                        adam,
                    )?;
                    let predictor_adam = AdamParams {
                        lr: adam.lr * 0.01,
                        ..*adam
                    };
                    let offset_state = host_adam_child(moments, 0);
                    apply_host_adam(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        &mut offset_state.weights,
                        &predictor_adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        &mut offset_state.biases,
                        &predictor_adam,
                    )?;
                    match (&mut params.mask_weights, &mut train.mask_weight_updates) {
                        (Some(weights), Some(updates)) => apply_host_adam(
                            weights,
                            updates,
                            &mut host_adam_child(moments, 1).weights,
                            &predictor_adam,
                        )?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "deform_conv Adam mask weight mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut params.mask_biases, &mut train.mask_bias_updates) {
                        (Some(biases), Some(updates)) => apply_host_adam_non_weight(
                            biases,
                            updates,
                            &mut host_adam_child(moments, 1).biases,
                            &predictor_adam,
                        )?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "deform_conv Adam mask bias mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerParams::DsConv(params), LayerTrainBuffers::DsConv(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(
                        &mut params.collapse,
                        &mut train.collapse,
                        moments,
                        adam,
                    )?;
                    let offset_adam = AdamParams {
                        lr: adam.lr * 0.01,
                        ..*adam
                    };
                    let offset_state = host_adam_child(moments, 0);
                    apply_host_adam(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        &mut offset_state.weights,
                        &offset_adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        &mut offset_state.biases,
                        &offset_adam,
                    )?;
                }
                (LayerParams::Dcnv4(params), LayerTrainBuffers::Dcnv4(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    let offset_adam = AdamParams {
                        lr: adam.lr * 0.01,
                        ..*adam
                    };
                    apply_host_adam(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        &mut moments.weights,
                        &offset_adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        &mut host_adam_child(moments, 0).biases,
                        &offset_adam,
                    )?;
                    if params.batch_norm.is_none() {
                        apply_host_adam_non_weight(
                            &mut params.biases,
                            &mut train.bias_updates,
                            &mut moments.biases,
                            adam,
                        )?;
                    }
                    match (&mut params.batch_norm, &mut train.batch_norm) {
                        (Some(params_bn), Some(train_bn)) => apply_host_adam_non_weight(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            &mut moments.scales,
                            adam,
                        )?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "DCNv4 Adam batchnorm mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerParams::Moe(params), LayerTrainBuffers::Moe(train)) => {
                    if params.experts.len() != train.experts.len() {
                        return Err(Error::InvalidConfig(format!(
                            "MoE Adam expert state mismatch at layer #{index}"
                        )));
                    }
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    let router_adam = AdamParams {
                        decay: 0.0,
                        ..*adam
                    };
                    apply_host_adam(
                        &mut params.router_weights,
                        &mut train.router_weight_updates,
                        &mut moments.weights,
                        &router_adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.router_biases,
                        &mut train.router_bias_updates,
                        &mut moments.biases,
                        &router_adam,
                    )?;
                    for (expert_index, (expert, updates)) in params
                        .experts
                        .iter_mut()
                        .zip(&mut train.experts)
                        .enumerate()
                    {
                        apply_host_adam_convolutional(
                            expert,
                            updates,
                            host_adam_child(moments, expert_index),
                            adam,
                        )?;
                    }
                }
                (LayerParams::Shortcut(params), LayerTrainBuffers::Shortcut(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam(
                        &mut params.weights,
                        &mut train.weight_updates,
                        &mut moments.weights,
                        adam,
                    )?;
                }
                (LayerParams::Local(params), LayerTrainBuffers::Local(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam(
                        &mut params.weights,
                        &mut train.weight_updates,
                        &mut moments.weights,
                        adam,
                    )?;
                    apply_host_adam_non_weight(
                        &mut params.biases,
                        &mut train.bias_updates,
                        &mut moments.biases,
                        adam,
                    )?;
                }
                (LayerParams::Gru(params), LayerTrainBuffers::Gru(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_connected(
                        &mut params.input_z,
                        &mut train.input_z,
                        host_adam_child(moments, 0),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.state_z,
                        &mut train.state_z,
                        host_adam_child(moments, 1),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.input_r,
                        &mut train.input_r,
                        host_adam_child(moments, 2),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.state_r,
                        &mut train.state_r,
                        host_adam_child(moments, 3),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.input_h,
                        &mut train.input_h,
                        host_adam_child(moments, 4),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.state_h,
                        &mut train.state_h,
                        host_adam_child(moments, 5),
                        adam,
                    )?;
                }
                (LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_connected(
                        &mut params.input,
                        &mut train.input,
                        host_adam_child(moments, 0),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.self_params,
                        &mut train.self_params,
                        host_adam_child(moments, 1),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.output,
                        &mut train.output,
                        host_adam_child(moments, 2),
                        adam,
                    )?;
                }
                (LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_connected(
                        &mut params.wf,
                        &mut train.wf,
                        host_adam_child(moments, 0),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.wi,
                        &mut train.wi,
                        host_adam_child(moments, 1),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.wg,
                        &mut train.wg,
                        host_adam_child(moments, 2),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.wo,
                        &mut train.wo,
                        host_adam_child(moments, 3),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.uf,
                        &mut train.uf,
                        host_adam_child(moments, 4),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.ui,
                        &mut train.ui,
                        host_adam_child(moments, 5),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.ug,
                        &mut train.ug,
                        host_adam_child(moments, 6),
                        adam,
                    )?;
                    apply_host_adam_connected(
                        &mut params.uo,
                        &mut train.uo,
                        host_adam_child(moments, 7),
                        adam,
                    )?;
                }
                (LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(
                        &mut params.input,
                        &mut train.input,
                        host_adam_child(moments, 0),
                        adam,
                    )?;
                    apply_host_adam_convolutional(
                        &mut params.self_params,
                        &mut train.self_params,
                        host_adam_child(moments, 1),
                        adam,
                    )?;
                    apply_host_adam_convolutional(
                        &mut params.output,
                        &mut train.output,
                        host_adam_child(moments, 2),
                        adam,
                    )?;
                }
                (
                    LayerParams::EmlConvolutional(params),
                    LayerTrainBuffers::EmlConvolutional(train),
                ) => {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(
                        &mut params.input,
                        &mut train.input,
                        host_adam_child(moments, 0),
                        adam,
                    )?;
                    apply_host_adam_convolutional(
                        &mut params.self_params,
                        &mut train.self_params,
                        host_adam_child(moments, 1),
                        adam,
                    )?;
                }
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train))
                    if matches!(layer.kind, LayerKind::Deconvolutional(_)) =>
                {
                    let moments =
                        moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_convolutional(params, train, moments, adam)?;
                }
                (LayerParams::None, _) | (LayerParams::Convolutional(_), _) => {}
                (params @ LayerParams::RecursiveBlock(_), train @ LayerTrainBuffers::RecursiveBlock(_)) => {
                    let moments = moments_all[index].get_or_insert_with(HostAdamLayerState::default);
                    apply_host_adam_layer(&layer.kind, layer.input_shape, params, train, moments, adam)?;
                }
                (LayerParams::DepthAnything(_), LayerTrainBuffers::None) => {}
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "adam=1 on the GPU path has no optimizer state for layer #{} ({})",
                        index,
                        layer.kind.name()
                    )));
                }
            }
        }
        // Host-updated container branches need re-uploading. Device optimizer
        // updates retain raw tensors for deferred synchronization and the next
        // Oxide forward pass.
        let mut cache = self.device_conv_params.borrow_mut();
        for (index, layer) in self.network.layers.iter().enumerate() {
            if (matches!(layer.kind, LayerKind::EmlConvolutional(_))
                && !device_eml_layers.get(index).copied().unwrap_or(false))
                || (matches!(layer.kind, LayerKind::Moe(_))
                    && !device_moe_layers.get(index).copied().unwrap_or(false))
            {
                if let Some(slots) = cache.get_mut(index) {
                    slots.clear();
                }
            }
        }
        Ok(())
    }

    pub(super) fn update_cpu_fallback_weights(
        &mut self,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        preserve_device_moe_params: &[bool],
        preserve_device_eml_params: &[bool],
        preserve_device_dcnv4_params: &[bool],
    ) -> Result<()> {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        let train_buffers = self.cpu_fallback_train_buffers.get_mut();
        for (layer_index, ((params, train), layer)) in self
            .params
            .iter_mut()
            .zip(train_buffers.iter_mut())
            .zip(&self.network.layers)
            .enumerate()
        {
            if matches!(layer.kind, LayerKind::Convolutional(_))
                || (matches!(layer.kind, LayerKind::Dcnv4(_))
                    && preserve_device_dcnv4_params
                        .get(layer_index)
                        .copied()
                        .unwrap_or(false))
            {
                continue;
            }
            if !layer.train.updates_at(iteration_num, burn_in) {
                continue;
            }
            match (params, train) {
                (
                    LayerParams::HyperbolicFoveation(params),
                    LayerTrainBuffers::HyperbolicFoveation(train),
                ) => {
                    let LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
                        unreachable!("hyperbolic-foveation train buffers match layer kind")
                    };
                    update_hyperbolic_foveation_sgd(
                        spec,
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        batch,
                    );
                }
                (LayerParams::FoveationFilm(params), LayerTrainBuffers::FoveationFilm(train)) => {
                    update_foveation_film_sgd(
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        batch,
                    );
                }
                (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
                    update_batchnorm_params(params, train, lr, momentum, batch);
                }
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
                    update_connected_params(params, train, lr, momentum, decay, batch);
                }
                (LayerParams::Local(params), LayerTrainBuffers::Local(train)) => {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                (LayerParams::Shortcut(params), LayerTrainBuffers::Shortcut(train)) => {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (
                    LayerParams::EmlConvolutional(params),
                    LayerTrainBuffers::EmlConvolutional(train),
                ) if !preserve_device_eml_params
                    .get(layer_index)
                    .copied()
                    .unwrap_or(false) =>
                {
                    let lr = lr * layer.train.learning_rate_scale;
                    update_convolutional_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Moe(params), LayerTrainBuffers::Moe(train)) => {
                    if !preserve_device_moe_params
                        .get(layer_index)
                        .copied()
                        .unwrap_or(false)
                    {
                        sgd_update(
                            &mut params.router_biases,
                            &mut train.router_bias_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                        sgd_update(
                            &mut params.router_weights,
                            &mut train.router_weight_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                        for (expert, updates) in params.experts.iter_mut().zip(&mut train.experts) {
                            update_convolutional_params(
                                expert, updates, lr, momentum, decay, batch,
                            );
                        }
                    }
                }
                (LayerParams::Vit(params), LayerTrainBuffers::Vit(train)) => {
                    let learned_pos =
                        matches!(layer.kind, LayerKind::Vit(ref spec) if spec.learned_pos_embed);
                    update_vit_params(
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                        learned_pos,
                    );
                }
                (LayerParams::Clifford(params), LayerTrainBuffers::Clifford(train)) => {
                    let LayerKind::Clifford(spec) = &layer.kind else {
                        return Err(Error::InvalidConfig(
                            "Clifford parameters do not match layer kind during GPU update".to_owned(),
                        ));
                    };
                    // Native backward stages positive gradients, as the CPU.
                    update_clifford_params(
                        spec,
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Swin(params), LayerTrainBuffers::Swin(train)) => {
                    update_swin_params(
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::DptHead(params), LayerTrainBuffers::DptHead(train)) => {
                    let LayerKind::DptHead(spec) = &layer.kind else {
                        return Err(Error::InvalidConfig("DPT optimizer layer kind mismatch".into()));
                    };
                    update_dpt_head_sgd(spec, layer.input_shape, params, train,
                        lr * layer.train.learning_rate_scale, momentum, decay, batch);
                }
                (LayerParams::RecursiveBlock(params), LayerTrainBuffers::RecursiveBlock(train)) => {
                    let LayerKind::RecursiveBlock(spec) = &layer.kind else {
                        return Err(Error::InvalidConfig("recursive optimizer layer kind mismatch".into()));
                    };
                    update_recursive_block_sgd(spec, params, train,
                        lr * layer.train.learning_rate_scale, momentum, decay, batch);
                }
                (
                    LayerParams::TuckerAttention(params),
                    LayerTrainBuffers::TuckerAttention(train),
                ) => {
                    update_tucker_attention_params(
                        params,
                        train,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::MambaVision(params), LayerTrainBuffers::MambaVision(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    let blocks = crate::mambavision::mambavision_param_blocks(params)
                        .expect("validated MambaVision parameter geometry");
                    for block in blocks {
                        sgd_update(
                            &mut params.values[block.range.clone()],
                            &mut train.updates[block.range],
                            layer_lr,
                            momentum,
                            if block.decay { decay } else { 0.0 },
                            batch,
                        );
                    }
                }
                (LayerParams::KimiKda(params), LayerTrainBuffers::KimiKda(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        layer_lr,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                (LayerParams::GraphConv(params), LayerTrainBuffers::GraphConv(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    update_convolutional_params(
                        &mut params.projection,
                        &mut train.projection,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                    match (&mut params.self_weights, &mut train.self_weight_updates) {
                        (Some(weights), Some(updates)) => {
                            sgd_update(weights, updates, layer_lr, momentum, decay, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated GraphConv self-weight buffers"),
                    }
                    match (&mut params.edge_kernel, &mut train.edge_kernel_updates) {
                        (Some(weights), Some(updates)) => {
                            sgd_update(weights, updates, layer_lr, momentum, decay, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated GraphConv edge-kernel buffers"),
                    }
                    match (&mut params.edge_biases, &mut train.edge_bias_updates) {
                        (Some(biases), Some(updates)) => {
                            sgd_update(biases, updates, layer_lr, momentum, 0.0, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated GraphConv edge-bias buffers"),
                    }
                }
                (LayerParams::DeformConv(params), LayerTrainBuffers::DeformConv(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    update_convolutional_params(
                        &mut params.main,
                        &mut train.main,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        layer_lr * 0.01,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        layer_lr * 0.01,
                        momentum,
                        0.0,
                        batch,
                    );
                    match (&mut params.mask_weights, &mut train.mask_weight_updates) {
                        (Some(weights), Some(updates)) => {
                            sgd_update(weights, updates, layer_lr * 0.01, momentum, decay, batch)
                        }
                        (None, None) => {}
                        _ => unreachable!("validated deform_conv mask train buffers"),
                    }
                    match (&mut params.mask_biases, &mut train.mask_bias_updates) {
                        (Some(biases), Some(updates)) => {
                            sgd_update(biases, updates, layer_lr * 0.01, momentum, 0.0, batch)
                        }
                        (None, None) => {}
                        _ => unreachable!("validated deform_conv mask bias buffers"),
                    }
                }
                (LayerParams::DsConv(params), LayerTrainBuffers::DsConv(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    update_convolutional_params(
                        &mut params.collapse,
                        &mut train.collapse,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        layer_lr * 0.01,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        layer_lr * 0.01,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                // Host-staged `[wmhf]` update, the CPU `update_weights` arm:
                // the sub-convolutions take the network rate and only the scan
                // weights take the layer's `learning_rate` scale.
                (LayerParams::Wmhf(params), LayerTrainBuffers::Wmhf(train)) => {
                    for (conv, updates) in [
                        (&mut params.pre, &mut train.pre),
                        (&mut params.local_dw3, &mut train.local_dw3),
                        (&mut params.local_dw5, &mut train.local_dw5),
                        (&mut params.local_dw7, &mut train.local_dw7),
                        (&mut params.local_mix, &mut train.local_mix),
                        (&mut params.hf_gate, &mut train.hf_gate),
                        (&mut params.fuse, &mut train.fuse),
                    ] {
                        update_convolutional_params(conv, updates, lr, momentum, decay, batch);
                    }
                    sgd_update(
                        &mut params.scan_weights,
                        &mut train.scan_weight_updates,
                        lr * layer.train.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Dcnv4(params), LayerTrainBuffers::Dcnv4(train)) => {
                    let layer_lr = lr * layer.train.learning_rate_scale;
                    sgd_update(
                        &mut params.offset_weights,
                        &mut train.offset_weight_updates,
                        layer_lr * 0.01,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.offset_biases,
                        &mut train.offset_bias_updates,
                        layer_lr * 0.01,
                        momentum,
                        0.0,
                        batch,
                    );
                    if let (Some(params_bn), Some(train_bn)) =
                        (&mut params.batch_norm, &mut train.batch_norm)
                    {
                        sgd_update(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            layer_lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    } else {
                        sgd_update(
                            &mut params.biases,
                            &mut train.bias_updates,
                            layer_lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    }
                }
                (LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
                    update_connected_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
                    update_connected_params(
                        &mut params.wf,
                        &mut train.wf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wi,
                        &mut train.wi,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wg,
                        &mut train.wg,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.wo,
                        &mut train.wo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uf,
                        &mut train.uf,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ui,
                        &mut train.ui,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.ug,
                        &mut train.ug,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_connected_params(
                        &mut params.uo,
                        &mut train.uo,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
                    update_convolutional_params(
                        &mut params.input,
                        &mut train.input,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.self_params,
                        &mut train.self_params,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut params.output,
                        &mut train.output,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train))
                    if matches!(layer.kind, LayerKind::Deconvolutional(_)) =>
                {
                    sgd_update(
                        &mut params.weights,
                        &mut train.weight_updates,
                        lr,
                        momentum,
                        decay,
                        batch,
                    );
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                (LayerParams::None, LayerTrainBuffers::None)
                | (LayerParams::Convolutional(_), LayerTrainBuffers::Convolutional(_)) => {}
                _ => {
                    let _ = layer_index;
                }
            }
        }
        // Host-updated EML branches and fallback MoE experts are reuploaded;
        // device-updated EML branches keep their refreshed branch slots.
        let mut cache = self.device_conv_params.borrow_mut();
        for (index, layer) in self.network.layers.iter().enumerate() {
            if (matches!(layer.kind, LayerKind::EmlConvolutional(_))
                && !preserve_device_eml_params
                    .get(index)
                    .copied()
                    .unwrap_or(false))
                || (matches!(layer.kind, LayerKind::Moe(_))
                    && !preserve_device_moe_params
                        .get(index)
                        .copied()
                        .unwrap_or(false))
            {
                if let Some(slots) = cache.get_mut(index) {
                    slots.clear();
                }
            }
        }
        Ok(())
    }

}
