use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(super) fn requires_cpu_backward_fallback(&self) -> bool {
        self.network.layers.iter().any(|layer| {
            match &layer.kind {
                // Recurrent projection/convolution backward supports BN through
                // rolling-stat affine derivatives when the activation is supported.
                LayerKind::Rnn(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none()
                        || (spec.logistic == 2 && Activation::Loggy.to_cuda_f32_kind().is_none())
                    {
                        return true;
                    }
                }
                LayerKind::Lstm(_) => {}
                LayerKind::Crnn(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                // MoE router and expert backward, including the rolling-stat
                // batch-norm affine derivative, use backend ops.
                LayerKind::Moe(spec) => {
                    if spec.expert.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                // Use the Rust host reference and have no native device backward.
                // `[deform_conv]` now has one (see the `has_host_only_deform`
                // guard for the variants that still need the host path).
                LayerKind::Dcnv4(spec) => {
                    if dcnv4_gpu::needs_host(spec) {
                        return true;
                    }
                }
                LayerKind::Deconvolutional(spec) => {
                    if !deconvolutional_spec_is_native(spec) {
                        return true;
                    }
                }
                // Weighted shortcut parameter reductions stay host-resident,
                // while input/source gradients use backend elementwise ops.
                // `[laya_act_head]` and the full ViT backward still use the
                // host fallback; standard-flash dQ/dK/dV dispatches to Oxide
                // from that fallback when the head dimension is supported.
                LayerKind::LayaActHead(_)
                | LayerKind::Vit(_)
                | LayerKind::Swin(_)
                | LayerKind::TuckerAttention(_)
                | LayerKind::MambaVision(_)
                | LayerKind::KimiKda(_) => return true,
                // Connected layers are handled on the GPU path with a host
                // round trip unless their activation or batch norm
                // (batch-statistics training) is unsupported.
                LayerKind::Connected(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                LayerKind::Local(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                LayerKind::Sam(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none() {
                        return true;
                    }
                }
                LayerKind::GraphConv(spec) if graph_conv_spec_needs_host(spec) => return true,
                _ => {}
            }
            if let LayerKind::ScaleChannels(spec) = &layer.kind
                && spec.activation.to_cuda_f32_kind().is_none()
            {
                return true;
            }
            false
        })
    }

    pub(super) fn backward_accumulate_seeded_gpu_cpu_fallback(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        let train_buffers_uninitialized = {
            let conv = self.device_train_buffers.borrow();
            let moe = self.device_moe_train_buffers.borrow();
            let eml = self.device_eml_train_buffers.borrow();
            conv.len() != self.network.layers.len()
                || moe.len() != self.network.layers.len()
                || eml.len() != self.network.layers.len()
                || (conv.iter().all(Option::is_none)
                    && moe.iter().all(Option::is_none)
                    && eml.iter().all(Option::is_none))
        };
        if train_buffers_uninitialized {
            self.init_train_buffers_gpu()?;
        }
        let input = self
            .download_tensor(&forward.input_gpu, self.network.net.input_shape())
            .map_err(|err| backend_error("download CPU fallback input", err))?;
        let mut cpu = ExecutableNetwork::new(self.network.clone());
        cpu.params = self.params.clone();
        cpu.train_buffers = self.cpu_fallback_train_buffers.borrow().clone();
        for (layer, train) in self.network.layers.iter().zip(&mut cpu.train_buffers) {
            if matches!(layer.kind, LayerKind::Convolutional(_)) {
                *train = default_layer_train_buffers(layer);
            }
        }
        let cpu_backward = cpu.backward_accumulate_seeded_with_attention(
            &input,
            &forward.cpu_forward,
            layer_output_gradient_seeds,
            |d_context, context, attention| {
                let Some(backward_inputs) = crate::vit::attention_flash_backward_inputs(attention)?
                else {
                    return crate::vit::attention_backward(d_context, attention);
                };
                if backward_inputs.channels / backward_inputs.heads > 128 {
                    return crate::vit::attention_backward(d_context, attention);
                }
                let qkv = self
                    .backend
                    .upload_f32(
                        &[
                            backward_inputs.batch,
                            backward_inputs.tokens,
                            3 * backward_inputs.channels,
                        ],
                        &backward_inputs.qkv,
                    )
                    .map_err(|err| format!("upload ViT flash backward QKV: {err}"))?;
                let context_tensor = self
                    .backend
                    .upload_f32(
                        &[
                            backward_inputs.batch,
                            backward_inputs.tokens,
                            backward_inputs.channels,
                        ],
                        context,
                    )
                    .map_err(|err| format!("upload ViT flash backward context: {err}"))?;
                let grad_tensor = self
                    .backend
                    .upload_f32(
                        &[
                            backward_inputs.batch,
                            backward_inputs.tokens,
                            backward_inputs.channels,
                        ],
                        d_context,
                    )
                    .map_err(|err| format!("upload ViT flash output gradient: {err}"))?;
                let lse_tensor = self
                    .backend
                    .upload_f32(
                        &[
                            backward_inputs.batch,
                            backward_inputs.heads,
                            backward_inputs.tokens,
                        ],
                        &backward_inputs.lse,
                    )
                    .map_err(|err| format!("upload ViT flash LSE: {err}"))?;
                let dqkv = self
                    .backend
                    .flash_attention_backward_f32(
                        &qkv,
                        &context_tensor,
                        &grad_tensor,
                        &lse_tensor,
                        &[
                            backward_inputs.batch,
                            backward_inputs.tokens,
                            3 * backward_inputs.channels,
                        ],
                        backward_inputs.batch,
                        backward_inputs.tokens,
                        backward_inputs.channels,
                        backward_inputs.heads,
                    )
                    .map_err(|err| format!("ViT flash backward: {err}"))?;
                self.backend
                    .download_f32(&dqkv)
                    .map_err(|err| format!("download ViT flash dQKV: {err}"))
            },
        )?;
        if self.capture_input_gradient.get() {
            *self.captured_input_gradient.borrow_mut() = Some(cpu_backward.input_gradient);
        }

        let mut device_conv_gradients = Vec::new();
        for (layer, train) in self.network.layers.iter().zip(&cpu.train_buffers) {
            if !matches!(layer.kind, LayerKind::Convolutional(_)) {
                continue;
            }
            let owner = match &layer.kind {
                LayerKind::Convolutional(spec) => spec.share_index.unwrap_or(layer.index),
                _ => continue,
            };
            if owner != layer.index {
                continue;
            }
            let LayerTrainBuffers::Convolutional(train) = train else {
                continue;
            };
            let Some(LayerParams::Convolutional(_)) = cpu.params.get(owner) else {
                return Err(Error::InvalidConfig(format!(
                    "missing CPU fallback convolution params for layer #{owner}"
                )));
            };
            let weight_grad = self
                .backend
                .upload_f32(&conv_weight_shape(layer), &train.weight_updates)
                .map_err(|err| backend_error("upload CPU convolution weight gradient", err))?;
            let bias_grad = self
                .backend
                .upload_f32(&[train.bias_updates.len()], &train.bias_updates)
                .map_err(|err| backend_error("upload CPU convolution bias gradient", err))?;
            let scale_grad = train
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    self.backend
                        .upload_f32(&[batch_norm.scale_updates.len()], &batch_norm.scale_updates)
                        .map_err(|err| backend_error("upload CPU batch-norm gradient", err))
                })
                .transpose()?;
            device_conv_gradients.push((owner, weight_grad, bias_grad, scale_grad));
        }
        for (owner, weight_grad, bias_grad, scale_grad) in device_conv_gradients {
            self.accumulate_device_train_buffer(owner, weight_grad, bias_grad, scale_grad)?;
        }

        let fallback_buffers = self.cpu_fallback_train_buffers.get_mut();
        for (layer_index, (layer, train)) in self
            .network
            .layers
            .iter()
            .zip(cpu.train_buffers)
            .enumerate()
        {
            if !matches!(layer.kind, LayerKind::Convolutional(_)) {
                fallback_buffers[layer_index] = train;
            }
        }
        self.params = cpu.params;
        self.clear_device_conv_params();
        self.warm_device_conv_params()?;
        Ok(())
    }

    pub(crate) fn download_tensor(
        &self,
        tensor: &B::Tensor,
        shape: TensorShape,
    ) -> std::result::Result<Tensor, B::Error> {
        let data = self.backend.download_f32(tensor)?;
        Ok(Tensor { shape, data })
    }

    pub(crate) fn accumulate_previous_gpu_gradient(
        &self,
        layer_index: usize,
        gradient: B::Tensor,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        if layer_index == 0 {
            if self.capture_input_gradient.get() {
                let input_gradient = self
                    .download_tensor(&gradient, self.network.net.input_shape())
                    .map_err(|err| backend_error("download captured GPU input gradient", err))?;
                let mut captured = self.captured_input_gradient.borrow_mut();
                if let Some(existing) = captured.as_mut() {
                    if existing.shape != input_gradient.shape {
                        return Err(Error::InvalidConfig(
                            "captured GPU input-gradient shape mismatch".to_owned(),
                        ));
                    }
                    accumulate_vec(&mut existing.data, &input_gradient.data)?;
                } else {
                    *captured = Some(input_gradient);
                }
            }
            return Ok(());
        }
        self.accumulate_gpu_gradient(&mut layer_output_gradients[layer_index - 1], gradient)
    }

    /// Download and accumulate deconvolution gradients into the host optimizer
    /// buffers. The layer math stays native; the existing optimizer owns host params.
    pub(crate) fn accumulate_deconvolutional_host_gradients(
        &self,
        layer_index: usize,
        weight_grad: &B::Tensor,
        bias_grad: &B::Tensor,
        weight_count: usize,
        bias_count: usize,
    ) -> Result<()> {
        let dw = self
            .backend
            .download_f32(weight_grad)
            .map_err(|err| backend_error("download deconvolutional weight gradient", err))?;
        let db = self
            .backend
            .download_f32(bias_grad)
            .map_err(|err| backend_error("download deconvolutional bias gradient", err))?;
        if dw.len() != weight_count || db.len() != bias_count {
            return Err(Error::InvalidConfig(format!(
                "deconvolutional gradient length mismatch at layer #{layer_index}"
            )));
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Convolutional(train)) = buffers.get_mut(layer_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing deconvolutional train buffers for layer #{layer_index}"
            )));
        };
        if train.weight_updates.len() != weight_count || train.bias_updates.len() != bias_count {
            return Err(Error::InvalidConfig(format!(
                "deconvolutional train buffer mismatch for layer #{layer_index}"
            )));
        }
        for (slot, value) in train.weight_updates.iter_mut().zip(dw) {
            *slot += value;
        }
        for (slot, value) in train.bias_updates.iter_mut().zip(db) {
            *slot += value;
        }
        Ok(())
    }

    /// Accumulate device-computed local gradients into the host optimizer
    /// buffers; each location retains its own weights and bias parameters.
    pub(crate) fn accumulate_local_host_gradients(
        &self,
        layer_index: usize,
        weight_grad: &B::Tensor,
        bias_grad: &B::Tensor,
        weight_count: usize,
        bias_count: usize,
    ) -> Result<()> {
        let dw = self
            .backend
            .download_f32(weight_grad)
            .map_err(|err| backend_error("download local weight gradient", err))?;
        let db = self
            .backend
            .download_f32(bias_grad)
            .map_err(|err| backend_error("download local bias gradient", err))?;
        if dw.len() != weight_count || db.len() != bias_count {
            return Err(Error::InvalidConfig(format!(
                "local gradient length mismatch at layer #{layer_index}"
            )));
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Local(train)) = buffers.get_mut(layer_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing local train buffers for layer #{layer_index}"
            )));
        };
        if train.weight_updates.len() != weight_count || train.bias_updates.len() != bias_count {
            return Err(Error::InvalidConfig(format!(
                "local train buffer mismatch for layer #{layer_index}"
            )));
        }
        for (slot, value) in train.weight_updates.iter_mut().zip(dw) {
            *slot += value;
        }
        for (slot, value) in train.bias_updates.iter_mut().zip(db) {
            *slot += value;
        }
        Ok(())
    }

    /// Accumulate a connected layer's host weight/bias gradient into its
    /// `LayerTrainBuffers` (the layer's parameters stay on the host; the
    /// optimizer's `update_cpu_fallback_weights` step consumes them).
    pub(crate) fn accumulate_connected_host_gradients(
        &self,
        layer_index: usize,
        weight_grad: &B::Tensor,
        bias_grad: &B::Tensor,
        inputs: usize,
        outputs: usize,
    ) -> Result<()> {
        let dw = self
            .backend
            .download_f32(weight_grad)
            .map_err(|err| backend_error("download connected weight gradient", err))?;
        let db = self
            .backend
            .download_f32(bias_grad)
            .map_err(|err| backend_error("download connected bias gradient", err))?;
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Connected(train)) = buffers.get_mut(layer_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing connected train buffers for layer #{layer_index}"
            )));
        };
        if train.weight_updates.len() != outputs * inputs || train.bias_updates.len() != outputs {
            return Err(Error::InvalidConfig(format!(
                "connected train buffer mismatch for layer #{layer_index}"
            )));
        }
        for (slot, value) in train.weight_updates.iter_mut().zip(dw) {
            *slot += value;
        }
        for (slot, value) in train.bias_updates.iter_mut().zip(db) {
            *slot += value;
        }
        Ok(())
    }

    pub(crate) fn accumulate_eml_device_gradients(
        &self,
        layer_index: usize,
        branch_slot: usize,
        weight_gradient: B::Tensor,
        bias_gradient: B::Tensor,
        scale_gradient: Option<B::Tensor>,
    ) -> Result<()> {
        let mut branch = {
            let buffers = self.device_eml_train_buffers.borrow();
            let train = buffers
                .get(layer_index)
                .and_then(Clone::clone)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing GPU EML train buffers for layer #{layer_index}"
                    ))
                })?;
            match branch_slot {
                0 => train.input,
                1 => train.self_params,
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "invalid EML branch slot {branch_slot} at layer #{layer_index}"
                    )));
                }
            }
        };
        if branch.scale_updates.is_some() != scale_gradient.is_some() {
            return Err(Error::InvalidConfig(format!(
                "EML GPU BN scale-gradient mismatch at layer #{layer_index}, slot {branch_slot}"
            )));
        }

        if self.defer_host_sync {
            if let Some(pending) = branch.pending.take() {
                let flushed = self.flush_pending_grads(&branch, &pending)?;
                branch.weight_updates = flushed.0;
                branch.bias_updates = flushed.1;
                branch.scale_updates = flushed.2;
            }
            branch.pending = Some(GpuPendingGrads {
                weights: weight_gradient,
                biases: bias_gradient,
                scales: scale_gradient,
            });
        } else {
            let config = ElementwiseLaunchConfig::default();
            branch.weight_updates = self
                .backend
                .saxpy_f32(1.0, &weight_gradient, &branch.weight_updates, config)
                .map_err(|err| backend_error("EML GPU weight-gradient accumulation", err))?;
            branch.bias_updates = self
                .backend
                .saxpy_f32(1.0, &bias_gradient, &branch.bias_updates, config)
                .map_err(|err| backend_error("EML GPU bias-gradient accumulation", err))?;
            branch.scale_updates = match (branch.scale_updates, scale_gradient) {
                (Some(updates), Some(gradient)) => Some(
                    self.backend
                        .saxpy_f32(1.0, &gradient, &updates, config)
                        .map_err(|err| {
                            backend_error("EML GPU BN scale-gradient accumulation", err)
                        })?,
                ),
                (None, None) => None,
                (Some(updates), None) => Some(updates),
                (None, Some(_)) => unreachable!("presence checked above"),
            };
            branch.pending = None;
        }
        let mut buffers = self.device_eml_train_buffers.borrow_mut();
        let train = buffers
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU EML train buffers for layer #{layer_index}"
                ))
            })?;
        match branch_slot {
            0 => train.input = branch,
            1 => train.self_params = branch,
            _ => unreachable!("branch slot validated above"),
        }
        Ok(())
    }

    pub(crate) fn accumulate_gpu_gradient(
        &self,
        slot: &mut Option<B::Tensor>,
        gradient: B::Tensor,
    ) -> Result<()> {
        *slot = Some(match slot.take() {
            Some(existing) => self
                .backend
                .vector_add_f32(&existing, &gradient, ElementwiseLaunchConfig::default())
                .map_err(|err| backend_error("GPU gradient accumulation", err))?,
            None => gradient,
        });
        Ok(())
    }

    pub(super) fn accumulate_device_train_buffer(
        &self,
        owner: usize,
        weight_grad: B::Tensor,
        bias_grad: B::Tensor,
        scale_grad: Option<B::Tensor>,
    ) -> Result<()> {
        let Some(train) = self
            .device_train_buffers
            .borrow()
            .get(owner)
            .and_then(Clone::clone)
        else {
            return Err(Error::InvalidConfig(format!(
                "missing GPU train buffers for layer #{owner}"
            )));
        };
        if self.defer_host_sync && scale_grad.is_some() == train.scale_updates.is_some() {
            // Keep this gradient pending: the optimizer step adds it to the
            // accumulators in its own launch.  An earlier pending gradient (a
            // subdivision) is added first, so the summation order is unchanged.
            let mut train = train;
            if let Some(pending) = train.pending.take() {
                let flushed = self.flush_pending_grads(&train, &pending)?;
                train.weight_updates = flushed.0;
                train.bias_updates = flushed.1;
                train.scale_updates = flushed.2;
            }
            train.pending = Some(GpuPendingGrads {
                weights: weight_grad,
                biases: bias_grad,
                scales: scale_grad,
            });
            self.device_train_buffers.borrow_mut()[owner] = Some(train);
            return Ok(());
        }
        let weight_updates = self
            .backend
            .saxpy_f32(
                1.0,
                &weight_grad,
                &train.weight_updates,
                ElementwiseLaunchConfig::default(),
            )
            .map_err(|err| backend_error("GPU weight gradient accumulation", err))?;
        let bias_updates = self
            .backend
            .saxpy_f32(
                1.0,
                &bias_grad,
                &train.bias_updates,
                ElementwiseLaunchConfig::default(),
            )
            .map_err(|err| backend_error("GPU bias gradient accumulation", err))?;
        let scale_updates = match (train.scale_updates, scale_grad) {
            (Some(scale_updates), Some(scale_grad)) => Some(
                self.backend
                    .saxpy_f32(
                        1.0,
                        &scale_grad,
                        &scale_updates,
                        ElementwiseLaunchConfig::default(),
                    )
                    .map_err(|err| {
                        backend_error("GPU batch-norm scale gradient accumulation", err)
                    })?,
            ),
            (None, None) => None,
            (Some(scale_updates), None) => Some(scale_updates),
            (None, Some(_)) => {
                return Err(Error::InvalidConfig(format!(
                    "GPU batch-norm scale gradient produced for non-BN layer #{owner}"
                )));
            }
        };
        self.device_train_buffers.borrow_mut()[owner] = Some(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
            pending: None,
        });
        Ok(())
    }

    pub(crate) fn accumulate_device_moe_train_buffer(
        &self,
        layer_index: usize,
        router_weight_gradient: &B::Tensor,
        router_bias_gradient: &B::Tensor,
        expert_weight_gradients: &[B::Tensor],
        expert_bias_gradients: &[B::Tensor],
        expert_scale_gradients: &[Option<B::Tensor>],
    ) -> Result<()> {
        let current = self
            .device_moe_train_buffers
            .borrow()
            .get(layer_index)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU MoE train buffers for layer #{layer_index}"
                ))
            })?;
        if current.expert_weight_updates.len() != expert_weight_gradients.len()
            || current.expert_bias_updates.len() != expert_bias_gradients.len()
            || current.expert_scale_updates.len() != expert_scale_gradients.len()
            || expert_weight_gradients.len() != expert_bias_gradients.len()
        {
            return Err(Error::InvalidConfig(format!(
                "MoE GPU gradient count mismatch for layer #{layer_index}"
            )));
        }
        let config = ElementwiseLaunchConfig::default();
        let add = |gradient: &B::Tensor, updates: &B::Tensor| {
            if current.dirty || !self.network.net.adam {
                self.backend
                    .saxpy_f32(1.0, gradient, updates, config)
                    .map_err(|err| backend_error("MoE GPU gradient accumulation", err))
            } else {
                Ok(gradient.clone())
            }
        };
        let router_weight_updates = add(router_weight_gradient, &current.router_weight_updates)?;
        let router_bias_updates = add(router_bias_gradient, &current.router_bias_updates)?;
        let mut expert_weight_updates = Vec::with_capacity(expert_weight_gradients.len());
        let mut expert_bias_updates = Vec::with_capacity(expert_bias_gradients.len());
        let mut expert_scale_updates = Vec::with_capacity(expert_scale_gradients.len());
        for (
            (((weight_gradient, bias_gradient), scale_gradient), (weight_updates, bias_updates)),
            scale_updates,
        ) in expert_weight_gradients
            .iter()
            .zip(expert_bias_gradients)
            .zip(expert_scale_gradients)
            .zip(
                current
                    .expert_weight_updates
                    .iter()
                    .zip(&current.expert_bias_updates),
            )
            .zip(&current.expert_scale_updates)
        {
            expert_weight_updates.push(add(weight_gradient, weight_updates)?);
            expert_bias_updates.push(add(bias_gradient, bias_updates)?);
            expert_scale_updates.push(match (scale_gradient, scale_updates) {
                (Some(gradient), Some(updates)) => Some(add(gradient, updates)?),
                (None, None) => None,
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "MoE GPU scale-gradient presence mismatch at layer #{layer_index}"
                    )));
                }
            });
        }
        let mut buffers = self.device_moe_train_buffers.borrow_mut();
        let slot = buffers
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU MoE train buffers for layer #{layer_index}"
                ))
            })?;
        slot.router_weight_updates = router_weight_updates;
        slot.router_bias_updates = router_bias_updates;
        slot.expert_weight_updates = expert_weight_updates;
        slot.expert_bias_updates = expert_bias_updates;
        slot.expert_scale_updates = expert_scale_updates;
        slot.dirty = true;
        Ok(())
    }

    pub(super) fn update_device_eml_optimizer(
        &mut self,
        layer_index: usize,
        step: &GpuOptimizerStep,
        sync_host_weights: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<bool> {
        let (spec, input_shape, train_options) = {
            let layer =
                self.network.layers.get(layer_index).ok_or_else(|| {
                    Error::InvalidConfig(format!("missing EML layer #{layer_index}"))
                })?;
            let LayerKind::EmlConvolutional(spec) = &layer.kind else {
                return Err(Error::InvalidConfig(format!(
                    "layer #{layer_index} is not an EML layer"
                )));
            };
            (spec.clone(), layer.input_shape, layer.train)
        };
        let LayerParams::EmlConvolutional(params) =
            self.params.get(layer_index).ok_or_else(|| {
                Error::InvalidConfig(format!("missing EML params for layer #{layer_index}"))
            })?
        else {
            return Err(Error::InvalidConfig(format!(
                "missing EML params for layer #{layer_index}"
            )));
        };
        let params = params.clone();
        let mut train = self
            .device_eml_train_buffers
            .borrow()
            .get(layer_index)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU EML train buffers for layer #{layer_index}"
                ))
            })?;
        if params.input.batch_norm.is_some() != spec.branch.batch_normalize
            || params.self_params.batch_norm.is_some() != spec.branch.batch_normalize
            || train.input.scale_updates.is_some() != spec.branch.batch_normalize
            || train.self_params.scale_updates.is_some() != spec.branch.batch_normalize
        {
            return Err(Error::InvalidConfig(format!(
                "EML parameter or train-buffer mismatch during device optimizer at layer #{layer_index}"
            )));
        }

        let weight_shape = [
            spec.branch.filters,
            input_shape.channels / spec.branch.groups.max(1),
            spec.branch.size,
            spec.branch.size,
        ];
        let filters_shape = [spec.branch.filters];
        let layer_step = step.with_lr_scale(train_options.learning_rate_scale);
        let mut adam_state = match step {
            GpuOptimizerStep::Adam(_) => self.eml_adam_state_for_update(layer_index)?,
            GpuOptimizerStep::Sgd { .. } => {
                self.eml_adam_state[layer_index].clone().unwrap_or_default()
            }
        };
        for slot in 0..2 {
            let branch_train = if slot == 0 {
                &mut train.input
            } else {
                &mut train.self_params
            };
            let cached = self.device_conv_params_at(layer_index, slot).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached EML convolution params for layer #{layer_index}, slot {slot}"
                ))
            })?;
            if cached.bn.is_some() != spec.branch.batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "EML device BN cache mismatch at layer #{layer_index}, slot {slot}"
                )));
            }
            let pending = branch_train.pending.clone();
            let mut weight_gradient = branch_train.weight_updates.clone();
            if train_options.grad_centr && branch_train.scale_updates.is_some() {
                if let Some(pending) = pending.as_ref() {
                    weight_gradient = self
                        .backend
                        .saxpy_f32(1.0, &pending.weights, &weight_gradient, config)
                        .map_err(|err| backend_error("EML optimizer gradient accumulation", err))?;
                }
                let filter_size = weight_shape[1] * weight_shape[2] * weight_shape[3];
                weight_gradient = self
                    .backend
                    .sanitize_nan_inf_f32(
                        &weight_gradient,
                        &weight_shape,
                        NanInfMode::Reset,
                        config,
                    )
                    .map_err(|err| backend_error("EML optimizer reset_nan_and_inf", err))?;
                weight_gradient = self
                    .backend
                    .gradient_centralization_f32(
                        &weight_gradient,
                        &weight_shape,
                        filter_size,
                        config,
                    )
                    .map_err(|err| backend_error("EML optimizer gradient centralization", err))?;
            }
            let pending_weight = if train_options.grad_centr && branch_train.scale_updates.is_some()
            {
                None
            } else {
                pending.as_ref().map(|pending| &pending.weights)
            };
            let moments = if slot == 0 {
                &mut adam_state.input
            } else {
                &mut adam_state.self_params
            };
            let mut targets = vec![
                GpuOptimTarget {
                    is_weight: true,
                    w: &cached.raw_weights,
                    u: &weight_gradient,
                    grad: pending_weight,
                    moments: &mut moments.weights,
                    shape: &weight_shape,
                },
                GpuOptimTarget {
                    is_weight: false,
                    w: &cached.raw_biases,
                    u: &branch_train.bias_updates,
                    grad: pending.as_ref().map(|pending| &pending.biases),
                    moments: &mut moments.biases,
                    shape: &filters_shape,
                },
            ];
            if let (Some(batch_norm), Some(scale_updates)) =
                (cached.bn.as_ref(), branch_train.scale_updates.as_ref())
            {
                targets.push(GpuOptimTarget {
                    is_weight: false,
                    w: &batch_norm.scales,
                    u: scale_updates,
                    grad: pending.as_ref().and_then(|pending| pending.scales.as_ref()),
                    moments: &mut moments.scales,
                    shape: &filters_shape,
                });
            }
            let mut updates =
                gpu_optimizer_multi_step(&self.backend, &layer_step, targets, config)?.into_iter();
            let (mut raw_weights, weight_updates) = updates.next().expect("EML weights result");
            if train_options.clip > 0.0 {
                raw_weights = self
                    .backend
                    .constrain_f32(&raw_weights, &weight_shape, train_options.clip, config)
                    .map_err(|err| backend_error("EML GPU weight clip", err))?;
            }
            let (raw_biases, bias_updates) = updates.next().expect("EML biases result");
            let (new_scales, scale_updates) = match updates.next() {
                Some((scales, updates)) => (Some(scales), Some(updates)),
                None => (None, None),
            };
            if new_scales.is_some() != spec.branch.batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "EML BN optimizer result mismatch at layer #{layer_index}, slot {slot}"
                )));
            }
            let mut refreshed = cached;
            refreshed.raw_weights = raw_weights.clone();
            refreshed.raw_biases = raw_biases.clone();
            if let (Some(scales), Some(batch_norm)) = (new_scales, refreshed.bn.as_mut()) {
                batch_norm.scales = scales;
                refreshed.folded_stale = true;
            } else {
                refreshed.weights = raw_weights;
                refreshed.biases = raw_biases;
                refreshed.folded_stale = false;
            }
            self.set_device_conv_params(layer_index, slot, refreshed.clone())?;
            if sync_host_weights {
                self.sync_eml_branch_params_from_device(layer_index, slot, &refreshed)?;
            }
            branch_train.weight_updates = weight_updates;
            branch_train.bias_updates = bias_updates;
            branch_train.scale_updates = scale_updates;
            branch_train.pending = None;
        }
        self.device_eml_train_buffers.borrow_mut()[layer_index] = Some(train);
        if !sync_host_weights {
            self.host_weights_stale = true;
        }
        if matches!(step, GpuOptimizerStep::Adam(_)) {
            self.eml_adam_state[layer_index] = Some(adam_state);
            self.host_adam_state.borrow_mut()[layer_index] = None;
        }
        Ok(true)
    }

    pub(super) fn update_device_moe_optimizer(
        &mut self,
        layer_index: usize,
        step: &GpuOptimizerStep,
        sync_host_weights: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<bool> {
        let Some(train) = self
            .device_moe_train_buffers
            .borrow()
            .get(layer_index)
            .and_then(Clone::clone)
        else {
            return Ok(false);
        };
        let layer = self
            .network
            .layers
            .get(layer_index)
            .ok_or_else(|| Error::InvalidConfig(format!("missing MoE layer #{layer_index}")))?;
        let LayerKind::Moe(spec) = &layer.kind else {
            return Err(Error::InvalidConfig(format!(
                "layer #{layer_index} is not a MoE layer"
            )));
        };
        let LayerParams::Moe(params) = self.params.get(layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing MoE params for layer #{layer_index}"))
        })?
        else {
            return Err(Error::InvalidConfig(format!(
                "missing MoE params for layer #{layer_index}"
            )));
        };
        if train.expert_weight_updates.len() != spec.experts
            || train.expert_bias_updates.len() != spec.experts
            || train.expert_scale_updates.len() != spec.experts
            || params.experts.len() != spec.experts
            || params
                .experts
                .iter()
                .any(|expert| expert.batch_norm.is_some() != spec.expert.batch_normalize)
            || train
                .expert_scale_updates
                .iter()
                .any(|scales| scales.is_some() != spec.expert.batch_normalize)
            || params.router_weights.len() != spec.experts * layer.input_shape.channels
            || params.router_biases.len() != spec.experts
        {
            return Err(Error::InvalidConfig(format!(
                "MoE parameter or gradient count mismatch during device optimizer at layer #{layer_index}"
            )));
        }
        let router_weights_host = params.router_weights.clone();
        let router_biases_host = params.router_biases.clone();
        let expert_count = spec.experts;
        let expert_batch_normalize = spec.expert.batch_normalize;
        let grouped_channels = layer.input_shape.channels / spec.expert.groups.max(1);
        let weight_shape = [
            spec.expert.filters,
            grouped_channels,
            spec.expert.size,
            spec.expert.size,
        ];
        let bias_shape = [spec.expert.filters];
        let router_weight_shape = [spec.experts, layer.input_shape.channels];
        let router_bias_shape = [spec.experts];
        let layer_step = match step {
            GpuOptimizerStep::Sgd { .. } => *step,
            GpuOptimizerStep::Adam(_) => step.with_lr_scale(layer.train.learning_rate_scale),
        };
        let mut adam_state = match step {
            GpuOptimizerStep::Sgd { .. } => GpuMoeAdamLayerState {
                router: GpuAdamLayerState::default(),
                experts: (0..spec.experts)
                    .map(|_| GpuAdamLayerState::default())
                    .collect(),
            },
            GpuOptimizerStep::Adam(_) => self.moe_adam_state_for_update(layer_index)?,
        };
        if adam_state.experts.len() != spec.experts
            || adam_state.router.scales.is_some()
            || (!spec.expert.batch_normalize
                && adam_state
                    .experts
                    .iter()
                    .any(|expert| expert.scales.is_some()))
        {
            return Err(Error::InvalidConfig(format!(
                "MoE optimizer state shape mismatch at layer #{layer_index}"
            )));
        }
        let router_weights = self
            .backend
            .upload_f32(&router_weight_shape, &router_weights_host)
            .map_err(|err| backend_error("upload MoE router weights for optimizer update", err))?;
        let router_biases = self
            .backend
            .upload_f32(&router_bias_shape, &router_biases_host)
            .map_err(|err| backend_error("upload MoE router biases for optimizer update", err))?;
        let router_targets = vec![
            GpuOptimTarget {
                is_weight: true,
                w: &router_weights,
                u: &train.router_weight_updates,
                grad: None,
                moments: &mut adam_state.router.weights,
                shape: &router_weight_shape,
            },
            GpuOptimTarget {
                is_weight: false,
                w: &router_biases,
                u: &train.router_bias_updates,
                grad: None,
                moments: &mut adam_state.router.biases,
                shape: &router_bias_shape,
            },
        ];
        let router_step = match layer_step {
            GpuOptimizerStep::Sgd {
                lr,
                momentum,
                batch,
                ..
            } => GpuOptimizerStep::Sgd {
                lr,
                momentum,
                decay: 0.0,
                batch,
            },
            GpuOptimizerStep::Adam(mut params) => {
                params.decay = 0.0;
                GpuOptimizerStep::Adam(params)
            }
        };
        let mut router_updates =
            gpu_optimizer_multi_step(&self.backend, &router_step, router_targets, config)?
                .into_iter();
        let (router_weights, router_weight_momentum) =
            router_updates.next().expect("MoE router weights result");
        let (router_biases, router_bias_momentum) =
            router_updates.next().expect("MoE router biases result");
        let router_weights_host = self.backend.download_f32(&router_weights).map_err(|err| {
            backend_error("download MoE router weights after optimizer update", err)
        })?;
        let router_biases_host = self.backend.download_f32(&router_biases).map_err(|err| {
            backend_error("download MoE router biases after optimizer update", err)
        })?;

        self.warm_device_conv_params()?;
        let mut updated_weight_moments = Vec::with_capacity(expert_count);
        let mut updated_bias_moments = Vec::with_capacity(expert_count);
        let mut updated_scale_moments = Vec::with_capacity(expert_count);
        for expert_index in 0..expert_count {
            let cached = self.device_conv_params_at(layer_index, expert_index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached GPU MoE expert #{expert_index} params for layer #{layer_index}"
                ))
            })?;
            if cached.bn.is_some() != expert_batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "batch-normalization cache mismatch at MoE layer #{layer_index}, expert #{expert_index}"
                )));
            }
            let expert_adam_state = &mut adam_state.experts[expert_index];
            let mut targets: Vec<GpuOptimTarget<'_, B::Tensor>> = vec![
                GpuOptimTarget {
                    is_weight: true,
                    w: &cached.raw_weights,
                    u: &train.expert_weight_updates[expert_index],
                    grad: None,
                    moments: &mut expert_adam_state.weights,
                    shape: &weight_shape,
                },
                GpuOptimTarget {
                    is_weight: false,
                    w: &cached.raw_biases,
                    u: &train.expert_bias_updates[expert_index],
                    grad: None,
                    moments: &mut expert_adam_state.biases,
                    shape: &bias_shape,
                },
            ];
            if let (Some(batch_norm), Some(scale_updates)) = (
                cached.bn.as_ref(),
                train.expert_scale_updates[expert_index].as_ref(),
            ) {
                targets.push(GpuOptimTarget {
                    is_weight: false,
                    w: &batch_norm.scales,
                    u: scale_updates,
                    grad: None,
                    moments: &mut expert_adam_state.scales,
                    shape: &bias_shape,
                });
            }
            let mut updates =
                gpu_optimizer_multi_step(&self.backend, &layer_step, targets, config)?.into_iter();
            let (raw_weights, weight_momentum) = updates.next().expect("expert weights result");
            let (raw_biases, bias_momentum) = updates.next().expect("expert biases result");
            let (updated_scales, scale_momentum) = match updates.next() {
                Some((scales, momentum)) => (Some(scales), Some(momentum)),
                None => (None, None),
            };
            if expert_batch_normalize != updated_scales.is_some() {
                return Err(Error::InvalidConfig(format!(
                    "MoE BN scale update mismatch at layer #{layer_index}, expert #{expert_index}"
                )));
            }
            let mut refreshed = cached;
            refreshed.raw_weights = raw_weights.clone();
            refreshed.raw_biases = raw_biases.clone();
            if let Some(scales) = updated_scales {
                refreshed
                    .bn
                    .as_mut()
                    .expect("BN cache validated above")
                    .scales = scales;
                refreshed.folded_stale = true;
            } else {
                refreshed.weights = raw_weights;
                refreshed.biases = raw_biases;
                refreshed.folded_stale = false;
            }
            self.set_device_conv_params(layer_index, expert_index, refreshed.clone())?;
            if sync_host_weights {
                self.sync_moe_expert_params_from_device(layer_index, expert_index, &refreshed)?;
            } else {
                self.host_weights_stale = true;
            }
            updated_weight_moments.push(weight_momentum);
            updated_bias_moments.push(bias_momentum);
            updated_scale_moments.push(scale_momentum);
        }
        let Some(LayerParams::Moe(params)) = self.params.get_mut(layer_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing MoE params for layer #{layer_index}"
            )));
        };
        params.router_weights = router_weights_host;
        params.router_biases = router_biases_host;
        if matches!(step, GpuOptimizerStep::Adam(_)) {
            self.moe_adam_state[layer_index] = Some(adam_state);
            self.host_adam_state.borrow_mut()[layer_index] = None;
        }
        let mut buffers = self.device_moe_train_buffers.borrow_mut();
        let slot = buffers
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU MoE train buffers for layer #{layer_index}"
                ))
            })?;
        slot.router_weight_updates = router_weight_momentum;
        slot.router_bias_updates = router_bias_momentum;
        slot.expert_weight_updates = updated_weight_moments;
        slot.expert_bias_updates = updated_bias_moments;
        slot.expert_scale_updates = updated_scale_moments;
        slot.dirty = false;
        Ok(true)
    }

    /// Run DCNv4 SGD on native tensors while keeping host parameters canonical.
    /// Adam remains on the host optimizer path after the device-gradient flush.
    pub(super) fn update_device_dcnv4_sgd(
        &mut self,
        layer_index: usize,
        step: &GpuOptimizerStep,
        config: ElementwiseLaunchConfig,
    ) -> Result<bool> {
        let GpuOptimizerStep::Sgd {
            lr,
            momentum,
            decay,
            batch,
        } = *step
        else {
            return Ok(false);
        };
        let layer =
            self.network.layers.get(layer_index).ok_or_else(|| {
                Error::InvalidConfig(format!("missing DCNv4 layer #{layer_index}"))
            })?;
        let (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) =
            (&layer.kind, &self.params[layer_index])
        else {
            return Ok(false);
        };
        if dcnv4_gpu::needs_host(spec) {
            return Ok(false);
        }
        let desc = dcnv4_gpu::descriptor(layer, spec, params)?;
        let Some(train) = self
            .device_dcnv4_train_buffers
            .borrow()
            .get(layer_index)
            .and_then(Clone::clone)
        else {
            return Ok(false);
        };
        let bias_shape = [spec.filters];
        let offset_weight_shape = [
            desc.geometry.offset_dim(),
            desc.geometry.channels,
            spec.size,
            spec.size,
        ];
        let offset_bias_shape = [desc.geometry.offset_dim()];
        let device_params = {
            let mut caches = self.dcnv4_device_params.borrow_mut();
            dcnv4_gpu::device_parameters(
                &self.backend,
                &mut caches[layer_index],
                layer,
                spec,
                params,
            )?
        };
        let offset_weights = device_params.offset_weights.clone();
        let offset_biases = device_params.offset_biases.clone();
        let layer_lr = lr * layer.train.learning_rate_scale;
        let offset_weight_sum = self
            .backend
            .saxpy_f32(
                1.0,
                &train.offset_weight_updates,
                &train.momentum_offset_weights,
                config,
            )
            .map_err(|err| backend_error("accumulate DCNv4 offset-weight momentum", err))?;
        let offset_bias_sum = self
            .backend
            .saxpy_f32(
                1.0,
                &train.offset_bias_updates,
                &train.momentum_offset_biases,
                config,
            )
            .map_err(|err| backend_error("accumulate DCNv4 offset-bias momentum", err))?;
        let (next_offset_weights, next_offset_weight_momentum) = self
            .backend
            .sgd_update_f32(
                &offset_weights,
                &offset_weight_sum,
                layer_lr * 0.01,
                momentum,
                decay,
                batch,
                config,
            )
            .map_err(|err| backend_error("DCNv4 native SGD offset weights", err))?;
        let (next_offset_biases, next_offset_bias_momentum) = self
            .backend
            .sgd_update_f32(
                &offset_biases,
                &offset_bias_sum,
                layer_lr * 0.01,
                momentum,
                0.0,
                batch,
                config,
            )
            .map_err(|err| backend_error("DCNv4 native SGD offset biases", err))?;
        let (next_biases, next_bias_momentum, next_scales, next_scale_momentum) =
            if params.batch_norm.is_some() {
                let updates = train.scale_updates.as_ref().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing DCNv4 BN scale updates at layer #{layer_index}"
                    ))
                })?;
                let momentum_buffer = train.momentum_scale.as_ref().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing DCNv4 BN scale momentum at layer #{layer_index}"
                    ))
                })?;
                let scale = device_params.scales.as_ref().ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing DCNv4 device BN scale at layer #{layer_index}"
                    ))
                })?;
                let scale_sum = self
                    .backend
                    .saxpy_f32(1.0, updates, momentum_buffer, config)
                    .map_err(|err| backend_error("accumulate DCNv4 scale momentum", err))?;
                let (next_scale, next_scale_momentum) = self
                    .backend
                    .sgd_update_f32(scale, &scale_sum, layer_lr, momentum, 0.0, batch, config)
                    .map_err(|err| backend_error("DCNv4 native SGD BN scales", err))?;
                (None, None, Some(next_scale), Some(next_scale_momentum))
            } else {
                let bias = &device_params.biases;
                let bias_sum = self
                    .backend
                    .saxpy_f32(1.0, &train.bias_updates, &train.momentum_bias, config)
                    .map_err(|err| backend_error("accumulate DCNv4 bias momentum", err))?;
                let (next_bias, next_bias_momentum) = self
                    .backend
                    .sgd_update_f32(bias, &bias_sum, layer_lr, momentum, 0.0, batch, config)
                    .map_err(|err| backend_error("DCNv4 native SGD biases", err))?;
                (Some(next_bias), Some(next_bias_momentum), None, None)
            };

        let offset_weights_host = self
            .backend
            .download_f32(&next_offset_weights)
            .map_err(|err| backend_error("download DCNv4 SGD offset weights", err))?;
        let offset_biases_host = self
            .backend
            .download_f32(&next_offset_biases)
            .map_err(|err| backend_error("download DCNv4 SGD offset biases", err))?;
        let biases_host = next_biases
            .as_ref()
            .map(|value| self.backend.download_f32(value))
            .transpose()
            .map_err(|err| backend_error("download DCNv4 SGD biases", err))?;
        let scales_host = next_scales
            .as_ref()
            .map(|value| self.backend.download_f32(value))
            .transpose()
            .map_err(|err| backend_error("download DCNv4 SGD BN scales", err))?;

        let (LayerKind::Dcnv4(_spec), LayerParams::Dcnv4(params)) = (
            &self.network.layers[layer_index].kind,
            &mut self.params[layer_index],
        ) else {
            unreachable!("DCNv4 layer metadata was validated above")
        };
        params.offset_weights = offset_weights_host;
        params.offset_biases = offset_biases_host;
        match (&mut params.batch_norm, scales_host, biases_host) {
            (Some(bn), Some(scales), None) => bn.scales = scales,
            (None, None, Some(biases)) => params.biases = biases,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "DCNv4 SGD parameter update mismatch at layer #{layer_index}"
                )));
            }
        }
        let LayerParams::Dcnv4(updated_params) = &self.params[layer_index] else {
            unreachable!("DCNv4 parameters validated above")
        };
        let cached_biases = next_biases
            .as_ref()
            .unwrap_or(&device_params.biases)
            .clone();
        let cached_scales = next_scales
            .as_ref()
            .cloned()
            .or_else(|| device_params.scales.clone());
        dcnv4_gpu::store_device_parameters(
            &mut self.dcnv4_device_params.borrow_mut()[layer_index],
            updated_params,
            next_offset_weights.clone(),
            next_offset_biases.clone(),
            cached_biases,
            cached_scales,
        );
        let mut buffers = self.device_dcnv4_train_buffers.borrow_mut();
        let slot = buffers
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .expect("DCNv4 train buffer validated above");
        slot.momentum_offset_weights = next_offset_weight_momentum;
        slot.momentum_offset_biases = next_offset_bias_momentum;
        if let Some(momentum) = next_bias_momentum {
            slot.momentum_bias = momentum;
        }
        if let Some(momentum) = next_scale_momentum {
            slot.momentum_scale = Some(momentum);
        }
        slot.offset_weight_updates = self
            .backend
            .zeros_f32(&offset_weight_shape)
            .map_err(|err| backend_error("clear DCNv4 offset-weight gradients", err))?;
        slot.offset_bias_updates = self
            .backend
            .zeros_f32(&offset_bias_shape)
            .map_err(|err| backend_error("clear DCNv4 offset-bias gradients", err))?;
        slot.bias_updates = self
            .backend
            .zeros_f32(&bias_shape)
            .map_err(|err| backend_error("clear DCNv4 bias gradients", err))?;
        if slot.scale_updates.is_some() {
            slot.scale_updates = Some(
                self.backend
                    .zeros_f32(&bias_shape)
                    .map_err(|err| backend_error("clear DCNv4 scale gradients", err))?,
            );
        }
        slot.dirty = false;
        Ok(true)
    }

    /// Apply DCNv4 Adam using native tensors and retain moments on-device.
    /// Parameters remain host-canonical; sidecar saves materialize moments.
    pub(super) fn update_device_dcnv4_adam(
        &mut self,
        layer_index: usize,
        step: &GpuOptimizerStep,
        config: ElementwiseLaunchConfig,
    ) -> Result<bool> {
        let GpuOptimizerStep::Adam(adam) = *step else {
            return Ok(false);
        };
        let layer =
            self.network.layers.get(layer_index).ok_or_else(|| {
                Error::InvalidConfig(format!("missing DCNv4 layer #{layer_index}"))
            })?;
        let (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) =
            (&layer.kind, &self.params[layer_index])
        else {
            return Ok(false);
        };
        if dcnv4_gpu::needs_host(spec) {
            return Ok(false);
        }
        let desc = dcnv4_gpu::descriptor(layer, spec, params)?;
        let Some(train) = self
            .device_dcnv4_train_buffers
            .borrow()
            .get(layer_index)
            .and_then(Clone::clone)
        else {
            return Ok(false);
        };
        let bias_shape = [spec.filters];
        let offset_weight_shape = [
            desc.geometry.offset_dim(),
            desc.geometry.channels,
            spec.size,
            spec.size,
        ];
        let offset_bias_shape = [desc.geometry.offset_dim()];
        let host_state = self
            .host_adam_state
            .borrow()
            .get(layer_index)
            .cloned()
            .flatten()
            .unwrap_or_default();
        let device_state = self.dcnv4_adam_state[layer_index].clone();
        let offset_bias_state = host_state
            .children
            .first()
            .and_then(|child| child.biases.clone());
        let upload_pair = |pair: &Option<(Vec<f32>, Vec<f32>)>, shape: &[usize]| {
            let Some((m, v)) = pair else {
                return Ok(None);
            };
            let expected = shape.iter().product::<usize>();
            if m.len() != expected || v.len() != expected {
                return Err(Error::InvalidConfig(format!(
                    "DCNv4 Adam moment shape mismatch at layer #{layer_index}"
                )));
            }
            Ok(Some((
                self.backend
                    .upload_f32(shape, m)
                    .map_err(|err| backend_error("upload DCNv4 Adam first moment", err))?,
                self.backend
                    .upload_f32(shape, v)
                    .map_err(|err| backend_error("upload DCNv4 Adam second moment", err))?,
            )))
        };
        let (
            mut offset_weight_moments,
            mut offset_bias_moments,
            mut bias_moments,
            mut scale_moments,
        ) = if let Some(state) = device_state {
            (
                state.offset_weights,
                state.offset_biases,
                state.biases,
                state.scales,
            )
        } else {
            (
                upload_pair(&host_state.weights, &offset_weight_shape)?,
                upload_pair(&offset_bias_state, &offset_bias_shape)?,
                upload_pair(&host_state.biases, &bias_shape)?,
                upload_pair(&host_state.scales, &bias_shape)?,
            )
        };
        let offset_step = GpuOptimizerStep::Adam(AdamParams {
            lr: adam.lr * layer.train.learning_rate_scale * 0.01,
            ..adam
        });
        let layer_step = GpuOptimizerStep::Adam(AdamParams {
            lr: adam.lr * layer.train.learning_rate_scale,
            ..adam
        });
        let device_params = {
            let mut caches = self.dcnv4_device_params.borrow_mut();
            dcnv4_gpu::device_parameters(
                &self.backend,
                &mut caches[layer_index],
                layer,
                spec,
                params,
            )?
        };
        let offset_weights = device_params.offset_weights.clone();
        let offset_biases = device_params.offset_biases.clone();
        let offset_targets = vec![
            GpuOptimTarget {
                is_weight: true,
                w: &offset_weights,
                u: &train.offset_weight_updates,
                grad: None,
                moments: &mut offset_weight_moments,
                shape: &offset_weight_shape,
            },
            GpuOptimTarget {
                is_weight: false,
                w: &offset_biases,
                u: &train.offset_bias_updates,
                grad: None,
                moments: &mut offset_bias_moments,
                shape: &offset_bias_shape,
            },
        ];
        let mut offset_results =
            gpu_optimizer_multi_step(&self.backend, &offset_step, offset_targets, config)?
                .into_iter();
        let (next_offset_weights, _) = offset_results.next().expect("offset weights result");
        let (next_offset_biases, _) = offset_results.next().expect("offset biases result");
        let (next_biases, next_scales) = if params.batch_norm.is_some() {
            let updates = train.scale_updates.as_ref().ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing DCNv4 Adam BN scale updates at layer #{layer_index}"
                ))
            })?;
            let scales = device_params.scales.as_ref().ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing DCNv4 device BN scale at layer #{layer_index}"
                ))
            })?;
            let target = GpuOptimTarget {
                is_weight: false,
                w: scales,
                u: updates,
                grad: None,
                moments: &mut scale_moments,
                shape: &bias_shape,
            };
            let mut results =
                gpu_optimizer_multi_step(&self.backend, &layer_step, vec![target], config)?
                    .into_iter();
            (None, Some(results.next().expect("BN scale result").0))
        } else {
            let biases = &device_params.biases;
            let target = GpuOptimTarget {
                is_weight: false,
                w: biases,
                u: &train.bias_updates,
                grad: None,
                moments: &mut bias_moments,
                shape: &bias_shape,
            };
            let mut results =
                gpu_optimizer_multi_step(&self.backend, &layer_step, vec![target], config)?
                    .into_iter();
            (Some(results.next().expect("bias result").0), None)
        };
        let download = |tensor: &B::Tensor, what| {
            self.backend
                .download_f32(tensor)
                .map_err(|err| backend_error(what, err))
        };
        let offset_weights_host =
            download(&next_offset_weights, "download DCNv4 Adam offset weights")?;
        let offset_biases_host =
            download(&next_offset_biases, "download DCNv4 Adam offset biases")?;
        let biases_host = next_biases
            .as_ref()
            .map(|value| download(value, "download DCNv4 Adam biases"))
            .transpose()?;
        let scales_host = next_scales
            .as_ref()
            .map(|value| download(value, "download DCNv4 Adam BN scales"))
            .transpose()?;
        let LayerParams::Dcnv4(params) = &mut self.params[layer_index] else {
            unreachable!("DCNv4 parameter metadata was validated above")
        };
        params.offset_weights = offset_weights_host;
        params.offset_biases = offset_biases_host;
        match (&mut params.batch_norm, scales_host, biases_host) {
            (Some(bn), Some(scales), None) => bn.scales = scales,
            (None, None, Some(biases)) => params.biases = biases,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "DCNv4 Adam parameter update mismatch at layer #{layer_index}"
                )));
            }
        }
        let LayerParams::Dcnv4(updated_params) = &self.params[layer_index] else {
            unreachable!("DCNv4 parameters validated above")
        };
        let cached_biases = next_biases
            .as_ref()
            .unwrap_or(&device_params.biases)
            .clone();
        let cached_scales = next_scales
            .as_ref()
            .cloned()
            .or_else(|| device_params.scales.clone());
        dcnv4_gpu::store_device_parameters(
            &mut self.dcnv4_device_params.borrow_mut()[layer_index],
            updated_params,
            next_offset_weights.clone(),
            next_offset_biases.clone(),
            cached_biases,
            cached_scales,
        );
        self.dcnv4_adam_state[layer_index] = Some(GpuDcnv4AdamState {
            offset_weights: offset_weight_moments,
            offset_biases: offset_bias_moments,
            biases: bias_moments,
            scales: scale_moments,
        });
        self.host_adam_state.borrow_mut()[layer_index] = None;
        let mut buffers = self.device_dcnv4_train_buffers.borrow_mut();
        let slot = buffers
            .get_mut(layer_index)
            .and_then(Option::as_mut)
            .expect("DCNv4 train buffer validated above");
        slot.offset_weight_updates = self
            .backend
            .zeros_f32(&offset_weight_shape)
            .map_err(|err| backend_error("clear DCNv4 Adam offset-weight gradients", err))?;
        slot.offset_bias_updates = self
            .backend
            .zeros_f32(&offset_bias_shape)
            .map_err(|err| backend_error("clear DCNv4 Adam offset-bias gradients", err))?;
        slot.bias_updates = self
            .backend
            .zeros_f32(&bias_shape)
            .map_err(|err| backend_error("clear DCNv4 Adam bias gradients", err))?;
        if slot.scale_updates.is_some() {
            slot.scale_updates = Some(
                self.backend
                    .zeros_f32(&bias_shape)
                    .map_err(|err| backend_error("clear DCNv4 Adam scale gradients", err))?,
            );
        }
        slot.dirty = false;
        Ok(true)
    }

    pub(super) fn flush_device_dcnv4_gradients_to_host(&mut self, only_layer: Option<usize>) -> Result<()> {
        type HostGradients = (Vec<f32>, Vec<f32>, Vec<f32>, Option<Vec<f32>>);
        let mut staged = Vec::with_capacity(self.network.layers.len());
        let mut reset = Vec::with_capacity(self.network.layers.len());
        {
            let buffers = self.device_dcnv4_train_buffers.borrow();
            for (layer_index, entry) in buffers.iter().enumerate() {
                let Some(entry) = entry else {
                    staged.push(None);
                    reset.push(None);
                    continue;
                };
                if only_layer.is_some_and(|target| target != layer_index) {
                    staged.push(None);
                    reset.push(Some(entry.clone()));
                    continue;
                }
                if !entry.dirty {
                    staged.push(None);
                    reset.push(Some(entry.clone()));
                    continue;
                }
                let gradients: HostGradients = (
                    self.backend
                        .download_f32(&entry.bias_updates)
                        .map_err(|err| backend_error("download DCNv4 bias updates", err))?,
                    self.backend
                        .download_f32(&entry.offset_weight_updates)
                        .map_err(|err| {
                            backend_error("download DCNv4 offset weight updates", err)
                        })?,
                    self.backend
                        .download_f32(&entry.offset_bias_updates)
                        .map_err(|err| backend_error("download DCNv4 offset bias updates", err))?,
                    entry
                        .scale_updates
                        .as_ref()
                        .map(|scales| {
                            self.backend
                                .download_f32(scales)
                                .map_err(|err| backend_error("download DCNv4 scale updates", err))
                        })
                        .transpose()?,
                );
                let (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) = (
                    &self.network.layers[layer_index].kind,
                    &self.params[layer_index],
                ) else {
                    return Err(Error::InvalidConfig(format!(
                        "missing DCNv4 layer/params for GPU gradients at layer #{layer_index}"
                    )));
                };
                let desc = dcnv4_gpu::descriptor(&self.network.layers[layer_index], spec, params)?;
                let bias_shape = [spec.filters];
                let offset_weight_shape = [
                    desc.geometry.offset_dim(),
                    desc.geometry.channels,
                    spec.size,
                    spec.size,
                ];
                let offset_bias_shape = [desc.geometry.offset_dim()];
                let zeros = GpuDcnv4TrainBuffers {
                    bias_updates: self
                        .backend
                        .zeros_f32(&bias_shape)
                        .map_err(|err| backend_error("reset DCNv4 bias updates", err))?,
                    offset_weight_updates: self
                        .backend
                        .zeros_f32(&offset_weight_shape)
                        .map_err(|err| backend_error("reset DCNv4 offset weight updates", err))?,
                    offset_bias_updates: self
                        .backend
                        .zeros_f32(&offset_bias_shape)
                        .map_err(|err| backend_error("reset DCNv4 offset bias updates", err))?,
                    scale_updates: params
                        .batch_norm
                        .as_ref()
                        .map(|_| {
                            self.backend
                                .zeros_f32(&bias_shape)
                                .map_err(|err| backend_error("reset DCNv4 scale updates", err))
                        })
                        .transpose()?,
                    momentum_bias: self
                        .backend
                        .zeros_f32(&bias_shape)
                        .map_err(|err| backend_error("reset DCNv4 bias momentum", err))?,
                    momentum_offset_weights: self
                        .backend
                        .zeros_f32(&offset_weight_shape)
                        .map_err(|err| backend_error("reset DCNv4 offset weight momentum", err))?,
                    momentum_offset_biases: self
                        .backend
                        .zeros_f32(&offset_bias_shape)
                        .map_err(|err| backend_error("reset DCNv4 offset bias momentum", err))?,
                    momentum_scale: params
                        .batch_norm
                        .as_ref()
                        .map(|_| {
                            self.backend
                                .zeros_f32(&bias_shape)
                                .map_err(|err| backend_error("reset DCNv4 scale momentum", err))
                        })
                        .transpose()?,
                    dirty: false,
                };
                staged.push(Some(gradients));
                reset.push(Some(zeros));
            }
        }
        {
            let buffers = self.cpu_fallback_train_buffers.borrow();
            for (layer_index, gradients) in staged.iter().enumerate() {
                let Some((biases, weights, offset_biases, scales)) = gradients else {
                    continue;
                };
                let Some(LayerTrainBuffers::Dcnv4(train)) = buffers.get(layer_index) else {
                    return Err(Error::InvalidConfig(format!(
                        "missing host DCNv4 train buffers for layer #{layer_index}"
                    )));
                };
                if train.bias_updates.len() != biases.len()
                    || train.offset_weight_updates.len() != weights.len()
                    || train.offset_bias_updates.len() != offset_biases.len()
                    || match (&train.batch_norm, scales) {
                        (Some(bn), Some(scales)) => bn.scale_updates.len() != scales.len(),
                        (None, None) => false,
                        _ => true,
                    }
                {
                    return Err(Error::InvalidConfig(format!(
                        "DCNv4 host train-buffer shape mismatch for layer #{layer_index}"
                    )));
                }
            }
        }
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            for (layer_index, gradients) in staged.iter().enumerate() {
                let Some((biases, weights, offset_biases, scales)) = gradients else {
                    continue;
                };
                let LayerTrainBuffers::Dcnv4(train) = &mut buffers[layer_index] else {
                    unreachable!("validated DCNv4 host train buffers")
                };
                for (dst, src) in train.bias_updates.iter_mut().zip(biases) {
                    *dst += src;
                }
                for (dst, src) in train.offset_weight_updates.iter_mut().zip(weights) {
                    *dst += src;
                }
                for (dst, src) in train.offset_bias_updates.iter_mut().zip(offset_biases) {
                    *dst += src;
                }
                if let (Some(bn), Some(scales)) = (&mut train.batch_norm, scales) {
                    for (dst, src) in bn.scale_updates.iter_mut().zip(scales) {
                        *dst += src;
                    }
                }
            }
        }
        *self.device_dcnv4_train_buffers.borrow_mut() = reset;
        Ok(())
    }

    pub(super) fn flush_device_moe_gradients_to_host(&mut self, keep_device_gradients: &[bool]) -> Result<()> {
        type ExpertUpdates = (Vec<f32>, Vec<f32>, Option<Vec<f32>>);
        type HostUpdates = (Vec<f32>, Vec<f32>, Vec<ExpertUpdates>);
        let mut staged: Vec<Option<HostUpdates>> = Vec::new();
        {
            let buffers = self.device_moe_train_buffers.borrow();
            staged.reserve(buffers.len());
            for (layer_index, maybe_train) in buffers.iter().enumerate() {
                if keep_device_gradients
                    .get(layer_index)
                    .copied()
                    .unwrap_or(false)
                {
                    staged.push(None);
                    continue;
                }
                let Some(train) = maybe_train.as_ref().filter(|train| train.dirty) else {
                    staged.push(None);
                    continue;
                };
                if train.expert_weight_updates.len() != train.expert_bias_updates.len()
                    || train.expert_weight_updates.len() != train.expert_scale_updates.len()
                {
                    return Err(Error::InvalidConfig(format!(
                        "MoE GPU update count mismatch for layer #{layer_index}"
                    )));
                }
                let router_weights = self
                    .backend
                    .download_f32(&train.router_weight_updates)
                    .map_err(|err| backend_error("download MoE router weight updates", err))?;
                let router_biases = self
                    .backend
                    .download_f32(&train.router_bias_updates)
                    .map_err(|err| backend_error("download MoE router bias updates", err))?;
                let mut expert_updates = Vec::with_capacity(train.expert_weight_updates.len());
                for ((weights, biases), scales) in train
                    .expert_weight_updates
                    .iter()
                    .zip(&train.expert_bias_updates)
                    .zip(&train.expert_scale_updates)
                {
                    expert_updates.push((
                        self.backend.download_f32(weights).map_err(|err| {
                            backend_error("download MoE expert weight updates", err)
                        })?,
                        self.backend.download_f32(biases).map_err(|err| {
                            backend_error("download MoE expert bias updates", err)
                        })?,
                        scales
                            .as_ref()
                            .map(|scales| {
                                self.backend.download_f32(scales).map_err(|err| {
                                    backend_error("download MoE expert scale updates", err)
                                })
                            })
                            .transpose()?,
                    ));
                }
                staged.push(Some((router_weights, router_biases, expert_updates)));
            }
        }

        {
            let buffers = self.cpu_fallback_train_buffers.borrow();
            for (layer_index, gradients) in staged.iter().enumerate() {
                let Some((router_weights, router_biases, experts)) = gradients else {
                    continue;
                };
                let Some(LayerTrainBuffers::Moe(train)) = buffers.get(layer_index) else {
                    return Err(Error::InvalidConfig(format!(
                        "missing host MoE train buffers for layer #{layer_index}"
                    )));
                };
                if train.router_weight_updates.len() != router_weights.len()
                    || train.router_bias_updates.len() != router_biases.len()
                    || train.experts.len() != experts.len()
                {
                    return Err(Error::InvalidConfig(format!(
                        "MoE host train-buffer shape mismatch for layer #{layer_index}"
                    )));
                }
                for (host, (weights, biases, scales)) in train.experts.iter().zip(experts) {
                    if host.weight_updates.len() != weights.len()
                        || host.bias_updates.len() != biases.len()
                    {
                        return Err(Error::InvalidConfig(format!(
                            "MoE host expert train-buffer shape mismatch for layer #{layer_index}"
                        )));
                    }
                    match (&host.batch_norm, scales) {
                        (Some(batch_norm), Some(scales))
                            if batch_norm.scale_updates.len() == scales.len() => {}
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "MoE host expert scale-buffer mismatch for layer #{layer_index}"
                            )));
                        }
                    }
                }
            }
        }
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            for (layer_index, gradients) in staged.iter().enumerate() {
                let Some((router_weights, router_biases, experts)) = gradients else {
                    continue;
                };
                let Some(LayerTrainBuffers::Moe(train)) = buffers.get_mut(layer_index) else {
                    return Err(Error::InvalidConfig(format!(
                        "missing host MoE train buffers for layer #{layer_index}"
                    )));
                };
                accumulate_vec(&mut train.router_weight_updates, router_weights)?;
                accumulate_vec(&mut train.router_bias_updates, router_biases)?;
                for (host, (weights, biases, scales)) in train.experts.iter_mut().zip(experts) {
                    accumulate_vec(&mut host.weight_updates, weights)?;
                    accumulate_vec(&mut host.bias_updates, biases)?;
                    if let (Some(batch_norm), Some(scales)) = (&mut host.batch_norm, scales) {
                        accumulate_vec(&mut batch_norm.scale_updates, scales)?;
                    }
                }
            }
        }
        let mut buffers = self.device_moe_train_buffers.borrow_mut();
        for (layer_index, (train, gradients)) in buffers.iter_mut().zip(staged).enumerate() {
            if train.is_none() || gradients.is_none() {
                continue;
            }
            let Some(LayerKind::Moe(spec)) = self.network.layers.get(layer_index).map(|l| &l.kind)
            else {
                return Err(Error::InvalidConfig(format!(
                    "device MoE buffers belong to non-MoE layer #{layer_index}"
                )));
            };
            let train = train.as_mut().expect("checked above");
            let grouped_channels =
                self.network.layers[layer_index].input_shape.channels / spec.expert.groups.max(1);
            train.router_weight_updates = self
                .backend
                .zeros_f32(&[
                    spec.experts,
                    self.network.layers[layer_index].input_shape.channels,
                ])
                .map_err(|err| backend_error("reset MoE router weight gradients", err))?;
            train.router_bias_updates = self
                .backend
                .zeros_f32(&[spec.experts])
                .map_err(|err| backend_error("reset MoE router bias gradients", err))?;
            for expert_index in 0..spec.experts {
                train.expert_weight_updates[expert_index] = self
                    .backend
                    .zeros_f32(&[
                        spec.expert.filters,
                        grouped_channels,
                        spec.expert.size,
                        spec.expert.size,
                    ])
                    .map_err(|err| backend_error("reset MoE expert weight gradients", err))?;
                train.expert_bias_updates[expert_index] = self
                    .backend
                    .zeros_f32(&[spec.expert.filters])
                    .map_err(|err| backend_error("reset MoE expert bias gradients", err))?;
                train.expert_scale_updates[expert_index] = spec
                    .expert
                    .batch_normalize
                    .then(|| {
                        self.backend
                            .zeros_f32(&[spec.expert.filters])
                            .map_err(|err| backend_error("reset MoE expert scale gradients", err))
                    })
                    .transpose()?;
            }
            train.dirty = false;
        }
        Ok(())
    }

    /// `updates + pending` for every tensor (the accumulate launches the fused
    /// optimizer step skips).
    #[allow(clippy::type_complexity)]
    pub(super) fn flush_pending_grads(
        &self,
        train: &GpuConvolutionTrainBuffers<B::Tensor>,
        pending: &GpuPendingGrads<B::Tensor>,
    ) -> Result<(B::Tensor, B::Tensor, Option<B::Tensor>)> {
        let config = ElementwiseLaunchConfig::default();
        let add = |grad: &B::Tensor, acc: &B::Tensor| {
            self.backend
                .saxpy_f32(1.0, grad, acc, config)
                .map_err(|err| backend_error("GPU gradient accumulation", err))
        };
        let scales = match (&pending.scales, &train.scale_updates) {
            (Some(g), Some(acc)) => Some(add(g, acc)?),
            (None, acc) => acc.clone(),
            (Some(_), None) => {
                return Err(Error::InvalidConfig(
                    "pending batch-norm scale gradient for a layer without scale updates"
                        .to_owned(),
                ));
            }
        };
        Ok((
            add(&pending.weights, &train.weight_updates)?,
            add(&pending.biases, &train.bias_updates)?,
            scales,
        ))
    }

}
