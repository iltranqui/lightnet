//! Device-resident Muon step for standard convolutions.
//!
//! Weight matrices (one per group) take the Muon momentum, Newton-Schulz and
//! decoupled-decay update entirely on the device. Biases and batch-norm scales
//! take AdamW through the same multi-tensor optimizer the SGD/Adam path uses.
//! No gradient or weight is downloaded; the momentum mirror is only
//! materialized when a checkpoint is written
//! ([`GpuExecutableNetwork::materialize_device_conv_muon_states`]).
use super::*;
use crate::muon::MUON_SCALE;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// True when layer `index` is an owning standard convolution whose device
    /// parameters and gradients exist, so [`Self::update_device_conv_muon`]
    /// can run without the host-staged Muon path.
    pub(super) fn device_conv_muon_eligible(&self, index: usize) -> bool {
        let Some(layer) = self.network.layers.get(index) else {
            return false;
        };
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return false;
        };
        spec.share_index.is_none()
            && matches!(self.params.get(index), Some(LayerParams::Convolutional(_)))
            && self.device_conv_params_at(index, 0).is_some()
            && self
                .device_train_buffers
                .borrow()
                .get(index)
                .is_some_and(Option::is_some)
            && !self.requires_cpu_backward_fallback()
    }

    /// One Muon step for convolution `index`. Returns `Ok(false)` without
    /// touching anything when the layer is not eligible.
    pub(super) fn update_device_conv_muon(
        &mut self,
        index: usize,
        adam: &AdamParams,
        loss_scale: f32,
        sync_host_weights: bool,
    ) -> Result<bool> {
        if !self.device_conv_muon_eligible(index) {
            return Ok(false);
        }
        let config = ElementwiseLaunchConfig::default();
        let layer = self.network.layers[index].clone();
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Ok(false);
        };
        let opts = layer.train;
        let groups = spec.groups.max(1);
        let in_per_group = layer.input_shape.channels / groups;
        if spec.filters == 0 || layer.input_shape.channels % groups != 0 || spec.filters % groups != 0 {
            return Err(Error::InvalidConfig(
                "Muon convolution has invalid grouped-channel geometry".to_owned(),
            ));
        }
        let rows = spec.filters / groups;
        let cols = in_per_group * spec.size * spec.size;
        let group_len = rows * cols;
        let total = groups * group_len;
        let weight_shape = [spec.filters, in_per_group, spec.size, spec.size];
        let filters_shape = [spec.filters];
        let layer_adam = AdamParams {
            lr: adam.lr * opts.learning_rate_scale,
            ..*adam
        };

        let cached = self.device_conv_params_at(index, 0).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing cached GPU convolution params for layer #{index}"
            ))
        })?;
        let train = self
            .device_train_buffers
            .borrow()
            .get(index)
            .and_then(Clone::clone)
            .ok_or_else(|| {
                Error::InvalidConfig(format!("missing cached GPU train buffers for layer #{index}"))
            })?;
        let pending = train.pending.clone();
        let has_bn = train.scale_updates.is_some();

        // Gradient: accumulated + pending, NaN/Inf reset, optional centralization.
        let mut weight_grad = train.weight_updates.clone();
        if let Some(p) = pending.as_ref() {
            weight_grad = self
                .backend
                .saxpy_f32(1.0, &p.weights, &weight_grad, config)
                .map_err(|err| backend_error("GPU Muon gradient accumulation", err))?;
        }
        weight_grad = self
            .backend
            .sanitize_nan_inf_f32(&weight_grad, &weight_shape, NanInfMode::Reset, config)
            .map_err(|err| backend_error("GPU Muon reset_nan_and_inf", err))?;
        if opts.grad_centr && has_bn {
            weight_grad = self
                .backend
                .gradient_centralization_f32(&weight_grad, &weight_shape, cols, config)
                .map_err(|err| backend_error("GPU Muon gradient centralization", err))?;
        }

        // Momentum: lazily seeded from the host mirror (checkpoint resume).
        let layer_count = self.network.layers.len();
        let momentum = {
            let mut device_state = self.device_muon_momentum.borrow_mut();
            if device_state.len() != layer_count {
                *device_state = vec![None; layer_count];
            }
            if device_state[index].is_none() {
                let mut flat = vec![0.0f32; total];
                if let Some(state) = self.muon_state.get(index).and_then(Option::as_ref)
                    && !state.matrices.is_empty()
                {
                    if state.matrices.len() != groups
                        || state
                            .matrices
                            .iter()
                            .any(|m| m.rows != rows || m.cols != cols || m.momentum.len() != group_len)
                    {
                        return Err(Error::InvalidConfig(format!(
                            "Muon state for convolution layer #{index} does not match {groups} group(s) of {rows}x{cols}"
                        )));
                    }
                    for (group, matrix) in state.matrices.iter().enumerate() {
                        flat[group * group_len..(group + 1) * group_len]
                            .copy_from_slice(&matrix.momentum);
                    }
                }
                device_state[index] = Some(
                    self.backend
                        .upload_f32(&[total], &flat)
                        .map_err(|err| backend_error("GPU Muon momentum upload", err))?,
                );
            }
            device_state[index].clone().expect("initialized GPU Muon momentum")
        };

        // Per-group Muon step. Groups are contiguous slabs of the flat tensor.
        let slice = |tensor: &B::Tensor, group: usize| -> Result<B::Tensor> {
            if groups == 1 {
                return Ok(tensor.clone());
            }
            self.backend
                .route_concat_f32(
                    &[tensor],
                    &[group * group_len],
                    &[group_len],
                    &[total],
                    &[1, group_len, 1, 1],
                    1,
                    config,
                )
                .map_err(|err| backend_error("GPU Muon group slice", err))
        };
        let direction_scale = MUON_SCALE * (rows.max(cols) as f32).sqrt();
        let matrix_shape = [rows, cols];
        let mut new_weights = Vec::with_capacity(groups);
        let mut new_momentum = Vec::with_capacity(groups);
        for group in 0..groups {
            let updates = slice(&weight_grad, group)?;
            let prior = slice(&momentum, group)?;
            let weights = slice(&cached.raw_weights, group)?;
            let (next_momentum, lookahead) = self
                .backend
                .muon_momentum_lookahead_f32(
                    &updates,
                    &prior,
                    &matrix_shape,
                    adam.batch,
                    loss_scale,
                    self.network.net.muon_momentum,
                )
                .map_err(|err| backend_error("GPU Muon momentum/lookahead", err))?;
            if self.backend.any_non_finite_f32(&next_momentum).map_err(|e| backend_error("GPU Muon finiteness check", e))?
                || self.backend.any_non_finite_f32(&lookahead).map_err(|e| backend_error("GPU Muon finiteness check", e))?
            {
                return Err(Error::InvalidConfig(
                    "GPU Muon momentum update became non-finite".to_owned(),
                ));
            }
            let direction = self
                .backend
                .muon_orthogonalize_f32(&lookahead, &matrix_shape, rows, cols)
                .map_err(|err| backend_error("GPU Muon Newton-Schulz", err))?;
            let updated = self
                .backend
                .muon_apply_update_f32(
                    &weights,
                    &direction,
                    &matrix_shape,
                    layer_adam.lr,
                    layer_adam.decay,
                    direction_scale,
                )
                .map_err(|err| backend_error("GPU Muon parameter update", err))?;
            if self.backend.any_non_finite_f32(&updated).map_err(|e| backend_error("GPU Muon finiteness check", e))? {
                return Err(Error::InvalidConfig(
                    "GPU Muon parameter update became non-finite".to_owned(),
                ));
            }
            new_weights.push(updated);
            new_momentum.push(next_momentum);
        }
        let join = |parts: &[B::Tensor]| -> Result<B::Tensor> {
            if parts.len() == 1 {
                return Ok(parts[0].clone());
            }
            let refs: Vec<&B::Tensor> = parts.iter().collect();
            let zeros = vec![0usize; parts.len()];
            let lens = vec![group_len; parts.len()];
            self.backend
                .route_concat_f32(&refs, &zeros, &lens, &lens, &[1, total, 1, 1], 1, config)
                .map_err(|err| backend_error("GPU Muon group concat", err))
        };
        let raw_weights = join(&new_weights)?;
        let next_momentum = join(&new_momentum)?;

        // Biases and BN scales: AdamW (no decay on non-matrix tensors).
        if self.adam_state.len() != layer_count {
            self.adam_state = (0..layer_count).map(|_| None).collect();
        }
        let mut moments = self.adam_state[index].take().unwrap_or_default();
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
        let mut targets = vec![GpuOptimTarget {
            is_weight: false,
            w: &cached.raw_biases,
            u: &train.bias_updates,
            grad: pending.as_ref().map(|p| &p.biases),
            moments: &mut moments.biases,
            shape: &filters_shape,
        }];
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
        let mut stepped = gpu_optimizer_multi_step(
            &self.backend,
            &GpuOptimizerStep::Adam(layer_adam),
            targets,
            config,
        )?
        .into_iter();
        let (raw_biases, bias_updates) = stepped.next().expect("bias result");
        let (new_scales, scale_updates) = match stepped.next() {
            Some((w, u)) => (Some(w), Some(u)),
            None => (None, None),
        };
        self.adam_state[index] = Some(moments);

        let weight_updates = self
            .backend
            .zeros_f32(&weight_shape)
            .map_err(|err| backend_error("zero convolution Muon gradients", err))?;
        let refreshed = self.refresh_device_convolution_params(
            index,
            raw_weights,
            raw_biases,
            new_scales,
            weight_updates,
            bias_updates,
            scale_updates,
        )?;
        self.device_muon_momentum.borrow_mut()[index] = Some(next_momentum);
        if sync_host_weights {
            self.sync_convolution_params_from_device(index, &refreshed)?;
        } else {
            self.host_weights_stale = true;
        }
        Ok(true)
    }

    /// Copy device convolution Muon momentum into `states` (checkpoint write).
    /// Only layers served by [`Self::update_device_conv_muon`] are touched;
    /// other families keep their host mirror.
    pub(super) fn materialize_device_conv_muon_states(
        &self,
        states: &mut [Option<MuonLayerState>],
    ) -> Result<()> {
        let device_state = self.device_muon_momentum.borrow();
        for (index, layer) in self.network.layers.iter().enumerate() {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if spec.share_index.is_some() || !self.device_conv_muon_eligible(index) {
                continue;
            }
            let Some(Some(tensor)) = device_state.get(index) else {
                continue;
            };
            let groups = spec.groups.max(1);
            let rows = spec.filters / groups;
            let cols = layer.input_shape.channels / groups * spec.size * spec.size;
            let group_len = rows * cols;
            let flat = self
                .backend
                .download_f32(tensor)
                .map_err(|err| backend_error("materialize convolution Muon momentum", err))?;
            if flat.len() != groups * group_len {
                return Err(Error::InvalidConfig(format!(
                    "device Muon momentum for convolution layer #{index} has {} values, expected {}",
                    flat.len(),
                    groups * group_len
                )));
            }
            let matrices = flat
                .chunks_exact(group_len)
                .map(|chunk| MuonMatrixState {
                    rows,
                    cols,
                    momentum: chunk.to_vec(),
                })
                .collect();
            *states.get_mut(index).ok_or_else(|| {
                Error::InvalidConfig("convolution Muon state index out of range".to_owned())
            })? = Some(MuonLayerState { matrices });
        }
        Ok(())
    }
}
