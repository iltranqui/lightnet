use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn new(network: Network, backend: B) -> Self {
        let params = network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        let device_conv_params =
            RefCell::new((0..network.layers.len()).map(|_| Vec::new()).collect());
        let f16_conv_weights = RefCell::new((0..network.layers.len()).map(|_| None).collect());
        let device_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        let device_moe_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        let device_eml_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        let device_dcnv4_train_buffers = RefCell::new(vec![None; network.layers.len()]);
        let dcnv4_device_params = RefCell::new(vec![None; network.layers.len()]);
        let cpu_fallback_train_buffers = RefCell::new(
            network
                .layers
                .iter()
                .map(default_layer_train_buffers)
                .collect(),
        );
        let moe_adam_state = (0..network.layers.len()).map(|_| None).collect();
        let eml_adam_state = (0..network.layers.len()).map(|_| None).collect();
        let dcnv4_adam_state = (0..network.layers.len()).map(|_| None).collect();
        let host_adam_state = RefCell::new((0..network.layers.len()).map(|_| None).collect());
        let muon_state = (0..network.layers.len()).map(|_| None).collect();
        let device_muon_momentum = RefCell::new(vec![None; network.layers.len()]);
        Self {
            backend,
            network,
            params,
            device_conv_params,
            f16_conv_weights,
            device_train_buffers,
            train_buffers_initialized: false,
            device_moe_train_buffers,
            device_eml_train_buffers,
            device_dcnv4_train_buffers,
            dcnv4_device_params,
            cpu_fallback_train_buffers,
            resident_parameter_banks: RefCell::new(Vec::new()),
            clifford_device_caches: RefCell::new(Vec::new()),
            capture_input_gradient: Cell::new(false),
            captured_input_gradient: RefCell::new(None),
            deform_device_caches: RefCell::new(Vec::new()),
            vit_device_caches: RefCell::new(Vec::new()),
            wmhf_device_caches: RefCell::new(Vec::new()),
            swin_device_caches: RefCell::new(Vec::new()),
            tucker_attention_device_caches: RefCell::new(Vec::new()),
            mambavision_device_caches: RefCell::new(Vec::new()),
            kimi_kda_device_caches: RefCell::new(Vec::new()),
            dpt_head_device_caches: RefCell::new(Vec::new()),
            depth_anything_device_caches: RefCell::new(Vec::new()),
            recursive_block_device_caches: RefCell::new(Vec::new()),
            adam_state: Vec::new(),
            moe_adam_state,
            eml_adam_state,
            dcnv4_adam_state,
            host_adam_state,
            muon_state,
            device_muon_momentum,
            adam_step: 0,
            train_iteration: 0,
            host_weights_stale: false,
            defer_host_sync: false,
        }
    }

    /// See [`ExecutableNetwork::set_train_iteration`].
    pub fn set_train_iteration(&mut self, iteration: usize) {
        self.train_iteration = iteration;
    }

    /// Inspect host-fallback layer train buffers. Device-resident gradients
    /// (for example native convolution buffers) are intentionally not included.
    /// The borrow reflects buffers after host fallback work and any explicit
    /// device-to-host gradient flushes.
    pub fn host_fallback_train_buffers(&self) -> std::cell::Ref<'_, [LayerTrainBuffers]> {
        std::cell::Ref::map(self.cpu_fallback_train_buffers.borrow(), |buffers| {
            buffers.as_slice()
        })
    }

    /// Adam updates applied so far (the `t` of the next update is this + 1).
    pub fn adam_step(&self) -> u32 {
        self.adam_step
    }

    /// Restores the Adam step counter (e.g. when resuming); moments are not
    /// part of the weights file and restart from zero.
    pub fn set_adam_step(&mut self, step: u32) {
        self.adam_step = step;
    }

    /// Loss scale applied to the yolo seed gradients before backward.
    pub fn loss_scale(&self) -> f32 {
        self.network.net.loss_scale
    }

    /// Update the loss scale (dynamic loss scaling).  Values are applied to the
    /// next backward and divided back out of the next optimizer step.
    pub fn set_loss_scale(&mut self, scale: f32) {
        if scale.is_finite() && scale > 0.0 {
            self.network.net.loss_scale = scale;
        }
    }

    /// True if any device convolution/MoE gradient or supported host gradient
    /// buffer is non-finite, for dynamic loss scaling. Other host-only layer
    /// families are not inspected.
    pub fn gradients_non_finite(&self) -> Result<bool> {
        for state in self.resident_parameter_banks.borrow().iter().flatten() {
            if self.backend.any_non_finite_f32(&state.updates)
                .map_err(|e|backend_error("resident gradient finiteness",e))? { return Ok(true); }
        }
        for entry in self.device_train_buffers.borrow().iter().flatten() {
            if self
                .backend
                .any_non_finite_f32(&entry.weight_updates)
                .map_err(|err| backend_error("gradient finiteness check", err))?
                || self
                    .backend
                    .any_non_finite_f32(&entry.bias_updates)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
            {
                return Ok(true);
            }
            if let Some(scales) = &entry.scale_updates
                && self
                    .backend
                    .any_non_finite_f32(scales)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
            {
                return Ok(true);
            }
            if let Some(pending) = &entry.pending {
                if self
                    .backend
                    .any_non_finite_f32(&pending.weights)
                    .map_err(|err| backend_error("gradient finiteness check", err))?
                    || self
                        .backend
                        .any_non_finite_f32(&pending.biases)
                        .map_err(|err| backend_error("gradient finiteness check", err))?
                {
                    return Ok(true);
                }
                if let Some(scales) = &pending.scales
                    && self
                        .backend
                        .any_non_finite_f32(scales)
                        .map_err(|err| backend_error("gradient finiteness check", err))?
                {
                    return Ok(true);
                }
            }
        }
        for entry in self
            .device_moe_train_buffers
            .borrow()
            .iter()
            .flatten()
            .filter(|entry| entry.dirty)
        {
            if self
                .backend
                .any_non_finite_f32(&entry.router_weight_updates)
                .map_err(|err| backend_error("MoE gradient finiteness check", err))?
                || self
                    .backend
                    .any_non_finite_f32(&entry.router_bias_updates)
                    .map_err(|err| backend_error("MoE gradient finiteness check", err))?
            {
                return Ok(true);
            }
            for ((weights, biases), scales) in entry
                .expert_weight_updates
                .iter()
                .zip(&entry.expert_bias_updates)
                .zip(&entry.expert_scale_updates)
            {
                if self
                    .backend
                    .any_non_finite_f32(weights)
                    .map_err(|err| backend_error("MoE gradient finiteness check", err))?
                    || self
                        .backend
                        .any_non_finite_f32(biases)
                        .map_err(|err| backend_error("MoE gradient finiteness check", err))?
                {
                    return Ok(true);
                }
                if let Some(scales) = scales
                    && self
                        .backend
                        .any_non_finite_f32(scales)
                        .map_err(|err| backend_error("MoE gradient finiteness check", err))?
                {
                    return Ok(true);
                }
            }
        }
        for train in self.device_eml_train_buffers.borrow().iter().flatten() {
            for branch in [&train.input, &train.self_params] {
                if self
                    .backend
                    .any_non_finite_f32(&branch.weight_updates)
                    .map_err(|err| backend_error("EML gradient finiteness check", err))?
                    || self
                        .backend
                        .any_non_finite_f32(&branch.bias_updates)
                        .map_err(|err| backend_error("EML gradient finiteness check", err))?
                {
                    return Ok(true);
                }
                if let Some(scales) = &branch.scale_updates
                    && self
                        .backend
                        .any_non_finite_f32(scales)
                        .map_err(|err| backend_error("EML BN gradient finiteness check", err))?
                {
                    return Ok(true);
                }
                if let Some(pending) = &branch.pending {
                    if self
                        .backend
                        .any_non_finite_f32(&pending.weights)
                        .map_err(|err| {
                            backend_error("EML pending gradient finiteness check", err)
                        })?
                        || self
                            .backend
                            .any_non_finite_f32(&pending.biases)
                            .map_err(|err| {
                                backend_error("EML pending gradient finiteness check", err)
                            })?
                    {
                        return Ok(true);
                    }
                    if let Some(scales) = &pending.scales
                        && self.backend.any_non_finite_f32(scales).map_err(|err| {
                            backend_error("EML pending BN gradient finiteness check", err)
                        })?
                    {
                        return Ok(true);
                    }
                }
            }
        }
        for entry in self
            .device_dcnv4_train_buffers
            .borrow()
            .iter()
            .flatten()
            .filter(|entry| entry.dirty)
        {
            if self
                .backend
                .any_non_finite_f32(&entry.offset_weight_updates)
                .map_err(|err| backend_error("DCNv4 gradient finiteness check", err))?
                || self
                    .backend
                    .any_non_finite_f32(&entry.offset_bias_updates)
                    .map_err(|err| backend_error("DCNv4 gradient finiteness check", err))?
                || self
                    .backend
                    .any_non_finite_f32(&entry.bias_updates)
                    .map_err(|err| backend_error("DCNv4 gradient finiteness check", err))?
            {
                return Ok(true);
            }
            if let Some(scales) = &entry.scale_updates
                && self
                    .backend
                    .any_non_finite_f32(scales)
                    .map_err(|err| backend_error("DCNv4 scale-gradient finiteness check", err))?
            {
                return Ok(true);
            }
        }
        let host_gradients = self.cpu_fallback_train_buffers.borrow();
        let finite = |values: &[f32]| values.iter().any(|value| !value.is_finite());
        let connected_non_finite = |train: &ConnectedTrainBuffers| {
            finite(&train.weight_updates)
                || finite(&train.bias_updates)
                || train.batch_norm.as_ref().is_some_and(|batch_norm| {
                    finite(&batch_norm.scale_updates)
                        || finite(&batch_norm.mean_delta)
                        || finite(&batch_norm.variance_delta)
                })
        };
        let convolutional_non_finite = |train: &ConvolutionalTrainBuffers| {
            finite(&train.weight_updates)
                || finite(&train.bias_updates)
                || train.batch_norm.as_ref().is_some_and(|batch_norm| {
                    finite(&batch_norm.scale_updates)
                        || finite(&batch_norm.mean_delta)
                        || finite(&batch_norm.variance_delta)
                })
        };
        for train in host_gradients.iter() {
            let non_finite = match train {
                LayerTrainBuffers::Connected(train) => connected_non_finite(train),
                LayerTrainBuffers::Rnn(train) => {
                    connected_non_finite(&train.input)
                        || connected_non_finite(&train.self_params)
                        || connected_non_finite(&train.output)
                }
                LayerTrainBuffers::Lstm(train) => [
                    &train.wf, &train.wi, &train.wg, &train.wo, &train.uf, &train.ui, &train.ug,
                    &train.uo,
                ]
                .into_iter()
                .any(connected_non_finite),
                LayerTrainBuffers::Crnn(train) => {
                    convolutional_non_finite(&train.input)
                        || convolutional_non_finite(&train.self_params)
                        || convolutional_non_finite(&train.output)
                }
                LayerTrainBuffers::EmlConvolutional(train) => {
                    convolutional_non_finite(&train.input)
                        || convolutional_non_finite(&train.self_params)
                }
                _ => false,
            };
            if non_finite {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Persist the Adam step counter and device/host moment tensors to a sidecar
    /// file. DNOPT4 adds device MoE expert BN-scale moments; DNOPT5 adds EML
    /// branch moments. DNOPT1–4 remain readable. The `.weights` format stays
    /// C++-compatible and carries only parameters plus `seen`; this sidecar
    /// preserves optimizer state on resume.
    pub fn save_optimizer_state(&self, path: impl AsRef<Path>) -> Result<()> {
        if self.network.net.muon && self.muon_state.len() != self.network.layers.len() {
            return Err(Error::Weights(
                "GPU Muon state layer count does not match the network".to_owned(),
            ));
        }
        let mut file = std::io::BufWriter::with_capacity(1 << 20, File::create(path)?);
        let has_moe_scale_moments = self
            .moe_adam_state
            .iter()
            .flatten()
            .any(|state| state.experts.iter().any(|expert| expert.scales.is_some()));
        let has_eml_moments = self.eml_adam_state.iter().any(Option::is_some);
        let version = if self.network.net.muon {
            6
        } else if has_eml_moments {
            5
        } else if has_moe_scale_moments {
            4
        } else {
            3
        };
        file.write_all(match version {
            6 => b"DNOPT6\0\0",
            5 => b"DNOPT5\0\0",
            4 => b"DNOPT4\0\0",
            _ => b"DNOPT3\0\0",
        })?;
        file.write_all(&self.adam_step.to_le_bytes())?;
        let entries: Vec<usize> = self
            .adam_state
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.as_ref().map(|_| index))
            .collect();
        file.write_all(&(entries.len() as u64).to_le_bytes())?;
        for index in entries {
            file.write_all(&(index as u64).to_le_bytes())?;
            let state = self.adam_state[index]
                .as_ref()
                .expect("filtered to Some above");
            let (weight_shape, filter_shape) = self.adam_parameter_shapes(index)?;
            self.write_moments(&mut file, state.weights.as_ref(), &weight_shape)?;
            self.write_moments(&mut file, state.biases.as_ref(), &filter_shape)?;
            self.write_moments(&mut file, state.scales.as_ref(), &filter_shape)?;
        }
        let mut host_states = self.host_adam_state.borrow().clone();
        self.materialize_resident_optimizer_states(&mut host_states)?;
        for (index, device_state) in self.dcnv4_adam_state.iter().enumerate() {
            let Some(device_state) = device_state else {
                continue;
            };
            let host_state = host_states[index].get_or_insert_with(HostAdamLayerState::default);
            let download_pair =
                |pair: &Option<GpuMoments<B::Tensor>>| -> Result<Option<(Vec<f32>, Vec<f32>)>> {
                    let Some((m, v)) = pair else {
                        return Ok(None);
                    };
                    Ok(Some((
                        self.backend.download_f32(m).map_err(|err| {
                            backend_error("download DCNv4 Adam first moment for sidecar", err)
                        })?,
                        self.backend.download_f32(v).map_err(|err| {
                            backend_error("download DCNv4 Adam second moment for sidecar", err)
                        })?,
                    )))
                };
            host_state.weights = download_pair(&device_state.offset_weights)?;
            host_adam_child(host_state, 0).biases = download_pair(&device_state.offset_biases)?;
            host_state.biases = download_pair(&device_state.biases)?;
            host_state.scales = download_pair(&device_state.scales)?;
        }
        let host_entries: Vec<usize> = host_states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.as_ref().map(|_| index))
            .collect();
        file.write_all(&(host_entries.len() as u64).to_le_bytes())?;
        for index in host_entries {
            file.write_all(&(index as u64).to_le_bytes())?;
            write_host_adam_state(
                &mut file,
                host_states[index].as_ref().expect("filtered to Some above"),
            )?;
        }
        let moe_entries: Vec<usize> = self
            .moe_adam_state
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.as_ref().map(|_| index))
            .collect();
        file.write_all(&(moe_entries.len() as u64).to_le_bytes())?;
        for index in moe_entries {
            file.write_all(&(index as u64).to_le_bytes())?;
            let state = self.moe_adam_state[index]
                .as_ref()
                .expect("filtered to Some above");
            let (router_weights, router_biases, expert_weights, expert_biases, expert_count) =
                self.moe_adam_parameter_shapes(index)?;
            self.write_moments(&mut file, state.router.weights.as_ref(), &router_weights)?;
            self.write_moments(&mut file, state.router.biases.as_ref(), &router_biases)?;
            if state.experts.len() != expert_count {
                return Err(Error::Weights(format!(
                    "MoE Adam expert moment count mismatch at layer #{index}"
                )));
            }
            file.write_all(&(state.experts.len() as u64).to_le_bytes())?;
            for expert in &state.experts {
                self.write_moments(&mut file, expert.weights.as_ref(), &expert_weights)?;
                self.write_moments(&mut file, expert.biases.as_ref(), &expert_biases)?;
                if version >= 4 {
                    self.write_moments(&mut file, expert.scales.as_ref(), &expert_biases)?;
                }
            }
        }
        if version >= 5 {
            let eml_entries: Vec<usize> = self
                .eml_adam_state
                .iter()
                .enumerate()
                .filter_map(|(index, state)| state.as_ref().map(|_| index))
                .collect();
            file.write_all(&(eml_entries.len() as u64).to_le_bytes())?;
            for index in eml_entries {
                file.write_all(&(index as u64).to_le_bytes())?;
                let state = self.eml_adam_state[index]
                    .as_ref()
                    .expect("filtered to Some above");
                let (weight_shape, filter_shape) = self.eml_adam_parameter_shapes(index)?;
                for branch in [&state.input, &state.self_params] {
                    self.write_moments(&mut file, branch.weights.as_ref(), &weight_shape)?;
                    self.write_moments(&mut file, branch.biases.as_ref(), &filter_shape)?;
                    self.write_moments(&mut file, branch.scales.as_ref(), &filter_shape)?;
                }
            }
        }
        if version >= 6 {
            let mut muon_states = self.muon_state.clone();
            self.materialize_resident_muon_states(&mut muon_states)?;
            let entries: Vec<usize> = muon_states
                .iter()
                .enumerate()
                .filter_map(|(index, state)| {
                    state
                        .as_ref()
                        .filter(|state| !state.matrices.is_empty())
                        .map(|_| index)
                })
                .collect();
            file.write_all(&(entries.len() as u64).to_le_bytes())?;
            for index in entries {
                file.write_all(&(index as u64).to_le_bytes())?;
                write_muon_layer_state(
                    &mut file,
                    muon_states[index]
                        .as_ref()
                        .expect("filtered to non-empty Muon state"),
                )?;
            }
        }
        file.flush()?;
        Ok(())
    }

    /// Restore state written by [`Self::save_optimizer_state`].  Layers absent
    /// from the sidecar keep `None` moments (Adam will initialise them on the
    /// next step), and the step counter is restored as stored.
    pub fn load_optimizer_state(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        let version = match &magic {
            b"DNOPT1\0\0" => 1,
            b"DNOPT2\0\0" => 2,
            b"DNOPT3\0\0" => 3,
            b"DNOPT4\0\0" => 4,
            b"DNOPT5\0\0" => 5,
            b"DNOPT6\0\0" => 6,
            _ => return Err(Error::Weights("bad optimizer state magic".to_owned())),
        };
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let step = u32::from_le_bytes(buf4);
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        let mut state: Vec<Option<GpuAdamLayerState<B::Tensor>>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        for _ in 0..count {
            file.read_exact(&mut buf8)?;
            let index = u64::from_le_bytes(buf8) as usize;
            if index >= self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer state layer index out of range".to_owned(),
                ));
            }
            let (weight_shape, filter_shape) = self.adam_parameter_shapes(index)?;
            let weights = self.read_moments(&mut file, &weight_shape)?;
            let biases = self.read_moments(&mut file, &filter_shape)?;
            let scales = self.read_moments(&mut file, &filter_shape)?;
            state[index] = Some(GpuAdamLayerState {
                weights,
                biases,
                scales,
            });
        }
        let mut host_state: Vec<Option<HostAdamLayerState>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 2 {
            file.read_exact(&mut buf8)?;
            let host_count = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| {
                Error::Weights("optimizer host layer count is too large".to_owned())
            })?;
            if host_count > self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer host layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..host_count {
                file.read_exact(&mut buf8)?;
                let index = u64::from_le_bytes(buf8) as usize;
                if index >= self.network.layers.len() || seen[index] {
                    return Err(Error::Weights(
                        "optimizer state host layer index is invalid or repeated".to_owned(),
                    ));
                }
                seen[index] = true;
                let max_scalars = host_adam_scalar_capacity(&self.params[index]);
                host_state[index] = Some(read_host_adam_state(&mut file, max_scalars)?);
            }
        }
        for (index, state) in host_state.iter().enumerate() {
            if let Some(state) = state
                && matches!(
                    self.network.layers[index].kind,
                    LayerKind::RecursiveBlock(_)
                )
            {
                validate_recursive_block_adam_state(
                    &self.network.layers[index],
                    &self.params[index],
                    state,
                )?;
            }
        }
        let mut moe_state: Vec<Option<GpuMoeAdamLayerState<B::Tensor>>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 3 {
            file.read_exact(&mut buf8)?;
            let moe_count = usize::try_from(u64::from_le_bytes(buf8))
                .map_err(|_| Error::Weights("optimizer MoE layer count is too large".to_owned()))?;
            if moe_count > self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer MoE layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..moe_count {
                file.read_exact(&mut buf8)?;
                let index = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| {
                    Error::Weights("optimizer MoE layer index is too large".to_owned())
                })?;
                if index >= self.network.layers.len() || seen[index] {
                    return Err(Error::Weights(
                        "optimizer MoE layer index is invalid or repeated".to_owned(),
                    ));
                }
                seen[index] = true;
                let (router_weights, router_biases, expert_weights, expert_biases, expert_count) =
                    self.moe_adam_parameter_shapes(index)?;
                let router = GpuAdamLayerState {
                    weights: self.read_moments(&mut file, &router_weights)?,
                    biases: self.read_moments(&mut file, &router_biases)?,
                    scales: None,
                };
                file.read_exact(&mut buf8)?;
                let saved_experts = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| {
                    Error::Weights("optimizer MoE expert count is too large".to_owned())
                })?;
                if saved_experts != expert_count {
                    return Err(Error::Weights(format!(
                        "optimizer MoE expert count mismatch at layer #{index}"
                    )));
                }
                let mut experts = Vec::with_capacity(expert_count);
                for _ in 0..expert_count {
                    experts.push(GpuAdamLayerState {
                        weights: self.read_moments(&mut file, &expert_weights)?,
                        biases: self.read_moments(&mut file, &expert_biases)?,
                        scales: if version >= 4 {
                            self.read_moments(&mut file, &expert_biases)?
                        } else {
                            None
                        },
                    });
                }
                moe_state[index] = Some(GpuMoeAdamLayerState { router, experts });
            }
        }
        let mut eml_state: Vec<Option<GpuEmlAdamLayerState<B::Tensor>>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 5 {
            file.read_exact(&mut buf8)?;
            let eml_count = usize::try_from(u64::from_le_bytes(buf8))
                .map_err(|_| Error::Weights("optimizer EML layer count is too large".to_owned()))?;
            if eml_count > self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer EML layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..eml_count {
                file.read_exact(&mut buf8)?;
                let index = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| {
                    Error::Weights("optimizer EML layer index is too large".to_owned())
                })?;
                if index >= self.network.layers.len() || seen[index] {
                    return Err(Error::Weights(
                        "optimizer EML layer index is invalid or repeated".to_owned(),
                    ));
                }
                seen[index] = true;
                let (weight_shape, filter_shape) = self.eml_adam_parameter_shapes(index)?;
                let mut read_branch = || -> Result<GpuAdamLayerState<B::Tensor>> {
                    Ok(GpuAdamLayerState {
                        weights: self.read_moments(&mut file, &weight_shape)?,
                        biases: self.read_moments(&mut file, &filter_shape)?,
                        scales: self.read_moments(&mut file, &filter_shape)?,
                    })
                };
                eml_state[index] = Some(GpuEmlAdamLayerState {
                    input: read_branch()?,
                    self_params: read_branch()?,
                });
            }
        }
        let mut muon_state: Vec<Option<MuonLayerState>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 6 {
            file.read_exact(&mut buf8)?;
            let muon_count = usize::try_from(u64::from_le_bytes(buf8))
                .map_err(|_| Error::Weights("Muon layer count is too large".to_owned()))?;
            if muon_count > self.network.layers.len() {
                return Err(Error::Weights(
                    "Muon layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..muon_count {
                file.read_exact(&mut buf8)?;
                let index = usize::try_from(u64::from_le_bytes(buf8))
                    .map_err(|_| Error::Weights("Muon layer index is too large".to_owned()))?;
                if index >= self.network.layers.len() || seen[index] {
                    return Err(Error::Weights(
                        "Muon layer index is invalid or repeated".to_owned(),
                    ));
                }
                seen[index] = true;
                let capacity = host_adam_scalar_capacity(&self.params[index]);
                muon_state[index] = Some(read_muon_layer_state(&mut file, capacity)?);
            }
        }
        if !self.network.net.muon
            && muon_state
                .iter()
                .flatten()
                .any(|state| !state.matrices.is_empty())
        {
            return Err(Error::Weights(
                "optimizer sidecar contains Muon state but [net] muon=0".to_owned(),
            ));
        }
        let mut restored_step = step;
        if self.network.net.muon && version < 6 {
            host_state.fill(None);
            state.fill(None);
            moe_state.fill(None);
            eml_state.fill(None);
            restored_step = 0;
        } else if self.network.net.muon {
            for (index, state) in muon_state.iter().enumerate() {
                let Some(state) = state else { continue };
                let layer = &self.network.layers[index];
                let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
                let expected = matrix_shapes(
                    &layer.kind,
                    layer.input_shape,
                    layer.output_shape,
                    &mut self.params[index],
                    &mut buffers[index],
                )
                .map_err(|error| Error::Weights(format!("invalid Muon layer layout: {error}")))?;
                let found: Vec<(usize, usize)> = state
                    .matrices
                    .iter()
                    .map(|matrix| (matrix.rows, matrix.cols))
                    .collect();
                if found != expected {
                    return Err(Error::Weights(format!(
                        "Muon state matrix shapes for layer {index} are {found:?}, expected {expected:?}"
                    )));
                }
            }
        }
        self.synchronize_resident_parameter_state()?;
        self.resident_parameter_banks.get_mut().clear();
        *self.host_adam_state.get_mut() = host_state;
        self.adam_state = state;
        self.moe_adam_state = moe_state;
        self.eml_adam_state = eml_state;
        self.dcnv4_adam_state = (0..self.network.layers.len()).map(|_| None).collect();
        self.muon_state = muon_state;
        *self.device_muon_momentum.borrow_mut() = vec![None; self.network.layers.len()];
        self.adam_step = restored_step;
        Ok(())
    }

    pub(super) fn eml_adam_parameter_shapes(&self, index: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let layer = self.network.layers.get(index).ok_or_else(|| {
            Error::Weights(format!("optimizer EML layer index {index} is out of range"))
        })?;
        let LayerKind::EmlConvolutional(spec) = &layer.kind else {
            return Err(Error::Weights(format!(
                "optimizer EML state for non-EML layer #{index}"
            )));
        };
        let channels = layer.input_shape.channels / spec.branch.groups.max(1);
        Ok((
            vec![
                spec.branch.filters,
                channels,
                spec.branch.size,
                spec.branch.size,
            ],
            vec![spec.branch.filters],
        ))
    }

    pub(super) fn adam_parameter_shapes(&self, index: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let layer = &self.network.layers[index];
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Err(Error::Weights(format!(
                "optimizer state for non-convolutional layer #{index}"
            )));
        };
        let grouped = layer.input_shape.channels / spec.groups.max(1);
        Ok((
            vec![spec.filters, grouped, spec.size, spec.size],
            vec![spec.filters],
        ))
    }

    pub(super) fn moe_adam_parameter_shapes(
        &self,
        index: usize,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, usize)> {
        let layer = self.network.layers.get(index).ok_or_else(|| {
            Error::Weights(format!(
                "MoE optimizer state layer #{index} is out of range"
            ))
        })?;
        let LayerKind::Moe(spec) = &layer.kind else {
            return Err(Error::Weights(format!(
                "optimizer state for non-MoE layer #{index}"
            )));
        };
        let grouped = layer.input_shape.channels / spec.expert.groups.max(1);
        Ok((
            vec![spec.experts, layer.input_shape.channels],
            vec![spec.experts],
            vec![
                spec.expert.filters,
                grouped,
                spec.expert.size,
                spec.expert.size,
            ],
            vec![spec.expert.filters],
            spec.experts,
        ))
    }

    pub(super) fn upload_host_adam_moments(
        &self,
        moments: Option<(Vec<f32>, Vec<f32>)>,
        shape: &[usize],
    ) -> Result<Option<GpuMoments<B::Tensor>>> {
        let Some((first, second)) = moments else {
            return Ok(None);
        };
        let expected = shape.iter().product::<usize>();
        if first.len() != expected || second.len() != expected {
            return Err(Error::InvalidConfig(
                "host-to-device Adam moment shape mismatch".to_owned(),
            ));
        }
        let first = self
            .backend
            .upload_f32(shape, &first)
            .map_err(|err| backend_error("upload restored Adam moment", err))?;
        let second = self
            .backend
            .upload_f32(shape, &second)
            .map_err(|err| backend_error("upload restored Adam moment", err))?;
        Ok(Some((first, second)))
    }

    pub(super) fn moe_adam_state_for_update(&self, index: usize) -> Result<GpuMoeAdamLayerState<B::Tensor>> {
        if let Some(state) = self.moe_adam_state.get(index).and_then(Option::as_ref) {
            return Ok(state.clone());
        }
        let (router_weights, router_biases, expert_weights, expert_biases, expert_count) =
            self.moe_adam_parameter_shapes(index)?;
        let LayerKind::Moe(spec) = &self.network.layers[index].kind else {
            unreachable!("MoE parameter-shape helper validated layer kind");
        };
        let host_state = self
            .host_adam_state
            .borrow()
            .get(index)
            .and_then(Clone::clone);
        let mut state = GpuMoeAdamLayerState {
            router: GpuAdamLayerState::default(),
            experts: (0..expert_count)
                .map(|_| GpuAdamLayerState::default())
                .collect(),
        };
        let Some(host_state) = host_state else {
            return Ok(state);
        };
        if host_state.scales.is_some() || host_state.children.len() > expert_count {
            return Err(Error::InvalidConfig(format!(
                "restored MoE Adam host state shape mismatch at layer #{index}"
            )));
        }
        state.router.weights =
            self.upload_host_adam_moments(host_state.weights, &router_weights)?;
        state.router.biases = self.upload_host_adam_moments(host_state.biases, &router_biases)?;
        for (expert_index, host_expert) in host_state.children.into_iter().enumerate() {
            if host_expert.scales.is_some() && !spec.expert.batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "restored BN MoE Adam state reached no-BN layer #{index}"
                )));
            }
            state.experts[expert_index].weights =
                self.upload_host_adam_moments(host_expert.weights, &expert_weights)?;
            state.experts[expert_index].biases =
                self.upload_host_adam_moments(host_expert.biases, &expert_biases)?;
            state.experts[expert_index].scales =
                self.upload_host_adam_moments(host_expert.scales, &expert_biases)?;
        }
        Ok(state)
    }

    pub(super) fn eml_adam_state_for_update(&self, index: usize) -> Result<GpuEmlAdamLayerState<B::Tensor>> {
        if let Some(state) = self.eml_adam_state.get(index).and_then(Option::as_ref) {
            return Ok(state.clone());
        }
        let (weight_shape, filter_shape) = self.eml_adam_parameter_shapes(index)?;
        let LayerKind::EmlConvolutional(spec) = &self.network.layers[index].kind else {
            unreachable!("EML parameter-shape helper validated layer kind");
        };
        let host_state = self
            .host_adam_state
            .borrow()
            .get(index)
            .and_then(Clone::clone);
        let mut state = GpuEmlAdamLayerState::default();
        let Some(host_state) = host_state else {
            return Ok(state);
        };
        if host_state.weights.is_some()
            || host_state.biases.is_some()
            || host_state.scales.is_some()
            || host_state.children.len() > 2
            || host_state
                .children
                .iter()
                .any(|branch| !branch.children.is_empty())
        {
            return Err(Error::InvalidConfig(format!(
                "restored EML Adam host state shape mismatch at layer #{index}"
            )));
        }
        for (slot, host_branch) in host_state.children.into_iter().enumerate() {
            if host_branch.scales.is_some() && !spec.branch.batch_normalize {
                return Err(Error::InvalidConfig(format!(
                    "restored BN EML Adam state reached no-BN layer #{index}"
                )));
            }
            let branch = if slot == 0 {
                &mut state.input
            } else {
                &mut state.self_params
            };
            branch.weights = self.upload_host_adam_moments(host_branch.weights, &weight_shape)?;
            branch.biases = self.upload_host_adam_moments(host_branch.biases, &filter_shape)?;
            branch.scales = self.upload_host_adam_moments(host_branch.scales, &filter_shape)?;
        }
        Ok(state)
    }

    pub(super) fn write_moments(
        &self,
        file: &mut impl Write,
        moments: Option<&GpuMoments<B::Tensor>>,
        shape: &[usize],
    ) -> Result<()> {
        match moments {
            None => {
                file.write_all(&[0u8])?;
                Ok(())
            }
            Some((first, second)) => {
                file.write_all(&[1u8])?;
                for tensor in [first, second] {
                    let values = self
                        .backend
                        .download_f32(tensor)
                        .map_err(|err| backend_error("download Adam moment", err))?;
                    if values.len() != shape.iter().product::<usize>() {
                        return Err(Error::Weights(
                            "Adam moment length does not match its parameter shape".to_owned(),
                        ));
                    }
                    crate::weights::write_f32_slice(file, &values)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn read_moments(
        &self,
        file: &mut File,
        shape: &[usize],
    ) -> Result<Option<GpuMoments<B::Tensor>>> {
        let mut flag = [0u8; 1];
        file.read_exact(&mut flag)?;
        if flag[0] == 0 {
            return Ok(None);
        }
        let len = shape.iter().product::<usize>();
        let mut first = vec![0f32; len];
        let mut second = vec![0f32; len];
        for buffer in [&mut first, &mut second] {
            let mut bytes = vec![0u8; len * 4];
            file.read_exact(&mut bytes)?;
            for (slot, chunk) in buffer.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        let first = self
            .backend
            .upload_f32(shape, &first)
            .map_err(|err| backend_error("upload Adam moment", err))?;
        let second = self
            .backend
            .upload_f32(shape, &second)
            .map_err(|err| backend_error("upload Adam moment", err))?;
        Ok(Some((first, second)))
    }

    /// The tensor backend (for benchmark statistics such as workspace size).
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn from_path(path: impl AsRef<Path>, backend: B) -> Result<Self> {
        Ok(Self::new(Network::from_path(path)?, backend))
    }

    pub fn zero_input(&self) -> Tensor {
        Tensor::zeros(self.network.net.input_shape())
    }

    pub fn load_weights_file(&mut self, path: impl AsRef<Path>) -> Result<WeightsHeader> {
        self.retire_resident_parameter_banks()?;
        let header = load_weights_into_params(&self.network, &mut self.params, path)?;
        self.clear_device_conv_params();
        self.clear_device_train_buffers();
        self.warm_device_conv_params()?;
        Ok(header)
    }

}
