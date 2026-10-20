use super::*;

/// Pre-allocate the shared backward scratch buffer.
/// Sized to max(filters + nweights) across all conv layers — mirrors C++ `net->workspace`.
pub fn compute_cpu_workspace_size(network: &Network) -> usize {
    network
        .layers
        .iter()
        .map(|l| match &l.kind {
            LayerKind::Convolutional(spec) => {
                let groups = spec.groups.max(1);
                let in_ch = l.input_shape.channels / groups;
                let nweights = spec.filters * in_ch * spec.size * spec.size;
                spec.filters + nweights
            }
            LayerKind::Connected(spec) => {
                let nweights = spec.output * l.input_shape.element_count();
                spec.output + nweights
            }
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutableNetwork {
    pub network: Network,
    pub params: Vec<LayerParams>,
    pub train_buffers: Vec<LayerTrainBuffers>,
    /// Shared scratch buffer for backward pass — avoids per-call Vec allocation.
    /// Equivalent to C++ `net->workspace`. Private; threaded through backward_accumulate paths.
    pub(super) workspace: Vec<f32>,
    /// Training iteration in progress (C++ `get_current_iteration`): drives the
    /// DropBlock ramp and the `dont_update`/`burnin_update` gates (see
    /// [`ExecutableNetwork::set_train_iteration`]).
    pub(super) train_iteration: usize,
    /// CPU AdamW fallback state, persisted separately from Darknet weights.
    pub(super) host_adam_state: Vec<Option<HostAdamLayerState>>,
    /// CPU Muon momentum, one state per eligible matrix block.
    pub(super) muon_state: Vec<Option<MuonLayerState>>,
    pub(super) adam_step: u32,
}

impl ExecutableNetwork {
    pub fn new(network: Network) -> Self {
        // Preserve one parameter/train-buffer slot per graph layer.  Layers
        // without weights use `None`, which keeps Rust indices aligned with
        // Darknet's C++ layer indices used by route and shortcut references.
        let mut params = network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        for (layer, params) in network.layers.iter().zip(&mut params) {
            if let (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(params)) =
                (&layer.kind, params)
                && spec.ouroboros == 1
            {
                let channels = layer.input_shape.channels;
                params.controller_biases[channels..2 * channels].fill(spec.gate_bias);
            }
        }
        let train_buffers = network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect::<Vec<_>>();
        let workspace = vec![0.0f32; compute_cpu_workspace_size(&network)];
        let layer_count = network.layers.len();
        Self {
            network,
            params,
            train_buffers,
            workspace,
            train_iteration: 0,
            host_adam_state: (0..layer_count).map(|_| None).collect(),
            muon_state: (0..layer_count).map(|_| None).collect(),
            adam_step: 0,
        }
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(Network::from_path(path)?))
    }

    pub fn zero_input(&self) -> Tensor {
        Tensor::zeros(self.network.net.input_shape())
    }

    pub fn load_weights_file(&mut self, path: impl AsRef<Path>) -> Result<WeightsHeader> {
        load_weights_into_params(&self.network, &mut self.params, path)
    }

    /// Adam updates applied so far (the `t` of the next update).
    pub fn adam_step(&self) -> u32 {
        self.adam_step
    }

    /// Restore the Adam bias-correction step counter.
    pub fn set_adam_step(&mut self, step: u32) {
        self.adam_step = step;
    }

    /// Save CPU AdamW fallback and Muon state in the shared DNOPT sidecar format.
    pub fn save_optimizer_state(&self, path: impl AsRef<Path>) -> Result<()> {
        if !self.network.net.muon
            && self
                .muon_state
                .iter()
                .flatten()
                .any(|state| !state.matrices.is_empty())
        {
            return Err(Error::Weights(
                "cannot save Muon momentum while [net] muon=0".to_owned(),
            ));
        }
        if self.network.net.muon && self.muon_state.len() != self.network.layers.len() {
            return Err(Error::Weights(
                "Muon optimizer state layer count does not match the network".to_owned(),
            ));
        }
        let mut file = std::io::BufWriter::with_capacity(1 << 20, File::create(path)?);
        file.write_all(if self.network.net.muon {
            b"DNOPT6\0\0"
        } else {
            b"DNOPT3\0\0"
        })?;
        file.write_all(&self.adam_step.to_le_bytes())?;
        file.write_all(&0u64.to_le_bytes())?; // no device-resident moments

        let entries: Vec<usize> = self
            .host_adam_state
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.as_ref().map(|_| index))
            .collect();
        file.write_all(&(entries.len() as u64).to_le_bytes())?;
        for index in entries {
            file.write_all(&(index as u64).to_le_bytes())?;
            write_host_adam_state(
                &mut file,
                self.host_adam_state[index]
                    .as_ref()
                    .expect("filtered to Some above"),
            )?;
        }
        file.write_all(&0u64.to_le_bytes())?; // no device MoE moments
        if self.network.net.muon {
            file.write_all(&0u64.to_le_bytes())?; // no device EML moments
            let entries: Vec<usize> = self
                .muon_state
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
                    self.muon_state[index]
                        .as_ref()
                        .expect("filtered to a non-empty state above"),
                )?;
            }
        }
        file.flush()?;
        Ok(())
    }

    /// Restore CPU AdamW fallback and Muon momentum from a host-only DNOPT sidecar.
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
        let mut bytes4 = [0u8; 4];
        file.read_exact(&mut bytes4)?;
        let step = u32::from_le_bytes(bytes4);
        let mut bytes8 = [0u8; 8];
        file.read_exact(&mut bytes8)?;
        if u64::from_le_bytes(bytes8) != 0 {
            return Err(Error::Weights(
                "CPU optimizer state cannot load device Adam moments".to_owned(),
            ));
        }

        let mut host_state: Vec<Option<HostAdamLayerState>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 2 {
            file.read_exact(&mut bytes8)?;
            let count = usize::try_from(u64::from_le_bytes(bytes8)).map_err(|_| {
                Error::Weights("optimizer host layer count is too large".to_owned())
            })?;
            if count > self.network.layers.len() {
                return Err(Error::Weights(
                    "optimizer host layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..count {
                file.read_exact(&mut bytes8)?;
                let index = usize::try_from(u64::from_le_bytes(bytes8)).map_err(|_| {
                    Error::Weights("optimizer host layer index is too large".to_owned())
                })?;
                if index >= self.network.layers.len() || seen[index] {
                    return Err(Error::Weights(
                        "optimizer host layer index is invalid or repeated".to_owned(),
                    ));
                }
                seen[index] = true;
                let capacity = host_adam_scalar_capacity(&self.params[index]);
                host_state[index] = Some(read_host_adam_state(&mut file, capacity)?);
            }
        }
        if version >= 3 {
            file.read_exact(&mut bytes8)?;
            if u64::from_le_bytes(bytes8) != 0 {
                return Err(Error::Weights(
                    "CPU optimizer state cannot load device MoE moments".to_owned(),
                ));
            }
        }
        if version >= 5 {
            file.read_exact(&mut bytes8)?;
            if u64::from_le_bytes(bytes8) != 0 {
                return Err(Error::Weights(
                    "CPU optimizer state cannot load device EML moments".to_owned(),
                ));
            }
        }
        let mut muon_state: Vec<Option<MuonLayerState>> =
            (0..self.network.layers.len()).map(|_| None).collect();
        if version >= 6 {
            file.read_exact(&mut bytes8)?;
            let count = usize::try_from(u64::from_le_bytes(bytes8))
                .map_err(|_| Error::Weights("Muon layer count is too large".to_owned()))?;
            if count > self.network.layers.len() {
                return Err(Error::Weights(
                    "Muon layer count exceeds network size".to_owned(),
                ));
            }
            let mut seen = vec![false; self.network.layers.len()];
            for _ in 0..count {
                file.read_exact(&mut bytes8)?;
                let index = usize::try_from(u64::from_le_bytes(bytes8))
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
        for (index, state) in host_state.iter().enumerate() {
            if let Some(state) = state {
                match self.network.layers[index].kind {
                    LayerKind::RecursiveBlock(_) => validate_recursive_block_adam_state(
                        &self.network.layers[index],
                        &self.params[index],
                        state,
                    )?,
                    LayerKind::HyperbolicFoveation(_) => validate_hyperbolic_foveation_adam_state(
                        &self.network.layers[index],
                        &self.params[index],
                        state,
                    )?,
                    LayerKind::FoveationFilm(_) => validate_foveation_film_adam_state(
                        &self.network.layers[index],
                        &self.params[index],
                        state,
                    )?,
                    LayerKind::DptHead(_) => validate_dpt_head_adam_state(
                        &self.network.layers[index],
                        &self.params[index],
                        state,
                        self.network.net.muon,
                    )?,
                    _ => {}
                }
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
            // DNOPT1-5 have no Muon momentum. Do not reinterpret their Adam
            // moments as Muon state; start both optimizer components fresh.
            host_state.fill(None);
            restored_step = 0;
        } else if self.network.net.muon {
            for (index, state) in muon_state.iter().enumerate() {
                let Some(state) = state else {
                    continue;
                };
                let layer = &self.network.layers[index];
                let expected = matrix_shapes(
                    &layer.kind,
                    layer.input_shape,
                    layer.output_shape,
                    &mut self.params[index],
                    &mut self.train_buffers[index],
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
        self.host_adam_state = host_state;
        self.muon_state = muon_state;
        self.adam_step = restored_step;
        Ok(())
    }

    /// Sets the 0-based index of the training iteration in progress (the
    /// CLI's `batch_num`).  The forward pass uses it for the DropBlock ramp;
    /// the update uses `iteration + 1` (C++ `seen / (batch * subdivisions)`
    /// after the iteration's batches) for `dont_update` / `burnin_update`.
    pub fn set_train_iteration(&mut self, iteration: usize) {
        self.train_iteration = iteration;
    }

    pub fn zero_train_buffers(&mut self) {
        self.train_buffers = self
            .network
            .layers
            .iter()
            .map(default_layer_train_buffers)
            .collect();
    }

    /// Apply one Darknet SGD step to the persistent parameter/update buffers.
    ///
    /// Darknet's backward pass stores a *descent direction* in `*_updates`:
    /// detector targets normally produce `target - prediction`, which is the
    /// negative of the mathematical loss derivative.  Consequently the
    /// update below adds `learning_rate / batch * update` to a parameter.  The
    /// convention is intentional and is shared by the CPU runtime and every
    /// backend implementing [`F32TensorOpsBackend::sgd_update_f32`].
    pub fn update_weights(&mut self, lr: f32, momentum: f32, decay: f32, batch: usize) {
        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        for (layer_index, (params, train)) in self
            .params
            .iter_mut()
            .zip(&mut self.train_buffers)
            .enumerate()
        {
            // C++ `update_network`: per-layer update gating.
            let layer = &self.network.layers[layer_index];
            let opts = layer.train;
            if !opts.updates_at(iteration_num, burn_in) {
                continue;
            }
            if let (
                LayerKind::DptHead(spec),
                LayerParams::DptHead(params),
                LayerTrainBuffers::DptHead(train),
            ) = (&layer.kind, &mut *params, &mut *train)
            {
                if train.has_gradient {
                    update_dpt_head_sgd(
                        spec,
                        layer.input_shape,
                        params,
                        train,
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                    train.has_gradient = false;
                }
                continue;
            }
            if let (
                LayerKind::RecursiveBlock(spec),
                LayerParams::RecursiveBlock(params),
                LayerTrainBuffers::RecursiveBlock(train),
            ) = (&layer.kind, &mut *params, &mut *train)
            {
                update_recursive_block_sgd(
                    spec,
                    params,
                    train,
                    lr * opts.learning_rate_scale,
                    momentum,
                    decay,
                    batch,
                );
                continue;
            }
            if let (
                LayerKind::HyperbolicFoveation(spec),
                LayerParams::HyperbolicFoveation(params),
                LayerTrainBuffers::HyperbolicFoveation(train),
            ) = (&layer.kind, &mut *params, &mut *train)
            {
                update_hyperbolic_foveation_sgd(
                    spec,
                    params,
                    train,
                    lr * opts.learning_rate_scale,
                    momentum,
                    batch,
                );
                continue;
            }
            if let (
                LayerKind::FoveationFilm(_),
                LayerParams::FoveationFilm(params),
                LayerTrainBuffers::FoveationFilm(train),
            ) = (&layer.kind, &mut *params, &mut *train)
            {
                update_foveation_film_sgd(
                    params,
                    train,
                    lr * opts.learning_rate_scale,
                    momentum,
                    batch,
                );
                continue;
            }
            match (params, train) {
                (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
                    sgd_update(
                        &mut params.biases,
                        &mut train.bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    sgd_update(
                        &mut params.scales,
                        &mut train.scale_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                }
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) => {
                    // Per-layer `learning_rate` scale, `grad_centr` (with BN)
                    // before the step and `clip` on the weights after it.
                    let lr = lr * opts.learning_rate_scale;
                    if opts.grad_centr && params.batch_norm.is_some() {
                        let f_size = params.weights.len() / params.biases.len().max(1);
                        if f_size > 0 {
                            train.weight_updates = darknet_cuda::host_ops::gradient_centralization(
                                &train.weight_updates,
                                f_size,
                            );
                        }
                    }
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
                    if let (Some(params_bn), Some(train_bn)) =
                        (&mut params.batch_norm, &mut train.batch_norm)
                    {
                        sgd_update(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    }
                    if opts.clip > 0.0 {
                        params.weights =
                            darknet_cuda::host_ops::constrain(&params.weights, opts.clip);
                    }
                }
                (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
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
                    if let (Some(params_bn), Some(train_bn)) =
                        (&mut params.batch_norm, &mut train.batch_norm)
                    {
                        sgd_update(
                            &mut params_bn.scales,
                            &mut train_bn.scale_updates,
                            lr,
                            momentum,
                            0.0,
                            batch,
                        );
                    }
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
                (LayerParams::Gru(params), LayerTrainBuffers::Gru(train)) => {
                    for (params, train) in [
                        (&mut params.input_z, &mut train.input_z),
                        (&mut params.state_z, &mut train.state_z),
                        (&mut params.input_r, &mut train.input_r),
                        (&mut params.state_r, &mut train.state_r),
                        (&mut params.input_h, &mut train.input_h),
                        (&mut params.state_h, &mut train.state_h),
                    ] {
                        update_connected_params(params, train, lr, momentum, decay, batch);
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
                // `[moe_convolutional]`: the router is a plain linear layer and
                // each expert is a plain convolution, so both use the ordinary
                // SGD rule.  The router's weights are not decayed in C++
                // (`update_moe_convolutional_layer` calls `axpy_cpu` with
                // `-decay * batch` only for the experts' offset/collapse
                // weights), so decay is passed as 0.0 for them.
                (LayerParams::Swin(params), LayerTrainBuffers::Swin(train)) => {
                    update_swin_params(
                        params,
                        train,
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (
                    LayerParams::TuckerAttention(params),
                    LayerTrainBuffers::TuckerAttention(train),
                ) => {
                    update_tucker_attention_params(
                        params,
                        train,
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::MambaVision(params), LayerTrainBuffers::MambaVision(train)) => {
                    let layer_lr = lr * opts.learning_rate_scale;
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
                    let layer_lr = lr * opts.learning_rate_scale;
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
                (LayerParams::DeformConv(params), LayerTrainBuffers::DeformConv(train)) => {
                    let layer_lr = lr * opts.learning_rate_scale;
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
                            sgd_update(weights, updates, layer_lr * 0.01, momentum, decay, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated deform_conv mask train buffers"),
                    }
                    match (&mut params.mask_biases, &mut train.mask_bias_updates) {
                        (Some(biases), Some(updates)) => {
                            sgd_update(biases, updates, layer_lr * 0.01, momentum, 0.0, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated deform_conv mask bias buffers"),
                    }
                }
                (LayerParams::DsConv(params), LayerTrainBuffers::DsConv(train)) => {
                    let layer_lr = lr * opts.learning_rate_scale;
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
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Clifford(params), LayerTrainBuffers::Clifford(train)) => {
                    let LayerKind::Clifford(spec) = &self.network.layers[layer_index].kind else {
                        unreachable!("Clifford params are owned by a Clifford layer");
                    };
                    update_clifford_params(
                        spec,
                        params,
                        train,
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::GraphConv(params), LayerTrainBuffers::GraphConv(train)) => {
                    let layer_lr = lr * opts.learning_rate_scale;
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
                        _ => unreachable!("validated graph_conv self-weight buffers"),
                    }
                    match (&mut params.edge_kernel, &mut train.edge_kernel_updates) {
                        (Some(weights), Some(updates)) => {
                            sgd_update(weights, updates, layer_lr, momentum, decay, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated graph_conv edge-kernel buffers"),
                    }
                    match (&mut params.edge_biases, &mut train.edge_bias_updates) {
                        (Some(biases), Some(updates)) => {
                            sgd_update(biases, updates, layer_lr, momentum, 0.0, batch);
                        }
                        (None, None) => {}
                        _ => unreachable!("validated graph_conv edge-bias buffers"),
                    }
                }
                (LayerParams::Dcnv4(params), LayerTrainBuffers::Dcnv4(train)) => {
                    let layer_lr = lr * opts.learning_rate_scale;
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
                (LayerParams::Vit(params), LayerTrainBuffers::Vit(train)) => {
                    let learned_pos = matches!(self.network.layers[layer_index].kind, LayerKind::Vit(ref spec) if spec.learned_pos_embed);
                    update_vit_params(
                        params,
                        train,
                        lr * opts.learning_rate_scale,
                        momentum,
                        decay,
                        batch,
                        learned_pos,
                    );
                }
                (
                    LayerParams::EmlConvolutional(eml),
                    LayerTrainBuffers::EmlConvolutional(train),
                ) => {
                    let layer_lr = lr * opts.learning_rate_scale;
                    update_convolutional_params(
                        &mut eml.input,
                        &mut train.input,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                    update_convolutional_params(
                        &mut eml.self_params,
                        &mut train.self_params,
                        layer_lr,
                        momentum,
                        decay,
                        batch,
                    );
                }
                (LayerParams::Moe(moe), LayerTrainBuffers::Moe(train)) => {
                    sgd_update(
                        &mut moe.router_biases,
                        &mut train.router_bias_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    sgd_update(
                        &mut moe.router_weights,
                        &mut train.router_weight_updates,
                        lr,
                        momentum,
                        0.0,
                        batch,
                    );
                    for (expert, updates) in moe.experts.iter_mut().zip(&mut train.experts) {
                        update_convolutional_params(expert, updates, lr, momentum, decay, batch);
                    }
                }
                _ => {}
            }
        }
    }

    /// Apply one host-only Adam update to accumulated CPU training buffers.
    /// The global step advances once for this optimizer update, independent of
    /// how many subdivisions contributed to the buffers.
    pub fn update_weights_adam(&mut self, lr: f32, batch: usize) -> Result<()> {
        if !self.network.net.adam || self.network.net.muon {
            return Err(Error::InvalidConfig(
                "CPU Adam update requires [net] adam=1 and muon=0; use update_weights_muon for muon=1"
                    .to_owned(),
            ));
        }
        if !lr.is_finite() || lr < 0.0 {
            return Err(Error::InvalidConfig(
                "CPU Adam learning rate must be finite and non-negative".to_owned(),
            ));
        }
        let step = self
            .adam_step
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConfig("CPU Adam step counter overflowed".to_owned()))?;
        let net = self.network.net.clone();
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
            decay: if net.decoupled_decay {
                net.decay
            } else {
                net.decay * loss_scale
            },
            batch,
            step,
            decoupled_decay: net.decoupled_decay,
        };
        if self.host_adam_state.len() != self.network.layers.len() {
            self.host_adam_state = (0..self.network.layers.len()).map(|_| None).collect();
        }

        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = self.network.net.burn_in;
        for index in 0..self.network.layers.len() {
            let layer = &self.network.layers[index];
            if matches!(layer.kind, LayerKind::Yolo3d(_)) {
                continue; // Weightless head: no Adam or DNOPT state.
            }
            let options = layer.train;
            if !options.updates_at(iteration_num, burn_in) {
                continue;
            }
            if let LayerTrainBuffers::DptHead(train) = &self.train_buffers[index]
                && !train.has_gradient
            {
                continue;
            }
            let layer_adam = AdamParams {
                lr: lr * options.learning_rate_scale,
                ..adam
            };
            if options.grad_centr
                && let (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) =
                    (&self.params[index], &mut self.train_buffers[index])
                && params.batch_norm.is_some()
            {
                let filter_size = params.weights.len() / params.biases.len().max(1);
                if filter_size > 0 {
                    train.weight_updates = darknet_cuda::host_ops::gradient_centralization(
                        &train.weight_updates,
                        filter_size,
                    );
                }
            }
            let state = self.host_adam_state[index].get_or_insert_with(HostAdamLayerState::default);
            apply_host_adam_layer(
                &layer.kind,
                layer.input_shape,
                &mut self.params[index],
                &mut self.train_buffers[index],
                state,
                &layer_adam,
            )?;
            if let LayerTrainBuffers::DptHead(train) = &mut self.train_buffers[index] {
                train.has_gradient = false;
            }
            if options.clip > 0.0
                && let LayerParams::Convolutional(params) = &mut self.params[index]
            {
                params.weights = darknet_cuda::host_ops::constrain(&params.weights, options.clip);
            }
        }
        self.adam_step = step;
        Ok(())
    }

    /// Apply the opt-in hybrid CPU optimizer: Muon for eligible matrix blocks,
    /// AdamW for all remaining trainable parameters.
    pub fn update_weights_muon(&mut self, lr: f32, batch: usize) -> Result<()> {
        let net = self.network.net.clone();
        if !net.muon || net.adam || !net.decoupled_decay {
            return Err(Error::InvalidConfig(
                "CPU Muon update requires [net] muon=1, adam=0, and AdamW fallback".to_owned(),
            ));
        }
        if !lr.is_finite() || lr < 0.0 {
            return Err(Error::InvalidConfig(
                "CPU Muon learning rate must be finite and non-negative".to_owned(),
            ));
        }
        if !net.muon_momentum.is_finite() || !(0.0..1.0).contains(&net.muon_momentum) {
            return Err(Error::InvalidConfig(
                "CPU Muon momentum must be finite and in [0, 1)".to_owned(),
            ));
        }
        if !net.decay.is_finite() || net.decay < 0.0 {
            return Err(Error::InvalidConfig(
                "CPU Muon AdamW decay must be finite and non-negative".to_owned(),
            ));
        }
        let step = self
            .adam_step
            .checked_add(1)
            .ok_or_else(|| Error::InvalidConfig("CPU Muon step counter overflowed".to_owned()))?;
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
            decay: net.decay,
            batch,
            step,
            decoupled_decay: true,
        };
        let layer_count = self.network.layers.len();
        if self.host_adam_state.len() != layer_count {
            self.host_adam_state = (0..layer_count).map(|_| None).collect();
        }
        if self.muon_state.len() != layer_count {
            self.muon_state = (0..layer_count).map(|_| None).collect();
        }

        let iteration_num = self.train_iteration as i64 + 1;
        let burn_in = net.burn_in;
        for index in 0..layer_count {
            let layer = self.network.layers[index].clone();
            if matches!(layer.kind, LayerKind::Yolo3d(_)) {
                continue; // No matrices or AdamW fallback parameters.
            }
            let options = layer.train;
            if !options.updates_at(iteration_num, burn_in) {
                continue;
            }
            if let LayerTrainBuffers::DptHead(train) = &self.train_buffers[index]
                && !train.has_gradient
            {
                continue;
            }
            prepare_muon_layer_gradients(
                &layer.kind,
                options,
                &mut self.params[index],
                &mut self.train_buffers[index],
            )?;
            let layer_adam = AdamParams {
                lr: lr * options.learning_rate_scale,
                ..adam
            };
            let staged = {
                let state = self.muon_state[index].get_or_insert_with(MuonLayerState::default);
                stage_layer(
                    &layer.kind,
                    layer.input_shape,
                    layer.output_shape,
                    &mut self.params[index],
                    &mut self.train_buffers[index],
                    state,
                    layer_adam.lr,
                    batch,
                    loss_scale,
                    net.muon_momentum,
                )?
            };
            let adam_state =
                self.host_adam_state[index].get_or_insert_with(HostAdamLayerState::default);
            if matches!(&layer.kind, LayerKind::DptHead(_)) {
                let (
                    LayerKind::DptHead(spec),
                    LayerParams::DptHead(params),
                    LayerTrainBuffers::DptHead(train),
                ) = (
                    &layer.kind,
                    &mut self.params[index],
                    &mut self.train_buffers[index],
                )
                else {
                    unreachable!("DPT layer kind and train data must match");
                };
                apply_host_adam_dpt(
                    spec,
                    layer.input_shape,
                    params,
                    train,
                    adam_state,
                    &layer_adam,
                    false,
                )?;
            } else {
                apply_host_adam_layer(
                    &layer.kind,
                    layer.input_shape,
                    &mut self.params[index],
                    &mut self.train_buffers[index],
                    adam_state,
                    &layer_adam,
                )?;
            }
            let muon_state = self.muon_state[index].get_or_insert_with(MuonLayerState::default);
            apply_staged_layer(
                &layer.kind,
                layer.input_shape,
                layer.output_shape,
                &mut self.params[index],
                &mut self.train_buffers[index],
                muon_state,
                staged,
            )?;
            if let LayerTrainBuffers::DptHead(train) = &mut self.train_buffers[index] {
                train.has_gradient = false;
            }
            if options.clip > 0.0
                && let LayerParams::Convolutional(params) = &mut self.params[index]
            {
                params.weights = darknet_cuda::host_ops::constrain(&params.weights, options.clip);
            }
        }
        self.adam_step = step;
        Ok(())
    }

}
