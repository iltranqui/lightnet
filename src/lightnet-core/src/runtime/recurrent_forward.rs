use super::*;

pub(super) fn recurrent_sequence_batch(batch: usize, steps: usize) -> Result<usize> {
    if steps == 0 {
        return Err(Error::InvalidConfig(
            "recurrent layer steps cannot be zero".to_owned(),
        ));
    }
    if batch == 0 || !batch.is_multiple_of(steps) {
        return Err(Error::InvalidConfig(format!(
            "recurrent layer batch {} must be a non-zero multiple of steps {}",
            batch, steps
        )));
    }
    Ok(batch / steps)
}

pub(super) fn sequence_slice(
    input: &Tensor,
    step: usize,
    sequence_batch: usize,
    step_shape: TensorShape,
) -> Result<Tensor> {
    if step_shape.batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "recurrent step shape batch mismatch".to_owned(),
        ));
    }
    let step_len = sequence_batch * step_shape.element_count();
    let start = step
        .checked_mul(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent step offset overflow".to_owned()))?;
    let end = start
        .checked_add(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent step end overflow".to_owned()))?;
    if end > input.data.len() {
        return Err(Error::InvalidConfig(
            "recurrent input does not contain all requested steps".to_owned(),
        ));
    }
    Tensor::from_data(step_shape, input.data[start..end].to_vec())
}

pub(super) fn sequence_write(
    output: &mut Tensor,
    step: usize,
    sequence_batch: usize,
    step_output: &Tensor,
) -> Result<()> {
    if step_output.shape.batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "recurrent output step batch mismatch".to_owned(),
        ));
    }
    let step_len = sequence_batch * step_output.shape.element_count();
    let start = step
        .checked_mul(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent output offset overflow".to_owned()))?;
    let end = start
        .checked_add(step_len)
        .ok_or_else(|| Error::InvalidConfig("recurrent output end overflow".to_owned()))?;
    if end > output.data.len() {
        return Err(Error::InvalidConfig(
            "recurrent output does not contain all requested steps".to_owned(),
        ));
    }
    output.data[start..end].copy_from_slice(&step_output.data);
    Ok(())
}

pub(super) fn recurrent_connected_spec(
    output: usize,
    batch_normalize: bool,
    activation: Activation,
) -> ConnectedSpec {
    ConnectedSpec {
        output,
        batch_normalize,
        activation,
    }
}

pub(super) fn recurrent_convolutional_spec(spec: &CrnnSpec, filters: usize) -> ConvolutionalSpec {
    ConvolutionalSpec {
        filters,
        size: spec.size,
        stride_x: spec.stride,
        stride_y: spec.stride,
        dilation: spec.dilation,
        padding: spec.padding,
        groups: spec.groups,
        share_index: None,
        batch_normalize: spec.batch_normalize,
        activation: spec.activation.clone(),
    }
}

pub(super) fn combine_tensors(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape != rhs.shape {
        return Err(Error::InvalidConfig(format!(
            "recurrent tensors have incompatible shapes {} and {}",
            lhs.shape, rhs.shape
        )));
    }
    let mut output = lhs.clone();
    for (dst, src) in output.data.iter_mut().zip(&rhs.data) {
        *dst += *src;
    }
    Ok(output)
}

pub(super) fn add_tensor_inplace(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.shape != src.shape {
        return Err(Error::InvalidConfig(format!(
            "recurrent gradient shapes do not match: {} and {}",
            dst.shape, src.shape
        )));
    }
    for (dst, src) in dst.data.iter_mut().zip(&src.data) {
        *dst += *src;
    }
    Ok(())
}

pub(super) fn forward_gru_layer(
    spec: &GruSpec,
    params: &GruParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "GRU output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input.shape.element_count();
    for (name, connected, expected_inputs) in [
        ("input_z", &params.input_z, inputs),
        ("state_z", &params.state_z, spec.output),
        ("input_r", &params.input_r, inputs),
        ("state_r", &params.state_r, spec.output),
        ("input_h", &params.input_h, inputs),
        ("state_h", &params.state_h, spec.output),
    ] {
        if connected.weights.len() != expected_inputs * spec.output
            || connected.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "GRU {name} parameter shape is inconsistent with its specification"
            )));
        }
    }

    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let input_linear =
        recurrent_connected_spec(spec.output, spec.batch_normalize, Activation::Linear);
    let state_linear =
        recurrent_connected_spec(spec.output, spec.batch_normalize, Activation::Linear);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(state_shape);

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let input_z =
            forward_connected_with_cache(&input_linear, &params.input_z, &step_input, state_shape)
                .0;
        let state_z = forward_connected_with_cache(
            &state_linear,
            &params.state_z,
            &previous_state,
            state_shape,
        )
        .0;
        let input_r =
            forward_connected_with_cache(&input_linear, &params.input_r, &step_input, state_shape)
                .0;
        let state_r = forward_connected_with_cache(
            &state_linear,
            &params.state_r,
            &previous_state,
            state_shape,
        )
        .0;
        let input_h =
            forward_connected_with_cache(&input_linear, &params.input_h, &step_input, state_shape)
                .0;
        let mut z = combine_tensors(&input_z, &state_z)?;
        let mut r = combine_tensors(&input_r, &state_r)?;
        for value in &mut z.data {
            *value = activate(Activation::Logistic, *value);
        }
        for value in &mut r.data {
            *value = activate(Activation::Logistic, *value);
        }
        let mut reset_state = previous_state.clone();
        for (value, reset) in reset_state.data.iter_mut().zip(&r.data) {
            *value *= *reset;
        }
        let state_h =
            forward_connected_with_cache(&state_linear, &params.state_h, &reset_state, state_shape)
                .0;
        let mut candidate = combine_tensors(&input_h, &state_h)?;
        for value in &mut candidate.data {
            *value = activate(Activation::Logistic, *value);
        }
        let mut state = Tensor::zeros(state_shape);
        for (((dst, previous), candidate), update) in state
            .data
            .iter_mut()
            .zip(&previous_state.data)
            .zip(&candidate.data)
            .zip(&z.data)
        {
            *dst = *update * *previous + (1.0 - *update) * *candidate;
        }
        sequence_write(&mut output, step, sequence_batch, &state)?;
        previous_state = state;
    }
    Ok(output)
}

pub(super) fn forward_gru_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    spec: &GruSpec,
    params: &GruParams,
    output_shape: TensorShape,
    layer_index: usize,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    let inputs = input_shape.element_count();
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "GRU output shape is inconsistent with its specification".to_owned(),
        ));
    }
    for (name, projection, expected_inputs) in [
        ("input_z", &params.input_z, inputs),
        ("state_z", &params.state_z, spec.output),
        ("input_r", &params.input_r, inputs),
        ("state_r", &params.state_r, spec.output),
        ("input_h", &params.input_h, inputs),
        ("state_h", &params.state_h, spec.output),
    ] {
        if projection.weights.len() != expected_inputs * spec.output
            || projection.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "GRU {name} parameter shape is inconsistent with its specification"
            )));
        }
    }
    let upload_projection =
        |projection: &ConnectedParams, input_size: usize| -> Result<(B::Tensor, B::Tensor)> {
            let (weights, biases) = if let Some(batch_norm) = &projection.batch_norm {
                validate_batchnorm_params(batch_norm, spec.output)?;
                let mut weights = projection.weights.clone();
                let mut biases = projection.biases.clone();
                for output in 0..spec.output {
                    let factor = batch_norm.scales[output]
                        / (batch_norm.rolling_variance[output] + BATCH_NORM_EPSILON).sqrt();
                    for weight in &mut weights[output * input_size..(output + 1) * input_size] {
                        *weight *= factor;
                    }
                    biases[output] -= batch_norm.rolling_mean[output] * factor;
                }
                (weights, biases)
            } else {
                (projection.weights.clone(), projection.biases.clone())
            };
            let weights = backend
                .upload_f32(&[spec.output, input_size], &weights)
                .map_err(|err| backend_error("upload GRU projection weights", err))?;
            let biases = backend
                .upload_f32(&[spec.output], &biases)
                .map_err(|err| backend_error("upload GRU projection biases", err))?;
            Ok((weights, biases))
        };
    let input_z = upload_projection(&params.input_z, inputs)?;
    let state_z = upload_projection(&params.state_z, spec.output)?;
    let input_r = upload_projection(&params.input_r, inputs)?;
    let state_r = upload_projection(&params.state_r, spec.output)?;
    let input_h = upload_projection(&params.input_h, inputs)?;
    let state_h = upload_projection(&params.state_h, spec.output)?;
    let state_shape = [sequence_batch, spec.output];
    let activation = gpu_activation_kind(layer_index, &Activation::Logistic)?;
    let mut previous = backend
        .zeros_f32(&state_shape)
        .map_err(|err| backend_error("allocate GRU initial state", err))?;
    let mut outputs = Vec::with_capacity(spec.steps);
    for step in 0..spec.steps {
        let step_input = backend
            .sequence_slice_f32(
                input_gpu,
                &tensor_backend_shape(input_shape),
                step,
                spec.steps,
                &[sequence_batch, inputs],
                config,
            )
            .map_err(|err| backend_error("slice GRU input sequence", err))?;
        let project = |x: &B::Tensor, pair: &(B::Tensor, B::Tensor), input_size: usize| {
            backend.connected_f32(
                x,
                &pair.0,
                &pair.1,
                sequence_batch,
                input_size,
                spec.output,
                config,
            )
        };
        let input_z_value = project(&step_input, &input_z, inputs)
            .map_err(|err| backend_error("GRU update input projection", err))?;
        let state_z_value = project(&previous, &state_z, spec.output)
            .map_err(|err| backend_error("GRU update state projection", err))?;
        let update_pre = backend
            .vector_add_f32(&input_z_value, &state_z_value, config)
            .map_err(|err| backend_error("sum GRU update projections", err))?;
        let update = backend
            .activation_f32(&update_pre, activation, config)
            .map_err(|err| backend_error("activate GRU update gate", err))?;
        let input_r_value = project(&step_input, &input_r, inputs)
            .map_err(|err| backend_error("GRU reset input projection", err))?;
        let state_r_value = project(&previous, &state_r, spec.output)
            .map_err(|err| backend_error("GRU reset state projection", err))?;
        let reset_pre = backend
            .vector_add_f32(&input_r_value, &state_r_value, config)
            .map_err(|err| backend_error("sum GRU reset projections", err))?;
        let reset = backend
            .activation_f32(&reset_pre, activation, config)
            .map_err(|err| backend_error("activate GRU reset gate", err))?;
        let reset_state = backend
            .vector_mul_f32(&reset, &previous, &state_shape, config)
            .map_err(|err| backend_error("apply GRU reset gate", err))?;
        let input_h_value = project(&step_input, &input_h, inputs)
            .map_err(|err| backend_error("GRU candidate input projection", err))?;
        let state_h_value = project(&reset_state, &state_h, spec.output)
            .map_err(|err| backend_error("GRU candidate state projection", err))?;
        let candidate_pre = backend
            .vector_add_f32(&input_h_value, &state_h_value, config)
            .map_err(|err| backend_error("sum GRU candidate projections", err))?;
        let candidate = backend
            .activation_f32(&candidate_pre, activation, config)
            .map_err(|err| backend_error("activate GRU candidate", err))?;
        let previous_minus_candidate = backend
            .saxpy_f32(-1.0, &candidate, &previous, config)
            .map_err(|err| backend_error("form GRU state difference", err))?;
        let gated_difference = backend
            .vector_mul_f32(&update, &previous_minus_candidate, &state_shape, config)
            .map_err(|err| backend_error("apply GRU update gate", err))?;
        let state = backend
            .vector_add_f32(&candidate, &gated_difference, config)
            .map_err(|err| backend_error("combine GRU candidate and previous state", err))?;
        outputs.push(state.clone());
        previous = state;
    }
    backend
        .sequence_concat_f32(
            &outputs.iter().collect::<Vec<_>>(),
            &tensor_backend_shape(output_shape),
            sequence_batch * spec.output,
            config,
        )
        .map_err(|err| backend_error("concatenate GRU outputs", err))
}

pub(super) fn recurrent_device_tensor<B: F32TensorOpsBackend>(
    backend: &B,
    tensor: &B::Tensor,
    shape: TensorShape,
    operation: &str,
) -> Result<Tensor>
where
    B::Error: std::fmt::Display,
{
    let data = backend
        .download_f32(tensor)
        .map_err(|err| backend_error(operation, err))?;
    Tensor::from_data(shape, data)
}

pub(super) fn forward_rnn_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    spec: &RnnSpec,
    params: &RnnParams,
    output_shape: TensorShape,
    layer_index: usize,
    capture_host_cache: bool,
    config: ElementwiseLaunchConfig,
) -> Result<(
    B::Tensor,
    Option<RnnForwardCache>,
    GpuRnnBackwardCache<B::Tensor>,
)>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "RNN output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input_shape.element_count();
    if params.input.weights.len() != inputs * spec.hidden
        || params.self_params.weights.len() != spec.hidden * spec.hidden
        || params.output.weights.len() != spec.hidden * spec.output
    {
        return Err(Error::InvalidConfig(
            "RNN parameter shapes are inconsistent with its specification".to_owned(),
        ));
    }

    // Inference batch norm is a per-output affine, so fold rolling statistics
    // into each recurrent projection before upload. The training BN path stays
    // on the host because it uses batch statistics and a separate cache contract.
    let upload_projection = |projection: &ConnectedParams,
                             outputs: usize,
                             projection_inputs: usize,
                             name: &str|
     -> Result<(B::Tensor, B::Tensor)> {
        let (weights, biases) = if let Some(batch_norm) = &projection.batch_norm {
            validate_batchnorm_params(batch_norm, outputs)?;
            let mut weights = projection.weights.clone();
            let mut biases = projection.biases.clone();
            for output in 0..outputs {
                let factor = batch_norm.scales[output]
                    / (batch_norm.rolling_variance[output] + BATCH_NORM_EPSILON).sqrt();
                for weight in
                    &mut weights[output * projection_inputs..(output + 1) * projection_inputs]
                {
                    *weight *= factor;
                }
                biases[output] -= batch_norm.rolling_mean[output] * factor;
            }
            (weights, biases)
        } else {
            (projection.weights.clone(), projection.biases.clone())
        };
        let weights = backend
            .upload_f32(&[outputs, projection_inputs], &weights)
            .map_err(|err| backend_error(&format!("upload RNN {name} weights"), err))?;
        let biases = backend
            .upload_f32(&[outputs], &biases)
            .map_err(|err| backend_error(&format!("upload RNN {name} biases"), err))?;
        Ok((weights, biases))
    };
    let (input_weights, input_biases) =
        upload_projection(&params.input, spec.hidden, inputs, "input")?;
    let (self_weights, self_biases) =
        upload_projection(&params.self_params, spec.hidden, spec.hidden, "recurrent")?;
    let (output_weights, output_biases) =
        upload_projection(&params.output, spec.output, spec.hidden, "output")?;

    let input_activation = gpu_activation_kind(layer_index, &spec.activation)?;
    let self_activation = match spec.logistic {
        1 => gpu_activation_kind(layer_index, &Activation::Logistic)?,
        2 => gpu_activation_kind(layer_index, &Activation::Loggy)?,
        _ => input_activation,
    };
    let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
    let output_step_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let mut output_steps = Vec::with_capacity(spec.steps);
    let mut previous_state_host = capture_host_cache.then(|| Tensor::zeros(hidden_shape));
    let mut previous_state_gpu = backend
        .zeros_f32(&[sequence_batch, spec.hidden])
        .map_err(|err| backend_error("allocate RNN initial state", err))?;
    let mut host_cache = capture_host_cache.then(|| RnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    });
    let mut device_cache = GpuRnnBackwardCache {
        sequence_batch,
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input_gpu = backend
            .sequence_slice_f32(
                input_gpu,
                &tensor_backend_shape(input_shape),
                step,
                spec.steps,
                &[sequence_batch, inputs],
                config,
            )
            .map_err(|err| backend_error("slice RNN input sequence", err))?;
        let input_pre = backend
            .connected_f32(
                &step_input_gpu,
                &input_weights,
                &input_biases,
                sequence_batch,
                inputs,
                spec.hidden,
                config,
            )
            .map_err(|err| backend_error("RNN input projection", err))?;
        let input_output_gpu = backend
            .activation_f32(&input_pre, input_activation, config)
            .map_err(|err| backend_error("activate RNN input projection", err))?;
        let self_pre = backend
            .connected_f32(
                &previous_state_gpu,
                &self_weights,
                &self_biases,
                sequence_batch,
                spec.hidden,
                spec.hidden,
                config,
            )
            .map_err(|err| backend_error("RNN recurrent projection", err))?;
        let self_output_gpu = backend
            .activation_f32(&self_pre, self_activation, config)
            .map_err(|err| backend_error("activate RNN recurrent projection", err))?;
        let mut state_gpu = backend
            .vector_add_f32(&input_output_gpu, &self_output_gpu, config)
            .map_err(|err| backend_error("sum RNN projections", err))?;
        if spec.shortcut {
            state_gpu = backend
                .shortcut_add_f32(&state_gpu, &previous_state_gpu, config)
                .map_err(|err| backend_error("apply RNN shortcut", err))?;
        }
        let output_pre = backend
            .connected_f32(
                &state_gpu,
                &output_weights,
                &output_biases,
                sequence_batch,
                spec.hidden,
                spec.output,
                config,
            )
            .map_err(|err| backend_error("RNN output projection", err))?;
        let step_output_gpu = backend
            .activation_f32(&output_pre, input_activation, config)
            .map_err(|err| backend_error("activate RNN output projection", err))?;

        device_cache.input_activation_inputs.push(input_pre.clone());
        device_cache.self_activation_inputs.push(self_pre.clone());
        device_cache
            .output_activation_inputs
            .push(output_pre.clone());
        device_cache
            .previous_states
            .push(previous_state_gpu.clone());
        device_cache.states.push(state_gpu.clone());
        if let Some(cache) = &mut host_cache {
            let input_output = recurrent_device_tensor(
                backend,
                &input_output_gpu,
                hidden_shape,
                "download RNN input projection",
            )?;
            let input_activation_input = recurrent_device_tensor(
                backend,
                &input_pre,
                hidden_shape,
                "download RNN input pre-activation",
            )?;
            let self_output = recurrent_device_tensor(
                backend,
                &self_output_gpu,
                hidden_shape,
                "download RNN recurrent projection",
            )?;
            let self_activation_input = recurrent_device_tensor(
                backend,
                &self_pre,
                hidden_shape,
                "download RNN recurrent pre-activation",
            )?;
            let state = recurrent_device_tensor(
                backend,
                &state_gpu,
                hidden_shape,
                "download RNN hidden state",
            )?;
            let output_activation_input = recurrent_device_tensor(
                backend,
                &output_pre,
                output_step_shape,
                "download RNN output pre-activation",
            )?;
            let step_output = recurrent_device_tensor(
                backend,
                &step_output_gpu,
                output_step_shape,
                "download RNN output",
            )?;
            cache.input_outputs.push(input_output);
            cache.input_activation_inputs.push(input_activation_input);
            cache.self_outputs.push(self_output);
            cache.self_activation_inputs.push(self_activation_input);
            cache.previous_states.push(
                previous_state_host
                    .as_ref()
                    .expect("capturing RNN host cache")
                    .clone(),
            );
            cache.states.push(state.clone());
            cache.output_activation_inputs.push(output_activation_input);
            cache.outputs.push(step_output);
            previous_state_host = Some(state);
        }
        output_steps.push(step_output_gpu.clone());
        previous_state_gpu = state_gpu;
    }

    let output_step_refs: Vec<_> = output_steps.iter().collect();
    let output_gpu = backend
        .sequence_concat_f32(
            &output_step_refs,
            &tensor_backend_shape(output_shape),
            sequence_batch * spec.output,
            config,
        )
        .map_err(|err| backend_error("assemble RNN output sequence", err))?;
    Ok((output_gpu, host_cache, device_cache))
}

pub(super) fn forward_rnn_layer(
    spec: &RnnSpec,
    params: &RnnParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, RnnForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "RNN output shape is inconsistent with its specification".to_owned(),
        ));
    }
    if params.input.weights.len() != input.shape.element_count() * spec.hidden
        || params.self_params.weights.len() != spec.hidden * spec.hidden
        || params.output.weights.len() != spec.hidden * spec.output
    {
        return Err(Error::InvalidConfig(
            "RNN parameter shapes are inconsistent with its specification".to_owned(),
        ));
    }

    let input_spec =
        recurrent_connected_spec(spec.hidden, spec.batch_normalize, spec.activation.clone());
    let self_activation = match spec.logistic {
        1 => Activation::Logistic,
        2 => Activation::Loggy,
        _ => spec.activation.clone(),
    };
    let self_spec = recurrent_connected_spec(spec.hidden, spec.batch_normalize, self_activation);
    let output_spec =
        recurrent_connected_spec(spec.output, spec.batch_normalize, spec.activation.clone());
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
    let output_step_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(hidden_shape);
    let mut cache = RnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (input_output, input_activation_input) =
            forward_connected_with_cache(&input_spec, &params.input, &step_input, hidden_shape);
        let (self_output, self_activation_input) = forward_connected_with_cache(
            &self_spec,
            &params.self_params,
            &previous_state,
            hidden_shape,
        );
        let mut state = combine_tensors(&input_output, &self_output)?;
        if spec.shortcut {
            add_tensor_inplace(&mut state, &previous_state)?;
        }
        let (step_output, output_activation_input) =
            forward_connected_with_cache(&output_spec, &params.output, &state, output_step_shape);
        sequence_write(&mut output, step, sequence_batch, &step_output)?;
        cache.input_outputs.push(input_output);
        cache.input_activation_inputs.push(
            input_activation_input.expect("connected forward always records an activation input"),
        );
        cache.self_outputs.push(self_output);
        cache.self_activation_inputs.push(
            self_activation_input.expect("connected forward always records an activation input"),
        );
        cache.previous_states.push(previous_state);
        cache.states.push(state.clone());
        cache.output_activation_inputs.push(
            output_activation_input.expect("connected forward always records an activation input"),
        );
        cache.outputs.push(step_output);
        previous_state = state;
    }

    Ok((output, cache))
}

pub(super) fn forward_lstm_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LstmSpec,
    params: &LstmParams,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    layer_index: usize,
    capture_host_cache: bool,
    config: ElementwiseLaunchConfig,
) -> Result<(
    B::Tensor,
    Option<LstmForwardCache>,
    GpuLstmBackwardCache<B::Tensor>,
)>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "LSTM output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input_shape.element_count();
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let upload_projection = |projection: &ConnectedParams, input_features: usize, name: &str| {
        if projection.weights.len() != spec.output * input_features
            || projection.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "LSTM {name} projection parameter shapes are inconsistent with its specification"
            )));
        }
        let (weights, biases) = if let Some(batch_norm) = &projection.batch_norm {
            validate_batchnorm_params(batch_norm, spec.output)?;
            let mut weights = projection.weights.clone();
            let mut biases = projection.biases.clone();
            for output in 0..spec.output {
                let factor = batch_norm.scales[output]
                    / (batch_norm.rolling_variance[output] + BATCH_NORM_EPSILON).sqrt();
                for weight in &mut weights[output * input_features..(output + 1) * input_features] {
                    *weight *= factor;
                }
                biases[output] -= batch_norm.rolling_mean[output] * factor;
            }
            (weights, biases)
        } else {
            (projection.weights.clone(), projection.biases.clone())
        };
        let weights = backend
            .upload_f32(&[spec.output, input_features], &weights)
            .map_err(|err| backend_error("upload LSTM projection weights", err))?;
        let biases = backend
            .upload_f32(&[spec.output], &biases)
            .map_err(|err| backend_error("upload LSTM projection biases", err))?;
        Ok((weights, biases))
    };
    let (wf_weights, wf_biases) = upload_projection(&params.wf, spec.output, "forget recurrent")?;
    let (wi_weights, wi_biases) = upload_projection(&params.wi, spec.output, "input recurrent")?;
    let (wg_weights, wg_biases) =
        upload_projection(&params.wg, spec.output, "candidate recurrent")?;
    let (wo_weights, wo_biases) = upload_projection(&params.wo, spec.output, "output recurrent")?;
    let (uf_weights, uf_biases) = upload_projection(&params.uf, inputs, "forget input")?;
    let (ui_weights, ui_biases) = upload_projection(&params.ui, inputs, "input input")?;
    let (ug_weights, ug_biases) = upload_projection(&params.ug, inputs, "candidate input")?;
    let (uo_weights, uo_biases) = upload_projection(&params.uo, inputs, "output input")?;
    let project = |x: &B::Tensor, weights: &B::Tensor, biases: &B::Tensor, features: usize| {
        backend
            .connected_f32(
                x,
                weights,
                biases,
                sequence_batch,
                features,
                spec.output,
                config,
            )
            .map_err(|err| backend_error("LSTM connected projection", err))
    };
    let logistic = gpu_activation_kind(layer_index, &Activation::Logistic)?;
    let tanh = gpu_activation_kind(layer_index, &Activation::Tanh)?;
    let mut state_outputs = Vec::with_capacity(spec.steps);
    let mut previous_state_gpu = backend
        .zeros_f32(&tensor_backend_shape(state_shape))
        .map_err(|err| backend_error("allocate LSTM initial state", err))?;
    let mut previous_cell_gpu = backend
        .zeros_f32(&tensor_backend_shape(state_shape))
        .map_err(|err| backend_error("allocate LSTM initial cell", err))?;
    let mut previous_state_host = capture_host_cache.then(|| Tensor::zeros(state_shape));
    let mut previous_cell_host = capture_host_cache.then(|| Tensor::zeros(state_shape));
    let mut host_cache = capture_host_cache.then(|| LstmForwardCache {
        sequence_batch,
        wf_outputs: Vec::with_capacity(spec.steps),
        wf_activation_inputs: Vec::with_capacity(spec.steps),
        wi_outputs: Vec::with_capacity(spec.steps),
        wi_activation_inputs: Vec::with_capacity(spec.steps),
        wg_outputs: Vec::with_capacity(spec.steps),
        wg_activation_inputs: Vec::with_capacity(spec.steps),
        wo_outputs: Vec::with_capacity(spec.steps),
        wo_activation_inputs: Vec::with_capacity(spec.steps),
        uf_outputs: Vec::with_capacity(spec.steps),
        uf_activation_inputs: Vec::with_capacity(spec.steps),
        ui_outputs: Vec::with_capacity(spec.steps),
        ui_activation_inputs: Vec::with_capacity(spec.steps),
        ug_outputs: Vec::with_capacity(spec.steps),
        ug_activation_inputs: Vec::with_capacity(spec.steps),
        uo_outputs: Vec::with_capacity(spec.steps),
        uo_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        previous_cells: Vec::with_capacity(spec.steps),
        forget: Vec::with_capacity(spec.steps),
        input_gate: Vec::with_capacity(spec.steps),
        candidate: Vec::with_capacity(spec.steps),
        output_gate: Vec::with_capacity(spec.steps),
        cells: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
    });
    let mut device_cache = GpuLstmBackwardCache {
        sequence_batch,
        previous_states: Vec::with_capacity(spec.steps),
        previous_cells: Vec::with_capacity(spec.steps),
        forget: Vec::with_capacity(spec.steps),
        input_gate: Vec::with_capacity(spec.steps),
        candidate: Vec::with_capacity(spec.steps),
        output_gate: Vec::with_capacity(spec.steps),
        cells: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input_gpu = backend
            .sequence_slice_f32(
                input_gpu,
                &tensor_backend_shape(input_shape),
                step,
                spec.steps,
                &[sequence_batch, inputs],
                config,
            )
            .map_err(|err| backend_error("slice LSTM input sequence", err))?;
        let wf_gpu = project(&previous_state_gpu, &wf_weights, &wf_biases, spec.output)?;
        let wi_gpu = project(&previous_state_gpu, &wi_weights, &wi_biases, spec.output)?;
        let wg_gpu = project(&previous_state_gpu, &wg_weights, &wg_biases, spec.output)?;
        let wo_gpu = project(&previous_state_gpu, &wo_weights, &wo_biases, spec.output)?;
        let uf_gpu = project(&step_input_gpu, &uf_weights, &uf_biases, inputs)?;
        let ui_gpu = project(&step_input_gpu, &ui_weights, &ui_biases, inputs)?;
        let ug_gpu = project(&step_input_gpu, &ug_weights, &ug_biases, inputs)?;
        let uo_gpu = project(&step_input_gpu, &uo_weights, &uo_biases, inputs)?;
        let forget_pre = backend
            .vector_add_f32(&wf_gpu, &uf_gpu, config)
            .map_err(|err| backend_error("sum LSTM forget projections", err))?;
        let input_pre = backend
            .vector_add_f32(&wi_gpu, &ui_gpu, config)
            .map_err(|err| backend_error("sum LSTM input projections", err))?;
        let candidate_pre = backend
            .vector_add_f32(&wg_gpu, &ug_gpu, config)
            .map_err(|err| backend_error("sum LSTM candidate projections", err))?;
        let output_pre = backend
            .vector_add_f32(&wo_gpu, &uo_gpu, config)
            .map_err(|err| backend_error("sum LSTM output projections", err))?;
        let forget_gpu = backend
            .activation_f32(&forget_pre, logistic, config)
            .map_err(|err| backend_error("activate LSTM forget gate", err))?;
        let input_gate_gpu = backend
            .activation_f32(&input_pre, logistic, config)
            .map_err(|err| backend_error("activate LSTM input gate", err))?;
        let candidate_gpu = backend
            .activation_f32(&candidate_pre, tanh, config)
            .map_err(|err| backend_error("activate LSTM candidate gate", err))?;
        let output_gate_gpu = backend
            .activation_f32(&output_pre, logistic, config)
            .map_err(|err| backend_error("activate LSTM output gate", err))?;
        let forgotten_cell = backend
            .vector_mul_f32(
                &forget_gpu,
                &previous_cell_gpu,
                &tensor_backend_shape(state_shape),
                config,
            )
            .map_err(|err| backend_error("multiply LSTM forget gate", err))?;
        let input_candidate = backend
            .vector_mul_f32(
                &input_gate_gpu,
                &candidate_gpu,
                &tensor_backend_shape(state_shape),
                config,
            )
            .map_err(|err| backend_error("multiply LSTM input and candidate gates", err))?;
        let cell_gpu = backend
            .vector_add_f32(&forgotten_cell, &input_candidate, config)
            .map_err(|err| backend_error("sum LSTM cell terms", err))?;
        let tanh_cell_gpu = backend
            .activation_f32(&cell_gpu, tanh, config)
            .map_err(|err| backend_error("activate LSTM cell", err))?;
        let state_gpu = backend
            .vector_mul_f32(
                &output_gate_gpu,
                &tanh_cell_gpu,
                &tensor_backend_shape(state_shape),
                config,
            )
            .map_err(|err| backend_error("multiply LSTM output gate", err))?;

        let state_host = if host_cache.is_some() {
            Some(recurrent_device_tensor(
                backend,
                &state_gpu,
                state_shape,
                "download LSTM state",
            )?)
        } else {
            None
        };
        state_outputs.push(state_gpu.clone());
        let cell_host = if host_cache.is_some() {
            Some(recurrent_device_tensor(
                backend,
                &cell_gpu,
                state_shape,
                "download LSTM cell",
            )?)
        } else {
            None
        };
        device_cache
            .previous_states
            .push(previous_state_gpu.clone());
        device_cache.previous_cells.push(previous_cell_gpu.clone());
        device_cache.forget.push(forget_gpu.clone());
        device_cache.input_gate.push(input_gate_gpu.clone());
        device_cache.candidate.push(candidate_gpu.clone());
        device_cache.output_gate.push(output_gate_gpu.clone());
        device_cache.cells.push(cell_gpu.clone());

        if let Some(cache) = &mut host_cache {
            let wf = recurrent_device_tensor(
                backend,
                &wf_gpu,
                state_shape,
                "download LSTM forget recurrent projection",
            )?;
            let wi = recurrent_device_tensor(
                backend,
                &wi_gpu,
                state_shape,
                "download LSTM input recurrent projection",
            )?;
            let wg = recurrent_device_tensor(
                backend,
                &wg_gpu,
                state_shape,
                "download LSTM candidate recurrent projection",
            )?;
            let wo = recurrent_device_tensor(
                backend,
                &wo_gpu,
                state_shape,
                "download LSTM output recurrent projection",
            )?;
            let uf = recurrent_device_tensor(
                backend,
                &uf_gpu,
                state_shape,
                "download LSTM forget input projection",
            )?;
            let ui = recurrent_device_tensor(
                backend,
                &ui_gpu,
                state_shape,
                "download LSTM input input projection",
            )?;
            let ug = recurrent_device_tensor(
                backend,
                &ug_gpu,
                state_shape,
                "download LSTM candidate input projection",
            )?;
            let uo = recurrent_device_tensor(
                backend,
                &uo_gpu,
                state_shape,
                "download LSTM output input projection",
            )?;
            let forget = recurrent_device_tensor(
                backend,
                &forget_gpu,
                state_shape,
                "download LSTM forget gate",
            )?;
            let input_gate = recurrent_device_tensor(
                backend,
                &input_gate_gpu,
                state_shape,
                "download LSTM input gate",
            )?;
            let candidate = recurrent_device_tensor(
                backend,
                &candidate_gpu,
                state_shape,
                "download LSTM candidate gate",
            )?;
            let output_gate = recurrent_device_tensor(
                backend,
                &output_gate_gpu,
                state_shape,
                "download LSTM output gate",
            )?;
            let cell = cell_host.as_ref().expect("capturing host cache");
            cache.wf_outputs.push(wf.clone());
            cache.wf_activation_inputs.push(wf);
            cache.wi_outputs.push(wi.clone());
            cache.wi_activation_inputs.push(wi);
            cache.wg_outputs.push(wg.clone());
            cache.wg_activation_inputs.push(wg);
            cache.wo_outputs.push(wo.clone());
            cache.wo_activation_inputs.push(wo);
            cache.uf_outputs.push(uf.clone());
            cache.uf_activation_inputs.push(uf);
            cache.ui_outputs.push(ui.clone());
            cache.ui_activation_inputs.push(ui);
            cache.ug_outputs.push(ug.clone());
            cache.ug_activation_inputs.push(ug);
            cache.uo_outputs.push(uo.clone());
            cache.uo_activation_inputs.push(uo);
            cache.previous_states.push(
                previous_state_host
                    .as_ref()
                    .expect("capturing host cache")
                    .clone(),
            );
            cache.previous_cells.push(
                previous_cell_host
                    .as_ref()
                    .expect("capturing host cache")
                    .clone(),
            );
            cache.forget.push(forget);
            cache.input_gate.push(input_gate);
            cache.candidate.push(candidate);
            cache.output_gate.push(output_gate);
            cache.cells.push(cell.clone());
            cache
                .states
                .push(state_host.as_ref().expect("capturing host cache").clone());
        }
        if host_cache.is_some() {
            previous_state_host = state_host;
            previous_cell_host = cell_host;
        }
        previous_state_gpu = state_gpu;
        previous_cell_gpu = cell_gpu;
    }

    let state_output_refs: Vec<_> = state_outputs.iter().collect();
    let output_gpu = backend
        .sequence_concat_f32(
            &state_output_refs,
            &tensor_backend_shape(output_shape),
            sequence_batch * spec.output,
            config,
        )
        .map_err(|err| backend_error("assemble LSTM output sequence", err))?;
    Ok((output_gpu, host_cache, device_cache))
}

pub(super) fn forward_lstm_layer(
    spec: &LstmSpec,
    params: &LstmParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, LstmForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if output_shape.width != 1 || output_shape.height != 1 || output_shape.channels != spec.output {
        return Err(Error::InvalidConfig(
            "LSTM output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let inputs = input.shape.element_count();
    let expected = inputs * spec.output;
    for nested in [
        &params.wf, &params.wi, &params.wg, &params.wo, &params.uf, &params.ui, &params.ug,
        &params.uo,
    ] {
        if nested.weights.len() != expected && nested.weights.len() != spec.output * spec.output {
            return Err(Error::InvalidConfig(
                "LSTM parameter shapes are inconsistent with its specification".to_owned(),
            ));
        }
    }

    let linear =
        |output| recurrent_connected_spec(output, spec.batch_normalize, Activation::Linear);
    let wf_spec = linear(spec.output);
    let wi_spec = linear(spec.output);
    let wg_spec = linear(spec.output);
    let wo_spec = linear(spec.output);
    let uf_spec = linear(spec.output);
    let ui_spec = linear(spec.output);
    let ug_spec = linear(spec.output);
    let uo_spec = linear(spec.output);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(state_shape);
    let mut previous_cell = Tensor::zeros(state_shape);
    let mut cache = LstmForwardCache {
        sequence_batch,
        wf_outputs: Vec::with_capacity(spec.steps),
        wf_activation_inputs: Vec::with_capacity(spec.steps),
        wi_outputs: Vec::with_capacity(spec.steps),
        wi_activation_inputs: Vec::with_capacity(spec.steps),
        wg_outputs: Vec::with_capacity(spec.steps),
        wg_activation_inputs: Vec::with_capacity(spec.steps),
        wo_outputs: Vec::with_capacity(spec.steps),
        wo_activation_inputs: Vec::with_capacity(spec.steps),
        uf_outputs: Vec::with_capacity(spec.steps),
        uf_activation_inputs: Vec::with_capacity(spec.steps),
        ui_outputs: Vec::with_capacity(spec.steps),
        ui_activation_inputs: Vec::with_capacity(spec.steps),
        ug_outputs: Vec::with_capacity(spec.steps),
        ug_activation_inputs: Vec::with_capacity(spec.steps),
        uo_outputs: Vec::with_capacity(spec.steps),
        uo_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        previous_cells: Vec::with_capacity(spec.steps),
        forget: Vec::with_capacity(spec.steps),
        input_gate: Vec::with_capacity(spec.steps),
        candidate: Vec::with_capacity(spec.steps),
        output_gate: Vec::with_capacity(spec.steps),
        cells: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (wf_output, wf_activation_input) =
            forward_connected_with_cache(&wf_spec, &params.wf, &previous_state, state_shape);
        let (wi_output, wi_activation_input) =
            forward_connected_with_cache(&wi_spec, &params.wi, &previous_state, state_shape);
        let (wg_output, wg_activation_input) =
            forward_connected_with_cache(&wg_spec, &params.wg, &previous_state, state_shape);
        let (wo_output, wo_activation_input) =
            forward_connected_with_cache(&wo_spec, &params.wo, &previous_state, state_shape);
        let (uf_output, uf_activation_input) =
            forward_connected_with_cache(&uf_spec, &params.uf, &step_input, state_shape);
        let (ui_output, ui_activation_input) =
            forward_connected_with_cache(&ui_spec, &params.ui, &step_input, state_shape);
        let (ug_output, ug_activation_input) =
            forward_connected_with_cache(&ug_spec, &params.ug, &step_input, state_shape);
        let (uo_output, uo_activation_input) =
            forward_connected_with_cache(&uo_spec, &params.uo, &step_input, state_shape);

        let mut forget = Tensor::zeros(state_shape);
        let mut input_gate = Tensor::zeros(state_shape);
        let mut candidate = Tensor::zeros(state_shape);
        let mut output_gate = Tensor::zeros(state_shape);
        for index in 0..forget.data.len() {
            forget.data[index] = activate(
                Activation::Logistic,
                wf_output.data[index] + uf_output.data[index],
            );
            input_gate.data[index] = activate(
                Activation::Logistic,
                wi_output.data[index] + ui_output.data[index],
            );
            candidate.data[index] = activate(
                Activation::Tanh,
                wg_output.data[index] + ug_output.data[index],
            );
            output_gate.data[index] = activate(
                Activation::Logistic,
                wo_output.data[index] + uo_output.data[index],
            );
        }
        let mut cell = Tensor::zeros(state_shape);
        let mut state = Tensor::zeros(state_shape);
        for index in 0..cell.data.len() {
            cell.data[index] = forget.data[index] * previous_cell.data[index]
                + input_gate.data[index] * candidate.data[index];
            state.data[index] =
                output_gate.data[index] * activate(Activation::Tanh, cell.data[index]);
        }
        sequence_write(&mut output, step, sequence_batch, &state)?;

        cache.wf_outputs.push(wf_output);
        cache
            .wf_activation_inputs
            .push(wf_activation_input.expect("connected forward cache"));
        cache.wi_outputs.push(wi_output);
        cache
            .wi_activation_inputs
            .push(wi_activation_input.expect("connected forward cache"));
        cache.wg_outputs.push(wg_output);
        cache
            .wg_activation_inputs
            .push(wg_activation_input.expect("connected forward cache"));
        cache.wo_outputs.push(wo_output);
        cache
            .wo_activation_inputs
            .push(wo_activation_input.expect("connected forward cache"));
        cache.uf_outputs.push(uf_output);
        cache
            .uf_activation_inputs
            .push(uf_activation_input.expect("connected forward cache"));
        cache.ui_outputs.push(ui_output);
        cache
            .ui_activation_inputs
            .push(ui_activation_input.expect("connected forward cache"));
        cache.ug_outputs.push(ug_output);
        cache
            .ug_activation_inputs
            .push(ug_activation_input.expect("connected forward cache"));
        cache.uo_outputs.push(uo_output);
        cache
            .uo_activation_inputs
            .push(uo_activation_input.expect("connected forward cache"));
        cache.previous_states.push(previous_state);
        cache.previous_cells.push(previous_cell);
        cache.forget.push(forget);
        cache.input_gate.push(input_gate);
        cache.candidate.push(candidate);
        cache.output_gate.push(output_gate);
        cache.cells.push(cell.clone());
        cache.states.push(state.clone());
        previous_state = state;
        previous_cell = cell;
    }

    Ok((output, cache))
}

pub(super) fn forward_crnn_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &CrnnSpec,
    params: &CrnnParams,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    layer_index: usize,
    capture_host_cache: bool,
    config: ElementwiseLaunchConfig,
) -> Result<(
    B::Tensor,
    Option<CrnnForwardCache>,
    GpuCrnnBackwardCache<B::Tensor>,
)>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    let input_width =
        convolution_output_dim(input_shape.width, spec.padding, spec.size, spec.stride);
    let input_height =
        convolution_output_dim(input_shape.height, spec.padding, spec.size, spec.stride);
    let hidden_shape = TensorShape::new(input_width, input_height, spec.hidden, sequence_batch);
    let output_step_shape =
        TensorShape::new(input_width, input_height, spec.output, sequence_batch);
    let input_step_shape = TensorShape::new(
        input_shape.width,
        input_shape.height,
        input_shape.channels,
        sequence_batch,
    );
    if output_shape != TensorShape::new(input_width, input_height, spec.output, input_shape.batch) {
        return Err(Error::InvalidConfig(
            "CRNN output shape is inconsistent with its specification".to_owned(),
        ));
    }

    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let groups = spec.groups.max(1);
    let upload_conv_params =
        |conv: &ConvolutionalParams, conv_spec: &ConvolutionalSpec, channels: usize, name: &str| {
            let grouped_channels = channels / groups;
            let expected = conv_spec.filters * grouped_channels * conv_spec.size * conv_spec.size;
            if conv.weights.len() != expected || conv.biases.len() != conv_spec.filters {
                return Err(Error::InvalidConfig(format!(
                    "CRNN {name} parameter shapes are inconsistent with its specification"
                )));
            }
            let (weights, biases) = if let Some(batch_norm) = &conv.batch_norm {
                validate_batchnorm_params(batch_norm, conv_spec.filters)?;
                let mut weights = conv.weights.clone();
                let mut biases = conv.biases.clone();
                let weights_per_filter = grouped_channels * conv_spec.size * conv_spec.size;
                for filter in 0..conv_spec.filters {
                    let factor = batch_norm.scales[filter]
                        / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
                    let start = filter * weights_per_filter;
                    for weight in &mut weights[start..start + weights_per_filter] {
                        *weight *= factor;
                    }
                    biases[filter] -= batch_norm.rolling_mean[filter] * factor;
                }
                (weights, biases)
            } else {
                (conv.weights.clone(), conv.biases.clone())
            };
            let weights = backend
                .upload_f32(
                    &[
                        conv_spec.filters,
                        grouped_channels,
                        conv_spec.size,
                        conv_spec.size,
                    ],
                    &weights,
                )
                .map_err(|err| backend_error("upload CRNN convolution weights", err))?;
            let biases = backend
                .upload_f32(&[conv_spec.filters], &biases)
                .map_err(|err| backend_error("upload CRNN convolution biases", err))?;
            Ok((weights, biases))
        };
    let (input_weights, input_biases) =
        upload_conv_params(&params.input, &input_spec, input_shape.channels, "input")?;
    let (self_weights, self_biases) =
        upload_conv_params(&params.self_params, &self_spec, spec.hidden, "recurrent")?;
    let (output_weights, output_biases) =
        upload_conv_params(&params.output, &output_spec, spec.hidden, "output")?;
    let activation = gpu_activation_kind(layer_index, &spec.activation)?;
    let conv_forward = |conv_input: &B::Tensor,
                        weights: &B::Tensor,
                        biases: &B::Tensor,
                        input_channels: usize,
                        input_height: usize,
                        input_width: usize,
                        filters: usize,
                        conv_output_shape: TensorShape|
     -> Result<(B::Tensor, B::Tensor)> {
        let raw = backend
            .conv2d_f32(
                conv_input,
                weights,
                &tensor_backend_shape(conv_output_shape),
                input_channels,
                input_height,
                input_width,
                filters,
                spec.size,
                spec.stride,
                spec.stride,
                spec.dilation,
                spec.padding,
                groups,
                config,
            )
            .map_err(|err| backend_error("CRNN convolution forward", err))?;
        let pre_activation = backend
            .bias_add_f32(
                &raw,
                biases,
                filters,
                conv_output_shape.width * conv_output_shape.height,
                config,
            )
            .map_err(|err| backend_error("CRNN convolution bias", err))?;
        let output = backend
            .activation_f32(&pre_activation, activation, config)
            .map_err(|err| backend_error("CRNN convolution activation", err))?;
        Ok((pre_activation, output))
    };

    let mut output_steps = Vec::with_capacity(spec.steps);
    let mut previous_state_host = capture_host_cache.then(|| Tensor::zeros(hidden_shape));
    let mut previous_state_gpu = backend
        .zeros_f32(&tensor_backend_shape(hidden_shape))
        .map_err(|err| backend_error("allocate CRNN initial state", err))?;
    let mut host_cache = capture_host_cache.then(|| CrnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    });
    let mut device_cache = GpuCrnnBackwardCache {
        sequence_batch,
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input_gpu = backend
            .sequence_slice_f32(
                input_gpu,
                &tensor_backend_shape(input_shape),
                step,
                spec.steps,
                &tensor_backend_shape(input_step_shape),
                config,
            )
            .map_err(|err| backend_error("slice CRNN input sequence", err))?;
        let (input_pre, input_output_gpu) = conv_forward(
            &step_input_gpu,
            &input_weights,
            &input_biases,
            input_shape.channels,
            input_shape.height,
            input_shape.width,
            spec.hidden,
            hidden_shape,
        )?;
        let (self_pre, self_output_gpu) = conv_forward(
            &previous_state_gpu,
            &self_weights,
            &self_biases,
            spec.hidden,
            input_height,
            input_width,
            spec.hidden,
            hidden_shape,
        )?;
        let mut state_gpu = backend
            .vector_add_f32(&input_output_gpu, &self_output_gpu, config)
            .map_err(|err| backend_error("sum CRNN projections", err))?;
        if spec.shortcut {
            state_gpu = backend
                .vector_add_f32(&state_gpu, &previous_state_gpu, config)
                .map_err(|err| backend_error("apply CRNN shortcut", err))?;
        }
        let (output_pre, step_output_gpu) = conv_forward(
            &state_gpu,
            &output_weights,
            &output_biases,
            spec.hidden,
            input_height,
            input_width,
            spec.output,
            output_step_shape,
        )?;

        device_cache.input_activation_inputs.push(input_pre.clone());
        device_cache.self_activation_inputs.push(self_pre.clone());
        device_cache
            .output_activation_inputs
            .push(output_pre.clone());
        device_cache
            .previous_states
            .push(previous_state_gpu.clone());
        device_cache.states.push(state_gpu.clone());
        if let Some(cache) = &mut host_cache {
            let input_output = recurrent_device_tensor(
                backend,
                &input_output_gpu,
                hidden_shape,
                "download CRNN input projection",
            )?;
            let input_activation_input = recurrent_device_tensor(
                backend,
                &input_pre,
                hidden_shape,
                "download CRNN input pre-activation",
            )?;
            let self_output = recurrent_device_tensor(
                backend,
                &self_output_gpu,
                hidden_shape,
                "download CRNN recurrent projection",
            )?;
            let self_activation_input = recurrent_device_tensor(
                backend,
                &self_pre,
                hidden_shape,
                "download CRNN recurrent pre-activation",
            )?;
            let state = recurrent_device_tensor(
                backend,
                &state_gpu,
                hidden_shape,
                "download CRNN hidden state",
            )?;
            let output_activation_input = recurrent_device_tensor(
                backend,
                &output_pre,
                output_step_shape,
                "download CRNN output pre-activation",
            )?;
            let step_output = recurrent_device_tensor(
                backend,
                &step_output_gpu,
                output_step_shape,
                "download CRNN output",
            )?;
            cache.input_outputs.push(input_output);
            cache.input_activation_inputs.push(input_activation_input);
            cache.self_outputs.push(self_output);
            cache.self_activation_inputs.push(self_activation_input);
            cache.previous_states.push(
                previous_state_host
                    .as_ref()
                    .expect("capturing CRNN host cache")
                    .clone(),
            );
            cache.states.push(state.clone());
            cache.output_activation_inputs.push(output_activation_input);
            cache.outputs.push(step_output);
            previous_state_host = Some(state);
        }
        output_steps.push(step_output_gpu.clone());
        previous_state_gpu = state_gpu;
    }

    let output_step_refs: Vec<_> = output_steps.iter().collect();
    let output_gpu = backend
        .sequence_concat_f32(
            &output_step_refs,
            &tensor_backend_shape(output_shape),
            sequence_batch * output_step_shape.element_count(),
            config,
        )
        .map_err(|err| backend_error("assemble CRNN output sequence", err))?;
    Ok((output_gpu, host_cache, device_cache))
}

pub(super) fn forward_crnn_layer(
    spec: &CrnnSpec,
    params: &CrnnParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, CrnnForwardCache)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    let input_width =
        convolution_output_dim(input.shape.width, spec.padding, spec.size, spec.stride);
    let input_height =
        convolution_output_dim(input.shape.height, spec.padding, spec.size, spec.stride);
    let hidden_shape = TensorShape::new(input_width, input_height, spec.hidden, sequence_batch);
    let output_step_shape =
        TensorShape::new(input_width, input_height, spec.output, sequence_batch);
    if output_shape != TensorShape::new(input_width, input_height, spec.output, input.shape.batch) {
        return Err(Error::InvalidConfig(
            "CRNN output shape is inconsistent with its specification".to_owned(),
        ));
    }
    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let mut output = Tensor::zeros(output_shape);
    let mut previous_state = Tensor::zeros(hidden_shape);
    let mut cache = CrnnForwardCache {
        sequence_batch,
        input_outputs: Vec::with_capacity(spec.steps),
        input_activation_inputs: Vec::with_capacity(spec.steps),
        self_outputs: Vec::with_capacity(spec.steps),
        self_activation_inputs: Vec::with_capacity(spec.steps),
        previous_states: Vec::with_capacity(spec.steps),
        states: Vec::with_capacity(spec.steps),
        output_activation_inputs: Vec::with_capacity(spec.steps),
        outputs: Vec::with_capacity(spec.steps),
    };

    for step in 0..spec.steps {
        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (input_output, input_activation_input) =
            forward_convolutional_with_cache(&input_spec, &params.input, &step_input, hidden_shape);
        let (self_output, self_activation_input) = forward_convolutional_with_cache(
            &self_spec,
            &params.self_params,
            &previous_state,
            hidden_shape,
        );
        let mut state = combine_tensors(&input_output, &self_output)?;
        if spec.shortcut {
            add_tensor_inplace(&mut state, &previous_state)?;
        }
        let (step_output, output_activation_input) = forward_convolutional_with_cache(
            &output_spec,
            &params.output,
            &state,
            output_step_shape,
        );
        sequence_write(&mut output, step, sequence_batch, &step_output)?;
        cache.input_outputs.push(input_output);
        cache.input_activation_inputs.push(
            input_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.self_outputs.push(self_output);
        cache.self_activation_inputs.push(
            self_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.previous_states.push(previous_state);
        cache.states.push(state.clone());
        cache.output_activation_inputs.push(
            output_activation_input
                .expect("convolutional forward always records an activation input"),
        );
        cache.outputs.push(step_output);
        previous_state = state;
    }
    Ok((output, cache))
}
