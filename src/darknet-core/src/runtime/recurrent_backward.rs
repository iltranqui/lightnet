use super::*;

pub(super) fn zero_connected_gradient(params: &ConnectedParams) -> ConnectedParams {
    ConnectedParams {
        biases: vec![0.0; params.biases.len()],
        batch_norm: params
            .batch_norm
            .as_ref()
            .map(|batch_norm| BatchNormParams {
                biases: vec![0.0; batch_norm.biases.len()],
                scales: vec![0.0; batch_norm.scales.len()],
                rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
                rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
            }),
        weights: vec![0.0; params.weights.len()],
    }
}

pub(super) fn zero_convolutional_gradient(params: &ConvolutionalParams) -> ConvolutionalParams {
    ConvolutionalParams {
        biases: vec![0.0; params.biases.len()],
        batch_norm: params
            .batch_norm
            .as_ref()
            .map(|batch_norm| BatchNormParams {
                biases: vec![0.0; batch_norm.biases.len()],
                scales: vec![0.0; batch_norm.scales.len()],
                rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
                rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
            }),
        weights: vec![0.0; params.weights.len()],
    }
}

pub(super) fn accumulate_connected_gradient(dst: &mut ConnectedParams, src: &ConnectedParams) -> Result<()> {
    accumulate_vec(&mut dst.biases, &src.biases)?;
    accumulate_vec(&mut dst.weights, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scales, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.rolling_mean, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.rolling_variance, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "connected recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn accumulate_convolutional_gradient(
    dst: &mut ConvolutionalParams,
    src: &ConvolutionalParams,
) -> Result<()> {
    accumulate_vec(&mut dst.biases, &src.biases)?;
    accumulate_vec(&mut dst.weights, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scales, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.rolling_mean, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.rolling_variance, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn upload_rnn_backward_cache<B: F32TensorOpsBackend>(
    backend: &B,
    cache: &RnnForwardCache,
) -> Result<GpuRnnBackwardCache<B::Tensor>>
where
    B::Error: std::fmt::Display,
{
    let upload = |tensors: &[Tensor], operation: &str| -> Result<Vec<B::Tensor>> {
        tensors
            .iter()
            .map(|tensor| {
                backend
                    .upload_f32(&tensor_backend_shape(tensor.shape), &tensor.data)
                    .map_err(|err| backend_error(operation, err))
            })
            .collect()
    };
    Ok(GpuRnnBackwardCache {
        sequence_batch: cache.sequence_batch,
        input_activation_inputs: upload(
            &cache.input_activation_inputs,
            "upload BN RNN input activation cache",
        )?,
        self_activation_inputs: upload(
            &cache.self_activation_inputs,
            "upload BN RNN recurrent activation cache",
        )?,
        output_activation_inputs: upload(
            &cache.output_activation_inputs,
            "upload BN RNN output activation cache",
        )?,
        previous_states: upload(&cache.previous_states, "upload BN RNN previous states")?,
        states: upload(&cache.states, "upload BN RNN states")?,
    })
}

pub(super) fn upload_lstm_backward_cache<B: F32TensorOpsBackend>(
    backend: &B,
    cache: &LstmForwardCache,
) -> Result<GpuLstmBackwardCache<B::Tensor>>
where
    B::Error: std::fmt::Display,
{
    let upload = |tensors: &[Tensor], operation: &str| -> Result<Vec<B::Tensor>> {
        tensors
            .iter()
            .map(|tensor| {
                backend
                    .upload_f32(&tensor_backend_shape(tensor.shape), &tensor.data)
                    .map_err(|err| backend_error(operation, err))
            })
            .collect()
    };
    Ok(GpuLstmBackwardCache {
        sequence_batch: cache.sequence_batch,
        previous_states: upload(&cache.previous_states, "upload BN LSTM previous states")?,
        previous_cells: upload(&cache.previous_cells, "upload BN LSTM previous cells")?,
        forget: upload(&cache.forget, "upload BN LSTM forget gates")?,
        input_gate: upload(&cache.input_gate, "upload BN LSTM input gates")?,
        candidate: upload(&cache.candidate, "upload BN LSTM candidate gates")?,
        output_gate: upload(&cache.output_gate, "upload BN LSTM output gates")?,
        cells: upload(&cache.cells, "upload BN LSTM cells")?,
    })
}

pub(super) fn upload_crnn_backward_cache<B: F32TensorOpsBackend>(
    backend: &B,
    cache: &CrnnForwardCache,
) -> Result<GpuCrnnBackwardCache<B::Tensor>>
where
    B::Error: std::fmt::Display,
{
    let upload = |tensors: &[Tensor], operation: &str| -> Result<Vec<B::Tensor>> {
        tensors
            .iter()
            .map(|tensor| {
                backend
                    .upload_f32(&tensor_backend_shape(tensor.shape), &tensor.data)
                    .map_err(|err| backend_error(operation, err))
            })
            .collect()
    };
    Ok(GpuCrnnBackwardCache {
        sequence_batch: cache.sequence_batch,
        input_activation_inputs: upload(
            &cache.input_activation_inputs,
            "upload BN CRNN input activation cache",
        )?,
        self_activation_inputs: upload(
            &cache.self_activation_inputs,
            "upload BN CRNN recurrent activation cache",
        )?,
        output_activation_inputs: upload(
            &cache.output_activation_inputs,
            "upload BN CRNN output activation cache",
        )?,
        previous_states: upload(&cache.previous_states, "upload BN CRNN previous states")?,
        states: upload(&cache.states, "upload BN CRNN states")?,
    })
}

pub(super) fn rolling_batch_norm_scale_gradient_device<B: F32TensorOpsBackend>(
    backend: &B,
    raw_output: &B::Tensor,
    delta: &B::Tensor,
    batch_norm: &BatchNormParams,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let channels = output_shape.channels;
    let channel_stride = output_shape.width * output_shape.height;
    if channels == 0
        || channel_stride == 0
        || batch_norm.rolling_mean.len() != channels
        || batch_norm.rolling_variance.len() != channels
    {
        return Err(Error::InvalidConfig(format!(
            "BN recurrent rolling-stat shape mismatch: output has {channels} channels"
        )));
    }

    let centered_bias: Vec<f32> = batch_norm.rolling_mean.iter().map(|mean| -*mean).collect();
    let inverse_std: Vec<f32> = batch_norm
        .rolling_variance
        .iter()
        .map(|variance| 1.0 / (variance + BATCH_NORM_EPSILON).sqrt())
        .collect();
    let centered_bias = backend
        .upload_f32(&[channels], &centered_bias)
        .map_err(|err| backend_error("upload BN recurrent rolling means", err))?;
    let inverse_std = backend
        .upload_f32(&[channels], &inverse_std)
        .map_err(|err| backend_error("upload BN recurrent inverse standard deviations", err))?;
    let centered = backend
        .bias_add_f32(raw_output, &centered_bias, channels, channel_stride, config)
        .map_err(|err| backend_error("center BN recurrent raw output", err))?;
    let normalized = backend
        .channel_scale_f32(&centered, &inverse_std, channels, channel_stride, config)
        .map_err(|err| backend_error("normalize BN recurrent raw output", err))?;
    let element_shape = tensor_backend_shape(output_shape);
    let scaled_delta = backend
        .vector_mul_f32(delta, &normalized, &element_shape, config)
        .map_err(|err| backend_error("multiply BN recurrent scale-gradient terms", err))?;
    backend
        .bias_backward_f32(&scaled_delta, channels, channel_stride, config)
        .map_err(|err| backend_error("reduce BN recurrent scale gradient", err))
}

pub(super) fn backward_connected_recurrent_projection_device<B: F32TensorOpsBackend>(
    backend: &B,
    input: &B::Tensor,
    weights: &B::Tensor,
    delta: &B::Tensor,
    params: &ConnectedParams,
    sequence_batch: usize,
    input_features: usize,
    output_features: usize,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, B::Tensor, B::Tensor, Option<B::Tensor>)>
where
    B::Error: std::fmt::Display,
{
    let (linear_delta, beta_gradient, scale_gradient) =
        if let Some(batch_norm) = params.batch_norm.as_ref() {
            if batch_norm.biases.len() != output_features
                || batch_norm.scales.len() != output_features
                || batch_norm.rolling_mean.len() != output_features
                || batch_norm.rolling_variance.len() != output_features
                || params.biases.len() != output_features
            {
                return Err(Error::InvalidConfig(
                    "BN recurrent projection parameter shape mismatch".to_owned(),
                ));
            }
            let beta_gradient = backend
                .bias_backward_f32(delta, output_features, 1, config)
                .map_err(|err| backend_error("BN recurrent beta gradient", err))?;
            let zero_biases = backend
                .zeros_f32(&[output_features])
                .map_err(|err| backend_error("allocate BN recurrent raw projection biases", err))?;
            let raw_output = backend
                .connected_f32(
                    input,
                    weights,
                    &zero_biases,
                    sequence_batch,
                    input_features,
                    output_features,
                    config,
                )
                .map_err(|err| backend_error("BN recurrent raw connected projection", err))?;
            let scale_gradient = rolling_batch_norm_scale_gradient_device(
                backend,
                &raw_output,
                delta,
                batch_norm,
                TensorShape::new(1, 1, output_features, sequence_batch),
                config,
            )?;
            let delta_scale: Vec<f32> = batch_norm
                .scales
                .iter()
                .zip(&batch_norm.rolling_variance)
                .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
                .collect();
            let delta_scale = backend
                .upload_f32(&[output_features], &delta_scale)
                .map_err(|err| backend_error("upload BN recurrent affine scales", err))?;
            let linear_delta = backend
                .channel_scale_f32(delta, &delta_scale, output_features, 1, config)
                .map_err(|err| backend_error("BN recurrent rolling-stat delta scale", err))?;
            (linear_delta, Some(beta_gradient), Some(scale_gradient))
        } else {
            (delta.clone(), None, None)
        };
    let (input_gradient, weight_gradient, connected_bias_gradient) = backend
        .connected_backward_f32(
            input,
            weights,
            &linear_delta,
            sequence_batch,
            input_features,
            output_features,
            config,
        )
        .map_err(|err| backend_error("recurrent connected projection backward", err))?;
    let bias_gradient = beta_gradient.unwrap_or(connected_bias_gradient);
    Ok((
        input_gradient,
        weight_gradient,
        bias_gradient,
        scale_gradient,
    ))
}

pub(super) fn backward_rnn_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &RnnSpec,
    params: &RnnParams,
    cache: &GpuRnnBackwardCache<B::Tensor>,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    output_gradient: &B::Tensor,
    layer_index: usize,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, RnnParams)>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "RNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let inputs = input_shape.element_count();
    let weights_valid = params.input.weights.len() == inputs * spec.hidden
        && params.self_params.weights.len() == spec.hidden * spec.hidden
        && params.output.weights.len() == spec.hidden * spec.output;
    if !weights_valid
        || cache.states.len() != spec.steps
        || cache.previous_states.len() != spec.steps
        || cache.input_activation_inputs.len() != spec.steps
        || cache.self_activation_inputs.len() != spec.steps
        || cache.output_activation_inputs.len() != spec.steps
    {
        return Err(Error::InvalidConfig(
            "RNN backward cache or parameter shapes are inconsistent".to_owned(),
        ));
    }

    let input_weights = backend
        .upload_f32(&[spec.hidden, inputs], &params.input.weights)
        .map_err(|err| backend_error("upload RNN backward input weights", err))?;
    let self_weights = backend
        .upload_f32(&[spec.hidden, spec.hidden], &params.self_params.weights)
        .map_err(|err| backend_error("upload RNN backward recurrent weights", err))?;
    let output_weights = backend
        .upload_f32(&[spec.output, spec.hidden], &params.output.weights)
        .map_err(|err| backend_error("upload RNN backward output weights", err))?;
    let input_activation = gpu_activation_kind(layer_index, &spec.activation)?;
    let self_activation = match spec.logistic {
        1 => gpu_activation_kind(layer_index, &Activation::Logistic)?,
        2 => gpu_activation_kind(layer_index, &Activation::Loggy)?,
        _ => input_activation,
    };
    let output_gradient_shape = TensorShape::new(1, 1, spec.output, input_shape.batch);
    let mut input_gradients: Vec<Option<B::Tensor>> = (0..spec.steps).map(|_| None).collect();
    let mut weight_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut bias_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut scale_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut future_state_gradient = backend
        .zeros_f32(&[sequence_batch, spec.hidden])
        .map_err(|err| backend_error("allocate RNN backward state gradient", err))?;
    let mut gradients = RnnParams {
        input: zero_connected_gradient(&params.input),
        self_params: zero_connected_gradient(&params.self_params),
        output: zero_connected_gradient(&params.output),
    };

    for step in (0..spec.steps).rev() {
        let output_gradient_step_gpu = backend
            .sequence_slice_f32(
                output_gradient,
                &tensor_backend_shape(output_gradient_shape),
                step,
                spec.steps,
                &[sequence_batch, spec.output],
                config,
            )
            .map_err(|err| backend_error("slice RNN output gradient", err))?;
        let output_activation_input = cache.output_activation_inputs[step].clone();
        let output_delta = backend
            .activation_backward_f32(
                &output_activation_input,
                &output_gradient_step_gpu,
                input_activation,
                config,
            )
            .map_err(|err| backend_error("RNN output activation backward", err))?;
        let state_gpu = cache.states[step].clone();
        let (
            mut state_gradient,
            output_weight_gradient,
            output_bias_gradient,
            output_scale_gradient,
        ) = backward_connected_recurrent_projection_device(
            backend,
            &state_gpu,
            &output_weights,
            &output_delta,
            &params.output,
            sequence_batch,
            spec.hidden,
            spec.output,
            config,
        )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[2],
            output_weight_gradient,
            config,
            "accumulate RNN output weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[2],
            output_bias_gradient,
            config,
            "accumulate RNN output bias gradient",
        )?;
        if let Some(scale_gradient) = output_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[2],
                scale_gradient,
                config,
                "accumulate RNN output scale gradient",
            )?;
        }
        state_gradient = backend
            .vector_add_f32(&state_gradient, &future_state_gradient, config)
            .map_err(|err| backend_error("accumulate RNN future state gradient", err))?;

        let self_activation_input = cache.self_activation_inputs[step].clone();
        let self_delta = backend
            .activation_backward_f32(
                &self_activation_input,
                &state_gradient,
                self_activation,
                config,
            )
            .map_err(|err| backend_error("RNN recurrent activation backward", err))?;
        let previous_state_gpu = cache.previous_states[step].clone();
        let (mut future_state, self_weight_gradient, self_bias_gradient, self_scale_gradient) =
            backward_connected_recurrent_projection_device(
                backend,
                &previous_state_gpu,
                &self_weights,
                &self_delta,
                &params.self_params,
                sequence_batch,
                spec.hidden,
                spec.hidden,
                config,
            )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[1],
            self_weight_gradient,
            config,
            "accumulate RNN recurrent weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[1],
            self_bias_gradient,
            config,
            "accumulate RNN recurrent bias gradient",
        )?;
        if let Some(scale_gradient) = self_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[1],
                scale_gradient,
                config,
                "accumulate RNN recurrent scale gradient",
            )?;
        }

        let input_activation_input = cache.input_activation_inputs[step].clone();
        let input_delta = backend
            .activation_backward_f32(
                &input_activation_input,
                &state_gradient,
                input_activation,
                config,
            )
            .map_err(|err| backend_error("RNN input activation backward", err))?;
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
        let (step_input_gradient, input_weight_gradient, input_bias_gradient, input_scale_gradient) =
            backward_connected_recurrent_projection_device(
                backend,
                &step_input_gpu,
                &input_weights,
                &input_delta,
                &params.input,
                sequence_batch,
                inputs,
                spec.hidden,
                config,
            )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[0],
            input_weight_gradient,
            config,
            "accumulate RNN input weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[0],
            input_bias_gradient,
            config,
            "accumulate RNN input bias gradient",
        )?;
        if let Some(scale_gradient) = input_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[0],
                scale_gradient,
                config,
                "accumulate RNN input scale gradient",
            )?;
        }
        input_gradients[step] = Some(step_input_gradient);

        if spec.shortcut {
            future_state = backend
                .vector_add_f32(&future_state, &state_gradient, config)
                .map_err(|err| backend_error("accumulate RNN shortcut gradient", err))?;
        }
        future_state_gradient = future_state;
    }

    let input_gradient_refs: Vec<_> = input_gradients
        .iter()
        .map(|gradient| {
            gradient.as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing RNN input gradient timestep".to_owned())
            })
        })
        .collect::<Result<_>>()?;
    let input_gradient = backend
        .sequence_concat_f32(
            &input_gradient_refs,
            &tensor_backend_shape(input_shape),
            sequence_batch * inputs,
            config,
        )
        .map_err(|err| backend_error("assemble RNN input gradient", err))?;
    for index in 0..3 {
        let weights = backend
            .download_f32(weight_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated RNN weight gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download accumulated RNN weight gradient", err))?;
        let biases = backend
            .download_f32(bias_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated RNN bias gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download accumulated RNN bias gradient", err))?;
        let gradient = match index {
            0 => &mut gradients.input,
            1 => &mut gradients.self_params,
            _ => &mut gradients.output,
        };
        if weights.len() != gradient.weights.len() || biases.len() != gradient.biases.len() {
            return Err(Error::InvalidConfig(
                "accumulated RNN parameter gradient shape mismatch".to_owned(),
            ));
        }
        gradient.weights = weights;
        gradient.biases = biases;
        match (&mut gradient.batch_norm, &scale_gradients[index]) {
            (Some(batch_norm), Some(scale_gradient)) => {
                let scales = backend
                    .download_f32(scale_gradient)
                    .map_err(|err| backend_error("download accumulated RNN scale gradient", err))?;
                if scales.len() != batch_norm.scales.len() {
                    return Err(Error::InvalidConfig(
                        "accumulated RNN scale gradient shape mismatch".to_owned(),
                    ));
                }
                batch_norm.scales = scales;
            }
            (None, None) => {}
            _ => {
                return Err(Error::InvalidConfig(
                    "RNN batch-normalization gradient presence mismatch".to_owned(),
                ));
            }
        }
    }
    Ok((input_gradient, gradients))
}

pub(super) fn accumulate_backend_gradient<B: F32TensorOpsBackend>(
    backend: &B,
    slot: &mut Option<B::Tensor>,
    gradient: B::Tensor,
    config: ElementwiseLaunchConfig,
    operation: &str,
) -> Result<()>
where
    B::Error: std::fmt::Display,
{
    *slot = Some(match slot.take() {
        Some(previous) => backend
            .vector_add_f32(&previous, &gradient, config)
            .map_err(|err| backend_error(operation, err))?,
        None => gradient,
    });
    Ok(())
}

pub(super) fn backward_lstm_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LstmSpec,
    params: &LstmParams,
    cache: &GpuLstmBackwardCache<B::Tensor>,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    output_gradient: &B::Tensor,
    layer_index: usize,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, LstmParams)>
where
    B::Error: std::fmt::Display,
{
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    let output_shape = TensorShape::new(1, 1, spec.output, input_shape.batch);
    let inputs = input_shape.element_count();
    let has_batch_norm = [
        params.wf.batch_norm.is_some(),
        params.wi.batch_norm.is_some(),
        params.wg.batch_norm.is_some(),
        params.wo.batch_norm.is_some(),
        params.uf.batch_norm.is_some(),
        params.ui.batch_norm.is_some(),
        params.ug.batch_norm.is_some(),
        params.uo.batch_norm.is_some(),
    ];
    if has_batch_norm
        .iter()
        .any(|present| *present != spec.batch_normalize)
    {
        return Err(Error::InvalidConfig(format!(
            "LSTM batch-normalization parameter mismatch at layer #{layer_index}"
        )));
    }
    if cache.sequence_batch != sequence_batch
        || cache.previous_states.len() != spec.steps
        || cache.previous_cells.len() != spec.steps
        || cache.forget.len() != spec.steps
        || cache.input_gate.len() != spec.steps
        || cache.candidate.len() != spec.steps
        || cache.output_gate.len() != spec.steps
        || cache.cells.len() != spec.steps
    {
        return Err(Error::InvalidConfig(format!(
            "LSTM backward cache shape mismatch at layer #{layer_index}"
        )));
    }

    let projections: [(&ConnectedParams, usize, &str); 8] = [
        (&params.wf, spec.output, "forget recurrent"),
        (&params.wi, spec.output, "input recurrent"),
        (&params.wg, spec.output, "candidate recurrent"),
        (&params.wo, spec.output, "output recurrent"),
        (&params.uf, inputs, "forget input"),
        (&params.ui, inputs, "input input"),
        (&params.ug, inputs, "candidate input"),
        (&params.uo, inputs, "output input"),
    ];
    let mut projection_weights = Vec::with_capacity(projections.len());
    for (projection, features, name) in projections {
        if projection.weights.len() != spec.output * features
            || projection.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "LSTM {name} projection shape mismatch at layer #{layer_index}"
            )));
        }
        projection_weights.push(
            backend
                .upload_f32(&[spec.output, features], &projection.weights)
                .map_err(|err| backend_error("upload LSTM backward weights", err))?,
        );
    }
    let mut input_gradients: Vec<Option<B::Tensor>> = (0..spec.steps).map(|_| None).collect();
    let mut next_hidden_gradient = backend
        .zeros_f32(&[sequence_batch, spec.output])
        .map_err(|err| backend_error("allocate LSTM hidden gradient", err))?;
    let mut next_cell_gradient = backend
        .zeros_f32(&[sequence_batch, spec.output])
        .map_err(|err| backend_error("allocate LSTM cell gradient", err))?;
    let mut weight_gradients: Vec<Option<B::Tensor>> =
        (0..projections.len()).map(|_| None).collect();
    let mut bias_gradients: Vec<Option<B::Tensor>> = (0..projections.len()).map(|_| None).collect();
    let mut scale_gradients: Vec<Option<B::Tensor>> =
        (0..projections.len()).map(|_| None).collect();

    for step in (0..spec.steps).rev() {
        let output_gradient_step_gpu = backend
            .sequence_slice_f32(
                output_gradient,
                &tensor_backend_shape(output_shape),
                step,
                spec.steps,
                &[sequence_batch, spec.output],
                config,
            )
            .map_err(|err| backend_error("slice LSTM output gradient", err))?;
        let hidden_gradient = backend
            .vector_add_f32(&output_gradient_step_gpu, &next_hidden_gradient, config)
            .map_err(|err| backend_error("accumulate LSTM hidden gradient", err))?;
        let (
            forget_gradient,
            input_gate_gradient,
            candidate_gradient,
            output_gate_gradient,
            cell_gradient,
        ) = backend
            .lstm_gate_backward_f32(
                &hidden_gradient,
                &cache.cells[step],
                &cache.previous_cells[step],
                &cache.forget[step],
                &cache.input_gate[step],
                &cache.candidate[step],
                &cache.output_gate[step],
                &next_cell_gradient,
                sequence_batch * spec.output,
                config,
            )
            .map_err(|err| backend_error("LSTM gate backward", err))?;
        next_cell_gradient = cell_gradient;
        let previous_state_gpu = &cache.previous_states[step];
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
        let gate_gradients = [
            forget_gradient,
            input_gate_gradient,
            candidate_gradient,
            output_gate_gradient,
        ];
        let mut previous_hidden_gradient: Option<B::Tensor> = None;
        let mut step_input_gradient: Option<B::Tensor> = None;
        for projection_index in 0..8 {
            let gate_index = projection_index % 4;
            let recurrent = projection_index < 4;
            let projection_input = if recurrent {
                &previous_state_gpu
            } else {
                &step_input_gpu
            };
            let features = if recurrent { spec.output } else { inputs };
            let gate_gradient_gpu = &gate_gradients[gate_index];
            let (input_gradient, weight_gradient, bias_gradient, scale_gradient) =
                backward_connected_recurrent_projection_device(
                    backend,
                    projection_input,
                    &projection_weights[projection_index],
                    gate_gradient_gpu,
                    projections[projection_index].0,
                    sequence_batch,
                    features,
                    spec.output,
                    config,
                )?;
            if recurrent {
                accumulate_backend_gradient(
                    backend,
                    &mut previous_hidden_gradient,
                    input_gradient,
                    config,
                    "accumulate LSTM previous hidden gradient",
                )?;
            } else {
                accumulate_backend_gradient(
                    backend,
                    &mut step_input_gradient,
                    input_gradient,
                    config,
                    "accumulate LSTM step input gradient",
                )?;
            }
            accumulate_backend_gradient(
                backend,
                &mut weight_gradients[projection_index],
                weight_gradient,
                config,
                "accumulate LSTM weight gradient",
            )?;
            accumulate_backend_gradient(
                backend,
                &mut bias_gradients[projection_index],
                bias_gradient,
                config,
                "accumulate LSTM bias gradient",
            )?;
            if let Some(scale_gradient) = scale_gradient {
                accumulate_backend_gradient(
                    backend,
                    &mut scale_gradients[projection_index],
                    scale_gradient,
                    config,
                    "accumulate LSTM scale gradient",
                )?;
            }
        }
        let step_input_gradient = step_input_gradient.ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing LSTM input gradient at layer #{layer_index}, step {step}"
            ))
        })?;
        input_gradients[step] = Some(step_input_gradient);
        next_hidden_gradient = previous_hidden_gradient.ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing LSTM hidden gradient at layer #{layer_index}, step {step}"
            ))
        })?;
    }

    let input_gradient_refs: Vec<_> = input_gradients
        .iter()
        .map(|gradient| {
            gradient.as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing LSTM input gradient timestep".to_owned())
            })
        })
        .collect::<Result<_>>()?;
    let input_gradient = backend
        .sequence_concat_f32(
            &input_gradient_refs,
            &tensor_backend_shape(input_shape),
            sequence_batch * inputs,
            config,
        )
        .map_err(|err| backend_error("assemble LSTM input gradient", err))?;
    let mut gradients = LstmParams {
        wf: zero_connected_gradient(&params.wf),
        wi: zero_connected_gradient(&params.wi),
        wg: zero_connected_gradient(&params.wg),
        wo: zero_connected_gradient(&params.wo),
        uf: zero_connected_gradient(&params.uf),
        ui: zero_connected_gradient(&params.ui),
        ug: zero_connected_gradient(&params.ug),
        uo: zero_connected_gradient(&params.uo),
    };
    for index in 0..8 {
        let weights = backend
            .download_f32(weight_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated LSTM weight gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download LSTM weight gradient", err))?;
        let biases = backend
            .download_f32(bias_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated LSTM bias gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download LSTM bias gradient", err))?;
        let gradient = match index {
            0 => &mut gradients.wf,
            1 => &mut gradients.wi,
            2 => &mut gradients.wg,
            3 => &mut gradients.wo,
            4 => &mut gradients.uf,
            5 => &mut gradients.ui,
            6 => &mut gradients.ug,
            _ => &mut gradients.uo,
        };
        if weights.len() != gradient.weights.len() || biases.len() != gradient.biases.len() {
            return Err(Error::InvalidConfig(format!(
                "LSTM accumulated parameter gradient shape mismatch at layer #{layer_index}"
            )));
        }
        gradient.weights = weights;
        gradient.biases = biases;
        match (&mut gradient.batch_norm, &scale_gradients[index]) {
            (Some(batch_norm), Some(scale_gradient)) => {
                let scales = backend
                    .download_f32(scale_gradient)
                    .map_err(|err| backend_error("download LSTM scale gradient", err))?;
                if scales.len() != batch_norm.scales.len() {
                    return Err(Error::InvalidConfig(format!(
                        "LSTM scale gradient shape mismatch at layer #{layer_index}"
                    )));
                }
                batch_norm.scales = scales;
            }
            (None, None) => {}
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "LSTM batch-normalization gradient mismatch at layer #{layer_index}"
                )));
            }
        }
    }
    Ok((input_gradient, gradients))
}

pub(super) fn backward_rnn_layer(
    spec: &RnnSpec,
    params: &RnnParams,
    cache: &RnnForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, RnnParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "RNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
    let output_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    if output_gradient.shape.batch != input.shape.batch
        || output_gradient.shape.width != 1
        || output_gradient.shape.height != 1
        || output_gradient.shape.channels != spec.output
    {
        return Err(Error::InvalidConfig(
            "RNN backward gradient shape mismatch".to_owned(),
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
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = RnnParams {
        input: zero_connected_gradient(&params.input),
        self_params: zero_connected_gradient(&params.self_params),
        output: zero_connected_gradient(&params.output),
    };
    let mut future_state_gradient = Tensor::zeros(hidden_shape);

    for step in (0..spec.steps).rev() {
        let output_gradient_step =
            sequence_slice(output_gradient, step, sequence_batch, output_shape)?;
        let (mut state_gradient, output_gradient_params) = backward_connected(
            &output_spec,
            &params.output,
            &cache.states[step],
            output_shape,
            &output_gradient_step,
            &cache.outputs[step],
            Some(&cache.output_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut state_gradient, &future_state_gradient)?;
        accumulate_connected_gradient(&mut gradients.output, &output_gradient_params)?;

        let (previous_from_self, self_gradient) = backward_connected(
            &self_spec,
            &params.self_params,
            &cache.previous_states[step],
            hidden_shape,
            &state_gradient,
            &cache.self_outputs[step],
            Some(&cache.self_activation_inputs[step]),
            workspace,
        )?;
        accumulate_connected_gradient(&mut gradients.self_params, &self_gradient)?;

        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (step_input_gradient, input_gradient_params) = backward_connected(
            &input_spec,
            &params.input,
            &step_input,
            hidden_shape,
            &state_gradient,
            &cache.input_outputs[step],
            Some(&cache.input_activation_inputs[step]),
            workspace,
        )?;
        accumulate_connected_gradient(&mut gradients.input, &input_gradient_params)?;
        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;

        future_state_gradient = previous_from_self;
        if spec.shortcut {
            add_tensor_inplace(&mut future_state_gradient, &state_gradient)?;
        }
        if step == 0 {
            future_state_gradient = Tensor::zeros(hidden_shape);
        }
    }

    Ok((input_gradient, gradients))
}

pub(super) fn backward_lstm_layer(
    spec: &LstmSpec,
    params: &LstmParams,
    cache: &LstmForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, LstmParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "LSTM forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let inputs_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
    if output_gradient.shape != TensorShape::new(1, 1, spec.output, input.shape.batch) {
        return Err(Error::InvalidConfig(
            "LSTM backward gradient shape mismatch".to_owned(),
        ));
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
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = LstmParams {
        wf: zero_connected_gradient(&params.wf),
        wi: zero_connected_gradient(&params.wi),
        wg: zero_connected_gradient(&params.wg),
        wo: zero_connected_gradient(&params.wo),
        uf: zero_connected_gradient(&params.uf),
        ui: zero_connected_gradient(&params.ui),
        ug: zero_connected_gradient(&params.ug),
        uo: zero_connected_gradient(&params.uo),
    };
    let mut next_hidden_gradient = Tensor::zeros(state_shape);
    let mut next_cell_gradient = Tensor::zeros(state_shape);

    for step in (0..spec.steps).rev() {
        let mut hidden_gradient =
            sequence_slice(output_gradient, step, sequence_batch, state_shape)?;
        add_tensor_inplace(&mut hidden_gradient, &next_hidden_gradient)?;
        let cell = &cache.cells[step];
        let previous_cell = &cache.previous_cells[step];
        let forget = &cache.forget[step];
        let input_gate = &cache.input_gate[step];
        let candidate = &cache.candidate[step];
        let output_gate = &cache.output_gate[step];

        let mut forget_gradient = Tensor::zeros(state_shape);
        let mut input_gate_gradient = Tensor::zeros(state_shape);
        let mut candidate_gradient = Tensor::zeros(state_shape);
        let mut output_gate_gradient = Tensor::zeros(state_shape);
        let mut cell_gradient = Tensor::zeros(state_shape);
        for index in 0..cell.data.len() {
            let tanh_cell = cell.data[index].tanh();
            output_gate_gradient.data[index] = hidden_gradient.data[index] * tanh_cell;
            cell_gradient.data[index] = hidden_gradient.data[index]
                * output_gate.data[index]
                * (1.0 - tanh_cell * tanh_cell)
                + next_cell_gradient.data[index];
            forget_gradient.data[index] = cell_gradient.data[index] * previous_cell.data[index];
            input_gate_gradient.data[index] = cell_gradient.data[index] * candidate.data[index];
            candidate_gradient.data[index] = cell_gradient.data[index] * input_gate.data[index];
            forget_gradient.data[index] *= forget.data[index] * (1.0 - forget.data[index]);
            input_gate_gradient.data[index] *=
                input_gate.data[index] * (1.0 - input_gate.data[index]);
            candidate_gradient.data[index] *= 1.0 - candidate.data[index] * candidate.data[index];
            output_gate_gradient.data[index] *=
                output_gate.data[index] * (1.0 - output_gate.data[index]);
        }

        let mut previous_hidden_gradient = Tensor::zeros(state_shape);
        let mut step_input_gradient = Tensor::zeros(inputs_shape);
        let previous_state = &cache.previous_states[step];
        let step_input = sequence_slice(input, step, sequence_batch, inputs_shape)?;

        let (gradient, parameter_gradient) = backward_connected(
            &wf_spec,
            &params.wf,
            previous_state,
            state_shape,
            &forget_gradient,
            &cache.wf_outputs[step],
            Some(&cache.wf_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wf, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wi_spec,
            &params.wi,
            previous_state,
            state_shape,
            &input_gate_gradient,
            &cache.wi_outputs[step],
            Some(&cache.wi_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wi, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wg_spec,
            &params.wg,
            previous_state,
            state_shape,
            &candidate_gradient,
            &cache.wg_outputs[step],
            Some(&cache.wg_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wg, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &wo_spec,
            &params.wo,
            previous_state,
            state_shape,
            &output_gate_gradient,
            &cache.wo_outputs[step],
            Some(&cache.wo_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut previous_hidden_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.wo, &parameter_gradient)?;

        let (gradient, parameter_gradient) = backward_connected(
            &uf_spec,
            &params.uf,
            &step_input,
            state_shape,
            &forget_gradient,
            &cache.uf_outputs[step],
            Some(&cache.uf_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.uf, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &ui_spec,
            &params.ui,
            &step_input,
            state_shape,
            &input_gate_gradient,
            &cache.ui_outputs[step],
            Some(&cache.ui_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.ui, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &ug_spec,
            &params.ug,
            &step_input,
            state_shape,
            &candidate_gradient,
            &cache.ug_outputs[step],
            Some(&cache.ug_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.ug, &parameter_gradient)?;
        let (gradient, parameter_gradient) = backward_connected(
            &uo_spec,
            &params.uo,
            &step_input,
            state_shape,
            &output_gate_gradient,
            &cache.uo_outputs[step],
            Some(&cache.uo_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut step_input_gradient, &gradient)?;
        accumulate_connected_gradient(&mut gradients.uo, &parameter_gradient)?;

        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;
        next_hidden_gradient = previous_hidden_gradient;
        for (index, (dst, value)) in next_cell_gradient
            .data
            .iter_mut()
            .zip(&cell_gradient.data)
            .enumerate()
        {
            *dst = *value * forget.data[index];
        }
    }

    Ok((input_gradient, gradients))
}

pub(super) fn backward_crnn_layer(
    spec: &CrnnSpec,
    params: &CrnnParams,
    cache: &CrnnForwardCache,
    input: &Tensor,
    output_gradient: &Tensor,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, CrnnParams)> {
    let sequence_batch = recurrent_sequence_batch(input.shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "CRNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
    let input_width =
        convolution_output_dim(input.shape.width, spec.padding, spec.size, spec.stride);
    let input_height =
        convolution_output_dim(input.shape.height, spec.padding, spec.size, spec.stride);
    let hidden_shape = TensorShape::new(input_width, input_height, spec.hidden, sequence_batch);
    let output_shape = TensorShape::new(input_width, input_height, spec.output, sequence_batch);
    let input_step_shape = TensorShape::new(
        input.shape.width,
        input.shape.height,
        input.shape.channels,
        sequence_batch,
    );
    if output_gradient.shape
        != TensorShape::new(input_width, input_height, spec.output, input.shape.batch)
    {
        return Err(Error::InvalidConfig(
            "CRNN backward gradient shape mismatch".to_owned(),
        ));
    }
    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = CrnnParams {
        input: zero_convolutional_gradient(&params.input),
        self_params: zero_convolutional_gradient(&params.self_params),
        output: zero_convolutional_gradient(&params.output),
    };
    let mut future_state_gradient = Tensor::zeros(hidden_shape);

    for step in (0..spec.steps).rev() {
        let output_gradient_step =
            sequence_slice(output_gradient, step, sequence_batch, output_shape)?;
        let (mut state_gradient, output_gradient_params) = backward_convolutional(
            &output_spec,
            &params.output,
            &cache.states[step],
            output_shape,
            &output_gradient_step,
            Some(&cache.output_activation_inputs[step]),
            workspace,
        )?;
        add_tensor_inplace(&mut state_gradient, &future_state_gradient)?;
        accumulate_convolutional_gradient(&mut gradients.output, &output_gradient_params)?;

        let (previous_from_self, self_gradient) = backward_convolutional(
            &self_spec,
            &params.self_params,
            &cache.previous_states[step],
            hidden_shape,
            &state_gradient,
            Some(&cache.self_activation_inputs[step]),
            workspace,
        )?;
        accumulate_convolutional_gradient(&mut gradients.self_params, &self_gradient)?;

        let step_input = sequence_slice(input, step, sequence_batch, input_step_shape)?;
        let (step_input_gradient, input_gradient_params) = backward_convolutional(
            &input_spec,
            &params.input,
            &step_input,
            hidden_shape,
            &state_gradient,
            Some(&cache.input_activation_inputs[step]),
            workspace,
        )?;
        accumulate_convolutional_gradient(&mut gradients.input, &input_gradient_params)?;
        sequence_write(
            &mut input_gradient,
            step,
            sequence_batch,
            &step_input_gradient,
        )?;

        future_state_gradient = previous_from_self;
        if spec.shortcut {
            add_tensor_inplace(&mut future_state_gradient, &state_gradient)?;
        }
        if step == 0 {
            future_state_gradient = Tensor::zeros(hidden_shape);
        }
    }

    Ok((input_gradient, gradients))
}

pub(super) fn backward_recurrent_convolution_projection_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    weights: &B::Tensor,
    input: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    delta: &B::Tensor,
    layer_index: usize,
    projection: &str,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, B::Tensor, B::Tensor, Option<B::Tensor>)>
where
    B::Error: std::fmt::Display,
{
    let groups = spec.groups.max(1);
    let filters = spec.filters;
    let spatial = output_shape.width * output_shape.height;
    let channels_per_group = input_shape.channels / groups;
    let expected_weights = filters * channels_per_group * spec.size * spec.size;
    if output_shape.channels != filters
        || input_shape.batch != output_shape.batch
        || input_shape.channels % groups != 0
        || filters % groups != 0
        || params.weights.len() != expected_weights
        || params.biases.len() != filters
    {
        return Err(Error::InvalidConfig(format!(
            "CRNN {projection} projection parameter/shape mismatch at layer #{layer_index}"
        )));
    }
    let bias_gradient = backend
        .bias_backward_f32(delta, filters, spatial, config)
        .map_err(|err| backend_error("CRNN recurrent convolution beta gradient", err))?;
    let (convolution_delta, scale_gradient) = if let Some(batch_norm) = params.batch_norm.as_ref() {
        if batch_norm.biases.len() != filters
            || batch_norm.scales.len() != filters
            || batch_norm.rolling_mean.len() != filters
            || batch_norm.rolling_variance.len() != filters
        {
            return Err(Error::InvalidConfig(format!(
                "CRNN {projection} batch-norm parameter mismatch at layer #{layer_index}"
            )));
        }
        let raw_output = backend
            .conv2d_f32(
                input,
                weights,
                &tensor_backend_shape(output_shape),
                input_shape.channels,
                input_shape.height,
                input_shape.width,
                filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                groups,
                config,
            )
            .map_err(|err| backend_error("CRNN recurrent raw convolution", err))?;
        let scale_gradient = rolling_batch_norm_scale_gradient_device(
            backend,
            &raw_output,
            delta,
            batch_norm,
            output_shape,
            config,
        )?;
        let delta_scales: Vec<f32> = batch_norm
            .scales
            .iter()
            .zip(&batch_norm.rolling_variance)
            .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
            .collect();
        let delta_scales = backend
            .upload_f32(&[filters], &delta_scales)
            .map_err(|err| backend_error("upload CRNN recurrent BN scales", err))?;
        let convolution_delta = backend
            .channel_scale_f32(delta, &delta_scales, filters, spatial, config)
            .map_err(|err| backend_error("scale CRNN recurrent BN delta", err))?;
        (convolution_delta, Some(scale_gradient))
    } else {
        (delta.clone(), None)
    };
    let weight_gradient = backend
        .conv2d_backward_weights_f32(
            input,
            &convolution_delta,
            &[filters, channels_per_group, spec.size, spec.size],
            input_shape.channels,
            input_shape.height,
            input_shape.width,
            filters,
            spec.size,
            spec.stride_x,
            spec.stride_y,
            spec.dilation,
            spec.padding,
            groups,
            config,
        )
        .map_err(|err| backend_error("CRNN recurrent convolution weight backward", err))?;
    let input_gradient = backend
        .conv2d_backward_input_f32(
            &convolution_delta,
            weights,
            &tensor_backend_shape(input_shape),
            input_shape.channels,
            input_shape.height,
            input_shape.width,
            filters,
            spec.size,
            spec.stride_x,
            spec.stride_y,
            spec.dilation,
            spec.padding,
            groups,
            config,
        )
        .map_err(|err| backend_error("CRNN recurrent convolution input backward", err))?;
    Ok((
        input_gradient,
        weight_gradient,
        bias_gradient,
        scale_gradient,
    ))
}

pub(super) fn backward_crnn_layer_device<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &CrnnSpec,
    params: &CrnnParams,
    cache: &GpuCrnnBackwardCache<B::Tensor>,
    input_gpu: &B::Tensor,
    input_shape: TensorShape,
    output_gradient: &B::Tensor,
    layer_index: usize,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, CrnnParams)>
where
    B::Error: std::fmt::Display,
{
    let has_batch_norm = [
        params.input.batch_norm.is_some(),
        params.self_params.batch_norm.is_some(),
        params.output.batch_norm.is_some(),
    ];
    if has_batch_norm
        .iter()
        .any(|present| *present != spec.batch_normalize)
    {
        return Err(Error::InvalidConfig(format!(
            "CRNN batch-normalization parameter mismatch at layer #{layer_index}"
        )));
    }
    let sequence_batch = recurrent_sequence_batch(input_shape.batch, spec.steps)?;
    if cache.sequence_batch != sequence_batch {
        return Err(Error::InvalidConfig(
            "CRNN forward cache sequence batch mismatch".to_owned(),
        ));
    }
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
    let expected_output_shape =
        TensorShape::new(input_width, input_height, spec.output, input_shape.batch);
    if cache.states.len() != spec.steps
        || cache.previous_states.len() != spec.steps
        || cache.input_activation_inputs.len() != spec.steps
        || cache.self_activation_inputs.len() != spec.steps
        || cache.output_activation_inputs.len() != spec.steps
    {
        return Err(Error::InvalidConfig(
            "CRNN backward cache or gradient shapes are inconsistent".to_owned(),
        ));
    }

    let groups = spec.groups.max(1);
    let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
    let output_spec = recurrent_convolutional_spec(spec, spec.output);
    let upload_weights = |conv: &ConvolutionalParams, channels: usize, name: &str| {
        let grouped_channels = channels / groups;
        backend
            .upload_f32(
                &[conv.biases.len(), grouped_channels, spec.size, spec.size],
                &conv.weights,
            )
            .map_err(|err| backend_error(name, err))
    };
    let input_weights = upload_weights(
        &params.input,
        input_shape.channels,
        "upload CRNN backward input weights",
    )?;
    let self_weights = upload_weights(
        &params.self_params,
        spec.hidden,
        "upload CRNN backward recurrent weights",
    )?;
    let output_weights = upload_weights(
        &params.output,
        spec.hidden,
        "upload CRNN backward output weights",
    )?;
    let activation = gpu_activation_kind(layer_index, &spec.activation)?;
    let mut input_gradients: Vec<Option<B::Tensor>> = (0..spec.steps).map(|_| None).collect();
    let mut weight_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut bias_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut scale_gradients: Vec<Option<B::Tensor>> = (0..3).map(|_| None).collect();
    let mut gradients = CrnnParams {
        input: zero_convolutional_gradient(&params.input),
        self_params: zero_convolutional_gradient(&params.self_params),
        output: zero_convolutional_gradient(&params.output),
    };
    let mut future_state_gradient = backend
        .zeros_f32(&tensor_backend_shape(hidden_shape))
        .map_err(|err| backend_error("allocate CRNN backward state gradient", err))?;

    for step in (0..spec.steps).rev() {
        let output_gradient_gpu = backend
            .sequence_slice_f32(
                output_gradient,
                &tensor_backend_shape(expected_output_shape),
                step,
                spec.steps,
                &tensor_backend_shape(output_step_shape),
                config,
            )
            .map_err(|err| backend_error("slice CRNN output gradient", err))?;
        let output_activation_input = cache.output_activation_inputs[step].clone();
        let output_delta = backend
            .activation_backward_f32(
                &output_activation_input,
                &output_gradient_gpu,
                activation,
                config,
            )
            .map_err(|err| backend_error("CRNN output activation backward", err))?;
        let state_gpu = cache.states[step].clone();
        let (
            mut state_gradient,
            output_weight_gradient,
            output_bias_gradient,
            output_scale_gradient,
        ) = backward_recurrent_convolution_projection_device(
            backend,
            &output_spec,
            &params.output,
            &output_weights,
            &state_gpu,
            hidden_shape,
            output_step_shape,
            &output_delta,
            layer_index,
            "output",
            config,
        )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[2],
            output_weight_gradient,
            config,
            "accumulate CRNN output weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[2],
            output_bias_gradient,
            config,
            "accumulate CRNN output beta gradient",
        )?;
        if let Some(scale_gradient) = output_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[2],
                scale_gradient,
                config,
                "accumulate CRNN output scale gradient",
            )?;
        }
        state_gradient = backend
            .vector_add_f32(&state_gradient, &future_state_gradient, config)
            .map_err(|err| backend_error("accumulate CRNN future state gradient", err))?;

        let self_activation_input = cache.self_activation_inputs[step].clone();
        let self_delta = backend
            .activation_backward_f32(&self_activation_input, &state_gradient, activation, config)
            .map_err(|err| backend_error("CRNN recurrent activation backward", err))?;
        let previous_state_gpu = cache.previous_states[step].clone();
        let (
            mut previous_state_gradient,
            self_weight_gradient,
            self_bias_gradient,
            self_scale_gradient,
        ) = backward_recurrent_convolution_projection_device(
            backend,
            &self_spec,
            &params.self_params,
            &self_weights,
            &previous_state_gpu,
            hidden_shape,
            hidden_shape,
            &self_delta,
            layer_index,
            "recurrent",
            config,
        )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[1],
            self_weight_gradient,
            config,
            "accumulate CRNN recurrent weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[1],
            self_bias_gradient,
            config,
            "accumulate CRNN recurrent beta gradient",
        )?;
        if let Some(scale_gradient) = self_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[1],
                scale_gradient,
                config,
                "accumulate CRNN recurrent scale gradient",
            )?;
        }

        let input_activation_input = cache.input_activation_inputs[step].clone();
        let input_delta = backend
            .activation_backward_f32(&input_activation_input, &state_gradient, activation, config)
            .map_err(|err| backend_error("CRNN input activation backward", err))?;
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
        let (step_input_gradient, input_weight_gradient, input_bias_gradient, input_scale_gradient) =
            backward_recurrent_convolution_projection_device(
                backend,
                &input_spec,
                &params.input,
                &input_weights,
                &step_input_gpu,
                input_step_shape,
                hidden_shape,
                &input_delta,
                layer_index,
                "input",
                config,
            )?;
        accumulate_backend_gradient(
            backend,
            &mut weight_gradients[0],
            input_weight_gradient,
            config,
            "accumulate CRNN input weight gradient",
        )?;
        accumulate_backend_gradient(
            backend,
            &mut bias_gradients[0],
            input_bias_gradient,
            config,
            "accumulate CRNN input beta gradient",
        )?;
        if let Some(scale_gradient) = input_scale_gradient {
            accumulate_backend_gradient(
                backend,
                &mut scale_gradients[0],
                scale_gradient,
                config,
                "accumulate CRNN input scale gradient",
            )?;
        }
        input_gradients[step] = Some(step_input_gradient);

        if spec.shortcut {
            previous_state_gradient = backend
                .vector_add_f32(&previous_state_gradient, &state_gradient, config)
                .map_err(|err| backend_error("CRNN shortcut state backward", err))?;
        }
        if step > 0 {
            future_state_gradient = previous_state_gradient;
        }
    }

    let input_gradient_refs: Vec<_> = input_gradients
        .iter()
        .map(|gradient| {
            gradient.as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing CRNN input gradient timestep".to_owned())
            })
        })
        .collect::<Result<_>>()?;
    let input_gradient_gpu = backend
        .sequence_concat_f32(
            &input_gradient_refs,
            &tensor_backend_shape(input_shape),
            sequence_batch * input_step_shape.element_count(),
            config,
        )
        .map_err(|err| backend_error("assemble CRNN input gradient", err))?;
    for index in 0..3 {
        let weights = backend
            .download_f32(weight_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated CRNN weight gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download accumulated CRNN weight gradient", err))?;
        let biases = backend
            .download_f32(bias_gradients[index].as_ref().ok_or_else(|| {
                Error::InvalidConfig("missing accumulated CRNN bias gradient".to_owned())
            })?)
            .map_err(|err| backend_error("download accumulated CRNN bias gradient", err))?;
        let gradient = match index {
            0 => &mut gradients.input,
            1 => &mut gradients.self_params,
            _ => &mut gradients.output,
        };
        if weights.len() != gradient.weights.len() || biases.len() != gradient.biases.len() {
            return Err(Error::InvalidConfig(
                "accumulated CRNN parameter gradient shape mismatch".to_owned(),
            ));
        }
        gradient.weights = weights;
        gradient.biases = biases;
        match (&mut gradient.batch_norm, &scale_gradients[index]) {
            (Some(batch_norm), Some(scale_gradient)) => {
                let scales = backend
                    .download_f32(scale_gradient)
                    .map_err(|err| backend_error("download CRNN scale gradient", err))?;
                if scales.len() != batch_norm.scales.len() {
                    return Err(Error::InvalidConfig(format!(
                        "CRNN scale gradient shape mismatch at layer #{layer_index}"
                    )));
                }
                batch_norm.scales = scales;
            }
            (None, None) => {}
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "CRNN batch-normalization gradient mismatch at layer #{layer_index}"
                )));
            }
        }
    }
    Ok((input_gradient_gpu, gradients))
}
