use super::*;

pub(super) fn forward_convolutional(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    forward_convolutional_with_cache(spec, params, input, output_shape).0
}

pub(super) fn convolution_output_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    let stride = stride.max(1);
    (input + padding * 2).saturating_sub(size) / stride + 1
}

pub(super) fn apply_convolution_channel_normalization(
    output: &mut Tensor,
    activation_input: &Tensor,
    activation: &Activation,
) {
    let Some(normalization) = activation.to_channel_normalization() else {
        return;
    };
    assert_eq!(output.shape, activation_input.shape);
    output.data = lightnet_cuda::host_ops::normalize_channels(
        &activation_input.data,
        output.shape.batch,
        output.shape.channels,
        output.shape.width * output.shape.height,
        normalization,
    );
}

pub(super) fn activate_convolution_value(activation: &Activation, value: f32) -> f32 {
    if activation.to_channel_normalization().is_some() {
        value
    } else {
        activate(activation.clone(), value)
    }
}

pub(super) fn forward_dsconv_with_cache(
    spec: &crate::model::DsConvSpec,
    params: &DsConvParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Tensor, DsConvRuntimeCache)> {
    let (raw_output, math) = crate::dsconv::dsconv_forward(
        input,
        output_shape,
        spec,
        &params.collapse.weights,
        &params.collapse.biases,
        &params.offset_weights,
        &params.offset_biases,
    )?;
    let mut activation_input = Tensor::zeros(output_shape);
    let mut output = Tensor::zeros(output_shape);
    for (index, raw) in raw_output.data.iter().copied().enumerate() {
        activation_input.data[index] = raw;
        output.data[index] = activate_convolution_value(&spec.activation, raw);
    }
    apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
    Ok((output, activation_input, DsConvRuntimeCache { math }))
}

pub(crate) fn graph_conv_aggregation_geometry(
    spec: &GraphConvSpec,
    input: TensorShape,
    output: TensorShape,
) -> lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry {
    lightnet_cuda::host_ops::graph_conv::GraphConvAggregationGeometry {
        batch: input.batch,
        channels: input.channels,
        input_height: input.height,
        input_width: input.width,
        output_height: output.height,
        output_width: output.width,
        size: spec.size,
        stride_x: spec.stride_x,
        stride_y: spec.stride_y,
        dilation: spec.dilation,
        padding: spec.padding,
    }
}

pub(crate) fn forward_graph_conv_with_cache(
    spec: &GraphConvSpec,
    params: &GraphConvParams,
    input: &Tensor,
    output_shape: TensorShape,
    training: bool,
) -> Result<(Tensor, Tensor, GraphConvRuntimeCache)> {
    let math = crate::graph_conv::graph_conv_forward(
        spec,
        input,
        output_shape,
        &params.projection.weights,
        params.self_weights.as_deref(),
        params.edge_kernel.as_deref(),
        params.edge_biases.as_deref(),
    )?;
    let activation_input = if let Some(batch_norm) = &params.projection.batch_norm {
        let mut batch_norm = batch_norm.clone();
        batch_norm.biases.clone_from(&params.projection.biases);
        forward_batchnorm_with_cache(&batch_norm, &math.raw_output, training)?.0
    } else {
        let mut biased = math.raw_output.clone();
        for batch in 0..output_shape.batch {
            for filter in 0..output_shape.channels {
                let start = (batch * output_shape.channels + filter)
                    * output_shape.height
                    * output_shape.width;
                for value in
                    &mut biased.data[start..start + output_shape.height * output_shape.width]
                {
                    *value += params.projection.biases[filter];
                }
            }
        }
        biased
    };
    let mut output = Tensor::zeros(output_shape);
    for (value, preactivation) in output.data.iter_mut().zip(&activation_input.data) {
        *value = activate_convolution_value(&spec.activation, *preactivation);
    }
    apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
    // Keep the preactivation independent from the BN input used in the math
    // cache; backward needs both tensors for activation and BN derivatives.
    let runtime_cache = GraphConvRuntimeCache { math, training };
    Ok((output, activation_input, runtime_cache))
}

pub(super) fn wmhf_internal_convolution_spec(
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
    activation: Activation,
) -> ConvolutionalSpec {
    ConvolutionalSpec {
        filters,
        size,
        stride_x: 1,
        stride_y: 1,
        dilation: 1,
        padding: size / 2,
        groups,
        share_index: None,
        batch_normalize,
        activation,
    }
}

pub(crate) fn forward_wmhf_convolution(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
    training: bool,
) -> Result<WmhfConvRuntimeCache> {
    let raw_output = raw_convolution_output(spec, params, input, output_shape);
    let activation_input = if let Some(batch_norm) = &params.batch_norm {
        let mut effective_bn = batch_norm.clone();
        effective_bn.biases.clone_from(&params.biases);
        forward_batchnorm_with_cache(&effective_bn, &raw_output, training)?.0
    } else {
        let mut biased = raw_output.clone();
        for b in 0..output_shape.batch {
            for channel in 0..output_shape.channels {
                let start = (b * output_shape.channels + channel)
                    * output_shape.height
                    * output_shape.width;
                for value in
                    &mut biased.data[start..start + output_shape.height * output_shape.width]
                {
                    *value += params.biases[channel];
                }
            }
        }
        biased
    };
    let mut output = Tensor::zeros(output_shape);
    for (value, preactivation) in output.data.iter_mut().zip(&activation_input.data) {
        *value = activate_convolution_value(&spec.activation, *preactivation);
    }
    Ok(WmhfConvRuntimeCache {
        input: input.clone(),
        raw_output,
        activation_input,
        output,
    })
}

pub(crate) fn backward_wmhf_convolution(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    cache: &WmhfConvRuntimeCache,
    output_gradient: &Tensor,
    training: bool,
) -> Result<(Tensor, ConvolutionalParams)> {
    let output_shape = cache.output.shape;
    let delta = apply_output_activation_gradient(
        &cache.output,
        output_gradient,
        &spec.activation,
        "wmhf sub-convolution",
        Some(&cache.activation_input),
    )?;
    let (raw_gradient, bias_gradient, batch_norm_gradient) =
        if let Some(batch_norm) = &params.batch_norm {
            let mut effective_bn = batch_norm.clone();
            effective_bn.biases.clone_from(&params.biases);
            let (raw_gradient, mut batch_norm_gradient) =
                backward_batchnorm(&effective_bn, &cache.raw_output, &delta, training)?;
            let bias_gradient = std::mem::take(&mut batch_norm_gradient.biases);
            (raw_gradient, bias_gradient, Some(batch_norm_gradient))
        } else {
            let mut bias_gradient = vec![0.0; output_shape.channels];
            for b in 0..output_shape.batch {
                for channel in 0..output_shape.channels {
                    for y in 0..output_shape.height {
                        for x in 0..output_shape.width {
                            bias_gradient[channel] += delta.get(b, channel, y, x);
                        }
                    }
                }
            }
            (delta, bias_gradient, None)
        };

    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = cache.input.shape.channels / spec.groups.max(1);
    let mut input_gradient = Tensor::zeros(cache.input.shape);
    let mut weight_gradient = vec![0.0; params.weights.len()];
    for b in 0..cache.input.shape.batch {
        for filter in 0..spec.filters {
            let group = filter / filters_per_group.max(1);
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let grad_value = raw_gradient.get(b, filter, oy, ox);
                    for ic in 0..channels_per_group {
                        let channel = group * channels_per_group + ic;
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let src_y = (oy * spec.stride_y + ky * spec.dilation) as isize
                                    - spec.padding as isize;
                                let src_x = (ox * spec.stride_x + kx * spec.dilation) as isize
                                    - spec.padding as isize;
                                if src_y < 0
                                    || src_y >= cache.input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= cache.input.shape.width as isize
                                {
                                    continue;
                                }
                                let weight_index = ((filter * channels_per_group + ic) * spec.size
                                    + ky)
                                    * spec.size
                                    + kx;
                                let src_y = src_y as usize;
                                let src_x = src_x as usize;
                                weight_gradient[weight_index] +=
                                    cache.input.get(b, channel, src_y, src_x) * grad_value;
                                let input_index = cache.input.offset(b, channel, src_y, src_x);
                                input_gradient.data[input_index] +=
                                    params.weights[weight_index] * grad_value;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((
        input_gradient,
        ConvolutionalParams {
            biases: bias_gradient,
            batch_norm: batch_norm_gradient,
            weights: weight_gradient,
        },
    ))
}

pub(crate) fn forward_wmhf_with_cache(
    spec: &crate::model::WmhfSpec,
    params: &WmhfParams,
    input: &Tensor,
    output_shape: TensorShape,
    training: bool,
) -> Result<(Tensor, Tensor, WmhfRuntimeCache)> {
    let batch = input.shape.batch;
    let spatial = input.shape.height * input.shape.width;
    let projected_shape = output_shape;
    let local_shape = TensorShape {
        batch,
        channels: spec.local_channels,
        height: input.shape.height,
        width: input.shape.width,
    };
    let global_shape = TensorShape {
        batch,
        channels: spec.global_channels,
        height: input.shape.height,
        width: input.shape.width,
    };
    let mut pre_spec = wmhf_internal_convolution_spec(
        spec.filters,
        1,
        1,
        spec.batch_normalize,
        Activation::Linear,
    );
    pre_spec.padding = 0;
    let pre = forward_wmhf_convolution(&pre_spec, &params.pre, input, projected_shape, training)?;

    let local_input = Tensor::from_data(
        local_shape,
        crate::wmhf::split_channels(
            &pre.output.data,
            batch,
            spec.filters,
            spatial,
            spec.identity_channels,
            spec.local_channels,
        )?,
    )?;
    let global_input = Tensor::from_data(
        global_shape,
        crate::wmhf::split_channels(
            &pre.output.data,
            batch,
            spec.filters,
            spatial,
            spec.identity_channels + spec.local_channels,
            spec.global_channels,
        )?,
    )?;

    let dw3_spec = wmhf_internal_convolution_spec(
        spec.local_channels,
        3,
        spec.local_channels,
        false,
        Activation::Linear,
    );
    let dw5_spec = wmhf_internal_convolution_spec(
        spec.local_channels,
        5,
        spec.local_channels,
        false,
        Activation::Linear,
    );
    let dw7_spec = wmhf_internal_convolution_spec(
        spec.local_channels,
        7,
        spec.local_channels,
        false,
        Activation::Linear,
    );
    let local_dw3 = forward_wmhf_convolution(
        &dw3_spec,
        &params.local_dw3,
        &local_input,
        local_shape,
        training,
    )?;
    let local_dw5 = forward_wmhf_convolution(
        &dw5_spec,
        &params.local_dw5,
        &local_input,
        local_shape,
        training,
    )?;
    let local_dw7 = forward_wmhf_convolution(
        &dw7_spec,
        &params.local_dw7,
        &local_input,
        local_shape,
        training,
    )?;
    let local_cat = Tensor::from_data(
        TensorShape {
            channels: 3 * spec.local_channels,
            ..local_shape
        },
        crate::wmhf::concat_local(
            &local_dw3.output.data,
            &local_dw5.output.data,
            &local_dw7.output.data,
            batch,
            spec.local_channels,
            spatial,
        )?,
    )?;
    let local_mix_spec = wmhf_internal_convolution_spec(
        spec.local_channels,
        1,
        1,
        spec.batch_normalize,
        Activation::Linear,
    );
    let local_mix = forward_wmhf_convolution(
        &local_mix_spec,
        &params.local_mix,
        &local_cat,
        local_shape,
        training,
    )?;

    let dwt = crate::wmhf::haar_dwt(
        &global_input.data,
        batch,
        spec.global_channels,
        input.shape.height,
        input.shape.width,
    )?;
    let scan_output = crate::wmhf::scan4_forward(
        &dwt.ll,
        &params.scan_weights,
        batch,
        spec.global_channels,
        dwt.height2,
        dwt.width2,
    )?;
    let mut inverse_coeffs = dwt.clone();
    inverse_coeffs.ll.clone_from(&scan_output);
    let global_values = crate::wmhf::haar_iwt(&inverse_coeffs)?;
    let energy = Tensor::from_data(
        global_shape,
        crate::wmhf::high_frequency_energy_upsample(&dwt)?,
    )?;
    let gate_spec = wmhf_internal_convolution_spec(spec.filters, 1, 1, false, Activation::Logistic);
    let hf_gate = forward_wmhf_convolution(
        &gate_spec,
        &params.hf_gate,
        &energy,
        projected_shape,
        training,
    )?;
    let fuse_input = Tensor::from_data(
        projected_shape,
        crate::wmhf::concat_fuse(
            &pre.output.data,
            &local_mix.output.data,
            &global_values,
            batch,
            spec.filters,
            spec.identity_channels,
            spec.local_channels,
            spatial,
        )?,
    )?;
    let fuse_spec = wmhf_internal_convolution_spec(
        spec.filters,
        1,
        1,
        spec.batch_normalize,
        Activation::Linear,
    );
    let fuse = forward_wmhf_convolution(
        &fuse_spec,
        &params.fuse,
        &fuse_input,
        output_shape,
        training,
    )?;
    let use_shortcut = spec.shortcut && input.shape.channels == spec.filters;
    let mut activation_input = fuse.output.clone();
    let mut output = Tensor::zeros(output_shape);
    for index in 0..output.data.len() {
        activation_input.data[index] +=
            spec.freq_scale * hf_gate.output.data[index] * pre.output.data[index];
        if use_shortcut {
            activation_input.data[index] += input.data[index];
        }
        output.data[index] = activate(spec.activation.clone(), activation_input.data[index]);
    }
    Ok((
        output,
        activation_input,
        WmhfRuntimeCache {
            pre,
            local_dw3,
            local_dw5,
            local_dw7,
            local_mix,
            hf_gate,
            fuse,
            dwt,
            training,
        },
    ))
}

pub(crate) fn forward_deform_conv_with_cache(
    spec: &DeformConvSpec,
    params: &DeformConvParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Tensor, DeformConvRuntimeCache)> {
    let (raw_output, math) = crate::deform_conv::deform_conv_forward(
        input,
        spec,
        output_shape,
        &params.main.weights,
        &params.offset_weights,
        &params.offset_biases,
        params.mask_weights.as_deref(),
        params.mask_biases.as_deref(),
    )?;
    let spatial = output_shape.width * output_shape.height;
    let mut activation_input = Tensor::zeros(output_shape);
    let mut output = Tensor::zeros(output_shape);
    let mut output_clamp_slope = vec![0.0; raw_output.data.len()];
    let mut output_nondiff_boundary = vec![false; raw_output.data.len()];
    for b in 0..output_shape.batch {
        for channel in 0..output_shape.channels {
            let bn = params.main.batch_norm.as_ref();
            let inv_std =
                bn.map(|bn| 1.0 / (bn.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt());
            for q in 0..spatial {
                let index = (b * output_shape.channels + channel) * spatial + q;
                let raw = raw_output.data[index];
                let preactivation = if let (Some(bn), Some(inv_std)) = (bn, inv_std) {
                    (raw - bn.rolling_mean[channel]) * inv_std * bn.scales[channel]
                        + params.main.biases[channel]
                } else {
                    raw + params.main.biases[channel]
                };
                activation_input.data[index] = preactivation;
                let activated = activate_convolution_value(&spec.activation, preactivation);
                output_clamp_slope[index] =
                    if activated.is_finite() && activated > -10_000.0 && activated < 10_000.0 {
                        1.0
                    } else {
                        0.0
                    };
                output_nondiff_boundary[index] =
                    !activated.is_finite() || activated.abs() == 10_000.0;
                output.data[index] = if activated.is_finite() {
                    activated.clamp(-10_000.0, 10_000.0)
                } else {
                    0.0
                };
            }
        }
    }
    apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
    Ok((
        output,
        activation_input,
        DeformConvRuntimeCache {
            math,
            output_clamp_slope,
            output_nondiff_boundary,
        },
    ))
}

pub(super) fn forward_dcnv4_with_cache(
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Tensor, crate::dcnv4::Dcnv4ForwardCache)> {
    let (raw_output, cache) = crate::dcnv4::dcnv4_forward(
        &input.data,
        input.shape,
        spec,
        &params.offset_weights,
        &params.offset_biases,
    )?;
    let spatial = output_shape.width * output_shape.height;
    let mut activation_input = Tensor::zeros(output_shape);
    let mut output = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for channel in 0..output_shape.channels {
            let bn = params.batch_norm.as_ref();
            let inv_std =
                bn.map(|bn| 1.0 / (bn.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt());
            for q in 0..spatial {
                let index = (b * output_shape.channels + channel) * spatial + q;
                let raw = raw_output[index];
                let preactivation = if let (Some(bn), Some(inv_std)) = (bn, inv_std) {
                    (raw - bn.rolling_mean[channel]) * inv_std * bn.scales[channel]
                } else {
                    raw + params.biases[channel]
                };
                activation_input.data[index] = preactivation;
                output.data[index] = activate_convolution_value(&spec.activation, preactivation);
            }
        }
    }
    apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
    Ok((output, activation_input, cache))
}

pub(crate) fn forward_convolutional_with_cache(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> (Tensor, Option<Tensor>) {
    // This direct loop is the correctness reference for convolution.  A
    // backend may replace it with im2col/GEMM or a CUDA kernel, but grouped
    // channel mapping, dilation, padding, and output order must remain equal.
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);

    // The cfg compatibility runner constructs networks before loading weights,
    // so the default parameter vectors are entirely zero.  For finite input
    // values every convolution product is consequently zero; retain the bias,
    // batch-normalization, and activation work while avoiding the otherwise
    // dominant kernel loop.  Non-finite inputs preserve IEEE multiplication
    // semantics through the reference loop below.
    if params.weights.iter().all(|weight| *weight == 0.0)
        && input.data.iter().all(|value| value.is_finite())
    {
        let channel_area = output_shape.width * output_shape.height;
        for b in 0..input.shape.batch {
            for filter in 0..spec.filters {
                let biased =
                    normalize_value(0.0, filter, &params.biases, params.batch_norm.as_ref());
                let activated = activate_convolution_value(&spec.activation, biased);
                let start = b * output.batch_stride() + filter * channel_area;
                activation_input.data[start..start + channel_area].fill(biased);
                output.data[start..start + channel_area].fill(activated);
            }
        }
        apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
        return (output, Some(activation_input));
    }

    for b in 0..input.shape.batch {
        for filter in 0..spec.filters {
            let group = filter / filters_per_group.max(1);
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut acc = 0.0;
                    for ic in 0..channels_per_group {
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let in_y = oy * spec.stride_y + ky * spec.dilation;
                                let in_x = ox * spec.stride_x + kx * spec.dilation;
                                let src_y = in_y as isize - spec.padding as isize;
                                let src_x = in_x as isize - spec.padding as isize;

                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }

                                let channel = group * channels_per_group + ic;
                                let input_value =
                                    input.get(b, channel, src_y as usize, src_x as usize);
                                let weight_index = ((filter * channels_per_group + ic) * spec.size
                                    + ky)
                                    * spec.size
                                    + kx;
                                acc += input_value * params.weights[weight_index];
                            }
                        }
                    }

                    let biased =
                        normalize_value(acc, filter, &params.biases, params.batch_norm.as_ref());
                    activation_input.set(b, filter, oy, ox, biased);
                    output.set(
                        b,
                        filter,
                        oy,
                        ox,
                        activate_convolution_value(&spec.activation, biased),
                    );
                }
            }
        }
    }

    apply_convolution_channel_normalization(&mut output, &activation_input, &spec.activation);
    (output, Some(activation_input))
}

pub(super) fn raw_convolution_output(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    let mut output = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for filter in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let value =
                        convolution_raw_value(spec, &params.weights, input, b, filter, y, x);
                    output.set(b, filter, y, x, value);
                }
            }
        }
    }
    output
}

pub(super) fn channel_mean_variance(tensor: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let spatial = tensor.shape.width * tensor.shape.height;
    let sample_count = tensor.shape.batch * spatial;
    let mut mean = vec![0.0; tensor.shape.channels];
    for b in 0..tensor.shape.batch {
        for (channel, slot) in mean.iter_mut().enumerate() {
            for idx in 0..spatial {
                let y = idx / tensor.shape.width;
                let x = idx % tensor.shape.width;
                *slot += tensor.get(b, channel, y, x);
            }
        }
    }
    if sample_count != 0 {
        let scale = 1.0 / sample_count as f32;
        for value in &mut mean {
            *value *= scale;
        }
    }

    let mut variance = vec![0.0; tensor.shape.channels];
    for b in 0..tensor.shape.batch {
        for (channel, slot) in variance.iter_mut().enumerate() {
            for idx in 0..spatial {
                let y = idx / tensor.shape.width;
                let x = idx % tensor.shape.width;
                let diff = tensor.get(b, channel, y, x) - mean[channel];
                *slot += diff * diff;
            }
        }
    }
    if sample_count > 1 {
        let scale = 1.0 / (sample_count - 1) as f32;
        for value in &mut variance {
            *value *= scale;
        }
    }

    (mean, variance)
}

pub(super) fn update_batch_norm_running_stats(
    batch_norm: &mut BatchNormParams,
    mean: &[f32],
    variance: &[f32],
    keep: f32,
) -> Result<()> {
    if batch_norm.rolling_mean.len() != mean.len()
        || batch_norm.rolling_variance.len() != variance.len()
    {
        return Err(Error::InvalidConfig(format!(
            "batch-norm running-stat mismatch: rolling mean {}, rolling variance {}, mean {}, variance {}",
            batch_norm.rolling_mean.len(),
            batch_norm.rolling_variance.len(),
            mean.len(),
            variance.len()
        )));
    }
    let update = 1.0 - keep;
    for (dst, src) in batch_norm.rolling_mean.iter_mut().zip(mean) {
        *dst = keep * *dst + update * *src;
    }
    for (dst, src) in batch_norm.rolling_variance.iter_mut().zip(variance) {
        *dst = keep * *dst + update * *src;
    }
    Ok(())
}

pub(super) fn update_connected_batch_norm_from_input(
    params: &mut ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<()> {
    let raw_output = raw_connected_output(params, input, output_shape);
    let Some(batch_norm) = params.batch_norm.as_mut() else {
        return Ok(());
    };
    let (mean, variance) = channel_mean_variance(&raw_output);
    update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)
}

pub(super) fn update_convolutional_batch_norm_from_input(
    params: &mut ConvolutionalParams,
    spec: &ConvolutionalSpec,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<()> {
    let raw_output = raw_convolution_output(spec, params, input, output_shape);
    let Some(batch_norm) = params.batch_norm.as_mut() else {
        return Ok(());
    };
    let (mean, variance) = channel_mean_variance(&raw_output);
    update_batch_norm_running_stats(batch_norm, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)
}

pub(super) fn update_recursive_body_batch_norm_running_stats(
    outer_layer: &crate::model::Layer,
    spec: &crate::model::RecursiveBlockSpec,
    params: &mut RecursiveBlockParams,
    cache: &RecursiveBlockForwardCache,
) -> Result<()> {
    if cache.body_passes.len() != spec.loops
        || cache.hidden_inputs.len() != spec.loops
        || params.body.len() != spec.body.len()
    {
        return Err(Error::InvalidConfig(format!(
            "[recursive_block] layer #{} body BN cache/parameter layout mismatch",
            outer_layer.index
        )));
    }
    for step in 0..spec.loops {
        let body_forward = &cache.body_passes[step];
        for (body_layer, body_params) in spec.body.iter().zip(&mut params.body) {
            if body_layer.index >= body_forward.layer_outputs.len() {
                return Err(Error::InvalidConfig(format!(
                    "[recursive_block] layer #{} body pass {step} lacks output for nested layer #{}",
                    outer_layer.index, body_layer.index
                )));
            }
            let body_input = if body_layer.index == 0 {
                &cache.hidden_inputs[step]
            } else {
                body_forward.layer_outputs.get(body_layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[recursive_block] layer #{} body pass {step} lacks input for nested layer #{}",
                        outer_layer.index, body_layer.index
                    ))
                })?
            };
            match (&body_layer.kind, body_params) {
                (LayerKind::Convolutional(body_spec), LayerParams::Convolutional(body_params))
                    if body_spec.batch_normalize =>
                {
                    let raw = raw_convolution_output(
                        body_spec,
                        body_params,
                        body_input,
                        body_layer.output_shape,
                    );
                    let (mean, variance) = channel_mean_variance(&raw);
                    let Some(batch_norm) = body_params.batch_norm.as_mut() else {
                        return Err(Error::InvalidConfig(format!(
                            "[recursive_block] layer #{} nested conv #{} has no batch-norm state",
                            outer_layer.index, body_layer.index
                        )));
                    };
                    update_batch_norm_running_stats(
                        batch_norm,
                        &mean,
                        &variance,
                        BATCH_NORM_ROLLING_KEEP_CPU,
                    )?;
                }
                (LayerKind::Connected(body_spec), LayerParams::Connected(body_params))
                    if body_spec.batch_normalize =>
                {
                    let raw =
                        raw_connected_output(body_params, body_input, body_layer.output_shape);
                    let (mean, variance) = channel_mean_variance(&raw);
                    let Some(batch_norm) = body_params.batch_norm.as_mut() else {
                        return Err(Error::InvalidConfig(format!(
                            "[recursive_block] layer #{} nested connected #{} has no batch-norm state",
                            outer_layer.index, body_layer.index
                        )));
                    };
                    update_batch_norm_running_stats(
                        batch_norm,
                        &mean,
                        &variance,
                        BATCH_NORM_ROLLING_KEEP_CPU,
                    )?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) fn update_recurrent_batch_norm_running_stats(
    network: &Network,
    params: &mut [LayerParams],
    input: &Tensor,
    forward: &ForwardResult,
) -> Result<()> {
    for layer in &network.layers {
        let current_input = if layer.index == 0 {
            input
        } else {
            forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous layer output for recurrent batch-norm stats at layer #{}",
                    layer.index
                ))
            })?
        };
        match (&layer.kind, params.get_mut(layer.index)) {
            (LayerKind::Gru(spec), Some(LayerParams::Gru(recurrent))) if spec.batch_normalize => {
                let tape = crate::layers::gru::cpu::tape(spec, recurrent, current_input, layer.output_shape)?;
                let batch = current_input.shape.batch / spec.steps;
                let input_shape = TensorShape::new(current_input.shape.width, current_input.shape.height, current_input.shape.channels, batch);
                let state_shape = TensorShape::new(1, 1, spec.output, batch);
                for step in tape {
                    let x = Tensor::from_data(input_shape, step.x)?;
                    let previous = Tensor::from_data(state_shape, step.previous)?;
                    let reset = Tensor::from_data(state_shape, step.reset)?;
                    for (p, source) in crate::layers::gru::cpu::projections_mut(recurrent).into_iter().zip([&x, &previous, &x, &previous, &x, &reset]) {
                        update_connected_batch_norm_from_input(p, source, state_shape)?;
                    }
                }
            }
            (LayerKind::Rnn(spec), Some(LayerParams::Rnn(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let hidden_shape = TensorShape::new(1, 1, spec.hidden, sequence_batch);
                let output_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Rnn(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing RNN cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.input,
                        &step_input,
                        hidden_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.self_params,
                        &cache.previous_states[step],
                        hidden_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.output,
                        &cache.states[step],
                        output_shape,
                    )?;
                }
            }
            (LayerKind::Lstm(spec), Some(LayerParams::Lstm(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let state_shape = TensorShape::new(1, 1, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Lstm(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing LSTM cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wf,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wi,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wg,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.wo,
                        &cache.previous_states[step],
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.uf,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.ui,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.ug,
                        &step_input,
                        state_shape,
                    )?;
                    update_connected_batch_norm_from_input(
                        &mut recurrent.uo,
                        &step_input,
                        state_shape,
                    )?;
                }
            }
            (LayerKind::Crnn(spec), Some(LayerParams::Crnn(recurrent))) if spec.batch_normalize => {
                let sequence_batch =
                    recurrent_sequence_batch(current_input.shape.batch, spec.steps)?;
                let input_shape = TensorShape::new(
                    current_input.shape.width,
                    current_input.shape.height,
                    current_input.shape.channels,
                    sequence_batch,
                );
                let width = convolution_output_dim(
                    current_input.shape.width,
                    spec.padding,
                    spec.size,
                    spec.stride,
                );
                let height = convolution_output_dim(
                    current_input.shape.height,
                    spec.padding,
                    spec.size,
                    spec.stride,
                );
                let hidden_shape = TensorShape::new(width, height, spec.hidden, sequence_batch);
                let output_shape = TensorShape::new(width, height, spec.output, sequence_batch);
                let Some(Some(RecurrentForwardCache::Crnn(cache))) =
                    forward.recurrent_caches.get(layer.index)
                else {
                    return Err(Error::InvalidConfig(format!(
                        "missing CRNN cache for recurrent batch-norm stats at layer #{}",
                        layer.index
                    )));
                };
                let input_spec = recurrent_convolutional_spec(spec, spec.hidden);
                let self_spec = recurrent_convolutional_spec(spec, spec.hidden);
                let output_spec = recurrent_convolutional_spec(spec, spec.output);
                for step in 0..spec.steps {
                    let step_input =
                        sequence_slice(current_input, step, sequence_batch, input_shape)?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.input,
                        &input_spec,
                        &step_input,
                        hidden_shape,
                    )?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.self_params,
                        &self_spec,
                        &cache.previous_states[step],
                        hidden_shape,
                    )?;
                    update_convolutional_batch_norm_from_input(
                        &mut recurrent.output,
                        &output_spec,
                        &cache.states[step],
                        output_shape,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn forward_local_with_cache(
    spec: &LocalSpec,
    params: &LocalParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    let locations = output_shape.width * output_shape.height;
    let kernel_values = input.shape.channels * spec.size * spec.size;
    let expected_biases = locations * spec.filters;
    let expected_weights = expected_biases * kernel_values;
    if params.biases.len() != expected_biases || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "local parameter shape mismatch: expected {expected_biases} biases and {expected_weights} weights, got {} and {}",
            params.biases.len(),
            params.weights.len()
        )));
    }

    let mut activation_input = Tensor::zeros(output_shape);
    // As with convolution, unweighted cfg construction has zero local kernels.
    // Preserve the per-location bias and activation exactly, while avoiding
    // spatial kernel traversal for finite inputs.  Keep the reference path for
    // non-finite inputs so IEEE zero-times-NaN/Inf behavior is unchanged.
    if params.weights.iter().all(|weight| *weight == 0.0)
        && input.data.iter().all(|value| value.is_finite())
    {
        for batch in 0..input.shape.batch {
            for out_y in 0..output_shape.height {
                for out_x in 0..output_shape.width {
                    let location = out_y * output_shape.width + out_x;
                    for filter in 0..spec.filters {
                        let parameter = location * spec.filters + filter;
                        activation_input.set(batch, filter, out_y, out_x, params.biases[parameter]);
                    }
                }
            }
        }
        let mut output = activation_input.clone();
        for value in &mut output.data {
            *value = activate(spec.activation.clone(), *value);
        }
        return Ok((output, Some(activation_input)));
    }

    for batch in 0..input.shape.batch {
        for out_y in 0..output_shape.height {
            for out_x in 0..output_shape.width {
                let location = out_y * output_shape.width + out_x;
                for filter in 0..spec.filters {
                    let parameter = location * spec.filters + filter;
                    let weight_base = parameter * kernel_values;
                    let mut value = params.biases[parameter];
                    for channel in 0..input.shape.channels {
                        for kernel_y in 0..spec.size {
                            for kernel_x in 0..spec.size {
                                let source_x = out_x * spec.stride_x + kernel_x;
                                let source_y = out_y * spec.stride_y + kernel_y;
                                let input_x = source_x as isize - spec.padding as isize;
                                let input_y = source_y as isize - spec.padding as isize;
                                if input_x < 0
                                    || input_y < 0
                                    || input_x as usize >= input.shape.width
                                    || input_y as usize >= input.shape.height
                                {
                                    continue;
                                }
                                let kernel_index =
                                    (channel * spec.size + kernel_y) * spec.size + kernel_x;
                                value += params.weights[weight_base + kernel_index]
                                    * input.get(batch, channel, input_y as usize, input_x as usize);
                            }
                        }
                    }
                    activation_input.set(batch, filter, out_y, out_x, value);
                }
            }
        }
    }
    let mut output = activation_input.clone();
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok((output, Some(activation_input)))
}

pub(super) fn backward_local(
    spec: &LocalSpec,
    params: &LocalParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    output: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, LocalParams)> {
    if output_gradient.shape != output_shape || output.shape != output_shape {
        return Err(Error::InvalidConfig(
            "local backward output shape mismatch".to_owned(),
        ));
    }
    let locations = output_shape.width * output_shape.height;
    let kernel_values = input.shape.channels * spec.size * spec.size;
    let expected_biases = locations * spec.filters;
    let expected_weights = expected_biases * kernel_values;
    if params.biases.len() != expected_biases || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(
            "local backward parameter shape mismatch".to_owned(),
        ));
    }
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "local",
        activation_input,
    )?;
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradients = LocalParams {
        biases: vec![0.0; expected_biases],
        weights: vec![0.0; expected_weights],
    };
    for batch in 0..input.shape.batch {
        for out_y in 0..output_shape.height {
            for out_x in 0..output_shape.width {
                let location = out_y * output_shape.width + out_x;
                for filter in 0..spec.filters {
                    let parameter = location * spec.filters + filter;
                    let weight_base = parameter * kernel_values;
                    let delta_value = delta.get(batch, filter, out_y, out_x);
                    gradients.biases[parameter] += delta_value;
                    for channel in 0..input.shape.channels {
                        for kernel_y in 0..spec.size {
                            for kernel_x in 0..spec.size {
                                let source_x = out_x * spec.stride_x + kernel_x;
                                let source_y = out_y * spec.stride_y + kernel_y;
                                let input_x = source_x as isize - spec.padding as isize;
                                let input_y = source_y as isize - spec.padding as isize;
                                if input_x < 0
                                    || input_y < 0
                                    || input_x as usize >= input.shape.width
                                    || input_y as usize >= input.shape.height
                                {
                                    continue;
                                }
                                let kernel_index =
                                    (channel * spec.size + kernel_y) * spec.size + kernel_x;
                                let weight_index = weight_base + kernel_index;
                                let input_value =
                                    input.get(batch, channel, input_y as usize, input_x as usize);
                                gradients.weights[weight_index] += input_value * delta_value;
                                let current = input_gradient.get(
                                    batch,
                                    channel,
                                    input_y as usize,
                                    input_x as usize,
                                );
                                input_gradient.set(
                                    batch,
                                    channel,
                                    input_y as usize,
                                    input_x as usize,
                                    current + params.weights[weight_index] * delta_value,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((input_gradient, gradients))
}

pub(crate) fn forward_connected_with_cache(
    spec: &ConnectedSpec,
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> (Tensor, Option<Tensor>) {
    // Darknet stores connected weights as [output][input].  Keeping this
    // layout makes binary weight loading and the transposed backward GEMM
    // match connected_layer.cpp without a hidden transpose.
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    let inputs = input.shape.element_count();

    for b in 0..input.shape.batch {
        let src_offset = b * inputs;
        let src = &input.data[src_offset..src_offset + inputs];

        for out_idx in 0..spec.output {
            let weight_base = out_idx * inputs;
            let mut acc = 0.0;
            for (idx, src_value) in src.iter().enumerate() {
                acc += *src_value * params.weights[weight_base + idx];
            }

            let normalized =
                normalize_value(acc, out_idx, &params.biases, params.batch_norm.as_ref());
            activation_input.data[b * spec.output + out_idx] = normalized;
            let value = activate(spec.activation.clone(), normalized);
            output.data[b * spec.output + out_idx] = value;
        }
    }

    (output, Some(activation_input))
}

pub(super) fn batchnorm_channel_stats(input: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let sample_count = (input.shape.batch * input.shape.width * input.shape.height).max(1);
    let mut mean = vec![0.0; input.shape.channels];
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    mean[channel] += input.get(b, channel, y, x);
                }
            }
        }
    }
    for value in &mut mean {
        *value /= sample_count as f32;
    }

    let mut variance = vec![0.0; input.shape.channels];
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let diff = input.get(b, channel, y, x) - mean[channel];
                    variance[channel] += diff * diff;
                }
            }
        }
    }
    for value in &mut variance {
        *value /= sample_count as f32;
    }
    (mean, variance)
}

pub(crate) fn validate_batchnorm_params(params: &BatchNormParams, channels: usize) -> Result<()> {
    if params.biases.len() != channels
        || params.scales.len() != channels
        || params.rolling_mean.len() != channels
        || params.rolling_variance.len() != channels
    {
        return Err(Error::InvalidConfig(format!(
            "batchnorm parameter length mismatch: expected {channels} channels, got biases {}, scales {}, rolling mean {}, rolling variance {}",
            params.biases.len(),
            params.scales.len(),
            params.rolling_mean.len(),
            params.rolling_variance.len()
        )));
    }
    Ok(())
}

pub(crate) fn folded_batchnorm_inference_params(
    params: &BatchNormParams,
    channels: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    validate_batchnorm_params(params, channels)?;
    let mut scales = Vec::with_capacity(channels);
    let mut biases = Vec::with_capacity(channels);
    for channel in 0..channels {
        let scale =
            params.scales[channel] / (params.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt();
        scales.push(scale);
        biases.push(params.biases[channel] - params.rolling_mean[channel] * scale);
    }
    Ok((scales, biases))
}

pub(super) fn forward_batchnorm_with_cache(
    params: &BatchNormParams,
    input: &Tensor,
    training: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    validate_batchnorm_params(params, input.shape.channels)?;
    let (mean, variance) = if training {
        batchnorm_channel_stats(input)
    } else {
        (params.rolling_mean.clone(), params.rolling_variance.clone())
    };
    let mut output = Tensor::zeros(input.shape);
    for b in 0..input.shape.batch {
        for channel in 0..input.shape.channels {
            let inv_std = 1.0 / (variance[channel] + BATCH_NORM_EPSILON).sqrt();
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let normalized = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    output.set(
                        b,
                        channel,
                        y,
                        x,
                        normalized * params.scales[channel] + params.biases[channel],
                    );
                }
            }
        }
    }
    // The cache is a training marker. Backward recomputes the batch statistics
    // from the saved graph input, while inference uses rolling statistics.
    Ok((output, training.then(|| input.clone())))
}

pub(super) fn backward_batchnorm(
    params: &BatchNormParams,
    input: &Tensor,
    output_gradient: &Tensor,
    training: bool,
) -> Result<(Tensor, BatchNormParams)> {
    if output_gradient.shape != input.shape {
        return Err(Error::InvalidConfig(format!(
            "batchnorm backward gradient shape mismatch: expected {}, got {}",
            input.shape, output_gradient.shape
        )));
    }
    validate_batchnorm_params(params, input.shape.channels)?;
    let (mean, variance) = if training {
        batchnorm_channel_stats(input)
    } else {
        (params.rolling_mean.clone(), params.rolling_variance.clone())
    };
    let count = (input.shape.batch * input.shape.width * input.shape.height).max(1) as f32;
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut gradient = BatchNormParams {
        biases: vec![0.0; input.shape.channels],
        scales: vec![0.0; input.shape.channels],
        rolling_mean: vec![0.0; input.shape.channels],
        rolling_variance: vec![0.0; input.shape.channels],
    };

    for channel in 0..input.shape.channels {
        let inv_std = 1.0 / (variance[channel] + BATCH_NORM_EPSILON).sqrt();
        let mut sum_gradient = 0.0;
        let mut sum_gradient_xhat = 0.0;
        for b in 0..input.shape.batch {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let dy = output_gradient.get(b, channel, y, x);
                    let xhat = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    gradient.biases[channel] += dy;
                    gradient.scales[channel] += dy * xhat;
                    sum_gradient += dy;
                    sum_gradient_xhat += dy * xhat;
                }
            }
        }

        for b in 0..input.shape.batch {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let dy = output_gradient.get(b, channel, y, x);
                    let xhat = (input.get(b, channel, y, x) - mean[channel]) * inv_std;
                    let dx = if training {
                        params.scales[channel] * inv_std / count
                            * (count * dy - sum_gradient - xhat * sum_gradient_xhat)
                    } else {
                        params.scales[channel] * inv_std * dy
                    };
                    input_gradient.set(b, channel, y, x, dx);
                }
            }
        }
    }
    Ok((input_gradient, gradient))
}
