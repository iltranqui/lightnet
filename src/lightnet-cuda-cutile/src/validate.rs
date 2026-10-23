use crate::error::CutileBackendError;

pub(crate) fn ensure_num_elements(shape: &[usize], len: usize) -> Result<(), CutileBackendError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| CutileBackendError::Shape("shape overflow".to_owned()))?;

    if expected != len {
        return Err(CutileBackendError::Shape(format!(
            "shape {:?} expects {expected} elements, got {len}",
            shape
        )));
    }

    Ok(())
}

pub(crate) fn ensure_same_shape(op: &str, lhs: &[i32], rhs: &[i32]) -> Result<(), CutileBackendError> {
    if lhs != rhs {
        return Err(CutileBackendError::Shape(format!(
            "{op} shape mismatch: {lhs:?} vs {rhs:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_bias_add_layout(
    x_len: usize,
    bias_len: usize,
    channels: usize,
    channel_stride: usize,
) -> Result<(usize, usize), CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "bias_add_f32 requires channels > 0".to_owned(),
        ));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "bias_add_f32 requires channel_stride > 0".to_owned(),
        ));
    }
    if bias_len != channels {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 bias length mismatch: expected {channels}, got {bias_len}"
        )));
    }
    if !x_len.is_multiple_of(channel_stride) {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 input length {x_len} is not divisible by channel_stride {channel_stride}"
        )));
    }

    let rows = x_len / channel_stride;
    if !rows.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "bias_add_f32 rows {rows} are not divisible by channels {channels}"
        )));
    }

    Ok((rows, channel_stride))
}

pub(crate) fn validate_bias_backward_layout(
    output_grad_len: usize,
    channels: usize,
    channel_stride: usize,
) -> Result<(usize, usize), CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "bias_backward_f32 requires channels > 0".to_owned(),
        ));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "bias_backward_f32 requires channel_stride > 0".to_owned(),
        ));
    }
    if !output_grad_len.is_multiple_of(channel_stride) {
        return Err(CutileBackendError::Shape(format!(
            "bias_backward_f32 output gradient length {output_grad_len} is not divisible by channel_stride {channel_stride}"
        )));
    }

    let rows = output_grad_len / channel_stride;
    if !rows.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "bias_backward_f32 rows {rows} are not divisible by channels {channels}"
        )));
    }

    Ok((rows, channel_stride))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RouteConcatLayout {
    pub(crate) batch: usize,
    pub(crate) total_channels: usize,
    pub(crate) output_len: usize,
}

pub(crate) fn validate_route_concat_layout(
    input_lens: &[usize],
    input_channel_offsets: &[usize],
    input_channels: &[usize],
    input_total_channels: &[usize],
    output_shape: &[usize],
    channel_stride: usize,
) -> Result<RouteConcatLayout, CutileBackendError> {
    if input_lens.is_empty() {
        return Err(CutileBackendError::Shape(
            "route_concat_f32 requires at least one input".to_owned(),
        ));
    }
    if input_lens.len() != input_channel_offsets.len()
        || input_lens.len() != input_channels.len()
        || input_lens.len() != input_total_channels.len()
    {
        return Err(CutileBackendError::Shape(format!(
            "route_concat_f32 metadata length mismatch: inputs={}, channel_offsets={}, channel_counts={}, total_channels={}",
            input_lens.len(),
            input_channel_offsets.len(),
            input_channels.len(),
            input_total_channels.len(),
        )));
    }
    if channel_stride == 0 {
        return Err(CutileBackendError::Shape(
            "route_concat_f32 requires channel_stride > 0".to_owned(),
        ));
    }

    let output_len = shape_num_elements(output_shape)?;
    let mut batch = None;
    let mut total_channels = 0usize;
    for (((input_len, channel_offset), channels), total_input_channels) in input_lens
        .iter()
        .copied()
        .zip(input_channel_offsets.iter().copied())
        .zip(input_channels.iter().copied())
        .zip(input_total_channels.iter().copied())
    {
        if channels == 0 {
            return Err(CutileBackendError::Shape(
                "route_concat_f32 requires input_channels > 0".to_owned(),
            ));
        }
        if total_input_channels == 0 {
            return Err(CutileBackendError::Shape(
                "route_concat_f32 requires input_total_channels > 0".to_owned(),
            ));
        }
        if channel_offset + channels > total_input_channels {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 channel slice [{}..{}) exceeds input total channels {}",
                channel_offset,
                channel_offset + channels,
                total_input_channels
            )));
        }
        if !input_len.is_multiple_of(channel_stride) {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 input length {input_len} is not divisible by channel_stride {channel_stride}"
            )));
        }
        let rows = input_len / channel_stride;
        if !rows.is_multiple_of(total_input_channels) {
            return Err(CutileBackendError::Shape(format!(
                "route_concat_f32 rows {rows} are not divisible by total input channels {total_input_channels}"
            )));
        }
        let input_batch = rows / total_input_channels;
        if let Some(expected) = batch {
            if expected != input_batch {
                return Err(CutileBackendError::Shape(format!(
                    "route_concat_f32 batch mismatch: expected {expected}, got {input_batch}"
                )));
            }
        } else {
            batch = Some(input_batch);
        }
        total_channels += channels;
    }

    let expected_output_len = batch.unwrap_or(0) * total_channels * channel_stride;
    if output_len != expected_output_len {
        return Err(CutileBackendError::Shape(format!(
            "route_concat_f32 output shape {:?} expects {output_len} elements, but inputs require {expected_output_len}",
            output_shape
        )));
    }

    Ok(RouteConcatLayout {
        batch: batch.unwrap_or(0),
        total_channels,
        output_len,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Conv2dLayout {
    pub(crate) batch: usize,
    pub(crate) channels_per_group: usize,
    pub(crate) filters_per_group: usize,
    pub(crate) input_len: usize,
    pub(crate) weight_len: usize,
    pub(crate) output_height: usize,
    pub(crate) output_width: usize,
    pub(crate) output_len: usize,
}

pub(crate) fn validate_conv2d_layout(
    input_len: usize,
    weight_len: usize,
    output_shape: &[usize],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    groups: usize,
) -> Result<Conv2dLayout, CutileBackendError> {
    if input_channels == 0 || input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires non-zero input dimensions".to_owned(),
        ));
    }
    if filters == 0 || kernel_size == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires non-zero filters and kernel size".to_owned(),
        ));
    }
    if groups == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 requires groups > 0".to_owned(),
        ));
    }
    if !input_channels.is_multiple_of(groups) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input channels {input_channels} are not divisible by groups {groups}"
        )));
    }
    if !filters.is_multiple_of(groups) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 filters {filters} are not divisible by groups {groups}"
        )));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("conv2d_f32 input plane overflow".to_owned()))?;
    let input_batch_stride = input_channels.checked_mul(input_plane).ok_or_else(|| {
        CutileBackendError::Shape("conv2d_f32 input batch stride overflow".to_owned())
    })?;
    if !input_len.is_multiple_of(input_batch_stride) {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 input length {input_len} is not divisible by input batch stride {input_batch_stride}"
        )));
    }
    let batch = input_len / input_batch_stride;

    let channels_per_group = input_channels / groups;
    let filters_per_group = filters / groups;
    let weights_per_filter = channels_per_group
        .checked_mul(kernel_size)
        .and_then(|value| value.checked_mul(kernel_size))
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d_f32 weights-per-filter overflow".to_owned())
        })?;
    let expected_weight_len = filters
        .checked_mul(weights_per_filter)
        .ok_or_else(|| CutileBackendError::Shape("conv2d_f32 weight length overflow".to_owned()))?;
    if weight_len != expected_weight_len {
        return Err(CutileBackendError::Shape(format!(
            "conv2d_f32 weight length mismatch: expected {expected_weight_len}, got {weight_len}"
        )));
    }

    if output_shape.len() < 4 {
        return Err(CutileBackendError::Shape(
            "conv2d_f32 output shape must be [batch, channels, height, width]".to_owned(),
        ));
    }
    let output_len = shape_num_elements(output_shape)?;
    Ok(Conv2dLayout {
        batch,
        channels_per_group,
        filters_per_group,
        input_len,
        weight_len,
        output_height: output_shape[output_shape.len() - 2],
        output_width: output_shape[output_shape.len() - 1],
        output_len,
    })
}

pub(crate) fn conv_output_dim(
    input: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
) -> Result<usize, CutileBackendError> {
    if stride == 0 || kernel_size == 0 || dilation == 0 {
        return Err(CutileBackendError::Shape(
            "conv2d backward requires non-zero kernel, stride, and dilation".to_owned(),
        ));
    }
    let kernel_extent = dilation
        .checked_mul(kernel_size.saturating_sub(1))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward kernel extent overflow".to_owned())
        })?;
    let padded = input
        .checked_add(padding.checked_mul(2).ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward padding overflow".to_owned())
        })?)
        .ok_or_else(|| {
            CutileBackendError::Shape("conv2d backward padded input overflow".to_owned())
        })?;
    if padded < kernel_extent {
        return Err(CutileBackendError::Shape(format!(
            "conv2d backward kernel extent {kernel_extent} exceeds padded input {padded}"
        )));
    }
    Ok((padded - kernel_extent) / stride + 1)
}

pub(crate) fn validate_conv2d_backward_input_layout(
    output_grad_len: usize,
    weight_len: usize,
    input_shape: &[usize],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    stride_x: usize,
    stride_y: usize,
    dilation: usize,
    padding: usize,
    groups: usize,
) -> Result<Conv2dLayout, CutileBackendError> {
    let input_len = shape_num_elements(input_shape)?;
    let output_height = conv_output_dim(input_height, kernel_size, stride_y, dilation, padding)?;
    let output_width = conv_output_dim(input_width, kernel_size, stride_x, dilation, padding)?;
    let batch = input_len / input_channels / input_height / input_width;
    validate_conv2d_layout(
        input_len,
        weight_len,
        &[batch, filters, output_height, output_width],
        input_channels,
        input_height,
        input_width,
        filters,
        kernel_size,
        groups,
    )
    .and_then(|layout| {
        if output_grad_len != layout.output_len {
            Err(CutileBackendError::Shape(format!(
                "conv2d_backward_input_f32 output gradient length mismatch: expected {}, got {output_grad_len}",
                layout.output_len
            )))
        } else {
            Ok(layout)
        }
    })
}

pub(crate) fn validate_conv2d_backward_weights_layout(
    input_len: usize,
    output_grad_len: usize,
    weight_shape: &[usize],
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    stride_x: usize,
    stride_y: usize,
    dilation: usize,
    padding: usize,
    groups: usize,
) -> Result<Conv2dLayout, CutileBackendError> {
    let weight_len = shape_num_elements(weight_shape)?;
    let output_height = conv_output_dim(input_height, kernel_size, stride_y, dilation, padding)?;
    let output_width = conv_output_dim(input_width, kernel_size, stride_x, dilation, padding)?;
    let batch = input_len / input_channels / input_height / input_width;
    validate_conv2d_layout(
        input_len,
        weight_len,
        &[batch, filters, output_height, output_width],
        input_channels,
        input_height,
        input_width,
        filters,
        kernel_size,
        groups,
    )
    .and_then(|layout| {
        if output_grad_len != layout.output_len {
            Err(CutileBackendError::Shape(format!(
                "conv2d_backward_weights_f32 output gradient length mismatch: expected {}, got {output_grad_len}",
                layout.output_len
            )))
        } else {
            Ok(layout)
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UpsampleLayout {
    pub(crate) output_height: usize,
    pub(crate) output_width: usize,
    pub(crate) output_len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MaxPoolLayout {
    pub(crate) output_height: usize,
    pub(crate) output_width: usize,
    pub(crate) output_len: usize,
}

pub(crate) fn validate_upsample_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    stride: usize,
    reverse: bool,
) -> Result<UpsampleLayout, CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires channels > 0".to_owned(),
        ));
    }
    if input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires non-zero input spatial dimensions".to_owned(),
        ));
    }
    if stride == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 requires stride > 0".to_owned(),
        ));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 input plane overflow".to_owned()))?;
    if !input_len.is_multiple_of(input_plane) {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 input length {input_len} is not divisible by input plane {input_plane}"
        )));
    }
    let batch_channels = input_len / input_plane;
    if !batch_channels.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 batch*channels {batch_channels} is not divisible by channels {channels}"
        )));
    }

    let output_len = shape_num_elements(output_shape)?;
    let output_plane = output_len
        .checked_div(batch_channels)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 invalid output plane".to_owned()))?;
    if output_plane == 0 {
        return Err(CutileBackendError::Shape(
            "upsample_f32 output plane cannot be zero".to_owned(),
        ));
    }

    let (output_height, output_width) = if output_shape.len() >= 2 {
        (
            output_shape[output_shape.len() - 2],
            output_shape[output_shape.len() - 1],
        )
    } else {
        return Err(CutileBackendError::Shape(
            "upsample_f32 output shape must include at least two spatial dimensions".to_owned(),
        ));
    };
    if output_height
        .checked_mul(output_width)
        .ok_or_else(|| CutileBackendError::Shape("upsample_f32 output plane overflow".to_owned()))?
        != output_plane
    {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 output shape {:?} does not match batch/channels-derived plane {output_plane}",
            output_shape
        )));
    }

    let expected_output_plane = if reverse {
        input_height.div_ceil(stride) * input_width.div_ceil(stride)
    } else {
        input_height
            .checked_mul(stride)
            .and_then(|height| input_width.checked_mul(stride).map(|width| height * width))
            .ok_or_else(|| {
                CutileBackendError::Shape("upsample_f32 expanded plane overflow".to_owned())
            })?
    };
    if output_plane != expected_output_plane {
        return Err(CutileBackendError::Shape(format!(
            "upsample_f32 output plane mismatch: expected {expected_output_plane}, got {output_plane}"
        )));
    }

    Ok(UpsampleLayout {
        output_height,
        output_width,
        output_len,
    })
}

pub(crate) fn validate_maxpool_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
) -> Result<MaxPoolLayout, CutileBackendError> {
    if channels == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires channels > 0".to_owned(),
        ));
    }
    if input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires non-zero input spatial dimensions".to_owned(),
        ));
    }
    if size == 0 || stride_x == 0 || stride_y == 0 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 requires non-zero kernel size and strides".to_owned(),
        ));
    }
    if output_shape.len() != 4 {
        return Err(CutileBackendError::Shape(
            "maxpool_f32 output shape must be [batch, channels, height, width]".to_owned(),
        ));
    }

    let input_plane = input_height
        .checked_mul(input_width)
        .ok_or_else(|| CutileBackendError::Shape("maxpool_f32 input plane overflow".to_owned()))?;
    if !input_len.is_multiple_of(input_plane) {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 input length {input_len} is not divisible by input plane {input_plane}"
        )));
    }
    let batch_channels = input_len / input_plane;
    if !batch_channels.is_multiple_of(channels) {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 batch*channels {batch_channels} is not divisible by channels {channels}"
        )));
    }
    let batch = batch_channels / channels;
    if output_shape[0] != batch {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output batch mismatch: expected {batch}, got {}",
            output_shape[0]
        )));
    }
    if output_shape[1] != channels {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output channels mismatch: expected {channels}, got {}",
            output_shape[1]
        )));
    }

    let output_len = shape_num_elements(output_shape)?;
    let output_height = output_shape[2];
    let output_width = output_shape[3];
    let expected_output_len = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or_else(|| {
            CutileBackendError::Shape("maxpool_f32 output length overflow".to_owned())
        })?;
    if output_len != expected_output_len {
        return Err(CutileBackendError::Shape(format!(
            "maxpool_f32 output shape {:?} expects {output_len} elements, got {expected_output_len}",
            output_shape
        )));
    }

    Ok(MaxPoolLayout {
        output_height,
        output_width,
        output_len,
    })
}

pub(crate) fn shape_num_elements(shape: &[usize]) -> Result<usize, CutileBackendError> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| CutileBackendError::Shape("shape overflow".to_owned()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CropLayout {
    pub(crate) batch: usize,
    pub(crate) output_height: usize,
    pub(crate) output_width: usize,
    pub(crate) output_len: usize,
}

pub(crate) fn validate_crop_layout(
    input_len: usize,
    output_shape: &[usize],
    channels: usize,
    input_height: usize,
    input_width: usize,
    source_x_len: usize,
    source_y_len: usize,
    scale_len: usize,
    offset_len: usize,
) -> Result<CropLayout, CutileBackendError> {
    if output_shape.len() != 4 || channels == 0 || input_height == 0 || input_width == 0 {
        return Err(CutileBackendError::Shape(
            "crop_bilinear_f32 requires [batch, channels, height, width] and non-zero dimensions"
                .to_owned(),
        ));
    }
    if output_shape[1] != channels {
        return Err(CutileBackendError::Shape(format!(
            "crop output channels mismatch: expected {channels}, got {}",
            output_shape[1]
        )));
    }
    let batch = output_shape[0];
    let output_height = output_shape[2];
    let output_width = output_shape[3];
    let output_len = shape_num_elements(output_shape)?;
    let expected_input = batch
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(input_height))
        .and_then(|n| n.checked_mul(input_width))
        .ok_or_else(|| CutileBackendError::Shape("crop input shape overflow".to_owned()))?;
    if input_len != expected_input {
        return Err(CutileBackendError::Shape(format!(
            "crop input length mismatch: expected {expected_input}, got {input_len}"
        )));
    }
    let expected_coords = batch * output_height * output_width;
    if source_x_len != expected_coords || source_y_len != expected_coords {
        return Err(CutileBackendError::Shape(format!(
            "crop coordinate length mismatch: expected {expected_coords}, got source_x={source_x_len}, source_y={source_y_len}"
        )));
    }
    if scale_len != batch || offset_len != batch {
        return Err(CutileBackendError::Shape(format!(
            "crop adjustment length mismatch: expected {batch}, got scale={scale_len}, offset={offset_len}"
        )));
    }
    Ok(CropLayout {
        batch,
        output_height,
        output_width,
        output_len,
    })
}
