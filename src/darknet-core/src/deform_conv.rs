//! CPU reference math for standard deformable convolution (DCNv1/DCNv2).
//!
//! The predictor convolutions use all input channels and share one offset/mask
//! field across convolution groups, matching `Jev_laya:src-lib/deform_conv_layer.cpp`.
//! Backward differentiates the Rust forward, including the conditional
//! per-image offset standardization that the C++ CPU backward omits.

use crate::error::{Error, Result};
use crate::model::{DeformConvSpec, TensorShape};
use crate::runtime::Tensor;

const VALUE_LIMIT: f32 = 10_000.0;
const SIGMOID_LIMIT: f32 = 80.0;
const MIN_OFFSET_STDDEV: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeformConvForwardCache {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    /// Sanitized/clamped input used by both predictor and main convolutions.
    input: Vec<f32>,
    input_slope: Vec<f32>,
    input_nondiff_boundary: Vec<bool>,
    /// Predictor receptive fields `[batch, C*K, output_spatial]`.
    columns: Vec<f32>,
    /// Raw predictor values after the first finite clamp, before optional
    /// per-image standardization.
    offset_stage1: Vec<f32>,
    offset_stage1_slope: Vec<f32>,
    offset_stage1_nondiff_boundary: Vec<bool>,
    offset_mean: Vec<f64>,
    offset_stddev: Vec<f64>,
    offset_normalized: Vec<bool>,
    offset_normalization_boundary: Vec<bool>,
    /// Values after optional normalization but before its second finite clamp.
    offset_pre_stage2_clamp: Vec<f32>,
    offset_stage2_slope: Vec<f32>,
    offset_stage2_nondiff_boundary: Vec<bool>,
    /// Centered-LHTAN output and its local derivative with respect to stage 2.
    offsets: Vec<f32>,
    offset_lhtan_slope: Vec<f32>,
    offset_lhtan_boundary: Vec<bool>,
    offset_map_slope: Vec<f32>,
    /// Sigmoid modulation probabilities and predictor clamp state.
    masks: Option<Vec<f32>>,
    mask_predictor_slope: Option<Vec<f32>>,
    mask_predictor_nondiff_boundary: Option<Vec<bool>>,
    mask_sigmoid_slope: Option<Vec<f32>>,
    mask_sigmoid_boundary: Option<Vec<bool>>,
    /// Sanitized convolution result before the bias/BN and activation epilogue.
    pub raw_output: Vec<f32>,
    pub raw_output_slope: Vec<f32>,
    raw_output_nondiff_boundary: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeformConvGradients {
    pub input: Vec<f32>,
    pub weights: Vec<f32>,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
    pub mask_weights: Option<Vec<f32>>,
    pub mask_biases: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    batches: usize,
    channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    channels_per_group: usize,
    filters_per_group: usize,
    kernel_area: usize,
    offset_filters: usize,
    output_height: usize,
    output_width: usize,
    output_spatial: usize,
    column_channels: usize,
    input_count: usize,
    output_count: usize,
    main_weight_count: usize,
    predictor_weight_count: usize,
    mask_weight_count: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[deform_conv]: {}", message.into()))
}

fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{what} element count overflows")))
    })
}

fn output_dim(
    input: usize,
    padding: usize,
    size: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    if input == 0 || size == 0 || stride == 0 || dilation == 0 {
        return None;
    }
    let padded = input.checked_add(padding.checked_mul(2)?)?;
    let kernel_extent = dilation.checked_mul(size.checked_sub(1)?)?;
    padded
        .checked_sub(kernel_extent)?
        .checked_sub(1)?
        .checked_div(stride)?
        .checked_add(1)
}

fn layout(
    input_shape: TensorShape,
    spec: &DeformConvSpec,
    output_shape: TensorShape,
) -> Result<Layout> {
    if input_shape.width == 0
        || input_shape.height == 0
        || input_shape.channels == 0
        || input_shape.batch == 0
    {
        return Err(invalid("input dimensions must be non-zero"));
    }
    if spec.filters == 0
        || spec.groups == 0
        || spec.size == 0
        || spec.stride_x == 0
        || spec.stride_y == 0
        || spec.dilation == 0
    {
        return Err(invalid(
            "filters, groups, size, strides, and dilation must be positive",
        ));
    }
    if input_shape.channels % spec.groups != 0 || spec.filters % spec.groups != 0 {
        return Err(invalid(
            "groups must divide both input channels and filters",
        ));
    }
    let output_width = output_dim(
        input_shape.width,
        spec.padding,
        spec.size,
        spec.stride_x,
        spec.dilation,
    )
    .ok_or_else(|| invalid("invalid or empty output width"))?;
    let output_height = output_dim(
        input_shape.height,
        spec.padding,
        spec.size,
        spec.stride_y,
        spec.dilation,
    )
    .ok_or_else(|| invalid("invalid or empty output height"))?;
    let expected = TensorShape::new(output_width, output_height, spec.filters, input_shape.batch);
    if output_shape != expected {
        return Err(invalid(format!(
            "output shape mismatch: expected {expected}, got {output_shape}"
        )));
    }
    let kernel_area = spec
        .size
        .checked_mul(spec.size)
        .ok_or_else(|| invalid("kernel area overflows"))?;
    let offset_filters = kernel_area
        .checked_mul(2)
        .ok_or_else(|| invalid("offset filter count overflows"))?;
    let column_channels = input_shape
        .channels
        .checked_mul(kernel_area)
        .ok_or_else(|| invalid("predictor column count overflows"))?;
    let predictor_weight_count =
        checked_product(&[offset_filters, column_channels], "offset weights")?;
    let mask_weight_count = if spec.use_mask {
        checked_product(&[kernel_area, column_channels], "mask weights")?
    } else {
        0
    };
    let main_weight_count = checked_product(
        &[
            spec.filters,
            input_shape.channels / spec.groups,
            kernel_area,
        ],
        "main weights",
    )?;
    let output_spatial = output_width
        .checked_mul(output_height)
        .ok_or_else(|| invalid("output spatial size overflows"))?;
    let input_count = checked_product(
        &[
            input_shape.batch,
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        ],
        "input",
    )?;
    let output_count =
        checked_product(&[input_shape.batch, spec.filters, output_spatial], "output")?;
    checked_product(
        &[input_shape.batch, column_channels, output_spatial],
        "predictor columns",
    )?;
    checked_product(
        &[input_shape.batch, offset_filters, output_spatial],
        "offset field",
    )?;
    if spec.use_mask {
        checked_product(
            &[input_shape.batch, kernel_area, output_spatial],
            "mask field",
        )?;
    }
    Ok(Layout {
        batches: input_shape.batch,
        channels: input_shape.channels,
        height: input_shape.height,
        width: input_shape.width,
        filters: spec.filters,
        channels_per_group: input_shape.channels / spec.groups,
        filters_per_group: spec.filters / spec.groups,
        kernel_area,
        offset_filters,
        output_height,
        output_width,
        output_spatial,
        column_channels,
        input_count,
        output_count,
        main_weight_count,
        predictor_weight_count,
        mask_weight_count,
    })
}

fn sanitize(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-VALUE_LIMIT, VALUE_LIMIT)
    } else {
        0.0
    }
}

fn open_clamp_slope(value: f32) -> f32 {
    if value.is_finite() && value > -VALUE_LIMIT && value < VALUE_LIMIT {
        1.0
    } else {
        0.0
    }
}

fn plane_index(
    batch: usize,
    plane: usize,
    position: usize,
    planes: usize,
    spatial: usize,
) -> usize {
    (batch * planes + plane) * spatial + position
}

fn column_index(
    batch: usize,
    column: usize,
    position: usize,
    columns: usize,
    spatial: usize,
) -> usize {
    (batch * columns + column) * spatial + position
}

fn input_index(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * channels + channel) * height + y) * width + x
}

fn predictor_columns(input: &[f32], spec: &DeformConvSpec, l: Layout) -> Vec<f32> {
    let mut columns = vec![0.0; l.batches * l.column_channels * l.output_spatial];
    for b in 0..l.batches {
        for channel in 0..l.channels {
            for ky in 0..spec.size {
                for kx in 0..spec.size {
                    let column = (channel * spec.size + ky) * spec.size + kx;
                    for oy in 0..l.output_height {
                        for ox in 0..l.output_width {
                            let y = oy as isize * spec.stride_y as isize
                                - (spec.padding * spec.dilation) as isize
                                + (ky * spec.dilation) as isize;
                            let x = ox as isize * spec.stride_x as isize
                                - (spec.padding * spec.dilation) as isize
                                + (kx * spec.dilation) as isize;
                            if y >= 0 && y < l.height as isize && x >= 0 && x < l.width as isize {
                                let pos = oy * l.output_width + ox;
                                columns[column_index(
                                    b,
                                    column,
                                    pos,
                                    l.column_channels,
                                    l.output_spatial,
                                )] = input[input_index(
                                    b, channel, y as usize, x as usize, l.channels, l.height,
                                    l.width,
                                )];
                            }
                        }
                    }
                }
            }
        }
    }
    columns
}

fn bilinear_sample(
    input: &[f32],
    channel_base: usize,
    height: usize,
    width: usize,
    y: f32,
    x: f32,
) -> (f32, f32, f32, [Option<(usize, f32)>; 4]) {
    if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
        return (0.0, 0.0, 0.0, [None; 4]);
    }
    let y0f = y.floor();
    let x0f = x.floor();
    let y0 = y0f as isize;
    let x0 = x0f as isize;
    let y1 = y0 + 1;
    let x1 = x0 + 1;
    let ly = y - y0f;
    let lx = x - x0f;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;
    let coordinates = [(y0, x0), (y0, x1), (y1, x0), (y1, x1)];
    let weights = [hy * hx, hy * lx, ly * hx, ly * lx];
    let mut values = [0.0; 4];
    let mut neighbors = [None; 4];
    for i in 0..4 {
        let (iy, ix) = coordinates[i];
        if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
            let index = channel_base + iy as usize * width + ix as usize;
            values[i] = input[index];
            neighbors[i] = Some((index, weights[i]));
        }
    }
    let value = weights.iter().zip(values).map(|(w, v)| w * v).sum();
    // At integer coordinates use the floor-selected cell (a deterministic
    // right-sided derivative convention); tests of the mathematical gradient
    // use non-integer sample coordinates.
    let dy = -hx * values[0] - lx * values[1] + hx * values[2] + lx * values[3];
    let dx = -hy * values[0] + hy * values[1] - ly * values[2] + ly * values[3];
    (value, dy, dx, neighbors)
}

fn centered_lhtan(value: f32, max_offset: f32) -> (f32, f32, bool, f32) {
    let scaled = value / (2.0 * max_offset) + 0.5;
    let (mapped, slope, boundary) = if scaled < 0.0 {
        (0.001 * scaled, 0.001, false)
    } else if scaled > 1.0 {
        (1.0 + 0.001 * (scaled - 1.0), 0.001, false)
    } else if scaled == 0.0 || scaled == 1.0 {
        (scaled, 1.0, true)
    } else {
        (scaled, 1.0, false)
    };
    let unclamped = (2.0 * mapped - 1.0) * max_offset;
    let clamp_slope = open_clamp_slope(unclamped);
    (sanitize(unclamped), slope, boundary, clamp_slope)
}

fn stable_sigmoid(value: f32) -> (f32, f32, bool) {
    if !value.is_finite() {
        return (0.5, 0.0, false);
    }
    if value == SIGMOID_LIMIT || value == -SIGMOID_LIMIT {
        let clipped = value;
        let probability = if clipped >= 0.0 {
            1.0 / (1.0 + (-clipped).exp())
        } else {
            let z = clipped.exp();
            z / (1.0 + z)
        };
        return (probability, 0.0, true);
    }
    let clipped = value.clamp(-SIGMOID_LIMIT, SIGMOID_LIMIT);
    let probability = if clipped >= 0.0 {
        1.0 / (1.0 + (-clipped).exp())
    } else {
        let z = clipped.exp();
        z / (1.0 + z)
    };
    (
        probability,
        if value > -SIGMOID_LIMIT && value < SIGMOID_LIMIT {
            1.0
        } else {
            0.0
        },
        false,
    )
}

pub(crate) fn deform_conv_forward(
    input: &Tensor,
    spec: &DeformConvSpec,
    output_shape: TensorShape,
    main_weights: &[f32],
    offset_weights: &[f32],
    offset_biases: &[f32],
    mask_weights: Option<&[f32]>,
    mask_biases: Option<&[f32]>,
) -> Result<(Tensor, DeformConvForwardCache)> {
    let l = layout(input.shape, spec, output_shape)?;
    if input.data.len() != l.input_count
        || main_weights.len() != l.main_weight_count
        || offset_weights.len() != l.predictor_weight_count
        || offset_biases.len() != l.offset_filters
    {
        return Err(invalid("input or parameter buffer length mismatch"));
    }
    match (spec.use_mask, mask_weights, mask_biases) {
        (true, Some(weights), Some(biases))
            if weights.len() == l.mask_weight_count && biases.len() == l.kernel_area => {}
        (false, None, None) => {}
        _ => return Err(invalid("mask parameter layout does not match use_mask")),
    }

    let input_slope: Vec<f32> = input.data.iter().copied().map(open_clamp_slope).collect();
    let input_nondiff_boundary: Vec<bool> = input
        .data
        .iter()
        .map(|value| !value.is_finite() || value.abs() == VALUE_LIMIT)
        .collect();
    let input_values: Vec<f32> = input.data.iter().copied().map(sanitize).collect();
    let columns = predictor_columns(&input_values, spec, l);
    let offset_count = l.batches * l.offset_filters * l.output_spatial;
    let mut offset_stage1 = vec![0.0; offset_count];
    let mut offset_stage1_slope = vec![0.0; offset_count];
    let mut offset_stage1_nondiff_boundary = vec![false; offset_count];
    for b in 0..l.batches {
        for out_filter in 0..l.offset_filters {
            for pos in 0..l.output_spatial {
                let mut value = offset_biases[out_filter];
                for column in 0..l.column_channels {
                    value += offset_weights[out_filter * l.column_channels + column]
                        * columns
                            [column_index(b, column, pos, l.column_channels, l.output_spatial)];
                }
                let index = plane_index(b, out_filter, pos, l.offset_filters, l.output_spatial);
                offset_stage1[index] = sanitize(value);
                offset_stage1_slope[index] = open_clamp_slope(value);
                offset_stage1_nondiff_boundary[index] =
                    !value.is_finite() || value.abs() == VALUE_LIMIT;
            }
        }
    }
    let mut offset_mean = vec![0.0f64; l.batches];
    let mut offset_stddev = vec![0.0f64; l.batches];
    let mut offset_normalized = vec![false; l.batches];
    let mut offset_normalization_boundary = vec![false; l.batches];
    let mut offset_pre_stage2_clamp = vec![0.0; offset_count];
    let max_offset = (spec.size as f32) * (spec.dilation as f32) * 2.0;
    if !max_offset.is_finite() || max_offset <= 0.0 {
        return Err(invalid("max offset is not finite and positive"));
    }
    let per_batch_offsets = l.offset_filters * l.output_spatial;
    for b in 0..l.batches {
        let start = b * per_batch_offsets;
        let end = start + per_batch_offsets;
        let sum = offset_stage1[start..end]
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>();
        let mean = sum / per_batch_offsets as f64;
        let sum_sq = offset_stage1[start..end]
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>();
        let variance = (sum_sq / per_batch_offsets as f64 - mean * mean).max(0.0);
        let stddev = variance.sqrt();
        offset_mean[b] = mean;
        offset_stddev[b] = stddev;
        let boundary = stddev == MIN_OFFSET_STDDEV
            || stddev == max_offset as f64
            || mean.abs() == max_offset as f64;
        offset_normalization_boundary[b] = boundary;
        let normalize = stddev > MIN_OFFSET_STDDEV
            && (stddev > max_offset as f64 || mean.abs() > max_offset as f64);
        offset_normalized[b] = normalize;
        let mean_f = mean as f32;
        let inv_std = if normalize { 1.0 / stddev as f32 } else { 1.0 };
        for i in 0..per_batch_offsets {
            let raw = offset_stage1[start + i];
            offset_pre_stage2_clamp[start + i] = if normalize {
                (raw - mean_f) * inv_std
            } else {
                raw
            };
        }
    }
    let offset_stage2_slope: Vec<f32> = offset_pre_stage2_clamp
        .iter()
        .copied()
        .map(open_clamp_slope)
        .collect();
    let offset_stage2_nondiff_boundary: Vec<bool> = offset_pre_stage2_clamp
        .iter()
        .map(|value| !value.is_finite() || value.abs() == VALUE_LIMIT)
        .collect();
    let mut offsets = vec![0.0; offset_count];
    let mut offset_lhtan_slope = vec![0.0; offset_count];
    let mut offset_lhtan_boundary = vec![false; offset_count];
    let mut offset_map_slope = vec![0.0; offset_count];
    for i in 0..offset_count {
        let clamped = sanitize(offset_pre_stage2_clamp[i]);
        let (mapped, lhtan_slope, boundary, map_slope) = centered_lhtan(clamped, max_offset);
        offsets[i] = mapped;
        offset_lhtan_slope[i] = lhtan_slope;
        offset_lhtan_boundary[i] = boundary;
        offset_map_slope[i] = map_slope;
    }

    let (
        masks,
        mask_predictor_slope,
        mask_predictor_nondiff_boundary,
        mask_sigmoid_slope,
        mask_sigmoid_boundary,
    ) = if spec.use_mask {
        let weights = mask_weights.expect("validated mask weights");
        let biases = mask_biases.expect("validated mask biases");
        let count = l.batches * l.kernel_area * l.output_spatial;
        let mut logits = vec![0.0; count];
        for b in 0..l.batches {
            for out_filter in 0..l.kernel_area {
                for pos in 0..l.output_spatial {
                    let mut value = biases[out_filter];
                    for column in 0..l.column_channels {
                        value += weights[out_filter * l.column_channels + column]
                            * columns
                                [column_index(b, column, pos, l.column_channels, l.output_spatial)];
                    }
                    logits[plane_index(b, out_filter, pos, l.kernel_area, l.output_spatial)] =
                        value;
                }
            }
        }
        let mut probabilities = vec![0.0; count];
        let mut predictor_slope = vec![0.0; count];
        let mut predictor_nondiff_boundary = vec![false; count];
        let mut sigmoid_slope = vec![0.0; count];
        let mut sigmoid_boundary = vec![false; count];
        for i in 0..count {
            let raw = logits[i];
            let sanitized = sanitize(raw);
            predictor_slope[i] = open_clamp_slope(raw);
            predictor_nondiff_boundary[i] = !raw.is_finite() || raw.abs() == VALUE_LIMIT;
            let (probability, sigmoid_derivative, boundary) = stable_sigmoid(sanitized);
            probabilities[i] = probability;
            sigmoid_slope[i] = sigmoid_derivative;
            sigmoid_boundary[i] = boundary;
            logits[i] = sanitized;
        }
        (
            Some(probabilities),
            Some(predictor_slope),
            Some(predictor_nondiff_boundary),
            Some(sigmoid_slope),
            Some(sigmoid_boundary),
        )
    } else {
        (None, None, None, None, None)
    };

    let mut raw_output = vec![0.0; l.output_count];
    let mut raw_output_slope = vec![0.0; l.output_count];
    let mut raw_output_nondiff_boundary = vec![false; l.output_count];
    for b in 0..l.batches {
        for filter in 0..l.filters {
            let group = filter / l.filters_per_group;
            for oy in 0..l.output_height {
                for ox in 0..l.output_width {
                    let pos = oy * l.output_width + ox;
                    let mut sum = 0.0f32;
                    for local_channel in 0..l.channels_per_group {
                        let channel = group * l.channels_per_group + local_channel;
                        let channel_base =
                            input_index(b, channel, 0, 0, l.channels, l.height, l.width);
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let k = ky * spec.size + kx;
                                let y_plane = 2 * k;
                                let x_plane = y_plane + 1;
                                let offset_y = offsets[plane_index(
                                    b,
                                    y_plane,
                                    pos,
                                    l.offset_filters,
                                    l.output_spatial,
                                )];
                                let offset_x = offsets[plane_index(
                                    b,
                                    x_plane,
                                    pos,
                                    l.offset_filters,
                                    l.output_spatial,
                                )];
                                let base_y = oy as f32 * spec.stride_y as f32
                                    - (spec.padding * spec.dilation) as f32
                                    + (ky * spec.dilation) as f32;
                                let base_x = ox as f32 * spec.stride_x as f32
                                    - (spec.padding * spec.dilation) as f32
                                    + (kx * spec.dilation) as f32;
                                let (sample, _, _, _) = bilinear_sample(
                                    &input_values,
                                    channel_base,
                                    l.height,
                                    l.width,
                                    base_y + offset_y,
                                    base_x + offset_x,
                                );
                                let mask = masks.as_ref().map_or(1.0, |values| {
                                    values[plane_index(b, k, pos, l.kernel_area, l.output_spatial)]
                                });
                                let weight_index = (filter * l.channels_per_group + local_channel)
                                    * l.kernel_area
                                    + k;
                                sum += main_weights[weight_index] * sample * mask;
                            }
                        }
                    }
                    let output_index = (b * l.filters + filter) * l.output_spatial + pos;
                    raw_output[output_index] = sanitize(sum);
                    raw_output_slope[output_index] = open_clamp_slope(sum);
                    raw_output_nondiff_boundary[output_index] =
                        !sum.is_finite() || sum.abs() == VALUE_LIMIT;
                }
            }
        }
    }
    Ok((
        Tensor::from_data(output_shape, raw_output.clone())?,
        DeformConvForwardCache {
            input_shape: input.shape,
            output_shape,
            input: input_values,
            input_slope,
            input_nondiff_boundary,
            columns,
            offset_stage1,
            offset_stage1_slope,
            offset_stage1_nondiff_boundary,
            offset_mean,
            offset_stddev,
            offset_normalized,
            offset_normalization_boundary,
            offset_pre_stage2_clamp,
            offset_stage2_slope,
            offset_stage2_nondiff_boundary,
            offsets,
            offset_lhtan_slope,
            offset_lhtan_boundary,
            offset_map_slope,
            masks,
            mask_predictor_slope,
            mask_predictor_nondiff_boundary,
            mask_sigmoid_slope,
            mask_sigmoid_boundary,
            raw_output,
            raw_output_slope,
            raw_output_nondiff_boundary,
        },
    ))
}

fn add_predictor_input_gradient(
    input_gradient: &mut [f32],
    predictor_gradient: &[f32],
    weights: &[f32],
    output_filters: usize,
    l: Layout,
    spec: &DeformConvSpec,
) {
    let mut column_gradient = vec![0.0; l.batches * l.column_channels * l.output_spatial];
    for b in 0..l.batches {
        for out_filter in 0..output_filters {
            for pos in 0..l.output_spatial {
                let gradient = predictor_gradient
                    [plane_index(b, out_filter, pos, output_filters, l.output_spatial)];
                for column in 0..l.column_channels {
                    column_gradient
                        [column_index(b, column, pos, l.column_channels, l.output_spatial)] +=
                        weights[out_filter * l.column_channels + column] * gradient;
                }
            }
        }
    }
    for b in 0..l.batches {
        for channel in 0..l.channels {
            for ky in 0..spec.size {
                for kx in 0..spec.size {
                    let column = (channel * spec.size + ky) * spec.size + kx;
                    for oy in 0..l.output_height {
                        for ox in 0..l.output_width {
                            let y = oy as isize * spec.stride_y as isize
                                - (spec.padding * spec.dilation) as isize
                                + (ky * spec.dilation) as isize;
                            let x = ox as isize * spec.stride_x as isize
                                - (spec.padding * spec.dilation) as isize
                                + (kx * spec.dilation) as isize;
                            if y >= 0 && y < l.height as isize && x >= 0 && x < l.width as isize {
                                let pos = oy * l.output_width + ox;
                                input_gradient[input_index(
                                    b, channel, y as usize, x as usize, l.channels, l.height,
                                    l.width,
                                )] += column_gradient[column_index(
                                    b,
                                    column,
                                    pos,
                                    l.column_channels,
                                    l.output_spatial,
                                )];
                            }
                        }
                    }
                }
            }
        }
    }
}

pub(crate) fn deform_conv_backward(
    input_shape: TensorShape,
    spec: &DeformConvSpec,
    output_shape: TensorShape,
    cache: &DeformConvForwardCache,
    main_weights: &[f32],
    offset_weights: &[f32],
    mask_weights: Option<&[f32]>,
    output_gradient: &Tensor,
) -> Result<DeformConvGradients> {
    let l = layout(input_shape, spec, output_shape)?;
    if cache.input_shape != input_shape
        || cache.output_shape != output_shape
        || output_gradient.shape != output_shape
        || output_gradient.data.len() != l.output_count
        || cache.input.len() != l.input_count
        || main_weights.len() != l.main_weight_count
        || offset_weights.len() != l.predictor_weight_count
    {
        return Err(invalid("backward shape or parameter buffer mismatch"));
    }
    if spec.use_mask != mask_weights.is_some() {
        return Err(invalid("mask weight presence does not match use_mask"));
    }
    let mut input_gradient = vec![0.0; l.input_count];
    let mut weight_gradient = vec![0.0; l.main_weight_count];
    let mut mapped_offset_gradient = vec![0.0; l.batches * l.offset_filters * l.output_spatial];
    let mut mask_probability_gradient = spec
        .use_mask
        .then(|| vec![0.0; l.batches * l.kernel_area * l.output_spatial]);

    for b in 0..l.batches {
        for filter in 0..l.filters {
            let group = filter / l.filters_per_group;
            for oy in 0..l.output_height {
                for ox in 0..l.output_width {
                    let pos = oy * l.output_width + ox;
                    let output_index = (b * l.filters + filter) * l.output_spatial + pos;
                    let incoming_gradient = output_gradient.data[output_index];
                    if incoming_gradient != 0.0 && cache.raw_output_nondiff_boundary[output_index] {
                        return Err(invalid(
                            "main convolution derivative is undefined at a nonfinite or finite-clamp boundary",
                        ));
                    }
                    let grad = incoming_gradient * cache.raw_output_slope[output_index];
                    for local_channel in 0..l.channels_per_group {
                        let channel = group * l.channels_per_group + local_channel;
                        let channel_base =
                            input_index(b, channel, 0, 0, l.channels, l.height, l.width);
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let k = ky * spec.size + kx;
                                let y_plane = 2 * k;
                                let x_plane = y_plane + 1;
                                let offset_y_index = plane_index(
                                    b,
                                    y_plane,
                                    pos,
                                    l.offset_filters,
                                    l.output_spatial,
                                );
                                let offset_x_index = plane_index(
                                    b,
                                    x_plane,
                                    pos,
                                    l.offset_filters,
                                    l.output_spatial,
                                );
                                let base_y = oy as f32 * spec.stride_y as f32
                                    - (spec.padding * spec.dilation) as f32
                                    + (ky * spec.dilation) as f32;
                                let base_x = ox as f32 * spec.stride_x as f32
                                    - (spec.padding * spec.dilation) as f32
                                    + (kx * spec.dilation) as f32;
                                let y = base_y + cache.offsets[offset_y_index];
                                let x = base_x + cache.offsets[offset_x_index];
                                let (sample, sample_dy, sample_dx, neighbors) = bilinear_sample(
                                    &cache.input,
                                    channel_base,
                                    l.height,
                                    l.width,
                                    y,
                                    x,
                                );
                                let mask_index =
                                    plane_index(b, k, pos, l.kernel_area, l.output_spatial);
                                let mask = cache
                                    .masks
                                    .as_ref()
                                    .map_or(1.0, |values| values[mask_index]);
                                let weight_index = (filter * l.channels_per_group + local_channel)
                                    * l.kernel_area
                                    + k;
                                let weight = main_weights[weight_index];
                                weight_gradient[weight_index] += grad * sample * mask;
                                mapped_offset_gradient[offset_y_index] +=
                                    grad * weight * mask * sample_dy;
                                mapped_offset_gradient[offset_x_index] +=
                                    grad * weight * mask * sample_dx;
                                if let Some(mask_gradient) = &mut mask_probability_gradient {
                                    mask_gradient[mask_index] += grad * weight * sample;
                                }
                                let sample_gradient = grad * weight * mask;
                                for (index, coefficient) in neighbors.into_iter().flatten() {
                                    input_gradient[index] += sample_gradient * coefficient;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Pull offset derivatives through the centered-LHTAN map, its second
    // finite clamp, optional per-image standardization, and the first clamp.
    let mut offset_predictor_gradient = vec![0.0; mapped_offset_gradient.len()];
    let per_batch_offsets = l.offset_filters * l.output_spatial;
    for b in 0..l.batches {
        let start = b * per_batch_offsets;
        let mut stage2_gradient = vec![0.0; per_batch_offsets];
        for i in 0..per_batch_offsets {
            let index = start + i;
            let gradient = mapped_offset_gradient[index];
            if gradient != 0.0 && cache.offset_lhtan_boundary[index] {
                return Err(invalid(
                    "offset derivative is undefined at a centered-LHTAN boundary",
                ));
            }
            if gradient != 0.0 && cache.offset_stage2_nondiff_boundary[index] {
                return Err(invalid(
                    "offset derivative is undefined at a nonfinite or finite-clamp boundary",
                ));
            }
            stage2_gradient[i] = gradient
                * cache.offset_lhtan_slope[index]
                * cache.offset_map_slope[index]
                * cache.offset_stage2_slope[index];
        }
        if cache.offset_normalization_boundary[b]
            && stage2_gradient.iter().any(|gradient| *gradient != 0.0)
        {
            return Err(invalid(
                "offset normalization derivative is undefined at a branch boundary",
            ));
        }
        let normalized_gradient = if cache.offset_normalized[b] {
            let count = per_batch_offsets as f64;
            let mean = cache.offset_mean[b];
            let stddev = cache.offset_stddev[b];
            let sum_gradient = stage2_gradient
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>();
            let centered_dot = stage2_gradient
                .iter()
                .enumerate()
                .map(|(i, value)| *value as f64 * (cache.offset_stage1[start + i] as f64 - mean))
                .sum::<f64>();
            (0..per_batch_offsets)
                .map(|i| {
                    let centered = cache.offset_stage1[start + i] as f64 - mean;
                    (((stage2_gradient[i] as f64 - sum_gradient / count) / stddev)
                        - centered * centered_dot / (count * stddev * stddev * stddev))
                        as f32
                })
                .collect::<Vec<_>>()
        } else {
            stage2_gradient
        };
        for i in 0..per_batch_offsets {
            let index = start + i;
            let gradient = normalized_gradient[i];
            if gradient != 0.0 && cache.offset_stage1_nondiff_boundary[index] {
                return Err(invalid(
                    "offset derivative is undefined at a nonfinite or finite-clamp boundary",
                ));
            }
            offset_predictor_gradient[index] = gradient * cache.offset_stage1_slope[index];
        }
    }

    let mut mask_predictor_gradient = if spec.use_mask {
        let probabilities = cache.masks.as_ref().expect("validated mask cache");
        let mut gradient = mask_probability_gradient.expect("mask derivative exists");
        let sigmoid_slope = cache
            .mask_sigmoid_slope
            .as_ref()
            .expect("sigmoid slopes exist");
        let sigmoid_boundary = cache
            .mask_sigmoid_boundary
            .as_ref()
            .expect("sigmoid boundaries exist");
        let predictor_slope = cache
            .mask_predictor_slope
            .as_ref()
            .expect("mask predictor slopes exist");
        let predictor_nondiff_boundary = cache
            .mask_predictor_nondiff_boundary
            .as_ref()
            .expect("mask predictor boundary flags exist");
        for i in 0..gradient.len() {
            if gradient[i] != 0.0 && sigmoid_boundary[i] {
                return Err(invalid(
                    "mask derivative is undefined at a sigmoid clamp boundary",
                ));
            }
            if gradient[i] != 0.0 && predictor_nondiff_boundary[i] {
                return Err(invalid(
                    "mask derivative is undefined at a nonfinite or finite-clamp boundary",
                ));
            }
            gradient[i] *=
                probabilities[i] * (1.0 - probabilities[i]) * sigmoid_slope[i] * predictor_slope[i];
        }
        Some(gradient)
    } else {
        None
    };

    let mut offset_weight_gradient = vec![0.0; l.predictor_weight_count];
    let mut offset_bias_gradient = vec![0.0; l.offset_filters];
    for b in 0..l.batches {
        for out_filter in 0..l.offset_filters {
            for pos in 0..l.output_spatial {
                let dy = offset_predictor_gradient
                    [plane_index(b, out_filter, pos, l.offset_filters, l.output_spatial)];
                offset_bias_gradient[out_filter] += dy;
                for column in 0..l.column_channels {
                    offset_weight_gradient[out_filter * l.column_channels + column] += dy
                        * cache.columns
                            [column_index(b, column, pos, l.column_channels, l.output_spatial)];
                }
            }
        }
    }
    add_predictor_input_gradient(
        &mut input_gradient,
        &offset_predictor_gradient,
        offset_weights,
        l.offset_filters,
        l,
        spec,
    );

    let mut mask_weight_gradient = spec.use_mask.then(|| vec![0.0; l.mask_weight_count]);
    let mut mask_bias_gradient = spec.use_mask.then(|| vec![0.0; l.kernel_area]);
    if let Some(predictor_gradient) = &mut mask_predictor_gradient {
        let weights = mask_weights.expect("validated mask weights");
        for b in 0..l.batches {
            for out_filter in 0..l.kernel_area {
                for pos in 0..l.output_spatial {
                    let dy = predictor_gradient
                        [plane_index(b, out_filter, pos, l.kernel_area, l.output_spatial)];
                    mask_bias_gradient
                        .as_mut()
                        .expect("mask bias gradient exists")[out_filter] += dy;
                    for column in 0..l.column_channels {
                        mask_weight_gradient
                            .as_mut()
                            .expect("mask weight gradient exists")
                            [out_filter * l.column_channels + column] += dy
                            * cache.columns
                                [column_index(b, column, pos, l.column_channels, l.output_spatial)];
                    }
                }
            }
        }
        add_predictor_input_gradient(
            &mut input_gradient,
            predictor_gradient,
            weights,
            l.kernel_area,
            l,
            spec,
        );
    }

    for (index, value) in input_gradient.iter_mut().enumerate() {
        if *value != 0.0 && cache.input_nondiff_boundary[index] {
            return Err(invalid(
                "input derivative is undefined at a nonfinite or finite-clamp boundary",
            ));
        }
        if !value.is_finite() {
            *value = 0.0;
        } else {
            *value *= cache.input_slope[index];
        }
    }
    Ok(DeformConvGradients {
        input: input_gradient,
        weights: weight_gradient,
        offset_weights: offset_weight_gradient,
        offset_biases: offset_bias_gradient,
        mask_weights: mask_weight_gradient,
        mask_biases: mask_bias_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::{deform_conv_backward, deform_conv_forward};
    use crate::model::{Activation, DeformConvSpec, TensorShape};
    use crate::runtime::Tensor;

    fn spec(
        _channels: usize,
        filters: usize,
        groups: usize,
        size: usize,
        mask: bool,
    ) -> DeformConvSpec {
        DeformConvSpec {
            filters,
            groups,
            size,
            stride_x: 1,
            stride_y: 1,
            dilation: 1,
            padding: 0,
            batch_normalize: false,
            use_mask: mask,
            activation: Activation::Linear,
        }
    }

    fn objective(
        input: &Tensor,
        spec: &DeformConvSpec,
        output_shape: TensorShape,
        weights: &[f32],
        offset_weights: &[f32],
        offset_biases: &[f32],
        mask_weights: Option<&[f32]>,
        mask_biases: Option<&[f32]>,
        upstream: &[f32],
    ) -> f32 {
        let (output, _) = deform_conv_forward(
            input,
            spec,
            output_shape,
            weights,
            offset_weights,
            offset_biases,
            mask_weights,
            mask_biases,
        )
        .unwrap();
        output.data.iter().zip(upstream).map(|(a, b)| a * b).sum()
    }

    fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance * (1.0 + actual.abs().max(expected.abs())),
            "{context}: actual={actual}, expected={expected}, error={error}"
        );
    }

    #[test]
    fn deform_conv_axis_layout_pins_cpp_offset_mask_groups_and_batch_nchw() {
        use crate::axis_fixture::{axis_tensor, nchw_index, probe_shape};

        // C++ `deformable_im2col_cpu` (`deform_conv_layer.cpp:518-560`) stores
        // offsets as interleaved (dy,dx) planes per kernel point, with each
        // plane `[out_h,out_w]`; mask planes are one per kernel point. The
        // forward at `:834-852` shares those fields across convolution groups.
        let input_shape = probe_shape(); // B=2,C=2,H=2,W=3
        let input = Tensor::from_data(input_shape, axis_tensor(input_shape)).unwrap();
        let filters = 2;
        let groups = 2;
        let size = 3;
        let kernel_area = size * size;
        let output_shape = TensorShape::new(
            input_shape.width,
            input_shape.height,
            filters,
            input_shape.batch,
        );
        let mut config = spec(input_shape.channels, filters, groups, size, true);
        config.padding = 1;
        let mut main_weights = vec![0.0; filters * (input_shape.channels / groups) * kernel_area];
        main_weights[1] = 1.25;
        main_weights[7] = -0.75;
        main_weights[kernel_area + 2] = 0.5;
        main_weights[kernel_area + 6] = 2.0;
        let predictor_weights = vec![0.0; 2 * kernel_area * input_shape.channels * kernel_area];
        let mut offset_biases = vec![0.0; 2 * kernel_area];
        let mut mask_biases = vec![0.0; kernel_area];
        for k in 0..kernel_area {
            let signed = k as f32 - 4.0;
            offset_biases[2 * k] = 0.05 * signed;
            offset_biases[2 * k + 1] = -0.07 * signed;
            mask_biases[k] = signed;
        }
        let mask_weights = vec![0.0; kernel_area * input_shape.channels * kernel_area];

        let (actual, _) = deform_conv_forward(
            &input,
            &config,
            output_shape,
            &main_weights,
            &predictor_weights,
            &offset_biases,
            Some(&mask_weights),
            Some(&mask_biases),
        )
        .unwrap();

        let bilinear_from_cpp = |n: usize, c: usize, y: f32, x: f32| {
            if y < -1.0 || y > input_shape.height as f32 || x < -1.0 || x > input_shape.width as f32
            {
                return 0.0;
            }
            let y0 = y.floor() as isize;
            let x0 = x.floor() as isize;
            let ly = y - y0 as f32;
            let lx = x - x0 as f32;
            let mut sample = 0.0;
            for (iy, wy) in [(y0, 1.0 - ly), (y0 + 1, ly)] {
                for (ix, wx) in [(x0, 1.0 - lx), (x0 + 1, lx)] {
                    if iy >= 0
                        && iy < input_shape.height as isize
                        && ix >= 0
                        && ix < input_shape.width as isize
                    {
                        sample += wy
                            * wx
                            * input.data[nchw_index(input_shape, n, c, iy as usize, ix as usize)];
                    }
                }
            }
            sample
        };

        for n in 0..input_shape.batch {
            for filter in 0..filters {
                // With C=F=groups=2, C++'s group loop maps output filter f
                // to input channel f; offsets and masks remain shared.
                let input_channel = filter;
                for oy in 0..output_shape.height {
                    for ox in 0..output_shape.width {
                        let mut expected = 0.0;
                        for ky in 0..size {
                            for kx in 0..size {
                                let k = ky * size + kx;
                                let weight = main_weights[filter * kernel_area + k];
                                if weight == 0.0 {
                                    continue;
                                }
                                let base_y = oy as f32 - config.padding as f32 + ky as f32;
                                let base_x = ox as f32 - config.padding as f32 + kx as f32;
                                let sample = bilinear_from_cpp(
                                    n,
                                    input_channel,
                                    base_y + offset_biases[2 * k],
                                    base_x + offset_biases[2 * k + 1],
                                );
                                let mask = 1.0 / (1.0 + (-mask_biases[k]).exp());
                                expected += weight * sample * mask;
                            }
                        }
                        let output_index = nchw_index(output_shape, n, filter, oy, ox);
                        assert_close(
                            actual.data[output_index],
                            expected,
                            2.0e-6,
                            &format!(
                                "C++ deform_conv axis/layout n={n}, filter={filter}, y={oy}, x={ox} (offset dy/dx, mask-k, groups, NCHW)"
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_zero_offset_sampling_matches_hand_computation() {
        let input_shape = TensorShape::new(3, 2, 2, 1);
        let output_shape = input_shape;
        let input = Tensor::from_data(
            input_shape,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
            ],
        )
        .unwrap();
        let config = spec(2, 2, 2, 1, false);
        let (output, _) = deform_conv_forward(
            &input,
            &config,
            output_shape,
            &[2.0, 3.0],
            &[0.0; 4],
            &[0.0; 2],
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            output.data,
            vec![
                2.0, 4.0, 6.0, 8.0, 10.0, 12.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0
            ]
        );
    }

    #[test]
    fn source_nchw_group_and_kernel_axes_match_hand_computation() {
        let input_shape = TensorShape::new(2, 2, 4, 1);
        let input = Tensor::from_data(
            input_shape,
            vec![
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, -1.0, -2.0,
                -3.0, -4.0,
            ],
        )
        .unwrap();
        let mut config = spec(4, 4, 2, 2, false);
        config.padding = 0;
        let output_shape = TensorShape::new(1, 1, 4, 1);
        // Main weights are [filter, input-channel-within-group, ky, kx].
        let mut weights = vec![0.0; 4 * 2 * 2 * 2];
        weights[0] = 1.0; // filter 0: channel 0, upper-left
        weights[4 + 1] = 1.0; // filter 0: channel 1, upper-right
        weights[8 + 2] = 1.0; // filter 1: channel 0, lower-left
        weights[8 + 4 + 3] = 1.0; // filter 1: channel 1, lower-right
        weights[16] = 1.0; // filter 2: channel 2, upper-left
        weights[16 + 4 + 1] = 1.0; // filter 2: channel 3, upper-right
        weights[24 + 2] = 1.0; // filter 3: channel 2, lower-left
        weights[24 + 4 + 3] = 1.0; // filter 3: channel 3, lower-right

        let (output, _) = deform_conv_forward(
            &input,
            &config,
            output_shape,
            &weights,
            &[0.0; 128],
            &[0.0; 8],
            None,
            None,
        )
        .unwrap();
        assert_eq!(output.data, vec![21.0, 43.0, 98.0, 296.0]);
    }

    #[test]
    fn boundary_zero_padded_sampling_matches_hand_computation() {
        let shape = TensorShape::new(1, 1, 1, 1);
        let input = Tensor::from_data(shape, vec![2.0]).unwrap();
        let config = spec(1, 1, 1, 1, false);
        let (output, _) = deform_conv_forward(
            &input,
            &config,
            shape,
            &[1.0],
            &[0.0; 2],
            &[-0.5, 0.0],
            None,
            None,
        )
        .unwrap();
        // The sample is halfway above the only row; zero padding contributes
        // half of 2.0 and half of 0.0.
        assert_eq!(output.data, vec![1.0]);
    }

    #[test]
    fn fractional_offset_and_sigmoid_mask_match_hand_computation() {
        let input_shape = TensorShape::new(2, 2, 1, 1);
        let input = Tensor::from_data(input_shape, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let config = spec(1, 1, 1, 1, true);
        let (output, _) = deform_conv_forward(
            &input,
            &config,
            input_shape,
            &[1.0],
            &[0.0; 2],
            &[0.25, 0.5],
            Some(&[0.0]),
            Some(&[0.0]),
        )
        .unwrap();
        // Bilinear sample at (0.25, 0.5) is 2.0; sigmoid(0) modulates it by 1/2.
        assert_close(output.data[0], 1.0, 1.0e-6, "masked bilinear value");
    }

    fn check_block(
        values: &[f32],
        analytic: &[f32],
        epsilon: f32,
        label: &str,
        mut evaluate: impl FnMut(&[f32]) -> f32,
    ) {
        for index in 0..values.len() {
            let mut candidate = values.to_vec();
            candidate[index] += epsilon;
            let plus = evaluate(&candidate);
            candidate[index] -= 2.0 * epsilon;
            let minus = evaluate(&candidate);
            assert_close(
                analytic[index],
                (plus - minus) / (2.0 * epsilon),
                8.0e-3,
                &format!("{label}[{index}]"),
            );
        }
    }

    fn finite_difference_case(
        input: Tensor,
        config: DeformConvSpec,
        output_shape: TensorShape,
        weights: Vec<f32>,
        offset_weights: Vec<f32>,
        offset_biases: Vec<f32>,
        mask_weights: Option<Vec<f32>>,
        mask_biases: Option<Vec<f32>>,
        upstream: Vec<f32>,
        label: &str,
    ) {
        let (_, cache) = deform_conv_forward(
            &input,
            &config,
            output_shape,
            &weights,
            &offset_weights,
            &offset_biases,
            mask_weights.as_deref(),
            mask_biases.as_deref(),
        )
        .unwrap();
        let gradient = deform_conv_backward(
            input.shape,
            &config,
            output_shape,
            &cache,
            &weights,
            &offset_weights,
            mask_weights.as_deref(),
            &Tensor::from_data(output_shape, upstream.clone()).unwrap(),
        )
        .unwrap();
        let epsilon = 5.0e-4;

        for index in 0..input.data.len() {
            let original = input.data[index];
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus.data[index] = original + epsilon;
            minus.data[index] = original - epsilon;
            let numeric = (objective(
                &plus,
                &config,
                output_shape,
                &weights,
                &offset_weights,
                &offset_biases,
                mask_weights.as_deref(),
                mask_biases.as_deref(),
                &upstream,
            ) - objective(
                &minus,
                &config,
                output_shape,
                &weights,
                &offset_weights,
                &offset_biases,
                mask_weights.as_deref(),
                mask_biases.as_deref(),
                &upstream,
            )) / (2.0 * epsilon);
            assert_close(
                gradient.input[index],
                numeric,
                8.0e-3,
                &format!("{label} input[{index}]"),
            );
        }
        check_block(
            &weights,
            &gradient.weights,
            epsilon,
            &format!("{label} main weights"),
            |candidate| {
                objective(
                    &input,
                    &config,
                    output_shape,
                    candidate,
                    &offset_weights,
                    &offset_biases,
                    mask_weights.as_deref(),
                    mask_biases.as_deref(),
                    &upstream,
                )
            },
        );
        check_block(
            &offset_weights,
            &gradient.offset_weights,
            epsilon,
            &format!("{label} offset weights"),
            |candidate| {
                objective(
                    &input,
                    &config,
                    output_shape,
                    &weights,
                    candidate,
                    &offset_biases,
                    mask_weights.as_deref(),
                    mask_biases.as_deref(),
                    &upstream,
                )
            },
        );
        check_block(
            &offset_biases,
            &gradient.offset_biases,
            epsilon,
            &format!("{label} offset biases"),
            |candidate| {
                objective(
                    &input,
                    &config,
                    output_shape,
                    &weights,
                    &offset_weights,
                    candidate,
                    mask_weights.as_deref(),
                    mask_biases.as_deref(),
                    &upstream,
                )
            },
        );
        if let (Some(values), Some(analytic)) =
            (mask_weights.as_deref(), gradient.mask_weights.as_deref())
        {
            check_block(
                values,
                analytic,
                epsilon,
                &format!("{label} mask weights"),
                |candidate| {
                    objective(
                        &input,
                        &config,
                        output_shape,
                        &weights,
                        &offset_weights,
                        &offset_biases,
                        Some(candidate),
                        mask_biases.as_deref(),
                        &upstream,
                    )
                },
            );
        }
        if let (Some(values), Some(analytic)) =
            (mask_biases.as_deref(), gradient.mask_biases.as_deref())
        {
            check_block(
                values,
                analytic,
                epsilon,
                &format!("{label} mask biases"),
                |candidate| {
                    objective(
                        &input,
                        &config,
                        output_shape,
                        &weights,
                        &offset_weights,
                        &offset_biases,
                        mask_weights.as_deref(),
                        Some(candidate),
                        &upstream,
                    )
                },
            );
        }
    }

    #[test]
    fn exact_backward_matches_finite_differences_with_conditional_offset_normalization() {
        let input_shape = TensorShape::new(3, 2, 2, 1);
        let input = Tensor::from_data(
            input_shape,
            vec![
                -0.37, 0.21, 0.56, -0.14, 0.44, 0.79, 0.18, -0.28, 0.67, 0.33, -0.55, 0.12,
            ],
        )
        .unwrap();
        let config = spec(2, 2, 2, 1, false);
        let offset_weights = vec![0.13, -0.07, 0.05, 0.09];
        let offset_biases = vec![5.2, -4.3];
        let upstream = vec![
            0.3, -0.2, 0.4, -0.5, 0.1, 0.25, -0.35, 0.45, 0.2, -0.15, 0.5, -0.3,
        ];
        let (_, cache) = deform_conv_forward(
            &input,
            &config,
            input_shape,
            &[0.8, -0.3],
            &offset_weights,
            &offset_biases,
            None,
            None,
        )
        .unwrap();
        assert!(
            cache.offset_normalized[0],
            "fixture must activate per-image normalization"
        );
        finite_difference_case(
            input,
            config,
            input_shape,
            vec![0.8, -0.3],
            offset_weights,
            offset_biases,
            None,
            None,
            upstream,
            "normalized DCNv1",
        );
    }

    #[test]
    fn exact_backward_matches_finite_differences_for_grouped_masked_sampling() {
        let input_shape = TensorShape::new(3, 3, 2, 1);
        let output_shape = TensorShape::new(2, 2, 2, 1);
        let input = Tensor::from_data(
            input_shape,
            vec![
                0.2, 0.7, -0.4, 0.5, -0.1, 0.8, -0.6, 0.3, 0.9, 0.4, -0.2, 0.6, 0.1, -0.7, 0.3,
                0.8, -0.5, 0.2,
            ],
        )
        .unwrap();
        let config = spec(2, 2, 1, 2, true);
        let weights = (0..16).map(|i| 0.11 + i as f32 * 0.013).collect::<Vec<_>>();
        let offset_weights = (0..64)
            .map(|i| ((i as f32 * 0.17).sin()) * 0.035)
            .collect::<Vec<_>>();
        let offset_biases = vec![0.11, -0.08, 0.17, -0.13, 0.09, 0.14, -0.06, 0.12];
        let mask_weights = (0..32)
            .map(|i| ((i as f32 * 0.23).cos()) * 0.025)
            .collect::<Vec<_>>();
        let mask_biases = vec![-0.17, 0.11, 0.23, -0.09];
        let upstream = vec![0.3, -0.1, 0.2, 0.4, -0.25, 0.15, 0.35, -0.3];
        finite_difference_case(
            input,
            config,
            output_shape,
            weights,
            offset_weights,
            offset_biases,
            Some(mask_weights),
            Some(mask_biases),
            upstream,
            "grouped DCNv2",
        );
    }

    #[test]
    fn nondifferentiable_sanitization_boundaries_are_rejected() {
        let shape = TensorShape::new(1, 1, 1, 1);
        let output_gradient = Tensor::filled(shape, 1.0);
        let unmasked = spec(1, 1, 1, 1, false);
        let input_boundary = Tensor::from_data(shape, vec![10_000.0]).unwrap();
        let (_, cache) = deform_conv_forward(
            &input_boundary,
            &unmasked,
            shape,
            &[0.001],
            &[0.0; 2],
            &[0.0; 2],
            None,
            None,
        )
        .unwrap();
        assert!(
            deform_conv_backward(
                shape,
                &unmasked,
                shape,
                &cache,
                &[0.001],
                &[0.0; 2],
                None,
                &output_gradient,
            )
            .is_err()
        );

        let ordinary_input = Tensor::from_data(shape, vec![1.0]).unwrap();
        let (_, cache) = deform_conv_forward(
            &ordinary_input,
            &unmasked,
            shape,
            &[10_000.0],
            &[0.0; 2],
            &[0.0; 2],
            None,
            None,
        )
        .unwrap();
        assert!(
            deform_conv_backward(
                shape,
                &unmasked,
                shape,
                &cache,
                &[10_000.0],
                &[0.0; 2],
                None,
                &output_gradient,
            )
            .is_err()
        );

        let nonfinite_input = Tensor::from_data(shape, vec![f32::NAN]).unwrap();
        let (output, cache) = deform_conv_forward(
            &nonfinite_input,
            &unmasked,
            shape,
            &[1.0],
            &[0.0; 2],
            &[0.0; 2],
            None,
            None,
        )
        .unwrap();
        assert_eq!(output.data, vec![0.0]);
        assert!(
            deform_conv_backward(
                shape,
                &unmasked,
                shape,
                &cache,
                &[1.0],
                &[0.0; 2],
                None,
                &output_gradient,
            )
            .is_err()
        );

        let masked = spec(1, 1, 1, 1, true);
        let (_, cache) = deform_conv_forward(
            &ordinary_input,
            &masked,
            shape,
            &[1.0],
            &[0.0; 2],
            &[0.0; 2],
            Some(&[0.0]),
            Some(&[80.0]),
        )
        .unwrap();
        assert!(
            deform_conv_backward(
                shape,
                &masked,
                shape,
                &cache,
                &[1.0],
                &[0.0; 2],
                Some(&[0.0]),
                &output_gradient,
            )
            .is_err()
        );
    }

    #[test]
    fn centered_lhtan_uses_source_slope_outside_its_linear_interval() {
        let (low, low_slope, _, _) = super::centered_lhtan(-8.0, 2.0);
        let (high, high_slope, _, _) = super::centered_lhtan(8.0, 2.0);
        assert_close(low, -2.006, 1.0e-6, "low stabilized offset");
        assert_close(high, 2.006, 1.0e-6, "high stabilized offset");
        assert_eq!((low_slope, high_slope), (0.001, 0.001));
    }
}
