//! CPU reference math for source-derived DCNv4 spatial aggregation.
//!
//! The module computes the offset projection and deformable sampling only. The
//! runtime applies the layer's bias/BN and activation epilogue and differentiates
//! that epilogue around the gradients returned here.

use crate::error::{Error, Result};
use crate::model::{Dcnv4Spec, TensorShape};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dcnv4ForwardCache {
    /// Dense `[batch, output_y, output_x, padded_offset_dim]` offset field.
    pub offsets: Vec<f32>,
    /// Unbiased, unnormalized deformable aggregation in NCHW order.
    pub raw_output: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct Dcnv4Gradients {
    pub input: Vec<f32>,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    output_width: usize,
    output_height: usize,
    kernel_area: usize,
    kernel_points: usize,
    offset_dim: usize,
    coarse_width: usize,
    coarse_height: usize,
    offset_weight_count: usize,
    input_count: usize,
    output_count: usize,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[dcnv4]: {}", message.into()))
}

fn checked_output_dim(
    input: usize,
    padding: usize,
    size: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    if size == 0 || stride == 0 || dilation == 0 {
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

fn checked_count(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |count, value| {
        count
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{name} element count overflows")))
    })
}

fn layout(input_shape: TensorShape, spec: &Dcnv4Spec) -> Result<Layout> {
    if input_shape.width == 0
        || input_shape.height == 0
        || input_shape.channels == 0
        || input_shape.batch == 0
    {
        return Err(invalid("input dimensions must be non-zero"));
    }
    if spec.filters != input_shape.channels {
        return Err(invalid(format!(
            "filters={} must equal input channels={}",
            spec.filters, input_shape.channels
        )));
    }
    if spec.groups == 0 || input_shape.channels % spec.groups != 0 {
        return Err(invalid(format!(
            "groups={} must divide input channels={}",
            spec.groups, input_shape.channels
        )));
    }
    if spec.stride_x == 0
        || spec.stride_y == 0
        || spec.size == 0
        || spec.dilation == 0
        || spec.d_stride == 0
    {
        return Err(invalid(
            "size, strides, dilation, and d_stride must be positive",
        ));
    }
    if !spec.offset_scale.is_finite() {
        return Err(invalid("offset_scale must be finite"));
    }

    let kernel_area = spec
        .size
        .checked_mul(spec.size)
        .ok_or_else(|| invalid("kernel area overflows"))?;
    let kernel_points = kernel_area
        .checked_sub(usize::from(spec.remove_center))
        .ok_or_else(|| invalid("remove_center leaves no active kernel points"))?;
    if kernel_points == 0 {
        return Err(invalid("remove_center leaves no active kernel points"));
    }
    let raw_offset_dim = spec
        .groups
        .checked_mul(kernel_points)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| invalid("offset dimension overflows"))?;
    let offset_dim = raw_offset_dim
        .checked_add(7)
        .and_then(|value| value.checked_div(8))
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid("padded offset dimension overflows"))?;

    let output_width = checked_output_dim(
        input_shape.width,
        spec.padding,
        spec.size,
        spec.stride_x,
        spec.dilation,
    )
    .ok_or_else(|| invalid("invalid or empty output width"))?;
    let output_height = checked_output_dim(
        input_shape.height,
        spec.padding,
        spec.size,
        spec.stride_y,
        spec.dilation,
    )
    .ok_or_else(|| invalid("invalid or empty output height"))?;
    let coarse_width = output_width
        .checked_add(spec.d_stride - 1)
        .and_then(|value| value.checked_div(spec.d_stride))
        .ok_or_else(|| invalid("coarse output width overflows"))?;
    let coarse_height = output_height
        .checked_add(spec.d_stride - 1)
        .and_then(|value| value.checked_div(spec.d_stride))
        .ok_or_else(|| invalid("coarse output height overflows"))?;
    let offset_weight_count = checked_count(
        &[offset_dim, input_shape.channels, kernel_area],
        "offset weights",
    )?;
    let input_count = checked_count(
        &[
            input_shape.batch,
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        ],
        "input",
    )?;
    let output_count = checked_count(
        &[
            input_shape.batch,
            input_shape.channels,
            output_height,
            output_width,
        ],
        "output",
    )?;
    // Also prove that the packed dense and coarse offset buffers are addressable.
    checked_count(
        &[input_shape.batch, output_height, output_width, offset_dim],
        "dense offsets",
    )?;
    checked_count(
        &[input_shape.batch, coarse_height, coarse_width, offset_dim],
        "coarse offsets",
    )?;

    Ok(Layout {
        output_width,
        output_height,
        kernel_area,
        kernel_points,
        offset_dim,
        coarse_width,
        coarse_height,
        offset_weight_count,
        input_count,
        output_count,
    })
}

fn input_coord(
    output_coord: usize,
    stride: usize,
    kernel_coord: usize,
    dilation: usize,
    padding: usize,
    limit: usize,
) -> Result<Option<usize>> {
    let raw = output_coord
        .checked_mul(stride)
        .and_then(|value| {
            kernel_coord
                .checked_mul(dilation)
                .and_then(|k| value.checked_add(k))
        })
        .ok_or_else(|| invalid("offset projection coordinate overflows"))?;
    Ok(raw
        .checked_sub(padding)
        .filter(|coordinate| *coordinate < limit))
}

fn coarse_stride(base: usize, d_stride: usize) -> Result<usize> {
    base.checked_mul(d_stride)
        .ok_or_else(|| invalid("coarse offset stride overflows"))
}

#[derive(Clone, Copy)]
struct Interpolation {
    y: usize,
    x: usize,
    weight: f32,
}

fn clamp_index(index: i64, limit: usize) -> usize {
    index.clamp(0, limit.saturating_sub(1) as i64) as usize
}

fn border_interpolation(height: usize, width: usize, y: f32, x: f32) -> [Interpolation; 4] {
    let y0f = y.floor();
    let x0f = x.floor();
    let y0 = y0f as i64;
    let x0 = x0f as i64;
    let y1 = y0.saturating_add(1);
    let x1 = x0.saturating_add(1);
    let ly = y - y0f;
    let lx = x - x0f;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;
    [
        Interpolation {
            y: clamp_index(y0, height),
            x: clamp_index(x0, width),
            weight: hy * hx,
        },
        Interpolation {
            y: clamp_index(y0, height),
            x: clamp_index(x1, width),
            weight: hy * lx,
        },
        Interpolation {
            y: clamp_index(y1, height),
            x: clamp_index(x0, width),
            weight: ly * hx,
        },
        Interpolation {
            y: clamp_index(y1, height),
            x: clamp_index(x1, width),
            weight: ly * lx,
        },
    ]
}

fn upsample_offsets(
    coarse: &[f32],
    batch: usize,
    coarse_height: usize,
    coarse_width: usize,
    dense_height: usize,
    dense_width: usize,
    offset_dim: usize,
    d_stride: usize,
) -> Vec<f32> {
    if coarse_height == dense_height && coarse_width == dense_width {
        return coarse.to_vec();
    }
    let mut dense = vec![0.0; batch * dense_height * dense_width * offset_dim];
    for b in 0..batch {
        for y in 0..dense_height {
            let src_y = y as f32 / d_stride as f32;
            for x in 0..dense_width {
                let src_x = x as f32 / d_stride as f32;
                let neighbors = border_interpolation(coarse_height, coarse_width, src_y, src_x);
                for p in 0..offset_dim {
                    let mut value = 0.0;
                    for neighbor in neighbors {
                        let index = ((b * coarse_height + neighbor.y) * coarse_width + neighbor.x)
                            * offset_dim
                            + p;
                        value += neighbor.weight * coarse[index];
                    }
                    let index = ((b * dense_height + y) * dense_width + x) * offset_dim + p;
                    dense[index] = value;
                }
            }
        }
    }
    dense
}

#[derive(Clone, Copy)]
struct BilinearSample {
    value: f32,
    dx: f32,
    dy: f32,
    indices: [Option<usize>; 4],
    weights: [f32; 4],
}

fn bilinear_zero(
    input: &[f32],
    shape: TensorShape,
    batch: usize,
    channel: usize,
    y: f32,
    x: f32,
) -> BilinearSample {
    let y0f = y.floor();
    let x0f = x.floor();
    let y0 = y0f as i64;
    let x0 = x0f as i64;
    let y1 = y0.saturating_add(1);
    let x1 = x0.saturating_add(1);
    let ly = y - y0f;
    let lx = x - x0f;
    let hy = 1.0 - ly;
    let hx = 1.0 - lx;

    let pixel_index = |yy: i64, xx: i64| {
        if yy < 0 || xx < 0 || yy >= shape.height as i64 || xx >= shape.width as i64 {
            None
        } else {
            Some(
                ((batch * shape.channels + channel) * shape.height + yy as usize) * shape.width
                    + xx as usize,
            )
        }
    };
    let indices = [
        pixel_index(y0, x0),
        pixel_index(y0, x1),
        pixel_index(y1, x0),
        pixel_index(y1, x1),
    ];
    let values = indices.map(|index| index.map_or(0.0, |index| input[index]));
    let weights = [hy * hx, hy * lx, ly * hx, ly * lx];
    let value = values
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum();
    let dx = hy * (values[1] - values[0]) + ly * (values[3] - values[2]);
    let dy = hx * (values[2] - values[0]) + lx * (values[3] - values[1]);
    BilinearSample {
        value,
        dx,
        dy,
        indices,
        weights,
    }
}

fn group_masks(row: &[f32], group_offset: usize, points: usize, softmax: bool) -> Vec<f32> {
    let logits_start = group_offset + 2 * points;
    let mut masks = row[logits_start..logits_start + points].to_vec();
    if softmax {
        let maximum = masks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in &mut masks {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let denominator = sum + 1.0e-6;
        for value in &mut masks {
            *value /= denominator;
        }
    }
    masks
}

/// DCNv4 offset projection plus grouped deformable aggregation, before the
/// runtime bias/BN and activation epilogue.
pub(crate) fn dcnv4_forward(
    input: &[f32],
    input_shape: TensorShape,
    spec: &Dcnv4Spec,
    offset_weights: &[f32],
    offset_biases: &[f32],
) -> Result<(Vec<f32>, Dcnv4ForwardCache)> {
    let layout = layout(input_shape, spec)?;
    if input.len() != layout.input_count {
        return Err(invalid(format!(
            "input has {} values, expected {}",
            input.len(),
            layout.input_count
        )));
    }
    if offset_weights.len() != layout.offset_weight_count {
        return Err(invalid(format!(
            "offset_weights has {} values, expected {}",
            offset_weights.len(),
            layout.offset_weight_count
        )));
    }
    if offset_biases.len() != layout.offset_dim {
        return Err(invalid(format!(
            "offset_biases has {} values, expected {}",
            offset_biases.len(),
            layout.offset_dim
        )));
    }

    let batch = input_shape.batch;
    let channels = input_shape.channels;
    let size = spec.size;
    let coarse_stride_y = coarse_stride(spec.stride_y, spec.d_stride)?;
    let coarse_stride_x = coarse_stride(spec.stride_x, spec.d_stride)?;
    let coarse_count = checked_count(
        &[
            batch,
            layout.coarse_height,
            layout.coarse_width,
            layout.offset_dim,
        ],
        "coarse offsets",
    )?;
    let mut coarse = vec![0.0; coarse_count];
    for b in 0..batch {
        for cy in 0..layout.coarse_height {
            for cx in 0..layout.coarse_width {
                for p in 0..layout.offset_dim {
                    let mut value = 0.0;
                    for c in 0..channels {
                        for ky in 0..size {
                            let iy = input_coord(
                                cy,
                                coarse_stride_y,
                                ky,
                                spec.dilation,
                                spec.padding,
                                input_shape.height,
                            )?;
                            let Some(iy) = iy else { continue };
                            for kx in 0..size {
                                let ix = input_coord(
                                    cx,
                                    coarse_stride_x,
                                    kx,
                                    spec.dilation,
                                    spec.padding,
                                    input_shape.width,
                                )?;
                                let Some(ix) = ix else { continue };
                                let input_index = ((b * channels + c) * input_shape.height + iy)
                                    * input_shape.width
                                    + ix;
                                let weight_index =
                                    (p * channels + c) * layout.kernel_area + ky * size + kx;
                                value += input[input_index] * offset_weights[weight_index];
                            }
                        }
                    }
                    let index = ((b * layout.coarse_height + cy) * layout.coarse_width + cx)
                        * layout.offset_dim
                        + p;
                    coarse[index] = value + offset_biases[p];
                }
            }
        }
    }
    let offsets = upsample_offsets(
        &coarse,
        batch,
        layout.coarse_height,
        layout.coarse_width,
        layout.output_height,
        layout.output_width,
        layout.offset_dim,
        spec.d_stride,
    );

    let spatial = layout.output_width * layout.output_height;
    let channels_per_group = channels / spec.groups;
    let center = (spec.dilation * (size - 1)) >> 1;
    let center_f = center as f32;
    let scale = spec.offset_scale;
    let mut output = vec![0.0; layout.output_count];
    for b in 0..batch {
        for q in 0..spatial {
            let out_y = q / layout.output_width;
            let out_x = q % layout.output_width;
            let p0_x = center as f32 - spec.padding as f32 + out_x as f32 * spec.stride_x as f32;
            let p0_y = center as f32 - spec.padding as f32 + out_y as f32 * spec.stride_y as f32;
            let p0_x_scaled = p0_x - center_f * scale;
            let p0_y_scaled = p0_y - center_f * scale;
            let row_start = (b * spatial + q) * layout.offset_dim;
            let row = &offsets[row_start..row_start + layout.offset_dim];
            for g in 0..spec.groups {
                let group_offset = g * layout.kernel_points * 3;
                let masks = group_masks(row, group_offset, layout.kernel_points, spec.softmax);
                let mut point = 0;
                for kw in 0..size {
                    for kh in 0..size {
                        if spec.remove_center && kw == size / 2 && kh == size / 2 {
                            continue;
                        }
                        let offset_x = row[group_offset + 2 * point];
                        let offset_y = row[group_offset + 2 * point + 1];
                        let sample_x =
                            p0_x_scaled + (kw as f32 * spec.dilation as f32 + offset_x) * scale;
                        let sample_y =
                            p0_y_scaled + (kh as f32 * spec.dilation as f32 + offset_y) * scale;
                        let mask = masks[point];
                        for d in 0..channels_per_group {
                            let channel = g * channels_per_group + d;
                            let sample =
                                bilinear_zero(input, input_shape, b, channel, sample_y, sample_x);
                            output[(b * channels + channel) * spatial + q] += mask * sample.value;
                        }
                        point += 1;
                    }
                }
            }
        }
    }

    Ok((
        output.clone(),
        Dcnv4ForwardCache {
            offsets,
            raw_output: output,
        },
    ))
}

/// Exact derivative of [`dcnv4_forward`] for input and the learned offset
/// projection. The runtime adds bias/BN/activation derivatives around it.
pub(crate) fn dcnv4_backward(
    input: &[f32],
    input_shape: TensorShape,
    spec: &Dcnv4Spec,
    offset_weights: &[f32],
    cache: &Dcnv4ForwardCache,
    output_gradient: &[f32],
) -> Result<Dcnv4Gradients> {
    let layout = layout(input_shape, spec)?;
    if input.len() != layout.input_count {
        return Err(invalid(format!(
            "input has {} values, expected {}",
            input.len(),
            layout.input_count
        )));
    }
    if offset_weights.len() != layout.offset_weight_count {
        return Err(invalid(format!(
            "offset_weights has {} values, expected {}",
            offset_weights.len(),
            layout.offset_weight_count
        )));
    }
    if output_gradient.len() != layout.output_count {
        return Err(invalid(format!(
            "output gradient has {} values, expected {}",
            output_gradient.len(),
            layout.output_count
        )));
    }
    let expected_offsets = checked_count(
        &[
            input_shape.batch,
            layout.output_height,
            layout.output_width,
            layout.offset_dim,
        ],
        "dense offsets",
    )?;
    if cache.offsets.len() != expected_offsets {
        return Err(invalid(format!(
            "cached offsets have {} values, expected {expected_offsets}",
            cache.offsets.len()
        )));
    }

    let batch = input_shape.batch;
    let channels = input_shape.channels;
    let size = spec.size;
    let spatial = layout.output_width * layout.output_height;
    let channels_per_group = channels / spec.groups;
    let center = (spec.dilation * (size - 1)) >> 1;
    let center_f = center as f32;
    let scale = spec.offset_scale;
    let mut input_gradient = vec![0.0; layout.input_count];
    let mut dense_offset_gradient = vec![0.0; expected_offsets];

    for b in 0..batch {
        for q in 0..spatial {
            let out_y = q / layout.output_width;
            let out_x = q % layout.output_width;
            let p0_x = center as f32 - spec.padding as f32 + out_x as f32 * spec.stride_x as f32;
            let p0_y = center as f32 - spec.padding as f32 + out_y as f32 * spec.stride_y as f32;
            let p0_x_scaled = p0_x - center_f * scale;
            let p0_y_scaled = p0_y - center_f * scale;
            let row_start = (b * spatial + q) * layout.offset_dim;
            let row = &cache.offsets[row_start..row_start + layout.offset_dim];
            let row_gradient_start = row_start;
            for g in 0..spec.groups {
                let group_offset = g * layout.kernel_points * 3;
                let masks = group_masks(row, group_offset, layout.kernel_points, spec.softmax);
                let mut mask_gradient = vec![0.0; layout.kernel_points];
                let mut point = 0;
                for kw in 0..size {
                    for kh in 0..size {
                        if spec.remove_center && kw == size / 2 && kh == size / 2 {
                            continue;
                        }
                        let offset_x = row[group_offset + 2 * point];
                        let offset_y = row[group_offset + 2 * point + 1];
                        let sample_x =
                            p0_x_scaled + (kw as f32 * spec.dilation as f32 + offset_x) * scale;
                        let sample_y =
                            p0_y_scaled + (kh as f32 * spec.dilation as f32 + offset_y) * scale;
                        let mask = masks[point];
                        let mut gradient_x = 0.0;
                        let mut gradient_y = 0.0;
                        let mut gradient_mask = 0.0;
                        for d in 0..channels_per_group {
                            let channel = g * channels_per_group + d;
                            let output_index = (b * channels + channel) * spatial + q;
                            let top = output_gradient[output_index];
                            let sample =
                                bilinear_zero(input, input_shape, b, channel, sample_y, sample_x);
                            for (neighbor, weight) in sample.indices.into_iter().zip(sample.weights)
                            {
                                if let Some(index) = neighbor {
                                    input_gradient[index] += top * mask * weight;
                                }
                            }
                            gradient_x += top * mask * sample.dx * scale;
                            gradient_y += top * mask * sample.dy * scale;
                            gradient_mask += top * sample.value;
                        }
                        dense_offset_gradient[row_gradient_start + group_offset + 2 * point] +=
                            gradient_x;
                        dense_offset_gradient[row_gradient_start + group_offset + 2 * point + 1] +=
                            gradient_y;
                        mask_gradient[point] = gradient_mask;
                        point += 1;
                    }
                }
                if spec.softmax {
                    let dot = mask_gradient
                        .iter()
                        .zip(&masks)
                        .map(|(gradient, mask)| gradient * mask)
                        .sum::<f32>();
                    for point in 0..layout.kernel_points {
                        let index =
                            row_gradient_start + group_offset + 2 * layout.kernel_points + point;
                        dense_offset_gradient[index] += masks[point] * (mask_gradient[point] - dot);
                    }
                } else {
                    for point in 0..layout.kernel_points {
                        let index =
                            row_gradient_start + group_offset + 2 * layout.kernel_points + point;
                        dense_offset_gradient[index] += mask_gradient[point];
                    }
                }
            }
        }
    }

    let mut coarse_offset_gradient = vec![
        0.0;
        checked_count(
            &[
                batch,
                layout.coarse_height,
                layout.coarse_width,
                layout.offset_dim,
            ],
            "coarse offset gradients",
        )?
    ];
    if layout.coarse_height == layout.output_height && layout.coarse_width == layout.output_width {
        coarse_offset_gradient.copy_from_slice(&dense_offset_gradient);
    } else {
        for b in 0..batch {
            for y in 0..layout.output_height {
                let src_y = y as f32 / spec.d_stride as f32;
                for x in 0..layout.output_width {
                    let src_x = x as f32 / spec.d_stride as f32;
                    let neighbors = border_interpolation(
                        layout.coarse_height,
                        layout.coarse_width,
                        src_y,
                        src_x,
                    );
                    for p in 0..layout.offset_dim {
                        let gradient = dense_offset_gradient[((b * layout.output_height + y)
                            * layout.output_width
                            + x)
                            * layout.offset_dim
                            + p];
                        for neighbor in neighbors {
                            let index = ((b * layout.coarse_height + neighbor.y)
                                * layout.coarse_width
                                + neighbor.x)
                                * layout.offset_dim
                                + p;
                            coarse_offset_gradient[index] += gradient * neighbor.weight;
                        }
                    }
                }
            }
        }
    }

    let coarse_stride_y = coarse_stride(spec.stride_y, spec.d_stride)?;
    let coarse_stride_x = coarse_stride(spec.stride_x, spec.d_stride)?;
    let mut offset_weight_gradient = vec![0.0; layout.offset_weight_count];
    let mut offset_bias_gradient = vec![0.0; layout.offset_dim];
    for b in 0..batch {
        for cy in 0..layout.coarse_height {
            for cx in 0..layout.coarse_width {
                for p in 0..layout.offset_dim {
                    let gradient = coarse_offset_gradient[((b * layout.coarse_height + cy)
                        * layout.coarse_width
                        + cx)
                        * layout.offset_dim
                        + p];
                    offset_bias_gradient[p] += gradient;
                    for c in 0..channels {
                        for ky in 0..size {
                            let iy = input_coord(
                                cy,
                                coarse_stride_y,
                                ky,
                                spec.dilation,
                                spec.padding,
                                input_shape.height,
                            )?;
                            let Some(iy) = iy else { continue };
                            for kx in 0..size {
                                let ix = input_coord(
                                    cx,
                                    coarse_stride_x,
                                    kx,
                                    spec.dilation,
                                    spec.padding,
                                    input_shape.width,
                                )?;
                                let Some(ix) = ix else { continue };
                                let input_index = ((b * channels + c) * input_shape.height + iy)
                                    * input_shape.width
                                    + ix;
                                let weight_index =
                                    (p * channels + c) * layout.kernel_area + ky * size + kx;
                                offset_weight_gradient[weight_index] +=
                                    gradient * input[input_index];
                                input_gradient[input_index] +=
                                    gradient * offset_weights[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Dcnv4Gradients {
        input: input_gradient,
        offset_weights: offset_weight_gradient,
        offset_biases: offset_bias_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::{Dcnv4Gradients, dcnv4_backward, dcnv4_forward};
    use crate::model::{Activation, Dcnv4Spec, TensorShape};

    fn spec(
        channels: usize,
        size: usize,
        padding: usize,
        d_stride: usize,
        remove_center: bool,
        softmax: bool,
        offset_scale: f32,
    ) -> Dcnv4Spec {
        Dcnv4Spec {
            filters: channels,
            groups: 1,
            size,
            stride_x: 1,
            stride_y: 1,
            dilation: 1,
            padding,
            batch_normalize: false,
            offset_scale,
            remove_center,
            d_stride,
            block_thread: 256,
            softmax,
            activation: Activation::Linear,
        }
    }

    fn parameter_counts(channels: usize, spec: &Dcnv4Spec) -> (usize, usize) {
        let kernel_area = spec.size * spec.size;
        let points = kernel_area - usize::from(spec.remove_center);
        let offset_dim = (spec.groups * points * 3).div_ceil(8) * 8;
        (offset_dim * channels * kernel_area, offset_dim)
    }

    #[test]
    fn one_by_one_unscaled_masks_are_a_hand_computed_channelwise_product() {
        let shape = TensorShape::new(2, 2, 1, 1);
        let spec = spec(1, 1, 0, 1, false, false, 1.0);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let weights = vec![0.0; weight_count];
        let mut biases = vec![0.0; offset_dim];
        biases[2] = 2.0;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (output, _) = dcnv4_forward(&input, shape, &spec, &weights, &biases).unwrap();
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn groups_keep_channel_partitions_and_sampling_uses_zero_padding() {
        let grouped_shape = TensorShape::new(2, 2, 2, 1);
        let mut grouped_spec = spec(2, 1, 0, 1, false, false, 1.0);
        grouped_spec.groups = 2;
        let (weight_count, offset_dim) = parameter_counts(2, &grouped_spec);
        let weights = vec![0.0; weight_count];
        let mut biases = vec![0.0; offset_dim];
        biases[2] = 2.0;
        biases[5] = 3.0;
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (output, _) =
            dcnv4_forward(&input, grouped_shape, &grouped_spec, &weights, &biases).unwrap();
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0]);

        let shape = TensorShape::new(2, 2, 1, 1);
        let spec = spec(1, 1, 0, 1, false, false, 1.0);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let weights = vec![0.0; weight_count];
        let mut biases = vec![0.0; offset_dim];
        biases[0] = -0.5;
        biases[2] = 1.0;
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (output, _) = dcnv4_forward(&input, shape, &spec, &weights, &biases).unwrap();
        assert!((output[0] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn center_removal_and_epsilon_softmax_match_hand_values() {
        let shape = TensorShape::new(3, 3, 1, 1);
        let mut spec = spec(1, 3, 1, 1, true, false, 1.0);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let weights = vec![0.0; weight_count];
        let mut biases = vec![0.0; offset_dim];
        let points = 8;
        for mask in &mut biases[2 * points..3 * points] {
            *mask = 1.0;
        }
        let input = (1..=9).map(|value| value as f32).collect::<Vec<_>>();
        let (output, _) = dcnv4_forward(&input, shape, &spec, &weights, &biases).unwrap();
        assert_eq!(output[4], 40.0);

        spec.softmax = true;
        biases[2 * points..3 * points].fill(0.0);
        let constant = vec![2.0; 9];
        let (output, _) = dcnv4_forward(&constant, shape, &spec, &weights, &biases).unwrap();
        let expected = 16.0 / (points as f32 + 1.0e-6);
        assert!((output[4] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn offset_scale_changes_the_reference_sampling_lattice() {
        let shape = TensorShape::new(3, 3, 1, 1);
        let spec = spec(1, 3, 1, 1, false, false, 0.5);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let weights = vec![0.0; weight_count];
        let mut biases = vec![0.0; offset_dim];
        for mask in &mut biases[18..27] {
            *mask = 1.0;
        }
        let mut input = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                input.push((x * x + y * y) as f32);
            }
        }
        let (output, _) = dcnv4_forward(&input, shape, &spec, &weights, &biases).unwrap();
        assert!((output[4] - 24.0).abs() < 1.0e-5);
    }

    fn objective(
        input: &[f32],
        shape: TensorShape,
        spec: &Dcnv4Spec,
        weights: &[f32],
        biases: &[f32],
        seed: &[f32],
    ) -> f32 {
        let (output, _) = dcnv4_forward(input, shape, spec, weights, biases).unwrap();
        output
            .iter()
            .zip(seed)
            .map(|(value, grad)| value * grad)
            .sum()
    }

    fn check_finite_differences(
        input: &[f32],
        shape: TensorShape,
        spec: &Dcnv4Spec,
        weights: &[f32],
        biases: &[f32],
        seed: &[f32],
        tolerance: f32,
    ) {
        let (_, cache) = dcnv4_forward(input, shape, spec, weights, biases).unwrap();
        let analytic: Dcnv4Gradients =
            dcnv4_backward(input, shape, spec, weights, &cache, seed).unwrap();
        let h = 1.0e-3;
        let check =
            |name: &str, values: &[f32], analytic: &[f32], vary: &dyn Fn(usize, f32) -> f32| {
                assert_eq!(values.len(), analytic.len());
                for index in 0..values.len() {
                    let plus = vary(index, h);
                    let minus = vary(index, -h);
                    let fd = (plus - minus) / (2.0 * h);
                    let error = (fd - analytic[index]).abs();
                    assert!(
                        error <= tolerance + tolerance * fd.abs(),
                        "{name}[{index}]: finite difference={fd}, analytic={}, error={error}",
                        analytic[index]
                    );
                }
            };
        // Separate closures keep each perturbation local to one parameter family.
        check("input", input, &analytic.input, &|index, delta| {
            let mut perturbed = input.to_vec();
            perturbed[index] += delta;
            objective(&perturbed, shape, spec, weights, biases, seed)
        });
        check(
            "offset_weights",
            weights,
            &analytic.offset_weights,
            &|index, delta| {
                let mut perturbed = weights.to_vec();
                perturbed[index] += delta;
                objective(input, shape, spec, &perturbed, biases, seed)
            },
        );
        check(
            "offset_biases",
            biases,
            &analytic.offset_biases,
            &|index, delta| {
                let mut perturbed = biases.to_vec();
                perturbed[index] += delta;
                objective(input, shape, spec, weights, &perturbed, seed)
            },
        );
    }

    #[test]
    fn input_and_offset_parameter_gradients_match_finite_differences() {
        let shape = TensorShape::new(5, 5, 1, 1);
        let spec = spec(1, 3, 1, 1, true, true, 0.75);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let input = (0..25)
            .map(|index| 0.2 + index as f32 * 0.071)
            .collect::<Vec<_>>();
        let weights = (0..weight_count)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.0002)
            .collect::<Vec<_>>();
        let mut biases = vec![0.0; offset_dim];
        for point in 0..8 {
            biases[2 * point] = 0.23 + point as f32 * 0.003;
            biases[2 * point + 1] = 0.31 - point as f32 * 0.002;
            biases[16 + point] = -0.2 + point as f32 * 0.05;
        }
        let seed = (0..25)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.17)
            .collect::<Vec<_>>();
        check_finite_differences(&input, shape, &spec, &weights, &biases, &seed, 5.0e-3);
    }

    #[test]
    fn coarse_offset_upsample_adjoint_matches_finite_differences() {
        let shape = TensorShape::new(4, 4, 1, 1);
        let spec = spec(1, 1, 0, 2, false, false, 0.6);
        let (weight_count, offset_dim) = parameter_counts(1, &spec);
        let mut weights = vec![0.0; weight_count];
        weights[0] = 0.002;
        weights[1] = 0.003;
        weights[2] = 0.001;
        let mut biases = vec![0.0; offset_dim];
        biases[0] = 0.13;
        biases[1] = 0.19;
        biases[2] = 0.8;
        let input = (0..16)
            .map(|index| 0.3 + index as f32 * 0.037)
            .collect::<Vec<_>>();
        let seed = (0..16)
            .map(|index| ((index % 5) as f32 - 2.0) * 0.21)
            .collect::<Vec<_>>();
        check_finite_differences(&input, shape, &spec, &weights, &biases, &seed, 4.0e-3);
    }
}
