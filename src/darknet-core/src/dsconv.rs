//! CPU reference math for the single-template Dynamic Snake Conv (`dsconv`).
//!
//! The forward follows `Jev_laya:src-lib/dsconv_layer.cpp`: a 3x3 tanh offset
//! predictor, center-anchored cumulative snake coordinates, border-clamped
//! bilinear sampling, and a learned 1-D collapse convolution. Backward is the
//! exact derivative of this Rust forward, including the cumulative-offset
//! Jacobian omitted by the source CUDA shortcut.

use crate::error::{Error, Result};
use crate::model::{DsConvSpec, TensorShape};
use crate::runtime::Tensor;

#[derive(Clone, Copy, Debug, PartialEq)]
struct AxisSample {
    low: usize,
    high: usize,
    fraction: f32,
    /// The coordinate is in the interior cell selected by floor(). At the
    /// final border and outside the image, border clamping makes this false.
    derivative_active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SamplePlan {
    y: AxisSample,
    x: AxisSample,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DsConvForwardCache {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    /// Tanh offset maps in `[B,2K,H,W]` layout.
    offsets: Vec<f32>,
    /// Morph-0 (x-snake) and morph-1 (y-snake) plans, each `[B,H,W,K]`.
    x_samples: Vec<SamplePlan>,
    y_samples: Vec<SamplePlan>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DsConvGradients {
    pub input: Vec<f32>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[dsconv]: {}", message.into()))
}

fn checked_count(shape: TensorShape, what: &str) -> Result<usize> {
    shape
        .batch
        .checked_mul(shape.channels)
        .and_then(|count| count.checked_mul(shape.height))
        .and_then(|count| count.checked_mul(shape.width))
        .ok_or_else(|| invalid(format!("{what} element count overflows")))
}

fn offset_index(
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

fn sample_index(
    batch: usize,
    y: usize,
    x: usize,
    k: usize,
    height: usize,
    width: usize,
    kernel: usize,
) -> usize {
    ((batch * height + y) * width + x) * kernel + k
}

fn sampled_channel_index(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    k: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel: usize,
) -> usize {
    (((batch * channels + channel) * height + y) * width + x) * kernel + k
}

fn axis_sample(coordinate: f32, length: usize) -> Result<AxisSample> {
    if !coordinate.is_finite() {
        return Err(invalid("snake sampling coordinate is non-finite"));
    }
    if coordinate < 0.0 {
        return Ok(AxisSample {
            low: 0,
            high: 0,
            fraction: 0.0,
            derivative_active: false,
        });
    }
    let last = length - 1;
    if coordinate >= last as f32 {
        return Ok(AxisSample {
            low: last,
            high: last,
            fraction: 0.0,
            derivative_active: false,
        });
    }
    // The interior coordinate is nonnegative and below `length-1`, hence its
    // floor safely converts to usize. At integer grid points this deliberately
    // chooses the right-hand cell derivative; the forward is continuous but
    // not differentiable there for arbitrary inputs.
    let low = coordinate.floor() as usize;
    Ok(AxisSample {
        low,
        high: low + 1,
        fraction: coordinate - low as f32,
        derivative_active: true,
    })
}

fn sample_plan(y: f32, x: f32, height: usize, width: usize) -> Result<SamplePlan> {
    Ok(SamplePlan {
        y: axis_sample(y, height)?,
        x: axis_sample(x, width)?,
    })
}

fn bilinear(input: &[f32], plan: SamplePlan, base: usize, width: usize) -> f32 {
    let p00 = input[base + plan.y.low * width + plan.x.low];
    let p01 = input[base + plan.y.low * width + plan.x.high];
    let p10 = input[base + plan.y.high * width + plan.x.low];
    let p11 = input[base + plan.y.high * width + plan.x.high];
    let top = p00 + plan.x.fraction * (p01 - p00);
    let bottom = p10 + plan.x.fraction * (p11 - p10);
    top + plan.y.fraction * (bottom - top)
}

fn cumulative_offsets(
    offsets: &[f32],
    batch: usize,
    y_pos: usize,
    x_pos: usize,
    kernel: usize,
    height: usize,
    width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let center = kernel / 2;
    let mut y = vec![0.0; kernel];
    let mut x = vec![0.0; kernel];
    for step in 1..=center {
        let positive = center + step;
        let negative = center - step;
        let y_positive = offset_index(batch, positive, y_pos, x_pos, 2 * kernel, height, width);
        let y_negative = offset_index(batch, negative, y_pos, x_pos, 2 * kernel, height, width);
        let x_positive = offset_index(
            batch,
            kernel + positive,
            y_pos,
            x_pos,
            2 * kernel,
            height,
            width,
        );
        let x_negative = offset_index(
            batch,
            kernel + negative,
            y_pos,
            x_pos,
            2 * kernel,
            height,
            width,
        );
        y[positive] = y[positive - 1] + offsets[y_positive];
        y[negative] = y[negative + 1] + offsets[y_negative];
        x[positive] = x[positive - 1] + offsets[x_positive];
        x[negative] = x[negative + 1] + offsets[x_negative];
    }
    (y, x)
}

fn validate(
    input: &Tensor,
    output_shape: TensorShape,
    spec: &DsConvSpec,
    weights: &[f32],
    biases: &[f32],
    offset_weights: &[f32],
    offset_biases: &[f32],
) -> Result<(usize, usize, usize, usize)> {
    let shape = input.shape;
    if shape.width == 0 || shape.height == 0 || shape.channels == 0 || shape.batch == 0 {
        return Err(invalid("input dimensions must be non-zero"));
    }
    if shape.width > isize::MAX as usize || shape.height > isize::MAX as usize {
        return Err(invalid("spatial dimensions exceed host index range"));
    }
    if spec.filters == 0 || spec.size == 0 || spec.size % 2 == 0 {
        return Err(invalid("filters and odd snake size must be positive"));
    }
    if spec.padding != spec.size / 2 || !spec.extend_scope.is_finite() {
        return Err(invalid("same padding and finite extend_scope are required"));
    }
    if spec.morph > 2 {
        return Err(invalid("morph must be 0, 1, or 2"));
    }
    if spec.templates != 1 {
        return Err(invalid(
            "templates>1 multi-view execution is not yet supported by the Rust CPU port",
        ));
    }
    if spec.morph == 2 && spec.filters < 2 {
        return Err(invalid("morph=2 requires at least two filters"));
    }
    let expected_output = TensorShape::new(shape.width, shape.height, spec.filters, shape.batch);
    checked_count(expected_output, "output")?;
    if output_shape != expected_output {
        return Err(invalid(format!(
            "output shape mismatch: expected {expected_output}, got {output_shape}"
        )));
    }
    if input.data.len() != checked_count(shape, "input")? {
        return Err(invalid("input data length does not match its NCHW shape"));
    }
    let kernel = spec.size;
    let expected_weights = spec
        .filters
        .checked_mul(shape.channels)
        .and_then(|count| count.checked_mul(kernel))
        .ok_or_else(|| invalid("collapse weight count overflows"))?;
    let expected_offset_weights = kernel
        .checked_mul(2)
        .and_then(|count| count.checked_mul(shape.channels))
        .and_then(|count| count.checked_mul(9))
        .ok_or_else(|| invalid("offset weight count overflows"))?;
    if weights.len() != expected_weights
        || biases.len() != spec.filters
        || offset_weights.len() != expected_offset_weights
        || offset_biases.len() != 2 * kernel
    {
        return Err(invalid(format!(
            "parameter lengths do not match filters={}, channels={}, size={kernel}",
            spec.filters, shape.channels
        )));
    }
    let sample_count = shape
        .batch
        .checked_mul(shape.height)
        .and_then(|count| count.checked_mul(shape.width))
        .and_then(|count| count.checked_mul(kernel))
        .ok_or_else(|| invalid("snake sample count overflows"))?;
    let offset_count = shape
        .batch
        .checked_mul(2 * kernel)
        .and_then(|count| count.checked_mul(shape.height))
        .and_then(|count| count.checked_mul(shape.width))
        .ok_or_else(|| invalid("offset map count overflows"))?;
    let sampled_count = sample_count
        .checked_mul(shape.channels)
        .ok_or_else(|| invalid("deformed tensor count overflows"))?;
    let _ = (offset_count, sampled_count);
    Ok((shape.channels, shape.height, shape.width, kernel))
}

pub(crate) fn dsconv_forward(
    input: &Tensor,
    output_shape: TensorShape,
    spec: &DsConvSpec,
    weights: &[f32],
    biases: &[f32],
    offset_weights: &[f32],
    offset_biases: &[f32],
) -> Result<(Tensor, DsConvForwardCache)> {
    let (channels, height, width, kernel) = validate(
        input,
        output_shape,
        spec,
        weights,
        biases,
        offset_weights,
        offset_biases,
    )?;
    let batches = input.shape.batch;
    let mut offsets = vec![0.0; batches * 2 * kernel * height * width];
    for batch in 0..batches {
        for out_channel in 0..2 * kernel {
            for y in 0..height {
                for x in 0..width {
                    let mut value = offset_biases[out_channel];
                    for channel in 0..channels {
                        for ky in 0..3 {
                            let iy = y as isize + ky as isize - 1;
                            if !(0..height as isize).contains(&iy) {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = x as isize + kx as isize - 1;
                                if !(0..width as isize).contains(&ix) {
                                    continue;
                                }
                                let weight_index =
                                    ((out_channel * channels + channel) * 3 + ky) * 3 + kx;
                                value += input.get(batch, channel, iy as usize, ix as usize)
                                    * offset_weights[weight_index];
                            }
                        }
                    }
                    let index = offset_index(batch, out_channel, y, x, 2 * kernel, height, width);
                    offsets[index] = value.tanh();
                }
            }
        }
    }

    let sample_count = batches * height * width * kernel;
    let empty_plan = SamplePlan {
        y: AxisSample {
            low: 0,
            high: 0,
            fraction: 0.0,
            derivative_active: false,
        },
        x: AxisSample {
            low: 0,
            high: 0,
            fraction: 0.0,
            derivative_active: false,
        },
    };
    let mut x_samples = vec![empty_plan; sample_count];
    let mut y_samples = vec![empty_plan; sample_count];
    let center = kernel / 2;
    for batch in 0..batches {
        for y in 0..height {
            for x in 0..width {
                let (y_cumulative, x_cumulative) =
                    cumulative_offsets(&offsets, batch, y, x, kernel, height, width);
                for k in 0..kernel {
                    let index = sample_index(batch, y, x, k, height, width, kernel);
                    if spec.morph == 0 || spec.morph == 2 {
                        x_samples[index] = sample_plan(
                            y as f32 + spec.extend_scope * y_cumulative[k],
                            x as f32 + k as isize as f32 - center as f32,
                            height,
                            width,
                        )?;
                    }
                    if spec.morph == 1 || spec.morph == 2 {
                        y_samples[index] = sample_plan(
                            y as f32 + k as isize as f32 - center as f32,
                            x as f32 + spec.extend_scope * x_cumulative[k],
                            height,
                            width,
                        )?;
                    }
                }
            }
        }
    }

    let mut output = Tensor::zeros(output_shape);
    let (x_filters, y_start) = if spec.morph == 2 {
        (spec.filters / 2, spec.filters / 2)
    } else {
        (spec.filters, 0)
    };
    let branches: &[(usize, usize, usize)] = if spec.morph == 2 {
        &[(0, x_filters, 0), (y_start, spec.filters - y_start, 1)]
    } else {
        &[(0, spec.filters, spec.morph)]
    };
    for &(output_start, output_count, morph) in branches {
        let branch_samples = if morph == 0 { &x_samples } else { &y_samples };
        for output_channel in output_start..output_start + output_count {
            for batch in 0..batches {
                for y in 0..height {
                    for x in 0..width {
                        let mut value = biases[output_channel];
                        for channel in 0..channels {
                            let input_base =
                                input_index(batch, channel, 0, 0, channels, height, width);
                            for k in 0..kernel {
                                let plan = branch_samples
                                    [sample_index(batch, y, x, k, height, width, kernel)];
                                let sample =
                                    bilinear(input.data.as_slice(), plan, input_base, width);
                                let weight_index =
                                    (output_channel * channels + channel) * kernel + k;
                                value += weights[weight_index] * sample;
                            }
                        }
                        output.set(batch, output_channel, y, x, value);
                    }
                }
            }
        }
    }
    Ok((
        output,
        DsConvForwardCache {
            input_shape: input.shape,
            output_shape,
            offsets,
            x_samples,
            y_samples,
        },
    ))
}

fn bilinear_derivatives(input: &[f32], base: usize, width: usize, plan: SamplePlan) -> (f32, f32) {
    let p00 = input[base + plan.y.low * width + plan.x.low];
    let p01 = input[base + plan.y.low * width + plan.x.high];
    let p10 = input[base + plan.y.high * width + plan.x.low];
    let p11 = input[base + plan.y.high * width + plan.x.high];
    let dx = if plan.x.derivative_active {
        (1.0 - plan.y.fraction) * (p01 - p00) + plan.y.fraction * (p11 - p10)
    } else {
        0.0
    };
    let dy = if plan.y.derivative_active {
        (1.0 - plan.x.fraction) * (p10 - p00) + plan.x.fraction * (p11 - p01)
    } else {
        0.0
    };
    (dy, dx)
}

pub(crate) fn dsconv_backward(
    input: &Tensor,
    spec: &DsConvSpec,
    weights: &[f32],
    offset_weights: &[f32],
    cache: &DsConvForwardCache,
    output_gradient: &[f32],
) -> Result<DsConvGradients> {
    if input.shape != cache.input_shape
        || output_gradient.len() != checked_count(cache.output_shape, "output gradient")?
    {
        return Err(invalid("forward cache or output-gradient shape mismatch"));
    }
    let batches = input.shape.batch;
    let channels = input.shape.channels;
    let height = input.shape.height;
    let width = input.shape.width;
    let kernel = spec.size;
    let output_channels = spec.filters;
    if cache.offsets.len() != batches * 2 * kernel * height * width
        || cache.x_samples.len() != batches * height * width * kernel
        || cache.y_samples.len() != batches * height * width * kernel
        || weights.len() != output_channels * channels * kernel
        || offset_weights.len() != 2 * kernel * channels * 9
    {
        return Err(invalid("forward cache or parameter length mismatch"));
    }
    let mut gradients = DsConvGradients {
        input: vec![0.0; input.data.len()],
        weights: vec![0.0; weights.len()],
        biases: vec![0.0; output_channels],
        offset_weights: vec![0.0; offset_weights.len()],
        offset_biases: vec![0.0; 2 * kernel],
    };
    let sampled_count = batches * channels * height * width * kernel;
    let mut sampled_gradients_x = vec![0.0; sampled_count];
    let mut sampled_gradients_y = vec![0.0; sampled_count];
    let (x_filters, y_start) = if spec.morph == 2 {
        (output_channels / 2, output_channels / 2)
    } else {
        (output_channels, 0)
    };
    let branches: &[(usize, usize, usize)] = if spec.morph == 2 {
        &[(0, x_filters, 0), (y_start, output_channels - y_start, 1)]
    } else {
        &[(0, output_channels, spec.morph)]
    };

    for &(output_start, output_count, morph) in branches {
        let (branch_samples, sampled_gradients) = if morph == 0 {
            (&cache.x_samples, &mut sampled_gradients_x)
        } else {
            (&cache.y_samples, &mut sampled_gradients_y)
        };
        for output_channel in output_start..output_start + output_count {
            for batch in 0..batches {
                for y in 0..height {
                    for x in 0..width {
                        let output_index =
                            ((batch * output_channels + output_channel) * height + y) * width + x;
                        let gradient = output_gradient[output_index];
                        gradients.biases[output_channel] += gradient;
                        for channel in 0..channels {
                            let input_base =
                                input_index(batch, channel, 0, 0, channels, height, width);
                            for k in 0..kernel {
                                let plan = branch_samples
                                    [sample_index(batch, y, x, k, height, width, kernel)];
                                let weight_index =
                                    (output_channel * channels + channel) * kernel + k;
                                let sample = bilinear(&input.data, plan, input_base, width);
                                gradients.weights[weight_index] += gradient * sample;
                                let sampled_index = sampled_channel_index(
                                    batch, channel, y, x, k, channels, height, width, kernel,
                                );
                                sampled_gradients[sampled_index] +=
                                    gradient * weights[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }

    let spatial_count = batches * height * width * kernel;
    let mut y_coordinate_gradients = vec![0.0; spatial_count];
    let mut x_coordinate_gradients = vec![0.0; spatial_count];
    for morph in [0, 1] {
        if (morph == 0 && spec.morph == 1) || (morph == 1 && spec.morph == 0) {
            continue;
        }
        let (branch_samples, sampled_gradients) = if morph == 0 {
            (&cache.x_samples, &sampled_gradients_x)
        } else {
            (&cache.y_samples, &sampled_gradients_y)
        };
        for batch in 0..batches {
            for y in 0..height {
                for x in 0..width {
                    for k in 0..kernel {
                        let plan =
                            branch_samples[sample_index(batch, y, x, k, height, width, kernel)];
                        let coordinate_index = sample_index(batch, y, x, k, height, width, kernel);
                        for channel in 0..channels {
                            let input_base =
                                input_index(batch, channel, 0, 0, channels, height, width);
                            let sampled_index = sampled_channel_index(
                                batch, channel, y, x, k, channels, height, width, kernel,
                            );
                            let gradient = sampled_gradients[sampled_index];
                            let p00 = input_base + plan.y.low * width + plan.x.low;
                            let p01 = input_base + plan.y.low * width + plan.x.high;
                            let p10 = input_base + plan.y.high * width + plan.x.low;
                            let p11 = input_base + plan.y.high * width + plan.x.high;
                            let wx0 = 1.0 - plan.x.fraction;
                            let wy0 = 1.0 - plan.y.fraction;
                            gradients.input[p00] += gradient * wy0 * wx0;
                            gradients.input[p01] += gradient * wy0 * plan.x.fraction;
                            gradients.input[p10] += gradient * plan.y.fraction * wx0;
                            gradients.input[p11] += gradient * plan.y.fraction * plan.x.fraction;
                            let (dy, dx) =
                                bilinear_derivatives(&input.data, input_base, width, plan);
                            if morph == 0 {
                                y_coordinate_gradients[coordinate_index] +=
                                    gradient * dy * spec.extend_scope;
                            } else {
                                x_coordinate_gradients[coordinate_index] +=
                                    gradient * dx * spec.extend_scope;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut offset_gradients = vec![0.0; cache.offsets.len()];
    let center = kernel / 2;
    for batch in 0..batches {
        for y in 0..height {
            for x in 0..width {
                let mut y_prefix = 0.0;
                let mut x_prefix = 0.0;
                for k in 0..center {
                    let index = sample_index(batch, y, x, k, height, width, kernel);
                    y_prefix += y_coordinate_gradients[index];
                    x_prefix += x_coordinate_gradients[index];
                    let y_offset_index = offset_index(batch, k, y, x, 2 * kernel, height, width);
                    let x_offset_index =
                        offset_index(batch, kernel + k, y, x, 2 * kernel, height, width);
                    if spec.morph == 0 || spec.morph == 2 {
                        offset_gradients[y_offset_index] = y_prefix;
                    }
                    if spec.morph == 1 || spec.morph == 2 {
                        offset_gradients[x_offset_index] = x_prefix;
                    }
                }
                let mut y_suffix = 0.0;
                let mut x_suffix = 0.0;
                for k in (center + 1..kernel).rev() {
                    let index = sample_index(batch, y, x, k, height, width, kernel);
                    y_suffix += y_coordinate_gradients[index];
                    x_suffix += x_coordinate_gradients[index];
                    let y_offset_index = offset_index(batch, k, y, x, 2 * kernel, height, width);
                    let x_offset_index =
                        offset_index(batch, kernel + k, y, x, 2 * kernel, height, width);
                    if spec.morph == 0 || spec.morph == 2 {
                        offset_gradients[y_offset_index] = y_suffix;
                    }
                    if spec.morph == 1 || spec.morph == 2 {
                        offset_gradients[x_offset_index] = x_suffix;
                    }
                }
            }
        }
    }

    for batch in 0..batches {
        for out_channel in 0..2 * kernel {
            for y in 0..height {
                for x in 0..width {
                    let index = offset_index(batch, out_channel, y, x, 2 * kernel, height, width);
                    let tanh_output = cache.offsets[index];
                    let gradient = offset_gradients[index] * (1.0 - tanh_output * tanh_output);
                    gradients.offset_biases[out_channel] += gradient;
                    for channel in 0..channels {
                        for ky in 0..3 {
                            let iy = y as isize + ky as isize - 1;
                            if !(0..height as isize).contains(&iy) {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = x as isize + kx as isize - 1;
                                if !(0..width as isize).contains(&ix) {
                                    continue;
                                }
                                let weight_index =
                                    ((out_channel * channels + channel) * 3 + ky) * 3 + kx;
                                let input_index = input_index(
                                    batch,
                                    channel,
                                    iy as usize,
                                    ix as usize,
                                    channels,
                                    height,
                                    width,
                                );
                                gradients.offset_weights[weight_index] +=
                                    gradient * input.data[input_index];
                                gradients.input[input_index] +=
                                    gradient * offset_weights[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(gradients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Activation;

    fn spec(morph: usize, filters: usize, size: usize) -> DsConvSpec {
        DsConvSpec {
            filters,
            size,
            padding: size / 2,
            extend_scope: 1.0,
            morph,
            templates: 1,
            drop_prob: 0.0,
            activation: Activation::Linear,
        }
    }

    fn input(batch: usize, channels: usize, height: usize, width: usize) -> Tensor {
        Tensor::from_data(
            TensorShape::new(width, height, channels, batch),
            (0..batch * channels * height * width)
                .map(|index| 0.17 * (index as f32 + 1.0) - 0.4)
                .collect(),
        )
        .unwrap()
    }

    fn zeros_for(input: &Tensor, spec: &DsConvSpec) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.0; spec.filters * input.shape.channels * spec.size],
            vec![0.0; spec.filters],
            vec![0.0; 2 * spec.size * input.shape.channels * 9],
            vec![0.0; 2 * spec.size],
        )
    }

    #[test]
    fn dsconv_source_nchw_axes_match_hand_computation() {
        let input = Tensor::from_data(
            TensorShape::new(3, 2, 1, 1),
            vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        )
        .unwrap();
        for morph in [0, 1] {
            let spec = spec(morph, 1, 3);
            let (mut weights, _, offset_weights, offset_biases) = zeros_for(&input, &spec);
            weights.copy_from_slice(&[1.0, 10.0, 100.0]);
            let (output, _) = dsconv_forward(
                &input,
                TensorShape::new(3, 2, 1, 1),
                &spec,
                &weights,
                &[0.0],
                &offset_weights,
                &offset_biases,
            )
            .unwrap();
            if morph == 0 {
                assert_eq!(
                    output.data,
                    vec![211.0, 421.0, 442.0, 1688.0, 3368.0, 3536.0]
                );
            } else {
                assert_eq!(
                    output.data,
                    vec![811.0, 1622.0, 3244.0, 881.0, 1762.0, 3524.0]
                );
            }
        }

        let spec = spec(2, 3, 1);
        let (mut weights, _, offset_weights, offset_biases) = zeros_for(&input, &spec);
        weights.copy_from_slice(&[2.0, 3.0, 5.0]);
        let (output, _) = dsconv_forward(
            &input,
            TensorShape::new(3, 2, 3, 1),
            &spec,
            &weights,
            &[0.0; 3],
            &offset_weights,
            &offset_biases,
        )
        .unwrap();
        assert_eq!(
            output.data,
            vec![
                2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 5.0, 10.0, 20.0,
                40.0, 80.0, 160.0
            ]
        );
    }

    #[test]
    fn dsconv_backward_uses_right_cell_derivative_at_integer_and_zero_at_border() {
        let input =
            Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 4.0, 7.0, 20.0]).unwrap();
        let spec = spec(0, 1, 3);
        let (mut weights, biases, offset_weights, offset_biases) = zeros_for(&input, &spec);
        weights.copy_from_slice(&[0.0, 0.0, 1.0]);
        let (_, cache) = dsconv_forward(
            &input,
            TensorShape::new(2, 2, 1, 1),
            &spec,
            &weights,
            &biases,
            &offset_weights,
            &offset_biases,
        )
        .unwrap();
        let mut gradient = vec![0.0; 4];
        gradient[0] = 1.0;
        let interior =
            dsconv_backward(&input, &spec, &weights, &offset_weights, &cache, &gradient).unwrap();
        assert_eq!(interior.offset_biases[2], 16.0);

        gradient.fill(0.0);
        gradient[2] = 1.0;
        let border =
            dsconv_backward(&input, &spec, &weights, &offset_weights, &cache, &gradient).unwrap();
        assert_eq!(border.offset_biases[2], 0.0);
    }

    #[test]
    fn dsconv_forward_clamps_bilinear_samples_to_the_border() {
        let input = Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![3.0, 11.0]).unwrap();
        let spec = spec(0, 1, 3);
        let (mut weights, _, offset_weights, offset_biases) = zeros_for(&input, &spec);
        weights.copy_from_slice(&[1.0, 1.0, 1.0]);
        // The fixed x positions sample border-clamped neighbors at both image edges.
        let (output, _) = dsconv_forward(
            &input,
            TensorShape::new(2, 1, 1, 1),
            &spec,
            &weights,
            &[0.0],
            &offset_weights,
            &offset_biases,
        )
        .unwrap();
        assert_eq!(output.data, vec![17.0, 25.0]);
    }

    fn objective(
        input: &Tensor,
        spec: &DsConvSpec,
        weights: &[f32],
        biases: &[f32],
        offset_weights: &[f32],
        offset_biases: &[f32],
        seed: &[f32],
    ) -> f32 {
        let (output, _) = dsconv_forward(
            input,
            TensorShape::new(
                input.shape.width,
                input.shape.height,
                spec.filters,
                input.shape.batch,
            ),
            spec,
            weights,
            biases,
            offset_weights,
            offset_biases,
        )
        .unwrap();
        output.data.iter().zip(seed).map(|(x, dy)| x * dy).sum()
    }

    #[test]
    fn dsconv_backward_matches_finite_differences_for_all_morphs() {
        for morph in [0, 1, 2] {
            let input = input(1, 1, 4, 5);
            let spec = spec(morph, if morph == 2 { 3 } else { 2 }, 3);
            let mut weights = (0..spec.filters * spec.size)
                .map(|index| 0.09 + index as f32 * 0.013)
                .collect::<Vec<_>>();
            let mut biases = (0..spec.filters)
                .map(|index| -0.03 + index as f32 * 0.02)
                .collect::<Vec<_>>();
            // Constant fractional offsets keep the finite-difference probe away
            // from grid kinks while leaving input gradients linear in `input`.
            let mut offset_weights = vec![0.0; 2 * spec.size * 9];
            let mut offset_biases = vec![0.0; 2 * spec.size];
            offset_biases[0] = -0.19;
            offset_biases[2] = 0.17;
            offset_biases[spec.size] = -0.16;
            offset_biases[spec.size + 2] = 0.14;
            let output_shape =
                TensorShape::new(input.shape.width, input.shape.height, spec.filters, 1);
            let (output, cache) = dsconv_forward(
                &input,
                output_shape,
                &spec,
                &weights,
                &biases,
                &offset_weights,
                &offset_biases,
            )
            .unwrap();
            let seed = (0..output.len())
                .map(|index| 0.03 + (index % 7) as f32 * 0.011)
                .collect::<Vec<_>>();
            let gradients =
                dsconv_backward(&input, &spec, &weights, &offset_weights, &cache, &seed).unwrap();
            let epsilon = 1.0e-3;
            let check = |kind: &str, analytic: f32, plus: f32, minus: f32| {
                let numeric = (plus - minus) / (2.0 * epsilon);
                assert!(
                    (analytic - numeric).abs() < 3.0e-2 * (1.0 + numeric.abs()),
                    "morph={morph} {kind}: analytic={analytic}, numeric={numeric}"
                );
            };
            for index in [0, input.data.len() / 2, input.data.len() - 1] {
                let mut plus = input.clone();
                let mut minus = input.clone();
                plus.data[index] += epsilon;
                minus.data[index] -= epsilon;
                check(
                    "input",
                    gradients.input[index],
                    objective(
                        &plus,
                        &spec,
                        &weights,
                        &biases,
                        &offset_weights,
                        &offset_biases,
                        &seed,
                    ),
                    objective(
                        &minus,
                        &spec,
                        &weights,
                        &biases,
                        &offset_weights,
                        &offset_biases,
                        &seed,
                    ),
                );
            }
            for index in [0, weights.len() / 2, weights.len() - 1] {
                let old = weights[index];
                weights[index] = old + epsilon;
                let plus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                weights[index] = old - epsilon;
                let minus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                weights[index] = old;
                check("collapse weight", gradients.weights[index], plus, minus);
            }
            for index in [0, biases.len() - 1] {
                let old = biases[index];
                biases[index] = old + epsilon;
                let plus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                biases[index] = old - epsilon;
                let minus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                biases[index] = old;
                check("collapse bias", gradients.biases[index], plus, minus);
            }
            for index in [0, offset_weights.len() / 2, offset_weights.len() - 1] {
                let old = offset_weights[index];
                offset_weights[index] = old + epsilon;
                let plus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                offset_weights[index] = old - epsilon;
                let minus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                offset_weights[index] = old;
                check(
                    "offset weight",
                    gradients.offset_weights[index],
                    plus,
                    minus,
                );
            }
            for index in [0, offset_biases.len() - 1] {
                let old = offset_biases[index];
                offset_biases[index] = old + epsilon;
                let plus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                offset_biases[index] = old - epsilon;
                let minus = objective(
                    &input,
                    &spec,
                    &weights,
                    &biases,
                    &offset_weights,
                    &offset_biases,
                    &seed,
                );
                offset_biases[index] = old;
                check("offset bias", gradients.offset_biases[index], plus, minus);
            }
        }
    }
}
