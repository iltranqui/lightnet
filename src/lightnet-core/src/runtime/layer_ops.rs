use super::*;

/// Elements per softmax group in one batch item.
pub(crate) fn softmax_group_size(shape: TensorShape, groups: usize) -> Result<usize> {
    let groups = groups.max(1);
    let per_batch = shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    Ok(per_batch / groups)
}

pub(super) fn forward_softmax(input: &Tensor, groups: usize) -> Result<Tensor> {
    let groups = groups.max(1);
    let per_batch = input.shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    let group_size = per_batch / groups;
    let mut output = Tensor::zeros(input.shape);
    for b in 0..input.shape.batch {
        let base = b * per_batch;
        for group in 0..groups {
            let start = base + group * group_size;
            let end = start + group_size;
            let max_value = input.data[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for (dst, value) in output.data[start..end]
                .iter_mut()
                .zip(&input.data[start..end])
            {
                *dst = (*value - max_value).exp();
                sum += *dst;
            }
            if sum > 0.0 {
                for value in &mut output.data[start..end] {
                    *value /= sum;
                }
            }
        }
    }
    Ok(output)
}

/// C++ `forward_region_layer` transposes `[channel][spatial]` into
/// `[spatial][anchor][entry]`, then activates objectness and optional classes.
pub(super) fn forward_region_layer(
    spec: &RegionSpec,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let spatial = input.shape.width * input.shape.height;
    let entries = spec.coords + spec.classes + 1;
    let channels = entries * spec.num;
    if input.shape.channels != channels || output_shape != input.shape {
        return Err(Error::InvalidConfig(
            "region tensor shape is inconsistent with its specification".to_owned(),
        ));
    }
    let mut output = Tensor::zeros(output_shape);
    let plane = spatial * channels;
    for batch in 0..input.shape.batch {
        let base = batch * plane;
        for channel in 0..channels {
            for location in 0..spatial {
                output.data[base + location * channels + channel] =
                    input.data[base + channel * spatial + location];
            }
        }
        for location in 0..spatial {
            for anchor in 0..spec.num {
                let offset = base + location * channels + anchor * entries;
                output.data[offset + 4] = 1.0 / (1.0 + (-output.data[offset + 4]).exp());
                if spec.softmax {
                    let class_start = offset + 5;
                    let max_value = output.data[class_start..class_start + spec.classes]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for value in &mut output.data[class_start..class_start + spec.classes] {
                        *value = (*value - max_value).exp();
                        sum += *value;
                    }
                    if sum > 0.0 {
                        for value in &mut output.data[class_start..class_start + spec.classes] {
                            *value /= sum;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(super) fn backward_region_layer(
    spec: &RegionSpec,
    output: &Tensor,
    output_grad: &Tensor,
) -> Result<Tensor> {
    if output.shape != output_grad.shape {
        return Err(Error::InvalidConfig(
            "region output/gradient shape mismatch".to_owned(),
        ));
    }
    let spatial = output.shape.width * output.shape.height;
    let entries = spec.coords + spec.classes + 1;
    let channels = entries * spec.num;
    if output.shape.channels != channels {
        return Err(Error::InvalidConfig(
            "region tensor shape is inconsistent with its specification".to_owned(),
        ));
    }
    let plane = spatial * channels;
    let mut input_grad = Tensor::zeros(output.shape);
    for batch in 0..output.shape.batch {
        let base = batch * plane;
        for location in 0..spatial {
            for anchor in 0..spec.num {
                let offset = base + location * channels + anchor * entries;
                let mut adjusted = output_grad.data[offset..offset + entries].to_vec();
                let objectness = output.data[offset + 4];
                adjusted[4] *= objectness * (1.0 - objectness);
                if spec.softmax {
                    let class_start = offset + 5;
                    let dot = output.data[class_start..class_start + spec.classes]
                        .iter()
                        .zip(&output_grad.data[class_start..class_start + spec.classes])
                        .map(|(prob, grad)| prob * grad)
                        .sum::<f32>();
                    for class in 0..spec.classes {
                        let prob = output.data[class_start + class];
                        adjusted[5 + class] = prob * (output_grad.data[class_start + class] - dot);
                    }
                }
                for entry in 0..entries {
                    let channel = anchor * entries + entry;
                    input_grad.data[base + channel * spatial + location] = adjusted[entry];
                }
            }
        }
    }
    Ok(input_grad)
}

pub(super) fn check_dropout_probability(probability: f32) -> Result<()> {
    if !(0.0..1.0).contains(&probability) {
        return Err(Error::InvalidConfig(format!(
            "dropout probability must be in [0, 1), got {probability}"
        )));
    }
    Ok(())
}

pub(super) fn forward_dropout(
    input: &Tensor,
    probability: f32,
    training: bool,
    rng_state: &mut u64,
) -> Result<(Tensor, Option<Tensor>)> {
    check_dropout_probability(probability)?;
    if !training || probability == 0.0 {
        return Ok((input.clone(), None));
    }

    // One seed per dropout layer, then a counter-hash per element
    // (`lightnet_cuda::dropout_unit`), so a GPU backend reproduces this mask
    // exactly from the same seed.
    let seed = advance_rng(rng_state);
    let scale = 1.0 / (1.0 - probability);
    let mut output = Tensor::zeros(input.shape);
    let mut mask = Tensor::zeros(input.shape);
    for (index, ((out, mask_value), value)) in output
        .data
        .iter_mut()
        .zip(mask.data.iter_mut())
        .zip(&input.data)
        .enumerate()
    {
        let multiplier = if dropout_unit(seed, index as u64) < probability {
            0.0
        } else {
            scale
        };
        *mask_value = multiplier;
        *out = *value * multiplier;
    }
    Ok((output, Some(mask)))
}

/// C++ `forward_dropout_layer_gpu` DropBlock schedule: during the first 85% of
/// `max_batches` the probability and block size ramp up linearly.  Returns the
/// per-cell block probability `p * ramp / block_size^2` and the square block
/// side (`min` of the clamped width/height, >= 1).
pub(super) fn dropblock_plan(
    spec: &DropblockSpec,
    probability: f32,
    shape: TensorShape,
    iteration: usize,
    max_batches: usize,
) -> (f32, usize) {
    let ramp_end = max_batches as f64 * 0.85;
    let multiplier = if (iteration as f64) < ramp_end {
        iteration as f32 / ramp_end as f32
    } else {
        1.0
    };
    let (mut block_width, mut block_height) = (
        (spec.size_abs as f32 * multiplier) as i64,
        (spec.size_abs as f32 * multiplier) as i64,
    );
    if spec.size_rel != 0.0 {
        block_width = (spec.size_rel * shape.width as f32 * multiplier) as i64;
        block_height = (spec.size_rel * shape.height as f32 * multiplier) as i64;
    }
    let block_width = block_width.clamp(1, shape.width.max(1) as i64);
    let block_height = block_height.clamp(1, shape.height.max(1) as i64);
    let block_size = block_width.min(block_height) as usize;
    (
        probability * multiplier / (block_size * block_size) as f32,
        block_size,
    )
}

/// Per-iteration DropBlock seed: the layer seed mixed with the iteration so
/// every step draws new blocks.
pub(super) fn dropblock_seed(layer_seed: u64, iteration: usize) -> u64 {
    layer_seed ^ (iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// DropBlock forward (see `lightnet_cuda::host_ops::dropblock`); like dropout
/// the returned mask is the per-element multiplier used by the backward.
pub(super) fn forward_dropblock(
    input: &Tensor,
    spec: &DropblockSpec,
    probability: f32,
    training: bool,
    rng_state: &mut u64,
    iteration: usize,
    max_batches: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    check_dropout_probability(probability)?;
    if !training || probability == 0.0 {
        return Ok((input.clone(), None));
    }
    let seed = advance_rng(rng_state);
    let (block_prob, block_size) =
        dropblock_plan(spec, probability, input.shape, iteration, max_batches);
    if block_prob <= 0.0 {
        return Ok((input.clone(), None));
    }
    let shape = tensor_backend_shape(input.shape);
    let (out, mask) = lightnet_cuda::host_ops::dropblock(
        &input.data,
        &shape,
        block_prob,
        block_size,
        dropblock_seed(seed, iteration),
    );
    Ok((
        Tensor {
            shape: input.shape,
            data: out,
        },
        Some(Tensor {
            shape: input.shape,
            data: mask,
        }),
    ))
}

pub(super) fn advance_rng(state: &mut u64) -> u64 {
    // A local generator keeps the core crate dependency-free.  Re-seeding per
    // forward call is intentional: inference remains deterministic and tests
    // can reproduce a training pass exactly.
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

pub(super) fn next_random_unit(state: &mut u64) -> f32 {
    ((advance_rng(state) >> 40) as u32) as f32 / (u32::MAX as f32 + 1.0)
}

pub(super) fn crop_sample(input: &Tensor, batch: usize, channel: usize, x: isize, y: isize) -> f32 {
    if x < 0 || y < 0 || x >= input.shape.width as isize || y >= input.shape.height as isize {
        0.0
    } else {
        input.get(batch, channel, y as usize, x as usize)
    }
}

#[derive(Clone, Copy)]
pub(super) struct CropLevelAdjustment {
    pub(super) saturation: f32,
    pub(super) exposure: f32,
    pub(super) rgb_shift: [f32; 3],
    pub(super) scale: f32,
    pub(super) translate: f32,
    pub(super) apply_hsv: bool,
}

pub(super) fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let delta = max - min;
    if max == 0.0 {
        return [-1.0, 0.0, 0.0];
    }
    let saturation = delta / max;
    let mut hue = if delta == 0.0 {
        0.0
    } else if rgb[0] == max {
        (rgb[1] - rgb[2]) / delta
    } else if rgb[1] == max {
        2.0 + (rgb[2] - rgb[0]) / delta
    } else {
        4.0 + (rgb[0] - rgb[1]) / delta
    };
    if hue < 0.0 {
        hue += 6.0;
    }
    [hue, saturation, max]
}

pub(super) fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let [hue, saturation, value] = hsv;
    let rgb = if saturation == 0.0 {
        [value, value, value]
    } else {
        let index = hue.floor() as i32;
        let fractional = hue - index as f32;
        let p = value * (1.0 - saturation);
        let q = value * (1.0 - saturation * fractional);
        let t = value * (1.0 - saturation * (1.0 - fractional));
        match index {
            0 => [value, t, p],
            1 => [q, value, p],
            2 => [p, value, t],
            3 => [p, q, value],
            4 => [t, p, value],
            _ => [value, p, q],
        }
    };
    rgb.map(|value| value.clamp(0.0, 1.0))
}

pub(super) fn crop_adjusted_sample(
    input: &Tensor,
    batch: usize,
    channel: usize,
    x: isize,
    y: isize,
    adjustment: CropLevelAdjustment,
) -> f32 {
    if x < 0 || y < 0 || x >= input.shape.width as isize || y >= input.shape.height as isize {
        return 0.0;
    }
    let mut value = crop_sample(input, batch, channel, x, y);
    if channel < 3 && input.shape.channels >= 3 {
        let mut rgb = [
            crop_sample(input, batch, 0, x, y),
            crop_sample(input, batch, 1, x, y),
            crop_sample(input, batch, 2, x, y),
        ];
        if adjustment.apply_hsv {
            let mut hsv = rgb_to_hsv(rgb);
            hsv[1] *= adjustment.saturation;
            hsv[2] *= adjustment.exposure;
            rgb = hsv_to_rgb(hsv);
        }
        value = rgb[channel];
    }
    let shift = if channel < 3 {
        adjustment.rgb_shift[channel]
    } else {
        0.0
    };
    value * adjustment.scale + adjustment.translate + shift
}

pub(super) fn crop_bilinear(
    input: &Tensor,
    batch: usize,
    channel: usize,
    x: f32,
    y: f32,
    adjustment: CropLevelAdjustment,
) -> f32 {
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let dx = x - ix as f32;
    let dy = y - iy as f32;
    (1.0 - dy) * (1.0 - dx) * crop_adjusted_sample(input, batch, channel, ix, iy, adjustment)
        + dy * (1.0 - dx) * crop_adjusted_sample(input, batch, channel, ix, iy + 1, adjustment)
        + (1.0 - dy) * dx * crop_adjusted_sample(input, batch, channel, ix + 1, iy, adjustment)
        + dy * dx * crop_adjusted_sample(input, batch, channel, ix + 1, iy + 1, adjustment)
}

/// Plans crop coordinates and color-level adjustments without touching tensor
/// data. The GPU executor uploads this metadata so image samples themselves
/// stay resident on the device.
pub(super) fn plan_crop_forward(
    spec: &CropSpec,
    input_shape: TensorShape,
    output_shape: TensorShape,
    training: bool,
    rng_state: &mut u64,
) -> Result<(CropForwardCache, Vec<CropLevelAdjustment>)> {
    if output_shape.batch != input_shape.batch || output_shape.channels != input_shape.channels {
        return Err(Error::InvalidConfig(format!(
            "crop output shape must preserve batch/channels: input {}, output {}",
            input_shape, output_shape
        )));
    }
    let samples = output_shape.batch * output_shape.width * output_shape.height;
    let mut cache = CropForwardCache {
        source_x: vec![0.0; samples],
        source_y: vec![0.0; samples],
        input_scales: vec![0.0; input_shape.batch],
        input_offsets: vec![0.0; input_shape.batch * input_shape.channels],
    };
    let mut adjustments = Vec::with_capacity(input_shape.batch);
    let center_x = input_shape.width as f32 / 2.0;
    let center_y = input_shape.height as f32 / 2.0;

    for batch in 0..input_shape.batch {
        let random: [f32; 8] = std::array::from_fn(|_| next_random_unit(rng_state));
        let adjustment = if training {
            let mut saturation = random[0] * (spec.saturation - 1.0) + 1.0;
            if random[1] > 0.5 {
                saturation = 1.0 / saturation;
            }
            let mut exposure = random[2] * (spec.exposure - 1.0) + 1.0;
            if random[3] > 0.5 {
                exposure = 1.0 / exposure;
            }
            CropLevelAdjustment {
                saturation,
                exposure,
                rgb_shift: [
                    (random[0] - 0.5) * spec.shift,
                    (random[1] - 0.5) * spec.shift,
                    (random[2] - 0.5) * spec.shift,
                ],
                scale: if spec.noadjust { 1.0 } else { 2.0 },
                translate: if spec.noadjust { 0.0 } else { -1.0 },
                apply_hsv: true,
            }
        } else {
            CropLevelAdjustment {
                saturation: 1.0,
                exposure: 1.0,
                rgb_shift: [0.0; 3],
                scale: if spec.noadjust { 1.0 } else { 2.0 },
                translate: if spec.noadjust { 0.0 } else { -1.0 },
                apply_hsv: false,
            }
        };
        cache.input_scales[batch] = adjustment.scale;
        for channel in 0..input_shape.channels {
            let shift = if channel < 3 {
                adjustment.rgb_shift[channel]
            } else {
                0.0
            };
            cache.input_offsets[batch * input_shape.channels + channel] =
                adjustment.translate + shift;
        }
        let (dw, dh, flip, angle) = if training {
            let dw = (input_shape.width.saturating_sub(spec.width)) as f32 * random[4];
            let dh = (input_shape.height.saturating_sub(spec.height)) as f32 * random[5];
            let flip = spec.flip && random[6] > 0.5;
            let angle = (2.0 * random[7] - 1.0) * spec.angle.to_radians();
            (dw, dh, flip, angle)
        } else {
            (
                (input_shape.width.saturating_sub(spec.width)) as f32 / 2.0,
                (input_shape.height.saturating_sub(spec.height)) as f32 / 2.0,
                false,
                0.0,
            )
        };
        let (sin, cos) = angle.sin_cos();
        for oy in 0..output_shape.height {
            for ox in 0..output_shape.width {
                let x = if flip {
                    input_shape.width as f32 - dw - ox as f32 - 1.0
                } else {
                    ox as f32 + dw
                };
                let y = oy as f32 + dh;
                let source_x = cos * (x - center_x) - sin * (y - center_y) + center_x;
                let source_y = sin * (x - center_x) + cos * (y - center_y) + center_y;
                let index = (batch * output_shape.height + oy) * output_shape.width + ox;
                cache.source_x[index] = source_x;
                cache.source_y[index] = source_y;
            }
        }
        adjustments.push(adjustment);
    }
    Ok((cache, adjustments))
}

pub(super) fn forward_crop(
    spec: &CropSpec,
    input: &Tensor,
    output_shape: TensorShape,
    training: bool,
    rng_state: &mut u64,
) -> Result<(Tensor, CropForwardCache)> {
    let (cache, adjustments) =
        plan_crop_forward(spec, input.shape, output_shape, training, rng_state)?;
    let mut output = Tensor::zeros(output_shape);
    for batch in 0..output_shape.batch {
        for channel in 0..output_shape.channels {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let index = (batch * output_shape.height + oy) * output_shape.width + ox;
                    output.set(
                        batch,
                        channel,
                        oy,
                        ox,
                        crop_bilinear(
                            input,
                            batch,
                            channel,
                            cache.source_x[index],
                            cache.source_y[index],
                            adjustments[batch],
                        ),
                    );
                }
            }
        }
    }
    Ok((output, cache))
}

pub(super) fn accumulate_crop_gradient(
    gradient: &mut Tensor,
    batch: usize,
    channel: usize,
    x: isize,
    y: isize,
    value: f32,
) {
    if x >= 0 && y >= 0 && x < gradient.shape.width as isize && y < gradient.shape.height as isize {
        let previous = gradient.get(batch, channel, y as usize, x as usize);
        gradient.set(batch, channel, y as usize, x as usize, previous + value);
    }
}

pub(super) fn backward_crop(
    cache: &CropForwardCache,
    input_shape: TensorShape,
    output_gradient: &Tensor,
) -> Result<Tensor> {
    let expected =
        output_gradient.shape.batch * output_gradient.shape.width * output_gradient.shape.height;
    if cache.source_x.len() != expected
        || cache.source_y.len() != expected
        || cache.input_scales.len() != output_gradient.shape.batch
        || cache.input_offsets.len() != output_gradient.shape.batch * output_gradient.shape.channels
    {
        return Err(Error::InvalidConfig(
            "crop cache shape does not match output gradient".to_owned(),
        ));
    }
    let mut input_gradient = Tensor::zeros(input_shape);
    for batch in 0..output_gradient.shape.batch {
        for channel in 0..output_gradient.shape.channels {
            for oy in 0..output_gradient.shape.height {
                for ox in 0..output_gradient.shape.width {
                    let index = (batch * output_gradient.shape.height + oy)
                        * output_gradient.shape.width
                        + ox;
                    let x = cache.source_x[index];
                    let y = cache.source_y[index];
                    let ix = x.floor() as isize;
                    let iy = y.floor() as isize;
                    let dx = x - ix as f32;
                    let dy = y - iy as f32;
                    let grad =
                        output_gradient.get(batch, channel, oy, ox) * cache.input_scales[batch];
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix,
                        iy,
                        grad * (1.0 - dy) * (1.0 - dx),
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix,
                        iy + 1,
                        grad * dy * (1.0 - dx),
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix + 1,
                        iy,
                        grad * (1.0 - dy) * dx,
                    );
                    accumulate_crop_gradient(
                        &mut input_gradient,
                        batch,
                        channel,
                        ix + 1,
                        iy + 1,
                        grad * dy * dx,
                    );
                }
            }
        }
    }
    Ok(input_gradient)
}

pub(super) fn forward_scale_channels(
    spec: &crate::model::ScaleChannelsSpec,
    input: &Tensor,
    source: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    if source.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "scale_channels source/output mismatch: {} vs {}",
            source.shape, output_shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for c in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let scale = if spec.scale_wh {
                        input.get(b, 0, y, x)
                    } else {
                        input.get(b, c, 0, 0)
                    };
                    let pre_activation = scale * source.get(b, c, y, x);
                    activation_input.set(b, c, y, x, pre_activation);
                    output.set(
                        b,
                        c,
                        y,
                        x,
                        activate(spec.activation.clone(), pre_activation),
                    );
                }
            }
        }
    }
    Ok((output, Some(activation_input)))
}

pub(super) fn forward_sam(
    spec: &crate::model::SamSpec,
    input: &Tensor,
    source: &Tensor,
    output_shape: TensorShape,
) -> Result<(Tensor, Option<Tensor>)> {
    if input.shape != output_shape || source.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "sam shape mismatch: input {}, source {}, output {}",
            input.shape, source.shape, output_shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    let mut activation_input = Tensor::zeros(output_shape);
    for ((out, pre_activation), (input_value, source_value)) in output
        .data
        .iter_mut()
        .zip(&mut activation_input.data)
        .zip(input.data.iter().zip(&source.data))
    {
        *pre_activation = *input_value * *source_value;
        *out = activate(spec.activation.clone(), *pre_activation);
    }
    Ok((output, Some(activation_input)))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn backward_connected(
    spec: &ConnectedSpec,
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    output: &Tensor,
    activation_input: Option<&Tensor>,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, ConnectedParams)> {
    // Return parameter-shaped gradients instead of mutating the model.  The
    // caller can use this for a standalone backward pass or merge it into
    // persistent Darknet-style train buffers for optimizer updates.
    let inputs = input.shape.element_count();
    if output.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "connected output shape mismatch: expected {}, got {}",
            output_shape, output.shape
        )));
    }
    let expected_weights = spec.output * inputs;
    if params.biases.len() != spec.output || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "connected parameter shape mismatch: expected {} biases and {} weights, got {} and {}",
            spec.output,
            expected_weights,
            params.biases.len(),
            params.weights.len()
        )));
    }
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "connected",
        activation_input,
    )?;

    let needed = spec.output + expected_weights;
    if workspace.len() < needed {
        workspace.resize(needed, 0.0);
    }
    workspace[..needed].fill(0.0);
    let (bias_grad, weight_grad) = workspace[..needed].split_at_mut(spec.output);
    let mut input_grad = Tensor::zeros(input.shape);
    let mut batch_norm_grad = params
        .batch_norm
        .as_ref()
        .map(|batch_norm| BatchNormParams {
            biases: vec![0.0; batch_norm.biases.len()],
            scales: vec![0.0; batch_norm.scales.len()],
            rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
            rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
        });

    for b in 0..input.shape.batch {
        let input_base = b * inputs;
        let output_base = b * spec.output;
        for (out_idx, bias_grad_slot) in bias_grad.iter_mut().enumerate().take(spec.output) {
            let mut grad_value = delta.data[output_base + out_idx];
            if let Some(batch_norm) = params.batch_norm.as_ref() {
                let raw_value = connected_raw_value(params, input, b, out_idx);
                let inv_std =
                    1.0 / (batch_norm.rolling_variance[out_idx] + BATCH_NORM_EPSILON).sqrt();
                let normalized = (raw_value - batch_norm.rolling_mean[out_idx]) * inv_std;
                *bias_grad_slot += grad_value;
                if let Some(batch_norm_grad) = batch_norm_grad.as_mut() {
                    batch_norm_grad.scales[out_idx] += grad_value * normalized;
                }
                grad_value *= batch_norm.scales[out_idx] * inv_std;
            } else {
                *bias_grad_slot += grad_value;
            }

            let weight_base = out_idx * inputs;
            for input_idx in 0..inputs {
                let input_value = input.data[input_base + input_idx];
                weight_grad[weight_base + input_idx] += input_value * grad_value;
                input_grad.data[input_base + input_idx] +=
                    params.weights[weight_base + input_idx] * grad_value;
            }
        }
    }

    Ok((
        input_grad,
        ConnectedParams {
            biases: bias_grad.to_vec(),
            batch_norm: batch_norm_grad,
            weights: weight_grad.to_vec(),
        },
    ))
}

pub(super) fn connected_raw_value(
    params: &ConnectedParams,
    input: &Tensor,
    batch: usize,
    output_index: usize,
) -> f32 {
    let inputs = input.shape.element_count();
    let input_base = batch * inputs;
    let weight_base = output_index * inputs;
    input.data[input_base..input_base + inputs]
        .iter()
        .zip(&params.weights[weight_base..weight_base + inputs])
        .map(|(input, weight)| input * weight)
        .sum()
}

pub(super) fn raw_connected_output(
    params: &ConnectedParams,
    input: &Tensor,
    output_shape: TensorShape,
) -> Tensor {
    let mut output = Tensor::zeros(output_shape);
    for batch in 0..input.shape.batch {
        for output_index in 0..output_shape.channels {
            output.data[batch * output_shape.channels + output_index] =
                connected_raw_value(params, input, batch, output_index);
        }
    }
    output
}

pub(super) fn backward_softmax(output: &Tensor, output_gradient: &Tensor, groups: usize) -> Result<Tensor> {
    if output.shape != output_gradient.shape {
        return Err(Error::InvalidConfig(format!(
            "softmax backward gradient shape mismatch: expected {}, got {}",
            output.shape, output_gradient.shape
        )));
    }
    let groups = groups.max(1);
    let per_batch = output.shape.element_count();
    if !per_batch.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "softmax input size {per_batch} is not divisible by groups {groups}"
        )));
    }
    let group_size = per_batch / groups;
    let mut input_grad = Tensor::zeros(output.shape);
    for b in 0..output.shape.batch {
        let base = b * per_batch;
        for group in 0..groups {
            let start = base + group * group_size;
            let end = start + group_size;
            let dot = output.data[start..end]
                .iter()
                .zip(&output_gradient.data[start..end])
                .map(|(probability, gradient)| probability * gradient)
                .sum::<f32>();
            for index in start..end {
                input_grad.data[index] = output.data[index] * (output_gradient.data[index] - dot);
            }
        }
    }
    Ok(input_grad)
}

pub(super) fn backward_dropout(output_gradient: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    if let Some(mask) = mask {
        if mask.shape != output_gradient.shape {
            return Err(Error::InvalidConfig(format!(
                "dropout mask shape mismatch: expected {}, got {}",
                output_gradient.shape, mask.shape
            )));
        }
        let mut input_grad = Tensor::zeros(output_gradient.shape);
        for ((dst, gradient), multiplier) in input_grad
            .data
            .iter_mut()
            .zip(&output_gradient.data)
            .zip(&mask.data)
        {
            *dst = *gradient * *multiplier;
        }
        Ok(input_grad)
    } else {
        Ok(output_gradient.clone())
    }
}

pub(super) fn backward_scale_channels(
    spec: &crate::model::ScaleChannelsSpec,
    input: &Tensor,
    source: &Tensor,
    output: &Tensor,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "scale_channels",
        activation_input,
    )?;
    let mut input_grad = Tensor::zeros(input.shape);
    let mut source_grad = Tensor::zeros(source.shape);
    for b in 0..output.shape.batch {
        for c in 0..output.shape.channels {
            for y in 0..output.shape.height {
                for x in 0..output.shape.width {
                    let index = output.offset(b, c, y, x);
                    let scale = if spec.scale_wh {
                        input.get(b, 0, y, x)
                    } else {
                        input.get(b, c, 0, 0)
                    };
                    let source_value = source.get(b, c, y, x);
                    let grad = delta.data[index];
                    let source_index = source.offset(b, c, y, x);
                    source_grad.data[source_index] = grad * scale;
                    if spec.scale_wh {
                        let input_index = input.offset(b, 0, y, x);
                        input_grad.data[input_index] += grad * source_value;
                    } else {
                        let input_index = input.offset(b, c, 0, 0);
                        input_grad.data[input_index] += grad * source_value;
                    }
                }
            }
        }
    }
    Ok((input_grad, source_grad))
}

pub(super) fn backward_sam(
    spec: &crate::model::SamSpec,
    input: &Tensor,
    source: &Tensor,
    output: &Tensor,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let delta = apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "sam",
        activation_input,
    )?;
    if input.shape != source.shape || input.shape != output.shape {
        return Err(Error::InvalidConfig(format!(
            "sam backward shape mismatch: input {}, source {}, output {}",
            input.shape, source.shape, output.shape
        )));
    }
    let mut input_grad = Tensor::zeros(input.shape);
    let mut source_grad = Tensor::zeros(source.shape);
    for index in 0..delta.data.len() {
        input_grad.data[index] = delta.data[index] * source.data[index];
        source_grad.data[index] = delta.data[index] * input.data[index];
    }
    Ok((input_grad, source_grad))
}

pub(super) fn backward_reorg(
    spec: &crate::model::ReorgSpec,
    output_gradient: &Tensor,
    input_shape: TensorShape,
) -> Result<Tensor> {
    forward_reorg(spec.stride, !spec.reverse, output_gradient, input_shape)
}

pub(super) fn backward_convolutional(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input: &Tensor,
    output_shape: TensorShape,
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
    workspace: &mut Vec<f32>,
) -> Result<(Tensor, ConvolutionalParams)> {
    if output_gradient.shape != output_shape {
        return Err(Error::InvalidConfig(format!(
            "convolution backward gradient shape mismatch: expected {output_shape}, got {}",
            output_gradient.shape
        )));
    }

    let output = forward_convolutional(spec, params, input, output_shape);
    let delta = if let Some(normalization) = spec.activation.to_channel_normalization() {
        let activation_input = activation_input.ok_or_else(|| {
            Error::InvalidConfig(
                "convolutional channel-normalizing backward requires cached pre-activation"
                    .to_owned(),
            )
        })?;
        if activation_input.shape != output_shape {
            return Err(Error::InvalidConfig(format!(
                "convolutional channel-normalization cache shape mismatch: expected {output_shape}, got {}",
                activation_input.shape
            )));
        }
        Tensor {
            shape: output_shape,
            data: lightnet_cuda::host_ops::normalize_channels_backward(
                &activation_input.data,
                &output.data,
                &output_gradient.data,
                output_shape.batch,
                output_shape.channels,
                output_shape.width * output_shape.height,
                normalization,
                GradMode::from_env() == GradMode::Cpp,
            ),
        }
    } else {
        apply_output_activation_gradient(
            &output,
            output_gradient,
            &spec.activation,
            "convolutional",
            activation_input,
        )?
    };

    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);
    // Use the pre-allocated workspace as scratch for bias_grad and weight_grad.
    // This avoids a heap allocation per backward call (C++ net->workspace parity).
    let bias_size = spec.filters;
    let weight_size = params.weights.len();
    let needed = bias_size + weight_size;
    if workspace.len() < needed {
        workspace.resize(needed, 0.0);
    }
    workspace[..bias_size].fill(0.0);
    workspace[bias_size..needed].fill(0.0);
    // Split the workspace into two mutable views used as bias_grad / weight_grad.
    let (bias_grad, weight_grad) = workspace[..needed].split_at_mut(bias_size);
    let mut input_grad = Tensor::zeros(input.shape);
    let mut batch_norm_grad = params
        .batch_norm
        .as_ref()
        .map(|batch_norm| BatchNormParams {
            biases: vec![0.0; batch_norm.biases.len()],
            scales: vec![0.0; batch_norm.scales.len()],
            rolling_mean: vec![0.0; batch_norm.rolling_mean.len()],
            rolling_variance: vec![0.0; batch_norm.rolling_variance.len()],
        });

    for b in 0..input.shape.batch {
        for (filter, bias_grad_value) in bias_grad.iter_mut().enumerate() {
            let group = filter / filters_per_group.max(1);
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut grad_value = delta.get(b, filter, oy, ox);
                    if let Some(batch_norm) = params.batch_norm.as_ref() {
                        let raw_value =
                            convolution_raw_value(spec, &params.weights, input, b, filter, oy, ox);
                        let inv_std =
                            1.0 / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
                        let normalized = (raw_value - batch_norm.rolling_mean[filter]) * inv_std;
                        *bias_grad_value += grad_value;
                        if let Some(batch_norm_grad) = batch_norm_grad.as_mut() {
                            batch_norm_grad.scales[filter] += grad_value * normalized;
                        }
                        grad_value *= batch_norm.scales[filter] * inv_std;
                    } else {
                        *bias_grad_value += grad_value;
                    }

                    for ic in 0..channels_per_group {
                        let channel = group * channels_per_group + ic;
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

                                let src_y = src_y as usize;
                                let src_x = src_x as usize;
                                let weight_index = ((filter * channels_per_group + ic) * spec.size
                                    + ky)
                                    * spec.size
                                    + kx;
                                weight_grad[weight_index] +=
                                    input.get(b, channel, src_y, src_x) * grad_value;

                                let input_index = input.offset(b, channel, src_y, src_x);
                                input_grad.data[input_index] +=
                                    params.weights[weight_index] * grad_value;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((
        input_grad,
        ConvolutionalParams {
            biases: bias_grad.to_vec(),
            batch_norm: batch_norm_grad,
            weights: weight_grad.to_vec(),
        },
    ))
}

pub(super) fn convolution_raw_value(
    spec: &ConvolutionalSpec,
    weights: &[f32],
    input: &Tensor,
    batch: usize,
    filter: usize,
    out_y: usize,
    out_x: usize,
) -> f32 {
    let filters_per_group = spec.filters / spec.groups.max(1);
    let channels_per_group = input.shape.channels / spec.groups.max(1);
    let group = filter / filters_per_group.max(1);
    let mut acc = 0.0;

    for ic in 0..channels_per_group {
        for ky in 0..spec.size {
            for kx in 0..spec.size {
                let in_y = out_y * spec.stride_y + ky * spec.dilation;
                let in_x = out_x * spec.stride_x + kx * spec.dilation;
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
                let weight_index =
                    ((filter * channels_per_group + ic) * spec.size + ky) * spec.size + kx;
                acc += input.get(batch, channel, src_y as usize, src_x as usize)
                    * weights[weight_index];
            }
        }
    }

    acc
}

pub(super) fn backward_maxpool(
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    input: &Tensor,
    output_gradient: &Tensor,
    average: bool,
    zero_nonmax: bool,
) -> Result<Tensor> {
    let mut input_grad = Tensor::zeros(input.shape);
    let mut is_max = vec![false; input.data.len()];
    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            for oy in 0..output_gradient.shape.height {
                for ox in 0..output_gradient.shape.width {
                    let grad_value = output_gradient.get(b, c, oy, ox);
                    if average {
                        let mut valid = Vec::new();
                        for ky in 0..size {
                            for kx in 0..size {
                                let in_y = oy * stride_y + ky;
                                let in_x = ox * stride_x + kx;
                                let src_y = in_y as isize - (padding / 2) as isize;
                                let src_x = in_x as isize - (padding / 2) as isize;
                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }
                                valid.push((src_y as usize, src_x as usize));
                            }
                        }
                        if !valid.is_empty() {
                            let share = grad_value / valid.len() as f32;
                            for (y, x) in valid {
                                let idx = input.offset(b, c, y, x);
                                input_grad.data[idx] += share;
                            }
                        }
                    } else {
                        let mut best = None;
                        let mut best_value = f32::NEG_INFINITY;
                        for ky in 0..size {
                            for kx in 0..size {
                                let in_y = oy * stride_y + ky;
                                let in_x = ox * stride_x + kx;
                                let src_y = in_y as isize - (padding / 2) as isize;
                                let src_x = in_x as isize - (padding / 2) as isize;
                                if src_y < 0
                                    || src_y >= input.shape.height as isize
                                    || src_x < 0
                                    || src_x >= input.shape.width as isize
                                {
                                    continue;
                                }
                                let value = input.get(b, c, src_y as usize, src_x as usize);
                                if value > best_value {
                                    best_value = value;
                                    best = Some((src_y as usize, src_x as usize));
                                }
                            }
                        }
                        if let Some((y, x)) = best {
                            let idx = input.offset(b, c, y, x);
                            input_grad.data[idx] += grad_value;
                            is_max[idx] = true;
                        }
                    }
                }
            }
        }
    }
    if zero_nonmax && !average {
        for (index, chosen) in is_max.iter().enumerate() {
            if !*chosen {
                input_grad.data[index] = 0.0;
            }
        }
    }
    Ok(input_grad)
}

pub(super) fn backward_route(
    spec: &RouteSpec,
    layer_index: usize,
    output_gradient: &Tensor,
    layer_output_gradients: &mut [Tensor],
    layers: &[crate::model::Layer],
) -> Result<()> {
    let mut out_channel = 0usize;
    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source_layer = layers.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        let chunk_channels = source_layer.output_shape.channels / spec.groups.max(1);
        let channel_start = chunk_channels * spec.group_id;
        let source_grad =
            layer_output_gradients
                .get_mut(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: *raw_index,
                })?;

        for b in 0..source_layer.output_shape.batch {
            for c in 0..chunk_channels {
                for y in 0..source_layer.output_shape.height {
                    for x in 0..source_layer.output_shape.width {
                        let grad = output_gradient.get(b, out_channel + c, y, x);
                        let idx = source_grad.offset(b, channel_start + c, y, x);
                        source_grad.data[idx] += grad;
                    }
                }
            }
        }

        out_channel += chunk_channels;
    }
    Ok(())
}

/// `[laya_marker_gather]` backward: scatter-add the output gradient into input
/// A's per-position gradient (the exact adjoint of the forward gather).
pub(super) fn backward_laya_marker_gather(
    spec: &crate::model::LayaMarkerGatherSpec,
    layer_index: usize,
    forward_outputs: &[Tensor],
    output_gradient: &Tensor,
    layer_output_gradients: &mut [Tensor],
    layers: &[crate::model::Layer],
) -> Result<()> {
    let a_index = resolve_relative_layer(layer_index, spec.layers[0])?;
    let b_index = resolve_relative_layer(layer_index, spec.layers[1])?;
    let a_layer = layers.get(a_index).ok_or(Error::InvalidRoute {
        layer_index,
        reference: spec.layers[0],
    })?;
    let b = forward_outputs.get(b_index).ok_or(Error::InvalidRoute {
        layer_index,
        reference: spec.layers[1],
    })?;
    let delta_a = crate::laya_marker_gather::backward(
        &b.data,
        &output_gradient.data,
        a_layer.output_shape.batch,
        spec.hidden,
        spec.max_options,
        a_layer.output_shape.width,
        b.shape.channels,
    )
    .map_err(Error::InvalidConfig)?;
    let a_gradient = layer_output_gradients
        .get_mut(a_index)
        .ok_or(Error::InvalidRoute {
            layer_index,
            reference: spec.layers[0],
        })?;
    for (dst, src) in a_gradient.data.iter_mut().zip(delta_a) {
        *dst += src;
    }
    Ok(())
}

/// `[laya_act_head]` backward: the MLP gradient is computed for its own (frozen)
/// weights and the pooled-CLS gradient is accumulated into the head source's
/// output gradient at column 0.  Scorer and marker get no gradient.
#[allow(clippy::too_many_arguments)]
pub(super) fn backward_laya_act_head(
    spec: &crate::model::LayaActHeadSpec,
    weights: &crate::laya_act_head::LayaActHeadWeights,
    layer_index: usize,
    forward_outputs: &[Tensor],
    output_gradient: &Tensor,
    layer_output_gradients: &mut [Tensor],
) -> Result<()> {
    let head_index = resolve_relative_layer(layer_index, spec.layers[0])?;
    let scorer_index = resolve_relative_layer(layer_index, spec.layers[1])?;
    let marker_index = resolve_relative_layer(layer_index, spec.layers[2])?;
    let head = forward_outputs.get(head_index).ok_or(Error::InvalidRoute {
        layer_index,
        reference: spec.layers[0],
    })?;
    let scorer = forward_outputs
        .get(scorer_index)
        .ok_or(Error::InvalidRoute {
            layer_index,
            reference: spec.layers[1],
        })?;
    let marker = forward_outputs
        .get(marker_index)
        .ok_or(Error::InvalidRoute {
            layer_index,
            reference: spec.layers[2],
        })?;
    let grad = crate::laya_act_head::act_head_backward(
        &head.data,
        &scorer.data,
        &marker.data,
        weights,
        &output_gradient.data,
        head.shape.batch,
        spec.hidden,
        spec.max_options,
        marker.shape.channels,
        head.shape.width,
        marker.shape.width,
        spec.fc_dim,
        spec.n_act,
    )
    .map_err(Error::InvalidConfig)?;
    let head_grad = layer_output_gradients
        .get_mut(head_index)
        .ok_or(Error::InvalidRoute {
            layer_index,
            reference: spec.layers[0],
        })?;
    for b in 0..head.shape.batch {
        for e in 0..spec.hidden {
            head_grad.data[(b * spec.hidden + e) * head.shape.width] +=
                grad.dcls[b * spec.hidden + e];
        }
    }
    Ok(())
}

pub(super) fn backward_shortcut(
    spec: &crate::model::ShortcutSpec,
    layer_index: usize,
    forward_outputs: &[Tensor],
    output_gradient: &Tensor,
    activation_input: Option<&Tensor>,
) -> Result<Tensor> {
    let output = forward_outputs.get(layer_index).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "missing saved forward output for shortcut layer #{layer_index}"
        ))
    })?;
    apply_output_activation_gradient(
        output,
        output_gradient,
        &spec.activation,
        "shortcut",
        activation_input,
    )
}

pub(super) fn backward_upsample(
    spec: &crate::model::UpsampleSpec,
    output_gradient: &Tensor,
    input_shape: TensorShape,
) -> Result<Tensor> {
    let stride = spec.stride.unsigned_abs();
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "upsample stride cannot be zero".to_owned(),
        ));
    }

    let mut input_grad = Tensor::zeros(input_shape);
    if spec.reverse {
        let covered_height = output_gradient
            .shape
            .height
            .checked_mul(stride)
            .ok_or_else(|| {
                Error::InvalidConfig("reverse upsample gradient height overflows".to_owned())
            })?;
        let covered_width = output_gradient
            .shape
            .width
            .checked_mul(stride)
            .ok_or_else(|| {
                Error::InvalidConfig("reverse upsample gradient width overflows".to_owned())
            })?;
        for b in 0..input_shape.batch {
            for c in 0..input_shape.channels {
                for y in 0..covered_height {
                    for x in 0..covered_width {
                        let src_y = y / stride;
                        let src_x = x / stride;
                        input_grad.set(
                            b,
                            c,
                            y,
                            x,
                            spec.scale * output_gradient.get(b, c, src_y, src_x),
                        );
                    }
                }
            }
        }
    } else {
        for b in 0..output_gradient.shape.batch {
            for c in 0..output_gradient.shape.channels {
                for y in 0..output_gradient.shape.height {
                    for x in 0..output_gradient.shape.width {
                        let dst_y = y / stride;
                        let dst_x = x / stride;
                        let idx = input_grad.offset(b, c, dst_y, dst_x);
                        input_grad.data[idx] += spec.scale * output_gradient.get(b, c, y, x);
                    }
                }
            }
        }
    }
    Ok(input_grad)
}

pub(super) fn apply_output_activation_gradient(
    output: &Tensor,
    output_gradient: &Tensor,
    activation: &Activation,
    layer_name: &str,
    activation_input: Option<&Tensor>,
) -> Result<Tensor> {
    if output.shape != output_gradient.shape {
        return Err(Error::InvalidConfig(format!(
            "{layer_name} backward gradient shape mismatch: expected {}, got {}",
            output.shape, output_gradient.shape
        )));
    }
    if let Some(cached_input) = activation_input
        && cached_input.shape != output.shape
    {
        return Err(Error::InvalidConfig(format!(
            "{layer_name} activation cache shape mismatch: expected {}, got {}",
            output.shape, cached_input.shape
        )));
    }

    let mut delta = Tensor::zeros(output.shape);
    for (index, (dst, (out_value, grad_value))) in delta
        .data
        .iter_mut()
        .zip(output.data.iter().zip(&output_gradient.data))
        .enumerate()
    {
        let cached_input = activation_input.map(|tensor| tensor.data[index]);
        *dst = activation_gradient_from_output(activation, *out_value, cached_input, layer_name)?
            * *grad_value;
    }
    Ok(delta)
}

pub(super) fn activation_gradient_from_output(
    activation: &Activation,
    output_value: f32,
    activation_input: Option<f32>,
    layer_name: &str,
) -> Result<f32> {
    match activation {
        Activation::Linear => Ok(1.0),
        Activation::Logistic => Ok(output_value * (1.0 - output_value)),
        Activation::Loggy => Ok(activation_input.map_or_else(
            || crate::activation_math::loggy_grad_from_output(output_value),
            crate::activation_math::loggy_grad,
        )),
        Activation::Relu => Ok((output_value > 0.0) as i32 as f32),
        Activation::Leaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
        Activation::Tanh => Ok(1.0 - output_value * output_value),
        // relu6(x) is in (0, 6) exactly when x is, so the output form is exact.
        Activation::Relu6 => Ok((output_value > 0.0 && output_value < 6.0) as i32 as f32),
        // ramp(x) > 0 exactly when x > 0.
        Activation::Ramp => Ok((output_value > 0.0) as i32 as f32 + 0.1),
        Activation::Swish => {
            let x = activation_input.ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "{layer_name} backward requires cached activation_input for {:?}",
                    activation
                ))
            })?;
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            Ok(sigmoid + x * sigmoid * (1.0 - sigmoid))
        }
        Activation::Mish => {
            let x = activation_input.ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "{layer_name} backward requires cached activation_input for {:?}",
                    activation
                ))
            })?;
            const MISH_THRESHOLD: f32 = 20.0;
            let softplus = if x > MISH_THRESHOLD {
                x
            } else if x < -MISH_THRESHOLD {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tsp = softplus.tanh();
            let grad_tsp = (1.0 - tsp * tsp) * grad_sp;
            Ok(x * grad_tsp + tsp)
        }
        Activation::Gelu => Ok(crate::activation_math::gelu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Elu => Ok(crate::activation_math::elu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Selu => Ok(crate::activation_math::selu_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Plse => Ok(crate::activation_math::plse_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Hardtan => Ok(crate::activation_math::hardtan_grad(
            cached_activation_input(activation, activation_input, layer_name)?,
        )),
        Activation::Lhtan => Ok(crate::activation_math::lhtan_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::Stair => Ok(crate::activation_math::stair_grad(cached_activation_input(
            activation,
            activation_input,
            layer_name,
        )?)),
        Activation::HardMish => Ok(crate::activation_math::hard_mish_grad(
            cached_activation_input(activation, activation_input, layer_name)?,
        )),
        // relie / revleaky are leaky-shaped: the output has the sign of the input.
        Activation::Relie => Ok(if output_value > 0.0 { 1.0 } else { 0.01 }),
        Activation::Revleaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
        Activation::NormalizeChannels
        | Activation::NormalizeChannelsSoftmax
        | Activation::NormalizeChannelsSoftmaxMaxval => Err(Error::InvalidConfig(format!(
            "{layer_name} channel-normalizing gradient requires the tensor-level convolution path"
        ))),
        Activation::Other(name) => Err(Error::UnsupportedLayer {
            layer_index: 0,
            name: format!("{layer_name}_{name}_backward"),
        }),
    }
}

pub(super) fn cached_activation_input(
    activation: &Activation,
    activation_input: Option<f32>,
    layer_name: &str,
) -> Result<f32> {
    activation_input.ok_or_else(|| {
        Error::InvalidConfig(format!(
            "{layer_name} backward requires cached activation_input for {activation:?}"
        ))
    })
}

pub(super) fn accumulate_previous_gradient(
    layer_index: usize,
    gradient: &Tensor,
    input_gradient: &mut Tensor,
    layer_output_gradients: &mut [Tensor],
) -> Result<()> {
    if layer_index == 0 {
        accumulate_tensor(input_gradient, gradient)
    } else {
        accumulate_tensor(
            layer_output_gradients
                .get_mut(layer_index - 1)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous layer gradient slot for layer #{layer_index}"
                    ))
                })?,
            gradient,
        )
    }
}

pub(super) fn accumulate_tensor(dst: &mut Tensor, src: &Tensor) -> Result<()> {
    if dst.shape != src.shape {
        return Err(Error::InvalidConfig(format!(
            "gradient accumulation shape mismatch: {} vs {}",
            dst.shape, src.shape
        )));
    }
    for (dst_value, src_value) in dst.data.iter_mut().zip(&src.data) {
        *dst_value += *src_value;
    }
    Ok(())
}

pub(super) fn normalize_value(
    value: f32,
    index: usize,
    biases: &[f32],
    batch_norm: Option<&BatchNormParams>,
) -> f32 {
    if let Some(batch_norm) = batch_norm {
        let normalized = (value - batch_norm.rolling_mean[index])
            / (batch_norm.rolling_variance[index] + BATCH_NORM_EPSILON).sqrt();
        normalized * batch_norm.scales[index] + biases[index]
    } else {
        value + biases[index]
    }
}

pub(super) fn activate(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::Linear => value,
        Activation::Logistic => 1.0 / (1.0 + (-value).exp()),
        Activation::Loggy => crate::activation_math::loggy(value),
        Activation::Relu => value.max(0.0),
        Activation::Leaky => {
            if value > 0.0 {
                value
            } else {
                0.1 * value
            }
        }
        Activation::Mish => value * (1.0 + value.exp()).ln().tanh(),
        Activation::Swish => value / (1.0 + (-value).exp()),
        Activation::Tanh => value.tanh(),
        Activation::Relu6 => value.clamp(0.0, 6.0),
        Activation::Ramp => value * (value > 0.0) as i32 as f32 + 0.1 * value,
        Activation::Gelu => crate::activation_math::gelu(value),
        Activation::Elu => crate::activation_math::elu(value),
        Activation::Selu => crate::activation_math::selu(value),
        Activation::Relie => crate::activation_math::relie(value),
        Activation::Plse => crate::activation_math::plse(value),
        Activation::Hardtan => crate::activation_math::hardtan(value),
        Activation::Lhtan => crate::activation_math::lhtan(value),
        Activation::Stair => crate::activation_math::stair(value),
        Activation::HardMish => crate::activation_math::hard_mish(value),
        Activation::Revleaky => {
            if value > 0.0 {
                value
            } else {
                0.1 * value
            }
        }
        Activation::NormalizeChannels
        | Activation::NormalizeChannelsSoftmax
        | Activation::NormalizeChannelsSoftmaxMaxval => {
            panic!("channel-normalizing activation requires convolution geometry")
        }
        Activation::Other(_) => value,
    }
}

pub(super) fn forward_maxpool(
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    input: &Tensor,
    output_shape: TensorShape,
    average: bool,
) -> Result<Tensor> {
    if output_shape.channels != input.shape.channels {
        return Err(Error::UnsupportedLayer {
            layer_index: 0,
            name: "maxpool_depth".to_owned(),
        });
    }

    let mut output = Tensor::zeros(output_shape);
    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let mut acc = if average { 0.0 } else { f32::NEG_INFINITY };
                    let mut count = 0usize;

                    for ky in 0..size {
                        for kx in 0..size {
                            let in_y = oy * stride_y + ky;
                            let in_x = ox * stride_x + kx;
                            let src_y = in_y as isize - (padding / 2) as isize;
                            let src_x = in_x as isize - (padding / 2) as isize;

                            if src_y < 0
                                || src_y >= input.shape.height as isize
                                || src_x < 0
                                || src_x >= input.shape.width as isize
                            {
                                continue;
                            }

                            let value = input.get(b, c, src_y as usize, src_x as usize);
                            if average {
                                acc += value;
                                count += 1;
                            } else {
                                acc = acc.max(value);
                            }
                        }
                    }

                    if average && count > 0 {
                        acc /= count as f32;
                    }

                    output.set(b, c, oy, ox, acc);
                }
            }
        }
    }
    Ok(output)
}

pub(super) fn forward_avgpool(input: &Tensor) -> Tensor {
    let mut output = Tensor::zeros(TensorShape::new(
        1,
        1,
        input.shape.channels,
        input.shape.batch,
    ));
    let area = input.channel_area() as f32;

    for b in 0..input.shape.batch {
        for c in 0..input.shape.channels {
            let mut acc = 0.0;
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    acc += input.get(b, c, y, x);
                }
            }
            output.data[b * input.shape.channels + c] = acc / area;
        }
    }

    output
}

pub(super) fn backward_avgpool(output_grad: &Tensor, input_shape: TensorShape) -> Result<Tensor> {
    let expected = TensorShape::new(1, 1, input_shape.channels, input_shape.batch);
    if output_grad.shape != expected {
        return Err(Error::InvalidConfig(format!(
            "avgpool backward gradient shape mismatch: expected {expected}, got {}",
            output_grad.shape
        )));
    }

    let mut input_grad = Tensor::zeros(input_shape);
    let area = (input_shape.width * input_shape.height) as f32;
    for b in 0..input_shape.batch {
        for c in 0..input_shape.channels {
            let grad = output_grad.data[b * input_shape.channels + c] / area;
            for y in 0..input_shape.height {
                for x in 0..input_shape.width {
                    input_grad.set(b, c, y, x, grad);
                }
            }
        }
    }

    Ok(input_grad)
}

pub(super) fn forward_route(
    spec: &RouteSpec,
    layer_index: usize,
    outputs: &[Tensor],
    output_shape: TensorShape,
) -> Result<Tensor> {
    let mut output = Tensor::zeros(output_shape);
    let mut out_channel = 0usize;

    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        let chunk_channels = source.shape.channels / spec.groups.max(1);
        let channel_start = chunk_channels * spec.group_id;

        for b in 0..source.shape.batch {
            for c in 0..chunk_channels {
                for y in 0..source.shape.height {
                    for x in 0..source.shape.width {
                        let value = source.get(b, channel_start + c, y, x);
                        output.set(b, out_channel + c, y, x, value);
                    }
                }
            }
        }

        out_channel += chunk_channels;
    }

    Ok(output)
}

pub(super) fn forward_route_compact(
    spec: &RouteSpec,
    layer_index: usize,
    outputs: &[Option<Tensor>],
    output_shape: TensorShape,
) -> Result<Tensor> {
    let mut output = Tensor::zeros(output_shape);
    let mut out_channel = 0usize;

    for raw_index in &spec.layers {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source =
            outputs
                .get(source_index)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: *raw_index,
                })?;
        let chunk_channels = source.shape.channels / spec.groups.max(1);
        let channel_start = chunk_channels * spec.group_id;

        for b in 0..source.shape.batch {
            for c in 0..chunk_channels {
                for y in 0..source.shape.height {
                    for x in 0..source.shape.width {
                        let value = source.get(b, channel_start + c, y, x);
                        output.set(b, out_channel + c, y, x, value);
                    }
                }
            }
        }
        out_channel += chunk_channels;
    }
    Ok(output)
}

/// `[laya_marker_gather]` forward over two already-resolved source tensors:
/// input A is `[batch, hidden, 1, T]`, input B is `[batch, marker_channels, 1, T]`.
pub(super) fn laya_marker_gather_forward(
    spec: &crate::model::LayaMarkerGatherSpec,
    a: &Tensor,
    b: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let data = crate::laya_marker_gather::forward(
        &a.data,
        &b.data,
        a.shape.batch,
        spec.hidden,
        spec.max_options,
        a.shape.width,
        b.shape.channels,
    )
    .map_err(Error::InvalidConfig)?;
    Tensor::from_data(output_shape, data)
}

/// `[laya_act_head]` forward over three already-resolved source tensors.
pub(super) fn laya_act_head_forward(
    spec: &crate::model::LayaActHeadSpec,
    weights: &crate::laya_act_head::LayaActHeadWeights,
    head: &Tensor,
    scorer: &Tensor,
    marker: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let data = crate::laya_act_head::act_head_forward(
        &head.data,
        &scorer.data,
        &marker.data,
        weights,
        head.shape.batch,
        spec.hidden,
        spec.max_options,
        marker.shape.channels,
        head.shape.width,
        marker.shape.width,
        spec.fc_dim,
        spec.n_act,
    )
    .map_err(Error::InvalidConfig)?;
    Tensor::from_data(output_shape, data)
}

/// Apply the explicit source-C++ stop-gradient deviation or the exact Rust
/// input derivative for the frozen Depth Anything encoder.
pub(super) fn depth_anything_input_gradient(
    mode: GradMode,
    spec: &crate::model::DepthAnythingSpec,
    input_shape: TensorShape,
    output_gradient: &[f32],
    weights: &crate::depth_anything::DepthAnythingWeights,
    cache: &crate::depth_anything::DepthAnythingForwardCache,
) -> Result<Option<Tensor>> {
    if mode == GradMode::Cpp {
        return Ok(None);
    }
    let geometry = spec.geometry(input_shape);
    let data = crate::depth_anything::backward(output_gradient, weights, &geometry, cache)
        .map_err(Error::InvalidConfig)?;
    Ok(Some(Tensor::from_data(input_shape, data)?))
}

/// `[laya_encoder]` forward: build the geometry from the spec + input shapes and
/// run the frozen ModernBERT stack on the host.
pub(super) fn laya_encoder_forward(
    spec: &crate::model::LayaEncoderSpec,
    weights: &crate::laya_encoder::LayaEncoderWeights,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let geometry = crate::laya_encoder::LayaEncoderGeometry {
        batch: input.shape.batch,
        seq_len: input.shape.width,
        hidden: spec.hidden,
        heads: spec.heads,
        n_layers: spec.n_layers,
        intermediate: spec.intermediate,
        vocab: spec.vocab,
        local_window: if spec.local_window > 0 {
            spec.local_window
        } else {
            crate::model::LAYA_ENCODER_LOCAL_WINDOW
        },
        rope_theta_full: if spec.rope_theta_full > 0.0 {
            spec.rope_theta_full
        } else {
            crate::model::LAYA_ENCODER_ROPE_THETA_FULL
        },
        rope_theta_local: if spec.rope_theta_local > 0.0 {
            spec.rope_theta_local
        } else {
            crate::model::LAYA_ENCODER_ROPE_THETA_LOCAL
        },
        ln_eps: if spec.ln_eps > 0.0 {
            spec.ln_eps
        } else {
            1.0e-5
        },
    };
    let data = crate::laya_encoder::encoder_forward(&input.data, weights, &geometry)
        .map_err(Error::InvalidConfig)?;
    Tensor::from_data(output_shape, data)
}

/// `[laya_head]` forward: build the geometry from the spec + input shape and run
/// the frozen decision head on the host.
pub(super) fn laya_head_forward(
    spec: &crate::model::LayaHeadSpec,
    weights: &crate::laya_head::LayaHeadWeights,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let geometry = crate::laya_head::LayaHeadGeometry {
        batch: input.shape.batch,
        seq_len: input.shape.width,
        hidden: spec.hidden,
        heads: spec.heads,
        n_layers: spec.n_layers,
        ffn_dim: spec.ffn_dim,
        ln_eps: if spec.ln_eps > 0.0 {
            spec.ln_eps
        } else {
            crate::model::LAYA_LAYERNORM_EPS
        },
        qtype: spec.qtype,
    };
    let data = crate::laya_head::head_forward(&input.data, weights, &geometry)
        .map_err(Error::InvalidConfig)?;
    Tensor::from_data(output_shape, data)
}

pub(super) fn shortcut_sources<'a>(
    spec: &crate::model::ShortcutSpec,
    layer_index: usize,
    outputs: &'a [Tensor],
    batch: usize,
) -> Result<Vec<&'a Tensor>> {
    spec.from
        .iter()
        .map(|raw_index| {
            let source_index = resolve_relative_layer(layer_index, *raw_index)?;
            let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
                layer_index,
                reference: *raw_index,
            })?;
            if source.shape.batch != batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{layer_index} mixes incompatible batch shapes"
                )));
            }
            Ok(source)
        })
        .collect()
}

pub(super) fn forward_shortcut_weighted(
    spec: &crate::model::ShortcutSpec,
    params: &ShortcutParams,
    input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<(Tensor, Option<Tensor>)> {
    let sources = shortcut_sources(spec, layer_index, outputs, input.shape.batch)?;
    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
    let pre = crate::shortcut_weights::weighted_forward(
        &input.data,
        &slices,
        input.shape.batch,
        input.shape.element_count(),
        &params.weights,
        spec.weights_normalization,
        spec.weight_slots(input.shape.channels),
    );
    let output: Vec<f32> = pre
        .iter()
        .map(|v| activate(spec.activation.clone(), *v))
        .collect();
    Ok((
        Tensor::from_data(input.shape, output)?,
        Some(Tensor::from_data(input.shape, pre)?),
    ))
}

pub(super) fn forward_shortcut_with_cache(
    spec: &crate::model::ShortcutSpec,
    current_input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<(Tensor, Option<Tensor>)> {
    let mut output = current_input.clone();
    for raw_index in &spec.from {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
            layer_index,
            reference: *raw_index,
        })?;
        if source.shape.batch != output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "shortcut layer #{layer_index} mixes incompatible batch shapes {} and {}",
                output.shape, source.shape
            )));
        }
        let output_stride = output.shape.element_count();
        let source_stride = source.shape.element_count();
        let copied = output_stride.min(source_stride);
        for batch in 0..output.shape.batch {
            let output_offset = batch * output_stride;
            let source_offset = batch * source_stride;
            for index in 0..copied {
                output.data[output_offset + index] += source.data[source_offset + index];
            }
        }
    }
    let activation_input = output.clone();
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok((output, Some(activation_input)))
}

pub(super) fn forward_shortcut_compact(
    spec: &crate::model::ShortcutSpec,
    current_input: &Tensor,
    layer_index: usize,
    outputs: &[Option<Tensor>],
) -> Result<Tensor> {
    let mut output = current_input.clone();
    for raw_index in &spec.from {
        let source_index = resolve_relative_layer(layer_index, *raw_index)?;
        let source =
            outputs
                .get(source_index)
                .and_then(Option::as_ref)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: *raw_index,
                })?;
        if source.shape.batch != output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "shortcut layer #{layer_index} mixes incompatible batch shapes {} and {}",
                output.shape, source.shape
            )));
        }
        let output_stride = output.shape.element_count();
        let source_stride = source.shape.element_count();
        let copied = output_stride.min(source_stride);
        for batch in 0..output.shape.batch {
            let output_offset = batch * output_stride;
            let source_offset = batch * source_stride;
            for index in 0..copied {
                output.data[output_offset + index] += source.data[source_offset + index];
            }
        }
    }
    for value in &mut output.data {
        *value = activate(spec.activation.clone(), *value);
    }
    Ok(output)
}

pub(super) fn shortcut_source_gradient(output_gradient: &Tensor, source_shape: TensorShape) -> Result<Tensor> {
    if output_gradient.shape.batch != source_shape.batch {
        return Err(Error::InvalidConfig(format!(
            "shortcut gradient batch {} is incompatible with source batch {}",
            output_gradient.shape, source_shape
        )));
    }

    let mut source_gradient = Tensor::zeros(source_shape);
    let output_stride = output_gradient.shape.element_count();
    let source_stride = source_shape.element_count();
    let copied = output_stride.min(source_stride);
    for batch in 0..source_shape.batch {
        let output_offset = batch * output_stride;
        let source_offset = batch * source_stride;
        source_gradient.data[source_offset..source_offset + copied]
            .copy_from_slice(&output_gradient.data[output_offset..output_offset + copied]);
    }
    Ok(source_gradient)
}

pub(super) fn forward_upsample(
    spec: &crate::model::UpsampleSpec,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    let stride = spec.stride.unsigned_abs();
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "upsample stride cannot be zero".to_owned(),
        ));
    }

    let mut output = Tensor::zeros(output_shape);
    if spec.reverse {
        let covered_height = output.shape.height.checked_mul(stride).ok_or_else(|| {
            Error::InvalidConfig("reverse upsample covered height overflows".to_owned())
        })?;
        let covered_width = output.shape.width.checked_mul(stride).ok_or_else(|| {
            Error::InvalidConfig("reverse upsample covered width overflows".to_owned())
        })?;
        for b in 0..output.shape.batch {
            for c in 0..output.shape.channels {
                for y in 0..covered_height {
                    for x in 0..covered_width {
                        let dst_y = y / stride;
                        let dst_x = x / stride;
                        let dst = output.offset(b, c, dst_y, dst_x);
                        output.data[dst] += spec.scale * input.get(b, c, y, x);
                    }
                }
            }
        }
    } else {
        for b in 0..input.shape.batch {
            for c in 0..input.shape.channels {
                for y in 0..output.shape.height {
                    for x in 0..output.shape.width {
                        let src_y = y / stride;
                        let src_x = x / stride;
                        let value = spec.scale * input.get(b, c, src_y, src_x);
                        output.set(b, c, y, x, value);
                    }
                }
            }
        }
    }

    Ok(output)
}

pub(super) fn forward_reorg(
    stride: usize,
    reverse: bool,
    input: &Tensor,
    output_shape: TensorShape,
) -> Result<Tensor> {
    if stride == 0 {
        return Err(Error::InvalidConfig(
            "reorg stride cannot be zero".to_owned(),
        ));
    }

    let mut output = Tensor::zeros(output_shape);
    if reverse {
        for b in 0..input.shape.batch {
            for c in 0..input.shape.channels {
                // Inverse of the non-reverse mapping below (Darknet reorg_cpu:
                // small channel k <-> big channel k % out_c, offset k / out_c).
                let out_c = c % output.shape.channels;
                let offset = c / output.shape.channels;
                let dy = offset / stride;
                let dx = offset % stride;
                for y in 0..input.shape.height {
                    for x in 0..input.shape.width {
                        output.set(
                            b,
                            out_c,
                            y * stride + dy,
                            x * stride + dx,
                            input.get(b, c, y, x),
                        );
                    }
                }
            }
        }
    } else {
        for b in 0..output.shape.batch {
            for c in 0..output.shape.channels {
                let in_c = c % input.shape.channels;
                let offset = c / input.shape.channels;
                let dy = offset / stride;
                let dx = offset % stride;
                for y in 0..output.shape.height {
                    for x in 0..output.shape.width {
                        output.set(
                            b,
                            c,
                            y,
                            x,
                            input.get(b, in_c, y * stride + dy, x * stride + dx),
                        );
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn resolve_relative_layer(layer_index: usize, raw_index: isize) -> Result<usize> {
    let absolute = if raw_index < 0 {
        layer_index
            .checked_add_signed(raw_index)
            .ok_or(Error::InvalidRoute {
                layer_index,
                reference: raw_index,
            })?
    } else {
        usize::try_from(raw_index).map_err(|_| Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        })?
    };

    if absolute >= layer_index {
        return Err(Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        });
    }

    Ok(absolute)
}
