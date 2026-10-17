//! CPU reference math for the `Jev_laya` graph-convolution layer.
//!
//! The layer reduces valid NCHW neighborhoods, optionally weighting each edge
//! with a learned reference/neighbor score, then applies grouped pointwise
//! projections. This module keeps the parameter-free math and its exact
//! derivative separate from runtime BN/activation and optimizer glue.

use crate::error::{Error, Result};
use crate::model::{GraphConvSpec, TensorShape};
use crate::runtime::Tensor;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphConvForwardCache {
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    /// Projection output before bias/BN, `[B, filters, out_h, out_w]`.
    pub raw_output: Tensor,
    /// Reference features at every output site, `[B, C, out_h, out_w]`.
    reference: Vec<f32>,
    /// Valid-neighbor aggregate at every output site, `[B, C, out_h, out_w]`.
    aggregate: Vec<f32>,
    /// Normalized edge weights, `[B, groups, out_h, out_w, size*size]`.
    alpha: Vec<f32>,
    /// Whether the corresponding neighbor coordinate is inside the input.
    valid: Vec<bool>,
    /// Whether the center/reference coordinate is inside the input.
    reference_valid: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GraphConvGradients {
    pub input: Vec<f32>,
    pub weights: Vec<f32>,
    pub self_weights: Option<Vec<f32>>,
    pub edge_kernel: Option<Vec<f32>>,
    pub edge_biases: Option<Vec<f32>>,
}

#[inline]
fn feature_index(
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

#[inline]
fn site_index(
    batch: usize,
    group: usize,
    y: usize,
    x: usize,
    groups: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * groups + group) * height + y) * width + x
}

#[inline]
fn output_index(
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

fn source_coordinate(
    output: usize,
    stride: usize,
    padding: usize,
    kernel: usize,
    dilation: usize,
) -> i128 {
    output as i128 * stride as i128 - padding as i128 + kernel as i128 * dilation as i128
}

fn validate_geometry(
    spec: &GraphConvSpec,
    input: TensorShape,
    output: TensorShape,
) -> Result<(usize, usize, usize, usize, usize)> {
    let groups = spec.groups.max(1);
    if spec.filters == 0
        || spec.size == 0
        || spec.size % 2 == 0
        || spec.stride_x == 0
        || spec.stride_y == 0
        || spec.dilation == 0
        || input.channels == 0
        || input.height == 0
        || input.width == 0
        || input.channels % groups != 0
        || spec.filters % groups != 0
        || !matches!(spec.edge_mode, 0 | 1)
    {
        return Err(Error::InvalidConfig(
            "invalid [graph_conv] math geometry or edge mode".to_owned(),
        ));
    }
    let expected_h = graph_conv_output_dim(
        input.height,
        spec.stride_y,
        spec.padding,
        spec.dilation,
        spec.size,
    )?;
    let expected_w = graph_conv_output_dim(
        input.width,
        spec.stride_x,
        spec.padding,
        spec.dilation,
        spec.size,
    )?;
    if output.batch != input.batch
        || output.channels != spec.filters
        || output.height != expected_h
        || output.width != expected_w
    {
        return Err(Error::InvalidConfig(format!(
            "[graph_conv] output shape mismatch: expected [B={}, C={}, H={expected_h}, W={expected_w}], got {output}",
            input.batch, spec.filters
        )));
    }
    let channels_per_group = input.channels / groups;
    let filters_per_group = spec.filters / groups;
    let kernel_area = spec.size.checked_mul(spec.size).ok_or_else(|| {
        Error::InvalidConfig("[graph_conv] kernel area overflows usize".to_owned())
    })?;
    let output_area = expected_h.checked_mul(expected_w).ok_or_else(|| {
        Error::InvalidConfig("[graph_conv] output area overflows usize".to_owned())
    })?;
    Ok((
        groups,
        channels_per_group,
        filters_per_group,
        kernel_area,
        output_area,
    ))
}

fn graph_conv_output_dim(
    input: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size: usize,
) -> Result<usize> {
    let numerator = input as i128 + 2 * padding as i128 - dilation as i128 * (size - 1) as i128 - 1;
    if numerator < 0 {
        return Err(Error::InvalidConfig(
            "[graph_conv] kernel has no output sites for the input geometry".to_owned(),
        ));
    }
    usize::try_from(numerator / stride as i128 + 1).map_err(|_| {
        Error::InvalidConfig("[graph_conv] output dimension overflows usize".to_owned())
    })
}

/// Run only graph aggregation and the grouped pointwise projections.
/// `weights` are `[filters, input_channels/groups]`; optional learned blocks
/// use the source layout `[groups, size*size, 2*input_channels/groups]` and
/// `[groups, size*size]`.
pub(crate) fn graph_conv_forward(
    spec: &GraphConvSpec,
    input: &Tensor,
    output_shape: TensorShape,
    weights: &[f32],
    self_weights: Option<&[f32]>,
    edge_kernel: Option<&[f32]>,
    edge_biases: Option<&[f32]>,
) -> Result<GraphConvForwardCache> {
    let (groups, channels_per_group, filters_per_group, kernel_area, _) =
        validate_geometry(spec, input.shape, output_shape)?;
    let channels = input.shape.channels;
    let output_height = output_shape.height;
    let output_width = output_shape.width;
    let area = output_height.checked_mul(output_width).ok_or_else(|| {
        Error::InvalidConfig("[graph_conv] output area overflows usize".to_owned())
    })?;
    let expected_weights = spec
        .filters
        .checked_mul(channels_per_group)
        .ok_or_else(|| {
            Error::InvalidConfig("[graph_conv] projection weight count overflows usize".to_owned())
        })?;
    if weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "[graph_conv] projection weight length mismatch: expected {expected_weights}, got {}",
            weights.len()
        )));
    }
    match (spec.use_self, self_weights) {
        (true, Some(values)) if values.len() == expected_weights => {}
        (false, None) => {}
        (true, Some(values)) => {
            return Err(Error::InvalidConfig(format!(
                "[graph_conv] self weight length mismatch: expected {expected_weights}, got {}",
                values.len()
            )));
        }
        (true, None) => {
            return Err(Error::InvalidConfig(
                "[graph_conv] self projection is enabled but its weights are missing".to_owned(),
            ));
        }
        (false, Some(_)) => {
            return Err(Error::InvalidConfig(
                "[graph_conv] self weights supplied while self projection is disabled".to_owned(),
            ));
        }
    }
    let edge_count = groups.checked_mul(kernel_area).ok_or_else(|| {
        Error::InvalidConfig("[graph_conv] edge parameter count overflows usize".to_owned())
    })?;
    let expected_edge_kernel = edge_count
        .checked_mul(2)
        .and_then(|count| count.checked_mul(channels_per_group))
        .ok_or_else(|| {
            Error::InvalidConfig("[graph_conv] edge-kernel count overflows usize".to_owned())
        })?;
    match (spec.edge_mode, edge_kernel, edge_biases) {
        (0, None, None) => {}
        (0, _, _) => {
            return Err(Error::InvalidConfig(
                "[graph_conv] uniform edge mode must not have learned edge parameters".to_owned(),
            ));
        }
        (1, Some(kernel), Some(biases))
            if kernel.len() == expected_edge_kernel && biases.len() == edge_count => {}
        (1, Some(kernel), Some(biases)) => {
            return Err(Error::InvalidConfig(format!(
                "[graph_conv] learned edge parameter length mismatch: expected kernel {}, biases {edge_count}; got kernel {}, biases {}",
                expected_edge_kernel,
                kernel.len(),
                biases.len()
            )));
        }
        (1, _, _) => {
            return Err(Error::InvalidConfig(
                "[graph_conv] learned edge mode requires edge kernel and bias parameters"
                    .to_owned(),
            ));
        }
        _ => unreachable!("edge mode validated above"),
    }

    let reference_len = input
        .shape
        .batch
        .checked_mul(channels)
        .and_then(|count| count.checked_mul(area))
        .ok_or_else(|| {
            Error::InvalidConfig("[graph_conv] feature-cache count overflows usize".to_owned())
        })?;
    let edge_len = input
        .shape
        .batch
        .checked_mul(groups)
        .and_then(|count| count.checked_mul(area))
        .and_then(|count| count.checked_mul(kernel_area))
        .ok_or_else(|| {
            Error::InvalidConfig("[graph_conv] edge-cache count overflows usize".to_owned())
        })?;
    let mut reference = vec![0.0; reference_len];
    let mut aggregate = vec![0.0; reference_len];
    let mut alpha = vec![0.0; edge_len];
    let mut valid = vec![false; edge_len];
    let mut reference_valid = vec![false; input.shape.batch * groups * area];
    let mut raw_output = Tensor::zeros(output_shape);
    let center = spec.size / 2;

    for batch in 0..input.shape.batch {
        for group in 0..groups {
            let site_base = site_index(batch, group, 0, 0, groups, output_height, output_width);
            let input_channel_base = group * channels_per_group;
            for oy in 0..output_height {
                for ox in 0..output_width {
                    let site = site_base + oy * output_width + ox;
                    let reference_y =
                        source_coordinate(oy, spec.stride_y, spec.padding, center, spec.dilation);
                    let reference_x =
                        source_coordinate(ox, spec.stride_x, spec.padding, center, spec.dilation);
                    let ref_ok = reference_y >= 0
                        && reference_y < input.shape.height as i128
                        && reference_x >= 0
                        && reference_x < input.shape.width as i128;
                    reference_valid[site] = ref_ok;
                    for channel in 0..channels_per_group {
                        let cache_index = feature_index(
                            batch,
                            input_channel_base + channel,
                            oy,
                            ox,
                            channels,
                            output_height,
                            output_width,
                        );
                        if ref_ok {
                            reference[cache_index] = input.get(
                                batch,
                                input_channel_base + channel,
                                reference_y as usize,
                                reference_x as usize,
                            );
                        }
                    }

                    let edge_site = site * kernel_area;
                    let mut maximum = f32::NEG_INFINITY;
                    let mut valid_count = 0usize;
                    for ky in 0..spec.size {
                        for kx in 0..spec.size {
                            let edge = ky * spec.size + kx;
                            let input_y = source_coordinate(
                                oy,
                                spec.stride_y,
                                spec.padding,
                                ky,
                                spec.dilation,
                            );
                            let input_x = source_coordinate(
                                ox,
                                spec.stride_x,
                                spec.padding,
                                kx,
                                spec.dilation,
                            );
                            if input_y < 0
                                || input_y >= input.shape.height as i128
                                || input_x < 0
                                || input_x >= input.shape.width as i128
                            {
                                continue;
                            }
                            valid[edge_site + edge] = true;
                            valid_count += 1;
                            if spec.edge_mode == 1 {
                                let kernel = edge_kernel.expect("learned edges validated");
                                let edge_param =
                                    (group * kernel_area + edge) * 2 * channels_per_group;
                                let mut logit = edge_biases.expect("learned edges validated")
                                    [group * kernel_area + edge];
                                for channel in 0..channels_per_group {
                                    let reference_value = reference[feature_index(
                                        batch,
                                        input_channel_base + channel,
                                        oy,
                                        ox,
                                        channels,
                                        output_height,
                                        output_width,
                                    )];
                                    let neighbor_value = input.get(
                                        batch,
                                        input_channel_base + channel,
                                        input_y as usize,
                                        input_x as usize,
                                    );
                                    logit += kernel[edge_param + channel] * reference_value
                                        + kernel[edge_param + channels_per_group + channel]
                                            * neighbor_value;
                                }
                                alpha[edge_site + edge] = logit;
                                maximum = maximum.max(logit);
                            }
                        }
                    }
                    if valid_count == 0 {
                        continue;
                    }
                    if spec.edge_mode == 0 {
                        let weight = 1.0 / valid_count as f32;
                        for edge in 0..kernel_area {
                            if valid[edge_site + edge] {
                                alpha[edge_site + edge] = weight;
                            }
                        }
                    } else {
                        let mut sum = 0.0;
                        for edge in 0..kernel_area {
                            let index = edge_site + edge;
                            if valid[index] {
                                alpha[index] = (alpha[index] - maximum).exp();
                                sum += alpha[index];
                            }
                        }
                        for edge in 0..kernel_area {
                            let index = edge_site + edge;
                            if valid[index] {
                                alpha[index] /= sum;
                            }
                        }
                    }

                    for channel in 0..channels_per_group {
                        let mut value = 0.0;
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let edge = ky * spec.size + kx;
                                let edge_cache_index = edge_site + edge;
                                if !valid[edge_cache_index] {
                                    continue;
                                }
                                let input_y = source_coordinate(
                                    oy,
                                    spec.stride_y,
                                    spec.padding,
                                    ky,
                                    spec.dilation,
                                ) as usize;
                                let input_x = source_coordinate(
                                    ox,
                                    spec.stride_x,
                                    spec.padding,
                                    kx,
                                    spec.dilation,
                                ) as usize;
                                value += alpha[edge_cache_index]
                                    * input.get(
                                        batch,
                                        input_channel_base + channel,
                                        input_y,
                                        input_x,
                                    );
                            }
                        }
                        let cache_index = feature_index(
                            batch,
                            input_channel_base + channel,
                            oy,
                            ox,
                            channels,
                            output_height,
                            output_width,
                        );
                        aggregate[cache_index] = value;
                    }
                }
            }
        }
    }

    for batch in 0..input.shape.batch {
        for group in 0..groups {
            let input_channel_base = group * channels_per_group;
            let output_channel_base = group * filters_per_group;
            for oy in 0..output_height {
                for ox in 0..output_width {
                    let site =
                        site_index(batch, group, oy, ox, groups, output_height, output_width);
                    if alpha[site * kernel_area..(site + 1) * kernel_area]
                        .iter()
                        .all(|value| *value == 0.0)
                    {
                        // This is a valid no-neighbor site: its graph branch is
                        // zero, while a valid self projection still contributes.
                    }
                    for output in 0..filters_per_group {
                        let out_channel = output_channel_base + output;
                        let mut value = 0.0;
                        for channel in 0..channels_per_group {
                            let cache_index = feature_index(
                                batch,
                                input_channel_base + channel,
                                oy,
                                ox,
                                channels,
                                output_height,
                                output_width,
                            );
                            let weight = weights[out_channel * channels_per_group + channel];
                            value += weight * aggregate[cache_index];
                            if let Some(self_weights) = self_weights {
                                value += self_weights[out_channel * channels_per_group + channel]
                                    * reference[cache_index];
                            }
                        }
                        raw_output.data[output_index(
                            batch,
                            out_channel,
                            oy,
                            ox,
                            spec.filters,
                            output_height,
                            output_width,
                        )] = value;
                    }
                    let _ = reference_valid[site];
                }
            }
        }
    }

    Ok(GraphConvForwardCache {
        input_shape: input.shape,
        output_shape,
        raw_output,
        reference,
        aggregate,
        alpha,
        valid,
        reference_valid,
    })
}

/// Differentiate the exact Rust forward math. `output_gradient` is the gradient
/// with respect to `raw_output`, after the runtime's bias/BN/activation path.
pub(crate) fn graph_conv_backward(
    spec: &GraphConvSpec,
    input: &Tensor,
    weights: &[f32],
    self_weights: Option<&[f32]>,
    edge_kernel: Option<&[f32]>,
    output_gradient: &Tensor,
    cache: &GraphConvForwardCache,
) -> Result<GraphConvGradients> {
    if cache.input_shape != input.shape || cache.output_shape != output_gradient.shape {
        return Err(Error::InvalidConfig(format!(
            "[graph_conv] backward cache/gradient shape mismatch: input {}, cached {}, gradient {}, expected {}",
            input.shape, cache.input_shape, output_gradient.shape, cache.output_shape
        )));
    }
    let (groups, channels_per_group, filters_per_group, kernel_area, _) =
        validate_geometry(spec, input.shape, cache.output_shape)?;
    let channels = input.shape.channels;
    let output_height = cache.output_shape.height;
    let output_width = cache.output_shape.width;
    let expected_weights = spec.filters * channels_per_group;
    if weights.len() != expected_weights
        || output_gradient.data.len() != cache.raw_output.data.len()
    {
        return Err(Error::InvalidConfig(
            "[graph_conv] backward parameter or output-gradient length mismatch".to_owned(),
        ));
    }
    if spec.use_self != self_weights.is_some()
        || (spec.use_self && self_weights.is_some_and(|values| values.len() != expected_weights))
        || (spec.edge_mode == 1
            && edge_kernel
                .is_none_or(|values| values.len() != groups * kernel_area * 2 * channels_per_group))
        || (spec.edge_mode == 0 && edge_kernel.is_some())
    {
        return Err(Error::InvalidConfig(
            "[graph_conv] backward parameter block mismatch".to_owned(),
        ));
    }
    let mut input_gradient = vec![0.0; input.data.len()];
    let mut weight_gradient = vec![0.0; weights.len()];
    let mut self_gradient = self_weights.map(|values| vec![0.0; values.len()]);
    let mut edge_gradient = edge_kernel.map(|values| vec![0.0; values.len()]);
    let mut edge_bias_gradient = (spec.edge_mode == 1).then(|| vec![0.0; groups * kernel_area]);
    let mut aggregate_gradient = vec![0.0; cache.aggregate.len()];
    let mut reference_gradient = vec![0.0; cache.reference.len()];
    let center = spec.size / 2;

    for batch in 0..input.shape.batch {
        for group in 0..groups {
            let input_channel_base = group * channels_per_group;
            let output_channel_base = group * filters_per_group;
            for oy in 0..output_height {
                for ox in 0..output_width {
                    let site =
                        site_index(batch, group, oy, ox, groups, output_height, output_width);
                    let ref_ok = cache.reference_valid[site];
                    for output in 0..filters_per_group {
                        let out_channel = output_channel_base + output;
                        let delta = output_gradient.data[output_index(
                            batch,
                            out_channel,
                            oy,
                            ox,
                            spec.filters,
                            output_height,
                            output_width,
                        )];
                        for channel in 0..channels_per_group {
                            let feature = feature_index(
                                batch,
                                input_channel_base + channel,
                                oy,
                                ox,
                                channels,
                                output_height,
                                output_width,
                            );
                            let parameter = out_channel * channels_per_group + channel;
                            weight_gradient[parameter] += delta * cache.aggregate[feature];
                            aggregate_gradient[feature] += delta * weights[parameter];
                            if let (Some(values), Some(gradient)) =
                                (self_weights, self_gradient.as_mut())
                            {
                                gradient[parameter] += delta * cache.reference[feature];
                                if ref_ok {
                                    reference_gradient[feature] += delta * values[parameter];
                                }
                            }
                        }
                    }

                    let edge_site = site * kernel_area;
                    let mut alpha_gradient = vec![0.0; kernel_area];
                    let mut softmax_dot = 0.0;
                    for ky in 0..spec.size {
                        for kx in 0..spec.size {
                            let edge = ky * spec.size + kx;
                            let edge_cache_index = edge_site + edge;
                            if !cache.valid[edge_cache_index] {
                                continue;
                            }
                            let input_y = source_coordinate(
                                oy,
                                spec.stride_y,
                                spec.padding,
                                ky,
                                spec.dilation,
                            ) as usize;
                            let input_x = source_coordinate(
                                ox,
                                spec.stride_x,
                                spec.padding,
                                kx,
                                spec.dilation,
                            ) as usize;
                            let alpha = cache.alpha[edge_cache_index];
                            for channel in 0..channels_per_group {
                                let feature = feature_index(
                                    batch,
                                    input_channel_base + channel,
                                    oy,
                                    ox,
                                    channels,
                                    output_height,
                                    output_width,
                                );
                                let neighbor_value = input.get(
                                    batch,
                                    input_channel_base + channel,
                                    input_y,
                                    input_x,
                                );
                                let d_aggregate = aggregate_gradient[feature];
                                let input_index = input.offset(
                                    batch,
                                    input_channel_base + channel,
                                    input_y,
                                    input_x,
                                );
                                input_gradient[input_index] += alpha * d_aggregate;
                                alpha_gradient[edge] += d_aggregate * neighbor_value;
                            }
                            softmax_dot += alpha * alpha_gradient[edge];
                        }
                    }

                    if spec.edge_mode == 1 {
                        let edge_kernel = edge_kernel.expect("learned edges validated");
                        let edge_gradient =
                            edge_gradient.as_mut().expect("learned gradient exists");
                        let edge_bias_gradient = edge_bias_gradient
                            .as_mut()
                            .expect("learned edge-bias gradient exists");
                        for ky in 0..spec.size {
                            for kx in 0..spec.size {
                                let edge = ky * spec.size + kx;
                                let edge_cache_index = edge_site + edge;
                                if !cache.valid[edge_cache_index] {
                                    continue;
                                }
                                let input_y = source_coordinate(
                                    oy,
                                    spec.stride_y,
                                    spec.padding,
                                    ky,
                                    spec.dilation,
                                ) as usize;
                                let input_x = source_coordinate(
                                    ox,
                                    spec.stride_x,
                                    spec.padding,
                                    kx,
                                    spec.dilation,
                                ) as usize;
                                let edge_param =
                                    (group * kernel_area + edge) * 2 * channels_per_group;
                                let d_logit = cache.alpha[edge_cache_index]
                                    * (alpha_gradient[edge] - softmax_dot);
                                edge_bias_gradient[group * kernel_area + edge] += d_logit;
                                for channel in 0..channels_per_group {
                                    let feature = feature_index(
                                        batch,
                                        input_channel_base + channel,
                                        oy,
                                        ox,
                                        channels,
                                        output_height,
                                        output_width,
                                    );
                                    let neighbor_index = input.offset(
                                        batch,
                                        input_channel_base + channel,
                                        input_y,
                                        input_x,
                                    );
                                    edge_gradient[edge_param + channel] +=
                                        d_logit * cache.reference[feature];
                                    edge_gradient[edge_param + channels_per_group + channel] +=
                                        d_logit
                                            * input.get(
                                                batch,
                                                input_channel_base + channel,
                                                input_y,
                                                input_x,
                                            );
                                    if ref_ok {
                                        input_gradient[input.offset(
                                            batch,
                                            input_channel_base + channel,
                                            source_coordinate(
                                                oy,
                                                spec.stride_y,
                                                spec.padding,
                                                center,
                                                spec.dilation,
                                            ) as usize,
                                            source_coordinate(
                                                ox,
                                                spec.stride_x,
                                                spec.padding,
                                                center,
                                                spec.dilation,
                                            ) as usize,
                                        )] += d_logit * edge_kernel[edge_param + channel];
                                    }
                                    input_gradient[neighbor_index] += d_logit
                                        * edge_kernel[edge_param + channels_per_group + channel];
                                }
                            }
                        }
                    }

                    if ref_ok {
                        let reference_y = source_coordinate(
                            oy,
                            spec.stride_y,
                            spec.padding,
                            center,
                            spec.dilation,
                        ) as usize;
                        let reference_x = source_coordinate(
                            ox,
                            spec.stride_x,
                            spec.padding,
                            center,
                            spec.dilation,
                        ) as usize;
                        for channel in 0..channels_per_group {
                            let feature = feature_index(
                                batch,
                                input_channel_base + channel,
                                oy,
                                ox,
                                channels,
                                output_height,
                                output_width,
                            );
                            let index = input.offset(
                                batch,
                                input_channel_base + channel,
                                reference_y,
                                reference_x,
                            );
                            input_gradient[index] += reference_gradient[feature];
                        }
                    }
                }
            }
        }
    }

    Ok(GraphConvGradients {
        input: input_gradient,
        weights: weight_gradient,
        self_weights: self_gradient,
        edge_kernel: edge_gradient,
        edge_biases: edge_bias_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::{graph_conv_backward, graph_conv_forward};
    use crate::model::{Activation, GraphConvSpec, TensorShape};
    use crate::runtime::Tensor;

    fn spec(
        _channels: usize,
        filters: usize,
        groups: usize,
        size: usize,
        padding: usize,
        edge_mode: usize,
        use_self: bool,
    ) -> GraphConvSpec {
        GraphConvSpec {
            filters,
            groups,
            size,
            stride_x: 1,
            stride_y: 1,
            dilation: 1,
            padding,
            edge_mode,
            use_self,
            batch_normalize: false,
            activation: Activation::Linear,
        }
    }

    #[test]
    fn graph_conv_source_nchw_axes_match_hand_computation() {
        let spec = spec(2, 2, 1, 3, 1, 0, false);
        let input = Tensor::from_data(
            TensorShape::new(3, 2, 2, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        )
        .unwrap();
        let cache = graph_conv_forward(
            &spec,
            &input,
            TensorShape::new(3, 2, 2, 1),
            &[1.0, 0.0, 0.0, 1.0],
            None,
            None,
            None,
        )
        .unwrap();
        let expected = [
            3.0, 3.5, 4.0, 3.0, 3.5, 4.0, 13.0, 13.5, 14.0, 13.0, 13.5, 14.0,
        ];
        for (actual, expected) in cache.raw_output.data.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-6);
        }
    }

    #[test]
    fn grouped_projection_keeps_nchw_channels_inside_each_group() {
        let spec = spec(4, 4, 2, 1, 0, 0, false);
        let input = Tensor::from_data(
            TensorShape::new(2, 1, 4, 1),
            vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 1000.0, 2000.0],
        )
        .unwrap();
        let cache = graph_conv_forward(
            &spec,
            &input,
            TensorShape::new(2, 1, 4, 1),
            &[2.0, 3.0, -1.0, 1.0, 4.0, -1.0, 0.5, 0.25],
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            cache.raw_output.data,
            vec![32.0, 64.0, 9.0, 18.0, -600.0, -1200.0, 300.0, 600.0]
        );
    }

    #[test]
    fn learned_edge_softmax_masks_invalid_neighbors_and_uses_row_major_edges() {
        let spec = spec(1, 1, 1, 3, 1, 1, false);
        let input =
            Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut edge_biases = vec![0.0; 9];
        edge_biases[4] = 0.0;
        edge_biases[5] = 2.0f32.ln();
        edge_biases[7] = 3.0f32.ln();
        edge_biases[8] = 4.0f32.ln();
        let cache = graph_conv_forward(
            &spec,
            &input,
            TensorShape::new(2, 2, 1, 1),
            &[1.0],
            None,
            Some(&vec![0.0; 18]),
            Some(&edge_biases),
        )
        .unwrap();
        assert!((cache.raw_output.data[0] - 3.0).abs() < 1e-6);
    }

    fn objective(
        spec: &GraphConvSpec,
        input: &Tensor,
        weights: &[f32],
        self_weights: Option<&[f32]>,
        edge_kernel: Option<&[f32]>,
        edge_biases: Option<&[f32]>,
        seed: &[f32],
    ) -> f32 {
        let shape = TensorShape::new(
            input.shape.width + 2 * spec.padding - spec.dilation * (spec.size - 1),
            input.shape.height + 2 * spec.padding - spec.dilation * (spec.size - 1),
            spec.filters,
            input.shape.batch,
        );
        let output = graph_conv_forward(
            spec,
            input,
            shape,
            weights,
            self_weights,
            edge_kernel,
            edge_biases,
        )
        .unwrap();
        output
            .raw_output
            .data
            .iter()
            .zip(seed)
            .map(|(value, grad)| value * grad)
            .sum()
    }

    fn finite_difference(values: &[f32], mut eval: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
        let epsilon = 1.0e-3;
        (0..values.len())
            .map(|index| {
                let mut plus = values.to_vec();
                let mut minus = values.to_vec();
                plus[index] += epsilon;
                minus[index] -= epsilon;
                (eval(&plus) - eval(&minus)) / (2.0 * epsilon)
            })
            .collect()
    }

    #[test]
    fn uniform_edge_backward_matches_boundary_finite_differences_and_zero_seed() {
        let spec = spec(1, 1, 1, 3, 1, 0, true);
        let input =
            Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![0.2, -0.4, 0.7, 1.1]).unwrap();
        let weights = vec![0.35];
        let self_weights = vec![-0.2];
        let seed = vec![0.3, -0.2, 0.45, -0.35];
        let shape = TensorShape::new(2, 2, 1, 1);
        let cache = graph_conv_forward(
            &spec,
            &input,
            shape,
            &weights,
            Some(&self_weights),
            None,
            None,
        )
        .unwrap();
        let output_gradient = Tensor::from_data(shape, seed.clone()).unwrap();
        let analytic = graph_conv_backward(
            &spec,
            &input,
            &weights,
            Some(&self_weights),
            None,
            &output_gradient,
            &cache,
        )
        .unwrap();
        let numeric_input = finite_difference(&input.data, |values| {
            let perturbed = Tensor::from_data(input.shape, values.to_vec()).unwrap();
            objective(
                &spec,
                &perturbed,
                &weights,
                Some(&self_weights),
                None,
                None,
                &seed,
            )
        });
        let numeric_weights = finite_difference(&weights, |values| {
            objective(
                &spec,
                &input,
                values,
                Some(&self_weights),
                None,
                None,
                &seed,
            )
        });
        let numeric_self = finite_difference(&self_weights, |values| {
            objective(&spec, &input, &weights, Some(values), None, None, &seed)
        });
        for (actual, expected) in analytic
            .input
            .iter()
            .chain(&analytic.weights)
            .chain(analytic.self_weights.as_ref().unwrap())
            .zip(
                numeric_input
                    .iter()
                    .chain(&numeric_weights)
                    .chain(&numeric_self),
            )
        {
            assert!(
                (actual - expected).abs() < 1.0e-3,
                "analytic={actual}, numeric={expected}"
            );
        }
        let zero_seed = Tensor::from_data(shape, vec![0.0; 4]).unwrap();
        let zero = graph_conv_backward(
            &spec,
            &input,
            &weights,
            Some(&self_weights),
            None,
            &zero_seed,
            &cache,
        )
        .unwrap();
        assert!(zero.input.iter().all(|value| *value == 0.0));
        assert!(zero.weights.iter().all(|value| *value == 0.0));
        assert!(zero.self_weights.unwrap().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn learned_edge_grouped_backward_matches_finite_differences_for_all_parameter_blocks() {
        let spec = spec(4, 4, 2, 3, 1, 1, true);
        let input = Tensor::from_data(
            TensorShape::new(2, 2, 4, 1),
            vec![
                0.2, -0.4, 0.7, 1.1, -0.3, 0.5, -0.8, 0.6, 0.9, -0.2, 0.4, -0.7, 0.3, 0.8, -0.5,
                0.1,
            ],
        )
        .unwrap();
        let weights = vec![0.15, -0.2, 0.3, 0.1, -0.12, 0.22, 0.17, -0.08];
        let self_weights = vec![-0.1, 0.25, 0.2, -0.15, 0.18, -0.07, 0.11, 0.09];
        let edge_kernel = (0..72)
            .map(|i| (i as f32 - 35.0) * 0.007)
            .collect::<Vec<_>>();
        let edge_biases = (0..18).map(|i| (i as f32 - 8.0) * 0.04).collect::<Vec<_>>();
        let seed = (0..16).map(|i| (i as f32 - 7.0) * 0.07).collect::<Vec<_>>();
        let shape = TensorShape::new(2, 2, 4, 1);
        let cache = graph_conv_forward(
            &spec,
            &input,
            shape,
            &weights,
            Some(&self_weights),
            Some(&edge_kernel),
            Some(&edge_biases),
        )
        .unwrap();
        let output_gradient = Tensor::from_data(shape, seed.clone()).unwrap();
        let analytic = graph_conv_backward(
            &spec,
            &input,
            &weights,
            Some(&self_weights),
            Some(&edge_kernel),
            &output_gradient,
            &cache,
        )
        .unwrap();
        let numeric_input = finite_difference(&input.data, |values| {
            let input = Tensor::from_data(input.shape, values.to_vec()).unwrap();
            objective(
                &spec,
                &input,
                &weights,
                Some(&self_weights),
                Some(&edge_kernel),
                Some(&edge_biases),
                &seed,
            )
        });
        let numeric_weights = finite_difference(&weights, |values| {
            objective(
                &spec,
                &input,
                values,
                Some(&self_weights),
                Some(&edge_kernel),
                Some(&edge_biases),
                &seed,
            )
        });
        let numeric_self = finite_difference(&self_weights, |values| {
            objective(
                &spec,
                &input,
                &weights,
                Some(values),
                Some(&edge_kernel),
                Some(&edge_biases),
                &seed,
            )
        });
        let numeric_kernel = finite_difference(&edge_kernel, |values| {
            objective(
                &spec,
                &input,
                &weights,
                Some(&self_weights),
                Some(values),
                Some(&edge_biases),
                &seed,
            )
        });
        let numeric_biases = finite_difference(&edge_biases, |values| {
            objective(
                &spec,
                &input,
                &weights,
                Some(&self_weights),
                Some(&edge_kernel),
                Some(values),
                &seed,
            )
        });
        for (analytic, numeric) in [
            (&analytic.input, &numeric_input),
            (&analytic.weights, &numeric_weights),
            (analytic.self_weights.as_ref().unwrap(), &numeric_self),
            (analytic.edge_kernel.as_ref().unwrap(), &numeric_kernel),
            (analytic.edge_biases.as_ref().unwrap(), &numeric_biases),
        ] {
            assert_eq!(analytic.len(), numeric.len());
            for (actual, expected) in analytic.iter().zip(numeric) {
                assert!(
                    (actual - expected).abs() < 2.0e-3,
                    "analytic={actual}, numeric={expected}"
                );
            }
        }
    }
}
