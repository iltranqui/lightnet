//! Uniform valid-neighbor aggregation used by `[graph_conv]` edge mode 0.
//!
//! This is one primitive in the layer pipeline, not the full graph-convolution
//! implementation. Tensors use NCHW layout and the reference value is the
//! arithmetic mean of the valid spatial neighbors.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphConvAggregationGeometry {
    pub batch: usize,
    pub channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
}

impl GraphConvAggregationGeometry {
    pub fn checked(self) -> Result<Self, String> {
        if self.batch == 0
            || self.channels == 0
            || self.input_height == 0
            || self.input_width == 0
            || self.output_height == 0
            || self.output_width == 0
            || self.size == 0
            || self.size % 2 == 0
            || self.stride_x == 0
            || self.stride_y == 0
            || self.dilation == 0
        {
            return Err("graph_conv aggregation has invalid geometry".to_owned());
        }
        let output_dim = |input: usize, stride: usize| -> Option<usize> {
            let numerator = input as i128 + 2 * self.padding as i128
                - self.dilation as i128 * (self.size - 1) as i128
                - 1;
            (numerator >= 0)
                .then(|| usize::try_from(numerator / stride as i128 + 1).ok())
                .flatten()
        };
        if output_dim(self.input_height, self.stride_y) != Some(self.output_height)
            || output_dim(self.input_width, self.stride_x) != Some(self.output_width)
        {
            return Err("graph_conv aggregation output shape mismatch".to_owned());
        }
        for value in [
            self.batch,
            self.channels,
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
            self.size,
            self.stride_x,
            self.stride_y,
            self.dilation,
            self.padding,
        ] {
            if value > i64::MAX as usize {
                return Err("graph_conv aggregation geometry exceeds i64 indexing".to_owned());
            }
        }
        let kernel_extent = (self.size - 1).checked_mul(self.dilation);
        let max_y = (self.output_height - 1)
            .checked_mul(self.stride_y)
            .and_then(|v| v.checked_add(kernel_extent?));
        let max_x = (self.output_width - 1)
            .checked_mul(self.stride_x)
            .and_then(|v| v.checked_add(kernel_extent?));
        let inverse_y = (self.input_height - 1).checked_add(self.padding);
        let inverse_x = (self.input_width - 1).checked_add(self.padding);
        if [kernel_extent, max_y, max_x, inverse_y, inverse_x]
            .into_iter()
            .flatten()
            .any(|value| value > i64::MAX as usize)
            || kernel_extent.is_none()
            || max_y.is_none()
            || max_x.is_none()
            || inverse_y.is_none()
            || inverse_x.is_none()
        {
            return Err("graph_conv aggregation coordinate arithmetic overflows i64".to_owned());
        }
        let input_len = self
            .batch
            .checked_mul(self.channels)
            .and_then(|n| n.checked_mul(self.input_height))
            .and_then(|n| n.checked_mul(self.input_width));
        let output_len = self
            .batch
            .checked_mul(self.channels)
            .and_then(|n| n.checked_mul(self.output_height))
            .and_then(|n| n.checked_mul(self.output_width));
        let kernel_area = self.size.checked_mul(self.size);
        if input_len.is_none() || output_len.is_none() || kernel_area.is_none() {
            return Err("graph_conv aggregation tensor size overflows usize".to_owned());
        }
        Ok(self)
    }

    pub fn input_len(self) -> usize {
        self.batch * self.channels * self.input_height * self.input_width
    }

    pub fn output_len(self) -> usize {
        self.batch * self.channels * self.output_height * self.output_width
    }
}

fn valid_neighbor_count(g: GraphConvAggregationGeometry, oy: usize, ox: usize) -> usize {
    let mut count = 0;
    for ky in 0..g.size {
        let iy = oy as i64 * g.stride_y as i64 - g.padding as i64 + ky as i64 * g.dilation as i64;
        for kx in 0..g.size {
            let ix =
                ox as i64 * g.stride_x as i64 - g.padding as i64 + kx as i64 * g.dilation as i64;
            if iy >= 0 && iy < g.input_height as i64 && ix >= 0 && ix < g.input_width as i64 {
                count += 1;
            }
        }
    }
    count
}

pub fn aggregate_forward(
    input: &[f32],
    geometry: GraphConvAggregationGeometry,
) -> Result<Vec<f32>, String> {
    let g = geometry.checked()?;
    if input.len() != g.input_len() {
        return Err(format!(
            "graph_conv aggregate input length mismatch: expected {}, got {}",
            g.input_len(),
            input.len()
        ));
    }
    let mut output = vec![0.0; g.output_len()];
    for b in 0..g.batch {
        for c in 0..g.channels {
            for oy in 0..g.output_height {
                for ox in 0..g.output_width {
                    let mut sum = 0.0;
                    let mut count = 0usize;
                    for ky in 0..g.size {
                        let iy = oy as i64 * g.stride_y as i64 - g.padding as i64
                            + ky as i64 * g.dilation as i64;
                        for kx in 0..g.size {
                            let ix = ox as i64 * g.stride_x as i64 - g.padding as i64
                                + kx as i64 * g.dilation as i64;
                            if iy < 0
                                || iy >= g.input_height as i64
                                || ix < 0
                                || ix >= g.input_width as i64
                            {
                                continue;
                            }
                            let index = ((b * g.channels + c) * g.input_height + iy as usize)
                                * g.input_width
                                + ix as usize;
                            sum += input[index];
                            count += 1;
                        }
                    }
                    if count != 0 {
                        let output_index =
                            ((b * g.channels + c) * g.output_height + oy) * g.output_width + ox;
                        output[output_index] = sum / count as f32;
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Exact adjoint of [`aggregate_forward`], gathering contributions per input
/// element so device implementations need no floating-point atomics.
pub fn aggregate_backward(
    output_gradient: &[f32],
    geometry: GraphConvAggregationGeometry,
) -> Result<Vec<f32>, String> {
    let g = geometry.checked()?;
    if output_gradient.len() != g.output_len() {
        return Err(format!(
            "graph_conv aggregate gradient length mismatch: expected {}, got {}",
            g.output_len(),
            output_gradient.len()
        ));
    }
    let mut input_gradient = vec![0.0; g.input_len()];
    for b in 0..g.batch {
        for c in 0..g.channels {
            for iy in 0..g.input_height {
                for ix in 0..g.input_width {
                    let mut sum = 0.0;
                    for ky in 0..g.size {
                        let oy_num = iy as i64 + g.padding as i64 - ky as i64 * g.dilation as i64;
                        if oy_num < 0 || oy_num % g.stride_y as i64 != 0 {
                            continue;
                        }
                        let oy = (oy_num / g.stride_y as i64) as usize;
                        if oy >= g.output_height {
                            continue;
                        }
                        for kx in 0..g.size {
                            let ox_num =
                                ix as i64 + g.padding as i64 - kx as i64 * g.dilation as i64;
                            if ox_num < 0 || ox_num % g.stride_x as i64 != 0 {
                                continue;
                            }
                            let ox = (ox_num / g.stride_x as i64) as usize;
                            if ox >= g.output_width {
                                continue;
                            }
                            let count = valid_neighbor_count(g, oy, ox);
                            if count != 0 {
                                let output_index = ((b * g.channels + c) * g.output_height + oy)
                                    * g.output_width
                                    + ox;
                                sum += output_gradient[output_index] / count as f32;
                            }
                        }
                    }
                    let input_index =
                        ((b * g.channels + c) * g.input_height + iy) * g.input_width + ix;
                    input_gradient[input_index] = sum;
                }
            }
        }
    }
    Ok(input_gradient)
}

/// Gather the center/reference feature used by GraphConv's optional self
/// projection. Out-of-bounds center coordinates produce zero.
pub fn center_reference_forward(
    input: &[f32],
    geometry: GraphConvAggregationGeometry,
) -> Result<Vec<f32>, String> {
    let g = geometry.checked()?;
    if input.len() != g.input_len() {
        return Err(format!(
            "graph_conv reference input length mismatch: expected {}, got {}",
            g.input_len(),
            input.len()
        ));
    }
    let mut reference = vec![0.0; g.output_len()];
    let center = g.size / 2;
    for batch in 0..g.batch {
        for channel in 0..g.channels {
            for oy in 0..g.output_height {
                let iy = oy as i64 * g.stride_y as i64 - g.padding as i64
                    + center as i64 * g.dilation as i64;
                if iy < 0 || iy >= g.input_height as i64 {
                    continue;
                }
                for ox in 0..g.output_width {
                    let ix = ox as i64 * g.stride_x as i64 - g.padding as i64
                        + center as i64 * g.dilation as i64;
                    if ix < 0 || ix >= g.input_width as i64 {
                        continue;
                    }
                    let input_index =
                        ((batch * g.channels + channel) * g.input_height + iy as usize)
                            * g.input_width
                            + ix as usize;
                    let output_index =
                        ((batch * g.channels + channel) * g.output_height + oy) * g.output_width
                            + ox;
                    reference[output_index] = input[input_index];
                }
            }
        }
    }
    Ok(reference)
}

/// Exact adjoint of [`center_reference_forward`].
pub fn center_reference_backward(
    output_gradient: &[f32],
    geometry: GraphConvAggregationGeometry,
) -> Result<Vec<f32>, String> {
    let g = geometry.checked()?;
    if output_gradient.len() != g.output_len() {
        return Err(format!(
            "graph_conv reference gradient length mismatch: expected {}, got {}",
            g.output_len(),
            output_gradient.len()
        ));
    }
    let mut input_gradient = vec![0.0; g.input_len()];
    let center_offset = (g.size / 2) as i64 * g.dilation as i64;
    for batch in 0..g.batch {
        for channel in 0..g.channels {
            for iy in 0..g.input_height {
                let oy_num = iy as i64 + g.padding as i64 - center_offset;
                if oy_num < 0 || oy_num % g.stride_y as i64 != 0 {
                    continue;
                }
                let oy = (oy_num / g.stride_y as i64) as usize;
                if oy >= g.output_height {
                    continue;
                }
                for ix in 0..g.input_width {
                    let ox_num = ix as i64 + g.padding as i64 - center_offset;
                    if ox_num < 0 || ox_num % g.stride_x as i64 != 0 {
                        continue;
                    }
                    let ox = (ox_num / g.stride_x as i64) as usize;
                    if ox >= g.output_width {
                        continue;
                    }
                    let input_index =
                        ((batch * g.channels + channel) * g.input_height + iy) * g.input_width
                            + ix;
                    let output_index =
                        ((batch * g.channels + channel) * g.output_height + oy) * g.output_width
                            + ox;
                    input_gradient[input_index] = output_gradient[output_index];
                }
            }
        }
    }
    Ok(input_gradient)
}

/// Validate the learned-edge aggregation layout. Edge kernels are row-major
/// `[groups, size*size, 2*channels_per_group]` (reference then neighbor).
fn checked_learned_geometry(
    geometry: GraphConvAggregationGeometry,
    groups: usize,
    edge_kernel: &[f32],
    edge_biases: &[f32],
) -> Result<(GraphConvAggregationGeometry, usize, usize), String> {
    let g = geometry.checked()?;
    if groups == 0 || g.channels % groups != 0 {
        return Err("graph_conv learned aggregation has invalid groups".to_owned());
    }
    let channels_per_group = g.channels / groups;
    let kernel_area = g
        .size
        .checked_mul(g.size)
        .ok_or_else(|| "graph_conv learned kernel area overflows usize".to_owned())?;
    let edge_count = groups
        .checked_mul(kernel_area)
        .ok_or_else(|| "graph_conv learned edge count overflows usize".to_owned())?;
    let expected_kernel = edge_count
        .checked_mul(channels_per_group)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| "graph_conv learned kernel length overflows usize".to_owned())?;
    if edge_kernel.len() != expected_kernel || edge_biases.len() != edge_count {
        return Err(format!(
            "graph_conv learned parameter length mismatch: expected kernel {expected_kernel}, bias {edge_count}; got kernel {}, bias {}",
            edge_kernel.len(),
            edge_biases.len()
        ));
    }
    Ok((g, channels_per_group, kernel_area))
}

#[inline]
fn learned_input_index(
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
fn learned_output_index(
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
fn learned_edge_coordinate(
    g: GraphConvAggregationGeometry,
    oy: usize,
    ox: usize,
    edge: usize,
) -> Option<(usize, usize)> {
    let ky = edge / g.size;
    let kx = edge % g.size;
    let iy = oy as i64 * g.stride_y as i64 - g.padding as i64 + ky as i64 * g.dilation as i64;
    let ix = ox as i64 * g.stride_x as i64 - g.padding as i64 + kx as i64 * g.dilation as i64;
    (iy >= 0 && iy < g.input_height as i64 && ix >= 0 && ix < g.input_width as i64)
        .then_some((iy as usize, ix as usize))
}

fn learned_reference_coordinate(
    g: GraphConvAggregationGeometry,
    oy: usize,
    ox: usize,
) -> Option<(usize, usize)> {
    learned_edge_coordinate(g, oy, ox, (g.size / 2) * g.size + g.size / 2)
}

fn learned_edge_alpha(
    input: &[f32],
    g: GraphConvAggregationGeometry,
    _groups: usize,
    channels_per_group: usize,
    kernel_area: usize,
    batch: usize,
    group: usize,
    oy: usize,
    ox: usize,
    edge_kernel: &[f32],
    edge_biases: &[f32],
) -> Vec<f32> {
    let channel_base = group * channels_per_group;
    let mut logits = vec![f32::NEG_INFINITY; kernel_area];
    let mut maximum = f32::NEG_INFINITY;
    for (edge, logit_slot) in logits.iter_mut().enumerate() {
        let Some((iy, ix)) = learned_edge_coordinate(g, oy, ox, edge) else {
            continue;
        };
        let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
        let reference = learned_reference_coordinate(g, oy, ox);
        let mut logit = edge_biases[group * kernel_area + edge];
        for channel in 0..channels_per_group {
            let reference_value = reference.map_or(0.0, |(ry, rx)| {
                input[learned_input_index(
                    batch,
                    channel_base + channel,
                    ry,
                    rx,
                    g.channels,
                    g.input_height,
                    g.input_width,
                )]
            });
            let neighbor_value = input[learned_input_index(
                batch,
                channel_base + channel,
                iy,
                ix,
                g.channels,
                g.input_height,
                g.input_width,
            )];
            logit += edge_kernel[parameter + channel] * reference_value
                + edge_kernel[parameter + channels_per_group + channel] * neighbor_value;
        }
        *logit_slot = logit;
        maximum = maximum.max(logit);
    }
    let mut sum = 0.0;
    for (edge, logit) in logits.iter_mut().enumerate() {
        if learned_edge_coordinate(g, oy, ox, edge).is_some() {
            *logit = (*logit - maximum).exp();
            sum += *logit;
        } else {
            *logit = 0.0;
        }
    }
    if sum > 0.0 {
        for alpha in &mut logits {
            *alpha /= sum;
        }
    }
    logits
}

/// Learned-edge softmax aggregation for edge mode 1. Invalid spatial edges are
/// masked before normalization; edge weights are group-local and row-major.
pub fn learned_aggregate_forward(
    input: &[f32],
    edge_kernel: &[f32],
    edge_biases: &[f32],
    geometry: GraphConvAggregationGeometry,
    groups: usize,
) -> Result<Vec<f32>, String> {
    let (g, channels_per_group, kernel_area) =
        checked_learned_geometry(geometry, groups, edge_kernel, edge_biases)?;
    if input.len() != g.input_len() {
        return Err(format!(
            "graph_conv learned input length mismatch: expected {}, got {}",
            g.input_len(),
            input.len()
        ));
    }
    let mut aggregate = vec![0.0; g.output_len()];
    for batch in 0..g.batch {
        for group in 0..groups {
            for oy in 0..g.output_height {
                for ox in 0..g.output_width {
                    let alpha = learned_edge_alpha(
                        input,
                        g,
                        groups,
                        channels_per_group,
                        kernel_area,
                        batch,
                        group,
                        oy,
                        ox,
                        edge_kernel,
                        edge_biases,
                    );
                    for channel in 0..channels_per_group {
                        let channel_index = group * channels_per_group + channel;
                        let output_index = learned_output_index(
                            batch,
                            channel_index,
                            oy,
                            ox,
                            g.channels,
                            g.output_height,
                            g.output_width,
                        );
                        for (edge, weight) in alpha.iter().enumerate() {
                            if *weight == 0.0 {
                                continue;
                            }
                            let (iy, ix) = learned_edge_coordinate(g, oy, ox, edge)
                                .expect("nonzero alpha corresponds to a valid edge");
                            aggregate[output_index] += weight
                                * input[learned_input_index(
                                    batch,
                                    channel_index,
                                    iy,
                                    ix,
                                    g.channels,
                                    g.input_height,
                                    g.input_width,
                                )];
                        }
                    }
                }
            }
        }
    }
    Ok(aggregate)
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphConvLearnedAggregationGradients {
    pub input: Vec<f32>,
    pub edge_kernel: Vec<f32>,
    pub edge_biases: Vec<f32>,
}

/// Exact adjoint of [`learned_aggregate_forward`], including the masked
/// softmax Jacobian, input fan-out, and group-local edge parameter gradients.
pub fn learned_aggregate_backward(
    input: &[f32],
    output_gradient: &[f32],
    edge_kernel: &[f32],
    edge_biases: &[f32],
    geometry: GraphConvAggregationGeometry,
    groups: usize,
) -> Result<GraphConvLearnedAggregationGradients, String> {
    let (g, channels_per_group, kernel_area) =
        checked_learned_geometry(geometry, groups, edge_kernel, edge_biases)?;
    if input.len() != g.input_len() || output_gradient.len() != g.output_len() {
        return Err(format!(
            "graph_conv learned tensor length mismatch: expected input {}, output {}; got input {}, output {}",
            g.input_len(),
            g.output_len(),
            input.len(),
            output_gradient.len()
        ));
    }
    let mut input_gradient = vec![0.0; g.input_len()];
    let mut edge_kernel_gradient = vec![0.0; edge_kernel.len()];
    let mut edge_bias_gradient = vec![0.0; edge_biases.len()];
    let mut alpha_gradient = vec![0.0; kernel_area];
    for batch in 0..g.batch {
        for group in 0..groups {
            let channel_base = group * channels_per_group;
            for oy in 0..g.output_height {
                for ox in 0..g.output_width {
                    let alpha = learned_edge_alpha(
                        input,
                        g,
                        groups,
                        channels_per_group,
                        kernel_area,
                        batch,
                        group,
                        oy,
                        ox,
                        edge_kernel,
                        edge_biases,
                    );
                    alpha_gradient.fill(0.0);
                    let mut softmax_dot = 0.0;
                    for edge in 0..kernel_area {
                        let Some((iy, ix)) = learned_edge_coordinate(g, oy, ox, edge) else {
                            continue;
                        };
                        for channel in 0..channels_per_group {
                            let output_index = learned_output_index(
                                batch,
                                channel_base + channel,
                                oy,
                                ox,
                                g.channels,
                                g.output_height,
                                g.output_width,
                            );
                            let input_index = learned_input_index(
                                batch,
                                channel_base + channel,
                                iy,
                                ix,
                                g.channels,
                                g.input_height,
                                g.input_width,
                            );
                            alpha_gradient[edge] +=
                                output_gradient[output_index] * input[input_index];
                        }
                        softmax_dot += alpha[edge] * alpha_gradient[edge];
                    }
                    let reference = learned_reference_coordinate(g, oy, ox);
                    for edge in 0..kernel_area {
                        if alpha[edge] == 0.0 {
                            continue;
                        }
                        let (iy, ix) = learned_edge_coordinate(g, oy, ox, edge)
                            .expect("nonzero alpha corresponds to a valid edge");
                        let d_logit = alpha[edge] * (alpha_gradient[edge] - softmax_dot);
                        let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
                        edge_bias_gradient[group * kernel_area + edge] += d_logit;
                        for channel in 0..channels_per_group {
                            let neighbor_index = learned_input_index(
                                batch,
                                channel_base + channel,
                                iy,
                                ix,
                                g.channels,
                                g.input_height,
                                g.input_width,
                            );
                            let neighbor = input[neighbor_index];
                            let output_index = learned_output_index(
                                batch,
                                channel_base + channel,
                                oy,
                                ox,
                                g.channels,
                                g.output_height,
                                g.output_width,
                            );
                            input_gradient[neighbor_index] += alpha[edge]
                                * output_gradient[output_index]
                                + d_logit * edge_kernel[parameter + channels_per_group + channel];
                            edge_kernel_gradient[parameter + channels_per_group + channel] +=
                                d_logit * neighbor;
                            if let Some((reference_y, reference_x)) = reference {
                                let reference_index = learned_input_index(
                                    batch,
                                    channel_base + channel,
                                    reference_y,
                                    reference_x,
                                    g.channels,
                                    g.input_height,
                                    g.input_width,
                                );
                                let reference_value = input[reference_index];
                                input_gradient[reference_index] +=
                                    d_logit * edge_kernel[parameter + channel];
                                edge_kernel_gradient[parameter + channel] +=
                                    d_logit * reference_value;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(GraphConvLearnedAggregationGradients {
        input: input_gradient,
        edge_kernel: edge_kernel_gradient,
        edge_biases: edge_bias_gradient,
    })
}

#[cfg(test)]
mod learned_tests {
    use super::{
        GraphConvAggregationGeometry, learned_aggregate_backward, learned_aggregate_forward,
    };

    fn geometry() -> GraphConvAggregationGeometry {
        GraphConvAggregationGeometry {
            batch: 1,
            channels: 2,
            input_height: 3,
            input_width: 4,
            output_height: 3,
            output_width: 2,
            size: 3,
            stride_x: 2,
            stride_y: 1,
            dilation: 1,
            padding: 1,
        }
    }

    fn objective(input: &[f32], kernel: &[f32], biases: &[f32], seed: &[f32]) -> f32 {
        learned_aggregate_forward(input, kernel, biases, geometry(), 2)
            .expect("learned aggregate forward")
            .iter()
            .zip(seed)
            .map(|(value, gradient)| value * gradient)
            .sum()
    }

    fn assert_gradient(name: &str, analytic: &[f32], numeric: &[f32]) {
        assert_eq!(analytic.len(), numeric.len());
        for (index, (actual, expected)) in analytic.iter().zip(numeric).enumerate() {
            let tolerance = 2.0e-4 + 3.0e-3 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: analytic {actual}, finite difference {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn learned_aggregate_masks_invalid_edges_and_matches_finite_differences() {
        let g = geometry();
        let mut input = (0..g.input_len())
            .map(|i| (i as f32 * 0.31).sin() * 0.4)
            .collect::<Vec<_>>();
        let mut kernel = (0..36)
            .map(|i| (i as f32 * 0.17).cos() * 0.08)
            .collect::<Vec<_>>();
        let mut biases = (0..18)
            .map(|i| (i as f32 * 0.11).sin() * 0.03)
            .collect::<Vec<_>>();
        let seed = (0..g.output_len())
            .map(|i| (i as f32 * 0.23).cos() * 0.3)
            .collect::<Vec<_>>();
        let analytic = learned_aggregate_backward(&input, &seed, &kernel, &biases, g, 2)
            .expect("learned aggregate backward");
        let epsilon = 1.0e-3;
        let mut numeric_input = vec![0.0; input.len()];
        for i in 0..input.len() {
            input[i] += epsilon;
            let plus = objective(&input, &kernel, &biases, &seed);
            input[i] -= 2.0 * epsilon;
            let minus = objective(&input, &kernel, &biases, &seed);
            input[i] += epsilon;
            numeric_input[i] = (plus - minus) / (2.0 * epsilon);
        }
        let mut numeric_kernel = vec![0.0; kernel.len()];
        for i in 0..kernel.len() {
            kernel[i] += epsilon;
            let plus = objective(&input, &kernel, &biases, &seed);
            kernel[i] -= 2.0 * epsilon;
            let minus = objective(&input, &kernel, &biases, &seed);
            kernel[i] += epsilon;
            numeric_kernel[i] = (plus - minus) / (2.0 * epsilon);
        }
        let mut numeric_biases = vec![0.0; biases.len()];
        for i in 0..biases.len() {
            biases[i] += epsilon;
            let plus = objective(&input, &kernel, &biases, &seed);
            biases[i] -= 2.0 * epsilon;
            let minus = objective(&input, &kernel, &biases, &seed);
            biases[i] += epsilon;
            numeric_biases[i] = (plus - minus) / (2.0 * epsilon);
        }
        assert_gradient("input", &analytic.input, &numeric_input);
        assert_gradient("edge kernel", &analytic.edge_kernel, &numeric_kernel);
        assert_gradient("edge biases", &analytic.edge_biases, &numeric_biases);
    }

    #[test]
    fn learned_aggregate_rejects_bad_groups_and_parameter_layouts() {
        let g = geometry();
        let kernel = vec![0.0; 36];
        let biases = vec![0.0; 18];
        let input = vec![0.0; g.input_len()];
        assert!(learned_aggregate_forward(&input, &kernel, &biases, g, 0).is_err());
        assert!(learned_aggregate_forward(&input, &kernel, &biases, g, 3).is_err());
        assert!(learned_aggregate_forward(&input, &kernel[..35], &biases, g, 2).is_err());
        assert!(learned_aggregate_forward(&input, &kernel, &biases[..17], g, 2).is_err());
        assert!(
            learned_aggregate_forward(&input[..input.len() - 1], &kernel, &biases, g, 2).is_err()
        );
    }
}
