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
