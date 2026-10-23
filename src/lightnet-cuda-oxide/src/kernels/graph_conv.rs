//! Uniform valid-neighbor aggregation kernels for graph-convolution mode 0.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn graph_aggregate_fwd(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let output_len = batch * channels * output_height * output_width;
        if i >= output_len {
            return;
        }
        let ox = i % output_width;
        let oy = (i / output_width) % output_height;
        let channel = (i / (output_width * output_height)) % channels;
        let b = i / (output_width * output_height * channels);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut ky = 0usize;
        while ky < size {
            let iy = oy as i64 * stride_y as i64 - padding as i64 + ky as i64 * dilation as i64;
            let mut kx = 0usize;
            while kx < size {
                let ix = ox as i64 * stride_x as i64 - padding as i64 + kx as i64 * dilation as i64;
                if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
                    let input_index = ((b * channels + channel) * input_height + iy as usize)
                        * input_width
                        + ix as usize;
                    sum += input[input_index];
                    count += 1;
                }
                kx += 1;
            }
            ky += 1;
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = if count == 0 { 0.0 } else { sum / count as f32 };
        }
    }

    /// Gather the center/reference feature for GraphConv self projection.
    #[kernel]
    pub fn graph_reference_fwd(
        input: &[f32],
        mut reference: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let output_len = batch * channels * output_height * output_width;
        if i >= output_len {
            return;
        }
        let ox = i % output_width;
        let oy = (i / output_width) % output_height;
        let channel = (i / (output_width * output_height)) % channels;
        let b = i / (output_width * output_height * channels);
        let center = size / 2;
        let iy = oy as i64 * stride_y as i64 - padding as i64 + center as i64 * dilation as i64;
        let ix = ox as i64 * stride_x as i64 - padding as i64 + center as i64 * dilation as i64;
        let value = if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
            input[((b * channels + channel) * input_height + iy as usize) * input_width
                + ix as usize]
        } else {
            0.0
        };
        if let Some(dst) = reference.get_mut(idx) {
            *dst = value;
        }
    }

    /// Exact gather adjoint of the center/reference extraction.
    #[kernel]
    pub fn graph_reference_bwd(
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let input_len = batch * channels * input_height * input_width;
        if i >= input_len {
            return;
        }
        let ix = i % input_width;
        let iy = (i / input_width) % input_height;
        let channel = (i / (input_width * input_height)) % channels;
        let b = i / (input_width * input_height * channels);
        let center_offset = (size / 2) as i64 * dilation as i64;
        let oy_num = iy as i64 + padding as i64 - center_offset;
        let ox_num = ix as i64 + padding as i64 - center_offset;
        let mut gradient = 0.0f32;
        if oy_num >= 0
            && oy_num % stride_y as i64 == 0
            && ox_num >= 0
            && ox_num % stride_x as i64 == 0
        {
            let oy = (oy_num / stride_y as i64) as usize;
            let ox = (ox_num / stride_x as i64) as usize;
            if oy < output_height && ox < output_width {
                gradient = output_gradient
                    [((b * channels + channel) * output_height + oy) * output_width + ox];
            }
        }
        if let Some(dst) = input_gradient.get_mut(idx) {
            *dst = gradient;
        }
    }

    /// Learned edge scoring with invalid-neighbor masking before stable softmax.
    #[kernel]
    pub fn graph_learned_aggregate_fwd(
        input: &[f32],
        edge_kernel: &[f32],
        edge_biases: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let output_len = batch * channels * output_height * output_width;
        if i >= output_len {
            return;
        }
        let ox = i % output_width;
        let oy = (i / output_width) % output_height;
        let channel = (i / (output_width * output_height)) % channels;
        let b = i / (output_width * output_height * channels);
        let channels_per_group = channels / groups;
        let group = channel / channels_per_group;
        let channel_base = group * channels_per_group;
        let kernel_area = size * size;
        let center_edge = (size / 2) * size + size / 2;
        let ref_y = oy as i64 * stride_y as i64 - padding as i64
            + (center_edge / size) as i64 * dilation as i64;
        let ref_x = ox as i64 * stride_x as i64 - padding as i64
            + (center_edge % size) as i64 * dilation as i64;
        let reference_valid =
            ref_y >= 0 && ref_y < input_height as i64 && ref_x >= 0 && ref_x < input_width as i64;
        let mut maximum = f32::NEG_INFINITY;
        let mut edge = 0usize;
        while edge < kernel_area {
            let ky = edge / size;
            let kx = edge % size;
            let iy = oy as i64 * stride_y as i64 - padding as i64 + ky as i64 * dilation as i64;
            let ix = ox as i64 * stride_x as i64 - padding as i64 + kx as i64 * dilation as i64;
            if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
                let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
                let mut logit = edge_biases[group * kernel_area + edge];
                let mut c = 0usize;
                while c < channels_per_group {
                    let neighbor = input[((b * channels + channel_base + c) * input_height
                        + iy as usize)
                        * input_width
                        + ix as usize];
                    let reference = if reference_valid {
                        input[((b * channels + channel_base + c) * input_height + ref_y as usize)
                            * input_width
                            + ref_x as usize]
                    } else {
                        0.0
                    };
                    logit += edge_kernel[parameter + c] * reference
                        + edge_kernel[parameter + channels_per_group + c] * neighbor;
                    c += 1;
                }
                maximum = maximum.max(logit);
            }
            edge += 1;
        }
        let mut denominator = 0.0f32;
        edge = 0;
        while edge < kernel_area {
            let ky = edge / size;
            let kx = edge % size;
            let iy = oy as i64 * stride_y as i64 - padding as i64 + ky as i64 * dilation as i64;
            let ix = ox as i64 * stride_x as i64 - padding as i64 + kx as i64 * dilation as i64;
            if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
                let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
                let mut logit = edge_biases[group * kernel_area + edge];
                let mut c = 0usize;
                while c < channels_per_group {
                    let neighbor = input[((b * channels + channel_base + c) * input_height
                        + iy as usize)
                        * input_width
                        + ix as usize];
                    let reference = if reference_valid {
                        input[((b * channels + channel_base + c) * input_height + ref_y as usize)
                            * input_width
                            + ref_x as usize]
                    } else {
                        0.0
                    };
                    logit += edge_kernel[parameter + c] * reference
                        + edge_kernel[parameter + channels_per_group + c] * neighbor;
                    c += 1;
                }
                denominator += (logit - maximum).exp();
            }
            edge += 1;
        }
        let mut aggregate = 0.0f32;
        if denominator > 0.0 {
            edge = 0;
            while edge < kernel_area {
                let ky = edge / size;
                let kx = edge % size;
                let iy = oy as i64 * stride_y as i64 - padding as i64 + ky as i64 * dilation as i64;
                let ix = ox as i64 * stride_x as i64 - padding as i64 + kx as i64 * dilation as i64;
                if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
                    let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
                    let mut logit = edge_biases[group * kernel_area + edge];
                    let mut c = 0usize;
                    while c < channels_per_group {
                        let neighbor = input[((b * channels + channel_base + c) * input_height
                            + iy as usize)
                            * input_width
                            + ix as usize];
                        let reference = if reference_valid {
                            input[((b * channels + channel_base + c) * input_height
                                + ref_y as usize)
                                * input_width
                                + ref_x as usize]
                        } else {
                            0.0
                        };
                        logit += edge_kernel[parameter + c] * reference
                            + edge_kernel[parameter + channels_per_group + c] * neighbor;
                        c += 1;
                    }
                    let alpha = (logit - maximum).exp() / denominator;
                    let value = input[((b * channels + channel) * input_height + iy as usize)
                        * input_width
                        + ix as usize];
                    aggregate += alpha * value;
                }
                edge += 1;
            }
        }
        if let Some(dst) = output.get_mut(idx) {
            *dst = aggregate;
        }
    }

    #[inline]
    fn learned_edge_coordinate(
        oy: usize,
        ox: usize,
        edge: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) -> (i64, i64) {
        let iy =
            oy as i64 * stride_y as i64 - padding as i64 + (edge / size) as i64 * dilation as i64;
        let ix =
            ox as i64 * stride_x as i64 - padding as i64 + (edge % size) as i64 * dilation as i64;
        (iy, ix)
    }

    #[inline]
    fn learned_edge_logit(
        input: &[f32],
        edge_kernel: &[f32],
        edge_biases: &[f32],
        batch_index: usize,
        group: usize,
        oy: usize,
        ox: usize,
        edge: usize,
        channels: usize,
        channels_per_group: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) -> f32 {
        let kernel_area = size * size;
        let (iy, ix) =
            learned_edge_coordinate(oy, ox, edge, size, stride_x, stride_y, dilation, padding);
        if iy < 0 || iy >= input_height as i64 || ix < 0 || ix >= input_width as i64 {
            return f32::NEG_INFINITY;
        }
        let center_edge = (size / 2) * size + size / 2;
        let (ry, rx) = learned_edge_coordinate(
            oy,
            ox,
            center_edge,
            size,
            stride_x,
            stride_y,
            dilation,
            padding,
        );
        let reference_valid =
            ry >= 0 && ry < input_height as i64 && rx >= 0 && rx < input_width as i64;
        let channel_base = group * channels_per_group;
        let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
        let mut logit = edge_biases[group * kernel_area + edge];
        let mut channel = 0usize;
        while channel < channels_per_group {
            let neighbor = input[((batch_index * channels + channel_base + channel)
                * input_height
                + iy as usize)
                * input_width
                + ix as usize];
            let reference = if reference_valid {
                input[((batch_index * channels + channel_base + channel) * input_height
                    + ry as usize)
                    * input_width
                    + rx as usize]
            } else {
                0.0
            };
            logit += edge_kernel[parameter + channel] * reference
                + edge_kernel[parameter + channels_per_group + channel] * neighbor;
            channel += 1;
        }
        logit
    }

    /// One thread per batch/group/output-site/edge writes its normalized weight.
    #[kernel]
    pub fn graph_learned_alpha_fwd(
        input: &[f32],
        edge_kernel: &[f32],
        edge_biases: &[f32],
        mut alpha: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let kernel_area = size * size;
        let sites = batch * groups * output_height * output_width;
        if linear >= sites * kernel_area {
            return;
        }
        let target_edge = linear % kernel_area;
        let site = linear / kernel_area;
        let ox = site % output_width;
        let oy = (site / output_width) % output_height;
        let group = (site / (output_width * output_height)) % groups;
        let b = site / (output_width * output_height * groups);
        let channels_per_group = channels / groups;
        let mut maximum = f32::NEG_INFINITY;
        let mut edge = 0usize;
        while edge < kernel_area {
            maximum = maximum.max(learned_edge_logit(
                input,
                edge_kernel,
                edge_biases,
                b,
                group,
                oy,
                ox,
                edge,
                channels,
                channels_per_group,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                dilation,
                padding,
            ));
            edge += 1;
        }
        let mut denominator = 0.0f32;
        edge = 0;
        while edge < kernel_area {
            let logit = learned_edge_logit(
                input,
                edge_kernel,
                edge_biases,
                b,
                group,
                oy,
                ox,
                edge,
                channels,
                channels_per_group,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                dilation,
                padding,
            );
            if logit != f32::NEG_INFINITY {
                denominator += (logit - maximum).exp();
            }
            edge += 1;
        }
        let target_logit = learned_edge_logit(
            input,
            edge_kernel,
            edge_biases,
            b,
            group,
            oy,
            ox,
            target_edge,
            channels,
            channels_per_group,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            dilation,
            padding,
        );
        let value = if denominator > 0.0 && target_logit != f32::NEG_INFINITY {
            (target_logit - maximum).exp() / denominator
        } else {
            0.0
        };
        if let Some(dst) = alpha.get_mut(idx) {
            *dst = value;
        }
    }

    /// Softmax-adjoint logits for each batch/group/output-site/edge tuple.
    #[kernel]
    pub fn graph_learned_logits_bwd(
        input: &[f32],
        output_gradient: &[f32],
        alpha: &[f32],
        mut logits_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let kernel_area = size * size;
        let sites = batch * groups * output_height * output_width;
        let total = sites * kernel_area;
        if linear >= total {
            return;
        }
        let edge = linear % kernel_area;
        let site = linear / kernel_area;
        let ox = site % output_width;
        let oy = (site / output_width) % output_height;
        let group = (site / (output_width * output_height)) % groups;
        let b = site / (output_width * output_height * groups);
        let channels_per_group = channels / groups;
        let (iy, ix) =
            learned_edge_coordinate(oy, ox, edge, size, stride_x, stride_y, dilation, padding);
        let mut edge_gradient = 0.0f32;
        if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
            let channel_base = group * channels_per_group;
            let mut channel = 0usize;
            while channel < channels_per_group {
                let out_index = ((b * channels + channel_base + channel) * output_height + oy)
                    * output_width
                    + ox;
                let input_index = ((b * channels + channel_base + channel) * input_height
                    + iy as usize)
                    * input_width
                    + ix as usize;
                edge_gradient += output_gradient[out_index] * input[input_index];
                channel += 1;
            }
        }
        let mut softmax_dot = 0.0f32;
        let mut other_edge = 0usize;
        while other_edge < kernel_area {
            let (ny, nx) = learned_edge_coordinate(
                oy, ox, other_edge, size, stride_x, stride_y, dilation, padding,
            );
            if ny >= 0 && ny < input_height as i64 && nx >= 0 && nx < input_width as i64 {
                let mut alpha_gradient = 0.0f32;
                let channel_base = group * channels_per_group;
                let mut channel = 0usize;
                while channel < channels_per_group {
                    let out_index = ((b * channels + channel_base + channel) * output_height + oy)
                        * output_width
                        + ox;
                    let input_index = ((b * channels + channel_base + channel) * input_height
                        + ny as usize)
                        * input_width
                        + nx as usize;
                    alpha_gradient += output_gradient[out_index] * input[input_index];
                    channel += 1;
                }
                softmax_dot += alpha[site * kernel_area + other_edge] * alpha_gradient;
            }
            other_edge += 1;
        }
        let dlogit = alpha[linear] * (edge_gradient - softmax_dot);
        if let Some(dst) = logits_gradient.get_mut(idx) {
            *dst = dlogit;
        }
    }

    /// Exact learned-edge aggregation input adjoint; each input element owns its sum.
    #[kernel]
    pub fn graph_learned_input_bwd(
        output_gradient: &[f32],
        edge_kernel: &[f32],
        alpha: &[f32],
        logits_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let input_len = batch * channels * input_height * input_width;
        if linear >= input_len {
            return;
        }
        let ix = linear % input_width;
        let iy = (linear / input_width) % input_height;
        let channel = (linear / (input_width * input_height)) % channels;
        let b = linear / (input_width * input_height * channels);
        let channels_per_group = channels / groups;
        let group = channel / channels_per_group;
        let channel_in_group = channel % channels_per_group;
        let kernel_area = size * size;
        let mut sum = 0.0f32;
        let mut oy = 0usize;
        while oy < output_height {
            let mut ox = 0usize;
            while ox < output_width {
                let site = ((b * groups + group) * output_height + oy) * output_width + ox;
                let center_edge = (size / 2) * size + size / 2;
                let (ry, rx) = learned_edge_coordinate(
                    oy,
                    ox,
                    center_edge,
                    size,
                    stride_x,
                    stride_y,
                    dilation,
                    padding,
                );
                let is_reference = ry == iy as i64 && rx == ix as i64;
                let mut edge = 0usize;
                while edge < kernel_area {
                    let (ny, nx) = learned_edge_coordinate(
                        oy, ox, edge, size, stride_x, stride_y, dilation, padding,
                    );
                    let edge_index = site * kernel_area + edge;
                    let parameter = (group * kernel_area + edge) * 2 * channels_per_group;
                    if ny == iy as i64 && nx == ix as i64 {
                        let output_index =
                            ((b * channels + channel) * output_height + oy) * output_width + ox;
                        sum += alpha[edge_index] * output_gradient[output_index]
                            + logits_gradient[edge_index]
                                * edge_kernel[parameter + channels_per_group + channel_in_group];
                    }
                    if is_reference
                        && ry >= 0
                        && ry < input_height as i64
                        && rx >= 0
                        && rx < input_width as i64
                        && ny >= 0
                        && ny < input_height as i64
                        && nx >= 0
                        && nx < input_width as i64
                    {
                        sum +=
                            logits_gradient[edge_index] * edge_kernel[parameter + channel_in_group];
                    }
                    edge += 1;
                }
                ox += 1;
            }
            oy += 1;
        }
        if let Some(dst) = input_gradient.get_mut(idx) {
            *dst = sum;
        }
    }

    /// Edge-kernel parameter gradients, reduced independently per parameter.
    #[kernel]
    pub fn graph_learned_kernel_bwd(
        input: &[f32],
        logits_gradient: &[f32],
        mut edge_kernel_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get();
        let channels_per_group = channels / groups;
        let kernel_area = size * size;
        let kernel_len = groups * kernel_area * 2 * channels_per_group;
        if linear >= kernel_len {
            return;
        }
        let channel_pair = linear % (2 * channels_per_group);
        let channel_in_group = channel_pair % channels_per_group;
        let is_neighbor = channel_pair >= channels_per_group;
        let edge = (linear / (2 * channels_per_group)) % kernel_area;
        let group = linear / (2 * channels_per_group * kernel_area);
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut oy = 0usize;
            while oy < output_height {
                let mut ox = 0usize;
                while ox < output_width {
                    let site = ((b * groups + group) * output_height + oy) * output_width + ox;
                    let (iy, ix) = learned_edge_coordinate(
                        oy, ox, edge, size, stride_x, stride_y, dilation, padding,
                    );
                    if iy >= 0 && iy < input_height as i64 && ix >= 0 && ix < input_width as i64 {
                        let (ry, rx) = learned_edge_coordinate(
                            oy,
                            ox,
                            (size / 2) * size + size / 2,
                            size,
                            stride_x,
                            stride_y,
                            dilation,
                            padding,
                        );
                        let (y, x) = if is_neighbor { (iy, ix) } else { (ry, rx) };
                        if is_neighbor
                            || (y >= 0
                                && y < input_height as i64
                                && x >= 0
                                && x < input_width as i64)
                        {
                            let channel = group * channels_per_group + channel_in_group;
                            let value = input[((b * channels + channel) * input_height
                                + y as usize)
                                * input_width
                                + x as usize];
                            sum += logits_gradient[site * kernel_area + edge] * value;
                        }
                    }
                    ox += 1;
                }
                oy += 1;
            }
            b += 1;
        }
        if let Some(dst) = edge_kernel_gradient.get_mut(idx) {
            *dst = sum;
        }
    }

    /// Edge-bias gradients, reduced independently per group and edge.
    #[kernel]
    pub fn graph_learned_bias_bwd(
        logits_gradient: &[f32],
        mut edge_bias_gradient: DisjointSlice<f32>,
        batch: usize,
        groups: usize,
        output_height: usize,
        output_width: usize,
        kernel_area: usize,
    ) {
        let idx = thread::index_1d();
        let parameter = idx.get();
        let edge_count = groups * kernel_area;
        if parameter >= edge_count {
            return;
        }
        let group = parameter / kernel_area;
        let edge = parameter % kernel_area;
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            let mut oy = 0usize;
            while oy < output_height {
                let mut ox = 0usize;
                while ox < output_width {
                    let site = ((b * groups + group) * output_height + oy) * output_width + ox;
                    sum += logits_gradient[site * kernel_area + edge];
                    ox += 1;
                }
                oy += 1;
            }
            b += 1;
        }
        if let Some(dst) = edge_bias_gradient.get_mut(idx) {
            *dst = sum;
        }
    }

    /// Each input element gathers the adjoint contributions of all output sites
    /// for which it is a valid neighbor; no atomics or scratch buffer required.
    #[kernel]
    pub fn graph_aggregate_bwd(
        output_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let input_len = batch * channels * input_height * input_width;
        if i >= input_len {
            return;
        }
        let ix = i % input_width;
        let iy = (i / input_width) % input_height;
        let channel = (i / (input_width * input_height)) % channels;
        let b = i / (input_width * input_height * channels);
        let mut sum = 0.0f32;
        let mut ky = 0usize;
        while ky < size {
            let oy_num = iy as i64 + padding as i64 - ky as i64 * dilation as i64;
            if oy_num >= 0 && oy_num % stride_y as i64 == 0 {
                let oy = (oy_num / stride_y as i64) as usize;
                if oy < output_height {
                    let mut kx = 0usize;
                    while kx < size {
                        let ox_num = ix as i64 + padding as i64 - kx as i64 * dilation as i64;
                        if ox_num >= 0 && ox_num % stride_x as i64 == 0 {
                            let ox = (ox_num / stride_x as i64) as usize;
                            if ox < output_width {
                                let mut count = 0usize;
                                let mut ky2 = 0usize;
                                while ky2 < size {
                                    let iy2 = oy as i64 * stride_y as i64 - padding as i64
                                        + ky2 as i64 * dilation as i64;
                                    let mut kx2 = 0usize;
                                    while kx2 < size {
                                        let ix2 = ox as i64 * stride_x as i64 - padding as i64
                                            + kx2 as i64 * dilation as i64;
                                        if iy2 >= 0
                                            && iy2 < input_height as i64
                                            && ix2 >= 0
                                            && ix2 < input_width as i64
                                        {
                                            count += 1;
                                        }
                                        kx2 += 1;
                                    }
                                    ky2 += 1;
                                }
                                if count > 0 {
                                    let output_index = ((b * channels + channel) * output_height
                                        + oy)
                                        * output_width
                                        + ox;
                                    sum += output_gradient[output_index] / count as f32;
                                }
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(dst) = input_gradient.get_mut(idx) {
            *dst = sum;
        }
    }
}
