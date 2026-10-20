//! Native primitives for the single-template Dynamic Snake Conv.

use super::resample::atomic_add_f32;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Build one morph branch's NCHW pixel sampling coordinates from the
    /// already-tanh-transformed `[B,2*K,H,W]` offset map.
    ///
    /// Output maps use `[B,H,W,K]`; call once with morph 0 and once with morph
    /// 1 for `morph=2`, then select the appropriate output-channel partition.
    #[kernel]
    pub fn dsconv_coordinate_map(
        offsets: &[f32],
        mut y_map: DisjointSlice<f32>,
        mut x_map: DisjointSlice<f32>,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= y_map.len() || index >= x_map.len() {
            return;
        }
        let k = index % kernel_size;
        let cell = index / kernel_size;
        let x = cell % width;
        let y = (cell / width) % height;
        let b = cell / (width * height);
        if b >= batch {
            return;
        }
        let center = kernel_size / 2;
        let spatial = height * width;
        let offset_batch = b * 2 * kernel_size * spatial;
        let mut cumulative = 0.0f32;
        if k > center {
            let mut j = center + 1;
            while j <= k {
                let channel = if morph == 0 { j } else { kernel_size + j };
                cumulative += offsets[offset_batch + channel * spatial + y * width + x];
                j += 1;
            }
        } else if k < center {
            let mut j = center;
            while j > k {
                j -= 1;
                let channel = if morph == 0 { j } else { kernel_size + j };
                cumulative += offsets[offset_batch + channel * spatial + y * width + x];
            }
        }
        if morph == 0 {
            // x-snake: offsets displace y; the kernel index advances x.
            unsafe {
                *y_map.as_mut_ptr().add(index) = y as f32 + extend_scope * cumulative;
                *x_map.as_mut_ptr().add(index) = x as f32 + k as f32 - center as f32;
            }
        } else {
            // y-snake: offsets displace x; the kernel index advances y.
            unsafe {
                *y_map.as_mut_ptr().add(index) = y as f32 + k as f32 - center as f32;
                *x_map.as_mut_ptr().add(index) = x as f32 + extend_scope * cumulative;
            }
        }
    }

    /// Gather one border-clamped bilinear sample per `(b,c,y,x,k)` into
    /// `[B,C,H,W,K]`, matching the CPU DSConv interpolation order.
    #[kernel]
    pub fn dsconv_bilinear_sample(
        input: &[f32],
        y_map: &[f32],
        x_map: &[f32],
        mut sampled: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= sampled.len() {
            return;
        }
        let k = index % kernel_size;
        let cell = index / kernel_size;
        let x = cell % width;
        let y = (cell / width) % height;
        let c = (cell / (width * height)) % channels;
        let b = cell / (width * height * channels);
        if b >= batch {
            return;
        }
        let coordinate_index = ((b * height + y) * width + x) * kernel_size + k;
        let sample_y = y_map[coordinate_index];
        let sample_x = x_map[coordinate_index];
        let max_y = (height - 1) as f32;
        let max_x = (width - 1) as f32;
        let mut y0 = 0usize;
        let mut y1 = 0usize;
        let mut fy = 0.0f32;
        if sample_y >= max_y {
            y0 = height - 1;
            y1 = height - 1;
        } else if sample_y >= 0.0 {
            y0 = sample_y as usize;
            y1 = y0 + 1;
            fy = sample_y - y0 as f32;
        }
        let mut x0 = 0usize;
        let mut x1 = 0usize;
        let mut fx = 0.0f32;
        if sample_x >= max_x {
            x0 = width - 1;
            x1 = width - 1;
        } else if sample_x >= 0.0 {
            x0 = sample_x as usize;
            x1 = x0 + 1;
            fx = sample_x - x0 as f32;
        }
        let base = ((b * channels + c) * height) * width;
        let p00 = input[base + y0 * width + x0];
        let p01 = input[base + y0 * width + x1];
        let p10 = input[base + y1 * width + x0];
        let p11 = input[base + y1 * width + x1];
        let top = p00 + fx * (p01 - p00);
        let bottom = p10 + fx * (p11 - p10);
        let value = top + fy * (bottom - top);
        // SAFETY: host validates exact input/coordinate/output tensor lengths.
        unsafe {
            *sampled.as_mut_ptr().add(index) = value;
        }
    }

    /// Collapse `[B,C,H,W,K]` samples with weights `[F,C,K]` and one bias
    /// per output channel into NCHW `[B,F,H,W]` output.
    #[kernel]
    pub fn dsconv_collapse(
        sampled: &[f32],
        weights: &[f32],
        biases: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let x = index % width;
        let y = (index / width) % height;
        let filter = (index / (width * height)) % filters;
        let b = index / (width * height * filters);
        if b >= batch {
            return;
        }
        let mut value = biases[filter];
        let mut channel = 0;
        while channel < channels {
            let mut k = 0;
            while k < kernel_size {
                let sample_index =
                    (((b * channels + channel) * height + y) * width + x) * kernel_size + k;
                let weight_index = (filter * channels + channel) * kernel_size + k;
                value += weights[weight_index] * sampled[sample_index];
                k += 1;
            }
            channel += 1;
        }
        // SAFETY: host validates operand lengths; each thread owns one output.
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Collapse two snake branches, choosing x/y samples per output channel
    /// for morph=2's floor/remainder partition.
    #[kernel]
    pub fn dsconv_collapse_morph(
        sampled_x: &[f32],
        sampled_y: &[f32],
        weights: &[f32],
        biases: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        morph: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let x = index % width;
        let y = (index / width) % height;
        let filter = (index / (width * height)) % filters;
        let b = index / (width * height * filters);
        if b >= batch {
            return;
        }
        let use_x = morph == 0 || (morph == 2 && filter < filters / 2);
        let sampled = if use_x { sampled_x } else { sampled_y };
        let mut value = biases[filter];
        let mut channel = 0;
        while channel < channels {
            let mut k = 0;
            while k < kernel_size {
                let sample_index =
                    (((b * channels + channel) * height + y) * width + x) * kernel_size + k;
                let weight_index = (filter * channels + channel) * kernel_size + k;
                value += weights[weight_index] * sampled[sample_index];
                k += 1;
            }
            channel += 1;
        }
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Adjoint of `dsconv_collapse`, owned by one sampled input element.
    #[kernel]
    pub fn dsconv_collapse_backward_sampled(
        weights: &[f32],
        output_gradient: &[f32],
        mut sampled_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        filter_start: usize,
        filter_count: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= sampled_gradient.len() {
            return;
        }
        let k = index % kernel_size;
        let cell = index / kernel_size;
        let x = cell % width;
        let y = (cell / width) % height;
        let channel = (cell / (width * height)) % channels;
        let b = cell / (width * height * channels);
        if b >= batch {
            return;
        }
        let mut value = 0.0f32;
        let mut f = filter_start;
        while f < filter_start + filter_count {
            let output_index = ((b * filters + f) * height + y) * width + x;
            let weight_index = (f * channels + channel) * kernel_size + k;
            value += output_gradient[output_index] * weights[weight_index];
            f += 1;
        }
        unsafe {
            *sampled_gradient.as_mut_ptr().add(index) = value;
        }
    }

    /// Weight-gradient reduction for one `[F,C,K]` parameter element.
    #[kernel]
    pub fn dsconv_collapse_backward_weights(
        sampled: &[f32],
        output_gradient: &[f32],
        mut weight_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        filters: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        filter_start: usize,
        filter_count: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= weight_gradient.len() {
            return;
        }
        let k = index % kernel_size;
        let channel = (index / kernel_size) % channels;
        let filter = index / (channels * kernel_size);
        let mut value = 0.0f32;
        if filter >= filter_start && filter < filter_start + filter_count {
            let mut b = 0;
            while b < batch {
                let mut y = 0;
                while y < height {
                    let mut x = 0;
                    while x < width {
                        let output_index = ((b * filters + filter) * height + y) * width + x;
                        let sample_index =
                            (((b * channels + channel) * height + y) * width + x) * kernel_size + k;
                        value += output_gradient[output_index] * sampled[sample_index];
                        x += 1;
                    }
                    y += 1;
                }
                b += 1;
            }
        }
        unsafe {
            *weight_gradient.as_mut_ptr().add(index) = value;
        }
    }

    /// Bias-gradient reduction for one output channel.
    #[kernel]
    pub fn dsconv_collapse_backward_biases(
        output_gradient: &[f32],
        mut bias_gradient: DisjointSlice<f32>,
        batch: usize,
        filters: usize,
        height: usize,
        width: usize,
        filter_start: usize,
        filter_count: usize,
    ) {
        let filter = thread::index_1d().get();
        if filter >= bias_gradient.len() {
            return;
        }
        let mut value = 0.0f32;
        if filter >= filter_start && filter < filter_start + filter_count {
            let mut b = 0;
            while b < batch {
                let mut y = 0;
                while y < height {
                    let mut x = 0;
                    while x < width {
                        let output_index = ((b * filters + filter) * height + y) * width + x;
                        value += output_gradient[output_index];
                        x += 1;
                    }
                    y += 1;
                }
                b += 1;
            }
        }
        unsafe {
            *bias_gradient.as_mut_ptr().add(filter) = value;
        }
    }

    /// Adjoint of the border-clamped gather. One thread owns a coordinate
    /// sample, reduces coordinate derivatives across channels, and atomically
    /// scatters the four bilinear contributions into `input_gradient`.
    #[kernel]
    pub fn dsconv_bilinear_backward(
        input: &[f32],
        y_map: &[f32],
        x_map: &[f32],
        sampled_gradient: &[f32],
        mut input_gradient: DisjointSlice<f32>,
        mut y_gradient: DisjointSlice<f32>,
        mut x_gradient: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) {
        let coordinate_index = thread::index_1d().get();
        if coordinate_index >= y_gradient.len() || coordinate_index >= x_gradient.len() {
            return;
        }
        let k = coordinate_index % kernel_size;
        let cell = coordinate_index / kernel_size;
        let x = cell % width;
        let y = (cell / width) % height;
        let b = cell / (width * height);
        if b >= batch {
            return;
        }
        let sample_y = y_map[coordinate_index];
        let sample_x = x_map[coordinate_index];
        let max_y = (height - 1) as f32;
        let max_x = (width - 1) as f32;
        let mut y0 = 0usize;
        let mut y1 = 0usize;
        let mut fy = 0.0f32;
        if sample_y >= max_y {
            y0 = height - 1;
            y1 = height - 1;
        } else if sample_y >= 0.0 {
            y0 = sample_y as usize;
            y1 = y0 + 1;
            fy = sample_y - y0 as f32;
        }
        let mut x0 = 0usize;
        let mut x1 = 0usize;
        let mut fx = 0.0f32;
        if sample_x >= max_x {
            x0 = width - 1;
            x1 = width - 1;
        } else if sample_x >= 0.0 {
            x0 = sample_x as usize;
            x1 = x0 + 1;
            fx = sample_x - x0 as f32;
        }
        let y_active = sample_y >= 0.0 && sample_y < max_y;
        let x_active = sample_x >= 0.0 && sample_x < max_x;
        let mut coordinate_dy = 0.0f32;
        let mut coordinate_dx = 0.0f32;
        let spatial = height * width;
        let mut channel = 0;
        while channel < channels {
            let base = (b * channels + channel) * spatial;
            let sample_index =
                ((b * channels + channel) * spatial + y * width + x) * kernel_size + k;
            let grad = sampled_gradient[sample_index];
            let p00 = input[base + y0 * width + x0];
            let p01 = input[base + y0 * width + x1];
            let p10 = input[base + y1 * width + x0];
            let p11 = input[base + y1 * width + x1];
            let wx0 = 1.0 - fx;
            let wy0 = 1.0 - fy;
            unsafe {
                let input_ptr = input_gradient.as_mut_ptr();
                atomic_add_f32(input_ptr.add(base + y0 * width + x0), grad * wy0 * wx0);
                atomic_add_f32(input_ptr.add(base + y0 * width + x1), grad * wy0 * fx);
                atomic_add_f32(input_ptr.add(base + y1 * width + x0), grad * fy * wx0);
                atomic_add_f32(input_ptr.add(base + y1 * width + x1), grad * fy * fx);
            }
            if y_active {
                coordinate_dy += grad * ((1.0 - fx) * (p10 - p00) + fx * (p11 - p01));
            }
            if x_active {
                coordinate_dx += grad * ((1.0 - fy) * (p01 - p00) + fy * (p11 - p10));
            }
            channel += 1;
        }
        unsafe {
            *y_gradient.as_mut_ptr().add(coordinate_index) = coordinate_dy;
            *x_gradient.as_mut_ptr().add(coordinate_index) = coordinate_dx;
        }
    }

    /// Adjoint from the two coordinate maps through center-anchored cumulative
    /// offsets and tanh. The center offset and unused morph axis have zero grad.
    #[kernel]
    pub fn dsconv_offset_map_backward(
        offsets_tanh: &[f32],
        y_coordinate_gradient: &[f32],
        x_coordinate_gradient: &[f32],
        mut offset_gradient: DisjointSlice<f32>,
        batch: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        extend_scope: f32,
        morph: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= offset_gradient.len() {
            return;
        }
        let spatial = height * width;
        let channel = (index / spatial) % (2 * kernel_size);
        let b = index / (spatial * 2 * kernel_size);
        let k = if channel < kernel_size {
            channel
        } else {
            channel - kernel_size
        };
        let center = kernel_size / 2;
        let position = index % spatial;
        if b >= batch || k == center {
            unsafe {
                *offset_gradient.as_mut_ptr().add(index) = 0.0;
            }
            return;
        }
        let y_axis = channel < kernel_size;
        if (y_axis && morph == 1) || (!y_axis && morph == 0) {
            unsafe {
                *offset_gradient.as_mut_ptr().add(index) = 0.0;
            }
            return;
        }
        let coord_batch = b * spatial * kernel_size;
        let coord_grad = if y_axis {
            y_coordinate_gradient
        } else {
            x_coordinate_gradient
        };
        let mut cumulative = 0.0f32;
        if k < center {
            let mut j = 0;
            while j <= k {
                cumulative += coord_grad[coord_batch + position * kernel_size + j];
                j += 1;
            }
        } else {
            let mut j = kernel_size;
            while j > k {
                j -= 1;
                cumulative += coord_grad[coord_batch + position * kernel_size + j];
            }
        }
        let tanh_index = index;
        let tanh_value = offsets_tanh[tanh_index];
        let value = cumulative * extend_scope * (1.0 - tanh_value * tanh_value);
        unsafe {
            *offset_gradient.as_mut_ptr().add(index) = value;
        }
    }
}
