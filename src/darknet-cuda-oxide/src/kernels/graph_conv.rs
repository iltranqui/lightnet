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
