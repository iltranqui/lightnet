//! Host references for transposed convolution in Darknet `[input][filter][ky][kx]` order.

/// Transposed convolution without bias, returning NCHW output.
#[allow(clippy::too_many_arguments)]
pub fn forward(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    stride: usize,
    output_height: usize,
    output_width: usize,
) -> Vec<f32> {
    assert!(
        batch > 0
            && channels > 0
            && input_height > 0
            && input_width > 0
            && filters > 0
            && kernel_size > 0
            && stride > 0
    );
    assert_eq!(input.len(), batch * channels * input_height * input_width);
    assert_eq!(
        weights.len(),
        channels * filters * kernel_size * kernel_size
    );
    assert_eq!(output_height, stride * (input_height - 1) + kernel_size);
    assert_eq!(output_width, stride * (input_width - 1) + kernel_size);

    let mut output = vec![0.0; batch * filters * output_height * output_width];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..output_height {
                for ox in 0..output_width {
                    let mut sum = 0.0;
                    for c in 0..channels {
                        for ky in 0..kernel_size {
                            let iy_num = oy as isize - ky as isize;
                            if iy_num < 0 || iy_num as usize % stride != 0 {
                                continue;
                            }
                            let iy = iy_num as usize / stride;
                            if iy >= input_height {
                                continue;
                            }
                            for kx in 0..kernel_size {
                                let ix_num = ox as isize - kx as isize;
                                if ix_num < 0 || ix_num as usize % stride != 0 {
                                    continue;
                                }
                                let ix = ix_num as usize / stride;
                                if ix >= input_width {
                                    continue;
                                }
                                let xi =
                                    ((b * channels + c) * input_height + iy) * input_width + ix;
                                let wi = ((c * filters + f) * kernel_size + ky) * kernel_size + kx;
                                sum += input[xi] * weights[wi];
                            }
                        }
                    }
                    output[((b * filters + f) * output_height + oy) * output_width + ox] = sum;
                }
            }
        }
    }
    output
}

/// `(d_input, d_weights, d_biases)` for the forward operation. All input
/// gradients are exact derivatives of [`forward`], and each output gradient
/// element is consumed once per contributing input/kernel position.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    input: &[f32],
    weights: &[f32],
    output_grad: &[f32],
    batch: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    filters: usize,
    kernel_size: usize,
    stride: usize,
    output_height: usize,
    output_width: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(
        batch > 0
            && channels > 0
            && input_height > 0
            && input_width > 0
            && filters > 0
            && kernel_size > 0
            && stride > 0
    );
    assert_eq!(input.len(), batch * channels * input_height * input_width);
    assert_eq!(
        weights.len(),
        channels * filters * kernel_size * kernel_size
    );
    assert_eq!(
        output_grad.len(),
        batch * filters * output_height * output_width
    );
    assert_eq!(output_height, stride * (input_height - 1) + kernel_size);
    assert_eq!(output_width, stride * (input_width - 1) + kernel_size);

    let mut d_input = vec![0.0; input.len()];
    let mut d_weights = vec![0.0; weights.len()];
    let mut d_biases = vec![0.0; filters];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..output_height {
                for ox in 0..output_width {
                    d_biases[f] +=
                        output_grad[((b * filters + f) * output_height + oy) * output_width + ox];
                }
            }
        }
        for c in 0..channels {
            for iy in 0..input_height {
                for ix in 0..input_width {
                    let xi = ((b * channels + c) * input_height + iy) * input_width + ix;
                    let x = input[xi];
                    for f in 0..filters {
                        for ky in 0..kernel_size {
                            for kx in 0..kernel_size {
                                let yi = ((b * filters + f) * output_height + iy * stride + ky)
                                    * output_width
                                    + ix * stride
                                    + kx;
                                let wi = ((c * filters + f) * kernel_size + ky) * kernel_size + kx;
                                let grad = output_grad[yi];
                                d_input[xi] += weights[wi] * grad;
                                d_weights[wi] += x * grad;
                            }
                        }
                    }
                }
            }
        }
    }
    (d_input, d_weights, d_biases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_and_backward_match_distinct_channel_filter_layout() {
        let (batch, channels, ih, iw, filters, k, stride) = (2, 2, 2, 3, 3, 2, 2);
        let (oh, ow) = (stride * (ih - 1) + k, stride * (iw - 1) + k);
        let input: Vec<_> = (0..batch * channels * ih * iw)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let weights: Vec<_> = (0..channels * filters * k * k)
            .map(|i| (i as f32 * 0.23).cos() * 0.3)
            .collect();
        let dy: Vec<_> = (0..batch * filters * oh * ow)
            .map(|i| (i as f32 * 0.11).sin() * 0.7)
            .collect();
        let y = forward(
            &input, &weights, batch, channels, ih, iw, filters, k, stride, oh, ow,
        );
        let (dx, dw, db) = backward(
            &input, &weights, &dy, batch, channels, ih, iw, filters, k, stride, oh, ow,
        );
        assert_eq!(y.len(), batch * filters * oh * ow);
        assert_eq!(dx.len(), input.len());
        assert_eq!(dw.len(), weights.len());
        assert_eq!(db.len(), filters);
        let h = 1e-3;
        for i in (0..input.len()).step_by(5) {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp: f32 = forward(
                &plus, &weights, batch, channels, ih, iw, filters, k, stride, oh, ow,
            )
            .iter()
            .zip(&dy)
            .map(|(y, d)| y * d)
            .sum();
            let lm: f32 = forward(
                &minus, &weights, batch, channels, ih, iw, filters, k, stride, oh, ow,
            )
            .iter()
            .zip(&dy)
            .map(|(y, d)| y * d)
            .sum();
            assert!(((lp - lm) / (2.0 * h) - dx[i]).abs() < 3e-3);
        }
        for i in (0..weights.len()).step_by(7) {
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            plus[i] += h;
            minus[i] -= h;
            let lp: f32 = forward(
                &input, &plus, batch, channels, ih, iw, filters, k, stride, oh, ow,
            )
            .iter()
            .zip(&dy)
            .map(|(y, d)| y * d)
            .sum();
            let lm: f32 = forward(
                &input, &minus, batch, channels, ih, iw, filters, k, stride, oh, ow,
            )
            .iter()
            .zip(&dy)
            .map(|(y, d)| y * d)
            .sum();
            assert!(((lp - lm) / (2.0 * h) - dw[i]).abs() < 3e-3);
        }
    }
}
