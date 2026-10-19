//! Host-reference operations for locally connected layers.
//!
//! Local weights are stored location-major as `[location, filter, channel, ky, kx]`;
//! unlike convolution, each output location owns a distinct kernel.

fn checked_len(dims: &[usize], what: &str) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| format!("local {what} element count overflows"))
    })
}

fn validate(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
) -> Result<(usize, usize), String> {
    if [
        batch, channels, height, width, filters, out_h, out_w, size, stride_x, stride_y,
    ]
    .contains(&0)
    {
        return Err("local dimensions and strides must be positive".to_owned());
    }
    let locations = out_h
        .checked_mul(out_w)
        .ok_or_else(|| "local output location count overflows".to_owned())?;
    let kernel = checked_len(&[channels, size, size], "kernel")?;
    let expected_input = checked_len(&[batch, channels, height, width], "input")?;
    let expected_weights = checked_len(&[locations, filters, kernel], "weights")?;
    if input.len() != expected_input || weights.len() != expected_weights {
        return Err(format!(
            "local input/weights have lengths {}/{}, expected {expected_input}/{expected_weights}",
            input.len(),
            weights.len()
        ));
    }
    Ok((locations, kernel))
}

#[allow(clippy::too_many_arguments)]
pub fn forward(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
) -> Result<Vec<f32>, String> {
    let (locations, kernel) = validate(
        input, weights, batch, channels, height, width, filters, out_h, out_w, size, stride_x,
        stride_y,
    )?;
    if biases.len() != locations * filters {
        return Err("local bias length mismatch".to_owned());
    }
    let mut output = vec![0.0; checked_len(&[batch, filters, locations], "output")?];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let location = oy * out_w + ox;
                    let mut sum = biases[location * filters + f];
                    for c in 0..channels {
                        for ky in 0..size {
                            let iy = (oy * stride_y + ky) as isize - padding as isize;
                            if iy < 0 || iy as usize >= height {
                                continue;
                            }
                            for kx in 0..size {
                                let ix = (ox * stride_x + kx) as isize - padding as isize;
                                if ix < 0 || ix as usize >= width {
                                    continue;
                                }
                                let ki = (c * size + ky) * size + kx;
                                let wi = (location * filters + f) * kernel + ki;
                                let xi = ((b * channels + c) * height + iy as usize) * width
                                    + ix as usize;
                                sum += weights[wi] * input[xi];
                            }
                        }
                    }
                    output[(b * filters + f) * locations + location] = sum;
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn backward(
    input: &[f32],
    weights: &[f32],
    delta: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let (locations, kernel) = validate(
        input, weights, batch, channels, height, width, filters, out_h, out_w, size, stride_x,
        stride_y,
    )?;
    if delta.len() != checked_len(&[batch, filters, locations], "delta")? {
        return Err("local output-gradient length mismatch".to_owned());
    }
    let mut dx = vec![0.0; input.len()];
    let mut dw = vec![0.0; weights.len()];
    let mut db = vec![0.0; locations * filters];
    for b in 0..batch {
        for f in 0..filters {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let location = oy * out_w + ox;
                    let dy = delta[(b * filters + f) * locations + location];
                    db[location * filters + f] += dy;
                    for c in 0..channels {
                        for ky in 0..size {
                            let iy = (oy * stride_y + ky) as isize - padding as isize;
                            if iy < 0 || iy as usize >= height {
                                continue;
                            }
                            for kx in 0..size {
                                let ix = (ox * stride_x + kx) as isize - padding as isize;
                                if ix < 0 || ix as usize >= width {
                                    continue;
                                }
                                let ki = (c * size + ky) * size + kx;
                                let wi = (location * filters + f) * kernel + ki;
                                let xi = ((b * channels + c) * height + iy as usize) * width
                                    + ix as usize;
                                dw[wi] += input[xi] * dy;
                                dx[xi] += weights[wi] * dy;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((dx, dw, db))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_forward_and_backward_match_hand_computed_location_major_layout() {
        let input = [1.0, 2.0, 3.0, 4.0];
        // Two output locations, one filter, one input channel, 1x1 kernels.
        let weights = [2.0, -1.0];
        let out = forward(
            &input,
            &weights,
            &[0.5, 1.0],
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            0,
        )
        .unwrap();
        assert_eq!(out, [2.5, -1.0]);
        let (dx, dw, db) = backward(
            &input,
            &weights,
            &[3.0, 5.0],
            1,
            1,
            2,
            2,
            1,
            1,
            2,
            1,
            1,
            1,
            0,
        )
        .unwrap();
        assert_eq!(dx, [6.0, -5.0, 0.0, 0.0]);
        assert_eq!(dw, [3.0, 10.0]);
        assert_eq!(db, [3.0, 5.0]);
    }

    #[test]
    fn local_rejects_invalid_storage() {
        assert!(forward(&[1.0], &[], &[], 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0).is_err());
    }
}
