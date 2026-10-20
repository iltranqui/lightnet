use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= 2e-6, "{label}[{index}]: {a} != {e}");
    }
}

fn coordinate_reference(
    offsets: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    kernel: usize,
    extend_scope: f32,
    morph: usize,
) -> (Vec<f32>, Vec<f32>) {
    let spatial = height * width;
    let center = kernel / 2;
    let mut y_map = vec![0.0; batch * height * width * kernel];
    let mut x_map = vec![0.0; y_map.len()];
    for b in 0..batch {
        for y in 0..height {
            for x in 0..width {
                for k in 0..kernel {
                    let mut cumulative = 0.0;
                    if k > center {
                        for j in center + 1..=k {
                            let c = if morph == 0 { j } else { kernel + j };
                            cumulative += offsets[(b * 2 * kernel + c) * spatial + y * width + x];
                        }
                    } else if k < center {
                        for j in (k..center).rev() {
                            let c = if morph == 0 { j } else { kernel + j };
                            cumulative += offsets[(b * 2 * kernel + c) * spatial + y * width + x];
                        }
                    }
                    let i = ((b * height + y) * width + x) * kernel + k;
                    if morph == 0 {
                        y_map[i] = y as f32 + extend_scope * cumulative;
                        x_map[i] = x as f32 + k as f32 - center as f32;
                    } else {
                        y_map[i] = y as f32 + k as f32 - center as f32;
                        x_map[i] = x as f32 + extend_scope * cumulative;
                    }
                }
            }
        }
    }
    (y_map, x_map)
}

#[test]
fn dsconv_bilinear_backward_matches_cpu_and_integer_boundary_convention() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, kernel) = (2usize, 2usize, 3usize, 4usize, 3usize);
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.07)
        .collect();
    let coords = batch * height * width * kernel;
    let y_map: Vec<f32> = (0..coords)
        .map(|i| match i % 6 {
            0 => -0.2,
            1 => 0.0,
            2 => 0.65,
            3 => 1.0,
            4 => 1.7,
            _ => 2.0,
        })
        .collect();
    let x_map: Vec<f32> = (0..coords)
        .map(|i| match i % 6 {
            0 => 0.0,
            1 => -0.3,
            2 => 0.4,
            3 => 1.0,
            4 => 2.5,
            _ => 3.0,
        })
        .collect();
    let sampled_gradient: Vec<f32> = (0..batch * channels * height * width * kernel)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.031)
        .collect();
    let t_input = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .unwrap();
    let t_y = backend
        .upload_f32(&[batch, height, width, kernel], &y_map)
        .unwrap();
    let t_x = backend
        .upload_f32(&[batch, height, width, kernel], &x_map)
        .unwrap();
    let t_grad = backend
        .upload_f32(&[batch, channels, height, width, kernel], &sampled_gradient)
        .unwrap();
    let (got_input, got_y, got_x) = backend
        .dsconv_bilinear_backward_f32(
            &t_input,
            &t_y,
            &t_x,
            &t_grad,
            batch,
            channels,
            height,
            width,
            kernel,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let axis = |coordinate: f32, length: usize| {
        if coordinate < 0.0 {
            (0, 0, 0.0, false)
        } else if coordinate >= (length - 1) as f32 {
            (length - 1, length - 1, 0.0, false)
        } else {
            let low = coordinate.floor() as usize;
            (low, low + 1, coordinate - low as f32, true)
        }
    };
    let mut want_input = vec![0.0; input.len()];
    let mut want_y = vec![0.0; coords];
    let mut want_x = vec![0.0; coords];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    for k in 0..kernel {
                        let ci = ((b * height + y) * width + x) * kernel + k;
                        let (y0, y1, fy, dy_active) = axis(y_map[ci], height);
                        let (x0, x1, fx, dx_active) = axis(x_map[ci], width);
                        let base = ((b * channels + c) * height) * width;
                        let p00 = input[base + y0 * width + x0];
                        let p01 = input[base + y0 * width + x1];
                        let p10 = input[base + y1 * width + x0];
                        let p11 = input[base + y1 * width + x1];
                        let si = (((b * channels + c) * height + y) * width + x) * kernel + k;
                        let g = sampled_gradient[si];
                        want_input[base + y0 * width + x0] += g * (1.0 - fy) * (1.0 - fx);
                        want_input[base + y0 * width + x1] += g * (1.0 - fy) * fx;
                        want_input[base + y1 * width + x0] += g * fy * (1.0 - fx);
                        want_input[base + y1 * width + x1] += g * fy * fx;
                        if dy_active {
                            want_y[ci] += g * ((1.0 - fx) * (p10 - p00) + fx * (p11 - p01));
                        }
                        if dx_active {
                            want_x[ci] += g * ((1.0 - fy) * (p01 - p00) + fy * (p11 - p10));
                        }
                    }
                }
            }
        }
    }
    close(
        "input adjoint",
        &backend.download_f32(&got_input).unwrap(),
        &want_input,
    );
    close(
        "y-coordinate adjoint",
        &backend.download_f32(&got_y).unwrap(),
        &want_y,
    );
    close(
        "x-coordinate adjoint",
        &backend.download_f32(&got_x).unwrap(),
        &want_x,
    );
}

#[test]
fn dsconv_collapse_matches_nchw_cpu_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, filters, height, width, kernel) =
        (2usize, 3usize, 4usize, 3usize, 5usize, 5usize);
    let sampled: Vec<f32> = (0..batch * channels * height * width * kernel)
        .map(|i| ((i * 19 % 47) as f32 - 23.0) * 0.025)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * kernel)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.009)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| i as f32 * 0.04 - 0.06).collect();
    let t_sampled = backend
        .upload_f32(&[batch, channels, height, width, kernel], &sampled)
        .unwrap();
    let t_weights = backend
        .upload_f32(&[filters, channels, kernel], &weights)
        .unwrap();
    let t_biases = backend.upload_f32(&[filters], &biases).unwrap();
    let got = backend
        .dsconv_collapse_f32(
            &t_sampled,
            &t_weights,
            &t_biases,
            batch,
            channels,
            filters,
            height,
            width,
            kernel,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let mut want = vec![0.0; batch * filters * height * width];
    for b in 0..batch {
        for f in 0..filters {
            for y in 0..height {
                for x in 0..width {
                    let mut value = biases[f];
                    for c in 0..channels {
                        for k in 0..kernel {
                            let si = (((b * channels + c) * height + y) * width + x) * kernel + k;
                            let wi = (f * channels + c) * kernel + k;
                            value += weights[wi] * sampled[si];
                        }
                    }
                    want[((b * filters + f) * height + y) * width + x] = value;
                }
            }
        }
    }
    close("collapse", &backend.download_f32(&got).unwrap(), &want);
}

#[test]
fn dsconv_collapse_backward_matches_cpu_for_channel_partition() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, filters, height, width, kernel) =
        (2usize, 3usize, 4usize, 3usize, 5usize, 5usize);
    let sampled: Vec<f32> = (0..batch * channels * height * width * kernel)
        .map(|i| ((i * 19 % 47) as f32 - 23.0) * 0.025)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * kernel)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.009)
        .collect();
    let gradient: Vec<f32> = (0..batch * filters * height * width)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) * 0.017)
        .collect();
    let (filter_start, filter_count) = (1usize, 2usize);
    let t_sampled = backend
        .upload_f32(&[batch, channels, height, width, kernel], &sampled)
        .unwrap();
    let t_weights = backend
        .upload_f32(&[filters, channels, kernel], &weights)
        .unwrap();
    let t_gradient = backend
        .upload_f32(&[batch, filters, height, width], &gradient)
        .unwrap();
    let (got_sampled, got_weights, got_biases) = backend
        .dsconv_collapse_backward_f32(
            &t_sampled,
            &t_weights,
            &t_gradient,
            batch,
            channels,
            filters,
            height,
            width,
            kernel,
            filter_start,
            filter_count,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let mut want_sampled = vec![0.0; sampled.len()];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    for k in 0..kernel {
                        let mut value = 0.0;
                        for f in filter_start..filter_start + filter_count {
                            let gi = ((b * filters + f) * height + y) * width + x;
                            let wi = (f * channels + c) * kernel + k;
                            value += gradient[gi] * weights[wi];
                        }
                        let si = (((b * channels + c) * height + y) * width + x) * kernel + k;
                        want_sampled[si] = value;
                    }
                }
            }
        }
    }
    let mut want_weights = vec![0.0; weights.len()];
    let mut want_biases = vec![0.0; filters];
    for f in filter_start..filter_start + filter_count {
        for c in 0..channels {
            for k in 0..kernel {
                let mut value = 0.0;
                for b in 0..batch {
                    for y in 0..height {
                        for x in 0..width {
                            let gi = ((b * filters + f) * height + y) * width + x;
                            let si = (((b * channels + c) * height + y) * width + x) * kernel + k;
                            value += gradient[gi] * sampled[si];
                        }
                    }
                }
                want_weights[(f * channels + c) * kernel + k] = value;
            }
        }
        for b in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    want_biases[f] += gradient[((b * filters + f) * height + y) * width + x];
                }
            }
        }
    }
    close(
        "sampled gradient",
        &backend.download_f32(&got_sampled).unwrap(),
        &want_sampled,
    );
    close(
        "weight gradient",
        &backend.download_f32(&got_weights).unwrap(),
        &want_weights,
    );
    close(
        "bias gradient",
        &backend.download_f32(&got_biases).unwrap(),
        &want_biases,
    );
}

#[test]
fn dsconv_offset_map_backward_matches_prefix_suffix_and_tanh_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, height, width, kernel, extend_scope) = (2usize, 3usize, 4usize, 5usize, 0.73f32);
    let spatial = height * width;
    let offsets: Vec<f32> = (0..batch * 2 * kernel * spatial)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.035)
        .collect();
    let coord_len = batch * height * width * kernel;
    let y_gradient: Vec<f32> = (0..coord_len)
        .map(|i| ((i * 13 % 23) as f32 - 11.0) * 0.021)
        .collect();
    let x_gradient: Vec<f32> = (0..coord_len)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.027)
        .collect();
    let t_offsets = backend
        .upload_f32(&[batch, 2 * kernel, height, width], &offsets)
        .unwrap();
    let t_y = backend
        .upload_f32(&[batch, height, width, kernel], &y_gradient)
        .unwrap();
    let t_x = backend
        .upload_f32(&[batch, height, width, kernel], &x_gradient)
        .unwrap();
    for morph in 0..=2 {
        let got = backend
            .dsconv_offset_map_backward_f32(
                &t_offsets,
                &t_y,
                &t_x,
                batch,
                height,
                width,
                kernel,
                extend_scope,
                morph,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let mut want = vec![0.0; offsets.len()];
        let center = kernel / 2;
        for b in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    for axis in 0..2 {
                        for k in 0..kernel {
                            if k == center || (axis == 0 && morph == 1) || (axis == 1 && morph == 0)
                            {
                                continue;
                            }
                            let mut cumulative = 0.0;
                            if k < center {
                                for j in 0..=k {
                                    let ci = ((b * height + y) * width + x) * kernel + j;
                                    cumulative += if axis == 0 {
                                        y_gradient[ci]
                                    } else {
                                        x_gradient[ci]
                                    };
                                }
                            } else {
                                for j in (k..kernel).rev() {
                                    let ci = ((b * height + y) * width + x) * kernel + j;
                                    cumulative += if axis == 0 {
                                        y_gradient[ci]
                                    } else {
                                        x_gradient[ci]
                                    };
                                }
                            }
                            let channel = axis * kernel + k;
                            let oi = ((b * 2 * kernel + channel) * height + y) * width + x;
                            want[oi] =
                                cumulative * extend_scope * (1.0 - offsets[oi] * offsets[oi]);
                        }
                    }
                }
            }
        }
        close(
            &format!("morph {morph} offset adjoint"),
            &backend.download_f32(&got).unwrap(),
            &want,
        );
    }
}

#[test]
fn dsconv_offset_predictor_backward_matches_cpu_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, filters) = (2usize, 2usize, 3usize, 5usize, 6usize);
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.09)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * 9)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.004)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| (i as f32 - 4.0) * 0.025).collect();
    let seed: Vec<f32> = (0..batch * filters * height * width)
        .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.041)
        .collect();
    let x = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .unwrap();
    let w = backend
        .upload_f32(&[filters, channels, 3, 3], &weights)
        .unwrap();
    let b = backend.upload_f32(&[filters], &biases).unwrap();
    let t_seed = backend
        .upload_f32(&[batch, filters, height, width], &seed)
        .unwrap();
    let output_shape = [batch, filters, height, width];
    let conv = backend
        .conv2d_f32(
            &x,
            &w,
            &output_shape,
            channels,
            height,
            width,
            filters,
            3,
            1,
            1,
            1,
            1,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let biased = backend
        .bias_add_f32(
            &conv,
            &b,
            filters,
            height * width,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let raw = backend.download_f32(&biased).unwrap();
    let delta = backend
        .activation_backward_f32(
            &biased,
            &t_seed,
            darknet_cuda::F32ActivationKind::Tanh,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let dx = backend
        .conv2d_backward_input_f32(
            &delta,
            &w,
            &[batch, channels, height, width],
            channels,
            height,
            width,
            filters,
            3,
            1,
            1,
            1,
            1,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let dw = backend
        .conv2d_backward_weights_f32(
            &x,
            &delta,
            &[filters, channels, 3, 3],
            channels,
            height,
            width,
            filters,
            3,
            1,
            1,
            1,
            1,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let db = backend
        .bias_backward_f32(
            &delta,
            filters,
            height * width,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let mut want_delta = vec![0.0; seed.len()];
    for i in 0..seed.len() {
        let t = raw[i].tanh();
        want_delta[i] = seed[i] * (1.0 - t * t);
    }
    let mut want_dx = vec![0.0; input.len()];
    let mut want_dw = vec![0.0; weights.len()];
    let mut want_db = vec![0.0; filters];
    for n in 0..batch {
        for f in 0..filters {
            for y in 0..height {
                for x_pos in 0..width {
                    let oi = ((n * filters + f) * height + y) * width + x_pos;
                    let g = want_delta[oi];
                    want_db[f] += g;
                    for c in 0..channels {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = y as isize + ky as isize - 1;
                                let ix = x_pos as isize + kx as isize - 1;
                                let wi = (f * channels + c) * 9 + ky * 3 + kx;
                                if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize
                                {
                                    let xi = ((n * channels + c) * height + iy as usize) * width
                                        + ix as usize;
                                    want_dw[wi] += g * input[xi];
                                    want_dx[xi] += g * weights[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    close(
        "offset d_input",
        &backend.download_f32(&dx).unwrap(),
        &want_dx,
    );
    close(
        "offset d_weights",
        &backend.download_f32(&dw).unwrap(),
        &want_dw,
    );
    close(
        "offset d_biases",
        &backend.download_f32(&db).unwrap(),
        &want_db,
    );
}

#[test]
fn dsconv_composed_forward_matches_cpu_reference_for_all_morphs() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, kernel, filters, extend_scope) =
        (2usize, 2usize, 3usize, 5usize, 3usize, 5usize, 0.61f32);
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.09)
        .collect();
    let offset_weights: Vec<f32> = (0..2 * kernel * channels * 9)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.004)
        .collect();
    let offset_biases: Vec<f32> = (0..2 * kernel).map(|i| (i as f32 - 2.0) * 0.025).collect();
    let weights: Vec<f32> = (0..filters * channels * kernel)
        .map(|i| ((i * 5 % 29) as f32 - 14.0) * 0.018)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| i as f32 * 0.035 - 0.07).collect();
    let t_input = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .unwrap();
    let t_offset_weights = backend
        .upload_f32(&[2 * kernel, channels, 3, 3], &offset_weights)
        .unwrap();
    let t_offset_biases = backend.upload_f32(&[2 * kernel], &offset_biases).unwrap();
    let t_weights = backend
        .upload_f32(&[filters, channels, kernel], &weights)
        .unwrap();
    let t_biases = backend.upload_f32(&[filters], &biases).unwrap();

    let mut offsets = vec![0.0; batch * 2 * kernel * height * width];
    for b in 0..batch {
        for f in 0..2 * kernel {
            for y in 0..height {
                for x in 0..width {
                    let mut value = offset_biases[f];
                    for c in 0..channels {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = y as isize + ky as isize - 1;
                                let ix = x as isize + kx as isize - 1;
                                if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize
                                {
                                    let xi = ((b * channels + c) * height + iy as usize) * width
                                        + ix as usize;
                                    let wi = ((f * channels + c) * 3 + ky) * 3 + kx;
                                    value += input[xi] * offset_weights[wi];
                                }
                            }
                        }
                    }
                    offsets[((b * 2 * kernel + f) * height + y) * width + x] = value.tanh();
                }
            }
        }
    }
    for morph in 0..=2 {
        let got = backend
            .dsconv_forward_native_f32(
                &t_input,
                &t_offset_weights,
                &t_offset_biases,
                &t_weights,
                &t_biases,
                batch,
                channels,
                filters,
                height,
                width,
                kernel,
                extend_scope,
                morph,
                darknet_cuda::F32ActivationKind::Linear,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let (y_map, x_map) = coordinate_reference(
            &offsets,
            batch,
            height,
            width,
            kernel,
            extend_scope,
            if morph == 1 { 1 } else { 0 },
        );
        let (y_map_alt, x_map_alt) = if morph == 2 {
            coordinate_reference(&offsets, batch, height, width, kernel, extend_scope, 1)
        } else {
            (y_map.clone(), x_map.clone())
        };
        let sample_ref = |ys: &[f32], xs: &[f32]| {
            let axis = |coordinate: f32, length: usize| {
                if coordinate < 0.0 {
                    (0, 0, 0.0)
                } else if coordinate >= (length - 1) as f32 {
                    (length - 1, length - 1, 0.0)
                } else {
                    let low = coordinate.floor() as usize;
                    (low, low + 1, coordinate - low as f32)
                }
            };
            let mut samples = vec![0.0; batch * channels * height * width * kernel];
            for b in 0..batch {
                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            for k in 0..kernel {
                                let ci = ((b * height + y) * width + x) * kernel + k;
                                let (y0, y1, fy) = axis(ys[ci], height);
                                let (x0, x1, fx) = axis(xs[ci], width);
                                let base = ((b * channels + c) * height) * width;
                                let p00 = input[base + y0 * width + x0];
                                let p01 = input[base + y0 * width + x1];
                                let p10 = input[base + y1 * width + x0];
                                let p11 = input[base + y1 * width + x1];
                                let top = p00 + fx * (p01 - p00);
                                let bottom = p10 + fx * (p11 - p10);
                                samples[(((b * channels + c) * height + y) * width + x) * kernel
                                    + k] = top + fy * (bottom - top);
                            }
                        }
                    }
                }
            }
            samples
        };
        let (samples_x, samples_y) = if morph == 0 {
            (sample_ref(&y_map, &x_map), Vec::new())
        } else if morph == 1 {
            (Vec::new(), sample_ref(&y_map, &x_map))
        } else {
            (
                sample_ref(&y_map, &x_map),
                sample_ref(&y_map_alt, &x_map_alt),
            )
        };
        let mut want = vec![0.0; batch * filters * height * width];
        let x_filters = filters / 2;
        for b in 0..batch {
            for f in 0..filters {
                for y in 0..height {
                    for x in 0..width {
                        let use_x = morph == 0 || (morph == 2 && f < x_filters);
                        let samples = if use_x { &samples_x } else { &samples_y };
                        let mut value = biases[f];
                        for c in 0..channels {
                            for k in 0..kernel {
                                let si =
                                    (((b * channels + c) * height + y) * width + x) * kernel + k;
                                value += weights[(f * channels + c) * kernel + k] * samples[si];
                            }
                        }
                        want[((b * filters + f) * height + y) * width + x] = value;
                    }
                }
            }
        }
        close(
            &format!("composed morph {morph}"),
            &backend.download_f32(&got).unwrap(),
            &want,
        );
    }
}

#[test]
fn dsconv_composed_backward_matches_finite_differences_for_all_morphs() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, kernel, filters) =
        (1usize, 1usize, 3usize, 4usize, 3usize, 3usize);
    let extend_scope = 0.7f32;
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.14)
        .collect();
    let offset_weights: Vec<f32> = (0..2 * kernel * channels * 9)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.012)
        .collect();
    let offset_biases: Vec<f32> = (0..2 * kernel).map(|i| (i as f32 - 2.0) * 0.027).collect();
    let weights: Vec<f32> = (0..filters * channels * kernel)
        .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.04)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| i as f32 * 0.05 - 0.04).collect();
    let output_gradient: Vec<f32> = (0..batch * filters * height * width)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.08)
        .collect();
    let epsilon = 1e-3f32;
    let loss = |morph: usize, input: &[f32], ow: &[f32], ob: &[f32], w: &[f32], b: &[f32]| {
        let t_input = backend
            .upload_f32(&[batch, channels, height, width], input)
            .unwrap();
        let t_ow = backend
            .upload_f32(&[2 * kernel, channels, 3, 3], ow)
            .unwrap();
        let t_ob = backend.upload_f32(&[2 * kernel], ob).unwrap();
        let t_w = backend.upload_f32(&[filters, channels, kernel], w).unwrap();
        let t_b = backend.upload_f32(&[filters], b).unwrap();
        let output = backend
            .dsconv_forward_native_f32(
                &t_input,
                &t_ow,
                &t_ob,
                &t_w,
                &t_b,
                batch,
                channels,
                filters,
                height,
                width,
                kernel,
                extend_scope,
                morph,
                darknet_cuda::F32ActivationKind::Linear,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        backend
            .download_f32(&output)
            .unwrap()
            .iter()
            .zip(&output_gradient)
            .map(|(x, g)| x * g)
            .sum::<f32>()
    };
    let finite_difference = |morph: usize, group: usize, index: usize| {
        let mut plus_input = input.clone();
        let mut minus_input = input.clone();
        let mut plus_ow = offset_weights.clone();
        let mut minus_ow = offset_weights.clone();
        let mut plus_ob = offset_biases.clone();
        let mut minus_ob = offset_biases.clone();
        let mut plus_w = weights.clone();
        let mut minus_w = weights.clone();
        let mut plus_b = biases.clone();
        let mut minus_b = biases.clone();
        let (plus, minus) = match group {
            0 => {
                plus_input[index] += epsilon;
                minus_input[index] -= epsilon;
                (
                    loss(
                        morph,
                        &plus_input,
                        &offset_weights,
                        &offset_biases,
                        &weights,
                        &biases,
                    ),
                    loss(
                        morph,
                        &minus_input,
                        &offset_weights,
                        &offset_biases,
                        &weights,
                        &biases,
                    ),
                )
            }
            1 => {
                plus_w[index] += epsilon;
                minus_w[index] -= epsilon;
                (
                    loss(
                        morph,
                        &input,
                        &offset_weights,
                        &offset_biases,
                        &plus_w,
                        &biases,
                    ),
                    loss(
                        morph,
                        &input,
                        &offset_weights,
                        &offset_biases,
                        &minus_w,
                        &biases,
                    ),
                )
            }
            2 => {
                plus_b[index] += epsilon;
                minus_b[index] -= epsilon;
                (
                    loss(
                        morph,
                        &input,
                        &offset_weights,
                        &offset_biases,
                        &weights,
                        &plus_b,
                    ),
                    loss(
                        morph,
                        &input,
                        &offset_weights,
                        &offset_biases,
                        &weights,
                        &minus_b,
                    ),
                )
            }
            3 => {
                plus_ow[index] += epsilon;
                minus_ow[index] -= epsilon;
                (
                    loss(morph, &input, &plus_ow, &offset_biases, &weights, &biases),
                    loss(morph, &input, &minus_ow, &offset_biases, &weights, &biases),
                )
            }
            _ => {
                plus_ob[index] += epsilon;
                minus_ob[index] -= epsilon;
                (
                    loss(morph, &input, &offset_weights, &plus_ob, &weights, &biases),
                    loss(morph, &input, &offset_weights, &minus_ob, &weights, &biases),
                )
            }
        };
        (plus - minus) / (2.0 * epsilon)
    };
    for morph in 0..=2 {
        let t_input = backend
            .upload_f32(&[batch, channels, height, width], &input)
            .unwrap();
        let t_ow = backend
            .upload_f32(&[2 * kernel, channels, 3, 3], &offset_weights)
            .unwrap();
        let t_ob = backend.upload_f32(&[2 * kernel], &offset_biases).unwrap();
        let t_w = backend
            .upload_f32(&[filters, channels, kernel], &weights)
            .unwrap();
        let t_b = backend.upload_f32(&[filters], &biases).unwrap();
        let t_grad = backend
            .upload_f32(&[batch, filters, height, width], &output_gradient)
            .unwrap();
        let (got_x, got_w, got_b, got_ow, got_ob) = backend
            .dsconv_backward_native_f32(
                &t_input,
                &t_ow,
                &t_ob,
                &t_w,
                &t_b,
                &t_grad,
                batch,
                channels,
                filters,
                height,
                width,
                kernel,
                extend_scope,
                morph,
                darknet_cuda::F32ActivationKind::Linear,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        let analytic = [
            backend.download_f32(&got_x).unwrap()[5],
            backend.download_f32(&got_w).unwrap()[2],
            backend.download_f32(&got_b).unwrap()[1],
            backend.download_f32(&got_ow).unwrap()
                [if morph == 1 { (kernel + 0) * 9 + 4 } else { 4 }],
            backend.download_f32(&got_ob).unwrap()[if morph == 1 { kernel } else { 0 }],
        ];
        let indices = [
            5usize,
            2,
            1,
            if morph == 1 { kernel * 9 + 4 } else { 4 },
            if morph == 1 { kernel } else { 0 },
        ];
        for group in 0..5 {
            let numeric = finite_difference(morph, group, indices[group]);
            let error = (analytic[group] - numeric).abs();
            assert!(
                error <= 4e-3 * (1.0 + numeric.abs()),
                "morph={morph} group={group} index={} analytic={} numeric={} error={}",
                indices[group],
                analytic[group],
                numeric,
                error
            );
        }
    }
}

#[test]
fn dsconv_offset_predictor_and_tanh_match_cpu_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, kernel) = (2usize, 2usize, 3usize, 5usize, 5usize);
    let filters = 2 * kernel;
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.09)
        .collect();
    let weights: Vec<f32> = (0..filters * channels * 9)
        .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.004)
        .collect();
    let biases: Vec<f32> = (0..filters).map(|i| (i as f32 - 4.0) * 0.025).collect();
    let x = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .unwrap();
    let w = backend
        .upload_f32(&[filters, channels, 3, 3], &weights)
        .unwrap();
    let b = backend.upload_f32(&[filters], &biases).unwrap();
    let shape = [batch, filters, height, width];
    let conv = backend
        .conv2d_f32(
            &x,
            &w,
            &shape,
            channels,
            height,
            width,
            filters,
            3,
            1,
            1,
            1,
            1,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let biased = backend
        .bias_add_f32(
            &conv,
            &b,
            filters,
            height * width,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let got = backend
        .activation_f32(
            &biased,
            darknet_cuda::F32ActivationKind::Tanh,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let mut want = vec![0.0; batch * filters * height * width];
    for n in 0..batch {
        for f in 0..filters {
            for y in 0..height {
                for x in 0..width {
                    let mut value = biases[f];
                    for c in 0..channels {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let iy = y as isize + ky as isize - 1;
                                let ix = x as isize + kx as isize - 1;
                                if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize
                                {
                                    let xi = ((n * channels + c) * height + iy as usize) * width
                                        + ix as usize;
                                    let wi = (f * channels + c) * 9 + ky * 3 + kx;
                                    value += input[xi] * weights[wi];
                                }
                            }
                        }
                    }
                    want[((n * filters + f) * height + y) * width + x] = value.tanh();
                }
            }
        }
    }
    close(
        "offset predictor",
        &backend.download_f32(&got).unwrap(),
        &want,
    );
}

#[test]
fn dsconv_bilinear_sampling_matches_cpu_with_border_clamps() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, channels, height, width, kernel) = (2usize, 3usize, 3usize, 5usize, 3usize);
    let input: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i * 17 % 41) as f32 - 20.0) * 0.11)
        .collect();
    let positions = batch * height * width * kernel;
    let y_map: Vec<f32> = (0..positions)
        .map(|i| match i % 5 {
            0 => -0.25,
            1 => 0.0,
            2 => 0.75,
            3 => 1.5,
            _ => 4.2,
        })
        .collect();
    let x_map: Vec<f32> = (0..positions)
        .map(|i| match i % 5 {
            0 => 0.3,
            1 => -1.0,
            2 => 1.25,
            3 => 3.8,
            _ => 4.0,
        })
        .collect();
    let t_input = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .unwrap();
    let t_y = backend
        .upload_f32(&[batch, height, width, kernel], &y_map)
        .unwrap();
    let t_x = backend
        .upload_f32(&[batch, height, width, kernel], &x_map)
        .unwrap();
    let got = backend
        .dsconv_bilinear_sample_f32(
            &t_input,
            &t_y,
            &t_x,
            batch,
            channels,
            height,
            width,
            kernel,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let axis = |coordinate: f32, length: usize| {
        if coordinate < 0.0 {
            (0, 0, 0.0)
        } else if coordinate >= (length - 1) as f32 {
            (length - 1, length - 1, 0.0)
        } else {
            let low = coordinate.floor() as usize;
            (low, low + 1, coordinate - low as f32)
        }
    };
    let mut want = vec![0.0; batch * channels * height * width * kernel];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    for k in 0..kernel {
                        let ci = ((b * height + y) * width + x) * kernel + k;
                        let (y0, y1, fy) = axis(y_map[ci], height);
                        let (x0, x1, fx) = axis(x_map[ci], width);
                        let base = ((b * channels + c) * height) * width;
                        let p00 = input[base + y0 * width + x0];
                        let p01 = input[base + y0 * width + x1];
                        let p10 = input[base + y1 * width + x0];
                        let p11 = input[base + y1 * width + x1];
                        let top = p00 + fx * (p01 - p00);
                        let bottom = p10 + fx * (p11 - p10);
                        want[(((b * channels + c) * height + y) * width + x) * kernel + k] =
                            top + fy * (bottom - top);
                    }
                }
            }
        }
    }
    close(
        "deformed tensor",
        &backend.download_f32(&got).unwrap(),
        &want,
    );
}

#[test]
fn dsconv_coordinate_maps_match_cpu_for_both_snake_axes() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, height, width, kernel, extend_scope) = (2usize, 3usize, 5usize, 5usize, 0.73f32);
    let spatial = height * width;
    let offsets: Vec<f32> = (0..batch * 2 * kernel * spatial)
        .map(|i| ((i * 11 % 29) as f32 - 14.0) * 0.013)
        .collect();
    let tensor = backend
        .upload_f32(&[batch, 2 * kernel, height, width], &offsets)
        .unwrap();
    for morph in 0..=1 {
        let (want_y, want_x) =
            coordinate_reference(&offsets, batch, height, width, kernel, extend_scope, morph);
        let (got_y, got_x) = backend
            .dsconv_coordinate_map_f32(
                &tensor,
                batch,
                height,
                width,
                kernel,
                extend_scope,
                morph,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        close("y_map", &backend.download_f32(&got_y).unwrap(), &want_y);
        close("x_map", &backend.download_f32(&got_x).unwrap(), &want_x);
    }
}

#[test]
fn dsconv_coordinate_maps_reject_invalid_geometry() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let tensor = backend.upload_f32(&[1, 6, 2, 3], &[0.0; 36]).unwrap();
    assert!(
        backend
            .dsconv_coordinate_map_f32(
                &tensor,
                1,
                2,
                3,
                3,
                1.0,
                2,
                ElementwiseLaunchConfig::default()
            )
            .is_err()
    );
    assert!(
        backend
            .dsconv_coordinate_map_f32(
                &tensor,
                1,
                2,
                3,
                3,
                f32::NAN,
                0,
                ElementwiseLaunchConfig::default()
            )
            .is_err()
    );
    assert!(
        backend
            .dsconv_coordinate_map_f32(
                &tensor,
                1,
                2,
                3,
                2,
                1.0,
                0,
                ElementwiseLaunchConfig::default()
            )
            .is_err()
    );
}
