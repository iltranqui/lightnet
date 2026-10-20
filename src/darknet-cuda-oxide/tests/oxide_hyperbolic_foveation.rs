use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn bilinear_warp_reference(input: &[f32], grid: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let [batch, channels, height, width] = shape;
    let spatial = height * width;
    let mut output = vec![0.0; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let pixel = y * width + x;
                    let grid_index = (b * spatial + pixel) * 2;
                    let sx = grid[grid_index];
                    let sy = grid[grid_index + 1];
                    let x0 = sx.floor() as isize;
                    let y0 = sy.floor() as isize;
                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;
                    let base = (b * channels + c) * spatial;
                    let mut value = 0.0;
                    for tap in 0..4 {
                        let tx = x0 + (tap & 1) as isize;
                        let ty = y0 + (tap >> 1) as isize;
                        if tx >= 0 && tx < width as isize && ty >= 0 && ty < height as isize {
                            let wx = if tap & 1 == 1 { dx } else { 1.0 - dx };
                            let wy = if tap >> 1 == 1 { dy } else { 1.0 - dy };
                            value += wy * wx * input[base + ty as usize * width + tx as usize];
                        }
                    }
                    output[base + pixel] = value;
                }
            }
        }
    }
    output
}

fn warp_backward_reference(
    input: &[f32], grid: &[f32], seed: &[f32], shape: [usize; 4],
) -> (Vec<f32>, Vec<f32>) {
    let [batch, channels, height, width] = shape;
    let spatial = height * width;
    let mut dx = vec![0.0; input.len()];
    let mut dg = vec![0.0; grid.len()];
    for b in 0..batch {
        for p in 0..spatial {
            let gi = (b * spatial + p) * 2;
            let x = grid[gi];
            let y = grid[gi + 1];
            let x0 = x.floor() as isize;
            let y0 = y.floor() as isize;
            let fx = x - x0 as f32;
            let fy = y - y0 as f32;
            for c in 0..channels {
                let base = (b * channels + c) * spatial;
                let at = |r: isize, col: isize| {
                    if r >= 0 && r < height as isize && col >= 0 && col < width as isize {
                        input[base + r as usize * width + col as usize]
                    } else { 0.0 }
                };
                let (a, b, c, d) = (at(y0, x0), at(y0, x0 + 1), at(y0 + 1, x0), at(y0 + 1, x0 + 1));
                let g = seed[base + p];
                dg[gi] += g * ((1.0 - fy) * (b - a) + fy * (d - c));
                dg[gi + 1] += g * ((1.0 - fx) * (c - a) + fx * (d - b));
                for (r, col, w) in [
                    (y0, x0, (1.0 - fy) * (1.0 - fx)),
                    (y0, x0 + 1, (1.0 - fy) * fx),
                    (y0 + 1, x0, fy * (1.0 - fx)),
                    (y0 + 1, x0 + 1, fy * fx),
                ] {
                    if r >= 0 && r < height as isize && col >= 0 && col < width as isize {
                        dx[base + r as usize * width + col as usize] += g * w;
                    }
                }
            }
        }
    }
    (dx, dg)
}

#[test]
fn hyperbolic_foveation_native_warp_backward_matches_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [1, 2, 3, 4];
    let input: Vec<f32> = (0..24).map(|i| (i as f32 * 0.31).sin()).collect();
    let grid = vec![
        -0.35, 0.25, 0.4, 0.3, 1.45, 0.4, 3.25, 0.2,
        0.2, 1.3, 1.4, 1.45, 2.3, 1.25, 3.2, 2.35,
        -0.2, 2.3, 0.65, 2.4, 1.6, 2.3, 3.4, 2.2,
    ];
    let seed: Vec<f32> = (0..24).map(|i| (i as f32 * 0.23).cos()).collect();
    let input_gpu = backend.upload_f32(&shape, &input).unwrap();
    let grid_gpu = backend.upload_f32(&[1, 3, 4, 2], &grid).unwrap();
    let seed_gpu = backend.upload_f32(&shape, &seed).unwrap();
    let (dx_gpu, dg_gpu) = backend.hyperbolic_foveation_warp_backward_f32(
        &input_gpu, &grid_gpu, &seed_gpu, &shape, ElementwiseLaunchConfig::default(),
    ).unwrap();
    let (expected_dx, expected_dg) = warp_backward_reference(&input, &grid, &seed, shape);
    let actual_dx = backend.download_f32(&dx_gpu).unwrap();
    let actual_dg = backend.download_f32(&dg_gpu).unwrap();
    for (i, (actual, expected)) in actual_dx.iter().zip(&expected_dx).enumerate() {
        assert!((actual - expected).abs() < 2e-6, "dx[{i}]: {actual} != {expected}");
    }
    for (i, (actual, expected)) in actual_dg.iter().zip(&expected_dg).enumerate() {
        assert!((actual - expected).abs() < 2e-6, "dgrid[{i}]: {actual} != {expected}");
    }
    let loss = |input: &[f32], grid: &[f32]| -> f32 {
        bilinear_warp_reference(input, grid, shape)
            .iter().zip(&seed).map(|(out, g)| out * g).sum()
    };
    let eps = 5.0e-3;
    for index in [3, 9, 17] {
        let mut plus = input.clone();
        let mut minus = input.clone();
        plus[index] += eps;
        minus[index] -= eps;
        let numeric = (loss(&plus, &grid) - loss(&minus, &grid)) / (2.0 * eps);
        assert!((actual_dx[index] - numeric).abs() < 1e-3, "finite-difference dx[{index}]: {} != {numeric}", actual_dx[index]);
    }
    for index in [4, 5, 12, 13] {
        let mut plus = grid.clone();
        let mut minus = grid.clone();
        plus[index] += eps;
        minus[index] -= eps;
        let numeric = (loss(&input, &plus) - loss(&input, &minus)) / (2.0 * eps);
        assert!((actual_dg[index] - numeric).abs() < 1e-3, "finite-difference dgrid[{index}]: {} != {numeric}", actual_dg[index]);
    }
}

#[test]
fn hyperbolic_foveation_native_warp_matches_nchw_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [2, 2, 2, 3];
    let input: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| (i as f32 * 0.37).sin() - 0.2)
        .collect();
    let grid = vec![
        -1.0, 0.0, 0.25, 0.0, 1.5, 0.0, 2.0, 0.5, 0.0, 1.0, 3.0, 1.0,
        0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.25, 1.0, 2.0, 1.0,
    ];
    let input_gpu = backend.upload_f32(&shape, &input).unwrap();
    let grid_gpu = backend.upload_f32(&[2, 2, 3, 2], &grid).unwrap();
    let output = backend
        .hyperbolic_foveation_warp_f32(
            &input_gpu,
            &grid_gpu,
            &shape,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let expected = bilinear_warp_reference(&input, &grid, shape);
    let actual = backend.download_f32(&output).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "HF warp[{index}]: {actual} != {expected}"
        );
    }
}

#[test]
fn hyperbolic_foveation_native_predictor_matches_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [2, 1, 2, 2];
    let input = [0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, 0.9];
    let w1 = [0.5, -0.3];
    let b1 = [0.1, 0.2];
    let w2 = [0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.7, -0.6];
    let b2 = [0.03, -0.04, 0.1, -0.2];
    let input_gpu = backend.upload_f32(&shape, &input).unwrap();
    let actual = backend
        .hyperbolic_foveation_predictor_f32(
            &input_gpu,
            &shape,
            2,
            2,
            4,
            &w1,
            &b1,
            &w2,
            &b2,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let actual = backend.download_f32(&actual).unwrap();
    let mut expected = Vec::new();
    for b in 0..2 {
        let pooled = input[b * 4..b * 4 + 4].iter().sum::<f32>() / 4.0;
        let hidden = [(w1[0] * pooled + b1[0]).max(0.0), (w1[1] * pooled + b1[1]).max(0.0)];
        let raw: Vec<f32> = (0..4)
            .map(|p| b2[p] + w2[p * 2] * hidden[0] + w2[p * 2 + 1] * hidden[1])
            .collect();
        for (p, value) in raw.into_iter().enumerate() {
            expected.push(if p < 2 {
                0.9 * value.tanh()
            } else if p == 3 {
                if value >= 0.0 { 1.0 / (1.0 + (-value).exp()) } else { let exp = value.exp(); exp / (1.0 + exp) }
            } else {
                (2.0 * value.exp().ln_1p() + 0.1).clamp(0.1, 2.0)
            });
        }
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() <= 3e-6, "predictor[{index}]: {actual} != {expected}");
    }
}

#[test]
fn hyperbolic_foveation_native_predictor_backward_matches_finite_differences() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let shape = [2, 1, 3, 5];
    let downsample = 2;
    let hidden_count = 2;
    let param_count = 5;
    let input: Vec<f32> = (0..30).map(|i| (i as f32 * 0.19).sin() * 0.7).collect();
    let w1 = [0.35, -0.2, -0.15, 0.4];
    let b1 = [0.55, 0.42];
    let w2 = [0.12, -0.07, -0.08, 0.1, 0.04, -0.06, 0.3, -0.2, 0.5, -0.4];
    let b2 = [0.05, -0.08, -0.15, 0.2, 0.35];
    let seed: Vec<f32> = (0..10).map(|i| (i as f32 * 0.37).cos() * 0.6).collect();
    let input_dim = 2;
    let mut pooled = vec![0.0; 4];
    for n in 0..2 {
        for x in 0..2 {
            let base = n * 15;
            pooled[n * 2 + x] = (input[base + x * 2] + input[base + x * 2 + 1]
                + input[base + 5 + x * 2] + input[base + 5 + x * 2 + 1]) / 4.0;
        }
    }
    let mut pre = vec![0.0; 4];
    let mut hidden = vec![0.0; 4];
    let mut raw = vec![0.0; 10];
    for n in 0..2 {
        for h in 0..hidden_count {
            pre[n * hidden_count + h] = b1[h] + w1[h * input_dim] * pooled[n * input_dim]
                + w1[h * input_dim + 1] * pooled[n * input_dim + 1];
            hidden[n * hidden_count + h] = pre[n * hidden_count + h].max(0.0);
        }
        for p in 0..param_count {
            raw[n * param_count + p] = b2[p] + w2[p * hidden_count] * hidden[n * hidden_count]
                + w2[p * hidden_count + 1] * hidden[n * hidden_count + 1];
        }
    }
    let pooled_gpu = backend.upload_f32(&[2, input_dim], &pooled).unwrap();
    let pre_gpu = backend.upload_f32(&[2, hidden_count], &pre).unwrap();
    let hidden_gpu = backend.upload_f32(&[2, hidden_count], &hidden).unwrap();
    let raw_gpu = backend.upload_f32(&[2, param_count], &raw).unwrap();
    let seed_gpu = backend.upload_f32(&[2, param_count], &seed).unwrap();
    let w1_gpu = backend.upload_f32(&[hidden_count, input_dim], &w1).unwrap();
    let w2_gpu = backend.upload_f32(&[param_count, hidden_count], &w2).unwrap();
    let (dsource_gpu, dw1_db1_gpu, dw2_db2_gpu) = backend.hyperbolic_foveation_predictor_backward_f32(
        &pooled_gpu, &pre_gpu, &hidden_gpu, &raw_gpu, &seed_gpu, &w1_gpu, &w2_gpu,
        &shape, downsample, hidden_count, param_count, ElementwiseLaunchConfig::default(),
    ).unwrap();
    let dsource = backend.download_f32(&dsource_gpu).unwrap();
    let dw1_db1 = backend.download_f32(&dw1_db1_gpu).unwrap();
    let dw2_db2 = backend.download_f32(&dw2_db2_gpu).unwrap();
    let loss = |input: &[f32], w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]| {
        let mut result = 0.0;
        for n in 0..2 {
            let base = n * 15;
            let pool = [
                (input[base] + input[base + 1] + input[base + 5] + input[base + 6]) * 0.25,
                (input[base + 2] + input[base + 3] + input[base + 7] + input[base + 8]) * 0.25,
            ];
            let h = [
                (b1[0] + w1[0] * pool[0] + w1[1] * pool[1]).max(0.0),
                (b1[1] + w1[2] * pool[0] + w1[3] * pool[1]).max(0.0),
            ];
            for p in 0..param_count {
                let r = b2[p] + w2[p * 2] * h[0] + w2[p * 2 + 1] * h[1];
                let value = if p < 2 { 0.9 * r.tanh() } else if p == 4 {
                    if r >= 0.0 { 1.0 / (1.0 + (-r).exp()) } else { let e = r.exp(); e / (1.0 + e) }
                } else { (2.0 * (1.0 + r.exp()).ln() + 0.1).clamp(0.1, 2.0) };
                result += value * seed[n * param_count + p];
            }
        }
        result
    };
    let eps = 2.0e-3;
    let check = |name: &str, actual: f32, numeric: f32| {
        assert!((actual - numeric).abs() < 2e-3, "{name}: analytic {actual}, numeric {numeric}");
    };
    for index in [0, 3, 5, 14, 17, 29] {
        let mut plus = input.clone(); let mut minus = input.clone();
        plus[index] += eps; minus[index] -= eps;
        check(&format!("dsource[{index}]"), dsource[index], (loss(&plus, &w1, &b1, &w2, &b2) - loss(&minus, &w1, &b1, &w2, &b2)) / (2.0 * eps));
    }
    for index in [0, 1, 2, 3] {
        let mut plus = w1; let mut minus = w1;
        plus[index] += eps; minus[index] -= eps;
        check(&format!("dw1[{index}]"), dw1_db1[index], (loss(&input, &plus, &b1, &w2, &b2) - loss(&input, &minus, &b1, &w2, &b2)) / (2.0 * eps));
    }
    for index in [0, 1] {
        let mut plus = b1; let mut minus = b1;
        plus[index] += eps; minus[index] -= eps;
        check(&format!("db1[{index}]"), dw1_db1[4 + index], (loss(&input, &w1, &plus, &w2, &b2) - loss(&input, &w1, &minus, &w2, &b2)) / (2.0 * eps));
    }
    for index in [0, 3, 7, 9] {
        let mut plus = w2; let mut minus = w2;
        plus[index] += eps; minus[index] -= eps;
        check(&format!("dw2[{index}]"), dw2_db2[index], (loss(&input, &w1, &b1, &plus, &b2) - loss(&input, &w1, &b1, &minus, &b2)) / (2.0 * eps));
    }
    for index in [0, 2, 4] {
        let mut plus = b2; let mut minus = b2;
        plus[index] += eps; minus[index] -= eps;
        check(&format!("db2[{index}]"), dw2_db2[10 + index], (loss(&input, &w1, &b1, &w2, &plus) - loss(&input, &w1, &b1, &w2, &minus)) / (2.0 * eps));
    }
}

#[test]
fn hyperbolic_foveation_native_fixed_point_param_grad_matches_finite_differences() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, height, width) = (2, 3, 4);
    let (alpha, p_exp, iterations, eta) = (1.3f32, 2.2f32, 5usize, 0.45f32);
    let config = ElementwiseLaunchConfig::default();
    let grid_seed: Vec<f32> = (0..batch * height * width * 2)
        .map(|i| (i as f32 * 0.31).cos() * 0.4)
        .collect();
    let grid_seed_gpu = backend.upload_f32(&[batch, height, width, 2], &grid_seed).unwrap();
    for param_count in [4usize, 5] {
        let mut params = Vec::new();
        for n in 0..batch {
            let base = [0.12 + n as f32 * 0.03, -0.09 + n as f32 * 0.02, 0.73 + n as f32 * 0.04];
            params.extend_from_slice(&base);
            if param_count == 5 { params.push(0.51 + n as f32 * 0.02); }
            params.push(0.32 + n as f32 * 0.03);
        }
        let params_gpu = backend.upload_f32(&[batch, param_count], &params).unwrap();
        let actual_gpu = backend.hyperbolic_foveation_fixed_point_param_grad_f32(
            &params_gpu, &grid_seed_gpu, &[batch, 1, height, width], param_count,
            alpha, p_exp, iterations, eta, config,
        ).unwrap().expect("native Oxide fixed-point parameter adjoint");
        let actual = backend.download_f32(&actual_gpu).unwrap();
        let loss = |values: &[f32]| {
            let values_gpu = backend.upload_f32(&[batch, param_count], values).unwrap();
            let grid = backend.hyperbolic_fixed_point_grid_f32(
                &values_gpu, batch, height, width, param_count, alpha, p_exp,
                iterations, eta, config,
            ).unwrap();
            backend.download_f32(&grid).unwrap().iter().zip(&grid_seed)
                .map(|(value, seed)| value * seed).sum::<f32>()
        };
        let eps = 1.0e-3;
        for n in 0..batch {
            for packed_lane in 0..param_count {
                let semantic_lane = if param_count == 4 && packed_lane == 3 { 4 } else { packed_lane };
                let index = n * param_count + packed_lane;
                let mut plus = params.clone();
                let mut minus = params.clone();
                plus[index] += eps;
                minus[index] -= eps;
                let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps);
                let analytic = actual[n * 5 + semantic_lane];
                assert!((analytic - numeric).abs() < 4e-3,
                    "param_count={param_count} batch={n} lane={packed_lane}: analytic {analytic}, numeric {numeric}");
            }
        }
    }
}

#[test]
fn hyperbolic_foveation_native_newton_grid_matches_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let (batch, height, width) = (2, 3, 4);
    let (alpha, p_exp, iterations, eta) = (1.3f32, 2.2f32, 5usize, 0.45f32);
    let params = [0.05, -0.04, 0.8, 0.3, -0.15, 0.1, 1.2, 0.7];
    let params_gpu = backend.upload_f32(&[batch, 4], &params).unwrap();
    let grid = backend
        .hyperbolic_newton_grid_f32(
            &params_gpu,
            batch,
            height,
            width,
            4,
            alpha,
            p_exp,
            iterations,
            eta,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let actual = backend.download_f32(&grid).unwrap();
    let mut expected = vec![0.0; batch * height * width * 2];
    for b in 0..batch {
        let cx = params[b * 4].clamp(-0.9, 0.9);
        let cy = params[b * 4 + 1].clamp(-0.9, 0.9);
        let radius = params[b * 4 + 2].clamp(0.1, 2.0).max(1e-4);
        let strength = params[b * 4 + 3].clamp(0.0, 1.0);
        for y in 0..height {
            let qy = 2.0 * y as f32 / (height - 1) as f32 - 1.0;
            for x in 0..width {
                let qx = 2.0 * x as f32 / (width - 1) as f32 - 1.0;
                let mut px = qx;
                let mut py = qy;
                for _ in 0..iterations {
                    let dx = px - cx;
                    let dy = py - cy;
                    let r = (dx * dx + dy * dy).sqrt();
                    let (u, du_dr) = if r > 1e-6 {
                        let ar = alpha.max(1e-4) * r;
                        let tanh = ar.tanh();
                        (tanh / r, (alpha.max(1e-4) * (1.0 - tanh * tanh) * r - tanh) / (r * r))
                    } else { (alpha.max(1e-4), 0.0) };
                    let (weight, dw_dr) = if r < radius {
                        let base = (1.0 - r / radius).max(0.0);
                        (base.powf(p_exp.max(1e-4)), if r > 1e-6 { -(p_exp.max(1e-4) / radius) * base.powf((p_exp.max(1e-4) - 1.0).max(0.0)) } else { 0.0 })
                    } else { (0.0, 0.0) };
                    let gain = 1.0 + strength * weight * (u - 1.0);
                    let dg = strength * (dw_dr * (u - 1.0) + weight * du_dr);
                    let mut jac = [[gain, 0.0], [0.0, gain]];
                    if r > 1e-6 {
                        let scale = dg / r;
                        jac[0][0] += scale * dx * dx;
                        jac[0][1] += scale * dx * dy;
                        jac[1][0] += scale * dy * dx;
                        jac[1][1] += scale * dy * dy;
                    }
                    let rx = qx - (cx + gain * dx);
                    let ry = qy - (cy + gain * dy);
                    let det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
                    if det.abs() < 1e-8 {
                        px = (px + eta * rx).clamp(-1.0, 1.0);
                        py = (py + eta * ry).clamp(-1.0, 1.0);
                    } else {
                        let sx = (jac[1][1] * rx - jac[0][1] * ry) / det;
                        let sy = (-jac[1][0] * rx + jac[0][0] * ry) / det;
                        px = (px + eta * sx).clamp(-1.0, 1.0);
                        py = (py + eta * sy).clamp(-1.0, 1.0);
                    }
                }
                let i = ((b * height + y) * width + x) * 2;
                expected[i] = (px + 1.0) * 0.5 * (width - 1) as f32;
                expected[i + 1] = (py + 1.0) * 0.5 * (height - 1) as f32;
            }
        }
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() <= 3e-6, "grid[{index}]: {actual} != {expected}");
    }
}

#[test]
fn hyperbolic_foveation_native_fixed_point_grid_matches_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let batch = 2;
    let height = 3;
    let width = 4;
    let alpha = 1.3f32;
    let p_exp = 2.2f32;
    let iterations = 7;
    let eta = 0.35f32;
    let params = [0.05, -0.04, 0.8, 0.7, 0.3, -0.15, 0.1, 1.2, 0.9, 0.7];
    let params_gpu = backend.upload_f32(&[batch, 5], &params).unwrap();
    let grid = backend
        .hyperbolic_fixed_point_grid_f32(
            &params_gpu,
            batch,
            height,
            width,
            5,
            alpha,
            p_exp,
            iterations,
            eta,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let actual = backend.download_f32(&grid).unwrap();
    let mut expected = vec![0.0; batch * height * width * 2];
    for b in 0..batch {
        let cx = params[b * 5].clamp(-0.9, 0.9);
        let cy = params[b * 5 + 1].clamp(-0.9, 0.9);
        let rx = params[b * 5 + 2].clamp(0.1, 2.0);
        let ry = params[b * 5 + 3].clamp(0.1, 2.0);
        let radius = (rx.powf(8.0) + ry.powf(8.0)).powf(0.125).clamp(0.1, 2.0);
        let strength = params[b * 5 + 4].clamp(0.0, 1.0);
        for y in 0..height {
            let qy = 2.0 * y as f32 / (height - 1) as f32 - 1.0;
            for x in 0..width {
                let qx = 2.0 * x as f32 / (width - 1) as f32 - 1.0;
                let mut px = qx;
                let mut py = qy;
                for _ in 0..iterations {
                    let dx = px - cx;
                    let dy = py - cy;
                    let r = (dx * dx + dy * dy).sqrt();
                    let u = if r > 1.0e-6 { (alpha.max(1.0e-4) * r).tanh() / r } else { alpha.max(1.0e-4) };
                    let weight = if r < radius { (1.0 - r / radius).max(0.0).powf(p_exp.max(1.0e-4)) } else { 0.0 };
                    let gain = 1.0 + strength * weight * (u - 1.0);
                    px = (px + eta * (qx - (cx + gain * dx))).clamp(-1.0, 1.0);
                    py = (py + eta * (qy - (cy + gain * dy))).clamp(-1.0, 1.0);
                }
                let index = ((b * height + y) * width + x) * 2;
                expected[index] = (px + 1.0) * 0.5 * (width - 1) as f32;
                expected[index + 1] = (py + 1.0) * 0.5 * (height - 1) as f32;
            }
        }
    }
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() <= 2e-6, "grid[{index}]: {actual} != {expected}");
    }
}

#[test]
fn hyperbolic_foveation_native_warp_rejects_invalid_shapes() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let input = backend.upload_f32(&[1, 1, 2, 2], &[0.0; 4]).unwrap();
    let short_grid = backend.upload_f32(&[1], &[0.0]).unwrap();
    let error = match backend.hyperbolic_foveation_warp_f32(
        &input,
        &short_grid,
        &[1, 1, 2, 2],
        ElementwiseLaunchConfig::default(),
    ) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("short HF sampling grid must be rejected"),
    };
    assert!(error.contains("hyperbolic_foveation_warp_f32"));
}
