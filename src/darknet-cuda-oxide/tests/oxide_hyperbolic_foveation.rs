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
fn hyperbolic_foveation_native_fixed_point_grid_matches_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let batch = 2;
    let height = 3;
    let width = 4;
    let alpha = 1.3f32;
    let p_exp = 2.2f32;
    let iterations = 7;
    let eta = 0.35f32;
    let params = [0.05, -0.04, 0.8, 0.3, -0.15, 0.1, 1.2, 0.7];
    let params_gpu = backend.upload_f32(&[batch, 4], &params).unwrap();
    let grid = backend
        .hyperbolic_fixed_point_grid_f32(
            &params_gpu,
            batch,
            height,
            width,
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
        let radius = params[b * 4 + 2].clamp(0.1, 2.0);
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
