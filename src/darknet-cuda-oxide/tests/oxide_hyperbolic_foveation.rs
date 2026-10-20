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
