//! Native f32 marker-gather checks. Independent source-derived reference;
//! no compiled-C++ golden. Run with tools/test_oxide.sh and one test thread.
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn position(value: f32, t: usize) -> usize {
    let rounded = f64::from(value).round();
    if rounded < 0.0 || rounded.is_nan() {
        0
    } else if rounded >= t as f64 {
        t - 1
    } else {
        rounded as usize
    }
}

#[test]
fn marker_gather_native_forward_backward_and_raw_input_fd() {
    let backend = OxideBackend::new().unwrap();
    let (batch, hidden, t, options, channels) = (2, 3, 7, 5, 12);
    let a_shape = [batch, hidden, 1, t];
    let b_shape = [batch, channels, 1, t];
    let out_shape = [batch, hidden, 1, options];
    let a: Vec<f32> = (0..batch * hidden * t)
        .map(|i| i as f32 * 0.125 - 2.0)
        .collect();
    // Only even channels at column zero are read. Masks and other columns
    // are poison, deliberately proving that they do not affect gathering.
    let mut b = vec![f32::NAN; batch * channels * t];
    let positions = [
        [-0.5, 0.5, 1.5, 1.5, 100.0],
        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 6.49, 4.5],
    ];
    for n in 0..batch {
        for j in 0..options {
            b[(n * channels + 2 * j) * t] = positions[n][j];
        }
    }
    let dy: Vec<f32> = (0..batch * hidden * options)
        .map(|i| (i % 11) as f32 * 0.125 - 0.5)
        .collect();
    let mut want = vec![0.0; dy.len()];
    let mut want_dx = vec![0.0; a.len()];
    for n in 0..batch {
        for j in 0..options {
            for c in 0..hidden {
                let src = (n * hidden + c) * t + position(positions[n][j], t);
                let dst = (n * hidden + c) * options + j;
                want[dst] = a[src];
                want_dx[src] += dy[dst];
            }
        }
    }
    let a_gpu = backend.upload_f32(&a_shape, &a).unwrap();
    let b_gpu = backend.upload_f32(&b_shape, &b).unwrap();
    let dy_gpu = backend.upload_f32(&out_shape, &dy).unwrap();
    for tile_len in [32, 128, 256] {
        let config = ElementwiseLaunchConfig { tile_len };
        let output = backend
            .marker_gather_f32(&a_gpu, &b_gpu, &a_shape, &b_shape, options, config)
            .unwrap();
        assert_eq!(backend.download_f32(&output).unwrap(), want);
        let dx = backend
            .marker_gather_backward_f32(&b_gpu, &dy_gpu, &a_shape, &b_shape, options, config)
            .unwrap();
        assert_eq!(backend.download_f32(&dx).unwrap(), want_dx);
    }
    // Actual GPU forward central differences, keeping integer metadata fixed.
    let config = ElementwiseLaunchConfig::default();
    for i in [0, 1, 2, 6, 14, 21, 22, 41] {
        let score = |offset: f32| {
            let mut perturbed = a.clone();
            perturbed[i] += offset;
            let input = backend.upload_f32(&a_shape, &perturbed).unwrap();
            let output = backend
                .marker_gather_f32(&input, &b_gpu, &a_shape, &b_shape, options, config)
                .unwrap();
            backend
                .download_f32(&output)
                .unwrap()
                .iter()
                .zip(&dy)
                .map(|(&x, &g)| f64::from(x) * f64::from(g))
                .sum::<f64>()
        };
        let fd = (score(0.001) - score(-0.001)) / 0.002;
        assert!(
            (fd - f64::from(want_dx[i])).abs() < 0.0003,
            "input {i}: {fd} vs {}",
            want_dx[i]
        );
    }
}

#[test]
fn marker_gather_native_duplicate_reduction_has_fixed_option_order() {
    let backend = OxideBackend::new().unwrap();
    let a_shape = [1, 1, 1, 3];
    let b_shape = [1, 8, 1, 3];
    let markers = backend.upload_f32(&b_shape, &vec![0.0; 24]).unwrap();
    let dy = backend
        .upload_f32(&[1, 1, 1, 4], &[1e20, 1.0, -1e20, 1.0])
        .unwrap();
    for _ in 0..3 {
        let dx = backend
            .marker_gather_backward_f32(
                &markers,
                &dy,
                &a_shape,
                &b_shape,
                4,
                ElementwiseLaunchConfig::default(),
            )
            .unwrap();
        assert_eq!(backend.download_f32(&dx).unwrap(), [1.0, 0.0, 0.0]);
    }
}

#[test]
fn marker_gather_native_rounds_adjacent_half_values_before_clamping() {
    let backend = OxideBackend::new().unwrap();
    let positions = [
        f32::from_bits(0x3eff_ffff),
        0.5,
        f32::from_bits(0x3f00_0001),
        f32::from_bits(0x3fbf_ffff),
        1.5,
        f32::from_bits(0x3fc0_0001),
        -0.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        3.5,
        f32::MAX,
    ];
    let a_shape = [1, 2, 1, 4];
    let b_shape = [1, 24, 1, 4];
    let features = backend
        .upload_f32(&a_shape, &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
        .unwrap();
    let mut markers = vec![f32::NAN; 96];
    for (j, &pos) in positions.iter().enumerate() {
        markers[2 * j * 4] = pos;
    }
    let markers = backend.upload_f32(&b_shape, &markers).unwrap();
    let output = backend
        .marker_gather_f32(
            &features,
            &markers,
            &a_shape,
            &b_shape,
            12,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    let expected: Vec<f32> = (0..2)
        .flat_map(|c| {
            positions
                .iter()
                .map(move |&p| 10.0 * c as f32 + position(p, 4) as f32)
        })
        .collect();
    assert_eq!(backend.download_f32(&output).unwrap(), expected);
    // In particular the f32 just below 0.5 must stay at column zero.
    assert_eq!(expected[0], 0.0);
    assert_eq!(expected[1], 1.0);
}

#[test]
fn marker_gather_native_invalid_layouts_and_storage_error_before_launch() {
    let backend = OxideBackend::new().unwrap();
    let config = ElementwiseLaunchConfig::default();
    let a_shape = [2, 3, 1, 7];
    let b_shape = [2, 12, 1, 7];
    let a = backend.upload_f32(&a_shape, &vec![0.0; 42]).unwrap();
    let b = backend.upload_f32(&b_shape, &vec![0.0; 168]).unwrap();
    for (a_dims, b_dims, k) in [
        (vec![2, 3, 1, 7], vec![2, 12, 1, 7], 0),
        (vec![2, 3, 2, 7], vec![2, 12, 1, 7], 5),
        (vec![2, 3, 1, 7], vec![1, 12, 1, 7], 5),
        (vec![2, 3, 1, 7], vec![2, 12, 1, 6], 5),
        (vec![2, 3, 1, 7], vec![2, 9, 1, 7], 5),
        (vec![2, 0, 1, 7], vec![2, 12, 1, 7], 5),
        (vec![usize::MAX, 3, 1, 7], vec![usize::MAX, 12, 1, 7], 5),
        (vec![2, 3, 7], vec![2, 12, 1, 7], 5),
    ] {
        assert!(
            backend
                .marker_gather_f32(&a, &b, &a_dims, &b_dims, k, config)
                .is_err()
        );
        assert!(
            backend
                .marker_gather_backward_f32(&b, &a, &a_dims, &b_dims, k, config)
                .is_err()
        );
    }
    let short = backend.upload_f32(&[1], &[0.0]).unwrap();
    assert!(
        backend
            .marker_gather_f32(&short, &b, &a_shape, &b_shape, 5, config)
            .is_err()
    );
    assert!(
        backend
            .marker_gather_f32(&a, &short, &a_shape, &b_shape, 5, config)
            .is_err()
    );
    assert!(
        backend
            .marker_gather_backward_f32(&b, &short, &a_shape, &b_shape, 5, config)
            .is_err()
    );
}
