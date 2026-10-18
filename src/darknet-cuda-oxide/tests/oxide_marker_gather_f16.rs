//! Native half-storage marker gather. Feature payloads are copied as bits;
//! marker rounding is applied after binary16 quantization, not before it.
use darknet_cuda::{ElementwiseLaunchConfig, F16TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn half_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let e = (bits >> 10) & 31;
    let f = bits & 1023;
    match e {
        0 => sign * f64::from(f) * 2.0_f64.powi(-24),
        31 if f == 0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + f64::from(f) / 1024.0) * 2.0_f64.powi(i32::from(e) - 15),
    }
}

fn position(bits: u16, t: usize) -> usize {
    let p = half_value(bits).round();
    if p.is_nan() || p < 0.0 {
        0
    } else if p >= t as f64 {
        t - 1
    } else {
        p as usize
    }
}

#[test]
fn marker_gather_f16_native_preserves_feature_bits_and_rounds_half_metadata() {
    let backend = OxideBackend::new().unwrap();
    let (batch, hidden, t, k, cb) = (2, 3, 9, 7, 16);
    let a_shape = [batch, hidden, 1, t];
    let b_shape = [batch, cb, 1, t];
    let pattern = [
        0x7e35, 0x7c01, 0x8000, 0x0001, 0x03ff, 0x0400, 0x7c00, 0xfc00, 0x3555,
    ];
    let features: Vec<u16> = (0..batch * hidden * t)
        .map(|i| pattern[(i + i / t) % pattern.len()])
        .collect();
    let positions = [
        [0x37ff, 0x3800, 0x3dff, 0x3e00, 0xb800, 0x7c00, 0x7e00],
        [0xfc00, 0x0001, 0x7bff, 0x3e01, 0x8000, 0x7e35, 0x4840],
    ];
    let mut markers = vec![0x7fff; batch * cb * t]; // masks/extra columns are poison
    for n in 0..batch {
        for j in 0..k {
            markers[(n * cb + 2 * j) * t] = positions[n][j];
        }
    }
    let mut want = Vec::new();
    for n in 0..batch {
        for c in 0..hidden {
            for j in 0..k {
                want.push(features[(n * hidden + c) * t + position(positions[n][j], t)]);
            }
        }
    }
    assert!(
        pattern.iter().all(|bits| want.contains(bits)),
        "every feature edge-case must actually be gathered"
    );
    let a = backend.upload_f16_bits(&a_shape, &features).unwrap();
    let b = backend.upload_f16_bits(&b_shape, &markers).unwrap();
    for tile_len in [32, 128, 256] {
        let output = F16TensorOpsBackend::marker_gather_f16(
            &backend,
            &a,
            &b,
            &a_shape,
            &b_shape,
            k,
            ElementwiseLaunchConfig { tile_len },
        )
        .unwrap();
        assert_eq!(backend.download_f16_bits(&output).unwrap(), want);
    }
}

#[test]
fn marker_gather_f16_native_uses_quantized_not_original_f32_positions() {
    let backend = OxideBackend::new().unwrap();
    let a_shape = [1, 1, 1, 3];
    let b_shape = [1, 4, 1, 3];
    let a = backend
        .upload_f16_bits(&a_shape, &[0x3c00, 0x4000, 0x4200])
        .unwrap();
    let mut markers = vec![0.0; 12];
    markers[0] = f32::from_bits(0x3eff_ffff); // below .5 -> half .5 -> column 1
    markers[6] = f32::from_bits(0x3fbf_ffff); // below 1.5 -> half 1.5 -> column 2
    let b = backend.upload_f16(&b_shape, &markers).unwrap();
    let quantized = backend.download_f16_bits(&b).unwrap();
    assert_eq!((quantized[0], quantized[6]), (0x3800, 0x3e00));
    let output = F16TensorOpsBackend::marker_gather_f16(
        &backend,
        &a,
        &b,
        &a_shape,
        &b_shape,
        2,
        ElementwiseLaunchConfig::default(),
    )
    .unwrap();
    assert_eq!(
        backend.download_f16_bits(&output).unwrap(),
        [0x4000, 0x4200]
    );
}

#[test]
fn marker_gather_f16_native_rejects_bad_layouts_and_short_storage() {
    let backend = OxideBackend::new().unwrap();
    let a_shape = [2, 3, 1, 9];
    let b_shape = [2, 16, 1, 9];
    let a = backend.upload_f16_bits(&a_shape, &vec![0; 54]).unwrap();
    let b = backend.upload_f16_bits(&b_shape, &vec![0; 288]).unwrap();
    let config = ElementwiseLaunchConfig::default();
    for (ash, bsh, k) in [
        (vec![2, 3, 1, 9], vec![2, 16, 1, 9], 0),
        (vec![2, 3, 2, 9], vec![2, 16, 1, 9], 7),
        (vec![2, 3, 1, 9], vec![1, 16, 1, 9], 7),
        (vec![2, 3, 1, 9], vec![2, 16, 1, 8], 7),
        (vec![2, 3, 1, 9], vec![2, 13, 1, 9], 7),
        (vec![2, 0, 1, 9], vec![2, 16, 1, 9], 7),
        (vec![usize::MAX, 3, 1, 9], vec![usize::MAX, 16, 1, 9], 7),
        (vec![2, 3, 9], vec![2, 16, 1, 9], 7),
    ] {
        assert!(
            F16TensorOpsBackend::marker_gather_f16(&backend, &a, &b, &ash, &bsh, k, config)
                .is_err()
        );
    }
    let short = backend.upload_f16_bits(&[1], &[0]).unwrap();
    assert!(
        F16TensorOpsBackend::marker_gather_f16(&backend, &short, &b, &a_shape, &b_shape, 7, config)
            .is_err()
    );
    assert!(
        F16TensorOpsBackend::marker_gather_f16(&backend, &a, &short, &a_shape, &b_shape, 7, config)
            .is_err()
    );
}

#[test]
fn marker_gather_f16_runtime_handles_nonprevious_sources_and_aliased_metadata() {
    use darknet_core::{ConfigFile, GpuExecutableNetwork, LayerParams, Network, Tensor};
    for sources in ["0,1", "-2,-2"] {
        let cfg = ConfigFile::parse_str(
            "half-gather.cfg",
            &format!(
                "[net]\nwidth=5\nheight=1\nchannels=16\nbatch=2\n\
             [convolutional]\nfilters=16\nsize=1\nactivation=linear\n\
             [convolutional]\nfilters=16\nsize=1\nactivation=linear\n\
             [laya_marker_gather]\nlayers={sources}\nhidden=16\nmax_options=4\n"
            ),
        )
        .unwrap();
        let net = Network::from_config(cfg).unwrap();
        let mut gpu = GpuExecutableNetwork::new(net, OxideBackend::new().unwrap());
        for params in &mut gpu.params {
            if let LayerParams::Convolutional(p) = params {
                p.weights.fill(0.0);
                p.biases.fill(0.0);
                for c in 0..16 {
                    p.weights[c * 16 + c] = 1.0;
                }
            }
        }
        let shape = gpu.network.net.input_shape();
        let mut data: Vec<f32> = (0..160).map(|i| (i % 13) as f32 * 0.125).collect();
        let positions = [
            [f32::from_bits(0x3eff_ffff), 1.5, 1.5, 10.0],
            [2.49, -0.25, 4.0, 3.5],
        ];
        for n in 0..2 {
            for j in 0..4 {
                data[(n * 16 + 2 * j) * 5] = positions[n][j];
            }
        }
        let input = Tensor::from_data(shape, data).unwrap();
        let half = gpu.backend.upload_f16(&[2, 16, 1, 5], &input.data).unwrap();
        let quantized = gpu.backend.download_f16(&half).unwrap();
        let expected =
            darknet_core::laya_marker_gather::forward(&quantized, &quantized, 2, 16, 4, 5, 16)
                .unwrap();
        let output = gpu
            .forward_compact_f16(&input)
            .unwrap()
            .final_output
            .unwrap();
        assert_eq!(output.shape, gpu.network.layers[2].output_shape);
        assert_eq!(output.data, expected, "sources {sources}");
    }
}
