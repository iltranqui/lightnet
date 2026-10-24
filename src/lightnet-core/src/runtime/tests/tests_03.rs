use super::*;

#[test]
fn fake_backend_activation_backward_matches_reference() {
    let backend = FakeBackend;
    let activation_input = FakeTensor {
        shape: vec![5],
        data: vec![-2.0, -0.5, 0.0, 0.5, 2.0],
    };
    let delta = FakeTensor {
        shape: vec![5],
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    let config = ElementwiseLaunchConfig::default();

    for activation in [
        F32ActivationKind::Linear,
        F32ActivationKind::Logistic,
        F32ActivationKind::Relu,
        F32ActivationKind::Leaky,
        F32ActivationKind::Mish,
        F32ActivationKind::Swish,
        F32ActivationKind::Tanh,
        F32ActivationKind::Elu,
        F32ActivationKind::Selu,
        F32ActivationKind::HardMish,
    ] {
        let actual = backend
            .activation_backward_f32(&activation_input, &delta, activation, config)
            .expect("backward activation");
        let expected = activation_input
            .data
            .iter()
            .zip(&delta.data)
            .map(|(value, grad)| grad * fake_activation_gradient(activation, *value))
            .collect::<Vec<_>>();
        assert_eq!(actual.shape, activation_input.shape);
        assert_eq!(actual.data.len(), expected.len());
        for (actual_value, expected_value) in actual.data.iter().zip(expected) {
            assert!((actual_value - expected_value).abs() < 1e-6);
        }
    }
}

#[test]
fn fake_backend_upsample_backward_accumulates_source_gradients() {
    let backend = FakeBackend;
    let output_grad = FakeTensor {
        shape: vec![1, 1, 4, 4],
        data: (1..=16).map(|value| value as f32).collect(),
    };

    let input_grad = backend
        .upsample_backward_f32(
            &output_grad,
            &[1, 1, 2, 2],
            1,
            4,
            4,
            2,
            1.0,
            ElementwiseLaunchConfig::default(),
        )
        .expect("upsample backward");

    assert_eq!(input_grad.shape, vec![1, 1, 2, 2]);
    assert_eq!(input_grad.data, vec![14.0, 22.0, 46.0, 54.0]);
}

#[test]
fn fake_backend_maxpool_backward_accumulates_overlapping_windows() {
    let backend = FakeBackend;
    let input = FakeTensor {
        shape: vec![1, 1, 3, 3],
        data: vec![1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
    };
    let output_grad = FakeTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };

    let input_grad = backend
        .maxpool_backward_f32(
            &input,
            &output_grad,
            1,
            3,
            3,
            2,
            1,
            1,
            0,
            ElementwiseLaunchConfig::default(),
        )
        .expect("maxpool backward");

    assert_eq!(input_grad.shape, input.shape);
    assert_eq!(
        input_grad.data,
        vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn logistic_axis_layout_keeps_batch_channel_hw_order() {
    // raptor:src-lib/logistic_layer.cpp copies the contiguous
    // batch*inputs array then activates each entry in place. The C++ input
    // shape is NCHW; distinct coordinates and non-square H/W catch any
    // batch/channel/spatial permutation.
    let network = Network::from_config(
        ConfigFile::parse_str(
            "logistic_axis.cfg",
            "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[logistic]\n",
        )
        .expect("logistic cfg"),
    )
    .expect("logistic network");
    let executable = ExecutableNetwork::new(network);
    let (batch, channels, height, width) = (2, 2, 2, 3);
    let input_values: Vec<f32> = (0..batch)
        .flat_map(|n| {
            (0..channels).flat_map(move |c| {
                (0..height).flat_map(move |y| {
                    (0..width).map(move |x| {
                        -1.1 + 0.6 * n as f32 + 0.3 * c as f32 + 0.12 * y as f32 + 0.05 * x as f32
                    })
                })
            })
        })
        .collect();
    let expected: Vec<f32> = input_values
        .iter()
        .map(|&value| 1.0 / (1.0 + (-value).exp()))
        .collect();
    let input = Tensor::from_data(
        TensorShape::new(width, height, channels, batch),
        input_values,
    )
    .expect("logistic axis input");
    let forward = executable.forward(&input).expect("logistic forward");
    assert_slice_close(
        "logistic NCHW batch/channel/HW order",
        &expected,
        &forward.layer_outputs[0].data,
        1e-7,
    );
}

#[test]
fn connected_axis_layout_keeps_batch_rows_and_output_weight_axes() {
    // C++ `connected_layer.cpp:186` presents each sample as a flattened
    // NCHW input row and weights as [output][input]. Distinct sample,
    // channel, and spatial sentinels make transposes visible.
    let network = Network::from_config(
            ConfigFile::parse_str(
                "connected_axis.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[connected]\noutput=3\nactivation=linear\n",
            )
            .expect("connected cfg"),
        )
        .expect("connected network");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Connected(params) = &mut executable.params[0] else {
        panic!("connected parameters expected");
    };
    params.weights[0] = 1.0;
    params.weights[11] = 0.5;
    params.weights[12 + 5] = 2.0;
    params.weights[12 + 6] = 1.0;
    params.weights[24 + 3] = 1.0;
    params.weights[24 + 8] = 1.0;
    params.weights[24 + 10] = -1.0;
    params.biases.copy_from_slice(&[0.25, -0.5, 1.0]);
    let input = Tensor::from_data(
        TensorShape::new(3, 2, 2, 2),
        vec![
            1., 2., 3., 4., 5., 6., 11., 12., 13., 14., 15., 16., 101., 102., 103., 104., 105.,
            106., 111., 112., 113., 114., 115., 116.,
        ],
    )
    .expect("connected axis input");
    let forward = executable.forward(&input).expect("connected forward");
    assert_slice_close(
        "connected batch/output rows",
        &[9.25, 22.5, 3.0, 159.25, 322.5, 103.0],
        &forward.layer_outputs[0].data,
        1e-6,
    );
    let seed = Tensor::from_data(
        TensorShape::new(1, 1, 3, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .expect("connected seed");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("connected backward");
    assert_slice_close(
        "connected [out][in] adjoint per batch row",
        &[
            1., 0., 0., 3., 0., 4., 2., 0., 3., 0., -3., 0.5, 4., 0., 0., 6., 0., 10., 5., 0., 6.,
            0., -6., 2.,
        ],
        &backward.input_gradient.data,
        1e-6,
    );
}

#[test]
fn local_avgpool_axis_layout_pins_window_and_batch_order() {
    // C++ `maxpool_layer.cpp:440` averages valid taps in each local
    // window. Non-square B=2/C=2 data and unequal axis strides expose any
    // x/y, channel, or sample permutation.
    let network = Network::from_config(
            ConfigFile::parse_str(
                "local_avgpool_axis.cfg",
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n[local_avgpool]\nsize=3\nstride_x=2\nstride_y=1\npadding=1\n",
            )
            .expect("local avgpool cfg"),
        )
        .expect("local avgpool network");
    let output_shape = TensorShape::new(2, 2, 2, 2);
    assert_eq!(network.layers[0].output_shape, output_shape);
    let executable = ExecutableNetwork::new(network);
    let values = (0..60)
        .map(|index| {
            let plane = index / 15;
            let position = index % 15;
            (plane * 100 + position + 1) as f32
        })
        .collect();
    let input =
        Tensor::from_data(TensorShape::new(5, 3, 2, 2), values).expect("local avgpool input");
    let forward = executable.forward(&input).expect("local avgpool forward");
    assert_slice_close(
        "local avgpool NCHW output",
        &[
            7.0, 9.0, 9.5, 11.5, 107.0, 109.0, 109.5, 111.5, 207.0, 209.0, 209.5, 211.5, 307.0,
            309.0, 309.5, 311.5,
        ],
        &forward.layer_outputs[0].data,
        1e-6,
    );
    let seeds = Tensor::from_data(
        output_shape,
        vec![
            1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 6.0, 4.0, 0.0, 0.0, 8.0,
        ],
    )
    .expect("local avgpool seeds");
    let backward = executable
        .backward(&input, &forward, &seeds)
        .expect("local avgpool backward");
    let one_plane = [
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        0.0,
        0.0,
        1.0 / 9.0,
        1.0 / 9.0,
        4.0 / 9.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 9.0,
        1.0 / 9.0,
        4.0 / 9.0,
        1.0 / 3.0,
        1.0 / 3.0,
    ];
    let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|scale| one_plane.map(|value| scale * value))
        .collect();
    assert_slice_close(
        "local avgpool per-window input gradient",
        &expected,
        &backward.input_gradient.data,
        1e-6,
    );
}

#[test]
fn avgpool_axis_layout_reduces_spatial_per_channel_and_batch() {
    // C++ `avgpool_layer.cpp:51` reduces each NCHW channel plane over H*W,
    // keeping batch and channel as distinct output axes.
    let network = Network::from_config(
        ConfigFile::parse_str(
            "avgpool_axis.cfg",
            "[net]\nwidth=3\nheight=2\nchannels=2\nbatch=2\n[avgpool]\n",
        )
        .expect("avgpool cfg"),
    )
    .expect("avgpool network");
    assert_eq!(network.layers[0].output_shape, TensorShape::new(1, 1, 2, 2));
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        TensorShape::new(3, 2, 2, 2),
        vec![
            1., 2., 3., 4., 5., 6., 10., 20., 30., 40., 50., 60., 101., 102., 103., 104., 105.,
            106., 110., 120., 130., 140., 150., 160.,
        ],
    )
    .expect("avgpool input");
    let forward = executable.forward(&input).expect("avgpool forward");
    assert_slice_close(
        "avgpool NCHW output",
        &[3.5, 35.0, 103.5, 135.0],
        &forward.layer_outputs[0].data,
        1e-6,
    );
    let seed = Tensor::from_data(TensorShape::new(1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
        .expect("avgpool seed");
    let backward = executable
        .backward(&input, &forward, &seed)
        .expect("avgpool backward");
    let expected = [
        vec![1.0 / 6.0; 6],
        vec![2.0 / 6.0; 6],
        vec![3.0 / 6.0; 6],
        vec![4.0 / 6.0; 6],
    ]
    .concat();
    assert_slice_close(
        "avgpool per-sample channel gradient",
        &expected,
        &backward.input_gradient.data,
        1e-6,
    );
}

#[test]
fn maxpool_axis_layout_pins_depth_channel_batch_and_spatial_order() {
    // Hand-derived from `Jev_laya:src-lib/maxpool_layer.cpp:327`: output
    // group g scans input channels g, g+out_c, ... independently at each
    // (batch,y,x). A tie at b0/g0/y0/x0 must keep channel 2, the first
    // maximum in that strided group.
    let depth_net = Network::from_config(
            ConfigFile::parse_str(
                "maxpool_depth_axis.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=5\nbatch=2\n[maxpool]\nmaxpool_depth=1\nout_channels=2\n",
            )
            .expect("depth cfg"),
        )
        .expect("depth network");
    assert_eq!(
        depth_net.layers[0].output_shape,
        TensorShape::new(3, 2, 2, 2)
    );
    let depth_shape = TensorShape::new(3, 2, 5, 2);
    let depth_input = Tensor::from_data(
        depth_shape,
        vec![
            0., 1., 10., 11., 20., 21., 100., 101., 110., 111., 120., 121., 400., 201., 210., 211.,
            220., 221., 300., 301., 310., 311., 320., 321., 400., 401., 410., 411., 420., 421.,
            1000., 1001., 1010., 1011., 1020., 1021., 1100., 1101., 1110., 1111., 1120., 1121.,
            1200., 1201., 1210., 1211., 1220., 1221., 1300., 1301., 1310., 1311., 1320., 1321.,
            1400., 1401., 1410., 1411., 1420., 1421.,
        ],
    )
    .expect("depth input");
    let depth_output_shape = TensorShape::new(3, 2, 2, 2);
    let depth_output =
        super::forward_maxpool_depth(&depth_input, depth_output_shape).expect("depth forward");
    assert_eq!(
        depth_output.data,
        vec![
            400., 401., 410., 411., 420., 421., 300., 301., 310., 311., 320., 321., 1400., 1401.,
            1410., 1411., 1420., 1421., 1300., 1301., 1310., 1311., 1320., 1321.,
        ]
    );
    let depth_seed = Tensor::from_data(
        depth_output_shape,
        (1..=24).map(|value| value as f32).collect(),
    )
    .expect("depth seed");
    let depth_dx =
        super::backward_maxpool_depth(&depth_input, &depth_seed).expect("depth backward");
    assert_eq!(
        depth_dx.data,
        vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 7., 8., 9.,
            10., 11., 12., 0., 2., 3., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 19., 20., 21., 22., 23., 24., 13., 14., 15., 16., 17., 18.,
        ]
    );

    // The ordinary maxpool branch scans ky then kx and stores NCHW planes.
    // Non-square 5x3, B=2, C=2 sentinels use odd size=3, unequal strides,
    // and padding=1. The top-left/bottom-right tie checks first-tap order;
    // overlapping windows test summed deltas and zero_nonmax.
    let spatial_net = Network::from_config(
            ConfigFile::parse_str(
                "maxpool_spatial_axis.cfg",
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n[maxpool]\nsize=3\nstride_x=2\nstride_y=1\npadding=1\nmaxpool_zero_nonmax=1\n",
            )
            .expect("spatial cfg"),
        )
        .expect("spatial network");
    assert_eq!(
        spatial_net.layers[0].output_shape,
        TensorShape::new(2, 2, 2, 2)
    );
    let shape = TensorShape::new(5, 3, 2, 2);
    let mut spatial_values: Vec<f32> = (0..60)
        .map(|index| {
            let batch = index / 30;
            let channel = index / 15 % 2;
            let position = index % 15;
            (batch * 200 + channel * 100 + position + 1) as f32
        })
        .collect();
    spatial_values[0] = 13.0; // ties the first tap with position 12
    let input = Tensor::from_data(shape, spatial_values).expect("spatial input");
    let output_shape = TensorShape::new(2, 2, 2, 2);
    let output = super::forward_maxpool(3, 2, 1, 1, &input, output_shape, false)
        .expect("spatial maxpool forward");
    assert_eq!(
        output.data,
        vec![
            13., 15., 13., 15., 113., 115., 113., 115., 213., 215., 213., 215., 313., 315., 313.,
            315.,
        ]
    );
    let seed = Tensor::from_data(output_shape, (1..=16).map(|value| value as f32).collect())
        .expect("spatial seed");
    let dx = super::backward_maxpool(3, 2, 1, 1, &input, &seed, false, true)
        .expect("zero_nonmax backward");
    assert_eq!(
        dx.data,
        vec![
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3., 0., 6., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 12., 0., 14., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 20.,
            0., 22., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 28., 0., 30.,
        ]
    );
}

#[test]
fn maxpool_zero_nonmax_zeroes_non_argmax_inputs() {
    let input = Tensor::from_data(
        TensorShape::new(3, 3, 1, 1),
        vec![1.0, 2.0, 1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
    )
    .expect("input");
    let mut grad = Tensor::zeros(TensorShape::new(2, 2, 1, 1));
    for (i, value) in grad.data.iter_mut().enumerate() {
        *value = (i + 1) as f32;
    }
    let exact = super::backward_maxpool(2, 1, 1, 0, &input, &grad, false, false).unwrap();
    let zeroed = super::backward_maxpool(2, 1, 1, 0, &input, &grad, false, true).unwrap();
    // Every window's argmax is the centre (1,1) -> flat index 4.
    assert_eq!(zeroed.data[4], exact.data[4]);
    for index in [0usize, 1, 2, 3, 5, 6, 7, 8] {
        assert_eq!(zeroed.data[index], 0.0, "index {index}");
    }
}

#[test]
fn maxpool_zero_nonmax_backward_matches_finite_differences() {
    let shape = TensorShape::new(3, 3, 1, 1);
    let output_shape = TensorShape::new(2, 2, 1, 1);
    let input =
        Tensor::from_data(shape, vec![0.1, 0.9, 0.2, 0.8, 0.7, 0.6, 0.3, 0.5, 0.4]).expect("input");
    let output_gradient =
        Tensor::from_data(output_shape, vec![0.2, -0.3, 0.7, 1.1]).expect("output gradient");
    let loss = |values: &[f32]| {
        let input = Tensor::from_data(shape, values.to_vec()).expect("perturbed input");
        let output = super::forward_maxpool(2, 1, 1, 0, &input, output_shape, false)
            .expect("maxpool forward");
        output
            .data
            .iter()
            .zip(&output_gradient.data)
            .map(|(value, grad)| value * grad)
            .sum::<f32>()
    };
    let gradients = [false, true].map(|zero_nonmax| {
        super::backward_maxpool(2, 1, 1, 0, &input, &output_gradient, false, zero_nonmax)
            .expect("maxpool backward")
    });
    assert_eq!(gradients[0], gradients[1]);
    let h = 1e-3;
    for (index, analytic) in gradients[1].data.iter().enumerate() {
        let mut plus = input.data.clone();
        let mut minus = input.data.clone();
        plus[index] += h;
        minus[index] -= h;
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * h);
        assert!(
            (analytic - numeric).abs() < 1e-3,
            "input[{index}]: analytic {analytic}, numeric {numeric}"
        );
    }
}

#[test]
fn maxpool_zero_nonmax_keeps_gpu_backward_eligible() {
    let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=1\nbatch=1\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[maxpool]\nsize=2\nstride=1\npadding=0\nmaxpool_zero_nonmax=1\n",
            )
            .expect("cfg"),
        )
        .expect("network");
    let gpu = GpuExecutableNetwork::new(network, FakeBackend);
    assert!(
        !gpu.requires_cpu_backward_fallback(),
        "argmax-only maxpool backward already zeroes non-argmax deltas"
    );
}

#[test]
fn fake_backend_route_backward_scatter_matches_forward_layout() {
    let backend = FakeBackend;
    let output_grad = FakeTensor {
        shape: vec![1, 3, 1, 2],
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };

    let input_grads = backend
        .route_backward_f32(
            &output_grad,
            &[vec![1, 4, 1, 2], vec![1, 2, 1, 2]],
            &[1, 0],
            &[2, 1],
            &[4, 2],
            2,
            ElementwiseLaunchConfig::default(),
        )
        .expect("route backward");

    assert_eq!(input_grads.len(), 2);
    assert_eq!(
        input_grads[0],
        FakeTensor {
            shape: vec![1, 4, 1, 2],
            data: vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
        }
    );
    assert_eq!(
        input_grads[1],
        FakeTensor {
            shape: vec![1, 2, 1, 2],
            data: vec![5.0, 6.0, 0.0, 0.0],
        }
    );
}

#[test]
fn fake_backend_bias_backward_reduces_channel_gradients() {
    let backend = FakeBackend;
    let output_grad = FakeTensor {
        shape: vec![1, 2, 1, 3],
        data: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    };
    let bias_grad = backend
        .bias_backward_f32(&output_grad, 2, 3, ElementwiseLaunchConfig::default())
        .expect("bias backward");
    assert_eq!(bias_grad.shape, vec![2]);
    assert_eq!(bias_grad.data, vec![6.0, 60.0]);
}

#[test]
fn fake_backend_conv_backward_matches_reference_for_pointwise_conv() {
    let backend = FakeBackend;
    let input = FakeTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![1.0, 2.0, 3.0, 4.0],
    };
    let weights = FakeTensor {
        shape: vec![1, 1, 1, 1],
        data: vec![2.0],
    };
    let output_grad = FakeTensor {
        shape: vec![1, 1, 2, 2],
        data: vec![10.0, 20.0, 30.0, 40.0],
    };

    let bias_grad = backend
        .bias_backward_f32(&output_grad, 1, 4, ElementwiseLaunchConfig::default())
        .expect("bias backward");
    let input_grad = backend
        .conv2d_backward_input_f32(
            &output_grad,
            &weights,
            &[1, 1, 2, 2],
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .expect("conv backward input");
    let weight_grad = backend
        .conv2d_backward_weights_f32(
            &input,
            &output_grad,
            &[1, 1, 1, 1],
            1,
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            ElementwiseLaunchConfig::default(),
        )
        .expect("conv backward weights");

    assert_eq!(bias_grad.data, vec![100.0]);
    assert_eq!(input_grad.shape, vec![1, 1, 2, 2]);
    assert_eq!(input_grad.data, vec![20.0, 40.0, 60.0, 80.0]);
    assert_eq!(weight_grad.shape, vec![1, 1, 1, 1]);
    assert_eq!(weight_grad.data, vec![300.0]);
}

#[test]
fn runs_single_conv_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-conv.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("tensor");
    let result = executable.forward(&input).expect("forward should succeed");

    assert_eq!(
        result.final_output().expect("final output").data,
        vec![2.5, 4.5]
    );

    let _ = fs::remove_file(path);
}

#[test]
fn runs_route_and_upsample_layers() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1

                [upsample]
                stride=2
                scale=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-route.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");
    let result = executable.forward(&input).expect("forward should succeed");
    let final_output = result.final_output().expect("final output");

    assert_eq!(final_output.shape.width, 4);
    assert_eq!(final_output.shape.height, 4);
    assert_eq!(final_output.get(0, 0, 0, 0), 1.0);
    assert_eq!(final_output.get(0, 0, 3, 3), 4.0);

    let _ = fs::remove_file(path);
}

#[test]
fn collects_yolo_head_outputs() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=6

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::filled(executable.network.net.input_shape(), 1.0);
    let result = executable.forward(&input).expect("forward should succeed");

    assert_eq!(result.yolo_heads.len(), 1);
    assert_eq!(result.yolo_heads[0].output.data, input.data);
}

#[test]
fn backward_runs_single_conv_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-conv.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    let output_gradient =
        Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![10.0, 20.0]).expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");

    assert_eq!(backward.input_gradient.data, vec![20.0, 40.0]);
    match &backward.param_gradients[0] {
        LayerParams::Convolutional(conv) => {
            assert_eq!(conv.biases, vec![30.0]);
            assert_eq!(conv.weights, vec![50.0]);
        }
        other => panic!("unexpected param gradients: {other:?}"),
    }

    let _ = fs::remove_file(path);
}

#[test]
fn convolutional_analytical_gradients_and_sgd_step() {
    // This test covers the complete CPU training contract for a padded
    // multi-channel convolution.  The analytical reference validates the
    // forward pass and dL/dx, dL/dw, and dL/db independently of the runtime
    // implementation.  The second half then uses Darknet's negative
    // upstream-gradient convention and verifies one real SGD update.
    let cfg = ConfigFile::parse_str(
        "conv-gradient.cfg",
        r#"
                [net]
                width=3
                height=2
                channels=2
                batch=2

                [convolutional]
                filters=2
                size=3
                stride=1
                pad=1
                activation=linear
            "#,
    )
    .expect("convolution config should parse");
    let network = Network::from_config(cfg).expect("convolution network should build");
    let mut executable = ExecutableNetwork::new(network);

    let weights = (0..36)
        .map(|index| 0.03 * (index as f32 - 17.0))
        .collect::<Vec<_>>();
    let biases = vec![0.17, -0.23];
    if let LayerParams::Convolutional(params) = &mut executable.params[0] {
        params.weights.clone_from_slice(&weights);
        params.biases.clone_from_slice(&biases);
    } else {
        panic!("expected a convolutional parameter block");
    }

    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        (0..24).map(|index| (index as f32 - 11.0) / 7.0).collect(),
    )
    .expect("convolution input should have the expected shape");
    let analytical_case = AnalyticalConvCase {
        batch: 2,
        channels: 2,
        height: 2,
        width: 3,
        filters: 2,
        kernel: 3,
        stride: 1,
        padding: 1,
        groups: 1,
        input: &input.data,
        weights: &weights,
        biases: &biases,
    };
    let analytical_output = analytical_convolution_output(&analytical_case);
    let forward = executable
        .forward(&input)
        .expect("convolution forward should succeed");
    let output = forward.final_output().expect("convolution output");
    assert_slice_close(
        "CPU convolution forward",
        &analytical_output,
        &output.data,
        2.0e-6,
    );
    let target = analytical_output
        .iter()
        .enumerate()
        .map(|(index, _)| 0.07 * index as f32 - 0.4)
        .collect::<Vec<_>>();
    let upstream = analytical_output
        .iter()
        .zip(&target)
        .map(|(prediction, target)| prediction - target)
        .collect::<Vec<_>>();
    let upstream_tensor = Tensor::from_data(output.shape, upstream.clone())
        .expect("upstream gradient should have the output shape");
    let backward = executable
        .backward(&input, &forward, &upstream_tensor)
        .expect("convolution backward should succeed");
    let (weight_gradient, bias_gradient) = match &backward.param_gradients[0] {
        LayerParams::Convolutional(gradient) => (gradient.weights.clone(), gradient.biases.clone()),
        other => panic!("unexpected convolution gradient block: {other:?}"),
    };
    // For L = 1/2 * sum((y - target)^2), dL/dy = y - target.  These
    // expected gradients come from the closed-form convolution equations
    // above, with no parameter or input perturbation.
    let (expected_input_gradient, expected_weight_gradient, expected_bias_gradient) =
        analytical_convolution_gradients(&analytical_case, &upstream);
    assert_slice_close(
        "convolution weight analytical gradient",
        &expected_weight_gradient,
        &weight_gradient,
        2.0e-6,
    );
    assert_slice_close(
        "convolution bias analytical gradient",
        &expected_bias_gradient,
        &bias_gradient,
        2.0e-6,
    );
    assert_slice_close(
        "convolution input analytical gradient",
        &expected_input_gradient,
        &backward.input_gradient.data,
        2.0e-6,
    );

    let before_loss = squared_error_loss(&executable, &input, &target);
    let before_weights = match &executable.params[0] {
        LayerParams::Convolutional(params) => params.weights.clone(),
        other => panic!("unexpected convolution parameters: {other:?}"),
    };
    let before_biases = match &executable.params[0] {
        LayerParams::Convolutional(params) => params.biases.clone(),
        other => panic!("unexpected convolution parameters: {other:?}"),
    };
    let descent_seed =
        Tensor::from_data(output.shape, upstream.iter().map(|value| -*value).collect())
            .expect("descent seed should have the output shape");
    executable.zero_train_buffers();
    let training_forward = executable
        .forward_training(&input)
        .expect("training forward should succeed");
    executable
        .backward_accumulate(&input, &training_forward, &descent_seed)
        .expect("training backward should accumulate");

    // Darknet accumulates -dL/dtheta in its update buffers and adds that
    // descent direction during update_weights().
    if let LayerTrainBuffers::Convolutional(train) = &executable.train_buffers[0] {
        let expected_weight_update = expected_weight_gradient
            .iter()
            .map(|gradient| -*gradient)
            .collect::<Vec<_>>();
        let expected_bias_update = expected_bias_gradient
            .iter()
            .map(|gradient| -*gradient)
            .collect::<Vec<_>>();
        assert_slice_close(
            "Darknet convolution weight descent buffer",
            &expected_weight_update,
            &train.weight_updates,
            2.0e-6,
        );
        assert_slice_close(
            "Darknet convolution bias descent buffer",
            &expected_bias_update,
            &train.bias_updates,
            2.0e-6,
        );
    } else {
        panic!("expected convolutional training buffers");
    }

    let learning_rate = 1.0e-3;
    let batch = executable.network.net.batch;
    executable.update_weights(learning_rate, 0.0, 0.0, batch);
    let expected_weights = before_weights
        .iter()
        .zip(&expected_weight_gradient)
        .map(|(weight, gradient)| weight - learning_rate * gradient / batch as f32)
        .collect::<Vec<_>>();
    let expected_biases = before_biases
        .iter()
        .zip(&expected_bias_gradient)
        .map(|(bias, gradient)| bias - learning_rate * gradient / batch as f32)
        .collect::<Vec<_>>();
    match &executable.params[0] {
        LayerParams::Convolutional(params) => {
            assert_slice_close(
                "CPU convolution SGD weights",
                &expected_weights,
                &params.weights,
                1.0e-6,
            );
            assert_slice_close(
                "CPU convolution SGD biases",
                &expected_biases,
                &params.biases,
                1.0e-6,
            );
        }
        other => panic!("unexpected convolution parameters after update: {other:?}"),
    }
    let after_loss = squared_error_loss(&executable, &input, &target);
    assert!(
        after_loss < before_loss,
        "one Darknet SGD step must lower the loss: before={before_loss}, after={after_loss}"
    );
}

#[test]
fn fake_gpu_convolution_backward_and_sgd_match_cpu_training() {
    // This uses the same F32TensorOpsBackend seam as the real cuTile
    // bridge.  The fake backend executes the scalar reference equations,
    // which lets CI verify device dispatch, gradient accumulation, and
    // the device SGD API without requiring a CUDA driver.
    let cfg = ConfigFile::parse_str(
        "gpu-conv-training.cfg",
        r#"
                [net]
                width=3
                height=2
                channels=2
                batch=2

                [convolutional]
                filters=2
                size=3
                stride=1
                pad=1
                activation=linear
            "#,
    )
    .expect("convolution config should parse");
    let network = Network::from_config(cfg).expect("convolution network should build");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let weights = (0..36)
        .map(|index| 0.02 * (index as f32 - 13.0))
        .collect::<Vec<_>>();
    let biases = vec![0.11, -0.19];
    for params in [&mut cpu.params[0], &mut gpu.params[0]] {
        let LayerParams::Convolutional(params) = params else {
            panic!("expected a convolutional parameter block");
        };
        params.weights.clone_from_slice(&weights);
        params.biases.clone_from_slice(&biases);
    }

    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        (0..24).map(|index| (index as f32 - 8.0) / 9.0).collect(),
    )
    .expect("convolution input should have the expected shape");
    let cpu_forward = cpu
        .forward_training(&input)
        .expect("CPU training forward should succeed");
    let cpu_output = cpu_forward.final_output().expect("CPU output");
    let target = cpu_output
        .data
        .iter()
        .enumerate()
        .map(|(index, _)| 0.03 * index as f32 - 0.2)
        .collect::<Vec<_>>();
    let descent_seed = Tensor::from_data(
        cpu_output.shape,
        cpu_output
            .data
            .iter()
            .zip(&target)
            .map(|(prediction, target)| target - prediction)
            .collect(),
    )
    .expect("GPU seed should have the output shape");
    cpu.zero_train_buffers();
    cpu.backward_accumulate(&input, &cpu_forward, &descent_seed)
        .expect("CPU training backward should succeed");
    cpu.update_weights(1.0e-3, 0.0, 0.0, cpu.network.net.batch);

    gpu.zero_train_buffers_gpu()
        .expect("GPU training buffers should initialize");
    let gpu_forward = gpu
        .forward_with_state(&input)
        .expect("GPU training forward should succeed");
    assert_slice_close(
        "fake GPU convolution forward",
        &cpu_forward.final_output().expect("CPU output").data,
        &gpu_forward
            .cpu_forward
            .final_output()
            .expect("GPU output")
            .data,
        1.0e-6,
    );
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[Some(descent_seed)])
        .expect("GPU convolution backward should succeed");
    gpu.update_weights_gpu(1.0e-3, 0.0, 0.0, gpu.network.net.batch)
        .expect("GPU convolution SGD update should succeed");

    let cpu_params = match &cpu.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected CPU parameters: {other:?}"),
    };
    let gpu_params = match &gpu.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected GPU parameters: {other:?}"),
    };
    assert_slice_close(
        "fake GPU convolution SGD weights",
        &cpu_params.weights,
        &gpu_params.weights,
        1.0e-6,
    );
    assert_slice_close(
        "fake GPU convolution SGD biases",
        &cpu_params.biases,
        &gpu_params.biases,
        1.0e-6,
    );
}

#[test]
fn fake_gpu_channel_normalizing_convolution_matches_cpu_training() {
    let cfg = ConfigFile::parse_str(
        "channel-normalization-conv.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=2
                batch=2

                [convolutional]
                filters=3
                size=1
                activation=normalize_channels_softmax_maxval
            "#,
    )
    .expect("channel-normalization config parses");
    let network = Network::from_config(cfg).expect("channel-normalization net builds");
    let mut cpu = ExecutableNetwork::new(network.clone());
    let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
    let weights = vec![0.3, -0.2, -0.1, 0.4, 0.15, -0.25];
    let biases = vec![-0.3, 0.1, 0.25];
    for params in [&mut cpu.params[0], &mut gpu.params[0]] {
        let LayerParams::Convolutional(params) = params else {
            panic!("expected convolutional parameters");
        };
        params.weights.clone_from_slice(&weights);
        params.biases.clone_from_slice(&biases);
    }

    let input = Tensor::from_data(
        cpu.network.net.input_shape(),
        vec![0.2, -0.7, 1.1, 0.4, -0.3, 0.8, 0.5, 1.3],
    )
    .expect("input tensor");
    let cpu_forward = cpu.forward_training(&input).expect("CPU forward");
    let gpu_forward = gpu.forward_with_state(&input).expect("FakeBackend forward");
    let cpu_output = cpu_forward.final_output().expect("CPU output");
    let gpu_output = gpu_forward.cpu_forward.final_output().expect("GPU output");
    assert_slice_close(
        "channel-normalization forward",
        &cpu_output.data,
        &gpu_output.data,
        1e-6,
    );
    let seed = Tensor::from_data(
        cpu_output.shape,
        (0..cpu_output.len())
            .map(|i| (i as f32 * 0.37).sin())
            .collect(),
    )
    .expect("output seed");

    cpu.zero_train_buffers();
    cpu.backward_accumulate(&input, &cpu_forward, &seed)
        .expect("CPU backward");
    gpu.zero_train_buffers_gpu().expect("GPU buffers");
    gpu.backward_accumulate_seeded_gpu(&gpu_forward, &[Some(seed)])
        .expect("FakeBackend backward");
    cpu.update_weights(1.0e-3, 0.0, 0.0, cpu.network.net.batch);
    gpu.update_weights_gpu(1.0e-3, 0.0, 0.0, gpu.network.net.batch)
        .expect("FakeBackend SGD");
    let (LayerParams::Convolutional(cpu_params), LayerParams::Convolutional(gpu_params)) =
        (&cpu.params[0], &gpu.params[0])
    else {
        panic!("expected convolutional parameters after update");
    };
    assert_slice_close(
        "channel-normalization weight update",
        &cpu_params.weights,
        &gpu_params.weights,
        1e-6,
    );
    assert_slice_close(
        "channel-normalization bias update",
        &cpu_params.biases,
        &gpu_params.biases,
        1e-6,
    );
}

#[test]
fn channel_normalization_activation_is_limited_to_convolution_family() {
    for activation in [
        "normalize_channels",
        "normalize_channels_softmax",
        "normalize_channels_softmax_maxval",
    ] {
        let cfg = ConfigFile::parse_str(
                "channel-normalization-activation.cfg",
                &format!(
                    "[net]\nwidth=2\nheight=1\nchannels=2\n[convolutional]\nfilters=3\nsize=1\nactivation={activation}\n"
                ),
            )
            .expect("config parses");
        assert!(
            Network::from_config(cfg).is_ok(),
            "{activation} convolution"
        );
        let cfg = ConfigFile::parse_str(
                "channel-normalization-activation.cfg",
                &format!(
                    "[net]\nwidth=2\nheight=2\nchannels=2\n[deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation={activation}\n"
                ),
            )
            .expect("config parses");
        assert!(
            Network::from_config(cfg).is_ok(),
            "{activation} deconvolution"
        );
    }
    let unsupported = ConfigFile::parse_str(
            "channel-normalization-activation.cfg",
            "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\nactivation=normalize_channels\n",
        )
        .expect("config parses");
    let err = Network::from_config(unsupported).expect_err("connected must reject NORM_CHAN");
    assert!(
        err.to_string()
            .contains("only supported on convolution-family layers")
    );
}

#[test]
fn backward_runs_avgpool_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [avgpool]
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    assert_eq!(
        forward.final_output().expect("final output").data,
        vec![2.5]
    );

    let output_gradient =
        Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![8.0]).expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");
    assert_eq!(backward.input_gradient.data, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn backward_runs_local_avgpool_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [local_avgpool]
                size=2
                stride=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = ExecutableNetwork::new(network);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    assert_eq!(
        forward.final_output().expect("final output").data,
        vec![2.5, 3.0, 3.5, 4.0]
    );

    let output_gradient = Tensor::from_data(
        executable.network.layers[0].output_shape,
        vec![4.0, 6.0, 8.0, 10.0],
    )
    .expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");
    assert_eq!(backward.input_gradient.data, vec![1.0, 4.0, 5.0, 18.0]);
}

#[test]
fn backward_route_scatter_matches_cpp_style_delta_flow() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [route]
                layers=-1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-route.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![3.0, 4.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    let output_gradient =
        Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![7.0, 11.0]).expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");

    assert_eq!(backward.layer_output_gradients[1].data, vec![7.0, 11.0]);
    assert_eq!(backward.layer_output_gradients[0].data, vec![7.0, 11.0]);
    assert_eq!(backward.input_gradient.data, vec![7.0, 11.0]);

    let _ = fs::remove_file(path);
}

#[test]
fn shared_convolutional_layers_count_one_parameter_block() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [convolutional]
                filters=1
                size=1
                stride=1
                share_index=0
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    assert_eq!(network.total_parameter_count(), 2);
}

#[test]
fn backward_accumulate_shared_conv_updates_into_owner_buffers() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear

                [convolutional]
                filters=1
                size=1
                stride=1
                share_index=0
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-shared.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![3.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    let output_gradient =
        Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![1.0]).expect("tensor");
    let backward = executable
        .backward_accumulate(&input, &forward, &output_gradient)
        .expect("backward should succeed");

    assert_eq!(backward.input_gradient.data, vec![4.0]);
    match &executable.train_buffers[0] {
        LayerTrainBuffers::Convolutional(conv) => {
            assert_eq!(conv.bias_updates, vec![3.0]);
            assert_eq!(conv.weight_updates, vec![12.5]);
        }
        other => panic!("unexpected owner train buffers: {other:?}"),
    }
    assert_eq!(executable.train_buffers[1], LayerTrainBuffers::None);

    let _ = fs::remove_file(path);
}

#[test]
fn sgd_update_applies_decay_step_and_momentum() {
    let mut weights = vec![2.0];
    let mut updates = vec![3.0];
    sgd_update(&mut weights, &mut updates, 0.1, 0.9, 0.01, 2);
    assert!((weights[0] - 2.148).abs() < 1e-6);
    assert!((updates[0] - 2.664).abs() < 1e-6);
}

#[test]
fn compute_learning_rate_handles_burn_in_steps_and_poly() {
    let base = NetConfig {
        width: 1,
        height: 1,
        channels: 1,
        batch: 1,
        time_steps: 1,
        label_smooth_eps: 0.0,
        adversarial: false,
        obb: false,
        subdivisions: 1,
        learning_rate: 0.01,
        learning_rate_min: 0.00001,
        momentum: 0.9,
        decay: 0.0005,
        max_batches: 100,
        burn_in: 10,
        power: 4.0,
        step: 1,
        scale: 1.0,
        gamma: 1.0,
        sgdr_cycle: 100,
        sgdr_mult: 2,
        policy: LrPolicy::Constant,
        steps: Vec::new(),
        scales: Vec::new(),
        equidistant_point: 0,
        badlabels_rejection_percentage: 0.0,
        num_sigmas_reject_badlabels: 0.0,
        adam: false,
        adam_b1: 0.9,
        adam_b2: 0.999,
        adam_eps: 1e-6,
        decoupled_decay: false,
        muon: false,
        muon_momentum: 0.95,
        loss_scale: 1.0,
        try_fix_nan: false,
    };
    assert_eq!(compute_learning_rate(&base, 0), 0.0);
    assert!((compute_learning_rate(&base, 5) - 0.000625).abs() < 1e-9);
    assert!((compute_learning_rate(&base, 10) - 0.01).abs() < 1e-9);

    let steps = NetConfig {
        policy: LrPolicy::Steps,
        burn_in: 0,
        steps: vec![20, 40],
        scales: vec![0.1, 0.1],
        ..base.clone()
    };
    assert!((compute_learning_rate(&steps, 10) - 0.01).abs() < 1e-9);
    assert!((compute_learning_rate(&steps, 20) - 0.001).abs() < 1e-9);
    assert!((compute_learning_rate(&steps, 40) - 0.0001).abs() < 1e-9);

    let poly = NetConfig {
        policy: LrPolicy::Poly,
        burn_in: 0,
        ..base
    };
    assert!((compute_learning_rate(&poly, 50) - 0.000625).abs() < 1e-9);
    assert_eq!(compute_learning_rate(&poly, 100), 0.0);
}

#[test]
fn backward_conv_supports_swish_with_cached_activation_input() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=swish
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-swish.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.0f32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    assert!(forward.layer_activation_inputs[0].is_some());

    let output_gradient =
        Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 1.0]).expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");

    let expected_bias_grad = {
        let s1 = 1.0 / (1.0 + (-1.0f32).exp());
        let s2 = 1.0 / (1.0 + (-2.0f32).exp());
        (s1 + 1.0 * s1 * (1.0 - s1)) + (s2 + 2.0 * s2 * (1.0 - s2))
    };
    let expected_weight_grad = {
        let s1 = 1.0 / (1.0 + (-1.0f32).exp());
        let s2 = 1.0 / (1.0 + (-2.0f32).exp());
        1.0 * (s1 + 1.0 * s1 * (1.0 - s1)) + 2.0 * (s2 + 2.0 * s2 * (1.0 - s2))
    };

    match &backward.param_gradients[0] {
        LayerParams::Convolutional(conv) => {
            assert!((conv.biases[0] - expected_bias_grad).abs() < 1e-6);
            assert!((conv.weights[0] - expected_weight_grad).abs() < 1e-6);
        }
        other => panic!("unexpected param gradients: {other:?}"),
    }

    let _ = fs::remove_file(path);
}

#[test]
fn backward_runs_batchnorm_conv_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-bn.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    bytes.extend_from_slice(&1.0f32.to_le_bytes());
    bytes.extend_from_slice(&3.0f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![3.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    let output_gradient =
        Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![4.0]).expect("tensor");
    let backward = executable
        .backward(&input, &forward, &output_gradient)
        .expect("backward should succeed");

    let inv_std = 1.0 / (3.0f32 + super::BATCH_NORM_EPSILON).sqrt();
    let normalized = (6.0 - 1.0) * inv_std;
    let grad_preconv = 4.0 * 2.0 * inv_std;

    assert!((backward.input_gradient.data[0] - 2.0 * grad_preconv).abs() < 1e-5);
    match &backward.param_gradients[0] {
        LayerParams::Convolutional(conv) => {
            assert!((conv.biases[0] - 4.0).abs() < 1e-6);
            assert!((conv.weights[0] - 3.0 * grad_preconv).abs() < 1e-5);
            let bn = conv.batch_norm.as_ref().expect("batchnorm grads");
            assert!((bn.scales[0] - 4.0 * normalized).abs() < 1e-5);
        }
        other => panic!("unexpected param gradients: {other:?}"),
    }

    let _ = fs::remove_file(path);
}

#[test]
fn backward_accumulate_updates_batchnorm_running_stats() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);
    match &mut executable.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 1.0;
            let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
            batch_norm.rolling_mean[0] = 10.0;
            batch_norm.rolling_variance[0] = 20.0;
        }
        other => panic!("unexpected params: {other:?}"),
    }

    let input = Tensor::from_data(executable.network.net.input_shape(), vec![2.0, 4.0])
        .expect("input tensor");
    let forward = executable.forward(&input).expect("forward should succeed");
    let seeds = vec![Some(Tensor::filled(
        executable.network.layers[0].output_shape,
        0.0,
    ))];
    executable
        .backward_accumulate_seeded(&input, &forward, &seeds)
        .expect("backward should update stats");

    let params = match &executable.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
    assert!((batch_norm.rolling_mean[0] - 9.3).abs() < 1.0e-6);
    assert!((batch_norm.rolling_variance[0] - 18.2).abs() < 1.0e-6);
}

#[test]
fn backward_seeded_accumulates_multiple_yolo_heads() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=6
                size=1
                stride=1
                activation=linear

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1

                [route]
                layers=-2

                [yolo]
                mask=0
                anchors=10,13
                classes=1
                num=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = ExecutableNetwork::new(network);

    let path = std::env::temp_dir().join("darknet-runtime-backward-multihead.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0.0f32.to_le_bytes(), 6).flatten());
    bytes.extend(std::iter::repeat_n(1.0f32.to_le_bytes(), 6).flatten());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input = Tensor::from_data(executable.network.net.input_shape(), vec![1.0]).expect("tensor");
    let forward = executable.forward(&input).expect("forward should succeed");

    let mut seeds = vec![None; executable.network.layers.len()];
    seeds[1] = Some(Tensor::filled(forward.layer_outputs[1].shape, 1.0));
    seeds[3] = Some(Tensor::filled(forward.layer_outputs[3].shape, 1.0));
    let backward = executable
        .backward_seeded(&input, &forward, &seeds)
        .expect("seeded backward should succeed");

    assert_eq!(backward.layer_output_gradients[1].data, vec![1.0; 6]);
    assert_eq!(backward.layer_output_gradients[3].data, vec![1.0; 6]);
    assert_eq!(backward.layer_output_gradients[2].data, vec![1.0; 6]);
    assert_eq!(backward.layer_output_gradients[0].data, vec![2.0; 6]);
    assert_eq!(backward.input_gradient.data, vec![12.0]);

    let _ = fs::remove_file(path);
}

#[test]
fn standalone_batchnorm_runs_inference_training_backward_update_and_weights() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1
                batch=2

                [batchnorm]
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    assert_eq!(
        network.layers[0]
            .weights()
            .expect("weights")
            .parameter_count(),
        4
    );
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::BatchNorm(params) = &mut executable.params[0] else {
        panic!("expected standalone batchnorm parameters");
    };
    params.biases[0] = 0.5;
    params.scales[0] = 2.0;
    params.rolling_mean[0] = 10.0;
    params.rolling_variance[0] = 4.0;

    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 3.0, 5.0, 7.0],
    )
    .expect("input tensor");
    let inference = executable
        .forward(&input)
        .expect("inference should succeed");
    let inference_inv_std = 1.0 / (4.0f32 + super::BATCH_NORM_EPSILON).sqrt();
    for (actual, expected) in inference
        .final_output()
        .expect("output")
        .data
        .iter()
        .zip([1.0f32, 3.0, 5.0, 7.0])
    {
        let expected = (expected - 10.0) * inference_inv_std * 2.0 + 0.5;
        assert!((actual - expected).abs() < 1e-5);
    }

    let training = executable
        .forward_training(&input)
        .expect("training forward should succeed");
    let output = training.final_output().expect("output");
    let inv_std = 1.0 / (5.0f32 + super::BATCH_NORM_EPSILON).sqrt();
    let expected = [
        (1.0 - 4.0) * inv_std * 2.0 + 0.5,
        (3.0 - 4.0) * inv_std * 2.0 + 0.5,
        (5.0 - 4.0) * inv_std * 2.0 + 0.5,
        (7.0 - 4.0) * inv_std * 2.0 + 0.5,
    ];
    for (actual, expected) in output.data.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }

    let seed = Tensor::from_data(output.shape, vec![1.0, 2.0, 3.0, 4.0]).expect("seed");
    let backward = executable
        .backward(&input, &training, &seed)
        .expect("backward should succeed");
    let xhat = [-3.0 * inv_std, -inv_std, inv_std, 3.0 * inv_std];
    let sum_dy: f32 = (1..=4).map(|value| value as f32).sum();
    let sum_dy_xhat: f32 = [1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .zip(xhat)
        .map(|(dy, xhat)| dy * xhat)
        .sum();
    for ((actual, dy), xhat) in backward
        .input_gradient
        .data
        .iter()
        .zip([1.0, 2.0, 3.0, 4.0])
        .zip(xhat)
    {
        let expected = 2.0 * inv_std / 4.0 * (4.0 * dy - sum_dy - xhat * sum_dy_xhat);
        assert!((actual - expected).abs() < 1e-5);
    }
    match &backward.param_gradients[0] {
        LayerParams::BatchNorm(gradient) => {
            assert!((gradient.biases[0] - 10.0).abs() < 1e-6);
            assert!((gradient.scales[0] - sum_dy_xhat).abs() < 1e-6);
            assert_eq!(gradient.rolling_mean, vec![0.0]);
            assert_eq!(gradient.rolling_variance, vec![0.0]);
        }
        other => panic!("unexpected param gradients: {other:?}"),
    }

    executable.zero_train_buffers();
    executable
        .backward_accumulate(&input, &training, &seed)
        .expect("accumulating backward should succeed");
    match &executable.train_buffers[0] {
        LayerTrainBuffers::BatchNorm(train) => {
            assert_eq!(train.bias_updates, vec![10.0]);
            assert!((train.scale_updates[0] - sum_dy_xhat).abs() < 1e-6);
        }
        other => panic!("unexpected train buffers: {other:?}"),
    }
    executable.update_weights(0.1, 0.0, 0.0, 1);
    let params = match &executable.params[0] {
        LayerParams::BatchNorm(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    assert!((params.biases[0] - 1.5).abs() < 1e-6);
    assert!((params.scales[0] - (2.0 + 0.1 * sum_dy_xhat)).abs() < 1e-5);
    assert!((params.rolling_mean[0] - 9.4).abs() < 1e-6);
    assert!((params.rolling_variance[0] - 4.1).abs() < 1e-6);
}

#[test]
fn gpu_standalone_batchnorm_uses_rolling_statistics_for_inference() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1
                batch=2

                [batchnorm]
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
    let LayerParams::BatchNorm(params) = &mut executable.params[0] else {
        panic!("expected standalone batchnorm parameters");
    };
    params.biases[0] = 0.5;
    params.scales[0] = 2.0;
    params.rolling_mean[0] = 10.0;
    params.rolling_variance[0] = 4.0;

    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 3.0, 5.0, 7.0],
    )
    .expect("input tensor");
    let result = executable
        .forward(&input)
        .expect("GPU inference should succeed");
    let inverse_std = 1.0 / (4.0f32 + super::BATCH_NORM_EPSILON).sqrt();
    let expected = [1.0f32, 3.0, 5.0, 7.0].map(|value| (value - 10.0) * inverse_std * 2.0 + 0.5);
    let actual = &result.final_output().expect("output").data;
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }

    let compact = executable
        .forward_compact(&input)
        .expect("compact GPU inference should succeed");
    let compact = &compact.final_output.expect("compact output").data;
    for (actual, expected) in compact.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn gpu_activation_adapter_maps_expected_variants() {
    assert_eq!(
        Activation::Mish.to_cuda_f32_kind(),
        Some(F32ActivationKind::Mish)
    );
    assert_eq!(
        Activation::Loggy.to_cuda_f32_kind(),
        Some(F32ActivationKind::Loggy)
    );
    assert_eq!(
        Activation::Hardtan.to_cuda_f32_kind(),
        Some(F32ActivationKind::Hardtan)
    );
    assert_eq!(
        Activation::Relie.to_cuda_f32_kind(),
        Some(F32ActivationKind::Relie)
    );
    assert_eq!(
        Activation::Lhtan.to_cuda_f32_kind(),
        Some(F32ActivationKind::Lhtan)
    );
    assert_eq!(
        Activation::Plse.to_cuda_f32_kind(),
        Some(F32ActivationKind::Plse)
    );
    assert_eq!(
        Activation::Stair.to_cuda_f32_kind(),
        Some(F32ActivationKind::Stair)
    );
    assert_eq!(
        Activation::Elu.to_cuda_f32_kind(),
        Some(F32ActivationKind::Elu)
    );
    assert_eq!(
        Activation::Selu.to_cuda_f32_kind(),
        Some(F32ActivationKind::Selu)
    );
    assert_eq!(
        Activation::HardMish.to_cuda_f32_kind(),
        Some(F32ActivationKind::HardMish)
    );
    assert_eq!(
        Activation::Other("hard_mish".to_owned()).to_cuda_f32_kind(),
        None
    );
}

#[test]
fn gpu_runtime_runs_route_shortcut_and_upsample_layers() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [upsample]
                stride=2
                scale=1

                [route]
                layers=-1

                [shortcut]
                from=-2
                activation=relu
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 2.0]).expect("tensor");
    let result = executable
        .forward(&input)
        .expect("gpu forward should succeed");
    let final_output = result.final_output().expect("final output");

    assert_eq!(final_output.shape, TensorShape::new(4, 2, 1, 1));
    assert_eq!(
        final_output.data,
        vec![2.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0]
    );
}

#[test]
fn gpu_runtime_runs_single_conv_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);

    let path = std::env::temp_dir().join("darknet-gpu-runtime-conv.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("tensor");
    let result = executable
        .forward(&input)
        .expect("gpu forward should succeed");

    assert_eq!(
        result.final_output().expect("final output").data,
        vec![2.5, 4.5]
    );

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_runtime_runs_maxpool_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=4
                height=4
                channels=1

                [maxpool]
                size=2
                stride=2
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(
        TensorShape::new(4, 4, 1, 1),
        vec![
            1.0, 5.0, 2.0, 4.0, 3.0, 7.0, 6.0, 8.0, 0.0, 9.0, 2.0, 1.0, 4.0, 3.0, 5.0, 6.0,
        ],
    )
    .expect("tensor");
    let result = executable
        .forward(&input)
        .expect("gpu forward should succeed");

    assert_eq!(
        result.final_output().expect("final output").data,
        vec![7.0, 8.0, 9.0, 6.0]
    );
}

#[test]
fn gpu_runtime_runs_avgpool_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [avgpool]
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");
    let result = executable
        .forward(&input)
        .expect("gpu forward should succeed");

    assert_eq!(result.final_output().expect("final output").data, vec![2.5]);
}

#[test]
fn gpu_runtime_runs_local_avgpool_layer() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=2
                channels=1

                [local_avgpool]
                size=2
                stride=1
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let executable = GpuExecutableNetwork::new(network, FakeBackend);
    let input = Tensor::from_data(
        executable.network.net.input_shape(),
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("tensor");
    let result = executable
        .forward(&input)
        .expect("gpu forward should succeed");

    assert_eq!(
        result.final_output().expect("final output").data,
        vec![2.5, 3.0, 3.5, 4.0]
    );
}

#[test]
fn gpu_runtime_reuses_uploaded_conv_params() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=2
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let backend = CountingBackend::default();
    let mut executable = GpuExecutableNetwork::new(network, backend.clone());

    let path = std::env::temp_dir().join("darknet-gpu-runtime-cache.weights");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    bytes.extend_from_slice(&5i32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0.5f32.to_le_bytes());
    bytes.extend_from_slice(&2.0f32.to_le_bytes());
    fs::write(&path, bytes).expect("weights file should write");

    executable
        .load_weights_file(&path)
        .expect("weights should load");
    assert_eq!(backend.upload_count(), 2);

    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("tensor");
    executable
        .forward(&input)
        .expect("first gpu forward should succeed");
    executable
        .forward(&input)
        .expect("second gpu forward should succeed");

    assert_eq!(backend.upload_count(), 4);

    let _ = fs::remove_file(path);
}

#[test]
fn gpu_weight_updates_preserve_momentum_between_iterations() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
    executable
        .init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![2.0]).expect("input tensor");
    let seeds = vec![Some(
        Tensor::from_data(executable.network.layers[0].output_shape, vec![3.0])
            .expect("seed tensor"),
    )];

    for _ in 0..2 {
        let forward = executable
            .forward_with_state(&input)
            .expect("GPU forward should succeed");
        executable
            .backward_accumulate_seeded_gpu(&forward, &seeds)
            .expect("GPU backward should accumulate");
        executable
            .update_weights_gpu(0.1, 0.9, 0.0, 1)
            .expect("GPU update should succeed");
    }

    let params = match &executable.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    assert!((params.weights[0] - 1.74).abs() < 1.0e-6);
    assert!((params.biases[0] - 0.87).abs() < 1.0e-6);

    let train_buffers = executable.device_train_buffers.borrow();
    let train = train_buffers[0].as_ref().expect("train buffers exist");
    assert!((train.weight_updates.data[0] - 10.26).abs() < 1.0e-6);
    assert!((train.bias_updates.data[0] - 5.13).abs() < 1.0e-6);
}

#[test]
fn gpu_batchnorm_conv_accumulates_and_updates_scale() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=1
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut executable = GpuExecutableNetwork::new(network, FakeBackend);
    match &mut executable.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights[0] = 2.0;
            params.biases[0] = 0.5;
            let batch_norm = params.batch_norm.as_mut().expect("batch norm params");
            batch_norm.scales[0] = 1.5;
            batch_norm.rolling_mean[0] = 1.0;
            batch_norm.rolling_variance[0] = 4.0 - super::BATCH_NORM_EPSILON;
        }
        other => panic!("unexpected params: {other:?}"),
    }
    executable
        .init_train_buffers_gpu()
        .expect("GPU train buffers should initialize");

    let input =
        Tensor::from_data(executable.network.net.input_shape(), vec![2.0]).expect("input tensor");
    let seeds = vec![Some(
        Tensor::from_data(executable.network.layers[0].output_shape, vec![3.0])
            .expect("seed tensor"),
    )];
    let forward = executable
        .forward_with_state(&input)
        .expect("GPU forward should succeed");
    executable
        .backward_accumulate_seeded_gpu(&forward, &seeds)
        .expect("GPU backward should accumulate");

    {
        let train_buffers = executable.device_train_buffers.borrow();
        let train = train_buffers[0].as_ref().expect("train buffers exist");
        let scale_updates = train.scale_updates.as_ref().expect("scale updates exist");
        assert!((scale_updates.data[0] - 4.5).abs() < 1.0e-6);
    }

    executable
        .update_weights_gpu(0.1, 0.0, 0.0, 1)
        .expect("GPU update should succeed");
    let params = match &executable.params[0] {
        LayerParams::Convolutional(params) => params,
        other => panic!("unexpected params: {other:?}"),
    };
    let batch_norm = params.batch_norm.as_ref().expect("batch norm params");
    assert!((batch_norm.scales[0] - 1.95).abs() < 1.0e-6);
}

#[test]
fn gpu_inference_folds_batchnorm_into_conv_like_cpu() {
    let cfg = ConfigFile::parse_str(
        "inline.cfg",
        r#"
                [net]
                width=3
                height=1
                channels=2

                [convolutional]
                filters=3
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
    )
    .expect("cfg should parse");
    let network = Network::from_config(cfg).expect("network should build");
    let mut gpu = GpuExecutableNetwork::new(network.clone(), FakeBackend);
    match &mut gpu.params[0] {
        LayerParams::Convolutional(params) => {
            params.weights = vec![1.0, -2.0, 0.5, 0.25, -1.0, 3.0];
            params.biases = vec![0.1, -0.2, 0.3];
            let bn = params.batch_norm.as_mut().expect("batch norm params");
            bn.scales = vec![1.5, 0.5, -2.0];
            bn.rolling_mean = vec![0.2, -0.4, 1.0];
            bn.rolling_variance = vec![4.0, 0.25, 9.0];
        }
        other => panic!("unexpected params: {other:?}"),
    }
    let mut cpu = ExecutableNetwork::new(network);
    cpu.params = gpu.params.clone();
    let input = Tensor::from_data(
        gpu.network.net.input_shape(),
        vec![1.0, 2.0, -1.0, 0.5, -3.0, 4.0],
    )
    .expect("input tensor");
    let want = cpu.forward(&input).expect("CPU forward");
    let got = gpu.forward(&input).expect("GPU forward");
    let (w, g) = (&want.layer_outputs[0], &got.layer_outputs[0]);
    assert_eq!(w.data.len(), g.data.len());
    for (a, b) in w.data.iter().zip(&g.data) {
        assert!((a - b).abs() < 1.0e-5, "cpu {a} vs gpu {b}");
    }
}
