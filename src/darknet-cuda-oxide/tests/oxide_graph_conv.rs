use darknet_cuda::{
    ElementwiseLaunchConfig, F32TensorOpsBackend,
    host_ops::graph_conv::{
        GraphConvAggregationGeometry, aggregate_backward, aggregate_forward,
        learned_aggregate_backward, learned_aggregate_forward,
    },
};
use darknet_cuda_oxide::OxideBackend;

fn patterned(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.37 + phase).sin() * 0.8) + (i as f32 % 5.0) * 0.03)
        .collect()
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name} length");
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        let error = (got - want).abs();
        assert!(
            error <= tol * (1.0 + want.abs()),
            "{name}[{i}]: got {got}, want {want}, error {error}"
        );
    }
}

#[test]
fn graph_conv_uniform_aggregate_forward_backward_matches_reference_and_fd() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    for geometry in [
        GraphConvAggregationGeometry {
            batch: 2,
            channels: 3,
            input_height: 3,
            input_width: 5,
            output_height: 3,
            output_width: 5,
            size: 3,
            stride_x: 1,
            stride_y: 1,
            dilation: 1,
            padding: 1,
        },
        GraphConvAggregationGeometry {
            batch: 1,
            channels: 4,
            input_height: 4,
            input_width: 5,
            output_height: 4,
            output_width: 3,
            size: 3,
            stride_x: 2,
            stride_y: 1,
            dilation: 2,
            padding: 2,
        },
    ] {
        let input = patterned(geometry.input_len(), 0.2);
        let input_device = backend
            .upload_f32(
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
                &input,
            )
            .expect("upload graph input");
        let output_device = backend
            .graph_conv_aggregate_f32(&input_device, geometry, config)
            .expect("native graph aggregation");
        let output = backend
            .download_f32(&output_device)
            .expect("download graph output");
        let expected = aggregate_forward(&input, geometry).expect("host aggregation");
        close("graph aggregation forward", &output, &expected, 2e-6);

        let output_gradient = patterned(geometry.output_len(), 0.8);
        let gradient_device = backend
            .upload_f32(
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.output_height,
                    geometry.output_width,
                ],
                &output_gradient,
            )
            .expect("upload graph output gradient");
        let input_gradient_device = backend
            .graph_conv_aggregate_backward_f32(&gradient_device, geometry, config)
            .expect("native graph aggregation backward");
        let input_gradient = backend
            .download_f32(&input_gradient_device)
            .expect("download graph input gradient");
        let expected_gradient =
            aggregate_backward(&output_gradient, geometry).expect("host aggregation backward");
        close(
            "graph aggregation backward",
            &input_gradient,
            &expected_gradient,
            2e-6,
        );

        let epsilon = 1e-3;
        for index in (0..input.len()).step_by(3) {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let plus = aggregate_forward(&plus, geometry).expect("FD plus");
            let minus = aggregate_forward(&minus, geometry).expect("FD minus");
            let numeric = plus
                .iter()
                .zip(&minus)
                .zip(&output_gradient)
                .map(|((&plus, &minus), &seed)| (plus - minus) * seed)
                .sum::<f32>()
                / (2.0 * epsilon);
            let error = (numeric - input_gradient[index]).abs();
            assert!(
                error < 2e-3 * (1.0 + numeric.abs()),
                "graph aggregation FD[{index}]: native {}, numeric {numeric}, error {error}",
                input_gradient[index]
            );
        }
    }
}

#[test]
fn graph_conv_learned_host_contract_masks_edges_and_matches_finite_differences() {
    let geometry = GraphConvAggregationGeometry {
        batch: 1,
        channels: 2,
        input_height: 3,
        input_width: 4,
        output_height: 3,
        output_width: 2,
        size: 3,
        stride_x: 2,
        stride_y: 1,
        dilation: 1,
        padding: 1,
    };
    let groups = 2;
    let mut input = patterned(geometry.input_len(), 0.1);
    let mut edge_kernel = patterned(36, 0.4)
        .into_iter()
        .map(|value| value * 0.12)
        .collect::<Vec<_>>();
    let mut edge_biases = patterned(18, 0.7)
        .into_iter()
        .map(|value| value * 0.04)
        .collect::<Vec<_>>();
    let seed = patterned(geometry.output_len(), 0.9);
    let ones = vec![1.0; geometry.input_len()];
    let masked_mean =
        learned_aggregate_forward(&ones, &vec![0.0; 36], &vec![0.0; 18], geometry, groups)
            .expect("masked softmax contract");
    close(
        "learned invalid-edge masking",
        &masked_mean,
        &vec![1.0; geometry.output_len()],
        1e-6,
    );
    let analytic =
        learned_aggregate_backward(&input, &seed, &edge_kernel, &edge_biases, geometry, groups)
            .expect("learned aggregate backward");
    let objective = |input: &[f32], kernel: &[f32], biases: &[f32]| {
        learned_aggregate_forward(input, kernel, biases, geometry, groups)
            .expect("learned aggregate forward")
            .iter()
            .zip(&seed)
            .map(|(value, gradient)| value * gradient)
            .sum::<f32>()
    };
    let epsilon = 1e-3;
    let mut numeric_input = vec![0.0; input.len()];
    for (index, value) in numeric_input.iter_mut().enumerate() {
        input[index] += epsilon;
        let plus = objective(&input, &edge_kernel, &edge_biases);
        input[index] -= 2.0 * epsilon;
        let minus = objective(&input, &edge_kernel, &edge_biases);
        input[index] += epsilon;
        *value = (plus - minus) / (2.0 * epsilon);
    }
    let mut numeric_kernel = vec![0.0; edge_kernel.len()];
    for (index, value) in numeric_kernel.iter_mut().enumerate() {
        edge_kernel[index] += epsilon;
        let plus = objective(&input, &edge_kernel, &edge_biases);
        edge_kernel[index] -= 2.0 * epsilon;
        let minus = objective(&input, &edge_kernel, &edge_biases);
        edge_kernel[index] += epsilon;
        *value = (plus - minus) / (2.0 * epsilon);
    }
    let mut numeric_biases = vec![0.0; edge_biases.len()];
    for (index, value) in numeric_biases.iter_mut().enumerate() {
        edge_biases[index] += epsilon;
        let plus = objective(&input, &edge_kernel, &edge_biases);
        edge_biases[index] -= 2.0 * epsilon;
        let minus = objective(&input, &edge_kernel, &edge_biases);
        edge_biases[index] += epsilon;
        *value = (plus - minus) / (2.0 * epsilon);
    }
    close(
        "learned input gradient",
        &analytic.input,
        &numeric_input,
        3e-3,
    );
    close(
        "learned edge-kernel gradient",
        &analytic.edge_kernel,
        &numeric_kernel,
        3e-3,
    );
    close(
        "learned edge-bias gradient",
        &analytic.edge_biases,
        &numeric_biases,
        3e-3,
    );
    assert!(learned_aggregate_forward(&input, &edge_kernel, &edge_biases, geometry, 3).is_err());
    assert!(
        learned_aggregate_forward(&input, &edge_kernel[..35], &edge_biases, geometry, groups)
            .is_err()
    );
}

#[test]
fn graph_conv_aggregation_rejects_coordinate_overflow() {
    let geometry = GraphConvAggregationGeometry {
        batch: 1,
        channels: 1,
        input_height: 1,
        input_width: 1,
        output_height: 3,
        output_width: 1,
        size: 1,
        stride_x: 1,
        stride_y: i64::MAX as usize,
        dilation: 1,
        padding: i64::MAX as usize,
    };
    assert!(geometry.checked().is_err());
}

#[test]
fn graph_conv_aggregation_rejects_malformed_shapes() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let geometry = GraphConvAggregationGeometry {
        batch: 1,
        channels: 1,
        input_height: 2,
        input_width: 2,
        output_height: 7,
        output_width: 2,
        size: 3,
        stride_x: 1,
        stride_y: 1,
        dilation: 1,
        padding: 1,
    };
    let input = backend.upload_f32(&[1, 1, 2, 2], &[1.0; 4]).unwrap();
    assert!(
        backend
            .graph_conv_aggregate_f32(&input, geometry, ElementwiseLaunchConfig::default())
            .is_err()
    );
}
