use std::env;

use darknet_cuda::F32ActivationKind;

use crate::DEFAULT_TILE_1D;
use crate::backend::{CutileBackend, grid_1d};
use crate::error::CutileBackendError;
use crate::validate::{
    validate_bias_add_layout, validate_conv2d_layout, validate_maxpool_layout,
    validate_route_concat_layout, validate_upsample_layout,
};

struct ConvCase {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    filters: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    input: Vec<f32>,
    weights: Vec<f32>,
    output_gradient: Vec<f32>,
}

/// Independent analytical NCHW convolution oracle used by the device
/// differential test.  It applies the chain rule directly to the same
/// cross-correlation convention as Darknet; it never perturbs parameters.
fn analytical_convolution_reference(
    case: &ConvCase,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
    let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
    let channels_per_group = case.channels / case.groups;
    let filters_per_group = case.filters / case.groups;
    let input_index = |batch: usize, channel: usize, y: usize, x: usize| {
        (((batch * case.channels + channel) * case.height) + y) * case.width + x
    };
    let output_index = |batch: usize, filter: usize, y: usize, x: usize| {
        (((batch * case.filters + filter) * output_height) + y) * output_width + x
    };
    let weight_index = |filter: usize, channel: usize, ky: usize, kx: usize| {
        (((filter * channels_per_group + channel) * case.kernel) + ky) * case.kernel + kx
    };
    let mut output = vec![0.0; case.batch * case.filters * output_height * output_width];
    let mut input_gradient = vec![0.0; case.input.len()];
    let mut weight_gradient = vec![0.0; case.weights.len()];
    let mut bias_gradient = vec![0.0; case.filters];

    for batch in 0..case.batch {
        for (filter, bias) in bias_gradient.iter_mut().enumerate() {
            let group = filter / filters_per_group;
            for y in 0..output_height {
                for x in 0..output_width {
                    let mut value = 0.0;
                    for channel_in_group in 0..channels_per_group {
                        let channel = group * channels_per_group + channel_in_group;
                        for ky in 0..case.kernel {
                            for kx in 0..case.kernel {
                                let src_y = y * case.stride + ky;
                                let src_x = x * case.stride + kx;
                                if src_y < case.padding
                                    || src_x < case.padding
                                    || src_y >= case.height + case.padding
                                    || src_x >= case.width + case.padding
                                {
                                    continue;
                                }
                                let src_y = src_y - case.padding;
                                let src_x = src_x - case.padding;
                                value += case.input[input_index(batch, channel, src_y, src_x)]
                                    * case.weights
                                        [weight_index(filter, channel_in_group, ky, kx)];
                            }
                        }
                    }
                    let output_offset = output_index(batch, filter, y, x);
                    output[output_offset] = value;
                    let grad = case.output_gradient[output_offset];
                    *bias += grad;
                    for channel_in_group in 0..channels_per_group {
                        let channel = group * channels_per_group + channel_in_group;
                        for ky in 0..case.kernel {
                            for kx in 0..case.kernel {
                                let src_y = y * case.stride + ky;
                                let src_x = x * case.stride + kx;
                                if src_y < case.padding
                                    || src_x < case.padding
                                    || src_y >= case.height + case.padding
                                    || src_x >= case.width + case.padding
                                {
                                    continue;
                                }
                                let src_y = src_y - case.padding;
                                let src_x = src_x - case.padding;
                                let input_offset = input_index(batch, channel, src_y, src_x);
                                let weight_offset =
                                    weight_index(filter, channel_in_group, ky, kx);
                                weight_gradient[weight_offset] +=
                                    case.input[input_offset] * grad;
                                input_gradient[input_offset] +=
                                    case.weights[weight_offset] * grad;
                            }
                        }
                    }
                }
            }
        }
    }

    (output, input_gradient, weight_gradient, bias_gradient)
}

fn assert_device_close(name: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{name} length mismatch");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let tolerance = 3.0e-4 + 3.0e-4 * expected.abs();
        assert!(
            (expected - actual).abs() <= tolerance,
            "{name}[{index}] differs: expected {expected:.8e}, actual {actual:.8e}, tolerance {tolerance:.8e}"
        );
    }
}

#[test]
fn computes_grid_1d() {
    assert_eq!(grid_1d(1024, DEFAULT_TILE_1D), 4);
    assert_eq!(grid_1d(1025, DEFAULT_TILE_1D), 5);
}

#[test]
fn validates_bias_add_layout_for_batched_channels() {
    assert_eq!(
        validate_bias_add_layout(24, 3, 3, 4).expect("valid"),
        (6, 4)
    );
}

#[test]
fn rejects_bias_length_mismatch() {
    let err = validate_bias_add_layout(12, 2, 3, 4).expect_err("expected mismatch");
    assert!(
        matches!(err, CutileBackendError::Shape(message) if message.contains("bias length mismatch"))
    );
}

#[test]
fn activation_kinds_are_stable_for_bridge_dispatch() {
    let supported = [
        F32ActivationKind::Linear,
        F32ActivationKind::Logistic,
        F32ActivationKind::Relu,
        F32ActivationKind::Leaky,
        F32ActivationKind::Mish,
        F32ActivationKind::Swish,
        F32ActivationKind::Tanh,
    ];
    assert_eq!(supported.len(), 7);
}

#[test]
fn validates_route_concat_layout() {
    let layout =
        validate_route_concat_layout(&[12, 4], &[1, 0], &[2, 1], &[3, 1], &[1, 3, 2, 2], 4)
            .expect("valid route layout");
    assert_eq!(layout.batch, 1);
    assert_eq!(layout.total_channels, 3);
    assert_eq!(layout.output_len, 12);
}

#[test]
fn validates_upsample_layout() {
    let layout = validate_upsample_layout(4, &[1, 1, 4, 4], 1, 2, 2, 2, false).expect("valid");
    assert_eq!(layout.output_height, 4);
    assert_eq!(layout.output_width, 4);
    assert_eq!(layout.output_len, 16);
}

#[test]
fn validates_conv2d_layout() {
    let layout = validate_conv2d_layout(8, 1, &[4, 1, 1, 2], 1, 1, 2, 1, 1, 1)
        .expect("valid conv layout");
    assert_eq!(layout.channels_per_group, 1);
    assert_eq!(layout.filters_per_group, 1);
    assert_eq!(layout.output_height, 1);
    assert_eq!(layout.output_width, 2);
    assert_eq!(layout.output_len, 8);
}

#[test]
fn validates_maxpool_layout() {
    let layout = validate_maxpool_layout(16, &[1, 1, 2, 2], 1, 4, 4, 2, 2, 2).expect("valid");
    assert_eq!(layout.output_height, 2);
    assert_eq!(layout.output_width, 2);
    assert_eq!(layout.output_len, 4);
}

#[test]
#[ignore = "requires a CUDA device; set DARKNET_CUTILE_GPU_TEST=1 and run ignored tests"]
fn device_convolution_backward_and_sgd_match_reference() {
    assert!(
        env::var_os("DARKNET_CUTILE_GPU_TEST").is_some(),
        "set DARKNET_CUTILE_GPU_TEST=1 to run the device test"
    );
    let case = ConvCase {
        batch: 2,
        channels: 2,
        height: 3,
        width: 3,
        filters: 2,
        kernel: 3,
        stride: 1,
        padding: 1,
        groups: 1,
        input: (0..36).map(|index| (index as f32 - 13.0) / 11.0).collect(),
        weights: (0..36).map(|index| 0.02 * (index as f32 - 17.0)).collect(),
        output_gradient: (0..36).map(|index| 0.03 * (index as f32 - 8.0)).collect(),
    };
    let output_height = (case.height + 2 * case.padding - case.kernel) / case.stride + 1;
    let output_width = (case.width + 2 * case.padding - case.kernel) / case.stride + 1;
    let (
        expected_output,
        expected_input_gradient,
        expected_weight_gradient,
        expected_bias_gradient,
    ) = analytical_convolution_reference(&case);
    let backend = CutileBackend::new(0).expect("CUDA device should initialize");
    let input = backend
        .upload_f32(
            &[case.batch, case.channels, case.height, case.width],
            &case.input,
        )
        .expect("input upload");
    let weights = backend
        .upload_f32(
            &[
                case.filters,
                case.channels / case.groups,
                case.kernel,
                case.kernel,
            ],
            &case.weights,
        )
        .expect("weight upload");
    let output_gradient = backend
        .upload_f32(
            &[case.batch, case.filters, output_height, output_width],
            &case.output_gradient,
        )
        .expect("output-gradient upload");
    let output = backend
        .conv2d_f32(
            &input,
            &weights,
            &[case.batch, case.filters, output_height, output_width],
            case.channels,
            case.height,
            case.width,
            case.filters,
            case.kernel,
            case.stride,
            case.stride,
            1,
            case.padding,
            case.groups,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile convolution forward");
    let input_gradient = backend
        .conv2d_backward_input_f32(
            &output_gradient,
            &weights,
            &[case.batch, case.channels, case.height, case.width],
            case.channels,
            case.height,
            case.width,
            case.filters,
            case.kernel,
            case.stride,
            case.stride,
            1,
            case.padding,
            case.groups,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile input gradient");
    let weight_gradient = backend
        .conv2d_backward_weights_f32(
            &input,
            &output_gradient,
            &[
                case.filters,
                case.channels / case.groups,
                case.kernel,
                case.kernel,
            ],
            case.channels,
            case.height,
            case.width,
            case.filters,
            case.kernel,
            case.stride,
            case.stride,
            1,
            case.padding,
            case.groups,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile weight gradient");
    let bias_gradient = backend
        .bias_backward_f32(
            &output_gradient,
            case.filters,
            output_height * output_width,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile bias gradient");

    assert_device_close(
        "cuTile convolution output",
        &expected_output,
        &backend.download_f32(&output).expect("output download"),
    );
    assert_device_close(
        "cuTile input gradient",
        &expected_input_gradient,
        &backend
            .download_f32(&input_gradient)
            .expect("input-gradient download"),
    );
    assert_device_close(
        "cuTile weight gradient",
        &expected_weight_gradient,
        &backend
            .download_f32(&weight_gradient)
            .expect("weight-gradient download"),
    );
    assert_device_close(
        "cuTile bias gradient",
        &expected_bias_gradient,
        &backend
            .download_f32(&bias_gradient)
            .expect("bias-gradient download"),
    );

    // Feed the negative mathematical gradient into cuTile's Darknet SGD
    // kernel.  With zero momentum/decay the update must be w - lr*g/batch.
    let descent_weight_update = backend
        .upload_f32(
            &[case.weights.len()],
            &expected_weight_gradient
                .iter()
                .map(|gradient| -*gradient)
                .collect::<Vec<_>>(),
        )
        .expect("weight update upload");
    let descent_bias_update = backend
        .upload_f32(
            &[case.filters],
            &expected_bias_gradient
                .iter()
                .map(|gradient| -*gradient)
                .collect::<Vec<_>>(),
        )
        .expect("bias update upload");
    let (updated_weights, _) = backend
        .sgd_update_f32(
            &weights,
            &descent_weight_update,
            1.0e-3,
            0.0,
            0.0,
            case.batch,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile weight SGD update");
    let biases = backend
        .upload_f32(&[case.filters], &[0.11, -0.19])
        .expect("bias upload");
    let (updated_biases, _) = backend
        .sgd_update_f32(
            &biases,
            &descent_bias_update,
            1.0e-3,
            0.0,
            0.0,
            case.batch,
            DEFAULT_TILE_1D,
        )
        .expect("cuTile bias SGD update");
    let expected_updated_weights = case
        .weights
        .iter()
        .zip(&expected_weight_gradient)
        .map(|(weight, gradient)| weight - 1.0e-3 * gradient / case.batch as f32)
        .collect::<Vec<_>>();
    let expected_updated_biases = [0.11f32, -0.19f32]
        .iter()
        .zip(&expected_bias_gradient)
        .map(|(bias, gradient)| bias - 1.0e-3 * gradient / case.batch as f32)
        .collect::<Vec<_>>();
    assert_device_close(
        "cuTile convolution SGD weights",
        &expected_updated_weights,
        &backend
            .download_f32(&updated_weights)
            .expect("updated-weight download"),
    );
    assert_device_close(
        "cuTile convolution SGD biases",
        &expected_updated_biases,
        &backend
            .download_f32(&updated_biases)
            .expect("updated-bias download"),
    );
}
