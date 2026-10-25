//! Native `[wmhf]` f32: per-primitive parity against host references, and
//! end-to-end `GpuExecutableNetwork<OxideBackend>` against the Rust CPU
//! reference for inference, the native training forward, the native backward
//! (input gradient, staged parameter gradients, BN rolling statistics) and
//! the following host-staged SGD step.
//!
//! Runtime tolerances are relative to `max(1, max |reference|)`: 2e-4 for
//! outputs and 5e-4 for gradients, rolling statistics and updated parameters
//! (device convolution, batch-norm and scan reductions sum in a different
//! order from the host loops). Rust CPU parity only, no compiled-C++ golden.
use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, LayerTrainBuffers, Network,
    Tensor,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

#[test]
fn wmhf_high_frequency_branch_forward_backward_matches_cpu_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let (batch, channels, height, width) = (2usize, 2usize, 3usize, 5usize);
    let height2 = height.div_ceil(2);
    let width2 = width.div_ceil(2);
    let coeff_len = batch * channels * height2 * width2;
    let coeff_shape = [batch, channels, height2, width2];
    let high_shape = [batch, channels, height, width];
    let lh = (0..coeff_len)
        .map(|i| (i as f32 - 3.0) * 0.13)
        .collect::<Vec<_>>();
    let hl = (0..coeff_len)
        .map(|i| (2.0 - i as f32) * 0.09)
        .collect::<Vec<_>>();
    let hh = (0..coeff_len)
        .map(|i| (i as f32 % 4.0 - 1.5) * 0.21)
        .collect::<Vec<_>>();
    let lh_device = backend.upload_f32(&coeff_shape, &lh).unwrap();
    let hl_device = backend.upload_f32(&coeff_shape, &hl).unwrap();
    let hh_device = backend.upload_f32(&coeff_shape, &hh).unwrap();
    let output = F32TensorOpsBackend::wmhf_high_frequency_forward_f32(
        &backend, &lh_device, &hl_device, &hh_device, batch, channels, height, width, config,
    )
    .expect("native high-frequency branch forward");
    let mut expected = vec![0.0; batch * channels * height * width];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let coeff = (b * channels + c) * height2 * width2 + (y / 2) * width2 + x / 2;
                    let index = (b * channels + c) * height * width + y * width + x;
                    expected[index] = lh[coeff].abs() + hl[coeff].abs() + hh[coeff].abs();
                }
            }
        }
    }
    let actual = backend.download_f32(&output).unwrap();
    for (index, (&actual, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() < 1.0e-6,
            "HF forward[{index}]: {actual} != {want}"
        );
    }

    let output_grad = (0..expected.len())
        .map(|i| (i as f32 * 0.19).sin())
        .collect::<Vec<_>>();
    let output_grad_device = backend.upload_f32(&high_shape, &output_grad).unwrap();
    let (lh_grad, hl_grad, hh_grad) = F32TensorOpsBackend::wmhf_high_frequency_backward_f32(
        &backend,
        &output_grad_device,
        &lh_device,
        &hl_device,
        &hh_device,
        batch,
        channels,
        height,
        width,
        config,
    )
    .expect("native high-frequency branch backward");
    let mut expected_grads = [
        vec![0.0; coeff_len],
        vec![0.0; coeff_len],
        vec![0.0; coeff_len],
    ];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let coeff = (b * channels + c) * height2 * width2 + (y / 2) * width2 + x / 2;
                    let index = (b * channels + c) * height * width + y * width + x;
                    let upstream = output_grad[index];
                    expected_grads[0][coeff] += upstream * sign_no_zero(lh[coeff]);
                    expected_grads[1][coeff] += upstream * sign_no_zero(hl[coeff]);
                    expected_grads[2][coeff] += upstream * sign_no_zero(hh[coeff]);
                }
            }
        }
    }
    for (name, grad, want) in [
        (
            "LH",
            backend.download_f32(&lh_grad).unwrap(),
            &expected_grads[0],
        ),
        (
            "HL",
            backend.download_f32(&hl_grad).unwrap(),
            &expected_grads[1],
        ),
        (
            "HH",
            backend.download_f32(&hh_grad).unwrap(),
            &expected_grads[2],
        ),
    ] {
        for (index, (&actual, &want)) in grad.iter().zip(want).enumerate() {
            assert!(
                (actual - want).abs() < 1.0e-6,
                "HF {name} gradient[{index}]: {actual} != {want}"
            );
        }
    }
}

fn scan4_backward_reference(
    input: &[f32],
    output_grad: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let area = height * width;
    let mut input_grad = vec![0.0; input.len()];
    let mut weight_grad = vec![0.0; 4 * channels];
    for b in 0..batch {
        for c in 0..channels {
            let sigmoid = if weights[c] >= 0.0 {
                1.0 / (1.0 + (-weights[c]).exp())
            } else {
                let z = weights[c].exp();
                z / (1.0 + z)
            };
            let a = 0.98 * sigmoid;
            let b_weight = weights[channels + c];
            let c_weight = weights[2 * channels + c];
            let d_weight = weights[3 * channels + c];
            let base = (b * channels + c) * area;
            let mut run = |positions: Vec<usize>| {
                let mut states = Vec::with_capacity(positions.len());
                let mut state = 0.0;
                for &spatial in &positions {
                    state = a * state + b_weight * input[base + spatial];
                    states.push(state);
                }
                let (mut da, mut db, mut dc, mut dd, mut ds_next) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for t in (0..positions.len()).rev() {
                    let spatial = positions[t];
                    let index = base + spatial;
                    let dy = 0.25 * output_grad[index];
                    let value = input[index];
                    let state_prev = if t == 0 { 0.0 } else { states[t - 1] };
                    dc += dy * states[t];
                    dd += dy * value;
                    let ds = ds_next + dy * c_weight;
                    da += ds * state_prev;
                    db += ds * value;
                    input_grad[index] += dy * d_weight + ds * b_weight;
                    ds_next = ds * a;
                }
                weight_grad[c] += da * 0.98 * sigmoid * (1.0 - sigmoid);
                weight_grad[channels + c] += db;
                weight_grad[2 * channels + c] += dc;
                weight_grad[3 * channels + c] += dd;
            };
            for y in 0..height {
                run((0..width).map(|x| y * width + x).collect());
                run((0..width).rev().map(|x| y * width + x).collect());
            }
            for x in 0..width {
                run((0..height).map(|y| y * width + x).collect());
                run((0..height).rev().map(|y| y * width + x).collect());
            }
        }
    }
    (input_grad, weight_grad)
}

#[test]
fn wmhf_gate_forward_backward_matches_cpu_and_trait_dispatch() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let shape = [2usize, 2usize, 2usize, 3usize];
    let count = shape.iter().product::<usize>();
    let fuse = (0..count)
        .map(|i| i as f32 * 0.07 - 0.3)
        .collect::<Vec<_>>();
    let gate = (0..count)
        .map(|i| (i as f32 * 0.13).sin())
        .collect::<Vec<_>>();
    let projected = (0..count)
        .map(|i| (i as f32 * 0.09).cos() * 0.5)
        .collect::<Vec<_>>();
    let shortcut = (0..count)
        .map(|i| (i as f32 - 2.0) * 0.025)
        .collect::<Vec<_>>();
    let output_grad = (0..count)
        .map(|i| (i as f32 * 0.21).sin() * 0.7)
        .collect::<Vec<_>>();
    let fuse_device = backend.upload_f32(&shape, &fuse).unwrap();
    let gate_device = backend.upload_f32(&shape, &gate).unwrap();
    let projected_device = backend.upload_f32(&shape, &projected).unwrap();
    let shortcut_device = backend.upload_f32(&shape, &shortcut).unwrap();
    let output_grad_device = backend.upload_f32(&shape, &output_grad).unwrap();
    let scale = 0.37;

    for (use_shortcut, shortcut_input) in [(false, None), (true, Some(&shortcut_device))] {
        let output = F32TensorOpsBackend::wmhf_apply_gate_f32(
            &backend,
            &fuse_device,
            &gate_device,
            &projected_device,
            shortcut_input,
            scale,
            &shape,
            config,
        )
        .expect("native WMHF gate forward");
        let actual = backend.download_f32(&output).unwrap();
        for i in 0..count {
            let want = fuse[i]
                + scale * gate[i] * projected[i]
                + if use_shortcut { shortcut[i] } else { 0.0 };
            assert!((actual[i] - want).abs() < 1.0e-6, "gate forward[{i}]");
        }

        let (dfuse, dgate, dprojected, dshortcut) =
            F32TensorOpsBackend::wmhf_apply_gate_backward_f32(
                &backend,
                &output_grad_device,
                &gate_device,
                &projected_device,
                scale,
                use_shortcut,
                &shape,
                config,
            )
            .expect("native WMHF gate backward");
        for (name, actual, want) in [
            (
                "fuse",
                backend.download_f32(&dfuse).unwrap(),
                output_grad.clone(),
            ),
            (
                "gate",
                backend.download_f32(&dgate).unwrap(),
                output_grad
                    .iter()
                    .zip(&projected)
                    .map(|(dy, p)| scale * p * dy)
                    .collect(),
            ),
            (
                "projected",
                backend.download_f32(&dprojected).unwrap(),
                output_grad
                    .iter()
                    .zip(&gate)
                    .map(|(dy, g)| scale * g * dy)
                    .collect(),
            ),
            (
                "shortcut",
                backend.download_f32(&dshortcut).unwrap(),
                output_grad
                    .iter()
                    .map(|dy| if use_shortcut { *dy } else { 0.0 })
                    .collect(),
            ),
        ] {
            for (i, (&actual, &want)) in actual.iter().zip(&want).enumerate() {
                assert!((actual - want).abs() < 1.0e-6, "gate {name} gradient[{i}]");
            }
        }
    }
}

#[test]
fn wmhf_channel_split_matches_batched_nchw_reference_via_trait() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let (batch, channels, height, width, begin, count) =
        (2usize, 4usize, 3usize, 5usize, 1usize, 2usize);
    let input = (0..batch * channels * height * width)
        .map(|i| i as f32 * 0.25 - 1.0)
        .collect::<Vec<_>>();
    let shape = [batch, channels, height, width];
    let device = backend.upload_f32(&shape, &input).unwrap();
    let actual = F32TensorOpsBackend::wmhf_split_channels_f32(
        &backend, &device, batch, channels, height, width, begin, count, config,
    )
    .expect("native WMHF channel split");
    let actual = backend.download_f32(&actual).unwrap();
    let area = height * width;
    let mut expected = Vec::with_capacity(batch * count * area);
    for b in 0..batch {
        let start = (b * channels + begin) * area;
        expected.extend_from_slice(&input[start..start + count * area]);
    }
    assert_eq!(actual, expected);
}

#[test]
fn wmhf_local_branch_concat_matches_batched_channel_major_reference_via_trait() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let (batch, channels, height, width) = (2usize, 2usize, 3usize, 5usize);
    let shape = [batch, channels, height, width];
    let branch_len = batch * channels * height * width;
    let branch3 = (0..branch_len).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
    let branch5 = (0..branch_len)
        .map(|i| i as f32 * 0.1 + 10.0)
        .collect::<Vec<_>>();
    let branch7 = (0..branch_len)
        .map(|i| i as f32 * 0.1 - 7.0)
        .collect::<Vec<_>>();
    let d3 = backend.upload_f32(&shape, &branch3).unwrap();
    let d5 = backend.upload_f32(&shape, &branch5).unwrap();
    let d7 = backend.upload_f32(&shape, &branch7).unwrap();
    let output = F32TensorOpsBackend::wmhf_concat_local_f32(
        &backend, &d3, &d5, &d7, batch, channels, height, width, config,
    )
    .expect("native WMHF local concat");
    let actual = backend.download_f32(&output).unwrap();
    let area = height * width;
    let mut expected = vec![0.0; batch * 3 * channels * area];
    for b in 0..batch {
        for c in 0..channels {
            for spatial in 0..area {
                let dst = (b * 3 * channels + c) * area + spatial;
                let src = (b * channels + c) * area + spatial;
                expected[dst] = branch3[src];
                expected[dst + channels * area] = branch5[src];
                expected[dst + 2 * channels * area] = branch7[src];
            }
        }
    }
    assert_eq!(actual, expected);
}

fn sign_no_zero(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[test]
fn wmhf_scan4_forward_f32_matches_four_direction_cpu_reference() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let (batch, channels, height, width) = (2usize, 2usize, 2usize, 3usize);
    let input = (0..batch * channels * height * width)
        .map(|i| ((i as f32 * 0.23).cos() * 0.9) + (i % 5) as f32 * 0.11)
        .collect::<Vec<_>>();
    let weights = [0.2, -0.4, 0.7, -0.3, 1.1, 0.6, 0.8, -0.2];
    let input_device = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .expect("upload input");
    let weights_device = backend
        .upload_f32(&[4, channels], &weights)
        .expect("upload scan weights");
    let output = backend
        .wmhf_scan4_forward_f32(
            &input_device,
            &weights_device,
            batch,
            channels,
            height,
            width,
            config,
        )
        .expect("native four-direction scan");

    let mut expected = vec![0.0; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            let raw_a = weights[c];
            let sigmoid = if raw_a >= 0.0 {
                1.0 / (1.0 + (-raw_a).exp())
            } else {
                let z = raw_a.exp();
                z / (1.0 + z)
            };
            let a = 0.98 * sigmoid;
            let b_weight = weights[channels + c];
            let c_weight = weights[2 * channels + c];
            let d_weight = weights[3 * channels + c];
            let base = (b * channels + c) * height * width;
            let mut run = |positions: Vec<usize>| {
                let mut state = 0.0;
                for spatial in positions {
                    let value = input[base + spatial];
                    state = a * state + b_weight * value;
                    expected[base + spatial] += 0.25 * (c_weight * state + d_weight * value);
                }
            };
            for y in 0..height {
                run((0..width).map(|x| y * width + x).collect());
                run((0..width).rev().map(|x| y * width + x).collect());
            }
            for x in 0..width {
                run((0..height).map(|y| y * width + x).collect());
                run((0..height).rev().map(|y| y * width + x).collect());
            }
        }
    }
    let actual = backend.download_f32(&output).unwrap();
    for (index, (&actual, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - want).abs() <= 3.0e-6 * (1.0 + want.abs()),
            "scan4[{index}]: got {actual}, want {want}"
        );
    }

    let output_grad = (0..input.len())
        .map(|i| ((i as f32 * 0.17).sin() * 0.8) + (i % 3) as f32 * 0.09)
        .collect::<Vec<_>>();
    let output_grad_device = backend
        .upload_f32(&[batch, channels, height, width], &output_grad)
        .expect("upload output gradient");
    let (input_grad, weight_grad) = backend
        .wmhf_scan4_backward_f32(
            &input_device,
            &output_grad_device,
            &weights_device,
            batch,
            channels,
            height,
            width,
            config,
        )
        .expect("native four-direction scan backward");
    let (expected_input_grad, expected_weight_grad) = scan4_backward_reference(
        &input,
        &output_grad,
        &weights,
        batch,
        channels,
        height,
        width,
    );
    for (name, actual, want) in [
        (
            "scan input gradient",
            backend.download_f32(&input_grad).unwrap(),
            expected_input_grad.as_slice(),
        ),
        (
            "scan weight gradient",
            backend.download_f32(&weight_grad).unwrap(),
            expected_weight_grad.as_slice(),
        ),
    ] {
        for (index, (&actual, &want)) in actual.iter().zip(want).enumerate() {
            assert!(
                (actual - want).abs() <= 2.0e-5 * (1.0 + want.abs()),
                "{name}[{index}]: got {actual}, want {want}"
            );
        }
    }
}

#[test]
fn wmhf_haar_dwt_f32_matches_nchw_reference_on_odd_non_square_batches() {
    let backend = OxideBackend::new().expect("Oxide backend");
    let config = ElementwiseLaunchConfig::default();
    let (batch, channels, height, width) = (2usize, 3usize, 3usize, 5usize);
    let height2 = height.div_ceil(2);
    let width2 = width.div_ceil(2);
    let input = (0..batch * channels * height * width)
        .map(|i| ((i as f32 * 0.31).sin() * 2.0) + (i % 7) as f32 * 0.17)
        .collect::<Vec<_>>();
    let input_device = backend
        .upload_f32(&[batch, channels, height, width], &input)
        .expect("upload input");
    let (ll, lh, hl, hh) = backend
        .wmhf_haar_dwt_f32(&input_device, batch, channels, height, width, config)
        .expect("native WMHF Haar DWT");

    let mut expected = [
        vec![0.0; batch * channels * height2 * width2],
        vec![0.0; batch * channels * height2 * width2],
        vec![0.0; batch * channels * height2 * width2],
        vec![0.0; batch * channels * height2 * width2],
    ];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height2 {
                let y0 = (2 * y).min(height - 1);
                let y1 = (y0 + 1).min(height - 1);
                for x in 0..width2 {
                    let x0 = (2 * x).min(width - 1);
                    let x1 = (x0 + 1).min(width - 1);
                    let src = (b * channels + c) * height * width;
                    let dst = (b * channels + c) * height2 * width2 + y * width2 + x;
                    let v00 = input[src + y0 * width + x0];
                    let v01 = input[src + y0 * width + x1];
                    let v10 = input[src + y1 * width + x0];
                    let v11 = input[src + y1 * width + x1];
                    expected[0][dst] = 0.5 * (v00 + v01 + v10 + v11);
                    expected[1][dst] = 0.5 * (v00 - v01 + v10 - v11);
                    expected[2][dst] = 0.5 * (v00 + v01 - v10 - v11);
                    expected[3][dst] = 0.5 * (v00 - v01 - v10 + v11);
                }
            }
        }
    }
    for (name, actual, want) in [
        (
            "LL",
            backend.download_f32(&ll).unwrap(),
            expected[0].as_slice(),
        ),
        (
            "LH",
            backend.download_f32(&lh).unwrap(),
            expected[1].as_slice(),
        ),
        (
            "HL",
            backend.download_f32(&hl).unwrap(),
            expected[2].as_slice(),
        ),
        (
            "HH",
            backend.download_f32(&hh).unwrap(),
            expected[3].as_slice(),
        ),
    ] {
        assert_eq!(actual.len(), want.len(), "{name} size");
        for (index, (&actual, &want)) in actual.iter().zip(want).enumerate() {
            assert!(
                (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
                "{name}[{index}]: got {actual}, want {want}"
            );
        }
    }

    let seeds = [
        (0..expected[0].len())
            .map(|i| ((i as f32 * 0.13).cos() * 0.7) - 0.1)
            .collect::<Vec<_>>(),
        (0..expected[0].len())
            .map(|i| ((i as f32 * 0.21).sin() * 0.4) + 0.05)
            .collect::<Vec<_>>(),
        (0..expected[0].len())
            .map(|i| ((i as f32 * 0.09).cos() * 0.3) + 0.2)
            .collect::<Vec<_>>(),
        (0..expected[0].len())
            .map(|i| ((i as f32 * 0.17).sin() * 0.5) - 0.07)
            .collect::<Vec<_>>(),
    ];
    let coefficient_shape = [batch, channels, height2, width2];
    let seed_devices = seeds
        .iter()
        .map(|seed| {
            backend
                .upload_f32(&coefficient_shape, seed)
                .expect("upload seed")
        })
        .collect::<Vec<_>>();
    let input_grad = backend
        .wmhf_haar_dwt_backward_f32(
            &seed_devices[0],
            &seed_devices[1],
            &seed_devices[2],
            &seed_devices[3],
            batch,
            channels,
            height,
            width,
            config,
        )
        .expect("native WMHF Haar DWT adjoint");
    let mut expected_input_grad = vec![0.0; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height2 {
                let y0 = (2 * y).min(height - 1);
                let y1 = (y0 + 1).min(height - 1);
                for x in 0..width2 {
                    let x0 = (2 * x).min(width - 1);
                    let x1 = (x0 + 1).min(width - 1);
                    let index = (b * channels + c) * height2 * width2 + y * width2 + x;
                    let a = 0.5 * seeds[0][index];
                    let b1 = 0.5 * seeds[1][index];
                    let c1 = 0.5 * seeds[2][index];
                    let d = 0.5 * seeds[3][index];
                    let base = (b * channels + c) * height * width;
                    expected_input_grad[base + y0 * width + x0] += a + b1 + c1 + d;
                    expected_input_grad[base + y0 * width + x1] += a - b1 + c1 - d;
                    expected_input_grad[base + y1 * width + x0] += a + b1 - c1 - d;
                    expected_input_grad[base + y1 * width + x1] += a - b1 - c1 + d;
                }
            }
        }
    }
    let actual_input_grad = backend.download_f32(&input_grad).unwrap();
    for (index, (&actual, &want)) in actual_input_grad
        .iter()
        .zip(&expected_input_grad)
        .enumerate()
    {
        assert!(
            (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
            "DWT adjoint[{index}]: got {actual}, want {want}"
        );
    }

    let synthesized = backend
        .wmhf_haar_iwt_f32(
            &seed_devices[0],
            &seed_devices[1],
            &seed_devices[2],
            &seed_devices[3],
            batch,
            channels,
            height,
            width,
            config,
        )
        .expect("native WMHF Haar IWT");
    let synthesized = backend.download_f32(&synthesized).unwrap();
    for (index, (&actual, &want)) in synthesized.iter().zip(&expected_input_grad).enumerate() {
        assert!(
            (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
            "IWT[{index}]: got {actual}, want {want}"
        );
    }

    let (iwt_dll, iwt_dlh, iwt_dhl, iwt_dhh) = backend
        .wmhf_haar_iwt_backward_f32(&input_device, batch, channels, height, width, config)
        .expect("native WMHF Haar IWT adjoint");
    for (name, actual, want) in [
        (
            "IWT dLL",
            backend.download_f32(&iwt_dll).unwrap(),
            expected[0].as_slice(),
        ),
        (
            "IWT dLH",
            backend.download_f32(&iwt_dlh).unwrap(),
            expected[1].as_slice(),
        ),
        (
            "IWT dHL",
            backend.download_f32(&iwt_dhl).unwrap(),
            expected[2].as_slice(),
        ),
        (
            "IWT dHH",
            backend.download_f32(&iwt_dhh).unwrap(),
            expected[3].as_slice(),
        ),
    ] {
        for (index, (&actual, &want)) in actual.iter().zip(want).enumerate() {
            assert!(
                (actual - want).abs() <= 2.0e-6 * (1.0 + want.abs()),
                "{name}[{index}]: got {actual}, want {want}"
            );
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * scale, "{name}[{i}]: got {g}, want {w}");
    }
}

/// A one-layer `[wmhf]` CPU network with seeded parameters for every block.
fn wmhf_network(cfg: &str) -> ExecutableNetwork {
    let cfg = ConfigFile::parse_str("native-wmhf.cfg", cfg).unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::Wmhf(params) = &mut cpu.params[0] else {
        panic!("expected WMHF params")
    };
    let mut rng = Rng(0x5EED_57A7);
    for conv in [
        &mut params.pre,
        &mut params.local_dw3,
        &mut params.local_dw5,
        &mut params.local_dw7,
        &mut params.local_mix,
        &mut params.hf_gate,
        &mut params.fuse,
    ] {
        for value in conv.weights.iter_mut() {
            *value = 0.4 * rng.next();
        }
        for value in conv.biases.iter_mut() {
            *value = 0.2 * rng.next();
        }
        if let Some(batch_norm) = &mut conv.batch_norm {
            for value in batch_norm.scales.iter_mut() {
                *value = 1.0 + 0.3 * rng.next();
            }
            for value in batch_norm.rolling_mean.iter_mut() {
                *value = 0.2 * rng.next();
            }
            for value in batch_norm.rolling_variance.iter_mut() {
                *value = 1.0 + 0.5 * rng.next();
            }
        }
    }
    for value in params.scan_weights.iter_mut() {
        *value = rng.next();
    }
    cpu
}

fn wmhf_input(cpu: &ExecutableNetwork, seed: u64) -> Tensor {
    let shape = cpu.network.net.input_shape();
    Tensor::from_data(shape, Rng(seed).vec(shape.batch * shape.element_count())).unwrap()
}

fn wmhf_updates(buffers: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::Wmhf(train) = buffers else {
        panic!("expected WMHF train buffers")
    };
    let mut blob = Vec::new();
    for conv in [
        &train.pre,
        &train.local_dw3,
        &train.local_dw5,
        &train.local_dw7,
        &train.local_mix,
        &train.hf_gate,
        &train.fuse,
    ] {
        blob.extend_from_slice(&conv.weight_updates);
        blob.extend_from_slice(&conv.bias_updates);
        if let Some(batch_norm) = &conv.batch_norm {
            blob.extend_from_slice(&batch_norm.scale_updates);
        }
    }
    blob.extend_from_slice(&train.scan_weight_updates);
    blob
}

/// Weights, biases, BN scales and rolling statistics, and scan weights.
fn wmhf_blob(params: &LayerParams) -> Vec<f32> {
    let LayerParams::Wmhf(params) = params else {
        panic!("expected WMHF params")
    };
    let mut blob = Vec::new();
    for conv in [
        &params.pre,
        &params.local_dw3,
        &params.local_dw5,
        &params.local_dw7,
        &params.local_mix,
        &params.hf_gate,
        &params.fuse,
    ] {
        blob.extend_from_slice(&conv.weights);
        blob.extend_from_slice(&conv.biases);
        if let Some(batch_norm) = &conv.batch_norm {
            blob.extend_from_slice(&batch_norm.scales);
            blob.extend_from_slice(&batch_norm.rolling_mean);
            blob.extend_from_slice(&batch_norm.rolling_variance);
        }
    }
    blob.extend_from_slice(&params.scan_weights);
    blob
}

// `filters=8`, 8 input channels: identity 2, local 3, global 3, shortcut used,
// batch-norm sub-convolutions, odd non-square plane.
const WMHF_BN_SHORTCUT: &str = "[net]\nbatch=2\nwidth=5\nheight=3\nchannels=8\n[wmhf]\nfilters=8\nbatch_normalize=1\nshortcut=1\nactivation=leaky\n";
// No identity channels (route concat without the projected slice), no BN,
// no shortcut (input channels differ), swish output.
const WMHF_NO_IDENTITY: &str = "[net]\nbatch=2\nwidth=4\nheight=4\nchannels=3\n[wmhf]\nfilters=6\nidentity_ratio=0\nbatch_normalize=0\nshortcut=1\nfreq_scale=0.4\nactivation=swish\n";
// Plane larger than the 7x7 depthwise kernel, linear output, BN and shortcut.
const WMHF_WIDE: &str = "[net]\nbatch=2\nwidth=9\nheight=7\nchannels=4\n[wmhf]\nfilters=4\nbatch_normalize=1\nshortcut=1\nactivation=linear\n";

fn wmhf_inference_check(cfg: &str) {
    let cpu = wmhf_network(cfg);
    let input = wmhf_input(&cpu, 99);
    let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let actual = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    assert_eq!(actual.shape, expected.shape, "{cfg}");
    close("WMHF inference", &actual.data, &expected.data, 2e-4);
}

#[test]
fn wmhf_native_inference_forward_matches_cpu() {
    for cfg in [WMHF_BN_SHORTCUT, WMHF_NO_IDENTITY, WMHF_WIDE] {
        wmhf_inference_check(cfg);
    }
}

/// Native training forward, native backward (input gradient, staged parameter
/// gradients, BN rolling statistics) and the host-staged SGD step.
fn wmhf_train_check(cfg: &str) {
    let mut cpu = wmhf_network(cfg);
    let input = wmhf_input(&cpu, 5);
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();

    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    let expected = cpu_forward.final_output().unwrap().clone();
    let actual = gpu_state.cpu_forward.final_output().unwrap().clone();
    close("WMHF training forward", &actual.data, &expected.data, 2e-4);

    let mut rng = Rng(17);
    let seed = Tensor::from_data(expected.shape, rng.vec(expected.data.len())).unwrap();
    let seeds = [Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let cpu_backward = cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &seeds)
        .unwrap()
        .expect("native WMHF input gradient");
    close(
        "WMHF input gradient",
        &gpu_input_gradient.data,
        &cpu_backward.input_gradient.data,
        5e-4,
    );
    close(
        "WMHF staged gradients",
        &wmhf_updates(&gpu.host_fallback_train_buffers()[0]),
        &wmhf_updates(&cpu.train_buffers[0]),
        5e-4,
    );
    close(
        "WMHF rolling statistics",
        &wmhf_blob(&gpu.params[0]),
        &wmhf_blob(&cpu.params[0]),
        5e-4,
    );

    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    close(
        "WMHF updated parameters",
        &wmhf_blob(&gpu.params[0]),
        &wmhf_blob(&cpu.params[0]),
        5e-4,
    );
}

#[test]
fn wmhf_native_training_backward_and_update_match_cpu() {
    for cfg in [WMHF_BN_SHORTCUT, WMHF_NO_IDENTITY, WMHF_WIDE] {
        wmhf_train_check(cfg);
    }
}
