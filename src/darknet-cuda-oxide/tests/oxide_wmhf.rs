use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

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
