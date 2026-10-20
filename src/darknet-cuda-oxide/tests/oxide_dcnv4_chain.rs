//! Native coarse predictor/upsample/aggregation vs real CPU runtime math.
//! No runtime GPU wiring, optimizer updates or compiled-C++ golden.

use darknet_core::{ConfigFile, ExecutableNetwork, LayerKind, LayerParams, Network, Tensor};
use darknet_cuda::host_ops::dcnv4::{Dcnv4Composition, Dcnv4Geometry};
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::OxideBackend;

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() < 3e-4 + 3e-4 * want.abs(),
            "{label}[{i}]: {got} vs {want}"
        );
    }
}

fn build(
    size: usize,
    d_stride: usize,
    dilation: usize,
    softmax: bool,
    remove_center: bool,
    groups: usize,
) -> (ExecutableNetwork, Dcnv4Composition) {
    let padding = dilation * (size / 2);
    let cfg = format!(
        "[net]\nwidth=7\nheight=5\nchannels=4\nbatch=2\ntry_fix_nan=0\n\
        [dcnv4]\nfilters=4\nsize={size}\nstride_x=2\nstride_y=1\npadding={padding}\n\
        dilation={dilation}\ngroups={groups}\nd_stride={d_stride}\noffset_scale=0.75\n\
        softmax={}\nremove_center={}\nactivation=linear\nbatch_normalize=0\n",
        u8::from(softmax),
        u8::from(remove_center)
    );
    let network = Network::from_config(ConfigFile::parse_str("inline.cfg", &cfg).unwrap()).unwrap();
    let mut cpu = ExecutableNetwork::new(network);
    let layer = &cpu.network.layers[0];
    let LayerKind::Dcnv4(spec) = &layer.kind else {
        panic!("not DCNv4");
    };
    let shape = layer.input_shape;
    let desc = Dcnv4Composition {
        geometry: Dcnv4Geometry {
            batch: shape.batch,
            channels: shape.channels,
            groups: spec.groups,
            input_height: shape.height,
            input_width: shape.width,
            output_height: layer.output_shape.height,
            output_width: layer.output_shape.width,
            size: spec.size,
            stride_x: spec.stride_x,
            stride_y: spec.stride_y,
            padding: spec.padding,
            dilation: spec.dilation,
            softmax: spec.softmax,
            remove_center: spec.remove_center,
        },
        d_stride: spec.d_stride,
        offset_scale: spec.offset_scale,
    };
    desc.checked().unwrap();
    let LayerParams::Dcnv4(params) = &mut cpu.params[0] else {
        unreachable!()
    };
    assert_eq!(params.offset_weights.len(), desc.weight_count());
    assert_eq!(params.offset_biases.len(), desc.geometry.offset_dim());
    for (i, w) in params.offset_weights.iter_mut().enumerate() {
        *w = ((i * 7 % 17) as f32 - 8.0) * 0.002;
    }
    for (i, b) in params.offset_biases.iter_mut().enumerate() {
        *b = 0.43 + (i % 7) as f32 * 0.03;
    }
    // Padding is not a point or a mask logit, even with softmax enabled.
    let raw = groups * desc.geometry.kernel_points() * 3;
    params.offset_biases[raw..].fill(20.0);
    params.biases.fill(0.0);
    (cpu, desc)
}

#[test]
fn native_composition_matches_real_cpu_outputs_and_all_math_gradients() {
    let backend = OxideBackend::new().unwrap();
    let config = ElementwiseLaunchConfig::default();
    for (size, d_stride, dilation, groups) in [
        (1, 2, 1, 2),
        (3, 1, 1, 1),
        (3, 2, 1, 2),
        (3, 3, 2, 2),
        (5, 2, 1, 2),
        (7, 1, 1, 2),
        (9, 1, 1, 2),
        (11, 1, 1, 2),
        (13, 1, 1, 2),
        (15, 1, 1, 2),
        (17, 1, 1, 2),
        (19, 1, 1, 2),
    ] {
        for softmax in [false, true] {
            for remove_center in [false, true] {
                if size == 1 && remove_center {
                    continue;
                }
                let (cpu, desc) = build(size, d_stride, dilation, softmax, remove_center, groups);
                let label = format!(
                    "size={size} ds={d_stride} dil={dilation} groups={groups} sm={softmax} rc={remove_center}"
                );
                let shape = cpu.network.net.input_shape();
                let data: Vec<_> = (0..shape.batch * shape.element_count())
                    .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.09)
                    .collect();
                let input = Tensor::from_data(shape, data.clone()).unwrap();
                let forward = cpu.forward(&input).unwrap();
                let out = forward.final_output().unwrap();
                let seed_data: Vec<_> = (0..out.data.len())
                    .map(|i| ((i * 11 % 19) as f32 - 9.0) * 0.17)
                    .collect();
                let seed = Tensor::from_data(out.shape, seed_data.clone()).unwrap();
                let backward = cpu.backward(&input, &forward, &seed).unwrap();
                let LayerParams::Dcnv4(params) = &cpu.params[0] else {
                    unreachable!()
                };
                let LayerParams::Dcnv4(want) = &backward.param_gradients[0] else {
                    unreachable!()
                };
                let upload = |data: &[f32]| backend.upload_f32(&[data.len()], data).unwrap();
                let x = upload(&data);
                let weights = upload(&params.offset_weights);
                let biases = upload(&params.offset_biases);
                let (output, offsets) = backend
                    .dcnv4_forward_native(&x, &weights, &biases, desc, config)
                    .unwrap();
                assert_close(
                    &format!("{label}/output"),
                    &backend.download_f32(&output).unwrap(),
                    &out.data,
                );
                let cache_before = backend.download_f32(&offsets).unwrap();
                // Cache owns its buffer independently of temporary coarse tensors
                // and forward operands. Force pool reuse before backward.
                drop((x, weights, biases));
                let x = upload(&data);
                let weights = upload(&params.offset_weights);
                let dy = upload(&seed_data);
                let got = backend
                    .dcnv4_backward_native(&x, &weights, &offsets, &dy, desc, config)
                    .unwrap();
                assert_close(
                    &format!("{label}/input"),
                    &backend.download_f32(&got.input).unwrap(),
                    &backward.input_gradient.data,
                );
                let dw = backend.download_f32(&got.offset_weights).unwrap();
                let db = backend.download_f32(&got.offset_biases).unwrap();
                assert_close(&format!("{label}/weights"), &dw, &want.offset_weights);
                assert_close(&format!("{label}/biases"), &db, &want.offset_biases);
                assert!(
                    dw.iter().any(|v| v.abs() > 1e-4),
                    "{label}: vacuous weights"
                );
                let raw = groups * desc.geometry.kernel_points() * 3;
                let row_weights = shape.channels * desc.geometry.size * desc.geometry.size;
                assert!(
                    dw[raw * row_weights..].iter().all(|v| *v == 0.0),
                    "padding weights"
                );
                assert!(db[raw..].iter().all(|v| *v == 0.0), "padding biases");
                assert_eq!(
                    backend.download_f32(&offsets).unwrap(),
                    cache_before,
                    "cache modified"
                );
                assert_close(
                    "live forward output",
                    &backend.download_f32(&output).unwrap(),
                    &out.data,
                );
                assert_eq!(backend.download_f32(&x).unwrap(), data, "input modified");
                assert_eq!(
                    backend.download_f32(&weights).unwrap(),
                    params.offset_weights,
                    "weights modified"
                );
            }
        }
    }
}

#[test]
fn native_composition_matches_device_finite_differences() {
    let backend = OxideBackend::new().unwrap();
    let cfg = ElementwiseLaunchConfig::default();
    let eps = 0.002;
    for softmax in [false, true] {
        for remove_center in [false, true] {
            let g = Dcnv4Geometry {
                batch: 1,
                channels: 2,
                groups: 1,
                input_height: 3,
                input_width: 4,
                output_height: 3,
                output_width: 4,
                size: 3,
                stride_x: 1,
                stride_y: 1,
                padding: 1,
                dilation: 1,
                softmax,
                remove_center,
            };
            let desc = Dcnv4Composition {
                geometry: g,
                d_stride: 2,
                offset_scale: 1.0,
            };
            let input: Vec<_> = (0..24).map(|i| (i % 7) as f32 * 0.15 - 0.45).collect();
            // Coarse offsets differ from 0.5 by at most 18*0.45*0.001=0.0081.
            // Including each FD perturbation, all sample coordinates stay well
            // within the same bilinear cell (distance to seam >0.48).
            let weights: Vec<_> = (0..desc.weight_count())
                .map(|i| (i % 3) as f32 * 0.001 - 0.001)
                .collect();
            let biases = vec![0.5; g.offset_dim()];
            let seeds: Vec<_> = (0..24).map(|i| (i % 5) as f32 * 0.3 - 0.6).collect();
            let upload = |data: &[f32]| backend.upload_f32(&[data.len()], data).unwrap();
            let x = upload(&input);
            let w = upload(&weights);
            let b = upload(&biases);
            let (_, cache) = backend.dcnv4_forward_native(&x, &w, &b, desc, cfg).unwrap();
            let grad = backend
                .dcnv4_backward_native(&x, &w, &cache, &upload(&seeds), desc, cfg)
                .unwrap();
            let dx = backend.download_f32(&grad.input).unwrap();
            let dw = backend.download_f32(&grad.offset_weights).unwrap();
            let db = backend.download_f32(&grad.offset_biases).unwrap();
            let objective = |x: &[f32], w: &[f32], b: &[f32]| -> f32 {
                let (y, _) = backend
                    .dcnv4_forward_native(&upload(x), &upload(w), &upload(b), desc, cfg)
                    .unwrap();
                backend
                    .download_f32(&y)
                    .unwrap()
                    .iter()
                    .zip(&seeds)
                    .map(|(v, s)| v * s)
                    .sum()
            };
            for (operand, values, gradient, step) in [
                (0, &input, &dx, 3),
                (1, &weights, &dw, 17),
                (2, &biases, &db, 1),
            ] {
                for i in (0..values.len()).step_by(step) {
                    let mut plus = values.clone();
                    let mut minus = values.clone();
                    plus[i] += eps;
                    minus[i] -= eps;
                    let (hi, lo) = match operand {
                        0 => (
                            objective(&plus, &weights, &biases),
                            objective(&minus, &weights, &biases),
                        ),
                        1 => (
                            objective(&input, &plus, &biases),
                            objective(&input, &minus, &biases),
                        ),
                        _ => (
                            objective(&input, &weights, &plus),
                            objective(&input, &weights, &minus),
                        ),
                    };
                    let fd = (hi - lo) / (2.0 * eps);
                    assert!(
                        (gradient[i] - fd).abs() < 2e-3,
                        "operand={operand} i={i} sm={softmax} rc={remove_center}: {} vs FD {fd}",
                        gradient[i]
                    );
                }
            }
        }
    }
}

#[test]
fn native_composition_rejects_bad_descriptors_and_storage() {
    let backend = OxideBackend::new().unwrap();
    let config = ElementwiseLaunchConfig::default();
    let (_, desc) = build(3, 2, 1, true, true, 2);
    let g = desc.geometry;
    let upload = |len| backend.upload_f32(&[len], &vec![0.0; len]).unwrap();
    let x_len = g.batch * g.channels * g.input_height * g.input_width;
    let x = upload(x_len);
    let w = upload(desc.weight_count());
    let b = upload(g.offset_dim());
    let offsets = upload(g.batch * g.spatial() * g.offset_dim());
    let seed = upload(g.batch * g.channels * g.spatial());
    for bad in [
        Dcnv4Composition {
            d_stride: 0,
            ..desc
        },
        Dcnv4Composition {
            d_stride: usize::MAX,
            ..desc
        },
        Dcnv4Composition {
            geometry: Dcnv4Geometry {
                size: 7,
                padding: 3,
                ..g
            },
            ..desc
        },
        Dcnv4Composition {
            offset_scale: f32::NAN,
            ..desc
        },
        Dcnv4Composition {
            geometry: Dcnv4Geometry {
                output_height: g.output_height + 1,
                ..g
            },
            ..desc
        },
        Dcnv4Composition {
            geometry: Dcnv4Geometry {
                padding: usize::MAX,
                ..g
            },
            ..desc
        },
    ] {
        assert!(
            backend
                .dcnv4_forward_native(&x, &w, &b, bad, config)
                .is_err()
        );
        assert!(
            backend
                .dcnv4_backward_native(&x, &w, &offsets, &seed, bad, config)
                .is_err()
        );
    }
    for (a, c, d) in [
        (x_len - 1, desc.weight_count(), g.offset_dim()),
        (x_len, desc.weight_count() - 1, g.offset_dim()),
        (x_len, desc.weight_count(), g.offset_dim() - 1),
    ] {
        assert!(
            backend
                .dcnv4_forward_native(&upload(a), &upload(c), &upload(d), desc, config)
                .is_err()
        );
    }
    assert!(
        backend
            .dcnv4_backward_native(&upload(x_len - 1), &w, &offsets, &seed, desc, config)
            .is_err()
    );
    assert!(
        backend
            .dcnv4_backward_native(
                &x,
                &upload(desc.weight_count() - 1),
                &offsets,
                &seed,
                desc,
                config
            )
            .is_err()
    );
    assert!(
        backend
            .dcnv4_backward_native(&x, &w, &upload(offsets.len() - 1), &seed, desc, config)
            .is_err()
    );
    assert!(
        backend
            .dcnv4_backward_native(&x, &w, &offsets, &upload(seed.len() - 1), desc, config)
            .is_err()
    );
}
