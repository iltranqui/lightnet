//! Native `[swin]` f32: per-kernel parity of the window layout and mHC kernels
//! against host references, and end-to-end `GpuExecutableNetwork<OxideBackend>`
//! against the Rust CPU reference for inference, the native training forward,
//! the native backward (input and staged parameter gradients) and the
//! following SGD update, plus the documented host fallback (`revleaky`).
//!
//! Tolerances are relative to `max(1, max |reference|)`: exact for the layout
//! gathers, 1e-5 for the mHC reductions, 2e-4 for outputs and 5e-4 for
//! gradients and updated parameters (f32 GEMM and reduction order differ from
//! the host loops). Hand-derived from the Rust CPU reference; no compiled-C++
//! golden.
use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, LayerTrainBuffers, Network,
    Tensor, TensorShape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

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

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

/// Reference of `pad_nchw` + `cyclic_shift_nchw(-shift)` + `window_partition_nchw`.
#[allow(clippy::too_many_arguments)]
fn ref_partition(
    x: &[f32], b: usize, c: usize, h: usize, w: usize, hp: usize, wp: usize, win: usize,
    shift: usize,
) -> Vec<f32> {
    let mut padded = vec![0.0; b * c * hp * wp];
    for bi in 0..b {
        for ch in 0..c {
            for y in 0..h {
                for xx in 0..w {
                    padded[((bi * c + ch) * hp + y) * wp + xx] = x[((bi * c + ch) * h + y) * w + xx];
                }
            }
        }
    }
    let (nwh, nww, t) = (hp / win, wp / win, win * win);
    let mut out = vec![0.0; b * nwh * nww * t * c];
    for bi in 0..b {
        for wh in 0..nwh {
            for ww in 0..nww {
                let wi = bi * nwh * nww + wh * nww + ww;
                for iy in 0..win {
                    for ix in 0..win {
                        let (y, xx) = (wh * win + iy, ww * win + ix);
                        let (sy, sx) = ((y + shift) % hp, (xx + shift) % wp);
                        for ch in 0..c {
                            out[(wi * t + iy * win + ix) * c + ch] =
                                padded[((bi * c + ch) * hp + sy) * wp + sx];
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn window_layout_kernels_match_reference_and_are_adjoint() {
    let be = OxideBackend::new().unwrap();
    let mut rng = Rng(17);
    for &(b, c, h, w, win, shift) in &[
        (1usize, 1usize, 4usize, 4usize, 2usize, 0usize),
        (2, 3, 3, 5, 2, 1),
        (2, 2, 7, 5, 3, 1),
        (1, 2, 2, 3, 1, 0),
        (1, 4, 6, 6, 4, 2),
    ] {
        let hp = h.div_ceil(win) * win;
        let wp = w.div_ceil(win) * win;
        let rows = b * (hp / win) * (wp / win) * win * win;
        let x = rng.vec(b * c * h * w);
        let xt = be.upload_f32(&[b, c, h, w], &x).unwrap();
        let windows = be
            .swin_window_partition_f32(&xt, b, c, h, w, hp, wp, win, shift, cfg())
            .unwrap();
        let got = be.download_f32(&windows).unwrap();
        close("partition", &got, &ref_partition(&x, b, c, h, w, hp, wp, win, shift), 0.0);

        // Unpartition inverts partition on the cropped extent.
        let back = be
            .swin_window_unpartition_f32(&windows, b, c, h, w, hp, wp, win, shift, 0.0, cfg())
            .unwrap();
        close("unpartition", &be.download_f32(&back).unwrap(), &x, 0.0);

        // Adjoint identity <P x, y> = <x, U y> with padded window slots in y.
        let y = rng.vec(rows * c);
        let yt = be.upload_f32(&[rows, c], &y).unwrap();
        let uy = be
            .download_f32(
                &be.swin_window_unpartition_f32(&yt, b, c, h, w, hp, wp, win, shift, 0.0, cfg())
                    .unwrap(),
            )
            .unwrap();
        let lhs: f32 = got.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.iter().zip(&uy).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() <= 1e-4 * lhs.abs().max(1.0), "adjoint {lhs} vs {rhs}");
    }

    // The output constrain maps NaN to -clip and clamps the rest.
    let (b, c, h, w, win) = (1usize, 1usize, 2usize, 2usize, 2usize);
    let raw = [250.0f32, f32::NAN, -3.0, f32::NEG_INFINITY];
    let rt = be.upload_f32(&[4, 1], &raw).unwrap();
    let got = be
        .download_f32(&be.swin_window_unpartition_f32(&rt, b, c, h, w, 2, 2, win, 0, 100.0, cfg()).unwrap())
        .unwrap();
    close("output constrain", &got, &[100.0, -100.0, -3.0, -100.0], 0.0);
    // A shift as large as the window is rejected.
    assert!(be.swin_window_partition_f32(&rt, 1, 1, 2, 2, 2, 2, 2, 2, cfg()).is_err());
}

#[test]
fn mhc_kernels_match_reference() {
    let be = OxideBackend::new().unwrap();
    let mut rng = Rng(29);
    let n = 150usize;
    let skip: Vec<f32> = rng.vec(n).iter().map(|v| v * 30.0).collect();
    let branch: Vec<f32> = rng.vec(n).iter().map(|v| v * 30.0).collect();
    let seed = rng.vec(n);
    let (a, bb, mix, ps, pb) = (0.7f32, 1.1f32, 0.3f32, 1.2f32, 0.6f32);
    let st = be.upload_f32(&[n], &skip).unwrap();
    let bt = be.upload_f32(&[n], &branch).unwrap();
    let et = be.upload_f32(&[n], &seed).unwrap();
    let got = be.download_f32(&be.swin_mhc_residual_f32(&st, &bt, a, bb, cfg()).unwrap()).unwrap();
    let want: Vec<f32> = skip.iter().zip(&branch).map(|(&s, &b)| a * s + bb * b).collect();
    close("mhc", &got, &want, 1e-6);

    let (ds, db, sums) = be
        .swin_mhc_residual_backward_f32(&st, &bt, &et, a, bb, mix, ps, pb, cfg())
        .unwrap();
    let mut want_sums = [0.0f32; 3];
    for i in 0..n {
        want_sums[0] += seed[i] * (mix * skip[i] + (1.0 - mix) * branch[i]);
        want_sums[1] += seed[i] * ((1.0 - mix) * skip[i] + mix * branch[i]);
        want_sums[2] += seed[i] * (ps - pb) * (skip[i] - branch[i]);
    }
    let want_ds: Vec<f32> = seed.iter().map(|g| g * a).collect();
    let want_db: Vec<f32> = seed.iter().map(|g| g * bb).collect();
    close("mhc d_skip", &be.download_f32(&ds).unwrap(), &want_ds, 0.0);
    close("mhc d_branch", &be.download_f32(&db).unwrap(), &want_db, 0.0);
    close("mhc sums", &be.download_f32(&sums).unwrap(), &want_sums, 1e-5);
}

fn network(extra: &str, width: usize, height: usize, channels: usize) -> ExecutableNetwork {
    let text = format!(
        "[net]\nbatch=2\nwidth={width}\nheight={height}\nchannels={channels}\n[swin]\n{extra}\n"
    );
    let cfg = ConfigFile::parse_str("native-swin.cfg", &text).unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::Swin(w) = &mut cpu.params[0] else { panic!("expected Swin weights") };
    let mut rng = Rng(0x5EED);
    let mut blocks = vec![
        &mut w.qkv_weight, &mut w.qkv_bias, &mut w.wo, &mut w.wo_bias, &mut w.ln1_gamma,
        &mut w.ln1_beta, &mut w.ln2_gamma, &mut w.ln2_beta, &mut w.ffn_w1, &mut w.ffn_b1,
        &mut w.ffn_w2, &mut w.ffn_b2, &mut w.relative_position_bias, &mut w.mhc,
    ];
    if let Some(res_proj) = w.res_proj.as_mut() {
        blocks.push(res_proj);
    }
    for v in blocks {
        for x in v.iter_mut() {
            *x = rng.next();
        }
    }
    for v in [&mut w.ln1_gamma, &mut w.ln2_gamma] {
        for x in v.iter_mut() {
            *x += 1.0;
        }
    }
    cpu
}

/// `(cfg body, width, height, channels)`: shifted + padded windows with a
/// residual projection, an identity skip, a 3x3 shifted window, and window 1.
const CASES: &[(&str, usize, usize, usize)] = &[
    ("filters=4\nsize=2\nheads=2\nshift=1\nffn_ratio=2\nactivation=gelu", 5, 3, 2),
    ("filters=4\nsize=2\nheads=2\nshift=0\nffn_ratio=2\nactivation=relu", 4, 4, 4),
    ("filters=6\nsize=3\nheads=3\nshift=1\nffn_ratio=1\nactivation=swish", 7, 5, 3),
    ("filters=2\nsize=1\nheads=1\nffn_ratio=2\nactivation=linear", 3, 2, 2),
];

fn input_for(width: usize, height: usize, channels: usize, seed: u64) -> Tensor {
    let shape = TensorShape::new(width, height, channels, 2);
    let mut rng = Rng(seed);
    Tensor::from_data(shape, rng.vec(shape.batch * shape.element_count())).unwrap()
}

#[test]
fn swin_native_forward_matches_cpu() {
    for &(extra, width, height, channels) in CASES {
        let cpu = network(extra, width, height, channels);
        let input = input_for(width, height, channels, 99);
        let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
        let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
        gpu.params = cpu.params.clone();
        let actual = gpu.forward(&input).unwrap().final_output().unwrap().clone();
        assert_eq!(actual.shape, expected.shape, "{extra}");
        close(extra, &actual.data, &expected.data, 2e-4);
        let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
        close("compact Swin", &compact.data, &expected.data, 2e-4);
    }
}

fn swin_updates(buffers: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::Swin(train) = buffers else { panic!("expected Swin train buffers") };
    train.updates.to_blob().unwrap()
}

fn swin_blob(params: &LayerParams) -> Vec<f32> {
    let LayerParams::Swin(w) = params else { panic!("expected Swin weights") };
    w.to_blob().unwrap()
}

/// Training forward, backward and the SGD step against the CPU.
fn train_check(extra: &str, width: usize, height: usize, channels: usize) {
    let mut cpu = network(extra, width, height, channels);
    let input = input_for(width, height, channels, 5);
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();

    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    let expected = cpu_forward.final_output().unwrap().clone();
    let actual = gpu_state.cpu_forward.final_output().unwrap().clone();
    close(&format!("{extra} forward"), &actual.data, &expected.data, 2e-4);

    let mut rng = Rng(7);
    let seed = Tensor::from_data(expected.shape, rng.vec(expected.len())).unwrap();
    let seeds = [Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let cpu_backward = cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &seeds)
        .unwrap()
        .expect("Swin input gradient");
    close(
        &format!("{extra} input gradient"),
        &gpu_input_gradient.data,
        &cpu_backward.input_gradient.data,
        5e-4,
    );
    let gpu_updates = swin_updates(&gpu.host_fallback_train_buffers()[0]);
    close(
        &format!("{extra} staged gradients"),
        &gpu_updates,
        &swin_updates(&cpu.train_buffers[0]),
        5e-4,
    );

    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    close(
        &format!("{extra} updated parameters"),
        &swin_blob(&gpu.params[0]),
        &swin_blob(&cpu.params[0]),
        5e-4,
    );
}

#[test]
fn swin_native_training_backward_and_update_match_cpu() {
    for &(extra, width, height, channels) in CASES {
        train_check(extra, width, height, channels);
    }
}

#[test]
fn swin_revleaky_keeps_the_host_fallback_and_matches_cpu() {
    // `revleaky` has no device activation kind: forward runs the host block,
    // backward the CPU fallback, so the same checks hold.
    train_check("filters=4\nsize=2\nheads=2\nshift=1\nffn_ratio=2\nactivation=revleaky", 5, 3, 2);
}
