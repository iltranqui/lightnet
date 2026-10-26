//! Native `[vit]` f32: per-kernel parity against host references, and
//! end-to-end `GpuExecutableNetwork<OxideBackend>` against the Rust CPU
//! reference for inference, the native training forward, the native backward
//! (input and staged parameter gradients) and the following SGD update.
//!
//! Tolerances are relative to `max(1, max |reference|)`: 2e-4 for outputs and
//! 5e-4 for gradients and updated parameters (f32 GEMM and reduction order
//! differ from the host loops; dense `flash_attention=0` also differs by the
//! CPU softmax's 1e-9 denominator epsilon).
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

fn clamp_feature(x: f32, clip: f32) -> f32 {
    if x.is_finite() { x.clamp(-clip, clip) } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
fn ref_patchify(
    x: &[f32], b: usize, c: usize, h: usize, w: usize, p: usize, s: usize, pad: usize,
) -> (usize, usize, Vec<f32>) {
    let oh = (h + 2 * pad - p) / s + 1;
    let ow = (w + 2 * pad - p) / s + 1;
    let pd = p * p * c;
    let mut out = vec![0.0; b * oh * ow * pd];
    for bi in 0..b {
        for ty in 0..oh {
            for tx in 0..ow {
                for dy in 0..p {
                    for dx in 0..p {
                        for ch in 0..c {
                            let y = ty * s + dy;
                            let xx = tx * s + dx;
                            if y < pad || y - pad >= h || xx < pad || xx - pad >= w {
                                continue;
                            }
                            out[((bi * oh * ow + ty * ow + tx) * pd) + (dy * p + dx) * c + ch] =
                                x[((bi * c + ch) * h + y - pad) * w + xx - pad];
                        }
                    }
                }
            }
        }
    }
    (oh, ow, out)
}

#[test]
fn patchify_matches_reference_for_pad_and_stride() {
    let be = OxideBackend::new().unwrap();
    let mut rng = Rng(7);
    for &(b, c, h, w, p, s, pad) in &[
        (1usize, 1usize, 4usize, 4usize, 1usize, 1usize, 0usize),
        (2, 3, 6, 8, 2, 2, 0),
        (2, 2, 5, 5, 3, 1, 1),
        (1, 3, 7, 7, 3, 2, 1),
        (2, 1, 4, 4, 2, 1, 0),
    ] {
        let x = rng.vec(b * c * h * w);
        let xt = be.upload_f32(&[b, c, h, w], &x).unwrap();
        let (oh, ow, got) = be.vit_patchify_f32(&xt, b, c, h, w, p, s, pad, cfg()).unwrap();
        let (roh, row, want) = ref_patchify(&x, b, c, h, w, p, s, pad);
        assert_eq!((oh, ow), (roh, row));
        close("patchify", &be.download_f32(&got).unwrap(), &want, 0.0);
    }
    let xt = be.upload_f32(&[1, 1, 4, 4], &[0.0; 16]).unwrap();
    assert!(be.vit_patchify_f32(&xt, 1, 1, 4, 4, 3, 2, 0, cfg()).is_err());
}

#[test]
fn elementwise_kernels_match_references() {
    let be = OxideBackend::new().unwrap();
    let mut rng = Rng(11);
    let (b, t, d) = (3usize, 5usize, 6usize);
    let tokens = rng.vec(b * t * d);
    let pos = rng.vec(t * d);
    let tt = be.upload_f32(&[b * t, d], &tokens).unwrap();
    let pt = be.upload_f32(&[t, d], &pos).unwrap();
    let got = be.download_f32(&be.vit_add_pos_embed_f32(&tt, &pt, cfg()).unwrap()).unwrap();
    let want: Vec<f32> = tokens.iter().enumerate().map(|(i, v)| v + pos[i % (t * d)]).collect();
    close("pos", &got, &want, 0.0);

    let mut x: Vec<f32> = rng.vec(64).iter().map(|v| v * 60.0).collect();
    x[3] = f32::NAN;
    x[9] = f32::INFINITY;
    x[10] = f32::NEG_INFINITY;
    let xt = be.upload_f32(&[64], &x).unwrap();
    let got = be.download_f32(&be.vit_clamp_feature_f32(&xt, 10.0, cfg()).unwrap()).unwrap();
    let want: Vec<f32> = x.iter().map(|&v| clamp_feature(v, 10.0)).collect();
    close("clamp", &got, &want, 0.0);

    let mut y: Vec<f32> = rng.vec(64).iter().map(|v| v * 40.0).collect();
    y[5] = f32::NAN;
    let yt = be.upload_f32(&[64], &y).unwrap();
    let got = be
        .download_f32(&be.vit_mhc_residual_f32(&xt, &yt, 0.8, 1.3, 10.0, cfg()).unwrap())
        .unwrap();
    let want: Vec<f32> = x
        .iter()
        .zip(&y)
        .map(|(&s, &br)| clamp_feature(0.8 * clamp_feature(s, 10.0) + 1.3 * clamp_feature(br, 10.0), 10.0))
        .collect();
    close("mhc", &got, &want, 1e-6);

    let (oh, ow) = (2usize, 3usize);
    let tok = rng.vec(b * oh * ow * d);
    let tkt = be.upload_f32(&[b * oh * ow, d], &tok).unwrap();
    let out = be.vit_tokens_to_nchw_f32(&tkt, b, oh, ow, d, cfg()).unwrap();
    let mut want = vec![0.0; tok.len()];
    for bi in 0..b {
        for tt in 0..oh * ow {
            for c in 0..d {
                want[(bi * d + c) * oh * ow + tt] = tok[(bi * oh * ow + tt) * d + c];
            }
        }
    }
    close("nchw", &be.download_f32(&out).unwrap(), &want, 0.0);
}

fn network(extra: &str, width: usize, height: usize) -> ExecutableNetwork {
    let text = format!(
        "[net]\nbatch=2\nwidth={width}\nheight={height}\nchannels=3\n[vit]\ndim=8\nheads=2\nmlp_dim=16\n{extra}\n"
    );
    let cfg = ConfigFile::parse_str("native-vit.cfg", &text).unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::Vit(w) = &mut cpu.params[0] else { panic!("expected Vit weights") };
    let mut rng = Rng(0xC0FFEE);
    for v in [
        &mut w.patch_embed, &mut w.patch_bias, &mut w.qkv_weight, &mut w.qkv_bias,
        &mut w.ln1_gamma, &mut w.ln1_beta, &mut w.wo, &mut w.wo_bias, &mut w.ln2_gamma,
        &mut w.ln2_beta, &mut w.ffn_w1, &mut w.ffn_b1, &mut w.ffn_w2, &mut w.ffn_b2,
        &mut w.pos_embed, &mut w.mhc,
    ] {
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

fn check(extra: &str, width: usize, height: usize) {
    let cpu = network(extra, width, height);
    let shape = TensorShape::new(width, height, 3, 2);
    let mut rng = Rng(99);
    let input = Tensor::from_data(shape, rng.vec(shape.batch * shape.element_count())).unwrap();
    let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let actual = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    assert_eq!(actual.shape, expected.shape, "{extra}");
    close(extra, &actual.data, &expected.data, 2e-4);
    let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
    close("compact ViT", &compact.data, &expected.data, 2e-4);
}

#[test]
fn vit_native_forward_matches_cpu() {
    check("patch_size=2\nflash_attention=1\nactivation=gelu", 4, 4);
    check("patch_size=2\nactivation=gelu", 4, 4);
    check("patch_size=2\npatch_stride=1\nactivation=leaky", 4, 4);
    check("patch_size=3\npatch_stride=2\npatch_pad=1\nflash_attention=1\nactivation=swish", 5, 5);
    check("patch_size=1\nflash_attention=1\nactivation=relu", 3, 4);
}

#[allow(clippy::too_many_arguments)]
fn ref_scatter(
    g: &[f32], b: usize, c: usize, h: usize, w: usize, p: usize, s: usize, pad: usize, oh: usize,
    ow: usize,
) -> Vec<f32> {
    let pd = p * p * c;
    let mut out = vec![0.0; b * c * h * w];
    for bi in 0..b {
        for ty in 0..oh {
            for tx in 0..ow {
                for dy in 0..p {
                    for dx in 0..p {
                        for ch in 0..c {
                            let y = ty * s + dy;
                            let xx = tx * s + dx;
                            if y < pad || y - pad >= h || xx < pad || xx - pad >= w {
                                continue;
                            }
                            out[((bi * c + ch) * h + y - pad) * w + xx - pad] +=
                                g[((bi * oh * ow + ty * ow + tx) * pd) + (dy * p + dx) * c + ch];
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn backward_kernels_match_references() {
    let be = OxideBackend::new().unwrap();
    let mut rng = Rng(23);

    let (b, oh, ow, d) = (2usize, 2usize, 3usize, 5usize);
    let x = rng.vec(b * d * oh * ow);
    let xt = be.upload_f32(&[b, d, oh, ow], &x).unwrap();
    let got = be
        .download_f32(&be.vit_nchw_to_tokens_f32(&xt, b, oh, ow, d, cfg()).unwrap())
        .unwrap();
    let mut want = vec![0.0; x.len()];
    for bi in 0..b {
        for t in 0..oh * ow {
            for c in 0..d {
                want[(bi * oh * ow + t) * d + c] = x[(bi * d + c) * oh * ow + t];
            }
        }
    }
    close("tokens", &got, &want, 0.0);

    for &(b, c, h, w, p, s, pad) in &[
        (1usize, 1usize, 4usize, 4usize, 1usize, 1usize, 0usize),
        (2, 3, 6, 8, 2, 2, 0),
        (2, 2, 5, 5, 3, 1, 1),
        (1, 3, 7, 7, 3, 2, 1),
        (2, 1, 4, 4, 2, 1, 0),
    ] {
        let (oh, ow, _) = ref_patchify(&vec![0.0; b * c * h * w], b, c, h, w, p, s, pad);
        let pg = rng.vec(b * oh * ow * p * p * c);
        let pt = be.upload_f32(&[b * oh * ow, p * p * c], &pg).unwrap();
        let got = be
            .download_f32(&be.vit_patch_scatter_f32(&pt, b, c, h, w, p, s, pad, oh, ow, cfg()).unwrap())
            .unwrap();
        close("scatter", &got, &ref_scatter(&pg, b, c, h, w, p, s, pad, oh, ow), 1e-6);
    }

    let mut raw: Vec<f32> = rng.vec(64).iter().map(|v| v * 40.0).collect();
    raw[2] = f32::NAN;
    raw[7] = f32::INFINITY;
    raw[8] = 10.0;
    let grad = rng.vec(64);
    let rt = be.upload_f32(&[64], &raw).unwrap();
    let gt = be.upload_f32(&[64], &grad).unwrap();
    let got = be
        .download_f32(&be.vit_clamp_backward_f32(&rt, &gt, 10.0, cfg()).unwrap())
        .unwrap();
    let want: Vec<f32> = raw
        .iter()
        .zip(&grad)
        .map(|(&r, &g)| if r.is_finite() && r.abs() < 10.0 { g } else { 0.0 })
        .collect();
    close("clamp backward", &got, &want, 0.0);

    // mHC adjoint against the `mhc_residual_backward` formulas.
    let mut skip: Vec<f32> = rng.vec(96).iter().map(|v| v * 30.0).collect();
    let branch: Vec<f32> = rng.vec(96).iter().map(|v| v * 30.0).collect();
    let mut seed = rng.vec(96);
    skip[4] = f32::NAN;
    seed[9] = f32::INFINITY;
    let (a, bb, mix, ps, pb) = (0.7f32, 1.1f32, 0.3f32, 1.2f32, 0.6f32);
    let st = be.upload_f32(&[96], &skip).unwrap();
    let bt = be.upload_f32(&[96], &branch).unwrap();
    let et = be.upload_f32(&[96], &seed).unwrap();
    let (ds, db, sums) = be
        .vit_mhc_residual_backward_f32(&st, &bt, &et, a, bb, mix, ps, pb, 10.0, cfg())
        .unwrap();
    let (mut want_ds, mut want_db, mut want_sums) = (vec![0.0; 96], vec![0.0; 96], [0.0f32; 3]);
    for i in 0..96 {
        let s = clamp_feature(skip[i], 10.0);
        let br = clamp_feature(branch[i], 10.0);
        let mixed = a * s + bb * br;
        if !mixed.is_finite() || mixed.abs() >= 10.0 || !seed[i].is_finite() {
            continue;
        }
        if skip[i].is_finite() && skip[i].abs() < 10.0 {
            want_ds[i] = seed[i] * a;
        }
        if branch[i].is_finite() && branch[i].abs() < 10.0 {
            want_db[i] = seed[i] * bb;
        }
        want_sums[0] += seed[i] * (mix * s + (1.0 - mix) * br);
        want_sums[1] += seed[i] * ((1.0 - mix) * s + mix * br);
        want_sums[2] += seed[i] * (ps - pb) * (s - br);
    }
    close("mhc d_skip", &be.download_f32(&ds).unwrap(), &want_ds, 0.0);
    close("mhc d_branch", &be.download_f32(&db).unwrap(), &want_db, 0.0);
    close("mhc sums", &be.download_f32(&sums).unwrap(), &want_sums, 1e-5);

    let (batch, table) = (3usize, 10usize);
    let g = rng.vec(batch * table);
    let tg = be.upload_f32(&[batch * table], &g).unwrap();
    let got = be
        .download_f32(&be.vit_pos_embed_backward_f32(&tg, batch, table, cfg()).unwrap())
        .unwrap();
    let want: Vec<f32> = (0..table)
        .map(|i| (0..batch).map(|bi| g[bi * table + i]).sum())
        .collect();
    close("pos backward", &got, &want, 1e-6);
}

fn vit_updates(buffers: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::Vit(train) = buffers else { panic!("expected Vit train buffers") };
    train.updates.to_blob()
}

fn vit_blob(params: &LayerParams) -> Vec<f32> {
    let LayerParams::Vit(w) = params else { panic!("expected Vit weights") };
    w.to_blob()
}

/// Native training forward, native backward and the SGD step against the CPU.
fn train_check(extra: &str, width: usize, height: usize) {
    let mut cpu = network(extra, width, height);
    let shape = TensorShape::new(width, height, 3, 2);
    let mut rng = Rng(5);
    let input = Tensor::from_data(shape, rng.vec(shape.batch * shape.element_count())).unwrap();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();

    let cpu_forward = cpu.forward_training(&input).unwrap();
    let gpu_state = gpu.forward_training_with_state(&input).unwrap();
    let expected = cpu_forward.final_output().unwrap().clone();
    let actual = gpu_state.cpu_forward.final_output().unwrap().clone();
    close(&format!("{extra} forward"), &actual.data, &expected.data, 2e-4);

    let seed = Tensor::from_data(expected.shape, rng.vec(expected.len())).unwrap();
    let seeds = [Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let cpu_backward = cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
    let gpu_input_gradient = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&gpu_state, &seeds)
        .unwrap()
        .expect("native ViT input gradient");
    close(
        &format!("{extra} input gradient"),
        &gpu_input_gradient.data,
        &cpu_backward.input_gradient.data,
        5e-4,
    );
    gpu.synchronize_resident_parameter_gradients().unwrap();
    let gpu_updates = vit_updates(&gpu.host_fallback_train_buffers()[0]);
    close(
        &format!("{extra} staged gradients"),
        &gpu_updates,
        &vit_updates(&cpu.train_buffers[0]),
        5e-4,
    );

    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    close(
        &format!("{extra} updated parameters"),
        &vit_blob(&gpu.params[0]),
        &vit_blob(&cpu.params[0]),
        5e-4,
    );
}

#[test]
fn vit_native_training_backward_and_update_match_cpu() {
    train_check("patch_size=2\nflash_attention=1\nactivation=gelu", 4, 4);
    train_check("patch_size=2\nactivation=gelu", 4, 4);
    train_check("patch_size=2\npatch_stride=1\nactivation=leaky\npos_embed=sinusoidal", 4, 4);
    train_check("patch_size=3\npatch_stride=2\npatch_pad=1\nflash_attention=1\nactivation=swish", 5, 5);
    train_check("patch_size=1\nflash_attention=1\nactivation=relu", 3, 4);
}

#[test]
fn vit_gpu_muon_hybrid_backward_and_adamw_update_match_cpu() {
    let mut cpu = network("patch_size=1\nflash_attention=1", 3, 2);
    cpu.network.net.muon = true;
    cpu.network.net.adam = false;
    cpu.network.net.decoupled_decay = true;
    cpu.network.net.decay = 0.02;
    cpu.network.net.muon_momentum = 0.8;
    let shape = cpu.network.net.input_shape();
    let mut rng = Rng(941);
    let input = Tensor::from_data(shape, rng.vec(shape.batch * shape.element_count())).unwrap();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    gpu.init_train_buffers_gpu().unwrap();
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();

    let cpu_forward = cpu.forward_training(&input).unwrap();
    let device_forward = gpu.forward_training_device(&input).unwrap();
    let expected = cpu_forward.final_output().unwrap();
    let actual = gpu.backend.download_f32(&device_forward.gpu_layer_outputs[0]).unwrap();
    close("ViT Muon native forward", &actual, &expected.data, 2e-4);

    let seed = Tensor::from_data(expected.shape, rng.vec(expected.len())).unwrap();
    let seeds = [Some(seed)];
    let cpu_backward = cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
    let gpu_input = gpu
        .backward_accumulate_seeded_gpu_with_input_gradient(&device_forward, &seeds)
        .unwrap()
        .expect("native ViT Muon input gradient");
    close("ViT Muon input gradient", &gpu_input.data, &cpu_backward.input_gradient.data, 5e-4);
    close(
        "ViT Muon staged gradients",
        &vit_updates(&gpu.host_fallback_train_buffers()[0]),
        &vit_updates(&cpu.train_buffers[0]),
        5e-4,
    );

    cpu.update_weights_muon(0.003, 1).unwrap();
    gpu.update_weights_gpu(0.003, 0.9, 0.02, 1).unwrap();
    close(
        "ViT Muon and AdamW parameters",
        &vit_blob(&gpu.params[0]),
        &vit_blob(&cpu.params[0]),
        5e-4,
    );
}

#[test]
fn vit_resident_clipping_preserves_frozen_accumulators_and_sanitizes_active_values() {
    let be=OxideBackend::new().unwrap();
    let input=be.upload_f32(&[7],&[f32::NAN,f32::INFINITY,f32::NEG_INFINITY,100.0,3.0,4.0,f32::NAN]).unwrap();
    let actual=be.download_f32(&be.vit_prepare_updates_f32(&input,6,7,5.0,cfg()).unwrap()).unwrap();
    let scale=5.0/(10025.0f64.sqrt() as f32);
    close("resident norm clip",&actual[..6],&[0.0,0.0,0.0,100.0*scale,3.0*scale,4.0*scale],1e-6);
    assert!(actual[6].is_nan());
}

#[test]
fn vit_resident_deferred_sgd_adam_and_adamw_match_cpu_and_resume() {
    for (adam,decoupled,sinusoidal) in [(false,false,false),(true,false,true),(true,true,false)] {
        let mut cpu=network(if sinusoidal {"patch_size=1\nflash_attention=1\npos_embed=sinusoidal"}
            else {"patch_size=1\nflash_attention=1"},3,2);
        cpu.network.net.adam=adam; cpu.network.net.decoupled_decay=decoupled;
        cpu.network.net.decay=0.03;
        let shape=cpu.network.net.input_shape(); let mut rng=Rng(781);
        let input=Tensor::from_data(shape,rng.vec(shape.batch*shape.element_count())).unwrap();
        let mut gpu=GpuExecutableNetwork::new(cpu.network.clone(),OxideBackend::new().unwrap());
        gpu.params=cpu.params.clone(); gpu.init_train_buffers_gpu().unwrap();
        cpu.zero_train_buffers(); gpu.zero_train_buffers_gpu().unwrap();
        for step in 0..3 {
            let host_snapshot=vit_blob(&gpu.params[0]);
            let host=cpu.forward_training(&input).unwrap();
            let device=gpu.forward_training_device(&input).unwrap();
            let expected=host.final_output().unwrap();
            close("resident forward",&gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap(),&expected.data,2e-4);
            let compact=gpu.forward_compact(&input).unwrap().final_output.unwrap();
            close("resident compact",&compact.data,&expected.data,2e-4);
            // The half path rounds activations at its storage boundaries, then
            // uses the same native f32 ViT kernels and resident master bank.
            let half_input = gpu.backend.upload_f16(&[shape.batch, shape.channels, shape.height, shape.width], &input.data).unwrap();
            let rounded_input = Tensor::from_data(shape, gpu.backend.download_f16(&half_input).unwrap()).unwrap();
            let f32_rounded_reference = gpu.forward_compact(&rounded_input).unwrap().final_output.unwrap();
            let rounded_output = gpu.backend.download_f16(
                &gpu.backend.upload_f16(&[
                    f32_rounded_reference.shape.batch,
                    f32_rounded_reference.shape.channels,
                    f32_rounded_reference.shape.height,
                    f32_rounded_reference.shape.width,
                ], &f32_rounded_reference.data).unwrap()
            ).unwrap();
            let half_summary = gpu.forward_compact_f16(&input).unwrap();
            let half_output = half_summary.final_output.unwrap();
            close("resident f16 compact storage", &half_output.data, &rounded_output, 0.0);
            let seeds=[Some(Tensor::from_data(expected.shape,rng.vec(expected.len()).into_iter().map(|v|v*200.0).collect()).unwrap())];
            for _ in 0..2 {
                let reference=cpu.backward_accumulate_seeded(&input,&host,&seeds).unwrap();
                let actual=gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device,&seeds).unwrap().unwrap();
                close("resident input adjoint",&actual.data,&reference.input_gradient.data,5e-4);
            }
            gpu.synchronize_resident_parameter_gradients().unwrap();
            close("resident accumulated gradients",&vit_updates(&gpu.host_fallback_train_buffers()[0]),
                &vit_updates(&cpu.train_buffers[0]),5e-4);
            if adam {cpu.update_weights_adam(0.003,2).unwrap();}
            else {cpu.update_weights(0.003,0.9,0.03,2);}
            gpu.update_weights_gpu_deferred(0.003,0.9,0.03,2).unwrap();
            assert_eq!(vit_blob(&gpu.params[0]),host_snapshot,"deferred host mirror changed");
            assert!(gpu.backward_accumulate_seeded_gpu(&device,&seeds).is_err(),"stale ViT tape accepted");
            let actual=gpu.forward_compact(&input).unwrap().final_output.unwrap();
            close("updated resident inference",&actual.data,&cpu.forward(&input).unwrap().final_output().unwrap().data,5e-4);
            let half_after_update = gpu.forward_compact_f16(&input).unwrap().final_output.unwrap();
            let current_f32 = gpu.forward_compact(&rounded_input).unwrap().final_output.unwrap();
            let current_half = gpu.backend.download_f16(
                &gpu.backend.upload_f16(&[
                    current_f32.shape.batch,
                    current_f32.shape.channels,
                    current_f32.shape.height,
                    current_f32.shape.width,
                ], &current_f32.data).unwrap()
            ).unwrap();
            close("updated resident f16 compact storage", &half_after_update.data, &current_half, 0.0);
            if adam && step==1 {
                let path=std::env::temp_dir().join(format!("lightnet-vit-resident-{}-{decoupled}.dnopt",std::process::id()));
                gpu.save_optimizer_state(&path).unwrap();
                let mut restored=GpuExecutableNetwork::new(cpu.network.clone(),OxideBackend::new().unwrap());
                restored.params=cpu.params.clone(); restored.init_train_buffers_gpu().unwrap();
                restored.load_optimizer_state(&path).unwrap(); std::fs::remove_file(path).unwrap();
                gpu=restored;
            }
        }
        gpu.synchronize_resident_parameter_state().unwrap();
        close("resident final parameters",&vit_blob(&gpu.params[0]),&vit_blob(&cpu.params[0]),5e-4);
    }
}
