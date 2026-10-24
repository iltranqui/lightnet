//! Native `[vit]` f32 inference: per-kernel parity against host references and
//! end-to-end `GpuExecutableNetwork<OxideBackend>` against the Rust CPU forward.
use lightnet_core::{
    ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor, TensorShape,
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
}

#[test]
fn vit_native_forward_matches_cpu() {
    check("patch_size=2\nflash_attention=1\nactivation=gelu", 4, 4);
    check("patch_size=2\nactivation=gelu", 4, 4);
    check("patch_size=2\npatch_stride=1\nactivation=leaky", 4, 4);
    check("patch_size=3\npatch_stride=2\npatch_pad=1\nflash_attention=1\nactivation=swish", 5, 5);
    check("patch_size=1\nflash_attention=1\nactivation=relu", 3, 4);
}
