//! Rust CPU versus native f32 Clifford inference, training forward, input and
//! parameter gradients, BN running statistics, and one host-staged SGD/Adam
//! update. Fourier explicitly exercises the reported host fallback.
//!
//! Tolerances use max(1, max |reference|): 2e-4 forward, 8e-4 backward,
//! statistics and updates; primitive products/adjoints use 2e-5. Reduction
//! and GEMM ordering differ. No compiled-C++ golden; nothing here claims it.
use lightnet_core::{
    CliffordParams, ConfigFile, ExecutableNetwork, GpuExecutableNetwork,
    LayerParams, LayerTrainBuffers, Network, Tensor, TensorShape,
};
use lightnet_core::clifford::CliffordBranchParams;
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
    }
    fn values(&mut self, len: usize) -> Vec<f32> { (0..len).map(|_| self.next()).collect() }
    fn fill(&mut self, values: &mut [f32], center: f32) {
        for v in values { *v = center + 0.3 * self.next(); }
    }
}

fn close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1.0f32, |s, v| s.max(v.abs()));
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tolerance * scale, "{name}[{i}]: {g} != {w}");
    }
}

fn fill_branch(rng: &mut Rng, branch: &mut CliffordBranchParams) {
    for values in [&mut branch.projection_weights, &mut branch.projection_biases,
        &mut branch.gate_weights, &mut branch.gate_biases] { rng.fill(values, 0.0); }
}

fn network(options: &str, width: usize, height: usize, channels: usize, adam: bool) -> ExecutableNetwork {
    let cfg = ConfigFile::parse_str("oxide-clifford.cfg", &format!(
        "[net]\nbatch=2\nwidth={width}\nheight={height}\nchannels={channels}\nadam={}\neps=0.001\n[clifford]\nfilters={channels}\n{options}\n",
        usize::from(adam))).unwrap();
    let mut cpu = ExecutableNetwork::new(Network::from_config(cfg).unwrap());
    let LayerParams::Clifford(p) = &mut cpu.params[0] else { panic!("Clifford weights") };
    let mut rng = Rng(17);
    rng.fill(&mut p.layer_norm_gamma, 1.0);
    rng.fill(&mut p.layer_norm_beta, 0.0);
    rng.fill(&mut p.layer_scale, 0.3);
    rng.fill(&mut p.determinant_weights, 0.0);
    rng.fill(&mut p.determinant_biases, 0.0);
    fill_branch(&mut rng, &mut p.local);
    for conv in &mut p.depthwise {
        rng.fill(&mut conv.weights, 0.0);
        rng.fill(&mut conv.biases, 0.0);
        let bn = conv.batch_norm.as_mut().unwrap();
        rng.fill(&mut bn.scales, 1.0);
        // Deliberately distinct from conv.biases: CPU uses the latter.
        rng.fill(&mut bn.biases, 0.4);
        rng.fill(&mut bn.rolling_mean, 0.0);
        rng.fill(&mut bn.rolling_variance, 1.0);
    }
    if let Some(branch) = &mut p.global { fill_branch(&mut rng, branch); }
    if let Some(fourier) = &mut p.fourier {
        rng.fill(&mut fourier.real, 0.2);
        rng.fill(&mut fourier.imaginary, 0.0);
    }
    cpu
}

fn branch_values(out: &mut Vec<f32>, branch: &CliffordBranchParams) {
    for v in [&branch.projection_weights, &branch.projection_biases,
        &branch.gate_weights, &branch.gate_biases] { out.extend_from_slice(v); }
}

fn parameter_values(params: &LayerParams) -> Vec<f32> {
    let LayerParams::Clifford(p) = params else { panic!("Clifford parameters") };
    values(p)
}

fn values(p: &CliffordParams) -> Vec<f32> {
    let mut out = Vec::new();
    for v in [&p.layer_norm_gamma, &p.layer_norm_beta, &p.determinant_weights,
        &p.determinant_biases, &p.layer_scale] { out.extend_from_slice(v); }
    for conv in &p.depthwise {
        out.extend_from_slice(&conv.weights);
        out.extend_from_slice(&conv.biases);
        if let Some(bn) = &conv.batch_norm {
            for v in [&bn.scales, &bn.biases, &bn.rolling_mean, &bn.rolling_variance] { out.extend_from_slice(v); }
        }
    }
    branch_values(&mut out, &p.local);
    if let Some(branch) = &p.global { branch_values(&mut out, branch); }
    if let Some(fourier) = &p.fourier {
        out.extend_from_slice(&fourier.real);
        out.extend_from_slice(&fourier.imaginary);
    }
    out
}

fn update_values(buffers: &LayerTrainBuffers) -> Vec<f32> {
    let LayerTrainBuffers::Clifford(t) = buffers else { panic!("Clifford train buffers") };
    let mut out = Vec::new();
    for v in [&t.layer_norm_gamma_updates, &t.layer_norm_beta_updates,
        &t.determinant_weight_updates, &t.determinant_bias_updates, &t.layer_scale_updates,
        &t.local.projection_weight_updates, &t.local.projection_bias_updates,
        &t.local.gate_weight_updates, &t.local.gate_bias_updates] { out.extend_from_slice(v); }
    for conv in &t.depthwise {
        out.extend_from_slice(&conv.weight_updates);
        out.extend_from_slice(&conv.bias_updates);
        if let Some(bn) = &conv.batch_norm {
            for v in [&bn.scale_updates, &bn.mean_delta, &bn.variance_delta] { out.extend_from_slice(v); }
        }
    }
    if let Some(b) = &t.global {
        for v in [&b.projection_weight_updates, &b.projection_bias_updates,
            &b.gate_weight_updates, &b.gate_bias_updates] { out.extend_from_slice(v); }
    }
    if let Some(f) = &t.fourier {
        out.extend_from_slice(&f.real_updates);
        out.extend_from_slice(&f.imaginary_updates);
    }
    out
}

fn train_check(options: &str, w: usize, h: usize, c: usize, adam: bool, fallback: bool) {
    let mut cpu = network(options, w, h, c, adam);
    let shape = TensorShape::new(w, h, c, 2);
    let input = Tensor::from_data(shape, Rng(99).values(2 * w * h * c)).unwrap();
    let mut gpu = GpuExecutableNetwork::new(cpu.network.clone(), OxideBackend::new().unwrap());
    gpu.params = cpu.params.clone();
    let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
    let actual = gpu.forward(&input).unwrap().final_output().unwrap().clone();
    close("inference", &actual.data, &expected.data, 2e-4);
    let compact = gpu.forward_compact(&input).unwrap().final_output.unwrap();
    close("compact inference", &compact.data, &expected.data, 2e-4);

    gpu.init_train_buffers_gpu().unwrap();
    let forward = cpu.forward_training(&input).unwrap();
    let device = gpu.forward_training_device(&input).unwrap();
    // Shape-only host outputs prove that the supported variants selected the
    // native backward rather than the network-wide CPU fallback.
    assert_eq!(device.cpu_forward.layer_outputs[0].data.is_empty(), !fallback);
    let actual = gpu.backend.download_f32(&device.gpu_layer_outputs[0]).unwrap();
    close("training forward", &actual, &forward.final_output().unwrap().data, 2e-4);
    let seed = Tensor::from_data(shape, Rng(71).values(input.len())).unwrap();
    let seeds = [Some(seed)];
    cpu.zero_train_buffers();
    gpu.zero_train_buffers_gpu().unwrap();
    let want = cpu.backward_accumulate_seeded(&input, &forward, &seeds).unwrap();
    let got = gpu.backward_accumulate_seeded_gpu_with_input_gradient(&device, &seeds).unwrap().unwrap();
    close("input gradient", &got.data, &want.input_gradient.data, 8e-4);
    close("parameter gradients", &update_values(&gpu.host_fallback_train_buffers()[0]),
        &update_values(&cpu.train_buffers[0]), 8e-4);
    close("BN running statistics", &parameter_values(&gpu.params[0]), &parameter_values(&cpu.params[0]), 8e-4);
    let before = parameter_values(&cpu.params[0]);
    cpu.update_weights(0.01, 0.9, 0.0005, 2);
    gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
    assert_ne!(before, parameter_values(&cpu.params[0]));
    close("updated parameters", &parameter_values(&gpu.params[0]), &parameter_values(&cpu.params[0]), 8e-4);
}

#[test]
fn clifford_native_forward_backward_and_host_staged_update_match_cpu() {
    for &(options, w, h, c) in &[
        ("gffn_mode=local\ncli_mode=full\nctx_mode=abs\ncli_higher_mode=off\ndwconv_size=3", 3, 2, 3),
        ("gffn_mode=local\ncli_mode=inner\nctx_mode=diff\nnum_dwconv=2\ndwconv_size=1", 3, 2, 3),
        ("gffn_mode=global\ncli_mode=wedge\nshifts=1,0\nglobal_op=avgpool", 3, 2, 3),
        ("gffn_mode=hybrid\ncli_mode=full\nctx_mode=diff\ncli_higher_mode=vb\nnum_dwconv=2\ndrop_path=0.25", 3, 2, 3),
        ("gffn_mode=hybrid\ncli_mode=full\nctx_mode=abs\ncli_higher_mode=vb\nshifts_wedge=1,1,0\nshifts_inner=2", 3, 2, 3),
        ("gffn_mode=hybrid\ncli_mode=full\nshifts_wedge=1\nshifts_inner=2,0,2\ndwconv_size=1", 1, 1, 3),
        ("gffn_mode=hybrid\ncli_mode=full\ncli_higher_mode=vb\ndwconv_size=1", 3, 1, 1),
    ] {
        train_check(options, w, h, c, false, false);
    }
}

#[test]
fn clifford_native_adam_update_matches_cpu() {
    train_check("gffn_mode=hybrid\ncli_mode=full\nctx_mode=diff\ncli_higher_mode=vb\ndwconv_size=1", 3, 2, 3, true, false);
}

#[test]
fn clifford_fourier_keeps_reported_host_fallback() {
    train_check("gffn_mode=hybrid\ncli_mode=full\nglobal_op=fourier\nfno_modes_h=1\nfno_modes_w=2\ndwconv_size=1", 3, 2, 3, false, true);
}

#[test]
fn clifford_interaction_kernels_match_cpu_products_and_adjoints() {
    let be = OxideBackend::new().unwrap();
    let cfg = ElementwiseLaunchConfig::default();
    let (batches, channels, spatial) = (2, 3, 5);
    let n = batches * channels * spatial;
    let x = Rng(17).values(n);
    let g = Rng(21).values(n);
    let xt = be.upload_f32(&[batches, channels, 1, spatial], &x).unwrap();
    let gt = be.upload_f32(&[batches, channels, 1, spatial], &g).unwrap();
    for pooled in [false, true] {
        let ctx = Rng(19).values(if pooled { batches * channels } else { n });
        let ct = be.upload_f32(&[ctx.len()], &ctx).unwrap();
        for inner in [false, true] {
            for shift in [0, 1, 7] {
                let mut want = vec![0.0; n];
                let mut dx = vec![0.0; n];
                let mut dc = vec![0.0; ctx.len()];
                for b in 0..batches {
                    for c in 0..channels {
                        let r = (c + shift) % channels;
                        for p in 0..spatial {
                            let i = (b * channels + c) * spatial + p;
                            let j = (b * channels + r) * spatial + p;
                            let ci = if pooled { b * channels + c } else { i };
                            let cj = if pooled { b * channels + r } else { j };
                            let product = x[i] * ctx[cj];
                            if inner {
                                let s = 1.0 / (1.0 + (-product).exp());
                                want[i] = product * s;
                                let d = g[i] * (s + product * s * (1.0 - s));
                                dx[i] += d * ctx[cj];
                                dc[cj] += d * x[i];
                            } else {
                                want[i] = product - ctx[ci] * x[j];
                                dx[i] += g[i] * ctx[cj];
                                dx[j] -= g[i] * ctx[ci];
                                dc[cj] += g[i] * x[i];
                                dc[ci] -= g[i] * x[j];
                            }
                        }
                    }
                }
                let out = be.clifford_interaction_f32(&xt, &ct, batches, channels, spatial, shift, inner, pooled, cfg).unwrap();
                let (ds, dctx) = be.clifford_interaction_backward_f32(&xt, &ct, &gt, batches, channels, spatial, shift, inner, pooled, cfg).unwrap();
                close("rolling product", &be.download_f32(&out).unwrap(), &want, 2e-5);
                close("rolling state gradient", &be.download_f32(&ds).unwrap(), &dx, 2e-5);
                close("rolling context gradient", &be.download_f32(&dctx).unwrap(), &dc, 2e-5);
            }
        }
    }
}

#[test]
fn clifford_residual_drops_whole_samples_and_differentiates_scale() {
    let be = OxideBackend::new().unwrap();
    let cfg = ElementwiseLaunchConfig::default();
    let (b, c, spatial) = (2, 3, 5);
    let n = b * c * spatial;
    let x = Rng(3).values(n);
    let mixed = Rng(5).values(n);
    let grad = Rng(7).values(n);
    let scale = [0.2, -0.3, 0.4];
    let mask = [0.0, 1.0 / 0.75];
    let xt = be.upload_f32(&[b, c, 1, spatial], &x).unwrap();
    let mt = be.upload_f32(&[b, c, 1, spatial], &mixed).unwrap();
    let gt = be.upload_f32(&[b, c, 1, spatial], &grad).unwrap();
    let st = be.upload_f32(&[c], &scale).unwrap();
    let maskt = be.upload_f32(&[b], &mask).unwrap();
    let out = be.clifford_residual_f32(&xt, &mt, &st, &maskt, b, c, spatial, cfg).unwrap();
    let (dm, ds) = be.clifford_residual_backward_f32(&mt, &gt, &st, &maskt, b, c, spatial, cfg).unwrap();
    let mut want = x.clone();
    let mut want_dm = vec![0.0; n];
    let mut want_ds = vec![0.0; c];
    for i in 0..n {
        let ch = (i / spatial) % c;
        let sample = i / (c * spatial);
        want[i] += mixed[i] * (scale[ch] * mask[sample]);
        want_dm[i] = grad[i] * mask[sample] * scale[ch];
        want_ds[ch] += grad[i] * mask[sample] * mixed[i];
    }
    close("residual", &be.download_f32(&out).unwrap(), &want, 2e-5);
    close("residual d_mixed", &be.download_f32(&dm).unwrap(), &want_dm, 2e-5);
    close("residual d_scale", &be.download_f32(&ds).unwrap(), &want_ds, 2e-5);
}
