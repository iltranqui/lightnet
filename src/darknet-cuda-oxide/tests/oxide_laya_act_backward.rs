//! Standalone native act-head gradient math; no runtime update integration.
use darknet_core::laya_act_head::{LayaActHeadWeights, act_head_backward};
use darknet_cuda::host_ops::laya_act::MlpLayout;
use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, LayaActBackwardInputs};
use darknet_cuda_oxide::{GemmMode, OxideBackend, OxideTensorF32};

const B: usize = 2;
const H: usize = 3;
const K: usize = 4;
const CB: usize = 10;
const TH: usize = 5;
const TM: usize = 3;
const F: usize = 5;
const A: usize = 3;
const HS: [usize; 4] = [B, H, 1, TH];
const SS: [usize; 4] = [B, 1, 1, K];
const MS: [usize; 4] = [B, CB, 1, TM];
fn backend() -> OxideBackend {
    OxideBackend::new().unwrap().with_gemm_mode(GemmMode::Fp32)
}
fn layout() -> MlpLayout {
    MlpLayout::new(&HS, &SS, &MS, F, A).unwrap()
}
fn close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < tol * (1.0 + w.abs()), "{i}: {g} vs {w}");
    }
}
fn fixture(masked: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, LayaActHeadWeights, Vec<f32>) {
    let head = (0..B * H * TH).map(|i| i as f32 * 0.02 - 0.2).collect();
    let logits = vec![0.2, -0.5, 1.1, 0.6, -0.7, 1.3, 0.4, -0.2];
    let mut marker = vec![0.0; B * CB * TM];
    if !masked {
        for b in 0..B {
            for j in 0..K {
                marker[(b * CB + 2 * j + 1) * TM] = if (b + j) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
    }
    let w = LayaActHeadWeights {
        fc1_w: (0..F * (H + 4))
            .map(|i| ((i % 11) as f32 - 5.0) * 0.04)
            .collect(),
        fc1_b: vec![-1.0, 0.0, 1.0, -0.5, 0.5],
        fc2_w: (0..A * F).map(|i| ((i % 7) as f32 - 3.0) * 0.07).collect(),
        fc2_b: vec![0.1, -0.2, 0.3],
    };
    (
        head,
        logits,
        marker,
        w,
        vec![1.0, -0.25, 0.0, -0.5, 0.75, -1.5],
    )
}
// The existing CPU full-MLP helper requires exactly 2*K marker channels,
// while the native feature math accepts extra ignored channels. Compact each
// sample's unused tail for the oracle without changing any read mask value.
fn cpu_grad(
    h: &[f32],
    s: &[f32],
    m: &[f32],
    w: &LayaActHeadWeights,
    dy: &[f32],
) -> darknet_core::laya_act_head::LayaActHeadGrad {
    let marker: Vec<_> = (0..B)
        .flat_map(|b| m[b * CB * TM..b * CB * TM + 2 * K * TM].iter().copied())
        .collect();
    act_head_backward(h, s, &marker, w, dy, B, H, K, 2 * K, TH, TM, F, A).unwrap()
}

struct Tensors {
    h: OxideTensorF32,
    s: OxideTensorF32,
    m: OxideTensorF32,
    w1: OxideTensorF32,
    b1: OxideTensorF32,
    w2: OxideTensorF32,
    dy: OxideTensorF32,
}
impl Tensors {
    fn new(
        be: &OxideBackend,
        h: &[f32],
        s: &[f32],
        m: &[f32],
        w: &LayaActHeadWeights,
        dy: &[f32],
        nchw: bool,
    ) -> Self {
        Self {
            h: be.upload_f32(&HS, h).unwrap(),
            s: be.upload_f32(&SS, s).unwrap(),
            m: be.upload_f32(&MS, m).unwrap(),
            w1: be.upload_f32(&[F, H + 4], &w.fc1_w).unwrap(),
            b1: be.upload_f32(&[F], &w.fc1_b).unwrap(),
            w2: be.upload_f32(&[A, F], &w.fc2_w).unwrap(),
            dy: be
                .upload_f32(if nchw { &[B, A, 1, 1] } else { &[B, A] }, dy)
                .unwrap(),
        }
    }
    fn inputs(&self) -> LayaActBackwardInputs<'_, OxideTensorF32> {
        LayaActBackwardInputs {
            head: &self.h,
            scorer: &self.s,
            markers: &self.m,
            fc1_w: &self.w1,
            fc1_b: &self.b1,
            fc2_w: &self.w2,
            seeds: &self.dy,
        }
    }
}

#[test]
fn laya_act_backward_native_all_blocks_and_cls_match_cpu_sums_without_updates() {
    let be = backend();
    for masked in [false, true] {
        let (h, s, m, w, dy) = fixture(masked);
        let t = Tensors::new(&be, &h, &s, &m, &w, &dy, masked);
        let cpu = cpu_grad(&h, &s, &m, &w, &dy);
        close(&cpu.fc2_b, &[0.5, 0.5, -1.5], 1e-7); // sum, not batch mean
        for tile_len in [32, 128, 256] {
            let g = be
                .laya_act_head_backward_f32(
                    &t.inputs(),
                    &layout(),
                    ElementwiseLaunchConfig { tile_len },
                )
                .unwrap();
            for (g, want) in [
                (&g.fc1_w, &cpu.fc1_w),
                (&g.fc1_b, &cpu.fc1_b),
                (&g.fc2_w, &cpu.fc2_w),
                (&g.fc2_b, &cpu.fc2_b),
                (&g.dcls, &cpu.dcls),
            ] {
                close(&be.download_f32(g).unwrap(), want, 2e-5);
            }
            assert_eq!(g.fc1_w.shape(), [F, H + 4]);
            assert_eq!(g.fc1_b.shape(), [F]);
            assert_eq!(g.fc2_w.shape(), [A, F]);
            assert_eq!(g.fc2_b.shape(), [A]);
            assert_eq!(g.dcls.shape(), [B, H]);
        }
        // Pure gradients must not apply updates or mutate resident operands.
        close(&be.download_f32(&t.h).unwrap(), &h, 1e-7);
        close(&be.download_f32(&t.s).unwrap(), &s, 1e-7);
        close(&be.download_f32(&t.m).unwrap(), &m, 1e-7);
        close(&be.download_f32(&t.w1).unwrap(), &w.fc1_w, 1e-7);
        close(&be.download_f32(&t.b1).unwrap(), &w.fc1_b, 1e-7);
        close(&be.download_f32(&t.w2).unwrap(), &w.fc2_w, 1e-7);
        close(&be.download_f32(&t.dy).unwrap(), &dy, 1e-7);
    }
}

fn gpu_score(
    be: &OxideBackend,
    h: &[f32],
    s: &[f32],
    m: &[f32],
    w: &LayaActHeadWeights,
    dy: &[f32],
) -> f64 {
    let t = Tensors::new(be, h, s, m, w, dy, false);
    let cfg = ElementwiseLaunchConfig::default();
    let x = be
        .laya_act_features_f32(&t.h, &t.s, &t.m, &HS, &SS, &MS, cfg)
        .unwrap();
    let z = be
        .connected_f32(&x, &t.w1, &t.b1, B, H + 4, F, cfg)
        .unwrap();
    let a = be.gelu_erf_f32(&z, &[B, F], cfg).unwrap();
    let b2 = be.upload_f32(&[A], &w.fc2_b).unwrap();
    let out = be.connected_f32(&a, &t.w2, &b2, B, F, A, cfg).unwrap();
    be.download_f32(&out)
        .unwrap()
        .iter()
        .zip(dy)
        .map(|(&y, &d)| f64::from(y) * f64::from(d))
        .sum()
}

#[test]
fn laya_act_backward_native_matches_actual_gpu_forward_parameter_and_cls_fd() {
    let be = backend();
    let (h, s, m, w, dy) = fixture(false);
    let t = Tensors::new(&be, &h, &s, &m, &w, &dy, false);
    let g = be
        .laya_act_head_backward_f32(&t.inputs(), &layout(), ElementwiseLaunchConfig::default())
        .unwrap();
    for (block, tensor) in [(0, &g.fc1_w), (1, &g.fc1_b), (2, &g.fc2_w), (3, &g.fc2_b)] {
        let gradient = be.download_f32(tensor).unwrap();
        for index in [0, gradient.len() - 1] {
            let score = |offset: f32| {
                let mut shifted = w.clone();
                let values = match block {
                    0 => &mut shifted.fc1_w,
                    1 => &mut shifted.fc1_b,
                    2 => &mut shifted.fc2_w,
                    _ => &mut shifted.fc2_b,
                };
                values[index] += offset;
                gpu_score(&be, &h, &s, &m, &shifted, &dy)
            };
            let fd = (score(0.001) - score(-0.001)) / 0.002;
            assert!(
                (fd - f64::from(gradient[index])).abs() < 0.001,
                "block {block} index {index}: {fd} vs {}",
                gradient[index]
            );
        }
    }
    let dcls = be.download_f32(&g.dcls).unwrap();
    // Hold scorer/marker buffers fixed, respecting detached summary features.
    for index in [0, 1, TH, (B * H - 1) * TH, B * H * TH - 1] {
        let score = |offset: f32| {
            let mut shifted = h.clone();
            shifted[index] += offset;
            gpu_score(&be, &shifted, &s, &m, &w, &dy)
        };
        let fd = (score(0.001) - score(-0.001)) / 0.002;
        let want = if index % TH == 0 {
            dcls[index / TH]
        } else {
            0.0
        };
        assert!(
            (fd - f64::from(want)).abs() < 0.001,
            "head {index}: {fd} vs {want}"
        );
    }
}

#[test]
fn laya_act_backward_native_checks_layout_all_operands_and_seed_metadata() {
    let be = backend();
    let (h, s, m, w, dy) = fixture(false);
    let t = Tensors::new(&be, &h, &s, &m, &w, &dy, false);
    let cfg = ElementwiseLaunchConfig::default();
    for (f, a) in [(0, A), (F, 0), (usize::MAX, A), (F, usize::MAX)] {
        assert!(MlpLayout::new(&HS, &SS, &MS, f, a).is_err());
    }
    assert!(MlpLayout::new(&[B, H, 2, TH], &SS, &MS, F, A).is_err());
    assert!(MlpLayout::new(&HS, &SS, &[B, 2 * K - 1, 1, TM], F, A).is_err());
    let short = be.upload_f32(&[1], &[0.0]).unwrap();
    for field in 0..7 {
        let mut args = t.inputs();
        match field {
            0 => args.head = &short,
            1 => args.scorer = &short,
            2 => args.markers = &short,
            3 => args.fc1_w = &short,
            4 => args.fc1_b = &short,
            5 => args.fc2_w = &short,
            _ => args.seeds = &short,
        };
        assert!(
            be.laya_act_head_backward_f32(&args, &layout(), cfg)
                .is_err()
        );
    }
    for field in 0..4 {
        let (shape, data) = match field {
            0 => (vec![H + 4, F], w.fc1_w.clone()),
            1 => (vec![1, F], w.fc1_b.clone()),
            2 => (vec![F, A], w.fc2_w.clone()),
            _ => (vec![A, B], dy.clone()),
        };
        let wrong = be.upload_f32(&shape, &data).unwrap();
        let mut args = t.inputs();
        match field {
            0 => args.fc1_w = &wrong,
            1 => args.fc1_b = &wrong,
            2 => args.fc2_w = &wrong,
            _ => args.seeds = &wrong,
        };
        assert!(
            be.laya_act_head_backward_f32(&args, &layout(), cfg)
                .is_err()
        );
    }
}

#[test]
fn laya_act_backward_native_results_outlive_temporary_operand_buffers() {
    let be = backend();
    let (h, s, m, w, dy) = fixture(true);
    let g = {
        let t = Tensors::new(&be, &h, &s, &m, &w, &dy, false);
        be.laya_act_head_backward_f32(&t.inputs(), &layout(), ElementwiseLaunchConfig::default())
            .unwrap()
    };
    // Force allocator-pool reuse after inputs/intermediates have been dropped.
    let mut churn = Vec::new();
    for n in [B * H, F * (H + 4), F, A * F, A] {
        for _ in 0..3 {
            churn.push(be.upload_f32(&[n], &vec![123.0; n]).unwrap());
        }
    }
    let want = cpu_grad(&h, &s, &m, &w, &dy);
    close(&be.download_f32(&g.dcls).unwrap(), &want.dcls, 2e-5);
    close(&be.download_f32(&g.fc1_b).unwrap(), &want.fc1_b, 2e-5);
    close(&be.download_f32(&g.fc2_w).unwrap(), &want.fc2_w, 2e-5);
    close(&be.download_f32(&g.fc1_w).unwrap(), &want.fc1_w, 2e-5);
    close(&be.download_f32(&g.fc2_b).unwrap(), &want.fc2_b, 2e-5);
}

#[test]
fn laya_act_cls_adjoint_scatter_matches_nchw_host_reference() {
    let be = backend();
    let batch = 2;
    let hidden = 5;
    let tokens = 7;
    let dcls: Vec<_> = (0..batch * hidden)
        .map(|i| (i as f32 * 0.31).sin() - 0.4)
        .collect();
    let mut existing: Vec<_> = (0..batch * hidden * tokens)
        .map(|i| (i as f32 * 0.17).cos() * 0.6)
        .collect();
    let want = darknet_cuda::host_ops::laya_act::scatter_cls_adjoint(
        &dcls, &existing, batch, hidden, tokens,
    )
    .unwrap();
    let dcls_gpu = be.upload_f32(&[batch, hidden], &dcls).unwrap();
    let existing_gpu = be
        .upload_f32(&[batch, hidden, 1, tokens], &existing)
        .unwrap();
    let got = be
        .laya_act_cls_scatter_f32(
            &dcls_gpu,
            &existing_gpu,
            batch,
            hidden,
            tokens,
            ElementwiseLaunchConfig::default(),
        )
        .unwrap();
    assert_eq!(got.shape(), [batch, hidden, 1, tokens]);
    assert_eq!(be.download_f32(&got).unwrap(), want);

    existing[0] += 0.5;
    let malformed = be
        .upload_f32(&[batch * hidden * tokens], &existing)
        .unwrap();
    assert!(
        be.laya_act_cls_scatter_f32(
            &dcls_gpu,
            &malformed,
            batch,
            hidden,
            tokens,
            ElementwiseLaunchConfig::default(),
        )
        .is_err()
    );
    let short = be.upload_f32(&[1], &[0.0]).unwrap();
    assert!(
        be.laya_act_cls_scatter_f32(
            &short,
            &existing_gpu,
            batch,
            hidden,
            tokens,
            ElementwiseLaunchConfig::default(),
        )
        .is_err()
    );
}
