//! Batch-norm training forward/backward and the Adam update: parity against
//! independent f64 CPU references, central finite differences through the full
//! forward, bit-exact determinism, and edge shapes.

use darknet_cuda::{AdamParams, ElementwiseLaunchConfig, F32TensorOpsBackend, NanInfMode};
use darknet_cuda_oxide::OxideBackend;

const TOL: f64 = 2e-4;
const EPS: f64 = 0.000_01;

struct Rng(u64);

impl Rng {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        lo + (hi - lo) * (r as f32 / (1u64 << 24) as f32)
    }

    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(lo, hi)).collect()
    }
}

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn assert_close(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (*g as f64 - w).abs();
        assert!(err <= TOL + TOL * w.abs(), "{name}[{i}]: got {g}, want {w}, err {err}");
    }
}

struct RefForward {
    out: Vec<f64>,
    x_hat: Vec<f64>,
    mean: Vec<f64>,
    var: Vec<f64>,
    unbiased: Vec<f64>,
}

/// Independent f64 reference over NCHW `[b, c, spatial]`.
fn ref_forward(x: &[f64], scale: &[f64], bias: &[f64], c: usize, spatial: usize) -> RefForward {
    let b = x.len() / (c * spatial);
    let n = (b * spatial) as f64;
    let at = |bi: usize, ci: usize, s: usize| x[(bi * c + ci) * spatial + s];
    let mut mean = vec![0.0; c];
    let mut var = vec![0.0; c];
    let mut unbiased = vec![0.0; c];
    for ci in 0..c {
        let m: f64 = (0..b).flat_map(|bi| (0..spatial).map(move |s| (bi, s))).map(|(bi, s)| at(bi, ci, s)).sum::<f64>() / n;
        let sq: f64 = (0..b)
            .flat_map(|bi| (0..spatial).map(move |s| (bi, s)))
            .map(|(bi, s)| (at(bi, ci, s) - m).powi(2))
            .sum();
        mean[ci] = m;
        var[ci] = sq / n;
        unbiased[ci] = sq / (n - 1.0).max(1.0);
    }
    let mut x_hat = vec![0.0; x.len()];
    let mut out = vec![0.0; x.len()];
    for i in 0..x.len() {
        let ci = (i / spatial) % c;
        x_hat[i] = (x[i] - mean[ci]) / (var[ci] + EPS).sqrt();
        out[i] = x_hat[i] * scale[ci] + bias[ci];
    }
    RefForward { out, x_hat, mean, var, unbiased }
}

struct Case {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
}

const CASES: &[Case] = &[
    Case { b: 2, c: 3, h: 4, w: 5 },
    Case { b: 1, c: 1, h: 1, w: 7 },
    Case { b: 3, c: 5, h: 17, w: 19 }, // stride 323 > 256: several strided partials
    Case { b: 4, c: 2, h: 1, w: 1 },   // stride 1: all spatial info is in the batch
];

#[test]
fn batchnorm_forward_matches_f64_reference() {
    let be = backend();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for (k, case) in CASES.iter().enumerate() {
        let (c, spatial) = (case.c, case.h * case.w);
        let shape = [case.b, c, case.h, case.w];
        let n = case.b * spatial;
        let x = rng.vec(n * c, -3.0, 5.0);
        let scale = rng.vec(c, 0.5, 2.0);
        let bias = rng.vec(c, -1.0, 1.0);
        let rm = rng.vec(c, -1.0, 1.0);
        let rv = rng.vec(c, 0.5, 2.0);
        let keep = 0.9f32;
        let t = |s: &[usize], v: &[f32]| be.upload_f32(s, v).unwrap();
        let got = be
            .batchnorm_train_forward_f32(
                &t(&shape, &x),
                &t(&[c], &scale),
                &t(&[c], &bias),
                &t(&[c], &rm),
                &t(&[c], &rv),
                &shape,
                keep,
                cfg(),
            )
            .unwrap();
        let want = ref_forward(&f64s(&x), &f64s(&scale), &f64s(&bias), c, spatial);
        let name = |s: &str| format!("case {k} {s}");
        assert_close(&name("out"), &be.download_f32(&got.output).unwrap(), &want.out);
        assert_close(&name("x_hat"), &be.download_f32(&got.x_hat).unwrap(), &want.x_hat);
        assert_close(&name("mean"), &be.download_f32(&got.mean).unwrap(), &want.mean);
        assert_close(&name("var"), &be.download_f32(&got.variance).unwrap(), &want.var);
        let keep = keep as f64;
        let want_rm: Vec<f64> = (0..c).map(|i| keep * rm[i] as f64 + (1.0 - keep) * want.mean[i]).collect();
        let want_rv: Vec<f64> = (0..c).map(|i| keep * rv[i] as f64 + (1.0 - keep) * want.unbiased[i]).collect();
        assert_close(&name("rolling_mean"), &be.download_f32(&got.rolling_mean).unwrap(), &want_rm);
        assert_close(&name("rolling_var"), &be.download_f32(&got.rolling_variance).unwrap(), &want_rv);

        // The trait's default host implementation must agree with the kernels.
        let host = darknet_cuda::host_ops::batchnorm_train_forward(&x, &scale, &bias, &rm, &rv, c, spatial, keep as f32);
        assert_close(&name("host out"), &host.output, &want.out);
        assert_close(&name("host rolling_var"), &host.rolling_variance, &want_rv);
    }
}

#[test]
fn batchnorm_backward_matches_reference_and_finite_difference() {
    let be = backend();
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    // Small cases so the O(n^2) finite difference stays cheap.
    for case in [Case { b: 2, c: 3, h: 2, w: 3 }, Case { b: 3, c: 2, h: 1, w: 4 }, Case { b: 1, c: 2, h: 3, w: 3 }] {
        let (c, spatial) = (case.c, case.h * case.w);
        let shape = [case.b, c, case.h, case.w];
        let len = case.b * c * spatial;
        let x = rng.vec(len, -2.0, 3.0);
        let scale = rng.vec(c, 0.5, 2.0);
        let bias = rng.vec(c, -1.0, 1.0);
        let dy = rng.vec(len, -1.0, 1.0);
        let t = |s: &[usize], v: &[f32]| be.upload_f32(s, v).unwrap();
        let fwd = be
            .batchnorm_train_forward_f32(
                &t(&shape, &x),
                &t(&[c], &scale),
                &t(&[c], &bias),
                &t(&[c], &vec![0.0; c]),
                &t(&[c], &vec![1.0; c]),
                &shape,
                0.9,
                cfg(),
            )
            .unwrap();
        let (dx, dscale, dbias) = be
            .batchnorm_train_backward_f32(&t(&shape, &dy), &fwd.x_hat, &fwd.variance, &t(&[c], &scale), &shape, cfg())
            .unwrap();
        let (dx, dscale, dbias) =
            (be.download_f32(&dx).unwrap(), be.download_f32(&dscale).unwrap(), be.download_f32(&dbias).unwrap());

        // Loss L = sum(dy * out), differentiated through the whole forward.
        let (xd, sd, bd, dyd) = (f64s(&x), f64s(&scale), f64s(&bias), f64s(&dy));
        let loss = |xv: &[f64], sv: &[f64], bv: &[f64]| -> f64 {
            ref_forward(xv, sv, bv, c, spatial).out.iter().zip(&dyd).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3f64;
        let fd = |vals: &[f64], f: &dyn Fn(&[f64]) -> f64, i: usize| -> f64 {
            let mut v = vals.to_vec();
            v[i] = vals[i] + eps;
            let lp = f(&v);
            v[i] = vals[i] - eps;
            let lm = f(&v);
            (lp - lm) / (2.0 * eps)
        };
        for i in 0..len {
            let want = fd(&xd, &|v| loss(v, &sd, &bd), i);
            assert!((dx[i] as f64 - want).abs() <= 2e-3 + 2e-3 * want.abs(), "dx fd[{i}]: {} vs {want}", dx[i]);
        }
        for ci in 0..c {
            let ws = fd(&sd, &|v| loss(&xd, v, &bd), ci);
            let wb = fd(&bd, &|v| loss(&xd, &sd, v), ci);
            assert!((dscale[ci] as f64 - ws).abs() <= 2e-3 + 2e-3 * ws.abs(), "dscale fd[{ci}]: {} vs {ws}", dscale[ci]);
            assert!((dbias[ci] as f64 - wb).abs() <= 2e-3 + 2e-3 * wb.abs(), "dbias fd[{ci}]: {} vs {wb}", dbias[ci]);
        }

        // And against the host reference the trait default uses.
        let want = ref_forward(&xd, &sd, &bd, c, spatial);
        let (hdx, hds, hdb) = darknet_cuda::host_ops::batchnorm_train_backward(
            &dy,
            &want.x_hat.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &want.var.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &scale,
            c,
            spatial,
        );
        assert_close("host dx", &hdx, &f64s(&dx));
        assert_close("host dscale", &hds, &f64s(&dscale));
        assert_close("host dbias", &hdb, &f64s(&dbias));
    }
}

#[test]
fn batchnorm_reductions_are_deterministic() {
    let be = backend();
    let mut rng = Rng(0x5eed_0000_1111_2222);
    let shape = [3, 4, 23, 29];
    let len: usize = shape.iter().product();
    let c = shape[1];
    let t = |s: &[usize], v: &[f32]| be.upload_f32(s, v).unwrap();
    let x = t(&shape, &rng.vec(len, -4.0, 4.0));
    let dy = t(&shape, &rng.vec(len, -1.0, 1.0));
    let (scale, bias) = (t(&[c], &rng.vec(c, 0.5, 1.5)), t(&[c], &rng.vec(c, -0.5, 0.5)));
    let (rm, rv) = (t(&[c], &vec![0.0; c]), t(&[c], &vec![1.0; c]));
    let run = || {
        let f = be.batchnorm_train_forward_f32(&x, &scale, &bias, &rm, &rv, &shape, 0.9, cfg()).unwrap();
        let (dx, ds, db) = be.batchnorm_train_backward_f32(&dy, &f.x_hat, &f.variance, &scale, &shape, cfg()).unwrap();
        [f.output, f.mean, f.variance, f.rolling_variance, dx, ds, db].map(|t| be.download_f32(&t).unwrap())
    };
    let (a, b) = (run(), run());
    for (i, (u, v)) in a.iter().zip(&b).enumerate() {
        assert_eq!(u.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "output {i} differs between runs");
    }
}

#[test]
fn batchnorm_rejects_bad_shapes() {
    let be = backend();
    let x = be.upload_f32(&[1, 2, 2, 2], &[0.0; 8]).unwrap();
    let c2 = be.upload_f32(&[2], &[1.0; 2]).unwrap();
    let c3 = be.upload_f32(&[3], &[1.0; 3]).unwrap();
    assert!(be.batchnorm_train_forward_f32(&x, &c3, &c2, &c2, &c2, &[1, 2, 2, 2], 0.9, cfg()).is_err());
    assert!(be.batchnorm_train_forward_f32(&x, &c2, &c2, &c2, &c2, &[1, 2, 2, 3], 0.9, cfg()).is_err());
}

// ------------------------------------------------------------------- Adam

fn ref_adam(w: &[f64], u: &[f64], m: &[f64], v: &[f64], p: &AdamParams) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (b1, b2, lr, eps) = (p.beta1 as f64, p.beta2 as f64, p.lr as f64, p.eps as f64);
    let t = p.step as i32;
    let (coupled, decoupled) = if p.decoupled_decay {
        (0.0, lr * p.decay as f64)
    } else {
        (p.decay as f64 * p.batch as f64, 0.0)
    };
    let (mut wo, mut mo, mut vo) = (w.to_vec(), m.to_vec(), v.to_vec());
    for i in 0..w.len() {
        let g = u[i] - coupled * w[i];
        mo[i] = b1 * m[i] + (1.0 - b1) * g;
        vo[i] = b2 * v[i] + (1.0 - b2) * g * g;
        let (mh, vh) = (mo[i] / (1.0 - b1.powi(t)), vo[i] / (1.0 - b2.powi(t)));
        wo[i] = w[i] + lr * mh / (vh.sqrt() + eps) - decoupled * w[i];
    }
    (wo, mo, vo)
}

#[test]
fn adam_update_matches_reference_over_steps() {
    let be = backend();
    let mut rng = Rng(0xada0_0000_0000_0001);
    for decoupled in [false, true] {
        let n = 1000; // not a multiple of the block size
        let mut w = rng.vec(n, -1.0, 1.0);
        let mut m = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let (mut rw, mut rm, mut rv) = (f64s(&w), f64s(&m), f64s(&v));
        let mut tw = be.upload_f32(&[n], &w).unwrap();
        let mut tm = be.upload_f32(&[n], &m).unwrap();
        let mut tv = be.upload_f32(&[n], &v).unwrap();
        let mut host = (w.clone(), m.clone(), v.clone());
        for step in 1..=4u32 {
            let u = rng.vec(n, -0.5, 0.5);
            let p = AdamParams {
                lr: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-6,
                decay: 5e-4,
                batch: 8,
                step,
                decoupled_decay: decoupled,
            };
            let tu = be.upload_f32(&[n], &u).unwrap();
            (tw, tm, tv) = be.adam_update_f32(&tw, &tu, &tm, &tv, &p, cfg()).unwrap();
            (rw, rm, rv) = ref_adam(&rw, &f64s(&u), &rm, &rv, &p);
            host = darknet_cuda::host_ops::adam_update(&host.0, &u, &host.1, &host.2, &p);
            w = be.download_f32(&tw).unwrap();
            m = be.download_f32(&tm).unwrap();
            v = be.download_f32(&tv).unwrap();
            let tag = format!("decoupled={decoupled} step {step}");
            assert_close(&format!("{tag} w"), &w, &rw);
            assert_close(&format!("{tag} m"), &m, &rm);
            assert_close(&format!("{tag} v"), &v, &rv);
            assert_close(&format!("{tag} host w"), &host.0, &rw);
        }
    }
}

#[test]
fn adam_rejects_length_mismatch() {
    let be = backend();
    let a = be.upload_f32(&[4], &[0.0; 4]).unwrap();
    let b = be.upload_f32(&[3], &[0.0; 3]).unwrap();
    let p = AdamParams { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-6, decay: 0.0, batch: 1, step: 1, decoupled_decay: false };
    assert!(be.adam_update_f32(&a, &a, &a, &b, &p, cfg()).is_err());
}

#[test]
fn sanitize_nan_inf_matches_host_reference() {
    let be = backend();
    let n = 1030; // not a multiple of the block size
    let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    for i in [0usize, 5, 255, 256, 1029] {
        x[i] = f32::NAN;
    }
    x[7] = f32::INFINITY;
    x[600] = f32::NEG_INFINITY;
    let t = be.upload_f32(&[10, 103], &x).unwrap();
    for mode in [NanInfMode::Reset, NanInfMode::Fix] {
        let out = be.sanitize_nan_inf_f32(&t, &[10, 103], mode, cfg()).unwrap();
        let got = be.download_f32(&out).unwrap();
        let want = darknet_cuda::host_ops::sanitize_nan_inf(&x, mode);
        assert_eq!(got, want, "{mode:?}");
        assert!(got.iter().all(|v| v.is_finite()));
    }
    assert!(be.sanitize_nan_inf_f32(&t, &[3], NanInfMode::Reset, cfg()).is_err());
}
