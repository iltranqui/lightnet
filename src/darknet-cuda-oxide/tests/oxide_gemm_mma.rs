//! Tensor-core (TF32 `mma.sync` m16n8k8, f32 accumulate) GEMM and conv tests.
//!
//! Tolerance.  Both operands are rounded to TF32 (10 explicit mantissa bits,
//! round-to-nearest: relative error <= 2^-11 each), so each product carries
//! <= ~2^-10 relative error; f32 accumulation adds ~K * 2^-24 relative to the
//! same scale.  Hence for every output
//!     |err| <= (2^-10 + K*2^-24) * (|alpha| * S + |beta| * |c0|),
//!     S = sum_p |a_ip| * |b_pj|        (the "row-norm" scale).
//! We accept `TOL_REL = 2e-3` of that scale (about 2x the analytic worst case
//! of 1e-3, far tighter than a generic 2e-2) and print the measured maximum
//! err/scale ratio.  For conv, S comes from the same reference evaluated on
//! |x|, |w|, |dy|.  Finite-difference checks are NOT run in TF32 mode: the
//! central difference at eps=1e-3 needs ~1e-6 accuracy, below TF32 rounding,
//! so it would only test the quantiser; exactness of the gradient formulas is
//! already covered by the fp32 finite-difference tests.

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::{GemmMode, OxideBackend};

const TOL_REL: f64 = 2e-3;

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

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn backend(mode: GemmMode) -> OxideBackend {
    OxideBackend::new().expect("backend").with_gemm_mode(mode)
}

/// Checks `|got - want| <= TOL_REL * scale` elementwise; returns max ratio.
fn check(name: &str, got: &[f32], want: &[f64], scale: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut worst = 0.0f64;
    for i in 0..got.len() {
        let err = (got[i] as f64 - want[i]).abs();
        let s = scale[i].max(1e-30);
        worst = worst.max(err / s);
        assert!(err <= TOL_REL * s, "{name}[{i}]: got {}, want {}, err {err}, scale {s}", got[i], want[i]);
    }
    worst
}

#[test]
fn gemm_tf32_parity_all_transposes() {
    let b = backend(GemmMode::Tf32);
    let mut rng = Rng(2024);
    let mut overall = 0.0f64;
    for &(m, n, k) in &[(37usize, 53usize, 29usize), (65, 70, 17), (1, 1, 1), (130, 3, 200), (256, 256, 256)] {
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.7, 0.3), (-1.3, 1.0)] {
                let a = rng.vec(m * k, -1.0, 1.0);
                let bm = rng.vec(k * n, -1.0, 1.0);
                let c0 = rng.vec(m * n, -1.0, 1.0);
                let a_store: Vec<f32> =
                    if ta { (0..k * m).map(|i| a[(i % m) * k + i / m]).collect() } else { a.clone() };
                let b_store: Vec<f32> =
                    if tb { (0..n * k).map(|i| bm[(i % k) * n + i / k]).collect() } else { bm.clone() };
                let at = b.upload_f32(&[a_store.len()], &a_store).unwrap();
                let bt = b.upload_f32(&[b_store.len()], &b_store).unwrap();
                let ct = b.upload_f32(&[m, n], &c0).unwrap();
                let out = b.gemm_f32(&at, &bt, &ct, m, n, k, ta, tb, alpha, beta).unwrap();
                let (mut want, mut scale) = (vec![0.0f64; m * n], vec![0.0f64; m * n]);
                for idx in 0..m * n {
                    let (i, j) = (idx / n, idx % n);
                    let (mut d, mut s) = (0.0f64, 0.0f64);
                    for p in 0..k {
                        let pr = a[i * k + p] as f64 * bm[p * n + j] as f64;
                        d += pr;
                        s += pr.abs();
                    }
                    want[idx] = alpha as f64 * d + beta as f64 * c0[idx] as f64;
                    scale[idx] = alpha.abs() as f64 * s + beta.abs() as f64 * c0[idx].abs() as f64;
                }
                let r = check(
                    &format!("tf32 m{m} n{n} k{k} ta{ta} tb{tb} a{alpha} b{beta}"),
                    &b.download_f32(&out).unwrap(),
                    &want,
                    &scale,
                );
                overall = overall.max(r);
            }
        }
    }
    println!("MEASURED tf32 gemm max err/scale = {overall:.3e} (tol {TOL_REL:e})");
}

#[test]
fn gemm_tf32_beta_zero_ignores_nan_in_c() {
    let b = backend(GemmMode::Tf32);
    let (m, n, k) = (9usize, 11usize, 5usize);
    let at = b.upload_f32(&[m, k], &vec![1.0; m * k]).unwrap();
    let bt = b.upload_f32(&[k, n], &vec![2.0; k * n]).unwrap();
    let ct = b.upload_f32(&[m, n], &vec![f32::NAN; m * n]).unwrap();
    let out = b.gemm_f32(&at, &bt, &ct, m, n, k, false, false, 1.0, 0.0).unwrap();
    assert!(b.download_f32(&out).unwrap().iter().all(|&v| v == 10.0));
}

#[test]
fn gemm_mma_agrees_with_fp32() {
    let (f, t) = (backend(GemmMode::Fp32), backend(GemmMode::Tf32));
    let mut rng = Rng(9);
    let mut worst = 0.0f64;
    for &(m, n, k) in &[(37usize, 53usize, 29usize), (130, 3, 200), (256, 256, 256)] {
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            let a = rng.vec(m * k, -1.0, 1.0);
            let bm = rng.vec(k * n, -1.0, 1.0);
            let c0 = rng.vec(m * n, -1.0, 1.0);
            let run = |be: &OxideBackend| {
                let at = be.upload_f32(&[a.len()], &a).unwrap();
                let bt = be.upload_f32(&[bm.len()], &bm).unwrap();
                let ct = be.upload_f32(&[m, n], &c0).unwrap();
                // Raw buffers are reinterpreted by trans flags (shape-agnostic),
                // which is fine since both backends see identical inputs.
                be.download_f32(&be.gemm_f32(&at, &bt, &ct, m, n, k, ta, tb, 0.9, 0.5).unwrap()).unwrap()
            };
            let (x, y) = (run(&f), run(&t));
            // scale: |alpha| * k * E|a||b| bound via exact sum |a||b| is costly;
            // use K * max|a| * max|b| = K (inputs in [-1,1]) as a safe upper scale.
            let scale = vec![0.9 * k as f64 + 0.5; m * n];
            let want: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            worst = worst.max(check(&format!("agree m{m} n{n} k{k} {ta}{tb}"), &y, &want, &scale));
        }
    }
    println!("MEASURED tf32-vs-fp32 max err/(alpha*K+beta) = {worst:.3e}");
}

// ---- conv through TF32 GEMM ------------------------------------------------

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    b: usize, c: usize, h: usize, w: usize, f: usize, k: usize,
    sx: usize, sy: usize, dil: usize, pad: usize, groups: usize,
}
impl Case {
    fn oh(&self) -> usize { (self.h + 2 * self.pad - self.k) / self.sy + 1 }
    fn ow(&self) -> usize { (self.w + 2 * self.pad - self.k) / self.sx + 1 }
    fn cpg(&self) -> usize { self.c / self.groups }
    fn x_len(&self) -> usize { self.b * self.c * self.h * self.w }
    fn y_len(&self) -> usize { self.b * self.f * self.oh() * self.ow() }
    fn w_len(&self) -> usize { self.f * self.cpg() * self.k * self.k }
}

const CASES: &[Case] = &[
    Case { name: "1x1", b: 2, c: 5, h: 7, w: 9, f: 6, k: 1, sx: 1, sy: 1, dil: 1, pad: 0, groups: 1 },
    Case { name: "3x3_pad1", b: 2, c: 3, h: 9, w: 11, f: 5, k: 3, sx: 1, sy: 1, dil: 1, pad: 1, groups: 1 },
    Case { name: "stride2", b: 1, c: 4, h: 13, w: 13, f: 6, k: 3, sx: 2, sy: 2, dil: 1, pad: 1, groups: 1 },
    Case { name: "dilation2", b: 1, c: 3, h: 11, w: 9, f: 4, k: 3, sx: 1, sy: 1, dil: 2, pad: 2, groups: 1 },
    Case { name: "groups2", b: 2, c: 4, h: 9, w: 7, f: 6, k: 3, sx: 1, sy: 1, dil: 1, pad: 1, groups: 2 },
    Case { name: "wide_tiles", b: 2, c: 16, h: 10, w: 10, f: 70, k: 3, sx: 1, sy: 1, dil: 1, pad: 1, groups: 1 },
    Case { name: "big", b: 1, c: 32, h: 40, w: 40, f: 48, k: 3, sx: 1, sy: 1, dil: 1, pad: 1, groups: 1 },
];

/// Reference `(y, dx, dw)` for inputs `x`, `w`, `dy` (f64).
fn refs(c: &Case, x: &[f64], w: &[f64], dy: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (oh, ow, cpg) = (c.oh(), c.ow(), c.cpg());
    let fpg = c.f / c.groups;
    let (mut y, mut dx, mut dw) = (vec![0.0; c.y_len()], vec![0.0; c.x_len()], vec![0.0; c.w_len()]);
    for b in 0..c.b {
        for f in 0..c.f {
            let g = f / fpg;
            for oy in 0..oh {
                for ox in 0..ow {
                    let yi = ((b * c.f + f) * oh + oy) * ow + ox;
                    for ic in 0..cpg {
                        for ky in 0..c.k {
                            for kx in 0..c.k {
                                let iy = (oy * c.sy + ky * c.dil) as isize - c.pad as isize;
                                let ix = (ox * c.sx + kx * c.dil) as isize - c.pad as isize;
                                if iy < 0 || iy >= c.h as isize || ix < 0 || ix >= c.w as isize {
                                    continue;
                                }
                                let xi = ((b * c.c + g * cpg + ic) * c.h + iy as usize) * c.w + ix as usize;
                                let wi = ((f * cpg + ic) * c.k + ky) * c.k + kx;
                                y[yi] += x[xi] * w[wi];
                                dx[xi] += w[wi] * dy[yi];
                                dw[wi] += x[xi] * dy[yi];
                            }
                        }
                    }
                }
            }
        }
    }
    (y, dx, dw)
}

#[test]
fn conv_fwd_dgrad_wgrad_through_tf32_gemm() {
    let be = OxideBackend::new()
        .unwrap()
        .with_conv_mode(darknet_cuda_oxide::ConvMode::Gemm)
        .with_gemm_mode(GemmMode::Tf32);
    let mut worst = 0.0f64;
    for (i, c) in CASES.iter().enumerate() {
        let mut rng = Rng(700 + i as u64);
        let (x, w, dy) = (rng.vec(c.x_len(), -1.0, 1.0), rng.vec(c.w_len(), -1.0, 1.0), rng.vec(c.y_len(), -1.0, 1.0));
        let d = |v: &[f32]| v.iter().map(|&e| e as f64).collect::<Vec<_>>();
        let ab = |v: &[f32]| v.iter().map(|&e| (e as f64).abs()).collect::<Vec<_>>();
        let (y, dx, dw) = refs(c, &d(&x), &d(&w), &d(&dy));
        let (sy, sdx, sdw) = refs(c, &ab(&x), &ab(&w), &ab(&dy));
        let xt = be.upload_f32(&[c.b, c.c, c.h, c.w], &x).unwrap();
        let wt = be.upload_f32(&[c.f, c.cpg(), c.k, c.k], &w).unwrap();
        let dyt = be.upload_f32(&[c.b, c.f, c.oh(), c.ow()], &dy).unwrap();
        let gy = be.conv2d_f32(&xt, &wt, &[c.b, c.f, c.oh(), c.ow()], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg()).unwrap();
        let gdx = be.conv2d_backward_input_f32(&dyt, &wt, &[c.b, c.c, c.h, c.w], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg()).unwrap();
        let gdw = be.conv2d_backward_weights_f32(&xt, &dyt, &[c.f, c.cpg(), c.k, c.k], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg()).unwrap();
        worst = worst.max(check(&format!("{} fwd", c.name), &be.download_f32(&gy).unwrap(), &y, &sy));
        worst = worst.max(check(&format!("{} dgrad", c.name), &be.download_f32(&gdx).unwrap(), &dx, &sdx));
        worst = worst.max(check(&format!("{} wgrad", c.name), &be.download_f32(&gdw).unwrap(), &dw, &sdw));
    }
    println!("MEASURED tf32 conv max err/scale = {worst:.3e} (tol {TOL_REL:e})");
}

#[test]
fn auto_mode_threshold() {
    // Small problem -> fp32 path, must equal the fp32 backend bit-for-bit.
    let (f, a) = (backend(GemmMode::Fp32), backend(GemmMode::Auto));
    let mut rng = Rng(5);
    let (m, n, k) = (40usize, 40usize, 40usize);
    let (x, y, c0) = (rng.vec(m * k, -1.0, 1.0), rng.vec(k * n, -1.0, 1.0), rng.vec(m * n, -1.0, 1.0));
    let run = |be: &OxideBackend| {
        let (xt, yt, ct) = (be.upload_f32(&[m, k], &x).unwrap(), be.upload_f32(&[k, n], &y).unwrap(), be.upload_f32(&[m, n], &c0).unwrap());
        be.download_f32(&be.gemm_f32(&xt, &yt, &ct, m, n, k, false, false, 1.0, 0.0).unwrap()).unwrap()
    };
    assert_eq!(run(&f), run(&a));
}

/// Interleaved A/B (3 rounds).  Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn perf_fp32_vs_tf32() {
    let (f, t) = (backend(GemmMode::Fp32), backend(GemmMode::Tf32));
    let mut rng = Rng(1);
    let n = 1024usize;
    let (a, b) = (rng.vec(n * n, -1.0, 1.0), rng.vec(n * n, -1.0, 1.0));
    let (af, bf) = (f.upload_f32(&[n, n], &a).unwrap(), f.upload_f32(&[n, n], &b).unwrap());
    let (at, bt) = (t.upload_f32(&[n, n], &a).unwrap(), t.upload_f32(&[n, n], &b).unwrap());
    let flops = 2.0 * (n as f64).powi(3);
    for round in 0..3 {
        let m1 = f.bench_gemm_f32(&af, &bf, n, n, n, 20).unwrap();
        let m2 = t.bench_gemm_f32(&at, &bt, n, n, n, 20).unwrap();
        println!("PERF round{round} gemm1024 fp32 {m1:.3} ms {:.0} GFLOP/s | tf32 {m2:.3} ms {:.0} GFLOP/s", flops / m1 / 1e6, flops / m2 / 1e6);
    }
    // conv 64->64 3x3 104x104 (fwd/dgrad/wgrad, gemm path)
    let mk = |mode| OxideBackend::new().unwrap().with_conv_mode(darknet_cuda_oxide::ConvMode::Gemm).with_gemm_mode(mode);
    let (cf, ct) = (mk(GemmMode::Fp32), mk(GemmMode::Tf32));
    let x = rng.vec(64 * 104 * 104, -1.0, 1.0);
    let w = rng.vec(64 * 64 * 9, -1.0, 1.0);
    let cflops = 2.0 * 64.0 * 64.0 * 9.0 * 104.0 * 104.0;
    let time = |be: &OxideBackend, which: usize| {
        let xt = be.upload_f32(&[1, 64, 104, 104], &x).unwrap();
        let wt = be.upload_f32(&[64, 64, 3, 3], &w).unwrap();
        let run = || match which {
            0 => drop(be.conv2d_f32(&xt, &wt, &[1, 64, 104, 104], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
            1 => drop(be.conv2d_backward_input_f32(&xt, &wt, &[1, 64, 104, 104], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
            _ => drop(be.conv2d_backward_weights_f32(&xt, &xt, &[64, 64, 3, 3], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
        };
        run();
        let _ = be.download_f32(&be.zeros_f32(&[1]).unwrap());
        let t0 = std::time::Instant::now();
        for _ in 0..10 { run(); }
        let _ = be.download_f32(&be.zeros_f32(&[1]).unwrap());
        t0.elapsed().as_secs_f64() * 1e3 / 10.0
    };
    for round in 0..3 {
        for (which, name) in ["fwd", "dgrad", "wgrad"].iter().enumerate() {
            let m1 = time(&cf, which);
            let m2 = time(&ct, which);
            println!("PERF round{round} conv64 {name} fp32 {m1:.3} ms {:.0} GFLOP/s | tf32 {m2:.3} ms {:.0} GFLOP/s", cflops / m1 / 1e6, cflops / m2 / 1e6);
        }
    }
}
