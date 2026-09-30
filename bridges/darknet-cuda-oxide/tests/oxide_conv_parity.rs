//! Convolution + GEMM parity for the cuda-oxide backend.
//!
//! References are written in f64 here (same indexing semantics as
//! `darknet-core`'s `convolution_raw_value` / `backward_convolutional`):
//! weights `[F, C/groups, k, k]`, NCHW, symmetric padding, per-axis strides.
//! Both conv paths (`Direct`, `Gemm`) are checked against them, against each
//! other, and against central finite differences of the f64 forward.

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use darknet_cuda_oxide::{ConvMode, OxideBackend};

const TOL: f32 = 1e-3;

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

fn backend(mode: ConvMode) -> OxideBackend {
    OxideBackend::new().expect("create oxide backend").with_conv_mode(mode)
}

fn assert_close(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (*g as f64 - w).abs();
        assert!(
            err <= TOL as f64 + TOL as f64 * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    k: usize,
    sx: usize,
    sy: usize,
    dil: usize,
    pad: usize,
    groups: usize,
}

impl Case {
    fn oh(&self) -> usize {
        (self.h + 2 * self.pad - (self.dil * (self.k - 1) + 1)) / self.sy + 1
    }
    fn ow(&self) -> usize {
        (self.w + 2 * self.pad - (self.dil * (self.k - 1) + 1)) / self.sx + 1
    }
    fn cpg(&self) -> usize {
        self.c / self.groups
    }
    fn x_len(&self) -> usize {
        self.b * self.c * self.h * self.w
    }
    fn y_len(&self) -> usize {
        self.b * self.f * self.oh() * self.ow()
    }
    fn w_len(&self) -> usize {
        self.f * self.cpg() * self.k * self.k
    }
}

#[allow(clippy::too_many_arguments)]
const fn case(
    name: &'static str,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    k: usize,
    sx: usize,
    sy: usize,
    dil: usize,
    pad: usize,
    groups: usize,
) -> Case {
    Case { name, b, c, h, w, f, k, sx, sy, dil, pad, groups }
}

const CASES: &[Case] = &[
    case("1x1", 2, 5, 7, 9, 6, 1, 1, 1, 1, 0, 1),
    case("3x3_pad1", 2, 3, 9, 11, 5, 3, 1, 1, 1, 1, 1),
    case("3x3_pad0", 1, 2, 8, 8, 3, 3, 1, 1, 1, 0, 1),
    case("stride2", 1, 4, 13, 13, 6, 3, 2, 2, 1, 1, 1),
    case("sx2_sy1", 2, 3, 10, 12, 4, 3, 2, 1, 1, 1, 1),
    case("sx1_sy3", 2, 3, 11, 10, 4, 3, 1, 3, 1, 1, 1),
    case("dilation2_pad2", 1, 3, 11, 9, 4, 3, 1, 1, 2, 2, 1),
    case("k5_s2_odd", 3, 3, 15, 13, 5, 5, 2, 2, 1, 2, 1),
    case("groups2", 2, 4, 9, 7, 6, 3, 1, 1, 1, 1, 2),
    case("depthwise", 2, 6, 9, 9, 6, 3, 1, 1, 1, 1, 6),
    case("depthwise_s2", 1, 5, 11, 11, 5, 3, 2, 2, 1, 1, 5),
    case("wide_tiles", 2, 16, 10, 10, 70, 3, 1, 1, 1, 1, 1),
    case("wide_groups", 1, 8, 9, 9, 80, 3, 1, 1, 1, 1, 2),
];

/// Smaller subset for the (O(n^2) in f64) finite-difference checks.
const FD_CASES: &[Case] = &[
    case("fd_1x1", 2, 3, 4, 5, 4, 1, 1, 1, 1, 0, 1),
    case("fd_3x3", 2, 2, 5, 6, 3, 3, 1, 1, 1, 1, 1),
    case("fd_stride2", 1, 2, 7, 6, 3, 3, 2, 2, 1, 1, 1),
    case("fd_sx2_sy1", 1, 2, 6, 7, 2, 3, 2, 1, 1, 1, 1),
    case("fd_dil2", 1, 2, 7, 7, 2, 3, 1, 1, 2, 2, 1),
    case("fd_groups2", 2, 4, 5, 5, 4, 3, 1, 1, 1, 1, 2),
    case("fd_depthwise", 1, 3, 6, 5, 3, 3, 1, 1, 1, 1, 3),
];

fn fwd_ref(c: &Case, x: &[f64], w: &[f64]) -> Vec<f64> {
    let (oh, ow, cpg) = (c.oh(), c.ow(), c.cpg());
    let fpg = c.f / c.groups;
    let mut y = vec![0.0; c.y_len()];
    for b in 0..c.b {
        for f in 0..c.f {
            let g = f / fpg;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0;
                    for ic in 0..cpg {
                        for ky in 0..c.k {
                            for kx in 0..c.k {
                                let iy = (oy * c.sy + ky * c.dil) as isize - c.pad as isize;
                                let ix = (ox * c.sx + kx * c.dil) as isize - c.pad as isize;
                                if iy < 0 || iy >= c.h as isize || ix < 0 || ix >= c.w as isize {
                                    continue;
                                }
                                let ch = g * cpg + ic;
                                acc += x[((b * c.c + ch) * c.h + iy as usize) * c.w + ix as usize]
                                    * w[((f * cpg + ic) * c.k + ky) * c.k + kx];
                            }
                        }
                    }
                    y[((b * c.f + f) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    y
}

/// Returns `(dx, dw)` for upstream gradient `dy`.
fn bwd_ref(c: &Case, x: &[f64], w: &[f64], dy: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (oh, ow, cpg) = (c.oh(), c.ow(), c.cpg());
    let fpg = c.f / c.groups;
    let mut dx = vec![0.0; c.x_len()];
    let mut dw = vec![0.0; c.w_len()];
    for b in 0..c.b {
        for f in 0..c.f {
            let g = f / fpg;
            for oy in 0..oh {
                for ox in 0..ow {
                    let d = dy[((b * c.f + f) * oh + oy) * ow + ox];
                    for ic in 0..cpg {
                        for ky in 0..c.k {
                            for kx in 0..c.k {
                                let iy = (oy * c.sy + ky * c.dil) as isize - c.pad as isize;
                                let ix = (ox * c.sx + kx * c.dil) as isize - c.pad as isize;
                                if iy < 0 || iy >= c.h as isize || ix < 0 || ix >= c.w as isize {
                                    continue;
                                }
                                let xi =
                                    ((b * c.c + g * cpg + ic) * c.h + iy as usize) * c.w + ix as usize;
                                let wi = ((f * cpg + ic) * c.k + ky) * c.k + kx;
                                dw[wi] += x[xi] * d;
                                dx[xi] += w[wi] * d;
                            }
                        }
                    }
                }
            }
        }
    }
    (dx, dw)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&e| e as f64).collect()
}

struct Gpu {
    y: Vec<f32>,
    dx: Vec<f32>,
    dw: Vec<f32>,
}

fn run_gpu(b: &OxideBackend, c: &Case, x: &[f32], w: &[f32], dy: &[f32]) -> Gpu {
    let xt = b.upload_f32(&[c.b, c.c, c.h, c.w], x).unwrap();
    let wt = b.upload_f32(&[c.f, c.cpg(), c.k, c.k], w).unwrap();
    let dyt = b.upload_f32(&[c.b, c.f, c.oh(), c.ow()], dy).unwrap();
    let y = b
        .conv2d_f32(
            &xt, &wt, &[c.b, c.f, c.oh(), c.ow()], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil,
            c.pad, c.groups, cfg(),
        )
        .unwrap();
    let dx = b
        .conv2d_backward_input_f32(
            &dyt, &wt, &[c.b, c.c, c.h, c.w], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad,
            c.groups, cfg(),
        )
        .unwrap();
    let dw = b
        .conv2d_backward_weights_f32(
            &xt, &dyt, &[c.f, c.cpg(), c.k, c.k], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil,
            c.pad, c.groups, cfg(),
        )
        .unwrap();
    Gpu {
        y: b.download_f32(&y).unwrap(),
        dx: b.download_f32(&dx).unwrap(),
        dw: b.download_f32(&dw).unwrap(),
    }
}

fn inputs(c: &Case, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    (
        rng.vec(c.x_len(), -1.0, 1.0),
        rng.vec(c.w_len(), -1.0, 1.0),
        rng.vec(c.y_len(), -1.0, 1.0),
    )
}

fn parity_for(mode: ConvMode) {
    let b = backend(mode);
    for (i, c) in CASES.iter().enumerate() {
        let (x, w, dy) = inputs(c, 100 + i as u64);
        let got = run_gpu(&b, c, &x, &w, &dy);
        let (xd, wd, dyd) = (f64s(&x), f64s(&w), f64s(&dy));
        let y = fwd_ref(c, &xd, &wd);
        let (dx, dw) = bwd_ref(c, &xd, &wd, &dyd);
        assert_close(&format!("{mode:?} {} fwd", c.name), &got.y, &y);
        assert_close(&format!("{mode:?} {} dgrad", c.name), &got.dx, &dx);
        assert_close(&format!("{mode:?} {} wgrad", c.name), &got.dw, &dw);
    }
}

#[test]
fn conv_parity_direct() {
    parity_for(ConvMode::Direct);
}

#[test]
fn conv_parity_gemm() {
    parity_for(ConvMode::Gemm);
}

#[test]
fn conv_parity_auto() {
    parity_for(ConvMode::Auto);
}

#[test]
fn conv_direct_and_gemm_agree() {
    let (d, g) = (backend(ConvMode::Direct), backend(ConvMode::Gemm));
    for (i, c) in CASES.iter().enumerate() {
        let (x, w, dy) = inputs(c, 500 + i as u64);
        let (a, b) = (run_gpu(&d, c, &x, &w, &dy), run_gpu(&g, c, &x, &w, &dy));
        assert_close(&format!("{} fwd", c.name), &a.y, &f64s(&b.y));
        assert_close(&format!("{} dgrad", c.name), &a.dx, &f64s(&b.dx));
        assert_close(&format!("{} wgrad", c.name), &a.dw, &f64s(&b.dw));
    }
}

/// Direct and GEMM paths must be run-to-run bit-identical (no racy sums).
#[test]
fn conv_is_deterministic() {
    for mode in [ConvMode::Direct, ConvMode::Gemm] {
        let b = backend(mode);
        let c = &CASES[7];
        let (x, w, dy) = inputs(c, 900);
        let first = run_gpu(&b, c, &x, &w, &dy);
        for _ in 0..3 {
            let again = run_gpu(&b, c, &x, &w, &dy);
            assert_eq!(first.y, again.y, "{mode:?} fwd");
            assert_eq!(first.dx, again.dx, "{mode:?} dgrad");
            assert_eq!(first.dw, again.dw, "{mode:?} wgrad");
        }
    }
}

/// Central differences of L = <fwd(x, w), r> on the f64 reference against the
/// GPU dgrad / wgrad with upstream gradient r.
fn fd_for(mode: ConvMode) {
    let b = backend(mode);
    let eps = 1e-3f64;
    for (i, c) in FD_CASES.iter().enumerate() {
        let (x, w, r) = inputs(c, 300 + i as u64);
        let got = run_gpu(&b, c, &x, &w, &r);
        let (xd, wd, rd) = (f64s(&x), f64s(&w), f64s(&r));
        let loss = |xv: &[f64], wv: &[f64]| -> f64 {
            fwd_ref(c, xv, wv).iter().zip(&rd).map(|(a, b)| a * b).sum()
        };
        let mut fd_dx = vec![0.0; xd.len()];
        for j in 0..xd.len() {
            let (mut p, mut m) = (xd.clone(), xd.clone());
            p[j] += eps;
            m[j] -= eps;
            fd_dx[j] = (loss(&p, &wd) - loss(&m, &wd)) / (2.0 * eps);
        }
        let mut fd_dw = vec![0.0; wd.len()];
        for j in 0..wd.len() {
            let (mut p, mut m) = (wd.clone(), wd.clone());
            p[j] += eps;
            m[j] -= eps;
            fd_dw[j] = (loss(&xd, &p) - loss(&xd, &m)) / (2.0 * eps);
        }
        assert_close(&format!("{mode:?} {} fd dgrad", c.name), &got.dx, &fd_dx);
        assert_close(&format!("{mode:?} {} fd wgrad", c.name), &got.dw, &fd_dw);
    }
}

#[test]
fn conv_finite_difference_direct() {
    fd_for(ConvMode::Direct);
}

#[test]
fn conv_finite_difference_gemm() {
    fd_for(ConvMode::Gemm);
}

#[test]
fn conv_rejects_bad_geometry() {
    let b = backend(ConvMode::Auto);
    let x = b.zeros_f32(&[1, 4, 6, 6]).unwrap();
    let w = b.zeros_f32(&[6, 2, 3, 3]).unwrap();
    // groups does not divide filters
    assert!(
        b.conv2d_f32(&x, &w, &[1, 6, 4, 4], 4, 6, 6, 6, 3, 1, 1, 1, 0, 4, cfg()).is_err()
    );
    // wrong output shape
    assert!(
        b.conv2d_f32(&x, &w, &[1, 6, 5, 5], 4, 6, 6, 6, 3, 1, 1, 1, 0, 2, cfg()).is_err()
    );
    // wrong weight length
    assert!(
        b.conv2d_f32(&x, &x, &[1, 6, 4, 4], 4, 6, 6, 6, 3, 1, 1, 1, 0, 2, cfg()).is_err()
    );
    // correct call succeeds
    assert!(
        b.conv2d_f32(&x, &w, &[1, 6, 4, 4], 4, 6, 6, 6, 3, 1, 1, 1, 0, 2, cfg()).is_ok()
    );
}

#[test]
fn backend_marker_traits() {
    fn is_send<T: Send>() {}
    is_send::<OxideBackend>();
    fn is_sync<T: Sync>() {}
    is_sync::<OxideBackend>();
}

#[test]
fn gemm_nn_nt_tn_tt_with_alpha_beta() {
    let b = backend(ConvMode::Auto);
    let mut rng = Rng(77);
    for &(m, n, k) in &[(37usize, 53usize, 29usize), (65, 70, 17), (1, 1, 1), (130, 3, 200), (64, 64, 16), (5, 129, 33)] {
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.7, 0.3), (-1.3, 1.0)] {
                let a = rng.vec(m * k, -1.0, 1.0); // logical m x k
                let bm = rng.vec(k * n, -1.0, 1.0); // logical k x n
                let c0 = rng.vec(m * n, -1.0, 1.0);
                // physical storage
                let a_store: Vec<f32> = if ta {
                    (0..k * m).map(|i| a[(i % m) * k + i / m]).collect()
                } else {
                    a.clone()
                };
                let b_store: Vec<f32> = if tb {
                    (0..n * k).map(|i| bm[(i % k) * n + i / k]).collect()
                } else {
                    bm.clone()
                };
                let at = b.upload_f32(&[a_store.len()], &a_store).unwrap();
                let bt = b.upload_f32(&[b_store.len()], &b_store).unwrap();
                let ct = b.upload_f32(&[m, n], &c0).unwrap();
                let out = b.gemm_f32(&at, &bt, &ct, m, n, k, ta, tb, alpha, beta).unwrap();
                let want: Vec<f64> = (0..m * n)
                    .map(|idx| {
                        let (i, j) = (idx / n, idx % n);
                        let dot: f64 = (0..k).map(|p| a[i * k + p] as f64 * bm[p * n + j] as f64).sum();
                        alpha as f64 * dot + beta as f64 * c0[idx] as f64
                    })
                    .collect();
                assert_close(
                    &format!("gemm m{m} n{n} k{k} ta{ta} tb{tb} a{alpha} b{beta}"),
                    &b.download_f32(&out).unwrap(),
                    &want,
                );
            }
        }
    }
}

#[test]
fn gemm_beta_zero_ignores_nan_in_c() {
    let b = backend(ConvMode::Auto);
    let (m, n, k) = (9usize, 11usize, 5usize);
    let a = vec![1.0f32; m * k];
    let bm = vec![2.0f32; k * n];
    let at = b.upload_f32(&[m, k], &a).unwrap();
    let bt = b.upload_f32(&[k, n], &bm).unwrap();
    let ct = b.upload_f32(&[m, n], &vec![f32::NAN; m * n]).unwrap();
    let out = b.gemm_f32(&at, &bt, &ct, m, n, k, false, false, 1.0, 0.0).unwrap();
    assert!(b.download_f32(&out).unwrap().iter().all(|&v| v == 10.0));
}

/// Timing note (not an assertion): 3x3 conv 64->64 at 104x104, batch 1.
#[test]
fn perf_note_conv_64_to_64() {
    use std::time::Instant;
    let c = case("perf", 1, 64, 104, 104, 64, 3, 1, 1, 1, 1, 1);
    let (x, w, dy) = inputs(&c, 4242);
    for mode in [ConvMode::Direct, ConvMode::Gemm, ConvMode::Direct, ConvMode::Gemm] {
        let b = backend(mode);
        let xt = b.upload_f32(&[1, 64, 104, 104], &x).unwrap();
        let wt = b.upload_f32(&[64, 64, 3, 3], &w).unwrap();
        let dyt = b.upload_f32(&[1, 64, 104, 104], &dy).unwrap();
        let run = |which: usize| {
            match which {
                0 => drop(b.conv2d_f32(&xt, &wt, &[1, 64, 104, 104], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
                1 => drop(b.conv2d_backward_input_f32(&dyt, &wt, &[1, 64, 104, 104], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
                _ => drop(b.conv2d_backward_weights_f32(&xt, &dyt, &[64, 64, 3, 3], 64, 104, 104, 64, 3, 1, 1, 1, 1, 1, cfg()).unwrap()),
            }
        };
        for (which, name) in ["fwd", "dgrad", "wgrad"].iter().enumerate() {
            run(which); // warm-up (module/JIT, workspace alloc)
            let n = 5;
            let t = Instant::now();
            for _ in 0..n {
                run(which);
            }
            // download forces a sync on the default stream
            let _ = b.download_f32(&b.zeros_f32(&[1]).unwrap()).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            let gflops = 2.0 * 64.0 * 64.0 * 9.0 * 104.0 * 104.0 / (ms * 1e-3) / 1e9;
            println!("PERF {mode:?} {name}: {ms:.3} ms ({gflops:.0} GFLOP/s)");
        }
    }
}
