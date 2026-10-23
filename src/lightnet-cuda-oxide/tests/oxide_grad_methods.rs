//! Gradient-method shootout (cheap version) and shortcut finite-difference test.
//!
//! For a handful of ops this compares, on the GPU, (a) the C++-style hand-written backward
//! (only where it differs from the exact one: softmax passes `dy` through), (b) the exact
//! analytical backward kernel and (c) a finite-difference gradient built from forwards
//! (forward difference = n+1 forwards, central = 2n), all measured with `time_gpu_ms`
//! (interleaved rounds) and compared with an independent f64 host reference.
//!
//! The FD loss `L(x) = sum(dy * f(x))` is evaluated entirely on the device: elementwise
//! product via `scale_channels_f32` (per-element scale, `[1, m, 1, 1]`) and the sum via
//! `bias_backward_f32` (one channel), so the FD path pays no D2H inside the timed region.
//!
//! `grad_methods_agree_with_f64_reference_small` runs in the normal suite (tiny shapes).
//! The timed table is `#[ignore]`d:
//!   `... oxide test -- --manifest-path src/lightnet-cuda-oxide/Cargo.toml -j 2 -- --ignored --nocapture --test-threads=1 grad_method_shootout`
//! Env: `LIGHTNET_BENCH_ROUNDS` (default 5), `LIGHTNET_BENCH_TARGET_MS` (default 10).

use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use std::cell::OnceCell;

use lightnet_cuda_oxide::{OxideBackend, OxideError, OxideTensorF32};

type T = OxideTensorF32;
type Res<R> = Result<R, OxideError>;

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn backend() -> OxideBackend {
    OxideBackend::new().expect("create oxide backend")
}

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

    /// `n` pairwise distinct values spaced 0.05 apart in a random order (no maxpool ties/kinks).
    fn distinct(&mut self, n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n).map(|i| i as f32 * 0.05 - n as f32 * 0.025).collect();
        for i in (1..n).rev() {
            let j = (self.next(0.0, 1.0) * (i + 1) as f32) as usize;
            v.swap(i, j.min(i));
        }
        v
    }
}

// ------------------------------------------------------------------ test shortcut (FD)

/// `shortcut_add` (no weights) is `out = x + y`: both backward branches are the identity on
/// `dy` (C++ `backward_shortcut_multilayer` adds `delta` unchanged when there are no weights).
/// Check that against a central finite difference of the device forward in f64-accumulated loss.
#[test]
fn shortcut_add_backward_is_identity_and_matches_finite_difference() {
    let be = backend();
    let mut rng = Rng(0x5c07_0001);
    let n = 37;
    let x = rng.vec(n, -1.0, 1.0);
    let y = rng.vec(n, -1.0, 1.0);
    let dy = rng.vec(n, -1.0, 1.0);
    let (xt, yt) = (be.upload_f32(&[n], &x).unwrap(), be.upload_f32(&[n], &y).unwrap());
    let out = be.download_f32(&be.shortcut_add_f32(&xt, &yt, cfg()).unwrap()).unwrap();
    for i in 0..n {
        assert!((out[i] - (x[i] + y[i])).abs() <= 1e-6, "shortcut fwd[{i}]");
    }
    let loss = |a: &[f32], b: &[f32]| -> f64 {
        let o = be
            .download_f32(
                &be.shortcut_add_f32(&be.upload_f32(&[n], a).unwrap(), &be.upload_f32(&[n], b).unwrap(), cfg()).unwrap(),
            )
            .unwrap();
        o.iter().zip(&dy).map(|(v, d)| *v as f64 * *d as f64).sum()
    };
    let h = 0.25f32; // the op is linear: a large step is exact up to f32 rounding
    for i in 0..n {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[i] += h;
        xm[i] -= h;
        let fd_x = (loss(&xp, &y) - loss(&xm, &y)) / (2.0 * h as f64);
        let (mut yp, mut ym) = (y.clone(), y.clone());
        yp[i] += h;
        ym[i] -= h;
        let fd_y = (loss(&x, &yp) - loss(&x, &ym)) / (2.0 * h as f64);
        assert!((fd_x - dy[i] as f64).abs() <= 2e-3, "d shortcut/dx[{i}]: fd {fd_x}, analytical {}", dy[i]);
        assert!((fd_y - dy[i] as f64).abs() <= 2e-3, "d shortcut/dy[{i}]: fd {fd_y}, analytical {}", dy[i]);
    }
}

// ------------------------------------------------------------------ shootout cases

type Fwd<'a> = Box<dyn Fn(&OxideBackend, &T) -> Res<T> + 'a>;
type Bwd<'a> = Box<dyn Fn(&OxideBackend, &T, &T) -> Res<T> + 'a>;

struct Case<'a> {
    name: String,
    shape: Vec<usize>,
    x: Vec<f32>,
    dy: Vec<f32>,
    /// Number of elements of `dy` (= output size) and its tensor shape.
    m: usize,
    dy_shape: Vec<usize>,
    fwd: Fwd<'a>,
    /// Exact analytical backward `(x, dy) -> dx`.
    ana: Bwd<'a>,
    /// C++-style backward when it differs from the exact one.
    cpp: Option<Bwd<'a>>,
    /// f64 reference gradient of `sum(dy * f(x))`.
    reference: Vec<f64>,
}

fn act64(kind: F32ActivationKind, x: f64) -> (f64, f64) {
    match kind {
        F32ActivationKind::Leaky => (if x > 0.0 { x } else { 0.1 * x }, if x > 0.0 { 1.0 } else { 0.1 }),
        F32ActivationKind::Logistic => {
            let s = 1.0 / (1.0 + (-x).exp());
            (s, s * (1.0 - s))
        }
        F32ActivationKind::Mish => {
            let sp = (1.0 + x.exp()).ln();
            let t = sp.tanh();
            let sig = 1.0 / (1.0 + (-x).exp());
            (x * t, t + x * (1.0 - t * t) * sig)
        }
        _ => unreachable!(),
    }
}

fn act_case<'a>(rng: &mut Rng, kind: F32ActivationKind, n: usize) -> Case<'a> {
    // away from the leaky kink at 0
    let x: Vec<f32> = rng.vec(n, -3.0, 3.0).into_iter().map(|v| if v.abs() < 0.1 { v + 0.3 } else { v }).collect();
    let dy = rng.vec(n, -1.0, 1.0);
    let reference = (0..n).map(|i| act64(kind, x[i] as f64).1 * dy[i] as f64).collect();
    Case {
        name: format!("activation {kind:?}"),
        shape: vec![n],
        x,
        dy,
        m: n,
        dy_shape: vec![n],
        fwd: Box::new(move |be, x| be.activation_f32(x, kind, cfg())),
        ana: Box::new(move |be, x, dy| be.activation_backward_f32(x, dy, kind, cfg())),
        cpp: None,
        reference,
    }
}

fn softmax_case<'a>(rng: &mut Rng, n: usize) -> Case<'a> {
    let group = 8;
    let groups = n / group;
    let x = rng.vec(n, -2.0, 2.0);
    let dy = rng.vec(n, -1.0, 1.0);
    let mut reference = vec![0.0f64; n];
    for g in 0..groups {
        let r = g * group..(g + 1) * group;
        let mx = r.clone().map(|i| x[i] as f64).fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = r.clone().map(|i| (x[i] as f64 - mx).exp()).sum();
        let y: Vec<f64> = r.clone().map(|i| ((x[i] as f64 - mx).exp()) / sum).collect();
        let dot: f64 = y.iter().zip(r.clone()).map(|(yy, i)| yy * dy[i] as f64).sum();
        for (k, i) in r.enumerate() {
            reference[i] = y[k] * (dy[i] as f64 - dot);
        }
    }
    let shape = vec![groups, group];
    let (s1, s2, s3) = (shape.clone(), shape.clone(), shape.clone());
    Case {
        name: "softmax".into(),
        shape,
        x,
        dy,
        m: n,
        dy_shape: vec![n],
        fwd: Box::new(move |be, x| be.softmax_f32(x, &s1, group, cfg())),
        ana: Box::new(move |be, x, dy| {
            let y = be.softmax_f32(x, &s2, group, cfg())?;
            be.softmax_backward_f32(&y, dy, &s2, group, cfg())
        }),
        // C++: `axpy(delta)`; the cost layer already holds `dL/dlogits`, so softmax is a copy.
        cpp: Some(Box::new(move |be, _x, dy| {
            let z = be.zeros_f32(&s3)?;
            be.saxpy_f32(1.0, dy, &z, cfg())
        })),
        reference,
    }
}

/// FD w.r.t. the per-channel bias; `x` here holds the bias, the activation tensor is fixed.
fn bias_case<'a>(rng: &mut Rng, channels: usize) -> Case<'a> {
    let (batch, stride) = (4usize, 16usize);
    let total = batch * channels * stride;
    let act = rng.vec(total, -1.0, 1.0);
    let bias = rng.vec(channels, -1.0, 1.0);
    let dy = rng.vec(total, -1.0, 1.0);
    let mut reference = vec![0.0f64; channels];
    for i in 0..total {
        reference[(i / stride) % channels] += dy[i] as f64;
    }
    let act2 = act.clone();
    let cell: OnceCell<T> = OnceCell::new();
    Case {
        name: "bias (d/dbias)".into(),
        shape: vec![channels],
        x: bias,
        dy,
        m: total,
        dy_shape: vec![total],
        fwd: Box::new(move |be, b| {
            // uploaded once: the FD path must not pay an H2D per forward
            if cell.get().is_none() {
                let _ = cell.set(be.upload_f32(&[total], &act2)?);
            }
            be.bias_add_f32(cell.get().unwrap(), b, channels, stride, cfg())
        }),
        ana: Box::new(move |be, _b, dy| be.bias_backward_f32(dy, channels, stride, cfg())),
        cpp: None,
        reference,
    }
}

fn maxpool_case<'a>(rng: &mut Rng, n: usize) -> Case<'a> {
    let (c, hw) = if n >= 512 { (2usize, 16usize) } else { (1usize, (n as f64).sqrt() as usize) };
    let (h, w) = (hw, hw);
    let n = c * h * w;
    let (oh, ow) = (h / 2, w / 2);
    let m = c * oh * ow;
    let x = rng.distinct(n);
    let dy = rng.vec(m, -1.0, 1.0);
    let mut reference = vec![0.0f64; n];
    for k in 0..c {
        for i in 0..oh {
            for j in 0..ow {
                let mut best = (f32::NEG_INFINITY, 0usize);
                for dyi in 0..2 {
                    for dxi in 0..2 {
                        let idx = (k * h + 2 * i + dyi) * w + 2 * j + dxi;
                        if x[idx] > best.0 {
                            best = (x[idx], idx);
                        }
                    }
                }
                reference[best.1] += dy[(k * oh + i) * ow + j] as f64;
            }
        }
    }
    let shape = vec![1, c, h, w];
    let oshape = vec![1, c, oh, ow];
    Case {
        name: "maxpool 2x2/2".into(),
        shape,
        x,
        dy,
        m,
        dy_shape: oshape.clone(),
        fwd: Box::new(move |be, x| be.maxpool_f32(x, &oshape, c, h, w, 2, 2, 2, 0, cfg())),
        ana: Box::new(move |be, x, dy| be.maxpool_backward_f32(x, dy, c, h, w, 2, 2, 2, 0, cfg())),
        cpp: None,
        reference,
    }
}

fn shortcut_case<'a>(rng: &mut Rng, n: usize) -> Case<'a> {
    let x = rng.vec(n, -1.0, 1.0);
    let other = rng.vec(n, -1.0, 1.0);
    let dy = rng.vec(n, -1.0, 1.0);
    let reference = dy.iter().map(|&v| v as f64).collect();
    let cell: OnceCell<T> = OnceCell::new();
    Case {
        name: "shortcut (x + y)".into(),
        shape: vec![n],
        x,
        dy,
        m: n,
        dy_shape: vec![n],
        fwd: Box::new(move |be, x| {
            if cell.get().is_none() {
                let _ = cell.set(be.upload_f32(&[n], &other)?);
            }
            be.shortcut_add_f32(x, cell.get().unwrap(), cfg())
        }),
        // exact == C++ for the unweighted shortcut: both branches receive `dy` unchanged
        ana: Box::new(move |be, _x, dy| {
            let z = be.zeros_f32(&[n])?;
            be.saxpy_f32(1.0, dy, &z, cfg())
        }),
        cpp: None,
        reference,
    }
}

// ------------------------------------------------------------------ FD on the device

struct FdCtx {
    units: Vec<T>,
    dy_t: T,
    m: usize,
}

/// `sum(dy * y)` as a 1-element device tensor (2 launches + 2 for the reduction).
fn device_loss(be: &OxideBackend, ctx: &FdCtx, y: &T) -> Res<T> {
    let prod = be.scale_channels_f32(&ctx.dy_t, y, &[1, ctx.m, 1, 1], false, cfg())?;
    be.bias_backward_f32(&prod, 1, ctx.m, cfg())
}

/// Enqueue the FD forwards; returns `(base, plus_i, minus_i)` loss tensors (`minus` empty for
/// the forward difference: n+1 forwards; central: 2n forwards).
fn fd_enqueue(be: &OxideBackend, case: &Case, ctx: &FdCtx, x_t: &T, h: f32, central: bool) -> Res<(Option<T>, Vec<T>, Vec<T>)> {
    let base = if central { None } else { Some(device_loss(be, ctx, &(case.fwd)(be, x_t)?)?) };
    let (mut plus, mut minus) = (Vec::new(), Vec::new());
    for e in &ctx.units {
        let xp = be.saxpy_f32(h, e, x_t, cfg())?;
        plus.push(device_loss(be, ctx, &(case.fwd)(be, &xp)?)?);
        if central {
            let xm = be.saxpy_f32(-h, e, x_t, cfg())?;
            minus.push(device_loss(be, ctx, &(case.fwd)(be, &xm)?)?);
        }
    }
    Ok((base, plus, minus))
}

fn fd_gradient(be: &OxideBackend, r: (Option<T>, Vec<T>, Vec<T>), h: f32) -> Vec<f32> {
    let dl = |t: &T| be.download_f32(t).unwrap()[0];
    let (base, plus, minus) = r;
    match base {
        Some(b) => {
            let b0 = dl(&b);
            plus.iter().map(|p| (dl(p) - b0) / h).collect()
        }
        None => plus.iter().zip(&minus).map(|(p, m)| (dl(p) - dl(m)) / (2.0 * h)).collect(),
    }
}

fn max_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter().zip(want).map(|(g, w)| (*g as f64 - w).abs()).fold(0.0, f64::max)
}

struct Row {
    op: String,
    n_inputs: usize,
    /// (variant label, median ms or NaN when not timed, max abs err vs f64 reference)
    variants: Vec<(String, f64, f64)>,
    forwards: (usize, usize),
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run_case(be: &OxideBackend, case: &Case, timed: Option<(usize, f64)>) -> Row {
    let n = case.x.len();
    let h = 1e-2f32;
    let x_t = be.upload_f32(&case.shape, &case.x).unwrap();
    let dy_t = be.upload_f32(&case.dy_shape, &case.dy).unwrap();
    let dy_flat = be.upload_f32(&[case.m], &case.dy).unwrap();
    let units = (0..n)
        .map(|i| {
            let mut u = vec![0.0f32; n];
            u[i] = 1.0;
            be.upload_f32(&case.shape, &u).unwrap()
        })
        .collect();
    let ctx = FdCtx { units, dy_t: dy_flat, m: case.m };

    // Variants: (label, enqueue closure).
    type Run<'r> = Box<dyn FnMut() -> Res<()> + 'r>;
    let mut labels: Vec<String> = Vec::new();
    let mut runs: Vec<Run> = Vec::new();
    let mut errs: Vec<f64> = Vec::new();

    let dx = be.download_f32(&(case.ana)(be, &x_t, &dy_t).unwrap()).unwrap();
    labels.push("analytical".into());
    errs.push(max_err(&dx, &case.reference));
    runs.push(Box::new(|| (case.ana)(be, &x_t, &dy_t).map(|_| ())));

    if let Some(cpp) = &case.cpp {
        let dxc = be.download_f32(&cpp(be, &x_t, &dy_t).unwrap()).unwrap();
        labels.push("C++ hand-written".into());
        errs.push(max_err(&dxc, &case.reference));
        runs.push(Box::new(|| cpp(be, &x_t, &dy_t).map(|_| ())));
    }

    for (label, central) in [("FD forward (n+1)", false), ("FD central (2n)", true)] {
        let g = fd_gradient(be, fd_enqueue(be, case, &ctx, &x_t, h, central).unwrap(), h);
        labels.push(label.into());
        errs.push(max_err(&g, &case.reference));
        let (xr, cr) = (&x_t, &ctx);
        runs.push(Box::new(move || fd_enqueue(be, case, cr, xr, h, central).map(|_| ())));
    }

    let mut ms = vec![f64::NAN; labels.len()];
    if let Some((rounds, target)) = timed {
        let iters: Vec<usize> = runs
            .iter_mut()
            .map(|r| ((target / be.time_gpu_ms(1, &mut **r).unwrap().max(1e-4)) as usize).clamp(1, 2000))
            .collect();
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); runs.len()];
        for _ in 0..rounds {
            for (i, r) in runs.iter_mut().enumerate() {
                samples[i].push(be.time_gpu_ms(iters[i], &mut **r).unwrap());
            }
        }
        for (i, s) in samples.iter_mut().enumerate() {
            ms[i] = median(s);
        }
    }
    Row {
        op: case.name.clone(),
        n_inputs: n,
        variants: labels.into_iter().zip(ms).zip(errs).map(|((l, t), e)| (l, t, e)).collect(),
        forwards: (n + 1, 2 * n),
    }
}

fn cases<'a>(rng: &mut Rng, n: usize) -> Vec<Case<'a>> {
    vec![
        act_case(rng, F32ActivationKind::Leaky, n),
        act_case(rng, F32ActivationKind::Logistic, n),
        act_case(rng, F32ActivationKind::Mish, n),
        softmax_case(rng, n),
        bias_case(rng, (n / 4).max(1)),
        maxpool_case(rng, n),
        shortcut_case(rng, n),
    ]
}

#[test]
fn grad_methods_agree_with_f64_reference_small() {
    let be = backend();
    let mut rng = Rng(0x6ead_0001);
    for case in cases(&mut rng, 16) {
        let row = run_case(&be, &case, None);
        for (label, _, err) in &row.variants {
            let tol = match label.as_str() {
                "analytical" => 1e-5,
                "C++ hand-written" => f64::INFINITY, // softmax passthrough is not the Jacobian
                "FD forward (n+1)" => 0.15,          // O(h) truncation, h = 1e-2
                _ => 0.03,                           // central: O(h^2) + f32 round-off / h
            };
            assert!(*err <= tol, "{}: {label} max err {err} > {tol}", row.op);
        }
    }
}

#[test]
#[ignore]
fn grad_method_shootout() {
    let be = backend();
    let rounds = std::env::var("LIGHTNET_BENCH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let target = std::env::var("LIGHTNET_BENCH_TARGET_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    println!("| op | inputs n | method | median ms | max abs err vs f64 | forwards |");
    println!("|---|---|---|---|---|---|");
    for n in [64usize, 512] {
        let mut rng = Rng(0x6ead_0100 + n as u64);
        for case in cases(&mut rng, n) {
            let row = run_case(&be, &case, Some((rounds, target)));
            for (label, ms, err) in &row.variants {
                let fw = match label.as_str() {
                    "FD forward (n+1)" => row.forwards.0.to_string(),
                    "FD central (2n)" => row.forwards.1.to_string(),
                    _ => "0".into(),
                };
                println!("| {} | {} | {} | {:.4} | {:.2e} | {} |", row.op, row.n_inputs, label, ms, err, fw);
            }
        }
    }
}
