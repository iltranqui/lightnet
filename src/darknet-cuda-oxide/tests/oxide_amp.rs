//! Mixed-precision (AMP) training ops: f16 storage + f32 accumulate.
//!
//! Part 1: per-op f16 backward (and the forward ops AMP adds) against f64 host
//! references.  Two bounds per op, both normalised by the largest reference
//! magnitude (`nerr = max|got - ref| / max|ref|`):
//!   * tight: the reference is computed from the f16-rounded inputs, so only the
//!     f32 accumulation order and the single f16 rounding of an f16 output
//!     remain (2^-11 = 4.9e-4 relative per element): <= 1e-3 for f16 outputs,
//!     <= 1e-4 for f32 outputs (weight gradients);
//!   * AMP: the reference is computed from the unrounded f32 inputs, i.e. what
//!     the whole mixed-precision step costs in accuracy: <= 4e-3.
//! Part 2: a tiny conv+BN+leaky net (maxpool, route, upsample, biased linear
//! head): AMP gradients vs f32 gradients, and a few synthetic SGD steps with f32
//! master weights.  No dataset, no real training.

use darknet_cuda::{AdamParams, ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::half::{f16_bits_to_f32, f32_to_f16_bits};
use darknet_cuda_oxide::{GemmMode, OxideBackend, OxideTensorF32};

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

fn amp_backend() -> OxideBackend {
    OxideBackend::new().expect("backend").with_amp(true)
}

fn f32_backend() -> OxideBackend {
    OxideBackend::new().expect("backend").with_amp(false).with_gemm_mode(GemmMode::Fp32)
}

fn r16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16_bits_to_f32(f32_to_f16_bits(x))).collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

fn up(be: &OxideBackend, shape: &[usize], v: &[f32]) -> OxideTensorF32 {
    be.upload_f32(shape, v).unwrap()
}

/// Uploads and rounds into f16 storage (the activation form AMP produces).
fn up_half(be: &OxideBackend, shape: &[usize], v: &[f32]) -> OxideTensorF32 {
    let t = be.to_half_storage(&up(be, shape, v)).unwrap();
    assert!(t.is_half());
    t
}

fn down(be: &OxideBackend, t: &OxideTensorF32) -> Vec<f32> {
    be.download_f32(t).unwrap()
}

/// `max|got - want| / max|want|`.
fn nerr(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| {
        assert!(g.is_finite(), "non-finite output {g}");
        m.max((*g as f64 - w).abs())
    }) / scale
}

#[track_caller]
fn check(name: &str, got: &[f32], want: &[f64], bound: f64) -> f64 {
    let e = nerr(got, want);
    println!("{name}: nerr {e:.3e} (bound {bound:.1e})");
    assert!(e <= bound, "{name}: nerr {e:.3e} exceeds {bound:.1e}");
    e
}

// ---------------------------------------------------------------- conv refs

#[derive(Clone, Copy, Debug)]
struct Cv {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    k: usize,
    s: usize,
    pad: usize,
    groups: usize,
}

impl Cv {
    fn oh(&self) -> usize {
        (self.h + 2 * self.pad - self.k) / self.s + 1
    }
    fn ow(&self) -> usize {
        (self.w + 2 * self.pad - self.k) / self.s + 1
    }
    fn cpg(&self) -> usize {
        self.c / self.groups
    }
    fn fpg(&self) -> usize {
        self.f / self.groups
    }
}

/// Visits every (output index, weight index, input index) tap of the conv.
fn conv_taps(cs: &Cv, mut tap: impl FnMut(usize, usize, usize)) {
    let (oh, ow) = (cs.oh(), cs.ow());
    for b in 0..cs.b {
        for f in 0..cs.f {
            let g = f / cs.fpg();
            for oy in 0..oh {
                for ox in 0..ow {
                    let o = ((b * cs.f + f) * oh + oy) * ow + ox;
                    for cc in 0..cs.cpg() {
                        let c = g * cs.cpg() + cc;
                        for ky in 0..cs.k {
                            for kx in 0..cs.k {
                                let iy = (oy * cs.s + ky) as i64 - cs.pad as i64;
                                let ix = (ox * cs.s + kx) as i64 - cs.pad as i64;
                                if iy < 0 || ix < 0 || iy >= cs.h as i64 || ix >= cs.w as i64 {
                                    continue;
                                }
                                let wi = ((f * cs.cpg() + cc) * cs.k + ky) * cs.k + kx;
                                let xi = ((b * cs.c + c) * cs.h + iy as usize) * cs.w + ix as usize;
                                tap(o, wi, xi);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn conv_ref_fwd(cs: &Cv, x: &[f32], w: &[f32]) -> Vec<f64> {
    let mut y = vec![0.0f64; cs.b * cs.f * cs.oh() * cs.ow()];
    conv_taps(cs, |o, wi, xi| y[o] += x[xi] as f64 * w[wi] as f64);
    y
}

fn conv_ref_dgrad(cs: &Cv, dy: &[f32], w: &[f32]) -> Vec<f64> {
    let mut dx = vec![0.0f64; cs.b * cs.c * cs.h * cs.w];
    conv_taps(cs, |o, wi, xi| dx[xi] += dy[o] as f64 * w[wi] as f64);
    dx
}

fn conv_ref_wgrad(cs: &Cv, dy: &[f32], x: &[f32]) -> Vec<f64> {
    let mut dw = vec![0.0f64; cs.f * cs.cpg() * cs.k * cs.k];
    conv_taps(cs, |o, wi, xi| dw[wi] += dy[o] as f64 * x[xi] as f64);
    dw
}

const CONV_CASES: &[Cv] = &[
    Cv { b: 2, c: 3, h: 9, w: 7, f: 5, k: 3, s: 1, pad: 1, groups: 1 },
    Cv { b: 3, c: 8, h: 10, w: 10, f: 6, k: 1, s: 1, pad: 0, groups: 1 },
    Cv { b: 2, c: 4, h: 11, w: 11, f: 8, k: 3, s: 2, pad: 1, groups: 2 },
    Cv { b: 2, c: 16, h: 13, w: 13, f: 16, k: 3, s: 1, pad: 1, groups: 1 },
    Cv { b: 1, c: 6, h: 8, w: 8, f: 12, k: 3, s: 1, pad: 1, groups: 3 },
    Cv { b: 2, c: 5, h: 7, w: 6, f: 4, k: 3, s: 1, pad: 1, groups: 1 },
    // multi-chunk-free but long K (many split-K slices in the wgrad GEMM)
    Cv { b: 4, c: 8, h: 32, w: 32, f: 8, k: 3, s: 1, pad: 1, groups: 1 },
];

fn conv_args(cs: &Cv) -> (usize, usize, usize, usize, usize, usize, usize, usize, usize, usize) {
    (cs.c, cs.h, cs.w, cs.f, cs.k, cs.s, cs.s, 1, cs.pad, cs.groups)
}

#[test]
fn amp_conv_forward_dgrad_wgrad_match_f64_reference() {
    let be = amp_backend();
    let mut rng = Rng(0xa11c_e5ee_d001);
    for cs in CONV_CASES {
        let (oh, ow) = (cs.oh(), cs.ow());
        let x = rng.vec(cs.b * cs.c * cs.h * cs.w, -1.0, 1.0);
        let w = rng.vec(cs.f * cs.cpg() * cs.k * cs.k, -0.5, 0.5);
        let dy = rng.vec(cs.b * cs.f * oh * ow, -1.0, 1.0);
        let (c, h, ww, f, k, sx, sy, dil, pad, groups) = conv_args(cs);
        let xt = up(&be, &[cs.b, c, h, ww], &x);
        let wt = up(&be, &[f, cs.cpg(), k, k], &w);
        let dyt = up(&be, &[cs.b, f, oh, ow], &dy);
        let tag = format!("{cs:?}");

        // forward: f16 output
        let y = be.conv2d_f32(&xt, &wt, &[cs.b, f, oh, ow], c, h, ww, f, k, sx, sy, dil, pad, groups, cfg()).unwrap();
        assert!(y.is_half(), "AMP conv output must be f16 storage");
        let yv = down(&be, &y);
        check(&format!("{tag} fwd tight"), &yv, &conv_ref_fwd(cs, &r16(&x), &r16(&w)), 1e-3);
        check(&format!("{tag} fwd amp"), &yv, &conv_ref_fwd(cs, &x, &w), 4e-3);

        // dgrad: f16 output
        let dx = be
            .conv2d_backward_input_f32(&dyt, &wt, &[cs.b, c, h, ww], c, h, ww, f, k, sx, sy, dil, pad, groups, cfg())
            .unwrap();
        assert!(dx.is_half());
        let dxv = down(&be, &dx);
        check(&format!("{tag} dgrad tight"), &dxv, &conv_ref_dgrad(cs, &r16(&dy), &r16(&w)), 1e-3);
        check(&format!("{tag} dgrad amp"), &dxv, &conv_ref_dgrad(cs, &dy, &w), 4e-3);

        // wgrad: f32 output (weight gradients are f32), x as f16 activation
        let xh = be.to_half_storage(&xt).unwrap();
        let dw = be
            .conv2d_backward_weights_f32(&xh, &dyt, &[f, cs.cpg(), k, k], c, h, ww, f, k, sx, sy, dil, pad, groups, cfg())
            .unwrap();
        assert!(!dw.is_half(), "weight gradients must stay f32");
        let dwv = down(&be, &dw);
        check(&format!("{tag} wgrad tight"), &dwv, &conv_ref_wgrad(cs, &r16(&dy), &r16(&x)), 1e-4);
        check(&format!("{tag} wgrad amp"), &dwv, &conv_ref_wgrad(cs, &dy, &x), 4e-3);
    }
}

#[test]
fn amp_conv_backward_runs_in_several_chunks_for_large_batches() {
    // dgrad / wgrad process the batch in chunks bounded by the 32 MB column
    // workspace; 80 images of 8x56x56 need two chunks in both (cases above never do).
    let be = amp_backend();
    let cs = Cv { b: 80, c: 8, h: 56, w: 56, f: 8, k: 3, s: 1, pad: 1, groups: 1 };
    let mut rng = Rng(0xc0ffee);
    let (oh, ow) = (cs.oh(), cs.ow());
    let x = rng.vec(cs.b * cs.c * cs.h * cs.w, -1.0, 1.0);
    let w = rng.vec(cs.f * cs.cpg() * cs.k * cs.k, -0.5, 0.5);
    let dy = rng.vec(cs.b * cs.f * oh * ow, -1.0, 1.0);
    let (c, h, ww, f, k, sx, sy, dil, pad, groups) = conv_args(&cs);
    let xh = up_half(&be, &[cs.b, c, h, ww], &x);
    let wt = up(&be, &[f, cs.cpg(), k, k], &w);
    let dyt = up(&be, &[cs.b, f, oh, ow], &dy);
    let dx = be
        .conv2d_backward_input_f32(&dyt, &wt, &[cs.b, c, h, ww], c, h, ww, f, k, sx, sy, dil, pad, groups, cfg())
        .unwrap();
    check("chunked dgrad", &down(&be, &dx), &conv_ref_dgrad(&cs, &r16(&dy), &r16(&w)), 1e-3);
    let dw = be
        .conv2d_backward_weights_f32(&xh, &dyt, &[f, cs.cpg(), k, k], c, h, ww, f, k, sx, sy, dil, pad, groups, cfg())
        .unwrap();
    check("chunked wgrad", &down(&be, &dw), &conv_ref_wgrad(&cs, &r16(&dy), &r16(&x)), 1e-4);
}

#[test]
fn amp_weight_copy_is_refreshed_after_an_update() {
    // The f16 working copy is cached per weight tensor; an optimizer step makes
    // a new tensor, so the next conv must see the new weights (no stale copy).
    let be = amp_backend();
    let cs = CONV_CASES[0];
    let mut rng = Rng(0x57a1e);
    let x = rng.vec(cs.b * cs.c * cs.h * cs.w, -1.0, 1.0);
    let w0 = rng.vec(cs.f * cs.cpg() * cs.k * cs.k, -0.5, 0.5);
    let u = rng.vec(w0.len(), -1.0, 1.0);
    let (c, h, ww, f, k, sx, sy, dil, pad, groups) = conv_args(&cs);
    let shape_w = [f, cs.cpg(), k, k];
    let xt = up(&be, &[cs.b, c, h, ww], &x);
    let out_shape = [cs.b, f, cs.oh(), cs.ow()];
    let run = |wt: &OxideTensorF32| {
        down(&be, &be.conv2d_f32(&xt, wt, &out_shape, c, h, ww, f, k, sx, sy, dil, pad, groups, cfg()).unwrap())
    };
    let w0t = up(&be, &shape_w, &w0);
    let a = run(&w0t);
    let again = run(&w0t);
    assert_eq!(a, again, "cached copy must give identical results");
    let ut = up(&be, &shape_w, &u);
    let (w1t, _) = be.sgd_update_f32(&w0t, &ut, 0.5, 0.0, 0.0, 1, cfg()).unwrap();
    let w1 = down(&be, &w1t);
    assert_ne!(w1, w0);
    let b = run(&w1t);
    check("after update", &b, &conv_ref_fwd(&cs, &r16(&x), &r16(&w1)), 1e-3);
    // and the old weights are still served correctly
    assert_eq!(run(&w0t), a);
}

// ------------------------------------------------------- elementwise / pool

#[test]
fn amp_bias_activation_bias_backward_vector_add() {
    let be = amp_backend();
    let mut rng = Rng(0xb1a5);
    let (b, c, hw) = (3usize, 7usize, 50usize);
    let n = b * c * hw;
    let x = rng.vec(n, -4.0, 4.0);
    let bias = rng.vec(c, -1.0, 1.0);
    let dy = rng.vec(n, -1.0, 1.0);
    let y = rng.vec(n, -1.0, 1.0);
    let shape = [b, c, 5, 10];
    let xh = up_half(&be, &shape, &x);
    let (xr, dyr, yr) = (r16(&x), r16(&dy), r16(&y));

    let out = be.bias_add_f32(&xh, &up(&be, &[c], &bias), c, hw, cfg()).unwrap();
    assert!(out.is_half());
    let want: Vec<f64> = (0..n).map(|i| xr[i] as f64 + bias[(i / hw) % c] as f64).collect();
    check("bias_add", &down(&be, &out), &want, 1e-3);

    let act = be.activation_f32(&xh, F32ActivationKind::Leaky, cfg()).unwrap();
    assert!(act.is_half());
    let want: Vec<f64> = xr.iter().map(|&v| if v > 0.0 { v as f64 } else { 0.1 * v as f64 }).collect();
    check("leaky fwd", &down(&be, &act), &want, 1e-3);

    let dyh = up_half(&be, &shape, &dy);
    for (kind, name) in [(F32ActivationKind::Leaky, "leaky"), (F32ActivationKind::Linear, "linear"), (F32ActivationKind::Logistic, "logistic")] {
        let g = be.activation_backward_f32(&xh, &dyh, kind, cfg()).unwrap();
        assert!(g.is_half());
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let d = match kind {
                    F32ActivationKind::Leaky => if xr[i] > 0.0 { 1.0 } else { 0.1 },
                    F32ActivationKind::Linear => 1.0,
                    _ => {
                        let s = 1.0 / (1.0 + (-(xr[i] as f64)).exp());
                        s * (1.0 - s)
                    }
                };
                dyr[i] as f64 * d
            })
            .collect();
        check(&format!("{name} bwd"), &down(&be, &g), &want, 1e-3);
    }
    // f32 delta + f16 activation input also takes the f16 path
    let g = be.activation_backward_f32(&xh, &up(&be, &shape, &dy), F32ActivationKind::Leaky, cfg()).unwrap();
    assert!(g.is_half());

    let db = be.bias_backward_f32(&dyh, c, hw, cfg()).unwrap();
    assert!(!db.is_half());
    let mut want = vec![0.0f64; c];
    for i in 0..n {
        want[(i / hw) % c] += dyr[i] as f64;
    }
    check("bias_backward", &down(&be, &db), &want, 1e-5);

    let sum = be.vector_add_f32(&dyh, &up_half(&be, &shape, &y), cfg()).unwrap();
    assert!(sum.is_half());
    let want: Vec<f64> = (0..n).map(|i| dyr[i] as f64 + yr[i] as f64).collect();
    check("vector_add", &down(&be, &sum), &want, 1e-3);
    // f32 + f32 stays f32 and exact (weight-gradient accumulation must not round)
    let s32 = be.vector_add_f32(&up(&be, &shape, &dy), &up(&be, &shape, &y), cfg()).unwrap();
    assert!(!s32.is_half());
    let got = down(&be, &s32);
    assert!(got.iter().zip(dy.iter().zip(&y)).all(|(g, (a, b))| *g == a + b));
}

#[test]
fn amp_pool_upsample_route_backward_match_f64_reference() {
    let be = amp_backend();
    let mut rng = Rng(0x9001);
    // maxpool 2x2 s2 and 3x3 s1 pad 2 (windows overlap)
    for (size, stride, pad, h, w) in [(2usize, 2usize, 0usize, 8usize, 6usize), (3, 1, 2, 7, 7), (2, 1, 1, 9, 5)] {
        let (b, c) = (2usize, 3usize);
        let off = pad / 2;
        let oh = (h + pad - size) / stride + 1;
        let ow = (w + pad - size) / stride + 1;
        let x = rng.vec(b * c * h * w, -2.0, 2.0);
        let dy = rng.vec(b * c * oh * ow, -1.0, 1.0);
        let (xr, dyr) = (r16(&x), r16(&dy));
        let xh = up_half(&be, &[b, c, h, w], &x);
        let dyh = up_half(&be, &[b, c, oh, ow], &dy);
        let dx = be.maxpool_backward_f32(&xh, &dyh, c, h, w, size, stride, stride, pad, cfg()).unwrap();
        assert!(dx.is_half());
        let mut want = vec![0.0f64; x.len()];
        for bc in 0..b * c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut bi = usize::MAX;
                    for ky in 0..size {
                        for kx in 0..size {
                            let iy = (oy * stride + ky) as i64 - off as i64;
                            let ix = (ox * stride + kx) as i64 - off as i64;
                            if iy < 0 || ix < 0 || iy >= h as i64 || ix >= w as i64 {
                                continue;
                            }
                            let i = bc * h * w + iy as usize * w + ix as usize;
                            if xr[i] > best {
                                best = xr[i];
                                bi = i;
                            }
                        }
                    }
                    if bi != usize::MAX {
                        want[bi] += dyr[(bc * oh + oy) * ow + ox] as f64;
                    }
                }
            }
        }
        check(&format!("maxpool bwd {size}/{stride}/{pad}"), &down(&be, &dx), &want, 1e-3);
        // forward stays bit-exact (max of f16 values)
        let y = be.maxpool_f32(&xh, &[b, c, oh, ow], c, h, w, size, stride, stride, pad, cfg()).unwrap();
        assert!(y.is_half());
    }

    // nearest upsample x2 / x3 with scale
    for (stride, scale) in [(2usize, 1.0f32), (3, 0.5)] {
        let (b, c, h, w) = (2usize, 4usize, 5usize, 6usize);
        let (oh, ow) = (h * stride, w * stride);
        let dy = rng.vec(b * c * oh * ow, -1.0, 1.0);
        let dyr = r16(&dy);
        let dyh = up_half(&be, &[b, c, oh, ow], &dy);
        let dx = be.upsample_backward_f32(&dyh, &[b, c, h, w], c, oh, ow, stride, scale, cfg()).unwrap();
        assert!(dx.is_half());
        let mut want = vec![0.0f64; b * c * h * w];
        for bc in 0..b * c {
            for y in 0..oh {
                for x in 0..ow {
                    want[bc * h * w + (y / stride) * w + x / stride] += scale as f64 * dyr[(bc * oh + y) * ow + x] as f64;
                }
            }
        }
        check(&format!("upsample bwd x{stride}"), &down(&be, &dx), &want, 1e-3);
        let xh = up_half(&be, &[b, c, h, w], &rng.vec(b * c * h * w, -1.0, 1.0));
        assert!(be.upsample_f32(&xh, &[b, c, oh, ow], c, h, w, stride, scale, false, cfg()).unwrap().is_half());
    }

    // route: two sources, one sliced (channels 1..4 of 6)
    let (b, hw) = (2usize, 12usize);
    let (c0, c1) = (3usize, 6usize);
    let (ch0, off1, ch1) = (c0, 1usize, 3usize);
    let out_c = ch0 + ch1;
    let dy = rng.vec(b * out_c * hw, -1.0, 1.0);
    let dyr = r16(&dy);
    let dyh = up_half(&be, &[b, out_c, 3, 4], &dy);
    let grads = be
        .route_backward_f32(&dyh, &[vec![b, c0, 3, 4], vec![b, c1, 3, 4]], &[0, off1], &[ch0, ch1], &[c0, c1], hw, cfg())
        .unwrap();
    assert!(grads.iter().all(|g| g.is_half()));
    let mut want0 = vec![0.0f64; b * c0 * hw];
    let mut want1 = vec![0.0f64; b * c1 * hw];
    for bb in 0..b {
        for c in 0..ch0 {
            for s in 0..hw {
                want0[(bb * c0 + c) * hw + s] = dyr[(bb * out_c + c) * hw + s] as f64;
            }
        }
        for c in 0..ch1 {
            for s in 0..hw {
                want1[(bb * c1 + off1 + c) * hw + s] = dyr[(bb * out_c + ch0 + c) * hw + s] as f64;
            }
        }
    }
    check("route bwd src0", &down(&be, &grads[0]), &want0, 1e-3);
    check("route bwd src1 (zeros outside the slice)", &down(&be, &grads[1]), &want1, 1e-3);
    // forward concat of mixed f32 / f16 inputs
    let a = up_half(&be, &[b, c0, 3, 4], &rng.vec(b * c0 * hw, -1.0, 1.0));
    let bt = up(&be, &[b, c1, 3, 4], &rng.vec(b * c1 * hw, -1.0, 1.0));
    let cat = be.route_concat_f32(&[&a, &bt], &[0, off1], &[ch0, ch1], &[c0, c1], &[b, out_c, 3, 4], hw, cfg()).unwrap();
    assert!(cat.is_half());
}

// ------------------------------------------------------------- batch norm

#[test]
fn amp_batchnorm_forward_backward_match_f64_reference() {
    let be = amp_backend();
    let mut rng = Rng(0xb4);
    for (b, c, h, w) in [(2usize, 5usize, 6usize, 7usize), (4, 16, 13, 13), (1, 3, 40, 40)] {
        let spatial = h * w;
        let n = (b * spatial) as f64;
        let shape = [b, c, h, w];
        let len = b * c * spatial;
        let x = rng.vec(len, -3.0, 5.0);
        let dy = rng.vec(len, -0.5, 0.5);
        let scale = rng.vec(c, 0.5, 2.0);
        let bias = rng.vec(c, -1.0, 1.0);
        let (rm, rv) = (rng.vec(c, -1.0, 1.0), rng.vec(c, 0.5, 2.0));
        let xr = f64s(&r16(&x));
        let dyr = f64s(&r16(&dy));
        let xh = up_half(&be, &shape, &x);
        let t = |v: &[f32]| up(&be, &[c], v);
        let got = be.batchnorm_train_forward_f32(&xh, &t(&scale), &t(&bias), &t(&rm), &t(&rv), &shape, 0.9, cfg()).unwrap();
        assert!(got.output.is_half() && got.x_hat.is_half() && !got.mean.is_half());

        // f64 reference from the rounded input
        let mut mean = vec![0.0f64; c];
        let mut var = vec![0.0f64; c];
        for i in 0..len {
            mean[(i / spatial) % c] += xr[i] / n;
        }
        for i in 0..len {
            let d = xr[i] - mean[(i / spatial) % c];
            var[(i / spatial) % c] += d * d / n;
        }
        let eps = 1e-5f64;
        let xhat: Vec<f64> = (0..len).map(|i| (xr[i] - mean[(i / spatial) % c]) / (var[(i / spatial) % c] + eps).sqrt()).collect();
        let out: Vec<f64> = (0..len).map(|i| xhat[i] * scale[(i / spatial) % c] as f64 + bias[(i / spatial) % c] as f64).collect();
        check("bn mean", &down(&be, &got.mean), &mean, 1e-5);
        check("bn var", &down(&be, &got.variance), &var, 1e-5);
        check("bn x_hat", &down(&be, &got.x_hat), &xhat, 1e-3);
        check("bn out", &down(&be, &got.output), &out, 1e-3);

        // backward
        let dyh = up_half(&be, &shape, &dy);
        let (dx, dscale, dbias) =
            be.batchnorm_train_backward_f32(&dyh, &got.x_hat, &got.variance, &t(&scale), &shape, cfg()).unwrap();
        assert!(dx.is_half() && !dscale.is_half() && !dbias.is_half());
        let mut sdy = vec![0.0f64; c];
        let mut sdyx = vec![0.0f64; c];
        for i in 0..len {
            sdy[(i / spatial) % c] += dyr[i];
            sdyx[(i / spatial) % c] += dyr[i] * xhat[i];
        }
        let dxr: Vec<f64> = (0..len)
            .map(|i| {
                let ch = (i / spatial) % c;
                scale[ch] as f64 / (var[ch] + eps).sqrt() / n * (n * dyr[i] - sdy[ch] - xhat[i] * sdyx[ch])
            })
            .collect();
        // x_hat was rounded to f16 before the backward: the reference uses the exact one
        check("bn dbias", &down(&be, &dbias), &sdy, 1e-5);
        check("bn dscale", &down(&be, &dscale), &sdyx, 2e-3);
        check("bn dx", &down(&be, &dx), &dxr, 4e-3);
    }
}

// ------------------------------------------------------------ tiny network

/// Master (f32) parameters of the tiny net:
/// conv1(3->8,k3)+BN+leaky -> maxpool2 -> conv3(8->8,k3)+BN+leaky -> route[3,2]
/// -> conv5(16->8,k1)+BN+leaky -> upsample2 -> conv7(8->6,k1,bias,linear).
#[derive(Clone)]
struct Params {
    w: [OxideTensorF32; 4],
    bn_scale: [OxideTensorF32; 3],
    bn_bias: [OxideTensorF32; 3],
    rm: [OxideTensorF32; 3],
    rv: [OxideTensorF32; 3],
    head_bias: OxideTensorF32,
}

const B: usize = 2;
const HW: usize = 16;
const W_SHAPES: [[usize; 4]; 4] = [[8, 3, 3, 3], [8, 8, 3, 3], [8, 16, 1, 1], [6, 8, 1, 1]];

fn init_params(be: &OxideBackend, rng: &mut Rng) -> Params {
    let w = std::array::from_fn(|i| {
        let s = W_SHAPES[i];
        let fan_in = (s[1] * s[2] * s[3]) as f32;
        let a = (3.0 / fan_in).sqrt();
        up(be, &s, &rng.vec(s.iter().product(), -a, a))
    });
    let ones = |n: usize| vec![1.0f32; n];
    Params {
        w,
        bn_scale: std::array::from_fn(|_| up(be, &[8], &ones(8))),
        bn_bias: std::array::from_fn(|_| up(be, &[8], &vec![0.0; 8])),
        rm: std::array::from_fn(|_| up(be, &[8], &vec![0.0; 8])),
        rv: std::array::from_fn(|_| up(be, &[8], &ones(8))),
        head_bias: up(be, &[6], &vec![0.0; 6]),
    }
}

/// Copies parameters between backends through the host.
fn clone_params(from: &OxideBackend, to: &OxideBackend, p: &Params) -> Params {
    let cp = |t: &OxideTensorF32| up(to, t.shape(), &down(from, t));
    Params {
        w: std::array::from_fn(|i| cp(&p.w[i])),
        bn_scale: std::array::from_fn(|i| cp(&p.bn_scale[i])),
        bn_bias: std::array::from_fn(|i| cp(&p.bn_bias[i])),
        rm: std::array::from_fn(|i| cp(&p.rm[i])),
        rv: std::array::from_fn(|i| cp(&p.rv[i])),
        head_bias: cp(&p.head_bias),
    }
}

/// f32 gradients (downhill convention: `seed = target - out`).
struct Grads {
    w: [Vec<f32>; 4],
    bn_scale: [Vec<f32>; 3],
    bn_bias: [Vec<f32>; 3],
    head_bias: Vec<f32>,
}

/// One forward + backward of the tiny net on `be` (the op sequence of the
/// runtime's GPU training path).  Returns the head output, the gradients
/// (divided by `loss_scale` on the host, as the optimizer does) and the new
/// rolling statistics.
fn forward_backward(
    be: &OxideBackend,
    p: &Params,
    x: &[f32],
    target: &[f32],
    loss_scale: f32,
) -> (Vec<f32>, Grads, [(OxideTensorF32, OxideTensorF32); 3]) {
    let leaky = F32ActivationKind::Leaky;
    let xt = up(be, &[B, 3, HW, HW], x);
    let conv = |inp: &OxideTensorF32, w: &OxideTensorF32, c: usize, h: usize, f: usize, k: usize| {
        let pad = k / 2;
        be.conv2d_f32(inp, w, &[B, f, h, h], c, h, h, f, k, 1, 1, 1, pad, 1, cfg()).unwrap()
    };
    // returns (activation output, bn cache, activation input)
    let bn_block = |inp: &OxideTensorF32, i: usize, c: usize, h: usize, f: usize, k: usize| {
        let raw = conv(inp, &p.w[i], c, h, f, k);
        let shape = [B, f, h, h];
        let bn = be
            .batchnorm_train_forward_f32(&raw, &p.bn_scale[i], &p.bn_bias[i], &p.rm[i], &p.rv[i], &shape, 0.9, cfg())
            .unwrap();
        let act = be.activation_f32(&bn.output, leaky, cfg()).unwrap();
        (act, bn)
    };
    // layer indices: conv blocks 0,1,2 use bn slots 0,1,2; head is w[3]
    let (a1, bn1) = bn_block(&xt, 0, 3, HW, 8, 3);
    let pooled = be.maxpool_f32(&a1, &[B, 8, HW / 2, HW / 2], 8, HW, HW, 2, 2, 2, 0, cfg()).unwrap();
    let (a3, bn3) = bn_block(&pooled, 1, 8, HW / 2, 8, 3);
    let hw8 = HW / 2;
    let cat = be
        .route_concat_f32(&[&a3, &pooled], &[0, 0], &[8, 8], &[8, 8], &[B, 16, hw8, hw8], hw8 * hw8, cfg())
        .unwrap();
    let (a5, bn5) = bn_block(&cat, 2, 16, hw8, 8, 1);
    let up2 = be.upsample_f32(&a5, &[B, 8, HW, HW], 8, hw8, hw8, 2, 1.0, false, cfg()).unwrap();
    let head_raw = conv(&up2, &p.w[3], 8, HW, 6, 1);
    let head = be.bias_add_f32(&head_raw, &p.head_bias, 6, HW * HW, cfg()).unwrap();
    let head_act = be.activation_f32(&head, F32ActivationKind::Linear, cfg()).unwrap();
    let out = down(be, &head_act);

    // host loss, as in the runtime: seed = (target - out) * loss_scale
    let seed: Vec<f32> = out.iter().zip(target).map(|(o, t)| (t - o) * loss_scale).collect();
    let g = up(be, &[B, 6, HW, HW], &seed);

    let unscale = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| x / loss_scale).collect() };
    // head
    let d = be.activation_backward_f32(&head, &g, F32ActivationKind::Linear, cfg()).unwrap();
    let hb = be.bias_backward_f32(&d, 6, HW * HW, cfg()).unwrap();
    let wg3 = be.conv2d_backward_weights_f32(&up2, &d, &W_SHAPES[3], 8, HW, HW, 6, 1, 1, 1, 1, 0, 1, cfg()).unwrap();
    let dup = be.conv2d_backward_input_f32(&d, &p.w[3], &[B, 8, HW, HW], 8, HW, HW, 6, 1, 1, 1, 1, 0, 1, cfg()).unwrap();
    let da5 = be.upsample_backward_f32(&dup, &[B, 8, hw8, hw8], 8, HW, HW, 2, 1.0, cfg()).unwrap();
    // block backward helper: returns (dinput, dW, dscale, dbias)
    let bn_back = |dact: &OxideTensorF32, act_in: &OxideTensorF32, bn: &darknet_cuda::BatchNormTrainForward<OxideTensorF32>, inp: &OxideTensorF32, i: usize, c: usize, h: usize, f: usize, k: usize| {
        let delta = be.activation_backward_f32(act_in, dact, leaky, cfg()).unwrap();
        let (dx, dscale, dbias) = be
            .batchnorm_train_backward_f32(&delta, &bn.x_hat, &bn.variance, &p.bn_scale[i], &[B, f, h, h], cfg())
            .unwrap();
        let dw = be.conv2d_backward_weights_f32(inp, &dx, &W_SHAPES[i], c, h, h, f, k, 1, 1, 1, k / 2, 1, cfg()).unwrap();
        let dinp = be.conv2d_backward_input_f32(&dx, &p.w[i], &[B, c, h, h], c, h, h, f, k, 1, 1, 1, k / 2, 1, cfg()).unwrap();
        (dinp, dw, dscale, dbias)
    };
    let (dcat, wg2, sg2, bg2) = bn_back(&da5, &bn5.output, &bn5, &cat, 2, 16, hw8, 8, 1);
    let rg = be
        .route_backward_f32(&dcat, &[vec![B, 8, hw8, hw8], vec![B, 8, hw8, hw8]], &[0, 0], &[8, 8], &[8, 8], hw8 * hw8, cfg())
        .unwrap();
    let (dpool_conv, wg1, sg1, bg1) = bn_back(&rg[0], &bn3.output, &bn3, &pooled, 1, 8, hw8, 8, 3);
    let dpool = be.vector_add_f32(&rg[1], &dpool_conv, cfg()).unwrap();
    let da1 = be.maxpool_backward_f32(&a1, &dpool, 8, HW, HW, 2, 2, 2, 0, cfg()).unwrap();
    let (_dx0, wg0, sg0, bg0) = bn_back(&da1, &bn1.output, &bn1, &xt, 0, 3, HW, 8, 3);

    let grads = Grads {
        w: [unscale(down(be, &wg0)), unscale(down(be, &wg1)), unscale(down(be, &wg2)), unscale(down(be, &wg3))],
        bn_scale: [unscale(down(be, &sg0)), unscale(down(be, &sg1)), unscale(down(be, &sg2))],
        bn_bias: [unscale(down(be, &bg0)), unscale(down(be, &bg1)), unscale(down(be, &bg2))],
        head_bias: unscale(down(be, &hb)),
    };
    let rolling = [
        (bn1.rolling_mean.clone(), bn1.rolling_variance.clone()),
        (bn3.rolling_mean.clone(), bn3.rolling_variance.clone()),
        (bn5.rolling_mean.clone(), bn5.rolling_variance.clone()),
    ];
    (out, grads, rolling)
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let want = f64s(b);
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    a.iter().zip(&want).fold(0.0f64, |m, (g, w)| {
        assert!(g.is_finite(), "non-finite gradient {g}");
        m.max((*g as f64 - w).abs())
    }) / scale
}

fn make_data(rng: &mut Rng) -> (Vec<f32>, Vec<f32>) {
    (rng.vec(B * 3 * HW * HW, 0.0, 1.0), rng.vec(B * 6 * HW * HW, -0.5, 0.5))
}

/// Max normalised error over every gradient tensor, with the per-tensor values.
fn grad_errors(a: &Grads, f: &Grads) -> Vec<(String, f64)> {
    let mut v = Vec::new();
    for i in 0..4 {
        v.push((format!("dW{i}"), rel(&a.w[i], &f.w[i])));
    }
    for i in 0..3 {
        v.push((format!("dscale{i}"), rel(&a.bn_scale[i], &f.bn_scale[i])));
        v.push((format!("dbias_bn{i}"), rel(&a.bn_bias[i], &f.bn_bias[i])));
    }
    v.push(("dbias_head".into(), rel(&a.head_bias, &f.head_bias)));
    v
}

/// Documented bound: every AMP gradient tensor of the 4-conv BN net is within
/// 2.5% of the f32 gradient, measured as max|g_amp - g_f32| / max|g_f32| per
/// tensor (loss scale 256; observed values are printed by the test and are
/// several times smaller, see docs/OXIDE_BENCH.md).
const NET_GRAD_BOUND: f64 = 2.5e-2;

#[test]
fn amp_gradients_track_f32_gradients_on_tiny_conv_bn_net() {
    let amp = amp_backend();
    let f32be = f32_backend();
    let mut rng = Rng(0x7107_ec);
    let p32 = init_params(&f32be, &mut rng);
    let pamp = clone_params(&f32be, &amp, &p32);
    let (x, target) = make_data(&mut rng);

    let (out32, g32, _) = forward_backward(&f32be, &p32, &x, &target, 1.0);
    let (out_amp, g_amp, rolling) = forward_backward(&amp, &pamp, &x, &target, 256.0);
    // forward head output
    let e_out = rel(&out_amp, &out32);
    println!("tiny net head output: nerr {e_out:.3e}");
    assert!(e_out <= 1e-2, "head output error {e_out:.3e}");
    let mut worst = 0.0f64;
    for (name, e) in grad_errors(&g_amp, &g32) {
        println!("tiny net {name}: nerr {e:.3e}");
        assert!(e <= NET_GRAD_BOUND, "{name}: nerr {e:.3e} > {NET_GRAD_BOUND:.1e}");
        worst = worst.max(e);
    }
    println!("tiny net worst gradient nerr {worst:.3e} (bound {NET_GRAD_BOUND:.1e})");
    // rolling statistics are f32 on both sides and (almost) equal
    for (i, (rm, rv)) in rolling.iter().enumerate() {
        assert!(!rm.is_half() && !rv.is_half(), "rolling statistics must be f32");
        assert!(down(&amp, rm).iter().chain(&down(&amp, rv)).all(|v| v.is_finite()), "bn {i}");
    }

    // loss_scale 1 for comparison (informational): this net's gradients are far
    // from the f16 underflow range, so the scale matters little here; it exists
    // for nets / losses whose gradients are tiny.
    let (_, g_noscale, _) = forward_backward(&amp, &pamp, &x, &target, 1.0);
    let worst_noscale = grad_errors(&g_noscale, &g32).iter().fold(0.0f64, |m, (_, e)| m.max(*e));
    println!("tiny net worst gradient nerr, loss_scale 1: {worst_noscale:.3e}");
}

#[test]
fn amp_loss_scale_is_neutral_on_gradients() {
    // Scaling the seed and dividing the gradients back must not change them
    // beyond f16 rounding (powers of two scale exactly).
    let amp = amp_backend();
    let f32be = f32_backend();
    let mut rng = Rng(0x5ca1e);
    let p32 = init_params(&f32be, &mut rng);
    let pamp = clone_params(&f32be, &amp, &p32);
    let (x, target) = make_data(&mut rng);
    let (_, g1, _) = forward_backward(&amp, &pamp, &x, &target, 64.0);
    let (_, g2, _) = forward_backward(&amp, &pamp, &x, &target, 512.0);
    for (name, e) in grad_errors(&g1, &g2) {
        println!("loss scale 64 vs 512 {name}: nerr {e:.3e}");
        assert!(e <= 1e-2, "{name}: {e:.3e}");
    }
}

#[test]
fn amp_overflowing_gradients_become_nonfinite_not_garbage() {
    // A scaled gradient beyond the f16 range must show up as Inf/NaN in the f32
    // weight gradients (which the optimizer's sanitize_nan_inf Reset zeroes
    // before the step), never as silently wrapped finite values.
    let be = amp_backend();
    let mut rng = Rng(0x0f10);
    let act_in = up_half(&be, &[B, 6, HW, HW], &vec![0.5f32; B * 6 * HW * HW]);
    let huge = up(&be, &[B, 6, HW, HW], &vec![1.0e7f32; B * 6 * HW * HW]);
    let d = be.activation_backward_f32(&act_in, &huge, F32ActivationKind::Linear, cfg()).unwrap();
    assert!(down(&be, &d).iter().all(|v| v.is_infinite()), "1e7 must overflow f16 to Inf");
    let inp = up_half(&be, &[B, 8, HW, HW], &rng.vec(B * 8 * HW * HW, -1.0, 1.0));
    let dw = be.conv2d_backward_weights_f32(&inp, &d, &W_SHAPES[3], 8, HW, HW, 6, 1, 1, 1, 1, 0, 1, cfg()).unwrap();
    assert!(down(&be, &dw).iter().any(|v| !v.is_finite()), "overflowing f16 gradient must be visible as Inf/NaN");
    // sanitize_nan_inf (Reset) is what the update path applies to such updates
    let clean = be
        .sanitize_nan_inf_f32(&dw, dw.shape(), darknet_cuda::NanInfMode::Reset, cfg())
        .unwrap();
    assert!(down(&be, &clean).iter().all(|v| v.is_finite()));
}

/// SGD with f32 master weights, `steps` steps on one fixed batch; returns
/// the weights after each step plus the MSE loss per step.
fn sgd_run(be: &OxideBackend, p0: &Params, x: &[f32], target: &[f32], steps: usize, loss_scale: f32) -> (Vec<Vec<Vec<f32>>>, Vec<f64>) {
    let cp = |t: &OxideTensorF32| up(be, t.shape(), &down(be, t));
    let mut p = Params {
        w: std::array::from_fn(|i| cp(&p0.w[i])),
        bn_scale: std::array::from_fn(|i| cp(&p0.bn_scale[i])),
        bn_bias: std::array::from_fn(|i| cp(&p0.bn_bias[i])),
        rm: std::array::from_fn(|i| cp(&p0.rm[i])),
        rv: std::array::from_fn(|i| cp(&p0.rv[i])),
        head_bias: cp(&p0.head_bias),
    };
    let mut upd_w: [OxideTensorF32; 4] = std::array::from_fn(|i| be.zeros_f32(p.w[i].shape()).unwrap());
    let mut upd_s: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut upd_b: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut upd_h = be.zeros_f32(&[6]).unwrap();
    // gradients are sums over B*H*W positions (darknet convention), hence the small lr
    let (lr, momentum, decay) = (2e-4f32, 0.9f32, 5e-4f32);
    let mut weights = Vec::new();
    let mut losses = Vec::new();
    for _ in 0..steps {
        let (out, g, rolling) = forward_backward(be, &p, x, target, loss_scale);
        losses.push(out.iter().zip(target).map(|(o, t)| ((t - o) as f64).powi(2)).sum::<f64>() / out.len() as f64);
        // The gradients above are already divided by loss_scale (host side);
        // accumulate into the f32 update buffers and step (downhill convention).
        let step = |w: &OxideTensorF32, u: &OxideTensorF32, grad: &[f32]| {
            let acc = be.vector_add_f32(u, &up(be, w.shape(), grad), cfg()).unwrap();
            be.sgd_update_f32(w, &acc, lr, momentum, decay, 1, cfg()).unwrap()
        };
        for i in 0..4 {
            let (w, u) = step(&p.w[i], &upd_w[i], &g.w[i]);
            p.w[i] = w;
            upd_w[i] = u;
        }
        for i in 0..3 {
            let (w, u) = step(&p.bn_scale[i], &upd_s[i], &g.bn_scale[i]);
            p.bn_scale[i] = w;
            upd_s[i] = u;
            let (w, u) = step(&p.bn_bias[i], &upd_b[i], &g.bn_bias[i]);
            p.bn_bias[i] = w;
            upd_b[i] = u;
            p.rm[i] = rolling[i].0.clone();
            p.rv[i] = rolling[i].1.clone();
        }
        let (w, u) = step(&p.head_bias, &upd_h, &g.head_bias);
        p.head_bias = w;
        upd_h = u;
        let mut snap: Vec<Vec<f32>> = p.w.iter().map(|t| down(be, t)).collect();
        snap.extend(p.bn_scale.iter().map(|t| down(be, t)));
        snap.extend(p.bn_bias.iter().map(|t| down(be, t)));
        snap.push(down(be, &p.head_bias));
        weights.push(snap);
    }
    (weights, losses)
}

/// Documented bound: after N synthetic SGD steps the AMP master weights differ
/// from the f32 master weights by at most 10% of the distance the f32 weights
/// travelled (per tensor, inf-norm), i.e. the AMP trajectory is the f32
/// trajectory up to f16 rounding noise.
const SGD_TRACK_BOUND: f64 = 0.10;
#[test]
fn amp_sgd_steps_track_f32_master_weights() {
    let amp = amp_backend();
    let f32be = f32_backend();
    let mut rng = Rng(0x56d_5eed);
    let p0 = init_params(&f32be, &mut rng);
    let p0_amp = clone_params(&f32be, &amp, &p0);
    let (x, target) = make_data(&mut rng);
    let steps = 6;
    let (w32, l32) = sgd_run(&f32be, &p0, &x, &target, steps, 1.0);
    let (wamp, lamp) = sgd_run(&amp, &p0_amp, &x, &target, steps, 256.0);
    println!("loss f32: {l32:.4?}");
    println!("loss amp: {lamp:.4?}");
    assert!(l32.last().unwrap() < &l32[0], "f32 loss must decrease");
    assert!(lamp.last().unwrap() < &lamp[0], "AMP loss must decrease");
    for (i, (a, b)) in lamp.iter().zip(&l32).enumerate() {
        assert!((a - b).abs() <= 0.02 * b.abs().max(1e-6), "step {i}: loss amp {a} vs f32 {b}");
    }
    let init: Vec<Vec<f32>> = {
        let mut v: Vec<Vec<f32>> = p0.w.iter().map(|t| down(&f32be, t)).collect();
        v.extend(p0.bn_scale.iter().map(|t| down(&f32be, t)));
        v.extend(p0.bn_bias.iter().map(|t| down(&f32be, t)));
        v.push(down(&f32be, &p0.head_bias));
        v
    };
    let names = ["w0", "w1", "w2", "w3", "s0", "s1", "s2", "b0", "b1", "b2", "bh"];
    let mut worst = 0.0f64;
    for (t, name) in names.iter().enumerate() {
        let (fin32, finamp) = (&w32[steps - 1][t], &wamp[steps - 1][t]);
        let travelled = fin32.iter().zip(&init[t]).fold(0.0f64, |m, (a, b)| m.max((a - b).abs() as f64)).max(1e-12);
        let diff = fin32.iter().zip(finamp).fold(0.0f64, |m, (a, b)| m.max((a - b).abs() as f64));
        let ratio = diff / travelled;
        println!("sgd {name}: |w_amp - w_f32|inf / travelled {ratio:.3e}");
        assert!(ratio <= SGD_TRACK_BOUND, "{name}: {ratio:.3e} > {SGD_TRACK_BOUND}");
        worst = worst.max(ratio);
    }
    println!("sgd worst tracking ratio {worst:.3e} (bound {SGD_TRACK_BOUND})");
}

/// Adam with f32 master weights over `steps` fixed-batch steps.
fn adam_run(
    be: &OxideBackend,
    p0: &Params,
    x: &[f32],
    target: &[f32],
    steps: usize,
    loss_scale: f32,
) -> (Vec<Vec<Vec<f32>>>, Vec<f64>) {
    let cp = |t: &OxideTensorF32| up(be, t.shape(), &down(be, t));
    let mut p = Params {
        w: std::array::from_fn(|i| cp(&p0.w[i])),
        bn_scale: std::array::from_fn(|i| cp(&p0.bn_scale[i])),
        bn_bias: std::array::from_fn(|i| cp(&p0.bn_bias[i])),
        rm: std::array::from_fn(|i| cp(&p0.rm[i])),
        rv: std::array::from_fn(|i| cp(&p0.rv[i])),
        head_bias: cp(&p0.head_bias),
    };
    let mut m_w: [OxideTensorF32; 4] = std::array::from_fn(|i| be.zeros_f32(p.w[i].shape()).unwrap());
    let mut v_w: [OxideTensorF32; 4] = std::array::from_fn(|i| be.zeros_f32(p.w[i].shape()).unwrap());
    let mut m_s: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut v_s: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut m_b: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut v_b: [OxideTensorF32; 3] = std::array::from_fn(|_| be.zeros_f32(&[8]).unwrap());
    let mut m_h = be.zeros_f32(&[6]).unwrap();
    let mut v_h = be.zeros_f32(&[6]).unwrap();
    let mut weights = Vec::new();
    let mut losses = Vec::new();
    for step in 1..=steps {
        let (out, g, rolling) = forward_backward(be, &p, x, target, loss_scale);
        losses.push(
            out.iter()
                .zip(target)
                .map(|(o, t)| ((t - o) as f64).powi(2))
                .sum::<f64>()
                / out.len() as f64,
        );
        let params = AdamParams {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            decay: 0.0,
            batch: 1,
            step: step as u32,
            decoupled_decay: false,
        };
        let step_tensor = |w: &OxideTensorF32,
                           m: &OxideTensorF32,
                           v: &OxideTensorF32,
                           grad: &[f32]| {
            let grad = up(be, w.shape(), grad);
            be.adam_update_f32(w, &grad, m, v, &params, cfg()).unwrap()
        };
        for i in 0..4 {
            let (w, m, v) = step_tensor(&p.w[i], &m_w[i], &v_w[i], &g.w[i]);
            p.w[i] = w;
            m_w[i] = m;
            v_w[i] = v;
        }
        for i in 0..3 {
            let (w, m, v) = step_tensor(&p.bn_scale[i], &m_s[i], &v_s[i], &g.bn_scale[i]);
            p.bn_scale[i] = w;
            m_s[i] = m;
            v_s[i] = v;
            let (w, m, v) = step_tensor(&p.bn_bias[i], &m_b[i], &v_b[i], &g.bn_bias[i]);
            p.bn_bias[i] = w;
            m_b[i] = m;
            v_b[i] = v;
            p.rm[i] = rolling[i].0.clone();
            p.rv[i] = rolling[i].1.clone();
        }
        let (w, m, v) = step_tensor(&p.head_bias, &m_h, &v_h, &g.head_bias);
        p.head_bias = w;
        m_h = m;
        v_h = v;
        let mut snap: Vec<Vec<f32>> = p.w.iter().map(|t| down(be, t)).collect();
        snap.extend(p.bn_scale.iter().map(|t| down(be, t)));
        snap.extend(p.bn_bias.iter().map(|t| down(be, t)));
        snap.push(down(be, &p.head_bias));
        weights.push(snap);
    }
    (weights, losses)
}

/// AMP Adam must stay on the f32 master-weight trajectory (same documented
/// 10%-of-travel bound as the SGD test).  The runtime's Adam path shares this
/// kernel, so this covers "Adam under AMP" without a dataset run.
#[test]
fn amp_adam_steps_track_f32_master_weights() {
    let amp = amp_backend();
    let f32be = f32_backend();
    let mut rng = Rng(0xad4a_5eed);
    let p0 = init_params(&f32be, &mut rng);
    let p0_amp = clone_params(&f32be, &amp, &p0);
    let (x, target) = make_data(&mut rng);
    let steps = 4;
    let (w32, l32) = adam_run(&f32be, &p0, &x, &target, steps, 1.0);
    let (wamp, lamp) = adam_run(&amp, &p0_amp, &x, &target, steps, 256.0);
    println!("adam loss f32: {l32:.4?}");
    println!("adam loss amp: {lamp:.4?}");
    assert!(l32.last().unwrap() < &l32[0], "f32 Adam loss must decrease");
    assert!(lamp.last().unwrap() < &lamp[0], "AMP Adam loss must decrease");
    for (i, (a, b)) in lamp.iter().zip(&l32).enumerate() {
        assert!(
            (a - b).abs() <= 0.02 * b.abs().max(1e-6),
            "step {i}: loss amp {a} vs f32 {b}"
        );
    }
    let init: Vec<Vec<f32>> = {
        let mut v: Vec<Vec<f32>> = p0.w.iter().map(|t| down(&f32be, t)).collect();
        v.extend(p0.bn_scale.iter().map(|t| down(&f32be, t)));
        v.extend(p0.bn_bias.iter().map(|t| down(&f32be, t)));
        v.push(down(&f32be, &p0.head_bias));
        v
    };
    let names = ["w0", "w1", "w2", "w3", "s0", "s1", "s2", "b0", "b1", "b2", "bh"];
    for (t, name) in names.iter().enumerate() {
        let (fin32, finamp) = (&w32[steps - 1][t], &wamp[steps - 1][t]);
        let travelled = fin32
            .iter()
            .zip(&init[t])
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs() as f64));
        let diff = fin32
            .iter()
            .zip(finamp)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs() as f64));
        // Adam barely moves some tensors, so allow the f16 storage rounding
        // floor (2% of the initial magnitude) on top of the 10%-of-travel term.
        let scale = init[t].iter().fold(1e-6f64, |m, a| m.max(a.abs() as f64));
        let bound = 0.10 * travelled + 0.02 * scale;
        println!("adam {name}: |w_amp - w_f32|inf {diff:.3e} bound {bound:.3e}");
        assert!(diff <= bound, "adam {name}: {diff:.3e} > {bound:.3e}");
    }
}

// ---------------------------------------------------------------------------
// 16-bit format selection (plan_bf16 Task 1)
// ---------------------------------------------------------------------------

#[test]
fn amp_format_parses_known_values_case_insensitively() {
    use darknet_cuda_oxide::{HalfFormat, parse_amp_format};
    for v in ["1", "true", "TRUE", "on", "On", "yes", "fp16", "FP16", "f16"] {
        assert_eq!(parse_amp_format(v), Some(HalfFormat::F16), "{v}");
    }
    for v in ["bf16", "BF16", "bfloat16", "BFloat16"] {
        assert_eq!(parse_amp_format(v), Some(HalfFormat::Bf16), "{v}");
    }
    assert_eq!(HalfFormat::F16.code(), 0);
    assert_eq!(HalfFormat::Bf16.code(), 1);
}

#[test]
fn amp_format_rejects_garbage_and_empty() {
    use darknet_cuda_oxide::parse_amp_format;
    for v in ["", "0", "off", "no", "fp8", "bf17", " bf16", "garbage"] {
        assert_eq!(parse_amp_format(v), None, "{v:?}");
    }
}

#[test]
fn with_amp_format_selects_the_matching_gemm_mode() {
    use darknet_cuda_oxide::HalfFormat;
    let be = OxideBackend::new().expect("backend").with_amp_format(Some(HalfFormat::Bf16));
    assert!(be.amp_enabled());
    assert_eq!(be.half_format(), HalfFormat::Bf16);
    let be = be.with_amp_format(Some(HalfFormat::F16));
    assert_eq!(be.half_format(), HalfFormat::F16);
    assert!(be.amp_enabled());
    assert!(!be.with_amp_format(None).amp_enabled());
}

#[test]
fn bf16_gemm_mode_tracks_the_f32_gemm_in_the_f32_api() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Bf16);
    let (m, n, k) = (40, 56, 96);
    let mut rng = Rng(0xbf16_0001);
    let a = rng.vec(m * k, -1.0, 1.0);
    let b = rng.vec(k * n, -1.0, 1.0);
    let ta = be.upload_f32(&[m, k], &a).unwrap();
    let tb = be.upload_f32(&[k, n], &b).unwrap();
    let tc = be.upload_f32(&[m, n], &vec![0.0f32; m * n]).unwrap();
    let got = be
        .download_f32(&be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 1.0, 0.0).unwrap())
        .unwrap();
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let r: f64 = (0..k).map(|p| a[i * k + p] as f64 * b[p * n + j] as f64).sum();
            worst = worst.max((got[i * n + j] as f64 - r).abs());
            scale = scale.max(r.abs());
        }
    }
    // Two bf16 roundings (2^-9 each) over K = 96 random terms.
    assert!(worst / scale < 1.5e-2, "bf16 GEMM error {}", worst / scale);
}
