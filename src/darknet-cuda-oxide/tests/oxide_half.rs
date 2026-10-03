//! P6 f16 inference path: conversions, f16 tensor-core GEMM, fused f16 conv,
//! pool / upsample / route on f16 storage.
//!
//! Tolerances (documented, binary16 = 11 significant bits, relative rounding
//! error u = 2^-11 per rounding, f32 accumulation):
//!  * `gemm_f16` on operands that are ALREADY f16 values (`round16`): the only
//!    error is the f32 accumulation, so |err| <= 2e-5 * S + 1e-6 with
//!    S = sum |a||b| (+ |beta c0|), checked against an f64 reference.
//!  * `GemmMode::Fp16` in the f32 API rounds the f32 operands too: each
//!    product has <= ~2u = 2^-10 relative error, so |err| <= 1.2e-3 * S.
//!  * Fused conv (operands f16-exact, f32 bias): the output is rounded to f16
//!    once more (u = 4.9e-4 of |value| after the activation) so
//!    |err| <= 5.2e-4 * |ref| + 2e-5 * S + 1e-7 (1e-7 ~ f16 subnormal spacing).
//!  * Overflow: values above 65504 become +-Inf (or +-65504 with saturate),
//!    magnitudes below 2^-25 round to zero, subnormals are kept.  Finite
//!    differences are not run on f16 (inference only; a central difference at
//!    eps 1e-3 is below f16 resolution).

use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::half::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits,
};
use darknet_cuda_oxide::{GemmMode, OxideBackend};

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

fn backend() -> OxideBackend {
    OxideBackend::new().expect("backend")
}

fn round16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16_bits_to_f32(f32_to_f16_bits(x))).collect()
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[test]
fn host_reference_converters_are_consistent() {
    // Exhaustive: every f16 pattern widens and rounds back to itself (NaN: any NaN).
    for h in 0..=u16::MAX {
        let f = f16_bits_to_f32(h);
        let back = f32_to_f16_bits(f);
        if f.is_nan() {
            assert!(f16_bits_to_f32(back).is_nan(), "{h:#06x}");
        } else {
            assert_eq!(back, h, "{h:#06x} -> {f}");
        }
    }
    // Known values.
    assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
    assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
    assert_eq!(f32_to_f16_bits(65519.0), 0x7bff); // below the midpoint to inf
    assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // tie -> even = inf
    assert_eq!(f32_to_f16_bits(5.960_464_5e-8), 0x0001); // smallest subnormal
    assert_eq!(f32_to_f16_bits(2.980_232_2e-8), 0x0000); // exact tie -> even (0)
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
}

#[test]
fn gpu_conversions_match_host_reference() {
    let b = backend();
    // 1. Every f16 pattern widened on the device.
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let t = b.upload_f16_bits(&[all.len()], &all).unwrap();
    let wide = b.download_f16(&t).unwrap();
    for (h, w) in all.iter().zip(&wide) {
        assert!(same_f32(*w, f16_bits_to_f32(*h)) || (w.is_nan() && f16_bits_to_f32(*h).is_nan()), "{h:#06x}");
    }
    // 2. f32 -> f16 / bf16 on edge cases + random values vs the host reference.
    let mut vals = vec![
        0.0, -0.0, 1.0, -1.0, 65504.0, 65519.99, 65520.0, 70000.0, -70000.0, 1e10, f32::INFINITY,
        f32::NEG_INFINITY, f32::NAN, 6.1035156e-5, 6.0e-5, 5.960_464_5e-8, 2.980_232_2e-8, 2.98e-8, 3.0e-8,
        1e-10, -1e-10, 0.1, 0.333_333_34, 1.000_488_3, 1.000_732_4, 2049.0, 2051.0, f32::MIN_POSITIVE,
    ];
    let mut rng = Rng(77);
    vals.extend(rng.vec(4000, -10.0, 10.0));
    vals.extend(rng.vec(2000, -1.0e-3, 1.0e-3));
    vals.extend(rng.vec(2000, -7.0e4, 7.0e4));
    // Random bit patterns (all exponents, NaN payloads).
    for i in 0..4000u32 {
        vals.push(f32::from_bits(i.wrapping_mul(2_654_435_761).rotate_left(7)));
    }
    let x = b.upload_f32(&[vals.len()], &vals).unwrap();
    let h = b.f32_to_f16(&x, false).unwrap();
    let hb = b.download_f16_bits(&h).unwrap();
    let hs = b.download_f16_bits(&b.f32_to_f16(&x, true).unwrap()).unwrap();
    let bf = b.download_f16_bits(&b.f32_to_bf16(&x).unwrap()).unwrap();
    let bf_back = b.download_f32(&b.bf16_to_f32(&b.upload_f16_bits(&[bf.len()], &bf).unwrap()).unwrap()).unwrap();
    for (i, v) in vals.iter().enumerate() {
        let want = f32_to_f16_bits(*v);
        if v.is_nan() {
            assert!(f16_bits_to_f32(hb[i]).is_nan(), "nan[{i}]");
            assert!(f16_bits_to_f32(hs[i]).is_nan(), "sat nan[{i}]");
            assert!(bf16_bits_to_f32(bf[i]).is_nan(), "bf16 nan[{i}]");
        } else {
            assert_eq!(hb[i], want, "f16[{i}] of {v:e}: got {:#06x} want {want:#06x}", hb[i]);
            let sat_want = f32_to_f16_bits(v.clamp(-65504.0, 65504.0));
            assert_eq!(hs[i], sat_want, "sat f16[{i}] of {v:e}");
            assert_eq!(bf[i], f32_to_bf16_bits(*v), "bf16[{i}] of {v:e}");
        }
        assert!(same_f32(bf_back[i], bf16_bits_to_f32(bf[i])), "bf16 widen[{i}]");
    }
    // Round trip through the tensor API.
    let rt = b.download_f16(&h).unwrap();
    for (v, r) in vals.iter().zip(&rt) {
        if v.is_finite() && v.abs() < 65504.0 && v.abs() >= 6.2e-5 {
            assert!((v - r).abs() <= v.abs() * 4.89e-4, "roundtrip {v} -> {r}");
        }
    }
}

struct GemmCase {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

const GEMM_CASES: &[GemmCase] = &[
    GemmCase { name: "aligned_64", m: 64, n: 64, k: 64 },
    GemmCase { name: "aligned_rect", m: 128, n: 72, k: 96 },
    GemmCase { name: "odd_all", m: 37, n: 53, k: 29 },
    GemmCase { name: "k_not_32", m: 40, n: 40, k: 40 },
    GemmCase { name: "tiny", m: 3, n: 5, k: 7 },
    GemmCase { name: "splitk_deep", m: 16, n: 24, k: 2048 },
    GemmCase { name: "splitk_odd", m: 33, n: 17, k: 1000 },
    GemmCase { name: "multi_tile", m: 150, n: 130, k: 200 },
    GemmCase { name: "n_not_8_k_8", m: 64, n: 35, k: 64 },
];

fn ref_gemm(
    a: &[f32], b: &[f32], c0: &[f32], m: usize, n: usize, k: usize, ta: bool, tb: bool, alpha: f32, beta: f32,
) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut scale = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            let mut sa = 0.0f64;
            for p in 0..k {
                let av = if ta { a[p * m + i] } else { a[i * k + p] } as f64;
                let bv = if tb { b[j * k + p] } else { b[p * n + j] } as f64;
                s += av * bv;
                sa += (av * bv).abs();
            }
            let c = c0[i * n + j] as f64;
            out[i * n + j] = alpha as f64 * s + beta as f64 * c;
            scale[i * n + j] = alpha.abs() as f64 * sa + beta.abs() as f64 * c.abs();
        }
    }
    (out, scale)
}

#[test]
fn gemm_f16_parity_all_transposes_alignments_alpha_beta() {
    let be = backend();
    let mut rng = Rng(4242);
    let mut worst = 0.0f64;
    for case in GEMM_CASES {
        let (m, n, k) = (case.m, case.n, case.k);
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.75, 0.5), (-1.5, 1.0)] {
                let a = round16(&rng.vec(m * k, -1.0, 1.0));
                let b = round16(&rng.vec(k * n, -1.0, 1.0));
                let c0 = rng.vec(m * n, -1.0, 1.0);
                let (want, scale) = ref_gemm(&a, &b, &c0, m, n, k, ta, tb, alpha, beta);
                let at = be.upload_f16(&[a.len()], &a).unwrap();
                let bt = be.upload_f16(&[b.len()], &b).unwrap();
                let ct = be.upload_f32(&[m, n], &c0).unwrap();
                let got = be.download_f32(&be.gemm_f16(&at, &bt, &ct, m, n, k, ta, tb, alpha, beta).unwrap()).unwrap();
                for i in 0..m * n {
                    let err = (got[i] as f64 - want[i]).abs();
                    let tol = 2e-5 * scale[i] + 1e-6;
                    worst = worst.max(err / (scale[i] + 1e-30));
                    assert!(
                        err <= tol,
                        "{} ta={ta} tb={tb} alpha={alpha} beta={beta} [{i}]: got {}, want {}, err {err}, tol {tol}",
                        case.name, got[i], want[i]
                    );
                }
            }
        }
    }
    println!("gemm_f16 worst err/S = {worst:.3e}");
}

fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| bf16_bits_to_f32(f32_to_bf16_bits(x))).collect()
}

/// bf16 operands (already bf16-exact), f32 accumulation: same error model as
/// the f16 GEMM (|err| <= 2e-5 * S + 1e-6, f64 reference), every transpose,
/// alignment, split-K and (alpha, beta) combination of `GEMM_CASES`.
#[test]
fn gemm_bf16_parity_all_transposes_alignments_alpha_beta() {
    let be = backend();
    let mut rng = Rng(777);
    let mut worst = 0.0f64;
    for case in GEMM_CASES {
        let (m, n, k) = (case.m, case.n, case.k);
        for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
            for &(alpha, beta) in &[(1.0f32, 0.0f32), (0.75, 0.5), (-1.5, 1.0)] {
                let a = round_bf16(&rng.vec(m * k, -1.0, 1.0));
                let b = round_bf16(&rng.vec(k * n, -1.0, 1.0));
                let c0 = rng.vec(m * n, -1.0, 1.0);
                let (want, scale) = ref_gemm(&a, &b, &c0, m, n, k, ta, tb, alpha, beta);
                let at = be.f32_to_bf16(&be.upload_f32(&[a.len()], &a).unwrap()).unwrap();
                let bt = be.f32_to_bf16(&be.upload_f32(&[b.len()], &b).unwrap()).unwrap();
                let ct = be.upload_f32(&[m, n], &c0).unwrap();
                let got = be.download_f32(&be.gemm_bf16(&at, &bt, &ct, m, n, k, ta, tb, alpha, beta).unwrap()).unwrap();
                for i in 0..m * n {
                    let err = (got[i] as f64 - want[i]).abs();
                    let tol = 2e-5 * scale[i] + 1e-6;
                    worst = worst.max(err / (scale[i] + 1e-30));
                    assert!(
                        err <= tol,
                        "{} ta={ta} tb={tb} alpha={alpha} beta={beta} [{i}]: got {}, want {}, err {err}, tol {tol}",
                        case.name, got[i], want[i]
                    );
                }
            }
        }
    }
    println!("gemm_bf16 worst err/S = {worst:.3e}");
}

#[test]
fn gemm_beta_zero_ignores_nan_in_c_f16() {
    let be = backend();
    let a = be.upload_f16(&[4, 8], &vec![1.0; 32]).unwrap();
    let b = be.upload_f16(&[8, 8], &vec![0.5; 64]).unwrap();
    let c = be.upload_f32(&[4, 8], &vec![f32::NAN; 32]).unwrap();
    let out = be.download_f32(&be.gemm_f16(&a, &b, &c, 4, 8, 8, false, false, 1.0, 0.0).unwrap()).unwrap();
    assert!(out.iter().all(|v| (*v - 4.0).abs() < 1e-6));
}

#[test]
fn gemm_mode_fp16_f32_api_within_documented_tolerance() {
    let be = OxideBackend::new().unwrap().with_gemm_mode(GemmMode::Fp16);
    let mut rng = Rng(99);
    let mut worst = 0.0f64;
    for case in GEMM_CASES {
        let (m, n, k) = (case.m, case.n, case.k);
        for &(ta, tb) in &[(false, false), (true, false), (false, true), (true, true)] {
            let a = rng.vec(m * k, -1.0, 1.0);
            let b = rng.vec(k * n, -1.0, 1.0);
            let c0 = rng.vec(m * n, -1.0, 1.0);
            let (want, scale) = ref_gemm(&a, &b, &c0, m, n, k, ta, tb, 0.5, 0.25);
            let at = be_upload(&be, &a, &[a.len()]);
            let bt = be_upload(&be, &b, &[b.len()]);
            let ct = be_upload(&be, &c0, &[m, n]);
            let got = be.download_f32(&be.gemm_f32(&at, &bt, &ct, m, n, k, ta, tb, 0.5, 0.25).unwrap()).unwrap();
            for i in 0..m * n {
                let err = (got[i] as f64 - want[i]).abs();
                worst = worst.max(err / (scale[i] + 1e-30));
                assert!(err <= 1.2e-3 * scale[i] + 1e-6, "{} ta={ta} tb={tb} [{i}] got {} want {} err {err}", case.name, got[i], want[i]);
            }
        }
    }
    println!("GemmMode::Fp16 f32 API worst err/S = {worst:.3e} (bound 1.2e-3)");
}

fn be_upload(be: &OxideBackend, v: &[f32], shape: &[usize]) -> darknet_cuda_oxide::OxideTensorF32 {
    be.upload_f32(shape, v).unwrap()
}

#[derive(Clone, Copy)]
struct ConvCase {
    name: &'static str,
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

const CONV_CASES: &[ConvCase] = &[
    ConvCase { name: "stem_c3_k3", b: 1, c: 3, h: 33, w: 29, f: 16, k: 3, s: 1, pad: 1, groups: 1 },
    ConvCase { name: "k3_c16_aligned", b: 2, c: 16, h: 16, w: 16, f: 32, k: 3, s: 1, pad: 1, groups: 1 },
    ConvCase { name: "k3_stride2", b: 1, c: 8, h: 21, w: 19, f: 24, k: 3, s: 2, pad: 1, groups: 1 },
    ConvCase { name: "k1_fast_path", b: 2, c: 16, h: 8, w: 16, f: 40, k: 1, s: 1, pad: 0, groups: 1 },
    ConvCase { name: "k1_hw35_unaligned", b: 1, c: 24, h: 7, w: 5, f: 20, k: 1, s: 1, pad: 0, groups: 1 },
    ConvCase { name: "k1_c_not_8", b: 1, c: 5, h: 8, w: 8, f: 9, k: 1, s: 1, pad: 0, groups: 1 },
    ConvCase { name: "groups2", b: 2, c: 8, h: 12, w: 10, f: 12, k: 3, s: 1, pad: 1, groups: 2 },
    ConvCase { name: "k1_groups2_aligned", b: 3, c: 32, h: 8, w: 8, f: 16, k: 1, s: 1, pad: 0, groups: 2 },
    ConvCase { name: "deep_splitk", b: 1, c: 128, h: 4, w: 4, f: 64, k: 3, s: 1, pad: 1, groups: 1 },
    ConvCase { name: "k5_pad2", b: 1, c: 4, h: 9, w: 9, f: 6, k: 5, s: 1, pad: 2, groups: 1 },
];

fn act_ref(kind: F32ActivationKind, x: f64) -> f64 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Leaky => if x > 0.0 { x } else { 0.1 * x },
        F32ActivationKind::Logistic => 1.0 / (1.0 + (-x).exp()),
        F32ActivationKind::Relu => x.max(0.0),
        F32ActivationKind::Mish => x * (1.0f64 + x.exp()).ln().tanh(),
        F32ActivationKind::Swish => x / (1.0 + (-x).exp()),
        F32ActivationKind::Tanh => x.tanh(),
        F32ActivationKind::Relu6 => x.clamp(0.0, 6.0),
        F32ActivationKind::Ramp => x * (x > 0.0) as i32 as f64 + 0.1 * x,
    }
}

/// f64 reference of conv + bias + activation; returns `(value, sum|w x| + |b|)`.
fn conv_ref(cs: &ConvCase, x: &[f32], w: &[f32], bias: &[f32], act: F32ActivationKind) -> (Vec<f64>, Vec<f64>) {
    let oh = (cs.h + 2 * cs.pad - cs.k) / cs.s + 1;
    let ow = (cs.w + 2 * cs.pad - cs.k) / cs.s + 1;
    let cpg = cs.c / cs.groups;
    let fpg = cs.f / cs.groups;
    let mut out = vec![0.0; cs.b * cs.f * oh * ow];
    let mut sc = vec![0.0; out.len()];
    for b in 0..cs.b {
        for f in 0..cs.f {
            let g = f / fpg;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = bias[f] as f64;
                    let mut sa = (bias[f] as f64).abs();
                    for ci in 0..cpg {
                        for ky in 0..cs.k {
                            for kx in 0..cs.k {
                                let iy = (oy * cs.s + ky) as i64 - cs.pad as i64;
                                let ix = (ox * cs.s + kx) as i64 - cs.pad as i64;
                                if iy < 0 || ix < 0 || iy >= cs.h as i64 || ix >= cs.w as i64 {
                                    continue;
                                }
                                let xv = x[((b * cs.c + g * cpg + ci) * cs.h + iy as usize) * cs.w + ix as usize] as f64;
                                let wv = w[((f * cpg + ci) * cs.k + ky) * cs.k + kx] as f64;
                                s += xv * wv;
                                sa += (xv * wv).abs();
                            }
                        }
                    }
                    let o = ((b * cs.f + f) * oh + oy) * ow + ox;
                    out[o] = act_ref(act, s);
                    sc[o] = sa;
                }
            }
        }
    }
    (out, sc)
}

fn run_conv(be: &OxideBackend, cs: &ConvCase, x: &[f32], w: &[f32], bias: &[f32], act: F32ActivationKind) -> Vec<f32> {
    let oh = (cs.h + 2 * cs.pad - cs.k) / cs.s + 1;
    let ow = (cs.w + 2 * cs.pad - cs.k) / cs.s + 1;
    let cpg = cs.c / cs.groups;
    let xt = be.upload_f16(&[cs.b, cs.c, cs.h, cs.w], x).unwrap();
    let wt = be.conv_weights_f16(&be_upload(be, w, &[w.len()]), cs.f, cpg * cs.k * cs.k).unwrap();
    let bt = be_upload(be, bias, &[cs.f]);
    let y = be
        .conv2d_bias_act_f16(&xt, &wt, &bt, &[cs.b, cs.f, oh, ow], cs.c, cs.h, cs.w, cs.f, cs.k, cs.s, cs.s, 1, cs.pad, cs.groups, act, cfg())
        .unwrap();
    be.download_f16(&y).unwrap()
}

#[test]
fn conv_f16_fused_matches_f64_reference() {
    let be = backend();
    let mut rng = Rng(555);
    let mut worst = 0.0f64;
    for cs in CONV_CASES {
        eprintln!("case {}", cs.name);
        let cpg = cs.c / cs.groups;
        for &act in &[F32ActivationKind::Leaky, F32ActivationKind::Linear, F32ActivationKind::Logistic] {
            let x = round16(&rng.vec(cs.b * cs.c * cs.h * cs.w, -1.0, 1.0));
            let w = round16(&rng.vec(cs.f * cpg * cs.k * cs.k, -0.5, 0.5));
            let bias = rng.vec(cs.f, -0.5, 0.5);
            let (want, scale) = conv_ref(cs, &x, &w, &bias, act);
            let got = run_conv(&be, cs, &x, &w, &bias, act);
            assert_eq!(got.len(), want.len(), "{}", cs.name);
            for i in 0..got.len() {
                let err = (got[i] as f64 - want[i]).abs();
                let tol = 5.2e-4 * want[i].abs() + 2e-5 * scale[i] + 1e-7;
                worst = worst.max(err / tol);
                assert!(err <= tol, "{} {act:?}[{i}]: got {}, want {}, err {err}, tol {tol}", cs.name, got[i], want[i]);
            }
        }
    }
    println!("conv f16 worst err/tol = {worst:.3}");
}

#[test]
fn conv_f16_close_to_f32_conv_on_unrounded_weights() {
    // End-to-end rounding of weights and activations (both f32 -> f16 on the
    // way in): compare with the f32 oxide conv; error ~ sqrt(K) * 2^-11 * |x||w|.
    let be = backend();
    let mut rng = Rng(31337);
    let cs = ConvCase { name: "agree", b: 1, c: 32, h: 20, w: 20, f: 32, k: 3, s: 1, pad: 1, groups: 1 };
    let x = rng.vec(cs.b * cs.c * cs.h * cs.w, -1.0, 1.0);
    let w = rng.vec(cs.f * cs.c * 9, -0.2, 0.2);
    let bias = rng.vec(cs.f, -0.2, 0.2);
    let (want, scale) = conv_ref(&cs, &x, &w, &bias, F32ActivationKind::Linear);
    let got = run_conv(&be, &cs, &x, &w, &bias, F32ActivationKind::Linear);
    let mut worst = 0.0f64;
    for i in 0..got.len() {
        let err = (got[i] as f64 - want[i]).abs();
        worst = worst.max(err / scale[i]);
        assert!(err <= 1.2e-3 * scale[i] + 5.2e-4 * want[i].abs(), "[{i}] err {err} scale {}", scale[i]);
    }
    println!("conv f16 (unrounded operands) worst err/S = {worst:.3e}");
}

#[test]
fn f16_overflow_underflow_behaviour() {
    let be = backend();
    // 1x1 conv summing 16 channels of 5000 -> 80000 overflows f16 -> +Inf; the
    // negative twin gives -Inf; a small sum stays finite; subnormal output kept.
    let cs = ConvCase { name: "ovf", b: 1, c: 16, h: 1, w: 8, f: 4, k: 1, s: 1, pad: 0, groups: 1 };
    let mut x = vec![0.0f32; 16 * 8];
    for c in 0..16 {
        for p in 0..8 {
            x[c * 8 + p] = 5000.0;
        }
    }
    let mut w = vec![0.0f32; 4 * 16];
    for c in 0..16 {
        w[c] = 1.0; // f0: +80000
        w[16 + c] = -1.0; // f1: -80000
        w[32 + c] = 0.001; // f2: 80 (finite)
    }
    for c in 0..16 {
        w[48 + c] = if c == 0 { 1e-5 } else { 0.0 }; // f3: 5000 * 1e-5 = 0.05
    }
    let bias = vec![0.0f32; 4];
    let got = run_conv(&be, &cs, &x, &w, &bias, F32ActivationKind::Linear);
    assert_eq!(got[0], f32::INFINITY);
    assert_eq!(got[8], f32::NEG_INFINITY);
    assert!((got[16] - 80.0).abs() < 0.1, "{}", got[16]);
    assert!((got[24] - 0.05).abs() < 2e-4, "{}", got[24]); // 1e-5 is an f16 subnormal (spacing 6e-8)
    // Underflow: tiny products flush to zero below 2^-25, subnormals survive.
    let cs2 = ConvCase { name: "udf", b: 1, c: 8, h: 1, w: 8, f: 3, k: 1, s: 1, pad: 0, groups: 1 };
    let x2 = vec![1.0f32; 8 * 8];
    let mut w2 = vec![0.0f32; 3 * 8];
    w2[0] = 1e-8; // f0 -> 1e-8 < 2^-25 -> 0
    w2[8] = 1e-7; // f1 -> 1e-7 subnormal (~2 ulp of 5.96e-8)
    w2[16] = 3.0e-5; // f2 -> subnormal 3e-5
    // weights themselves are rounded to f16 on upload: 1e-8 -> 0
    let got2 = run_conv(&be, &cs2, &x2, &w2, &vec![0.0; 3], F32ActivationKind::Linear);
    assert_eq!(got2[0], 0.0);
    assert!(got2[8] > 0.0 && (got2[8] - 1e-7).abs() <= 6e-8, "{}", got2[8]);
    assert!((got2[16] - 3.0e-5).abs() <= 3.0e-8, "{}", got2[16]);
    // NaN propagates through the GEMM.
    let mut xn = x2.clone();
    xn[3] = f32::NAN;
    let gn = run_conv(&be, &cs2, &xn, &w2, &vec![0.0; 3], F32ActivationKind::Linear);
    assert!(gn[3].is_nan());
    assert!(!gn[2].is_nan());
}

#[test]
fn f16_pool_upsample_route_match_f32_reference() {
    let be = backend();
    let mut rng = Rng(8);
    // maxpool 2x2 s2, 3x3 s2 pad 2, 2x2 s1 pad 1 (the LegoGears / cfg combos).
    for &(size, stride, pad) in &[(2usize, 2usize, 0usize), (3, 2, 2), (2, 1, 1)] {
        let (b, c, h, w) = (2usize, 3usize, 9usize, 11usize);
        let x = round16(&rng.vec(b * c * h * w, -4.0, 4.0));
        let off = pad / 2;
        let oh = (h + pad - size) / stride + 1;
        let ow = (w + pad - size) / stride + 1;
        let xt = be.upload_f16(&[b, c, h, w], &x).unwrap();
        let y = be.maxpool_f16(&xt, &[b, c, oh, ow], c, h, w, size, stride, stride, pad, cfg()).unwrap();
        let got = be.download_f16(&y).unwrap();
        for bc in 0..b * c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    for ky in 0..size {
                        for kx in 0..size {
                            let iy = (oy * stride + ky) as i64 - off as i64;
                            let ix = (ox * stride + kx) as i64 - off as i64;
                            if iy >= 0 && ix >= 0 && iy < h as i64 && ix < w as i64 {
                                best = best.max(x[(bc * h + iy as usize) * w + ix as usize]);
                            }
                        }
                    }
                    assert_eq!(got[(bc * oh + oy) * ow + ox], best, "maxpool {size}/{stride}/{pad}");
                }
            }
        }
    }
    // upsample x2
    let (b, c, h, w) = (2usize, 3usize, 4usize, 5usize);
    let x = round16(&rng.vec(b * c * h * w, -2.0, 2.0));
    let xt = be.upload_f16(&[b, c, h, w], &x).unwrap();
    let up = be.upsample_f16(&xt, &[b, c, 2 * h, 2 * w], c, h, w, 2, 1.0, false, cfg()).unwrap();
    let got = be.download_f16(&up).unwrap();
    for bc in 0..b * c {
        for oy in 0..2 * h {
            for ox in 0..2 * w {
                assert_eq!(got[(bc * 2 * h + oy) * 2 * w + ox], x[(bc * h + oy / 2) * w + ox / 2]);
            }
        }
    }
    // route: concat of [full a (3ch)] + [channels 2..4 of b (4ch)]
    let a = round16(&rng.vec(2 * 3 * 6, -1.0, 1.0));
    let bb = round16(&rng.vec(2 * 4 * 6, -1.0, 1.0));
    let at = be.upload_f16(&[2, 3, 2, 3], &a).unwrap();
    let bt = be.upload_f16(&[2, 4, 2, 3], &bb).unwrap();
    let r = be.route_concat_f16(&[&at, &bt], &[0, 2], &[3, 2], &[3, 4], &[2, 5, 2, 3], 6, cfg()).unwrap();
    let got = be.download_f16(&r).unwrap();
    for n in 0..2 {
        for ch in 0..5 {
            for s in 0..6 {
                let want = if ch < 3 { a[(n * 3 + ch) * 6 + s] } else { bb[(n * 4 + 2 + ch - 3) * 6 + s] };
                assert_eq!(got[(n * 5 + ch) * 6 + s], want);
            }
        }
    }
}

#[test]
fn f16_ops_reject_bad_shapes() {
    let be = backend();
    let x = be.upload_f16(&[1, 2, 4, 4], &vec![0.0; 32]).unwrap();
    let w = be.upload_f16(&[3, 8], &vec![0.0; 24]).unwrap();
    let bias = be_upload(&be, &[0.0; 3], &[3]);
    // kpad for c=2,k=3 is 24, not 8
    assert!(be.conv2d_bias_act_f16(&x, &w, &bias, &[1, 3, 4, 4], 2, 4, 4, 3, 3, 1, 1, 1, 1, 1, F32ActivationKind::Linear, cfg()).is_err());
    assert!(be.conv_weights_f16(&be_upload(&be, &[0.0; 5], &[5]), 2, 3).is_err());
    assert!(be.upload_f16(&[3], &[0.0; 2]).is_err());
    assert!(be.upsample_f16(&x, &[1, 2, 8, 8], 2, 4, 4, 2, 1.0, true, cfg()).is_err());
}

#[test]
fn conv_f16_legogears_v2_real_layer_shapes() {
    // The 21 LegoGears_v2 convs at full size (batch 1, leaky), f64 reference.
    let csv = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legogears_v2_convs.csv")).unwrap();
    let be = backend();
    let mut rng = Rng(2026);
    let mut worst = 0.0f64;
    let mut n = 0;
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let u = |i: usize| f[i].parse::<usize>().unwrap();
        assert_eq!(u(6), u(7));
        let cs = ConvCase { name: "lego", b: 1, c: u(1), h: u(2), w: u(3), f: u(4), k: u(5), s: u(6), pad: u(9), groups: u(10) };
        let kk = cs.c / cs.groups * cs.k * cs.k;
        let amp = 1.0 / (kk as f32).sqrt();
        let x = round16(&rng.vec(cs.c * cs.h * cs.w, -1.0, 1.0));
        let w = round16(&rng.vec(cs.f * kk, -amp, amp));
        let bias = rng.vec(cs.f, -0.5, 0.5);
        let (want, scale) = conv_ref(&cs, &x, &w, &bias, F32ActivationKind::Leaky);
        let got = run_conv(&be, &cs, &x, &w, &bias, F32ActivationKind::Leaky);
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            let err = (got[i] as f64 - want[i]).abs();
            let tol = 5.2e-4 * want[i].abs() + 2e-5 * scale[i] + 1e-7;
            worst = worst.max(err / tol);
            assert!(err <= tol, "layer {} [{i}] got {} want {} err {err} tol {tol}", u(0), got[i], want[i]);
        }
        n += 1;
    }
    assert_eq!(n, 21);
    println!("lego 21 layers worst err/tol = {worst:.3}");
}

/// |got - ref| <= 5.2e-4 |ref| + 1e-7: one binary16 rounding of the f32/f64
/// result (u = 2^-11 = 4.88e-4) plus f16 subnormal spacing; the operands are
/// f16-exact so the f32 arithmetic error (~1e-6 relative) is far below it.
fn assert_f16_close(name: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = 5.2e-4 * w.abs() + 1e-7;
        assert!((g as f64 - w).abs() <= tol, "{name}[{i}]: got {g} want {w} (tol {tol})");
    }
}

#[test]
fn f16_shortcut_avgpool_affine_reverse_match_f64_reference() {
    let be = backend();
    let mut rng = Rng(21);
    let acts = [
        F32ActivationKind::Linear, F32ActivationKind::Leaky, F32ActivationKind::Logistic, F32ActivationKind::Relu,
        F32ActivationKind::Mish, F32ActivationKind::Swish, F32ActivationKind::Tanh, F32ActivationKind::Relu6,
        F32ActivationKind::Ramp,
    ];
    // shortcut: act(a + b), odd length so the launch tail is exercised.
    for act in acts {
        let n = 2 * 5 * 7 * 3 + 1;
        let a = round16(&rng.vec(n, -5.0, 5.0));
        let b = round16(&rng.vec(n, -5.0, 5.0));
        let at = be.upload_f16(&[n], &a).unwrap();
        let bt = be.upload_f16(&[n], &b).unwrap();
        let got = be.download_f16(&be.shortcut_add_act_f16(&at, &bt, act, cfg()).unwrap()).unwrap();
        let want: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| act_ref(act, x as f64 + y as f64)).collect();
        assert_f16_close(&format!("shortcut {act:?}"), &got, &want);
    }
    // standalone bias + activation and folded BN (scale, bias) per channel.
    let (b, c, h, w) = (2usize, 5usize, 6usize, 7usize);
    let x = round16(&rng.vec(b * c * h * w, -4.0, 4.0));
    let scale = rng.vec(c, 0.3, 1.7);
    let bias = rng.vec(c, -1.0, 1.0);
    let xt = be.upload_f16(&[b, c, h, w], &x).unwrap();
    let st = be_upload(&be, &scale, &[c]);
    let bt = be_upload(&be, &bias, &[c]);
    for act in acts {
        let got = be.download_f16(&be.bias_activation_f16(&xt, &bt, c, h * w, act, cfg()).unwrap()).unwrap();
        let want: Vec<f64> =
            x.iter().enumerate().map(|(i, &v)| act_ref(act, v as f64 + bias[(i / (h * w)) % c] as f64)).collect();
        assert_f16_close(&format!("bias_act {act:?}"), &got, &want);
        let got = be.download_f16(&be.channel_affine_act_f16(&xt, Some(&st), &bt, c, h * w, act, cfg()).unwrap()).unwrap();
        let want: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = (i / (h * w)) % c;
                act_ref(act, v as f64 * scale[ch] as f64 + bias[ch] as f64)
            })
            .collect();
        assert_f16_close(&format!("affine {act:?}"), &got, &want);
    }
    // local avgpool (3x3 s2 pad 2, 2x2 s1 pad 1, 2x2 s2) and global avgpool.
    for &(size, stride, pad) in &[(3usize, 2usize, 2usize), (2, 1, 1), (2, 2, 0)] {
        let off = pad / 2;
        let oh = (h + pad - size) / stride + 1;
        let ow = (w + pad - size) / stride + 1;
        let y = be.local_avgpool_f16(&xt, &[b, c, oh, ow], c, h, w, size, stride, stride, pad, cfg()).unwrap();
        let got = be.download_f16(&y).unwrap();
        let mut want = Vec::new();
        for bc in 0..b * c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let (mut s, mut cnt) = (0.0f64, 0usize);
                    for ky in 0..size {
                        for kx in 0..size {
                            let iy = (oy * stride + ky) as i64 - off as i64;
                            let ix = (ox * stride + kx) as i64 - off as i64;
                            if iy >= 0 && ix >= 0 && iy < h as i64 && ix < w as i64 {
                                s += x[(bc * h + iy as usize) * w + ix as usize] as f64;
                                cnt += 1;
                            }
                        }
                    }
                    want.push(if cnt > 0 { s / cnt as f64 } else { 0.0 });
                }
            }
        }
        assert_f16_close(&format!("local_avgpool {size}/{stride}/{pad}"), &got, &want);
    }
    let g = be.avgpool_f16(&xt, &[b, c, 1, 1], c, h, w, cfg()).unwrap();
    let want: Vec<f64> =
        (0..b * c).map(|bc| x[bc * h * w..(bc + 1) * h * w].iter().map(|&v| v as f64).sum::<f64>() / (h * w) as f64).collect();
    assert_f16_close("avgpool", &be.download_f16(&g).unwrap(), &want);
    // reverse upsample: floor-sized (7 / 2 = 3) and exact outputs, with scale.
    for &(stride, oh, ow, sc) in &[(2usize, 3usize, 3usize, 1.0f32), (2, 3, 4, 0.5), (3, 2, 2, 1.0)] {
        let (ih, iw) = (6usize, 7usize);
        let x = round16(&rng.vec(b * c * ih * iw, -3.0, 3.0));
        let xt = be.upload_f16(&[b, c, ih, iw], &x).unwrap();
        let y = be.upsample_f16(&xt, &[b, c, oh, ow], c, ih, iw, stride, sc, true, cfg());
        if oh > ih.div_ceil(stride) || ow > iw.div_ceil(stride) {
            assert!(y.is_err());
            continue;
        }
        let got = be.download_f16(&y.unwrap()).unwrap();
        let mut want = Vec::new();
        for bc in 0..b * c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = 0.0f64;
                    for dy in 0..stride {
                        for dx in 0..stride {
                            let (iy, ix) = (oy * stride + dy, ox * stride + dx);
                            if iy < ih && ix < iw {
                                s += x[(bc * ih + iy) * iw + ix] as f64;
                            }
                        }
                    }
                    want.push(sc as f64 * s);
                }
            }
        }
        assert_f16_close(&format!("reverse upsample s{stride}"), &got, &want);
    }
    // bad shapes
    assert!(be.shortcut_add_act_f16(&xt, &be.upload_f16(&[3], &[0.0; 3]).unwrap(), F32ActivationKind::Linear, cfg()).is_err());
    assert!(be.channel_affine_act_f16(&xt, None, &be_upload(&be, &[0.0; 4], &[4]), c, h * w, F32ActivationKind::Linear, cfg()).is_err());
    assert!(be.avgpool_f16(&xt, &[b, c, 2, 1], c, h, w, cfg()).is_err());
}

/// f16 GEMM micro-benchmark (`... -- --ignored --nocapture f16_gemm_bench`):
/// median over `DARKNET_BENCH_ROUNDS` (default 7) of mean device ms, 2048^3,
/// 4096^3 and the sum over the 21 LegoGears_v2 conv shapes.
#[test]
#[ignore]
fn f16_gemm_bench() {
    let be = backend();
    let rounds: usize = std::env::var("DARKNET_BENCH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let lego: [(usize, usize, usize); 21] = [
        (32, 8960, 27), (64, 2240, 288), (64, 2240, 576), (32, 2240, 288), (32, 2240, 288), (64, 2240, 64),
        (128, 560, 576), (64, 560, 576), (64, 560, 576), (128, 560, 128), (256, 140, 1152), (128, 140, 1152),
        (128, 140, 1152), (256, 140, 256), (512, 35, 2304), (256, 35, 512), (512, 35, 2304), (30, 35, 512),
        (128, 35, 256), (256, 140, 3456), (30, 140, 256),
    ];
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    for &s in &[2048usize, 4096] {
        let iters = if s == 4096 { 10 } else { 40 };
        let ms = med((0..rounds).map(|_| be.bench_gemm_f16(s, s, s, iters).unwrap()).collect());
        println!("BENCH f16 {s}^3: {ms:.4} ms {:.1} TF/s", 2.0 * (s as f64).powi(3) / ms / 1e9);
    }
    let total: f64 = lego
        .iter()
        .map(|&(m, n, k)| med((0..rounds).map(|_| be.bench_gemm_f16(m, n, k, 100).unwrap()).collect()))
        .sum();
    println!("BENCH f16 lego21 total: {total:.4} ms");
}
