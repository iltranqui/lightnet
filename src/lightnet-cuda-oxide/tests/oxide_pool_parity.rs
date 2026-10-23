//! Pooling / resample / route / crop: forward parity, backward parity and
//! central finite-difference gradient checks against f64 CPU references.
//!
//! The references are written from `lightnet-core` (`forward_maxpool`,
//! `backward_maxpool`, `forward_upsample`, `forward_route`, `crop_bilinear`)
//! and the cuTile bridge; window rule: input row = `oy*stride_y + ky - pad/2`,
//! taps outside the image are skipped.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda_oxide::OxideBackend;

const TOL: f32 = 1e-4;

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

    /// Pairwise distinct values (gap >= 0.01, far above the FD step) in a
    /// shuffled order, so max-pool has no ties.
    fn distinct(&mut self, n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - n as f32 * 0.005).collect();
        for i in (1..n).rev() {
            let j = (self.next(0.0, 1.0) * (i + 1) as f32) as usize % (i + 1);
            v.swap(i, j);
        }
        v
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
        assert!(
            err <= TOL as f64 + TOL as f64 * w.abs(),
            "{name}[{i}]: got {g}, want {w}, err {err}"
        );
    }
}

/// Central finite difference of `L(x) = sum(dy * f(x))` compared with `got`
/// (the op's backward for upstream gradient `dy`).
fn fd_check(name: &str, x: &[f32], dy: &[f32], f: impl Fn(&[f64]) -> Vec<f64>, got: &[f32]) {
    assert_eq!(x.len(), got.len(), "{name}: gradient length");
    let dy = f64s(dy);
    let loss = |xv: &[f64]| -> f64 { f(xv).iter().zip(&dy).map(|(a, b)| a * b).sum() };
    let eps = 1e-3f64;
    let mut xv = f64s(x);
    for i in 0..x.len() {
        let orig = xv[i];
        xv[i] = orig + eps;
        let lp = loss(&xv);
        xv[i] = orig - eps;
        let lm = loss(&xv);
        xv[i] = orig;
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (got[i] as f64 - fd).abs() <= 1e-3 + 1e-3 * fd.abs(),
            "{name} fd[{i}]: backward {}, finite difference {fd}",
            got[i]
        );
    }
}

// ---------------------------------------------------------------- pooling

#[derive(Clone, Copy, Debug)]
struct PoolCase {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    size: usize,
    sx: usize,
    sy: usize,
    pad: usize,
}

impl PoolCase {
    /// Same rule as C++ `make_maxpool_layer` and `lightnet-core::pool_dim`.
    fn oh(&self) -> usize {
        (self.h + self.pad - self.size) / self.sy + 1
    }
    fn ow(&self) -> usize {
        (self.w + self.pad - self.size) / self.sx + 1
    }
    fn in_shape(&self) -> [usize; 4] {
        [self.b, self.c, self.h, self.w]
    }
    fn out_shape(&self) -> [usize; 4] {
        [self.b, self.c, self.oh(), self.ow()]
    }
}

const POOL_CASES: [PoolCase; 9] = [
    // yolov4-tiny: size=2 stride=2 (default padding = size-1 = 1), odd input.
    PoolCase { b: 2, c: 3, h: 7, w: 9, size: 2, sx: 2, sy: 2, pad: 1 },
    // yolov4-tiny: size=2 stride=1 (output keeps the input size).
    PoolCase { b: 2, c: 3, h: 7, w: 9, size: 2, sx: 1, sy: 1, pad: 1 },
    PoolCase { b: 3, c: 2, h: 5, w: 6, size: 3, sx: 1, sy: 1, pad: 2 },
    PoolCase { b: 2, c: 4, h: 9, w: 7, size: 3, sx: 2, sy: 2, pad: 2 },
    PoolCase { b: 2, c: 2, h: 7, w: 9, size: 2, sx: 2, sy: 2, pad: 0 },
    PoolCase { b: 1, c: 3, h: 8, w: 8, size: 3, sx: 2, sy: 2, pad: 0 },
    PoolCase { b: 2, c: 2, h: 6, w: 7, size: 3, sx: 2, sy: 1, pad: 2 },
    // odd padding: offset pad/2 = 1 with an extra window at the far edge.
    PoolCase { b: 2, c: 2, h: 5, w: 5, size: 3, sx: 1, sy: 1, pad: 3 },
    // window larger than stride * 2 (overlap factor 3).
    PoolCase { b: 1, c: 2, h: 10, w: 11, size: 3, sx: 1, sy: 2, pad: 2 },
];

fn cpu_maxpool(x: &[f64], p: &PoolCase) -> Vec<f64> {
    let (oh, ow, off) = (p.oh(), p.ow(), (p.pad / 2) as isize);
    let mut out = vec![f64::NEG_INFINITY; p.b * p.c * oh * ow];
    for bc in 0..p.b * p.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f64::NEG_INFINITY;
                for ky in 0..p.size {
                    for kx in 0..p.size {
                        let iy = (oy * p.sy + ky) as isize - off;
                        let ix = (ox * p.sx + kx) as isize - off;
                        if iy < 0 || iy >= p.h as isize || ix < 0 || ix >= p.w as isize {
                            continue;
                        }
                        best = best.max(x[(bc * p.h + iy as usize) * p.w + ix as usize]);
                    }
                }
                out[(bc * oh + oy) * ow + ox] = best;
            }
        }
    }
    out
}

/// `lightnet-core::backward_maxpool`: first strictly greater element wins.
fn cpu_maxpool_bwd(x: &[f64], dy: &[f64], p: &PoolCase) -> Vec<f64> {
    let (oh, ow, off) = (p.oh(), p.ow(), (p.pad / 2) as isize);
    let mut dx = vec![0.0; x.len()];
    for bc in 0..p.b * p.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f64::NEG_INFINITY;
                let mut at = None;
                for ky in 0..p.size {
                    for kx in 0..p.size {
                        let iy = (oy * p.sy + ky) as isize - off;
                        let ix = (ox * p.sx + kx) as isize - off;
                        if iy < 0 || iy >= p.h as isize || ix < 0 || ix >= p.w as isize {
                            continue;
                        }
                        let idx = (bc * p.h + iy as usize) * p.w + ix as usize;
                        if x[idx] > best {
                            best = x[idx];
                            at = Some(idx);
                        }
                    }
                }
                if let Some(idx) = at {
                    dx[idx] += dy[(bc * oh + oy) * ow + ox];
                }
            }
        }
    }
    dx
}

fn cpu_local_avg(x: &[f64], p: &PoolCase) -> Vec<f64> {
    let (oh, ow, off) = (p.oh(), p.ow(), (p.pad / 2) as isize);
    let mut out = vec![0.0; p.b * p.c * oh * ow];
    for bc in 0..p.b * p.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let (mut sum, mut n) = (0.0, 0usize);
                for ky in 0..p.size {
                    for kx in 0..p.size {
                        let iy = (oy * p.sy + ky) as isize - off;
                        let ix = (ox * p.sx + kx) as isize - off;
                        if iy < 0 || iy >= p.h as isize || ix < 0 || ix >= p.w as isize {
                            continue;
                        }
                        sum += x[(bc * p.h + iy as usize) * p.w + ix as usize];
                        n += 1;
                    }
                }
                if n > 0 {
                    out[(bc * oh + oy) * ow + ox] = sum / n as f64;
                }
            }
        }
    }
    out
}

fn cpu_local_avg_bwd(dy: &[f64], p: &PoolCase) -> Vec<f64> {
    let (oh, ow, off) = (p.oh(), p.ow(), (p.pad / 2) as isize);
    let mut dx = vec![0.0; p.b * p.c * p.h * p.w];
    for bc in 0..p.b * p.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut taps = Vec::new();
                for ky in 0..p.size {
                    for kx in 0..p.size {
                        let iy = (oy * p.sy + ky) as isize - off;
                        let ix = (ox * p.sx + kx) as isize - off;
                        if iy < 0 || iy >= p.h as isize || ix < 0 || ix >= p.w as isize {
                            continue;
                        }
                        taps.push((bc * p.h + iy as usize) * p.w + ix as usize);
                    }
                }
                for &t in &taps {
                    dx[t] += dy[(bc * oh + oy) * ow + ox] / taps.len() as f64;
                }
            }
        }
    }
    dx
}

#[test]
fn pool_output_size_rule_matches_cpp_and_core() {
    // C++: out = (in + pad - size) / stride + 1 with C integer division;
    // lightnet-core pool_dim is the same floor rule (no ceil mode anywhere).
    let p = POOL_CASES[1]; // yolov4-tiny size=2 stride=1 keeps the size
    assert_eq!((p.oh(), p.ow()), (7, 9));
    let p = POOL_CASES[0]; // size=2 stride=2 on 7x9 -> 4x5 (covers the odd row)
    assert_eq!((p.oh(), p.ow()), (4, 5));
    let p = POOL_CASES[4]; // pad 0 floors: 7x9 -> 3x4
    assert_eq!((p.oh(), p.ow()), (3, 4));
}

#[test]
fn maxpool_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(101);
    for p in POOL_CASES {
        let n = p.b * p.c * p.h * p.w;
        let no = p.b * p.c * p.oh() * p.ow();
        let x = rng.distinct(n);
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&p.in_shape(), &x).unwrap();
        let dyt = b.upload_f32(&p.out_shape(), &dy).unwrap();
        let name = format!("maxpool {p:?}");

        let y = b
            .maxpool_f32(&xt, &p.out_shape(), p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        assert_eq!(y.shape(), &p.out_shape());
        assert_close(&format!("{name} fwd"), &b.download_f32(&y).unwrap(), &cpu_maxpool(&f64s(&x), &p));

        let dx = b
            .maxpool_backward_f32(&xt, &dyt, p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        assert_eq!(dx.shape(), &p.in_shape());
        let dxh = b.download_f32(&dx).unwrap();
        assert_close(&format!("{name} bwd"), &dxh, &cpu_maxpool_bwd(&f64s(&x), &f64s(&dy), &p));
        fd_check(&name, &x, &dy, |xv| cpu_maxpool(xv, &p), &dxh);
    }
}

#[test]
fn maxpool_ties_resolve_to_first_tap_like_cpu_reference() {
    let b = backend();
    let mut rng = Rng(102);
    for p in [POOL_CASES[0], POOL_CASES[1], POOL_CASES[3]] {
        let n = p.b * p.c * p.h * p.w;
        let no = p.b * p.c * p.oh() * p.ow();
        // Few distinct values -> many ties inside every window.
        let x: Vec<f32> = (0..n).map(|_| (rng.next(0.0, 3.0) as i32) as f32).collect();
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&p.in_shape(), &x).unwrap();
        let dyt = b.upload_f32(&p.out_shape(), &dy).unwrap();
        let dx = b
            .maxpool_backward_f32(&xt, &dyt, p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        assert_close(
            &format!("maxpool ties {p:?}"),
            &b.download_f32(&dx).unwrap(),
            &cpu_maxpool_bwd(&f64s(&x), &f64s(&dy), &p),
        );
        // Deterministic: a second launch is bit-identical.
        let dx2 = b
            .maxpool_backward_f32(&xt, &dyt, p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        assert_eq!(b.download_f32(&dx).unwrap(), b.download_f32(&dx2).unwrap());
    }
}

#[test]
fn local_avgpool_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(103);
    for p in POOL_CASES {
        let n = p.b * p.c * p.h * p.w;
        let no = p.b * p.c * p.oh() * p.ow();
        let x = rng.vec(n, -1.0, 1.0);
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&p.in_shape(), &x).unwrap();
        let dyt = b.upload_f32(&p.out_shape(), &dy).unwrap();
        let name = format!("local_avgpool {p:?}");

        let y = b
            .local_avgpool_f32(&xt, &p.out_shape(), p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        assert_close(&format!("{name} fwd"), &b.download_f32(&y).unwrap(), &cpu_local_avg(&f64s(&x), &p));

        let dx = b
            .local_avgpool_backward_f32(&dyt, &p.in_shape(), p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg())
            .unwrap();
        let dxh = b.download_f32(&dx).unwrap();
        assert_close(&format!("{name} bwd"), &dxh, &cpu_local_avg_bwd(&f64s(&dy), &p));
        fd_check(&name, &x, &dy, |xv| cpu_local_avg(xv, &p), &dxh);
    }
}

fn cpu_global_avg(x: &[f64], planes: usize, area: usize) -> Vec<f64> {
    (0..planes).map(|k| x[k * area..(k + 1) * area].iter().sum::<f64>() / area as f64).collect()
}

#[test]
fn avgpool_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(104);
    for (batch, c, h, w) in [(3usize, 5usize, 7usize, 5usize), (2, 3, 1, 1), (1, 7, 13, 9), (4, 2, 3, 4)] {
        let (n, area) = (batch * c * h * w, h * w);
        let x = rng.vec(n, -1.0, 1.0);
        let dy = rng.vec(batch * c, -1.0, 1.0);
        let xt = b.upload_f32(&[batch, c, h, w], &x).unwrap();
        let dyt = b.upload_f32(&[batch, c, 1, 1], &dy).unwrap();
        let name = format!("avgpool {batch}x{c}x{h}x{w}");

        let y = b.avgpool_f32(&xt, &[batch, c, 1, 1], c, h, w, cfg()).unwrap();
        assert_eq!(y.shape(), &[batch, c, 1, 1]);
        assert_close(&format!("{name} fwd"), &b.download_f32(&y).unwrap(), &cpu_global_avg(&f64s(&x), batch * c, area));

        let dx = b.avgpool_backward_f32(&dyt, &[batch, c, h, w], c, h, w, cfg()).unwrap();
        let dxh = b.download_f32(&dx).unwrap();
        let want: Vec<f64> = (0..n).map(|i| dy[i / area] as f64 / area as f64).collect();
        assert_close(&format!("{name} bwd"), &dxh, &want);
        fd_check(&name, &x, &dy, |xv| cpu_global_avg(xv, batch * c, area), &dxh);
    }
}

// --------------------------------------------------------------- upsample

#[derive(Clone, Copy, Debug)]
struct UpCase {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    stride: usize,
    scale: f32,
}

fn cpu_upsample(x: &[f64], u: &UpCase) -> Vec<f64> {
    let (oh, ow, s) = (u.h * u.stride, u.w * u.stride, u.stride);
    let mut out = vec![0.0; u.b * u.c * oh * ow];
    for bc in 0..u.b * u.c {
        for y in 0..oh {
            for xx in 0..ow {
                out[(bc * oh + y) * ow + xx] = u.scale as f64 * x[(bc * u.h + y / s) * u.w + xx / s];
            }
        }
    }
    out
}

fn cpu_upsample_bwd(dy: &[f64], u: &UpCase) -> Vec<f64> {
    let (oh, ow, s) = (u.h * u.stride, u.w * u.stride, u.stride);
    let mut dx = vec![0.0; u.b * u.c * u.h * u.w];
    for bc in 0..u.b * u.c {
        for y in 0..oh {
            for xx in 0..ow {
                dx[(bc * u.h + y / s) * u.w + xx / s] += u.scale as f64 * dy[(bc * oh + y) * ow + xx];
            }
        }
    }
    dx
}

/// `lightnet-core::forward_upsample(reverse)` with taps whose target lies
/// outside the `oh x ow` output skipped (the C++ kernel's floor-sized case).
fn cpu_upsample_rev(x: &[f64], u: &UpCase, oh: usize, ow: usize) -> Vec<f64> {
    let s = u.stride;
    let mut out = vec![0.0; u.b * u.c * oh * ow];
    for bc in 0..u.b * u.c {
        for y in 0..u.h {
            for xx in 0..u.w {
                if y / s < oh && xx / s < ow {
                    out[(bc * oh + y / s) * ow + xx / s] += u.scale as f64 * x[(bc * u.h + y) * u.w + xx];
                }
            }
        }
    }
    out
}

fn cpu_upsample_rev_bwd(dy: &[f64], u: &UpCase, oh: usize, ow: usize) -> Vec<f64> {
    let s = u.stride;
    let mut dx = vec![0.0; u.b * u.c * u.h * u.w];
    for bc in 0..u.b * u.c {
        for y in 0..u.h {
            for xx in 0..u.w {
                if y / s < oh && xx / s < ow {
                    dx[(bc * u.h + y) * u.w + xx] = u.scale as f64 * dy[(bc * oh + y / s) * ow + xx / s];
                }
            }
        }
    }
    dx
}

#[test]
fn upsample_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(105);
    let cases = [
        UpCase { b: 2, c: 3, h: 5, w: 3, stride: 2, scale: 1.0 }, // yolov4-tiny upsample
        UpCase { b: 3, c: 2, h: 4, w: 7, stride: 3, scale: 0.5 },
        UpCase { b: 1, c: 5, h: 3, w: 3, stride: 1, scale: 2.0 },
        UpCase { b: 2, c: 2, h: 1, w: 1, stride: 4, scale: -1.5 },
    ];
    for u in cases {
        let (n, oh, ow) = (u.b * u.c * u.h * u.w, u.h * u.stride, u.w * u.stride);
        let no = u.b * u.c * oh * ow;
        let x = rng.vec(n, -1.0, 1.0);
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&[u.b, u.c, u.h, u.w], &x).unwrap();
        let dyt = b.upload_f32(&[u.b, u.c, oh, ow], &dy).unwrap();
        let name = format!("upsample {u:?}");

        let y = b
            .upsample_f32(&xt, &[u.b, u.c, oh, ow], u.c, u.h, u.w, u.stride, u.scale, false, cfg())
            .unwrap();
        assert_close(&format!("{name} fwd"), &b.download_f32(&y).unwrap(), &cpu_upsample(&f64s(&x), &u));

        let dx = b
            .upsample_backward_f32(&dyt, &[u.b, u.c, u.h, u.w], u.c, oh, ow, u.stride, u.scale, cfg())
            .unwrap();
        let dxh = b.download_f32(&dx).unwrap();
        assert_close(&format!("{name} bwd"), &dxh, &cpu_upsample_bwd(&f64s(&dy), &u));
        fd_check(&name, &x, &dy, |xv| cpu_upsample(xv, &u), &dxh);
    }
}

#[test]
fn upsample_reverse_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(106);
    // (case, out_h, out_w): floor-sized (C++) and ceil-sized (cuTile) outputs.
    let cases = [
        (UpCase { b: 2, c: 3, h: 8, w: 6, stride: 2, scale: 1.0 }, 4, 3),
        (UpCase { b: 2, c: 2, h: 7, w: 9, stride: 2, scale: 0.5 }, 3, 4),
        (UpCase { b: 2, c: 2, h: 7, w: 9, stride: 2, scale: 0.5 }, 4, 5),
        (UpCase { b: 1, c: 3, h: 9, w: 10, stride: 3, scale: 2.0 }, 3, 3),
    ];
    for (u, oh, ow) in cases {
        let n = u.b * u.c * u.h * u.w;
        let no = u.b * u.c * oh * ow;
        let x = rng.vec(n, -1.0, 1.0);
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&[u.b, u.c, u.h, u.w], &x).unwrap();
        let dyt = b.upload_f32(&[u.b, u.c, oh, ow], &dy).unwrap();
        let name = format!("upsample reverse {u:?} -> {oh}x{ow}");

        let y = b
            .upsample_f32(&xt, &[u.b, u.c, oh, ow], u.c, u.h, u.w, u.stride, u.scale, true, cfg())
            .unwrap();
        assert_close(&format!("{name} fwd"), &b.download_f32(&y).unwrap(), &cpu_upsample_rev(&f64s(&x), &u, oh, ow));

        let dx = b
            .upsample_backward_f32(&dyt, &[u.b, u.c, u.h, u.w], u.c, oh, ow, u.stride, u.scale, cfg())
            .unwrap();
        let dxh = b.download_f32(&dx).unwrap();
        assert_close(&format!("{name} bwd"), &dxh, &cpu_upsample_rev_bwd(&f64s(&dy), &u, oh, ow));
        fd_check(&name, &x, &dy, |xv| cpu_upsample_rev(xv, &u, oh, ow), &dxh);
    }
}

// ------------------------------------------------------------------ route

/// One route input: `channels` channels starting at `offset` of a source with
/// `total` channels (`groups`/`group_id` => `channels = total/groups`,
/// `offset = channels*group_id`, exactly as the runtime computes them).
#[derive(Clone, Copy)]
struct RouteIn {
    total: usize,
    offset: usize,
    channels: usize,
}

fn cpu_route(srcs: &[Vec<f64>], ins: &[RouteIn], batch: usize, stride: usize) -> Vec<f64> {
    let out_c: usize = ins.iter().map(|i| i.channels).sum();
    let mut out = vec![0.0; batch * out_c * stride];
    let mut base = 0;
    for (src, i) in srcs.iter().zip(ins) {
        for bb in 0..batch {
            for c in 0..i.channels {
                for s in 0..stride {
                    out[(bb * out_c + base + c) * stride + s] =
                        src[(bb * i.total + i.offset + c) * stride + s];
                }
            }
        }
        base += i.channels;
    }
    out
}

fn cpu_route_bwd(dy: &[f64], ins: &[RouteIn], batch: usize, stride: usize) -> Vec<Vec<f64>> {
    let out_c: usize = ins.iter().map(|i| i.channels).sum();
    let mut base = 0;
    ins.iter()
        .map(|i| {
            let mut g = vec![0.0; batch * i.total * stride];
            for bb in 0..batch {
                for c in 0..i.channels {
                    for s in 0..stride {
                        g[(bb * i.total + i.offset + c) * stride + s] =
                            dy[(bb * out_c + base + c) * stride + s];
                    }
                }
            }
            base += i.channels;
            g
        })
        .collect()
}

#[test]
fn route_forward_backward_and_fd_including_grouped() {
    let b = backend();
    let mut rng = Rng(107);
    let (batch, stride) = (3usize, 5 * 7);
    let grouped = |total: usize, groups: usize, id: usize| RouteIn {
        total,
        offset: total / groups * id,
        channels: total / groups,
    };
    let full = |total: usize| RouteIn { total, offset: 0, channels: total };
    let cases: Vec<(&str, Vec<RouteIn>)> = vec![
        ("plain concat", vec![full(3), full(5)]),
        ("groups=2 group_id=0", vec![grouped(8, 2, 0)]),
        ("groups=2 group_id=1", vec![grouped(8, 2, 1)]),
        ("group_id=1 + plain", vec![grouped(8, 2, 1), full(3)]),
        ("both halves of one source", vec![grouped(6, 2, 0), grouped(6, 2, 1)]),
        ("three inputs", vec![full(2), grouped(10, 2, 1), full(1)]),
    ];
    for (name, ins) in cases {
        let srcs: Vec<Vec<f32>> = ins.iter().map(|i| rng.vec(batch * i.total * stride, -1.0, 1.0)).collect();
        let out_c: usize = ins.iter().map(|i| i.channels).sum();
        let dy = rng.vec(batch * out_c * stride, -1.0, 1.0);
        let tensors: Vec<_> = srcs
            .iter()
            .zip(&ins)
            .map(|(s, i)| b.upload_f32(&[batch, i.total, 5, 7], s).unwrap())
            .collect();
        let refs: Vec<_> = tensors.iter().collect();
        let offsets: Vec<usize> = ins.iter().map(|i| i.offset).collect();
        let chans: Vec<usize> = ins.iter().map(|i| i.channels).collect();
        let totals: Vec<usize> = ins.iter().map(|i| i.total).collect();

        let y = b
            .route_concat_f32(&refs, &offsets, &chans, &totals, &[batch, out_c, 5, 7], stride, cfg())
            .unwrap();
        assert_eq!(y.shape(), &[batch, out_c, 5, 7]);
        let srcs64: Vec<Vec<f64>> = srcs.iter().map(|s| f64s(s)).collect();
        assert_close(&format!("route {name} fwd"), &b.download_f32(&y).unwrap(), &cpu_route(&srcs64, &ins, batch, stride));

        let dyt = b.upload_f32(&[batch, out_c, 5, 7], &dy).unwrap();
        let shapes: Vec<Vec<usize>> = ins.iter().map(|i| vec![batch, i.total, 5, 7]).collect();
        let grads = b.route_backward_f32(&dyt, &shapes, &offsets, &chans, &totals, stride, cfg()).unwrap();
        let want = cpu_route_bwd(&f64s(&dy), &ins, batch, stride);
        assert_eq!(grads.len(), ins.len());
        for (k, g) in grads.iter().enumerate() {
            assert_eq!(g.shape(), &shapes[k][..]);
            let gh = b.download_f32(g).unwrap();
            assert_close(&format!("route {name} bwd[{k}]"), &gh, &want[k]);
            // FD w.r.t. input k, the other inputs held fixed.
            let f = |xv: &[f64]| {
                let mut all = srcs64.clone();
                all[k] = xv.to_vec();
                cpu_route(&all, &ins, batch, stride)
            };
            fd_check(&format!("route {name}[{k}]"), &srcs[k], &dy, f, &gh);
        }
    }
}

// ------------------------------------------------------------------- crop

struct CropCase {
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
}

/// Bilinear sample of per-tap affine-adjusted values, preserving zero taps
/// outside the image like `lightnet-core`'s crop sampler.
fn cpu_crop(x: &[f64], k: &CropCase, sx: &[f32], sy: &[f32], scales: &[f32], offs: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; k.b * k.c * k.oh * k.ow];
    for bb in 0..k.b {
        for c in 0..k.c {
            for p in 0..k.oh * k.ow {
                let (fx, fy) = (sx[bb * k.oh * k.ow + p] as f64, sy[bb * k.oh * k.ow + p] as f64);
                let (ix, iy) = (fx.floor(), fy.floor());
                let (dx, dy) = (fx - ix, fy - iy);
                let tap = |tx: f64, ty: f64| -> f64 {
                    if tx < 0.0 || ty < 0.0 || tx >= k.w as f64 || ty >= k.h as f64 {
                        0.0
                    } else {
                        x[((bb * k.c + c) * k.h + ty as usize) * k.w + tx as usize]
                            * scales[bb] as f64
                            + offs[bb] as f64
                    }
                };
                let v = (1.0 - dy) * (1.0 - dx) * tap(ix, iy)
                    + dy * (1.0 - dx) * tap(ix, iy + 1.0)
                    + (1.0 - dy) * dx * tap(ix + 1.0, iy)
                    + dy * dx * tap(ix + 1.0, iy + 1.0);
                out[((bb * k.c + c) * k.oh * k.ow) + p] = v;
            }
        }
    }
    out
}

#[test]
fn crop_bilinear_forward_backward_and_fd() {
    let b = backend();
    let mut rng = Rng(108);
    let cases = [
        CropCase { b: 2, c: 3, h: 6, w: 7, oh: 4, ow: 5 },
        CropCase { b: 3, c: 1, h: 5, w: 5, oh: 7, ow: 3 },
        CropCase { b: 1, c: 2, h: 9, w: 4, oh: 9, ow: 4 },
    ];
    for k in cases {
        let n = k.b * k.c * k.h * k.w;
        let coords = k.b * k.oh * k.ow;
        let no = k.b * k.c * k.oh * k.ow;
        let x = rng.vec(n, -1.0, 1.0);
        // Coordinates reach outside the image on every side (zero taps).
        let sx = rng.vec(coords, -1.5, k.w as f32 + 0.5);
        let sy = rng.vec(coords, -1.5, k.h as f32 + 0.5);
        let scales = rng.vec(k.b, 0.5, 2.0);
        let offs = rng.vec(k.b, -1.0, 1.0);
        let channel_offsets: Vec<f32> = (0..k.b)
            .flat_map(|batch| std::iter::repeat_n(offs[batch], k.c))
            .collect();
        let dy = rng.vec(no, -1.0, 1.0);
        let xt = b.upload_f32(&[k.b, k.c, k.h, k.w], &x).unwrap();
        let sxt = b.upload_f32(&[coords], &sx).unwrap();
        let syt = b.upload_f32(&[coords], &sy).unwrap();
        let st = b.upload_f32(&[k.b], &scales).unwrap();
        let ot = b.upload_f32(&[k.b, k.c], &channel_offsets).unwrap();
        let sat = b.upload_f32(&[k.b], &vec![1.0; k.b]).unwrap();
        let exp = b.upload_f32(&[k.b], &vec![1.0; k.b]).unwrap();
        let dyt = b.upload_f32(&[k.b, k.c, k.oh, k.ow], &dy).unwrap();
        let name = format!("crop {}x{}x{}x{}->{}x{}", k.b, k.c, k.h, k.w, k.oh, k.ow);

        let y = b
            .crop_bilinear_f32(
                &xt,
                &[k.b, k.c, k.oh, k.ow],
                k.c,
                k.h,
                k.w,
                &sxt,
                &syt,
                &st,
                &ot,
                &sat,
                &exp,
                false,
                cfg(),
            )
            .unwrap();
        assert_close(
            &format!("{name} fwd"),
            &b.download_f32(&y).unwrap(),
            &cpu_crop(&f64s(&x), &k, &sx, &sy, &scales, &offs),
        );

        let dx = b
            .crop_bilinear_backward_f32(&dyt, &[k.b, k.c, k.h, k.w], k.c, k.h, k.w, &sxt, &syt, &st, cfg())
            .unwrap();
        let dxh = b.download_f32(&dx).unwrap();
        // Backward reference: gradient of the linear map = transpose, i.e. the
        // forward applied to unit vectors (offsets do not matter for dL/dx).
        let zero_offs = vec![0.0f32; k.b];
        let dy64 = f64s(&dy);
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = vec![0.0; n];
                e[i] = 1.0;
                cpu_crop(&e, &k, &sx, &sy, &scales, &zero_offs).iter().zip(&dy64).map(|(a, b)| a * b).sum()
            })
            .collect();
        assert_close(&format!("{name} bwd"), &dxh, &want);
        fd_check(&name, &x, &dy, |xv| cpu_crop(xv, &k, &sx, &sy, &scales, &offs), &dxh);
    }
}
