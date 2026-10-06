//! softmax / reorg / dropout / scale_channels / connected: forward parity,
//! backward parity, central finite-difference checks (f64 CPU references) and
//! edge shapes.  The references are written independently of
//! `darknet_cuda::host_ops` (which only serves the dropout RNG bit-match).

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend, dropout_unit};
use darknet_cuda_oxide::OxideBackend;

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

/// Central finite difference of `L(x) = sum(dy * f(x))` compared with `got`.
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

// ---------------------------------------------------------------- softmax

fn cpu_softmax(x: &[f64], group: usize) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (s, d) in x.chunks(group).zip(out.chunks_mut(group)) {
        let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = s.iter().map(|v| (v - max).exp()).sum();
        for (o, v) in d.iter_mut().zip(s) {
            *o = (v - max).exp() / sum;
        }
    }
    out
}

/// (batch * groups, group_size) pairs, incl. size-1 groups and one big group.
const SOFTMAX_CASES: [(usize, usize); 6] = [(1, 1), (3, 1), (1, 10), (6, 7), (2, 1000), (257, 3)];

#[test]
fn softmax_forward_backward_and_fd() {
    let be = backend();
    let mut rng = Rng(0x5eed_0001);
    for &(n_groups, group) in &SOFTMAX_CASES {
        let n = n_groups * group;
        let x = rng.vec(n, -3.0, 3.0);
        let dy = rng.vec(n, -1.0, 1.0);
        let xt = be.upload_f32(&[n_groups, group], &x).unwrap();
        let y = be.softmax_f32(&xt, &[n_groups, group], group, cfg()).unwrap();
        let yh = be.download_f32(&y).unwrap();
        let want = cpu_softmax(&f64s(&x), group);
        assert_close(&format!("softmax {n_groups}x{group}"), &yh, &want);
        let dyt = be.upload_f32(&[n_groups, group], &dy).unwrap();
        let dx = be.download_f32(&be.softmax_backward_f32(&y, &dyt, &[n_groups, group], group, cfg()).unwrap()).unwrap();
        // backward parity: y * (dy - <y, dy>) per group
        let mut want_dx = vec![0.0f64; n];
        for g in 0..n_groups {
            let r = g * group..(g + 1) * group;
            let dot: f64 = r.clone().map(|i| want[i] * dy[i] as f64).sum();
            for i in r {
                want_dx[i] = want[i] * (dy[i] as f64 - dot);
            }
        }
        assert_close(&format!("softmax bwd {n_groups}x{group}"), &dx, &want_dx);
        if n <= 64 {
            fd_check("softmax", &x, &dy, |xv| cpu_softmax(xv, group), &dx);
        }
    }
}

#[test]
fn softmax_is_stable_for_large_logits() {
    let be = backend();
    let x = [1000.0f32, 1001.0, 999.0, -1000.0, -1001.0, -999.0];
    let xt = be.upload_f32(&[2, 3], &x).unwrap();
    let y = be.download_f32(&be.softmax_f32(&xt, &[2, 3], 3, cfg()).unwrap()).unwrap();
    assert!(y.iter().all(|v| v.is_finite()));
    assert_close("softmax large", &y, &cpu_softmax(&f64s(&x), 3));
    for g in y.chunks(3) {
        assert!((g.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn softmax_rejects_bad_group_size() {
    let be = backend();
    let xt = be.upload_f32(&[7], &[0.0; 7]).unwrap();
    assert!(be.softmax_f32(&xt, &[7], 0, cfg()).is_err());
    assert!(be.softmax_f32(&xt, &[7], 3, cfg()).is_err());
    assert!(be.softmax_backward_f32(&xt, &xt, &[7], 2, cfg()).is_err());
}

// ------------------------------------------------------------------ reorg

/// Non-reverse reorg forward, written as the CPU runtime does (gather).
fn cpu_reorg_fwd(x: &[f64], b: usize, c: usize, h: usize, w: usize, s: usize) -> Vec<f64> {
    let (oc, oh, ow) = (c * s * s, h / s, w / s);
    let mut out = vec![0.0; b * oc * oh * ow];
    for bi in 0..b {
        for k in 0..oc {
            let (in_c, off) = (k % c, k / c);
            for y in 0..oh {
                for xx in 0..ow {
                    let (iy, ix) = (y * s + off / s, xx * s + off % s);
                    out[((bi * oc + k) * oh + y) * ow + xx] = x[((bi * c + in_c) * h + iy) * w + ix];
                }
            }
        }
    }
    out
}

/// Reverse reorg forward (small `[b, cs, hs, ws]` -> big), written as a scatter.
fn cpu_reorg_rev_fwd(x: &[f64], b: usize, cs: usize, hs: usize, ws: usize, s: usize) -> Vec<f64> {
    let cb = cs / (s * s);
    let (hb, wb) = (hs * s, ws * s);
    let mut out = vec![0.0; b * cb * hb * wb];
    for bi in 0..b {
        for k in 0..cs {
            let (c2, off) = (k % cb, k / cb);
            for j in 0..hs {
                for i in 0..ws {
                    let (h2, w2) = (j * s + off / s, i * s + off % s);
                    out[((bi * cb + c2) * hb + h2) * wb + w2] = x[((bi * cs + k) * hs + j) * ws + i];
                }
            }
        }
    }
    out
}

/// (batch, c_in, h, w, stride) of the non-reverse layer; h/w not always divisible.
const REORG_CASES: [(usize, usize, usize, usize, usize); 6] = [
    (1, 1, 2, 2, 2),
    (2, 3, 6, 8, 2),
    (2, 4, 9, 7, 2), // odd spatial dims: last row/column unused
    (1, 2, 6, 6, 3),
    (3, 2, 4, 5, 1), // stride 1 = identity
    (2, 5, 8, 12, 4),
];

fn reorg_shapes(b: usize, c: usize, h: usize, w: usize, s: usize) -> ([usize; 4], [usize; 4]) {
    ([b, c, h, w], [b, c * s * s, h / s, w / s])
}

#[test]
fn reorg_forward_backward_and_fd() {
    let be = backend();
    let mut rng = Rng(0x5eed_0002);
    for &(b, c, h, w, s) in &REORG_CASES {
        let (ish, osh) = reorg_shapes(b, c, h, w, s);
        let x = rng.vec(ish.iter().product(), -1.0, 1.0);
        let dy = rng.vec(osh.iter().product(), -1.0, 1.0);
        let xt = be.upload_f32(&ish, &x).unwrap();
        let y = be.reorg_f32(&xt, &ish, &osh, s, false, cfg()).unwrap();
        assert_eq!(y.shape(), &osh);
        let name = format!("reorg {b}x{c}x{h}x{w} s{s}");
        assert_close(&name, &be.download_f32(&y).unwrap(), &cpu_reorg_fwd(&f64s(&x), b, c, h, w, s));
        let dyt = be.upload_f32(&osh, &dy).unwrap();
        let dx = be.reorg_backward_f32(&dyt, &ish, &osh, s, false, cfg()).unwrap();
        assert_eq!(dx.shape(), &ish);
        let dxh = be.download_f32(&dx).unwrap();
        fd_check(&name, &x, &dy, |xv| cpu_reorg_fwd(xv, b, c, h, w, s), &dxh);

        // reverse layer: input is the small tensor, output the big one.
        let xr = rng.vec(osh.iter().product(), -1.0, 1.0);
        // reverse output = [b, c, (h/s)*s, (w/s)*s]
        let rsh = [b, c, h / s * s, w / s * s];
        let dyr = rng.vec(rsh.iter().product(), -1.0, 1.0);
        let xrt = be.upload_f32(&osh, &xr).unwrap();
        let yr = be.reorg_f32(&xrt, &osh, &rsh, s, true, cfg()).unwrap();
        let rname = format!("reorg reverse {b}x{c}x{h}x{w} s{s}");
        assert_close(&rname, &be.download_f32(&yr).unwrap(), &cpu_reorg_rev_fwd(&f64s(&xr), b, c * s * s, h / s, w / s, s));
        let dyrt = be.upload_f32(&rsh, &dyr).unwrap();
        let dxr = be.download_f32(&be.reorg_backward_f32(&dyrt, &osh, &rsh, s, true, cfg()).unwrap()).unwrap();
        fd_check(&rname, &xr, &dyr, |xv| cpu_reorg_rev_fwd(xv, b, c * s * s, h / s, w / s, s), &dxr);
    }
}

#[test]
fn reorg_round_trip_is_identity_on_covered_area() {
    let be = backend();
    let mut rng = Rng(0x5eed_0003);
    let (ish, osh) = reorg_shapes(2, 3, 8, 6, 2);
    let x = rng.vec(ish.iter().product(), -1.0, 1.0);
    let xt = be.upload_f32(&ish, &x).unwrap();
    let small = be.reorg_f32(&xt, &ish, &osh, 2, false, cfg()).unwrap();
    // reverse layer applied to the small tensor restores the original.
    let back = be.reorg_f32(&small, &osh, &ish, 2, true, cfg()).unwrap();
    assert_eq!(be.download_f32(&back).unwrap(), x);
}

#[test]
fn reorg_rejects_inconsistent_shapes() {
    let be = backend();
    let xt = be.upload_f32(&[1, 2, 4, 4], &[0.0; 32]).unwrap();
    assert!(be.reorg_f32(&xt, &[1, 2, 4, 4], &[1, 8, 2, 2], 0, false, cfg()).is_err());
    assert!(be.reorg_f32(&xt, &[1, 2, 4, 4], &[1, 7, 2, 2], 2, false, cfg()).is_err());
    assert!(be.reorg_f32(&xt, &[1, 2, 4, 4], &[1, 8, 3, 2], 2, false, cfg()).is_err());
    // element count of the tensor must match the declared input shape
    assert!(be.reorg_f32(&xt, &[1, 2, 6, 6], &[1, 8, 3, 3], 2, false, cfg()).is_err());
}

// ---------------------------------------------------------------- dropout

#[test]
fn dropout_mask_matches_counter_hash_and_backward() {
    let be = backend();
    let mut rng = Rng(0x5eed_0004);
    // lengths straddling the 256-thread block; seeds incl. 0 and high bits set
    for &(n, p, seed) in &[(1usize, 0.5f32, 0u64), (255, 0.3, 7), (256, 0.5, u64::MAX), (1000, 0.9, 0x9e37_79b9), (4099, 0.1, 42)] {
        let x = rng.vec(n, -2.0, 2.0);
        let dy = rng.vec(n, -1.0, 1.0);
        let xt = be.upload_f32(&[n], &x).unwrap();
        let (y, mask) = be.dropout_f32(&xt, &[n], p, seed, cfg()).unwrap();
        let (yh, mh) = (be.download_f32(&y).unwrap(), be.download_f32(&mask).unwrap());
        let scale = 1.0 / (1.0 - p);
        for i in 0..n {
            let want_m = if dropout_unit(seed, i as u64) < p { 0.0 } else { scale };
            assert_eq!(mh[i], want_m, "mask[{i}] n={n} p={p}");
            assert_eq!(yh[i], x[i] * want_m, "out[{i}]");
        }
        let dyt = be.upload_f32(&[n], &dy).unwrap();
        let dx = be.download_f32(&be.dropout_backward_f32(&dyt, &mask, &[n], cfg()).unwrap()).unwrap();
        for i in 0..n {
            assert_eq!(dx[i], dy[i] * mh[i], "dx[{i}]");
        }
        // FD of the (linear) masked scale with the mask held fixed
        if n <= 300 {
            let mf = f64s(&mh);
            fd_check("dropout", &x, &dy, |xv| xv.iter().zip(&mf).map(|(a, b)| a * b).collect(), &dx);
        }
    }
}

#[test]
fn dropout_is_deterministic_seeded_and_unbiased() {
    let be = backend();
    let n = 200_000;
    let xt = be.upload_f32(&[n], &vec![1.0; n]).unwrap();
    let run = |seed| {
        let (_, m) = be.dropout_f32(&xt, &[n], 0.25, seed, cfg()).unwrap();
        be.download_f32(&m).unwrap()
    };
    let (a, b, c) = (run(11), run(11), run(12));
    assert_eq!(a, b, "same seed must reproduce the mask");
    assert_ne!(a, c, "different seeds must differ");
    let kept = a.iter().filter(|&&m| m != 0.0).count() as f64 / n as f64;
    assert!((kept - 0.75).abs() < 0.01, "keep rate {kept}");
    // inverted dropout: expected output equals the input
    let mean = a.iter().map(|&m| m as f64).sum::<f64>() / n as f64;
    assert!((mean - 1.0).abs() < 0.02, "mean multiplier {mean}");
}

#[test]
fn dropout_zero_probability_is_identity_and_bad_probability_errors() {
    let be = backend();
    let x = [0.5f32, -1.5, 2.0];
    let xt = be.upload_f32(&[3], &x).unwrap();
    let (y, m) = be.dropout_f32(&xt, &[3], 0.0, 99, cfg()).unwrap();
    assert_eq!(be.download_f32(&y).unwrap(), x);
    assert_eq!(be.download_f32(&m).unwrap(), vec![1.0; 3]);
    assert!(be.dropout_f32(&xt, &[3], 1.0, 1, cfg()).is_err());
    assert!(be.dropout_f32(&xt, &[3], -0.1, 1, cfg()).is_err());
    assert!(be.dropout_f32(&xt, &[4], 0.5, 1, cfg()).is_err()); // shape/len mismatch
}

// --------------------------------------------------------- scale_channels

fn sc_index(i: usize, c: usize, hw: usize, wh: bool) -> usize {
    let (b, r) = (i / (c * hw), i % (c * hw));
    if wh { b * hw + r % hw } else { b * c + r / hw }
}

fn cpu_scale(scale: &[f64], src: &[f64], c: usize, hw: usize, wh: bool) -> Vec<f64> {
    (0..src.len()).map(|i| scale[sc_index(i, c, hw, wh)] * src[i]).collect()
}

/// (batch, c, h, w): includes hw = 1 and a large plane.
const SC_CASES: [[usize; 4]; 5] = [[1, 1, 1, 1], [2, 3, 4, 5], [3, 1, 3, 3], [2, 17, 1, 1], [2, 8, 33, 31]];

#[test]
fn scale_channels_forward_backward_and_fd() {
    let be = backend();
    let mut rng = Rng(0x5eed_0005);
    for shape in SC_CASES {
        let [b, c, h, w] = shape;
        let hw = h * w;
        let n = b * c * hw;
        for wh in [false, true] {
            let sl = if wh { b * hw } else { b * c };
            let (scale, src, dy) = (rng.vec(sl, -1.5, 1.5), rng.vec(n, -1.0, 1.0), rng.vec(n, -1.0, 1.0));
            let st = be.upload_f32(&[sl], &scale).unwrap();
            let srt = be.upload_f32(&shape, &src).unwrap();
            let name = format!("scale_channels {shape:?} wh={wh}");
            let out = be.scale_channels_f32(&st, &srt, &shape, wh, cfg()).unwrap();
            assert_eq!(out.shape(), &shape);
            assert_close(&name, &be.download_f32(&out).unwrap(), &cpu_scale(&f64s(&scale), &f64s(&src), c, hw, wh));

            let dyt = be.upload_f32(&shape, &dy).unwrap();
            let (ds, dsrc) = be.scale_channels_backward_f32(&dyt, &st, &srt, &shape, wh, cfg()).unwrap();
            assert_eq!(ds.shape(), &[sl]);
            let (dsh, dsrch) = (be.download_f32(&ds).unwrap(), be.download_f32(&dsrc).unwrap());
            let mut want_ds = vec![0.0f64; sl];
            for i in 0..n {
                want_ds[sc_index(i, c, hw, wh)] += dy[i] as f64 * src[i] as f64;
            }
            assert_close(&format!("{name} dscale"), &dsh, &want_ds);
            let want_dsrc: Vec<f64> = (0..n).map(|i| dy[i] as f64 * scale[sc_index(i, c, hw, wh)] as f64).collect();
            assert_close(&format!("{name} dsource"), &dsrch, &want_dsrc);
            if n <= 120 {
                let sf = f64s(&src);
                fd_check(&format!("{name} scale"), &scale, &dy, |sv| cpu_scale(sv, &sf, c, hw, wh), &dsh);
                let scf = f64s(&scale);
                fd_check(&format!("{name} source"), &src, &dy, |xv| cpu_scale(&scf, xv, c, hw, wh), &dsrch);
            }
        }
    }
}

#[test]
fn scale_channels_rejects_bad_layouts() {
    let be = backend();
    let src = be.upload_f32(&[1, 2, 2, 2], &[1.0; 8]).unwrap();
    let scale = be.upload_f32(&[2], &[1.0; 2]).unwrap();
    assert!(be.scale_channels_f32(&scale, &src, &[1, 2, 2, 2], true, cfg()).is_err()); // needs 4 scales
    assert!(be.scale_channels_f32(&scale, &src, &[1, 2, 2], false, cfg()).is_err());
    assert!(be.scale_channels_f32(&scale, &src, &[1, 2, 2, 3], false, cfg()).is_err());
}

// -------------------------------------------------------------- connected

fn cpu_connected(x: &[f64], w: &[f64], bias: &[f64], batch: usize, inputs: usize, outputs: usize) -> Vec<f64> {
    let mut out = vec![0.0; batch * outputs];
    for b in 0..batch {
        for o in 0..outputs {
            out[b * outputs + o] = bias[o] + (0..inputs).map(|i| x[b * inputs + i] * w[o * inputs + i]).sum::<f64>();
        }
    }
    out
}

/// (batch, inputs, outputs): 1x1x1, vectors, ragged vs the 64x64x16 GEMM tile, wide.
const CONNECTED_CASES: [(usize, usize, usize); 6] =
    [(1, 1, 1), (1, 5, 1), (4, 1, 3), (3, 130, 70), (65, 17, 66), (2, 300, 9)];

#[test]
fn connected_forward_backward_and_fd() {
    let be = backend();
    let mut rng = Rng(0x5eed_0006);
    for &(batch, inputs, outputs) in &CONNECTED_CASES {
        let x = rng.vec(batch * inputs, -1.0, 1.0);
        let w = rng.vec(outputs * inputs, -0.5, 0.5);
        let bias = rng.vec(outputs, -1.0, 1.0);
        let delta = rng.vec(batch * outputs, -1.0, 1.0);
        let name = format!("connected b{batch} i{inputs} o{outputs}");
        let xt = be.upload_f32(&[batch, inputs], &x).unwrap();
        let wt = be.upload_f32(&[outputs, inputs], &w).unwrap();
        let bt = be.upload_f32(&[outputs], &bias).unwrap();
        let out = be.connected_f32(&xt, &wt, &bt, batch, inputs, outputs, cfg()).unwrap();
        assert_eq!(out.shape(), &[batch, outputs]);
        assert_close(&name, &be.download_f32(&out).unwrap(), &cpu_connected(&f64s(&x), &f64s(&w), &f64s(&bias), batch, inputs, outputs));

        let dt = be.upload_f32(&[batch, outputs], &delta).unwrap();
        let (dx, dw, db) = be.connected_backward_f32(&xt, &wt, &dt, batch, inputs, outputs, cfg()).unwrap();
        assert_eq!((dx.shape(), dw.shape(), db.shape()), (&[batch, inputs][..], &[outputs, inputs][..], &[outputs][..]));
        let (dxh, dwh, dbh) = (be.download_f32(&dx).unwrap(), be.download_f32(&dw).unwrap(), be.download_f32(&db).unwrap());
        let (xf, wf, bf) = (f64s(&x), f64s(&w), f64s(&bias));
        let mut want = (vec![0.0f64; batch * inputs], vec![0.0f64; outputs * inputs], vec![0.0f64; outputs]);
        for b in 0..batch {
            for o in 0..outputs {
                let g = delta[b * outputs + o] as f64;
                want.2[o] += g;
                for i in 0..inputs {
                    want.1[o * inputs + i] += g * xf[b * inputs + i];
                    want.0[b * inputs + i] += g * wf[o * inputs + i];
                }
            }
        }
        assert_close(&format!("{name} dx"), &dxh, &want.0);
        assert_close(&format!("{name} dw"), &dwh, &want.1);
        assert_close(&format!("{name} db"), &dbh, &want.2);
        if batch * inputs * outputs <= 200 {
            fd_check(&format!("{name} x"), &x, &delta, |v| cpu_connected(v, &wf, &bf, batch, inputs, outputs), &dxh);
            fd_check(&format!("{name} w"), &w, &delta, |v| cpu_connected(&xf, v, &bf, batch, inputs, outputs), &dwh);
            fd_check(&format!("{name} b"), &bias, &delta, |v| cpu_connected(&xf, &wf, v, batch, inputs, outputs), &dbh);
        }
    }
}

#[test]
fn connected_rejects_bad_shapes() {
    let be = backend();
    let x = be.upload_f32(&[2, 3], &[0.0; 6]).unwrap();
    let w = be.upload_f32(&[4, 3], &[0.0; 12]).unwrap();
    let b = be.upload_f32(&[4], &[0.0; 4]).unwrap();
    let bad_b = be.upload_f32(&[3], &[0.0; 3]).unwrap();
    assert!(be.connected_f32(&x, &w, &b, 2, 3, 4, cfg()).is_ok());
    assert!(be.connected_f32(&x, &w, &bad_b, 2, 3, 4, cfg()).is_err());
    assert!(be.connected_f32(&x, &w, &b, 2, 4, 3, cfg()).is_err());
    assert!(be.connected_f32(&x, &w, &b, 0, 3, 4, cfg()).is_err());
    let delta = be.upload_f32(&[2, 3], &[0.0; 6]).unwrap();
    assert!(be.connected_backward_f32(&x, &w, &delta, 2, 3, 4, cfg()).is_err());
}

// -------------------------------------------------------------- dropblock

/// Independent DropBlock reference (C++ `dropblock_fast_kernel` semantics with
/// per-plane random values `dropout_unit(seed, plane * h * w + i)`): returns
/// the per-element multiplier mask.
fn ref_dropblock_mask(shape: [usize; 4], prob: f32, bs: usize, seed: u64) -> Vec<f32> {
    let [n, c, h, w] = shape;
    let mut mask = vec![1.0f32; n * c * h * w];
    for b in 0..n {
        let mut dropped = 0usize;
        for ch in 0..c {
            let plane = b * c + ch;
            let mut best: Option<(i32, usize)> = None;
            for i in 0..h * w {
                let u = dropout_unit(seed, (plane * h * w + i) as u64);
                if u < prob {
                    let q = (u * 1_000_000.0) as i32;
                    if q < 1_000_000 && best.is_none_or(|(bq, _)| q < bq) {
                        best = Some((q, i));
                    }
                }
            }
            if let Some((_, i)) = best {
                let mut bx = i % w;
                let mut by = i / w;
                if bx > w - bs {
                    bx -= w - bs;
                }
                if by > h - bs {
                    by -= h - bs;
                }
                bx = bx.min(w - bs);
                by = by.min(h - bs);
                for y in by..by + bs {
                    for x in bx..bx + bs {
                        mask[plane * h * w + y * w + x] = 0.0;
                    }
                }
                dropped += bs * bs;
            }
        }
        let keep = 1.0 - dropped as f32 / (c * h * w) as f32;
        let scale = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        for m in &mut mask[b * c * h * w..(b + 1) * c * h * w] {
            *m *= scale;
        }
    }
    mask
}

#[test]
fn dropblock_matches_reference_backward_is_exact_and_fd() {
    let be = backend();
    let mut rng = Rng(0x5eed_00d8);
    // (shape, per-cell prob, block size, seed): non-square planes, blocks as
    // large as the plane, a block clipped at the border, many planes
    let cases: [([usize; 4], f32, usize, u64); 7] = [
        ([1, 1, 4, 4], 1.0, 2, 1),
        ([2, 3, 5, 7], 0.2, 3, 7),
        ([2, 4, 6, 6], 0.05, 1, 99),
        ([3, 2, 3, 3], 0.5, 3, u64::MAX),
        ([2, 5, 9, 4], 0.1, 4, 0),
        ([4, 64, 13, 13], 0.02, 5, 0x9e37_79b9),
        ([1, 3, 2, 8], 0.3, 2, 5),
    ];
    for (shape, prob, bs, seed) in cases {
        let len: usize = shape.iter().product();
        let x = rng.vec(len, -2.0, 2.0);
        let dy = rng.vec(len, -1.0, 1.0);
        let xt = be.upload_f32(&shape, &x).unwrap();
        let (y, mask) = be.dropblock_f32(&xt, &shape, prob, bs, seed, cfg()).unwrap();
        let (yh, mh) = (be.download_f32(&y).unwrap(), be.download_f32(&mask).unwrap());
        let want = ref_dropblock_mask(shape, prob, bs, seed);
        let name = format!("dropblock {shape:?} p={prob} bs={bs}");
        assert_eq!(mh, want, "{name}: mask");
        for i in 0..len {
            assert_eq!(yh[i], x[i] * want[i], "{name}: out[{i}]");
        }
        if prob >= 0.05 {
            assert!(mh.contains(&0.0), "{name}: mask should drop something");
        }
        // exact backward: dx = dy * mask
        let dyt = be.upload_f32(&shape, &dy).unwrap();
        let dx = be.download_f32(&be.dropout_backward_f32(&dyt, &mask, &shape, cfg()).unwrap()).unwrap();
        for i in 0..len {
            assert_eq!(dx[i], dy[i] * want[i], "{name}: dx[{i}]");
        }
        if len <= 200 {
            let mf = f64s(&want);
            fd_check(&name, &x, &dy, |xv| xv.iter().zip(&mf).map(|(a, b)| a * b).collect(), &dx);
        }
        // host default (trait) implementation agrees bit for bit
        let (hy, hm) = darknet_cuda::host_ops::dropblock(&x, &shape, prob, bs, seed);
        assert_eq!(hm, mh, "{name}: host mask");
        assert_eq!(hy, yh, "{name}: host out");
    }
}

#[test]
fn dropblock_is_deterministic_and_rejects_bad_arguments() {
    let be = backend();
    let shape = [2usize, 8, 6, 6];
    let x = vec![1.0f32; shape.iter().product()];
    let xt = be.upload_f32(&shape, &x).unwrap();
    let run = |seed| {
        let (_, m) = be.dropblock_f32(&xt, &shape, 0.1, 3, seed, cfg()).unwrap();
        be.download_f32(&m).unwrap()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
    // probability 0: identity
    let (y, m) = be.dropblock_f32(&xt, &shape, 0.0, 3, 5, cfg()).unwrap();
    assert_eq!(be.download_f32(&y).unwrap(), x);
    assert!(be.download_f32(&m).unwrap().iter().all(|&v| v == 1.0));
    assert!(be.dropblock_f32(&xt, &shape, 0.1, 0, 5, cfg()).is_err());
    assert!(be.dropblock_f32(&xt, &shape, 0.1, 7, 5, cfg()).is_err());
    assert!(be.dropblock_f32(&xt, &[2, 8, 6], 0.1, 3, 5, cfg()).is_err());
}

// ------------------------------------------- constrain / gradient centralisation

#[test]
fn constrain_clamps_like_cpp_including_nan() {
    let be = backend();
    let x = [0.5f32, -3.0, 3.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0, -1.0, 0.0];
    let xt = be.upload_f32(&[x.len()], &x).unwrap();
    let got = be.download_f32(&be.constrain_f32(&xt, &[x.len()], 1.0, cfg()).unwrap()).unwrap();
    // fminf(1, fmaxf(-1, v)): NaN -> -1
    assert_eq!(got, vec![0.5, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.0]);
    assert_eq!(got, darknet_cuda::host_ops::constrain(&x, 1.0));
    let big = Rng(0x5eed_00c1).vec(10_000, -5.0, 5.0);
    let bt = be.upload_f32(&[big.len()], &big).unwrap();
    let got = be.download_f32(&be.constrain_f32(&bt, &[big.len()], 0.25, cfg()).unwrap()).unwrap();
    for (g, v) in got.iter().zip(&big) {
        assert_eq!(*g, v.clamp(-0.25, 0.25));
    }
}

#[test]
fn gradient_centralization_matches_reference_and_is_noop_off_warp_sizes() {
    let be = backend();
    let mut rng = Rng(0x5eed_00c2);
    for &(filters, f_size) in &[(1usize, 32usize), (5, 64), (70, 288), (3, 18), (4, 9)] {
        let x = rng.vec(filters * f_size, -1.0, 1.0);
        let xt = be.upload_f32(&[filters, f_size], &x).unwrap();
        let got = be.download_f32(&be.gradient_centralization_f32(&xt, &[filters, f_size], f_size, cfg()).unwrap()).unwrap();
        assert_eq!(got, darknet_cuda::host_ops::gradient_centralization(&x, f_size), "{filters}x{f_size}: host");
        if f_size % 32 == 0 {
            for (f, (g, src)) in got.chunks(f_size).zip(x.chunks(f_size)).enumerate() {
                let mean: f64 = src.iter().map(|&v| v as f64).sum::<f64>() / f_size as f64;
                for (a, b) in g.iter().zip(src) {
                    assert!((*a as f64 - (*b as f64 - mean)).abs() < 1e-5, "filter {f}");
                }
                assert!(g.iter().map(|&v| v as f64).sum::<f64>().abs() < 1e-4, "filter {f} mean not removed");
            }
        } else {
            assert_eq!(got, x, "C++ skips filters that are not a multiple of the warp");
        }
    }
    let xt = be.upload_f32(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(be.gradient_centralization_f32(&xt, &[4], 3, cfg()).is_err());
}

#[test]
fn channel_shuffle_matches_reference_and_inverse_round_trips() {
    let b = backend();
    let mut rng = Rng(77);
    for (batch, c, h, w, groups) in [(2usize, 6usize, 3usize, 5usize, 2usize), (1, 12, 2, 2, 3), (3, 8, 1, 7, 4)] {
        let n = batch * c * h * w;
        let x = rng.vec(n, -1.0, 1.0);
        let shape = [batch, c, h, w];
        let xt = b.upload_f32(&shape, &x).unwrap();
        let shuffled = b.channel_shuffle_f32(&xt, &shape, groups, false, cfg()).unwrap();
        // independent reference: output channel j*groups + i takes input channel i*(c/groups) + j
        let (cols, len) = (c / groups, h * w);
        let mut want = vec![0.0f64; n];
        for bb in 0..batch {
            for i in 0..groups {
                for j in 0..cols {
                    for s in 0..len {
                        want[(bb * c + j * groups + i) * len + s] = x[(bb * c + i * cols + j) * len + s] as f64;
                    }
                }
            }
        }
        assert_close("channel_shuffle", &b.download_f32(&shuffled).unwrap(), &want);
        let back = b.channel_shuffle_f32(&shuffled, &shape, groups, true, cfg()).unwrap();
        assert_eq!(b.download_f32(&back).unwrap(), x);
    }
}

fn host_moe_router_gates(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    experts: usize,
    top_k: usize,
    temperature: f32,
) -> Vec<f32> {
    let inv_spatial = 1.0 / spatial as f32;
    let mut result = vec![0.0; batch * experts];
    for b in 0..batch {
        let mut pooled = vec![0.0; channels];
        for c in 0..channels {
            let base = (b * channels + c) * spatial;
            pooled[c] = input[base..base + spatial].iter().sum::<f32>() * inv_spatial;
        }
        let mut logits = (0..experts)
            .map(|e| {
                let mut value = biases[e];
                for c in 0..channels {
                    value += weights[e * channels + c] * pooled[c];
                }
                value / temperature.max(1e-6)
            })
            .collect::<Vec<_>>();
        if top_k > 0 && top_k < experts {
            let mut ranked = logits.clone();
            ranked.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let threshold = ranked[top_k - 1];
            for value in &mut logits {
                if *value < threshold {
                    *value = -f32::MAX;
                }
            }
        }
        let max = logits.iter().copied().fold(-f32::MAX, f32::max);
        let mut sum = 0.0f32;
        for (e, value) in logits.iter().enumerate() {
            let exp = if *value <= -f32::MAX * 0.5 { 0.0 } else { (*value - max).exp() };
            result[b * experts + e] = exp;
            sum += exp;
        }
        if sum <= 0.0 || !sum.is_finite() {
            result[b * experts..(b + 1) * experts].fill(1.0 / experts as f32);
        } else {
            for gate in &mut result[b * experts..(b + 1) * experts] {
                *gate /= sum;
            }
        }
    }
    result
}

#[test]
fn moe_router_gates_match_host_for_soft_and_tied_top_k_routes() {
    let be = backend();
    let (batch, channels, spatial, experts) = (3usize, 4usize, 5usize, 4usize);
    let mut rng = Rng(0x4d4f_455f_524f_5554);
    let input = rng.vec(batch * channels * spatial, -1.5, 1.5);
    let weights = rng.vec(experts * channels, -0.7, 0.7);
    let biases = rng.vec(experts, -0.3, 0.3);
    let input_t = be.upload_f32(&[batch, channels, spatial], &input).unwrap();
    let weights_t = be.upload_f32(&[experts, channels], &weights).unwrap();
    let biases_t = be.upload_f32(&[experts], &biases).unwrap();

    for top_k in [0usize, 1, 2, experts, experts + 2] {
        let want = host_moe_router_gates(
            &input, &weights, &biases, batch, channels, spatial, experts, top_k, 1.7,
        );
        let got = be
            .moe_router_gates_f32(
                &input_t,
                &weights_t,
                &biases_t,
                batch,
                channels,
                spatial,
                experts,
                top_k,
                1.7,
                cfg(),
            )
            .unwrap();
        assert_close(
            &format!("moe_router top_k={top_k}"),
            &be.download_f32(&got).unwrap(),
            &f64s(&want),
        );
    }

    // C++ keeps every expert tied with the kth logit instead of arbitrarily
    // selecting only one of the equal maxima.
    let tied_biases = vec![2.0; experts];
    let tied_biases_t = be.upload_f32(&[experts], &tied_biases).unwrap();
    let tied_weights_t = be
        .upload_f32(&[experts, channels], &vec![0.0; experts * channels])
        .unwrap();
    let tied = be
        .moe_router_gates_f32(
            &input_t,
            &tied_weights_t,
            &tied_biases_t,
            batch,
            channels,
            spatial,
            experts,
            1,
            1.0,
            cfg(),
        )
        .unwrap();
    let uniform = vec![0.25; batch * experts]
        .iter()
        .map(|&v| v as f64)
        .collect::<Vec<_>>();
    assert_close(
        "moe_router tied top_k",
        &be.download_f32(&tied).unwrap(),
        &uniform,
    );
}

#[test]
fn moe_pool_and_gate_dots_match_host_for_non_multiple_reductions() {
    let be = backend();
    let (batch, channels, spatial) = (3usize, 4usize, 17usize);
    let mut rng = Rng(0x4d4f_455f_504f_4f4c);
    let input = rng.vec(batch * channels * spatial, -1.0, 1.0);
    let input_t = be.upload_f32(&[batch, channels, spatial], &input).unwrap();
    let pooled = be
        .moe_router_pool_f32(&input_t, batch, channels, spatial, cfg())
        .unwrap();
    let mut pooled_ref = Vec::with_capacity(batch * channels);
    for b in 0..batch {
        for c in 0..channels {
            let base = (b * channels + c) * spatial;
            pooled_ref.push(
                input[base..base + spatial]
                    .iter()
                    .map(|&x| x as f64)
                    .sum::<f64>()
                    / spatial as f64,
            );
        }
    }
    assert_close(
        "moe router pool",
        &be.download_f32(&pooled).unwrap(),
        &pooled_ref,
    );
    let pooled_gradient = rng.vec(batch * channels, -0.6, 0.6);
    let pooled_gradient_t = be.upload_f32(&[batch, channels], &pooled_gradient).unwrap();
    let input_shape = [batch, channels, 1, spatial];
    let input_gradient = be
        .moe_router_pool_backward_f32(&pooled_gradient_t, &input_shape, cfg())
        .unwrap();
    let input_gradient_ref = (0..batch * channels * spatial)
        .map(|i| pooled_gradient[i / spatial] as f64 / spatial as f64)
        .collect::<Vec<_>>();
    assert_close(
        "moe router pool backward",
        &be.download_f32(&input_gradient).unwrap(),
        &input_gradient_ref,
    );

    let per_image = 777usize;
    let gradient = rng.vec(batch * per_image, -0.8, 0.8);
    let output = rng.vec(batch * per_image, -0.9, 0.9);
    let gradient_t = be.upload_f32(&[batch, per_image], &gradient).unwrap();
    let output_t = be.upload_f32(&[batch, per_image], &output).unwrap();
    let dots = be
        .moe_gate_dot_f32(&gradient_t, &output_t, batch, per_image, cfg())
        .unwrap();
    let expected = (0..batch)
        .map(|b| {
            let start = b * per_image;
            gradient[start..start + per_image]
                .iter()
                .zip(&output[start..start + per_image])
                .map(|(&dy, &y)| dy as f64 * y as f64)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    assert_close("moe gate dot", &be.download_f32(&dots).unwrap(), &expected);
}

#[test]
fn moe_router_backward_and_gate_columns_match_reference() {
    let be = backend();
    let (batch, experts, per_image) = (4usize, 3usize, 73usize);
    let gates = vec![0.4, 0.6, 0.0, 0.0, 0.3, 0.7, 0.25, 0.0, 0.75, 0.6, 0.4, 0.0];
    let mut rng = Rng(0x4d4f_455f_4257_445f);
    let gradient = rng.vec(batch * per_image, -0.8, 0.8);
    let gate_t = be.upload_f32(&[batch, experts], &gates).unwrap();
    let gradient_t = be.upload_f32(&[batch, per_image], &gradient).unwrap();
    let expert_outputs = (0..experts)
        .map(|_| {
            let values = rng.vec(batch * per_image, -0.9, 0.9);
            be.upload_f32(&[batch, per_image], &values).unwrap()
        })
        .collect::<Vec<_>>();
    let expert_values = expert_outputs
        .iter()
        .map(|output| be.download_f32(output).unwrap())
        .collect::<Vec<_>>();
    for expert in 0..experts {
        let column = be
            .moe_gate_column_f32(&gate_t, batch, experts, expert, cfg())
            .unwrap();
        let expected = (0..batch)
            .map(|b| gates[b * experts + expert] as f64)
            .collect::<Vec<_>>();
        assert_close(
            "moe gate column",
            &be.download_f32(&column).unwrap(),
            &expected,
        );
    }

    let load_balance = 0.03f64;
    let temperature = 1.7f64;
    let dlogit = be
        .moe_router_backward_f32(
            &gate_t,
            &gradient_t,
            &expert_outputs,
            batch,
            experts,
            per_image,
            load_balance as f32,
            temperature as f32,
            cfg(),
        )
        .unwrap();
    let mut raw = vec![0.0f64; batch * experts];
    for b in 0..batch {
        for e in 0..experts {
            let grad_base = b * per_image;
            raw[b * experts + e] = gradient[grad_base..grad_base + per_image]
                .iter()
                .zip(&expert_values[e][grad_base..grad_base + per_image])
                .map(|(&dy, &y)| dy as f64 * y as f64)
                .sum();
        }
    }
    for e in 0..experts {
        let mean_gate = (0..batch)
            .map(|b| gates[b * experts + e] as f64)
            .sum::<f64>()
            / batch as f64;
        let balance_gradient = load_balance * 2.0 * experts as f64 * mean_gate / batch as f64;
        for b in 0..batch {
            raw[b * experts + e] += balance_gradient;
        }
    }
    let mut expected = vec![0.0f64; batch * experts];
    for b in 0..batch {
        let base = b * experts;
        let weighted = (0..experts)
            .map(|e| gates[base + e] as f64 * raw[base + e])
            .sum::<f64>();
        for e in 0..experts {
            expected[base + e] = gates[base + e] as f64 * (raw[base + e] - weighted) / temperature;
        }
    }
    assert_close(
        "moe router backward",
        &be.download_f32(&dlogit).unwrap(),
        &expected,
    );
}
