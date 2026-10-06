//! Real-cfg shape coverage for the cuda-oxide backend.
//!
//! Every geometry comes from `tests/data/*.csv`, produced by
//! `tools/cfg_conv_shapes.py` from the ~100 cfgs under `cfg/` plus the
//! LegoGears_v2 cfg (`legogears_v2_convs.csv`, regenerate with
//! `tools/cfg_conv_shapes.py --cfg <cfg> --cfg-out tests/data/legogears_v2_convs.csv`).
//!
//! Run with `--nocapture` to see the summaries (geometries run / skipped /
//! failed, max errors, unsupported list, LegoGears per-layer errors).
//!
//! # Conv selection and size rule
//! * Selected: `dilation == 1`, `stride_x == stride_y`, `padding == size/2`,
//!   `groups | in_c`, `groups | filters` (983 of the 990 unique geometries).
//!   Batch 1 (batch 2 for every 50th geometry).
//! * CPU-reference cost cap: a geometry whose MACs exceed `SPATIAL_TRIGGER`
//!   (20M) keeps channels / kernel / stride / padding / groups but its
//!   `in_h`, `in_w` shrink to `32 + (h % 2)` (parity of the real size kept so
//!   odd-size stride-2 behaviour is exercised).  If it still exceeds `MAC_CAP`
//!   (40M), channels are sampled deterministically: channels-per-group and
//!   filters-per-group are first capped at 64 (the leading channels of the
//!   group are kept; `groups` itself is never changed), then the larger is
//!   halved until <= 8, then the spatial size halves down to >= 16.
//!   The output is always >= 1 (spatial size never drops below the kernel
//!   extent minus the padding).
//! * LegoGears_v2 conv layers always run at REAL size (224x160 network input).
//! * fp32 modes (`Direct`, `Gemm` with `GemmMode::Fp32`) must match an f64
//!   reference within 1e-3 abs + 1e-3 rel.  TF32 (`Gemm` + `GemmMode::Tf32`)
//!   runs on a deterministic subset (every 8th geometry + all mandatory
//!   picks + LegoGears) with the scaled tolerance of `oxide_gemm_mma.rs`
//!   (`|err| <= 2e-3 * reference-on-|x|,|w|,|dy|`).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Mutex;

use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::{ConvMode, GemmMode, OxideBackend};

// ---------------------------------------------------------------- helpers

const TOL_ABS: f64 = 1e-3;
const TOL_REL: f64 = 1e-3;
const TOL_TF32: f64 = 2e-3;
const SPATIAL_TRIGGER: usize = 20_000_000;
const MAC_CAP: usize = 40_000_000;

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

    /// Pairwise distinct values in a shuffled order (no max-pool ties).
    fn distinct(&mut self, n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - n as f32 * 0.005).collect();
        for i in (1..n).rev() {
            let j = (self.next(0.0, 1.0) * (i + 1) as f32) as usize % (i + 1);
            v.swap(i, j);
        }
        v
    }
}

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&e| e as f64).collect()
}

fn abs64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&e| (e as f64).abs()).collect()
}

/// Tiny std-only CSV reader (no quoting needed: the files never contain
/// commas inside fields).
struct Table {
    name: String,
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn load(name: &str) -> Table {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/data");
        path.push(name);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<String> = lines
            .next()
            .unwrap_or_else(|| panic!("{name}: empty"))
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let rows: Vec<Vec<String>> = lines
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>())
            .inspect(|r| assert_eq!(r.len(), header.len(), "{name}: ragged row {r:?}"))
            .collect();
        Table { name: name.to_string(), header, rows }
    }

    fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("{}: no column {name}", self.name))
    }

    fn s<'a>(&'a self, row: usize, col: &str) -> &'a str {
        &self.rows[row][self.col(col)]
    }

    fn u(&self, row: usize, col: &str) -> usize {
        self.s(row, col)
            .parse()
            .unwrap_or_else(|_| panic!("{}: row {row} col {col} not an integer", self.name))
    }
}

/// Work-shared parallel loop over equally sized output planes.
fn par_planes<F>(out: &mut [f64], plane: usize, f: F)
where
    F: Fn(usize, &mut [f64]) + Sync,
{
    if plane == 0 || out.is_empty() {
        return;
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let it = Mutex::new(out.chunks_mut(plane).enumerate());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let next = it.lock().unwrap().next();
                    match next {
                        Some((i, chunk)) => f(i, chunk),
                        None => break,
                    }
                }
            });
        }
    });
}

// ------------------------------------------------------------ conv cases

#[derive(Clone, Debug)]
struct Case {
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
    fn extent(&self) -> usize {
        self.dil * (self.k - 1) + 1
    }
    fn oh(&self) -> usize {
        (self.h + 2 * self.pad - self.extent()) / self.sy + 1
    }
    fn ow(&self) -> usize {
        (self.w + 2 * self.pad - self.extent()) / self.sx + 1
    }
    fn cpg(&self) -> usize {
        self.c / self.groups
    }
    fn fpg(&self) -> usize {
        self.f / self.groups
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
    fn macs(&self) -> usize {
        self.b * self.f * self.cpg() * self.k * self.k * self.oh() * self.ow()
    }
    /// Smallest spatial size with output >= 1.
    fn min_hw(&self) -> usize {
        self.extent().saturating_sub(2 * self.pad).max(1)
    }
    fn desc(&self) -> String {
        format!(
            "b{} {}x{}x{} -> f{} k{} s{}x{} d{} p{} g{}",
            self.b, self.c, self.h, self.w, self.f, self.k, self.sx, self.sy, self.dil, self.pad,
            self.groups
        )
    }
}

/// One unique geometry row of `conv_shapes.csv` (or a LegoGears layer).
#[derive(Clone, Debug)]
struct Geom {
    row: usize,
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
    out_h: usize,
    out_w: usize,
    origin: String,
}

impl Geom {
    fn case(&self, b: usize) -> Case {
        Case {
            b,
            c: self.c,
            h: self.h,
            w: self.w,
            f: self.f,
            k: self.k,
            sx: self.sx,
            sy: self.sy,
            dil: self.dil,
            pad: self.pad,
            groups: self.groups,
        }
    }
    fn in_rule(&self) -> bool {
        self.dil == 1
            && self.sx == self.sy
            && self.pad == self.k / 2
            && self.c % self.groups == 0
            && self.f % self.groups == 0
    }
    fn depthwise(&self) -> bool {
        self.groups > 1 && self.groups == self.c && self.c == self.f
    }
}

fn load_geoms() -> Vec<Geom> {
    let t = Table::load("conv_shapes.csv");
    (0..t.rows.len())
        .map(|r| Geom {
            row: r + 2, // 1-based CSV line number including the header
            c: t.u(r, "in_c"),
            h: t.u(r, "in_h"),
            w: t.u(r, "in_w"),
            f: t.u(r, "filters"),
            k: t.u(r, "size"),
            sx: t.u(r, "stride_x"),
            sy: t.u(r, "stride_y"),
            dil: t.u(r, "dilation"),
            pad: t.u(r, "padding"),
            groups: t.u(r, "groups"),
            out_h: t.u(r, "out_h"),
            out_w: t.u(r, "out_w"),
            origin: t.s(r, "example_cfg").to_string(),
        })
        .collect()
}

fn load_lego() -> Vec<(usize, String, String, Geom)> {
    let t = Table::load("legogears_v2_convs.csv");
    (0..t.rows.len())
        .map(|r| {
            let layer = t.u(r, "layer");
            (
                layer,
                t.s(r, "activation").to_string(),
                t.s(r, "batch_normalize").to_string(),
                Geom {
                    row: r + 2,
                    c: t.u(r, "in_c"),
                    h: t.u(r, "in_h"),
                    w: t.u(r, "in_w"),
                    f: t.u(r, "filters"),
                    k: t.u(r, "size"),
                    sx: t.u(r, "stride_x"),
                    sy: t.u(r, "stride_y"),
                    dil: t.u(r, "dilation"),
                    pad: t.u(r, "padding"),
                    groups: t.u(r, "groups"),
                    out_h: t.u(r, "out_h"),
                    out_w: t.u(r, "out_w"),
                    origin: format!("LegoGears_v2.cfg#{layer}"),
                },
            )
        })
        .collect()
}

/// Applies the documented size rule; `trigger`/`cap` are MAC limits.
fn shrink(g: &Geom, batch: usize, trigger: usize, cap: usize) -> (Case, bool) {
    let mut c = g.case(batch);
    let orig = c.clone();
    if c.macs() > trigger {
        let min_hw = c.min_hw();
        c.h = c.h.min(32 + c.h % 2).max(min_hw);
        c.w = c.w.min(32 + c.w % 2).max(min_hw);
    }
    loop {
        if c.macs() <= cap {
            break;
        }
        let (mut cpg, mut fpg) = (c.cpg(), c.fpg());
        if cpg > 64 || fpg > 64 {
            cpg = cpg.min(64);
            fpg = fpg.min(64);
        } else if cpg.max(fpg) > 8 {
            if cpg >= fpg {
                cpg = (cpg / 2).max(1);
            } else {
                fpg = (fpg / 2).max(1);
            }
        } else if c.h > 16 || c.w > 16 {
            let min_hw = c.min_hw();
            c.h = (c.h / 2).max(16).max(min_hw);
            c.w = (c.w / 2).max(16).max(min_hw);
            continue;
        } else {
            break;
        }
        c.c = cpg * c.groups;
        c.f = fpg * c.groups;
    }
    let changed = (c.h, c.w, c.c, c.f) != (orig.h, orig.w, orig.c, orig.f);
    (c, changed)
}

// ------------------------------------------------------ conv CPU reference

/// First and one-past-last output column whose tap `ox*sx + kx*dil - pad`
/// lies inside `[0, w)`.
fn ox_range(c: &Case, kx: usize) -> (usize, usize) {
    let off = (kx * c.dil) as isize - c.pad as isize; // ix = ox*sx + off
    let lo = if off >= 0 { 0 } else { ((-off) as usize).div_ceil(c.sx) };
    let top = c.w as isize - 1 - off; // need ox*sx <= top
    let hi = if top < 0 { 0 } else { (top as usize / c.sx + 1).min(c.ow()) };
    (lo.min(hi), hi)
}

/// Input row for output row `oy` and tap row `ky`, if inside the image.
fn in_row(c: &Case, oy: usize, ky: usize) -> Option<usize> {
    let iy = (oy * c.sy + ky * c.dil) as isize - c.pad as isize;
    (iy >= 0 && iy < c.h as isize).then_some(iy as usize)
}

fn col_off(c: &Case, kx: usize) -> isize {
    (kx * c.dil) as isize - c.pad as isize
}

/// Same indexing semantics as darknet-core `convolution_raw_value`: weights
/// `[F, C/groups, k, k]`, NCHW, symmetric padding, per-axis strides.
/// Returns `(y, dx, dw)`; threaded over output planes.
fn conv_refs_impl(c: &Case, x: &[f64], w: &[f64], dy: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (oh, ow, cpg, fpg, k) = (c.oh(), c.ow(), c.cpg(), c.fpg(), c.k);
    let (h, wd) = (c.h, c.w);

    let mut y = vec![0.0; c.y_len()];
    par_planes(&mut y, oh * ow, |i, plane| {
        let (b, f) = (i / c.f, i % c.f);
        let g = f / fpg;
        for ic in 0..cpg {
            let xp = &x[(b * c.c + g * cpg + ic) * h * wd..][..h * wd];
            for ky in 0..k {
                for kx in 0..k {
                    let wv = w[((f * cpg + ic) * k + ky) * k + kx];
                    let (lo, hi) = ox_range(c, kx);
                    let off = col_off(c, kx);
                    for oy in 0..oh {
                        let Some(iy) = in_row(c, oy, ky) else { continue };
                        let xr = &xp[iy * wd..][..wd];
                        let yr = &mut plane[oy * ow..][..ow];
                        for ox in lo..hi {
                            yr[ox] += wv * xr[((ox * c.sx) as isize + off) as usize];
                        }
                    }
                }
            }
        }
    });

    // dgrad: plane = (b, input channel)
    let mut dx = vec![0.0; c.x_len()];
    par_planes(&mut dx, h * wd, |i, plane| {
        let (b, ch) = (i / c.c, i % c.c);
        let (g, ic) = (ch / cpg, ch % cpg);
        for f in g * fpg..(g + 1) * fpg {
            let dyp = &dy[(b * c.f + f) * oh * ow..][..oh * ow];
            for ky in 0..k {
                for kx in 0..k {
                    let wv = w[((f * cpg + ic) * k + ky) * k + kx];
                    let (lo, hi) = ox_range(c, kx);
                    let off = col_off(c, kx);
                    for oy in 0..oh {
                        let Some(iy) = in_row(c, oy, ky) else { continue };
                        let dr = &dyp[oy * ow..][..ow];
                        let xr = &mut plane[iy * wd..][..wd];
                        for ox in lo..hi {
                            xr[((ox * c.sx) as isize + off) as usize] += wv * dr[ox];
                        }
                    }
                }
            }
        }
    });

    // wgrad: plane = filter
    let mut dw = vec![0.0; c.w_len()];
    par_planes(&mut dw, cpg * k * k, |f, plane| {
        let g = f / fpg;
        for b in 0..c.b {
            let dyp = &dy[(b * c.f + f) * oh * ow..][..oh * ow];
            for ic in 0..cpg {
                let xp = &x[(b * c.c + g * cpg + ic) * h * wd..][..h * wd];
                for ky in 0..k {
                    for kx in 0..k {
                        let (lo, hi) = ox_range(c, kx);
                        let off = col_off(c, kx);
                        let mut acc = 0.0;
                        for oy in 0..oh {
                            let Some(iy) = in_row(c, oy, ky) else { continue };
                            let xr = &xp[iy * wd..][..wd];
                            let dr = &dyp[oy * ow..][..ow];
                            for ox in lo..hi {
                                acc += xr[((ox * c.sx) as isize + off) as usize] * dr[ox];
                            }
                        }
                        plane[(ic * k + ky) * k + kx] += acc;
                    }
                }
            }
        }
    });
    (y, dx, dw)
}

struct Gpu {
    y: Vec<f32>,
    dx: Vec<f32>,
    dw: Vec<f32>,
}

fn run_gpu(be: &OxideBackend, c: &Case, x: &[f32], w: &[f32], dy: &[f32]) -> Result<Gpu, String> {
    let e = |op: &str, err: darknet_cuda_oxide::OxideError| format!("{op}: {err}");
    let xt = be.upload_f32(&[c.b, c.c, c.h, c.w], x).map_err(|er| e("upload x", er))?;
    let wt = be.upload_f32(&[c.f, c.cpg(), c.k, c.k], w).map_err(|er| e("upload w", er))?;
    let dyt = be.upload_f32(&[c.b, c.f, c.oh(), c.ow()], dy).map_err(|er| e("upload dy", er))?;
    let y = be
        .conv2d_f32(
            &xt, &wt, &[c.b, c.f, c.oh(), c.ow()], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil,
            c.pad, c.groups, cfg(),
        )
        .map_err(|er| e("fwd", er))?;
    let dx = be
        .conv2d_backward_input_f32(
            &dyt, &wt, &[c.b, c.c, c.h, c.w], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad,
            c.groups, cfg(),
        )
        .map_err(|er| e("dgrad", er))?;
    let dw = be
        .conv2d_backward_weights_f32(
            &xt, &dyt, &[c.f, c.cpg(), c.k, c.k], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil,
            c.pad, c.groups, cfg(),
        )
        .map_err(|er| e("wgrad", er))?;
    Ok(Gpu {
        y: be.download_f32(&y).map_err(|er| e("download y", er))?,
        dx: be.download_f32(&dx).map_err(|er| e("download dx", er))?,
        dw: be.download_f32(&dw).map_err(|er| e("download dw", er))?,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Flavor {
    Fp32,
    Tf32,
}

/// `(max error metric, number of bad elements, first bad message)`.
/// Metric: absolute error (fp32) or error / scale (tf32).
fn compare(name: &str, got: &[f32], want: &[f64], scale: Option<&[f64]>) -> (f64, usize, Option<String>) {
    if got.len() != want.len() {
        return (f64::INFINITY, got.len().max(1), Some(format!("{name}: length {} != {}", got.len(), want.len())));
    }
    let (mut worst, mut bad, mut first) = (0.0f64, 0usize, None);
    for i in 0..got.len() {
        let err = (got[i] as f64 - want[i]).abs();
        let (metric, ok) = match scale {
            None => (err, err <= TOL_ABS + TOL_REL * want[i].abs()),
            Some(s) => {
                let s = s[i].max(1e-30);
                (err / s, err <= TOL_TF32 * s)
            }
        };
        if !ok {
            bad += 1;
            if first.is_none() {
                first = Some(format!("{name}[{i}] got {} want {} err {err:.3e}", got[i], want[i]));
            }
        }
        if metric > worst || metric.is_nan() {
            worst = if metric.is_nan() { f64::INFINITY } else { metric };
        }
    }
    (worst, bad, first)
}

struct Outcome {
    max: [f64; 3],
    failures: Vec<String>,
}

fn run_conv_case(be: &OxideBackend, c: &Case, seed: u64, flavor: Flavor) -> Outcome {
    let mut rng = Rng(seed);
    let (x, w, dy) = (
        rng.vec(c.x_len(), -1.0, 1.0),
        rng.vec(c.w_len(), -1.0, 1.0),
        rng.vec(c.y_len(), -1.0, 1.0),
    );
    let mut out = Outcome { max: [0.0; 3], failures: vec![] };
    let got = match run_gpu(be, c, &x, &w, &dy) {
        Ok(g) => g,
        Err(msg) => {
            out.failures.push(format!("backend error: {msg}"));
            out.max = [f64::INFINITY; 3];
            return out;
        }
    };
    let (y, dx, dw) = conv_refs_impl(c, &f64s(&x), &f64s(&w), &f64s(&dy));
    let scales = (flavor == Flavor::Tf32).then(|| conv_refs_impl(c, &abs64(&x), &abs64(&w), &abs64(&dy)));
    let checks: [(&str, &[f32], &[f64], Option<&[f64]>); 3] = [
        ("fwd", &got.y, &y, scales.as_ref().map(|s| &s.0[..])),
        ("dgrad", &got.dx, &dx, scales.as_ref().map(|s| &s.1[..])),
        ("wgrad", &got.dw, &dw, scales.as_ref().map(|s| &s.2[..])),
    ];
    for (i, (name, g, r, s)) in checks.iter().enumerate() {
        let (worst, bad, first) = compare(name, g, r, *s);
        out.max[i] = worst;
        if bad > 0 {
            out.failures.push(format!("{name}: {bad} bad elems, e.g. {}", first.unwrap_or_default()));
        }
    }
    out
}

// ------------------------------------------------------- mandatory picks

/// Indices (into `sel`) that every mode - including the TF32 subset - must
/// cover, each with the reason.  See the task list: every distinct (k, s),
/// depthwise, groups 2 / 32 / odd filters-per-group, stem in_c 1 or 3, the
/// large-filters 1x1, plus the extremes of each of those classes.
fn mandatory(sel: &[Geom]) -> Vec<(usize, String)> {
    let mut picks: Vec<(usize, String)> = vec![];
    let macs = |g: &Geom| g.case(1).macs();
    let by = |pred: &dyn Fn(&Geom) -> bool| -> Vec<usize> {
        (0..sel.len()).filter(|&i| pred(&sel[i])).collect()
    };
    let (mut ks, mut groups_seen) = (BTreeSet::new(), BTreeSet::new());
    for g in sel {
        ks.insert((g.k, g.sx));
        groups_seen.insert(g.groups);
    }
    let mut add_min_max = |label: String, idx: Vec<usize>| {
        if let (Some(&lo), Some(&hi)) = (
            idx.iter().min_by_key(|&&i| macs(&sel[i])),
            idx.iter().max_by_key(|&&i| macs(&sel[i])),
        ) {
            picks.push((lo, format!("{label} (smallest)")));
            if hi != lo {
                picks.push((hi, format!("{label} (largest)")));
            }
        }
    };
    for (k, s) in ks {
        let mut idx = by(&|g| g.k == k && g.sx == s && g.groups == 1);
        if idx.is_empty() {
            idx = by(&|g| g.k == k && g.sx == s); // e.g. k5 s2 exists only grouped
        }
        add_min_max(format!("k{k} s{s}"), idx);
    }
    add_min_max("depthwise".into(), by(&|g| g.depthwise()));
    for gv in groups_seen.into_iter().filter(|&g| g > 1) {
        add_min_max(format!("groups={gv}"), by(&|g| g.groups == gv));
    }
    add_min_max("odd filters-per-group".into(), by(&|g| g.groups > 1 && (g.f / g.groups) % 2 == 1 && g.f / g.groups > 1));
    add_min_max("grouped, filters-per-group 1".into(), by(&|g| g.groups > 1 && g.f / g.groups == 1 && !g.depthwise()));
    for ic in [1usize, 3] {
        add_min_max(format!("stem in_c={ic}"), by(&|g| g.c == ic));
    }
    // large-filters 1x1 (e.g. YOLO9000 28269 filters, classifier heads)
    let mut big: Vec<usize> = by(&|g| g.k == 1 && g.groups == 1);
    big.sort_by_key(|&i| std::cmp::Reverse(sel[i].f));
    let mut extra: Vec<(usize, String)> = vec![];
    for &i in big.iter().take(3) {
        extra.push((i, format!("large-filters 1x1 (f={})", sel[i].f)));
    }
    let mut wide: Vec<usize> = by(&|g| g.k >= 3 && g.groups == 1);
    wide.sort_by_key(|&i| std::cmp::Reverse(sel[i].c.max(sel[i].f)));
    for &i in wide.iter().take(2) {
        extra.push((i, format!("widest 3x3+ (c={} f={})", sel[i].c, sel[i].f)));
    }
    // largest real spatial size and an odd-size stride-2 case
    add_min_max("stride>1 odd input".into(), by(&|g| g.sx > 1 && (g.h % 2 == 1 || g.w % 2 == 1)));
    picks.extend(extra);
    picks
}

fn selected() -> (Vec<Geom>, Vec<Geom>) {
    let (mut sel, mut other) = (vec![], vec![]);
    for g in load_geoms() {
        if g.in_rule() { sel.push(g) } else { other.push(g) }
    }
    (sel, other)
}

fn is_dilated(g: &Geom) -> bool {
    g.dil > 1
}

/// Geometries outside the rule that are still plain (dil == 1) convs: they
/// are run through the ordinary parity check (e.g. AlexNet k11 s4 p0).
fn plain_extras(other: &[Geom]) -> Vec<&Geom> {
    other
        .iter()
        .filter(|g| !is_dilated(g) && g.c % g.groups == 0 && g.f % g.groups == 0)
        .collect()
}

// -------------------------------------------------------- conv mode tests

struct ModeReport {
    run: usize,
    skipped: Vec<String>,
    failed: Vec<String>,
    worst: [(f64, String); 3],
}

impl ModeReport {
    fn new() -> Self {
        Self {
            run: 0,
            skipped: vec![],
            failed: vec![],
            worst: [(0.0, String::new()), (0.0, String::new()), (0.0, String::new())],
        }
    }

    fn record(&mut self, label: &str, o: &Outcome) {
        self.run += 1;
        for i in 0..3 {
            if o.max[i] >= self.worst[i].0 {
                self.worst[i] = (o.max[i], label.to_string());
            }
        }
        if !o.failures.is_empty() {
            self.failed.push(format!("{label}: {}", o.failures.join("; ")));
        }
    }

    fn finish(&self, title: &str, metric: &str) {
        println!("==== {title} ====");
        println!("geometries run: {}, skipped: {}, failed: {}", self.run, self.skipped.len(), self.failed.len());
        for s in &self.skipped {
            println!("  skipped: {s}");
        }
        for (i, n) in ["fwd", "dgrad", "wgrad"].iter().enumerate() {
            println!("  max {metric} {n}: {:.3e}  at {}", self.worst[i].0, self.worst[i].1);
        }
        for f in &self.failed {
            println!("  FAIL {f}");
        }
        assert!(self.failed.is_empty(), "{title}: {} geometry failures (listed above)\n{}", self.failed.len(), self.failed.join("\n"));
    }
}

fn label(g: &Geom, c: &Case, shrunk: bool) -> String {
    format!(
        "csv line {} [{}] real {}x{}x{}->f{} k{} s{} p{} g{} | ran {}{}",
        g.row, g.origin, g.c, g.h, g.w, g.f, g.k, g.sx, g.pad, g.groups, c.desc(),
        if shrunk { " (shrunk)" } else { "" }
    )
}

fn run_lego(be: &OxideBackend, flavor: Flavor, rep: &mut ModeReport) {
    let metric = if flavor == Flavor::Fp32 { "abs err" } else { "err/scale" };
    for (layer, act, bn, g) in load_lego() {
        assert!(g.in_rule(), "LegoGears layer {layer} must be inside the rule");
        assert_eq!((g.case(1).oh(), g.case(1).ow()), (g.out_h, g.out_w), "LegoGears layer {layer} out dims");
        let c = g.case(1); // REAL size, no shrink
        let o = run_conv_case(be, &c, 7000 + layer as u64, flavor);
        println!(
            "  LEGO L{layer:<2} {} ({act}, bn={bn}) {metric}: fwd {:.2e} dgrad {:.2e} wgrad {:.2e}{}",
            c.desc(), o.max[0], o.max[1], o.max[2],
            if o.failures.is_empty() { "" } else { "  <-- FAIL" }
        );
        rep.record(&format!("LegoGears_v2 layer {layer} {}", c.desc()), &o);
    }
}

fn conv_mode_test(title: &str, be: OxideBackend, flavor: Flavor) {
    let t0 = std::time::Instant::now();
    let (sel, other) = selected();
    let picks = mandatory(&sel);
    let pick_set: BTreeSet<usize> = picks.iter().map(|p| p.0).collect();
    let mut rep = ModeReport::new();
    println!("{title}: {} selected geometries, {} outside the rule", sel.len(), other.len());

    run_lego(&be, flavor, &mut rep);

    let indices: Vec<usize> = match flavor {
        Flavor::Fp32 => (0..sel.len()).collect(),
        Flavor::Tf32 => (0..sel.len()).filter(|i| i % 8 == 0 || pick_set.contains(i)).collect(),
    };
    for i in indices {
        let g = &sel[i];
        let batch = if i % 50 == 0 { 2 } else { 1 };
        let (c, shrunk) = shrink(g, batch, SPATIAL_TRIGGER, MAC_CAP);
        if c.oh() == 0 || c.ow() == 0 {
            rep.skipped.push(format!("csv line {}: output < 1 after shrink", g.row));
            continue;
        }
        let o = run_conv_case(&be, &c, 1000 + g.row as u64, flavor);
        rep.record(&label(g, &c, shrunk), &o);
    }
    // Plain (dilation 1) geometries outside the selection rule.
    for g in plain_extras(&other) {
        let (c, shrunk) = shrink(g, 1, SPATIAL_TRIGGER, MAC_CAP);
        let o = run_conv_case(&be, &c, 1000 + g.row as u64, flavor);
        rep.record(&format!("non-rule {}", label(g, &c, shrunk)), &o);
    }
    // Synthetic grouped geometries (NOT in any cfg): odd filters-per-group etc.
    for (n, g) in synthetic_grouped().iter().enumerate() {
        let c = g.case(if n == 0 { 2 } else { 1 });
        let o = run_conv_case(&be, &c, 4000 + n as u64, flavor);
        rep.record(&format!("synthetic grouped {}", c.desc()), &o);
    }
    let mut covered = BTreeSet::new();
    for (i, why) in &picks {
        covered.insert(why.clone());
        if flavor == Flavor::Tf32 {
            assert!(pick_set.contains(i));
        }
    }
    println!("mandatory picks ({} geometries, {} classes) all executed:", pick_set.len(), covered.len());
    for (i, why) in &picks {
        println!("    {why}: csv line {}", sel[*i].row);
    }
    println!("elapsed: {:.1}s", t0.elapsed().as_secs_f64());
    let metric = if flavor == Flavor::Fp32 { "abs err" } else { "err/scale" };
    rep.finish(title, metric);
}

#[test]
fn cfg_conv_direct() {
    conv_mode_test(
        "cfg conv: Direct (fp32)",
        OxideBackend::new().unwrap().with_conv_mode(ConvMode::Direct).with_gemm_mode(GemmMode::Fp32),
        Flavor::Fp32,
    );
}

#[test]
fn cfg_conv_gemm() {
    conv_mode_test(
        "cfg conv: im2col+GEMM (fp32)",
        OxideBackend::new().unwrap().with_conv_mode(ConvMode::Gemm).with_gemm_mode(GemmMode::Fp32),
        Flavor::Fp32,
    );
}

#[test]
fn cfg_conv_gemm_tf32() {
    conv_mode_test(
        "cfg conv: im2col+GEMM (tf32 subset)",
        OxideBackend::new().unwrap().with_conv_mode(ConvMode::Gemm).with_gemm_mode(GemmMode::Tf32),
        Flavor::Tf32,
    );
}

/// Grouped geometries absent from the cfgs: odd filters-per-group, odd
/// channels-per-group, grouped stride 2, filters != channels.
fn synthetic_grouped() -> Vec<Geom> {
    let mk = |c, h, w, f, k, s, g| Geom {
        row: 0, c, h, w, f, k, sx: s, sy: s, dil: 1, pad: k / 2, groups: g, out_h: 0, out_w: 0,
        origin: "synthetic".into(),
    };
    vec![mk(9, 17, 17, 15, 3, 1, 3), mk(10, 19, 19, 15, 3, 2, 5), mk(12, 16, 16, 28, 1, 1, 4), mk(21, 15, 15, 21, 5, 1, 7)]
}

/// Sanity of the selection rule / picks themselves (no GPU).
/// fp16 (P6) over the real cfg geometries: every 10th in-rule geometry of
/// `conv_shapes.csv` (shrunk like the fp32 runs) through the fused f16 conv
/// (f16 storage, f32 accumulate, linear epilogue, zero bias) against the f64
/// reference computed from the f16-ROUNDED operands.  Tolerance model of
/// `tests/oxide_half.rs`: `5.2e-4|ref| + 2e-5 * sum|w x| + 1e-7`.
#[test]
fn cfg_conv_fp16_sampled() {
    use darknet_cuda_oxide::half::{f16_bits_to_f32, f32_to_f16_bits};
    let round16 = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16_bits_to_f32(f32_to_f16_bits(x))).collect() };
    let (sel, _) = selected();
    let be = OxideBackend::new().unwrap();
    let (mut run, mut worst, mut fails) = (0usize, 0.0f64, vec![]);
    for (n, g) in sel.iter().enumerate().filter(|(n, _)| n % 10 == 0) {
        let (c, shrunk) = shrink(g, 1, 10_000_000, 16_000_000);
        let (oh, ow, cpg) = (c.oh(), c.ow(), c.cpg());
        let kk = cpg * c.k * c.k;
        let mut rng = Rng(5150 + n as u64);
        let amp = 1.0 / (kk as f32).sqrt();
        let x = round16(&rng.vec(c.x_len(), -1.0, 1.0));
        let w = round16(&rng.vec(c.w_len(), -amp, amp));
        let (xd, wd) = (f64s(&x), f64s(&w));
        let (want, _, _) = conv_refs_impl(&c, &xd, &wd, &vec![0.0; c.y_len()]);
        let (scale, _, _) = conv_refs_impl(&c, &abs64(&x), &abs64(&w), &vec![0.0; c.y_len()]);
        let xt = be.upload_f16(&[c.b, c.c, c.h, c.w], &x).unwrap();
        let wt = be.conv_weights_f16(&be.upload_f32(&[w.len()], &w).unwrap(), c.f, kk).unwrap();
        let bt = be.upload_f32(&[c.f], &vec![0.0; c.f]).unwrap();
        let res = be.conv2d_bias_act_f16(
            &xt, &wt, &bt, &[c.b, c.f, oh, ow], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups,
            F32ActivationKind::Linear, cfg(),
        );
        let y = match res.and_then(|y| be.download_f16(&y)) {
            Ok(y) => y,
            Err(e) => {
                fails.push(format!("{}: {e}", label(g, &c, shrunk)));
                continue;
            }
        };
        let mut worst_here = 0.0f64;
        for i in 0..y.len() {
            let err = (y[i] as f64 - want[i]).abs();
            let tol = 5.2e-4 * want[i].abs() + 2e-5 * scale[i] + 1e-7;
            worst_here = worst_here.max(err / tol);
            if !(err <= tol) {
                fails.push(format!("{} [{i}] got {} want {} err {err:.3e} tol {tol:.3e}", label(g, &c, shrunk), y[i], want[i]));
                break;
            }
        }
        worst = worst.max(worst_here);
        run += 1;
    }
    println!("==== cfg conv fp16 (sampled) ==== geometries run {run}, worst err/tol {worst:.3}, failed {}", fails.len());
    assert!(run >= 90, "expected ~98 sampled geometries, ran {run}");
    assert!(fails.is_empty(), "{}", fails[..fails.len().min(20)].join("\n"));
}

#[test]
fn cfg_conv_selection_rule_and_picks() {
    let (sel, other) = selected();
    assert_eq!(sel.len() + other.len(), 990, "conv_shapes.csv geometry count");
    assert_eq!(sel.len(), 983, "rule-selected geometries");
    let picks = mandatory(&sel);
    let has = |needle: &str| picks.iter().any(|(_, w)| w.contains(needle));
    for needle in ["k1 s1", "k3 s1", "k3 s2", "k5 s1", "k5 s2", "k7 s2", "depthwise", "groups=2", "groups=32",
        "stem in_c=1", "stem in_c=3", "large-filters 1x1"] {
        assert!(has(needle), "mandatory class missing: {needle}");
    }
    // The real cfgs contain NO grouped conv with odd filters-per-group > 1
    // (non-depthwise groups are 2 or 32 with power-of-two filters-per-group);
    // odd fpg is covered by the synthetic geometries in `synthetic_grouped()`.
    assert!(sel.iter().all(|g| g.groups == 1 || g.f / g.groups == 1 || (g.f / g.groups) % 2 == 0));
    // every mandatory pick is reachable through the shrink rule with output >= 1
    for (i, _) in &picks {
        let (c, _) = shrink(&sel[*i], 1, SPATIAL_TRIGGER, MAC_CAP);
        assert!(c.oh() >= 1 && c.ow() >= 1);
    }
    // LegoGears_v2: all layers in rule at real size
    let lego = load_lego();
    assert_eq!(lego.len(), 21);
    assert!(lego.iter().all(|(_, _, _, g)| g.in_rule() && g.h <= 160 && g.w <= 224));
    assert_eq!((lego[0].3.h, lego[0].3.w), (160, 224));
}

// ---------------------------------------- dilation and non-rule geometries

/// `dilation > 1` (6 resnet152_trident geometries).
///
/// Findings (asserted below):
/// * C++ `convolutional_out_width` and darknet-core ignore dilation:
///   `out = (in + 2*pad - size)/stride + 1`.  C++ cuDNN uses
///   `pad*dilation` symmetric padding, so for pad = size/2 its output equals
///   the core formula; the darknet-core CPU path instead uses `pad` (not
///   `pad*dilation`) with those output dims, which is a shifted window that
///   the C++ GPU path does not compute.
/// * The oxide conv uses the true dilated extent `d*(k-1)+1` in its output
///   formula.  With the darknet-core arguments (pad = csv padding, out dims
///   from the csv) it therefore returns `Err` (shape mismatch) for fwd, dgrad
///   and wgrad - asserted.
/// * Called the way C++ cuDNN does (`pad*dilation`) it matches the dilated f64
///   reference and reproduces the cfg output dims - asserted.
#[test]
fn cfg_conv_dilated_and_non_rule_geometries() {
    let (_, other) = selected();
    let mut unsupported = vec![];
    let mut failures = vec![];
    let mut ran = 0usize;
    for mode in [ConvMode::Direct, ConvMode::Gemm] {
        let be = OxideBackend::new().unwrap().with_conv_mode(mode).with_gemm_mode(GemmMode::Fp32);
        for g in other.iter().filter(|g| is_dilated(g)) {
            // (a) darknet-core convention -> Err on all three entry points.
            let core = g.case(1);
            let xt = be.zeros_f32(&[1, core.c, core.h, core.w]).unwrap();
            let wt = be.zeros_f32(&[core.f, core.cpg(), core.k, core.k]).unwrap();
            let dyt = be.zeros_f32(&[1, core.f, g.out_h, g.out_w]).unwrap();
            let fwd = be.conv2d_f32(&xt, &wt, &[1, core.f, g.out_h, g.out_w], core.c, core.h, core.w, core.f,
                core.k, core.sx, core.sy, core.dil, core.pad, core.groups, cfg());
            let dgr = be.conv2d_backward_input_f32(&dyt, &wt, &[1, core.c, core.h, core.w], core.c, core.h,
                core.w, core.f, core.k, core.sx, core.sy, core.dil, core.pad, core.groups, cfg());
            let wgr = be.conv2d_backward_weights_f32(&xt, &dyt, &[core.f, core.cpg(), core.k, core.k], core.c,
                core.h, core.w, core.f, core.k, core.sx, core.sy, core.dil, core.pad, core.groups, cfg());
            if !(fwd.is_err() && dgr.is_err() && wgr.is_err()) {
                failures.push(format!("{mode:?} line {}: darknet-core-convention call did not return Err on all of fwd/dgrad/wgrad", g.row));
            }
            if mode == ConvMode::Direct {
                unsupported.push(format!(
                    "line {} [{}] {}x{}x{}->f{} k{} s{} dil{} pad{}: Err under darknet-core dims (oxide dilated dims {}x{} != cfg {}x{})",
                    g.row, g.origin, core.c, core.h, core.w, core.f, core.k, core.sx, core.dil, core.pad,
                    core.oh(), core.ow(), g.out_h, g.out_w
                ));
            }
            // (b) C++ cuDNN convention pad*dilation: same dims as the cfg, and parity.
            let mut eff = g.clone();
            eff.pad = g.pad * g.dil;
            if (eff.case(1).oh(), eff.case(1).ow()) != (g.out_h, g.out_w) {
                failures.push(format!("{mode:?} line {}: pad*dil output dims {}x{} != cfg {}x{}", g.row,
                    eff.case(1).oh(), eff.case(1).ow(), g.out_h, g.out_w));
            }
            let (c, shrunk) = shrink(&eff, 1, SPATIAL_TRIGGER, MAC_CAP / 2);
            let o = run_conv_case(&be, &c, 3000 + g.row as u64, Flavor::Fp32);
            ran += 1;
            println!("  dilated {mode:?} line {}: {} fwd {:.2e} dgrad {:.2e} wgrad {:.2e}", g.row, label(&eff, &c, shrunk), o.max[0], o.max[1], o.max[2]);
            if !o.failures.is_empty() {
                failures.push(format!("{mode:?} {}: {}", label(&eff, &c, shrunk), o.failures.join("; ")));
            }
        }
    }
    println!("unsupported/skipped under the darknet-core dilation convention (asserted Err):");
    for u in &unsupported {
        println!("  {u}");
    }
    println!("dilated geometries run via pad*dilation: {ran} (2 modes), failed: {}", failures.len());
    println!("non-rule plain geometries (run in the mode tests): {}", plain_extras(&other).len());
    assert_eq!(unsupported.len(), 6, "expected 6 dilated geometries");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ------------------------------------------------- pool / resample / route

fn assert_close_report(name: &str, got: &[f32], want: &[f64], tol: f64, fails: &mut Vec<String>, worst: &mut f64) {
    let (w, bad, first) = {
        // same metric as the pool tests: 1e-4 abs + 1e-4 rel
        let mut w = 0.0f64;
        let mut bad = 0usize;
        let mut first = None;
        if got.len() != want.len() {
            fails.push(format!("{name}: length {} != {}", got.len(), want.len()));
            return;
        }
        for i in 0..got.len() {
            let err = (got[i] as f64 - want[i]).abs();
            if !(err <= tol + tol * want[i].abs()) {
                bad += 1;
                first.get_or_insert(format!("[{i}] got {} want {}", got[i], want[i]));
            }
            if err > w || err.is_nan() {
                w = if err.is_nan() { f64::INFINITY } else { err };
            }
        }
        (w, bad, first)
    };
    *worst = worst.max(w);
    if bad > 0 {
        fails.push(format!("{name}: {bad} bad elems, e.g. {}", first.unwrap_or_default()));
    }
}

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
    fn work(&self) -> usize {
        self.b * self.c * self.oh() * self.ow() * self.size * self.size
    }
}

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

fn cpu_maxpool_bwd(x: &[f64], dy: &[f64], p: &PoolCase) -> Vec<f64> {
    let (oh, ow, off) = (p.oh(), p.ow(), (p.pad / 2) as isize);
    let mut dx = vec![0.0; x.len()];
    for bc in 0..p.b * p.c {
        for oy in 0..oh {
            for ox in 0..ow {
                let (mut best, mut at) = (f64::NEG_INFINITY, None);
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

#[test]
fn cfg_maxpool_shapes() {
    let t = Table::load("maxpool_shapes.csv");
    let be = OxideBackend::new().unwrap();
    let (mut run, mut fails, mut worst) = (0usize, vec![], 0.0f64);
    let mut combos = BTreeSet::new();
    let mut capped = 0usize;
    for r in 0..t.rows.len() {
        let mut p = PoolCase {
            b: 1,
            c: t.u(r, "in_c"),
            h: t.u(r, "in_h"),
            w: t.u(r, "in_w"),
            size: t.u(r, "size"),
            sx: t.u(r, "stride_x"),
            sy: t.u(r, "stride_y"),
            pad: t.u(r, "padding"),
        };
        let (csv_oh, csv_ow) = (t.u(r, "out_h"), t.u(r, "out_w"));
        if (p.oh(), p.ow()) != (csv_oh, csv_ow) {
            fails.push(format!("line {}: output dims rule gives {}x{}, csv {csv_oh}x{csv_ow}", r + 2, p.oh(), p.ow()));
            continue;
        }
        combos.insert((p.size, p.sx, p.sy, p.pad));
        // Cap the CPU work at 8M window taps: channels first (leading
        // channels kept), then the spatial size (down to >= 64).
        let real = p;
        while p.work() > 8_000_000 && p.c > 1 {
            p.c = p.c.div_ceil(2);
        }
        while p.work() > 8_000_000 && (p.h > 64 || p.w > 64) {
            p.h = (p.h / 2).max(64);
            p.w = (p.w / 2).max(64);
        }
        if (p.c, p.h, p.w) != (real.c, real.h, real.w) {
            capped += 1;
        }
        let mut rng = Rng(5000 + r as u64);
        let x = rng.distinct(p.b * p.c * p.h * p.w);
        let dy = rng.vec(p.b * p.c * p.oh() * p.ow(), -1.0, 1.0);
        let name = format!("maxpool line {} {p:?}", r + 2);
        let xt = be.upload_f32(&p.in_shape(), &x).unwrap();
        let dyt = be.upload_f32(&p.out_shape(), &dy).unwrap();
        let y = be.maxpool_f32(&xt, &p.out_shape(), p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg());
        let dx = be.maxpool_backward_f32(&xt, &dyt, p.c, p.h, p.w, p.size, p.sx, p.sy, p.pad, cfg());
        match (y, dx) {
            (Ok(y), Ok(dx)) => {
                assert_close_report(&format!("{name} fwd"), &be.download_f32(&y).unwrap(), &cpu_maxpool(&f64s(&x), &p), 1e-4, &mut fails, &mut worst);
                assert_close_report(&format!("{name} bwd"), &be.download_f32(&dx).unwrap(), &cpu_maxpool_bwd(&f64s(&x), &f64s(&dy), &p), 1e-4, &mut fails, &mut worst);
                run += 1;
            }
            (a, b) => fails.push(format!("{name}: backend error fwd={:?} bwd={:?}", a.err(), b.err())),
        }
    }
    println!("==== cfg maxpool ==== rows run {run}/{} (size-capped {capped}), distinct (size,sx,sy,pad) combos {}, failed {}, max abs err {worst:.2e}", t.rows.len(), combos.len(), fails.len());
    for f in &fails {
        println!("  FAIL {f}");
    }
    assert_eq!(combos.len(), 7, "all 7 maxpool combos expected");
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

fn up_ref(x: &[f64], b: usize, c: usize, h: usize, w: usize, s: usize) -> Vec<f64> {
    let (oh, ow) = (h * s, w * s);
    let mut out = vec![0.0; b * c * oh * ow];
    for bc in 0..b * c {
        for y in 0..oh {
            for xx in 0..ow {
                out[(bc * oh + y) * ow + xx] = x[(bc * h + y / s) * w + xx / s];
            }
        }
    }
    out
}

fn up_ref_bwd(dy: &[f64], b: usize, c: usize, h: usize, w: usize, s: usize) -> Vec<f64> {
    let (oh, ow) = (h * s, w * s);
    let mut dx = vec![0.0; b * c * h * w];
    for bc in 0..b * c {
        for y in 0..oh {
            for xx in 0..ow {
                dx[(bc * h + y / s) * w + xx / s] += dy[(bc * oh + y) * ow + xx];
            }
        }
    }
    dx
}

#[test]
fn cfg_upsample_shapes() {
    let t = Table::load("upsample_shapes.csv");
    let be = OxideBackend::new().unwrap();
    let (mut run, mut fails, mut worst) = (0usize, vec![], 0.0f64);
    for r in 0..t.rows.len() {
        let (s, mut c, h, w) = (t.u(r, "stride"), t.u(r, "in_c"), t.u(r, "in_h"), t.u(r, "in_w"));
        if (t.u(r, "out_h"), t.u(r, "out_w")) != (h * s, w * s) {
            fails.push(format!("line {}: csv out dims != h*stride", r + 2));
            continue;
        }
        while c * h * w * s * s > 4_000_000 && c > 1 {
            c = c.div_ceil(2);
        }
        let b = 2usize;
        let (oh, ow) = (h * s, w * s);
        let mut rng = Rng(6000 + r as u64);
        let x = rng.vec(b * c * h * w, -1.0, 1.0);
        let dy = rng.vec(b * c * oh * ow, -1.0, 1.0);
        let xt = be.upload_f32(&[b, c, h, w], &x).unwrap();
        let dyt = be.upload_f32(&[b, c, oh, ow], &dy).unwrap();
        let name = format!("upsample line {} s{s} {c}x{h}x{w}", r + 2);
        let y = be.upsample_f32(&xt, &[b, c, oh, ow], c, h, w, s, 1.0, false, cfg());
        let dx = be.upsample_backward_f32(&dyt, &[b, c, h, w], c, oh, ow, s, 1.0, cfg());
        match (y, dx) {
            (Ok(y), Ok(dx)) => {
                assert_close_report(&format!("{name} fwd"), &be.download_f32(&y).unwrap(), &up_ref(&f64s(&x), b, c, h, w, s), 1e-4, &mut fails, &mut worst);
                assert_close_report(&format!("{name} bwd"), &be.download_f32(&dx).unwrap(), &up_ref_bwd(&f64s(&dy), b, c, h, w, s), 1e-4, &mut fails, &mut worst);
                run += 1;
            }
            (a, b2) => fails.push(format!("{name}: backend error fwd={:?} bwd={:?}", a.err(), b2.err())),
        }
    }
    println!("==== cfg upsample ==== rows run {run}/{}, failed {}, max abs err {worst:.2e}", t.rows.len(), fails.len());
    for f in &fails {
        println!("  FAIL {f}");
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

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
                    out[(bb * out_c + base + c) * stride + s] = src[(bb * i.total + i.offset + c) * stride + s];
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
                        g[(bb * i.total + i.offset + c) * stride + s] = dy[(bb * out_c + base + c) * stride + s];
                    }
                }
            }
            base += i.channels;
            g
        })
        .collect()
}

#[test]
fn cfg_route_shapes() {
    let t = Table::load("route_shapes.csv");
    let be = OxideBackend::new().unwrap();
    let (mut run, mut fails, mut worst) = (0usize, vec![], 0.0f64);
    let (mut grouped, mut capped) = (0usize, 0usize);
    let batch = 2usize;
    for r in 0..t.rows.len() {
        let src: Vec<usize> = t.s(r, "src_channels").split('|').map(|v| v.parse().unwrap()).collect();
        let (groups, gid, out_c) = (t.u(r, "groups"), t.u(r, "group_id"), t.u(r, "out_c"));
        let (mut h, mut w) = (t.u(r, "h"), t.u(r, "w"));
        assert_eq!(src.len(), t.u(r, "n_sources"));
        // darknet: each source contributes total/groups channels starting at channels*group_id
        let ins: Vec<RouteIn> = src
            .iter()
            .map(|&total| RouteIn { total, offset: total / groups * gid, channels: total / groups })
            .collect();
        let sum_c: usize = ins.iter().map(|i| i.channels).sum();
        if sum_c != out_c {
            fails.push(format!("line {}: channel sum {sum_c} != csv out_c {out_c}", r + 2));
            continue;
        }
        if groups > 1 {
            grouped += t.u(r, "count");
        }
        // cap at ~3M source elements per batch item by shrinking h, w
        let tot: usize = src.iter().sum();
        while batch * (tot + sum_c) * h * w > 6_000_000 && (h > 8 || w > 8) {
            h = h.div_ceil(2);
            w = w.div_ceil(2);
            capped += 1;
        }
        let stride = h * w;
        let mut rng = Rng(8000 + r as u64);
        let srcs: Vec<Vec<f32>> = ins.iter().map(|i| rng.vec(batch * i.total * stride, -1.0, 1.0)).collect();
        let dy = rng.vec(batch * sum_c * stride, -1.0, 1.0);
        let tensors: Vec<_> = srcs.iter().zip(&ins).map(|(s, i)| be.upload_f32(&[batch, i.total, h, w], s).unwrap()).collect();
        let refs: Vec<_> = tensors.iter().collect();
        let (offsets, chans, totals): (Vec<usize>, Vec<usize>, Vec<usize>) = (
            ins.iter().map(|i| i.offset).collect(),
            ins.iter().map(|i| i.channels).collect(),
            ins.iter().map(|i| i.total).collect(),
        );
        let name = format!("route line {} src{src:?} g{groups}/{gid} {h}x{w}", r + 2);
        let y = match be.route_concat_f32(&refs, &offsets, &chans, &totals, &[batch, sum_c, h, w], stride, cfg()) {
            Ok(y) => y,
            Err(e) => {
                fails.push(format!("{name}: fwd error {e}"));
                continue;
            }
        };
        let srcs64: Vec<Vec<f64>> = srcs.iter().map(|s| f64s(s)).collect();
        assert_close_report(&format!("{name} fwd"), &be.download_f32(&y).unwrap(), &cpu_route(&srcs64, &ins, batch, stride), 1e-4, &mut fails, &mut worst);
        let dyt = be.upload_f32(&[batch, sum_c, h, w], &dy).unwrap();
        let shapes: Vec<Vec<usize>> = ins.iter().map(|i| vec![batch, i.total, h, w]).collect();
        match be.route_backward_f32(&dyt, &shapes, &offsets, &chans, &totals, stride, cfg()) {
            Ok(grads) => {
                let want = cpu_route_bwd(&f64s(&dy), &ins, batch, stride);
                for (k, g) in grads.iter().enumerate() {
                    assert_close_report(&format!("{name} bwd[{k}]"), &be.download_f32(g).unwrap(), &want[k], 1e-4, &mut fails, &mut worst);
                }
                run += 1;
            }
            Err(e) => fails.push(format!("{name}: bwd error {e}")),
        }
    }
    println!("==== cfg route ==== rows run {run}/{} (grouped-route layers {grouped}, spatially capped steps {capped}), failed {}, max abs err {worst:.2e}", t.rows.len(), fails.len());
    for f in &fails {
        println!("  FAIL {f}");
    }
    assert_eq!(grouped, 30, "expected the 30 groups=2 route layers");
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

fn parse_chw(s: &str) -> (usize, usize, usize) {
    let v: Vec<usize> = s.split('x').map(|p| p.parse().unwrap()).collect();
    (v[0], v[1], v[2])
}

/// Shortcut: same-shape `shortcut_add_f32` forward (and the delta
/// accumulation used by its backward) against a CPU sum, plus the 30
/// mismatched-shape layers (18 distinct rows): Darknet adds a differently
/// shaped source FLAT per batch sample (`src_i < min(outputs, src_outputs)`,
/// `shortcut_multilayer_*`), i.e. the source is cropped / zero-padded to the
/// layer's per-sample length.  `shortcut_fit_f32` implements that resize;
/// forward = `shortcut_add(x, fit(src))`, source gradient = `fit(delta)`.
/// Batch 2 so the per-sample strides differ (a flat add over the whole batch
/// would be wrong).  Reference = independent host loops.
#[test]
fn cfg_shortcut_shapes() {
    let t = Table::load("shortcut_shapes.csv");
    let be = OxideBackend::new().unwrap();
    let (mut same, mut mismatched, mut mismatch_layers) = (0usize, 0usize, 0usize);
    let (mut crop_rows, mut pad_rows, mut equal_flat_rows) = (0usize, 0usize, 0usize);
    let (mut fails, mut worst) = (vec![], 0.0f64);
    for r in 0..t.rows.len() {
        let cur = parse_chw(t.s(r, "cur_chw"));
        let srcs: Vec<(usize, usize, usize)> = t.s(r, "from_chw_list").split('|').map(parse_chw).collect();
        assert_eq!(srcs.len(), 1, "single-source shortcuts only");
        let from = srcs[0];
        let count = t.u(r, "count");
        let mut rng = Rng(9000 + r as u64);
        let (xn, yn) = (cur.0 * cur.1 * cur.2, from.0 * from.1 * from.2);
        if cur == from {
            let (x, y, d) = (rng.vec(xn, -1.0, 1.0), rng.vec(xn, -1.0, 1.0), rng.vec(xn, -1.0, 1.0));
            let xt = be.upload_f32(&[1, cur.0, cur.1, cur.2], &x).unwrap();
            let yt = be.upload_f32(&[1, cur.0, cur.1, cur.2], &y).unwrap();
            let want: Vec<f64> = x.iter().zip(&y).map(|(a, b)| *a as f64 + *b as f64).collect();
            let got = be.download_f32(&be.shortcut_add_f32(&xt, &yt, cfg()).unwrap()).unwrap();
            assert_close_report(&format!("shortcut {} fwd", t.s(r, "cur_chw")), &got, &want, 1e-6, &mut fails, &mut worst);
            // backward: delta of both branches accumulates the incoming delta
            let prev = rng.vec(xn, -1.0, 1.0);
            let pt = be.upload_f32(&[1, cur.0, cur.1, cur.2], &prev).unwrap();
            let dt = be.upload_f32(&[1, cur.0, cur.1, cur.2], &d).unwrap();
            let want: Vec<f64> = prev.iter().zip(&d).map(|(a, b)| *a as f64 + *b as f64).collect();
            let got = be.download_f32(&be.vector_add_f32(&pt, &dt, cfg()).unwrap()).unwrap();
            assert_close_report(&format!("shortcut {} bwd-accumulate", t.s(r, "cur_chw")), &got, &want, 1e-6, &mut fails, &mut worst);
            same += 1;
        } else {
            mismatched += 1;
            mismatch_layers += count;
            match yn.cmp(&xn) {
                std::cmp::Ordering::Less => pad_rows += 1,
                std::cmp::Ordering::Greater => crop_rows += 1,
                std::cmp::Ordering::Equal => equal_flat_rows += 1,
            }
            let what = format!("{} + {} (x{count})", t.s(r, "cur_chw"), t.s(r, "from_chw_list"));
            let batch = 2usize;
            let (x, y, d) = (rng.vec(batch * xn, -1.0, 1.0), rng.vec(batch * yn, -1.0, 1.0), rng.vec(batch * xn, -1.0, 1.0));
            let (xs, ys) = ([batch, cur.0, cur.1, cur.2], [batch, from.0, from.1, from.2]);
            let (xt, yt, dt) = (
                be.upload_f32(&xs, &x).unwrap(),
                be.upload_f32(&ys, &y).unwrap(),
                be.upload_f32(&xs, &d).unwrap(),
            );
            let m = xn.min(yn);
            // forward: out[b*xn + i] = x + (i < m ? y[b*yn + i] : 0)
            let mut want = vec![0.0f64; batch * xn];
            for b in 0..batch {
                for i in 0..xn {
                    want[b * xn + i] = x[b * xn + i] as f64 + if i < m { y[b * yn + i] as f64 } else { 0.0 };
                }
            }
            let fitted = be.shortcut_fit_f32(&yt, batch, &xs, cfg()).unwrap();
            assert_eq!(fitted.shape(), &xs, "{what}: fitted shape");
            let got = be.download_f32(&be.shortcut_add_f32(&xt, &fitted, cfg()).unwrap()).unwrap();
            assert_close_report(&format!("shortcut {what} fwd"), &got, &want, 1e-6, &mut fails, &mut worst);
            // backward: source gradient = delta cropped / zero-padded to the source size
            let mut want = vec![0.0f64; batch * yn];
            for b in 0..batch {
                for i in 0..m {
                    want[b * yn + i] = d[b * xn + i] as f64;
                }
            }
            let sg = be.shortcut_fit_f32(&dt, batch, &ys, cfg()).unwrap();
            assert_eq!(sg.shape(), &ys, "{what}: source gradient shape");
            let got = be.download_f32(&sg).unwrap();
            assert_close_report(&format!("shortcut {what} bwd source grad"), &got, &want, 0.0, &mut fails, &mut worst);
        }
    }
    println!(
        "==== cfg shortcut ==== same-shape rows {same}, mismatched rows {mismatched} ({mismatch_layers} layers: {pad_rows} zero-pad, {crop_rows} crop, {equal_flat_rows} equal flat length); max abs err {worst:.2e}"
    );
    assert_eq!(mismatch_layers, 30, "30 mismatched-shape shortcut layers expected");
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn shortcut_fit_edge_cases_and_errors() {
    let be = OxideBackend::new().unwrap();
    let x: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect(); // batch 2 x 3
    let xt = be.upload_f32(&[2, 3], &x).unwrap();
    // identity, pad, crop, 1-element crop
    let run = |len: usize| be.download_f32(&be.shortcut_fit_f32(&xt, 2, &[2, len], cfg()).unwrap()).unwrap();
    assert_eq!(run(3), x);
    assert_eq!(run(5), vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    assert_eq!(run(2), vec![1.0, 2.0, 4.0, 5.0]);
    assert_eq!(run(1), vec![1.0, 4.0]);
    // matches the host default implementation used by backends without a kernel
    assert_eq!(run(5), darknet_cuda::host_ops::shortcut_fit(&x, 2, 5));
    // Inconsistent arguments are errors, not panics.
    assert!(be.shortcut_fit_f32(&xt, 4, &[2, 3], cfg()).is_err(), "output not divisible");
    assert!(be.shortcut_fit_f32(&xt, 4, &[4, 3], cfg()).is_err(), "source not divisible by batch 4");
    assert!(be.shortcut_fit_f32(&xt, 0, &[2, 3], cfg()).is_err(), "batch 0");
}

// -------------------------------------------------------------- activations

const KINDS: [F32ActivationKind; 16] = [
    F32ActivationKind::Linear,
    F32ActivationKind::Loggy,
    F32ActivationKind::Hardtan,
    F32ActivationKind::Relie,
    F32ActivationKind::Logistic,
    F32ActivationKind::Relu,
    F32ActivationKind::Leaky,
    F32ActivationKind::Mish,
    F32ActivationKind::Swish,
    F32ActivationKind::Tanh,
    F32ActivationKind::Relu6,
    F32ActivationKind::Ramp,
    F32ActivationKind::Gelu,
    F32ActivationKind::Elu,
    F32ActivationKind::Selu,
    F32ActivationKind::HardMish,
];

fn kind_of(name: &str) -> Option<F32ActivationKind> {
    // Exhaustive match below turns any new trait variant into a compile error
    // here, forcing this test (and activations.csv coverage) to be revisited.
    let k = match name {
        "linear" => F32ActivationKind::Linear,
        "loggy" => F32ActivationKind::Loggy,
        "hardtan" => F32ActivationKind::Hardtan,
        "relie" => F32ActivationKind::Relie,
        "logistic" => F32ActivationKind::Logistic,
        "relu" => F32ActivationKind::Relu,
        "leaky" => F32ActivationKind::Leaky,
        "mish" => F32ActivationKind::Mish,
        "swish" => F32ActivationKind::Swish,
        "tanh" => F32ActivationKind::Tanh,
        "relu6" => F32ActivationKind::Relu6,
        "ramp" => F32ActivationKind::Ramp,
        "gelu" => F32ActivationKind::Gelu,
        "elu" => F32ActivationKind::Elu,
        "selu" => F32ActivationKind::Selu,
        "hard_mish" => F32ActivationKind::HardMish,
        _ => return None,
    };
    match k {
        F32ActivationKind::Linear
        | F32ActivationKind::Loggy
        | F32ActivationKind::Hardtan
        | F32ActivationKind::Relie
        | F32ActivationKind::Logistic
        | F32ActivationKind::Relu
        | F32ActivationKind::Leaky
        | F32ActivationKind::Mish
        | F32ActivationKind::Swish
        | F32ActivationKind::Tanh
        | F32ActivationKind::Relu6
        | F32ActivationKind::Ramp
        | F32ActivationKind::Gelu
        | F32ActivationKind::Elu
        | F32ActivationKind::Selu
        | F32ActivationKind::HardMish => {}
    }
    Some(k)
}

fn act64(kind: F32ActivationKind, x: f64) -> f64 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        F32ActivationKind::Hardtan => x.clamp(-1.0, 1.0),
        F32ActivationKind::Relie => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        F32ActivationKind::Logistic => 1.0 / (1.0 + (-x).exp()),
        F32ActivationKind::Relu => x.max(0.0),
        F32ActivationKind::Leaky => if x > 0.0 { x } else { 0.1 * x },
        F32ActivationKind::Mish => {
            let sp = if x > 30.0 { x } else { x.exp().ln_1p() };
            x * sp.tanh()
        }
        F32ActivationKind::Swish => x / (1.0 + (-x).exp()),
        F32ActivationKind::Tanh => x.tanh(),
        F32ActivationKind::Relu6 => x.clamp(0.0, 6.0),
        F32ActivationKind::Ramp => x * (x > 0.0) as i32 as f64 + 0.1 * x,
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507 * x
            } else {
                1.0507 * 1.6732 * (x.exp() - 1.0)
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                x
            } else if x > -2.0 {
                x * x / 2.0 + x
            } else {
                0.0
            }
        }
    }
}

fn grad64(kind: F32ActivationKind, x: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-x).exp());
    match kind {
        F32ActivationKind::Linear => 1.0,
        F32ActivationKind::Loggy => 2.0 * sig * (1.0 - sig),
        F32ActivationKind::Hardtan => (x > -1.0 && x < 1.0) as i32 as f64,
        F32ActivationKind::Relie => {
            if x > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        F32ActivationKind::Logistic => sig * (1.0 - sig),
        F32ActivationKind::Relu => if x > 0.0 { 1.0 } else { 0.0 },
        F32ActivationKind::Leaky => if x > 0.0 { 1.0 } else { 0.1 },
        F32ActivationKind::Mish => {
            let sp = if x > 30.0 { x } else { x.exp().ln_1p() };
            let t = sp.tanh();
            t + x * (1.0 - t * t) * sig
        }
        F32ActivationKind::Swish => sig + x * sig * (1.0 - sig),
        F32ActivationKind::Tanh => 1.0 - x.tanh() * x.tanh(),
        F32ActivationKind::Relu6 => if x > 0.0 && x < 6.0 { 1.0 } else { 0.0 },
        F32ActivationKind::Ramp => if x > 0.0 { 1.1 } else { 0.1 },
        F32ActivationKind::Gelu => {
            let t = (0.797885 * x + 0.035677 * x * x * x).tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (0.797885 + 3.0 * 0.035677 * x * x)
        }
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                1.0
            } else {
                x.exp()
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507
            } else {
                1.0507 * 1.6732 * x.exp()
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                1.0
            } else if x > -2.0 {
                x + 1.0
            } else {
                0.0
            }
        }
    }
}

/// f32 mirror of the kernel formulas (for non-finite inputs).
fn act32(kind: F32ActivationKind, x: f32) -> f32 {
    match kind {
        F32ActivationKind::Linear => x,
        F32ActivationKind::Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        F32ActivationKind::Hardtan => x.clamp(-1.0, 1.0),
        F32ActivationKind::Relie => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        F32ActivationKind::Logistic => 1.0 / (1.0 + (-x).exp()),
        F32ActivationKind::Relu => x.max(0.0),
        F32ActivationKind::Leaky => if x > 0.0 { x } else { 0.1 * x },
        F32ActivationKind::Mish => x * (1.0 + x.exp()).ln().tanh(),
        F32ActivationKind::Swish => x / (1.0 + (-x).exp()),
        F32ActivationKind::Tanh => x.tanh(),
        // `.max(0).min(6)` (device fmax/fmin): NaN -> 0 like relu.
        F32ActivationKind::Relu6 => x.max(0.0).min(6.0),
        F32ActivationKind::Ramp => (if x > 0.0 { x } else { 0.0 }) + 0.1 * x,
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
        F32ActivationKind::Elu => {
            if x >= 0.0 {
                x
            } else {
                x.exp() - 1.0
            }
        }
        F32ActivationKind::Selu => {
            if x >= 0.0 {
                1.0507 * x
            } else {
                1.0507 * 1.6732 * (x.exp() - 1.0)
            }
        }
        F32ActivationKind::HardMish => {
            if x > 0.0 {
                x
            } else if x > -2.0 {
                x * x / 2.0 + x
            } else {
                0.0
            }
        }
    }
}

fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-6 * b.abs()
}

fn edge_values() -> Vec<f32> {
    let mut v = vec![0.0f32, -0.0];
    for m in [1e-30f32, 1e-6, 0.05, 0.5, 1.0, 2.5, 5.0, 5.999, 6.0, 6.001, 10.0, 19.99, 20.0, 20.01, 25.0, 30.0, 88.0, 89.0, 100.0, 1e4] {
        v.push(m);
        v.push(-m);
    }
    v
}

#[test]
fn cfg_activations_supported_and_unsupported() {
    let t = Table::load("activations.csv");
    let be = OxideBackend::new().unwrap();
    let edges = edge_values();
    let sizes = [1usize, 2, 31, 32, 33, 255, 257, 4099, 65_537, (1 << 20) + 3];
    let (mut supported, mut unsupported, mut fails) = (vec![], vec![], vec![]);
    for r in 0..t.rows.len() {
        let name = t.s(r, "activation").to_string();
        match kind_of(&name) {
            Some(k) => supported.push((name, k, t.u(r, "count"))),
            None => unsupported.push((name, t.u(r, "count"))),
        }
    }
    // tanh is supported but unused by any cfg: still checked.
    if !supported.iter().any(|s| s.1 == F32ActivationKind::Tanh) {
        supported.push(("tanh".into(), F32ActivationKind::Tanh, 0));
    }
    for (name, kind, uses) in &supported {
        let mut worst = [0.0f64; 2];
        for &n in &sizes {
            let mut rng = Rng(11 + n as u64);
            let mut x = rng.vec(n, -8.0, 8.0);
            for (i, e) in edges.iter().enumerate().take(n) {
                x[(i * 7) % n] = *e; // spread edge values through the tensor
            }
            let d = rng.vec(n, -2.0, 2.0);
            let xt = be.upload_f32(&[n], &x).unwrap();
            let dt = be.upload_f32(&[n], &d).unwrap();
            let f = be.download_f32(&be.activation_f32(&xt, *kind, cfg()).unwrap()).unwrap();
            let g = be.download_f32(&be.activation_backward_f32(&xt, &dt, *kind, cfg()).unwrap()).unwrap();
            for i in 0..n {
                let (w, wg) = (act64(*kind, x[i] as f64), d[i] as f64 * grad64(*kind, x[i] as f64));
                let (ef, eg) = ((f[i] as f64 - w).abs(), (g[i] as f64 - wg).abs());
                worst[0] = worst[0].max(ef);
                worst[1] = worst[1].max(eg);
                if !(ef <= 1e-4 + 1e-4 * w.abs()) {
                    fails.push(format!("{name} fwd n={n} x={} got {} want {w}", x[i], f[i]));
                }
                if !(eg <= 1e-4 + 1e-4 * wg.abs()) {
                    fails.push(format!("{name} bwd n={n} x={} got {} want {wg}", x[i], g[i]));
                }
            }
        }
        // Non-finite inputs: must equal the f32 mirror of the kernel formula.
        let nf = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let xt = be.upload_f32(&[4], &nf).unwrap();
        let f = be.download_f32(&be.activation_f32(&xt, *kind, cfg()).unwrap()).unwrap();
        for i in 0..4 {
            if !same_f32(f[i], act32(*kind, nf[i])) {
                fails.push(format!("{name} fwd non-finite x={} got {} want {}", nf[i], f[i], act32(*kind, nf[i])));
            }
        }
        println!("  activation {name:<8} cfg uses {uses:>5}: max abs err fwd {:.2e} bwd {:.2e}; fwd(NaN,+inf,-inf,1) = {:?}", worst[0], worst[1], f);
    }
    println!("==== cfg activations ==== supported+run: {}, failed: {}", supported.len(), fails.len());
    // relu6 and ramp are now expressible (F32ActivationKind::{Relu6, Ramp}); no
    // cfg activation may be left without a kind except the ones C++ itself
    // does not give a Rust CPU path (none occur in activations.csv).
    for (name, uses) in &unsupported {
        println!("  activation {name}: {uses} cfg uses -> NOT expressible via F32ActivationKind (CPU fallback)");
    }
    assert!(unsupported.is_empty(), "cfg activations without an F32ActivationKind: {unsupported:?}");
    assert!(fails.is_empty(), "{}", fails[..fails.len().min(40)].join("\n"));
}

// ------------------------------------------ conv + bias/BN + activation chain

struct Pair {
    label: String,
    geom: Geom,
    kind: F32ActivationKind,
    bn: bool,
    batch: usize,
    real_size: bool,
}

fn run_pair(be: &OxideBackend, p: &Pair, seed: u64) -> Result<[f64; 3], String> {
    let (c, _) = if p.real_size { (p.geom.case(p.batch), false) } else { shrink(&p.geom, p.batch, 10_000_000, 20_000_000) };
    let (ohw, kk) = (c.oh() * c.ow(), c.cpg() * c.k * c.k);
    let mut rng = Rng(seed);
    let a = (3.0 / kk as f32).sqrt();
    let x = rng.vec(c.x_len(), -1.0, 1.0);
    let w = rng.vec(c.w_len(), -a, a);
    let scale = if p.bn { rng.vec(c.f, 1.0, 3.0) } else { vec![1.0; c.f] };
    let bias = rng.vec(c.f, -0.5, 0.5);
    let da = rng.vec(c.y_len(), -1.0, 1.0);
    let er = |op: &str, e: darknet_cuda_oxide::OxideError| format!("{op}: {e}");

    // ---- f64 chain
    let (xd, wd) = (f64s(&x), f64s(&w));
    let (y, _, _) = conv_refs_impl(&c, &xd, &wd, &vec![0.0; c.y_len()]);
    let chan = |i: usize| (i / ohw) % c.f;
    let z: Vec<f64> = y.iter().enumerate().map(|(i, v)| v * scale[chan(i)] as f64 + bias[chan(i)] as f64).collect();
    let a_out: Vec<f64> = z.iter().map(|&v| act64(p.kind, v)).collect();

    // ---- GPU chain
    let xt = be.upload_f32(&[c.b, c.c, c.h, c.w], &x).map_err(|e| er("upload", e))?;
    let wt = be.upload_f32(&[c.f, c.cpg(), c.k, c.k], &w).map_err(|e| er("upload", e))?;
    let st = be.upload_f32(&[c.f], &scale).map_err(|e| er("upload", e))?;
    let bt = be.upload_f32(&[c.f], &bias).map_err(|e| er("upload", e))?;
    let dat = be.upload_f32(&[c.b, c.f, c.oh(), c.ow()], &da).map_err(|e| er("upload", e))?;
    let oshape = [c.b, c.f, c.oh(), c.ow()];
    let yt = be
        .conv2d_f32(&xt, &wt, &oshape, c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg())
        .map_err(|e| er("conv fwd", e))?;
    let yt = if p.bn { be.channel_scale_f32(&yt, &st, c.f, ohw, cfg()).map_err(|e| er("scale", e))? } else { yt };
    let zt = be.bias_add_f32(&yt, &bt, c.f, ohw, cfg()).map_err(|e| er("bias", e))?;
    let at = be.activation_f32(&zt, p.kind, cfg()).map_err(|e| er("act", e))?;
    // The reference gradient is evaluated at the GPU's own f32 pre-activation:
    // ReLU / Leaky have a derivative discontinuity at 0, so a pre-activation
    // within f32 rounding of 0 could otherwise flip 0.1 <-> 1 between the f32
    // chain and the f64 chain (seen on real-size LegoGears layer 1, 573k elems).
    let z_gpu = be.download_f32(&zt).map_err(|e| er("dl z", e))?;
    let dy: Vec<f64> = z_gpu
        .iter()
        .enumerate()
        .map(|(i, &v)| da[i] as f64 * grad64(p.kind, v as f64) * scale[chan(i)] as f64)
        .collect();
    let (_, dx, dw) = conv_refs_impl(&c, &xd, &wd, &dy);
    let dz = be.activation_backward_f32(&zt, &dat, p.kind, cfg()).map_err(|e| er("act bwd", e))?;
    let dzt = if p.bn { be.channel_scale_f32(&dz, &st, c.f, ohw, cfg()).map_err(|e| er("scale bwd", e))? } else { dz };
    let dxt = be
        .conv2d_backward_input_f32(&dzt, &wt, &[c.b, c.c, c.h, c.w], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg())
        .map_err(|e| er("dgrad", e))?;
    let dwt = be
        .conv2d_backward_weights_f32(&xt, &dzt, &[c.f, c.cpg(), c.k, c.k], c.c, c.h, c.w, c.f, c.k, c.sx, c.sy, c.dil, c.pad, c.groups, cfg())
        .map_err(|e| er("wgrad", e))?;
    let mut max = [0.0f64; 3];
    let mut bad = vec![];
    let outs = [
        ("act out", be.download_f32(&at).map_err(|e| er("dl", e))?, a_out),
        ("dgrad", be.download_f32(&dxt).map_err(|e| er("dl", e))?, dx),
        ("wgrad", be.download_f32(&dwt).map_err(|e| er("dl", e))?, dw),
    ];
    for (i, (n, got, want)) in outs.iter().enumerate() {
        let (worst, nbad, first) = compare(n, got, want, None);
        max[i] = worst;
        if nbad > 0 {
            bad.push(format!("{n}: {nbad} bad, e.g. {}", first.unwrap_or_default()));
        }
    }
    if bad.is_empty() { Ok(max) } else { Err(format!("{} [{}]: {}", p.label, c.desc(), bad.join("; "))) }
}

#[test]
fn cfg_conv_bias_activation_chain() {
    let geoms = load_geoms();
    let pairs_csv = Table::load("conv_activation_pairs.csv");
    let (sel, _) = selected();
    let mut pairs: Vec<Pair> = vec![];

    // (1) deterministic sample of the real (geometry, activation, bn) pairs:
    //     up to 10 per supported activation, evenly spaced over its csv rows,
    //     restricted to rule-selected geometries.
    for (name, kind) in [("linear", F32ActivationKind::Linear), ("logistic", F32ActivationKind::Logistic),
        ("relu", F32ActivationKind::Relu), ("leaky", F32ActivationKind::Leaky), ("mish", F32ActivationKind::Mish),
        ("swish", F32ActivationKind::Swish)] {
        let rows: Vec<usize> = (0..pairs_csv.rows.len())
            .filter(|&r| pairs_csv.s(r, "activation") == name)
            .filter(|&r| {
                let g = &geoms[pairs_csv.u(r, "geom_id")];
                g.in_rule()
            })
            .collect();
        let take = rows.len().min(10);
        for j in 0..take {
            let r = rows[j * rows.len() / take.max(1)];
            let g = geoms[pairs_csv.u(r, "geom_id")].clone();
            pairs.push(Pair {
                label: format!("pairs.csv line {} geom {} {name} bn{}", r + 2, pairs_csv.u(r, "geom_id"), pairs_csv.u(r, "batch_normalize")),
                geom: g,
                kind,
                bn: pairs_csv.u(r, "batch_normalize") != 0,
                batch: 1,
                real_size: false,
            });
        }
    }
    let n_csv = pairs.len();

    // (2) every activation (all 7 incl. tanh) x several fixed geometries.
    let mk = |c, h, w, f, k, s, g| Geom {
        row: 0, c, h, w, f, k, sx: s, sy: s, dil: 1, pad: k / 2, groups: g, out_h: 0, out_w: 0, origin: "synthetic".into(),
    };
    let fixed = [
        ("stem k3 s1 c3", mk(3, 32, 32, 8, 3, 1, 1)),
        ("1x1", mk(16, 24, 24, 32, 1, 1, 1)),
        ("k3 s2 odd", mk(16, 25, 25, 16, 3, 2, 1)),
        ("depthwise", mk(8, 20, 20, 8, 3, 1, 8)),
        ("groups3 odd fpg", mk(9, 17, 17, 15, 3, 1, 3)),
        ("k5 s2", mk(6, 21, 21, 12, 5, 2, 2)),
    ];
    for (ai, kind) in KINDS.iter().enumerate() {
        for (gi, (n, g)) in fixed.iter().enumerate() {
            pairs.push(Pair {
                label: format!("cross {n} x {kind:?}"),
                geom: g.clone(),
                kind: *kind,
                bn: (ai + gi) % 2 == 1,
                batch: if gi == 2 { 2 } else { 1 },
                real_size: true,
            });
        }
    }
    let n_cross = pairs.len() - n_csv;

    // (3) LegoGears_v2 layers at real size with their real activation / bn.
    for (layer, act, bn, g) in load_lego() {
        pairs.push(Pair {
            label: format!("LegoGears_v2 layer {layer} {act} bn{bn}"),
            geom: g,
            kind: kind_of(&act).expect("LegoGears activation supported"),
            bn: bn != "0",
            batch: 1,
            real_size: true,
        });
    }
    let n_lego = pairs.len() - n_csv - n_cross;

    let t0 = std::time::Instant::now();
    let (mut fails, mut per_kind) = (vec![], std::collections::BTreeMap::<String, usize>::new());
    let mut worst = [0.0f64; 3];
    let modes = [ConvMode::Direct, ConvMode::Gemm];
    let backends: Vec<OxideBackend> = modes
        .iter()
        .map(|&m| OxideBackend::new().unwrap().with_conv_mode(m).with_gemm_mode(GemmMode::Fp32))
        .collect();
    for (i, p) in pairs.iter().enumerate() {
        let be = &backends[i % 2];
        match run_pair(be, p, 20_000 + i as u64) {
            Ok(m) => {
                for j in 0..3 {
                    worst[j] = worst[j].max(m[j]);
                }
            }
            Err(e) => fails.push(format!("{:?}: {e}", modes[i % 2])),
        }
        *per_kind.entry(format!("{:?}", p.kind)).or_default() += 1;
    }
    println!(
        "==== cfg conv+bias(+BN scale)+activation chain ==== pairs {} (csv sample {n_csv}, cross {n_cross}, LegoGears {n_lego}), failed {}, {:.1}s",
        pairs.len(), fails.len(), t0.elapsed().as_secs_f64()
    );
    println!("  max abs err: activation out {:.2e}, dgrad {:.2e}, wgrad {:.2e}; pairs per activation {per_kind:?}", worst[0], worst[1], worst[2]);
    for f in &fails {
        println!("  FAIL {f}");
    }
    assert!(pairs.len() >= 60);
    assert!(sel.len() > 900);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}
