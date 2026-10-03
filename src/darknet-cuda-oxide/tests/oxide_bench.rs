//! P7 benchmark harness for the cuda-oxide backend (all tests `#[ignore]`d; driven by
//! `tools/oxide_bench.py`).  Every result is one line `BENCHJSON {...}` on stdout.
//!
//! Run by hand:
//!   `... oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 1 -- --ignored --nocapture --test-threads=1 bench_micro`
//!   (tests: `bench_micro`, `bench_conv`)
//!
//! Method: every sample is the mean device time (CUDA events on the backend stream)
//! of `iters` back-to-back launches after one warm-up launch; `iters` is calibrated so a
//! sample lasts about `DARKNET_BENCH_TARGET_MS` (default 20).  `DARKNET_BENCH_ROUNDS`
//! (default 7) samples are taken per (item, mode); modes are interleaved round by round
//! (laptop clocks drift) and median / min / p95 over the rounds are reported.
//! Env: `DARKNET_BENCH_MODES` (micro: gemm modes `fp32,tf32`; conv: `direct,gemm-fp32,gemm-tf32`),
//! `DARKNET_BENCH_FILTER` (substring on the item name), `DARKNET_BENCH_CONV_BATCH` (default 1).

use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};
use darknet_cuda_oxide::{ConvMode, GemmMode, OxideBackend, OxideTensorF32};

fn cfg() -> ElementwiseLaunchConfig {
    ElementwiseLaunchConfig::default()
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn env_list(k: &str, d: &str) -> Vec<String> {
    std::env::var(k).unwrap_or_else(|_| d.into()).split(',').map(|s| s.trim().to_string()).collect()
}

fn filter_ok(name: &str) -> bool {
    std::env::var("DARKNET_BENCH_FILTER").map(|f| name.contains(&f)).unwrap_or(true)
}

struct Stats {
    median: f64,
    min: f64,
    p95: f64,
}

fn stats(v: &mut [f64]) -> Stats {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = v[(((v.len() as f64) * 0.95).ceil() as usize).clamp(1, v.len()) - 1];
    Stats { median: v[v.len() / 2], min: v[0], p95 }
}

/// One benchmark subject: how to run it once (enqueue only) on a given backend.
type Op<'a> = Box<dyn FnMut() -> Result<(), darknet_cuda_oxide::OxideError> + 'a>;

/// Interleaved measurement of one item over several (label, backend, op) variants.
#[allow(clippy::too_many_arguments)]
fn measure(
    group: &str,
    name: &str,
    params: &str,
    flops: f64,
    bytes: f64,
    variants: &mut [(String, &OxideBackend, Op<'_>)],
) {
    if !filter_ok(name) {
        return;
    }
    let rounds = env_usize("DARKNET_BENCH_ROUNDS", 7);
    let target = env_usize("DARKNET_BENCH_TARGET_MS", 20) as f64;
    // Calibrate iterations per variant from a short probe.
    let mut iters = Vec::new();
    for (_, be, op) in variants.iter_mut() {
        let probe = be.time_gpu_ms(3, &mut **op).unwrap().max(1e-4);
        iters.push(((target / probe) as usize).clamp(3, 2000));
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
    for _ in 0..rounds {
        for (i, (_, be, op)) in variants.iter_mut().enumerate() {
            samples[i].push(be.time_gpu_ms(iters[i], &mut **op).unwrap());
        }
    }
    for (i, (mode, _, _)) in variants.iter().enumerate() {
        let st = stats(&mut samples[i]);
        let gflops = if flops > 0.0 { flops / st.median / 1e6 } else { 0.0 };
        let gbs = if bytes > 0.0 { bytes / st.median / 1e6 } else { 0.0 };
        println!(
            "BENCHJSON {{\"group\":\"{group}\",\"name\":\"{name}\",\"mode\":\"{mode}\",\"params\":\"{params}\",\
\"flops\":{flops:.0},\"bytes\":{bytes:.0},\"median_ms\":{:.5},\"min_ms\":{:.5},\"p95_ms\":{:.5},\
\"iters\":{},\"rounds\":{rounds},\"gflops\":{gflops:.2},\"gbs\":{gbs:.2}}}",
            st.median, st.min, st.p95, iters[i]
        );
    }
}

fn single(group: &str, name: &str, params: &str, flops: f64, bytes: f64, be: &OxideBackend, op: Op<'_>) {
    measure(group, name, params, flops, bytes, &mut [("-".to_string(), be, op)]);
}

fn pattern(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i.wrapping_mul(7).wrapping_add(seed * 13)) % 17) as f32 - 8.0) / 8.0).collect()
}

fn gemm_modes(list: &[String]) -> Vec<(String, GemmMode)> {
    list.iter()
        .map(|m| {
            let mode = match m.as_str() {
                "tf32" => GemmMode::Tf32,
                "fp16" => GemmMode::Fp16,
                _ => GemmMode::Fp32,
            };
            (m.clone(), mode)
        })
        .collect()
}

/// (m, n, k) = (filters, out_h*out_w, in_c/groups*size^2) of the 21 LegoGears_v2 convs.
const LEGO_GEMM: [(usize, usize, usize); 21] = [
    (32, 8960, 27), (64, 2240, 288), (64, 2240, 576), (32, 2240, 288), (32, 2240, 288), (64, 2240, 64),
    (128, 560, 576), (64, 560, 576), (64, 560, 576), (128, 560, 128),
    (256, 140, 1152), (128, 140, 1152), (128, 140, 1152), (256, 140, 256),
    (512, 35, 2304), (256, 35, 512), (512, 35, 2304), (30, 35, 512),
    (128, 35, 256), (256, 140, 3456), (30, 140, 256),
];

#[test]
#[ignore]
fn bench_micro() {
    let modes = gemm_modes(&env_list("DARKNET_BENCH_MODES", "fp32,tf32"));
    let backends: Vec<OxideBackend> =
        modes.iter().map(|(_, m)| OxideBackend::new().unwrap().with_gemm_mode(*m)).collect();
    let b0 = &backends[0];

    // ---- GEMM: square + conv shaped, per mode.
    let mut shapes: Vec<(String, usize, usize, usize)> = [256usize, 512, 1024, 2048, 4096]
        .iter()
        .map(|&s| (format!("gemm_sq{s}"), s, s, s))
        .collect();
    for (i, &(m, n, k)) in LEGO_GEMM.iter().enumerate() {
        shapes.push((format!("gemm_lego{i:02}_m{m}n{n}k{k}"), m, n, k));
    }
    for (name, m, n, k) in shapes {
        if !filter_ok(&name) {
            continue;
        }
        let (a, b) = (pattern(m * k, 1), pattern(k * n, 2));
        let ts: Vec<_> = backends
            .iter()
            .map(|be| (be.upload_f32(&[m, k], &a).unwrap(), be.upload_f32(&[k, n], &b).unwrap()))
            .collect();
        let rounds = env_usize("DARKNET_BENCH_ROUNDS", 7);
        let target = env_usize("DARKNET_BENCH_TARGET_MS", 20) as f64;
        let flops = 2.0 * (m * n * k) as f64;
        let bytes = 4.0 * (m * k + k * n + m * n) as f64;
        let mut iters = Vec::new();
        for (i, be) in backends.iter().enumerate() {
            let p = be.bench_gemm_f32(&ts[i].0, &ts[i].1, m, n, k, 3).unwrap().max(1e-4);
            iters.push(((target / p) as usize).clamp(3, 2000));
        }
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); backends.len()];
        for _ in 0..rounds {
            for (i, be) in backends.iter().enumerate() {
                samples[i].push(be.bench_gemm_f32(&ts[i].0, &ts[i].1, m, n, k, iters[i]).unwrap());
            }
        }
        for (i, (mode, _)) in modes.iter().enumerate() {
            let st = stats(&mut samples[i]);
            println!(
                "BENCHJSON {{\"group\":\"gemm\",\"name\":\"{name}\",\"mode\":\"{mode}\",\"params\":\"m={m} n={n} k={k}\",\
\"flops\":{flops:.0},\"bytes\":{bytes:.0},\"median_ms\":{:.5},\"min_ms\":{:.5},\"p95_ms\":{:.5},\
\"iters\":{},\"rounds\":{rounds},\"gflops\":{:.2},\"gbs\":{:.2}}}",
                st.median, st.min, st.p95, iters[i], flops / st.median / 1e6, bytes / st.median / 1e6
            );
        }
    }

    // ---- im2col (LegoGears conv geometries, single image).
    let csv = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legogears_v2_convs.csv")).unwrap();
    let convs = parse_convs(&csv);
    for (i, cv) in convs.iter().enumerate() {
        let name = format!("im2col_lego{i:02}");
        if !filter_ok(&name) {
            continue;
        }
        let x = b0.upload_f32(&[1, cv.c, cv.h, cv.w], &pattern(cv.c * cv.h * cv.w, 3)).unwrap();
        let ((_, cols_bytes), mut st) = {
            let probe = b0.bench_im2col_f32(&x, cv.c, cv.h, cv.w, cv.k, cv.stride, cv.pad, 3).unwrap();
            let iters = ((20.0 / probe.0.max(1e-4)) as usize).clamp(3, 2000);
            let mut v = Vec::new();
            for _ in 0..env_usize("DARKNET_BENCH_ROUNDS", 7) {
                v.push(b0.bench_im2col_f32(&x, cv.c, cv.h, cv.w, cv.k, cv.stride, cv.pad, iters).unwrap().0);
            }
            (probe, v)
        };
        let s = stats(&mut st);
        let bytes = (cv.c * cv.h * cv.w * 4 + cols_bytes) as f64;
        println!(
            "BENCHJSON {{\"group\":\"im2col\",\"name\":\"{name}\",\"mode\":\"-\",\"params\":\"c={} {}x{} k={} s={} p={}\",\
\"flops\":0,\"bytes\":{bytes:.0},\"median_ms\":{:.5},\"min_ms\":{:.5},\"p95_ms\":{:.5},\"iters\":0,\"rounds\":{},\
\"gflops\":0,\"gbs\":{:.2}}}",
            cv.c, cv.h, cv.w, cv.k, cv.stride, cv.pad, s.median, s.min, s.p95, env_usize("DARKNET_BENCH_ROUNDS", 7), bytes / s.median / 1e6
        );
    }

    // ---- elementwise / activation / bias / shortcut (16M floats = 64 MB per tensor).
    let n = 16 * 1024 * 1024usize;
    let (c, hw) = (64usize, 64 * 64usize);
    let x = b0.upload_f32(&[n], &pattern(n, 4)).unwrap();
    let y = b0.upload_f32(&[n], &pattern(n, 5)).unwrap();
    let bias = b0.upload_f32(&[c], &pattern(c, 6)).unwrap();
    let nb = (n * 4) as f64;
    let ew = |name: &str, params: &str, flops: f64, bytes: f64, op: Op<'_>| {
        single("elementwise", name, params, flops, bytes, b0, op);
    };
    ew("vector_add", "n=16M", n as f64, 3.0 * nb, Box::new(|| b0.vector_add_f32(&x, &y, cfg()).map(|_| ())));
    ew("saxpy", "n=16M", 2.0 * n as f64, 3.0 * nb, Box::new(|| b0.saxpy_f32(0.5, &x, &y, cfg()).map(|_| ())));
    ew("shortcut_add", "n=16M", n as f64, 3.0 * nb, Box::new(|| b0.shortcut_add_f32(&x, &y, cfg()).map(|_| ())));
    ew("bias_add", "n=16M c=64 hw=4096", n as f64, 2.0 * nb, Box::new(|| b0.bias_add_f32(&x, &bias, c, hw, cfg()).map(|_| ())));
    ew("channel_scale", "n=16M c=64 hw=4096", n as f64, 2.0 * nb, Box::new(|| b0.channel_scale_f32(&x, &bias, c, hw, cfg()).map(|_| ())));
    ew("bias_backward", "n=16M c=64 hw=4096", n as f64, nb, Box::new(|| b0.bias_backward_f32(&x, c, hw, cfg()).map(|_| ())));
    for (nm, k) in [
        ("act_linear", F32ActivationKind::Linear),
        ("act_leaky", F32ActivationKind::Leaky),
        ("act_relu", F32ActivationKind::Relu),
        ("act_logistic", F32ActivationKind::Logistic),
        ("act_mish", F32ActivationKind::Mish),
        ("act_swish", F32ActivationKind::Swish),
        ("act_tanh", F32ActivationKind::Tanh),
    ] {
        ew(nm, "n=16M", 0.0, 2.0 * nb, Box::new({ let xr = &x; move || b0.activation_f32(xr, k, cfg()).map(|_| ()) }));
    }
    ew("act_leaky_backward", "n=16M", 0.0, 3.0 * nb, Box::new(|| b0.activation_backward_f32(&x, &y, F32ActivationKind::Leaky, cfg()).map(|_| ())));
    ew("sgd_update", "n=16M", 4.0 * n as f64, 4.0 * nb, Box::new(|| b0.sgd_update_f32(&x, &y, 0.01, 0.9, 5e-4, 16, cfg()).map(|_| ())));

    // ---- pooling / upsample / route on [8,64,128,128] (8.4M floats).
    let (pb, pc, ph, pw) = (8usize, 64usize, 128usize, 128usize);
    let pn = pb * pc * ph * pw;
    let px = b0.upload_f32(&[pb, pc, ph, pw], &pattern(pn, 7)).unwrap();
    let pbytes = (pn * 4) as f64;
    let sp = |name: &str, params: &str, bytes: f64, op: Op<'_>| single("spatial", name, params, 0.0, bytes, b0, op);
    sp("maxpool_2x2s2", "[8,64,128,128]", pbytes * 1.25, Box::new(|| b0.maxpool_f32(&px, &[pb, pc, ph / 2, pw / 2], pc, ph, pw, 2, 2, 2, 0, cfg()).map(|_| ())));
    sp("maxpool_3x3s1", "[8,64,128,128]", pbytes * 2.0, Box::new(|| b0.maxpool_f32(&px, &[pb, pc, ph, pw], pc, ph, pw, 3, 1, 1, 2, cfg()).map(|_| ())));
    sp("avgpool_global", "[8,64,128,128]", pbytes, Box::new(|| b0.avgpool_f32(&px, &[pb, pc, 1, 1], pc, ph, pw, cfg()).map(|_| ())));
    sp("upsample_x2", "[8,64,128,128]", pbytes * 5.0, Box::new(|| b0.upsample_f32(&px, &[pb, pc, ph * 2, pw * 2], pc, ph, pw, 2, 1.0, false, cfg()).map(|_| ())));
    let half = b0.upload_f32(&[pb, 32, ph, pw], &pattern(pb * 32 * ph * pw, 8)).unwrap();
    sp("route_concat", "2x[8,32,128,128]->[8,64,128,128]", pbytes * 2.0, Box::new(|| {
        b0.route_concat_f32(&[&half, &half], &[0, 0], &[32, 32], &[32, 32], &[pb, 64, ph, pw], ph * pw, cfg()).map(|_| ())
    }));
    // ---- softmax and batch norm (training).
    let (groups, gs) = (8192usize, 1000usize);
    let sx = b0.upload_f32(&[groups, gs], &pattern(groups * gs, 9)).unwrap();
    single("norm", "softmax_8192x1000", "groups=8192 size=1000", 0.0, (groups * gs * 8) as f64, b0,
        Box::new(|| b0.softmax_f32(&sx, &[groups, gs], gs, cfg()).map(|_| ())));
    let shape = [8usize, 64, 64, 64];
    let bn = shape.iter().product::<usize>();
    let bx = b0.upload_f32(&shape, &pattern(bn, 10)).unwrap();
    let bdy = b0.upload_f32(&shape, &pattern(bn, 11)).unwrap();
    let (sc, bi, rm, rv) = (
        b0.upload_f32(&[64], &vec![1.0; 64]).unwrap(),
        b0.upload_f32(&[64], &vec![0.0; 64]).unwrap(),
        b0.upload_f32(&[64], &vec![0.0; 64]).unwrap(),
        b0.upload_f32(&[64], &vec![1.0; 64]).unwrap(),
    );
    single("norm", "bn_train_fwd", "[8,64,64,64]", 0.0, (bn * 4 * 3) as f64, b0,
        Box::new(|| b0.batchnorm_train_forward_f32(&bx, &sc, &bi, &rm, &rv, &shape, 0.9, cfg()).map(|_| ())));
    let fwd = b0.batchnorm_train_forward_f32(&bx, &sc, &bi, &rm, &rv, &shape, 0.9, cfg()).unwrap();
    single("norm", "bn_train_bwd", "[8,64,64,64]", 0.0, (bn * 4 * 3) as f64, b0,
        Box::new(|| b0.batchnorm_train_backward_f32(&bdy, &fwd.x_hat, &fwd.variance, &sc, &shape, cfg()).map(|_| ())));
}

struct ConvGeom {
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    k: usize,
    stride: usize,
    pad: usize,
    groups: usize,
    oh: usize,
    ow: usize,
}

fn parse_convs(csv: &str) -> Vec<ConvGeom> {
    // layer,in_c,in_h,in_w,filters,size,stride_x,stride_y,dilation,padding,groups,activation,batch_normalize,out_h,out_w
    csv.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            let u = |i: usize| f[i].parse::<usize>().unwrap();
            ConvGeom { c: u(1), h: u(2), w: u(3), f: u(4), k: u(5), stride: u(6), pad: u(9), groups: u(10), oh: u(13), ow: u(14) }
        })
        .collect()
}

#[test]
#[ignore]
fn bench_conv() {
    let csv = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/legogears_v2_convs.csv")).unwrap();
    let convs = parse_convs(&csv);
    assert_eq!(convs.len(), 21);
    let batch = env_usize("DARKNET_BENCH_CONV_BATCH", 1);
    let modes = env_list("DARKNET_BENCH_MODES", "direct,gemm-fp32,gemm-tf32");
    let backends: Vec<OxideBackend> = modes
        .iter()
        .map(|m| {
            let (cm, gm) = match m.as_str() {
                "direct" => (ConvMode::Direct, GemmMode::Fp32),
                "gemm-tf32" => (ConvMode::Gemm, GemmMode::Tf32),
                // fwd only, native f16 storage path (fused bias + leaky), see below.
                "gemm-fp16" => (ConvMode::Gemm, GemmMode::Fp16),
                "auto" => (ConvMode::Auto, GemmMode::Auto),
                _ => (ConvMode::Gemm, GemmMode::Fp32),
            };
            OxideBackend::new().unwrap().with_conv_mode(cm).with_gemm_mode(gm)
        })
        .collect();
    for (li, g) in convs.iter().enumerate() {
        let xs = [batch, g.c, g.h, g.w];
        let ys = [batch, g.f, g.oh, g.ow];
        let ws = [g.f, g.c / g.groups, g.k, g.k];
        let (xl, yl, wl) = (xs.iter().product(), ys.iter().product(), ws.iter().product());
        let (xh, yh, wh) = (pattern(xl, 1), pattern(yl, 2), pattern(wl, 3));
        let flops = 2.0 * (batch * g.f * (g.c / g.groups) * g.k * g.k * g.oh * g.ow) as f64;
        let bytes = 4.0 * (xl + yl + wl) as f64;
        let params = format!("c={} {}x{} f={} k={} s={} g={} batch={batch}", g.c, g.h, g.w, g.f, g.k, g.stride, g.groups);
        let ts: Vec<[OxideTensorF32; 3]> = backends
            .iter()
            .map(|be| [be.upload_f32(&xs, &xh).unwrap(), be.upload_f32(&ys, &yh).unwrap(), be.upload_f32(&ws, &wh).unwrap()])
            .collect();
        for pass in ["fwd", "dgrad", "wgrad"] {
            let mut vs: Vec<(String, &OxideBackend, Op<'_>)> = Vec::new();
            for (i, be) in backends.iter().enumerate() {
                let [x, dy, w] = &ts[i];
                if be.f16_storage_enabled() {
                    // Inference-only path: forward through conv2d_bias_act_f16 (linear act,
                    // f32 bias), including the fused epilogue the f32 rows do not have.
                    if pass == "fwd" {
                        let x16 = be.f32_to_f16(x, false).unwrap();
                        let w16 = be.conv_weights_f16(w, g.f, g.c / g.groups * g.k * g.k).unwrap();
                        let bias = be.upload_f32(&[g.f], &vec![0.1; g.f]).unwrap();
                        let op: Op = Box::new(move || {
                            be.conv2d_bias_act_f16(
                                &x16, &w16, &bias, &ys, g.c, g.h, g.w, g.f, g.k, g.stride, g.stride, 1, g.pad,
                                g.groups, F32ActivationKind::Linear, cfg(),
                            )
                            .map(|_| ())
                        });
                        vs.push((modes[i].clone(), be, op));
                    }
                    continue;
                }
                let op: Op = match pass {
                    "fwd" => Box::new(move || {
                        be.conv2d_f32(x, w, &ys, g.c, g.h, g.w, g.f, g.k, g.stride, g.stride, 1, g.pad, g.groups, cfg()).map(|_| ())
                    }),
                    "dgrad" => Box::new(move || {
                        be.conv2d_backward_input_f32(dy, w, &xs, g.c, g.h, g.w, g.f, g.k, g.stride, g.stride, 1, g.pad, g.groups, cfg()).map(|_| ())
                    }),
                    _ => Box::new(move || {
                        be.conv2d_backward_weights_f32(x, dy, &ws, g.c, g.h, g.w, g.f, g.k, g.stride, g.stride, 1, g.pad, g.groups, cfg()).map(|_| ())
                    }),
                };
                vs.push((modes[i].clone(), be, op));
            }
            measure("conv", &format!("conv{li:02}_{pass}"), &params, flops, bytes, &mut vs);
        }
    }
}
