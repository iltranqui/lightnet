//! GEMM micro-benchmark, interleaved A/B over modes (laptop clocks drift).
//! Run: `... oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 1 -- --ignored --nocapture gemm_bench`
//! Env: DARKNET_BENCH_MODES=fp32,tf32 (default), DARKNET_BENCH_ROUNDS=5.

use darknet_cuda::F32TensorOpsBackend;
use darknet_cuda_oxide::{GemmMode, OxideBackend};

/// (m, n, k) = (filters, out_h*out_w, in_c/groups*size^2) of the 21 LegoGears_v2 convs.
const LEGO: [(usize, usize, usize); 21] = [
    (32, 8960, 27), (64, 2240, 288), (64, 2240, 576), (32, 2240, 288), (32, 2240, 288), (64, 2240, 64),
    (128, 560, 576), (64, 560, 576), (64, 560, 576), (128, 560, 128),
    (256, 140, 1152), (128, 140, 1152), (128, 140, 1152), (256, 140, 256),
    (512, 35, 2304), (256, 35, 512), (512, 35, 2304), (30, 35, 512),
    (128, 35, 256), (256, 140, 3456), (30, 140, 256),
];

fn modes() -> Vec<(String, GemmMode)> {
    let s = std::env::var("DARKNET_BENCH_MODES").unwrap_or_else(|_| "fp32,tf32".into());
    s.split(',')
        .map(|m| {
            let mode = match m {
                "tf32" => GemmMode::Tf32,
                "auto" => GemmMode::Auto,
                "fp16" | "f16" | "half" => GemmMode::Fp16,
                "fp8" | "e4m3" => GemmMode::Fp8,
                "fp4" | "nvfp4" => GemmMode::Fp4,
                _ => GemmMode::Fp32,
            };
            (m.to_string(), mode)
        })
        .collect()
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run(shapes: &[(usize, usize, usize)], label: &str) {
    let rounds: usize = std::env::var("DARKNET_BENCH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let ms = modes();
    let backends: Vec<OxideBackend> =
        ms.iter().map(|(_, m)| OxideBackend::new().unwrap().with_gemm_mode(*m)).collect();
    let mut tot = vec![0.0f64; ms.len()];
    for &(m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0).collect();
        let ts: Vec<_> = backends
            .iter()
            .map(|be| (be.upload_f32(&[m, k], &a).unwrap(), be.upload_f32(&[k, n], &b).unwrap()))
            .collect();
        let iters = ((2e9 / (2.0 * (m * n * k) as f64)) as usize).clamp(20, 400);
        let mut res: Vec<Vec<f64>> = vec![Vec::new(); ms.len()];
        for _ in 0..rounds {
            for (i, be) in backends.iter().enumerate() {
                res[i].push(be.bench_gemm_f32(&ts[i].0, &ts[i].1, m, n, k, iters).unwrap());
            }
        }
        let flops = 2.0 * (m * n * k) as f64;
        let mut line = format!("BENCH {label} m{m} n{n} k{k}:");
        for (i, (name, _)) in ms.iter().enumerate() {
            let med = median(&mut res[i]);
            tot[i] += med;
            line += &format!(" {name} {:.4} ms {:.0} GF/s |", med, flops / med / 1e6);
        }
        println!("{line}");
    }
    let mut line = format!("BENCH {label} TOTAL:");
    for (i, (name, _)) in ms.iter().enumerate() {
        line += &format!(" {name} {:.4} ms |", tot[i]);
    }
    println!("{line}");
}

#[test]
#[ignore]
fn gemm_bench_square() {
    run(&[(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)], "sq");
}

#[test]
#[ignore]
fn gemm_bench_lego() {
    run(&LEGO, "lego");
}

/// `tasks/rust/fp8_fp4.md` Phase D: the f32 / tf32 / f16 / fp8 / fp4 table.
/// End-to-end through the GEMM seam, so the fp8 and fp4 columns include their
/// quantisation (the FP4 column additionally includes the host transpose and
/// host round trip of the packed path - it is an emulation, not a kernel).
#[test]
#[ignore]
fn gemm_bench_precisions() {
    run(&[(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)], "prec-sq");
    run(&LEGO, "prec-lego");
}

/// Kernel-only numbers for the low-precision paths: the operands are
/// quantised once up front, so the FP8 column is the tensor-core `mma` alone
/// and the FP4 column is the `__dp4a` loop alone.  These are the numbers to
/// compare against each other; `gemm_bench_precisions` is what a network sees.
#[test]
#[ignore]
fn gemm_bench_low_precision_kernels() {
    use darknet_cuda_oxide::fp8::{f32_to_fp8, Fp8Format};
    let rounds: usize = std::env::var("DARKNET_BENCH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    for &(m, n, k) in &[(512usize, 512usize, 512usize), (1024, 1024, 1024), (2048, 2048, 2048)] {
        let be = OxideBackend::new().unwrap();
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32 - 5.0) / 8.0).collect();
        let aq: Vec<u8> = a.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
        let bq: Vec<u8> = b.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
        let flops = 2.0 * (m * n * k) as f64;

        let ta = be.upload_f32(&[m, k], &a).unwrap();
        let tb = be.upload_f32(&[k, n], &b).unwrap();
        let iters = ((2e9 / flops) as usize).clamp(20, 400);
        let mut fp32 = Vec::new();
        for _ in 0..rounds {
            fp32.push(be.bench_gemm_f32(&ta, &tb, m, n, k, iters).unwrap());
        }
        // Pre-quantised FP8 operands; the upload happens outside the timed loop.
        let mut dev8_ms = Vec::new();
        for _ in 0..rounds {
            let (aq_dev, bq_dev) = (aq.clone(), bq.clone());
            dev8_ms.push(be.bench_gemm_fp8_host(&aq_dev, &bq_dev, m, n, k, iters, Fp8Format::E4M3, Fp8Format::E4M3).unwrap());
        }
        let fp8_med = median(&mut dev8_ms);

        // Packed NVFP4, pre-quantised.
        let pa4 = darknet_cuda_oxide::fp4::quantize_1x16(&a, m, k);
        let pb4 = darknet_cuda_oxide::fp4::quantize_1x16(&b, n, k);
        let mut fp4_ms = Vec::new();
        for _ in 0..rounds {
            fp4_ms.push(be.bench_gemm_fp4(&pa4, &pb4, m, n, k, iters).unwrap());
        }
        let fp4_med = median(&mut fp4_ms);
        let fp32_med = median(&mut fp32);
        println!(
            "KERNELBENCH m{m} n{n} k{k}: fp32 {:.4} ms {:.0} GF/s | fp8-mma {:.4} ms {:.0} GF/s | fp4-dp4a {:.4} ms {:.0} GF/s",
            fp32_med,
            flops / fp32_med / 1e6,
            fp8_med,
            flops / fp8_med / 1e6,
            fp4_med,
            flops / fp4_med / 1e6
        );
    }
}
