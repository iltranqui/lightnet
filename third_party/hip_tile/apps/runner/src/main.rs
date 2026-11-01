use std::process::ExitCode;

use hip_rt::{DeviceBuffer, Event};
use kernels::saxpy::saxpy;

const USAGE: &str = "usage: runner saxpy [--n N] | runner gemm [--n N] [--bench] | runner conv-bench";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("saxpy") => parse_n(&args[1..]).and_then(run_saxpy),
        Some("gemm") => parse_gemm(&args[1..]).and_then(|(n, bench)| run_gemm(n, bench)),
        Some("conv-bench") => conv_bench().map(|()| true),
        _ => Err(USAGE.to_owned()),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}

fn parse_n(args: &[String]) -> Result<usize, String> {
    match args {
        [] => Ok(1 << 20),
        [flag, n] if flag == "--n" => n.parse().map_err(|e| format!("bad --n `{n}`: {e}")),
        _ => Err(USAGE.to_owned()),
    }
}

/// Returns whether the result matched the CPU reference.
fn run_saxpy(n: usize) -> Result<bool, String> {
    let e = |e: hip_rt::HipError| e.to_string();
    let (device, module) = runner::load().map_err(e)?;
    println!("device: {} ({})", device.name(), device.arch());

    let a = 2.5f32;
    let x: Vec<f32> = (0..n).map(|i| (i % 97) as f32 * 0.25).collect();
    let y: Vec<f32> = (0..n).map(|i| (i % 31) as f32 - 7.0).collect();
    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| a * x + y).collect();

    let dx = DeviceBuffer::from_slice(&x).map_err(e)?;
    let mut dy = DeviceBuffer::from_slice(&y).map_err(e)?;
    let grid = (n as u32).div_ceil(saxpy::BLOCK_THREADS).max(1);

    let (start, stop) = (Event::new().map_err(e)?, Event::new().map_err(e)?);
    start.record(None).map_err(e)?;
    saxpy::launch(
        &module,
        None,
        grid,
        a,
        dx.tensor(1, n as u32),
        dy.tensor_mut(1, n as u32),
    )
    .map_err(e)?;
    stop.record(None).map_err(e)?;
    let ms = stop.elapsed_ms(&start).map_err(e)?;

    let got = dy.copy_to_vec().map_err(e)?;
    let mismatches = got.iter().zip(&expected).filter(|(g, w)| g != w).count();
    println!("n = {n}, mismatches = {mismatches}, time = {ms:.4} ms");
    Ok(mismatches == 0)
}

fn parse_gemm(args: &[String]) -> Result<(usize, bool), String> {
    let (mut n, mut bench) = (1024, false);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bench" => bench = true,
            "--n" => {
                let v = it.next().ok_or_else(|| USAGE.to_owned())?;
                n = v.parse().map_err(|e| format!("bad --n `{v}`: {e}"))?;
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    Ok((n, bench))
}

/// Square `n x n x n` gemm; prints the error against the CPU, and with `bench` the
/// average time and GFLOP/s of the tile kernel next to rocBLAS.
fn run_gemm(n: usize, bench: bool) -> Result<bool, String> {
    let e = |e: hip_rt::HipError| e.to_string();
    let (device, module) = runner::load().map_err(e)?;
    println!("device: {} ({})", device.name(), device.arch());

    let a = runner::random_vec(n * n, 1);
    let b = runner::random_vec(n * n, 2);
    let (da, db) = (
        DeviceBuffer::from_slice(&a).map_err(e)?,
        DeviceBuffer::from_slice(&b).map_err(e)?,
    );
    let mut dc = DeviceBuffer::<f32>::zeroed(n * n).map_err(e)?;
    runner::launch_gemm(&module, &da, &db, &mut dc, (n, n, n)).map_err(e)?;
    let got = dc.copy_to_vec().map_err(e)?;
    let err = runner::max_rel_err_vs_cpu((&a, &b, &got), (n, n, n), 4096);
    println!("n = {n}, max rel err vs f64 CPU = {err:.2e}");

    if bench {
        let flops = 2.0 * (n as f64).powi(3);
        let blas = hip_rt::rocblas::Handle::new().map_err(e)?;
        let (warmup, iters, reps) = (10, 20, 5);
        // The iGPU shares its power budget with the CPU and ramps clocks slowly, so warm
        // up properly and report the best of several repetitions.
        let time =
            |name: &str, run: &mut dyn FnMut() -> hip_rt::Result<()>| -> Result<(), String> {
                for _ in 0..warmup {
                    run().map_err(e)?;
                }
                let (start, stop) = (Event::new().map_err(e)?, Event::new().map_err(e)?);
                let mut best = f64::INFINITY;
                for _ in 0..reps {
                    start.record(None).map_err(e)?;
                    for _ in 0..iters {
                        run().map_err(e)?;
                    }
                    stop.record(None).map_err(e)?;
                    let ms = f64::from(stop.elapsed_ms(&start).map_err(e)?) / f64::from(iters);
                    best = best.min(ms);
                }
                println!("{name:>8}: {best:.4} ms  {:.1} GFLOP/s", flops / best / 1e6);
                Ok(())
            };
        time("tile", &mut || {
            runner::launch_gemm(&module, &da, &db, &mut dc, (n, n, n))
        })?;
        time("rocBLAS", &mut || blas.sgemm(n, n, n, &da, &db, &mut dc))?;
    }
    Ok(err <= 1e-5)
}

/// Time every conv tile config on the heavy YOLOv4-tiny layers (224x160 input).
/// Best of 5 repetitions of 20 launches after 10 warm-up launches.
fn conv_bench() -> Result<(), String> {
    use kernels::yolo::conv::{Config, ConvParams, launch_conv_with};
    let e = |e: hip_rt::HipError| e.to_string();
    let (device, module) = runner::load().map_err(e)?;
    println!("device: {} ({})", device.name(), device.arch());
    // (cin, cout, h, w, k, stride)
    let layers = [
        (3u32, 32u32, 160u32, 224u32, 3u32, 2u32),
        (32, 64, 80, 112, 3, 2),
        (64, 64, 40, 56, 3, 1),
        (32, 32, 40, 56, 3, 1),
        (128, 128, 20, 28, 3, 1),
        (64, 64, 20, 28, 3, 1),
        (256, 256, 10, 14, 3, 1),
        (128, 128, 10, 14, 3, 1),
        (512, 512, 5, 7, 3, 1),
        (256, 512, 5, 7, 3, 1),
        (512, 256, 5, 7, 1, 1),
        (512, 30, 5, 7, 1, 1),
    ];
    let (warmup, iters, reps) = (10, 20, 5);
    for (cin, cout, h, w, k, s) in layers {
        let p = ConvParams::new(cin, h, w, k, s, k / 2, ConvParams::LEAKY);
        let (hw, ohw, kd) = ((h * w) as usize, (p.oh * p.ow) as usize, (cin * k * k) as usize);
        let x = DeviceBuffer::from_slice(&runner::random_vec(cin as usize * hw, 1)).map_err(e)?;
        let wt = DeviceBuffer::from_slice(&runner::random_vec(cout as usize * kd, 2)).map_err(e)?;
        let b = DeviceBuffer::from_slice(&runner::random_vec(cout as usize, 3)).map_err(e)?;
        let mut y = DeviceBuffer::<f32>::zeroed(cout as usize * ohw).map_err(e)?;
        let flops = 2.0 * f64::from(cout) * kd as f64 * ohw as f64;
        println!("conv {cin}->{cout} k{k} s{s} {w}x{h} -> {}x{}  ({:.1} MFLOP)", p.ow, p.oh, flops / 1e6);
        for cfg in Config::ALL {
            let mut run = || {
                launch_conv_with(
                    cfg,
                    &module,
                    None,
                    x.tensor(cin, hw as u32),
                    wt.tensor(cout, kd as u32),
                    hip_tile::Tensor::vector(b.as_ptr(), cout),
                    y.tensor_mut(cout, ohw as u32),
                    &p,
                )
            };
            for _ in 0..warmup {
                run().map_err(e)?;
            }
            let (start, stop) = (Event::new().map_err(e)?, Event::new().map_err(e)?);
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                start.record(None).map_err(e)?;
                for _ in 0..iters {
                    run().map_err(e)?;
                }
                stop.record(None).map_err(e)?;
                best = best.min(f64::from(stop.elapsed_ms(&start).map_err(e)?) / f64::from(iters));
            }
            let auto = if cfg == Config::choose(cout, p.oh * p.ow) { " <- auto" } else { "" };
            println!("  {cfg:?}: {best:.4} ms  {:.1} GFLOP/s{auto}", flops / best / 1e6);
        }
    }
    Ok(())
}
