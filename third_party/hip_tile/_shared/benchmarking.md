# Benchmarking

How to measure performance on this machine without fooling yourself. Every performance number in a run file, the README or a commit message follows this method.

## Contents

- [Why numbers lie here](#why-numbers-lie-here)
- [Method](#method)
- [Reference numbers](#reference-numbers)

## Why numbers lie here

The Radeon 680M is an iGPU: its clock follows a power budget shared with the CPU. With a concurrent `cargo build`, the same binary measured tile gemm at 1285 vs 2118 GFLOP/s and rocBLAS at 1538 vs 2706. Results are bimodal, not just noisy. An early "tile gemm beats rocBLAS at N=1024" claim was a clock artifact.

## Method

- Keep the CPU idle: no builds or agents running.
- Warm up properly (at least 10 launches), then report the **best of several repetitions** (runner: 10 warm-up, best of 5×20; `hip_darknet --bench`: best of 5×N on GPU, 3×20 on CPU).
- Compare two kernels only within the same process run, back to back.
- Time with HIP events on the stream; report ms and GFLOP/s (or ms/frame).
- Run the measurement at least twice; if the runs disagree by more than ~10%, find out why before reporting.

## Reference numbers

Measured 2026-10-25, CPU idle, pure-Rust barrier (see the run files for details):

| Measurement | Result |
|---|---|
| Tile gemm 128×128×16, N=1024 | 2.05 TFLOP/s (rocBLAS 2.69) |
| Tile gemm 128×128×16, N=2048 | 2.28 TFLOP/s (rocBLAS 1.91) |
| YOLO conv, 56×40 maps / 7×5 maps | 850–970 / 250–320 GFLOP/s |
| LegoGears YOLOv4-tiny, GPU forward / CPU forward | 4.78 / 459.84 ms per frame (2026-10-26, default-features build; a `--no-default-features` binary has no GPU path at all) |

Commands: `cargo run -p runner --release -- gemm --n N --bench`, `cargo run -p runner --release -- conv-bench`, `cargo run -p hip_darknet --release -- --gpu --bench 200 ...`.
