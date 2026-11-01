# 03_executor — the whole network on the GPU

One job: `hip_darknet --gpu` produces the same layer outputs and detections as the CPU path.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<model>/01_cfg-cpu.md`, `runs/<model>/02_gpu-ops.md`.
- Reference (every run): `_shared/rules.md`, `crates/hip_darknet/CONTEXT.md`.

Do NOT load: kernel internals (`kernels/src/**` bodies), benchmark files.

## Process

1. Extend `crates/hip_darknet/src/gpu.rs` (`GpuModel`) for any new layer type: buffer allocation, launch, route views.
2. Run `cargo test -p hip_darknet --release -- --nocapture`: per-layer parity (every layer, max rel err ≤ 1e-3) and detection parity (GPU == CPU, plus ground truth).
3. Also run `--no-default-features` to keep the CPU-only build working.

## Outputs

- Code in `crates/hip_darknet/src/gpu.rs` (+ tests).
- `runs/<model>/03_executor.md`: per-layer error table (worst layers), detection parity result, ground-truth numbers, `test result` lines.

## Human check

Find the worst layer in the table: is it under 1e-3, and do GPU and CPU detections agree on every image?
