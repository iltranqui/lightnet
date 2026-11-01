# 04_executor — one GPU training step

One job: a `TrainModel` that runs forward → loss/deltas → backward → SGD update for a whole batch on the GPU, agreeing with the stage-02 CPU trainer.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/02_cpu-backward.md`, `runs/<name>/03_gpu-backward.md`.
- Reference (every run): `_shared/rules.md`, `crates/hip_darknet/CONTEXT.md`, `crates/hip_darknet/src/gpu.rs` (the inference executor this one mirrors).

Do NOT load: kernel bodies, benchmark numbers.

## Process

1. Add `src/train/gpu.rs`: buffers per activation and per gradient sized for the batch, resident weights + momentum, BN stats/running buffers, all f32.
2. One stream sync per iteration; weights, momentum and BN stats are updated on the device (no round-trip through the host per step).
3. Parity: GPU training forward == CPU training forward per layer; GPU per-layer gradients == CPU reference gradients; whole-net loss for a fixed seeded batch matches the CPU number.
4. Timing probe of ~200 iterations to size the real run before committing to it.

## Outputs

- `crates/hip_darknet/src/train/gpu.rs` (+ tests).
- `runs/<name>/04_executor.md`: parity table, loss comparison, ms/iteration probe, `test result` lines.

## Human check

Does the gradient parity table hold at the stated tolerance for every layer, and does the 200-iteration probe give a wall clock you would accept for the full run?
