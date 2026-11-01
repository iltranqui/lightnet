# 02_gpu-ops — a GPU kernel for every layer type

One job: every layer type the model uses has a tested GPU kernel.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<model>/01_cfg-cpu.md` (layer inventory and shapes).
- Reference (every run): `_shared/rules.md`, `kernels/CONTEXT.md`, `workflows/add-kernel/CONTEXT.md`.

Do NOT load: `crates/hip_darknet/src/gpu.rs` (that is stage 03), the weights.

## Process

1. Map each layer type to an existing kernel in `kernels/src/yolo/` (conv, maxpool, upsample, concat; route groups are zero-copy views).
2. For each missing one, run the `add-kernel` workflow (`workflows/add-kernel/runs/<op>/`), with the model's real layer shapes in its spec.
3. For an existing kernel with new shapes or params, add those shapes to `apps/runner/tests/yolo_ops.rs` and re-run.

## Outputs

- Kernels and tests (through `add-kernel` runs).
- `runs/<model>/02_gpu-ops.md`: layer type → kernel → test → add-kernel run link; `test result` lines.

## Human check

Every layer type in the inventory has a row with a passing test.
