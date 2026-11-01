# 03_gpu-backward — tile ops and kernels for the backward pass

One job: every operation the trainer needs exists as a tile op plus a thin kernel, tested against the stage-02 CPU reference on the GPU.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/02_cpu-backward.md`.
- Reference (every run): `_shared/rules.md`, `_shared/tile-model.md`, `_shared/darknet-semantics.md`, `crates/hip-tile/CONTEXT.md`, `kernels/CONTEXT.md`, `workflows/add-kernel/CONTEXT.md`.

Do NOT load: `crates/hip_darknet/src/train/reference.rs` internals (only use it as the test oracle), `gpu.rs`.

## Process

1. List the primitives the backward pass needs that `hip-tile` does not have yet (accumulating store, channel-strided reduction, batched addressing, in-place scale/axpy, accumulating GEMM) and add each as an op — not as kernel-local code.
2. For each kernel, run the `add-kernel` workflow (`workflows/add-kernel/runs/<op>/`) with the model's real shapes in its spec. Kernels stay thin bodies over tile ops.
3. Prefer formulations that need no atomics: `dX` as a gather transposed convolution, `dW` with one block per output tile walking the whole batch. If a spec cannot avoid a cross-block accumulation, say so and pick the deterministic alternative.
4. Add batch support to the forward kernels without changing the batch-1 path, and prove it by re-running the existing inference parity suites unchanged.
5. Parity per op/kernel: GPU vs the f32 CPU reference, plus finite differences for the backward ones at the same tolerance as stage 02.

## Outputs

- Ops in `crates/hip-tile/src/ops/`, kernels in `kernels/src/`, tests in `apps/runner/tests/`, one `add-kernel` run per op.
- `runs/<name>/03_gpu-backward.md`: op/kernel → spec run → parity error → FD error table, `test result` lines.

## Human check

Every row's parity error is inside the stated tolerance, the batch-1 inference suites still pass, and no kernel contains a hand-rolled reduction that should have been a tile op.
