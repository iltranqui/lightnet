# lightnet-cuda

Backend-neutral CUDA tensor contracts. Builds without a CUDA driver.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

The seam between `lightnet-core` and concrete GPU backends. Defines operation traits and tensor metadata; concrete device code lives in `lightnet-cuda-cutile` and `lightnet-cuda-oxide`.

## Responsibilities

- Traits such as `F32TensorOpsBackend` and tensor metadata for the executor.
- Host fallback ops (`host_ops/`, for example `bilinear_gather`) used while a device kernel is pending.
- Shared activation math (`laya_act.rs`).

## Dependencies

- Depends on: nothing in this repo.
- Depended on by: `lightnet-core`, `lightnet-cuda-cutile`, `lightnet-cuda-oxide`, `lightnet-cli`.

## Constraints

- Must stay buildable on CPU-only machines.
- A host-fallback layer still needs a CPU-parity test before it counts as GPU-complete (`tasks/rust/gpu_layers_plan.md`).
- Open question: keep exact-erf math here and in `lightnet-core` in sync (both use `float_erf`).
