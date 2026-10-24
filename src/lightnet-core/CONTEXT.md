# lightnet-core

CPU-first Rust implementation of the Darknet/YOLO runtime. Largest crate; holds every layer.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Owns Darknet's observable contracts: `.cfg` parsing, planar `B x C x H x W` tensors, layer ordering, `.weights` byte ordering, detector target assignment, and optimizer update semantics (SGD, Adam, Muon).

## Responsibilities

- One folder per layer under `src/layers/<layer>/` (for example `deform_conv`, `dcnv4`, `clifford`, `carpn`, `depth_anything`, `vit`): `cpu.rs` is the CPU reference, `gpu.rs` is the runtime glue that drives the Oxide/CUDA backend (present for `dcnv4`, `hyperbolic_foveation`, `laya_act_head`, `laya_encoder`, `laya_head`, `laya_scorer`, `vit`), and `mod.rs` re-exports both. `lib.rs` re-exports each layer at its old path (`crate::vit`, `lightnet_core::clifford`), so callers did not change. `swin` also holds `block.rs`. GPU kernels stay in `lightnet-cuda-oxide`, which cannot host CPU code because it depends on this crate. Per-layer arms inlined in the large `runtime/gpu_forward.rs`, `gpu_backward.rs` and `exec_*.rs` dispatchers have not been extracted. Other modules in `src/`: `cfg.rs`, `image.rs`, `detect.rs`, `evaluate.rs`, `augmentation.rs`.
- Forward and exact-derivative backward for each layer; C++ deviations only under `GradMode::Cpp`.
- Dispatch to GPU backends through the traits in `lightnet-cuda`.

## Dependencies

- Depends on: `lightnet-cuda`.
- Depended on by: `lightnet-cli`, `lightnet-examples`, `lightnet-cuda-oxide`.

## Constraints

- Unknown config is an error, never silent identity.
- Needs pinned nightly (`#![feature(float_erf)]`, see `rust-toolchain.toml`).
- No C++ parity claim without a compiled-C++ golden. Per-layer status: `tasks/rust/layer-support.md`.
- Axis and layout semantics must be pinned to the C++ source by a named test.
