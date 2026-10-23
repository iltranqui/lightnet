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

- One module per layer or feature in `src/` (for example `deform_conv.rs`, `dcnv4.rs`, `clifford.rs`, `carpn.rs`, `depth_anything.rs`), plus `cfg.rs`, `image.rs`, `detect.rs`, `evaluate.rs`, `augmentation.rs`.
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
