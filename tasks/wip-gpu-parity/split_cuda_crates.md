# Split `lightnet-cuda` into per-layer crates

Plan only. Nothing here is implemented, built or tested. Goal: modular crates and faster incremental builds. One crate per layer group (`lightnet-cuda-conv`, `lightnet-cuda-train`, ...), each with its own `Compiling` line, with `lightnet-cuda` compiled last as a thin re-export.

## Table of Contents

- [Problem](#problem)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
- [Design](#design)
- [Implementation plan](#implementation-plan)
- [Verification](#verification)
- [Open questions](#open-questions)

## Problem

`src/lightnet-cuda/src/lib.rs` is about 6.5k lines. About 5.3k of them are one trait, `F32TensorOpsBackend`, with 187 methods (conv2d, vit, laya, deform, wmhf, swin, ...). Any edit recompiles the whole crate. Rust cannot split one trait across crates, so the split must use supertraits.

Method groups by name prefix (top): vit 15, laya 15, deform 13, wmhf 12, swin 8, hyperbolic 7, shortcut/moe/mambavision/graph 6 each, conv2d 6, maxpool/local/foveation 4 each.

## Project tree

```
src/
├── lightnet-cuda-base/        NEW  enums, descriptors, TensorStore trait, CudaBackend
├── lightnet-cuda-conv/        NEW  pilot: F32ConvOps (conv2d x6)
├── lightnet-cuda-train/       NEW  second: optimizer / update / loss ops
├── lightnet-cuda-<group>/     NEW  one per remaining group (vit, laya, deform, ...)
├── lightnet-cuda/                  THIN: re-exports; keeps F32TensorOpsBackend (shrinking)
│   ├── src/lib.rs                  shrinks from 6.5k lines
│   └── src/host_ops/               moves into the group crate that uses each file
├── lightnet-core/                  imports unchanged (via lightnet-cuda)
├── lightnet-cuda-cutile/           impl blocks split per group trait
├── lightnet-cuda-oxide/            impl blocks split per group trait (not a workspace member)
└── lightnet-cli/                   imports unchanged
```

## Component breakdown

### lightnet-cuda-base

- Role: bottom of the graph; depends on nothing in this repo.
- Contents: `ScalarType`, `TensorLayout`, `KernelLanguage`, `ConvPrecision`, `DeviceTensorDesc`, `ScalarArg`, `ElementwiseLaunchConfig`, `CudaBackend`, and a new `TensorStore` trait (`type Error`, `type Tensor`, `upload_f32`, `zeros_f32`, `download_f32`, `tensor_shape`).
- Constraint: buildable without a CUDA driver.

### lightnet-cuda-conv (pilot)

- Role: `F32ConvOps: TensorStore` with the 6 `conv2d_*` methods. The 3 `_with_precision` methods keep their default bodies.
- Depends on: `lightnet-cuda-base` only. Checked: the 6 methods use only `Self::Tensor`, `Self::Error`, `ElementwiseLaunchConfig`, `ConvPrecision`, so no host fallback moves with them.
- Used by: `lightnet-cuda`.

### lightnet-cuda-train

- Role: `F32TrainOps` for SGD/Adam/Muon updates, batch-norm train, dropout.
- Open: needs `host_ops/muon.rs`, `AdamParams`, `NanInfMode`. Confirm they have no dependency on other groups before moving.

### lightnet-cuda (thin)

- Re-exports every group crate and base. `F32TensorOpsBackend` gains each group trait as a supertrait, so existing generic bounds in `lightnet-core` keep compiling.
- Keeps the not-yet-split methods until their turn.

### Backends: lightnet-cuda-cutile, lightnet-cuda-oxide

- Today: one `impl F32TensorOpsBackend for X` block holds every method.
- After: the block splits into `impl TensorStore`, `impl F32ConvOps`, and the rest. This is a mechanical edit but it is a downstream change. My earlier claim of "no downstream impl changes" was wrong; only callers using `F32TensorOpsBackend` bounds stay unchanged.

## Design

```
base:  TensorStore { Error, Tensor, upload/download/zeros }
conv:  trait F32ConvOps: TensorStore { conv2d_* }
train: trait F32TrainOps: TensorStore { ... }
cuda:  trait F32TensorOpsBackend: TensorStore + F32ConvOps + F32TrainOps { legacy methods }
```

Each step is independently shippable. Call sites need the trait in scope; `lightnet-cuda` re-exports it, and the supertrait bound makes the methods callable through `F32TensorOpsBackend` generics.

## Implementation plan

Branch for the work: `wip-gpu-parity`. At plan time `src/lightnet-cuda/src/lib.rs` was being edited by another process in the main checkout (uncommitted, 6551 to 6665 lines). This plan was written in a worktree at commit `f003801`, so line numbers may differ.

1. [ ] Settle the in-flight `lib.rs` edits; start from a clean `src/lightnet-cuda`.
2. [ ] Create `lightnet-cuda-base`: move the enums, descriptors, `CudaBackend`, `ElementwiseLaunchConfig`. Add `TensorStore`. Re-export from `lightnet-cuda`. Add to workspace `members`. `cargo xtask check`.
3. [ ] Make `F32TensorOpsBackend: TensorStore`; move the base methods and associated types. Split the backend impls (cutile, oxide). `cargo xtask check` and `gpu-check`.
4. [ ] Create `lightnet-cuda-conv` with `F32ConvOps`. Move 6 methods. Add as supertrait. Split backend impls. `check` and `gpu-check`.
5. [ ] Measure: touch the conv file, confirm only `lightnet-cuda-conv` and dependents recompile. Record timings.
6. [ ] Repeat for `train`.
7. [ ] Repeat per group in order of size (vit, laya, deform, wmhf, ...), one `check` after each.
8. [ ] When the legacy trait is empty, drop it or make it a pure combination trait.
9. [ ] Update `src/lightnet-cuda/CONTEXT.md`, the root `CLAUDE.md` crate table, and add `CONTEXT.md` to each new crate.
10. [ ] Commit per group with only the files that group touches.

## Verification

- `cargo xtask check` and `cargo xtask gpu-check` after every group (4 threads, 3G cap).
- `cargo xtask test` before and after step 4; results must match.
- `cargo xtask oxide` after steps 3 and 4 if a GPU is available.
- Incremental-build proof in step 5.

## Open questions

- Granularity: one crate per group (about 40) or merge small ones (for example all pooling ops)? 40 crates add workspace overhead.
- `lightnet-cuda-oxide` and `lightnet-cuda-cutile` are excluded from the workspace: confirm they can path-depend on the new crates without a lockfile problem.
