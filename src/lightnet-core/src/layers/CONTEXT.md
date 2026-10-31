# layers

One folder per layer. This file is the single home of the layer layout contract; task state (what is done, what is next) lives in `tasks/rust/`, not here.

## Table of Contents

- [Layout](#layout)
- [Where each piece plugs in](#where-each-piece-plugs-in)
- [Recipe: add or finish a layer](#recipe-add-or-finish-a-layer)
- [Verify](#verify)
- [Rules](#rules)
- [Where to look next](#where-to-look-next)

## Layout

```
src/lightnet-model/src/<layer>/geometry.rs  data-only geometry, validation, weight counts (some layers)
src/lightnet-layers/src/<layer>/cpu.rs   host reference and CPU tests
src/lightnet-runtime/src/layers/<layer>/
  mod.rs   re-exports the CPU items from lightnet-layers and declares `gpu`
  gpu.rs   GpuExecutableNetwork methods: forward_<layer>_gpu, forward_<layer>_compact_gpu,
           backward_<layer>_gpu, update (present once the layer's arms are moved)
src/lightnet-cuda/src/host_ops.rs      host math shared with the backend trait defaults
src/lightnet-cuda-oxide/src/kernels/   device kernels, one file per layer as they are split
src/lightnet-cuda-oxide/tests/oxide_<layer>.rs   CPU-versus-GPU parity for that layer
```

CPU reference and GPU glue no longer share a folder: `cpu.rs` is in `lightnet-layers`, `gpu.rs` is in `lightnet-runtime`, because the glue needs `GpuExecutableNetwork` and the CPU math must stay below it (production graph `model <- layers <- runtime <- train <- core`). This replaces the old one-folder co-location rule.

Exception: `clifford/cpu.rs` stays in `lightnet-runtime/src/layers/clifford/` because it holds the executor-side Clifford code (`CliffordRuntimeCache`, train buffers); its pure data is in `lightnet-layers/src/clifford_params.rs`.

The runtime `layers/` tree is `#[doc(hidden)] pub`; `lightnet-core` re-exports it, so `lightnet_core::...` paths are unchanged. Layer tests that need both the executor and the CPU reference (`carpn/tests.rs`, `gru/tests.rs`, `laya_scorer/wiring_tests.rs`) stay in this folder, `lightnet-core/src/layers/<layer>/`, and are wired by `lightnet-core/src/layer_tests.rs`. The runtime test tree is `lightnet-core/src/runtime_tests/`.

`lightnet-cuda-oxide` depends on `lightnet-core`, so device kernels cannot sit beside the CPU code. It is not a workspace member and builds only through `cargo xtask oxide`.

## Where each piece plugs in

| Piece | File | Wiring |
|---|---|---|
| Module | `lightnet-runtime/src/layers/mod.rs` and `lightnet-runtime/src/lib.rs` | `pub mod <layer>;` here, `pub use layers::<layer>;` at the crate root |
| Old path | `lightnet-core/src/lib.rs` | `pub use lightnet_train::*;` keeps every `lightnet_core::...` path; nothing to add per layer |
| Full forward | `lightnet-runtime/src/runtime/gpu_forward.rs` | match arm calls `self.forward_<layer>_gpu(..)` |
| Compact (f16) forward | `runtime/gpu_forward_compact.rs` | arm calls `forward_<layer>_compact_gpu`; unsupported layers return the "compact f16 GPU execution is unsupported" error |
| Backward | `runtime/gpu_backward.rs` | arm calls `backward_<layer>_gpu` |
| Device cache | `runtime/gpu_types.rs`, `gpu_state.rs`, `gpu_forward.rs` | a `<layer>_caches` field, initialised in every constructor |
| Backend op | `lightnet-cuda/src/lib.rs` (trait with host default), `lightnet-cuda-oxide/src/lib.rs` (device method) | one method per new kernel |
| Kernel | `lightnet-cuda-oxide/src/kernels/<layer>.rs` | `pub mod <layer>;` in `kernels/mod.rs` |
| Parity test | `lightnet-cuda-oxide/tests/oxide_<layer>.rs` | compares device forward, backward and update with the CPU result |
| Layer map row | `tasks/rust/02_state/layer-map.md` | add the row when a directory or file appears |

## Recipe: add or finish a layer

1. Read the layer's row in `tasks/rust/02_state/layer-support.md` and its `cpu.rs`. The device arm must be the exact derivative of that forward.
2. Write `gpu.rs` with `forward_<layer>_gpu`, `backward_<layer>_gpu` and, only if the layer is on the f16 inference list, `forward_<layer>_compact_gpu`. Follow `dsconv/gpu.rs` or `deform_conv/gpu.rs` for style.
3. Add each missing kernel to `kernels/<layer>.rs`, with a backend method and a host default in the trait.
4. Wire the arms and the cache field as in the table above.
5. Add `tests/oxide_<layer>.rs` with a stated tolerance.
6. Update `layer-map.md`, `layer-support.md`, then the version docs named in `tasks/rust/AGENTS.md` (Versioning).

## Verify

One command at a time, `JOBS=4 MEM=4G`:

```
cargo xtask check
cargo xtask gpu-check
cargo xtask test <layer>
cargo xtask oxide --test oxide_<layer>
```

`cargo xtask oxide` rewrites `src/lightnet-cuda-oxide/Cargo.lock`; revert it before committing.

## Rules

- Keep every old path working through re-exports.
- Expose cross-crate implementation items only as needed with `pub` and `#[doc(hidden)]`.
- Backward is the exact derivative of the Rust forward. C++ deviations only under `GradMode::Cpp`.
- Unknown config is an error, never silent identity.
- A native primitive or forward alone does not complete a layer; see the Layer Completion Contract in `tasks/rust/AGENTS.md`.

## Where to look next

- Per-layer files and tests: `tasks/rust/02_state/layer-map.md`.
- Per-layer CPU/GPU/f16/axis status: `tasks/rust/02_state/layer-support.md`.
- What is open: `tasks/rust/AGENTS.md` (Layout Move Queue, Open Work).
