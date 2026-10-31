# lightnet-core

Facade over the Darknet/YOLO crates. It holds no production code: `pub use lightnet_train::*` re-exports the whole stack so every old `lightnet_core::...` path keeps working.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

The stable public name of the port. The code lives in four crates, each depending only on the ones before it:

```
lightnet-model    .cfg parsing, Network, TensorShape/Tensor, error, names, shared geometry and label types
lightnet-layers   CPU reference math for each layer family (layers/*/cpu.rs, LayerParams data)
lightnet-runtime  ExecutableNetwork, GpuExecutableNetwork, layers/*/gpu.rs glue, weights, initialize,
                  Muon state, ONNX export, image and loader_profile
lightnet-train    losses, augmentation, detect, evaluate, obb_*, cuboid_labels, yolo_3d, chart
lightnet-core     facade (this crate) + cross-crate unit tests
```

All four sit on `lightnet-cuda`. Per-layer wiring is in `src/layers/CONTEXT.md`.

## Responsibilities

- `src/lib.rs`: the re-export. Nothing else belongs in it.
- Unit tests that need items from several of the crates: `src/runtime_tests/` (executor against training, detection and dense loss), `src/model_tests.rs`, and the layer tests wired by `src/layer_tests.rs`. They cannot live lower down without a dependency cycle.
- Integration tests in `tests/` and their fixtures (`tests/fixtures`, `tests/data`), which the runtime and train unit tests also read.

## Dependencies

- Depends on: `lightnet-model`, `lightnet-layers`, `lightnet-runtime`, `lightnet-train`, `lightnet-cuda`.
- Depended on by: `lightnet-cli`, `lightnet-examples`, `lightnet-gui*`, `lightnet-lm`, `lightnet-cuda-oxide` (dev, separate workspace).

## Constraints

- Production graph is acyclic: `model <- layers <- runtime <- train <- core`. `tools/check_module_layers.sh` guards it; run it after moving code.
- Items that were crate-private before the split are `pub` plus `#[doc(hidden)]` in the owning crate when a sibling crate or these tests need them. Inventory: `tasks/wip-gpu-parity/crate-split-plan.md`.
- Unknown config is an error, never silent identity.
- Needs pinned nightly (`#![feature(float_erf)]` is declared in `lightnet-layers`, see `rust-toolchain.toml`).
- No C++ parity claim without a compiled-C++ golden. Per-layer status: `tasks/rust/02_state/layer-support.md`.
- Axis and layout semantics must be pinned to the C++ source by a named test.
