# lightnet-runtime

ExecutableNetwork and GpuExecutableNetwork, the per-layer GPU glue (layers/*/gpu.rs), .weights I/O, initialisation, Muon state, ONNX export, image loading and loader profiling. Never names train or detect; guarded by tools/check_module_layers.sh.

## Table of Contents

- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Dependencies

- Graph: `lightnet-model` <- `lightnet-layers` <- `lightnet-runtime` <- `lightnet-train` <- `lightnet-core` (facade). Every crate also depends on `lightnet-cuda`.
- Plan and visibility inventory: `tasks/wip-gpu-parity/crate-split-plan.md`.

## Constraints

- Keep the production graph acyclic; run `tools/check_module_layers.sh`.
- Items needed by an upper crate or by the cross-crate tests in `lightnet-core` are `pub` and `#[doc(hidden)]`.
- Tests that need several crates live in `lightnet-core`, not here.
