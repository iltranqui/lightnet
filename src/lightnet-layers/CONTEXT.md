# lightnet-layers

CPU reference math for every layer family (layers/*/cpu.rs), LayerParams and the other pure-data parameter types. Depends on lightnet-model only; never names runtime, train or detect.

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
