# lightnet-train

Training and post-processing: YOLO and dense-head losses, augmentation and batch loading, detection decode, evaluation (mAP), OBB and cuboid labels, yolo_3d, segmentation data, training chart. Re-exports lightnet-runtime.

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
