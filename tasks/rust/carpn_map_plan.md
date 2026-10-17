# CARPN Polygon mAP Dispatch Plan

## Table of Contents

- [Purpose and boundary](#purpose-and-boundary)
- [Current evidence](#current-evidence)
- [Scope](#scope)
- [Compatibility decisions](#compatibility-decisions)
- [Implementation tasks](#implementation-tasks)
- [Verification gates](#verification-gates)
- [Explicitly deferred](#explicitly-deferred)
- [Completion criteria](#completion-criteria)

## Purpose and boundary

Enable the existing public OBB mAP entry point to score caller-provided CARPN quadrilateral detections against OBB labels, using the already-implemented CARPN polygon IoU helper. The current explicit rejection is in `src/darknet-core/src/evaluate.rs::obb_detector_map` (`CARPN polygon mAP is not implemented for OBB YOLO labels`).

This plan is for evaluator dispatch on sample records. It does **not** claim end-to-end inference or CLI/training mAP for a CARPN network: Rust currently has no `obb_refine`/stage-2 model layer or CARPN network decoder, and `decode_obb_yolo_detections` emits only legacy front-point detections.

## Current evidence

- `carpn_detector_map` already ranks normalized `CarpnQuad` detections, reconstructs oriented truth quads from front-point OBB labels, checks required geometry/labels, and shares AP/matching/report generation with the standard evaluator.
- `carpn_polygon_iou` already computes convex quadrilateral intersection/union and has tests for both windings, partial/contained/disjoint overlaps, and invalid geometry.
- Existing `evaluate.rs` tests already cover CARPN AP and duplicate detections, rotated truth versus AABB overlap, missing geometry/front-label rejection, and the current `obb_detector_map` rejection.
- `ObbValidationDataset::evaluate`, `detector map`, and training `-map` all use `decode_obb_yolo_detections` and therefore cannot currently produce CARPN quads. Keep this a separately blocked inference/runtime integration item.
- Reference source `raptor:src-lib/detector_map_obb.cpp` identifies CARPN detections as 8 points with a mask, rebuilds truth quads from front-point labels, and uses convex polygon IoU. No compiled C++ oracle is available; any source comparison is hand-derived only.

## Scope

In scope:

- Dispatch uniformly typed caller-provided OBB samples through `obb_detector_map`: legacy front-point geometry continues through its existing OBB IoU path; CARPN quad geometry delegates to `carpn_detector_map`.
- Reject mixed `FrontPoint` and `CarpnQuad` predictions instead of silently comparing incompatible shapes or falling back to AABB/legacy OBB IoU.
- Preserve current class-count, score, IoU-threshold, duplicate matching, AP, report-counter, and normalized-coordinate behavior.
- Add focused unit coverage at the `obb_detector_map` public boundary.

Out of scope:

- Implementing CARPN `[obb_refine]` parsing, stage-2 forward/runtime, proposal generation, or network decode.
- Claiming that standalone dispatcher tests prove `detector map` or training `-map` can evaluate a CARPN model.
- Training, GPU support, dataset-dependent numeric parity, or compiled-C++ parity.

## Compatibility decisions

1. If at least one prediction is CARPN, all predictions across all samples must be CARPN; delegate to `carpn_detector_map`, which validates front-point labels and other inputs.
2. If there are no CARPN predictions, preserve the existing front-point OBB route. A fully empty prediction set carries no representation tag; callers that need explicit CARPN validation (including missing-front checks) should call `carpn_detector_map` directly. Document this ambiguity rather than infer mode from the network or silently change empty-set behavior.
3. Mixed prediction geometry is an input error. Do not apply legacy OBB IoU to CARPN vertices or AABB fallback.
4. Preserve empty image samples, thresholds, AP integration, and detection counters by delegating to the existing shared evaluator rather than duplicating matching logic.
5. Treat this as Rust-runtime/API behavior. Do not claim C++ parity without a compiled reference executable and matching fixture.

## Implementation tasks

- [x] 1. Add failing `evaluate.rs` tests through `obb_detector_map`: all-CARPN predictions produce the same `MapReport` fields as direct `carpn_detector_map`; include duplicate prediction/false-positive behavior and a rotated truth whose AABB-only overlap must not pass the polygon threshold.
- [x] 2. Add mixed-geometry rejection tests, including CARPN predictions split across separate image samples and mixed predictions within one sample. Require a clear invalid-input error.
- [x] 3. Preserve and test legacy front-point behavior and the no-prediction ambiguity. Keep explicit `carpn_detector_map` tests for empty CARPN predictions and missing-front labels if its current behavior is intended.
- [x] 4. Implement only the `obb_detector_map` dispatch/validation needed to satisfy the tests; reuse `carpn_detector_map` and `detector_map_by` rather than copy matching/AP logic.
- [x] 5. Run `MEM=3G ./tools/test_host.sh obb_map`, `MEM=3G ./tools/test_host.sh carpn_map`, and `MEM=3G ./tools/test_host.sh carpn_polygon_iou` separately; never run build/test commands in parallel. Results: 5 core tests passed for `obb_map`, 3 for direct CARPN mAP, and 2 for polygon IoU.
- [x] 6. Review `git diff` for accidental changes to `ObbValidationDataset`, CLI routing, or training `-map`; those changes would wrongly imply network-level support in this task.
- [x] 7. Record implementation evidence in the ledgers and task docs, with task-named code commit `7681f478` (`v0.0.258`). `version-map.tsv`, `implemented.md`, `todo.md`, and `STATUS.md` reflect the evidence; unrelated user edits are not included.

## Verification gates

Required for this bounded task:

- The existing direct `carpn_detector_map` and `carpn_polygon_iou` tests remain green.
- New dispatcher tests prove all-CARPN polygon matching, strict mixed-geometry rejection, and unchanged legacy front-point results.
- The focused host test command passes; no GPU or training command is needed.
- No numeric compiled-C++ parity claim is made. Source-derived behavior must be labelled as hand-derived from source.

Not a pass for this task:

- A fake callback that fabricates CARPN detections from a legacy network output.
- A passing helper-only unit test described as real model/CLI CARPN mAP.
- A full workspace/cfg-corpus run; it is not required and known resource-limited failures remain documented.

## Explicitly deferred

End-to-end CARPN inference and direct `detector map` are implemented separately in [`carpn_network_map_plan.md`](carpn_network_map_plan.md) (v0.0.259): CPU `[obb_refine]` inference, quadrilateral decoding, `detect-image`, full-forward validation, and generated cfg/weights-backed polygon-mAP fixtures. Rust CARPN training and scheduled `detector train -map` remain deferred until Stage-2 assignment/loss/backward/update support is implemented. Neither scope is hidden inside this caller-supplied evaluator dispatch.

Dataset/model smoke checks and compiled-C++ comparison are blocked until matching assets and a compiled reference build exist. Preserve the known non-green `cfg_corpus` baseline; do not rerun it blindly.

## Completion criteria

This plan is complete when each implementation task is checked with its evidence, existing OBB mAP behavior remains unchanged, the host-focused tests pass, docs record the bounded helper/API scope, and all network-level CARPN limitations remain explicit. Completion must not be reported as end-to-end CARPN model mAP or compiled-C++ parity.
