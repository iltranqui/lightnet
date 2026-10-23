# Notes: CARPN Network mAP Plan Expansion

## Table of Contents

- [Sources](#sources)
  - [Rust evaluator and inference](#rust-evaluator-and-inference)
  - [Rust CLI and training boundary](#rust-cli-and-training-boundary)
  - [Reference configuration and weights](#reference-configuration-and-weights)
  - [Reference proposals and decoding](#reference-proposals-and-decoding)
- [Synthesized Findings](#synthesized-findings)
  - [Completed sample API](#completed-sample-api)
  - [Network inference contract](#network-inference-contract)
  - [Training, data, and parity boundaries](#training-data-and-parity-boundaries)

## Sources

### Rust evaluator and inference

- `src/lightnet-core/src/evaluate.rs::ObbValidationDataset::evaluate` currently calls `decode_obb_yolo_detections` with only the inference callback's `yolo_heads`; it normalizes the results and invokes `obb_detector_map`.
- `carpn_detector_map` already checks for CARPN quads and front-point labels, computes polygon IoU, and reuses `detector_map_by` for ranking, matching, duplicate handling, AP, and report counters. `obb_detector_map` now dispatches homogeneous CARPN samples and rejects mixed geometry (v0.0.258).
- `src/lightnet-core/src/runtime.rs::ForwardResult` already carries `layer_outputs` and `yolo_heads` in a full forward. `from_heads` is only a compact compatibility constructor and has no feature tensors.
- `src/lightnet-core/src/detect.rs::decode_obb_yolo_detections` only decodes legacy Stage-1 OBB heads to `FrontPoint`. It already has typed `CarpnQuad`, polygon IoU, and geometry-aware OBB NMS utilities but no network CARPN decoder.
- `model.rs` has `LayerKind::ObbYolo`, `Layer::weights`, and shape inference, but no `[obb_refine]` model/spec or Stage-2 parameter variant. `weights.rs` supports generic layer inspection/round-trip, so the new layer must join the same path.

### Rust CLI and training boundary

- `src/lightnet-cli/src/main.rs::run_detector_map` routes OBB cfgs through `ObbValidationDataset::evaluate`, but its callback drops everything except `ForwardResult.yolo_heads`.
- `detect-image` also passes only `yolo_heads` to the legacy OBB decoder. Both routes need full Stage-2 outputs for CARPN.
- The shared training-validation enum in `main.rs` reuses `ObbValidationDataset`. The CPU OBB trainer computes Stage-1 OBB deltas only; OBB CPU `-map` support does not imply `[obb_refine]` training support. The plan must reject training graphs with `[obb_refine]` until Stage-2 assignment/loss/backward/update are implemented.
- `GpuExecutableNetwork::forward` can retain full outputs; `forward_compact` retains only detector-head summaries. GPU/Oxide CARPN is not included in the CPU-first milestone.

### Reference configuration and weights (`raptor`)

- `src-lib/darknet_cfg_obb.cpp::parse_obb_refine_section` requires `[net] obb=1`. `from` defaults to `-2` and must resolve to a prior spatial layer. `proposal_from=-1` is resolved by the parser to the nearest preceding `[obb_yolo]`; it does not mean “all prior heads” in the implementation, despite stale header comments. Other negative values are relative indexes; an explicit source must be a prior OBB-YOLO layer.
- Classes default to the proposal layer and must match. Source defaults include `proposal_topk_train=128`, `proposal_topk_infer=300`, `proposal_jitter=1`, `sample_grid=3` with a minimum of one, `proposal_thresh=0.01`, `front_range=1.0`, `quality_normalizer=0`, and `csda_topk=9`. Default CARPN anchor scales are `[1,2,4]`; at most three are used, with nonpositive entries treated as 1. Supported `representation` is `carpn`; `loss` accepts `carpn` and `ciou_obb`.
- `src-lib/obb_stage2_refine_layer.cpp::make_obb_stage2_refine_layer` creates a five-output linear head with `inputs = sample_grid² * feature_channels`. `src-lib/weights.cpp::save_obb_refine_weights`/`load_obb_refine_weights` store five biases then `5 * inputs` weights.

### Reference proposals and decoding (`raptor`)

- `src-lib/obb_stage2_proposal.cpp::get_obb_stage2_proposals` scans the selected OBB-YOLO head's raw output. It filters objectness and `objectness * class_probability`, creates one proposal per surviving class, sorts by score, and truncates top-K.
- Stage-1 decoded bbox geometry is used to sample the feature map: a clamped bilinear `sample_grid × sample_grid` with channel-major features and row-major grid positions. The CARPN regression anchor instead uses the static cell center and the prior-anchor half-diagonal, not the predicted bbox center/size.
- `src-lib/obb_stage2_refine_layer.cpp::decode_carpn` clamps `δr` to `[-5,5]`, the other four raw components to `[-front_range,front_range]`, then bounds `|Δα|` and `|Δβ|` by the refined radius and emits four vertices. The final detection carries the proposal class/score and the quad AABB.
- `src-lib/darknet_network.cpp::fill_network_boxes_v3` returns refine outputs instead of Stage-1 detections if any `[obb_refine]` exists; it visits each refine layer. This means the Rust CARPN path must not append unrefined Stage-1 front-point detections to the final CARPN set. Source NMS follows detection generation.
- `src-lib/detector_map_obb.cpp` and `detector_map.cpp` confirm polygon IoU for CARPN quads and front-point requirements for truth reconstruction. No compiled C++ executable/golden is available.

## Synthesized Findings

### Completed sample API

- The previously rejected `obb_detector_map` caller-supplied API gap is closed in v0.0.258. Focused evidence: `obb_map` (5 core tests), `carpn_map` (3), and `carpn_polygon_iou` (2). This proves sample-level scoring only, not network decoding.

### Network inference contract

- The direct missing pieces are typed cfg/model/weight support, Stage-2 CPU forward using raw OBB head output plus a separately routed feature tensor, preservation of per-proposal metadata, bounded quad decoding, and full-forward plumbing into `detect-image` and `ObbValidationDataset`.
- `ObbValidationDataset` should choose explicit CARPN map mode from the graph's `[obb_refine]` layer rather than infer it from the prediction list; this must enforce front-point labels even when no proposal survives.
- The generated end-to-end test should run a tiny fixed-weight `ExecutableNetwork` through its real forward and polygon mAP path. A callback that directly fabricates `CarpnQuad` values is not sufficient evidence.
- The expanded implementation plan is `tasks/rust/carpn_network_map_plan.md`; no implementation has started.

### Training, data, and parity boundaries

- This plan is CPU inference, `detect-image`, and direct `detector map`. CARPN `detector train -map` is explicitly deferred because the Rust OBB trainer has no Stage-2 CSDA/loss/backward/update path. Reject `[obb_refine]` training rather than silently freezing its weights or training Stage 1 only.
- Build deterministic local cfg/weights/image/label fixtures. Canonical CARPN assets are not available as required test oracles; keep any matching dataset smoke optional and report it blocked if assets remain absent.
- Reference behavior is hand-derived from `raptor` source. No compiled-C++ parity claim is allowed, and the known non-green `cfg_corpus` baseline remains unrelated to these focused gates.
