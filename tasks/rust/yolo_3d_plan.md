# Plan: `[yolo_3d]` Stage-1 CPU port

This is the Opus writing-plans pass for `[yolo_3d]` (`tasks/rust/plan-open-items.md`, "Layer parity backlog" item 1). It replaces `yolo_3d_plan_draft.md` as the dispatch document. The draft and `yolo_3d_scope_audit.md` stay as preparation and evidence; do not edit them. The binding rules are `tasks/rust/AGENTS.md` and the Layer Port Contract in `tasks/rust/layers.md` (sections A–G), plus the axis checklist in `plan-open-items.md`.

## Contents

- [Status and evidence boundary](#status-and-evidence-boundary)
- [Objective, scope, and non-goals](#objective-scope-and-non-goals)
- [Decisions (resolves the draft's open decisions)](#decisions-resolves-the-drafts-open-decisions)
- [Source contract](#source-contract)
  - [Facts already established](#facts-already-established)
  - [S1–S13 transcription checklist (Step 0 complete)](#s1s13-transcription-checklist-step-0-complete)
- [File tree and ownership](#file-tree-and-ownership)
- [Public API sketch](#public-api-sketch)
- [Implementation steps and gates](#implementation-steps-and-gates)
  - [Increment 1: v0.1.015 (projected head)](#increment-1-v01015-projected-head)
  - [Increment 2: v0.1.016 (intrinsics, depth, metric)](#increment-2-v01016-intrinsics-depth-metric)
- [Named test matrix](#named-test-matrix)
- [Verification commands](#verification-commands)
- [Bookkeeping](#bookkeeping)
- [Stop criteria](#stop-criteria)
- [Self-review](#self-review)

## Status and evidence boundary

- **Accepted for dispatch; implementation partial.** Source transcription is complete; implemented portions of Steps 1, 2, 4 and 5 and their evidence are recorded in `implemented.md` §§38–40. Next is Step 3 projected loss math and its training integration, then decode/evaluation/A2 axis pin and depth/metric support. The complete layer DoD remains open. The operator requests direct implementation without subagents.
- **Source verification.** The original planning pass relied on the scope audit and local design documents. The implementation pass has since verified `git rev-parse depth_3d` as `9c9e0b4b5e0836330a70a0667fd9e161bd1aac1a`, retrieved the relevant C++ files with `git show depth_3d:<path>`, and transcribed S1–S13 in `tasks/rust/yolo_3d_notes.md`. That notes file is the source contract for implementation; it records discrepancies and deliberate Rust-only strictness. An independent read-only plan review also completed; its findings were resolved below.
- **Version state (checked in `version-map.tsv`).** The latest implementation row is idx 300, `v0.1.026` (`755e5145…`); next available is `v0.1.027`. This plan originally projected `v0.1.015` and `v0.1.016` below; those numbers are historical and must not be reused. Allocate from the current ledger before each commit.
- **Evidence wording.** Every result is "hand-derived from source, no compiled-C++ golden". No compiled-C++ parity is claimed. The C++ branch's own numbers (e.g. OKS8 0.867 vs a Python value of 0.868, from `HANDOFF.md`) are context only, not Rust test oracles.

## Objective, scope, and non-goals

Port `[yolo_3d]` to the Rust CPU path to the Stage-1 standard:

- cfg, shape, checkpoint, and weightless runtime;
- cuboid labels and truth encoding;
- the full training loss with an exact Rust backward;
- decode;
- a clearly named projected-corner evaluator;
- explicit GPU rejection.

All of this is CPU and library-level in `lightnet-core`.

**In scope (both increments together are the layer's Definition of Done):**

1. Aliases `[yolo_3d]`, `[yolo3d]`, `[cuboid]`, `[cuboid_yolo]`. Source defaults, strict errors for unknown or invalid keys, and the channel formula.
2. A weightless head. No `.weights` block and no optimizer state. The checkpoint stream is unchanged, and upstream convolution weights round-trip.
3. Channel, anchor, entry, vertex, and metric layouts pinned by A2 axis tests.
4. Cuboid label parsing, the source truth layout (11-float prefix + `3*K`), per-object intrinsics, and mixed 2D/3D truth ownership.
5. Projected loss terms: objectness/class/box, vertex, visibility, cuboid consistency, and OKS8. Optional terms: `use_3d` metric, depth tilt, orthogonality, and their warm-up. Rust mode uses exact derivatives. Source deviations live only under `GradMode::Cpp`.
6. Decode into a typed cuboid detection. The evaluator uses AP@OKS8 (primary 0.50, secondary mean over 0.50:0.05:0.95), with silhouette IoU as a secondary metric. All names say "projected".
7. DPT reference validation for `depth_layer`. Depth is stop-gradient.
8. Upstream optimizer eligibility. The head owns nothing. Upstream 2D params use the existing SGD, Adam, and Muon (`[net] muon=1`) paths.
9. Explicit GPU rejection for execution, compact, f16, backward, and updates.

**Non-goals (explicitly rejected, never silently ignored):**

- Native GPU or Oxide kernels and host GPU fallback: Stage 2.
- CLI `detector train`/`map`/`test` for `[yolo_3d]` networks. These must **reject** with a clear error naming the library APIs, and the follow-up goes in `todo.md`. This follows the CARPN precedent, where network/CLI wiring is a separate plan.
- An image data loader with augmentation sampling. The truth encoder takes explicit crop/flip/mosaic parameters and is fully tested; random augmentation policy is not ported.
- ONNX export. The existing catch-all in `onnx/export.rs:84` already returns `unsupported`. Do not edit `onnx/export.rs` or `onnx/proto.rs`: both carry uncommitted user changes.
- Training runs, `cfg_corpus`, Oxide tests, and any compiled-C++ parity claim.

## Decisions (resolves the draft's open decisions)

| # | Decision | Evidence / reason |
|---|---|---|
| D1 | **Ship as two code increments.** v0.1.015 is the projected head: cfg, runtime, labels, full truth layout including intrinsics/`geo` encoding, projected and OKS8 loss, decode, evaluator, and GPU rejection. It accepts `focal_35mm` and `winding` because the truth encoder needs them, and `oks_normalizer`/`oks_kappa` because increment 1 implements OKS8. v0.1.016 adds *loss/decode consumption* of intrinsics/`geo`, depth tilt, ortho, `depth_layer`, and `use_3d` metric. Until v0.1.016, nondefault settings for terms that increment 2 alone implements must fail with `InvalidConfig` saying they are "not implemented until v0.1.016"; see the exact split in S2. The layer is "done" only after v0.1.016. | The audit allows a projected-first slice if acceptance and strict errors are defined (`yolo_3d_scope_audit.md` "Port boundaries"). DPT is already ported (v0.1.013), so this is a size split, not a dependency deferral. OKS8 needs only GT corners (`HANDOFF.md` §5), so it belongs in increment 1. |
| D2 | **Metric mode (`use_3d=1`) is in scope.** The `yolo_3d.hpp:145-151` "reserved/not implemented in v1" comment is stale and must not be followed. | `Y3D_METRIC_CHANNELS=7`, parser keys, metric decode/loss, `Yolo3DMetric` tests, and `book_cuboid_depth_anything_yolo3d_metric.cfg` (audit "Important source discrepancy"). |
| D3 | **Forward is a pass-through.** The head output is the raw preceding tensor. Activations are applied inside the pure loss/decode math, as Rust `[yolo]` does (`runtime.rs:3177-3183`; `train.rs:2734` `yolo_objectness_output`). Activation math is still taken from the C++ source (S4). | This keeps one Rust convention for every YOLO-family head, the existing seed pass-through backward (`runtime.rs:5293`), and ONNX/compact behavior. The axis tests pin raw-channel indexing, so the storage difference from C++ in-place activation cannot hide a layout bug. |
| D4 | **Seeds follow the existing loss-head convention: seed = −∂L_rust/∂raw** (`layers.md` Global Constraints: "seed = Darknet's descent direction"). `Yolo3dTrainingLoss` reports both `loss` (the Rust scalar whose gradient the seeds are) and `source_cost` (the C++ printed cost formula, for logging). | Mirrors `ObbYoloTrainingDeltaBatch::source_loss_cost` (`train.rs:288`). It lets finite-difference tests target `loss` exactly. |
| D5 | **Rust mode vs `GradMode::Cpp`.** For each source delta, Step 0 classifies it: (a) an exact gradient of an identifiable per-term loss, which is the same in both modes; or (b) a shortcut. Shortcuts are logistic-gradient omissions, finite-difference geometry gradients (tilt/ortho use FD in C++, `HANDOFF.md` §5), and any non-gradient scaling. Rust mode is the analytic derivative of the named loss. `GradMode::Cpp` reproduces the source delta, including the source FD step size for tilt/ortho. Anchor assignment, ignore masks, visibility masks, face selection, and `geo_burn_in` are piecewise constant and are held fixed at the forward point; FD tests choose points away from those boundaries. Delta clipping (`max_delta`, if the source has it for this head) is a gradient transform applied in both modes, documented as such, and disabled in FD tests. | AGENTS.md rule "Backward = exact derivative of the Rust forward. C++ deviations only under `GradMode::Cpp`". |
| D6 | **Depth is stop-gradient.** The `[dpt_head]` output feeds tilt targets only. No gradient flows into the DPT layer or its inputs from `[yolo_3d]`. | `loss_design.md` "L_tilt = … t_depth stop-grad". The C++ DPT backward is a no-op, and the source cfg sets DPT `onlyforward=1` (`HANDOFF.md` §3). |
| D7 | **Typed, separate APIs; no reuse of OBB types.** Add `Yolo3dSpec`, `LayerKind::Yolo3d`, a `ForwardResult::yolo_3d_heads` vector, `CuboidLabel`, `Yolo3dTruthBatch`, `Yolo3dDetection`, and `Yolo3dValidationDataset`. Do not push `[yolo_3d]` into `yolo_heads`, `ObbLabel`, or `ObbDetectionGeometry`. | Semantics differ: 8 vertices + visibility + intrinsics (audit "Existing Rust integration boundaries"). A separate vector keeps every existing `yolo_heads` consumer (`train.rs:841`, `detect.rs:95`, the HF auxiliary at `runtime.rs:18515`) unchanged. |
| D8 | **Existing 2D APIs must not silently ignore 3D heads.** `compute_yolo_training_loss_with_state`, `decode_yolo_detections`/`decode_network_detections`, `ValidationDataset::from_data`, and the OBB entry points return `InvalidConfig` naming the yolo_3d API when the network contains `LayerKind::Yolo3d`. The exception is a mixed network routed through the mixed API (D9). | Rule "Unknown config = error, never silent identity". A 2D-only loss on a mixed net would silently drop 3D supervision. |
| D9 | **Mixed `[yolo]` + `[yolo_3d]` training** goes through one entry point, `compute_mixed_2d_3d_training_loss`. It splits truths by the source ownership flags (S7), calls the 2D loss core (an internal variant that skips the D8 guard) for the 2D heads and the yolo_3d loss for the 3D heads, and **sums** seeds per layer (lesson: "each pass owns only its partial gradient"). Shared classes and keypoints are validated at network build, as in `darknet_cfg.cpp:1603-1687`. | Audit evidence on the C++ flags slot. |
| D10 | **Mixed-network decode.** `decode_yolo_3d_network_detections` returns `Yolo3dNetworkDetections { boxes, cuboids }`: the pinned source's network decoder emits both `[yolo]` boxes and `[yolo_3d]` cuboids, then uses separate AABB-NMS pools when combined truth is configured (S8). The projected evaluator requires K=8; the library decoder may expose K<8 projected points, whose legacy public C++ `Prediction` does not set `has_cuboid`. | `darknet_network.cpp:1846-1947`; `darknet.cpp:1490-1514,1570-1588`; `box.cpp:1010-1068`; `yolo_3d.cpp:2694-2724`. |
| D11 | **The evaluator is named projected-only.** Use `Yolo3dProjectedMapReport` with `ap_oks8_50`, `ap_oks8_50_95`, `ap_silhouette_iou_50`, and `mean_oks8`. Docs and test names say "projected corner", never "3D IoU"/"3D mAP". OKS8 uses all valid labeled corners, including occluded ones, `D = max(w_gt, h_gt)`, and default `kappa = 0.0849`. The image scale is a caller-supplied original W/H. When only network W/H is known, the report must flag `scale_source: NetworkSize` (C++ uses network W/H, `HANDOFF.md` "Known issues"). | `cuboid_metrics_research.md` §4; audit "Settled cuboid evaluation design". |
| D12 | **Optimizers.** The head has `LayerParams::None` and `LayerTrainBuffers::None`, and no Muon visitor arm. An upstream convolution keeps its existing SGD, Adam/AdamW, and Muon (`[net] muon=1`, `model.rs:78,182`) treatment. Tests prove that the conv updates and that no head state is written to the DNOPT sidecar. | Audit: `weights.cpp` has no `YOLO_3D` case; anchors are metadata. `weights.rs:378` already skips `LayerParams::None`. |
| D13 | **GPU rejection** reuses the existing Stage-1 helper by adding `LayerKind::Yolo3d` to the `matches!` in `reject_gpu_hyperbolic_foveation` (`runtime.rs:80-99`). That one helper already guards f16 (8359), updates (9903), execution (11329), compact (13163), and backward/updates (13785). Do **not** add `Yolo3d` to the GPU pass-through arms (`runtime.rs:8697, 12309, 13683`) or to `requires_cpu_backward_fallback`. | AGENTS.md: CPU-only Stage 1, no host fallback. |

## Source contract

### Facts already established

Source facts were independently verified against the pinned `depth_3d` snapshot and transcribed with exact formulas/citations in `yolo_3d_notes.md`.

- **Enum/aliases.** `darknet_enums.cpp:147-151`, `define_layers.hpp:36` (`LAYER_YOLO_3D` enabled).
- **Parser.** `darknet_cfg.cpp:2555-2699` covers defaults, validation, vertex-loss kinds, depth/focal keys, and anchors. `darknet_cfg.cpp:1549-1593` checks that `depth_layer` points to an earlier `[dpt_head]` and checks the tilt/ortho prerequisites. `darknet_cfg.cpp:1603-1687` covers the mixed-head flags slot, shared class count, and shared keypoint count.
- **Channels.** Each anchor has `5 + C + 3*K` channels, and `use_3d=1` adds 7 per anchor (`yolo_3d.hpp:181-228`, `Y3D_METRIC_CHANNELS=7`). Layout and decode contracts are in `yolo_3d.hpp:54-151`.
- **Shape and CPU path.** Shape, allocation, and defaults are in `yolo_3d.cpp:1275-1410`. CPU activation, training loss, cost, and backward into the previous layer are in `yolo_3d.cpp:1501-1672`. Decode, geometry, tilt, ortho, OKS8, and detection decode are in `yolo_3d.cpp:1679-1755,1976-2598,2654-2853`. The GPU paths (`yolo_3d.cu:369-708`, `yolo_3d.cpp:2865-2984`) are reference only and are not ported.
- **Truth per object.** `[cx,cy,w,h,cls,track, fx,fy,px,py, geo]` followed by `u,v,c` repeated K times. Offsets come from the `Y3D_TRUTH_*` constants and `y3d_truth_prefix_size()` in `yolo_3d.hpp` (`HANDOFF.md` §4).
- **Loader.** `data.cpp:907-964,1510-1768`. It computes `f_px = focal_35mm/43.27*hypot(ow,oh)`, `fx=f_px/swidth`, `fy=f_px/sheight`, and `px=(ow/2-pleft)/swidth` (`py` analogous). On flip it sets `px=1-px`. It sets `geo = winding*(flip?-1:1)`, and `geo=0` when there is no focal, a corner is clipped, or mosaic is used. Under mosaic, 3D corners are dropped (unsupervised); the source deliberately does not blend them (`todo.md` Review). The 3D loader is selected by `yolo_3d_keypoints > 0`.
- **Visibility/face flags.** `c == 0` means absent; `c < 0` means unknown and is treated as present by vertex/metric coordinate loss, while also serving as a hidden-corner marker for face inference. If negatives are present, only valid hidden sets are accepted (1 corner, 2 edge-adjacent corners, or 4 corners of one face); an inconsistent set returns invalid and does not fall back to winding. Winding is used only when no c is negative. See `yolo_3d_notes.md` S5/S11 and `yolo_3d.cpp:587-652,1976-2085`.
- **Intrinsics/depth keys.** Defaults and validations are recorded in `yolo_3d_notes.md` S2. `focal_35mm` and `winding` feed truth encoding; OKS8 is a projected term; depth tilt/ortho/depth-layer and metric channels are opt-in. `geo_burn_in` gates tilt/ortho at `iteration >= geo_burn_in` and does not gate OKS8.
- **Tilt.** Exact source formulas, including least-squares fit of the sampled DPT output scalar (without an extra reciprocal), reliability saturation `w_depth=R2*s/(1+s)` with `s=|(A,B)|*L/sigma_res`, vanishing-point construction, face-width filter, normalized area weighting, 9x9/20%-margin depth sampling, and finite-difference step, are in `yolo_3d_notes.md` S11. This supersedes the shorthand in `loss_design.md`; use the pinned executable source (`yolo_3d.cpp:2089-2455`).
- **Ortho.** The exact three edge families, longest-two-GT-edge vanishing-point construction, unit K-backprojected directions, pair weights, and C++ FD step are transcribed in `yolo_3d_notes.md` S11 (`yolo_3d.cpp:35-46,2463-2548`).
- **OKS8.** `1 - OKS8`, `kappa = 0.0849`, `D` = max side of the GT corner box, analytic gradient (`HANDOFF.md` §5).
- **Weights.** `weights.cpp` has no `YOLO_3D` case. The head has no serialized parameters, and anchors are metadata.

### S1–S13 transcription checklist (Step 0 complete)

Write each item into `tasks/rust/yolo_3d_notes.md` with a pinned file:line citation and the exact formula or value. **Each item is a gate for the steps listed.** If an item cannot be resolved from the source, apply [Stop criteria](#stop-criteria).

| ID | Item | Gates |
|---|---|---|
| S1 | Exact alias match spelling and section-name normalisation. | 1 |
| S2 | Every key read by `parse_yolo_3d_section`, with spelling, type, default, source validation/fallback, ignored status where relevant, and increment split: dimensions/`num`/`mask`/`anchors`/K, all generic YOLO loss and data keys, all projected/visibility/geometry/metric keys, and `map`. Distinguish `keypoints` from the runtime field name; K range and which terms require K=8. | 1, 6 |
| S3 | Channel order within one anchor block (box, objectness, classes, `(u,v,vis)` triplets, then the 7 metric channels) and the meaning of each metric channel. | 1, 2, 3, 6 |
| S4 | Activation per channel group (logistic, linear, exp, `scale_x_y`), and the decode equations for box, vertices, and metric values in network and image space, including letterbox correction. | 3, 4 |
| S5 | Pure cuboid and mixed 2D row grammar, field counts/order and source precedence (2D 5/7/9-field rows are classified before cuboid counts, so K=2 leading-class 7-field and K=3 implicit-class 9-field cuboids are ambiguous), coordinate clamping, actual rejects, and distinct visibility semantics: `c==0` absent; `c<0` unknown but treated as present by vertex/metric loss, skipped by visibility BCE, and used as a separate hidden-face marker; visibility target threshold is `c>0.5`. | 2 |
| S6 | Truth-slot encoding: `max` boxes, the 11-float prefix, `track`, and how `cx,cy,w,h` derive from the corners or the row. | 2 |
| S7 | Mixed flags slot: its position, its values (`EY3DTruthFlags`), and which heads consume which truths. | 2, 5 |
| S8 | Mixed-network decode (both head types), objectness×class confidence, letterbox correction, `has_cuboid`/K<8 behavior, and the actual class-sorted AABB-NMS pool/threshold path (including which parsed NMS keys are unused). | 4 |
| S9 | Anchor assignment, ignore-threshold rule, positive selection, and the per-term deltas and cost for objectness, class, box, vertex, visibility, and cuboid consistency. Classify each delta per D5. | 3 |
| S10 | Flip transforms every vertex u coordinate independently without permuting indices; it mirrors principal point and negates the configured winding sign in `geo`. | 2, 6 |
| S11 | Tilt and ortho exact formulas, face list/hidden-set semantics, `sigma_res` and saturating reliability, vanishing-point construction, edge families, source FD step, depth-map pixel mapping, and `geo_burn_in` comparison. Confirm whether a full-normal loss exists (none in this snapshot). | 6 |
| S12 | Metric (`use_3d`) loss terms, the metric truth they need, and where that truth comes from (the label row or derived data); explicitly resolve the stale header comment. | 6 |
| S13 | `depth_layer` resolution (absolute vs relative index) and the exact prerequisite checks (e.g. tilt needs `depth_layer` and `focal_35mm`; ortho needs `focal_35mm`). | 6 |

## File tree and ownership

```
src/lightnet-core/src/
├── lib.rs                MODIFY  declare yolo_3d, yolo_3d_geometry, cuboid_labels; re-export public API
├── model.rs              MODIFY  Yolo3dSpec, LayerKind::Yolo3d, name(), parse arm + reject_unknown_yolo_3d_keys,
│                                 infer_output_shape arm, depth_layer + mixed-head build validation, tests
├── yolo_3d.rs            CREATE  layout/index helpers, activation, decode math, anchor assignment,
│                                 projected loss + seeds (Rust and Cpp modes), unit/FD/axis tests
├── yolo_3d_geometry.rs   CREATE  intrinsics, visible faces, face width, depth plane fit, tilt, ortho,
│                                 OKS8 + gradient, silhouette IoU, metric terms, tests
├── cuboid_labels.rs      CREATE  CuboidLabel row parser, truth encoder (crop/flip/mosaic), mixed ownership split, tests
├── runtime.rs            MODIFY  forward pass-through arm (3177 group), yolo_3d_heads capture (≈3294),
│                                 ForwardResult field + from_heads + every constructor, backward pass-through (5293 group),
│                                 reject_gpu_hyperbolic_foveation matches! (80-99), runtime test module registration (≈45618)
├── runtime/tests/yolo_3d_runtime_tests.rs  CREATE  runtime, checkpoint, optimizer, GPU-rejection, mixed-loss tests
├── train.rs              MODIFY  Yolo3dTrainingLoss, compute_yolo_3d_training_loss(_with_state),
│                                 compute_mixed_2d_3d_training_loss, D8 guard in compute_yolo_training_loss_with_state
├── detect.rs             MODIFY  Yolo3dDetection, decode_yolo_3d_network_detections, D8 guard in 2D decoders
└── evaluate.rs           MODIFY  Yolo3dValidationDataset, Yolo3dProjectedMapReport, yolo_3d_projected_map, D8 guard
src/lightnet-cli/src/main.rs  MODIFY  explicit rejection in detector train/map/test for yolo_3d networks + one CLI test
tasks/rust/yolo_3d_notes.md  CREATE (Step 0)  source transcription S1–S13 (local-only, git add -f when committed)
```

Do **not** touch: `onnx/export.rs` and `onnx/proto.rs` (uncommitted user edits), `Cargo.toml`, `Cargo.lock`, `.gitignore`, root `notes.md`/`task_plan.md`/`CLAUDE.md`, `weights.rs` and `initialize.rs` (no new params; the `LayerParams::None` arms at `weights.rs:378` and `initialize.rs:14` already cover the head), `muon_runtime.rs` (no head matrices), and `lightnet-cuda*`. If the compile inventory (Step 1) shows an exhaustive match in one of these files, add only the minimal `LayerKind::Yolo3d` arm there, and stop and ask if the file is in the "uncommitted user edits" list.

**Component roles:**

- **`model.rs`.** Owns cfg truth, which is shape plus validation. It depends on S1–S3 and S13. Everything else depends on it.
- **`yolo_3d.rs`.** Pure projected math with no I/O. It depends on `Yolo3dSpec`. `train.rs`, `detect.rs`, and `evaluate.rs` depend on it.
- **`yolo_3d_geometry.rs`.** Pure geometry with an optional depth slice. `yolo_3d.rs` (OKS8 loss term) and `evaluate.rs` (OKS8/silhouette) depend on it. Increment 2 adds the tilt, ortho, and metric parts.
- **`cuboid_labels.rs`.** Labels to truth. Depends on S5–S7 and S10. `train.rs` and `evaluate.rs` depend on it.
- **`runtime.rs`.** Thin arms only. The design constraint is that no new `LayerParams` variant is allowed.
- **CLI.** Rejection only.

## Public API sketch

Names are binding. Fields marked "S#" are completed from Step 0.

```rust
// model.rs
pub struct Yolo3dSpec {
    pub classes: usize, pub num: usize, pub mask: Vec<usize>, pub anchors: Vec<f32>,
    pub keypoints: usize,            // K (S2)
    pub use_3d: bool,                // +7 channels per anchor
    pub max_boxes: usize,
    /* source normalizers/thresholds/vertex-loss kind: S2, S9 */
    pub geometry: Yolo3dGeometrySpec,
}
pub struct Yolo3dGeometrySpec {
    pub focal_35mm: f32, pub winding: i32, pub depth_layer: Option<usize>,   // absolute, validated
    pub tilt_normalizer: f32, pub tilt_min_face_width: f32, pub ortho_normalizer: f32,
    pub oks_normalizer: f32, pub oks_kappa: f32, pub geo_burn_in: usize,
}
impl Yolo3dSpec { pub fn entries_per_anchor(&self) -> usize; /* 5+C+3K (+7) */ }

// runtime.rs
pub struct Yolo3dHeadOutput { pub layer_index: usize, pub output: Tensor }
// ForwardResult gains `pub yolo_3d_heads: Vec<Yolo3dHeadOutput>`

// cuboid_labels.rs
pub struct CuboidLabel { pub class_id: usize, pub vertices: Vec<[f32; 3]> /* u,v,c */, /* S5 */ }
pub struct Yolo3dImageMeta { pub original_width: usize, pub original_height: usize,
                             pub crop: Yolo3dCrop /* pleft, ptop, swidth, sheight, flip, mosaic */ }
pub struct Yolo3dTruthBatch { pub truths: Vec<f32>, pub truth_size: usize, pub max_boxes: usize, pub batch: usize }
pub fn load_cuboid_labels(path, classes: Option<usize>, keypoints: usize) -> Result<Vec<MixedLabel>>;
pub fn encode_yolo_3d_truths(labels: &[Vec<MixedLabel>], meta: &[Yolo3dImageMeta], spec: &Yolo3dSpec) -> Result<Yolo3dTruthBatch>;
pub enum MixedLabel { Box2d(YoloLabel), Cuboid(CuboidLabel) }  // ownership per S7

// train.rs
pub struct Yolo3dTrainingLoss { pub loss: f32, pub source_cost: f32,
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>, pub positive_matches: usize,
    pub term_costs: Yolo3dTermCosts /* obj, class, box, vertex, vis, consistency, oks, tilt, ortho, metric */ }
pub fn compute_yolo_3d_training_loss_with_state(network, forward, truths: &Yolo3dTruthBatch,
    iteration: usize, grad_mode: GradMode) -> Result<Yolo3dTrainingLoss>;
pub fn compute_mixed_2d_3d_training_loss(network, forward, labels: &[Vec<MixedLabel>],
    meta: &[Yolo3dImageMeta], iteration: usize, grad_mode: GradMode) -> Result<Yolo3dTrainingLoss>;

// detect.rs
pub struct Yolo3dDetection { pub detection: Detection, pub projected_vertices: Vec<[f32; 2]> /* K per S2 */,
    pub visibility: Vec<f32>, pub metric: Option<[cx, cy, cz, length, width, height, yaw]> }
pub fn decode_yolo_3d_network_detections(network, forward, batch_index, image_w, image_h,
    options: DetectionOptions) -> Result<Yolo3dNetworkDetections>;

// evaluate.rs
pub struct Yolo3dProjectedMapReport { pub ap_oks8_50: f32, pub ap_oks8_50_95: f32,
    pub ap_silhouette_iou_50: f32, pub mean_oks8: f32, pub per_class: Vec<..>, pub scale_source: OksScaleSource }
```

Pass `grad_mode` explicitly instead of reading the environment inside the pure math. Callers use `GradMode::from_env()`. This keeps both modes testable in one process.

## Implementation steps and gates

Each step is test-first. Write the named tests and see them fail (or fail to compile) before you implement. Run one command at a time.

### Increment 1: v0.1.015 (projected head)

**Step 0: Source transcription (completed; gate for all later steps).**
- The source pass refreshed paths with `git grep`, retrieved the cited files via `git show depth_3d:<path>`, and verified `git rev-parse depth_3d` = `9c9e0b4b5e0836330a70a0667fd9e161bd1aac1a`.
- `tasks/rust/yolo_3d_notes.md` resolves S1–S13 with source citations and D5 delta classifications. It records the stale `use_3d` header comment (D2), the unknown/hidden visibility distinction, exact `>=` burn-in boundary, mixed decode/NMS, and executable-source tilt reliability formula.
- Gate passed: no S-item remains unresolved; the plan is accepted for dispatch after these source-backed corrections. The `yolo_3d.hpp:145-151` `use_3d` reserved-term comment is stale relative to executable code and tests. No compiled-C++ parity is established.

**Step 1: Model and cfg.**
- Add `Yolo3dSpec`, `LayerKind::Yolo3d`, the `name()` arm (`"yolo_3d"`), and the parse arm with all four aliases (`model.rs:3707` match). Add `reject_unknown_yolo_3d_keys` following the `reject_unknown_dcnv4_keys` pattern (`model.rs:6169`).
- Use checked arithmetic for `mask.len()*entries` and require it to equal `current.channels`. Model this on `[obb_yolo]` (`model.rs:4117-4131`). Shape is identity.
- Build-time checks: shared classes/keypoints for mixed networks; nondefault increment-2-only feature requests rejected (D1). `focal_35mm` and `winding` remain accepted for increment-1 truth encoding; OKS8 keys are increment-1 features.
- Run `MEM=3G ./tools/test_host.sh does_not_exist` once. Its compile errors list every non-exhaustive `match` (Contract A). Add arms only where the compiler demands them, then apply the "Do not touch" rule.
- Gate: all model tests in the matrix pass with `MEM=3G ./tools/test_host.sh yolo_3d_cfg`.

**Step 2: Labels and truth.**
- Implement `cuboid_labels.rs`: the row parser (S5), the truth encoder with the full 11 + 3K prefix including intrinsics, crop, flip, and mosaic (S6, S10, the loader formulas above), and the mixed split (S7).
- Gate: `MEM=3G ./tools/test_host.sh yolo_3d_label` and `MEM=3G ./tools/test_host.sh yolo_3d_truth` pass.

**Step 3: Projected loss math.**
- In `yolo_3d.rs`: the layout/index helper `entry_index(batch, anchor, entry, row, col)` using the C++ formula from S3; activation (S4); assignment and per-term losses (S9); the OKS8 term via `yolo_3d_geometry::oks8` (+ gradient); Rust-mode seeds; and the Cpp-mode source deltas (D5). Increment 1 implements none of the geometry terms (tilt, ortho, metric).
- Gate: hand-computed, finite-difference, and axis tests pass via `MEM=3G ./tools/test_host.sh yolo_3d`.

**Step 4: Runtime, training API, decode, evaluator.**
- Runtime: add the pass-through forward arm, `yolo_3d_heads` capture, the `ForwardResult` field (initialise it in **every** constructor, including `from_heads` at `runtime.rs:847` and the GPU result at ≈13149), and the backward pass-through arm.
- `train.rs`: add the APIs and the D8 guard. `detect.rs`: add decode and D10, then NMS per S8. `evaluate.rs`: add AP@OKS8, silhouette IoU, and the report (D11).
- CLI: add the rejection.
- Gate: runtime, evaluator, and CLI tests pass.

**Step 5: CPU-only boundary and optimizer/checkpoint proof.**
- Make the D13 change. Add the GPU rejection test using the `dpt_head_rejects_gpu_execution_backward_and_updates` pattern (`runtime/tests/dpt_head_runtime_tests.rs:719`). Add the checkpoint and SGD/Adam/Muon tests (D12).
- Gate: `MEM=3G ./tools/test_host.sh yolo_3d` passes, plus the regression filters in [Verification commands](#verification-commands).

**Step 6: Review and commit v0.1.015.**
- Run targeted rustfmt and `git diff --check`, review every new error path, and confirm the "Do not touch" list has no diff.
- Commit `v0.1.015: add yolo_3d projected head Stage-1 CPU support`, staging each file by path. Then do the separate docs commit ([Bookkeeping](#bookkeeping)).

### Increment 2: v0.1.016 (intrinsics, depth, metric)

**Step 7: Geometry math.**
- In `yolo_3d_geometry.rs`: `Yolo3dIntrinsics` from the truth prefix, visible faces (negative hidden-corner set first; invalid sets return invalid; winding is only the no-negative fallback), face width, the affine plane fit of sampled DPT output values in ray coordinates over the 9x9/20%-margin samples, tilt with source reliability and area weighting, and ortho with edge-length weighting. There is no full-normal term in the pinned source.
- Each function gets a hand test. Each loss gets an analytic gradient with an FD test.
- Gate: `MEM=3G ./tools/test_host.sh yolo_3d_geometry` passes.

**Step 8: Integration.**
- Remove the D1 rejections. Validate `depth_layer` (an earlier `LayerKind::DptHead`; future, self, out-of-range, and wrong-kind references are `InvalidConfig`) and the S13 prerequisites.
- Read the depth tensor from `forward.layer_outputs[depth_layer]` as stop-gradient (D6). Gate tilt/ortho with `iteration >= geo_burn_in` per S11; OKS8 is not behind this gate.
- Add the `use_3d` channels to the layout, decode (`Yolo3dDetection::metric`), and the loss (S12). Add Cpp-mode FD deltas.
- Gate: all increment-2 matrix tests pass, plus the full `yolo_3d` filter.

**Step 9: Review and commit v0.1.016.** Do the same as Step 6, with message `v0.1.016: add yolo_3d depth/focal geometry and metric terms`, then the docs commit.

## Named test matrix

Every test is new. "Hand" means the numbers are written in the test and derived independently of the implementation, from source formulas in `yolo_3d_notes.md`. "FD" means central finite differences of `L_rust` against `-seed`, with tolerance ≤ 1e-3 relative, in f64 where the helper allows, at points away from assignment, mask, and kink boundaries.

| Area | Test (file) | Inc |
|---|---|---|
| cfg | `yolo_3d_cfg_aliases_parse_to_one_kind`, `yolo_3d_cfg_defaults_match_source`, `yolo_3d_cfg_rejects_unknown_key`, `yolo_3d_cfg_rejects_invalid_values` (zero classes/K/num, mask index ≥ total, non-positive or non-finite anchors/normalizers, overflow; these strict errors are Rust boundaries where C++ may fall back or leave values unchecked) (`model.rs`) | 1 |
| shape | `yolo_3d_cfg_projected_channels_are_a_times_5_plus_c_plus_3k`, `yolo_3d_cfg_rejects_preceding_channel_mismatch`, `yolo_3d_cfg_metric_adds_seven_channels_per_anchor` (inc 1: asserts rejection of `use_3d=1` per D1; inc 2: asserts count) (`model.rs`) | 1/2 |
| staging | `yolo_3d_cfg_increment_two_keys_rejected_until_implemented` (inc 1; deleted/replaced in inc 2 by acceptance tests) | 1 |
| mixed cfg | `yolo_3d_cfg_mixed_network_requires_shared_classes_and_keypoints` (`model.rs`) | 1 |
| depth ref | `yolo_3d_cfg_depth_layer_must_be_earlier_dpt_head` (future, self, out-of-range, wrong kind, valid), `yolo_3d_cfg_geometry_prerequisites_match_source` (S13) (`model.rs`) | 2 |
| axis A2 | `yolo_3d_axis_layout_anchor_entry_vertex_channels_are_channel_major`: 3x5 grid, B=2, A=2, C=2, K per S2, unique sentinel per (batch, anchor, entry, row, col); checks the `entry_index` and decode readback, with no cross-batch leakage (`yolo_3d.rs`) | 1 |
| axis A2 | `yolo_3d_axis_layout_metric_channels_follow_projected_block` (`yolo_3d.rs`) | 2 |
| activation/decode | `yolo_3d_decode_box_and_vertices_hand_computed` (non-square net, letterbox on/off), `yolo_3d_decode_scales_each_image_independently`, `yolo_3d_mixed_decode_returns_boxes_and_cuboids`, `yolo_3d_decode_nms_and_confidence_follow_source` (`yolo_3d.rs`/`detect.rs`) | 1 |
| decode metric | `yolo_3d_decode_metric_values_hand_computed` (`detect.rs`) | 2 |
| labels | `yolo_3d_label_row_parses_source_layout`, `yolo_3d_mixed_label_ambiguous_row_precedence_matches_source`, `yolo_3d_label_rejects_malformed_rows` (field count, class bounds, degenerate hull; Rust-only finite-value validation is identified as a safety extension), `yolo_3d_label_clamps_out_of_range_coordinates_like_source`, `yolo_3d_label_unknown_visibility_is_not_absent` (`cuboid_labels.rs`) | 1 |
| truth | `yolo_3d_truth_prefix_offsets_match_source` (sentinels in all 11 + 3K slots, `max` cap/padding with the documented deterministic Rust selection policy), `yolo_3d_truth_intrinsics_hand_computed` (focal_35mm=24 at 1280x960 gives f_px ≈ 888 px), `yolo_3d_truth_flip_mirrors_px_and_negates_geo_without_reordering_vertices`, `yolo_3d_truth_mosaic_drops_projected_fields`, `yolo_3d_truth_clipped_corner_zeroes_geo_but_keeps_clamped_points` (`cuboid_labels.rs`) | 1 |
| ownership | `yolo_3d_truth_mixed_ownership_routes_rows_to_heads` (`cuboid_labels.rs`), `yolo_3d_mixed_loss_each_head_owns_partial_gradient` (2D seed equals the 2D-only loss on 2D-owned rows; 3D seed equals the 3D-only loss; the summed seed has no double count) (`yolo_3d_runtime_tests.rs`) | 1 |
| loss hand | `yolo_3d_projected_loss_hand_computed` (one GT, one positive cell, every term value), `yolo_3d_ignore_threshold_masks_objectness_hand_case` (`yolo_3d.rs`) | 1 |
| FD | `yolo_3d_projected_seed_matches_finite_difference` (all channels of a 2x3 grid), `yolo_3d_visibility_seed_matches_finite_difference`, `yolo_3d_oks8_gradient_matches_finite_difference` (`yolo_3d.rs`/`yolo_3d_geometry.rs`) | 1 |
| Cpp mode | `yolo_3d_cpp_mode_reproduces_source_deltas` (each class-(b) delta from S9; Rust mode differs where expected) | 1 |
| OKS8 | `yolo_3d_oks8_hand_computed_with_kappa_0849` (half-similarity at 0.1·D gives 0.5), `yolo_3d_oks8_uses_all_valid_labeled_corners` (`yolo_3d_geometry.rs`) | 1 |
| geometry | `yolo_3d_geometry_visible_faces_hidden_unknown_and_winding`, `yolo_3d_geometry_rejects_inconsistent_hidden_set`, `yolo_3d_geometry_face_width_threshold_skips_thin_faces`, `yolo_3d_geometry_depth_tilt_is_affine_invariant`, `yolo_3d_geometry_tilt_loss_hand_computed`, `yolo_3d_geometry_tilt_gradient_matches_finite_difference`, `yolo_3d_geometry_ortho_loss_hand_computed`, `yolo_3d_geometry_ortho_gradient_matches_finite_difference` | 2 |
| metric | `yolo_3d_metric_loss_hand_computed`, `yolo_3d_metric_seed_matches_finite_difference` | 2 |
| gating | `yolo_3d_geo_burn_in_gates_geometry_terms` (at and around the boundary per S11), `yolo_3d_cpp_mode_geometry_uses_source_finite_difference_step` | 2 |
| stop-grad | `yolo_3d_depth_layer_receives_no_gradient` (`[dpt_head]` → … → `[yolo_3d]` chain; DPT and upstream-of-DPT gradients from the 3D loss are zero) (`yolo_3d_runtime_tests.rs`) | 2 |
| runtime | `yolo_3d_forward_is_pass_through_and_records_head`, `yolo_3d_backward_passes_seed_to_previous_convolution` (plus FD on two upstream conv weights through the full loss), `yolo_3d_from_heads_and_all_result_constructors_have_field` (`yolo_3d_runtime_tests.rs`) | 1 |
| checkpoint | `yolo_3d_weights_round_trip_exact_length_with_upstream_conv` (file = `20 + 4*conv_params`; the head adds 0 bytes; `trailing_bytes == 0`) | 1 |
| optimizers | `yolo_3d_sgd_and_adam_update_upstream_conv_only`, `yolo_3d_muon_updates_upstream_conv_and_writes_no_head_state` (`[net] muon=1`; DNOPT round trip) | 1 |
| guards | `yolo_3d_existing_2d_and_obb_apis_reject_yolo_3d_networks` (train loss, decode, `ValidationDataset::from_data`), `yolo_3d_onnx_export_rejects` (calls `export_onnx`; asserts the existing unsupported error; no export.rs edit) | 1 |
| GPU | `yolo_3d_rejects_gpu_execution_backward_and_updates` (execution, compact, f16, backward, update) | 1 |
| eval | `yolo_3d_projected_map_oks8_threshold_boundary`, `yolo_3d_projected_map_duplicate_detection_is_false_positive`, `yolo_3d_projected_map_reports_ap50_and_ap50_95`, `yolo_3d_silhouette_iou_hand_computed`, `yolo_3d_projected_map_flags_network_size_scale` (`evaluate.rs`) | 1 |
| CLI | `yolo_3d_cli_detector_train_map_test_reject` (`lightnet-cli`) | 1 |

All names start with `yolo_3d_`, so the `yolo_3d` filter selects the whole set. Axis tests also match `axis_layout`.

## Verification commands

Run these serially, never two at once. Each one runs under `MEM=3G` (the script wraps `systemd-run --user --scope -p MemoryHigh=3G` and builds with `-j 2`; `tools/test_host.sh:9-15`). If no user systemd bus is available, the script fails; then run uncapped and say so (AGENTS.md Rules). Check the matched test counts in the output, not just exit codes.

```
MEM=3G ./tools/test_host.sh does_not_exist        # Step 1 only: compile-error inventory, expected to fail
MEM=3G ./tools/test_host.sh yolo_3d_cfg
MEM=3G ./tools/test_host.sh yolo_3d_label
MEM=3G ./tools/test_host.sh yolo_3d_truth
MEM=3G ./tools/test_host.sh yolo_3d_geometry      # increment 2
MEM=3G ./tools/test_host.sh yolo_3d               # full new set
MEM=3G ./tools/test_host.sh axis_layout           # A1 + DPT + new A2 tests stay green
MEM=3G ./tools/test_host.sh dpt_head              # DPT unaffected
MEM=3G ./tools/test_host.sh weights               # checkpoint regression
MEM=3G ./tools/test_host.sh yolo                  # existing YOLO/OBB/train/decode regression (D8 guards)
MEM=3G ./tools/test_host.sh map                   # evaluator regression
MEM=3G ./tools/test_host.sh onnx                  # export regression (no export.rs change)
rustfmt --edition 2024 --check <each changed .rs file>
git diff --check
git status --short                                # confirm the Do-not-touch files show only the user's prior edits
```

Not run: training, `./tools/test_oxide.sh`, `cargo test --test cfg_corpus`, and the unfiltered host suite. The baseline is known non-green (AGENTS.md Current State).

## Bookkeeping

After each code commit, make a separate `docs(rust): record v0.1.01N yolo_3d …` commit.

- **`implemented.md`.** Add a new section with one evidence line: the tests run, their counts, and "hand-derived from source, no compiled-C++ golden; GPU status: none (Stage 2)".
- **`STATUS.md`.** Add a "Latest increments" entry, and record the evidence gaps: CLI wiring, the augmentation loader, and the native GPU.
- **`todo.md`.** Tick the item, and add the follow-ups: CLI `detector train/map` for yolo_3d, and Stage-2 GPU.
- **`version-map.tsv`.** Add the row `idx<TAB>v0.1.01N<TAB>-<TAB><full sha>`.
- **`plan-open-items.md` and `AGENTS.md` Open Work.** Mark `[yolo_3d]` done only after v0.1.016.
- **`git add -f`.** Use it for the local-only files `implemented.md`, `version-map.tsv`, `yolo_3d_notes.md`, `yolo_3d_plan.md`, and `plan-open-items.md`.
- **STATUS.md conflict.** `STATUS.md` currently carries uncommitted user edits. If it still does at bookkeeping time, **stop and ask** before committing, rather than mixing the user's hunks into the docs commit.
- **Other edits.** Never `git add -A`. Clean up any `.weights`/temporary files the tests created.

## Stop criteria

Stop, record the evidence in `yolo_3d_notes.md`, and ask the user before continuing if any of these happens:

1. `depth_3d` is missing, or `git rev-parse depth_3d` ≠ `9c9e0b4b…` and the cited ranges moved materially.
2. Any of S3, S4, S5, S6, S7, or S9 cannot be determined from the source. These define layouts and the loss, so guessing is not allowed.
3. S12 shows that metric mode needs truth the source loader never produces, or that the metric loss is dead code. Report it; do not invent truth. Only the metric sub-feature stops; it stays rejected with `InvalidConfig`.
4. A source delta has no identifiable loss, so a Rust-mode exact derivative is undefined (D5). Propose a named loss and wait for approval.
5. Tilt or ortho need a quantity that is non-differentiable everywhere in a way not covered by the D5 fixed-mask rule.
6. The compile inventory requires edits to `onnx/export.rs`, `onnx/proto.rs`, `Cargo.toml`, or `Cargo.lock`.
7. An existing regression filter (`yolo`, `map`, `weights`, `axis_layout`, `dpt_head`, `onnx`) fails in a way the D8 guards cannot explain.
8. Any request or temptation to run training, Oxide tests, or `cfg_corpus`, or to claim C++ parity.

## Self-review

- **Coverage.** Every bullet of the audit's "Plan gate" is covered: stale header (D2), staging (D1), truth/detection/mAP (D7–D11), optimizers (D12), and GPU rejection (D13). All five of the draft's open decisions are resolved: types (D7), staging (D1), key mapping (S2/S11/S13 with gates), flags/dimensions flow (D9, `Yolo3dImageMeta`), and GPU/Cpp boundary (D5, D13).
- **Honest gap.** The pinned source files were retrieved and S1–S13 are transcribed in `yolo_3d_notes.md`; source-only tests remain hand-derived and there is no compiled-C++ golden. The file records strict Rust input/config validation where C++ silently clamps, falls back, or leaves values unchecked.
- **Consistency.** All names (`Yolo3dSpec`, `LayerKind::Yolo3d`, `yolo_3d_heads`, `compute_yolo_3d_training_loss_with_state`, `compute_mixed_2d_3d_training_loss`, `decode_yolo_3d_network_detections`, `Yolo3dProjectedMapReport`) are defined once in [Public API sketch](#public-api-sketch). Versions v0.1.015 and v0.1.016 follow `version-map.tsv` idx 288 = v0.1.014.
- **Minimal impact.** There are no new `LayerParams` variants and no edits to `weights.rs`, `initialize.rs`, `muon_runtime.rs`, ONNX, or the CUDA crates. Existing 2D consumers are untouched apart from explicit guards.
