# Porting progress — durable state

Read this first after a context compaction. It records what is verified and what
is next. The authoritative plan is `tasks/rust/layers.md`; the FP8/FP4 plan is
`tasks/rust/fp8_fp4.md`; the **full implemented inventory is
`tasks/rust/implemented.md`** (this file no longer repeats it).

## Contents

- [Ground rules](#ground-rules-do-not-violate)
- [User decisions](#user-decisions-2026-10-04--all-open-questions-answered-yes)
- [Implemented](#implemented)
- [Verification commands](#verification-commands)
- [Next steps](#next-in-dependency-order-nothing-blocked)
- [Known evidence gaps](#known-evidence-gaps-not-user-questions)
- [Task plans not started](#task-plans-not-started)

## Ground rules (do not violate)

- Every cargo command uses `-j 4` or fewer. `tools/test_host.sh` and
  `tools/test_oxide.sh` use `-j 2`.
- `src/darknet-cuda-oxide` is **not** a workspace member; build/test it with
  `./tools/test_oxide.sh` (which wraps `cargo oxide`).
- Backward is always the **exact derivative of the Rust forward**. C++
  deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Commit only the files a change names.

## User decisions (2026-10-04) — all Open Questions answered YES

"port everything in rust". Recorded in full in `tasks/rust/layers.md`
("Open questions for the user — RESOLVED 2026-10-04"). Short version:
ship the cfgs, port LAYA, port the `flash_attention`/`vc_attention` variants,
port the whole OBB family, build a dense-truth trainer, keep the legacy layers,
and let Rust define the swin weights order. **Nothing is blocked on a question
any more** — the remaining blockers are evidence gaps only.

## Implemented

Everything delivered so far is consolidated in
[`tasks/rust/implemented.md`](implemented.md): the cuda-oxide backend (P0–P9),
FP8/FP4, layer-port Phases 0 / 1b / 2A / 2B-CPU and the Phase 3A shared
primitives, LegoGears_v2 inference parity, the training loop + data pipeline,
and housekeeping — each with its evidence.

Latest increments:
- v0.0.251 (`968c1beb`; gate incomplete): added source-derived OBB weighted NMS. Reference is `raptor:src-lib/box.cpp` (`diounms_sort`, not `yolo_layer.cpp`): `overlap = obb_iou - (d²/c²)^beta_nms`, where `obb_iou = detection_obb_iou(i,j)`, `d²` is center-distance squared, and `c²` is the enclosing AABB diagonal squared; merge weights are `w_i = prob_i[k]` and `w_j = prob_j[k] * obb_iou`, used to average box x/y/w/h and the keeper front point. Source enum spelling is `OBB_W_NMS`; cfg aliases are `obb_w` and `obb_wnms`. The four `weighted_merge_*` tests and the cfg/multi-head-beta tests passed. Direct `cargo test -p darknet-core -p darknet-cli -j 2 -- obb_` (uncapped because `systemd-run` has no user bus) ran 132 core tests: 131 passed, 1 failed; CLI ran 3 passed. The unrelated OBB-training failure is `train::tests::obb_batch_loader_applies_configured_seeded_flips_to_images_and_labels`, which rejects geometric option `jitter`; stop here and resolve/report before advancing T1.4. No compiled-C++ parity is claimed.
- v164: OBB NMS now supports source `greedynms` (fixed 0.6 DIoU exponent) and `diounms` (configurable `beta_nms`, default 0.6), using the source AABB-based formulas. Multi-head DIoU configs with mismatched betas and non-finite `beta_nms` are rejected. The NMS scoring/cfg/multi-head tests passed; `MEM=3G ./tools/test_host.sh obb_` passed 122 core + 2 CLI tests. Weighted OBB merge remains unsupported; no compiled-C++ parity is claimed.
- v163: OBB NMS now accepts source `default` and `cornersnms` kinds and routes both through the source AABB-IoU suppression path; existing `obb` remains orientation-aware, and source `obb_nms`/`obbnms` aliases parse. The NMS-kind cfg and overlap tests passed; `MEM=3G ./tools/test_host.sh obb_` passed 120 core + 2 CLI tests. `greedynms`, `diounms`, and weighted OBB NMS remain unsupported; no compiled-C++ parity is claimed.
- v162: `detect-image` now routes `[net] obb=1` inference through the orientation-aware OBB decoder and reports the legacy front point in human output and full-precision `det_obb` TSV rows; unsupported mixed OBB/modern heads fail explicitly. The dispatch/formatting CLI test passed, `detect_` passed 2 CLI tests, and `obb_` passed 119 core + 2 CLI tests. CARPN network decoding remains unsupported; no compiled-C++ parity is claimed.
- v161: CPU OBB training now routes `-map` through the orientation-aware `ObbValidationDataset` and shared training `MapSchedule`; interval and final evaluations write `_map.csv`, update loss-chart mAP, print reports, and save `_best.weights` on improvement. The CLI test checks `-map_every 2` at iterations 2 and final 3; `detector_train_` passed 4/4 CLI tests, `map_` passed 4 CLI + 11 core tests, and `obb_` passed 119 core + 1 CLI test. CARPN validation remains unsupported; no compiled-C++ parity is claimed.
- v160: the CPU OBB trainer now writes a separate `<cfg>_obb_metrics.csv` with subdivision-aggregated average OBB IoU/GIoU, class output, assignment counts, classification/box cost components, front loss, and orientation accuracy/error buckets. The end-to-end CLI test validates the CSV schema and two-subdivision counts; `detector_train_` passed 4/4 CLI tests and `obb_` passed 119 core plus 1 CLI test. This is source-mapped telemetry, not compiled-C++ parity.
- v159: `[net] obb=1` now routes CPU `detector train` through a bounded OBB SGD loop using source OBB losses, configured subdivisions, direct resize, seeded horizontal flips, checkpoints, and loss CSV logging. It explicitly rejects `-map`, Adam, dynamic loss scaling, and unsupported augmentation; OBB GPU training is rejected. The CLI regression test covers one iteration and `-map` rejection; all 119 core OBB-filtered tests plus the CLI test passed. `cargo check -p darknet-cli --features oxide -j 4` could not complete because the existing f16 `ForwardResult` initializer at `main.rs:309` omits current cache fields; no oxide-compiled or compiled-C++ parity is claimed. OBB `-map` cadence and broader optimizer/augmentation integration remain open.
- v158: OBB delta batches now expose source-style loss cost for MSE and supported non-MSE modes. Non-MSE cost uses source best/additional anchor `1 - box_ious` loss averages (truth-threshold-only cells do not enter the count) plus objectness-normalized non-box squared deltas; per-head and subdivision score-loss sums are aggregated. The focused cost tests and all 119 OBB-filtered host tests passed. `ciou_obb_mul` training and CLI/schedule integration remain open; no compiled-C++ parity is established.
- v157: OBB truth-threshold dispatch now permits distinct matching and box-delta score configs, required for weighted MSE matching with the source CIoU-OBB regression fallback. The MSE weighted end-to-end delta test and all 119 OBB-filtered host tests passed.
- v156: OBB delta batches now report the source objectness-normalized sum of squared deltas outside the six box channels as the classification/objectness loss component, including non-MSE modes. Coverage checks default and non-unit `obj_normalizer`; all 118 OBB-filtered host tests passed. The remaining non-MSE IoU scalar and CLI/schedule integration are open.
- v155: OBB delta batches now expose the source raw `sum(delta²)` cost when every OBB YOLO head uses MSE; other loss modes return `None` rather than a misleading partial scalar. The MSE cost test and all 118 OBB-filtered host tests passed. Full non-MSE loss/CLI integration remains open.
- v154: covariance and COBB matching scores now combine using their configured normalizer weights when both are active, matching the source independent-branch matcher; CIoU-OBB override and single-branch paths are unchanged. The weighted-score test and all 117 OBB-filtered host tests passed. No compiled-C++ parity is established.
- v153: when covariance and COBB are both active, box regression now adds their independent contributions instead of returning after covariance. A combined-vs-separate gradient test and all 117 OBB-filtered host tests passed. This does not establish compiled-C++ or full-loss parity.
- v152: separate OBB matching-score and box-delta dispatch for `iou_loss=mse`: weighted covariance/COBB scores still select matches, while box regression follows source CIoU-OBB fallback instead of applying the covariance analytic gradient. Added routing and regression-path tests; all 116 OBB-filtered host tests passed. This does not establish compiled-C++ or full-loss parity.
- v151: front-aware OBB training stats now aggregate source angle-error mean/buckets (`<5°`, `<20°`, `>30°`) across assignments, heads, and subdivisions, using configured anchor offsets. The focused angle/merge tests and all 114 OBB-filtered host tests passed.
- v150: `ciou_obb_mul` is distinguished from additive CIoU-OBB during parsing; OBB training config now rejects that unsupported variant instead of silently using the additive path. Focused coverage and all 113 OBB-filtered host tests passed.
- v149: OBB `iou_thresh_kind` now parses to the spec with the source `iou` default; a configured `diou` value is covered by the adapter test. This is config retention; training-threshold semantics remain part of broader integration.
- v148: standard OBB `iou_loss` values are now retained in the parsed YOLO spec and routed into front-point delta configuration; `ciou_obb` and `ciou_obb_mul` map to CIoU while retaining their OBB score-mode flag. The focused adapter tests and all 113 OBB-filtered host tests passed.
- v147: the parsed OBB `front_loss_normalizer` now feeds covariance matching-score length weighting instead of a hard-coded 0.35. The focused adapter test and all 113 OBB-filtered host tests passed.
- v146: `[obb_yolo] iou_loss` now parses CIoU-OBB/CIoU-OBB-mul and standard IoU names; per-head config selects the CIoU-OBB match branch when requested or when covariance/COBB weighting is absent. Unsupported names fail. The focused tests and all 113 OBB-filtered host tests passed; this is score-path coverage, not full loss parity.
- v145: parsed covariance/COBB normalizers, covariance length ratio, COBB weights, and clamped QBBV vIoU weight now populate both matching and box-delta score configs. The focused parser/adapter test and all 112 OBB-filtered host tests passed.
- v144: `[obb_yolo] map=` now resolves relative class-map files and routes parsed IDs into the existing assignment config, matching the OBB source loader path. The focused parser/adapter test and all 111 OBB-filtered host tests passed.
- v143: `[obb_yolo]` now parses `counters_per_class` into bounded class multipliers and routes them through the existing per-head assignment config; malformed class counts are rejected. The focused parser/adapter test and all 110 OBB-filtered host tests passed.
- v142: added `train_obb_yolo_subdivisions`, which accumulates OBB gradients across all configured subdivisions, merges per-head/source metrics, and applies one SGD update normalized by total images. Tests cover two-subdivision accumulation and update. The focused test and all 109 OBB-filtered host tests passed; no C++ parity is implied.
- v141: added `train_obb_yolo_batch_step`, a one-subdivision CPU SGD path from an `ObbTrainingBatch` through forward, OBB delta generation, accumulated backprop, and parameter update. It rejects Adam and multi-subdivision configurations rather than silently using incorrect semantics. The focused parameter-update test and all 108 OBB-filtered host tests passed; full loss scalar and CLI/schedule integration remain open.
- v140: added `backward_obb_yolo_training`, connecting OBB output deltas to `ExecutableNetwork::backward_seeded`; the test verifies finite input gradients and nonzero convolution weight gradients. It still stops before optimizer updates and a complete OBB loss scalar. The focused backward test and all 107 OBB-filtered host tests passed; no compiled-C++ parity is established.
- v139: added `compute_obb_yolo_training_deltas`, which consumes OBB `ForwardResult` heads and labels, builds per-head configs, and returns per-layer output-gradient seeds plus aggregated stats. It explicitly requires shared focal/class options across heads and does not compute the full loss scalar or run backward/update. The focused integration test and all 106 OBB-filtered host tests passed; no compiled-C++ parity is established.
- v138: multi-head front-aware dispatch now accepts a separate front-delta config per head; the shared-config API remains as a compatibility wrapper. The multi-head test compares per-head outputs/stats and rejects a config-count mismatch. The focused test and all 105 OBB-filtered host tests passed. Network-loop/backprop integration remains open; no compiled-C++ parity is established.
- v137: added a parsed-spec adapter that binds an `[obb_yolo]` head to runtime grid/network dimensions and produces matching, assignment, box-delta, and front-delta configs, including parsed directional offsets and front-loss normalization. The focused adapter/parser test and all 105 OBB-filtered host tests passed. It does not run a network trainer or backprop; no compiled-C++ parity is established.
- v136: parsed `[obb_yolo]` `front_loss_normalizer` and its `front_point_normalizer` alias with source default `0.35` and source precedence. The focused parser tests and all 105 OBB-filtered host tests passed. Routing this setting into a network trainer remains open; no compiled-C++ parity is established.
- v135: `[obb_yolo]` now parses source `obb_anchors` in width/height-pair and width/height/offset-quadruple forms, stores finite directional offsets in `ObbYoloSpec`, rejects ambiguous/invalid forms, and retains the Rust `anchors` alias. The focused parser tests and all 105 OBB-filtered host tests passed. Offset routing into training remains open; the inference decoder remains cell-relative. No compiled-C++ parity is established.
- v134: front-point writeback now reports source-normalized loss sums/counts for explicit-front matches, composed across truth-threshold and best/additional passes and aggregated across heads. The 10 front-point-filtered tests and all 104 OBB-filtered host tests passed. This is only one loss term; full OBB loss/metric aggregation and network-loop/backprop integration remain open. No compiled-C++ parity is established.
- v133: added the multi-head front-aware training dispatcher by applying the ordered one-head composition to each head and aggregating its stats. Tests compare multi-head outputs/stats with per-head calls; all 104 OBB-filtered host tests passed. Network-loop/backprop integration remains open; no compiled-C++ parity is established.
- v132: added an opt-in one-head training composition that inserts truth-threshold and best/additional front-point deltas after their corresponding box passes; the test checks the exact order even with `accumulate=false`. The focused composition test and all 104 OBB-filtered host tests passed. Multi-head front dispatch and network-loop/backprop integration remain open; no compiled-C++ parity is established.
- v131: added truth-threshold front-point delta writeback for matched batch/head cells with class-scaled normalization and source center-chain behavior. The focused writeback test and all 103 OBB-filtered host tests passed. Composing the front passes into the full head/network training loop remains open; no compiled-C++ parity is established.
- v130: added positive-assignment front-point delta writeback for best/additional anchors, including class-scaled normalization, preserved unrelated channels, and optional center-chain updates when offsets are configured. The focused writeback test and all 102 OBB-filtered host tests passed. Truth-threshold front writeback and composed network-loop integration remain open; no compiled-C++ parity is established.
- v129: added standalone source front-point CIoU channel deltas with grid chain factors, normalizer/clipping, absent-front gating, and optional center-chain contributions when directional priors are present. The focused test and all 101 OBB-filtered host tests passed. Batch/head writeback and network-loop integration remain open; no compiled-C++ parity is established.
- v128: one-/multi-head OBB stats now expose source `net.total_bbox` counts by summing truth-threshold cells and best/additional anchor box assignments. The focused composition tests and all 100 OBB-filtered host tests passed. Raw front-channel gradients and full network-loop/loss integration remain open.
- v127: multi-head OBB delta dispatch now returns both per-head stats and source-style aggregate `avg_cat` sums/counts, threshold-positive counts, and box-positive counts. The focused multi-head test and all 100 OBB-filtered host tests passed. Raw front-channel gradients and full network-loop/loss integration remain open.
- v126: routed optional per-anchor front offsets through full-head score matching and truth/positive box deltas, including the covariance center-chain contribution; the focused training test and all 100 OBB-filtered host tests passed. cfg ingestion/inference decode and raw front-channel gradients remain open.
- v125: added standalone `decode_obb_yolo_front_point` for legacy cell-relative or source directional-anchor-prior decoding, with finite-offset validation. The focused test and all 99 OBB-filtered host tests passed. Config loading and routing these offsets through matcher/trainer/network decode remain open.
- v124: added multi-head delta dispatch across separate head tensors with shared global anchors/classes/network dimensions and per-head stats; the focused dispatch test and all 98 OBB-filtered host tests passed. Network-loop/backprop integration remains open.
- v123: composed the one-head delta passes in source order with `apply_obb_yolo_batch_head_training_deltas`, returning the anchor `avg_cat` stats and separate threshold/anchor counts. The focused composition test and all 97 OBB-filtered host tests passed. Multi-head trainer dispatch, front-point deltas, and broader losses/metrics remain open.
- v122: added `apply_obb_yolo_batch_head_truth_threshold_deltas` for per-cell truth-threshold positives, applying mapped class deltas (including the source smoothed `best_score²` override) and class-scaled box deltas while leaving the previously emitted objectness entry intact. The focused test and all 97 OBB-filtered host tests passed. Trainer sequencing, front-point deltas, and broader losses/metrics remain open.
- v121: positive-anchor box-delta writeback now scales `iou_normalizer` by the mapped class multiplier, matching the source `obb_yolo_delta_training` call for COBB and covariance branches. The focused writeback test and all 96 OBB-filtered host tests passed. Multi-head/trainer integration remains open.
- v120: `apply_obb_yolo_batch_head_positive_class_deltas` now returns source `avg_cat` sum/count contributions for its best/additional anchor assignments, while still writing class deltas. The focused stats/writeback test and all 96 OBB-filtered host tests passed. Truth-threshold metric aggregation and multi-head/trainer integration remain open.
- v119: integrated the source covariance analytic-gradient branch into `obb_yolo_box_regression_deltas` and positive batch/head box writeback, including covariance/IoU normalizer scaling and YOLO width/coordinate chain factors. Focused tests and all 96 OBB-filtered host tests passed. `center_front_anchor` and trainer/loss-loop integration remain open.
- v118: ported `box_covariance_w2_obb_loss_and_gradient` with source forward loss terms and analytic center/size/front gradients. Analytic gradients match finite-difference checks across all six inputs, and returned terms match the existing decoded-geometry loss. Focused test and all 95 OBB-filtered host tests passed. Batch delta integration remains open.
- v117: added `obb_yolo_box_regression_deltas` and per-batch/head positive box-delta writeback for the source finite-difference COBB/legacy score path, including normalizer scaling, sigmoid chain factors, clipping, and accumulation. The focused tests and all 94 OBB-filtered host tests passed. Covariance analytic gradients, multi-head trainer integration, and full loss/loop wiring remain open.
- v116: added `apply_obb_yolo_batch_head_positive_class_deltas`, applying source class-channel deltas at best/additional positive anchors with class mapping, multipliers, label smoothing, and existing-target behavior while preserving other entries. The focused writeback test and all 91 OBB-filtered host tests passed. Multi-head integration, class metrics, box deltas/losses, and trainer wiring remain open.
- v115: added `ObbYoloPositiveObjectnessConfig` and `apply_obb_yolo_batch_head_positive_objectness_deltas` for source best/additional positive-anchor overrides, including class mapping/multipliers and smooth-mode preservation of nonzero deltas. Tests verify mask slots, assignments, smooth-mode behavior, and bad-map rejection; all 90 OBB-filtered host tests passed.
- v114: added `apply_obb_yolo_batch_head_objectness_deltas`, writing source-derived negative/ignore/positive objectness deltas into entry 6 of the per-head delta tensor while preserving all other entries. Tests cover positive/negative batch slots, exact write locations, and delta-shape rejection; all 88 OBB-filtered host tests passed. Class/box delta writeback and full trainer integration remain open.
- v113: integrated source objectness-delta generation into batch-cell matching and full head traversal, using the output objectness channel, `obj_normalizer`, smoothing, and positive/ignore gate precedence. Batch/head-focused tests and all 86 OBB-filtered host tests passed. Class/box deltas and full trainer integration remain open.
- v112: added `ObbYoloHeadMatchConfig` and `visit_obb_yolo_batch_head_matches` to traverse a complete batch-major head tensor in source batch/cell/anchor order with bounded callback memory, sanitize consumed prediction/class channels, and emit per-cell match gates. Tests cover score/channel extraction, batch tags, traversal ordering, and malformed tensor length; all 86 OBB-filtered host tests passed. Multi-head trainer dispatch, deltas, losses, and optimizer integration remain open.
- v111: added `match_obb_yolo_batch_cell_predictions` and tagged batch/head/anchor/cell match records, mapping caller-selected cells to the correct image truths and preserving input order with per-cell score config. Focused tests cover distinct batch truths, order, thresholds, and invalid batch indices; all 83 OBB-filtered host tests passed. Full head-tensor traversal, loss deltas, and trainer integration remain open.
- v110: added `ObbYoloHeadAssignmentSpec`, batch/head-tagged positive records, and `enumerate_obb_yolo_batch_positives` in source head-then-batch order. Tests verify primary/additional anchors, local truth indices, per-head masks, order, and empty-input rejection; all 81 OBB-filtered host tests passed. Prediction-conditioned batch matching and full loss/trainer integration remain open.
- v109: added `match_obb_yolo_cell_prediction_with_score_config`, composing per-truth covariance/COBB/angular branch scores with source best-truth, class-gated ignore, positive thresholds, truth order, and empty-row sentinel for one cell. All 79 OBB-filtered host tests passed. Batch/multi-head assignment and training integration remain open.
- v108: added `ObbYoloMatchScoreConfig`/`ObbYoloMatchScoreBranch` and `obb_yolo_match_score`, dispatching one cell through covariance-first, then COBB, then legacy-angular source branch priority. All 78 OBB-filtered host tests passed; batch/multi-truth assignment and training integration remain open.
- v107: added `obb_yolo_cobb_prediction_match_score_with_viou`, decoding one cell's six post-YOLO channels, enforcing covariance-priority/COBB branch gates, and applying the source COBB plus QBBV score. The raw-match test and all 77 OBB-filtered host tests passed. Batch assignment/loss integration and compiled-C++ parity remain open.
- v106: added source COBB decoded-geometry loss/quality helpers (`ObbCobbLossConfig`, `ObbCobbLossTerms`, `box_cobb_obb_loss_terms`, and QBBV-blended quality), including candidate type selection, analytic score rows, Smooth-L1 terms, and invalid-geometry fallback. Two focused COBB tests and all 76 OBB-filtered host tests passed. Raw-channel decode/training integration and compiled-C++ parity remain open.
- v105: added `box_ciou_angular_quality_score` and `obb_yolo_angular_prediction_match_score_with_viou` for the source legacy weighted-branch fallback, including raw-channel cell decoding and QBBV blending. Focused geometry/decode tests passed; all 74 OBB-filtered host tests passed. COBB scoring/integration and compiled-C++ parity remain open.
- v104: added `blend_obb_score_with_qbbv_viou` and `obb_yolo_covariance_prediction_match_score_with_viou`, applying the source weight clamp, default 24×24 sampling, and non-finite sanitization to the covariance match branch. QBBV blending and raw covariance integration tests passed; all 72 OBB-filtered host tests passed. Multi-branch/other-branch integration and compiled-C++ parity remain open.
- v103: added `obb_qbbv_heatmap_value` and `box_obb_qbbv_viou`, porting the source-oriented tent heatmap and center-sampled squared-heatmap intersection-over-union (sample count clamped to 4–128). Two focused QBBV tests and all 71 OBB-filtered host tests passed; score blending and compiled-C++ parity remain open.
- v102: added `obb_yolo_covariance_prediction_match_score`, decoding one cell's post-YOLO channels and scoring them against a truth with the covariance-W2 branch. The branch gate, direction-free front placeholder, and anchor/grid decode are covered. One focused test passed; weighted covariance/COBB aggregation, QBBV-vIoU blending, and compiled-C++ parity remain open.
- v101: added `box_covariance_w2_obb_loss_terms` and `box_covariance_w2_obb_quality_score` for already-decoded OBB geometry, porting the source covariance, center, and front-length terms with source epsilons/clamps. Two focused formula tests passed; raw matching decode, QBBV-vIoU blending, COBB scoring, and compiled-C++ parity remain open.
- v100: added `obb_yolo_class_deltas`, a source-derived class-channel delta primitive for positive OBB cells, including the existing-delta gate, focal/non-focal branches, label smoothing, optional class weights/normalization, and average-class output. Four focused class-delta tests and all 66 OBB-filtered host tests passed. Full training/loss integration and compiled-C++ parity remain open.
- v99: added `match_obb_yolo_cell_prediction` to compose per-truth CIoU scoring with the source best-truth, class-gated ignore, and positive-threshold gates for one cell. It preserves truth order, first-on-tie selection, and the `truth.x == 0` empty-slot sentinel. All 62 OBB-filtered host tests passed; weighted branches, batch/positive-anchor integration, losses, and compiled-C++ parity remain open.
- v98: added `obb_yolo_ciou_prediction_match_score`, decoding one cell's post-YOLO six-channel output for the source CIoU/default-score branch and passing its normalized geometry to v97's quality metric. Legacy/new-coordinates, nonzero-cell, absent-front, and invalid-input cases are covered. All 60 OBB-filtered host tests passed; weighted covariance/COBB scoring, batch-loop integration, and compiled-C++ parity remain open.
- v97: added `box_ciou_obb_quality_score`, matching the source `box_ciou_obb_loss_terms` score for already-decoded OBB geometries: rotated IoU, CIoU center/aspect penalties, and the cosine half-angle orientation factor. Two focused source-formula tests passed; raw-channel matching, covariance/COBB scoring, and training integration remain open. No compiled-C++ parity is claimed.
- v96: added `obb_yolo_cell_objectness_delta` to translate the v93 per-cell match selection into the source OBB YOLO negative, ignore, and positive objectness deltas. It preserves the smoothed ignore max, positive-target override, non-finite output/delta sanitization, and leaves raw score calculation plus class/box loss and trainer integration open. All 55 OBB-filtered host tests passed; no compiled-C++ parity is claimed.
- v95: audited the fp16 convolution-backward TODO against `mixed.rs` and the real-device `oxide_amp` suite. AMP convolution dgrad uses f16 GEMM with f32 accumulation before f16 storage; wgrad uses f16 inputs with f32 accumulation and returns f32 weight gradients. All 11 `oxide_amp` tests passed, including grouped and multi-chunk convolution. This closes the tracked fp16 dgrad/wgrad follow-up; half2 elementwise remains unnecessary.
- v94: added `carpn_detector_map` for normalized CARPN quadrilateral detections
  and front-point OBB labels. It reuses oriented polygon IoU and the shared AP
  report/matching policy, rejects non-CARPN predictions and missing-front labels,
  and reconstructs ground-truth vertices in the source BoxOBB order. The
  evaluation tests passed 12/12 (including all three CARPN tests); the OBB-focused
  suite passed 53/53. CARPN network
  decoding/validation integration and compiled-C++ parity remain open.
- v93: added `select_obb_yolo_cell_truth_match` for validated single-cell
  selection from precomputed source-style scores. It preserves the first best
  truth, gates ignore matching on any class output strictly above 0.25, and
  reports strict `ignore_thresh` and `truth_thresh` outcomes. Focused OBB tests
  passed 53/53. Raw prediction-match score math, delta/loss generation, trainer
  integration, and compiled-C++ parity remain open.
- v92: added `enumerate_obb_yolo_head_positives` with typed best-anchor and
  additional-anchor matches. It preserves truth and global-anchor order, emits
  extra masked anchors only under the source `iou_thresh < 1` and strict score
  comparison, and can return qualifying extras even when the global best is not
  in the head mask. Focused OBB tests passed 49/49. Prediction-conditioned
  ignore/truth-threshold matching, loss/deltas, trainer integration, and
  compiled-C++ parity remain open.
- v91: added `ObbTrainingBatch::encode_truths` and public seven-float OBB YOLO
  truth packing in the source `[batch][box-slot][7]` layout. Rows preserve label
  order, use the center placeholder for absent front points, and zero-pad unused
  slots; invalid shapes/classes/capacity fail explicitly. Focused OBB tests passed
  44/44. The package-wide host test attempt timed out after 20 minutes in the
  existing `laya_scorer::real_weights` test, so its final result is inconclusive.
  Complete anchor assignment, OBB losses/trainer integration, and compiled-C++
  parity remain open.
- v90: `DetectorAugmentation` now accepts `[obb_yolo]` cfg sections and exposes
  a deterministic per-sample horizontal-flip mask matching its existing Rust
  augmentation draw order. Added `load_obb_training_batch_with_flips_from_paths`,
  which applies the configured flips to resized NCHW images and labels together.
  Crop/resize jitter, HSV, mosaic, training/loss integration, and C++ RNG parity
  remain unsupported. The focused OBB suite passed 42/42 and all 465
  `darknet-core` library tests passed.
- v89: added public direction-free OBB YOLO anchor-match scoring and
  best-anchor/cell assignment primitives in `obb_training.rs`. Matching compares
  normalized anchor size with truth size; global-best selection, first-on-tie
  ordering, head-mask handling, and cell clamping follow the audited source.
  This is not a complete OBB truth assignment/loss/training path, and no
  compiled-C++ parity is claimed. The focused OBB suite passed 40/40 and all 463
  `darknet-core` library tests passed.
- v88: added `ObbTrainingBatch::apply_horizontal_flips`, which reflects selected
  NCHW samples and their OBB labels together using a validated per-image mask.
  The focused OBB suite passed 36/36 and all 459 `darknet-core` library tests
  passed. Flip selection remains caller-owned; config randomness and crop/mosaic
  augmentation are still open, with no compiled-C++ parity claim.
- v87: added public `flip_obb_labels`, an exact horizontal reflection of label
  centers, front points, and stored bounds that preserves OBB dimensions and
  absent-front state. The focused OBB suite passed 35/35 and all 458
  `darknet-core` library tests passed. This is label geometry only: loader
  integration and crop/mosaic transforms remain open; no compiled-C++ parity is
  claimed.
- v86: added typed `[obb_yolo] nms_kind` parsing with source default `obb`,
  case-insensitive explicit `obb`, and an explicit error for unwired alternate
  kinds. The OBB decoder dispatches the configured oriented kind to its
  orientation-aware NMS; focused OBB tests passed 33/33 and all 456 `darknet-core`
  library tests passed. No compiled-C++ parity is claimed.
- v85: added a standalone CPU `detector map` command with optional weights,
  confidence/IoU/NMS thresholds, and letterbox preprocessing. It routes standard
  graphs through `ValidationDataset` and OBB graphs through
  `ObbValidationDataset`; `darknet-cli` tests passed 19/19 and temporary standard
  and OBB cfg/data command smoke tests both completed. Training `-map` cadence,
  GPU evaluation, CARPN mAP, and compiled-C++ parity remain out of scope.
- v84: added `ObbValidationDataset`, including OBB-only head/class validation,
  OBB label preflight, one inference/decode per image, pixel-to-normalized
  conversion for boxes/front points, and orientation-aware AP evaluation. The
  existing axis-aligned evaluator still refuses OBB graphs; the CPU CLI route was
  added in v85, while training `-map` cadence and CARPN mAP remain open. The focused
  OBB host suite passed 33/33 and all 456 `darknet-core` library tests passed; no
  compiled-C++ parity is claimed.
- v83: added `obb_detector_map` for normalized legacy front-point predictions,
  using orientation-aware matching and existing AP/report semantics. Five-field
  labels are treated as axis-aligned truth; CARPN polygon detections are rejected
  until their truth-vertex encoder is wired. This is a standalone scorer, not
  network/CLI evaluation. The focused OBB host suite passed 31/31 and all 454
  `darknet-core` library tests passed; no compiled-C++ parity is claimed.
- v82: added a separate OBB training-batch type/loader using direct image resize
  and the existing five/seven-field parser. It preserves optional front-point
  state; class IDs are bounded by the smallest OBB head and optional data class
  count. Augmentation and OBB losses remain unsupported. The focused OBB host
  suite passed 29/29 and all 452 `darknet-core` unit tests passed; no compiled-C++
  parity is claimed.
- v81: wired `[net] obb=1` and `[obb_yolo]` cfg parsing into a typed
  six-coordinate head with filter/mask/anchor validation; CPU/runtime head
  collection feeds a network-level multi-head decoder and orientation-aware
  NMS. AABB YOLO training and mAP explicitly reject OBB networks rather than
  silently using axis-aligned semantics. The focused OBB host suite passed
  26/26, and all 449 `darknet-core` unit tests passed. Batch
  loading/augmentation, OBB truth assignment/loss, NMS-kind config wiring, and
  mAP remain open; no compiled-C++ golden/parity claim.
- v80: extended `ObbLabel` with source-clamped AABB bounds and added a
  source-derived normalized-center label rotation utility. Explicit front points
  rotate oriented rectangle corners while preserving OBB sizes; missing or
  degenerate front vectors use the source AABB fallback. The focused OBB suite
  passed 21/21, including both rotation paths and the source no-op threshold.
  Image augmentation/loader/training integration and compiled-C++ parity remain
  open.
- v79: added standalone OBB label-file parsing for five-field box rows and
  seven-field rows with explicit front points. The parser tracks `has_front`,
  defaults absent front coordinates to the center placeholder, clamps finite
  fields to `[0,1]`, and validates class bounds and positive dimensions. The
  focused OBB suite passed 17/17. This is not `TrainingBatch`, augmentation,
  truth assignment, OBB loss, or mAP integration; no compiled-C++ parity claim.
- v78: added a source-derived per-head legacy `[obb_yolo]` decoder for six box
  channels, objectness, class scores, and raw linear front offsets. It handles
  `new_coords`, `scale_x_y`, strict objectness/class thresholds, output
  sanitization, integer-rounded C++ letterbox dimensions, and image-pixel front
  points; OBB NMS is deliberately separate. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb` suite passed 12/12,
  covering logistic and `new_coords` decode, letterbox correction, integer
  resize rounding, malformed shapes, geometry, and NMS. At the v78 point this
  remained a standalone primitive; cfg/runtime integration was added in v81.
  No compiled-C++ golden or parity claim is available.
- v77: added an OBB detection wrapper/geometry enum, CARPN-vs-front-point IoU
  dispatch, and classwise greedy OBB NMS with descending per-class confidence,
  exact-zero skip behavior, and strict `IoU > threshold`. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb` suite passed 8/8, including
  orientation-sensitive suppression, mixed CARPN/front-point dispatch, class
  isolation, and zero-objectness/zero-score cases. This is not decoder, cfg, or
  dataset/mAP integration; no compiled-C++ golden or parity claim is available.
- v76: added host-only CARPN convex-quad IoU for four interleaved vertex pairs,
  reusing the pure-Rust convex clipper and rejecting degenerate/non-finite inputs.
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh carpn_polygon_iou` passed 2/2
  for hand-derived overlap, containment, disjoint, diamond, reversed-winding, and
  invalid cases. This is geometry only; CARPN detection dispatch, OBB NMS/mAP,
  cfg/layer integration, and losses remain open. No compiled-C++ golden/parity
  claim is available.
- v75: added host-only vertices and convex IoU for legacy front-point OBB rectangles,
  with source-ordered corners and explicit invalid-geometry behavior. The focused
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh obb_geometry` tests pass 3/3,
  covering rotated/partial overlaps, containment, disjoint boxes, and malformed
  inputs. This is only a geometry prerequisite: OBB cfg/layer/data loading,
  CARPN explicit polygons, losses, NMS, and mAP remain open. No compiled-C++
  golden or parity claim is available.
- v74: EML Adam now applies both convolution branches and BN scales on-device,
  retains per-branch moments, and supports deferred host synchronization. Adam
  moments are persisted in additive DNOPT5 sidecars while DNOPT1–4 remain
  readable; DNOPT3 EML host moments migrate to device state, and host fallback
  remains active when another layer requires CPU backward. Focused host EML tests
  pass 12/12, optimizer-state tests pass 5/5, and `oxide_runtime_chain` passes
  20/20, including no-BN Adam update parity, BN scale updates, deferred sync,
  and optimizer-state output. No compiled-C++ golden is available; CPU EML BN still
  uses rolling statistics while native GPU training uses batch statistics, so
  CPU/C++ training parity is not claimed.
- v73: EML SGD now accumulates per-branch weight/beta/BN-scale gradients and
  momentum buffers on-device, applies both branch updates through the backend,
  and supports deferred host synchronization. Device finiteness checks cover
  EML branch and BN-scale buffers. Focused host EML tests pass 9/9; the full
  Oxide `oxide_runtime_chain` suite passes 19/19, including two-step momentum
  parity, BN/no-BN updates, and deferred sync. At that point Adam remained
  host-side for EML. No CPU/C++ BN-training parity claim.
- v72: Batch-normalized EML now stays on native Oxide forward/backward: each
  branch retains batch-stat BN caches, computes exact scale/beta gradients and
  convolution adjoints, and applies device-produced rolling EMA stats to host
  params. Gradients still accumulate into host train buffers and optimizers
  remain host-side. Focused host EML tests pass 3/3; the real Oxide
  `oxide_runtime_chain` suite passes 19/19, including both branch EMA updates,
  finite BN gradients, and host optimizer/cache refresh. CPU EML training still
  uses rolling statistics while native GPU training uses batch statistics, so
  CPU/C++ training parity is not claimed. Device optimizers and a compiled-C++
  golden remain open.
- v71: No-BN EML GPU backward now uses backend EML adjoints and both branch
  convolution input/weight-gradient ops, adds the eligible residual gradient,
  and accumulates parameter gradients into host train buffers for SGD/Adam.
  BN EML retains the explicit CPU backward fallback; host EML gradients now
  participate in dynamic-loss-scale finiteness checks. `oxide_runtime_chain`
  passes 19/19, including two accumulated no-BN backwards and BN fallback.
  Optimizer updates remain host-side; no C++ parity claim is made.
- v70: EML f32 GPU forward now runs both branch convolutions, folded rolling-stat
  BN parameters, bias addition, and the combine op through the backend; both
  branch parameter blocks use separate per-owner device-cache slots. CPU branch
  caches remain available for the explicit CPU backward/optimizer fallback.
  `./tools/test_oxide.sh --test oxide_runtime_chain` passed 19/19, including BN
  and no-BN EML outputs plus device-cache invalidation after host updates.
  Native backward/optimizer and C++ batch-stat/
  EMA training behavior remain open; this is not a C++ parity claim.
- v69: Integrated `[eml_convolutional]` cfg/shape parsing, two branch-convolution
  parameter and training-buffer blocks, C++ `input_layer` then `self_layer`
  `.weights` order, CPU forward/backward, SGD/Adam updates, and an explicit GPU
  host-fallback path. Focused EML host tests pass 3/3, including finite-difference
  parameter gradients and FakeBackend CPU/GPU-fallback SGD parity; the EML Adam
  fallback test passes. The EML weights round-trip is covered by
  `recurrent_weight_blocks_round_trip_in_darknet_order`. Remaining: native Oxide
  whole-layer execution and batch-stat/rolling-EMA BN training behavior; the
  existing host convolution helper uses rolling-stat BN. The standalone full
  `cfg_corpus` run exceeded a 40-minute timeout without a final result. No
  compiled-C++ golden is available, so no C++ parity claim is made.
- v68: Added host-reference and native Oxide f32 EML branch-combine forward
  and backward ops with optional residual, shape checks, exact branch
  derivatives, and the source-derived nonpositive clamp/epsilon defaults.
  `MEM=3G PKGS=darknet-cuda ./tools/test_host.sh` passed 12/12; real-device
  `oxide_layers_extra` passed 24/24. Config/layer/runtime/optimizer wiring and
  compiled-C++ golden parity remain open.
- v67: Added `eml_conv_math.rs`, a source-derived pure reference for the
  `[eml_convolutional]` two-branch combine, stable softplus/sigmoid, optional
  residual, and exact default branch gradients. Hand-reference, shape/default,
  strict-boundary, and finite-difference tests passed (4/4). This is only the
  math prerequisite; cfg/layer/runtime/backend/optimizer support and compiled-C++
  parity remain open; no compiled-C++ golden was available.
- v66: Enabled `normalize_channels`, `normalize_channels_softmax`, and
  `normalize_channels_softmax_maxval` on convolution layers in CPU and f32 GPU
  inference/training forward/backward. Exact gradients remain default;
  `DARKNET_GRAD_MODE=cpp` selects the source-derived CUDA diagonal shortcut.
  Other layer types and f16 inference reject the activation explicitly. The
  FakeBackend CPU/GPU training parity test, Oxide chained forward/backward/SGD
  (including compact inference), C++-mode Oxide runtime, and f16 rejection tests
  pass. This is source-derived runtime parity, not a compiled-C++ golden.
- v65: Added shape-aware host-reference and native Oxide primitives for
  `normalize_channels`, `normalize_channels_softmax`, and
  `normalize_channels_softmax_maxval`, including exact normalization-chain
  backward and the legacy C++ CUDA diagonal shortcut when requested. Source
  inspection found C++ CPU and CUDA gradients differ; `cpp_grad` intentionally
  follows the CUDA kernels. Direct host finite differences/parity and Oxide
  forward/backward/C++ tests pass; full `oxide_layers_extra` passed (23/23).
  At the v65 boundary, cfg parsing/runtime integration remained open; v66 below
  closes the supported f32 convolution path without claiming compiled-C++ parity.
- v64: Weighted-shortcut backward now assembles per-source `delta * activation`
  products and applies exact None/ReLU/softmax normalization-chain derivatives
  through `F32TensorOpsBackend`; `GradMode::Cpp` still returns raw products.
  Oxide transfers only the final per-input/per-slot weight-gradient vector to
  host train buffers; parameter updates remain host-side. Host finite-difference
  math passed (4/4), FakeBackend runtime parity passed (1/1), the direct Oxide
  product/normalization/C++-mode test passed (1/1) and full `oxide_layers_extra`
  passed (22/22), and real-device runtime forward/backward/SGD parity passed
  (1/1), including short-source prefixes.
- v63: Audited convolution `grad_centr` against the C++ GPU update path and
  removed it from the open parity gaps. `gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate`
  passed (host 1/1), and `gradient_centralization_matches_reference_and_is_noop_off_warp_sizes`
  passed on real Oxide (1/1), covering BN gating, per-filter subtraction, and
  C++'s non-warp-size skip.
- v62: Replaced approximate mosaic pasting/box remapping with the source-level
  C++ quadrant crop alignment, flip-aware shifts, optional corner relocation,
  integer-pixel box clipping, and strict filtering. Rust RNG remains deterministic
  but not C++-identical; no compiled-C++ golden claim. Augmentation/loader host
  filter passed (12/12), including OpenCV single-image fixtures and hand-derived
  mosaic pixel/box geometry tests.
- v61: BN RNN/LSTM/CRNN scale-gradient reductions now stay on backend using
  existing centering, channel-scale, pointwise-multiply, and channel-reduction
  ops; Oxide no longer transfers raw projection/convolution outputs and deltas to
  host just for dscale. Forward/cache, rolling-stat update, and optimizer behavior
  remain unchanged. Host recurrent filters passed (RNN 7/7, LSTM 3/3, CRNN 3/3),
  all three real Oxide BN recurrent parity tests passed (3/3), and full real
  Oxide `oxide_runtime_chain` passed (17/17).
- v60: BN CRNN backward now uses backend convolution/BPTT ops with rolling-stat
  affine input/beta derivatives; forward/cache capture, scale-gradient reductions,
  and optimizer remain host-side. FakeBackend covers three projection blocks,
  scales, rolling stats, upstream convolution updates, and loss-scale overflow;
  host CRNN tests passed (3/3), host `rnn` filter (7/7), host LSTM tests (3/3),
  and full real Oxide `oxide_runtime_chain` passed (17/17).
- v59: BN LSTM backward now uses backend gate/BPTT and projection operations,
  with rolling-stat affine input/beta derivatives; CPU forward/cache capture and
  optimizer remain host-side, and scale-gradient reductions transfer raw
  projections/deltas to host. BN CRNN remains on CPU backward. Host LSTM tests
  passed (3/3), host RNN regression tests passed (6/6), and real Oxide
  `oxide_runtime_chain` passed (16/16), including BN LSTM parameter, rolling-stat,
  and upstream-convolution parity.
- v58: BN RNN backward uses backend projection/BPTT ops and rolling-stat affine
  input/beta derivatives; scale-gradient reduction still transfers raw projections
  and deltas to host. Forward/cache capture and optimizer remain host-side; BN
  LSTM/CRNN still use CPU backward. Dynamic loss-scale checks now inspect
  host-buffered connected/recurrent gradients. Host RNN tests passed (6/6),
  including FakeBackend parameter/stat parity; focused real Oxide parity passed
  (1/1); full `oxide_runtime_chain` passed
  (15/15).
- v57: `[maxpool] maxpool_zero_nonmax=1` now stays on the GPU backward path.
  The existing kernel writes gradients only for input argmaxes, so non-argmax
  deltas are already zero. Host maxpool tests passed (8/8); focused real Oxide
  parity passed (1/1), and the full `oxide_runtime_chain` suite passed (14/14).
- v56: BN MoE router/expert SGD, including BN scale updates and momentum, now
  runs on-device. Deferred sync preserves the expert cache and copies raw params
  plus BN stats to host. FakeBackend MoE tests passed (21/21); the full real-device
  `oxide_runtime_chain` suite passed (13/13), including BN backward/SGD parity.
- v55: BN MoE router/expert Adam now updates raw weights, beta, and BN scales
  on-device; scale moments are persisted in DNOPT4 while DNOPT1-3 remain
  readable, including migration of nested host BN moments from DNOPT3. Deferred
  sync now retains device-updated BN expert caches and copies raw params plus
  BN stats back to host. FakeBackend MoE tests passed (20/20), including host-to-
  device scale-moment migration and v4 resume; optimizer-state compatibility
  tests passed (4/4). The full real-device `oxide_runtime_chain` suite passed
  (13/13), including BN Adam parity and deferred synchronization.
- v54: BN MoE expert backward is no longer routed through whole-network CPU
  fallback. The backend computes BN affine/input/weight derivatives and
  accumulates beta/scale gradients with router/expert gradients on device;
  host-side SGD/Adam consume them after flush. No-BN router/expert optimizer
  paths remain device-native. FakeBackend MoE tests passed (19/19), including BN
  gradient/SGD parity, BN Adam host fallback, and BN-scale overflow detection;
  the full real-device `oxide_runtime_chain` suite passed (12/12). The focused
  `moe_batchnorm_expert_device_backward_matches_cpu_on_oxide` test passed (1/1),
  comparing upstream conv, router, expert, and BN scale updates against CPU.
  BN expert Adam remains host-side.
- v53: moved no-BN MoE router/expert Adam updates and moment storage onto the
  device; DNOPT3 persists those moments, DNOPT1/2 remain readable, and legacy
  DNOPT2 host moments migrate on the first device update. Fixed host nested-Adam
  child resizing so later MoE experts retain moments across steps. The focused
  host MoE suite passed (18/18), host optimizer compatibility passed, and the
  full real-device `oxide_runtime_chain` suite passed (11/11), including
  `moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients`.
- Added no-BN LSTM device backward; v59 adds its BN rolling-stat projection
  derivative path using an uploaded CPU forward cache. Gate derivatives, connected projection
  gradients, and recurrent/input-gradient accumulation use backend ops. The
  training path retains BPTT caches on-device rather than downloading projection,
  gate, and cell caches, and concatenates timestep outputs directly on-device.
  LSTM and RNN forward/backward sequence slicing and timestep input-gradient
  assembly stay on-device; RNN projection gradients also accumulate on-device
  across steps, and no-BN RNN/CRNN training pre-activation/state caches now
  remain device-resident for backward (the host cache is captured only for
  host-facing forwards). CRNN forward input slicing and output assembly also
  stay on-device. CRNN backward sequence slicing and input-gradient assembly
  also stay on-device; convolution parameter gradients accumulate on-device across
  steps. Recurrent cache/state and parameter-gradient transfers, host
  accumulation/SGD, and some cache handling remain host-side. Targeted host
  RNN/CRNN tests and real-device `oxide_runtime_chain` tests
  `rnn_recurrent_projections_and_training_backward_match_cpu` and
  `crnn_recurrent_convolutions_and_training_backward_match_cpu` pass. LSTM
  device-backward parity also passes, including upstream convolution updates.
- Added an oxide Stair activation kernel matching Darknet's floor-based
  alternating-step formula and parity gradient, including negative odd intervals.
  Verified by the 391-test core library suite, `oxide_parity` (11 passed,
  including finite differences away from integer discontinuities), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (195 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. Every
  oxide test target also compiled with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide PLSE activation kernel with the C++ piecewise-linear sigmoid
  and exact ±4 endpoint gradients. Verified by the 391-test core library suite,
  `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (189 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests.
- Added an oxide Lhtan activation kernel with its exact 0.001 tail slopes and
  Darknet's endpoint gradients at 0 and 1. Verified by the 391-test core library
  suite, `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (183 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests. Every oxide test target also compiled
  with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide Relie activation kernel (leaky ReLU with slope 0.01), including
  the exact zero-point derivative. Verified by `oxide_parity` (11 passed,
  including finite differences and input/output derivative parity), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (177 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. The
  391-test core library suite also passed.
- Added an oxide HardTanh kernel with Darknet's inclusive clamp and zero
  derivative at exactly -1 and 1. Verified by `oxide_parity` (11 passed,
  including finite differences and endpoint parity), `oxide_half` (13 passed, 1
  ignored), all 13 `oxide_cfg_shapes` tests (171 convolution-activation pairs),
  the real-device chained-convolution activation test, the 391-test core
  library suite, and a no-run build of every oxide test target.
- Added an oxide loggy activation kernel and exact derivative, wiring it through
  cfg parsing, f32/f16 convolution activation paths, and the real-device runtime.
  Verified by `oxide_parity` (11 passed, including finite differences and
  input-vs-output derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (165 convolution-activation pairs), all 7 real-device
  `oxide_runtime_chain` tests, and the 391-test core library suite. The oxide
  `--no-run` build compiled every test target.
- Added an oxide Hard Mish activation kernel and exact derivative, including
  Darknet's strict threshold behavior at -2 and 0. Verified by `oxide_parity`
  (11 passed, including finite differences and explicit x=-2), `oxide_half` (13
  passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (159 forward/backward
  convolution-activation pairs), and all 7 real-device `oxide_runtime_chain`
  tests. The core library suite passed (391 tests).
- Added an oxide SELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), all 13 `oxide_cfg_shapes` tests (including
  153 forward/backward convolution-activation pairs), and all 7 real-device
  `oxide_runtime_chain` tests. `cargo test -j 4 -p darknet-core --lib` passed
  (391 tests).
- Added an oxide ELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), and the real-device
  `oxide_runtime_chain::chained_conv_activation_stacks_match_cpu_backward` test.
  `MEM=3G ./tools/test_oxide.sh --no-run` also compiled every oxide test target.
- YOLO IoU/GIoU box geometry and objectness/class sigmoid-BCE deltas have
  central finite-difference coverage; the 394-test core library suite passed at
  that milestone. The box test records Darknet's legacy 0.5 width/height
  comparison factor. DIoU/CIoU deltas now have source-derived parity coverage
  for contained, edge-crossing/tied, and horizontal/vertical disjoint boxes;
  all 24 `yolo_`-filtered host tests pass. Do not use naive forward finite
  differences as their C++ reference: `box_diou` uses `(d/c)^0.6`, while `dx_box_iou`
  implements the legacy derivative of `d/c`; CIoU approximates the `alpha`
  contribution.
- f16 standalone batchnorm (via channel-affine), bias + activation, and reverse
  upsample forward are already implemented. Verified by
  `oxide_half::f16_shortcut_avgpool_affine_reverse_match_f64_reference` and
  `oxide_runtime_f16::tiny_net_with_avgpool_local_avgpool_batchnorm_reverse_upsample_matches_f32`
  (13 + 5 tests passed; 1 ignored in the half suite).
- `NORM_CHAN*` f32 convolution forward/backward is wired through CPU references
  and native Oxide ops; exact f32 adjoints are default and C++ mode uses the
  inspected CUDA diagonal shortcuts (C++ CPU custom gradients differ). The cfg
  parser permits these activations only on `[convolutional]`; f16 inference and
  other layer types reject them explicitly.
- Weighted `[shortcut]` GPU forward composes oxide channel-scale + add. Input and
  source gradients use backend elementwise multiply + shortcut-fit ops. Oxide
  assembles raw `delta * source` products and applies exact None/ReLU/softmax
  normalization derivatives on-device, including the `GradMode::Cpp` raw-product
  behavior. Only the final per-input/per-slot weight-gradient vector transfers
  to host train buffers; parameter updates/SGD remain host-resident. FakeBackend
  and real-device
  `weighted_shortcut_runtime_matches_cpu_forward_backward_and_sgd` parity pass.
- AMP runtime with dropout and dropblock now has real-device output/mask/backward/
  SGD comparison against f32 (`oxide_runtime_chain` test).
- Deferred optimizer accumulation has a real-device `subdivisions=2` CPU parity
  test across two iterations, including pending-gradient flush.
- Device convolution params support per-owner expert slots for container reuse;
  warm-up and host-update invalidation are covered by the
  `container_convolution_cache_is_keyed_by_owner_and_expert_slot` FakeBackend test.
  MoE computes router pooling/logits/top-k softmax, expert convolutions, and
  per-sample gate blending on Oxide. No-BN router connected-backward,
  device-selected gate columns, expert gate-dot reductions, load-balance plus
  top-k softmax-Jacobian backward, router-pool input-gradient broadcast, and
  expert convolution backward use backend ops. Gates and pooled/expert
  output/activation caches stay device-resident; no gate/dot values cross the
  host in normal Oxide training. Router/expert parameter gradients accumulate
  on-device across backward calls. For SGD, no-BN router/expert parameter
  updates plus momentum run on-device without gradient transfers. Router values
  sync to host after each step for the next forward; expert values support
  deferred host sync. No-BN MoE Adam updates also run on-device without gradient
  transfers; DNOPT3 persists their moments and legacy DNOPT2 host moments migrate
  on first use. DNOPT4 adds BN MoE scale moments; DNOPT1-3 remain readable, and
  nested host BN moments migrate on the first BN Adam device update. BN expert
  backward uses backend ops; BN expert SGD updates weights/beta/scales with
  device-resident momentum, and BN Adam uses DNOPT4 scale moments. The dynamic-
  loss-scale finiteness check covers dirty router/expert/BN-scale buffers.
  FakeBackend
  parity and real-device
  `moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  pass, including two backward calls accumulated before the first deferred SGD
  step, followed by a second step with upstream convolution updates. Real-device
  `moe_batchnorm_expert_device_backward_matches_cpu_on_oxide` also checks BN
  expert gradients and updates against CPU, including upstream convolution
  updates. `moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients`
  checks BN expert/scale Adam and deferred synchronization. The FakeBackend test
  `gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check` verifies
  dirty router/expert/BN-scale accumulators are checked for overflow; FakeBackend
  SGD parity also checks router momentum residency and `dont_update` gradient
  retention. `oxide_layers_extra`
  verifies soft/tied top-k, pool and pool-backward, non-multiple gate-dot
  reductions, and router backward parity with load balancing.
- Overlapping-window maxpool backward, including `maxpool_zero_nonmax`, matches
  central finite differences; real-device maxpool forward/backward/SGD parity
  also passes for `maxpool_zero_nonmax=1`.
- No-BN RNN and CRNN forward/backward use backend ops: RNN projections and
  CRNN recurrent convolutions, activation gradients, state accumulation, and
  shortcuts. No-BN LSTM forward runs gates/cell products through backend ops,
  including `vector_mul_f32`; its backward now uses a backend gate-derivative
  operation plus connected ops for projection gradients, time-accumulated
  parameters, and recurrent/input-gradient accumulation. Device training keeps
  BPTT caches resident instead of downloading projection/gate/cell caches,
  and timestep outputs are concatenated on-device. RNN/LSTM forward/backward
  sequence slicing and timestep input-gradient assembly also stay on-device;
  RNN projection gradients accumulate on-device across steps; CRNN forward
  input slicing and output assembly also stay on-device. CRNN backward sequence
  slicing and input-gradient assembly also stay on-device, with convolution
  parameter gradients accumulated across steps. Recurrent cache/state and
  parameter-gradient transfers, host accumulation/SGD, and some cache handling
  still cross the host. BN RNN/LSTM backward uses backend projection/BPTT ops;
  BN CRNN backward uses convolution/BPTT ops. All use rolling-stat affine
  input/beta derivatives; scale-gradient reductions use backend centering,
  channel-scale, multiply, and channel-reduction ops (native on Oxide), so raw
  recurrent values/deltas no longer return to host for dscale. Recurrent BN
  forward/cache capture, rolling-stat update, and optimizer remain host-side.
  FakeBackend and real-device
  `oxide_runtime_chain::batch_normalized_rnn_device_backward_matches_cpu_on_oxide`,
  `oxide_runtime_chain::batch_normalized_lstm_device_backward_matches_cpu_on_oxide`,
  and `oxide_runtime_chain::batch_normalized_crnn_device_backward_matches_cpu_on_oxide`
  compare parameter and rolling-stat updates with CPU, including upstream
  connected/convolution updates. Dynamic loss-scale finiteness checks include
  host-buffered connected/recurrent gradients.
- Host Adam updates standalone batchnorm, local, connected, weighted shortcut,
  MoE fallback, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT parameters with
  distinct moment slots and per-layer rate scaling. DNOPT3 persists no-BN device
  MoE moments and host moments; DNOPT4 adds device BN scale moments. DNOPT1-3
  remain loadable, with nested BN host moments migrating on first BN device Adam
  update. The ViT reference test covers C++ weight
  order, decay groups, clipping, and reduced mHC rate; host resume has a
  multi-step continuation test.
- `[vit]` reference path: cfg/shape, C++-ordered header-aware weights, seeded init,
  CPU analytical forward/backward + SGD, and GPU f32 host fallback; 23 focused
  tests pass, targeted package check/build pass with `-j 4`.
- Host suites: `darknet-core --lib` 390/390 and `darknet-cli` 18/18 pass. The
  real-device `oxide_runtime_chain` suite passes all 7 tests.
  `cfg_corpus` started but the combined host run hit its 30-minute command timeout;
  this is not evidence of a code failure. Historical `raptor-fp4` C++ commit
  `ceb7c974` was located for MSA and flash/VC source audit.
- `[moe_convolutional]` CPU path (commit `cdf68d00`) — 13 tests green.
- `[laya_marker_gather]` — 10 tests; `[laya_act_head]` — 9 tests;
  `[laya_loss]` — 11 tests; `[laya_encoder]` — 9 tests; `[laya_head]` — 8 tests;
  `darknet-core --lib` 354/354.
- Docs reconciled: `todo.md` now tracks open work only; `eml_convolutional`
  save order verified; `onnx.md` / `gui_iced_plan.md` marked not started.

## Verification commands

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G cargo test -j 4 -p darknet-core --lib segmentation   # the fast inner loop
MEM=3G ./tools/test_oxide.sh --test oxide_gemm_f8  # ~50 s
MEM=3G ./tools/test_oxide.sh                      # ~15 min, 18 suites
```

`cfg_corpus` needs ~145 s **alone**; five unrelated cfgs hit their 300 s
per-cfg timeout when the suite shares the machine, so run it on its own before
believing a `cfg_corpus` failure.

## NEXT (in dependency order, nothing blocked)

1. The LAYA family is **complete** (`laya_scorer`, `laya_marker_gather`,
   `laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head`). The `[vit]`
   reference path is now wired; remaining Phase 3A work is flash/vc-attention
   fast paths and `swin`/`transformer`, `tucker_attention`, `mambavision`,
   `kimi_kda`, `minimax_sparse_attention` on the shared LayerNorm/GEMM set.
   **Order matters:** `[laya_scorer]`'s `weights=` is only usable together with
   `[laya_encoder]` + `[laya_head]` (the shared-file offset is computed past
   their blobs), so those two come next even though the plan files them under
   Phase 4. They also need the Phase 3A prerequisite kernel set.
2. Phase 2A is done, including `detect::decode_modern_detections` and the
   `detect-image` merge. Phase 2B `moe_convolutional` has a verified CPU path
   and partial Oxide forward/backward; no-BN router/expert SGD and Adam are now
   device-side; BN expert backward and Adam use backend ops, while BN expert
   SGD remains host-side.
   Remaining across the phases:
   `cfg/LegoGears_centernet.cfg` end to end (the cfg lives on `Jev_laya`), then
   2B's `eml_convolutional`, 2C (OBB), 3A/3B (transformer + deformable families,
   shared LayerNorm/GEMM kernel set first) and 4.
3. **Phase 3A prerequisite (2) is complete**: LayerNorm fwd/bwd, the fused
   residual/bias epilogue, the row-softmax backward and the batched-GEMM wrapper
   (`linear_f32`) are all in, each with a host default plus GPU parity tests.
   That unblocks `vit`, `swin`/`transformer`, `tucker_attention`, `mambavision`,
   `kimi_kda`, `minimax_sparse_attention` and the LAYA encoder/head. Still to
   decide when writing those layers: whether the `flash_attention` /
   `vc_attention` fast paths (decision 3) get their own kernels or reuse the
   reference attention path.
4. Phase 2B `moe_convolutional` (weights order verified) has partial Oxide
   training support; BN backward and SGD/Adam updates use backend ops. Then
   port `eml_convolutional` (an activation, not a `LayerKind`; save order **re-read
   2026-10-05**: `input_layer` then `self_layer`, two full conv blocks).
5. Phase 3B deformable family — start with `dcnv4` (real C++ forward and
   kernels; weights order `weights.cpp:428`).
6. Phase 2C OBB, then Phase 4.

## Known evidence gaps (not user questions)

- `[dcnv3]`: forward/backward/update are stubs in **all five** branches that
  carry it (`deform`, `Jev_laya`, `fog`, `jepa`, `depth_3d`), and
  `docs/dcnv3_layer.md` says so. Only its `.weights` order can be ported
  against C++; the numerics would have to come from InternImage. Details and
  the evidence are in `tasks/rust/layers.md`.
- `[contrastive]` detection branch needs the yolo head's per-box class ids;
  `YoloHeadOutput` only carries the raw output tensor.
- DETR is still broken in C++, so Phase 4's `detr_decoder` also has no
  reference numerics.
- No C++ build of any owning branch is available here, so C++ dump parity is
  "NOT RUN" everywhere.

## Task plans not started

Two self-contained task plans in this directory have **no implementation**:

- `tasks/rust/onnx.md` (`darknet-rs export-onnx`): 0/23 task steps done. No
  `src/darknet-core/src/onnx/` module and no `export-onnx` CLI arm exist.
- `tasks/rust/gui_iced_plan.md` (iced DarkMark-compatible GUI): 0/84 task
  steps done. None of `darknet-gui-project` / `darknet-gui-engine` /
  `darknet-gui` exist, and there is no `detect-server` CLI arm.

Both are greenfield; the layer-port work above does not depend on them.
