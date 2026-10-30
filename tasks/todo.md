# Latin pretraining preparation

## ONNX detection and segmentation extensions

### Detection port: `raptor-fp4`

```text
Darknet raptor-fp4                   Lightnet
src-lib/yolo_layer.{cpp,cu}         src/lightnet-core/src/{model.rs,train.rs,runtime/}
src-lib/{modern_yolo*,region*,      src/lightnet-core/src/{detect.rs,layers/}
  centernet*,detector*}             src/lightnet-cli/src/main.rs
src-test/{test_yolo_layer_cuda,      src/lightnet-cuda/src/lib.rs
  test_modern_yolo_heads}           src/lightnet-cuda-oxide/src/{kernels/,lib.rs}
                                    src/lightnet-cuda-oxide/tests/
```

The model parser describes supported YOLO options; the device backend owns activation, assignment, loss, deltas and metrics; GPU runtime owns device tensor lifetime, loss scaling and backward seed flow; CLI selects the device path and reports metrics; existing CPU logic supplies a comparison while the C++ branch remains the parity reference. Unsupported reference options must be rejected explicitly, collisions must preserve assignment order, and FP32 head semantics must coexist with the user's FP8 backbone requirement.

Initial inventory: classic YOLO, Gaussian YOLO, OBB, YOLOX, PP-YOLOE, YOLO-NAS, CenterNet, Region and Detection have distinct dispatch and loss behavior. Current support table identifies truth-driven Region loss/decode and legacy Detection loss/decode as open; modern-head losses remain host-owned. LegoGears uses classic CIoU and two YOLO heads. The reference GPU tests exercise baseline, extra anchors, objectness smoothing, new coordinates, truth-threshold positives and collisions. This inventory is not yet a complete branch parity audit.

- [ ] Audit branch detection head types, decode, datasets, config options, training and tests against current Rust; record confirmed gaps and exact supported scope.
- [ ] Port classic YOLO GPU loss and supported-option checks, including objectness/anchor assignment, threshold positives, collisions, CIoU, per-image metrics and CUDA-resident output.
- [ ] Connect GPU loss tensors to backward without host download/re-upload; preserve loss scaling, gradient direction, ownership and fallback errors.
- [ ] Wire the device loss into detector training and validate unsupported flags before training.
- [ ] Add focused CPU/reference and SM 89 parity tests, then run an authorized short LegoGears smoke test and compare FP32 and FP8 when the precision policy exists.

Reference: `/home/kerrigan/src/DarkSuite/darknet`, branch `raptor-fp4` (read without switching the user's checkout).

User requirements: port all relevant detection/segmentation layers and workflows; SM 89 native training and inference use FP32 master weights and FP8 tensor-core convolution, with immediately preceding head convolutions FP32. FP8 may remain between eligible convolution, pool and route operations; other layers retain higher precision. Export FP32 and FP16 ONNX with NCHW/NHWC external layouts; offer raw or decoded YOLO outputs, leave NMS to the application, and default segmentation activation on with a raw-logit option. Acceptance datasets: `/home/kerrigan/src/DarkSuite/datasets/{LegoGears_v2,semantic_binary,semantic_multi}`. Start with short smoke tests; compare native FP8 against FP32 and ONNX outputs against native FP32.

```text
Darknet raptor-fp4                 Lightnet
src-lib/{yolo*,region*,            src/lightnet-core/src/{detect.rs,dense_loss.rs,
  centernet*,detector*}              layers/,runtime/}
src-seg/{segmentation_layer*,      src/lightnet-core/src/layers/segmentation/
  segmentation_data*,seg_gate*,   src/lightnet-core/src/runtime/gpu_dense_loss.rs
  segmentation_visualization*,    src/lightnet-cli/src/
  segmenter*}                     src/lightnet-cuda-oxide/src/{dense_loss_backend.rs,kernels/}
src-onnx/{darknet_onnx*,          src/lightnet-core/src/onnx/{export.rs,proto.rs}
  darknet_segmenter_onnx*}        tools/onnx_check.py
src-test/{test_segmentation*,     src/lightnet-cuda-oxide/tests/oxide_runtime_chain.rs
  test_seg_gate*,test_yolo*}      src/lightnet-core/src/runtime/tests/
```

Detection layers own head activation, assignment, losses, gradients and decoding; runtime and data loaders feed them, GPU backends implement device work, and CLI/evaluation consume results. Segmentation layers own semantic activation and loss/gating; data loading, augmentation, visualization and CLI depend on their mask conventions. ONNX export depends on network shapes, weights and head semantics and must preserve NCHW internal operators while exposing the selected layout. Tests need native and ONNX comparisons against the named C++ branch, including CPU and GPU training behavior. Inventory exact branch variants and unsupported combinations before assigning completion.

- [ ] Inventory all detection and segmentation layers, data, training, inference, GPU, and ONNX features in Darknet `raptor-fp4` against Rust; port missing behavior and test each path.
- [ ] Port `src-lib/yolo_layer.cu` native device training loss and its supported-option checks, then compare assignments, deltas, and metrics against C++ and Rust references.
- [ ] Port `src-seg` data/mask discovery, semantic segmenter training/inference and visualization, with binary and multiclass fixtures.
- [ ] Port `[softmax] spatial=1` forward and backward on CPU and Oxide, and handle legacy `[logistic]`/spatial `[softmax]` segmentation heads in ONNX. The supplied `semantic_binary` and `semantic_multi` configurations use these heads.
- [ ] Preserve and apply `[softmax]` temperature from `raptor-fp4` on CPU, GPU, and ONNX; test non-default temperatures and single-channel spatial softmax.
- [ ] Export semantic segmentation activation and verify against the native runtime.
- [ ] Add decoded detection outputs and verify against native detection, preserving raw-head mode.
- [ ] Complete FP16/NCHW/NHWC ONNX Runtime and native numerical comparisons for all three acceptance models; test activation and raw-logit variants.
- [ ] Implement per-convolution FP8 selection for forward and backward without mutable global precision state. Select FP32 for the convolution immediately preceding each detection/segmentation head; preserve FP32 master weights and GPU updates.
- [ ] Implement and validate the eligible FP8 pool/route storage and conversion boundaries, keeping all other operations in higher precision.
- [ ] Validate supported combinations with ONNX Runtime and record limitations and results.

Current slice: semantic and legacy segmentation ONNX heads, activation selection, NHWC boundary transposes and FP16 export are implemented. Thirteen exporter tests, the CLI flag test and the pixel-mask fixture passed; CPU spatial-softmax forward/adjoint and one Oxide spatial-backward kernel test passed. Real binary segmentation FP16 NHWC ONNX executed with finite output; on a seeded synthetic input its difference from FP32 ONNX was max 0.0024834, mean 0.0001714. Native CPU comparison was interrupted because debug inference exceeded a short smoke test. Detection decoding, COCO data, CLI integration, FP8 policy, ONNX-versus-native parity and the branch inventory remain open.

Real multiclass segmentation FP16 NHWC ONNX also ran on the CPU ONNX Runtime: finite `[1,224,224,4]` output, maximum class-sum error 0.000402. This does not establish parity against the native runtime.

The updated Oxide spatial-softmax kernel test passed on SM 89 for multiclass forward, single-channel forward, and backward. FP16 native spatial softmax and configurable temperature are still open.

## Contents

- [Design](#design)
- [Plan](#plan)
- [Review](#review)
- [Layer host and device implementation](#layer-host-and-device-implementation)
- [Resident Muon and dense AMP training](#resident-muon-and-dense-amp-training)
- [Resident forward cache ownership](#resident-forward-cache-ownership)

## Design

Use the downloaded pinned snapshots under `datasets/latin/raw`. Extract UTF-8 NFC surface text, retain macrons and u/v/i/j, remove TEI notes/apparatus/translations and annotation-only enclitic copies. Prefer canonical Perseus over annotated mirrors by CTS work ID. Include additional lascivaroma works only with an explicit source licence; quarantine unresolved sources. Keep all Seneca in validation, all Tacitus in test, remaining authors together in deterministic roughly 90/5/5 allocation. LLCT is a separate diagnostic, never training. Audit original and folded five-word shingles against held-out text; quarantine training documents with substantial overlap. Produce seven-field document TSVs, original/extracted hashes, source revisions, exclusions, bounded train-only tokenizer sample, 8192 BPE artifact and per-split native shards. No model training.

Relevant components: `tools/prepare_latin.py` handles source extraction and audit, `tools/test_prepare_latin.py` verifies boundary cases, and `src/lightnet-examples/examples/llm_prepare_corpus.rs` uses the existing tokenizer/data API behind the optional `llm` feature. Raw data depends on `sources.tsv`; Rust output depends on the prepared document TSVs. Main constraints are work-level leakage, mixed original-source licences, bounded memory and compatibility with existing loaders. User scope correction: prepare code only; no build or corpus execution yet. See `tools/latin_pretraining.md` for the output layout and future commands.

## Plan

- [x] Implement TEI/annotated XML/CoNLL-U extraction and split rules; five focused extraction fixtures passed.
- [x] Implement work/text deduplication, training-versus-heldout shingle audit, provenance and exclusions.
- [x] Implement a bounded, deterministic training-only tokenizer sample.
- [x] Implement BPE fitting and native token shards using the existing Rust APIs.
- [x] Gate the preparation example and optional LM dependencies behind `lightnet-examples/llm`.
- [x] Add manifest/loader/round-trip validation code and document usage and current evidence.
- [ ] Deferred by user: compile, execute the real preparation pipeline, and validate resulting tokenizer/shards.

## Review

Source-only delivery per user correction. The real extraction attempt was stopped during shingle auditing. No BPE fitting or model pretraining ran. Rust example/feature wiring remains uncompiled; five Python fixtures passed earlier. `git diff --check` passed after the feature changes. Unrelated concurrent GPU/ViT edits were left untouched. No commit/tag: the required Rust validation is deferred by the user.

## Layer host and device implementation

Priority update: full backward, native GPU updates, FP16 and every-layer
completion come first. CPU detection-oracle work is deferred. Current concrete
sequence: [device_completion_plan.md](rust/device_completion_plan.md).

User scope: implement the host/device contract in `tasks/rust/AGENTS.md`, source only. No Cargo checks, builds, tests, execution or training. User subsequently authorized local WIP commits for source-complete features without Cargo validation; no release tags. Existing CPU references and pending device edits are retained. The existing Latin preparation task above is independent.

### Project tree

```text
src/
  lightnet-core/src/
    layers/
      CONTEXT.md
      mambavision/{mod.rs,cpu.rs,gpu.rs}
      kimi_kda/{mod.rs,cpu.rs,gpu.rs}
      depth_anything/{mod.rs,cpu.rs}
      dpt_head/{mod.rs,cpu.rs}
      recursive_block/{mod.rs,cpu.rs}
    runtime/
      gpu_types.rs
      gpu_state.rs
      gpu_forward.rs
      gpu_forward_compact.rs
      gpu_backward.rs
      gpu_backward_support.rs
  lightnet-cuda/src/
    lib.rs
    mambavision.rs
  lightnet-cuda-oxide/src/
    lib.rs
    mambavision.rs
    kernels/{mod.rs,mambavision.rs}
tasks/
  todo.md
  lessons.md
  rust/{AGENTS.md,layer-support.md,gpu_layers_plan.md}
```

### Components and constraints

- Layer CPU code owns config geometry, serialized parameter order and exact forward/backward semantics. Existing CPU optimizer arms consume the same parameter gradients; no compiled-C++ parity is implied.
- Layer GPU glue composes backend operations, stores device training caches, and routes mathematical gradients to runtime accumulation. Forward/backward selection must use the same support gate.
- Runtime types/state/dispatch own cache lifetime, layer selection and gradient fan-out. Every constructor must initialize new caches; unsupported variants retain explicit documented fallback.
- Backend trait defaults supply host primitive references; Oxide must override every newly used primitive to avoid hidden host downloads.
- Oxide wrappers validate dimensions and buffer lengths; kernels implement depthwise SiLU, selective scan and sparse attention with exact piecewise derivatives. Mamba's selective projection deliberately consumes the CPU BDT storage as flat rows.
- Remaining layers follow the same contract, retaining CPU gaps and pending f16/optimizer residency requirements. Verification is deferred by explicit user instruction.

### Implementation sequence

- [x] Inspect live instructions, lessons, dirty files and existing device primitives.
- [x] Write MambaVision native forward/training cache/backward and all Oxide overrides, preserving scan and MSA derivative guards (uncompiled).
- [x] Add source parity fixtures for MambaVision layout, causal convolution, MSA and parameter gradients without running them.
- [x] Write KimiKDA native scan/router/runtime, CPU residual-adjoint correction and unrun fixtures.
- [x] Write DPT native four-tap forward/backward, optimizer dispatch and unrun fixtures.
- [x] Write frozen DepthAnything native forward/input adjoint, tap clamp and unrun fixtures.
- [x] Write compact f32 inference dispatch for ViT/Swin/Tucker/Mamba/Kimi/depth/DPT/WMHF/Clifford and unrun parity fixtures.
- [x] Write bounded static RecursiveBlock native f32 forward/backward/compact/optimizer integration and unrun fixtures.
- [x] Write native Mamba SGD/Adam resident parameters/gradients/moments, synchronization/checkpoint lifecycle and scalar finiteness reduction (uncompiled; fixtures unrun).
- [ ] Continue RecursiveBlock controller/lattice/BN/other body variants and remaining layers according to the completion queue.
- [ ] Continue remaining host/device gaps and native updates/residency without claiming full completion.
- [ ] Review source diffs and document exact implemented scope and unvalidated gaps.
- [ ] Deferred: compile and run host/device parity gates; no layer may be certified complete before those pass.

### Kimi residency and half-storage continuation

Relevant tree: `layers/{kimi_kda,mambavision}/gpu.rs`,
`runtime/{gpu_resident_params,gpu_f16,mod}.rs`, and Oxide
`tests/{oxide_resident_params,oxide_kimi_kda}.rs`. Layer glue owns exact
serialization and device adjoints; resident banks own decay/checkpoint mapping
and synchronization; half inference owns device-only casts and cache isolation.
The detailed project tree and component contracts are in
`rust/device_completion_plan.md`.

- [x] Write unrun Kimi fixtures for deferred SGD/Adam/AdamW, diagnostics and resume.
- [x] Extend resident layout with separate coupled decay and moment slots.
- [x] Wire Kimi device slices, packed gradients and host synchronization.
- [x] Extend Tucker residency with device factor packing/unpacking and unrun update fixtures.
- [x] Add Kimi/Mamba/Tucker half-storage inference with explicit device f32 islands.
- [x] Inspect source diffs and record unvalidated local WIP checkpoint `eaeb292` (staged `git diff --check` passed; no compilation or tests).
- [ ] Deferred: compile and run all relevant fixtures; execution remains prohibited.

Source review: checked bias-first layouts, exact gradient-range coverage,
source decay rules, root versus child DNOPT moments, device cast overrides and
training-cache isolation. Host staleness is marked immediately per resident
layer publication, including partial-update error exits. No compilation,
numeric execution or every-layer completion claim. Muon remains hybrid; native
half arithmetic/training and other families remain open.

## Resident Muon and dense AMP training

Source-only continuation requested explicitly. Work directly, retain unrelated
files and use local WIP checkpoints without release tags.

Verification authorization update: user now permits Cargo checks/tests, with
at most two threads and only one check/test invocation running at a time.
Retain the 3G soft RAM cap. Start with cargo xtask check, then focused host
tests; GPU validation uses the Oxide wrapper and remains separately reported.

- [x] Define serialized Muon matrix ranges matching the CPU visitor order; host order/value fixture passed.
- [x] Write resident matrix momentum, native Muon/AdamW updates, gates and deferred synchronization (device validation pending).
- [x] Wire resident Muon DNOPT6 save/load and host materialization (device validation pending).
- [x] Write native dense AMP forward/dgrad/wgrad with f32 parameter gradients/master state (device validation pending).
- [x] Write source fixtures for repeated steps, resume, gates, loss scaling and half overflow.
- [x] Review the remaining layer queue, implement non-BN GRU and document limitations.
- [x] Write exact non-BN GRU host/native BPTT, resident SGD/Adam/Muon and half-storage inference; retain explicit BN rejection.
- [ ] Inspect source/diffs, update ledgers and create local WIP checkpoints.
- [x] Host cargo xtask check passed; focused GRU 4/4 and Muon 26/26 passed with two threads.
- [x] Resolve broad runtime failures and rerun its full filter: 324/324 passed after initialization, routing and Swin cache fixes.
- [x] GPU-enabled cargo xtask gpu-check passed; new multi-head Swin finite-difference fixture passed.
- [ ] Run relevant Oxide suites one at a time, at most two jobs (dense AMP first).
- [x] Oxide dense AMP suite: 3/3 passed on GPU; f16/bf16 operands and f32 parameter/master state verified.
- [x] Oxide GRU training suite: 1/1 passed on GPU across SGD/Adam/Muon, AMP off/on, repeated BPTT and resume.
- [x] Diagnose resident suite (12 passed, 3 failed): fix CPU Mamba scan C-coefficient lookup; new finite-difference test failed before fix, 19 Mamba host tests passed after fix.
- [x] Rerun the three failed resident optimizer tests after the CPU reference fix: Muon filter 3/3 and SGD/Adam gate 1/1 passed.
- [x] Full core unit suite: 1,016 passed; cfg-corpus integration interrupted by user stop, CLI full suite pending.
- [ ] Remaining all-layer/variant completion and full native-half arithmetic coverage.

Verified runtime root-cause fixes: explicit device-training-buffer allocation
state with invalidation on weight reload, fake routing forward/backward dimensions
derived from metadata rather than nominal token shapes, explicit unsupported
activations in host-fallback fixtures, and compact Swin head-cache strides matching
the adjoint. A two-/three-head finite-difference fixture was added. Training is
still outside scope.

## Resident forward cache ownership

Stopped by user request, 2026-10-26. Source checkpoint `0e93bcb` is
uncompiled/unrun; the prior corpus/Cargo scope was stopped. No task-owned test
process remains. Resume only on an explicit request; see HANDOFF.md.

```text
src/lightnet-core/src/
  layers/{mambavision,kimi_kda,tucker_attention}/gpu.rs
  layers/{deform_conv,vit,swin,wmhf,clifford,dpt_head,depth_anything,recursive_block}/gpu.rs
  runtime/
    gpu_types.rs
    gpu_state.rs
    gpu_forward.rs
    gpu_backward.rs
    gpu_resident_params.rs
    tests/yolo_3d_runtime_tests.rs
src/lightnet-cuda-oxide/tests/{oxide_resident_params.rs,oxide_forward_tapes.rs}
tasks/rust/{source-wip.md,device_completion_plan.md}
```

Layer device caches must belong to the returned forward state, as recurrent
GRU tapes already do. Forward dispatch produces them; backward consumes the
matching state's tape. The network retains only reusable parameters/optimizer
state. A monotonic network epoch identifies each parameter-bank allocation,
edit and update, including after sidecar reload; bank retirement must not reset
that epoch or allow an old forward to match a newly allocated bank.
The same ownership rule applies to all remaining native layer tapes. WMHF
and Clifford rolling-stat accumulation must consume the same forward-owned
BN statistics as their backward call, including overlapping forwards.

- [x] Add distinct-input overlapping-forward and stale-reload regressions for all three resident mixers (unrun).
- [ ] On explicit resume, run new lifecycle regressions; the earlier corpus command was interrupted and is no longer active.
- [x] Move the three mixer caches to GpuForwardState and wire exact backward ownership (uncompiled).
- [x] Allocate non-repeating resident generations across bank retirement/reload and updates (uncompiled).
- [x] Move the remaining eight native cache families and their BN-stat consumers into the forward state (uncompiled).
- [x] Add overlapping-forward/inference input-adjoint and BN rolling-stat regressions for those native families (unrun).
- [ ] Run focused host/device lifecycle checks serially, capped at two jobs and MemoryHigh=3G.
- [ ] Record exact evidence and make a local WIP checkpoint; leave all-layer completion open.

## Explicit resume verification, 2026-10-26

Plan: execute HANDOFF.md's queue serially on main at 0e93bcb. Preserve the
independent Latin/example/root-lockfile work. No subagents, Python edits,
training, benchmarks or pushing. All build/check/test commands use the user
systemd scope with MemoryHigh=3G, JOBS=2, CARGO_BUILD_JOBS=2,
RAYON_NUM_THREADS=2 and host/GPU RUST_TEST_THREADS=2/1.

Relevant tree and component contracts: see the Resident forward cache ownership
section above and tasks/rust/03_plans/gpu/device_completion_plan.md. xtask/src/main.rs and
 tools/{test_host.sh,test_oxide.sh} own the serial verification launchers;
lightnet-cli checks downstream runtime integration. Layer code produces and
consumes tapes; runtime owns generations/resident state; Oxide tests execute
the lifecycle contract on device. Preserve inactive DPT slots and per-child
optimizer checkpoint semantics when its implementation slice is reached.

- [x] Read handoff, Rust instructions/source plan/layer context/CLAUDE and lessons; inspect main, HEAD and dirty state.
- [x] Compile 0e93bcb with ln_host check: core/CLI passed; no compile fixes needed.
- [x] Run core runtime:: host tests (324 passed, 0 failed), then gpu-check (passed).
- [x] Run resident_overlapping_forwards (1/1, 84.86s) and resident_reload_cannot (1/1, 72.91s) serially; both passed across Mamba/Kimi/Tucker.
- [x] Run oxide_forward_tapes: 1/1 passed (333.34s), eight additional native families; lifecycle equivalence only.
- [ ] Run full CLI tests, then relevant full layer/resident suites serially.
  - Full CLI tests passed: 36/36 (0.44s).
  - Full resident suite passed: 17/17 (1278.44s) at 0e93bcb; layer numeric suites begin with oxide_mambavision.
  - oxide_mambavision passed: 2/2 (98.50s); oxide_kimi_kda now running.
  - oxide_kimi_kda passed: 3/3 (100.85s); oxide_tucker_attention now running.
  - oxide_tucker_attention passed: 3/3 (175.91s); oxide_vit now running.
  - oxide_vit passed: 5/5 (258.35s); oxide_swin now running.
  - oxide_swin passed: 5/5 (224.26s); oxide_wmhf now running.
  - oxide_wmhf passed: 8/8 (247.76s); oxide_clifford now running.
  - Initial oxide_clifford failed its Adam update comparison (1.1428108 != 1.1332897 at parameter 0); CPU fixture used SGD. Corrected CPU optimizer selection, rerun pending old binary exit; tolerances unchanged.
  - Initial Clifford aggregate 4 passed/1 failed (220.09s); corrected full rerun 5/5 passed (216.17s).
  - Validated fixture checkpoint blocked: named-path git add failed because .git/index.lock is on a read-only filesystem. No commit/tag created. Continue checks without bypassing filesystem permissions.
  - oxide_dpt_head passed: 1/1 (59.33s), original host-staged path; oxide_depth_anything passed 2/2 (63.13s).
  - oxide_depth_anything passed: 2/2 (63.13s); oxide_recursive_block now running.
  - Initial oxide_recursive_block failed 0/1 (22.47s): native body accumulators negated CPU convolution/connected signs. Corrected two loops to +=. Host check (4.44s), runtime:: (324/324), gpu-check (4.41s), and full recursive suite 1/1 (59.48s) passed.
- [x] Add partial DPT resident SGD/Adam/Muon parameters, gradients and frozen-slot masks; check, runtime:: (325/325), gpu-check, oxide_dpt_head (1/1, 65.57s), and oxide_resident_params (17/17, 1117.58s) passed.
- [x] Add DPT inactive-accumulator and optimizer-specific moment regression (1/1); verify Muon skips matrix Adam moments.
- [x] Round-trip resident DPT Adam/Muon child moments through DNOPT; verify inactive and Muon matrix moments stay absent.
- [ ] Test switching a saved DPT Adam checkpoint into Muon and decide/verify its intended validation behavior.
- [ ] Continue other-family residency, FP16 coverage and remaining backward variants with verification after each slice.
  - [x] ViT supported native SGD/Adam/AdamW residency: preserve 16-array order, global clipping, frozen positions and reduced/clamped mHC updates; `oxide_vit` 7/7.
  - [x] ViT compact f16 activation-storage dispatch around f32 arithmetic; tested before/after deferred resident updates in `oxide_vit` 7/7.
  - [ ] ViT MSA/VC native backward/residency and ViT Muon remain open; current native support gate excludes MSA/VC.
  - [ ] Next transformer-family residency slice: Swin; preserve optional residual projection and variable serialized block order.
  - [ ] Full CLI/layer reruns after current continuation edits.
- [ ] Run cfg_corpus separately and last.
- [x] Refresh HANDOFF.md/source-wip.md/device_completion_plan.md with exact DPT and RecursiveBlock evidence.
- [ ] Commit named green slices if the filesystem permits; named-path DPT retry failed again because `.git/index.lock` creation reports read-only filesystem.

### Resume review

Host check, runtime:: (324/324) and gpu-check passed at 0e93bcb on explicit
resume, then passed again after DPT residency edits. Full CLI tests passed
36/36 earlier in the resume. oxide_recursive_block passed 1/1 after its sign
fix; oxide_dpt_head passed 1/1 after the partial resident SGD/Adam addition.
The first DPT runtime rerun found a tensor-shape mismatch and passed after the
flat-packing fix. cfg_corpus remains separately queued last and is not covered
by the runtime filter.
Stop if a covered failure/cannot-run condition remains unresolved
after one root-cause pass. Do not infer full-layer completion from a subset.

### Swin resident Muon slice

- [x] Add a device regression comparing CPU and resident Muon for Swin with
  and without residual projection, including deferred updates and resume.
- [x] Map serialized matrices in CPU visitor order and retain the resident
  bank through Muon forward, backward, update and optimizer switching.
- [x] Preserve Swin-wide gradient sanitization/clipping, AdamW fallback and
  mHC clamping; validate current-tree host/runtime/Swin/Oxide gates serially.
- [x] Record the exact evidence and remaining gaps without claiming full Swin
  completion or compiled-C++ parity.

### ViT native MSA slice

- [ ] Add an end-to-end native MSA regression with actual sparse selection,
  multiple heads, a partial final block, backward and optimizer update.
- [ ] Implement device-resident selection and exact ViT MSA forward/backward,
  preserving Q/K and score clamps, deterministic ties and forward-owned tape.
- [ ] Validate host and GPU checks, ViT/runtime filters and the full ViT Oxide
  suite serially; retain VC attention and other uncovered variants as open.
- [ ] Record evidence and limits without claiming complete ViT or C++ parity.
