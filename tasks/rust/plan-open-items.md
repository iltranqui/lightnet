# Rust Port — Open-Items Execution Plan

> **Current operator instructions:** implement directly without subagents and focus on GPU layers now. This overrides historical delegation/CPU-first sequencing below, without completing or dropping CPU requirements. Resume native GPU work from `gpu_layers_plan.md`; implementation evidence is in `implemented.md` §§41–43. Native act-head features are only a primitive, not runtime integration; the source-training mismatch and unanswered behavior question are recorded in §43. Historical role assignments are not authorization to delegate.

**Goal:** Finish only the remaining **layer parity and axis/shape alignment** work in `tasks/rust/` (branch `rust`): every unported layer follows the Layer Port Contract, and every ported layer has an explicit axis/layout check against the C++ source. Waves 0–2 are done; all other open items are out of scope (see [Scope](#scope)).

**Architecture:** Strictly sequential waves. Wave 0 establishes the baseline (not green; see Progress). Wave 1 closes OBB CPU-training gaps (all in `obb_training.rs` / `train.rs` / `detect.rs` / CLI). Wave 2 executes the two already-written step plans. Everything else is a backlog stub that must be expanded into its own plan before dispatch (`layers.md` rule).

**Tech Stack:** Rust workspace (`darknet-core`, `darknet-cli`, `darknet-cuda`), `src/darknet-cuda-oxide` (not a workspace member, built with `cargo oxide`), nightly-2026-08-28.

**Spec:** `tasks/rust/STATUS.md` (resume point), `tasks/rust/todo.md` (open items), `tasks/rust/layers.md` (Layer Port Contract, phase tables), `tasks/rust/implemented.md` (done ledger).

## Table of Contents

- [Progress (as of 2026-10-16)](#progress-as-of-2026-10-16)
- [Context](#context)
- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Project Tree](#project-tree)
- [Component Breakdown](#component-breakdown)
- [Execution Model](#execution-model)
- [Wave 0 — Baseline establishment](#wave-0--baseline-establishment)
  - [Task 0.1: Stop the oxide CLI build break and make it unrepeatable](#task-01-stop-the-oxide-cli-build-break-and-make-it-unrepeatable)
  - [Task 0.2: Host-suite baseline and WIP triage report](#task-02-host-suite-baseline-and-wip-triage-report)
- [Wave 1 — OBB CPU training gaps](#wave-1--obb-cpu-training-gaps)
  - [Task 1.1: Adam for CPU OBB training](#task-11-adam-for-cpu-obb-training)
  - [Task 1.2: Photometric augmentation for OBB training](#task-12-photometric-augmentation-for-obb-training)
  - [Task 1.3: Weighted OBB NMS merge](#task-13-weighted-obb-nms-merge)
  - [Task 1.4: `ciou_obb_mul` training path](#task-14-ciou_obb_mul-training-path)
- [Wave 2 — Execute existing plans](#wave-2--execute-existing-plans)
  - [Task W2.1: Audit `onnx.md` and `gui_plan.md` for dispatchability](#task-w21-audit-onnxmd-and-gui_planmd-for-dispatchability)
  - [Task W2.2: Execute `onnx.md` (21/21 steps)](#task-w22-execute-onnxmd-211-steps)
- [Scope](#scope)
- [Axis alignment checklist](#axis-alignment-checklist)
- [Layer parity backlog (expand before dispatch)](#layer-parity-backlog-expand-before-dispatch)
- [Out of scope / blocked](#out-of-scope--blocked)
- [Verification](#verification)
- [Self-review](#self-review)
- [Review follow-up (2026-10-17)](#review-follow-up-2026-10-17)

## Progress (as of 2026-10-16)

- **Task 0.1 complete:** `ForwardResult::from_heads` and the CLI f16 caller were committed as `98003667` (`v0.0.247`); focused test and oxide CLI check passed.
- **Task 0.2 baseline not green; disposition recorded:** direct uncapped `cargo test -p darknet-core -p darknet-cli -j 2` passed 562/562 core unit tests and 23/23 CLI tests, and other integration targets passed, but `cfg_corpus` failed after 3943.36 s on `cfg/cspx-p7-mish.cfg` and `cfg/cspx-p7-mish_hp.cfg` (child CPU inference failure with empty error) and timed out on `cfg/lstm.train.cfg` and `cfg/rnn.train.cfg` (300 s each). Full evidence is in `STATUS.md`. At the time of this baseline run, the `MEM=3G` wrapper could not reach the WSL user systemd bus, so the command was uncapped. A later user-provided `./tools/test_oxide.sh` log showed `Running scope as unit: ...scope`. Then `MEM=3G ./tools/test_host.sh obb_` ran in a systemd scope and passed 135 core + 4 CLI tests. The full host baseline and `cfg_corpus` have not been rerun under the cap, so this focused pass does not resolve the four corpus cases; they remain non-green and unresolved as behavior. Disposition: accept them as known resource/runtime residuals for scoped follow-on work only; they are not passes or fixed. Do not claim a green full-host baseline.
- **Wave 1 tasks 1.1–1.4 complete:** OBB CPU Adam (`f9664e63`), photometric augmentation (`5b98fe1c`), weighted OBB NMS (`968c1beb`), OBB default augmentation settings (`fae49267`), and `ciou_obb_mul` (`31219aba`, `v0.0.253`). The previous `jitter` failure is resolved by the zero default for omitted OBB jitter. Current `MEM=3G ./tools/test_host.sh obb_` filter: 135 core + 4 CLI tests pass. Multiplicative source formula and numerical-gradient details are recorded in `STATUS.md`; no compiled-C++ parity is claimed.
- **WIP snapshot audit:** `git show --stat 8fb5bb1b` lists 35 files. The `obb_` filter exercises OBB model/training/detection/evaluation/augmentation plus CLI OBB paths, not the snapshot's unrelated render scalar, CenterNet, chart, contrastive, deconvolution, dense-loss, LAYA, MoE, Modern YOLO, segmentation, generic weights/runtime, and CUDA areas.
- **Baseline triage/disposition:** the two CSPX child failures are SIGKILL/global-OOM events (~7 GiB RSS confirmed by kernel logs); `lstm.train.cfg` and `rnn.train.cfg` each request batch 128 × 576 time steps × three 1024-unit recurrent layers and hit the 300 s CPU-inference timeout. A targeted child run of `cspx-p7-mish.cfg` reproduced SIGKILL. Full baseline remains non-green; no parser/model error was reproduced. The controller accepts these four known residuals for scoped follow-on tasks only; no full-host green claim is allowed. The standalone corpus command is not rerun because the full run took >1 hour.

## Context

`tasks/rust/` holds ~12.9k lines of plans and ledgers. The open items span trivially small (a 3-field struct literal), medium (OBB Adam), and XL (OBB GPU training, DETR, clifford, GUI). Dispatching "the open items" as one job would hand subagents ambiguous, huge tasks and would break the repo's one-build-at-a-time rule. This plan fixes ordering, interfaces, and test gates so each dispatch is small and checkable.

Verified facts that shaped it (status as of 2026-10-17; items marked FIXED were the original blockers):
- FIXED (`98003667`, v0.0.247): `cargo check -p darknet-cli --features oxide` was broken by `ForwardResult` literal at `main.rs:310` missing `moe_caches`, `eml_caches`, `vit_caches`.
- The workspace itself did not load at HEAD (dangling `darknet-gui*` members); `Cargo.toml` now has them commented out (uncommitted).
- FIXED (`f9664e63`, `5b98fe1c`, `fae49267`): OBB CPU trainer rejected Adam and photometric augmentation; geometric augmentation (`jitter, resize, letter_box, mixup, mosaic, mosaic_bound, random`) is still rejected by design.
- FIXED (`31219aba`, `968c1beb`): `ciou_obb_mul` was rejected for training; weighted OBB NMS merge was unsupported.
- Sample-level API dispatch is complete in `v0.0.258` (`evaluate.rs`, `carpn_map_plan.md`): `obb_detector_map` routes homogeneous CARPN sample sets to `carpn_detector_map` and rejects mixed geometry. The source audit and bounded CPU network inference / `detect-image` / direct `detector map` plan are in `carpn_network_map_plan.md`; implementation is not started. CARPN training and scheduled `detector train -map` are deferred until Stage-2 training/backward/update support is separately planned.
- No compiled C++ exists in the repo, so every "parity" claim is hand-derived or finite-difference only.
- Last version commit before Wave 2 was `v0.0.253` (`31219aba`); `v0.0.248` (`2941a2ee`, CUDA toolkit path) remains the intentional out-of-plan gap. ONNX implementation is recorded in `v0.0.254`–`v0.0.257`; documentation is committed as `2b998051`. Next version is `v0.0.259`. HEAD at plan time `8fb5bb1b` is a WIP snapshot ("before switching to raptor-dev"); do not rewrite it.

## Global Constraints

- Build/test: max 4 threads; repo scripts use `-j 2`. Cap with `systemd-run --user --scope -p MemoryHigh=3G` (scripts take `MEM=3G`) when a user systemd session exists; otherwise run uncapped and say so in the report. ONE build/test command at a time, never two in parallel.
- Host tests: `MEM=3G ./tools/test_host.sh [filter]`. Oxide tests: `MEM=3G ./tools/test_oxide.sh --test <name>` (~15 min for all; run one suite). `cfg_corpus`: run alone with `-j 2`; a timeout is not a pass and not a failure.
- Backward is the exact derivative of the Rust forward; C++ deviations only under `GradMode::Cpp`. Unknown config = error, never silent identity.
- Commit only files a task names. Message: `v0.0.NNN: <what>` (next is `v0.0.259`), ending with the attribution line the harness gives. Never `git add -A`. Leave the user's uncommitted `.gitignore` edit alone.
- Never claim C++ parity; write "hand-derived from source, no compiled-C++ golden".
- Every `.md` file edited/created gets a TOC (user CLAUDE.md). Keep `tasks/rust/lessons.md` updated after any correction.

## Review Focus

Likely-to-bite inputs the spec does not obviously test; each is pinned by a test in the owning task.

1. OBB train with `adam=1` and `subdivisions>1`: Adam step counter must advance once per update, not per subdivision; weights unchanged when `learning_rate=0` (T1.1).
2. Resuming an OBB Adam run: moments must round-trip through the optimizer sidecar (T1.1).
3. Photometric augmentation must leave labels byte-identical and be seed-deterministic (T1.2).
4. OBB NMS merge with one box, zero overlap, tied/NaN scores must neither panic nor drop all boxes (T1.3).
5. The f16 inference path must still return heads with empty caches after `ForwardResult` gains fields (T0.1).

## Project Tree

Only components touched by Waves 0–2.

```
darknet/
├── Cargo.toml                    (gui members removed, uncommitted)
├── tasks/rust/
│   ├── STATUS.md · todo.md · implemented.md · layers.md · lessons.md
│   ├── plan-open-items.md        (this plan)
│   ├── onnx.md                   (21 executable checklist steps, Wave 2)
│   └── gui_plan.md          (egui plan M0–M6, Wave 2)
├── tools/test_host.sh · test_oxide.sh
└── src/
    ├── darknet-core/src/
    │   ├── runtime.rs            ForwardResult (498), ExecutableNetwork::update_weights (745), Adam (5396, 5637, 13236)
    │   ├── train.rs              train_obb_yolo_batch_step (551), train_obb_yolo_subdivisions (728), load_obb_training_batch* (1778-1872)
    │   ├── obb_training.rs       delta writeback passes, loss/metrics
    │   ├── detect.rs             decode_obb_yolo_detections (133), NMS kinds
    │   ├── model.rs              ObbNmsKind (881), iou_loss parse (3726)
    │   ├── evaluate.rs           obb_detector_map (419), carpn_detector_map (342)
    │   └── augmentation.rs       photometric + geometric transforms
    ├── darknet-cli/src/main.rs   f16 ForwardResult (310), run_detector_train_cpu_obb (2392)
    ├── darknet-cuda/src/host_ops.rs   AdamParams (807), adam_update
    └── darknet-cuda-oxide/       (Wave 0 only: compile check)
```

## Component Breakdown

| Component | Role | Responsibilities now needed | Depends on / used by | Constraint / open question |
|---|---|---|---|---|
| `ForwardResult` (`runtime.rs`) | Per-forward cache bag | Add `from_heads` constructor so callers stop listing every cache | CLI f16 path, runtime tests | Fixing the literal alone re-breaks on the next field |
| `train_obb_yolo_*` (`train.rs`) | CPU OBB SGD loop | Accept Adam via existing host Adam machinery | `ExecutableNetwork`, `AdamParams`, optimizer sidecar | Host Adam is entangled with GPU path (`update_cpu_fallback_adam`); spike first |
| OBB batch loader (`train.rs`) | Image+label batches | Photometric hue/sat/exposure only | `augmentation.rs` | Geometric aug needs OBB label transforms — out of scope |
| OBB NMS (`detect.rs`, `model.rs`) | Suppression | Weighted-merge kind | `ObbNmsKind`, decoder | Source only on `raptor*` branches; read via `git show` |
| `ciou_obb_mul` (`obb_training.rs`) | Loss variant | Distinct multiplicative CIoU-OBB gradient | `model.rs` parse | Derive from raptor source; stay rejected if underivable |
| `onnx.md`, `gui_plan.md` | Pre-written step plans | Execute as-is | independent crates | Audit task format first (Task W2.1) |

## Execution Model

Per user CLAUDE.md: **Sonnet** implementers (one per task), **Haiku** for read-only exploration, fresh reviewer per task, **Opus** only for plan expansion of backlog stubs. Tasks are sequential because they share `train.rs`/`main.rs` and because only one build may run at a time. After each task: reviewer runs the task's verify command, then the controller ticks the checkbox here and in `todo.md`.

## Wave 0 — Baseline establishment

### Task 0.1: Stop the oxide CLI build break and make it unrepeatable

**Files:**
- Modify: `src/darknet-core/src/runtime.rs` (add constructor next to `impl ForwardResult` at ~518; test module already added after it)
- Modify: `src/darknet-cli/src/main.rs:310`

**Interfaces:**
- Produces: `ForwardResult::from_heads(yolo_heads: Vec<YoloHeadOutput>, modern_heads: Vec<ModernHeadOutput>) -> ForwardResult` — all per-layer cache vectors empty.

- [x] **Step 1:** Write test `forward_result_from_heads_has_empty_caches` (done; asserts all cache vectors empty and `final_output().is_none()`).
- [x] **Step 2:** Run the filter — compile FAIL (E0599, `from_heads` missing). Done.
- [x] **Step 3:** Implemented `from_heads`; replaced the literal at `main.rs:310` with `darknet_core::ForwardResult::from_heads(summary.yolo_heads, summary.modern_heads)`.
- [x] **Step 4:** `cargo test -p darknet-core -j 2 -- forward_result_from_heads` — passed.
- [x] **Step 5:** `cargo check -p darknet-cli --features oxide -j 4` — passed.
- [x] **Step 6:** Committed as `98003667` (`v0.0.247: add ForwardResult::from_heads`).

### Task 0.2: Host-suite baseline and WIP triage report

**Files:** Modify `tasks/rust/STATUS.md` (new "Baseline 2026-10-16" section with TOC entry if headings change).

- [x] **Step 1 (attempted):** at the time, the `MEM=3G` wrapper could not reach `systemd-run --user`; direct uncapped `cargo test -p darknet-core -p darknet-cli -j 2` passed core unit 562/562 and CLI 23/23, then failed the corpus target on two child CPU-inference failures and two 300 s timeouts. Later, the user Oxide log confirmed systemd scopes start, and `MEM=3G ./tools/test_host.sh obb_` passed 135 core + 4 CLI. The full host baseline/cfg corpus has not been rerun under the cap. Exact original outcomes are in `STATUS.md`.
- [x] **Step 2 (observed in the full run, not rerun standalone):** `cfg_corpus` ran for 3943.36 s and reported the same four cfg failures; standalone rerun omitted to avoid repeating the >1-hour run.
- [x] **Step 3:** `git show --stat 8fb5bb1b` audited (35 files); filtered OBB coverage gaps are listed in Progress above.
- [x] **Step 4:** Recorded the baseline result and explicit disposition in `STATUS.md`: the four failures are accepted as known resource/runtime residuals for scoped follow-on work only. Task 0.2 remains non-green and is not a passing baseline; do not claim full-host green.

Gate: Wave 1 starts only if Step 1 has zero failures that were absent at `d1972af5` (compare against the `v164` claim "122 core + 2 CLI OBB tests passed" using `test_host.sh obb_`).

## Wave 1 — OBB CPU training gaps

All four tasks edit overlapping files; run strictly in order.

### Task 1.1: Adam for CPU OBB training

**Files:** Modify `src/darknet-core/src/train.rs:564,741`, `src/darknet-core/src/runtime.rs` (host Adam entry), `src/darknet-cli/src/main.rs:2400`; tests in `train.rs` tests.

**Interfaces:**
- Consumes: `darknet_cuda::host_ops::AdamParams` (`lr, beta1, beta2, eps, decay, batch, step, decoupled_decay`), `apply_host_adam` (`runtime.rs:13236`), `ExecutableNetwork::adam_step()/set_adam_step()`.
- Produces: `ExecutableNetwork::update_weights_adam(&mut self, learning_rate: f32, batch: usize) -> Result<()>` (host-only; increments `adam_step` exactly once).

- [x] **Step 1 (spike):** host Adam can be applied without a GPU backend using the existing `apply_host_adam*` helpers; implementation and evidence are in commit `f9664e63`.
- [x] **Step 2:** Tests (names and assertions):
  - `obb_adam_step_changes_weights_and_advances_step_once`: 1-conv OBB net, `adam=1`, `subdivisions=2`; after `train_obb_yolo_subdivisions`, weights differ and `adam_step()==1`.
  - `obb_adam_zero_lr_is_a_noop`: `learning_rate=0` leaves weights equal.
  - `obb_adam_matches_host_ops_reference`: for one weight, result equals `host_ops::adam_update` with the same inputs.
  - `obb_adam_moments_round_trip_through_sidecar`: save/load optimizer state; next step equals the uninterrupted run.
- [x] **Step 3–6:** Implemented/verified and committed as `f9664e63` (`v0.0.249: OBB CPU Adam`); Adam tests and OBB host-filter coverage passed.

### Task 1.2: Photometric augmentation for OBB training

**Files:** Modify `src/darknet-core/src/train.rs` (`load_obb_training_batch_with_flips_from_paths`, ~1872), `src/darknet-cli/src/main.rs:2406-2418`.

**Interfaces:** Consumes the existing photometric path in `augmentation.rs` (`apply`, line 257) — reuse; do not copy it. Produces: loader accepts `hue/saturation/exposure` from `DetectorAugmentation`; geometric options (`jitter, resize, letter_box, mixup, mosaic, mosaic_bound, random`) still error.

- [x] **Step 1–5:** Implemented/reused the photometric augmentation path, retained geometric-option rejection, and committed as `5b98fe1c` (`v0.0.250: OBB photometric augmentation`); focused and OBB-filtered tests passed.

### Task 1.3: Weighted OBB NMS merge

**Files:** Modify `src/darknet-core/src/model.rs:881-906` (`ObbNmsKind`), `src/darknet-core/src/detect.rs` (NMS dispatch near line 165-195).

**Interfaces:** Produces `ObbNmsKind::Weighted` parsed from the source spelling found in step 1.

- [x] **Step 1:** The plan's `yolo_layer.cpp` path was stale. Located the routine with `git show raptor:src-lib/box.cpp` (`diounms_sort`, around line 2000); formula, enum spelling, and cfg aliases are recorded in `STATUS.md`. The reference exists, so continue.
- [x] **Step 2–4:** Added the four weighted-merge tests and implementation; committed as `968c1beb` (`v0.0.251: OBB weighted NMS merge`). The later OBB jitter test failure was fixed by v0.0.252.

### Task 1.4: `ciou_obb_mul` training path

**Files:** Modify `src/darknet-core/src/obb_training.rs` (box regression), `src/darknet-core/src/model.rs` (the rejection test at ~5045 flips to acceptance).

- [x] **Step 1:** Read `raptor:src-lib/box.cpp` and `raptor:src-lib/yolo_layer_obb.cpp`; the source formula is documented in `STATUS.md`.
- [x] **Step 2:** Added the three-geometry/six-coordinate finite-difference check, `ciou_obb_mul_differs_from_additive`, and parse/build-config assertions.
- [x] **Step 3–4:** Implemented the distinct multiplicative loss/gradient and routed box/front channels; `ciou_obb_mul` and the 135-core + 3-CLI `obb_` filter pass uncapped.
- [x] **Step 5:** Committed as `31219aba` (`v0.0.253: OBB ciou_obb_mul training`).

## Wave 2 — Execute existing plans

### Task W2.1: Audit `onnx.md` and `gui_plan.md` for dispatchability

- [x] Read both plans and audit paths, tests, and verification commands. Most ONNX implementation steps are dispatchable; numeric verification's canonical LegoGears assets are missing locally, so equivalent checked-in cfg/weights and artwork were used. GUI manual/GPU steps remain externally gated: the named DarkMark fixture and LegoGears dataset assets are absent, and GPU checks require oxide hardware. The iced build gate's 3G cap conflicts with its global 2G iced/wgpu cap; resolve that before GUI dispatch.
- [x] Dispatch decision: W2.2 was executed; defer the GUI plan pending user confirmation and its data/hardware prerequisites. No plan expansion was needed for the ONNX implementation.

### Task W2.2: Execute `onnx.md` (21/21 steps)

- [x] Implemented and verified Tasks 1–5 in commits `fc50aa57`–`778242ab` (`v0.0.254`–`v0.0.257`); documentation and the completed checklist are committed as `2b998051` (`docs(rust): complete ONNX export plan`). `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; the CLI filter passed 1 test; host CLI build passed. ONNX Runtime numeric checks against Rust CPU dumps: fused `3.257e-06`, unfused `3.908e-05`, Mish variant `3.831e-06` worst relative errors (all ≤ `1e-4`). No compiled-C++ parity is claimed.

`gui_plan.md` is dispatched only after W2.2 completes and the user confirms the GUI is wanted now.

## Scope

In scope (only this remains):
1. **Layer parity** — port the layers still missing from Rust, each under the Layer Port Contract (`layers.md:61`): config parse, output shape, `.weights` order, forward, exact backward, `GradMode::Cpp` deviations only behind the flag.
2. **Axis alignment** — for every layer (new and already ported) pin the axis/layout semantics to the C++ source with a named test.

3. **GPU for all (end goal)** — Stage 1 ports every missing layer on CPU only; Stage 2 then gives every layer, including those on a host-fallback today, a native cuda-oxide kernel with a CPU-parity test.
Release lines: `v0.1.NNN` = Stage 1 (CPU parity of all layers + CPU Muon); `v0.2.NNN` = Stage 2 (all GPU paths + GPU Muon); `v0.3.NNN` = sparse MoE (`MOE.md`). `v0.0.NNN` closed at `v0.0.274`; each line starts only after the previous is complete.

4. **Muon optimizer** on every 2D weight matrix (conv as `[out, in*kh*kw]`, `connected`, projections of attention/MLP/MoE/EML/`dcnv4`/`laya_*` layers); 1D params stay on Adam/AdamW. Spec first (no C++ reference exists), CPU then GPU, state in the DNOPT sidecar.

Dropped from this plan (not deleted from `todo.md`, just not dispatched here): GUI, CARPN Stage-2 training and `detector train -map`, OBB GPU training/`-map`, recurrent BN device move, FP8/FP4/wgmma/NVFP4/multi-GPU, CUDA graph capture, real convergence runs.

## Axis alignment checklist

Applies to each layer in the backlog below, and as an audit pass over ported layers that reduce, split, or concatenate along an axis. Each row needs a test that fails when the axis is wrong (non-square, non-power-of-two sizes; `batch>1`; channels != spatial).

| Check | What to pin | Typical bug |
|---|---|---|
| Tensor layout | NCHW flat index `((n*C + c)*H + h)*W + w` for input, output, weights, deltas | NHWC assumption in token layers |
| Token layout | `[vit]`/`swin`/`mambavision`: C++ order of `(h,w)` -> token index and channel-major vs token-major | transposed token/channel |
| Reduction axis | LayerNorm / softmax / RMSNorm reduce over the C++ axis (channel vs last dim), `eps` placement | reducing over wrong dim |
| Head split | attention heads: `C -> heads x head_dim` order, Q/K/V packing order, output merge order | head/dim interleave swapped |
| Concat/slice axis | `route`, `channel_slice`, `shortcut` crop/pad take shape from the named source layer | shape from previous layer |
| Scan/window axis | `mambavision` scan direction, `swin` window partition and shift roll direction | roll sign, window row/col swap |
| Offset/grid axis | `deform_conv`/`dcnv4`: offset channel order `(dy,dx)` vs `(dx,dy)`, group split, sampling coordinate (x=w, y=h) | swapped x/y |
| Weights axis | stored weight layout `[out][in][kh][kw]` vs C++ save order per layer | transposed matrix |
| Batch axis | `batch>1` uses independent samples; no cross-sample reduction unless C++ does it | leaked statistics |

- [x] **A1:** Audited every already-ported family named by `AGENTS.md` (`vit`, `swin`, `tucker_attention`, `mambavision`, `kimi_kda`, `moe_convolutional`, `eml_convolutional`, `dcnv4`, `laya_*`, `channel_slice`, `deconvolutional`, `shortcut`) with `git grep` + `git show Jev_laya:<path>`; documented each axis contract and named pinning test in `STATUS.md` and added the missing hand-computed tests. `MEM=3G ./tools/test_host.sh axis_layout` passed 19 core tests: 18 A1 tests plus DPT's backlog axis test. No behavior changes or compiled-C++ parity claims; remaining missing-layer backlogs still carry their own axis DoDs.
- [x] **A1.5 (v0.1.015):** Coverage is machine-checked. `LayerKind::axis_pin` is an exhaustive match over `AXIS_PINS`, so a new variant fails the build until it records `Pinned` / `Open` / `RustOnly`; `model::tests::every_layer_kind_has_an_axis_pin_decision` pins the table, and `axis_fixture.rs` supplies the shared NCHW sentinel/index helpers. Baseline: 18 variants pinned by 19 tests, 41 C++-backed variants open, 5 Rust-only extensions, 3 unported. Evidence ledger: `axis-coverage.md`.
- [ ] **A2:** Per backlog layer below, the axis rows above are part of that layer's Definition of Done. The open tiers in `axis-coverage.md` are the standing queue; a behavior fix found by a new pin gets its own version.

Evidence rule: write "hand-derived from source, no compiled-C++ golden". Never claim C++ parity.

## Layer parity backlog (expand before dispatch)

Each item needs its own writing-plans pass (Opus) using the Layer Port Contract plus the axis checklist. Existing sub-plans: `vit_plan.md`, `swin_plan.md`, `tucker_attention_plan.md`, `mambavision_plan.md`, `kimi_kda_plan.md`.

**Scope check:** `depth_3d` at `9c9e0b4b` confirms `[yolo_3d]` is a real C++ layer with CPU/GPU code, training loss, decode, loader integration, configs, and tests, so the conditional v0.1 item in `AGENTS.md` is in scope. The evidence-only audit is `tasks/rust/yolo_3d_scope_audit.md`; it is not an Opus plan and does not authorize dispatch. Its metric/depth path depends on `[dpt_head]`, and the `use_3d`-is-reserved comment in `yolo_3d.hpp` is contradicted by the implementation/tests.

Current Stage-1 gaps, in dependency order:

1. **`[yolo_3d]`** — scope is confirmed; `[dpt_head]` is complete in v0.1.013. `yolo_3d_plan.md` is accepted for dispatch with completed source transcription S1–S13 in `yolo_3d_notes.md`; it supersedes the draft. The config-helper portion of Step 1 is implemented in v0.1.024, standalone Step-2 label/truth helpers in v0.1.025, and raw network/runtime integration in v0.1.026 (`implemented.md` §§38–40). The latter has typed cuboid head capture, shared class/K checks, exact raw identity backward, weightless checkpoint/seeded optimizer checks and fail-closed legacy API/CLI/GPU guards. Projected math/assignments/loss with exact derivatives, truth/map/counter consumption, mixed loss routing, decode/evaluator and the A2 output-axis pin remain open before depth/metric CPU support. Seeded optimizers are not loss-to-conv training evidence. The `yolo_3d` gate is 42 core + 1 CLI; axis stays Open and GPU none. Follow the plan, but allocate fresh versions from `version-map.tsv` rather than its stale projected numbers. The user requested direct parent implementation without subagents for this continuation.
2. **DSConv multi-template `morph=0/1` is required under full source parity** — executable C++ CPU forward averages `2*templates` views and may drop views during training, with separate extra offset-weight blocks (`Jev_laya:src-lib/dsconv_layer.cpp:416-433,597-638`). Rust v0.1.002 explicitly rejects the mode (`implemented.md` §19; `todo.md`). Add it as a separate v0.1 backlog item with an Opus planning pass; the plan must cover active-view cache/averaging and exact Rust gradients for every view/offset block. CPU C++ backward is a stub and CUDA multi-template backward is incomplete, so do not claim compiled-C++ gradient parity.
3. **All-layer axis contracts and A2 tests** — the bounded A1 list is complete in v0.1.012 and DPT's pin is in v0.1.013, not the all-layer gate. The current registry still has 41 C++-backed Open variants, alongside 19 pinned variants and 5 Rust-only kinds (plus Unsupported). Resolve applicable Open contracts from `axis-coverage.md` and include source-anchored A2 layout tests in each missing or expanded layer's DoD before Stage 1 closes.

**Resolved C++ enum boundary (not a layer gap):** `NETWORK` is cfg metadata (`Jev_laya:src-lib/darknet_cfg.cpp:817-820`); `BLANK` is an internal GPU-fusion sentinel (`darknet_network.cpp:2662-2672`); `EMPTY`/`SILENCE` is marked unused and, like `[blank]`, falls through the C++ layer parser's unsupported default (`darknet_enums.hpp:51-52`, `darknet_enums.cpp:60-62`, `darknet_cfg.cpp:947-1157`). Rust accepts `[network]` as metadata and rejects `[blank]`, `[empty]`, and `[silence]` rather than treating them as identity layers. Regression test `model::tests::unused_cpp_layer_enum_names_are_not_rust_noop_layers` passed with `MEM=3G ./tools/test_host.sh unused_cpp_layer_enum_names_are_not_rust_noop_layers`.

Do not start v0.2 until all v0.1 CPU/variant gaps are resolved. Do not start v0.3 until v0.2 (all native GPU paths plus GPU Muon) is complete. The sparse MoE CPU/GPU extension remains sequenced in `MOE.md`.

Done when: every in-scope v0.1 layer and required source variant is parse+shape+weights+forward+backward tested, its axis rows pinned (A2), and `todo.md`/`implemented.md`/`STATUS.md`/`version-map.tsv` updated per version.

## Out of scope / blocked

- Evidence: `[dcnv3]` (no reference numerics), `detr_decoder` (C++ broken, paper only), contrastive detection branch (needs per-box class ids). Not part of parity until a reference exists.
- Hardware and non-layer work: see [Scope](#scope) for the dropped list.

## Verification

- Per task: the stated filter fails first, then passes; then `MEM=3G ./tools/test_host.sh obb_` (OBB tasks) or full `./tools/test_host.sh` (Wave 0).
- Wave 0 end: `cargo check -p darknet-cli --features oxide -j 4` exits 0.
- Wave 1 end: `MEM=3G ./tools/test_host.sh obb_` passed 135 core + 4 CLI tests, including an end-to-end OBB `detector train` CLI test with `adam=1`, exposure augmentation, and two iterations. `Running scope as unit: run-r5e0590e5177c4b5f9508c6e86ca4f304.scope` confirms the host filter ran under the user systemd scope.
- Bookkeeping per wave: tick `todo.md`, add to `implemented.md` and `STATUS.md` "Latest increments" (vNNN entry stating tests run and "no compiled-C++ parity"), update `version-map.tsv`.

## Self-review

- Spec coverage: every open layer-parity item in `todo.md` appears in the backlog or Out of scope; non-layer items are explicitly dropped in Scope.
- Type consistency: `from_heads`, `update_weights_adam`, `ObbNmsKind::Weighted` are defined once and referenced by name only.
- Known weakness: Tasks 1.3 and 1.4 depend on reference source in `raptor*` branches I did not open while planning; each has an explicit stop condition. Task 1.1's spike resolves how entangled host Adam is.
- Proportion: Waves 0–1 are bite-sized; everything XL is deliberately a stub, matching `layers.md`'s own rule.

## Review follow-up (2026-10-17)

Verification of this plan against git and source found the following; dispositions:

1. **Bookkeeping** — fixed: `version-map.tsv` rows 247–253 added (`old_sha` is `-`, no pre-rewrite sha exists) and `implemented.md` section 13 added. `STATUS.md` already had the entries.
2. **v0.0.248 gap** — `2941a2ee` (`.cargo/config.toml` CUDA path) is outside the plan; recorded in Context and `implemented.md`. The gap is intentional; after ONNX Wave 2 and CARPN sample dispatch v0.0.258, next is `v0.0.259`.
3. **Heading/TOC mismatch** — fixed: Wave 0 TOC entry now matches the heading "Baseline establishment"; task `###` entries added under Waves 0–2.
4. **Stale Context** — fixed: resolved blockers marked FIXED with commits; "Last version commit" updated to `v0.0.253`.
5. **Baseline not green** — unchanged and still true: `cfg_corpus` has 4 unresolved resource/runtime failures (2 SIGKILL/OOM on CSPX, 2 timeouts on lstm/rnn). The full host baseline was not rerun under the 3G cap. Do not claim green.
6. **Uncommitted files** — `STATUS.md`, `todo.md`, `Cargo.toml`, `Cargo.lock`, `.gitignore` carry uncommitted edits (plus the files touched by this follow-up). Commit only task-named files; leave `.gitignore` alone.
