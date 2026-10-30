# fp16 Training Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `cargo xtask train_legogears -fp16` a safe, measured mixed-precision (f16) training mode for LegoGears_v2.

**Architecture:** The f16 AMP path already exists in the oxide backend (`LIGHTNET_OXIDE_AMP=1`, `src/lightnet-cuda-oxide/src/mixed.rs`). This plan adds what it lacks: a tested loss scaler, an initial loss scale from the command line, detection of overflow in every gradient the AMP path produces, and a recorded convergence comparison against fp32.

**Tech Stack:** Rust (workspace crates `lightnet-core`, `lightnet-cli`, `xtask`; non-workspace crate `lightnet-cuda-oxide`), cuda-oxide kernels, sm_89 GPU.

**Spec:** [ROADMAP_FP8.md](ROADMAP_FP8.md) (goal and rules), section "Phase 1: fp16".

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Verified starting state](#verified-starting-state)
- [File structure](#file-structure)
- [Tasks](#tasks)
  - [Task 1: LossScaler type](#task-1-lossscaler-type)
  - [Task 2: CLI uses LossScaler and takes -loss_scale](#task-2-cli-uses-lossscaler-and-takes--loss_scale)
  - [Task 3: Overflow detection covers every AMP gradient](#task-3-overflow-detection-covers-every-amp-gradient)
  - [Task 4: xtask -fp16 passes the scaler flags](#task-4-xtask--fp16-passes-the-scaler-flags)
  - [Task 5: Convergence and step-time record](#task-5-convergence-and-step-time-record)

## Global Constraints

- Build and test only through `cargo xtask` (`check`, `test`, `oxide`, `clippy`, `fmt`); one build or test command at a time, never two in parallel.
- Max 4 build threads (`JOBS` is clamped to 4 except for `train_legogears`, which the operator set to 8); soft memory cap `MemoryHigh=3G` (applied by `cargo xtask`; it warns and runs uncapped without a user systemd bus).
- `src/lightnet-cuda-oxide` is not a workspace member: its tests run only through `cargo xtask oxide --test <file> [-- <filter>]`.
- Commit only the files a task names; never `git add -A`. `src/lightnet-core/src/runtime/mod.rs` may carry unrelated user edits: do not stage it.
- Do not start a training run unless the user asks; Task 5 is user-gated.
- Delete weights, logs and sidecars a run produces once the evidence is recorded.
- Every version-mapped increment updates `implemented.md`, `STATUS.md`, `todo.md` and `version-map.tsv` (next free `v0.2.NNN` number from `version-map.tsv`; never reuse a number).
- No parity claim without a golden; fp16 results are compared against the Rust fp32 run, not against C++.

## Review Focus

- Loss scale overflow on the very first step (scale 1024, large initial gradients): the step is skipped, nothing is written to the weights, the run continues.
- A non-finite gradient that exists only in a BN scale or bias gradient (not a conv weight gradient): must still skip the step.
- A user passing both `-loss_scale 512` and `-dyn_scale`: the given value is the starting scale, not overwritten by the cfg value.
- `-loss_scale` without `-dyn_scale`: static scaling, never changes.
- `-loss_scale 0`, negative, NaN or inf: rejected with a clear error, not silently ignored.

## Verified starting state

These were read from the code (the exploration agents' summaries were checked; one was wrong about dynamic scaling):

- AMP: `with_amp` and `amp_from_env` at `src/lightnet-cuda-oxide/src/mixed.rs:46,69`; AMP also forces `GemmMode::Fp16`.
- Loss scale in the runtime: `net.loss_scale` parsed at `src/lightnet-core/src/model.rs:296` (default 1.0); seeds multiplied in `runtime/gpu_backward.rs:25-43`; divided out in `runtime/gpu_update.rs` and `runtime/exec_core.rs` (`lr/loss_scale`, `decay*loss_scale`, `eps*loss_scale`).
- Dynamic scaling already exists in the CLI as the `-dyn_scale` flag: `src/lightnet-cli/src/main.rs:1274` (flag), `:3131-3156` (loop: on non-finite gradients zero the buffers, halve the scale with floor 1.0, skip the step; after 1000 finite steps double the scale up to 65536). It is inline in the training loop and untested. The initial scale comes from the cfg `loss_scale` (default 1.0).
- `GpuExecutableNetwork::loss_scale`, `set_loss_scale`, `gradients_non_finite` are in `src/lightnet-core/src/runtime/gpu_state.rs:91-106`. `gradients_non_finite` inspects device conv and MoE gradient buffers and, per its doc comment, "other host-only layer families are not inspected".
- xtask: `-fp16` currently only sets `LIGHTNET_OXIDE_AMP=1` (`xtask/src/main.rs`, `precision_env`).
- `docs/OXIDE_BENCH.md` section 7.2: AMP is 1.3x faster at batch 16, 27% less device memory; "no convergence run".

## File structure

- Create `src/lightnet-core/src/runtime/loss_scaler.rs`: the `LossScaler` type and its unit tests.
- Modify `src/lightnet-core/src/runtime/mod.rs` (add `mod loss_scaler;` and a re-export; this file may have unrelated user edits: stage only your hunk with `git add -p`).
- Modify `src/lightnet-cli/src/main.rs`: use `LossScaler` in the training loop; add the `-loss_scale <f32>` option to `TrainOptions`.
- Modify `src/lightnet-core/src/runtime/gpu_state.rs`: widen `gradients_non_finite` if Task 3 finds a gap.
- Modify `xtask/src/main.rs`: `-fp16` adds scaler flags.
- Modify `docs/OXIDE_BENCH.md`, `tasks/rust/02_state/implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`, `CLAUDE.md` (Commands table).

## Tasks

### Task 1: LossScaler type

**Files:**
- Create: `src/lightnet-core/src/runtime/loss_scaler.rs`
- Modify: `src/lightnet-core/src/runtime/mod.rs` (module declaration and `pub use`)
- Test: unit tests inside `loss_scaler.rs`

**Interfaces:**
- Produces: `pub enum ScaleStep { Apply, Skip }` and `pub struct LossScaler` with `pub fn new(initial: f32) -> Self`, `pub fn scale(&self) -> f32`, `pub fn update(&mut self, grads_finite: bool) -> ScaleStep`. Behaviour is the CLI loop's, lifted verbatim: on non-finite gradients, scale becomes `max(scale * 0.5, 1.0)`, the growth counter resets, result is `Skip`; on finite gradients, the growth counter increments, and when it reaches 1000 and the scale is below 65536 the scale doubles and the counter resets; result is `Apply`.

- [x] **Step 1: Write the failing tests** in `loss_scaler.rs`: `halves_and_skips_on_overflow` (new(1024.0), `update(false)` returns `Skip`, `scale()` is 512.0), `never_drops_below_one` (new(1.0), `update(false)` leaves `scale()` at 1.0), `doubles_after_1000_finite_steps` (999 finite updates leave the scale unchanged; the 1000th doubles it), `growth_counter_resets_after_overflow` (500 finite, one overflow, then 999 finite leaves the scale unchanged), `caps_at_65536` (new(65536.0) never doubles).
- [x] **Step 2: Run to verify it fails.** `PKGS=lightnet-core cargo xtask test loss_scaler` — Expected: compile error, type not defined.
- [x] **Step 3: Implement** `LossScaler` as described in Interfaces, in `loss_scaler.rs`; one `u32` growth counter field.
- [x] **Step 4: Run to verify it passes.** Same command — Expected: 5 passed.
- [x] **Step 5: Commit** `loss_scaler.rs` and the `mod.rs` hunk: `git add src/lightnet-core/src/runtime/loss_scaler.rs && git add -p src/lightnet-core/src/runtime/mod.rs`, message `feat(core): add LossScaler with unit tests`.

### Task 2: CLI uses LossScaler and takes -loss_scale

**Files:**
- Modify: `src/lightnet-cli/src/main.rs` (`TrainOptions` near line 1222; option parsing near line 1274; training loop near lines 2955 and 3131-3156)
- Test: the existing `TrainOptions` parse tests in `src/lightnet-cli/src/main.rs` (find them with `grep -n "fn parse_train" src/lightnet-cli/src/main.rs`)

**Interfaces:**
- Consumes: `LossScaler::{new, scale, update}` and `ScaleStep` from Task 1; `GpuExecutableNetwork::set_loss_scale`.
- Produces: `TrainOptions.loss_scale: Option<f32>`, set by `-loss_scale <f32>`.

- [ ] **Step 1: Write the failing tests** next to the existing parse tests: `parse_train_loss_scale_option` (`-loss_scale 512` gives `Some(512.0)`), `parse_train_loss_scale_rejects_non_positive_and_non_finite` (`0`, `-4`, `nan`, `inf` each return `Err`), `parse_train_without_loss_scale_is_none`.
- [ ] **Step 2: Run to verify it fails.** `cargo xtask test parse_train_loss_scale` — Expected: FAIL (field and option do not exist).
- [ ] **Step 3: Implement.** Add the `loss_scale` field and `-loss_scale` / `--loss_scale` parsing (reject values that are not finite or not greater than 0, same error style as `-save_every`). Before the training loop, if `options.loss_scale` is `Some(v)`, call `executable.set_loss_scale(v)`. Replace the inline `loss_scale_growth` logic with a `LossScaler::new(executable.loss_scale())` created after that call; for `ScaleStep::Skip` keep the existing zero-buffers-and-println behaviour, and apply `scaler.scale()` with `set_loss_scale` after each `update`.
- [ ] **Step 4: Run to verify it passes.** `cargo xtask test parse_train` then `cargo xtask check` — Expected: tests pass, no warnings.
- [ ] **Step 5: Commit** `src/lightnet-cli/src/main.rs` with `feat(cli): -loss_scale option; training loop uses LossScaler`.

### Task 3: Overflow detection covers every AMP gradient

**Files:**
- Modify: `src/lightnet-core/src/runtime/gpu_state.rs` (only if the test below fails)
- Test: `src/lightnet-core/src/runtime/tests/tests_06.rs` (it already hosts `set_loss_scale` tests near line 2056)

**Interfaces:**
- Consumes: `GpuExecutableNetwork::gradients_non_finite() -> Result<bool>`.

- [ ] **Step 1: Write the failing test** `gradients_non_finite_sees_bn_scale_gradient`: build a small conv + batch-norm GPU network (copy the construction used by the neighbouring loss-scale tests), run backward with a seed that makes only the BN scale gradient non-finite, assert `gradients_non_finite()` is `true`. If the harness cannot isolate the BN gradient, assert the same for a conv-weight gradient and a BN-bias gradient in two further tests with the names `gradients_non_finite_sees_conv_weight_gradient` and `gradients_non_finite_sees_bn_bias_gradient`.
- [ ] **Step 2: Run to see the result.** `cargo xtask test gradients_non_finite` — Expected: either PASS (detection already covers BN, record that in the commit message) or FAIL (gap found).
- [ ] **Step 3: If it failed, extend `gradients_non_finite`** to scan the BN scale, bias and mean/variance gradient buffers, using the same `backend` non-finite helper it already calls for conv buffers.
- [ ] **Step 4: Run to verify it passes.** Same command — Expected: all pass.
- [ ] **Step 5: Commit** the test file (and `gpu_state.rs` if changed), `test(core): pin non-finite gradient detection for conv and BN buffers`.

### Task 4: xtask -fp16 passes the scaler flags

**Files:**
- Modify: `xtask/src/main.rs` (`precision_env`)
- Test: `#[cfg(test)]` module in `xtask/src/main.rs`

**Interfaces:**
- Consumes: the CLI flags `-dyn_scale` and `-loss_scale <f32>` from Task 2.
- Produces: `precision_env` returns the extra training arguments in addition to the env pairs (the return type gains a third element, `Vec<String>`, appended after the pass-through args).

- [ ] **Step 1: Write the failing tests:** `fp16_adds_dynamic_loss_scale_flags` (`["-fp16"]` yields env `LIGHTNET_OXIDE_AMP=1` and extra args `["-dyn_scale", "-loss_scale", "1024"]`), `fp16_respects_user_loss_scale` (`["-fp16", "-loss_scale", "256"]` adds only `-dyn_scale`), `fp16_respects_user_dyn_scale` (`["-fp16", "-dyn_scale"]` does not repeat it), `fp8_adds_no_scaler_flags`.
- [ ] **Step 2: Run to verify it fails.** `PKGS=xtask cargo xtask test` — Expected: FAIL.
- [ ] **Step 3: Implement** the extra args in `precision_env`; update the `train_legogears` arm to append them; update the help text and the `CLAUDE.md` Commands row for `train_legogears`.
- [ ] **Step 4: Run to verify it passes.** `PKGS=xtask cargo xtask test` — Expected: 4 passed.
- [ ] **Step 5: Commit** `xtask/src/main.rs` and `CLAUDE.md`, `feat(xtask): -fp16 enables dynamic loss scaling at 1024`.

### Task 5: Convergence and step-time record

**Files:**
- Modify: `docs/OXIDE_BENCH.md` (new subsection under section 7.2), `tasks/rust/02_state/implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`

User-gated: ask the user to run (or confirm you may run) the two trainings; do not start them unprompted.

- [x] **Step 1: Baseline run.** `cargo xtask train_legogears -map_every 100` (fp32), note the final `avg loss`, the `-map` results, and the `train=... milliseconds` column. Stop after the iteration count the user agrees on.
- [x] **Step 2: fp16 run.** `cargo xtask train_legogears -fp16 -map_every 100` with the same seed and iteration count.
- [x] **Step 3: Compare** loss at equal iterations (at least iterations 100, 500 and the final one), mAP, step time, and the number of `non-finite gradients` skip lines. Pass criterion: fp16 loss within 10% of fp32 at the final iteration and mAP within 2 points; otherwise record the failure and stop (do not start Phase 2 claims).
- [x] **Step 4: Record** the table and verdict in `docs/OXIDE_BENCH.md`, and the version entry in `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`. Delete the weights and logs.
- [x] **Step 5: Commit** the doc files, `docs: record fp16 convergence run vs fp32`.
