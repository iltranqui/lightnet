# bf16 Mixed-Precision Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cargo xtask train_legogears -bf16` trains LegoGears_v2 in bf16 mixed precision (bf16 activations and gradients, f32 master weights), reusing the f16 AMP machinery.

**Architecture:** One 16-bit format per backend, `HalfFormat { F16, Bf16 }`, chosen by `LIGHTNET_OXIDE_AMP`. Every device kernel that reads or writes 16-bit data takes a `fmt: u32` argument (0 = f16, 1 = bf16), the same pattern `hgemm_f16_mma` already uses for its `bf16: u32` flag. `OxideTensorF16` stays format-agnostic; the backend knows the format.

**Tech Stack:** Rust, cuda-oxide kernels (`cuda_device::convert::cvt_bf16x2_f32`, `mma_m16n8k16_f32_bf16`), sm_89.

**Spec:** [ROADMAP_FP8.md](ROADMAP_FP8.md), section "Phase 2: bf16". Do Phase 1 ([plan_fp16.md](plan_fp16.md)) first.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Verified starting state](#verified-starting-state)
- [File structure](#file-structure)
- [Progress](#progress)
- [Tasks](#tasks)
  - [Task 1: HalfFormat, env parsing, GemmMode::Bf16](#task-1-halfformat-env-parsing-gemmmodebf16)
  - [Task 2: Format-aware conversions](#task-2-format-aware-conversions)
  - [Task 3: bf16 output in the 16-bit GEMM](#task-3-bf16-output-in-the-16-bit-gemm)
  - [Task 4: amp.rs kernels take fmt](#task-4-ampers-kernels-take-fmt)
  - [Task 5: Training half_ops kernels take fmt](#task-5-training-half_ops-kernels-take-fmt)
  - [Task 6: Backend wiring and the AMP suite for bf16](#task-6-backend-wiring-and-the-amp-suite-for-bf16)
  - [Task 7: xtask -bf16](#task-7-xtask--bf16)
  - [Task 8: Convergence and step-time record](#task-8-convergence-and-step-time-record)

## Global Constraints

- Same constraints as [plan_fp16.md](plan_fp16.md#global-constraints): `cargo xtask` only, one build or test at a time, 4 build threads (8 for training), `MemoryHigh=3G`, stage only named files, no unprompted training runs, delete weights and logs after use, update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv` for each version-mapped increment.
- `src/lightnet-cuda-oxide` tests run only through `cargo xtask oxide --test <file> [-- <filter>]`.
- f16 behaviour must not change: every existing test in `oxide_amp.rs`, `oxide_half.rs`, `oxide_runtime_f16.rs` passes unmodified at every task boundary.
- bf16 mode uses no loss scaling (same exponent range as f32); `-bf16` must not add `-dyn_scale`.
- bf16 has an 8-bit significand against f16's 11: tolerances for bf16 tests are 8x the f16 bounds, no looser.

## Review Focus

- A bf16 run that reaches an f16-only code path (inference `forward_compact_f16`, `half.rs` f16 inference ops) must fail with a clear error, not silently reinterpret bf16 bits as f16.
- bf16 value rounding: round-to-nearest-even, NaN stays NaN, large f32 values (above f16 max 65504) stay finite in bf16, infinities stay infinities.
- `LIGHTNET_OXIDE_AMP` values `bf16`, `BF16`, `fp16`, `1`, `0`, empty, garbage: garbage and empty mean AMP off, not a panic.
- `LIGHTNET_OXIDE_GEMM=fp8` together with `LIGHTNET_OXIDE_AMP=bf16`: the AMP format wins and the combination is reported, not silently ignored (fp8 plan owns the combination; here it must at least not crash).
- Weight cache refresh after an optimizer step in bf16 (stale bf16 copies would train on old weights).

## Verified starting state

- bf16 exists only as: `mma_m16n8k16_f32_bf16` selected by the `bf16: u32` flag of `hgemm_f16_mma` (`src/lightnet-cuda-oxide/src/kernels/gemm_f16.rs:53,334-338`; operands bf16, output f32 only); `cvt_f32_to_bf16`, `cvt_bf16_to_f32` (`kernels/convert.rs:81,90`); host `f32_to_bf16_bits`, `bf16_bits_to_f32`, `f32_to_bf16`, `bf16_to_f32`, `gemm_bf16` (`half.rs:93,102,245,263,359`); `hgemm_ex(a, b, out: HOut, g: &GemmArgs, bf16: bool)` at `half.rs:447`, which rejects bf16 with a non-f32 output at `half.rs:459`. Tests: `gpu_conversions_match_host_reference`, `gemm_bf16_parity_all_transposes_alignments_alpha_beta` in `tests/oxide_half.rs`.
- The f16 epilogue stores are at `gemm_f16.rs:367,417` using `cvt_f16x2_f32`; the split-K reduce is `hgemm_splitk_reduce_act16` (`gemm_f16.rs:379`).
- AMP training launches these kernels (from `mixed.rs`): `amp_bias_add`, `amp_activation`, `amp_activation_backward`, `amp_vector_add`, `amp_sum_partial`, `amp_sq_dev_partial`, `amp_dot_partial`, `amp_bn_normalize`, `amp_bn_backward_dx`, `amp_col2im`, `amp_maxpool_backward`, `amp_upsample_backward` (all in `kernels/amp.rs`), plus `half_ops.im2col_f16`, `half_ops.route_copy_f16`, and `hgemm`. The forward maxpool and upsample AMP branches are at `lib.rs:7603` and `lib.rs:9888` and use `maxpool_f16` and `upsample_nearest_f16` (`kernels/half_ops.rs:80,167`).
- f16 conversion helpers used by those kernels: `f32_to_f16_bits` (`kernels/convert.rs:16`), `f16_bits_to_f32` (`kernels/convert.rs:30`), imported as `h2f` in `amp.rs`.
- `amp_from_env() -> bool` at `mixed.rs:46`; `with_amp(bool)` at `mixed.rs:69` (sets `GemmMode::Fp16`); `GemmMode` at `lib.rs:140`, `from_env` at `lib.rs:168`, `use_tf32` at `lib.rs:184`.

## File structure

- Modify `src/lightnet-cuda-oxide/src/mixed.rs`: `HalfFormat`, `parse_amp_format`, backend field, call sites.
- Modify `src/lightnet-cuda-oxide/src/lib.rs`: `GemmMode::Bf16`, constructor wiring.
- Modify `src/lightnet-cuda-oxide/src/kernels/convert.rs`: format-aware device helpers and cvt kernels.
- Modify `src/lightnet-cuda-oxide/src/kernels/gemm_f16.rs`, `half.rs`: bf16 output.
- Modify `src/lightnet-cuda-oxide/src/kernels/amp.rs`, `kernels/half_ops.rs`: `fmt` arguments (training kernels only; inference-only f16 kernels stay f16).
- Modify tests: `tests/oxide_amp.rs`, `tests/oxide_half.rs`, `tests/oxide_gemm_f8.rs` (GemmMode parsing test lives there).
- Modify `xtask/src/main.rs`, `CLAUDE.md`, docs as in plan_fp16.

## Progress

Status as of 2026-10-22.

| Task | Status | Commit or tag |
|---|---|---|
| 1. HalfFormat, env parsing, GemmMode::Bf16 | done | `v0.2.170` (`97414f8`) |
| 2. Format-aware conversions | done | `v0.2.173` (`f983c8e`) |
| 3. bf16 output in the 16-bit GEMM | done | `v0.2.174` (`c51d5b9`) |
| 4. amp.rs kernels take fmt | done | `v0.2.175` (`dedebb2`) |
| 5. Training half_ops kernels take fmt | done | `v0.2.175` (`dedebb2`); only `maxpool_f16` and `upsample_nearest_f16` needed `fmt`, the copy kernels are format-agnostic |
| 6. Backend wiring and the AMP suite | done | `v0.2.175` (`dedebb2`) |
| 7. xtask `-bf16` | done | `v0.2.176` (`7ef8322`) |
| 8. Convergence and step-time record | done (run by the fp8/bf16 goal) | 2026-10-23, `docs/OXIDE_BENCH.md` section 10: 138.6 s, loss 0.0524, mAP 99.63%, F1 0.9935 against fp32 147.1 s |

Task 1 notes: `oxide_amp` passes 15 of 15 and `oxide_gemm_f8` 26 of 26. `GemmMode::Bf16` in the f32 API goes through `gemm_f32_operands_as_half(.., bf16 = true)`. `LIGHTNET_OXIDE_AMP=bf16` parses and selects the format, but the AMP activation kernels are still f16 only, so it must not be used for training before Task 4 to 6.

## Tasks

### Task 1: HalfFormat, env parsing, GemmMode::Bf16

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/mixed.rs`, `src/lightnet-cuda-oxide/src/lib.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_amp.rs`, `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Produces: `pub enum HalfFormat { F16, Bf16 }` with `pub fn code(self) -> u32` (F16 = 0, Bf16 = 1); `pub fn parse_amp_format(value: &str) -> Option<HalfFormat>` (`1`, `true`, `on`, `yes`, `fp16`, `f16` give F16, case-insensitive; `bf16`, `bfloat16` give Bf16; anything else None); `OxideBackend::with_amp_format(self, f: Option<HalfFormat>) -> Self`; `OxideBackend::half_format(&self) -> HalfFormat`; `GemmMode::Bf16`.
- `amp_from_env()` keeps returning `bool` (true when `parse_amp_format` is Some) so existing callers compile; `with_amp(on)` becomes `with_amp_format(on.then_some(HalfFormat::F16))`.

- [ ] **Step 1: Write failing tests:** `amp_format_parses_known_values_case_insensitively` (table of the values above), `amp_format_rejects_garbage_and_empty`, `gemm_mode_from_env_parses_bf16` (mirror `gemm_mode_from_env_parses_fp8_and_fp4` in `oxide_gemm_f8.rs`; `LIGHTNET_OXIDE_GEMM=bf16` gives `GemmMode::Bf16`).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_amp -- amp_format` — Expected: compile error (type missing).
- [ ] **Step 3: Implement** the items in Interfaces; add the `Bf16` arm to `use_tf32` (false) and any exhaustive `match` the compiler reports; `GemmMode::Bf16` behaves as `Fp16` in the f32-API GEMM path until Task 3 (route it through `gemm_bf16`).
- [ ] **Step 4: Run to verify they pass,** plus the whole `oxide_amp` and `oxide_gemm_f8` files — Expected: all pass.
- [ ] **Step 5: Commit** `mixed.rs`, `lib.rs`, the two test files: `feat(oxide): HalfFormat and GemmMode::Bf16 selection`.

### Task 2: Format-aware conversions

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/kernels/convert.rs`, `src/lightnet-cuda-oxide/src/half.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_half.rs`

**Interfaces:**
- Consumes: `HalfFormat::code()` from Task 1.
- Produces, device helpers in `convert.rs`: `pub fn half_bits_to_f32(bits: u16, fmt: u32) -> f32` and `pub fn f32_to_half_bits(x: f32, fmt: u32) -> u16` (fmt 0 uses the existing f16 helpers, 1 the bf16 conversion; round-to-nearest-even). New kernels `cvt_f32_to_half(.., fmt: u32)` and `cvt_half_to_f32(.., fmt: u32)`; the existing `cvt_f32_to_f16`, `cvt_f16_to_f32`, `cvt_f32_to_bf16`, `cvt_bf16_to_f32` stay and keep their tests. Host: `OxideBackend::to_half(&self, x: &OxideTensorF32, fmt: HalfFormat) -> Result<OxideTensorF16, OxideError>` and `OxideBackend::from_half(&self, x: &OxideTensorF16, fmt: HalfFormat) -> Result<OxideTensorF32, OxideError>`.

- [ ] **Step 1: Write failing tests:** `to_half_from_half_round_trip_both_formats` (values: 0, -0, 1, 65504, 70000, 1e-8, 3.0e38, inf, -inf, NaN; compare to `f32_to_f16_bits` / `f32_to_bf16_bits` host references bit-for-bit), `bf16_keeps_values_above_f16_max_finite` (70000 stays finite in bf16, becomes inf in f16).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_half -- to_half` — Expected: compile error.
- [ ] **Step 3: Implement** the helpers, kernels and host methods in the Interfaces block, in the style of the existing `cvt_*` kernels and `f32_to_bf16` (`half.rs:245`).
- [ ] **Step 4: Run to verify they pass,** then all of `oxide_half` — Expected: all pass.
- [ ] **Step 5: Commit** `convert.rs`, `half.rs`, `oxide_half.rs`: `feat(oxide): format-aware f32<->16-bit conversions`.

### Task 3: bf16 output in the 16-bit GEMM

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/kernels/gemm_f16.rs` (epilogue stores at lines 367 and 417; `hgemm_splitk_reduce_act16` at line 379), `src/lightnet-cuda-oxide/src/half.rs` (`hgemm_ex`, line 459)
- Test: `src/lightnet-cuda-oxide/tests/oxide_half.rs`

**Interfaces:**
- Consumes: `f32_to_half_bits` from Task 2.
- Produces: `hgemm_ex(a, b, out: HOut::F16(_), g, bf16: true)` is accepted and writes bf16 bits; `hgemm_splitk_reduce_act16` takes a trailing `fmt: u32`.

- [ ] **Step 1: Write the failing test** `gemm_bf16_with_bf16_output_matches_reference`: for the transposes and alignments the existing `gemm_bf16_parity_all_transposes_alignments_alpha_beta` covers, run with `HOut::F16` output and `bf16 = true`, widen with `bf16_bits_to_f32`, compare to the f64 reference with the bf16 bound (8x the f16 bound used by the neighbouring f16 test), including a split-K shape (small M, N, large K).
- [ ] **Step 2: Run to verify it fails.** `cargo xtask oxide --test oxide_half -- gemm_bf16_with_bf16_output` — Expected: FAIL with "bf16 operands support the f32 output only".
- [ ] **Step 3: Implement.** In the two epilogue stores branch on the existing `bf16` kernel argument (0: `cvt_f16x2_f32`, otherwise `f32_to_half_bits(.., 1)`); add `fmt` to the split-K reduce and pass `bf as u32`; remove the rejection in `hgemm_ex`.
- [ ] **Step 4: Run to verify it passes,** then all of `oxide_half` and `oxide_gemm_mma` — Expected: all pass, f16 tests unchanged.
- [ ] **Step 5: Commit** `gemm_f16.rs`, `half.rs`, `oxide_half.rs`: `feat(oxide): bf16 output from the 16-bit tensor-core GEMM`.

### Task 4: amp.rs kernels take fmt

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/kernels/amp.rs` (12 kernels listed in Verified starting state), their call sites in `src/lightnet-cuda-oxide/src/mixed.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_amp.rs`

**Interfaces:**
- Consumes: `half_bits_to_f32`, `f32_to_half_bits` from Task 2; `OxideBackend::half_format()` from Task 1.
- Produces: every `amp_*` kernel gains a trailing `fmt: u32` argument; every `*_amp` host method passes `self.half_format().code()`.

- [ ] **Step 1: Parametrize the existing tests:** add `amp_backend_fmt(fmt: HalfFormat) -> OxideBackend` to `oxide_amp.rs` and run each of `amp_bias_activation_bias_backward_vector_add`, `amp_pool_upsample_route_backward_match_f64_reference`, `amp_batchnorm_forward_backward_match_f64_reference`, `amp_conv_forward_dgrad_wgrad_match_f64_reference` for both formats through a loop, using bound `bound_f16` for F16 and `8.0 * bound_f16` for Bf16 (named `bf16_gradient_bounds_are_8x_f16` in a shared helper so the factor is stated once).
- [ ] **Step 2: Run to verify the bf16 cases fail.** `cargo xtask oxide --test oxide_amp` — Expected: the F16 cases pass, the Bf16 cases fail (values read as f16).
- [ ] **Step 3: Implement** the `fmt` argument in each kernel by replacing the direct `f16_bits_to_f32` / `f32_to_f16_bits` calls with the Task 2 helpers, and thread it through `mixed.rs`. Do the mechanical edit kernel by kernel, running the file after each three kernels.
- [ ] **Step 4: Run to verify it passes.** `cargo xtask oxide --test oxide_amp` — Expected: all pass, both formats.
- [ ] **Step 5: Commit** `amp.rs`, `mixed.rs`, `oxide_amp.rs`: `feat(oxide): amp kernels are 16-bit-format aware`.

### Task 5: Training half_ops kernels take fmt

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/kernels/half_ops.rs` (`im2col_f16`, `route_copy_f16`, `maxpool_f16`, `upsample_nearest_f16`), their callers in `src/lightnet-cuda-oxide/src/mixed.rs`, `src/lightnet-cuda-oxide/src/half.rs`, `src/lightnet-cuda-oxide/src/lib.rs` (AMP branches near lines 7603 and 9888)
- Test: `src/lightnet-cuda-oxide/tests/oxide_half.rs`, `src/lightnet-cuda-oxide/tests/oxide_amp.rs`

**Interfaces:**
- Produces: the four kernels gain a trailing `fmt: u32`. Inference-only f16 kernels (softmax, reorg, region, shortcut, avgpool, `reverse` upsample, gelu, laya) are not touched; their host entry points return `OxideError::Unsupported`-style errors when `half_format()` is Bf16 (use the error type and message style of `unsupported_layers_still_error_cleanly` in `oxide_runtime_f16.rs`).

- [ ] **Step 1: Write failing tests:** `bf16_pool_upsample_route_match_f32_reference` (copy of `f16_pool_upsample_route_match_f32_reference` for bf16 with the 8x bound), `bf16_im2col_matches_reference`, `bf16_mode_rejects_f16_only_inference_ops` (calling an inference-only f16 op, for example `softmax_f16`, on a bf16 backend returns `Err`).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_half -- bf16_` — Expected: FAIL.
- [ ] **Step 3: Implement** the `fmt` argument for the four kernels, the call sites, and the guard on inference-only entry points.
- [ ] **Step 4: Run to verify they pass,** then all of `oxide_half`, `oxide_amp`, `oxide_runtime_f16` — Expected: all pass.
- [ ] **Step 5: Commit** the files above: `feat(oxide): bf16 for im2col, route, maxpool, upsample; guard f16-only ops`.

### Task 6: Backend wiring and the AMP suite for bf16

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/mixed.rs` (`as_half`, `wrap_half`, `weights_half`, `to_half_storage`), `src/lightnet-cuda-oxide/src/lib.rs` (`download_f32`/`upload_f32` widening, constructor reading `amp_format_from_env`)
- Test: `src/lightnet-cuda-oxide/tests/oxide_amp.rs`

**Interfaces:**
- Consumes: `to_half`, `from_half` (Task 2), `HalfFormat` (Task 1).
- Produces: `OxideBackend::new()` reads `LIGHTNET_OXIDE_AMP` through `parse_amp_format`; the constructor sets `gemm_mode` to `Fp16` or `Bf16` to match.

- [ ] **Step 1: Write failing tests** by running the remaining AMP tests for both formats: `amp_weight_copy_is_refreshed_after_an_update`, `amp_gradients_track_f32_gradients_on_tiny_conv_bn_net`, `amp_sgd_steps_track_f32_master_weights`, `amp_adam_steps_track_f32_master_weights`, `amp_overflowing_gradients_become_nonfinite_not_garbage` (for bf16 the overflow test uses a value above 3.4e38 / f32 max is not representable, so assert instead that 1.0e7 stays finite in bf16: name it `amp_bf16_does_not_overflow_where_f16_does`).
- [ ] **Step 2: Run to verify the bf16 cases fail.** `cargo xtask oxide --test oxide_amp` — Expected: bf16 cases fail on the weight cache and widening paths.
- [ ] **Step 3: Implement** the format use in the listed methods; `weights_half` converts through `to_half` with the backend format (cache key stays buffer identity; one format per backend means no extra key).
- [ ] **Step 4: Run to verify it passes.** `cargo xtask oxide --test oxide_amp`, then `cargo xtask oxide --test oxide_half`, then `cargo xtask test` — Expected: all pass.
- [ ] **Step 5: Commit** `mixed.rs`, `lib.rs`, `oxide_amp.rs`: `feat(oxide): bf16 mixed-precision training mode (LIGHTNET_OXIDE_AMP=bf16)`.

### Task 7: xtask -bf16

**Files:**
- Modify: `xtask/src/main.rs` (`precision_env`), `CLAUDE.md` (Commands table row)
- Test: `#[cfg(test)]` in `xtask/src/main.rs`

**Interfaces:**
- Consumes: the `LIGHTNET_OXIDE_AMP=bf16` value from Task 6; the three-element return of `precision_env` from plan_fp16 Task 4.

- [ ] **Step 1: Write failing tests:** `bf16_sets_amp_bf16_without_scaler_flags` (`["-bf16"]` yields env `LIGHTNET_OXIDE_AMP=bf16` and no extra args), `two_precision_flags_are_rejected` (`["-fp16", "-bf16"]` is an `Err`).
- [ ] **Step 2: Run to verify they fail.** `PKGS=xtask cargo xtask test` — Expected: FAIL (`-bf16` is rejected today).
- [ ] **Step 3: Implement** the mapping and remove the "no runtime switch" rejection and help text.
- [ ] **Step 4: Run to verify they pass.** `PKGS=xtask cargo xtask test` — Expected: pass.
- [ ] **Step 5: Commit** `xtask/src/main.rs`, `CLAUDE.md`: `feat(xtask): -bf16 selects bf16 mixed precision`.

### Task 8: Convergence and step-time record

User-gated, same procedure as [plan_fp16.md Task 5](plan_fp16.md#task-5-convergence-and-step-time-record).

- [ ] **Step 1:** With the fp32 baseline and the fp16 run from Phase 1 available, run `cargo xtask train_legogears -bf16 -map_every 100` with the same seed and iteration count.
- [ ] **Step 2:** Compare loss at equal iterations, mAP, step time against both. Pass criterion: loss within 10% of fp32 at the final iteration, mAP within 2 points; step time not worse than fp16 by more than 5% (the same kernels run).
- [ ] **Step 3:** Record the table in `docs/OXIDE_BENCH.md`; update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`; delete weights and logs.
- [ ] **Step 4: Commit** the docs: `docs: record bf16 convergence run`.
