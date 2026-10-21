# fp8 Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fp8 (e4m3 forward, optional e5m2 gradients) a faster training mode than f16 AMP on the sm_89 GPU for LegoGears_v2, or prove it cannot be and record that.

**Architecture:** Remove the per-GEMM overhead first (host readbacks, per-call weight quantisation), then tune the `mma.m16n8k32` kernel, then compose fp8 convolution GEMMs with the f16/bf16 AMP activation storage. Every task is measured against the numbers recorded below; a change that does not help is reverted.

**Tech Stack:** Rust, cuda-oxide kernels (`mma_m16n8k32_fp8_*`, `cp.async`, `ldmatrix`), sm_89.

**Spec:** [AGENTS_fp8.md](AGENTS_fp8.md), section "Phase 3: fp8", and [fp8_fp4.md](fp8_fp4.md) (earlier fp8 design, Phase D numbers). Do Phases 1 and 2 first ([plan_fp16.md](plan_fp16.md), [plan_bf16.md](plan_bf16.md)); Tasks 4 and 6 depend on plan_bf16 Tasks 2 and 6.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Baseline numbers](#baseline-numbers)
- [Verified starting state](#verified-starting-state)
- [File structure](#file-structure)
- [Tasks](#tasks)
  - [Task 1: Device-resident scales](#task-1-device-resident-scales)
  - [Task 2: Cached fp8 weights](#task-2-cached-fp8-weights)
  - [Task 3: Delayed scaling](#task-3-delayed-scaling)
  - [Task 4: Quantise straight from 16-bit storage](#task-4-quantise-straight-from-16-bit-storage)
  - [Task 5: Kernel tuning](#task-5-kernel-tuning)
  - [Task 6: Compose with AMP](#task-6-compose-with-amp)
  - [Task 7: Backward pass review](#task-7-backward-pass-review)
  - [Task 8: xtask -fp8](#task-8-xtask--fp8)
  - [Task 9: Accuracy, speed gate, record](#task-9-accuracy-speed-gate-record)

## Global Constraints

- Same constraints as [plan_fp16.md](plan_fp16.md#global-constraints): `cargo xtask` only, one build or test at a time, 4 build threads (8 for training), `MemoryHigh=3G`, stage only named files, no unprompted training or long benchmark runs beyond the named bench tests, delete weights and logs, update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv` per version-mapped increment.
- Do not use `mma_m16n8k32_{f16,f32}_e4m3_*` (no `fp8_` in the name): they lower to `kind::f8f6f4` and need sm_120a/121a. Use the `mma_m16n8k32_fp8_*` family (see [fp8_fp4.md](fp8_fp4.md)).
- The quantisation factor is `max_finite / amax` (a multiplier). `fp8::amax_to_scale` returns the reciprocal (the C++ divisor). Passing the wrong one flushes e5m2 values to zero (`lib.rs` comment above `fp8_quant_factor`).
- Existing fp8 behaviour must hold at every task boundary: `cargo xtask oxide --test oxide_gemm_f8` (20 tests) passes unmodified, except tests this plan names as updated.
- Keep `DARKNET_OXIDE_GEMM=fp8` semantics (f32 storage path) working throughout; Task 6 adds the AMP combination, it does not remove the existing mode.
- A kernel change is kept only if the kernel-only 2048^3 time (`gemm_bench_low_precision_kernels`) improves by at least 5% and all fp8 tests pass; otherwise revert it and record the result in the task commit message.
- No C++ FP8 build exists, so accuracy is judged against the Rust fp16 run, never claimed as C++ parity.

## Review Focus

- All-zero, constant and single-spike tensors: amax of 0 or non-finite must give factor 1.0 and a finite result; a spike larger than the scale saturates to the format's max finite value, never NaN or inf.
- Stale cached weights: after an optimizer step, the next forward and dgrad must use the new weights (a missed refresh trains on old weights silently).
- Mixed shapes in one network: the same weight buffer used as A in forward and in dgrad (transposed) must give consistent results from one cached copy.
- A NaN or inf in an activation: the amax kernel ignores non-finite values (existing test `amax_ignores_non_finite_values`); the quantiser must not turn them into finite garbage that hides the overflow from the loss scaler.
- Split-K shapes (small M and N, large K): scale handling is identical with and without split-K.

## Baseline numbers

From `tasks/rust/implemented.md` (section 3 and the Phase D table in [fp8_fp4.md](fp8_fp4.md)); re-measure with `cargo xtask oxide --test oxide_gemm_bench -- --ignored gemm_bench_precisions` before Task 1 and record the new values in the Task 1 commit message.

| Case | f32 | tf32 | fp16 | fp8 |
|---|---|---|---|---|
| 2048^3 end to end | 5.50 ms | 1.80 ms | 1.274 ms | 5.90 ms |
| 2048^3 fp8 kernel only | | | | 2.45 ms (7020 GF/s); fp16 kernel 13485 GF/s |
| LegoGears_v2, 21 convs | 5.95 ms | 4.82 ms | 3.87 ms | 24.2 ms |

## Verified starting state

- Dispatch: `GemmMode::from_env` (`DARKNET_OXIDE_GEMM=fp8`) at `src/darknet-cuda-oxide/src/lib.rs:168`; `gemm` routes to `gemm_f32_via_fp8` (`lib.rs:1564`) then `gemm_f32_via_fp8_fmt` (`lib.rs:1537`).
- Per call: `fp8_quant_factor` (`lib.rs:1480`) allocates a one-word slot, launches `fp8_amax_f32_kernel`, then `slot.to_host_vec(..)` (a blocking device-to-host copy), for each of A and B; `fp8_quantize_buffer` (`lib.rs:1510`) launches `fp8_quantize_kernel` for each; `gemm_fp8_args` (`lib.rs:2460`) launches `sgemm_fp8_mma`, optionally followed by `gemm_splitk_reduce`. The scale `alpha/(fa*fb)` is folded into the mma epilogue.
- No weight cache and no amax history for fp8. The f16 equivalent is `weights_half` (`mixed.rs:148`, keyed on buffer identity).
- Backward: `DARKNET_OXIDE_GEMM_BWD=fp8` sets `bwd_fp8 = (E4M3, E5M2)` (`lib.rs:199,1417`); `backward_gemm` (`lib.rs:1455`) is called from `conv.rs:514` (dgrad: A = weights, B = dy) and `conv.rs:646` (wgrad: A = dy, B = column buffer). Forward calls `self.gemm` with A = weights at `conv.rs:417`.
- Kernel `sgemm_fp8_mma` (`kernels/gemm_f8.rs:362`): 64x64 tile, BK 32, 128 threads, single shared-memory buffer, global loads through `fp8_stage16` into registers then `u32` shared stores, two `sync_threads` per K step, scalar fragment reads, a per-byte global loop for the transposed (`trans == 1`) path (`gemm_f8.rs:193-206`), runtime branches on format in `mma_f8_k32`. The fp16 kernel (`kernels/gemm_f16.rs:53`) uses `cp.async` 16-byte copies, two shared stages, `ldmatrix_x4`.
- AMP forces `GemmMode::Fp16` (`lib.rs:1393`, `mixed.rs:68-72`), so `DARKNET_OXIDE_GEMM=fp8` is ignored under `DARKNET_OXIDE_AMP`.
- Tests: `tests/oxide_gemm_f8.rs` (20 tests including `fragment_layout_*`, `gemm_fp8_e4m3_all_transposes`, `gemm_mode_fp8_selector_matches_the_dequantised_reference`, `fp8_backward_dgrad_and_wgrad_track_the_f32_backward`, `amax_ignores_non_finite_values`); `tests/oxide_gemm_bench.rs` (`gemm_bench_precisions`, `gemm_bench_lego`, `gemm_bench_low_precision_kernels`, all `#[ignore]`).

## File structure

- Modify `src/darknet-cuda-oxide/src/lib.rs` (fp8 helpers, weight cache, mode wiring), `src/darknet-cuda-oxide/src/fp8.rs` (scale state types), `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs` (kernels), `src/darknet-cuda-oxide/src/conv.rs` (weight-cache call sites), `src/darknet-cuda-oxide/src/mixed.rs` (AMP composition).
- Modify tests: `tests/oxide_gemm_f8.rs`, `tests/oxide_amp.rs`, `tests/oxide_gemm_bench.rs`.
- Modify `xtask/src/main.rs`, `CLAUDE.md`, `tasks/rust/fp8_fp4.md`, `docs/OXIDE_BENCH.md`, `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`.

## Tasks

### Task 1: Device-resident scales

**Files:**
- Modify: `src/darknet-cuda-oxide/src/lib.rs` (`fp8_quant_factor`, `fp8_quantize_buffer`, `gemm_f32_via_fp8_fmt`, `gemm_fp8_args`), `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs` (`fp8_quantize_kernel`, `sgemm_fp8_mma`, new `fp8_scale_kernel`)
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Produces: `fn fp8_scale_dev(&self, buf: &DeviceBuffer<f32>, len: usize, fmt: Fp8Format) -> Result<DeviceBuffer<f32>, OxideError>`: a two-word device buffer `[factor, 1/factor]` computed by a new `fp8_scale_kernel` from the amax slot (factor 1.0 and 1.0 when amax is 0 or non-finite). `fp8_quantize_kernel` and `sgemm_fp8_mma` read the factors from device pointers instead of by-value scalars; the mma epilogue computes `alpha * inv_a * inv_b` on the device. The host never reads the scale.

- [ ] **Step 1: Write the failing test** `fp8_gemm_does_not_read_back_to_host`: add `OxideBackend::host_readbacks(&self) -> u64` (an atomic counter incremented by every `to_host_vec` the backend wraps, `pub` for tests) and assert it is unchanged across a `gemm_f32_via_fp8` call at 256x256x256. Also keep `gemm_mode_fp8_selector_matches_the_dequantised_reference` as the numeric check.
- [ ] **Step 2: Run to verify it fails.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_gemm_does_not_read_back` — Expected: compile error, then FAIL with a count of 2.
- [ ] **Step 3: Implement** the interfaces above; remove `to_host_vec` from the fp8 path; keep results bit-identical to the previous implementation (same factor formula, computed on the device).
- [ ] **Step 4: Run to verify it passes.** `cargo xtask oxide --test oxide_gemm_f8` — Expected: all 21 pass. Then `gemm_bench_precisions` and record before and after in the commit message.
- [ ] **Step 5: Commit** the four source files and the test: `perf(oxide): fp8 scales stay on the device (no per-GEMM readback)`.

### Task 2: Cached fp8 weights

**Files:**
- Modify: `src/darknet-cuda-oxide/src/lib.rs` (cache and method), `src/darknet-cuda-oxide/src/conv.rs` (call sites at lines 417 and 514)
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Consumes: `fp8_scale_dev` from Task 1.
- Produces: `struct Fp8Weights { bytes: Arc<DeviceBuffer<u8>>, scale: Arc<DeviceBuffer<f32>> }`; `fn fp8_weights(&self, w: &OxideTensorF32, fmt: Fp8Format) -> Result<Fp8Weights, OxideError>` (cache keyed on the weight buffer identity and `fmt`, same mechanism and cap as `weights_half`/`W16_CACHE_MAX`); `fn gemm_fp8_weights_a(&self, w: &OxideTensorF32, b: &DeviceBuffer<f32>, c: &mut DeviceBuffer<f32>, g: &GemmArgs, fa: Fp8Format, fb: Fp8Format) -> Result<(), OxideError>`; `OxideBackend::fp8_quantize_launches(&self) -> u64` (atomic counter, `pub` for tests). The conv forward (A = weights) and dgrad (A = weights) call `gemm_fp8_weights_a` when the mode is fp8.

- [ ] **Step 1: Write the failing tests:** `fp8_weight_cache_hits_on_second_forward` (two forwards with the same weight tensor: the launch counter grows by the activation quantise only on the second), `fp8_weight_cache_refreshes_after_update` (replace the weight tensor with an updated one, as the optimizer does: the counter grows by the weight quantise and the output changes), `fp8_cached_weights_serve_forward_and_transposed_dgrad` (one cached copy gives results equal to the uncached path for both).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_weight_cache` — Expected: compile error.
- [ ] **Step 3: Implement** as in Interfaces. The quantisation is elementwise and layout preserving, so one cached byte buffer serves both the plain and transposed uses through `GemmArgs`.
- [ ] **Step 4: Run to verify they pass.** `cargo xtask oxide --test oxide_gemm_f8`, then `gemm_bench_lego` and record the 21-conv time — Expected: all pass; 21-conv time lower than the Task 1 value.
- [ ] **Step 5: Commit:** `perf(oxide): cache quantised fp8 weights per optimizer step`.

### Task 3: Delayed scaling

**Files:**
- Modify: `src/darknet-cuda-oxide/src/fp8.rs`, `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs`, `src/darknet-cuda-oxide/src/lib.rs`
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Produces: `pub struct Fp8ScaleState` (device buffer of a 16-entry amax history plus a write cursor, one per quantised tensor role); kernel `fp8_quantize_delayed_kernel` that derives the factor from the history maximum, quantises, and records this call's amax into the history in the same pass (one pass over the data instead of amax then quantise). `pub enum Fp8Scaling { Current, Delayed }` chosen by `DARKNET_OXIDE_FP8_SCALING=current|delayed` (default `current`; `Delayed` stays opt-in until Task 9 validates it). Reference recipe: the C++ `fp8_scaling.cpp` (read with `git show v6-dev-fp8:src-fp8/fp8_scaling.cpp`; its history length constant is `kFp8AmaxHistoryLength`).

- [ ] **Step 1: Write the failing tests:** `delayed_scaling_matches_current_scaling_on_stationary_data` (same tensor repeated 20 times: after warm-up, outputs equal current scaling within one e4m3 ulp), `delayed_scaling_saturates_on_a_spike` (an element 100x the history maximum quantises to max finite 448, no NaN or inf), `delayed_scaling_first_call_uses_current_amax` (empty history falls back to the tensor's own amax).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_gemm_f8 -- delayed_scaling` — Expected: compile error.
- [ ] **Step 3: Implement** the state, kernel and env switch as in Interfaces.
- [ ] **Step 4: Run to verify they pass,** then all of `oxide_gemm_f8` — Expected: pass, default mode unchanged.
- [ ] **Step 5: Commit:** `feat(oxide): opt-in delayed fp8 scaling with an amax history`.

### Task 4: Quantise straight from 16-bit storage

**Files:**
- Modify: `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs`, `src/darknet-cuda-oxide/src/lib.rs`
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Consumes: `half_bits_to_f32(bits: u16, fmt: u32)` from plan_bf16 Task 2.
- Produces: kernels `fp8_amax_half_kernel` and `fp8_quantize_half_kernel` reading `&[u16]` with a `fmt: u32` argument; `fn gemm_fp8_half_operands(&self, a: &OxideTensorF16, b: &OxideTensorF16, c, g: &GemmArgs, fa: Fp8Format, fb: Fp8Format) -> Result<(), OxideError>`.

- [ ] **Step 1: Write the failing tests:** `fp8_from_f16_matches_fp8_from_widened_f32` (quantising f16 storage gives byte-identical output to widening to f32 first), same for bf16.
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_from_` — Expected: compile error.
- [ ] **Step 3: Implement** the kernels and the host method; no f32 intermediate buffer.
- [ ] **Step 4: Run to verify they pass.** `cargo xtask oxide --test oxide_gemm_f8` — Expected: pass.
- [ ] **Step 5: Commit:** `feat(oxide): fp8 quantisation reads f16/bf16 storage directly`.

### Task 5: Kernel tuning

**Files:**
- Modify: `src/darknet-cuda-oxide/src/kernels/gemm_f8.rs`
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`, `src/darknet-cuda-oxide/tests/oxide_gemm_bench.rs`

One sub-step per commit; each is kept only under the 5% rule in Global Constraints. Measure with `cargo xtask oxide --test oxide_gemm_bench -- --ignored gemm_bench_low_precision_kernels` before and after.

- [ ] **Step 1: Sub-step A, staging.** Replace `fp8_stage16` register staging with `cp.async` 16-byte copies into two shared stages and one `sync_threads` per K step, the pattern of `hgemm_f16_mma` (`gemm_f16.rs:161-268`). Tests: all `oxide_gemm_f8` tests unchanged and passing.
- [ ] **Step 2: Sub-step B, transposed staging.** Replace the per-byte global loop for `trans == 1` (`gemm_f8.rs:193-206`) with a vectorised load plus a shared-memory transpose. Test: `gemm_fp8_e4m3_all_transposes` plus a new `gemm_fp8_transposed_large_k_matches_reference` at K = 4096.
- [ ] **Step 3: Sub-step C, format specialisation.** Add `sgemm_fp8_mma_e4m3` (both operands e4m3, no runtime format branch) and select it in `gemm_fp8_args` when `fa == fb == E4M3`. Test: `fp8_e4m3_specialised_kernel_matches_generic` bitwise.
- [ ] **Step 4: Sub-step D, `ldmatrix` fragments (hypothesis).** Two adjacent e4m3 bytes form one 16-bit element, so `ldmatrix.x4` can load the A and B fragments as CUTLASS does for 8-bit types; the header comment of `gemm_f8.rs` says otherwise, so verify with the `fragment_layout_*` tests (`oxide_gemm_f8.rs:158-177`) first. If the layout test passes, keep it; if not, revert and record why in the commit message.
- [ ] **Step 5: Record** the kernel-only 2048^3 time after each sub-step and the final value against the fp16 kernel (13485 GF/s); update [fp8_fp4.md](fp8_fp4.md) Phase D with the new table.

### Task 6: Compose with AMP

**Files:**
- Modify: `src/darknet-cuda-oxide/src/mixed.rs` (`with_amp_format`, `conv_forward_amp`, `conv_backward_input_amp`, `conv_backward_weights_amp`), `src/darknet-cuda-oxide/src/lib.rs` (constructor, `gemm_mode` selection)
- Test: `src/darknet-cuda-oxide/tests/oxide_amp.rs`

**Interfaces:**
- Consumes: `fp8_weights` (Task 2), `gemm_fp8_half_operands` (Task 4), `HalfFormat` (plan_bf16 Task 1).
- Produces: under AMP, when `DARKNET_OXIDE_GEMM=fp8`, the conv GEMMs run fp8 on f16/bf16 activations; `with_amp_format` no longer overwrites an `Fp8` gemm mode with `Fp16`/`Bf16`. Non-conv ops keep their AMP paths.

- [ ] **Step 1: Write the failing tests:** `amp_conv_with_fp8_gemm_tracks_f32_reference` (forward, dgrad, wgrad against the f64 reference with the error bound used by `gemm_mode_fp8_selector_matches_the_dequantised_reference`, because e4m3 keeps 3 mantissa bits), `amp_fp8_weight_copy_is_refreshed_after_an_update`, `amp_without_fp8_env_still_uses_fp16_gemm` (no behaviour change).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_amp -- fp8` — Expected: FAIL (AMP overrides the GEMM mode).
- [ ] **Step 3: Implement** the interfaces; fp8 applies only to conv GEMMs.
- [ ] **Step 4: Run to verify they pass,** then `oxide_amp`, `oxide_half`, `oxide_gemm_f8` — Expected: pass.
- [ ] **Step 5: Commit:** `feat(oxide): fp8 conv GEMMs under mixed-precision storage`.

### Task 7: Backward pass review

**Files:**
- Modify: `src/darknet-cuda-oxide/src/conv.rs` (lines 514 and 646 call sites), `src/darknet-cuda-oxide/src/lib.rs` (`backward_gemm`)
- Test: `src/darknet-cuda-oxide/tests/oxide_gemm_f8.rs`

- [ ] **Step 1: Write the failing test** `fp8_backward_formats_follow_the_operand_role`: dgrad uses (weights e4m3, dy e5m2) and wgrad uses (dy e5m2, columns e4m3), asserting through the quantise-launch counter and a result compared with the f32 backward at the e4m3/e5m2 error level. Today wgrad passes `(fa, fb) = (E4M3, E5M2)` positionally while its A operand is `dy`, so the formats follow position, not role; the test pins the correct role mapping.
- [ ] **Step 2: Run to verify it fails.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_backward_formats` — Expected: FAIL for wgrad.
- [ ] **Step 3: Implement** a `BackwardRole { Dgrad, Wgrad }` argument to `backward_gemm` selecting the format pair; use the weight cache for dgrad's A operand.
- [ ] **Step 4: Run to verify it passes,** including `fp8_backward_dgrad_and_wgrad_track_the_f32_backward` — Expected: pass.
- [ ] **Step 5: Commit:** `fix(oxide): fp8 backward formats follow operand roles`.

### Task 8: xtask -fp8

**Files:**
- Modify: `xtask/src/main.rs`, `CLAUDE.md`
- Test: `#[cfg(test)]` in `xtask/src/main.rs`

- [ ] **Step 1: Write the failing test** `fp8_sets_amp_and_fp8_gemm` (`["-fp8"]` yields env `DARKNET_OXIDE_AMP=1` and `DARKNET_OXIDE_GEMM=fp8`, no scaler flags beyond the fp16 task's rule: fp8 uses `-dyn_scale -loss_scale 1024` because activations are f16).
- [ ] **Step 2: Run to verify it fails.** `PKGS=xtask cargo xtask test` — Expected: FAIL (today only `DARKNET_OXIDE_GEMM=fp8` is set).
- [ ] **Step 3: Implement** the mapping and update help text and the `CLAUDE.md` Commands row.
- [ ] **Step 4: Run to verify it passes.** `PKGS=xtask cargo xtask test` — Expected: pass.
- [ ] **Step 5: Commit:** `feat(xtask): -fp8 trains with f16 storage and fp8 convolutions`.

### Task 9: Accuracy, speed gate, record

User-gated, same procedure as [plan_fp16.md Task 5](plan_fp16.md#task-5-convergence-and-step-time-record).

- [ ] **Step 1:** With the fp32 and fp16 runs available, run `cargo xtask train_legogears -fp8 -map_every 100` with the same seed and iteration count; also run once with `DARKNET_OXIDE_FP8_SCALING=delayed`.
- [ ] **Step 2: Gate.** fp8 is declared a speed feature only if all hold: the batch-16 step time is lower than the fp16 AMP step time; loss within 10% of fp32 at the final iteration; mAP within 2 points of fp16. If the speed condition fails after Tasks 1 to 6, document fp8 as numerics-only in [fp8_fp4.md](fp8_fp4.md) and stop; do not keep tuning without a profile (use `nsys` or the bench tests to name the remaining bottleneck first).
- [ ] **Step 3:** Decide the default scaling mode from the two runs; record the table and verdict in `docs/OXIDE_BENCH.md`; update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`; delete weights and logs.
- [ ] **Step 4: Commit** the docs: `docs: record fp8 accuracy and speed gate`.
