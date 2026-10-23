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
- [Progress](#progress)
- [Open issues (repo-wide)](#open-issues-repo-wide)
- [Kernel experiment log](#kernel-experiment-log)
- [LegoGears conv shapes](#legogears-conv-shapes)
- [Overhead optimisation](#overhead-optimisation)
- [Review backlog](#review-backlog)

## Global Constraints

- Same constraints as [plan_fp16.md](plan_fp16.md#global-constraints): `cargo xtask` only, one build or test at a time, 4 build threads (8 for training), `MemoryHigh=3G`, stage only named files, no unprompted training or long benchmark runs beyond the named bench tests, delete weights and logs, update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv` per version-mapped increment.
- Do not use `mma_m16n8k32_{f16,f32}_e4m3_*` (no `fp8_` in the name): they lower to `kind::f8f6f4` and need sm_120a/121a. Use the `mma_m16n8k32_fp8_*` family (see [fp8_fp4.md](fp8_fp4.md)).
- The quantisation factor is `max_finite / amax` (a multiplier). `fp8::amax_to_scale` returns the reciprocal (the C++ divisor). Passing the wrong one flushes e5m2 values to zero (`lib.rs` comment above `fp8_quant_factor`).
- Existing fp8 behaviour must hold at every task boundary: `cargo xtask oxide --test oxide_gemm_f8` (20 tests) passes unmodified, except tests this plan names as updated.
- Keep `LIGHTNET_OXIDE_GEMM=fp8` semantics (f32 storage path) working throughout; Task 6 adds the AMP combination, it does not remove the existing mode.
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

- Dispatch: `GemmMode::from_env` (`LIGHTNET_OXIDE_GEMM=fp8`) at `src/lightnet-cuda-oxide/src/lib.rs:168`; `gemm` routes to `gemm_f32_via_fp8` (`lib.rs:1564`) then `gemm_f32_via_fp8_fmt` (`lib.rs:1537`).
- Per call: `fp8_quant_factor` (`lib.rs:1480`) allocates a one-word slot, launches `fp8_amax_f32_kernel`, then `slot.to_host_vec(..)` (a blocking device-to-host copy), for each of A and B; `fp8_quantize_buffer` (`lib.rs:1510`) launches `fp8_quantize_kernel` for each; `gemm_fp8_args` (`lib.rs:2460`) launches `sgemm_fp8_mma`, optionally followed by `gemm_splitk_reduce`. The scale `alpha/(fa*fb)` is folded into the mma epilogue.
- No weight cache and no amax history for fp8. The f16 equivalent is `weights_half` (`mixed.rs:148`, keyed on buffer identity).
- Backward: `LIGHTNET_OXIDE_GEMM_BWD=fp8` sets `bwd_fp8 = (E4M3, E5M2)` (`lib.rs:199,1417`); `backward_gemm` (`lib.rs:1455`) is called from `conv.rs:514` (dgrad: A = weights, B = dy) and `conv.rs:646` (wgrad: A = dy, B = column buffer). Forward calls `self.gemm` with A = weights at `conv.rs:417`.
- Kernel `sgemm_fp8_mma` (`kernels/gemm_f8.rs:362`): 64x64 tile, BK 32, 128 threads, single shared-memory buffer, global loads through `fp8_stage16` into registers then `u32` shared stores, two `sync_threads` per K step, scalar fragment reads, a per-byte global loop for the transposed (`trans == 1`) path (`gemm_f8.rs:193-206`), runtime branches on format in `mma_f8_k32`. The fp16 kernel (`kernels/gemm_f16.rs:53`) uses `cp.async` 16-byte copies, two shared stages, `ldmatrix_x4`.
- (Before `v0.2.181`) AMP forced `GemmMode::Fp16`, so `LIGHTNET_OXIDE_GEMM=fp8` was ignored under `LIGHTNET_OXIDE_AMP`; now an Fp8 mode is kept.
- Tests: `tests/oxide_gemm_f8.rs` (20 tests including `fragment_layout_*`, `gemm_fp8_e4m3_all_transposes`, `gemm_mode_fp8_selector_matches_the_dequantised_reference`, `fp8_backward_dgrad_and_wgrad_track_the_f32_backward`, `amax_ignores_non_finite_values`); `tests/oxide_gemm_bench.rs` (`gemm_bench_precisions`, `gemm_bench_lego`, `gemm_bench_low_precision_kernels`, all `#[ignore]`).

## File structure

- Modify `src/lightnet-cuda-oxide/src/lib.rs` (fp8 helpers, weight cache, mode wiring), `src/lightnet-cuda-oxide/src/fp8.rs` (scale state types), `src/lightnet-cuda-oxide/src/kernels/gemm_f8.rs` (kernels), `src/lightnet-cuda-oxide/src/conv.rs` (weight-cache call sites), `src/lightnet-cuda-oxide/src/mixed.rs` (AMP composition).
- Modify tests: `tests/oxide_gemm_f8.rs`, `tests/oxide_amp.rs`, `tests/oxide_gemm_bench.rs`.
- Modify `xtask/src/main.rs`, `CLAUDE.md`, `tasks/rust/fp8_fp4.md`, `docs/OXIDE_BENCH.md`, `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`.

## Tasks

### Task 1: Device-resident scales

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/lib.rs` (`fp8_quant_factor`, `fp8_quantize_buffer`, `gemm_f32_via_fp8_fmt`, `gemm_fp8_args`), `src/lightnet-cuda-oxide/src/kernels/gemm_f8.rs` (`fp8_quantize_kernel`, `sgemm_fp8_mma`, new `fp8_scale_kernel`)
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Interfaces:**
- Produces: `fn fp8_scale_dev(&self, buf: &DeviceBuffer<f32>, len: usize, fmt: Fp8Format) -> Result<DeviceBuffer<f32>, OxideError>`: a two-word device buffer `[factor, 1/factor]` computed by a new `fp8_scale_kernel` from the amax slot (factor 1.0 and 1.0 when amax is 0 or non-finite). `fp8_quantize_kernel` and `sgemm_fp8_mma` read the factors from device pointers instead of by-value scalars; the mma epilogue computes `alpha * inv_a * inv_b` on the device. The host never reads the scale.

- [ ] **Step 1: Write the failing test** `fp8_gemm_does_not_read_back_to_host`: add `OxideBackend::host_readbacks(&self) -> u64` (an atomic counter incremented by every `to_host_vec` the backend wraps, `pub` for tests) and assert it is unchanged across a `gemm_f32_via_fp8` call at 256x256x256. Also keep `gemm_mode_fp8_selector_matches_the_dequantised_reference` as the numeric check.
- [ ] **Step 2: Run to verify it fails.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_gemm_does_not_read_back` — Expected: compile error, then FAIL with a count of 2.
- [ ] **Step 3: Implement** the interfaces above; remove `to_host_vec` from the fp8 path; keep results bit-identical to the previous implementation (same factor formula, computed on the device).
- [ ] **Step 4: Run to verify it passes.** `cargo xtask oxide --test oxide_gemm_f8` — Expected: all 21 pass. Then `gemm_bench_precisions` and record before and after in the commit message.
- [ ] **Step 5: Commit** the four source files and the test: `perf(oxide): fp8 scales stay on the device (no per-GEMM readback)`.

### Task 2: Cached fp8 weights

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/lib.rs` (cache and method), `src/lightnet-cuda-oxide/src/conv.rs` (call sites at lines 417 and 514)
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

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
- Modify: `src/lightnet-cuda-oxide/src/fp8.rs`, `src/lightnet-cuda-oxide/src/kernels/gemm_f8.rs`, `src/lightnet-cuda-oxide/src/lib.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Done as `v0.2.178`:** no `Fp8ScaleState` struct; the history is one zeroed device buffer (16 slots per key, 4096 keys) in a mutex-guarded map keyed by (source device address, length, format code). Call `n` of a key writes slot `n % 16` and zeroes slot `(n + 2) % 16`; the kernel reads the other 14 slots, so the effective window is 14 calls. Call 0 runs the two-pass current-scaling path with its amax landing in the history. Pair quantisation runs the single-operand path twice; cached weights stay on current scaling. Test seam `fp8_quantize_probe`.

**Interfaces:**
- Produces: `pub struct Fp8ScaleState` (device buffer of a 16-entry amax history plus a write cursor, one per quantised tensor role); kernel `fp8_quantize_delayed_kernel` that derives the factor from the history maximum, quantises, and records this call's amax into the history in the same pass (one pass over the data instead of amax then quantise). `pub enum Fp8Scaling { Current, Delayed }` chosen by `LIGHTNET_OXIDE_FP8_SCALING=current|delayed` (default `current`; `Delayed` stays opt-in until Task 9 validates it). Reference recipe: the C++ `fp8_scaling.cpp` (read with `git show v6-dev-fp8:src-fp8/fp8_scaling.cpp`; its history length constant is `kFp8AmaxHistoryLength`).

- [ ] **Step 1: Write the failing tests:** `delayed_scaling_matches_current_scaling_on_stationary_data` (same tensor repeated 20 times: after warm-up, outputs equal current scaling within one e4m3 ulp), `delayed_scaling_saturates_on_a_spike` (an element 100x the history maximum quantises to max finite 448, no NaN or inf), `delayed_scaling_first_call_uses_current_amax` (empty history falls back to the tensor's own amax).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_gemm_f8 -- delayed_scaling` — Expected: compile error.
- [ ] **Step 3: Implement** the state, kernel and env switch as in Interfaces.
- [ ] **Step 4: Run to verify they pass,** then all of `oxide_gemm_f8` — Expected: pass, default mode unchanged.
- [ ] **Step 5: Commit:** `feat(oxide): opt-in delayed fp8 scaling with an amax history`.

### Task 4: Quantise straight from 16-bit storage

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/kernels/gemm_f8.rs`, `src/lightnet-cuda-oxide/src/lib.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

**Done as `v0.2.180`:** kernels `fp8_amax_half_kernel` and `fp8_quantize_half_kernel` (extra `hfmt: u32`); host `fp8_quantize_half` (single operand, current scaling), `gemm_fp8_half_operands(a, b, c, g, fa, fb, hfmt)` (the `hfmt` argument is an addition to the signature below) and the test seams `fp8_quantize_half_probe` and `gemm_fp8_half`. Delayed scaling was not applied to 16-bit operands at `v0.2.180`; `v0.2.187` added `fp8_quantize_delayed_half_kernel` and the delayed dispatch in `fp8_quantize_half` (see the Task 4 follow-up in the status table and `implemented.md` §251). `oxide_gemm_f8` 35 of 35 (two new twins, which also check that the GEMM on 16-bit operands is bit-equal to the f32 fp8 selector on the widened values, all four transposes).

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
- Modify: `src/lightnet-cuda-oxide/src/kernels/gemm_f8.rs`
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`, `src/lightnet-cuda-oxide/tests/oxide_gemm_bench.rs`

One sub-step per commit; each is kept only under the 5% rule in Global Constraints. Measure with `cargo xtask oxide --test oxide_gemm_bench -- --ignored gemm_bench_low_precision_kernels` before and after.

- [x] **Step 1: Sub-step A, staging.** Done as `v0.2.185`, applied *after* B. Standalone on the old kernel it was slower (2.68 ms against 2.46 ms, twice) and was reverted: B was still gathered byte by byte, so the cp.async copy of A could not help. On top of B: A is copied with 16-byte `cp.async` (zero fill) when K-contiguous and 16-byte aligned, B stays on registers (stored after the mma of the current tile), two shared stages, one `sync_threads` per K step, pitch 48. 1.26 ms to 0.991 ms.
- [x] **Step 2: Sub-step B, transposed staging.** Done first, as `v0.2.184`, with a different shape from the text above: no shared-memory transpose. Each thread loads a 4 (k) x 4 (n) byte block as four aligned words, transposes it in registers (`fp8_stage_t4`) and stores four K-contiguous words. Edge and unaligned blocks keep the guarded byte path. Tests added: `gemm_fp8_transposed_large_k_matches_reference` (K = 4096) and `gemm_fp8_nn_is_bitwise_equal_to_nt_on_the_same_data`. 2.46 ms to 1.26 ms.
- [x] **Step 3: Sub-step C, format specialisation.** Done as `v0.2.186`. `sgemm_fp8_mma_e4m3` is a copy of the generic body with `mma_f8_k32_e4m3` (a macro cannot generate it: `cuda_module` does not see macro-expanded kernels), launched when both operands are e4m3. The runtime format `match` around the 8 mma per K step was the largest single cost left: 0.991 ms to 0.613 ms. Test: `fp8_e4m3_specialised_kernel_matches_generic` (bitwise, through the `set_fp8_generic_kernel` seam).
- [x] **Step 4: Sub-step D, `ldmatrix` fragments.** Hypothesis confirmed on layout, rejected on speed. Two adjacent 8-bit values are one 16-bit element in the m16n8k32 fragment layout (a0..a3 = rows g / g+8, K bytes 4t.. / 16+4t.. is the m16n8k16 f16 A layout over 16 b16 columns), so `ldmatrix.x4` loads A and B fragments (the header comment's claim that `ldmatrix` cannot be used was wrong). `oxide_gemm_f8` 38 of 38. But 0.613 ms to 0.697 ms (+14%) at 2048^3 and the LegoGears kernel-only fp8 / fp32 ratio 0.72-0.75 to 0.79: reverted. The patch is not kept.
- [x] **Step 5: Record.** Table below and [fp8_fp4.md](fp8_fp4.md) Phase D. Final 2048^3 kernel-only: 0.613 ms = 28000 GF/s, against 13485 GF/s for the fp16 kernel and 4946 GF/s for fp32 (same run).

### Task 6: Compose with AMP

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/mixed.rs` (`with_amp_format`, `conv_forward_amp`, `conv_backward_input_amp`, `conv_backward_weights_amp`), `src/lightnet-cuda-oxide/src/lib.rs` (constructor, `gemm_mode` selection)
- Test: `src/lightnet-cuda-oxide/tests/oxide_amp.rs`

**Done as `v0.2.181`:** `with_amp_format` and the constructor keep an `Fp8` gemm mode (AMP + `LIGHTNET_OXIDE_GEMM=fp8`). The fp8 conv paths live in `mixed.rs`: `conv_forward_amp_fp8` (weights from `fp8_weights` on the f32 master, activation columns quantised from the 16-bit im2col with `fp8_quantize_half`, f32 accumulate into an f32 buffer that is rounded once into the AMP storage format; 1x1 stride-1 convs quantise the activations once and skip im2col), and the dgrad and wgrad arms of `conv_backward_input_amp` / `conv_backward_weights_amp` (formats `bwd_fp8` or e4m3 x e4m3; `dy` quantised once for both through the new `fp8_dy_half` cache). Under AMP + fp8 the f32 GEMM seam (`gemm`, `gemm_weights_a`) keeps the 16-bit compute path, so non-conv ops are unchanged. Delayed scaling was not applied at `v0.2.181`; `v0.2.187` applies it to the 16-bit activations and `dy`, not to the weights. `oxide_amp` 18 of 18 in f16 and in bf16 (`LIGHTNET_TEST_AMP_FMT=bf16`); the fp8 conv error against the f64 reference is 3e-2 to 7e-2 of the largest element (bound 0.1), against about 1e-3 for the 16-bit GEMM. No speed measurement.

**Interfaces:**
- Consumes: `fp8_weights` (Task 2), `gemm_fp8_half_operands` (Task 4), `HalfFormat` (plan_bf16 Task 1).
- Produces: under AMP, when `LIGHTNET_OXIDE_GEMM=fp8`, the conv GEMMs run fp8 on f16/bf16 activations; `with_amp_format` no longer overwrites an `Fp8` gemm mode with `Fp16`/`Bf16`. Non-conv ops keep their AMP paths.

- [ ] **Step 1: Write the failing tests:** `amp_conv_with_fp8_gemm_tracks_f32_reference` (forward, dgrad, wgrad against the f64 reference with the error bound used by `gemm_mode_fp8_selector_matches_the_dequantised_reference`, because e4m3 keeps 3 mantissa bits), `amp_fp8_weight_copy_is_refreshed_after_an_update`, `amp_without_fp8_env_still_uses_fp16_gemm` (no behaviour change).
- [ ] **Step 2: Run to verify they fail.** `cargo xtask oxide --test oxide_amp -- fp8` — Expected: FAIL (AMP overrides the GEMM mode).
- [ ] **Step 3: Implement** the interfaces; fp8 applies only to conv GEMMs.
- [ ] **Step 4: Run to verify they pass,** then `oxide_amp`, `oxide_half`, `oxide_gemm_f8` — Expected: pass.
- [ ] **Step 5: Commit:** `feat(oxide): fp8 conv GEMMs under mixed-precision storage`.

### Task 7: Backward pass review

**Files:**
- Modify: `src/lightnet-cuda-oxide/src/conv.rs` (lines 514 and 646 call sites), `src/lightnet-cuda-oxide/src/lib.rs` (`backward_gemm`)
- Test: `src/lightnet-cuda-oxide/tests/oxide_gemm_f8.rs`

- [ ] **Step 1: Write the failing test** `fp8_backward_formats_follow_the_operand_role`: dgrad uses (weights e4m3, dy e5m2) and wgrad uses (dy e5m2, columns e4m3), asserting through the quantise-launch counter and a result compared with the f32 backward at the e4m3/e5m2 error level. Today wgrad passes `(fa, fb) = (E4M3, E5M2)` positionally while its A operand is `dy`, so the formats follow position, not role; the test pins the correct role mapping.
- [ ] **Step 2: Run to verify it fails.** `cargo xtask oxide --test oxide_gemm_f8 -- fp8_backward_formats` — Expected: FAIL for wgrad.
- [ ] **Step 3: Implement** a `BackwardRole { Dgrad, Wgrad }` argument to `backward_gemm` selecting the format pair; use the weight cache for dgrad's A operand.
- [ ] **Step 4: Run to verify it passes,** including `fp8_backward_dgrad_and_wgrad_track_the_f32_backward` — Expected: pass.
- [ ] **Step 5: Commit:** `fix(oxide): fp8 backward formats follow operand roles`.

### Task 8: xtask -fp8

**Done as `v0.2.182`, with one deviation:** plain `-fp8` is unchanged (f32 storage plus fp8 GEMMs, the recorded benchmark config), so the step 1 test below was not written as stated. The combination is selected with two flags, `-fp8 -fp16` (adds `LIGHTNET_OXIDE_AMP=1` and the `-dyn_scale -loss_scale 1024` rule) or `-fp8 -bf16` (`LIGHTNET_OXIDE_AMP=bf16`, no scaler flags), in either order; `-fp16 -bf16` and repeated flags stay errors. `PKGS=xtask cargo xtask test` 11 of 11 (five new). `CLAUDE.md` Commands row and `xtask help` updated.

**Files:**
- Modify: `xtask/src/main.rs`, `CLAUDE.md`
- Test: `#[cfg(test)]` in `xtask/src/main.rs`

- [ ] **Step 1: Write the failing test** `fp8_sets_amp_and_fp8_gemm` (`["-fp8"]` yields env `LIGHTNET_OXIDE_AMP=1` and `LIGHTNET_OXIDE_GEMM=fp8`, no scaler flags beyond the fp16 task's rule: fp8 uses `-dyn_scale -loss_scale 1024` because activations are f16).
- [ ] **Step 2: Run to verify it fails.** `PKGS=xtask cargo xtask test` — Expected: FAIL (today only `LIGHTNET_OXIDE_GEMM=fp8` is set).
- [ ] **Step 3: Implement** the mapping and update help text and the `CLAUDE.md` Commands row.
- [ ] **Step 4: Run to verify it passes.** `PKGS=xtask cargo xtask test` — Expected: pass.
- [ ] **Step 5: Commit:** `feat(xtask): -fp8 trains with f16 storage and fp8 convolutions`.

### Task 9: Accuracy, speed gate, record

User-gated, same procedure as [plan_fp16.md Task 5](plan_fp16.md#task-5-convergence-and-step-time-record).

- [ ] **Step 1:** With the fp32 and fp16 runs available, run `cargo xtask train_legogears -fp8 -map_every 100` with the same seed and iteration count; also run once with `LIGHTNET_OXIDE_FP8_SCALING=delayed`.
- [ ] **Step 2: Gate.** fp8 is declared a speed feature only if all hold: the batch-16 step time is lower than the fp16 AMP step time; loss within 10% of fp32 at the final iteration; mAP within 2 points of fp16. If the speed condition fails after Tasks 1 to 6, document fp8 as numerics-only in [fp8_fp4.md](fp8_fp4.md) and stop; do not keep tuning without a profile (use `nsys` or the bench tests to name the remaining bottleneck first).
- [ ] **Step 3:** Decide the default scaling mode from the two runs; record the table and verdict in `docs/OXIDE_BENCH.md`; update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`; delete weights and logs.
- [ ] **Step 4: Commit** the docs: `docs: record fp8 accuracy and speed gate`.

## Progress

Status as of 2026-10-22. After Task 2 the overhead work in [Overhead optimisation](#overhead-optimisation) (`v0.2.167` to `v0.2.169`) cut `prec-lego` fp8 from 21.97 ms to 2.23 ms; Tasks 4, 6 and 8 landed as `v0.2.180` to `v0.2.182` (bf16 was already done, `v0.2.170`). Task 5 landed as `v0.2.184` to `v0.2.186` on 2026-10-23: the 2048^3 kernel went from 2.46 ms to 0.613 ms.

| Task | Status | Commit or tag |
|---|---|---|
| 1. Device-resident scales | done | `87fef93` (`perf(oxide): fp8 scales stay on the device`) |
| 2. Cached fp8 weights | done | `v0.2.166` (`420dfd9`), docs `05953ba` |
| 3. Delayed scaling | done, opt-in (`LIGHTNET_OXIDE_FP8_SCALING=delayed`); speed unmeasured | `v0.2.178` (`b74279e`) |
| 4. Quantise from 16-bit storage | done; current scaling at `v0.2.180`, delayed scaling for 16-bit sources added at `v0.2.187` (`fp8_quantize_delayed_half_kernel`; `oxide_gemm_f8` 46 of 46, `oxide_amp` 20 of 20 in f16 and bf16; speed unmeasured) | `v0.2.180` (`5c5dc81`), `v0.2.187` (`8056b16`) |
| 5. Kernel tuning | done: sub-steps B, A (on B) and C kept, D rejected; 2048^3 kernel 2.46 ms to 0.613 ms (7000 to 28000 GF/s) | `v0.2.184` (`87ae104`), `v0.2.185` (`01ae58f`), `v0.2.186` (`b690fb9`) |
| 6. Compose with AMP | done, f16 and bf16; speed unmeasured | `v0.2.181` (`eb8919a`) |
| 7. Backward pass review | done: `BackwardRole` places `dy` per GEMM; wgrad on the f32-column path gave `dy` e4m3 and the column e5m2 before; `oxide_gemm_f8` 52 of 52 | this commit |
| 8. xtask `-fp8` | done as `-fp8 -fp16` / `-fp8 -bf16`; plain `-fp8` unchanged | `v0.2.182` (`7fdb817`) |
| 9. Accuracy and speed gate | user-gated (training run) | |

Task 2 details: the forward and dgrad weight operands are quantised once per weight tensor. The cache is keyed on buffer identity and format, and is reachable only through the weight seams (`gemm_weights_a`, `backward_gemm_weights_a`). `oxide_gemm_f8` passes 24 of 24 (three new cache tests). The 21-conv bench does not exercise the cache, so its speed gain is still unmeasured.

Open issues:
- **Task 2 speed gain unmeasured.** Needs a conv-level bench that runs the weight-cache path (`prec-lego` goes through the uncached tensor path).
- **Bench test `gemm_bench_low_precision_lego_kernels` uncommitted** (`tests/oxide_gemm_bench.rs`). Added this session.
- **`bench/` folder untracked.** Written by Haiku agents: `README.md`, `run_bench.sh` (`bash -n` only, never run end to end), and three header-only CSVs. Open decisions: symlink of `third_party/cuda-oxide` into scratch worktrees (script refuses unless `BENCH_SHARE_OXIDE=1`), and deleting the leftover `bench/.work/`. Neither decided.
- **LegoGears kernel time.** Before Task 5 the fp8 kernel only matched fp32 on the 21 shapes (ratio 1.00); after `v0.2.186` it is 0.72-0.75 of fp32. End-to-end `prec-lego` is still overhead-bound, about 1.7 ms against 1.0 ms (see [LegoGears conv shapes](#legogears-conv-shapes)).
- **Local-memory spill removed, no speed gain.** Root cause was `fp8_stage16` taking `&mut` register arrays and a runtime-indexed `dst[j / 4]` (see [Kernel experiment log](#kernel-experiment-log)). Both fixes were reverted because kernel time did not improve.
- **Profile covered one launch only.** The ncu run used the 1024^3 launch (grid 256), not 2048^3.
- **Delayed history keyed on a recycled device pointer (open, not fixed).** `fp8_quantize_delayed` keys the amax history on (source device pointer, length, format). The buffer pool recycles device buffers, so two unrelated tensors of the same length can share a key and the second inherits the first one's history (a stale scale, covered by saturation but not by accuracy). Found in the codex review after `v0.2.187`; the fixes `v0.2.188` to `v0.2.190` (stale-tail amax, non-finite propagation, tiny-amax factor clamp; `implemented.md` §252) do not touch it. Options when it is taken up: key on a call-site id instead of the pointer, or invalidate on pool reuse.
- **Codex review ran against an edited file.** It read `gemm_f8.rs` while the padding experiment was applied, so some line references may be off.

## Open issues (repo-wide)

Issues outside the fp8 path that affect how fp8 work is verified or committed. Recorded here so they are not lost; none is fixed by the fp8 work.

- **Default `cargo xtask clippy` fails.** Three `approx_constant` errors in `src/lightnet-core/src/vc_attention.rs`, which this work does not touch. Touched crates (`xtask`, `lightnet-cuda-oxide`) are clean on their own (source: `implemented.md`, section 237).
- **Default `cargo xtask test` status unverified.** `implemented.md` (section 237) reports 77 core runtime test-harness errors; `AGENTS_fp8.md` says commit `37ec4b5` made it compile again. Not re-run in this session. Re-check before citing host tests as a gate.
- **Flaky bf16 parity test.** `oxide_half` `gemm_bf16_parity_all_transposes_alignments_alpha_beta` failed once on `OxideBackend::new()` and passed in 10 later runs; cause unknown (source: `AGENTS_fp8.md`, Current state). Blocks Phase 2 verification.
- **Uncommitted user edit in `src/lightnet-core/src/runtime/mod.rs`** (28 lines, not ours). It was present at the start of the session and is not staged. Commits must stage only named hunks.
- **Uncommitted bench test** `gemm_bench_low_precision_lego_kernels` in `src/lightnet-cuda-oxide/tests/oxide_gemm_bench.rs`. Ours; needs a commit once accepted.
- **`bench/` folder untracked.** Haiku-written; the script has not been run end to end (see the Progress section for the two undecided items).
- **Phase 1 Task 5 verdict open.** fp32 and fp16 convergence runs exist (seed 42, 3000 iterations, final mAP 99.55% against 99.63%) but the loss gate (11.7% apart) and the record in `docs/OXIDE_BENCH.md` are not done. Training runs are user-gated.
- **Commit attribution.** Session reminders named different models (Sonnet 5.5 and Haiku 5.5) for the `Co-Authored-By` line. Commits so far use Sonnet 5.5. Needs a decision from the user.
- **Process: background waits.** Several long `cargo xtask` runs exceeded the 120 s foreground timeout and were moved to background; results were read from output files. No data loss, but future runs should use `run_in_background` directly.

## Kernel experiment log

Each experiment is measured kernel-only at 2048^3 with `gemm_bench_low_precision_kernels`; a change is kept only if it is at least 5% faster than the current HEAD baseline (2.53 ms, re-measured 2026-10-22).

| Experiment | Result | Decision |
|---|---|---|
| `cp.async` 4-byte staging of A into two shared stages, B still via registers | 3.09 ms (slower) | reverted |
| `fp8_stage16` returns `[u32; 4]` by value (removes the 32-byte `areg`/`breg` local depot) | 2.77 ms and 3.10 ms (slower) | reverted |
| Same as above plus scalar word gather (removes the 16-byte `dst[j / 4]` local array; 0 local ops in PTX) | 3.10 ms (slower) | reverted |
| Shared pitch padding, `LD_A` = `LD_B` = 44 bytes (bank conflicts 16 to 4 extra wavefronts per store) | 2.47 ms (2.4%, within the fp32 control's 1.7% drift) | reverted |

Takeaway: removing local memory and reducing store bank conflicts did not reduce kernel time at 2048^3.

Task 5 session (2026-10-23, Sonnet 5.5). Baseline at HEAD `0afcbe1`: 2.456, 2.459, 2.4705 ms (2048^3 kernel-only). Each row is the 2048^3 kernel-only time (two runs when both are given), the LegoGears kernel-only fp8 total as a ratio to the fp32 control in the same run (the laptop clock drifts: absolute 21-conv totals move by 25% between runs while the ratio holds to about 0.05), and `prec-sq` fp8 end to end (fp32 4.43 ms in every run).

| Experiment | 2048^3 kernel | lego kernel fp8 / fp32 | prec-sq fp8 | Decision |
|---|---|---|---|---|
| Baseline | 2.456 / 2.459 / 2.4705 ms (7000 GF/s) | 1.00, 1.00 | 4.36 / 4.61 ms | |
| A standalone: `cp.async` 16 B for A, 2 shared stages, 1 sync per K step, B per-byte gather as before | 2.682 / 2.682 ms (+9%) | 0.91, 1.08 | 4.77 / 4.75 ms | reverted (no gain: B gather is the bottleneck) |
| B: 4x4 byte blocks loaded as words, register transpose (`v0.2.184`) | 1.258 / 1.266 ms (-49%) | 0.84, 0.91 | 2.63 / 2.75 ms | kept |
| A on top of B (`v0.2.185`) | 0.991 / 0.991 ms (-21%) | 0.82, 0.83 | 2.41 / 2.39 ms | kept |
| C probe: generic kernel with the format hard-wired to (0, 0) | 0.612 ms | 0.80 | 1.88 ms | confirms the branch is the cost |
| C: `sgemm_fp8_mma_e4m3` (`v0.2.186`) | 0.613 / 0.616 ms (-38%) | 0.72, 0.75 | 1.87 / 1.87 ms | kept |
| D: `ldmatrix.x4` for A and B on top of C | 0.697 ms (+14%) | 0.79 | 2.07 ms | reverted |

Reading: the byte-wise `[k][n]` gather was the bottleneck at 2048^3 (it was NN: B is `[k][n]`), then the per-mma format branch. The earlier local-memory and bank-conflict experiments attacked costs that were hidden behind these two. No ncu run was made; ncu and nsys exist on this machine (`/usr/local/cuda-13.4/bin`) but were not used, so the explanation is from the before/after times and the C probe, not from stall counters. `prec-lego` (end to end) stayed in the 1.6-1.9 ms range against fp32 0.9-1.3 ms in every run: it is overhead-bound and the kernel changes do not show in it. The training-step effect is not measured (training is user-gated).

## LegoGears conv shapes

Measured 2026-10-22 on the 21 LegoGears_v2 conv GEMMs (`LEGO` in `tests/oxide_gemm_bench.rs`). Two benches: `gemm_bench_low_precision_lego_kernels` (kernel only, operands pre-quantised, new) and `gemm_bench_precisions` `prec-lego` (end to end, includes quantise and allocations).

| Bench | fp32 | fp8 | fp8 / fp32 |
|---|---|---|---|
| Kernel only, 21 shapes total | 2.04 ms | 2.17 ms | 1.06 |
| End to end, 21 shapes total | 2.52 ms | 21.97 ms | 8.7 |

Findings:
- Kernel time is about 10% of end-to-end fp8 time on these shapes. About 90% is quantise passes, amax and scale kernels, per-call allocations and launches.
- The kernel alone does not beat fp32 on these shapes. Small M (30 to 128) leaves the 64x64 tile partly empty; N = 35 wastes about 45%; K = 27, 288 and 576 pad to the 32-element BK.

Decision: optimise the overhead first (pool the FP8 temporaries, remove redundant quantise passes, fuse amax into quantise), then retune the kernel for small M and N. Measure each change with `gemm_bench_precisions` and `gemm_bench_low_precision_lego_kernels`, not only 2048^3.

## Overhead optimisation

Order set 2026-10-22. Each step is measured with `LIGHTNET_BENCH_MODES=fp32,fp8 cargo xtask oxide --test oxide_gemm_bench -- --ignored gemm_bench_precisions` (`prec-lego` TOTAL) and kept only if the fp8 total drops.

| Step | What | Status | prec-lego fp8 total |
|---|---|---|---|
| 0 | Baseline | measured | 21.97 ms (fp32 2.52 ms) |
| 1 | Profile with `nsys` (see the profile below) | done | |
| 2 | Pool the fp8 temporaries: amax slot and scale pair from the f32 pool, quantised bytes from a new u8 free list (`PooledBuf8`, shares the byte cap). The amax slot is now an f32 buffer read as `u32` bits by the atomic | done, `oxide_gemm_f8` 24 of 24 | 3.84 ms (fp32 1.12 ms, re-measured the same run) |
| 3 | Drop `fp8_scale_kernel`: `fp8_quantize_kernel` reads the amax slot, derives the factor and writes the `[factor, 1/factor]` scale pair itself (per operand: amax + quantise, 2 launches instead of 3). `fp8_scale_dev` and `fp8_quantize_buffer` merged into `fp8_quantize_operand`; the host-side `quantize_f32_to_fp8` keeps an explicit-scale kernel. The amax-slot zero-fill is still a memset | done, `oxide_gemm_f8` 24 of 24 | 2.95 ms (fp32 1.02 ms, same run) |
| 3b | Fuse A and B: `fp8_amax_pair_kernel` (one launch, per-block shared-memory reduction, one atomic per block, A and B slots 128 bytes apart) and `fp8_quantize_pair_kernel` (one launch). Per GEMM 7 host calls become 4. First attempt (one atomic per element, adjacent slots) was slower: lego 5.89 ms, amax kernel avg 140 us against 3.4 us. The codex `gpt-6-sol` review also found that the kernels split operands at buffer length instead of GEMM extent (out-of-bounds write when a buffer is longer than its extent); fixed and pinned by `fp8_pair_quantise_ignores_elements_beyond_the_gemm_extent` | done, `oxide_gemm_f8` 25 of 25 | 2.23 ms (fp32 0.98 ms, same run); square total 4.16 ms against fp32 4.21 ms |
| 4 | Byte im2col: `im2col_fp8` quantises while it gathers, using the image amax (conv forward and wgrad). v0.2.171 | done, `oxide_gemm_f8` 26 of 26; training step 57.0 to 50.5 ms (fp32 53.4 ms), 3000 iterations 153.7 s against fp32 160.9 s | not measured by the GEMM bench (it feeds f32 operands) |
| 4b | Pre-zeroed amax slot ring (one memset per 256 operands) and block-reduced single-operand amax. v0.2.172 | done, `oxide_gemm_f8` 26 of 26; `bench-train` fp8 49.0 ms against fp32 50.9 ms; 3000 iterations 143.0 s against 147.1 s | |
| 4c | wgrad byte column with the e5m2 gradient recipe (column e4m3, `dy` e5m2); `-fp8` sets `LIGHTNET_OXIDE_GEMM_BWD=fp8`. v0.2.177 | done, `oxide_gemm_f8` 26 of 26; 3000 iterations 142.1 s and 142.9 s against fp32 146.4 s, mAP 99.80% | |
| 5 | Delayed scaling ([Task 3](#task-3-delayed-scaling)): one pass over activations instead of two. v0.2.178 | done, opt-in; `oxide_gemm_f8` 31 of 31 (five new). Not yet measured on training; Task 9 decides the default | |
| 5b | Quantise `dy` once per conv backward: dgrad (B) and wgrad (A) share one e5m2 byte copy and scale pair (`fp8_dy`, one-entry cache keyed on the dy buffer identity, length and format; wgrad quantises the whole `dy` instead of the per-chunk prefix). One keyed quantise per backward, so delayed scaling also advances its history once. v0.2.179 | done, `oxide_gemm_f8` 33 of 33 (two new), `oxide_amp` 15 of 15, `oxide_half` 17 of 17; quantise launches per backward 2 to 1, results bitwise equal in the single-chunk case. Not timed on training | |
| 6 | CUDA Graphs for the per-step launch sequence, only if launch overhead is still visible after 3; first check that cuda-oxide exposes graph capture. Pooling (step 2) already makes addresses stable | open | |

Step 1 profile (2026-10-22, `nsys profile` on the `gemm_bench_precisions` binary, `LIGHTNET_BENCH_MODES=fp8`, 1 round, after steps 2 and 3; square and LegoGears shapes together, 2402 GEMMs). nsys adds tracing overhead to the host API times.

| What | Count | Host API time | GPU time |
|---|---|---|---|
| `cuLaunchKernel` | 14022 (5.8 per GEMM) | 238 ms (17 us each) | |
| `cuMemsetD8Async` (amax slot zero) | 4906 | 71 ms (14.5 us each) | 2 ms |
| `cuMemAlloc` | 94 | 31 ms | |
| `sgemm_fp8_mma` | 2402 | | 126 ms |
| `fp8_quantize_kernel` | 4804 | | 17.5 ms |
| `fp8_amax_f32_kernel` | 4804 | | 16.4 ms |
| `gemm_splitk_reduce` | 2012 | | 6.2 ms |

Findings: pooling worked (94 allocations for 2402 GEMMs). The remaining cost is host-side: launches plus memsets cost about 310 ms of API time against about 166 ms of GPU time, so the run is launch-bound. The amax and quantise kernels are about 20% of GPU time. Per GEMM the call sequence is memset x2, amax x2, quantise x2, mma (+ split-K reduce): fusing the two operands into one amax launch and one quantise launch (7 calls to 4) cuts host calls more than byte im2col does, and CUDA Graphs would remove the rest. Byte im2col (step 4) targets GPU time and traffic, which is the smaller share here.

Step 2 removed one `cudaMalloc`/`cudaFree` per operand per GEMM (amax slot, scale pair, byte output). The remaining gap to fp32 (3.4x) is launches and the quantise passes.

After step 4b (2026-10-23) the training runs are recorded in `docs/OXIDE_BENCH.md` section 10: fp8 142.8 s and 143.0 s against fp32 147.2 s and 147.0 s, mAP 99.55%, F1 0.9902.

Training-step profile (2026-10-22, `nsys` on `bench-train`, batch 16, 21 steps, before step 4): the step is GPU-bound. GPU kernel time per step: fp8 about 51 ms (`sgemm_fp8_mma` 17.8, `im2col` 8.1, `col2im` 3.8, amax and quantise kernels 11.6), fp32 about 45 ms (`sgemm_tiled` 23.8, `im2col` 8.1, `col2im` 3.6). So fp8 lost to fp32 on the amax and quantise passes, and CUDA Graphs (step 6) would not have helped. After step 4 the fp8 step is about 50 ms. Remaining levers: delayed scaling (drops `fp8_amax_*`, about 129 ms of 1075 in the profile), quantising `dy` without a second pass, and the fp8 kernel itself (only 1.3x faster than fp32 `sgemm_tiled` here).

## Review backlog

Ranked by the codex review of the fp8 GEMM (gpt-6-sol, high effort, read-only), against the ncu profile in the log above. Confidence is the reviewer's. The order below is the review's; the LegoGears measurements above move item 4 ahead of item 1, because overhead dominates end to end.

1. **Pack B as K-contiguous `[N, K]` while quantising.** Largest kernel gain expected, medium confidence. *Partly done differently as `v0.2.184`: the GEMM kernel now transposes 4x4 byte blocks in registers while staging (2.46 ms to 1.26 ms), so the packing pass is not needed for the kernel; packing at quantise time remains an option for the end-to-end path.* The NN path gathers B with `ldb`-spaced byte loads before each K tile (`kernels/gemm_f8.rs` `fp8_stage16`, `trans == 1`). Measure the packing pass cost in the full call, since the kernel-only bench assumes operands are pre-quantised. Bench: `gemm_bench_precisions` (fp8 column is end-to-end).
2. **Two shared K tiles plus `cp.async`**, after item 1. Medium confidence. *Done as `v0.2.185` (A only; B stays on registers), 1.26 ms to 0.991 ms.* Verify register count and stalls per variant.
3. **Shared-store layout, then BK and output tile size.** Modest, low to medium confidence. Locate the conflicting store instruction first. 122 registers per thread limits larger tiles.
4. **Pool FP8 temporaries; CUDA Graphs for repeated calls.** High confidence for end-to-end gain (`lib.rs` allocates amax slot, scale pair and byte output per uncached operand; weights are cached, f32 outputs and split-K scratch are pooled). Graphs cut launch overhead across about 21 convs per step, not the time inside one GEMM.
5. **Quantise inside the GEMM load path.** Low expected gain, medium confidence. Repeats conversion across reused tiles and keeps the amax/scale dependency. Try item 1 first.

Instruction boundary (from the same review): FP8 `mma.sync.m16n8k32` is available on sm_89; FP8 `wgmma` requires sm_90a; native `ldmatrix.b8` is not available on Ada. An `ldmatrix.b16` plus repacking variant is possible but has the least direct evidence. *Tried in Task 5 sub-step D: no repacking is needed (two 8-bit values are one b16 element in this fragment layout), it is correct, and it is 14% slower than the byte-wide shared loads.*
