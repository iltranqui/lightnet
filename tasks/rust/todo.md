# Todo (branch rust)

Plan: ~/.claude/plans/https-github-com-nvidia-cuda-rust-port-t-imperative-pumpkin.md
Goal: cuda-oxide backend `src/darknet-cuda-oxide` implementing `F32TensorOpsBackend` with Rust kernels ported from `src-lib/*.cu`.

> **All implemented work has moved to [`implemented.md`](implemented.md).** This
> file tracks only open / partial / deferred items. Phase tables live in
> [`layers.md`](layers.md); the FP8/FP4 plan is [`fp8_fp4.md`](fp8_fp4.md).
> Per-`LayerKind` CPU/GPU/f16/axis status is in [`layer-support.md`](layer-support.md).

## Contents

- [Build/test (one command at a time, repo root)](#buildtest-one-command-at-a-time-repo-root)
- [Hardware-blocked (cannot be compiled/tested on this sm_89 GPU)](#hardware-blocked-cannot-be-compiledtested-on-this-sm_89-gpu)
- [Backend / kernel follow-ups](#backend--kernel-follow-ups)
- [Gradients](#gradients)
- [P8/P9 training follow-ups](#p8p9-training-follow-ups)
- [Data pipeline](#data-pipeline)
- [FP8 / FP4 (`fp8_fp4.md`)](#fp8--fp4-fp8_fp4md)
- [Layer port (Phases 1b–4 — see `layers.md` phase tables)](#layer-port-phases-1b4--see-layersmd-phase-tables)
- [Active and unstarted plans](#active-and-unstarted-plans)
- [ONNX export](#onnx-export)
- [Known evidence gaps (not user questions)](#known-evidence-gaps-not-user-questions)
- [Not implemented by design / no scanned cfg needs it](#not-implemented-by-design--no-scanned-cfg-needs-it)
- [Known small diffs vs C++](#known-small-diffs-vs-c)
- [Version rewrite v0.0.001 – v0.0.246](#version-rewrite-v00001--v00246)

## Build/test (one command at a time, repo root)

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G ./tools/test_oxide.sh                      # ~15 min, all suites
systemd-run --user --scope -p MemoryHigh=3G --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
  target/debug/cargo-oxide oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 4
```

Run `cargo test -p darknet-core --test cfg_corpus -j 2` alone before believing a failure. The full isolated corpus run exceeded a 40-minute timeout during v69 with no final result; do not record that as a pass or a test failure.

## Hardware-blocked (cannot be compiled/tested on this sm_89 GPU)

- [ ] wgmma (sm_90) / tcgen05 (sm_100 Blackwell) GEMM kernels and per-arch
  dispatch (the `GemmArgs` + `GemmMode` seam is in place).
- [ ] Native NVFP4 `mma` (needs sm_120a/121a); the sm_89 path is the `__dp4a`
  emulation.
- [ ] Multi-GPU (2 GPUs planned); keep GEMM/optimizer behind interfaces.

## Backend / kernel follow-ups

- [ ] Deformable-family wave 1 (shared bilinear sampling), parts 1–2 of 3 done. **Part 1 (host contract):** `host_ops::bilinear_gather::{BilinearGatherGeometry, forward, backward}` (`src/darknet-cuda`) plus `F32TensorOpsBackend::deform_bilinear_gather_f32` / `_backward_f32` host-reference defaults returning `[B,S,C,OH,OW]` columns and `(d_input, d_sample_y, d_sample_x)`. Semantics copied from `darknet_core::deform_conv::bilinear_sample`: far-outside zero-gradient early-out, floor-cell one-sided convention at integer coordinates, NaN propagation. Evidence: `cargo test -p darknet-cuda -j 2 --lib` 18 passed (6 new). **Part 2 (native device):** `src/darknet-cuda-oxide/src/kernels/deform_sample.rs` (`deform_gather`, `deform_gather_backward`) and Oxide overrides with checked operands; `d_input` is a float-atomic scatter (same documented trade-off as `crop_bilinear_backward`: exact result set, non-deterministic summation order), while `d_sample_y`/`d_sample_x` are per-sample owned writes. The forward/backward contract is documented in `darknet_cuda::host_ops::bilinear_gather`; the device side re-derives the floor cell, so this is device-vs-host-reference agreement, not C++ parity. Evidence: `MEM=3G ./tools/test_oxide.sh --test oxide_deform_gather` **5 passed** on the RTX 4060 (forward parity, backward parity, far-outside zeros in both directions, device finite differences, operand rejection); rustfmt clean for the two new files; `cargo test -p darknet-cuda --lib` still 18 passed; regression suites green: `oxide_layers_extra` 24 passed and `oxide_runtime_chain` 25 passed, the latter including `deform_conv_f32_host_fallback_matches_cpu_chain_and_updates` — so the pre-wiring deform_conv host-fallback baseline is verified before that arm is touched. Note: oxide `lib.rs`/existing kernel files are **not** rustfmt-clean repo-wide, so only the new files were formatted (no unrelated churn). **Part 3 (wiring, accumulation stage only):** `kernels/deform_sample.rs::deform_conv_accumulate` is native — one thread per `(b, filter, oy, ox)` sums `weight[f,c,k] * bilinear_sample(base + offset) * mask` over the group's channels and kernel points and applies the source `sanitize` clamp (±10000, non-finite → 0); `deform_sample_at` is the shared `#[device]` sampler. `OxideBackend::deform_conv_accumulate` validates batches/offsets/masks/weights and launches it. Evidence: `MEM=3G ./tools/test_oxide.sh --test oxide_deform_conv_accumulate` **5 passed** (zero-offset vs a plain 3x3 conv incl. a hand-computed output, grouped+displaced+sigmoid-masked, strided+dilated, samples pushed past the edge forcing zeros, operand rejection) with an independently written reference, plus rustfmt. **Not wired:** the predictor columns, per-batch conditional offset normalization and lhtan/sigmoid envelopes are still CPU-side, so `layer-support.md` keeps `[deform_conv]` at `host`/`cpu`. Remaining assembly order: native predictor columns → `linear_f32` for offset/mask predictors → one elementwise normalization+lhtan+sigmoid kernel → this accumulate kernel → existing conv epilogue. **Envelope stage landed:** `deform_offset_envelope` (sanitize → optional per-batch normalize → sanitize → centered lhtan; caller supplies `mean`/`inv_std`/`normalize`) and `deform_mask_sigmoid` (stable sigmoid of the sanitized logit), with 5 device tests in `oxide_deform_envelope` against an independent reference (clamp saturates, NaN/inf, per-batch stats switch, operand rejection, ±80 clamp symmetry). **Predictor stage landed:** `deform_offset_predict` = `conv2d_f32` (`groups = 1`, source weight order `[filters][channels][ky][kx]`, zero-fill im2col) + `bias_add_f32`, matching `predictor_columns` → dense matmul → bias; used for both the `2*size*size` offset filters and the `size*size` mask filters. It **rejects `dilation != 1`** explicitly rather than returning a mis-shaped tensor, because the source extent includes the dilated kernel extent and the conv `Geom` does not. `oxide_deform_envelope` now has 12 device tests (adds two predictor parity cases incl. stride-without-padding, the dilation rejection, and **two composed-forward cases** through `deform_conv_forward_native`). Composition order is `sanitize(input) → offset predictor → per-batch stats → envelope → (mask predictor → sigmoid) → accumulate`, returning the raw output before the layer's own bias/BN/activation epilogue — matching what `deform_conv_forward` returns. **Trait plumbing started:** the composition currently lives as an `OxideBackend`
inherent method, and `GpuExecutableNetwork<B: F32TensorOpsBackend>` is generic,
so the runtime arm **cannot call it** — the stages must be trait methods with
host-reference defaults first. Those defaults now exist:
`darknet_cuda::host_ops::deform_conv` (`sanitize`, `centered_lhtan`,
`stable_sigmoid`, `OffsetStats`/`offset_stats`, `offset_envelope`,
`mask_sigmoid`, `DeformGeometry`/`accumulate`) with 6 unit tests, and
`host_ops::bilinear_gather::sample_at` was extracted so the gather and the
accumulation share one sampler (verified by re-running `oxide_deform_gather`,
5 passed). `cargo test -p darknet-cuda -j 2 --lib` 24 passed.

**Next:** move the five primitives onto the trait (host default + Oxide override)
and compose in `runtime.rs`; only then can the arm change. **Trait migration done:**
`F32TensorOpsBackend` now declares `deform_sanitize_f32`,
`deform_offset_stats`, `deform_offset_envelope`, `deform_mask_sigmoid` and
`deform_conv_accumulate_f32` with the new host-reference defaults, and
`OxideBackend` forwards each to its native implementation (shape-checked on the
accumulation). `cargo check -p darknet-core -j 4` and the full oxide build behind
`oxide_deform_gather` (5 passed) confirm both sides compile and still agree.
**Blocking finding:** the arm change still cannot happen. The CPU backward reads
`DeformConvRuntimeCache`, which is produced by the host forward;
`GpuForwardState.cpu_forward` is assembled from the GPU loop's own host
artifacts (`runtime.rs:13174`), so nothing else can supply it. A device forward
would silently starve the fallback backward, and recomputing the host forward
for the cache would erase the gain. **Backward stage 1 (host reference) landed:** `host_ops::deform_conv::accumulate_backward` returns `DeformGradients { d_input, d_offsets, d_masks, d_weights }` and is verified by finite differences over **all four operand groups** (masks included), with the source's `sanitize` slopes on the output sum and on the sanitized input. **Backward stage 2 (host reference) landed:** `centered_lhtan_slope` and `offset_envelope_backward` (adjoint w.r.t. `raw`, stats treated as constants) verified by finite differences in **both** normalize modes. `cargo test -p darknet-cuda -j 2 --lib` 28 passed. Fixture note: values on a 0.5 grid with `max_offset = 3` land exactly on `scaled == 0`, where the source's slope is a one-sided choice (1.0) and a central difference averages 1.0 with 0.001 — the fixture is now offset by 0.17 so no sample sits on a piecewise boundary. **Backward stage 3 (host reference) landed:** `offset_normalization_backward` implements the source's batch-coupled Jacobian `(d_i - Σd/n)/std - (x_i - mean)·dot(d, x-mean)/(n·std³)` in f64, with pass-through for unnormalized batches and a hard error on a non-positive stddev. FD-verified **through the recomputed batch statistics** (the test's forward calls `offset_stats` itself, so the coupling term is genuinely exercised), spanning one normalizing and one raw batch. `OffsetStats` gained a `stddev` field. `cargo test -p darknet-cuda -j 2 --lib` 30 passed. **Backward stage 4 (host reference) landed:** `predictor_columns` (the plain im2col the predictors consume) and `predictor_backward` → `PredictorGradients { d_input, d_weights, d_biases, d_columns }`, covering the dense predictor adjoint **and** the column-gather adjoint back to the layer input (with the input `sanitize` slope). FD-verified over every weight and bias and a sampled subset of inputs, recomputing the columns inside the objective. `cargo test -p darknet-cuda -j 2 --lib` 32 passed. **All four backward stages now exist as FD-exact host references.** **Unit baseline re-verified after v0.2.008:** `darknet-core --lib` 950 passed, `darknet-cli` 29 passed; `cfg_corpus` still not re-run (four known residuals). This closes the gate that v0.2.007 skipped and that the vit-guard regression proved necessary.

- [x] **v0.2.017 (`8a3ef3b0`): DCNv4 7x7 runtime optimizer parity.** Actual native runtime 7x7/two-step SGD+Adam CPU parity with BN on/off; targeted `oxide_runtime_chain` test passed. See implemented.md §54.
- [x] **v0.2.016 (`a96d6d95`): native DCNv4 size bound extended to 7x7 (49 points).** Fixed mask capacity + native composition CPU-parity matrix; runtime gate aligned. `oxide_dcnv4_chain` 3 and host `dcnv4` 28 passed. >49 and unsupported activations remain host. See implemented.md §53.
- [x] **v0.2.015 (`1fc80f27`): native DCNv4 channel-normalization activations/adjoints.** See implemented.md §52.
- [x] **v0.2.014 (`ceef2037`): native DCNv4 rolling-stat BN.** Affine/gamma adjoints use native primitives; post-backward native stats reduction preserves 0.9 EMA/unbiased variance, including unseeded layers. Ordinary bias/beta remain ignored; zero/negative gamma work. Host DCNv4 **26**, runtime:: **282**, host-cuda **37**; device BN **2** (GPU FD, fixed rolling stats, N=1/unseeded), runtime DCNv4 **3** (two SGD/Adam steps, BN on/off, masks/center, upstream/all active blocks), composition **3**, aggregate **6**; CLI oxide-feature check/format/diff gates passed. Audit `implemented.md` §51. **Next: native NormalizeChannels* activation/adjoints**. Large/unsupported/mixed fallback, compact/f16, device parameter updates/residency and GPU Muon remain open. Prior full runtime-chain timeout remains unverified; no C++ training-BN/performance/full-suite claim.

- [x] **v0.2.013 (`3f3c2226`): bounded DCNv4 native f32 runtime math.** No BN, <=25 active points, supported elementwise activation; mixed/excluded variants retain host caches. Both directions share the global gate. Offsets recompute on device; parameter gradients/SGD/Adam stay host-resident, so not complete native layer/residency. Host-cuda **37**, host `dcnv4` **22**, mandatory `runtime::` **278**; focused device runtime **3** (two-step SGD/Adam, all blocks/upstream conv motion, BN/mixed fallback), composition **3**, aggregate **6**, CLI oxide-feature check and targeted formatting/diff checks passed. Full runtime-chain **timed out at 600 s in ViT** after 25 preceding tests reported ok; no full-suite green claim. Audit `implemented.md` §50. **BN is now native in v0.2.014 above**; normalized/large variants, compact/f16, native parameter updates/residency and GPU Muon remain open.

- [x] **v0.2.012 (`c51b25f6`): DCNv4 native raw forward/backward composition.** Coarse predictor + border lattice interpolation + aggregation and input/offset parameter adjoints. Corrects old shared K/row-stride bug (active points and padding to eight); prior stage-only parity did not prove CPU-layer parity. Chain **3** (18 real-CPU cases, full parameter/input gradients, device FD and error/lifetime guards), aggregate **6**, host-cuda **36**, host `dcnv4` **17**, Oxide fallback regression **1** passed. Audit `implemented.md` §49. **Bounded runtime integration/two-step parity is now v0.2.013 above; BN and other unsupported variants still use host fallback.** no f16/benchmark/training/C++ golden.

- [x] **v0.2.011 (`8fe32a09`): DCNv4 native aggregation adjoint (offsets given).** Fixed pending batch-zero-only launch guard; native row-owned offset/mask gradients and atomic input scatter. `oxide_dcnv4_aggregate` **6 passed** (host parity, every-operand actual-device FD, signed seeds, B=2/groups, all mask/center modes, sizes 1/3/5, stride/dilation/edges, malformed/overflow rejection); host-cuda **35**, host `dcnv4` **17**, Oxide runtime `dcnv4` fallback **1**. `implemented.md` §48 holds the audit. **Composition/layout corrected in v0.2.012 above; trait/runtime wiring and update parity remain open.** Native primitives are not a complete native layer; fallback and optimizer behavior unchanged, no f16/benchmark/C++ golden.

**`[dcnv4]` device aggregation landed (v0.2.010, `7874c225`):** `kernels/dcnv4.rs::dcnv4_aggregate` + `OxideBackend::dcnv4_aggregate_native`, parity-green against the host reference for 2 group configs x 4 softmax/remove_center combos (`oxide_dcnv4_aggregate` 2 passed). Fixed 25-point mask buffer; >5x5 kernels rejected by the launcher. Forward-only at that release; the standalone adjoint is now v0.2.011 above. Full composition and the runtime gate remain open.

**`[dcnv4]` host reference landed (v0.2.009, `e3cfb27d`):** `host_ops::dcnv4` provides `aggregate` / `aggregate_backward` (given offsets), FD-green over all four `softmax`/`remove_center` combinations; `cargo test -p darknet-cuda --lib` 35 passed. The two earlier failures were fixture defects, resolved by isolated replication (ratio 0.9999) rather than by rewriting the code. Next: device kernel + parity, then composed backward vs `dcnv4_backward` gradients, then the runtime gate. Original audit:

**`[dcnv4]` audit: its forward cache has only two fields (`offsets`, `raw_output` — `src/darknet-core/src/dcnv4.rs:11`), so unlike deform_conv it needs no cache producer and the template applies cheaply. The genuinely new work is the aggregation kernel itself (`d_stride` coarse offsets + interpolation, grouped partitions, optional softmax masks, centre removal) and its adjoint; the offset predictor is a `d_stride` projection, so predictor-backward reuse is only partial. Slice order and the "do not assume deform_conv evidence transfers" warning are recorded in `gpu_layers_plan.md`. No `dcnv4` code was changed.

**`[deform_conv]` is wired end to end (device forward + backward), v0.2.007 (`15fc3f8c`).** The runtime now selects the device path when the network has no host-only deform variant (`dilation == 1`, no batch norm) and no other fallback layer; `DeformConv` left `requires_cpu_backward_fallback`, and a single shared predicate (`has_host_only_deform`) gates both directions so they cannot disagree. The backward arm composes the five trait primitives, performs exactly one host hop (batch-coupled normalization Jacobian + stage-1 slope), applies the activation derivative and `bias_backward` for the epilogue, and accumulates into the existing `LayerTrainBuffers::DeformConv` fields so the optimizer paths are unchanged. Evidence: `test_host.sh deform_conv` **23 passed** (on FakeBackend, i.e. through the host-default trait bodies), `oxide_runtime_chain` **26 passed** including the new `deform_conv_device_path_matches_cpu_chain_and_updates` (forward outputs, one SGD step, main/offset/mask parameter blocks all matching CPU, plus "parameters actually moved" checks) and the retained BN host-fallback test. Two defects this surfaced, both real: the backward's fallback gate initially ignored `has_host_only_deform` (device backward + host forward → missing cache), and the envelope/mask backward calls passed the *layer output* shape instead of the offset/mask tensor shapes (caught by the host-default bodies, invisible to the Oxide overrides which ignore the shape argument). `layer-support.md` now records the gate.

**Backward trait surface complete:** `F32TensorOpsBackend` now also declares `deform_conv_accumulate_backward_f32`, `deform_envelope_pre_backward_f32`, `deform_mask_sigmoid_backward_f32`, `deform_predictor_backward_f32` and `deform_scale_input_gradient_f32`, each with a host-reference default, with `OxideBackend` forwarding to the device primitives. This is what makes the runtime composition possible at all: `GpuExecutableNetwork<B: F32TensorOpsBackend>` is generic and cannot call inherent Oxide methods. The normalization Jacobian needs no trait method — the runtime performs that one host hop with the free function in `host_ops`. Evidence: `cargo check -p darknet-core -j 4` clean, `oxide_deform_conv_accumulate` **8 passed**, `oxide_deform_conv_backward_chain` **3 passed** (the latter also confirms the simplified `deform_predictor_backward` signature, which now rebuilds column values from the sanitized input instead of reading a columns buffer). Remaining: the `runtime.rs` backward arm, removing `DeformConv` from `requires_cpu_backward_fallback`, and the two-step weight-parity test.

**Runtime step 1 landed (device path present but inert):** `DeformDeviceCache<B::Tensor>` holds the device intermediates; `forward_with_state_impl` now has a global `has_host_only_deform` predicate (`dilation != 1 || batch_normalize`) folded into `host_copies` so the forward and backward can never disagree about which path ran; and the GPU forward arm composes the device pipeline (`deform_sanitize_f32` → `conv2d_f32`+`bias_add_f32` predictors → `deform_offset_stats` → `deform_offset_envelope` → `deform_mask_sigmoid` → `deform_conv_accumulate_f32` → bias + activation) **only when `!host_copies`**. `DeformConv` is deliberately still in `requires_cpu_backward_fallback`, so the device branch is currently unreachable — that is what keeps the change safe. Also simplified `deform_predictor_backward` to read the sanitized input and rebuild the column value in-kernel, removing the need for a materialised columns buffer on device. Gates: `cargo check -p darknet-core` clean, `MEM=3G ./tools/test_host.sh deform_conv` **23 passed**, `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain` **25 passed** (including `deform_conv_f32_host_fallback_matches_cpu_chain_and_updates`, so the fallback path is unchanged). Remaining: the backward arm + removing the fallback entry + the two-step weight-parity test. `OxideBackend::deform_conv_backward_native` chains the device primitives (accumulation adjoint → envelope slopes → **one host hop** for the batch-coupled normalization Jacobian + stage-1 slope → predictor/column adjoint → mask sigmoid adjoint → predictor → summed input gradient scaled once by the input slope). Test `native_backward_composition_matches_the_cpu_gradients` compares `d_main_weights`, `d_offset_weights`, `d_offset_biases`, `d_mask_weights`, `d_mask_biases` against the **CPU train buffers** and `d_input` against the FD-exact host composition (the CPU keeps no input gradient), masked and unmasked — `oxide_deform_conv_backward_chain` **3 passed**. The only host-resident step is the batch-serial reduction; everything per-element runs on device.

**Device backward stage 2 landed:** `deform_envelope_pre_backward` and `deform_mask_sigmoid_backward` kernels plus the shared `deform_sanitize_slope` / `deform_centered_lhtan_slope` / `deform_mask_sigmoid_slope` device helpers, exposed through `OxideBackend::deform_envelope_pre_backward` / `deform_mask_sigmoid_backward`. Host `mask_sigmoid_slope` / `mask_sigmoid_backward` were added and the composed host backward now calls them instead of repeating the inline slope. Evidence: `oxide_deform_envelope` **14 passed** (envelope pre-adjoint in both normalize modes; mask sigmoid adjoint at 0, ±1.5, ±20, ±79, ±80, 1e6 and NaN), `cargo test -p darknet-cuda --lib` 33 passed. Implementation note: `#[device]` helpers must live at file scope — a helper nested inside one `#[cuda_module]` is unreachable from a sibling module via `super::*`, so `deform_sanitize`/`deform_stable_sigmoid` and the slope helpers were hoisted out.

**Device backward stage 1 landed:** `kernels/deform_sample.rs` module `backward` provides `deform_conv_accumulate_backward` (one thread per `(b, filter, pos)`; recomputes the output slope, then atomically scatters `d_input`, `d_offsets`, `d_masks`, `d_weights`), exposed as `OxideBackend::deform_conv_accumulate_backward` with validated operand lengths. The input `sanitize` slope is deliberately **not** applied in the kernel (the caller does it once on the summed input gradient). Evidence: `oxide_deform_conv_accumulate` now has **7 passed**, the two new cases comparing all four gradients against the FD-verified host reference, masked and unmasked, at atomic-order tolerance `2e-5`. Implementation notes: the shared sampler moved to file scope so the forward kernels and the `backward` module can both use it (a nested `#[device]` fn is not reachable via `super::*`), and the scatter targets reuse the `deform_atomic_add_f32` helper.

**Backward composition (host) landed:** `host_ops::deform_conv::forward` / `backward` chain all four stages (sanitize → predictor → stats → envelope → mask sigmoid → accumulate, then back through accumulate → lhtan/sanitize slopes → normalization Jacobian → stage1 slope → predictor/column adjoint), with the input `sanitize` slope applied once. FD-verified over `d_input`, `d_main_weights`, `d_offset_weights`, `d_offset_biases`, `d_mask_weights`, `d_mask_biases` in one composed test. `cargo test -p darknet-cuda -j 2 --lib` 33 passed. Two findings while getting there: (1) chaining the *full* `offset_envelope_backward` (which already contains `inv_std`) into `offset_normalization_backward` **double-counts** the normalization; a dedicated `offset_envelope_pre_backward` now supplies the pre-normalization adjoint. (2) The composed FD only converged once the fixture pinned the predictor output to a narrow band around offset ≈ 0.45 — the sampling base is integral, so offsets drifting near 0/1 put bilinear cells on the one-sided seam and a central difference disagrees with the analytic slope. **Backward verified against the real CPU backward:** `oxide_deform_conv_backward_chain` builds a cfg, runs `forward_training` + `backward_accumulate_seeded`, and compares every `LayerTrainBuffers::DeformConv` gradient (`main.weight_updates`, `offset_weight_updates`, `offset_bias_updates`, `mask_weight_updates`, `mask_bias_updates`) against the composed reference — **2 passed** (masked and unmasked). This is the check that validates the four-stage decomposition against the repo's own backward. Coverage limits: `batch_normalize=0` and `activation=linear` in both cfgs, so the layer epilogue and the `raw_output_slope` bookkeeping are **not** exercised, and the normalization gate is off in these fixtures. The composed
reference tolerance is `TOL_COMPOSED = 1e-4` rather than the single-stage `2e-6`: the device predictor runs through a GEMM whose summation order differs from the reference loop, so offsets differ in the last f32 bits and bilinear sampling amplifies that (measured worst case 2.4e-6 absolute on values ~0.16). **True-oracle test:** `oxide_deform_conv_chain` builds a real `ExecutableNetwork` from inline cfg text, runs its CPU forward, and compares the deform layer's output (bias + linear epilogue) against `deform_conv_forward_native` using the runtime's own initialized `DeformConvParams` — **3 passed** (unmasked, masked, and the dilated-layer rejection that keeps the host fallback available). That validates parameter layout and epilogue order, not just the math. The per-batch mean/stddev reduction is now native as well: `deform_offset_stats` (one thread per batch, f64 accumulation matching the source, serial by design) returns `(mean, inv_std, normalize)`; `oxide_deform_envelope` now has 7 device tests, including per-batch gate coverage and the subtlety that a **constant** batch never normalizes (`stddev > 1e-12` precedes the `|mean| > max_offset` gate, so a large mean with zero spread does nothing). Extent caveat: `deform_conv`'s output extent includes the dilated kernel extent while the oxide conv `Geom` intentionally does not, so `conv2d_f32` can only stand in for the predictors at `dilation == 1`; see `gpu_layers_plan.md` for the finding and the fixture that caught it. Note: a strict `(0,1)` bound at |x| = 80 is wrong in f32 — `1/(1+exp(-80))` rounds to exactly 1.0, so the test asserts clamp-equality and symmetry instead. No version claimed and nothing committed.

- [x] v0.2.006: standalone native f32 act-head parameter/CLS gradient composition. New RTX 4060 suite 4 passed (all blocks, actual GPU FD, guards, ownership/pool reuse); forward regression 2, exact-erf 4, host act-head 11. Audit and oracle-domain adaptation: `implemented.md` §46. Runtime backward/updates remain unchanged.

- [x] v0.2.005: native f32 act-head full/compact forward composition. RTX 4060 tests 2 + exact-erf regression 4; host act-head 11/runtime 273 passed. CPU-call probe and mutable/block/blob/error checks are in `implemented.md` §45. Four weight uploads each call; no persistent-cache/speedup claim. CPU backward and frozen params remain unchanged; f16 rejected.

- [x] v0.2.004: native f32 exact-erf GELU forward/backward via libdevice, separate from unchanged tanh GELU. Four new RTX 4060 tests passed; gradient regression 2 passed/1 ignored benchmark, act-feature regression 3, host GELU 2. Scope/audit: `implemented.md` §44; no runtime integration or f16 addition.

- [x] v0.2.003: native detached act-head feature primitive, independently verified on RTX 4060 (3 new tests, 24 extra-layer + 4 half-gather regressions; 9 host act-head tests). Audit: `implemented.md` §43. This is not runtime integration or a completed native act-head layer.
- [ ] Act-head remaining work: native full-sequence CLS-adjoint scatter, then runtime backward/update integration once eligibility is resolved. Standalone four-block/dcls math is delivered in v0.2.006 but is not called by runtime backward. F32 forward composition is delivered; shared-file loading is not proven by in-memory blob replacement. Keep the old tanh GELU unchanged; keep CPU backward until all required backward/update behavior is implemented. Resolve source training/update gap before changing eligibility/removing fallback: C++ updates all four blocks; Rust runtime discards those gradients. The question about including that training fix remains unanswered; preserve current training behavior meanwhile. Independent native math can continue.

- [x] v0.2.002: native f16 marker-gather compact inference, bit-preserving u16 feature copies and quantized marker rounding. RTX 4060 tests: new f16 suite 4, existing f16 runtime 6, f32 gather 4; host gather 12. Scope/limits and named tests: `implemented.md` §42. No f16 training/full-state API or global GPU completion claim.

- [x] v0.2.001: native f32 `[laya_marker_gather]` full/compact forward and deterministic backward, removing its per-operator host round trip and whole-network CPU-backward fallback. Real RTX 4060 tests: direct ops 4, runtime chain 1, extra-layer regression 24; host marker 12/runtime 271. Evidence and limits: `implemented.md` §41.
- [ ] **Current operator priority: GPU layers now**, directly without subagents; CPU-first sequencing is explicitly overridden, not completed. Continue `gpu_layers_plan.md` for remaining LAYA/transformer/deformable/CPU-only native paths and GPU Muon. Marker gather f16 compact inference is native in v0.2.002; act-head feature/exact-erf primitives and f32 runtime forward are native in v0.2.003–005; backward/update integration and f16 remain open. Keep parked, uncommitted YOLO-3D activation work separate; all CPU/config/axis requirements remain open, and sparse MoE stays unstarted.

- [ ] CUDA graph capture: skipped — cuda-core only has raw begin/end capture (no
  instantiate/launch wrapper) and pool addresses are not static.
- [x] fp16 convolution dgrad/wgrad: Oxide AMP stores convolution inputs/gradients in f16, uses f32 accumulation for dgrad col2im and wgrad, and returns f32 weight gradients. Verified by `MEM=3G ./tools/test_oxide.sh --test oxide_amp` (11/11 passed, including grouped and multi-chunk convolution cases). Half2 elementwise remains unnecessary.
  Standalone f16 batchnorm uses the channel-affine kernel, standalone bias +
  activation and reverse upsample forward are implemented, and all three have
  `oxide_half` / runtime f16 parity coverage.
- [ ] Implicit-GEMM conv as default: implemented (`DARKNET_OXIDE_CONV=implicit`)
  but ~5% slower (materialised B tile uses 16-byte `cp.async`, implicit only
  4-byte scalar copies); stays opt-in, `Auto` keeps the materialised path.
- [ ] f32 GEMM epilogue bias/activation and training-forward BN-apply +
  activation fusion: blocked because backward needs the pre-activation.

## Gradients

- [~] Analytical gradients everywhere: the C++-shortcut audit is done and
  `DARKNET_GRAD_MODE=analytical|cpp` gates softmax and scale_channels (see
  `implemented.md` §12). v65 added host/Oxide `NORM_CHAN*` primitives and v66
  integrated them into convolution CPU/f32-GPU forward/backward, with exact
  default derivatives and explicit C++ CUDA shortcuts. Other layer types and
  f16 inference reject these activations; no compiled-C++ golden is available.
  Other parity gaps include conv `reverse`/`deform`. Convolution `grad_centr` matches the C++ BN gate,
  per-filter mean subtraction, and `f_size % 32` skip;
  `gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate` and
  `gradient_centralization_matches_reference_and_is_noop_off_warp_sizes` pass
  on host/FakeBackend and real Oxide. YOLO IoU/GIoU geometry and
  objectness/class sigmoid-BCE deltas now have finite-difference
  coverage; the box test records Darknet's legacy 0.5 width/height comparison
  factor. DIoU/CIoU deltas have source-derived parity coverage for contained,
  edge-crossing/tied, and horizontal/vertical disjoint boxes. Do not force
  DIoU/CIoU deltas to equal naive finite differences:
  C++ `box_diou` uses `(d/c)^0.6` while `dx_box_iou` supplies the legacy gradient
  of `d/c`; CIoU similarly uses an approximate `alpha` gradient. Overlapping-window
  maxpool with `maxpool_zero_nonmax` also has finite-difference and real Oxide
  runtime-chain forward/backward/SGD parity coverage; the argmax-only kernel
  inherently writes zero gradients to non-argmax inputs.
  Weighted shortcut CPU gradients have finite-difference coverage; GPU input
  and source gradients use backend elementwise multiply/fit operations. Oxide
  assembles raw weight dot products and applies exact None/ReLU/softmax
  normalization-chain derivatives on-device, preserving `GradMode::Cpp` raw-z
  behavior; only the final per-input/per-slot weight-gradient vector crosses to
  host train buffers. Parameter/optimizer updates remain host-side.

## P8/P9 training follow-ups

- [~] Recurrent GPU execution: no-BN RNN projections and CRNN recurrent
  convolutions use backend forward/backward ops. No-BN LSTM gates/cell products,
  BPTT gate derivatives, connected backward, and recurrent/input-gradient
  accumulation use backend ops. Training retains LSTM backward caches on-device
  instead of downloading projection/gate/cell caches; timestep outputs are
  concatenated directly on-device; LSTM backward slices input/output gradients
  and assembles input gradients on-device too. RNN now also slices input/output
  sequences and assembles input gradients on-device, accumulating projection
  gradients on-device across steps before one transfer per block. CRNN forward
  and backward sequence slicing and input-gradient assembly are on-device, and
  convolution parameter gradients accumulate on-device across steps before one
  transfer per block. No-BN RNN and CRNN training pre-activation/state caches
  now stay on-device for backward, avoiding per-timestep cache downloads and
  re-uploads; host-facing forwards still capture the legacy CPU cache. Device
  parameter-gradient transfers, host accumulation/SGD, and some sequence handling
  still cross the host. BN RNN/LSTM backward uses backend projection/BPTT ops;
  BN CRNN backward uses backend convolution/BPTT ops with rolling-stat affine
  input/beta derivatives. BN scale-gradient reductions now use existing backend
  centering, channel-scale, pointwise-multiply, and channel-reduction ops; the
  Oxide path no longer transfers raw projections/convolutions and deltas to host
  just for dscale.
  Recurrent BN forward/cache capture, rolling-stat update, and optimizer remain
  host-side. FakeBackend and real Oxide parity cover parameter, rolling-stat,
  and upstream connected/convolution updates. Dynamic loss-scale finiteness
  checks include host-buffered connected/recurrent gradients. Unsupported
  recurrent activations still use CPU backward.
- [x] Host Adam: connected, standalone batchnorm, local, weighted shortcut,
  MoE fallback, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT host params use
  per-layer rate scaling and persisted host moments. DNOPT3 also persists
  device MoE Adam moments; DNOPT1/2 remain readable, and legacy DNOPT2 MoE
  moments migrate to the device on the first no-BN MoE Adam update. ViT
  preserves weight order, decay groups, gradient clipping, and its reduced mHC
  learning rate.
- [~] weighted shortcut: per-feature/per-channel oxide forward composes
  channel-scale + add; input/source gradients use backend elementwise multiply
  and shortcut-fit ops. Oxide assembles raw weight dot products and applies the
  exact normalization derivative, including `GradMode::Cpp`; only the final
  per-input/per-slot gradient transfers to host. Parameter/SGD updates remain
  host-resident. FakeBackend and real-device runtime tests cover short-source
  prefixes, forward, backward, and SGD parity.
- [ ] `reverse` conv, `deform`, adversarial yolo branch.
- [x] End-to-end real-device `subdivisions=2` deferred-gradient flush matches
  CPU weights/biases through two iterations in
  `oxide_runtime_chain::subdivisions_deferred_gradient_flush_matches_cpu`.
- [x] AMP with dropout/dropblock: real-device end-to-end runtime coverage in
  `oxide_runtime_chain::amp_dropout_and_dropblock_runtime_chain_matches_f32_reference`
  checks layer outputs, masks, backward, and SGD against f32 (these ops use the
  documented f32 boundary path under AMP).
- [ ] Real convergence run: only when the user asks.

## Data pipeline

- [~] mixup, multi-scale `random`, and source-level faithful mosaic geometry are
  implemented. Mosaic follows C++ crop alignment, flip-aware quadrant shifts,
  corner relocation, integer box clipping, and strict filtering; the seeded Rust
  RNG is not C++-bit-identical and compiled-C++ mosaic parity is not established.
  Cutmix (which C++ itself rejects), blur/noise/fog remain deferred/rejected.
- [~] Resume image-order replay is not yet guaranteed; Rust SplitMix64 remains
  deterministic but not C++-bit-identical. Hand-derived mosaic geometry and
  single-image OpenCV fixture tests pass.

## FP8 / FP4 (`fp8_fp4.md`)

- [ ] Step 5: FP8 conv via im2col + LegoGears_v2 mAP vs the C++ FP8 path. Not
  runnable here (no C++ build of `v6-dev-fp8`).

## Layer port (Phases 1b–4 — see `layers.md` phase tables)

Immediately next, in dependency order:

- [x] LAYA family (`laya_scorer`, `laya_marker_gather`, `laya_act_head`,
  `laya_loss`, `laya_encoder`, `laya_head`) — all done; see `implemented.md` §5.
- [~] Phase 3A: `[vit]` reference path is implemented: CPU forward/backward,
  header-aware weights, SGD, and GPU f32 host round-trip. Standard flash
  attention has native Oxide forward and dQ/dK/dV primitives inside the
  whole-network CPU fallback (see `implemented.md` §8). VC source audit confirmed
  the key-centering/LSE and padded-query `sq` risks; Rust avoids the padded query
  and defines the centered/clamped-score derivative contract explicitly.
  v0.0.261–v0.0.266 added and tested the VC quantization, V-Smooth, online-row,
  Q/K-scale, head composition, and B×H packing primitives. v0.0.267 wires strict
  cfg aliases/modes/permutations and CPU/GPU host-reference forward/backward;
  `vc_attention` (16 filtered tests) and `vit` (32) pass. GPU VC deliberately
  falls back to the Rust host reference on all devices/head sizes; no native VC
  kernel or compiled-C++ parity is claimed. Swin/transformer integration is
  complete in `model.rs`, `swin.rs`, `swin_block.rs`, `runtime.rs`, and
  `weights.rs`; all four aliases, CPU exact backward, Rust-defined weights,
  SGD/Adam, and GPU f32 host fallback are covered. `MEM=3G
  ./tools/test_host.sh swin` passed 25 tests, `runtime` passed 166, and `weights`
  passed 29. No native Swin kernel, C++ weights writer, or compiled-C++ parity is
  available; see `swin_plan.md`. v0.0.270 (`ecd4e54e`) completes `[tucker_attention]` on the
  host reference path; `MEM=3G ./tools/test_host.sh tucker_attention` passes 14
  tests, `weights` 32, and `runtime` 168. v0.0.271 completes MambaVision with
  strict config, explicit recurrent state, exact CPU forward/backward, seeded
  initialization, SGD/Adam, source-order `.weights`, optimizer persistence, and
  GPU f32 host fallback. `MEM=3G ./tools/test_host.sh mambavision` passed 16
  tests, `runtime` 171, and `weights` 33. Coverage includes input/all-parameter/
  state finite differences, causal convolution, MSA clip derivatives, weight
  initialization/decay, serialization, Adam-state persistence, and FakeBackend
  parity. No native kernel or compiled-C++ golden. Kimi KDA is integrated in
  v0.0.272; see `implemented.md` §8. MiniMax sparse attention is a shared
  option, not a standalone layer in the
  audited C++ branch; ViT, Tucker, and MambaVision now integrate it according
  to their source contracts.
- [x] Container-conv cache prerequisite: device convolution params are keyed by
  owner layer + expert slot; `warm_device_conv_params` warms each MoE expert,
  and host-updated experts invalidate/re-upload their slots. FakeBackend test
  covers slot separation and rewarm. The generic convolution optimizer/BN
  refresh still handles top-level owners only; MoE router/expert SGD and Adam
  have device-side paths, including BN scales, using owner/expert convolution slots.
  Router parameter values sync to host after each update because the current
  router forward consumes host params; gradients and optimizer moments stay
  device-side. DNOPT3 persists no-BN device MoE Adam moments; DNOPT4 adds BN
  expert scale moments and migrates nested host BN moments to device. DNOPT1-3
  remain readable. BN expert backward uses backend ops; BN MoE SGD/Adam update
  experts/router and BN scales on-device, with SGD momentum retained in device buffers.
- [~] Phase 2B `moe_convolutional`: real-device forward computes router pooling,
  logits, top-k masking, softmax, expert convolutions, and gate blending. For
  no-BN experts, router connected-backward, gate-column selection and gate-dot
  reductions, load-balance plus top-k softmax-Jacobian backward, router-pool
  input-gradient broadcast, and expert convolution backward use backend ops;
  gates and pooled/expert activation/output caches remain device-resident, with
  no gate/dot transfers in the normal Oxide training path. Router/expert
  parameter gradients accumulate on-device across backward calls. For SGD and
  Adam, no-BN router and expert updates run on-device without gradient transfers.
  Router parameter values sync to host after each step for the next forward,
  while expert values support deferred host sync. BN expert backward uses
  backend ops and accumulates scale/beta gradients on device;
  BN expert SGD/Adam update weights, beta, and scale on-device; SGD momentum
  stays in device buffers, while Adam scale moments use DNOPT4 (DNOPT1-3 remain
  compatible). The
  dynamic-loss-scale finiteness check covers dirty router/expert/BN-scale
  buffers (`gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check`);
  FakeBackend
  parity also checks router momentum residency, `dont_update` gradient
  retention, and BN expert backward/SGD parity.
  Verified against CPU in
  `oxide_runtime_chain::moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  (two backward calls accumulated before the first deferred SGD step, followed
  by a second deferred step, including upstream convolution updates) and
  `oxide_runtime_chain::moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients`,
  `oxide_runtime_chain::moe_batchnorm_expert_device_backward_matches_cpu_on_oxide`,
  and `oxide_runtime_chain::moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients`;
  BN Adam parity checks scales, upstream convolution, and deferred host sync.
  FakeBackend optimizer-state tests cover host BN migration and DNOPT4 scale
  round-trip; router-specific soft/tied top-k parity is in `oxide_layers_extra`.
  CPU path is done (`cdf68d00`). v67 adds source-derived pure forward/backward
  math for the EML two-branch combination (`eml_conv_math.rs`, four host tests).
  v68 adds host-reference and native Oxide f32 branch-combine ops, optional
  identity residual, and the C++ nonpositive clamp/epsilon fallbacks. v69 adds
  `[eml_convolutional]` parsing/shape inference, paired convolution parameters,
  CPU forward/backward, C++ `input_layer` then `self_layer` weight serialization,
  and SGD/Adam updates. Focused host EML tests pass 3/3; EML weights round-trip
  passes in `recurrent_weight_blocks_round_trip_in_darknet_order`. v70 runs both
  EML branch convolutions and the combine through the f32 backend on GPU
  forward, including rolling-stat BN folding; `oxide_runtime_chain` passes
  19/19, including BN/no-BN outputs and cache invalidation after host updates.
  CPU branch caches still feed fallback paths when needed. v71 adds native
  Oxide backward for no-BN EML: backend EML adjoints and both convolution
  input/weight gradients, residual routing, and host accumulation for SGD/Adam.
  v72 adds per-branch batch-stat BN forward/backward caches, exact BN scale/beta
  gradients, rolling-stat EMA propagation, and host train-buffer accumulation;
  BN EML no longer forces a CPU backward fallback. `MEM=3G PKGS=darknet-core
  ./tools/test_host.sh eml_convolutional` passes 3/3 and
  `./tools/test_oxide.sh --test oxide_runtime_chain` passes 19/19, including
  BN rolling-stat inference, finite native BN backward, EMA updates, and host
  optimizer updates. v73 adds per-branch device gradient/momentum buffers,
  native SGD for BN and no-BN EML, and deferred synchronization for both branch
  parameter blocks; SGD accumulation and finiteness checks stay device-side.
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh eml_` passed 9/9 and
  `./tools/test_oxide.sh --test oxide_runtime_chain` passed 19/19, including
  multi-step SGD momentum parity and deferred host sync. v74 adds native Adam
  for both branches and BN scales, per-branch moments, deferred synchronization,
  host fallback for CPU-backward networks, and DNOPT5 sidecar persistence while
  retaining DNOPT1–4 readers and migrating DNOPT3 EML host moments. Host EML
  tests pass 12/12, optimizer-state tests pass 5/5, and `oxide_runtime_chain`
  passes 20/20, including no-BN Adam update parity, BN-scale updates, and
  deferred sync. No compiled-C++ golden is available; CPU EML BN training still
  uses rolling rather than batch statistics, so no CPU/C++ training parity is
  claimed.
- [ ] Phase 2B `dcnv3` blocked: no reference numerics (stub in all five branches).
- [ ] Phase 2C OBB family (unblocked by decision 4; XL — also data loading, NMS,
  mAP). v75 adds legacy front-point rectangle geometry; v76 adds standalone
  CARPN convex-quad IoU; v77 adds typed geometry dispatch and an OBB NMS helper;
  v78 adds standalone legacy `[obb_yolo]` six-coordinate per-head decoding with
  source-derived front-point and integer-letterbox transforms; v79 adds a
  standalone five/seven-field OBB label parser preserving the optional front
  point; v80 adds source-derived label bounds and rotation; v81 adds
  `[net] obb=1` and `[obb_yolo]` parsing, runtime head capture, network-level
  multi-head OBB decode/NMS, and explicit AABB-training/mAP rejection; v82 adds
  separate no-augmentation OBB batch loading with direct image resize and
  preserved optional-front labels; v83 adds standalone AP matching for normalized
  legacy front-point OBB detections; v84 adds an OBB validation dataset/evaluation
  API with network decode and normalized orientation-aware scoring; v85 adds a CPU
  `detector map` CLI route for standard and legacy OBB evaluators; v86 adds the
  source-default oriented `nms_kind=obb` path and rejects unsupported alternatives;
  v87 adds a standalone horizontal-mirror transform for OBB labels; v88 applies
  caller-selected per-image flips to OBB batch tensors and labels together; v89
  adds source-derived direction-free anchor-size matching and best-anchor/cell
  assignment primitives; v90 adds seeded config-driven horizontal flips through
  an OBB-specific batch loader; v91 adds seven-float truth-row packing in the
  source batch/box-slot layout with absent-front placeholders and zero padding;
  v92 enumerates primary and strict-threshold additional per-head positive
  anchors; v93 selects best-truth and ignore/positive gates from precomputed
  source-style prediction scores; v94 adds a standalone CARPN polygon mAP helper
  for normalized quad detections and front-point labels; v96 adds source-style
  per-cell objectness deltas from the v93 match gates, including ignore smoothing
  and positive-target override; v97 adds source CIoU-OBB match-quality scoring for
  decoded geometries; v98 decodes one cell's post-YOLO channels for the source
  CIoU/default score path; v99 composes those per-truth scores with the source
  class-gated ignore and positive-threshold selection for one cell; v100 adds
  source-style class-channel deltas for positive cells; v101 adds standalone
  covariance-W2 loss terms and an `exp(-loss)` match quality for decoded OBB
  geometry; v102 adds a raw-channel covariance-branch match score for one cell;
  v103 adds the source QBBV heatmap and sampled vIoU metric; v104 adds the
  source QBBV blend and covariance-branch integration; v105 adds the source
  legacy angular fallback scorer and one-cell decode/blend; v106 adds decoded
  COBB loss/quality and QBBV blending; v107 adds raw-channel COBB match decoding
  and source branch gates; v108 adds one-cell covariance-first/COBB/angular
  score dispatch; v109 composes branch scores with per-truth cell gates; v110
  enumerates primary/additional positives across batch slots and multiple heads;
  v111 maps caller-selected head/cell predictions to batch-specific truths and
  returns tagged match gates; v112 traverses a full batch-major head tensor in
  source cell/anchor order; v113 emits source objectness deltas for batch/head
  cells with ignore and positive overrides; v114 writes those deltas into entry 6
  of one head's delta tensor without altering other entries; v115 applies
  source best/additional positive-anchor objectness overrides, class mapping and
  smooth-mode preservation; v116 writes source class-channel deltas at those
  positives with existing-target behavior; v117 adds source finite-difference
  COBB/legacy box deltas and positive-anchor writeback; v118 ports source
  covariance-W2 analytic geometry gradients; v119 routes those gradients into
  positive-anchor box-delta writeback; v120 reports `avg_cat` contributions
  from best/additional class assignments; v121 applies mapped class multipliers
  to box-delta normalization; v122 applies truth-threshold cell class and box
  deltas after the all-cell objectness pass; v123 composes ordered delta passes
  for one complete head; v124 dispatches those passes across multiple head
  tensors; v125 adds a standalone center-front-anchor point decoder; v126 routes
  optional anchor offsets through head matching and box deltas; v127 aggregates
  multi-head class and positive-assignment metrics; v128 reports source total-bbox
  counts across threshold and anchor assignments; v129 adds a standalone source
  front-point delta primitive; v130 writes those deltas to best/additional positive
  anchor channels; v131 writes them to truth-threshold-positive cells; v132
  composes front deltas in the one-head pass order; v133 dispatches that path
  across multiple heads; v134 reports source-normalized front-loss sums/counts
  across those passes and heads; v135 parses source `obb_anchors` pair/quadruple
  forms and retains per-anchor direction priors; v136 parses the front-loss
  normalizer and alias/default; v137 builds per-head delta configs from parsed
  OBB specs plus runtime dimensions; v138 supports per-head front-loss configs
  during multi-head dispatch; v139 builds network-layer delta seeds from OBB
  forward heads and labels; v140 propagates those seeds through runtime
  backpropagation; v141 adds a one-subdivision CPU SGD batch step with explicit
  Adam/subdivision rejection; v142 accumulates configured subdivisions and applies
  one normalized SGD update; v143 parses OBB `counters_per_class` into
  assignment class multipliers; v144 parses and routes OBB relative class-map
  files; v145 parses and routes covariance/COBB normalizers and QBBV blending
  options into per-head scoring; v146 wires parsed `iou_loss` to CIoU-OBB score
  dispatch; v147 routes the parsed front-loss normalizer into covariance
  match-score weighting; v148 retains standard `iou_loss` choices in the parsed
  spec and routes them to front-point deltas; v149 retains `iou_thresh_kind` in
  the parsed spec; v150 explicitly rejects `ciou_obb_mul` training until its
  distinct loss/gradient path is implemented; v151 aggregates front-vector angle
  error mean/threshold counts in front-aware stats across passes and subdivisions;
  v152 separates weighted MSE matching scores from the source CIoU-OBB box-delta
  fallback; v153 adds covariance and COBB box-delta contributions independently
  when both weighted branches are active; v154 combines their matching scores by
  configured weights; v155 reports source raw `sum(delta²)` for all-MSE OBB heads;
  v156 reports the objectness-normalized non-box delta-squared component; v157
  allows distinct weighted MSE matching and CIoU-OBB regression configs through
  batch dispatch; v158 aggregates source-style MSE and supported non-MSE loss
  costs across heads, with non-MSE box losses counted only for best/additional
  anchor assignments; v159 routes `detector train` through a bounded CPU OBB
  SGD loop with configured subdivisions, direct resize, seeded horizontal flips,
  checkpoints and loss CSV logging, and explicitly rejects `-map`, Adam, dynamic
  loss scaling, unsupported augmentation, and OBB GPU training. The CLI test
  covers one iteration and the `-map` rejection; all 119 OBB-filtered core tests
  plus the CLI test passed. v160 adds a separate subdivision-aggregated OBB
  metrics CSV (average IoU/GIoU, class output, assignment counts, loss components,
  front loss, and orientation accuracy/error buckets); the CLI schema/count test,
  four detector-train CLI tests, and 119 OBB-filtered core tests pass. v161 routes
  CPU OBB training `-map` through orientation-aware validation and shared cadence;
  the test verifies `-map_every 2` and the final-iteration evaluation, best weights,
  and map log. `detector_train_` passed 4 CLI tests, `map_` passed 4 CLI + 11 core
  tests, and `obb_` passed 119 core + 1 CLI test. v162 routes `detect-image`
  `[net] obb=1` outputs through the orientation-aware decoder and prints legacy
  front-point geometry in human and `det_obb` TSV output; mixed OBB/modern heads
  fail explicitly. The focused CLI tests and `detect_` passed 2 CLI tests; `obb_`
  passed 119 core + 2 CLI tests. v163 accepts source `default` and `cornersnms`
  OBB NMS kinds with AABB-IoU suppression, retaining orientation-aware `obb` and
  source aliases; `obb_` passed 120 core + 2 CLI tests. v164 implements source
  `greedynms` (fixed 0.6 DIoU exponent) and `diounms` (`beta_nms`, default 0.6)
  using AABB box formulas; non-finite betas and mismatched multi-head DIoU betas
  fail explicitly. v0.0.249 adds CPU OBB Adam with subdivision-correct updates and
  DNOPT sidecar moments; v0.0.250 adds photometric augmentation while retaining
  explicit rejection of unsupported geometry; v0.0.251 adds weighted OBB NMS
  (`obb_w`/`obb_wnms`); v0.0.252 defaults omitted OBB jitter to zero; v0.0.253
  implements multiplicative CIoU-OBB loss and its six-coordinate finite-difference
  gradients. The direct `cargo test -p darknet-core -p darknet-cli -j 2 -- obb_` filter
  passed 135 core + 4 CLI tests; after systemd scope availability was confirmed,
  `MEM=3G ./tools/test_host.sh obb_` also passed 135 core + 4 CLI tests.
  No compiled-C++ parity is established.
  v0.0.258 adds sample-level CARPN mAP dispatch through `obb_detector_map`; focused
  `obb_map` (5 core), `carpn_map` (3), and `carpn_polygon_iou` (2) host filters pass.
  v0.0.259 completes CPU CARPN network inference, `detect-image`, and direct
  `detector map` via real cfg/weights-backed fixture tests; `carpn_refine_cfg` (2),
  `carpn_refine_weights` (1), `carpn_refine_forward` (3), `carpn_refine_decode` (5),
  `carpn_map` (6), `carpn_cli` (1), `detect_image_routes_carpn` (1), and
  `detector_train_rejects_carpn` (1) pass. The legacy `obb_` filter passes 138 core
  + 4 CLI tests. CARPN Stage-2 training and scheduled `detector train -map` remain
  deferred. Still open: broader OBB GPU training/validation, remaining cfg/training-
  option wiring, source-faithful geometric augmentation/label transforms, and any
  unimplemented OBB source paths.
- [x] Phase 3B bounded Stage-1 CPU layer ports: `graph_conv` v0.1.003
  (`implemented.md` §20), WMHF v0.1.004 (§21), Clifford v0.1.006 (§23), and
  `recursive_block` v0.1.007 (§24) are source-derived from `Jev_laya`; no
  compiled-C++ golden/parity or GPU path is claimed. CPU Muon is integrated in
  v0.1.005 (§22). This closes the bounded CPU-port task set, not every source
  variant: DSConv multi-template `morph=0/1` remains open, and DCNv1/2 plus
  `[dcnv4]` still use the bounded CPU/reference or host-fallback scope recorded
  in §§17–19. Native GPU support and GPU Muon remain Stage 2.
- [x] CPU Muon: spec and Stage-1 hybrid implementation complete in v0.1.005
  (`implemented.md` §22). Eligible matrices use paper-derived Muon; other
  trainable parameters use AdamW fallback. DNOPT6 persists shape-checked
  momentum while DNOPT1–5 remain readable. `muon_` (13), `weights` (43), and
  `runtime` (204) host tests passed. No training, Oxide tests, or compiled-C++
  golden/parity. GPU Muon is deferred to Stage 2.
- [~] Phase 4 foveation / depth / DETR: Stage-1 CPU `[hyperbolic_foveation]` is complete in v0.1.008 (`implemented.md` §25), `[foveation_film]` in v0.1.009 (§26), `[depth_anything]` in v0.1.010 (§27), and `[dpt_head]` in v0.1.013 (§30); LAYA encoder/head are already implemented. `[yolo_3d]` and DETR remain open. DETR is still broken in C++ (paper only).
- [x] Stage-1 A1 axis/layout audit of every already-ported family named in `AGENTS.md`: completed in v0.1.012 (`implemented.md` §29), with the initial subset recorded in §28. `STATUS.md` tabulates the source contracts and named tests. Per-backlog axis DoDs for missing-layer ports and the remaining v0.1–v0.3 work stay open.
- [x] v0.1.014 C++ enum-boundary regression: `[network]` remains top-level metadata, and the unused `[blank]`, `[empty]`, `[silence]` aliases reject rather than becoming identity layers; `model::tests::unused_cpp_layer_enum_names_are_not_rust_noop_layers` passed (see `implemented.md` §31). No compiled-C++ parity is claimed.
- [x] v0.1.015 axis-coverage gate and shared fixtures: every `LayerKind` variant now records a `Pinned`/`Open`/`RustOnly` decision in `AXIS_PINS`, enforced by the exhaustive `LayerKind::axis_pin` match, and `axis_fixture.rs` supplies the shared sentinel/index helpers (see `implemented.md` §32 and `axis-coverage.md`). Baseline is 18 pinned / 41 open / 5 Rust-only / 3 unported.

- [x] v0.1.016 convolutional axis pins: NCHW/group partitioning and the dilated-extent contract are pinned; finding F1 was retracted (C++ CPU geometry ignores dilation, cuDNN pads by `pad*dilation`). See `implemented.md` §33 and `axis-coverage.md`.- [ ] Axis tiers 1-4 (see `axis-coverage.md`): add the missing named pins, starting with `[convolutional]`, which also carries finding F1 (dilated output extent ignores `dilation` in `conv_dim`). Each behavior fix gets its own version; do not weaken a pin to match the implementation.

- [x] v0.1.017 GPU conv parity: the device extent now follows the CPU for `dilation > 1` (the kernels were already correct). See `implemented.md` §34.
- [~] `gui_plan.md` — implemented in v0.1.019 across `darknet-gui-project` / `darknet-gui-engine` / `darknet-gui`: JSON+txt sidecars per `G3`-`G7`, scan/filter/nav/undo, the `G9`-`G16` edit model, actions (interpolate/clipboard/snap/homography), project creation, atomic class remap, `detect-server` + worker, and the launcher/canvas/views/class-editor/overlay shell. Still open: add/delete classes from the dialog (engine supports it), and the behaviours behind keys that are mapped but unimplemented (mass-delete, homography, merge, heatmaps, screenshots, tiling, jump, delete-image/undo-delete), the train runner, and the Task 1.5 DarkMark file diff. Hotkeys are transcribed from `DMContent::keyPressed` as of v0.1.021.
## Active and unstarted plans

- [x] `onnx.md` — `darknet-rs export-onnx`; all 21 implementation, verification, and bookkeeping steps are complete.
- [x] `carpn_map_plan.md` — bounded caller-supplied sample dispatch implemented in v0.0.258.
- [x] `carpn_network_map_plan.md` — v0.0.259 completes CPU network inference, `detect-image`, and direct `detector map`; CARPN training/`detector train -map` remain explicitly deferred.
- [x] `dcnv4_plan.md` — source-audited CPU/reference + GPU f32 host-fallback implementation completed in v0.0.273; v0.0.274 matches the deformable-layer `try_fix_nan` config default. No native kernel or compiled-C++ parity.
- [x] `deform_conv_plan.md` — bounded DCNv1/2 CPU reference and f32 host-fallback implementation completed in v0.1.001. No native kernel or compiled-C++ parity; CPU Muon was added separately in v0.1.005.
- [x] `dsconv_plan.md` — bounded single-template CPU implementation completed in v0.1.002; multi-template modes remain open. CPU Muon was added separately in v0.1.005. No native kernel or compiled-C++ parity.
- [x] `graph_conv_plan.md` — Stage-1 CPU implementation completed in v0.1.003; CPU Muon was added separately in v0.1.005 and native GPU support remains Stage 2. No compiled-C++ parity.
- [x] `wmhf_plan.md` — bounded Stage-1 CPU implementation completed in v0.1.004; source/config/backward/weights audit is in `wmhf_notes.md`. CPU Muon was added in v0.1.005. No compiled-C++ golden.
- [x] `clifford_plan.md` — source-derived Stage-1 CPU implementation and focused verification completed in v0.1.006; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `recursive_block_plan.md` — source-derived Stage-1 CPU implementation and focused verification completed in v0.1.007; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `hyperbolic_foveation_plan.md` — source-derived Stage-1 CPU support and verification completed in v0.1.008; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `foveation_film_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.009; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `depth_anything_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.010; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `dpt_head_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.013; no compiled-C++ golden/parity. Native GPU and GPU Muon remain Stage 2; `[yolo_3d]` remains separate.
- [x] `[yolo_3d]` config-helper preparation only — v0.1.024: strict source-backed spec/parser and checked shape helpers; 12 `yolo_3d_cfg` and 88 `model::` tests passed under 3G scopes. See `implemented.md` §38; hand-derived from source, no compiled-C++ golden.
- [x] `[yolo_3d]` standalone label/truth helpers — v0.1.025: pure/mixed parsing, source tracks/ownership, deterministic cap/filtering, crop/flip intrinsics and mosaic-zeroing; 31 `yolo_3d` tests (19 new) and 43 `label` tests passed in 3G scopes. See `implemented.md` §39 for the scoped audit and sampling/API boundaries; no compiled-C++ golden.
- [x] `[yolo_3d]` raw model/runtime foundation — v0.1.026: aliases/identity shape and shared class/K validation, separate raw cuboid forward outputs, exact supplied-seed backward, zero-byte weightless head, seeded upstream SGD/Adam/Muon and DNOPT checks, legacy API/CLI rejection and GPU guards. 42 core + 1 CLI `yolo_3d` tests passed; all stated regression filters and oxide-feature CLI check passed under 3G scopes. Scope/evidence audit: `implemented.md` §40. No full layer-port or compiled-C++ parity claim.
- [ ] `yolo_3d_plan.md` — implement projected math/assignments/loss with exact Rust derivatives and Cpp deltas, standalone/mixed training APIs consuming the existing truth helpers, decode/evaluator and the A2 anchor/entry/vertex test, then depth/metric CPU consumption/reference validation. Resolve parsed map/counter metadata in the loss/decode integration. Raw forward and seeded optimizer tests do not establish full loss-to-conv training. Legacy detector APIs/CLI commands reject cuboid networks until their dedicated APIs exist; GPU remains Stage 2.
- [x] `muon_plan.md` — Stage-1 CPU hybrid Muon implementation and focused verification completed in v0.1.005. GPU Muon remains Stage 2; no compiled-C++ golden/parity.
- [~] `gui_plan.md` — egui DarkMark replacement; M0–M6 partly implemented in v0.1.019–v0.1.021 (crates, `detect-server`, sidecars, scan/filter/nav/undo, edit model, actions, class remap, shell). Open: class add/delete dialog, mapped-but-unimplemented hotkeys, train runner, DarkMark file diff.

## Known evidence gaps (not user questions)

- `[dcnv3]`: forward/backward/update are stubs in all five branches that carry it
  (`deform`, `Jev_laya`, `fog`, `jepa`, `depth_3d`); only the `.weights` order is
  portable against C++.
- `[contrastive]` detection branch needs the yolo head's per-box class ids
  (`YoloHeadOutput` only carries the raw output tensor).
- DETR is broken in C++, so Phase 4's `detr_decoder` has no reference numerics.
- No C++ build of any owning branch is available here, so C++ dump parity is
  "NOT RUN" everywhere.

## Not implemented by design / no scanned cfg needs it

- All parsed scalar activations with CPU forward/derivative support now have
  oxide kernels, including `gelu`, `elu`, `selu`, `hard_mish`, `loggy`,
  `hardtan`, `relie`, `lhtan`, `plse`, and `stair`. `normalize_channels*` is
  supported on `[convolutional]` for CPU and f32 GPU runtimes; the parser rejects
  other layer types and f16 inference rejects it explicitly.
- xnor (74 layers), share_index (54), `new_coords=1` yolo (path looks wrong),
  antialiasing, conv grad clamps / `try_fix_nan` clamps,
  `rewritten_bbox`/`total_bbox` logging.

## Known small diffs vs C++

- CPU resize uses `image` Triangle (differs for non-network-size inputs); EXIF
  orientation ignored.
- Dilated conv: oxide needs `pad*dilation` (only resnet152_trident); darknet-core
  output dims ignore dilation.
- relu/relu6 NaN -> 0 on oxide (C++ NaN); mish/swish at -inf -> NaN.
- C++ CUDA reference runs `cudnn_half=1`, so layer dumps differ from pure fp32 by
  up to 5.7e-3 abs (not an oxide error).

## Version rewrite v0.0.001 – v0.0.246

- [x] Snapshot main tree (status, index, refs, stash)
- [x] Backup tag `backup/rust-pre-version` -> `a4c40375` (old tip)
- [x] Guard: 246 commits `e05545c1..a4c40375`, 0 merges, none reachable from other branches
- [x] `git rebase --exec` in temp worktree (branch `rust-versioned`); hook checks each replayed tree equals the original before amending
- [x] Verify: 246/246 rows; parent chain; trees; author + committer name/email/date; raw message bytes after subject; subject sequence `v0.0.001…246`; tip tree == old tip tree; main tree and other refs unchanged
- [x] Swap `refs/heads/rust`: `a4c40375` -> `d1972af5` (compare-and-swap); temp worktree and branch removed
- Map (old -> new per index): `tasks/rust/version-map.tsv` (git-excluded)

Review:
- Only subject lines changed. `rust milestone vN: ` prefix stripped on 163 commits, 83 subjects kept as-is, body bytes identical.
- Committer identity and date restored from the original commit (the amend would otherwise use the current git config identity).
- Not pushed. `rust` is now 246 ahead / 7 behind `origin/rust`; updating the remote needs `git push --force-with-lease origin rust` by hand.
- Left alone: 25 old-hash references in `tasks/rust/` (`implemented.md` 16, `layers.md` 7, `todo.md` 1, `STATUS.md` 1). They still resolve via the backup tag; map above gives replacements.
- Rollback: `git update-ref refs/heads/rust backup/rust-pre-version`

## ONNX export

- [x] `darknet-rs export-onnx` is implemented: protobuf encoder (`v0.0.254`), supported-layer exporter (`v0.0.255`), CLI (`v0.0.256`), and Rust-runtime checker (`v0.0.257`). Focused results: `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; `MEM=3G ./tools/test_host.sh export_onnx_args` passed 1 CLI test; `cargo build -p darknet-cli -j 2` passed. The CLI exported both fused and unfused models from `build/logs/verify/lego_cur_cpu/run.cfg` and `run_final.weights`.
- [x] `python3 -m py_compile tools/onnx_check.py` passed. ONNX Runtime numeric checks against Rust CPU layer dumps passed at tolerance `1e-4`: fused worst relative error `3.257e-06`; unfused `3.908e-05`; Mish-opset-18 variant `3.831e-06`. The Mish test changed Leaky activations to Mish in a temporary cfg while retaining the same real weights; image was `artwork/dog.jpg`.
- Canonical `/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2` assets are absent in this checkout, so verification used the local matching cfg/weights plus `build/logs/bench/LegoGears.names`; no compiled-C++ parity is claimed. `onnx` Python package is absent, but `onnxruntime`, NumPy, and OpenCV are installed and the emitted models loaded and executed successfully in ONNX Runtime.
- [x] ONNX plan bookkeeping and verification are complete; version map, `implemented.md`, and `STATUS.md` record the milestone evidence. Follow-ups listed in `tasks/rust/onnx.md` remain out of scope.
