# Layer support matrix (branch `rust`)

One row per `LayerKind` variant, with its CPU and GPU status, so a missing CPU
port and a missing native GPU kernel are never confused with each other. This
file is the per-kind complement of `implemented.md` (what landed, with
evidence), `todo.md` (open work), `axis-coverage.md` (axis pins) and
`gpu_layers_plan.md` (native GPU work queue).

Evidence rule, inherited from `AGENTS.md`: **hand-derived from source, no
compiled-C++ golden**. A `native` cell means "no host round trip in the GPU
arm", **not** "the layer is complete on GPU". A `device` backward cell likewise
does not imply runtime optimizer integration.

## Contents

- [How this file was derived](#how-this-file-was-derived)
- [Legend](#legend)
- [Counts](#counts)
- [Matrix](#matrix)
- [GPU hard rejections (no device path at all)](#gpu-hard-rejections-no-device-path-at-all)
- [GPU forward but CPU backward](#gpu-forward-but-cpu-backward)
- [Partial device paths](#partial-device-paths)
- [Optimizer status](#optimizer-status)
- [f16 inference coverage](#f16-inference-coverage)
- [CPU-only gaps (Stage 1)](#cpu-only-gaps-stage-1)
- [Blocked, not plannable](#blocked-not-plannable)
- [Defects found while deriving this matrix](#defects-found-while-deriving-this-matrix)
- [Maintenance](#maintenance)

## How this file was derived

Read from code on 2026-10-18, not from plan prose. Line numbers drift; re-derive
with the command given rather than trusting the number.

| Fact | Source | Re-derive |
|---|---|---|
| The 65 represented variants + `Unsupported` | `src/darknet-core/src/model.rs:1579` | `grep -n "pub enum LayerKind" -A 90 src/darknet-core/src/model.rs` |
| CPU forward arms (all variants) | `src/darknet-core/src/runtime.rs:2310-3300` | `grep -nE "^ {16}\(LayerKind::" src/darknet-core/src/runtime.rs` |
| CPU backward arms | `runtime.rs:3395-6700` | same command, rows between 3395 and 6700 |
| GPU full/training forward arms | `runtime.rs:11649-13070` (inside `forward_with_state_impl`) | same command, rows between 11537 and 13100 |
| GPU compact/inference forward arms | `runtime.rs:13436-13717` (inside `forward_compact`, starts 13206) | same command, rows between 13420 and 13740 |
| GPU backward arms | `runtime.rs:14079-15660` (inside `backward_accumulate_gpu_unscaled`, starts 13822); catch-all at 15668 returns `UnsupportedLayer` | same command, rows between 13820 and 15670 |
| GPU pre-flight hard rejections | `runtime.rs:83-102`, `runtime.rs:11402-11490`, `runtime.rs:13214-13295` | `grep -n "GPU execution is not implemented" src/darknet-core/src/runtime.rs` |
| Whole-network CPU-backward fallback | `runtime.rs:15707` | `grep -n "fn requires_cpu_backward_fallback" -A 40 src/darknet-core/src/runtime.rs` |
| Activation device coverage | `src/darknet-core/src/model.rs:367` | `grep -n "fn to_cuda_f32_kind" -A 30 src/darknet-core/src/model.rs` |
| f16 compact inference allowlist | `runtime.rs:8389-8440` | `grep -n "fn forward_compact_f16" -A 50 src/darknet-core/src/runtime.rs` |
| Axis pin status per variant | `tasks/rust/axis-coverage.md` | that file, "Pinned"/"Open"/"Rust-only" tables |

## Legend

| Column | Values |
|---|---|
| `cfg` | `yes` = section parses to this kind; `alias` = several section names map here; `pass` = parsed but only as identity/pass-through; `no` = no representation |
| `wts` | `C++` = source weight order, round-trip tested; `Rust` = Rust defines the order (no C++ writer); `none` = weightless |
| `cpu fwd` | `fwd` real forward; `pass` identity pass-through only; `frozen` real forward, weights frozen; `inf` inference only |
| `cpu bwd` | `exact` exact derivative of the Rust forward; `head` gradient owned by the head/loss module; `frozen` no parameter/input gradient by design; `pass` identity delta; `none` absent |
| `axis` | `pin` named `axis_layout` test exists; `T1..T4` open tier in `axis-coverage.md`; `rust` Rust-only contract, no C++ anchor |
| `gpu fwd` | `native` device kernels; `host` host round trip inside the GPU arm; `pass` identity pass-through; `none` hard rejection |
| `gpu bwd` | `device` backend primitives; `mixed` device where supported, CPU fallback otherwise; `cpu` forces whole-network CPU backward; `staged` only forwards the incoming gradient to the previous layer; `head` loss/head module; `none` |
| `f16` | `yes` accepted by `forward_compact_f16`; `no` explicit `UnsupportedLayer` |

## Counts

| Status | Count |
|---|---|
| Represented `LayerKind` variants | 65 (+`Unsupported`) |
| CPU forward arm | 65 |
| CPU standalone parameter backward | all parameterized kinds; head/pass-through kinds own no standalone backward |
| CPU-only forward, no GPU path (hard reject) | 11 |
| GPU forward but CPU/mixed backward | 9 |
| GPU native forward + device backward | 23 |
| f16 inference | 11 variants (12 section names) |
| Axis pinned | 19 variants / 21 tests |
| Axis open | 41 (Tier 1: 14, Tier 2: 16, Tier 3: 3, Tier 4: 8) |
| Axis Rust-only | 5 |
| Incomplete CPU port | 1 (`Yolo3d`); blocked: `Dcnv3`, `DetrDecoder` |
| GPU Muon | partial hybrid integration: connected matrices use native Oxide momentum/lookahead, Newton–Schulz and direction+decay update; standard non-BN convolution, local per-location, RNN, LSTM, deconvolutional, and transformer-family matrices use native Oxide Newton–Schulz with gradients/state/final updates staged through the host. Convolution gradients are downloaded from device at the optimizer boundary, and updated parameters are re-uploaded to the device cache. BN convolutions remain rejected. Connected momentum is cached between steps but mirrored to host each update; weights remain host-canonical. Standalone BatchNorm uses host AdamW fallback; other parameter families open |

## Matrix

| Kind | cfg | wts | cpu fwd | cpu bwd | axis | gpu fwd | gpu bwd | f16 |
|---|---|---|---|---|---|---|---|---|
| `Convolutional` | yes | C++ | fwd | exact | pin | native | device | yes |
| `Deconvolutional` | yes | C++ | fwd | exact | pin | native / host (unsupported act) | native / cpu fallback (unsupported act) | no |
| `Connected` | yes | C++ | fwd | exact | T1 | native / host (unsupported act) | mixed | no |
| `MaxPool` | alias | none | fwd | exact | T1 | native (`depth` rejected) | device | yes |
| `LocalAvgPool` | alias | none | fwd | exact | T1 | native (channel-changing rejected) | device | yes |
| `AvgPool` | alias | none | fwd | exact | T1 | native | device | yes |
| `Route` | alias | none | fwd | exact | T1 | native | device | yes |
| `Shortcut` | alias | C++ | fwd | exact | pin | native | mixed (weight grad host) | yes |
| `Upsample` | alias | none | fwd | exact | T1 | native | device | yes |
| `Reorg` | alias | none | fwd | exact | T1 | native | device | no |
| `ChannelSlice` | yes | none | fwd | exact | pin | native | device | no |
| `ChannelShuffle` | yes | none | fwd | exact | T1 | native | device | no |
| `Softmax` | alias | none | fwd | exact | T1 | native | device | no |
| `Dropout` | yes | none | fwd | exact | T1 | native | device | no |
| `ScaleChannels` | yes | none | fwd | exact | T1 | native | mixed | no |
| `Sam` | yes | none | fwd | exact | T1 | native f32 / host (unsupported activation) | device / CPU fallback | no |
| `BatchNorm` | yes | C++ | fwd | exact | rust | inf (training rejected) | device | yes |
| `Crop` | yes | none | fwd | exact | rust | inf (training rejected) | device | no |
| `Local` | yes | C++ | fwd | exact | rust | native f32 / host (unsupported activation) | device / CPU fallback | no |
| `Gru` | yes | C++ | fwd | exact | rust | **none** (falls to "inconsistent") | cpu | no |
| `Rnn` | yes | C++ | fwd | exact | T3 | mixed (device w/o BN) | mixed | no |
| `Lstm` | yes | C++ | fwd | exact | T3 | mixed (device w/o BN) | mixed | no |
| `Crnn` | yes | C++ | fwd | exact | T3 | mixed (device w/o BN) | mixed | no |
| `Moe` | alias | C++ | fwd | exact | pin | native | device | no |
| `EmlConvolutional` | alias | C++ | fwd | exact | pin | native | device / CPU (BN) | no |
| `GraphConv` | alias | C++ | fwd | exact | T4 | none | none | no |
| `Dcnv4` | alias | C++ | fwd | exact | pin | native (incl. rolling-stat BN, <=225 points, supported scalar/NormalizeChannels* acts incl. Revleaky) / host otherwise | mixed (native math and SGD/Adam update; learned parameter tensors cached across GPU calls; Adam moments resident between updates, sidecar on save) | no |
| `DeformConv` | alias | C++ | fwd | exact | T4 | native (dilation 1, no BN) / host otherwise | native (same gate; one host hop for the normalization Jacobian) | no |
| `DsConv` | alias | C++ | fwd | exact | T4 | none | none | no |
| `Wmhf` | alias | C++ | fwd | exact | T4 | none | none | no |
| `Clifford` | yes | C++ | fwd | exact | T4 | none | none | no |
| `RecursiveBlock` | alias | body only | fwd | exact | T1 | none | none | no |
| `HyperbolicFoveation` | yes | C++ | fwd | exact | T4 | none | none | no |
| `FoveationFilm` | alias | C++ | fwd | exact | T4 | none | none | no |
| `Vit` | yes | C++ | fwd | exact | pin | host | cpu | no |
| `Swin` | alias | Rust | fwd | exact | pin | host | cpu | no |
| `TuckerAttention` | alias | C++ | fwd | exact | pin | host | cpu | no |
| `MambaVision` | alias | C++ | fwd | exact | pin | host | cpu | no |
| `KimiKda` | alias | C++ | fwd | exact | pin | host | cpu | no |
| `DepthAnything` | yes | C++ | frozen | frozen | T4 | none | none | no |
| `DptHead` | alias | C++ | fwd | exact | pin | none | none | no |
| `LayaEncoder` | yes | C++ | frozen | frozen | pin | host | cpu | no |
| `LayaHead` | yes | C++ | frozen | frozen | pin | host | cpu | no |
| `LayaScorer` | yes | C++ | frozen | frozen | pin | native | device | no |
| `LayaMarkerGather` | yes | none | fwd | exact | pin | native | device | yes |
| `LayaActHead` | yes | C++ | fwd | exact (CPU path) | pin | native | cpu (primitive native) | no |
| `LayaLoss` | yes | none | head | head | pin | native (loss on host) | head | no |
| `JevCe` | alias | none | head | head | T2 | native (loss on host) | head | no |
| `Segmentation` | alias | none | head | head | T2 | native (loss on host) | head | no |
| `OmdLoss` | alias | none | pass + head | head | T2 | pass | head | no |
| `Contrastive` | yes | none | pass + head | head | T2 | pass | head | no |
| `Logistic` | yes | none | head | head | T2 | native | head | no |
| `Regression` | yes | none | pass | pass | T2 | pass | pass | no |
| `Cost` | yes | none | pass + head | head | T1 | pass | pass | no |
| `Region` | yes | none (C++ has block) | pass | pass | T2 | pass | pass | no |
| `Detection` | yes | none | pass | pass | rust | pass | pass | no |
| `Yolo` | yes | none | pass | head | T2 | pass | head | yes |
| `GaussianYolo` | alias | none | pass | head | T2 | pass | head | yes |
| `ObbYolo` | yes | none | pass | head | T2 | pass | head | yes |
| `ObbRefine` | yes | C++ | inf | none | T2 | none | none | no |
| `Yolo3d` | alias | none | pass (raw logits) | pass (supplied seed) | T2 | none | none | no |
| `Yolox` | yes | none | pass + head | head | T2 | pass | head | no |
| `Ppyoloe` | yes | none | pass + head | head | T2 | pass | head | no |
| `Yolonas` | yes | none | pass + head | head | T2 | pass | head | no |
| `CenterNet` | yes | none | pass + head | head | T2 | pass | head | no |
| `Unsupported` | n/a | n/a | error | n/a | n/a | error | n/a | error |

## GPU hard rejections (no device path at all)

Eleven variants abort before any kernel runs. Source: `runtime.rs:83-102`
(`reject_gpu_hyperbolic_foveation`) and `runtime.rs:11402-11490`.

| Variant | Rejection message | Why it exists | Work item |
|---|---|---|---|
| `GraphConv` | `GPU execution is not implemented in the Stage-1 CPU port` | Stage-1 CPU port only | shared deformable/neighbour sampling wave |
| `DsConv` | same | Stage-1 CPU port only (multi-template also unsupported on CPU) | shared sampling wave |
| `Wmhf` | same | Stage-1 CPU only (DWT + EMA scan + 7 sub-convs) | container/XL wave |
| `RecursiveBlock` | same | Stage-1 CPU only; full-recurrence gradient | container/XL wave |
| `Clifford` | same | Stage-1 CPU only (XL mode matrix) | container/XL wave |
| `HyperbolicFoveation` | `Stage-1 CPU support` | needs Newton-inverse warp kernel + detect/mAP reprojection | XL wave |
| `FoveationFilm` | same | depends on the HF forward cache | XL wave |
| `DepthAnything` | same | frozen DINOv2 ViT-S/14 | attention wave (after ViT backward) |
| `DptHead` | same | four-tap fusion neck over DPT features | attention wave |
| `Yolo3d` | same | raw CPU forward only; loss/decode absent | CPU first, then GPU |
| `ObbRefine` (CARPN) | `GPU inference is not implemented` / `GPU CARPN training is unsupported` | CPU inference only; Stage-2 training deferred | after CARPN CPU training |

`Gru` is not in that list but has **no GPU arm** either; it falls through to
the match's catch-all `InvalidConfig("... and parameter set are inconsistent")`
(`runtime.rs:13070-13084`). That message is wrong and should be an
`UnsupportedLayer`. `Local` has a bounded native f32 forward/backward path since
v0.2.048; its optimizer remains host-staged and unsupported activations retain
the legacy host route.

> **Wave 1 status (deformable sampling) — `[deform_conv]` is wired.** The device
> forward and backward are live for networks with `dilation == 1` and no batch
> norm; anything else keeps the host path via a single shared predicate
> (`has_host_only_deform`) used by both the forward and the backward, so they can
> never disagree about which path ran. Evidence: the five device primitives, the
> host twins (33 unit tests), the composed backward against the real CPU
> gradients (`oxide_deform_conv_backward_chain`, 3), and the end-to-end
> `oxide_runtime_chain` device test (forward outputs, one SGD step, every
> parameter block, masked) plus the retained host-fallback test for the BN case.
> The only host-resident step in the device path is the batch-serial
> normalization Jacobian. `[dsconv]` and `[graph_conv]` remain unwired.
> **DCNv4 is bounded-native in v0.2.013–014:** <=25 active points, supported
> elementwise activation, optional rolling-stat BN, no other CPU fallback/
> host-only deform_conv. Both directions share the global gate. BN affine/
> adjoint and post-backward 0.9 EMA reduce on device (per-channel metadata syncs
> to host); no batch-stat normalization substitution. Offsets/raw outputs
> recompute; parameter gradients download to host SGD/Adam buffers. Large/
> unsupported-activation/mixed networks keep host caches. Host runtime 282,
> host DCNv4 26, focused device BN 2/runtime 3, composition 3 and aggregate 6
> passed. Previous full runtime-chain timed out at 600 s in ViT; not a pass.
> Layout/source-oracle evidence is in §49; runtime/BN audits in `implemented.md`
> §§50–51. Not native optimizer/resident-weight or C++ training-BN completion.

## GPU forward but CPU backward

`runtime.rs:17107 requires_cpu_backward_fallback` forces the **whole network**
onto the CPU backward path when any of these is present. One such layer
de-optimizes an entire model, which is why this list matters more than the raw
layer count.

| Variant | GPU forward | Missing backward piece |
|---|---|---|
| `Dcnv4` excluded/mixed variants | host round trip (>225 points, unsupported activation or another fallback layer) | remaining unsupported/unknown activations and large-kernel paths; BN and NormalizeChannels* math native since v0.2.014/015; Revleaky native since v0.2.026; 225-point bound since v0.2.027 |
| `Vit` | host round trip | device attention/MLP backward (flash dQ/dK/dV primitives exist) |
| `Swin` | host round trip | device windowed-attention backward |
| `TuckerAttention` | host round trip | device low-rank factor backward |
| `MambaVision` | host round trip | device selective-scan backward |
| `KimiKda` | host round trip | device delta-rule backward |
| `LayaActHead` | **native f32** composition | runtime backward integration blocked on the open eligibility question (`implemented.md` §43) |
| `Sam` (unsupported activation) | host round trip | unsupported activation uses the whole-network CPU-backward fallback |

`DeformConv` **left this table** in v0.2.007: it now has a native forward *and*
backward, gated on `dilation == 1` without batch norm. Its remaining limitation
(one host hop for the batch-coupled normalization Jacobian) is recorded in the
main matrix row instead. `Deconvolutional` left this table in v0.2.039: supported
f32 activations use native forward/backward; unsupported activations retain the
whole-network host fallback.
| `Connected` | native, but host round trip when activation is unsupported | no separate Revleaky gap since its v0.2.026 mapping to Leaky |
| `Rnn`/`Crnn` | device, but host when activation unsupported | device kernels for the remaining activations |
| `Moe` | device | device path for unsupported expert activations |
| `ScaleChannels` | device | device path for unsupported activations |

Activation coverage is the quiet limiter: `model.rs:367` maps all scalar
activations, including `Revleaky` (reusing the Leaky f32 op since v0.2.026),
to device kinds. The `NormalizeChannels*` family uses separate tensor-level
primitives; unknown `Other` values remain unsupported.

`vc_attention` is worse than "forward only": it is host-reference on **all**
devices and head sizes (no native kernel at all), even though standard
`flash_attention` has native Oxide forward and dQ/dK/dV primitives.

## Partial device paths

| Area | State | Gap |
|---|---|---|
| MoE | device forward/backward, device SGD/Adam for router and experts, DNOPT3/4 moments | BN-expert **SGD** still host-side per `STATUS.md`; router parameter values sync to host each step |
| EML | device forward, device SGD/Adam (DNOPT5), BN scale/beta gradients | BN backward CPU fallback per v71; documents disagree on later state — verify against `implemented.md` §7 before planning |
| Recurrent | device projections/BPTT, on-device caches and gradient accumulation without BN | BN forward/cache, rolling-stat update and optimizer remain host-side; some parameter-gradient transfers still cross the host |
| DCNv4 | bounded native f32 forward/backward through 15x15/225 points incl. rolling-stat BN and 0.9 EMA, NormalizeChannels*/Revleaky activations, device gradient accumulation and native SGD/Adam updates; learned parameter tensors persist across forwards/updates with snapshot invalidation (v0.2.030), Adam moments bridge through sidecar only at save/load (v0.2.028); two-step CPU parity | >225/unsupported variants host; host-canonical mirrors still download after optimizer updates and rolling stats remain host-managed; compact/f16 absent |
| Deconvolutional | native Oxide f32 forward, activation, input/weight/bias gradients (v0.2.039); actual-GPU CPU parity for Muon and two-step SGD/Adam updates (v0.2.040) | unmapped activations retain legacy host route; weights and gradients cross host boundary for optimizer; no f16 |
| Shortcut (weighted) | device input/source gradients, exact normalization derivatives | per-input/per-slot weight gradient vector crosses to host; optimizer host-side |
| Connected | device GEMM + activation, BN folded at inference | BN training and unsupported/unknown activations fall back |
| `Crop`, `BatchNorm` | device inference | training explicitly rejected |
| `MaxPool`, `LocalAvgPool` | device | `maxpool_depth` and channel-changing `local_avgpool` rejected |

## Optimizer status

| Optimizer | CPU | GPU |
|---|---|---|
| SGD (momentum, decay, per-layer rate, gates) | all parameterized layers | device for conv/MoE/EML and eligible native DCNv4; host for connected, deconv, deform_conv, recurrent, weighted shortcut, vit, swin, tucker, mamba, kimi, local, batchnorm; n/a for hard rejects |
| Adam/AdamW | all parameterized layers; DNOPT1–5 sidecars | device for conv/MoE/EML/DCNv4 (+BN scales); DCNv4 moments persist on-device between updates and materialize through the existing host sidecar on save; DNOPT1–5 readable |
| Muon | v0.1.005 hybrid on eligible 2-D matrices, DNOPT6 | **partial runtime slice (v0.2.032–052)** — connected matrices use native Oxide momentum/lookahead, Newton–Schulz and final direction+decay update, with actual-GPU multi-step, resume, two-layer, and all three update-gate parity tests. Standard non-BN convolution, local per-location, RNN, LSTM, deconvolution, ViT, Swin, TuckerAttention, MambaVision and KimiKda matrices use native Oxide orthogonalization with host-staged gradients/state/final updates and actual-GPU CPU parity. Convolution gradients transfer from device at the optimizer boundary, and updated parameters are re-uploaded to the device cache; BN convolutions remain explicitly unsupported. Connected BN has actual-device output/parameter parity through its existing host-backward/host-optimizer fallback (not native connected BN backward). Connected momentum persists on-device but is mirrored to host each update; parameters remain host-canonical. Standalone BatchNorm scales/biases use host AdamW fallback; other families remain open |

Muon eligibility (CPU, from `AGENTS.md`): convolution weights viewed as
`[out, in*kh*kw]`, `connected`, `deconvolutional`, attention/MLP projections in
`vit`/`swin`/`transformer`/`tucker_attention`/`kimi_kda`/`mambavision`,
MoE/EML expert convs, `dcnv4` projections, `laya_*` projections. Biases,
BN/norm scales, positional/embedding tables and 1-D params stay Adam/AdamW.

## f16 inference coverage

Accepted by `forward_compact_f16` (`runtime.rs:8389`):

`convolutional` (not `NORM_CHAN*`), `maxpool`, `local_avgpool`, `avgpool`,
`batchnorm`, `route`, `laya_marker_gather`, `upsample` (incl. `reverse`),
`shortcut` (with sources), `yolo`, `obb_yolo`, `gaussian_yolo`.

Everything else returns `UnsupportedLayer`. Note this is inference only: f16
training is the AMP path (`DARKNET_OXIDE_AMP=1`), which keeps f32 masters, and
f16 convolution dgrad/wgrad are implemented there.

## CPU-only gaps (Stage 1)

Only one layer port is genuinely incomplete; the rest are variants or
non-layer work.

| Item | Detail | Owner |
|---|---|---|
| `Yolo3d` | raw forward + label/config helpers + supplied-seed optimizer tests exist; projected math/assignments/loss derivatives, mixed training APIs, decode/evaluator, A2 axis pin, depth/metric consumption do not | `yolo_3d_plan.md` |
| `DsConv` multi-template `morph=0/1` | executable C++ averages views with training-time view dropout; Rust rejects it; C++ CPU backward is itself a stub | needs its own plan |
| `Region` / `Detection` | parsed as identity; C++ decode + loss unimplemented in Rust | needs its own plan |
| `Contrastive` detection branch | needs the yolo head's per-box class ids; `YoloHeadOutput` carries only the raw tensor | blocked on a head-metadata change |
| CARPN Stage-2 training | `[obb_refine]` inference only; blocks `detector train -map` for CARPN | after `carpn_network_map_plan.md` |
| Axis pins | 41 open variants (`T1`–`T4`) plus a per-backlog A2 test | `axis-coverage.md` |
| Conv `reverse` / `deform`, adversarial yolo branch | gradients missing | `todo.md` |
| `Local`, `Gru`, `Crop`, `BatchNorm` training | Rust-only extensions, partial | `todo.md` |
| EML BN batch-stat training | CPU uses rolling stats, not C++ batch statistics | `implemented.md` §7 |

## Blocked, not plannable

| Item | Reason |
|---|---|
| `Dcnv3` | forward/backward/update are stubs in all five C++ branches; only the `.weights` order is portable |
| `DetrDecoder` | C++ implementation is broken; numerics would have to come from the paper |
| wgmma (sm_90) / tcgen05 (sm_100) GEMM | cannot be compiled or tested on the sm_89 GPU here |
| Native NVFP4 `mma` | needs sm_120a/121a; sm_89 path is `__dp4a` emulation |
| Multi-GPU | hardware absent |
| Any C++ golden | no C++ build of any owning branch in this environment |

## Defects found while deriving this matrix

1. `Gru` has no GPU forward arm and no pre-flight rejection, so it surfaces as
   `InvalidConfig("... and parameter set are inconsistent")` instead of
   `UnsupportedLayer`. `Local` has native f32 forward/backward math, but its
   optimizer remains host-staged.
2. `Sam` now uses native elementwise forward/backward for supported activations;
   backward accumulates both the previous-layer and selected-source gradients.
   Unsupported activations remain on whole-network CPU fallback. Evidence:
   `sam_native_gpu_forward_backward_fans_out_source_gradient_matches_cpu_across_steps`.
3. Docs disagree about EML BN backward (v71 says CPU fallback; later notes say
   it was removed) and about MoE BN-expert SGD. Re-check `implemented.md` §7 and
   the `oxide_runtime_chain` BN tests before scheduling.
4. `STATUS.md` still says the GUI crates do not exist and that `[moe]` has no
   device-side BN path; both are stale relative to `todo.md`/`implemented.md`.

## Maintenance

- Update a row when a version lands. Cite the version in the row's evidence
  column (or in `implemented.md`) rather than duplicating prose here.
- Re-derive, do not guess: the `grep` commands in
  [How this file was derived](#how-this-file-was-derived) produce the current
  truth in seconds.
- This file is local-only (`tasks/` is in `.git/info/exclude`); commit it with
  `git add -f` if it should be tracked.
- Never write "C++ parity" for any cell. No compiled-C++ golden exists.
