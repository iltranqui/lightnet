# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Act-head contract finding](#act-head-contract-finding)
- [Act-head native gradient math and CLS scatter](#act-head-native-gradient-math-and-cls-scatter)
- [Next slice: deformable conv forward (wave 1, part 3)](#next-slice-deformable-conv-forward-wave-1-part-3)
- [Runtime integration plan for `[deform_conv]`](#runtime-integration-plan-for-deform_conv)
- [Next slice: `[dcnv4]` device path (audit result)](#next-slice-dcnv4-device-path-audit-result)
- [Verification gates](#verification-gates)
- [Remaining scope](#remaining-scope)

## Priority and boundaries

The operator explicitly requested GPU layers now, overriding CPU-first sequencing.
CPU completeness is not implied. Implement directly without subagents. Pending,
uncommitted YOLO-3D activation math is parked: do not discard or include it in GPU
commits (`darknet-core/src/lib.rs`, `model/yolo_3d_config.rs`, `yolo_3d.rs` and its
tests). Preserve unrelated user edits. One build/test at a time, max four build
threads, MemoryHigh=3G; Oxide builds/tests only through `tools/test_oxide.sh` /
cargo-oxide. No training runs. Native math, not new host fallback, is the GPU
deliverable. Sparse MoE remains unstarted.

## Resume point

Marker gather is native in f32 full/compact forward/backward and f16 compact
inference (v0.2.001–002). Act-head features/exact-erf primitives are native f32
(v0.2.003–004), and full/compact act-head f32 forward is native in v0.2.005.
Evidence/limits are in `implemented.md` §§41–45. Do not repeat these ports.
Act-head uses four uncached weight uploads per forward; no residency/speedup
claim. Runtime backward still uses CPU, act params remain frozen, f16 rejected.
Latest GPU slices are v0.2.032–037 (`2ddcaa6b`, `d73a87e7`, `f27b370c`, `cc0c00c1`, `86b2fbda`, `281cdc7c`): connected-layer `[net] muon=1` uses native Oxide momentum/lookahead, Newton–Schulz, and direction+decay updates; momentum tensors persist between steps, with host mirrors for DNOPT6 and host-canonical connected weights. Actual-GPU two-step, sidecar-resume, two-layer and `dont_update` gate CPU parity passed (§69). Deconvolutional matrix eligibility uses native Oxide orthogonalization with host-fallback gradients, momentum, final update and state; this is not native deconvolution layer support. Standalone BatchNorm remains host AdamW fallback; other eligible families explicitly reject. v0.2.031 added the standalone orthogonalization primitive (§68). Remaining GPU Muon work: extend matrix eligibility/runtime coverage and test other gate variants/connected BN on device; weights and state mirrors still cross the host boundary. Prior DCNv4 parameter tensors persist across forwards and SGD/Adam updates (v0.2.030, §67), with host snapshots detecting public edits and host-canonical mirrors refreshed after device updates. Adam moments likewise remain device-resident between updates (v0.2.028, §65). Host downloads after optimizer updates and rolling-stat host management remain; no speedup claim. v0.2.029 (`9b722d35`) first retained offset-projection weights per forward for backward (§66). v0.2.027 (`e5217f8a`) extended native DCNv4 capacity and runtime eligibility through 15x15/225 active points with GPU composition and two-step optimizer CPU parity (§64); >225 remains host fallback. v0.2.026 (`fafa95ae`) added native Revleaky mapping (§63); v0.2.025 (`7c96d45a`) added native DCNv4 Adam operations (§62), and v0.2.024 (`0087d109`) added native SGD updates (§61). v0.2.023 (`2ff4544e`) added standalone act-head MLP gradients' native full-sequence CLS-adjoint scatter (§60; runtime integration remains open). DCNv4 stores raw/offset tensors in each native `GpuForwardState` (§58) and v0.2.022 accumulates bias/offset/BN-scale gradients on-device across microbatches (§59). Next bounded slice: extend device Muon to other eligible matrices (first handle each family's device-gradient/weight lifecycle), then continue other native layer gaps. v0.2.020 (`c99f83d8`) extended native aggregation/composition/runtime through 13x13/169 active points, with two-step SGD/Adam parity and BN on/off (§57); >169 remains host fallback. v0.2.019 (`76242245`) extended through 11x11/121 points (§56). v0.2.018 (`613b243e`) extended support through 9x9/81 (§55). v0.2.017 (`8a3ef3b0`) added equivalent runtime parity at 7x7 (§54), and v0.2.016 (`a96d6d95`) expanded the kernel/runtime bound to 49 points (§53). v0.2.015 added NormalizeChannels* activations/adjoints (§52). Previous, v0.2.014: DCNv4 native rolling-stat BN affine/adjoint plus
post-backward 0.9 EMA, including unseeded layers. Eligible math (<=169 active
points, supported elementwise activation, no other fallback/host-only deform_conv)
is native with BN off/on; ignored ordinary bias/beta and update gates preserved.
Forward/gradient normalization uses the old rolling values, NOT batch statistics.
Host-cuda 37, host DCNv4 26, mandatory runtime:: 282; focused device BN 2,
runtime 3 (eight configs × two SGD/Adam steps), composition 3, aggregate 6 passed.
Details/source boundaries: `implemented.md` §51; previous runtime/layout audits
in §§49–50. Parameter gradients accumulate on-device; v0.2.024/025 apply eligible SGD/Adam updates through device operations. Since v0.2.028, Adam moments remain device-resident between steps and materialize through the host-sidecar representation only on save/load. v0.2.030 caches learned parameters across forwards and optimizer updates, refreshing host-canonical mirrors after the update; rolling statistics remain separately host-managed. v0.2.029 first retained the offset-projection weights in the per-forward cache for backward. The per-forward raw/offset cache avoids repeated projection/aggregation. No end-to-end device-only residency or speedup claim. Previous
full runtime-chain timed out at 600 s in ViT; **not a full-suite pass**.
The NormalizeChannels* activation/adjoint slice is done (v0.2.015). Successive
bounded increments extended native DCNv4 through 15x15/225 points; >225,
unsupported variants and mixed networks remain host-cached. Compact/f16 and GPU Muon remain open; DCNv4 learned tensors now persist on-device across forwards/updates, but host mirrors still download each optimizer step and rolling stats remain host-managed. Adam moments cross the host boundary only for sidecar save/load. Native deform_conv was already wired in v0.2.007; do not repeat these ports.
Recheck `version-map.tsv` before allocating another version. Earlier wave
sections below retain historical findings; this resume point takes precedence.

## Act-head contract finding

`Jev_laya@4af013d2:src-lib/laya_act_head_layer.cpp` accumulates/updates all four
weight/bias blocks, while Rust runtime drops pure helper parameter gradients and
retains only dcls. Existing module comments call weights frozen. The question
about including this training fix has not been answered; automatic continuation
is not approval to change existing training behavior. Preserve it while doing
independent native math. Do not remove CPU-backward fallback or start applying
new updates until the full required contract is resolved and implemented.

Shared-file loading is another evidence gap: tests cover in-memory blob
replacement, not a file loader. The runtime hint names load_laya_act_head_blob,
but no such core loader implementation was found. Do not accept the hint as proof.
Native forward precision inherits existing connected/GEMM settings; only explicit
Fp32 mode was validated for parity. No new global AMP/precision behavior was added.

## Act-head native gradient math and CLS scatter

Standalone act-head gradients are independent of optimizer eligibility. The
CPU helper computes all four parameter derivatives even though runtime currently
uses only the CLS adjoint.

- [x] v0.2.006: typed native MLP composition returns four positive batch-summed
  parameter gradients plus `[B,H]` `dcls`; CPU parity, actual-GPU finite
  differences, malformed-storage guards and allocator-lifetime tests are in
  `implemented.md` §46.
- [x] v0.2.023: checked backend host reference and native Oxide scatter add
  `[B,H]` into token zero of an existing NCHW `[B,H,1,T]` gradient; scorer/marker
  summary derivatives stay detached. `oxide_laya_act_backward` has 5 passing
  tests, including actual-device scatter parity and shape errors; host references
  have 2 focused tests. Evidence: `implemented.md` §60.
- [ ] Runtime backward/update integration remains separate. C++ updates all four
  act-head parameter blocks while Rust runtime drops those gradients; preserve
  current behavior until the source-training contract is resolved and do not
  remove the CPU fallback or start applying updates by inference.
- [ ] Shared-file loading remains unproven (in-memory blob replacement is not a
  file-loader test); f16 remains rejected. Native forward parity was validated
  with explicit Fp32 GEMM only. No full runtime-training or speedup claim.

## Next slice: deformable conv forward (wave 1, part 3)

Status: the shared sampling primitive and the `[deform_conv]` accumulation
stage are native and green. The host contract lives in
`darknet_cuda::host_ops::bilinear_gather` (`BilinearGatherGeometry`, `forward`,
`backward`, trait defaults `deform_bilinear_gather_f32` / `_backward_f32`); the
device kernels live in `src/darknet-cuda-oxide/src/kernels/deform_sample.rs`.
`todo.md` holds the test counts and the atomic-scatter caveat. **No layer calls
any of it yet.** This section is the contract for the first wiring.

Status: the accumulation stage is native and green —
`deform_conv_accumulate` (kernel) + `OxideBackend::deform_conv_accumulate`
(checked launcher), 5 device tests in `tests/oxide_deform_conv_accumulate.rs`
with an independently written reference, plus the earlier `oxide_deform_gather`
(5) and host contract (18 host tests). Remaining before the runtime can select
the device path, in dependency order: (1)+(2) predictor columns and the
offset/mask predictors — **done as `deform_offset_predict`**, which is
`conv2d_f32` (`groups = 1`, source weight order `[filters][channels][ky][kx]`) +
`bias_add_f32`; it **rejects `dilation != 1` explicitly** instead of returning a
mis-shaped tensor (see the finding below); (3) one elementwise kernel for the per-batch conditional
normalization (`stddev > 1e-12 && (stddev > max_offset || |mean| > max_offset)`),
`sanitize`, `centered_lhtan` and the mask sigmoid — **done**:
`deform_offset_envelope` (caller supplies per-batch `mean`/`inv_std`/`normalize`)
and `deform_mask_sigmoid`, verified by `oxide_deform_envelope` (7 passed); the
per-batch mean/stddev reduction is now native too (`deform_offset_stats`, one
thread per batch, f64 accumulation matching the source, serial by design);
(4) this
accumulation kernel; (5) the
existing conv epilogue (bias/BN/activation).

**Composition landed:** `OxideBackend::deform_conv_forward_native` chains all five
stages and returns the raw output (before the layer's own bias/BN/activation),
verified by two composed device tests at `TOL_COMPOSED = 1e-4` (GEMM summation
order in the predictor vs the reference loop; single-stage tests stay at
`2e-6`). Still deliberately unwired: the composed oracle is a stage-by-stage
host reference. **True oracle reached:** `oxide_deform_conv_chain` builds a real
`ExecutableNetwork` from inline cfg text, runs its CPU forward with the
runtime's own initialized parameters, and matches the native composition (3
passed: masked, unmasked, and the dilated rejection). Remaining: add the runtime
forward arm (keeping `DeformConv` in `requires_cpu_backward_fallback` and
falling back to the host for `dilation > 1`), then update `layer-support.md` to
`gpu fwd native / gpu bwd cpu`.

### Blocking finding: a native forward alone cannot serve `[deform_conv]`

The CPU backward is the reference and it reads `DeformConvRuntimeCache`, which is
produced **by the host forward** (`forward_deform_conv_with_cache`).
`GpuForwardState.cpu_forward` (`runtime.rs:13174`) is assembled from the GPU
loop's own host artifacts, not from a separate CPU pass, so there is no other
producer of that cache. Two consequences:

1. Swapping the arm to a device forward removes the cache and breaks the
   fallback backward (empty or missing cache) — a silent correctness hole, not a
   compile error.
2. Recomputing the host forward just to fill the cache would keep the cost the
   device path is meant to remove, so it buys nothing.

Therefore the wave order must change: **native backward (or a device cache
producer) comes before the forward wiring**, for `[deform_conv]` and for
`[dcnv4]` (same structure). The five forward stages stay as tested, unused
primitives until then; the trait methods are now in place so the wiring is
mechanical once the backward exists. Do not flip the arm on forward-only
reasoning.

**Audit result: a cache producer is not cheaper than the backward.**
`deform_conv_backward` reads 27 cache fields — every intermediate *and* its
derivative bookkeeping: `raw_output_slope`/`_nondiff_boundary`, `offsets`,
`input`, `input_slope`/`_nondiff_boundary`, `offset_lhtan_slope`/`_boundary`,
`offset_map_slope`, `offset_stage2_slope`/`_nondiff_boundary`,
`offset_normalization_boundary`, `offset_normalized`, `offset_mean`,
`offset_stddev`, `offset_stage1`, `offset_stage1_slope`/`_nondiff_boundary`,
`columns`, `masks`. Producing those on device *is* the input-side half of the
backward, so there is no cheap subset. A forward-only device path for
`[deform_conv]` is therefore not viable in any form.

**Strategic consequence for the whole wave.** Every layer in
`requires_cpu_backward_fallback` builds its cache inside its own host forward arm
(`cpu_forward` is assembled from GPU-loop artifacts, `runtime.rs:13174`), so the
same wall applies to `dcnv4`, `vit`, `swin`, `tucker`, `mambavision`, `kimi_kda`
and `laya_act_head`: a native forward cannot be wired without a native backward.
The "native forward first" staging used for the deformable family is wrong for
this repo; pick targets by backward tractability instead.

### Finding: the runtime arm needs trait methods, not inherent ones

`GpuExecutableNetwork<B: F32TensorOpsBackend>` is generic over the backend, so a
`runtime.rs` forward arm cannot call an `OxideBackend`-inherent method such as
`deform_conv_forward_native`. Wiring the device path therefore requires the five
stages to be **trait** methods with host-reference defaults first. Those defaults
now exist (`darknet_cuda::host_ops::deform_conv`: `sanitize`, `centered_lhtan`,
`stable_sigmoid`, `OffsetStats`/`offset_stats`, `offset_envelope`,
`mask_sigmoid`, `DeformGeometry`/`accumulate`, 6 unit tests; `bilinear_gather`
exposes `sample_at` so the gather and the accumulation share one sampler). The
next slice is mechanical but broad: trait declarations + default bodies + Oxide
overrides (already written, just need forwarding), then composition in
`runtime.rs`. Do not try to special-case the Oxide backend in generic code.

### Finding: predictor columns cannot simply reuse `conv2d_f32`

`deform_conv::output_dim` includes the dilated kernel extent:
`oh = (H + 2*pad - dilation*(size-1) - 1) / stride + 1`. The oxide conv `Geom`
(`src/darknet-cuda-oxide/src/conv.rs:145`) intentionally does **not** — it uses
`(H + 2*pad - size) / stride + 1`, matching `[convolutional]` (see
`axis-coverage.md` finding F1). For `dilation == 1` the two agree, so the offset
and mask predictors can be built with `conv2d_f32` + `bias_add_f32`. For
`dilation > 1` they do not: the conv would return a differently shaped tensor,
so the predictor columns need their own im2col launch (or a conv variant that
takes an explicit output extent). Do not "fix" `Geom` to include dilation —
that would break `[convolutional]` and `cfg/resnet152_trident.cfg`; select the
extent explicitly at the call site instead.

Caught by a fixture assertion in `oxide_deform_conv_accumulate`: the original
strided+dilated case used a shape `deform_conv` would never produce (4x5 in,
size 3, stride 2, pad 1, dilation 2 => source extent 1x1, not 2x2). The kernel
takes `out_h`/`out_w` as given, so it still passed; the test now derives and
asserts the extent before comparing.

Scope: `[deform_conv]` **forward only** (device), CPU backward unchanged. Do not
touch `requires_cpu_backward_fallback`, do not remove the host round trip until
the backward is native, and do not claim the layer is GPU-complete — the whole
point of this slice is that a native forward alone still de-optimises a network.

Operands and math, from `src/darknet-core/src/deform_conv.rs` (hand-derived from
`Jev_laya`): output `[B, filters, OH, OW]`; for each output point the base
sample is the input grid position derived from `stride`/`padding`/`dilation`,
displaced by the predicted offsets `groups * size^2 * 2` and scaled by the
optional sigmoid masks `groups * size^2`; offsets pass through the source
conditional per-image normalization and the `centered_lhtan`/`stable_sigmoid`
clamps; the sampled value is bilinear with zero padding and the far-outside
early-out; `out += weight[f, c, ky, kx] * sample`, then the existing conv
epilogue (bias, BN fold or batch stats, activation).

Kernel shape: one thread per `(b, f, oy, ox)` that loops the kernel points and
the group's input channels, calling a shared `#[device]` sampler extracted from
`deform_sample.rs` — do **not** materialise `[B,S,C,OH,OW]` columns for this
layer (that is `f32 * B*S*C*OH*OW`); the column primitive stays available for
layers that need it. Reuse the existing epilogue kernels; do not add a second
activation implementation. Guard every address with the same in-bounds checks
as the CPU reference and keep the `deform_` device-symbol prefix (PTX symbols
are global).

Test gates, in this order:

1. New `tests/oxide_deform_conv.rs`: parity against `deform_conv_forward` on a
tiny real cfg (a `[deform_conv]` section with `groups > 1`, `size=3`,
stride/dilation/padding variants, masks on and off, `modifier` on and off) plus
a non-square `W != H` case; tolerance comparable to the conv suites, and an
explicit assertion that the far-edge samples agree (that is where the early-out
lives).
2. `MEM=3G ./tools/test_oxide.sh --test oxide_deform_gather` (regression).
3. `MEM=3G ./tools/test_host.sh deform_conv` (23 core tests today; must not
shrink).
4. `MEM=3G ./tools/test_host.sh runtime::` if the runtime arm changed at all.

Definition of done for this slice: device forward selected for `[deform_conv]`
networks, CPU backward still selected, every host and device test above passing,
`layer-support.md` updated to `gpu fwd native / gpu bwd cpu`, and one line in
`todo.md` with the counts. No compiled-C++ golden exists, so write
"device vs Rust CPU reference", never "matches C++".

## Runtime integration plan for `[deform_conv]`

The device forward and backward both exist and are verified (`todo.md` holds the
counts). What remains is the runtime wiring, and it has one condition that the
earlier "flip the arm" plan missed.

**The cache is the hinge.** `DeformConvRuntimeCache` is only produced by the host
forward (`runtime.rs:2755`) and only consumed by the CPU backward
(`runtime.rs:4103`); its length is validated at `3555`. A device forward that
stops calling the host forward starves that path. So the arm must be
conditional, not replaced:

- **Fallback active** (any layer in `requires_cpu_backward_fallback`, including
  `DeformConv` itself while it stays listed): keep the host round trip and the
  cache exactly as today. No behaviour change.
- **Whole network device-trainable**: compose the forward from the trait
  primitives (`deform_sanitize_f32`, `conv2d_f32` + `bias_add_f32` for the
  predictors, `deform_offset_stats`, `deform_offset_envelope`,
  `deform_mask_sigmoid`, `deform_conv_accumulate_f32`), keep the intermediates as
  device tensors, and call `deform_conv_backward_native` from the backward arm.
  No host cache is produced or needed.

**Ordering that avoids a broken intermediate state:**

1. Add the device backward arm for the device-trainable case, gated on
   `!requires_cpu_backward_fallback()` **after** removing `DeformConv` from that
   list — but keep the mixed case working by computing the host cache only when
   `host_copies` is true.
2. Gradients must land in the existing `LayerTrainBuffers::DeformConv` fields
   (`runtime.rs:6441` accumulates them today) so optimizer code at `7576` and
   `10065` keeps working unchanged.
3. Then remove `DeformConv` from the GPU-optimizer rejection list (`10065`) only
   after a two-step weight-parity test passes.

**Tests, in this order:** (a) device backward vs CPU gradients on real params
(already green, `oxide_deform_conv_backward_chain`); (b) a new
`oxide_runtime_chain` case running two training steps on a tiny `[deform_conv]`
network and comparing updated weights against CPU; (c) `test_host.sh deform_conv`
(23 tests, must not shrink) and `test_host.sh runtime::` because the
fallback/optimizer arms changed; (d) only then flip `layer-support.md` to
`gpu fwd native / gpu bwd native (normalization host hop)`.

**Do not** wire the forward alone, and do not remove `DeformConv` from the
fallback list before the device backward arm exists for the mixed case — either
mistake produces a network that trains silently wrong rather than failing.

## Next slice: `[dcnv4]` device path (audit result)

`[deform_conv]` is wired (`layer-support.md`), so the question for `[dcnv4]` is
whether the same template applies. The blocking factor for deform_conv was its
27-field forward cache; **dcnv4's cache has two fields**
(`src/darknet-core/src/dcnv4.rs:11`):

```text
Dcnv4ForwardCache { offsets: Vec<f32>, raw_output: Vec<f32> }
```

and `dcnv4_backward` reads only those (`:580`, `:607`). So no cache producer is
needed: `offsets` is the offset-predictor output and `raw_output` is the
aggregation output, both of which the device forward would already hold.

**What is genuinely new** (cannot be reused from deform_conv):

- the aggregation differs — `d_stride` coarse offsets with interpolation,
  grouped channel partitions, optional softmax-normalised masks and centre
  removal (`src-lib/dcnv4_layer.cpp::dcnv4_deformable_forward_cpu`), so the
  `deform_conv_accumulate` kernel does not apply;
- the offset predictor is a projection with `d_stride` reduction rather than a
  same-resolution conv, so the predictor-backward reuse is partial.

**Slice order, mirroring what worked twice already:** (1) host reference for the
aggregation and its adjoint in `darknet_cuda::host_ops` with finite-difference
tests; (2) device kernel for aggregation + adjoint, parity against (1);
(3) compose the full forward/backward and compare against the CPU
`dcnv4_backward` gradients on a real cfg (the `oxide_deform_conv_backward_chain`
pattern); (4) only then remove `Dcnv4` from `requires_cpu_backward_fallback` and
extend the shared path predicate — the current exact fallback path after this
change is `Dcnv4` in `requires_cpu_backward_fallback` plus the deform-only gate,
so nothing about dcnv4 should be touched until step (3) is green.

Do not assume deform_conv's accumulated evidence transfers: the sampling
geometry, mask semantics and offset layout are different code paths in the
source.

**First host-reference attempt failed its own FD check (negative result), and
the cause is now identified as a fixture defect, not a math error.** The
aggregation was re-derived from `dcnv4_forward`; the analytics disagreed with
finite differences on `d_offsets` (analytic `0` where the FD said `-0.0080`, at
a mid-kernel point with `softmax` and `remove_center` both off). After re-reading
`dcnv4_backward` and comparing term by term, the adjoint matched the source —
`dx = hy*(v01-v00) + ly*(v11-v10)`, `dy = hx*(v10-v00) + lx*(v11-v01)`, the packed
`[2K offsets, K mask logits]` layout, the all-`K` mask write and the softmax
`p*(d - dot)` transform all agree. What differed was the **fixture**: offsets
built as `((i*5) % 11) * 0.03 - 0.15` include **exactly zero** values, and the
base position is integral, so those samples land precisely on the floor-cell
seam where the analytic slope is one-sided. Same defect class as the four earlier
fixture failures (`lessons.md`).

So the module was removed for the right reason (no failing test in the tree) but
the diagnosis was wrong: this is not evidence that a forward-only derivation
fails. **Second attempt (direct transcription, offsets shifted +0.37) still
fails FD**, so the seam hypothesis is disproven as the sole cause: with
`softmax=false, remove_center=false` the analytic `d_offsets[14]` is `0.0637`
versus an FD of `0.0499` — same sign, ~1.28x, at an x-offset of kernel point 7.
Both attempts were removed and the green baseline re-verified (33 passed).

That pattern (analytic consistently larger, no seam, no zero) points at a term
counted more than once rather than a sign or convention error.

**Instrumented term-level check (done, isolated):** a standalone program computed
the analytic `d_offset_x` for one position/point from `hy*(v01-v00) +
ly*(v11-v10)`, summed over the two channels, and compared it with a central
difference of the same expression: `analytic 0.093750 vs fd 0.093751`,
**ratio 1.0000**. So the gradient *formula* per term is right; the discrepancy in
the aggregate implementation is therefore an **indexing/bookkeeping** error —
which row/point/channel a contribution is written to — not arithmetic.

**Resolved by isolated replication (not an index bug).** A standalone program
containing the same forward and backward logic (all positions, all points, both
channels, mask and input scatter included) was run with a **cell-centre fixture**
(offsets ≈ 0.5…0.58, target offset chosen so `sx = 1.5`, fractional part 0.5):

```text
sx=1.5 frac=0.5
analytic=0.093750 fd=0.093758 ratio=0.9999
```

So the arithmetic **and** the write mapping are correct; the earlier ~1.28× came
from the fixture, whose target sample sat at `sx = 1.005` and was pushed across
the integer cell boundary at `1.0` by the ±0.0075 FD perturbation — a
discontinuity of the *comparison*, not of the code. This is the sixth instance of
the fixture-seam class in `lessons.md`, and the first one resolved by isolation
instead of by rewriting.

**Landed:** `host_ops::dcnv4` (`aggregate` / `aggregate_backward`, plus
`Dcnv4Geometry` and `group_masks`) is in the tree as v0.2.009 (`e3cfb27d`) and
FD-green across all four `softmax`/`remove_center` combinations (`cargo test -p
darknet-cuda --lib` 35 passed) using the cell-centre fixture. Native aggregation
forward (v0.2.010) and adjoint (v0.2.011) now pass host parity and actual-device
finite differences; 6 device tests cover batches/groups, masks/center removal,
strided/dilated/edge geometry and malformed operands. Still not wired.
v0.2.012 now implements native full raw composition, with real CPU output/input/
parameter-gradient checks and device finite differences. It also corrects the
shared old aggregation layout (K=size²-remove_center, padded row width). This
supersedes the stage-only parity claims above; audit in `implemented.md` §49.
v0.2.013 wires eligible f32 runtime math and verifies two-step SGD/Adam parity;
parameter gradients/updates remain host-side and offsets recompute on device.
`requires_cpu_backward_fallback` is conditional for DCNv4: >25 active points
or unsupported activation still force the host path. v0.2.014 now supports
rolling-stat BN affine/adjoint and post-backward 0.9 EMA; audit §51. Next:
NormalizeChannels* activation/adjoint composition; retain all other limits.

**Concrete fixture numbers for that attempt** (recomputed from the failing case,
`softmax=false, remove_center=false`, `size=3, pad=1, scale=0.75`, output cell
`q=0`, point 7 = `(kw=2, kh=1)`): the fixture offsets are
`0.37 + ((i*5)%11)*0.03 - 0.15`, so `offsets[14]=0.34` and `offsets[15]=0.49`.
With `p0_scaled = -0.75` that gives `sx = 1.005` and `sy = 0.3675`. The FD
perturbs `sx` by `±0.0075`, i.e. it crosses the integer cell boundary at
`sx = 1.0` (stencil `{0,1}` vs `{1,2}`, both in-bounds, so the interpolant should
be C1 there — but this is exactly the kind of near-boundary fixture that has
fooled five earlier comparisons, and it must be excluded before blaming the
code). Shift the fixture so `sx` lands near a cell *centre* (fractional part
≈ 0.5) for every point, then re-run the FD before any index work.

## Verification gates

Use test-first, serial test threads, script `-j 2` and the 3G cap. Add focused real
GPU gradient-math tests; existing related regressions are:

```
MEM=3G ./tools/test_host.sh laya_act_head
MEM=3G ./tools/test_host.sh runtime::
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_forward
MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features
```

Targeted Rustfmt/diff checks, requirement-to-artifact audit, implemented/STATUS/
todo/version-map updates and disposable-artifact cleanup gate each release.
Distinguish native forward/gradient primitives from complete native layers,
selected tests from full-suite coverage and Rust CPU agreement from C++ parity.
No performance/compiled-C++ claim without actual measurements/golden evidence.

## Remaining scope

Act-head runtime backward/update/file loading and f16, other LAYA/transformer
layers, DCNv4 Adam and excluded-variant host fallbacks, deform/deconvolution
host fallbacks, CPU-only layer GPU paths and GPU Muon remain open. Each layer needs device forward/backward/update tests as
applicable and explicit limits. CPU/config/axis gaps remain open. No green full-
host/corpus/full-Oxide suite is established; sparse MoE stays unstarted.
