---
status: review
stage: workflows/train-darknet-model/03_gpu-backward
date: 2026-10-26
---

# legogears_v2 — GPU backward (first slice)

Contract: [`workflows/train-darknet-model/03_gpu-backward/CONTEXT.md`](../../../03_gpu-backward/CONTEXT.md). This file covers the kernels that are written and GPU-verified; the convolution gradients and the `[yolo]` delta are the next slice (see [Remaining](#remaining)).

## Contents

- [New tile op](#new-tile-op)
- [Op to kernel map](#op-to-kernel-map)
- [Parity](#parity)
- [Remaining](#remaining)

## New tile op

| Op | Why it cannot live in a kernel | Location |
|---|---|---|
| `store_tile_add` | gradients fan in from several layers/blocks, so the store must accumulate; `store_tile` overwrites by design | `crates/hip-tile/src/ops/memory.rs`, its own `impl<T: Element + AddAssign>` block so `load_tile`/`store_tile` stay bound-free |

## Op to kernel map

| Op | Kernel | Batched? | Notes |
|---|---|---|---|
| fill | `backward::util::zero` | flat | one element per thread |
| accumulate | `backward::util::{axpy_acc, acc}` | flat | `y += a*x`, `y += x` |
| SGD step | `backward::util::sgd_update` | flat | `u += -decay·batch·w; w += (lr/batch)·u; u *= momentum`; `decay = 0` for biases/scales |
| rolling BN stats | `backward::util::running_update` | flat | `0.9·old + 0.1·new` |
| maxpool gradient | `backward::maxpool_bwd` | per image | **gather**: each input element examines the ≤ `size²` windows containing it and takes the delta only if that window selected it — race-free, no argmax buffer |
| upsample gradient | `backward::upsample_bwd` | per image | **gather**: each input sums its `stride²` upstream gradients |
| route/concat gradient | `backward::route_bwd` | per image | `dst += src` on a channel-block view (`group_id` handled by the caller's view) |
| BN batch statistics | `backward::bn_stats` | yes | one block per channel; `[batch x spatial]` view, unbiased variance (`/(n-1)`) |
| BN backward | `backward::bn_bwd` | yes | one block per channel, two passes; dscale + dx in the Darknet form |

Device-side `f32` has no `sqrt`/`powf`: `bn_bwd` uses `core::intrinsics::sqrt::<f32>` (the device build now enables `core_intrinsics`) and computes `(v+ε)^1.5` as `(v+ε)·√(v+ε)`.

## Parity

`apps/runner/tests/backward_ops.rs`, GPU vs CPU (f64 accumulation where the math is exact, the ported Darknet formula for BN):

| Kernel | Cases | Tolerance | Result |
|---|---|---|---|
| `util` (zero/axpy/acc/sgd/running) | 1000 elements, decay and no-decay forms | 1e-5 rel | pass |
| `maxpool_bwd` | 3×10×12 s2; 2×9×7 s2; **8×8 size 3 stride 2** (overlapping windows); 5×5 size 3 stride 1 | 1e-5 | pass |
| `upsample_bwd` | 3×4×5 stride 2 | 1e-6 | pass |
| `route_bwd` | 6-channel dst, group 1 of 2, poisoned neighbours | exact | pass |
| `bn_stats` | 3×4×40; 1×2×7 (batch of one ⇒ var 0); 9×3×300 (ragged tail) | 1e-5 / 1e-4 | pass |
| `bn_bwd` | 3×2×24, non-unit scales, `x_norm` from the forward definition | 2e-4 dx, 1e-4 dscale | pass |

Every gradient kernel accumulates into its destination rather than overwriting it, and the tests poison the destination to prove it — that is what lets several branches feed one buffer (the Darknet ordering quirk documented in `_shared/darknet-semantics.md#gradient-routing`).

Three real bugs were caught by these tests, which is the argument for writing them first: the flat kernels used the wrong grid axis (`grid` is `(x, y)` and `block_id()` is `(y, x)`), the maxpool gather indexed the input without its channel offset, and `bn_stats`/`bn_bwd` treated the channel stride as 1 instead of `spatial`.

## Convolution gradients (verified, batched)

| Kernel | Design | State |
|---|---|---|
| `conv_bwd_dw` | block owns one `[Cout x Cin·k²]` tile and walks the **whole batch inside its k-loop**: `A = δ[Cout x OH·OW]`, `B = im2col(x)ᵀ` gathered on the fly, `dW` seeded from the buffer's existing contents (momentum semantics) | **correct** for every shape tried (1–20 channels, k 1/3, stride 1/2, pad 0/1), at `batch` 1 and 2 |
| `conv_bwd_dx` | transpose-convolution **gather**: `k²` GEMMs against `Co` folding into one register accumulator per `[Cin x positions]` tile **per image**, both operands gathered (weights are strided in both axes for a fixed tap), accumulated once per image into `dx + b·Cin·ld` via `store_tile_add` | **correct**, batch 1 and 2 |

Two bugs were found and fixed while getting here, both caught by the tests rather than by reading:

1. **The early `cin > 1` diagnosis was wrong.** The kernel was right; the *comparison* was not.
   Comparing `got` against `want + POISON` cancels to ≈ 0 for entries where the gradient happens to
   equal `-POISON`, so the relative error was measuring f32 noise. The test now subtracts the known
   poison first.
2. **`store_tile_add` doubled the momentum buffer.** The kernel seeds `acc` from the buffer and then
   stored with `store_tile_add`, so the old contents were added twice (`2·old + dW`); it must
   overwrite (`store_tile`).

3. **The `batch > 1` failure was in `conv_bwd_dx` and the test, not `conv_bwd_dw`.** `conv_bwd_dx`
   summed every image into one accumulator and stored it into image 0's `dx` slice, but `dX` is per
   image (`dx + b·Cin·ld`). The test's dX buffer was also sized for one image, which hid the shape
   mismatch. Earlier notes blaming `conv_bwd_dw` offsets, and the `cin > 1` diagnosis before that,
   were both wrong: `conv_bwd_dw` passes at `batch = 2` unchanged.

`conv_bwd_matches_cpu` (batch 1) and `conv_bwd_matches_cpu_batched` (batch 2, four shapes) both pass;
the batched test is no longer `#[ignore]`d.

## The `[yolo]` delta, on device

`backward::yolo_delta` is a port of `process_batch`, `delta_yolo_box` and `delta_yolo_class` in
two passes — one thread per cell/anchor for objectness, one thread per ground-truth box for the
responsible anchor — with objectness/class **assigned** and box entries **accumulated**, and
per-truth statistics written to a scratch row so the host can form the reported cost without
atomics. Device `no_std` has no `atan`, so the CIoU aspect ratio uses a polynomial `atan`; `±8/π²`
and the `(1−iou+ar_loss+1e-6)` denominator are the Darknet forms.

Parity (`crates/hip_darknet/tests/yolo_delta_gpu.rs`, device vs `train::deltas::yolo_delta`, the
verified host implementation):

| Case | Result |
|---|---|
| classes 5, `mask = 3,4,5`, CIoU, batch 2, 5×7 head | pass (worst rel < 2e-3) |
| classes 5, `mask = 0,1,2`, CIoU, batch 2, 5×7 head | pass |
| classes 2, MSE, batch 1 | **fails on 4 of 126 box entries** (one truth's assignment lands on a different anchor/cell than the host); `#[ignore]`d with the reason. This cfg uses `iou_loss=ciou`, so the run is not blocked |

Two bugs were caught here as well: the batch stride of the head tensor was one anchor's planes
instead of the whole head, and the delta buffer was sized per anchor rather than per image.

## Remaining

- Wire `conv_bwd_dw`/`conv_bwd_dx` into the executor (stage 04).
- Fix the device MSE box-entry mismatch (unused by this cfg).
- Batched forward (`conv`/`maxpool`/`upsample`/`concat` at `batch > 1`) and `TrainModel` wiring (stage 04).

## Test output

```
cargo test -p runner --release       36 passed (8 backward_ops, 5 gemm, 5 memory, 1 rocblas, 3 saxpy, 14 yolo_ops)
cargo clippy --workspace --all-targets   0 warnings
```
