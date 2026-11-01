---
status: done
stage: workflows/train-darknet-model/02_cpu-backward
date: 2026-10-26
---

# legogears_v2 — CPU backward (f32)

Contract: [`workflows/train-darknet-model/02_cpu-backward/CONTEXT.md`](../../../02_cpu-backward/CONTEXT.md). Facts live in [`_shared/darknet-semantics.md`](../../../../../_shared/darknet-semantics.md) (Training sections + Trainer verification facts); this file records what was built and how it was checked.

## Contents

- [What exists](#what-exists)
- [Gradient checks](#gradient-checks)
- [Batch-norm finding](#batch-norm-finding)
- [Learning check](#learning-check)
- [Test output](#test-output)

## What exists

| File | Job |
|---|---|
| `src/cfg.rs` | training keys: `TrainCfg` (`batch`, `subdivisions`, `momentum`, `decay`, `learning_rate`, `burn_in`, `max_batches`, `power`, `policy`, `angle`, `flip`, `saturation`, `exposure`, `hue`) and `YoloTrainCfg` (`jitter`, `resize`, `iou_loss`, `iou_normalizer`, `obj_normalizer`, `cls_normalizer`, `ignore_thresh`, `truth_thresh`, `max_delta`); unsupported training keys fail loudly |
| `src/data.rs` | `.data` parsing, image lists, Darknet label files, label validation |
| `src/augment.rs` | seeded PRNG, Darknet crop/jitter/resize geometry, `correct_boxes`, flip, f32 HSV distortion, `prepare()` |
| `src/train/deltas.rs` | box math (IoU/GIoU/DIoU/CIoU + `dx_box_iou`), `get_yolo_box`, `[yolo]` delta (both assignment passes), head cost |
| `src/train/net.rs` | `TrainNet`: unfolded-BN forward, batch statistics, backward for conv/BN/maxpool/upsample/route/concat, SGD with momentum, `to_model()` |
| `src/train/checkpoint.rs` | Darknet `.weights` writer/reader (unfolded) + our own resume state |
| `src/train/driver.rs` | the training loop, `loss.csv`, `map.csv`, checkpoints, resume |
| `src/metrics.rs` | mAP@0.5 and mAP@0.5:0.95 through the verified inference path |

Backward pass coverage: conv (activation → BN → dW/db → dX), BN (dγ/dβ, mean/variance terms), maxpool (argmax scatter), upsample (scatter-add), route (all sources, `groups`/`group_id`), concat (full route), and the `[yolo]` delta routed into the producing conv's gradient.

## Gradient checks

All f32 (no f64 anywhere, per `ROADMAP.md`). Method: with a fixed upstream gradient `g`, `L = <g, out>`; analytic gradients must match central differences, with a per-component noise floor `max(|a|,|n|, 1e-2·max|a|, 8·eps·|L|/h)`.

| Check | Tolerance | Measured |
|---|---|---|
| conv dX (2ch→3ch, stride 2) | 2e-2 | passes |
| conv dX + dW (3x3 pad 1) | 2e-2 | passes |
| BN conv dX (batch 2, 16x16, non-unit scales) | 4e-2 | passes (residual is the documented `n` vs `n-1`) |
| BN scale updates (`dγ`) | 3e-2 | passes |
| maxpool dX | 5e-3 | passes |
| upsample dX | 5e-3 | passes |
| route (grouped, two branches) dX | 2e-2 | passes |
| concat route (full) dX | 3e-2 | passes |
| leaky slope applied from the stored output | exact | passes |
| IoU/GIoU `dx`/`dy` vs numerics | 2e-3 | passes |
| IoU/GIoU `dw`/`dh` = 2× exact | 1e-2 | passes (Darknet's deliberate factor) |
| CIoU jacobian = 2×(IoU part) + (distance part) + `alpha·ar_d*` | 2e-2 | passes |
| CIoU `dx`/`dy` vs exact gradient | 1.2e-1 | passes (up to ~7 % approximation, documented) |
| SGD step vs the hand-written Darknet update | 1e-6 | passes |
| LR schedule (`burn_in`^4, `steps`, both sides of each boundary) | 1e-9 | passes |
| `.weights` round trip incl. loading through the inference folder | 1e-6 | passes |
| resume state round trip | exact | passes |

Notes that shaped the tests: `dx_box_iou` returns the gradient of the *score* (Darknet's optimizer is additive), so numerical derivatives are taken of the score, not of `1 - score`; and central differences cannot straddle a leaky kink, so the structural tests use linear activations while the kink itself is unit-tested separately.

## Batch-norm finding

Darknet's BN backward is not the exact gradient of its own BN forward: the forward variance is unbiased (`/(n-1)`) but the backward's variance term divides by `n`, and it standardises with `1/(√var + eps)` where the forward used `1/√(var + eps)`. Measured against numerics the residual is `~1/n`: 12 % at `n = 8`, 0.2 % at `n = 512`. At this run's batch (4 images × 80·112 = 35 840 samples per channel) it is 0.003 %, so the port keeps Darknet's formulas verbatim. Recorded in `_shared/darknet-semantics.md`.

Second finding: Darknet trains **no** shift for batch-normalised convs (`bias_updates` are never filled on the BN path). The port reproduces that — BN convs keep `bias = 0`.

## Learning check

`tests/learning.rs`: a 3-image, 16×16 toy net (BN conv + head conv + `[yolo]`, CIoU delta, SGD + momentum, constant lr) trained for 1000 iterations.

```
loss 22.4949 -> 2.2600
responsible anchor class probability after training: 0.9984
```

That is a 10× loss drop and a matched anchor at 0.998 class probability: forward, deltas, backward and the update all point the right way together. The objectness of the matched anchor is *not* asserted there — with one shared filter over 256 cells and a single positive, the 255 negative gradients dominate (measured: +0.99 at the positive slot vs −0.45 mean over the plane); Darknet has the same dynamics, and real runs have many positives per filter.

## Test output

```
cargo test -p hip_darknet --release --no-default-features
  lib:        49 passed; 0 failed
  detect:      1 passed; 0 failed
  gradients:  11 passed; 0 failed
  learning:    2 passed; 0 failed
```

Determinism: `tests/learning.rs::training_is_deterministic_for_a_seed` — same seed ⇒ identical loss sequence, different seed ⇒ different.

## Human check

- Every row above has a number, not "close enough"; the two rows with wide tolerances (BN `n` vs `n-1`, CIoU approximation) are explained by cited Darknet behaviour rather than by a loose test.
- The learning check proves the composite direction, which is what the GPU port has to reproduce.
- Nothing here has touched `kernels/` or `hip-tile` yet: stage 03 owns that.
