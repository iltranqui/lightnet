---
status: review
stage: workflows/train-darknet-model/01_train-semantics
date: 2026-10-26
---

# legogears_v2 — training semantics

Contract: [`workflows/train-darknet-model/01_train-semantics/CONTEXT.md`](../../01_train-semantics/CONTEXT.md). Facts live in [`_shared/darknet-semantics.md`](../../../../_shared/darknet-semantics.md) (Training sections); this file records only what is specific to this run.

## Contents

- [Paths](#paths)
- [Key inventory](#key-inventory)
- [Loss scaling](#loss-scaling)
- [Run decisions](#run-decisions)
- [Accepted deviations](#accepted-deviations)
- [Reference history](#reference-history)

## Paths

- cfg: `~/src/DarkSuite/datasets/LegoGears_v2/LegoGears_v2.cfg`
- data: `~/src/DarkSuite/datasets/LegoGears_v2/LegoGears_v2.data` (classes = 5)
- names: `~/src/DarkSuite/datasets/LegoGears_v2/LegoGears.names` (`red light, pin, center, small gear, medium gear`)
- train list: `LegoGears_v2_train.txt` (80 images) → `darkmark_image_cache/resize/*.png` + `*.txt` labels
- valid list: `LegoGears_v2_valid.txt` (80 images), same cache
- reference history: `LegoGears_v2_loss.csv`, `LegoGears_v2_map.csv`

The lists already point at images pre-resized to the network input (224×160), so the resize path is a no-op for this run and augmentation works on cache-sized images with labels in normalized coordinates.

## Key inventory

`[net]`:

| Key | Value | This run |
|---|---|---|
| `batch` / `subdivisions` | 16 / 1 | override to 4 / 1 (see Run decisions) |
| `width` / `height` / `channels` | 224 / 160 / 3 | implement |
| `momentum` | 0.9 | implement (SGD momentum) |
| `decay` | 0.0005 | implement, weights only |
| `learning_rate` | 0.002610 | implement |
| `burn_in` | 1000 | implement, with `power = 4` (default) |
| `max_batches` | 3000 | override to 12000 |
| `policy` / `steps` / `scales` | steps / 2400,2700 / 0.2,0.1 | override steps to 9600,10800 |
| `angle` | 0 | reject non-zero (rotated crop is out of scope) |
| `flip` | 0 | implement, off by default; reject non-zero |
| `saturation` / `exposure` / `hue` | 1.5 / 1.5 / 0.1 | implement |
| `mosaic` / `mixup` / `cutmix` | 0 | reject non-zero |
| `max_chart_loss` | 4.0 | ignore (DarkMark chart aid) |
| `use_cuda_graph` | 0 | ignore |
| `weights_reject_freq`, `ema_alpha`, `equidistant_point`, `num_sigmas_reject_badlabels`, `badlabels_rejection_percentage` | commented out | ignore |
| optimizer | absent | SGD + momentum only, by decision |

`[yolo]` (both heads, identical):

| Key | Value | This run |
|---|---|---|
| `mask` / `anchors` / `num` | `3,4,5` / `0,1` / `0,1,2`; six anchors | implement (already parsed for inference) |
| `classes` | 5 | implement |
| `jitter` | 0.3 | implement (crop jitter) |
| `resize` | 1.5 | implement (crop resize range) |
| `ignore_thresh` / `truth_thresh` | 0.7 / 1.0 | implement |
| `iou_loss` / `iou_normalizer` | ciou / 0.07 | implement |
| `cls_normalizer` / `obj_normalizer` | 1.0 / default 1.0 (absent) | implement |
| `max_delta` | absent (FLT_MAX) | implement as "no clipping" |
| `objectness_smooth`, `label_smooth_eps`, `focal_loss`, `classes_multipliers`, `new_coords`, `map` | absent/0 | reject non-zero |
| `random` / `scale_x_y` | 0 / 1.05 | implement (scale_x_y affects the delta target too) |
| `nms_kind` / `beta_nms` | greedynms / 0.6 | inference only, unchanged |

## Loss scaling

- Darknet's per-layer cost is `Σδ²` over the **whole batch** (`outputs × batch`), not a mean; the reported loss is the mean of the two `[yolo]` costs (`_shared/darknet-semantics.md#training-loss`).
- With `iou_loss=ciou` the cost is `iou_normalizer·(tot_iou_loss/count) + obj_normalizer·Σ(δ_box-zeroed)²`, so it is **not** proportional to `batch` the way an MSE cost is — the classification part is, the IoU part is an average.
- `avg_loss` in the reference csv is an EMA `0.9·avg + 0.1·loss` seeded with the first iteration.
- Therefore: our `loss.csv` writes `loss` (Darknet formula, for comparability) **and** `loss_per_image = loss / batch` (the only column that is comparable across batch sizes for the quadratic part). Comparisons in stage 06 use `loss_per_image` and the mAP table, never the raw `loss` column against a different batch size.
- LR checkpoints for the schedule port: at iteration 500 the reference is `0.00261·0.5⁴ = 1.63e-4`; the first 1000 iterations are all `≤ 2.61e-3` and effectively ~0 early, which is why the reference csv's `learning_rate` reads `0.00000000` for its first iterations.

## Run decisions

| Decision | Value | Why |
|---|---|---|
| Init | from scratch (no `--weights`) | reproduces the reference run's starting point |
| Batch | 4 | user decision; fits the iGPU with room for gradients at 224×160 |
| Iterations | 12000 | reference saw `16 × 3000 = 48000` images; `4 × 12000` matches that exposure |
| Schedule | `steps = 9600,10800`, `scales = 0.2,0.1`, `burn_in = 1000`, `max_batches = 12000` | same *fraction* of the run as `2400,2700` of 3000, so the LR curve is comparable at equal exposure |
| Optimizer | SGD + momentum, `decay` on weights only | user decision; no Adam path exists in the code |
| Precision | f32 everywhere | user decision; no f64 master, no mixed precision |
| Save | Darknet `.weights`, BN unfolded, every 500 iterations | loadable by DarkSuite and by our own loader |
| Seed | fixed, recorded with the run | reproducibility |

## Accepted deviations

| Deviation | Why | Tolerance it implies |
|---|---|---|
| Colour augmentation does not match OpenCV's 8-bit RGB↔HSV rounding | pure Rust, no OpenCV, and `image_data_augmentation` round-trips through 8-bit HSV | per-pixel augmentation differences of a few LSB; compared by distribution/shape, not bit-exact. Never a test failure by itself |
| Crop resize uses our bilinear (`image.rs`) rather than `cv::resize` INTER_LINEAR | same reason; our resize is already verified against Darknet's `resize_image` for the inference path | small numeric differences on augmented pixels only |
| Box order within an image is not shuffled the way `data.cpp` shuffles | only affects delta accumulation order on anchors hit twice | none (order-independent in exact arithmetic, f32 differences below gradient tolerance) |
| No `optimizer.state` compatibility | Darknet's blob has no published layout | resume only within this port, stated in stage 05 |
| Batch 4 BN statistics differ from the reference's batch 16 | fewer samples per channel per step, same 0.9 momentum | expected small trajectory shift, measured in stage 06 |

## Reference history

| Quantity | Reference |
|---|---|
| Iterations × batch | 3000 × 16 = 48000 images |
| First iteration loss / avg_loss | 57.64 / 57.64 |
| Final iteration loss / avg_loss | 0.0164 / 0.0645 (lr 5.22e-5) |
| mAP@0.5 at iteration 1000 / 2000 / 2400 / 3000 | 0.329 / 0.877 / 0.938 / 0.995 |
| Best mAP@0.5 | 0.9955 (iteration 2800) |
| Loss-magnitude reality check | iteration 1 of the reference is ≈58 for 16 images, i.e. ≈3.6 per image — our batch-4 run should be in the same band once scaled |

## Human check

- Every implemented key above points at a cited fact in `_shared/darknet-semantics.md`; every rejected key fails loudly rather than being ignored.
- Given the loss-scaling section: our first iteration's `loss_per_image` should be within roughly a factor of two of 3.6. If it is orders of magnitude off, stage 02 has the wrong formula and no run should start.
- The schedule checkpoint numbers (500 → 1.63e-4, first 1000 ~0) are the ones stage 06 compares against the reference csv.
