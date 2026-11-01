---
status: review
stage: workflows/train-darknet-model/05_cli-checkpoints
date: 2026-10-26
---

# legogears_v2 — CLI, checkpoints, metrics

Contract: [`workflows/train-darknet-model/05_cli-checkpoints/CONTEXT.md`](../../../05_cli-checkpoints/CONTEXT.md). This stage was built ahead of 03/04 because everything in it is device-independent: the GPU trainer will drive the same loop.

## Contents

- [Command line](#command-line)
- [File formats](#file-formats)
- [Round trip and resume](#round-trip-and-resume)
- [First real run](#first-real-run)
- [Dataset findings](#dataset-findings)

## Command line

```
hip_darknet train --data <…>.data [--cfg <…>.cfg] [--iters N] [--batch N] [--seed S]
                  [--out DIR] [--save-every N] [--eval-every N] [--eval]
                  [--weights init.weights] [--resume last.state]

# Darknet's shape, same trainer:
hip_darknet detector train <…>.cfg <…>.data [-map] [-dont_show] [-gpus N]
                             [-iters N] [-batch N] [-seed S] [-out DIR]
                             [-save_every N] [-map_every N] [-weights W]
```

`detector train` takes the two paths in either order (they are told apart by extension),
ignores `-dont_show`, `-gpus`, `-clear` and `-show_imgs` like Darknet does, turns `-map` into
`--eval --eval-every 1000`, and defaults the output directory to `<data dir>/hip_train` — the
spiritual equivalent of Darknet's `backup`.

- `--cfg` defaults to the `.data` path with a `.cfg` extension (this dataset ships `LegoGears_v2.cfg`).
- `--iters` defaults to the cfg's `max_batches`.
- The inference CLI (`--gpu/--bench/--out/--thresh/--nms`) is unchanged; `train` is a subcommand.

## File formats

| File | Contents | Compatible with |
|---|---|---|
| `iter_NNNNNN.weights`, `last.weights` | Darknet binary: `i32 major/minor/revision`, `u64 seen`, per conv `bias, [scales, rolling mean, rolling var,] weights` — **unfolded** | DarkLedger/DarkSuite and `hip_darknet`'s own loader (verified: `train/checkpoint.rs` test folds to the same values as `Model::load`) |
| `last.state` | our own: magic, `seen`, then per conv `weights, bias, scales, rolling mean/var, weight_updates, bias_updates, scale_updates` | this port only (Darknet's `optimizer.state` layout is unpublished) |
| `loss.csv` | `iteration,seen,loss,avg_loss,loss_per_image,learning_rate,milliseconds` | reference columns + `loss_per_image` (the only cross-batch-comparable one) |
| `map.csv` | `iteration,seen,iou,threshold,map,best_map`, one row at `iou=0.5` and one at `0.5:0.95` | reference columns; the reference only logs `iou=0.5` |

## Round trip and resume

- `.weights` save → load → per-layer equality (all layers), and save → `Model::from_parts` equality with the verified inference folder, to `1e-6`.
- `last.state` save → load → equal weights, momentum buffers and `seen`.
- Resume is proven by the format round trip; the split-run vs single-run comparison belongs to stage 06 (it needs a run long enough to split).

## First real run

```
hip_darknet train --data …/LegoGears_v2.data --iters 3 --batch 2 --out /tmp/hipd_smoke
iter 1 loss 144.6892 (avg 144.6892) lr 0.00000000 11042 ms
trained 3 iterations (6 images seen) in 35.9s: loss 144.6892 -> 144.2452 (avg 144.6455)
```

Real dataset, real augmentation, real deltas, real checkpoint, real csv. The learning rate is 0 because `burn_in = 1000` with `power = 4` (`(1/1000)^4`), exactly as the reference csv's first rows show — the schedule port is behaving.

CPU cost: ~5.5 s per image at 224×160 on one core (forward + backward + Adam-free SGD). The full 12 000-iteration batch-4 run is therefore a GPU job (stages 03/04); the CPU path exists to validate and to give a reference implementation.

## Dataset findings

- `LegoGears_v2_train.txt` and `LegoGears_v2_valid.txt` list the **same 80 images in the same order**. The reference `map.csv` (mAP 0.995) is therefore measured on training images. Any honest generalisation claim needs a different split; stage 06 must state this.
- The only full weights file shipped is `LegoGears_v2_last.weights`; `*_final.weights.fp8scales` are quantisation scales, not weights. `LegoGears_v2_last.weights` loads cleanly to EOF through both loaders but produces **no detections** on the dataset's own images (CPU path, threshold 0.25) — so it is not a reference checkpoint we can score against. Stage 06's comparison must go through the csv trajectory, not through a shipped model.
- Images are pre-resized to 224×160 in `darkmark_image_cache/resize/`, so the resize path is a no-op and augmentation works from cache-sized images.

## Human check

- Open the produced `loss.csv` next to `LegoGears_v2_loss.csv`: same columns for the shared fields, plus `loss_per_image`.
- Does the first iteration's loss look sane for the batch size? 144.7 for 2 images at random init is the classification term over 175 cells × 3 anchors × 2 heads; it is the same quantity Darknet prints, which at batch 16 was 57.6 in the reference run (different RNG, and that run's very first iterations are also lr = 0).
- The two dataset findings above are decisions for stage 06: which images to score on, and that there is no trustworthy shipped checkpoint.
