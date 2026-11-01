---
status: done
stage: workflows/port-darknet-model/01_cfg-cpu
date: 2026-10-25
---

# legogears_v2 — cfg and CPU

Contract: [`workflows/port-darknet-model/01_cfg-cpu/CONTEXT.md`](../../01_cfg-cpu/CONTEXT.md).

## Contents

- [Model files](#model-files)
- [Layer inventory](#layer-inventory)
- [New semantics](#new-semantics)
- [Ground truth](#ground-truth)
- [CPU speed](#cpu-speed)

## Model files

`~/src/DarkSuite/datasets/LegoGears_v2/`: `LegoGears_v2.cfg`, `LegoGears_v2_last.weights` (24 MB, header 0.2.5, seen 48000), `LegoGears.names` (red light, pin, center, small gear, medium gear), `LegoGears_v2_valid.txt` (80 images, 224×160 PNGs in `darkmark_image_cache/resize/` with YOLO label files).

## Layer inventory

YOLOv4-tiny, 224×160×3: 21 convolutional (19 batch-norm + leaky, 2 linear heads with 30 filters), 3 maxpool (size 2), 11 route (3 with `groups=2 group_id=1`), 1 upsample (stride 2), 2 yolo (`scale_x_y=1.05`, greedynms).

## New semantics

All layers were new; verified semantics recorded in `_shared/darknet-semantics.md` (BN ε 1e-5, DIoU-NMS for greedynms, maxpool origin, absolute route indices, corner-aligned resize).

## Ground truth

`tests/detect.rs`, 10 labelled validation images, thresh 0.25, NMS 0.45, IoU 0.5: gt 38, det 38, matched 38, recall 1.000, precision 1.000, class errors 0. Matched-box IoU against hand labels 0.66–0.97. 25 unit tests pass.

## CPU speed

About 420–445 ms/frame, single-threaded, release.
