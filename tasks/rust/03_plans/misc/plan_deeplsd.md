# DeepLSD line segment detection plan

Plan to run DeepLSD (learned line segment detection) in lightnet, using the pretrained
weights. Reference implementation: https://github.com/sanket-pixel/lines-metis (DeepLSD
on the Axelera Metis accelerator, C++ postprocessing).

Status: proposal. Nothing is implemented. Details marked "unverified" come from the repo
README summary and from memory of the DeepLSD paper; confirm them against the ONNX export
and the repo source before writing code.

## Table of Contents

- [Goal](#goal)
- [Model and output](#model-and-output)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
- [Postprocessing algorithm](#postprocessing-algorithm)
- [Ground truth and verification](#ground-truth-and-verification)
- [Implementation plan](#implementation-plan)
- [Open questions](#open-questions)

## Goal

Detect 2D straight line segments (wall edges, door frames, table edges) in a single
image and output `[x1, y1, x2, y2]` in original-image pixels. A line here is a geometric
feature with no class. It differs from a Canny edge map: Canny marks every intensity
gradient as binary pixels, while DeepLSD marks only long straight structures and returns
vector segments with endpoints.

Out of scope: tracking, depth, SLAM, ROS, training from scratch, Metis INT8 and the
Voyager SDK. The lightnet equivalent of quantization is the existing fp8/fp16/bf16 path.

## Model and output

Backbone: VGG-UNet style CNN (unverified layer counts).

- Encoder: conv3x3, BN, ReLU blocks with 2x2 maxpool, about 4 stages (channels about
  64, 128, 256, 512). Pre-pool maps are kept as skips.
- Decoder: upsample x2, concat with the matching skip, conv3x3, BN, ReLU, down to input
  resolution (stride 1).
- Two 1x1 heads off the same features.

Head output, at input resolution (240x320 after crop), combined `[2, H, W]`:

| Head | Shape | Raw value | Decoded |
|---|---|---|---|
| Distance | `[1, H, W]` | `d_norm`, about [0, 1] | `dist = 5 * exp(-d_norm)` px |
| Angle | `[1, H, W]` | `angle_frac`, [0, 1] | `theta = pi * angle_frac` rad |

Each pixel states how far it is from the nearest line and the orientation of that line.
No endpoints exist until the CPU extractor runs.

Darknet mapping:

| DeepLSD op | Darknet layer | Status |
|---|---|---|
| conv, BN, ReLU | `[convolutional] batch_normalize=1 activation=relu` | check that `relu` exists |
| maxpool | `[maxpool]` | exists |
| upsample | `[upsample]` | nearest only; DeepLSD may use bilinear |
| skip concat | `[route] layers=-1,N` | exists |
| 1x1 heads | `[convolutional] activation=linear` or `logistic` | exists; check sigmoid |

Two parallel heads need two `[convolutional]` layers, routed and concatenated into one
2-channel output, since Darknet emits the last layer only.

## Project tree

```
lightnet/
├── cfg/
│   └── deeplsd.cfg                 (new) VGG-UNet cfg, two heads concatenated
├── tools/
│   └── convert_deeplsd.py          (new) checkpoint/ONNX -> .weights, BN folded
├── src/
│   ├── lightnet-core/
│   │   └── src/
│   │       ├── lines.rs            (new) decode + SegmentExtractor (CPU)
│   │       └── (upsample, activation) (maybe) bilinear upsample, relu
│   └── lightnet-cli/
│       └── src/                    (edit) `lines` subcommand: image -> segments
└── tasks/rust/
    └── plan_deeplsd.md             this file
```

## Component breakdown

### cfg/deeplsd.cfg
- Role: network definition in Darknet format.
- Responsibilities: encoder, decoder with routes, two heads, final 2-channel concat.
- Depends on: layer coverage in `layer-support.md`. Used by the weights converter and CLI.
- Constraints: relu and upsample mode must match the checkpoint, or fields drift.

### tools/convert_deeplsd.py
- Role: turn the pretrained MegaDepth DeepLSD checkpoint (MIT licensed) into `.weights`.
- Responsibilities: export or read tensors, fold BN into conv as Darknet does, write in
  cfg order.
- Depends on: PyTorch and the DeepLSD repo. Used by the parity test.
- Open: ONNX versus direct state_dict read. See [Open questions](#open-questions).

### Backbone layers (lightnet-core)
- Role: forward pass for the cfg.
- Responsibilities: reuse existing conv, maxpool, route, upsample. Add relu and bilinear
  upsample only if the checkpoint needs them.
- Depends on: `layers.md`. GPU versions follow `gpu_layers_plan.md`; CPU first.

### lines.rs (decode + SegmentExtractor)
- Role: dense fields to vector segments. Pure CPU, no Darknet equivalent.
- Responsibilities: see [Postprocessing algorithm](#postprocessing-algorithm).
- Depends on: the 2-channel output tensor. Used by the CLI.
- Constraints: threshold presets (strict, balanced, relaxed) as a config struct, with
  balanced as the default, matching the reference repo.

### lightnet-cli `lines` subcommand
- Role: end-to-end entry: load image, resize to 320x240, grayscale, run net, extract,
  map back, print or draw segments.
- Depends on: all of the above.

## Postprocessing algorithm

1. Decode per pixel: `dist = 5 * exp(-d_norm)`, `theta = pi * angle_frac`.
2. Region growing: seed at pixels with small `dist`; grow to 8-neighbours when `dist` is
   under a threshold and the angle differs from the region angle by less than a tolerance.
   Angles wrap mod pi: use `min(|d|, pi - |d|)`.
3. Weighted PCA per region: covariance of pixel coordinates weighted by `exp(-dist)` or
   similar. The principal eigenvector is the direction; the projection extent gives the
   endpoints.
4. Validate: minimum pixel support, minimum length, eigenvalue ratio test for
   elongation, optional agreement between PCA angle and mean `theta`.
5. Join: merge nearly collinear segments with a small angle difference and an endpoint
   gap under the join-gap threshold.
6. Dedup: drop overlapping segments with close angle and perpendicular offset; keep the
   longer or better supported one.
7. Map back: scale endpoints from 320x240 to the original image, accounting for pad and crop.

The exact growing rule and thresholds are in the reference repo's C++ `SegmentExtractor`.
Read it before implementing.

## Ground truth and verification

With the pretrained weights no labels are needed. The postprocessing is heuristic.

- Parity, no labels: run one image through PyTorch DeepLSD and through lightnet. Compare
  the distance and angle fields (max abs error), then the final segments.
- Quality, optional: Wireframe (ShanghaiTech) or YorkUrban line annotations; metrics are
  structural AP and repeatability. The reference repo reports no accuracy number.
- Calibration for fp8 scales, optional: images only (the reference uses 200 TUM RGB-D
  Freiburg 2 frames).

Training or fine-tuning is out of scope. It would need per-pixel targets: distance to the
nearest GT segment (clipped, normalized) and the angle of that segment mod pi, an L1/L2
loss with wrap-aware angle error, and a label source (the authors reportedly used LSD
pseudo-labels with homography adaptation; unverified).

## Implementation plan

- [ ] Read the reference repo source (`SegmentExtractor`, pre/postprocessing) and export
      the DeepLSD ONNX; record exact layer list, activations, upsample mode, head ranges.
- [ ] Check `tasks/rust/02_state/layer-support.md` against that list; list the missing layers.
- [ ] Write `cfg/deeplsd.cfg`.
- [ ] Write `tools/convert_deeplsd.py` and produce `.weights` (do not commit the weights).
- [ ] Add missing layers (relu, bilinear upsample) on CPU, with unit tests.
- [ ] Forward-pass parity: fields versus PyTorch on one image.
- [ ] Implement decode in `lines.rs`, with unit tests on synthetic fields.
- [ ] Implement region growing, PCA, validate, join, dedup, with synthetic-line tests.
- [ ] Add the CLI `lines` subcommand and the segment overlay output.
- [ ] End-to-end parity: segments versus the reference repo on a few images.
- [ ] Optional: GPU layers per `gpu_layers_plan.md`; fp16/bf16 per `plan_fp16.md`.
- [ ] Update `tasks/rust/02_state/STATUS.md`, `layer-support.md`, and commit with the feature tag.

Build and test through `cargo xtask` only (4-thread and 3G caps apply).

## Open questions

- Does the checkpoint use bilinear upsample, and does the Rust port have one?
- Do the heads end in a sigmoid or a linear output? Is there a third channel?
- ONNX export or direct state_dict read for the converter?
- Where does the extractor live: `lightnet-core` or a separate small crate?
- Is the reference repo licence compatible with porting its extractor logic (check
  before copying any code; reimplement from the algorithm description otherwise)?
