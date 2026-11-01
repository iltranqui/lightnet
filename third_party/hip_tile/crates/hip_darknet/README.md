# hip_darknet

Pure-Rust Darknet YOLO: `.cfg` and `.weights` loading with batch-norm folding, a CPU reference forward, YOLO decode with NMS, and (feature `gpu`, default) a GPU executor on the hip-tile kernels.

## Contents

- [Usage](#usage)
- [Features](#features)
- [Library API](#library-api)
- [Tests](#tests)

## Usage

```sh
cargo run -p hip_darknet --release -- [--gpu] --cfg C --weights W --names N \
    [--thresh 0.25] [--nms 0.45] [--out PATH] [--bench N] image...
```

- `--gpu`: forward on the GPU (error if built without the `gpu` feature).
- `--out PATH`: annotated image with class-colored rectangles; a directory when several images are given.
- `--bench N`: N forwards on the first image after warm-up, best of several rounds; prints CPU and (with `--gpu`) GPU ms/frame, forward only and end to end.

## Features

- `gpu` (default): `gpu::GpuModel`, needs ROCm and the pinned nightly. `--no-default-features` builds CPU-only.

## Library API

- `Model::load(cfg, weights)`: layers with shapes, folded conv weights.
- `reference::forward`: every layer's output on the CPU.
- `gpu::GpuModel::new(&model)`, `forward` (yolo inputs), `forward_all` (every layer).
- `detect_cpu`, `detect_gpu`: resize, forward, decode, NMS.

## Tests

`tests/detect.rs` checks CPU detections against dataset labels; `tests/gpu.rs` checks per-layer GPU vs CPU parity and detection parity. Both skip when the LegoGears dataset is missing.
