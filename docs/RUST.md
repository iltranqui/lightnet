# Rust Darknet Port

Detailed design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).  Direct
C++/Rust CPU differential testing: [`docs/CPP_PARITY.md`](docs/CPP_PARITY.md).

This directory contains the Rust implementation of the Darknet runtime and
CLI. The CPU runtime is the behavioral reference; the GPU runtime uses native
backend operations where available and a correctness-first CPU-reference
fallback for operations that do not yet have device kernels.

## Implemented

- Darknet `.cfg`, `.names`, and `.weights` parsing, inspection, loading, and
  saving, including recurrent nested-layer weight layouts.
- Effective Darknet mini-batch and `time_steps` handling.
- Legacy configuration aliases, inline comments, whitespace-separated lists,
  and Darknet learning-rate policies (`constant`, `step`, `steps`, `poly`,
  `exp`, `sig`, and `sgdr`).
- CPU forward and backward execution for:
  - convolutional layers, grouped convolution, batch normalization, and all
    supported activations;
  - connected layers;
  - maxpool, local average pool, and global average pool;
  - route, shortcut, upsample, and reorg;
  - softmax, dropout, scale-channels, and SAM;
  - RNN, LSTM, and CRNN recurrent layers with cached BPTT;
  - YOLO/region/gaussian-YOLO/cost/contrastive staging.
- CPU training buffers, SGD with momentum/decay, learning-rate scheduling,
  detector target assignment and loss, checkpoint writing, image loading, YOLO
  decode, and CPU NMS.
- A backend-generic `GpuExecutableNetwork` with native dispatch for the
  available convolution, pooling, route, shortcut, upsample, and YOLO paths.
  Connected, reorg, softmax, dropout, scale-channels, SAM, recurrent layers,
  and mismatched-shape shortcuts use a CPU-reference bridge while retaining
  GPU convolution gradients and updates where possible.
- GPU training-state retention, seeded backward accumulation, batch-norm
  running-stat updates, and persistent SGD update buffers.

The Rust workspace has warning-clean checks and tests. The core crate contains
69 unit tests plus a direct C++ CPU parity integration test, including explicit
CPU inference/training forward-backward coverage, recurrent update tests,
weights round trips, GPU fallback coverage, and an analytical convolution/SGD
check. That convolution test uses an independent closed-form NCHW reference
for input, weight, and bias gradients, then verifies that one Darknet-style
stochastic-gradient update moves the parameters in the expected direction and
lowers the loss. The parity test compiles the repository's actual
`src-lib/connected_layer.cpp` and compares Rust/C++ outputs and gradients.

## Workspace Layout

- `src/darknet-core`: model parsing, tensors, weights, inference, training,
  and CPU/GPU runtime orchestration.
- `src/darknet-cuda`: backend traits and CUDA-facing tensor abstractions.
- `src/darknet-cli`: the `darknet-rs` command-line application.
- `src/darknet-examples`: Rust equivalents of all 11 programs in
  ../src-examples, including image, video, JSON, YOLO-label, keypoint,
  RTSP, and webcam workflows.
- `src/darknet-cuda-cutile`: optional `cutile-rs` backend.

## Build and Verify

From the Darknet repository root:

```bash
source toolchain.env
cd rust
RUSTFLAGS=-Dwarnings cargo check --workspace --all-targets --locked -j 2
RUSTFLAGS=-Dwarnings cargo test --workspace --all-targets --locked -j 2
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check

# The optional cuTile bridge is checked separately from the main workspace.
./src/darknet-cuda-cutile/scripts/check-env.sh
RUSTFLAGS=-Dwarnings cargo test \
  --manifest-path src/darknet-cuda-cutile/Cargo.toml --locked --lib

# Require the direct C++/Rust fixture instead of allowing a missing compiler
# to be reported as a portable-development skip.
DARKNET_CPP_PARITY_REQUIRED=1 \
  RUSTFLAGS=-Dwarnings cargo test -p darknet-core --test cpp_parity -- --nocapture
```

`toolchain.env` selects the workspace-local Rust 1.97.1 toolchain, Cargo,
Clippy, rustfmt, and CUDA 12.9 `nvcc`. `nvcc` compilation can be verified
without a GPU; executing CUDA kernels still requires a compatible GPU.

## CLI

```text
version
cfg-validate <cfg>
cfg-summary <cfg>
names <names>
weights-inspect <cfg> <weights>
forward-dummy <cfg> [weights]
detect-image <cfg> <weights> <names> <image> [threshold] [nms] [letterbox|resize]
```

Examples:

```bash
cargo run -p darknet-cli -- cfg-summary ../cfg/yolov4-tiny.cfg
cargo run -p darknet-cli -- forward-dummy ../cfg/yolov4-tiny.cfg
cargo run -p darknet-cli -- detect-image ../cfg/yolov4-tiny.cfg model.weights ../cfg/coco.names frame.ppm
```

The example binaries use explicit model arguments:

    cargo run -p darknet-examples --bin darknet_01_inference_images -- <cfg> <weights|-> <names|-> image...
    cargo run -p darknet-examples --bin darknet_04_process_videos -- <cfg> <weights|-> <names|-> video...
    cargo run -p darknet-examples --bin darknet_06_images_to_json -- <cfg> <weights|-> <names|-> image...
    cargo run -p darknet-examples --bin darknet_11_images_to_yolo -- <cfg> <weights|-> <names|-> image...

All 11 binaries are named after their C++ counterparts. Video, RTSP, and
webcam equivalents use ffmpeg and ffprobe for media I/O, while display
variants use ffplay. Model execution, decoding, annotation, and output
ordering remain in Rust. Common options are --threshold, --nms, --resize,
--output, and --seconds.

| Darknet C++ example | Rust equivalent |
| --- | --- |
| darknet_01_inference_images.cpp | darknet_01_inference_images |
| darknet_02_display_annotated_images.cpp | darknet_02_display_annotated_images |
| darknet_03_display_videos.cpp | darknet_03_display_videos |
| darknet_04_process_videos.cpp | darknet_04_process_videos |
| darknet_05_process_videos_multithreaded.cpp | darknet_05_process_videos_multithreaded |
| darknet_06_images_to_json.cpp | darknet_06_images_to_json |
| darknet_07_display_rtsp_stream.cpp | darknet_07_display_rtsp_stream |
| darknet_08_display_webcam.cpp | darknet_08_display_webcam |
| darknet_09_keypoints_images.cpp | darknet_09_keypoints_images |
| darknet_10_keypoints_videos.cpp | darknet_10_keypoints_videos |
| darknet_11_images_to_yolo.cpp | darknet_11_images_to_yolo |

## GPU and cuTile

The optional bridge is built separately:

```bash
./src/darknet-cuda-cutile/scripts/check-env.sh
cargo check --manifest-path src/darknet-cuda-cutile/Cargo.toml --locked
```

The bridge is pinned to cutile-rs v0.2.0. Its default JIT path uses the
pure-Rust cutile-ir, so LLVM/MLIR 21 and Polly are no longer required for the
normal build. CUDA 13.3 is the upstream-recommended target; CUDA 13.2 or newer
is appropriate for Ada sm_89. CUDA 12.9 remains useful for compiler-only
checks but is too old for the current low-precision path. CUDA headers and a
checks but is too old for the current low-precision path. CUDA, cuRAND, and
libclang installations are needed by the generated CUDA binding step. The
workspace setup includes the cuRAND development headers/library alongside the
compiler-only CUDA tree.
See the [cuTile Rust repository](https://github.com/NVlabs/cutile-rs) and the
[v0.2.0 release notes](https://github.com/NVlabs/cutile-rs/releases/tag/v0.2.0).

The cuTile bridge currently exposes device kernels for tensor upload/download,
elementwise activation, bias and channel scaling, shortcut add, route,
upsample, convolution, maxpool, and the backend trait. Global/local average
pooling still include conservative host fallbacks, and detector decoding/NMS
remains CPU-side.

The bridge's default tests validate layout and API contracts without requiring a
GPU. A real-device differential test compares cuTile convolution forward,
input/weight/bias backward gradients, and one SGD update against a scalar
reference. Run it only on a CUDA host with the driver visible:

```bash
export DARKNET_CUTILE_GPU_TEST=1
RUSTFLAGS=-Dwarnings cargo test \
  --manifest-path src/darknet-cuda-cutile/Cargo.toml --locked --lib \
  -- --ignored --nocapture
```

## How to extend the port

For a C++ behavior change, update the typed model/parser, shape inference, CPU
forward, caches, CPU backward, weights order, backend dispatch/fallback, tests,
and documentation as one change. The CPU implementation is the reference path;
native CUDA code is an optimization layer and must preserve its tensor layout
and gradients. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#extension-rules).

## Scope Boundary

The practical Darknet layer families used by the repository’s standard YOLO,
ResNet, CSP, EfficientNet, and recurrent configs are modeled and executable.
Historical/nonstandard config sections outside the current C++ parser’s layer
set still return `UnsupportedLayer`. Native GPU kernels for every CPU operation
are also not claimed: the GPU API preserves functional parity through its
explicit CPU-reference fallback until those kernels and their LLVM/CUDA
environment are available.
