# Rust Darknet Port

This directory contains the Rust port-in-progress of Darknet.

It is not a full replacement for the original C/CUDA code yet. The current
state is:

- Darknet `.cfg` parsing
- network and layer modeling
- `.names` loading
- `.weights` header parsing, layout inspection, and parameter loading into a CPU executable network
- a CPU execution path for a useful YOLO-style layer subset
- a Rust CLI for inspection, validation, and dummy forward execution
- a separate opt-in CUDA bridge built against `cutile-rs`

## Status

What works today:

- validate and summarize Darknet config files
- inspect weight files against a parsed network
- load Darknet weights into executable convolutional and connected layers
- run `forward-dummy` on supported models with zero or loaded weights
- collect YOLO head tensors during forward execution for later decoding work
- decode YOLO head tensors into class detections with CPU-side NMS
- load common image formats through the Rust `image` crate for preprocessing
- run end-to-end Rust image inference through the CLI on the CPU runtime
- construct a backend-generic `GpuExecutableNetwork` for the current GPU layer subset:
  convolutional, maxpool, route, shortcut, upsample, and YOLO staging
- run a compact GPU forward path for `cfg/yolov4-tiny.cfg` and `cfg/yolov4-tiny-3l.cfg`
  without retaining every intermediate CPU tensor
- build the optional `cutile-rs` bridge crate with normal Cargo commands from this repo

What is not finished:

- training, backward pass, and optimizer support
- full CUDA parity with original Darknet kernels
- recurrent and less common Darknet layer types

## Workspace Layout

- `crates/darknet-core`
  Rust model parsing, tensor/image types, names loading, weights inspection, executable-network construction, and CPU runtime support.
- `crates/darknet-cuda`
  Shared CUDA-side abstractions and typed backend traits used by Rust runtimes.
- `crates/darknet-cli`
  Command-line entry point exposed as `darknet-rs`.
- `bridges/darknet-cuda-cutile`
  Optional `cutile-rs` backend bridge. This is intentionally outside the main Cargo workspace so the git dependency does not affect every workspace build.

## CPU Runtime Coverage

The current `darknet-core::ExecutableNetwork` can execute:

- `convolutional`
- `connected`
- `maxpool`
- `local_avgpool`
- `avgpool`
- `route`
- `shortcut`
- `upsample`
- `reorg`

The runtime currently treats these as pass-through staging layers so configs can
continue to execute structurally while detector decoding is still missing:

- `yolo`
- `gaussian_yolo`
- `region`
- `softmax`
- `cost`
- `dropout`
- `scale_channels`
- `sam`
- `contrastive`

The runtime currently rejects recurrent families such as:

- `rnn`
- `lstm`
- `crnn`

## Build

From the repo root:

```bash
cargo check --manifest-path rust/Cargo.toml --workspace
```

From this `rust/` directory:

```bash
cargo check --workspace
```

Verified in this repo:

- `cargo check --workspace --locked`
- `cargo test --workspace --locked`

## CLI

The Rust CLI currently supports:

- `version`
- `cfg-validate <cfg>`
- `cfg-summary <cfg>`
- `names <names>`
- `weights-inspect <cfg> <weights>`
- `forward-dummy <cfg> [weights]`
- `detect-image <cfg> <weights> <names> <image> [thresh] [nms] [letterbox|resize]`

Examples:

```bash
cargo run -p darknet-cli -- cfg-summary cfg/yolov3-tiny.cfg
cargo run -p darknet-cli -- weights-inspect cfg/yolov4-tiny.cfg yolov4-tiny.weights
cargo run -p darknet-cli -- forward-dummy cfg/yolov3-tiny.cfg
cargo run -p darknet-cli -- detect-image cfg/yolov4-tiny.cfg yolov4-tiny.weights cfg/coco.names frame.ppm
```

The default CLI build does not compile the optional `cutile-rs` CUDA bridge. In
that mode, `detector train` uses the CPU path. To build the CLI with CUDA bridge
support, enable the opt-in feature:

```bash
cargo run -p darknet-cli --features cutile -- detector train cfg/yolov4-tiny.cfg data/train.data
```

`forward-dummy` constructs an `ExecutableNetwork`, optionally loads a Darknet
weights file, feeds a zero input tensor shaped from the `[net]` section, and
prints the final tensor shape plus any YOLO head tensors encountered during the
forward pass.

`detect-image` runs the CPU Rust inference path end to end: image load,
letterbox-or-resize preprocessing, forward execution, YOLO decode, and greedy
NMS. The Rust image loader uses the `image` crate, so common formats such as
PNG, JPEG, BMP, NetPBM, GIF, TIFF, TGA, WebP, HDR, EXR, QOI, and AVIF go
through the same preprocessing path.

## Optional CUDA Bridge

The CUDA bridge uses `cutile-rs` as the backend reference and execution path:

- `https://github.com/NVlabs/cutile-rs`

The bridge crate lives at:

- `bridges/darknet-cuda-cutile`

It currently provides:

- a concrete `CutileBackend`
- typed `f32` upload, zero-init, download, vector-add, and saxpy paths
- channel-wise `f32` bias add over row-major activation views
- elementwise `f32` activation kernels for linear, logistic, relu, leaky, mish, swish, and tanh
- `shortcut` add on top of the bridge tensor ops
- fully on-device `route` concatenation and `upsample` pointer kernels
- a direct `f32` convolution kernel path exposed through `F32TensorOpsBackend`
- a direct `f32` maxpool kernel path exposed through `F32TensorOpsBackend`
- global avgpool forward/backward exposed through `F32TensorOpsBackend` with a conservative host fallback until a device kernel is ported
- a bridge implementation of `darknet_cuda::F32TensorOpsBackend`
- a smoke example that exercises context creation, kernel launch, and device-to-host copy
- bridge-side shape validation tests for convolution, maxpool, route, and upsample layouts

What it does not provide yet:

- a direct device kernel for global avgpool
- `local_avgpool` on the GPU path
- a real detector decode / NMS path after YOLO head staging
- direct end-to-end image inference from the Rust CLI

### Requirements

- CUDA 13.2
- LLVM 21
- MLIR 21 development headers
- Polly static libraries for LLVM 21

On Debian or Ubuntu that typically means:

```bash
sudo apt install llvm-21 llvm-21-dev libmlir-21-dev mlir-21-tools libpolly-21-dev libzstd-dev
```

### Normal Cargo Usage

After the CUDA and LLVM environment variables are set, the bridge can be built
with normal Cargo commands:

```bash
cargo check --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml
cargo build --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml
cargo run --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml --example cutile_smoke
```

If the bridge does not build on another machine, run:

```bash
./rust/bridges/darknet-cuda-cutile/scripts/check-env.sh
```

For bridge-specific details, see:

- [bridges/darknet-cuda-cutile/README.md](bridges/darknet-cuda-cutile/README.md)

## Notes

- The main Rust workspace excludes the `cutile` bridge on purpose. That keeps `cargo check --workspace` fast and independent from the large external CUDA/MLIR dependency graph.
- The CLI's CUDA training path is behind the `cutile` feature for the same reason. The default CLI build falls back to CPU training when `detector train` is used.
- On this machine, `./rust/bridges/darknet-cuda-cutile/scripts/check-env.sh` finds `/usr/local/cuda-13.2` but does not find LLVM 21, MLIR 21, or Polly. The bridge build remains blocked until those dependencies are installed or exported.
- The current CPU path is the behavioral reference for the future GPU runtime. The next work is in [TODO.md](TODO.md).
