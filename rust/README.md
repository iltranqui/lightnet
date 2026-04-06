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
- construct a backend-generic `GpuExecutableNetwork` for the current GPU layer subset:
  convolutional, route, shortcut, upsample, and YOLO staging
- build the optional `cutile-rs` bridge crate with normal Cargo commands from this repo

What is not finished:

- full detector inference output decoding and NMS
- image preprocessing and end-to-end image inference
- training, backward pass, and optimizer support
- a real GPU execution path for Darknet layers
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
cargo check
```

Equivalent explicit form:

```bash
cargo check --workspace
```

Verified in this repo:

- `cargo check`
- `cargo test`

## CLI

The Rust CLI currently supports:

- `version`
- `cfg-validate <cfg>`
- `cfg-summary <cfg>`
- `names <names>`
- `weights-inspect <cfg> <weights>`
- `forward-dummy <cfg> [weights]`

Examples:

```bash
cargo run -p darknet-cli -- cfg-summary cfg/yolov3-tiny.cfg
cargo run -p darknet-cli -- weights-inspect cfg/yolov4-tiny.cfg yolov4-tiny.weights
cargo run -p darknet-cli -- forward-dummy cfg/yolov3-tiny.cfg
```

`forward-dummy` constructs an `ExecutableNetwork`, optionally loads a Darknet
weights file, feeds a zero input tensor shaped from the `[net]` section, and
prints the final tensor shape plus any YOLO head tensors encountered during the
forward pass.

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
- a bridge implementation of `darknet_cuda::F32TensorOpsBackend`
- a smoke example that exercises context creation, kernel launch, and device-to-host copy

What it does not provide yet:

- pooling kernels
- weight caching for executable GPU layers
- full YOLO-subset GPU execution for configs such as `cfg/yolov4-tiny.cfg`, which still stop on `maxpool`

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

The repo-level Cargo config in [`.cargo/config.toml`](/home/helldiver/src/darknet/.cargo/config.toml) supplies the local CUDA and LLVM prefix variables needed by the bridge, so from this repo you can use normal Cargo commands:

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

- [bridges/darknet-cuda-cutile/README.md](/home/helldiver/src/darknet/rust/bridges/darknet-cuda-cutile/README.md)

## Notes

- The main Rust workspace excludes the `cutile` bridge on purpose. That keeps `cargo check --workspace` fast and independent from the large external CUDA/MLIR dependency graph.
- The bridge now compiles as a normal Cargo crate, but running GPU examples still depends on real NVIDIA driver and device access in the current shell.
- The current CPU path is the behavioral reference for the future GPU runtime. The next work is in [TODO.md](/home/helldiver/src/darknet/rust/TODO.md).
