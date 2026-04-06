# Rust Darknet CUDA TODO

This document is the implementation plan for the next stage of the Rust port:
move from a working `cutile-rs` bridge to real Darknet tensor operators and
eventually end-to-end GPU inference.

## Current Baseline

Already working:

- Rust workspace and CLI
- repo-root Cargo workspace commands:
  - `cargo check`
  - `cargo test`
- `.cfg` parsing and network/layer modeling
- `.names` loading
- `.weights` inspection
- `.weights` loading into executable convolutional and connected layers
- CPU runtime for a useful YOLO-style subset
- CPU support for:
  - convolutional
  - connected
  - maxpool
  - local_avgpool
  - avgpool
  - route
  - shortcut
  - upsample
  - reorg
- YOLO head output collection during CPU forward execution
- optional `cutile-rs` bridge crate
- typed `f32` tensor upload, zero-init, download, vector-add, and saxpy on the bridge
- bridge implementation of `darknet_cuda::F32TensorOpsBackend`
- GPU bridge support for:
  - convolutional
  - route
  - shortcut
  - upsample
  - yolo passthrough staging
- backend-generic `GpuExecutableNetwork`
- workspace tests covering config parsing, weights inspection, and basic runtime execution

Still missing on the CUDA side:

- maxpool and the remaining pooling family
- weight upload and persistent device parameter storage
- full-model GPU forward on:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- CPU/GPU parity checks on real model layers
- detector decoding and post-processing
- end-to-end image inference

## Goal

Build a minimal but real GPU inference path in Rust that can execute the
practical YOLO-tiny targets for this repo with `cutile-rs`.

Primary model targets:

- `cfg/yolov4-tiny.cfg`
- `cfg/yolov4-tiny-3l.cfg`

The first target should be:

- convolution
- bias add
- activation
- route
- shortcut
- upsample
- maxpool
- yolo output staging

## Guiding Principles

- Keep the main `rust` workspace independent from `cutile-rs`.
- Keep the `cutile` bridge in `bridges/darknet-cuda-cutile`.
- Start with inference-only support.
- Prefer correctness and debuggability before optimization.
- Reuse the existing CPU runtime as the reference implementation for behavior.
- Add layer-by-layer parity tests before trying full-model parity.

## Phase 1: Harden The CUDA Bridge

Objective:
turn the bridge from a smoke demo into a reusable backend layer for Darknet ops.

Status:

- complete enough for the next stage

Done:

- concrete `CutileBackend`
- `CutileTensorF32`
- explicit shape validation on upload and elementwise ops
- reusable typed upload, zero-init, download, vector-add, and saxpy helpers
- `darknet_cuda::F32TensorOpsBackend` implementation
- smoke example proving context creation and device-to-host round trip

Deliverable:

- a stable `darknet_cuda::F32TensorOpsBackend` implementation suitable for higher-level ops

Carry-over cleanup that can happen opportunistically:

- add focused bridge tests for round trips, vector-add, saxpy, and shape mismatch failures
- document allocation and ownership rules for activations, workspaces, and persistent weights
- add more typed tensor wrappers only when a real Darknet op needs them

## Phase 2: Add Core Tensor Ops

Objective:
implement the first GPU primitives that the existing CPU runtime already uses.

Status:

- mostly complete except for pooling

Implemented in the `cutile` bridge:

- fused channel-wise bias add over row-major activation views
- activation kernels for:
  - linear
  - logistic
  - relu
  - leaky
  - mish
  - swish
  - tanh
- shortcut add exposed through the backend trait
- fully on-device route concatenation and upsample kernels
- a direct convolution kernel path exposed through the backend trait

Implemented in `darknet-core`:

- `Activation::to_cuda_f32_kind()` adapter for the GPU backend
- a backend-generic `GpuExecutableNetwork`
- GPU convolution dispatch with host-side batch-norm folding into inference weights
- fake-backend tests covering convolution, route, shortcut, and upsample dispatch
- a repo-backed `cfg/yolov4-tiny.cfg` coverage-boundary test that currently stops at `maxpool`

Priority order:

1. bias add
2. activation
3. pointwise shortcut add
4. route concatenation
5. upsample
6. maxpool
7. convolution support path

Tasks:

- implement a fused bias-add kernel
- implement activation kernels for:
  - linear
  - leaky
  - relu
  - logistic
  - mish
  - swish and tanh if target configs require them
- implement shortcut as elementwise add with shape checks
- implement route as tensor concatenation along channel dimension
- implement upsample nearest-neighbor behavior matching Darknet
- implement maxpool with Darknet-compatible padding/stride behavior

Deliverable:

- GPU implementations for the cheap non-convolution layers used by YOLO models, plus the first direct convolution path

## Phase 3: First Convolution Path

Objective:
get the first real `convolutional` layer running on GPU and make it usable from
the existing executable-network structure.

Status:

- partially complete

Decision:

Start with inference-only forward convolution. Do not attempt backward or
training kernels yet.

Implementation options:

1. direct convolution kernel in `cutile`
2. im2col + GEMM path
3. special-case `1x1` convolution first, then general kernels

Recommended order:

1. `1x1` convolution
2. general `kxk` convolution
3. optional fused conv+bias+activation path

Tasks:

- keep the current direct convolution kernel correct while extending coverage beyond the basic path
- upload convolution weights and biases into persistent device tensors instead of re-uploading them during `forward()`
- define a Rust-side layer parameter cache keyed by layer index
- compare GPU output against CPU output on small deterministic inputs
- extend to general kernel sizes and strides
- decide whether to use:
  - pure `cutile` kernel path
  - im2col in Rust + `cutile` GEMM
  - mixed strategy depending on layer shape

Deliverable:

- GPU forward support for `convolutional` layers with loaded weights

## Phase 4: GPU Executable Network

Objective:
connect per-op kernels to the existing Rust network runtime without forking the
model semantics.

Status:

- in progress

Already done:

- define a backend-generic `GpuExecutableNetwork`
- mirror CPU dispatch for:
  - convolutional
  - route
  - shortcut
  - upsample
  - yolo passthrough staging
- keep the GPU path structurally aligned with the CPU runtime for the supported subset
- add a repo-backed `cfg/yolov4-tiny.cfg` test that identifies `maxpool` as the current first unsupported GPU layer

Tasks:

- allocate activation tensors per layer
- upload all layer weights during model initialization
- implement layer dispatch for:
  - maxpool
- avoid per-forward temporary parameter uploads for supported weighted layers

Deliverable:

- a Rust GPU runtime that can execute a useful YOLO subset

## Phase 5: Verification And Parity

Objective:
prove the GPU path matches the CPU path closely enough to build on it.

Tasks:

- add per-layer parity tests between CPU and GPU outputs
- add small synthetic fixtures for:
  - convolution
  - route
  - shortcut
  - maxpool
  - upsample
- add end-to-end parity on the primary target configs:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- define tolerances for floating-point comparison
- add debug utilities to dump intermediate layer outputs when parity fails

Deliverable:

- repeatable CPU/GPU correctness checks

## Phase 6: Detector Output And Images

Objective:
make the Rust path useful for actual inference output, even if some post-process
steps stay on CPU initially.

Tasks:

- port YOLO head decoding
- port box/objectness/class score extraction
- implement NMS on CPU first
- add image loading
- add Darknet-compatible resize and letterbox behavior
- convert image data into network tensor layout
- add CLI support for real inference outputs
- optionally move post-processing to GPU later

Deliverable:

- end-to-end Rust inference from image path to detections

## Phase 7: Training Roadmap

Objective:
extend the `cutile-rs` path from inference into real training, but only after
forward execution is solid on `cfg/yolov4-tiny.cfg` and `cfg/yolov4-tiny-3l.cfg`.

Status:

- not started

Hard prerequisite before training work:

- full GPU forward coverage for:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- parity checks for the supported forward path
- persistent device weight storage and runtime parameter caching

Stage 1: Training-Capable Runtime

- define a training runtime or extend `GpuExecutableNetwork` with train/eval modes
- keep forward activations needed by backward instead of discarding them after each layer
- define device-side parameter bundles for:
  - weights
  - biases
  - gradients
  - optimizer state
- define workspace/lifetime rules for activations, gradients, and scratch buffers

Stage 2: Backward Ops

- implement backward kernels for:
  - convolutional
  - activation
  - shortcut
  - route
  - upsample
  - maxpool
- define gradient accumulation rules that match Darknet semantics
- add CPU-reference checks on synthetic tensors before model-level tests

Stage 3: YOLO Losses And Targets

- port target assignment and anchor matching
- port objectness, class, and box regression losses
- support both 2-head and 3-head tiny layouts used by:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- verify loss values and per-head gradients against the CPU/original Darknet path

Stage 4: Optimizers And Updates

- implement SGD with momentum first
- add weight decay, learning-rate scheduling, and EMA only after basic training works
- decide whether Adam/AdamW are needed, or whether Darknet-compatible SGD is sufficient
- support mixed host/device bookkeeping for logging without moving parameter tensors back to CPU

Stage 5: Data And Training Loop

- add dataset loading and label parsing
- add batching, shuffling, and epoch iteration
- add training-time resize/letterbox behavior
- add augmentation only after non-augmented training is correct
- add CLI support for training runs and checkpoint saves

Stage 6: Verification

- add per-op gradient checks where practical
- run short overfit tests on tiny subsets for:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- confirm loss decreases and checkpoints reload correctly
- compare early training curves against the existing Darknet implementation before optimizing

Deliverable:

- a first end-to-end `cutile-rs` training path for `yolov4-tiny` and `yolov4-tiny-3l`

## Suggested Immediate Next Steps

Recommended short-term sequence:

1. implement GPU `maxpool`
2. add persistent device weight storage and a layer-parameter cache
3. run `cfg/yolov4-tiny.cfg` and `cfg/yolov4-tiny-3l.cfg` through the GPU runtime past the current `maxpool` boundary
4. add CPU/GPU parity tests for convolution, route, upsample, and the first real YOLOv4-tiny slices
5. decide whether later convolution work stays pure `cutile` or switches to im2col + GEMM for some shapes

This order attacks the current blocking layer first, then removes the obvious runtime inefficiency before expanding coverage on the two target tiny models.

## Non-Goals For Now

Do not spend time on these until inference is solid on `yolov4-tiny` and `yolov4-tiny-3l`:

- backward pass
- optimizer logic
- data augmentation pipeline
- multi-GPU training
- full Darknet feature parity
- aggressive kernel autotuning

## Open Design Questions

- should GPU execution live in `darknet-core` behind a backend trait, or in a separate executable runtime crate?
- should weights be uploaded eagerly for all layers, or lazily on first use?
- should convolution start as pure `cutile` kernels, or should the first implementation use im2col + GEMM for faster progress?
- how much operator fusion is worth doing before correctness is proven?
- should YOLO decode remain CPU-side initially even when the backbone/head run on GPU?
