# Rust Darknet CUDA TODO

This document tracks the next stage of the Rust port:
move from a working `cutile-rs` bridge to a C++-aligned Darknet runtime with
usable GPU inference and training.

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
- YOLO decode, CPU NMS, image resize/letterbox preprocessing, and CLI image inference
- CPU backward/runtime support for:
  - convolutional
  - maxpool
  - route
  - shortcut
  - upsample
- CPU training buffers that follow the C++ per-layer update-buffer scheme
- CPU `detector train` loop with:
  - `.data` parsing
  - image-list loading
  - batch loading
  - learning-rate scheduling
  - SGD with momentum and decay
  - checkpoint save support
- optional `cutile-rs` bridge crate
- typed `f32` tensor upload, zero-init, download, vector-add, and saxpy on the bridge
- bridge implementation of `darknet_cuda::F32TensorOpsBackend`
- GPU bridge support for:
  - convolutional
  - activation backward
  - route backward
  - maxpool
  - maxpool backward
  - route
  - shortcut
  - upsample
  - upsample backward
  - bias backward
  - convolution backward input
  - convolution backward weights
  - in-place SGD-style parameter update
  - yolo passthrough staging
- backend-generic `GpuExecutableNetwork`
- persistent device-side convolution parameter caching
- compact GPU forward that can run:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- initial GPU training scaffolding:
  - retained forward state for backward
  - device-side train buffers for convolution owners
  - GPU backward accumulation for the supported subset
  - GPU weight update path
- workspace tests covering config parsing, weights inspection, and basic runtime execution

Still missing on the CUDA side:

- a working cutile/CUDA runtime on the target machine
- detector training with real YOLO loss instead of placeholder output seeds
- batch-norm training-time running-stat updates
- momentum-preserving GPU train buffers across iterations
- a shared C++-style reusable workspace buffer for CPU/GPU training scratch
- GPU support for the remaining pooling family and unsupported train-time layers
- CPU/GPU parity checks on more realistic end-to-end detector slices

## Goal

Build a C++-aligned Rust runtime that can execute the practical YOLO-tiny
targets for this repo on GPU with `cutile-rs`, then extend that path into real
detector training.

Primary model targets:

- `cfg/yolov4-tiny.cfg`
- `cfg/yolov4-tiny-3l.cfg`

The first complete GPU target should cover:

- convolution
- bias add
- activation
- route
- shortcut
- upsample
- maxpool
- yolo output staging

The first complete training target should additionally cover:

- retained activations for backward
- device-side update buffers with C++ ownership semantics
- real YOLO target assignment and loss
- SGD with momentum on GPU

## Guiding Principles

- Keep the main `rust` workspace independent from `cutile-rs`.
- Keep the `cutile` bridge in `bridges/darknet-cuda-cutile`.
- Prefer correctness and debuggability before optimization.
- Reuse the existing CPU runtime as the reference implementation for behavior.
- Add layer-by-layer parity tests before trying full-model parity.
- Follow the C++ memory and ownership scheme closely:
  - per-layer parameter ownership
  - `share_layer`-style aliasing
  - persistent `*_updates` buffers
  - shared workspace reuse

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

- substantially complete for inference

Already done:

- define a backend-generic `GpuExecutableNetwork`
- mirror CPU dispatch for:
  - convolutional
  - maxpool
  - route
  - shortcut
  - upsample
  - yolo passthrough staging
- keep the GPU path structurally aligned with the CPU runtime for the supported subset
- retain compact-forward coverage for:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- add parity and runtime tests for supported GPU layer slices

Tasks:

- keep extending coverage to the remaining unsupported runtime layers as needed
- reduce unnecessary host/device round-trips in the training path
- add explicit workspace allocation/reuse rules

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

- in progress, but not yet usable for real detector training

Hard prerequisite before training work:

- keep GPU backend startup working on the target machine
- keep parity checks passing for the supported forward path
- maintain persistent device weight storage and runtime parameter caching

Stage 1: Training-Capable Runtime

Done or mostly done:

- extend `GpuExecutableNetwork` with retained forward state for backward
- keep forward activations needed by backward instead of discarding them after each layer
- define device-side parameter bundles for:
  - weights
  - biases
  - gradients

Still missing:

- explicit train/eval mode split where behavior differs
- optimizer state beyond basic SGD momentum buffers
- workspace/lifetime rules for activations, gradients, and scratch buffers
- preserving GPU momentum buffers across iterations the way C++ does

Stage 2: Backward Ops

Done for the currently supported subset:

- convolutional
- activation
- shortcut
- route
- upsample
- maxpool
- CPU-reference checks on synthetic tensors for the supported subset

Still missing:

- broader layer coverage beyond the current subset
- more end-to-end backward parity on real detector slices
- batch-norm training semantics beyond folded-scale handling

Stage 3: YOLO Losses And Targets

- port target assignment and anchor matching
- port objectness, class, and box regression losses
- support both 2-head and 3-head tiny layouts used by:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- verify loss values and per-head gradients against the CPU/original Darknet path

Current blocker:

- the training loop still seeds backward with placeholder `1.0` output gradients, so this stage is the main gap between “weight updates happen” and “real detector training”

Stage 4: Optimizers And Updates

Done:

- CPU SGD with momentum, decay, and LR scheduling
- GPU SGD-style in-place parameter update primitive

Still missing:

- preserve GPU momentum buffers across iterations instead of recreating them
- EMA support
- broader optimizer/state coverage only if needed after Darknet-compatible SGD is correct
- cleaner mixed host/device bookkeeping for logging and checkpointing

Stage 5: Data And Training Loop

Done on CPU:

- dataset loading and label parsing
- batching, shuffling, and epoch iteration
- training-time resize/letterbox behavior
- CLI support for training runs and checkpoint saves

In progress:

- GPU-backed `detector train` entry that attempts to initialize cutile first

Still missing:

- stable GPU execution on the target machine
- real YOLO loss integration into the train loop
- augmentation only after non-augmented training is correct

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

1. fix cutile/CUDA backend startup on the target machine so `detector train` stays on GPU instead of falling back to CPU
2. preserve GPU `*_updates` buffers across iterations so momentum matches the C++ training scheme
3. port real YOLO target assignment and loss instead of using placeholder backward seeds
4. add training-time batch-norm running mean/variance updates
5. add explicit shared workspace allocation/reuse rules for CPU/GPU training scratch

This order removes the current operational blocker first, then closes the biggest remaining correctness gaps for real detector training.

## Non-Goals For Now

Do not spend time on these until inference is solid on `yolov4-tiny` and `yolov4-tiny-3l`:

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
- how closely should the Rust runtime mirror the C++ `workspace` allocation strategy before optimization work starts?
