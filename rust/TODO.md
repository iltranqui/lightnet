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
  - avgpool
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
- Rust YOLO training-loss parity for:
  - best-anchor assignment
  - `truth_thresh` positive matching
  - extra positive anchors via `iou_thresh` / `iou_thresh_kind`
  - focal loss
  - label smoothing
  - `counters_per_class` -> class multipliers
  - `map=...` class remapping
  - objectness smoothing
  - max-delta clipping
  - multi-anchor box-delta averaging
  - persistent YOLO training-state post-processing for:
    - `badlabels_rejection_percentage`
    - `num_sigmas_reject_badlabels`
    - `equidistant_point`
    - `delta_rolling_max`
    - `delta_rolling_avg`
    - `delta_rolling_std`
    - `badlabels_reject_threshold`
- optional `cutile-rs` bridge crate
- typed `f32` tensor upload, zero-init, download, vector-add, and saxpy on the bridge
- bridge implementation of `darknet_cuda::F32TensorOpsBackend`
- GPU bridge support for:
  - convolutional
  - activation backward
  - route backward
  - maxpool
  - maxpool backward
  - avgpool
  - avgpool backward
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
  - momentum-preserving GPU update buffers across iterations
  - batch-normalized convolution scale-gradient accumulation and scale updates
  - batch-norm training-time running-stat updates
  - GPU batch-norm running-stat updates from retained forward state, independent of gradient-seed reachability
  - GPU global avgpool forward/backward through the backend trait
  - CPU/GPU train-buffer parity for a small YOLO-loss detector slice
- workspace tests covering config parsing, weights inspection, and basic runtime execution

Still missing on the CUDA side:

- a working cutile/CUDA runtime on the target machine
- a shared C++-style reusable workspace buffer for CPU/GPU training scratch
- `local_avgpool` GPU backward is wired and the cutile bridge has a kernel, but the kernel
  is a CPU-side fallback (download → loop → upload); replace with a proper GPU kernel
- parity checks against the original C Darknet path on realistic end-to-end detector slices

Still missing on the Rust training-parity side:

- parity checks for the persistent YOLO training-state post-processing against the original C++ Darknet path on realistic detector slices
- logging/diagnostics for YOLO bad-label rejection thresholds, if needed for training-curve comparison

Explicitly out of scope for this TODO item:

- C++ adversarial training mode parity

## Backpropagation Status

### Layer-by-layer backward coverage (as of 2026-04-26)

| Layer | Forward | CPU backward | GPU backward | Notes |
|-------|---------|-------------|-------------|-------|
| convolutional | ✅ | ✅ | ✅ | weights + biases + BN scales; activation grad applied |
| maxpool | ✅ | ✅ | ✅ | gradient routed to max-activated positions |
| local_avgpool | ✅ | ✅ | ⚠️ | GPU call wired; cutile kernel is CPU-fallback — needs real GPU kernel |
| avgpool | ✅ | ✅ | ✅ | uniform gradient distribution |
| route | ✅ | ✅ | ✅ | splits gradient back to source layers |
| shortcut | ✅ | ✅ | ✅ | activation grad + accumulates to both paths |
| upsample | ✅ | ✅ | ✅ | forward and reverse (`spec.reverse`) modes |
| yolo / region | ✅ | passthrough | passthrough | loss layers — gradient seeded externally, no param update |
| dropout | ✅ | ⚠️ wrong | ⚠️ wrong | passthrough ignores mask; incorrect during training |
| scale_channels | ✅ | ⚠️ wrong | ⚠️ wrong | passthrough; should route grad to both input and source layer |
| sam | ✅ | ⚠️ wrong | ⚠️ wrong | same issue as scale_channels |
| connected | ✅ | ❌ | ❌ | throws `Error::UnsupportedLayer`; weight/bias/BN grad missing |
| reorg | ✅ | ❌ | ❌ | throws `Error::UnsupportedLayer`; inverse permutation needed |
| rnn / lstm / crnn | ✅ | ❌ | ❌ | throws `Error::UnsupportedLayer`; out of scope for YOLO-tiny |

### What "full backpropagation" still needs

For the primary YOLO-tiny targets (`yolov4-tiny.cfg`, `yolov4-tiny-3l.cfg`) backprop is
**functionally complete** — neither config uses `connected`, `reorg`, or recurrent layers.

For general detector training parity with C++ Darknet the remaining gaps are:

1. **`connected` backward (CPU + GPU)** — highest priority for non-tiny detectors
   - CPU: implement `backward_connected()` mirroring `backward_convolutional()` structure
     (weight grad via outer-product, bias grad via row-sum, input grad via weight-transpose matmul)
   - GPU: add `backend.connected_backward_f32()` trait method and `cutile` kernel
   - Add `LayerTrainBuffers::Connected` variant with `weights_updates` and `biases_updates`
   - Wire into `sgd_update()` alongside the existing convolutional path

2. **`local_avgpool` GPU backward** — needed for full GPU training on configs that use it
   - Backend trait already has `local_avgpool_backward_f32()` for GPU
   - The `cutile` bridge kernel needs to be written and wired

3. **`reorg` backward** — low priority; not used by YOLO-tiny targets
   - Inverse of the space-to-depth permutation done in forward

4. **End-to-end gradient parity checks** against C++ Darknet on realistic detector slices
   - Per-layer numerical gradient checks (finite-difference vs analytic)
   - Full-network backward on a single batch with fixed seed
   - Tolerance: ~1e-4 relative for fp32

5. **`BackwardResult` / gradient accumulation coverage** for connected and reorg
   - `param_gradients: Vec<LayerParams>` must include connected-layer entries
   - Ensure `layer_output_gradients` routing is correct when connected layers sit in-path

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
- YOLO training-state parity for bad-label rejection and equidistant-point filtering
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

Stage 2: Backward Ops

Done for the currently supported subset:

- convolutional
- activation
- shortcut
- route
- upsample
- maxpool
- batch-normalized convolution scale gradients
- batch-norm training-time running mean/variance updates
- CPU-reference checks on synthetic tensors for the supported subset

Still missing:

- broader layer coverage beyond the current subset
- backward parity against the original C Darknet path on real detector slices

Stage 3: YOLO Losses And Targets

- target assignment and anchor matching are ported on the Rust CPU side
- objectness, class, and box regression losses are ported on the Rust CPU side
- both 2-head and 3-head tiny layouts are supported by:
  - `cfg/yolov4-tiny.cfg`
  - `cfg/yolov4-tiny-3l.cfg`
- still verify loss values and per-head gradients against the original Darknet path

Stage 4: Optimizers And Updates

Done:

- CPU SGD with momentum, decay, and LR scheduling
- GPU SGD-style in-place parameter update primitive
- GPU convolution update buffers preserve momentum across iterations
- GPU batch-normalized convolution scales update through the same SGD path

Still missing:

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

## Phase 8: Full Backpropagation

Objective:
close every remaining gap so that any config the forward path can execute can
also train end-to-end.  The YOLO-tiny targets already work; this phase covers
the remaining layer types and corrects silent errors in layers currently treated
as passthrough.

### 8-A: `connected` layer backward (CPU)

The forward exists at `forward_connected` (runtime.rs:3224).
`ConnectedTrainBuffers`, `LayerTrainBuffers::Connected`, and the SGD update path
in `update_weights` all exist and are wired.  Only the backward compute is
missing — `backward_seeded_core` returns `Error::UnsupportedLayer` for it.

Steps:

1. Add `fn backward_connected(spec, params, input, output_shape, output_gradient, activation_input, workspace)
   -> Result<(Tensor, ConnectedParams)>` mirroring the signature of `backward_convolutional`.

   Math (per batch item `b`, flat input length `n`, output length `m`):
   - apply activation gradient: `delta[b, j] = output_grad[b, j] * act'(pre_act[b, j])`
   - weight grad: `weight_grad[j * n + i] += delta[b, j] * input[b, i]`
   - bias grad: `bias_grad[j] += delta[b, j]`
   - input grad: `input_grad[b, i] += weight[j * n + i] * delta[b, j]`
   - batch-norm path: same chain-rule as convolutional — scale grad accumulates,
     `grad_value *= scales[j] * inv_std` before the weight/input accumulation

2. Wire the result into `backward_seeded_core` under `LayerKind::Connected(_)`.
   Replace the current `Error::UnsupportedLayer` arm with a call to
   `backward_connected` and route the returned `ConnectedParams` gradient through
   `accumulate_connected_param_gradients` (add a new helper analogous to
   `accumulate_convolutional_param_gradients`).

3. Extend `compute_cpu_workspace_size` (runtime.rs:185) to account for connected
   layers: `spec.output + spec.output * spec.inputs` (bias + weight scratch).
   Connected layers currently contribute 0 to workspace sizing, which is wrong
   once they start writing into `workspace`.

4. Extend `update_convolution_batch_norm_running_stats` to handle
   `LayerKind::Connected` with `batch_normalize=true`.  Currently it only
   iterates conv layers; connected-BN running mean/variance will not update
   otherwise.

5. Add unit tests in the existing `#[cfg(test)]` block:
   - single connected layer, no BN, check input grad via finite-difference
   - single connected layer, with BN, same check
   - weight grad sign matches expected outer-product direction

### 8-B: `connected` layer backward (GPU)

The GPU backward (`backward_accumulate_seeded_gpu`) hits the catch-all
`Error::UnsupportedLayer` arm for connected layers.

Steps:

1. Add a `connected_backward_f32` method to the `F32TensorOpsBackend` trait
   (`darknet-cuda/src/lib.rs`).  Signature should return `(input_grad, weight_grad, bias_grad)`.

2. Implement it in `CutileBackend` (`bridges/darknet-cuda-cutile/src/lib.rs`).
   The first implementation may use a CPU-side fallback (download → compute → upload)
   identical to the approach used by `local_avgpool_backward_f32` in that bridge.
   A proper GPU gemm-based kernel can replace it later.

3. Wire the call into `backward_accumulate_seeded_gpu`, accumulating gradient
   tensors into device train buffers via the existing
   `accumulate_device_train_buffer` helper (which currently only handles
   convolutional owners — extend it to also accept connected owners).

4. Add a fake-backend test covering a connected backward dispatch, analogous to
   the existing `gpu_runtime_runs_conv_layer` tests.

### 8-C: `reorg` backward

`forward_reorg` (runtime.rs:3971) is a space-to-depth permutation.  Its inverse
is simply `forward_reorg` with `reverse` flipped.

Steps:

1. Add `fn backward_reorg(stride, reverse, output_gradient, input_shape) -> Result<Tensor>`:
   call `forward_reorg(stride, !reverse, output_gradient, input_shape)`.

2. Wire into `backward_seeded_core` under `LayerKind::Reorg(_)`.

3. For GPU backward add a `reorg_backward_f32` backend method.  Initial
   implementation: CPU fallback (same as the forward kernel which is already
   CPU-side in the bridge).

4. Wire into `backward_accumulate_seeded_gpu` under `LayerKind::Reorg(_)`.

5. Unit test: roundtrip — `backward_reorg(forward_reorg(x)) ≈ x`.

### 8-D: `ScaleChannels` and `Sam` backward

Both layers perform an elementwise multiply between the current path and a
source layer's output.  The current passthrough backward is incorrect —
gradients should be routed to both paths.

`ScaleChannels` forward: `output[i] = input[i] * source[i]` (then activation).
`Sam` forward: identical structure.

Steps:

1. Implement `backward_scale_channels` and `backward_sam` functions.
   For each element `i`:
   - activation gradient applied first (use `apply_output_activation_gradient`)
   - `input_grad[i] = delta[i] * source[i]`
   - `source_grad[i] = delta[i] * input[i]`
   Both gradients must be accumulated: input_grad into the previous layer's
   gradient slot, source_grad into the source layer's gradient slot.

2. Replace the passthrough arm in `backward_seeded_core` for these two variants.
   The forward pass does not currently cache the pre-activation tensor for these
   layers — add it to `layer_activation_inputs` if needed (same pattern as
   shortcut).

3. Same fix in `backward_accumulate_seeded_gpu`.  The backend already has
   `activation_backward_f32` and `vector_add_f32`; the elementwise multiply can
   reuse `saxpy_f32` (alpha=1, accumulate) after computing the partial grads.

4. Unit tests: verify that gradient reaches both upstream paths.

### 8-E: `Dropout` backward

Dropout is currently passthrough in both CPU and GPU backward.  During training
the forward pass should store the binary mask and backward should apply it.

Steps:

1. Add a `dropout_mask: Option<Vec<bool>>` field to `ForwardResult` (or store
   it inside `layer_activation_inputs` as a packed tensor with 0.0/1.0 values).

2. Modify `forward` to generate and store the mask when `probability` is `Some`.
   During inference (non-training forward) keep the current pass-through behavior.

3. Modify `backward_seeded_core` to multiply the gradient by the stored mask and
   scale by `1 / (1 - probability)`.

4. Note: `yolov4-tiny` does not use dropout, so this is low priority for the
   primary target.  Implement after connected backward is stable.

### 8-F: Numerical gradient check harness

Without a reference implementation on the same machine, finite-difference checks
are the fastest way to catch errors in new backward implementations.

Steps:

1. Add a helper `fn finite_difference_input_gradient(net, input, seeds, eps) -> Tensor`
   that perturbs each element of `input`, runs `forward`, computes loss from
   `seeds`, and returns the finite-difference estimate.

2. Add a helper `fn finite_difference_weight_gradient(net, layer_index, input, seeds, eps) -> LayerParams`
   for weight gradients.

3. Use both in tests for every newly implemented backward pass (connected, reorg,
   scale_channels, sam, dropout).

4. Tolerance: relative error < 1e-3 for fp32 at eps=1e-4.

### 8-G: Update Phase 7 Stage 2 to reflect actual gaps

The following items in Stage 2 are still open after Phase 8:

- `local_avgpool` GPU kernel is CPU-fallback; replace with a real CUDA kernel
  once the broader training path is stable
- `ScaleChannels` and `Sam` GPU backward (step 8-D-3 above)
- `Dropout` GPU backward mask application
- Connected GPU backward beyond the CPU-fallback first pass (step 8-B-2)

## Suggested Immediate Next Steps

Recommended short-term sequence:

1. Fix cutile/CUDA backend startup on the target machine so `detector train` stays on GPU instead of falling back to CPU.
2. Implement `backward_connected` CPU (Phase 8-A) — highest-impact correctness gap for non-tiny detectors.
3. Extend `compute_cpu_workspace_size` and `update_convolution_batch_norm_running_stats` for connected layers.
4. Add numerical gradient check harness (Phase 8-F) before extending to GPU.
5. Fix `ScaleChannels` and `Sam` backward (Phase 8-D) — currently silent wrong gradients when these layers are present.
6. Implement `reorg` backward (Phase 8-C).
7. Wire connected GPU backward (Phase 8-B).
8. Add explicit shared workspace allocation/reuse rules for CPU/GPU training scratch.
9. Verify Rust YOLO-loss gradient seeds against the original Darknet path on realistic detector slices.
10. Run short overfit checks once the target machine can keep `detector train` on GPU.

This order fixes silent correctness bugs first (connected, scale_channels), then
extends coverage to GPU, then closes the remaining operational blocker.

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
