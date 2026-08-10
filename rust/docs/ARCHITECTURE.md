# Rust Darknet port architecture

This document explains how the Rust implementation maps the existing Darknet
C++ runtime into a safe, testable workspace. It is intended to be read beside
the C++ tree.

## Design goal

The goal is behavioral compatibility, not a superficial C++ wrapper. The Rust
runtime reads the same `.cfg` and `.weights` formats, uses the same layer order
and tensor layout, computes the same CPU forward/backward contracts, and keeps
the same optimizer and detector-training boundaries.

The implementation does not claim that every ownership detail is identical:

- C++ owns graph objects through raw pointers and callback fields.
- Rust owns graph metadata, parameters, caches, and buffers through values and
  `Vec<f32>`.
- C++ may terminate through fatal logging; Rust returns `Result<T, Error>`.
- C++ GPU execution is selected by global/network state; Rust receives a
  backend object and uses an explicit CPU-reference fallback when a device
  operation is unavailable.

Those are implementation choices. The observable numerical contracts are the
compatibility boundary.

## Workspace map

| Rust path | Responsibility | C++ analogue |
| --- | --- | --- |
| `crates/darknet-core/src/cfg.rs` | Parse raw sections and key/value pairs | `darknet_cfg.cpp/.hpp`, option parsing |
| `crates/darknet-core/src/model.rs` | Infer layer shapes and model metadata | network/layer construction in `darknet_cfg.cpp` |
| `crates/darknet-core/src/runtime.rs` | CPU forward/backward, caches, train buffers, GPU orchestration | layer callbacks and network execution |
| `crates/darknet-core/src/weights.rs` | Read/write binary weights | `weights.cpp/.hpp` |
| `crates/darknet-core/src/train.rs` | Dataset paths, labels, target assignment, detector loss | detector/data/training code |
| `crates/darknet-core/src/detect.rs` | Decode YOLO heads and NMS | YOLO detection helpers |
| `crates/darknet-core/src/image.rs` | RGB loading and planar preprocessing | image loading/letterbox helpers |
| `crates/darknet-cuda` | Backend traits and tensor operation contract | CUDA function boundary |
| `bridges/darknet-cuda-cutile` | cuTile implementation of the backend contract | CUDA kernels exposed as Rust operations |
| `crates/darknet-cli` | CLI commands and training/inference reporting | `darknet` executable commands |
| `crates/darknet-examples` | Rust equivalents of the 11 C++ examples | `src-examples/darknet_*.cpp` |

## C++ to Rust ownership map

| Darknet C++ concept | Rust representation | Why it matters |
| --- | --- | --- |
| `struct layer` | `model::Layer` + `runtime::LayerParams` + `LayerTrainBuffers` | Separates immutable graph metadata from mutable parameters and optimizer state |
| `struct network` | `model::Network` + `ExecutableNetwork` | A parsed graph can be reused by CPU/GPU executors |
| `float *output` | `Tensor { shape, data: Vec<f32> }` | Shape and allocation travel together; no dangling output pointer |
| `float *delta` | `BackwardResult` and train buffers | A backward pass can be inspected before it is accumulated or applied |
| `layer.forward` callback | `match LayerKind` in `forward_with_mode` | Dispatch is exhaustive and visible to the compiler |
| `layer.backward` callback | Reverse `match LayerKind` in `backward_seeded_core` | Reverse graph traversal is explicit |
| `layer.weights` / `biases` | `LayerParams` variants | Nested RNN/LSTM/CRNN parameters retain C++ file order |
| `net.workspace` | `ExecutableNetwork::workspace` | Reused scratch storage avoids per-pass allocations |
| fatal parser error | `Error` returned through `Result` | Library callers control recovery |
| `cuda_*` global operations | `F32TensorOpsBackend` | The core does not depend on one CUDA implementation |

## Tensor layout

All runtime tensors use contiguous planar storage:

```text
offset(batch, channel, y, x) =
    batch * (channels * height * width)
  + channel * (height * width)
  + y * width
  + x
```

This is the layout expected by the C++ Darknet CPU code. A connected layer
flattens one sample to `width * height * channels`; a convolution reads the
same buffer through explicit `(channel, y, x)` indexing. `PlanarRgbImage` is the
only image-boundary type that converts interleaved RGB pixels into this layout.

## Model construction

`ConfigFile::parse_str` first creates raw sections. It does not try to
understand every layer. `Network::from_config` then:

1. parses `[net]` dimensions, batch/subdivision/time-step semantics, and the
   learning-rate policy;
2. walks sections in source order;
3. converts each section to a typed `LayerKind`;
4. infers its `TensorShape` from the current shape and earlier layers;
5. stores the source line and index for diagnostics and route resolution.

The separation is important. Config validation can run without allocating
weights or touching CUDA, and all executors consume the same shape decisions.

## CPU forward pass

`ExecutableNetwork::forward` is deterministic for inference. It creates one
output/cache slot per graph layer and dispatches by `(LayerKind, LayerParams)`.
The caches have a concrete purpose:

- activation inputs are needed for Swish/Mish and other backward derivatives;
- dropout masks preserve the exact inverted-dropout choices made in training;
- recurrent caches store every time-step state for BPTT;
- all layer outputs remain available for route, shortcut, SAM, and backward.

The CPU convolution loop is deliberately straightforward. It is the reference
implementation for grouped channels, padding, stride, and dilation. Native
backends may use a different kernel, but they must produce the same tensor.

Connected weights remain `[output][input]`, matching Darknet's binary file and
the transposed GEMM in `connected_layer.cpp`.

## CPU backward pass

`backward_seeded` accepts a gradient seed for any layer. `backward` is the
convenience form that seeds the final layer. The implementation then walks the
graph in reverse:

1. validate saved forward state and seed shapes;
2. allocate parameter-gradient slots with the same layer ordering;
3. apply the layer's output activation derivative;
4. compute input and parameter gradients;
5. add the input gradient to the previous layer's gradient slot;
6. accumulate fan-in contributions instead of overwriting them.

This last rule is essential. A route or shortcut can feed one earlier layer
from multiple paths, and C++ Darknet accumulates those deltas in place.

Backward returns gradients without changing model parameters. The separate
`backward_accumulate` path merges them into `LayerTrainBuffers`; then
`update_weights` applies Darknet's SGD rule:

```text
update = momentum * update
       + gradient
       - decay * batch * weight
weight += learning_rate / batch * update
```

Biases and batch-norm scales use the same update path without weight decay.
Keeping calculation, accumulation, and update separate is useful for both
training and numerical differential tests.

## Batch normalization and recurrent layers

Batch normalization stores scales and rolling mean/variance in
`BatchNormParams`. Training forward computes batch statistics and updates the
rolling values; inference uses the rolling values. The forward cache preserves
the pre-normalized values needed by backward.

RNN, LSTM, and CRNN are represented as nested parameter blocks. Their caches
store one tensor per time step, so backward-through-time can traverse steps in
reverse without reconstructing hidden states. The nested blocks are also the
reason the weights module has explicit recursive read/write functions: the
serialized order must remain the C++ order.

## GPU execution and fallback

`GpuExecutableNetwork<B>` is generic over a backend implementing
`F32TensorOpsBackend`. It uploads convolution parameters lazily, dispatches
native operations where the backend supports them, and retains a CPU forward
state for correctness and backward bookkeeping.

The fallback is intentional and explicit. It means a network remains
functionally usable before every device kernel exists, but it is not a claim
that the fallback is fast. The cuTile bridge is optional and is not required to
compile or test the CPU port.

## Training output

The CLI training loop emits Darknet-style fields—loss, running average loss,
learning rate, elapsed time, image count, and ETA—at the same iteration-level
boundary as the C++ loop. Formatting lives in the CLI/training layer rather
than tensor code so applications can reuse the runtime without inheriting
terminal output.

## Examples

The 11 Rust example binaries intentionally mirror the C++ filenames. They
share a common argument parser and detector helper, but remain separate
executables so existing scripts can migrate one example at a time. Media
decode/display is delegated to `ffmpeg`, `ffprobe`, and `ffplay`; model
loading, inference, decoding, annotations, and output naming stay in Rust.

## Extension rules

When adding a layer or changing a C++ behavior, update all of these in order:

1. add/extend the typed `LayerKind` and parser;
2. add output-shape inference;
3. add default parameter and train-buffer construction;
4. implement CPU forward and its cache requirements;
5. implement CPU backward and parameter gradients;
6. add weight-file entries if the layer is trainable;
7. add backend dispatch or document the CPU fallback;
8. add unit tests plus a C++ differential fixture when the C++ source is
   available;
9. update this document and `TODO.md` with the exact parity boundary.

Do not add a backend-only implementation first. The CPU path is the reference
against which accelerated kernels and the C++ port are checked.
