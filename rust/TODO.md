# Rust Darknet Port Status

This file records the remaining work after the CPU runtime, recurrent layers,
training path, and GPU fallback path were completed.

## Completed port work

- C++-style effective mini-batch and recurrent `time_steps` semantics.
- Config parsing compatible with Darknet comments, aliases, list syntax, and
  learning-rate policies.
- Convolutional, grouped convolution, connected, pooling, route, shortcut,
  upsample, reorg, softmax, dropout, scale-channels, SAM, YOLO staging, and
  recurrent RNN/LSTM/CRNN CPU forward paths.
- CPU backward propagation for the same trainable execution path, including
  connected-layer gradients, recurrent BPTT, reorg/shortcut routing,
  dropout masks, scale-channel/SAM source gradients, and optimizer buffers.
- Darknet recurrent weight loading/saving in C++ order.
- CPU detector target assignment, loss, bad-label training-state handling,
  SGD/momentum/decay, scheduling, checkpointing, image inference, YOLO decode,
  and NMS.
- GPU forward/backward state and persistent convolution update buffers.
- Native backend GPU dispatch for the available tensor operations.
- Functional GPU CPU-reference fallback for operations without a native device
  kernel. The fallback preserves device convolution gradients and updates and
  keeps the public GPU runtime API usable for general networks.

## Remaining work

## Direct C++ differential coverage

`tests/cpp_parity.rs` now directly compiles and exercises the repository's CPU
`src-lib/connected_layer.cpp` for six activations and compares forward outputs
plus input/weight/bias gradients. It is an executable C++/Rust differential
gate, not just a second Rust reference formula.

Remaining direct fixtures should be added in this order:

1. connected batch-normalization forward/backward;
2. convolutional forward/backward, including grouped and padded cases;
3. recurrent CPU steps and BPTT;
4. detector target assignment/loss on a fixed tiny batch;
5. serialized weight blocks for every supported layer family.

Run the existing fixture with `DARKNET_CPP_PARITY_REQUIRED=1` when a missing
compiler or source tree must fail the build instead of being a portable-
development skip. See `docs/CPP_PARITY.md` for the protocol.

### Native GPU parity

The GPU API is functionally complete for the supported layer model, but not
every operation is a device kernel yet. The following are the main performance
and deployment tasks:

- replace CPU-reference fallback for connected, reorg, softmax, dropout,
  scale-channels, SAM, recurrent layers, and mismatched-shape shortcuts with
  backend kernels;
- replace local/global average-pool host fallback in the cuTile bridge with
  device kernels;
- reuse a shared C++-style workspace for activation, gradient, and scratch
  allocations instead of allocating per operation;
- run deterministic CPU/GPU parity tests on real detector weights and compare
  intermediate tensors and gradients against the original C++ runtime.

### Optional cuTile environment

The bridge is intentionally outside the main workspace and is pinned to
cutile-rs v0.2.0. The default JIT now uses the pure-Rust cutile-ir, so LLVM 21,
MLIR headers, and Polly are no longer required. CUDA 13.3 is the recommended
upstream target; CUDA 13.2 or newer is suitable for Ada sm_89. The local
workspace toolchain currently provides CUDA 12.9 nvcc for compiler-only checks,
but a CUDA 13.2+ installation is required for the current low-precision path.
Generated CUDA bindings still require CUDA headers and libclang.

### cutile AOT compilation (investigated, not started)

Every fresh process pays a one-time JIT cost the first time each unique tile
kernel shape is used (observed ~108s on batch 1 of LegoGears_v2 training,
amortizing to <1% over a 3000-batch run). Root cause confirmed by reading the
vendored `cutile-rs` v0.2.0 source (commit `de6c5cd`, pinned in
`bridges/darknet-cuda-cutile/Cargo.toml`):

- `cutile::tile_kernel::compile_from_context` caches compiled kernels
  **thread-local / in-process only**, keyed by module+function name, type
  generics, stride args, and compile-time grid dims. There is no on-disk
  cubin cache, so every new process recompiles every kernel shape it touches
  from scratch via `tileiras`.
- `tileiras` (`cuda-13.3/bin/tileiras`) is a genuine *offline* assembler
  (`tile bytecode file -> cubin`), not an in-process JIT like `nvrtc`. The
  "JIT" behavior is a policy choice in `cutile-rs`'s Rust wrapper (compile
  lazily on first kernel use), not a hard requirement of the CUDA toolchain.
- CUDA 13.3's `nvcc` already exposes AOT-oriented flags —
  `--enable-tile`, `--tilebc`, `--tilecubin`, `--tilefatbin`,
  `-Xtileiras` (verified via `nvcc --help` against our installed
  `/usr/local/cuda-13.3`) — but the pinned `cutile-rs` v0.2.0 tree has zero
  references to any of them; there is no AOT path to opt into today.

Getting real build-time compilation would mean forking `cutile-rs` to move
the existing `compile_from_context` -> `compile_tile_ir_module` pipeline
(currently invoked lazily at first kernel launch) into a `build.rs`, emitting
`include_bytes!()`-embedded cubins per target SM architecture instead of
JIT-compiling at runtime.

The catch specific to this port: kernel tile shapes (the `const S: [i32; N]`
generics on kernels like `bias_add_2d`, `channel_scale_2d`, `saxpy` in
`bridges/darknet-cuda-cutile/src/lib.rs`) are derived from network layer
output dimensions, which come from whatever `.cfg` file is loaded at
*runtime*. `cargo build` has no way to know which cfg a user will pass, so a
naive AOT fork would need to either bake kernels for one fixed, hardcoded
network shape (defeats the point of a general darknet-cfg loader) or keep
kernels on fully dynamic/wildcard shapes throughout (losing the per-shape
`tileiras -O3` specialization, and pushing more code through the
symbolic-shape path that already caused the `bias_add_2d` broadcast-type bug
fixed above). Worth doing only for a fixed, repeatedly-run deployment target,
not for general cfg flexibility — not started.

### Historical configuration families

The repository’s standard YOLO, ResNet, CSP, EfficientNet, and recurrent
configuration files validate. Old or nonstandard sections outside the current
C++ parser’s supported layer registry still need dedicated model semantics if
they are to be accepted rather than reported as `UnsupportedLayer`.

### Behavioral parity

- compare recurrent and connected outputs/gradients against fixed-seed C++
  fixtures;
- compare detector loss and persistent bad-label thresholds on realistic
  batches;
- add checkpoint compatibility fixtures covering every supported layer family;
- add GPU execution tests on a CUDA-capable host once the cuTile prerequisites
  are installed.

## Verification gate

From the Darknet repository root:

```bash
source toolchain.env
cd rust
RUSTFLAGS=-Dwarnings cargo check --workspace --all-targets --locked -j 2
RUSTFLAGS=-Dwarnings cargo test --workspace --all-targets --locked -j 2
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```
