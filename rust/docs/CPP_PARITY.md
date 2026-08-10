# C++/Rust CPU parity tests

The Rust unit tests cover graph parsing, individual layer behavior, gradient
accumulation, recurrent updates, weights round trips, and GPU fallback routing.
This document describes the separate differential test that executes the C++
implementation itself.

## What is compiled

`crates/darknet-core/tests/cpp_parity.rs` copies the repository's real
`src-lib/connected_layer.cpp` into a temporary build directory and compiles it
with:

- `tests/cpp_compat/darknet_internal.hpp`: a narrow CPU compatibility surface;
- `tests/cpp_compat/gemm.hpp`: the scalar GEMM declaration;
- `tests/cpp_compat/oracle_main.cpp`: deterministic input/output protocol.

The compatibility surface exists only to avoid pulling OpenCV, CUDA, and the
full Darknet executable into a unit test. The connected-layer source remains
the C++ code under test; the shim supplies the `Layer`, `NetworkState`, CPU
helper functions, and callback declarations that this translation unit reads.

This is stronger than writing the same equation twice in Rust and C++: a
change to `src-lib/connected_layer.cpp` changes the executable used by the
test.

## Compared values

For each activation case, Rust and C++ receive identical:

- batch size `2`;
- flattened input width `4`;
- output width `3`;
- input tensor values;
- `[output][input]` weights;
- biases;
- output-gradient seed.

The test compares:

- forward output;
- post-activation output delta;
- input gradient;
- weight gradient;
- bias gradient.

The activation matrix currently includes `linear`, `logistic`, `relu`,
`leaky`, `tanh`, and `loggy`. This exercises both the matrix path and the
activation derivative path in the actual C++ layer.

The current direct fixture intentionally uses `batch_normalize=0`. Batch norm,
convolution, recurrent layers, and detector target assignment remain covered
by Rust tests and should receive additional direct C++ fixtures as their
minimal compatibility surfaces are added.

## Running it

From `darknet/rust`:

```bash
source ../toolchain.env
RUSTFLAGS=-Dwarnings cargo test -p darknet-core --test cpp_parity -- --nocapture
```

The test discovers `c++`, `g++`, or `clang++`. Use explicit paths when the
compiler is not on `PATH`:

```bash
DARKNET_CXX=/usr/bin/g++ cargo test -p darknet-core --test cpp_parity
```

If the Rust crate is copied away from the repository root, point it at the C++
tree explicitly:

```bash
DARKNET_CPP_ROOT=/path/to/darknet cargo test -p darknet-core --test cpp_parity
```

Without a C++ compiler, the test reports a skip so ordinary CPU-only Rust
development remains portable. Set `DARKNET_CPP_PARITY_REQUIRED=1` in CI or a
release gate to turn that skip into a failure.

## Numerical policy

The assertion uses an absolute-plus-relative tolerance of
`2e-5 + 2e-5 * abs(reference)`. The oracle is compiled without fast-math so
the comparison catches layout/sign/gradient mistakes while allowing normal
compiler/libm last-bit differences for sigmoid and tanh.

When a parity case fails, run with `--nocapture`. The test prints the compiler
diagnostic if the actual C++ source no longer compiles against the shim, and
the assertion identifies the array, element, C++ value, Rust value, and
tolerance.

## Adding another direct fixture

For another C++ layer:

1. identify the smallest production `.cpp` translation unit and the exact
   fields/helpers it reads;
2. extend or add a compatibility header rather than modifying production
   headers;
3. make the oracle protocol deterministic and text-readable;
4. compare forward, backward, and parameter/update state separately;
5. document unsupported branches explicitly;
6. run the fixture with `DARKNET_CPP_PARITY_REQUIRED=1`.

Never silently compare only final detections for a new trainable layer. A wrong
intermediate layout can still produce plausible boxes while its gradients are
wrong.
