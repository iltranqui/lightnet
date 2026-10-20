# lightnet

A Rust port of the Darknet/YOLO runtime. The CPU path is the behavioural reference; GPU backends (cuTile and cuda-oxide) add native kernels layer by layer. It reads Darknet `.cfg`, `.names` and `.weights` files and covers parsing, inference, training, detector loss, YOLO decoding and NMS.

## Table of Contents

- [Status](#status)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Build and run](#build-and-run)
- [Tests](#tests)
- [Documentation](#documentation)
- [License](#license)

## Status

Work in progress. Per-layer CPU, GPU, f16 and axis status is tracked in `tasks/rust/layer-support.md`; the overall roadmap is in `tasks/rust/`. No parity with the C++ implementation is claimed beyond what `docs/CPP_PARITY.md` describes.

## Repository layout

| Path | Contents |
|---|---|
| `src/darknet-core` | Layers, cfg parsing, weights, training, evaluation |
| `src/darknet-cuda` | Backend-neutral GPU tensor traits |
| `src/darknet-cuda-cutile` | Optional cuTile GPU backend (not a workspace member) |
| `src/darknet-cuda-oxide` | Optional cuda-oxide GPU backend (not a workspace member) |
| `src/darknet-cli` | The `darknet-rs` command-line binary |
| `src/darknet-examples` | Rust equivalents of the Darknet example programs |
| `src/darknet-gui*` | Annotation GUI (opt-in) |
| `bridges/` | Vendored [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) crates |
| `cfg/` | Model configs and dataset descriptors |
| `tools/` | Test, benchmark and parity scripts |
| `docs/` | Architecture, parity and benchmark notes |
| `tasks/rust/` | Roadmap, plans and status |

Each folder has a `CONTEXT.md` describing its role; start from `CLAUDE.md` and `CONTEXT.md` at the root.

## Requirements

- Rust nightly pinned in `rust-toolchain.toml` (installed automatically by `rustup`).
- For GPU backends: CUDA 13.2 or newer (the repo defaults to `/usr/local/cuda-13.4`, overridable through `CUDA_TOOLKIT_PATH`) and libclang. See `src/darknet-cuda-cutile/README.md` and `src/darknet-cuda-oxide/README.md`.

## Build and run

CPU build of the command-line tool:

```
cargo build --release -p darknet-cli -j 4
./target/release/darknet-rs version
./target/release/darknet-rs cfg-validate cfg/<model>.cfg
./target/release/darknet-rs detector train <cfg> <data>
./target/release/darknet-rs detector map <cfg> <data>
```

Other subcommands include `cfg-summary`, `names` and `weights-inspect`; see `src/darknet-cli/src/main.rs`. The annotation GUI is opt-in: `cargo build -p darknet-gui`.

The cuda-oxide backend is built only through `./tools/test_oxide.sh`.

## Tests

```
./tools/test_host.sh
```

Use at most 4 build threads. GPU tests need a CUDA device and run single-threaded; see `src/darknet-cuda-oxide/README.md`.

## Documentation

- `docs/ARCHITECTURE.md`: how the C++ runtime maps to the workspace.
- `docs/CPP_PARITY.md`: C++/Rust CPU parity tests.
- `docs/OXIDE_BENCH.md`: cuda-oxide benchmark evidence.
- `tasks/rust/AGENTS.md`: rules, commands and versioning for contributors and agents.

## License

Apache-2.0, see `LICENSE`. Vendored cuda-oxide code keeps its own notices in `LICENSES/`.
