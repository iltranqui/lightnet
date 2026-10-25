# docs

Reference shelf: stable explanations and benchmark evidence. Plans and status live in `tasks/rust/`, not here.

## Table of Contents

- [Contents](#contents)
- [Who cites what](#who-cites-what)
- [Constraints](#constraints)

## Contents

| File | What it holds |
|---|---|
| `ARCHITECTURE.md` | How the Rust port is organised |
| `CPP_PARITY.md` | C++/Rust CPU parity tests |
| `RUST.md` | Rust port overview |
| `OXIDE_BENCH.md` | cuda-oxide benchmark and convergence evidence |
| `windows_cudnn_paths.md` | Windows cuDNN extraction and path setup |

The layer layout contract is not here: it lives in `src/lightnet-core/src/layers/CONTEXT.md`, with the per-layer file map in `tasks/rust/layer-map.md`.

## Who cites what

`OXIDE_BENCH.md` is cited by `tools/oxide_parity_*.py`, `src/lightnet-cli/src/main.rs`, `src/lightnet-cuda-oxide/README.md`, `src/lightnet-cuda-oxide/tests/oxide_amp.rs` and `tasks/rust/implemented.md`. Do not rename or split it without updating those.

## Constraints

- One home per fact: status goes in `tasks/rust/STATUS.md`, not here.
- Each `.md` here needs a table of contents.
