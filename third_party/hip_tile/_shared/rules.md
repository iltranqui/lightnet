# Rules

The hard rules for every change in this repo. They apply to every workflow stage; nothing overrides them except the user.

## Contents

- [Code](#code)
- [Verification](#verification)
- [Builds](#builds)
- [Environment](#environment)

## Code

- **Pure Rust, end to end.** Kernels, host code, tests, benches and tools are Rust. The only allowed non-Rust dependency is the HIP runtime library (`libamdhip64.so`), plus rocBLAS as a test and benchmark reference. No C/C++, no `hipcc`, no bindgen.
- **No inline assembly.** Never use `asm!` in `crates/hip-tile` or `kernels`. Let rustc and LLVM generate AMDGCN. Use `core::arch::amdgpu` intrinsics and `core::sync::atomic::fence` instead (see [tile-model.md](tile-model.md#barrier)).
- **Kernel bodies use the tile API** (`load_tile`, `Tile::from_fn`, `map`, `mma`, `store_tile`). Per-thread scalar code only for inherently elementwise ops.
- **Model ports are direct Rust ports.** Never run `/usr/bin/darknet` or `darknet_*` binaries, not even as a test reference. Reading Darknet's C++ source (`~/src/DarkSuite/darknet/src-lib`) to port its semantics is how ports are done.
- **One home per fact.** Settled technical facts live in `_shared/`; contracts link to them instead of copying.

## Verification

- A task is done only when its tests ran on the GPU and passed, with the output shown.
- GPU results are compared against an f64 (or f32 naive) CPU reference in Rust; model results also against the dataset's ground-truth label files.
- Performance claims follow [benchmarking.md](benchmarking.md).

## Builds

- At most 4 jobs: `-j 4` (also set in `.cargo/config.toml`). Never bare `-j`.
- Wrap every cargo build, test, run, check or clippy in a soft memory cap: `systemd-run --user --scope -q -p MemoryHigh=3G cargo ...`.
- Toolchain: `rust-toolchain.toml` pins `nightly-2026-10-25` with `rust-src`. Device builds need `-Ctarget-cpu=<gfx>`; `hip-tile-build` passes it (default `gfx1031`, override `HIP_TILE_ARCH`).
- Commit only when the user asks.

## Environment

- GPU: Radeon 680M iGPU (Ryzen 7 7735HS), reported by HIP as `gfx1031`, wave32, 12 CUs (HIP prints 6 WGPs). WSL2, ROCm 10.1 at `/opt/rocm` (override `ROCM_PATH`).
- Never run `find /` on WSL: it walks `/mnt/c` and hangs. Search `/opt/rocm`, `/usr`, or the repo.
- `sudo` cannot run through the `!` prompt prefix; give the user the command to run in their own terminal.
- zsh does not word-split `$var`; use bash for scratch scripts that rely on it.
