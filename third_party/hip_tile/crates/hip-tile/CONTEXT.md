# hip-tile — the tile library

One job: the cuTile-style programming model used inside kernels; on the host only the kernel ABI types.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- Both targets: `tensor.rs` (`Tensor`, `TensorMut`), `arg.rs` (`KernelArg`), `config.rs` (block size, LDS size helpers), `lib.rs` (`prelude`, panic handler).
- Device only (`target_arch = "amdgpu"`): `arch.rs` (ids, `sync_block`), `element.rs`, `layout.rs` (thread layout rule), `smem.rs` (LDS), `tile.rs` (`Tile`, `map`, `zip`, ops), `ops/{memory,mma,transpose,reduce}.rs`.

## Dependencies

- Depends on: nightly `core` only (`#![no_std]`).
- Used by: `kernels` (device), `hip-rt`, launchers (host ABI types).

## Rules for changes

- Read `_shared/rules.md` first.
- How tiles map to threads, LDS and barriers: `_shared/tile-model.md` (keep it in sync when this crate changes).
- No `asm!`. New ops are general (any valid tile shape), not model-specific.
- A new op needs a parity test via the `add-kernel` workflow (stage 03).

## Human check

`cargo test -p runner --release` passes and the gemm bench did not regress (method: `_shared/benchmarking.md`).
