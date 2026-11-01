# kernels — all GPU kernels

One job: hold every `#[kernel]`, built for the device (ELF) and for the host (launchers).

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `src/lib.rs`: module list; `#![no_std]`.
- `saxpy.rs`, `copy.rs`, `transpose.rs`, `gemm.rs` (128×128×16).
- `yolo/`: `conv.rs` (implicit GEMM, 5 tile configs, `Config::choose`, `ConvParams`), `maxpool.rs` (`PoolParams`), `upsample.rs`, `concat.rs`.
- `backward/`: training gradients — `util.rs` (zero/axpy/acc, `SgdParams` + `sgd_update`, `running_update`), `maxpool_bwd.rs`, `upsample_bwd.rs`, `route_bwd.rs`, `bn_stats.rs`, `bn_bwd.rs`. Gradient kernels accumulate (`store_tile_add`), never overwrite.
- Each kernel's record: `workflows/add-kernel/runs/<name>/`.

## Dependencies

- Depends on: `hip-tile`, `hip-tile-macros`; host side also `hip-rt`.
- Built by: `hip-tile-build` (from `apps/runner` and `hip_darknet`, path `../../kernels`; do not move this folder).
- Used by: `apps/runner`, `hip_darknet`.

## Rules for changes

- Read `_shared/rules.md` first.
- New kernels go through `workflows/add-kernel/`; start from `_templates/kernel/kernel.rs`.
- Tile ops only, no `asm!`; kernel symbol names unique; `#[kernel]` cannot be generic (use `macro_rules!`).

## Human check

The kernel has a passing parity test in `apps/runner/tests` and a run folder (`backward` kernels: `apps/runner/tests/backward_ops.rs`).
