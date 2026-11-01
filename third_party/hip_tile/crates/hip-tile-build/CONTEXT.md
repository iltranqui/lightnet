# hip-tile-build — device build helper

One job: compile the `kernels` crate for the GPU from a `build.rs` and hand back the ELF path.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `src/lib.rs`: `build_kernels(dir, arch)`, `build_kernels_watching(dir, arch, extra)` (nested `cargo rustc --crate-type cdylib -Zbuild-std=core --target amdgcn-amd-amdhsa`, `-Ctarget-cpu`, `-Cpanic=abort`, `-j 4`, scrubbed env, `rerun-if-changed`), `link_rocm()` (rpath).

## Dependencies

- Depends on: the outer cargo (`CARGO` env).
- Used by: `apps/runner/build.rs`, `crates/hip_darknet/build.rs`.

## Rules for changes

- Read `_shared/rules.md` first.
- `HIP_TILE_ARCH` overrides the arch; `ROCM_PATH` the ROCm root.
- Host `RUSTFLAGS`/target env must never leak into the device build.

## Human check

Editing a file in `kernels/src` triggers a device rebuild on the next `cargo build -p runner`.
