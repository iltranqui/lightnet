# hip-tile-macros — #[kernel]

One job: turn one Rust function into a GPU entry point and a typed host launcher.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `src/lib.rs`: `#[kernel(lds = <const expr>)]`. Device: `#[unsafe(no_mangle)] extern "gpu-kernel" fn`; host: `pub mod <fn> { NAME, BLOCK_THREADS, LDS_BYTES, launch(...) }` plus a `KernelArg` check per parameter.

## Dependencies

- Depends on: `syn`, `quote`, `proc-macro2`.
- Used by: `kernels`.

## Rules for changes

- Read `_shared/rules.md` first.
- Generated host code may only use `::hip_rt` and `::core` paths (`kernels` is `no_std`).
- Launcher path shape `kernels::<file>::<fn>::launch` is relied on everywhere; changing it is a cross-crate change.

## Human check

`cargo build -p runner --release` and `cargo test -p runner --release` pass.
