# runner — kernel tests and benchmarks

One job: prove kernels correct on the GPU and measure them.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `build.rs`: builds and embeds the kernels ELF (`HIP_TILE_KERNELS_ELF`), `link_rocm()`.
- `src/lib.rs`: `load()`, `random_vec`, `launch_gemm`, `max_rel_err_vs_cpu`.
- `src/main.rs`: `runner saxpy [--n N]`, `runner gemm [--n N] [--bench]`, `runner conv-bench`.
- `tests/`: `saxpy.rs`, `memory.rs` (copy, transpose), `gemm.rs`, `rocblas.rs`, `yolo_ops.rs` — 28 tests, all on the GPU.

## Dependencies

- Depends on: `hip-rt` (feature `rocblas`), `hip-tile`, `kernels`, build-dep `hip-tile-build`.

## Rules for changes

- Read `_shared/rules.md` first.
- New parity tests start from `_templates/kernel/parity_test.rs`.
- Benchmarks follow `_shared/benchmarking.md` (warm-up, best of N, idle CPU).

## Human check

`systemd-run --user --scope -q -p MemoryHigh=3G cargo test -p runner --release -j 4`: every `test result` line says `ok`.
