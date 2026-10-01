# darknet-cuda-oxide

cuda-oxide (Rust SIMT kernels) backend for the Darknet Rust port. Excluded from the root workspace; it
is built through the vendored `cargo-oxide` driver, which sets up the custom rustc codegen backend.

## Building and testing

Use the pinned nightly and build the driver once from the repo root:

```
export RUSTUP_TOOLCHAIN=nightly-2026-08-28
cargo build -p cargo-oxide                      # -> target/debug/cargo-oxide
```

* `cargo oxide ...` (the alias in `.cargo/config.toml`) does NOT work from this directory: the alias
  resolves relative to the workspace root and this crate is excluded from it. Call the binary and pass the
  manifest instead, from the repo root:

  ```
  target/debug/cargo-oxide oxide test  -- --manifest-path bridges/darknet-cuda-oxide/Cargo.toml -j 1 -- --test-threads=1
  target/debug/cargo-oxide oxide build -- --release -p darknet-cli --features oxide -j 1
  ```

* Run the test binaries with `--test-threads=1` (one CUDA context, large buffers).
* Never more than 4 build threads (`-j 4` max); `-j 1`/`-j 2` is preferred because codegen is memory
  hungry. Run one build/test command at a time under a soft cap:
  `systemd-run --user --scope -p MemoryHigh=2G target/debug/cargo-oxide oxide test -- ...`.
  `lib` and `lib --test` used to write the same `darknet_cuda_oxide*.ll/.ptx` files and raced (SIGBUS in
  `opt` with `-j > 1`); the test unit now uses `darknet_cuda_oxide_test*` names
  (`crates/rustc-codegen-cuda/src/lib.rs`) and `-j 4 --test oxide_parity` passed (2026-10-01).
* One suite: add `--test oxide_parity` after `-j N`, e.g.
  `... -- --manifest-path bridges/darknet-cuda-oxide/Cargo.toml -j 4 --test oxide_parity -- --test-threads=1`.
* The generated `darknet_cuda_oxide*.ll` / `.ptx` files in this directory are build products.

## Environment knobs

* `DARKNET_OXIDE_GEMM={fp32,tf32,fp16,auto}` - GEMM precision/path.
* `DARKNET_OXIDE_AMP=1` - opt-in mixed-precision training (implies the fp16 GEMM): f32 master weights/optimizer, f16 activations and gradients, f32 accumulation (see `src/mixed.rs`; default off).
* `DARKNET_OXIDE_CONV={direct,gemm,auto}` - convolution algorithm.
* `DARKNET_OXIDE_CONV_CHUNK=<n>` - max images per batched im2col + GEMM launch (`1` = old per-image loop, for A/B).
* `DARKNET_OXIDE_POOL_MB=<n>` - device memory pool size in MB.
* `DARKNET_BACKEND=oxide` selects this backend in `darknet-cli` (built with `--features oxide`).

## Benchmarks

`tests/oxide_bench.rs` and `tests/oxide_gemm_bench.rs` are `#[ignore]`d; run them with
`... -j 1 -- --ignored --nocapture --test-threads=1 bench_micro` (see the header of each bench file for test names).
End-to-end scripts: `tools/oxide_bench.py`, `tools/oxide_map.py` (repo-root `tools/`); results in
`docs/OXIDE_BENCH.md`. `bridges/darknet-cuda-oxide/tools/cfg_conv_shapes.py` dumps cfg conv geometries
(`--crosscheck` compares with the C++ layer table; use `python3 -u`).
