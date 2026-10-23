# third_party

Pinned git submodules. Treat as upstream code: never edit inside, apply patches from `patches/`.

## Table of Contents

- [Contents](#contents)
- [cuda-oxide crate groups](#cuda-oxide-crate-groups)
- [Who uses it](#who-uses-it)
- [Constraints](#constraints)

## Contents

| Path | Upstream | Pin |
|---|---|---|
| `cuda-oxide/` | https://github.com/NVlabs/cuda-oxide | `728539f65` (first upstream fix after `7d9c19405`; its root Cargo.toml paths are repaired) |
| `patches/` | local | patches to apply on top of the pin, see `patches/README.md` |

The cuTile submodule lives at `src/cutile-rs-aot` (NVlabs/cutile-rs v0.2.0) because `lightnet-cuda-cutile` uses a relative path to it.

Crates are at `third_party/cuda-oxide/cuda-oxide/crates/<name>`.

## cuda-oxide crate groups

| Group | Crates |
|---|---|
| Device and host API | `cuda-device`, `cuda-host`, `cuda-macros`, `cuda-target-spec`, `oxide-artifacts`, `cuda-artifact-finalizer` |
| Compiler front end | `rustc-codegen-cuda` (excluded from the workspace), `mir-importer`, `dialect-mir`, `mir-transforms`, `cuda-oxide-codegen` |
| Lowering and dialects | `mir-lower`, `dialect-nvvm`, `dialect-ptx`, `dialect-iket`, `iket-lower`, `nvvm-transforms`, `llvm-export` |
| PTX tooling | `ptx-parse`, `ptx-schedule` |
| Intrinsics | `cuda-intrinsics`, `cuda-intrinsics-gen` (upstream has it again; not a workspace member here) |
| FFI and naming | `libnvvm-sys`, `nvjitlink-sys`, `reserved-oxide-symbols` |
| Driver and testing | `cargo-oxide`, `fuzzer` |

## Who uses it

`src/lightnet-cuda-oxide` (kernels) and the `oxide` feature of `src/lightnet-cli`. Build via `./tools/test_oxide.sh`.

## Constraints

- Fresh clone needs `git submodule update --init --recursive`.
- Changing the pin can break `lightnet-cuda-oxide`; re-run its tests with `-j 1` or `-j 2`.
- The root `Cargo.toml` `[workspace.package]` supplies `authors`, `license` and `repository` that the crates inherit.
