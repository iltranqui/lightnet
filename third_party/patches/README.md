# third_party/patches

## Table of Contents

- [Patches](#patches)
- [Applying](#applying)

## Patches

| File | Target | Why |
|---|---|---|
| `0001-rustc-codegen-cuda-test-unit-output-name.patch` | `third_party/cuda-oxide` pinned at `7d9c19405` | Names the `lib --test` unit `<crate>_test` so parallel codegen (`-j > 1`) does not race on the shared `.ll`/`.ptx`/`.o` files. Not yet upstream. |

## Applying

Submodules cannot carry local edits, so apply on demand from the submodule root:

```
cd third_party/cuda-oxide && git apply -p1 ../patches/0001-rustc-codegen-cuda-test-unit-output-name.patch
```

Use `-j 1` for oxide builds until it is applied (this is already the repo guidance).
