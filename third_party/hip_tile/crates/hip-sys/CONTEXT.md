# hip-sys — raw HIP FFI

One job: hand-written `extern "C"` bindings to `libamdhip64` (and rocBLAS behind feature `rocblas`).

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `src/lib.rs`: types, enums, functions. Some HIP APIs are versioned symbols (e.g. `hipGetDevicePropertiesR0600`); check with `nm -D /opt/rocm/lib/libamdhip64.so`.
- `build.rs`: link search `$ROCM_PATH/lib` (default `/opt/rocm`), link libs, rpath.

## Dependencies

- Depends on: ROCm libraries.
- Used by: `hip-rt`, `hip_darknet` (feature `gpu`).

## Rules for changes

- Read `_shared/rules.md` first.
- No bindgen: verify every new signature and enum value against `/opt/rocm/include/hip/hip_runtime_api.h`.
- Keep the surface minimal; add only what `hip-rt` needs.

## Human check

New bindings match the header byte for byte (types, argument order, enum values).
