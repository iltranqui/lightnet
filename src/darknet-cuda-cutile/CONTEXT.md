# darknet-cuda-cutile

Opt-in cuTile GPU backend. Not a workspace member.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Concrete `CutileBackend` implementing `darknet_cuda::F32TensorOpsBackend` on top of `cutile-rs` v0.2.0. Setup and requirements: `README.md` in this folder.

## Responsibilities

- Typed `f32` upload and download.
- Device ops: bias add, activation, shortcut add, route concat, upsample, direct convolution.
- Global avgpool via host fallback until the device kernel lands.

## Dependencies

- Depends on: `darknet-cuda` and the git submodule `src/cutile-rs-aot` (NVlabs/cutile-rs, pinned to v0.2.0).
- Depended on by: nothing in the default build.

## Constraints

- Fresh clone needs `git submodule update --init src/cutile-rs-aot` (or `git clone --recurse-submodules`).
- Needs CUDA 13.2+ (13.3 recommended), `CUDA_TOOLKIT_PATH`, libclang.
- Exercises GPU hardware: run only when the task asks for GPU verification.
