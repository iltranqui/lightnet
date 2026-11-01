# Tile model

How `crates/hip-tile` maps tiles onto the GPU, and the nightly-Rust facts it depends on. Read this before writing or changing a kernel or a tile op.

## Contents

- [Blocks and tiles](#blocks-and-tiles)
- [Thread layout](#thread-layout)
- [LDS](#lds)
- [mma](#mma)
- [Barrier](#barrier)
- [Kernels and launchers](#kernels-and-launchers)
- [Nightly Rust facts](#nightly-rust-facts)

## Blocks and tiles

- Every kernel runs 256 threads per block (`config::BLOCK_THREADS`), 8 waves of 32. All threads of a block execute the same tile code with the same tile coordinates.
- `Tile<T, M, N>` is a block-wide value held in registers. `T: Element` (`f32`, `f64`).
- Each thread stores its fragment in a fixed array `[T; FRAG_MAX = 64]`; only `frag_len(M, N) = M·N/256` slots are used. Loops unroll and LLVM removes the unused slots, so there are no spills. A tile above 64 elements per thread (for example 256×128 f32) fails a const assert.
- `Tensor<T>` / `TensorMut<T>` are `#[repr(C)] {ptr, rows: u32, cols: u32, ld: u32}`, row-major with leading dimension `ld`. They are kernel arguments by value. `load_tile`/`store_tile` take element offsets and mask out-of-range elements (zero on load, skipped on store).

## Thread layout

- The threads form a `TR × TC` grid: `TC` is the largest power of two `≤ 256 / min(M, 16)` that divides `N`, and `TR = 256 / TC`; requires `M % TR == 0` (`layout::check_shape`, const-evaluated).
- Thread `(ty, tx)` owns elements `(ty + TR·i, tx + TC·j)`, so neighbouring threads touch neighbouring columns and global accesses coalesce.
- Verified exact (copy and transpose, ragged edges): 64×16, 16×64, 64×64, 64×8, 8×64, 32×32, 1×256, 64×32, 32×64.

## LDS

- Rust cannot declare static LDS. All LDS is the launch-sized region from `core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<T>()`.
- Every LDS op starts at offset 0, begins with `sync_block()` and ends after its last LDS read.
- A kernel declares its need with `#[kernel(lds = ...)]` using `config::{mma_lds_bytes, transpose_lds_bytes, reduce_lds_bytes, lds_tile_bytes, lds_max}`. `smem::require` reads `group_segment_size` from the HSA dispatch packet (offset 28) and traps if the launch gave too little.
- LDS rows are padded by `LDS_PAD = 4` elements against bank conflicts.

## mma

- RDNA2 has no matrix cores (no WMMA/MFMA). `mma(a, b, acc)` stages both operands in padded LDS, then each thread accumulates its fragment of C with `core::intrinsics::fma` (register-blocked outer product, up to 8×8 per thread).

## Barrier

- `arch::sync_block()` is `fence(Release)` + `core::arch::amdgpu::s_barrier()` + `fence(Acquire)`. The fence is system-scope, so LLVM adds `buffer_gl0_inv`/`buffer_gl1_inv`.
- An `asm!("s_waitcnt lgkmcnt(0)", "s_barrier")` barrier was about 5% faster on gemm but violates the no-asm rule ([rules.md](rules.md#code)); it was reverted. Do not reintroduce it.

## Kernels and launchers

- `#[kernel(lds = <const expr>)] fn name(args)` emits, on the device, `extern "gpu-kernel" fn name` and, on the host, `pub mod name { NAME, BLOCK_THREADS, LDS_BYTES, fn launch(module, stream, grid, args...) }`.
- Because the macro emits `pub mod <fn>` inside the file module, launchers live at `kernels::<file>::<fn>::launch`, e.g. `kernels::gemm::gemm::launch`.
- `#[kernel]` cannot be generic: emit one kernel per tile config with `macro_rules!` (see `kernels/src/yolo/conv.rs`). Kernel symbol names must be unique across the crate.
- Argument types must implement `hip_tile::KernelArg` (`unsafe trait`, `Copy + repr(C)`). `#[derive(Clone, Copy)]` on a generic `Tensor<T>` adds a `T: Copy` bound; use manual impls.
- Grid convention: `(blocks along columns, blocks along rows)`, e.g. gemm `(ceil(n/BN), ceil(m/BM))`.
- `hip_tile_build::link_rocm()` in a `build.rs` sets the ROCm rpath so binaries run without `LD_LIBRARY_PATH`.

## Nightly Rust facts

- The amdgcn target needs an explicit `-C target-cpu`, or `core` fails to build. Output is `<crate>.elf`, loadable by `hipModuleLoadData`.
- `generic_const_exprs` is not usable: it ICEs under the next solver (`rustc_infer` `infer/mod.rs:1694`) and overflows under `-Znext-solver=no`. Hence `FRAG_MAX`.
- FMA is `core::intrinsics::fma(a, b, c)` (`fmaf32` no longer exists). Abort is `core::process::abort_immediate()` (feature `abort_immediate`).
- Device features in use: `abi_gpu_kernel`, `stdarch_amdgpu`, `gpu_intrinsics`, `gpu_launch_sized_workgroup_mem`, `core_intrinsics`, `abort_immediate`.
