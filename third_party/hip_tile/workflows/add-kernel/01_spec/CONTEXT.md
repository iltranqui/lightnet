# 01_spec — define what the kernel computes

One job: write a spec precise enough that the kernel and its test can be written from it alone.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): the request, and `runs/<name>/01_spec.md` (copied from the template).
- Reference (every run): `_shared/rules.md`, `_shared/tile-model.md`.
- Reference (similar kernels): `kernels/src/gemm.rs`, `kernels/src/yolo/conv.rs`.

Do NOT load: other runs, `crates/hip_darknet`, the whole `crates/hip-tile` source.

## Process

1. State the math with index notation, the data layout (row-major `Tensor` views, `ld`), and dtype.
2. List the real shapes it must handle, including ragged ones and edge cases.
3. Pick tile shapes that satisfy the layout rule (`tile-model.md#thread-layout`) and the 64-elements-per-thread limit; compute the LDS bytes with the `config` helpers.
4. Name the kernel symbol(s) (unique), the launcher path, the grid, and any `#[repr(C)]` params struct.
5. Define the CPU reference and the tolerance (exact for data movement; ~1e-4 relative for reductions/GEMM-like).

## Outputs

- `runs/<name>/01_spec.md`

## Human check

Read the shape list and the tolerance. Are the real shapes all there, and is the tolerance strict enough to catch an indexing bug?
