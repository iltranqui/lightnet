# 02_kernel — write the kernel

One job: implement the spec as a `#[kernel]` with tile ops, plus a host launch helper.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/01_spec.md`.
- Reference (every run): `_shared/rules.md`, `_shared/tile-model.md`, `kernels/CONTEXT.md`.
- Starter: `_templates/kernel/kernel.rs`.

Do NOT load: `crates/hip_darknet`, benchmark results, other runs.

## Process

1. Copy `_templates/kernel/kernel.rs` to `kernels/src/<name>.rs` (or `kernels/src/<group>/<name>.rs`) and add the `pub mod` line.
2. Write the body with tile ops only; no `asm!`, no per-thread scalar code unless the op is elementwise.
3. If a new tile op is truly needed, add it to `crates/hip-tile` in general (cuTile-style) form, with its own parity test in stage 03.
4. Build: `systemd-run --user --scope -q -p MemoryHigh=3G cargo build -p runner --release -j 4`.
5. Check the ISA: `llvm-objdump -d --mcpu=gfx1031` and `llvm-readelf --notes` on the embedded `kernels.elf` — `private_segment_fixed_size: 0` (no spills).

## Outputs

- Code: `kernels/src/<name>.rs` (+ `crates/hip-tile` changes if any).
- `runs/<name>/02_kernel.md`: files touched, launcher path, LDS bytes, VGPRs, spills.

## Human check

Skim the kernel body: does it read like the spec's math, with no scalar loops hiding in it?
