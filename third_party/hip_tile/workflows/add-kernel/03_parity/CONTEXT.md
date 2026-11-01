# 03_parity — prove it is correct

One job: GPU vs CPU reference tests at every shape in the spec, passing on the GPU.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/01_spec.md`, `runs/<name>/02_kernel.md`.
- Reference (every run): `_shared/rules.md`, `apps/runner/CONTEXT.md`.
- Starter: `_templates/kernel/parity_test.rs`.

Do NOT load: benchmark files, `crates/hip_darknet`.

## Process

1. Copy `_templates/kernel/parity_test.rs` into `apps/runner/tests/` (or extend the matching test file).
2. Implement the CPU reference straightforwardly (f64 accumulation where it matters).
3. Cover every spec shape, ragged sizes, and writes into a view with `ld` if the kernel supports it. Poison memory outside the output region and check it survives.
4. Run: `systemd-run --user --scope -q -p MemoryHigh=3G cargo test -p runner --release -j 4`. All tests, not only the new ones, must pass.

## Outputs

- Tests in `apps/runner/tests/`.
- `runs/<name>/03_parity.md`: test names, shapes, worst error per shape, the `test result` lines.

## Human check

Read the worst-error column: is every value under the spec's tolerance, and do the shapes match the spec list?
