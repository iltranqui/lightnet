# 07_bench-docs — measure, then make the repo say the truth

One job: honest speed and memory numbers for training, and every folder's CONTEXT.md plus the README updated so the repo describes what now exists.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): all earlier stage files.
- Reference (every run): `_shared/benchmarking.md`, `_shared/rules.md`, `README.md`, `crates/hip_darknet/CONTEXT.md`, `kernels/CONTEXT.md`, `crates/hip-tile/CONTEXT.md`, `CONTEXT.md`, `docs/ROADMAP.md`.

Do NOT load: other runs.

## Process

1. Measure per `_shared/benchmarking.md`: CPU idle, warm-up, best of several repetitions, HIP events, twice. Per-iteration time, images/s, effective TFLOP/s for forward and backward separately, peak device memory, and the CPU reference trainer for scale.
2. Update the goal table's numbers in `README.md` and, if a number becomes a new reference, `_shared/benchmarking.md`.
3. Update the CONTEXT.md files of every folder whose file list changed, the root `CONTEXT.md` workflow table, and the status column of `docs/ROADMAP.md`.

## Outputs

- `runs/<name>/07_bench-docs.md`: both measurement rounds, the memory figure, what changed in the docs.
- Updated README and CONTEXT files.

## Human check

Do the two measurement rounds agree within the method's tolerance, and does every CONTEXT.md file now list a file that actually exists?
