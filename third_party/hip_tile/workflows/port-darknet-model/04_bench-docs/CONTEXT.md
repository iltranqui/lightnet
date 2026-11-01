# 04_bench-docs — measure and publish

One job: honest speed numbers, an annotated sample, and the README updated.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<model>/03_executor.md`.
- Reference (every run): `_shared/benchmarking.md`, `_shared/rules.md`, `README.md` (Results section).

Do NOT load: kernel sources, other runs.

## Process

1. `cargo run -p hip_darknet --release -- --gpu --bench 200 ...` twice with an idle CPU (method: `_shared/benchmarking.md`).
2. `--out docs/<model>_gpu.png` on one validation image; look at it.
3. Update the README Results table and, if the numbers are a new reference, `_shared/benchmarking.md`.

## Outputs

- `runs/<model>/04_bench-docs.md`: GPU/CPU ms/frame (forward and end to end), both runs, image path.
- `docs/<model>_gpu.png`, README update.

## Human check

Open the annotated image: are the boxes on the objects? Do the two bench runs agree?
