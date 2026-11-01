# 05_cli-checkpoints — drive a run, save what a run must save

One job: `hip_darknet train` starts, resumes and stops a run, and the files it writes are the ones the dataset's tooling and our own loader expect.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/04_executor.md`.
- Reference (every run): `_shared/rules.md`, `_shared/darknet-semantics.md` (Training-state files), `crates/hip_darknet/CONTEXT.md`, `README.md` (existing CLI usage that must keep working).

Do NOT load: kernel bodies.

## Process

1. Add the `train` subcommand to `main.rs`, keeping the file thin: parse flags, call the library. Existing `--gpu/--bench/--out/--thresh/--nms` behaviour must not change.
2. Add the `.weights` writer in Darknet's binary format with BN unfolded at save time, and a save→load→per-layer-equality test.
3. Add `metrics.rs`: mAP on the valid list through the existing decode + DIoU-NMS, emitting `map.csv` with the reference column set (plus a 0.5:0.95 sweep row); `loss.csv` with the reference columns plus a per-image loss column.
4. Add resume (own documented format) and prove it by splitting a short run in two and comparing against the same run done in one go.

## Outputs

- `src/main.rs`, weights writer, `src/metrics.rs`, resume support.
- `runs/<name>/05_cli-checkpoints.md`: csv headers written, round-trip result, resume-equality result, `test result` lines.

## Human check

Open the produced `loss.csv` and `map.csv` next to the dataset's files: same columns, same meaning, and a resumed run that lands where the uninterrupted one did.
