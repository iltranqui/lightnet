# 06_run — the training run, judged against the reference history

One job: train the model from scratch and show, number by number, how it compares with the dataset's own reference run.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/05_cli-checkpoints.md`, the dataset's `*_loss.csv` and `*_map.csv`.
- Reference (every run): `docs/ROADMAP.md` (acceptance numbers), `_shared/rules.md`, `_shared/benchmarking.md` (for the wall clock statement only).

Do NOT load: kernel sources, the C++ porting source.

## Process

1. Fix the seed and the iteration count with the image-exposure conversion stated in `01_train-semantics.md`; record the cfg overrides used (schedule steps/scales/max_batches) so the run is reproducible from the command line alone.
2. Run to completion, checkpointing at a fixed interval, with `loss.csv`/`map.csv` written as the run goes.
3. Compare against the reference on a same-images-seen axis: learning rate (must match by construction), per-image loss trajectory, mAP@0.5 and mAP@0.5:0.95.
4. Load the final `.weights` back through the inference path and run the ground-truth detection test on the valid images.
5. If the acceptance numbers are missed, state the diagnosis order (gradient check → augmentation parity → loss scaling) and stop rather than tuning hyperparameters.

## Outputs

- Trained `.weights`, `loss.csv`, `map.csv`, command line, seed, wall clock.
- `runs/<name>/06_run.md`: both curves side by side, mAP table, round-trip detection numbers.

## Human check

Do the two loss curves occupy the same band, is the final mAP inside the acceptance bar, and would you bet the saved weights are the ones that produced that last csv line?
