# 01_train-semantics — the facts the whole run rests on

One job: every training behaviour this run depends on is read out of Darknet's C++ source, written down with a citation, and the loss scaling is pinned so our numbers are comparable to the dataset's `loss.csv`.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): the model's `.cfg`, `.data`, `.names`, train/valid lists, label files, and the dataset's `*_loss.csv` / `*_map.csv` reference history (paths in `runs/<name>/01_train-semantics.md`).
- Reference (every run): `_shared/rules.md`, `_shared/darknet-semantics.md`, `docs/ROADMAP.md`, `crates/hip_darknet/CONTEXT.md`.
- Porting source (read only): `~/src/DarkSuite/darknet/src-lib/` — `yolo_layer.cpp`, `batchnorm_layer.cpp`, `convolutional_layer.cpp`, `blas.cpp`, `darknet_network.cpp`, `darknet_cfg.cpp`, `data.cpp`, `image_opencv.cpp`, `detector.cpp`.

Do NOT load: `kernels/`, anything GPU, `/usr/bin/darknet` (never run it).

## Process

1. List every training-only key in the cfg (net section and `[yolo]` sections) and say for each whether this run implements it or rejects it.
2. For each behaviour the run depends on — loss/cost accumulation, delta assignment order, BN train path, SGD step, LR schedule, truth packing, augmentation geometry and colour — read the C++ and add the fact to the **Training** sections of `_shared/darknet-semantics.md`, one line per fact with `file:line`. Never restate a fact that already lives there.
3. Pin the loss scaling: what exactly the `loss`/`avg_loss` columns of the reference `loss.csv` mean, how many images per update, and how our batch size converts to theirs.
4. Write down the deviations this run accepts (for example 8-bit HSV rounding) with the tolerance they imply.

## Outputs

- `_shared/darknet-semantics.md` Training sections (the facts).
- `runs/<name>/01_train-semantics.md`: key inventory (implement / reject), loss-scaling definition, accepted deviations, reference-history summary.

## Human check

Does every implemented key point at a cited fact, and does the loss-scaling section let you predict roughly where the first iteration's loss should land for this batch size?
