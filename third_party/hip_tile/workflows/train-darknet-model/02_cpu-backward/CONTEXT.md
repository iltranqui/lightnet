# 02_cpu-backward — a gradient-checked CPU trainer

One job: a training-mode CPU forward, the loss and delta tensor, and every backward pass, all f32, all verified against numerical differentiation.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/01_train-semantics.md`.
- Reference (every run): `_shared/rules.md`, `_shared/darknet-semantics.md` (Training sections), `crates/hip_darknet/CONTEXT.md`, `crates/hip_darknet/src/reference.rs` (the inference forward this one must stay consistent with).

Do NOT load: `kernels/`, `gpu.rs`.

## Process

1. Add `src/train/reference.rs`: training-mode forward that keeps BN unfolded, saves pre-activations and per-channel batch stats, then the loss + deltas exactly as `01_train-semantics.md` specifies.
2. Add `src/train/optim.rs`: SGD with momentum only — `updates += −decay·batch·w`, `w += (lr/batch)·updates`, `updates *= momentum`; bias and BN params without decay; `steps` policy with `burn_in` and `power`; BN rolling-stat update.
3. Check gradients three ways, all f32: two independently written CIoU derivative formulations agreeing; central differences per layer type and for a tiny net at a magnitude-scaled step; and analytic identities (BN 3-element hand case, maxpool/upsample/route gradient sums equal the forward sums).
4. Record the tolerance each check passes at — a number, not "close enough".

## Outputs

- `crates/hip_darknet/src/train/{reference,optim}.rs` (+ tests).
- `runs/<name>/02_cpu-backward.md`: per-check table (what, tolerance, measured error), `test result` lines.

## Human check

Look at the tolerance column: is every check at or below the f32 figure stated in the file, and does the CIoU pair agree without having been written by copy-paste from one another?
