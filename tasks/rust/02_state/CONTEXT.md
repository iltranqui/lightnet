# 02_state

What exists and what is open. Updated in the same pass as every version commit.

## Table of Contents

- [Files](#files)
- [Per-version update](#per-version-update)

## Files

| File | Holds | Open when |
|---|---|---|
| [STATUS.md](STATUS.md) | Resume point, per-version log, evidence gaps | after a context reset |
| [open-work.md](open-work.md) | Current state and open work by release line | choosing next work |
| [layout-move-queue.md](layout-move-queue.md) | Layer-layout refactor status | before a move |
| [implemented.md](implemented.md) | Done ledger, one section per version | checking whether X exists |
| [layer-support.md](layer-support.md) | Per-`LayerKind` CPU/GPU/f16/axis matrix | checking one layer |
| [layer-map.md](layer-map.md) | Per layer: CPU file, GPU glue, kernel, parity test | finding code |
| [axis-coverage.md](axis-coverage.md) | Axis/layout pin ledger | adding an axis pin |
| [version-map.tsv](version-map.tsv) | idx, version, old sha, new sha | version to commit |

The checklist form of open work is [../todo.md](../todo.md).

## Per-version update

Append a `version-map.tsv` row, one line in `implemented.md`, an entry in `STATUS.md` "Latest increments", tick `../todo.md`, and update the layer's row in `layer-support.md`.
