# hip-tile lessons (branch: main)

Corrections and the rules they produced. Settled technical facts moved to `_shared/` (one home per fact); this file keeps the history of how they were learned.

## Contents

- [Corrections](#corrections)
- [Where the facts live](#where-the-facts-live)
- [History](#history)

## Corrections

- **No Darknet binary.** The YOLO work is a direct Rust port: never invoke `/usr/bin/darknet` or `darknet_*` binaries for references or tests. Rule: `_shared/rules.md#code`.
- **No inline asm.** An agent added an `asm!` LDS barrier (+5% gemm); the repo `CLAUDE.md` forbids hand-written AMDGCN. Reverted; rule: `_shared/rules.md#code`, details `_shared/tile-model.md#barrier`.
- **Don't trust one benchmark run on the iGPU.** "Tile gemm beats rocBLAS at N=1024" was a clock artifact. Method: `_shared/benchmarking.md`.
- **Check claims from subagents.** Re-run their tests and benches in the real repo before marking a step done (the bench artifact above was caught this way).
- **Names the user gives are exact.** The model-port package is `hip_darknet` (underscore), one package with lib + bin.
- **Device-equals-CPU parity cannot catch a bug both share.** `[yolo]` positive cells only got a delta for the truth class, so the other classes never learned to go to 0: every box scored all 5 classes high, about 4 FP per object at conf 0.25. The reference `delta_yolo_class` sets every class to `(n == class) - output`. The CPU port and the kernel were written from the same misreading, so every parity test passed. Rule: for each loss term, read the reference C++ loop (not a summary of it) and add a hand-computed test of the negatives; look at raw detections (`GT` next to `DET`) before blaming NMS or the metric.

## Where the facts live

| Topic | Home |
|---|---|
| Hard rules, environment, build caps | `_shared/rules.md` |
| Tile layout, LDS, mma, barrier, launchers, nightly-Rust facts | `_shared/tile-model.md` |
| Darknet inference semantics (verified in C++ source) | `_shared/darknet-semantics.md` |
| Benchmark method and reference numbers | `_shared/benchmarking.md` |
| Per-kernel / per-model results | `workflows/*/runs/<name>/` |

## History

2026-10-25, in order:
1. Spike: hand-written Rust GEMM on gfx1031 at 1347.7 GFLOP/s proved rustc → AMDGCN works.
2. `generic_const_exprs` tile fragments ICE'd the compiler; replaced by `FRAG_MAX` arrays.
3. Tile-API gemm tuned to 128×128×16 (run: `workflows/add-kernel/runs/gemm/`).
4. LegoGears YOLOv4-tiny ported, CPU then GPU (run: `workflows/port-darknet-model/runs/legogears_v2/`).
5. Repo restructured to ICM (this file split into `_shared/`).
