# hip-tile todo (branch: main)

Plan: `~/.claude/plans/crystalline-dreaming-lightning.md`. Run the LegoGears YOLOv4-tiny Darknet model on AMD gfx1031 in pure Rust, using hip-tile (cuTile-style tile kernels).

## Contents

- [Phase A: hip-tile](#phase-a-hip-tile)
- [Phase B: LegoGears YOLO](#phase-b-legogears-yolo)
- [Review](#review)

## Phase A: hip-tile

- [x] Bootstrap: git, toolchain pin, workspace skeleton, task files
- [x] Spike: tile fragment typing, dynamic LDS, ISA check (1347.7 GFLOP/s)
- [x] `hip-sys` + `hip-rt`, `hip-tile-build`, `#[kernel]` macro, runner + saxpy (verified in scratch copy)
- [x] `hip-tile` core ops: load/store, map, mma, transpose, reduce (tile GEMM ~1475 GFLOP/s)
- [x] Step 0: generalized thread layout, `kernels/src/{copy,transpose,gemm}.rs`, real-repo build + tests + clippy, lessons

## Phase B: LegoGears YOLO

- [x] 1. `crates/hip_darknet` (cargo package `hip_darknet`, lib + bin): cfg, weights + BN fold, shapes, CPU reference forward, yolo decode + NMS, image; Rust unit tests + ground-truth label check (no Darknet binary)
- [x] 2. GPU kernels: conv (implicit GEMM + fused bias/leaky), maxpool, upsample, concat; per-op parity
- [x] 3. GPU executor in `hip_darknet` (`--gpu`): per-layer parity, detections equal to the CPU path
- [x] 4. App polish: annotated PNG, `--bench`, README results
- [x] 5. Training, stage 01-02 + the device-independent half of 05 (`ROADMAP.md`): Darknet training
      semantics with citations, cfg training keys, dataset + augmentation, `[yolo]` deltas and cost,
      training-mode forward/backward (f32), SGD + momentum, numeric gradient checks, Darknet-format
      `.weights` writer, mAP, `hip_darknet train` — 63 tests pass, clippy clean
- [ ] 6. GPU training (conv backward verified batched 2026-10-26; remaining: batched forward, `TrainModel`, run): `hip-tile` ops (accumulating store, channel reduction, batched addressing,
      accumulating GEMM), conv/bn/yolo/maxpool/upsample/route backward kernels, batched forward,
      `TrainModel`, 12 000-iteration reference run (`ROADMAP.md` stages 03, 04, 06, 07)
- [ ] 7. Optional: fused route stores, per-layer conv tiles, f16

## Review

2026-10-25. Steps 0-4 done and verified in the real repo:
- `cargo test -p runner --release`: 28 GPU tests pass (gemm vs f64 CPU + rocBLAS, copy, transpose, saxpy, YOLO ops).
- `cargo test -p hip_darknet --release`: 25 unit + ground-truth + GPU per-layer parity (max rel err 1.7e-4) + detection parity (GPU == CPU on 10 images; GT recall 1.000, precision 1.000, 38/38).
- LegoGears 224x160: GPU 4.8 ms/frame forward, 5.8 ms end to end; CPU 443 ms (about 92x).
- Tile GEMM (pure Rust barrier, CPU idle): N=1024 2.05 TFLOP/s (rocBLAS 2.69), N=2048 2.28 (rocBLAS 1.91).
- No inline asm, no C/C++ in the pipeline, no Darknet binary.
Open: step 5 (fused route stores, per-layer conv tiles / split-K for 7x5 maps, f16).
