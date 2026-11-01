# HANDOFF — GPU training of hip_darknet

State of the GPU trainer at the end of the session of 2026-10-26, for whoever picks it up next. Read `CLAUDE.md` and `docs/ROADMAP.md` first; this file only records what changed, what is proven, and what is open.

## Contents

- [Where things stand](#where-things-stand)
- [How to run it](#how-to-run-it)
- [What was built](#what-was-built)
- [What is verified](#what-is-verified)
- [What is not done](#what-is-not-done)
- [Things that will surprise you](#things-that-will-surprise-you)
- [Next steps](#next-steps)

## Where things stand

`GpuTrainNet` (`crates/hip_darknet/src/train/gpu.rs`) runs a complete training step on the AMD GPU: batched forward, `[yolo]` loss and deltas, backward, and the SGD update. Every stage matches the CPU reference trainer (`TrainNet`) on the real `LegoGears_v2.cfg`. The training loop (`train/driver.rs`) can use it with `--gpu`, and `cargo xtask train_hip_legogears` wraps the whole thing.

What has not happened: no long run. The longest GPU run so far is 20 iterations, so there is no evidence yet that the loss falls over a real schedule or that mAP reaches the acceptance bar. Stage 06 of `ROADMAP.md` is still open.

Commits from this session, oldest first:

| Commit | What |
|---|---|
| `efd9c47` | `conv_bwd_dx` writes each image to its own `dx` slice (the `batch > 1` bug); batched parity test un-ignored |
| `cdf0348` | batched forward launches for conv, maxpool, upsample, concat (`kernels/src/yolo/batched.rs`) |
| `7499738` | `bn_fwd` kernel (normalise, scale/shift, fused leaky) |
| `f2b9cc6` | CPU maxpool forward/backward handle the whole batch (a bug in the stage-02 reference) |
| `da3c644` | `GpuTrainNet` forward and the `yolo_act` kernel |
| `580c7fd` | `GpuTrainNet::loss_and_deltas`; shared `head_cost_from_sums` |
| `8f40890` | `GpuTrainNet::backward` and `update`; `leaky_bwd`, `bias_bwd`, batched backward launches |

Uncommitted at the time of writing (see `git status`): the `xtask` crate, the `.cargo/config.toml` alias, `--gpu` / `-gpu` in `main.rs`, the GPU path in `driver.rs`, the doc edits in `CONTEXT.md` and `crates/hip_darknet/CONTEXT.md`, and this file. A final clippy and full-test run was in flight when the session ended; rerun it before committing.

## How to run it

```
cargo xtask train_hip_legogears                 # cfg defaults: batch 16, 3000 iterations
cargo xtask train_hip_legogears --iters 20      # quick check
cargo xtask train_hip_legogears --map           # periodic mAP on the validation split
cargo xtask --help                              # all options
```

The task builds `hip_darknet` in release mode and runs `hip_darknet detector train CFG DATA -gpu`. It keeps everything the `.cfg` says (batch, iterations, learning-rate schedule, augmentation) unless an option overrides it. Output goes to `target/legogears_v2_gpu` (`loss.csv`, `map.csv`, `iter_NNNNNN.weights`, `last.state`). Both steps run under `systemd-run --user --scope -p MemoryHigh=3G` when it exists.

Direct form: `hip_darknet train --data DATA --gpu ...` or `hip_darknet detector train CFG DATA -gpu ...`. `--no-default-features` builds are CPU-only and refuse `--gpu`.

Measured speed (3 and 20 iteration runs, including data loading and augmentation): about 100 ms per iteration at batch 4 and about 300 ms at batch 16, against roughly 22 s per iteration on the CPU at batch 4. A 3000-iteration run at batch 16 should take around 15 minutes, but that is an extrapolation, not a measurement.

## What was built

- **Kernels** (`kernels/src/`): `yolo/batched.rs` and `backward/batched.rs` (per-image launches over offset views; the batch-1 inference kernels are untouched), `backward/bn_fwd.rs`, `backward/yolo_act.rs`, `backward/act_bwd.rs` (`leaky_bwd`), `backward/bias_bwd.rs`, and the `conv_bwd_dx` fix.
- **`GpuTrainNet`**: built from a CPU `TrainNet` so both start from identical weights. Device-resident weights, momentum buffers (the `*_updates` buffers), BN state, activations and gradients. `download_into(&mut TrainNet)` copies everything back; the driver calls it at checkpoints and mAP evaluations so the `.weights` writer and resume format are unchanged. `load_activations(&TrainNet)` loads the CPU's forward values and exists only for parity tests.
- **Driver**: `TrainOptions::gpu`, a `GpuHandle` that is `GpuTrainNet` with the `gpu` feature and `()` without it.
- **`xtask`**: see `xtask/CONTEXT.md`.
- **Shared cost formula**: `head_cost_from_sums` in `train/deltas.rs`, used by both trainers.

## What is verified

All on `LegoGears_v2.cfg`, batch 2, against the CPU reference (`crates/hip_darknet/tests/train_gpu.rs`). Errors are the largest absolute difference as a fraction of the tensor's largest magnitude.

| Check | Result | Assert |
|---|---|---|
| Forward, all 38 layers, plus BN batch and rolling statistics | worst 2.2e-5 | 1e-4 |
| Network loss, fixed batch with ground truth | 138.590454 (CPU) vs 138.590439 (GPU) | 1e-3 relative |
| Head gradients at the two `[yolo]` heads | 7.5e-6 and 8.8e-6 | 1e-3 |
| Per-layer output gradients | worst 3.8e-6 | 1e-4 |
| Parameter gradients (weights, BN scales, plain-conv bias) | worst 7.3e-6 | 1e-4 |
| Weights, bias, scales, momentum after one SGD step | worst 7.3e-6 | 1e-4 |

Kernel-level tests: `apps/runner/tests/backward_ops.rs` (9 tests, including batched conv backward and `bn_fwd`) and `yolo_batched.rs` (5 tests, bitwise equal to single-image launches). The CPU regression for the maxpool batch bug is `maxpool_batched_equals_per_image` in `tests/gradients.rs`.

## What is not done

- **No long GPU run.** Stage 06 acceptance (mAP@0.5 at least 0.99, loss band, stability, reproducibility, round trip, performance) is untested.
- **No `04_executor.md` run file.** `workflows/train-darknet-model/runs/legogears_v2/04_executor.md` should hold the parity table above, the timing probe, and the `test result` lines. `ROADMAP.md`'s status table and `tasks/main/todo.md` still describe stage 04 as not started.
- **Stage 01 and stage 03 run files are still `status: review`.** The project rule says a person reads each stage's output before the next stage starts. Stage 03's file was updated this session (the `cin > 1` / `batch > 1` diagnoses it carried were both wrong; see below) and should be reread.
- **Device `iou_loss=mse` mismatch** (4 of 126 box entries) is still open. `GpuTrainNet` refuses `mse` heads. This cfg uses `ciou`.
- **Limits in `GpuTrainNet`**: at most 200 ground-truth boxes per image, only `linear` and `leaky` conv activations, no input-gradient computation for layer 0, `-gpus N` is accepted and ignored.
- **`ROADMAP.md` decisions conflict with the cfg.** The roadmap fixes batch 4 and 12000 iterations; the cfg says batch 16 and 3000 iterations with `steps=2400,2700`. The user asked the task to keep the cfg's values, so `cargo xtask train_hip_legogears` follows the cfg. Reconcile the roadmap or the cfg.
- **Dataset caveats from stage 05 stand**: the train and valid lists are the same 80 images, and the only shipped weights file produces no detections.

## Things that will surprise you

- **GPU and CPU forwards disagree at leaky-ReLU kinks.** Two f32 forwards differ by rounding. A value within that rounding of zero lands on opposite sides of the kink, which flips its gradient factor between 1 and 0.1. One such element in layer 10 once showed up as a 1.7e-2 gradient mismatch spread over the earlier layers. It is not a bug. The backward parity test therefore calls `load_activations(&net)` so both sides use the CPU's forward values. Expect GPU and CPU loss curves to drift slightly apart in real runs for the same reason.
- **Earlier notes blamed the wrong kernel.** `ROADMAP.md` said `conv_bwd_dw` was wrong for `cin > 1`; the stage 03 file then said it was wrong for `batch > 1`. Neither was true. `conv_bwd_dw` was correct; `conv_bwd_dx` folded all images into image 0 and the test's dX buffer was sized for one image.
- **The CPU trainer had a batch bug.** Its maxpool ignored images after the first, so any CPU training run at `batch > 1` through a maxpool net before commit `f2b9cc6` used wrong gradients. Treat older CPU run outputs with suspicion.
- **`bn_stats` computes variance in one pass** (`(sum of squares - sum squared / n) / (n - 1)`) while the CPU uses two passes. It passes at the current scale; with large conv sums relative to their spread it can lose f32 precision. Look here first if layer parity drifts on a different net.
- **The loss is formed on the host.** Each iteration the per-head delta and per-truth statistics come back to the host to compute the reported cost. The gradient itself stays on the device. The cost of this has not been measured.
- **Batch handling is per-image launches**, so a layer costs `batch` launches. Measured throughput includes this, but a fused batched kernel may be faster.
- **The learning rate prints as `0.00000000` for the first iterations.** That is `burn_in=1000` with a power-4 ramp, not a bug: at iteration 20 the rate is about 4e-10.
- **Memory**: this machine has about 7.4 GB RAM shared with the iGPU. Batch 16 ran fine for 3 iterations; a long run's footprint has not been watched.

## Next steps

1. Rerun `cargo clippy --workspace --all-targets` and `cargo test --workspace --release` (and `-p hip_darknet --no-default-features`), then commit the uncommitted work.
2. Run `cargo xtask train_hip_legogears --map` for the full cfg schedule. Watch `loss.csv` for finite, falling loss and compare it with `LegoGears_v2_loss.csv` in the dataset folder.
3. Write `04_executor.md` and update the `ROADMAP.md` status table and `tasks/main/todo.md` (stage 04 done, stage 06 in progress).
4. Decide the batch/iterations question (roadmap versus cfg) and record it in `ROADMAP.md` Decisions.
5. Have a person read the stage 01, 03 and 04 run files before stage 06 counts as started.
6. Optional: fix the device MSE mismatch, move the loss reduction onto the device, fuse the per-image launches.
