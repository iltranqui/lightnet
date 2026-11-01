# hip_darknet — Darknet YOLO port

One job: load a Darknet model and detect objects, on the CPU (reference) or the GPU, in pure Rust.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `cfg.rs`, `weights.rs` (BN fold), `model.rs` (layers, shapes), `reference.rs` (CPU forward, every layer), `yolo.rs` (decode, DIoU-NMS), `image.rs`, `error.rs`, `lib.rs` (`detect_cpu`, `detect_gpu`).
- Training (CPU reference today; `docs/ROADMAP.md`): `data.rs` (`.data`, image lists, labels), `augment.rs` (jitter/flip/colour, seeded), `metrics.rs` (mAP@0.5 and 0.5:0.95), `train/` (`gpu.rs` `GpuTrainNet`: the GPU trainer, created from a `TrainNet` and copied back at checkpoints; `--gpu` / `-gpu` selects it; `deltas.rs` box math + `[yolo]` deltas + cost, `net.rs` unfolded-BN forward/backward/SGD, `checkpoint.rs` Darknet `.weights` writer + resume state, `driver.rs` the loop and csvs).
- Feature `gpu` (default): `gpu.rs` (`GpuModel`), `draw.rs`, `build.rs` (kernels ELF).
- `main.rs`: CLI (`--gpu`, `--out`, `--bench`, `--thresh`, `--nms`), the `train` subcommand, and the Darknet-shaped `detector train CFG DATA [-map] [-dont_show] [-gpus N]` entry point (paths accepted in either order, checkpoints next to the `.data` file unless `-out`).
- `tests/detect.rs` (ground truth), `tests/gpu.rs` (per-layer and detection parity), `tests/gradients.rs` (numeric gradient checks), `tests/learning.rs` (a toy net must learn).
- `README.md`: usage and API.

## Features (read before building)

- Default features include `gpu`: the HIP executor, the kernels code object, and `--gpu`.
- `--no-default-features` is the CPU-only build. It compiles **no** HIP code, so `--gpu` is absent
  and `train` is CPU-only — a binary built this way can never touch the GPU. Use it only to prove
  the CPU path still builds.
- Consequence for training: today's trainer is CPU-only in *both* builds. The GPU inference path
  measured 4.78 ms/frame against 459.84 ms/frame on the CPU (96x) in the default-features build,
  which is what the missing GPU trainer would buy.

## Dependencies

- Depends on: `image` (png, jpeg), `thiserror`; with `gpu`: `hip-sys`, `hip-rt`, `hip-tile`, `kernels`, build-dep `hip-tile-build`.

## Rules for changes

- Read `_shared/rules.md` first.
- Model work goes through `workflows/port-darknet-model/`; Darknet behaviour comes from `_shared/darknet-semantics.md`.
- Never run the Darknet binary. Keep `main.rs` thin; logic lives in the library.
- Keep `--no-default-features` (CPU only) building.

## Rules for training code

- f32 end to end, SGD with momentum only (`docs/ROADMAP.md` Decisions). No Adam, no f64 master copy.
- Darknet's formulas verbatim, including its quirks (BN backward `n` vs `n-1`, untrained BN shift, the 2× `w`/`h` box slots); deviations are listed in the stage run file and in `_shared/darknet-semantics.md`.
- Gradients are verified numerically (`tests/gradients.rs`), never against the Darknet binary.

## Human check

`cargo test -p hip_darknet --release -- --nocapture` and `cargo test -p hip_darknet --release --no-default-features` pass (49 lib + 1 detect + 11 gradient + 2 learning tests on the CPU-only build).
