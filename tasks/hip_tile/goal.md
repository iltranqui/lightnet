# hip_tile goal spec

Full spec for the `/goal` of the `hip_tile` branch. The short goal text points here.

## Table of Contents

- [Goal](#goal)
- [Context](#context)
- [Scope](#scope)
- [Rules](#rules)
- [Done when](#done-when)
- [Open checks](#open-checks)

## Goal

Run LegoGears inference on AMD ROCm (Radeon 680M, gfx1031) through hip_tile, with the same or very close results as the existing lightnet paths, for three cases: detector, segmentation, and ONNX export. The kernels are one Rust source compiled differently per backend: cuda-oxide (PTX) for NVIDIA, rustc `amdgcn` (hip_tile runtime) for AMD.

## Context

- Repo: `/home/kerrigan/src/DarkSuite/lightnet`, branch `hip_tile`.
- hip_tile is a plain copy at `third_party/hip_tile` (from c2c0f7e, no .git), a standalone workspace excluded in the root `Cargo.toml`. It already has a LegoGears YOLOv4-tiny detector on the 680M (`crates/hip_darknet`, tile GEMM, conv kernels in `kernels/src/yolo`). Read first: its `CLAUDE.md`, `CONTEXT.md`, `_shared/tile-model.md`, `_shared/rules.md`, `_shared/darknet-semantics.md`.
- cuda-oxide kernels: `src/lightnet-cuda-oxide/src/kernels/*.rs` (per-thread SIMT). cuda-oxide is a pinned submodule, not editable.
- Reference paths in lightnet: CPU reference layers, the oxide GPU path, the segmentation head (`[segmentation]`, or legacy `[logistic]` / `[softmax] spatial=1`; `src/lightnet-cli/src/segmentation_train.rs` for training), and ONNX export (`tools/export_onnx.sh`, `lightnet-rs export-onnx`).
- Existing parity tooling to reuse: `cargo xtask export_test_onnx_match` (detector heads vs ONNX Runtime), `export_test_onnx_match_segmentation_bin|multi|all` (segmentation vs ONNX Runtime), `tools/onnx_check.py`, `tools/onnx_match_seg.sh`.
- Also read `tasks/hip_tile/todo.md`, `src/lightnet-cuda-oxide/CONTEXT.md`, `tasks/rust/AGENTS.md`, `tasks/rust/02_state/layer-support.md`.

## Scope

In order. Stop at the first failing gate.

1. Design. Write into `tasks/hip_tile/todo.md`: the adapter surface (thread/block index, input slice, disjoint output slice, barrier), which ops are shared SIMT source, and which need a per-backend kernel (tile GEMM and conv). Prove both toolchains compile a trivial shared kernel before anything else.
2. Shared kernel crate. New `no_std` crate (not a workspace member) with two thin adapters: `cuda_device` and hip_tile / `core::arch::amdgpu`. Port bias add first, run it on both GPUs, compare with the CPU reference.
3. Detector. Port the layers the LegoGears YOLO cfg needs: elementwise (bias, activation, shortcut, route, upsample), maxpool, conv and GEMM. Elementwise ops are one shared source. For conv and GEMM, reuse hip_tile's tile kernels on AMD and keep the tuned oxide ones on NVIDIA; justify every exception in the todo. Gate: detections on the LegoGears test images match the CPU reference (same boxes and classes, score difference <= 1e-3, report the max).
4. Segmentation. Add the segmentation head on the AMD path for the binary and multi-class datasets (`semantic_binary`, `semantic_multi`). Gate: class agreement and max error against the native forward, using the existing `export_test_onnx_match_segmentation_*` method.
5. ONNX export. Export is CPU-only today, so the gate is that the exported ONNX (detector, segmentation, FP32 and FP16) run in ONNX Runtime matches the AMD hip_tile forward as well as it matches the existing paths. Reuse `onnx_match.sh` and `onnx_match_seg.sh`; add a flag or mode that takes the hip_tile forward as the reference side. Do not rewrite the exporter.
6. No training on AMD in this goal. No tuning for speed beyond a single latency number per model.

## Rules

- Use `cargo xtask` subcommands (`check`, `clippy`, `oxide`, `fmt`, and the export/match commands above) for lightnet. For `third_party/hip_tile` use its own `cargo xtask`. Max 4 build threads and a soft RAM cap: `JOBS=4 MEM=4G cargo xtask ...`; raw cargo only as `systemd-run --user --scope -p MemoryHigh=3G cargo ... -j 4`.
- The main session does the design, the adapter surface, review, builds, tests and commits. Codex does bounded subtasks: port one op or layer per task, write its parity test, run read-only lookups.
- Codex runs with model `gpt-6-luna`, at most 1 background task and 2 threads, through the `codex` skill. Give each task one op, the exact files it may touch and the adapter API it must use. Codex cannot cap memory and cannot write `.git`, so it never builds or commits. Flags go before `resume`. Use plain `mv`.
- After each Codex task, the main session reads the diff, runs `cargo xtask check` (4 threads) and the parity test, and fixes or rejects the result. Do not start the next op while it fails.
- No Python scripts for edits. Main-session edits use shell tools, `Edit` and `Write`.
- Production crate graph stays acyclic (`tools/check_module_layers.sh`).
- Commit only files the task names, never `git add -A`. One commit per working op, with the feature tag required by `tasks/rust/AGENTS.md`. Delete weights and other big files when no longer needed.
- Update `third_party/CONTEXT.md` (it still says everything there is a pinned submodule), `CLAUDE.md` routing and `tasks/rust/02_state/layer-support.md` once the layout settles.
- Do not start training runs.
- Record every correction from the operator in `tasks/hip_tile/lessons.md`.

## Done when

- LegoGears detector, segmentation (binary and multi) and the ONNX exports each have a passing gate above, with the numbers reported (max error, class agreement, box/score differences, one latency per model).
- Elementwise ops exist once in the shared crate, with no duplicated kernel body per backend. Every other exception is listed in the todo with the reason.
- `cargo xtask check` and `clippy` pass for the workspace; `tools/check_module_layers.sh` passes.
- `tasks/hip_tile/todo.md` has a review section: what ran, what did not, numbers.
- If the AMD GPU is not usable (ROCm missing, device not found), stop and say so. Do not report the AMD side as verified.

## Open checks

- Segmentation: not yet checked that hip_tile has a segmentation head. Assumed to need adding. Step 4 only works if the AMD path can run the `[segmentation]` layer.
- ONNX export: the exporter runs on CPU, so "ONNX on AMD" means the exported model matches the AMD forward in ONNX Runtime. It does not mean exporting from the GPU.
