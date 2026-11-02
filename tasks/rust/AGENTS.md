# Rust Port — Agent Handbook

Entry point and ordered work list for any agent working in `tasks/rust/` (branch `rust`). Rules, state and plans live on the shelves it links to; this file says what to do next and how to know it is done.

## Table of Contents

- [Goal](#goal)
- [State snapshot](#state-snapshot)
- [Where to go](#where-to-go)
- [Running this file as a goal](#running-this-file-as-a-goal)
  - [Loop](#loop)
  - [Definition of done for one item](#definition-of-done-for-one-item)
  - [Stop conditions](#stop-conditions)
- [Agent Todo](#agent-todo)
  - [Phase 0: Baseline](#phase-0-baseline)
  - [Phase 1: Validate and version the source-wired GPU layers](#phase-1-validate-and-version-the-source-wired-gpu-layers)
  - [Phase 2: Layout refactor leftovers](#phase-2-layout-refactor-leftovers)
  - [Phase 3: Finish the GPU paths](#phase-3-finish-the-gpu-paths)
  - [Phase 4: CPU gaps](#phase-4-cpu-gaps)
  - [Phase 5: Sparse MoE](#phase-5-sparse-moe)
  - [Stop and ask the user](#stop-and-ask-the-user)
- [Keeping this file current](#keeping-this-file-current)

## Goal

Full Rust parity with every C++ layer, in three release lines:

1. `v0.1.NNN`: CPU parity. Every layer ported on the CPU with SGD, Adam and Muon arms, and axis/layout semantics pinned to the C++ source by a named test.
2. `v0.2.NNN`: every layer has a native GPU path with a CPU-parity test, host-fallback layers included, plus GPU Muon.
3. `v0.3.NNN`: opt-in true sparse MoE, inference first, never slower than dense.

A layer is finished only when it meets the [Layer Completion Contract](01_rules/layer-contract.md): host and device code both complete, living together in `src/lightnet-core/src/layers/<layer>/`, with named tests and updated docs.

Current operator override: GPU layers first, edited directly in the main session without subagents, while every open CPU requirement stays open. Native GPU work is authorized before CPU completion; it does not close a CPU gap. No parity claim without a compiled-C++ golden: write "hand-derived from source, no compiled-C++ golden".

## State snapshot

State on 2026-10-27, from a static audit of source against the docs (no cargo run):

- Latest versions: `v0.1.044` (CPU, DSConv multi-template) and `v0.2.204` (GPU, ViT resident Muon and sparse MSA). Next free numbers: `v0.1.045`, `v0.2.205`. Never reuse a number; confirm in `02_state/version-map.tsv`.
- 25 commits sit after `v0.2.202`: layout refactors, `v0.1.043`, `v0.1.044`, the dcnv4 test fix (`a62e123`), the Swin residency WIP (`0362bea`), and `v0.2.203`/`v0.2.204`.
- Wired in the runtime with no recorded pass: native GPU forward, backward and compact f32 for `wmhf`, `tucker_attention`, `mambavision`, `kimi_kda`, `depth_anything`, `dpt_head`. They come from WIP commits with no version. `swin` (v0.2.203) and `vit` (v0.2.204) now have recorded passes.
- No GPU arm: `yolo_3d`, `obb_refine`, GRU training with batch norm. Host-only on GPU: `vc_attention`, `recursive_block` variants other than static `channel_shuffle`, the `clifford` network-wide backward.
- Host suite baseline is not green: four `cfg_corpus` cases are unresolved (two OOM at about 7 GiB, two 300 s timeouts). Do not claim a green host suite.
- Details: [02_state/STATUS.md](02_state/STATUS.md) "Audit 2026-10-27", [02_state/layer-support.md](02_state/layer-support.md).

## Where to go

| Task | Read |
|---|---|
| Any build, edit or commit | [01_rules/rules.md](01_rules/rules.md), [01_rules/commands.md](01_rules/commands.md) |
| Cut a version, commit, tag | [01_rules/versioning.md](01_rules/versioning.md) |
| Port or finish a layer | [01_rules/layer-contract.md](01_rules/layer-contract.md), `src/lightnet-core/src/layers/CONTEXT.md`, [02_state/layer-support.md](02_state/layer-support.md) |
| Resume after a context reset | [02_state/STATUS.md](02_state/STATUS.md) (top), [02_state/open-work.md](02_state/open-work.md) |
| Long backlog by area, with history | [todo.md](todo.md) |
| Check whether X exists | [02_state/implemented.md](02_state/implemented.md), [02_state/layer-map.md](02_state/layer-map.md) |
| GPU layer work | [03_plans/gpu/gpu_layers_plan.md](03_plans/gpu/gpu_layers_plan.md) |
| Low-precision work (fp16, bf16, fp8; own roadmap and todo, versions from `v0.2.224`) | [03_plans/lowprec/ROADMAP_FP8.md](03_plans/lowprec/ROADMAP_FP8.md) |
| LLM work (own roadmap and todo; versions from `v0.1.067` and `v0.2.227`) | [03_plans/llm/ROADMAP_LLM.md](03_plans/llm/ROADMAP_LLM.md) |
| Any other plan | [03_plans/CONTEXT.md](03_plans/CONTEXT.md) |
| Layer Port Contract, TDD recipe, source notes | [04_reference/CONTEXT.md](04_reference/CONTEXT.md) |
| Past corrections | [lessons.md](lessons.md) |

Full folder map: [CONTEXT.md](CONTEXT.md).

## Running this file as a goal

Use it as `/goal tasks/rust/AGENTS.md`.

### Loop

1. Read `lessons.md`, `git status`, `git log -5`, the top of `02_state/STATUS.md` and `02_state/layer-support.md`.
2. Take the first unchecked, unblocked item in [Agent Todo](#agent-todo). One layer or slice at a time. An item marked blocked is skipped, not worked around.
3. Edit in the main session with shell tools, `Edit` and `Write`; no subagents, no Python scripts. Run one command at a time, at most 4 threads: `JOBS=4 MEM=4G cargo xtask check`, then `gpu-check`, the layer's test filter and, for device work, `cargo xtask oxide --test <suite>`. Never run two builds in parallel.
4. On green, commit and tag per [01_rules/versioning.md](01_rules/versioning.md) and update the docs in the same pass. A refactor move is `refactor(rust): ...` with no tag.
5. Tick the item below, then go back to step 2.

### Definition of done for one item

An item is done only when all of these hold:

1. The item's "Done when" line is true, with a named test that ran and passed. A timeout, a filtered-out test or a test that did not run is not a pass; say which ran and which did not.
2. `cargo xtask check` and `gpu-check` pass after the change.
3. The code commit is `v0.M.NNN: <what>` with the tests and results in the body, followed by a local lightweight tag `v0.M.NNN`. Commit only the files the item names, never `git add -A`. Leave the user's uncommitted edits alone.
4. The docs commit `docs(rust): <what>` updates `02_state/implemented.md` (one line with evidence), `02_state/STATUS.md` ("Latest increments"), `02_state/layer-support.md` (the layer's row), `02_state/version-map.tsv` (`git rev-parse v0.M.NNN` as `new_sha`) and `todo.md`.
5. The item is ticked here.
6. Nothing claims C++ parity without a compiled-C++ golden.

Revert `src/lightnet-cuda-oxide/Cargo.lock` before committing: `cargo xtask oxide` rewrites it. Do not run `cargo xtask fmt`; format edited code by hand. After each finished step, delete big files nothing references (`.weights`, `.backup`, `.onnx`, sidecars, logs, A/B build trees).

### Stop conditions

Stop and report, without continuing, when:

- every layer meets the Completion Contract, or the next item needs a decision not written down here;
- a covered test fails, times out or cannot run (for example no GPU), and the cause is not clear after one root-cause pass;
- an action is hard to reverse or outward-facing: push, pull request, history rewrite, deleting data, training run, anything under [Stop and ask the user](#stop-and-ask-the-user).

If something goes sideways, stop and re-plan; do not keep pushing. Never report a layer complete from a primitive, a forward-only path or a passing subset.

## Agent Todo

Ordered checklist. Each item starts with the version it will ship as. `--` means no release (a check or an investigation); `refactor` means a `refactor(rust): ...` commit with no tag. Planned numbers are provisional: at commit time take the next free number in `02_state/version-map.tsv`, never reuse one, and fix the number here if it moved. Take the first unchecked item whose gate is not blocked. Rules: [01_rules/rules.md](01_rules/rules.md); commands: [01_rules/commands.md](01_rules/commands.md).

### Phase 0: Baseline

- [ ] -- Read `lessons.md`, `git status`, `git log -5`, and the top of `02_state/STATUS.md`.
- [ ] -- `JOBS=4 MEM=4G cargo xtask check`, then `cargo xtask gpu-check`. Done when both pass. If either fails, fix the cause before any other item.
- [ ] -- Note which files are the user's uncommitted work (`git status`: `muon_runtime.rs`, `gpu_resident_params.rs`, `yolo_3d*.rs`, `vit/gpu.rs`, `swin/gpu.rs`, oxide tests). Do not stage them unless an item names them.

### Phase 1: Validate and version the source-wired GPU layers

The runtime already dispatches native forward, backward and compact f32 for these layers, but no pass is recorded. For each item: run the Oxide suite and the host filter, fix failures at the root cause, then commit, tag and document. Layer files live in `src/lightnet-core/src/layers/<layer>/`; run a suite with `cargo xtask oxide --test <suite>`.

- [x] v0.2.203 `swin`: version the resident SGD/Adam/AdamW slice (WIP `0362bea`). Run `oxide_swin` and host filter `swin`. Done when both pass and the version row exists.
- [x] v0.2.204 `vit`: native training forward/backward, compact f32, resident parameters. Run `oxide_vit` and host filter `vit`. Done when native training matches the CPU result and the host fallback is not used for supported variants.
- [x] v0.2.205 `tucker_attention`: run `oxide_tucker_attention`, host filter `tucker`.
- [x] v0.2.206 `mambavision`: run `oxide_mambavision`, host filter `mamba`.
- [ ] v0.2.207 `kimi_kda`: run `oxide_kimi_kda`, host filter `kimi`.
- [ ] v0.2.208 Resident parameters for v0.2.205 to v0.2.207: run `oxide_resident_params`.
- [ ] v0.2.209 `depth_anything`: run `oxide_depth_anything`, host filter `depth_anything`. Head dimension above 128 stays on the CPU; say so in the docs.
- [ ] v0.2.210 `dpt_head`: run `oxide_dpt_head`, host filter `dpt`.
- [ ] v0.2.211 `wmhf`: run `oxide_wmhf`, host filter `wmhf`. Done when the `needs_host` cases are listed in `02_state/layer-support.md`.
- [ ] -- Run the host `runtime::` filter once after v0.2.203 to v0.2.211. Done when it passes (332 passed at the last record).

### Phase 2: Layout refactor leftovers

Each move is a `refactor(rust): ...` commit with no version tag. A move changes no behaviour; a behaviour change is a separate versioned commit. Run `cargo xtask check` after each move and the layer's Oxide suite before committing. Status: [02_state/layout-move-queue.md](02_state/layout-move-queue.md).

- [ ] refactor Move the eight `muon_*` methods and `fp4_quantize_1x16` out of `src/lightnet-cuda-oxide/src/lib.rs`. Blocked until the user's Muon residency edits (`muon_runtime.rs`, `gpu_resident_params.rs`) are committed.
- [ ] refactor Move the remaining thin compact arms into `layers/<layer>/gpu.rs` (`runtime/gpu_forward_compact.rs`).
- [ ] refactor Regenerate `02_state/layer-map.md` from the file tree after the moves.

### Phase 3: Finish the GPU paths

Each item ends with the layer meeting the [Layer Completion Contract](01_rules/layer-contract.md). Plan: [03_plans/gpu/gpu_layers_plan.md](03_plans/gpu/gpu_layers_plan.md). Host fallbacks are interim until the layer passes the contract.

- [ ] v0.2.212 `recursive_block`: native forward/backward for variants other than static `channel_shuffle` bodies (BN, controllers, lattice). Suite `oxide_recursive_block`.
- [ ] v0.2.213 `clifford`: remove the network-wide CPU backward fallback. Suite `oxide_clifford`.
- [ ] v0.2.214 GRU with batch norm: native GPU training (CPU reference is v0.1.043). Suite `oxide_gru_training`.
- [ ] v0.2.215 `deform_conv` with batch norm: native path, including the batch-serial normalization Jacobian. Suite `oxide_deform_conv_chain`.
- [ ] v0.2.216 `graph_conv`: unsupported variants. Suite `oxide_graph_conv`.
- [ ] v0.2.217 `hyperbolic_foveation` and `foveation_film`: move the HF source and cache off the host; native SGD/Adam updates. Suites `oxide_hyperbolic_foveation`, `oxide_foveation_film`.
- [ ] v0.2.218 `dcnv4`: unsupported activations and mixed fallback networks. Suites `oxide_dcnv4_*`.
- [ ] v0.2.219 `dsconv`: multi-template GPU path (CPU landed in v0.1.044). Suite `oxide_dsconv`.
- [ ] v0.2.220 Dense-head loss on the device for CenterNet, modern-YOLO and LayaLoss. Suite `oxide_centernet` and the matching dense-loss tests.
- [ ] v0.2.221 GPU Muon for the remaining matrix families, and native update plus residency where parameters are still host-staged (connected, deconvolutional, deform_conv, recurrent, weighted shortcut, HF/FiLM). Suites `oxide_muon`, `oxide_resident_params`.
- [ ] v0.2.222 `vc_attention`: native kernel (host-only today).
- [ ] v0.2.223 Native half-storage arithmetic for the transformer family (today: device f32 islands between half boundaries).

### Phase 4: CPU gaps

Kept open under the GPU-first override. Plans: [03_plans/cpu-layers/](03_plans/cpu-layers/CONTEXT.md). Backward is the exact derivative of the Rust forward; C++ deviations only under `GradMode::Cpp`. Unknown config is an error, never a silent identity.

- [ ] v0.1.045 `yolo_3d` projected math and assignments. Plan: `03_plans/cpu-layers/yolo_3d_plan.md`. Filter `yolo_3d`.
- [ ] v0.1.046 `yolo_3d` loss with exact Rust derivatives and `GradMode::Cpp` deltas.
- [ ] v0.1.047 `yolo_3d` standalone and mixed training APIs consuming the truth helpers.
- [ ] v0.1.048 `yolo_3d` decode and evaluator.
- [ ] v0.1.049 `yolo_3d` A2 output-axis pin (`axis_layout`).
- [ ] v0.1.050 to v0.1.063 (one per open variant) Axis pins: the 14 open C++-backed variants in `02_state/axis-coverage.md`. One named test each; a behaviour fix gets its own version. Never weaken a pin to match the code. Filter `axis_layout`.
- [ ] v0.1.064 `dsconv` multi-template: Muon for the extra offset arrays (explicit error today). Blocked until the user's matrix visitor edit in `muon_runtime.rs` lands.
- [ ] v0.1.065 `[region]` truth loss and decode; `[detection]` completeness.
- [ ] v0.1.066 `obb_refine`/CARPN: CPU training first, then GPU.
- [ ] -- `cfg_corpus`: investigate the four residuals. Run it alone: `cargo test -p lightnet-core --test cfg_corpus -j 2`. A timeout is neither a pass nor a failure.

### Phase 5: Sparse MoE

- [ ] v0.3.001 Opt-in true sparse `[moe_convolutional]` on the CPU, dense-masked result as the oracle. Plan file `MOE.md` is referenced by older docs but does not exist in the repo: write the plan first and get the user to review it. Start only when Phases 1 and 3 are complete.
- [ ] v0.3.002 Sparse `[moe_convolutional]` on the GPU, equal to the dense-masked result within a stated tolerance.

### Stop and ask the user

Stop and report, without continuing, for:

- the act-head training fix (executable C++ updates four parameter blocks, Rust uses only the CLS adjoint): the operator question is unanswered;
- any push, pull request, history rewrite, or deletion of data;
- any training run, convergence run, or the FP8 default decision (`03_plans/lowprec/ROADMAP_FP8.md`);
- a failing test whose cause is unclear after one root-cause pass;
- work outside this list: GUI, CARPN Stage-2 training, OBB GPU training, CUDA graph capture, `dcnv3` (no reference numerics), `detr_decoder` (C++ broken).

## Keeping this file current

- Tick an item only after its definition of done holds. Add a new item at the end of its phase with the next planned version; a ticked item keeps the version it shipped as.
- Update [State snapshot](#state-snapshot) whenever a version lands: latest versions, next free numbers, wired-but-unvalidated layers.
- Finished work goes in `02_state/implemented.md`, not here. Open work stays here and in `todo.md`.
- After a user correction, add the rule to `lessons.md`.
- Each edited `.md` keeps its table of contents in sync with its headings.
