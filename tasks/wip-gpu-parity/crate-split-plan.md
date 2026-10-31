# Splitting lightnet-core into crates

Plan only. Nothing here is implemented or measured. Numbers marked "measured" come from `grep`/`wc` on the tree at `bb98f0f` (2026-10-31); everything else is a proposal. It answers the question asked on 2026-10-31: can `lightnet-core` become `lightnet-core-conv` and `lightnet-core-train`, mainly to cut rebuild time?

## Table of Contents

- [Verdict](#verdict)
- [Current structure](#current-structure)
  - [Size](#size)
  - [Dependency edges that block a split](#dependency-edges-that-block-a-split)
- [Target crate tree](#target-crate-tree)
- [Component breakdown](#component-breakdown)
  - [lightnet-model](#lightnet-model)
  - [lightnet-layers](#lightnet-layers)
  - [lightnet-runtime](#lightnet-runtime)
  - [lightnet-train](#lightnet-train)
  - [lightnet-core facade](#lightnet-core-facade)
- [Implementation plan](#implementation-plan)
  - [Phase 0 measure and decide](#phase-0-measure-and-decide)
  - [Phase 1 cheap wins](#phase-1-cheap-wins)
  - [Phase 2 layer the modules inside one crate](#phase-2-layer-the-modules-inside-one-crate)
  - [Phase 3 extract lightnet-model](#phase-3-extract-lightnet-model)
  - [Phase 4 extract lightnet-layers](#phase-4-extract-lightnet-layers)
  - [Phase 5 extract lightnet-train](#phase-5-extract-lightnet-train)
- [Verification](#verification)
- [Risks](#risks)
- [Open questions](#open-questions)
- [Execution decisions and Phase 0 status (2026-10-31)](#execution-decisions-and-phase-0-status-2026-10-31)
- [Phase 2 LayerParams inventory](#phase-2-layerparams-inventory)

## Verdict

Possible, but only in the order below, and only if Phase 0 shows the gain is worth it. The split cannot follow today's `crate::` references, because they form cycles. The cycles must be removed inside the single crate first (Phase 2); each of those steps is an ordinary refactor that `cargo xtask check` verifies, and the crate moves after that are mostly file moves.

The names `lightnet-core-conv` and `lightnet-core-train` are kept as the intent (layers below, training above), but the crates are named by role: `lightnet-layers` and `lightnet-train`. Rename freely; the graph matters, not the names.

## Current structure

### Size

Measured:

| Part | Lines |
|---|---|
| whole `lightnet-core` (222 files) | 153,701 |
| `layers/` (59 directories) | 54,478 |
| `runtime/` | 54,592 |
| `model.rs` | 12,243 |
| `obb_training.rs` | 6,985 |
| `train.rs` | 5,101 |
| `detect.rs` | 3,015 |
| `onnx/` | 1,584 |

There are 45 `gpu.rs` files and 37 `cpu.rs` files under `layers/`, and 1,059 `#[test]` functions. Downstream crates that depend on `lightnet-core`: `lightnet-cli`, `lightnet-cuda-oxide` (dev-dependency, in its own cargo workspace) and `lightnet-examples`.

### Dependency edges that block a split

`crate::X` references, counted by `grep`:

| From | To | Count | What it is |
|---|---|---|---|
| `layers/` | `model` | 135 | specs and shapes |
| `layers/` | `runtime` | 77 | `Tensor`, `LayerParams`, `GpuExecutableNetwork`, `GpuForwardState`, `tensor_backend_shape`, `backend_error` |
| `runtime/` | `model` | 106 | specs |
| `runtime/` | `layers` | 62 | CPU reference math and GPU glue |
| `model.rs` | `layers` | about 25 | weight-count and geometry helpers: `depth_anything`, `dpt_head`, `tucker_attention`, `mambavision`, `kimi_kda`, `vit`, `laya_act_head` (the `obb_training` uses are in tests) |
| `layers/` | `train`, `detect`, `dense_loss` | 1 each | `hyperbolic_foveation/gpu.rs` (`YoloLabel`), `carpn/cpu.rs` (`detect` helpers), `segmentation/cpu.rs` (doc comment only) |
| `runtime/` | `train` | 12 | label and loss types |
| `train.rs` | `runtime` | 7 | executable networks |

Three cycles result:

1. `model` and `layers`: `model.rs` calls layer geometry helpers; layers read `model` specs.
2. `layers` and `runtime`: the 45 `gpu.rs` files take `&GpuExecutableNetwork<B>`/`GpuForwardState`, while the runtime dispatches into them. Only 11 of the 37 `cpu.rs` files import `runtime`, so the CPU half is nearly independent.
3. `runtime` and `train`: shared label and loss types.

`LayerParams` (in `runtime/types.rs`) has a variant per layer holding layer-defined weight structs. It is the knot that ties `runtime` to every layer, so it decides where the base crate boundary falls.

## Target crate tree

```text
lightnet-model        error, cfg, names, TensorShape, Tensor, model.rs, activation_math,
   │                  per-layer geometry/spec/param types (data only), label types
   ▼
lightnet-layers       layers/*/cpu.rs  (CPU reference math, no backend)
   ▼
lightnet-runtime      runtime/, layers/*/gpu.rs glue, weights, initialize, muon_runtime, onnx
   ▼
lightnet-train        train, dense_loss, augmentation, loader_profile, evaluate, detect,
   │                  obb_*, chart, image, cuboid_labels, yolo_3d, segmentation_data
   ▼
lightnet-core         facade: pub use of everything above, unchanged public paths
   ▲
lightnet-cli, lightnet-cuda-oxide (dev), lightnet-examples   (no source changes)
```

`lightnet-cuda` (the backend trait crate) stays below `lightnet-runtime` as it is today.

## Component breakdown

### lightnet-model

- What it is: the shared vocabulary of the project.
- Must do: hold every type that both `layers` and `runtime` name, with no math and no backend. That includes `TensorShape`, `Tensor` (moved from `runtime/types.rs`), `error`, `cfg`, `names`, `model.rs`, `YoloLabel` and the other label types, `activation_math`, and the geometry, validation and weight-count helpers that `model.rs` calls today.
- Depends on: `lightnet-cuda` only if `Tensor` needs it (check; probably not).
- Depended on by: everything else.
- Constraint: per-layer weight structs used by `LayerParams` have to live here, or `LayerParams` has to move up. This is the main open design question.
- Cost: any edit here rebuilds all four upper crates. Keep it small and rarely edited, which is hard with a 12k-line `model.rs`.

### lightnet-layers

- What it is: the CPU reference implementation of the 59 layer families.
- Must do: hold `layers/*/cpu.rs` and the tests that exercise only CPU code.
- Depends on: `lightnet-model`.
- Depended on by: `lightnet-runtime`, `lightnet-train`.
- Constraint: no `GpuExecutableNetwork`, no `crate::train`, no `crate::detect`. `carpn/cpu.rs` needs the `detect` helpers it uses moved down (`anchor_pair`, `decode_obb_bbox`, `finite_or_zero`, `head_activation`).
- Conflict with project rule: `CLAUDE.md` says each layer's CPU reference and GPU glue live together in `layers/<layer>/` (`cpu.rs`, `gpu.rs`, `mod.rs`). Moving `gpu.rs` into the runtime crate breaks that rule. See [Open questions](#open-questions).

### lightnet-runtime

- What it is: executable networks on CPU and on a backend, plus the model-file and export code that needs them.
- Must do: hold `runtime/`, the `gpu.rs` glue (so the layer-to-runtime cycle becomes a one-way edge), `weights`, `initialize`, `muon_runtime`, and `onnx`.
- Depends on: `lightnet-model`, `lightnet-layers`, `lightnet-cuda`.
- Depended on by: `lightnet-train`.
- Constraint: `runtime` to `train` edges (12 imports) must be cut by moving the shared label and loss types into `lightnet-model`.

### lightnet-train

- What it is: training loops, losses, data augmentation, evaluation, detection decode and OBB code.
- Must do: hold `train`, `dense_loss`, `augmentation`, `loader_profile`, `evaluate`, `detect`, `obb_training`, `obb_geometry`, `obb_labels`, `cuboid_labels`, `yolo_3d`, `segmentation_data`, `chart`, `image`.
- Depends on: `lightnet-model`, `lightnet-layers`, `lightnet-runtime`.
- Depended on by: the facade only.
- Constraint: `detect.rs` also feeds `layers/carpn`; the shared helpers move down as noted above.

### lightnet-core facade

- What it is: the existing crate name, reduced to `pub use`.
- Must do: re-export every moved module under its old path (`pub use lightnet_model::model;` keeps `lightnet_core::model` and `crate::model` working), so the CLI, the Oxide crate and the examples need no source change.
- Depends on: all four new crates.
- Constraint: `pub(crate)` items (`activation_math`, `muon`, `muon_runtime`) become `pub` in their new crate and are marked `#[doc(hidden)]`.

## Implementation plan

### Phase 0 measure and decide

- [ ] Record a baseline on this machine (4 threads, `MemoryHigh=3G` as usual): clean `cargo build -p lightnet-core --timings`, then the incremental time after touching one file each in `train.rs`, a `runtime/` file, a `layers/*/cpu.rs` file and `model.rs`; also the host test binary link time (`cargo xtask test <filter>` with a one-test filter).
- [ ] Write the numbers into this file.
- [ ] Decide go or no-go. Suggested rule: continue only if touching `train.rs` or a layer file rebuilds most of the crate today. If rebuilds are dominated by the link step or debug info, Phase 1 alone is the answer.

### Phase 1 cheap wins

- [ ] `[profile.dev] debug = "line-tables-only"` in the workspace `Cargo.toml`.
- [ ] Try `lld` or `mold` as linker through `.cargo/config.toml`. Note that `third_party/cuda-oxide` and `lightnet-cuda-oxide` use their own workspace and profile; check that the change does not disturb them (`opt-level = 1` there is required for atomics).
- [ ] Re-measure as in Phase 0. Stop here if the gain is enough.

### Phase 2 layer the modules inside one crate

Each step is a refactor in the existing crate, verified with `cargo xtask check` after it (project rule: check after each layer when moving layer code).

- [ ] Add `tools/check_module_layers.sh`: greps for forbidden `crate::` edges (for example `layers/**/cpu.rs` importing `detect`, `train` or `runtime::Gpu*`) and fails on a new one. Run it in every later step.
- [ ] Inventory `LayerParams`: list each variant and where its payload type is defined. Decide base-crate placement (see [Open questions](#open-questions)).
- [ ] Break cycle 1: for each of `depth_anything`, `dpt_head`, `tucker_attention`, `mambavision`, `kimi_kda`, `vit`, `laya_act_head`, move the geometry, validation and weight-count code that `model.rs` calls into `layers/<x>/geometry.rs`, with no runtime or math dependency. Then `model.rs` imports only those files.
- [ ] Break cycle 3: move `YoloLabel` and the other shared label and loss types out of `train.rs` into one module that `runtime` and `layers` import.
- [ ] Remove the three stray upward edges from `layers/` (`hyperbolic_foveation/gpu.rs` to `train`, `carpn/cpu.rs` to `detect`, the `segmentation/cpu.rs` doc reference).
- [ ] Move `Tensor` from `runtime/types.rs` next to `TensorShape` so `layers/*/cpu.rs` no longer needs `runtime` for it.
- [ ] Run `cargo xtask check`, `gpu-check`, full `cargo xtask test`.

### Phase 3 extract lightnet-model

- [ ] Create the crate; move the modules listed in [lightnet-model](#lightnet-model) with `git mv` so history follows.
- [ ] In `lightnet-core/src/lib.rs`, add `pub use lightnet_model::{...}` for each moved module path.
- [ ] Fix visibility: `pub(crate)` to `pub` plus `#[doc(hidden)]` where needed.
- [ ] Tests that use `super::*` move with their module. Tests that reach into a private item across the new boundary get a small `pub(crate)`-to-`pub` change or move to the crate that owns the item.
- [ ] Check that `lightnet-cuda-oxide` (separate workspace and `Cargo.lock`) still builds: the new crate must be a path dependency there as well. Run one Oxide test through `cargo xtask oxide`.

### Phase 4 extract lightnet-layers

- [ ] Create the crate; move `layers/*/cpu.rs`, each layer's `mod.rs` and CPU-only tests.
- [ ] Decide `gpu.rs` placement first (see [Open questions](#open-questions)); the default in this plan is to move them to `lightnet-runtime`.
- [ ] Re-export through the facade.

### Phase 5 extract lightnet-train

- [ ] Create `lightnet-runtime` (if not already split out in Phase 4) and `lightnet-train` with the modules listed above.
- [ ] Re-export through the facade. Confirm `lightnet-cli` builds with no source edits.

## Verification

Run after every phase, serially, at 4 threads (project rule), through `cargo xtask`:

- `cargo xtask check` and `cargo xtask gpu-check`, no new errors.
- `cargo xtask test` over the whole workspace: the count of passing tests must equal the count before the phase (1,059 `#[test]` functions at the start; record the exact passing count in Phase 0).
- `cargo xtask clippy`, no new warnings.
- One Oxide test file that exercises the moved code, for example `cargo xtask oxide --test oxide_runtime_chain -- native_dense_head_loss` (needs `CARGO_INCREMENTAL=0`, which `tools/test_oxide.sh` now sets).
- `lightnet-cli`, `lightnet-cuda-oxide` and `lightnet-examples` compile with no source change. If one needs an edit, the facade is missing a re-export.
- `tools/check_module_layers.sh` passes.
- Re-run the Phase 0 timings after Phase 3 and again after Phase 5 and compare with the baseline. If the numbers did not improve, say so in this file.

## Risks

- A base crate edit rebuilds everything above it, so the split helps only edits made above it. Most of this project's recent edits sit in the middle (`runtime/`, `gpu_update.rs`), which is why Phase 0 comes first.
- About 1,059 unit tests use private items through `super::*`. Each module boundary needs a pass over them.
- `LayerParams` placement can force a lot of weight-struct code into the base crate, which makes it large and hard to keep stable.
- The Oxide workspace has its own `Cargo.lock` and `[profile.dev]`; a new crate can fail to resolve or change codegen there.
- The linker failure fixed on 2026-10-31 (undefined hidden `anon.*.llvm.*` symbol from swin `Arc<[usize]>::drop_slow`) shows incremental codegen is fragile in this tree; a crate split changes codegen-unit partitioning and could move it.
- Splitting adds public surface (`pub(crate)` to `pub`). Mark it `#[doc(hidden)]` and keep the facade as the only documented API.

## Open questions

- Where should `layers/*/gpu.rs` live? Leaving them in `lightnet-layers` needs a trait or registry so the runtime can call them without a dependency cycle; moving them to `lightnet-runtime` is simpler but breaks the project rule that a layer's CPU and GPU code sit together.
- Should `LayerParams` and the per-layer weight structs live in `lightnet-model`, or should `lightnet-runtime` own `LayerParams` and layers return plain weight blobs?
- Keep the crate names `lightnet-model` / `lightnet-layers` / `lightnet-runtime` / `lightnet-train`, or use the names from the original question (`lightnet-core-conv`, `lightnet-core-train`)?
- Is this worth doing for compile time alone? Phase 0 and Phase 1 exist to answer that before any file moves.

## Execution decisions and Phase 0 status (2026-10-31)

- Use `lightnet-model`, `lightnet-layers`, `lightnet-runtime`, and `lightnet-train`; `lightnet-core` remains a pure re-export facade.
- Move `layers/*/gpu.rs` into `lightnet-runtime`; keep `cpu.rs` in `lightnet-layers`. This explicitly deviates from the CPU/GPU co-location rule in `CLAUDE.md` and the layer layout contract.
- Inventory `LayerParams` first in Phase 2. Prefer keeping per-layer weight structs in `lightnet-model` if workable; stop and report if that would place more than about one third of `layers/` in the base crate.
- Phase 0 is **blocked, not completed**: its clean build with Cargo timings is incompatible with the instruction to use only the `check`, `gpu-check`, `clippy`, `test`, and `oxide` xtask commands and not run `cargo clean`. No clean or incremental build timings, link time, or go/no-go decision were obtained.
- Attempted baseline `JOBS=2 MEM=3G cargo xtask test`: xtask reported `systemd-run --user unavailable, running UNCAPPED`; the command was interrupted immediately (exit 130) to honor the required soft memory cap. No passing test count or test timing was obtained. No phase gates were run and no source files were changed.

## Phase 2 LayerParams inventory

User decision: GO without Phase 0 measurements; skip Phase 1. No build or test is authorized in this editing turn. The inventory below records definitions before any crate extraction; `runtime/types.rs` means `src/lightnet-core/src/runtime/types.rs`, and layer paths are relative to `src/lightnet-core/src/layers/`.

| Variant | Payload | Definition |
|---|---|---|
| None | none | — |
| RecursiveBlock | RecursiveBlockParams | runtime/types.rs |
| FoveationFilm | FoveationFilmParams | runtime/types.rs |
| HyperbolicFoveation | HyperbolicFoveationParams | runtime/types.rs |
| BatchNorm | BatchNormParams | runtime/types.rs |
| Convolutional | ConvolutionalParams | runtime/types.rs |
| Connected | ConnectedParams | runtime/types.rs |
| Local | LocalParams | runtime/types.rs |
| Shortcut | ShortcutParams | runtime/types.rs |
| Gru | GruParams | runtime/types.rs |
| Rnn | RnnParams | runtime/types.rs |
| Lstm | LstmParams | runtime/types.rs |
| Crnn | CrnnParams | runtime/types.rs |
| LayaScorer | LayaScorerWeights | laya_scorer/cpu.rs |
| LayaActHead | LayaActHeadWeights | laya_act_head/cpu.rs |
| LayaEncoder | LayaEncoderWeights | laya_encoder/cpu.rs |
| LayaHead | LayaHeadWeights | laya_head/cpu.rs |
| DepthAnything | DepthAnythingWeights | depth_anything/cpu.rs |
| DptHead | DptHeadWeights | dpt_head/cpu.rs |
| Vit | VitWeights | vit/cpu.rs |
| Swin | SwinWeights | swin/cpu.rs |
| TuckerAttention | TuckerAttentionParams | tucker_attention/cpu.rs |
| MambaVision | MambaVisionParams | mambavision/cpu.rs |
| KimiKda | KimiKdaParams | kimi_kda/cpu.rs |
| Moe | MoeParams | runtime/types.rs |
| EmlConvolutional | EmlConvolutionalParams | runtime/types.rs |
| Dcnv4 | Dcnv4Params | runtime/types.rs |
| GraphConv | GraphConvParams | runtime/types.rs |
| DeformConv | DeformConvParams | runtime/types.rs |
| DsConv | DsConvParams | runtime/types.rs |
| Wmhf | WmhfParams | runtime/types.rs |
| Clifford | CliffordParams | clifford/cpu.rs |

Twelve payload definitions occur in 12 of the 59 layer directories; 19 are already in runtime types. This count does not show that moving the data definitions would put over one third of `layers/` into the model crate. Continue Phase 2; reassess the actual supporting code before moving those payloads in a later phase. Some types have private fields or geometry-dependent methods, so the inventory alone does not prove extraction is workable.

Phase 2 source-edit result: added the boundary guard and inventory; extracted the seven requested geometry groups, shared training/detection types and helpers, and `Tensor`, retaining prior paths via re-exports. Verification is pending the lead: before/after test counts and timings are unavailable because Phase 0 and all builds/tests were skipped by user instruction. The grep-based guard itself has not been run. No Phase 3 work was started.
