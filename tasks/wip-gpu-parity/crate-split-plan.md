# Splitting lightnet-core into crates

The split is implemented (see Result); sections after it are the original plan and phase notes. Numbers marked "measured" come from `grep`/`wc` on the tree at `bb98f0f` (2026-10-31); everything else is a proposal. It answers the question asked on 2026-10-31: can `lightnet-core` become `lightnet-core-conv` and `lightnet-core-train`, mainly to cut rebuild time?

## Table of Contents

- [Result](#result)
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
- [Phase 3 closure result](#phase-3-closure-result)
- [Phase 4 source extraction result](#phase-4-source-extraction-result)
- [Phase 5 result](#phase-5-result)

## Result

The split is done (2026-10-31). The sections below are the original plan and the per-phase notes; this paragraph is the outcome.

Final graph: `lightnet-model <- lightnet-layers <- lightnet-runtime <- lightnet-train <- lightnet-core`, all on `lightnet-cuda`; `lightnet-core` is a pure facade (`pub use lightnet_train::*`) so `lightnet-cli`, `lightnet-examples`, `lightnet-gui*`, `lightnet-lm`, `lightnet-cuda-oxide` and `xtask` needed no source edit. The production graph is acyclic and `tools/check_module_layers.sh` enforces it. Tests that span several crates (runtime tests, model tests, carpn, gru and laya_scorer wiring tests) live in `lightnet-core`. Items they and the upper crates needed from lower crates were widened to `pub` plus `#[doc(hidden)]`; the list is in [Phase 5 result](#phase-5-result).

Deviation from the old `CLAUDE.md` rule: a layer's CPU reference and GPU glue no longer share one folder. `cpu.rs` is in `lightnet-layers/src/<layer>/`, `gpu.rs` is in `lightnet-runtime/src/layers/<layer>/`, because the glue needs `GpuExecutableNetwork` and the CPU math must sit below it. `clifford/cpu.rs` stayed in the runtime crate with the executor-side Clifford code. `CLAUDE.md` and `src/lightnet-core/src/layers/CONTEXT.md` now state the new rule. Phase 0 timings were never taken, so the rebuild-time gain is unmeasured.

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

## Phase 3 closure result

The user reports Phase 2 was committed as `c9d1fe0` and passed check, GPU check, clippy, and 1,059 host tests. Phase 3 is stopped before extraction under the explicit prohibited-dependency rule:

- `model.rs:13` imports `lightnet_cuda::{F32ActivationKind, F32ChannelNormalization}`; public methods at lines 431 and 462 use those backend types.
- `model.rs:2811` calls `clifford::raw_channels`, defined in `layers/clifford/cpu.rs:219`; line 5058 uses `laya_scorer::LAYA_LAYERNORM_EPS`, defined in `layers/laya_scorer/cpu.rs:34`; line 8139 calls `swin::padded_shape`, defined in `layers/swin/cpu.rs:429`.
- The tests in the moved `model.rs` would also reach layer CPU parameters (`crate::KimiKdaParams` at line 9290 and `crate::mambavision::MambaVisionParams` at line 9382) and OBB training/geometry (`crate::obb_training` at line 10985 and `crate::obb_geometry` at line 11066). These tests cannot be dropped or weakened.
- `model/yolo_3d_config.rs` is a child module of `model.rs`. The seven existing geometry modules reference model/error types; `training_types.rs` has a documentation link to `dense_loss` that would need adjustment if moved.

No crate, manifest, source move, visibility change, or guard update was made. The Oxide manifest has a path dev-dependency on `lightnet-core`, its own workspace and a dev profile with `opt-level = 1`; its separate lockfile currently lists `lightnet-core` and `lightnet-cuda`. If extraction is subsequently authorized and completed, the lead must verify path resolution and lockfile handling in that workspace and preserve that profile. No Phase 3 build or test was run.

Follow-up decisions: `lightnet-model` may depend on `lightnet-cuda`; extract the Clifford, LAYA scorer and Swin helpers into geometry modules; keep tests requiring core in a separate core test module. The three helper extractions were inspected and require no runtime code. Eight model tests need the core-only home. The first required `git mv` failed because `.git/index.lock` cannot be created on the read-only filesystem. Partial source edits were removed and the source restored to the starting state. Phase 3 remains incomplete; no visibility changes or build/test results were produced. When resumed, include `lightnet-model` in the host test package selection so its moved tests run.

### Phase 3 source extraction result

The user authorized plain `mv` because the Git index is read-only, and allowed the `lightnet-cuda` dependency. Extracted the model crate and retained old module paths in core. The three additional helper geometry files were created during this extraction. Eight core-dependent tests were relocated to `model_tests.rs`. Static source inventory finds 1,059 `#[test]` functions before and after across core plus model; this is not a passing test count. No build, check, lint, guard script or test was run. The four Phase 2 re-export warnings were left alone. Preserve the Oxide workspace profile and let Cargo resolve both lockfiles during lead verification. Host test selection must explicitly include `lightnet-model` because the xtask default does not.

Cross-crate visibility widened to `pub` with `#[doc(hidden)]`: the `activation_math` module and its 20 previously crate-private functions; `Tensor::{channel_area,batch_stride,offset}`; `DptHeadSpec::geometry`; `dcnv4_parameter_layout`; the new `clifford::geometry::raw_channels` and `swin::geometry::padded_shape`; and the geometry declarations below. The geometry declarations were widened conservatively and require compiler review for minimum visibility. Paths below refer to the corresponding files under `src/lightnet-model/src/layers/`; line numbers are the pre-widening line numbers.

```text
src/lightnet-model/src/layers/vit/geometry.rs:6:pub(super) fn vit_weight_sizes(g: &VitGeometry, out_h: usize, out_w: usize) -> Result<[usize; 16], String> {
src/lightnet-model/src/layers/tucker_attention/geometry.rs:6:pub(crate) struct TuckerGeometry {
src/lightnet-model/src/layers/tucker_attention/geometry.rs:23:pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
src/lightnet-model/src/layers/tucker_attention/geometry.rs:30:pub(crate) fn tucker_weight_count(
src/lightnet-model/src/layers/tucker_attention/geometry.rs:67:pub(crate) fn validate_geometry(g: &TuckerGeometry) -> Result<(), String> {
src/lightnet-model/src/layers/mambavision/geometry.rs:5:pub(crate) struct MambaVisionGeometry {
src/lightnet-model/src/layers/mambavision/geometry.rs:6:    pub(super) batch: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:7:    pub(super) channels: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:8:    pub(super) height: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:9:    pub(super) width: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:10:    pub(super) filters: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:11:    pub(super) state_size: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:12:    pub(super) conv_size: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:13:    pub(super) dt_rank: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:14:    pub(super) ffn_ratio: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:15:    pub(super) activation: Activation,
src/lightnet-model/src/layers/mambavision/geometry.rs:16:    pub(super) msa_enabled: bool,
src/lightnet-model/src/layers/mambavision/geometry.rs:17:    pub(super) msa_block_size: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:18:    pub(super) msa_topk: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:19:    pub(super) msa_heads: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:20:    pub(super) causal: bool,
src/lightnet-model/src/layers/mambavision/geometry.rs:24:    pub(crate) fn from_layer(input: TensorShape, spec: &MambaVisionSpec) -> Result<Self, String> {
src/lightnet-model/src/layers/mambavision/geometry.rs:88:    pub(super) fn tokens(&self) -> usize {
src/lightnet-model/src/layers/mambavision/geometry.rs:92:    pub(super) fn rows(&self) -> usize {
src/lightnet-model/src/layers/mambavision/geometry.rs:96:    pub(super) fn half(&self) -> usize {
src/lightnet-model/src/layers/mambavision/geometry.rs:100:    pub(super) fn hidden(&self) -> usize {
src/lightnet-model/src/layers/mambavision/geometry.rs:104:    pub(super) fn projection_width(&self) -> usize {
src/lightnet-model/src/layers/mambavision/geometry.rs:109:pub(crate) fn validate_geometry(input: TensorShape, spec: &MambaVisionSpec) -> Result<(), String> {
src/lightnet-model/src/layers/mambavision/geometry.rs:113:pub(crate) struct MambaParamBlock {
src/lightnet-model/src/layers/mambavision/geometry.rs:120:pub(super) struct MambaOffsets {
src/lightnet-model/src/layers/mambavision/geometry.rs:121:    pub(super) in_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:122:    pub(super) in_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:123:    pub(super) conv_x_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:124:    pub(super) conv_x_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:125:    pub(super) conv_z_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:126:    pub(super) conv_z_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:127:    pub(super) x_proj_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:128:    pub(super) dt_proj_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:129:    pub(super) dt_proj_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:130:    pub(super) a_log: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:131:    pub(super) d_skip: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:132:    pub(super) out_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:133:    pub(super) out_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:134:    pub(super) res_w: Option<Range<usize>>,
src/lightnet-model/src/layers/mambavision/geometry.rs:135:    pub(super) ln1_gamma: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:136:    pub(super) ln1_beta: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:137:    pub(super) ln2_gamma: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:138:    pub(super) ln2_beta: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:139:    pub(super) ffn1_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:140:    pub(super) ffn1_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:141:    pub(super) ffn2_w: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:142:    pub(super) ffn2_b: Range<usize>,
src/lightnet-model/src/layers/mambavision/geometry.rs:143:    pub(super) total: usize,
src/lightnet-model/src/layers/mambavision/geometry.rs:147:    pub(super) fn new(
src/lightnet-model/src/layers/mambavision/geometry.rs:238:    pub(super) fn blocks(&self) -> Vec<MambaParamBlock> {
src/lightnet-model/src/layers/mambavision/geometry.rs:359:pub(crate) fn mambavision_weight_blocks(
src/lightnet-model/src/layers/mambavision/geometry.rs:374:pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:6:pub(crate) struct KimiKdaGeometry {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:21:    pub(crate) fn validate(&self) -> Result<(), String> {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:71:    pub(crate) fn from_layer(
src/lightnet-model/src/layers/kimi_kda/geometry.rs:98:    pub(crate) fn tokens(&self) -> usize {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:102:    pub(crate) fn rows(&self) -> usize {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:107:pub(crate) fn validate_geometry(
src/lightnet-model/src/layers/kimi_kda/geometry.rs:115:pub(crate) struct KimiKdaWeightBlock {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:120:pub(crate) fn kimi_kda_parameter_counts(
src/lightnet-model/src/layers/kimi_kda/geometry.rs:130:pub(super) struct KimiOffsets {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:131:    pub(super) wq: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:132:    pub(super) wk: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:133:    pub(super) wv: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:134:    pub(super) wo: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:135:    pub(super) wbeta: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:136:    pub(super) wdecay: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:137:    pub(super) wrouter: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:138:    pub(super) w1: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:139:    pub(super) w2: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:140:    pub(super) total_weights: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:141:    pub(super) ln1_gamma: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:142:    pub(super) ln1_beta: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:143:    pub(super) ln2_gamma: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:144:    pub(super) ln2_beta: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:145:    pub(super) bq: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:146:    pub(super) bk: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:147:    pub(super) bv: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:148:    pub(super) bbeta: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:149:    pub(super) bdecay: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:150:    pub(super) bo: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:151:    pub(super) brouter: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:152:    pub(super) b1: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:153:    pub(super) b2: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:154:    pub(super) total_biases: usize,
src/lightnet-model/src/layers/kimi_kda/geometry.rs:158:    pub(super) fn new(channels: usize, rank: usize, experts: usize, hidden: usize) -> Result<Self, String> {
src/lightnet-model/src/layers/kimi_kda/geometry.rs:234:pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:3:pub(crate) const DPT_WEIGHT_SLOTS: usize = 66;
src/lightnet-model/src/layers/dpt_head/geometry.rs:7:pub(crate) struct DptHeadGeometry {
src/lightnet-model/src/layers/dpt_head/geometry.rs:21:    pub(crate) fn output_height(self) -> Result<usize> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:27:    pub(crate) fn output_width(self) -> Result<usize> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:33:    pub(crate) fn validate(self) -> Result<()> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:105:pub(crate) struct DptWeightRange {
src/lightnet-model/src/layers/dpt_head/geometry.rs:111:pub(crate) struct DptHeadWeightLayout {
src/lightnet-model/src/layers/dpt_head/geometry.rs:117:    pub(crate) fn get(&self, slot: usize) -> DptWeightRange {
src/lightnet-model/src/layers/dpt_head/geometry.rs:122:pub(super) fn invalid(message: impl Into<String>) -> Error {
src/lightnet-model/src/layers/dpt_head/geometry.rs:126:pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:139:pub(crate) fn dpt_head_weight_layout(geometry: DptHeadGeometry) -> Result<DptHeadWeightLayout> {
src/lightnet-model/src/layers/dpt_head/geometry.rs:208:pub(crate) fn dpt_weight_slot_name(slot: usize) -> String {
src/lightnet-model/src/layers/dpt_head/geometry.rs:212:pub(super) fn reassembled_size(size: usize, factor: usize) -> Result<usize> {
src/lightnet-model/src/layers/depth_anything/geometry.rs:3:pub(super) const RGB_CHANNELS: usize = 3;
src/lightnet-model/src/layers/depth_anything/geometry.rs:152:    pub(super) values: Vec<f32>,
src/lightnet-model/src/layers/depth_anything/geometry.rs:236:pub(super) struct WeightField {
src/lightnet-model/src/layers/depth_anything/geometry.rs:237:    pub(super) start: usize,
src/lightnet-model/src/layers/depth_anything/geometry.rs:238:    pub(super) len: usize,
src/lightnet-model/src/layers/depth_anything/geometry.rs:239:    pub(super) stride: usize,
src/lightnet-model/src/layers/depth_anything/geometry.rs:243:    pub(super) fn range(self) -> std::ops::Range<usize> {
src/lightnet-model/src/layers/depth_anything/geometry.rs:247:    pub(super) fn block(self, index: usize) -> std::ops::Range<usize> {
src/lightnet-model/src/layers/depth_anything/geometry.rs:254:pub(super) struct WeightOffsets {
src/lightnet-model/src/layers/depth_anything/geometry.rs:255:    pub(super) patch_w: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:256:    pub(super) patch_b: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:257:    pub(super) cls_token: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:258:    pub(super) pos_embed: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:259:    pub(super) norm_gamma: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:260:    pub(super) norm_beta: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:261:    pub(super) ln1_gamma: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:262:    pub(super) ln1_beta: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:263:    pub(super) qkv_w: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:264:    pub(super) qkv_b: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:265:    pub(super) proj_w: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:266:    pub(super) proj_b: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:267:    pub(super) ls1: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:268:    pub(super) ln2_gamma: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:269:    pub(super) ln2_beta: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:270:    pub(super) fc1_w: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:271:    pub(super) fc1_b: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:272:    pub(super) fc2_w: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:273:    pub(super) fc2_b: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:274:    pub(super) ls2: WeightField,
src/lightnet-model/src/layers/depth_anything/geometry.rs:275:    pub(super) total: usize,
src/lightnet-model/src/layers/depth_anything/geometry.rs:278:pub(super) fn weight_offsets(g: &DepthAnythingGeometry) -> Result<WeightOffsets, String> {
src/lightnet-model/src/layers/depth_anything/geometry.rs:357:pub(super) fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
src/lightnet-model/src/layers/depth_anything/geometry.rs:367:pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {

```

The widened activation functions are `loggy`, `loggy_grad`, `loggy_grad_from_output`, `gelu`, `gelu_grad`, `elu`, `elu_grad`, `selu`, `selu_grad`, `relie`, `plse`, `plse_grad`, `hardtan`, `hardtan_grad`, `lhtan`, `lhtan_grad`, `stair`, `stair_grad`, `hard_mish`, and `hard_mish_grad`. The model crate also exposes its new `layers` and layer geometry modules as public hidden infrastructure; the old core layer module paths remain as re-exports.

## Phase 4 source extraction result

The user reports Phase 3 check, gpu-check, and clippy passed; 973 core and 36 CLI unit tests passed before the host run was interrupted. Phase 4 remains unverified: no build, lint, boundary script, or test was run during this source-edit turn. A static count of `#[test]` declarations across core, model, and layers remains 1,059; that is not a passing test count.

Moved `src/lightnet-core/src/layers/<name>/cpu.rs` to `src/lightnet-layers/src/<name>/cpu.rs` for every name below; each new directory has a CPU module wrapper while the core wrapper retains GPU code and old paths:
- `carpn`
- `centernet`
- `channel_shuffle`
- `channel_slice`
- `contrastive`
- `dcnv4`
- `deconvolutional`
- `deform_conv`
- `depth_anything`
- `dpt_head`
- `dsconv`
- `eml_conv_math`
- `foveation_film`
- `graph_conv`
- `gru`
- `hyperbolic_foveation`
- `jev_ce`
- `kimi_kda`
- `laya_act_head`
- `laya_encoder`
- `laya_head`
- `laya_loss`
- `laya_marker_gather`
- `laya_scorer`
- `mambavision`
- `maxpool_depth`
- `modern_yolo`
- `moe_conv`
- `omd_loss`
- `segmentation`
- `shortcut_weights`
- `swin`
- `tucker_attention`
- `vc_attention`
- `vit`
- `wmhf`

Also moved `src/lightnet-core/src/layers/swin/block.rs` to `src/lightnet-layers/src/swin/block.rs`; `src/lightnet-core/src/axis_fixture.rs` to `src/lightnet-model/src/axis_fixture.rs`; and `src/lightnet-core/src/detection_geometry.rs` to `src/lightnet-model/src/detection_geometry.rs`. Extracted runtime pure parameter and training-buffer definitions to `src/lightnet-layers/src/params.rs` and Clifford pure parameter definitions to `src/lightnet-layers/src/clifford_params.rs`; these are partial-file moves. Clifford CPU execution remains in core because it uses `WmhfConvRuntimeCache`, `forward_wmhf_convolution`, and `backward_wmhf_convolution`. Runtime-dependent GRU, CARPN and LAYA scorer tests remain in core in their respective `tests.rs`/`wiring_tests.rs` modules.

Visibility changes: restricted declarations in the moved CPU files are public with `#[doc(hidden)]` for cross-crate use, conservatively widened pending compiler review. The following inventory identifies those declarations and their current paths; it also includes previously hidden declarations retained by the moved files. The `axis_fixture` constant/functions, four detection geometry functions, `BATCH_NORM_EPSILON`, and Clifford helpers `zero_gradients_like`, `has_local`, `has_global`, `has_fourier` were likewise made public hidden.

```text
src/lightnet-layers/src/gru/cpu.rs:9:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-10-pub struct Step<T> {
--
src/lightnet-layers/src/gru/cpu.rs:14:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-15-pub fn projections(p:&GruParams)->[&ConnectedParams;6] {
--
src/lightnet-layers/src/gru/cpu.rs:18:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-19-pub fn projections_mut(p:&mut GruParams)->[&mut ConnectedParams;6] {
--
src/lightnet-layers/src/gru/cpu.rs:22:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-23-pub fn serialized(p:&GruParams)->Vec<f32> {
--
src/lightnet-layers/src/gru/cpu.rs:26:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-27-pub fn set_serialized(p:&mut GruParams,values:&[f32])->Result<()> {
--
src/lightnet-layers/src/gru/cpu.rs:37:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-38-pub fn validate(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
--
src/lightnet-layers/src/gru/cpu.rs:49:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-50-pub fn validate_cpu(s:&GruSpec,p:&GruParams,input:TensorShape,output:TensorShape)->Result<usize> {
--
src/lightnet-layers/src/gru/cpu.rs:68:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-69-pub fn tape(s:&GruSpec,p:&GruParams,input:&Tensor,output:TensorShape)->Result<Vec<Step<Vec<f32>>>> {
--
src/lightnet-layers/src/gru/cpu.rs:98:#[doc(hidden)]
src/lightnet-layers/src/gru/cpu.rs-99-pub fn backward(s:&GruSpec,p:&GruParams,input:&Tensor,grad:&Tensor)->Result<(Tensor,GruParams)> {
--
src/lightnet-layers/src/carpn/cpu.rs:53:#[doc(hidden)]
src/lightnet-layers/src/carpn/cpu.rs-54-pub fn forward_refine_layer(
--
src/lightnet-layers/src/carpn/cpu.rs:169:#[doc(hidden)]
src/lightnet-layers/src/carpn/cpu.rs-170-pub fn collect_proposals(
--
src/lightnet-layers/src/carpn/cpu.rs:308:#[doc(hidden)]
src/lightnet-layers/src/carpn/cpu.rs-309-pub fn sample_proposal_features(
--
src/lightnet-layers/src/carpn/cpu.rs:356:#[doc(hidden)]
src/lightnet-layers/src/carpn/cpu.rs-357-pub fn sample_feature_bilinear(
--
src/lightnet-layers/src/dcnv4/cpu.rs:11:#[doc(hidden)]
src/lightnet-layers/src/dcnv4/cpu.rs-12-pub struct Dcnv4ForwardCache {
--
src/lightnet-layers/src/dcnv4/cpu.rs:20:#[doc(hidden)]
src/lightnet-layers/src/dcnv4/cpu.rs-21-pub struct Dcnv4Gradients {
--
src/lightnet-layers/src/dcnv4/cpu.rs:388:#[doc(hidden)]
src/lightnet-layers/src/dcnv4/cpu.rs-389-pub fn dcnv4_forward(
--
src/lightnet-layers/src/dcnv4/cpu.rs:544:#[doc(hidden)]
src/lightnet-layers/src/dcnv4/cpu.rs-545-pub fn dcnv4_backward(
--
src/lightnet-layers/src/params.rs:3:#[doc(hidden)]
src/lightnet-layers/src/params.rs-4-pub const BATCH_NORM_EPSILON: f32 = 0.000_01;
--
src/lightnet-layers/src/channel_slice/cpu.rs:11:#[doc(hidden)]
src/lightnet-layers/src/channel_slice/cpu.rs-12-pub fn slice_channels(source: &Tensor, start: usize, end: usize) -> Result<Tensor> {
--
src/lightnet-layers/src/channel_slice/cpu.rs:31:#[doc(hidden)]
src/lightnet-layers/src/channel_slice/cpu.rs-32-pub fn slice_channels_backward(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:19:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-20-pub struct HfParams {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:33:    #[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-34-    pub fn effective_radius(self) -> f32 {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:43:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-44-pub struct HfPointEval {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:51:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-52-pub struct HfInverseResult {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:60:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-61-pub struct HfInverseParamJacobian {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:69:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-70-pub struct BilinearSample {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:77:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-78-pub struct WarpGradients {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:86:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-87-pub struct PredictorForwardCache {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:96:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-97-pub struct PredictorGradients {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:136:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-137-pub fn smooth_max_radius(radius_x: f32, radius_y: f32) -> f32 {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:147:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-148-pub fn phi_eval(point: [f32; 2], params: HfParams) -> HfPointEval {
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:202:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-203-pub fn inverse_point(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:459:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-460-pub fn inverse_point_param_jacobian(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:513:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-514-pub fn bilinear_sample_plane(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:559:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-560-pub fn warp_nchw(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:609:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-610-pub fn warp_nchw_backward(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:675:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-676-pub fn average_pool_downsample_nchw(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:729:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-730-pub fn predictor_forward_nchw(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:842:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-843-pub fn predictor_backward_nchw(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:991:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-992-pub fn sampling_grid_nchw(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:1025:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-1026-pub fn transform_box_with_jacobian(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:1071:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-1072-pub fn warp_parameter_gradients_from_grid_gradient(
--
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs:1117:#[doc(hidden)]
src/lightnet-layers/src/hyperbolic_foveation/cpu.rs-1118-pub fn warp_backward_with_param_gradients(
--
src/lightnet-layers/src/wmhf/cpu.rs:6:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-7-pub struct HaarCoefficients {
--
src/lightnet-layers/src/wmhf/cpu.rs:21:    #[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-22-    pub fn zeros(
--
src/lightnet-layers/src/wmhf/cpu.rs:46:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-47-pub struct FuseConcatGradients {
--
src/lightnet-layers/src/wmhf/cpu.rs:54:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-55-pub struct GateGradients {
--
src/lightnet-layers/src/wmhf/cpu.rs:152:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-153-pub fn split_channels(
--
src/lightnet-layers/src/wmhf/cpu.rs:180:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-181-pub fn insert_channels(
--
src/lightnet-layers/src/wmhf/cpu.rs:215:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-216-pub fn concat_local(
--
src/lightnet-layers/src/wmhf/cpu.rs:242:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-243-pub fn split_local_gradient(
--
src/lightnet-layers/src/wmhf/cpu.rs:267:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-268-pub fn concat_fuse(
--
src/lightnet-layers/src/wmhf/cpu.rs:324:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-325-pub fn split_fuse_gradient(
--
src/lightnet-layers/src/wmhf/cpu.rs:373:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-374-pub fn haar_dwt(
--
src/lightnet-layers/src/wmhf/cpu.rs:412:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-413-pub fn haar_dwt_backward(coeff_grads: &HaarCoefficients) -> Result<Vec<f32>> {
--
src/lightnet-layers/src/wmhf/cpu.rs:492:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-493-pub fn haar_iwt(coeffs: &HaarCoefficients) -> Result<Vec<f32>> {
--
src/lightnet-layers/src/wmhf/cpu.rs:534:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-535-pub fn haar_iwt_backward(
--
src/lightnet-layers/src/wmhf/cpu.rs:573:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-574-pub fn scan4_forward(
--
src/lightnet-layers/src/wmhf/cpu.rs:688:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-689-pub fn scan4_backward(
--
src/lightnet-layers/src/wmhf/cpu.rs:849:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-850-pub fn high_frequency_energy_upsample(coeffs: &HaarCoefficients) -> Result<Vec<f32>> {
--
src/lightnet-layers/src/wmhf/cpu.rs:882:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-883-pub fn high_frequency_energy_backward(
--
src/lightnet-layers/src/wmhf/cpu.rs:933:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-934-pub fn apply_gate(
--
src/lightnet-layers/src/wmhf/cpu.rs:951:#[doc(hidden)]
src/lightnet-layers/src/wmhf/cpu.rs-952-pub fn apply_gate_backward(
--
src/lightnet-layers/src/graph_conv/cpu.rs:13:#[doc(hidden)]
src/lightnet-layers/src/graph_conv/cpu.rs-14-pub struct GraphConvForwardCache {
--
src/lightnet-layers/src/graph_conv/cpu.rs:32:#[doc(hidden)]
src/lightnet-layers/src/graph_conv/cpu.rs-33-pub struct GraphConvGradients {
--
src/lightnet-layers/src/graph_conv/cpu.rs:176:#[doc(hidden)]
src/lightnet-layers/src/graph_conv/cpu.rs-177-pub fn graph_conv_forward(
--
src/lightnet-layers/src/graph_conv/cpu.rs:522:#[doc(hidden)]
src/lightnet-layers/src/graph_conv/cpu.rs-523-pub fn graph_conv_backward(
--
src/lightnet-layers/src/vit/cpu.rs:400:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-401-pub struct FlashAttentionBackwardInputs {
--
src/lightnet-layers/src/vit/cpu.rs:413:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-414-pub fn attention_flash_backward_inputs(
--
src/lightnet-layers/src/vit/cpu.rs:523:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-524-pub fn attention_forward_vc(
--
src/lightnet-layers/src/vit/cpu.rs:1031:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-1032-pub fn mhc_coefficients(params: &[f32], site: usize) -> Result<(f32, f32, f32, f32, f32), String> {
--
src/lightnet-layers/src/vit/cpu.rs:1283:    #[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-1284-    pub fn serialized_parts(&self) -> [&[f32]; 16] {
--
src/lightnet-layers/src/vit/cpu.rs:1291:    #[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-1292-    pub fn set_serialized(&mut self, values: &[f32]) -> Result<(), String> {
--
src/lightnet-layers/src/vit/cpu.rs:1501:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-1502-pub fn tensor_activation(activation: &Activation, x: f32) -> Result<f32, String> {
--
src/lightnet-layers/src/vit/cpu.rs:1540:#[doc(hidden)]
src/lightnet-layers/src/vit/cpu.rs-1541-pub fn tensor_activation_grad(
--
src/lightnet-layers/src/deform_conv/cpu.rs:17:#[doc(hidden)]
src/lightnet-layers/src/deform_conv/cpu.rs-18-pub struct DeformConvForwardCache {
--
src/lightnet-layers/src/deform_conv/cpu.rs:58:#[doc(hidden)]
src/lightnet-layers/src/deform_conv/cpu.rs-59-pub struct DeformConvGradients {
--
src/lightnet-layers/src/deform_conv/cpu.rs:422:#[doc(hidden)]
src/lightnet-layers/src/deform_conv/cpu.rs-423-pub fn deform_conv_forward(
--
src/lightnet-layers/src/deform_conv/cpu.rs:752:#[doc(hidden)]
src/lightnet-layers/src/deform_conv/cpu.rs-753-pub fn deform_conv_backward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:6:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-7-pub enum ExpCastMode {
--
src/lightnet-layers/src/vc_attention/cpu.rs:30:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-31-pub fn e4m3_encode_bits(value: f32) -> u8 {
--
src/lightnet-layers/src/vc_attention/cpu.rs:87:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-88-pub fn e4m3_to_f32(code: u8) -> f32 {
--
src/lightnet-layers/src/vc_attention/cpu.rs:104:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-105-pub fn expcast_probability_code(u: f32, mode: ExpCastMode) -> Result<u8, String> {
--
src/lightnet-layers/src/vc_attention/cpu.rs:149:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-150-pub fn spatial8_permutation(grid_h: usize, grid_w: usize) -> Result<Vec<usize>, String> {
--
src/lightnet-layers/src/vc_attention/cpu.rs:197:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-198-pub struct VsmoothCache {
--
src/lightnet-layers/src/vc_attention/cpu.rs:213:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-214-pub fn vsmooth_forward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:341:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-342-pub fn vsmooth_backward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:460:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-461-pub struct QuantizedQkCache {
--
src/lightnet-layers/src/vc_attention/cpu.rs:486:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-487-pub struct QkScaleGradients {
--
src/lightnet-layers/src/vc_attention/cpu.rs:494:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-495-pub fn quantize_qk(
--
src/lightnet-layers/src/vc_attention/cpu.rs:674:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-675-pub fn quantized_qk_log2_scores(
--
src/lightnet-layers/src/vc_attention/cpu.rs:717:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-718-pub fn new_qk_scale_gradients(cache: &QuantizedQkCache) -> QkScaleGradients {
--
src/lightnet-layers/src/vc_attention/cpu.rs:727:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-728-pub fn accumulate_qk_scale_gradients(
--
src/lightnet-layers/src/vc_attention/cpu.rs:790:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-791-pub fn quantized_qk_scale_input_backward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:860:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-861-pub struct VcHeadCache {
--
src/lightnet-layers/src/vc_attention/cpu.rs:877:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-878-pub fn vc_head_forward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:924:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-925-pub fn vc_head_backward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:970:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-971-pub struct QuantizedOnlineRowCache {
--
src/lightnet-layers/src/vc_attention/cpu.rs:987:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-988-pub fn quantized_online_row_forward(
--
src/lightnet-layers/src/vc_attention/cpu.rs:1095:#[doc(hidden)]
src/lightnet-layers/src/vc_attention/cpu.rs-1096-pub fn quantized_online_row_backward(
--
src/lightnet-layers/src/dsconv/cpu.rs:30:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-31-pub struct DsConvForwardCache {
--
src/lightnet-layers/src/dsconv/cpu.rs:42:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-43-pub struct DsConvGradients {
--
src/lightnet-layers/src/dsconv/cpu.rs:53:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-54-pub struct DsConvMultiViewCache {
--
src/lightnet-layers/src/dsconv/cpu.rs:60:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-61-pub struct DsConvMultiViewGradients {
--
src/lightnet-layers/src/dsconv/cpu.rs:66:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-67-pub fn dsconv_multi_view_forward(
--
src/lightnet-layers/src/dsconv/cpu.rs:110:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-111-pub fn dsconv_multi_view_backward(
--
src/lightnet-layers/src/dsconv/cpu.rs:387:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-388-pub fn dsconv_forward(
--
src/lightnet-layers/src/dsconv/cpu.rs:551:#[doc(hidden)]
src/lightnet-layers/src/dsconv/cpu.rs-552-pub fn dsconv_backward(
--
src/lightnet-layers/src/jev_ce/cpu.rs:14:#[doc(hidden)]
src/lightnet-layers/src/jev_ce/cpu.rs-15-pub fn jev_ce_probs(logits: &[f32], spec: &JevCeSpec, batch: usize) -> Vec<f32> {
--
src/lightnet-layers/src/jev_ce/cpu.rs:38:#[doc(hidden)]
src/lightnet-layers/src/jev_ce/cpu.rs-39-pub struct JevCeLoss {
--
src/lightnet-layers/src/jev_ce/cpu.rs:49:#[doc(hidden)]
src/lightnet-layers/src/jev_ce/cpu.rs-50-pub fn jev_ce_loss(
--
src/lightnet-layers/src/mambavision/cpu.rs:7:#[doc(hidden)]
src/lightnet-layers/src/mambavision/cpu.rs-8-pub use super::geometry::{MambaOffsets, MambaVisionGeometry};
--
src/lightnet-layers/src/mambavision/cpu.rs:95:    #[doc(hidden)]
src/lightnet-layers/src/mambavision/cpu.rs-96-    pub fn validate(&self) -> Result<(), String> {
--
src/lightnet-layers/src/mambavision/cpu.rs:115:    #[doc(hidden)]
src/lightnet-layers/src/mambavision/cpu.rs-116-    pub fn matches(&self, g: &MambaVisionGeometry) -> bool {
--
src/lightnet-layers/src/mambavision/cpu.rs:1252:#[doc(hidden)]
src/lightnet-layers/src/mambavision/cpu.rs-1253-pub fn mambavision_param_blocks(
--
src/lightnet-layers/src/eml_conv_math/cpu.rs:7:#[doc(hidden)]
src/lightnet-layers/src/eml_conv_math/cpu.rs-8-pub struct EmlConvMathParams {
--
src/lightnet-layers/src/eml_conv_math/cpu.rs:17:    #[doc(hidden)]
src/lightnet-layers/src/eml_conv_math/cpu.rs-18-    pub fn from_cpp_values(clamp: f32, eps: f32, scale: f32) -> Self {
--
src/lightnet-layers/src/eml_conv_math/cpu.rs:28:#[doc(hidden)]
src/lightnet-layers/src/eml_conv_math/cpu.rs-29-pub struct EmlConvMathGrad {
--
src/lightnet-layers/src/eml_conv_math/cpu.rs:59:#[doc(hidden)]
src/lightnet-layers/src/eml_conv_math/cpu.rs-60-pub fn forward(
--
src/lightnet-layers/src/eml_conv_math/cpu.rs:87:#[doc(hidden)]
src/lightnet-layers/src/eml_conv_math/cpu.rs-88-pub fn backward(
--
src/lightnet-layers/src/deconvolutional/cpu.rs:8:#[doc(hidden)]
src/lightnet-layers/src/deconvolutional/cpu.rs-9-pub fn deconv_forward(
--
src/lightnet-layers/src/deconvolutional/cpu.rs:49:#[doc(hidden)]
src/lightnet-layers/src/deconvolutional/cpu.rs-50-pub struct DeconvGrad {
--
src/lightnet-layers/src/deconvolutional/cpu.rs:58:#[doc(hidden)]
src/lightnet-layers/src/deconvolutional/cpu.rs-59-pub fn deconv_backward(
--
src/lightnet-layers/src/swin/cpu.rs:13:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-14-pub struct WindowAttentionCache {
--
src/lightnet-layers/src/swin/cpu.rs:31:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-32-pub struct WindowAttentionGradients {
--
src/lightnet-layers/src/swin/cpu.rs:127:    #[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-128-    pub fn serialized_parts(&self) -> Vec<&[f32]> {
--
src/lightnet-layers/src/swin/cpu.rs:138:    #[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-139-    pub fn set_serialized(&mut self, values: &[f32]) -> Result<(), String> {
--
src/lightnet-layers/src/swin/cpu.rs:144:    #[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-145-    pub fn matches_geometry(
--
src/lightnet-layers/src/swin/cpu.rs:426:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-427-pub fn pad_nchw(
--
src/lightnet-layers/src/swin/cpu.rs:465:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-466-pub fn crop_nchw(
--
src/lightnet-layers/src/swin/cpu.rs:504:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-505-pub fn cyclic_shift_nchw(
--
src/lightnet-layers/src/swin/cpu.rs:542:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-543-pub fn window_partition_nchw(
--
src/lightnet-layers/src/swin/cpu.rs:595:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-596-pub fn window_unpartition_nchw(
--
src/lightnet-layers/src/swin/cpu.rs:650:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-651-pub fn build_attention_mask(
--
src/lightnet-layers/src/swin/cpu.rs:729:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-730-pub fn relative_position_indices(window: usize) -> Result<Vec<usize>, String> {
--
src/lightnet-layers/src/swin/cpu.rs:768:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-769-pub fn mhc_parameters(
--
src/lightnet-layers/src/swin/cpu.rs:790:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-791-pub fn mhc_residual_forward(
--
src/lightnet-layers/src/swin/cpu.rs:810:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-811-pub fn mhc_residual_backward(
--
src/lightnet-layers/src/swin/cpu.rs:895:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-896-pub fn window_attention_forward(
--
src/lightnet-layers/src/swin/cpu.rs:916:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-917-pub fn window_attention_forward_with_indices(
--
src/lightnet-layers/src/swin/cpu.rs:1045:#[doc(hidden)]
src/lightnet-layers/src/swin/cpu.rs-1046-pub fn window_attention_backward(
--
src/lightnet-layers/src/laya_act_head/cpu.rs:36:#[doc(hidden)]
src/lightnet-layers/src/laya_act_head/cpu.rs-37-pub fn cpu_forward_call_count() -> usize {
--
src/lightnet-layers/src/foveation_film/cpu.rs:7:#[doc(hidden)]
src/lightnet-layers/src/foveation_film/cpu.rs-8-pub struct FoveationFilmForwardCache {
--
src/lightnet-layers/src/foveation_film/cpu.rs:18:#[doc(hidden)]
src/lightnet-layers/src/foveation_film/cpu.rs-19-pub struct FoveationFilmGradients {
--
src/lightnet-layers/src/foveation_film/cpu.rs:74:#[doc(hidden)]
src/lightnet-layers/src/foveation_film/cpu.rs-75-pub fn forward_nchw(
--
src/lightnet-layers/src/foveation_film/cpu.rs:156:#[doc(hidden)]
src/lightnet-layers/src/foveation_film/cpu.rs-157-pub fn backward_nchw(
--
src/lightnet-layers/src/dpt_head/cpu.rs:7:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-8-pub use super::geometry::{DPT_WEIGHT_SLOTS, DptHeadGeometry, DptHeadWeightLayout, dpt_head_weight_layout};
--
src/lightnet-layers/src/dpt_head/cpu.rs:11:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-12-pub const DPT_MUON_WEIGHT_SLOTS: [usize; 32] = [
--
src/lightnet-layers/src/dpt_head/cpu.rs:16:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-17-pub const DPT_BIAS_SLOTS: [usize; 28] = [
--
src/lightnet-layers/src/dpt_head/cpu.rs:28:    #[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-29-    pub fn zeroed(geometry: DptHeadGeometry) -> Result<Self> {
--
src/lightnet-layers/src/dpt_head/cpu.rs:38:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-39-pub struct DptHeadGradients {
--
src/lightnet-layers/src/dpt_head/cpu.rs:46:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-47-pub struct DptHeadForwardCache {
--
src/lightnet-layers/src/dpt_head/cpu.rs:138:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-139-pub fn forward(
--
src/lightnet-layers/src/dpt_head/cpu.rs:398:#[doc(hidden)]
src/lightnet-layers/src/dpt_head/cpu.rs-399-pub fn backward(
--
src/lightnet-layers/src/modern_yolo/cpu.rs:1153:    #[doc(hidden)]
src/lightnet-layers/src/modern_yolo/cpu.rs-1154-    pub fn spec() -> ModernYoloSpec {
--
src/lightnet-layers/src/modern_yolo/cpu.rs:1168:    #[doc(hidden)]
src/lightnet-layers/src/modern_yolo/cpu.rs-1169-    pub fn dfl_spec(kind: ModernYoloKind, classes: usize, reg_max: usize) -> ModernYoloSpec {
--
src/lightnet-layers/src/maxpool_depth/cpu.rs:34:#[doc(hidden)]
src/lightnet-layers/src/maxpool_depth/cpu.rs-35-pub fn forward_maxpool_depth(input: &Tensor, output_shape: TensorShape) -> Result<Tensor> {
--
src/lightnet-layers/src/maxpool_depth/cpu.rs:61:#[doc(hidden)]
src/lightnet-layers/src/maxpool_depth/cpu.rs-62-pub fn backward_maxpool_depth(input: &Tensor, output_gradient: &Tensor) -> Result<Tensor> {
--
src/lightnet-layers/src/kimi_kda/cpu.rs:7:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-8-pub use super::geometry::{KimiKdaGeometry, KimiOffsets, checked_product};
--
src/lightnet-layers/src/kimi_kda/cpu.rs:88:    #[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-89-    pub fn validate(&self) -> Result<(), String> {
--
src/lightnet-layers/src/kimi_kda/cpu.rs:104:    #[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-105-    pub fn matches(&self, g: &KimiKdaGeometry) -> bool {
--
src/lightnet-layers/src/kimi_kda/cpu.rs:141:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-142-pub struct KimiKdaGradients {
--
src/lightnet-layers/src/kimi_kda/cpu.rs:149:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-150-pub fn kimi_kda_weight_blocks(
--
src/lightnet-layers/src/kimi_kda/cpu.rs:197:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-198-pub fn initialize_kimi_kda_params(
--
src/lightnet-layers/src/kimi_kda/cpu.rs:294:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-295-pub fn kimi_kda_forward(
--
src/lightnet-layers/src/kimi_kda/cpu.rs:538:#[doc(hidden)]
src/lightnet-layers/src/kimi_kda/cpu.rs-539-pub fn kimi_kda_backward(
--
src/lightnet-layers/src/lib.rs:10:#[doc(hidden)]
src/lightnet-layers/src/lib.rs-11-pub mod clifford_params;
--
src/lightnet-layers/src/shortcut_weights/cpu.rs:10:#[doc(hidden)]
src/lightnet-layers/src/shortcut_weights/cpu.rs-11-pub fn effective_weights(
--
src/lightnet-layers/src/shortcut_weights/cpu.rs:45:#[doc(hidden)]
src/lightnet-layers/src/shortcut_weights/cpu.rs-46-pub fn weighted_forward(
--
src/lightnet-layers/src/shortcut_weights/cpu.rs:74:#[doc(hidden)]
src/lightnet-layers/src/shortcut_weights/cpu.rs-75-pub struct WeightedGrad {
--
src/lightnet-layers/src/shortcut_weights/cpu.rs:87:#[doc(hidden)]
src/lightnet-layers/src/shortcut_weights/cpu.rs-88-pub fn weighted_backward(
--
src/lightnet-layers/src/shortcut_weights/cpu.rs:129:#[doc(hidden)]
src/lightnet-layers/src/shortcut_weights/cpu.rs-130-pub fn normalized_weight_gradient(
--
src/lightnet-layers/src/clifford_params.rs:106:    #[doc(hidden)]
src/lightnet-layers/src/clifford_params.rs-107-    pub fn zero_gradients_like(&self) -> Self {
--
src/lightnet-layers/src/clifford_params.rs:207:#[doc(hidden)]
src/lightnet-layers/src/clifford_params.rs-208-pub fn has_local(spec: &CliffordSpec) -> bool {
--
src/lightnet-layers/src/clifford_params.rs:212:#[doc(hidden)]
src/lightnet-layers/src/clifford_params.rs-213-pub fn has_global(spec: &CliffordSpec) -> bool {
--
src/lightnet-layers/src/clifford_params.rs:217:#[doc(hidden)]
src/lightnet-layers/src/clifford_params.rs-218-pub fn has_fourier(spec: &CliffordSpec) -> bool {
--
src/lightnet-layers/src/swin/block.rs:21:#[doc(hidden)]
src/lightnet-layers/src/swin/block.rs-22-pub struct SwinGeometry {
--
src/lightnet-layers/src/swin/block.rs:219:#[doc(hidden)]
src/lightnet-layers/src/swin/block.rs-220-pub fn swin_forward(
--
src/lightnet-layers/src/swin/block.rs:442:#[doc(hidden)]
src/lightnet-layers/src/swin/block.rs-443-pub fn swin_backward(
--
src/lightnet-layers/src/tucker_attention/cpu.rs:110:    #[doc(hidden)]
src/lightnet-layers/src/tucker_attention/cpu.rs-111-    pub fn matches_geometry(&self, g: &TuckerGeometry) -> bool {
--
src/lightnet-layers/src/tucker_attention/cpu.rs:145:#[doc(hidden)]
src/lightnet-layers/src/tucker_attention/cpu.rs-146-pub struct TuckerAttentionGradients {
--
src/lightnet-layers/src/tucker_attention/cpu.rs:377:#[doc(hidden)]
src/lightnet-layers/src/tucker_attention/cpu.rs-378-pub fn tucker_attention_forward(
--
src/lightnet-layers/src/tucker_attention/cpu.rs:475:#[doc(hidden)]
src/lightnet-layers/src/tucker_attention/cpu.rs-476-pub fn tucker_attention_backward(
--
src/lightnet-layers/src/channel_shuffle/cpu.rs:11:#[doc(hidden)]
src/lightnet-layers/src/channel_shuffle/cpu.rs-12-pub fn shuffle_channels(x: &Tensor, groups: usize, inverse: bool) -> Result<Tensor> {

```

The root and core manifests now include `lightnet-layers`; Cargo must update lockfiles, including the separate Oxide workspace lockfile, and its existing dev `opt-level = 1` must be preserved. Host `PKGS` must include `lightnet-layers` and `lightnet-model` along with core and CLI.

## Phase 5 result

Phase 5 is implemented. `lightnet-runtime` and `lightnet-train` exist, `lightnet-core` is a facade (`src/lib.rs` is `pub use lightnet_train::*;` plus three `#[cfg(test)]` modules), and both new crates are root workspace members and default members. The separate Oxide workspace needed no manifest change: it depends on `lightnet-core` by path, and the new crates resolve transitively; its `[profile.dev] opt-level = 1` is untouched.

### Where the cycle gate landed

The gate reported earlier (runtime tests need training and detection while keeping access to private runtime items) is resolved by keeping those tests above the runtime, not by adding a `runtime -> train` edge:

- `image` and `loader_profile` moved into `lightnet-runtime` (the runtime names `PlanarRgbImage`; `image` uses `loader_profile`). `lightnet-train` re-exports them.
- `dense_loss` is only a doc reference from the runtime, so it moved to `lightnet-train`.
- Tests that touch several layers of the stack stay in `lightnet-core` as `#[cfg(test)]` modules and reach the crates by public path: `src/runtime_tests/` (the old `runtime/tests/` tree), `src/model_tests.rs`, and, wired by `src/layer_tests.rs`, `layers/carpn/tests.rs`, `layers/gru/tests.rs` and `layers/laya_scorer/wiring_tests.rs`. Runtime-private items they used are now `pub` and `#[doc(hidden)]` (inventory below).

### Files moved (old to new)

| From `lightnet-core/src` | To |
|---|---|
| `runtime/` (including `CONTEXT.md`) | `lightnet-runtime/src/runtime/` (its `tests/` subtree stays in core as `runtime_tests/`) |
| `layers/mod.rs`, every `layers/<layer>/mod.rs` and `gpu.rs`, `layers/clifford/cpu.rs` | `lightnet-runtime/src/layers/` |
| `weights.rs`, `initialize.rs`, `muon.rs`, `muon_runtime.rs`, `onnx/`, `image.rs`, `loader_profile.rs` | `lightnet-runtime/src/` |
| `train.rs`, `dense_loss.rs`, `augmentation.rs`, `evaluate.rs`, `detect.rs`, `obb_training.rs`, `obb_geometry.rs`, `obb_labels.rs`, `cuboid_labels{.rs,/}`, `yolo_3d{.rs,/}`, `segmentation_data.rs`, `chart.rs` | `lightnet-train/src/` |

`layers/CONTEXT.md`, the cross-crate test files and the untracked draft files (`centernet/train.rs`, `modern_yolo/*_train.rs`) were not moved. The `lightnet-core/tests/` integration tests and their `fixtures/` and `data/` stay; the moved unit tests in `image.rs` and `augmentation.rs` read them through a `../lightnet-core/tests/...` path.

Each new crate's `lib.rs` re-exports the one below with the same module names (`lightnet-train` does `pub use lightnet_runtime::*;`), so `crate::runtime::`, `crate::model::` and so on inside moved code needed no edit.

### Visibility inventory

All changes are `pub` plus `#[doc(hidden)]`, in `lightnet-runtime` unless stated. None changes behaviour.

- Layer modules: every `pub(crate) mod` and `pub(crate) use` in `layers/mod.rs` and `layers/*/mod.rs` (including each `gpu` module and the `cpu`, `geometry`, `block` aliases) is now `pub`. The `layers` module is `#[doc(hidden)] pub`, and `lib.rs` re-exports the old crate-private layer names (`channel_shuffle`, `dsconv`, `swin`, `vit`, `wmhf` and the rest) as `#[doc(hidden)] pub use`. `muon`, `muon_runtime`, `activation_math`, `detection_geometry` and `axis_fixture` are `#[doc(hidden)] pub`.
- `runtime/mod.rs`: the glob re-exports of `host_adam`, `param_updates`, `weights_io`, `forward_layers`, `layer_ops`, `layers::hyperbolic_foveation::gpu` and the named `layer_ops` list are `#[doc(hidden)] pub use`; `BATCH_NORM_ROLLING_KEEP` and `BATCH_NORM_ROLLING_KEEP_CPU` are `pub`.
- Types: every `pub(crate)` or `pub(super)` struct, enum, type alias or trait in `runtime/*.rs`, `layers/*/gpu.rs`, `layers/clifford/cpu.rs`, `muon.rs` and `muon_runtime.rs` (about 45 definitions, for example `HyperbolicFoveationForwardCache`, `MuonLayerState`, `MuonMatrixState`, the per-layer `DeviceCache` types).
- Fields: every non-public field of `ForwardResult`, `GpuForwardState`, `ExecutableNetwork` and `GpuExecutableNetwork` (`runtime/types.rs`, `gpu_types.rs`, `exec_core.rs`); `HostAdamLayerState.{weights,biases,scales,children}`; `dcnv4::gpu::ForwardCache.weights`.
- Functions and methods (name collisions across files widened a few extra overloads, for example `needs_host` in `clifford`, `wmhf` and `dcnv4`, and `activate` in `clifford/gpu.rs`): `accumulate_train_buffers`, `activate`, `activation_delta`, `activation_gradient_from_output`, `activation_output`, `advance_rng`, `backward_dropout`, `backward_maxpool`, `backward_reorg`, `backward_route`, `backward_sam`, `backward_scale_channels`, `backward_seeded_core`, `backward_seeded_with_grad_mode`, `backward_softmax`, `backward_upsample`, `batchnorm_channel_stats`, `dcnv4::gpu::backward`, `depth_anything_input_gradient`, `device_conv_muon_eligible`, `dropblock_plan`, `eml_adam_state_for_update`, `forward_convolutional`, `forward_dropout`, `forward_maxpool`, `forward_reorg`, `forward_route`, `forward_sam`, `forward_scale_channels`, `forward_softmax`, `forward_training_with_rng_seed`, `forward_upsample`, `has_host_only_deform`, `hyperbolic_foveation_predictor_initial_biases`, `matrix_shapes`, `needs_host`, `plan_crop_forward`, `reject_gpu_hyperbolic_foveation`, `requires_cpu_backward_fallback`, `resolve_relative_layer`, `sanitize_vit_updates`, `scale_vit_updates`, `sgd_update`, `softmax_group_size`, `tensor_backend_shape`, `update_batch_norm_running_stats`, `update_cpu_fallback_weights`, `visit_muon_matrices`, `vit_update_clip_scale`, `warm_device_conv_params`, `write_muon_layer_state`.
- `lightnet-train`: no visibility change. `detect.rs` needed `ForwardResult.hyperbolic_foveation_caches` and its type, covered above.
- `lightnet-layers`: new cargo feature `test-hooks` (off by default) gates `laya_act_head::cpu_forward_call_count` and its counter with `cfg(any(test, feature = "test-hooks"))`, because a test in `lightnet-core` reads it and `cfg(test)` does not reach a dependency. `lightnet-core` enables it through a dev-dependency.
- Other: `lightnet-core` no longer lists the unused `imageproc` dependency. `tools/test_host.sh`, `xtask` and `CLAUDE.md` default package lists name all six crates.

### Final crate graph

```text
lightnet-model       (-> lightnet-cuda)
   ▲
lightnet-layers      (-> model, lightnet-cuda)
   ▲
lightnet-runtime     (-> model, layers, lightnet-cuda; image, mozjpeg, half)
   ▲
lightnet-train       (-> model, layers, runtime, lightnet-cuda; image, rayon, half)
   ▲
lightnet-core        facade (-> all of the above); dev-dependency on layers with test-hooks
   ▲
lightnet-cli, lightnet-examples, lightnet-gui*, lightnet-lm, lightnet-cuda-oxide (dev)
```

`tools/check_module_layers.sh` guards the edges: model, layers and runtime never name `train` or `detect`, runtime never names `lightnet_train` or `lightnet_core`, and `layers/*/cpu.rs` never names `runtime::Gpu*`.

### Verification

Run during the work (all through `cargo xtask`, `JOBS=4 MEM=4G`): `tools/check_module_layers.sh` passes; `cargo xtask check` (default six packages plus CLI) and `cargo xtask gpu-check` compiled with no errors; the test binaries for all packages compiled (`cargo xtask test <no-match filter>`). A full `cargo xtask test` was started and stopped on request before the slow `every_cfg_file_parses_constructs_and_executes_cpu_inference` corpus test finished: up to that point `lightnet-cli` 36 passed and `lightnet-core` unit tests 342 passed, with 0 failures. The `lightnet-model`, `lightnet-layers`, `lightnet-runtime` and `lightnet-train` unit-test binaries did not run.

Not run: `cargo xtask clippy`, the full pass counts for model, layers, runtime and train, and the Oxide check (`cargo xtask oxide --test oxide_runtime_chain -- native_dense_head_loss`), so the separate Oxide workspace resolving and linking the new path dependencies is unverified. Static `#[test]` count is unchanged at 1,059 in sources (plus 2 integration tests in `lightnet-core/tests`). Remaining warnings known: unused `use self::tests_03::*` and `use self::tests_07::*` in `lightnet-core/src/runtime_tests/mod.rs`.
