# Port Every Missing Darknet Layer Type to Rust — Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Implementation status (2026-10-05).** Phase 0 (Tasks 1–12) is **delivered**
> — its step checkboxes below are left as the plan's TDD recipe, not as a status
> list. Phase 1b (`segmentation`, `omd_loss`, `contrastive` classifier,
> `laya_scorer`, `laya_marker_gather`), Phase 2A (all anchor-free heads) and
> Phase 2B `moe_convolutional`'s CPU path are done; the Phase 3A shared
> primitives (LayerNorm, residual/bias epilogue, row-softmax backward, GEMM
> wrapper) are in. **The consolidated inventory is
> [`implemented.md`](implemented.md)**; the phase tables at the end of this
> document list only what remains.

**Plan structure (read first).** This is ONE master plan for several independent subsystems, deliberately not split into separate files:
- **Phase 0** (Tier 0 = Tasks 1-6, Tier 1 = Tasks 7-11, bookkeeping = Task 12) is fully detailed: bite-sized TDD steps, real code, exact commands, expected output, commits. It also defines the reusable **Layer Port Contract** that every later layer follows.
- **Phases 1b-4** (Tiers 1b-4, below "Phase sections") are NOT bite-sized. Each lists layers, C++ source, dependencies, exact `.weights` order, math summary, test strategy, kernel needs and difficulty, and ends with **"expand into its own plan before starting"**. Do not execute them from this document.

**Goal:** `darknet-rs` parses, runs (CPU reference + cuda-oxide GPU), trains and loads/saves `.weights` for every Darknet layer type that exists on the other git branches (`Jev_laya`, `raptor*`, `fog`, `jepa`, `v6-dev-seg`, `depth_3d`, `deform`, ...) but is not fully supported on branch `rust` today, with numerics checked against C++ wherever a C++ build of the owning branch is available.

**Architecture:** Add each layer by the same recipe (the Layer Port Contract): `LayerKind` variant + `*Spec` + parse/shape inference in `model.rs`; pure host math in a small focused module (new file per family, not more lines in the 17k-line `runtime.rs`); thin CPU forward/backward arms in `runtime.rs`; a default host implementation on `F32TensorOpsBackend` (`darknet-cuda`) with an optional cuda-oxide kernel override; GPU runtime arms; `.weights` I/O in the existing `weights.rs`/`load_weights_into_params` order. Backward is always the exact analytical derivative; C++ shortcuts are gated behind `DARKNET_GRAD_MODE=cpp` (existing `GradMode`). Layers with no GPU kernel run on the GPU through a host round trip (download, host op, upload) exactly like `connected`/`sam` do today.

**Tech Stack:** Rust (`darknet-core`, `darknet-cuda`, `darknet-cli`, `src/darknet-cuda-oxide` with cuda-oxide `#[kernel]`s), Python only for the existing `tools/compare_layer_dumps.py`.

**Spec:** No separate spec; this document is the spec. Sources to ADAPT (never copy wholesale): C++ on branch `Jev_laya` (`src-lib/`, superset of most layers), `raptor` / `raptor-dev` / `raptor-fp4` / `fog` / `v6-dev-seg` (logistic, regression, segmentation, OBB), `jepa` (swin), `depth_3d` (yolo_3d), `deform`. Read with `git show "branch:path"` (quote it, zsh). Prior art for plan style: `tasks/rust/onnx.md`. State and conventions: `tasks/rust/todo.md` (P5 "CPU-fallback gaps", P8, P9, "C++ parity gap summary"). Per user memory `port-features-adapt-not-replace`: extend the current Rust code in its current style, minimal extras, no wholesale file replacement, no `*_expected` twins; ask before adding optional cfgs/examples (see Open Questions).

## Global Constraints

- Builds/tests: one cargo command at a time, `-j 2` at most, memory-capped. Host tests ONLY via `./tools/test_host.sh <filter>` (darknet-core + darknet-cli, no GPU; `PKGS=darknet-cuda ./tools/test_host.sh <filter>` for the darknet-cuda crate). GPU tests ONLY via `./tools/test_oxide.sh --test <file> -- <filter>`. Benches via `./tools/bench_oxide.sh`. Never run `cargo` directly, never `-j` above 4, never launch training runs for debugging (static analysis + unit tests + finite differences).
- Backward = exact analytical derivative of the Rust forward. C++ deviations (grad=1 for normalised shortcut weights, deconv double `1/batch`, softmax pass-through, ...) are reproduced ONLY when `GradMode::from_env() == GradMode::Cpp`.
- `.weights` byte order must match the C++ `save_*_weights` of the owning branch exactly; every layer with weights gets a round-trip test that also asserts the exact file length (`20` header bytes + `4 * parameter_count`).
- Loss heads follow the existing yolo convention: loss is computed on the host in `darknet-core`, the "seed" gradient is Darknet's descent direction (`target - prediction`), and the layer's backward arm passes the seed through to the previous layer.
- Unknown / unsupported config must be an error, never a silent identity (this plan removes the identity fallback for unknown activations in Task 2).
- Commit only the files a task names. Every commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. `docs/` is gitignored (`git add -f` if a doc is ever added; this plan adds none).
- No new Rust dependencies.

## Review Focus

The five failure modes most likely to bite a user and not obviously covered by "happy path" tests. Each is pinned by a named test in the owning task.

1. **`.weights` order mismatch when loading real C++ weights** (shortcut weights, deconvolutional, later vit/moe/dcnv3...). Expected: loading consumes the file exactly (`trailing_bytes == 0`), saving produces `20 + 4*count` bytes. Pinned: Task 6 Step 1 (`shortcut_weights_round_trip_exact_length`), Task 9 Step 1 (`deconvolutional_weights_round_trip_exact_length`), Task 5.
2. **Shape inference of new layers consumed by `route` / `shortcut` / `scale_channels`.** `channel_slice` takes its shape from the layer named by `from`, NOT from the previous layer; `deconvolutional` output is `stride*(h-1)+size` (not `h*stride`), so a following mismatched `shortcut` hits the flat crop/pad path. Expected: downstream `route`/`shortcut` see the new shape. Pinned: Task 7 Step 1 (`channel_slice_shape_comes_from_source_layer_and_feeds_route`), Task 9 Step 1 (`deconvolutional_output_is_stride_times_h_minus_one_plus_size`).
3. **C++ defaults that are invalid or keys spelled differently.** `[channel_slice]` C++ defaults `start=end=2` (an empty slice that aborts), `[deconvolutional]` rejects `pad`/`padding`, `[maxpool]` key is `maxpool_zero_nonmax` (Rust read `zero_nonmax`, so C++ cfgs were silently ignored), `[jev_ce]` `groups=81` with `candidates` inferred. Expected: a clear `InvalidConfig` error naming the section and line, or the C++ key honoured. Pinned: Tasks 4, 7, 9, 11 parse tests.
4. **Silent identity activation and wrong C++ gradients.** Rust `activate(Other(_))` returns the input unchanged; C++ falls back to relu and C++ `gelu_gradient` is evaluated at the layer OUTPUT with a mismatched formula. Expected: unknown activation = config error; gelu gradient = exact derivative evaluated at the cached pre-activation. Pinned: Task 1 (finite-difference), Task 2 (`unknown_activation_is_a_config_error_not_identity`).
5. **GPU/CPU divergence on host-round-trip layers and loss heads.** A GPU arm that forgets to upload the activation cache, or a loss seed in the wrong space (logit vs probability), trains silently wrong. Expected: GPU forward/backward equal CPU to 1e-5 on `FakeBackend` chains. Pinned: Task 7 Step 5, Task 8 Step 6, Task 9 Step 7, Task 10 Step 6, Task 11 Step 6.

## Findings that correct or refine the research report (spot-checked against the sources)

| Report claim | What the sources say |
|---|---|
| Unknown sections hit `UnsupportedLayer` at runtime (~1119/1747/4322/4700) | They already fail earlier: `infer_output_shape` returns `Error::UnsupportedLayer` for `LayerKind::Unsupported` inside `Network::from_config`, so such a cfg never builds a `Network`. The runtime arms are defensive. |
| `batchnorm`: "no weights block in weights.rs" | Half true. `Layer::weights()` and `load_weights_into_params` handle `BatchNorm`; `save_weights` has NO `BatchNorm` arm, so saving any net with `[batchnorm]` fails with "inconsistent while saving". Task 5. |
| gru / local / crop / batchnorm / detection are "legacy layers to finish" | They are NOT in `ELayerType` of `master` or `Jev_laya` (verified in `src-lib/darknet_enums.hpp`), `weights.cpp` has no case for them, and C++ never writes standalone batchnorm/gru/local weights. They are Rust-only extensions with no C++ reference to validate against. Deprioritised to the Deferred appendix. |
| `maxpool_depth` Unsupported | Rust never parses the flag: `out_channels` (C++ default 1, Rust default = input channels) is used as a proxy and spatial dims are still pooled. Also the C++ key is `maxpool_zero_nonmax`; Rust reads `zero_nonmax`. Task 4. |
| `DARKNET_LAYER_DUMP_DIR` + `tools/compare_layer_dumps.py` give C++ numerics | The Rust side (`detect-image`) writes dumps. The C++ side exists only in commit `96f662f0` (`src-lib/network_kernels.cu`, GPU inference path `not state.train`), deleted from this branch by `2514cf53`, and absent from every layer branch. See "Verification against C++". |
| Jev_laya `weights.cpp` has no TRANSFORMER/SWIN case | Verified (`jepa` has none either). The Rust order must be defined by us (Phase 3). |
| vit / moe / dcnv3 weight orders | Verified against `Jev_laya:src-lib/weights.cpp` (`save_vit_weights`, `save_moe_convolutional_weights`, `save_dcnv3_weights`). Others in Phases 2-4 are marked "from research report, not re-verified". |
| Shortcut `weights_normalization` | C++ weight gradient ignores the normalisation (`grad = 1`; the correct sum is commented out in `blas.cpp`). Exact derivative in Task 6, C++ behaviour behind `GradMode::Cpp`. |
| Loss layers (logistic, regression, jev_ce, cost) | Rust has no generic loss hook: `train.rs` only computes the yolo loss. Task 10 adds `dense_loss.rs`; wiring into the `detector train` loop is an Open Question. |

## Layer Port Contract (the reusable template; later tasks refer to it by letter)

Adding one layer touches the sites below. `grep -n "LayerKind::Reorg" src/darknet-core/src/*.rs` is a good way to find every site of an existing comparable layer. After adding a `LayerKind` variant run `./tools/test_host.sh does_not_exist` once: the compile errors list every non-exhaustive `match`.

**A. Config / model (`src/darknet-core/src/model.rs`)**
1. `pub struct <Name>Spec { .. }` next to the other specs (~line 255-470) and `LayerKind::<Name>(<Name>Spec)`.
2. `LayerKind::name()` arm (the cfg section name; it feeds error messages and `weights-inspect`).
3. `parse_layer_kind` arm (~line 1071-1380) with the C++ keys and defaults. If a C++ default is invalid, require the key and return `Error::InvalidConfig(format!("[{name}] at line {}: ...", section.line))`. Aliases go in the same match arm (`"conv" | "convolutional"` style).
4. `infer_output_shape` arm (~line 1384-1555): must use the same shape the C++ `parse_*_section` computes. `current` is the previous layer's output; layers reading another layer (`from`) must use `built_layers[resolve_relative_layer(layer_index, raw)?].output_shape` (see `ScaleChannels`).
5. `Layer::weights()` entries (~line 664-870) IN `.weights` ORDER, only when the layer has weights.
6. Tests in `model.rs` `mod tests`: parse, shape, invalid-config.

**B. Parameters (only layers with weights).** Reuse an existing `LayerParams`/`LayerTrainBuffers` variant when the byte layout is identical (deconvolutional reuses `Convolutional`). A new variant touches exactly: `runtime.rs` `LayerParams` (~218) and `LayerTrainBuffers` (~290) enums; `default_layer_params` / `default_layer_train_buffers` (~6330/6450; the zero default doubles as the zero gradient accumulator in `backward_seeded_core`); `update_weights` arm (~560-700); `accumulate_train_buffers` arm (~1920-2100); `load_weights_into_params` arm (~7079); `weights.rs` `save_weights` arm AND the exhaustive `LayerParams` list in the `recurrent_weight_blocks_round_trip_in_darknet_order` test; `initialize.rs` arm; `darknet-cli/src/main.rs` (~2361 `LayerTrainBuffers` match). The GPU update path keeps non-convolutional params in `cpu_fallback_train_buffers` (see Task 9 Step 6).

**C. CPU reference (`runtime.rs`)**
1. Forward arm in `forward_with_mode` (~line 880-1130). Return `(Tensor, Option<Tensor>)`: the second value is the pre-activation cache (`Some` whenever the layer has an activation).
2. Backward arm in `backward_seeded_core` (~1204-1760): accumulate with `accumulate_previous_gradient` (previous layer) and `accumulate_tensor(&mut layer_output_gradients[source_index], ..)` (other layers), never overwrite (C++ `copy_cpu` overwrites in `channel_slice`/`channel_shuffle`; accumulation is the only correct fan-in semantics).
3. Compact CPU inference (`forward_compact`, preflight list ~line 820-840) automatically falls back to the full `forward` for kinds not in that list. Do NOT add the new kind there.

**D. Backend op (only if there is device work).** Add a method with a DEFAULT host implementation to `F32TensorOpsBackend` in `src/darknet-cuda/src/lib.rs` calling a reference function in `src/darknet-cuda/src/host_ops.rs` (download, compute, upload; see `reorg_f32`). Then override in `src/darknet-cuda-oxide/src/lib.rs` with a kernel from `src/kernels/*.rs`, and add `src/darknet-cuda-oxide/tests/oxide_*.rs` parity + finite-difference tests. Prefer reusing existing ops (e.g. `route_concat_f32` slices channels, `conv2d_backward_input_f32` is a transposed convolution).

**E. GPU runtime (`GpuExecutableNetwork`, `runtime.rs`)**
1. Training forward arm (`forward_with_state_impl`, ~line 3790-4330). Host-round-trip layers use `self.download_tensor(current_input, layer.input_shape)`, run the CPU function, `self.backend.upload_f32(&tensor_backend_shape(layer.output_shape), &data)`, and set `activation_input_gpu = Some(uploaded pre-activation)` when the layer has an activation (see the `Sam` arm ~4248).
2. Backward arm (`backward_accumulate_seeded_gpu`, ~4780-5450) using `accumulate_previous_gpu_gradient` / `accumulate_gpu_gradient`; pass-through loss heads join the list at ~5438. Layers whose parameters live on the host go into `requires_cpu_backward_fallback` (~5460): then `forward_with_state_impl` downloads all host copies automatically and the whole backward runs on the CPU.
3. GPU compact inference (`forward_compact`, the `matches!` list ~line 4410): add the new kind there so inference falls back to the full state path. f16 inference (`forward_f16`, ~2390) already rejects unknown kinds.
4. `layer_uses_previous_input` (~7234) only matters for the compact path; kinds that do not read the previous layer (channel_slice) should return `false` if they are ever added to a compact arm.

**F. Tests (all must exist for every layer)**
- parse + shape (`model.rs` tests); weights round-trip with exact length (`weights.rs` tests, uses `ExecutableNetwork::new` + `save_weights` + `load_weights_file`);
- CPU forward vs a hand-computed reference (numbers written in the test, derived independently of the implementation);
- CPU backward vs central finite differences of `L = sum(seed * forward)` (the pure-function module owns this test);
- GPU vs CPU: `assert_gpu_layer_chain_matches_cpu(cfg_text, seed_layer)` in `runtime.rs` tests (needs a convolution before the layer so weight gradients are compared and a chain with no CPU-fallback layer), or a fallback-style test modelled on `gpu_local_avgpool_backward_feeds_previous_convolution` (~15640) for host-param layers; plus `tests/oxide_*.rs` parity when a kernel exists.
- cfg fixture: inline cfg strings inside the tests. Shipping a file under `cfg/` is an Open Question (user previously rejected some experimental cfgs).

**G. Bookkeeping.** One line in `tasks/rust/todo.md` pointing at the finished task; the C++ parity check from "Verification against C++" is run when a C++ build of the branch exists, otherwise the skipped check is written down in `todo.md`.

## File Structure (Phase 0)

```
src/darknet-core/src/
├── lib.rs                  MODIFY  declare the new modules; re-export dense loss API
├── model.rs                MODIFY  Activation variants; LayerKind variants + Specs; parse/shape; weights()
├── runtime.rs              MODIFY  activate()/gradient arms; thin CPU+GPU arms per layer; ShortcutParams; tests
├── weights.rs              MODIFY  save arms (batchnorm, shortcut, deconvolutional) + round-trip tests
├── initialize.rs           MODIFY  shortcut weights init = 1.0
├── activation_math.rs      CREATE  scalar forward + exact derivative of 10 activations (+ tests)
├── channel_ops.rs          CREATE  channel_slice / channel_shuffle host math (+ tests)
├── deconv.rs               CREATE  transposed-convolution forward/backward (+ tests)
├── shortcut_weights.rs     CREATE  weighted-shortcut forward/backward with exact normalisation derivative (+ tests)
├── jev_ce.rs               CREATE  jev_ce softmax + soft-target CE (+ tests)
└── dense_loss.rs           CREATE  host loss + seeds for [cost] [logistic] [regression] [jev_ce] (+ tests)
src/darknet-cuda/src/
├── lib.rs                  MODIFY  F32ActivationKind::Gelu; trait op channel_shuffle_f32 (default host impl)
└── host_ops.rs             MODIFY  channel_permute (+ tests)
src/darknet-cuda-oxide/
├── src/kernels/activations.rs  MODIFY  ACT_GELU
├── src/kernels/layers_extra.rs MODIFY  channel_permute kernel
├── src/lib.rs                  MODIFY  activation_code(Gelu); channel_shuffle_f32 override
└── tests/oxide_parity.rs, oxide_cfg_shapes.rs, oxide_half.rs, oxide_batchnorm_adam.rs, oxide_layers_extra.rs  MODIFY
src/darknet-cli/src/main.rs  MODIFY  LayerTrainBuffers::Shortcut arm (~2361)
tasks/rust/todo.md              MODIFY  pointer to this plan (final task)
```

Responsibilities: the `*.rs` math modules know only tensors and numbers (no `LayerKind`); `runtime.rs` arms only glue them to `Tensor`/`LayerParams`/backend calls; `model.rs` knows only cfg semantics.

---

# PHASE 0 — Tier 0: finish what is parsed but wrong or missing

### Task 1: Ten missing elementwise activations (CPU) with exact derivatives

Adds `gelu elu selu relie plse hardtan lhtan stair hard_mish revleaky`. All ten are cheap on CPU; only `gelu` gets a GPU kernel (Task 3) because every transformer-family layer defaults to it. `normalize_channels*` are NOT elementwise (they need the layer geometry and exist only in `[convolutional]`); Task 2 initially rejected them. v65/v66 now integrate them for CPU and f32 GPU convolution, while other layer types and f16 inference remain explicit errors. `eml` belongs to `eml_convolutional` (Phase 2).

**Files:**
- Create: `src/darknet-core/src/activation_math.rs`
- Modify: `src/darknet-core/src/lib.rs` (add `pub(crate) mod activation_math;` after `pub mod error;`)
- Modify: `src/darknet-core/src/model.rs:213-264` (`Activation` enum, `parse`, `to_cuda_f32_kind`)
- Modify: `src/darknet-core/src/runtime.rs:10498-10550` (`activation_gradient_from_output`) and `:10601-10622` (`activate`)
- Test: `activation_math.rs` `mod tests`; `runtime.rs` `mod tests` (after `relu6_and_ramp_follow_cpp_activation_and_gradient`, ~16500)

**Interfaces:**
- Produces (`pub(crate)`, all `fn(f32) -> f32`): `gelu`, `gelu_grad`, `elu`, `elu_grad`, `selu`, `selu_grad`, `relie`, `plse`, `plse_grad`, `hardtan`, `hardtan_grad`, `lhtan`, `lhtan_grad`, `stair`, `stair_grad`, `hard_mish`, `hard_mish_grad`. Every `*_grad` takes the PRE-activation `x`.
- Produces: `Activation::{Gelu, Elu, Selu, Relie, Plse, Hardtan, Lhtan, Stair, HardMish, Revleaky}` (parse names `gelu elu selu relie plse hardtan lhtan stair hard_mish revleaky`).
- Consumes: existing `activate(activation: Activation, value: f32) -> f32`, `activation_gradient_from_output(&Activation, output_value, Option<f32> activation_input, layer_name) -> Result<f32>` in `runtime.rs`.

Math (C++ `activations.hpp` forward; derivative = exact derivative of that forward):
- gelu: `0.5 x (1 + tanh(0.797885 x + 0.035677 x^3))`; `d/dx = 0.5(1+t) + 0.5 x (1-t^2)(0.797885 + 3*0.035677 x^2)`, `t = tanh(u)`. (C++ `gelu_gradient` uses different constants, is applied to the OUTPUT by the conv code, and is wrong; we do not reproduce it.)
- elu: `x>=0 ? x : e^x-1`, grad `x>=0 ? 1 : e^x`. selu: `1.0507*(x>=0 ? x : 1.6732*(e^x-1))`, grad `x>=0 ? 1.0507 : 1.0507*1.6732*e^x`.
- relie: `x>0 ? x : 0.01x`. plse: `x<-4: 0.01(x+4)`, `x>4: 0.01(x-4)+1`, else `0.125x+0.5`; grad `|x|>4 ? 0.01 : 0.125` (C++ `(x<0||x>1)` is wrong).
- hardtan: clamp(-1,1), grad `|x|<1`. lhtan: `x<0: 0.001x`, `x>1: 0.001(x-1)+1`, else `x`; grad `0<x<1 ? 1 : 0.001`.
- stair: `n=floor(x)`; `n` even: `floor(x/2)`; odd: `(x-n)+floor(x/2)`; grad `n` odd ? 1 : 0 (C++ returns 1 for non-integer x regardless of parity).
- hard_mish: `x>0: x`, `x>-2: x^2/2+x`, else 0; grad `x>0: 1`, `x>-2: x+1`, else 0.
- revleaky: forward = leaky (`x>0 ? x : 0.1x`), gradient = leaky gradient (C++ CPU does this; the C++ CUDA kernel divides by it, which is not the derivative of anything).

- [ ] **Step 1: Write the failing tests** — create `src/darknet-core/src/activation_math.rs` containing only the test module (the functions do not exist yet):

```rust
//! Scalar activations missing from the original Rust set (see tasks/rust/layers.md Task 1).

#[cfg(test)]
mod tests {
    use super::*;

    fn near(name: &str, got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{name}: got {got}, want {want}");
    }

    #[test]
    fn forward_values_match_cpp_formulas() {
        near("gelu(0)", gelu(0.0), 0.0, 1e-7);
        near("gelu(1)", gelu(1.0), 0.841_192, 1e-4);
        near("gelu(-1)", gelu(-1.0), -0.158_808, 1e-4);
        near("elu(-1)", elu(-1.0), -0.632_120_6, 1e-6);
        near("elu(2)", elu(2.0), 2.0, 0.0);
        near("selu(1)", selu(1.0), 1.0507, 1e-6);
        near("selu(-1)", selu(-1.0), -1.111_3, 1e-3);
        near("relie(-2)", relie(-2.0), -0.02, 1e-7);
        near("relie(3)", relie(3.0), 3.0, 0.0);
        near("plse(-5)", plse(-5.0), -0.01, 1e-7);
        near("plse(0)", plse(0.0), 0.5, 1e-7);
        near("plse(5)", plse(5.0), 1.01, 1e-6);
        near("hardtan(-3)", hardtan(-3.0), -1.0, 0.0);
        near("hardtan(0.25)", hardtan(0.25), 0.25, 0.0);
        near("lhtan(-2)", lhtan(-2.0), -0.002, 1e-7);
        near("lhtan(0.5)", lhtan(0.5), 0.5, 0.0);
        near("lhtan(3)", lhtan(3.0), 1.002, 1e-6);
        // n = floor(x): even n -> floor(x/2); odd n -> (x-n) + floor(x/2)
        near("stair(0.5)", stair(0.5), 0.0, 1e-7);
        near("stair(1.5)", stair(1.5), 0.5, 1e-7);
        near("stair(2.5)", stair(2.5), 1.0, 1e-7);
        near("stair(3.5)", stair(3.5), 1.5, 1e-7);
        near("stair(-0.5)", stair(-0.5), -1.0 + 0.5, 1e-7); // n=-1 (odd): 0.5 + floor(-0.25) = -0.5
        near("hard_mish(-3)", hard_mish(-3.0), 0.0, 0.0);
        near("hard_mish(-1)", hard_mish(-1.0), -0.5, 1e-7);
        near("hard_mish(2)", hard_mish(2.0), 2.0, 0.0);
    }

    // Away from every kink/discontinuity of the ten functions (kinks at 0, +-1, -2, +-4, integers).
    const POINTS: [f32; 8] = [-5.3, -3.2, -1.7, -0.6, 0.4, 0.9, 2.3, 4.6];

    #[test]
    fn analytic_gradients_match_central_differences() {
        let h = 1e-2f32;
        let cases: [(&str, fn(f32) -> f32, fn(f32) -> f32); 8] = [
            ("gelu", gelu, gelu_grad),
            ("elu", elu, elu_grad),
            ("selu", selu, selu_grad),
            ("plse", plse, plse_grad),
            ("hardtan", hardtan, hardtan_grad),
            ("lhtan", lhtan, lhtan_grad),
            ("stair", stair, stair_grad),
            ("hard_mish", hard_mish, hard_mish_grad),
        ];
        for (name, f, g) in cases {
            for x in POINTS {
                let fd = (f(x + h) - f(x - h)) / (2.0 * h);
                assert!(
                    (fd - g(x)).abs() <= 2e-3 + 2e-3 * fd.abs(),
                    "{name} x={x}: finite difference {fd}, analytic {}",
                    g(x)
                );
            }
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tools/test_host.sh activation_math`
Expected: FAIL (compile error `cannot find function gelu in this scope`, or module not declared). Add `pub(crate) mod activation_math;` to `lib.rs` first so the failure is the missing functions.

- [ ] **Step 3: Implement the scalar math** — insert above the test module in `activation_math.rs`:

```rust
// C++ literals are f32 (`0.797885f`, `0.035677f`); the derivative below is the exact
// derivative of THIS forward, so it uses the same constants.
const GELU_A: f32 = 0.797_885;
const GELU_B: f32 = 0.035_677;
const SELU_SCALE: f32 = 1.0507;
const SELU_ALPHA: f32 = 1.6732;

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_A * x + GELU_B * x * x * x).tanh())
}

pub(crate) fn gelu_grad(x: f32) -> f32 {
    let t = (GELU_A * x + GELU_B * x * x * x).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (GELU_A + 3.0 * GELU_B * x * x)
}

pub(crate) fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

pub(crate) fn elu_grad(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { x.exp() }
}

pub(crate) fn selu(x: f32) -> f32 {
    if x >= 0.0 { SELU_SCALE * x } else { SELU_SCALE * SELU_ALPHA * (x.exp() - 1.0) }
}

pub(crate) fn selu_grad(x: f32) -> f32 {
    if x >= 0.0 { SELU_SCALE } else { SELU_SCALE * SELU_ALPHA * x.exp() }
}

pub(crate) fn relie(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 * x }
}

pub(crate) fn plse(x: f32) -> f32 {
    if x < -4.0 {
        0.01 * (x + 4.0)
    } else if x > 4.0 {
        0.01 * (x - 4.0) + 1.0
    } else {
        0.125 * x + 0.5
    }
}

pub(crate) fn plse_grad(x: f32) -> f32 {
    if x.abs() > 4.0 { 0.01 } else { 0.125 }
}

pub(crate) fn hardtan(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

pub(crate) fn hardtan_grad(x: f32) -> f32 {
    if x > -1.0 && x < 1.0 { 1.0 } else { 0.0 }
}

pub(crate) fn lhtan(x: f32) -> f32 {
    if x < 0.0 {
        0.001 * x
    } else if x > 1.0 {
        0.001 * (x - 1.0) + 1.0
    } else {
        x
    }
}

pub(crate) fn lhtan_grad(x: f32) -> f32 {
    if x > 0.0 && x < 1.0 { 1.0 } else { 0.001 }
}

pub(crate) fn stair(x: f32) -> f32 {
    let n = x.floor() as i32;
    if n % 2 == 0 {
        (x / 2.0).floor()
    } else {
        (x - n as f32) + (x / 2.0).floor()
    }
}

pub(crate) fn stair_grad(x: f32) -> f32 {
    if (x.floor() as i32).rem_euclid(2) == 1 { 1.0 } else { 0.0 }
}

pub(crate) fn hard_mish(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else if x > -2.0 {
        x * x / 2.0 + x
    } else {
        0.0
    }
}

pub(crate) fn hard_mish_grad(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x > -2.0 {
        x + 1.0
    } else {
        0.0
    }
}
```

- [ ] **Step 4: Run to verify the math passes**

Run: `./tools/test_host.sh activation_math`
Expected: `test activation_math::tests::forward_values_match_cpp_formulas ... ok`, `analytic_gradients_match_central_differences ... ok`, `test result: ok. 2 passed`.

- [ ] **Step 5: Write the failing dispatch test** — in `runtime.rs` `mod tests`, after `relu6_and_ramp_follow_cpp_activation_and_gradient`:

```rust
    #[test]
    fn extra_activations_parse_dispatch_and_use_the_exact_input_gradient() {
        let all = [
            ("gelu", Activation::Gelu),
            ("elu", Activation::Elu),
            ("selu", Activation::Selu),
            ("relie", Activation::Relie),
            ("plse", Activation::Plse),
            ("hardtan", Activation::Hardtan),
            ("lhtan", Activation::Lhtan),
            ("stair", Activation::Stair),
            ("hard_mish", Activation::HardMish),
            ("revleaky", Activation::Revleaky),
        ];
        for (name, expected) in &all {
            let network = Network::from_config(
                ConfigFile::parse_str(
                    "inline.cfg",
                    &format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={name}\n"),
                )
                .expect("cfg"),
            )
            .expect("network");
            let LayerKind::Convolutional(spec) = &network.layers[0].kind else {
                panic!("expected convolution");
            };
            assert_eq!(&spec.activation, expected, "{name}");
            // dispatch -> central difference of `activate` vs the runtime gradient
            for x in [-1.7f32, -0.6, 0.4, 0.9, 2.3] {
                let h = 1e-2;
                let f = |v: f32| super::activate(expected.clone(), v);
                let fd = (f(x + h) - f(x - h)) / (2.0 * h);
                let grad = super::activation_gradient_from_output(expected, f(x), Some(x), "t")
                    .expect("gradient");
                assert!((fd - grad).abs() <= 3e-3 + 3e-3 * fd.abs(), "{name} x={x}: fd {fd}, grad {grad}");
            }
        }
        // input-form gradients must refuse to run without the cached pre-activation
        assert!(super::activation_gradient_from_output(&Activation::Gelu, 0.5, None, "t").is_err());
    }
```

- [ ] **Step 6: Run to verify it fails**

Run: `./tools/test_host.sh extra_activations_parse_dispatch`
Expected: FAIL (compile error: no variant `Gelu` in `Activation`).

- [ ] **Step 7: Wire the variants** — `model.rs`: add to `enum Activation` after `Ramp,` the variants `Gelu, Elu, Selu, Relie, Plse, Hardtan, Lhtan, Stair, HardMish, Revleaky,`; in `Activation::parse` add

```rust
            "gelu" => Self::Gelu,
            "elu" => Self::Elu,
            "selu" => Self::Selu,
            "relie" => Self::Relie,
            "plse" => Self::Plse,
            "hardtan" => Self::Hardtan,
            "lhtan" => Self::Lhtan,
            "stair" => Self::Stair,
            "hard_mish" => Self::HardMish,
            "revleaky" => Self::Revleaky,
```

and in `to_cuda_f32_kind` make the ten new variants return `None` (`Self::Loggy => None,` line: change to `Self::Loggy | Self::Gelu | Self::Elu | Self::Selu | Self::Relie | Self::Plse | Self::Hardtan | Self::Lhtan | Self::Stair | Self::HardMish | Self::Revleaky => None,`). `runtime.rs` `activate`: add before `Activation::Other(_)`

```rust
        Activation::Gelu => crate::activation_math::gelu(value),
        Activation::Elu => crate::activation_math::elu(value),
        Activation::Selu => crate::activation_math::selu(value),
        Activation::Relie => crate::activation_math::relie(value),
        Activation::Plse => crate::activation_math::plse(value),
        Activation::Hardtan => crate::activation_math::hardtan(value),
        Activation::Lhtan => crate::activation_math::lhtan(value),
        Activation::Stair => crate::activation_math::stair(value),
        Activation::HardMish => crate::activation_math::hard_mish(value),
        Activation::Revleaky => {
            if value > 0.0 { value } else { 0.1 * value }
        }
```

`activation_gradient_from_output`: add a small helper above it and the arms before `Activation::Other(name)`:

```rust
fn cached_activation_input(
    activation: &Activation,
    activation_input: Option<f32>,
    layer_name: &str,
) -> Result<f32> {
    activation_input.ok_or_else(|| {
        Error::InvalidConfig(format!(
            "{layer_name} backward requires cached activation_input for {activation:?}"
        ))
    })
}
```

```rust
        Activation::Gelu => Ok(crate::activation_math::gelu_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Elu => Ok(crate::activation_math::elu_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Selu => Ok(crate::activation_math::selu_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Plse => Ok(crate::activation_math::plse_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Hardtan => Ok(crate::activation_math::hardtan_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Lhtan => Ok(crate::activation_math::lhtan_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::Stair => Ok(crate::activation_math::stair_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        Activation::HardMish => Ok(crate::activation_math::hard_mish_grad(cached_activation_input(activation, activation_input, layer_name)?)),
        // relie / revleaky are leaky-shaped: the output has the sign of the input.
        Activation::Relie => Ok(if output_value > 0.0 { 1.0 } else { 0.01 }),
        Activation::Revleaky => Ok(if output_value > 0.0 { 1.0 } else { 0.1 }),
```

- [ ] **Step 8: Run to verify everything passes**

Run: `./tools/test_host.sh activation`
Expected: all `activation*`, `relu6_and_ramp...`, `extra_activations_parse_dispatch...` tests `ok`; no other test regresses (`test result: ok`).

- [ ] **Step 9: Commit**

```bash
git add src/darknet-core/src/activation_math.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs
git commit -m "feat(activations): gelu elu selu relie plse hardtan lhtan stair hard_mish revleaky with exact derivatives

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Unknown activation names are a config error, not identity

C++ prints an error and falls back to relu; Rust `activate(Other(_))` silently returned the input unchanged, so a typo or an unsupported name trained and inferred wrongly without any message. This task initially made `normalize_channels*` and `eml` explicit errors. Current status: v66 permits `normalize_channels*` only on f32 `[convolutional]`; other layer types and f16 inference remain rejected.

**Files:**
- Modify: `src/darknet-core/src/model.rs` (new `LayerKind::activation()`; check in `Network::from_config` loop ~line 975-990)
- Test: `src/darknet-core/src/model.rs` `mod tests`

**Interfaces:**
- Produces: `impl LayerKind { pub fn activation(&self) -> Option<&Activation> }`.

- [ ] **Step 1: Write the failing test** — append to `model.rs` `mod tests` (add `Activation` to the `use super::{...}` list there if needed):

```rust
    fn build(text: &str) -> crate::error::Result<Network> {
        Network::from_config(ConfigFile::parse_str("t.cfg", text)?)
    }

    #[test]
    fn unknown_activation_is_a_config_error_not_identity() {
        let cfg = |act: &str| {
            format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={act}\n")
        };
        for ok in [
            "linear", "leaky", "mish", "swish", "gelu", "elu", "selu", "relie", "plse", "hardtan",
            "lhtan", "stair", "hard_mish", "revleaky",
        ] {
            assert!(build(&cfg(ok)).is_ok(), "{ok} must parse");
        }
        for bad in ["banana", "eml"] {
            let err = build(&cfg(bad)).expect_err(bad).to_string();
            assert!(err.contains(bad) && err.contains("line"), "{err}");
        }
        // the same check covers shortcut / connected / scale_channels / sam / rnn / crnn / local
        let shortcut = "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[shortcut]\nfrom=-1\nactivation=banana\n";
        assert!(build(shortcut).is_err());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tools/test_host.sh unknown_activation_is_a_config_error`
Expected: FAIL (`banana` currently builds fine: `thread ... panicked: banana`).

- [ ] **Step 3: Implement** — in `model.rs` add next to `impl LayerKind`:

```rust
impl LayerKind {
    /// The elementwise activation this layer applies, if it has one.
    pub fn activation(&self) -> Option<&Activation> {
        match self {
            Self::Convolutional(spec) => Some(&spec.activation),
            Self::Connected(spec) => Some(&spec.activation),
            Self::Shortcut(spec) => Some(&spec.activation),
            Self::ScaleChannels(spec) => Some(&spec.activation),
            Self::Sam(spec) => Some(&spec.activation),
            Self::Local(spec) => Some(&spec.activation),
            Self::Rnn(spec) => Some(&spec.activation),
            Self::Crnn(spec) => Some(&spec.activation),
            _ => None,
        }
    }
}
```

and in `Network::from_config`, right after `let kind = parse_layer_kind(...)?;`:

```rust
            if let Some(Activation::Other(name)) = kind.activation() {
                let hint = if name.starts_with("normalize_channels") {
                    " (channel-normalising activations need the layer geometry and are not implemented)"
                } else if name == "eml" {
                    " (eml is only used by the separate [eml_convolutional] layer, not scalar activations)"
                } else {
                    ""
                };
                return Err(Error::InvalidConfig(format!(
                    "[{}] at line {}: unsupported activation \"{name}\"{hint}; C++ would silently use relu",
                    section.name, section.line
                )));
            }
```

- [ ] **Step 4: Run to verify it passes and nothing else regressed**

Run: `./tools/test_host.sh unknown_activation_is_a_config_error` then `./tools/test_host.sh parses_`
Expected: `ok` and `test result: ok` (every shipped cfg uses only known names: linear leaky mish swish logistic relu6 relu ramp).

- [ ] **Step 5: Commit**

```bash
git add src/darknet-core/src/model.rs
git commit -m "fix(cfg): unknown activation is an error instead of a silent identity

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: GELU on the GPU (enum, oxide kernel, parity, finite-difference)

Contract D/E. Every shared device activation (`kernels::activations::activate`/`gradient`) feeds the fused bias+activation, f16, AMP and plain activation kernels, so one device-function edit covers them all.

**Files:**
- Modify: `src/darknet-cuda/src/lib.rs:14-26` (`F32ActivationKind::Gelu`)
- Modify: `src/darknet-core/src/model.rs` (`to_cuda_f32_kind`)
- Modify: `src/darknet-cuda-oxide/src/kernels/activations.rs`, `src/darknet-cuda-oxide/src/lib.rs:200-211`
- Modify: `src/darknet-core/src/runtime.rs` (`FakeBackend` helpers ~12098-12150; new test)
- Test: `src/darknet-cuda-oxide/tests/oxide_parity.rs` (+ the three other files with exhaustive `F32ActivationKind` matches)

**Interfaces:**
- Produces: `F32ActivationKind::Gelu`; `activations::ACT_GELU: u32 = 9`.
- Consumes: Task 1 `Activation::Gelu`.

- [ ] **Step 1: Write the failing tests.** (a) `src/darknet-core/src/runtime.rs` `mod tests`:

```rust
    #[test]
    fn gpu_gelu_convolution_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=gelu\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n",
            1,
        );
    }
```

(b) `src/darknet-cuda-oxide/tests/oxide_parity.rs`: change `const KINDS: [F32ActivationKind; 9]` to `; 10]` and append `F32ActivationKind::Gelu,`; add arms to the three reference functions:

```rust
        // cpu_activate (f32)
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
        // cpu_gradient (f32)
        F32ActivationKind::Gelu => {
            let t = (0.797885 * x + 0.035677 * x * x * x).tanh();
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (0.797885 + 3.0 * 0.035677 * x * x)
        }
        // act64 (f64, used by the finite-difference test)
        F32ActivationKind::Gelu => 0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh()),
```

(c) The other exhaustive matches fail to compile until they get a `Gelu` arm: `tests/oxide_cfg_shapes.rs` (`KINDS` array at ~1440, `kind_of` match + its `match k {}` list, `act64`, `grad64`, `act32`), `tests/oxide_half.rs` (`act_ref` ~323, the `acts` array ~590), `tests/oxide_batchnorm_adam.rs` (activation list ~550). Add the same f64/f32 formulas as above (`grad64`: `let t = (0.797885*x + 0.035677*x*x*x).tanh(); 0.5*(1.0+t) + 0.5*x*(1.0-t*t)*(0.797885 + 3.0*0.035677*x*x)`), `"gelu" => F32ActivationKind::Gelu` in `kind_of`, and `Gelu` in each array/`|` list.

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh gpu_gelu_convolution` — Expected: FAIL (`layer #0 uses activation Gelu, which the GPU f32 backend does not support`).
Run: `./tools/test_oxide.sh --test oxide_parity -- activation` — Expected: FAIL (compile error: no variant `Gelu` in `F32ActivationKind`).

- [ ] **Step 3: Implement.**
`src/darknet-cuda/src/lib.rs` — after `Ramp,`:
```rust
    /// tanh-approximation GELU `0.5x(1+tanh(0.797885x+0.035677x^3))`; exact derivative (C++ `gelu_*`).
    Gelu,
```
`model.rs` `to_cuda_f32_kind`: remove `Self::Gelu` from the `=> None` list and add `Self::Gelu => Some(F32ActivationKind::Gelu),`.
`src/darknet-cuda-oxide/src/kernels/activations.rs`: add `pub const ACT_GELU: u32 = 9;`; in `activate` add before the final `else`:
```rust
    } else if kind == 9 {
        0.5 * x * (1.0 + (0.797885 * x + 0.035677 * x * x * x).tanh())
```
and in `gradient` before the final `else`:
```rust
    } else if kind == 9 {
        let t = (0.797885 * x + 0.035677 * x * x * x).tanh();
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (0.797885 + 3.0 * 0.035677 * x * x)
```
`src/darknet-cuda-oxide/src/lib.rs` `activation_code`: `F32ActivationKind::Gelu => activations::ACT_GELU,`.
`runtime.rs` FakeBackend: in `apply_fake_activation` add `F32ActivationKind::Gelu => Activation::Gelu,`; in `fake_activation_gradient` add `F32ActivationKind::Gelu => crate::activation_math::gelu_grad(value),`.

- [ ] **Step 4: Run to verify they pass**

Run: `./tools/test_host.sh gpu_gelu_convolution` — Expected: `ok`.
Run: `./tools/test_oxide.sh --test oxide_parity -- activation` — Expected: `activations_forward_and_backward ... ok`, `activation_backward_matches_finite_difference ... ok`.
Run: `./tools/test_oxide.sh --test oxide_cfg_shapes -- activation` and `--test oxide_half -- shortcut` and `--test oxide_batchnorm_adam -- bias` (the files touched) — Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/darknet-cuda/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs src/darknet-cuda-oxide/src src/darknet-cuda-oxide/tests
git commit -m "feat(oxide): GELU activation kernel (forward + exact derivative)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `maxpool_zero_nonmax` key and `maxpool_depth`

Fixes the key spelling (C++ `maxpool_zero_nonmax`) and implements depth pooling: output channel `g` is the max over input channels `g, g+out_c, g+2*out_c, ...` at the same pixel; spatial size unchanged; C++ ignores `size/stride/padding` in this mode. Depth pooling remains CPU-only; ordinary maxpool with `zero_nonmax` uses the existing GPU argmax-backward operation, which writes zero to every non-argmax input.

**Files:**
- Modify: `src/darknet-core/src/model.rs` (`MaxPoolSpec` ~283, parse ~1139-1170, `infer_output_shape` ~1408-1424)
- Modify: `src/darknet-core/src/runtime.rs` (CPU forward ~1017, backward ~1344, compact preflight ~829, three GPU channel checks ~2501/3788/4510)
- Test: `model.rs` `mod tests`, `runtime.rs` `mod tests`

**Interfaces:**
- Produces: `MaxPoolSpec.depth: bool`; `runtime.rs`: `fn forward_maxpool_depth(input: &Tensor, output_shape: TensorShape) -> Result<Tensor>`, `fn backward_maxpool_depth(input: &Tensor, output_gradient: &Tensor) -> Result<Tensor>`.

- [ ] **Step 1: Write the failing tests.** `model.rs` tests:

```rust
    #[test]
    fn maxpool_depth_and_zero_nonmax_keys_follow_cpp() {
        let net = build(
            "[net]\nwidth=4\nheight=6\nchannels=4\n\
             [maxpool]\nsize=2\nstride=2\nmaxpool_depth=1\nout_channels=2\n\
             [maxpool]\nsize=2\nstride=2\nmaxpool_zero_nonmax=1\n",
        )
        .expect("builds");
        let LayerKind::MaxPool(depth) = &net.layers[0].kind else { panic!() };
        assert!(depth.depth);
        // depth pooling keeps w x h and sets the channels to out_channels
        let s = net.layers[0].output_shape;
        assert_eq!((s.width, s.height, s.channels), (4, 6, 2));
        let LayerKind::MaxPool(plain) = &net.layers[1].kind else { panic!() };
        assert!(plain.zero_nonmax && !plain.depth);
        let s = net.layers[1].output_shape;
        assert_eq!((s.width, s.height, s.channels), (2, 3, 2)); // out_channels ignored without depth
        // out_channels larger than the input channels is rejected
        assert!(build("[net]\nwidth=4\nheight=4\nchannels=2\n[maxpool]\nmaxpool_depth=1\nout_channels=3\n").is_err());
    }
```

`runtime.rs` tests:

```rust
    #[test]
    fn maxpool_depth_picks_the_first_max_over_strided_channels_and_routes_the_gradient() {
        // w=2 h=1 c=4: channel k = [a_k, b_k]; out_channels=2 -> group0 = {ch0, ch2}, group1 = {ch1, ch3}
        let input = Tensor::from_data(
            TensorShape::new(2, 1, 4, 1),
            vec![1.0, 5.0, 2.0, 0.0, 3.0, 4.0, 0.0, 9.0],
        )
        .unwrap();
        let out_shape = TensorShape::new(2, 1, 2, 1);
        let out = super::forward_maxpool_depth(&input, out_shape).unwrap();
        assert_eq!(out.data, vec![3.0, 5.0, 2.0, 9.0]);
        let grad = Tensor::from_data(out_shape, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let dx = super::backward_maxpool_depth(&input, &grad).unwrap();
        // winners: g0x0 -> ch2, g0x1 -> ch0, g1x0 -> ch1, g1x1 -> ch3
        assert_eq!(dx.data, vec![0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0]);
        // tie: the first channel wins (C++ `val > max`)
        let tie = Tensor::from_data(TensorShape::new(1, 1, 4, 1), vec![7.0, 1.0, 7.0, 1.0]).unwrap();
        let g = Tensor::from_data(TensorShape::new(1, 1, 2, 1), vec![1.0, 1.0]).unwrap();
        assert_eq!(super::backward_maxpool_depth(&tie, &g).unwrap().data, vec![1.0, 1.0, 0.0, 0.0]);
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh maxpool_depth`
Expected: FAIL (compile error: no field `depth`, no function `forward_maxpool_depth`).

- [ ] **Step 3: Implement the model.** `MaxPoolSpec`: add after `zero_nonmax`:
```rust
    /// C++ `maxpool_depth=1`: pool over strided channel groups; w x h unchanged, channels = `out_channels`.
    pub depth: bool,
```
Parse (`"max" | "maxpool"` arm): replace the `out_channels` and `zero_nonmax` lines and add `depth`:
```rust
                // C++ default is 1 and the value only matters with maxpool_depth=1.
                out_channels: section.get_usize("out_channels", 1)?,
                // C++ key; the pre-port Rust key `zero_nonmax` is still accepted.
                zero_nonmax: section.get_bool_int(
                    "maxpool_zero_nonmax",
                    section.get_bool_int("zero_nonmax", false)?,
                )?,
                depth: section.get_bool_int("maxpool_depth", false)?,
```
`LocalAvgPool` literal: `out_channels: current.channels, zero_nonmax: false, depth: false,`. `infer_output_shape` pool arm becomes:
```rust
        LayerKind::MaxPool(spec) | LayerKind::LocalAvgPool(spec) => {
            if spec.depth {
                if spec.out_channels == 0 || spec.out_channels > current.channels {
                    return Err(Error::InvalidConfig(format!(
                        "maxpool_depth layer #{layer_index}: out_channels={} must be in 1..={}",
                        spec.out_channels, current.channels
                    )));
                }
                return Ok(TensorShape::new(
                    current.width,
                    current.height,
                    spec.out_channels,
                    current.batch,
                ));
            }
            Ok(TensorShape::new(
                pool_dim(current.width, spec.padding, spec.size, spec.stride_x),
                pool_dim(current.height, spec.padding, spec.size, spec.stride_y),
                current.channels,
                current.batch,
            ))
        }
```

- [ ] **Step 4: Implement the runtime.** Add near `forward_maxpool`:

```rust
/// First maximum over the input channels `g, g + out_c, ...` at one pixel
/// (C++ `val > max` keeps the first on ties). Returns `(value, input_channel)`.
fn depth_argmax(input: &Tensor, b: usize, g: usize, out_c: usize, y: usize, x: usize) -> (f32, usize) {
    let mut best_channel = g;
    let mut best = input.get(b, g, y, x);
    let mut k = g + out_c;
    while k < input.shape.channels {
        let value = input.get(b, k, y, x);
        if value > best {
            best = value;
            best_channel = k;
        }
        k += out_c;
    }
    (best, best_channel)
}

fn forward_maxpool_depth(input: &Tensor, output_shape: TensorShape) -> Result<Tensor> {
    if output_shape.width != input.shape.width
        || output_shape.height != input.shape.height
        || output_shape.batch != input.shape.batch
        || output_shape.channels == 0
        || output_shape.channels > input.shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "maxpool_depth shape mismatch: input {}, output {output_shape}",
            input.shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for g in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let (value, _) = depth_argmax(input, b, g, output_shape.channels, y, x);
                    output.set(b, g, y, x, value);
                }
            }
        }
    }
    Ok(output)
}

fn backward_maxpool_depth(input: &Tensor, output_gradient: &Tensor) -> Result<Tensor> {
    let mut grad = Tensor::zeros(input.shape);
    let out_c = output_gradient.shape.channels;
    for b in 0..output_gradient.shape.batch {
        for g in 0..out_c {
            for y in 0..output_gradient.shape.height {
                for x in 0..output_gradient.shape.width {
                    let (_, k) = depth_argmax(input, b, g, out_c, y, x);
                    let current = grad.get(b, k, y, x);
                    grad.set(b, k, y, x, current + output_gradient.get(b, g, y, x));
                }
            }
        }
    }
    Ok(grad)
}
```

Dispatch: in `forward_with_mode` put BEFORE the `(LayerKind::MaxPool(spec), LayerParams::None)` arm (~1017):
```rust
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    forward_maxpool_depth(current_input, layer.output_shape).map(|output| (output, None))?
                }
```
and before the backward `MaxPool` arm (~1344):
```rust
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    let input_grad = backward_maxpool_depth(current_input, &grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
```
Compact CPU preflight (~829): replace `| LayerKind::MaxPool(_)` by `| LayerKind::MaxPool(crate::model::MaxPoolSpec { depth: false, .. })` (otherwise a depth layer with `out_channels == channels` would be silently pooled as a normal maxpool). In the three GPU `MaxPool` arms (~2501, ~3788, ~4510) replace `if layer.output_shape.channels != layer.input_shape.channels {` by `if spec.depth {` (keep the `UnsupportedLayer { name: "maxpool_depth" }` body).

- [ ] **Step 5: Run to verify**

Run: `./tools/test_host.sh maxpool` — Expected: `maxpool_depth_and_zero_nonmax_keys_follow_cpp ... ok`, `maxpool_depth_picks_the_first_max...  ok`, existing `maxpool_zero_nonmax_zeroes_non_argmax_inputs ... ok`, `test result: ok`.

- [ ] **Step 6: Commit**

```bash
git add src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs
git commit -m "fix(maxpool): C++ key maxpool_zero_nonmax and real maxpool_depth (CPU)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Standalone `[batchnorm]` can be saved (Rust-only extension, documented)

`load_weights_into_params` reads `biases, scales, rolling_mean, rolling_variance`; `save_weights` has no arm, so saving any net with `[batchnorm]` fails. C++ never persists a standalone batchnorm layer (no `BATCHNORM` in `ELayerType`, none in `save_weights_upto`), so this is a Rust extension: a `.weights` file with a `[batchnorm]` layer is NOT readable by C++ and vice versa. Keep it simple: add the missing symmetric arm and state the divergence in the doc comment.

**Files:**
- Modify: `src/darknet-core/src/weights.rs` (`save_weights` match ~line 98-160; module doc)
- Test: `weights.rs` `mod tests`

- [ ] **Step 1: Write the failing test** (append to `weights.rs` `mod tests`):

```rust
    #[test]
    fn standalone_batchnorm_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str("bn.cfg", "[net]\nwidth=2\nheight=2\nchannels=3\n[batchnorm]\n")
                .expect("cfg"),
        )
        .expect("network");
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::BatchNorm(bn) = &mut executable.params[0] else {
            panic!("expected batchnorm params");
        };
        bn.biases = vec![0.1, 0.2, 0.3];
        bn.scales = vec![1.0, 2.0, 3.0];
        bn.rolling_mean = vec![4.0, 5.0, 6.0];
        bn.rolling_variance = vec![7.0, 8.0, 9.0];

        let path = std::env::temp_dir().join("darknet-batchnorm-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 3).expect("save must not fail");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * 12);
        let mut reloaded = ExecutableNetwork::new(network);
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        let _ = fs::remove_file(path);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `./tools/test_host.sh standalone_batchnorm_weights`
Expected: FAIL with `layer #0 (batchnorm) and parameter set are inconsistent while saving weights`.

- [ ] **Step 3: Implement** — in `save_weights`, before `(LayerKind::Local(_), LayerParams::Local(local)) =>`:

```rust
            // Rust-only: C++ never writes a standalone [batchnorm]; order mirrors
            // `load_weights_into_params` and the C++ `save_batchnorm_weights` helper.
            (LayerKind::BatchNorm(_), LayerParams::BatchNorm(bn)) => {
                write_f32_slice(&mut file, &bn.biases)?;
                write_f32_slice(&mut file, &bn.scales)?;
                write_f32_slice(&mut file, &bn.rolling_mean)?;
                write_f32_slice(&mut file, &bn.rolling_variance)?;
            }
```

- [ ] **Step 4: Run to verify it passes**

Run: `./tools/test_host.sh weights` — Expected: all `weights::tests::*` `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/darknet-core/src/weights.rs
git commit -m "fix(weights): save standalone batchnorm params (Rust-only extension)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `[shortcut]` `weights_type` / `weights_normalization` (CPU, `.weights`, GPU runtime)

C++: `weights_type = none | per_feature | per_channel`, `weights_normalization = none | relu | avg_relu | softmax`. `nweights = (n+1)` (per_feature) or `(n+1)*c` (per_channel), `n = from.len()`, `c` = layer channels; saved right after the layer's turn in layer order (`save_shortcut_weights`, only when `nweights > 0`); initialised to `1.0`. Weight index of input `i` (`i=0` is the previous layer, `1..=n` the `from` layers) at flat element `e`: `e/step + i*layer_step` with `layer_step = nweights/(n+1)`, `step = per_sample/layer_step`. `eps = 1e-4`. Forward: `out[e] = w_0*x_0[e] + sum_i [e < len_i] w_i*x_i[e]`, `w_i` = raw, `relu(w_i)/(eps+sum_j relu(w_j))`, or `exp(w_i-max)/(eps+sum_j exp(w_j-max))`, per slot. The C++ weight gradient is `sum delta*x_i` (ignores the normalisation); the exact gradient is derived in the code comment below. GPU forward composes `channel_scale_f32` + `shortcut_add_f32` in the backend default weighted-sum operation; exact backward and SGD update use the CPU fallback because shortcut parameters are host-resident. FakeBackend and real-device runtime tests cover mismatched fit, forward, backward, and SGD. A dedicated fused oxide kernel/device weight-gradient reduction is only an optional performance follow-up.

**Files:**
- Create: `src/darknet-core/src/shortcut_weights.rs`
- Modify: `src/darknet-core/src/lib.rs`, `model.rs` (specs, parse, `Layer::weights()`), `runtime.rs` (params/buffers/arms), `weights.rs`, `initialize.rs`, `src/darknet-cli/src/main.rs`
- Test: `shortcut_weights.rs` `mod tests`; `weights.rs`, `model.rs`, `runtime.rs` tests

**Interfaces:**
- Produces (`model.rs`): `enum ShortcutWeightsType { None, PerFeature, PerChannel }`, `enum ShortcutWeightsNorm { None, Relu, Softmax }` (both `Copy, Default, PartialEq, Eq, Debug, Clone`), `ShortcutSpec { from, activation, weights_type, weights_normalization }`, `ShortcutSpec::weight_count(&self, channels: usize) -> usize`.
- Produces (`shortcut_weights.rs`, `pub(crate)`): `effective_weights(raw: &[f32], norm: ShortcutWeightsNorm, layer_step: usize) -> Vec<f32>`; `weighted_forward(input: &[f32], sources: &[&[f32]], batch: usize, per_sample: usize, raw: &[f32], norm: ShortcutWeightsNorm, layer_step: usize) -> Vec<f32>`; `struct WeightedGrad { pub input: Vec<f32>, pub sources: Vec<Vec<f32>>, pub weights: Vec<f32> }`; `weighted_backward(delta: &[f32], input: &[f32], sources: &[&[f32]], batch: usize, per_sample: usize, raw: &[f32], norm: ShortcutWeightsNorm, layer_step: usize, cpp_grad: bool) -> WeightedGrad`.
- Produces (`runtime.rs`): `pub struct ShortcutParams { pub weights: Vec<f32> }`, `pub struct ShortcutTrainBuffers { pub weight_updates: Vec<f32> }`, `LayerParams::Shortcut(ShortcutParams)`, `LayerTrainBuffers::Shortcut(ShortcutTrainBuffers)`.

- [ ] **Step 1: Write the failing tests.** Create `shortcut_weights.rs` with only the test module:

```rust
//! Weighted `[shortcut]` (tasks/rust/layers.md Task 6).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ShortcutWeightsNorm::{None as NoNorm, Relu, Softmax};

    #[test]
    fn per_feature_relu_matches_hand_computation() {
        // raw [2, 1], eps 1e-4: eff = [2/3.0001, 1/3.0001]; x = 3, y = 6 -> 12/3.0001
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[2.0, 1.0], Relu, 1);
        assert!((out[0] - 12.0 / 3.0001).abs() < 1e-5, "{out:?}");
        // softmax of equal weights is 1/(2+eps) each: out = (3 + 6) / 2.0001
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[0.0, 0.0], Softmax, 1);
        assert!((out[0] - 9.0 / 2.0001).abs() < 1e-5, "{out:?}");
        // no normalisation: plain weighted sum
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[2.0, 0.5], NoNorm, 1);
        assert_eq!(out, vec![9.0]);
    }

    #[test]
    fn mismatched_source_adds_only_its_prefix() {
        // per_sample 4, source has 2 elements: elements 2,3 only see the layer input
        let out = weighted_forward(&[1.0, 1.0, 1.0, 1.0], &[&[10.0, 10.0]], 1, 4, &[1.0, 1.0], NoNorm, 1);
        assert_eq!(out, vec![11.0, 11.0, 1.0, 1.0]);
    }

    fn loss(input: &[f32], sources: &[&[f32]], raw: &[f32], norm: ShortcutWeightsNorm, seed: &[f32]) -> f32 {
        // batch 2, per_sample 6, per_channel with 2 channels (layer_step 2, step 3)
        weighted_forward(input, sources, 2, 6, raw, norm, 2)
            .iter()
            .zip(seed)
            .map(|(o, s)| o * s)
            .sum()
    }

    #[test]
    fn exact_gradients_match_central_differences_for_every_normalisation() {
        let input: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
        let s0: Vec<f32> = (0..12).map(|i| (i as f32 * 0.71).cos()).collect(); // len 6/sample
        let s1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.53).sin() + 0.3).collect(); // len 4/sample (prefix)
        let seed: Vec<f32> = (0..12).map(|i| (i as f32 * 0.91).cos()).collect();
        let raw = vec![0.5f32, -0.3, 0.8, 1.2, 0.4, 0.9]; // (n+1)=3 inputs x 2 slots
        let h = 1e-2f32;
        for norm in [NoNorm, Relu, Softmax] {
            let g = weighted_backward(&seed, &input, &[&s0, &s1], 2, 6, &raw, norm, 2, false);
            let check = |name: &str, analytic: f32, plus: f32, minus: f32| {
                let fd = (plus - minus) / (2.0 * h);
                assert!((fd - analytic).abs() <= 5e-3 + 5e-3 * fd.abs(), "{norm:?} {name}: fd {fd}, analytic {analytic}");
            };
            for k in 0..raw.len() {
                let (mut p, mut m) = (raw.clone(), raw.clone());
                p[k] += h;
                m[k] -= h;
                check(&format!("w[{k}]"), g.weights[k], loss(&input, &[&s0, &s1], &p, norm, &seed), loss(&input, &[&s0, &s1], &m, norm, &seed));
            }
            for k in 0..input.len() {
                let (mut p, mut m) = (input.clone(), input.clone());
                p[k] += h;
                m[k] -= h;
                check(&format!("x[{k}]"), g.input[k], loss(&p, &[&s0, &s1], &raw, norm, &seed), loss(&m, &[&s0, &s1], &raw, norm, &seed));
            }
            for k in 0..s1.len() {
                let (mut p, mut m) = (s1.clone(), s1.clone());
                p[k] += h;
                m[k] -= h;
                check(&format!("s1[{k}]"), g.sources[1][k], loss(&input, &[&s0, &p], &raw, norm, &seed), loss(&input, &[&s0, &m], &raw, norm, &seed));
            }
        }
    }

    #[test]
    fn cpp_grad_mode_ignores_the_normalisation_derivative() {
        let raw = [2.0f32, 1.0];
        let g = weighted_backward(&[1.0], &[3.0], &[&[6.0]], 1, 1, &raw, Relu, 1, true);
        assert_eq!(g.weights, vec![3.0, 6.0]); // sum(delta * x_i) exactly, like blas.cpp (grad = 1)
    }
}
```
`weights.rs` test (append):

```rust
    #[test]
    fn shortcut_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "sc.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=softmax\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
        // (n+1)*c = 2*2 = 4 shortcut weights after 2 convs of (2 bias + 4 weights)
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Shortcut(sc) = &mut executable.params[2] else { panic!("shortcut params") };
        sc.weights = vec![0.5, 1.5, 2.5, 3.5];
        let path = std::env::temp_dir().join("darknet-shortcut-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * (2 * (2 + 4) + 4));
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        let inspection = inspect_weights_file(&network, &path).expect("inspect");
        assert_eq!(inspection.trailing_bytes, 0);
        let _ = fs::remove_file(path);
    }
```

`model.rs` test:

```rust
    #[test]
    fn shortcut_weight_keys_follow_cpp_names_and_counts() {
        let mk = |extra: &str| {
            build(&format!(
                "[net]\nwidth=2\nheight=2\nchannels=3\n[convolutional]\nfilters=3\nsize=1\n[convolutional]\nfilters=3\nsize=1\n[shortcut]\nfrom=-2\n{extra}\n"
            ))
        };
        let n = mk("weights_type=per_feature\nweights_normalization=avg_relu").unwrap();
        assert_eq!(n.layers[2].weights().unwrap().parameter_count(), 2); // n+1
        let n = mk("weights_type=per_channel").unwrap();
        assert_eq!(n.layers[2].weights().unwrap().parameter_count(), 6); // (n+1)*c
        assert!(mk("").unwrap().layers[2].weights().is_none());
        assert!(mk("weights_type=banana").is_err());
        assert!(mk("weights_normalization=banana").is_err());
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh shortcut_weight` — Expected: FAIL (compile errors: `weighted_forward` / `LayerParams::Shortcut` / `ShortcutWeightsNorm` undefined).

- [ ] **Step 3: Implement the pure math** — in `shortcut_weights.rs` above the tests (declare `pub(crate) mod shortcut_weights;` in `lib.rs`):

```rust
use crate::model::ShortcutWeightsNorm;

/// C++ `eps` in the weight normalisation.
const EPS: f32 = 0.0001;

/// Per-slot normalised weights. `raw` is laid out `[input i][slot]` with
/// `layer_step` slots per input (1 for per_feature, C for per_channel).
pub(crate) fn effective_weights(raw: &[f32], norm: ShortcutWeightsNorm, layer_step: usize) -> Vec<f32> {
    let n1 = raw.len() / layer_step;
    let mut eff = raw.to_vec();
    for slot in 0..layer_step {
        let at = |i: usize| slot + i * layer_step;
        match norm {
            ShortcutWeightsNorm::None => {}
            ShortcutWeightsNorm::Relu => {
                let sum = EPS + (0..n1).map(|i| raw[at(i)].max(0.0)).sum::<f32>();
                for i in 0..n1 {
                    eff[at(i)] = raw[at(i)].max(0.0) / sum;
                }
            }
            ShortcutWeightsNorm::Softmax => {
                let max = (0..n1).map(|i| raw[at(i)]).fold(f32::NEG_INFINITY, f32::max);
                let sum = EPS + (0..n1).map(|i| (raw[at(i)] - max).exp()).sum::<f32>();
                for i in 0..n1 {
                    eff[at(i)] = (raw[at(i)] - max).exp() / sum;
                }
            }
        }
    }
    eff
}

/// `out[b,e] = eff[e/step]*input[b,e] + sum_i [e < len_i] eff[e/step + (i+1)*layer_step]*sources[i][b*len_i+e]`
/// (`len_i` = per-sample length of source `i`; a shorter source only covers the flat prefix,
/// like the C++ `src_i < add_outputs` test).
pub(crate) fn weighted_forward(
    input: &[f32],
    sources: &[&[f32]],
    batch: usize,
    per_sample: usize,
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
) -> Vec<f32> {
    let eff = effective_weights(raw, norm, layer_step);
    let step = per_sample / layer_step;
    let mut out = vec![0.0f32; batch * per_sample];
    for b in 0..batch {
        for e in 0..per_sample {
            let slot = e / step;
            let mut value = eff[slot] * input[b * per_sample + e];
            for (i, source) in sources.iter().enumerate() {
                let len = source.len() / batch;
                if e < len {
                    value += eff[slot + (i + 1) * layer_step] * source[b * len + e];
                }
            }
            out[b * per_sample + e] = value;
        }
    }
    out
}

pub(crate) struct WeightedGrad {
    pub input: Vec<f32>,
    pub sources: Vec<Vec<f32>>,
    pub weights: Vec<f32>,
}

/// Exact backward. `z[k]` = dL/d(effective weight k) = sum over the slot's elements of
/// `delta * x_i`. Chain through the normalisation:
/// - none:    dL/dw_k = z_k
/// - relu:    eff_i = r_i/S, r = relu(w), S = eps + sum r  ->  dL/dw_k = [w_k>0] (z_k/S - sum_i z_i r_i / S^2)
/// - softmax: eff_i = e_i/S                                 ->  dL/dw_k = eff_k (z_k - sum_i z_i eff_i)
/// `cpp_grad` reproduces C++ (`grad = 1`, i.e. dL/dw_k = z_k for every normalisation).
pub(crate) fn weighted_backward(
    delta: &[f32],
    input: &[f32],
    sources: &[&[f32]],
    batch: usize,
    per_sample: usize,
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
    cpp_grad: bool,
) -> WeightedGrad {
    let n1 = raw.len() / layer_step;
    let eff = effective_weights(raw, norm, layer_step);
    let step = per_sample / layer_step;
    let mut grad = WeightedGrad {
        input: vec![0.0; delta.len()],
        sources: sources.iter().map(|s| vec![0.0; s.len()]).collect(),
        weights: vec![0.0; raw.len()],
    };
    let mut z = vec![0.0f32; raw.len()];
    for b in 0..batch {
        for e in 0..per_sample {
            let slot = e / step;
            let d = delta[b * per_sample + e];
            grad.input[b * per_sample + e] = d * eff[slot];
            z[slot] += d * input[b * per_sample + e];
            for (i, source) in sources.iter().enumerate() {
                let len = source.len() / batch;
                if e < len {
                    let idx = slot + (i + 1) * layer_step;
                    grad.sources[i][b * len + e] = d * eff[idx];
                    z[idx] += d * source[b * len + e];
                }
            }
        }
    }
    for slot in 0..layer_step {
        let at = |i: usize| slot + i * layer_step;
        match (norm, cpp_grad) {
            (ShortcutWeightsNorm::None, _) | (_, true) => {
                for i in 0..n1 {
                    grad.weights[at(i)] = z[at(i)];
                }
            }
            (ShortcutWeightsNorm::Relu, false) => {
                let sum = EPS + (0..n1).map(|i| raw[at(i)].max(0.0)).sum::<f32>();
                let dot: f32 = (0..n1).map(|i| z[at(i)] * raw[at(i)].max(0.0)).sum();
                for k in 0..n1 {
                    grad.weights[at(k)] = if raw[at(k)] > 0.0 { z[at(k)] / sum - dot / (sum * sum) } else { 0.0 };
                }
            }
            (ShortcutWeightsNorm::Softmax, false) => {
                let dot: f32 = (0..n1).map(|i| z[at(i)] * eff[at(i)]).sum();
                for k in 0..n1 {
                    grad.weights[at(k)] = eff[at(k)] * (z[at(k)] - dot);
                }
            }
        }
    }
    grad
}
```

- [ ] **Step 4: Run the math tests** (the model enums must exist first; do Step 5 before this if the compile order requires) — Run: `./tools/test_host.sh shortcut_weights::tests` — Expected: `per_feature_relu_matches_hand_computation ... ok`, `mismatched_source_adds_only_its_prefix ... ok`, `exact_gradients_match_central_differences_for_every_normalisation ... ok`, `cpp_grad_mode_ignores_the_normalisation_derivative ... ok`.

- [ ] **Step 5: Implement model + params + weights I/O.**
`model.rs` — before `ShortcutSpec` add:

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShortcutWeightsType {
    #[default]
    None,
    PerFeature,
    PerChannel,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShortcutWeightsNorm {
    #[default]
    None,
    Relu,
    Softmax,
}

impl ShortcutWeightsType {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "per_feature" => Ok(Self::PerFeature),
            "per_channel" => Ok(Self::PerChannel),
            other => Err(Error::InvalidConfig(format!("weights type \"{other}\" is not supported"))),
        }
    }
}

impl ShortcutWeightsNorm {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "relu" | "avg_relu" => Ok(Self::Relu),
            "softmax" => Ok(Self::Softmax),
            other => Err(Error::InvalidConfig(format!("weights normalization \"{other}\" is not supported"))),
        }
    }
}
```
`ShortcutSpec` gains `pub weights_type: ShortcutWeightsType, pub weights_normalization: ShortcutWeightsNorm,` and

```rust
impl ShortcutSpec {
    /// C++ `nweights`: `n+1` (per_feature) or `(n+1)*c` (per_channel), 0 without weights.
    pub fn weight_count(&self, channels: usize) -> usize {
        match self.weights_type {
            ShortcutWeightsType::None => 0,
            ShortcutWeightsType::PerFeature => self.from.len() + 1,
            ShortcutWeightsType::PerChannel => (self.from.len() + 1) * channels,
        }
    }

    /// Weight slots per input (C++ `layer_step`).
    pub fn weight_slots(&self, channels: usize) -> usize {
        match self.weights_type {
            ShortcutWeightsType::None => 0,
            ShortcutWeightsType::PerFeature => 1,
            ShortcutWeightsType::PerChannel => channels,
        }
    }
}
```
parse arm: add `weights_type: ShortcutWeightsType::parse(&section.get_str("weights_type", "none"))?, weights_normalization: ShortcutWeightsNorm::parse(&section.get_str("weights_normalization", "none"))?,`. `Layer::weights()` new arm:

```rust
            LayerKind::Shortcut(spec) => {
                let len = spec.weight_count(self.input_shape.channels);
                (len > 0).then(|| LayerWeights {
                    entries: vec![WeightsEntry { array: WeightsArray::Weights, len }],
                })
            }
```
Re-export in `lib.rs`: add `ShortcutWeightsNorm, ShortcutWeightsType` to the `pub use model::{...}` list and `ShortcutParams, ShortcutTrainBuffers` to `pub use runtime::{...}`.

`runtime.rs`: add structs `ShortcutParams` / `ShortcutTrainBuffers` (derive `Clone, Debug, PartialEq`) near `LocalParams`; enum variants `LayerParams::Shortcut(ShortcutParams)` and `LayerTrainBuffers::Shortcut(ShortcutTrainBuffers)`; defaults:

```rust
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerParams::None,
            len => LayerParams::Shortcut(ShortcutParams { weights: vec![0.0; len] }),
        },
```
in `default_layer_params`, and in `default_layer_train_buffers`:
```rust
        LayerKind::Shortcut(spec) => match spec.weight_count(layer.input_shape.channels) {
            0 => LayerTrainBuffers::None,
            len => LayerTrainBuffers::Shortcut(ShortcutTrainBuffers { weight_updates: vec![0.0; len] }),
        },
```
`update_weights` arm (~after the `Local` arm): `(LayerParams::Shortcut(params), LayerTrainBuffers::Shortcut(train)) => { sgd_update(&mut params.weights, &mut train.weight_updates, lr, momentum, decay, batch); }`. `accumulate_train_buffers` arm:

```rust
                (LayerKind::Shortcut(_), LayerParams::Shortcut(gradient)) => {
                    let Some(LayerTrainBuffers::Shortcut(train)) = self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.weight_updates, &gradient.weights)?;
                }
```
`load_weights_into_params`: `LayerParams::Shortcut(shortcut) => read_f32_into(&mut file, &mut shortcut.weights)?,`. `weights.rs` `save_weights`: `(LayerKind::Shortcut(_), LayerParams::Shortcut(shortcut)) => write_f32_slice(&mut file, &shortcut.weights)?,` and add `| LayerParams::Shortcut(_)` to the exhaustive list in `recurrent_weight_blocks_round_trip_in_darknet_order`. `initialize.rs`: `LayerParams::Shortcut(p) => p.weights.fill(1.0),` (C++ init). `darknet-cli/src/main.rs` (~2361): `darknet_core::LayerTrainBuffers::Shortcut(shortcut) => shortcut.weight_updates.iter().any(|value| *value != 0.0),`.

- [ ] **Step 6: Implement the CPU arms.** Helper glue in `runtime.rs` near `forward_shortcut_with_cache`:

```rust
fn shortcut_sources<'a>(
    spec: &crate::model::ShortcutSpec,
    layer_index: usize,
    outputs: &'a [Tensor],
    batch: usize,
) -> Result<Vec<&'a Tensor>> {
    spec.from
        .iter()
        .map(|raw_index| {
            let source_index = resolve_relative_layer(layer_index, *raw_index)?;
            let source = outputs.get(source_index).ok_or(Error::InvalidRoute {
                layer_index,
                reference: *raw_index,
            })?;
            if source.shape.batch != batch {
                return Err(Error::InvalidConfig(format!(
                    "shortcut layer #{layer_index} mixes incompatible batch shapes"
                )));
            }
            Ok(source)
        })
        .collect()
}

fn forward_shortcut_weighted(
    spec: &crate::model::ShortcutSpec,
    params: &ShortcutParams,
    input: &Tensor,
    layer_index: usize,
    outputs: &[Tensor],
) -> Result<(Tensor, Option<Tensor>)> {
    let sources = shortcut_sources(spec, layer_index, outputs, input.shape.batch)?;
    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
    let pre = crate::shortcut_weights::weighted_forward(
        &input.data,
        &slices,
        input.shape.batch,
        input.shape.element_count(),
        &params.weights,
        spec.weights_normalization,
        spec.weight_slots(input.shape.channels),
    );
    let output: Vec<f32> = pre.iter().map(|v| activate(spec.activation.clone(), *v)).collect();
    Ok((Tensor::from_data(input.shape, output)?, Some(Tensor::from_data(input.shape, pre)?)))
}
```
Forward arm in `forward_with_mode` (next to the existing `Shortcut` arm): `(LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => forward_shortcut_weighted(spec, params, current_input, layer.index, &layer_outputs)?,`. Backward arm in `backward_seeded_core` (next to the existing `Shortcut` arm):

```rust
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    let output = &forward.layer_outputs[layer.index];
                    let delta = apply_output_activation_gradient(
                        output,
                        &grad,
                        &spec.activation,
                        "shortcut",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    let sources = shortcut_sources(spec, layer.index, &forward.layer_outputs, current_input.shape.batch)?;
                    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
                    let g = crate::shortcut_weights::weighted_backward(
                        &delta.data,
                        &current_input.data,
                        &slices,
                        current_input.shape.batch,
                        current_input.shape.element_count(),
                        &params.weights,
                        spec.weights_normalization,
                        spec.weight_slots(current_input.shape.channels),
                        GradMode::from_env() == GradMode::Cpp,
                    );
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Shortcut(ShortcutParams { weights: g.weights });
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, g.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    for (raw_index, source_grad) in spec.from.iter().zip(g.sources) {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let shape = forward.layer_outputs[source_index].shape;
                        accumulate_tensor(
                            &mut layer_output_gradients[source_index],
                            &Tensor::from_data(shape, source_grad)?,
                        )?;
                    }
                }
```

Preflights: the CPU compact list is restricted to `weights_type=None` because weighted parameters need their own forward. GPU `forward_with_state_impl` gathers/fits all sources, computes normalized host weights, and calls `shortcut_weighted_sum_f32`; its default implementation composes per-channel scale and add operations. Weighted shortcut makes `requires_cpu_backward_fallback()` true so analytical parameter/source gradients and SGD remain on the host. The weighted runtime path has FakeBackend and real-Oxide integration tests; compact inference falls back to the full forward path.

- [ ] **Step 7: Add the end-to-end CPU test** (`runtime.rs` tests) that a weighted shortcut trains consistently through the whole stack:

```rust
    #[test]
    fn weighted_shortcut_layer_backward_matches_finite_differences_on_its_weights() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=relu\nactivation=leaky\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut net = ExecutableNetwork::new(network);
        fill_conv_params(&mut net.params);
        let LayerParams::Shortcut(p) = &mut net.params[2] else { panic!() };
        p.weights = vec![0.9, 0.4, 0.7, 0.6]; // (n+1)=2 inputs x 2 channels
        let input = patterned_input(net.network.net.input_shape());
        let seed = Tensor::from_data(
            net.network.layers[2].output_shape,
            (0..16).map(|i| (i as f32 * 0.61).cos()).collect(),
        )
        .unwrap();
        let loss = |net: &ExecutableNetwork| -> f32 {
            let out = net.forward_training(&input).unwrap();
            out.layer_outputs[2].data.iter().zip(&seed.data).map(|(o, s)| o * s).sum()
        };
        let forward = net.forward_training(&input).unwrap();
        let mut seeds = vec![None; 3];
        seeds[2] = Some(seed.clone());
        let backward = net.backward_accumulate_seeded(&input, &forward, &seeds).unwrap();
        let LayerParams::Shortcut(g) = &backward.param_gradients[2] else { panic!() };
        for k in 0..4 {
            let h = 1e-2f32;
            let (mut np, mut nm) = (net.clone(), net.clone()); // ExecutableNetwork derives Clone
            if let LayerParams::Shortcut(p) = &mut np.params[2] {
                p.weights[k] += h;
            }
            if let LayerParams::Shortcut(p) = &mut nm.params[2] {
                p.weights[k] -= h;
            }
            let fd = (loss(&np) - loss(&nm)) / (2.0 * h);
            assert!(
                (fd - g.weights[k]).abs() <= 5e-3 + 5e-3 * fd.abs(),
                "w[{k}]: fd {fd}, analytic {}",
                g.weights[k]
            );
        }
    }
```

- [ ] **Step 8: Run everything** — `./tools/test_host.sh shortcut` — Expected: `shortcut_weights::tests::*`, `shortcut_weights_round_trip_exact_length`, `shortcut_weight_keys_follow_cpp_names_and_counts`, `weighted_shortcut_layer_backward_matches_finite_differences_on_its_weights` all `ok`; then `./tools/test_host.sh gpu_` to confirm the GPU runtime tests did not regress.

- [ ] **Step 9: Commit**

```bash
git add src/darknet-core/src/shortcut_weights.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs src/darknet-core/src/weights.rs src/darknet-core/src/initialize.rs src/darknet-cli/src/main.rs
git commit -m "feat(shortcut): weights_type / weights_normalization with exact normalised gradient and .weights block

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

# PHASE 0 — Tier 1: small layers (each follows the Layer Port Contract)

### Task 7: `[channel_slice]`

C++ `channel_slice_layer.cpp` (branches `Jev_laya`, `fog`, `raptor`): copies channels `[start, end)` of the OUTPUT of layer `from` (not of the previous layer). Keys `from` (required), `start`, `end`, `axis` (only `1`). C++ defaults `start = end = 2` describe an empty slice that aborts, so both are required here. No weights. C++ backward `copy_cpu`s into `from`'s delta (overwrite); we scatter-ADD (exact, and correct when `from` also feeds other layers). GPU: no new kernel; it is a one-input grouped route, so `route_concat_f32` / `route_backward_f32` (existing, kernels exist) slice and scatter.

**Files:**
- Create: `src/darknet-core/src/channel_ops.rs`
- Modify: `src/darknet-core/src/lib.rs` (`pub(crate) mod channel_ops;`), `model.rs`, `runtime.rs`
- Test: `channel_ops.rs` `mod tests`, `model.rs` tests, `runtime.rs` tests

**Interfaces:**
- Produces: `ChannelSliceSpec { pub from: isize, pub start: usize, pub end: usize }`, `LayerKind::ChannelSlice(ChannelSliceSpec)`; `channel_ops::slice_channels(source: &Tensor, start: usize, end: usize) -> Result<Tensor>`; `channel_ops::slice_channels_backward(grad: &Tensor, source_shape: TensorShape, start: usize) -> Result<Tensor>`; `model.rs` private `fn require_usize(section: &RawSection, key: &str) -> Result<usize>`.
- Consumes: `resolve_relative_layer`, `Tensor::{zeros, from_data}`.

- [ ] **Step 1: Write the failing tests.** `channel_ops.rs` (only tests so far):

```rust
//! Channel-axis host math for [channel_slice] and [channel_shuffle] (tasks/rust/layers.md Tasks 7-8).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorShape;

    fn ramp(shape: TensorShape) -> Tensor {
        let n = shape.batch * shape.element_count();
        Tensor::from_data(shape, (0..n).map(|i| i as f32).collect()).unwrap()
    }

    #[test]
    fn slice_copies_the_channel_range_of_every_sample() {
        // batch 2, 4 channels, 2x1 pixels: value = b*8 + c*2 + p
        let y = slice_channels(&ramp(TensorShape::new(2, 1, 4, 2)), 1, 3).unwrap();
        assert_eq!(y.shape, TensorShape::new(2, 1, 2, 2));
        assert_eq!(y.data, vec![2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0]);
        assert!(slice_channels(&ramp(TensorShape::new(2, 1, 4, 2)), 3, 5).is_err());
    }

    #[test]
    fn slice_backward_is_the_adjoint_and_zero_outside_the_range() {
        let source_shape = TensorShape::new(2, 1, 4, 2);
        let x = ramp(source_shape);
        let y_shape = TensorShape::new(2, 1, 2, 2);
        let y = Tensor::from_data(y_shape, (0..8).map(|i| 0.5 + i as f32 * 0.25).collect()).unwrap();
        let sliced = slice_channels(&x, 1, 3).unwrap();
        let back = slice_channels_backward(&y, source_shape, 1).unwrap();
        let lhs: f32 = sliced.data.iter().zip(&y.data).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.data.iter().zip(&back.data).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-4, "<S x, y> = {lhs}, <x, S^T y> = {rhs}");
        assert!(back.data[0..2].iter().chain(&back.data[6..8]).all(|v| *v == 0.0)); // channels 0 and 3
    }
}
```

`model.rs` tests:

```rust
    #[test]
    fn channel_slice_shape_comes_from_source_layer_and_feeds_route() {
        let net = build(
            "[net]\nwidth=4\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\n\
             [convolutional]\nfilters=5\nsize=1\n\
             [channel_slice]\nfrom=-2\nstart=1\nend=4\n\
             [route]\nlayers=-1,-2\n",
        )
        .unwrap();
        let s = net.layers[2].output_shape; // 3 channels of layer 0 (6 ch), not of the previous layer (5 ch)
        assert_eq!((s.width, s.height, s.channels, s.batch), (4, 3, 3, 2));
        assert_eq!(net.layers[3].output_shape.channels, 3 + 5);
        for bad in [
            "from=-2\nstart=2\nend=2",          // C++ default start=end=2: empty slice
            "from=-2\nstart=0\nend=7",          // beyond the source channels
            "from=-2\nstart=1",                 // end missing
            "start=1\nend=2",                   // from missing
            "from=-2\nstart=0\nend=2\naxis=0",  // only axis 1
            "from=5\nstart=0\nend=1",           // not a previous layer
        ] {
            let cfg = format!("[net]\nwidth=4\nheight=3\nchannels=2\n[convolutional]\nfilters=6\nsize=1\n[convolutional]\nfilters=5\nsize=1\n[channel_slice]\n{bad}\n");
            assert!(build(&cfg).is_err(), "{bad}");
        }
    }
```

`runtime.rs` tests (GPU on `FakeBackend` vs CPU, two chains: previous layer as source, and a non-adjacent source so the previous layer receives no gradient):

```rust
    #[test]
    fn gpu_channel_slice_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [channel_slice]\nfrom=-1\nstart=1\nend=3\n",
            1,
        );
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=4\nsize=1\nstride=1\nactivation=linear\n\
             [convolutional]\nfilters=2\nsize=1\nstride=1\nactivation=linear\n\
             [channel_slice]\nfrom=-2\nstart=0\nend=2\n",
            2,
        );
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh channel_slice` — Expected: FAIL (compile errors: `slice_channels` undefined, `[channel_slice]` is `Unsupported`).

- [ ] **Step 3: Implement the model.** `model.rs`: add the spec and variant:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSliceSpec {
    /// Layer whose output is sliced (relative negative or absolute, like route).
    pub from: isize,
    pub start: usize,
    /// Exclusive.
    pub end: usize,
}
```
`LayerKind::ChannelSlice(ChannelSliceSpec)`, `name()` arm `"channel_slice"`. Helper (free fn near `resolve_relative_layer`):

```rust
fn require_usize(section: &RawSection, key: &str) -> Result<usize> {
    section
        .get(key)
        .ok_or_else(|| {
            Error::InvalidConfig(format!("[{}] at line {} requires {key}=", section.name, section.line))
        })?
        .trim()
        .parse::<usize>()
        .map_err(|_| {
            Error::InvalidConfig(format!("[{}] at line {}: invalid {key}", section.name, section.line))
        })
}
```
Parse arm:

```rust
        "channel_slice" => {
            let from = section
                .get("from")
                .and_then(|value| value.trim().parse::<isize>().ok())
                .ok_or_else(|| {
                    Error::InvalidConfig(format!("[channel_slice] at line {} requires an integer from=", section.line))
                })?;
            if section.get_usize("axis", 1)? != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[channel_slice] at line {}: only axis=1 is supported",
                    section.line
                )));
            }
            LayerKind::ChannelSlice(ChannelSliceSpec {
                from,
                start: require_usize(section, "start")?,
                end: require_usize(section, "end")?,
            })
        }
```
`infer_output_shape` arm:

```rust
        LayerKind::ChannelSlice(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            let source = built_layers
                .get(source_index)
                .ok_or(Error::InvalidRoute { layer_index, reference: spec.from })?
                .output_shape;
            if spec.end <= spec.start || spec.end > source.channels {
                return Err(Error::InvalidConfig(format!(
                    "channel_slice layer #{layer_index}: range [{}, {}) is invalid for {} channels",
                    spec.start, spec.end, source.channels
                )));
            }
            Ok(TensorShape::new(source.width, source.height, spec.end - spec.start, source.batch))
        }
```

- [ ] **Step 4: Implement `channel_ops.rs` slice math and the runtime arms.** Above the tests in `channel_ops.rs`:

```rust
use crate::error::{Error, Result};
use crate::model::TensorShape;
use crate::runtime::Tensor;

/// Channels `[start, end)` of every sample of `source`.
pub(crate) fn slice_channels(source: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let s = source.shape;
    if start >= end || end > s.channels {
        return Err(Error::InvalidConfig(format!(
            "channel_slice range [{start}, {end}) is outside {s}"
        )));
    }
    let area = s.width * s.height;
    let run = (end - start) * area;
    let mut out = Tensor::zeros(TensorShape::new(s.width, s.height, end - start, s.batch));
    for b in 0..s.batch {
        let src = (b * s.channels + start) * area;
        out.data[b * run..(b + 1) * run].copy_from_slice(&source.data[src..src + run]);
    }
    Ok(out)
}

/// Exact adjoint of [`slice_channels`]: a zero tensor of `source_shape` with `grad`
/// written into channels `[start, start + grad.channels)`. The caller accumulates it.
pub(crate) fn slice_channels_backward(grad: &Tensor, source_shape: TensorShape, start: usize) -> Result<Tensor> {
    let g = grad.shape;
    if g.batch != source_shape.batch
        || g.width != source_shape.width
        || g.height != source_shape.height
        || start + g.channels > source_shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "channel_slice gradient {g} does not fit source {source_shape} at channel {start}"
        )));
    }
    let area = g.width * g.height;
    let run = g.channels * area;
    let mut out = Tensor::zeros(source_shape);
    for b in 0..g.batch {
        let dst = (b * source_shape.channels + start) * area;
        out.data[dst..dst + run].copy_from_slice(&grad.data[b * run..(b + 1) * run]);
    }
    Ok(out)
}
```
`runtime.rs` CPU forward arm (before the `Reorg` arm in `forward_with_mode`):

```rust
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    (crate::channel_ops::slice_channels(source, spec.start, spec.end)?, None)
                }
```
CPU backward arm (before `Reorg` in `backward_seeded_core`):

```rust
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_shape = self
                        .network
                        .layers
                        .get(source_index)
                        .ok_or(Error::InvalidRoute { layer_index: layer.index, reference: spec.from })?
                        .output_shape;
                    let source_grad =
                        crate::channel_ops::slice_channels_backward(&grad, source_shape, spec.start)?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
```
GPU training forward arm (in `forward_with_state_impl`, next to `Reorg`):

```rust
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = gpu_layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    let source_channels = self.network.layers[source_index].output_shape.channels;
                    self.backend
                        .route_concat_f32(
                            &[source],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_channels],
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("channel_slice", err))?
                }
```
GPU backward arm (in `backward_accumulate_seeded_gpu`, next to `Reorg`):

```rust
                (LayerKind::ChannelSlice(spec), Some(LayerParams::None)) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer = &self.network.layers[source_index];
                    let grads = self
                        .backend
                        .route_backward_f32(
                            &grad,
                            &[tensor_backend_shape(source_layer.output_shape).to_vec()],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_layer.output_shape.channels],
                            layer.output_shape.width * layer.output_shape.height,
                            config,
                        )
                        .map_err(|err| backend_error("GPU channel_slice backward", err))?;
                    for source_grad in grads {
                        self.accumulate_gpu_gradient(&mut layer_output_gradients[source_index], source_grad)?;
                    }
                }
```
Add `| LayerKind::ChannelSlice(_)` to the `forward_compact` GPU fallback `matches!` list (~line 4410).

- [ ] **Step 5: Run to verify they pass**

Run: `./tools/test_host.sh channel_slice` — Expected: `slice_copies_the_channel_range_of_every_sample ... ok`, `slice_backward_is_the_adjoint_and_zero_outside_the_range ... ok`, `channel_slice_shape_comes_from_source_layer_and_feeds_route ... ok`, `gpu_channel_slice_matches_cpu_forward_and_backward ... ok`.

- [ ] **Step 6: Commit**

```bash
git add src/darknet-core/src/channel_ops.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs
git commit -m "feat(layers): channel_slice (CPU + GPU via route ops)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `[channel_shuffle]`

C++ `channel_shuffle_layer.cpp`: key `groups` (default 2, `c % groups == 0`). With `rows = groups`, `cols = c / groups`: `out_channel[j*rows + i] = in_channel[i*cols + j]`. Backward is the same permutation with `rows`/`cols` swapped (C++ overwrites `state.delta`; we accumulate). No weights. New backend op with a default host implementation and a one-thread-per-output-element oxide kernel.

**Files:**
- Modify: `src/darknet-cuda/src/host_ops.rs` (`channel_permute` + tests), `src/darknet-cuda/src/lib.rs` (trait op), `src/darknet-cuda-oxide/src/kernels/layers_extra.rs`, `src/darknet-cuda-oxide/src/lib.rs`
- Modify: `src/darknet-core/src/channel_ops.rs`, `model.rs`, `runtime.rs`
- Test: `host_ops.rs`, `src/darknet-cuda-oxide/tests/oxide_layers_extra.rs`, `channel_ops.rs`, `model.rs`, `runtime.rs` tests

**Interfaces:**
- Produces: `host_ops::channel_permute(x: &[f32], batch: usize, rows: usize, cols: usize, len: usize) -> Vec<f32>`; trait op `F32TensorOpsBackend::channel_shuffle_f32(&self, x: &Self::Tensor, shape: &[usize], groups: usize, inverse: bool, config: ElementwiseLaunchConfig) -> Result<Self::Tensor, Self::Error>`; `ChannelShuffleSpec { pub groups: usize }`, `LayerKind::ChannelShuffle(ChannelShuffleSpec)`; `channel_ops::shuffle_channels(x: &Tensor, groups: usize, inverse: bool) -> Result<Tensor>`.

- [ ] **Step 1: Write the failing tests.** `host_ops.rs` (append a test module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_permute_is_the_group_transpose_and_its_inverse() {
        // 6 channels, groups = 2 -> rows 2, cols 3: out[j*2+i] = in[i*3+j]
        let x: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let shuffled = channel_permute(&x, 1, 2, 3, 1);
        assert_eq!(shuffled, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(channel_permute(&shuffled, 1, 3, 2, 1), x);
        // two samples, len 2: whole pixel runs move together
        let x2: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let y2 = channel_permute(&x2, 2, 2, 3, 2);
        assert_eq!(&y2[0..4], &[0.0, 1.0, 6.0, 7.0]); // channels 0 and 3 of sample 0
        assert_eq!(&y2[12..16], &[12.0, 13.0, 18.0, 19.0]);
    }
}
```

`channel_ops.rs` tests (append):

```rust
    #[test]
    fn shuffle_permutes_channels_and_inverse_restores() {
        let x = ramp(TensorShape::new(1, 1, 6, 1)); // channel c holds value c
        let y = shuffle_channels(&x, 2, false).unwrap();
        assert_eq!(y.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(shuffle_channels(&y, 2, true).unwrap().data, x.data);
        assert!(shuffle_channels(&x, 4, false).is_err()); // 6 % 4 != 0
    }
```

`model.rs` test:

```rust
    #[test]
    fn channel_shuffle_requires_divisible_channels() {
        let ok = build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=3\n").unwrap();
        assert_eq!(ok.layers[0].output_shape.channels, 6);
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=4\n").is_err());
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=0\n").is_err());
        // C++ default groups=2
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\n").is_ok());
    }
```

`runtime.rs` test:

```rust
    #[test]
    fn gpu_channel_shuffle_matches_cpu_forward_and_backward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\nstride=1\nactivation=linear\n\
             [channel_shuffle]\ngroups=3\n",
            1,
        );
    }
```

`oxide_layers_extra.rs` test (append):

```rust
#[test]
fn channel_shuffle_matches_reference_and_inverse_round_trips() {
    let b = backend();
    let mut rng = Rng(77);
    for (batch, c, h, w, groups) in [(2usize, 6usize, 3usize, 5usize, 2usize), (1, 12, 2, 2, 3), (3, 8, 1, 7, 4)] {
        let n = batch * c * h * w;
        let x = rng.vec(n, -1.0, 1.0);
        let shape = [batch, c, h, w];
        let xt = b.upload_f32(&shape, &x).unwrap();
        let shuffled = b.channel_shuffle_f32(&xt, &shape, groups, false, cfg()).unwrap();
        // independent reference: output channel j*groups + i takes input channel i*(c/groups) + j
        let (cols, len) = (c / groups, h * w);
        let mut want = vec![0.0f64; n];
        for bb in 0..batch {
            for i in 0..groups {
                for j in 0..cols {
                    for s in 0..len {
                        want[(bb * c + j * groups + i) * len + s] = x[(bb * c + i * cols + j) * len + s] as f64;
                    }
                }
            }
        }
        assert_close("channel_shuffle", &b.download_f32(&shuffled).unwrap(), &want);
        let back = b.channel_shuffle_f32(&shuffled, &shape, groups, true, cfg()).unwrap();
        assert_eq!(b.download_f32(&back).unwrap(), x);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `PKGS=darknet-cuda ./tools/test_host.sh channel_permute` — Expected: FAIL (`channel_permute` undefined).
Run: `./tools/test_oxide.sh --test oxide_layers_extra -- channel_shuffle` — Expected: FAIL (compile error: no method `channel_shuffle_f32`).

- [ ] **Step 3: Implement the host reference and trait op.** `host_ops.rs`:

```rust
/// Channel permutation of `[batch][rows*cols][len]`: output channel `j*rows + i` is input
/// channel `i*cols + j` (C++ `channel_shuffle_op`). With `rows = groups, cols = c/groups` this is
/// `[channel_shuffle]`; swapping `rows`/`cols` gives the exact inverse (the backward pass).
pub fn channel_permute(x: &[f32], batch: usize, rows: usize, cols: usize, len: usize) -> Vec<f32> {
    let per = rows * cols * len;
    assert!(per > 0 && x.len() == batch * per, "channel_permute: bad shape");
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                let src = b * per + (i * cols + j) * len;
                let dst = b * per + (j * rows + i) * len;
                out[dst..dst + len].copy_from_slice(&x[src..src + len]);
            }
        }
    }
    out
}
```
`src/darknet-cuda/src/lib.rs` — add to `F32TensorOpsBackend` next to `reorg_f32`:

```rust
    /// `[channel_shuffle]` on NCHW `shape` (see [`host_ops::channel_permute`]): forward
    /// (`inverse = false`) uses rows = groups, cols = c / groups; `inverse = true` is the exact
    /// adjoint (rows = c / groups, cols = groups) used by the backward pass.
    fn channel_shuffle_f32(
        &self,
        x: &Self::Tensor,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        let _ = config;
        assert!(
            shape.len() == 4 && groups > 0 && shape[1] % groups == 0,
            "channel_shuffle_f32: bad shape {shape:?} or groups {groups}"
        );
        let (rows, cols) = if inverse { (shape[1] / groups, groups) } else { (groups, shape[1] / groups) };
        let host = self.download_f32(x)?;
        let out = host_ops::channel_permute(&host, shape[0], rows, cols, shape[2] * shape[3]);
        self.upload_f32(shape, &out)
    }
```

- [ ] **Step 4: Implement the oxide kernel and override.** `kernels/layers_extra.rs`, inside `pub mod kernels` next to `reorg_to_big`:

```rust
    /// `out[b, j*rows + i, s] = x[b, i*cols + j, s]`: one thread per output element.
    #[kernel]
    pub fn channel_permute(x: &[f32], mut out: DisjointSlice<f32>, rows: usize, cols: usize, len: usize) {
        let idx = thread::index_1d();
        let g = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let per = rows * cols * len;
            let b = g / per;
            let r = g % per;
            let p = r / len; // output channel j*rows + i
            let s = r % len;
            let j = p / rows;
            let i = p % rows;
            *o = x[b * per + (i * cols + j) * len + s];
        }
    }
```
`src/darknet-cuda-oxide/src/lib.rs`, in `impl F32TensorOpsBackend for OxideBackend` next to `reorg_f32`:

```rust
    fn channel_shuffle_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        groups: usize,
        inverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if shape.len() != 4 || groups == 0 || shape[1] % groups != 0 || x.len() != shape.iter().product::<usize>() {
            return Err(shape_err(
                "channel_shuffle_f32",
                format!("bad shape {shape:?} / groups {groups} for {} elements", x.len()),
            ));
        }
        let (rows, cols) = if inverse { (shape[1] / groups, groups) } else { (groups, shape[1] / groups) };
        let len = shape[2] * shape[3];
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; every source index is < x.len() by the shape check.
        unsafe {
            self.layers_extra.channel_permute(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                rows,
                cols,
                len,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }
```

- [ ] **Step 5: Implement the model + CPU + GPU arms.** `model.rs`: `pub struct ChannelShuffleSpec { pub groups: usize }` (`Clone, Debug, PartialEq, Eq`), `LayerKind::ChannelShuffle(ChannelShuffleSpec)` + `name()` `"channel_shuffle"`; parse `"channel_shuffle" => LayerKind::ChannelShuffle(ChannelShuffleSpec { groups: section.get_usize("groups", 2)? })`; shape:

```rust
        LayerKind::ChannelShuffle(spec) => {
            if spec.groups == 0 || current.channels % spec.groups != 0 {
                return Err(Error::InvalidConfig(format!(
                    "channel_shuffle layer #{layer_index} requires channels={} divisible by groups={}",
                    current.channels, spec.groups
                )));
            }
            Ok(current)
        }
```
`channel_ops.rs`:

```rust
/// `[channel_shuffle]` forward (`inverse = false`) or its exact adjoint (`inverse = true`).
pub(crate) fn shuffle_channels(x: &Tensor, groups: usize, inverse: bool) -> Result<Tensor> {
    let s = x.shape;
    if groups == 0 || s.channels % groups != 0 {
        return Err(Error::InvalidConfig(format!(
            "channel_shuffle needs channels={} divisible by groups={groups}",
            s.channels
        )));
    }
    let (rows, cols) = if inverse { (s.channels / groups, groups) } else { (groups, s.channels / groups) };
    Tensor::from_data(
        s,
        darknet_cuda::host_ops::channel_permute(&x.data, s.batch, rows, cols, s.width * s.height),
    )
}
```
`runtime.rs` CPU forward `(LayerKind::ChannelShuffle(spec), LayerParams::None) => (crate::channel_ops::shuffle_channels(current_input, spec.groups, false)?, None),` and backward:

```rust
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => {
                    let input_grad = crate::channel_ops::shuffle_channels(&grad, spec.groups, true)?;
                    accumulate_previous_gradient(layer.index, &input_grad, &mut input_gradient, &mut layer_output_gradients)?;
                }
```
GPU training forward:

```rust
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => self
                    .backend
                    .channel_shuffle_f32(
                        current_input,
                        &tensor_backend_shape(layer.input_shape),
                        spec.groups,
                        false,
                        launch_config,
                    )
                    .map_err(|err| backend_error("channel_shuffle", err))?,
```
GPU backward:

```rust
                (LayerKind::ChannelShuffle(spec), Some(LayerParams::None)) => {
                    let input_grad = self
                        .backend
                        .channel_shuffle_f32(&grad, &tensor_backend_shape(layer.output_shape), spec.groups, true, config)
                        .map_err(|err| backend_error("GPU channel_shuffle backward", err))?;
                    self.accumulate_previous_gpu_gradient(layer.index, input_grad, &mut layer_output_gradients)?;
                }
```
plus `| LayerKind::ChannelShuffle(_)` in the `forward_compact` GPU fallback list.

- [ ] **Step 6: Run to verify**

Run: `PKGS=darknet-cuda ./tools/test_host.sh channel_permute` — Expected: `ok`.
Run: `./tools/test_host.sh channel_shuffle` — Expected: `shuffle_permutes_channels_and_inverse_restores`, `channel_shuffle_requires_divisible_channels`, `gpu_channel_shuffle_matches_cpu_forward_and_backward` all `ok`.
Run: `./tools/test_oxide.sh --test oxide_layers_extra -- channel_shuffle` — Expected: `channel_shuffle_matches_reference_and_inverse_round_trips ... ok`.

- [ ] **Step 7: Commit**

```bash
git add src/darknet-cuda src/darknet-core/src/channel_ops.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs src/darknet-cuda-oxide/src src/darknet-cuda-oxide/tests/oxide_layers_extra.rs
git commit -m "feat(layers): channel_shuffle with host reference op and oxide kernel

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `[deconvolutional]`

C++ `deconvolutional_layer.cpp` (`Jev_laya`): transposed convolution. Keys `filters` (n), `size` (k), `stride` (s), `activation` (default `logistic`); `pad`/`padding` are rejected. Output `n x (s*(h-1)+k) x (s*(w-1)+k)`. Weights `[c][n][k][k]`, biases `[n]`, no BN; `.weights` = biases then weights (same layout as a convolution without BN, so `LayerParams::Convolutional` is reused and load/save need no new variant). Math: `out[o, y*s+ky, x*s+kx] += in[ci, y, x] * W[ci, o, ky, kx]`, then `+ bias[o]`, then activation. Backward (delta after the activation derivative): `db[o] = sum delta`, `dW[ci,o,ky,kx] = sum in[ci,y,x]*delta[o,y*s+ky,x*s+kx]`, `dx[ci,y,x] = sum W[ci,o,ky,kx]*delta[o,y*s+ky,x*s+kx]`. C++ additionally multiplies `dW` by `1/batch` and the optimizer divides by `batch` again (double normalisation); the exact gradient omits it; `GradMode::Cpp` reproduces it. GPU: forward via host round trip, backward via the whole-net CPU fallback (`requires_cpu_backward_fallback`), update through `update_cpu_fallback_weights`. (A later optimisation can map this onto `conv2d_backward_input_f32` / `conv2d_f32` / `conv2d_backward_weights_f32`, which are exactly the transposed pair with `filters = c`, `input_channels = n`.)

**Files:**
- Create: `src/darknet-core/src/deconv.rs`
- Modify: `src/darknet-core/src/lib.rs`, `model.rs`, `runtime.rs`, `weights.rs`
- Test: `deconv.rs` `mod tests`, `model.rs`, `weights.rs`, `runtime.rs` tests

**Interfaces:**
- Produces: `DeconvolutionalSpec { pub filters: usize, pub size: usize, pub stride: usize, pub activation: Activation }`, `LayerKind::Deconvolutional(..)`; `deconv::deconv_forward(input: &Tensor, weights: &[f32], biases: &[f32], size: usize, stride: usize, output_shape: TensorShape) -> Tensor` (pre-activation); `deconv::DeconvGrad { pub input: Tensor, pub weights: Vec<f32>, pub biases: Vec<f32> }`; `deconv::deconv_backward(input: &Tensor, weights: &[f32], delta: &Tensor, size: usize, stride: usize, weight_scale: f32) -> DeconvGrad`.
- Consumes: `LayerParams::Convolutional(ConvolutionalParams { biases, batch_norm, weights })`, `default_convolutional_params`, `default_convolutional_train_buffers`, `apply_output_activation_gradient`.

- [ ] **Step 1: Write the failing tests.** `deconv.rs`:

```rust
//! Transposed convolution host math (tasks/rust/layers.md Task 9).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorShape;

    #[test]
    fn two_by_two_stride_two_scatters_each_input_pixel_into_its_own_block() {
        // c = 1 -> n = 1, k = 2, s = 2: out is 4x4, each input pixel paints a 2x2 block with W
        let input = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let weights = [1.0, 10.0, 100.0, 1000.0]; // W[0][0][ky][kx]
        let out = deconv_forward(&input, &weights, &[0.5], 2, 2, TensorShape::new(4, 4, 1, 1));
        let row = |y: usize| (0..4).map(|x| out.get(0, 0, y, x)).collect::<Vec<_>>();
        assert_eq!(row(0), vec![1.5, 10.5, 2.5, 20.5]);
        assert_eq!(row(1), vec![100.5, 1000.5, 200.5, 2000.5]);
        assert_eq!(row(2), vec![3.5, 30.5, 4.5, 40.5]);
        assert_eq!(row(3), vec![300.5, 3000.5, 400.5, 4000.5]);
    }

    #[test]
    fn overlapping_blocks_accumulate() {
        // k = 3, s = 1 on a 1x2 input: out width 4; the middle columns receive both pixels
        let input = Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 1.0]).unwrap();
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = deconv_forward(&input, &weights, &[0.0], 3, 1, TensorShape::new(4, 3, 1, 1));
        assert_eq!(out.get(0, 0, 0, 1), 2.0 + 1.0); // pixel0 kx=1 + pixel1 kx=0
    }

    #[test]
    fn backward_matches_central_differences() {
        let (c, n, k, s) = (2usize, 3usize, 3usize, 2usize);
        let ishape = TensorShape::new(3, 2, c, 2);
        let oshape = TensorShape::new(s * (3 - 1) + k, s * (2 - 1) + k, n, 2);
        let fill = |len: usize, f: f32| (0..len).map(|i| (i as f32 * f).sin()).collect::<Vec<f32>>();
        let input = Tensor::from_data(ishape, fill(2 * ishape.element_count(), 0.37)).unwrap();
        let weights = fill(c * n * k * k, 0.53);
        let biases = fill(n, 1.1);
        let seed = Tensor::from_data(oshape, fill(2 * oshape.element_count(), 0.91)).unwrap();
        let loss = |x: &Tensor, w: &[f32], b: &[f32]| -> f32 {
            deconv_forward(x, w, b, k, s, oshape).data.iter().zip(&seed.data).map(|(o, g)| o * g).sum()
        };
        let g = deconv_backward(&input, &weights, &seed, k, s, 1.0);
        let h = 1e-2f32;
        let check = |name: &str, analytic: f32, plus: f32, minus: f32| {
            let fd = (plus - minus) / (2.0 * h);
            assert!((fd - analytic).abs() <= 5e-3 + 5e-3 * fd.abs(), "{name}: fd {fd}, analytic {analytic}");
        };
        for i in (0..weights.len()).step_by(7) {
            let (mut p, mut m) = (weights.clone(), weights.clone());
            p[i] += h;
            m[i] -= h;
            check(&format!("w[{i}]"), g.weights[i], loss(&input, &p, &biases), loss(&input, &m, &biases));
        }
        for i in 0..n {
            let (mut p, mut m) = (biases.clone(), biases.clone());
            p[i] += h;
            m[i] -= h;
            check(&format!("b[{i}]"), g.biases[i], loss(&input, &weights, &p), loss(&input, &weights, &m));
        }
        for i in (0..input.data.len()).step_by(5) {
            let (mut p, mut m) = (input.clone(), input.clone());
            p.data[i] += h;
            m.data[i] -= h;
            check(&format!("x[{i}]"), g.input.data[i], loss(&p, &weights, &biases), loss(&m, &weights, &biases));
        }
    }
}
```

`model.rs` test:

```rust
    #[test]
    fn deconvolutional_output_is_stride_times_h_minus_one_plus_size() {
        let net = build(
            "[net]\nwidth=3\nheight=2\nchannels=2\n\
             [deconvolutional]\nfilters=4\nsize=3\nstride=2\nactivation=leaky\n",
        )
        .unwrap();
        let s = net.layers[0].output_shape;
        assert_eq!((s.width, s.height, s.channels), (2 * 2 + 3, 2 * 1 + 3, 4));
        // weights(): biases[n] then weights[c*n*k*k] (C++ save_deconvolutional_weights order)
        let w = net.layers[0].weights().unwrap();
        assert_eq!(w.entries[0].len, 4);
        assert_eq!(w.entries[1].len, 2 * 4 * 9);
        // C++ default activation is logistic; pad/padding are rejected
        let net = build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\n").unwrap();
        assert!(matches!(&net.layers[0].kind, LayerKind::Deconvolutional(s) if s.activation == super::Activation::Logistic));
        assert!(build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npad=1\n").is_err());
        assert!(build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npadding=1\n").is_err());
    }
```

`weights.rs` test:

```rust
    #[test]
    fn deconvolutional_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str("d.cfg", "[net]\nwidth=2\nheight=2\nchannels=2\n[deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation=linear\n").expect("cfg"),
        )
        .expect("network");
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Convolutional(p) = &mut executable.params[0] else { panic!("conv-shaped params") };
        p.biases = vec![0.5, 1.5, 2.5];
        p.weights = (0..2 * 3 * 4).map(|i| i as f32).collect();
        let path = std::env::temp_dir().join("darknet-deconv-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * (3 + 24));
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        assert_eq!(inspect_weights_file(&network, &path).unwrap().trailing_bytes, 0);
        let _ = fs::remove_file(path);
    }
```

`runtime.rs` tests: CPU activation + SGD step, then GPU:

```rust
    #[test]
    fn deconvolutional_layer_trains_on_cpu_and_gpu_fallback_identically() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation=leaky\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        fill_conv_params(&mut cpu.params); // fills Convolutional-shaped params of both layers
        let mut gpu = GpuExecutableNetwork::new(network, FakeBackend);
        gpu.params = cpu.params.clone();
        gpu.init_train_buffers_gpu().unwrap();
        assert!(gpu.requires_cpu_backward_fallback(), "deconvolutional trains through the CPU fallback");

        let input = patterned_input(cpu.network.net.input_shape());
        let cpu_forward = cpu.forward_training(&input).unwrap();
        let gpu_forward = gpu.forward_training_with_state(&input).unwrap();
        for (i, (a, b)) in cpu_forward.layer_outputs.iter().zip(&gpu_forward.cpu_forward.layer_outputs).enumerate() {
            assert_slice_close(&format!("layer {i}"), &a.data, &b.data, 1e-5);
        }
        let shape = cpu.network.layers[1].output_shape;
        let mut seeds = vec![None; 2];
        seeds[1] = Some(Tensor::from_data(shape, (0..2 * shape.element_count()).map(|i| (i as f32 * 0.91).cos()).collect()).unwrap());
        cpu.backward_accumulate_seeded(&input, &cpu_forward, &seeds).unwrap();
        gpu.backward_accumulate_seeded_gpu(&gpu_forward, &seeds).unwrap();
        let LayerTrainBuffers::Convolutional(cpu_train) = &cpu.train_buffers[1] else { panic!() };
        let fallback = gpu.cpu_fallback_train_buffers.borrow();
        let LayerTrainBuffers::Convolutional(gpu_train) = &fallback[1] else { panic!() };
        assert!(cpu_train.weight_updates.iter().any(|g| g.abs() > 1e-6));
        assert_slice_close("weight_updates", &cpu_train.weight_updates, &gpu_train.weight_updates, 1e-4);
        assert_slice_close("bias_updates", &cpu_train.bias_updates, &gpu_train.bias_updates, 1e-4);
        drop(fallback);

        // one SGD step moves the host-resident deconvolution weights on both paths alike
        cpu.update_weights(0.01, 0.9, 0.0005, 2);
        gpu.update_weights_gpu(0.01, 0.9, 0.0005, 2).unwrap();
        let (LayerParams::Convolutional(a), LayerParams::Convolutional(b)) = (&cpu.params[1], &gpu.params[1]) else { panic!() };
        assert_slice_close("updated weights", &a.weights, &b.weights, 1e-5);
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh deconv` — Expected: FAIL (compile errors: `deconv_forward` undefined; `[deconvolutional]` is `Unsupported`).

- [ ] **Step 3: Implement `deconv.rs`** (above the tests; `pub(crate) mod deconv;` in `lib.rs`):

```rust
use crate::model::TensorShape;
use crate::runtime::Tensor;

/// Pre-activation transposed convolution: `weights` is `[c][n][k][k]`, `output_shape` is
/// `n x (s*(h-1)+k) x (s*(w-1)+k)`.
pub(crate) fn deconv_forward(
    input: &Tensor,
    weights: &[f32],
    biases: &[f32],
    size: usize,
    stride: usize,
    output_shape: TensorShape,
) -> Tensor {
    let (c, n) = (input.shape.channels, output_shape.channels);
    let mut out = Tensor::zeros(output_shape);
    for b in 0..input.shape.batch {
        for ci in 0..c {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let value = input.get(b, ci, y, x);
                    for o in 0..n {
                        for ky in 0..size {
                            for kx in 0..size {
                                let w = weights[((ci * n + o) * size + ky) * size + kx];
                                let (oy, ox) = (y * stride + ky, x * stride + kx);
                                let current = out.get(b, o, oy, ox);
                                out.set(b, o, oy, ox, current + value * w);
                            }
                        }
                    }
                }
            }
        }
        for o in 0..n {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let current = out.get(b, o, oy, ox);
                    out.set(b, o, oy, ox, current + biases[o]);
                }
            }
        }
    }
    out
}

pub(crate) struct DeconvGrad {
    pub input: Tensor,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

/// Exact backward for `delta` = output gradient after the activation derivative.
/// `weight_scale` multiplies the weight gradient (1.0; `1/batch` reproduces the C++ quirk).
pub(crate) fn deconv_backward(
    input: &Tensor,
    weights: &[f32],
    delta: &Tensor,
    size: usize,
    stride: usize,
    weight_scale: f32,
) -> DeconvGrad {
    let (c, n) = (input.shape.channels, delta.shape.channels);
    let mut grad = DeconvGrad {
        input: Tensor::zeros(input.shape),
        weights: vec![0.0; weights.len()],
        biases: vec![0.0; n],
    };
    for b in 0..input.shape.batch {
        for o in 0..n {
            for oy in 0..delta.shape.height {
                for ox in 0..delta.shape.width {
                    grad.biases[o] += delta.get(b, o, oy, ox);
                }
            }
        }
        for ci in 0..c {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let value = input.get(b, ci, y, x);
                    let mut dx = 0.0f32;
                    for o in 0..n {
                        for ky in 0..size {
                            for kx in 0..size {
                                let d = delta.get(b, o, y * stride + ky, x * stride + kx);
                                let wi = ((ci * n + o) * size + ky) * size + kx;
                                dx += weights[wi] * d;
                                grad.weights[wi] += weight_scale * value * d;
                            }
                        }
                    }
                    grad.input.set(b, ci, y, x, dx);
                }
            }
        }
    }
    grad
}
```

- [ ] **Step 4: Implement the model.** `model.rs`: spec + variant + `name()` `"deconvolutional"`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionalSpec {
    pub filters: usize,
    pub size: usize,
    pub stride: usize,
    pub activation: Activation,
}
```
parse (`"deconvolutional"`; C++ defaults filters=1 size=1 stride=1 activation=logistic):

```rust
        "deconvolutional" => {
            if section.get_usize("pad", 0)? != 0 || section.get_usize("padding", 0)? != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[deconvolutional] at line {} does not support pad/padding",
                    section.line
                )));
            }
            LayerKind::Deconvolutional(DeconvolutionalSpec {
                filters: section.get_usize("filters", 1)?,
                size: section.get_usize("size", 1)?,
                stride: section.get_usize("stride", 1)?,
                activation: Activation::parse(&section.get_str("activation", "logistic")),
            })
        }
```
shape: reject zeros, then `TensorShape::new(spec.stride * (current.width - 1) + spec.size, spec.stride * (current.height - 1) + spec.size, spec.filters, current.batch)` (error `InvalidConfig` when `current.width == 0 || current.height == 0 || filters == 0 || size == 0 || stride == 0`). `Layer::weights()` arm:

```rust
            LayerKind::Deconvolutional(spec) => Some(LayerWeights {
                entries: vec![
                    WeightsEntry { array: WeightsArray::Biases, len: spec.filters },
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: self.input_shape.channels * spec.filters * spec.size * spec.size,
                    },
                ],
            }),
```
Task 2's `LayerKind::activation()`: add `Self::Deconvolutional(spec) => Some(&spec.activation),`.

- [ ] **Step 5: Implement params, weights and CPU arms.** `runtime.rs`: `default_layer_params`:
```rust
        LayerKind::Deconvolutional(spec) => LayerParams::Convolutional(default_convolutional_params(
            layer.input_shape.channels, spec.filters, spec.size, 1, false,
        )),
```
(`filters * channels * size^2` weights = `[c][n][k][k]`); `default_layer_train_buffers`: `LayerKind::Deconvolutional(spec) => LayerTrainBuffers::Convolutional(default_convolutional_train_buffers(layer.input_shape.channels, spec.filters, spec.size, 1, false)),`. `accumulate_train_buffers` (next to the `Convolutional` arm):

```rust
                (LayerKind::Deconvolutional(_), LayerParams::Convolutional(conv_grad)) => {
                    let Some(LayerTrainBuffers::Convolutional(train)) = self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deconvolutional train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &conv_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &conv_grad.weights)?;
                }
```
`weights.rs` `save_weights`: `(LayerKind::Deconvolutional(_), LayerParams::Convolutional(conv)) => { write_f32_slice(&mut file, &conv.biases)?; write_f32_slice(&mut file, &conv.weights)?; }`. (Loading is keyed on the params variant and already reads `biases, weights`; `initialize.rs` needs no change: He-uniform with `fan = c*k*k`.) CPU forward arm:

```rust
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let pre = crate::deconv::deconv_forward(
                        current_input, &p.weights, &p.biases, spec.size, spec.stride, layer.output_shape,
                    );
                    let data = pre.data.iter().map(|v| activate(spec.activation.clone(), *v)).collect();
                    (Tensor::from_data(pre.shape, data)?, Some(pre))
                }
```
CPU backward arm:

```rust
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let delta = apply_output_activation_gradient(
                        &forward.layer_outputs[layer.index],
                        &grad,
                        &spec.activation,
                        "deconvolutional",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    // C++ also scales the weight gradient by 1/batch (and the optimizer divides again).
                    let weight_scale = if GradMode::from_env() == GradMode::Cpp {
                        1.0 / current_input.shape.batch as f32
                    } else {
                        1.0
                    };
                    let g = crate::deconv::deconv_backward(current_input, &p.weights, &delta, spec.size, spec.stride, weight_scale);
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!("missing deconvolutional gradient slot for layer #{}", layer.index)));
                    };
                    *slot = LayerParams::Convolutional(ConvolutionalParams { biases: g.biases, batch_norm: None, weights: g.weights });
                    accumulate_previous_gradient(layer.index, &g.input, &mut input_gradient, &mut layer_output_gradients)?;
                }
```

- [ ] **Step 6: GPU arms (host round trip + CPU fallback).** `forward_with_state_impl` arm (next to `Sam`; `activation_input_gpu` is the existing per-layer cache variable):

```rust
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download deconvolutional input", err))?;
                    let pre = crate::deconv::deconv_forward(
                        &host_input, &p.weights, &p.biases, spec.size, spec.stride, layer.output_shape,
                    );
                    activation_input_gpu = Some(
                        self.backend
                            .upload_f32(&tensor_backend_shape(layer.output_shape), &pre.data)
                            .map_err(|err| backend_error("upload deconvolutional activation cache", err))?,
                    );
                    let activated: Vec<f32> = pre.data.iter().map(|v| activate(spec.activation.clone(), *v)).collect();
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &activated)
                        .map_err(|err| backend_error("upload deconvolutional output", err))?
                }
```
`requires_cpu_backward_fallback`: add `LayerKind::Deconvolutional(_) => return true,` to the first `match` (next to the recurrent layers). `update_cpu_fallback_weights` (it skips `LayerKind::Convolutional` kinds because they are device-resident; a deconvolution is host-resident): add the arm

```rust
                (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train))
                    if matches!(layer.kind, LayerKind::Deconvolutional(_)) =>
                {
                    sgd_update(&mut params.weights, &mut train.weight_updates, lr, momentum, decay, batch);
                    sgd_update(&mut params.biases, &mut train.bias_updates, lr, momentum, 0.0, batch);
                }
```
(the surrounding `match (params, train)` has `layer` in scope from the `.zip(&self.network.layers)`). `update_cpu_fallback_connected_adam`: the existing arm `(LayerParams::None, _) | (LayerParams::Convolutional(_), _) => {}` would silently skip a deconvolution under Adam; add before it
```rust
                (LayerParams::Convolutional(_), _) if matches!(layer.kind, LayerKind::Deconvolutional(_)) => {
                    return Err(Error::InvalidConfig(format!(
                        "adam=1 on the GPU path has no optimizer state for deconvolutional layer #{index}"
                    )));
                }
```
Add `| LayerKind::Deconvolutional(_)` to the `forward_compact` GPU fallback list.

- [ ] **Step 7: Run to verify**

Run: `./tools/test_host.sh deconv` — Expected: `deconv::tests::*` (3), `deconvolutional_output_is_stride_times_h_minus_one_plus_size`, `deconvolutional_weights_round_trip_exact_length`, `deconvolutional_layer_trains_on_cpu_and_gpu_fallback_identically` all `ok`. Then `./tools/test_host.sh weights` and `./tools/test_host.sh gpu_` — Expected: no regressions. (If the GPU fallback test reveals that `cpu_fallback_train_buffers` for the deconvolution is the zeroed default because `backward_accumulate_seeded_gpu_cpu_fallback` only copies non-`Convolutional` kinds back — it does include `Deconvolutional`, see the final loop at ~5561 — no change is needed; if it fails, fix THAT loop, do not special-case the test.)

- [ ] **Step 8: Commit**

```bash
git add src/darknet-core/src/deconv.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs src/darknet-core/src/weights.rs
git commit -m "feat(layers): deconvolutional (CPU exact backward, GPU host round trip, conv-layout weights)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Dense-truth loss heads: `[cost]` scale, `[logistic]`, `[regression]`

Rust has no generic loss hook (only `compute_yolo_training_loss`). This task adds `dense_loss.rs`: given the forward result and a per-layer truth tensor it returns Darknet descent-direction seeds and the mean cost, exactly like the C++ `forward_*_layer` with `state.truth`. Convention (same as yolo): the seed is placed in the loss layer's output-gradient slot already expressed w.r.t. the layer's INPUT (logit) and the layer's backward passes it through. Wiring into `detector train` is out of scope (Open Question 5).

C++ references: `cost_layer.cpp` (`master`: types `sse` (default), `masked`, `smooth`; `scale` key; `delta = scale * (truth - input)`, `masked` sets input to `SECRET_NUM = -1234` where truth is `SECRET_NUM`), `logistic_layer.cpp` (`raptor`: output = sigmoid(input); with truth `delta = t - p`, `error = -t ln p - (1-t) ln(1-p)`; backward adds delta), `regression_layer.cpp` (`fog`: l2, key `scale` default 1; C++ leaves `l.output` holding the per-element error; Rust passes the prediction through so inference is usable).

**Files:**
- Create: `src/darknet-core/src/dense_loss.rs`
- Modify: `src/darknet-core/src/lib.rs` (`pub mod dense_loss;` + re-export `DenseTrainingLoss, compute_dense_training_loss`), `model.rs`, `runtime.rs`
- Test: `dense_loss.rs` `mod tests`, `model.rs`, `runtime.rs`

**Interfaces:**
- Produces: `LayerKind::Cost { kind: String, scale: f32 }` (field added), `LayerKind::Logistic`, `LayerKind::Regression(RegressionSpec { pub scale: f32 })`; `pub const SECRET_NUM: f32 = -1234.0`; `pub struct DenseTrainingLoss { pub loss: f32, pub layer_output_gradient_seeds: Vec<Option<Tensor>> }`; `pub fn compute_dense_training_loss(network: &Network, forward: &ForwardResult, input: &Tensor, truth: &[Option<Tensor>]) -> Result<DenseTrainingLoss>` (`truth[i] = Some(t)` for loss layer `i`; `t` has the layer's output shape).

- [ ] **Step 1: Write the failing tests.** `dense_loss.rs`:

```rust
//! Host loss for dense-truth heads (tasks/rust/layers.md Task 10).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigFile, ExecutableNetwork, LayerParams, Network, TensorShape};

    fn run(cfg: &str, set: impl FnOnce(&mut ExecutableNetwork), truth: Vec<f32>, head: usize) -> DenseTrainingLoss {
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", cfg).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        set(&mut exec);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let shape = exec.network.layers[head].output_shape;
        let mut t = vec![None; exec.network.layers.len()];
        t[head] = Some(Tensor::from_data(shape, truth).unwrap());
        compute_dense_training_loss(&exec.network, &forward, &input, &t).unwrap()
    }

    fn set_bias(exec: &mut ExecutableNetwork, bias: Vec<f32>) {
        let LayerParams::Connected(p) = &mut exec.params[0] else { panic!() };
        p.biases = bias;
    }

    #[test]
    fn cost_sse_smooth_masked_match_c_formulas() {
        let cfg = |kind: &str| format!("[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\nactivation=linear\n[cost]\ntype={kind}\nscale=0.5\n");
        // prediction = bias = [1, 2]; truth [3, 0]: diff [2, -2]
        let sse = run(&cfg("sse"), |e| set_bias(e, vec![1.0, 2.0]), vec![3.0, 0.0], 1);
        assert_eq!(sse.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![1.0, -1.0]); // scale * diff
        assert_eq!(sse.loss, 8.0); // 4 + 4
        // truth [4, 2.5]: diff [3, 0.5] -> |3| >= 1: 2*3-1 = 5, 0.25; deltas 1, 0.5 (sign for |diff|>=1) * scale
        let smooth = run(&cfg("smooth"), |e| set_bias(e, vec![1.0, 2.0]), vec![4.0, 2.5], 1);
        assert_eq!(smooth.loss, 5.25);
        assert_eq!(smooth.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![0.5, 0.25]);
        // masked: first truth is SECRET_NUM -> prediction masked, no loss/gradient there
        let masked = run(&cfg("masked"), |e| set_bias(e, vec![1.0, 2.0]), vec![SECRET_NUM, 0.0], 1);
        assert_eq!(masked.loss, 4.0);
        assert_eq!(masked.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![0.0, -1.0]);
    }

    #[test]
    fn logistic_head_uses_sigmoid_output_and_logit_space_seed() {
        // zero connected params -> logit 0 -> p = 0.5 for every element, batch 2
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=3\nbatch=2\n[connected]\noutput=3\nactivation=linear\n[logistic]\n";
        let loss = run(cfg, |_| {}, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        assert_eq!(seed.data, vec![0.5, -0.5, 0.5, -0.5, -0.5, 0.5]); // t - p
        assert!((loss.loss - 6.0 * std::f32::consts::LN_2).abs() < 1e-5, "{}", loss.loss);
    }

    #[test]
    fn regression_head_is_l2_with_scale_and_passes_the_prediction_through() {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\nactivation=linear\n[regression]\nscale=2\n";
        let loss = run(cfg, |e| set_bias(e, vec![1.0, 2.0]), vec![3.0, 1.0], 1);
        assert_eq!(loss.layer_output_gradient_seeds[1].as_ref().unwrap().data, vec![4.0, -2.0]); // 2 * (t - p)
        assert_eq!(loss.loss, 4.0 + 1.0);
    }

    #[test]
    fn layers_without_truth_are_skipped_and_wrong_shapes_rejected() {
        let network = Network::from_config(ConfigFile::parse_str("d.cfg", "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\n[cost]\n").unwrap()).unwrap();
        let exec = ExecutableNetwork::new(network);
        let input = Tensor::zeros(exec.network.net.input_shape());
        let forward = exec.forward_training(&input).unwrap();
        let none = compute_dense_training_loss(&exec.network, &forward, &input, &[None, None]).unwrap();
        assert_eq!(none.loss, 0.0);
        assert!(none.layer_output_gradient_seeds.iter().all(Option::is_none));
        let bad = Tensor::zeros(TensorShape::new(1, 1, 3, 1));
        assert!(compute_dense_training_loss(&exec.network, &forward, &input, &[None, Some(bad)]).is_err());
    }
}
```
`model.rs` test:

```rust
    #[test]
    fn loss_head_sections_parse_with_scale() {
        let net = build(
            "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\n\
             [cost]\ntype=masked\nscale=0.25\n[logistic]\n[regression]\nscale=3\n",
        )
        .unwrap();
        assert!(matches!(&net.layers[1].kind, LayerKind::Cost { kind, scale } if kind == "masked" && *scale == 0.25));
        assert!(matches!(net.layers[2].kind, LayerKind::Logistic));
        assert!(matches!(&net.layers[3].kind, LayerKind::Regression(r) if r.scale == 3.0));
        assert_eq!(net.layers[3].output_shape, net.layers[2].output_shape);
    }
```
`runtime.rs` test (GPU logistic head):

```rust
    #[test]
    fn gpu_logistic_head_matches_cpu_forward_and_passes_the_seed_through() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=3\nsize=1\nstride=1\nactivation=linear\n\
             [logistic]\n",
            1,
        );
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh dense_loss` / `loss_head_sections` / `gpu_logistic_head` — Expected: FAIL (compile errors: `compute_dense_training_loss` undefined, `Cost` has no field `scale`).

- [ ] **Step 3: Model changes.** `model.rs`: `Cost { kind: String }` becomes `Cost { kind: String, scale: f32 }`; its parse arm adds `scale: section.get_f32("scale", 1.0)?`; add

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionSpec {
    pub scale: f32,
}
```
variants `LayerKind::Logistic`, `LayerKind::Regression(RegressionSpec)`, `name()` arms `"logistic"` / `"regression"`, parse arms `"logistic" => LayerKind::Logistic`, `"regression" => LayerKind::Regression(RegressionSpec { scale: section.get_f32("scale", 1.0)? })`, and `Logistic | Regression(_)` in the `Ok(current)` shape arm that already lists `Softmax | Cost | Dropout | Contrastive`.

- [ ] **Step 4: `dense_loss.rs`** (above the tests; add `pub mod dense_loss;` and `pub use dense_loss::{DenseTrainingLoss, compute_dense_training_loss};` to `lib.rs`):

```rust
use crate::error::{Error, Result};
use crate::model::{LayerKind, Network};
use crate::runtime::{ForwardResult, Tensor};

/// C++ `SECRET_NUM`: a truth value that masks the element out of a `masked` cost layer.
pub const SECRET_NUM: f32 = -1234.0;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseTrainingLoss {
    /// Mean of the per-layer `cost` over the loss layers that had truth (C++ `get_network_cost`).
    pub loss: f32,
    /// Descent-direction seeds, `Some` for every loss layer that had truth. Feed them to
    /// `backward_accumulate_seeded`; the loss layers' backward arms pass them through.
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
}

pub fn compute_dense_training_loss(
    network: &Network,
    forward: &ForwardResult,
    input: &Tensor,
    truth: &[Option<Tensor>],
) -> Result<DenseTrainingLoss> {
    if truth.len() != network.layers.len() {
        return Err(Error::InvalidConfig(format!(
            "dense truth count mismatch: expected {}, got {}",
            network.layers.len(),
            truth.len()
        )));
    }
    let mut seeds = vec![None; network.layers.len()];
    let (mut total, mut count) = (0.0f32, 0usize);
    for layer in &network.layers {
        let Some(t) = truth[layer.index].as_ref() else { continue };
        if t.shape != layer.output_shape {
            return Err(Error::InvalidConfig(format!(
                "layer #{} ([{}]) truth shape {} != output shape {}",
                layer.index,
                layer.kind.name(),
                t.shape,
                layer.output_shape
            )));
        }
        let prev = if layer.index == 0 { input } else { &forward.layer_outputs[layer.index - 1] };
        let (delta, cost) = match &layer.kind {
            LayerKind::Cost { kind, scale } => elementwise_cost(prev, t, kind, *scale),
            LayerKind::Regression(spec) => elementwise_cost(prev, t, "sse", spec.scale),
            LayerKind::Logistic => logistic_cost(&forward.layer_outputs[layer.index], t),
            other => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([{}]) is not a dense-truth loss layer",
                    layer.index,
                    other.name()
                )));
            }
        };
        seeds[layer.index] = Some(delta);
        total += cost;
        count += 1;
    }
    Ok(DenseTrainingLoss {
        loss: if count == 0 { 0.0 } else { total / count as f32 },
        layer_output_gradient_seeds: seeds,
    })
}

/// C++ `l2_cpu` / `smooth_l1_cpu` (+ `masked`): `delta = scale * (truth - pred)`, `cost = sum error`.
fn elementwise_cost(pred: &Tensor, truth: &Tensor, kind: &str, scale: f32) -> (Tensor, f32) {
    let (masked, smooth) = (kind == "masked", kind == "smooth");
    let mut delta = Tensor::zeros(pred.shape);
    let mut cost = 0.0f32;
    for (i, (p, t)) in pred.data.iter().zip(&truth.data).enumerate() {
        // C++ overwrites the prediction with SECRET_NUM where the truth is SECRET_NUM => diff 0.
        let p = if masked && *t == SECRET_NUM { SECRET_NUM } else { *p };
        let diff = t - p;
        let (error, d) = if smooth && diff.abs() >= 1.0 {
            (2.0 * diff.abs() - 1.0, if diff > 0.0 { 1.0 } else { -1.0 })
        } else {
            (diff * diff, diff)
        };
        cost += error;
        delta.data[i] = d * scale;
    }
    (delta, cost)
}

/// C++ `logistic_x_ent_cpu` on the sigmoid output: `delta = t - p` (logit space), cross-entropy cost.
/// `ln` arguments are clamped to `[1e-12, 1 - 1e-12]` for the reported cost only (C++ yields inf).
fn logistic_cost(pred: &Tensor, truth: &Tensor) -> (Tensor, f32) {
    let mut delta = Tensor::zeros(pred.shape);
    let mut cost = 0.0f32;
    for (i, (p, t)) in pred.data.iter().zip(&truth.data).enumerate() {
        let pc = p.clamp(1e-12, 1.0 - 1e-12);
        cost += -t * pc.ln() - (1.0 - t) * (1.0 - pc).ln();
        delta.data[i] = t - p;
    }
    (delta, cost)
}
```

- [ ] **Step 5: Runtime arms.** CPU forward: `Logistic` computes the sigmoid, `Regression` passes through (add `| (LayerKind::Regression(_), LayerParams::None)` to the existing yolo/cost pass-through forward arm at ~1053):

```rust
                (LayerKind::Logistic, LayerParams::None) => {
                    let data = current_input.data.iter().map(|v| activate(Activation::Logistic, *v)).collect();
                    (Tensor::from_data(current_input.shape, data)?, None)
                }
```
CPU backward: add `| (LayerKind::Logistic, LayerParams::None) | (LayerKind::Regression(_), LayerParams::None)` to the pass-through backward arm (the one listing `Yolo | GaussianYolo | Region | Detection | Cost | Contrastive`, ~1651). GPU training forward: add `Regression(_)` to the pass-through list at ~4017 and the arm

```rust
                (LayerKind::Logistic, LayerParams::None) => self
                    .backend
                    .activation_f32(current_input, darknet_cuda::F32ActivationKind::Logistic, launch_config)
                    .map_err(|err| backend_error("logistic", err))?,
```
GPU backward: add `| (LayerKind::Logistic, Some(LayerParams::None)) | (LayerKind::Regression(_), Some(LayerParams::None))` to the pass-through list at ~5438. Add `| LayerKind::Logistic | LayerKind::Regression(_)` to the `forward_compact` GPU fallback list.

- [ ] **Step 6: Run to verify**

Run: `./tools/test_host.sh dense_loss` / `loss_head_sections` / `gpu_logistic_head` — Expected: all `ok`. Then `./tools/test_host.sh cost` to confirm existing `[cost]` cfgs still parse (22 shipped cfgs use `[cost]`; `parses_` tests).

- [ ] **Step 7: Commit**

```bash
git add src/darknet-core/src/dense_loss.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/runtime.rs
git commit -m "feat(layers): logistic and regression heads, cost scale, host dense-truth loss

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: `[jev_ce]` (soft-target cross-entropy over `groups x candidates`)

C++ `jev_ce_layer.cpp` (`Jev_laya`). Input `groups*candidates` logits per sample; output = `softmax(logits / T)` per group; with truth: loss per active group `-sum t_i ln p_i` (targets normalised per group, optional label smoothing, empty group = masked unless `ignore_empty=0`), `delta = (t - p) / T`, each sample's delta divided by its number of active groups when `normalize=1`; layer `cost = mean CE per active group`; backward adds `scale * delta`. Keys: `groups` (81), `candidates` (inferred as `inputs/groups`), `temperature`/`temp` (1), `normalize` (1), `ignore_empty` (1), `label_smoothing`/`smoothing` ([0,1)), `scale` (1), `layout = group | channel` (+ aliases). GPU: forward host round trip (C++ does the same), backward pass-through.

**Files:**
- Create: `src/darknet-core/src/jev_ce.rs`
- Modify: `lib.rs` (`pub(crate) mod jev_ce;`), `model.rs`, `dense_loss.rs`, `runtime.rs`
- Test: `jev_ce.rs`, `model.rs`, `dense_loss.rs`, `runtime.rs`

**Interfaces:**
- Produces: `JevCeSpec { pub groups: usize, pub candidates: usize, pub temperature: f32, pub normalize: bool, pub ignore_empty: bool, pub label_smoothing: f32, pub scale: f32, pub channel_major: bool }`, `LayerKind::JevCe(JevCeSpec)`; `jev_ce::jev_ce_probs(logits: &[f32], spec: &JevCeSpec, batch: usize) -> Vec<f32>`; `jev_ce::JevCeLoss { pub mean_ce: f32, pub active_groups: usize, pub correct: usize, pub delta: Vec<f32> }`; `jev_ce::jev_ce_loss(logits: &[f32], truth: &[f32], spec: &JevCeSpec, batch: usize) -> JevCeLoss`.
- Consumes: Task 10 `compute_dense_training_loss` (gets a `LayerKind::JevCe` arm).

- [ ] **Step 1: Write the failing tests.** `jev_ce.rs`:

```rust
//! jev_ce softmax + soft-target cross-entropy (tasks/rust/layers.md Task 11).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::JevCeSpec;

    fn spec(groups: usize, candidates: usize) -> JevCeSpec {
        JevCeSpec { groups, candidates, temperature: 1.0, normalize: true, ignore_empty: true, label_smoothing: 0.0, scale: 1.0, channel_major: false }
    }

    #[test]
    fn uniform_logits_give_ln3_loss_and_descent_delta() {
        let s = spec(1, 3);
        let l = jev_ce_loss(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &s, 1);
        assert!((l.mean_ce - 3f32.ln()).abs() < 1e-6);
        let third = 1.0 / 3.0;
        for (got, want) in l.delta.iter().zip([1.0 - third, -third, -third]) {
            assert!((got - want).abs() < 1e-6);
        }
        assert_eq!((l.active_groups, l.correct), (1, 1));
        let p = jev_ce_probs(&[0.0, 0.0, 0.0], &s, 1);
        assert!(p.iter().all(|v| (v - third).abs() < 1e-6));
    }

    #[test]
    fn empty_target_groups_are_masked_and_normalisation_counts_active_groups() {
        let s = spec(2, 3);
        // group 1 has an all-zero target -> ignored: delta 0, one active group
        let l = jev_ce_loss(&[0.0; 6], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &s, 1);
        assert_eq!(l.active_groups, 1);
        assert!(l.delta[3..].iter().all(|v| *v == 0.0));
        // two active groups with normalize: every delta is halved
        let l2 = jev_ce_loss(&[0.0; 6], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &s, 1);
        assert!((l2.delta[0] - (1.0 - 1.0 / 3.0) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_label_smoothing_and_channel_major_layout() {
        let mut s = spec(1, 3);
        s.label_smoothing = 0.3; // t = 0.7*[1,0,0] + 0.3/3 = [0.8, 0.1, 0.1]
        let l = jev_ce_loss(&[0.0; 3], &[1.0, 0.0, 0.0], &s, 1);
        assert!((l.delta[0] - (0.8 - 1.0 / 3.0)).abs() < 1e-6);
        let mut hot = spec(1, 2);
        hot.temperature = 2.0; // p = softmax([1, 0]); delta = (t - p) / 2
        let p_first = 1.0 / (1.0 + (-1.0f32).exp());
        let l = jev_ce_loss(&[2.0, 0.0], &[1.0, 0.0], &hot, 1);
        assert!((l.delta[0] - (1.0 - p_first) / 2.0).abs() < 1e-6);
        // channel-major: element (g, i) lives at i*groups + g. Group 0 = (idx0, idx2) = (1, 2).
        let mut cm = spec(2, 2);
        cm.channel_major = true;
        let p = jev_ce_probs(&[1.0, 0.0, 2.0, 0.0], &cm, 1);
        let e = 1.0f32.exp();
        assert!((p[0] - 1.0 / (1.0 + e)).abs() < 1e-6 && (p[2] - e / (1.0 + e)).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6 && (p[3] - 0.5).abs() < 1e-6); // group 1 = (0, 0)
    }

    #[test]
    fn delta_is_minus_the_gradient_of_the_summed_cross_entropy() {
        let mut s = spec(2, 3);
        s.normalize = false;
        let logits = [0.3f32, -0.2, 0.9, 1.1, 0.0, -0.7];
        let truth = [0.2f32, 0.5, 0.3, 1.0, 0.0, 0.0];
        let ce = |z: &[f32]| {
            let l = jev_ce_loss(z, &truth, &s, 1);
            l.mean_ce * l.active_groups as f32
        };
        let analytic = jev_ce_loss(&logits, &truth, &s, 1).delta;
        let h = 1e-3f32;
        for k in 0..6 {
            let (mut p, mut m) = (logits, logits);
            p[k] += h;
            m[k] -= h;
            let fd = (ce(&p) - ce(&m)) / (2.0 * h);
            assert!((fd + analytic[k]).abs() < 2e-3, "dz[{k}]: -dCE/dz = {} vs delta {}", -fd, analytic[k]);
        }
    }
}
```
`model.rs` test:

```rust
    #[test]
    fn jev_ce_keys_follow_cpp_defaults_and_validation() {
        let base = "[net]\nwidth=1\nheight=1\nchannels=12\n[connected]\noutput=12\n";
        let net = build(&format!("{base}[jev_ce]\ngroups=4\ntemp=2\nsmoothing=0.1\nlayout=cm\n")).unwrap();
        let LayerKind::JevCe(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.groups, s.candidates, s.temperature, s.label_smoothing, s.channel_major), (4, 3, 2.0, 0.1, true));
        assert!(s.normalize && s.ignore_empty && s.scale == 1.0);
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);
        assert!(build(&format!("{base}[jev_ce]\n")).is_err()); // default groups=81 does not divide 12
        assert!(build(&format!("{base}[jev_ce]\ngroups=5\n")).is_err());
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\ncandidates=4\n")).is_err()); // 4*4 != 12
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\nlabel_smoothing=1.0\n")).is_err());
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\nlayout=zigzag\n")).is_err());
    }
```
`dense_loss.rs` test (append):

```rust
    #[test]
    fn jev_ce_head_seed_is_scale_times_delta_and_cost_is_mean_ce() {
        let cfg = "[net]\nwidth=1\nheight=1\nchannels=3\n[connected]\noutput=3\nactivation=linear\n[jev_ce]\ngroups=1\nscale=2\n";
        let loss = run(cfg, |_| {}, vec![1.0, 0.0, 0.0], 1);
        let seed = loss.layer_output_gradient_seeds[1].as_ref().unwrap();
        let third = 1.0f32 / 3.0;
        assert!((seed.data[0] - 2.0 * (1.0 - third)).abs() < 1e-6);
        assert!((loss.loss - 3f32.ln()).abs() < 1e-6);
    }
```
`runtime.rs` GPU test (a convolution head so the helper can compare weight gradients; 6 outputs = 2 groups x 3 candidates):

```rust
    #[test]
    fn gpu_jev_ce_head_matches_cpu_forward() {
        assert_gpu_layer_chain_matches_cpu(
            "[net]\nwidth=1\nheight=1\nchannels=4\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\nstride=1\nactivation=linear\n\
             [jev_ce]\ngroups=2\ntemperature=1.5\n",
            1,
        );
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `./tools/test_host.sh jev_ce` — Expected: FAIL (compile errors: `JevCeSpec`/`jev_ce_loss` undefined).

- [ ] **Step 3: Model.** `model.rs`: `JevCeSpec` as in Interfaces (derive `Clone, Debug, PartialEq`), `LayerKind::JevCe(JevCeSpec)`, `name()` `"jev_ce"`, shape `Ok(current)` (add to the pass-through shape arm). Parse:

```rust
        "jev_ce" => {
            let inputs = current.element_count();
            let groups = section.get_usize("groups", 81)?;
            let line = section.line;
            if groups == 0 {
                return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: groups must be > 0")));
            }
            let mut candidates = section.get_usize("candidates", 0)?;
            if candidates == 0 {
                if inputs % groups != 0 {
                    return Err(Error::InvalidConfig(format!(
                        "[jev_ce] at line {line}: cannot infer candidates, {inputs} outputs are not divisible by groups={groups}"
                    )));
                }
                candidates = inputs / groups;
            }
            if groups * candidates != inputs {
                return Err(Error::InvalidConfig(format!(
                    "[jev_ce] at line {line}: inputs ({inputs}) must equal groups*candidates ({groups}*{candidates})"
                )));
            }
            let label_smoothing =
                section.get_f32("label_smoothing", section.get_f32("smoothing", 0.0)?)?;
            if !(0.0..1.0).contains(&label_smoothing) {
                return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: label_smoothing must be in [0,1)")));
            }
            let layout = section.get_str("layout", "group");
            let channel_major = match layout.as_str() {
                "channel" | "channel_major" | "cm" | "chw" => true,
                "group" | "group_major" | "gm" => section.get_usize("channel_major", 0)? != 0,
                other => {
                    return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: unknown layout '{other}' (expected group or channel)")));
                }
            };
            LayerKind::JevCe(JevCeSpec {
                groups,
                candidates,
                temperature: section.get_f32("temperature", section.get_f32("temp", 1.0)?)?,
                normalize: section.get_usize("normalize", 1)? != 0,
                ignore_empty: section.get_usize("ignore_empty", 1)? != 0,
                label_smoothing,
                scale: section.get_f32("scale", 1.0)?,
                channel_major,
            })
        }
```

- [ ] **Step 4: `jev_ce.rs` math** (above the tests; `pub(crate) mod jev_ce;`):

```rust
use crate::model::JevCeSpec;

fn at(spec: &JevCeSpec, base: usize, g: usize, i: usize) -> usize {
    if spec.channel_major { base + i * spec.groups + g } else { base + g * spec.candidates + i }
}

/// `softmax(logits / T)` per group (layer output; also the inference output).
pub(crate) fn jev_ce_probs(logits: &[f32], spec: &JevCeSpec, batch: usize) -> Vec<f32> {
    let temp = spec.temperature.max(1e-6);
    let mut probs = vec![0.0f32; logits.len()];
    for b in 0..batch {
        let base = b * spec.groups * spec.candidates;
        for g in 0..spec.groups {
            let max = (0..spec.candidates).map(|i| logits[at(spec, base, g, i)]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..spec.candidates {
                let e = ((logits[at(spec, base, g, i)] - max) / temp).exp();
                probs[at(spec, base, g, i)] = e;
                sum += e;
            }
            for i in 0..spec.candidates {
                probs[at(spec, base, g, i)] /= sum;
            }
        }
    }
    probs
}

pub(crate) struct JevCeLoss {
    pub mean_ce: f32,
    pub active_groups: usize,
    pub correct: usize,
    /// Darknet descent direction `(t - p) / T`, normalised per sample when `spec.normalize`.
    pub delta: Vec<f32>,
}

pub(crate) fn jev_ce_loss(logits: &[f32], truth: &[f32], spec: &JevCeSpec, batch: usize) -> JevCeLoss {
    let temp = spec.temperature.max(1e-6);
    let eps = if spec.label_smoothing > 0.0 { spec.label_smoothing.min(0.999) } else { 0.0 };
    let probs = jev_ce_probs(logits, spec, batch);
    let mut delta = vec![0.0f32; logits.len()];
    let (mut ce_sum, mut active, mut correct) = (0.0f64, 0usize, 0usize);
    for b in 0..batch {
        let base = b * spec.groups * spec.candidates;
        let mut sample_active = 0usize;
        for g in 0..spec.groups {
            let target_sum: f32 = (0..spec.candidates).map(|i| truth[at(spec, base, g, i)]).sum();
            let empty = target_sum <= 1e-8;
            if !empty || !spec.ignore_empty {
                sample_active += 1;
            }
            if empty && spec.ignore_empty {
                continue; // masked group: no loss, no gradient
            }
            let (mut arg_p, mut arg_t, mut loss) = (0usize, 0usize, 0.0f32);
            for i in 1..spec.candidates {
                if probs[at(spec, base, g, i)] > probs[at(spec, base, g, arg_p)] {
                    arg_p = i;
                }
                if truth[at(spec, base, g, i)] > truth[at(spec, base, g, arg_t)] {
                    arg_t = i;
                }
            }
            for i in 0..spec.candidates {
                let idx = at(spec, base, g, i);
                let mut t = if empty { 1.0 / spec.candidates as f32 } else { truth[idx] / target_sum };
                if eps > 0.0 {
                    t = (1.0 - eps) * t + eps / spec.candidates as f32;
                }
                loss += -t * probs[idx].max(1e-12).ln();
                delta[idx] = (t - probs[idx]) / temp;
            }
            ce_sum += loss as f64;
            active += 1;
            correct += (arg_p == arg_t) as usize;
        }
        if spec.normalize {
            let inv = 1.0 / sample_active.max(1) as f32;
            for v in &mut delta[base..base + spec.groups * spec.candidates] {
                *v *= inv;
            }
        }
    }
    JevCeLoss {
        mean_ce: if active > 0 { (ce_sum / active as f64) as f32 } else { 0.0 },
        active_groups: active,
        correct,
        delta,
    }
}
```

- [ ] **Step 5: Wire runtime + dense loss.** `dense_loss.rs`: new arm in the `match &layer.kind` of `compute_dense_training_loss`:

```rust
            LayerKind::JevCe(spec) => {
                let result = crate::jev_ce::jev_ce_loss(&prev.data, &t.data, spec, prev.shape.batch);
                let mut delta = Tensor::from_data(prev.shape, result.delta)?;
                for v in &mut delta.data {
                    *v *= spec.scale; // C++ backward: axpy(scale, delta)
                }
                (delta, result.mean_ce)
            }
```
CPU forward arm: `(LayerKind::JevCe(spec), LayerParams::None) => (Tensor::from_data(current_input.shape, crate::jev_ce::jev_ce_probs(&current_input.data, spec, current_input.shape.batch))?, None),`. CPU backward: add `| (LayerKind::JevCe(_), LayerParams::None)` to the pass-through backward arm. GPU forward arm (host round trip, like C++):

```rust
                (LayerKind::JevCe(spec), LayerParams::None) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download jev_ce input", err))?;
                    let probs = crate::jev_ce::jev_ce_probs(&host_input.data, spec, layer.input_shape.batch);
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &probs)
                        .map_err(|err| backend_error("upload jev_ce output", err))?
                }
```
GPU backward: add `| (LayerKind::JevCe(_), Some(LayerParams::None))` to the pass-through list; add `| LayerKind::JevCe(_)` to the `forward_compact` GPU fallback list.

- [ ] **Step 6: Run to verify**

Run: `./tools/test_host.sh jev_ce` — Expected: `jev_ce::tests::*` (4), `jev_ce_keys_follow_cpp_defaults_and_validation`, `jev_ce_head_seed_is_scale_times_delta_and_cost_is_mean_ce`, `gpu_jev_ce_head_matches_cpu_forward` all `ok`. Then `./tools/test_host.sh` (whole host suite, several minutes) — Expected: `test result: ok` for `darknet-core` and `darknet-cli` (the whole Phase 0 is now in).

- [ ] **Step 7: Commit**

```bash
git add src/darknet-core/src/jev_ce.rs src/darknet-core/src/lib.rs src/darknet-core/src/model.rs src/darknet-core/src/dense_loss.rs src/darknet-core/src/runtime.rs
git commit -m "feat(layers): jev_ce soft-target cross-entropy head

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Bookkeeping and Phase 0 exit check

**Files:** Modify `tasks/rust/todo.md`.

- [ ] **Step 1:** Run the whole host suite once: `./tools/test_host.sh` — Expected: `test result: ok.` for both packages. Run the touched oxide suites one at a time: `./tools/test_oxide.sh --test oxide_parity`, `--test oxide_layers_extra`, `--test oxide_cfg_shapes`, `--test oxide_half`, `--test oxide_batchnorm_adam` — Expected: all `ok`.
- [ ] **Step 2:** Add to `tasks/rust/todo.md` (under "Not implemented by design", replacing the "Activations without kernels" bullet) a short block listing what Phase 0 delivered (ten activations + GELU kernel, unknown-activation error, maxpool keys/depth, batchnorm save, weighted shortcut, channel_slice, channel_shuffle, deconvolutional, logistic/regression/cost-scale/jev_ce + `dense_loss`) and a pointer `see tasks/rust/layers.md for the remaining phases`. Record any C++ parity check that could not run (see "Verification against C++").
- [ ] **Step 3: Commit**

```bash
git add tasks/rust/todo.md
git commit -m "docs(tasks): record layer-port Phase 0 status

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

# Verification against C++ (applies to every task; read before claiming "matches C++")

What the repo can and cannot do today, stated honestly:

1. **Rust side.** `darknet-rs detect-image` with `DARKNET_LAYER_DUMP_DIR=<dir>` writes `layers.tsv` + `layer_NNN.f32` (raw f32, one file per layer, inference only). `tools/compare_layer_dumps.py <left> <right> --label ..` compares two dump dirs layer by layer (dims must match, prints `max_abs`/`max_rel`).
2. **C++ side does not exist in the repo any more.** The dump hook was added to `src-lib/network_kernels.cu` (`forward_network_gpu`, branch `rust` history only) by commit `96f662f0`, and the whole C++ tree was removed by `2514cf53`. It is NOT on `Jev_laya`, `raptor`, `fog`, `jepa`, `depth_3d`, ... The hook only runs in inference (`not state.train`) on a CUDA build.
3. **Therefore, for a layer that lives only on another branch:** (a) `git worktree add ../darknet-<branch> <branch>`; (b) `git -C ../darknet-<branch> apply <(git show 96f662f0 -- src-lib/network_kernels.cu)` (expect to resolve a small conflict); (c) build that C++ (user installs toolchains; ONE build at a time, `-j 2`, memory-capped, per `build_one_at_a_time_low_memory`); (d) run the C++ `darknet detector test` on a tiny cfg with the new layer plus a fixed image with `DARKNET_LAYER_DUMP_DIR=/tmp/cpp`; (e) run the Rust `detect-image` on the same cfg/weights/image with `DARKNET_LAYER_DUMP_DIR=/tmp/rust` and `tools/compare_layer_dumps.py /tmp/cpp /tmp/rust`. Expect `max_abs` around 1e-5 on f32 layers; the C++ CUDA reference runs `cudnn_half=1`, which alone explains up to ~5.7e-3 after convolutions (`tasks/rust/todo.md`), so compare the new layer's output against its own input rather than only the last layer.
4. **Weights order** is the strongest cross-check and needs no C++ forward: generate weights with the C++ build (`darknet detector train` for one iteration, or any saved `.weights`), run `darknet-rs weights-inspect <cfg> <weights>`: it must report `trailing_bytes 0` and the per-layer array names/lengths must equal the plan. For layers where Rust is the first writer (swin), do the reverse: `save_weights` in Rust, load in the C++ branch build, dump, compare.
5. **Backward and losses cannot be compared through dumps** (inference only). They are validated by: central finite differences of `L = sum(seed * forward)` in the pure-math module (every task above), CPU-vs-GPU parity on `FakeBackend`/oxide, and hand-derived numbers in tests. C++ gradient shortcuts are reproduced under `DARKNET_GRAD_MODE=cpp` so a future same-init training comparison (needs explicit user OK per the no-debug-training-runs rule) can use them.
6. **When no C++ build of the owning branch is available** (the default in this environment): do the by-construction checks (this plan's formulas were copied from the C++ source with the file named in each task), run all Rust tests, and write one line per skipped parity check under the task in `tasks/rust/todo.md` ("C++ dump parity: NOT RUN, no build of `<branch>`"). Never write "matches C++" without a dump or an inspected `.weights`.

---

# Phase sections (NOT bite-sized — each must be expanded into its own plan before starting)

Conventions for every phase below: follow the Layer Port Contract; pure math in its own module; GPU first via host round trip, kernels second; each layer gets parse/shape/weights/CPU/FD/GPU tests; weights order below is quoted from the C++ source named, `[verified]` = re-read for this plan, `[report]` = from the research report and not re-read.

Cross-cutting prerequisites: (1) container layers (moe, eml, recursive_block, clifford, wmhf, depth_anything) reuse convolution parameter/layout helpers where possible; the GPU device-conv cache prerequisite for MoE is now in place. `GpuExecutableNetwork::device_conv_params` is a per-owner row of slots (`Vec<Vec<Option<GpuConvolutionLayerParams>>>`), ordinary convolutions use slot 0, and `cached_convolution_params_at` / `cached_convolution_params_raw_at` accept explicit slots. `warm_device_conv_params` warms every `LayerParams::Moe` expert, and host-updated Moe experts invalidate their row before the next warm/upload. The FakeBackend test `container_convolution_cache_is_keyed_by_owner_and_expert_slot` verifies distinct expert-slot params and rewarm. The generic convolution optimizer/BN-refresh path still handles top-level owners only; separate SGD and Adam paths update MoE routers and experts, including BN scales, on-device using owner/expert slots for convolution parameters. Router parameter values sync to host after each step for the current host-parameter router forward; gradients and momentum stay on-device. No-BN MoE Adam updates also run on-device; DNOPT3 persists device moments, and DNOPT2 host moments migrate on first no-BN device use. BN expert backward uses backend ops; BN SGD updates weights, beta, and scale on-device with momentum in device buffers, while BN Adam also persists scale moments in DNOPT4. The `moe` Oxide arm computes pool, routing, expert convolutions, and gate blending on-device. No-BN backward computes router/expert and input gradients through backend ops; gates and pooled/expert caches stay device-resident. Router/expert gradients accumulate on-device across backward calls. SGD and Adam updates for no-BN routers/experts run on-device without gradient transfers; Adam moments persist in DNOPT3, with DNOPT2 host moments migrated on first device update. Router parameter values sync to host after each step, while deferred synchronization refreshes no-BN expert weights. BN expert backward uses backend ops and accumulates gradients on-device; BN expert SGD/Adam updates are device-native, with SGD momentum in device buffers and Adam scale moments in DNOPT4. The real-device `oxide_runtime_chain::moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu` test checks two backward calls accumulated before the first deferred SGD step, then a second step, against CPU; `oxide_runtime_chain::moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients` checks Adam updates against CPU-derived gradients; `oxide_runtime_chain::moe_batchnorm_expert_device_backward_matches_cpu_on_oxide` checks BN backward/SGD and upstream-conv parity; `oxide_runtime_chain::moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients` checks BN Adam updates and deferred sync. v69 integrated `eml_convolutional` parsing, paired-convolution CPU forward/backward, C++ weight order, and SGD/Adam. v70 runs both branch convolutions and the combine through the f32 backend on GPU forward, using separate cache slots; CPU branch caches still feed the explicit backward/optimizer fallback. v71 adds native no-BN Oxide backward and host parameter-gradient accumulation; BN EML backward retains CPU fallback, and optimizer updates remain host-side. C++ batch-stat/EMA BN training behavior remains open. (2) A shared oxide kernel set for ~12 layers: LayerNorm fwd/bwd, row softmax (exists: `softmax_f32`), batched/strided GEMM (`gemm.rs` already takes batch strides), GELU (Task 3), residual add, bias+activation epilogues. Build it ONCE in Phase 3A. (3) Loss/seed plumbing: `dense_loss.rs` (Task 10) for dense truth; detection heads extend `ForwardResult.yolo_heads`-style capture + their own `compute_*_training_loss`.

## Phase 1b — rest of Tier 1 (expand first; all medium)

- **segmentation** (alias `segmenter`; `raptor`/`v6-dev-seg`, `src-seg/segmentation_layer.cpp/.hpp`, `seg_gate.cpp/.hpp`, `seg_gate_kernels.cu`; ~657 LOC; difficulty M). Keys `classes`(2), `max_delta`, `delta_normalizer`, `cls_normalizer`, `loss`, `lambda_ce`(1), `ce_normalizer`(1), `gate_enabled`, `gate_t_correct`(.9), `gate_warmup_iters`(1000), `gate_ramp_iters`(2000), `gate_max_compensation`(10). Math: `classes==1` sigmoid + logistic cross-entropy (the Task 10 `logistic` math per pixel), else per-pixel channel softmax (temperature) + softmax CE. Weights none. Needs: a dense mask truth (pixel classes) and a segmentation label loader in the data pipeline (not read yet), `Seg_resnet_1head`/`yolov4-tiny-seg*`/`unet_*` cfgs only if the user wants them (Open Question 1). The gate (`seg_gate*`) was not read; expand-plan step 1 is to read it and decide whether to port or reject `gate_enabled=1`. Kernels: none required (host loss). Depends on Task 10.
- **omd_loss** (`Jev_laya` `omd_loss_layer.cpp`, 753 LOC; M). Pass-through output; in training builds a supervised pair set (keys `classes`, `truth_size`(6), `lambda`(.03), `margin`(.5), `small_boost`, `max_samples`(128), `pool`, `eps`, `max_delta`(10)) and ADDS metric-learning deltas to the PREVIOUS layer's delta (so it needs the previous layer's features and an extra seed on layer `idx-1`, unlike Task 10 heads). No weights. Tests: hand-built 2-sample pair sets + finite differences of the metric loss. Source not read in detail yet.
- **LAYA small pieces** (`laya_marker_gather` S, `laya_scorer` S, `laya_act_head` S/M, `laya_loss` M; `Jev_laya`). Scorer = LayerNorm -> Linear(H,H) -> GELU -> Linear(H,1), frozen weights (`load_laya_scorer_weights`, `weights.cpp:1705`); head weights `load_laya_act_head_weights:1736`; loss `ce | rlcd`. **`laya_scorer`/`laya_marker_gather`/`laya_act_head`/`laya_loss`/`laya_encoder`/`laya_head` all done 2026-10-04/05** (`laya_head.rs` + `LayaHeadSpec`, 8 tests: bias LayerNorm, ReLU FFN blocks, type_emb, per-sample qtype, blob round trip, CPU runtime forward/parse). The full LAYA family is complete. **Blocked on Open Question 2**: user memory says the LAYA C++ port (inference `c14aed31`, training `3f903fca`/`b1299940`) lives on another branch and AdamW is required for its training; do not start until the user confirms LAYA belongs in Rust. Needs the shared LayerNorm/GEMM set (Phase 3A) and the exact (erf) vs tanh GELU question settled by reading the C++ scorer.
- **Deferred Tier 0 leftovers** (do only on demand): (a) `maxpool_zero_nonmax` needs no separate GPU kernel: the existing argmax-only maxpool backward operation inherently zeros non-argmax deltas; real Oxide forward/backward/SGD parity is covered. (b) Rust-only legacy layers `gru` (backward + GPU), `local` (GPU), `batchnorm` (GPU training), `crop` (GPU training): not in any C++ branch's `ELayerType` (verified), 1-4 `cfg/unsupported` files use them, no reference to validate against. (c) `region` (YOLOv2; 12 cfgs under `cfg/unsupported`, `tiny-yolo_xnor.cfg`), `detection` (v1), `cost` classifier training and `contrastive`: parsed as identity pass-through; their loss/decode are unimplemented (cost now has a host loss via Task 10; region/detection/contrastive need `region_layer.cpp` decode + loss, `contrastive_layer.cpp` read first). (d) `normalize_channels*` activations are conv-only and geometry-dependent (`activate_array_normalize_channels*` in `activations.cpp`; ReLU/softmax over channels per pixel, `eps=1e-4`). v65 added f32 host/Oxide primitives and v66 integrated CPU/f32-GPU convolution forward/backward, with exact default adjoints and source-derived CUDA shortcut gradients under `DARKNET_GRAD_MODE=cpp`. The parser rejects non-convolution uses; f16 inference is explicitly unsupported. No compiled-C++ golden is available. (e) shortcut optimizer/parameter updates remain host-side.

## Phase 2 — detection heads, MoE/EML/DCNv3, OBB

**2A. Anchor-free heads** (`Jev_laya` `src-lib/modern_yolo_layer.cpp` 1254 LOC + 76-line wrappers `yolox_layer`, `ppyoloe_layer`, `yolonas_layer`; `centernet_layer.cpp` 857; L / M). No weights, no kernels: host decode/assign/loss. Keys `classes`, `max`, `jitter`, `resize`, `reg_max`(16); centernet adds `center_min_radius`, `peak_nms`, `anisotropic_gaussian`, `small_*`, `focal_alpha/beta`, `hm/wh/off_normalizer`, `max_delta`, `delta_normalizer` and needs `classes + 4` filters before it. Math: YOLOX SimOTA assignment + IoU/obj/cls loss; PP-YOLOE/YOLO-NAS distribution-focal-loss box regression with task-aligned assignment; CenterNet focal heatmap + wh + offset. Rust needs: new `LayerKind`s, a decode path next to `decode_yolo_detections` (`detect.rs` only understands yolo heads), evaluation (`evaluate.rs` mAP) for the new outputs, `compute_*_training_loss` modules. **Port the LegoGears nested-object fix** that lives in this file (`class_quality`, smallest-area-wins, min-cell-size gate; user memory `legogears_nested_object_fix`) together with the head. Tests: hand-built targets for assignment, finite differences of each loss term, decode round trip. Order: yolox, then ppyoloe/yolonas (shared DFL), then centernet.

**2B. moe_convolutional** (692 + 439 kernel LOC; M). Experts are convolutions; router = global-average-pool -> `Wr*pooled + br`, top-k, temperature softmax gates; `out = sum_e gate_e * expert_e(x)`; load-balance loss `alpha * max(0, E*sum(mean_gate^2) - 1)`. Keys `experts`, `top_k`, `load_balance`, `temperature`. **Weights `[verified]`** (`save_moe_convolutional_weights`): for each expert `e` in `0..total`: the full `save_convolutional_weights` block (biases, [BN scales, rolling mean, rolling variance], weights); then router `biases` (`nbiases`), router `weights` (`nweights`). **CPU done 2026-10-05** (commit `cdf68d00`): `src/darknet-core/src/moe_conv.rs` + `MoeSpec` + `LayerKind::Moe` + runtime `LayerParams::Moe`/`MoeTrainBuffers` (CPU fwd/bwd/update, hand-computed blend test, FD gradient tests). Container-conv reuse turned out **not** to be needed for the CPU path - the experts reuse the existing conv forward/backward directly. For GPU, the `(owner, slot)` cache prerequisite and no-BN Oxide forward/backend backward are implemented. Router/expert gradients and SGD momentum accumulate on-device, and both parameter blocks update there; router parameter values sync to host after each step for forward. No-BN Adam updates also run on-device and persist moments in DNOPT3 (DNOPT2 host moments migrate on first device update); BN expert backward uses backend ops; BN SGD updates weight, beta, and scale on-device with momentum in device buffers; BN Adam also persists scale moments in DNOPT4. **eml_convolutional** (inside `convolutional_layer.cpp:1210-2104`, no cfg users; M): two parallel convs `x`,`y`; `out = scale*(exp(clamp(x,+-c)) - ln(softplus(y)+eps))` (+ input if `residual`); keys `eml_clamp`(4), `eml_eps`(1e-6), `eml_scale`(.1), `residual`(1); activation `eml` = `e^x - ln1p(e^x) - (1 - ln2)` with threshold 2 (`activations.cpp:205`); save order **verified 2026-10-05** (`weights.cpp:775-793` save, `:2000-2015` load): the `EML_CONV` case writes/reads `save_convolutional_weights(l.input_layer)` then `save_convolutional_weights(l.self_layer)`, i.e. two complete conv blocks in that order (biases, [BN scales, mean, var], weights) - the two parallel convs are named `input_layer` and `self_layer`. **dcnv3** (247 + 479; M): conv keys + `offset_scale`; per group `groups*size^2*2` offsets and `groups*size^2` masks (softmax over `K`). **Weights `[verified]`** (`save_dcnv3_weights`): conv `biases(n)`, [BN scales, mean, var], `weights`, then `offset_weights` (`offset_filters*c*k*k`), `offset_biases(offset_filters)`, `mask_weights` (`mask_filters*c*k*k`), `mask_biases(mask_filters)` with `offset_filters = groups*k*k*2`, `mask_filters = groups*k*k`. Kernels: bilinear sampling im2col (reuse `im2col.rs`) for dcnv3.

**2C. OBB family** (`raptor`, `raptor-dev`, `raptor-fp4`, `origin-github/obb*`; XL, scope-gated by Open Question 4). `obb_yolo` (`yolo_layer.cpp` 1851 lines on raptor, `darknet_cfg_obb.cpp`, `obb_loss.cpp`, `box_obb.cpp`, `obb_covariance_loss_kernels.cu`, `detector_map_obb.cpp`, `calcanchors_obb.cpp`; coords 6, truth size 7 with a front point; needs `[net] obb=1`/`rotate_obb`), `obb_anchorless` (840), `yolo_2stage` (549, after obb_yolo), `obb_refine` (1280, depends on obb_yolo, weights layout not read). Touches data loading (rotated boxes), augmentation, NMS and mAP (`evaluate.rs`), not just the layer. v75 adds legacy front-point rectangle vertices/IoU; v76 adds CARPN quad IoU; v77 adds typed OBB dispatch/NMS; v78 adds head decode; v79 adds label parsing; v80 adds source-derived label rotation; v81 adds cfg/runtime decode; v82 adds direct-resize OBB batch loading; v83 adds standalone OBB AP; v84 adds validation evaluation; v85 adds CPU `detector map`; v86 parses/defaults `nms_kind=obb` and rejects unsupported alternatives; v87 adds a standalone horizontal-mirror label transform; v88 applies caller-selected per-image flips to OBB batch tensors and labels together; v89 adds source-derived direction-free anchor size matching and best-anchor/cell assignment primitives; v90 adds seeded config-driven flips through an OBB-specific batch loader; v91 adds source-layout seven-float truth-row packing with absent-front placeholders and zero padding; v92 enumerates the primary best-anchor and strict-threshold additional per-head positives; v93 selects best-truth and ignore/positive gates from precomputed source-style scores; v94 adds standalone CARPN polygon mAP for normalized quad detections and front-point labels; v96 adds source-style per-cell objectness deltas from the v93 match gates, including ignore smoothing and positive-target override; v97 adds source CIoU-OBB match-quality scoring for decoded geometries; v98 decodes one cell's post-YOLO channels for the source CIoU/default score path; v99 composes per-truth scores with source class-gated ignore and positive-threshold selection for one cell; v100 adds source-style class-channel deltas for positive cells; v101 adds standalone covariance-W2 loss terms and `exp(-loss)` quality scoring for decoded geometries; v102 adds one-cell raw covariance-branch score decoding; v103 adds the source QBBV heatmap and sampled vIoU metric; v104 adds the source QBBV blend and covariance-branch integration; v105 adds the source legacy angular fallback scorer with one-cell decode/blend; v106 adds decoded COBB loss/quality and QBBV blending; v107 adds raw-channel COBB match decoding and source branch gates; v108 adds one-cell covariance-first/COBB/angular score dispatch; v109 composes branch scores with per-truth cell gates; v110 enumerates primary/additional positives across batch slots and multiple heads; v111 maps caller-selected head/cell predictions to batch-specific truths and tagged match gates; v112 traverses a complete batch-major head tensor in source cell/anchor order with callback-bounded memory; v113 emits source objectness deltas for batch/head cells, including ignore and positive overrides; v114 writes them to the objectness channel of one head's delta tensor while preserving other entries; v115 applies source best/additional positive-anchor objectness overrides with class mapping and smooth-mode preservation; v116 writes source class-channel deltas at those positives with existing-target behavior; v117 adds source finite-difference COBB/legacy box deltas and positive-anchor writeback; v118 ports source covariance-W2 analytic geometry gradients; v119 routes those gradients into positive-anchor box-delta writeback; v120 returns `avg_cat` contributions from best/additional class assignments; v121 applies mapped class multipliers to box-delta normalization; v122 applies truth-threshold cell class and box deltas after the all-cell objectness pass; v123 composes ordered delta passes for one complete head; v124 dispatches across multiple head tensors; v125 adds a standalone source center-front-anchor point decoder; v126 routes optional anchor offsets through head matching and box deltas; v127 aggregates per-head `avg_cat`, threshold-positive, and box-positive stats across multi-head dispatch; v128 reports source `net.total_bbox` counts across threshold-cell and best/additional-anchor assignments; v129 adds standalone source front-point CIoU deltas, including raw-channel grid scaling and optional anchor center-chain contributions; v130 writes these deltas to best/additional positive anchor channels with class scaling; v131 writes them to truth-threshold-positive cells with class scaling; v132 composes both front writeback passes in one-head source order; v133 dispatches that composition across multiple heads; v134 aggregates source-normalized front-loss sums/counts across truth-threshold and best/additional passes; v135 parses source `obb_anchors` pair/quadruple forms and retains directional priors in the OBB spec; v136 parses `front_loss_normalizer` with the source alias and default. Still open: remaining cfg/training option wiring, inference/network decode integration, network-loop/backprop integration, remaining loss/metric aggregation and training-loop wiring, crop/mosaic transforms, alternate NMS-kind implementations, training `-map` cadence, CARPN network decoding/validation integration, and the remaining family layers. No compiled-C++ parity is established. Expand the family incrementally and keep unsupported semantics explicit.

## Phase 3 — attention family and the rest of the conv family

**3A. Shared kernels + transformer layers** (all L). First deliverable: oxide LayerNorm fwd/bwd, batched-GEMM wrapper, row-softmax backward, residual/bias epilogues, exposed as `F32TensorOpsBackend` ops with host defaults (so `FakeBackend` tests work), with GELU from Task 3. Then:
- **vit** (2597 + 1557): patch embed -> `+ pos_embed` -> pre-norm MHSA (`QKV`, `wo`) -> pre-norm FFN (`GELU`) with a 2x2 Sinkhorn "mHC" residual mixer (`scales[12]`, present only in newer weights: `load_vit_weights(l, fp, has_mhc_scales)`). **Reference Rust path implemented** in `vit.rs`/`model.rs`/`runtime.rs`: validated cfg parsing + shape inference, exact analytical CPU backward, seeded init, SGD updates, header-aware load/save, and GPU f32 host fallback. `MEM=3G ./tools/test_host.sh vit` passes 23 focused tests; FakeBackend covers a preceding conv chain. **Weights `[verified]`** (`save_vit_weights`): `vit_patch_embed (out_c*patch_dim)`, `vit_patch_bias (out_c)`, `biases (nbiases = QKV bias)`, `ln1_gamma`, `ln1_beta`, `weights (nweights = QKV)`, `wo (out_c*out_c)`, `wo_bias`, `ln2_gamma`, `ln2_beta`, `ffn_w1 (out_c*mlp_dim)`, `ffn_b1 (mlp_dim)`, `ffn_w2 (out_c*mlp_dim)`, `ffn_b2 (out_c)`, `pos_embed (out_h*out_w*out_c)`, `scales (12)`; Rust gates that final block on header version. Still open: `flash_attention`/`vc_attention` fast paths and native oxide kernels; C++ dump parity NOT RUN.
- **transformer / swin** (`jepa` renames it `SWIN`; 2624 + 1556): windowed MHSA with cyclic shift mask `-100`, same mHC mixer. **Verified gap: no `weights.cpp` case exists on `Jev_laya` or `jepa`**, so the Rust `.weights` order is OURS to define (propose: the `vit` order with window/shift parameters implied by the cfg, document it in `Layer::weights()`), and a C++ `.weights` cannot exist yet. User decision: YES, Rust defines the vit-like order; no question is blocking implementation.
- **tucker_attention** (823 + 2488; low-rank Q/K/V/O): weights are a flat arena `biases[nbiases]`, `weights[nweights]` `[report]` (`save_tucker_attention_weights`, `weights.cpp:220`). **mambavision** (1271 + 783; selective scan): order `[report]`: `weights, biases, conv_x, conv_x_bias, conv_z, conv_z_bias, x_proj, dt_proj, dt_bias, A_log, D, out_proj, out_bias, res_proj (only if c != n), ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, ffn_w1, ffn_b1, ffn_w2, ffn_b2`. **kimi_kda** (906 + 687; delta-rule attention + MoE; `biases`, `weights` flat `[report]`) and `minimax_sparse_attention` (auxiliary `msa_*` keys). Tests for every one: FD of the pure-math module, `FakeBackend` parity, tiny hand-computed attention (softmax of a 2x2 score matrix), mask/shift tests for swin.

**3B. Conv family** (all L):
- **deform_conv** (DCNv1/v2; 1654 + 1087): offset conv `[2K]` + optional mask conv `[K]` (sigmoid), bilinear im2col; keys `use_mask`(1), `flipped`, `dot`, `angle`, `grad_centr`, `reverse`, `coordconv`; weights = conv block + `offset_weights/biases` (+ `mask_weights/biases`) `[report]`. **dcnv4** (939 + 1597; `LegoGears_deform` cfg) and **dsconv** (dynamic snake conv; 901 + 714; `extend_scope`, `morph`, templates) and **graph_conv** (1966 + 1619; `graph_edge_mode`, `graph_use_self`; weights include `graph_self_weights`, `graph_edge_kernel/biases`) and **wmhf** (1375 + 548; Haar DWT + EMA scan + IWT, sub-convs; weights order not read) — each gets its own plan after reading its `save_*_weights`.
- **clifford** (3554 LOC total, XL) and **recursive_block** (3417, XL; body sections, `ouroboros` 0/1/2, conv-LoRA, lattice projection). **Unverified C++ suspicion:** `recursive_block` saves only the body weights for `ouroboros > 0` (`weights.cpp:859-876` shows loops over body layers: conv / connected); confirm before defining the Rust layout. Both are containers: need nested-layer parsing (body sections are consumed by a pre-pass in `darknet_cfg.cpp`), which has no Rust analogue yet.

## Phase 4 — hyperbolic / depth / LAYA backbone / DETR

- **hyperbolic_foveation** (2659 LOC; L/XL; learnable radial warp, Newton inverse; also touches detect/mAP reprojection) + **foveation_film** (402; M; needs the former; weights `w_embed, b_embed, w_gamma, b_gamma, w_beta, b_beta` `[report]`).
- **depth_anything** (1056 + 536; L; frozen DINOv2 ViT-S/14, exact erf GELU, bicubic/bilinear pos-embed interpolation; loader `load_depth_anything_weights` at `weights.cpp:1433`) + **dpt_head** (664 + 309; L; reassemble + fusion neck + depth head) + **yolo_3d** (use the `depth_3d` branch version: 2989 + 470 + 708; XL; projected 8-vertex cuboids; per user memory the depth-focal-loss design is settled in `yolo3d_depth_focal_loss_design`). `tools/depth_anything/README.md` documents the existing Depth-Anything work; read it before porting.
- **laya_encoder / laya_head** (L): ModernBERT-style encoder (`tok_embed, embed_norm`, per layer `attn_norm (i>0), Wqkv, Wo, mlp_norm, Wi, Wo_mlp`, `final_norm`; RoPE full/local theta; GeGLU) and head (two pre-norm TransformerEncoderLayers with ReLU FFN + `type_emb[3*H]`) `[report]`. Gated by Open Question 2.
- **detr_decoder** (1428 + 773; XL; Hungarian matching on the host). Last: user memory says DETR is still broken in C++; port only after the C++ reference works.

---

# Out of scope (not layers; do not plan them as layers)

`fog`'s `[net]` augmentation key; `raptor-dev` augmentation sections `[Brightness] [GaussianBlur] [GaussianNoise] [PepperNoise] [augmatch]` (see `tasks/rust/augmentation-audit.md`); `[net]` keys `obb`, `rotate_obb`, `tf32`, `cudnn_bf16`, `random_init_weights`, `thread_val` (except where a phase above needs them); `raptor-fp4`'s `fp8`/`fp4` keys and `src-fp8` conv path (see `tasks/rust/fp8_fp4.md`); the `flash_attention`/`vc_attention` flags on `[vit]`/`[tucker_attention]` (Open Question 3); `src-jepa`'s standalone I-JEPA trainer.

# Open questions for the user — RESOLVED 2026-10-04

The user answered **all of them yes, and asked for everything to be ported to
Rust** ("q2, a3, a7: all yes and port everything in rust").  The decisions are
recorded here so no later step re-opens them; the *consequences* are folded into
the phase tables above.

1. **Which cfgs to ship?** Port everything. Shipping the `unet_*` /
   `Seg_resnet_1head` / `deconvolutional` cfgs under `cfg/` is now in scope
   alongside the layer itself (they were previously gated on this question).
2. **Does LAYA belong in Rust at all? YES.** Unblocks the Phase 1b `laya_*`
   family (`laya_marker_gather`, `laya_scorer`, `laya_act_head`, `laya_loss`)
   and Phase 4's `laya_encoder` / `laya_head`.  The plan's note that LAYA
   training needs AdamW on the C++ side becomes a Rust requirement, not a
   blocker.
3. **Port the `flash_attention` / `vc_attention` variants of `vit` / `tucker`?
   YES.** They are no longer deferred; the shared attention kernel set in the
   Phase 3A prerequisites must cover the fast paths, not only reference
   attention.
4. **OBB scope: port the whole family.** `obb_yolo`, `obb_anchorless`,
   `yolo_2stage` and `obb_refine` are all in scope, together with the rotated-box
   data loading, augmentation, NMS and mAP work they imply.
5. **Where do dense-truth losses plug in? Port a trainer.** `logistic`,
   `regression`, `cost`, `jev_ce`, `segmentation`, `omd_loss` and `contrastive`
   currently have a library API only; a CLI trainer (and a segmenter command) is
   now part of the scope rather than a follow-up.
6. **Rust-only legacy layers: keep them as extensions.** `batchnorm` stays
   saveable; `gru`, `local`, `crop`, `detection` and `region` stay supported
   (with `region` / `detection` decode and loss implemented properly, since
   22 corpus cfgs use them).
7. **swin `.weights` order: Rust defines it.** No C++ writer exists, so the
   vit-like order proposed in this document becomes the reference and is
   documented in `Layer::weights()`.

## Consequences for the phase tables

Everything above is now unblocked, so the "Blocked on" column only records
genuine *evidence* gaps rather than questions for the user:

| Phase | Real blocker left | Notes |
|---|---|---|
| 1b `segmentation` | - | done 2026-10-04 (label loader, `seg_gate` GPU kernel and the cfgs still open) |
| 1b `omd_loss` | - | done 2026-10-04 (`show_details` reporting still open) |
| 1b deferred (c) `contrastive` | - | classifier branch done 2026-10-04; detection branch needs the yolo head's per-box class ids; `region` / `detection` decode + loss still open |
| 1b `laya_*` | - | **complete 2026-10-05**: `laya_scorer`, `laya_marker_gather`, `laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head` |
| 2A anchor-free heads | - | `yolox` done 2026-10-04; `ppyoloe` / `yolonas` (shared DFL) and `centernet` to go, plus the LegoGears nested-object fix |
| 2B `moe` / `eml` | - | `moe_convolutional` CPU and Oxide forward/backward plus SGD/Adam paths implemented as detailed above. v67 added source-derived EML math; v68 adds host-reference and native Oxide f32 branch-combine ops; v69 adds EML cfg/shape parsing, paired-convolution CPU forward/backward, weights, SGD/Adam; v70 adds native f32 backend forward for both branches and their combine, including rolling-stat BN folding. `oxide_runtime_chain` passes 19/19, including BN/no-BN outputs and cache invalidation after host updates. Focused host EML tests pass 3/3 and EML weights round-trip passes. v71 adds native no-BN Oxide backward and host parameter-gradient accumulation; BN EML backward retains CPU fallback, and optimizer updates remain host-side. Still open: BN native backward, device-side optimizer, and C++ batch-stat/EMA BN training behavior; no compiled-C++ golden is available, so C++ parity is unverified. Save order verified as `input_layer` then `self_layer`, two full conv blocks. |
| 2B `dcnv3` | **no reference numerics** | see the finding below - all five branches that carry it have a stub forward/backward/update, so it cannot be a verified port |
| 2C OBB family | - | unblocked by decision 4; XL, and it also changes data loading, NMS and mAP |
| 3A transformer family | - | unblocked by decisions 3 and 7; needs the shared kernel set (LayerNorm fwd/bwd, batched GEMM, row-softmax backward, residual/bias epilogues) built once first |
| 3B conv family | - | `dcnv4` has a real C++ forward; `deform_conv`, `dsconv`, `graph_conv`, `wmhf`, `clifford`, `recursive_block` still to go |
| 4 foveation / depth / LAYA / DETR | - | unblocked by decision 2; DETR is still broken in C++, so its reference is the paper, like `dcnv3` |

### Finding: `[dcnv3]` has no reference implementation anywhere in this repo

Verified 2026-10-04 on `deform`, `Jev_laya`, `fog`, `jepa` and `depth_3d`:
`forward_dcnv3_layer`, `backward_dcnv3_layer` and `update_dcnv3_layer` are all
`display_warning_msg("... not fully implemented")` placeholders, and
`docs/dcnv3_layer.md` says so explicitly ("**Status**: STUB",
"1. No CPU implementation yet").  The plan's "[verified]" weights order
(`save_dcnv3_weights`, confirmed correct) is the only part that can be ported
against C++; the numerics would have to come from the InternImage reference
instead, which is a different contract.  `[dcnv4]` (Phase 3B) is the tractable
deformable target instead: its CPU forward and GPU kernels are real.

# Self-review (writing-plans checklist, run against this document)

- **Spec coverage.** Requested items: (1) Layer Port Contract = "Layer Port Contract" section (letters A-G, referenced from tasks); (2) Tier 0/1 detailed = Tasks 1-12 (activations first: Task 1-3; gru/local/batchnorm/crop/zero_nonmax-GPU/region demoted with evidence to "Deferred Tier 0 leftovers"); (3) Review Focus = 5 items each pinned to a named test; (4) Verification against C++ = dedicated section with the honest prerequisite (C++ hook only in `96f662f0`); (5) Phases 2-4 + Out of scope + Open questions = present. Tier 1 items not fully detailed: `segmentation`, `omd_loss`, `laya_*` are in "Phase 1b" because their C++ was not read in enough detail to write correct code (stated, with the exact reason).
- **Placeholder scan.** No `TBD`/`TODO`; every code step has code. The two places that depend on something I could not confirm statically are concrete instructions with a fallback: Task 9 Step 7 (the fallback copy-back loop) and Task 3 Step 1(c) (the remaining exhaustive `F32ActivationKind` matches are listed by the compiler).
- **Type consistency.** `ShortcutParams/ShortcutTrainBuffers/ShortcutWeightsType/ShortcutWeightsNorm/weight_count/weight_slots` (Task 6), `ChannelSliceSpec/ChannelShuffleSpec/DeconvolutionalSpec/RegressionSpec/JevCeSpec`, `slice_channels/slice_channels_backward/shuffle_channels`, `deconv_forward/deconv_backward/DeconvGrad`, `compute_dense_training_loss/DenseTrainingLoss/SECRET_NUM`, `jev_ce_probs/jev_ce_loss/JevCeLoss`, `channel_permute`, `channel_shuffle_f32` are each defined once and used with the same signature in later tasks.
- **Review Focus coverage.** (1) Tasks 5, 6, 9; (2) Tasks 7, 9; (3) Tasks 4, 7, 9, 11; (4) Tasks 1, 2; (5) Tasks 7-11 GPU parity tests.

# Progress checklist

Phase 0 (Tasks 1–12) is **delivered** — see
[`implemented.md`](implemented.md) §4 for the per-task inventory and evidence.

- [ ] Phase 1b / 2 / 3 / 4: expand into their own plans (one per bullet group) before starting

**All Open Questions are resolved** (2026-10-04, "port everything in Rust") -
see the RESOLVED section below. Phase 1b progress: `segmentation`, `omd_loss`,
the `contrastive` classifier branch, and all `laya_*` layers are done; Phase 2A
is complete (`yolox`, `ppyoloe`, `yolonas`, `centernet`, the nested-object fix
and decode/mAP integration). Phase 2B's MoE has CPU support and partial Oxide
forward/backward plus no-BN router/expert SGD and Adam on-device; BN expert
backward remains. Cross-phase state: `tasks/rust/STATUS.md`; full
inventory: `tasks/rust/implemented.md`.

Phase 0 landed in commits `8e4b0f24` .. `be71eb6e`; the per-task evidence is in
`implemented.md` §4 (verified by `./tools/test_host.sh` and
`./tools/test_oxide.sh`, all green).

Phases 1b-4 are **in progress / not started** (recording the state at commit
`cdf68d00`).  Phase 2B's `moe_convolutional` CPU path is done (see its bullet
and `tasks/rust/todo.md`); each remaining phase section above still says it must
be expanded into its own plan first, and several bullets are blocked on the Open
Questions at the end of this document rather than on code:

| Phase | Blocked on | Notes |
|---|---|---|
| 1b `segmentation` | - | **done 2026-10-04**: `src/darknet-core/src/segmentation.rs` + `SegmentationSpec`, including the C++ correct-pixel gate. The label loader, the `seg_gate` GPU kernel and the cfgs stay open (Open Questions 1 and 5) |
| 1b `omd_loss` | - | **done 2026-10-04**: `src/darknet-core/src/omd_loss.rs` + `OmdLossSpec`, all three samplers and the metric pair loss. It turned out **not** to need the extra seed on layer `idx-1` the plan assumed: the layer is a plain pass-through in both directions, so its own seed already reaches the previous layer. `show_details` reporting and a real detector cfg are still open |
| 1b `laya_*` | - | **done 2026-10-05**: whole family (`laya_scorer`, `laya_marker_gather`, `laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head`) |
| 1b deferred (c) `contrastive` | - | **partly done 2026-10-04**: classifier branch ported (`src/darknet-core/src/contrastive.rs`). The detection branch needs the yolo head's per-box class ids, so it reports itself as unported. `region` / `detection` decode+loss still open |
| 2A anchor-free heads | nothing but plan expansion | needs `decode.rs` / `evaluate.rs` work, not just a layer |
| 2B `moe_convolutional`, `eml_convolutional`, `dcnv3` | `dcnv3` has no reference | MoE CPU and Oxide forward/backward plus SGD/Adam for no-BN and BN experts; BN expert SGD momentum is device-resident and BN Adam scale moments persist in DNOPT4. `eml` save order verified 2026-10-05 (`input_layer` then `self_layer`) |
| 2C OBB family | Open Question 4 | XL; also changes data loading, NMS and mAP |
| 3A/3B transformer + conv family | shared oxide kernel set (prerequisite (2)), Open Question 3 (`flash_attention`), Open Question 7 (swin weights order is ours to define) | |
| 4 foveation / depth / LAYA / DETR | Open Question 2 (LAYA), and DETR is broken in C++ | XL each |

The honest ordering is: answer Open Questions 1-7, then expand one phase at a
time.  Doing it the other way round produces layers whose weights order and
inference contract nobody agreed to.
