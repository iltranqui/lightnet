# Device-resident convolution Muon update

Plan only. Nothing here is implemented, built or tested. It closes the "Optimizer boundary" item in `PARITY_RAPTOR.md`: convolution gradients must update natively on the device, and batch-norm convolutions must stop being rejected on the Muon path.

## Table of Contents

- [Problem](#problem)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
- [Implementation plan](#implementation-plan)
- [Verification](#verification)
- [Open questions](#open-questions)

## Problem

Today a standard convolution under Muon takes this path in `update_weights_gpu_muon` (`src/lightnet-core/src/runtime/gpu_update.rs`, about lines 390 to 1016):

1. The device weight and bias gradients are downloaded to the host (about line 535).
2. `stage_layer_with_orthogonalizer` (`src/lightnet-core/src/muon_runtime.rs:1408`) runs momentum and lookahead on the host. Only the Newton-Schulz step runs on the device, through an upload and a download per matrix.
3. Host AdamW updates the biases, `apply_staged_layer` writes the host weights, and `cached_convolution_upload` re-uploads them to the device cache.

Two limits follow. Every step pays a gradient download and a weight upload per convolution, which the user has ruled out. A convolution with batch norm is rejected at `gpu_update.rs:549` (the `!spec.batch_normalize` guard) and in the supported-families error text.

The `Connected` branch of the same function (about lines 706 to 840) already does the matrix step with device momentum (`device_muon_momentum`), `muon_momentum_lookahead_f32`, `muon_orthogonalize_f32` and `muon_apply_update_f32`. It still uploads the gradient and weights from the host and downloads the result. The resident-bank path in `gpu_resident_params.rs:444` (`update_resident_muon_parameters`) is fully device-resident, but it works on one flat serialized bank per layer, which convolutions do not use.

## Project tree

```text
src/lightnet-core/src/
├── muon_runtime.rs            host staging; visit_grouped_convolution = matrix layout (rows, cols, groups)
├── muon.rs                    MUON_SCALE, MuonLayerState, MuonMatrixState
└── runtime/
    ├── gpu_update.rs          update_weights_gpu_muon: the function to change
    ├── gpu_params.rs          device_conv_params_at / set_device_conv_params / refresh_device_convolution_params
    ├── gpu_types.rs           GpuConvolutionLayerParams (raw_*, folded weights/biases, bn, folded_stale)
    ├── gpu_resident_params.rs reference for a fully device Muon step (do not reuse the bank format)
    ├── gpu_state.rs           device_train_buffers, device_muon_momentum, host_weights_stale
    ├── param_updates.rs       host Adam for BN scales and biases (apply_host_adam_convolutional)
    └── tests/tests_06.rs      gpu_muon_rejects_batch_normalized_convolution_... (must change)
src/lightnet-cuda/src/lib.rs   F32TensorOpsBackend: muon_* and adam_update_f32 ops (reuse, no new op expected)
src/lightnet-cuda-oxide/tests/oxide_runtime_chain.rs   new Oxide parity tests
```

## Component breakdown

### `update_weights_gpu_muon` (gpu_update.rs)

- Role: per-step Muon driver for every layer family.
- Needs to do: add a device-resident arm for `LayerKind::Convolutional` (no `share_index`) before the host-staging fallback. For each group: slice the gradient and the raw weights, run momentum and lookahead, orthogonalize, apply the update with decoupled decay, write the result back into the device raw weights. Then update the bias with device AdamW, zero the gradients, and mark folded copies stale.
- Depends on: backend ops `muon_momentum_lookahead_f32`, `muon_orthogonalize_f32`, `muon_apply_update_f32`, `adam_update_f32`, `route_concat_f32` for slicing groups.
- Depended on by: `update_weights_gpu` and the deferred variant.
- Constraint: keep the host arm for families that stay host-staged. Do not change their behaviour.

### Device convolution parameters (`GpuConvolutionLayerParams`)

- Role: device source of truth for raw weights, biases, BN state, plus BN-folded inference copies.
- Needs to do: expose the raw weights and biases for in-place replacement, and set `folded_stale = true` after a step so the next forward re-folds on the device.
- Open design point: whether `raw_weights` is a shared tensor or a clone. A step must publish new tensors through `set_device_conv_params`, not mutate shared storage.

### Momentum state

- Role: per-matrix Muon momentum. Host mirror is `MuonLayerState` in `muon_state`; device copy is `device_muon_momentum`.
- Needs to do: for grouped convolutions keep one momentum tensor per group, or one tensor of shape `[groups * rows, cols]` sliced per group. Lazily upload from the host mirror on first use (this keeps checkpoint resume working). Download only when a checkpoint is written, as `materialize_resident_muon_states` does for banks.
- Constraint: the checkpoint sidecar format must not change.

### Bias and BN parameters

- Role: biases and BN scales are not matrices, so Muon uses the AdamW fallback for them.
- Needs to do: run `adam_update_f32` on device for biases. For BN convolutions also run it for scales, using `scale_updates` from the device train buffers. Rolling mean and variance are not trained and stay as they are.
- Decision to confirm: the host Adam state (`host_adam_state`, `HostAdamLayerState`) holds the moments today. The device arm needs device moments with a host mirror for the sidecar, or it must reuse the existing Adam device moments if the SGD/Adam path already keeps them for convolutions.

### Gradient conditioning

- Role: `prepare_muon_layer_gradients` applies `grad_centr` on the host for BN convolutions.
- Needs to do: do the same on the device (a per-filter mean subtraction), or reject `grad_centr` on the device arm and keep that case on the host arm until a kernel exists.

### Tests

- `gpu_muon_rejects_batch_normalized_convolution_eml_moe_dcnv4_and_deform_before_updating` (tests_06.rs:2557) must be split. The BN convolution case becomes a parity test. EML, MoE, DCNv4 and deform stay rejected.
- New Oxide tests compare the device arm with the existing host-staged arm and with the Rust CPU `update_weights_muon`.

## Implementation plan

1. Baseline. Record the current conv Muon result on Oxide for a two-step run (weights and biases) using the existing host-staged arm. This is the oracle for the new arm. Save it as a test, not a comment.
2. Matrix layout helper. Add a small function that returns, for a convolution, the group count, `rows`, `cols` and the per-group element ranges, shared by the host visitor and the device arm so both agree. Check `visit_grouped_convolution` is the only definition of the layout.
3. Device arm for non-BN convolution.
   - Read the gradient from `device_train_buffers` without downloading.
   - Lazily create device momentum from `muon_state`.
   - Per group: momentum and lookahead, finiteness check, orthogonalize, `muon_apply_update_f32` with `MUON_SCALE * sqrt(max(rows, cols))`, decay as the host arm uses.
   - Concatenate groups into the new raw weights and publish them with `set_device_conv_params`.
   - Run device AdamW on the bias, zero the gradient buffers, clear `pending`.
   - Skip the layer when `layer.train.updates_at(...)` is false and do not advance its state.
4. Mirror state back lazily. Keep `muon_state` and the host Adam state authoritative only when `sync_host_weights` is true or a checkpoint is written. Reuse the `host_weights_stale` flag; do not add a second flag.
5. BN convolutions. Remove the `!spec.batch_normalize` guard for the device arm. Add device AdamW for BN scales. Handle `grad_centr` (see the open question). Keep the folded-copy refresh through `folded_stale`.
6. Tests. Replace the BN rejection test, add the Oxide parity tests listed under verification, and add a host test with the fake backend that proves no `download_f32` of a gradient or weight happens in a step.
7. Docs. Update `PARITY_RAPTOR.md` (optimizer boundary status and the line references), `tasks/rust/02_state/layer-support.md`, and the error text at `gpu_update.rs:461`.
8. Commit with the feature tag required by `AGENTS.md`, only the files this task names.

## Verification

All commands use `cargo xtask`, 4 threads, soft RAM cap.

- `cargo xtask check` and `cargo xtask clippy`.
- `cargo xtask test muon` for the host and fake-backend tests.
- `cargo xtask oxide --test oxide_runtime_chain -- muon` for the new device arm. Cases: ungrouped conv, grouped conv, BN conv, two steps (momentum carries), `learning_rate_scale`, a gated layer, and a checkpoint round trip.
- Pass criterion: the weights and biases from the device arm match the host-staged arm within 1e-5 on one step and within 8e-4 on two steps, which is the tolerance the existing DSConv Muon test uses against the CPU.
- A transfer check: count device-to-host copies during `update_weights_gpu` on a conv net and require zero for the convolution gradients and weights.
- No training run, no loss or speed claim, until the user asks.

## Open questions

- Should `grad_centr` get a device kernel now, or stay on the host arm and be rejected in the device arm?
- Do convolution AdamW moments already exist on the device for the Adam path, or does this change add them?
- Should the host arm be deleted once the device arm covers all convolutions, or kept as the fallback for backends that lack the Muon ops (the default trait methods return an error)?
