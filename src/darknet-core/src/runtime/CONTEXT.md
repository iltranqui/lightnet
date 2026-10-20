# runtime

CPU and backend-generic Darknet execution, split out of the former single
`runtime.rs`. `mod.rs` keeps the module docs, imports, shared constants, module
declarations and `pub use self::<file>::*` re-exports, so every path such as
`darknet_core::runtime::X` or `crate::runtime::X` is unchanged. Sibling files
use `use super::*;` and expose formerly private items as `pub(super)`.

## Table of contents

- [Core types](#core-types)
- [CPU executable network](#cpu-executable-network)
- [GPU executable network](#gpu-executable-network)
- [Free functions](#free-functions)
- [Pre-existing submodules](#pre-existing-submodules)
- [Tests](#tests)

## Core types

- `types.rs`: `GradMode`, `Tensor`, per-layer params / train buffers, forward caches, `ForwardResult`, `BackwardResult`.
- `gpu_types.rs`: GPU forward/backward cache structs, `GpuExecutableNetwork` struct, device Adam state types, GPU guard helpers, `gpu_optimizer_multi_step`.
- `host_adam.rs`: `HostAdamLayerState`, its validation and optimizer-state (de)serialisation.

## CPU executable network

- `exec_core.rs`: `ExecutableNetwork` struct, construction, weight / optimizer-state IO, `update_weights*` (SGD, Adam, Muon).
- `exec_forward.rs`: CPU forward entry points and `forward_with_mode_and_states`.
- `exec_backward.rs`: `backward*` entry points and `backward_seeded_core`.
- `exec_accumulate.rs`: `backward_accumulate*`, BN running stats, train-buffer accumulation, recursive-block helpers, `compute_learning_rate`.

## GPU executable network

All files are `impl GpuExecutableNetwork<B>` blocks (F32 backend) except `gpu_f16.rs`.

- `gpu_f16.rs`: f16 inference impl (`forward_compact_f16`).
- `gpu_state.rs`: constructor, accessors, optimizer-state save/load, moments IO.
- `gpu_update.rs`: train-buffer allocation, device / CPU-fallback weight updates.
- `gpu_forward.rs`: BN/EML forward helpers and `forward_with_state_impl`.
- `gpu_forward_compact.rs`: `forward_compact`.
- `gpu_backward.rs`: `backward_accumulate_seeded_gpu*` and `backward_accumulate_gpu_unscaled`.
- `gpu_backward_support.rs`: CPU-fallback detection, gradient accumulation, EML / MoE / DCNv4 device optimizers and flushes.
- `gpu_params.rs`: device BN state and cached device convolution parameters / sync to host.

## Free functions

- `defaults.rs`: default layer params and train buffers.
- `hyperbolic_foveation.rs`: hyperbolic foveation param helpers, forward and backward.
- `param_updates.rs`: gradient accumulation, SGD and host Adam / Muon updates.
- `weights_io.rs`: weight loading, binary readers, folding for inference, backend helpers.
- `forward_layers.rs`: CPU conv / local / connected / batchnorm forward and BN stats.
- `recurrent_forward.rs`: GRU / RNN / LSTM / CRNN forward (host and device).
- `recurrent_backward.rs`: recurrent backward (host and device).
- `layer_ops.rs`: softmax, region, dropout, crop, pooling, route, shortcut, upsample and other per-layer forward/backward helpers.

## Pre-existing submodules

- `dcnv4_gpu.rs`, `laya_act_gpu.rs`, `laya_encoder_gpu.rs`, `laya_head_gpu.rs`, `laya_scorer_gpu.rs`: device paths for individual layers (unchanged).

## Tests

- `tests/mod.rs`: shared imports and module list.
- `tests/tests_01.rs` to `tests/tests_07.rs`: the former inline `mod tests` body, in original order (test fn names unchanged).
- `tests/*_tests.rs`: pre-existing per-feature test files.
