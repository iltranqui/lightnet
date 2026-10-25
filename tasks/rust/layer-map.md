# Layer map (branch `rust`)

Per layer directory under `src/lightnet-core/src/layers/`: which files hold the CPU reference and the GPU runtime glue, which `lightnet-cuda-oxide` kernel file holds its device kernels, and which Oxide test covers it. Maintained by hand from the file tree (no generator yet); update it when a layer directory or file appears. The layout contract is in `src/lightnet-core/src/layers/CONTEXT.md`. A `-` means no dedicated file of that kind (kernels may then live in a shared file such as `conv_direct.rs`, `elementwise.rs` or `layers_extra.rs`, and parity may be covered by `oxide_runtime_chain.rs`, `oxide_runtime_f16.rs`, `oxide_parity.rs` or `oxide_layers_extra.rs`). This file records layout only, not completeness: completeness per kind lives in [layer-support.md](layer-support.md).

## Contents

- [Map](#map)
- [Shared kernel and test files](#shared-kernel-and-test-files)

## Map

| Layer dir | CPU (`cpu.rs`) | GPU glue (`gpu.rs`) | Kernel file (`kernels/`) | Dedicated Oxide tests |
|---|---|---|---|---|
| `batch_norm` | - | yes | - | - |
| `carpn` | yes | - | - | - |
| `centernet` | yes | - | - | `oxide_centernet` |
| `channel_shuffle` | yes | yes | `channel_shuffle.rs` | - |
| `channel_slice` | yes | yes | - | - |
| `clifford` | yes | - | - | - |
| `connected` | - | yes | - | - |
| `contrastive` | yes | - | - | - |
| `convolutional` | - | yes | - | - |
| `crnn` | - | yes | - | - |
| `crop` | - | yes | - | - |
| `dcnv4` | yes | yes | `dcnv4.rs` | `oxide_dcnv4_aggregate`, `oxide_dcnv4_bn`, `oxide_dcnv4_chain`, `oxide_dcnv4_normalization` |
| `deconvolutional` | yes | yes | `deconvolutional.rs` | `oxide_deconvolutional` |
| `deform_conv` | yes | yes | - | `oxide_deform_conv_accumulate`, `oxide_deform_conv_backward_chain`, `oxide_deform_conv_chain` |
| `depth_anything` | yes | yes | shared `vit.rs`, convolution and attention | `oxide_depth_anything` (unrun WIP) |
| `dpt_head` | yes | yes | shared convolution, deconvolution and bilinear gather | `oxide_dpt_head` (unrun WIP) |
| `dropout` | - | yes | `dropout.rs` | - |
| `dsconv` | yes | yes | `dsconv.rs` | `oxide_dsconv` |
| `eml_conv_math` | yes | - | - | - |
| `eml_convolutional` | - | yes | - | - |
| `foveation_film` | yes | yes | `foveation_film.rs` | `oxide_foveation_film` |
| `graph_conv` | yes | yes | `graph_conv.rs` | `oxide_graph_conv` |
| `gru` | - | yes | - | - |
| `hyperbolic_foveation` | yes | yes | - | `oxide_hyperbolic_foveation` |
| `jev_ce` | yes | - | - | - |
| `kimi_kda` | yes | yes | `kimi_kda.rs` | `oxide_kimi_kda` (unrun WIP) |
| `laya_act_head` | yes | yes | - | - |
| `laya_encoder` | yes | yes | - | `oxide_laya_encoder_forward` |
| `laya_head` | yes | yes | - | `oxide_laya_head_attention`, `oxide_laya_head_forward` |
| `laya_loss` | yes | - | - | - |
| `laya_marker_gather` | yes | yes | - | - |
| `laya_scorer` | yes | yes | - | `oxide_laya_scorer_forward` |
| `local` | - | yes | `local.rs` | `oxide_local` |
| `lstm` | - | yes | - | - |
| `mambavision` | yes | yes | `mambavision.rs` | `oxide_mambavision` (unrun WIP) |
| `maxpool_depth` | yes | - | - | - |
| `modern_yolo` | yes | - | - | - |
| `moe_conv` | yes | yes | - | - |
| `omd_loss` | yes | - | - | - |
| `pool` | - | yes | `pool.rs` | `oxide_pool_parity` |
| `recursive_block` | host in runtime | yes | shared convolution, connected and elementwise | `oxide_recursive_block` (unrun WIP) |
| `region` | - | yes | - | - |
| `reorg` | - | yes | `reorg.rs` | - |
| `rnn` | - | yes | - | - |
| `route` | - | yes | `route.rs` | - |
| `sam` | - | yes | - | - |
| `scale_channels` | - | yes | `scale_channels.rs` | - |
| `segmentation` | yes | - | - | - |
| `shortcut` | - | yes | `shortcut.rs` | - |
| `shortcut_weights` | yes | - | - | - |
| `softmax` | - | yes | `softmax.rs` | - |
| `swin` | yes | yes | - | - |
| `tucker_attention` | yes | yes | `tucker_attention.rs` | `oxide_tucker_attention` (unrun WIP) |
| `upsample` | - | yes | - | - |
| `vc_attention` | yes | - | - | - |
| `vit` | yes | yes | `vit.rs` | `oxide_vit` |
| `wmhf` | yes | - | `wmhf.rs` | `oxide_wmhf` |

## Shared kernel and test files

- Kernels shared across layers: `activations.rs`, `amp.rs`, `batchnorm.rs`, `bias_scale.rs`, `conv_direct.rs`, `conv_implicit.rs`, `convert.rs`, `elementwise.rs`, `gemm*.rs`, `half_ops.rs`, `im2col.rs`, `layers_extra.rs` (dropout unit, marker position, constrain, gradient centralize, residual), `muon.rs`, `optim.rs`, `quant_fp4.rs`, `resample.rs`, `unroll.rs`.
- Chain tests: `oxide_runtime_chain.rs` (98 tests), `oxide_runtime_f16.rs` (23), `oxide_parity.rs`, `oxide_layers_extra.rs`.
- Open: the `OxideBackend` methods in `lightnet-cuda-oxide/src/lib.rs` are not yet split per layer.
