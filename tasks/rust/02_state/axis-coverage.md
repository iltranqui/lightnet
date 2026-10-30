# Axis coverage ledger (branch rust)

One row per `LayerKind` variant, with the C++ source that defines its axis
contract and the named test that pins it. This is the evidence file for the
"axis alignment" half of the v0.1 scope; `plan-open-items.md` owns the work
queue, `STATUS.md` owns the version log.

Measured 2026-10-24 against `model.rs` (65 represented variants plus
`Unsupported`), `Jev_laya:src-lib/darknet_enums.hpp` and the `depth_3d`
YOLO-3D source. Represented does not imply a complete layer port:

- **46 C++-backed LayerKind variants pinned**, with an additional DSConv multi-template contract row; `axis_layout` selects 52 named tests,
  including the Rust-only Crop contract pin and additional convolution/Swin
  probes.
- **14 C++-backed variants open** (12 on `Jev_laya`, plus `Regression` on
  `fog` and the partial `Yolo3d` on `depth_3d`).
- **5 Rust-only extensions** with no C++ layer type to anchor against.
- **4 layer ports incomplete/partial**: `Yolo3d` has only its raw runtime and
  label/config helpers; Region truth-loss/decode is absent; `Dcnv3` and
  `DetrDecoder` have no variants.

46 + 14 + 5 = 65 represented variants; `Unsupported` is a separate sentinel.
`Yolo3d` remains open: identity/seed/checkpoint tests are not a decode-axis pin.

Evidence rule for every row and every new test: **hand-derived from source, no
compiled-C++ golden**. Do not write "C++ parity".

## Contents

- [What counts as an axis pin](#what-counts-as-an-axis-pin)
- [Axis classes](#axis-classes)
- [Fixture rules](#fixture-rules)
- [Summary](#summary)
- [Pinned variants (46)](#pinned-variants-46)
- [Open: Tier 1, shape and geometry](#open-tier-1-shape-and-geometry)
- [Open: Tier 2, heads and losses](#open-tier-2-heads-and-losses)
- [Open: Tier 3, sequence and recurrent](#open-tier-3-sequence-and-recurrent)
- [Open: Tier 4, recent CPU ports](#open-tier-4-recent-cpu-ports)
- [Rust-only extensions (5)](#rust-only-extensions-5)
- [Not complete (4)](#not-complete-4)
- [Weights axis](#weights-axis)
- [Findings](#findings)
- [Adding a pin](#adding-a-pin)

## What counts as an axis pin

A test is a pin when it satisfies all of these:

1. The name contains `axis_layout`, so `MEM=3G ./tools/test_host.sh axis_layout`
   selects it.
2. The fixture is **non-square** (`W != H`), uses `batch > 1`, and has
   `channels != spatial`. Square, single-image, `C == H == W` fixtures hide
   transposes.
3. At least one value differs between every pair of axes that the layer could
   confuse (x vs y, batch vs channel, token vs channel, input vs output
   channel). Distinct per-index sentinels satisfy this.
4. The expectation is hand-computed from the C++ loop order, not captured from
   the Rust implementation.
5. It fails when the suspect axis is swapped. Where practical, state the wrong
   value the test would have observed (see the existing DCNv4 and Swin tests).

A finite-difference test is **not** an axis pin: it catches sign and scale
errors, and is invariant to a consistent axis transpose.

## Axis classes

From `plan-open-items.md` "Axis alignment checklist". Each row below lists the
classes that apply.

| Class | What is pinned |
|---|---|
| `layout` | NCHW flat index `((n*C + c)*H + h)*W + w` for input, output, deltas |
| `token` | `(h,w)` to token index order and channel-major vs token-major packing |
| `reduce` | reduction axis for norm/softmax/pool, and `eps` placement |
| `head` | `C -> heads x head_dim` split, Q/K/V packing, output merge |
| `concat` | `route`/`channel_slice`/`shortcut` source-layer shape and slot order |
| `window` | window partition and shift/roll direction and sign |
| `offset` | offset channel order `(dy,dx)` vs `(dx,dy)`, group split, `x=w`, `y=h` |
| `geometry` | output extent from `size`/`stride`/`padding`/`dilation` |
| `weights` | stored order `[out][in][kh][kw]` and the save order per layer |
| `batch` | independent samples, no cross-sample reduction unless C++ does it |

## Fixture rules

- Use the shared helpers in `src/lightnet-core/src/axis_fixture.rs`
  (`axis_tensor`, `nchw_index`, `probe_shape`, `axis_value`) instead of
  hand-writing nested loops. These helpers exist; older pins may retain their
  original explicit fixtures.
- Prefer `W = 3, H = 2, C = 2, batch = 2` as the default probing shape, and add
  a second shape when the layer has parity behaviour (`C != filters`,
  `size` even vs odd).
- Assert on **indices**, not only values: a pin should show that output element
  `(n,c,h,w)` came from the intended input element.
- Keep the assertion message specific: say which axis was swapped and what the
  wrong value would be.

## Summary

| Status | Count | Notes |
|---|---|---|
| Pinned | 46 C++-backed LayerKind variants, 47 contract rows; 52 `axis_layout` tests total | transformer, LAYA, MoE/EML, DCNv4/DSConv/DeformConv/GraphConv/WMHF/Clifford/DepthAnything, DPT, conv/connected, maxpool/avgpool, localavgpool, route/upsample/reorg/shuffle/softmax/scale/SAM, slice/shortcut/deconv, Logistic, RNN/LSTM/CRNN, Contrastive/JevCe/OMD, FoveationFilm, HyperbolicFoveation; one Rust-only Crop contract test |
| Open, Tier 1 | 3 | dropout, cost, recursive block |
| Open, Tier 2 | 11 | yolo family including partial yolo_3d, modern heads, OBB, losses, segmentation, regression |
| Open, Tier 3 | 0 | — |
| Open, Tier 4 | 0 | — |
| Rust-only | 5 | no C++ layer type; pin Rust's own contract |
| Ports incomplete/partial | 4 | Region truth-loss/decode; yolo_3d partial; dcnv3 and detr_decoder blocked |

`axis_layout` current baseline after the GRU Rust-contract, geometry/layout pins, DSConv, DeformConv, GraphConv, WMHF, Clifford and DepthAnything pins:
**52 passed** in `cargo xtask test axis_layout` for v0.1.044; the prior baseline was 50 passed, 946 filtered out (996 lib tests; run under `MEM=3G`). Logistic, RNN/LSTM/CRNN, Contrastive/JevCe/OMD, FoveationFilm and HyperbolicFoveation have hand-computed source-derived pins; Crop's Rust-contract pin is included. Every tier below grows that count, which must never shrink.

## Pinned variants (46)

| Variant | C++ reference | Axis classes | Pinning test |
|---|---|---|---|
| `Convolutional` | `Jev_laya:src-lib/convolutional_layer.cpp:1615`, extent from `convolutional_out_width` at `:317` | geometry, layout, weights, batch | `model.rs` `convolutional_axis_layout_pins_dilated_extent_and_groups`; `runtime.rs` `convolutional_axis_layout_keeps_nchw_and_group_partitions` |
| `Logistic` | `raptor:src-lib/logistic_layer.cpp:72-77`, elementwise loop in `activations.cpp:127-143` | layout, batch | `runtime.rs` `logistic_axis_layout_keeps_batch_channel_hw_order` |
| `ChannelSlice` | `Jev_laya:src-lib/channel_slice_layer.cpp:95` | layout, concat, batch | `channel_slice.rs:65` `slice_axis_layout_preserves_batch_channel_and_spatial_order` |
| `Deconvolutional` | `Jev_laya:src-lib/deconvolutional_layer.cpp:154` | layout, weights, batch | `deconvolutional.rs:120` `deconvolutional_axis_layout_maps_input_channels_to_filters_and_batch` |
| `Connected` | `Jev_laya:src-lib/connected_layer.cpp:186` | weights, layout, batch | `runtime.rs` `connected_axis_layout_keeps_batch_rows_and_output_weight_axes` |
| `MaxPool` | `Jev_laya:src-lib/maxpool_layer.cpp:309,327` | geometry, reduce, layout, batch | `runtime.rs` `maxpool_axis_layout_pins_depth_channel_batch_and_spatial_order` |
| `AvgPool` | `Jev_laya:src-lib/avgpool_layer.cpp:51` | reduce, layout, batch | `runtime.rs` `avgpool_axis_layout_reduces_spatial_per_channel_and_batch` |
| `LocalAvgPool` | `Jev_laya:src-lib/maxpool_layer.cpp:440` | geometry, reduce, layout, batch | `runtime.rs` `local_avgpool_axis_layout_pins_window_and_batch_order` |
| `Route` | `Jev_laya:src-lib/route_layer.cpp:91-139` | concat, layout, batch | `runtime.rs` `route_axis_layout_pins_named_source_order_group_slice_and_delta_scatter` |
| `Upsample` | `Jev_laya:src-lib/upsample_layer.cpp:6-28`; `blas.cpp:669-695` | geometry, layout, reduce, batch | `runtime.rs` `upsample_axis_layout_pins_forward_repeat_reverse_sum_and_adjoint` |
| `Reorg` | `Jev_laya:src-lib/reorg_layer.cpp:112-128`; `blas.cpp:69-100` | layout, batch | `runtime.rs` `reorg_axis_layout_pins_batch_channel_offset_and_adjoint` (normal forward path; reverse source defect noted in F2) |
| `ChannelShuffle` | `Jev_laya:src-lib/channel_shuffle_layer.cpp:11-20,95-113` | layout, batch | `channel_shuffle.rs` `shuffle_axis_layout_preserves_batch_spatial_planes_and_group_interleave` |
| `Softmax` | `Jev_laya:src-lib/softmax_layer.cpp:75-85`; `blas.cpp:655-665` | reduce, layout, batch | `runtime.rs` `softmax_axis_layout_pins_per_batch_contiguous_groups_and_adjoint` |
| `ScaleChannels` | `Jev_laya:src-lib/scale_channels_layer.cpp:12-35,73-135`; `blas_kernels.cu:1664-1676` | layout, concat, batch | `runtime.rs` `scale_channels_axis_layout_pins_named_source_and_channel_or_spatial_broadcast` |
| `Sam` | `Jev_laya:src-lib/sam_layer.cpp:69-105`; `blas_kernels.cu:1752-1798` | layout, batch | `runtime.rs` `sam_axis_layout_pins_elementwise_nchw_source_and_adjoint` |
| `Shortcut` | `Jev_laya:src-lib/shortcut_layer.cpp:171` via `blas.cpp::shortcut_multilayer_cpu` | layout, concat, weights | `shortcut_weights.rs:182` `shortcut_axis_layout_uses_nchw_channel_slots_per_batch` |
| `Vit` | `Jev_laya:src-lib/vit_layer.cpp:1260` (`vit_patchify_cpu`) | token, layout | `vit.rs:2208` `patchify_axis_layout_keeps_channel_fastest_order_across_batches` |
| `Swin` | `Jev_laya:src-lib/transformer_layer.cpp:888` (`window_partition`, `cyclic_shift`) | window, layout | `swin.rs:1186` `window_partition_axis_layout_is_batch_major_raster_then_channel`, `swin.rs:1200` `cyclic_shift_axis_layout_wraps_each_nchw_plane_with_source_signs` |
| `TuckerAttention` | `Jev_laya:src-lib/tucker_attention_layer.cpp:357` | token, window, batch | `tucker_attention.rs:891` `axis_layout_keeps_batch_window_token_and_channel_axes_independent` |
| `MambaVision` | `Jev_laya:src-lib/mambavision_layer.cpp:885` (`scan_forward`) | token, window, batch | `mambavision.rs:1716` `axis_layout_maps_nchw_rows_and_scan_state_per_batch` |
| `KimiKda` | `Jev_laya:src-lib/kimi_kda_layer.cpp` (`flatten_cpu`, delta scan) | token, batch | `kimi_kda.rs:1167` `axis_layout_keeps_nchw_raster_scans_independent_per_batch` |
| `Moe` | `Jev_laya:src-lib/moe_convolutional_layer.cpp` (`compute_cpu_router`) | reduce, batch | `runtime.rs:39815` `cpu_moe_convolutional_axis_layout_keeps_router_and_blend_per_batch` |
| `EmlConvolutional` | `Jev_laya:src-lib/convolutional_layer.cpp:1850` | layout, batch | `runtime.rs:39248` `eml_convolutional_axis_layout_preserves_batch_channel_and_spatial_order` |
| `Dcnv4` | `Jev_laya:src-lib/dcnv4_layer.cpp:660` (`dcnv4_deformable_forward_cpu`) | offset, layout, batch | `dcnv4.rs:860` `offset_axis_layout_reads_dx_then_dy_and_keeps_batches_nchw` |
| `DeformConv` | `Jev_laya:src-lib/deform_conv_layer.cpp:518-560,834-852` | offset, layout, batch | `deform_conv.rs` `deform_conv_axis_layout_pins_cpp_offset_mask_groups_and_batch_nchw` |
| `GraphConv` | `Jev_laya:src-lib/graph_conv_layer.cpp:55-83,215-224,621-808,823-855` | layout, weights, reduce, batch | `graph_conv.rs` `graph_conv_axis_layout_pins_cpp_batch_group_neighbor_raster_and_nchw` |
| `Wmhf` | `Jev_laya:src-lib/wmhf_layer.cpp:274-303` (`dwt_cpu`) | layout, batch, reduce | `wmhf.rs` `wmhf_axis_layout_pins_cpp_batch_channel_spatial_and_subband_order` |
| `Clifford` | `Jev_laya:src-lib/clifford_layer.cpp:311-346,1578-1580` (`layernorm_nchw_forward`) | layout, reduce, batch | `clifford.rs` `clifford_axis_layout_pins_cpp_nchw_token_channel_normalization` |
| `DsConv` | `Jev_laya:src-lib/dsconv_layer.cpp:103-167,223-247,324-336` | offset, geometry, batch, layout | `dsconv.rs` `dsconv_axis_layout_pins_cpp_snake_branches_batch_channels_and_nchw_outputs` |
| `DsConv` multi-template | `Jev_laya:src-lib/dsconv_layer.cpp:597-638` | alternating snake views, offset-block order, batch/channel/spatial layout | `runtime/tests/dsconv_multi_tests.rs` `dsconv_multi_template_axis_layout_pins_view_order_and_nchw` |
| `DptHead` | `Jev_laya:src-lib/dpt_head_layer.cpp:534` (`dpt_forward_impl`) | layout, geometry, batch | `runtime/tests/dpt_head_runtime_tests.rs:274` `dpt_head_axis_layout_batched_nchw_forward_matches_independent_non_square_samples` |
| `DepthAnything` | `Jev_laya:src-lib/depth_anything_layer.cpp:300-339`; weights in `src-lib/weights.cpp:1433-1464` | token, layout, weights, batch | `depth_anything.rs` `depth_anything_axis_layout_pins_cpp_patch_tokens_cls_and_nchw_batches`; source-order length and blob boundary in `depth_anything_type_major_weight_arrays_match_source_order_and_lengths`, `depth_anything_external_weights_resolve_and_are_excluded_from_network_stream`, `depth_anything_sequential_network_weights_round_trip_type_major` |
| `LayaEncoder` | `Jev_laya:src-lib/laya_encoder_layer.cpp:576` (`la_forward_impl`) | token, batch | `laya_encoder.rs:624` `encoder_axis_layout_keeps_batch_and_token_rows_independent` |
| `LayaHead` | `Jev_laya:src-lib/laya_head_layer.cpp:486` (`lh_forward_impl`) | token, batch | `laya_head.rs:548` `head_axis_layout_preserves_batch_sequence_and_qtype_channels` |
| `LayaMarkerGather` | `Jev_laya:src-lib/laya_marker_gather_layer.cpp:111` | layout, batch | `laya_marker_gather.rs:200` `marker_gather_axis_layout_uses_sample_specific_positions` |
| `LayaActHead` | `Jev_laya:src-lib/laya_act_head_layer.cpp:223` | layout, batch | `laya_act_head.rs:547` `act_head_axis_layout_is_batch_major_over_the_two_layer_mlp` |
| `LayaScorer` | `Jev_laya:src-lib/laya_scorer_layer.cpp:262` (`sc_forward_impl`) | reduce, batch | `laya_scorer.rs:575` `scorer_axis_layout_keeps_batches_channels_and_options_independent` |
| `LayaLoss` | `Jev_laya:src-lib/laya_loss_layer.cpp:296` (`loss_forward_host`) | batch, reduce | `laya_loss.rs:459` `loss_axis_layout_excludes_padding_rows_from_batch_mean` |
| `Region` | `Jev_laya:src-lib/region_layer.cpp:209`, `blas.cpp::flatten:104` | head, layout, batch | `runtime.rs` `region_axis_layout_matches_cpp_flatten_objectness_and_classes` |
| `FoveationFilm` | `Jev_laya:src-lib/foveation_film_layer.cpp:160-222` | layout, reduce | `foveation_film.rs` `foveation_film_axis_layout_maps_batch_hf_embedding_and_channel_affines` |
| `HyperbolicFoveation` | `Jev_laya:src-lib/hyperbolic_foveation_layer.cpp:1341-1441` (`map_x/map_y`, `hf_grid`, per-channel `cv::remap`) | layout, geometry, batch | `hyperbolic_foveation.rs` `hyperbolic_foveation_axis_layout_keeps_batch_channel_hw_and_grid_xy` |
| `Rnn` | `Jev_laya:src-lib/rnn_layer.cpp:92-133` (`forward_rnn_layer`) | sequence, batch, input-feature | `runtime.rs` `rnn_axis_layout_pins_cpp_step_batch_and_input_channels` |
| `Lstm` | `Jev_laya:src-lib/lstm_layer.cpp:46-83,177-260` (`forward_lstm_layer`) | gate, sequence, batch, cell/state | `runtime.rs` `lstm_axis_layout_pins_cpp_gate_blocks_steps_and_samples` |
| `Crnn` | `Jev_laya:src-lib/crnn_layer.cpp:28-83,165-218` (`forward_crnn_layer`) | spatial, sequence, batch, channel | `runtime.rs` `crnn_axis_layout_pins_cpp_spatial_step_and_sample_order` |
| `Contrastive` | `Jev_laya:src-lib/softmax_layer.cpp:265-430` (`forward_contrastive_layer`) | reduce, batch, feature | `contrastive.rs` `contrastive_axis_layout_pins_cpp_batch_rows_and_embedding_features` |
| `JevCe` | `Jev_laya:src-lib/jev_ce_layer.cpp:177-183,311-323` | reduce, layout, batch | `jev_ce.rs` `jev_ce_axis_layout_pins_cpp_batch_group_candidate_and_layout_order` |
| `OmdLoss` | `Jev_laya:src-lib/omd_loss_layer.cpp:86-95,143-185,189-260` | reduce, layout, batch | `omd_loss.rs` `omd_axis_layout_pins_cpp_nchw_sample_rows_pairs_and_scatter`; clipping: `max_delta_clips_the_injected_delta` |

## Open: Tier 1, shape and geometry

Highest risk per line of code: every cfg uses these, and a wrong extent
silently changes a downstream `route`/`shortcut`.

| Variant | Aliases | C++ reference | Axis classes | Required pin |
|---|---|---|---|---|
| `Dropout` | `dropout` | `Jev_laya:src-lib/dropout_layer.cpp:87-133`; `dropout_layer_kernels.cu:127-167` | layout, batch | Regular per-element mask/inference identity has focused coverage in `runtime.rs` `dropout_element_mask_and_adjoint_preserve_nchw_slots`; complete pin remains open because the DropBlock GPU source indexing is unresolved (F3). |
| `RecursiveBlock` | `recursive_block`, `recurrent_block` | `recursive_block_layer.cpp:1586` | geometry, batch | nested body shapes/`current` propagation; per-pass lattice spatial mapping; `batch=2` independent |
| `Cost` | `cost` | `cost_layer.cpp:85` | reduce | `MASKED`/`SMOOTH` reduction axis and the `scale` factor; `C` vs `N` layout |

## Open: Tier 2, heads and losses

These feed training losses, so a transposed head trains silently wrong rather
than failing. Pins here are the highest-value per test.

| Variant | Aliases | C++ reference | Axis classes | Required pin |
|---|---|---|---|---|
| `Yolo3d` | `yolo_3d`, `yolo3d`, `cuboid`, `cuboid_yolo` | `depth_3d@9c9e0b4b:src-lib/yolo_3d.cpp:102-109,1679-1755` | head, layout, batch | raw runtime exists; pin anchor/entry/vertex indexing together with decode; metric channels pending |
| `Yolo` | `yolo` | `yolo_layer.cpp:957` | head, layout, batch | anchor-major vs entry-major output; `mask` order; cell `(y,x)` iteration; class offset per anchor; `batch=2` |
| `GaussianYolo` | `gaussian_yolo` | `gaussian_yolo_layer.cpp:505` | head, layout | 4 extra mean/variance channels interleaved per entry |
| `Yolox` | `yolox` | `src-lib/yolox_layer.cpp` via `modern_yolo_layer.cpp:395` | head, layout | class/obj/box channel split; grid `(y,x)` order; `batch=2` |
| `Ppyoloe` | `ppyoloe` | `src-lib/ppyoloe_layer.cpp` via `modern_yolo_layer.cpp` | head, layout | DFL bin axis (`4*(reg_max+1)`) ordering; TAL/grid index order |
| `Yolonas` | `yolonas` | `src-lib/yolonas_layer.cpp` via `modern_yolo_layer.cpp` | head, layout | per-level channel layout and decode order |
| `CenterNet` | `centernet` | `centernet_layer.cpp:395` | head, layout | heatmap plane `(y,x)` not `(x,y)`; `HM`/`WH`/`REG` block order |
| `ObbYolo` | `obb_yolo` | `raptor:src-lib/yolo_layer_obb.cpp` | head, layout | six-coordinate per-entry order; front-point channel placement; `raptor` only |
| `ObbRefine` | `obb_refine` | `raptor:src-lib/` CARPN head | head, layout | per-sample proposal row to output row mapping |
| `Segmentation` | `segmentation`, `segmenter` | `raptor:src-seg/segmentation_layer.cpp` | layout, reduce | per-pixel class axis: `(n,c,h,w)` softmax over `c`, mask index `h*W+w` |
| `Regression` | `regression` | `fog`:src-lib/regression_layer.cpp (absent on `Jev_laya`) | reduce, layout | target/seed axis and `C` vs `N` layout; audit the branch before pinning |

## Open: Tier 3, sequence and recurrent

| Variant | Aliases | C++ reference | Axis classes | Required pin |
|---|---|---|---|---|

## Open: Tier 4, recent CPU ports

These have finite-difference and hand-math tests; they lack a *named axis* pin.
Weights order is covered for all of them (see [Weights axis](#weights-axis)).

| Variant | C++ reference | Axis classes | Required pin |
|---|---|---|---|

## Rust-only extensions (4)

Not in `ELayerType` on any branch (`layers.md:79`), so there is no C++
layer type to anchor against. Pin Rust's documented contract and record the
named test; use `no-C++-reference` rather than claiming C++ parity.

| Variant | Aliases | Reference | Required pin | Contract test |
|---|---|---|---|---|
| `Gru` | `gru` | none | per-sample reset; gate block order | `runtime.rs` `gru_axis_layout_keeps_samples_independent_and_gate_blocks_ordered` |
| `Crop` | `crop` | Rust-only layer; GPU-only `Jev_laya:src-lib/crop_layer_kernels.cu` | crop window to output index mapping, `batch=2` | `runtime.rs` `crop_axis_layout_maps_batches_channels_and_center_window` (Rust contract; no C++ golden) |
| `Local` | `local` | none | per-sample weight sharing; `C*H*W` to `out` layout | open |
| `BatchNorm` | `batchnorm` | `Jev_laya:src-lib/batchnorm_layer.cpp` exists but has no `ELayerType` | channel-axis reduction; `(C)` scale/beta order | open |

## Not complete (4)

| Variant | Reference | Status |
|---|---|---|
| `Region` | `Jev_laya:src-lib/region_layer.cpp:209-235,256-390` | CPU flatten/objectness/softmax and seeded adjoint are implemented; truth-driven loss/decode and GPU path remain open |
| `Yolo3d` | `depth_3d@9c9e0b4b:src-lib/yolo_3d.cpp:1356` | config/labels plus raw runtime only in v0.1.024–026; loss/decode/evaluator and A2 axis DoD remain in `yolo_3d_plan.md` |
| `Dcnv3` | `dcnv3_layer.cpp:199` | blocked: stubs in all five branches |
| `DetrDecoder` | `detr_decoder_layer.cpp:479` | blocked: C++ broken |

## Weights axis

Save-order is the one axis class that is already well covered. Round-trip tests
that assert exact length and source order exist for: `deconvolutional`
(`weights.rs`), `shortcut`, `standalone batchnorm`, `vit`, `swin`,
`tucker_attention`, `mambavision`, `kimi_kda`, `moe_convolutional`,
`eml_convolutional`, `deform_conv`, `dsconv`, `dcnv4`, `graph_conv`,
`clifford`, `wmhf`, `recursive_block` (Adam state), `hyperbolic_foveation`,
`foveation_film`, `depth_anything`, `dpt_head`, `carpn_refine`, and the
`rnn`/`lstm`/`crnn` nested blocks (`recurrent_weight_blocks_round_trip_in_darknet_order`).

Open weights-axis gaps: `[convolutional]` and `[connected]` have no named
order test of their own (they rely on network-level round trips), and
`[region]` has no weights block in Rust. Adding an explicit `[convolutional]`
`[out][in][kh][kw]` order test belongs to Tier 1.

## Findings

### F1: dilated convolution geometry — resolved, not a bug

Originally recorded here as "`conv_dim` is missing a `dilation*(size-1)` term".
That was wrong. `Jev_laya:src-lib/convolutional_layer.cpp:317` computes
`(w + 2*pad - size)/stride + 1` and ignores dilation, while `im2col` samples
with dilation offsets and the cuDNN path pads by `pad*dilation`. The CPU
contract is therefore "dilation changes sampling, not extent".

Applying the textbook term in Rust shrank the extent and broke
`cfg/resnet152_trident.cfg` at route #244 (`16x16` vs `38x38`), which would have
read as a route bug. `conv_dim` now carries that reasoning as a comment, and
`convolutional_axis_layout_pins_dilated_extent_and_groups` fails if the term is
reintroduced. `cfg/resnet152_trident.cfg` builds; layers #167-#207 report
`in=38x38x256 out=38x38x256` at `dilation=3, size=3, pad=1`.

Remaining divergence, deliberately not changed: cuDNN/oxide pad by
`pad*dilation`, so GPU and CPU extents can differ for dilated convolutions.
That is a source-level inconsistency, not a Rust defect.

### F2: Reorg reverse-channel argument — source-level defect, unresolved

`make_reorg_layer` sets reverse output channels to `c / (stride*stride)`
(`Jev_laya:src-lib/reorg_layer.cpp:21-29`), then calls `reorg_cpu` with that
`l.out_c` (`:112-116`). The helper computes `in_c = out_c/(stride*stride)`
(`blas.cpp:73`) and uses it as a divisor (`:84-86`). For ordinary reverse
shapes such as input C=8, stride=2, `out_c=2` and `in_c=0`, so the C++ reverse
mapping is not a defensible pin without a compiled oracle. The new Reorg pin
therefore covers only normal forward plus its adjoint, whose `forward=0/1`
helper mapping is well-defined. Rust's reverse path remains a separate
source-inconsistent behavior; do not claim C++ parity for it.

### F3: DropBlock GPU plane indexing — unresolved

The regular CPU dropout loop visits the flat `batch * inputs` range and stores
one random value per element (`Jev_laya:src-lib/dropout_layer.cpp:87-133`);
`dropout_element_mask_and_adjoint_preserve_nchw_slots` exercises that Rust
contract, including inference identity and reuse of the exact mask in backward.
This does not close Dropout's axis row: the GPU DropBlock kernel at
`dropout_layer_kernels.cu:25-37` computes `index = b*spatial*f + f*spatial + i`,
which does not match the ordinary NCHW plane offset `(b*filters + f)*spatial + i`
for general `batch, filters`. Without a compiled-C++ oracle, the DropBlock
batch/channel mapping is unresolved; do not treat host DropBlock fixtures as
C++ parity evidence.

### F4: axis registry drift — resolved

`AXIS_PINS` had sixteen variants marked `Open` even though their named
`*axis_layout*` tests existed and this ledger already listed them as pinned:
`Logistic`, `Route`, `Upsample`, `Reorg`, `ChannelShuffle`, `Softmax`,
`ScaleChannels`, `Sam`, `FoveationFilm`, `HyperbolicFoveation`, `DeformConv`,
`GraphConv`, `Wmhf`, `Clifford`, `DsConv`, and `DepthAnything`. The coverage
test hardcoded the stale `30 pinned / 30 open / 6 Rust-only` counts, so it could
not detect the mismatch with this ledger's `46 / 14 / 5` (plus `Unsupported`).
The registry now points those sixteen variants at their existing pins and the
test asserts `46 pinned / 14 open / 6 Rust-only` (the six include
`Unsupported`). `every_layer_kind_has_an_axis_pin_decision` passes.

## Adding a pin

1. Pick the row and confirm the C++ reference with
   `git show Jev_laya:src-lib/<file>:...` (quote paths in zsh).
2. Write the test in the module that owns the math (per-layer module, or
   `runtime.rs` for layers whose forward lives there).
3. Build the fixture with `axis_fixture` helpers; assert indices, not just
   values.
4. Run `MEM=3G ./tools/test_host.sh <layer>_axis_layout` then
   `MEM=3G ./tools/test_host.sh axis_layout`; record both counts.
5. Update this ledger row to the pinning test name, and add the increment to
   `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`.
6. If the pin fails, fix the layer in its own version before adding more pins.
   Do not weaken the pin to match the implementation.
