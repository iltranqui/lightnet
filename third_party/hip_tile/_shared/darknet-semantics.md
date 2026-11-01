# Darknet semantics

Behaviour of Darknet that a Rust port must reproduce, each verified in the C++ source under `~/src/DarkSuite/darknet/src-lib/`. Read this in stage 01 of a model port or of a training port; add any newly verified fact here (one home per fact).

## Contents

- [Weights](#weights)
- [Layers](#layers)
- [YOLO decode and NMS](#yolo-decode-and-nms)
- [Images](#images)
- [Training loss](#training-loss)
- [Training deltas](#training-deltas)
- [Batch norm, training path](#batch-norm-training-path)
- [SGD and learning rate](#sgd-and-learning-rate)
- [Training data and augmentation](#training-data-and-augmentation)
- [Training-state files](#training-state-files)
- [Gradient routing](#gradient-routing)
- [Trainer verification facts](#trainer-verification-facts)

## Weights

- Header (`weights.cpp:1006-1023`): `major`, `minor`, `revision` as i32, then `seen` as u64 when `major*10 + minor >= 2` (else u32).
- Per conv layer (`weights.cpp:768-785`): biases, then (batch-norm layers only) scales, rolling mean, rolling variance, then weights `[filters][in/groups][kh][kw]`.
- Batch-norm folding for inference (`darknet_network.cpp:2905-2940`, `fuse_conv_batchnorm`): `ε = 1e-5`, computed in f64 then rounded to f32: `w' = w·scale/√(var+ε)`, `b' = bias − mean·scale/√(var+ε)`. The 1e-6 in `convolutional_layer.cpp` is the training path.
- A correct loader consumes the file exactly to EOF.

## Layers

- **convolutional** (`darknet_cfg.cpp:2022-2027`): `pad=1` means `size/2` per side. Default activation is `logistic`. Leaky slope 0.1 (`activations.hpp:129-133`).
- **maxpool** (`darknet_cfg.cpp:2336+`, `maxpool_layer.cpp:129,370-395`): `padding` defaults to `size-1`; output `(w + padding − size)/stride + 1`; window origin `−(padding/2)` (integer division); out-of-image taps are ignored.
- **route** (`route_layer.cpp:120-137`, `darknet_cfg.cpp:2204-2250`): negative indices are relative to the current layer, indices `≥ 0` are absolute. With `groups`/`group_id`, every source contributes channel block `group_id` of `groups` equal blocks (contiguous in CHW), concatenated.
- **upsample**: nearest neighbour, `stride` 2.

## YOLO decode and NMS

- `get_yolo_box` (`yolo_layer.cpp:54-88`): `x = (col + x')/lw`, with `x' = σ(tx)·s − 0.5·(s−1)` for `s = scale_x_y` (`yolo_layer.cpp:991`); `w = exp(clamp(tw, ±10))·anchor_w/net_w`, floored at 1e-6.
- Objectness threshold on `σ(obj)`; class probability `σ(obj)·σ(cls)`, zeroed below the threshold (`yolo_layer.cpp:1479-1515`).
- `nms_kind=greedynms` is DIoU-NMS (`box_diou`, hard-coded exponent 0.6; `box.cpp:1063-1081, 2102-2109`). `beta_nms` only affects `diounms`. `default` is plain-IoU `do_nms_sort`.
- NMS threshold 0.45 (`detector.cpp:2524`); the kind comes from the last yolo layer (`detector.cpp:2599-2606`). Sorting is per class by probability; later boxes are suppressed.

## Images

- Resize (`darknet_image.cpp:1720-1790`): separable bilinear with corner-aligned scale `(in−1)/(out−1)`; a no-op at network size. RGB, `u8/255` into planar CHW f32.
- No letterbox unless the cfg sets `letter_box` (`detector.cpp:2557-2561`).

## Training loss

- Per `[yolo]` layer, `l.cost = pow(mag_array(l.delta, outputs*batch), 2)`, and `mag_array` is `sqrt(Σx²)` (`yolo_layer.cpp:1271-1276`, `utils.cpp:1019-1029`) — so the cost is `Σδ²` over the whole batch, **not** a mean.
- For an IoU-type `iou_loss`, the cost is instead `iou_normalizer·(tot_iou_loss/count) + obj_normalizer·Σ(δ with the four box entries zeroed)²` (`yolo_layer.cpp:1231-1291`). `count` is the number of matched anchors.
- The printed network loss is the mean of the per-layer costs over the layers that have one, i.e. the mean of the two heads (`darknet_network.cpp:get_network_cost`).
- `loss.csv`'s `loss` column is that number for one iteration; `avg_loss` is an EMA `0.9·avg + 0.1·loss` seeded with the first loss (`detector.cpp:774-778`).
- [yolo] semantics that change what the delta is taken against: in training the layer applies the activation itself — logistic on `x,y` and on `obj`+classes, then `x' = s·σ(x) − 0.5(s−1)`; `w,h` stay raw logits (`yolo_layer.cpp:968-1000`). Delta is stored into a buffer zeroed at the start of every forward (`yolo_layer.cpp:1012`).

## Training deltas

- Box entries are **accumulated** (`+=`) so several truths can hit one anchor; objectness/class entries are **assigned** (`=`) (`yolo_layer.cpp:786-800, 763-771`). Order matters: objectness/class are written before the box delta accumulates.
- Objectness: default `δ_obj = obj_normalizer·(0 − σ(obj))`; if the best IoU of any truth with this box exceeds `ignore_thresh` → `δ_obj = 0`; if it exceeds `truth_thresh` → `δ_obj = obj_normalizer·(1 − σ(obj))`, and the class and box deltas are written (`yolo_layer.cpp:709-790`).
- Class: only the matched class entry is written, `δ_cls = cls_normalizer·(1 − σ(cls))` times the optional per-class multiplier (`yolo_layer.cpp:352-380, 763-770`).
- Box, for the non-MSE (CIoU/DIoU/GIoU) branch: `dx = dIoU/dt`, `dy = dIoU/db`, `dw = dIoU/dl·e^{tw}`, `dh = dIoU/dr·e^{th}` — a jacobian mapping that is deliberately **not** the "true" transpose (which is commented out above it) — each scaled by `iou_normalizer`, then NaN/inf-guarded and clipped to `±max_delta` when that is finite; `tw,th` are clamped to ±10 first (`yolo_layer.cpp:154-310`).
- The `scale` argument `(2 − truth.w·truth.h)` reaches `delta_yolo_box` but is discarded (`(void)scale`), so box deltas are *not* weighted by object size in this branch (`yolo_layer.cpp:159-160`).
- Defaults: `iou_normalizer=0.75`, `obj_normalizer=1`, `cls_normalizer=1`, `ignore_thresh=0.5`, `truth_thresh=1`, `iou_loss=mse`, `max_delta=FLT_MAX` (`darknet_cfg.cpp:2437-2468`).
- A matched anchor's box delta for `iou_loss=mse` is the classic `(target − x)·iou_normalizer` on all four entries (`yolo_layer.cpp:222-230`), which is what inference-time ports usually expect.

## Batch norm, training path

- Forward (`batchnorm_layer.cpp:106-133`): per-channel mean over `batch·spatial`; variance over `batch·spatial − 1` (**unbiased**, `blas.cpp:391-432`); rolling stats `0.9·old + 0.1·batch`; normalise with `sqrt(var + 1e-5)` (`blas.cpp:435-450`); then `y = scale·x_norm + bias`. The epsilon differs from the 1e-6 in `convolutional_layer.cpp` and from the inference fold's `1e-5`.
- Backward (`batchnorm_layer.cpp:134-147, 18-66`): `scale_updates += x_norm·δ`; `δ *= scale`; `mean_delta = −Σδ/√(var+1e-5)`; `variance_delta = −0.5·Σδ(x−mean)/(var+1e-5)^{3/2}`; then `δ = δ/(√var + 1e-5) + variance_delta·2(x−mean)/(spatial·batch) + mean_delta/(spatial·batch)`. The `√var + 1e-5` form (rather than `√(var+1e-5)`) is an inconsistency in Darknet itself — port it as written.
- Batch norm params take the SGD step without weight decay (`batchnorm_layer.cpp:148-160`).

## SGD and learning rate

- One update per iteration, after all `subdivisions` micro-batches (`darknet_network.cpp:1146-1170`); `batch` passed to the update is `net.batch·subdivisions` (`darknet_network.cpp:update_network`).
- Conv update (`convolutional_layer.cpp:2983-3005`): `updates += −decay·batch·w`; `w += (lr/batch)·updates`; `updates *= momentum`. Bias and BN `scales`: `w += (lr/batch)·updates`; `updates *= momentum`, **no** decay. So momentum buffers store batch-summed gradients, and the `1/batch` lives in the update, not in the backward.
- Per-layer `learning_rate` key multiplies the global rate (`darknet_cfg.cpp:1222`); default 1.
- Schedule (`darknet_network.cpp:711-738`): `batch_num = seen/(batch·subdivisions)` (`:589-596`); while `batch_num < burn_in`, `lr·(batch_num/burn_in)^power` with **`power` defaulting to 4** (`darknet_cfg.cpp:1678`), so early iterations have an effectively zero rate; then `policy=steps` multiplies the rate by `scales[i]` once `steps[i]` has passed (`steps[i] > batch_num` returns early).

## Training data and augmentation

- Truth layout per box: normalised `x, y, w, h, class_id, track_id`, `truth_size` floats apart, bounded by `max_boxes` (`data.cpp:911-1030`). Boxes smaller than one network pixel are dropped; `x,y` are clamped into `(0,1)`; out-of-range class ids and coordinates are fatal for Darknet (our port errors instead).
- Box order within an image is shuffled before transformation (`data.cpp:934`), so a port's delta accumulation order can differ without changing the result.
- Crop/augment geometry (`data.cpp:1510-1690`): `dw = ow·jitter`, `dh = oh·jitter`; crop offsets `rand(−dw..dw)` per side plus a resize term from `resize` (`min_rdw/max_rdw` derived from `1/resize` and `resize`); the cropped region is resized to the network size with the crop's mean colour filling any out-of-image part (`image_opencv.cpp:308-340`); optional horizontal flip (`cv::flip` with axis 1).
- Colour augmentation (`image_opencv.cpp:340-380`, `darknet_image.cpp:1618-1629`): on the **8-bit** RGB image, convert to HSV, `h += 179·dhue`, `s *= dsat`, `v *= dexp`, convert back. `dhue = rand_uniform(−hue, hue)`; `dsat = rand_scale(saturation)`, `dexp = rand_scale(exposure)`; `rand_scale(s)` is `rand_uniform(1,s)` with a coin-flip to invert it (`utils.cpp:1223-1235`). Reproducing it bit-exactly needs OpenCV's 8-bit RGB↔HSV rounding; a Rust port matches to a small tolerance instead, which is a recorded deviation and not an error.
- `flip=0`/`angle=0`/`mosaic=0`/`mixup=0`/`cutmix=0` disable their augmentations entirely; a port must reject them rather than silently apply a no-op variant.

## Training-state files

- `.weights` (`weights.cpp:1006-1023, 768-785`): the same file written during training holds the **unfolded** BN parameters (bias, scales, rolling mean, rolling variance, weights), i.e. a training checkpoint and an inference checkpoint are the same format; only `seen` (u64 when `major*10+minor >= 2`) records progress.
- Darknet's own `optimizer.state` blob (momentum buffers + loss history) has no published layout; a port may write its own resume file, documented as such, without claiming compatibility.

## Gradient routing

- Every non-conv layer's backward is an **add** into its sources' delta buffers, never an overwrite: `[yolo]` does `state.delta += l.delta` (`yolo_layer.cpp:1325-1328`); `[route]` does `axpy` into each source's delta at the source's channel block (with `groups`/`group_id`, `route_layer.cpp:140-159`). So a conv output consumed by both a head and a route accumulates gradient from both branches, and a gradient buffer must be zeroed once per iteration.
- Consequence for a port: a conv's backward starts from a delta buffer that may already hold contributions from several consumers, and `dW`/`db` are accumulated as batch sums (the `1/batch` lives in the optimizer step, see [SGD and learning rate](#sgd-and-learning-rate)).

## Trainer verification facts

Verified while building the CPU trainer (`crates/hip_darknet/src/train/`); these constrain how a port may be tested.

- `dx_box_iou` returns the gradient of the **score** (IoU/GIoU/DIoU/CIoU), not of the loss: Darknet's optimizer is additive (`w += lr * updates`), so a positive IoU gradient is the descending direction of `1 - score` (`box.cpp:1410-1595`, `convolutional_layer.cpp:2991`).
- The `x`/`y` slots (`dt`, `db`) are the exact score gradient; the `w`/`h` slots are **twice** the exact gradient on purpose ("For dw and dh, we do not divided by 2", `box.cpp:1560-1565`), and for CIoU the enclosing-box distance term enters at its true scale while the IoU term is doubled.
- The CIoU/DIoU box gradient is an approximation: `alpha`'s own dependence on the box is dropped, the aspect-ratio derivative uses `±8/π²·(atan(ar_gt) − atan(ar_pred))·{h, −w}` without the `1/(w²+h²)` denominator (`box.cpp:1552-1588`), and DIoU's `(d/c)^0.6` exponent is not differentiated. A port must reproduce these numbers; measured deviation from the exact gradient is up to ~7 % on `x/y` and larger on `w/h` for boxes with very different aspect ratios.
- Batch-norm backward is **not** the exact gradient of the BN forward: the forward variance is unbiased (`/(n−1)`, `blas.cpp:417`) but the backward's variance term divides by `n` (`batchnorm_layer.cpp:62`), and the standardisation uses `1/(√var + 1e-5)` where the forward used `1/√(var + 1e-5)`. The residual is ~`1/n` — 0.03 % at this run's batch, but it must be accounted for when comparing against numerical differentiation of small batches.
- A conv's BN shift (`l.biases` after `scale_bias`) is applied in the forward but its `bias_updates` are never filled on the BN path, so Darknet trains **no** shift for batch-normalised convs (`batchnorm_layer.cpp:148-160` vs `convolutional_layer.cpp:2890-2897`).
- `col2im` zeroes its destination before accumulating (`col2im.cpp:85`), and a conv writes `dX` into the previous layer's delta buffer, so a route's earlier-added contribution to that same buffer is discarded. This is a Darknet ordering quirk; our port **accumulates** `dX` instead (verified by numerical differentiation), which is the mathematically correct gradient for a fan-out.
