# Darknet `.cfg` + `.weights` → ONNX Export Implementation Plan

> **Status: COMPLETE (2026-10-17).** All 21 executable checklist steps are complete. Numeric checks used the available `build/logs/verify/lego_cur_cpu/run.cfg` + `run_final.weights` and `artwork/dog.jpg`, because the plan's external LegoGears dataset paths are absent in this checkout. ONNX Runtime verified fused, unfused, and a weight-compatible Mish variant at tolerance `1e-4`. This is Rust-runtime parity only, not compiled-C++ parity.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `lightnet-rs export-onnx <cfg> <weights> [out.onnx] [--no-fuse]` writes an ONNX file whose raw YOLO head outputs match the Rust CPU runtime with `max_abs / max(max_abs(reference), 1.0) <= 1e-4`, as implemented by `tools/onnx_check.py`. This is Rust-runtime parity only; do not infer compiled-C++ parity.

**Architecture:** A new module `src/lightnet-core/src/onnx/` with (1) `proto.rs`: a ~100-line hand-written protobuf wire encoder (no `prost`/`protobuf` dependency), (2) `export.rs`: walks `Network.layers` + `LayerParams` (the same structures the CPU/oxide runtimes use) and emits ONNX nodes/initializers, (3) a CLI subcommand in `lightnet-cli`. Batch-norm is folded into conv weights by default using the runtime's existing `fold_convolutional_params_for_inference`. Numeric verification is a Python script (`onnxruntime`) comparing against the existing `LIGHTNET_LAYER_DUMP_DIR` dumps.

**Tech Stack:** Rust (lightnet-core, lightnet-cli), ONNX IR 8 / opset 13 (18 when `mish` is used), Python 3 + `onnx` + `onnxruntime` + `numpy` + `opencv-python` for the check script only (user installs; none are installed today).

**Spec:** No separate spec. This document is the spec. Reference implementation to ADAPT (not copy): C++ exporter `src-onnx/darknet_onnx.cpp` on branch `master` (`git show master:src-onnx/darknet_onnx.cpp`), design notes in `git show master:README.md` (ONNX section). Per memory `port-features-adapt-not-replace`: extend the Rust code in its current style, no `tl::expected` twins, no extra cfgs/examples.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [File Structure](#file-structure)
  - [Task 1: Protobuf wire encoder + ONNX message builders](#task-1-protobuf-wire-encoder--onnx-message-builders)
  - [Task 2: Exporter core — conv (+BN fold), activations, maxpool, routes, shortcut, upsample, yolo](#task-2-exporter-core--conv-bn-fold-activations-maxpool-routes-shortcut-upsample-yolo)
  - [Task 3: `export-onnx` CLI subcommand](#task-3-export-onnx-cli-subcommand)
  - [Task 4: Numeric verification against the Rust runtime (onnxruntime)](#task-4-numeric-verification-against-the-rust-runtime-onnxruntime)
  - [Task 5: Record the work](#task-5-record-the-work)
- [Out of scope (explicit YAGNI list)](#out-of-scope-explicit-yagni-list)
- [Self-Review](#self-review)

## Global Constraints

- No new Rust dependencies (protobuf is hand-encoded).
- Builds: host crates only (`lightnet-core`, `lightnet-cli`, no `--features oxide`); one cargo command at a time, `-j 2`. Prefer `systemd-run --user --scope -p MemoryHigh=3G` or `./tools/test_host.sh <single-filter>` when the WSL user systemd bus is available. If it is unavailable, run the same Cargo command directly, uncapped, and record that limitation; never run two build/test commands concurrently.
- Never run builds with more than 4 jobs; do not launch training runs.
- ONNX: `ir_version = 8`, default `opset = 13`, `opset = 18` iff any layer uses `mish` (ONNX `Mish` op appeared in opset 18).
- Tensor layout NCHW, float32 initializers stored as `raw_data` little-endian; dynamic batch dimension named `"batch"`, fixed H/W/C from the cfg.
- Tensor names: net input `input`; layer `i` output `layer_{i:03}_out`; YOLO head outputs `yolo_{k}` (k = 0.. in cfg order).
- Commit only files this plan names. Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- `docs/` is gitignored (`git add -f`); this plan adds no docs files.

## Review Focus

1. **Maxpool padding is asymmetric**: Darknet pads `padding/2` at the start and the rest at the end (`size=2 stride=1 padding=1` → ONNX `pads=[0,0,1,1]`, NOT `[1,1,1,1]`). Expected: exported output H/W equals `layer.output_shape`.
2. **Non-1 `upsample` `scale`** multiplies values in Darknet; `reverse=1` is a downsample. Expected: `scale != 1` emits a `Mul`; `reverse` returns a clear `UnsupportedLayer` error.
3. **Mismatched-shape shortcut** (30 layers across the shipped cfgs) is a flat crop/pad in Darknet, not expressible as `Add`. Expected: clear error naming the layer, never a silently wrong graph.
4. **Grouped route** (`groups=2 group_id=1`, used by YOLOv4-tiny / LegoGears) must become a channel `Slice` using the *source layer's* channel count, not the current layer's.
5. **Dilated conv** (only `resnet152_trident`): Rust output dims ignore dilation, so exporting would be silently wrong. Expected: `UnsupportedLayer` error.
6. **Activations without an ONNX equivalent here** (`ramp`, `loggy`, `Other(..)`): error, not a silent identity (the CPU runtime silently treats unknown names as identity).
7. **Graph without YOLO layers** (classifier cfg): output must fall back to the last layer rather than an empty output list.

## File Structure

```
src/lightnet-core/src/
├── lib.rs                      MODIFY  pub mod onnx; pub use onnx::{export_onnx, OnnxOptions};
├── runtime.rs                  MODIFY  fold_convolutional_params_for_inference + BATCH_NORM_EPSILON -> pub(crate)
└── onnx/
    ├── mod.rs                  CREATE  re-exports
    ├── proto.rs                CREATE  protobuf wire encoder + ONNX message builders (+ test-only decoder)
    └── export.rs               CREATE  Exporter: layers -> nodes/initializers, export_onnx()
src/lightnet-cli/src/main.rs  MODIFY  "export-onnx" subcommand
tools/onnx_check.py             CREATE  onnxruntime vs LIGHTNET_LAYER_DUMP_DIR numeric check
tasks/rust/todo.md              MODIFY  one line pointing at this plan (final task)
```

Responsibilities: `proto.rs` knows ONNX *field numbers* only; `export.rs` knows Darknet *layer semantics* only; the CLI only parses arguments and writes bytes.

---

### Task 1: Protobuf wire encoder + ONNX message builders

**Files:**
- Create: `src/lightnet-core/src/onnx/mod.rs`
- Create: `src/lightnet-core/src/onnx/proto.rs`
- Modify: `src/lightnet-core/src/lib.rs` (add `pub mod onnx;` next to `pub mod weights;` at line 30)

**Interfaces:**
- Produces (all `pub(crate)` in `proto.rs`):
  - `struct Msg(Vec<u8>)` with `Msg::new()`, `.int(field: u32, v: i64) -> &mut Self`, `.float(field, f32)`, `.bytes(field, &[u8])`, `.string(field, &str)`, `.msg(field, &Msg)`, `.packed_ints(field, &[i64])`, `.finish(self) -> Vec<u8>`
  - `fn tensor_f32(name: &str, dims: &[i64], data: &[f32]) -> Msg`
  - `fn tensor_i64(name: &str, dims: &[i64], data: &[i64]) -> Msg`
  - `fn attr_int(name, i64) -> Msg`, `attr_float(name, f32)`, `attr_str(name, &str)`, `attr_ints(name, &[i64])`
  - `fn node(op: &str, name: &str, inputs: &[String], outputs: &[String], attrs: Vec<Msg>) -> Msg`
  - `fn value_info(name: &str, dims: &[Dim]) -> Msg` with `enum Dim { Fixed(i64), Param(&'static str) }`
  - `fn model(opset: i64, graph_name: &str, nodes: &[Msg], initializers: &[Msg], inputs: &[Msg], outputs: &[Msg]) -> Vec<u8>`
  - `#[cfg(test)] pub(crate) mod testing` with `enum Wire { Varint(u64), Len(Vec<u8>), Fixed32(u32) }` and `fn decode(bytes: &[u8]) -> Vec<(u32, Wire)>`

ONNX field numbers used (from `onnx.proto3`): ModelProto `ir_version=1 producer_name=2 opset_import=8 graph=7`; OperatorSetIdProto `domain=1 version=2`; GraphProto `node=1 name=2 initializer=5 input=11 output=12`; NodeProto `input=1 output=2 name=3 op_type=4 attribute=5`; AttributeProto `name=1 f=2 i=3 s=4 floats=7 ints=8 type=20` (type enum FLOAT=1 INT=2 STRING=3 INTS=7); TensorProto `dims=1 data_type=2 name=8 raw_data=9` (data_type FLOAT=1 INT64=7); ValueInfoProto `name=1 type=2`; TypeProto `tensor_type=1`; TypeProto.Tensor `elem_type=1 shape=2`; TensorShapeProto `dim=1`; Dimension `dim_value=1 dim_param=2`.

- [x] **Step 1: Write the failing tests** — create `src/lightnet-core/src/onnx/mod.rs`:

```rust
//! ONNX export of a Darknet network (see tasks/rust/03_plans/misc/onnx.md).
pub(crate) mod proto;
```

and `src/lightnet-core/src/onnx/proto.rs` containing only the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::testing::{Wire, decode};
    use super::*;

    #[test]
    fn varint_matches_protobuf_documentation_example() {
        // https://protobuf.dev/programming-guides/encoding/ : field 1 = 150 -> 08 96 01
        let mut m = Msg::new();
        m.int(1, 150);
        assert_eq!(m.finish(), vec![0x08, 0x96, 0x01]);
    }

    #[test]
    fn negative_int_uses_ten_byte_twos_complement() {
        let mut m = Msg::new();
        m.int(1, -1);
        let bytes = m.finish();
        assert_eq!(bytes.len(), 1 + 10);
        assert_eq!(&bytes[1..10], &[0xff; 9]);
        assert_eq!(bytes[10], 0x01);
    }

    #[test]
    fn model_round_trips_through_test_decoder() {
        let init = tensor_f32("w", &[2], &[1.0, -2.0]);
        let n = node("Relu", "relu0", &["x".into()], &["y".into()], vec![attr_float("alpha", 0.5)]);
        let input = value_info("x", &[Dim::Param("batch"), Dim::Fixed(3)]);
        let output = value_info("y", &[Dim::Param("batch"), Dim::Fixed(3)]);
        let bytes = model(13, "g", &[n], &[init], &[input], &[output]);
        let top = decode(&bytes);
        assert!(top.iter().any(|(f, w)| *f == 1 && matches!(w, Wire::Varint(8))), "ir_version 8");
        let graph = top.iter().find_map(|(f, w)| match (f, w) { (7, Wire::Len(b)) => Some(decode(b)), _ => None }).unwrap();
        let node = graph.iter().find_map(|(f, w)| match (f, w) { (1, Wire::Len(b)) => Some(decode(b)), _ => None }).unwrap();
        let op = node.iter().find_map(|(f, w)| match (f, w) { (4, Wire::Len(b)) => Some(String::from_utf8(b.clone()).unwrap()), _ => None }).unwrap();
        assert_eq!(op, "Relu");
        let init = graph.iter().find_map(|(f, w)| match (f, w) { (5, Wire::Len(b)) => Some(decode(b)), _ => None }).unwrap();
        let raw = init.iter().find_map(|(f, w)| match (f, w) { (9, Wire::Len(b)) => Some(b.clone()), _ => None }).unwrap();
        assert_eq!(raw, [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat());
    }
}
```

- [x] **Step 2: Run to verify failure**

Run: `./tools/test_host.sh varint_matches` (script from the earlier request; equivalent: `systemd-run --user --scope -p MemoryHigh=3G cargo test -p lightnet-core -j 2 -- varint_matches`)
Expected: FAIL to compile (`Msg`, `testing` not defined).

- [x] **Step 3: Implement** — put this ABOVE the test module in `proto.rs`:

```rust
//! Minimal protobuf wire encoder and the ONNX messages the exporter needs.
//! Field numbers follow onnx.proto3 (IR version 8).

pub(crate) struct Msg(Vec<u8>);

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

impl Msg {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }
    fn tag(&mut self, field: u32, wire: u8) {
        put_varint(&mut self.0, ((field as u64) << 3) | wire as u64);
    }
    pub(crate) fn int(&mut self, field: u32, v: i64) -> &mut Self {
        self.tag(field, 0);
        put_varint(&mut self.0, v as u64);
        self
    }
    pub(crate) fn float(&mut self, field: u32, v: f32) -> &mut Self {
        self.tag(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub(crate) fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.tag(field, 2);
        put_varint(&mut self.0, b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }
    pub(crate) fn string(&mut self, field: u32, s: &str) -> &mut Self {
        self.bytes(field, s.as_bytes())
    }
    pub(crate) fn msg(&mut self, field: u32, m: &Msg) -> &mut Self {
        self.bytes(field, &m.0)
    }
    pub(crate) fn packed_ints(&mut self, field: u32, vs: &[i64]) -> &mut Self {
        let mut body = Vec::new();
        for v in vs {
            put_varint(&mut body, *v as u64);
        }
        self.bytes(field, &body)
    }
    pub(crate) fn finish(self) -> Vec<u8> {
        self.0
    }
}

const FLOAT: i64 = 1;
const INT64: i64 = 7;

pub(crate) fn tensor_f32(name: &str, dims: &[i64], data: &[f32]) -> Msg {
    let mut raw = Vec::with_capacity(data.len() * 4);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut m = Msg::new();
    m.packed_ints(1, dims).int(2, FLOAT).string(8, name).bytes(9, &raw);
    m
}

pub(crate) fn tensor_i64(name: &str, dims: &[i64], data: &[i64]) -> Msg {
    let mut raw = Vec::with_capacity(data.len() * 8);
    for v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut m = Msg::new();
    m.packed_ints(1, dims).int(2, INT64).string(8, name).bytes(9, &raw);
    m
}

pub(crate) fn attr_int(name: &str, v: i64) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).int(3, v).int(20, 2);
    m
}
pub(crate) fn attr_float(name: &str, v: f32) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).float(2, v).int(20, 1);
    m
}
pub(crate) fn attr_str(name: &str, v: &str) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).string(4, v).int(20, 3);
    m
}
pub(crate) fn attr_ints(name: &str, vs: &[i64]) -> Msg {
    let mut m = Msg::new();
    m.string(1, name).packed_ints(8, vs).int(20, 7);
    m
}

pub(crate) fn node(op: &str, name: &str, inputs: &[String], outputs: &[String], attrs: Vec<Msg>) -> Msg {
    let mut m = Msg::new();
    for i in inputs {
        m.string(1, i);
    }
    for o in outputs {
        m.string(2, o);
    }
    m.string(3, name).string(4, op);
    for a in &attrs {
        m.msg(5, a);
    }
    m
}

pub(crate) enum Dim {
    Fixed(i64),
    Param(&'static str),
}

pub(crate) fn value_info(name: &str, dims: &[Dim]) -> Msg {
    let mut shape = Msg::new();
    for d in dims {
        let mut dim = Msg::new();
        match d {
            Dim::Fixed(v) => dim.int(1, *v),
            Dim::Param(p) => dim.string(2, p),
        };
        shape.msg(1, &dim);
    }
    let mut tensor = Msg::new();
    tensor.int(1, FLOAT).msg(2, &shape);
    let mut ty = Msg::new();
    ty.msg(1, &tensor);
    let mut m = Msg::new();
    m.string(1, name).msg(2, &ty);
    m
}

pub(crate) fn model(
    opset: i64,
    graph_name: &str,
    nodes: &[Msg],
    initializers: &[Msg],
    inputs: &[Msg],
    outputs: &[Msg],
) -> Vec<u8> {
    let mut graph = Msg::new();
    for n in nodes {
        graph.msg(1, n);
    }
    graph.string(2, graph_name);
    for t in initializers {
        graph.msg(5, t);
    }
    for i in inputs {
        graph.msg(11, i);
    }
    for o in outputs {
        graph.msg(12, o);
    }
    let mut opset_import = Msg::new();
    opset_import.string(1, "").int(2, opset);
    let mut m = Msg::new();
    m.int(1, 8).string(2, "lightnet-rs").msg(8, &opset_import).msg(7, &graph);
    m.finish()
}

#[cfg(test)]
pub(crate) mod testing {
    pub(crate) enum Wire {
        Varint(u64),
        Len(Vec<u8>),
        Fixed32(u32),
    }

    pub(crate) fn decode(mut b: &[u8]) -> Vec<(u32, Wire)> {
        fn varint(b: &mut &[u8]) -> u64 {
            let (mut v, mut shift) = (0u64, 0);
            loop {
                let byte = b[0];
                *b = &b[1..];
                v |= ((byte & 0x7f) as u64) << shift;
                if byte & 0x80 == 0 {
                    return v;
                }
                shift += 7;
            }
        }
        let mut out = Vec::new();
        while !b.is_empty() {
            let key = varint(&mut b);
            let field = (key >> 3) as u32;
            match key & 7 {
                0 => out.push((field, Wire::Varint(varint(&mut b)))),
                2 => {
                    let n = varint(&mut b) as usize;
                    out.push((field, Wire::Len(b[..n].to_vec())));
                    b = &b[n..];
                }
                5 => {
                    out.push((field, Wire::Fixed32(u32::from_le_bytes(b[..4].try_into().unwrap()))));
                    b = &b[4..];
                }
                other => panic!("unsupported wire type {other}"),
            }
        }
        out
    }
}
```

Add `pub mod onnx;` to `lib.rs`. Test-module imports (`use super::*;`) already cover the builders.

- [x] **Step 4: Run to verify pass**

Run these separately because `cargo test` accepts one positional test filter:

```bash
./tools/test_host.sh varint_matches
./tools/test_host.sh negative_int
./tools/test_host.sh model_round_trips
```

Expected: 1 test passes per command (3 total).

- [x] **Step 5: Commit**

```bash
git add src/lightnet-core/src/onnx src/lightnet-core/src/lib.rs
git commit -m "feat(onnx): protobuf wire encoder and ONNX message builders"
```

---

### Task 2: Exporter core — conv (+BN fold), activations, maxpool, routes, shortcut, upsample, yolo

**Files:**
- Modify: `src/lightnet-core/src/runtime.rs` (`const BATCH_NORM_EPSILON` line 37 and `fn fold_convolutional_params_for_inference` line ~7017: add `pub(crate)`)
- Create: `src/lightnet-core/src/onnx/export.rs`
- Modify: `src/lightnet-core/src/onnx/mod.rs`, `src/lightnet-core/src/lib.rs`

**Interfaces:**
- Consumes: `proto::{Msg, tensor_f32, tensor_i64, attr_*, node, value_info, Dim, model}` (Task 1); `Network { net: NetConfig{channels,height,width}, layers: Vec<Layer> }`, `Layer { input_shape, output_shape: TensorShape{batch,channels,height,width}, kind }`, `LayerParams::Convolutional(ConvolutionalParams{biases, batch_norm: Option<BatchNormParams{scales,rolling_mean,rolling_variance,biases}>, weights})`, `Error::UnsupportedLayer { layer_index, name }`.
- Produces: `pub struct OnnxOptions { pub fuse_batch_norm: bool }` (`Default` = fuse true); `pub fn export_onnx(network: &Network, params: &[LayerParams], options: &OnnxOptions) -> Result<Vec<u8>>`.

Semantics to implement (all verified against the Rust runtime):
- **Conv**: ONNX `Conv`, attrs `group=spec.groups.max(1)`, `pads=[p,p,p,p]` with `p=spec.padding`, `strides=[sy,sx]`, `dilations=[1,1]`, `kernel_shape=[size,size]`. Weights dims `[filters, in_c/groups, size, size]` (Darknet order is identical). `dilation != 1` or `share_index.is_some()` → `UnsupportedLayer`. Fused: `(bias, weights) = fold_convolutional_params_for_inference(spec, params, in_c)`; unfused with BN: `Conv`(no bias) → `BatchNormalization(epsilon = 1e-5)` with inputs `[x, scales, biases(beta), rolling_mean, rolling_variance]`.
- **Activation** after conv/shortcut: `Linear` nothing; `Leaky` → `LeakyRelu alpha=0.1`; `Logistic` → `Sigmoid`; `Relu` → `Relu`; `Tanh` → `Tanh`; `Mish` → `Mish` (opset 18); `Swish` → `Sigmoid` then `Mul(x, sigmoid(x))`; `Relu6` → `Clip` with initializer inputs `min=0.0`, `max=6.0` (opset 13 takes them as inputs); `Loggy`/`Ramp`/`Other` → `UnsupportedLayer`.
- **MaxPool**: `kernel_shape=[size,size]`, `strides=[sy,sx]`, `pads=[pad/2, pad/2, pad-pad/2, pad-pad/2]` (ONNX order `[y_begin, x_begin, y_end, x_end]`) where `pad = spec.padding`, `ceil_mode=0`.
- **Route**: layer indices absolute via `idx = if l < 0 { i as isize + l } else { l }`. One source and `groups==1` → alias (no node). One source and `groups>1` → `Slice` over axis 1 with `starts = group_id*c/groups`, `ends = starts + c/groups` where `c = layers[src].output_shape.channels` (initializers as int64 `[1]` tensors). Two or more sources and `groups==1` → `Concat axis=1`. Anything else → `UnsupportedLayer`.
- **Shortcut**: for each `from` source (absolute index) require `layers[src].output_shape == layer.output_shape` else `UnsupportedLayer`; chain `Add(prev, src)`; then the shortcut's activation.
- **Upsample**: `reverse` → `UnsupportedLayer`. `Resize` (opset 13 form: inputs `[x, roi, scales]`, `roi` = float tensor dims `[0]`, `scales = [1,1,stride,stride]`), attrs `mode="nearest"`, `coordinate_transformation_mode="asymmetric"`, `nearest_mode="floor"`. `scale != 1.0` → extra `Mul` by a scalar float initializer.
- **Yolo**: `Identity` from the previous output to `yolo_{k}`; recorded as a graph output with the layer's `output_shape` (`[batch, C, H, W]`). If the network has no yolo layer, the last layer's output is the single graph output.
- Everything else (`Connected`, `Softmax`, `AvgPool`, ...) → `UnsupportedLayer { layer_index, name: kind.name() }`.

- [x] **Step 1: Write the failing tests** — in `export.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::onnx::proto::testing::{Wire, decode};
    use crate::runtime::{ExecutableNetwork, LayerParams};

    const CFG: &str = r#"
[net]
width=16
height=16
channels=3
batch=1

[convolutional]
batch_normalize=1
filters=4
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=1

[convolutional]
filters=4
size=1
stride=1
pad=1
activation=linear

[route]
layers=-1
groups=2
group_id=1

[convolutional]
filters=8
size=1
stride=1
pad=1
activation=linear

[route]
layers=-1,-3

[upsample]
stride=2

[yolo]
mask=0
anchors=2,3
classes=1
num=1
"#;

    fn exec() -> ExecutableNetwork {
        let network = Network::from_config(ConfigFile::parse_str("t.cfg", CFG).unwrap()).unwrap();
        let mut e = ExecutableNetwork::new(network);
        for p in &mut e.params {
            if let LayerParams::Convolutional(c) = p {
                for (i, w) in c.weights.iter_mut().enumerate() {
                    *w = ((i % 7) as f32 - 3.0) * 0.1;
                }
                if let Some(bn) = c.batch_norm.as_mut() {
                    bn.scales.iter_mut().for_each(|v| *v = 1.5);
                    bn.rolling_mean.iter_mut().for_each(|v| *v = 0.25);
                    bn.rolling_variance.iter_mut().for_each(|v| *v = 2.0);
                }
            }
        }
        e
    }

    /// Returns (op_types, node input lists, node attribute (name,ints) lists, initializer names, output names)
    fn summarize(bytes: &[u8]) -> Summary {
        let top = decode(bytes);
        let graph = top.iter().find_map(|(f, w)| match (f, w) { (7, Wire::Len(b)) => Some(decode(b)), _ => None }).unwrap();
        let mut s = Summary::default();
        for (f, w) in graph {
            let Wire::Len(b) = w else { continue };
            let text = |fields: &[(u32, Wire)], id: u32| -> Vec<String> {
                fields.iter().filter_map(|(f, w)| match (f, w) { (x, Wire::Len(b)) if *x == id => Some(String::from_utf8_lossy(b).into_owned()), _ => None }).collect()
            };
            let fields = decode(&b);
            match f {
                1 => {
                    s.ops.push(text(&fields, 4).remove(0));
                    s.inputs.push(text(&fields, 1));
                    let mut attrs = Vec::new();
                    for (af, aw) in &fields {
                        if let (5, Wire::Len(ab)) = (af, aw) {
                            let a = decode(ab);
                            let name = text(&a, 1).remove(0);
                            let ints = a.iter().find_map(|(x, w)| match (x, w) { (8, Wire::Len(p)) => Some(unpack(p)), _ => None }).unwrap_or_default();
                            attrs.push((name, ints));
                        }
                    }
                    s.attrs.push(attrs);
                    s.node_outputs.extend(text(&fields, 2));
                }
                5 => {
                    let name = text(&fields, 8).into_iter().next().unwrap_or_default();
                    s.initializers.push(name.clone());
                    let is_i64 = fields.iter().any(|(field, wire)| *field == 2 && matches!(wire, Wire::Varint(7)));
                    if is_i64 {
                        if let Some(Wire::Len(raw)) = fields.iter().find_map(|(field, wire)| if *field == 9 { Some(wire) } else { None }) {
                            let values = raw.chunks_exact(8).map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap())).collect();
                            s.int_initializers.push((name, values));
                        }
                    }
                }
                12 => s.outputs.extend(text(&fields, 1)),
                _ => {}
            }
        }
        s
    }
    fn unpack(mut b: &[u8]) -> Vec<i64> {
        let mut out = Vec::new();
        while !b.is_empty() {
            let (mut v, mut sh) = (0u64, 0);
            loop { let x = b[0]; b = &b[1..]; v |= ((x & 0x7f) as u64) << sh; if x & 0x80 == 0 { break } sh += 7; }
            out.push(v as i64);
        }
        out
    }
    #[derive(Default)]
    struct Summary { ops: Vec<String>, inputs: Vec<Vec<String>>, attrs: Vec<Vec<(String, Vec<i64>)>>, initializers: Vec<String>, int_initializers: Vec<(String, Vec<i64>)>, node_outputs: Vec<String>, outputs: Vec<String> }

    #[test]
    fn exports_expected_op_sequence_and_topological_graph() {
        let e = exec();
        let bytes = export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap();
        let s = summarize(&bytes);
        assert_eq!(s.ops, ["Conv", "LeakyRelu", "MaxPool", "Conv", "Slice", "Conv", "Concat", "Resize", "Identity"]);
        assert_eq!(s.outputs, ["yolo_0"]);
        // every node input is the net input, an initializer, or an earlier node output
        let mut known: Vec<&str> = vec!["input"];
        known.extend(s.initializers.iter().map(String::as_str));
        for (i, ins) in s.inputs.iter().enumerate() {
            for name in ins {
                assert!(known.contains(&name.as_str()), "node {i} ({}) uses undefined {name}", s.ops[i]);
            }
            // Every exported node currently has exactly one output.
            known.push(s.node_outputs[i].as_str());
        }
    }

    #[test]
    fn maxpool_padding_is_asymmetric_like_darknet() {
        let e = exec();
        let s = summarize(&export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap());
        let idx = s.ops.iter().position(|o| o == "MaxPool").unwrap();
        let pads = &s.attrs[idx].iter().find(|(n, _)| n == "pads").unwrap().1;
        assert_eq!(pads, &[0, 0, 1, 1]);
    }

    #[test]
    fn grouped_route_slices_source_channels() {
        let e = exec();
        let bytes = export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap();
        let s = summarize(&bytes);
        let starts = s.int_initializers.iter().find(|(n, _)| n.ends_with("_starts")).unwrap();
        let ends = s.int_initializers.iter().find(|(n, _)| n.ends_with("_ends")).unwrap();
        // The source layer has four channels; group_id=1 selects [2, 4).
        assert_eq!(starts.1, [2]);
        assert_eq!(ends.1, [4]);
    }

    #[test]
    fn no_fuse_emits_batch_normalization_node() {
        let e = exec();
        let s = summarize(&export_onnx(&e.network, &e.params, &OnnxOptions { fuse_batch_norm: false }).unwrap());
        assert_eq!(&s.ops[..3], ["Conv", "BatchNormalization", "LeakyRelu"]);
    }

    #[test]
    fn unsupported_constructs_are_errors_not_silent() {
        for (needle, replacement) in [
            ("[upsample]\nstride=2", "[upsample]\nstride=2\nreverse=1"),
            ("activation=leaky", "activation=ramp"),
        ] {
            let cfg = CFG.replace(needle, replacement);
            let network = Network::from_config(ConfigFile::parse_str("t.cfg", &cfg).unwrap()).unwrap();
            let e = ExecutableNetwork::new(network);
            let err = export_onnx(&e.network, &e.params, &OnnxOptions::default());
            assert!(err.is_err(), "{replacement} must be rejected");
        }
    }

    #[test]
    fn upsample_nonunit_scale_emits_mul() {
        let cfg = CFG.replace("[upsample]\nstride=2", "[upsample]\nstride=2\nscale=2");
        let network = Network::from_config(ConfigFile::parse_str("t.cfg", &cfg).unwrap()).unwrap();
        let e = ExecutableNetwork::new(network);
        let s = summarize(&export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap());
        assert!(s.ops.contains(&"Mul".to_owned()));
    }

    #[test]
    fn mismatched_shortcut_is_unsupported() {
        let cfg = r#"
[net]
width=16
height=16
channels=3
batch=1

[convolutional]
filters=4
size=1
stride=1
pad=1
activation=linear

[maxpool]
size=2
stride=2

[shortcut]
from=-2
activation=linear
"#;
        let network = Network::from_config(ConfigFile::parse_str("shortcut.cfg", cfg).unwrap()).unwrap();
        let e = ExecutableNetwork::new(network);
        assert!(export_onnx(&e.network, &e.params, &OnnxOptions::default()).is_err());
    }

    #[test]
    fn dilated_conv_is_unsupported() {
        let cfg = CFG.replace("filters=4\nsize=3\nstride=1", "filters=4\nsize=3\nstride=1\ndilation=2");
        let network = Network::from_config(ConfigFile::parse_str("dilated.cfg", &cfg).unwrap()).unwrap();
        let e = ExecutableNetwork::new(network);
        assert!(export_onnx(&e.network, &e.params, &OnnxOptions::default()).is_err());
    }

    #[test]
    fn no_yolo_network_uses_last_layer_output() {
        let cfg = r#"
[net]
width=16
height=16
channels=3
batch=1

[convolutional]
filters=4
size=1
stride=1
pad=1
activation=linear
"#;
        let network = Network::from_config(ConfigFile::parse_str("bare.cfg", cfg).unwrap()).unwrap();
        let e = ExecutableNetwork::new(network);
        let s = summarize(&export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap());
        assert_eq!(s.outputs.len(), 1);
        assert_eq!(s.outputs, [s.node_outputs.last().unwrap()]);
    }

    #[test]
    fn mish_selects_opset_18() {
        let cfg = CFG.replace("activation=leaky", "activation=mish");
        let network = Network::from_config(ConfigFile::parse_str("t.cfg", &cfg).unwrap()).unwrap();
        let e = ExecutableNetwork::new(network);
        let bytes = export_onnx(&e.network, &e.params, &OnnxOptions::default()).unwrap();
        let top = decode(&bytes);
        let opset = top.iter().find_map(|(f, w)| match (f, w) { (8, Wire::Len(b)) => Some(decode(b)), _ => None }).unwrap();
        assert!(opset.iter().any(|(f, w)| *f == 2 && matches!(w, Wire::Varint(18))));
    }
}
```

(If a cfg key in the fixture is rejected by the parser — e.g. `reverse` or `ramp` handling — adjust the *fixture string only*, keeping the intent: one reversed upsample, one `ramp` activation.)

- [x] **Step 2: Run to verify failure**

Run: `./tools/test_host.sh exports_expected`
Expected: FAIL to compile (`export_onnx`, `OnnxOptions` missing). This single filter is sufficient to verify the new exporter API is absent.

- [x] **Step 3: Implement**

In `runtime.rs` change `const BATCH_NORM_EPSILON` and `fn fold_convolutional_params_for_inference` to `pub(crate)`. In `onnx/mod.rs` add `mod export;` and `pub use export::{OnnxOptions, export_onnx};`; in `lib.rs` add `pub use onnx::{OnnxOptions, export_onnx};`.

`export.rs` implementation (above the tests):

```rust
//! Walks a loaded `Network` + `LayerParams` and emits an ONNX graph.
use crate::error::{Error, Result};
use crate::model::{Activation, ConvolutionalSpec, Layer, LayerKind, MaxPoolSpec, Network, RouteSpec, ShortcutSpec, TensorShape, UpsampleSpec};
use crate::onnx::proto::*;
use crate::runtime::{BATCH_NORM_EPSILON, LayerParams, fold_convolutional_params_for_inference};

#[derive(Clone, Copy, Debug)]
pub struct OnnxOptions {
    /// Fold batch-norm into the conv weights/bias (default). `false` emits a
    /// separate `BatchNormalization` node (C++ `dontfuse`).
    pub fuse_batch_norm: bool,
}
impl Default for OnnxOptions {
    fn default() -> Self {
        Self { fuse_batch_norm: true }
    }
}

struct Exporter<'a> {
    network: &'a Network,
    params: &'a [LayerParams],
    options: OnnxOptions,
    nodes: Vec<Msg>,
    initializers: Vec<Msg>,
    outs: Vec<String>,
    yolo_outputs: Vec<(String, TensorShape)>,
}

fn unsupported(layer_index: usize, what: &str) -> Error {
    Error::UnsupportedLayer { layer_index, name: what.to_owned() }
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

pub fn export_onnx(network: &Network, params: &[LayerParams], options: &OnnxOptions) -> Result<Vec<u8>> {
    if params.len() != network.layers.len() {
        return Err(Error::Weights("parameter count does not match layer count".to_owned()));
    }
    let uses_mish = network.layers.iter().any(|l| {
        matches!(&l.kind, LayerKind::Convolutional(s) if s.activation == Activation::Mish)
            || matches!(&l.kind, LayerKind::Shortcut(s) if s.activation == Activation::Mish)
    });
    let opset = if uses_mish { 18 } else { 13 };
    let mut e = Exporter { network, params, options: *options, nodes: vec![], initializers: vec![], outs: vec![], yolo_outputs: vec![] };
    for (i, layer) in network.layers.iter().enumerate() {
        let out = match &layer.kind {
            LayerKind::Convolutional(s) => e.conv(i, layer, s)?,
            LayerKind::MaxPool(s) => e.maxpool(i, s),
            LayerKind::Route(s) => e.route(i, s)?,
            LayerKind::Shortcut(s) => e.shortcut(i, layer, s)?,
            LayerKind::Upsample(s) => e.upsample(i, s)?,
            LayerKind::Yolo(_) => e.yolo(i, layer),
            other => return Err(unsupported(i, other.name())),
        };
        e.outs.push(out);
    }
    let dims = |s: &TensorShape| [Dim::Param("batch"), Dim::Fixed(s.channels as i64), Dim::Fixed(s.height as i64), Dim::Fixed(s.width as i64)];
    let net = &network.net;
    let input = value_info("input", &[Dim::Param("batch"), Dim::Fixed(net.channels as i64), Dim::Fixed(net.height as i64), Dim::Fixed(net.width as i64)]);
    let outputs: Vec<Msg> = if e.yolo_outputs.is_empty() {
        let last = network.layers.last().ok_or_else(|| Error::InvalidConfig("network has no layers".into()))?;
        vec![value_info(e.outs.last().unwrap(), &dims(&last.output_shape))]
    } else {
        e.yolo_outputs.iter().map(|(n, s)| value_info(n, &dims(s))).collect()
    };
    Ok(model(opset, "darknet", &e.nodes, &e.initializers, &[input], &outputs))
}

impl Exporter<'_> {
    fn input_of(&self, i: usize) -> String {
        if i == 0 { "input".to_owned() } else { self.outs[i - 1].clone() }
    }
    fn absolute(i: usize, l: isize) -> usize {
        (if l < 0 { i as isize + l } else { l }) as usize
    }
    fn add_f32(&mut self, name: &str, dims: &[i64], data: &[f32]) {
        self.initializers.push(tensor_f32(name, dims, data));
    }
    fn add_node(&mut self, op: &str, name: &str, inputs: &[String], out: &str, attrs: Vec<Msg>) {
        self.nodes.push(node(op, name, inputs, &[out.to_owned()], attrs));
    }

    fn conv(&mut self, i: usize, layer: &Layer, spec: &ConvolutionalSpec) -> Result<String> {
        if spec.dilation != 1 || spec.share_index.is_some() {
            return Err(unsupported(i, "convolutional with dilation/share_index"));
        }
        let LayerParams::Convolutional(p) = &self.params[i] else {
            return Err(unsupported(i, "convolutional without parameters"));
        };
        let in_c = layer.input_shape.channels;
        let groups = spec.groups.max(1);
        let name = format!("layer_{i:03}");
        let w_dims = [spec.filters as i64, (in_c / groups) as i64, spec.size as i64, spec.size as i64];
        let pad = spec.padding as i64;
        let attrs = vec![
            attr_int("group", groups as i64),
            attr_ints("pads", &[pad, pad, pad, pad]),
            attr_ints("strides", &[spec.stride_y as i64, spec.stride_x as i64]),
            attr_ints("dilations", &[1, 1]),
            attr_ints("kernel_shape", &[spec.size as i64, spec.size as i64]),
        ];
        let x = self.input_of(i);
        let canonical_linear = matches!(&spec.activation, Activation::Linear);
        let conv_out = if canonical_linear { format!("{name}_out") } else { format!("{name}_conv") };
        let out = if p.batch_norm.is_some() && !self.options.fuse_batch_norm {
            let bn = p.batch_norm.as_ref().unwrap();
            self.add_f32(&format!("{name}_W"), &w_dims, &p.weights);
            self.add_node("Conv", &format!("{name}_conv"), &[x, format!("{name}_W")], &conv_out, attrs);
            let f = [spec.filters as i64];
            self.add_f32(&format!("{name}_bn_scale"), &f, &bn.scales);
            self.add_f32(&format!("{name}_bn_B"), &f, &p.biases);
            self.add_f32(&format!("{name}_bn_mean"), &f, &bn.rolling_mean);
            self.add_f32(&format!("{name}_bn_var"), &f, &bn.rolling_variance);
            let bn_out = if canonical_linear { format!("{name}_out") } else { format!("{name}_bn") };
            self.add_node(
                "BatchNormalization",
                &bn_out.clone(),
                &[conv_out, format!("{name}_bn_scale"), format!("{name}_bn_B"), format!("{name}_bn_mean"), format!("{name}_bn_var")],
                &bn_out,
                vec![attr_float("epsilon", BATCH_NORM_EPSILON)],
            );
            bn_out
        } else {
            let (bias, weights) = fold_convolutional_params_for_inference(spec, p, in_c)?;
            self.add_f32(&format!("{name}_W"), &w_dims, &weights);
            self.add_f32(&format!("{name}_B"), &[spec.filters as i64], &bias);
            self.add_node("Conv", &format!("{name}_conv"), &[x, format!("{name}_W"), format!("{name}_B")], &conv_out, attrs);
            conv_out
        };
        self.activation(i, &spec.activation, out)
    }

    fn activation(&mut self, i: usize, activation: &Activation, x: String) -> Result<String> {
        let name = format!("layer_{i:03}_act");
        let out = format!("layer_{i:03}_out");
        match activation {
            Activation::Linear => {
                // keep the canonical per-layer name without adding a node
                return Ok(x);
            }
            Activation::Leaky => self.add_node("LeakyRelu", &name, &[x], &out, vec![attr_float("alpha", 0.1)]),
            Activation::Logistic => self.add_node("Sigmoid", &name, &[x], &out, vec![]),
            Activation::Relu => self.add_node("Relu", &name, &[x], &out, vec![]),
            Activation::Tanh => self.add_node("Tanh", &name, &[x], &out, vec![]),
            Activation::Mish => self.add_node("Mish", &name, &[x], &out, vec![]),
            Activation::Swish => {
                let s = format!("{name}_sigmoid");
                self.add_node("Sigmoid", &s, &[x.clone()], &s, vec![]);
                self.add_node("Mul", &name, &[x, s], &out, vec![]);
            }
            Activation::Relu6 => {
                self.add_f32(&format!("{name}_min"), &[], &[0.0]);
                self.add_f32(&format!("{name}_max"), &[], &[6.0]);
                self.add_node("Clip", &name, &[x, format!("{name}_min"), format!("{name}_max")], &out, vec![]);
            }
            other => return Err(unsupported(i, &format!("activation {other:?}"))),
        }
        Ok(out)
    }

    fn maxpool(&mut self, i: usize, spec: &MaxPoolSpec) -> String {
        let pad = spec.padding as i64;
        let (begin, end) = (pad / 2, pad - pad / 2);
        let out = format!("layer_{i:03}_out");
        let x = self.input_of(i);
        self.add_node(
            "MaxPool",
            &format!("layer_{i:03}_maxpool"),
            &[x],
            &out,
            vec![
                attr_int("ceil_mode", 0),
                attr_ints("pads", &[begin, begin, end, end]),
                attr_ints("kernel_shape", &[spec.size as i64, spec.size as i64]),
                attr_ints("strides", &[spec.stride_y as i64, spec.stride_x as i64]),
            ],
        );
        out
    }

    fn route(&mut self, i: usize, spec: &RouteSpec) -> Result<String> {
        let sources: Vec<usize> = spec.layers.iter().map(|l| Self::absolute(i, *l)).collect();
        let out = format!("layer_{i:03}_out");
        match (sources.len(), spec.groups) {
            (1, 1) => Ok(self.outs[sources[0]].clone()),
            (1, groups) => {
                if spec.group_id >= groups {
                    return Err(unsupported(i, "route group_id out of range"));
                }
                let channels = self.network.layers[sources[0]].output_shape.channels;
                let per = channels / groups;
                let (start, end) = ((spec.group_id * per) as i64, ((spec.group_id + 1) * per) as i64);
                let name = format!("layer_{i:03}_slice");
                for (suffix, value) in [("starts", start), ("ends", end), ("axes", 1), ("steps", 1)] {
                    self.initializers.push(tensor_i64(&format!("{name}_{suffix}"), &[1], &[value]));
                }
                let inputs = [self.outs[sources[0]].clone(), format!("{name}_starts"), format!("{name}_ends"), format!("{name}_axes"), format!("{name}_steps")];
                self.add_node("Slice", &name, &inputs, &out, vec![]);
                Ok(out)
            }
            (n, 1) if n >= 2 => {
                let inputs: Vec<String> = sources.iter().map(|s| self.outs[*s].clone()).collect();
                self.add_node("Concat", &format!("layer_{i:03}_concat"), &inputs, &out, vec![attr_int("axis", 1)]);
                Ok(out)
            }
            _ => Err(unsupported(i, "route with several sources and groups > 1")),
        }
    }

    fn shortcut(&mut self, i: usize, layer: &Layer, spec: &ShortcutSpec) -> Result<String> {
        let mut acc = self.input_of(i);
        for (k, from) in spec.from.iter().enumerate() {
            let src = Self::absolute(i, *from);
            if self.network.layers[src].output_shape != layer.output_shape {
                return Err(unsupported(i, "shortcut with mismatched source shape (flat crop/pad)"));
            }
            let out = format!("layer_{i:03}_add{k}");
            self.add_node("Add", &out.clone(), &[acc, self.outs[src].clone()], &out, vec![]);
            acc = out;
        }
        self.activation(i, &spec.activation, acc)
    }

    fn upsample(&mut self, i: usize, spec: &UpsampleSpec) -> Result<String> {
        if spec.reverse {
            return Err(unsupported(i, "upsample reverse (downsample)"));
        }
        let name = format!("layer_{i:03}_resize");
        let s = spec.stride as f32;
        self.add_f32(&format!("{name}_roi"), &[0], &[]);
        self.add_f32(&format!("{name}_scales"), &[4], &[1.0, 1.0, s, s]);
        let resized = if spec.scale == 1.0 { format!("layer_{i:03}_out") } else { format!("{name}_out") };
        let x = self.input_of(i);
        self.add_node(
            "Resize",
            &name,
            &[x, format!("{name}_roi"), format!("{name}_scales")],
            &resized,
            vec![attr_str("mode", "nearest"), attr_str("coordinate_transformation_mode", "asymmetric"), attr_str("nearest_mode", "floor")],
        );
        if spec.scale == 1.0 {
            return Ok(resized);
        }
        let out = format!("layer_{i:03}_out");
        self.add_f32(&format!("{name}_scale"), &[], &[spec.scale]);
        self.add_node("Mul", &format!("{name}_mul"), &[resized, format!("{name}_scale")], &out, vec![]);
        Ok(out)
    }

    fn yolo(&mut self, i: usize, layer: &Layer) -> String {
        let out = format!("yolo_{}", self.yolo_outputs.len());
        let x = self.input_of(i);
        self.add_node("Identity", &format!("layer_{i:03}_yolo"), &[x], &out, vec![]);
        self.yolo_outputs.push((out.clone(), layer.output_shape));
        out
    }
}
```

Implementation notes: `names()` helper is unused — delete it if clippy complains. `TensorShape` must be `Copy + PartialEq`; if it is not `Copy`, clone it in `yolo`. If `Activation` lacks `PartialEq` for `== Activation::Mish`, use `matches!(.., Activation::Mish)`. A zero-length `roi` initializer is intentional (ONNX requires the input to exist).

- [x] **Step 4: Run to verify pass**

Run each focused test separately (one filter per invocation):

```bash
./tools/test_host.sh exports_expected
./tools/test_host.sh maxpool_padding
./tools/test_host.sh grouped_route
./tools/test_host.sh no_fuse
./tools/test_host.sh unsupported_constructs
./tools/test_host.sh mish_selects
./tools/test_host.sh upsample_nonunit_scale
./tools/test_host.sh mismatched_shortcut_is_unsupported
./tools/test_host.sh dilated_conv_is_unsupported
./tools/test_host.sh no_yolo_network_uses_last_layer_output
./tools/test_host.sh onnx
```

Expected: each focused filter passes its matching test, then the `onnx` filter passes all exporter tests.

- [x] **Step 5: Commit**

```bash
git add src/lightnet-core/src/onnx src/lightnet-core/src/lib.rs src/lightnet-core/src/runtime.rs
git commit -m "feat(onnx): export conv/maxpool/route/shortcut/upsample/yolo layers to ONNX"
```

---

### Task 3: `export-onnx` CLI subcommand

**Files:**
- Modify: `src/lightnet-cli/src/main.rs` (new match arm after `"detect-image"` block; usage/help text if there is a list of subcommands — grep `bench-train` to find it)
- Test: `src/lightnet-cli/src/main.rs` `mod tests`

**Interfaces:**
- Consumes: `lightnet_core::{export_onnx, OnnxOptions, Network, ExecutableNetwork}` (`ExecutableNetwork::new(network)`, `.load_weights_file(path)`, pub fields `.network`, `.params`).
- Produces: CLI `lightnet-rs export-onnx <cfg> <weights> [out.onnx] [--no-fuse]`; helper `fn parse_export_onnx_args(args: Vec<String>) -> Result<ExportOnnxArgs, String>`.

- [x] **Step 1: Write the failing test** (in `main.rs` tests):

```rust
#[test]
fn export_onnx_args_parse_defaults_and_flags() {
    let a = super::parse_export_onnx_args(vec!["m.cfg".into(), "m.weights".into()]).unwrap();
    assert_eq!(a.out, std::path::PathBuf::from("m.onnx"));
    assert!(a.fuse);
    let b = super::parse_export_onnx_args(vec!["m.cfg".into(), "m.weights".into(), "o.onnx".into(), "--no-fuse".into()]).unwrap();
    assert_eq!(b.out, std::path::PathBuf::from("o.onnx"));
    assert!(!b.fuse);
    assert!(super::parse_export_onnx_args(vec!["m.cfg".into()]).is_err());
    assert!(super::parse_export_onnx_args(vec!["a.cfg".into(), "b.weights".into(), "--bogus".into()]).is_err());
}
```

- [x] **Step 2: Run to verify failure** — `./tools/test_host.sh export_onnx_args` → FAIL (no such function).

- [x] **Step 3: Implement**

```rust
struct ExportOnnxArgs {
    cfg: String,
    weights: String,
    out: PathBuf,
    fuse: bool,
}

fn parse_export_onnx_args(args: Vec<String>) -> Result<ExportOnnxArgs, String> {
    let mut positional = Vec::new();
    let mut fuse = true;
    for arg in args {
        match arg.as_str() {
            "--no-fuse" | "-dontfuse" => fuse = false,
            other if other.starts_with('-') => return Err(format!("unknown export-onnx flag: {other}")),
            _ => positional.push(arg),
        }
    }
    let mut it = positional.into_iter();
    let cfg = it.next().ok_or("missing required argument: cfg path")?;
    let weights = it.next().ok_or("missing required argument: weights path")?;
    let out = it.next().map(PathBuf::from).unwrap_or_else(|| PathBuf::from(&cfg).with_extension("onnx"));
    Ok(ExportOnnxArgs { cfg, weights, out, fuse })
}
```

and the match arm:

```rust
"export-onnx" => {
    let parsed = parse_export_onnx_args(args.collect())?;
    let network = Network::from_path(&parsed.cfg)?.with_batch(1);
    let mut executable = ExecutableNetwork::new(network);
    executable.load_weights_file(&parsed.weights)?;
    let bytes = lightnet_core::export_onnx(
        &executable.network,
        &executable.params,
        &lightnet_core::OnnxOptions { fuse_batch_norm: parsed.fuse },
    )?;
    fs::write(&parsed.out, &bytes)?;
    println!("wrote {} ({} bytes, batch_norm {})", parsed.out.display(), bytes.len(), if parsed.fuse { "fused" } else { "separate" });
}
```

(Adjust `args.collect()` to however the surrounding code iterates arguments — the existing arms use `args.next()` on the same iterator.)

- [x] **Step 4: Run to verify pass** — `./tools/test_host.sh export_onnx_args` → 1 passed. Build the host CLI before the smoke test:

```bash
cargo build -p lightnet-cli -j 2
D=/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2
target/debug/lightnet-rs export-onnx "$D/LegoGears_v2.cfg" "$D/LegoGears_augmatch_best.weights" /tmp/lego.onnx
test -s /tmp/lego.onnx
```

Expected: a clear `wrote ...` message and a non-empty `/tmp/lego.onnx`.

- [x] **Step 5: Commit**

```bash
git add src/lightnet-cli/src/main.rs
git commit -m "feat(cli): export-onnx subcommand"
```

---

### Task 4: Numeric verification against the Rust runtime (onnxruntime)

**Files:**
- Create: `tools/onnx_check.py`

**Interfaces:**
- Consumes: `lightnet-rs export-onnx`, `lightnet-rs detect-image` with `LIGHTNET_LAYER_DUMP_DIR` (writes `layers.tsv` + `layer_NNN.f32`, raw head for yolo layers), `onnxruntime`.
- Produces: exit code 0 iff every `yolo_k` output matches its dumped layer within tolerance.

Prerequisite (user installs, not done by this plan): `pip install onnx onnxruntime numpy opencv-python`.

- [x] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Check an exported ONNX model against the Rust runtime's own layer dumps.

Usage: tools/onnx_check.py CFG WEIGHTS NAMES IMAGE [--binary target/release/lightnet-rs] [--no-fuse] [--tol 1e-4]
Runs `export-onnx`, runs `detect-image` with LIGHTNET_LAYER_DUMP_DIR (preprocess
resize-exact = cv2 INTER_LINEAR_EXACT, same as `detector map`), feeds the same
pixels to onnxruntime and compares every yolo head.
"""
import argparse, os, re, subprocess, sys, tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def yolo_layer_indices(cfg: Path) -> list[int]:
    sections = re.findall(r"^\s*\[([^\]]+)\]", cfg.read_text(), flags=re.M)
    layers = [s.strip().lower() for s in sections if s.strip().lower() != "net"]
    return [i for i, s in enumerate(layers) if s == "yolo"]


def net_size(cfg: Path) -> tuple[int, int]:
    text = cfg.read_text()
    w = int(re.search(r"^\s*width\s*=\s*(\d+)", text, flags=re.M).group(1))
    h = int(re.search(r"^\s*height\s*=\s*(\d+)", text, flags=re.M).group(1))
    return w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=Path)
    ap.add_argument("weights", type=Path)
    ap.add_argument("names", type=Path)
    ap.add_argument("image", type=Path)
    ap.add_argument("--binary", default="target/release/lightnet-rs")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        dump = Path(tmp) / "dump"
        dump.mkdir()
        cmd = [args.binary, "export-onnx", str(args.cfg), str(args.weights), str(onnx_path)]
        if args.no_fuse:
            cmd.append("--no-fuse")
        subprocess.run(cmd, check=True)
        env = dict(os.environ, LIGHTNET_LAYER_DUMP_DIR=str(dump))
        subprocess.run(
            [args.binary, "detect-image", str(args.cfg), str(args.weights), str(args.names),
             str(args.image), "0.25", "0.45", "resize-exact"],
            check=True, env=env, stdout=subprocess.DEVNULL)

        w, h = net_size(args.cfg)
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        x = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = [o.name for o in session.get_outputs()]
        outputs = session.run(None, {"input": x})
        indices = yolo_layer_indices(args.cfg)
        if len(indices) != len(names):
            print(f"expected {len(indices)} yolo outputs, model has {names}", file=sys.stderr)
            return 1
        worst = 0.0
        for name, layer_index, got in zip(names, indices, outputs):
            ref = np.fromfile(dump / f"layer_{layer_index:03}.f32", dtype=np.float32).reshape(got.shape)
            err = float(np.max(np.abs(got - ref)))
            scale = max(float(np.max(np.abs(ref))), 1.0)
            print(f"{name} (layer {layer_index}): shape {got.shape} max_abs {err:.3e} rel_to_max {err / scale:.3e}")
            worst = max(worst, err / scale)
        print("OK" if worst <= args.tol else "FAIL", f"(worst {worst:.3e}, tol {args.tol:g})")
        return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: Syntax check** — `python3 -m py_compile tools/onnx_check.py` → no output.

- [x] **Step 3: Run on LegoGears_v2 (needs the deps installed; no GPU required, Rust CPU takes ~40 s per image)**

```bash
cargo build --release -p lightnet-cli -j 2   # wrap with systemd-run when available; otherwise run uncapped and record it
D=/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2
IMG="$D/set_03/DSCN2167_frame_000238.jpg"
python3 tools/onnx_check.py "$D/LegoGears_v2.cfg" "$D/LegoGears_augmatch_best.weights" "$D/LegoGears.names" "$IMG"
python3 tools/onnx_check.py "$D/LegoGears_v2.cfg" "$D/LegoGears_augmatch_best.weights" "$D/LegoGears.names" "$IMG" --no-fuse
```
Expected: two `yolo_k` lines, `OK`, worst ≤ 1e-4 for both fused and unfused exports. If a shape-mismatch exception comes from onnxruntime, the offending layer is the one whose declared `value_info` differs from the computed one — re-check Review Focus 1/4 first.

Also run `--no-fuse` once and, for coverage of `Swish`/`Sigmoid`/`Mish`/`Resize`, one more cfg with those layers (any `cfg/*.cfg` with `[upsample]`, e.g. `cfg/yolov4-tiny-3l.cfg`) using random weights is NOT meaningful numerically — use only cfgs that have real weights on disk.

- [x] **Step 4: Commit**

```bash
git add tools/onnx_check.py
git commit -m "feat(onnx): onnxruntime check script against Rust layer dumps"
```

---

### Task 5: Record the work

**Files:**
- Modify: `tasks/rust/todo.md` (append under a new `## ONNX export` heading at the end, before `## Review` sections if present)

- [x] **Step 1: Add the entry** (tick boxes with measured numbers once Task 4 ran):

```markdown
## ONNX export (plan: tasks/rust/03_plans/misc/onnx.md)
- [x] Task 1 protobuf encoder (`v0.0.254`), Task 2 exporter (`v0.0.255`), Task 3 CLI (`v0.0.256`), Task 4 numeric checker (`v0.0.257`). `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; `MEM=3G ./tools/test_host.sh export_onnx_args` passed 1 CLI test; `cargo build -p lightnet-cli -j 2` passed. `tools/onnx_check.py` passed fused (worst relative error `3.257e-06`), unfused (`3.908e-05`), and Mish (`3.831e-06`) against Rust layer dumps, all under `1e-4`. Local fixture: `build/logs/verify/lego_cur_cpu/run.cfg` + `run_final.weights`, `build/logs/bench/LegoGears.names`, `artwork/dog.jpg`; the plan's canonical dataset paths are absent. No compiled-C++ parity is claimed.
- Follow-ups NOT in scope: postprocess boxes/confs sub-graph (C++ `populate_graph_postprocess`), fp16 initializers (`bit_size`), INT8, weighted shortcut, mismatched-shape shortcut, dilated conv, connected/softmax/avgpool/reorg.
```

- [x] **Step 2: Verify and commit**

No runtime test applies to this bookkeeping-only step. Verify the Markdown/task entry with `git diff --check` and confirm the measured tolerance and run results match Task 4 before committing:

```bash
git add tasks/rust/todo.md tasks/rust/03_plans/misc/onnx.md
git commit -m "docs(todo): ONNX export plan"
```

---

## Out of scope (explicit YAGNI list)

- YOLO decode/NMS inside the graph (C++ `postprocess_boxes`): heads are exported raw; decoding stays in the consumer (`decode_yolo_detections` equivalent). Add later as a separate plan.
- FP16 / INT8 weights; ONNX `ir_version` other than 8; ONNX files for training.
- Layers other than conv/maxpool/route/shortcut/upsample/yolo (`connected`, `softmax`, `avgpool`, `reorg`, recurrent, `scale_channels`, `sam`).

## Self-Review

- **Spec coverage:** Goal → Tasks 1–3 produce the file, Task 4 proves Rust-runtime numeric equality, Task 5 records it. Review Focus 1 → `maxpool_padding_is_asymmetric_like_darknet`; 2 → reverse rejection plus `upsample_nonunit_scale_emits_mul`; 3 → `mismatched_shortcut_is_unsupported`; 4 → grouped-route start/end assertions; 5 → `dilated_conv_is_unsupported`; 6 → `unsupported_constructs_are_errors_not_silent`; 7 → `no_yolo_network_uses_last_layer_output`. No compiled-C++ parity is claimed.
- **Placeholders:** none; the only conditional notes are about parser/trait derive differences, each with the concrete fallback.
- **Type consistency:** `OnnxOptions { fuse_batch_norm }` used identically in Tasks 2 and 3; `export_onnx(&Network, &[LayerParams], &OnnxOptions)` identical in the CLI call; tensor names `layer_{i:03}_out`, `yolo_{k}` consistent with `onnx_check.py` (it reads outputs by position, in cfg order).
