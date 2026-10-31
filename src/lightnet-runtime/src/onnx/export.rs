//! Walks a loaded `Network` and its parameters and emits a raw-head ONNX graph.

use crate::error::{Error, Result};
use crate::model::{
    Activation, ConvolutionalSpec, Layer, LayerKind, MaxPoolSpec, Network, RouteSpec, ShortcutSpec,
    TensorShape, UpsampleSpec,
};
use crate::onnx::proto::*;
use crate::runtime::{BATCH_NORM_EPSILON, LayerParams, fold_convolutional_params_for_inference};

#[derive(Clone, Copy, Debug)]
pub struct OnnxOptions {
    /// Fold batch normalization into convolution weights and bias (default).
    pub fuse_batch_norm: bool,
    /// Expose image inputs and segmentation outputs in NHWC layout.
    pub nhwc: bool,
    /// Include the segmentation activation in the graph.
    pub segmentation_activation: bool,
    /// Export FP16 graph inputs, outputs, and floating-point parameters.
    pub fp16: bool,
    /// Decode YOLO heads in-graph into `boxes` (normalized x1,y1,x2,y2) and `confs`
    /// (objectness x class); NMS stays with the application. Default exports raw heads.
    pub decode_detections: bool,
}

impl Default for OnnxOptions {
    fn default() -> Self {
        Self {
            fuse_batch_norm: true,
            nhwc: false,
            segmentation_activation: true,
            fp16: false,
            decode_detections: false,
        }
    }
}

struct Exporter<'a> {
    network: &'a Network,
    params: &'a [LayerParams],
    options: OnnxOptions,
    nodes: Vec<Msg>,
    initializers: Vec<Msg>,
    layer_outputs: Vec<String>,
    yolo_outputs: Vec<(String, TensorShape)>,
    yolo_layers: Vec<usize>,
}

fn unsupported(layer_index: usize, name: impl Into<String>) -> Error {
    Error::UnsupportedLayer {
        layer_index,
        name: name.into(),
    }
}

/// Export supported layers as an ONNX model. Unsupported constructs fail rather
/// than producing a graph with silently altered semantics.
pub fn export_onnx(
    network: &Network,
    params: &[LayerParams],
    options: &OnnxOptions,
) -> Result<Vec<u8>> {
    if params.len() != network.layers.len() {
        return Err(Error::Weights(format!(
            "parameter count {} does not match layer count {}",
            params.len(),
            network.layers.len()
        )));
    }
    if network.layers.is_empty() {
        return Err(Error::InvalidConfig("network has no layers".into()));
    }

    let uses_mish = network.layers.iter().any(|layer| {
        matches!(&layer.kind, LayerKind::Convolutional(spec) if spec.activation == Activation::Mish)
            || matches!(&layer.kind, LayerKind::Shortcut(spec) if spec.activation == Activation::Mish)
    });
    let opset = if uses_mish { 18 } else { 13 };
    let mut exporter = Exporter {
        network,
        params,
        options: *options,
        nodes: Vec::new(),
        initializers: Vec::new(),
        layer_outputs: Vec::with_capacity(network.layers.len()),
        yolo_outputs: Vec::new(),
        yolo_layers: Vec::new(),
    };

    if options.nhwc {
        exporter.add_node("Transpose", "input_to_nchw", &["input".to_owned()], "input_nchw", vec![attr_ints("perm", &[0, 3, 1, 2])]);
    }

    for (index, layer) in network.layers.iter().enumerate() {
        let output = match &layer.kind {
            LayerKind::Convolutional(spec) => exporter.conv(index, layer, spec)?,
            LayerKind::MaxPool(spec) if !spec.depth => exporter.maxpool(index, spec),
            LayerKind::MaxPool(_) => return Err(unsupported(index, "depth maxpool")),
            LayerKind::Route(spec) => exporter.route(index, spec)?,
            LayerKind::Shortcut(spec) => exporter.shortcut(index, layer, spec)?,
            LayerKind::Upsample(spec) => exporter.upsample(index, spec)?,
            LayerKind::Yolo(_) => exporter.yolo(index, layer),
            LayerKind::Logistic if index + 1 == network.layers.len() => {
                let input = exporter.input_of(index);
                let output = format!("layer_{index:03}_out");
                let op = if options.segmentation_activation { "Sigmoid" } else { "Identity" };
                exporter.add_node(op, &output, &[input], &output, vec![]);
                output
            }
            LayerKind::Softmax { spatial: true, .. } if index + 1 == network.layers.len() => {
                let input = exporter.input_of(index);
                let output = format!("layer_{index:03}_out");
                if options.segmentation_activation {
                    exporter.add_node("Softmax", &output, &[input], &output, vec![attr_int("axis", 1)]);
                } else {
                    exporter.add_node("Identity", &output, &[input], &output, vec![]);
                }
                output
            }
            LayerKind::Segmentation(spec) => {
                let input = exporter.input_of(index);
                let output = format!("layer_{index:03}_out");
                if !options.segmentation_activation {
                    exporter.add_node("Identity", &output, &[input], &output, vec![]);
                } else if spec.classes == 1 {
                    let nan = format!("layer_{index:03}_nan");
                    let inf = format!("layer_{index:03}_inf");
                    let invalid = format!("layer_{index:03}_invalid");
                    let clean = format!("layer_{index:03}_clean");
                    let zero = format!("layer_{index:03}_zero");
                    exporter.add_f32(&zero, &[], &[0.0]);
                    exporter.add_node("IsNaN", &nan, &[input.clone()], &nan, vec![]);
                    exporter.add_node("IsInf", &inf, &[input.clone()], &inf, vec![]);
                    exporter.add_node("Or", &invalid, &[nan, inf], &invalid, vec![]);
                    exporter.add_node("Where", &clean, &[invalid, zero, input], &clean, vec![]);
                    exporter.add_node("Sigmoid", &output, &[clean], &output, vec![]);
                } else {
                    let temperature = format!("layer_{index:03}_temperature");
                    let scaled = format!("layer_{index:03}_scaled");
                    exporter.add_f32(&temperature, &[], &[spec.temperature.max(1e-6)]);
                    exporter.add_node("Div", &scaled, &[input, temperature], &scaled, vec![]);
                    exporter.add_node("Softmax", &output, &[scaled], &output, vec![attr_int("axis", 1)]);
                }
                output
            }
            other => return Err(unsupported(index, other.name())),
        };
        exporter.layer_outputs.push(output);
    }

    let shape_dims = |shape: &TensorShape| {
        [
            Dim::Param("batch"),
            Dim::Fixed(shape.channels as i64),
            Dim::Fixed(shape.height as i64),
            Dim::Fixed(shape.width as i64),
        ]
    };
    let input_dims = if options.nhwc {
        [Dim::Param("batch"), Dim::Fixed(network.net.height as i64), Dim::Fixed(network.net.width as i64), Dim::Fixed(network.net.channels as i64)]
    } else {
        [Dim::Param("batch"), Dim::Fixed(network.net.channels as i64), Dim::Fixed(network.net.height as i64), Dim::Fixed(network.net.width as i64)]
    };
    let input = value_info_typed("input", &input_dims, options.fp16);
    let segmentation_heads: Vec<_> = network.layers.iter().enumerate()
        .filter(|(index, layer)| matches!(layer.kind, LayerKind::Segmentation(_)) || (*index + 1 == network.layers.len() && matches!(layer.kind, LayerKind::Logistic | LayerKind::Softmax { spatial: true, .. })))
        .map(|(index, layer)| (index, layer.output_shape.clone()))
        .collect();
    let outputs = if !segmentation_heads.is_empty() {
        let mut results = Vec::with_capacity(segmentation_heads.len());
        for (index, shape) in &segmentation_heads {
            let source = exporter.layer_outputs[*index].clone();
            let name = if segmentation_heads.len() == 1 { "output".to_owned() } else { format!("segmentation_{index}") };
            if options.nhwc {
                exporter.add_node("Transpose", &format!("segmentation_{index}_nhwc"), &[source], &name, vec![attr_ints("perm", &[0, 2, 3, 1])]);
                results.push(value_info_typed(&name, &[Dim::Param("batch"), Dim::Fixed(shape.height as i64), Dim::Fixed(shape.width as i64), Dim::Fixed(shape.channels as i64)], options.fp16));
            } else {
                exporter.add_node("Identity", &format!("segmentation_{index}_output"), &[source], &name, vec![]);
                results.push(value_info_typed(&name, &shape_dims(shape), options.fp16));
            }
        }
        results
    } else if options.decode_detections && !exporter.yolo_outputs.is_empty() {
        exporter.decode_detections()?
    } else if exporter.yolo_outputs.is_empty() {
        let last = network.layers.last().expect("nonempty checked above");
        vec![value_info_typed(
            exporter.layer_outputs.last().expect("one output per layer"),
            &shape_dims(&last.output_shape),
            options.fp16,
        )]
    } else {
        exporter
            .yolo_outputs
            .iter()
            .map(|(name, shape)| value_info_typed(name, &shape_dims(shape), options.fp16))
            .collect()
    };

    Ok(model(
        opset,
        "darknet",
        &exporter.nodes,
        &exporter.initializers,
        &[input],
        &outputs,
    ))
}

impl Exporter<'_> {
    fn input_of(&self, index: usize) -> String {
        if index == 0 {
            if self.options.nhwc { "input_nchw" } else { "input" }.to_owned()
        } else {
            self.layer_outputs[index - 1].clone()
        }
    }

    fn source_index(&self, index: usize, source: isize) -> Result<usize> {
        let absolute = if source < 0 {
            index as isize + source
        } else {
            source
        };
        if absolute < 0 || absolute as usize >= index {
            return Err(unsupported(index, format!("invalid source layer {source}")));
        }
        Ok(absolute as usize)
    }

    fn add_f32(&mut self, name: &str, dims: &[i64], data: &[f32]) {
        self.initializers.push(if self.options.fp16 { tensor_f16(name, dims, data) } else { tensor_f32(name, dims, data) });
    }

    fn add_i64(&mut self, name: &str, dims: &[i64], data: &[i64]) {
        self.initializers.push(tensor_i64(name, dims, data));
    }

    fn add_node(&mut self, op: &str, name: &str, inputs: &[String], output: &str, attrs: Vec<Msg>) {
        self.nodes
            .push(node(op, name, inputs, &[output.to_owned()], attrs));
    }

    fn conv(&mut self, index: usize, layer: &Layer, spec: &ConvolutionalSpec) -> Result<String> {
        if spec.dilation != 1 || spec.share_index.is_some() {
            return Err(unsupported(
                index,
                "convolutional with dilation/share_index",
            ));
        }
        let LayerParams::Convolutional(params) = &self.params[index] else {
            return Err(unsupported(index, "convolutional without parameters"));
        };
        let input_channels = layer.input_shape.channels;
        let groups = spec.groups.max(1);
        if input_channels % groups != 0 || spec.filters % groups != 0 {
            return Err(unsupported(
                index,
                "convolution with invalid group dimensions",
            ));
        }
        let layer_name = format!("layer_{index:03}");
        let weight_dims = [
            spec.filters as i64,
            (input_channels / groups) as i64,
            spec.size as i64,
            spec.size as i64,
        ];
        let pad = spec.padding as i64;
        let attrs = vec![
            attr_int("group", groups as i64),
            attr_ints("pads", &[pad, pad, pad, pad]),
            attr_ints("strides", &[spec.stride_y as i64, spec.stride_x as i64]),
            attr_ints("dilations", &[1, 1]),
            attr_ints("kernel_shape", &[spec.size as i64, spec.size as i64]),
        ];
        let input = self.input_of(index);
        let has_separate_bn = params.batch_norm.is_some() && !self.options.fuse_batch_norm;
        let conv_output = if has_separate_bn || spec.activation != Activation::Linear {
            format!("{layer_name}_conv")
        } else {
            format!("{layer_name}_out")
        };

        if has_separate_bn {
            let bn = params.batch_norm.as_ref().expect("checked above");
            self.add_f32(&format!("{layer_name}_W"), &weight_dims, &params.weights);
            self.add_node(
                "Conv",
                &format!("{layer_name}_conv_node"),
                &[input, format!("{layer_name}_W")],
                &conv_output,
                attrs,
            );
            let channels = [spec.filters as i64];
            self.add_f32(&format!("{layer_name}_bn_scale"), &channels, &bn.scales);
            self.add_f32(&format!("{layer_name}_bn_B"), &channels, &bn.biases);
            self.add_f32(
                &format!("{layer_name}_bn_mean"),
                &channels,
                &bn.rolling_mean,
            );
            self.add_f32(
                &format!("{layer_name}_bn_var"),
                &channels,
                &bn.rolling_variance,
            );
            let bn_output = if spec.activation == Activation::Linear {
                format!("{layer_name}_out")
            } else {
                format!("{layer_name}_bn")
            };
            self.add_node(
                "BatchNormalization",
                &format!("{layer_name}_bn_node"),
                &[
                    conv_output,
                    format!("{layer_name}_bn_scale"),
                    format!("{layer_name}_bn_B"),
                    format!("{layer_name}_bn_mean"),
                    format!("{layer_name}_bn_var"),
                ],
                &bn_output,
                vec![attr_float("epsilon", BATCH_NORM_EPSILON)],
            );
            self.activation(index, &spec.activation, bn_output)
        } else {
            let (bias, weights) =
                fold_convolutional_params_for_inference(spec, params, input_channels)?;
            self.add_f32(&format!("{layer_name}_W"), &weight_dims, &weights);
            self.add_f32(&format!("{layer_name}_B"), &[spec.filters as i64], &bias);
            self.add_node(
                "Conv",
                &format!("{layer_name}_conv_node"),
                &[input, format!("{layer_name}_W"), format!("{layer_name}_B")],
                &conv_output,
                attrs,
            );
            self.activation(index, &spec.activation, conv_output)
        }
    }

    fn activation(
        &mut self,
        index: usize,
        activation: &Activation,
        input: String,
    ) -> Result<String> {
        let name = format!("layer_{index:03}_act");
        let output = format!("layer_{index:03}_out");
        match activation {
            Activation::Linear => return Ok(input),
            Activation::Leaky => self.add_node(
                "LeakyRelu",
                &name,
                &[input],
                &output,
                vec![attr_float("alpha", 0.1)],
            ),
            Activation::Logistic => self.add_node("Sigmoid", &name, &[input], &output, vec![]),
            Activation::Relu => self.add_node("Relu", &name, &[input], &output, vec![]),
            Activation::Tanh => self.add_node("Tanh", &name, &[input], &output, vec![]),
            Activation::Mish => self.add_node("Mish", &name, &[input], &output, vec![]),
            Activation::Swish => {
                let sigmoid = format!("{name}_sigmoid");
                self.add_node("Sigmoid", &sigmoid, &[input.clone()], &sigmoid, vec![]);
                self.add_node("Mul", &name, &[input, sigmoid], &output, vec![]);
            }
            Activation::Relu6 => {
                self.add_f32(&format!("{name}_min"), &[], &[0.0]);
                self.add_f32(&format!("{name}_max"), &[], &[6.0]);
                self.add_node(
                    "Clip",
                    &name,
                    &[input, format!("{name}_min"), format!("{name}_max")],
                    &output,
                    vec![],
                );
            }
            other => return Err(unsupported(index, format!("activation {other:?}"))),
        }
        Ok(output)
    }

    fn maxpool(&mut self, index: usize, spec: &MaxPoolSpec) -> String {
        let pad = spec.padding as i64;
        let (begin, end) = (pad / 2, pad - pad / 2);
        let output = format!("layer_{index:03}_out");
        let input = self.input_of(index);
        self.add_node(
            "MaxPool",
            &format!("layer_{index:03}_maxpool"),
            &[input],
            &output,
            vec![
                attr_int("ceil_mode", 0),
                attr_ints("pads", &[begin, begin, end, end]),
                attr_ints("kernel_shape", &[spec.size as i64, spec.size as i64]),
                attr_ints("strides", &[spec.stride_y as i64, spec.stride_x as i64]),
            ],
        );
        output
    }

    fn route(&mut self, index: usize, spec: &RouteSpec) -> Result<String> {
        let sources: Vec<usize> = spec
            .layers
            .iter()
            .map(|source| self.source_index(index, *source))
            .collect::<Result<_>>()?;
        let output = format!("layer_{index:03}_out");
        match (sources.len(), spec.groups) {
            (1, 1) => Ok(self.layer_outputs[sources[0]].clone()),
            (1, groups) if groups > 1 => {
                if spec.group_id >= groups {
                    return Err(unsupported(index, "route group_id out of range"));
                }
                let channels = self.network.layers[sources[0]].output_shape.channels;
                if channels % groups != 0 {
                    return Err(unsupported(
                        index,
                        "route groups do not divide source channels",
                    ));
                }
                let per_group = channels / groups;
                let start = (spec.group_id * per_group) as i64;
                let end = start + per_group as i64;
                let slice_name = format!("layer_{index:03}_slice");
                for (suffix, value) in [("starts", start), ("ends", end), ("axes", 1), ("steps", 1)]
                {
                    self.add_i64(&format!("{slice_name}_{suffix}"), &[1], &[value]);
                }
                let inputs = [
                    self.layer_outputs[sources[0]].clone(),
                    format!("{slice_name}_starts"),
                    format!("{slice_name}_ends"),
                    format!("{slice_name}_axes"),
                    format!("{slice_name}_steps"),
                ];
                self.add_node("Slice", &slice_name, &inputs, &output, vec![]);
                Ok(output)
            }
            (count, 1) if count >= 2 => {
                let inputs: Vec<String> = sources
                    .iter()
                    .map(|source| self.layer_outputs[*source].clone())
                    .collect();
                self.add_node(
                    "Concat",
                    &format!("layer_{index:03}_concat"),
                    &inputs,
                    &output,
                    vec![attr_int("axis", 1)],
                );
                Ok(output)
            }
            _ => Err(unsupported(index, "route with unsupported groups/sources")),
        }
    }

    fn shortcut(&mut self, index: usize, layer: &Layer, spec: &ShortcutSpec) -> Result<String> {
        if spec.weights_type != crate::model::ShortcutWeightsType::None {
            return Err(unsupported(index, "weighted shortcut"));
        }
        let mut accumulated = self.input_of(index);
        for (slot, from) in spec.from.iter().enumerate() {
            let source = self.source_index(index, *from)?;
            if self.network.layers[source].output_shape != layer.output_shape {
                return Err(unsupported(
                    index,
                    "shortcut with mismatched source shape (flat crop/pad)",
                ));
            }
            let output = format!("layer_{index:03}_add{slot}");
            self.add_node(
                "Add",
                &output,
                &[accumulated, self.layer_outputs[source].clone()],
                &output,
                vec![],
            );
            accumulated = output;
        }
        self.activation(index, &spec.activation, accumulated)
    }

    fn upsample(&mut self, index: usize, spec: &UpsampleSpec) -> Result<String> {
        if spec.reverse || spec.stride <= 0 {
            return Err(unsupported(
                index,
                "upsample reverse/downsample or invalid stride",
            ));
        }
        let name = format!("layer_{index:03}_resize");
        let stride = spec.stride as f32;
        self.initializers.push(tensor_f32(&format!("{name}_roi"), &[0], &[]));
        self.initializers.push(tensor_f32(&format!("{name}_scales"), &[4], &[1.0, 1.0, stride, stride]));
        let resized = if spec.scale == 1.0 {
            format!("layer_{index:03}_out")
        } else {
            format!("{name}_out")
        };
        let input = self.input_of(index);
        self.add_node(
            "Resize",
            &name,
            &[input, format!("{name}_roi"), format!("{name}_scales")],
            &resized,
            vec![
                attr_str("mode", "nearest"),
                attr_str("coordinate_transformation_mode", "asymmetric"),
                attr_str("nearest_mode", "floor"),
            ],
        );
        if spec.scale == 1.0 {
            return Ok(resized);
        }
        let output = format!("layer_{index:03}_out");
        self.add_f32(&format!("{name}_scale"), &[], &[spec.scale]);
        self.add_node(
            "Mul",
            &format!("{name}_mul"),
            &[resized, format!("{name}_scale")],
            &output,
            vec![],
        );
        Ok(output)
    }

    fn yolo(&mut self, index: usize, layer: &Layer) -> String {
        let output = format!("yolo_{}", self.yolo_outputs.len());
        let input = self.input_of(index);
        self.add_node(
            "Identity",
            &format!("layer_{index:03}_yolo"),
            &[input],
            &output,
            vec![],
        );
        self.yolo_outputs.push((output.clone(), layer.output_shape));
        self.yolo_layers.push(index);
        output
    }

    fn emit(&mut self, op: &str, name: &str, inputs: &[&str], attrs: Vec<Msg>) -> String {
        let output = format!("{name}_out");
        let inputs: Vec<String> = inputs.iter().map(|s| (*s).to_owned()).collect();
        self.add_node(op, name, &inputs, &output, attrs);
        output
    }

    fn slice(&mut self, name: &str, input: &str, start: i64, end: i64, axis: i64) -> String {
        self.add_i64(&format!("{name}_starts"), &[1], &[start]);
        self.add_i64(&format!("{name}_ends"), &[1], &[end]);
        self.add_i64(&format!("{name}_axes"), &[1], &[axis]);
        let (starts, ends, axes) = (format!("{name}_starts"), format!("{name}_ends"), format!("{name}_axes"));
        self.emit("Slice", name, &[input, &starts, &ends, &axes], vec![])
    }

    /// Reshape with `0` copying the (dynamic) batch dimension.
    fn reshape(&mut self, name: &str, input: &str, dims: &[i64]) -> String {
        self.add_i64(&format!("{name}_shape"), &[dims.len() as i64], dims);
        let shape = format!("{name}_shape");
        self.emit("Reshape", name, &[input, &shape], vec![])
    }

    fn constant(&mut self, name: &str, dims: &[i64], data: &[f32]) -> String {
        self.add_f32(name, dims, data);
        name.to_owned()
    }

    /// Decode every YOLO head into `boxes` [batch, N, 1, 4] (normalized x1,y1,x2,y2)
    /// and `confs` [batch, N, classes] (objectness x class), mirroring the Darknet ONNX
    /// post-process. Anchor order is (mask, y, x) within a head, heads concatenated.
    fn decode_detections(&mut self) -> Result<Vec<Msg>> {
        let (net_w, net_h) = (self.network.net.width as f32, self.network.net.height as f32);
        let fp16 = self.options.fp16;
        let mut boxes = Vec::new();
        let mut confs = Vec::new();
        let (mut total, mut classes_out) = (0usize, 0usize);
        for (head, &index) in self.yolo_layers.clone().iter().enumerate() {
            let layer = &self.network.layers[index];
            let LayerKind::Yolo(spec) = &layer.kind else {
                return Err(unsupported(index, "yolo head bookkeeping"));
            };
            let (masks, classes) = (spec.mask.len(), spec.classes);
            let (h, w) = (layer.output_shape.height, layer.output_shape.width);
            let hw = h * w;
            if masks == 0 || layer.output_shape.channels != masks * (5 + classes) {
                return Err(unsupported(index, "yolo head with unexpected channel count"));
            }
            if head > 0 && classes != classes_out {
                return Err(unsupported(index, "decoded yolo heads with different class counts"));
            }
            classes_out = classes;
            if spec.anchors.len() < 2 * spec.mask.iter().max().map_or(0, |m| m + 1) {
                return Err(unsupported(index, "yolo mask indexes past anchors"));
            }
            // The previous layer's logistic activation already covers tx, ty, objectness, classes.
            let activated = index > 0
                && matches!(&self.network.layers[index - 1].kind, LayerKind::Convolutional(s) if s.activation == Activation::Logistic);
            let p = format!("yolo{head}");
            let source = self.input_of(index);
            let x = self.reshape(&format!("{p}_split"), &source, &[0, masks as i64, (5 + classes) as i64, hw as i64]);
            let xy = self.slice(&format!("{p}_xy"), &x, 0, 2, 2);
            let wh = self.slice(&format!("{p}_wh"), &x, 2, 4, 2);
            let obj = self.slice(&format!("{p}_obj"), &x, 4, 5, 2);
            let cls = self.slice(&format!("{p}_cls"), &x, 5, (5 + classes) as i64, 2);
            let sigmoid = |this: &mut Self, name: &str, input: String| {
                if activated { input } else { this.emit("Sigmoid", name, &[&input], vec![]) }
            };

            // centers: (sigmoid(t) * s - (s - 1) / 2 + grid) / [W, H]
            let mut xy = sigmoid(self, &format!("{p}_xy_sig"), xy);
            let s = spec.scale_x_y;
            if s != 1.0 {
                let scale = self.constant(&format!("{p}_xy_scale"), &[], &[s]);
                let offset = self.constant(&format!("{p}_xy_offset"), &[], &[(s - 1.0) / 2.0]);
                let scaled = self.emit("Mul", &format!("{p}_xy_mul"), &[&xy, &scale], vec![]);
                xy = self.emit("Sub", &format!("{p}_xy_sub"), &[&scaled, &offset], vec![]);
            }
            let mut grid = Vec::with_capacity(2 * hw);
            grid.extend((0..hw).map(|pos| (pos % w) as f32));
            grid.extend((0..hw).map(|pos| (pos / w) as f32));
            let grid = self.constant(&format!("{p}_grid"), &[1, 1, 2, hw as i64], &grid);
            let extent = self.constant(&format!("{p}_extent"), &[1, 1, 2, 1], &[w as f32, h as f32]);
            let xy = self.emit("Add", &format!("{p}_xy_grid"), &[&xy, &grid], vec![]);
            let xy = self.emit("Div", &format!("{p}_xy_norm"), &[&xy, &extent], vec![]);

            // sizes: exp(t) or (2t)^2, times anchor / network size
            let wh = if spec.new_coords {
                let two = self.constant(&format!("{p}_two"), &[], &[2.0]);
                let doubled = self.emit("Mul", &format!("{p}_wh_double"), &[&wh, &two], vec![]);
                self.emit("Mul", &format!("{p}_wh_square"), &[&doubled, &doubled], vec![])
            } else {
                self.emit("Exp", &format!("{p}_wh_exp"), &[&wh], vec![])
            };
            let mut anchors = Vec::with_capacity(2 * masks);
            for &m in &spec.mask {
                anchors.push(spec.anchors[2 * m] / net_w);
                anchors.push(spec.anchors[2 * m + 1] / net_h);
            }
            let anchors = self.constant(&format!("{p}_anchors"), &[1, masks as i64, 2, 1], &anchors);
            let wh = self.emit("Mul", &format!("{p}_wh_anchor"), &[&wh, &anchors], vec![]);

            // corners
            let half = self.constant(&format!("{p}_half"), &[], &[0.5]);
            let half_wh = self.emit("Mul", &format!("{p}_half_wh"), &[&wh, &half], vec![]);
            let top_left = self.emit("Sub", &format!("{p}_tl"), &[&xy, &half_wh], vec![]);
            let bottom_right = self.emit("Add", &format!("{p}_br"), &[&top_left, &wh], vec![]);
            let corners = self.emit("Concat", &format!("{p}_corners"), &[&top_left, &bottom_right], vec![attr_int("axis", 2)]);
            let corners = self.emit("Transpose", &format!("{p}_corners_t"), &[&corners], vec![attr_ints("perm", &[0, 1, 3, 2])]);
            let count = (masks * hw) as i64;
            boxes.push(self.reshape(&format!("{p}_boxes"), &corners, &[0, count, 1, 4]));

            // confidences
            let obj = sigmoid(self, &format!("{p}_obj_sig"), obj);
            let cls = sigmoid(self, &format!("{p}_cls_sig"), cls);
            let conf = self.emit("Mul", &format!("{p}_conf"), &[&cls, &obj], vec![]);
            let conf = self.emit("Transpose", &format!("{p}_conf_t"), &[&conf], vec![attr_ints("perm", &[0, 1, 3, 2])]);
            confs.push(self.reshape(&format!("{p}_confs"), &conf, &[0, count, classes as i64]));
            total += masks * hw;
        }
        let (boxes_in, confs_in) = (boxes, confs);
        self.add_node("Concat", "boxes_concat", &boxes_in, "boxes", vec![attr_int("axis", 1)]);
        self.add_node("Concat", "confs_concat", &confs_in, "confs", vec![attr_int("axis", 1)]);
        Ok(vec![
            value_info_typed("boxes", &[Dim::Param("batch"), Dim::Fixed(total as i64), Dim::Fixed(1), Dim::Fixed(4)], fp16),
            value_info_typed("confs", &[Dim::Param("batch"), Dim::Fixed(total as i64), Dim::Fixed(classes_out as i64)], fp16),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::onnx::proto::testing::{Wire, decode};
    use crate::runtime::ExecutableNetwork;

    const CFG: &str = "\n[net]\nwidth=16\nheight=16\nchannels=3\nbatch=1\n\n[convolutional]\nbatch_normalize=1\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=leaky\n\n[maxpool]\nsize=2\nstride=1\n\n[convolutional]\nfilters=4\nsize=1\nstride=1\npad=1\nactivation=linear\n\n[route]\nlayers=-1\ngroups=2\ngroup_id=1\n\n[convolutional]\nfilters=8\nsize=1\nstride=1\npad=1\nactivation=linear\n\n[route]\nlayers=-1,-3\n\n[upsample]\nstride=2\n\n[yolo]\nmask=0\nanchors=2,3\nclasses=1\nnum=1\n";

    fn build(cfg: &str) -> ExecutableNetwork {
        let network =
            Network::from_config(ConfigFile::parse_str("test.cfg", cfg).unwrap()).unwrap();
        ExecutableNetwork::new(network)
    }

    fn graph(bytes: &[u8]) -> Vec<(u32, Wire)> {
        let top = decode(bytes);
        top.iter()
            .find_map(|(field, wire)| match (field, wire) {
                (7, Wire::Len(bytes)) => Some(decode(bytes)),
                _ => None,
            })
            .unwrap()
    }

    fn ops(bytes: &[u8]) -> Vec<String> {
        let graph = graph(bytes);
        graph
            .iter()
            .filter_map(|(field, wire)| match (field, wire) {
                (1, Wire::Len(bytes)) => {
                    let node = decode(bytes);
                    node.iter().find_map(|(f, w)| match (f, w) {
                        (4, Wire::Len(op)) => Some(String::from_utf8(op.clone()).unwrap()),
                        _ => None,
                    })
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn exports_binary_and_multiclass_segmentation_activations() {
        for (classes, expected) in [(1, "Sigmoid"), (3, "Softmax")] {
            let cfg = format!("[net]\nwidth=3\nheight=2\nchannels={classes}\nbatch=1\n[segmentation]\nclasses={classes}\n");
            let executable = build(&cfg);
            let bytes = export_onnx(&executable.network, &executable.params, &OnnxOptions::default()).unwrap();
            assert!(ops(&bytes).contains(&expected.to_owned()));
        }
    }

    #[test]
    fn segmentation_nhwc_and_raw_logits_use_boundary_transposes() {
        let executable = build("[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n[segmentation]\nclasses=1\n");
        let options = OnnxOptions { nhwc: true, segmentation_activation: false, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        assert_eq!(ops(&bytes), ["Transpose", "Identity", "Transpose"]);
    }

    #[test]
    fn legacy_logistic_segmentation_can_export_activated_or_raw() {
        let executable = build("[net]\nwidth=3\nheight=2\nchannels=1\nbatch=1\n[logistic]\n");
        for (enabled, expected) in [(true, "Sigmoid"), (false, "Identity")] {
            let options = OnnxOptions { segmentation_activation: enabled, ..OnnxOptions::default() };
            let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
            assert_eq!(ops(&bytes)[0], expected);
        }
    }

    #[test]
    fn legacy_spatial_softmax_exports_on_channel_axis() {
        let executable = build("[net]\nwidth=3\nheight=2\nchannels=4\nbatch=1\n[softmax]\nspatial=1\n");
        for (enabled, expected) in [(true, "Softmax"), (false, "Identity")] {
            let options = OnnxOptions { segmentation_activation: enabled, nhwc: true, ..OnnxOptions::default() };
            let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
            assert_eq!(ops(&bytes), ["Transpose", expected, "Transpose"]);
        }
    }

    #[test]
    fn fp16_export_keeps_resize_metadata_float32() {
        let executable = build(CFG);
        let options = OnnxOptions { fp16: true, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        let tensors: Vec<(String, u64)> = graph(&bytes).iter().filter_map(|(field, wire)| {
            if *field != 5 { return None; }
            let Wire::Len(bytes) = wire else { return None; };
            let fields = decode(bytes);
            let name = fields.iter().find_map(|(f, w)| match (f, w) { (8, Wire::Len(v)) => Some(String::from_utf8(v.clone()).unwrap()), _ => None })?;
            let dtype = fields.iter().find_map(|(f, w)| match (f, w) { (2, Wire::Varint(v)) => Some(*v), _ => None })?;
            Some((name, dtype))
        }).collect();
        assert!(tensors.iter().any(|(name, dtype)| name.ends_with("_W") && *dtype == 10));
        assert!(tensors.iter().any(|(name, dtype)| name.ends_with("_scales") && *dtype == 1));
    }

    #[test]
    fn exports_supported_layers_and_preserves_routes() {
        let executable = build(CFG);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions::default(),
        )
        .unwrap();
        assert_eq!(
            ops(&bytes),
            [
                "Conv",
                "LeakyRelu",
                "MaxPool",
                "Conv",
                "Slice",
                "Conv",
                "Concat",
                "Resize",
                "Identity"
            ]
        );
    }

    #[test]
    fn decoded_detection_export_replaces_raw_heads_with_boxes_and_confs() {
        let executable = build(DECODE_CFG);
        let export = |decode_detections| export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions { decode_detections, ..OnnxOptions::default() },
        ).unwrap();
        let contains = |bytes: &[u8], needle: &str| bytes.windows(needle.len()).any(|w| w == needle.as_bytes());
        let raw = export(false);
        assert!(contains(&raw, "yolo_0") && !contains(&raw, "confs"));
        let decoded = export(true);
        let actual = ops(&decoded);
        for op in ["Reshape", "Slice", "Sigmoid", "Exp", "Transpose", "Concat"] {
            assert!(actual.contains(&op.to_owned()), "missing {op}");
        }
        assert!(contains(&decoded, "boxes") && contains(&decoded, "confs"));
    }

    // One 16x16 head with 2 classes: a 1x1 conv to 7 = 1 mask * (5 + 2) channels.
    const DECODE_CFG: &str = "\n[net]\nwidth=16\nheight=16\nchannels=3\nbatch=1\n\n[convolutional]\nfilters=7\nsize=1\nstride=1\npad=0\nactivation=linear\n\n[yolo]\nmask=0\nanchors=2,3\nclasses=2\nnum=1\n";

    #[test]
    fn decoded_detection_export_fp32_nchw() {
        let executable = build(DECODE_CFG);
        let options = OnnxOptions { decode_detections: true, fp16: false, nhwc: false, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        let mut found = Vec::new();
        for (field, wire) in graph(&bytes) {
            let Wire::Len(info) = wire else { continue };
            if field != 11 && field != 12 {
                continue;
            }
            let info = decode(&info);
            let name = info.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(String::from_utf8(v.clone()).unwrap()),
                _ => None,
            }).unwrap();
            let ty = info.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let tensor = ty.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let elem = tensor.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Varint(v)) => Some(*v as i64),
                _ => None,
            }).unwrap();
            let shape = tensor.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let dims: Vec<String> = shape.iter().filter_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v).iter().find_map(|(f, w)| match (f, w) {
                    (1, Wire::Varint(d)) => Some(d.to_string()),
                    (2, Wire::Len(p)) => Some(String::from_utf8(p.clone()).unwrap()),
                    _ => None,
                }).unwrap()),
                _ => None,
            }).collect();
            found.push((name, elem, dims.join(",")));
        }
        assert_eq!(found, vec![
            ("input".to_owned(), 1, "batch,3,16,16".to_owned()),
            ("boxes".to_owned(), 1, "batch,256,1,4".to_owned()),
            ("confs".to_owned(), 1, "batch,256,2".to_owned()),
        ]);
    }

    #[test]
    fn decoded_detection_export_fp32_nhwc() {
        let executable = build(DECODE_CFG);
        let options = OnnxOptions { decode_detections: true, fp16: false, nhwc: true, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        let mut found = Vec::new();
        for (field, wire) in graph(&bytes) {
            let Wire::Len(info) = wire else { continue };
            if field != 11 && field != 12 {
                continue;
            }
            let info = decode(&info);
            let name = info.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(String::from_utf8(v.clone()).unwrap()),
                _ => None,
            }).unwrap();
            let ty = info.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let tensor = ty.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let elem = tensor.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Varint(v)) => Some(*v as i64),
                _ => None,
            }).unwrap();
            let shape = tensor.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let dims: Vec<String> = shape.iter().filter_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v).iter().find_map(|(f, w)| match (f, w) {
                    (1, Wire::Varint(d)) => Some(d.to_string()),
                    (2, Wire::Len(p)) => Some(String::from_utf8(p.clone()).unwrap()),
                    _ => None,
                }).unwrap()),
                _ => None,
            }).collect();
            found.push((name, elem, dims.join(",")));
        }
        assert_eq!(found, vec![
            ("input".to_owned(), 1, "batch,16,16,3".to_owned()),
            ("boxes".to_owned(), 1, "batch,256,1,4".to_owned()),
            ("confs".to_owned(), 1, "batch,256,2".to_owned()),
        ]);
    }

    #[test]
    fn decoded_detection_export_fp16_nchw() {
        let executable = build(DECODE_CFG);
        let options = OnnxOptions { decode_detections: true, fp16: true, nhwc: false, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        let mut found = Vec::new();
        for (field, wire) in graph(&bytes) {
            let Wire::Len(info) = wire else { continue };
            if field != 11 && field != 12 {
                continue;
            }
            let info = decode(&info);
            let name = info.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(String::from_utf8(v.clone()).unwrap()),
                _ => None,
            }).unwrap();
            let ty = info.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let tensor = ty.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let elem = tensor.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Varint(v)) => Some(*v as i64),
                _ => None,
            }).unwrap();
            let shape = tensor.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let dims: Vec<String> = shape.iter().filter_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v).iter().find_map(|(f, w)| match (f, w) {
                    (1, Wire::Varint(d)) => Some(d.to_string()),
                    (2, Wire::Len(p)) => Some(String::from_utf8(p.clone()).unwrap()),
                    _ => None,
                }).unwrap()),
                _ => None,
            }).collect();
            found.push((name, elem, dims.join(",")));
        }
        assert_eq!(found, vec![
            ("input".to_owned(), 10, "batch,3,16,16".to_owned()),
            ("boxes".to_owned(), 10, "batch,256,1,4".to_owned()),
            ("confs".to_owned(), 10, "batch,256,2".to_owned()),
        ]);
    }

    #[test]
    fn decoded_detection_export_fp16_nhwc() {
        let executable = build(DECODE_CFG);
        let options = OnnxOptions { decode_detections: true, fp16: true, nhwc: true, ..OnnxOptions::default() };
        let bytes = export_onnx(&executable.network, &executable.params, &options).unwrap();
        let mut found = Vec::new();
        for (field, wire) in graph(&bytes) {
            let Wire::Len(info) = wire else { continue };
            if field != 11 && field != 12 {
                continue;
            }
            let info = decode(&info);
            let name = info.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(String::from_utf8(v.clone()).unwrap()),
                _ => None,
            }).unwrap();
            let ty = info.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let tensor = ty.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let elem = tensor.iter().find_map(|(f, w)| match (f, w) {
                (1, Wire::Varint(v)) => Some(*v as i64),
                _ => None,
            }).unwrap();
            let shape = tensor.iter().find_map(|(f, w)| match (f, w) {
                (2, Wire::Len(v)) => Some(decode(v)),
                _ => None,
            }).unwrap();
            let dims: Vec<String> = shape.iter().filter_map(|(f, w)| match (f, w) {
                (1, Wire::Len(v)) => Some(decode(v).iter().find_map(|(f, w)| match (f, w) {
                    (1, Wire::Varint(d)) => Some(d.to_string()),
                    (2, Wire::Len(p)) => Some(String::from_utf8(p.clone()).unwrap()),
                    _ => None,
                }).unwrap()),
                _ => None,
            }).collect();
            found.push((name, elem, dims.join(",")));
        }
        assert_eq!(found, vec![
            ("input".to_owned(), 10, "batch,16,16,3".to_owned()),
            ("boxes".to_owned(), 10, "batch,256,1,4".to_owned()),
            ("confs".to_owned(), 10, "batch,256,2".to_owned()),
        ]);
    }

    #[test]
    fn batch_norm_can_be_unfused_and_maxpool_padding_is_asymmetric() {
        let executable = build(CFG);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions {
                fuse_batch_norm: false,
                ..OnnxOptions::default()
            },
        )
        .unwrap();
        let actual = ops(&bytes);
        assert_eq!(&actual[..3], ["Conv", "BatchNormalization", "LeakyRelu"]);
        assert!(actual.contains(&"MaxPool".to_owned()));
    }

    #[test]
    fn unsupported_layers_and_activations_fail_explicitly() {
        let ramp = CFG.replace("activation=leaky", "activation=ramp");
        let executable = build(&ramp);
        assert!(
            export_onnx(
                &executable.network,
                &executable.params,
                &OnnxOptions::default()
            )
            .is_err()
        );
        let reversed = CFG.replace("[upsample]\nstride=2", "[upsample]\nstride=-2");
        let executable = build(&reversed);
        assert!(
            export_onnx(
                &executable.network,
                &executable.params,
                &OnnxOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn mish_uses_opset_18() {
        let cfg = CFG.replace("activation=leaky", "activation=mish");
        let executable = build(&cfg);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions::default(),
        )
        .unwrap();
        let top = decode(&bytes);
        let opset = top
            .iter()
            .find_map(|(field, wire)| match (field, wire) {
                (8, Wire::Len(bytes)) => Some(decode(bytes)),
                _ => None,
            })
            .unwrap();
        assert!(
            opset
                .iter()
                .any(|(field, wire)| *field == 2 && matches!(wire, Wire::Varint(18)))
        );
    }

    fn unpack_varints(mut bytes: &[u8]) -> Vec<i64> {
        let mut values = Vec::new();
        while !bytes.is_empty() {
            let (mut value, mut shift) = (0u64, 0);
            loop {
                let byte = bytes[0];
                bytes = &bytes[1..];
                value |= ((byte & 0x7f) as u64) << shift;
                if byte & 0x80 == 0 {
                    break;
                }
                shift += 7;
            }
            values.push(value as i64);
        }
        values
    }

    #[test]
    fn maxpool_padding_is_asymmetric_and_grouped_route_uses_source_channels() {
        let executable = build(CFG);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions::default(),
        )
        .unwrap();
        let fields = graph(&bytes);
        let maxpool_pads = fields
            .iter()
            .find_map(|(field, wire)| {
                let Wire::Len(node_bytes) = wire else {
                    return None;
                };
                if *field != 1 {
                    return None;
                }
                let node = decode(node_bytes);
                let op = node.iter().find_map(|(f, w)| match (f, w) {
                    (4, Wire::Len(value)) => Some(value.as_slice()),
                    _ => None,
                });
                if op != Some(b"MaxPool".as_slice()) {
                    return None;
                }
                node.iter()
                    .filter_map(|(f, w)| match (f, w) {
                        (5, Wire::Len(attr_bytes)) => Some(decode(attr_bytes)),
                        _ => None,
                    })
                    .find_map(|attr| {
                        let is_pads = attr
                            .iter()
                            .any(|(f, w)| *f == 1 && matches!(w, Wire::Len(b) if b == b"pads"));
                        if !is_pads {
                            return None;
                        }
                        attr.into_iter().find_map(|(f, w)| match (f, w) {
                            (8, Wire::Len(values)) => Some(unpack_varints(&values)),
                            _ => None,
                        })
                    })
            })
            .unwrap();
        assert_eq!(maxpool_pads, [0, 0, 1, 1]);

        let starts = fields
            .iter()
            .find_map(|(field, wire)| {
                let Wire::Len(init_bytes) = wire else {
                    return None;
                };
                if *field != 5 {
                    return None;
                }
                let init = decode(init_bytes);
                let name = init.iter().find_map(|(f, w)| match (f, w) {
                    (8, Wire::Len(name)) => Some(String::from_utf8_lossy(name).into_owned()),
                    _ => None,
                })?;
                if !name.ends_with("_starts") {
                    return None;
                }
                init.iter().find_map(|(f, w)| match (f, w) {
                    (9, Wire::Len(raw)) => Some((
                        name.clone(),
                        raw.chunks_exact(8)
                            .map(|v| i64::from_le_bytes(v.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                    )),
                    _ => None,
                })
            })
            .unwrap();
        assert_eq!(starts.1, [2]);
    }

    #[test]
    fn nonunit_upsample_scale_emits_mul() {
        let cfg = CFG.replace("[upsample]\nstride=2", "[upsample]\nstride=2\nscale=2");
        let executable = build(&cfg);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions::default(),
        )
        .unwrap();
        assert!(ops(&bytes).contains(&"Mul".to_owned()));
    }

    #[test]
    fn dilated_convolution_and_mismatched_shortcut_are_rejected() {
        let cfg = CFG.replace(
            "filters=4\nsize=3\nstride=1",
            "filters=4\nsize=3\nstride=1\ndilation=2",
        );
        let executable = build(&cfg);
        assert!(
            export_onnx(
                &executable.network,
                &executable.params,
                &OnnxOptions::default()
            )
            .is_err()
        );

        let shortcut_cfg = "[net]\nwidth=16\nheight=16\nchannels=3\nbatch=1\n[convolutional]\nfilters=4\nsize=1\nactivation=linear\n[maxpool]\nsize=2\nstride=2\n[shortcut]\nfrom=-2\nactivation=linear\n";
        let executable = build(shortcut_cfg);
        assert!(
            export_onnx(
                &executable.network,
                &executable.params,
                &OnnxOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn classifier_exports_last_layer_when_no_yolo_exists() {
        let cfg = "[net]\nwidth=8\nheight=8\nchannels=3\nbatch=1\n[convolutional]\nfilters=2\nsize=1\nactivation=linear\n";
        let executable = build(cfg);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions::default(),
        )
        .unwrap();
        let top = decode(&bytes);
        let graph = top
            .iter()
            .find_map(|(field, wire)| match (field, wire) {
                (7, Wire::Len(bytes)) => Some(decode(bytes)),
                _ => None,
            })
            .unwrap();
        assert!(
            graph
                .iter()
                .any(|(field, wire)| *field == 12 && matches!(wire, Wire::Len(_)))
        );
    }
}
