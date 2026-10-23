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
}

impl Default for OnnxOptions {
    fn default() -> Self {
        Self {
            fuse_batch_norm: true,
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
    };

    for (index, layer) in network.layers.iter().enumerate() {
        let output = match &layer.kind {
            LayerKind::Convolutional(spec) => exporter.conv(index, layer, spec)?,
            LayerKind::MaxPool(spec) if !spec.depth => exporter.maxpool(index, spec),
            LayerKind::MaxPool(_) => return Err(unsupported(index, "depth maxpool")),
            LayerKind::Route(spec) => exporter.route(index, spec)?,
            LayerKind::Shortcut(spec) => exporter.shortcut(index, layer, spec)?,
            LayerKind::Upsample(spec) => exporter.upsample(index, spec)?,
            LayerKind::Yolo(_) => exporter.yolo(index, layer),
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
    let input = value_info(
        "input",
        &[
            Dim::Param("batch"),
            Dim::Fixed(network.net.channels as i64),
            Dim::Fixed(network.net.height as i64),
            Dim::Fixed(network.net.width as i64),
        ],
    );
    let outputs = if exporter.yolo_outputs.is_empty() {
        let last = network.layers.last().expect("nonempty checked above");
        vec![value_info(
            exporter.layer_outputs.last().expect("one output per layer"),
            &shape_dims(&last.output_shape),
        )]
    } else {
        exporter
            .yolo_outputs
            .iter()
            .map(|(name, shape)| value_info(name, &shape_dims(shape)))
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
            "input".to_owned()
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
        self.initializers.push(tensor_f32(name, dims, data));
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
        self.add_f32(&format!("{name}_roi"), &[0], &[]);
        self.add_f32(&format!("{name}_scales"), &[4], &[1.0, 1.0, stride, stride]);
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
        output
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
    fn batch_norm_can_be_unfused_and_maxpool_padding_is_asymmetric() {
        let executable = build(CFG);
        let bytes = export_onnx(
            &executable.network,
            &executable.params,
            &OnnxOptions {
                fuse_batch_norm: false,
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
