//! CPU reference math for the source-derived DPT neck and depth head.
//!
//! The math is deliberately independent of cfg/runtime types. Tensor storage is
//! NCHW; the four input taps are contiguous channel blocks within each sample.

use crate::error::{Error, Result};

pub(crate) const DPT_WEIGHT_SLOTS: usize = 66;
pub(crate) const DPT_MUON_WEIGHT_SLOTS: [usize; 32] = [
    0, 2, 4, 6, 8, 10, 14, 16, 17, 18, 19, 20, 22, 24, 26, 30, 32, 34, 38, 40, 42, 44, 46, 48, 50,
    52, 54, 56, 58, 60, 62, 64,
];
pub(crate) const DPT_BIAS_SLOTS: [usize; 28] = [
    1, 3, 5, 7, 9, 11, 15, 21, 23, 25, 27, 31, 33, 35, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
    61, 63, 65,
];
const RESIZE_KERNELS: [usize; 4] = [4, 2, 0, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DptHeadGeometry {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub embed_dim: usize,
    pub fusion: usize,
    pub head_hidden: usize,
    pub patch_size: usize,
    pub neck: [usize; 4],
    /// Effective C++ factors after its float-to-integer coercion.
    pub reassemble: [usize; 4],
}

impl DptHeadGeometry {
    pub(crate) fn output_height(self) -> Result<usize> {
        self.height
            .checked_mul(self.patch_size)
            .ok_or_else(|| invalid("output height overflows"))
    }

    pub(crate) fn output_width(self) -> Result<usize> {
        self.width
            .checked_mul(self.patch_size)
            .ok_or_else(|| invalid("output width overflows"))
    }

    pub(crate) fn validate(self) -> Result<()> {
        if [
            self.batch,
            self.height,
            self.width,
            self.embed_dim,
            self.fusion,
            self.head_hidden,
            self.patch_size,
        ]
        .contains(&0)
        {
            return Err(invalid(
                "batch, input geometry, widths, and patch_size must be positive",
            ));
        }
        if self.fusion % 2 != 0 {
            return Err(invalid(
                "fusion must be even because the head uses fusion/2 channels",
            ));
        }
        if self.neck.contains(&0) {
            return Err(invalid("all four neck widths must be positive"));
        }
        if self.reassemble != [4, 2, 1, 0] {
            return Err(invalid(format!(
                "only effective reassemble=[4,2,1,0] is supported by the fixed source weight slots, got {:?}",
                self.reassemble
            )));
        }
        let output_height = self.output_height()?;
        let output_width = self.output_width()?;
        checked_product(&[self.batch, output_height, output_width], "output tensor")?;
        checked_product(
            &[self.batch, 4, self.embed_dim, self.height, self.width],
            "input tensor",
        )?;
        for tap in 0..4 {
            let height = reassembled_size(self.height, self.reassemble[tap])?;
            let width = reassembled_size(self.width, self.reassemble[tap])?;
            checked_product(
                &[self.batch, self.neck[tap], self.height, self.width],
                "projected feature map",
            )?;
            checked_product(
                &[self.batch, self.fusion, height, width],
                "reassembled feature map",
            )?;
        }
        let final_height = reassembled_size(self.height, 4)?
            .checked_mul(2)
            .ok_or_else(|| invalid("final fusion height overflows"))?;
        let final_width = reassembled_size(self.width, 4)?
            .checked_mul(2)
            .ok_or_else(|| invalid("final fusion width overflows"))?;
        checked_product(
            &[self.batch, self.fusion, final_height, final_width],
            "final fusion feature map",
        )?;
        checked_product(
            &[self.batch, self.fusion / 2, final_height, final_width],
            "head conv1 feature map",
        )?;
        checked_product(
            &[self.batch, self.head_hidden, output_height, output_width],
            "head conv2 feature map",
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DptWeightRange {
    pub start: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DptHeadWeightLayout {
    pub slots: [DptWeightRange; DPT_WEIGHT_SLOTS],
    pub total: usize,
}

impl DptHeadWeightLayout {
    pub(crate) fn get(&self, slot: usize) -> DptWeightRange {
        self.slots[slot]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DptHeadWeights {
    pub values: Vec<f32>,
}

impl DptHeadWeights {
    pub(crate) fn zeroed(geometry: DptHeadGeometry) -> Result<Self> {
        let layout = dpt_head_weight_layout(geometry)?;
        Ok(Self {
            values: vec![0.0; layout.total],
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DptHeadGradients {
    pub input: Vec<f32>,
    /// Mathematical dL/dw gradients in the same 66-slot order as the weights.
    pub weight_updates: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DptHeadForwardCache {
    samples: Vec<DptSampleCache>,
}

#[derive(Clone, Debug, PartialEq)]
struct DptSampleCache {
    input: Vec<f32>,
    projected: [Vec<f32>; 4],
    reassembled: [Vec<f32>; 4],
    neck: [Vec<f32>; 4],
    fusion: Vec<DptFusionCache>,
    head1_input: Vec<f32>,
    head1_pre: Vec<f32>,
    head_up: Vec<f32>,
    head2_pre: Vec<f32>,
    head2_relu: Vec<f32>,
    head3_pre: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
struct DptFusionCache {
    residual: Option<DptResConfCache>,
    resconf: DptResConfCache,
    projection_input: Vec<f32>,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct DptResConfCache {
    input: Vec<f32>,
    conv1_pre: Vec<f32>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[dpt_head]: {}", message.into()))
}

fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{what} element count overflows")))
    })
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid(format!("{what} size overflows")))
}

pub(crate) fn dpt_head_weight_layout(geometry: DptHeadGeometry) -> Result<DptHeadWeightLayout> {
    geometry.validate()?;
    let mut slots = [DptWeightRange::default(); DPT_WEIGHT_SLOTS];
    let mut cursor = 0usize;
    let mut set = |slot: usize, len: usize| -> Result<()> {
        slots[slot] = DptWeightRange { start: cursor, len };
        cursor = checked_add(cursor, len, "weight arena")?;
        Ok(())
    };

    for tap in 0..4 {
        set(
            2 * tap,
            checked_product(&[geometry.neck[tap], geometry.embed_dim], "projection")?,
        )?;
        set(2 * tap + 1, geometry.neck[tap])?;
    }
    for tap in 0..4 {
        let kernel = RESIZE_KERNELS[tap];
        let weight_count = checked_product(
            &[geometry.neck[tap], geometry.neck[tap], kernel, kernel],
            "resize weights",
        )?;
        let bias_count = if kernel == 0 { 0 } else { geometry.neck[tap] };
        set(8 + 2 * tap, weight_count)?;
        set(9 + 2 * tap, bias_count)?;
    }
    for tap in 0..4 {
        set(
            16 + tap,
            checked_product(&[geometry.fusion, geometry.neck[tap], 9], "neck weights")?,
        )?;
    }
    for base in [20usize, 28, 36, 44, 52] {
        for layer in 0..4 {
            let weight_count = if base == 20 {
                checked_product(&[geometry.fusion, geometry.fusion], "fusion projection")?
            } else {
                checked_product(
                    &[geometry.fusion, geometry.fusion, 9],
                    "residual convolution",
                )?
            };
            set(base + 2 * layer, weight_count)?;
            set(base + 2 * layer + 1, geometry.fusion)?;
        }
    }
    set(
        60,
        checked_product(&[geometry.fusion / 2, geometry.fusion, 9], "head conv1")?,
    )?;
    set(61, geometry.fusion / 2)?;
    set(
        62,
        checked_product(
            &[geometry.head_hidden, geometry.fusion / 2, 9],
            "head conv2",
        )?,
    )?;
    set(63, geometry.head_hidden)?;
    set(64, geometry.head_hidden)?;
    set(65, 1)?;
    debug_assert_eq!(slots[65].start + slots[65].len, cursor);
    Ok(DptHeadWeightLayout {
        slots,
        total: cursor,
    })
}

pub(crate) fn dpt_weight_slot_name(slot: usize) -> String {
    format!("dpt_tensor_{slot:02}")
}

fn slot<'a>(weights: &'a [f32], layout: &DptHeadWeightLayout, index: usize) -> &'a [f32] {
    let range = layout.get(index);
    &weights[range.start..range.start + range.len]
}

fn add_slot(
    updates: &mut [f32],
    layout: &DptHeadWeightLayout,
    index: usize,
    values: &[f32],
) -> Result<()> {
    let range = layout.get(index);
    if values.len() != range.len {
        return Err(invalid(format!(
            "gradient slot {index} has {} values, expected {}",
            values.len(),
            range.len
        )));
    }
    for (dst, value) in updates[range.start..range.start + range.len]
        .iter_mut()
        .zip(values)
    {
        *dst += *value;
    }
    Ok(())
}

fn reassembled_size(size: usize, factor: usize) -> Result<usize> {
    match factor {
        0 => Ok(size.div_ceil(2)),
        1 => Ok(size),
        scale => size
            .checked_mul(scale)
            .ok_or_else(|| invalid("reassembled spatial size overflows")),
    }
}

fn reassembled_shapes(geometry: DptHeadGeometry) -> Result<[(usize, usize); 4]> {
    let mut shapes = [(0, 0); 4];
    for tap in 0..4 {
        shapes[tap] = (
            reassembled_size(geometry.height, geometry.reassemble[tap])?,
            reassembled_size(geometry.width, geometry.reassemble[tap])?,
        );
    }
    Ok(shapes)
}

fn validate_weights(
    weights: &DptHeadWeights,
    geometry: DptHeadGeometry,
) -> Result<DptHeadWeightLayout> {
    let layout = dpt_head_weight_layout(geometry)?;
    if weights.values.len() != layout.total {
        return Err(invalid(format!(
            "weight arena has {} values, expected {}",
            weights.values.len(),
            layout.total
        )));
    }
    Ok(layout)
}

pub(crate) fn forward(
    input: &[f32],
    weights: &DptHeadWeights,
    geometry: DptHeadGeometry,
) -> Result<(Vec<f32>, DptHeadForwardCache)> {
    let layout = validate_weights(weights, geometry)?;
    let input_len = checked_product(
        &[
            geometry.batch,
            4,
            geometry.embed_dim,
            geometry.height,
            geometry.width,
        ],
        "input",
    )?;
    if input.len() != input_len {
        return Err(invalid(format!(
            "input has {} values, expected {input_len}",
            input.len()
        )));
    }
    let (output, cache) = forward_impl(input, &weights.values, geometry, &layout)?;
    Ok((output, cache))
}

fn forward_impl(
    input: &[f32],
    weights: &[f32],
    geometry: DptHeadGeometry,
    layout: &DptHeadWeightLayout,
) -> Result<(Vec<f32>, DptHeadForwardCache)> {
    let shapes = reassembled_shapes(geometry)?;
    let output_h = geometry.output_height()?;
    let output_w = geometry.output_width()?;
    let output_area = checked_product(&[output_h, output_w], "output")?;
    let per_sample_input = checked_product(
        &[4, geometry.embed_dim, geometry.height, geometry.width],
        "per-sample input",
    )?;
    let mut output = Vec::with_capacity(geometry.batch * output_area);
    let mut samples = Vec::with_capacity(geometry.batch);
    for batch in 0..geometry.batch {
        let input_start = batch * per_sample_input;
        let sample_input = &input[input_start..input_start + per_sample_input];
        let mut projected: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
        let mut reassembled: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
        let mut neck: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
        for tap in 0..4 {
            let input_start = tap * geometry.embed_dim * geometry.height * geometry.width;
            let tap_input = &sample_input
                [input_start..input_start + geometry.embed_dim * geometry.height * geometry.width];
            let projection = conv2d_forward(
                tap_input,
                geometry.embed_dim,
                geometry.height,
                geometry.width,
                slot(weights, layout, 2 * tap),
                Some(slot(weights, layout, 2 * tap + 1)),
                geometry.neck[tap],
                1,
                1,
                0,
            )?;
            projected[tap] = projection;
            reassembled[tap] = match geometry.reassemble[tap] {
                4 | 2 => deconv_forward(
                    &projected[tap],
                    geometry.neck[tap],
                    geometry.height,
                    geometry.width,
                    slot(weights, layout, 8 + 2 * tap),
                    slot(weights, layout, 9 + 2 * tap),
                    geometry.reassemble[tap],
                )?,
                1 => projected[tap].clone(),
                0 => conv2d_forward(
                    &projected[tap],
                    geometry.neck[tap],
                    geometry.height,
                    geometry.width,
                    slot(weights, layout, 8 + 2 * tap),
                    Some(slot(weights, layout, 9 + 2 * tap)),
                    geometry.neck[tap],
                    3,
                    2,
                    1,
                )?,
                _ => unreachable!("geometry validates the effective factor tuple"),
            };
            let (neck_h, neck_w) = shapes[tap];
            neck[tap] = conv2d_forward(
                &reassembled[tap],
                geometry.neck[tap],
                neck_h,
                neck_w,
                slot(weights, layout, 16 + tap),
                None,
                geometry.fusion,
                3,
                1,
                1,
            )?;
        }

        let mut current = neck[3].clone();
        let mut current_h = shapes[3].0;
        let mut current_w = shapes[3].1;
        let mut fusion = Vec::with_capacity(4);
        for index in 0..4 {
            let residual = if index == 0 {
                None
            } else {
                let tap = 3 - index;
                let (rh, rw) = shapes[tap];
                let (residual_output, cache) = resconf_forward(
                    &neck[tap],
                    geometry.fusion,
                    rh,
                    rw,
                    slot(weights, layout, 28 + 2 * index),
                    slot(weights, layout, 28 + 2 * index + 1),
                    slot(weights, layout, 36 + 2 * index),
                    slot(weights, layout, 36 + 2 * index + 1),
                )?;
                Some((residual_output, cache))
            };
            if let Some((residual_output, _)) = &residual {
                if residual_output.len() != current.len() {
                    return Err(invalid(format!(
                        "fusion residual {index} shape does not match current tensor"
                    )));
                }
                for (value, residual) in current.iter_mut().zip(residual_output) {
                    *value += *residual;
                }
            }
            let (resconf_output, resconf_cache) = resconf_forward(
                &current,
                geometry.fusion,
                current_h,
                current_w,
                slot(weights, layout, 44 + 2 * index),
                slot(weights, layout, 44 + 2 * index + 1),
                slot(weights, layout, 52 + 2 * index),
                slot(weights, layout, 52 + 2 * index + 1),
            )?;
            let (next_h, next_w) = if index < 3 {
                shapes[2 - index]
            } else {
                (
                    current_h
                        .checked_mul(2)
                        .ok_or_else(|| invalid("final fusion height overflows"))?,
                    current_w
                        .checked_mul(2)
                        .ok_or_else(|| invalid("final fusion width overflows"))?,
                )
            };
            let resized = bilinear_forward(
                &resconf_output,
                geometry.fusion,
                current_h,
                current_w,
                next_h,
                next_w,
            )?;
            current = conv2d_forward(
                &resized,
                geometry.fusion,
                next_h,
                next_w,
                slot(weights, layout, 20 + 2 * index),
                Some(slot(weights, layout, 21 + 2 * index)),
                geometry.fusion,
                1,
                1,
                0,
            )?;
            fusion.push(DptFusionCache {
                residual: residual.map(|(_, cache)| cache),
                resconf: resconf_cache,
                projection_input: resized,
                input_h: current_h,
                input_w: current_w,
                output_h: next_h,
                output_w: next_w,
            });
            current_h = next_h;
            current_w = next_w;
        }

        let head1_input = current;
        let head1_pre = conv2d_forward(
            &head1_input,
            geometry.fusion,
            current_h,
            current_w,
            slot(weights, layout, 60),
            Some(slot(weights, layout, 61)),
            geometry.fusion / 2,
            3,
            1,
            1,
        )?;
        let head_up = bilinear_forward(
            &head1_pre,
            geometry.fusion / 2,
            current_h,
            current_w,
            output_h,
            output_w,
        )?;
        let head2_pre = conv2d_forward(
            &head_up,
            geometry.fusion / 2,
            output_h,
            output_w,
            slot(weights, layout, 62),
            Some(slot(weights, layout, 63)),
            geometry.head_hidden,
            3,
            1,
            1,
        )?;
        let head2_relu = head2_pre
            .iter()
            .map(|&value| value.max(0.0))
            .collect::<Vec<_>>();
        let head3_pre = conv2d_forward(
            &head2_relu,
            geometry.head_hidden,
            output_h,
            output_w,
            slot(weights, layout, 64),
            Some(slot(weights, layout, 65)),
            1,
            1,
            1,
            0,
        )?;
        output.extend(head3_pre.iter().map(|&value| value.max(0.0)));
        samples.push(DptSampleCache {
            input: sample_input.to_vec(),
            projected,
            reassembled,
            neck,
            fusion,
            head1_input,
            head1_pre,
            head_up,
            head2_pre,
            head2_relu,
            head3_pre,
        });
    }
    Ok((output, DptHeadForwardCache { samples }))
}

pub(crate) fn backward(
    output_gradient: &[f32],
    weights: &DptHeadWeights,
    geometry: DptHeadGeometry,
    cache: &DptHeadForwardCache,
) -> Result<DptHeadGradients> {
    let layout = validate_weights(weights, geometry)?;
    let output_h = geometry.output_height()?;
    let output_w = geometry.output_width()?;
    let output_area = checked_product(&[output_h, output_w], "output")?;
    let expected_output = checked_product(&[geometry.batch, output_area], "output gradient")?;
    if output_gradient.len() != expected_output || cache.samples.len() != geometry.batch {
        return Err(invalid("output gradient or forward-cache shape mismatch"));
    }
    let per_sample_input = checked_product(
        &[4, geometry.embed_dim, geometry.height, geometry.width],
        "per-sample input",
    )?;
    let mut input_gradient = vec![0.0; geometry.batch * per_sample_input];
    let mut weight_updates = vec![0.0; layout.total];
    let shapes = reassembled_shapes(geometry)?;
    let fusion_h = shapes[0]
        .0
        .checked_mul(2)
        .ok_or_else(|| invalid("final fusion height overflows"))?;
    let fusion_w = shapes[0]
        .1
        .checked_mul(2)
        .ok_or_else(|| invalid("final fusion width overflows"))?;
    for batch in 0..geometry.batch {
        let seed = &output_gradient[batch * output_area..(batch + 1) * output_area];
        let sample = &cache.samples[batch];
        let head3_gradient = seed
            .iter()
            .zip(&sample.head3_pre)
            .map(|(&grad, &pre)| if pre > 0.0 { grad } else { 0.0 })
            .collect::<Vec<_>>();
        let (mut head2_relu_gradient, dw, db) = conv2d_backward(
            &sample.head2_relu,
            geometry.head_hidden,
            output_h,
            output_w,
            slot(&weights.values, &layout, 64),
            &head3_gradient,
            1,
            1,
            1,
            0,
        )?;
        add_slot(&mut weight_updates, &layout, 64, &dw)?;
        add_slot(&mut weight_updates, &layout, 65, &db)?;
        for (grad, &pre) in head2_relu_gradient.iter_mut().zip(&sample.head2_pre) {
            if pre <= 0.0 {
                *grad = 0.0;
            }
        }
        let (head_up_gradient, dw, db) = conv2d_backward(
            &sample.head_up,
            geometry.fusion / 2,
            output_h,
            output_w,
            slot(&weights.values, &layout, 62),
            &head2_relu_gradient,
            geometry.head_hidden,
            3,
            1,
            1,
        )?;
        add_slot(&mut weight_updates, &layout, 62, &dw)?;
        add_slot(&mut weight_updates, &layout, 63, &db)?;
        let head1_pre_gradient = bilinear_backward(
            &head_up_gradient,
            geometry.fusion / 2,
            fusion_h,
            fusion_w,
            output_h,
            output_w,
        )?;
        let (mut current_gradient, dw, db) = conv2d_backward(
            &sample.head1_input,
            geometry.fusion,
            fusion_h,
            fusion_w,
            slot(&weights.values, &layout, 60),
            &head1_pre_gradient,
            geometry.fusion / 2,
            3,
            1,
            1,
        )?;
        add_slot(&mut weight_updates, &layout, 60, &dw)?;
        add_slot(&mut weight_updates, &layout, 61, &db)?;

        let mut neck_gradients: [Vec<f32>; 4] =
            std::array::from_fn(|tap| vec![0.0; geometry.fusion * shapes[tap].0 * shapes[tap].1]);
        for index in (0..4).rev() {
            let stage = &sample.fusion[index];
            let (bilinear_gradient, dw, db) = conv2d_backward(
                &stage.projection_input,
                geometry.fusion,
                stage.output_h,
                stage.output_w,
                slot(&weights.values, &layout, 20 + 2 * index),
                &current_gradient,
                geometry.fusion,
                1,
                1,
                0,
            )?;
            add_slot(&mut weight_updates, &layout, 20 + 2 * index, &dw)?;
            add_slot(&mut weight_updates, &layout, 21 + 2 * index, &db)?;
            let resconf_gradient = bilinear_backward(
                &bilinear_gradient,
                geometry.fusion,
                stage.input_h,
                stage.input_w,
                stage.output_h,
                stage.output_w,
            )?;
            let (combined_gradient, dw1, db1, dw2, db2) = resconf_backward(
                &resconf_gradient,
                geometry.fusion,
                stage.input_h,
                stage.input_w,
                &stage.resconf,
                slot(&weights.values, &layout, 44 + 2 * index),
                slot(&weights.values, &layout, 44 + 2 * index + 1),
                slot(&weights.values, &layout, 52 + 2 * index),
                slot(&weights.values, &layout, 52 + 2 * index + 1),
            )?;
            add_slot(&mut weight_updates, &layout, 44 + 2 * index, &dw1)?;
            add_slot(&mut weight_updates, &layout, 45 + 2 * index, &db1)?;
            add_slot(&mut weight_updates, &layout, 52 + 2 * index, &dw2)?;
            add_slot(&mut weight_updates, &layout, 53 + 2 * index, &db2)?;
            if index == 0 {
                add_in_place(&mut neck_gradients[3], &combined_gradient)?;
            } else {
                let tap = 3 - index;
                let residual_cache = stage
                    .residual
                    .as_ref()
                    .ok_or_else(|| invalid("missing residual fusion cache"))?;
                let (residual_gradient, dw1, db1, dw2, db2) = resconf_backward(
                    &combined_gradient,
                    geometry.fusion,
                    shapes[tap].0,
                    shapes[tap].1,
                    residual_cache,
                    slot(&weights.values, &layout, 28 + 2 * index),
                    slot(&weights.values, &layout, 29 + 2 * index),
                    slot(&weights.values, &layout, 36 + 2 * index),
                    slot(&weights.values, &layout, 37 + 2 * index),
                )?;
                add_in_place(&mut neck_gradients[tap], &residual_gradient)?;
                add_slot(&mut weight_updates, &layout, 28 + 2 * index, &dw1)?;
                add_slot(&mut weight_updates, &layout, 29 + 2 * index, &db1)?;
                add_slot(&mut weight_updates, &layout, 36 + 2 * index, &dw2)?;
                add_slot(&mut weight_updates, &layout, 37 + 2 * index, &db2)?;
            }
            current_gradient = combined_gradient;
        }

        let input_start = batch * per_sample_input;
        for tap in 0..4 {
            let (neck_h, neck_w) = shapes[tap];
            let (reassembled_gradient, dw) = conv2d_backward_no_bias(
                &sample.reassembled[tap],
                geometry.neck[tap],
                neck_h,
                neck_w,
                slot(&weights.values, &layout, 16 + tap),
                &neck_gradients[tap],
                geometry.fusion,
                3,
                1,
                1,
            )?;
            add_slot(&mut weight_updates, &layout, 16 + tap, &dw)?;
            let projected_gradient = match geometry.reassemble[tap] {
                4 | 2 => {
                    let (input_grad, dw, db) = deconv_backward(
                        &sample.projected[tap],
                        geometry.neck[tap],
                        geometry.height,
                        geometry.width,
                        slot(&weights.values, &layout, 8 + 2 * tap),
                        &reassembled_gradient,
                        geometry.reassemble[tap],
                    )?;
                    add_slot(&mut weight_updates, &layout, 8 + 2 * tap, &dw)?;
                    add_slot(&mut weight_updates, &layout, 9 + 2 * tap, &db)?;
                    input_grad
                }
                1 => reassembled_gradient,
                0 => {
                    let (input_grad, dw, db) = conv2d_backward(
                        &sample.projected[tap],
                        geometry.neck[tap],
                        geometry.height,
                        geometry.width,
                        slot(&weights.values, &layout, 8 + 2 * tap),
                        &reassembled_gradient,
                        geometry.neck[tap],
                        3,
                        2,
                        1,
                    )?;
                    add_slot(&mut weight_updates, &layout, 8 + 2 * tap, &dw)?;
                    add_slot(&mut weight_updates, &layout, 9 + 2 * tap, &db)?;
                    input_grad
                }
                _ => unreachable!(),
            };
            let tap_start = tap * geometry.embed_dim * geometry.height * geometry.width;
            let tap_input = &sample.input
                [tap_start..tap_start + geometry.embed_dim * geometry.height * geometry.width];
            let (tap_gradient, dw, db) = conv2d_backward(
                tap_input,
                geometry.embed_dim,
                geometry.height,
                geometry.width,
                slot(&weights.values, &layout, 2 * tap),
                &projected_gradient,
                geometry.neck[tap],
                1,
                1,
                0,
            )?;
            add_slot(&mut weight_updates, &layout, 2 * tap, &dw)?;
            add_slot(&mut weight_updates, &layout, 2 * tap + 1, &db)?;
            let input_offset = input_start + tap_start;
            input_gradient[input_offset..input_offset + tap_gradient.len()]
                .copy_from_slice(&tap_gradient);
        }
    }
    Ok(DptHeadGradients {
        input: input_gradient,
        weight_updates,
    })
}

fn add_in_place(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(invalid(format!(
            "gradient accumulation length mismatch: {} != {}",
            dst.len(),
            src.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
    Ok(())
}

fn relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&value| value.max(0.0)).collect()
}

fn conv2d_forward(
    input: &[f32],
    input_channels: usize,
    input_h: usize,
    input_w: usize,
    weights: &[f32],
    biases: Option<&[f32]>,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Vec<f32>> {
    if [
        input_channels,
        input_h,
        input_w,
        output_channels,
        kernel,
        stride,
    ]
    .contains(&0)
    {
        return Err(invalid("convolution dimensions must be positive"));
    }
    let input_len = checked_product(&[input_channels, input_h, input_w], "convolution input")?;
    let weight_len = checked_product(
        &[output_channels, input_channels, kernel, kernel],
        "convolution weights",
    )?;
    if input.len() != input_len || weights.len() != weight_len {
        return Err(invalid("convolution input/weight layout mismatch"));
    }
    if let Some(biases) = biases
        && biases.len() != output_channels
    {
        return Err(invalid("convolution bias layout mismatch"));
    }
    let padded_h = input_h
        .checked_add(2 * padding)
        .ok_or_else(|| invalid("convolution height overflows"))?;
    let padded_w = input_w
        .checked_add(2 * padding)
        .ok_or_else(|| invalid("convolution width overflows"))?;
    if padded_h < kernel || padded_w < kernel {
        return Err(invalid("convolution kernel exceeds padded input"));
    }
    let output_h = (padded_h - kernel) / stride + 1;
    let output_w = (padded_w - kernel) / stride + 1;
    let output_area = checked_product(&[output_h, output_w], "convolution output")?;
    let mut output = vec![0.0; output_channels * output_area];
    for out_channel in 0..output_channels {
        let output_start = out_channel * output_area;
        let bias = biases.map_or(0.0, |values| values[out_channel]);
        for out_y in 0..output_h {
            for out_x in 0..output_w {
                let mut sum = bias;
                for in_channel in 0..input_channels {
                    for kernel_y in 0..kernel {
                        let in_y =
                            out_y as isize * stride as isize + kernel_y as isize - padding as isize;
                        if !(0..input_h as isize).contains(&in_y) {
                            continue;
                        }
                        for kernel_x in 0..kernel {
                            let in_x = out_x as isize * stride as isize + kernel_x as isize
                                - padding as isize;
                            if !(0..input_w as isize).contains(&in_x) {
                                continue;
                            }
                            let input_index =
                                (in_channel * input_h + in_y as usize) * input_w + in_x as usize;
                            let weight_index =
                                ((out_channel * input_channels + in_channel) * kernel + kernel_y)
                                    * kernel
                                    + kernel_x;
                            sum += input[input_index] * weights[weight_index];
                        }
                    }
                }
                output[output_start + out_y * output_w + out_x] = sum;
            }
        }
    }
    Ok(output)
}

fn conv2d_backward(
    input: &[f32],
    input_channels: usize,
    input_h: usize,
    input_w: usize,
    weights: &[f32],
    output_gradient: &[f32],
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let padded_h = input_h
        .checked_add(2 * padding)
        .ok_or_else(|| invalid("convolution height overflows"))?;
    let padded_w = input_w
        .checked_add(2 * padding)
        .ok_or_else(|| invalid("convolution width overflows"))?;
    if padded_h < kernel || padded_w < kernel || stride == 0 {
        return Err(invalid("invalid convolution geometry"));
    }
    let output_h = (padded_h - kernel) / stride + 1;
    let output_w = (padded_w - kernel) / stride + 1;
    let output_area = output_h * output_w;
    let expected_output = output_channels * output_area;
    if output_gradient.len() != expected_output
        || input.len() != input_channels * input_h * input_w
        || weights.len() != output_channels * input_channels * kernel * kernel
    {
        return Err(invalid("convolution backward layout mismatch"));
    }
    let mut input_gradient = vec![0.0; input.len()];
    let mut weight_gradient = vec![0.0; weights.len()];
    let mut bias_gradient = vec![0.0; output_channels];
    for out_channel in 0..output_channels {
        for out_y in 0..output_h {
            for out_x in 0..output_w {
                let gradient =
                    output_gradient[out_channel * output_area + out_y * output_w + out_x];
                bias_gradient[out_channel] += gradient;
                for in_channel in 0..input_channels {
                    for kernel_y in 0..kernel {
                        let in_y =
                            out_y as isize * stride as isize + kernel_y as isize - padding as isize;
                        if !(0..input_h as isize).contains(&in_y) {
                            continue;
                        }
                        for kernel_x in 0..kernel {
                            let in_x = out_x as isize * stride as isize + kernel_x as isize
                                - padding as isize;
                            if !(0..input_w as isize).contains(&in_x) {
                                continue;
                            }
                            let input_index =
                                (in_channel * input_h + in_y as usize) * input_w + in_x as usize;
                            let weight_index =
                                ((out_channel * input_channels + in_channel) * kernel + kernel_y)
                                    * kernel
                                    + kernel_x;
                            input_gradient[input_index] += weights[weight_index] * gradient;
                            weight_gradient[weight_index] += input[input_index] * gradient;
                        }
                    }
                }
            }
        }
    }
    Ok((input_gradient, weight_gradient, bias_gradient))
}

fn conv2d_backward_no_bias(
    input: &[f32],
    input_channels: usize,
    input_h: usize,
    input_w: usize,
    weights: &[f32],
    output_gradient: &[f32],
    output_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (input_gradient, weight_gradient, _) = conv2d_backward(
        input,
        input_channels,
        input_h,
        input_w,
        weights,
        output_gradient,
        output_channels,
        kernel,
        stride,
        padding,
    )?;
    Ok((input_gradient, weight_gradient))
}

fn deconv_forward(
    input: &[f32],
    channels: usize,
    input_h: usize,
    input_w: usize,
    weights: &[f32],
    biases: &[f32],
    scale: usize,
) -> Result<Vec<f32>> {
    if scale == 0
        || input.len() != channels * input_h * input_w
        || weights.len() != channels * channels * scale * scale
        || biases.len() != channels
    {
        return Err(invalid("transposed-convolution layout mismatch"));
    }
    let output_h = input_h
        .checked_mul(scale)
        .ok_or_else(|| invalid("transposed-convolution height overflows"))?;
    let output_w = input_w
        .checked_mul(scale)
        .ok_or_else(|| invalid("transposed-convolution width overflows"))?;
    let output_area = output_h * output_w;
    let mut output = vec![0.0; channels * output_area];
    for channel in 0..channels {
        output[channel * output_area..(channel + 1) * output_area].fill(biases[channel]);
    }
    for input_channel in 0..channels {
        for input_y in 0..input_h {
            for input_x in 0..input_w {
                let value = input[(input_channel * input_h + input_y) * input_w + input_x];
                if value == 0.0 {
                    continue;
                }
                for output_channel in 0..channels {
                    for kernel_y in 0..scale {
                        for kernel_x in 0..scale {
                            let weight_index =
                                ((input_channel * channels + output_channel) * scale + kernel_y)
                                    * scale
                                    + kernel_x;
                            let out_y = input_y * scale + kernel_y;
                            let out_x = input_x * scale + kernel_x;
                            let output_index =
                                output_channel * output_area + out_y * output_w + out_x;
                            output[output_index] += value * weights[weight_index];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn deconv_backward(
    input: &[f32],
    channels: usize,
    input_h: usize,
    input_w: usize,
    weights: &[f32],
    output_gradient: &[f32],
    scale: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let output_h = input_h
        .checked_mul(scale)
        .ok_or_else(|| invalid("transposed-convolution height overflows"))?;
    let output_w = input_w
        .checked_mul(scale)
        .ok_or_else(|| invalid("transposed-convolution width overflows"))?;
    let output_area = output_h * output_w;
    if scale == 0
        || input.len() != channels * input_h * input_w
        || weights.len() != channels * channels * scale * scale
        || output_gradient.len() != channels * output_area
    {
        return Err(invalid("transposed-convolution backward layout mismatch"));
    }
    let mut input_gradient = vec![0.0; input.len()];
    let mut weight_gradient = vec![0.0; weights.len()];
    let mut bias_gradient = vec![0.0; channels];
    for output_channel in 0..channels {
        for output_y in 0..output_h {
            for output_x in 0..output_w {
                bias_gradient[output_channel] +=
                    output_gradient[output_channel * output_area + output_y * output_w + output_x];
            }
        }
    }
    for input_channel in 0..channels {
        for input_y in 0..input_h {
            for input_x in 0..input_w {
                let input_index = (input_channel * input_h + input_y) * input_w + input_x;
                let value = input[input_index];
                for output_channel in 0..channels {
                    for kernel_y in 0..scale {
                        for kernel_x in 0..scale {
                            let output_index = output_channel * output_area
                                + (input_y * scale + kernel_y) * output_w
                                + input_x * scale
                                + kernel_x;
                            let gradient = output_gradient[output_index];
                            let weight_index =
                                ((input_channel * channels + output_channel) * scale + kernel_y)
                                    * scale
                                    + kernel_x;
                            input_gradient[input_index] += weights[weight_index] * gradient;
                            weight_gradient[weight_index] += value * gradient;
                        }
                    }
                }
            }
        }
    }
    Ok((input_gradient, weight_gradient, bias_gradient))
}

fn bilinear_coordinates(input: usize, output: usize, position: usize) -> (usize, usize, f32) {
    let source = if output > 1 {
        position as f32 * (input.saturating_sub(1)) as f32 / (output - 1) as f32
    } else {
        0.0
    };
    let low = source.floor() as usize;
    let high = (low + 1).min(input - 1);
    (low, high, source - low as f32)
}

fn bilinear_forward(
    input: &[f32],
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) -> Result<Vec<f32>> {
    if [channels, input_h, input_w, output_h, output_w].contains(&0)
        || input.len() != channels * input_h * input_w
    {
        return Err(invalid("bilinear resize layout mismatch"));
    }
    let input_area = input_h * input_w;
    let output_area = output_h * output_w;
    let mut output = vec![0.0; channels * output_area];
    for channel in 0..channels {
        for y in 0..output_h {
            let (y0, y1, wy) = bilinear_coordinates(input_h, output_h, y);
            for x in 0..output_w {
                let (x0, x1, wx) = bilinear_coordinates(input_w, output_w, x);
                let source = channel * input_area;
                let a = input[source + y0 * input_w + x0];
                let b = input[source + y0 * input_w + x1];
                let c = input[source + y1 * input_w + x0];
                let d = input[source + y1 * input_w + x1];
                output[channel * output_area + y * output_w + x] =
                    (1.0 - wy) * ((1.0 - wx) * a + wx * b) + wy * ((1.0 - wx) * c + wx * d);
            }
        }
    }
    Ok(output)
}

fn bilinear_backward(
    output_gradient: &[f32],
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) -> Result<Vec<f32>> {
    if [channels, input_h, input_w, output_h, output_w].contains(&0)
        || output_gradient.len() != channels * output_h * output_w
    {
        return Err(invalid("bilinear backward layout mismatch"));
    }
    let input_area = input_h * input_w;
    let output_area = output_h * output_w;
    let mut input_gradient = vec![0.0; channels * input_area];
    for channel in 0..channels {
        for y in 0..output_h {
            let (y0, y1, wy) = bilinear_coordinates(input_h, output_h, y);
            for x in 0..output_w {
                let (x0, x1, wx) = bilinear_coordinates(input_w, output_w, x);
                let gradient = output_gradient[channel * output_area + y * output_w + x];
                let base = channel * input_area;
                input_gradient[base + y0 * input_w + x0] += gradient * (1.0 - wy) * (1.0 - wx);
                input_gradient[base + y0 * input_w + x1] += gradient * (1.0 - wy) * wx;
                input_gradient[base + y1 * input_w + x0] += gradient * wy * (1.0 - wx);
                input_gradient[base + y1 * input_w + x1] += gradient * wy * wx;
            }
        }
    }
    Ok(input_gradient)
}

fn resconf_forward(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    conv1_weights: &[f32],
    conv1_biases: &[f32],
    conv2_weights: &[f32],
    conv2_biases: &[f32],
) -> Result<(Vec<f32>, DptResConfCache)> {
    let input_relu = relu(input);
    let conv1_pre = conv2d_forward(
        &input_relu,
        channels,
        height,
        width,
        conv1_weights,
        Some(conv1_biases),
        channels,
        3,
        1,
        1,
    )?;
    let conv1_relu = relu(&conv1_pre);
    let conv2 = conv2d_forward(
        &conv1_relu,
        channels,
        height,
        width,
        conv2_weights,
        Some(conv2_biases),
        channels,
        3,
        1,
        1,
    )?;
    if conv2.len() != input.len() {
        return Err(invalid("residual-conf output geometry mismatch"));
    }
    let output = input.iter().zip(&conv2).map(|(a, b)| a + b).collect();
    Ok((
        output,
        DptResConfCache {
            input: input.to_vec(),
            conv1_pre,
        },
    ))
}

fn resconf_backward(
    output_gradient: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    cache: &DptResConfCache,
    conv1_weights: &[f32],
    conv1_biases: &[f32],
    conv2_weights: &[f32],
    conv2_biases: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let conv1_relu = relu(&cache.conv1_pre);
    let (mut conv1_relu_gradient, conv2_weight_gradient, conv2_bias_gradient) = conv2d_backward(
        &conv1_relu,
        channels,
        height,
        width,
        conv2_weights,
        output_gradient,
        channels,
        3,
        1,
        1,
    )?;
    for (gradient, &pre) in conv1_relu_gradient.iter_mut().zip(&cache.conv1_pre) {
        if pre <= 0.0 {
            *gradient = 0.0;
        }
    }
    let input_relu = relu(&cache.input);
    let (mut input_gradient, conv1_weight_gradient, conv1_bias_gradient) = conv2d_backward(
        &input_relu,
        channels,
        height,
        width,
        conv1_weights,
        &conv1_relu_gradient,
        channels,
        3,
        1,
        1,
    )?;
    for ((gradient, &input), &skip) in input_gradient
        .iter_mut()
        .zip(&cache.input)
        .zip(output_gradient)
    {
        *gradient = if input > 0.0 { *gradient } else { 0.0 } + skip;
    }
    let _ = conv1_biases;
    let _ = conv2_biases;
    Ok((
        input_gradient,
        conv1_weight_gradient,
        conv1_bias_gradient,
        conv2_weight_gradient,
        conv2_bias_gradient,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_geometry() -> DptHeadGeometry {
        DptHeadGeometry {
            batch: 1,
            height: 1,
            width: 1,
            embed_dim: 1,
            fusion: 2,
            head_hidden: 1,
            patch_size: 1,
            neck: [1; 4],
            reassemble: [4, 2, 1, 0],
        }
    }

    fn positive_weights(geometry: DptHeadGeometry) -> DptHeadWeights {
        let layout = dpt_head_weight_layout(geometry).unwrap();
        let mut values = vec![0.025; layout.total];
        for slot_index in 0..DPT_WEIGHT_SLOTS {
            if matches!(
                slot_index,
                1 | 3
                    | 9
                    | 11
                    | 15
                    | 21
                    | 23
                    | 25
                    | 27
                    | 29
                    | 31
                    | 33
                    | 35
                    | 37
                    | 39
                    | 41
                    | 43
                    | 45
                    | 47
                    | 49
                    | 51
                    | 53
                    | 55
                    | 57
                    | 59
                    | 61
                    | 63
                    | 65
            ) {
                let range = layout.get(slot_index);
                values[range.start..range.start + range.len].fill(0.2);
            }
        }
        DptHeadWeights { values }
    }

    fn objective(output: &[f32], seed: &[f32]) -> f32 {
        output.iter().zip(seed).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn dpt_head_default_layout_has_source_slot_count_and_size() {
        let mut geometry = tiny_geometry();
        geometry.embed_dim = 384;
        geometry.fusion = 64;
        geometry.head_hidden = 32;
        geometry.neck = [48, 96, 192, 384];
        let layout = dpt_head_weight_layout(geometry).unwrap();
        assert_eq!(layout.slots.len(), 66);
        assert_eq!(layout.slots[12].len, 0);
        assert_eq!(layout.slots[13].len, 0);
        assert_eq!(layout.total, 2_728_513);
        assert_eq!(layout.slots[0].start, 0);
        assert!(
            layout
                .slots
                .windows(2)
                .all(|pair| { pair[0].start + pair[0].len == pair[1].start })
        );
    }

    #[test]
    fn dpt_head_tap_major_channel_axis_matches_hand_computed_reference() {
        let geometry = DptHeadGeometry {
            batch: 2,
            height: 1,
            width: 1,
            embed_dim: 2,
            fusion: 2,
            head_hidden: 1,
            patch_size: 2,
            neck: [1; 4],
            reassemble: [4, 2, 1, 0],
        };
        let layout = dpt_head_weight_layout(geometry).unwrap();
        let mut weights = DptHeadWeights::zeroed(geometry).unwrap();
        // Isolate tap 0, channel 1.  The rest of the path is a single-channel
        // identity through reassemble, neck, final fusion, and the depth head.
        weights.values[layout.get(0).start + 1] = 1.0;
        let resize = layout.get(8);
        weights.values[resize.start..resize.start + resize.len].fill(1.0);
        weights.values[layout.get(16).start + 4] = 1.0;
        weights.values[layout.get(26).start] = 1.0;
        weights.values[layout.get(60).start + 4] = 1.0;
        weights.values[layout.get(62).start + 4] = 1.0;
        weights.values[layout.get(64).start] = 1.0;
        let input = [
            11.0, 2.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // sample 0
            21.0, 7.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, // sample 1
        ];
        let (output, _) = forward(&input, &weights, geometry).unwrap();
        assert_eq!(
            output,
            [2.0; 4].into_iter().chain([7.0; 4]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dpt_head_align_corners_bilinear_and_adjoint() {
        let input = [0.0, 2.0, 4.0, 6.0];
        let resized = bilinear_forward(&input, 1, 2, 2, 3, 3).unwrap();
        assert_eq!(resized, [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0]);
        let gradient = bilinear_backward(&[1.0; 9], 1, 2, 2, 3, 3).unwrap();
        assert_eq!(gradient, [2.25, 2.25, 2.25, 2.25]);
        let seed = [0.5, -0.25, 0.75, 1.0, -0.5, 0.2, 0.3, -0.7, 1.1];
        let analytical = bilinear_backward(&seed, 1, 2, 2, 3, 3).unwrap();
        let epsilon = 1.0e-3;
        for index in 0..input.len() {
            let mut high = input;
            let mut low = input;
            high[index] += epsilon;
            low[index] -= epsilon;
            let high_value = objective(&bilinear_forward(&high, 1, 2, 2, 3, 3).unwrap(), &seed);
            let low_value = objective(&bilinear_forward(&low, 1, 2, 2, 3, 3).unwrap(), &seed);
            let finite = (high_value - low_value) / (2.0 * epsilon);
            assert!((finite - analytical[index]).abs() < 1.0e-3);
        }
    }

    #[test]
    fn dpt_head_full_backward_matches_finite_differences_for_input_and_all_slots() {
        let geometry = tiny_geometry();
        let mut weights = positive_weights(geometry);
        let input = [0.2, 0.3, 0.4, 0.5];
        let seed = [0.7];
        let (output, cache) = forward(&input, &weights, geometry).unwrap();
        let gradients = backward(&seed, &weights, geometry, &cache).unwrap();
        let epsilon = 2.0e-3;
        for (index, analytical) in gradients.weight_updates.iter().enumerate() {
            let original = weights.values[index];
            weights.values[index] = original + epsilon;
            let high = objective(&forward(&input, &weights, geometry).unwrap().0, &seed);
            weights.values[index] = original - epsilon;
            let low = objective(&forward(&input, &weights, geometry).unwrap().0, &seed);
            weights.values[index] = original;
            let finite = (high - low) / (2.0 * epsilon);
            let tolerance = 3.0e-3 + 2.0e-2 * finite.abs();
            assert!(
                (finite - analytical).abs() <= tolerance,
                "weight[{index}] analytical={analytical} finite={finite}"
            );
        }
        for (index, analytical) in gradients.input.iter().enumerate() {
            let mut high = input;
            let mut low = input;
            high[index] += epsilon;
            low[index] -= epsilon;
            let high_value = objective(&forward(&high, &weights, geometry).unwrap().0, &seed);
            let low_value = objective(&forward(&low, &weights, geometry).unwrap().0, &seed);
            let finite = (high_value - low_value) / (2.0 * epsilon);
            let tolerance = 3.0e-3 + 2.0e-2 * finite.abs();
            assert!(
                (finite - analytical).abs() <= tolerance,
                "input[{index}] analytical={analytical} finite={finite}"
            );
        }
        assert_eq!(output.len(), 1);
    }
}
