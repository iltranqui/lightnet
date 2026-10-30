//! Dense detection-head forward backend operations.

use cuda_core::DeviceBuffer;
use lightnet_cuda::{ElementwiseLaunchConfig, YoloGpuLossArgs, YoloImageMetrics};

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    /// Native classic-YOLO CIoU loss: four kernels (background + threshold positives,
    /// sequential per-image truth assignment, optional per-cell averaging, reduction).
    pub(crate) fn yolo_loss_backend_f32(
        &self, output: &OxideTensorF32, truth: &OxideTensorF32, shape: &[usize],
        args: &YoloGpuLossArgs, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, Vec<YoloImageMetrics>), OxideError> {
        let op = "yolo_loss_f32";
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let truth_f = self.f32v(truth)?;
        let truth = &*truth_f;
        let entries = args.classes + 5;
        if shape.len() != 4 || shape.contains(&0) || output.shape() != shape || shape[1] != args.anchors * entries
            || args.anchors == 0 || args.mask.len() != args.anchors
        {
            return Err(shape_err(op, format!("incompatible YOLO shape {shape:?}, classes {}, anchors {}", args.classes, args.anchors)));
        }
        if args.total_anchors > 32 || args.max_boxes > 1024 || args.biases.len() < 2 * args.total_anchors
            || args.mask.iter().any(|&m| m >= args.total_anchors) || args.truth_size < 5
            || args.truths < args.max_boxes * args.truth_size || truth.len() < shape[0] * args.truths
        {
            return Err(shape_err(op, "unsupported anchor/box counts or truth layout".into()));
        }
        let batches = shape[0];
        let spatial = shape[2] * shape[3];
        let outputs = shape[1] * spatial;
        let dims: Vec<u32> = [
            shape[3], shape[2], args.anchors, args.total_anchors, args.classes, args.max_boxes, args.truth_size,
            args.truths, outputs, args.net_width, args.net_height, args.new_coords as usize, args.objectness_smooth as usize,
        ].iter().map(|&v| v as u32).collect();
        let params = [args.ignore_thresh, args.iou_normalizer, args.obj_normalizer, args.max_delta, args.iou_thresh, args.truth_thresh];
        let mask: Vec<u32> = args.mask.iter().map(|&m| m as u32).collect();
        let dims = DeviceBuffer::<u32>::from_host(&self.stream, &dims)?;
        let params = DeviceBuffer::<f32>::from_host(&self.stream, &params)?;
        let mask = DeviceBuffer::<u32>::from_host(&self.stream, &mask)?;
        let biases = DeviceBuffer::<f32>::from_host(&self.stream, &args.biases)?;
        let per_image = args.anchors * spatial;
        let mut delta = self.alloc(batches * outputs)?;
        let mut flags = self.alloc(batches * per_image)?;
        let mut metrics = self.alloc(batches * 7)?;
        // SAFETY: shapes validated above; each kernel's threads own disjoint delta/flag/metric slots, and
        // the sequential kernels are ordered on one stream.
        unsafe {
            self.dense_loss.yolo_loss_background(
                &self.stream, launch::elementwise(batches * per_image, config),
                &output.buf, &truth.buf, &mut delta, &mut flags, &dims, &params, &mask, &biases, batches,
            ).map_err(launch_err)?;
            self.dense_loss.yolo_loss_positives(
                &self.stream, launch::elementwise(batches, config),
                &output.buf, &truth.buf, &mut delta, &mut metrics, &dims, &params, &mask, &biases, batches,
            ).map_err(launch_err)?;
            if args.iou_thresh < 1.0 {
                self.dense_loss.yolo_loss_average(
                    &self.stream, launch::elementwise(batches * per_image, config), &mut delta, &dims, batches,
                ).map_err(launch_err)?;
            }
            self.dense_loss.yolo_loss_reduce(
                &self.stream, launch::elementwise(batches, config), &delta, &flags, &mut metrics, &dims, batches,
            ).map_err(launch_err)?;
        }
        let records = metrics.to_host_vec(&self.stream)?;
        let per_image_metrics = records.chunks_exact(7).map(YoloImageMetrics::from_record).collect();
        Ok((OxideTensorF32::new(delta, shape), per_image_metrics))
    }

    pub(crate) fn yolo_activate_backend_f32(
        &self, x: &OxideTensorF32, shape: &[usize], classes: usize,
        scale_xy: f32, new_coords: bool, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "yolo_activate_f32";
        let entries = classes.checked_add(5).ok_or_else(|| shape_err(op, "entry count overflow".into()))?;
        if shape.len() != 4 || shape.contains(&0) || shape.iter().product::<usize>() != x.len() || x.shape() != shape || shape[1] % entries != 0 {
            return Err(shape_err(op, format!("incompatible YOLO shape {shape:?}, classes {classes}")));
        }
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: shape validation bounds input and output; one thread per element.
        unsafe { self.dense_loss.yolo_activate(&self.stream, launch::elementwise(x.len(), config), &x.buf, &mut output, shape[2] * shape[3], entries, scale_xy, new_coords) }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }
    pub(crate) fn centernet_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "centernet_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] < classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible CenterNet shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.centernet_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn modern_yolo_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "modern_yolo_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] <= classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible modern-YOLO shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions make the channel/spatial mapping valid;
        // one thread writes one disjoint output element.
        unsafe {
            self.dense_loss.modern_yolo_forward(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }
}

impl OxideBackend {
    pub(crate) fn segmentation_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() && shape.len() == 4 && shape[1] == classes && classes > 1 {
            let out = self.spatial_softmax_f16(&self.as_half(x)?, shape, temperature, config)?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "segmentation_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || shape[1] != classes
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("incompatible segmentation shape/classes: {shape:?}, {classes}"),
            ));
        }
        let spatial = shape[2] * shape[3];
        let pixels = shape[0] * spatial;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated dimensions ensure one thread per batch/pixel owns
        // exactly one slot in every class plane.
        unsafe {
            self.dense_loss.segmentation_forward(
                &self.stream,
                launch::elementwise(pixels, config),
                &x.buf,
                &mut output,
                shape[1],
                spatial,
                classes,
                temperature,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn jev_ce_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "jev_ce_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || groups == 0
            || candidates == 0
            || groups
                .checked_mul(candidates)
                .and_then(|per_sample| per_sample.checked_mul(shape[0]))
                != Some(x.len())
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible JevCe shape/groups/candidates: {shape:?}, {groups}, {candidates}"
                ),
            ));
        }
        let work = shape[0] * groups;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: every sample/group owns a disjoint candidate range in the
        // selected storage layout.
        unsafe {
            self.dense_loss.jev_ce_forward(
                &self.stream,
                launch::elementwise(work, config),
                &x.buf,
                &mut output,
                groups,
                candidates,
                temperature,
                channel_major as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn laya_loss_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        max_options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "laya_loss_forward_f32";
        if shape.len() != 4
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || shape[0] == 0
            || max_options == 0
            || max_options.checked_mul(shape[0]) != Some(x.len())
        {
            return Err(shape_err(
                op,
                format!("incompatible LayaLoss shape/options: {shape:?}, {max_options}"),
            ));
        }
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: validated batch/options spans are disjoint; one thread owns
        // one complete sample row.
        unsafe {
            self.dense_loss.laya_loss_forward(
                &self.stream,
                launch::elementwise(shape[0], config),
                &x.buf,
                &mut output,
                max_options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn region_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "region_forward_f32";
        if shape.len() != 4
            || shape[1] != anchors.saturating_mul(entries)
            || shape.iter().product::<usize>() != x.len()
            || x.shape() != shape
            || anchors == 0
            || entries < classes.saturating_add(5)
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible Region shape/parameters: {shape:?}, anchors={anchors}, entries={entries}, classes={classes}"
                ),
            ));
        }
        let spatial = shape[2] * shape[3];
        let groups = shape[0] * spatial * anchors;
        let mut output = self.alloc_uninit(x.len())?;
        // SAFETY: shape and Region layout checks establish valid input/output
        // rows; one thread writes one disjoint anchor/location row.
        unsafe {
            self.dense_loss.region_forward(
                &self.stream,
                launch::elementwise(groups, config),
                &x.buf,
                &mut output,
                spatial,
                anchors,
                entries,
                classes,
                softmax_classes as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, shape))
    }

    pub(crate) fn region_backward_backend_f32(
        &self,
        output: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "region_backward_f32";
        if shape.len() != 4
            || shape[1] != anchors.saturating_mul(entries)
            || shape.iter().product::<usize>() != output.len()
            || output.shape() != shape
            || output_grad.shape() != shape
            || output_grad.len() != output.len()
            || anchors == 0
            || entries < classes.saturating_add(5)
            || classes == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!(
                    "incompatible Region shape/parameters: {shape:?}, anchors={anchors}, entries={entries}, classes={classes}"
                ),
            ));
        }
        let spatial = shape[2] * shape[3];
        let groups = shape[0] * spatial * anchors;
        let mut input_grad = self.alloc_uninit(output.len())?;
        // SAFETY: each anchor/location thread writes unique channel-major input
        // slots, and the output/gradient storage shapes were checked above.
        unsafe {
            self.dense_loss.region_backward(
                &self.stream,
                launch::elementwise(groups, config),
                &output.buf,
                &output_grad.buf,
                &mut input_grad,
                spatial,
                anchors,
                entries,
                classes,
                softmax_classes as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_grad, shape))
    }

    pub(crate) fn jev_ce_loss_backend_f32(
        &self,
        logits: &OxideTensorF32,
        truth: &OxideTensorF32,
        shape: &[usize],
        groups: usize,
        candidates: usize,
        temperature: f32,
        normalize: bool,
        ignore_empty: bool,
        label_smoothing: f32,
        scale: f32,
        channel_major: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, OxideTensorF32), OxideError> {
        let logits_f = self.f32v(logits)?;
        let logits = &*logits_f;
        let truth_f = self.f32v(truth)?;
        let truth = &*truth_f;
        const OP: &str = "jev_ce_loss_f32";
        let batch = if shape.len() == 4 { shape[0] } else { 0 };
        if shape.len() != 4
            || batch == 0
            || groups == 0
            || candidates == 0
            || groups
                .checked_mul(candidates)
                .and_then(|per_sample| per_sample.checked_mul(batch))
                != Some(logits.len())
            || logits.len() != truth.len()
            || logits.shape() != shape
            || truth.shape() != shape
        {
            return Err(shape_err(
                OP,
                format!(
                    "incompatible JevCe shape/groups/candidates: {shape:?}, {groups}, {candidates}"
                ),
            ));
        }
        let work = batch * groups;
        let mut delta = self.alloc_uninit(logits.len())?;
        let mut losses = self.alloc(work)?;
        let mut active = self.alloc(work)?;
        // SAFETY: one thread per sample/group owns its candidate entries and
        // its loss/active slot.
        unsafe {
            self.dense_loss.jev_ce_loss(
                &self.stream,
                launch::elementwise(work, config),
                &logits.buf,
                &truth.buf,
                &mut delta,
                &mut losses,
                &mut active,
                batch,
                groups,
                candidates,
                temperature,
                normalize as u32,
                ignore_empty as u32,
                label_smoothing,
                scale,
                channel_major as u32,
            )
        }
        .map_err(launch_err)?;
        let loss_host = losses.to_host_vec(&self.stream)?;
        let active_host = active.to_host_vec(&self.stream)?;
        let (mut sum, mut count) = (0.0f64, 0usize);
        for (loss, active) in loss_host.iter().zip(&active_host) {
            if *active != 0.0 {
                sum += *loss as f64;
                count += 1;
            }
        }
        let mean = if count == 0 {
            0.0
        } else {
            (sum / count as f64) as f32
        };
        Ok((mean, OxideTensorF32::new(delta, shape)))
    }
}

impl OxideBackend {
    pub(crate) fn segmentation_loss_backend_f32(
        &self,
        output: &OxideTensorF32,
        truth: &OxideTensorF32,
        shape: &[usize],
        classes: usize,
        spatial: usize,
        softmax_loss: bool,
        scale: f32,
        gate_strength: f32,
        gate_t_correct: f32,
        gate_max_compensation: f32,
        max_delta: f32,
        clipping: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(f32, OxideTensorF32), OxideError> {
        let output_f = self.f32v(output)?;
        let output = &*output_f;
        let truth_f = self.f32v(truth)?;
        let truth = &*truth_f;
        const OP: &str = "segmentation_loss_f32";
        let batch = if shape.len() == 4 { shape[0] } else { 0 };
        let expected_spatial = if shape.len() == 4 {
            shape[2] * shape[3]
        } else {
            0
        };
        if shape.len() != 4
            || batch == 0
            || classes == 0
            || spatial == 0
            || spatial != expected_spatial
            || classes
                .checked_mul(spatial)
                .and_then(|per_sample| per_sample.checked_mul(batch))
                != Some(output.len())
            || output.len() != truth.len()
            || output.shape() != shape
            || truth.shape() != shape
        {
            return Err(shape_err(
                OP,
                format!(
                    "incompatible segmentation shape/classes/spatial: {shape:?}, {classes}, {spatial}"
                ),
            ));
        }
        let total = output.len();
        let mut delta = self.alloc_uninit(total)?;
        let mut partial_loss = self.alloc(batch)?;
        let mut suppressed = self.alloc(batch)?;
        // SAFETY: one thread per sample owns its output/truth reads and its
        // per-sample loss/suppressed slots; the delta slots are disjoint.
        unsafe {
            self.dense_loss.segmentation_loss(
                &self.stream,
                launch::elementwise(batch, config),
                &output.buf,
                &truth.buf,
                &mut delta,
                &mut partial_loss,
                &mut suppressed,
                batch,
                classes,
                spatial,
                softmax_loss as u32,
                scale,
                gate_strength,
                gate_t_correct,
            )
        }
        .map_err(launch_err)?;
        let partial_host = partial_loss.to_host_vec(&self.stream)?;
        let suppressed_host = suppressed.to_host_vec(&self.stream)?;
        let mut loss_sum = 0.0f32;
        let mut suppressed_total = 0.0f32;
        for (loss, count) in partial_host.iter().zip(&suppressed_host) {
            loss_sum += *loss;
            suppressed_total += *count;
        }
        let total_pixels = batch * spatial;
        let active = total_pixels as f32 - suppressed_total;
        let mut compensation = if active > 0.0 && active < total_pixels as f32 {
            (total_pixels as f32 / active).min(gate_max_compensation)
        } else {
            1.0
        };
        if compensation <= 1.0 {
            compensation = 1.0;
        }
        // SAFETY: one thread per delta element.
        unsafe {
            self.dense_loss.segmentation_scale_delta(
                &self.stream,
                launch::elementwise(total, config),
                &mut delta,
                compensation,
                max_delta,
                clipping as u32,
            )
        }
        .map_err(launch_err)?;
        let mean = if total_pixels == 0 {
            0.0
        } else {
            loss_sum / total_pixels as f32
        };
        Ok((mean, OxideTensorF32::new(delta, shape)))
    }

}
