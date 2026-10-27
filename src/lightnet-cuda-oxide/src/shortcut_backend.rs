//! Shortcut backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32ShortcutWeightNormalization, F32TensorOpsBackend};

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn shortcut_add_backend_f32(
        &self,
        x: &OxideTensorF32,
        y: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let y_f = self.f32v(y)?;
        let y = &*y_f;
        self.vector_add_f32(x, y, config)
    }

    pub(crate) fn shortcut_weight_gradient_backend_f32(
        &self,
        delta: &OxideTensorF32,
        source: &OxideTensorF32,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        layer_step: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "shortcut_weight_gradient_f32";
        if batch == 0 || per_sample == 0 || layer_step == 0 || per_sample % layer_step != 0 {
            return Err(shape_err(
                op,
                "invalid batch or shortcut slot geometry".to_owned(),
            ));
        }
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        let source = self.f32v(source)?;
        let source = &*source;
        if delta.len() != batch * per_sample || source.len() != batch * source_per_sample {
            return Err(shape_err(
                op,
                format!(
                    "delta/source lengths are {}/{}, expected {}/{}",
                    delta.len(),
                    source.len(),
                    batch * per_sample,
                    batch * source_per_sample
                ),
            ));
        }
        let mut gradient = self.alloc_uninit(layer_step)?;
        // SAFETY: one thread owns each coefficient slot and reads only the
        // validated delta/source extents.
        unsafe {
            self.shortcut.shortcut_weight_gradient(
                &self.stream,
                launch::elementwise(layer_step, config),
                &delta.buf,
                &source.buf,
                &mut gradient,
                batch,
                per_sample,
                source_per_sample,
                layer_step,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, &[layer_step]))
    }

    pub(crate) fn shortcut_weight_gradient_accumulate_backend_f32(
        &self,
        accumulated: &OxideTensorF32,
        delta: &OxideTensorF32,
        source: &OxideTensorF32,
        batch: usize,
        per_sample: usize,
        source_per_sample: usize,
        num_inputs: usize,
        layer_step: usize,
        source_index: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "shortcut_weight_gradient_accumulate_f32";
        if batch == 0
            || per_sample == 0
            || layer_step == 0
            || per_sample % layer_step != 0
            || num_inputs == 0
            || source_index >= num_inputs
        {
            return Err(shape_err(
                op,
                "invalid shortcut gradient geometry".to_owned(),
            ));
        }
        let total = num_inputs
            .checked_mul(layer_step)
            .ok_or_else(|| shape_err(op, "shortcut weight count overflows usize".to_owned()))?;
        let accumulated = self.f32v(accumulated)?;
        let accumulated = &*accumulated;
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        let source = self.f32v(source)?;
        let source = &*source;
        if accumulated.len() != total
            || delta.len() != batch * per_sample
            || source.len() != batch * source_per_sample
        {
            return Err(shape_err(
                op,
                format!(
                    "tensor lengths are {}/{}/{}, expected {}/{}/{}",
                    accumulated.len(),
                    delta.len(),
                    source.len(),
                    total,
                    batch * per_sample,
                    batch * source_per_sample
                ),
            ));
        }
        let mut out = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each output coefficient; all input extents
        // and source-row bounds are validated above.
        unsafe {
            self.shortcut.shortcut_weight_gradient_accumulate(
                &self.stream,
                launch::elementwise(total, config),
                &accumulated.buf,
                &delta.buf,
                &source.buf,
                &mut out,
                batch,
                per_sample,
                source_per_sample,
                num_inputs,
                layer_step,
                source_index,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[num_inputs, layer_step]))
    }

    pub(crate) fn shortcut_weight_normalized_gradient_backend_f32(
        &self,
        raw_weights: &OxideTensorF32,
        products: &OxideTensorF32,
        num_inputs: usize,
        layer_step: usize,
        normalization: F32ShortcutWeightNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "shortcut_weight_normalized_gradient_f32";
        if num_inputs == 0 || layer_step == 0 {
            return Err(shape_err(op, "empty shortcut weight geometry".to_owned()));
        }
        let total = num_inputs
            .checked_mul(layer_step)
            .ok_or_else(|| shape_err(op, "shortcut weight count overflows usize".to_owned()))?;
        let raw_weights = self.f32v(raw_weights)?;
        let raw_weights = &*raw_weights;
        let products = self.f32v(products)?;
        let products = &*products;
        if raw_weights.len() != total || products.len() != total {
            return Err(shape_err(
                op,
                format!(
                    "raw/product lengths are {}/{}, expected {total}",
                    raw_weights.len(),
                    products.len()
                ),
            ));
        }
        let normalization_code = match normalization {
            F32ShortcutWeightNormalization::None => 0,
            F32ShortcutWeightNormalization::Relu => 1,
            F32ShortcutWeightNormalization::Softmax => 2,
        };
        let mut gradient = self.alloc_uninit(total)?;
        // SAFETY: one thread owns each result coefficient; input rows and slots
        // are bounded by the validated total length.
        unsafe {
            self.shortcut.shortcut_weight_normalized_gradient(
                &self.stream,
                launch::elementwise(total, config),
                &raw_weights.buf,
                &products.buf,
                &mut gradient,
                num_inputs,
                layer_step,
                normalization_code,
                cpp_grad as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, &[num_inputs, layer_step]))
    }

    pub(crate) fn shortcut_fit_backend_f32(
        &self,
        x: &OxideTensorF32,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "shortcut_fit_f32";
        let out_len: usize = output_shape.iter().product();
        if batch == 0
            || x.len() % batch != 0
            || out_len % batch != 0
            || out_len == 0
            || x.is_empty()
        {
            return Err(shape_err(
                op,
                format!(
                    "batch {batch}, {} source elements and output shape {output_shape:?} are inconsistent",
                    x.len()
                ),
            ));
        }
        let (from_len, to_len) = (x.len() / batch, out_len / batch);
        let mut out = self.alloc_uninit(out_len)?;
        // SAFETY: one thread per output element; reads stay inside `x`
        // (`i < from_len`, `b < batch`).
        unsafe {
            self.shortcut.shortcut_fit(
                &self.stream,
                launch::elementwise(out_len, config),
                &x.buf,
                &mut out,
                from_len,
                to_len,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }
}
