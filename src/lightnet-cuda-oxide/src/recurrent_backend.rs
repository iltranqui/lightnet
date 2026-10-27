//! Recurrent backend dispatch.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, same_len, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn lstm_gate_backward_f32_backend(
        &self,
        hidden_gradient: &OxideTensorF32,
        cell: &OxideTensorF32,
        previous_cell: &OxideTensorF32,
        forget: &OxideTensorF32,
        input_gate: &OxideTensorF32,
        candidate: &OxideTensorF32,
        output_gate: &OxideTensorF32,
        next_cell_gradient: &OxideTensorF32,
        elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let hidden_gradient = self.f32v(hidden_gradient)?;
        let cell = self.f32v(cell)?;
        let previous_cell = self.f32v(previous_cell)?;
        let forget = self.f32v(forget)?;
        let input_gate = self.f32v(input_gate)?;
        let candidate = self.f32v(candidate)?;
        let output_gate = self.f32v(output_gate)?;
        let next_cell_gradient = self.f32v(next_cell_gradient)?;
        let op = "lstm_gate_backward_f32";
        for input in [
            &*cell,
            &*previous_cell,
            &*forget,
            &*input_gate,
            &*candidate,
            &*output_gate,
            &*next_cell_gradient,
        ] {
            same_len(op, &hidden_gradient, input)?;
        }
        if hidden_gradient.len() != elements {
            return Err(shape_err(
                op,
                format!(
                    "expected {elements} elements, got {}",
                    hidden_gradient.len()
                ),
            ));
        }
        let mut forget_gradient = self.alloc_uninit(elements)?;
        let mut input_gate_gradient = self.alloc_uninit(elements)?;
        let mut candidate_gradient = self.alloc_uninit(elements)?;
        let mut output_gate_gradient = self.alloc_uninit(elements)?;
        let mut cell_gradient = self.alloc_uninit(elements)?;
        // SAFETY: one thread owns the same element across five disjoint outputs.
        unsafe {
            self.recurrent.lstm_gate_backward(
                &self.stream,
                launch::elementwise(elements, config),
                &hidden_gradient.buf,
                &cell.buf,
                &previous_cell.buf,
                &forget.buf,
                &input_gate.buf,
                &candidate.buf,
                &output_gate.buf,
                &next_cell_gradient.buf,
                &mut forget_gradient,
                &mut input_gate_gradient,
                &mut candidate_gradient,
                &mut output_gate_gradient,
                &mut cell_gradient,
            )
        }
        .map_err(launch_err)?;
        let shape = hidden_gradient.shape();
        Ok((
            OxideTensorF32::new(forget_gradient, shape),
            OxideTensorF32::new(input_gate_gradient, shape),
            OxideTensorF32::new(candidate_gradient, shape),
            OxideTensorF32::new(output_gate_gradient, shape),
            OxideTensorF32::new(cell_gradient, shape),
        ))
    }

    pub(crate) fn sequence_slice_f32_backend(
        &self,
        input: &OxideTensorF32,
        input_shape: &[usize],
        step_index: usize,
        steps: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "sequence_slice_f32";
        if steps == 0 || step_index >= steps {
            return Err(shape_err(
                op,
                "invalid recurrent step index/count".to_owned(),
            ));
        }
        let input_elements = input_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "input shape element count overflow".to_owned()))?;
        if input_elements % steps != 0 {
            return Err(shape_err(
                op,
                format!("input has {input_elements} elements, not divisible by {steps} steps"),
            ));
        }
        let step_elements = input_elements / steps;
        let output_elements = output_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "output shape element count overflow".to_owned()))?;
        if output_elements != step_elements {
            return Err(shape_err(
                op,
                format!("output has {output_elements} elements, expected {step_elements}"),
            ));
        }
        let input = self.f32v(input)?;
        if input.len() != input_elements {
            return Err(shape_err(
                op,
                format!(
                    "input tensor has {} elements, expected {input_elements}",
                    input.len()
                ),
            ));
        }
        let offset = step_index
            .checked_mul(step_elements)
            .ok_or_else(|| shape_err(op, "sequence input offset overflow".to_owned()))?;
        let mut output = self.alloc_uninit(step_elements)?;
        // SAFETY: checked shapes and step index keep this copy within input.
        unsafe {
            self.recurrent.sequence_slice_step(
                &self.stream,
                launch::elementwise(step_elements, config),
                &input.buf,
                &mut output,
                offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn sequence_concat_f32_backend(
        &self,
        steps: &[&OxideTensorF32],
        output_shape: &[usize],
        step_elements: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "sequence_concat_f32";
        if steps.is_empty() {
            return Err(shape_err(
                op,
                "at least one recurrent step is required".to_owned(),
            ));
        }
        let expected = steps
            .len()
            .checked_mul(step_elements)
            .ok_or_else(|| shape_err(op, "output element count overflow".to_owned()))?;
        let shape_elements = output_shape
            .iter()
            .try_fold(1usize, |product, dim| product.checked_mul(*dim))
            .ok_or_else(|| shape_err(op, "output shape element count overflow".to_owned()))?;
        if shape_elements != expected {
            return Err(shape_err(
                op,
                format!(
                    "shape {output_shape:?} has {shape_elements} elements, expected {expected}"
                ),
            ));
        }
        let mut output = self.alloc_uninit(expected)?;
        for (step_index, input) in steps.iter().enumerate() {
            let input = self.f32v(input)?;
            if input.len() != step_elements {
                return Err(shape_err(
                    op,
                    format!(
                        "step {step_index} has {} elements, expected {step_elements}",
                        input.len()
                    ),
                ));
            }
            let offset = step_index
                .checked_mul(step_elements)
                .ok_or_else(|| shape_err(op, "step output offset overflow".to_owned()))?;
            // SAFETY: each ordered launch writes a distinct output slice.
            unsafe {
                self.recurrent.sequence_concat_step(
                    &self.stream,
                    launch::elementwise(step_elements, config),
                    &input.buf,
                    &mut output,
                    offset,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(output, output_shape))
    }
}
