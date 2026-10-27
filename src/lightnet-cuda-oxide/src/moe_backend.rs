//! MoE router and gate backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn moe_router_gates_backend_f32(
        &self,
        input: &OxideTensorF32,
        router_weights: &OxideTensorF32,
        router_biases: &OxideTensorF32,
        batch: usize,
        channels: usize,
        spatial: usize,
        experts: usize,
        top_k: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let router_weights_f = self.f32v(router_weights)?;
        let router_weights = &*router_weights_f;
        let router_biases_f = self.f32v(router_biases)?;
        let router_biases = &*router_biases_f;
        let op = "moe_router_gates_f32";
        if batch == 0 || channels == 0 || spatial == 0 || experts == 0 {
            return Err(shape_err(
                op,
                "batch, channels, spatial, and experts must be nonzero".into(),
            ));
        }
        if input.len() != batch * channels * spatial
            || router_weights.len() != experts * channels
            || router_biases.len() != experts
        {
            return Err(shape_err(
                op,
                format!(
                    "input/weights/biases have lengths {}/{}/{}, expected {}/{}/{}",
                    input.len(),
                    router_weights.len(),
                    router_biases.len(),
                    batch * channels * spatial,
                    experts * channels,
                    experts
                ),
            ));
        }
        let mut logits = self.alloc_uninit(batch * experts)?;
        // SAFETY: one thread per (sample, expert) logit, each uniquely written.
        unsafe {
            self.moe.moe_router_logits(
                &self.stream,
                launch::elementwise(logits.len(), config),
                &input.buf,
                &router_weights.buf,
                &router_biases.buf,
                &mut logits,
                channels,
                spatial,
                experts,
                batch,
                temperature,
            )
        }
        .map_err(launch_err)?;
        let mut gates = self.alloc_uninit(batch * experts)?;
        // SAFETY: one thread per batch row, each uniquely writes `experts` gates.
        unsafe {
            self.moe.moe_router_softmax_topk(
                &self.stream,
                launch::elementwise(batch, config),
                &logits,
                &mut gates,
                batch,
                experts,
                top_k,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gates, &[batch, experts]))
    }
    pub(crate) fn moe_router_pool_backend_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input = self.f32v(input)?;
        let op = "moe_router_pool_f32";
        if batch == 0 || channels == 0 || spatial == 0 {
            return Err(shape_err(
                op,
                "batch, channels, and spatial must be nonzero".into(),
            ));
        }
        if input.len() != batch * channels * spatial {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {}",
                    input.len(),
                    batch * channels * spatial
                ),
            ));
        }
        let mut pooled = self.alloc_uninit(batch * channels)?;
        // SAFETY: one thread uniquely writes one sample/channel descriptor.
        unsafe {
            self.moe.moe_router_pool(
                &self.stream,
                launch::elementwise(batch * channels, config),
                &input.buf,
                &mut pooled,
                channels,
                spatial,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(pooled, &[batch, channels]))
    }
    pub(crate) fn moe_router_pool_backward_backend_f32(
        &self,
        pooled_gradient: &OxideTensorF32,
        input_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let pooled_gradient = self.f32v(pooled_gradient)?;
        let op = "moe_router_pool_backward_f32";
        if input_shape.len() != 4
            || input_shape[0] == 0
            || input_shape[1] == 0
            || input_shape[2] == 0
            || input_shape[3] == 0
        {
            return Err(shape_err(
                op,
                format!("input shape must be nonzero NCHW, got {input_shape:?}"),
            ));
        }
        let (batch, channels) = (input_shape[0], input_shape[1]);
        let spatial = input_shape[2] * input_shape[3];
        if pooled_gradient.len() != batch * channels {
            return Err(shape_err(
                op,
                format!(
                    "pooled gradient has {} elements, expected {}",
                    pooled_gradient.len(),
                    batch * channels
                ),
            ));
        }
        let mut input_gradient = self.alloc_uninit(batch * channels * spatial)?;
        // SAFETY: one thread uniquely writes one input-gradient element.
        unsafe {
            self.moe.moe_router_pool_backward(
                &self.stream,
                launch::elementwise(input_gradient.len(), config),
                &pooled_gradient.buf,
                &mut input_gradient,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(input_gradient, input_shape))
    }
    pub(crate) fn moe_gate_dot_backend_f32(
        &self,
        output_gradient: &OxideTensorF32,
        expert_output: &OxideTensorF32,
        batch: usize,
        per_image: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_gradient = self.f32v(output_gradient)?;
        let expert_output = self.f32v(expert_output)?;
        let op = "moe_gate_dot_f32";
        let expected = batch * per_image;
        if batch == 0 || per_image == 0 {
            return Err(shape_err(
                op,
                "batch and per-image size must be nonzero".into(),
            ));
        }
        if output_gradient.len() != expected || expert_output.len() != expected {
            return Err(shape_err(
                op,
                format!(
                    "gradient/output lengths are {}/{}, expected {}",
                    output_gradient.len(),
                    expert_output.len(),
                    expected
                ),
            ));
        }
        let mut partial = self.alloc_uninit(batch * 256)?;
        // SAFETY: one thread owns each (sample, reduction-lane) partial.
        unsafe {
            self.moe.moe_gate_dot_partial(
                &self.stream,
                launch::elementwise(batch * 256, config),
                &output_gradient.buf,
                &expert_output.buf,
                &mut partial,
                per_image,
                batch,
            )
        }
        .map_err(launch_err)?;
        let mut dots = self.alloc_uninit(batch)?;
        // SAFETY: one thread uniquely folds each sample's 256 partial sums.
        unsafe {
            self.moe.moe_gate_dot_reduce(
                &self.stream,
                launch::elementwise(batch, config),
                &partial,
                &mut dots,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dots, &[batch]))
    }
    pub(crate) fn moe_gate_column_backend_f32(
        &self,
        gates: &OxideTensorF32,
        batch: usize,
        experts: usize,
        expert: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let gates = self.f32v(gates)?;
        let op = "moe_gate_column_f32";
        if batch == 0 || experts == 0 || expert >= experts || gates.len() != batch * experts {
            return Err(shape_err(
                op,
                format!(
                    "invalid gate shape/slot: batch={batch}, experts={experts}, expert={expert}, len={}",
                    gates.len()
                ),
            ));
        }
        let mut column = self.alloc_uninit(batch)?;
        // SAFETY: one thread per sample reads its gate and writes one column element.
        unsafe {
            self.moe.moe_gate_column(
                &self.stream,
                launch::elementwise(batch, config),
                &gates.buf,
                &mut column,
                experts,
                expert,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(column, &[batch]))
    }
    pub(crate) fn moe_router_backward_backend_f32(
        &self,
        gates: &OxideTensorF32,
        output_gradient: &OxideTensorF32,
        expert_outputs: &[OxideTensorF32],
        batch: usize,
        experts: usize,
        per_image: usize,
        load_balance: f32,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let gates = self.f32v(gates)?;
        let output_gradient = self.f32v(output_gradient)?;
        let op = "moe_router_backward_f32";
        if batch == 0 || experts == 0 || per_image == 0 || expert_outputs.len() != experts {
            return Err(shape_err(
                op,
                format!(
                    "invalid dimensions or expert count: {batch}x{experts}x{per_image}, {} outputs",
                    expert_outputs.len()
                ),
            ));
        }
        let gate_count = batch * experts;
        let output_count = batch * per_image;
        if gates.len() != gate_count || output_gradient.len() != output_count {
            return Err(shape_err(
                op,
                format!(
                    "gate/gradient lengths are {}/{}, expected {gate_count}/{output_count}",
                    gates.len(),
                    output_gradient.len()
                ),
            ));
        }
        let mut partial = self.alloc_uninit(batch * 256)?;
        let mut dots = self.alloc_uninit(gate_count)?;
        for (expert, output) in expert_outputs.iter().enumerate() {
            let output = self.f32v(output)?;
            if output.len() != output_count {
                return Err(shape_err(
                    op,
                    format!(
                        "expert {expert} has {} values, expected {output_count}",
                        output.len()
                    ),
                ));
            }
            // SAFETY: one thread owns each (sample, reduction-lane) partial. The
            // workspace is reused on the same stream before folding this expert.
            unsafe {
                self.moe.moe_gate_dot_partial(
                    &self.stream,
                    launch::elementwise(batch * 256, config),
                    &output_gradient.buf,
                    &output.buf,
                    &mut partial,
                    per_image,
                    batch,
                )
            }
            .map_err(launch_err)?;
            // SAFETY: each expert launch writes a disjoint column of the dots matrix.
            unsafe {
                self.moe.moe_gate_dot_reduce_column(
                    &self.stream,
                    launch::elementwise(batch, config),
                    &partial,
                    &mut dots,
                    experts,
                    expert,
                )
            }
            .map_err(launch_err)?;
        }
        let mut dlogit = self.alloc_uninit(gate_count)?;
        // SAFETY: one thread owns one complete batch row of router derivatives.
        unsafe {
            self.moe.moe_router_gate_backward(
                &self.stream,
                launch::elementwise(batch, config),
                &gates.buf,
                &dots,
                &mut dlogit,
                batch,
                experts,
                load_balance,
                temperature,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dlogit, &[batch, experts]))
    }

}
