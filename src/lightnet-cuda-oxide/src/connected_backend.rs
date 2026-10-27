//! Native connected-layer backend operations.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::{launch, launch_err, shape_err, GemmArgs, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn connected_backend_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if self.amp {
            return self.connected_forward_amp(x,weights,biases,batch,inputs,outputs,config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let biases_f = self.f32v(biases)?;
        let biases = &*biases_f;
        let op = "connected_f32";
        self.connected_layout(op, x.len(), weights.len(), batch, inputs, outputs)?;
        if biases.len() != outputs {
            return Err(shape_err(
                op,
                format!("expected {outputs} biases, got {}", biases.len()),
            ));
        }
        let mut raw = self.alloc_uninit(batch * outputs)?;
        // out[b, o] = x[b, :] . w[o, :]  (NT GEMM; beta = 0 writes every element)
        self.gemm(
            &x.buf,
            &weights.buf,
            &mut raw,
            &GemmArgs {
                m: batch,
                n: outputs,
                k: inputs,
                lda: inputs,
                ldb: inputs,
                ldc: outputs,
                trans_b: true,
                ..GemmArgs::default()
            },
        )?;
        let mut out = self.alloc_uninit(batch * outputs)?;
        // SAFETY: one thread per output element; channel_stride 1 = per-column bias.
        unsafe {
            self.bias_scale.bias_add(
                &self.stream,
                launch::elementwise(out.len(), config),
                &raw,
                &biases.buf,
                &mut out,
                outputs,
                1,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[batch, outputs]))
    }

    pub(crate) fn connected_backend_backward_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        delta: &OxideTensorF32,
        batch: usize,
        inputs: usize,
        outputs: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        if self.amp {
            return self.connected_backward_amp(x,weights,delta,batch,inputs,outputs);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        let op = "connected_backward_f32";
        self.connected_layout(op, x.len(), weights.len(), batch, inputs, outputs)?;
        if delta.len() != batch * outputs {
            return Err(shape_err(
                op,
                format!(
                    "expected {} delta elements, got {}",
                    batch * outputs,
                    delta.len()
                ),
            ));
        }
        // dx = delta . W  (NN GEMM)
        let mut dx = self.alloc_uninit(batch * inputs)?;
        self.gemm(
            &delta.buf,
            &weights.buf,
            &mut dx,
            &GemmArgs {
                m: batch,
                n: inputs,
                k: outputs,
                lda: outputs,
                ldb: inputs,
                ldc: inputs,
                ..GemmArgs::default()
            },
        )?;
        // dW = delta^T . x  (TN GEMM)
        let mut dw = self.alloc_uninit(outputs * inputs)?;
        self.gemm(
            &delta.buf,
            &x.buf,
            &mut dw,
            &GemmArgs {
                m: outputs,
                n: inputs,
                k: batch,
                lda: outputs,
                ldb: inputs,
                ldc: inputs,
                trans_a: true,
                ..GemmArgs::default()
            },
        )?;
        let db = self.bias_backward_f32(delta, outputs, 1, config)?;
        Ok((
            OxideTensorF32::new(dx, &[batch, inputs]),
            OxideTensorF32::new(dw, &[outputs, inputs]),
            db,
        ))
    }
}
