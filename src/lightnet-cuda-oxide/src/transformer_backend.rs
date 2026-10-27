//! Transformer primitive backend dispatch.

use cuda_core::simt::LaunchConfig;
use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{kernels, launch, launch_err, GemmArgs, OxideBackend, OxideError, OxideTensorF32};

/// Reject a row count / width the LayerNorm kernels cannot serve before
/// launching, instead of reading out of bounds.
fn check_layernorm_shape(
    op: &str,
    shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), OxideError> {
    let total: usize = shape.iter().product();
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(total) {
        return Err(OxideError::Shape(format!(
            "{op}: rows*cols must equal the tensor length ({rows}*{cols} vs {total})"
        )));
    }
    if cols > kernels::activations::layernorm::LAYERNORM_MAX_COLS {
        return Err(OxideError::Shape(format!(
            "{op}: cols={cols} exceeds the kernel limit {}",
            kernels::activations::layernorm::LAYERNORM_MAX_COLS
        )));
    }
    Ok(())
}

impl OxideBackend {
    pub(crate) fn linear_f32_backend(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        bias: Option<&OxideTensorF32>,
        shape: &[usize],
        rows: usize,
        in_features: usize,
        out_features: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let _ = config;
        if rows == 0 || in_features == 0 || out_features == 0 {
            return Err(OxideError::Shape(
                "linear_f32: rows, in_features and out_features must be > 0".to_owned(),
            ));
        }
        if shape.iter().product::<usize>() != rows * out_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: shape {shape:?} must hold rows*out_features = {}",
                rows * out_features
            )));
        }
        let x_f = self.f32v(x)?.into_owned();
        let w_f = self.f32v(weights)?.into_owned();
        if x_f.len() < rows * in_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: activation has {} entries, expected {}",
                x_f.len(),
                rows * in_features
            )));
        }
        if w_f.len() < out_features * in_features {
            return Err(OxideError::Shape(format!(
                "linear_f32: weights have {} entries, expected {}",
                w_f.len(),
                out_features * in_features
            )));
        }
        // Fold the bias into a beta-1 accumulator copy of the output, which the
        // GEMM already accumulates into; that avoids a second epilogue pass.
        let mut acc = self.alloc_uninit(rows * out_features)?;
        if let Some(bias) = bias {
            let b = self.f32v(bias)?.into_owned();
            if b.len() < out_features {
                return Err(OxideError::Shape(format!(
                    "linear_f32: bias has {} entries, expected {out_features}",
                    b.len()
                )));
            }
            let host = b.buf.to_host_vec(&self.stream)?;
            let mut values = vec![0.0f32; rows * out_features];
            for r in 0..rows {
                values[r * out_features..(r + 1) * out_features]
                    .copy_from_slice(&host[..out_features]);
            }
            acc.copy_from_host(&self.stream, &values)?;
        } else {
            acc.copy_from_host(&self.stream, &vec![0.0f32; rows * out_features])?;
        }
        // `gemm` takes a `&mut DeviceBuffer`, so drive it on the pooled buffer
        // directly and keep the tensor only for the returned shape.
        let mut acc_raw = acc;
        self.gemm(
            &x_f.buf,
            &w_f.buf,
            &mut acc_raw,
            &GemmArgs {
                m: rows,
                n: out_features,
                k: in_features,
                lda: in_features,
                // `w` is [out_features, in_features], i.e. B^T of the
                // [in_features, out_features] matrix the GEMM wants.
                ldb: in_features,
                ldc: out_features,
                trans_a: false,
                trans_b: true,
                alpha: 1.0,
                beta: 1.0,
                ..GemmArgs::default()
            },
        )?;
        Ok(OxideTensorF32::new(acc_raw, shape))
    }

    /// Fused residual + bias epilogue, one thread per output element:
    /// `out[i] = residual[i] + y[i] + bias[i % cols]`.
    pub(crate) fn residual_bias_f32_backend(
        &self,
        residual: &OxideTensorF32,
        y: &OxideTensorF32,
        bias: Option<&OxideTensorF32>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let len = rows.checked_mul(cols).ok_or_else(|| {
            OxideError::Shape("residual_bias_f32: rows * cols overflowed".to_owned())
        })?;
        if rows == 0 || cols == 0 || len != shape.iter().product::<usize>() {
            return Err(OxideError::Shape(format!(
                "residual_bias_f32: rows*cols must equal the tensor length ({rows}*{cols} vs {:?})",
                shape
            )));
        }
        let r = self.f32v(residual)?.into_owned();
        let yv = self.f32v(y)?.into_owned();
        // A missing bias is a one-element placeholder the kernel never indexes.
        let bias_tensor = match bias {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1usize]),
        };
        let has_bias = u32::from(bias.is_some());
        let mut out = self.alloc_uninit(len)?;
        // SAFETY: one thread per element, and the kernel only reads `bias` when
        // `has_bias` is set (then it has `cols` entries).
        unsafe {
            self.residual_epilogue.residual_bias(
                &self.stream,
                launch::elementwise(len, config),
                &r.buf,
                &yv.buf,
                &bias_tensor.buf,
                &mut out,
                cols,
                has_bias,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    pub(crate) fn layernorm_forward_f32_backend(
        &self,
        x: &OxideTensorF32,
        gamma: Option<&OxideTensorF32>,
        beta: Option<&OxideTensorF32>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let _ = config;
        check_layernorm_shape("layernorm_forward_f32", shape, rows, cols)?;
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        // A missing gamma / beta becomes a one-element placeholder tensor the
        // kernel never reads (the flag gates the access), so there is no `cols`
        // allocation and no extra launch.  Both arms are `OxideTensorF32` so the
        // borrowed buffer below coerces exactly like every other operand.
        let gamma_tensor = match gamma {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let beta_tensor = match beta {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let has_gamma = u32::from(gamma.is_some());
        let has_beta = u32::from(beta.is_some());
        let mut out = self.alloc_uninit(rows * cols)?;
        // SAFETY: one thread per row, each writing only its own `cols` outputs.
        unsafe {
            self.layernorm.layernorm_forward(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &x.buf,
                &gamma_tensor.buf,
                &beta_tensor.buf,
                &mut out,
                rows,
                cols,
                eps,
                has_gamma,
                has_beta,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, shape))
    }

    /// Row-wise LayerNorm input gradient, same one-thread-per-row reduction.
    pub(crate) fn layernorm_backward_f32_backend(
        &self,
        output_grad: &OxideTensorF32,
        x: &OxideTensorF32,
        gamma: Option<&OxideTensorF32>,
        shape: &[usize],
        rows: usize,
        cols: usize,
        eps: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let _ = config;
        check_layernorm_shape("layernorm_backward_f32", shape, rows, cols)?;
        let dy_f = self.f32v(output_grad)?;
        let x_f = self.f32v(x)?;
        let dy = &*dy_f;
        let x = &*x_f;
        let gamma_tensor = match gamma {
            Some(t) => self.f32v(t)?.into_owned(),
            None => OxideTensorF32::new(self.alloc_uninit(1)?, &[1]),
        };
        let has_gamma = u32::from(gamma.is_some());
        let mut dx = self.alloc_uninit(rows * cols)?;
        // SAFETY: one thread per row, each writing only its own `cols` outputs.
        unsafe {
            self.layernorm.layernorm_backward(
                &self.stream,
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &dy.buf,
                &x.buf,
                &gamma_tensor.buf,
                &mut dx,
                rows,
                cols,
                eps,
                has_gamma,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, shape))
    }

}
