//! Checked launch wrappers for native MambaVision primitives.
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::mambavision::{MambaScanForward, MambaScanGeometry, MambaScanGrad};
use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

fn shape(message: &str) -> OxideError { OxideError::Shape(format!("MambaVision: {message}")) }
fn product(dims: &[usize]) -> Result<usize, OxideError> {
    if dims.contains(&0) { return Err(shape("dimensions must be positive")); }
    dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| shape("geometry overflow"))
}
fn lengths(actual: &[usize], expected: &[usize]) -> Result<(), OxideError> {
    if actual != expected { return Err(shape("tensor length mismatch")); }
    Ok(())
}
fn depth_geometry(b: usize, t: usize, d: usize, branch: usize, k: usize) -> Result<usize, OxideError> {
    if branch > 1 || k == 0 || k % 2 == 0 { return Err(shape("invalid depthwise branch/kernel")); }
    product(&[d, k])?;
    product(&[b, t, d, 2])?;
    product(&[b, d, t])
}
fn msa_geometry(b: usize, t: usize, c: usize, h: usize) -> Result<(usize, usize), OxideError> {
    if h == 0 || c % h != 0 { return Err(shape("MSA channels must be divisible by positive heads")); }
    Ok((product(&[b, t, c])?, product(&[b, h, t, t])?))
}

impl OxideBackend {
    pub(crate) fn mambavision_depthwise_native(
        &self, input: &OxideTensorF32, weights: &OxideTensorF32, bias: &OxideTensorF32,
        b: usize, t: usize, d: usize, branch: usize, k: usize, causal: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let n = depth_geometry(b, t, d, branch, k)?;
        let x = self.f32v(input)?;
        let w = self.f32v(weights)?;
        let bias = self.f32v(bias)?;
        lengths(&[x.len(), w.len(), bias.len()], &[2 * n, d * k, d])?;
        let mut pre = self.alloc_uninit(n)?;
        let mut out = self.alloc_uninit(n)?;
        // SAFETY: validated BDT outputs; each thread writes one element.
        unsafe { self.mambavision.mamba_depthwise(&self.stream, launch::elementwise(n, config),
            &x.buf, &w.buf, &bias.buf, &mut pre, &mut out, t, d, branch, k,
            if causal { k - 1 } else { k / 2 }) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(pre, &[b, d, t]), OxideTensorF32::new(out, &[b, d, t])))
    }

    pub(crate) fn mambavision_depthwise_backward_native(
        &self, input: &OxideTensorF32, weights: &OxideTensorF32, pre: &OxideTensorF32,
        grad: &OxideTensorF32, b: usize, t: usize, d: usize, branch: usize,
        k: usize, causal: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let n = depth_geometry(b, t, d, branch, k)?;
        let x = self.f32v(input)?;
        let w = self.f32v(weights)?;
        let pre = self.f32v(pre)?;
        let grad = self.f32v(grad)?;
        lengths(&[x.len(), w.len(), pre.len(), grad.len()], &[2 * n, d * k, n, n])?;
        let mut dx = self.alloc_uninit(2 * n)?;
        let mut dw = self.alloc_uninit(d * k)?;
        let mut db = self.alloc_uninit(d)?;
        let pad = if causal { k - 1 } else { k / 2 };
        // SAFETY: input gradients gather valid taps; channel reductions own parameter rows.
        unsafe { self.mambavision.mamba_depthwise_dx(&self.stream, launch::elementwise(2 * n, config),
            &w.buf, &pre.buf, &grad.buf, &mut dx, t, d, branch, k, pad) }.map_err(launch_err)?;
        unsafe { self.mambavision.mamba_depthwise_params(&self.stream, launch::elementwise(d, config),
            &x.buf, &pre.buf, &grad.buf, &mut dw, &mut db, b, t, d, branch, k, pad) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(dx, &[b * t, 2 * d]),
            OxideTensorF32::new(dw, &[d, k]), OxideTensorF32::new(db, &[d])))
    }

    pub(crate) fn mambavision_scan_native(
        &self, x: &OxideTensorF32, projection: &OxideTensorF32, dt_weights: &OxideTensorF32,
        dt_bias: &OxideTensorF32, a_log: &OxideTensorF32, d_skip: &OxideTensorF32,
        state_in: &OxideTensorF32, g: MambaScanGeometry, config: ElementwiseLaunchConfig,
    ) -> Result<MambaScanForward<OxideTensorF32>, OxideError> {
        g.validate().map_err(OxideError::Shape)?;
        let (b, t, d, s, r) = (g.batch, g.tokens, g.channels, g.state, g.rank);
        let n = b * d * t;
        let x = self.f32v(x)?;
        let p = self.f32v(projection)?;
        let w = self.f32v(dt_weights)?;
        let bias = self.f32v(dt_bias)?;
        let a = self.f32v(a_log)?;
        let skip = self.f32v(d_skip)?;
        let initial = self.f32v(state_in)?;
        lengths(&[x.len(), p.len(), w.len(), bias.len(), a.len(), skip.len(), initial.len()],
            &[n, b * t * g.width(), d * r, d, d * s, d, b * d * s])?;
        let mut out = self.alloc_uninit(n)?;
        let mut pre = self.alloc_uninit(n)?;
        let mut states = self.alloc_uninit(n * s)?;
        let mut final_state = self.alloc_uninit(b * d * s)?;
        let mut boundary = self.alloc_uninit(b * d)?;
        // SAFETY: each lane owns a full time/state slab; all buffers are fully written.
        unsafe { self.mambavision.mamba_scan(&self.stream, launch::elementwise(b * d, config),
            &x.buf, &p.buf, &w.buf, &bias.buf, &a.buf, &skip.buf, &initial.buf,
            &mut out, &mut pre, &mut states, &mut final_state, &mut boundary, b, t, d, s, r)
        }.map_err(launch_err)?;
        Ok(MambaScanForward {
            output: OxideTensorF32::new(out, &[b, d, t]),
            dt_pre: OxideTensorF32::new(pre, &[b, d, t]),
            states: OxideTensorF32::new(states, &[b, d, t, s]),
            state_out: OxideTensorF32::new(final_state, &[b, d, s]),
            boundary: OxideTensorF32::new(boundary, &[b, d]),
        })
    }

    pub(crate) fn mambavision_scan_backward_native(
        &self, x: &OxideTensorF32, projection: &OxideTensorF32, dt_weights: &OxideTensorF32,
        a_log: &OxideTensorF32, d_skip: &OxideTensorF32, state_in: &OxideTensorF32,
        dt_pre: &OxideTensorF32, states: &OxideTensorF32, grad: &OxideTensorF32,
        state_grad: &OxideTensorF32, g: MambaScanGeometry, config: ElementwiseLaunchConfig,
    ) -> Result<MambaScanGrad<OxideTensorF32>, OxideError> {
        g.validate().map_err(OxideError::Shape)?;
        let (b, t, d, s, r, w) = (g.batch, g.tokens, g.channels, g.state, g.rank, g.width());
        let n = b * d * t;
        let x = self.f32v(x)?;
        let p = self.f32v(projection)?;
        let dtw = self.f32v(dt_weights)?;
        let a = self.f32v(a_log)?;
        let skip = self.f32v(d_skip)?;
        let initial = self.f32v(state_in)?;
        let pre = self.f32v(dt_pre)?;
        let states = self.f32v(states)?;
        let grad = self.f32v(grad)?;
        let sg = self.f32v(state_grad)?;
        lengths(&[x.len(), p.len(), dtw.len(), a.len(), skip.len(), initial.len(),
            pre.len(), states.len(), grad.len(), sg.len()],
            &[n, b * t * w, d * r, d * s, d, b * d * s, n, n * s, n, b * d * s])?;
        let mut dx = self.alloc_uninit(n)?;
        let mut parts = self.alloc_uninit(n * w)?;
        let mut dp = self.alloc_uninit(b * t * w)?;
        let mut dw = self.alloc_uninit(d * r)?;
        let mut db = self.alloc_uninit(d)?;
        let mut da = self.alloc_uninit(d * s)?;
        let mut dd = self.alloc_uninit(d)?;
        let mut dh = self.alloc_uninit(b * d * s)?;
        // SAFETY: each channel owns its parameter rows and per-batch slabs;
        // the next launch reduces the private selective-projection gradients.
        unsafe { self.mambavision.mamba_scan_backward(&self.stream, launch::elementwise(d, config),
            &x.buf, &p.buf, &dtw.buf, &a.buf, &skip.buf, &initial.buf, &pre.buf,
            &states.buf, &grad.buf, &sg.buf, &mut dx, &mut parts, &mut dw, &mut db,
            &mut da, &mut dd, &mut dh, b, t, d, s, r) }.map_err(launch_err)?;
        unsafe { self.mambavision.mamba_scan_projection_reduce(&self.stream,
            launch::elementwise(b * t * w, config), &parts, &mut dp, t, d, w) }.map_err(launch_err)?;
        Ok(MambaScanGrad {
            input: OxideTensorF32::new(dx, &[b, d, t]),
            projection: OxideTensorF32::new(dp, &[b * t, w]),
            dt_weights: OxideTensorF32::new(dw, &[d, r]), dt_bias: OxideTensorF32::new(db, &[d]),
            a_log: OxideTensorF32::new(da, &[d, s]), d_skip: OxideTensorF32::new(dd, &[d]),
            state_in: OxideTensorF32::new(dh, &[b, d, s]),
        })
    }

    pub(crate) fn mambavision_msa_native(
        &self, input: &OxideTensorF32, b: usize, t: usize, c: usize, h: usize,
        block: usize, topk: usize, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let (n, scores) = msa_geometry(b, t, c, h)?;
        if block == 0 || topk == 0 { return Err(shape("MSA block/topk must be positive")); }
        let block = block.min(t);
        let blocks = t.div_ceil(block);
        let x = self.f32v(input)?;
        lengths(&[x.len()], &[n])?;
        let rows = b * h * t;
        let mut out = self.alloc_uninit(n)?;
        let mut probabilities = self.alloc_uninit(scores)?;
        let mut active = self.alloc_uninit(scores)?;
        let mut undefined = self.alloc_uninit(rows)?;
        let mut selected = self.alloc_uninit(rows * blocks)?;
        // SAFETY: one row/head thread owns all of its outputs and selection scratch.
        unsafe { self.mambavision.mamba_msa(&self.stream, launch::elementwise(rows, config),
            &x.buf, &mut out, &mut probabilities, &mut active, &mut undefined, &mut selected,
            b, t, c, h, block, topk.min(blocks)) }.map_err(launch_err)?;
        Ok((OxideTensorF32::new(out, &[b * t, c]),
            OxideTensorF32::new(probabilities, &[b, h, t, t]),
            OxideTensorF32::new(active, &[b, h, t, t]), OxideTensorF32::new(undefined, &[rows])))
    }

    pub(crate) fn mambavision_msa_backward_native(
        &self, input: &OxideTensorF32, probabilities: &OxideTensorF32, active: &OxideTensorF32,
        grad: &OxideTensorF32, b: usize, t: usize, c: usize, h: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let (n, scores) = msa_geometry(b, t, c, h)?;
        let x = self.f32v(input)?;
        let p = self.f32v(probabilities)?;
        let active = self.f32v(active)?;
        let grad = self.f32v(grad)?;
        lengths(&[x.len(), p.len(), active.len(), grad.len()], &[n, scores, scores, n])?;
        let mut ds = self.alloc_uninit(scores)?;
        let mut dx = self.alloc_uninit(n)?;
        // SAFETY: row-owned softmax adjoints followed by a gather per input element.
        unsafe { self.mambavision.mamba_msa_ds(&self.stream, launch::elementwise(b * h * t, config),
            &x.buf, &p.buf, &active.buf, &grad.buf, &mut ds, b, t, c, h) }.map_err(launch_err)?;
        unsafe { self.mambavision.mamba_msa_dx(&self.stream, launch::elementwise(n, config),
            &x.buf, &p.buf, &ds, &grad.buf, &mut dx, t, c, h) }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, &[b * t, c]))
    }
}
