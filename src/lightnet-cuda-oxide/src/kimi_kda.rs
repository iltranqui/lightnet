//! Checked device launch wrappers for Kimi's scan and deterministic router.
use lightnet_cuda::{ElementwiseLaunchConfig, kimi_kda::{KdaForward, KdaGeometry, KdaGrad}};
use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

fn lengths(actual: &[usize], expected: &[usize]) -> Result<(), OxideError> {
    if actual != expected { return Err(OxideError::Shape("Kimi KDA tensor length mismatch".into())); }
    Ok(())
}

impl OxideBackend {
    pub(crate) fn kimi_scan_native(
        &self, q: &OxideTensorF32, k: &OxideTensorF32, v: &OxideTensorF32,
        beta: &OxideTensorF32, decay: &OxideTensorF32, g: KdaGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<KdaForward<OxideTensorF32>, OxideError> {
        g.validate().map_err(OxideError::Shape)?;
        let rows = g.batch * g.tokens;
        let (q, k, v, beta, decay) = (self.f32v(q)?, self.f32v(k)?, self.f32v(v)?, self.f32v(beta)?, self.f32v(decay)?);
        lengths(&[q.len(), k.len(), v.len(), beta.len(), decay.len()],
            &[rows * g.rank, rows * g.rank, rows * g.channels, rows * g.rank, rows * g.rank])?;
        let mut keys = self.alloc_uninit(rows * g.rank)?;
        let mut inverse = self.alloc_uninit(rows)?;
        let mut norm = self.alloc_uninit(rows)?;
        let mut states = self.alloc_uninit(rows * g.rank * g.channels)?;
        let mut out = self.alloc_uninit(rows * g.channels)?;
        // SAFETY: checked sizes; batch threads own disjoint time/state slabs.
        unsafe { self.kimi_kda.scan(&self.stream, launch::elementwise(g.batch, config),
            &q.buf, &k.buf, &v.buf, &beta.buf, &decay.buf, &mut keys, &mut inverse,
            &mut norm, &mut states, &mut out, g.batch, g.tokens, g.channels, g.rank) }.map_err(launch_err)?;
        Ok(KdaForward { output: OxideTensorF32::new(out, &[rows, g.channels]),
            keys: OxideTensorF32::new(keys, &[rows, g.rank]), inverse: OxideTensorF32::new(inverse, &[rows]),
            norm_squared: OxideTensorF32::new(norm, &[rows]),
            states: OxideTensorF32::new(states, &[rows, g.rank, g.channels]) })
    }

    pub(crate) fn kimi_scan_backward_native(
        &self, q: &OxideTensorF32, v: &OxideTensorF32, beta: &OxideTensorF32, decay: &OxideTensorF32,
        f: &KdaForward<OxideTensorF32>, dy: &OxideTensorF32, g: KdaGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<KdaGrad<OxideTensorF32>, OxideError> {
        g.validate().map_err(OxideError::Shape)?;
        let rows = g.batch * g.tokens;
        let (q, k, v, beta, decay) = (self.f32v(q)?, self.f32v(&f.keys)?, self.f32v(v)?, self.f32v(beta)?, self.f32v(decay)?);
        let (inv, norm, states, dy) = (self.f32v(&f.inverse)?, self.f32v(&f.norm_squared)?, self.f32v(&f.states)?, self.f32v(dy)?);
        lengths(&[q.len(), k.len(), v.len(), beta.len(), decay.len(), inv.len(), norm.len(), states.len(), dy.len()],
            &[rows * g.rank, rows * g.rank, rows * g.channels, rows * g.rank, rows * g.rank,
                rows, rows, rows * g.rank * g.channels, rows * g.channels])?;
        // The public primitive follows the CPU seam contract as well as runtime callers.
        if norm.buf.to_host_vec(&self.stream)?.contains(&1e-12) {
            return Err(OxideError::Shape("Kimi KDA key-normalization derivative is undefined at the floor boundary".into()));
        }
        let mut dq = self.alloc_uninit(rows * g.rank)?;
        let mut dk = self.alloc_uninit(rows * g.rank)?;
        let mut dv = self.alloc_uninit(rows * g.channels)?;
        let mut db = self.alloc_uninit(rows * g.rank)?;
        let mut dd = self.alloc_uninit(rows * g.rank)?;
        let mut carry = self.alloc_uninit(g.batch * g.rank * g.channels)?;
        // SAFETY: validated rank <= 64 and lengths; carry is per batch.
        unsafe { self.kimi_kda.backward(&self.stream, launch::elementwise(g.batch, config),
            &q.buf, &k.buf, &v.buf, &beta.buf, &decay.buf, &inv.buf, &norm.buf, &states.buf,
            &dy.buf, &mut dq, &mut dk, &mut dv, &mut db, &mut dd, &mut carry,
            g.batch, g.tokens, g.channels, g.rank) }.map_err(launch_err)?;
        Ok(KdaGrad { q: OxideTensorF32::new(dq, &[rows, g.rank]),
            k: OxideTensorF32::new(dk, &[rows, g.rank]), v: OxideTensorF32::new(dv, &[rows, g.channels]),
            beta: OxideTensorF32::new(db, &[rows, g.rank]), decay: OxideTensorF32::new(dd, &[rows, g.rank]) })
    }

    pub(crate) fn kimi_router_native(&self, logits: &OxideTensorF32, rows: usize,
        experts: usize, top_k: usize, temperature: f32, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let n = rows.checked_mul(experts).ok_or_else(|| OxideError::Shape("Kimi router overflow".into()))?;
        if rows == 0 || experts == 0 || experts > 64 || top_k > experts
            || !temperature.is_finite() || temperature <= 0.0 {
            return Err(OxideError::Shape("Kimi router invalid geometry".into()));
        }
        let logits = self.f32v(logits)?;
        lengths(&[logits.len()], &[n])?;
        let mut out = self.alloc_uninit(n)?;
        // SAFETY: one thread owns each row, experts <= local array length.
        unsafe { self.kimi_kda.router(&self.stream, launch::elementwise(rows, config),
            &logits.buf, &mut out, rows, experts, top_k, temperature) }.map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[rows, experts]))
    }
}
