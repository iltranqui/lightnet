//! Optimizer update backend dispatch.

use lightnet_cuda::{AdamParams, AdamUpdateItem, ElementwiseLaunchConfig, SgdUpdateItem};

use crate::{launch, launch_err, same_len, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn sgd_update_backend_f32(
        &self,
        weights: &OxideTensorF32,
        updates: &OxideTensorF32,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        same_len("sgd_update_f32", weights, updates)?;
        let batch = batch.max(1) as f32;
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.optim.sgd_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &mut new_weights,
                &mut new_updates,
                lr / batch,
                momentum,
                decay * batch,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, weights.shape()),
            OxideTensorF32::new(new_updates, updates.shape()),
        ))
    }

    pub(crate) fn adam_update_backend_f32(
        &self,
        weights: &OxideTensorF32,
        updates: &OxideTensorF32,
        m: &OxideTensorF32,
        v: &OxideTensorF32,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let m_f = self.f32v(m)?;
        let m = &*m_f;
        let v_f = self.f32v(v)?;
        let v = &*v_f;
        let op = "adam_update_f32";
        same_len(op, weights, updates)?;
        same_len(op, weights, m)?;
        same_len(op, weights, v)?;
        let (coupled, decoupled, bc1, bc2) = params.kernel_scalars();
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_m = self.alloc_uninit(weights.len())?;
        let mut new_v = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element writes its own slot of each output.
        unsafe {
            self.optim.adam_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &m.buf,
                &v.buf,
                &mut new_weights,
                &mut new_m,
                &mut new_v,
                params.lr,
                params.beta1,
                params.beta2,
                params.eps,
                coupled,
                decoupled,
                bc1,
                bc2,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, weights.shape()),
            OxideTensorF32::new(new_m, weights.shape()),
            OxideTensorF32::new(new_v, weights.shape()),
        ))
    }

    pub(crate) fn sanitized_sgd_update_backend_f32(
        &self,
        weights: &OxideTensorF32,
        updates: &OxideTensorF32,
        shape: &[usize],
        fix_weights: bool,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let op = "sanitized_sgd_update_f32";
        same_len(op, weights, updates)?;
        if shape.iter().product::<usize>() != weights.len() {
            return Err(OxideError::Shape(format!(
                "{op}: shape {shape:?} does not match {} elements",
                weights.len()
            )));
        }
        let batch = batch.max(1) as f32;
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element; every output slot is written.
        unsafe {
            self.optim.sanitized_sgd_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &mut new_weights,
                &mut new_updates,
                lr / batch,
                momentum,
                decay * batch,
                fix_weights as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, shape),
            OxideTensorF32::new(new_updates, shape),
        ))
    }

    pub(crate) fn sanitized_adam_update_backend_f32(
        &self,
        weights: &OxideTensorF32,
        updates: &OxideTensorF32,
        m: &OxideTensorF32,
        v: &OxideTensorF32,
        shape: &[usize],
        fix_weights: bool,
        params: &AdamParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        let updates_f = self.f32v(updates)?;
        let updates = &*updates_f;
        let m_f = self.f32v(m)?;
        let m = &*m_f;
        let v_f = self.f32v(v)?;
        let v = &*v_f;
        let op = "sanitized_adam_update_f32";
        same_len(op, weights, updates)?;
        same_len(op, weights, m)?;
        same_len(op, weights, v)?;
        if shape.iter().product::<usize>() != weights.len() {
            return Err(OxideError::Shape(format!(
                "{op}: shape {shape:?} does not match {} elements",
                weights.len()
            )));
        }
        let (coupled, decoupled, bc1, bc2) = params.kernel_scalars();
        let mut new_weights = self.alloc_uninit(weights.len())?;
        let mut new_m = self.alloc_uninit(weights.len())?;
        let mut new_v = self.alloc_uninit(weights.len())?;
        let mut new_updates = self.alloc_uninit(weights.len())?;
        // SAFETY: one thread per element; every output slot is written.
        unsafe {
            self.optim.sanitized_adam_update(
                &self.stream,
                launch::elementwise(weights.len(), config),
                &weights.buf,
                &updates.buf,
                &m.buf,
                &v.buf,
                &mut new_weights,
                &mut new_m,
                &mut new_v,
                &mut new_updates,
                params.lr,
                params.beta1,
                params.beta2,
                params.eps,
                coupled,
                decoupled,
                bc1,
                bc2,
                fix_weights as u32,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(new_weights, shape),
            OxideTensorF32::new(new_m, shape),
            OxideTensorF32::new(new_v, shape),
            OxideTensorF32::new(new_updates, shape),
        ))
    }

    pub(crate) fn sanitized_sgd_update_multi_backend_f32(
        &self,
        items: &[SgdUpdateItem<'_, OxideTensorF32>],
        lr: f32,
        momentum: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(OxideTensorF32, OxideTensorF32)>, OxideError> {
        let op = "sanitized_sgd_update_multi_f32";
        if items.is_empty() || items.len() > 3 {
            return Err(OxideError::Shape(format!(
                "{op}: 1..=3 tensors per launch, got {}",
                items.len()
            )));
        }
        // Resolve every operand to f32 (AMP tensors are widened by `f32v`).
        let mut w_f = Vec::new();
        let mut u_f = Vec::new();
        let mut g_f = Vec::new();
        for it in items {
            let w = self.f32v(it.weights)?;
            let u = self.f32v(it.updates)?;
            same_len(op, &*w, &*u)?;
            if it.shape.iter().product::<usize>() != w.len() {
                return Err(OxideError::Shape(format!(
                    "{op}: shape {:?} does not match {} elements",
                    it.shape,
                    w.len()
                )));
            }
            let g = match it.grad {
                Some(g) => {
                    let g = self.f32v(g)?;
                    same_len(op, &*w, &*g)?;
                    Some(g)
                }
                None => None,
            };
            w_f.push(w);
            u_f.push(u);
            g_f.push(g);
        }
        let batch = batch.max(1) as f32;
        // Unused segments reuse segment 0's inputs (never read, n == 0) and a
        // one-element dummy output.
        let seg = |k: usize| if k < items.len() { k } else { 0 };
        let n = |k: usize| if k < items.len() { w_f[k].len() } else { 0 };
        let mut w_out = Vec::new();
        let mut u_out = Vec::new();
        for k in 0..3 {
            let len = if k < items.len() { w_f[k].len() } else { 1 };
            w_out.push(self.alloc_uninit(len)?);
            u_out.push(self.alloc_uninit(len)?);
        }
        let gbuf = |k: usize| match &g_f[seg(k)] {
            Some(g) => &g.buf,
            None => &u_f[seg(k)].buf,
        };
        let hg = |k: usize| (k < items.len() && g_f[k].is_some()) as u32;
        let decay = |k: usize| {
            if k < items.len() {
                items[k].decay * batch
            } else {
                0.0
            }
        };
        let total = n(0) + n(1) + n(2);
        let [wo0, wo1, wo2] = &mut *w_out else {
            unreachable!()
        };
        let [uo0, uo1, uo2] = &mut *u_out else {
            unreachable!()
        };
        // SAFETY: one thread per element of the concatenation; every output
        // slot of every used segment is written.
        unsafe {
            self.optim.sanitized_sgd_update3(
                &self.stream,
                launch::elementwise(total, config),
                &w_f[0].buf,
                &u_f[0].buf,
                gbuf(0),
                wo0,
                uo0,
                &w_f[seg(1)].buf,
                &u_f[seg(1)].buf,
                gbuf(1),
                wo1,
                uo1,
                &w_f[seg(2)].buf,
                &u_f[seg(2)].buf,
                gbuf(2),
                wo2,
                uo2,
                n(0),
                n(1),
                n(2),
                hg(0),
                hg(1),
                hg(2),
                items[0].fix_weights as u32,
                lr / batch,
                momentum,
                decay(0),
                decay(1),
                decay(2),
            )
        }
        .map_err(launch_err)?;
        let mut out = Vec::with_capacity(items.len());
        let mut wi = w_out.into_iter();
        let mut ui = u_out.into_iter();
        for it in items {
            out.push((
                OxideTensorF32::new(wi.next().expect("w_out"), it.shape),
                OxideTensorF32::new(ui.next().expect("u_out"), it.shape),
            ));
        }
        Ok(out)
    }

    pub(crate) fn sanitized_adam_update_multi_backend_f32(
        &self,
        items: &[AdamUpdateItem<'_, OxideTensorF32>],
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<(OxideTensorF32, OxideTensorF32, OxideTensorF32, OxideTensorF32)>, OxideError> {
        let op = "sanitized_adam_update_multi_f32";
        if items.is_empty() || items.len() > 3 {
            return Err(OxideError::Shape(format!(
                "{op}: 1..=3 tensors per launch, got {}",
                items.len()
            )));
        }
        let mut w_f = Vec::new();
        let mut u_f = Vec::new();
        let mut g_f = Vec::new();
        let mut m_f = Vec::new();
        let mut v_f = Vec::new();
        for it in items {
            let w = self.f32v(it.weights)?;
            let u = self.f32v(it.updates)?;
            let m = self.f32v(it.m)?;
            let v = self.f32v(it.v)?;
            same_len(op, &*w, &*u)?;
            same_len(op, &*w, &*m)?;
            same_len(op, &*w, &*v)?;
            if it.shape.iter().product::<usize>() != w.len() {
                return Err(OxideError::Shape(format!(
                    "{op}: shape {:?} does not match {} elements",
                    it.shape,
                    w.len()
                )));
            }
            let g = match it.grad {
                Some(g) => {
                    let g = self.f32v(g)?;
                    same_len(op, &*w, &*g)?;
                    Some(g)
                }
                None => None,
            };
            w_f.push(w);
            u_f.push(u);
            g_f.push(g);
            m_f.push(m);
            v_f.push(v);
        }
        let seg = |k: usize| if k < items.len() { k } else { 0 };
        let n = |k: usize| if k < items.len() { w_f[k].len() } else { 0 };
        let mut w_out = Vec::new();
        let mut m_out = Vec::new();
        let mut v_out = Vec::new();
        let mut u_out = Vec::new();
        for k in 0..3 {
            let len = if k < items.len() { w_f[k].len() } else { 1 };
            w_out.push(self.alloc_uninit(len)?);
            m_out.push(self.alloc_uninit(len)?);
            v_out.push(self.alloc_uninit(len)?);
            u_out.push(self.alloc_uninit(len)?);
        }
        let gbuf = |k: usize| match &g_f[seg(k)] {
            Some(g) => &g.buf,
            None => &u_f[seg(k)].buf,
        };
        let hg = |k: usize| (k < items.len() && g_f[k].is_some()) as u32;
        let scalars = |k: usize| {
            if k < items.len() {
                let (c, d, _, _) = items[k].params.kernel_scalars();
                (c, d)
            } else {
                (0.0, 0.0)
            }
        };
        let p0 = items[0].params;
        let (_, _, bc1, bc2) = p0.kernel_scalars();
        let total = n(0) + n(1) + n(2);
        let [wo0, wo1, wo2] = &mut *w_out else {
            unreachable!()
        };
        let [mo0, mo1, mo2] = &mut *m_out else {
            unreachable!()
        };
        let [vo0, vo1, vo2] = &mut *v_out else {
            unreachable!()
        };
        let [uo0, uo1, uo2] = &mut *u_out else {
            unreachable!()
        };
        // SAFETY: as in `sanitized_sgd_update_multi_f32`.
        unsafe {
            self.optim.sanitized_adam_update3(
                &self.stream,
                launch::elementwise(total, config),
                &w_f[0].buf,
                &u_f[0].buf,
                gbuf(0),
                &m_f[0].buf,
                &v_f[0].buf,
                wo0,
                mo0,
                vo0,
                uo0,
                &w_f[seg(1)].buf,
                &u_f[seg(1)].buf,
                gbuf(1),
                &m_f[seg(1)].buf,
                &v_f[seg(1)].buf,
                wo1,
                mo1,
                vo1,
                uo1,
                &w_f[seg(2)].buf,
                &u_f[seg(2)].buf,
                gbuf(2),
                &m_f[seg(2)].buf,
                &v_f[seg(2)].buf,
                wo2,
                mo2,
                vo2,
                uo2,
                n(0),
                n(1),
                n(2),
                hg(0),
                hg(1),
                hg(2),
                items[0].fix_weights as u32,
                p0.lr,
                p0.beta1,
                p0.beta2,
                p0.eps,
                bc1,
                bc2,
                scalars(0).0,
                scalars(0).1,
                scalars(1).0,
                scalars(1).1,
                scalars(2).0,
                scalars(2).1,
            )
        }
        .map_err(launch_err)?;
        let mut out = Vec::with_capacity(items.len());
        let mut wi = w_out.into_iter();
        let mut mi = m_out.into_iter();
        let mut vi = v_out.into_iter();
        let mut ui = u_out.into_iter();
        for it in items {
            out.push((
                OxideTensorF32::new(wi.next().expect("w_out"), it.shape),
                OxideTensorF32::new(mi.next().expect("m_out"), it.shape),
                OxideTensorF32::new(vi.next().expect("v_out"), it.shape),
                OxideTensorF32::new(ui.next().expect("u_out"), it.shape),
            ));
        }
        Ok(out)
    }
}
