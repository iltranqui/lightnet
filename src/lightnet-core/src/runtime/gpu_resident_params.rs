//! Flat serialized device parameter banks. Native SGD/Adam retain weights,
//! descending accumulators and moments between steps; host mirrors are only
//! materialized by explicit diagnostics, eager updates or checkpoint writing.
use super::*;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResidentMomentSlot {
    ChildWeights(usize),
    Weights,
    Biases,
}

impl ResidentMomentSlot {
    fn get<'a>(&self, state: &'a HostAdamLayerState) -> Option<&'a (Vec<f32>, Vec<f32>)> {
        match self {
            Self::ChildWeights(index) => state.children.get(*index)?.weights.as_ref(),
            Self::Weights => state.weights.as_ref(),
            Self::Biases => state.biases.as_ref(),
        }
    }

    fn set(&self, state: &mut HostAdamLayerState, pair: (Vec<f32>, Vec<f32>)) {
        match self {
            Self::ChildWeights(index) => host_adam_child(state, *index).weights = Some(pair),
            Self::Weights => state.weights = Some(pair),
            Self::Biases => state.biases = Some(pair),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidentParameterRange {
    pub range: Range<usize>,
    /// SGD and decoupled AdamW decay eligibility.
    pub decay: bool,
    /// Coupled Adam can decay biases even when SGD and AdamW do not.
    pub coupled_decay: bool,
    pub moment_slot: ResidentMomentSlot,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentParameterState<T> {
    pub weights: T,
    pub updates: T,
    pub moments: Option<(T, T)>,
    decay_mask: T,
    coupled_decay_mask: T,
    layout: Vec<ResidentParameterRange>,
    snapshot: Vec<f32>,
    pub generation: u64,
    pub dirty: bool,
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a,b)| a.to_bits() == b.to_bits())
}

fn host_values_match(params: &LayerParams, snapshot: &[f32]) -> Result<bool> {
    let split_matches = |biases: &[f32], weights: &[f32]| {
        snapshot.len() == biases.len() + weights.len()
            && same_bits(biases, &snapshot[..biases.len()])
            && same_bits(weights, &snapshot[biases.len()..])
    };
    match params {
        LayerParams::MambaVision(p) => Ok(same_bits(&p.values, snapshot)),
        LayerParams::KimiKda(p) => Ok(split_matches(&p.biases, &p.weights)),
        LayerParams::TuckerAttention(p) => Ok(split_matches(&p.biases, &p.weights)),
        _ => Err(Error::InvalidConfig("resident parameter host kind mismatch".into())),
    }
}

fn host_updates(buffer: &LayerTrainBuffers) -> Result<Vec<f32>> {
    match buffer {
        LayerTrainBuffers::MambaVision(t) => Ok(t.updates.clone()),
        LayerTrainBuffers::KimiKda(t) => Ok([t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()),
        LayerTrainBuffers::TuckerAttention(t) => Ok([t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()),
        _ => Err(Error::InvalidConfig("resident accumulator host kind mismatch".into())),
    }
}

fn set_host_updates(buffer: &mut LayerTrainBuffers, values: &[f32]) -> Result<()> {
    match buffer {
        LayerTrainBuffers::MambaVision(t) if t.updates.len() == values.len() => t.updates.copy_from_slice(values),
        LayerTrainBuffers::KimiKda(t) if t.bias_updates.len() + t.weight_updates.len() == values.len() => {
            let biases = t.bias_updates.len();
            t.bias_updates.copy_from_slice(&values[..biases]);
            t.weight_updates.copy_from_slice(&values[biases..]);
        }
        LayerTrainBuffers::TuckerAttention(t) if t.bias_updates.len() + t.weight_updates.len() == values.len() => {
            let biases = t.bias_updates.len();
            t.bias_updates.copy_from_slice(&values[..biases]);
            t.weight_updates.copy_from_slice(&values[biases..]);
        }
        _ => return Err(Error::InvalidConfig("resident host accumulator mirror mismatch".into())),
    }
    Ok(())
}

fn set_host_values(params: &mut LayerParams, values: &[f32]) -> Result<()> {
    match params {
        LayerParams::MambaVision(p) if p.values.len() == values.len() => p.values.copy_from_slice(values),
        LayerParams::KimiKda(p) if p.parameter_count() == values.len() => {
            let biases = p.biases.len();
            p.biases.copy_from_slice(&values[..biases]);
            p.weights.copy_from_slice(&values[biases..]);
        }
        LayerParams::TuckerAttention(p) if p.parameter_count() == values.len() => {
            let biases = p.biases.len();
            p.biases.copy_from_slice(&values[..biases]);
            p.weights.copy_from_slice(&values[biases..]);
        }
        _ => return Err(Error::InvalidConfig("resident parameter host length/kind mismatch".into())),
    }
    Ok(())
}

fn validate_layout(layout: &[ResidentParameterRange], total: usize) -> Result<()> {
    let mut end = 0;
    for block in layout {
        if block.range.start != end || block.range.end <= end || block.range.end > total {
            return Err(Error::InvalidConfig("resident parameter ranges must cover the serialized blob exactly once".into()));
        }
        end = block.range.end;
    }
    if end != total || total == 0 {
        return Err(Error::InvalidConfig("resident parameter range total mismatch".into()));
    }
    Ok(())
}

impl<B> GpuExecutableNetwork<B> where B: F32TensorOpsBackend, B::Error: std::fmt::Display {
    /// Muon deliberately keeps its existing hybrid implementation in this
    /// slice. CPU-fallback networks must use host caches/accumulators together.
    pub(crate) fn resident_parameter_path_enabled(&self) -> bool {
        !self.network.net.muon && !self.requires_cpu_backward_fallback()
            && !has_host_only_deform(&self.network)
    }

    /// Public host edits are accepted when the mirror is current. Editing a
    /// stale mirror is an error, since either copy could otherwise be lost.
    pub(crate) fn resident_parameter_slices(&self, index: usize, host: &[f32],
        layout: Vec<ResidentParameterRange>, slices: &[Range<usize>], config: ElementwiseLaunchConfig,
    ) -> Result<(Vec<B::Tensor>, u64)> {
        validate_layout(&layout, host.len())?;
        for range in slices {
            if range.start > range.end || range.end > host.len() {
                return Err(Error::InvalidConfig("resident parameter slice outside blob".into()));
            }
        }
        let be = &self.backend;
        let err = |e| backend_error("resident parameter upload/slice", e);
        let mut banks = self.resident_parameter_banks.borrow_mut();
        if banks.len() != self.network.layers.len() {
            if banks.iter().any(Option::is_some) {
                return Err(Error::InvalidConfig("resident parameter network layout changed; synchronize and reload first".into()));
            }
            *banks = vec![None; self.network.layers.len()];
        }
        let slot = banks.get_mut(index).ok_or_else(|| Error::InvalidConfig("resident parameter layer index".into()))?;
        if let Some(state) = slot.as_mut() {
            if state.layout != layout {
                return Err(Error::InvalidConfig("resident parameter geometry changed; reload weights before reuse".into()));
            }
            if !same_bits(host, &state.snapshot) {
                if state.dirty {
                    return Err(Error::InvalidConfig("host parameters changed while device parameters were newer; synchronize before editing params".into()));
                }
                state.weights = be.upload_f32(&[host.len()], host).map_err(err)?;
                state.snapshot = host.to_vec();
                state.generation = state.generation.wrapping_add(1);
            }
        } else {
            let mut mask = vec![0.0;host.len()];
            let mut coupled_mask = vec![0.0;host.len()];
            let mut first = vec![0.0;host.len()];
            let mut second = first.clone();
            let mut has_moments = false;
            let host_moments = self.host_adam_state.borrow();
            let parent = host_moments.get(index).and_then(Option::as_ref);
            for block in &layout {
                if block.decay { mask[block.range.clone()].fill(1.0); }
                if block.coupled_decay { coupled_mask[block.range.clone()].fill(1.0); }
                if let Some((m,v)) = parent.and_then(|p|block.moment_slot.get(p)) {
                    if m.len()!=block.range.len() || v.len()!=block.range.len() {
                        return Err(Error::InvalidConfig("resident Adam sidecar moment length mismatch".into()));
                    }
                    first[block.range.clone()].copy_from_slice(m);
                    second[block.range.clone()].copy_from_slice(v);
                    has_moments = true;
                }
            }
            let moments = if has_moments { Some((be.upload_f32(&[host.len()],&first).map_err(err)?,
                be.upload_f32(&[host.len()],&second).map_err(err)?)) } else {None};
            let host_buffers = self.cpu_fallback_train_buffers.borrow();
            let updates = match host_buffers.get(index) {
                Some(buffer) => {
                    let values = host_updates(buffer)?;
                    if values.len() != host.len() {
                        return Err(Error::InvalidConfig("resident accumulator length mismatch".into()));
                    }
                    be.upload_f32(&[host.len()], &values).map_err(err)?
                }
                None => be.zeros_f32(&[host.len()]).map_err(err)?,
            };
            *slot = Some(ResidentParameterState { weights:be.upload_f32(&[host.len()],host).map_err(err)?,
                updates, moments, decay_mask:be.upload_f32(&[host.len()],&mask).map_err(err)?, layout,
                coupled_decay_mask:be.upload_f32(&[host.len()],&coupled_mask).map_err(err)?,
                snapshot:host.to_vec(), generation:0, dirty:false });
        }
        let state = slot.as_ref().ok_or_else(||Error::InvalidConfig("missing resident parameter bank".into()))?;
        let mut output = Vec::with_capacity(slices.len());
        for range in slices {
            output.push(if range.is_empty() {be.zeros_f32(&[1]).map_err(err)?} else {
                be.route_concat_f32(&[&state.weights],&[range.start],&[range.len()],&[host.len()],
                    &[1,range.len(),1,1],1,config).map_err(err)?
            });
        }
        Ok((output,state.generation))
    }

    /// Concatenate mathematical gradient blocks directly on device, then
    /// accumulate the descending direction. Generation pins the forward's
    /// parameter snapshot; two backward calls before an update remain legal.
    pub(crate) fn accumulate_resident_parameter_gradients(&self,index:usize,generation:u64,
        mut gradients:Vec<(Range<usize>,B::Tensor)>,config:ElementwiseLaunchConfig)->Result<()> {
        gradients.sort_by_key(|(r,_)|r.start);
        let mut banks=self.resident_parameter_banks.borrow_mut();
        let state=banks.get_mut(index).and_then(Option::as_mut)
            .ok_or_else(||Error::InvalidConfig("missing resident gradient bank".into()))?;
        // Gradient operation boundaries need not equal optimizer group
        // boundaries (Kimi has many projections but two checkpoint arrays).
        let mut end = 0;
        let coverage = gradients.iter().all(|(range, _)| {
            let valid = range.start == end && range.end > end && range.end <= state.snapshot.len();
            end = range.end;
            valid
        }) && end == state.snapshot.len();
        if state.generation!=generation || !coverage {
            return Err(Error::InvalidConfig("resident gradient snapshot/range mismatch; rerun training forward after parameter updates".into()));
        }
        let refs=gradients.iter().map(|(_,t)|t).collect::<Vec<_>>();
        let lengths=gradients.iter().map(|(r,_)|r.len()).collect::<Vec<_>>();
        let count=gradients.len();
        let err=|e|backend_error("resident gradient accumulation",e);
        let packed=self.backend.route_concat_f32(&refs,&vec![0;count],&lengths,&lengths,
            &[1,state.snapshot.len(),1,1],1,config).map_err(err)?;
        state.updates=self.backend.saxpy_f32(-1.0,&packed,&state.updates,config).map_err(err)?;
        Ok(())
    }

    /// Diagnostic only: ordinary native backward does not refresh host buffers.
    /// Repeated calls replace the mirror, so inspection cannot double-count.
    pub fn synchronize_resident_parameter_gradients(&self)->Result<()> {
        let banks=self.resident_parameter_banks.borrow();
        let mut buffers=self.cpu_fallback_train_buffers.borrow_mut();
        for (index,state) in banks.iter().enumerate() {
            let Some(state)=state else {continue;};
            let values=self.backend.download_f32(&state.updates)
                .map_err(|e|backend_error("download resident gradient diagnostic",e))?;
            let buffer = buffers.get_mut(index)
                .ok_or_else(|| Error::InvalidConfig("resident host accumulator index".into()))?;
            set_host_updates(buffer, &values)?;
        }
        Ok(())
    }

    pub(super) fn materialize_resident_optimizer_states(&self,states:&mut[Option<HostAdamLayerState>])->Result<()> {
        let banks=self.resident_parameter_banks.borrow();
        for (index,state) in banks.iter().enumerate() {
            let Some(state)=state else{continue;};
            let Some((m,v))=&state.moments else{continue;};
            let m=self.backend.download_f32(m).map_err(|e|backend_error("download resident Adam first moment",e))?;
            let v=self.backend.download_f32(v).map_err(|e|backend_error("download resident Adam second moment",e))?;
            if m.len()!=state.snapshot.len() || v.len()!=m.len(){return Err(Error::InvalidConfig("resident moment length mismatch".into()));}
            let parent=states.get_mut(index).ok_or_else(||Error::InvalidConfig("resident Adam host-state index".into()))?
                .get_or_insert_with(HostAdamLayerState::default);
            for block in &state.layout {
                block.moment_slot.set(parent, (m[block.range.clone()].to_vec(), v[block.range.clone()].to_vec()));
            }
        }
        Ok(())
    }

    /// Materialize weights, gradients/momentum and Adam state at an explicit
    /// boundary. It leaves device banks alive for the next forward/update.
    pub fn synchronize_resident_parameter_state(&mut self)->Result<()> {
        self.synchronize_resident_parameter_gradients()?;
        let mut moments=self.host_adam_state.borrow().clone();
        self.materialize_resident_optimizer_states(&mut moments)?;
        let mut banks=self.resident_parameter_banks.borrow_mut();
        for (index,state) in banks.iter_mut().enumerate(){
            let Some(state)=state else{continue;};
            if !state.dirty{continue;}
            let params = self.params.get_mut(index)
                .ok_or_else(|| Error::InvalidConfig("resident parameter host index".into()))?;
            if !host_values_match(params,&state.snapshot)?{return Err(Error::InvalidConfig("host parameters were edited before deferred device synchronization".into()));}
            let values=self.backend.download_f32(&state.weights).map_err(|e|backend_error("download resident parameter mirror",e))?;
            set_host_values(params, &values)?;
            state.snapshot=values;state.dirty=false;
        }
        *self.host_adam_state.get_mut()=moments;
        Ok(())
    }

    pub(super) fn retire_resident_parameter_banks(&mut self)->Result<()> {
        self.synchronize_resident_parameter_state()?;
        self.resident_parameter_banks.get_mut().clear();
        Ok(())
    }

    pub(super) fn reset_resident_parameter_gradients(&mut self)->Result<()> {
        for state in self.resident_parameter_banks.get_mut().iter_mut().flatten(){
            state.updates=self.backend.zeros_f32(&[state.snapshot.len()])
                .map_err(|e|backend_error("reset resident parameter gradients",e))?;
        }
        Ok(())
    }

    pub(super) fn update_resident_parameters(&mut self,step:&GpuOptimizerStep,
        sync_host:bool,config:ElementwiseLaunchConfig)->Result<()> {
        let iteration=self.train_iteration as i64+1;
        let be=&self.backend;
        let err=|e|backend_error("resident parameter optimizer",e);
        let banks=self.resident_parameter_banks.get_mut();
        // Clone the handles so a failed layer update never installs only one
        // of its weights/updates/moments. Each layer publishes one new state.
        for (index,slot) in banks.iter_mut().enumerate(){
            let Some(old)=slot.as_ref()else{continue;};
            let layer=&self.network.layers[index];
            if !layer.train.updates_at(iteration,self.network.net.burn_in){continue;}
            let host = self.params.get(index)
                .ok_or_else(|| Error::InvalidConfig("resident optimizer parameter index".into()))?;
            if !host_values_match(host,&old.snapshot)?{return Err(Error::InvalidConfig("host parameters changed after native forward; rerun forward or synchronize before updating".into()));}
            let mut next=old.clone();
            let masked=be.vector_mul_f32(&old.weights,&old.decay_mask,&[old.snapshot.len()],config).map_err(err)?;
            match step {
                GpuOptimizerStep::Sgd{lr,momentum,decay,batch}=>{
                    let effective=be.saxpy_f32(-*decay*(*batch).max(1) as f32,&masked,&old.updates,config).map_err(err)?;
                    let (weights,updates)=be.sgd_update_f32(&old.weights,&effective,
                        *lr*layer.train.learning_rate_scale,*momentum,0.0,*batch,config).map_err(err)?;
                    next.weights=weights;next.updates=updates;
                }
                GpuOptimizerStep::Adam(adam)=>{
                    let adam=AdamParams{lr:adam.lr*layer.train.learning_rate_scale,..*adam};
                    let (coupled,decoupled,_,_)=adam.kernel_scalars();
                    let effective=if coupled==0.0{old.updates.clone()} else {
                        let coupled_weights = be.vector_mul_f32(&old.weights, &old.coupled_decay_mask,
                            &[old.snapshot.len()], config).map_err(err)?;
                        be.saxpy_f32(-coupled,&coupled_weights,&old.updates,config).map_err(err)?
                    };
                    let moments=match &old.moments{
                        Some(pair)=>pair.clone(),
                        None=>(be.zeros_f32(&[old.snapshot.len()]).map_err(err)?,be.zeros_f32(&[old.snapshot.len()]).map_err(err)?),
                    };
                    let (weights,m,v)=be.adam_update_f32(&old.weights,&effective,&moments.0,&moments.1,
                        &AdamParams{decay:0.0,..adam},config).map_err(err)?;
                    next.weights=if decoupled==0.0{weights}else{be.saxpy_f32(-decoupled,&masked,&weights,config).map_err(err)?};
                    next.moments=Some((m,v));next.updates=be.zeros_f32(&[old.snapshot.len()]).map_err(err)?;
                }
            }
            next.dirty=true;next.generation=next.generation.wrapping_add(1);*slot=Some(next);
            // A later layer can fail. Preserve the staleness marker as soon
            // as this layer publishes a new authoritative device state.
            self.host_weights_stale=true;
        }
        if sync_host{self.synchronize_resident_parameter_state()?;}
        Ok(())
    }
}
