//! Flat serialized device parameter banks. Native SGD/Adam/Muon retain weights,
//! descending accumulators and moments between steps; host mirrors are only
//! materialized by explicit diagnostics, eager updates or checkpoint writing.
use super::*;
use std::ops::Range;
use crate::muon_runtime::{SerializedMuonMatrix, serialized_mixer_muon_layout};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResidentMomentSlot {
    ChildWeights(usize),
    ChildBiases(usize),
    Weights,
    Biases,
}

impl ResidentMomentSlot {
    fn get<'a>(&self, state: &'a HostAdamLayerState) -> Option<&'a (Vec<f32>, Vec<f32>)> {
        match self {
            Self::ChildWeights(index) => state.children.get(*index)?.weights.as_ref(),
            Self::ChildBiases(index) => state.children.get(*index)?.biases.as_ref(),
            Self::Weights => state.weights.as_ref(),
            Self::Biases => state.biases.as_ref(),
        }
    }

    fn set(&self, state: &mut HostAdamLayerState, pair: (Vec<f32>, Vec<f32>)) {
        match self {
            Self::ChildWeights(index) => host_adam_child(state, *index).weights = Some(pair),
            Self::ChildBiases(index) => host_adam_child(state, *index).biases = Some(pair),
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
    pub moment_slot: Option<ResidentMomentSlot>,
    /// False ranges are serialized but frozen by the CPU optimizer.
    pub active: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentParameterState<T> {
    pub weights: T,
    pub updates: T,
    pub moments: Option<(T, T)>,
    decay_mask: T,
    coupled_decay_mask: T,
    update_mask: T,
    muon_matrices: Vec<(SerializedMuonMatrix, T)>,
    muon_adam_mask: Option<T>,
    layout: Vec<ResidentParameterRange>,
    snapshot: Vec<f32>,
    pub generation: u64,
    pub dirty: bool,
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a,b)| a.to_bits() == b.to_bits())
}

fn next_resident_generation(epoch: &Cell<u64>) -> Result<u64> {
    let next = epoch.get().checked_add(1)
        .ok_or_else(|| Error::InvalidConfig("resident parameter generation exhausted".into()))?;
    epoch.set(next);
    Ok(next)
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
        LayerParams::Gru(p) => Ok(same_bits(&crate::layers::gru::cpu::serialized(p),snapshot)),
        LayerParams::DptHead(p) => Ok(same_bits(&p.values,snapshot)),
        LayerParams::Vit(p) => Ok(same_bits(&p.to_blob(),snapshot)),
        _ => Err(Error::InvalidConfig("resident parameter host kind mismatch".into())),
    }
}

fn host_updates(buffer: &LayerTrainBuffers) -> Result<Vec<f32>> {
    match buffer {
        LayerTrainBuffers::MambaVision(t) => Ok(t.updates.clone()),
        LayerTrainBuffers::KimiKda(t) => Ok([t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()),
        LayerTrainBuffers::TuckerAttention(t) => Ok([t.bias_updates.as_slice(), t.weight_updates.as_slice()].concat()),
        LayerTrainBuffers::Gru(t) => Ok([&t.input_z,&t.state_z,&t.input_r,&t.state_r,&t.input_h,&t.state_h]
            .into_iter().flat_map(|t|t.bias_updates.iter().chain(&t.weight_updates).copied()).collect()),
        LayerTrainBuffers::DptHead(t) => Ok(t.weight_updates.clone()),
        LayerTrainBuffers::Vit(t) => Ok(t.updates.to_blob()),
        _ => Err(Error::InvalidConfig("resident accumulator host kind mismatch".into())),
    }
}

fn set_host_updates(buffer: &mut LayerTrainBuffers, values: &[f32]) -> Result<()> {
    if let LayerTrainBuffers::Gru(t)=buffer {
        let fields=[&mut t.input_z,&mut t.state_z,&mut t.input_r,&mut t.state_r,&mut t.input_h,&mut t.state_h];
        let total=fields.iter().map(|t|t.bias_updates.len()+t.weight_updates.len()).sum::<usize>();
        if total!=values.len(){return Err(Error::InvalidConfig("resident GRU accumulator length mismatch".into()));}
        let mut cursor=0;
        for t in fields {
            let end=cursor+t.bias_updates.len();t.bias_updates.copy_from_slice(&values[cursor..end]);cursor=end;
            let end=cursor+t.weight_updates.len();t.weight_updates.copy_from_slice(&values[cursor..end]);cursor=end;
        }
        return Ok(());
    }
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
        LayerTrainBuffers::DptHead(t) if t.weight_updates.len() == values.len() => t.weight_updates.copy_from_slice(values),
        LayerTrainBuffers::Vit(t) => t.updates.set_serialized(values).map_err(Error::InvalidConfig)?,
        _ => return Err(Error::InvalidConfig("resident host accumulator mirror mismatch".into())),
    }
    Ok(())
}

fn set_host_values(params: &mut LayerParams, values: &[f32]) -> Result<()> {
    match params {
        LayerParams::Gru(p) => crate::layers::gru::cpu::set_serialized(p,values)?,
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
        LayerParams::DptHead(p) if p.values.len() == values.len() => p.values.copy_from_slice(values),
        LayerParams::Vit(p) => p.set_serialized(values).map_err(Error::InvalidConfig)?,
        _ => return Err(Error::InvalidConfig("resident parameter host length/kind mismatch".into())),
    }
    Ok(())
}

fn validate_layout(layout: &[ResidentParameterRange], total: usize) -> Result<()> {
    let mut end = 0;
    for block in layout {
        if block.range.start != end || block.range.end < end || block.range.end > total {
            return Err(Error::InvalidConfig("resident parameter ranges must cover the serialized blob exactly once".into()));
        }
        end = block.range.end;
    }
    if end != total || total == 0 {
        return Err(Error::InvalidConfig("resident parameter range total mismatch".into()));
    }
    Ok(())
}

/// ViT's serialized groups share one clip norm, but mHC has a smaller learning
/// rate and a parameter clamp. Keep those rules outside the uniform-bank step.
fn stage_vit_resident_optimizer<B: F32TensorOpsBackend>(be: &B,
    old: &ResidentParameterState<B::Tensor>, step: &GpuOptimizerStep,
    layer_rate: f32, config: ElementwiseLaunchConfig,
) -> Result<ResidentParameterState<B::Tensor>> where B::Error: std::fmt::Display {
    let err=|e|backend_error("resident ViT optimizer",e);
    let total=old.snapshot.len();
    let positions=old.layout.get(14).ok_or_else(||Error::InvalidConfig("ViT resident position group missing".into()))?;
    let excluded=if positions.active {0..0} else {positions.range.clone()};
    let prepared=be.vit_prepare_updates_f32(&old.updates,excluded.start,excluded.end,5.0,config).map_err(err)?;
    let slice=|t:&B::Tensor,r:Range<usize>|be.route_concat_f32(&[t],&[r.start],&[r.len()],&[total],
        &[r.len()],1,config).map_err(err);
    let mut weights=Vec::new(); let mut updates=Vec::new();
    let mut first=Vec::new(); let mut second=Vec::new();
    for (index,block) in old.layout.iter().enumerate() {
        let w=slice(&old.weights,block.range.clone())?;
        let u=slice(&prepared,block.range.clone())?;
        let (m,v)=match &old.moments {
            Some((m,v))=>(slice(m,block.range.clone())?,slice(v,block.range.clone())?),
            None=>(be.zeros_f32(&[block.range.len()]).map_err(err)?,be.zeros_f32(&[block.range.len()]).map_err(err)?),
        };
        if !block.active { weights.push(w); updates.push(u); first.push(m); second.push(v); continue; }
        let rate=layer_rate*if index==15 {0.1} else {1.0};
        let (w,u,m,v)=match step {
            GpuOptimizerStep::Sgd{lr,momentum,decay,batch}=>{
                let (w,u)=be.sgd_update_f32(&w,&u,*lr*rate,*momentum,
                    if block.decay {*decay} else {0.0},*batch,config).map_err(err)?;
                (w,u,m,v)
            }
            GpuOptimizerStep::Adam(adam)=>{
                let decay=if adam.decoupled_decay {block.decay} else {block.coupled_decay};
                let params=AdamParams{lr:adam.lr*rate,decay:if decay {adam.decay} else {0.0},..*adam};
                let (w,m,v)=be.adam_update_f32(&w,&u,&m,&v,&params,config).map_err(err)?;
                (w,be.zeros_f32(&[block.range.len()]).map_err(err)?,m,v)
            }
        };
        weights.push(if index==15 {be.vit_clamp_feature_f32(&w,8.0,config).map_err(err)?} else {w});
        updates.push(u); first.push(m); second.push(v);
    }
    let lengths=old.layout.iter().map(|b|b.range.len()).collect::<Vec<_>>();
    let pack=|v:&[B::Tensor]|be.route_concat_f32(&v.iter().collect::<Vec<_>>(),&vec![0;v.len()],
        &lengths,&lengths,&[total],1,config).map_err(err);
    let mut next=old.clone();
    next.weights=pack(&weights)?; next.updates=pack(&updates)?;
    if matches!(step,GpuOptimizerStep::Adam(_)) {next.moments=Some((pack(&first)?,pack(&second)?));}
    Ok(next)
}

impl<B> GpuExecutableNetwork<B> where B: F32TensorOpsBackend, B::Error: std::fmt::Display {
    /// CPU-fallback networks must use host caches/accumulators together.
    pub(crate) fn resident_parameter_path_enabled(&self) -> bool {
        !self.requires_cpu_backward_fallback()
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
                state.generation = next_resident_generation(&self.resident_parameter_epoch)?;
            }
        } else {
            let mut mask = vec![0.0;host.len()];
            let mut coupled_mask = vec![0.0;host.len()];
            let mut active_mask = vec![0.0;host.len()];
            let mut first = vec![0.0;host.len()];
            let mut second = first.clone();
            let mut has_moments = false;
            let host_moments = self.host_adam_state.borrow();
            let parent = host_moments.get(index).and_then(Option::as_ref);
            for block in &layout {
                if block.decay { mask[block.range.clone()].fill(1.0); }
                if block.coupled_decay { coupled_mask[block.range.clone()].fill(1.0); }
                if block.active { active_mask[block.range.clone()].fill(1.0); }
                if let Some((m,v)) = block.moment_slot.as_ref().and_then(|slot| parent.and_then(|p|slot.get(p))) {
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
                update_mask:be.upload_f32(&[host.len()],&active_mask).map_err(err)?,
                muon_matrices:Vec::new(), muon_adam_mask:None,
                snapshot:host.to_vec(), generation:next_resident_generation(&self.resident_parameter_epoch)?, dirty:false });
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
            &[state.snapshot.len()],1,config).map_err(err)?;
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
                if let Some(slot) = &block.moment_slot {
                    slot.set(parent, (m[block.range.clone()].to_vec(), v[block.range.clone()].to_vec()));
                }
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
        let mut muon = self.muon_state.clone();
        self.materialize_resident_muon_states(&mut muon)?;
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
        self.muon_state=muon;
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

    pub(super) fn materialize_resident_muon_states(&self, states: &mut [Option<MuonLayerState>]) -> Result<()> {
        for (index, bank) in self.resident_parameter_banks.borrow().iter().enumerate() {
            let Some(bank) = bank else { continue; };
            if bank.muon_matrices.is_empty() { continue; }
            let mut matrices = Vec::with_capacity(bank.muon_matrices.len());
            for (view, tensor) in &bank.muon_matrices {
                let momentum = self.backend.download_f32(tensor)
                    .map_err(|e| backend_error("materialize resident Muon momentum",e))?;
                if momentum.len() != view.range.len() {
                    return Err(Error::InvalidConfig("resident Muon momentum length mismatch".into()));
                }
                matrices.push(MuonMatrixState { rows:view.rows, cols:view.cols, momentum });
            }
            *states.get_mut(index).ok_or_else(|| Error::InvalidConfig("resident Muon state index".into()))? =
                Some(MuonLayerState { matrices });
        }
        Ok(())
    }

    /// Matrix momentum/orthogonalization and AdamW fallback stay on device.
    /// Stage one complete layer before publishing any of its new state.
    pub(super) fn update_resident_muon_parameters(&mut self, adam: &AdamParams,
        loss_scale: f32, config: ElementwiseLaunchConfig) -> Result<()> {
        let be = &self.backend;
        let err = |e| backend_error("resident Muon optimizer",e);
        let iteration = self.train_iteration as i64 + 1;
        for (index, slot) in self.resident_parameter_banks.get_mut().iter_mut().enumerate() {
            let Some(old) = slot.as_ref() else { continue; };
            let layer = &self.network.layers[index];
            if !layer.train.updates_at(iteration,self.network.net.burn_in) { continue; }
            if !host_values_match(&self.params[index],&old.snapshot)? {
                return Err(Error::InvalidConfig("host parameters changed before resident Muon update; synchronize or rerun forward".into()));
            }
            let views = serialized_mixer_muon_layout(&layer.kind,layer.input_shape,&self.params[index])?;
            let dpt_policy = matches!(layer.kind, LayerKind::DptHead(_));
            let total = old.snapshot.len();
            let slice = |tensor: &B::Tensor, range: Range<usize>| -> Result<B::Tensor> {
                be.route_concat_f32(&[tensor],&[range.start],&[range.len()],&[total],
                    &[1,range.len(),1,1],1,config).map_err(err)
            };
            let mut next = old.clone();
            if next.muon_matrices.is_empty() {
                let prior = self.muon_state.get(index).and_then(Option::as_ref);
                if prior.is_some_and(|p| !p.matrices.is_empty() && p.matrices.len()!=views.len()) {
                    return Err(Error::InvalidConfig("resident Muon sidecar matrix count mismatch".into()));
                }
                let mut mask = vec![0.0;total];
                for block in &next.layout {
                    if block.active { mask[block.range.clone()].fill(1.0); }
                }
                for (i,view) in views.iter().enumerate() {
                    mask[view.range.clone()].fill(0.0);
                    let momentum = if let Some(p) = prior.and_then(|p|p.matrices.get(i)) {
                        if p.rows!=view.rows || p.cols!=view.cols || p.momentum.len()!=view.range.len() {
                            return Err(Error::InvalidConfig("resident Muon sidecar matrix shape mismatch".into()));
                        }
                        be.upload_f32(&[view.rows,view.cols],&p.momentum).map_err(err)?
                    } else { be.zeros_f32(&[view.rows,view.cols]).map_err(err)? };
                    next.muon_matrices.push((view.clone(),momentum));
                }
                next.muon_adam_mask=Some(be.upload_f32(&[total],&mask).map_err(err)?);
            }
            if next.muon_matrices.iter().map(|(v,_)|v).ne(views.iter()) {
                return Err(Error::InvalidConfig("resident Muon matrix layout changed".into()));
            }
            if be.any_non_finite_f32(&old.weights).map_err(err)? || be.any_non_finite_f32(&old.updates).map_err(err)? {
                return Err(Error::InvalidConfig("resident Muon parameters/gradients must be finite".into()));
            }
            let layer_adam=AdamParams { lr:adam.lr*layer.train.learning_rate_scale,..*adam };
            // The CPU stages matrix directions, zeros their Adam gradients,
            // then advances AdamW for the whole layer. Existing matrix Adam
            // moments must still decay/advance, including resumed sidecars.
            let effective=be.vector_mul_f32(&old.updates,next.muon_adam_mask.as_ref()
                .ok_or_else(||Error::InvalidConfig("missing resident Muon Adam mask".into()))?,&[total],config).map_err(err)?;
            let moments=match &old.moments {
                Some(pair)=>pair.clone(),
                None=>(be.zeros_f32(&[total]).map_err(err)?,be.zeros_f32(&[total]).map_err(err)?),
            };
            let (base,m,v)=be.adam_update_f32(&old.weights,&effective,&moments.0,&moments.1,
                &AdamParams{decay:0.0,..layer_adam},config).map_err(err)?;
            let masked=be.vector_mul_f32(&old.weights,&old.decay_mask,&[total],config).map_err(err)?;
            let base=be.saxpy_f32(-layer_adam.lr*layer_adam.decay,&masked,&base,config).map_err(err)?;
            let mut pieces=Vec::with_capacity(2*views.len()+1);
            let mut cursor=0;
            for (view,momentum) in &mut next.muon_matrices {
                let shape=[view.rows,view.cols];
                let updates=slice(&old.updates,view.range.clone())?;
                let (next_momentum,lookahead)=be.muon_momentum_lookahead_f32(&updates,momentum,&shape,
                    adam.batch,loss_scale,self.network.net.muon_momentum).map_err(err)?;
                if be.any_non_finite_f32(&next_momentum).map_err(err)? || be.any_non_finite_f32(&lookahead).map_err(err)? {
                    return Err(Error::InvalidConfig("resident Muon momentum became nonfinite".into()));
                }
                let direction=be.muon_orthogonalize_f32(&lookahead,&shape,view.rows,view.cols).map_err(err)?;
                let weights=slice(&base,view.range.clone())?;
                let updated=be.muon_apply_update_f32(&weights,&direction,&shape,layer_adam.lr,0.0,
                    crate::muon::MUON_SCALE*(view.rows.max(view.cols) as f32).sqrt()).map_err(err)?;
                if be.any_non_finite_f32(&updated).map_err(err)? {
                    return Err(Error::InvalidConfig("resident Muon update became nonfinite".into()));
                }
                if cursor<view.range.start { pieces.push((cursor..view.range.start,slice(&base,cursor..view.range.start)?)); }
                pieces.push((view.range.clone(),updated)); cursor=view.range.end;
                *momentum=next_momentum;
            }
            if cursor<total { pieces.push((cursor..total,slice(&base,cursor..total)?)); }
            let refs=pieces.iter().map(|(_,t)|t).collect::<Vec<_>>();
            let lengths=pieces.iter().map(|(r,_)|r.len()).collect::<Vec<_>>();
            next.weights=be.route_concat_f32(&refs,&vec![0;pieces.len()],&lengths,&lengths,
                &[1,total,1,1],1,config).map_err(err)?;
            if dpt_policy {
                let inactive=be.saxpy_f32(-1.0,&old.update_mask,
                    &be.upload_f32(&[total],&vec![1.0;total]).map_err(err)?,config).map_err(err)?;
                let active_weights=be.vector_mul_f32(&next.weights,&old.update_mask,&[total],config).map_err(err)?;
                let frozen_weights=be.vector_mul_f32(&old.weights,&inactive,&[total],config).map_err(err)?;
                next.weights=be.vector_add_f32(&active_weights,&frozen_weights,config).map_err(err)?;
                let (old_m,old_v)=match &old.moments { Some((m,v))=>(m.clone(),v.clone()),
                    None=>(be.zeros_f32(&[total]).map_err(err)?,be.zeros_f32(&[total]).map_err(err)?) };
                let ones=be.upload_f32(&[total],&vec![1.0;total]).map_err(err)?;
                let adam_mask=next.muon_adam_mask.as_ref()
                    .ok_or_else(||Error::InvalidConfig("missing resident Muon Adam mask".into()))?;
                let inactive_adam=be.saxpy_f32(-1.0,adam_mask,&ones,config).map_err(err)?;
                let m_active=be.vector_mul_f32(&m,adam_mask,&[total],config).map_err(err)?;
                let m_frozen=be.vector_mul_f32(&old_m,&inactive_adam,&[total],config).map_err(err)?;
                let v_active=be.vector_mul_f32(&v,adam_mask,&[total],config).map_err(err)?;
                let v_frozen=be.vector_mul_f32(&old_v,&inactive_adam,&[total],config).map_err(err)?;
                next.moments=Some((be.vector_add_f32(&m_active,&m_frozen,config).map_err(err)?,
                    be.vector_add_f32(&v_active,&v_frozen,config).map_err(err)?));
                next.updates=be.vector_mul_f32(&old.updates,&inactive,&[total],config).map_err(err)?;
            } else {
                next.moments=Some((m,v)); next.updates=be.zeros_f32(&[total]).map_err(err)?;
            }
            if be.any_non_finite_f32(&next.weights).map_err(err)? {
                return Err(Error::InvalidConfig("resident Muon AdamW fallback became nonfinite".into()));
            }
            next.generation=next_resident_generation(&self.resident_parameter_epoch)?; next.dirty=true;
            *slot=Some(next); self.host_weights_stale=true;
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
            if matches!(layer.kind,LayerKind::Vit(_)) {
                let mut next=stage_vit_resident_optimizer(be,old,step,layer.train.learning_rate_scale,config)?;
                next.dirty=true; next.generation=next_resident_generation(&self.resident_parameter_epoch)?;
                *slot=Some(next); self.host_weights_stale=true;
                continue;
            }
            let mut next=old.clone();
            let inactive_mask=be.saxpy_f32(-1.0,&old.update_mask,
                &be.upload_f32(&[old.snapshot.len()],&vec![1.0;old.snapshot.len()]).map_err(err)?,config).map_err(err)?;
            let masked=be.vector_mul_f32(&old.weights,&old.decay_mask,&[old.snapshot.len()],config).map_err(err)?;
            match step {
                GpuOptimizerStep::Sgd{lr,momentum,decay,batch}=>{
                    let effective=be.saxpy_f32(-*decay*(*batch).max(1) as f32,&masked,&old.updates,config).map_err(err)?;
                    let (weights,updates)=be.sgd_update_f32(&old.weights,&effective,
                        *lr*layer.train.learning_rate_scale,*momentum,0.0,*batch,config).map_err(err)?;
                    let active_weights=be.vector_mul_f32(&weights,&old.update_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let frozen_weights=be.vector_mul_f32(&old.weights,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                    next.weights=be.vector_add_f32(&active_weights,&frozen_weights,config).map_err(err)?;
                    let active_updates=be.vector_mul_f32(&updates,&old.update_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let frozen_updates=be.vector_mul_f32(&old.updates,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                    next.updates=be.vector_add_f32(&active_updates,&frozen_updates,config).map_err(err)?;
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
                    let weights=if decoupled==0.0{weights}else{be.saxpy_f32(-decoupled,&masked,&weights,config).map_err(err)?};
                    let active_weights=be.vector_mul_f32(&weights,&old.update_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let frozen_weights=be.vector_mul_f32(&old.weights,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                    next.weights=be.vector_add_f32(&active_weights,&frozen_weights,config).map_err(err)?;
                    let (m0,v0)=match &old.moments {Some((m0,v0))=>(m0.clone(),v0.clone()),None=>(be.zeros_f32(&[old.snapshot.len()]).map_err(err)?,be.zeros_f32(&[old.snapshot.len()]).map_err(err)?)};
                    let am=be.vector_mul_f32(&m,&old.update_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let fm=be.vector_mul_f32(&m0,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let av=be.vector_mul_f32(&v,&old.update_mask,&[old.snapshot.len()],config).map_err(err)?;
                    let fv=be.vector_mul_f32(&v0,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                    next.moments=Some((be.vector_add_f32(&am,&fm,config).map_err(err)?,be.vector_add_f32(&av,&fv,config).map_err(err)?));
                    next.updates=be.vector_mul_f32(&old.updates,&inactive_mask,&[old.snapshot.len()],config).map_err(err)?;
                }
            }
            next.dirty=true;next.generation=next_resident_generation(&self.resident_parameter_epoch)?;*slot=Some(next);
            // A later layer can fail. Preserve the staleness marker as soon
            // as this layer publishes a new authoritative device state.
            self.host_weights_stale=true;
        }
        if sync_host{self.synchronize_resident_parameter_state()?;}
        Ok(())
    }
}
