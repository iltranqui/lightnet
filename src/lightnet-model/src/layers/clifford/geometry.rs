use crate::model::{CliffordInteractionMode, CliffordSpec};

#[doc(hidden)]
pub fn raw_channels(spec: &CliffordSpec, channels: usize) -> usize {
    let shifts = match spec.interaction_mode {
        CliffordInteractionMode::Inner => spec.inner_shifts.len(),
        CliffordInteractionMode::Wedge => spec.wedge_shifts.len(),
        CliffordInteractionMode::Full => spec.wedge_shifts.len() + spec.inner_shifts.len(),
    };
    channels * shifts
}
