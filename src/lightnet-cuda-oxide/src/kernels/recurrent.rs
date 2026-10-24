//! Recurrent kernels: LSTM gate backward and sequence slice/concat steps.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Copy one contiguous sequence chunk to an independent output tensor.
    #[kernel]
    pub fn sequence_slice_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < output.len() {
            // SAFETY: the host validates the timestep range against input.
            unsafe {
                *output.as_mut_ptr().add(i) = input[offset + i];
            }
        }
    }

    /// Copy one contiguous recurrent timestep into its disjoint output range.
    #[kernel]
    pub fn sequence_concat_step(input: &[f32], mut output: DisjointSlice<f32>, offset: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            // SAFETY: host-provided offsets partition the output by timestep.
            unsafe {
                *output.as_mut_ptr().add(offset + i) = input[i];
            }
        }
    }

    /// Exact LSTM cell/gate derivative for one BPTT step. Each thread owns one
    /// hidden element and writes all gate deltas plus the cell gradient for the
    /// preceding step.
    #[kernel]
    pub fn lstm_gate_backward(
        hidden_gradient: &[f32],
        cell: &[f32],
        previous_cell: &[f32],
        forget: &[f32],
        input_gate: &[f32],
        candidate: &[f32],
        output_gate: &[f32],
        next_cell_gradient: &[f32],
        mut forget_gradient: DisjointSlice<f32>,
        mut input_gate_gradient: DisjointSlice<f32>,
        mut candidate_gradient: DisjointSlice<f32>,
        mut output_gate_gradient: DisjointSlice<f32>,
        mut cell_gradient_out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= hidden_gradient.len() {
            return;
        }
        let tanh_cell = cell[i].tanh();
        let output_delta = hidden_gradient[i] * tanh_cell;
        let cell_delta = hidden_gradient[i] * output_gate[i] * (1.0 - tanh_cell * tanh_cell)
            + next_cell_gradient[i];
        let mut forget_delta = cell_delta * previous_cell[i];
        let mut input_delta = cell_delta * candidate[i];
        let mut candidate_delta = cell_delta * input_gate[i];
        forget_delta *= forget[i] * (1.0 - forget[i]);
        input_delta *= input_gate[i] * (1.0 - input_gate[i]);
        candidate_delta *= 1.0 - candidate[i] * candidate[i];
        let mut output_delta = output_delta;
        output_delta *= output_gate[i] * (1.0 - output_gate[i]);
        // SAFETY: each thread writes the matching element in each output.
        unsafe {
            *forget_gradient.as_mut_ptr().add(i) = forget_delta;
            *input_gate_gradient.as_mut_ptr().add(i) = input_delta;
            *candidate_gradient.as_mut_ptr().add(i) = candidate_delta;
            *output_gate_gradient.as_mut_ptr().add(i) = output_delta;
            *cell_gradient_out.as_mut_ptr().add(i) = cell_delta * forget[i];
        }
    }
}
