//! Newton–Schulz matrix operations for the native Muon optimizer path.

use super::resample::atomic_add_f32;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Transpose row-major `rows x cols` into row-major `cols x rows`.
    #[kernel]
    pub fn muon_transpose_f32(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < rows * cols {
            let out_row = i / rows;
            let out_col = i % rows;
            if let Some(value) = output.get_mut(idx) {
                *value = input[out_col * cols + out_row];
            }
        }
    }

    /// Accumulate `sum(input[i]^2)` into a zero-initialized scalar.
    #[kernel]
    pub fn muon_sum_squares_f32(input: &[f32], output: *mut f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            unsafe { atomic_add_f32(output, input[i] * input[i]) };
        }
    }

    /// Update momentum and form `mu * next_momentum + gradient` lookahead.
    #[kernel]
    pub fn muon_momentum_lookahead_f32(
        updates: &[f32],
        old_momentum: &[f32],
        mut next_momentum: DisjointSlice<f32>,
        mut lookahead: DisjointSlice<f32>,
        divisor: f32,
        coefficient: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < updates.len() {
            let gradient = updates[i] / divisor;
            let next = coefficient * old_momentum[i] + gradient;
            let nesterov = coefficient * next + gradient;
            if let Some(dst) = next_momentum.get_mut(idx) {
                *dst = next;
            }
            if let Some(dst) = lookahead.get_mut(thread::index_1d()) {
                *dst = nesterov;
            }
        }
    }

    /// Apply decoupled decay and a scaled Muon direction.
    #[kernel]
    pub fn muon_apply_update_f32(
        weights: &[f32],
        direction: &[f32],
        mut output: DisjointSlice<f32>,
        learning_rate: f32,
        decay_factor: f32,
        direction_scale: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < weights.len() {
            if let Some(dst) = output.get_mut(idx) {
                *dst = decay_factor * weights[i] + learning_rate * direction_scale * direction[i];
            }
        }
    }

    /// Divide a matrix by its Frobenius norm; an all-zero input remains zero.
    #[kernel]
    pub fn muon_normalize_f32(input: &[f32], norm_squared: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            let norm_squared = norm_squared[0];
            let value = if norm_squared > 0.0 {
                input[i] / norm_squared.sqrt()
            } else {
                0.0
            };
            if let Some(dst) = output.get_mut(idx) {
                *dst = value;
            }
        }
    }

    /// `out[i] = a*x[i] + b*y[i] + c*z[i]`.
    #[kernel]
    pub fn muon_linear_combination_f32(
        a: f32,
        x: &[f32],
        b: f32,
        y: &[f32],
        c: f32,
        z: &[f32],
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < x.len() {
            if let Some(dst) = output.get_mut(idx) {
                *dst = a * x[i] + b * y[i] + c * z[i];
            }
        }
    }
}
