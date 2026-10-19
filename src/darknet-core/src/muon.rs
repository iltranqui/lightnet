//! Paper-derived CPU Muon primitives.
//!
//! The update follows *Muon is Scalable for LLM Training* (arXiv:2502.16982),
//! equations (1)–(4). Runtime dispatch maps each eligible parameter block to a
//! separate matrix; this module has no layer-specific layout assumptions.

use crate::error::{Error, Result};

const NS_STEPS: usize = 5;
const NS_A: f32 = 3.4445;
const NS_B: f32 = -4.7750;
const NS_C: f32 = 2.0315;
pub(crate) const MUON_SCALE: f32 = 0.2;

/// Persistent momentum state for one logical Muon matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MuonMatrixState {
    pub rows: usize,
    pub cols: usize,
    pub momentum: Vec<f32>,
}

impl MuonMatrixState {
    pub(crate) fn zeroed(rows: usize, cols: usize) -> Result<Self> {
        let len = checked_matrix_len(rows, cols)?;
        Ok(Self {
            rows,
            cols,
            momentum: vec![0.0; len],
        })
    }
}

/// Approximate the polar factor of a row-major `rows × cols` matrix using the
/// five-step polynomial Newton–Schulz iteration from the selected Muon paper.
pub(crate) fn newton_schulz_orthogonalize(
    matrix: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>> {
    let len = checked_matrix_len(rows, cols)?;
    if matrix.len() != len {
        return Err(Error::InvalidConfig(format!(
            "Muon matrix has {} values, expected {len} for {rows}x{cols}",
            matrix.len()
        )));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(
            "Muon matrix contains a non-finite value".to_owned(),
        ));
    }

    // The polar factor is transpose-equivariant. Use the smaller Gram
    // dimension so the expensive Gram matrix has min(rows, cols)^2 entries.
    let (work_rows, work_cols, transposed) = if rows > cols {
        (cols, rows, true)
    } else {
        (rows, cols, false)
    };
    let mut x = if transposed {
        transpose(matrix, rows, cols)
    } else {
        matrix.to_vec()
    };
    let norm_squared = x
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if !norm_squared.is_finite() {
        return Err(Error::InvalidConfig(
            "Muon matrix Frobenius norm is non-finite".to_owned(),
        ));
    }
    if norm_squared == 0.0 {
        return Ok(vec![0.0; len]);
    }
    let norm = norm_squared.sqrt();
    for value in &mut x {
        *value = (f64::from(*value) / norm) as f32;
    }

    let gram_len = checked_matrix_len(work_rows, work_rows)?;
    for _ in 0..NS_STEPS {
        let gram = row_gram(&x, work_rows, work_cols, gram_len);
        let ax = multiply(&gram, work_rows, work_rows, &x, work_cols);
        let a2x = multiply(&gram, work_rows, work_rows, &ax, work_cols);
        for i in 0..x.len() {
            x[i] = NS_A * x[i] + NS_B * ax[i] + NS_C * a2x[i];
            if !x[i].is_finite() {
                return Err(Error::InvalidConfig(
                    "Muon Newton-Schulz iteration became non-finite".to_owned(),
                ));
            }
        }
    }
    Ok(if transposed {
        transpose(&x, work_rows, work_cols)
    } else {
        x
    })
}

/// One computed Muon matrix direction. Computing it does not mutate parameters
/// or optimizer state, allowing a hybrid AdamW/Muon layer update to stage all
/// matrix directions before applying AdamW to the non-Muon parameters.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MuonMatrixStep {
    rows: usize,
    cols: usize,
    direction: Vec<f32>,
    pub next_momentum: Vec<f32>,
}

impl MuonMatrixStep {
    /// Add the scaled Muon direction after a caller has applied decoupled
    /// weight decay to the matrix.
    pub(crate) fn add_direction(&self, weights: &mut [f32], learning_rate: f32) -> Result<()> {
        let len = checked_matrix_len(self.rows, self.cols)?;
        if weights.len() != len || self.direction.len() != len {
            return Err(Error::InvalidConfig(format!(
                "Muon direction/weights have lengths {}/{}, expected {len} for {}x{}",
                self.direction.len(),
                weights.len(),
                self.rows,
                self.cols
            )));
        }
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(Error::InvalidConfig(
                "Muon learning rate must be finite and non-negative".to_owned(),
            ));
        }
        let mut next_weights = Vec::with_capacity(len);
        for (&weight, &direction) in weights.iter().zip(&self.direction) {
            let next = weight + learning_rate * direction;
            if !next.is_finite() {
                return Err(Error::InvalidConfig(
                    "Muon parameter update became non-finite".to_owned(),
                ));
            }
            next_weights.push(next);
        }
        weights.copy_from_slice(&next_weights);
        Ok(())
    }
}

/// Compute one paper-derived Muon direction for a row-major logical matrix.
///
/// `descent_updates` are Darknet's accumulated *descent* directions (the
/// opposite sign from a mathematical gradient). They are averaged by `batch`
/// and unscaled before entering momentum.
#[cfg(test)]
pub(crate) fn compute_muon_matrix_step(
    descent_updates: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    loss_scale: f32,
    momentum_coefficient: f32,
    state: &MuonMatrixState,
) -> Result<MuonMatrixStep> {
    compute_muon_matrix_step_with_orthogonalizer(
        descent_updates,
        rows,
        cols,
        batch,
        loss_scale,
        momentum_coefficient,
        state,
        newton_schulz_orthogonalize,
    )
}

pub(crate) fn compute_muon_matrix_step_with_orthogonalizer(
    descent_updates: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
    loss_scale: f32,
    momentum_coefficient: f32,
    state: &MuonMatrixState,
    orthogonalize: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>>,
) -> Result<MuonMatrixStep> {
    let len = checked_matrix_len(rows, cols)?;
    if descent_updates.len() != len {
        return Err(Error::InvalidConfig(format!(
            "Muon updates have {} values, expected {len} for {rows}x{cols}",
            descent_updates.len()
        )));
    }
    if state.rows != rows || state.cols != cols || state.momentum.len() != len {
        return Err(Error::InvalidConfig(format!(
            "Muon momentum shape is {}x{} ({} values), expected {rows}x{cols} ({len} values)",
            state.rows,
            state.cols,
            state.momentum.len()
        )));
    }
    if !momentum_coefficient.is_finite() || !(0.0..1.0).contains(&momentum_coefficient) {
        return Err(Error::InvalidConfig(
            "Muon momentum coefficient must be finite and in [0, 1)".to_owned(),
        ));
    }
    if descent_updates.iter().any(|value| !value.is_finite())
        || state.momentum.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidConfig(
            "Muon updates and momentum must be finite".to_owned(),
        ));
    }

    let scale = if loss_scale.is_finite() && loss_scale > 0.0 {
        loss_scale
    } else {
        1.0
    };
    let divisor = batch.max(1) as f64 * f64::from(scale);
    let mut next_momentum = Vec::with_capacity(len);
    let mut lookahead = Vec::with_capacity(len);
    for (&old_momentum, &descent) in state.momentum.iter().zip(descent_updates.iter()) {
        let gradient = (f64::from(descent) / divisor) as f32;
        let momentum = momentum_coefficient * old_momentum + gradient;
        let nesterov_descent = momentum_coefficient * momentum + gradient;
        if !momentum.is_finite() || !nesterov_descent.is_finite() {
            return Err(Error::InvalidConfig(
                "Muon momentum update became non-finite".to_owned(),
            ));
        }
        next_momentum.push(momentum);
        lookahead.push(nesterov_descent);
    }

    let orthogonal = orthogonalize(&lookahead, rows, cols)?;
    let update_scale = MUON_SCALE * (rows.max(cols) as f32).sqrt();
    let direction = orthogonal
        .into_iter()
        .map(|value| value * update_scale)
        .collect::<Vec<_>>();
    if direction.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(
            "Muon scaled direction became non-finite".to_owned(),
        ));
    }
    Ok(MuonMatrixStep {
        rows,
        cols,
        direction,
        next_momentum,
    })
}

/// Apply one paper-derived Muon step directly, including decoupled weight
/// decay. Darknet hybrid runtime code uses `compute_muon_matrix_step` so it can
/// delegate matrix decay to the existing AdamW parameter update exactly once.
#[cfg(test)]
pub(crate) fn apply_muon_matrix(
    weights: &mut [f32],
    descent_updates: &mut [f32],
    rows: usize,
    cols: usize,
    learning_rate: f32,
    decay: f32,
    batch: usize,
    loss_scale: f32,
    momentum_coefficient: f32,
    state: &mut MuonMatrixState,
) -> Result<()> {
    let len = checked_matrix_len(rows, cols)?;
    if weights.len() != len || descent_updates.len() != len {
        return Err(Error::InvalidConfig(format!(
            "Muon weights/updates have lengths {}/{}, expected {len} for {rows}x{cols}",
            weights.len(),
            descent_updates.len()
        )));
    }
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(Error::InvalidConfig(
            "Muon learning rate must be finite and non-negative".to_owned(),
        ));
    }
    if !decay.is_finite() || decay < 0.0 {
        return Err(Error::InvalidConfig(
            "Muon weight decay must be finite and non-negative".to_owned(),
        ));
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(
            "Muon weights must be finite".to_owned(),
        ));
    }
    let step = compute_muon_matrix_step(
        descent_updates,
        rows,
        cols,
        batch,
        loss_scale,
        momentum_coefficient,
        state,
    )?;
    let decay_scale = 1.0 - learning_rate * decay;
    let mut next_weights = weights
        .iter()
        .map(|weight| weight * decay_scale)
        .collect::<Vec<_>>();
    if next_weights.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(
            "Muon decoupled weight decay became non-finite".to_owned(),
        ));
    }
    step.add_direction(&mut next_weights, learning_rate)?;
    weights.copy_from_slice(&next_weights);
    descent_updates.fill(0.0);
    state.momentum = step.next_momentum;
    Ok(())
}

fn checked_matrix_len(rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidConfig(
            "Muon matrices require nonzero rows and columns".to_owned(),
        ));
    }
    rows.checked_mul(cols)
        .ok_or_else(|| Error::InvalidConfig("Muon matrix size overflow".to_owned()))
}

fn transpose(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; matrix.len()];
    for row in 0..rows {
        for col in 0..cols {
            result[col * rows + row] = matrix[row * cols + col];
        }
    }
    result
}

fn row_gram(matrix: &[f32], rows: usize, cols: usize, len: usize) -> Vec<f32> {
    let mut gram = vec![0.0; len];
    for row in 0..rows {
        for other in 0..=row {
            let mut dot = 0.0;
            for col in 0..cols {
                dot += matrix[row * cols + col] * matrix[other * cols + col];
            }
            gram[row * rows + other] = dot;
            gram[other * rows + row] = dot;
        }
    }
    gram
}

/// Row-major matrix product `(rows × inner) * (inner × cols)`.
fn multiply(left: &[f32], rows: usize, inner: usize, right: &[f32], cols: usize) -> Vec<f32> {
    let mut result = vec![0.0; rows * cols];
    for row in 0..rows {
        for k in 0..inner {
            let factor = left[row * inner + k];
            let right_row = &right[k * cols..(k + 1) * cols];
            let output_row = &mut result[row * cols..(row + 1) * cols];
            for col in 0..cols {
                output_row[col] += factor * right_row[col];
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{name} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn muon_newton_schulz_matches_the_hand_computed_scalar_polynomial() {
        let actual = newton_schulz_orthogonalize(&[3.0], 1, 1).unwrap();
        assert_close("NS scalar", &actual, &[0.6964364], 2.0e-6);
        let negative = newton_schulz_orthogonalize(&[-3.0], 1, 1).unwrap();
        assert_close("negative NS scalar", &negative, &[-0.6964364], 2.0e-6);
    }

    #[test]
    fn muon_newton_schulz_is_transpose_equivariant_for_rectangular_matrices() {
        let matrix = vec![0.2, -0.4, 0.7, 1.1, -0.3, 0.5];
        let result = newton_schulz_orthogonalize(&matrix, 2, 3).unwrap();
        let transposed = transpose(&matrix, 2, 3);
        let transpose_result = newton_schulz_orthogonalize(&transposed, 3, 2).unwrap();
        assert_close(
            "transpose equivariance",
            &result,
            &transpose(&transpose_result, 3, 2),
            2.0e-6,
        );
    }

    #[test]
    fn muon_zero_matrix_stays_zero_and_bad_shapes_or_nonfinite_values_error() {
        assert_eq!(
            newton_schulz_orthogonalize(&[0.0; 6], 2, 3).unwrap(),
            vec![0.0; 6]
        );
        assert!(newton_schulz_orthogonalize(&[0.0; 5], 2, 3).is_err());
        assert!(newton_schulz_orthogonalize(&[f32::NAN], 1, 1).is_err());
        assert!(newton_schulz_orthogonalize(&[], 0, 2).is_err());
    }

    #[test]
    fn muon_step_uses_darknet_descent_sign_nesterov_batch_and_decay() {
        let mut weights = vec![2.0];
        let mut updates = vec![4.0];
        let mut state = MuonMatrixState::zeroed(1, 1).unwrap();
        apply_muon_matrix(
            &mut weights,
            &mut updates,
            1,
            1,
            0.1,
            0.05,
            2,
            2.0,
            0.95,
            &mut state,
        )
        .unwrap();
        assert_close("momentum", &state.momentum, &[1.0], 1.0e-6);
        assert_eq!(updates, vec![0.0]);
        // Nesterov input is 0.95*1 + 1 = 1.95. Its five-step scalar
        // Newton–Schulz factor is the same as the positive unit scalar.
        assert_close("weight", &weights, &[2.0039287], 2.0e-6);
    }

    #[test]
    fn muon_momentum_persists_across_steps_and_nesterov_uses_the_new_buffer() {
        let mut weights = vec![0.0];
        let mut state = MuonMatrixState::zeroed(1, 1).unwrap();
        let mut first_updates = vec![1.0];
        apply_muon_matrix(
            &mut weights,
            &mut first_updates,
            1,
            1,
            0.0,
            0.0,
            1,
            1.0,
            0.95,
            &mut state,
        )
        .unwrap();
        assert_close("first momentum", &state.momentum, &[1.0], 1.0e-6);

        let mut second_updates = vec![-0.5];
        apply_muon_matrix(
            &mut weights,
            &mut second_updates,
            1,
            1,
            0.1,
            0.0,
            1,
            1.0,
            0.95,
            &mut state,
        )
        .unwrap();
        assert_close("second momentum", &state.momentum, &[0.45], 1.0e-6);
        // Nesterov descent = .95*.45 - .5 = -.0725; scalar orthogonalization
        // preserves only its sign, then Muon's 0.2*sqrt(max(rows, cols)) scale.
        assert_close("second weight", &weights, &[-0.013928728], 2.0e-6);
    }

    #[test]
    fn muon_bad_update_is_transactional() {
        let mut weights = vec![1.0, 2.0];
        let mut updates = vec![f32::INFINITY, 0.0];
        let mut state = MuonMatrixState::zeroed(1, 2).unwrap();
        let old_state = state.clone();
        assert!(
            apply_muon_matrix(
                &mut weights,
                &mut updates,
                1,
                2,
                0.1,
                0.0,
                1,
                1.0,
                0.95,
                &mut state
            )
            .is_err()
        );
        assert_eq!(weights, vec![1.0, 2.0]);
        assert_eq!(updates, vec![f32::INFINITY, 0.0]);
        assert_eq!(state, old_state);
    }
}
