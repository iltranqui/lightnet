//! CPU references for the native Muon update and Newton–Schulz matrix primitives.

/// Update momentum and form the Nesterov lookahead from Darknet descent updates.
pub fn momentum_lookahead(
    updates: &[f32],
    momentum: &[f32],
    batch: usize,
    loss_scale: f32,
    coefficient: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if updates.len() != momentum.len() {
        return Err("Muon updates/momentum length mismatch".to_owned());
    }
    if !coefficient.is_finite() || !(0.0..1.0).contains(&coefficient) {
        return Err("Muon momentum coefficient must be finite and in [0, 1)".to_owned());
    }
    if updates
        .iter()
        .chain(momentum)
        .any(|value| !value.is_finite())
    {
        return Err("Muon updates and momentum must be finite".to_owned());
    }
    let scale = if loss_scale.is_finite() && loss_scale > 0.0 {
        loss_scale
    } else {
        1.0
    };
    let divisor = batch.max(1) as f64 * f64::from(scale);
    let mut next_momentum = Vec::with_capacity(updates.len());
    let mut lookahead = Vec::with_capacity(updates.len());
    for (&update, &previous) in updates.iter().zip(momentum) {
        let gradient = (f64::from(update) / divisor) as f32;
        let next = coefficient * previous + gradient;
        let nesterov = coefficient * next + gradient;
        if !next.is_finite() || !nesterov.is_finite() {
            return Err("Muon momentum update became non-finite".to_owned());
        }
        next_momentum.push(next);
        lookahead.push(nesterov);
    }
    Ok((next_momentum, lookahead))
}

/// Apply the Muon direction and decoupled AdamW decay to a matrix.
pub fn apply_update(
    weights: &[f32],
    direction: &[f32],
    learning_rate: f32,
    decay: f32,
    direction_scale: f32,
) -> Result<Vec<f32>, String> {
    if weights.len() != direction.len() {
        return Err("Muon weights/direction length mismatch".to_owned());
    }
    if !learning_rate.is_finite() || learning_rate < 0.0 || !decay.is_finite() || decay < 0.0 {
        return Err("Muon learning rate and decay must be finite and non-negative".to_owned());
    }
    if !direction_scale.is_finite() || direction_scale < 0.0 {
        return Err("Muon direction scale must be finite and non-negative".to_owned());
    }
    if weights
        .iter()
        .chain(direction)
        .any(|value| !value.is_finite())
    {
        return Err("Muon weights and direction must be finite".to_owned());
    }
    let decay_factor = 1.0 - learning_rate * decay;
    let mut output = Vec::with_capacity(weights.len());
    for (&weight, &direction) in weights.iter().zip(direction) {
        let next = decay_factor * weight + learning_rate * direction_scale * direction;
        if !next.is_finite() {
            return Err("Muon parameter update became non-finite".to_owned());
        }
        output.push(next);
    }
    Ok(output)
}

/// Five-iteration Newton–Schulz orthogonalization used by Muon.
pub fn orthogonalize(matrix: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| "Muon matrix extent overflows".to_owned())?;
    if rows == 0 || cols == 0 || matrix.len() != len {
        return Err(format!(
            "Muon expected a non-empty {rows}x{cols} matrix, got {} elements",
            matrix.len()
        ));
    }
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err("Muon matrix contains a non-finite value".to_owned());
    }
    let norm_squared = matrix
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    if !norm_squared.is_finite() {
        return Err("Muon matrix Frobenius norm is non-finite".to_owned());
    }
    if norm_squared == 0.0 {
        return Ok(vec![0.0; len]);
    }
    let norm = norm_squared.sqrt();
    let (mut x, small, large, transposed) = if rows > cols {
        let mut transposed = vec![0.0; len];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = matrix[r * cols + c];
            }
        }
        for value in &mut transposed {
            *value = (f64::from(*value) / norm) as f32;
        }
        (transposed, cols, rows, true)
    } else {
        (
            matrix
                .iter()
                .map(|&value| (f64::from(value) / norm) as f32)
                .collect(),
            rows,
            cols,
            false,
        )
    };

    let mut gram = vec![0.0; small * small];
    let mut ax = vec![0.0; small * large];
    let mut a2x = vec![0.0; small * large];
    for _ in 0..5 {
        for i in 0..small {
            for j in 0..small {
                let mut sum = 0.0;
                for k in 0..large {
                    sum += x[i * large + k] * x[j * large + k];
                }
                gram[i * small + j] = sum;
            }
        }
        for i in 0..small {
            for j in 0..large {
                let mut sum = 0.0;
                for k in 0..small {
                    sum += gram[i * small + k] * x[k * large + j];
                }
                ax[i * large + j] = sum;
            }
        }
        for i in 0..small {
            for j in 0..large {
                let mut sum = 0.0;
                for k in 0..small {
                    sum += gram[i * small + k] * ax[k * large + j];
                }
                a2x[i * large + j] = sum;
            }
        }
        for i in 0..x.len() {
            x[i] = 3.4445 * x[i] - 4.7750 * ax[i] + 2.0315 * a2x[i];
        }
    }
    if transposed {
        let mut output = vec![0.0; len];
        for r in 0..rows {
            for c in 0..cols {
                output[r * cols + c] = x[c * rows + r];
            }
        }
        Ok(output)
    } else {
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_update, momentum_lookahead, orthogonalize};

    #[test]
    fn muon_momentum_lookahead_and_decoupled_update_match_the_reference_rule() {
        let (momentum, lookahead) =
            momentum_lookahead(&[0.4, -0.2], &[0.1, 0.3], 2, 2.0, 0.9).unwrap();
        for (actual, expected) in momentum.iter().zip([0.19, 0.22]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in lookahead.iter().zip([0.271, 0.148]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let updated = apply_update(&[2.0, -1.0], &[0.5, -0.25], 0.1, 0.2, 0.4).unwrap();
        for (actual, expected) in updated.iter().zip([1.98, -0.99]) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn muon_momentum_and_update_reject_invalid_inputs() {
        assert!(momentum_lookahead(&[1.0], &[], 1, 1.0, 0.9).is_err());
        assert!(momentum_lookahead(&[f32::NAN], &[0.0], 1, 1.0, 0.9).is_err());
        assert!(momentum_lookahead(&[1.0], &[0.0], 1, 1.0, 1.0).is_err());
        assert!(apply_update(&[1.0], &[], 0.1, 0.0, 1.0).is_err());
        assert!(apply_update(&[1.0], &[0.0], -0.1, 0.0, 1.0).is_err());
    }

    #[test]
    fn muon_orthogonalize_handles_zero_and_non_finite_inputs() {
        assert_eq!(orthogonalize(&[0.0; 6], 2, 3).unwrap(), vec![0.0; 6]);
        assert!(orthogonalize(&[0.0; 5], 2, 3).is_err());
        assert!(orthogonalize(&[f32::NAN], 1, 1).is_err());
    }

    #[test]
    fn muon_orthogonalize_is_transpose_equivariant() {
        let matrix = [0.2, -0.7, 1.1, 0.8, 0.4, -0.3];
        let transposed = [0.2, 0.8, -0.7, 0.4, 1.1, -0.3];
        let wide = orthogonalize(&matrix, 2, 3).unwrap();
        let tall = orthogonalize(&transposed, 3, 2).unwrap();
        for r in 0..2 {
            for c in 0..3 {
                assert!((wide[r * 3 + c] - tall[c * 2 + r]).abs() < 1e-6);
            }
        }
    }
}
