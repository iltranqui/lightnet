//! CPU reference for the native Muon Newton–Schulz matrix primitive.

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
    use super::orthogonalize;

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
