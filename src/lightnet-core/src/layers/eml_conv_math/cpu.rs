//! Pure elementwise math for the two convolution branches of
//! `[eml_convolutional]` (see `convolutional_layer.cpp` and
//! `convolutional_kernels.cu`). Layer parsing, convolution execution, and
//! optimizer integration are deliberately handled by later port milestones.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EmlConvMathParams {
    pub clamp: f32,
    pub eps: f32,
    pub scale: f32,
}

impl EmlConvMathParams {
    /// Apply the C++ layer's defensive defaults for nonpositive `eml_clamp`
    /// and `eml_eps`. The cfg parser normally supplies 4 and 1e-6 directly.
    pub(crate) fn from_cpp_values(clamp: f32, eps: f32, scale: f32) -> Self {
        Self {
            clamp: if clamp > 0.0 { clamp } else { 4.0 },
            eps: if eps > 0.0 { eps } else { 1.0e-6 },
            scale,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EmlConvMathGrad {
    pub x_branch: Vec<f32>,
    pub self_branch: Vec<f32>,
    /// Present when the forward included an identity residual.
    pub residual: Option<Vec<f32>>,
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// EML branch combination:
/// `scale * (exp(clamp(x, -c, c)) - ln(softplus(y) + eps)) + residual`.
/// A supplied residual must have the same length as both branch outputs.
pub(crate) fn forward(
    x_branch: &[f32],
    self_branch: &[f32],
    residual: Option<&[f32]>,
    params: EmlConvMathParams,
) -> Result<Vec<f32>, &'static str> {
    if x_branch.len() != self_branch.len() {
        return Err("EML branch output lengths differ");
    }
    if residual.is_some_and(|values| values.len() != x_branch.len()) {
        return Err("EML residual length differs from branch outputs");
    }
    let mut output = Vec::with_capacity(x_branch.len());
    for (index, (&x, &y)) in x_branch.iter().zip(self_branch).enumerate() {
        let clamped_x = x.max(-params.clamp).min(params.clamp);
        let mut value = params.scale * (clamped_x.exp() - (softplus(y) + params.eps).ln());
        if let Some(residual) = residual {
            value += residual[index];
        }
        output.push(value);
    }
    Ok(output)
}

/// Exact adjoint of [`forward`] with the source-defined strict clamp gate.
/// At `x == +/- clamp`, the derivative is zero; the residual's derivative is
/// one. `delta` is the incoming output gradient (not pre-scaled).
pub(crate) fn backward(
    x_branch: &[f32],
    self_branch: &[f32],
    delta: &[f32],
    residual: Option<&[f32]>,
    params: EmlConvMathParams,
) -> Result<EmlConvMathGrad, &'static str> {
    let len = x_branch.len();
    if self_branch.len() != len || delta.len() != len {
        return Err("EML branch and delta lengths differ");
    }
    if residual.is_some_and(|values| values.len() != len) {
        return Err("EML residual length differs from branch outputs");
    }
    let mut x_grad = Vec::with_capacity(len);
    let mut self_grad = Vec::with_capacity(len);
    for ((&x, &y), &dout) in x_branch.iter().zip(self_branch).zip(delta) {
        let dx = if x > -params.clamp && x < params.clamp {
            x.max(-params.clamp).min(params.clamp).exp()
        } else {
            0.0
        };
        let dy = -sigmoid(y) / (softplus(y) + params.eps);
        let scaled_delta = dout * params.scale;
        x_grad.push(scaled_delta * dx);
        self_grad.push(scaled_delta * dy);
    }
    Ok(EmlConvMathGrad {
        x_branch: x_grad,
        self_branch: self_grad,
        residual: residual.map(|_| delta.to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(name: &str, got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{name}: got {got}, want {want}");
    }

    #[test]
    fn cpp_forward_and_branch_gradients_match_hand_values() {
        let params = EmlConvMathParams::from_cpp_values(2.0, 0.25, 0.5);
        let x = [0.0, 3.0, -3.0];
        let y = [0.0, 1.0, -1.0];
        let residual = [0.25, -0.5, 1.0];
        let output = forward(&x, &y, Some(&residual), params).unwrap();
        let expected = [
            0.5 * (1.0 - (std::f32::consts::LN_2 + 0.25).ln()) + 0.25,
            0.5 * (2.0f32.exp() - (1.0f32.exp().ln_1p() + 0.25).ln()) - 0.5,
            0.5 * ((-2.0f32).exp() - ((-1.0f32).exp().ln_1p() + 0.25).ln()) + 1.0,
        ];
        for (i, (&got, &want)) in output.iter().zip(&expected).enumerate() {
            near(&format!("forward[{i}]"), got, want, 2e-6);
        }

        let delta = [2.0, -1.0, 0.5];
        let grad = backward(&x, &y, &delta, Some(&residual), params).unwrap();
        near("dx interior", grad.x_branch[0], delta[0] * 0.5, 1e-7);
        assert_eq!(grad.x_branch[1], 0.0, "x beyond +clamp is gated");
        assert_eq!(grad.x_branch[2], 0.0, "x beyond -clamp is gated");
        near(
            "dy",
            grad.self_branch[0],
            -delta[0] * 0.5 * 0.5 / (std::f32::consts::LN_2 + 0.25),
            2e-6,
        );
        assert_eq!(grad.residual.as_deref(), Some(delta.as_slice()));
    }

    #[test]
    fn exact_branch_gradients_match_finite_differences() {
        let params = EmlConvMathParams::from_cpp_values(2.5, 1e-4, 0.7);
        let xs = [-2.0, -0.7, 0.3, 1.6];
        let ys = [-30.0, -1.2, 0.6, 25.0];
        let deltas = [0.4, -0.3, 0.8, -0.5];
        let residual = [0.2, 0.1, -0.6, 0.7];
        let analytic = backward(&xs, &ys, &deltas, Some(&residual), params).unwrap();
        let h = 1e-3;
        for i in 0..xs.len() {
            let objective = |x: &[f32], y: &[f32], r: &[f32]| {
                forward(x, y, Some(r), params)
                    .unwrap()
                    .iter()
                    .zip(deltas)
                    .map(|(out, delta)| out * delta)
                    .sum::<f32>()
            };
            let mut plus_x = xs;
            let mut minus_x = xs;
            plus_x[i] += h;
            minus_x[i] -= h;
            let fd_x = (objective(&plus_x, &ys, &residual) - objective(&minus_x, &ys, &residual))
                / (2.0 * h);
            near(
                &format!("finite difference dx[{i}]"),
                fd_x,
                analytic.x_branch[i],
                2e-3,
            );

            let mut plus_y = ys;
            let mut minus_y = ys;
            plus_y[i] += h;
            minus_y[i] -= h;
            let fd_y = (objective(&xs, &plus_y, &residual) - objective(&xs, &minus_y, &residual))
                / (2.0 * h);
            near(
                &format!("finite difference dy[{i}]"),
                fd_y,
                analytic.self_branch[i],
                2e-3,
            );

            let mut plus_r = residual;
            let mut minus_r = residual;
            plus_r[i] += h;
            minus_r[i] -= h;
            let fd_r = (objective(&xs, &ys, &plus_r) - objective(&xs, &ys, &minus_r)) / (2.0 * h);
            near(
                &format!("finite difference residual[{i}]"),
                fd_r,
                deltas[i],
                2e-3,
            );
        }
    }

    #[test]
    fn cpp_parameter_fallbacks_and_shapes_are_explicit() {
        let defaults = EmlConvMathParams::from_cpp_values(0.0, -1.0, 0.1);
        assert_eq!(defaults.clamp, 4.0);
        assert_eq!(defaults.eps, 1e-6);
        assert_eq!(defaults.scale, 0.1);
        assert_eq!(
            forward(&[1.0], &[], None, defaults),
            Err("EML branch output lengths differ")
        );
        assert_eq!(
            forward(&[1.0], &[1.0], Some(&[]), defaults),
            Err("EML residual length differs from branch outputs")
        );
        assert_eq!(
            backward(&[1.0], &[1.0], &[], None, defaults),
            Err("EML branch and delta lengths differ")
        );
    }

    #[test]
    fn strict_clamp_boundaries_have_zero_x_gradient() {
        let params = EmlConvMathParams::from_cpp_values(2.0, 1e-6, 1.0);
        let grad = backward(&[-2.0, 2.0], &[0.0, 0.0], &[1.0, 1.0], None, params).unwrap();
        assert_eq!(grad.x_branch, [0.0, 0.0]);
    }
}
