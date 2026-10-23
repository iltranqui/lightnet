//! Scalar activations missing from the original Rust set (see tasks/rust/layers.md Task 1).

// C++ literals are f32 (`0.797885f`, `0.035677f`); the derivative below is the exact
// derivative of THIS forward, so it uses the same constants.
const GELU_A: f32 = 0.797_885;
const GELU_B: f32 = 0.035_677;
const SELU_SCALE: f32 = 1.0507;
const SELU_ALPHA: f32 = 1.6732;

pub(crate) fn loggy(x: f32) -> f32 {
    2.0 / (1.0 + (-x).exp()) - 1.0
}

pub(crate) fn loggy_grad(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    2.0 * sigmoid * (1.0 - sigmoid)
}

pub(crate) fn loggy_grad_from_output(output: f32) -> f32 {
    0.5 * (1.0 - output * output)
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_A * x + GELU_B * x * x * x).tanh())
}

pub(crate) fn gelu_grad(x: f32) -> f32 {
    let t = (GELU_A * x + GELU_B * x * x * x).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * (GELU_A + 3.0 * GELU_B * x * x)
}

pub(crate) fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

pub(crate) fn elu_grad(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { x.exp() }
}

pub(crate) fn selu(x: f32) -> f32 {
    if x >= 0.0 {
        SELU_SCALE * x
    } else {
        SELU_SCALE * SELU_ALPHA * (x.exp() - 1.0)
    }
}

pub(crate) fn selu_grad(x: f32) -> f32 {
    if x >= 0.0 {
        SELU_SCALE
    } else {
        SELU_SCALE * SELU_ALPHA * x.exp()
    }
}

pub(crate) fn relie(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 * x }
}

pub(crate) fn plse(x: f32) -> f32 {
    if x < -4.0 {
        0.01 * (x + 4.0)
    } else if x > 4.0 {
        0.01 * (x - 4.0) + 1.0
    } else {
        0.125 * x + 0.5
    }
}

pub(crate) fn plse_grad(x: f32) -> f32 {
    if x.abs() > 4.0 { 0.01 } else { 0.125 }
}

pub(crate) fn hardtan(x: f32) -> f32 {
    x.clamp(-1.0, 1.0)
}

pub(crate) fn hardtan_grad(x: f32) -> f32 {
    if x > -1.0 && x < 1.0 { 1.0 } else { 0.0 }
}

pub(crate) fn lhtan(x: f32) -> f32 {
    if x < 0.0 {
        0.001 * x
    } else if x > 1.0 {
        0.001 * (x - 1.0) + 1.0
    } else {
        x
    }
}

pub(crate) fn lhtan_grad(x: f32) -> f32 {
    if x > 0.0 && x < 1.0 { 1.0 } else { 0.001 }
}

pub(crate) fn stair(x: f32) -> f32 {
    let n = x.floor() as i32;
    if n % 2 == 0 {
        (x / 2.0).floor()
    } else {
        (x - n as f32) + (x / 2.0).floor()
    }
}

pub(crate) fn stair_grad(x: f32) -> f32 {
    if (x.floor() as i32).rem_euclid(2) == 1 {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn hard_mish(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else if x > -2.0 {
        x * x / 2.0 + x
    } else {
        0.0
    }
}

pub(crate) fn hard_mish_grad(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x > -2.0 {
        x + 1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(name: &str, got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{name}: got {got}, want {want}");
    }

    #[test]
    fn forward_values_match_cpp_formulas() {
        near("loggy(0)", loggy(0.0), 0.0, 0.0);
        near("loggy(1)", loggy(1.0), 0.462_117_17, 1e-7);
        near("loggy(-1)", loggy(-1.0), -0.462_117_17, 1e-7);
        near("gelu(0)", gelu(0.0), 0.0, 1e-7);
        near("gelu(1)", gelu(1.0), 0.841_192, 1e-4);
        near("gelu(-1)", gelu(-1.0), -0.158_808, 1e-4);
        near("elu(-1)", elu(-1.0), -0.632_120_6, 1e-6);
        near("elu(2)", elu(2.0), 2.0, 0.0);
        near("selu(1)", selu(1.0), 1.0507, 1e-6);
        near("selu(-1)", selu(-1.0), -1.111_3, 1e-3);
        near("relie(-2)", relie(-2.0), -0.02, 1e-7);
        near("relie(3)", relie(3.0), 3.0, 0.0);
        near("plse(-5)", plse(-5.0), -0.01, 1e-7);
        near("plse(0)", plse(0.0), 0.5, 1e-7);
        near("plse(5)", plse(5.0), 1.01, 1e-6);
        near("hardtan(-3)", hardtan(-3.0), -1.0, 0.0);
        near("hardtan(0.25)", hardtan(0.25), 0.25, 0.0);
        near("lhtan(-2)", lhtan(-2.0), -0.002, 1e-7);
        near("lhtan(0.5)", lhtan(0.5), 0.5, 0.0);
        near("lhtan(3)", lhtan(3.0), 1.002, 1e-6);
        // n = floor(x): even n -> floor(x/2); odd n -> (x-n) + floor(x/2)
        near("stair(0.5)", stair(0.5), 0.0, 1e-7);
        near("stair(1.5)", stair(1.5), 0.5, 1e-7);
        near("stair(2.5)", stair(2.5), 1.0, 1e-7);
        near("stair(3.5)", stair(3.5), 1.5, 1e-7);
        near("stair(-0.5)", stair(-0.5), -1.0 + 0.5, 1e-7); // n=-1 (odd): 0.5 + floor(-0.25) = -0.5
        near("hard_mish(-3)", hard_mish(-3.0), 0.0, 0.0);
        near("hard_mish(-1)", hard_mish(-1.0), -0.5, 1e-7);
        near("hard_mish(2)", hard_mish(2.0), 2.0, 0.0);
    }

    // Away from every kink/discontinuity of the functions (kinks at 0, +-1, -2, +-4, integers).
    const POINTS: [f32; 8] = [-5.3, -3.2, -1.7, -0.6, 0.4, 0.9, 2.3, 4.6];

    #[test]
    fn analytic_gradients_match_central_differences() {
        let h = 1e-2f32;
        let cases: [(&str, fn(f32) -> f32, fn(f32) -> f32); 9] = [
            ("loggy", loggy, loggy_grad),
            ("gelu", gelu, gelu_grad),
            ("elu", elu, elu_grad),
            ("selu", selu, selu_grad),
            ("plse", plse, plse_grad),
            ("hardtan", hardtan, hardtan_grad),
            ("lhtan", lhtan, lhtan_grad),
            ("stair", stair, stair_grad),
            ("hard_mish", hard_mish, hard_mish_grad),
        ];
        for (name, f, g) in cases {
            for x in POINTS {
                let fd = (f(x + h) - f(x - h)) / (2.0 * h);
                assert!(
                    (fd - g(x)).abs() <= 2e-3 + 2e-3 * fd.abs(),
                    "{name} x={x}: finite difference {fd}, analytic {}",
                    g(x)
                );
            }
        }
    }
}
