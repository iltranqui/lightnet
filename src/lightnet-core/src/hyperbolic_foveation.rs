//! Scalar geometry and NCHW sampling primitives for hyperbolic foveation.
//!
//! The source layer maps image coordinates in normalized `[-1, 1]` space,
//! builds an inverse sampling grid in `(x, y)` order, and applies that shared
//! grid independently to every channel plane and batch item. The CPU model,
//! runtime, detection, and training paths consume these independently testable
//! math helpers.

use crate::error::{Error, Result};
use crate::model::{HfInverseMethod, HfRadiusMode};

const MIN_RADIUS: f32 = 0.1;
const MAX_RADIUS: f32 = 2.0;
const RADIUS_SMOOTH_K: f32 = 8.0;
const EPSILON: f32 = 1.0e-6;
const MIN_JACOBIAN_DETERMINANT: f32 = 1.0e-8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HfParams {
    pub center_x: f32,
    pub center_y: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub strength: f32,
    pub alpha: f32,
    pub p_exp: f32,
    pub radius_mode: HfRadiusMode,
}

impl HfParams {
    #[inline]
    pub(crate) fn effective_radius(self) -> f32 {
        match self.radius_mode {
            HfRadiusMode::Isotropic => clamp_radius(self.radius_x),
            HfRadiusMode::Anisotropic => smooth_max_radius(self.radius_x, self.radius_y),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HfPointEval {
    pub point: [f32; 2],
    /// Row-major Jacobian `[[dPhi_x/dx, dPhi_x/dy], [dPhi_y/dx, dPhi_y/dy]]`.
    pub jacobian: [[f32; 2]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HfInverseResult {
    pub point: [f32; 2],
    /// False when Newton encountered a singular Jacobian and used the source's
    /// damped residual fallback for at least one iteration.
    pub nonsingular: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HfInverseParamJacobian {
    pub point: [f32; 2],
    /// Rows are inverse-point coordinates; columns are `[cx, cy, rx, ry, strength]`.
    pub params: [[f32; 5]; 2],
    pub nonsingular: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BilinearSample {
    pub value: f32,
    pub grad_x: f32,
    pub grad_y: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WarpGradients {
    /// Adjoint in the same NCHW order as the image input.
    pub input: Vec<f32>,
    /// Adjoint of the shared per-pixel grid, stored as `(src_x, src_y)`.
    pub grid: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PredictorForwardCache {
    pub pooled: Vec<f32>,
    pub hidden_preactivation: Vec<f32>,
    pub hidden: Vec<f32>,
    pub raw_output: Vec<f32>,
    pub constrained_params: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PredictorGradients {
    pub source: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[hyperbolic_foveation]: {}", message.into()))
}

fn checked_numel(dims: &[usize], name: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| invalid(format!("{name} element count overflows")))
    })
}

fn validate_nchw(batch: usize, channels: usize, height: usize, width: usize) -> Result<usize> {
    if batch == 0 || channels == 0 || height == 0 || width == 0 {
        return Err(invalid("NCHW dimensions must be positive"));
    }
    checked_numel(&[batch, channels, height, width], "NCHW tensor")
}

#[inline]
fn clamp_poincare(value: f32) -> f32 {
    value.clamp(-1.0, 1.0)
}

#[inline]
fn clamp_radius(value: f32) -> f32 {
    value.clamp(MIN_RADIUS, MAX_RADIUS)
}

/// Source-compatible k=8 soft maximum for anisotropic radii.
#[inline]
pub(crate) fn smooth_max_radius(radius_x: f32, radius_y: f32) -> f32 {
    let rx = clamp_radius(radius_x);
    let ry = clamp_radius(radius_y);
    clamp_radius((rx.powf(RADIUS_SMOOTH_K) + ry.powf(RADIUS_SMOOTH_K)).powf(1.0 / RADIUS_SMOOTH_K))
}

/// Evaluate the source radial map and its analytic spatial Jacobian.
///
/// Config construction is responsible for rejecting non-finite or invalid
/// parameters. This helper preserves the C++ epsilon floors used in the map.
pub(crate) fn phi_eval(point: [f32; 2], params: HfParams) -> HfPointEval {
    let [x, y] = point;
    let dx = x - params.center_x;
    let dy = y - params.center_y;
    let r = (dx * dx + dy * dy).sqrt();
    let radius = params.effective_radius().max(1.0e-4);
    let alpha = params.alpha.max(1.0e-4);
    let p_exp = params.p_exp.max(1.0e-4);
    let inside = r < radius;

    let (u, du_dr) = if r > EPSILON {
        let ar = alpha * r;
        let tanh = ar.tanh();
        let sech2 = 1.0 - tanh * tanh;
        (tanh / r, (alpha * sech2 * r - tanh) / (r * r))
    } else {
        (alpha, 0.0)
    };

    let (weight, dw_dr) = if inside {
        let base = (1.0 - r / radius).max(0.0);
        let weight = base.powf(p_exp);
        let dw_dr = if r > EPSILON {
            -(p_exp / radius) * base.powf((p_exp - 1.0).max(0.0))
        } else {
            0.0
        };
        (weight, dw_dr)
    } else {
        (0.0, 0.0)
    };

    let gain = 1.0 + params.strength * weight * (u - 1.0);
    let dg_dr = params.strength * (dw_dr * (u - 1.0) + weight * du_dr);
    let mut jacobian = [[gain, 0.0], [0.0, gain]];
    if r > EPSILON {
        let d_gain_over_r = dg_dr / r;
        jacobian[0][0] += d_gain_over_r * dx * dx;
        jacobian[0][1] += d_gain_over_r * dx * dy;
        jacobian[1][0] += d_gain_over_r * dy * dx;
        jacobian[1][1] += d_gain_over_r * dy * dy;
    }

    HfPointEval {
        point: [params.center_x + gain * dx, params.center_y + gain * dy],
        jacobian,
    }
}

/// Invert one foveated point with the configured finite source iteration.
///
/// Newton uses the analytic Jacobian, with the source's damped residual step
/// when its determinant is singular. Fixed-point uses the residual step on all
/// iterations. Both methods clamp each iterate to the Poincare square.
pub(crate) fn inverse_point(
    query: [f32; 2],
    params: HfParams,
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> HfInverseResult {
    let mut point = [clamp_poincare(query[0]), clamp_poincare(query[1])];
    let mut nonsingular = true;
    for _ in 0..iterations {
        let eval = phi_eval(point, params);
        let residual = [query[0] - eval.point[0], query[1] - eval.point[1]];
        if method == HfInverseMethod::FixedPoint {
            point[0] = clamp_poincare(point[0] + eta * residual[0]);
            point[1] = clamp_poincare(point[1] + eta * residual[1]);
            continue;
        }

        let [[j00, j01], [j10, j11]] = eval.jacobian;
        let determinant = j00 * j11 - j01 * j10;
        if determinant.abs() < MIN_JACOBIAN_DETERMINANT {
            nonsingular = false;
            point[0] = clamp_poincare(point[0] + eta * residual[0]);
            point[1] = clamp_poincare(point[1] + eta * residual[1]);
            continue;
        }
        let step_x = (j11 * residual[0] - j01 * residual[1]) / determinant;
        let step_y = (-j10 * residual[0] + j00 * residual[1]) / determinant;
        point[0] = clamp_poincare(point[0] + eta * step_x);
        point[1] = clamp_poincare(point[1] + eta * step_y);
    }
    HfInverseResult { point, nonsingular }
}

/// First-order dual number used to differentiate the finite inverse iteration.
///
/// Newton updates depend on the analytic spatial Jacobian, so differentiating
/// the configured finite sequence must also differentiate that Jacobian as
/// its point and foveation parameters change. Five tangent lanes cover the
/// trainable `[cx, cy, rx, ry, strength]`; alpha and the exponent are cfg values.
#[derive(Clone, Copy, Debug)]
struct Dual5 {
    value: f32,
    grad: [f32; 5],
}

impl Dual5 {
    #[inline]
    fn constant(value: f32) -> Self {
        Self {
            value,
            grad: [0.0; 5],
        }
    }

    #[inline]
    fn variable(value: f32, lane: usize) -> Self {
        let mut result = Self::constant(value);
        result.grad[lane] = 1.0;
        result
    }

    #[inline]
    fn sqrt(self) -> Self {
        let value = self.value.sqrt();
        let grad = if value <= EPSILON {
            [0.0; 5]
        } else {
            let scale = 0.5 / value;
            self.grad.map(|gradient| gradient * scale)
        };
        Self { value, grad }
    }

    #[inline]
    fn tanh(self) -> Self {
        let value = self.value.tanh();
        let scale = 1.0 - value * value;
        Self {
            value,
            grad: self.grad.map(|gradient| gradient * scale),
        }
    }

    #[inline]
    fn powf(self, exponent: f32) -> Self {
        let value = self.value.powf(exponent);
        let scale = exponent * self.value.powf(exponent - 1.0);
        Self {
            value,
            grad: self.grad.map(|gradient| gradient * scale),
        }
    }

    #[inline]
    fn clamp(self, minimum: f32, maximum: f32) -> Self {
        if self.value <= minimum {
            Self::constant(minimum)
        } else if self.value >= maximum {
            Self::constant(maximum)
        } else {
            self
        }
    }
}

impl std::ops::Add for Dual5 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value + rhs.value,
            grad: std::array::from_fn(|lane| self.grad[lane] + rhs.grad[lane]),
        }
    }
}

impl std::ops::Sub for Dual5 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value - rhs.value,
            grad: std::array::from_fn(|lane| self.grad[lane] - rhs.grad[lane]),
        }
    }
}

impl std::ops::Mul for Dual5 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            grad: std::array::from_fn(|lane| {
                self.grad[lane] * rhs.value + self.value * rhs.grad[lane]
            }),
        }
    }
}

impl std::ops::Div for Dual5 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            grad: std::array::from_fn(|lane| {
                (self.grad[lane] * rhs.value - self.value * rhs.grad[lane]) / denominator
            }),
        }
    }
}

impl std::ops::Neg for Dual5 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            grad: self.grad.map(|gradient| -gradient),
        }
    }
}

#[derive(Clone, Copy)]
struct HfDualEval {
    point: [Dual5; 2],
    jacobian: [[Dual5; 2]; 2],
}

fn phi_eval_dual(
    point: [Dual5; 2],
    params: [Dual5; 5],
    alpha: f32,
    p_exp: f32,
    mode: HfRadiusMode,
) -> HfDualEval {
    let [center_x, center_y, radius_x, radius_y, strength] = params;
    let dx = point[0] - center_x;
    let dy = point[1] - center_y;
    let radius = match mode {
        HfRadiusMode::Isotropic => radius_x.clamp(MIN_RADIUS, MAX_RADIUS),
        HfRadiusMode::Anisotropic => {
            let rx = radius_x.clamp(MIN_RADIUS, MAX_RADIUS);
            let ry = radius_y.clamp(MIN_RADIUS, MAX_RADIUS);
            (rx.powf(RADIUS_SMOOTH_K) + ry.powf(RADIUS_SMOOTH_K))
                .powf(1.0 / RADIUS_SMOOTH_K)
                .clamp(MIN_RADIUS, MAX_RADIUS)
        }
    };
    let radius = if radius.value < 1.0e-4 {
        Dual5::constant(1.0e-4)
    } else {
        radius
    };
    let alpha = alpha.max(1.0e-4);
    let p_exp = p_exp.max(1.0e-4);
    let r = (dx * dx + dy * dy).sqrt();
    let inside = r.value < radius.value;

    let (u, du_dr) = if r.value > EPSILON {
        let ar = Dual5::constant(alpha) * r;
        let tanh = ar.tanh();
        let sech2 = Dual5::constant(1.0) - tanh * tanh;
        (
            tanh / r,
            (Dual5::constant(alpha) * sech2 * r - tanh) / (r * r),
        )
    } else {
        (Dual5::constant(alpha), Dual5::constant(0.0))
    };
    let (weight, dw_dr) = if inside {
        let unbounded_base = Dual5::constant(1.0) - r / radius;
        let base = if unbounded_base.value <= 0.0 {
            Dual5::constant(0.0)
        } else {
            unbounded_base
        };
        let weight = base.powf(p_exp);
        let dw_dr = if r.value > EPSILON {
            -(Dual5::constant(p_exp) / radius) * base.powf((p_exp - 1.0).max(0.0))
        } else {
            Dual5::constant(0.0)
        };
        (weight, dw_dr)
    } else {
        (Dual5::constant(0.0), Dual5::constant(0.0))
    };

    let gain = Dual5::constant(1.0) + strength * weight * (u - Dual5::constant(1.0));
    let dg_dr = strength * (dw_dr * (u - Dual5::constant(1.0)) + weight * du_dr);
    let mut jacobian = [[gain, Dual5::constant(0.0)], [Dual5::constant(0.0), gain]];
    if r.value > EPSILON {
        let d_gain_over_r = dg_dr / r;
        jacobian[0][0] = jacobian[0][0] + d_gain_over_r * dx * dx;
        jacobian[0][1] = d_gain_over_r * dx * dy;
        jacobian[1][0] = d_gain_over_r * dy * dx;
        jacobian[1][1] = jacobian[1][1] + d_gain_over_r * dy * dy;
    }

    HfDualEval {
        point: [center_x + gain * dx, center_y + gain * dy],
        jacobian,
    }
}

/// Differentiate the exact configured finite inverse iterations w.r.t. the
/// five trainable layer parameters. Clamps and branch choices match the scalar
/// forward; comparisons at their boundaries are nondifferentiable by contract.
pub(crate) fn inverse_point_param_jacobian(
    query: [f32; 2],
    params: HfParams,
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> HfInverseParamJacobian {
    let parameter_duals = [
        Dual5::variable(params.center_x, 0),
        Dual5::variable(params.center_y, 1),
        Dual5::variable(params.radius_x, 2),
        Dual5::variable(params.radius_y, 3),
        Dual5::variable(params.strength, 4),
    ];
    let query = query.map(Dual5::constant);
    let mut point = [query[0].clamp(-1.0, 1.0), query[1].clamp(-1.0, 1.0)];
    let mut nonsingular = true;
    for _ in 0..iterations {
        let eval = phi_eval_dual(
            point,
            parameter_duals,
            params.alpha,
            params.p_exp,
            params.radius_mode,
        );
        let residual = [query[0] - eval.point[0], query[1] - eval.point[1]];
        if method == HfInverseMethod::FixedPoint {
            point[0] = (point[0] + Dual5::constant(eta) * residual[0]).clamp(-1.0, 1.0);
            point[1] = (point[1] + Dual5::constant(eta) * residual[1]).clamp(-1.0, 1.0);
            continue;
        }
        let [[j00, j01], [j10, j11]] = eval.jacobian;
        let determinant = j00 * j11 - j01 * j10;
        if determinant.value.abs() < MIN_JACOBIAN_DETERMINANT {
            nonsingular = false;
            point[0] = (point[0] + Dual5::constant(eta) * residual[0]).clamp(-1.0, 1.0);
            point[1] = (point[1] + Dual5::constant(eta) * residual[1]).clamp(-1.0, 1.0);
            continue;
        }
        let step_x = (j11 * residual[0] - j01 * residual[1]) / determinant;
        let step_y = (-(j10 * residual[0]) + j00 * residual[1]) / determinant;
        point[0] = (point[0] + Dual5::constant(eta) * step_x).clamp(-1.0, 1.0);
        point[1] = (point[1] + Dual5::constant(eta) * step_y).clamp(-1.0, 1.0);
    }
    HfInverseParamJacobian {
        point: [point[0].value, point[1].value],
        params: [point[0].grad, point[1].grad],
        nonsingular,
    }
}

/// Bilinearly sample one channel plane with zero padding outside its bounds.
/// `src_x` is the width coordinate and `src_y` is the height coordinate.
pub(crate) fn bilinear_sample_plane(
    plane: &[f32],
    height: usize,
    width: usize,
    src_x: f32,
    src_y: f32,
) -> Result<BilinearSample> {
    let expected = validate_nchw(1, 1, height, width)?;
    if plane.len() != expected {
        return Err(invalid(format!(
            "channel plane has {} values, expected {expected}",
            plane.len()
        )));
    }
    if !src_x.is_finite() || !src_y.is_finite() {
        return Err(invalid("sampling coordinates must be finite"));
    }

    let x0 = src_x.floor() as isize;
    let y0 = src_y.floor() as isize;
    let dx = src_x - x0 as f32;
    let dy = src_y - y0 as f32;
    let sample = |y: isize, x: isize| -> f32 {
        if y < 0 || x < 0 || y >= height as isize || x >= width as isize {
            0.0
        } else {
            plane[y as usize * width + x as usize]
        }
    };

    let v00 = sample(y0, x0);
    let v01 = sample(y0, x0 + 1);
    let v10 = sample(y0 + 1, x0);
    let v11 = sample(y0 + 1, x0 + 1);
    Ok(BilinearSample {
        value: (1.0 - dy) * (1.0 - dx) * v00
            + (1.0 - dy) * dx * v01
            + dy * (1.0 - dx) * v10
            + dy * dx * v11,
        grad_x: (1.0 - dy) * (v01 - v00) + dy * (v11 - v10),
        grad_y: (1.0 - dx) * (v10 - v00) + dx * (v11 - v01),
    })
}

/// Sample a shared `(src_x, src_y)` grid independently over an NCHW tensor.
pub(crate) fn warp_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    grid: &[f32],
) -> Result<Vec<f32>> {
    let count = validate_nchw(batch, channels, height, width)?;
    let grid_count = checked_numel(&[batch, height, width, 2], "sampling grid")?;
    if input.len() != count {
        return Err(invalid(format!(
            "input has {} values, expected {count}",
            input.len()
        )));
    }
    if grid.len() != grid_count {
        return Err(invalid(format!(
            "sampling grid has {} values, expected {grid_count}",
            grid.len()
        )));
    }

    let mut output = vec![0.0; count];
    for n in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let grid_index = ((n * height + y) * width + x) * 2;
                let src_x = grid[grid_index];
                let src_y = grid[grid_index + 1];
                for c in 0..channels {
                    let plane_start = (n * channels + c) * height * width;
                    let out_index = plane_start + y * width + x;
                    let sampled = bilinear_sample_plane(
                        &input[plane_start..plane_start + height * width],
                        height,
                        width,
                        src_x,
                        src_y,
                    )?;
                    output[out_index] = sampled.value;
                }
            }
        }
    }
    Ok(output)
}

/// Adjoint of [`warp_nchw`] for image values and shared sampling coordinates.
pub(crate) fn warp_nchw_backward(
    input: &[f32],
    output_seed: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    grid: &[f32],
) -> Result<WarpGradients> {
    let count = validate_nchw(batch, channels, height, width)?;
    let grid_count = checked_numel(&[batch, height, width, 2], "sampling grid")?;
    if input.len() != count || output_seed.len() != count {
        return Err(invalid(format!(
            "input and output seed must each have {count} values"
        )));
    }
    if grid.len() != grid_count {
        return Err(invalid(format!(
            "sampling grid has {} values, expected {grid_count}",
            grid.len()
        )));
    }

    let mut input_grad = vec![0.0; count];
    let mut grid_grad = vec![0.0; grid_count];
    for n in 0..batch {
        for y in 0..height {
            for x in 0..width {
                let grid_index = ((n * height + y) * width + x) * 2;
                let src_x = grid[grid_index];
                let src_y = grid[grid_index + 1];
                for c in 0..channels {
                    let plane_start = (n * channels + c) * height * width;
                    let output_index = plane_start + y * width + x;
                    let seed = output_seed[output_index];
                    let plane = &input[plane_start..plane_start + height * width];
                    let sample = bilinear_sample_plane(plane, height, width, src_x, src_y)?;
                    grid_grad[grid_index] += seed * sample.grad_x;
                    grid_grad[grid_index + 1] += seed * sample.grad_y;

                    let x0 = src_x.floor() as isize;
                    let y0 = src_y.floor() as isize;
                    let dx = src_x - x0 as f32;
                    let dy = src_y - y0 as f32;
                    let mut scatter = |row: isize, col: isize, weight: f32| {
                        if row >= 0 && col >= 0 && row < height as isize && col < width as isize {
                            input_grad[plane_start + row as usize * width + col as usize] +=
                                seed * weight;
                        }
                    };
                    scatter(y0, x0, (1.0 - dy) * (1.0 - dx));
                    scatter(y0, x0 + 1, (1.0 - dy) * dx);
                    scatter(y0 + 1, x0, dy * (1.0 - dx));
                    scatter(y0 + 1, x0 + 1, dy * dx);
                }
            }
        }
    }
    Ok(WarpGradients {
        input: input_grad,
        grid: grid_grad,
    })
}

/// C++ `avgpool_downsample` layout and tile semantics for the predictor input.
pub(crate) fn average_pool_downsample_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    factor: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let count = validate_nchw(batch, channels, height, width)?;
    if input.len() != count {
        return Err(invalid(format!(
            "input has {} values, expected {count}",
            input.len()
        )));
    }
    if factor == 0 {
        return Err(invalid("downsample factor must be positive"));
    }

    let out_h = (height / factor).max(1);
    let out_w = (width / factor).max(1);
    let output_count = checked_numel(&[batch, channels, out_h, out_w], "pooled tensor")?;
    let mut output = vec![0.0; output_count];
    for n in 0..batch {
        for c in 0..channels {
            let input_start = (n * channels + c) * height * width;
            let output_start = (n * channels + c) * out_h * out_w;
            for dy in 0..out_h {
                for dx in 0..out_w {
                    let mut sum = 0.0;
                    let mut samples = 0usize;
                    for ky in 0..factor {
                        let y = dy * factor + ky;
                        for kx in 0..factor {
                            let x = dx * factor + kx;
                            if y < height && x < width {
                                sum += input[input_start + y * width + x];
                                samples += 1;
                            }
                        }
                    }
                    output[output_start + dy * out_w + dx] = if samples == 0 {
                        0.0
                    } else {
                        sum / samples as f32
                    };
                }
            }
        }
    }
    Ok((output, out_h, out_w))
}

pub(crate) fn predictor_forward_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    downsample: usize,
    hidden_count: usize,
    param_count: usize,
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> Result<PredictorForwardCache> {
    if hidden_count == 0 || param_count < 4 {
        return Err(invalid(
            "predictor requires hidden units and at least four parameters",
        ));
    }
    let (pooled, pool_h, pool_w) =
        average_pool_downsample_nchw(input, batch, channels, height, width, downsample)?;
    let input_dim = checked_numel(&[channels, pool_h, pool_w], "predictor input")?;
    let expected = [
        hidden_count.checked_mul(input_dim),
        Some(hidden_count),
        param_count.checked_mul(hidden_count),
        Some(param_count),
    ];
    let actual = [
        Some(w1.len()),
        Some(b1.len()),
        Some(w2.len()),
        Some(b2.len()),
    ];
    if expected != actual {
        return Err(invalid(format!(
            "predictor parameter lengths are {:?}, expected {:?}",
            actual.map(|value| value.unwrap_or(usize::MAX)),
            expected.map(|value| value.unwrap_or(usize::MAX))
        )));
    }

    let mut hidden_preactivation = vec![0.0; batch * hidden_count];
    let mut hidden = vec![0.0; batch * hidden_count];
    let mut raw_output = vec![0.0; batch * param_count];
    let mut constrained_params = vec![0.0; batch * param_count];
    for n in 0..batch {
        let pooled_start = n * input_dim;
        let hidden_start = n * hidden_count;
        for h in 0..hidden_count {
            let mut value = b1[h];
            let weight_start = h * input_dim;
            for i in 0..input_dim {
                value += w1[weight_start + i] * pooled[pooled_start + i];
            }
            hidden_preactivation[hidden_start + h] = value;
            hidden[hidden_start + h] = value.max(0.0);
        }
        let output_start = n * param_count;
        for p in 0..param_count {
            let mut value = b2[p];
            let weight_start = p * hidden_count;
            for h in 0..hidden_count {
                value += w2[weight_start + h] * hidden[hidden_start + h];
            }
            raw_output[output_start + p] = value;
            constrained_params[output_start + p] =
                constrain_predictor_output(value, p, param_count);
        }
    }
    Ok(PredictorForwardCache {
        pooled,
        hidden_preactivation,
        hidden,
        raw_output,
        constrained_params,
    })
}

fn constrain_predictor_output(value: f32, index: usize, param_count: usize) -> f32 {
    if index < 2 {
        0.9 * value.tanh()
    } else if index + 1 == param_count {
        let sigmoid = if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp = value.exp();
            exp / (1.0 + exp)
        };
        sigmoid
    } else {
        (2.0 * softplus(value) + MIN_RADIUS).clamp(MIN_RADIUS, MAX_RADIUS)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(crate) fn predictor_backward_nchw(
    input_shape: (usize, usize, usize, usize),
    downsample: usize,
    hidden_count: usize,
    param_count: usize,
    w1: &[f32],
    w2: &[f32],
    cache: &PredictorForwardCache,
    constrained_seed: &[f32],
) -> Result<PredictorGradients> {
    let (batch, channels, height, width) = input_shape;
    let input_count = validate_nchw(batch, channels, height, width)?;
    if downsample == 0 {
        return Err(invalid("predictor downsample factor must be positive"));
    }
    if hidden_count == 0 || param_count < 4 {
        return Err(invalid(
            "predictor backward requires hidden units and at least four parameters",
        ));
    }
    let out_h = (height / downsample).max(1);
    let out_w = (width / downsample).max(1);
    let input_dim = checked_numel(&[channels, out_h, out_w], "predictor input")?;
    let expected = [
        input_count,
        batch * input_dim,
        batch * hidden_count,
        batch * hidden_count,
        batch * param_count,
        batch * param_count,
        hidden_count.checked_mul(input_dim).unwrap_or(usize::MAX),
        hidden_count,
        param_count.checked_mul(hidden_count).unwrap_or(usize::MAX),
        batch * param_count,
    ];
    let actual = [
        input_count,
        cache.pooled.len(),
        cache.hidden_preactivation.len(),
        cache.hidden.len(),
        cache.raw_output.len(),
        cache.constrained_params.len(),
        w1.len(),
        hidden_count,
        w2.len(),
        constrained_seed.len(),
    ];
    if actual != expected {
        return Err(invalid(format!(
            "predictor backward layout mismatch: expected {expected:?}, got {actual:?}"
        )));
    }

    let mut gradients = PredictorGradients {
        source: vec![0.0; input_count],
        w1: vec![0.0; w1.len()],
        b1: vec![0.0; hidden_count],
        w2: vec![0.0; w2.len()],
        b2: vec![0.0; param_count],
    };
    let mut pooled_grad = vec![0.0; cache.pooled.len()];
    let mut hidden_grad = vec![0.0; cache.hidden.len()];
    let mut raw_grad = vec![0.0; cache.raw_output.len()];
    for n in 0..batch {
        let hidden_start = n * hidden_count;
        let output_start = n * param_count;
        for p in 0..param_count {
            let raw = cache.raw_output[output_start + p];
            let derivative = if p < 2 {
                let tanh = raw.tanh();
                0.9 * (1.0 - tanh * tanh)
            } else if p + 1 == param_count {
                let strength = sigmoid(raw);
                strength * (1.0 - strength)
            } else {
                let unconstrained = 2.0 * softplus(raw) + MIN_RADIUS;
                if unconstrained <= MIN_RADIUS || unconstrained >= MAX_RADIUS {
                    0.0
                } else {
                    2.0 * sigmoid(raw)
                }
            };
            raw_grad[output_start + p] = constrained_seed[output_start + p] * derivative;
            gradients.b2[p] += raw_grad[output_start + p];
            let weight_start = p * hidden_count;
            for h in 0..hidden_count {
                gradients.w2[weight_start + h] +=
                    raw_grad[output_start + p] * cache.hidden[hidden_start + h];
                hidden_grad[hidden_start + h] += raw_grad[output_start + p] * w2[weight_start + h];
            }
        }
    }

    for n in 0..batch {
        let pooled_start = n * input_dim;
        let hidden_start = n * hidden_count;
        for h in 0..hidden_count {
            let preactivation = cache.hidden_preactivation[hidden_start + h];
            let grad = if preactivation > 0.0 {
                hidden_grad[hidden_start + h]
            } else {
                0.0
            };
            gradients.b1[h] += grad;
            let weight_start = h * input_dim;
            for i in 0..input_dim {
                gradients.w1[weight_start + i] += grad * cache.pooled[pooled_start + i];
                pooled_grad[pooled_start + i] += grad * w1[weight_start + i];
            }
        }
    }

    for n in 0..batch {
        for c in 0..channels {
            let input_start = (n * channels + c) * height * width;
            let pooled_start = (n * channels + c) * out_h * out_w;
            for pool_y in 0..out_h {
                for pool_x in 0..out_w {
                    let mut count = 0usize;
                    for ky in 0..downsample {
                        let y = pool_y * downsample + ky;
                        for kx in 0..downsample {
                            let x = pool_x * downsample + kx;
                            if y < height && x < width {
                                count += 1;
                            }
                        }
                    }
                    if count == 0 {
                        continue;
                    }
                    let share = pooled_grad[pooled_start + pool_y * out_w + pool_x] / count as f32;
                    for ky in 0..downsample {
                        let y = pool_y * downsample + ky;
                        for kx in 0..downsample {
                            let x = pool_x * downsample + kx;
                            if y < height && x < width {
                                gradients.source[input_start + y * width + x] += share;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(gradients)
}

pub(crate) fn sampling_grid_nchw(
    params_per_batch: &[HfParams],
    batch: usize,
    height: usize,
    width: usize,
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> Result<Vec<f32>> {
    if params_per_batch.len() != batch || batch == 0 || height == 0 || width == 0 {
        return Err(invalid(
            "sampling grid dimensions or per-batch parameter count are invalid",
        ));
    }
    let count = checked_numel(&[batch, height, width, 2], "sampling grid")?;
    let mut grid = vec![0.0; count];
    let width_scale = width.saturating_sub(1) as f32;
    let height_scale = height.saturating_sub(1) as f32;
    for n in 0..batch {
        for y in 0..height {
            let qy = 2.0 * y as f32 / (height - 1).max(1) as f32 - 1.0;
            for x in 0..width {
                let qx = 2.0 * x as f32 / (width - 1).max(1) as f32 - 1.0;
                let inverse = inverse_point([qx, qy], params_per_batch[n], method, iterations, eta);
                let index = ((n * height + y) * width + x) * 2;
                grid[index] = (inverse.point[0] + 1.0) * 0.5 * width_scale;
                grid[index + 1] = (inverse.point[1] + 1.0) * 0.5 * height_scale;
            }
        }
    }
    Ok(grid)
}

pub(crate) fn transform_box_with_jacobian(
    bbox: [f32; 4],
    params: HfParams,
    inverse: bool,
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> [f32; 4] {
    if params.strength.abs() <= 1.0e-7 {
        return bbox;
    }
    let [unit_x, unit_y, width, height] = bbox;
    let point = [2.0 * unit_x - 1.0, 2.0 * unit_y - 1.0];
    let (center, width_tangent, height_tangent) = if inverse {
        let center = inverse_point(point, params, method, iterations, eta).point;
        let eval = phi_eval(center, params);
        let width_scale = (eval.jacobian[0][0].powi(2) + eval.jacobian[1][0].powi(2))
            .sqrt()
            .max(1.0e-6);
        let height_scale = (eval.jacobian[0][1].powi(2) + eval.jacobian[1][1].powi(2))
            .sqrt()
            .max(1.0e-6);
        (
            center,
            (2.0 * width) / width_scale,
            (2.0 * height) / height_scale,
        )
    } else {
        let eval = phi_eval(point, params);
        let width_scale = (eval.jacobian[0][0].powi(2) + eval.jacobian[1][0].powi(2)).sqrt();
        let height_scale = (eval.jacobian[0][1].powi(2) + eval.jacobian[1][1].powi(2)).sqrt();
        (
            eval.point,
            width_scale * (2.0 * width),
            height_scale * (2.0 * height),
        )
    };
    [
        ((center[0] + 1.0) * 0.5).clamp(0.0, 1.0),
        ((center[1] + 1.0) * 0.5).clamp(0.0, 1.0),
        (width_tangent * 0.5).max(1.0e-6),
        (height_tangent * 0.5).max(1.0e-6),
    ]
}

pub(crate) fn warp_parameter_gradients_from_grid_gradient(
    grid_gradient: &[f32],
    batch: usize,
    height: usize,
    width: usize,
    params_per_batch: &[HfParams],
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> Result<Vec<[f32; 5]>> {
    let grid_count = checked_numel(&[batch, height, width, 2], "sampling-grid gradient")?;
    if grid_gradient.len() != grid_count {
        return Err(invalid(format!(
            "sampling-grid gradient has {} values, expected {grid_count}", grid_gradient.len()
        )));
    }
    if params_per_batch.len() != batch {
        return Err(invalid("per-batch parameter count does not match warp batch"));
    }
    let mut param_gradients = vec![[0.0; 5]; batch];
    let width_scale = width.saturating_sub(1) as f32 * 0.5;
    let height_scale = height.saturating_sub(1) as f32 * 0.5;
    for n in 0..batch {
        for y in 0..height {
            let qy = 2.0 * y as f32 / (height - 1).max(1) as f32 - 1.0;
            for x in 0..width {
                let qx = 2.0 * x as f32 / (width - 1).max(1) as f32 - 1.0;
                let inverse = inverse_point_param_jacobian(
                    [qx, qy], params_per_batch[n], method, iterations, eta,
                );
                let grid_index = ((n * height + y) * width + x) * 2;
                let grid_seed = [
                    grid_gradient[grid_index] * width_scale,
                    grid_gradient[grid_index + 1] * height_scale,
                ];
                for parameter in 0..5 {
                    param_gradients[n][parameter] += grid_seed[0] * inverse.params[0][parameter]
                        + grid_seed[1] * inverse.params[1][parameter];
                }
            }
        }
    }
    Ok(param_gradients)
}

pub(crate) fn warp_backward_with_param_gradients(
    input: &[f32],
    output_seed: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    grid: &[f32],
    params_per_batch: &[HfParams],
    method: HfInverseMethod,
    iterations: usize,
    eta: f32,
) -> Result<(WarpGradients, Vec<[f32; 5]>)> {
    let warp = warp_nchw_backward(input, output_seed, batch, channels, height, width, grid)?;
    let param_gradients = warp_parameter_gradients_from_grid_gradient(
        &warp.grid, batch, height, width, params_per_batch, method, iterations, eta,
    )?;
    Ok((warp, param_gradients))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isotropic_params(strength: f32) -> HfParams {
        HfParams {
            center_x: 0.15,
            center_y: -0.10,
            radius_x: 0.9,
            radius_y: 0.9,
            strength,
            alpha: 2.0,
            p_exp: 2.0,
            radius_mode: HfRadiusMode::Isotropic,
        }
    }

    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn phi_center_and_zero_strength_match_hand_values() {
        let params = isotropic_params(0.35);
        let center = phi_eval([params.center_x, params.center_y], params);
        assert_eq!(center.point, [params.center_x, params.center_y]);
        assert!(center.jacobian[0][0].is_finite());
        assert_eq!(center.jacobian[0][1], 0.0);
        assert_eq!(center.jacobian[1][0], 0.0);

        let identity = isotropic_params(0.0);
        for point in [[-0.7, 0.4], [0.2, -0.3], [0.9, 0.8]] {
            let mapped = phi_eval(point, identity);
            assert!((mapped.point[0] - point[0]).abs() < 1.0e-7);
            assert!((mapped.point[1] - point[1]).abs() < 1.0e-7);
            assert_eq!(mapped.jacobian, [[1.0, 0.0], [0.0, 1.0]]);
        }
    }

    #[test]
    fn hyperbolic_foveation_axis_layout_keeps_batch_channel_hw_and_grid_xy() {
        // C++ forward_hyperbolic_foveation_layer builds map_x/map_y as [H,W],
        // stores grid[(b*H*W + y*W + x)*2 + {x,y}], then remaps each NCHW
        // channel plane independently with OpenCV. Hand fixture: B=2,C=2,H=2,W=3.
        // Coordinates use multiples of 1/32, so OpenCV's INTER_LINEAR lookup
        // table and the source-derived bilinear hand calculation agree exactly.
        let (batch, channels, height, width) = (2, 2, 2, 3);
        let input: Vec<f32> = (0..batch)
            .flat_map(|n| {
                (0..channels).flat_map(move |c| {
                    (0..height).flat_map(move |y| {
                        (0..width).map(move |x| (1000 * n + 100 * c + 10 * y + x) as f32)
                    })
                })
            })
            .collect();
        let mut grid = vec![0.0; batch * height * width * 2];
        for n in 0..batch {
            for y in 0..height {
                for x in 0..width {
                    let index = ((n * height + y) * width + x) * 2;
                    grid[index] = 0.25 + 0.5 * x as f32 + 0.125 * n as f32;
                    grid[index + 1] = 0.125 + 0.5 * y as f32 + 0.0625 * n as f32;
                }
            }
        }
        let actual = warp_nchw(&input, batch, channels, height, width, &grid).unwrap();
        let mut expected = Vec::new();
        for n in 0..batch {
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let index = ((n * height + y) * width + x) * 2;
                        let src_x = grid[index];
                        let src_y = grid[index + 1];
                        expected.push(1000.0 * n as f32 + 100.0 * c as f32 + 10.0 * src_y + src_x);
                    }
                }
            }
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 2e-4,
                "NCHW/grid XY axis at flat {index}: {actual} != {expected}"
            );
        }
        let n1_c0_y1_x2 = ((1 * channels) * height + 1) * width + 2;
        assert!(
            (actual[n1_c0_y1_x2] - 1008.25).abs() < 2e-5,
            "batch/channel/HW probe must sample x=1.375,y=0.6875, not swapped x/y"
        );
        assert_ne!(batch, 1);
        assert_ne!(width, height);
        assert_ne!(channels, height * width);
    }

    #[test]
    fn phi_jacobian_matches_coordinate_finite_differences() {
        let params = isotropic_params(0.35);
        let point = [0.28, 0.17];
        let analytic = phi_eval(point, params).jacobian;
        let epsilon = 1.0e-3;
        for axis in 0..2 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            let mapped_plus = phi_eval(plus, params).point;
            let mapped_minus = phi_eval(minus, params).point;
            for output_axis in 0..2 {
                let numeric =
                    (mapped_plus[output_axis] - mapped_minus[output_axis]) / (2.0 * epsilon);
                assert!(
                    (numeric - analytic[output_axis][axis]).abs() < 2.0e-3,
                    "axis={axis}, out={output_axis}, numeric={numeric}, analytic={}",
                    analytic[output_axis][axis]
                );
            }
        }
    }

    #[test]
    fn radial_warp_magnifies_inside_and_is_identity_outside_radius() {
        let params = HfParams {
            center_x: 0.0,
            center_y: 0.0,
            radius_x: 0.8,
            radius_y: 0.8,
            strength: 1.0,
            alpha: 2.0,
            p_exp: 2.0,
            radius_mode: HfRadiusMode::Isotropic,
        };
        let inside = phi_eval([0.3, 0.0], params).point;
        assert!(inside[0] > 0.3);
        assert_eq!(phi_eval([0.9, 0.0], params).point, [0.9, 0.0]);
    }

    #[test]
    fn anisotropic_radius_uses_the_source_smooth_max() {
        let radius = smooth_max_radius(0.3, 0.7);
        assert!(radius > 0.7);
        assert!(radius < 0.701);
        assert_eq!(smooth_max_radius(0.0, 9.0), 2.0);
    }

    #[test]
    fn newton_inverse_round_trips_a_noncentral_point() {
        let params = isotropic_params(0.35);
        let source = [0.38, -0.27];
        let warped = phi_eval(source, params).point;
        let inverse = inverse_point(warped, params, HfInverseMethod::Newton, 20, 1.0);
        assert!(inverse.nonsingular);
        assert!((inverse.point[0] - source[0]).abs() < 1.0e-5);
        assert!((inverse.point[1] - source[1]).abs() < 1.0e-5);
    }

    #[test]
    fn fixed_point_inverse_round_trips_with_damped_iterations() {
        let params = HfParams {
            center_x: 0.1,
            center_y: -0.15,
            radius_x: 0.8,
            radius_y: 0.8,
            strength: 0.3,
            alpha: 1.5,
            p_exp: 2.0,
            radius_mode: HfRadiusMode::Isotropic,
        };
        let source = [0.25, -0.20];
        let warped = phi_eval(source, params).point;
        let inverse = inverse_point(warped, params, HfInverseMethod::FixedPoint, 120, 0.5);
        assert!((inverse.point[0] - source[0]).abs() < 2.0e-3);
        assert!((inverse.point[1] - source[1]).abs() < 2.0e-3);
    }

    fn perturb_hf_param(params: &mut HfParams, lane: usize, delta: f32) {
        match lane {
            0 => params.center_x += delta,
            1 => params.center_y += delta,
            2 => params.radius_x += delta,
            3 => params.radius_y += delta,
            4 => params.strength += delta,
            _ => unreachable!("five hyperbolic-foveation parameter lanes"),
        }
    }

    #[test]
    fn inverse_param_jacobian_matches_finite_configured_iterations() {
        for radius_mode in [HfRadiusMode::Isotropic, HfRadiusMode::Anisotropic] {
            for method in [HfInverseMethod::Newton, HfInverseMethod::FixedPoint] {
                let params = HfParams {
                    center_x: 0.1,
                    center_y: -0.12,
                    radius_x: 0.83,
                    radius_y: 0.65,
                    strength: 0.27,
                    alpha: 1.7,
                    p_exp: 2.3,
                    radius_mode,
                };
                let query = [-0.35, 0.47];
                let (iterations, eta) = (4, 0.61);
                let analytic = inverse_point_param_jacobian(query, params, method, iterations, eta);
                let scalar = inverse_point(query, params, method, iterations, eta);
                assert!(scalar.nonsingular);
                assert!((analytic.point[0] - scalar.point[0]).abs() < 1.0e-6);
                assert!((analytic.point[1] - scalar.point[1]).abs() < 1.0e-6);
                assert_eq!(analytic.nonsingular, scalar.nonsingular);

                let epsilon = 1.0e-3;
                for lane in 0..5 {
                    let mut plus = params;
                    let mut minus = params;
                    perturb_hf_param(&mut plus, lane, epsilon);
                    perturb_hf_param(&mut minus, lane, -epsilon);
                    let plus = inverse_point(query, plus, method, iterations, eta).point;
                    let minus = inverse_point(query, minus, method, iterations, eta).point;
                    for axis in 0..2 {
                        let numeric = (plus[axis] - minus[axis]) / (2.0 * epsilon);
                        let expected = analytic.params[axis][lane];
                        assert!(
                            (numeric - expected).abs() < 7.0e-3,
                            "mode={radius_mode:?}, method={method:?}, lane={lane}, axis={axis}, numeric={numeric}, analytic={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn box_transform_uses_jacobian_columns_and_round_trips() {
        for radius_mode in [HfRadiusMode::Isotropic, HfRadiusMode::Anisotropic] {
            let params = HfParams {
                center_x: 0.12,
                center_y: -0.08,
                radius_x: 1.1,
                radius_y: 0.9,
                strength: 0.42,
                alpha: 1.8,
                p_exp: 2.4,
                radius_mode,
            };
            let box_in = [0.64, 0.43, 0.07, 0.11];
            let input_point = [2.0 * box_in[0] - 1.0, 2.0 * box_in[1] - 1.0];
            let eval = phi_eval(input_point, params);
            let expected = [
                (eval.point[0] + 1.0) * 0.5,
                (eval.point[1] + 1.0) * 0.5,
                box_in[2] * (eval.jacobian[0][0].powi(2) + eval.jacobian[1][0].powi(2)).sqrt(),
                box_in[3] * (eval.jacobian[0][1].powi(2) + eval.jacobian[1][1].powi(2)).sqrt(),
            ];
            let mapped = transform_box_with_jacobian(
                box_in,
                params,
                false,
                HfInverseMethod::Newton,
                20,
                1.0,
            );
            for (actual, expected) in mapped.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            let restored =
                transform_box_with_jacobian(mapped, params, true, HfInverseMethod::Newton, 20, 1.0);
            for (actual, expected) in restored.into_iter().zip(box_in) {
                assert!((actual - expected).abs() < 2.0e-4);
            }
        }
    }

    #[test]
    fn nchw_warp_preserves_channel_batch_and_xy_axes() {
        let (batch, channels, height, width) = (2, 2, 2, 3);
        let mut input = vec![0.0; batch * channels * height * width];
        for n in 0..batch {
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let index = ((n * channels + c) * height + y) * width + x;
                        input[index] =
                            1000.0 * n as f32 + 100.0 * c as f32 + 10.0 * y as f32 + x as f32;
                    }
                }
            }
        }
        let mut grid = vec![0.0; batch * height * width * 2];
        for point in grid.chunks_exact_mut(2) {
            point[0] = 1.25;
            point[1] = 0.5;
        }

        let output = warp_nchw(&input, batch, channels, height, width, &grid).unwrap();
        for n in 0..batch {
            for c in 0..channels {
                let expected = 6.25 + 1000.0 * n as f32 + 100.0 * c as f32;
                let start = (n * channels + c) * height * width;
                assert!(
                    output[start..start + height * width]
                        .iter()
                        .all(|value| (*value - expected).abs() < 1.0e-5)
                );
            }
        }
    }

    #[test]
    fn bilinear_warp_adjoint_matches_input_and_grid_finite_differences() {
        let (batch, channels, height, width) = (1, 2, 2, 3);
        let input = vec![
            0.2, -0.4, 0.7, 0.9, 0.1, -0.3, -0.2, 0.6, 0.4, -0.8, 0.3, 0.5,
        ];
        let mut grid = vec![0.0; batch * height * width * 2];
        for (index, point) in grid.chunks_exact_mut(2).enumerate() {
            point[0] = 0.7 + index as f32 * 0.03;
            point[1] = 0.4 + index as f32 * 0.02;
        }
        let seed = vec![
            0.3, -0.1, 0.5, 0.8, -0.4, 0.2, -0.7, 0.6, 0.1, -0.2, 0.9, 0.4,
        ];
        let gradients =
            warp_nchw_backward(&input, &seed, batch, channels, height, width, &grid).unwrap();
        let epsilon = 1.0e-3;

        for index in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let loss_plus = dot(
                &warp_nchw(&plus, batch, channels, height, width, &grid).unwrap(),
                &seed,
            );
            let loss_minus = dot(
                &warp_nchw(&minus, batch, channels, height, width, &grid).unwrap(),
                &seed,
            );
            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - gradients.input[index]).abs() < 2.0e-4);
        }
        for index in 0..grid.len() {
            let mut plus = grid.clone();
            let mut minus = grid.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let loss_plus = dot(
                &warp_nchw(&input, batch, channels, height, width, &plus).unwrap(),
                &seed,
            );
            let loss_minus = dot(
                &warp_nchw(&input, batch, channels, height, width, &minus).unwrap(),
                &seed,
            );
            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert!((numeric - gradients.grid[index]).abs() < 2.0e-4);
        }
    }

    #[test]
    fn average_pool_downsample_matches_nchw_floor_tiles_and_small_inputs() {
        let (batch, channels, height, width) = (2, 2, 3, 5);
        let mut input = vec![0.0; batch * channels * height * width];
        for n in 0..batch {
            for c in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let index = ((n * channels + c) * height + y) * width + x;
                        input[index] =
                            1000.0 * n as f32 + 100.0 * c as f32 + 10.0 * y as f32 + x as f32;
                    }
                }
            }
        }
        let (pooled, out_h, out_w) =
            average_pool_downsample_nchw(&input, batch, channels, height, width, 2).unwrap();
        assert_eq!((out_h, out_w), (1, 2));
        for n in 0..batch {
            for c in 0..channels {
                let start = (n * channels + c) * out_h * out_w;
                let base = 1000.0 * n as f32 + 100.0 * c as f32;
                assert_eq!(pooled[start], base + 5.5);
                assert_eq!(pooled[start + 1], base + 7.5);
            }
        }

        let (single, out_h, out_w) = average_pool_downsample_nchw(&[3.0], 1, 1, 1, 1, 8).unwrap();
        assert_eq!((out_h, out_w), (1, 1));
        assert_eq!(single, [3.0]);
    }

    #[test]
    fn predictor_forward_matches_hand_computed_nchw_fc_values() {
        let input = vec![
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
            31.0,
        ];
        let cache = predictor_forward_nchw(
            &input,
            1,
            2,
            2,
            4,
            2,
            1,
            4,
            &[0.1, 0.2, 0.3, 0.4],
            &[0.5],
            &[0.1, -0.2, 0.3, 0.4],
            &[0.0; 4],
        )
        .unwrap();
        assert_eq!(cache.pooled, [6.0, 10.0, 22.0, 26.0]);
        assert!((cache.hidden_preactivation[0] - 20.1).abs() < 1.0e-5);
        assert!((cache.hidden[0] - 20.1).abs() < 1.0e-5);
        for (actual, expected) in cache.raw_output.iter().zip([2.01, -4.02, 6.03, 8.04]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        assert!((cache.constrained_params[0] - 0.9 * 2.01f32.tanh()).abs() < 1.0e-6);
        assert!((cache.constrained_params[1] - 0.9 * (-4.02f32).tanh()).abs() < 1.0e-6);
        assert_eq!(cache.constrained_params[2], MAX_RADIUS);
        assert!((cache.constrained_params[3] - sigmoid(8.04)).abs() < 1.0e-6);
    }

    #[test]
    fn predictor_nchw_fc_layout_and_backward_match_finite_differences() {
        let (batch, channels, height, width) = (2, 2, 3, 5);
        let downsample = 2;
        let hidden_count = 2;
        let param_count = 4;
        let input = (0..batch * channels * height * width)
            .map(|index| 0.1 + index as f32 * 0.013)
            .collect::<Vec<_>>();
        let w1 = vec![0.12, 0.08, 0.06, 0.09, 0.04, 0.11, 0.07, 0.05];
        let b1 = vec![0.4, 0.6];
        let w2 = vec![0.03, -0.02, -0.01, 0.04, 0.02, 0.01, -0.03, 0.05];
        let b2 = vec![0.2, -0.15, 0.1, 0.3];
        let forward = predictor_forward_nchw(
            &input,
            batch,
            channels,
            height,
            width,
            downsample,
            hidden_count,
            param_count,
            &w1,
            &b1,
            &w2,
            &b2,
        )
        .unwrap();
        assert_eq!(forward.pooled.len(), batch * channels * 1 * 2);
        assert_eq!(forward.hidden.len(), batch * hidden_count);
        assert_eq!(forward.constrained_params.len(), batch * param_count);
        assert!(
            forward
                .constrained_params
                .iter()
                .all(|value| value.is_finite())
        );
        let seed = vec![0.2, -0.3, 0.4, 0.1, -0.1, 0.5, -0.2, 0.3];
        let gradients = predictor_backward_nchw(
            (batch, channels, height, width),
            downsample,
            hidden_count,
            param_count,
            &w1,
            &w2,
            &forward,
            &seed,
        )
        .unwrap();
        let epsilon = 1.0e-3;
        let loss = |input: &[f32], w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]| {
            dot(
                &predictor_forward_nchw(
                    input,
                    batch,
                    channels,
                    height,
                    width,
                    downsample,
                    hidden_count,
                    param_count,
                    w1,
                    b1,
                    w2,
                    b2,
                )
                .unwrap()
                .constrained_params,
                &seed,
            )
        };
        for (analytic, input_index) in [(&gradients.source, 0), (&gradients.source, 31)] {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[input_index] += epsilon;
            minus[input_index] -= epsilon;
            let numeric = (loss(&plus, &w1, &b1, &w2, &b2) - loss(&minus, &w1, &b1, &w2, &b2))
                / (2.0 * epsilon);
            assert!((numeric - analytic[input_index]).abs() < 2.0e-3);
        }
        for (index, analytic) in [(0, gradients.w1[0]), (5, gradients.w1[5])] {
            let mut plus = w1.clone();
            let mut minus = w1.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (loss(&input, &plus, &b1, &w2, &b2)
                - loss(&input, &minus, &b1, &w2, &b2))
                / (2.0 * epsilon);
            assert!((numeric - analytic).abs() < 2.0e-3);
        }
        for (index, analytic) in [(0, gradients.w2[0]), (6, gradients.w2[6])] {
            let mut plus = w2.clone();
            let mut minus = w2.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numeric = (loss(&input, &w1, &b1, &plus, &b2)
                - loss(&input, &w1, &b1, &minus, &b2))
                / (2.0 * epsilon);
            assert!((numeric - analytic).abs() < 2.0e-3);
        }
        for (index, analytic) in [(1, gradients.b1[1]), (2, gradients.b2[2])] {
            let mut plus_b1 = b1.clone();
            let mut minus_b1 = b1.clone();
            let mut plus_b2 = b2.clone();
            let mut minus_b2 = b2.clone();
            if index == 1 {
                plus_b1[index] += epsilon;
                minus_b1[index] -= epsilon;
            } else {
                plus_b2[index] += epsilon;
                minus_b2[index] -= epsilon;
            }
            let numeric = (loss(&input, &w1, &plus_b1, &w2, &plus_b2)
                - loss(&input, &w1, &minus_b1, &w2, &minus_b2))
                / (2.0 * epsilon);
            assert!((numeric - analytic).abs() < 2.0e-3);
        }

        let mut changed_second_sample = input.clone();
        changed_second_sample[channels * height * width..].fill(7.0);
        let changed = predictor_forward_nchw(
            &changed_second_sample,
            batch,
            channels,
            height,
            width,
            downsample,
            hidden_count,
            param_count,
            &w1,
            &b1,
            &w2,
            &b2,
        )
        .unwrap();
        assert_eq!(
            &changed.constrained_params[..param_count],
            &forward.constrained_params[..param_count]
        );
    }
}
