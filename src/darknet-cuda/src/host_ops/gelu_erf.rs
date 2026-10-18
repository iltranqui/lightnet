//! Exact-erf GELU reference for LAYA, distinct from the tanh activation.
//! Nonfinite values follow the literal CPU formulas, without clipping/limits.

pub fn checked_len(shape: &[usize]) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err("exact GELU requires nonempty positive dimensions".to_owned());
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| "exact GELU storage exceeds bounded indexing".to_owned())
}

pub fn value(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * std::f32::consts::FRAC_1_SQRT_2).erf())
}

pub fn slope(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + (x * std::f32::consts::FRAC_1_SQRT_2).erf());
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
    cdf + x * pdf
}
