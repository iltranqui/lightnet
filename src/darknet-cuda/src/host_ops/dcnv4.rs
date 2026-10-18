//! Host reference for the `[dcnv4]` deformable **aggregation** and its adjoint.
//!
//! Scope: offsets are given (the coarse `d_stride` projection and its upsample
//! are separate stages). The contract is transcribed from
//! `darknet_core::dcnv4::{dcnv4_forward, dcnv4_backward}`, themselves derived
//! from `dcnv4_deformable_forward_cpu`: per output position the row holds
//! `[2*K offset pairs, K mask logits]` per group, sampling is bilinear with zero
//! padding, `remove_center` skips the middle kernel point (whose mask gradient
//! then stays zero), the softmax denominator adds `1e-6`, and the coordinate
//! gradient carries the `offset_scale` factor.
//!
//! Fixtures must keep every sample off a cell boundary: a finite difference that
//! pushes `sx` across an integer compares a one-sided derivative with a central
//! one and reports a false mismatch (see `tasks/rust/lessons.md`).

use super::bilinear_gather::sample_at;

/// Geometry and switches of one aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dcnv4Geometry {
    pub batch: usize,
    pub channels: usize,
    pub groups: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding: usize,
    pub dilation: usize,
    pub softmax: bool,
    pub remove_center: bool,
}

impl Dcnv4Geometry {
    pub fn kernel_points(&self) -> usize {
        self.size * self.size
    }
    pub fn offset_dim(&self) -> usize {
        self.groups * self.kernel_points() * 3
    }
    pub fn spatial(&self) -> usize {
        self.output_height * self.output_width
    }
    pub fn checked(&self) -> Result<(), String> {
        let axes = [
            self.batch,
            self.channels,
            self.groups,
            self.input_height,
            self.input_width,
            self.output_height,
            self.output_width,
            self.size,
            self.stride_x,
            self.stride_y,
            self.dilation,
        ];
        if axes.contains(&0) {
            return Err("dcnv4 aggregation requires positive dimensions".to_owned());
        }
        if self.channels % self.groups != 0 {
            return Err("dcnv4 aggregation: groups must divide channels".to_owned());
        }
        Ok(())
    }
}

/// Masks of one `(position, group)` row: logits, or softmax probabilities.
pub fn group_masks(row: &[f32], group_offset: usize, points: usize, softmax: bool) -> Vec<f32> {
    let start = group_offset + 2 * points;
    let mut masks = row[start..start + points].to_vec();
    if softmax {
        let maximum = masks.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for value in &mut masks {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let denominator = sum + 1.0e-6;
        for value in &mut masks {
            *value /= denominator;
        }
    }
    masks
}

/// Bilinear value plus coordinate slopes, zero padded (dcnv4 has no far-outside
/// early-out, and the slopes keep their zero-padded values).
fn sample_and_slopes(
    input: &[f32],
    base: usize,
    h: usize,
    w: usize,
    y: f32,
    x: f32,
) -> (f32, f32, f32) {
    let y0f = y.floor();
    let x0f = x.floor();
    let (y0, x0) = (y0f as i64, x0f as i64);
    let (ly, lx) = (y - y0f, x - x0f);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let get = |yy: i64, xx: i64| -> f32 {
        if yy < 0 || xx < 0 || yy >= h as i64 || xx >= w as i64 {
            0.0
        } else {
            input[base + yy as usize * w + xx as usize]
        }
    };
    let (v00, v01, v10, v11) = (
        get(y0, x0),
        get(y0, x0 + 1),
        get(y0 + 1, x0),
        get(y0 + 1, x0 + 1),
    );
    (
        hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11,
        hx * (v10 - v00) + lx * (v11 - v01),
        hy * (v01 - v00) + ly * (v11 - v10),
    )
}

/// Aggregates `[B, C, H, W]` at `[B, OH*OW, offset_dim]` offsets.
pub fn aggregate(
    input: &[f32],
    offsets: &[f32],
    g: &Dcnv4Geometry,
    offset_scale: f32,
) -> Result<Vec<f32>, String> {
    g.checked()?;
    let plane = g.input_height * g.input_width;
    let spatial = g.spatial();
    let offset_dim = g.offset_dim();
    if input.len() != g.batch * g.channels * plane
        || offsets.len() != g.batch * spatial * offset_dim
    {
        return Err("dcnv4 aggregate: operand length mismatch".to_owned());
    }
    let cpg = g.channels / g.groups;
    let center = (g.dilation * (g.size - 1)) >> 1;
    let center_f = center as f32;
    let mut out = vec![0.0f32; g.batch * g.channels * spatial];
    for b in 0..g.batch {
        for q in 0..spatial {
            let (out_y, out_x) = (q / g.output_width, q % g.output_width);
            let p0_x = center as f32 - g.padding as f32 + out_x as f32 * g.stride_x as f32;
            let p0_y = center as f32 - g.padding as f32 + out_y as f32 * g.stride_y as f32;
            let p0_x_scaled = p0_x - center_f * offset_scale;
            let p0_y_scaled = p0_y - center_f * offset_scale;
            let row_start = (b * spatial + q) * offset_dim;
            let row = &offsets[row_start..row_start + offset_dim];
            for group in 0..g.groups {
                let group_offset = group * g.kernel_points() * 3;
                let masks = group_masks(row, group_offset, g.kernel_points(), g.softmax);
                let mut point = 0;
                for kw in 0..g.size {
                    for kh in 0..g.size {
                        if g.remove_center && kw == g.size / 2 && kh == g.size / 2 {
                            continue;
                        }
                        let ox = row[group_offset + 2 * point];
                        let oy = row[group_offset + 2 * point + 1];
                        let sx =
                            p0_x_scaled + (kw as f32 * g.dilation as f32 + ox) * offset_scale;
                        let sy =
                            p0_y_scaled + (kh as f32 * g.dilation as f32 + oy) * offset_scale;
                        let mask = masks[point];
                        for d in 0..cpg {
                            let channel = group * cpg + d;
                            let base = (b * g.channels + channel) * plane;
                            let value =
                                sample_at(input, base, g.input_height, g.input_width, sy, sx);
                            out[(b * g.channels + channel) * spatial + q] += mask * value;
                        }
                        point += 1;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Exact adjoint of [`aggregate`] for the input and the offsets.
pub fn aggregate_backward(
    input: &[f32],
    offsets: &[f32],
    output_grad: &[f32],
    g: &Dcnv4Geometry,
    offset_scale: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    g.checked()?;
    let plane = g.input_height * g.input_width;
    let spatial = g.spatial();
    let offset_dim = g.offset_dim();
    if input.len() != g.batch * g.channels * plane
        || offsets.len() != g.batch * spatial * offset_dim
        || output_grad.len() != g.batch * g.channels * spatial
    {
        return Err("dcnv4 aggregate_backward: operand length mismatch".to_owned());
    }
    let cpg = g.channels / g.groups;
    let points = g.kernel_points();
    let center = (g.dilation * (g.size - 1)) >> 1;
    let center_f = center as f32;
    let mut d_input = vec![0.0f32; input.len()];
    let mut d_offsets = vec![0.0f32; offsets.len()];
    for b in 0..g.batch {
        for q in 0..spatial {
            let (out_y, out_x) = (q / g.output_width, q % g.output_width);
            let p0_x = center as f32 - g.padding as f32 + out_x as f32 * g.stride_x as f32;
            let p0_y = center as f32 - g.padding as f32 + out_y as f32 * g.stride_y as f32;
            let p0_x_scaled = p0_x - center_f * offset_scale;
            let p0_y_scaled = p0_y - center_f * offset_scale;
            let row_start = (b * spatial + q) * offset_dim;
            let row = &offsets[row_start..row_start + offset_dim];
            for group in 0..g.groups {
                let group_offset = group * points * 3;
                let masks = group_masks(row, group_offset, points, g.softmax);
                let mut mask_gradient = vec![0.0f32; points];
                let mut point = 0;
                for kw in 0..g.size {
                    for kh in 0..g.size {
                        if g.remove_center && kw == g.size / 2 && kh == g.size / 2 {
                            continue;
                        }
                        let ox = row[group_offset + 2 * point];
                        let oy = row[group_offset + 2 * point + 1];
                        let sx =
                            p0_x_scaled + (kw as f32 * g.dilation as f32 + ox) * offset_scale;
                        let sy =
                            p0_y_scaled + (kh as f32 * g.dilation as f32 + oy) * offset_scale;
                        let mask = masks[point];
                        let (mut gx, mut gy, mut gm) = (0.0f32, 0.0f32, 0.0f32);
                        for d in 0..cpg {
                            let channel = group * cpg + d;
                            let base = (b * g.channels + channel) * plane;
                            let top = output_grad[(b * g.channels + channel) * spatial + q];
                            let (value, dy, dx) =
                                sample_and_slopes(input, base, g.input_height, g.input_width, sy, sx);
                            scatter(
                                &mut d_input,
                                base,
                                g.input_height,
                                g.input_width,
                                sy,
                                sx,
                                top * mask,
                            );
                            gx += top * mask * dx * offset_scale;
                            gy += top * mask * dy * offset_scale;
                            gm += top * value;
                        }
                        d_offsets[row_start + group_offset + 2 * point] += gx;
                        d_offsets[row_start + group_offset + 2 * point + 1] += gy;
                        mask_gradient[point] = gm;
                        point += 1;
                    }
                }
                let mask_start = row_start + group_offset + 2 * points;
                if g.softmax {
                    let dot: f32 = mask_gradient.iter().zip(&masks).map(|(g, m)| g * m).sum();
                    for i in 0..points {
                        d_offsets[mask_start + i] += masks[i] * (mask_gradient[i] - dot);
                    }
                } else {
                    for i in 0..points {
                        d_offsets[mask_start + i] += mask_gradient[i];
                    }
                }
            }
        }
    }
    Ok((d_input, d_offsets))
}

fn scatter(d_input: &mut [f32], base: usize, h: usize, w: usize, y: f32, x: f32, scale: f32) {
    let y0f = y.floor();
    let x0f = x.floor();
    let (y0, x0) = (y0f as i64, x0f as i64);
    let (ly, lx) = (y - y0f, x - x0f);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let mut add = |yy: i64, xx: i64, weight: f32| {
        if yy >= 0 && xx >= 0 && yy < h as i64 && xx < w as i64 {
            d_input[base + yy as usize * w + xx as usize] += scale * weight;
        }
    };
    add(y0, x0, hy * hx);
    add(y0, x0 + 1, hy * lx);
    add(y0 + 1, x0, ly * hx);
    add(y0 + 1, x0 + 1, ly * lx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(softmax: bool, remove_center: bool) -> Dcnv4Geometry {
        Dcnv4Geometry {
            batch: 1,
            channels: 2,
            groups: 1,
            input_height: 3,
            input_width: 4,
            output_height: 2,
            output_width: 3,
            size: 3,
            stride_x: 1,
            stride_y: 1,
            padding: 1,
            dilation: 1,
            softmax,
            remove_center,
        }
    }

    #[test]
    fn aggregation_backward_matches_finite_differences() {
        for softmax in [false, true] {
            for remove_center in [false, true] {
                let g = geometry(softmax, remove_center);
                let scale = 0.75f32;
                let input: Vec<f32> = (0..g.batch * g.channels * g.input_height * g.input_width)
                    .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
                    .collect();
                // Offsets stay near a cell centre (0.5..0.58): a finite
                // difference must not push a sample across an integer, where the
                // analytic slope is one-sided and the comparison is invalid.
                let offsets: Vec<f32> = (0..g.batch * g.spatial() * g.offset_dim())
                    .map(|i| 0.5 + ((i * 3 % 5) as f32) * 0.02)
                    .collect();
                let grad: Vec<f32> = (0..g.batch * g.channels * g.spatial())
                    .map(|i| 0.3 + ((i % 5) as f32) * 0.1)
                    .collect();
                let (d_input, d_offsets) =
                    aggregate_backward(&input, &offsets, &grad, &g, scale).unwrap();
                let objective = |inp: &[f32], off: &[f32]| -> f32 {
                    aggregate(inp, off, &g, scale)
                        .unwrap()
                        .iter()
                        .zip(&grad)
                        .map(|(v, s)| v * s)
                        .sum()
                };
                let check = |analytic: f32,
                             base: f32,
                             what: &str,
                             index: usize,
                             f: &dyn Fn(f32) -> f32| {
                    let eps = 1e-2f32;
                    let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
                    assert!(
                        (analytic - want).abs() < 5e-3 + 5e-3 * want.abs(),
                        "{what}[{index}] sm={softmax} rc={remove_center}: \
                         analytic {analytic} vs fd {want}"
                    );
                };
                for i in (0..input.len()).step_by(3) {
                    check(d_input[i], input[i], "d_input", i, &|v| {
                        let mut x = input.clone();
                        x[i] = v;
                        objective(&x, &offsets)
                    });
                }
                for i in 0..offsets.len() {
                    check(d_offsets[i], offsets[i], "d_offsets", i, &|v| {
                        let mut o = offsets.clone();
                        o[i] = v;
                        objective(&input, &o)
                    });
                }
            }
        }
    }

    #[test]
    fn aggregation_rejects_bad_operands() {
        let g = geometry(false, false);
        let input = vec![0.0f32; g.batch * g.channels * g.input_height * g.input_width];
        let offsets = vec![0.0f32; g.batch * g.spatial() * g.offset_dim()];
        let grad = vec![0.0f32; g.batch * g.channels * g.spatial()];
        assert!(aggregate(&input, &offsets[1..], &g, 1.0).is_err());
        assert!(aggregate_backward(&input, &offsets, &grad[1..], &g, 1.0).is_err());
    }
}
