//! Host reference for the `[dcnv4]` deformable **aggregation** and its adjoint.
//!
//! Scope: offsets are given (the coarse `d_stride` projection and its upsample
//! are separate stages). The contract is transcribed from
//! `darknet_core::dcnv4::{dcnv4_forward, dcnv4_backward}`, themselves derived
//! from `dcnv4_deformable_forward_cpu`: per output position the row holds
//! `[2*K coordinates, K mask logits]` per group, K=size²-remove_center, with
//! the complete row padded to a multiple of eight. Sampling is zero-padded;
//! `remove_center` skips the middle kernel location without consuming a packed
//! point slot. The softmax denominator adds `1e-6`, and the coordinate
//! gradient carries the `offset_scale` factor.
//!
//! Fixtures must keep every sample off a cell boundary: a finite difference that
//! pushes `sx` across an integer compares a one-sided derivative with a central
//! one and reports a false mismatch (see `tasks/rust/lessons.md`).

use super::bilinear_gather::sample_at;
#[path = "dcnv4_composition.rs"]
mod composition;
pub use composition::{CompositionGradients, composition_backward, composition_forward};

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
        self.size * self.size - usize::from(self.remove_center)
    }
    pub fn offset_dim(&self) -> usize {
        (self.groups * self.kernel_points() * 3 + 7) / 8 * 8
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
        // Launchers use these extents for allocations and unsafe device indexing.
        // Reject overflow before any unchecked product in the accessors above.
        let product = |dims: &[usize]| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= i64::MAX as usize)
                .ok_or_else(|| "dcnv4 aggregation: dimension product overflow".to_owned())
        };
        product(&[self.size, self.size])?;
        if self.kernel_points() == 0 {
            return Err("dcnv4 aggregation requires an active kernel point".to_owned());
        }
        let raw = product(&[self.groups, self.kernel_points(), 3])?;
        raw.checked_add(7)
            .ok_or_else(|| "dcnv4 aggregation: row alignment overflow".to_owned())?;
        product(&[
            self.batch,
            self.output_height,
            self.output_width,
            self.offset_dim(),
        ])?;
        product(&[self.input_height, self.input_width])?;
        product(&[
            self.batch,
            self.channels,
            self.input_height,
            self.input_width,
        ])?;
        product(&[self.output_height, self.output_width])?;
        product(&[
            self.batch,
            self.channels,
            self.output_height,
            self.output_width,
        ])?;
        product(&[
            self.batch,
            self.output_height,
            self.output_width,
            self.groups,
            self.size,
            self.size,
            3,
        ])?;
        product(&[self.dilation, self.size - 1])?;
        Ok(())
    }
}

/// Full projection/upsample/aggregation descriptor (before bias/BN/activation).
/// Launchers must call `checked` before using derived extents.
#[derive(Clone, Copy, Debug)]
pub struct Dcnv4Composition {
    pub geometry: Dcnv4Geometry,
    pub d_stride: usize,
    pub offset_scale: f32,
}

impl Dcnv4Composition {
    pub fn coarse_height(&self) -> usize {
        self.geometry.output_height.div_ceil(self.d_stride)
    }
    pub fn coarse_width(&self) -> usize {
        self.geometry.output_width.div_ceil(self.d_stride)
    }
    pub fn weight_count(&self) -> usize {
        self.geometry.offset_dim()
            * self.geometry.channels
            * self.geometry.size
            * self.geometry.size
    }
    pub fn checked(&self) -> Result<(), String> {
        let g = &self.geometry;
        g.checked()?;
        if self.d_stride == 0 || !self.offset_scale.is_finite() {
            return Err(
                "dcnv4 composition: positive d_stride and finite scale required".to_owned(),
            );
        }
        let bad = || "dcnv4 composition: invalid or overflowing geometry".to_owned();
        let output_dim = |input: usize, stride: usize| -> Option<usize> {
            input
                .checked_add(g.padding.checked_mul(2)?)?
                .checked_sub(g.dilation.checked_mul(g.size - 1)?)?
                .checked_sub(1)?
                .checked_div(stride)?
                .checked_add(1)
        };
        if output_dim(g.input_height, g.stride_y) != Some(g.output_height)
            || output_dim(g.input_width, g.stride_x) != Some(g.output_width)
        {
            return Err(bad());
        }
        let count = |dims: &[usize]| -> Result<(), String> {
            let n = dims
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(bad)?;
            if n > i64::MAX as usize / 4 {
                return Err(bad());
            }
            Ok(())
        };
        count(&[g.offset_dim(), g.channels, g.size, g.size])?;
        count(&[
            g.batch,
            self.coarse_height(),
            self.coarse_width(),
            g.offset_dim(),
        ])?;
        count(&[g.batch, g.input_height, g.input_width, g.channels])?;
        count(&[g.batch, g.output_height, g.output_width, g.channels])?;
        count(&[g.batch, g.output_height, g.output_width, g.offset_dim()])?;
        for (coarse, stride) in [
            (self.coarse_height(), g.stride_y),
            (self.coarse_width(), g.stride_x),
        ] {
            let step = stride.checked_mul(self.d_stride).ok_or_else(bad)?;
            let last = (coarse - 1)
                .checked_mul(step)
                .and_then(|v| v.checked_add((g.size - 1).checked_mul(g.dilation)?))
                .ok_or_else(bad)?;
            if last > i64::MAX as usize {
                return Err(bad());
            }
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
                        let sx = p0_x_scaled + (kw as f32 * g.dilation as f32 + ox) * offset_scale;
                        let sy = p0_y_scaled + (kh as f32 * g.dilation as f32 + oy) * offset_scale;
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
                        let sx = p0_x_scaled + (kw as f32 * g.dilation as f32 + ox) * offset_scale;
                        let sy = p0_y_scaled + (kh as f32 * g.dilation as f32 + oy) * offset_scale;
                        let mask = masks[point];
                        let (mut gx, mut gy, mut gm) = (0.0f32, 0.0f32, 0.0f32);
                        for d in 0..cpg {
                            let channel = group * cpg + d;
                            let base = (b * g.channels + channel) * plane;
                            let top = output_grad[(b * g.channels + channel) * spatial + q];
                            let (value, dy, dx) = sample_and_slopes(
                                input,
                                base,
                                g.input_height,
                                g.input_width,
                                sy,
                                sx,
                            );
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
                let check =
                    |analytic: f32, base: f32, what: &str, index: usize, f: &dyn Fn(f32) -> f32| {
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
    fn aggregation_layout_uses_active_points_and_eight_aligned_rows() {
        let mut g = geometry(false, false);
        assert_eq!(g.kernel_points(), 9);
        assert_eq!(g.offset_dim(), 32);
        g.remove_center = true;
        assert_eq!(g.kernel_points(), 8);
        assert_eq!(g.offset_dim(), 24);
        g.groups = 2;
        assert_eq!(g.offset_dim(), 48);
        g.size = 1;
        assert!(
            g.checked().is_err(),
            "remove_center must leave an active point"
        );
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
