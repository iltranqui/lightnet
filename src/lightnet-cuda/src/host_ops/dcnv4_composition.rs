//! Host defaults for generic DCNv4 backends. Oxide overrides these entry points.
use super::{Dcnv4Composition, aggregate, aggregate_backward};

pub struct CompositionGradients {
    pub input: Vec<f32>,
    pub offset_weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
}

fn validate(input: &[f32], weights: &[f32], d: Dcnv4Composition) -> Result<(), String> {
    d.checked()?;
    let g = d.geometry;
    if input.len() != g.batch * g.channels * g.input_height * g.input_width
        || weights.len() != d.weight_count()
    {
        return Err("DCNv4 composition input/weight length mismatch".to_owned());
    }
    Ok(())
}

fn neighbors(y: usize, x: usize, d: Dcnv4Composition) -> [(usize, f32); 4] {
    let sy = y as f32 / d.d_stride as f32;
    let sx = x as f32 / d.d_stride as f32;
    let (y0, x0) = (sy.floor() as usize, sx.floor() as usize);
    let (ly, lx) = (sy - y0 as f32, sx - x0 as f32);
    let h = d.coarse_height();
    let w = d.coarse_width();
    let index = |yy: usize, xx: usize| yy.min(h - 1) * w + xx.min(w - 1);
    [
        (index(y0, x0), (1.0 - ly) * (1.0 - lx)),
        (index(y0, x0 + 1), (1.0 - ly) * lx),
        (index(y0 + 1, x0), ly * (1.0 - lx)),
        (index(y0 + 1, x0 + 1), ly * lx),
    ]
}

/// Raw NCHW output and dense NHWC offsets, before the runtime epilogue.
pub fn composition_forward(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    d: Dcnv4Composition,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    validate(input, weights, d)?;
    let g = d.geometry;
    let p = g.offset_dim();
    if biases.len() != p {
        return Err("DCNv4 offset bias length mismatch".to_owned());
    }
    let (ch, cw) = (d.coarse_height(), d.coarse_width());
    let mut coarse = vec![0.0; g.batch * ch * cw * p];
    for b in 0..g.batch {
        for cy in 0..ch {
            for cx in 0..cw {
                for t in 0..p {
                    let mut sum = 0.0;
                    for c in 0..g.channels {
                        for ky in 0..g.size {
                            let yy = cy * g.stride_y * d.d_stride + ky * g.dilation;
                            for kx in 0..g.size {
                                let xx = cx * g.stride_x * d.d_stride + kx * g.dilation;
                                if yy < g.padding || xx < g.padding {
                                    continue;
                                }
                                let (y, x) = (yy - g.padding, xx - g.padding);
                                if y < g.input_height && x < g.input_width {
                                    let i = ((b * g.channels + c) * g.input_height + y)
                                        * g.input_width
                                        + x;
                                    let w =
                                        (t * g.channels + c) * g.size * g.size + ky * g.size + kx;
                                    sum += input[i] * weights[w];
                                }
                            }
                        }
                    }
                    coarse[((b * ch + cy) * cw + cx) * p + t] = sum + biases[t];
                }
            }
        }
    }
    let mut offsets = vec![0.0; g.batch * g.spatial() * p];
    for b in 0..g.batch {
        for y in 0..g.output_height {
            for x in 0..g.output_width {
                for t in 0..p {
                    let i = ((b * g.output_height + y) * g.output_width + x) * p + t;
                    for (q, weight) in neighbors(y, x, d) {
                        offsets[i] += weight * coarse[(b * ch * cw + q) * p + t];
                    }
                }
            }
        }
    }
    Ok((aggregate(input, &offsets, &g, d.offset_scale)?, offsets))
}

/// Exact Rust-mode input/offset-projection adjoints, positive batch sums.
pub fn composition_backward(
    input: &[f32],
    weights: &[f32],
    offsets: &[f32],
    seed: &[f32],
    d: Dcnv4Composition,
) -> Result<CompositionGradients, String> {
    validate(input, weights, d)?;
    let g = d.geometry;
    let p = g.offset_dim();
    let (mut dx, dense) = aggregate_backward(input, offsets, seed, &g, d.offset_scale)?;
    let (ch, cw) = (d.coarse_height(), d.coarse_width());
    let mut coarse = vec![0.0; g.batch * ch * cw * p];
    for b in 0..g.batch {
        for y in 0..g.output_height {
            for x in 0..g.output_width {
                for t in 0..p {
                    let grad = dense[((b * g.output_height + y) * g.output_width + x) * p + t];
                    for (q, weight) in neighbors(y, x, d) {
                        coarse[(b * ch * cw + q) * p + t] += grad * weight;
                    }
                }
            }
        }
    }
    let mut dw = vec![0.0; weights.len()];
    let mut db = vec![0.0; p];
    for b in 0..g.batch {
        for cy in 0..ch {
            for cx in 0..cw {
                for t in 0..p {
                    let grad = coarse[((b * ch + cy) * cw + cx) * p + t];
                    db[t] += grad;
                    for c in 0..g.channels {
                        for ky in 0..g.size {
                            let yy = cy * g.stride_y * d.d_stride + ky * g.dilation;
                            for kx in 0..g.size {
                                let xx = cx * g.stride_x * d.d_stride + kx * g.dilation;
                                if yy < g.padding || xx < g.padding {
                                    continue;
                                }
                                let (y, x) = (yy - g.padding, xx - g.padding);
                                if y < g.input_height && x < g.input_width {
                                    let i = ((b * g.channels + c) * g.input_height + y)
                                        * g.input_width
                                        + x;
                                    let w =
                                        (t * g.channels + c) * g.size * g.size + ky * g.size + kx;
                                    dx[i] += grad * weights[w];
                                    dw[w] += grad * input[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(CompositionGradients {
        input: dx,
        offset_weights: dw,
        offset_biases: db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_ops::dcnv4::Dcnv4Geometry;

    #[test]
    fn composition_defaults_match_finite_differences_and_reject_bad_storage() {
        let d = Dcnv4Composition {
            geometry: Dcnv4Geometry {
                batch: 2,
                channels: 2,
                groups: 1,
                input_height: 3,
                input_width: 4,
                output_height: 3,
                output_width: 4,
                size: 1,
                stride_x: 1,
                stride_y: 1,
                padding: 0,
                dilation: 1,
                softmax: false,
                remove_center: false,
            },
            d_stride: 2,
            offset_scale: 1.0,
        };
        let input: Vec<_> = (0..48).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let weights = vec![0.001; d.weight_count()];
        let biases = vec![0.5; d.geometry.offset_dim()];
        let seed: Vec<_> = (0..48).map(|i| (i % 5) as f32 * 0.3 - 0.6).collect();
        let (_, off) = composition_forward(&input, &weights, &biases, d).unwrap();
        let grad = composition_backward(&input, &weights, &off, &seed, d).unwrap();
        let objective = |x: &[f32], w: &[f32], b: &[f32]| -> f32 {
            composition_forward(x, w, b, d)
                .unwrap()
                .0
                .iter()
                .zip(&seed)
                .map(|(v, s)| v * s)
                .sum()
        };
        for (operand, values, gradient) in [
            (0, &input, &grad.input),
            (1, &weights, &grad.offset_weights),
            (2, &biases, &grad.offset_biases),
        ] {
            for i in 0..values.len() {
                let mut plus = values.clone();
                let mut minus = values.clone();
                let eps = 0.002;
                plus[i] += eps;
                minus[i] -= eps;
                let (hi, lo) = match operand {
                    0 => (
                        objective(&plus, &weights, &biases),
                        objective(&minus, &weights, &biases),
                    ),
                    1 => (
                        objective(&input, &plus, &biases),
                        objective(&input, &minus, &biases),
                    ),
                    _ => (
                        objective(&input, &weights, &plus),
                        objective(&input, &weights, &minus),
                    ),
                };
                assert!(
                    (gradient[i] - (hi - lo) / (2.0 * eps)).abs() < 2e-3,
                    "operand={operand} i={i}"
                );
            }
        }
        assert!(composition_forward(&input[1..], &weights, &biases, d).is_err());
        assert!(composition_forward(&input, &weights[1..], &biases, d).is_err());
        assert!(composition_forward(&input, &weights, &biases[1..], d).is_err());
        assert!(composition_backward(&input, &weights, &off[1..], &seed, d).is_err());
        assert!(composition_backward(&input, &weights, &off, &seed[1..], d).is_err());
    }
}
