//! Transposed convolution host math (tasks/rust/layers.md Task 9).

use crate::model::TensorShape;
use crate::runtime::Tensor;

/// Pre-activation transposed convolution: `weights` is `[c][n][k][k]`, `output_shape` is
/// `n x (s*(h-1)+k) x (s*(w-1)+k)`.
pub(crate) fn deconv_forward(
    input: &Tensor,
    weights: &[f32],
    biases: &[f32],
    size: usize,
    stride: usize,
    output_shape: TensorShape,
) -> Tensor {
    let (c, n) = (input.shape.channels, output_shape.channels);
    let mut out = Tensor::zeros(output_shape);
    for b in 0..input.shape.batch {
        for ci in 0..c {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let value = input.get(b, ci, y, x);
                    for o in 0..n {
                        for ky in 0..size {
                            for kx in 0..size {
                                let w = weights[((ci * n + o) * size + ky) * size + kx];
                                let (oy, ox) = (y * stride + ky, x * stride + kx);
                                let current = out.get(b, o, oy, ox);
                                out.set(b, o, oy, ox, current + value * w);
                            }
                        }
                    }
                }
            }
        }
        for o in 0..n {
            for oy in 0..output_shape.height {
                for ox in 0..output_shape.width {
                    let current = out.get(b, o, oy, ox);
                    out.set(b, o, oy, ox, current + biases[o]);
                }
            }
        }
    }
    out
}

pub(crate) struct DeconvGrad {
    pub input: Tensor,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

/// Exact backward for `delta` = output gradient after the activation derivative.
/// `weight_scale` multiplies the weight gradient (1.0; `1/batch` reproduces the C++ quirk).
pub(crate) fn deconv_backward(
    input: &Tensor,
    weights: &[f32],
    delta: &Tensor,
    size: usize,
    stride: usize,
    weight_scale: f32,
) -> DeconvGrad {
    let (c, n) = (input.shape.channels, delta.shape.channels);
    let mut grad = DeconvGrad {
        input: Tensor::zeros(input.shape),
        weights: vec![0.0; weights.len()],
        biases: vec![0.0; n],
    };
    for b in 0..input.shape.batch {
        for o in 0..n {
            for oy in 0..delta.shape.height {
                for ox in 0..delta.shape.width {
                    grad.biases[o] += delta.get(b, o, oy, ox);
                }
            }
        }
        for ci in 0..c {
            for y in 0..input.shape.height {
                for x in 0..input.shape.width {
                    let value = input.get(b, ci, y, x);
                    let mut dx = 0.0f32;
                    for o in 0..n {
                        for ky in 0..size {
                            for kx in 0..size {
                                let d = delta.get(b, o, y * stride + ky, x * stride + kx);
                                let wi = ((ci * n + o) * size + ky) * size + kx;
                                dx += weights[wi] * d;
                                grad.weights[wi] += weight_scale * value * d;
                            }
                        }
                    }
                    grad.input.set(b, ci, y, x, dx);
                }
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_two_stride_two_scatters_each_input_pixel_into_its_own_block() {
        // c = 1 -> n = 1, k = 2, s = 2: out is 4x4, each input pixel paints a 2x2 block with W
        let input = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let weights = [1.0, 10.0, 100.0, 1000.0]; // W[0][0][ky][kx]
        let out = deconv_forward(&input, &weights, &[0.5], 2, 2, TensorShape::new(4, 4, 1, 1));
        let row = |y: usize| (0..4).map(|x| out.get(0, 0, y, x)).collect::<Vec<_>>();
        assert_eq!(row(0), vec![1.5, 10.5, 2.5, 20.5]);
        assert_eq!(row(1), vec![100.5, 1000.5, 200.5, 2000.5]);
        assert_eq!(row(2), vec![3.5, 30.5, 4.5, 40.5]);
        assert_eq!(row(3), vec![300.5, 3000.5, 400.5, 4000.5]);
    }

    #[test]
    fn overlapping_blocks_accumulate() {
        // k = 3, s = 1 on a 1x2 input: out width 4; the middle columns receive both pixels
        let input = Tensor::from_data(TensorShape::new(2, 1, 1, 1), vec![1.0, 1.0]).unwrap();
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let out = deconv_forward(&input, &weights, &[0.0], 3, 1, TensorShape::new(4, 3, 1, 1));
        assert_eq!(out.get(0, 0, 0, 1), 2.0 + 1.0); // pixel0 kx=1 + pixel1 kx=0
    }

    #[test]
    fn backward_matches_central_differences() {
        let (c, n, k, s) = (2usize, 3usize, 3usize, 2usize);
        let ishape = TensorShape::new(3, 2, c, 2);
        let oshape = TensorShape::new(s * (3 - 1) + k, s * (2 - 1) + k, n, 2);
        let fill = |len: usize, f: f32| (0..len).map(|i| (i as f32 * f).sin()).collect::<Vec<f32>>();
        let input = Tensor::from_data(ishape, fill(2 * ishape.element_count(), 0.37)).unwrap();
        let weights = fill(c * n * k * k, 0.53);
        let biases = fill(n, 1.1);
        let seed = Tensor::from_data(oshape, fill(2 * oshape.element_count(), 0.91)).unwrap();
        let loss = |x: &Tensor, w: &[f32], b: &[f32]| -> f32 {
            deconv_forward(x, w, b, k, s, oshape).data.iter().zip(&seed.data).map(|(o, g)| o * g).sum()
        };
        let g = deconv_backward(&input, &weights, &seed, k, s, 1.0);
        let h = 1e-2f32;
        let check = |name: &str, analytic: f32, plus: f32, minus: f32| {
            let fd = (plus - minus) / (2.0 * h);
            assert!((fd - analytic).abs() <= 5e-3 + 5e-3 * fd.abs(), "{name}: fd {fd}, analytic {analytic}");
        };
        for i in (0..weights.len()).step_by(7) {
            let (mut p, mut m) = (weights.clone(), weights.clone());
            p[i] += h;
            m[i] -= h;
            check(&format!("w[{i}]"), g.weights[i], loss(&input, &p, &biases), loss(&input, &m, &biases));
        }
        for i in 0..n {
            let (mut p, mut m) = (biases.clone(), biases.clone());
            p[i] += h;
            m[i] -= h;
            check(&format!("b[{i}]"), g.biases[i], loss(&input, &weights, &p), loss(&input, &weights, &m));
        }
        for i in (0..input.data.len()).step_by(5) {
            let (mut p, mut m) = (input.clone(), input.clone());
            p.data[i] += h;
            m.data[i] -= h;
            check(&format!("x[{i}]"), g.input.data[i], loss(&p, &weights, &biases), loss(&m, &weights, &biases));
        }
    }
}
