//! `maxpool_depth` host math (tasks/rust/layers.md Task 4).
//!
//! C++ `maxpool_depth=1`: output channel `g` is the max over input channels
//! `g, g + out_c, g + 2*out_c, ...` at the same pixel; spatial size is unchanged.

use crate::error::{Error, Result};
use crate::model::TensorShape;
use crate::runtime::Tensor;

/// First maximum over the input channels `g, g + out_c, ...` at one pixel
/// (C++ `val > max` keeps the first on ties). Returns `(value, input_channel)`.
fn depth_argmax(input: &Tensor, b: usize, g: usize, out_c: usize, y: usize, x: usize) -> (f32, usize) {
    let mut best_channel = g;
    let mut best = input.get(b, g, y, x);
    let mut k = g + out_c;
    while k < input.shape.channels {
        let value = input.get(b, k, y, x);
        if value > best {
            best = value;
            best_channel = k;
        }
        k += out_c;
    }
    (best, best_channel)
}

pub(crate) fn forward_maxpool_depth(input: &Tensor, output_shape: TensorShape) -> Result<Tensor> {
    if output_shape.width != input.shape.width
        || output_shape.height != input.shape.height
        || output_shape.batch != input.shape.batch
        || output_shape.channels == 0
        || output_shape.channels > input.shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "maxpool_depth shape mismatch: input {}, output {output_shape}",
            input.shape
        )));
    }
    let mut output = Tensor::zeros(output_shape);
    for b in 0..output_shape.batch {
        for g in 0..output_shape.channels {
            for y in 0..output_shape.height {
                for x in 0..output_shape.width {
                    let (value, _) = depth_argmax(input, b, g, output_shape.channels, y, x);
                    output.set(b, g, y, x, value);
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn backward_maxpool_depth(input: &Tensor, output_gradient: &Tensor) -> Result<Tensor> {
    let mut grad = Tensor::zeros(input.shape);
    let out_c = output_gradient.shape.channels;
    for b in 0..output_gradient.shape.batch {
        for g in 0..out_c {
            for y in 0..output_gradient.shape.height {
                for x in 0..output_gradient.shape.width {
                    let (_, k) = depth_argmax(input, b, g, out_c, y, x);
                    let current = grad.get(b, k, y, x);
                    grad.set(b, k, y, x, current + output_gradient.get(b, g, y, x));
                }
            }
        }
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorShape;

    #[test]
    fn maxpool_depth_picks_the_first_max_over_strided_channels_and_routes_the_gradient() {
        // w=2 h=1 c=4: channel k = [a_k, b_k]; out_channels=2 -> group0 = {ch0, ch2}, group1 = {ch1, ch3}
        let input = Tensor::from_data(
            TensorShape::new(2, 1, 4, 1),
            vec![1.0, 5.0, 2.0, 0.0, 3.0, 4.0, 0.0, 9.0],
        )
        .unwrap();
        let out_shape = TensorShape::new(2, 1, 2, 1);
        let out = forward_maxpool_depth(&input, out_shape).unwrap();
        assert_eq!(out.data, vec![3.0, 5.0, 2.0, 9.0]);
        let grad = Tensor::from_data(out_shape, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let dx = backward_maxpool_depth(&input, &grad).unwrap();
        // winners: g0x0 -> ch2, g0x1 -> ch0, g1x0 -> ch1, g1x1 -> ch3
        assert_eq!(dx.data, vec![0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0]);
        // tie: the first channel wins (C++ `val > max`)
        let tie = Tensor::from_data(TensorShape::new(1, 1, 4, 1), vec![7.0, 1.0, 7.0, 1.0]).unwrap();
        let g = Tensor::from_data(TensorShape::new(1, 1, 2, 1), vec![1.0, 1.0]).unwrap();
        assert_eq!(super::backward_maxpool_depth(&tie, &g).unwrap().data, vec![1.0, 1.0, 0.0, 0.0]);
    }
}
