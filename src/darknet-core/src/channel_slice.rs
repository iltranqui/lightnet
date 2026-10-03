//! `[channel_slice]` host math (tasks/rust/layers.md Task 7).
//!
//! Copies channels `[start, end)` of the OUTPUT of the layer named by `from`
//! (not of the previous layer).

use crate::error::{Error, Result};
use crate::model::TensorShape;
use crate::runtime::Tensor;

/// Channels `[start, end)` of every sample of `source`.
pub(crate) fn slice_channels(source: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let s = source.shape;
    if start >= end || end > s.channels {
        return Err(Error::InvalidConfig(format!(
            "channel_slice range [{start}, {end}) is outside {s}"
        )));
    }
    let area = s.width * s.height;
    let run = (end - start) * area;
    let mut out = Tensor::zeros(TensorShape::new(s.width, s.height, end - start, s.batch));
    for b in 0..s.batch {
        let src = (b * s.channels + start) * area;
        out.data[b * run..(b + 1) * run].copy_from_slice(&source.data[src..src + run]);
    }
    Ok(out)
}

/// Exact adjoint of [`slice_channels`]: a zero tensor of `source_shape` with `grad`
/// written into channels `[start, start + grad.channels)`. The caller accumulates it.
pub(crate) fn slice_channels_backward(
    grad: &Tensor,
    source_shape: TensorShape,
    start: usize,
) -> Result<Tensor> {
    let g = grad.shape;
    if g.batch != source_shape.batch
        || g.width != source_shape.width
        || g.height != source_shape.height
        || start + g.channels > source_shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "channel_slice gradient {g} does not fit source {source_shape} at channel {start}"
        )));
    }
    let area = g.width * g.height;
    let run = g.channels * area;
    let mut out = Tensor::zeros(source_shape);
    for b in 0..g.batch {
        let dst = (b * source_shape.channels + start) * area;
        out.data[dst..dst + run].copy_from_slice(&grad.data[b * run..(b + 1) * run]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(shape: TensorShape) -> Tensor {
        let n = shape.batch * shape.element_count();
        Tensor::from_data(shape, (0..n).map(|i| i as f32).collect()).unwrap()
    }

    #[test]
    fn slice_copies_the_channel_range_of_every_sample() {
        // batch 2, 4 channels, 2x1 pixels: value = b*8 + c*2 + p
        let y = slice_channels(&ramp(TensorShape::new(2, 1, 4, 2)), 1, 3).unwrap();
        assert_eq!(y.shape, TensorShape::new(2, 1, 2, 2));
        assert_eq!(y.data, vec![2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0]);
        assert!(slice_channels(&ramp(TensorShape::new(2, 1, 4, 2)), 3, 5).is_err());
    }

    #[test]
    fn slice_backward_is_the_adjoint_and_zero_outside_the_range() {
        let source_shape = TensorShape::new(2, 1, 4, 2);
        let x = ramp(source_shape);
        let y_shape = TensorShape::new(2, 1, 2, 2);
        let y = Tensor::from_data(y_shape, (0..8).map(|i| 0.5 + i as f32 * 0.25).collect()).unwrap();
        let sliced = slice_channels(&x, 1, 3).unwrap();
        let back = slice_channels_backward(&y, source_shape, 1).unwrap();
        let lhs: f32 = sliced.data.iter().zip(&y.data).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.data.iter().zip(&back.data).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-4, "<S x, y> = {lhs}, <x, S^T y> = {rhs}");
        assert!(back.data[0..2].iter().chain(&back.data[6..8]).all(|v| *v == 0.0)); // channels 0 and 3
    }
}
