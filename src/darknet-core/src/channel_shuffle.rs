//! `[channel_shuffle]` host math (tasks/rust/layers.md Task 8).
//!
//! With `rows = groups`, `cols = c / groups`: `out_channel[j*rows + i] =
//! in_channel[i*cols + j]`. Backward is the same permutation with `rows`/`cols`
//! swapped.

use crate::error::{Error, Result};
use crate::runtime::Tensor;

/// `[channel_shuffle]` forward (`inverse = false`) or its exact adjoint (`inverse = true`).
pub(crate) fn shuffle_channels(x: &Tensor, groups: usize, inverse: bool) -> Result<Tensor> {
    let s = x.shape;
    if groups == 0 || s.channels % groups != 0 {
        return Err(Error::InvalidConfig(format!(
            "channel_shuffle needs channels={} divisible by groups={groups}",
            s.channels
        )));
    }
    let (rows, cols) = if inverse {
        (s.channels / groups, groups)
    } else {
        (groups, s.channels / groups)
    };
    Tensor::from_data(
        s,
        darknet_cuda::host_ops::channel_permute(&x.data, s.batch, rows, cols, s.width * s.height),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TensorShape;

    fn ramp(shape: TensorShape) -> Tensor {
        let n = shape.batch * shape.element_count();
        Tensor::from_data(shape, (0..n).map(|i| i as f32).collect()).unwrap()
    }

    #[test]
    fn shuffle_permutes_channels_and_inverse_restores() {
        let x = ramp(TensorShape::new(1, 1, 6, 1)); // channel c holds value c
        let y = shuffle_channels(&x, 2, false).unwrap();
        assert_eq!(y.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(shuffle_channels(&y, 2, true).unwrap().data, x.data);
        assert!(shuffle_channels(&x, 4, false).is_err()); // 6 % 4 != 0
    }
}
