//! `[channel_shuffle]` host math (tasks/rust/04_reference/layers.md Task 8).
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
        lightnet_cuda::host_ops::channel_permute(&x.data, s.batch, rows, cols, s.width * s.height),
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
    fn shuffle_axis_layout_preserves_batch_spatial_planes_and_group_interleave() {
        use crate::axis_fixture::{axis_tensor, axis_value};

        let shape = TensorShape::new(3, 2, 6, 2);
        let input = Tensor::from_data(shape, axis_tensor(shape)).expect("axis input");
        let output = shuffle_channels(&input, 3, false).expect("forward shuffle");
        for n in 0..2 {
            for c in 0..6 {
                let input_channel = (c % 3) * 2 + c / 3;
                for h in 0..2 {
                    for w in 0..3 {
                        let index = ((n * 6 + c) * 2 + h) * 3 + w;
                        assert_eq!(
                            output.data[index],
                            axis_value(n, input_channel, h, w),
                            "channel-shuffle axis mismatch at (n={n}, c={c}, h={h}, w={w})"
                        );
                    }
                }
            }
        }

        let seed = Tensor::from_data(shape, axis_tensor(shape)).expect("axis seed");
        let input_gradient = shuffle_channels(&seed, 3, true).expect("shuffle adjoint");
        for n in 0..2 {
            for c in 0..6 {
                let output_channel = (c % 2) * 3 + c / 2;
                for h in 0..2 {
                    for w in 0..3 {
                        let index = ((n * 6 + c) * 2 + h) * 3 + w;
                        assert_eq!(
                            input_gradient.data[index],
                            axis_value(n, output_channel, h, w),
                            "channel-shuffle adjoint mismatch at (n={n}, c={c}, h={h}, w={w})"
                        );
                    }
                }
            }
        }
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
