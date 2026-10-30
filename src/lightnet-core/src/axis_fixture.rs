//! Shared fixtures for axis/layout pins.
//!
//! Every axis test needs the same three properties: a non-square spatial
//! extent, `batch > 1`, and a value that identifies which `(n, c, h, w)` it
//! came from. Tests that predate this module hand-roll those loops, which
//! makes "distinct" mean something slightly different in each one. New pins
//! should use these helpers; see `tasks/rust/02_state/axis-coverage.md`.
//!
//! A pin built on these fixtures is still **hand-derived from source, no
//! compiled-C++ golden**. The sentinel only makes a wrong axis visible; it is
//! not evidence of parity.

use crate::model::TensorShape;

/// Default probing shape `(width, height, channels, batch)`: non-square,
/// `batch > 1`, and `channels != spatial` so a channel/spatial transpose
/// cannot pass.
pub(crate) const PROBE: (usize, usize, usize, usize) = (3, 2, 2, 2);

/// [`PROBE`] as a [`TensorShape`].
pub(crate) fn probe_shape() -> TensorShape {
    TensorShape::new(PROBE.0, PROBE.1, PROBE.2, PROBE.3)
}

/// Sentinel value for position `(n, c, h, w)`.
///
/// Values are `n*1000 + c*100 + h*10 + w`, so a failure message names the
/// position directly and no two positions collide while every extent stays
/// below ten. That limit is deliberate: it keeps the packing readable rather
/// than clever.
pub(crate) fn axis_value(n: usize, c: usize, h: usize, w: usize) -> f32 {
    assert!(
        n < 10 && c < 10 && h < 10 && w < 10,
        "axis_value packs one decimal digit per axis; keep every extent below 10"
    );
    (n * 1000 + c * 100 + h * 10 + w) as f32
}

/// Inverse of [`axis_value`]; panics on values it did not produce.
pub(crate) fn decode_axis_value(value: f32) -> (usize, usize, usize, usize) {
    let packed = value as usize;
    assert_eq!(packed as f32, value, "not an axis_value sentinel: {value}");
    (
        packed / 1000,
        (packed / 100) % 10,
        (packed / 10) % 10,
        packed % 10,
    )
}

/// NCHW flat index `((n*C + c)*H + h)*W + w`, the layout every layer's input,
/// output and delta buffers use.
pub(crate) fn nchw_index(shape: TensorShape, n: usize, c: usize, h: usize, w: usize) -> usize {
    assert!(n < shape.batch, "batch index {n} out of {shape:?}");
    assert!(c < shape.channels, "channel index {c} out of {shape:?}");
    assert!(h < shape.height, "row index {h} out of {shape:?}");
    assert!(w < shape.width, "column index {w} out of {shape:?}");
    ((n * shape.channels + c) * shape.height + h) * shape.width + w
}

/// A full `NCHW` tensor of [`axis_value`] sentinels for `shape`.
pub(crate) fn axis_tensor(shape: TensorShape) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.batch * shape.channels * shape.height * shape.width);
    for n in 0..shape.batch {
        for c in 0..shape.channels {
            for h in 0..shape.height {
                for w in 0..shape.width {
                    values.push(axis_value(n, c, h, w));
                }
            }
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_values_are_distinct_and_decode_to_their_position() {
        for n in 0..3 {
            for c in 0..4 {
                for h in 0..3 {
                    for w in 0..4 {
                        let value = axis_value(n, c, h, w);
                        assert_eq!(decode_axis_value(value), (n, c, h, w));
                    }
                }
            }
        }
    }

    #[test]
    fn nchw_index_matches_an_independently_written_formula() {
        let shape = TensorShape::new(5, 3, 2, 4);
        for n in 0..shape.batch {
            for c in 0..shape.channels {
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        let expected = n * (shape.channels * shape.height * shape.width)
                            + c * (shape.height * shape.width)
                            + h * shape.width
                            + w;
                        assert_eq!(nchw_index(shape, n, c, h, w), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn axis_tensor_is_nchw_order_and_complete() {
        let shape = probe_shape();
        let values = axis_tensor(shape);
        assert_eq!(
            values.len(),
            shape.batch * shape.channels * shape.height * shape.width
        );
        for n in 0..shape.batch {
            for c in 0..shape.channels {
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        assert_eq!(
                            values[nchw_index(shape, n, c, h, w)],
                            axis_value(n, c, h, w)
                        );
                    }
                }
            }
        }
    }
}
