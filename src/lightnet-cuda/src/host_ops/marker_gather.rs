//! Checked layout and generic-backend reference for LAYA marker gather.

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub batch: usize,
    pub hidden: usize,
    pub sequence: usize,
    pub marker_channels: usize,
    pub options: usize,
    pub input_len: usize,
    pub markers_len: usize,
    pub output_len: usize,
}

impl Layout {
    pub fn new(a: &[usize], b: &[usize], options: usize) -> Result<Self, String> {
        if a.len() != 4
            || b.len() != 4
            || a.contains(&0)
            || b.contains(&0)
            || a[2] != 1
            || b[2] != 1
            || a[0] != b[0]
            || a[3] != b[3]
            || options == 0
            || options.checked_mul(2).is_none_or(|n| b[1] < n)
        {
            return Err(format!(
                "marker_gather: incompatible shapes {a:?}, {b:?}, options={options}"
            ));
        }
        let count = |dims: &[usize]| -> Result<usize, String> {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| "marker_gather: tensor storage exceeds bounded indexing".to_owned())
        };
        Ok(Self {
            batch: a[0],
            hidden: a[1],
            sequence: a[3],
            marker_channels: b[1],
            options,
            input_len: count(a)?,
            markers_len: count(b)?,
            output_len: count(&[a[0], a[1], options])?,
        })
    }

    pub fn output_shape(&self) -> [usize; 4] {
        [self.batch, self.hidden, 1, self.options]
    }

    pub fn forward(&self, features: &[f32], markers: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), self.input_len);
        assert_eq!(markers.len(), self.markers_len);
        let mut out = vec![0.0; self.output_len];
        for b in 0..self.batch {
            for c in 0..self.hidden {
                for j in 0..self.options {
                    let pos = position(
                        markers[(b * self.marker_channels + 2 * j) * self.sequence],
                        self.sequence,
                    );
                    out[(b * self.hidden + c) * self.options + j] =
                        features[(b * self.hidden + c) * self.sequence + pos];
                }
            }
        }
        out
    }

    pub fn backward(&self, markers: &[f32], dy: &[f32]) -> Vec<f32> {
        assert_eq!(markers.len(), self.markers_len);
        assert_eq!(dy.len(), self.output_len);
        let mut dx = vec![0.0; self.input_len];
        for b in 0..self.batch {
            for j in 0..self.options {
                let pos = position(
                    markers[(b * self.marker_channels + 2 * j) * self.sequence],
                    self.sequence,
                );
                for c in 0..self.hidden {
                    dx[(b * self.hidden + c) * self.sequence + pos] +=
                        dy[(b * self.hidden + c) * self.options + j];
                }
            }
        }
        dx
    }
}

/// Rust CPU reference policy: half away from zero, clamp, NaN -> zero.
pub fn position(value: f32, sequence: usize) -> usize {
    let rounded = f64::from(value).round();
    if rounded < 0.0 || rounded.is_nan() {
        0
    } else if rounded >= sequence as f64 {
        sequence - 1
    } else {
        rounded as usize
    }
}
