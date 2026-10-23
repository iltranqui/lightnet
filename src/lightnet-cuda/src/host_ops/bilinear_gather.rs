//! Shared bilinear deformable-sampling reference for the deformable layer
//! family (`[deform_conv]`, `[dcnv4]`, `[dsconv]`, `[graph_conv]`).
//!
//! Every one of those layers needs the same primitive: gather `C` channels at
//! one fractional `(y, x)` position per output cell, with zero padding outside
//! the feature map, and propagate the exact adjoint back to both the input
//! feature map and the sample coordinates.
//!
//! The coordinate contract is copied from the existing Rust CPU reference
//! `lightnet_core::deform_conv::bilinear_sample`, which is itself hand-derived
//! from `Jev_laya:src-lib/deform_conv_layer.cpp`:
//!
//! * `y < -1 || y > H || x < -1 || x > W` is a **far-outside** sample: value and
//!   both coordinate gradients are zero (C++ early-out, not the mathematical
//!   bilinear extension).
//! * Otherwise the floor-selected cell contributes its four corners, and each
//!   corner contributes only if it is inside the feature map.
//! * Fractional offsets are `ly = y - floor(y)`, `lx = x - floor(x)`; integer
//!   coordinates therefore use the right-sided cell, matching the CPU
//!   convention (and its one-sided derivative).
//!
//! Layout: `input` is `[B, C, H, W]`, the sample fields are `[B, S, OH, OW]`
//! (`S` = kernel points for a convolution, one for a plain gather), and the
//! output is `[B, S, C, OH, OW]` — the im2col-style column tensor the layer
//! then multiplies by its kernel weights. Materialising those columns is the
//! same memory trade the existing im2col kernels already make; streaming them
//! into the convolution is a later optimisation, not part of this contract.

/// Checked geometry of a deformable bilinear gather.
///
/// Construction validates every dimension and the products used for indexing,
/// so callers cannot build a geometry that overflows or has a zero axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BilinearGatherGeometry {
    pub batch: usize,
    pub channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub samples: usize,
    pub output_height: usize,
    pub output_width: usize,
}

impl BilinearGatherGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch: usize,
        channels: usize,
        input_height: usize,
        input_width: usize,
        samples: usize,
        output_height: usize,
        output_width: usize,
    ) -> Result<Self, String> {
        let geometry = Self {
            batch,
            channels,
            input_height,
            input_width,
            samples,
            output_height,
            output_width,
        };
        geometry.checked_products()?;
        Ok(geometry)
    }

    pub fn input_len(&self) -> usize {
        self.batch * self.channels * self.input_height * self.input_width
    }

    pub fn sample_len(&self) -> usize {
        self.batch * self.samples * self.output_height * self.output_width
    }

    pub fn output_len(&self) -> usize {
        self.sample_len() * self.channels
    }

    /// Reject empty axes and any indexing product that does not fit `usize`
    /// (and stays inside the bounded `i32` range the device kernels index with).
    pub fn checked_products(&self) -> Result<(), String> {
        let axes = [
            ("batch", self.batch),
            ("channels", self.channels),
            ("input height", self.input_height),
            ("input width", self.input_width),
            ("samples", self.samples),
            ("output height", self.output_height),
            ("output width", self.output_width),
        ];
        for (name, value) in axes {
            if value == 0 {
                return Err(format!("deformable gather requires a positive {name}"));
            }
        }
        for len in [self.input_len(), self.output_len()] {
            if len > i32::MAX as usize {
                return Err("deformable gather storage exceeds bounded indexing".to_owned());
            }
        }
        Ok(())
    }

    #[inline]
    pub fn input_index(&self, b: usize, c: usize, y: usize, x: usize) -> usize {
        ((b * self.channels + c) * self.input_height + y) * self.input_width + x
    }

    #[inline]
    pub fn sample_index(&self, b: usize, s: usize, y: usize, x: usize) -> usize {
        ((b * self.samples + s) * self.output_height + y) * self.output_width + x
    }

    #[inline]
    pub fn output_index(&self, b: usize, s: usize, c: usize, y: usize, x: usize) -> usize {
        (((b * self.samples + s) * self.channels + c) * self.output_height + y) * self.output_width
            + x
    }
}

/// One resolved bilinear cell: four corner weights plus the value and the
/// coordinate derivatives of the sampled value for a single channel.
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    corners: [(isize, isize, f32); 4],
    /// `false` when the far-outside early-out fired; then every gradient is 0.
    active: bool,
}

#[inline]
fn resolve_cell(y: f32, x: f32, height: usize, width: usize) -> Cell {
    // NaN/infinite coordinates deliberately follow the CPU reference: the
    // comparison is false, the saturating `as isize` cast selects cell 0 and
    // the fractional parts propagate NaN. Do not add an `is_finite` guard.
    if y < -1.0 || y > height as f32 || x < -1.0 || x > width as f32 {
        return Cell::default();
    }
    let y0f = y.floor();
    let x0f = x.floor();
    let (y0, x0) = (y0f as isize, x0f as isize);
    let (ly, lx) = (y - y0f, x - x0f);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    Cell {
        corners: [
            (y0, x0, hy * hx),
            (y0, x0 + 1, hy * lx),
            (y0 + 1, x0, ly * hx),
            (y0 + 1, x0 + 1, ly * lx),
        ],
        active: true,
    }
}

#[inline]
fn inside(y: isize, x: isize, height: usize, width: usize) -> bool {
    y >= 0 && y < height as isize && x >= 0 && x < width as isize
}

/// Bilinear sample of one channel at `(y, x)`, following the CPU reference:
/// far-outside samples are zero, corners outside the feature map contribute
/// nothing, and integer coordinates use the floor-selected cell. NaN/infinite
/// coordinates propagate (no `is_finite` guard), matching the CPU.
pub fn sample_at(
    input: &[f32],
    channel_base: usize,
    height: usize,
    width: usize,
    y: f32,
    x: f32,
) -> f32 {
    let cell = resolve_cell(y, x, height, width);
    if !cell.active {
        return 0.0;
    }
    let mut value = 0.0f32;
    for (cy, cx, weight) in cell.corners {
        if inside(cy, cx, height, width) {
            value += weight * input[channel_base + cy as usize * width + cx as usize];
        }
    }
    value
}

/// Reference forward: `[B,C,H,W]` gathered at `[B,S,OH,OW]` coordinates into
/// `[B,S,C,OH,OW]`. Returns zeros for far-outside samples.
pub fn forward(
    input: &[f32],
    sample_y: &[f32],
    sample_x: &[f32],
    geometry: &BilinearGatherGeometry,
) -> Result<Vec<f32>, String> {
    geometry.checked_products()?;
    if input.len() != geometry.input_len()
        || sample_y.len() != geometry.sample_len()
        || sample_x.len() != geometry.sample_len()
    {
        return Err("deformable gather operand length does not match its geometry".to_owned());
    }
    let mut output = vec![0.0f32; geometry.output_len()];
    for b in 0..geometry.batch {
        for s in 0..geometry.samples {
            for oy in 0..geometry.output_height {
                for ox in 0..geometry.output_width {
                    let sample = geometry.sample_index(b, s, oy, ox);
                    for c in 0..geometry.channels {
                        let value = sample_at(
                            input,
                            geometry.input_index(b, c, 0, 0),
                            geometry.input_height,
                            geometry.input_width,
                            sample_y[sample],
                            sample_x[sample],
                        );
                        let out = geometry.output_index(b, s, c, oy, ox);
                        output[out] = value;
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Reference adjoint of [`forward`]. Returns `(d_input, d_sample_y,
/// d_sample_x)`, with the coordinate gradients summed over channels.
pub fn backward(
    input: &[f32],
    sample_y: &[f32],
    sample_x: &[f32],
    output_grad: &[f32],
    geometry: &BilinearGatherGeometry,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    geometry.checked_products()?;
    if input.len() != geometry.input_len()
        || sample_y.len() != geometry.sample_len()
        || sample_x.len() != geometry.sample_len()
        || output_grad.len() != geometry.output_len()
    {
        return Err(
            "deformable gather backward operand length does not match its geometry".to_owned(),
        );
    }
    let mut d_input = vec![0.0f32; geometry.input_len()];
    let mut d_y = vec![0.0f32; geometry.sample_len()];
    let mut d_x = vec![0.0f32; geometry.sample_len()];
    for b in 0..geometry.batch {
        for s in 0..geometry.samples {
            for oy in 0..geometry.output_height {
                for ox in 0..geometry.output_width {
                    let sample = geometry.sample_index(b, s, oy, ox);
                    let y = sample_y[sample];
                    let x = sample_x[sample];
                    let cell = resolve_cell(y, x, geometry.input_height, geometry.input_width);
                    if !cell.active {
                        continue;
                    }
                    let y0f = y.floor();
                    let x0f = x.floor();
                    let (ly, lx) = (y - y0f, x - x0f);
                    let (hy, hx) = (1.0 - ly, 1.0 - lx);
                    let mut acc_y = 0.0f32;
                    let mut acc_x = 0.0f32;
                    for c in 0..geometry.channels {
                        let out = geometry.output_index(b, s, c, oy, ox);
                        let grad = output_grad[out];
                        // Corner values are per channel: the coordinate
                        // derivative of this channel's sampled value is what
                        // that channel's gradient weights.
                        let mut values = [0.0f32; 4];
                        for (slot, (cy, cx, _)) in cell.corners.iter().enumerate() {
                            if inside(*cy, *cx, geometry.input_height, geometry.input_width) {
                                let index = geometry.input_index(b, c, *cy as usize, *cx as usize);
                                values[slot] = input[index];
                            }
                        }
                        if grad == 0.0 {
                            continue;
                        }
                        for (cy, cx, weight) in cell.corners {
                            if inside(cy, cx, geometry.input_height, geometry.input_width) {
                                let index = geometry.input_index(b, c, cy as usize, cx as usize);
                                d_input[index] += grad * weight;
                            }
                        }
                        let [v00, v01, v10, v11] = values;
                        acc_y += grad * (-hx * v00 - lx * v01 + hx * v10 + lx * v11);
                        acc_x += grad * (-hy * v00 + hy * v01 - ly * v10 + ly * v11);
                    }
                    d_y[sample] = acc_y;
                    d_x[sample] = acc_x;
                }
            }
        }
    }
    Ok((d_input, d_y, d_x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> BilinearGatherGeometry {
        BilinearGatherGeometry::new(1, 2, 2, 2, 1, 2, 2).expect("valid geometry")
    }

    #[test]
    fn integer_samples_copy_each_channel_and_zero_fill_outside() {
        let g = BilinearGatherGeometry::new(1, 2, 2, 2, 1, 2, 2).unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        // Exact pixel (1,1) in channel 0/1, then a far-outside sample.
        let sample_y = vec![1.0, -2.0, 0.0, 0.0];
        let sample_x = vec![1.0, 0.0, 0.0, 0.0];
        let out = forward(&input, &sample_y, &sample_x, &g).unwrap();
        assert_eq!(out[g.output_index(0, 0, 0, 0, 0)], 4.0);
        assert_eq!(out[g.output_index(0, 0, 1, 0, 0)], 40.0);
        assert_eq!(out[g.output_index(0, 0, 0, 0, 1)], 0.0);
        assert_eq!(out[g.output_index(0, 0, 0, 1, 0)], 1.0);
    }

    #[test]
    fn fractional_centre_averages_the_four_corners() {
        let g = geometry();
        let input = vec![0.0, 10.0, 20.0, 30.0, 0.0, 100.0, 200.0, 300.0];
        let sample_y = vec![0.5; 4];
        let sample_x = vec![0.5; 4];
        let out = forward(&input, &sample_y, &sample_x, &g).unwrap();
        // Only the top-left output cell has all four corners inside.
        assert!((out[g.output_index(0, 0, 0, 0, 0)] - 15.0).abs() < 1e-6);
        assert!((out[g.output_index(0, 0, 1, 0, 0)] - 150.0).abs() < 1e-6);
    }

    #[test]
    fn far_outside_samples_have_zero_coordinate_gradients() {
        let g = geometry();
        let input = vec![1.0; 8];
        let sample_y = vec![-2.0; 4];
        let sample_x = vec![0.0; 4];
        let grad = vec![1.0; 8];
        let (d_input, d_y, d_x) = backward(&input, &sample_y, &sample_x, &grad, &g).unwrap();
        assert!(d_input.iter().all(|&v| v == 0.0));
        assert!(d_y.iter().all(|&v| v == 0.0));
        assert!(d_x.iter().all(|&v| v == 0.0));
    }

    /// Deterministic LCG so the finite-difference check needs no dev-dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (u32::MAX >> 1) as f32 - 1.0
        }
    }

    #[test]
    fn backward_matches_finite_differences_of_forward() {
        let g = BilinearGatherGeometry::new(2, 3, 3, 4, 2, 2, 3).unwrap();
        let mut rng = Lcg(0x5eed);
        let input: Vec<f32> = (0..g.input_len()).map(|_| rng.next()).collect();
        // Keep every fractional part in [0.3, 0.7) and every coordinate inside
        // the feature map: the floor-cell convention is one-sided at integers
        // and the far-outside early-out is a cliff, so a central difference
        // with eps = 1e-2 must not cross either discontinuity.
        let mut coordinate = |extent: usize| -> Vec<f32> {
            (0..g.sample_len())
                .map(|i| {
                    let base = (i % extent) as f32;
                    base + 0.3 + 0.4 * (rng.next() * 0.5 + 0.5)
                })
                .collect()
        };
        let sample_y = coordinate(g.input_height);
        let sample_x = coordinate(g.input_width);
        let seed: Vec<f32> = (0..g.output_len()).map(|_| rng.next()).collect();

        let objective = |input: &[f32], sy: &[f32], sx: &[f32]| -> f32 {
            forward(input, sy, sx, &g)
                .unwrap()
                .iter()
                .zip(&seed)
                .map(|(v, s)| v * s)
                .sum()
        };
        let (d_input, d_y, d_x) = backward(&input, &sample_y, &sample_x, &seed, &g).unwrap();

        let eps = 1e-2f32;
        for i in (0..g.input_len()).step_by(7) {
            let mut plus = input.clone();
            plus[i] += eps;
            let mut minus = input.clone();
            minus[i] -= eps;
            let fd = (objective(&plus, &sample_y, &sample_x)
                - objective(&minus, &sample_y, &sample_x))
                / (2.0 * eps);
            assert!(
                (d_input[i] - fd).abs() < 2e-3,
                "d_input[{i}]: analytic {} vs fd {fd}",
                d_input[i]
            );
        }
        for i in 0..g.sample_len() {
            let mut plus = sample_y.clone();
            plus[i] += eps;
            let mut minus = sample_y.clone();
            minus[i] -= eps;
            let fd = (objective(&input, &plus, &sample_x) - objective(&input, &minus, &sample_x))
                / (2.0 * eps);
            assert!(
                (d_y[i] - fd).abs() < 5e-3,
                "d_sample_y[{i}]: analytic {} vs fd {fd}",
                d_y[i]
            );

            let mut plus = sample_x.clone();
            plus[i] += eps;
            let mut minus = sample_x.clone();
            minus[i] -= eps;
            let fd = (objective(&input, &sample_y, &plus) - objective(&input, &sample_y, &minus))
                / (2.0 * eps);
            assert!(
                (d_x[i] - fd).abs() < 5e-3,
                "d_sample_x[{i}]: analytic {} vs fd {fd}",
                d_x[i]
            );
        }
    }

    #[test]
    fn coordinate_gradients_are_summed_over_channels() {
        let g = BilinearGatherGeometry::new(1, 2, 2, 2, 1, 1, 1).unwrap();
        // Asymmetric fixture so a dy/dx swap or a per-channel overwrite fails.
        let input = vec![0.0, 1.0, 2.0, 3.0, 0.0, 10.0, 20.0, 30.0];
        let sample_y = vec![0.5];
        let sample_x = vec![0.5];
        let grad = vec![1.0, 1.0];
        let (_, d_y, d_x) = backward(&input, &sample_y, &sample_x, &grad, &g).unwrap();
        // Per channel: dvalue/dy = 0.5*(v10 - v00) + 0.5*(v11 - v01).
        // Channel 0: 0.5*(2-0) + 0.5*(3-1) = 2. Channel 1: 0.5*(20-0) + 0.5*(30-10) = 20.
        // dvalue/dx = 0.5*(v01 - v00) + 0.5*(v11 - v10): channel 0 = 1, channel 1 = 10.
        assert!((d_y[0] - 22.0).abs() < 1e-6, "d_y = {}", d_y[0]);
        assert!((d_x[0] - 11.0).abs() < 1e-6, "d_x = {}", d_x[0]);
    }

    #[test]
    fn rejects_mismatched_operands_and_degenerate_geometry() {
        let g = geometry();
        let input = vec![0.0; g.input_len()];
        let sample = vec![0.0; g.sample_len()];
        assert!(forward(&input, &sample, &sample[1..], &g).is_err());
        assert!(backward(&input, &sample, &sample, &sample[1..], &g).is_err());
        assert!(BilinearGatherGeometry::new(0, 1, 1, 1, 1, 1, 1).is_err());
        assert!(BilinearGatherGeometry::new(1, 1, 1, 1, 0, 1, 1).is_err());
    }
}
