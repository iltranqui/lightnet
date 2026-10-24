//! Softmax forward and backward kernels.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// One thread per softmax group of `group_size` contiguous elements
    /// (`n_groups = batch * groups`): max, exp + sum, normalise.
    #[kernel]
    pub fn softmax_forward(
        x: &[f32],
        mut out: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut max = f32::NEG_INFINITY;
        let mut i = 0usize;
        while i < group_size {
            let v = x[base + i];
            if v > max {
                max = v;
            }
            i += 1;
        }
        let mut sum = 0.0f32;
        i = 0;
        while i < group_size {
            let e = (x[base + i] - max).exp();
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`;
            // the host checks `out` covers `n_groups * group_size`.
            unsafe {
                *out.as_mut_ptr().add(base + i) = e;
            }
            sum += e;
            i += 1;
        }
        if sum > 0.0 {
            i = 0;
            while i < group_size {
                // SAFETY: as above; this thread already wrote the element.
                unsafe {
                    *out.as_mut_ptr().add(base + i) /= sum;
                }
                i += 1;
            }
        }
    }

    /// `dx = y * (dy - sum(y * dy))`, one thread per group.
    #[kernel]
    pub fn softmax_backward(
        y: &[f32],
        dy: &[f32],
        mut dx: DisjointSlice<f32>,
        group_size: usize,
        n_groups: usize,
    ) {
        let g = thread::index_1d().get();
        if g >= n_groups {
            return;
        }
        let base = g * group_size;
        let mut dot = 0.0f32;
        let mut i = 0usize;
        while i < group_size {
            dot += y[base + i] * dy[base + i];
            i += 1;
        }
        i = 0;
        while i < group_size {
            // SAFETY: group `g` exclusively owns `[base, base + group_size)`.
            unsafe {
                *dx.as_mut_ptr().add(base + i) = y[base + i] * (dy[base + i] - dot);
            }
            i += 1;
        }
    }
}
