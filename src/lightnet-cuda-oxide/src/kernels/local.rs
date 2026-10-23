//! Locally connected forward and backward kernels (location-specific kernels).

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn local_fwd(
        x: &[f32],
        weights: &[f32],
        biases: &[f32],
        mut output: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        filters: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let locations = out_h * out_w;
        if i >= batch * filters * locations {
            return;
        }
        let location = i % locations;
        let ox = location % out_w;
        let oy = location / out_w;
        let filter = (i / locations) % filters;
        let b = i / (locations * filters);
        let kernel = channels * size * size;
        let weight_base = (location * filters + filter) * kernel;
        let mut sum = biases[location * filters + filter];
        let mut c = 0usize;
        while c < channels {
            let mut ky = 0usize;
            while ky < size {
                let iy = (oy * stride_y + ky) as i64 - padding as i64;
                if iy >= 0 && iy < height as i64 {
                    let mut kx = 0usize;
                    while kx < size {
                        let ix = (ox * stride_x + kx) as i64 - padding as i64;
                        if ix >= 0 && ix < width as i64 {
                            let ki = (c * size + ky) * size + kx;
                            let xi =
                                ((b * channels + c) * height + iy as usize) * width + ix as usize;
                            sum += x[xi] * weights[weight_base + ki];
                        }
                        kx += 1;
                    }
                }
                ky += 1;
            }
            c += 1;
        }
        if let Some(slot) = output.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    pub fn local_dgrad(
        delta: &[f32],
        weights: &[f32],
        mut dx: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        filters: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= batch * channels * height * width {
            return;
        }
        let ix = i % width;
        let iy = (i / width) % height;
        let c = (i / (width * height)) % channels;
        let b = i / (width * height * channels);
        let locations = out_h * out_w;
        let kernel = channels * size * size;
        let mut sum = 0.0f32;
        let mut ky = 0usize;
        while ky < size {
            let ny = iy as i64 + padding as i64 - ky as i64;
            if ny >= 0 && (ny as usize) % stride_y == 0 {
                let oy = ny as usize / stride_y;
                if oy < out_h {
                    let mut kx = 0usize;
                    while kx < size {
                        let nx = ix as i64 + padding as i64 - kx as i64;
                        if nx >= 0 && (nx as usize) % stride_x == 0 {
                            let ox = nx as usize / stride_x;
                            if ox < out_w {
                                let location = oy * out_w + ox;
                                let ki = (c * size + ky) * size + kx;
                                let mut f = 0usize;
                                while f < filters {
                                    let yi = (b * filters + f) * locations + location;
                                    let wi = (location * filters + f) * kernel + ki;
                                    sum += delta[yi] * weights[wi];
                                    f += 1;
                                }
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(slot) = dx.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    pub fn local_wgrad(
        x: &[f32],
        delta: &[f32],
        mut dw: DisjointSlice<f32>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        filters: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let locations = out_h * out_w;
        let kernel = channels * size * size;
        if i >= locations * filters * kernel {
            return;
        }
        let ki = i % kernel;
        let filter = (i / kernel) % filters;
        let location = i / (kernel * filters);
        let ox = location % out_w;
        let oy = location / out_w;
        let c = ki / (size * size);
        let ky = (ki / size) % size;
        let kx = ki % size;
        let iy = (oy * stride_y + ky) as i64 - padding as i64;
        let ix = (ox * stride_x + kx) as i64 - padding as i64;
        let mut sum = 0.0f32;
        if iy >= 0 && iy < height as i64 && ix >= 0 && ix < width as i64 {
            let mut b = 0usize;
            while b < batch {
                let xi = ((b * channels + c) * height + iy as usize) * width + ix as usize;
                let yi = (b * filters + filter) * locations + location;
                sum += x[xi] * delta[yi];
                b += 1;
            }
        }
        if let Some(slot) = dw.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    pub fn local_bgrad(
        delta: &[f32],
        mut db: DisjointSlice<f32>,
        batch: usize,
        filters: usize,
        locations: usize,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= filters * locations {
            return;
        }
        let location = i / filters;
        let filter = i % filters;
        let mut sum = 0.0f32;
        let mut b = 0usize;
        while b < batch {
            sum += delta[(b * filters + filter) * locations + location];
            b += 1;
        }
        if let Some(slot) = db.get_mut(idx) {
            *slot = sum;
        }
    }
}
