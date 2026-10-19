//! Source-derived CPU math for WMHF channel routing and wavelet/scan branches.

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HaarCoefficients {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub height2: usize,
    pub width2: usize,
    pub ll: Vec<f32>,
    pub lh: Vec<f32>,
    pub hl: Vec<f32>,
    pub hh: Vec<f32>,
}

impl HaarCoefficients {
    pub(crate) fn zeros(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Self> {
        let (height2, width2) = half_shape(height, width)?;
        let len = checked_product(&[batch, channels, height2, width2], "wavelet")?;
        Ok(Self {
            batch,
            channels,
            height,
            width,
            height2,
            width2,
            ll: vec![0.0; len],
            lh: vec![0.0; len],
            hl: vec![0.0; len],
            hh: vec![0.0; len],
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FuseConcatGradients {
    pub projected: Vec<f32>,
    pub local: Vec<f32>,
    pub global: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GateGradients {
    pub fuse: Vec<f32>,
    pub gate: Vec<f32>,
    pub projected: Vec<f32>,
    pub shortcut: Option<Vec<f32>>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[wmhf]: {}", message.into()))
}

fn checked_product(dims: &[usize], what: &str) -> Result<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .ok_or_else(|| invalid(format!("{what} element count overflows")))
    })
}

fn half_shape(height: usize, width: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(invalid("input height and width must be positive"));
    }
    Ok((height / 2 + height % 2, width / 2 + width % 2))
}

fn expect_len(values: &[f32], expected: usize, what: &str) -> Result<()> {
    if values.len() != expected {
        return Err(invalid(format!(
            "{what} length mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    Ok(())
}

fn nchw_len(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    what: &str,
) -> Result<usize> {
    if batch == 0 || channels == 0 || height == 0 || width == 0 {
        return Err(invalid(format!("{what} dimensions must be positive")));
    }
    checked_product(&[batch, channels, height, width], what)
}

fn spatial_len(batch: usize, channels: usize, spatial: usize, what: &str) -> Result<usize> {
    if batch == 0 || channels == 0 || spatial == 0 {
        return Err(invalid(format!("{what} dimensions must be positive")));
    }
    checked_product(&[batch, channels, spatial], what)
}

#[inline]
fn index4(
    batch: usize,
    channel: usize,
    y: usize,
    x: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * channels + channel) * height + y) * width + x
}

#[inline]
fn index4_spatial(
    batch: usize,
    channel: usize,
    spatial: usize,
    channels: usize,
    area: usize,
) -> usize {
    (batch * channels + channel) * area + spatial
}

fn validate_coefficients(coeffs: &HaarCoefficients) -> Result<()> {
    let (height2, width2) = half_shape(coeffs.height, coeffs.width)?;
    if coeffs.batch == 0
        || coeffs.channels == 0
        || coeffs.height2 != height2
        || coeffs.width2 != width2
    {
        return Err(invalid("wavelet coefficient geometry is inconsistent"));
    }
    let len = checked_product(&[coeffs.batch, coeffs.channels, height2, width2], "wavelet")?;
    expect_len(&coeffs.ll, len, "LL coefficients")?;
    expect_len(&coeffs.lh, len, "LH coefficients")?;
    expect_len(&coeffs.hl, len, "HL coefficients")?;
    expect_len(&coeffs.hh, len, "HH coefficients")?;
    Ok(())
}

pub(crate) fn split_channels(
    input: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    begin: usize,
    count: usize,
) -> Result<Vec<f32>> {
    let input_len = spatial_len(batch, channels, spatial, "channel split input")?;
    expect_len(input, input_len, "channel split input")?;
    let end = begin
        .checked_add(count)
        .ok_or_else(|| invalid("channel split range overflows"))?;
    if count == 0 || end > channels {
        return Err(invalid("channel split range is empty or out of bounds"));
    }
    let mut output = vec![0.0; spatial_len(batch, count, spatial, "channel split output")?];
    let src_batch = channels * spatial;
    let dst_batch = count * spatial;
    for b in 0..batch {
        let src = b * src_batch + begin * spatial;
        let dst = b * dst_batch;
        output[dst..dst + dst_batch].copy_from_slice(&input[src..src + dst_batch]);
    }
    Ok(output)
}

pub(crate) fn insert_channels(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    output_channels: usize,
    input_channels: usize,
    spatial: usize,
    begin: usize,
    scale: f32,
) -> Result<()> {
    let output_len = spatial_len(batch, output_channels, spatial, "channel insert output")?;
    let input_len = spatial_len(batch, input_channels, spatial, "channel insert input")?;
    expect_len(output, output_len, "channel insert output")?;
    expect_len(input, input_len, "channel insert input")?;
    let end = begin
        .checked_add(input_channels)
        .ok_or_else(|| invalid("channel insert range overflows"))?;
    if end > output_channels {
        return Err(invalid("channel insert range is out of bounds"));
    }
    let src_batch = input_channels * spatial;
    let dst_batch = output_channels * spatial;
    for b in 0..batch {
        for c in 0..input_channels {
            let src = b * src_batch + c * spatial;
            let dst = b * dst_batch + (begin + c) * spatial;
            for i in 0..spatial {
                output[dst + i] += scale * input[src + i];
            }
        }
    }
    Ok(())
}

pub(crate) fn concat_local(
    branch3: &[f32],
    branch5: &[f32],
    branch7: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> Result<Vec<f32>> {
    let branch_len = spatial_len(batch, channels, spatial, "local branch")?;
    expect_len(branch3, branch_len, "3x local branch")?;
    expect_len(branch5, branch_len, "5x local branch")?;
    expect_len(branch7, branch_len, "7x local branch")?;
    let batch_branch = channels * spatial;
    let mut output = vec![0.0; checked_product(&[batch, 3, batch_branch], "local concatenation")?];
    for b in 0..batch {
        let src = b * batch_branch;
        let dst = b * 3 * batch_branch;
        output[dst..dst + batch_branch].copy_from_slice(&branch3[src..src + batch_branch]);
        output[dst + batch_branch..dst + 2 * batch_branch]
            .copy_from_slice(&branch5[src..src + batch_branch]);
        output[dst + 2 * batch_branch..dst + 3 * batch_branch]
            .copy_from_slice(&branch7[src..src + batch_branch]);
    }
    Ok(output)
}

pub(crate) fn split_local_gradient(
    concatenated: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let branch_len = spatial_len(batch, channels, spatial, "local branch gradient")?;
    let total = checked_product(&[3, branch_len], "local concatenation gradient")?;
    expect_len(concatenated, total, "local concatenation gradient")?;
    let mut branch3 = vec![0.0; branch_len];
    let mut branch5 = vec![0.0; branch_len];
    let mut branch7 = vec![0.0; branch_len];
    for b in 0..batch {
        let dst = b * channels * spatial;
        let src = b * 3 * channels * spatial;
        let branch = channels * spatial;
        branch3[dst..dst + branch].copy_from_slice(&concatenated[src..src + branch]);
        branch5[dst..dst + branch].copy_from_slice(&concatenated[src + branch..src + 2 * branch]);
        branch7[dst..dst + branch]
            .copy_from_slice(&concatenated[src + 2 * branch..src + 3 * branch]);
    }
    Ok((branch3, branch5, branch7))
}

pub(crate) fn concat_fuse(
    projected: &[f32],
    local: &[f32],
    global: &[f32],
    batch: usize,
    filters: usize,
    identity_channels: usize,
    local_channels: usize,
    spatial: usize,
) -> Result<Vec<f32>> {
    let global_channels = filters
        .checked_sub(identity_channels)
        .and_then(|remaining| remaining.checked_sub(local_channels))
        .ok_or_else(|| invalid("fuse channel split exceeds filter count"))?;
    if identity_channels + local_channels + global_channels != filters
        || local_channels == 0
        || global_channels == 0
    {
        return Err(invalid("invalid fuse channel split"));
    }
    expect_len(
        projected,
        spatial_len(batch, filters, spatial, "projected features")?,
        "projected features",
    )?;
    expect_len(
        local,
        spatial_len(batch, local_channels, spatial, "local features")?,
        "local features",
    )?;
    expect_len(
        global,
        spatial_len(batch, global_channels, spatial, "global features")?,
        "global features",
    )?;
    let mut output = vec![0.0; spatial_len(batch, filters, spatial, "fuse features")?];
    let src_projected_batch = filters * spatial;
    let src_local_batch = local_channels * spatial;
    let src_global_batch = global_channels * spatial;
    for b in 0..batch {
        let dst = b * src_projected_batch;
        let proj = b * src_projected_batch;
        let local_src = b * src_local_batch;
        let global_src = b * src_global_batch;
        let id_len = identity_channels * spatial;
        let local_len = local_channels * spatial;
        let global_len = global_channels * spatial;
        output[dst..dst + id_len].copy_from_slice(&projected[proj..proj + id_len]);
        output[dst + id_len..dst + id_len + local_len]
            .copy_from_slice(&local[local_src..local_src + local_len]);
        output[dst + id_len + local_len..dst + id_len + local_len + global_len]
            .copy_from_slice(&global[global_src..global_src + global_len]);
    }
    Ok(output)
}

pub(crate) fn split_fuse_gradient(
    gradient: &[f32],
    batch: usize,
    filters: usize,
    identity_channels: usize,
    local_channels: usize,
    spatial: usize,
) -> Result<FuseConcatGradients> {
    let global_channels = filters
        .checked_sub(identity_channels)
        .and_then(|remaining| remaining.checked_sub(local_channels))
        .ok_or_else(|| invalid("fuse gradient split exceeds filter count"))?;
    if local_channels == 0 || global_channels == 0 {
        return Err(invalid("invalid fuse gradient split"));
    }
    expect_len(
        gradient,
        spatial_len(batch, filters, spatial, "fuse gradient")?,
        "fuse gradient",
    )?;
    let mut projected = vec![0.0; spatial_len(batch, filters, spatial, "projected gradient")?];
    let mut local = vec![0.0; spatial_len(batch, local_channels, spatial, "local gradient")?];
    let mut global = vec![0.0; spatial_len(batch, global_channels, spatial, "global gradient")?];
    let filters_batch = filters * spatial;
    let local_batch = local_channels * spatial;
    let global_batch = global_channels * spatial;
    let id_len = identity_channels * spatial;
    let local_len = local_channels * spatial;
    let global_len = global_channels * spatial;
    for b in 0..batch {
        let src = b * filters_batch;
        let proj = b * filters_batch;
        let local_dst = b * local_batch;
        let global_dst = b * global_batch;
        projected[proj..proj + id_len].copy_from_slice(&gradient[src..src + id_len]);
        local[local_dst..local_dst + local_len]
            .copy_from_slice(&gradient[src + id_len..src + id_len + local_len]);
        global[global_dst..global_dst + global_len].copy_from_slice(
            &gradient[src + id_len + local_len..src + id_len + local_len + global_len],
        );
    }
    Ok(FuseConcatGradients {
        projected,
        local,
        global,
    })
}

pub(crate) fn haar_dwt(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<HaarCoefficients> {
    expect_len(
        input,
        nchw_len(batch, channels, height, width, "DWT input")?,
        "DWT input",
    )?;
    let mut coeffs = HaarCoefficients::zeros(batch, channels, height, width)?;
    let area2 = coeffs.height2 * coeffs.width2;
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..coeffs.height2 {
                let y0 = y.saturating_mul(2).min(height - 1);
                let y1 = y0.saturating_add(1).min(height - 1);
                for x in 0..coeffs.width2 {
                    let x0 = x.saturating_mul(2).min(width - 1);
                    let x1 = x0.saturating_add(1).min(width - 1);
                    let v00 = input[index4(b, c, y0, x0, channels, height, width)];
                    let v01 = input[index4(b, c, y0, x1, channels, height, width)];
                    let v10 = input[index4(b, c, y1, x0, channels, height, width)];
                    let v11 = input[index4(b, c, y1, x1, channels, height, width)];
                    let o = index4_spatial(b, c, y * coeffs.width2 + x, channels, area2);
                    coeffs.ll[o] = 0.5 * (v00 + v01 + v10 + v11);
                    coeffs.lh[o] = 0.5 * (v00 - v01 + v10 - v11);
                    coeffs.hl[o] = 0.5 * (v00 + v01 - v10 - v11);
                    coeffs.hh[o] = 0.5 * (v00 - v01 - v10 + v11);
                }
            }
        }
    }
    Ok(coeffs)
}

pub(crate) fn haar_dwt_backward(coeff_grads: &HaarCoefficients) -> Result<Vec<f32>> {
    validate_coefficients(coeff_grads)?;
    let mut input_grad = vec![
        0.0;
        nchw_len(
            coeff_grads.batch,
            coeff_grads.channels,
            coeff_grads.height,
            coeff_grads.width,
            "DWT input gradient",
        )?
    ];
    let area2 = coeff_grads.height2 * coeff_grads.width2;
    for b in 0..coeff_grads.batch {
        for c in 0..coeff_grads.channels {
            for y in 0..coeff_grads.height2 {
                let y0 = y.saturating_mul(2).min(coeff_grads.height - 1);
                let y1 = y0.saturating_add(1).min(coeff_grads.height - 1);
                for x in 0..coeff_grads.width2 {
                    let x0 = x.saturating_mul(2).min(coeff_grads.width - 1);
                    let x1 = x0.saturating_add(1).min(coeff_grads.width - 1);
                    let o = index4_spatial(
                        b,
                        c,
                        y * coeff_grads.width2 + x,
                        coeff_grads.channels,
                        area2,
                    );
                    let a = 0.5 * coeff_grads.ll[o];
                    let b1 = 0.5 * coeff_grads.lh[o];
                    let c1 = 0.5 * coeff_grads.hl[o];
                    let d = 0.5 * coeff_grads.hh[o];
                    let i00 = index4(
                        b,
                        c,
                        y0,
                        x0,
                        coeff_grads.channels,
                        coeff_grads.height,
                        coeff_grads.width,
                    );
                    let i01 = index4(
                        b,
                        c,
                        y0,
                        x1,
                        coeff_grads.channels,
                        coeff_grads.height,
                        coeff_grads.width,
                    );
                    let i10 = index4(
                        b,
                        c,
                        y1,
                        x0,
                        coeff_grads.channels,
                        coeff_grads.height,
                        coeff_grads.width,
                    );
                    let i11 = index4(
                        b,
                        c,
                        y1,
                        x1,
                        coeff_grads.channels,
                        coeff_grads.height,
                        coeff_grads.width,
                    );
                    input_grad[i00] += a + b1 + c1 + d;
                    input_grad[i01] += a - b1 + c1 - d;
                    input_grad[i10] += a + b1 - c1 - d;
                    input_grad[i11] += a - b1 - c1 + d;
                }
            }
        }
    }
    Ok(input_grad)
}

pub(crate) fn haar_iwt(coeffs: &HaarCoefficients) -> Result<Vec<f32>> {
    validate_coefficients(coeffs)?;
    let mut output = vec![
        0.0;
        nchw_len(
            coeffs.batch,
            coeffs.channels,
            coeffs.height,
            coeffs.width,
            "IWT output"
        )?
    ];
    let area2 = coeffs.height2 * coeffs.width2;
    for b in 0..coeffs.batch {
        for c in 0..coeffs.channels {
            for y in 0..coeffs.height2 {
                let y0 = y.saturating_mul(2).min(coeffs.height - 1);
                let y1 = y0.saturating_add(1).min(coeffs.height - 1);
                for x in 0..coeffs.width2 {
                    let x0 = x.saturating_mul(2).min(coeffs.width - 1);
                    let x1 = x0.saturating_add(1).min(coeffs.width - 1);
                    let o = index4_spatial(b, c, y * coeffs.width2 + x, coeffs.channels, area2);
                    let ll = coeffs.ll[o];
                    let lh = coeffs.lh[o];
                    let hl = coeffs.hl[o];
                    let hh = coeffs.hh[o];
                    output[index4(b, c, y0, x0, coeffs.channels, coeffs.height, coeffs.width)] +=
                        0.5 * (ll + lh + hl + hh);
                    output[index4(b, c, y0, x1, coeffs.channels, coeffs.height, coeffs.width)] +=
                        0.5 * (ll - lh + hl - hh);
                    output[index4(b, c, y1, x0, coeffs.channels, coeffs.height, coeffs.width)] +=
                        0.5 * (ll + lh - hl - hh);
                    output[index4(b, c, y1, x1, coeffs.channels, coeffs.height, coeffs.width)] +=
                        0.5 * (ll - lh - hl + hh);
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn haar_iwt_backward(
    output_grad: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<HaarCoefficients> {
    expect_len(
        output_grad,
        nchw_len(batch, channels, height, width, "IWT output gradient")?,
        "IWT output gradient",
    )?;
    let mut coeff_grads = HaarCoefficients::zeros(batch, channels, height, width)?;
    let area2 = coeff_grads.height2 * coeff_grads.width2;
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..coeff_grads.height2 {
                let y0 = y.saturating_mul(2).min(height - 1);
                let y1 = y0.saturating_add(1).min(height - 1);
                for x in 0..coeff_grads.width2 {
                    let x0 = x.saturating_mul(2).min(width - 1);
                    let x1 = x0.saturating_add(1).min(width - 1);
                    let o = index4_spatial(b, c, y * coeff_grads.width2 + x, channels, area2);
                    let d00 = output_grad[index4(b, c, y0, x0, channels, height, width)];
                    let d01 = output_grad[index4(b, c, y0, x1, channels, height, width)];
                    let d10 = output_grad[index4(b, c, y1, x0, channels, height, width)];
                    let d11 = output_grad[index4(b, c, y1, x1, channels, height, width)];
                    coeff_grads.ll[o] += 0.5 * (d00 + d01 + d10 + d11);
                    coeff_grads.lh[o] += 0.5 * (d00 - d01 + d10 - d11);
                    coeff_grads.hl[o] += 0.5 * (d00 + d01 - d10 - d11);
                    coeff_grads.hh[o] += 0.5 * (d00 - d01 - d10 + d11);
                }
            }
        }
    }
    Ok(coeff_grads)
}

pub(crate) fn scan4_forward(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>> {
    expect_len(
        input,
        nchw_len(batch, channels, height, width, "scan input")?,
        "scan input",
    )?;
    expect_len(
        weights,
        checked_product(&[4, channels], "scan weights")?,
        "scan weights",
    )?;
    let mut output = vec![0.0; input.len()];
    let mut indices = Vec::with_capacity(height.max(width));
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                indices.clear();
                indices.extend((0..width).map(|x| y * width + x));
                scan_sequence_forward(
                    input,
                    &mut output,
                    weights,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
                indices.clear();
                indices.extend((0..width).rev().map(|x| y * width + x));
                scan_sequence_forward(
                    input,
                    &mut output,
                    weights,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
            }
            for x in 0..width {
                indices.clear();
                indices.extend((0..height).map(|y| y * width + x));
                scan_sequence_forward(
                    input,
                    &mut output,
                    weights,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
                indices.clear();
                indices.extend((0..height).rev().map(|y| y * width + x));
                scan_sequence_forward(
                    input,
                    &mut output,
                    weights,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
            }
        }
    }
    Ok(output)
}

fn scan_sequence_forward(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    _batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    batch: usize,
    channel: usize,
    sequence: &[usize],
) {
    let raw_a = weights[channel];
    let a = 0.98 * stable_sigmoid(raw_a);
    let b_weight = weights[channels + channel];
    let c_weight = weights[2 * channels + channel];
    let d_weight = weights[3 * channels + channel];
    let mut state = 0.0;
    let area = height * width;
    for &spatial in sequence {
        let index = index4_spatial(batch, channel, spatial, channels, area);
        let value = input[index];
        state = a * state + b_weight * value;
        output[index] += 0.25 * (c_weight * state + d_weight * value);
    }
}

pub(crate) fn scan4_backward(
    input: &[f32],
    output_grad: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let len = nchw_len(batch, channels, height, width, "scan input")?;
    expect_len(input, len, "scan input")?;
    expect_len(output_grad, len, "scan output gradient")?;
    expect_len(
        weights,
        checked_product(&[4, channels], "scan weights")?,
        "scan weights",
    )?;
    let mut input_grad = vec![0.0; len];
    let mut weight_grad = vec![0.0; 4 * channels];
    let mut indices = Vec::with_capacity(height.max(width));
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                indices.clear();
                indices.extend((0..width).map(|x| y * width + x));
                scan_sequence_backward(
                    input,
                    output_grad,
                    weights,
                    &mut weight_grad,
                    &mut input_grad,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
                indices.clear();
                indices.extend((0..width).rev().map(|x| y * width + x));
                scan_sequence_backward(
                    input,
                    output_grad,
                    weights,
                    &mut weight_grad,
                    &mut input_grad,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
            }
            for x in 0..width {
                indices.clear();
                indices.extend((0..height).map(|y| y * width + x));
                scan_sequence_backward(
                    input,
                    output_grad,
                    weights,
                    &mut weight_grad,
                    &mut input_grad,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
                indices.clear();
                indices.extend((0..height).rev().map(|y| y * width + x));
                scan_sequence_backward(
                    input,
                    output_grad,
                    weights,
                    &mut weight_grad,
                    &mut input_grad,
                    batch,
                    channels,
                    height,
                    width,
                    b,
                    c,
                    &indices,
                );
            }
        }
    }
    Ok((input_grad, weight_grad))
}

fn scan_sequence_backward(
    input: &[f32],
    output_grad: &[f32],
    weights: &[f32],
    weight_grad: &mut [f32],
    input_grad: &mut [f32],
    _batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    batch: usize,
    channel: usize,
    sequence: &[usize],
) {
    if sequence.is_empty() {
        return;
    }
    let raw_a = weights[channel];
    let sigmoid = stable_sigmoid(raw_a);
    let a = 0.98 * sigmoid;
    let b_weight = weights[channels + channel];
    let c_weight = weights[2 * channels + channel];
    let d_weight = weights[3 * channels + channel];
    let area = height * width;
    let mut states = vec![0.0; sequence.len()];
    let mut values = vec![0.0; sequence.len()];
    let mut state = 0.0;
    for (t, &spatial) in sequence.iter().enumerate() {
        let index = index4_spatial(batch, channel, spatial, channels, area);
        let value = input[index];
        values[t] = value;
        state = a * state + b_weight * value;
        states[t] = state;
    }
    let (mut da, mut db, mut dc, mut dd, mut ds_next) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for t in (0..sequence.len()).rev() {
        let index = index4_spatial(batch, channel, sequence[t], channels, area);
        let dy = 0.25 * output_grad[index];
        let value = values[t];
        let state_t = states[t];
        let state_prev = if t == 0 { 0.0 } else { states[t - 1] };
        dc += dy * state_t;
        dd += dy * value;
        let ds = ds_next + dy * c_weight;
        da += ds * state_prev;
        db += ds * value;
        input_grad[index] += dy * d_weight + ds * b_weight;
        ds_next = ds * a;
    }
    weight_grad[channel] += da * 0.98 * sigmoid * (1.0 - sigmoid);
    weight_grad[channels + channel] += db;
    weight_grad[2 * channels + channel] += dc;
    weight_grad[3 * channels + channel] += dd;
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

pub(crate) fn high_frequency_energy_upsample(coeffs: &HaarCoefficients) -> Result<Vec<f32>> {
    validate_coefficients(coeffs)?;
    let mut output = vec![
        0.0;
        nchw_len(
            coeffs.batch,
            coeffs.channels,
            coeffs.height,
            coeffs.width,
            "high-frequency energy"
        )?
    ];
    let area2 = coeffs.height2 * coeffs.width2;
    for b in 0..coeffs.batch {
        for c in 0..coeffs.channels {
            for y in 0..coeffs.height {
                let y2 = (y / 2).min(coeffs.height2 - 1);
                for x in 0..coeffs.width {
                    let x2 = (x / 2).min(coeffs.width2 - 1);
                    let coeff =
                        index4_spatial(b, c, y2 * coeffs.width2 + x2, coeffs.channels, area2);
                    let output_index =
                        index4(b, c, y, x, coeffs.channels, coeffs.height, coeffs.width);
                    output[output_index] =
                        coeffs.lh[coeff].abs() + coeffs.hl[coeff].abs() + coeffs.hh[coeff].abs();
                }
            }
        }
    }
    Ok(output)
}

pub(crate) fn high_frequency_energy_backward(
    output_grad: &[f32],
    coeffs: &HaarCoefficients,
) -> Result<HaarCoefficients> {
    validate_coefficients(coeffs)?;
    expect_len(
        output_grad,
        nchw_len(
            coeffs.batch,
            coeffs.channels,
            coeffs.height,
            coeffs.width,
            "high-frequency energy gradient",
        )?,
        "high-frequency energy gradient",
    )?;
    let mut grads =
        HaarCoefficients::zeros(coeffs.batch, coeffs.channels, coeffs.height, coeffs.width)?;
    let area2 = coeffs.height2 * coeffs.width2;
    for b in 0..coeffs.batch {
        for c in 0..coeffs.channels {
            for y in 0..coeffs.height {
                let y2 = (y / 2).min(coeffs.height2 - 1);
                for x in 0..coeffs.width {
                    let x2 = (x / 2).min(coeffs.width2 - 1);
                    let coeff =
                        index4_spatial(b, c, y2 * coeffs.width2 + x2, coeffs.channels, area2);
                    let upstream = output_grad
                        [index4(b, c, y, x, coeffs.channels, coeffs.height, coeffs.width)];
                    grads.lh[coeff] += upstream * sign_no_zero(coeffs.lh[coeff]);
                    grads.hl[coeff] += upstream * sign_no_zero(coeffs.hl[coeff]);
                    grads.hh[coeff] += upstream * sign_no_zero(coeffs.hh[coeff]);
                }
            }
        }
    }
    Ok(grads)
}

fn sign_no_zero(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
pub(crate) fn apply_gate(
    fuse: &[f32],
    gate: &[f32],
    projected: &[f32],
    shortcut: Option<&[f32]>,
    freq_scale: f32,
) -> Result<Vec<f32>> {
    expect_len(gate, fuse.len(), "gate values")?;
    expect_len(projected, fuse.len(), "projected values")?;
    if let Some(shortcut) = shortcut {
        expect_len(shortcut, fuse.len(), "shortcut values")?;
    }
    Ok((0..fuse.len())
        .map(|i| fuse[i] + freq_scale * gate[i] * projected[i] + shortcut.map_or(0.0, |x| x[i]))
        .collect())
}

pub(crate) fn apply_gate_backward(
    output_grad: &[f32],
    gate: &[f32],
    projected: &[f32],
    use_shortcut: bool,
    freq_scale: f32,
) -> Result<GateGradients> {
    expect_len(gate, output_grad.len(), "gate values")?;
    expect_len(projected, output_grad.len(), "projected values")?;
    let mut gate_grad = vec![0.0; output_grad.len()];
    let mut projected_grad = vec![0.0; output_grad.len()];
    for i in 0..output_grad.len() {
        gate_grad[i] = freq_scale * projected[i] * output_grad[i];
        projected_grad[i] = freq_scale * gate[i] * output_grad[i];
    }
    Ok(GateGradients {
        fuse: output_grad.to_vec(),
        gate: gate_grad,
        projected: projected_grad,
        shortcut: use_shortcut.then(|| output_grad.to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(name: &str, got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{name}: got {got}, want {want}");
    }

    fn assert_slice_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        for (i, (got, want)) in got.iter().zip(want).enumerate() {
            assert_close(&format!("{name}[{i}]"), *got, *want, tol);
        }
    }

    #[test]
    fn wmhf_source_nchw_axes_match_hand_computation() {
        let input = (0..48).map(|v| v as f32).collect::<Vec<_>>();
        let split = split_channels(&input, 2, 4, 6, 1, 2).unwrap();
        assert_slice_close(
            "batch/channel split",
            &split,
            &(6..18).chain(30..42).map(|v| v as f32).collect::<Vec<_>>(),
            0.0,
        );

        let branch = (0..12).map(|v| v as f32).collect::<Vec<_>>();
        let other = branch.iter().map(|v| v + 100.0).collect::<Vec<_>>();
        let third = branch.iter().map(|v| v + 200.0).collect::<Vec<_>>();
        let local = concat_local(&branch, &other, &third, 2, 1, 6).unwrap();
        assert_slice_close(
            "local branch channel order",
            &local,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 200.0,
                201.0, 202.0, 203.0, 204.0, 205.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 106.0, 107.0,
                108.0, 109.0, 110.0, 111.0, 206.0, 207.0, 208.0, 209.0, 210.0, 211.0,
            ],
            0.0,
        );

        let projected = vec![1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 3.0, 4.0];
        let local = vec![5.0, 6.0];
        let global = vec![7.0, 8.0, 9.0, 10.0];
        let fused = concat_fuse(&projected, &local, &global, 1, 4, 1, 1, 2).unwrap();
        assert_slice_close(
            "fuse input identity/local/global order",
            &fused,
            &[1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            0.0,
        );
    }

    #[test]
    fn wmhf_haar_odd_edge_matches_source_hand_values() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let coeffs = haar_dwt(&input, 1, 1, 3, 2).unwrap();
        assert_eq!((coeffs.height2, coeffs.width2), (2, 1));
        assert_slice_close("LL", &coeffs.ll, &[5.0, 11.0], 0.0);
        assert_slice_close("LH", &coeffs.lh, &[-1.0, -1.0], 0.0);
        assert_slice_close("HL", &coeffs.hl, &[-2.0, 0.0], 0.0);
        assert_slice_close("HH", &coeffs.hh, &[0.0, 0.0], 0.0);
        assert_slice_close(
            "IWT accumulates duplicated odd edge",
            &haar_iwt(&coeffs).unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0],
            0.0,
        );
    }

    #[test]
    fn wmhf_haar_forward_and_inverse_backward_are_adjoint_on_odd_shapes() {
        let input = pattern(2 * 2 * 3 * 5, 0.17, -0.31);
        let coeffs = haar_dwt(&input, 2, 2, 3, 5).unwrap();
        let coeff_seed = HaarCoefficients {
            batch: coeffs.batch,
            channels: coeffs.channels,
            height: coeffs.height,
            width: coeffs.width,
            height2: coeffs.height2,
            width2: coeffs.width2,
            ll: pattern(coeffs.ll.len(), 0.11, 0.2),
            lh: pattern(coeffs.lh.len(), -0.13, -0.1),
            hl: pattern(coeffs.hl.len(), 0.07, 0.4),
            hh: pattern(coeffs.hh.len(), 0.19, -0.2),
        };
        let dwt_dx = haar_dwt_backward(&coeff_seed).unwrap();
        let lhs = dot4(&coeffs, &coeff_seed);
        let rhs = dot(&input, &dwt_dx);
        assert_close("DWT adjoint", lhs, rhs, 2.0e-5);

        let synthesis = haar_iwt(&coeff_seed).unwrap();
        let image_seed = pattern(input.len(), -0.09, 0.3);
        let iwt_adjoint = haar_iwt_backward(&image_seed, 2, 2, 3, 5).unwrap();
        assert_close(
            "IWT adjoint",
            dot(&synthesis, &image_seed),
            dot4(&coeff_seed, &iwt_adjoint),
            2.0e-5,
        );
    }

    #[test]
    fn wmhf_four_direction_scan_matches_hand_computation() {
        let input = [1.0, 2.0];
        let weights = [0.0, 2.0, 3.0, 0.25];
        let output = scan4_forward(&input, &weights, 1, 1, 1, 2).unwrap();
        assert_slice_close("row/column scan sum", &output, &[7.72, 13.235], 2.0e-6);
    }

    #[test]
    fn wmhf_scan_backward_matches_finite_differences() {
        let (batch, channels, height, width) = (2, 2, 2, 3);
        let input = pattern(batch * channels * height * width, 0.13, -0.22);
        let weights = pattern(4 * channels, 0.09, 0.31);
        let upstream = pattern(input.len(), -0.07, 0.1);
        let (input_grad, weight_grad) =
            scan4_backward(&input, &upstream, &weights, batch, channels, height, width).unwrap();
        let eps = 1.0e-3;
        for (i, analytic) in input_grad.iter().enumerate() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (scan_loss(&plus, &weights, &upstream, batch, channels, height, width)
                - scan_loss(&minus, &weights, &upstream, batch, channels, height, width))
                / (2.0 * eps);
            assert_close(&format!("scan input grad {i}"), *analytic, fd, 2.0e-3);
        }
        for (i, analytic) in weight_grad.iter().enumerate() {
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (scan_loss(&input, &plus, &upstream, batch, channels, height, width)
                - scan_loss(&input, &minus, &upstream, batch, channels, height, width))
                / (2.0 * eps);
            assert_close(&format!("scan weight grad {i}"), *analytic, fd, 2.0e-3);
        }
    }

    #[test]
    fn wmhf_high_frequency_energy_uses_nchw_nearest_upsampling() {
        let mut coeffs = HaarCoefficients::zeros(1, 1, 3, 3).unwrap();
        coeffs.lh.copy_from_slice(&[1.0, -2.0, 0.5, 3.0]);
        coeffs.hl.copy_from_slice(&[0.0, 1.0, -1.0, -2.0]);
        coeffs.hh.copy_from_slice(&[2.0, 0.0, 0.5, 1.0]);
        let energy = high_frequency_energy_upsample(&coeffs).unwrap();
        assert_slice_close(
            "odd H/W energy replication",
            &energy,
            &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 6.0],
            0.0,
        );

        let upstream = pattern(9, 0.2, -0.1);
        let grads = high_frequency_energy_backward(&upstream, &coeffs).unwrap();
        assert_eq!(grads.ll, vec![0.0; 4]);
        assert_eq!(grads.lh.len(), 4);
        assert_eq!(grads.hl.len(), 4);
        assert_eq!(grads.hh.len(), 4);
    }

    #[test]
    fn wmhf_gate_backward_matches_finite_differences_and_shortcut() {
        let fuse = vec![0.2, -0.4, 0.7];
        let gate = vec![0.3, 0.6, 0.8];
        let projected = vec![-0.5, 0.9, 0.1];
        let shortcut = vec![0.4, 0.1, -0.2];
        let upstream = vec![0.7, -0.3, 0.2];
        let scale = 0.25;
        let out = apply_gate(&fuse, &gate, &projected, Some(&shortcut), scale).unwrap();
        assert_slice_close(
            "gate plus shortcut",
            &out,
            &[
                0.2 + 0.25 * 0.3 * -0.5 + 0.4,
                -0.4 + 0.25 * 0.6 * 0.9 + 0.1,
                0.7 + 0.25 * 0.8 * 0.1 - 0.2,
            ],
            1.0e-7,
        );
        let grads = apply_gate_backward(&upstream, &gate, &projected, true, scale).unwrap();
        assert_slice_close("fuse grad", &grads.fuse, &upstream, 0.0);
        assert_slice_close(
            "gate grad",
            &grads.gate,
            &[
                upstream[0] * scale * projected[0],
                upstream[1] * scale * projected[1],
                upstream[2] * scale * projected[2],
            ],
            0.0,
        );
        assert_slice_close(
            "projected grad",
            &grads.projected,
            &[
                upstream[0] * scale * gate[0],
                upstream[1] * scale * gate[1],
                upstream[2] * scale * gate[2],
            ],
            0.0,
        );
        assert_eq!(grads.shortcut.as_deref(), Some(upstream.as_slice()));
    }

    fn pattern(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 7 % 19) as f32 - 8.0) * scale + offset)
            .collect()
    }

    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn dot4(left: &HaarCoefficients, right: &HaarCoefficients) -> f32 {
        dot(&left.ll, &right.ll)
            + dot(&left.lh, &right.lh)
            + dot(&left.hl, &right.hl)
            + dot(&left.hh, &right.hh)
    }

    fn scan_loss(
        input: &[f32],
        weights: &[f32],
        upstream: &[f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> f32 {
        dot(
            &scan4_forward(input, weights, batch, channels, height, width).unwrap(),
            upstream,
        )
    }
}
