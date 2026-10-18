//! `[laya_act_head]`: the LAYA act/escalate head (Phase 1b of
//! `tasks/rust/layers.md`), ported from
//! `Jev_laya:src-lib/laya_act_head_layer.cpp`.
//!
//! A small frozen head that turns the scorer's option distribution into a
//! summary feature vector and runs a 2-layer MLP over it:
//!
//! ```text
//! logits = scorer_output
//! p      = softmax(masked_fill(logits, ~marker_mask, -1e4))   over all max_options slots
//! k      = max(marker_mask.sum(), 2)
//! feats  = [top1, top1 - top2, entropy / log(k), k / 255]
//! pooled = head_output[:, :, 0, 0]                            (CLS position)
//! x      = cat(pooled, feats)                                 (hidden + 4)
//! out    = fc2( gelu_erf( fc1(x) ) )
//! ```
//!
//! Exact erf GELU (torch `nn.GELU()`), same as `[laya_scorer]`.  The softmax
//! features are **constants for the backward** (upstream detaches the logits),
//! so only the pooled CLS vector receives a gradient; the scorer and marker
//! inputs get none.
//!
//! Three inputs, route-style `layers=head,scorer,marker`:
//!  * head   `[batch, hidden, 1, T_head]`  (only column 0 is read)
//!  * scorer `[batch, 1, 1, max_options]`  raw logits
//!  * marker `[batch, 2*max_options, 1, T_marker]` (channel `2j+1` is the mask)

use crate::laya_scorer::{blob_float_count as scorer_blob_float_count, gelu_erf, gelu_erf_grad};

/// C++ `LAYA_ACT_HEAD_FC_DIM`.
pub const LAYA_ACT_HEAD_FC_DIM: usize = 256;
/// C++ `LAYA_ACT_HEAD_N_ACT`.
pub const LAYA_ACT_HEAD_N_ACT: usize = 2;
/// C++ `LAYA_ACT_HEAD_N_FEATS`: `[top1, top1-top2, entropy, k/255]`.
pub const LAYA_ACT_HEAD_N_FEATS: usize = 4;

/// C++ `laya_act_head_blob_float_count(hidden, fc_dim, n_act)`.
pub fn blob_float_count(hidden: usize, fc_dim: usize, n_act: usize) -> usize {
    let input = hidden + LAYA_ACT_HEAD_N_FEATS;
    fc_dim * input + fc_dim + n_act * fc_dim + n_act
}

/// Byte offset of the `[laya_act_head]` blob inside the shared `.weights` file:
/// `20` header bytes plus the encoder, head (incl. `type_emb`), scorer blobs and
/// the decision blob's `temperature[3]`.  Mirrors C++ `parse_laya_act_head_section`.
#[allow(clippy::too_many_arguments)]
pub fn blob_offset_bytes(
    enc_hidden: usize,
    enc_layers: usize,
    enc_intermediate: usize,
    enc_vocab: usize,
    head_hidden: usize,
    head_layers: usize,
    head_ffn_dim: usize,
    this_hidden: usize,
    f16: bool,
) -> usize {
    let skip =
        crate::laya_scorer::encoder_blob_float_count(
            enc_hidden,
            enc_layers,
            enc_intermediate,
            enc_vocab,
        ) + crate::laya_scorer::head_blob_float_count(head_hidden, head_layers, head_ffn_dim)
            + scorer_blob_float_count(this_hidden)
            + 3;
    20 + skip * if f16 { 2 } else { 4 }
}

/// The four frozen tensors, in C++ blob order
/// (`act_head.0.weight`, `.0.bias`, `.2.weight`, `.2.bias`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaActHeadWeights {
    /// `[fc_dim, hidden + 4]`, row-major.
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    /// `[n_act, fc_dim]`, row-major.
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl LayaActHeadWeights {
    /// All-zero weights of the right sizes (C++ zero-initialises and expects the
    /// blob loader to fill them).
    pub fn zeroed(hidden: usize, fc_dim: usize, n_act: usize) -> Self {
        let input = hidden + LAYA_ACT_HEAD_N_FEATS;
        Self {
            fc1_w: vec![0.0; fc_dim * input],
            fc1_b: vec![0.0; fc_dim],
            fc2_w: vec![0.0; n_act * fc_dim],
            fc2_b: vec![0.0; n_act],
        }
    }

    /// Per-tensor sizes in blob order.
    pub fn sizes(&self) -> [usize; 4] {
        [
            self.fc1_w.len(),
            self.fc1_b.len(),
            self.fc2_w.len(),
            self.fc2_b.len(),
        ]
    }

    /// Length validation for a given `(hidden, fc_dim, n_act)`.
    pub fn matches(&self, hidden: usize, fc_dim: usize, n_act: usize) -> bool {
        self.sizes() == {
            let input = hidden + LAYA_ACT_HEAD_N_FEATS;
            [fc_dim * input, fc_dim, n_act * fc_dim, n_act]
        }
    }

    /// The blob flattened in C++ read order, for `.weights` round trips.
    pub fn to_blob(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.sizes().iter().sum());
        out.extend_from_slice(&self.fc1_w);
        out.extend_from_slice(&self.fc1_b);
        out.extend_from_slice(&self.fc2_w);
        out.extend_from_slice(&self.fc2_b);
        out
    }

    /// Inverse of [`Self::to_blob`] for the given geometry.
    pub fn from_blob(
        blob: &[f32],
        hidden: usize,
        fc_dim: usize,
        n_act: usize,
    ) -> Result<Self, String> {
        let expected = blob_float_count(hidden, fc_dim, n_act);
        if blob.len() != expected {
            return Err(format!(
                "laya_act_head blob has {} floats, expected {expected} for hidden={hidden} fc_dim={fc_dim} n_act={n_act}",
                blob.len()
            ));
        }
        let input = hidden + LAYA_ACT_HEAD_N_FEATS;
        let mut at = 0usize;
        let mut take = |len: usize| {
            let slice = blob[at..at + len].to_vec();
            at += len;
            slice
        };
        Ok(Self {
            fc1_w: take(fc_dim * input),
            fc1_b: take(fc_dim),
            fc2_w: take(n_act * fc_dim),
            fc2_b: take(n_act),
        })
    }
}

/// C++ `ah_features`: build `x[hidden + 4]` for one sample.
///
/// `head_out` is `[batch, hidden, 1, t_head]`, `scorer_logits` is
/// `[batch, max_options]`, `marker` is `[batch, marker_channels, 1, t_marker]`.
pub fn features_for_sample(
    head_out: &[f32],
    scorer_logits: &[f32],
    marker: &[f32],
    b: usize,
    hidden: usize,
    max_options: usize,
    marker_channels: usize,
    t_head: usize,
    t_marker: usize,
    x: &mut [f32],
) {
    debug_assert_eq!(x.len(), hidden + LAYA_ACT_HEAD_N_FEATS);
    // Masked softmax over all K slots.
    let mut p = vec![0.0f32; max_options];
    let mut n_valid = 0usize;
    for j in 0..max_options {
        let valid = marker[(b * marker_channels + 2 * j + 1) * t_marker] > 0.5;
        if valid {
            n_valid += 1;
        }
        p[j] = if valid {
            scorer_logits[b * max_options + j]
        } else {
            -1.0e4
        };
    }
    // `std::max_element` with no -INFINITY seed (the C++ build uses -Ofast).
    let zmax = p.iter().copied().fold(p[0], f32::max);
    let mut sum = 0.0f32;
    for value in p.iter_mut() {
        *value = (*value - zmax).exp();
        sum += *value;
    }
    let mut top1 = -1.0f32;
    let mut top2 = -1.0f32;
    let mut ent = 0.0f32;
    for value in p.iter_mut() {
        *value /= sum;
        ent -= *value * value.max(1.0e-9).ln();
        if *value > top1 {
            top2 = top1;
            top1 = *value;
        } else if *value > top2 {
            top2 = *value;
        }
    }
    let k = n_valid.max(2) as f32;

    for e in 0..hidden {
        x[e] = head_out[(b * hidden + e) * t_head];
    }
    x[hidden] = top1;
    x[hidden + 1] = top1 - top2;
    x[hidden + 2] = ent / k.ln();
    x[hidden + 3] = k / 255.0;
}

/// C++ `forward_laya_act_head_layer`.
#[allow(clippy::too_many_arguments)]
pub fn act_head_forward(
    head_out: &[f32],
    scorer_logits: &[f32],
    marker: &[f32],
    weights: &LayaActHeadWeights,
    batch: usize,
    hidden: usize,
    max_options: usize,
    marker_channels: usize,
    t_head: usize,
    t_marker: usize,
    fc_dim: usize,
    n_act: usize,
) -> Result<Vec<f32>, String> {
    validate_shapes(
        head_out,
        scorer_logits,
        marker,
        weights,
        batch,
        hidden,
        max_options,
        marker_channels,
        t_head,
        t_marker,
        fc_dim,
        n_act,
    )?;
    let input = hidden + LAYA_ACT_HEAD_N_FEATS;
    let mut output = vec![0.0f32; batch * n_act];
    let mut x = vec![0.0f32; input];
    let mut hid = vec![0.0f32; fc_dim];

    for b in 0..batch {
        features_for_sample(
            head_out,
            scorer_logits,
            marker,
            b,
            hidden,
            max_options,
            marker_channels,
            t_head,
            t_marker,
            &mut x,
        );
        for f in 0..fc_dim {
            let w = &weights.fc1_w[f * input..(f + 1) * input];
            let mut acc = weights.fc1_b[f];
            for i in 0..input {
                acc += w[i] * x[i];
            }
            hid[f] = gelu_erf(acc);
        }
        for a in 0..n_act {
            let w = &weights.fc2_w[a * fc_dim..(a + 1) * fc_dim];
            let mut acc = weights.fc2_b[a];
            for f in 0..fc_dim {
                acc += w[f] * hid[f];
            }
            output[b * n_act + a] = acc;
        }
    }
    Ok(output)
}

/// The gradient of one backward pass: `-dL/dparam` in C++ is accumulated as
/// `dL/dparam` here (the Rust convention is a positive gradient), plus the
/// gradient that flows to the head's CLS column.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaActHeadGrad {
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    /// `[batch, hidden]`: add to `head.delta[b, e, 0]`.
    pub dcls: Vec<f32>,
}

/// C++ `ah_backward_sample`, for the whole batch at once.
///
/// `output_gradient` is `[batch, n_act]`.  The softmax features are constants,
/// so the scorer and marker inputs get no gradient.
#[allow(clippy::too_many_arguments)]
pub fn act_head_backward(
    head_out: &[f32],
    scorer_logits: &[f32],
    marker: &[f32],
    weights: &LayaActHeadWeights,
    output_gradient: &[f32],
    batch: usize,
    hidden: usize,
    max_options: usize,
    marker_channels: usize,
    t_head: usize,
    t_marker: usize,
    fc_dim: usize,
    n_act: usize,
) -> Result<LayaActHeadGrad, String> {
    validate_shapes(
        head_out,
        scorer_logits,
        marker,
        weights,
        batch,
        hidden,
        max_options,
        marker_channels,
        t_head,
        t_marker,
        fc_dim,
        n_act,
    )?;
    if output_gradient.len() != batch * n_act {
        return Err(format!(
            "laya_act_head output gradient has {} values, expected batch*n_act={}",
            output_gradient.len(),
            batch * n_act
        ));
    }
    let input = hidden + LAYA_ACT_HEAD_N_FEATS;
    let mut grad = LayaActHeadGrad {
        fc1_w: vec![0.0f32; fc_dim * input],
        fc1_b: vec![0.0f32; fc_dim],
        fc2_w: vec![0.0f32; n_act * fc_dim],
        fc2_b: vec![0.0f32; n_act],
        dcls: vec![0.0f32; batch * hidden],
    };
    let mut x = vec![0.0f32; input];
    let mut u = vec![0.0f32; fc_dim];
    let mut gu = vec![0.0f32; fc_dim];

    for b in 0..batch {
        features_for_sample(
            head_out,
            scorer_logits,
            marker,
            b,
            hidden,
            max_options,
            marker_channels,
            t_head,
            t_marker,
            &mut x,
        );
        let d = &output_gradient[b * n_act..(b + 1) * n_act];
        for value in gu.iter_mut() {
            *value = 0.0;
        }
        // Pre-GELU activations.
        for f in 0..fc_dim {
            let w = &weights.fc1_w[f * input..(f + 1) * input];
            let mut acc = weights.fc1_b[f];
            for i in 0..input {
                acc += w[i] * x[i];
            }
            u[f] = acc;
        }
        for a in 0..n_act {
            grad.fc2_b[a] += d[a];
            for f in 0..fc_dim {
                grad.fc2_w[a * fc_dim + f] += d[a] * gelu_erf(u[f]);
                gu[f] += d[a] * weights.fc2_w[a * fc_dim + f];
            }
        }
        let dcls = &mut grad.dcls[b * hidden..(b + 1) * hidden];
        for f in 0..fc_dim {
            let g = gu[f] * gelu_erf_grad(u[f]);
            grad.fc1_b[f] += g;
            let w = &weights.fc1_w[f * input..(f + 1) * input];
            let uw = &mut grad.fc1_w[f * input..(f + 1) * input];
            for i in 0..input {
                uw[i] += g * x[i];
            }
            for e in 0..hidden {
                dcls[e] += g * w[e];
            }
        }
    }
    Ok(grad)
}

#[allow(clippy::too_many_arguments)]
fn validate_shapes(
    head_out: &[f32],
    scorer_logits: &[f32],
    marker: &[f32],
    weights: &LayaActHeadWeights,
    batch: usize,
    hidden: usize,
    max_options: usize,
    marker_channels: usize,
    t_head: usize,
    t_marker: usize,
    fc_dim: usize,
    n_act: usize,
) -> Result<(), String> {
    if hidden == 0 || max_options < 2 || fc_dim == 0 || n_act == 0 || batch == 0 {
        return Err(format!(
            "laya_act_head: hidden={hidden} max_options={max_options} (>=2) fc_dim={fc_dim} n_act={n_act} batch={batch} invalid"
        ));
    }
    if marker_channels != 2 * max_options {
        return Err(format!(
            "laya_act_head: marker input has {marker_channels} channels, expected 2*max_options={}",
            2 * max_options
        ));
    }
    if t_head == 0 || t_marker == 0 {
        return Err("laya_act_head: sequence lengths must be > 0".to_owned());
    }
    if head_out.len() != batch * hidden * t_head {
        return Err(format!(
            "laya_act_head: head input has {} values, expected batch*hidden*t_head={}",
            head_out.len(),
            batch * hidden * t_head
        ));
    }
    if scorer_logits.len() != batch * max_options {
        return Err(format!(
            "laya_act_head: scorer input has {} values, expected batch*max_options={}",
            scorer_logits.len(),
            batch * max_options
        ));
    }
    if marker.len() != batch * marker_channels * t_marker {
        return Err(format!(
            "laya_act_head: marker input has {} values, expected batch*marker_channels*t_marker={}",
            marker.len(),
            batch * marker_channels * t_marker
        ));
    }
    if !weights.matches(hidden, fc_dim, n_act) {
        return Err("laya_act_head: weight tensors do not match the layer geometry".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights(hidden: usize, fc_dim: usize, n_act: usize) -> LayaActHeadWeights {
        let input = hidden + LAYA_ACT_HEAD_N_FEATS;
        let mut next = 0.0f32;
        let mut fill = |len: usize, scale: f32| {
            (0..len)
                .map(|_| {
                    next += 0.017;
                    (next * 0.31).sin() * scale
                })
                .collect::<Vec<f32>>()
        };
        LayaActHeadWeights {
            fc1_w: fill(fc_dim * input, 0.5),
            fc1_b: fill(fc_dim, 0.1),
            fc2_w: fill(n_act * fc_dim, 0.2),
            fc2_b: fill(n_act, 0.1),
        }
    }

    /// `[batch, hidden, 1, t_head]` channel-major.
    fn head_data(batch: usize, hidden: usize, t: usize) -> Vec<f32> {
        (0..batch * hidden * t)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect()
    }

    /// `[batch, marker_channels, 1, t]` channel-major, masks in odd channels.
    fn marker_data(batch: usize, k: usize, t: usize, valid: &[bool], pos: &[f32]) -> Vec<f32> {
        let cb = 2 * k;
        let mut data = vec![0.0f32; batch * cb * t];
        for b in 0..batch {
            for j in 0..k {
                for tt in 0..t {
                    data[(b * cb + 2 * j) * t + tt] = pos[j];
                    data[(b * cb + 2 * j + 1) * t + tt] = if valid[j] { 1.0 } else { 0.0 };
                }
            }
        }
        data
    }

    #[test]
    fn features_follow_the_masked_softmax_definition() {
        let hidden = 3usize;
        let k = 3usize;
        let t = 2usize;
        let head = head_data(1, hidden, t);
        // Valid options 0 and 2, invalid 1.  Logits [0, 5, 0] -> the invalid
        // 5 is masked to -1e4, so p = softmax([0, 5, 0]) over the two valid.
        let logits = vec![0.0f32, 5.0, 0.0];
        let marker = marker_data(1, k, t, &[true, false, true], &[0.0, 0.0, 0.0]);
        let mut x = vec![0.0f32; hidden + 4];
        features_for_sample(&head, &logits, &marker, 0, hidden, k, 2 * k, t, t, &mut x);

        // CLS column 0 of every head channel.
        for e in 0..hidden {
            assert_eq!(x[e], head[e * t]);
        }
        let e0 = 0.0f32.exp();
        let p0 = e0 / (e0 + e0);
        let p2 = p0;
        assert!((x[hidden] - p0).abs() < 1e-6, "{x:?}");
        // top2 is p[2] (also 0.5) since p[0] and p[2] tie; the strict scan gives
        // top1 = p[0], top2 = p[2].
        assert!((x[hidden + 1] - (p0 - p2)).abs() < 1e-6);
        // entropy of a 2-way 0.5/0.5 distribution is ln2; k = 2 (not 3, the
        // invalid option is not counted, and the floor is 2).
        assert!((x[hidden + 2] - 1.0).abs() < 1e-5, "{x:?}");
        assert!((x[hidden + 3] - 2.0 / 255.0).abs() < 1e-9);
    }

    #[test]
    fn all_invalid_options_give_a_uniform_distribution_and_k_floor() {
        let hidden = 2usize;
        let k = 4usize;
        let t = 1usize;
        let head = head_data(1, hidden, t);
        let logits = vec![3.0f32, 1.0, 0.0, -2.0];
        let marker = marker_data(1, k, t, &[false; 4], &[0.0; 4]);
        let mut x = vec![0.0f32; hidden + 4];
        features_for_sample(&head, &logits, &marker, 0, hidden, k, 2 * k, t, t, &mut x);
        assert!((x[hidden] - 0.25).abs() < 1e-6, "{x:?}");
        // Uniform over K=4: entropy = ln(4); k floors to 2, so feats = ln4/ln2 = 2.
        assert!((x[hidden + 2] - 2.0).abs() < 1e-5, "{x:?}");
        assert!((x[hidden + 3] - 2.0 / 255.0).abs() < 1e-9, "k floor is 2");
    }

    #[test]
    fn act_head_axis_layout_is_batch_major_over_the_two_layer_mlp() {
        let (batch, hidden, k, t, fc_dim, n_act) = (2usize, 4usize, 3usize, 2usize, 5usize, 2usize);
        let head = head_data(batch, hidden, t);
        let logits: Vec<f32> = (0..batch * k).map(|i| (i as f32) * 0.3 - 0.5).collect();
        let marker = marker_data(batch, k, t, &[true, true, false], &[0.0; 3]);
        let w = weights(hidden, fc_dim, n_act);
        let out = act_head_forward(
            &head,
            &logits,
            &marker,
            &w,
            batch,
            hidden,
            k,
            2 * k,
            t,
            t,
            fc_dim,
            n_act,
        )
        .unwrap();
        // Hand-compute both batch rows to pin the batch-major [B,n_act] output
        // and the channel-major feature gathering for each sample.
        let input = hidden + LAYA_ACT_HEAD_N_FEATS;
        for b in 0..batch {
            let mut x = vec![0.0f32; input];
            features_for_sample(&head, &logits, &marker, b, hidden, k, 2 * k, t, t, &mut x);
            let mut hid = vec![0.0f32; fc_dim];
            for f in 0..fc_dim {
                let mut acc = w.fc1_b[f];
                for i in 0..input {
                    acc += w.fc1_w[f * input + i] * x[i];
                }
                hid[f] = gelu_erf(acc);
            }
            for a in 0..n_act {
                let mut acc = w.fc2_b[a];
                for f in 0..fc_dim {
                    acc += w.fc2_w[a * fc_dim + f] * hid[f];
                }
                let index = b * n_act + a;
                assert!(
                    (out[index] - acc).abs() < 1e-5,
                    "out[{index}]={} vs {acc}",
                    out[index]
                );
            }
        }
    }

    #[test]
    fn backward_matches_central_differences_for_every_tensor() {
        let (batch, hidden, k, t, fc_dim, n_act) = (2usize, 3usize, 3usize, 2usize, 4usize, 2usize);
        let head = head_data(batch, hidden, t);
        let logits: Vec<f32> = (0..batch * k).map(|i| (i as f32) * 0.21 - 0.3).collect();
        let marker = marker_data(batch, k, t, &[true, false, true], &[0.0; 3]);
        let seed: Vec<f32> = (0..batch * n_act).map(|i| 0.4 + 0.2 * i as f32).collect();
        let w = weights(hidden, fc_dim, n_act);

        let grad = act_head_backward(
            &head,
            &logits,
            &marker,
            &w,
            &seed,
            batch,
            hidden,
            k,
            2 * k,
            t,
            t,
            fc_dim,
            n_act,
        )
        .unwrap();

        let loss_of = |weights: &LayaActHeadWeights| -> f32 {
            let out = act_head_forward(
                &head,
                &logits,
                &marker,
                weights,
                batch,
                hidden,
                k,
                2 * k,
                t,
                t,
                fc_dim,
                n_act,
            )
            .unwrap();
            out.iter().zip(&seed).map(|(o, s)| o * s).sum()
        };

        let eps = 1e-2f32;
        let check = |name: &str,
                     analytic: &[f32],
                     base: &[f32],
                     make: &dyn Fn(&mut LayaActHeadWeights, usize, f32)| {
            for i in 0..analytic.len() {
                let mut hi = w.clone();
                make(&mut hi, i, eps);
                let mut lo = w.clone();
                make(&mut lo, i, -eps);
                let numeric = (loss_of(&hi) - loss_of(&lo)) / (2.0 * eps);
                let _ = base;
                assert!(
                    (analytic[i] - numeric).abs() <= 5e-2 * numeric.abs().max(2e-4),
                    "{name}[{i}]: analytic {} numeric {numeric}",
                    analytic[i]
                );
            }
        };
        check("fc1_w", &grad.fc1_w, &w.fc1_w, &|weights, i, d| {
            weights.fc1_w[i] += d
        });
        check("fc1_b", &grad.fc1_b, &w.fc1_b, &|weights, i, d| {
            weights.fc1_b[i] += d
        });
        check("fc2_w", &grad.fc2_w, &w.fc2_w, &|weights, i, d| {
            weights.fc2_w[i] += d
        });
        check("fc2_b", &grad.fc2_b, &w.fc2_b, &|weights, i, d| {
            weights.fc2_b[i] += d
        });
    }

    #[test]
    fn blob_round_trip_and_offset() {
        let (hidden, fc_dim, n_act) = (8usize, 4usize, 3usize);
        let w = weights(hidden, fc_dim, n_act);
        assert_eq!(w.to_blob().len(), blob_float_count(hidden, fc_dim, n_act));
        let back = LayaActHeadWeights::from_blob(&w.to_blob(), hidden, fc_dim, n_act).unwrap();
        assert_eq!(w, back);
        assert!(LayaActHeadWeights::from_blob(&w.to_blob()[..10], hidden, fc_dim, n_act).is_err());
        // Offset is 20 + (encoder + head + scorer + 3) * 4 for f32.
        let offset = blob_offset_bytes(1024, 28, 2624, 50368, 1024, 2, 4096, 1024, false);
        let expected = 20
            + (crate::laya_scorer::encoder_blob_float_count(1024, 28, 2624, 50368)
                + crate::laya_scorer::head_blob_float_count(1024, 2, 4096)
                + scorer_blob_float_count(1024)
                + 3)
                * 4;
        assert_eq!(offset, expected);
    }

    #[test]
    fn shape_errors_are_reported() {
        let (batch, hidden, k, t, fc_dim, n_act) = (1usize, 2usize, 3usize, 2usize, 2usize, 2usize);
        let head = head_data(batch, hidden, t);
        let logits = vec![0.0f32; batch * k];
        let marker = marker_data(batch, k, t, &[true; 3], &[0.0; 3]);
        let w = weights(hidden, fc_dim, n_act);
        // Wrong marker channel count.
        assert!(
            act_head_forward(
                &head, &logits, &marker, &w, batch, hidden, k, k, t, t, fc_dim, n_act
            )
            .is_err()
        );
        // Wrong scorer length.
        assert!(
            act_head_forward(
                &head,
                &logits[..1],
                &marker,
                &w,
                batch,
                hidden,
                k,
                2 * k,
                t,
                t,
                fc_dim,
                n_act
            )
            .is_err()
        );
        // max_options must be >= 2.
        assert!(
            act_head_forward(
                &head, &logits, &marker, &w, batch, hidden, 1, 2, t, t, fc_dim, n_act
            )
            .is_err()
        );
        // Backward needs the output gradient.
        assert!(
            act_head_backward(
                &head,
                &logits,
                &marker,
                &w,
                &[],
                batch,
                hidden,
                k,
                2 * k,
                t,
                t,
                fc_dim,
                n_act
            )
            .is_err()
        );
    }
}
