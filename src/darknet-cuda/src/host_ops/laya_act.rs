//! Checked act-head feature reference and immutable MLP geometry.
//! Gradient composition lives in crate::laya_act; this adds no optimizer updates.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub batch: usize,
    pub hidden: usize,
    pub options: usize,
    pub marker_channels: usize,
    pub head_width: usize,
    pub marker_width: usize,
    pub head_len: usize,
    pub logits_len: usize,
    pub marker_len: usize,
    pub output_len: usize,
}

impl Layout {
    pub fn new(head: &[usize], scorer: &[usize], marker: &[usize]) -> Result<Self, String> {
        if head.len() != 4
            || scorer.len() != 4
            || marker.len() != 4
            || head.contains(&0)
            || scorer.contains(&0)
            || marker.contains(&0)
            || head[2] != 1
            || scorer[1] != 1
            || scorer[2] != 1
            || marker[2] != 1
            || head[0] != scorer[0]
            || head[0] != marker[0]
            || scorer[3] < 2
            || scorer[3].checked_mul(2).is_none_or(|n| marker[1] < n)
        {
            return Err(format!(
                "laya_act_features: incompatible shapes {head:?}, {scorer:?}, {marker:?}"
            ));
        }
        let count = |dims: &[usize]| {
            dims.iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| "laya_act_features: bounded storage overflow".to_owned())
        };
        let width = head[1]
            .checked_add(4)
            .ok_or_else(|| "laya_act_features: hidden+4 overflow".to_owned())?;
        Ok(Self {
            batch: head[0],
            hidden: head[1],
            options: scorer[3],
            marker_channels: marker[1],
            head_width: head[3],
            marker_width: marker[3],
            head_len: count(head)?,
            logits_len: count(scorer)?,
            marker_len: count(marker)?,
            output_len: count(&[head[0], width])?,
        })
    }

    pub fn output_shape(&self) -> [usize; 4] {
        [self.batch, self.hidden + 4, 1, 1]
    }

    pub fn forward(&self, head: &[f32], logits: &[f32], marker: &[f32]) -> Vec<f32> {
        assert_eq!(head.len(), self.head_len);
        assert_eq!(logits.len(), self.logits_len);
        assert_eq!(marker.len(), self.marker_len);
        let mut out = vec![0.0; self.output_len];
        let mut p = vec![0.0; self.options];
        for b in 0..self.batch {
            let mut valid = 0usize;
            for (j, v) in p.iter_mut().enumerate() {
                let active =
                    marker[(b * self.marker_channels + 2 * j + 1) * self.marker_width] > 0.5;
                valid += usize::from(active);
                *v = if active {
                    logits[b * self.options + j]
                } else {
                    -1e4
                };
            }
            let max = p.iter().copied().fold(p[0], f32::max);
            let mut sum = 0.0;
            for v in &mut p {
                *v = (*v - max).exp();
                sum += *v;
            }
            let (mut top1, mut top2, mut entropy) = (-1.0, -1.0, 0.0);
            for v in &mut p {
                *v /= sum;
                entropy -= *v * v.max(1e-9).ln();
                if *v > top1 {
                    top2 = top1;
                    top1 = *v;
                } else if *v > top2 {
                    top2 = *v;
                }
            }
            let count = valid.max(2) as f32;
            let base = b * (self.hidden + 4);
            for c in 0..self.hidden {
                out[base + c] = head[(b * self.hidden + c) * self.head_width];
            }
            out[base + self.hidden..base + self.hidden + 4].copy_from_slice(&[
                top1,
                top1 - top2,
                entropy / count.ln(),
                count / 255.0,
            ]);
        }
        out
    }
}

/// Immutable, checked geometry for standalone act-head gradient math.
/// Construct afresh from the declared source views; callers cannot mutate it
/// into an unchecked layout after construction.
#[derive(Clone, Copy, Debug)]
pub struct MlpLayout {
    features: Layout,
    fc_dim: usize,
    n_act: usize,
}

impl MlpLayout {
    pub fn new(
        head: &[usize],
        scorer: &[usize],
        marker: &[usize],
        fc_dim: usize,
        n_act: usize,
    ) -> Result<Self, String> {
        let features = Layout::new(head, scorer, marker)?;
        let inputs = features.hidden + 4; // feature layout checked this addition
        for (a, b) in [
            (fc_dim, inputs),
            (n_act, fc_dim),
            (features.batch, fc_dim),
            (features.batch, n_act),
            (features.batch, features.hidden),
        ] {
            if a.checked_mul(b)
                .is_none_or(|n| n == 0 || n > i32::MAX as usize)
            {
                return Err("act-head backward: zero/overflow MLP dimensions".into());
            }
        }
        Ok(Self {
            features,
            fc_dim,
            n_act,
        })
    }
    pub fn features(&self) -> Layout {
        self.features
    }
    pub fn fc_dim(&self) -> usize {
        self.fc_dim
    }
    pub fn n_act(&self) -> usize {
        self.n_act
    }
    pub fn inputs(&self) -> usize {
        self.features.hidden + 4
    }
    pub fn fc1_len(&self) -> usize {
        self.fc_dim * self.inputs()
    }
    pub fn fc2_len(&self) -> usize {
        self.n_act * self.fc_dim
    }
    pub fn seed_len(&self) -> usize {
        self.features.batch * self.n_act
    }
    pub fn head_shape(&self) -> [usize; 4] {
        [
            self.features.batch,
            self.features.hidden,
            1,
            self.features.head_width,
        ]
    }
    pub fn scorer_shape(&self) -> [usize; 4] {
        [self.features.batch, 1, 1, self.features.options]
    }
    pub fn marker_shape(&self) -> [usize; 4] {
        [
            self.features.batch,
            self.features.marker_channels,
            1,
            self.features.marker_width,
        ]
    }
}

/// Add the pooled CLS adjoint into token zero of an existing NCHW head
/// gradient. `dcls` is `[batch, hidden]`; `head_grad` is `[batch, hidden, 1,
/// tokens]`. Scorer/marker summary features are detached and receive no gradient.
pub fn scatter_cls_adjoint(
    dcls: &[f32],
    head_grad: &[f32],
    batch: usize,
    hidden: usize,
    tokens: usize,
) -> Result<Vec<f32>, String> {
    let cls_len = batch
        .checked_mul(hidden)
        .filter(|&len| len > 0 && len <= i32::MAX as usize)
        .ok_or_else(|| "laya_act_cls_scatter: invalid batch/hidden dimensions".to_owned())?;
    let full_len = cls_len
        .checked_mul(tokens)
        .filter(|&len| tokens > 0 && len <= i32::MAX as usize)
        .ok_or_else(|| "laya_act_cls_scatter: invalid token count/storage".to_owned())?;
    if dcls.len() != cls_len || head_grad.len() != full_len {
        return Err(format!(
            "laya_act_cls_scatter: expected dcls={cls_len}, head_grad={full_len}, got dcls={}, head_grad={}",
            dcls.len(),
            head_grad.len()
        ));
    }
    let mut out = head_grad.to_vec();
    for b in 0..batch {
        for h in 0..hidden {
            let index = (b * hidden + h) * tokens;
            out[index] += dcls[b * hidden + h];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::scatter_cls_adjoint;

    #[test]
    fn cls_adjoint_scatter_adds_only_token_zero_in_nchw_layout() {
        let got = scatter_cls_adjoint(
            &[1.0, -2.0, 0.5, 3.0],
            &[
                0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            2,
            2,
            5,
        )
        .unwrap();
        assert_eq!(
            got,
            vec![
                1.1, 0.2, 0.3, 0.4, 1.0, 0.0, 3.0, 4.0, -1.0, -2.0, -2.5, -4.0, 5.0, 6.0, 7.0,
                11.0, 9.0, 10.0, 11.0, 12.0,
            ]
        );
    }

    #[test]
    fn cls_adjoint_scatter_rejects_bad_storage_and_overflow() {
        assert!(scatter_cls_adjoint(&[1.0], &[0.0; 2], 1, 2, 1).is_err());
        assert!(scatter_cls_adjoint(&[1.0], &[0.0], usize::MAX, 2, 1).is_err());
        assert!(scatter_cls_adjoint(&[1.0], &[0.0], 1, 1, 0).is_err());
    }
}
