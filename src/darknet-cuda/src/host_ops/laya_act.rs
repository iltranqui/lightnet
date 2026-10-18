//! Checked layout/reference for the act-head's detached CLS + four features.
//! This is not the MLP or its backward/update contract.
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
