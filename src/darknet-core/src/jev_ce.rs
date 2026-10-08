//! jev_ce softmax + soft-target cross-entropy (tasks/rust/layers.md Task 11).

use crate::model::JevCeSpec;

fn at(spec: &JevCeSpec, base: usize, g: usize, i: usize) -> usize {
    if spec.channel_major {
        base + i * spec.groups + g
    } else {
        base + g * spec.candidates + i
    }
}

/// `softmax(logits / T)` per group (layer output; also the inference output).
pub(crate) fn jev_ce_probs(logits: &[f32], spec: &JevCeSpec, batch: usize) -> Vec<f32> {
    let temp = spec.temperature.max(1e-6);
    let mut probs = vec![0.0f32; logits.len()];
    for b in 0..batch {
        let base = b * spec.groups * spec.candidates;
        for g in 0..spec.groups {
            let max = (0..spec.candidates)
                .map(|i| logits[at(spec, base, g, i)])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..spec.candidates {
                let e = ((logits[at(spec, base, g, i)] - max) / temp).exp();
                probs[at(spec, base, g, i)] = e;
                sum += e;
            }
            for i in 0..spec.candidates {
                probs[at(spec, base, g, i)] /= sum;
            }
        }
    }
    probs
}

pub(crate) struct JevCeLoss {
    pub mean_ce: f32,
    pub active_groups: usize,
    pub correct: usize,
    /// Darknet descent direction `(t - p) / T`, normalised per sample when `spec.normalize`.
    pub delta: Vec<f32>,
}

pub(crate) fn jev_ce_loss(
    logits: &[f32],
    truth: &[f32],
    spec: &JevCeSpec,
    batch: usize,
) -> JevCeLoss {
    let temp = spec.temperature.max(1e-6);
    let eps = if spec.label_smoothing > 0.0 {
        spec.label_smoothing.min(0.999)
    } else {
        0.0
    };
    let probs = jev_ce_probs(logits, spec, batch);
    let mut delta = vec![0.0f32; logits.len()];
    let (mut ce_sum, mut active, mut correct) = (0.0f64, 0usize, 0usize);
    for b in 0..batch {
        let base = b * spec.groups * spec.candidates;
        let mut sample_active = 0usize;
        for g in 0..spec.groups {
            let target_sum: f32 = (0..spec.candidates)
                .map(|i| truth[at(spec, base, g, i)])
                .sum();
            let empty = target_sum <= 1e-8;
            if !empty || !spec.ignore_empty {
                sample_active += 1;
            }
            if empty && spec.ignore_empty {
                continue; // masked group: no loss, no gradient
            }
            let (mut arg_p, mut arg_t, mut loss) = (0usize, 0usize, 0.0f32);
            for i in 1..spec.candidates {
                if probs[at(spec, base, g, i)] > probs[at(spec, base, g, arg_p)] {
                    arg_p = i;
                }
                if truth[at(spec, base, g, i)] > truth[at(spec, base, g, arg_t)] {
                    arg_t = i;
                }
            }
            for i in 0..spec.candidates {
                let idx = at(spec, base, g, i);
                let mut t = if empty {
                    1.0 / spec.candidates as f32
                } else {
                    truth[idx] / target_sum
                };
                if eps > 0.0 {
                    t = (1.0 - eps) * t + eps / spec.candidates as f32;
                }
                loss += -t * probs[idx].max(1e-12).ln();
                delta[idx] = (t - probs[idx]) / temp;
            }
            ce_sum += loss as f64;
            active += 1;
            correct += (arg_p == arg_t) as usize;
        }
        if spec.normalize {
            let inv = 1.0 / sample_active.max(1) as f32;
            for v in &mut delta[base..base + spec.groups * spec.candidates] {
                *v *= inv;
            }
        }
    }
    JevCeLoss {
        mean_ce: if active > 0 {
            (ce_sum / active as f64) as f32
        } else {
            0.0
        },
        active_groups: active,
        correct,
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::JevCeSpec;

    fn spec(groups: usize, candidates: usize) -> JevCeSpec {
        JevCeSpec {
            groups,
            candidates,
            temperature: 1.0,
            normalize: true,
            ignore_empty: true,
            label_smoothing: 0.0,
            scale: 1.0,
            channel_major: false,
        }
    }

    #[test]
    fn uniform_logits_give_ln3_loss_and_descent_delta() {
        let s = spec(1, 3);
        let l = jev_ce_loss(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &s, 1);
        assert!((l.mean_ce - 3f32.ln()).abs() < 1e-6);
        let third = 1.0 / 3.0;
        for (got, want) in l.delta.iter().zip([1.0 - third, -third, -third]) {
            assert!((got - want).abs() < 1e-6);
        }
        assert_eq!((l.active_groups, l.correct), (1, 1));
        let p = jev_ce_probs(&[0.0, 0.0, 0.0], &s, 1);
        assert!(p.iter().all(|v| (v - third).abs() < 1e-6));
    }

    #[test]
    fn empty_target_groups_are_masked_and_normalisation_counts_active_groups() {
        let s = spec(2, 3);
        // group 1 has an all-zero target -> ignored: delta 0, one active group
        let l = jev_ce_loss(&[0.0; 6], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &s, 1);
        assert_eq!(l.active_groups, 1);
        assert!(l.delta[3..].iter().all(|v| *v == 0.0));
        // two active groups with normalize: every delta is halved
        let l2 = jev_ce_loss(&[0.0; 6], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &s, 1);
        assert!((l2.delta[0] - (1.0 - 1.0 / 3.0) / 2.0).abs() < 1e-6);
    }

    #[test]
    fn temperature_label_smoothing_and_channel_major_layout() {
        let mut s = spec(1, 3);
        s.label_smoothing = 0.3; // t = 0.7*[1,0,0] + 0.3/3 = [0.8, 0.1, 0.1]
        let l = jev_ce_loss(&[0.0; 3], &[1.0, 0.0, 0.0], &s, 1);
        assert!((l.delta[0] - (0.8 - 1.0 / 3.0)).abs() < 1e-6);
        let mut hot = spec(1, 2);
        hot.temperature = 2.0; // p = softmax([1, 0]); delta = (t - p) / 2
        let p_first = 1.0 / (1.0 + (-1.0f32).exp());
        let l = jev_ce_loss(&[2.0, 0.0], &[1.0, 0.0], &hot, 1);
        assert!((l.delta[0] - (1.0 - p_first) / 2.0).abs() < 1e-6);
        // channel-major: element (g, i) lives at i*groups + g. Group 0 = (idx0, idx2) = (1, 2).
        let mut cm = spec(2, 2);
        cm.channel_major = true;
        let p = jev_ce_probs(&[1.0, 0.0, 2.0, 0.0], &cm, 1);
        let e = 1.0f32.exp();
        assert!((p[0] - 1.0 / (1.0 + e)).abs() < 1e-6 && (p[2] - e / (1.0 + e)).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6 && (p[3] - 0.5).abs() < 1e-6); // group 1 = (0, 0)
    }

    #[test]
    fn delta_is_minus_the_gradient_of_the_summed_cross_entropy() {
        let mut s = spec(2, 3);
        s.normalize = false;
        let logits = [0.3f32, -0.2, 0.9, 1.1, 0.0, -0.7];
        let truth = [0.2f32, 0.5, 0.3, 1.0, 0.0, 0.0];
        let ce = |z: &[f32]| {
            let l = jev_ce_loss(z, &truth, &s, 1);
            l.mean_ce * l.active_groups as f32
        };
        let analytic = jev_ce_loss(&logits, &truth, &s, 1).delta;
        let h = 1e-3f32;
        for k in 0..6 {
            let (mut p, mut m) = (logits, logits);
            p[k] += h;
            m[k] -= h;
            let fd = (ce(&p) - ce(&m)) / (2.0 * h);
            assert!(
                (fd + analytic[k]).abs() < 2e-3,
                "dz[{k}]: -dCE/dz = {} vs delta {}",
                -fd,
                analytic[k]
            );
        }
    }
}
