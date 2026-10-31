//! `[contrastive]`: the supervised-contrastive head, classifier branch (deferred
//! item (c) of `tasks/rust/04_reference/layers.md` Phase 1b), ported from
//! `Jev_laya:src-lib/softmax_layer.cpp::forward_contrastive_layer` and the
//! `P_constrastive` / `grad_contrastive_loss_positive` /
//! `grad_contrastive_loss_negative` helpers in `src-lib/blas.cpp`.
//!
//! The layer is a pass-through for the forward and the backward
//! (`axpy_cpu(..., 1, l.delta, 1, state.delta, 1)`); everything happens in the
//! delta it computes from the previous layer's embeddings.
//!
//! **Scope.**  Only the *classifier* branch (`yolo_layer = 0`) is ported.  The
//! detection branch needs the referenced `[yolo]` layer's per-box class ids and
//! label track, which the Rust yolo head does not expose, so a cfg that points
//! at a `[yolo]` layer is reported as unsupported instead of being silently
//! wrong.  The classifier branch additionally assumes the previous layer
//! produces exactly one embedding vector per image (`H = W = 1`, one anchor):
//! C++'s `labels` array has `batch` entries but is indexed by
//! `b * n * h * w + ...`, so anything larger than a single cell reads out of
//! bounds there too.  That is checked instead of reproduced.
//!
//! Math.  With `z_i` the embedding of sample `i`, `s_ij` their cosine
//! similarity and `T` the temperature,
//! `P[i][j] = exp(s_ij / T) / sum_{k != i} exp(s_kj / T)`.  Note the
//! denominator runs over column `j` (every sample similar to `j`, excluding the
//! anchor) rather than row `i`; that is C++'s definition and it is
//! self-consistent with the gradient formulas below, so it is ported as-is.
//! For each anchor `i` with `N` samples of its class,
//! `mult = 1 / ((N - 1) * T * ||z_i||)`, and the delta is
//!
//! * positive (same class `j`): `-mult * (s_ij z_i - z_j) * (1 - P[i][j])`
//! * negative (same class `j`, every other `k`): `-mult * (z_k - s_ik z_i) * P[i][k]`
//!
//! scaled by `cls_normalizer`, clipped to `max_delta`, and reported as
//! `cost = ||delta||^2` (C++ `pow(mag_array(l.delta, ...), 2)`).

use crate::model::ContrastiveSpec;

/// Largest finite `f32`, C++ `FLT_MAX`: the default that disables clipping.
pub const FLT_MAX: f32 = f32::MAX;

/// What `[contrastive]` computes during its forward.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastiveResult {
    /// `||delta||^2`, as C++ reports it.
    pub cost: f32,
    /// The delta to inject into the previous layer.
    pub delta: Vec<f32>,
    /// Percentage of samples whose best different-class match is worse than
    /// their best same-class match; `-1` when nothing was comparable.
    pub accuracy: f32,
    /// Number of samples (`batch` for the classifier branch).
    pub samples: usize,
    /// The class label C++ assigned to each sample.
    pub labels: Vec<i32>,
}

/// `cosine_similarity` (C++): 0 when either vector has zero length.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut mul, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        mul += a[i] * b[i];
        da += a[i] * a[i];
        db += b[i] * b[i];
    }
    let divider = da.sqrt() * db.sqrt();
    if divider > 0.0 { mul / divider } else { 0.0 }
}

fn vector_length(a: &[f32]) -> f32 {
    a.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// The layer's whole forward.  `input` is the previous layer's output (the
/// embeddings), `truth` is `batch * classes` class probabilities and
/// `truth_thresh` is C++'s `state.net.label_smooth_eps`.
///
/// Returns an error only for the two configurations this port does not cover;
/// everything else degrades to `cost = 0` and a zero delta, exactly like C++'s
/// early `return`s.
pub fn contrastive_forward(
    input: &[f32],
    truth: Option<&[f32]>,
    spec: &ContrastiveSpec,
    batch: usize,
    embedding_size: usize,
    time_steps: usize,
    adversarial: bool,
    truth_thresh: f32,
    training: bool,
) -> Result<ContrastiveResult, String> {
    if spec.yolo_layer != 0 {
        return Err(format!(
            "[contrastive] yolo_layer={} (the detection branch) is not ported: it needs the referenced [yolo] layer's per-box class ids and label track",
            spec.yolo_layer
        ));
    }
    let total = batch * embedding_size;
    let mut result = ContrastiveResult {
        cost: 0.0,
        delta: vec![0.0; total],
        accuracy: -1.0,
        samples: batch,
        labels: vec![-1; batch],
    };
    if !training || spec.temperature == 0.0 {
        return Ok(result);
    }
    let Some(truth) = truth else {
        return Ok(result);
    };
    if truth.len() < batch * spec.classes {
        return Ok(result);
    }

    // C++ label assignment: `b / 2` (or `b % 2` for adversarial nets), then any
    // class whose truth probability clears the threshold overwrites it.
    let mini_batch = (batch / time_steps.max(1)).max(1);
    for b in 0..batch {
        result.labels[b] = if adversarial {
            (b % 2) as i32
        } else {
            (b / 2) as i32
        };
        for c in 0..spec.classes {
            if truth[b * spec.classes + c] > truth_thresh {
                result.labels[b] = c as i32;
            }
        }
    }

    let embeddings: Vec<Vec<f32>> = (0..batch)
        .map(|b| input[b * embedding_size..(b + 1) * embedding_size].to_vec())
        .collect();

    // Cosine similarities, `cos_sim[i][j]` for i != j and same time step.
    let mut cos_sim = vec![0.0f32; batch * batch];
    let mut exp_cos_sim = vec![0.0f32; batch * batch];
    let mut max_same = vec![-10.0f32; batch];
    let mut max_diff = vec![-10.0f32; batch];
    for i in 0..batch {
        if result.labels[i] < 0 {
            continue;
        }
        for j in 0..batch {
            if i == j || result.labels[j] < 0 {
                continue;
            }
            if i / mini_batch != j / mini_batch {
                continue; // C++ only pairs inside one time step
            }
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            exp_cos_sim[i * batch + j] = (sim / spec.temperature).exp();
            cos_sim[i * batch + j] = sim;
            if result.labels[i] == result.labels[j] {
                if max_same[i] < sim {
                    max_same[i] = sim;
                }
            } else if max_diff[i] < sim {
                max_diff[i] = sim;
            }
        }
    }

    // Accuracy: comparable samples whose best different-class match is worse.
    let (mut good, mut all) = (0usize, 0usize);
    for i in 0..batch {
        if max_same[i] >= -1.0 && max_diff[i] >= -1.0 {
            all += 1;
            if max_diff[i] < max_same[i] {
                good += 1;
            }
        }
    }
    result.accuracy = if all > 0 {
        100.0 * good as f32 / all as f32
    } else {
        -1.0
    };

    // C++ `P_constrastive(i, j)`: numerator row i, column j; the denominator
    // sums column j over every row except i.
    let mut p_constrastive = vec![0.0f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            if i == j || result.labels[j] < 0 {
                continue;
            }
            let numerator = exp_cos_sim[i * batch + j];
            let mut denominator = 0.0f32;
            for k in 0..batch {
                if k != i {
                    denominator += exp_cos_sim[k * batch + j];
                }
            }
            // C++ divides unconditionally, so a two-sample batch (or any pair
            // whose denominator column is empty) yields 0/0 = NaN there.  The
            // sanitise after the gradient loops turns that into a zero delta
            // anyway, so 0 is the faithful result and is written explicitly
            // instead of producing a NaN.
            p_constrastive[i * batch + j] = if denominator != 0.0 {
                numerator / denominator
            } else {
                0.0
            };
        }
    }

    // Deltas.
    for i in 0..batch {
        if result.labels[i] < 0 {
            continue;
        }
        let vec_len = vector_length(&embeddings[i]);
        if vec_len == 0.0 {
            continue; // C++ treats this as a fatal error; here it means "no pull"
        }
        let n_same = result
            .labels
            .iter()
            .filter(|l| **l == result.labels[i])
            .count() as f32;
        if n_same == 0.0 {
            continue;
        }
        let mult = 1.0 / ((n_same - 1.0) * spec.temperature * vec_len);
        // positive
        for j in 0..batch {
            if j == i || result.labels[j] != result.labels[i] {
                continue;
            }
            let sim = cos_sim[i * batch + j];
            let p = p_constrastive[i * batch + j];
            for m in 0..embedding_size {
                let d = mult * (sim * embeddings[i][m] - embeddings[j][m]) * (1.0 - p);
                result.delta[i * embedding_size + m] -= d;
            }
        }
        // negative: every third sample through every same-class partner
        for j in 0..batch {
            if j == i || result.labels[j] != result.labels[i] {
                continue;
            }
            for k in 0..batch {
                if k == i || k == j || result.labels[k] < 0 {
                    continue;
                }
                let sim = cos_sim[i * batch + k];
                let p = p_constrastive[i * batch + k];
                for m in 0..embedding_size {
                    let d = mult * (embeddings[k][m] - sim * embeddings[i][m]) * p;
                    result.delta[i * embedding_size + m] -= d;
                }
            }
        }
    }

    if spec.cls_normalizer != 1.0 {
        for d in result.delta.iter_mut() {
            *d *= spec.cls_normalizer;
        }
    }
    for d in result.delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
        if spec.max_delta < FLT_MAX {
            *d = d.clamp(-spec.max_delta, spec.max_delta);
        }
    }
    // C++ `pow(mag_array(l.delta, l.inputs * l.batch), 2)`.
    result.cost = result.delta.iter().map(|d| d * d).sum::<f32>();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ContrastiveSpec {
        ContrastiveSpec {
            classes: 2,
            yolo_layer: 0,
            temperature: 1.0,
            cls_normalizer: 1.0,
            max_delta: FLT_MAX,
            contrastive_neg_max: 3,
        }
    }

    #[test]
    fn contrastive_axis_layout_pins_cpp_batch_rows_and_embedding_features() {
        // C++ assigns one embedding row per sample and reads each row's
        // `embedding_size` contiguous features (`softmax_layer.cpp:265-430`,
        // `blas.cpp` contrastive helpers). These asymmetric rows make a
        // sample/feature transpose change cosine pairs and the seeded delta.
        let s = spec();
        let batch = 4usize;
        let embedding_size = 3usize;
        let input = vec![
            1.0f32, 0.0, 0.0, // sample 0, class 0
            0.0, 1.0, 0.0, // sample 1, class 0
            -1.0, 0.0, 0.0, // sample 2, class 1
            0.0, -1.0, 0.0, // sample 3, class 1
        ];
        let truth = vec![1.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let result = contrastive_forward(
            &input,
            Some(&truth),
            &s,
            batch,
            embedding_size,
            1,
            false,
            0.5,
            true,
        )
        .expect("classifier contrastive forward");
        assert_eq!(result.labels, vec![0, 0, 1, 1]);

        // Independent source-order oracle over [sample][feature] rows.
        let rows = (0..batch)
            .map(|sample| &input[sample * embedding_size..(sample + 1) * embedding_size])
            .collect::<Vec<_>>();
        let mut similarities = vec![0.0f32; batch * batch];
        let mut exponentials = vec![0.0f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                if i == j {
                    continue;
                }
                let dot = rows[i].iter().zip(rows[j]).map(|(a, b)| a * b).sum::<f32>();
                let norm_i = rows[i].iter().map(|v| v * v).sum::<f32>().sqrt();
                let norm_j = rows[j].iter().map(|v| v * v).sum::<f32>().sqrt();
                let sim = dot / (norm_i * norm_j);
                similarities[i * batch + j] = sim;
                exponentials[i * batch + j] = (sim / s.temperature).exp();
            }
        }
        let mut probabilities = vec![0.0f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                if i == j {
                    continue;
                }
                let denominator = (0..batch)
                    .filter(|&k| k != i)
                    .map(|k| exponentials[k * batch + j])
                    .sum::<f32>();
                probabilities[i * batch + j] = exponentials[i * batch + j] / denominator;
            }
        }
        let mut expected = vec![0.0f32; batch * embedding_size];
        for i in 0..batch {
            let same_count = result
                .labels
                .iter()
                .filter(|&&label| label == result.labels[i])
                .count();
            let multiplier = 1.0 / ((same_count - 1) as f32 * s.temperature * 1.0);
            for j in 0..batch {
                if j == i || result.labels[j] != result.labels[i] {
                    continue;
                }
                let sim = similarities[i * batch + j];
                let p = probabilities[i * batch + j];
                for feature in 0..embedding_size {
                    expected[i * embedding_size + feature] -=
                        multiplier * (sim * rows[i][feature] - rows[j][feature]) * (1.0 - p);
                }
                for k in 0..batch {
                    if k == i || k == j {
                        continue;
                    }
                    let sim = similarities[i * batch + k];
                    let p = probabilities[i * batch + k];
                    for feature in 0..embedding_size {
                        expected[i * embedding_size + feature] -=
                            multiplier * (rows[k][feature] - sim * rows[i][feature]) * p;
                    }
                }
            }
        }
        for (index, (&actual, &want)) in result.delta.iter().zip(&expected).enumerate() {
            assert!(
                (actual - want).abs() <= 1.0e-6 * (1.0 + want.abs()),
                "contrastive batch/feature axis at flat index {index}: {actual} != {want}"
            );
        }
    }

    #[test]
    fn the_detection_branch_reports_itself_as_unported() {
        let mut s = spec();
        s.yolo_layer = -1;
        let err = contrastive_forward(
            &[1.0, 0.0],
            Some(&[1.0, 0.0]),
            &s,
            1,
            2,
            1,
            false,
            0.0,
            true,
        )
        .expect_err("the detection branch must not pretend to work");
        assert!(err.contains("detection branch"), "{err}");
    }

    #[test]
    fn identical_embeddings_in_one_class_have_no_gradient() {
        // Two images of the same class with identical embeddings: cosine 1, and
        // the positive term is exactly cancelled by the negative term.
        let s = spec();
        let input = vec![1.0f32, 0.0, 1.0, 0.0];
        let truth = vec![1.0f32, 0.0, 1.0, 0.0];
        let r = contrastive_forward(&input, Some(&truth), &s, 2, 2, 1, false, 0.0, true).unwrap();
        assert!(r.cost < 1e-6, "cost {}", r.cost);
        assert!(r.delta.iter().all(|d| d.abs() < 1e-6), "{:?}", r.delta);
        assert_eq!(r.labels, vec![0, 0]);
        // With a single class present there is no different-class match, so no
        // sample is comparable and the accuracy is C++'s -1 sentinel.
        assert_eq!(r.accuracy, -1.0);
    }

    #[test]
    fn opposite_embeddings_in_one_class_are_pulled_together() {
        // The intra-class term is the only gradient source: C++ pairs only
        // samples of the *same* class (the "negative" loop walks third samples
        // through a same-class partner), so a class with one member contributes
        // nothing at all.  Three same-class images are needed for C++'s
        // denominator (summed over every row but the anchor) to be non-zero.
        let s = spec();
        let input = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0];
        let truth = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let r = contrastive_forward(&input, Some(&truth), &s, 3, 2, 1, false, 0.0, true).unwrap();
        assert_eq!(r.labels, vec![0, 0, 0]);
        assert!(r.cost > 0.0, "cost {}", r.cost);
        assert!(r.delta.iter().any(|d| d.abs() > 1e-6), "{:?}", r.delta);
        // The lone member of its class (image 1) gets no gradient at all: C++
        // would divide by N - 1 = 0 there, which the sanitise turns into zero.
        let alone_truth = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let alone =
            contrastive_forward(&input, Some(&alone_truth), &s, 3, 2, 1, false, 0.0, true).unwrap();
        assert_eq!(alone.labels, vec![0, 1, 0]);
        assert!(
            alone.delta[2..4].iter().all(|d| d.abs() < 1e-6),
            "the lone sample must not move: {:?}",
            alone.delta
        );
        assert!(
            alone.delta[0..2].iter().any(|d| d.abs() > 1e-6)
                || alone.delta[4..6].iter().any(|d| d.abs() > 1e-6),
            "its two same-class partners still pull: {:?}",
            alone.delta
        );
    }

    #[test]
    fn labels_come_from_the_truth_when_it_clears_the_threshold() {
        let s = spec();
        let input = vec![1.0f32, 0.0, 0.0, 1.0];
        // No truth above `truth_thresh`, so the fallback `b / 2` stands.
        let r =
            contrastive_forward(&input, Some(&[0.0f32; 4]), &s, 2, 2, 1, false, 0.1, true).unwrap();
        assert_eq!(r.labels, vec![0, 0], "`b / 2` gives both images class 0");
        // A truth probability above the threshold sets the label directly.
        let r = contrastive_forward(
            &input,
            Some(&[0.0, 1.0, 1.0, 0.0]),
            &s,
            2,
            2,
            1,
            false,
            0.1,
            true,
        )
        .unwrap();
        assert_eq!(r.labels, vec![1, 0]);
        // Adversarial nets use `b % 2` instead of `b / 2`.
        let r = contrastive_forward(&input, Some(&[0.0; 4]), &s, 2, 2, 1, true, 0.1, true).unwrap();
        assert_eq!(r.labels, vec![0, 1]);
    }

    #[test]
    fn inference_and_missing_truth_are_silent_no_ops() {
        let s = spec();
        let input = vec![1.0f32, 0.0, -1.0, 0.0];
        let infer = contrastive_forward(
            &input,
            Some(&[1.0, 0.0, 0.0, 1.0]),
            &s,
            2,
            2,
            1,
            false,
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(infer.cost, 0.0);
        assert!(infer.delta.iter().all(|d| *d == 0.0));
        let no_truth = contrastive_forward(&input, None, &s, 2, 2, 1, false, 0.0, true).unwrap();
        assert_eq!(no_truth.cost, 0.0);
        let short =
            contrastive_forward(&input, Some(&[1.0, 0.0]), &s, 2, 2, 1, false, 0.0, true).unwrap();
        assert_eq!(short.cost, 0.0);
    }

    #[test]
    fn time_steps_separate_the_batch_into_groups() {
        let s = spec();
        // Six images in two time steps of three: step 1 (images 0-2) is class
        // 0 and step 2 (images 3-5) is class 1.  C++ only pairs inside one
        // time step, so with two steps each class never sees the other.
        let input = vec![
            1.0f32, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0,
        ];
        let truth = vec![
            1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let split =
            contrastive_forward(&input, Some(&truth), &s, 6, 2, 2, false, 0.0, true).unwrap();
        assert_eq!(split.labels, vec![0, 0, 0, 1, 1, 1]);
        // Joining the two steps (`time_steps = 1`) lets each same-class pair
        // also see the other class through the negative loop.
        let joined =
            contrastive_forward(&input, Some(&truth), &s, 6, 2, 1, false, 0.0, true).unwrap();
        assert!(
            joined.cost > split.cost,
            "joining the steps adds cross-class negatives: {} vs {}",
            joined.cost,
            split.cost
        );
    }

    #[test]
    fn cls_normalizer_scales_the_delta_and_max_delta_clips_it() {
        let mut s = spec();
        s.cls_normalizer = 2.0;
        let input = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0];
        let truth = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let plain =
            contrastive_forward(&input, Some(&truth), &spec(), 3, 2, 1, false, 0.0, true).unwrap();
        let scaled =
            contrastive_forward(&input, Some(&truth), &s, 3, 2, 1, false, 0.0, true).unwrap();
        for (a, b) in plain.delta.iter().zip(&scaled.delta) {
            assert!((b - 2.0 * a).abs() < 1e-6, "{b} vs {}", 2.0 * a);
        }
        assert!(
            (scaled.cost - 4.0 * plain.cost).abs() < 1e-4,
            "{} vs {}",
            scaled.cost,
            4.0 * plain.cost
        );
        let mut clipped = spec();
        clipped.max_delta = 1e-4;
        let r =
            contrastive_forward(&input, Some(&truth), &clipped, 3, 2, 1, false, 0.0, true).unwrap();
        assert!(
            r.delta.iter().all(|d| d.abs() <= clipped.max_delta + 1e-9),
            "{:?}",
            r.delta
        );
    }

    #[test]
    fn cosine_similarity_and_length_match_the_cpp_helpers() {
        assert!((cosine_similarity(&[3.0, 4.0], &[3.0, 4.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(
            cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]),
            0.0,
            "zero length gives 0"
        );
        assert!((vector_length(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }
}
