//! Shared training labels, loss results, and dense-head truth preparation.

use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, Tensor, TensorShape};

#[derive(Clone, Debug, PartialEq)]
pub struct YoloLabel {
    pub class_id: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloTrainingLoss {
    pub loss: f32,
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
    pub positive_matches: usize,
    pub ignored_objectness: usize,
    /// Mean IoU of the box-loss assignments (C++ `tot_iou / count`, summed over heads).
    pub avg_iou: f32,
    /// Number of box-loss assignments (C++ `count`, summed over heads).
    pub box_count: usize,
    /// Number of class-delta assignments (C++ `class_count`; incremented together with `count`).
    pub class_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseTrainingLoss {
    /// Mean of the per-layer `cost` over the loss layers that had truth (C++ `get_network_cost`).
    pub loss: f32,
    /// Descent-direction seeds, `Some` for every loss layer that had truth. Feed them to
    /// `backward_accumulate_seeded`; the loss layers' backward arms pass them through.
    pub layer_output_gradient_seeds: Vec<Option<Tensor>>,
}

/// Truth row width of the dense box-table heads (`float_to_box` + class id +
/// padding), as `[yolox]`, `[ppyoloe]`, `[yolonas]` and `[centernet]` read it.
const DENSE_TRUTH_SIZE: usize = 6;

/// Per-layer truth tensors for the dense box-table heads.
///
/// Every `[yolox]`, `[ppyoloe]`, `[yolonas]` and `[centernet]` layer gets
/// `batch` images of `max * truth_size` floats: the labels in image-normalized
/// `x, y, w, h, class` order, then an all-zero row (C++ stops at the first row
/// whose `x` is zero, which is how a short list is terminated). Other layers get
/// `None`, which is what [`crate::dense_loss::compute_dense_training_loss_at`]
/// expects.
///
/// Labels beyond the head's `max` are dropped, matching C++'s truth loop.
pub fn dense_head_truth(
    network: &Network,
    labels: &[Vec<YoloLabel>],
) -> Result<Vec<Option<Tensor>>> {
    let batch = network.net.batch.max(1);
    if labels.len() != batch {
        return Err(Error::InvalidConfig(format!(
            "dense truth batch mismatch: expected {batch}, got {}",
            labels.len()
        )));
    }
    let mut truth = vec![None; network.layers.len()];
    for layer in &network.layers {
        let max_boxes = match &layer.kind {
            LayerKind::Yolox(spec) | LayerKind::Ppyoloe(spec) | LayerKind::Yolonas(spec) => {
                spec.max_boxes
            }
            LayerKind::CenterNet(spec) => spec.max_boxes,
            _ => continue,
        };
        let rows = max_boxes * DENSE_TRUTH_SIZE;
        let mut data = vec![0.0f32; batch * rows];
        for (image, image_labels) in labels.iter().enumerate() {
            for (row, label) in image_labels.iter().take(max_boxes).enumerate() {
                let at = image * rows + row * DENSE_TRUTH_SIZE;
                data[at..at + 5].copy_from_slice(&[
                    label.x,
                    label.y,
                    label.w,
                    label.h,
                    label.class_id as f32,
                ]);
            }
        }
        truth[layer.index] = Some(Tensor::from_data(
            TensorShape::new(rows, 1, 1, batch),
            data,
        )?);
    }
    Ok(truth)
}

/// True for a network whose only detection heads are the dense box-table heads.
pub fn network_uses_dense_heads(network: &Network) -> bool {
    network.layers.iter().any(|layer| {
        matches!(
            layer.kind,
            LayerKind::Yolox(_)
                | LayerKind::Ppyoloe(_)
                | LayerKind::Yolonas(_)
                | LayerKind::CenterNet(_)
        )
    })
}
