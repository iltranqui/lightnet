//! Model integration tests that require core-owned training or layer code.

use std::fs;
use crate::cfg::ConfigFile;
use lightnet_model::model::{self, LayerKind, Network};

fn build(text: &str) -> crate::error::Result<Network> {
    Network::from_config(ConfigFile::parse_str("t.cfg", text)?)
}

#[test]
fn kimi_kda_aliases_defaults_identity_shape_and_weight_metadata() {
    for alias in ["kimi_kda", "kimi_delta_attention"] {
        let network = build(&format!(
            "[net]\nwidth=3\nheight=2\nchannels=8\n[{alias}]\n"
        ))
        .unwrap();
        let layer = &network.layers[0];
        let LayerKind::KimiKda(spec) = &layer.kind else {
            panic!("expected Kimi KDA for [{alias}]")
        };
        assert_eq!(
            (
                spec.filters,
                spec.rank,
                spec.experts,
                spec.top_k,
                spec.hidden,
                spec.temperature,
                spec.load_balance,
            ),
            (8, 2, 4, 2, 16, 1.0, 0.01)
        );
        assert_eq!(spec.activation, model::Activation::Gelu);
        assert_eq!(layer.kind.name(), "kimi_kda");
        assert_eq!(layer.output_shape, model::TensorShape::new(3, 2, 8, 1));
        let weights = layer.weights().unwrap();
        assert_eq!(weights.entries.len(), 2);
        assert_eq!(weights.entries[0].array.name(), "biases");
        assert_eq!(weights.entries[1].array.name(), "weights");
        let params = crate::KimiKdaParams::zeroed(8, 2, 4, 16).unwrap();
        assert_eq!(weights.parameter_count(), params.parameter_count());
    }
}

#[test]
fn mambavision_aliases_defaults_shape_and_weight_metadata() {
    for alias in ["mambavision", "mambavision_block"] {
        let network = build(&format!(
            "[net]\nwidth=3\nheight=2\nchannels=8\n[{alias}]\n"
        ))
        .unwrap();
        let layer = &network.layers[0];
        let LayerKind::MambaVision(spec) = &layer.kind else {
            panic!("expected MambaVision for [{alias}]")
        };
        assert_eq!(
            (
                spec.filters,
                spec.state,
                spec.conv_size,
                spec.dt_rank,
                spec.ffn_ratio,
                spec.msa_enabled,
                spec.msa_block_size,
                spec.msa_topk,
                spec.msa_heads,
                spec.causal,
            ),
            (8, 8, 3, 1, 4, false, 16, 4, 1, false)
        );
        assert_eq!(spec.activation, model::Activation::Gelu);
        assert_eq!(layer.kind.name(), "mambavision");
        assert_eq!(layer.output_shape, model::TensorShape::new(3, 2, 8, 1));
        let weights = layer.weights().unwrap();
        assert_eq!(
            weights.entries.first().unwrap().array.name(),
            "mambavision_in_proj_weights"
        );
        assert_eq!(
            weights.entries.last().unwrap().array.name(),
            "mambavision_ffn_b2"
        );
        assert_eq!(
            weights.parameter_count(),
            crate::mambavision::MambaVisionParams::zeroed(8, 8, 8, 3, 1, 4)
                .unwrap()
                .parameter_count()
        );
    }

    let network = build(
        "[net]\nwidth=4\nheight=2\nchannels=8\n\
         [mambavision]\nfilters=10\nstate=4\nd_state=5\nconv_size=5\nsize=7\n\
         dt_rank=2\nffn_ratio=2\nactivation=linear\nmsa=0\nminimax_sparse_attention=1\n\
         msa_block=6\nmsa_block_size=3\nmsa_topk=5\nmsa_top_blocks=2\nmsa_heads=3\ncausal=1\n",
    )
    .unwrap();
    let LayerKind::MambaVision(spec) = &network.layers[0].kind else {
        panic!("expected MambaVision")
    };
    assert_eq!(
        (
            spec.filters,
            spec.state,
            spec.conv_size,
            spec.dt_rank,
            spec.ffn_ratio
        ),
        (10, 4, 5, 2, 2)
    );
    assert_eq!(
        (
            spec.msa_enabled,
            spec.msa_block_size,
            spec.msa_topk,
            spec.msa_heads,
            spec.causal
        ),
        (false, 6, 5, 1, true)
    );
    assert_eq!(spec.activation, model::Activation::Linear);
}

#[test]
fn obb_yolo_parses_class_multipliers_from_per_class_counters() {
    let network = build(
        "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\ncounters_per_class=1,4\nmax_delta=1.5\n",
    )
    .expect("OBB class counters parse");
    let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert_eq!(spec.class_multipliers, Some(vec![1.5, 1.0]));
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("parsed class multipliers route into training assignment");
    assert_eq!(
        training.assignment_config.class_multipliers,
        spec.class_multipliers
    );

    let invalid = build(
        "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\ncounters_per_class=3\n",
    );
    assert!(invalid.is_err(), "counter cardinality must match classes");
}

#[test]
fn obb_yolo_parses_and_routes_class_map_files() {
    let dir = std::env::temp_dir().join("darknet-model-obb-class-map");
    let _ = fs::create_dir_all(&dir);
    let map_path = dir.join("labels.map");
    let cfg_path = dir.join("obb.cfg");
    fs::write(&map_path, "1\n0\n").expect("map writes");
    fs::write(
        &cfg_path,
        "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\nmap=labels.map\n",
    )
    .expect("cfg writes");

    let network = Network::from_path(&cfg_path).expect("OBB map path resolves");
    let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert_eq!(spec.class_map, Some(vec![1, 0]));
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("parsed class map routes into training assignment");
    assert_eq!(training.assignment_config.class_map, spec.class_map);
}

#[test]
fn obb_yolo_routes_covariance_cobb_and_viou_options() {
    let network = build(
        "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=giou\niou_thresh_kind=diou\nfront_loss_normalizer=0.7\ncovariance_normalizer=1.3\ncobb_normalizer=0.4\nobb_length_ratio_k=1.4\nobb_viou_weight=1.5\ncobb_box_weight=1.2\ncobb_r_weight=0.8\ncobb_s_weight=1.4\ncobb_iou_lambda=2.2\n",
    )
    .expect("OBB score options parse");
    let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("OBB score options route into training configs");
    assert_eq!(
        training.match_config.score_config,
        training.box_delta_config.score
    );
    assert_eq!(
        training.match_config.score_config.covariance_normalizer,
        1.3
    );
    assert!(!training.match_config.score_config.ciou_obb);
    assert_eq!(spec.iou_thresh_kind, model::YoloIouLoss::Diou);
    assert_eq!(
        training.front_point_delta_config.iou_loss,
        model::YoloIouLoss::Giou
    );
    assert_eq!(
        training
            .match_config
            .score_config
            .covariance_front_length_weight,
        0.7
    );
    assert_eq!(training.match_config.score_config.cobb_normalizer, 0.4);
    assert_eq!(
        training
            .match_config
            .score_config
            .covariance_front_length_ratio_k,
        1.4
    );
    assert_eq!(training.match_config.score_config.viou_weight, 1.0);
    assert_eq!(
        training.match_config.score_config.cobb_loss,
        crate::obb_geometry::ObbCobbLossConfig {
            box_weight: 1.2,
            ratio_weight: 0.8,
            score_weight: 1.4,
            iou_lambda: 2.2,
        }
    );
}

#[test]
fn obb_yolo_mse_uses_ciou_obb_box_delta_branch_with_weighted_matching() {
    let network = build(
        "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=mse\ncovariance_normalizer=1.0\n",
    )
    .expect("weighted MSE OBB config parses");
    let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("weighted MSE OBB config builds");

    assert!(!training.match_config.score_config.ciou_obb);
    assert!(training.box_delta_config.score.ciou_obb);
    assert_eq!(
        training.match_config.score_config.covariance_normalizer,
        training.box_delta_config.score.covariance_normalizer
    );
}

#[test]
fn obb_yolo_defaults_to_ciou_obb_match_scoring() {
    let network = build(
        "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\n",
    )
    .expect("default OBB loss parses");
    let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert!(spec.ciou_obb_loss);
    assert_eq!(spec.iou_loss, model::YoloIouLoss::Ciou);
    assert_eq!(spec.iou_thresh_kind, model::YoloIouLoss::Iou);
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("default OBB score config builds");
    assert!(training.match_config.score_config.ciou_obb);

    let multiplicative = build(
        "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=ciou_obb_mul\n",
    )
    .expect("multiplicative CIoU-OBB parses");
    let LayerKind::ObbYolo(spec) = &multiplicative.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert!(spec.ciou_obb_loss);
    assert!(spec.ciou_obb_multiplicative_loss);
    assert_eq!(spec.iou_loss, model::YoloIouLoss::Ciou);
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
            .expect("multiplicative CIoU-OBB training config builds");
    assert!(training.box_delta_config.ciou_obb_multiplicative_loss);
    assert!(
        training
            .front_point_delta_config
            .ciou_obb_multiplicative_loss
    );

    let invalid = build(
        "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=unsupported\n",
    );
    assert!(invalid.is_err(), "unsupported OBB iou_loss must fail");
}

#[test]
fn parses_obb_anchor_pair_and_directional_quadruple_forms() {
    let build = |anchor_key: &str, anchor_values: &str| {
        let cfg = format!(
            "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\n{anchor_key}={anchor_values}\n"
        );
        Network::from_config(ConfigFile::parse_str("obb-anchor-forms.cfg", &cfg).unwrap())
    };

    let directional = build("obb_anchors", "4,2,0.25,-0.5,8,6,-0.4,0.3")
        .expect("four-value source anchors parse");
    let LayerKind::ObbYolo(spec) = &directional.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert_eq!(spec.anchors, vec![4.0, 2.0, 8.0, 6.0]);
    assert_eq!(
        spec.front_anchor_offsets,
        Some(vec![[0.25, -0.5], [-0.4, 0.3]])
    );
    let training =
        crate::obb_training::build_obb_yolo_training_head_config(spec, 7, 2, 3, 640, 480)
            .expect("parsed spec builds matching and assignment configs");
    assert_eq!(training.match_config.head_index, 7);
    assert_eq!(
        (
            training.match_config.grid_width,
            training.match_config.grid_height
        ),
        (2, 3)
    );
    assert_eq!(
        training.match_config.front_anchor_offsets,
        spec.front_anchor_offsets
    );
    assert_eq!(
        training.assignment_config.front_anchor_offsets,
        spec.front_anchor_offsets
    );
    assert_eq!(training.front_point_delta_config.normalizer, 0.35);
    assert_eq!(
        training.box_delta_config.iou_normalizer,
        spec.iou_normalizer
    );

    let direction_free =
        build("obb_anchors", "4,2,8,6").expect("two-value source anchors parse");
    let LayerKind::ObbYolo(spec) = &direction_free.layers[0].kind else {
        panic!("expected OBB YOLO layer");
    };
    assert_eq!(spec.anchors, vec![4.0, 2.0, 8.0, 6.0]);
    assert_eq!(spec.front_anchor_offsets, None);

    for values in [
        "4,2,0.25",                  // wrong cardinality
        "4,2,NaN,-0.5,8,6,-0.4,0.3", // non-finite directional offset
    ] {
        assert!(build("obb_anchors", values).is_err(), "accepted {values}");
    }
    let both_keys = Network::from_config(
        ConfigFile::parse_str(
            "obb-anchor-forms.cfg",
            "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\nanchors=4,2,8,6\nobb_anchors=4,2,8,6\n",
        )
        .unwrap(),
    );
    assert!(both_keys.is_err(), "ambiguous anchor keys must fail");

    let base = "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\nobb_anchors=4,2,8,6\n";
    let parse_front_normalizer = |options: &str| {
        let cfg = format!("{base}{options}");
        Network::from_config(ConfigFile::parse_str("obb-front-loss.cfg", &cfg).unwrap())
            .unwrap()
    };
    let read_front_normalizer = |network: &Network| {
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        spec.front_loss_normalizer
    };
    assert_eq!(read_front_normalizer(&parse_front_normalizer("")), 0.35);
    assert_eq!(
        read_front_normalizer(&parse_front_normalizer("front_point_normalizer=0.8\n")),
        0.8
    );
    assert_eq!(
        read_front_normalizer(&parse_front_normalizer("front_loss_normalizer=0.6\n")),
        0.6
    );
    assert_eq!(
        read_front_normalizer(&parse_front_normalizer(
            "front_point_normalizer=0.8\nfront_loss_normalizer=0.4\n"
        )),
        0.4,
        "front_loss_normalizer takes precedence over its alias"
    );
}
