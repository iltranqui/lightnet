    use crate::TensorShape;
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{ExecutableNetwork, LayerParams, Tensor};

    use lightnet_runtime::carpn::{
        ObbRefineProposal, collect_proposals, sample_feature_bilinear, sample_proposal_features,
    };

    fn tiny_exec(objectness: f32) -> ExecutableNetwork {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "carpn-forward.cfg",
                r#"
                    [net]
                    width=2
                    height=1
                    channels=1
                    batch=1
                    obb=1

                    [convolutional]
                    filters=2
                    size=1
                    batch_normalize=0
                    activation=linear

                    [convolutional]
                    filters=8
                    size=1
                    batch_normalize=0
                    activation=linear

                    [obb_yolo]
                    classes=1
                    num=1
                    mask=0
                    anchors=1,1
                    new_coords=1
                    max=1

                    [obb_refine]
                    from=0
                    proposal_from=2
                    proposal_topk_train=2
                    proposal_topk_infer=2
                    proposal_capacity=2
                    sample_grid=1
                    front_range=1
                "#,
            )
            .expect("CARPN cfg"),
        )
        .expect("CARPN network");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("expected feature convolution parameters");
        };
        feature.weights.copy_from_slice(&[2.0, 3.0]);
        feature.biases.fill(0.0);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("expected proposal convolution parameters");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.9, 0.4, 0.5, 0.5, 0.0, 0.0, objectness, 0.8]);
        let LayerParams::Connected(refine) = &mut executable.params[3] else {
            panic!("expected CARPN parameters");
        };
        refine.biases.fill(0.0);
        refine.weights.fill(0.0);
        refine.weights[0] = 0.1;
        refine.weights[3] = 0.1;
        refine.weights[4] = 0.05;
        refine.weights[7] = 0.2;
        refine.weights[8] = -0.1;
        executable
    }

    #[test]
    fn carpn_refine_forward_executes_routed_feature_and_proposal_sources() {
        let executable = tiny_exec(0.9);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("CARPN forward");
        assert_eq!(forward.yolo_heads.len(), 1);
        assert_eq!(forward.yolo_heads[0].layer_index, 2);
        assert_eq!(forward.obb_refine_outputs.len(), 1);

        let refine = &forward.obb_refine_outputs[0];
        assert_eq!(refine.layer_index, 3);
        assert_eq!(refine.proposal_count, 2);
        assert_eq!(refine.proposals.len(), 2);
        let proposal = &refine.proposals[0];
        assert_eq!(proposal.source_layer_index, 2);
        assert_eq!(proposal.anchor_index, 0);
        assert_eq!(
            (proposal.cell_x, proposal.cell_y, proposal.class_id),
            (0, 0, 0)
        );
        assert_eq!(proposal.bbox, [0.45, 0.4, 0.5, 1.0]);
        assert_eq!((proposal.anchor_x, proposal.anchor_y), (0.25, 0.5));
        assert!((proposal.anchor_radius - 0.5 * 1.25f32.sqrt()).abs() < 1e-6);
        assert_eq!(proposal.objectness, 0.9);
        assert_eq!(proposal.class_probability, 0.8);
        assert!((proposal.score - 0.72).abs() < 1e-6);
        let expected_first = [0.29, 0.435, 0.145, 0.87, -0.29];
        for (actual, expected) in proposal.raw_regression.iter().zip(expected_first) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let second = &refine.proposals[1];
        assert_eq!((second.cell_x, second.cell_y), (1, 0));
        assert_eq!(second.bbox, [0.95, 0.4, 0.5, 1.0]);
        assert_eq!((second.anchor_x, second.anchor_y), (0.75, 0.5));
        let expected_second = [0.39, 0.585, 0.195, 1.17, -0.39];
        for (actual, expected) in second.raw_regression.iter().zip(expected_second) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let raw = &forward.layer_outputs[3];
        assert_eq!(raw.shape, TensorShape::new(2, 1, 5, 1));
        let expected_channel_major = [
            0.29, 0.39, 0.435, 0.585, 0.145, 0.195, 0.87, 1.17, -0.29, -0.39,
        ];
        for (actual, expected) in raw.data.iter().zip(expected_channel_major) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn carpn_refine_forward_keeps_zero_proposals_as_padded_output() {
        let executable = tiny_exec(0.005);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("CARPN forward");
        assert_eq!(forward.obb_refine_outputs[0].proposal_count, 0);
        assert!(forward.obb_refine_outputs[0].proposals.is_empty());
        assert_eq!(forward.layer_outputs[3].shape, TensorShape::new(2, 1, 5, 1));
        assert_eq!(forward.layer_outputs[3].data, [0.0; 10]);
    }

    #[test]
    fn carpn_proposals_apply_topk_source_order_ties_and_nonfinite_filtering() {
        use crate::model::LayerKind;

        let cfg = r#"
            [net]
            width=2
            height=1
            channels=1
            batch=1
            obb=1
            [convolutional]
            filters=2
            size=1
            batch_normalize=0
            activation=linear
            [convolutional]
            filters=9
            size=1
            batch_normalize=0
            activation=linear
            [obb_yolo]
            classes=2
            num=1
            mask=0
            anchors=1,1
            new_coords=1
            [obb_refine]
            from=0
            proposal_from=2
            proposal_topk_train=1
            proposal_topk_infer=1
            proposal_capacity=1
            sample_grid=1
        "#;
        let network =
            Network::from_config(ConfigFile::parse_str("carpn-topk.cfg", cfg).expect("cfg"))
                .expect("network");
        let LayerKind::ObbRefine(refine) = &network.layers[3].kind else {
            panic!("expected CARPN refine layer");
        };
        let LayerKind::ObbYolo(source) = &network.layers[2].kind else {
            panic!("expected OBB proposal head");
        };

        let mut data = vec![0.0; 9 * 2];
        for cell in 0..2 {
            data[6 * 2 + cell] = 0.9;
            data[7 * 2 + cell] = 0.8;
            data[8 * 2 + cell] = 0.8;
        }
        data[6 * 2 + 1] = 0.0;
        let head =
            Tensor::from_data(TensorShape::new(2, 1, 9, 1), data.clone()).expect("head tensor");
        let tied = collect_proposals(&network, refine, 2, source, &head).expect("tied proposals");
        assert_eq!(tied.len(), 1);
        assert_eq!((tied[0].cell_x, tied[0].class_id), (0, 0));
        assert_eq!(tied[0].source_layer_index, 2);

        data[7 * 2] = 0.7;
        data[8 * 2] = 0.6;
        data[6 * 2 + 1] = 0.9;
        data[7 * 2 + 1] = 0.5;
        data[8 * 2 + 1] = 0.9;
        let ranked = Tensor::from_data(TensorShape::new(2, 1, 9, 1), data.clone())
            .expect("ranked head tensor");
        let top =
            collect_proposals(&network, refine, 2, source, &ranked).expect("ranked proposals");
        assert_eq!(top.len(), 1);
        assert_eq!((top[0].cell_x, top[0].class_id), (1, 1));

        data[6 * 2] = f32::NAN;
        data[6 * 2 + 1] = 0.0;
        let non_finite =
            Tensor::from_data(TensorShape::new(2, 1, 9, 1), data).expect("non-finite head tensor");
        assert!(
            collect_proposals(&network, refine, 2, source, &non_finite)
                .expect("non-finite proposals")
                .is_empty()
        );
    }

    #[test]
    fn carpn_feature_sampling_uses_grid_order_bilinear_values_and_clamping() {
        let feature = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 3.0, 5.0, 7.0])
            .expect("feature");
        assert_eq!(
            sample_feature_bilinear(&feature, 0, 0, 0.5, 0.5).unwrap(),
            4.0
        );
        assert_eq!(
            sample_feature_bilinear(&feature, 0, 0, -1.0, 2.0).unwrap(),
            5.0
        );
        let proposal = ObbRefineProposal {
            proposal_index: 0,
            source_layer_index: 0,
            anchor_index: 0,
            cell_x: 0,
            cell_y: 0,
            class_id: 0,
            objectness: 1.0,
            class_probability: 1.0,
            score: 1.0,
            bbox: [0.5, 0.5, 1.0, 1.0],
            anchor_x: 0.5,
            anchor_y: 0.5,
            anchor_radius: 1.0,
            raw_regression: [0.0; 5],
        };
        assert_eq!(
            sample_proposal_features(&feature, &proposal, 2).unwrap(),
            [1.0, 3.0, 5.0, 7.0]
        );
        let non_finite = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![f32::NAN])
            .expect("non-finite feature");
        assert_eq!(
            sample_feature_bilinear(&non_finite, 0, 0, 0.0, 0.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn carpn_refine_forward_rejects_training_until_stage_two_updates_exist() {
        let executable = tiny_exec(0.9);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let error = executable
            .forward_training(&input)
            .expect_err("CARPN training is unsupported");
        assert!(error.to_string().contains("[obb_refine]"));
    }
