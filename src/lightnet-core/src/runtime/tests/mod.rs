use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::cfg::ConfigFile;
use crate::model::{
    Activation, ConvolutionalSpec, LayerKind, LrPolicy, NetConfig, Network, TensorShape,
};
use crate::muon::MuonMatrixState;
use crate::train::{YoloLabel, compute_yolo_training_loss};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use super::*;
use super::{
    ConvolutionalParams, ExecutableNetwork, ForwardResult, GpuExecutableNetwork,
    HostAdamLayerState, LayerParams, LayerTrainBuffers, MuonLayerState, Tensor, backward_reorg,
    backward_route, backward_softmax, backward_upsample, compute_learning_rate, forward_reorg,
    forward_route, forward_softmax, forward_upsample, sgd_update, write_muon_layer_state,
};


mod tests_01;
mod tests_02;
mod tests_03;
mod tests_04;
mod tests_05;
mod tests_06;
mod tests_07;

use self::tests_01::*;
use self::tests_02::*;
use self::tests_03::*;
use self::tests_04::*;
use self::tests_05::*;
use self::tests_06::*;
use self::tests_07::*;

#[path = "dcnv4_gpu_tests.rs"]
mod dcnv4_gpu_tests;
#[path = "depth_anything_runtime_tests.rs"]
mod depth_anything_runtime_tests;
#[path = "dpt_head_runtime_tests.rs"]
mod dpt_head_runtime_tests;
#[path = "foveation_film_runtime_tests.rs"]
mod foveation_film_runtime_tests;
#[path = "hyperbolic_foveation_runtime_tests.rs"]
mod hyperbolic_foveation_runtime_tests;
#[path = "laya_act_gpu_tests.rs"]
mod laya_act_gpu_tests;
#[path = "marker_gather_gpu_tests.rs"]
mod marker_gather_gpu_tests;
#[path = "yolo_3d_runtime_tests.rs"]
mod yolo_3d_runtime_tests;
