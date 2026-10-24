//! CPU and backend-generic Darknet execution.
//!
//! The execution lifecycle mirrors C++ Darknet without reproducing its raw
//! pointer ownership:
//!
//! 1. [`crate::model::Network`] describes layers and shapes.
//! 2. [`ExecutableNetwork`] owns parameter values, train buffers, and CPU
//!    scratch storage.
//! 3. `forward` records activation, recurrent, and dropout state needed by
//!    backward.
//! 4. `backward` walks the graph in reverse and returns input/parameter
//!    gradients; accumulation and SGD update persistent buffers separately.
//! 5. `GpuExecutableNetwork` uploads the same parameter model and uses the CPU
//!    reference for operations without a native backend kernel.
//!
//! Tensor indexing is planar and contiguous: each batch contains all channels,
//! each channel contains `height * width` values, and channels are stored in
//! channel-major order exactly like Darknet's `float *output` arrays.

use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::clifford::{
    CliffordBranchParams, CliffordBranchTrainBuffers, CliffordParams, CliffordRuntimeCache,
    CliffordTrainBuffers,
};
use crate::error::{Error, Result};
use crate::image::PlanarRgbImage;
use crate::maxpool_depth::{backward_maxpool_depth, forward_maxpool_depth};
use crate::model::{
    Activation, ConnectedSpec, ConvolutionalSpec, CrnnSpec, CropSpec, Dcnv4Spec, DeformConvSpec,
    DptHeadSpec, DropblockSpec, EmlConvolutionalSpec, GraphConvSpec, GruSpec, LayerKind, LocalSpec,
    LrPolicy, LstmSpec, NetConfig, Network, RegionSpec, RnnSpec, RouteSpec, TensorShape,
};
use crate::muon::MuonMatrixState;
use crate::muon_runtime::{
    MuonLayerState, apply_staged_layer, matrix_shapes, stage_layer, stage_layer_with_orthogonalizer,
};
use crate::weights::WeightsHeader;
use lightnet_cuda::{
    AdamParams, AdamUpdateItem, DenseLossBackend, ElementwiseLaunchConfig, F16TensorOpsBackend,
    F32ActivationKind, F32TensorOpsBackend, NanInfMode, SgdUpdateItem, dropout_unit,
};

pub(crate) const BATCH_NORM_EPSILON: f32 = 0.000_01;
/// Rolling-stat EMA keep factor of the GPU batch-statistics training path:
/// C++ CUDA `scal_ongpu(.99)` + `axpy_ongpu(.01)` / cuDNN factor `.01`.
pub(crate) const BATCH_NORM_ROLLING_KEEP: f32 = 0.99;
/// Rolling-stat EMA keep factor of the host (CPU) training path: C++
/// `forward_batchnorm_layer` uses `scal_cpu(.9)` + `axpy_cpu(.1)`.
pub(crate) const BATCH_NORM_ROLLING_KEEP_CPU: f32 = 0.9;

use crate::layers::dcnv4::gpu as dcnv4_gpu;
mod loss_scaler;
use crate::layers::laya_act_head::gpu as laya_act_gpu;
use crate::layers::laya_encoder::gpu as laya_encoder_gpu;
use crate::layers::vit::gpu as vit_gpu;
use crate::layers::laya_head::gpu as laya_head_gpu;
use crate::layers::laya_scorer::gpu as laya_scorer_gpu;

mod types;
mod gpu_types;
mod host_adam;
mod exec_core;
mod exec_forward;
mod exec_backward;
mod exec_accumulate;
mod gpu_f16;
mod gpu_state;
mod gpu_update;
mod gpu_forward;
mod gpu_forward_compact;
mod gpu_backward;
mod gpu_backward_support;
mod gpu_dense_loss;
mod gpu_params;
mod defaults;
mod param_updates;
mod weights_io;
mod forward_layers;
mod recurrent_forward;
mod recurrent_backward;
mod layer_ops;

#[cfg(test)]
mod tests;

pub use self::types::*;
pub use self::gpu_types::*;
pub use self::loss_scaler::{LossScaler, ScaleStep};
use self::host_adam::*;
pub use self::exec_core::*;
pub use self::exec_accumulate::*;
use self::defaults::*;
pub(crate) use crate::layers::hyperbolic_foveation::gpu::*;
use self::param_updates::*;
pub(crate) use self::weights_io::*;
pub(crate) use self::forward_layers::*;
pub(crate) use self::recurrent_forward::*;
use self::recurrent_backward::*;
use self::layer_ops::*;
pub(crate) use self::layer_ops::{
    activate, forward_sam, forward_scale_channels, plan_crop_forward, resolve_relative_layer,
    softmax_group_size,
};
