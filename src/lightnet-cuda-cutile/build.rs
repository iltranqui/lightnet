//! Optional build-time AOT smoke compiler.
//!
//! The hello-world probe verifies embedding works, while `sgd_update_flat`
//! provides the first useful flat-pointer runtime AOT kernel. The runtime
//! kernel catalogue can grow independently from these build-time artifacts.

use std::env;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;

use cutile::compile_api::KernelCompiler;
use cutile::tile_kernel::{CompileOptions, FunctionKey, TileFunctionKey};
use cutile_compiler::cuda_tile_runtime_utils::{compile_tile_ir_module, tileiras_binary};

const DEFAULT_SM: &str = "sm_89";

#[cutile::module]
mod cutile_lightnet_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    #[allow(unused_variables)] // consumed by `cuda_tile_print!` during cuTile lowering
    fn hello_world_kernel() {
        let pids: (i32, i32, i32) = get_tile_block_id();
        let npids: (i32, i32, i32) = get_num_tile_blocks();
        cuda_tile_print!(
            "lightnet-cuda tile <{}, {}, {}> of <{}, {}, {}>\n",
            pids.0,
            pids.1,
            pids.2,
            npids.0,
            npids.1,
            npids.2
        );
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn crop_bilinear_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        source_x_ptr: *mut f32,
        source_y_ptr: *mut f32,
        scale_ptr: *mut f32,
        offset_ptr: *mut f32,
        output_len: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let sx_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_x_ptr);
        let sx_base: PointerTile<*mut f32, { [1] }> = sx_root.reshape(const_shape![1]);
        let sx_ptrs: PointerTile<*mut f32, { [128] }> = sx_base.broadcast(const_shape![128]);
        let sy_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_y_ptr);
        let sy_base: PointerTile<*mut f32, { [1] }> = sy_root.reshape(const_shape![1]);
        let sy_ptrs: PointerTile<*mut f32, { [128] }> = sy_base.broadcast(const_shape![128]);
        let scale_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(scale_ptr);
        let scale_base: PointerTile<*mut f32, { [1] }> = scale_root.reshape(const_shape![1]);
        let scale_ptrs: PointerTile<*mut f32, { [128] }> = scale_base.broadcast(const_shape![128]);
        let offset_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(offset_ptr);
        let offset_base: PointerTile<*mut f32, { [1] }> = offset_root.reshape(const_shape![1]);
        let offset_ptrs: PointerTile<*mut f32, { [128] }> =
            offset_base.broadcast(const_shape![128]);
        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let plane = output_height * output_width;
        let output_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(plane, const_shape![128]);
        let channel_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels * plane, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let input_width_limit: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_idx: Tile<i32, { [128] }> = output_idx / channel_plane_tile;
            let in_batch: Tile<i32, { [128] }> = output_idx % channel_plane_tile;
            let channel_idx: Tile<i32, { [128] }> = in_batch / output_plane_tile;
            let pixel_idx: Tile<i32, { [128] }> = in_batch % output_plane_tile;
            let coord_idx: Tile<i32, { [128] }> = batch_idx * output_plane_tile + pixel_idx;
            let source_x_src: PointerTile<*mut f32, { [128] }> = sx_ptrs.offset_tile(coord_idx);
            let source_x_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_x_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_x = source_x_load.0;
            let source_y_src: PointerTile<*mut f32, { [128] }> = sy_ptrs.offset_tile(coord_idx);
            let source_y_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_y_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_y = source_y_load.0;
            let ix: Tile<i32, { [128] }> = convert_tile(floor(source_x));
            let iy: Tile<i32, { [128] }> = convert_tile(floor(source_y));
            let dx: Tile<f32, { [128] }> = source_x - convert_tile(ix);
            let dy: Tile<f32, { [128] }> = source_y - convert_tile(iy);
            let one = broadcast_scalar(1.0f32, const_shape![128]);
            let w00 = (one - dy) * (one - dx);
            let w01 = dy * (one - dx);
            let w10 = (one - dy) * dx;
            let w11 = dy * dx;
            let scale_src: PointerTile<*mut f32, { [128] }> = scale_ptrs.offset_tile(batch_idx);
            let scale_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(1.0f32),
                None,
                Latency::<0>,
            );
            let scale = scale_load.0;
            let offset_src: PointerTile<*mut f32, { [128] }> = offset_ptrs.offset_tile(batch_idx);
            let adjustment_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                offset_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let adjustment = adjustment_load.0;
            let input_base = batch_idx
                * broadcast_scalar(channels * input_height * input_width, const_shape![128])
                + channel_idx * input_plane_tile;
            let mut value: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            let valid00 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let offsets00 = input_base + iy * input_width_tile + ix;
            let sample00_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(select(valid00, offsets00, zero_i32));
            let sample00_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample00_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid00),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let sample00 = sample00_load.0;
            value = value + sample00 * w00;
            let iy1 = iy + broadcast_scalar(1i32, const_shape![128]);
            let valid01 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets01 = input_base + iy1 * input_width_tile + ix;
            let sample01_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(select(valid01, offsets01, zero_i32));
            let sample01_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample01_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid01),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let sample01 = sample01_load.0;
            value = value + sample01 * w01;
            let ix1 = ix + broadcast_scalar(1i32, const_shape![128]);
            let valid10 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let offsets10 = input_base + iy * input_width_tile + ix1;
            let sample10_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(select(valid10, offsets10, zero_i32));
            let sample10_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample10_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid10),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let sample10 = sample10_load.0;
            value = value + sample10 * w10;
            let valid11 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets11 = input_base + iy1 * input_width_tile + ix1;
            let sample11_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(select(valid11, offsets11, zero_i32));
            let sample11_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample11_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid11),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let sample11 = sample11_load.0;
            value = value + sample11 * w11;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store: Token = store_ptr_tko(
                dst,
                value * scale + adjustment,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn crop_bilinear_backward_flat(
        input_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        source_x_ptr: *mut f32,
        source_y_ptr: *mut f32,
        scale_ptr: *mut f32,
        output_len: i32,
        channels: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();
        let grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let grad_base: PointerTile<*mut f32, { [1] }> = grad_root.reshape(const_shape![1]);
        let grad_ptrs: PointerTile<*mut f32, { [128] }> = grad_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let sx_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_x_ptr);
        let sx_base: PointerTile<*mut f32, { [1] }> = sx_root.reshape(const_shape![1]);
        let sx_ptrs: PointerTile<*mut f32, { [128] }> = sx_base.broadcast(const_shape![128]);
        let sy_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(source_y_ptr);
        let sy_base: PointerTile<*mut f32, { [1] }> = sy_root.reshape(const_shape![1]);
        let sy_ptrs: PointerTile<*mut f32, { [128] }> = sy_base.broadcast(const_shape![128]);
        let scale_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(scale_ptr);
        let scale_base: PointerTile<*mut f32, { [1] }> = scale_root.reshape(const_shape![1]);
        let scale_ptrs: PointerTile<*mut f32, { [128] }> = scale_base.broadcast(const_shape![128]);
        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let plane = output_height * output_width;
        let output_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(plane, const_shape![128]);
        let channel_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels * plane, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let input_width_limit: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask = lt_tile(output_idx, output_len_tile);
            let batch_idx: Tile<i32, { [128] }> = output_idx / channel_plane_tile;
            let in_batch: Tile<i32, { [128] }> = output_idx % channel_plane_tile;
            let channel_idx: Tile<i32, { [128] }> = in_batch / output_plane_tile;
            let pixel_idx: Tile<i32, { [128] }> = in_batch % output_plane_tile;
            let coord_idx: Tile<i32, { [128] }> = batch_idx * output_plane_tile + pixel_idx;
            let source_x_ptr: PointerTile<*mut f32, { [128] }> = sx_ptrs.offset_tile(coord_idx);
            let source_x_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_x_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_x: Tile<f32, { [128] }> = source_x_load.0;
            let source_y_ptr: PointerTile<*mut f32, { [128] }> = sy_ptrs.offset_tile(coord_idx);
            let source_y_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                source_y_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let source_y: Tile<f32, { [128] }> = source_y_load.0;
            let ix: Tile<i32, { [128] }> = convert_tile(floor(source_x));
            let iy: Tile<i32, { [128] }> = convert_tile(floor(source_y));
            let dx = source_x - convert_tile(ix);
            let dy = source_y - convert_tile(iy);
            let one = broadcast_scalar(1.0f32, const_shape![128]);
            let w00 = (one - dy) * (one - dx);
            let w01 = dy * (one - dx);
            let w10 = (one - dy) * dx;
            let w11 = dy * dx;
            let grad_ptr: PointerTile<*mut f32, { [128] }> = grad_ptrs.offset_tile(output_idx);
            let grad_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scale_src: PointerTile<*mut f32, { [128] }> = scale_ptrs.offset_tile(batch_idx);
            let scale_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(1.0f32),
                None,
                Latency::<0>,
            );
            let grad: Tile<f32, { [128] }> = grad_load.0 * scale_load.0;
            let input_base = batch_idx
                * broadcast_scalar(channels * input_height * input_width, const_shape![128])
                + channel_idx * input_plane_tile;
            let ix1 = ix + broadcast_scalar(1i32, const_shape![128]);
            let iy1 = iy + broadcast_scalar(1i32, const_shape![128]);
            let valid00 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let valid01 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let valid10 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let valid11 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let off00 = input_base + iy * input_width_tile + ix;
            let off01 = input_base + iy1 * input_width_tile + ix;
            let off10 = input_base + iy * input_width_tile + ix1;
            let off11 = input_base + iy1 * input_width_tile + ix1;
            let dst00: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid00, off00, zero_i32));
            let _a00: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst00,
                grad * w00,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid00),
                None,
            );
            let dst01: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid01, off01, zero_i32));
            let _a01: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst01,
                grad * w01,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid01),
                None,
            );
            let dst10: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid10, off10, zero_i32));
            let _a10: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst10,
                grad * w10,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid10),
                None,
            );
            let dst11: PointerTile<*mut f32, { [128] }> =
                input_ptrs.offset_tile(select(valid11, off11, zero_i32));
            let _a11: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst11,
                grad * w11,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(valid11),
                None,
            );
        }
    }

    #[cutile::entry(generic_aot = true)]
    unsafe fn sgd_update_flat(
        weights_ptr: *mut f32,
        updates_ptr: *mut f32,
        len: i32,
        lr_over_batch: f32,
        momentum: f32,
        decay_times_batch: f32,
    ) {
        let pid = get_tile_block_id();
        let base_offset = pid.0 * 128i32;
        let idx: Tile<i32, { [128] }> =
            iota(const_shape![128]) + broadcast_scalar(base_offset, const_shape![128]);
        let mask: Tile<bool, { [128] }> = lt_tile(idx, broadcast_scalar(len, const_shape![128]));

        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weights_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);
        let weight_ptrs_offset: PointerTile<*mut f32, { [128] }> = weight_ptrs.offset_tile(idx);

        let update_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(updates_ptr);
        let update_base: PointerTile<*mut f32, { [1] }> = update_root.reshape(const_shape![1]);
        let update_ptrs: PointerTile<*mut f32, { [128] }> =
            update_base.broadcast(const_shape![128]);
        let update_ptrs_offset: PointerTile<*mut f32, { [128] }> = update_ptrs.offset_tile(idx);

        let loaded_weights: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            weight_ptrs_offset,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );
        let loaded_updates: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
            update_ptrs_offset,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            Some(0.0f32),
            None,
            Latency::<0>,
        );

        let weights_tile = loaded_weights.0;
        let updates_tile = loaded_updates.0;
        let decayed_updates =
            updates_tile - broadcast_scalar(decay_times_batch, const_shape![128]) * weights_tile;
        let new_weights =
            weights_tile + broadcast_scalar(lr_over_batch, const_shape![128]) * decayed_updates;
        let new_updates = broadcast_scalar(momentum, const_shape![128]) * decayed_updates;

        let _store_weights = store_ptr_tko(
            weight_ptrs_offset,
            new_weights,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
        let _store_updates = store_ptr_tko(
            update_ptrs_offset,
            new_updates,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

fn parse_sms(raw: Option<&str>) -> Result<Vec<String>, String> {
    let value = raw.unwrap_or(DEFAULT_SM);
    let mut targets = Vec::new();
    for target in value.split(',').map(str::trim) {
        if target.is_empty() {
            return Err("LIGHTNET_CUTILE_AOT_SMS contains an empty target".to_owned());
        }
        let Some(digits) = target.strip_prefix("sm_") else {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        };
        if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!("invalid CUDA SM target `{target}`; expected sm_NN"));
        }
        if !targets.iter().any(|existing| existing == target) {
            targets.push(target.to_owned());
        }
    }
    Ok(targets)
}

fn required_build() -> bool {
    env::var("LIGHTNET_CUTILE_AOT_REQUIRED_BUILD").as_deref() == Ok("1")
}

fn toolchain_available() -> bool {
    let tileiras = tileiras_binary();
    let tileiras_ok = Command::new(&tileiras).arg("--version").output().is_ok();
    let cuda_ok = env::var_os("CUDA_TOOLKIT_PATH")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .map(|path| path.exists())
        .unwrap_or_else(|| Path::new("/usr/local/cuda").exists());
    tileiras_ok && cuda_ok
}

fn hello_cache_key() -> String {
    TileFunctionKey::new(
        "cutile_lightnet_kernels".to_owned(),
        "hello_world_kernel".to_owned(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        CompileOptions::default(),
    )
    .get_hash_string()
}

fn sgd_cache_key() -> String {
    TileFunctionKey::new(
        "cutile_lightnet_kernels".to_owned(),
        "sgd_update_flat".to_owned(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        CompileOptions::default(),
    )
    .get_hash_string()
}

fn crop_cache_key() -> String {
    TileFunctionKey::new(
        "cutile_lightnet_kernels".to_owned(),
        "crop_bilinear_flat".to_owned(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        CompileOptions::default(),
    )
    .get_hash_string()
}

fn crop_backward_cache_key() -> String {
    TileFunctionKey::new(
        "cutile_lightnet_kernels".to_owned(),
        "crop_bilinear_backward_flat".to_owned(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
        CompileOptions::default(),
    )
    .get_hash_string()
}

fn compile_one(sm: &str, out_dir: &Path) -> Result<(PathBuf, String), String> {
    let artifacts = KernelCompiler::new(
        cutile_lightnet_kernels::__module_ast_self,
        "cutile_lightnet_kernels",
        "hello_world_kernel",
    )
    .target(sm)
    .compile()
    .map_err(|error| format!("cuTile compiler failed for {sm}: {error}"))?;

    let temporary_cubin = compile_tile_ir_module(&artifacts.into_module(), sm);
    let destination = out_dir.join(format!("cutile_hello_world_{sm}.cubin"));
    fs::copy(&temporary_cubin, &destination).map_err(|error| {
        format!(
            "copying {temporary_cubin} to {}: {error}",
            destination.display()
        )
    })?;
    let _ = fs::remove_file(temporary_cubin);
    Ok((destination, hello_cache_key()))
}

fn compile_sgd_one(sm: &str, out_dir: &Path) -> Result<(PathBuf, String), String> {
    let artifacts = KernelCompiler::new(
        cutile_lightnet_kernels::__module_ast_self,
        "cutile_lightnet_kernels",
        "sgd_update_flat",
    )
    .target(sm)
    .compile()
    .map_err(|error| format!("cuTile compiler failed for {sm} sgd_update_flat: {error}"))?;

    let temporary_cubin = compile_tile_ir_module(&artifacts.into_module(), sm);
    let destination = out_dir.join(format!("cutile_sgd_update_flat_{sm}.cubin"));
    fs::copy(&temporary_cubin, &destination).map_err(|error| {
        format!(
            "copying {temporary_cubin} to {}: {error}",
            destination.display()
        )
    })?;
    let _ = fs::remove_file(temporary_cubin);
    Ok((destination, sgd_cache_key()))
}

fn compile_crop_one(sm: &str, out_dir: &Path) -> Result<(PathBuf, String), String> {
    let artifacts = KernelCompiler::new(
        cutile_lightnet_kernels::__module_ast_self,
        "cutile_lightnet_kernels",
        "crop_bilinear_flat",
    )
    .target(sm)
    .compile()
    .map_err(|error| format!("cuTile compiler failed for {sm} crop_bilinear_flat: {error}"))?;

    let temporary_cubin = compile_tile_ir_module(&artifacts.into_module(), sm);
    let destination = out_dir.join(format!("cutile_crop_bilinear_flat_{sm}.cubin"));
    fs::copy(&temporary_cubin, &destination).map_err(|error| {
        format!(
            "copying {temporary_cubin} to {}: {error}",
            destination.display()
        )
    })?;
    let _ = fs::remove_file(temporary_cubin);
    Ok((destination, crop_cache_key()))
}

fn compile_crop_backward_one(sm: &str, out_dir: &Path) -> Result<(PathBuf, String), String> {
    let artifacts = KernelCompiler::new(
        cutile_lightnet_kernels::__module_ast_self,
        "cutile_lightnet_kernels",
        "crop_bilinear_backward_flat",
    )
    .target(sm)
    .compile()
    .map_err(|error| {
        format!("cuTile compiler failed for {sm} crop_bilinear_backward_flat: {error}")
    })?;

    let temporary_cubin = compile_tile_ir_module(&artifacts.into_module(), sm);
    let destination = out_dir.join(format!("cutile_crop_bilinear_backward_flat_{sm}.cubin"));
    fs::copy(&temporary_cubin, &destination).map_err(|error| {
        format!(
            "copying {temporary_cubin} to {}: {error}",
            destination.display()
        )
    })?;
    let _ = fs::remove_file(temporary_cubin);
    Ok((destination, crop_backward_cache_key()))
}

fn write_registry(
    out_dir: &Path,
    entries: &[(String, PathBuf, String, &str, &str)],
) -> Result<(), String> {
    let registry = out_dir.join("cutile_aot_registry.rs");
    let mut source = String::from("// @generated by lightnet-cuda-cutile/build.rs\n");
    for (index, (sm, cubin, cache_key, function_entry, kernel_name)) in entries.iter().enumerate() {
        source.push_str(&format!(
            "#[linkme::distributed_slice(cutile::tile_kernel::CUTILE_AOT_KERNELS)]\n"
        ));
        source.push_str(&format!(
            "static CUTILE_AOT_{kernel_name}_{index}: cutile::tile_kernel::AotKernel = cutile::tile_kernel::AotKernel {{\n"
        ));
        source.push_str(&format!("    target_sm: {sm:?},\n"));
        source.push_str(&format!("    cache_key: {cache_key:?},\n"));
        source.push_str(&format!("    function_entry: {function_entry:?},\n"));
        source.push_str(&format!("    cubin: include_bytes!({:?}),\n", cubin));
        source.push_str("};\n\n");
    }
    fs::write(registry, source).map_err(|error| format!("writing AOT registry: {error}"))
}

fn main() {
    println!("cargo:rerun-if-env-changed=LIGHTNET_CUTILE_AOT_SMS");
    println!("cargo:rerun-if-env-changed=LIGHTNET_CUTILE_AOT_REQUIRED_BUILD");
    println!("cargo:rerun-if-env-changed=CUTILE_TILEIRAS_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let required = required_build();
    let sms = match parse_sms(env::var("LIGHTNET_CUTILE_AOT_SMS").ok().as_deref()) {
        Ok(sms) => sms,
        Err(error) => panic!("{error}"),
    };

    let mut entries = Vec::new();
    if !toolchain_available() {
        let message = "CUDA/tileiras unavailable; emitting an empty cuTile AOT registry";
        if required {
            panic!("{message}; LIGHTNET_CUTILE_AOT_REQUIRED_BUILD=1");
        }
        println!("cargo:warning={message}");
    } else {
        for sm in &sms {
            let compiled = catch_unwind(AssertUnwindSafe(|| compile_one(sm, &out_dir)))
                .map_err(|_| format!("cuTile AOT compilation panicked for {sm}"))
                .and_then(|result| result);
            match compiled {
                Ok((cubin, key)) => {
                    entries.push((sm.clone(), cubin, key, "hello_world_kernel_entry", "HELLO"))
                }
                Err(error) if required => panic!("{error}"),
                Err(error) => println!("cargo:warning={error}; emitting an empty AOT registry"),
            }
        }
        for sm in &sms {
            let compiled = catch_unwind(AssertUnwindSafe(|| compile_sgd_one(sm, &out_dir)))
                .map_err(|_| format!("cuTile AOT compilation panicked for {sm} sgd_update_flat"))
                .and_then(|result| result);
            match compiled {
                Ok((cubin, key)) => entries.push((
                    sm.clone(),
                    cubin,
                    key,
                    "sgd_update_flat_entry",
                    "SGD_UPDATE_FLAT",
                )),
                Err(error) if required => panic!("{error}"),
                Err(error) => {
                    println!("cargo:warning={error}; emitting no sgd_update_flat AOT artifact")
                }
            }
        }
        for sm in &sms {
            let compiled = catch_unwind(AssertUnwindSafe(|| compile_crop_one(sm, &out_dir)))
                .map_err(|_| format!("cuTile AOT compilation panicked for {sm} crop_bilinear_flat"))
                .and_then(|result| result);
            match compiled {
                Ok((cubin, key)) => entries.push((
                    sm.clone(),
                    cubin,
                    key,
                    "crop_bilinear_flat_entry",
                    "CROP_BILINEAR_FLAT",
                )),
                Err(error) if required => panic!("{error}"),
                Err(error) => {
                    println!("cargo:warning={error}; emitting no crop_bilinear_flat AOT artifact")
                }
            }
        }
        for sm in &sms {
            let compiled =
                catch_unwind(AssertUnwindSafe(|| compile_crop_backward_one(sm, &out_dir)))
                    .map_err(|_| {
                        format!(
                            "cuTile AOT compilation panicked for {sm} crop_bilinear_backward_flat"
                        )
                    })
                    .and_then(|result| result);
            match compiled {
                Ok((cubin, key)) => entries.push((
                    sm.clone(),
                    cubin,
                    key,
                    "crop_bilinear_backward_flat_entry",
                    "CROP_BILINEAR_BACKWARD_FLAT",
                )),
                Err(error) if required => panic!("{error}"),
                Err(error) => println!(
                    "cargo:warning={error}; emitting no crop_bilinear_backward_flat AOT artifact"
                ),
            }
        }
    }

    if let Err(error) = write_registry(&out_dir, &entries) {
        panic!("{error}");
    }
}
