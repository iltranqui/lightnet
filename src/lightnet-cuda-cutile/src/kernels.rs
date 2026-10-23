#[cutile::module]
pub(crate) mod cutile_lightnet_kernels {
    use cutile::core::*;

    unsafe fn get_tensor_1d<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        unsafe { make_tensor_view(ptr_tile, shape, strides, new_token_unordered()) }
    }

    #[cutile::entry()]
    fn hello_world_kernel() {
        let _pids: (i32, i32, i32) = get_tile_block_id();
        let _npids: (i32, i32, i32) = get_num_tile_blocks();
        cuda_tile_print!(
            "lightnet-cuda tile <{}, {}, {}> of <{}, {}, {}>\n",
            _pids.0,
            _pids.1,
            _pids.2,
            _npids.0,
            _npids.1,
            _npids.2
        );
    }

    #[cutile::entry()]
    fn vector_add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }

    #[cutile::entry()]
    fn saxpy<const S: [i32; 1]>(
        alpha: f32,
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_alpha: Tile<f32, S> = alpha.broadcast(z.shape());
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_alpha * tile_x + tile_y);
    }

    #[cutile::entry()]
    fn bias_add_2d<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        bias: &Tensor<f32, { [-1] }>,
        channels: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_idx = pid.0;
        let channel_idx = row_idx % channels;
        let tile_x: Tile<f32, S> = load_tile_like(x, z);
        let bias_scalar = bias.load_tile(const_shape![1], [channel_idx]);
        let bias_tile: Tile<f32, S> = bias_scalar.reshape(const_shape![1, 1]).broadcast(z.shape());
        z.store(tile_x + bias_tile);
    }

    #[cutile::entry()]
    fn channel_scale_2d<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        scale: &Tensor<f32, { [-1] }>,
        channels: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row_idx = pid.0;
        let channel_idx = row_idx % channels;
        let tile_x: Tile<f32, S> = load_tile_like(x, z);
        let scale_scalar = scale.load_tile(const_shape![1], [channel_idx]);
        let scale_tile: Tile<f32, S> = scale_scalar
            .reshape(const_shape![1, 1])
            .broadcast(z.shape());
        z.store(tile_x * scale_tile);
    }

    #[cutile::entry()]
    unsafe fn bias_backward_flat(
        bias_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        len: i32,
        channels: i32,
        channel_stride: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let grad_base: PointerTile<*mut f32, { [1] }> = grad_root.reshape(const_shape![1]);
        let grad_ptrs: PointerTile<*mut f32, { [128] }> = grad_base.broadcast(const_shape![128]);
        let bias_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(bias_grad_ptr);
        let bias_base: PointerTile<*mut f32, { [1] }> = bias_root.reshape(const_shape![1]);
        let bias_ptrs: PointerTile<*mut f32, { [128] }> = bias_base.broadcast(const_shape![128]);

        let len_tile: Tile<i32, { [128] }> = broadcast_scalar(len, const_shape![128]);
        let channel_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channel_stride, const_shape![128]);
        let channels_tile: Tile<i32, { [128] }> = broadcast_scalar(channels, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..len).step_by(step as usize) {
            let idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(idx, len_tile);
            let channel_idx = (idx / channel_stride_tile) % channels_tile;
            let grad_src: PointerTile<*mut f32, { [128] }> = grad_ptrs.offset_tile(idx);
            let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let grad_values: Tile<f32, { [128] }> = grad_tile.0;
            let bias_dst: PointerTile<*mut f32, { [128] }> = bias_ptrs.offset_tile(channel_idx);
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                bias_dst,
                grad_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(mask),
                None,
            );
        }
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
            let source_x: Tile<f32, { [128] }> = source_x_load.0;
            let source_y: Tile<f32, { [128] }> = source_y_load.0;
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
            let batch_scales: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(1.0f32),
                None,
                Latency::<0>,
            );
            let offset_src: PointerTile<*mut f32, { [128] }> = offset_ptrs.offset_tile(batch_idx);
            let batch_offsets: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                offset_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scale: Tile<f32, { [128] }> = batch_scales.0;
            let adjustment: Tile<f32, { [128] }> = batch_offsets.0;
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
            let safe_offsets00: Tile<i32, { [128] }> = select(valid00, offsets00, zero_i32);
            let sample00_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets00);
            let sample00: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample00_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid00),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample00.0 * w00;
            let iy1 = iy + broadcast_scalar(1i32, const_shape![128]);
            let valid01 = output_mask
                & ge_tile(ix, zero_i32)
                & lt_tile(ix, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets01 = input_base + iy1 * input_width_tile + ix;
            let safe_offsets01: Tile<i32, { [128] }> = select(valid01, offsets01, zero_i32);
            let sample01_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets01);
            let sample01: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample01_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid01),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample01.0 * w01;
            let ix1 = ix + broadcast_scalar(1i32, const_shape![128]);
            let valid10 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy, zero_i32)
                & lt_tile(iy, input_height_tile);
            let offsets10 = input_base + iy * input_width_tile + ix1;
            let safe_offsets10: Tile<i32, { [128] }> = select(valid10, offsets10, zero_i32);
            let sample10_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets10);
            let sample10: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample10_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid10),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample10.0 * w10;
            let valid11 = output_mask
                & ge_tile(ix1, zero_i32)
                & lt_tile(ix1, input_width_limit)
                & ge_tile(iy1, zero_i32)
                & lt_tile(iy1, input_height_tile);
            let offsets11 = input_base + iy1 * input_width_tile + ix1;
            let safe_offsets11: Tile<i32, { [128] }> = select(valid11, offsets11, zero_i32);
            let sample11_ptr: PointerTile<*mut f32, { [128] }> =
                in_ptrs.offset_tile(safe_offsets11);
            let sample11: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                sample11_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(valid11),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            value = value + sample11.0 * w11;
            let dst = out_ptrs.offset_tile(output_idx);
            let _store = store_ptr_tko(
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
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let input_width_limit: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
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
            let scale_ptr: PointerTile<*mut f32, { [128] }> = scale_ptrs.offset_tile(batch_idx);
            let scale_load: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                scale_ptr,
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

    #[cutile::entry()]
    fn activation_linear<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        z.store(tile_x);
    }

    #[cutile::entry()]
    fn activation_logistic<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(one / (one + exp_neg_x));
    }

    #[cutile::entry()]
    fn activation_relu<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        z.store(max_tile(tile_x, zero));
    }

    #[cutile::entry()]
    fn activation_leaky<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let slope: Tile<f32, S> = constant(0.1f32, z.shape());
        z.store(select(gt_tile(tile_x, zero), tile_x, slope * tile_x));
    }

    #[cutile::entry()]
    fn activation_mish<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let exp_x: Tile<f32, S> = exp(tile_x);
        let softplus: Tile<f32, S> = log(one + exp_x);
        let softplus_tanh: Tile<f32, S> = tanh(softplus);
        z.store(tile_x * softplus_tanh);
    }

    #[cutile::entry()]
    fn activation_swish<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        z.store(tile_x / (one + exp_neg_x));
    }

    #[cutile::entry()]
    fn activation_tanh<const S: [i32; 1]>(z: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let tile_x = load_tile_like(x, z);
        let tanh_x: Tile<f32, S> = tanh(tile_x);
        z.store(tanh_x);
    }

    #[cutile::entry()]
    fn activation_backward_linear<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        _x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_delta = load_tile_like(delta, z);
        z.store(tile_delta);
    }

    #[cutile::entry()]
    fn activation_backward_logistic<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        let sigmoid: Tile<f32, S> = one / (one + exp_neg_x);
        z.store(tile_delta * sigmoid * (one - sigmoid));
    }

    #[cutile::entry()]
    fn activation_backward_relu<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let grad = select(gt_tile(tile_x, zero), one, zero);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_leaky<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let zero: Tile<f32, S> = constant(0.0f32, z.shape());
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let slope: Tile<f32, S> = constant(0.1f32, z.shape());
        let grad = select(gt_tile(tile_x, zero), one, slope);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_mish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let threshold: Tile<f32, S> = constant(20.0f32, z.shape());
        let neg_threshold: Tile<f32, S> = constant(-20.0f32, z.shape());
        let exp_x: Tile<f32, S> = exp(tile_x);
        let log_term: Tile<f32, S> = log(one + exp_x);
        let softplus: Tile<f32, S> = select(
            gt_tile(tile_x, threshold),
            tile_x,
            select(lt_tile(tile_x, neg_threshold), exp_x, log_term),
        );
        let neg_softplus: Tile<f32, S> = negf(softplus);
        let exp_neg_softplus: Tile<f32, S> = exp(neg_softplus);
        let grad_sp: Tile<f32, S> = one - exp_neg_softplus;
        let tsp: Tile<f32, S> = tanh(softplus);
        let grad_tsp: Tile<f32, S> = (one - tsp * tsp) * grad_sp;
        let grad: Tile<f32, S> = tile_x * grad_tsp + tsp;
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_swish<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let neg_x: Tile<f32, S> = negf(tile_x);
        let exp_neg_x: Tile<f32, S> = exp(neg_x);
        let sigmoid: Tile<f32, S> = one / (one + exp_neg_x);
        let grad: Tile<f32, S> = sigmoid + tile_x * sigmoid * (one - sigmoid);
        z.store(tile_delta * grad);
    }

    #[cutile::entry()]
    fn activation_backward_tanh<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        delta: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_delta = load_tile_like(delta, z);
        let one: Tile<f32, S> = constant(1.0f32, z.shape());
        let tanh_x: Tile<f32, S> = tanh(tile_x);
        z.store(tile_delta * (one - tanh_x * tanh_x));
    }

    #[cutile::entry()]
    unsafe fn route_copy_flat(
        dst_ptr: *mut f32,
        src_ptr: *mut f32,
        copy_len: i32,
        source_total_channels: i32,
        source_channel_offset: i32,
        source_channels: i32,
        output_total_channels: i32,
        output_channel_offset: i32,
        channel_stride: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let src_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(src_ptr);
        let src_base: PointerTile<*mut f32, { [1] }> = src_root.reshape(const_shape![1]);
        let src_ptrs: PointerTile<*mut f32, { [128] }> = src_base.broadcast(const_shape![128]);
        let dst_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(dst_ptr);
        let dst_base: PointerTile<*mut f32, { [1] }> = dst_root.reshape(const_shape![1]);
        let dst_ptrs: PointerTile<*mut f32, { [128] }> = dst_base.broadcast(const_shape![128]);

        let copy_len_tile: Tile<i32, { [128] }> = broadcast_scalar(copy_len, const_shape![128]);
        let source_group_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_channels * channel_stride, const_shape![128]);
        let source_batch_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_total_channels * channel_stride, const_shape![128]);
        let source_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(source_channel_offset * channel_stride, const_shape![128]);
        let output_batch_stride: Tile<i32, { [128] }> =
            broadcast_scalar(output_total_channels * channel_stride, const_shape![128]);
        let output_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_channel_offset * channel_stride, const_shape![128]);
        let channel_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channel_stride, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..copy_len).step_by(step as usize) {
            let local_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(local_idx, copy_len_tile);
            let batch_idx = local_idx / source_group_stride;
            let within_batch = local_idx % source_group_stride;
            let channel_idx = within_batch / channel_stride_tile;
            let elem_idx = within_batch % channel_stride_tile;

            let src_offsets = batch_idx * source_batch_stride
                + source_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let src: PointerTile<*mut f32, { [128] }> = src_ptrs.offset_tile(src_offsets);
            let src_values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let src_tile: Tile<f32, { [128] }> = src_values.0;

            let dst_offsets = batch_idx * output_batch_stride
                + output_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let dst: PointerTile<*mut f32, { [128] }> = dst_ptrs.offset_tile(dst_offsets);
            let _store_token: Token = store_ptr_tko(
                dst,
                src_tile,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn route_scatter_flat(
        dst_ptr: *mut f32,
        src_ptr: *mut f32,
        copy_len: i32,
        source_total_channels: i32,
        source_channel_offset: i32,
        source_channels: i32,
        output_total_channels: i32,
        output_channel_offset: i32,
        channel_stride: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let src_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(src_ptr);
        let src_base: PointerTile<*mut f32, { [1] }> = src_root.reshape(const_shape![1]);
        let src_ptrs: PointerTile<*mut f32, { [128] }> = src_base.broadcast(const_shape![128]);
        let dst_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(dst_ptr);
        let dst_base: PointerTile<*mut f32, { [1] }> = dst_root.reshape(const_shape![1]);
        let dst_ptrs: PointerTile<*mut f32, { [128] }> = dst_base.broadcast(const_shape![128]);

        let copy_len_tile: Tile<i32, { [128] }> = broadcast_scalar(copy_len, const_shape![128]);
        let source_selected_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_channels * channel_stride, const_shape![128]);
        let source_group_stride: Tile<i32, { [128] }> =
            broadcast_scalar(source_total_channels * channel_stride, const_shape![128]);
        let source_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(source_channel_offset * channel_stride, const_shape![128]);
        let output_batch_stride: Tile<i32, { [128] }> =
            broadcast_scalar(output_total_channels * channel_stride, const_shape![128]);
        let output_channel_offset_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_channel_offset * channel_stride, const_shape![128]);
        let channel_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channel_stride, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..copy_len).step_by(step as usize) {
            let local_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(local_idx, copy_len_tile);
            let batch_idx = local_idx / source_selected_stride;
            let within_batch = local_idx % source_selected_stride;
            let channel_idx = within_batch / channel_stride_tile;
            let elem_idx = within_batch % channel_stride_tile;

            let src_offsets = batch_idx * source_group_stride
                + source_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let src: PointerTile<*mut f32, { [128] }> = src_ptrs.offset_tile(src_offsets);
            let src_values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let src_tile: Tile<f32, { [128] }> = src_values.0;

            let dst_offsets = batch_idx * output_batch_stride
                + output_channel_offset_tile
                + channel_idx * channel_stride_tile
                + elem_idx;
            let dst: PointerTile<*mut f32, { [128] }> = dst_ptrs.offset_tile(dst_offsets);
            let _store_token: Token = store_ptr_tko(
                dst,
                src_tile,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn upsample_nearest_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        stride: i32,
        scale: f32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let stride_tile: Tile<i32, { [128] }> = broadcast_scalar(stride, const_shape![128]);
        let scale_tile: Tile<f32, { [128] }> = broadcast_scalar(scale, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let input_y = output_y / stride_tile;
            let input_x = output_x / stride_tile;
            let input_offsets = batch_channel
                * broadcast_scalar(input_height * input_width, const_shape![128])
                + input_y * input_width_tile
                + input_x;
            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(input_offsets);
            let values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                scaled_values,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn upsample_reverse_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        input_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        stride: i32,
        scale: f32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let input_len_tile: Tile<i32, { [128] }> = broadcast_scalar(input_len, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let stride_tile: Tile<i32, { [128] }> = broadcast_scalar(stride, const_shape![128]);
        let scale_tile: Tile<f32, { [128] }> = broadcast_scalar(scale, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..input_len).step_by(step as usize) {
            let input_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let mask: Tile<bool, { [128] }> = lt_tile(input_idx, input_len_tile);
            let batch_channel = input_idx / input_plane_tile;
            let within_plane = input_idx % input_plane_tile;
            let input_y = within_plane / input_width_tile;
            let input_x = within_plane % input_width_tile;
            let output_y = input_y / stride_tile;
            let output_x = input_x / stride_tile;
            let output_offsets =
                batch_channel * output_plane_tile + output_y * output_width_tile + output_x;
            let src: PointerTile<*mut f32, { [128] }> = in_ptrs.offset_tile(input_idx);
            let values: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let scaled_values: Tile<f32, { [128] }> = values.0 * scale_tile;
            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_offsets);
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                dst,
                scaled_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(mask),
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn maxpool_flat(
        out_ptr: *mut f32,
        in_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        size: i32,
        stride_x: i32,
        stride_y: i32,
        pad_offset: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let in_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(in_ptr);
        let in_base: PointerTile<*mut f32, { [1] }> = in_root.reshape(const_shape![1]);
        let in_ptrs: PointerTile<*mut f32, { [128] }> = in_base.broadcast(const_shape![128]);
        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let mut acc: Tile<f32, { [128] }> = constant(-1.0e30f32, const_shape![128]);

            for ky in 0i32..size {
                let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                let input_y = output_y * broadcast_scalar(stride_y, const_shape![128]) + ky_tile
                    - broadcast_scalar(pad_offset, const_shape![128]);
                for kx in 0i32..size {
                    let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                    let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                        + kx_tile
                        - broadcast_scalar(pad_offset, const_shape![128]);
                    let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let input_y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, input_height_tile);
                    let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                    let input_x_lt_width: Tile<bool, { [128] }> =
                        lt_tile(input_x, input_width_tile);
                    let valid: Tile<bool, { [128] }> = output_mask
                        & input_y_ge_zero
                        & input_y_lt_height
                        & input_x_ge_zero
                        & input_x_lt_width;
                    let input_offsets =
                        batch_channel * input_plane_tile + input_y * input_width_tile + input_x;
                    let safe_input_offsets = select(valid, input_offsets, zero_i32);
                    let src: PointerTile<*mut f32, { [128] }> =
                        in_ptrs.offset_tile(safe_input_offsets);
                    let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                        src,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(valid),
                        Some(-1.0e30f32),
                        None,
                        Latency::<0>,
                    );
                    let values: Tile<f32, { [128] }> = input_tile.0;
                    let better: Tile<bool, { [128] }> = gt_tile(values, acc);
                    acc = select(better, values, acc);
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn maxpool_backward_flat(
        input_grad_ptr: *mut f32,
        input_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        output_len: i32,
        input_height: i32,
        input_width: i32,
        output_height: i32,
        output_width: i32,
        size: i32,
        stride_x: i32,
        stride_y: i32,
        pad_offset: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);
        let input_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_grad_base: PointerTile<*mut f32, { [1] }> =
            input_grad_root.reshape(const_shape![1]);
        let input_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            input_grad_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let false_tile: Tile<bool, { [128] }> = broadcast_scalar(false, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);
            let batch_channel = output_idx / output_plane_tile;
            let within_plane = output_idx % output_plane_tile;
            let output_y = within_plane / output_width_tile;
            let output_x = within_plane % output_width_tile;
            let mut best_value: Tile<f32, { [128] }> = constant(-1.0e30f32, const_shape![128]);
            let mut best_offset: Tile<i32, { [128] }> = zero_i32;
            let mut best_found: Tile<bool, { [128] }> = false_tile;

            for ky in 0i32..size {
                let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                let input_y = output_y * broadcast_scalar(stride_y, const_shape![128]) + ky_tile
                    - broadcast_scalar(pad_offset, const_shape![128]);
                for kx in 0i32..size {
                    let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                    let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                        + kx_tile
                        - broadcast_scalar(pad_offset, const_shape![128]);
                    let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let input_y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, input_height_tile);
                    let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                    let input_x_lt_width: Tile<bool, { [128] }> =
                        lt_tile(input_x, input_width_tile);
                    let valid: Tile<bool, { [128] }> = output_mask
                        & input_y_ge_zero
                        & input_y_lt_height
                        & input_x_ge_zero
                        & input_x_lt_width;
                    let input_offsets =
                        batch_channel * input_plane_tile + input_y * input_width_tile + input_x;
                    let safe_input_offsets = select(valid, input_offsets, zero_i32);
                    let src: PointerTile<*mut f32, { [128] }> =
                        input_ptrs.offset_tile(safe_input_offsets);
                    let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                        src,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(valid),
                        Some(-1.0e30f32),
                        None,
                        Latency::<0>,
                    );
                    let values: Tile<f32, { [128] }> = input_tile.0;
                    let better: Tile<bool, { [128] }> = valid & gt_tile(values, best_value);
                    best_value = select(better, values, best_value);
                    best_offset = select(better, input_offsets, best_offset);
                    best_found = select(better, valid, best_found);
                }
            }

            let grad_src: PointerTile<*mut f32, { [128] }> =
                output_grad_ptrs.offset_tile(output_idx);
            let output_grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                grad_src,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            let grad_values: Tile<f32, { [128] }> = output_grad_tile.0;
            let grad_dst: PointerTile<*mut f32, { [128] }> =
                input_grad_ptrs.offset_tile(best_offset);
            let write_mask = output_mask & best_found;
            let _atomic_token: (Tile<f32, { [128] }>, Token) = atomic_rmw_tko(
                grad_dst,
                grad_values,
                atomic::AddF,
                ordering::Relaxed,
                scope::Device,
                Some(write_mask),
                None,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_flat(
        out_ptr: *mut f32,
        input_ptr: *mut f32,
        weight_ptr: *mut f32,
        output_len: i32,
        input_channels: i32,
        input_height: i32,
        input_width: i32,
        filters: i32,
        output_height: i32,
        output_width: i32,
        kernel_size: i32,
        stride_x: i32,
        stride_y: i32,
        dilation: i32,
        padding: i32,
        _groups: i32,
        channels_per_group: i32,
        filters_per_group: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let out_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let out_base: PointerTile<*mut f32, { [1] }> = out_root.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut f32, { [128] }> = out_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);

        let output_len_tile: Tile<i32, { [128] }> = broadcast_scalar(output_len, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let filter_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters * output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);
        let input_height_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height, const_shape![128]);
        let input_width_limit_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..output_len).step_by(step as usize) {
            let output_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let output_mask: Tile<bool, { [128] }> = lt_tile(output_idx, output_len_tile);

            let batch_idx = output_idx / filter_plane_tile;
            let batch_rem = output_idx % filter_plane_tile;
            let filter_idx = batch_rem / output_plane_tile;
            let output_rem = batch_rem % output_plane_tile;
            let output_y = output_rem / output_width_tile;
            let output_x = output_rem % output_width_tile;
            let group_idx = filter_idx / filters_per_group_tile;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for ic in 0i32..channels_per_group {
                let ic_tile: Tile<i32, { [128] }> = broadcast_scalar(ic, const_shape![128]);
                let input_channel = group_idx * channels_per_group_tile + ic_tile;
                for ky in 0i32..kernel_size {
                    let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                    let input_y = output_y * broadcast_scalar(stride_y, const_shape![128])
                        + ky_tile * broadcast_scalar(dilation, const_shape![128])
                        - broadcast_scalar(padding, const_shape![128]);
                    for kx in 0i32..kernel_size {
                        let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                        let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                            + kx_tile * broadcast_scalar(dilation, const_shape![128])
                            - broadcast_scalar(padding, const_shape![128]);
                        let input_y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                        let input_y_lt_height: Tile<bool, { [128] }> =
                            lt_tile(input_y, input_height_tile);
                        let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                        let input_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(input_x, input_width_limit_tile);
                        let input_valid: Tile<bool, { [128] }> = output_mask
                            & input_y_ge_zero
                            & input_y_lt_height
                            & input_x_ge_zero
                            & input_x_lt_width;
                        let input_offsets = batch_idx
                            * broadcast_scalar(
                                input_channels * input_height * input_width,
                                const_shape![128],
                            )
                            + input_channel * input_plane_tile
                            + input_y * input_width_tile
                            + input_x;
                        let safe_input_offsets = select(input_valid, input_offsets, zero_i32);
                        let src: PointerTile<*mut f32, { [128] }> =
                            input_ptrs.offset_tile(safe_input_offsets);
                        let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(input_valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let weight_offsets = ((filter_idx * channels_per_group_tile + ic_tile)
                            * broadcast_scalar(kernel_size, const_shape![128])
                            + ky_tile)
                            * broadcast_scalar(kernel_size, const_shape![128])
                            + kx_tile;
                        let weight_ptrs_offset: PointerTile<*mut f32, { [128] }> =
                            weight_ptrs.offset_tile(weight_offsets);
                        let weight_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            weight_ptrs_offset,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(output_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );
                        let product: Tile<f32, { [128] }> = input_tile.0 * weight_tile.0;
                        acc = acc + product;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = out_ptrs.offset_tile(output_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_backward_input_flat(
        input_grad_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        weight_ptr: *mut f32,
        input_len: i32,
        input_channels: i32,
        input_height: i32,
        input_width: i32,
        filters: i32,
        output_height: i32,
        output_width: i32,
        kernel_size: i32,
        stride_x: i32,
        stride_y: i32,
        dilation: i32,
        padding: i32,
        _groups: i32,
        channels_per_group: i32,
        filters_per_group: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let input_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_grad_ptr);
        let input_grad_base: PointerTile<*mut f32, { [1] }> =
            input_grad_root.reshape(const_shape![1]);
        let input_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            input_grad_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);
        let weight_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_ptr);
        let weight_base: PointerTile<*mut f32, { [1] }> = weight_root.reshape(const_shape![1]);
        let weight_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_base.broadcast(const_shape![128]);

        let input_len_tile: Tile<i32, { [128] }> = broadcast_scalar(input_len, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let input_channel_plane_tile: Tile<i32, { [128] }> = broadcast_scalar(
            input_channels * input_height * input_width,
            const_shape![128],
        );
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let kernel_size_tile: Tile<i32, { [128] }> =
            broadcast_scalar(kernel_size, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..input_len).step_by(step as usize) {
            let input_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let input_mask: Tile<bool, { [128] }> = lt_tile(input_idx, input_len_tile);

            let batch_idx = input_idx / input_channel_plane_tile;
            let batch_rem = input_idx % input_channel_plane_tile;
            let input_channel = batch_rem / input_plane_tile;
            let within_plane = batch_rem % input_plane_tile;
            let input_y = within_plane / input_width_tile;
            let input_x = within_plane % input_width_tile;
            let group_idx = input_channel / channels_per_group_tile;
            let input_channel_in_group = input_channel % channels_per_group_tile;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for filter_in_group in 0i32..filters_per_group {
                let filter_in_group_tile: Tile<i32, { [128] }> =
                    broadcast_scalar(filter_in_group, const_shape![128]);
                let filter_idx = group_idx * filters_per_group_tile + filter_in_group_tile;
                for ky in 0i32..kernel_size {
                    let ky_tile: Tile<i32, { [128] }> = broadcast_scalar(ky, const_shape![128]);
                    let oy_num = input_y + broadcast_scalar(padding, const_shape![128])
                        - ky_tile * broadcast_scalar(dilation, const_shape![128]);
                    let oy_ge_zero: Tile<bool, { [128] }> = ge_tile(oy_num, zero_i32);
                    let oy_on_stride: Tile<bool, { [128] }> = eq_tile(
                        oy_num % broadcast_scalar(stride_y, const_shape![128]),
                        zero_i32,
                    );
                    let oy_valid: Tile<bool, { [128] }> = oy_ge_zero & oy_on_stride;
                    let output_y = oy_num / broadcast_scalar(stride_y, const_shape![128]);
                    for kx in 0i32..kernel_size {
                        let kx_tile: Tile<i32, { [128] }> = broadcast_scalar(kx, const_shape![128]);
                        let ox_num = input_x + broadcast_scalar(padding, const_shape![128])
                            - kx_tile * broadcast_scalar(dilation, const_shape![128]);
                        let ox_ge_zero: Tile<bool, { [128] }> = ge_tile(ox_num, zero_i32);
                        let ox_on_stride: Tile<bool, { [128] }> = eq_tile(
                            ox_num % broadcast_scalar(stride_x, const_shape![128]),
                            zero_i32,
                        );
                        let ox_valid: Tile<bool, { [128] }> = ox_ge_zero & ox_on_stride;
                        let output_x = ox_num / broadcast_scalar(stride_x, const_shape![128]);
                        let output_y_ge_zero: Tile<bool, { [128] }> = ge_tile(output_y, zero_i32);
                        let output_y_lt_height: Tile<bool, { [128] }> =
                            lt_tile(output_y, broadcast_scalar(output_height, const_shape![128]));
                        let output_x_ge_zero: Tile<bool, { [128] }> = ge_tile(output_x, zero_i32);
                        let output_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(output_x, broadcast_scalar(output_width, const_shape![128]));
                        let valid: Tile<bool, { [128] }> = input_mask
                            & oy_valid
                            & ox_valid
                            & output_y_ge_zero
                            & output_y_lt_height
                            & output_x_ge_zero
                            & output_x_lt_width;

                        let grad_offsets = batch_idx
                            * broadcast_scalar(
                                filters * output_height * output_width,
                                const_shape![128],
                            )
                            + filter_idx * output_plane_tile
                            + output_y * output_width_tile
                            + output_x;
                        let safe_grad_offsets = select(valid, grad_offsets, zero_i32);
                        let grad_src: PointerTile<*mut f32, { [128] }> =
                            output_grad_ptrs.offset_tile(safe_grad_offsets);
                        let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            grad_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let weight_offsets = ((filter_idx * channels_per_group_tile
                            + input_channel_in_group)
                            * kernel_size_tile
                            + ky_tile)
                            * kernel_size_tile
                            + kx_tile;
                        let weight_src: PointerTile<*mut f32, { [128] }> =
                            weight_ptrs.offset_tile(weight_offsets);
                        let weight_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            weight_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(input_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );
                        acc = acc + grad_tile.0 * weight_tile.0;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = input_grad_ptrs.offset_tile(input_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(input_mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry()]
    unsafe fn conv2d_backward_weights_flat(
        weight_grad_ptr: *mut f32,
        input_ptr: *mut f32,
        output_grad_ptr: *mut f32,
        weight_len: i32,
        input_channels: i32,
        input_height: i32,
        input_width: i32,
        filters: i32,
        output_height: i32,
        output_width: i32,
        kernel_size: i32,
        stride_x: i32,
        stride_y: i32,
        dilation: i32,
        padding: i32,
        _groups: i32,
        batch: i32,
        channels_per_group: i32,
        filters_per_group: i32,
    ) {
        let grid = get_num_tile_blocks();
        let pid = get_tile_block_id();

        let weight_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(weight_grad_ptr);
        let weight_grad_base: PointerTile<*mut f32, { [1] }> =
            weight_grad_root.reshape(const_shape![1]);
        let weight_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            weight_grad_base.broadcast(const_shape![128]);
        let input_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(input_ptr);
        let input_base: PointerTile<*mut f32, { [1] }> = input_root.reshape(const_shape![1]);
        let input_ptrs: PointerTile<*mut f32, { [128] }> = input_base.broadcast(const_shape![128]);
        let output_grad_root: PointerTile<*mut f32, { [] }> = pointer_to_tile(output_grad_ptr);
        let output_grad_base: PointerTile<*mut f32, { [1] }> =
            output_grad_root.reshape(const_shape![1]);
        let output_grad_ptrs: PointerTile<*mut f32, { [128] }> =
            output_grad_base.broadcast(const_shape![128]);

        let weight_len_tile: Tile<i32, { [128] }> = broadcast_scalar(weight_len, const_shape![128]);
        let input_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_height * input_width, const_shape![128]);
        let input_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(input_width, const_shape![128]);
        let output_plane_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_height * output_width, const_shape![128]);
        let output_width_tile: Tile<i32, { [128] }> =
            broadcast_scalar(output_width, const_shape![128]);
        let channels_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(channels_per_group, const_shape![128]);
        let filters_per_group_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters_per_group, const_shape![128]);
        let kernel_size_tile: Tile<i32, { [128] }> =
            broadcast_scalar(kernel_size, const_shape![128]);
        let input_batch_stride_tile: Tile<i32, { [128] }> = broadcast_scalar(
            input_channels * input_height * input_width,
            const_shape![128],
        );
        let output_batch_stride_tile: Tile<i32, { [128] }> =
            broadcast_scalar(filters * output_height * output_width, const_shape![128]);
        let zero_i32: Tile<i32, { [128] }> = broadcast_scalar(0i32, const_shape![128]);

        let start: i32 = pid.0 * 128i32;
        let step: i32 = grid.0 * 128i32;
        for offset in (start..weight_len).step_by(step as usize) {
            let weight_idx: Tile<i32, { [128] }> =
                iota(const_shape![128]) + broadcast_scalar(offset, const_shape![128]);
            let weight_mask: Tile<bool, { [128] }> = lt_tile(weight_idx, weight_len_tile);

            let filter_idx =
                weight_idx / (channels_per_group_tile * kernel_size_tile * kernel_size_tile);
            let filter_rem =
                weight_idx % (channels_per_group_tile * kernel_size_tile * kernel_size_tile);
            let input_channel_in_group = filter_rem / (kernel_size_tile * kernel_size_tile);
            let kernel_rem = filter_rem % (kernel_size_tile * kernel_size_tile);
            let ky = kernel_rem / kernel_size_tile;
            let kx = kernel_rem % kernel_size_tile;
            let group_idx = filter_idx / filters_per_group_tile;
            let input_channel = group_idx * channels_per_group_tile + input_channel_in_group;

            let mut acc: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
            for batch_idx_scalar in 0i32..batch {
                let batch_idx: Tile<i32, { [128] }> =
                    broadcast_scalar(batch_idx_scalar, const_shape![128]);
                for output_y_scalar in 0i32..output_height {
                    let output_y: Tile<i32, { [128] }> =
                        broadcast_scalar(output_y_scalar, const_shape![128]);
                    let input_y = output_y * broadcast_scalar(stride_y, const_shape![128])
                        + ky * broadcast_scalar(dilation, const_shape![128])
                        - broadcast_scalar(padding, const_shape![128]);
                    let y_ge_zero: Tile<bool, { [128] }> = ge_tile(input_y, zero_i32);
                    let y_lt_height: Tile<bool, { [128] }> =
                        lt_tile(input_y, broadcast_scalar(input_height, const_shape![128]));
                    let y_valid: Tile<bool, { [128] }> = y_ge_zero & y_lt_height;
                    for output_x_scalar in 0i32..output_width {
                        let output_x: Tile<i32, { [128] }> =
                            broadcast_scalar(output_x_scalar, const_shape![128]);
                        let input_x = output_x * broadcast_scalar(stride_x, const_shape![128])
                            + kx * broadcast_scalar(dilation, const_shape![128])
                            - broadcast_scalar(padding, const_shape![128]);
                        let input_x_ge_zero: Tile<bool, { [128] }> = ge_tile(input_x, zero_i32);
                        let input_x_lt_width: Tile<bool, { [128] }> =
                            lt_tile(input_x, broadcast_scalar(input_width, const_shape![128]));
                        let valid: Tile<bool, { [128] }> =
                            weight_mask & y_valid & input_x_ge_zero & input_x_lt_width;

                        let input_offsets = batch_idx * input_batch_stride_tile
                            + input_channel * input_plane_tile
                            + input_y * input_width_tile
                            + input_x;
                        let safe_input_offsets = select(valid, input_offsets, zero_i32);
                        let input_src: PointerTile<*mut f32, { [128] }> =
                            input_ptrs.offset_tile(safe_input_offsets);
                        let input_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            input_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(valid),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );

                        let grad_offsets = batch_idx * output_batch_stride_tile
                            + filter_idx * output_plane_tile
                            + output_y * output_width_tile
                            + output_x;
                        let output_grad_src: PointerTile<*mut f32, { [128] }> =
                            output_grad_ptrs.offset_tile(grad_offsets);
                        let grad_tile: (Tile<f32, { [128] }>, Token) = load_ptr_tko(
                            output_grad_src,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(weight_mask),
                            Some(0.0f32),
                            None,
                            Latency::<0>,
                        );
                        acc = acc + input_tile.0 * grad_tile.0;
                    }
                }
            }

            let dst: PointerTile<*mut f32, { [128] }> = weight_grad_ptrs.offset_tile(weight_idx);
            let _store_token: Token = store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(weight_mask),
                None,
                Latency::<0>,
            );
        }
    }
}
