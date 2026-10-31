fn padded_extent(extent: usize, window: usize) -> Result<usize, String> {
    if extent == 0 || window == 0 {
        return Err("Swin padding requires positive extents and window size".into());
    }
    let rounded = extent
        .checked_add(window - 1)
        .ok_or("Swin padded extent overflow")?
        / window;
    rounded
        .checked_mul(window)
        .ok_or_else(|| "Swin padded extent overflow".into())
}

#[doc(hidden)]
pub fn padded_shape(
    height: usize,
    width: usize,
    window: usize,
) -> Result<(usize, usize), String> {
    Ok((
        padded_extent(height, window)?,
        padded_extent(width, window)?,
    ))
}
