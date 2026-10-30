//! Pixel-mask loading for semantic segmentation, following Darknet raptor-fp4.
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Find the first existing mask using Darknet's candidate order.
pub fn segmentation_mask_path(image: &Path) -> PathBuf {
    let parent = image.parent().unwrap_or_else(|| Path::new(""));
    let stem = image.file_stem().unwrap_or_default().to_string_lossy();
    let mut candidates = vec![
        parent.join(format!("{stem}_mask.png")),
        parent.join(format!("{stem}_seg.png")),
        parent.join("mask.png"),
        parent.join("masks").join(format!("{stem}_mask.png")),
        parent.join("masks").join(format!("{stem}.png")),
        parent.join("masks").join("mask.png"),
    ];
    let normalized = image.to_string_lossy().replace('\\', "/");
    if let Some((prefix, suffix)) = normalized.split_once("/images/") {
        let replaced = PathBuf::from(format!("{prefix}/masks/{suffix}"));
        let dir = replaced.parent().unwrap_or_else(|| Path::new(""));
        candidates.push(dir.join(format!("{stem}_mask.png")));
        candidates.push(dir.join(format!("{stem}.png")));
    }
    candidates.push(parent.join(format!("{stem}.png")));
    candidates.iter().find(|candidate| candidate.as_path() != image && candidate.exists())
        .cloned().unwrap_or_else(|| candidates[0].clone())
}

/// Load an integer class-ID mask and return binary or channel-major one-hot truth.
pub fn load_segmentation_truth(path: &Path, width: usize, height: usize, classes: usize) -> Result<Vec<f32>> {
    if width == 0 || height == 0 || classes == 0 {
        return Err(Error::InvalidConfig("segmentation dimensions and classes must be nonzero".into()));
    }
    let image = image::open(path).map_err(|e| Error::Parse(format!("cannot load mask {}: {e}", path.display())))?;
    let source_width = image.width() as usize;
    let source_height = image.height() as usize;
    let values: Vec<u16> = match image {
        image::DynamicImage::ImageLuma16(pixels) => pixels.into_raw(),
        image::DynamicImage::ImageLumaA16(pixels) => pixels.pixels().map(|p| p.0[0]).collect(),
        image::DynamicImage::ImageRgb16(pixels) => image::DynamicImage::ImageRgb16(pixels).to_luma16().into_raw(),
        image::DynamicImage::ImageRgba16(pixels) => image::DynamicImage::ImageRgba16(pixels).to_luma16().into_raw(),
        other => other.to_luma8().into_raw().into_iter().map(u16::from).collect(),
    };
    let spatial = width.checked_mul(height).ok_or_else(|| Error::InvalidConfig("segmentation size overflow".into()))?;
    let length = spatial.checked_mul(classes).ok_or_else(|| Error::InvalidConfig("segmentation truth size overflow".into()))?;
    let mut truth = vec![0.0; length];
    for y in 0..height {
        for x in 0..width {
            let class = values[(y * source_height / height) * source_width + x * source_width / width] as usize;
            let pixel = y * width + x;
            if classes == 1 {
                truth[pixel] = if class != 0 { 1.0 } else { 0.0 };
            } else {
                truth[class.min(classes - 1) * spatial + pixel] = 1.0;
            }
        }
    }
    Ok(truth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    #[test]
    fn pixel_masks_resize_and_encode_binary_and_one_hot() {
        let dir = std::env::temp_dir().join(format!("lightnet-seg-mask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample_mask.png");
        let image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(2, 1, vec![0, 2]).unwrap();
        image.save(&path).unwrap();
        assert_eq!(segmentation_mask_path(&dir.join("sample.jpg")), path);
        assert_eq!(load_segmentation_truth(&path, 4, 1, 1).unwrap(), [0., 0., 1., 1.]);
        assert_eq!(load_segmentation_truth(&path, 4, 1, 3).unwrap(), [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
}
