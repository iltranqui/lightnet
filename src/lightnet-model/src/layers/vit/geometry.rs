use crate::model::Activation;

pub const MHC_PARAM_COUNT: usize = 12;

/// Parameter lengths ordered as `save_vit_weights` (last item is mHC scales).
#[doc(hidden)]
pub fn vit_weight_sizes(g: &VitGeometry, out_h: usize, out_w: usize) -> Result<[usize; 16], String> {
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or("vit parameter count overflow");
    let patch_dim = mul(mul(g.patch_size, g.patch_size)?, g.channels)?;
    let t = mul(out_h, out_w)?;
    Ok([
        mul(g.hidden, patch_dim)?,
        g.hidden,
        mul(3, g.hidden)?,
        g.hidden,
        g.hidden,
        mul(3, mul(g.hidden, g.hidden)?)?,
        mul(g.hidden, g.hidden)?,
        g.hidden,
        g.hidden,
        g.hidden,
        mul(g.mlp_dim, g.hidden)?,
        g.mlp_dim,
        mul(g.hidden, g.mlp_dim)?,
        g.hidden,
        mul(t, g.hidden)?,
        MHC_PARAM_COUNT,
    ])
}

pub fn vit_blob_float_count(
    g: &VitGeometry,
    out_h: usize,
    out_w: usize,
    has_mhc_scales: bool,
) -> Result<usize, String> {
    let sizes = vit_weight_sizes(g, out_h, out_w)?;
    let base = sizes[..15].iter().try_fold(0usize, |n, &v| {
        n.checked_add(v).ok_or("vit blob size overflow")
    })?;
    if has_mhc_scales {
        base.checked_add(MHC_PARAM_COUNT)
            .ok_or_else(|| "vit blob size overflow".into())
    } else {
        Ok(base)
    }
}

/// Geometry and execution choices for one global ViT block.
#[derive(Clone, Debug, PartialEq)]
pub struct VitGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub patch_size: usize,
    pub patch_stride: usize,
    pub patch_pad: usize,
    pub hidden: usize,
    pub heads: usize,
    pub mlp_dim: usize,
    pub ln_eps: f32,
    pub activation: Activation,
    pub learned_pos_embed: bool,
    pub pos_init_zero: bool,
    pub msa: Option<(usize, usize)>,
    pub flash_attention: bool,
    pub vc_attention: bool,
    pub vc_expcast: u8,
    pub vc_perm: crate::model::VitVcPermutation,
}
