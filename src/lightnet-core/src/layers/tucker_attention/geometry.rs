//! Geometry and parameter sizing for Tucker attention.

use crate::model::Activation;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TuckerGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub size: usize,
    pub heads: usize,
    pub rank_q: usize,
    pub rank_k: usize,
    pub rank_v: usize,
    pub rank_o: usize,
    pub activation: Activation,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
}

pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .ok_or_else(|| format!("Tucker {what} size overflow"))
}

pub(crate) fn tucker_weight_count(
    channels: usize,
    heads: usize,
    rank_q: usize,
    rank_k: usize,
    rank_v: usize,
    rank_o: usize,
) -> Result<usize, String> {
    if channels == 0
        || heads == 0
        || channels % heads != 0
        || rank_q == 0
        || rank_k == 0
        || rank_v == 0
        || rank_o == 0
    {
        return Err(
            "Tucker weight geometry requires positive ranks and channels divisible by heads".into(),
        );
    }
    let dim = channels / heads;
    let sizes = [
        checked_product(&[channels, rank_q], "Q basis")?,
        checked_product(&[channels, rank_k], "K basis")?,
        checked_product(&[channels, rank_v], "V basis")?,
        checked_product(&[heads, rank_q, dim], "Q core")?,
        checked_product(&[heads, rank_k, dim], "K core")?,
        checked_product(&[heads, rank_v, dim], "V core")?,
        checked_product(&[heads, dim, rank_o], "O core")?,
        checked_product(&[rank_o, channels], "O basis")?,
    ];
    sizes
        .into_iter()
        .try_fold(0usize, |sum, size| sum.checked_add(size))
        .ok_or_else(|| String::from("Tucker total weight count overflow"))
}

pub(crate) fn validate_geometry(g: &TuckerGeometry) -> Result<(), String> {
    if g.batch == 0
        || g.channels == 0
        || g.height == 0
        || g.width == 0
        || g.size == 0
        || g.heads == 0
        || g.channels % g.heads != 0
        || g.rank_q == 0
        || g.rank_k == 0
        || g.rank_v == 0
        || g.rank_o == 0
    {
        return Err(
            "Tucker geometry requires positive dimensions/ranks and channels divisible by heads"
                .into(),
        );
    }
    let tokens = checked_product(&[g.size, g.size], "window token count")?;
    let padded_h = g
        .height
        .div_ceil(g.size)
        .checked_mul(g.size)
        .ok_or("Tucker padded height overflow")?;
    let padded_w = g
        .width
        .div_ceil(g.size)
        .checked_mul(g.size)
        .ok_or("Tucker padded width overflow")?;
    let windows = checked_product(
        &[g.batch, padded_h / g.size, padded_w / g.size],
        "window count",
    )?;
    checked_product(&[windows, tokens, g.channels], "window input")?;
    checked_product(&[windows, g.heads, tokens, tokens], "attention scores")?;
    checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    for rank in [g.rank_q, g.rank_k, g.rank_v, g.rank_o] {
        checked_product(&[tokens, rank], "window rank workspace")?;
    }
    checked_product(&[tokens, g.channels], "window channel workspace")?;
    Ok(())
}
