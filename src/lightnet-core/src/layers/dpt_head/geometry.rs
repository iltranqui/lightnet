use crate::error::{Error, Result};

pub(crate) const DPT_WEIGHT_SLOTS: usize = 66;
const RESIZE_KERNELS: [usize; 4] = [4, 2, 0, 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DptHeadGeometry {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub embed_dim: usize,
    pub fusion: usize,
    pub head_hidden: usize,
    pub patch_size: usize,
    pub neck: [usize; 4],
    /// Effective C++ factors after its float-to-integer coercion.
    pub reassemble: [usize; 4],
}

impl DptHeadGeometry {
    pub(crate) fn output_height(self) -> Result<usize> {
        self.height
            .checked_mul(self.patch_size)
            .ok_or_else(|| invalid("output height overflows"))
    }

    pub(crate) fn output_width(self) -> Result<usize> {
        self.width
            .checked_mul(self.patch_size)
            .ok_or_else(|| invalid("output width overflows"))
    }

    pub(crate) fn validate(self) -> Result<()> {
        if [
            self.batch,
            self.height,
            self.width,
            self.embed_dim,
            self.fusion,
            self.head_hidden,
            self.patch_size,
        ]
        .contains(&0)
        {
            return Err(invalid(
                "batch, input geometry, widths, and patch_size must be positive",
            ));
        }
        if self.fusion % 2 != 0 {
            return Err(invalid(
                "fusion must be even because the head uses fusion/2 channels",
            ));
        }
        if self.neck.contains(&0) {
            return Err(invalid("all four neck widths must be positive"));
        }
        if self.reassemble != [4, 2, 1, 0] {
            return Err(invalid(format!(
                "only effective reassemble=[4,2,1,0] is supported by the fixed source weight slots, got {:?}",
                self.reassemble
            )));
        }
        let output_height = self.output_height()?;
        let output_width = self.output_width()?;
        checked_product(&[self.batch, output_height, output_width], "output tensor")?;
        checked_product(
            &[self.batch, 4, self.embed_dim, self.height, self.width],
            "input tensor",
        )?;
        for tap in 0..4 {
            let height = reassembled_size(self.height, self.reassemble[tap])?;
            let width = reassembled_size(self.width, self.reassemble[tap])?;
            checked_product(
                &[self.batch, self.neck[tap], self.height, self.width],
                "projected feature map",
            )?;
            checked_product(
                &[self.batch, self.fusion, height, width],
                "reassembled feature map",
            )?;
        }
        let final_height = reassembled_size(self.height, 4)?
            .checked_mul(2)
            .ok_or_else(|| invalid("final fusion height overflows"))?;
        let final_width = reassembled_size(self.width, 4)?
            .checked_mul(2)
            .ok_or_else(|| invalid("final fusion width overflows"))?;
        checked_product(
            &[self.batch, self.fusion, final_height, final_width],
            "final fusion feature map",
        )?;
        checked_product(
            &[self.batch, self.fusion / 2, final_height, final_width],
            "head conv1 feature map",
        )?;
        checked_product(
            &[self.batch, self.head_hidden, output_height, output_width],
            "head conv2 feature map",
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DptWeightRange {
    pub start: usize,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DptHeadWeightLayout {
    pub slots: [DptWeightRange; DPT_WEIGHT_SLOTS],
    pub total: usize,
}

impl DptHeadWeightLayout {
    pub(crate) fn get(&self, slot: usize) -> DptWeightRange {
        self.slots[slot]
    }
}

pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidConfig(format!("[dpt_head]: {}", message.into()))
}

pub(super) fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{what} element count overflows")))
    })
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid(format!("{what} size overflows")))
}

pub(crate) fn dpt_head_weight_layout(geometry: DptHeadGeometry) -> Result<DptHeadWeightLayout> {
    geometry.validate()?;
    let mut slots = [DptWeightRange::default(); DPT_WEIGHT_SLOTS];
    let mut cursor = 0usize;
    let mut set = |slot: usize, len: usize| -> Result<()> {
        slots[slot] = DptWeightRange { start: cursor, len };
        cursor = checked_add(cursor, len, "weight arena")?;
        Ok(())
    };

    for tap in 0..4 {
        set(
            2 * tap,
            checked_product(&[geometry.neck[tap], geometry.embed_dim], "projection")?,
        )?;
        set(2 * tap + 1, geometry.neck[tap])?;
    }
    for tap in 0..4 {
        let kernel = RESIZE_KERNELS[tap];
        let weight_count = checked_product(
            &[geometry.neck[tap], geometry.neck[tap], kernel, kernel],
            "resize weights",
        )?;
        let bias_count = if kernel == 0 { 0 } else { geometry.neck[tap] };
        set(8 + 2 * tap, weight_count)?;
        set(9 + 2 * tap, bias_count)?;
    }
    for tap in 0..4 {
        set(
            16 + tap,
            checked_product(&[geometry.fusion, geometry.neck[tap], 9], "neck weights")?,
        )?;
    }
    for base in [20usize, 28, 36, 44, 52] {
        for layer in 0..4 {
            let weight_count = if base == 20 {
                checked_product(&[geometry.fusion, geometry.fusion], "fusion projection")?
            } else {
                checked_product(
                    &[geometry.fusion, geometry.fusion, 9],
                    "residual convolution",
                )?
            };
            set(base + 2 * layer, weight_count)?;
            set(base + 2 * layer + 1, geometry.fusion)?;
        }
    }
    set(
        60,
        checked_product(&[geometry.fusion / 2, geometry.fusion, 9], "head conv1")?,
    )?;
    set(61, geometry.fusion / 2)?;
    set(
        62,
        checked_product(
            &[geometry.head_hidden, geometry.fusion / 2, 9],
            "head conv2",
        )?,
    )?;
    set(63, geometry.head_hidden)?;
    set(64, geometry.head_hidden)?;
    set(65, 1)?;
    debug_assert_eq!(slots[65].start + slots[65].len, cursor);
    Ok(DptHeadWeightLayout {
        slots,
        total: cursor,
    })
}

pub(crate) fn dpt_weight_slot_name(slot: usize) -> String {
    format!("dpt_tensor_{slot:02}")
}

pub(super) fn reassembled_size(size: usize, factor: usize) -> Result<usize> {
    match factor {
        0 => Ok(size.div_ceil(2)),
        1 => Ok(size),
        scale => size
            .checked_mul(scale)
            .ok_or_else(|| invalid("reassembled spatial size overflows")),
    }
}
