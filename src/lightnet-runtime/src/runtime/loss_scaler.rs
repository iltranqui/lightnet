/// Result of updating a dynamic loss scale after checking gradients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleStep {
    Apply,
    Skip,
}

/// Tracks the dynamic loss scale used during mixed-precision training.
#[derive(Debug, Clone, Copy)]
pub struct LossScaler {
    scale: f32,
    growth_steps: u32,
}

impl LossScaler {
    pub fn new(initial: f32) -> Self {
        Self {
            scale: initial.min(65536.0),
            growth_steps: 0,
        }
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn update(&mut self, grads_finite: bool) -> ScaleStep {
        if !grads_finite {
            self.scale = (self.scale * 0.5).max(1.0);
            self.growth_steps = 0;
            return ScaleStep::Skip;
        }

        if self.scale >= 65536.0 {
            self.growth_steps = 0;
            return ScaleStep::Apply;
        }

        self.growth_steps += 1;
        if self.growth_steps >= 1000 && self.scale < 65536.0 {
            self.scale = (self.scale * 2.0).min(65536.0);
            self.growth_steps = 0;
        }
        ScaleStep::Apply
    }
}

#[cfg(test)]
mod tests {
    use super::{LossScaler, ScaleStep};

    #[test]
    fn halves_and_skips_on_overflow() {
        let mut scaler = LossScaler::new(1024.0);

        assert_eq!(scaler.update(false), ScaleStep::Skip);
        assert_eq!(scaler.scale(), 512.0);
    }

    #[test]
    fn never_drops_below_one() {
        let mut scaler = LossScaler::new(1.0);

        assert_eq!(scaler.update(false), ScaleStep::Skip);
        assert_eq!(scaler.scale(), 1.0);
    }

    #[test]
    fn doubles_after_1000_finite_steps() {
        let mut scaler = LossScaler::new(1024.0);

        for _ in 0..999 {
            assert_eq!(scaler.update(true), ScaleStep::Apply);
        }
        assert_eq!(scaler.scale(), 1024.0);
        assert_eq!(scaler.update(true), ScaleStep::Apply);
        assert_eq!(scaler.scale(), 2048.0);
    }

    #[test]
    fn growth_counter_resets_after_overflow() {
        let mut scaler = LossScaler::new(1024.0);

        for _ in 0..500 {
            scaler.update(true);
        }
        assert_eq!(scaler.update(false), ScaleStep::Skip);
        assert_eq!(scaler.scale(), 512.0);
        for _ in 0..999 {
            assert_eq!(scaler.update(true), ScaleStep::Apply);
        }
        assert_eq!(scaler.scale(), 512.0);
    }

    #[test]
    fn growth_clamps_to_65536_from_below() {
        let mut scaler = LossScaler::new(40000.0);

        for _ in 0..1000 {
            assert_eq!(scaler.update(true), ScaleStep::Apply);
        }
        assert_eq!(scaler.scale(), 65536.0);
    }

    #[test]
    fn constructor_caps_initial_scale_at_65536() {
        assert_eq!(LossScaler::new(100_000.0).scale(), 65536.0);
    }

    #[test]
    fn finite_update_at_cap_resets_growth_counter_without_overflow() {
        let mut scaler = LossScaler::new(65536.0);
        scaler.growth_steps = u32::MAX;

        assert_eq!(scaler.update(true), ScaleStep::Apply);
        assert_eq!(scaler.scale(), 65536.0);
        assert_eq!(scaler.growth_steps, 0);
    }

    #[test]
    fn caps_at_65536() {
        let mut scaler = LossScaler::new(65536.0);

        for _ in 0..1000 {
            assert_eq!(scaler.update(true), ScaleStep::Apply);
        }
        assert_eq!(scaler.scale(), 65536.0);
    }
}
