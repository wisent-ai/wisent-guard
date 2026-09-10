//! What one ranked pair costs, and the running totals a run reports from.

use anyhow::Result;
use candle_core::Tensor;

use super::super::super::preflight::softplus;
use super::RewardHead;

/// The loss for one pair, plus the scalars the report is built from.
pub(super) struct Step {
    pub(super) tensor: Tensor,
    pub(super) loss: f64,
    pub(super) chosen: f64,
    pub(super) rejected: f64,
}

/// One pair's contribution: two scored rows and one Bradley-Terry loss.
///
/// The residual streams arrive already read out of whatever forward produced
/// them, so this function is identical whether one pair went through the model
/// or eight did.
pub(super) fn step_loss(head: &RewardHead, chosen: &Tensor, rejected: &Tensor) -> Result<Step> {
    let chosen = head.score(chosen)?;
    let rejected = head.score(rejected)?;
    // -log sigmoid(chosen - rejected), through the softplus that survives a
    // head confident enough to overflow the direct form.
    let tensor = softplus(&(&rejected - &chosen)?)?;
    Ok(Step {
        loss: tensor.to_scalar::<f32>()? as f64,
        chosen: chosen.to_scalar::<f32>()? as f64,
        rejected: rejected.to_scalar::<f32>()? as f64,
        tensor,
    })
}

/// Running totals over one epoch.
#[derive(Debug, Default)]
pub(super) struct Summary {
    pub(super) pairs: usize,
    pub(super) loss: f64,
    pub(super) correct: usize,
    /// Pairs whose two sides scored bit-identically. The comparison is exact
    /// on purpose: the only tie this is here to name is the one a
    /// zero-initialised head produces, where both sides are the same
    /// arithmetic on the same weights and land on the same float. A tolerance
    /// would start folding in pairs the head merely finds close, which is a
    /// different statement.
    pub(super) tied: usize,
    pub(super) chosen: f64,
    pub(super) rejected: f64,
}

impl Summary {
    pub(super) fn record(&mut self, step: &Step) {
        self.pairs += 1;
        self.loss += step.loss;
        self.chosen += step.chosen;
        self.rejected += step.rejected;
        if step.chosen > step.rejected {
            self.correct += 1;
        } else if step.chosen == step.rejected {
            self.tied += 1;
        }
    }

    /// Every mean below divides by the pair count, guarded at one so an epoch
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    fn mean(&self, total: f64) -> f32 {
        (total / self.pairs.max(1) as f64) as f32
    }

    pub(super) fn mean_loss(&self) -> f32 {
        self.mean(self.loss)
    }

    pub(super) fn mean_chosen(&self) -> f32 {
        self.mean(self.chosen)
    }

    pub(super) fn mean_rejected(&self) -> f32 {
        self.mean(self.rejected)
    }

    pub(super) fn mean_margin(&self) -> f32 {
        self.mean(self.chosen - self.rejected)
    }

    pub(super) fn accuracy(&self) -> f32 {
        self.correct as f32 / self.pairs.max(1) as f32
    }
}
