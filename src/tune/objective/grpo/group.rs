//! One prompt's group: the completions drawn for it, the advantage each one
//! carries against the group's own mean, and the loss the group produces.

use anyhow::{bail, Result};
use candle_core::Tensor;

use crate::{
    model::Route,
    runtime::{Completion, GenerationOptions, Runtime},

};

use super::super::super::preflight::token_logprobs;
use super::{GrpoIteration, GrpoOptions, Reward};

/// One sampled completion with everything the step needs about it.
pub(super) struct Draw {
    pub(super) completion: Completion,
    pub(super) advantage: f64,
    /// The frozen reference's per-token log-probabilities of this completion.
    /// Constant: the reference cannot move, and it is scored here rather than
    /// inside the loss so the tensor carries no autograd tape.
    pub(super) reference: Tensor,
}

/// A whole group for one prompt.
pub(super) struct Group {
    pub(super) draws: Vec<Draw>,
    pub(super) mean_reward: f64,
    pub(super) spread: f64,
}

/// Draws `--group` completions, scores them, and normalizes within the group.
pub(super) fn sample_group(
    runtime: &Runtime,
    prompt: &str,
    reward: &Reward,
    options: &GrpoOptions,
    draw: &mut u64,
) -> Result<Group> {
    let mut completions = Vec::with_capacity(options.group);
    let mut rewards = Vec::with_capacity(options.group);
    for _ in 0..options.group {
        let generation = GenerationOptions {
            seed: options.generation.seed.wrapping_add(*draw),
            ..options.generation
        };
        *draw = draw.wrapping_add(1);
        let completion = runtime.sample(prompt, None, generation)?;
        if completion.tokens.is_empty() {
            // The first sampled token was the end of sequence. There is nothing
            // to take a gradient through, and dropping the completion would
            // bias the baseline upward by removing the group's worst member, so
            // the whole group is refused instead.
            bail!("a sampled completion was empty, so this group has nothing to score");
        }
        rewards.push(reward.score(&completion)?);
        completions.push(completion);
    }

    let count = rewards.len() as f64;
    let mean = rewards.iter().sum::<f64>() / count;
    let variance = rewards.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / count;
    let spread = variance.sqrt();
    // No epsilon. A group whose completions all scored the same has no
    // preference to express, and its advantages are exactly zero; adding a
    // floor to the denominator would turn that silence into amplified rounding.
    let normalize = |value: f64| if spread > 0.0 { (value - mean) / spread } else { 0.0 };

    let mut draws = Vec::with_capacity(completions.len());
    for (completion, value) in completions.into_iter().zip(&rewards) {
        let ids = sequence(&completion);
        let logits = runtime.forward_scored(&ids, Route::Base)?;
        let reference = token_logprobs(&logits, &ids, completion.prompt.len(), runtime.device())?;
        draws.push(Draw {
            advantage: normalize(*value),
            reference,
            completion,
        });
    }
    Ok(Group { draws, mean_reward: mean, spread })
}

/// The loss for one group, plus the scalars the report is built from.
pub(super) struct Loss {
    pub(super) tensor: Tensor,
    pub(super) value: f64,
    pub(super) kl: f64,
}

pub(super) fn group_loss(runtime: &Runtime, group: &Group, options: &GrpoOptions) -> Result<Loss> {
    let mut summed: Option<Tensor> = None;
    let mut kl_total = 0f64;
    for draw in &group.draws {
        let ids = sequence(&draw.completion);
        let logits = runtime.forward_train(&ids)?;
        let policy = token_logprobs(&logits, &ids, draw.completion.prompt.len(), runtime.device())?;

        // pi_old is pi_theta at this exact step, so the ratio is one in value
        // and its gradient is the policy gradient. Detaching is what states
        // that, and it costs no second forward pass.
        let ratio = (&policy - policy.detach())?.exp()?;
        let advantage = (ratio * draw.advantage)?;

        // k3: exp(d) - d - 1 with d = log pi_ref - log pi_theta. Non-negative
        // for every sample and unbiased for the divergence, where the naive -d
        // is neither.
        let divergence = (&draw.reference - &policy)?;
        let penalty = ((divergence.exp()? - &divergence)? - 1.0)?;
        kl_total += penalty.mean_all()?.to_scalar::<f32>()? as f64;

        // Averaged over the completion's own tokens before the group mean, so
        // a long completion does not outvote a short one on length alone.
        let objective = (advantage - (penalty * options.beta)?)?.mean_all()?;
        let scaled = (objective.neg()? / group.draws.len() as f64)?;
        summed = Some(match summed {
            Some(total) => (total + scaled)?,
            None => scaled,
        });
    }
    let Some(tensor) = summed else {
        bail!("a sampled group contained no completions");
    };
    Ok(Loss {
        value: tensor.to_scalar::<f32>()? as f64,
        kl: kl_total / group.draws.len() as f64,
        tensor,
    })
}

/// Prompt then completion, the exact sequence the sampler produced.
fn sequence(completion: &Completion) -> Vec<u32> {
    let mut ids = Vec::with_capacity(completion.prompt.len() + completion.tokens.len());
    ids.extend_from_slice(&completion.prompt);
    ids.extend_from_slice(&completion.tokens);
    ids
}

/// Running totals over one iteration.
#[derive(Debug, Default)]
pub(super) struct Totals {
    pub(super) groups: usize,
    pub(super) completions: usize,
    pub(super) reward: f64,
    pub(super) spread: f64,
    pub(super) kl: f64,
    pub(super) loss: f64,
    pub(super) tokens: usize,
}

impl Totals {
    pub(super) fn record(&mut self, group: &Group, loss: &Loss) {
        self.groups += 1;
        self.completions += group.draws.len();
        self.reward += group.mean_reward;
        self.spread += group.spread;
        self.kl += loss.kl;
        self.loss += loss.value;
        self.tokens += group.draws.iter().map(|draw| draw.completion.tokens.len()).sum::<usize>();
    }

    /// Every mean divides by the group count, guarded at one so an iteration
    /// that recorded nothing reports zero rather than a JSON `NaN` no client
    /// can parse.
    pub(super) fn mean(&self, total: f64) -> f32 {
        (total / self.groups.max(1) as f64) as f32
    }

    pub(super) fn mean_reward(&self) -> f32 {
        self.mean(self.reward)
    }

    pub(super) fn mean_kl(&self) -> f32 {
        self.mean(self.kl)
    }

    pub(super) fn finish(&self, iteration: usize) -> GrpoIteration {
        GrpoIteration {
            iteration,
            groups: self.groups,
            completions: self.completions,
            mean_reward: self.mean_reward(),
            reward_spread: self.mean(self.spread),
            mean_kl: self.mean_kl(),
            policy_loss: self.mean(self.loss),

            mean_completion_tokens: self.tokens as f32 / self.completions.max(1) as f32,
        }
    }
}
