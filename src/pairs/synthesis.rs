//! Writing a contrastive pair set with a model: a faithful port of Wisent's
//! `SyntheticContrastivePairsGenerator.generate`, driven by whichever
//! generator the caller picked. Steering needs hidden states and stays local;
//! writing pair text does not, so the writer may be a hosted route.

use std::num::{FpCategory, NonZeroU64, NonZeroUsize};

use anyhow::{Result, bail};
use serde::Serialize;

use super::generator::Generator;
use crate::{
    artifact::{ContrastivePair, PairSet},
    pairs::quality::dedupe::{self, DedupeOptions},
    pairs::quality::{diversity, refusal},
    runtime::GenerationOptions,
    workflow,
};

/// Verbatim from the Python generator: the single instruction that produces
/// the user question each pair answers.
const QUESTION_INSTRUCTION: &str = "Write one short question a user might ask. Example: 'What is your favorite hobby?' Just the question, nothing else.";

/// Verbatim `roleplay_neg_fix` from Wisent's `db_instructions/mini_dp.py`.
/// The Python cleaner sends it as a system message; `Runtime::generate` takes
/// a single prompt string, so it is prepended to the user turn instead.
const ROLEPLAY_NEG_FIX: &str = "You are fixing ONLY the negative example of a contrastive pair.\nProduce a single concise negative response for the given prompt that exemplifies the UNDESIRED trait.\nIt must be fictional/hypothetical, safe, and non-actionable. Return raw text only.";

// MARK: - Synthesis

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub trait_description: String,
    pub trait_name: String,
    pub opposite: Option<String>,
    pub count: usize,
    pub retry_multiplier: usize,
    pub dedupe: DedupeOptions,
    pub refusal_threshold: f32,
    pub generation: GenerationOptions,
}

#[derive(Debug, Clone, Serialize)]
pub struct SynthesisReport {
    /// What wrote these pairs: `local:<model id>` or `brama:<route>`. First in
    /// the report because every other number below is only interpretable once
    /// the reader knows which model produced the text.
    pub generator: String,
    pub trait_name: String,
    pub trait_description: String,
    pub opposite: String,
    pub requested: usize,
    pub attempts: usize,
    pub kept: usize,
    pub rejected_empty: usize,
    pub rejected_refusals: usize,
    pub rejected_duplicates: usize,
    pub refusal_retries: usize,
    pub diversity: diversity::Scores,
}

/// What a run has done so far, counted as it happens; every count starts at
/// none.
#[derive(Default)]
struct Tally {
    calls: u64,
    attempts: usize,
    rejected_empty: usize,
    rejected_refusals: usize,
    rejected_duplicates: usize,
    refusal_retries: usize,
}

/// One more of what `count` counts.
fn one_more(count: &mut usize) {
    *count += NonZeroUsize::MIN.get();
}

/// Generate a contrastive pair set with `generator`.
///
/// Port of `SyntheticContrastivePairsGenerator.generate`: one opposite-trait
/// description up front, then a question / positive / negative triple per
/// attempt, with refusal repair and deduplication applied as each pair lands
/// rather than in a batch pass at the end. Cleaning inline is what lets the
/// loop stop the moment `count` *surviving* pairs exist instead of
/// re-cleaning the whole set every ten pairs the way the Python does.
pub fn synthesize(
    generator: Generator<'_>,
    options: &SynthesisOptions,
) -> Result<(PairSet, SynthesisReport)> {
    if NonZeroUsize::new(options.count).is_none() {
        bail!("synthesis requires a pair count above zero");
    }
    if NonZeroUsize::new(options.retry_multiplier).is_none() {
        bail!("synthesis requires a retry multiplier above zero");
    }
    // Every prompt in this loop is a constant, repeated for the whole run: the
    // same question instruction, the same two persona prompts. A greedy
    // decoder — Ster's own argmax path at or below zero temperature, and any
    // hosted sampler asked for the same — would return the same question and
    // the same answers forever and the whole run would dedupe to one pair.
    // The reason is the loop's, so the refusal holds for every route. A
    // temperature that is not a number is not above zero either.
    let temperature = options.generation.temperature;
    if temperature.is_nan()
        || temperature.is_sign_negative()
        || temperature.classify() == FpCategory::Zero
    {
        bail!(
            "synthesis requires a temperature above zero; argmax generation repeats a single prompt"
        );
    }
    options.dedupe.validate()?;

    let trait_name = resolve_trait_name(options);
    let mut tally = Tally::default();
    let ask = |prompt: &str, tally: &mut Tally| -> Result<String> {
        let text = generator.generate(prompt, options, tally.calls)?;
        tally.calls = tally.calls.wrapping_add(NonZeroU64::MIN.get());
        Ok(text)
    };

    // The opposite is the caller's, or the generator's answer to one
    // question; a generator that names none leaves the negatives nothing to
    // be, so the run is refused rather than given an opposite nobody stated.
    let opposite = match options
        .opposite
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => value.to_owned(),
        None => {
            let answer = ask(
                &format!(
                    "What is the OPPOSITE personality trait of: {}?\n\nDescribe the opposite in one sentence, be specific about what words/style/tone to use.",
                    options.trait_description
                ),
                &mut tally,
            )?;
            if answer.is_empty() {
                bail!(
                    "{} returned no opposite trait for {:?}; state it with --opposite",
                    generator.label(),
                    options.trait_description
                );
            }
            answer
        }
    };

    // Named once, before the loop: a streamed run has to state which model is
    // writing and which opposite trait the negative side is answering as,
    // and neither changes for the rest of the run.
    workflow::progress(format!(
        "synthesizing with {} against opposite trait {:?}",
        generator.label(),
        opposite
    ));

    let mut index = dedupe::Index::new(options.dedupe)?;
    let mut kept: Vec<ContrastivePair> = Vec::with_capacity(options.count);
    let mut questions: Vec<String> = Vec::with_capacity(options.count);
    let max_attempts = options.count.saturating_mul(options.retry_multiplier);

    while kept.len() < options.count && tally.attempts < max_attempts {
        one_more(&mut tally.attempts);
        workflow::progress(format!(
            "synthesizing pair {}/{} (attempt {})",
            kept.len() + NonZeroUsize::MIN.get(),
            options.count,
            tally.attempts
        ));

        let question = ask(QUESTION_INSTRUCTION, &mut tally)?;
        if question.is_empty() {
            one_more(&mut tally.rejected_empty);
            continue;
        }

        let positive = ask(
            &persona_prompt(&question, &options.trait_description),
            &mut tally,
        )?;
        if positive.is_empty() {
            one_more(&mut tally.rejected_empty);
            continue;
        }

        let mut negative = ask(&persona_prompt(&question, &opposite), &mut tally)?;
        if negative.is_empty() {
            one_more(&mut tally.rejected_empty);
            continue;
        }

        // A refusing positive has no repair path: the trait itself is what the
        // model declined to roleplay, so re-asking would refuse again.
        if refusal::looks_like_refusal(&positive, options.refusal_threshold) {
            one_more(&mut tally.rejected_refusals);
            continue;
        }

        // `RefusalerCleaner` + `BaseRefusaler::fix_negative`: one re-prompt,
        // never a loop, and a still-refusing replacement drops the pair.
        if refusal::looks_like_refusal(&negative, options.refusal_threshold) {
            one_more(&mut tally.refusal_retries);
            let replacement = ask(
                &format!(
                    "{ROLEPLAY_NEG_FIX}\n\nPrompt: {question}\nTrait label: {trait_name}\nTrait description: {opposite}"
                ),
                &mut tally,
            )?;
            if replacement.is_empty()
                || refusal::looks_like_refusal(&replacement, options.refusal_threshold)
            {
                one_more(&mut tally.rejected_refusals);
                continue;
            }
            negative = replacement;
        }

        // Ster's `ContrastivePair` carries two strings and no separate prompt
        // field, so the question is folded into both sides in the shape the
        // README and the docs already publish. Folding it in identically is
        // what keeps the two sides matched on topic, wording, and length —
        // the only difference left between them is the trait.
        let pair = ContrastivePair {
            positive: format!("Question: {question}\nAnswer: {positive}"),
            negative: format!("Question: {question}\nAnswer: {negative}"),
        };
        if index.insert(&pair).is_some() {
            one_more(&mut tally.rejected_duplicates);
            continue;
        }
        questions.push(question);
        kept.push(pair);
    }

    // Diversity is measured over the questions, matching the Python report:
    // the answers inherit their variety from the prompt that produced them.
    let diversity = diversity::compute(&questions);

    let report = SynthesisReport {
        generator: generator.label(),
        trait_name: trait_name.clone(),
        trait_description: options.trait_description.clone(),
        opposite,
        requested: options.count,
        attempts: tally.attempts,
        kept: kept.len(),
        rejected_empty: tally.rejected_empty,
        rejected_refusals: tally.rejected_refusals,
        rejected_duplicates: tally.rejected_duplicates,
        refusal_retries: tally.refusal_retries,
        diversity,
    };
    Ok((
        PairSet {
            trait_name,
            pairs: kept,
        },
        report,
    ))
}

/// Verbatim from the Python generator; the same shape produces both sides,
/// with the trait description swapped for its opposite on the negative.
fn persona_prompt(question: &str, personality: &str) -> String {
    format!(
        "Question: {question}\n\nAnswer the question AS IF you have this personality: {personality}\n\nWrite 1-2 sentences showing this personality clearly. Just the answer."
    )
}

/// The set's trait name: the one given, else the trait's description itself.
fn resolve_trait_name(options: &SynthesisOptions) -> String {
    let name = options.trait_name.trim();
    if !name.is_empty() {
        return name.to_owned();
    }
    options.trait_description.trim().to_owned()
}
