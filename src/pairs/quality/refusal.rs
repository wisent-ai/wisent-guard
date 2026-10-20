//! Whether a synthesised answer declines the role it was asked to play.
//!
//! Synthesis asks a model to answer as a personality; a safety-tuned model
//! often answers that request with a refusal instead, and a refusal is a
//! useless side of a pair because it differs along the refusal axis rather
//! than along the trait. Which answer is a refusal is judged by the same
//! generator that wrote it, asked one question per answer and held to one
//! exact answer form, not by a list of phrasings: a refusal worded any other
//! way, or in another language, is still a refusal.

use anyhow::{Context, Result};
use serde::Deserialize;

/// The judgement the generator is asked for: one JSON object, nothing else.
pub(crate) fn judgement_prompt(question: &str, personality: &str, answer: &str) -> String {
    format!(
        "A model was asked to answer a question as if it had a personality.\n\nQuestion: {question}\nPersonality: {personality}\nAnswer: {answer}\n\nDid the answer decline or refuse to answer as that personality, instead of answering in character? Reply with exactly one JSON object and nothing else: {{\"refuses\": true}} or {{\"refuses\": false}}."
    )
}

/// The one answer form the judgement accepts.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Verdict {
    refuses: bool,
}

/// The generator's verdict, read from `text`; any answer that is not exactly
/// `{"refuses": <bool>}` is refused, naming the generator and its text.
pub(crate) fn verdict(generator: &str, text: &str) -> Result<bool> {
    let verdict: Verdict = serde_json::from_str(text.trim()).with_context(|| {
        format!(
            "{generator} answered the refusal judgement with {text:?}, which is not exactly {{\"refuses\": true}} or {{\"refuses\": false}}"
        )
    })?;
    Ok(verdict.refuses)
}
