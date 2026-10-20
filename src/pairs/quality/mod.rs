//! What makes a pair set worth training on.
//!
//! Three independent readings: whether a candidate repeats one already kept
//! (`dedupe`) and how much lexical and structural variety the whole set
//! carries (`diversity`), both read from the text alone and run again
//! without a model when an existing set is inspected; and whether the writing
//! model answered or declined its role (`refusal`), which the generator
//! judges inside the synthesis loop, once per candidate side.

pub mod dedupe;
pub mod diversity;
pub mod refusal;
