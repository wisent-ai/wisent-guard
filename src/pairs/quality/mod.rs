//! What makes a pair set worth training on, judged from the text alone.
//!
//! Three independent readings, each a port of the Wisent original: whether a
//! candidate repeats one already kept (`dedupe`), how much lexical and
//! structural variety the whole set carries (`diversity`), and whether the
//! writing model answered or refused (`refusal`). They run inside the
//! synthesis loop, once per candidate, while a model is loaded, and again
//! without a model when an existing set is inspected.

pub mod dedupe;
pub mod diversity;
pub mod refusal;
