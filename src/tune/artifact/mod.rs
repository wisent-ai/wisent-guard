//! What can be done with an adapter once it exists: measure it on held-out
//! text, or fold it into the base weights it was trained beside.

mod evaluate;
mod merge;

pub use evaluate::{
    evaluate, warn_on_provenance, EvaluateOptions, EvaluateReport, EvaluatedExample,
};
pub use merge::{merge, MergeReport};
