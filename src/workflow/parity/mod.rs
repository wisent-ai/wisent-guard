//! Comparing this model's forward pass with hidden states another
//! implementation recorded for the same tokens: the measure of whether Ster
//! computes a family the way its reference implementation does.
//!
//! The input is `{"records": [{"tokenIds": [...] | "tokenIdsFile":
//! "ids.json", "hidden": "states.safetensors", "tensor": "hidden_states"}]}`,
//! paths relative to the input file. Each reference tensor holds the
//! residual stream after the final norm for the first `P` positions, `[P,
//! hidden]` or `[1, P, hidden]`. Ster reads the same token ids — no
//! tokenizer, no chat template — and reports how far its states lie from the
//! reference ones: the relative error over every compared position, the
//! worst position's, the largest single difference, and the cosine of each
//! position's pair. A cosine is only measured where both states are nonzero;
//! when no position has one, the mean and smallest cosine are reported as not
//! measured (null) rather than as an agreement of zero. A reference with no
//! positions is refused. A reference computed at a narrower precision than
//! Ster's sets the floor these can reach.

use std::{
    fs,
    num::FpCategory,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::runtime::Runtime;

use super::progress;

/// The tensor a reference file holds its states under when the record names
/// none, as Transformers' `output_hidden_states` captures name them.
const DEFAULT_TENSOR: &str = "hidden_states";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParitySet {
    records: Vec<ParityInput>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ParityInput {
    #[serde(default)]
    token_ids: Option<Vec<u32>>,
    #[serde(default)]
    token_ids_file: Option<String>,
    hidden: String,
    #[serde(default)]
    tensor: Option<String>,
}

/// How far Ster's states lie from one record's reference states, or from
/// every record's together. The cosines are `None` when no compared position
/// had a nonzero state on both sides, so nothing was measured.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityMeasure {
    pub positions: usize,
    pub relative_error: f64,
    pub worst_position_relative_error: f64,
    pub largest_difference: f64,
    pub mean_cosine: Option<f64>,
    pub smallest_cosine: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityRecord {
    pub hidden: String,
    pub tokens: usize,
    #[serde(flatten)]
    pub measure: ParityMeasure,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParityReport {
    pub model: String,
    pub model_revision: Option<String>,
    pub records: Vec<ParityRecord>,
    pub overall: ParityMeasure,
}

pub fn parity(runtime: &Runtime, input: &Path) -> Result<ParityReport> {
    let set: ParitySet = serde_json::from_slice(
        &fs::read(input).with_context(|| format!("failed to read {}", input.display()))?,
    )
    .with_context(|| format!("invalid parity JSON in {}", input.display()))?;
    if set.records.is_empty() {
        bail!("{} lists no records to compare", input.display());
    }
    let base = input
        .parent()
        .with_context(|| format!("{} is not a file inside a directory", input.display()))?
        .to_path_buf();
    let vocabulary = runtime.vocab_size();
    let mut records = Vec::with_capacity(set.records.len());
    let mut totals = Totals::default();
    for (index, record) in set.records.iter().enumerate() {
        progress(format!("comparing record {index} ({})", record.hidden));
        let ids = token_ids(record, &base, index)?;
        if let Some(id) = ids.iter().find(|id| **id as usize >= vocabulary) {
            bail!("record {index} holds token id {id}, past the model's {vocabulary} tokens");
        }
        // The record names its tensor, or the documented Transformers name.
        let tensor = match record.tensor.as_deref() {
            Some(named) => named,
            None => DEFAULT_TENSOR,
        };
        let reference = reference_states(&base.join(&record.hidden), tensor)?;
        let (positions, width) = reference.dims2()?;
        if positions > ids.len() {
            bail!(
                "record {index}'s reference holds {positions} positions for {} tokens",
                ids.len()
            );
        }
        let ours = runtime.forward_hidden_scored(&ids)?;
        let (_, tokens, ours_width) = ours.dims3()?;
        if ours_width != width {
            bail!(
                "record {index}'s reference states are {width} wide and this model's {ours_width}"
            );
        }
        let ours = ours
            .reshape((tokens, ours_width))?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let theirs = reference.to_vec2::<f32>()?;
        if theirs.is_empty() {
            bail!(
                "record {index}'s {tensor:?} in {} holds no positions to compare",
                record.hidden
            );
        }
        let mut record_totals = Totals::default();
        for (mine, reference) in ours.iter().zip(&theirs) {
            record_totals.add(mine, reference);
        }
        totals.merge(&record_totals);
        records.push(ParityRecord {
            hidden: record.hidden.clone(),
            tokens: ids.len(),
            measure: record_totals.measure(),
        });
    }
    Ok(ParityReport {
        model: runtime.model_id.clone(),
        model_revision: runtime.revision.clone(),
        records,
        overall: totals.measure(),
    })
}

fn token_ids(record: &ParityInput, base: &Path, index: usize) -> Result<Vec<u32>> {
    let ids = match (&record.token_ids, &record.token_ids_file) {
        (Some(ids), None) => ids.clone(),
        (None, Some(file)) => {
            let path: PathBuf = base.join(file);
            serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
            )
            .with_context(|| format!("{} is not a JSON list of token ids", path.display()))?
        }
        (Some(_), Some(_)) => {
            bail!("record {index} names both tokenIds and tokenIdsFile; give one")
        }
        (None, None) => bail!("record {index} names neither tokenIds nor tokenIdsFile"),
    };
    if ids.is_empty() {
        bail!("record {index} holds no token ids");
    }
    Ok(ids)
}

/// The reference states, `[positions, hidden]` in F32. A three-dimensional
/// tensor is accepted only when its leading batch holds a single sequence.
fn reference_states(path: &Path, tensor: &str) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let Some(states) = tensors.get(tensor) else {
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort();
        bail!(
            "{} holds no tensor {tensor:?}; it holds {names:?}",
            path.display()
        );
    };
    let states = states.to_dtype(DType::F32)?;
    match *states.dims() {
        [_, _] => Ok(states),
        [_, positions, width] if states.elem_count() == positions * width => {
            Ok(states.reshape((positions, width))?)
        }
        ref other => bail!(
            "{}'s {tensor:?} is {other:?}; Ster compares [positions, hidden] states",
            path.display()
        ),
    }
}

#[derive(Debug, Default)]
struct Totals {
    difference_squares: f64,
    reference_squares: f64,
    worst_relative: f64,
    largest: f64,
    /// One entry per compared position: its cosine, or `None` when either
    /// state is all zeros and the angle between them is undefined.
    cosines: Vec<Option<f64>>,
}

impl Totals {
    fn add(&mut self, mine: &[f32], reference: &[f32]) {
        let pairs = || {
            mine.iter()
                .zip(reference)
                .map(|(a, b)| (f64::from(*a), f64::from(*b)))
        };
        let difference: f64 = pairs().map(|(a, b)| (a - b) * (a - b)).sum();
        let norm_mine: f64 = pairs().map(|(a, _)| a * a).sum();
        let norm_reference: f64 = pairs().map(|(_, b)| b * b).sum();
        let dot: f64 = pairs().map(|(a, b)| a * b).sum();
        self.largest = pairs()
            .map(|(a, b)| (a - b).abs())
            .fold(self.largest, f64::max);
        self.difference_squares += difference;
        self.reference_squares += norm_reference;
        self.worst_relative = self
            .worst_relative
            .max(ratio(difference.sqrt(), norm_reference.sqrt()));
        let norms = (norm_mine * norm_reference).sqrt();
        self.cosines
            .push((norms.classify() != FpCategory::Zero).then(|| dot / norms));
    }

    fn merge(&mut self, other: &Self) {
        self.difference_squares += other.difference_squares;
        self.reference_squares += other.reference_squares;
        self.worst_relative = self.worst_relative.max(other.worst_relative);
        self.largest = self.largest.max(other.largest);
        self.cosines.extend_from_slice(&other.cosines);
    }

    fn measure(&self) -> ParityMeasure {
        let measured: Vec<f64> = self.cosines.iter().flatten().copied().collect();
        ParityMeasure {
            positions: self.cosines.len(),
            relative_error: ratio(
                self.difference_squares.sqrt(),
                self.reference_squares.sqrt(),
            ),
            worst_position_relative_error: self.worst_relative,
            largest_difference: self.largest,
            mean_cosine: (!measured.is_empty())
                .then(|| measured.iter().sum::<f64>() / measured.len() as f64),
            smallest_cosine: measured.iter().copied().reduce(f64::min),
        }
    }
}

/// `numerator / denominator`. Over a zero denominator, a zero numerator is
/// itself the answer (identical zero states differ by nothing) and any other
/// numerator is infinitely far off.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.classify() != FpCategory::Zero {
        return numerator / denominator;
    }
    if numerator.classify() == FpCategory::Zero {
        numerator
    } else {
        f64::INFINITY
    }
}
