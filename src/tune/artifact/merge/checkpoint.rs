//! Reading a checkpoint's tensors and writing them back as an ordinary
//! checkpoint directory: what a merge and an ablation both rewrite.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

use crate::{runtime::Checkpoint, workflow};

/// Every tensor of `source`'s weight files, by name; a name stored in two
/// shards is refused rather than letting the last shard win.
pub(crate) fn read_tensors(
    source: &Checkpoint,
    model: &str,
    device: &Device,
) -> Result<BTreeMap<String, Tensor>> {
    workflow::progress(format!(
        "reading {} weight file(s) from {model}",
        source.weights.len()
    ));
    let mut tensors: BTreeMap<String, Tensor> = BTreeMap::new();
    for file in &source.weights {
        let loaded = candle_core::safetensors::load(file, device)
            .with_context(|| format!("failed to read model weights {}", file.display()))?;
        for (name, tensor) in loaded {
            if tensors.insert(name.clone(), tensor).is_some() {
                // Sharded checkpoints partition the tensors; the same name in
                // two shards means the shards disagree, and silently keeping
                // the last one merges half a model.
                bail!("model weight {name} appears in more than one weight file");
            }
        }
    }
    Ok(tensors)
}

/// What [`write_checkpoint`] wrote.
pub(crate) struct Written {
    pub(crate) files: Vec<String>,
    pub(crate) parameters: usize,
    pub(crate) total: usize,
}

/// `tensors` written as an ordinary checkpoint directory at `output`, beside
/// copies of `source`'s own config, tokenizer and chat template.
pub(crate) fn write_checkpoint(
    source: &Checkpoint,
    output: &Path,
    tensors: &BTreeMap<String, Tensor>,
) -> Result<Written> {
    fs::create_dir_all(output).with_context(|| format!("failed to create {}", output.display()))?;
    let weights_file = source.layout.merged_weights();
    let weights_path = output.join(weights_file);
    let flat: HashMap<String, Tensor> = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    let parameters: usize = tensors.values().map(|tensor| tensor.elem_count()).sum();
    let total = tensors.len();
    candle_core::safetensors::save(&flat, &weights_path)
        .with_context(|| format!("failed to write {}", weights_path.display()))?;

    // Copied rather than regenerated. The tokenizer and the config are the
    // source's own, and a rewritten checkpoint that tokenized differently from
    // the model it came from would be a different model wearing its name.
    //
    // `tokenizer_config.json` and `chat_template.jinja` are part of that
    // statement and not extras. The chat template lives in one or the other,
    // and an adapter trained with `--chat-template auto` was trained on the
    // markers that template renders — so a merge that dropped it produced a
    // directory whose own `tune evaluate` silently scored raw text and
    // disagreed with the same adapter attached to the source checkpoint. Both
    // are optional because plenty of checkpoints publish neither; a base model
    // with no template merges to a directory with no template, which is the
    // same statement in the other direction.
    let mut files = vec![weights_file.to_owned()];
    let optional = [
        (source.tokenizer_config.as_deref(), "tokenizer_config.json"),
        (source.chat_template.as_deref(), "chat_template.jinja"),
    ];
    let tokenizer_leaf = source
        .tokenizer
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| {
            format!(
                "tokenizer path {} has no file name",
                source.tokenizer.display()
            )
        })?;
    let required = [
        (&source.config, source.layout.config_file()),
        (&source.tokenizer, tokenizer_leaf),
    ];
    for (from, leaf) in required
        .into_iter()
        .map(|(from, leaf)| (from.as_path(), leaf))
        .chain(
            optional
                .into_iter()
                .filter_map(|(from, leaf)| from.map(|from| (from, leaf))),
        )
    {
        let to = output.join(leaf);
        fs::copy(from, &to)
            .with_context(|| format!("failed to copy {} to {}", from.display(), to.display()))?;
        files.push(leaf.to_owned());
    }
    workflow::progress(format!(
        "wrote {} to {}",
        files.join(", "),
        output.display()
    ));
    Ok(Written {
        files,
        parameters,
        total,
    })
}
