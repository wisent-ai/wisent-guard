use std::path::Path;

use crate::{artifact::Document, lora};

/// The object recording how an artifact was produced. A sidecar read or JSON
/// failure is observable; missing metadata and binary adapter weights are not
/// evidence that the training configuration matched the current runtime.
pub(super) fn read(artifact: &Path) -> Result<Option<serde_json::Value>, String> {
    let sidecar = lora::Artifact::sidecar_path(artifact);
    let (bytes, from_sidecar) = match std::fs::read(&sidecar) {
        Ok(bytes) => (bytes, true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::read(artifact) {
                Ok(bytes) => (bytes, false),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(format!("{}: {error}", artifact.display())),
            }
        }
        Err(error) => return Err(format!("{}: {error}", sidecar.display())),
    };
    let document = match serde_json::from_slice::<serde_json::Value>(&bytes) {
        Ok(document) => document,
        Err(error) if from_sidecar => {
            return Err(format!("{} is not JSON: {error}", sidecar.display()))
        }
        Err(_) => return Ok(None),
    };
    // Steering documents already use the JSON extension, so their sidecar
    // path is the artifact itself. The document shape, not its path, decides
    // whether the record is at the top level or inside an adapter's train.
    Ok(Some(match Document::recognise(&bytes) {
        Document::Steering => document,
        Document::AdapterSidecar | Document::Unrecognised => document["train"].clone(),
    }))
}
