//! HuggingFace Hub model download support.
//!
//! Downloads model files (config.json, tokenizer.json, safetensors) from
//! the HuggingFace Hub using the HTTP API. Supports caching to avoid
//! re-downloading.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use log::info;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("model not found: {0}")]
    NotFound(String),
    #[error("huggingface-cli not available and HTTP download failed: {0}")]
    NoDownloadMethod(String),
    #[error("invalid model ID: {0}")]
    InvalidModelId(String),
}

/// Parsed model identifier (owner/name or just name).
#[derive(Debug, Clone)]
pub struct ModelId {
    pub repo_id: String,
    pub revision: String,
}

impl ModelId {
    pub fn parse(input: &str) -> Result<Self, HubError> {
        let input = input.trim();
        // Check if it's a local path.
        if Path::new(input).exists() {
            return Err(HubError::InvalidModelId(
                "input is a local path, not a model ID".into(),
            ));
        }

        let (repo, rev) = if input.contains('@') {
            let parts: Vec<&str> = input.splitn(2, '@').collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (input.to_string(), "main".to_string())
        };

        // Basic validation: should contain at most one slash.
        if repo.matches('/').count() > 1 {
            return Err(HubError::InvalidModelId(format!(
                "invalid repo ID: '{}'", repo
            )));
        }

        Ok(ModelId {
            repo_id: repo,
            revision: rev,
        })
    }
}

/// Resolve a model path — either a local directory or a HuggingFace model ID.
/// If it's a model ID, downloads to cache_dir and returns the cached path.
/// `hf_token` takes precedence over the HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars.
pub fn resolve_model_path(
    model_path: &str,
    cache_dir: Option<&Path>,
    hf_token: Option<&str>,
) -> Result<PathBuf, HubError> {
    let path = Path::new(model_path);

    // If it's already a local directory with config.json, use it directly.
    if path.is_dir() && path.join("config.json").exists() {
        return Ok(path.to_path_buf());
    }

    // Parse as a HuggingFace model ID.
    let model_id = ModelId::parse(model_path)?;

    // Determine cache directory.
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);

    let model_cache = cache
        .join("models")
        .join(model_id.repo_id.replace('/', "--"));

    // Check if already cached.
    if model_cache.join("config.json").exists() {
        info!("Using cached model at {:?}", model_cache);
        return Ok(model_cache);
    }

    // Download the model.
    info!("Downloading model '{}' to {:?}", model_id.repo_id, model_cache);
    download_model(&model_id, &model_cache, hf_token)?;

    Ok(model_cache)
}

fn default_cache_dir() -> PathBuf {
    // Use HF_HOME or ~/.cache/nve
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home).join("nve")
    } else if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("nve")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("nve")
    } else {
        PathBuf::from("/tmp/nve_cache")
    }
}

/// Download a model from HuggingFace Hub.
fn download_model(model_id: &ModelId, dest: &Path, hf_token: Option<&str>) -> Result<(), HubError> {
    // Try huggingface-cli first (handles auth, LFS, etc.).
    if try_hf_cli_download(model_id, dest)? {
        return Ok(());
    }

    // Fall back to direct HTTP download.
    download_via_http(model_id, dest, hf_token)
}

fn try_hf_cli_download(model_id: &ModelId, dest: &Path) -> Result<bool, HubError> {
    // Check if huggingface-cli is available.
    let check = Command::new("huggingface-cli").arg("--help").output();
    if check.is_err() {
        return Ok(false);
    }

    info!("Using huggingface-cli to download {}", model_id.repo_id);
    fs::create_dir_all(dest)?;

    let mut cmd = Command::new("huggingface-cli");
    cmd.arg("download")
        .arg(&model_id.repo_id)
        .arg("--local-dir")
        .arg(dest)
        .arg("--revision")
        .arg(&model_id.revision);

    // Only download essential files.
    cmd.arg("--include")
        .arg("config.json")
        .arg("--include")
        .arg("tokenizer.json")
        .arg("--include")
        .arg("tokenizer_config.json")
        .arg("--include")
        .arg("*.safetensors")
        .arg("--include")
        .arg("*.safetensors.index.json");

    let output = cmd.output()?;
    if output.status.success() {
        info!("Download complete via huggingface-cli");
        Ok(true)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        info!("huggingface-cli failed: {}", stderr.trim());
        Ok(false)
    }
}

fn download_via_http(model_id: &ModelId, dest: &Path, hf_token: Option<&str>) -> Result<(), HubError> {
    fs::create_dir_all(dest)?;

    let base_url = format!(
        "https://huggingface.co/{}/resolve/{}",
        model_id.repo_id, model_id.revision
    );

    // Explicit token takes precedence over env vars.
    let auth_token = hf_token
        .map(|t| t.to_string())
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

    // Required files.
    let required = vec!["config.json", "tokenizer.json"];
    // Optional files to try.
    let optional = vec![
        "tokenizer_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
    ];

    for filename in &required {
        let url = format!("{}/{}", base_url, filename);
        let dest_path = dest.join(filename);
        download_file(&url, &dest_path, auth_token.as_deref()).map_err(|e| {
            HubError::Http(format!("failed to download {}: {}", filename, e))
        })?;
    }

    for filename in &optional {
        let url = format!("{}/{}", base_url, filename);
        let dest_path = dest.join(filename);
        let _ = download_file(&url, &dest_path, auth_token.as_deref());
    }

    // If we got an index file, download the shard files.
    let index_path = dest.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_json = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_json)
            .map_err(|e| HubError::Http(format!("invalid index JSON: {}", e)))?;

        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            let shard_files: std::collections::HashSet<&str> = weight_map
                .values()
                .filter_map(|v| v.as_str())
                .collect();

            for shard in shard_files {
                let shard_dest = dest.join(shard);
                if !shard_dest.exists() {
                    let url = format!("{}/{}", base_url, shard);
                    info!("Downloading shard: {}", shard);
                    download_file(&url, &shard_dest, auth_token.as_deref()).map_err(|e| {
                        HubError::Http(format!("failed to download shard {}: {}", shard, e))
                    })?;
                }
            }
        }
    }

    Ok(())
}

fn download_file(url: &str, dest: &Path, auth_token: Option<&str>) -> Result<(), HubError> {
    info!("Downloading: {} -> {:?}", url, dest);

    // Use curl since it's universally available and handles redirects/LFS.
    let mut cmd = Command::new("curl");
    cmd.arg("-fSL") // fail on HTTP errors, show errors, follow redirects
        .arg("-o")
        .arg(dest)
        .arg("--progress-bar");

    if let Some(token) = auth_token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {}", token));
    }

    cmd.arg(url);

    let output = cmd.output()?;
    if output.status.success() {
        Ok(())
    } else {
        // Clean up partial download.
        let _ = fs::remove_file(dest);
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(HubError::Http(format!("curl failed: {}", stderr.trim())))
    }
}

/// Check if a path looks like a HuggingFace model ID (org/model or just model).
pub fn is_model_id(s: &str) -> bool {
    !Path::new(s).exists() && !s.starts_with('/') && !s.starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_id() {
        let id = ModelId::parse("meta-llama/Llama-3.2-1B").unwrap();
        assert_eq!(id.repo_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(id.revision, "main");
    }

    #[test]
    fn test_parse_model_id_with_revision() {
        let id = ModelId::parse("meta-llama/Llama-3.2-1B@refs/pr/123").unwrap();
        assert_eq!(id.repo_id, "meta-llama/Llama-3.2-1B");
        assert_eq!(id.revision, "refs/pr/123");
    }
}
