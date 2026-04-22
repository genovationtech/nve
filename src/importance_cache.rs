//! Layer importance score cache.
//!
//! Saves and loads per-layer profiling results so that re-launching NVE on the
//! same model skips the profiling pass entirely.
//!
//! Cache location: `~/.cache/nve/importance/<model_key>.json`
//! where `model_key` is the 16-hex-char FNV-1a-64 fingerprint of the
//! canonical model directory path.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const NVE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Serialize, Deserialize)]
struct CacheFile {
    nve_version: String,
    model_path: String,
    num_layers: usize,
    scores: Vec<f64>,
    timestamp: u64,
}

/// FNV-1a 64-bit hash of a byte slice.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn model_key(model_dir: &Path) -> String {
    let canonical = model_dir
        .canonicalize()
        .unwrap_or_else(|_| model_dir.to_path_buf());
    let s = canonical.to_string_lossy();
    format!("{:016x}", fnv1a_64(s.as_bytes()))
}

fn cache_dir() -> PathBuf {
    // Prefer XDG_CACHE_HOME, fall back to ~/.cache
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("nve").join("importance")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home)
            .join(".cache")
            .join("nve")
            .join("importance")
    } else {
        PathBuf::from("/tmp/nve_importance_cache")
    }
}

pub struct ImportanceCache;

impl ImportanceCache {
    /// Full path to the cache file for this model.
    pub fn cache_path(model_dir: &Path) -> PathBuf {
        cache_dir().join(format!("{}.json", model_key(model_dir)))
    }

    /// Load cached importance scores.
    ///
    /// Returns `None` if:
    /// - The cache file doesn't exist
    /// - The NVE version changed (scores may be incompatible)
    /// - The number of layers doesn't match
    /// - The file is corrupt
    pub fn load(model_dir: &Path, num_layers: usize) -> Option<Vec<f64>> {
        let path = Self::cache_path(model_dir);
        if !path.exists() {
            return None;
        }
        let contents = std::fs::read_to_string(&path).ok()?;
        let cache: CacheFile = serde_json::from_str(&contents).ok()?;

        // Reject if layer count or version changed.
        if cache.num_layers != num_layers {
            log::debug!(
                "Importance cache layer count mismatch ({} vs {}), ignoring",
                cache.num_layers,
                num_layers
            );
            return None;
        }
        if cache.nve_version != NVE_VERSION {
            log::debug!(
                "Importance cache version mismatch ({} vs {}), ignoring",
                cache.nve_version,
                NVE_VERSION
            );
            return None;
        }

        log::info!(
            "Loaded layer importance cache from {:?} ({} layers)",
            path,
            num_layers
        );
        Some(cache.scores)
    }

    /// Persist importance scores to disk.
    pub fn save(
        model_dir: &Path,
        scores: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = cache_dir();
        std::fs::create_dir_all(&dir)?;

        let path = Self::cache_path(model_dir);
        let canonical = model_dir
            .canonicalize()
            .unwrap_or_else(|_| model_dir.to_path_buf());

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let cache = CacheFile {
            nve_version: NVE_VERSION.to_string(),
            model_path: canonical.to_string_lossy().into_owned(),
            num_layers: scores.len(),
            scores: scores.to_vec(),
            timestamp,
        };

        let json = serde_json::to_string_pretty(&cache)?;
        std::fs::write(&path, json)?;
        log::info!("Saved layer importance cache to {:?}", path);
        Ok(())
    }

    /// Delete the cache entry for this model (e.g. after a /profile REPL command).
    pub fn invalidate(model_dir: &Path) {
        let path = Self::cache_path(model_dir);
        let _ = std::fs::remove_file(path);
    }
}
