//! Persistent NVE configuration stored at ~/.config/nve/config.toml.

use std::path::PathBuf;

use directories::BaseDirs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NveConfig {
    pub hf_token: Option<String>,
    /// Hot-tier memory budget in MB (always-resident RAM).
    pub hot_budget_mb: usize,
    /// Warm-tier memory budget in MB (LRU-cached RAM).
    pub warm_budget_mb: usize,
    /// Auto-detect budgets from available system RAM.
    pub auto_budget: bool,
    /// Default quantization mode: none, q4, q8, q3, q2, q1.
    pub quantize: String,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Maximum tokens to generate per turn.
    pub max_tokens: usize,
}

impl Default for NveConfig {
    fn default() -> Self {
        Self {
            hf_token: None,
            hot_budget_mb: 512,
            warm_budget_mb: 2048,
            auto_budget: false,
            quantize: "none".into(),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
        }
    }
}

impl NveConfig {
    /// Return the platform-correct path to the config file.
    pub fn config_path() -> PathBuf {
        if let Some(base) = BaseDirs::new() {
            base.config_dir().join("nve").join("config.toml")
        } else {
            // Fallback: ~/.config/nve/config.toml
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home)
                .join(".config")
                .join("nve")
                .join("config.toml")
        }
    }

    /// Load config from disk. Returns defaults if the file is missing or invalid.
    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(cfg) = toml::from_str::<Self>(&contents) {
                    return cfg;
                }
            }
        }
        Self::default()
    }

    /// Persist config to disk.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Return the HF token from config, falling back to env vars.
    pub fn hf_token_resolved(&self) -> Option<String> {
        if let Some(ref t) = self.hf_token {
            if !t.is_empty() {
                return Some(t.clone());
            }
        }
        std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok()
    }

    /// Reset to defaults and save.
    pub fn reset() -> Result<(), Box<dyn std::error::Error>> {
        Self::default().save()
    }
}
