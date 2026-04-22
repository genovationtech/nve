//! Llama model configuration parser.
//!
//! Parses HuggingFace `config.json` into a validated `LlamaConfig` struct.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid config: {0}")]
    Validation(String),
}

/// Llama model configuration (compatible with HuggingFace config.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,

    #[serde(default)]
    pub bos_token_id: Option<u32>,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,

    #[serde(default = "default_rope_scaling_factor")]
    pub factor: f64,

    #[serde(default = "default_low_freq_factor")]
    pub low_freq_factor: f64,

    #[serde(default = "default_high_freq_factor")]
    pub high_freq_factor: f64,

    #[serde(default = "default_original_max_position_embeddings")]
    pub original_max_position_embeddings: usize,
}

// Llama 3.2 1B defaults
fn default_hidden_size() -> usize { 2048 }
fn default_num_hidden_layers() -> usize { 16 }
fn default_num_attention_heads() -> usize { 32 }
fn default_num_key_value_heads() -> usize { 8 }
fn default_intermediate_size() -> usize { 8192 }
fn default_vocab_size() -> usize { 128256 }
fn default_max_position_embeddings() -> usize { 131072 }
fn default_rope_theta() -> f64 { 500000.0 }
fn default_rms_norm_eps() -> f64 { 1e-5 }
fn default_tie_word_embeddings() -> bool { true }
fn default_torch_dtype() -> String { "bfloat16".to_string() }
fn default_rope_type() -> String { "llama3".to_string() }
fn default_rope_scaling_factor() -> f64 { 32.0 }
fn default_low_freq_factor() -> f64 { 1.0 }
fn default_high_freq_factor() -> f64 { 4.0 }
fn default_original_max_position_embeddings() -> usize { 8192 }

impl Default for LlamaConfig {
    fn default() -> Self {
        LlamaConfig {
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            intermediate_size: default_intermediate_size(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_scaling: Some(RopeScalingConfig::default()),
            tie_word_embeddings: default_tie_word_embeddings(),
            torch_dtype: default_torch_dtype(),
            bos_token_id: Some(128000),
            eos_token_id: Some(serde_json::Value::Number(128001.into())),
        }
    }
}

impl Default for RopeScalingConfig {
    fn default() -> Self {
        RopeScalingConfig {
            rope_type: default_rope_type(),
            factor: default_rope_scaling_factor(),
            low_freq_factor: default_low_freq_factor(),
            high_freq_factor: default_high_freq_factor(),
            original_max_position_embeddings: default_original_max_position_embeddings(),
        }
    }
}

impl LlamaConfig {
    /// Load config from a JSON file path.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path)?;
        let config: LlamaConfig = serde_json::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Load config from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let config: LlamaConfig = serde_json::from_str(json)?;
        config.validate()?;
        Ok(config)
    }

    /// Load from a model directory (looks for config.json).
    pub fn from_model_dir(dir: &Path) -> Result<Self, ConfigError> {
        Self::from_file(&dir.join("config.json"))
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 {
            return Err(ConfigError::Validation("hidden_size must be > 0".into()));
        }
        if self.num_hidden_layers == 0 {
            return Err(ConfigError::Validation("num_hidden_layers must be > 0".into()));
        }
        if self.num_attention_heads == 0 {
            return Err(ConfigError::Validation("num_attention_heads must be > 0".into()));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::Validation(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ConfigError::Validation(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }

    /// Number of query heads per KV head group.
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Total model parameters (approximate, for Llama architecture).
    pub fn estimated_params(&self) -> usize {
        let d = self.hidden_size;
        let n = self.num_hidden_layers;
        let v = self.vocab_size;
        let ff = self.intermediate_size;
        let kv_heads = self.num_key_value_heads;
        let head_dim = self.head_dim();

        // Embedding
        let embed = v * d;
        // Per layer: Q, K, V projections + output proj + gate/up/down FFN + 2 norms
        let q_proj = d * d;
        let k_proj = kv_heads * head_dim * d;
        let v_proj = kv_heads * head_dim * d;
        let o_proj = d * d;
        let gate = d * ff;
        let up = d * ff;
        let down = ff * d;
        let norms = 2 * d; // 2 RMSNorm per layer
        let per_layer = q_proj + k_proj + v_proj + o_proj + gate + up + down + norms;
        // Final norm + lm_head (may be tied)
        let final_norm = d;
        let lm_head = if self.tie_word_embeddings { 0 } else { d * v };

        embed + n * per_layer + final_norm + lm_head
    }

    /// Total size in bytes for bf16 weights.
    pub fn estimated_size_bytes(&self) -> usize {
        self.estimated_params() * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_llama_1b() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 16);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.intermediate_size, 8192);
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.num_queries_per_kv(), 4);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-05
        }"#;
        let cfg = LlamaConfig::from_json(json).unwrap();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_validation_catches_bad_config() {
        let mut cfg = LlamaConfig::default();
        cfg.hidden_size = 100;
        cfg.num_attention_heads = 32;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_estimated_params() {
        let cfg = LlamaConfig::default();
        let params = cfg.estimated_params();
        // Llama 3.2 1B should be ~1.2B params
        assert!(params > 1_000_000_000);
        assert!(params < 1_500_000_000);
    }
}
