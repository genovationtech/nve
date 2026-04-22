//! Architecture detection and unified model configuration.
//!
//! Reads HuggingFace `config.json` and determines the model architecture,
//! producing a normalized `UnifiedConfig` that the generic model can use.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArchError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported architecture: {0}")]
    UnsupportedArch(String),
    #[error("missing required config field: {0}")]
    MissingField(String),
    #[error("invalid config: {0}")]
    Validation(String),
}

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Mistral,
    Qwen2,
    Phi3,
    Gemma,
    Gemma2,
    GPTNeoX,
    GPT2,
    Falcon,
    StableLM,
    StarCoder2,
    InternLM2,
    OLMo,
    DeepSeek,
}

impl Architecture {
    /// Detect architecture from HuggingFace config.json `architectures` or `model_type` field.
    pub fn detect(raw: &serde_json::Value) -> Result<Self, ArchError> {
        // Try architectures array first.
        if let Some(archs) = raw.get("architectures").and_then(|v| v.as_array()) {
            for a in archs {
                if let Some(s) = a.as_str() {
                    if let Some(arch) = Self::from_arch_string(s) {
                        return Ok(arch);
                    }
                }
            }
        }
        // Fallback to model_type.
        if let Some(mt) = raw.get("model_type").and_then(|v| v.as_str()) {
            if let Some(arch) = Self::from_model_type(mt) {
                return Ok(arch);
            }
            return Err(ArchError::UnsupportedArch(mt.to_string()));
        }
        Err(ArchError::UnsupportedArch(
            "no 'architectures' or 'model_type' found in config.json".into(),
        ))
    }

    fn from_arch_string(s: &str) -> Option<Self> {
        match s {
            s if s.contains("Llama") => Some(Architecture::Llama),
            s if s.contains("Mistral") => Some(Architecture::Mistral),
            s if s.contains("Qwen2") => Some(Architecture::Qwen2),
            s if s.contains("Phi3") || s.contains("PhiForCausalLM") => Some(Architecture::Phi3),
            s if s.contains("Gemma2") => Some(Architecture::Gemma2),
            s if s.contains("Gemma") => Some(Architecture::Gemma),
            s if s.contains("GPTNeoX") => Some(Architecture::GPTNeoX),
            s if s.contains("GPT2") => Some(Architecture::GPT2),
            s if s.contains("Falcon") || s.contains("RWForCausalLM") => Some(Architecture::Falcon),
            s if s.contains("StableLm") || s.contains("StableLM") => Some(Architecture::StableLM),
            s if s.contains("Starcoder2") => Some(Architecture::StarCoder2),
            s if s.contains("InternLM2") => Some(Architecture::InternLM2),
            s if s.contains("OLMo") => Some(Architecture::OLMo),
            s if s.contains("DeepSeek") => Some(Architecture::DeepSeek),
            _ => None,
        }
    }

    fn from_model_type(s: &str) -> Option<Self> {
        match s {
            "llama" => Some(Architecture::Llama),
            "mistral" => Some(Architecture::Mistral),
            "qwen2" => Some(Architecture::Qwen2),
            "phi3" | "phi" => Some(Architecture::Phi3),
            "gemma2" => Some(Architecture::Gemma2),
            "gemma" => Some(Architecture::Gemma),
            "gpt_neox" => Some(Architecture::GPTNeoX),
            "gpt2" => Some(Architecture::GPT2),
            "falcon" | "RefinedWeb" | "RefinedWebModel" => Some(Architecture::Falcon),
            "stablelm" | "stablelm_epoch" => Some(Architecture::StableLM),
            "starcoder2" => Some(Architecture::StarCoder2),
            "internlm2" => Some(Architecture::InternLM2),
            "olmo" => Some(Architecture::OLMo),
            "deepseek" => Some(Architecture::DeepSeek),
            _ => None,
        }
    }
}

/// What kind of normalization the model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    RMSNorm,
    LayerNorm,
}

/// What kind of position encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosEncoding {
    RoPE,
    Learned,
    ALiBi,
}

/// What kind of FFN activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnType {
    /// SiLU-gated: down(silu(gate(x)) * up(x))  — 3 weight matrices
    SwiGLU,
    /// GELU-gated: down(gelu(gate(x)) * up(x))  — 3 weight matrices
    GeGLU,
    /// Standard 2-layer FFN with GELU: down(gelu(up(x)))  — 2 weight matrices (+ optional bias)
    GELU,
    /// Standard 2-layer FFN with ReLU
    ReLU,
}

/// Unified model configuration derived from any supported HuggingFace config.json.
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    pub arch: Architecture,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub norm_type: NormType,
    pub pos_encoding: PosEncoding,
    pub ffn_type: FfnType,
    pub rope_scaling: Option<crate::config::RopeScalingConfig>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    /// Whether the model uses bias in attention projections.
    pub attn_bias: bool,
    /// Whether the model uses bias in FFN layers.
    pub ffn_bias: bool,
    /// Whether the model uses bias in norm layers.
    pub norm_bias: bool,
    /// Whether attention and FFN are computed in parallel (GPT-NeoX style).
    pub parallel_attn_ffn: bool,
    /// Partial rotary dimension (for Phi models that only rotate part of head_dim).
    pub rotary_dim: Option<usize>,
}

impl UnifiedConfig {
    pub fn from_model_dir(dir: &Path) -> Result<Self, ArchError> {
        let contents = std::fs::read_to_string(dir.join("config.json"))?;
        Self::from_json(&contents)
    }

    pub fn from_json(json: &str) -> Result<Self, ArchError> {
        let raw: serde_json::Value = serde_json::from_str(json)?;
        let arch = Architecture::detect(&raw)?;
        Self::from_raw(arch, &raw)
    }

    fn from_raw(arch: Architecture, raw: &serde_json::Value) -> Result<Self, ArchError> {
        let hidden_size = get_usize(raw, "hidden_size")
            .or_else(|| get_usize(raw, "n_embd"))
            .or_else(|| get_usize(raw, "d_model"))
            .ok_or_else(|| ArchError::MissingField("hidden_size".into()))?;

        let num_hidden_layers = get_usize(raw, "num_hidden_layers")
            .or_else(|| get_usize(raw, "n_layer"))
            .or_else(|| get_usize(raw, "num_layers"))
            .ok_or_else(|| ArchError::MissingField("num_hidden_layers".into()))?;

        let num_attention_heads = get_usize(raw, "num_attention_heads")
            .or_else(|| get_usize(raw, "n_head"))
            .ok_or_else(|| ArchError::MissingField("num_attention_heads".into()))?;

        let num_key_value_heads = get_usize(raw, "num_key_value_heads")
            .or_else(|| get_usize(raw, "num_kv_heads"))
            .or_else(|| {
                // Falcon uses multi_query
                if raw.get("multi_query").and_then(|v| v.as_bool()).unwrap_or(false) {
                    Some(1)
                } else {
                    None
                }
            })
            .unwrap_or(num_attention_heads); // Default: MHA

        let intermediate_size = get_usize(raw, "intermediate_size")
            .or_else(|| get_usize(raw, "n_inner"))
            .or_else(|| get_usize(raw, "ffn_dim"))
            .unwrap_or(hidden_size * 4);

        let vocab_size = get_usize(raw, "vocab_size")
            .ok_or_else(|| ArchError::MissingField("vocab_size".into()))?;

        let max_position_embeddings = get_usize(raw, "max_position_embeddings")
            .or_else(|| get_usize(raw, "n_positions"))
            .or_else(|| get_usize(raw, "max_seq_len"))
            .unwrap_or(2048);

        let rope_theta = get_f64(raw, "rope_theta").unwrap_or(10000.0);

        let norm_eps = get_f64(raw, "rms_norm_eps")
            .or_else(|| get_f64(raw, "layer_norm_eps"))
            .or_else(|| get_f64(raw, "layer_norm_epsilon"))
            .unwrap_or(1e-5);

        let tie_word_embeddings = raw
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(arch_default_tie(arch));

        let rope_scaling = raw.get("rope_scaling").and_then(|v| {
            serde_json::from_value::<crate::config::RopeScalingConfig>(v.clone()).ok()
        });

        let bos_token_id = raw.get("bos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32);
        let eos_token_id = raw.get("eos_token_id").and_then(|v| {
            v.as_u64().map(|v| v as u32).or_else(|| {
                v.as_array().and_then(|a| a.first()).and_then(|v| v.as_u64()).map(|v| v as u32)
            })
        });

        let (norm_type, pos_encoding, ffn_type, attn_bias, ffn_bias, norm_bias, parallel_attn_ffn) =
            arch_defaults(arch, raw);

        let rotary_dim = get_usize(raw, "rotary_dim")
            .or_else(|| get_usize(raw, "partial_rotary_factor").map(|_| {
                let frac = get_f64(raw, "partial_rotary_factor").unwrap_or(1.0);
                ((hidden_size / num_attention_heads) as f64 * frac) as usize
            }));

        let config = UnifiedConfig {
            arch,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            norm_eps,
            tie_word_embeddings,
            norm_type,
            pos_encoding,
            ffn_type,
            rope_scaling,
            bos_token_id,
            eos_token_id,
            attn_bias,
            ffn_bias,
            norm_bias,
            parallel_attn_ffn,
            rotary_dim,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ArchError> {
        if self.hidden_size == 0 {
            return Err(ArchError::Validation("hidden_size must be > 0".into()));
        }
        if self.num_hidden_layers == 0 {
            return Err(ArchError::Validation("num_hidden_layers must be > 0".into()));
        }
        if self.num_attention_heads == 0 {
            return Err(ArchError::Validation("num_attention_heads must be > 0".into()));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ArchError::Validation(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_queries_per_kv(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Total model parameters (approximate).
    pub fn estimated_params(&self) -> usize {
        let d = self.hidden_size;
        let n = self.num_hidden_layers;
        let v = self.vocab_size;
        let ff = self.intermediate_size;
        let kv_heads = self.num_key_value_heads;
        let head_dim = self.head_dim();

        let embed = v * d;
        let q_proj = d * d;
        let k_proj = kv_heads * head_dim * d;
        let v_proj = kv_heads * head_dim * d;
        let o_proj = d * d;

        let ffn_params = match self.ffn_type {
            FfnType::SwiGLU | FfnType::GeGLU => d * ff * 3, // gate + up + down
            FfnType::GELU | FfnType::ReLU => d * ff * 2,    // up + down
        };
        let norms = 2 * d;
        let per_layer = q_proj + k_proj + v_proj + o_proj + ffn_params + norms;
        let final_norm = d;
        let lm_head = if self.tie_word_embeddings { 0 } else { d * v };

        embed + n * per_layer + final_norm + lm_head
    }

    pub fn estimated_size_bytes(&self) -> usize {
        self.estimated_params() * 2 // bf16
    }

    /// Convert to a LlamaConfig for backward compatibility.
    pub fn to_llama_config(&self) -> crate::config::LlamaConfig {
        crate::config::LlamaConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.norm_eps,
            rope_scaling: self.rope_scaling.clone(),
            tie_word_embeddings: self.tie_word_embeddings,
            torch_dtype: "bfloat16".into(),
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id.map(|v| serde_json::Value::Number(v.into())),
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Architecture::Llama => write!(f, "Llama"),
            Architecture::Mistral => write!(f, "Mistral"),
            Architecture::Qwen2 => write!(f, "Qwen2"),
            Architecture::Phi3 => write!(f, "Phi-3"),
            Architecture::Gemma => write!(f, "Gemma"),
            Architecture::Gemma2 => write!(f, "Gemma2"),
            Architecture::GPTNeoX => write!(f, "GPT-NeoX"),
            Architecture::GPT2 => write!(f, "GPT-2"),
            Architecture::Falcon => write!(f, "Falcon"),
            Architecture::StableLM => write!(f, "StableLM"),
            Architecture::StarCoder2 => write!(f, "StarCoder2"),
            Architecture::InternLM2 => write!(f, "InternLM2"),
            Architecture::OLMo => write!(f, "OLMo"),
            Architecture::DeepSeek => write!(f, "DeepSeek"),
        }
    }
}

// ── Helpers ──

fn get_usize(v: &serde_json::Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
}

fn get_f64(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|v| v.as_f64())
}

fn arch_default_tie(arch: Architecture) -> bool {
    matches!(arch, Architecture::Llama | Architecture::Gemma | Architecture::Gemma2 | Architecture::GPT2)
}

fn arch_defaults(
    arch: Architecture,
    raw: &serde_json::Value,
) -> (NormType, PosEncoding, FfnType, bool, bool, bool, bool) {
    match arch {
        Architecture::Llama | Architecture::Mistral | Architecture::DeepSeek => {
            (NormType::RMSNorm, PosEncoding::RoPE, FfnType::SwiGLU, false, false, false, false)
        }
        Architecture::Qwen2 => {
            (NormType::RMSNorm, PosEncoding::RoPE, FfnType::SwiGLU, true, false, false, false)
        }
        Architecture::Phi3 => {
            (NormType::RMSNorm, PosEncoding::RoPE, FfnType::SwiGLU, false, false, false, false)
        }
        Architecture::Gemma | Architecture::Gemma2 => {
            (NormType::RMSNorm, PosEncoding::RoPE, FfnType::GeGLU, false, false, false, false)
        }
        Architecture::GPTNeoX => {
            let parallel = raw.get("use_parallel_residual")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            (NormType::LayerNorm, PosEncoding::RoPE, FfnType::GELU, true, true, true, parallel)
        }
        Architecture::GPT2 => {
            (NormType::LayerNorm, PosEncoding::Learned, FfnType::GELU, true, true, true, false)
        }
        Architecture::Falcon => {
            let parallel = raw.get("parallel_attn")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            (NormType::LayerNorm, PosEncoding::RoPE, FfnType::GELU, false, false, true, parallel)
        }
        Architecture::StableLM => {
            (NormType::LayerNorm, PosEncoding::RoPE, FfnType::SwiGLU, false, false, false, false)
        }
        Architecture::StarCoder2 => {
            (NormType::LayerNorm, PosEncoding::RoPE, FfnType::GELU, true, true, true, false)
        }
        Architecture::InternLM2 => {
            (NormType::RMSNorm, PosEncoding::RoPE, FfnType::SwiGLU, false, false, false, false)
        }
        Architecture::OLMo => {
            (NormType::LayerNorm, PosEncoding::RoPE, FfnType::SwiGLU, false, false, false, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_llama() {
        let json = r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","hidden_size":2048,"num_hidden_layers":16,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":8192,"vocab_size":128256}"#;
        let config = UnifiedConfig::from_json(json).unwrap();
        assert_eq!(config.arch, Architecture::Llama);
        assert_eq!(config.norm_type, NormType::RMSNorm);
        assert_eq!(config.ffn_type, FfnType::SwiGLU);
    }

    #[test]
    fn test_detect_gpt2() {
        let json = r#"{"architectures":["GPT2LMHeadModel"],"model_type":"gpt2","n_embd":768,"n_layer":12,"n_head":12,"vocab_size":50257,"n_positions":1024}"#;
        let config = UnifiedConfig::from_json(json).unwrap();
        assert_eq!(config.arch, Architecture::GPT2);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.norm_type, NormType::LayerNorm);
        assert_eq!(config.pos_encoding, PosEncoding::Learned);
        assert_eq!(config.ffn_type, FfnType::GELU);
    }

    #[test]
    fn test_detect_mistral() {
        let json = r#"{"architectures":["MistralForCausalLM"],"model_type":"mistral","hidden_size":4096,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,"vocab_size":32000}"#;
        let config = UnifiedConfig::from_json(json).unwrap();
        assert_eq!(config.arch, Architecture::Mistral);
        assert_eq!(config.ffn_type, FfnType::SwiGLU);
    }
}
