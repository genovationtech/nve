//! Llama model implementation — full native Rust forward pass.
//!
//! Loads weights from safetensors and runs inference without PyTorch.
//! All projection weights stored as CompactTensor (bf16) for 2x memory reduction.

use std::path::Path;
use std::time::Instant;

use log::info;
use thiserror::Error;

use crate::attention::{gqa_forward_prefill, gqa_forward_single, AttentionWeights, KvCache};
use crate::config::LlamaConfig;
use crate::ops::{self, RopeFreqs};
use crate::quantize::WeightStorage;
use crate::safetensors::{ModelWeights, SafetensorsError};
use crate::tensor::{compact_linear_vec, CompactTensor, Tensor, TensorError};
use crate::tokenizer::TokenizerError;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] SafetensorsError),
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("weight not found: {0}")]
    WeightNotFound(String),
    #[error("model not loaded")]
    NotLoaded,
}

/// Weights for a single transformer block.
/// Projection weights stored as CompactTensor (bf16) for memory efficiency.
/// Norm weights remain f32 (tiny: just [hidden_size] each).
pub struct TransformerBlockWeights {
    // Attention projections (CompactTensor — bf16)
    pub q_proj: CompactTensor,
    pub k_proj: CompactTensor,
    pub v_proj: CompactTensor,
    pub o_proj: CompactTensor,
    // FFN projections (CompactTensor — bf16)
    pub gate_proj: CompactTensor,
    pub up_proj: CompactTensor,
    pub down_proj: CompactTensor,
    // Norms (f32 — small, ~12 KB each)
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
}

/// All model weights.
pub struct LlamaWeights {
    /// Embedding stored in native dtype (bf16) for memory efficiency.
    pub embed_tokens: CompactTensor,
    pub layers: Vec<TransformerBlockWeights>,
    pub norm: Tensor,
    pub lm_head: Option<CompactTensor>,
}

impl LlamaWeights {
    /// Load all weights from safetensors files.
    /// Projection weights loaded as CompactTensor (bf16), norms as f32 Tensor.
    pub fn load(weights: &ModelWeights, config: &LlamaConfig) -> Result<Self, ModelError> {
        let start = Instant::now();

        let embed_tokens = weights
            .load_compact("model.embed_tokens.weight")
            .map_err(|e| ModelError::WeightNotFound(format!("embed_tokens: {}", e)))?;

        info!(
            "Loaded embedding: {:?} ({:.1} MB, {} dtype)",
            embed_tokens.shape(),
            embed_tokens.size_bytes() as f64 / 1024.0 / 1024.0,
            match embed_tokens.dtype() {
                crate::tensor::DType::BF16 => "bf16",
                crate::tensor::DType::F16 => "f16",
                crate::tensor::DType::F32 => "f32",
            }
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let block = TransformerBlockWeights {
                q_proj: Self::load_compact(weights, &format!("{}.self_attn.q_proj.weight", prefix))?,
                k_proj: Self::load_compact(weights, &format!("{}.self_attn.k_proj.weight", prefix))?,
                v_proj: Self::load_compact(weights, &format!("{}.self_attn.v_proj.weight", prefix))?,
                o_proj: Self::load_compact(weights, &format!("{}.self_attn.o_proj.weight", prefix))?,
                gate_proj: Self::load_compact(weights, &format!("{}.mlp.gate_proj.weight", prefix))?,
                up_proj: Self::load_compact(weights, &format!("{}.mlp.up_proj.weight", prefix))?,
                down_proj: Self::load_compact(weights, &format!("{}.mlp.down_proj.weight", prefix))?,
                input_layernorm: Self::load_weight(weights, &format!("{}.input_layernorm.weight", prefix))?,
                post_attention_layernorm: Self::load_weight(weights, &format!("{}.post_attention_layernorm.weight", prefix))?,
            };
            layers.push(block);
        }

        info!("Loaded {} transformer layers (compact bf16)", layers.len());

        let norm = Self::load_weight(weights, "model.norm.weight")?;

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(Self::load_compact(weights, "lm_head.weight")?)
        };

        let elapsed = start.elapsed();
        let total_bytes = embed_tokens.size_bytes()
            + layers.iter().map(|l| Self::layer_size_bytes(l)).sum::<usize>()
            + norm.numel() * 4
            + lm_head.as_ref().map(|t| t.size_bytes()).unwrap_or(0);

        info!(
            "All weights loaded in {:.2}s ({} layers, {:.0} MB total)",
            elapsed.as_secs_f64(),
            config.num_hidden_layers,
            total_bytes as f64 / 1024.0 / 1024.0
        );

        Ok(LlamaWeights {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn load_weight(weights: &ModelWeights, name: &str) -> Result<Tensor, ModelError> {
        weights
            .load_tensor(name)
            .map_err(|e| ModelError::WeightNotFound(format!("{}: {}", name, e)))
    }

    fn load_compact(weights: &ModelWeights, name: &str) -> Result<CompactTensor, ModelError> {
        weights
            .load_compact(name)
            .map_err(|e| ModelError::WeightNotFound(format!("{}: {}", name, e)))
    }

    fn layer_size_bytes(layer: &TransformerBlockWeights) -> usize {
        layer.q_proj.size_bytes()
            + layer.k_proj.size_bytes()
            + layer.v_proj.size_bytes()
            + layer.o_proj.size_bytes()
            + layer.gate_proj.size_bytes()
            + layer.up_proj.size_bytes()
            + layer.down_proj.size_bytes()
            + (layer.input_layernorm.numel() + layer.post_attention_layernorm.numel()) * 4
    }

    pub fn embed(&self, token_id: u32) -> Tensor {
        let data = self.embed_tokens.row_to_f32(token_id as usize);
        let dim = self.embed_tokens.shape()[1];
        Tensor::new(data, vec![dim]).unwrap()
    }

    /// Compute logits: hidden_state → vocabulary scores.
    pub fn compute_logits(&self, hidden: &Tensor) -> Tensor {
        if let Some(ref lm_head) = self.lm_head {
            return compact_linear_vec(hidden, lm_head);
        }
        let logits_data = self.embed_tokens.matvec_f32(hidden.data());
        let vocab_size = self.embed_tokens.shape()[0];
        Tensor::new(logits_data, vec![vocab_size]).unwrap()
    }
}

/// The Llama model — complete inference engine.
pub struct LlamaModel {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub rope_freqs: RopeFreqs,
    pub kv_cache: KvCache,
    pos: usize,
}

impl LlamaModel {
    pub fn from_dir(model_dir: &Path) -> Result<Self, ModelError> {
        info!("Loading model from {:?}", model_dir);

        let config = LlamaConfig::from_model_dir(model_dir)?;
        info!(
            "Config: {}d, {}L, {}H ({}KV), {}ff, {}V",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.intermediate_size,
            config.vocab_size,
        );

        let safetensors = ModelWeights::load(model_dir)?;
        info!("Safetensors: {} tensors", safetensors.len());

        let weights = LlamaWeights::load(&safetensors, &config)?;

        let rope_freqs = RopeFreqs::new(
            config.head_dim(),
            config.max_position_embeddings.min(8192),
            config.rope_theta,
            config.rope_scaling.as_ref(),
        );

        let kv_cache = KvCache::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim(),
        );

        Ok(LlamaModel {
            config,
            weights,
            rope_freqs,
            kv_cache,
            pos: 0,
        })
    }

    pub fn new(config: LlamaConfig, weights: LlamaWeights) -> Self {
        let rope_freqs = RopeFreqs::new(
            config.head_dim(),
            config.max_position_embeddings.min(8192),
            config.rope_theta,
            config.rope_scaling.as_ref(),
        );

        let kv_cache = KvCache::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim(),
        );

        LlamaModel {
            config,
            weights,
            rope_freqs,
            kv_cache,
            pos: 0,
        }
    }

    pub fn forward_single(&mut self, token_id: u32) -> Tensor {
        let pos = self.pos;
        let eps = self.config.rms_norm_eps;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        let mut hidden = self.weights.embed(token_id);

        for i in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[i];

            let normed = ops::rms_norm(&hidden, &layer.input_layernorm, eps);

            // Wrap CompactTensor references in WeightStorage for the generic attention API.
            let q_ws = WeightStorage::Compact(layer.q_proj.clone());
            let k_ws = WeightStorage::Compact(layer.k_proj.clone());
            let v_ws = WeightStorage::Compact(layer.v_proj.clone());
            let o_ws = WeightStorage::Compact(layer.o_proj.clone());

            let attn_weights = AttentionWeights {
                q_proj: &q_ws,
                k_proj: &k_ws,
                v_proj: &v_ws,
                o_proj: &o_ws,
                q_bias: None,
                k_bias: None,
                v_bias: None,
            };

            let attn_out = gqa_forward_single(
                &normed, &attn_weights, Some(&self.rope_freqs),
                &mut self.kv_cache, i, pos, n_heads, n_kv, head_dim,
            );

            let mut h = hidden.add(&attn_out);
            let normed2 = ops::rms_norm(&h, &layer.post_attention_layernorm, eps);
            let ffn_out = ops::compact_swiglu_ffn(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj);
            h.add_inplace(&ffn_out);
            hidden = h;
        }

        hidden = ops::rms_norm(&hidden, &self.weights.norm, eps);
        let logits = self.weights.compute_logits(&hidden);

        self.pos += 1;
        self.kv_cache.advance();

        logits
    }

    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let start_pos = self.pos;
        let eps = self.config.rms_norm_eps;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        let mut embed_data = Vec::with_capacity(seq_len * hidden_size);
        for &tid in token_ids {
            let emb = self.weights.embed(tid);
            embed_data.extend_from_slice(emb.data());
        }
        let mut hidden = Tensor::new(embed_data, vec![seq_len, hidden_size]).unwrap();

        for i in 0..self.weights.layers.len() {
            let layer = &self.weights.layers[i];

            let normed = ops::rms_norm_batch(&hidden, &layer.input_layernorm, eps);

            let q_ws = WeightStorage::Compact(layer.q_proj.clone());
            let k_ws = WeightStorage::Compact(layer.k_proj.clone());
            let v_ws = WeightStorage::Compact(layer.v_proj.clone());
            let o_ws = WeightStorage::Compact(layer.o_proj.clone());

            let attn_weights = AttentionWeights {
                q_proj: &q_ws,
                k_proj: &k_ws,
                v_proj: &v_ws,
                o_proj: &o_ws,
                q_bias: None,
                k_bias: None,
                v_bias: None,
            };

            let attn_out = gqa_forward_prefill(
                &normed, &attn_weights, Some(&self.rope_freqs),
                &mut self.kv_cache, i, start_pos, n_heads, n_kv, head_dim,
            );

            let mut h = hidden.add(&attn_out);
            let normed2 = ops::rms_norm_batch(&h, &layer.post_attention_layernorm, eps);
            let ffn_out = ops::compact_swiglu_ffn(&normed2, &layer.gate_proj, &layer.up_proj, &layer.down_proj);
            h.add_inplace(&ffn_out);
            hidden = h;
        }

        hidden = ops::rms_norm_batch(&hidden, &self.weights.norm, eps);
        let last_hidden = hidden.slice_rows(seq_len - 1, seq_len);
        let last_vec = Tensor::new(last_hidden.data().to_vec(), vec![hidden_size]).unwrap();
        let logits = self.weights.compute_logits(&last_vec);

        self.pos += seq_len;
        for _ in 0..seq_len {
            self.kv_cache.advance();
        }

        logits
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> GenerationResult {
        let start = Instant::now();
        let mut generated = Vec::new();
        let eos_ids = vec![128001u32, 128009];

        let prefill_start = Instant::now();
        let mut logits = if prompt_tokens.len() > 1 {
            self.forward_prefill(prompt_tokens)
        } else {
            self.forward_single(prompt_tokens[0])
        };
        let prefill_time = prefill_start.elapsed();

        let first_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
        generated.push(first_token);

        if eos_ids.contains(&first_token) {
            return GenerationResult {
                tokens: generated,
                prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
                decode_time_ms: 0.0,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                tokens_per_sec: 0.0,
                prompt_tokens: prompt_tokens.len(),
            };
        }

        let decode_start = Instant::now();
        let mut next_token = first_token;

        for _ in 1..max_new_tokens {
            logits = self.forward_single(next_token);
            next_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
            generated.push(next_token);

            if eos_ids.contains(&next_token) {
                break;
            }
        }

        let decode_time = decode_start.elapsed();
        let total_time = start.elapsed();
        let decode_tokens = generated.len().saturating_sub(1);
        let tokens_per_sec = if decode_time.as_secs_f64() > 0.0 {
            decode_tokens as f64 / decode_time.as_secs_f64()
        } else {
            0.0
        };

        GenerationResult {
            tokens: generated,
            prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
            decode_time_ms: decode_time.as_secs_f64() * 1000.0,
            total_time_ms: total_time.as_secs_f64() * 1000.0,
            tokens_per_sec,
            prompt_tokens: prompt_tokens.len(),
        }
    }

    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.pos = 0;
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }
}

#[derive(Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
}

impl GenerationResult {
    pub fn display(&self) {
        println!("  Prompt tokens:    {}", self.prompt_tokens);
        println!("  Generated tokens: {}", self.tokens.len());
        println!("  Prefill time:     {:.1} ms", self.prefill_time_ms);
        println!("  Decode time:      {:.1} ms", self.decode_time_ms);
        println!("  Total time:       {:.1} ms", self.total_time_ms);
        println!("  Decode speed:     {:.1} tok/s", self.tokens_per_sec);
    }
}
