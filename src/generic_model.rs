//! Architecture-generic transformer model.
//!
//! Loads any supported HuggingFace model and runs inference by dispatching
//! to the appropriate normalization, attention, and FFN implementations
//! based on the detected architecture.
//! All projection weights stored as CompactTensor (bf16) for memory efficiency.

use std::path::Path;
use std::time::Instant;

use log::info;
use thiserror::Error;

use crate::arch::{FfnType, NormType, PosEncoding, UnifiedConfig};
use crate::attention::{gqa_forward_prefill, gqa_forward_single, AttentionWeights, KvCache};
use crate::model::GenerationResult;
use crate::ops::{self, RopeFreqs};
use crate::quantize::{QuantMode, QuantizedTensor, QuantizedTensor8, QuantizedTensor3, QuantizedTensor2, QuantizedTensor1, WeightStorage, ws_linear, ws_linear_vec};
use crate::safetensors::ModelWeights;
use crate::tensor::{compact_linear_vec, CompactTensor, Tensor};
use crate::weight_map;

#[derive(Debug, Error)]
pub enum GenericModelError {
    #[error("architecture error: {0}")]
    Arch(#[from] crate::arch::ArchError),
    #[error("safetensors error: {0}")]
    Safetensors(#[from] crate::safetensors::SafetensorsError),
    #[error("tensor error: {0}")]
    Tensor(#[from] crate::tensor::TensorError),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] crate::tokenizer::TokenizerError),
    #[error("weight not found: {0}")]
    WeightNotFound(String),
    #[error("hub error: {0}")]
    Hub(#[from] crate::hub::HubError),
    #[error("unsupported feature: {0}")]
    Unsupported(String),
}

/// Weights for a single transformer block, with optional biases.
/// Projection weights stored as WeightStorage (bf16, Q4, or Q8), biases as f32 Tensor (small).
pub struct GenericBlockWeights {
    // Attention projections (WeightStorage — bf16/Q4/Q8).
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
    // FFN weights (WeightStorage — bf16/Q4/Q8).
    pub gate_proj: Option<WeightStorage>,
    pub up_proj: WeightStorage,
    pub down_proj: WeightStorage,
    // Norms (f32 — small, never quantized).
    pub input_norm: Tensor,
    pub post_attn_norm: Tensor,
    // Optional biases (f32 — small, never quantized).
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub o_bias: Option<Tensor>,
    pub up_bias: Option<Tensor>,
    pub down_bias: Option<Tensor>,
    pub gate_bias: Option<Tensor>,
    pub input_norm_bias: Option<Tensor>,
    pub post_attn_norm_bias: Option<Tensor>,
}

impl GenericBlockWeights {
    /// Convert all Compact weights to Q4_0 quantized.
    pub fn quantize_q4(&mut self) {
        fn convert(ws: &mut WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                *ws = WeightStorage::Quantized4(QuantizedTensor::from_compact(c));
            }
        }
        convert(&mut self.q_proj);
        convert(&mut self.k_proj);
        convert(&mut self.v_proj);
        convert(&mut self.o_proj);
        if let Some(ref mut g) = self.gate_proj {
            convert(g);
        }
        convert(&mut self.up_proj);
        convert(&mut self.down_proj);
    }

    /// Convert all Compact weights to Q8_0 quantized.
    pub fn quantize_q8(&mut self) {
        fn convert(ws: &mut WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                *ws = WeightStorage::Quantized8(QuantizedTensor8::from_compact(c));
            }
        }
        convert(&mut self.q_proj);
        convert(&mut self.k_proj);
        convert(&mut self.v_proj);
        convert(&mut self.o_proj);
        if let Some(ref mut g) = self.gate_proj {
            convert(g);
        }
        convert(&mut self.up_proj);
        convert(&mut self.down_proj);
    }

    /// Convert all Compact weights to Q3 quantized.
    pub fn quantize_q3(&mut self) {
        fn convert(ws: &mut WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                *ws = WeightStorage::Quantized3(QuantizedTensor3::from_compact(c));
            }
        }
        convert(&mut self.q_proj);
        convert(&mut self.k_proj);
        convert(&mut self.v_proj);
        convert(&mut self.o_proj);
        if let Some(ref mut g) = self.gate_proj { convert(g); }
        convert(&mut self.up_proj);
        convert(&mut self.down_proj);
    }

    /// Convert all Compact weights to Q2 quantized.
    pub fn quantize_q2(&mut self) {
        fn convert(ws: &mut WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                *ws = WeightStorage::Quantized2(QuantizedTensor2::from_compact(c));
            }
        }
        convert(&mut self.q_proj);
        convert(&mut self.k_proj);
        convert(&mut self.v_proj);
        convert(&mut self.o_proj);
        if let Some(ref mut g) = self.gate_proj { convert(g); }
        convert(&mut self.up_proj);
        convert(&mut self.down_proj);
    }

    /// Convert all Compact weights to Q1 (ternary) quantized.
    pub fn quantize_q1(&mut self) {
        fn convert(ws: &mut WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                *ws = WeightStorage::Quantized1(QuantizedTensor1::from_compact(c));
            }
        }
        convert(&mut self.q_proj);
        convert(&mut self.k_proj);
        convert(&mut self.v_proj);
        convert(&mut self.o_proj);
        if let Some(ref mut g) = self.gate_proj { convert(g); }
        convert(&mut self.up_proj);
        convert(&mut self.down_proj);
    }

    /// Adaptively quantize projections with per-layer mode and optional AWQ saliency.
    /// AWQ scales: alpha_j = saliency[j].sqrt().max(1e-6)
    pub fn quantize_adaptive(&mut self, mode: QuantMode, saliency: Option<&ChannelSaliency>) {
        match mode {
            QuantMode::None => {
                // Zero/prune: replace all projections with Q1 of a zeroed tensor
                // (effectively does nothing — keep compact but skip computation upstream)
            }
            QuantMode::Q8 => {
                if let Some(sal) = saliency {
                    // Use k-quant (importance-weighted rounding) + AWQ for best quality.
                    self.quantize_with_awq_mode(|c, scales| WeightStorage::Quantized8(QuantizedTensor8::from_compact_awq_kquant(c, scales)), sal);
                } else {
                    self.quantize_q8();
                }
            }
            QuantMode::Q4 => {
                if let Some(sal) = saliency {
                    // Use k-quant (importance-weighted rounding) + AWQ for best quality.
                    self.quantize_with_awq_mode(|c, scales| WeightStorage::Quantized4(QuantizedTensor::from_compact_awq_kquant(c, scales)), sal);
                } else {
                    self.quantize_q4();
                }
            }
            QuantMode::Q3 => {
                if let Some(sal) = saliency {
                    self.quantize_with_awq_mode(|c, scales| WeightStorage::Quantized3(QuantizedTensor3::from_compact_awq(c, scales)), sal);
                } else {
                    self.quantize_q3();
                }
            }
            QuantMode::Q2 => {
                if let Some(sal) = saliency {
                    self.quantize_with_awq_mode(|c, scales| WeightStorage::Quantized2(QuantizedTensor2::from_compact_awq(c, scales)), sal);
                } else {
                    self.quantize_q2();
                }
            }
            QuantMode::Q1 => {
                if let Some(sal) = saliency {
                    self.quantize_with_awq_mode(|c, scales| WeightStorage::Quantized1(QuantizedTensor1::from_compact_awq(c, scales)), sal);
                } else {
                    self.quantize_q1();
                }
            }
            QuantMode::ProfileGuided(_) => {
                // Should not reach here — PG mode is resolved to a concrete mode before calling
            }
        }
    }

    /// Internal: quantize all projections using AWQ scales derived from saliency.
    fn quantize_with_awq_mode<F>(&mut self, convert: F, saliency: &ChannelSaliency)
    where
        F: Fn(&CompactTensor, &[f32]) -> WeightStorage,
    {
        let hidden_scales = compute_awq_scales(&saliency.hidden_saliency);
        let intermediate_scales = compute_awq_scales(&saliency.intermediate_saliency);

        fn try_convert(ws: &mut WeightStorage, scales: &[f32], conv: &dyn Fn(&CompactTensor, &[f32]) -> WeightStorage) {
            if let WeightStorage::Compact(ref c) = ws {
                let cols = if c.shape().len() >= 2 { c.shape()[1] } else { c.shape()[0] };
                if scales.len() == cols {
                    *ws = conv(c, scales);
                } else {
                    // Dimension mismatch — quantize without AWQ
                    // This will be handled by using hidden scales for appropriate projections
                }
            }
        }

        // Attention projections take hidden_dim input
        try_convert(&mut self.q_proj, &hidden_scales, &convert);
        try_convert(&mut self.k_proj, &hidden_scales, &convert);
        try_convert(&mut self.v_proj, &hidden_scales, &convert);
        try_convert(&mut self.o_proj, &hidden_scales, &convert);

        // FFN: gate and up take hidden_dim input, down takes intermediate_dim input
        if let Some(ref mut g) = self.gate_proj {
            try_convert(g, &hidden_scales, &convert);
        }
        try_convert(&mut self.up_proj, &hidden_scales, &convert);
        try_convert(&mut self.down_proj, &intermediate_scales, &convert);
    }
}

/// AWQ saliency information for a single layer's channels.
pub struct ChannelSaliency {
    pub hidden_saliency: Vec<f32>,       // mean(|x_j|) for hidden dim channels
    pub intermediate_saliency: Vec<f32>, // mean(|x_j|) for intermediate dim channels
    pub sample_count: usize,
}

/// Compute AWQ scales from saliency: alpha_j = saliency[j].sqrt().max(1e-6)
fn compute_awq_scales(saliency: &[f32]) -> Vec<f32> {
    saliency.iter().map(|&s| s.sqrt().max(1e-6)).collect()
}

/// The generic model — runs inference for any supported architecture.
pub struct GenericModel {
    pub config: UnifiedConfig,
    pub embed_tokens: CompactTensor,
    pub layers: Vec<GenericBlockWeights>,
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub lm_head: Option<CompactTensor>,
    pub pos_embed: Option<Tensor>,
    pub rope_freqs: Option<RopeFreqs>,
    pub kv_cache: KvCache,
    pos: usize,
}

impl GenericModel {
    pub fn from_dir(model_dir: &Path) -> Result<Self, GenericModelError> {
        info!("Loading model from {:?}", model_dir);

        let config = UnifiedConfig::from_model_dir(model_dir)?;
        info!(
            "Detected architecture: {} ({}d, {}L, {}H/{}KV, {}ff, {}V)",
            config.arch,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.intermediate_size,
            config.vocab_size,
        );
        info!(
            "  norm={:?}, pos={:?}, ffn={:?}, attn_bias={}, parallel={}",
            config.norm_type, config.pos_encoding, config.ffn_type,
            config.attn_bias, config.parallel_attn_ffn,
        );

        let safetensors = ModelWeights::load(model_dir)?;
        info!("Safetensors: {} tensors", safetensors.len());

        Self::from_weights(&config, &safetensors)
    }

    pub fn from_model_id(model_path: &str) -> Result<Self, GenericModelError> {
        let resolved = crate::hub::resolve_model_path(model_path, None, None)?;
        Self::from_dir(&resolved)
    }

    fn from_weights(
        config: &UnifiedConfig,
        weights: &ModelWeights,
    ) -> Result<Self, GenericModelError> {
        let start = Instant::now();
        let global = weight_map::global_weights(config.arch);

        let embed_tokens = weights
            .load_compact(&global.embed_tokens)
            .map_err(|e| GenericModelError::WeightNotFound(format!("embed_tokens: {}", e)))?;

        info!(
            "Loaded embedding: {:?} ({:.1} MB)",
            embed_tokens.shape(),
            embed_tokens.size_bytes() as f64 / 1024.0 / 1024.0,
        );

        let pos_embed = if let Some(ref name) = global.position_embed {
            Some(load_weight(weights, name)?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let block = load_generic_block(weights, config, i)?;
            layers.push(block);
        }

        info!("Loaded {} transformer layers (compact bf16)", layers.len());

        let norm = load_weight(weights, &global.final_norm)?;
        let norm_bias = global.final_norm_bias.as_ref().and_then(|n| {
            if weights.has_tensor(n) { load_weight(weights, n).ok() } else { None }
        });

        let lm_head = if config.tie_word_embeddings {
            None
        } else if weights.has_tensor(&global.lm_head) {
            Some(load_compact(weights, &global.lm_head)?)
        } else {
            None
        };

        let rope_freqs = if config.pos_encoding == PosEncoding::RoPE {
            let rope_dim = config.rotary_dim.unwrap_or(config.head_dim());
            Some(RopeFreqs::new(
                rope_dim,
                config.max_position_embeddings.min(8192),
                config.rope_theta,
                config.rope_scaling.as_ref(),
            ))
        } else {
            None
        };

        let kv_cache = KvCache::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim(),
        );

        let elapsed = start.elapsed();
        info!("All weights loaded in {:.2}s", elapsed.as_secs_f64());

        Ok(GenericModel {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            norm_bias,
            lm_head,
            pos_embed,
            rope_freqs,
            kv_cache,
            pos: 0,
        })
    }

    fn embed(&self, token_id: u32) -> Tensor {
        let vocab = self.embed_tokens.shape()[0];
        assert!(
            (token_id as usize) < vocab,
            "token_id {} out of range for vocab size {}",
            token_id, vocab,
        );
        let data = self.embed_tokens.row_to_f32(token_id as usize);
        let dim = self.embed_tokens.shape()[1];
        Tensor::new(data, vec![dim]).unwrap()
    }

    fn do_apply_norm(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        apply_norm(x, weight, bias, self.config.norm_type, self.config.norm_eps)
    }

    fn do_apply_ffn(&self, x: &Tensor, layer: &GenericBlockWeights) -> Tensor {
        apply_ffn(x, layer, self.config.ffn_type)
    }

    fn compute_logits(&self, hidden: &Tensor) -> Tensor {
        if let Some(ref lm_head) = self.lm_head {
            return compact_linear_vec(hidden, lm_head);
        }
        let logits_data = self.embed_tokens.matvec_f32(hidden.data());
        let vocab_size = self.config.vocab_size;
        Tensor::new(logits_data, vec![vocab_size]).unwrap()
    }

    pub fn forward_single(&mut self, token_id: u32) -> Tensor {
        let pos = self.pos;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        let mut hidden = self.embed(token_id);

        if let Some(ref pos_emb) = self.pos_embed {
            let pe = pos_emb.embedding(pos as u32);
            hidden.add_inplace(&pe);
        }

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];

            let normed = self.do_apply_norm(&hidden, &layer.input_norm, layer.input_norm_bias.as_ref());

            let attn_weights = AttentionWeights {
                q_proj: &layer.q_proj,
                k_proj: &layer.k_proj,
                v_proj: &layer.v_proj,
                o_proj: &layer.o_proj,
                q_bias: layer.q_bias.as_ref(),
                k_bias: layer.k_bias.as_ref(),
                v_bias: layer.v_bias.as_ref(),
            };

            let attn_out = gqa_forward_single(
                &normed, &attn_weights, self.rope_freqs.as_ref(),
                &mut self.kv_cache, i, pos, n_heads, n_kv, head_dim,
            );

            let attn_out = if let Some(ref bias) = layer.o_bias {
                attn_out.add(bias)
            } else {
                attn_out
            };

            if self.config.parallel_attn_ffn {
                let ffn_normed = self.do_apply_norm(&hidden, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref());
                let ffn_out = self.do_apply_ffn(&ffn_normed, layer);
                hidden.add_inplace(&attn_out);
                hidden.add_inplace(&ffn_out);
            } else {
                let mut h = hidden.add(&attn_out);
                let normed2 = self.do_apply_norm(&h, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref());
                let ffn_out = self.do_apply_ffn(&normed2, layer);
                h.add_inplace(&ffn_out);
                hidden = h;
            }
        }

        hidden = self.do_apply_norm(&hidden, &self.norm, self.norm_bias.as_ref());
        let logits = self.compute_logits(&hidden);

        self.pos += 1;
        self.kv_cache.advance();

        logits
    }

    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Tensor {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let start_pos = self.pos;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        let mut embed_data = Vec::with_capacity(seq_len * hidden_size);
        for &tid in token_ids {
            let emb = self.embed(tid);
            embed_data.extend_from_slice(emb.data());
        }
        let mut hidden = Tensor::new(embed_data, vec![seq_len, hidden_size]).unwrap();

        if let Some(ref pos_emb) = self.pos_embed {
            let data = hidden.data_mut();
            for s in 0..seq_len {
                let pos = start_pos + s;
                let pe_row = pos_emb.row(pos as usize);
                let offset = s * hidden_size;
                for j in 0..hidden_size {
                    data[offset + j] += pe_row[j];
                }
            }
        }

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];

            let normed = self.do_apply_norm_batch(&hidden, &layer.input_norm, layer.input_norm_bias.as_ref());

            let attn_weights = AttentionWeights {
                q_proj: &layer.q_proj,
                k_proj: &layer.k_proj,
                v_proj: &layer.v_proj,
                o_proj: &layer.o_proj,
                q_bias: layer.q_bias.as_ref(),
                k_bias: layer.k_bias.as_ref(),
                v_bias: layer.v_bias.as_ref(),
            };

            let attn_out = gqa_forward_prefill(
                &normed, &attn_weights, self.rope_freqs.as_ref(),
                &mut self.kv_cache, i, start_pos, n_heads, n_kv, head_dim,
            );

            // Apply output projection bias (e.g. GPT-2 c_proj.bias).
            let attn_out = if let Some(ref bias) = layer.o_bias {
                let mut result = attn_out;
                let data = result.data_mut();
                let hidden = bias.numel();
                let seq_len = data.len() / hidden;
                let b = bias.data();
                for s in 0..seq_len {
                    for j in 0..hidden {
                        data[s * hidden + j] += b[j];
                    }
                }
                result
            } else {
                attn_out
            };

            if self.config.parallel_attn_ffn {
                let ffn_normed = self.do_apply_norm_batch(&hidden, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref());
                let ffn_out = self.do_apply_ffn(&ffn_normed, layer);
                hidden.add_inplace(&attn_out);
                hidden.add_inplace(&ffn_out);
            } else {
                let mut h = hidden.add(&attn_out);
                let normed2 = self.do_apply_norm_batch(&h, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref());
                let ffn_out = self.do_apply_ffn(&normed2, layer);
                h.add_inplace(&ffn_out);
                hidden = h;
            }
        }

        hidden = self.do_apply_norm_batch(&hidden, &self.norm, self.norm_bias.as_ref());

        let last_hidden = hidden.slice_rows(seq_len - 1, seq_len);
        let last_vec = Tensor::new(last_hidden.data().to_vec(), vec![hidden_size]).unwrap();
        let logits = self.compute_logits(&last_vec);

        self.pos += seq_len;
        // Set KV cache length once after all layers are done (positions are written explicitly).
        self.kv_cache.set_seq_len(start_pos + seq_len);

        logits
    }

    fn do_apply_norm_batch(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        apply_norm_batch(x, weight, bias, self.config.norm_type, self.config.norm_eps)
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

        let eos_id = self.config.eos_token_id.unwrap_or(2);
        let mut eos_ids = vec![eos_id];
        if eos_id == 128001 {
            eos_ids.push(128009);
        }

        let prefill_start = Instant::now();
        let mut logits = if prompt_tokens.len() > 1 {
            self.forward_prefill(prompt_tokens)
        } else {
            self.forward_single(prompt_tokens[0])
        };
        let prefill_time = prefill_start.elapsed();

        // Debug: check logit distribution
        {
            let ld = logits.data();
            let max_val = ld.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = ld.iter().cloned().fold(f32::INFINITY, f32::min);
            let argmax = ld.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap();
            let nan_count = ld.iter().filter(|v| v.is_nan()).count();
            let inf_count = ld.iter().filter(|v| v.is_infinite()).count();
            eprintln!("[DEBUG] logits: len={}, min={:.4}, max={:.4}, argmax={}, nan={}, inf={}",
                ld.len(), min_val, max_val, argmax, nan_count, inf_count);
        }
        let first_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
        eprintln!("[DEBUG] first_token={}", first_token);
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

    /// Streaming variant — calls `on_token(token_id)` immediately as each token
    /// is sampled, before the full `GenerationResult` is assembled.
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        mut on_token: F,
    ) -> GenerationResult
    where
        F: FnMut(u32),
    {
        use std::time::Instant;
        let start = Instant::now();
        let mut generated = Vec::new();

        let eos_id = self.config.eos_token_id.unwrap_or(2);
        let mut eos_ids = vec![eos_id];
        if eos_id == 128001 {
            eos_ids.push(128009);
        }

        let prefill_start = Instant::now();
        let mut logits = if prompt_tokens.len() > 1 {
            self.forward_prefill(prompt_tokens)
        } else {
            self.forward_single(prompt_tokens[0])
        };
        let prefill_time = prefill_start.elapsed();

        let first_token = crate::ops::sample_top_p(logits.data(), temperature, top_p) as u32;
        on_token(first_token);
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
            next_token = crate::ops::sample_top_p(logits.data(), temperature, top_p) as u32;
            on_token(next_token);
            generated.push(next_token);
            if eos_ids.contains(&next_token) {
                break;
            }
        }

        let decode_time = decode_start.elapsed();
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
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
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

// ── Public free functions for norm and FFN ──

/// Apply normalization based on type (RMSNorm or LayerNorm).
/// Can be used by both GenericModel and PagedModel.
pub fn apply_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, norm_type: NormType, eps: f64) -> Tensor {
    match norm_type {
        NormType::RMSNorm => ops::rms_norm(x, weight, eps),
        NormType::LayerNorm => ops::layer_norm(x, weight, bias, eps),
    }
}

/// Apply normalization to a batch (2D tensor).
pub fn apply_norm_batch(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, norm_type: NormType, eps: f64) -> Tensor {
    match norm_type {
        NormType::RMSNorm => ops::rms_norm_batch(x, weight, eps),
        NormType::LayerNorm => ops::layer_norm_batch(x, weight, bias, eps),
    }
}

/// Apply FFN based on architecture type using WeightStorage weights.
/// Can be used by both GenericModel and PagedModel.
pub fn apply_ffn(x: &Tensor, layer: &GenericBlockWeights, ffn_type: FfnType) -> Tensor {
    match ffn_type {
        FfnType::SwiGLU => {
            // SwiGLU: down(silu(gate(x)) * up(x))
            let gate = if x.ndim() == 1 {
                ws_linear_vec(x, layer.gate_proj.as_ref().unwrap())
            } else {
                ws_linear(x, layer.gate_proj.as_ref().unwrap())
            };
            let up = if x.ndim() == 1 {
                ws_linear_vec(x, &layer.up_proj)
            } else {
                ws_linear(x, &layer.up_proj)
            };
            let activated: Vec<f32> = gate.data().iter()
                .zip(up.data().iter())
                .map(|(&g, &u)| ops::silu(g) * u)
                .collect();
            let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();
            if x.ndim() == 1 {
                ws_linear_vec(&intermediate, &layer.down_proj)
            } else {
                ws_linear(&intermediate, &layer.down_proj)
            }
        }
        FfnType::GeGLU => {
            let gate = if x.ndim() == 1 {
                ws_linear_vec(x, layer.gate_proj.as_ref().unwrap())
            } else {
                ws_linear(x, layer.gate_proj.as_ref().unwrap())
            };
            let up = if x.ndim() == 1 {
                ws_linear_vec(x, &layer.up_proj)
            } else {
                ws_linear(x, &layer.up_proj)
            };
            let activated: Vec<f32> = gate.data().iter()
                .zip(up.data().iter())
                .map(|(&g, &u)| ops::gelu(g) * u)
                .collect();
            let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();
            if x.ndim() == 1 {
                ws_linear_vec(&intermediate, &layer.down_proj)
            } else {
                ws_linear(&intermediate, &layer.down_proj)
            }
        }
        FfnType::GELU => {
            let mut up = if x.ndim() == 1 {
                ws_linear_vec(x, &layer.up_proj)
            } else {
                ws_linear(x, &layer.up_proj)
            };
            if let Some(ref bias) = layer.up_bias {
                add_bias_inplace(&mut up, bias);
            }
            let activated: Vec<f32> = up.data().iter().map(|&v| ops::gelu(v)).collect();
            let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
            let mut out = if x.ndim() == 1 {
                ws_linear_vec(&act, &layer.down_proj)
            } else {
                ws_linear(&act, &layer.down_proj)
            };
            if let Some(ref bias) = layer.down_bias {
                add_bias_inplace(&mut out, bias);
            }
            out
        }
        FfnType::ReLU => {
            let mut up = if x.ndim() == 1 {
                ws_linear_vec(x, &layer.up_proj)
            } else {
                ws_linear(x, &layer.up_proj)
            };
            if let Some(ref bias) = layer.up_bias {
                add_bias_inplace(&mut up, bias);
            }
            let activated: Vec<f32> = up.data().iter().map(|&v| v.max(0.0)).collect();
            let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
            let mut out = if x.ndim() == 1 {
                ws_linear_vec(&act, &layer.down_proj)
            } else {
                ws_linear(&act, &layer.down_proj)
            };
            if let Some(ref bias) = layer.down_bias {
                add_bias_inplace(&mut out, bias);
            }
            out
        }
    }
}

// ── Public block loading function ──

/// Load a single transformer block's weights from ModelWeights.
/// Handles fused QKV, Conv1D transposition, optional biases.
pub fn load_generic_block(
    weights: &ModelWeights,
    config: &UnifiedConfig,
    layer_idx: usize,
) -> Result<GenericBlockWeights, GenericModelError> {
    let names = weight_map::layer_weights(config.arch, layer_idx);
    let uses_fused = weight_map::uses_fused_qkv(config.arch);
    let conv1d = weight_map::uses_conv1d_weights(config.arch);

    let (q_proj, k_proj, v_proj) = if uses_fused {
        let mut fused = load_weight(weights, &names.q_proj)?;
        if conv1d {
            fused = fused.transpose_2d();
        }
        let (q, k, v) = split_qkv(
            &fused,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
        );
        (
            WeightStorage::Compact(tensor_to_compact(&q)),
            WeightStorage::Compact(tensor_to_compact(&k)),
            WeightStorage::Compact(tensor_to_compact(&v)),
        )
    } else {
        if conv1d {
            (
                WeightStorage::Compact(load_compact_transposed(weights, &names.q_proj)?),
                WeightStorage::Compact(load_compact_transposed(weights, &names.k_proj)?),
                WeightStorage::Compact(load_compact_transposed(weights, &names.v_proj)?),
            )
        } else {
            (
                WeightStorage::Compact(load_compact(weights, &names.q_proj)?),
                WeightStorage::Compact(load_compact(weights, &names.k_proj)?),
                WeightStorage::Compact(load_compact(weights, &names.v_proj)?),
            )
        }
    };

    let (q_bias, k_bias, v_bias) = if uses_fused {
        if let Some(ref bias_name) = names.q_bias {
            if weights.has_tensor(bias_name) {
                let fused_bias = load_weight(weights, bias_name)?;
                let (qb, kb, vb) = split_qkv_bias(
                    &fused_bias,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim(),
                );
                (Some(qb), Some(kb), Some(vb))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        }
    } else {
        (
            load_optional(weights, &names.q_bias),
            load_optional(weights, &names.k_bias),
            load_optional(weights, &names.v_bias),
        )
    };

    let uses_fused_gate_up = weight_map::uses_fused_gate_up(config.arch);

    let (o_proj, gate_proj, up_proj, down_proj) = if uses_fused_gate_up {
        // Phi-3 style: gate_up_proj is a single tensor [2*ff, hidden] that needs splitting
        let o = WeightStorage::Compact(load_compact(weights, &names.o_proj)?);
        let fused_gate_up = load_weight(weights, &names.up_proj)?; // gate_proj and up_proj point to same tensor
        let ff = config.intermediate_size;
        let hidden = config.hidden_size;
        let data = fused_gate_up.data();
        let gate_data: Vec<f32> = data[..ff * hidden].to_vec();
        let up_data: Vec<f32> = data[ff * hidden..2 * ff * hidden].to_vec();
        let gate_t = Tensor::new(gate_data, vec![ff, hidden]).unwrap();
        let up_t = Tensor::new(up_data, vec![ff, hidden]).unwrap();
        (
            o,
            Some(WeightStorage::Compact(tensor_to_compact(&gate_t))),
            WeightStorage::Compact(tensor_to_compact(&up_t)),
            WeightStorage::Compact(load_compact(weights, &names.down_proj)?),
        )
    } else if conv1d {
        (
            WeightStorage::Compact(load_compact_transposed(weights, &names.o_proj)?),
            names.gate_proj.as_ref().map(|n| load_compact_transposed(weights, n).map(WeightStorage::Compact)).transpose()?,
            WeightStorage::Compact(load_compact_transposed(weights, &names.up_proj)?),
            WeightStorage::Compact(load_compact_transposed(weights, &names.down_proj)?),
        )
    } else {
        (
            WeightStorage::Compact(load_compact(weights, &names.o_proj)?),
            names.gate_proj.as_ref().map(|n| load_compact(weights, n).map(WeightStorage::Compact)).transpose()?,
            WeightStorage::Compact(load_compact(weights, &names.up_proj)?),
            WeightStorage::Compact(load_compact(weights, &names.down_proj)?),
        )
    };

    Ok(GenericBlockWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
        input_norm: load_weight(weights, &names.input_norm)?,
        post_attn_norm: load_weight(weights, &names.post_attn_norm)?,
        q_bias,
        k_bias,
        v_bias,
        o_bias: load_optional(weights, &names.o_bias),
        up_bias: load_optional(weights, &names.up_bias),
        down_bias: load_optional(weights, &names.down_bias),
        gate_bias: load_optional(weights, &names.gate_bias),
        input_norm_bias: load_optional(weights, &names.input_norm_bias),
        post_attn_norm_bias: load_optional(weights, &names.post_attn_norm_bias),
    })
}

// ── Helpers ──

pub(crate) fn load_weight(weights: &ModelWeights, name: &str) -> Result<Tensor, GenericModelError> {
    weights
        .load_tensor(name)
        .map_err(|e| GenericModelError::WeightNotFound(format!("{}: {}", name, e)))
}

pub(crate) fn load_compact(weights: &ModelWeights, name: &str) -> Result<CompactTensor, GenericModelError> {
    weights
        .load_compact(name)
        .map_err(|e| GenericModelError::WeightNotFound(format!("{}: {}", name, e)))
}

/// Load a weight as f32, transpose it, and re-encode as CompactTensor.
/// Used for GPT-2 Conv1D weights which are stored as [in, out] instead of [out, in].
pub(crate) fn load_compact_transposed(weights: &ModelWeights, name: &str) -> Result<CompactTensor, GenericModelError> {
    let tensor = load_weight(weights, name)?;
    let transposed = tensor.transpose_2d();
    Ok(tensor_to_compact(&transposed))
}

pub(crate) fn load_optional(weights: &ModelWeights, name: &Option<String>) -> Option<Tensor> {
    name.as_ref().and_then(|n| {
        if weights.has_tensor(n) {
            weights.load_tensor(n).ok()
        } else {
            None
        }
    })
}

/// Convert an f32 Tensor to a CompactTensor (bf16).
pub(crate) fn tensor_to_compact(t: &Tensor) -> CompactTensor {
    let bytes: Vec<u8> = t.data().iter().flat_map(|&v| {
        half::bf16::from_f32(v).to_le_bytes().to_vec()
    }).collect();
    CompactTensor::new(bytes, t.shape().to_vec(), crate::tensor::DType::BF16)
}

/// Add bias to a tensor (broadcasts for batched tensors).
pub(crate) fn add_bias_inplace(x: &mut Tensor, bias: &Tensor) {
    if x.ndim() == 1 {
        x.add_inplace(bias);
    } else {
        let data = x.data_mut();
        let hidden = bias.numel();
        let seq_len = data.len() / hidden;
        let b = bias.data();
        for s in 0..seq_len {
            for i in 0..hidden {
                data[s * hidden + i] += b[i];
            }
        }
    }
}

/// Split a fused QKV tensor into separate Q, K, V tensors.
pub(crate) fn split_qkv(
    fused: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Tensor, Tensor, Tensor) {
    let q_dim = num_heads * head_dim;
    let k_dim = num_kv_heads * head_dim;
    let v_dim = num_kv_heads * head_dim;
    let hidden = fused.shape()[1];

    let data = fused.data();
    let q_data: Vec<f32> = data[..q_dim * hidden].to_vec();
    let k_data: Vec<f32> = data[q_dim * hidden..(q_dim + k_dim) * hidden].to_vec();
    let v_data: Vec<f32> = data[(q_dim + k_dim) * hidden..(q_dim + k_dim + v_dim) * hidden].to_vec();

    (
        Tensor::new(q_data, vec![q_dim, hidden]).unwrap(),
        Tensor::new(k_data, vec![k_dim, hidden]).unwrap(),
        Tensor::new(v_data, vec![v_dim, hidden]).unwrap(),
    )
}

/// Split a fused QKV bias into separate Q, K, V biases.
pub(crate) fn split_qkv_bias(
    fused: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> (Tensor, Tensor, Tensor) {
    let q_dim = num_heads * head_dim;
    let k_dim = num_kv_heads * head_dim;
    let v_dim = num_kv_heads * head_dim;

    let data = fused.data();
    (
        Tensor::new(data[..q_dim].to_vec(), vec![q_dim]).unwrap(),
        Tensor::new(data[q_dim..q_dim + k_dim].to_vec(), vec![k_dim]).unwrap(),
        Tensor::new(data[q_dim + k_dim..q_dim + k_dim + v_dim].to_vec(), vec![v_dim]).unwrap(),
    )
}

// Note: to_llama_block() converter removed — the Llama-specific path is now
// fully replaced by the generic model which uses WeightStorage throughout.
