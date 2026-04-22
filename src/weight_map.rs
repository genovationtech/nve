//! Per-architecture weight name mapping.
//!
//! Maps a canonical set of weight names to the architecture-specific names
//! used in HuggingFace safetensors files.

use crate::arch::Architecture;

/// Canonical weight names used by the generic model.
pub struct LayerWeightNames {
    pub q_proj: String,
    pub k_proj: String,
    pub v_proj: String,
    pub o_proj: String,
    /// For gated FFNs (SwiGLU/GeGLU): gate projection. For standard FFN: None.
    pub gate_proj: Option<String>,
    /// Up projection (or fc1 in standard FFN).
    pub up_proj: String,
    /// Down projection (or fc2 in standard FFN).
    pub down_proj: String,
    pub input_norm: String,
    pub post_attn_norm: String,
    // Bias tensors (None if architecture doesn't use them).
    pub q_bias: Option<String>,
    pub k_bias: Option<String>,
    pub v_bias: Option<String>,
    pub o_bias: Option<String>,
    pub up_bias: Option<String>,
    pub down_bias: Option<String>,
    pub gate_bias: Option<String>,
    pub input_norm_bias: Option<String>,
    pub post_attn_norm_bias: Option<String>,
}

/// Global weight names (not per-layer).
pub struct GlobalWeightNames {
    pub embed_tokens: String,
    pub final_norm: String,
    pub lm_head: String,
    pub final_norm_bias: Option<String>,
    /// Learned position embeddings (GPT-2 style).
    pub position_embed: Option<String>,
}

/// Map layer index to weight names for a given architecture.
pub fn layer_weights(arch: Architecture, layer_idx: usize) -> LayerWeightNames {
    match arch {
        Architecture::Llama
        | Architecture::Mistral
        | Architecture::DeepSeek => llama_layer(layer_idx),

        Architecture::Qwen2 => qwen2_layer(layer_idx),
        Architecture::Phi3 => phi3_layer(layer_idx),
        Architecture::Gemma | Architecture::Gemma2 => gemma_layer(layer_idx),
        Architecture::GPTNeoX => gpt_neox_layer(layer_idx),
        Architecture::GPT2 => gpt2_layer(layer_idx),
        Architecture::Falcon => falcon_layer(layer_idx),
        Architecture::StableLM => stablelm_layer(layer_idx),
        Architecture::StarCoder2 => starcoder2_layer(layer_idx),
        Architecture::InternLM2 => internlm2_layer(layer_idx),
        Architecture::OLMo => olmo_layer(layer_idx),
    }
}

pub fn global_weights(arch: Architecture) -> GlobalWeightNames {
    match arch {
        Architecture::Llama
        | Architecture::Mistral
        | Architecture::DeepSeek
        | Architecture::Qwen2
        | Architecture::InternLM2 => GlobalWeightNames {
            embed_tokens: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: None,
            position_embed: None,
        },
        Architecture::Phi3 => GlobalWeightNames {
            embed_tokens: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: None,
            position_embed: None,
        },
        Architecture::Gemma | Architecture::Gemma2 => GlobalWeightNames {
            embed_tokens: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: None,
            position_embed: None,
        },
        Architecture::GPTNeoX => GlobalWeightNames {
            embed_tokens: "gpt_neox.embed_in.weight".into(),
            final_norm: "gpt_neox.final_layer_norm.weight".into(),
            lm_head: "embed_out.weight".into(),
            final_norm_bias: Some("gpt_neox.final_layer_norm.bias".into()),
            position_embed: None,
        },
        Architecture::GPT2 => GlobalWeightNames {
            embed_tokens: "wte.weight".into(),
            final_norm: "ln_f.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: Some("ln_f.bias".into()),
            position_embed: Some("wpe.weight".into()),
        },
        Architecture::Falcon => GlobalWeightNames {
            embed_tokens: "transformer.word_embeddings.weight".into(),
            final_norm: "transformer.ln_f.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: Some("transformer.ln_f.bias".into()),
            position_embed: None,
        },
        Architecture::StableLM | Architecture::OLMo => GlobalWeightNames {
            embed_tokens: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: None,
            position_embed: None,
        },
        Architecture::StarCoder2 => GlobalWeightNames {
            embed_tokens: "model.embed_tokens.weight".into(),
            final_norm: "model.norm.weight".into(),
            lm_head: "lm_head.weight".into(),
            final_norm_bias: Some("model.norm.bias".into()),
            position_embed: None,
        },
    }
}

// ── Per-architecture layer mappings ──

fn llama_layer(i: usize) -> LayerWeightNames {
    let p = format!("model.layers.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.self_attn.q_proj.weight", p),
        k_proj: format!("{}.self_attn.k_proj.weight", p),
        v_proj: format!("{}.self_attn.v_proj.weight", p),
        o_proj: format!("{}.self_attn.o_proj.weight", p),
        gate_proj: Some(format!("{}.mlp.gate_proj.weight", p)),
        up_proj: format!("{}.mlp.up_proj.weight", p),
        down_proj: format!("{}.mlp.down_proj.weight", p),
        input_norm: format!("{}.input_layernorm.weight", p),
        post_attn_norm: format!("{}.post_attention_layernorm.weight", p),
        q_bias: None, k_bias: None, v_bias: None, o_bias: None,
        up_bias: None, down_bias: None, gate_bias: None,
        input_norm_bias: None, post_attn_norm_bias: None,
    }
}

fn qwen2_layer(i: usize) -> LayerWeightNames {
    let mut names = llama_layer(i);
    let p = format!("model.layers.{}", i);
    names.q_bias = Some(format!("{}.self_attn.q_proj.bias", p));
    names.k_bias = Some(format!("{}.self_attn.k_proj.bias", p));
    names.v_bias = Some(format!("{}.self_attn.v_proj.bias", p));
    names
}

fn phi3_layer(i: usize) -> LayerWeightNames {
    let p = format!("model.layers.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.self_attn.qkv_proj.weight", p),   // fused QKV
        k_proj: format!("{}.self_attn.qkv_proj.weight", p),
        v_proj: format!("{}.self_attn.qkv_proj.weight", p),
        o_proj: format!("{}.self_attn.o_proj.weight", p),
        gate_proj: Some(format!("{}.mlp.gate_up_proj.weight", p)), // fused gate+up
        up_proj: format!("{}.mlp.gate_up_proj.weight", p),
        down_proj: format!("{}.mlp.down_proj.weight", p),
        input_norm: format!("{}.input_layernorm.weight", p),
        post_attn_norm: format!("{}.post_attention_layernorm.weight", p),
        q_bias: None, k_bias: None, v_bias: None, o_bias: None,
        up_bias: None, down_bias: None, gate_bias: None,
        input_norm_bias: None, post_attn_norm_bias: None,
    }
}

fn gemma_layer(i: usize) -> LayerWeightNames {
    llama_layer(i) // Gemma uses same naming as Llama
}

fn gpt_neox_layer(i: usize) -> LayerWeightNames {
    let p = format!("gpt_neox.layers.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.attention.query_key_value.weight", p),
        k_proj: format!("{}.attention.query_key_value.weight", p), // fused QKV
        v_proj: format!("{}.attention.query_key_value.weight", p), // fused QKV
        o_proj: format!("{}.attention.dense.weight", p),
        gate_proj: None,
        up_proj: format!("{}.mlp.dense_h_to_4h.weight", p),
        down_proj: format!("{}.mlp.dense_4h_to_h.weight", p),
        input_norm: format!("{}.input_layernorm.weight", p),
        post_attn_norm: format!("{}.post_attention_layernorm.weight", p),
        q_bias: Some(format!("{}.attention.query_key_value.bias", p)),
        k_bias: None, v_bias: None,
        o_bias: Some(format!("{}.attention.dense.bias", p)),
        up_bias: Some(format!("{}.mlp.dense_h_to_4h.bias", p)),
        down_bias: Some(format!("{}.mlp.dense_4h_to_h.bias", p)),
        gate_bias: None,
        input_norm_bias: Some(format!("{}.input_layernorm.bias", p)),
        post_attn_norm_bias: Some(format!("{}.post_attention_layernorm.bias", p)),
    }
}

fn gpt2_layer(i: usize) -> LayerWeightNames {
    let p = format!("h.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.attn.c_attn.weight", p),   // fused QKV
        k_proj: format!("{}.attn.c_attn.weight", p),
        v_proj: format!("{}.attn.c_attn.weight", p),
        o_proj: format!("{}.attn.c_proj.weight", p),
        gate_proj: None,
        up_proj: format!("{}.mlp.c_fc.weight", p),
        down_proj: format!("{}.mlp.c_proj.weight", p),
        input_norm: format!("{}.ln_1.weight", p),
        post_attn_norm: format!("{}.ln_2.weight", p),
        q_bias: Some(format!("{}.attn.c_attn.bias", p)),
        k_bias: None, v_bias: None,
        o_bias: Some(format!("{}.attn.c_proj.bias", p)),
        up_bias: Some(format!("{}.mlp.c_fc.bias", p)),
        down_bias: Some(format!("{}.mlp.c_proj.bias", p)),
        gate_bias: None,
        input_norm_bias: Some(format!("{}.ln_1.bias", p)),
        post_attn_norm_bias: Some(format!("{}.ln_2.bias", p)),
    }
}

fn falcon_layer(i: usize) -> LayerWeightNames {
    let p = format!("transformer.h.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.self_attention.query_key_value.weight", p),
        k_proj: format!("{}.self_attention.query_key_value.weight", p),
        v_proj: format!("{}.self_attention.query_key_value.weight", p),
        o_proj: format!("{}.self_attention.dense.weight", p),
        gate_proj: None,
        up_proj: format!("{}.mlp.dense_h_to_4h.weight", p),
        down_proj: format!("{}.mlp.dense_4h_to_h.weight", p),
        input_norm: format!("{}.input_layernorm.weight", p),
        post_attn_norm: format!("{}.post_attention_layernorm.weight", p),
        q_bias: None, k_bias: None, v_bias: None, o_bias: None,
        up_bias: None, down_bias: None, gate_bias: None,
        input_norm_bias: Some(format!("{}.input_layernorm.bias", p)),
        post_attn_norm_bias: Some(format!("{}.post_attention_layernorm.bias", p)),
    }
}

fn stablelm_layer(i: usize) -> LayerWeightNames {
    llama_layer(i) // StableLM uses Llama-style naming
}

fn starcoder2_layer(i: usize) -> LayerWeightNames {
    let p = format!("model.layers.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.self_attn.q_proj.weight", p),
        k_proj: format!("{}.self_attn.k_proj.weight", p),
        v_proj: format!("{}.self_attn.v_proj.weight", p),
        o_proj: format!("{}.self_attn.o_proj.weight", p),
        gate_proj: None,
        up_proj: format!("{}.mlp.c_fc.weight", p),
        down_proj: format!("{}.mlp.c_proj.weight", p),
        input_norm: format!("{}.input_layernorm.weight", p),
        post_attn_norm: format!("{}.post_attention_layernorm.weight", p),
        q_bias: Some(format!("{}.self_attn.q_proj.bias", p)),
        k_bias: Some(format!("{}.self_attn.k_proj.bias", p)),
        v_bias: Some(format!("{}.self_attn.v_proj.bias", p)),
        o_bias: Some(format!("{}.self_attn.o_proj.bias", p)),
        up_bias: Some(format!("{}.mlp.c_fc.bias", p)),
        down_bias: Some(format!("{}.mlp.c_proj.bias", p)),
        gate_bias: None,
        input_norm_bias: Some(format!("{}.input_layernorm.bias", p)),
        post_attn_norm_bias: Some(format!("{}.post_attention_layernorm.bias", p)),
    }
}

fn internlm2_layer(i: usize) -> LayerWeightNames {
    let p = format!("model.layers.{}", i);
    LayerWeightNames {
        q_proj: format!("{}.attention.wqkv.weight", p), // fused QKV
        k_proj: format!("{}.attention.wqkv.weight", p),
        v_proj: format!("{}.attention.wqkv.weight", p),
        o_proj: format!("{}.attention.wo.weight", p),
        gate_proj: Some(format!("{}.feed_forward.w1.weight", p)),
        up_proj: format!("{}.feed_forward.w3.weight", p),
        down_proj: format!("{}.feed_forward.w2.weight", p),
        input_norm: format!("{}.attention_norm.weight", p),
        post_attn_norm: format!("{}.ffn_norm.weight", p),
        q_bias: None, k_bias: None, v_bias: None, o_bias: None,
        up_bias: None, down_bias: None, gate_bias: None,
        input_norm_bias: None, post_attn_norm_bias: None,
    }
}

fn olmo_layer(i: usize) -> LayerWeightNames {
    llama_layer(i) // OLMo uses Llama-style naming in newer versions
}

/// Check if architecture uses fused QKV weights (single tensor for Q, K, V).
pub fn uses_fused_qkv(arch: Architecture) -> bool {
    matches!(
        arch,
        Architecture::GPTNeoX | Architecture::GPT2 | Architecture::Falcon | Architecture::InternLM2 | Architecture::Phi3
    )
}

/// Check if architecture uses fused gate+up projection (single tensor for gate and up in SwiGLU FFN).
pub fn uses_fused_gate_up(arch: Architecture) -> bool {
    matches!(arch, Architecture::Phi3)
}

/// Check if architecture uses Conv1D-style weights [in_features, out_features]
/// instead of the standard [out_features, in_features] layout.
/// These need to be transposed before use with our linear ops.
pub fn uses_conv1d_weights(arch: Architecture) -> bool {
    matches!(arch, Architecture::GPT2)
}
