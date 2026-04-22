//! Grouped-Query Attention (GQA) with KV cache.
//!
//! Implements Llama's multi-head attention with:
//! - Grouped-query attention (32 Q heads, 8 KV heads for 1B)
//! - Rotary position embeddings
//! - Contiguous KV cache for cache-friendly attention
//! - Causal masking
//! - CompactTensor (bf16) weight projections

use crate::ops::{self, RopeFreqs};
use crate::quantize::{WeightStorage, ws_linear, ws_linear_vec};
use crate::tensor::{dot, Tensor};

/// KV Cache for autoregressive generation.
///
/// Uses a contiguous buffer layout for cache-friendly access:
/// key_data[layer][kv_head][position] = [head_dim] f32 values
pub struct KvCache {
    /// Contiguous key storage: [num_layers * num_kv_heads * max_seq * head_dim]
    key_data: Vec<f32>,
    /// Contiguous value storage: same layout
    val_data: Vec<f32>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq: usize,
    seq_len: usize,
}

impl KvCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let max_seq = 4096; // Pre-allocate for typical max sequence
        let total = num_layers * num_kv_heads * max_seq * head_dim;
        KvCache {
            key_data: vec![0.0f32; total],
            val_data: vec![0.0f32; total],
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq,
            seq_len: 0,
        }
    }

    /// Ensure capacity for at least `needed` positions.
    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.max_seq {
            return;
        }
        let new_max = (needed * 2).max(self.max_seq * 2);
        let new_total = self.num_layers * self.num_kv_heads * new_max * self.head_dim;
        let mut new_keys = vec![0.0f32; new_total];
        let mut new_vals = vec![0.0f32; new_total];

        // Copy existing data
        for l in 0..self.num_layers {
            for h in 0..self.num_kv_heads {
                for s in 0..self.seq_len {
                    let old_offset = ((l * self.num_kv_heads + h) * self.max_seq + s) * self.head_dim;
                    let new_offset = ((l * self.num_kv_heads + h) * new_max + s) * self.head_dim;
                    new_keys[new_offset..new_offset + self.head_dim]
                        .copy_from_slice(&self.key_data[old_offset..old_offset + self.head_dim]);
                    new_vals[new_offset..new_offset + self.head_dim]
                        .copy_from_slice(&self.val_data[old_offset..old_offset + self.head_dim]);
                }
            }
        }

        self.key_data = new_keys;
        self.val_data = new_vals;
        self.max_seq = new_max;
    }

    /// Append a KV pair for one layer, one KV head, at the current sequence position.
    #[inline]
    pub fn append(&mut self, layer: usize, kv_head: usize, key: &[f32], value: &[f32]) {
        let pos = self.seq_len;
        self.ensure_capacity(pos + 1);
        let offset = ((layer * self.num_kv_heads + kv_head) * self.max_seq + pos) * self.head_dim;
        self.key_data[offset..offset + self.head_dim].copy_from_slice(key);
        self.val_data[offset..offset + self.head_dim].copy_from_slice(value);
    }

    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Insert a KV pair at a specific position (used by prefill).
    #[inline]
    pub fn insert_at(&mut self, layer: usize, kv_head: usize, pos: usize, key: &[f32], value: &[f32]) {
        self.ensure_capacity(pos + 1);
        let offset = ((layer * self.num_kv_heads + kv_head) * self.max_seq + pos) * self.head_dim;
        self.key_data[offset..offset + self.head_dim].copy_from_slice(key);
        self.val_data[offset..offset + self.head_dim].copy_from_slice(value);
    }

    /// Set the sequence length directly (used after prefill completes all layers).
    pub fn set_seq_len(&mut self, len: usize) {
        self.seq_len = len;
    }

    /// Get a pointer to cached key at [layer, kv_head, position].
    #[inline]
    pub fn key_at(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let offset = ((layer * self.num_kv_heads + kv_head) * self.max_seq + pos) * self.head_dim;
        &self.key_data[offset..offset + self.head_dim]
    }

    /// Get a pointer to cached value at [layer, kv_head, position].
    #[inline]
    pub fn val_at(&self, layer: usize, kv_head: usize, pos: usize) -> &[f32] {
        let offset = ((layer * self.num_kv_heads + kv_head) * self.max_seq + pos) * self.head_dim;
        &self.val_data[offset..offset + self.head_dim]
    }

    /// Number of cached key positions for a given layer/head.
    #[inline]
    pub fn cached_len(&self) -> usize {
        self.seq_len
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
        // No need to zero the buffer — we track length
    }

    pub fn memory_bytes(&self) -> usize {
        self.key_data.len() * 4 + self.val_data.len() * 4
    }
}

/// Weights for a single attention layer (WeightStorage for projections).
pub struct AttentionWeights<'a> {
    pub q_proj: &'a WeightStorage, // [num_heads * head_dim, hidden_size]
    pub k_proj: &'a WeightStorage, // [num_kv_heads * head_dim, hidden_size]
    pub v_proj: &'a WeightStorage, // [num_kv_heads * head_dim, hidden_size]
    pub o_proj: &'a WeightStorage, // [hidden_size, num_heads * head_dim]
    pub q_bias: Option<&'a Tensor>,
    pub k_bias: Option<&'a Tensor>,
    pub v_bias: Option<&'a Tensor>,
}

/// Run grouped-query attention for a single token (autoregressive decode step).
/// If rope_freqs is None, no rotary position embedding is applied (e.g., GPT-2 with learned pos embeddings).
pub fn gqa_forward_single(
    x: &Tensor,
    weights: &AttentionWeights,
    rope_freqs: Option<&RopeFreqs>,
    kv_cache: &mut KvCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let num_groups = num_heads / num_kv_heads;

    // Project Q, K, V using WeightStorage.
    let mut q_all = ws_linear_vec(x, weights.q_proj);
    let mut k_all = ws_linear_vec(x, weights.k_proj);
    let mut v_all = ws_linear_vec(x, weights.v_proj);

    // Apply QKV biases if present.
    if let Some(bias) = weights.q_bias { q_all.add_inplace(bias); }
    if let Some(bias) = weights.k_bias { k_all.add_inplace(bias); }
    if let Some(bias) = weights.v_bias { v_all.add_inplace(bias); }

    let mut q_data = q_all.data().to_vec();
    let mut k_data = k_all.data().to_vec();
    let v_data = v_all.data();

    // Apply RoPE if present.
    if let Some(rope) = rope_freqs {
        for h in 0..num_heads {
            let q_head = &mut q_data[h * head_dim..(h + 1) * head_dim];
            rope.apply_q(q_head, pos);
        }
        for h in 0..num_kv_heads {
            let k_head = &mut k_data[h * head_dim..(h + 1) * head_dim];
            rope.apply_k(k_head, pos);
        }
    }

    // Store K, V in contiguous cache.
    for h in 0..num_kv_heads {
        kv_cache.append(
            layer_idx, h,
            &k_data[h * head_dim..(h + 1) * head_dim],
            &v_data[h * head_dim..(h + 1) * head_dim],
        );
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let cache_len = kv_cache.cached_len() + 1; // includes position just appended (before advance)

    // Compute attention for each Q head.
    let mut output = vec![0.0f32; num_heads * head_dim];

    for qh in 0..num_heads {
        let kv_head = qh / num_groups;
        let q = &q_data[qh * head_dim..(qh + 1) * head_dim];

        // Compute attention scores: Q @ K^T / sqrt(d)
        let mut scores = vec![0.0f32; cache_len];
        for t in 0..cache_len {
            scores[t] = dot(q, kv_cache.key_at(layer_idx, kv_head, t)) * scale;
        }

        ops::softmax(&mut scores);

        // Weighted sum of values.
        let out_head = &mut output[qh * head_dim..(qh + 1) * head_dim];
        for t in 0..cache_len {
            let v = kv_cache.val_at(layer_idx, kv_head, t);
            for d in 0..head_dim {
                out_head[d] += scores[t] * v[d];
            }
        }
    }

    // Output projection.
    let attn_out = Tensor::new(output, vec![num_heads * head_dim]).unwrap();
    ws_linear_vec(&attn_out, weights.o_proj)
}

/// Run grouped-query attention for prefill (multiple tokens at once).
/// If rope_freqs is None, no rotary position embedding is applied.
pub fn gqa_forward_prefill(
    x: &Tensor,
    weights: &AttentionWeights,
    rope_freqs: Option<&RopeFreqs>,
    kv_cache: &mut KvCache,
    layer_idx: usize,
    start_pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor {
    let seq_len = x.shape()[0];
    let hidden_size = x.shape()[1];
    let num_groups = num_heads / num_kv_heads;

    // Batch project using WeightStorage.
    let mut q_all = ws_linear(x, weights.q_proj);
    let mut k_all = ws_linear(x, weights.k_proj);
    let mut v_all = ws_linear(x, weights.v_proj);

    // Apply QKV biases if present (broadcast across seq_len).
    if let Some(bias) = weights.q_bias {
        let data = q_all.data_mut();
        let bias_data = bias.data();
        let bias_len = bias_data.len();
        for s in 0..(data.len() / bias_len) {
            for j in 0..bias_len { data[s * bias_len + j] += bias_data[j]; }
        }
    }
    if let Some(bias) = weights.k_bias {
        let data = k_all.data_mut();
        let bias_data = bias.data();
        let bias_len = bias_data.len();
        for s in 0..(data.len() / bias_len) {
            for j in 0..bias_len { data[s * bias_len + j] += bias_data[j]; }
        }
    }
    if let Some(bias) = weights.v_bias {
        let data = v_all.data_mut();
        let bias_data = bias.data();
        let bias_len = bias_data.len();
        for s in 0..(data.len() / bias_len) {
            for j in 0..bias_len { data[s * bias_len + j] += bias_data[j]; }
        }
    }

    let mut q_data = q_all.data().to_vec();
    let mut k_data = k_all.data().to_vec();
    let v_data = v_all.data();

    let q_row_size = num_heads * head_dim;
    let kv_row_size = num_kv_heads * head_dim;

    // Apply RoPE (if present) and cache K, V for each position.
    for s in 0..seq_len {
        let pos = start_pos + s;

        if let Some(rope) = rope_freqs {
            for h in 0..num_heads {
                let offset = s * q_row_size + h * head_dim;
                let q_head = &mut q_data[offset..offset + head_dim];
                rope.apply_q(q_head, pos);
            }

            for h in 0..num_kv_heads {
                let k_offset = s * kv_row_size + h * head_dim;
                let k_head = &mut k_data[k_offset..k_offset + head_dim];
                rope.apply_k(k_head, pos);
            }
        }

        for h in 0..num_kv_heads {
            let k_offset = s * kv_row_size + h * head_dim;
            let v_offset = s * kv_row_size + h * head_dim;
            kv_cache.insert_at(layer_idx, h, pos, &k_data[k_offset..k_offset + head_dim], &v_data[v_offset..v_offset + head_dim]);
        }
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute attention for each position and head.
    let mut output = vec![0.0f32; seq_len * hidden_size];

    for s in 0..seq_len {
        let total_len = start_pos + s + 1;

        for qh in 0..num_heads {
            let kv_head = qh / num_groups;
            let q_offset = s * q_row_size + qh * head_dim;
            let q = &q_data[q_offset..q_offset + head_dim];

            let mut scores = vec![0.0f32; total_len];
            for t in 0..total_len {
                scores[t] = dot(q, kv_cache.key_at(layer_idx, kv_head, t)) * scale;
            }

            ops::softmax(&mut scores);

            let out_offset = s * hidden_size + qh * head_dim;
            let out_head = &mut output[out_offset..out_offset + head_dim];
            for t in 0..total_len {
                let v = kv_cache.val_at(layer_idx, kv_head, t);
                for d in 0..head_dim {
                    out_head[d] += scores[t] * v[d];
                }
            }
        }
    }

    // Output projection.
    let attn_concat = Tensor::new(output, vec![seq_len, num_heads * head_dim]).unwrap();
    ws_linear(&attn_concat, weights.o_proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{CompactTensor, DType};
    use half::bf16;

    fn make_compact_weight(rows: usize, cols: usize, val: f32) -> CompactTensor {
        let bytes: Vec<u8> = (0..rows * cols)
            .flat_map(|_| bf16::from_f32(val).to_le_bytes().to_vec())
            .collect();
        CompactTensor::new(bytes, vec![rows, cols], DType::BF16)
    }

    fn make_ws_weight(rows: usize, cols: usize, val: f32) -> WeightStorage {
        WeightStorage::Compact(make_compact_weight(rows, cols, val))
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KvCache::new(2, 4, 8);
        assert_eq!(cache.seq_len(), 0);

        cache.append(0, 0, &vec![1.0; 8], &vec![2.0; 8]);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.key_at(0, 0, 0), &vec![1.0; 8][..]);
    }

    #[test]
    fn test_gqa_single_token() {
        let hidden = 16;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden / num_heads;

        let q_proj = make_ws_weight(num_heads * head_dim, hidden, 0.1);
        let k_proj = make_ws_weight(num_kv_heads * head_dim, hidden, 0.1);
        let v_proj = make_ws_weight(num_kv_heads * head_dim, hidden, 0.1);
        let o_proj = make_ws_weight(hidden, num_heads * head_dim, 0.1);

        let weights = AttentionWeights {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            o_proj: &o_proj,
            q_bias: None,
            k_bias: None,
            v_bias: None,
        };

        let rope = RopeFreqs::new(head_dim, 128, 10000.0, None);
        let mut cache = KvCache::new(1, num_kv_heads, head_dim);

        let x = Tensor::new(vec![1.0; hidden], vec![hidden]).unwrap();
        let out = gqa_forward_single(
            &x, &weights, Some(&rope), &mut cache, 0, 0,
            num_heads, num_kv_heads, head_dim,
        );

        assert_eq!(out.shape(), &[hidden]);
        assert_eq!(cache.seq_len(), 0); // advance not called yet by gqa_forward_single
    }
}
