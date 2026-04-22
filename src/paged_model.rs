//! Paged model — NVE tier-aware inference for any supported architecture.
//!
//! Wraps the model with NVE's virtual weight paging system.
//! Weights are loaded as CompactTensor (bf16) for 2x memory reduction,
//! enabling all layers to fit in RAM for most models.

use std::path::Path;
use std::time::Instant;

use log::{debug, info};

use crate::arch::{FfnType, PosEncoding, UnifiedConfig};
use crate::attention::{gqa_forward_prefill, gqa_forward_single, AttentionWeights, KvCache};
use crate::generic_model::{self, apply_ffn, apply_norm, apply_norm_batch, ChannelSaliency, GenericBlockWeights};
use crate::model::{GenerationResult, ModelError};
use crate::ops::{self, RopeFreqs};
use crate::quantize::{QuantMode, allocate_bits, ws_linear};
use crate::safetensors::ModelWeights;
use crate::tensor::{compact_linear_vec, CompactTensor, Tensor};
use crate::weight_map;

/// Memory budget configuration for paged inference.
#[derive(Debug, Clone)]
pub struct PagedConfig {
    pub hot_budget_bytes: usize,
    pub warm_budget_bytes: usize,
    pub prefetch_ahead: usize,
    pub profile_activations: bool,
    /// Hot-only mode: skip layers that don't fit in memory instead of paging
    /// from SSD. Trades quality for speed — only active layers are computed,
    /// inactive layers pass the residual stream through unchanged.
    pub hot_only_mode: bool,
    /// Override number of active layers (default: fit as many as budget allows).
    /// Lower = faster but lower quality. Even for models that fit in RAM,
    /// setting this reduces compute for faster inference.
    pub active_layers: Option<usize>,
    /// Quantization mode: None (bf16), Q4, or Q8. Applied after each layer load.
    pub quant_mode: QuantMode,
    /// Compute device for GPU-accelerated profiling matmuls.
    /// Accepts the same strings as `Device::resolve`: "auto", "cpu", "cuda:0", etc.
    /// "auto" picks the best available device at runtime (CUDA → HIP → Metal → CPU).
    /// Only takes effect when compiled with a GPU feature (`--features cuda/hip/metal`).
    pub device: String,
}

impl Default for PagedConfig {
    fn default() -> Self {
        PagedConfig {
            hot_budget_bytes: 512 * 1024 * 1024,
            warm_budget_bytes: 2048 * 1024 * 1024,
            prefetch_ahead: 2,
            profile_activations: false,
            hot_only_mode: false,
            active_layers: None,
            quant_mode: QuantMode::None,
            device: "auto".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerTier {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Default)]
pub struct PagingStats {
    pub page_hits: u64,
    pub page_faults: u64,
    pub layers_loaded: u64,
    pub layers_evicted: u64,
    pub total_load_time_ms: f64,
    pub total_evict_time_ms: f64,
    pub prefetch_hits: u64,
}

impl PagingStats {
    pub fn fault_rate(&self) -> f64 {
        let total = self.page_hits + self.page_faults;
        if total == 0 {
            return 0.0;
        }
        self.page_faults as f64 / total as f64
    }
}

/// AWQ calibration data collected during profiling.
pub struct AwqCalibration {
    pub layer_saliency: Vec<ChannelSaliency>,
}

/// Paged model — loads layer weights on demand as GenericBlockWeights.
/// Works with any supported architecture.
pub struct PagedModel {
    config: UnifiedConfig,
    paged_config: PagedConfig,
    embed_tokens: CompactTensor,
    norm: Tensor,
    norm_bias: Option<Tensor>,
    lm_head: Option<CompactTensor>,
    pos_embed: Option<Tensor>,
    /// Layer weights loaded as GenericBlockWeights. None = cold.
    layers: Vec<Option<GenericBlockWeights>>,
    layer_tiers: Vec<LayerTier>,
    layer_access: Vec<u64>,
    weight_source: ModelWeights,
    rope_freqs: Option<RopeFreqs>,
    kv_cache: KvCache,
    pos: usize,
    tick: u64,
    stats: PagingStats,
    hot_used_bytes: usize,
    warm_used_bytes: usize,
    layer_size_bytes: usize,
    layer_importance: Vec<f64>,
    /// Which layers are active in hot-only mode. If empty, all layers are active.
    active_layer_set: Vec<bool>,
    layers_skipped: usize,
    /// AWQ calibration data collected during profiling.
    awq_calibration: Option<AwqCalibration>,
    /// Per-layer quantization assignments for profile-guided mode.
    layer_quant_assignments: Option<Vec<QuantMode>>,
    /// GPU inference state — populated by `init_gpu()` when a GPU device is configured.
    /// Hot layers execute entirely on GPU; CPU↔GPU transfers happen only at tier boundaries.
    #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
    gpu_state: Option<crate::gpu_layer::GpuInferenceState>,
}

// ── Scorer comparison types and helpers ─────────────────────────────────────

/// Side-by-side comparison of four layer-importance scoring methods produced
/// by a single streaming profiling pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoringComparison {
    /// Per-layer scores for each method (averaged over tokens).
    pub scores_proxy:     Vec<f64>,  // FFN L2 + attn-proxy L2  (production scorer)
    pub scores_ffn_only:  Vec<f64>,  // FFN L2 only
    pub scores_attn_proxy: Vec<f64>, // Attn-proxy L2 only (Q/V projections)
    pub scores_input_l2:  Vec<f64>,  // ‖RMSNorm(hidden)‖₂ — zero extra compute

    /// Kendall's τ rank correlation of each alternative vs the proxy scorer.
    /// 1.0 = identical ranking, 0.0 = uncorrelated, -1.0 = reversed.
    pub tau_ffn_vs_proxy:   f64,
    pub tau_attn_vs_proxy:  f64,
    pub tau_input_vs_proxy: f64,

    /// Fraction of top-k layers (k = num_layers/2) shared with the proxy ranking.
    pub topk_ffn_vs_proxy:   f64,
    pub topk_attn_vs_proxy:  f64,
    pub topk_input_vs_proxy: f64,

    /// Wall-time spent purely on the Q/V projections (ms) across all tokens and layers.
    pub attn_proxy_cost_ms: f64,
    /// Total profiling loop time (ms).
    pub total_time_ms: f64,
    /// `attn_proxy_cost_ms / total_time_ms` — how much overhead the proxy adds.
    pub attn_proxy_fraction: f64,
}

/// Kendall's τ rank correlation between two score vectors.
/// O(n²) — fine for n ≤ ~64 layers.
fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 { return 1.0; }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let sa = (a[i] - a[j]).signum();
            let sb = (b[i] - b[j]).signum();
            match (sa * sb).partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => concordant += 1,
                Some(std::cmp::Ordering::Less)    => discordant += 1,
                _ => {} // tie in at least one scorer
            }
        }
    }
    let total = (n * (n - 1) / 2) as f64;
    if total == 0.0 { 1.0 } else { (concordant - discordant) as f64 / total }
}

/// Fraction of the top-k elements (by score) that match between two vectors.
fn top_k_overlap_vecs(a: &[f64], b: &[f64], k: usize) -> f64 {
    if a.is_empty() || k == 0 { return 1.0; }
    let k = k.min(a.len());
    let top_k = |v: &[f64]| -> std::collections::HashSet<usize> {
        let mut ranked: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        ranked.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.iter().take(k).map(|&(i, _)| i).collect()
    };
    let set_a = top_k(a);
    let set_b = top_k(b);
    set_a.intersection(&set_b).count() as f64 / k as f64
}

// ── GPU-accelerated profiling pass ───────────────────────────────────────────
//
// Dispatches the five heavy matmuls (Q, V, gate, up, down) to the GPU via
// candle-core while keeping norms and activations on the CPU.  Falls back
// gracefully to the CPU batch path on any error.
//
// Only compiled when at least one GPU feature is enabled.

#[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
fn profile_layer_gpu(
    hidden_batch: &Tensor,                         // [n_tok, d_h] f32, on CPU
    layer: &GenericBlockWeights,
    config: &crate::arch::UnifiedConfig,
    device: &crate::device::Device,
) -> Result<(Tensor, Vec<f32>, Vec<f32>, f64), candle_core::Error> {
    use candle_core::{Tensor as CT, D};
    use crate::arch::{FfnType, NormType};
    use crate::quantize::WeightStorage;

    let candle_dev = device.to_candle()?;

    let shape = hidden_batch.shape();
    let n_tok = shape[0];
    let d_h   = shape[1];

    // Helper: CompactTensor (bf16) → candle F32 on the target device.
    // We convert to f32 on CPU first to avoid dtype plumbing; this is a one-off
    // per-layer operation so the overhead is acceptable.
    let compact_to_gpu = |ws: &WeightStorage| -> candle_core::Result<CT> {
        let c = match ws {
            WeightStorage::Compact(c) => c,
            _ => return Err(candle_core::Error::Msg(
                "weight is not compact bf16 — GPU profiling requires unquantized layers".into()
            )),
        };
        let f32t = c.to_f32();
        let rows = c.shape()[0];
        let cols = if c.shape().len() >= 2 { c.shape()[1] } else { c.shape()[0] };
        CT::from_slice(f32t.data(), (rows, cols), &candle_dev)
    };

    // ── RMSNorm / LayerNorm on CPU (fast, avoids candle norm plumbing) ────────
    let normed = crate::generic_model::apply_norm_batch(
        hidden_batch,
        &layer.input_norm,
        layer.input_norm_bias.as_ref(),
        config.norm_type,
        config.norm_eps,
    );
    let normed2 = crate::generic_model::apply_norm_batch(
        hidden_batch,
        &layer.post_attn_norm,
        layer.post_attn_norm_bias.as_ref(),
        config.norm_type,
        config.norm_eps,
    );

    // Hidden saliency: mean |normed[tok][j]| over tokens.
    let h_sal: Vec<f32> = (0..d_h)
        .map(|j| {
            (0..n_tok).map(|t| normed.data()[t * d_h + j].abs()).sum::<f32>()
                / n_tok as f32
        })
        .collect();

    // ── Q / V projection on GPU ───────────────────────────────────────────────
    let normed_gpu = CT::from_slice(normed.data(), (n_tok, d_h), &candle_dev)?;
    let q_w = compact_to_gpu(&layer.q_proj)?;
    let v_w = compact_to_gpu(&layer.v_proj)?;
    let q_out = normed_gpu.matmul(&q_w.t()?)?;  // [n_tok, d_q]
    let v_out = normed_gpu.matmul(&v_w.t()?)?;  // [n_tok, d_v]

    let attn_per_tok: Vec<f32> = q_out.sqr()?.sum(D::Minus1)?
        .add(&v_out.sqr()?.sum(D::Minus1)?)?
        .sqrt()?
        .to_vec1()?;
    let attn_l2_sum: f64 = attn_per_tok.iter().map(|&x| x as f64).sum();

    // ── FFN on GPU ────────────────────────────────────────────────────────────
    let normed2_gpu = CT::from_slice(normed2.data(), (n_tok, d_h), &candle_dev)?;

    let ffn_out_gpu = match config.ffn_type {
        FfnType::SwiGLU => {
            let gate_w = compact_to_gpu(layer.gate_proj.as_ref().unwrap())?;
            let up_w   = compact_to_gpu(&layer.up_proj)?;
            let down_w = compact_to_gpu(&layer.down_proj)?;
            let gate   = normed2_gpu.matmul(&gate_w.t()?)?.silu()?;
            let up     = normed2_gpu.matmul(&up_w.t()?)?;
            gate.mul(&up)?.matmul(&down_w.t()?)?
        }
        FfnType::GeGLU => {
            let gate_w = compact_to_gpu(layer.gate_proj.as_ref().unwrap())?;
            let up_w   = compact_to_gpu(&layer.up_proj)?;
            let down_w = compact_to_gpu(&layer.down_proj)?;
            let gate   = normed2_gpu.matmul(&gate_w.t()?)?.gelu_erf()?;
            let up     = normed2_gpu.matmul(&up_w.t()?)?;
            gate.mul(&up)?.matmul(&down_w.t()?)?
        }
        FfnType::GELU => {
            let up_w   = compact_to_gpu(&layer.up_proj)?;
            let down_w = compact_to_gpu(&layer.down_proj)?;
            normed2_gpu.matmul(&up_w.t()?)?.gelu_erf()?.matmul(&down_w.t()?)?
        }
        _ => return Err(candle_core::Error::Msg(
            format!("FFN type {:?} not yet supported in GPU profiling", config.ffn_type)
        )),
    };

    // Per-token FFN L2 norms.
    let ffn_per_tok: Vec<f32> = ffn_out_gpu.sqr()?.sum(D::Minus1)?.sqrt()?.to_vec1()?;
    let ffn_l2_sum: f64 = ffn_per_tok.iter().map(|&x| x as f64).sum();
    let layer_score = (attn_l2_sum + ffn_l2_sum) / n_tok as f64;

    // FFN saliency: mean |ffn_out[tok][j]| over tokens (proxy for intermediate dim).
    let ffn_out_flat: Vec<f32> = ffn_out_gpu.flatten_all()?.to_vec1()?;
    let ffn_sal: Vec<f32> = (0..d_h)
        .map(|j| {
            (0..n_tok).map(|t| ffn_out_flat[t * d_h + j].abs()).sum::<f32>()
                / n_tok as f32
        })
        .collect();

    // Residual: new_hidden = hidden + ffn_out.
    let h_gpu  = CT::from_slice(hidden_batch.data(), (n_tok, d_h), &candle_dev)?;
    let new_h_flat: Vec<f32> = h_gpu.add(&ffn_out_gpu)?.flatten_all()?.to_vec1()?;
    let new_hidden = Tensor::new(new_h_flat, vec![n_tok, d_h]).unwrap();

    Ok((new_hidden, h_sal, ffn_sal, layer_score))
}

// ─────────────────────────────────────────────────────────────────────────────

impl PagedModel {
    pub fn from_dir(
        model_dir: &Path,
        paged_config: PagedConfig,
    ) -> Result<Self, ModelError> {
        info!("Loading paged model from {:?}", model_dir);
        let config = UnifiedConfig::from_model_dir(model_dir)
            .map_err(|e| ModelError::WeightNotFound(format!("config: {}", e)))?;

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

        let weight_source = ModelWeights::load(model_dir)?;

        let global = weight_map::global_weights(config.arch);

        let embed_tokens = weight_source
            .load_compact(&global.embed_tokens)
            .map_err(|e| ModelError::WeightNotFound(format!("embed_tokens: {}", e)))?;
        let norm = weight_source
            .load_tensor(&global.final_norm)
            .map_err(|e| ModelError::WeightNotFound(format!("norm: {}", e)))?;
        let norm_bias = global.final_norm_bias.as_ref().and_then(|n| {
            if weight_source.has_tensor(n) { weight_source.load_tensor(n).ok() } else { None }
        });

        let lm_head = if config.tie_word_embeddings {
            None
        } else if weight_source.has_tensor(&global.lm_head) {
            Some(
                weight_source
                    .load_compact(&global.lm_head)
                    .map_err(|e| ModelError::WeightNotFound(format!("lm_head: {}", e)))?,
            )
        } else {
            None
        };

        let pos_embed = if let Some(ref name) = global.position_embed {
            if weight_source.has_tensor(name) {
                Some(weight_source.load_tensor(name)
                    .map_err(|e| ModelError::WeightNotFound(format!("pos_embed: {}", e)))?)
            } else {
                None
            }
        } else {
            None
        };

        let num_layers = config.num_hidden_layers;

        // Estimate per-layer size in COMPACT (bf16) format.
        let d = config.hidden_size;
        let kv = config.num_key_value_heads * config.head_dim();
        let ff = config.intermediate_size;

        // Attention projections: q(d*d) + k(kv*d) + v(kv*d) + o(d*d)
        let attn_params = d * d + kv * d + kv * d + d * d;
        // FFN projections depend on type
        let ffn_params = match config.ffn_type {
            FfnType::SwiGLU | FfnType::GeGLU => d * ff * 3,  // gate + up + down
            FfnType::GELU | FfnType::ReLU => d * ff * 2,     // up + down
        };
        let norm_params = 2 * d;
        let layer_params = attn_params + ffn_params + norm_params;
        // bf16 projections (2 bytes) + f32 norms (4 bytes for 2*d params)
        let layer_size_bytes = (layer_params - norm_params) * 2 + norm_params * 4;

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

        info!(
            "Paged model: {} layers, {:.1} MB/layer (bf16), hot budget {:.0} MB, warm budget {:.0} MB",
            num_layers,
            layer_size_bytes as f64 / 1024.0 / 1024.0,
            paged_config.hot_budget_bytes as f64 / 1024.0 / 1024.0,
            paged_config.warm_budget_bytes as f64 / 1024.0 / 1024.0,
        );

        let hot_layers = paged_config.hot_budget_bytes / layer_size_bytes.max(1);
        let warm_layers = paged_config.warm_budget_bytes / layer_size_bytes.max(1);

        let mut layer_tiers = vec![LayerTier::Cold; num_layers];
        for i in 0..num_layers.min(hot_layers) {
            layer_tiers[i] = LayerTier::Hot;
        }
        for i in hot_layers..num_layers.min(hot_layers + warm_layers) {
            layer_tiers[i] = LayerTier::Warm;
        }

        // Hot-only mode: compute which layers are active.
        let active_layer_set = if paged_config.hot_only_mode {
            let max_active = if let Some(n) = paged_config.active_layers {
                n.min(num_layers)
            } else {
                (hot_layers + warm_layers).min(num_layers)
            };
            let set = Self::select_active_layers(num_layers, max_active);
            let active_count = set.iter().filter(|&&x| x).count();
            info!(
                "Hot-only mode: {}/{} layers active (evenly spaced), {} skipped",
                active_count, num_layers, num_layers - active_count,
            );
            layer_tiers = vec![LayerTier::Cold; num_layers];
            let mut hot_remaining = hot_layers;
            let mut warm_remaining = warm_layers;
            for i in 0..num_layers {
                if set[i] {
                    if hot_remaining > 0 {
                        layer_tiers[i] = LayerTier::Hot;
                        hot_remaining -= 1;
                    } else if warm_remaining > 0 {
                        layer_tiers[i] = LayerTier::Warm;
                        warm_remaining -= 1;
                    }
                }
            }
            set
        } else if let Some(n) = paged_config.active_layers {
            let max_active = n.min(num_layers);
            let set = Self::select_active_layers(num_layers, max_active);
            let active_count = set.iter().filter(|&&x| x).count();
            info!(
                "Active layers limited to {}/{} (evenly spaced)",
                active_count, num_layers,
            );
            set
        } else {
            vec![true; num_layers]
        };

        info!(
            "Initial tiers: {} hot, {} warm, {} cold (bf16 compact weights)",
            layer_tiers.iter().filter(|&&t| t == LayerTier::Hot).count(),
            layer_tiers.iter().filter(|&&t| t == LayerTier::Warm).count(),
            layer_tiers.iter().filter(|&&t| t == LayerTier::Cold).count(),
        );

        Ok(PagedModel {
            config,
            paged_config,
            embed_tokens,
            norm,
            norm_bias,
            lm_head,
            pos_embed,
            layers: (0..num_layers).map(|_| None).collect(),
            layer_tiers,
            layer_access: vec![0; num_layers],
            weight_source,
            rope_freqs,
            kv_cache,
            pos: 0,
            tick: 0,
            stats: PagingStats::default(),
            hot_used_bytes: 0,
            warm_used_bytes: 0,
            layer_size_bytes,
            layer_importance: vec![0.0; num_layers],
            active_layer_set,
            layers_skipped: 0,
            awq_calibration: None,
            layer_quant_assignments: None,
            #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
            gpu_state: None,
        })
    }

    /// Initialize GPU inference state for hot-tier layers.
    ///
    /// Must be called after model loading and (optionally) profiling. Uploads
    /// hot-layer weights to VRAM and builds the GPU KV cache. On CPU-only
    /// builds this is a no-op.
    ///
    /// After `init_gpu()` returns, `forward_single` and `forward_prefill`
    /// automatically route hot-layer ops to the GPU.
    pub fn init_gpu(&mut self) -> Result<(), ModelError> {
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        {
            use crate::device::Device;
            use crate::gpu_layer::build_gpu_state;

            let device = Device::resolve(&self.paged_config.device);
            if device.is_cpu() {
                info!("[GPU] Device resolved to CPU — skipping GPU init");
                return Ok(());
            }

            let candle_dev = device.to_candle()
                .map_err(|e| ModelError::WeightNotFound(format!("GPU device init: {}", e)))?;

            // Determine which layers run on GPU (hot tier, active).
            // Ensure all hot layers are loaded first.
            let hot_set: Vec<bool> = (0..self.config.num_hidden_layers)
                .map(|i| self.layer_tiers[i] == LayerTier::Hot && self.active_layer_set[i])
                .collect();

            // Load any hot layers that aren't in memory yet.
            for i in 0..self.config.num_hidden_layers {
                if hot_set[i] && self.layers[i].is_none() {
                    self.load_layer_direct(i)?;
                }
            }

            let gpu_hot_count = hot_set.iter().filter(|&&x| x).count();
            info!(
                "[GPU] Initializing GPU inference: {} hot layers → {}, uploading weights",
                gpu_hot_count, device,
            );

            let state = build_gpu_state(
                candle_dev,
                &self.layers,
                &hot_set,
                self.rope_freqs.as_ref(),
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                self.config.ffn_type,
                self.config.norm_type,
                self.config.norm_eps,
            ).map_err(|e| ModelError::WeightNotFound(format!("GPU state init: {}", e)))?;

            info!("[GPU] GPU inference ready: {} layers on device", gpu_hot_count);
            self.gpu_state = Some(state);
        }
        Ok(())
    }

    /// Profile layer importance in streaming mode.
    ///
    /// **Batch-first, layer-outer loop**: each layer is loaded exactly once and
    /// all calibration tokens are processed through it in a single batched
    /// forward pass before the layer is evicted.  This is O(n_layers) loads
    /// rather than the old O(n_layers × n_tokens) approach.
    ///
    /// When compiled with a GPU feature (`cuda`, `hip`, or `metal`) and a GPU
    /// is detected at runtime, the five large matrix multiplications per layer
    /// (Q, V, gate, up, down projections) are dispatched to the GPU via
    /// candle-core.  Norms and activations stay on CPU (they are negligible).
    /// On any GPU failure the function falls back to the CPU batch path silently.
    pub fn profile_layer_importance(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f64>, ModelError> {
        let num_layers = self.config.num_hidden_layers;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let n_tok = prompt_tokens.len();

        let mut scores = vec![0.0f64; num_layers];
        let mut layer_hidden_saliency: Vec<Vec<f64>> =
            (0..num_layers).map(|_| vec![0.0f64; hidden_size]).collect();
        let mut layer_intermediate_saliency: Vec<Vec<f64>> =
            (0..num_layers).map(|_| vec![0.0f64; intermediate_size]).collect();

        // Resolve device once before the layer loop — honours the PagedConfig
        // device string so callers can pin to "cpu", "cuda:0", etc.
        let compute_device = crate::device::Device::resolve(&self.paged_config.device);
        info!(
            "Streaming profiler: {} layers, {} tokens, batch mode (device={})",
            num_layers, n_tok, compute_device,
        );

        // Pre-embed all calibration tokens into a single [n_tok, hidden_size] batch.
        // Each token's hidden state evolves independently through the layers.
        let mut flat: Vec<f32> = Vec::with_capacity(n_tok * hidden_size);
        for &tok in prompt_tokens {
            flat.extend_from_slice(&self.embed_tokens.row_to_f32(tok as usize));
        }
        let mut hidden_batch = Tensor::new(flat, vec![n_tok, hidden_size]).unwrap();

        for i in 0..num_layers {
            // Load this layer ONCE — process every token before evicting.
            self.evict_all_layers();
            self.load_layer_direct(i)?;

            // ── GPU path ──────────────────────────────────────────────────────
            #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
            if compute_device.is_gpu() {
                match profile_layer_gpu(
                    &hidden_batch,
                    self.layers[i].as_ref().unwrap(),
                    &self.config,
                    &compute_device,
                ) {
                    Ok((new_hidden, h_sal, ffn_sal, layer_score)) => {
                        hidden_batch = new_hidden;
                        scores[i] = layer_score;
                        for j in 0..h_sal.len().min(hidden_size) {
                            layer_hidden_saliency[i][j] += h_sal[j] as f64;
                        }
                        for j in 0..ffn_sal.len().min(intermediate_size) {
                            layer_intermediate_saliency[i][j] += ffn_sal[j] as f64;
                        }
                        self.layers[i] = None;
                        self.warm_used_bytes =
                            self.warm_used_bytes.saturating_sub(self.layer_size_bytes);
                        self.hot_used_bytes =
                            self.hot_used_bytes.saturating_sub(self.layer_size_bytes);
                        continue;
                    }
                    Err(e) => {
                        log::warn!(
                            "GPU profiling failed for layer {} ({}); falling back to CPU",
                            i, e
                        );
                    }
                }
            }

            // ── CPU batch path ────────────────────────────────────────────────
            {
                let layer = self.layers[i].as_ref().unwrap();

                // Norm over the whole batch: [n_tok, hidden_size]
                let normed = apply_norm_batch(
                    &hidden_batch,
                    &layer.input_norm,
                    layer.input_norm_bias.as_ref(),
                    self.config.norm_type,
                    self.config.norm_eps,
                );

                // Hidden saliency: sum |normed[tok][j]| over tokens.
                for (flat_j, &v) in normed.data().iter().enumerate() {
                    layer_hidden_saliency[i][flat_j % hidden_size] += v.abs() as f64;
                }

                // Q/V projection — batched matmul [n_tok, d_h] × [d_q, d_h]^T
                let q_out = ws_linear(&normed, &layer.q_proj); // [n_tok, d_q]
                let v_out = ws_linear(&normed, &layer.v_proj); // [n_tok, d_v]
                let d_q = q_out.shape()[1];
                let d_v = v_out.shape()[1];
                let attn_l2_sum: f64 = (0..n_tok)
                    .map(|t| {
                        let q_sq: f64 = q_out.data()[t * d_q..(t + 1) * d_q]
                            .iter().map(|&x| (x as f64) * (x as f64)).sum();
                        let v_sq: f64 = v_out.data()[t * d_v..(t + 1) * d_v]
                            .iter().map(|&x| (x as f64) * (x as f64)).sum();
                        (q_sq + v_sq).sqrt()
                    })
                    .sum();

                // FFN — apply_ffn dispatches to ws_linear (batched) for 2-D input
                let normed2 = apply_norm_batch(
                    &hidden_batch,
                    &layer.post_attn_norm,
                    layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type,
                    self.config.norm_eps,
                );
                let ffn_out = apply_ffn(&normed2, layer, self.config.ffn_type);
                let d_out = ffn_out.shape()[1];

                let ffn_l2_sum: f64 = (0..n_tok)
                    .map(|t| {
                        ffn_out.data()[t * d_out..(t + 1) * d_out]
                            .iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
                    })
                    .sum();

                // Intermediate saliency: sum |ffn_out[tok][j]| over tokens.
                for (flat_j, &v) in ffn_out.data().iter().enumerate() {
                    let col = flat_j % d_out;
                    if col < intermediate_size {
                        layer_intermediate_saliency[i][col] += v.abs() as f64;
                    }
                }

                scores[i] = (attn_l2_sum + ffn_l2_sum) / n_tok as f64;

                // Residual update: hidden_batch += ffn_out  (both [n_tok, hidden_size])
                hidden_batch.add_inplace(&ffn_out);
            }

            self.layers[i] = None;
            self.warm_used_bytes =
                self.warm_used_bytes.saturating_sub(self.layer_size_bytes);
            self.hot_used_bytes =
                self.hot_used_bytes.saturating_sub(self.layer_size_bytes);
        }

        // Saliency was accumulated as sums; divide by n_tok to get per-channel means.
        let sc = n_tok as f64;
        for i in 0..num_layers {
            for j in 0..hidden_size      { layer_hidden_saliency[i][j]        /= sc; }
            for j in 0..intermediate_size{ layer_intermediate_saliency[i][j]  /= sc; }
        }

        // Store AWQ calibration.
        let mut layer_saliency = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layer_saliency.push(ChannelSaliency {
                hidden_saliency:       layer_hidden_saliency[i].iter().map(|&v| v as f32).collect(),
                intermediate_saliency: layer_intermediate_saliency[i].iter().map(|&v| v as f32).collect(),
                sample_count: n_tok,
            });
        }
        self.awq_calibration  = Some(AwqCalibration { layer_saliency });
        self.layer_importance = scores.clone();

        info!("Layer importance scores:");
        for (i, &s) in scores.iter().enumerate() {
            info!("  Layer {:2}: {:.4}", i, s);
        }

        let _ = crate::importance_cache::ImportanceCache::save(
            self.weight_source.model_dir(),
            &scores,
        );

        if let QuantMode::ProfileGuided(target_bpw) = self.paged_config.quant_mode {
            let d  = self.config.hidden_size;
            let kv = self.config.num_key_value_heads * self.config.head_dim();
            let ff = self.config.intermediate_size;
            let layer_params = d * d + kv * d + kv * d + d * d + d * ff * 3;
            let assignments = allocate_bits(&scores, target_bpw, layer_params);
            info!("Profile-guided bit allocation (target {:.1} bpw):", target_bpw);
            for (i, mode) in assignments.iter().enumerate() {
                info!("  Layer {:2}: {}", i, mode);
            }
            self.layer_quant_assignments = Some(assignments);
        }

        Ok(scores)
    }

    /// Profile layer importance and simultaneously compare four scoring methods in
    /// a single streaming pass with no extra layer loads.
    ///
    /// The four methods computed are:
    /// - **Proxy** (current production scorer): `‖FFN(x)‖₂ + ‖[Qx, Vx]‖₂`
    /// - **FFN-only**: `‖FFN(x)‖₂`  — is the Q/V proxy overhead justified?
    /// - **Attn-proxy-only**: `‖[Qx, Vx]‖₂`
    /// - **Input-L2**: `‖RMSNorm(x)‖₂`  — zero extra cost beyond the existing norm
    ///
    /// Returns `(proxy_scores, comparison)`.  `proxy_scores` is identical to the
    /// output of `profile_layer_importance` and is stored in `self.layer_importance`.
    ///
    /// Uses the same layer-outer batched loop as `profile_layer_importance`.
    pub fn profile_layer_importance_detailed(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<(Vec<f64>, ScoringComparison), ModelError> {
        let num_layers = self.config.num_hidden_layers;
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let n_tok = prompt_tokens.len();

        let mut scores_proxy  = vec![0.0f64; num_layers];
        let mut scores_ffn    = vec![0.0f64; num_layers];
        let mut scores_attn   = vec![0.0f64; num_layers];
        let mut scores_input  = vec![0.0f64; num_layers];

        let mut layer_hidden_saliency: Vec<Vec<f64>> =
            (0..num_layers).map(|_| vec![0.0f64; hidden_size]).collect();
        let mut layer_intermediate_saliency: Vec<Vec<f64>> =
            (0..num_layers).map(|_| vec![0.0f64; intermediate_size]).collect();

        let loop_start    = Instant::now();
        let mut attn_proxy_ns: u128 = 0;

        #[allow(unused_variables)]
        let compute_device = crate::device::Device::resolve(&self.paged_config.device);

        // Pre-embed all tokens into a [n_tok, hidden_size] batch.
        let mut flat: Vec<f32> = Vec::with_capacity(n_tok * hidden_size);
        for &tok in prompt_tokens {
            flat.extend_from_slice(&self.embed_tokens.row_to_f32(tok as usize));
        }
        let mut hidden_batch = Tensor::new(flat, vec![n_tok, hidden_size]).unwrap();

        for i in 0..num_layers {
            self.evict_all_layers();
            self.load_layer_direct(i)?;

            // GPU path — same as profile_layer_importance (no scorer timing on GPU).
            #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
            if compute_device.is_gpu() {
                match profile_layer_gpu(
                    &hidden_batch,
                    self.layers[i].as_ref().unwrap(),
                    &self.config,
                    &compute_device,
                ) {
                    Ok((new_hidden, h_sal, ffn_sal, layer_score)) => {
                        hidden_batch = new_hidden;
                        // For the scorer comparison, attribute all to proxy/ffn (attn=0 on GPU path).
                        scores_proxy[i] = layer_score;
                        scores_ffn[i]   = layer_score;
                        for j in 0..h_sal.len().min(hidden_size) {
                            layer_hidden_saliency[i][j] += h_sal[j] as f64;
                        }
                        for j in 0..ffn_sal.len().min(intermediate_size) {
                            layer_intermediate_saliency[i][j] += ffn_sal[j] as f64;
                        }
                        self.layers[i] = None;
                        self.warm_used_bytes =
                            self.warm_used_bytes.saturating_sub(self.layer_size_bytes);
                        self.hot_used_bytes =
                            self.hot_used_bytes.saturating_sub(self.layer_size_bytes);
                        continue;
                    }
                    Err(e) => {
                        log::warn!(
                            "GPU profiling failed for layer {} ({}); falling back to CPU",
                            i, e
                        );
                    }
                }
            }

            // CPU batch path.
            {
                let layer = self.layers[i].as_ref().unwrap();

                let normed = apply_norm_batch(
                    &hidden_batch, &layer.input_norm, layer.input_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps,
                );

                // Input-L2: mean per-token norm of the normalized hidden state.
                let input_l2_sum: f64 = (0..n_tok).map(|t| {
                    normed.data()[t * hidden_size..(t + 1) * hidden_size]
                        .iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
                }).sum();
                scores_input[i] = input_l2_sum / n_tok as f64;

                for (flat_j, &v) in normed.data().iter().enumerate() {
                    layer_hidden_saliency[i][flat_j % hidden_size] += v.abs() as f64;
                }

                let normed2 = apply_norm_batch(
                    &hidden_batch, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps,
                );
                let ffn_out = apply_ffn(&normed2, layer, self.config.ffn_type);
                let d_out = ffn_out.shape()[1];

                let ffn_l2_sum: f64 = (0..n_tok).map(|t| {
                    ffn_out.data()[t * d_out..(t + 1) * d_out]
                        .iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
                }).sum();
                scores_ffn[i] = ffn_l2_sum / n_tok as f64;

                for (flat_j, &v) in ffn_out.data().iter().enumerate() {
                    let col = flat_j % d_out;
                    if col < intermediate_size {
                        layer_intermediate_saliency[i][col] += v.abs() as f64;
                    }
                }

                // Q/V projection (timed for overhead measurement).
                let attn_start = Instant::now();
                let q_out = ws_linear(&normed, &layer.q_proj);
                let v_out = ws_linear(&normed, &layer.v_proj);
                attn_proxy_ns += attn_start.elapsed().as_nanos();

                let d_q = q_out.shape()[1];
                let d_v = v_out.shape()[1];
                let attn_l2_sum: f64 = (0..n_tok).map(|t| {
                    let q_sq: f64 = q_out.data()[t * d_q..(t + 1) * d_q]
                        .iter().map(|&x| (x as f64) * (x as f64)).sum();
                    let v_sq: f64 = v_out.data()[t * d_v..(t + 1) * d_v]
                        .iter().map(|&x| (x as f64) * (x as f64)).sum();
                    (q_sq + v_sq).sqrt()
                }).sum();
                scores_attn[i]  = attn_l2_sum / n_tok as f64;
                scores_proxy[i] = scores_ffn[i] + scores_attn[i];

                hidden_batch.add_inplace(&ffn_out);
                self.layers[i] = None;
                self.warm_used_bytes =
                    self.warm_used_bytes.saturating_sub(self.layer_size_bytes);
                self.hot_used_bytes =
                    self.hot_used_bytes.saturating_sub(self.layer_size_bytes);
            }
        }

        let total_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
        let attn_ms  = attn_proxy_ns as f64 / 1_000_000.0;

        let sc = n_tok as f64;
        for i in 0..num_layers {
            for j in 0..hidden_size      { layer_hidden_saliency[i][j]        /= sc; }
            for j in 0..intermediate_size{ layer_intermediate_saliency[i][j]  /= sc; }
        }

        let mut layer_saliency = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layer_saliency.push(ChannelSaliency {
                hidden_saliency:       layer_hidden_saliency[i].iter().map(|&v| v as f32).collect(),
                intermediate_saliency: layer_intermediate_saliency[i].iter().map(|&v| v as f32).collect(),
                sample_count: n_tok,
            });
        }
        self.awq_calibration  = Some(AwqCalibration { layer_saliency });
        self.layer_importance = scores_proxy.clone();

        if let QuantMode::ProfileGuided(target_bpw) = self.paged_config.quant_mode {
            let d  = self.config.hidden_size;
            let kv = self.config.num_key_value_heads * self.config.head_dim();
            let ff = self.config.intermediate_size;
            let layer_params = d * d + kv * d + kv * d + d * d + d * ff * 3;
            let assignments = allocate_bits(&scores_proxy, target_bpw, layer_params);
            self.layer_quant_assignments = Some(assignments);
        }

        // Rank correlations and top-k overlaps vs the proxy.
        let k = (num_layers / 2).max(1);
        let comparison = ScoringComparison {
            tau_ffn_vs_proxy:     kendall_tau(&scores_ffn,   &scores_proxy),
            tau_attn_vs_proxy:    kendall_tau(&scores_attn,  &scores_proxy),
            tau_input_vs_proxy:   kendall_tau(&scores_input, &scores_proxy),
            topk_ffn_vs_proxy:    top_k_overlap_vecs(&scores_ffn,   &scores_proxy, k),
            topk_attn_vs_proxy:   top_k_overlap_vecs(&scores_attn,  &scores_proxy, k),
            topk_input_vs_proxy:  top_k_overlap_vecs(&scores_input, &scores_proxy, k),
            attn_proxy_cost_ms:   attn_ms,
            total_time_ms:        total_ms,
            attn_proxy_fraction:  if total_ms > 0.0 { attn_ms / total_ms } else { 0.0 },
            scores_proxy:         scores_proxy.clone(),
            scores_ffn_only:      scores_ffn,
            scores_attn_proxy:    scores_attn,
            scores_input_l2:      scores_input,
        };

        Ok((scores_proxy, comparison))
    }

    /// Evict all loaded layers to free memory.
    fn evict_all_layers(&mut self) {
        for i in 0..self.layers.len() {
            if self.layers[i].is_some() {
                self.layers[i] = None;
            }
        }
        self.hot_used_bytes = 0;
        self.warm_used_bytes = 0;
    }

    /// Load a single layer directly without LRU or tier logic, using generic block loader.
    fn load_layer_direct(&mut self, layer_idx: usize) -> Result<(), ModelError> {
        let mut block = generic_model::load_generic_block(&self.weight_source, &self.config, layer_idx)
            .map_err(|e| ModelError::WeightNotFound(format!("layer {}: {}", layer_idx, e)))?;
        // Quantize on load if configured — loads bf16, quantizes, drops bf16.
        match self.paged_config.quant_mode {
            QuantMode::Q4 => block.quantize_q4(),
            QuantMode::Q8 => block.quantize_q8(),
            QuantMode::Q3 => block.quantize_q3(),
            QuantMode::Q2 => block.quantize_q2(),
            QuantMode::Q1 => block.quantize_q1(),
            QuantMode::ProfileGuided(_) => {
                if let Some(ref assignments) = self.layer_quant_assignments {
                    let layer_mode = assignments[layer_idx];
                    let awq = self.awq_calibration.as_ref()
                        .map(|c| &c.layer_saliency[layer_idx]);
                    block.quantize_adaptive(layer_mode, awq);
                }
            }
            QuantMode::None => {}
        }
        self.layers[layer_idx] = Some(block);
        self.warm_used_bytes += self.layer_size_bytes;
        Ok(())
    }

    /// Switch to hot-only mode using profiled importance to select the best layers.
    /// Must call `profile_layer_importance` first.
    pub fn apply_profiled_hot_only(&mut self, active_count: usize) {
        let num_layers = self.config.num_hidden_layers;
        let n = active_count.min(num_layers);

        // Rank layers by importance (descending).
        let mut ranked: Vec<(usize, f64)> = self.layer_importance.iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Always include first and last layer, then fill with highest importance.
        let mut selected = vec![false; num_layers];
        selected[0] = true;
        if num_layers > 1 {
            selected[num_layers - 1] = true;
        }
        let mut count = selected.iter().filter(|&&x| x).count();

        for (idx, _score) in &ranked {
            if count >= n {
                break;
            }
            if !selected[*idx] {
                selected[*idx] = true;
                count += 1;
            }
        }

        // Reassign tiers.
        let hot_layers = self.paged_config.hot_budget_bytes / self.layer_size_bytes.max(1);
        let warm_layers = self.paged_config.warm_budget_bytes / self.layer_size_bytes.max(1);

        self.layer_tiers = vec![LayerTier::Cold; num_layers];
        let mut hot_remaining = hot_layers;
        let mut warm_remaining = warm_layers;
        for i in 0..num_layers {
            if selected[i] {
                if hot_remaining > 0 {
                    self.layer_tiers[i] = LayerTier::Hot;
                    hot_remaining -= 1;
                } else if warm_remaining > 0 {
                    self.layer_tiers[i] = LayerTier::Warm;
                    warm_remaining -= 1;
                }
            }
        }

        // Evict layers no longer active.
        for i in 0..num_layers {
            if !selected[i] && self.layers[i].is_some() {
                self.layers[i] = None;
                self.warm_used_bytes = self.warm_used_bytes.saturating_sub(self.layer_size_bytes);
            }
        }

        // Reset KV cache and position since profiling consumed tokens.
        self.kv_cache.clear();
        self.pos = 0;
        self.stats = PagingStats::default();
        self.layers_skipped = 0;

        self.active_layer_set = selected;
        let active = self.active_layer_set.iter().filter(|&&x| x).count();
        let selected_indices: Vec<usize> = self.active_layer_set.iter()
            .enumerate()
            .filter(|(_, &x)| x)
            .map(|(i, _)| i)
            .collect();
        info!(
            "Profiled hot-only: {}/{} layers active (by importance), layers: {:?}",
            active, num_layers, selected_indices,
        );
    }

    /// Inject externally-computed importance scores, bypassing the profiling pass.
    ///
    /// This enables the "profile once on a large machine, run on a small device"
    /// workflow: profile on a 16 GB cloud instance, export the scores, then import
    /// them on a 2–4 GB edge device so layer selection is still optimal without
    /// needing to run a full forward pass at the constrained site.
    ///
    /// Panics if `scores.len()` does not match the model's layer count.
    pub fn inject_importance_scores(&mut self, scores: Vec<f64>) {
        assert_eq!(
            scores.len(),
            self.config.num_hidden_layers,
            "injected profile has {} scores but model has {} layers",
            scores.len(),
            self.config.num_hidden_layers
        );
        info!(
            "Injected {} pre-computed importance scores (profiling pass skipped)",
            scores.len()
        );
        self.layer_importance = scores;
        // AWQ calibration is not available from an injected profile; clear it so
        // profile-guided quantization (config C) falls back to pure bit allocation.
        self.awq_calibration = None;
    }

    /// Select which layers to keep active using evenly-spaced selection.
    /// Always includes the first and last layer for best quality.
    fn select_active_layers(total: usize, active_count: usize) -> Vec<bool> {
        let mut set = vec![false; total];
        if active_count == 0 || total == 0 {
            return set;
        }
        if active_count >= total {
            return vec![true; total];
        }

        // Always include first and last.
        set[0] = true;
        if total > 1 {
            set[total - 1] = true;
        }

        // Spread remaining evenly across the middle.
        let remaining = active_count.saturating_sub(2);
        if remaining > 0 && total > 2 {
            let middle = total - 2;
            for i in 0..remaining {
                let idx = 1 + (i * middle) / remaining;
                set[idx] = true;
            }
        }

        // If we still don't have enough due to rounding, fill from front.
        let mut count = set.iter().filter(|&&x| x).count();
        if count < active_count {
            for i in 0..total {
                if count >= active_count {
                    break;
                }
                if !set[i] {
                    set[i] = true;
                    count += 1;
                }
            }
        }

        set
    }

    fn ensure_layer_loaded(&mut self, layer_idx: usize) -> Result<(), ModelError> {
        self.ensure_layer_loaded_protecting(layer_idx, layer_idx)
    }

    fn ensure_layer_loaded_protecting(&mut self, layer_idx: usize, protect_layer: usize) -> Result<(), ModelError> {
        if self.layers[layer_idx].is_some() {
            self.stats.page_hits += 1;
            return Ok(());
        }

        self.stats.page_faults += 1;
        let load_start = Instant::now();

        self.evict_if_needed(protect_layer)?;

        // Load layer weights using the generic block loader.
        let mut block = generic_model::load_generic_block(&self.weight_source, &self.config, layer_idx)
            .map_err(|e| ModelError::WeightNotFound(format!("layer {}: {}", layer_idx, e)))?;
        // Quantize on load if configured.
        match self.paged_config.quant_mode {
            QuantMode::Q4 => block.quantize_q4(),
            QuantMode::Q8 => block.quantize_q8(),
            QuantMode::Q3 => block.quantize_q3(),
            QuantMode::Q2 => block.quantize_q2(),
            QuantMode::Q1 => block.quantize_q1(),
            QuantMode::ProfileGuided(_) => {
                if let Some(ref assignments) = self.layer_quant_assignments {
                    let layer_mode = assignments[layer_idx];
                    let awq = self.awq_calibration.as_ref()
                        .map(|c| &c.layer_saliency[layer_idx]);
                    block.quantize_adaptive(layer_mode, awq);
                }
            }
            QuantMode::None => {}
        }

        self.layers[layer_idx] = Some(block);
        self.stats.layers_loaded += 1;

        match self.layer_tiers[layer_idx] {
            LayerTier::Hot => self.hot_used_bytes += self.layer_size_bytes,
            LayerTier::Warm => self.warm_used_bytes += self.layer_size_bytes,
            LayerTier::Cold => {
                self.layer_tiers[layer_idx] = LayerTier::Warm;
                self.warm_used_bytes += self.layer_size_bytes;
            }
        }

        let load_time = load_start.elapsed().as_secs_f64() * 1000.0;
        self.stats.total_load_time_ms += load_time;
        debug!("Loaded layer {} (bf16 compact) in {:.1}ms", layer_idx, load_time);

        Ok(())
    }

    fn evict_if_needed(&mut self, protect_layer: usize) -> Result<(), ModelError> {
        let total_budget = self.paged_config.hot_budget_bytes + self.paged_config.warm_budget_bytes;
        let current_used = self.hot_used_bytes + self.warm_used_bytes;

        if current_used + self.layer_size_bytes <= total_budget {
            return Ok(());
        }

        let mut lru_idx = None;
        let mut lru_tick = u64::MAX;

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_some() && i != protect_layer && self.layer_access[i] < lru_tick {
                lru_tick = self.layer_access[i];
                lru_idx = Some(i);
            }
        }

        if let Some(idx) = lru_idx {
            let evict_start = Instant::now();
            self.layers[idx] = None;

            match self.layer_tiers[idx] {
                LayerTier::Hot => self.hot_used_bytes = self.hot_used_bytes.saturating_sub(self.layer_size_bytes),
                LayerTier::Warm => self.warm_used_bytes = self.warm_used_bytes.saturating_sub(self.layer_size_bytes),
                _ => {}
            }
            self.layer_tiers[idx] = LayerTier::Cold;
            self.stats.layers_evicted += 1;
            self.stats.total_evict_time_ms += evict_start.elapsed().as_secs_f64() * 1000.0;
            debug!("Evicted layer {}", idx);
        }

        Ok(())
    }

    fn prefetch_layers(&mut self, current_layer: usize) -> Result<(), ModelError> {
        let mut prefetched = 0;
        let mut target = current_layer + 1;
        while prefetched < self.paged_config.prefetch_ahead && target < self.config.num_hidden_layers {
            if self.active_layer_set[target] && self.layers[target].is_none() {
                self.ensure_layer_loaded_protecting(target, current_layer)?;
                self.stats.prefetch_hits += 1;
                prefetched += 1;
            }
            target += 1;
        }
        Ok(())
    }

    pub fn forward_single(&mut self, token_id: u32) -> Result<Tensor, ModelError> {
        let pos = self.pos;
        self.tick += 1;

        let emb_data = self.embed_tokens.row_to_f32(token_id as usize);
        let hidden_size = self.config.hidden_size;
        let mut hidden = Tensor::new(emb_data, vec![hidden_size]).unwrap();

        // Add learned position embeddings if present (GPT-2 style).
        if let Some(ref pos_emb) = self.pos_embed {
            let pe = pos_emb.embedding(pos as u32);
            hidden.add_inplace(&pe);
        }

        // GPU residual accumulator — Some(_) while the hidden state lives on GPU.
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        let mut gpu_hidden: Option<candle_core::Tensor> = None;

        let parallel = self.config.parallel_attn_ffn;

        for i in 0..self.config.num_hidden_layers {
            // Hot-only mode: skip inactive layers — pass residual through unchanged.
            if !self.active_layer_set[i] {
                self.layers_skipped += 1;
                continue;
            }

            // ── GPU hot-layer path ────────────────────────────────────────────
            #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
            {
                let is_gpu_layer = self.gpu_state.as_ref()
                    .map(|g| g.layer_weights[i].is_some())
                    .unwrap_or(false);

                if is_gpu_layer {
                    let state = self.gpu_state.as_ref().unwrap();
                    // Upload CPU hidden to GPU (as BF16, matching weight dtype).
                    if gpu_hidden.is_none() {
                        gpu_hidden = Some(
                            candle_core::Tensor::from_slice(
                                hidden.data(), (1, hidden_size), &state.device,
                            ).and_then(|t| t.to_dtype(candle_core::DType::F16))
                            .map_err(|e| ModelError::WeightNotFound(
                                format!("GPU upload layer {}: {}", i, e)
                            ))?
                        );
                    }

                    let importance = self.layer_importance.get(i).copied().unwrap_or(0.0);
                    let result = crate::gpu_layer::gpu_layer_forward_decode(
                        gpu_hidden.take().unwrap(),
                        i,
                        self.gpu_state.as_mut().unwrap(),
                        pos,
                        parallel,
                        importance,
                    ).map_err(|e| ModelError::WeightNotFound(
                        format!("GPU layer {} forward: {}", i, e)
                    ))?;
                    gpu_hidden = Some(result);
                    continue;
                }

                // Transitioning from GPU to CPU: download residual (convert BF16 → F32).
                if let Some(gh) = gpu_hidden.take() {
                    let data = crate::gpu_layer::gpu_to_cpu_vec(
                        gh.to_dtype(candle_core::DType::F32)
                          .map_err(|e| ModelError::WeightNotFound(format!("GPU dtype convert: {}", e)))?
                    ).map_err(|e| ModelError::WeightNotFound(
                        format!("GPU download before layer {}: {}", i, e)
                    ))?;
                    hidden = Tensor::new(data, vec![hidden_size]).unwrap();
                }
            }

            // ── CPU layer path (existing code) ────────────────────────────────
            self.ensure_layer_loaded(i)?;
            self.layer_access[i] = self.tick;

            if i + 1 < self.config.num_hidden_layers {
                self.prefetch_layers(i)?;
            }

            let layer = self.layers[i].as_ref().unwrap();

            let normed = apply_norm(&hidden, &layer.input_norm, layer.input_norm_bias.as_ref(),
                self.config.norm_type, self.config.norm_eps);

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
                &normed,
                &attn_weights,
                self.rope_freqs.as_ref(),
                &mut self.kv_cache,
                i,
                pos,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
            );

            // Apply output projection bias if present.
            let attn_out = if let Some(ref bias) = layer.o_bias {
                attn_out.add(bias)
            } else {
                attn_out
            };

            if parallel {
                // Parallel attention + FFN (GPT-NeoX, Falcon style).
                let ffn_normed = apply_norm(&hidden, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps);
                let ffn_out = apply_ffn(&ffn_normed, layer, self.config.ffn_type);
                hidden.add_inplace(&attn_out);
                hidden.add_inplace(&ffn_out);
            } else {
                // Sequential attention then FFN.
                let mut h = hidden.add(&attn_out);
                let normed2 = apply_norm(&h, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps);
                let ffn_out = apply_ffn(&normed2, layer, self.config.ffn_type);
                h.add_inplace(&ffn_out);
                hidden = h;
            }
        }

        // Download GPU residual after all layers (BF16 → F32).
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        if let Some(gh) = gpu_hidden {
            let data = crate::gpu_layer::gpu_to_cpu_vec(
                gh.to_dtype(candle_core::DType::F32)
                  .map_err(|e| ModelError::WeightNotFound(format!("GPU dtype convert final: {}", e)))?
            ).map_err(|e| ModelError::WeightNotFound(
                format!("GPU download final: {}", e)
            ))?;
            hidden = Tensor::new(data, vec![hidden_size]).unwrap();
        }

        hidden = apply_norm(&hidden, &self.norm, self.norm_bias.as_ref(),
            self.config.norm_type, self.config.norm_eps);

        let logits = if let Some(ref lm) = self.lm_head {
            compact_linear_vec(&hidden, lm)
        } else {
            let logits_data = self.embed_tokens.matvec_f32(hidden.data());
            let vocab_size = self.config.vocab_size;
            Tensor::new(logits_data, vec![vocab_size]).unwrap()
        };

        self.pos += 1;
        self.kv_cache.advance();

        Ok(logits)
    }

    /// Internal: embed tokens, run all transformer layers, return full [seq_len, hidden_size]
    /// hidden-state matrix. Caller must update self.pos and self.kv_cache afterward.
    fn prefill_hidden_states(&mut self, token_ids: &[u32]) -> Result<Tensor, ModelError> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let start_pos = self.pos;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();

        // Build the [seq_len, hidden_size] embedding matrix.
        let mut embed_data = Vec::with_capacity(seq_len * hidden_size);
        for &tid in token_ids {
            let emb = self.embed_tokens.row_to_f32(tid as usize);
            embed_data.extend_from_slice(&emb);
        }
        let mut hidden = Tensor::new(embed_data, vec![seq_len, hidden_size]).unwrap();

        // Add learned position embeddings if present (GPT-2 style).
        if let Some(ref pos_emb) = self.pos_embed {
            let data = hidden.data_mut();
            for s in 0..seq_len {
                let pos = start_pos + s;
                let pe_row = pos_emb.row(pos);
                let offset = s * hidden_size;
                for j in 0..hidden_size {
                    data[offset + j] += pe_row[j];
                }
            }
        }

        self.tick += 1;

        // GPU residual accumulator for prefill (shape [seq_len, hidden_size] on GPU).
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        let mut gpu_hidden: Option<candle_core::Tensor> = None;

        let parallel = self.config.parallel_attn_ffn;

        for i in 0..self.config.num_hidden_layers {
            if !self.active_layer_set[i] {
                self.layers_skipped += 1;
                continue;
            }

            // ── GPU hot-layer path ────────────────────────────────────────────
            #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
            {
                let is_gpu_layer = self.gpu_state.as_ref()
                    .map(|g| g.layer_weights[i].is_some())
                    .unwrap_or(false);

                if is_gpu_layer {
                    let state = self.gpu_state.as_ref().unwrap();
                    // Upload prefill batch to GPU as BF16.
                    if gpu_hidden.is_none() {
                        gpu_hidden = Some(
                            candle_core::Tensor::from_slice(
                                hidden.data(), (seq_len, hidden_size), &state.device,
                            ).and_then(|t| t.to_dtype(candle_core::DType::F16))
                            .map_err(|e| ModelError::WeightNotFound(
                                format!("GPU upload prefill layer {}: {}", i, e)
                            ))?
                        );
                    }

                    let result = crate::gpu_layer::gpu_layer_forward_prefill(
                        gpu_hidden.take().unwrap(),
                        i,
                        self.gpu_state.as_mut().unwrap(),
                        start_pos,
                        parallel,
                    ).map_err(|e| ModelError::WeightNotFound(
                        format!("GPU prefill layer {} forward: {}", i, e)
                    ))?;
                    gpu_hidden = Some(result);
                    continue;
                }

                // Transitioning from GPU to CPU (BF16 → F32).
                if let Some(gh) = gpu_hidden.take() {
                    let data = crate::gpu_layer::gpu_to_cpu_vec(
                        gh.to_dtype(candle_core::DType::F32)
                          .map_err(|e| ModelError::WeightNotFound(format!("GPU dtype convert: {}", e)))?
                    ).map_err(|e| ModelError::WeightNotFound(
                        format!("GPU download prefill before layer {}: {}", i, e)
                    ))?;
                    hidden = Tensor::new(data, vec![seq_len, hidden_size]).unwrap();
                }
            }

            // ── CPU layer path ────────────────────────────────────────────────
            self.ensure_layer_loaded(i)?;
            self.layer_access[i] = self.tick;

            if i + 1 < self.config.num_hidden_layers {
                self.prefetch_layers(i)?;
            }

            let layer = self.layers[i].as_ref().unwrap();

            let normed = apply_norm_batch(&hidden, &layer.input_norm, layer.input_norm_bias.as_ref(),
                self.config.norm_type, self.config.norm_eps);

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

            // Apply output projection bias if present.
            let attn_out = if let Some(ref bias) = layer.o_bias {
                let mut result = attn_out;
                let data = result.data_mut();
                let h = bias.numel();
                let sl = data.len() / h;
                let b = bias.data();
                for s in 0..sl {
                    for j in 0..h {
                        data[s * h + j] += b[j];
                    }
                }
                result
            } else {
                attn_out
            };

            if parallel {
                let ffn_normed = apply_norm_batch(&hidden, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps);
                let ffn_out = apply_ffn(&ffn_normed, layer, self.config.ffn_type);
                hidden.add_inplace(&attn_out);
                hidden.add_inplace(&ffn_out);
            } else {
                let mut h = hidden.add(&attn_out);
                let normed2 = apply_norm_batch(&h, &layer.post_attn_norm, layer.post_attn_norm_bias.as_ref(),
                    self.config.norm_type, self.config.norm_eps);
                let ffn_out = apply_ffn(&normed2, layer, self.config.ffn_type);
                h.add_inplace(&ffn_out);
                hidden = h;
            }
        }

        // Download GPU residual after all layers (BF16 → F32).
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        if let Some(gh) = gpu_hidden {
            let data = crate::gpu_layer::gpu_to_cpu_vec(
                gh.to_dtype(candle_core::DType::F32)
                  .map_err(|e| ModelError::WeightNotFound(format!("GPU dtype convert prefill final: {}", e)))?
            ).map_err(|e| ModelError::WeightNotFound(
                format!("GPU download prefill final: {}", e)
            ))?;
            hidden = Tensor::new(data, vec![seq_len, hidden_size]).unwrap();
        }

        Ok(hidden)
    }

    /// Batched prefill: process all prompt tokens in a single forward pass per layer.
    /// Returns logits for the last token only (for next-token sampling).
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Result<Tensor, ModelError> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let start_pos = self.pos;

        let hidden = self.prefill_hidden_states(token_ids)?;

        // Norm the last token's hidden state.
        let last_hidden_data = hidden.data()[((seq_len - 1) * hidden_size)..].to_vec();
        let mut last_hidden = Tensor::new(last_hidden_data, vec![hidden_size]).unwrap();
        last_hidden = apply_norm(&last_hidden, &self.norm, self.norm_bias.as_ref(),
            self.config.norm_type, self.config.norm_eps);

        let logits = if let Some(ref lm) = self.lm_head {
            compact_linear_vec(&last_hidden, lm)
        } else {
            let logits_data = self.embed_tokens.matvec_f32(last_hidden.data());
            let vocab_size = self.config.vocab_size;
            Tensor::new(logits_data, vec![vocab_size]).unwrap()
        };

        self.pos += seq_len;
        self.kv_cache.set_seq_len(start_pos + seq_len);

        Ok(logits)
    }

    /// Fast sequence scoring using a single prefill pass (O(1) forward passes instead of O(seq_len)).
    ///
    /// Processes all tokens in parallel via batched attention, then applies norm + LM head
    /// to each position to extract per-token NLL. ~10-50× faster than `score_sequence`
    /// for long sequences.
    ///
    /// Returns per-token NLL of length `tokens.len() - 1`, where `result[i] = -log P(tokens[i+1] | tokens[..=i])`.
    pub fn score_sequence_fast(&mut self, tokens: &[u32]) -> Result<Vec<f64>, ModelError> {
        if tokens.len() < 2 {
            return Ok(vec![]);
        }
        self.reset();

        let seq_len = tokens.len();
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size;
        let start_pos = self.pos;

        let hidden = self.prefill_hidden_states(tokens)?;

        // Apply norm + LM head to each position, extract NLL for next-token prediction.
        let mut nll = Vec::with_capacity(seq_len - 1);
        for i in 0..seq_len - 1 {
            let row = hidden.data()[(i * hidden_size)..((i + 1) * hidden_size)].to_vec();
            let mut h = Tensor::new(row, vec![hidden_size]).unwrap();
            h = apply_norm(&h, &self.norm, self.norm_bias.as_ref(),
                self.config.norm_type, self.config.norm_eps);
            let logits = if let Some(ref lm) = self.lm_head {
                compact_linear_vec(&h, lm)
            } else {
                let logits_data = self.embed_tokens.matvec_f32(h.data());
                Tensor::new(logits_data, vec![vocab_size]).unwrap()
            };
            nll.push(-log_softmax_token(logits.data(), tokens[i + 1] as usize));
        }

        self.pos += seq_len;
        self.kv_cache.set_seq_len(start_pos + seq_len);

        Ok(nll)
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<GenerationResult, ModelError> {
        let start = Instant::now();
        let eos_id = self.config.eos_token_id.unwrap_or(2);
        let mut eos_ids = vec![eos_id];
        if eos_id == 128001 {
            eos_ids.push(128009);
        }

        let prefill_start = Instant::now();
        // Use batched prefill when multiple prompt tokens (much faster than one-at-a-time).
        let mut logits = if prompt_tokens.len() > 1 {
            self.forward_prefill(prompt_tokens)?
        } else {
            self.forward_single(prompt_tokens[0])?
        };
        let prefill_time = prefill_start.elapsed();

        let mut generated = Vec::new();
        let mut next_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
        generated.push(next_token);

        if eos_ids.contains(&next_token) {
            return Ok(GenerationResult {
                tokens: generated,
                prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
                decode_time_ms: 0.0,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                tokens_per_sec: 0.0,
                prompt_tokens: prompt_tokens.len(),
            });
        }

        let decode_start = Instant::now();
        for _ in 1..max_new_tokens {
            logits = self.forward_single(next_token)?;
            next_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
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

        Ok(GenerationResult {
            tokens: generated,
            prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
            decode_time_ms: decode_time.as_secs_f64() * 1000.0,
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            tokens_per_sec,
            prompt_tokens: prompt_tokens.len(),
        })
    }

    /// Streaming variant of `generate` — calls `on_token` for each token as it
    /// is produced, before the full result is assembled.
    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        mut on_token: F,
    ) -> Result<GenerationResult, ModelError>
    where
        F: FnMut(u32),
    {
        let start = Instant::now();
        let eos_id = self.config.eos_token_id.unwrap_or(2);
        let mut eos_ids = vec![eos_id];
        if eos_id == 128001 {
            eos_ids.push(128009);
        }

        let prefill_start = Instant::now();
        let mut logits = if prompt_tokens.len() > 1 {
            self.forward_prefill(prompt_tokens)?
        } else {
            self.forward_single(prompt_tokens[0])?
        };
        let prefill_time = prefill_start.elapsed();

        let mut generated = Vec::new();
        let mut next_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
        on_token(next_token);
        generated.push(next_token);

        if eos_ids.contains(&next_token) {
            return Ok(GenerationResult {
                tokens: generated,
                prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
                decode_time_ms: 0.0,
                total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                tokens_per_sec: 0.0,
                prompt_tokens: prompt_tokens.len(),
            });
        }

        let decode_start = Instant::now();
        for _ in 1..max_new_tokens {
            logits = self.forward_single(next_token)?;
            next_token = ops::sample_top_p(logits.data(), temperature, top_p) as u32;
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

        Ok(GenerationResult {
            tokens: generated,
            prefill_time_ms: prefill_time.as_secs_f64() * 1000.0,
            decode_time_ms: decode_time.as_secs_f64() * 1000.0,
            total_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            tokens_per_sec,
            prompt_tokens: prompt_tokens.len(),
        })
    }

    pub fn stats(&self) -> &PagingStats {
        &self.stats
    }

    pub fn layer_tiers(&self) -> &[LayerTier] {
        &self.layer_tiers
    }

    pub fn layer_size_bytes(&self) -> usize {
        self.layer_size_bytes
    }

    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.pos = 0;
        self.tick = 0;
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        if let Some(ref mut gpu_state) = self.gpu_state {
            gpu_state.reset();
        }
    }

    /// Directly inject pre-computed importance scores (from cache) and activate
    /// the top `active_count` layers without running a profiling pass.
    pub fn apply_importance_scores(&mut self, scores: &[f64], active_count: usize) {
        if scores.len() == self.config.num_hidden_layers {
            self.layer_importance = scores.to_vec();
        }
        self.apply_profiled_hot_only(active_count);
    }

    /// Return a copy of the current per-layer importance scores.
    pub fn layer_importance_scores(&self) -> &[f64] {
        &self.layer_importance
    }

    /// Get human-readable labels for per-layer quant assignments (for reporting).
    pub fn layer_quant_assignment_labels(&self) -> Option<Vec<String>> {
        self.layer_quant_assignments.as_ref().map(|assignments| {
            assignments.iter().map(|m| format!("{}", m)).collect()
        })
    }

    /// Score a sequence: returns per-token NLL (negative log-likelihood).
    ///
    /// Given tokens `[t0, t1, ..., tn]`, returns a Vec of length `n` where
    /// `result[i] = -log P(t_{i+1} | t_0..=t_i)`.
    ///
    /// Perplexity = exp(mean(result)).
    ///
    /// Resets model state before scoring so this can be called repeatedly.
    pub fn score_sequence(&mut self, tokens: &[u32]) -> Result<Vec<f64>, ModelError> {
        if tokens.len() < 2 {
            return Ok(vec![]);
        }
        self.reset();

        let mut nll = Vec::with_capacity(tokens.len() - 1);
        for i in 0..tokens.len() - 1 {
            let logits = self.forward_single(tokens[i])?;
            let log_prob = log_softmax_token(logits.data(), tokens[i + 1] as usize);
            nll.push(-log_prob);
        }
        Ok(nll)
    }

    pub fn memory_report(&self) -> String {
        let loaded = self.layers.iter().filter(|l| l.is_some()).count();
        let total = self.config.num_hidden_layers;
        let active = self.active_layer_set.iter().filter(|&&x| x).count();
        let mut report = format!(
            "Layers: {}/{} loaded ({} hot, {} warm, {} cold) | \
             Memory: {:.0} MB hot, {:.0} MB warm | \
             KV Cache: {:.1} MB | \
             Faults: {} ({:.1}%)",
            loaded,
            total,
            self.layer_tiers.iter().filter(|&&t| t == LayerTier::Hot).count(),
            self.layer_tiers.iter().filter(|&&t| t == LayerTier::Warm).count(),
            self.layer_tiers.iter().filter(|&&t| t == LayerTier::Cold).count(),
            self.hot_used_bytes as f64 / 1024.0 / 1024.0,
            self.warm_used_bytes as f64 / 1024.0 / 1024.0,
            self.kv_cache.memory_bytes() as f64 / 1024.0 / 1024.0,
            self.stats.page_faults,
            self.stats.fault_rate() * 100.0,
        );
        if self.paged_config.hot_only_mode || self.paged_config.active_layers.is_some() {
            report.push_str(&format!(
                " | Active: {}/{} layers, {} layer-evals skipped",
                active, total, self.layers_skipped,
            ));
        }
        report
    }
}

// Keep the old name as a type alias for backward compatibility.
pub type PagedLlamaModel = PagedModel;

/// Compute log P(token_id) = logits[token_id] - log(sum_j exp(logits[j])).
/// Uses log-sum-exp trick for numerical stability.
fn log_softmax_token(logits: &[f32], token_id: usize) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum_exp: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
    logits[token_id] as f64 - log_sum_exp
}
