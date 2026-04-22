//! GPU-accelerated transformer layer inference via candle-core.
//!
//! This module provides production-grade GPU inference for NVE's hot-tier layers.
//! When a layer is marked as "hot" and a GPU device is configured, all tensor
//! operations (attention, FFN, norms) run on-device with no CPU round-trips
//! in the hot path.
//!
//! ## Architecture
//!
//! - `GpuLayerWeights`: F32 candle tensors for each weight matrix, uploaded once at init.
//! - `GpuKvCache`: Per-layer accumulated K/V tensors on GPU, concatenated each decode step.
//! - `GpuInferenceState`: Owns device, kv_cache, per-layer weights, and RoPE tables.
//! - `gpu_layer_forward_decode`: Full single-token forward pass on GPU (no CPU round-trips).
//! - `gpu_layer_forward_prefill`: Batched prefill forward pass on GPU.
//!
//! ## CPU↔GPU boundaries
//!
//! Only two transfers per forward call:
//! 1. Upload embedding → GPU at the first hot layer.
//! 2. Download GPU hidden state → CPU after the last hot layer (for final norm + lm_head).
//!
//! ## Compilation
//!
//! Gated on `#[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]`.
//! On CPU-only builds this module compiles to nothing.

#[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
pub use inner::*;

#[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
mod inner {
    use candle_core::{DType, Device as CandleDev, Tensor as CT, D};

    use crate::arch::{FfnType, NormType};
    use crate::generic_model::GenericBlockWeights;
    use crate::ops::RopeFreqs;
    use crate::quantize::{WeightStorage, Q4_BLOCK_SIZE};

    // Import fused kernel wrappers when the cuda feature is enabled.
    // Falls back silently to unfused candle ops on non-CUDA targets.
    #[cfg(feature = "cuda")]
    use crate::cuda_kernels::{
        dequant_w4a16, flash_decode_f16, fused_layer_norm, fused_matvec,
        fused_matvec_w4a16, fused_qkv_matvec_w4a16,
        fused_rms_norm, fused_rope_decode, fused_rope_prefill,
        matvec_w4a8, quantize_f16_q8,
    };

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Upload a `WeightStorage` weight matrix to the GPU.
    ///
    /// - Q4_0 weights on a CUDA device → `GpuWeight::W4` (packed nibbles + F32 scales).
    ///   Decode uses the W4A16 fused kernel (~4× less VRAM bandwidth).
    ///   Prefill dequantizes on-the-fly via `nve_dequant_w4a16`.
    /// - All other formats (Compact, Q8, etc.) → `GpuWeight::F16` (dequantize to F16).
    fn ws_to_gpu(ws: &WeightStorage, device: &CandleDev) -> candle_core::Result<GpuWeight> {
        // Q4_0 on CUDA: keep weights packed; decode with W4A16 kernel
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            if let WeightStorage::Quantized4(q) = ws {
                let (nibbles_data, scales_data, awq) = q.extract_for_gpu();
                // ws.shape() returns [usize; 2] by value (WeightStorage trait method)
                let [n, k] = ws.shape();
                let blocks_per_row = k / Q4_BLOCK_SIZE;
                let nibbles = CT::from_slice(&nibbles_data, (n, k / 2), device)?;
                let scales  = CT::from_slice(&scales_data,  (n, blocks_per_row), device)?;
                let awq_scales = awq.map(|a| CT::from_slice(a, (k,), device)).transpose()?;
                return Ok(GpuWeight::W4(GpuW4 { nibbles, scales, awq_scales, n, k }));
            }
        }
        // Fallback: dequantize to F16 on CPU then upload
        let [rows, cols] = ws.shape();
        let data = ws.to_f32_vec();
        let t = CT::from_slice(&data, (rows, cols), device)?.to_dtype(DType::F16)?;
        Ok(GpuWeight::F16(t))
    }

    /// Numerically stable softmax over the last dimension.
    fn softmax_last_dim(x: &CT) -> candle_core::Result<CT> {
        let max_val = x.max_keepdim(D::Minus1)?;
        let shifted = x.broadcast_sub(&max_val)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(D::Minus1)?;
        exp.broadcast_div(&sum)
    }

    /// RMSNorm on GPU: y = x / rms(x) * weight
    /// x shape: [seq, hidden] or [1, hidden]; weight: [hidden]
    /// Computation is promoted to F32 for numerical stability (hidden_size can be
    /// large enough that squared sums overflow BF16/F16), then cast back.
    fn rms_norm_gpu(x: &CT, weight: &CT, eps: f64) -> candle_core::Result<CT> {
        let orig = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let wf = weight.to_dtype(DType::F32)?;
        let mean_sq = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = mean_sq.affine(1.0, eps)?.sqrt()?;
        let normed = xf.broadcast_div(&rms)?;
        normed.broadcast_mul(&wf.unsqueeze(0)?)?.to_dtype(orig)
    }

    /// LayerNorm on GPU. Computation promoted to F32 for stability.
    fn layer_norm_gpu(
        x: &CT,
        weight: &CT,
        bias: Option<&CT>,
        eps: f64,
    ) -> candle_core::Result<CT> {
        let orig = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let wf = weight.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(D::Minus1)?;
        let centered = xf.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = var.affine(1.0, eps)?.sqrt()?;
        let normed = centered.broadcast_div(&std)?;
        let scaled = normed.broadcast_mul(&wf.unsqueeze(0)?)?;
        let result = if let Some(b) = bias {
            let bf = b.to_dtype(DType::F32)?;
            scaled.broadcast_add(&bf.unsqueeze(0)?)?
        } else {
            scaled
        };
        result.to_dtype(orig)
    }

    fn apply_norm_gpu(
        x: &CT,
        weight: &CT,
        bias: Option<&CT>,
        norm_type: NormType,
        eps: f64,
    ) -> candle_core::Result<CT> {
        // Use fused single-pass CUDA kernels when running on a CUDA device.
        // The fused kernels reduce 6-7 separate kernel launches to 1.
        // Each kernel normalizes exactly one row (hidden_size elements).
        // For prefill [seq_len, hidden] we run the kernel once per token.
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && x.dtype() == candle_core::DType::F16
            && std::env::var("NVE_NO_FUSED").is_err()
        {
            let x_shape = x.shape().clone();
            let dims    = x_shape.dims();
            // Last dim is hidden_size; everything before is batch / seq dimensions.
            let hidden = dims[dims.len() - 1];
            let n_rows = x.elem_count() / hidden;
            let w_flat = weight.flatten_all()?;
            let b_flat = bias.map(|b| b.flatten_all()).transpose()?;

            // Apply the kernel independently per row (per token).
            // narrow(0, i, 1) slices the leading batch/seq dim.
            let mut rows: Vec<CT> = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                // Works for [1, hidden] (decode) and [seq, hidden] (prefill).
                let row = x.narrow(0, i, 1)?.flatten_all()?;
                let normed = match norm_type {
                    NormType::RMSNorm => fused_rms_norm(&row, &w_flat, eps)?,
                    NormType::LayerNorm => fused_layer_norm(&row, &w_flat, b_flat.as_ref(), eps)?,
                };
                rows.push(normed.unsqueeze(0)?);
            }
            let stacked = CT::cat(&rows, 0)?; // [n_rows, hidden]
            return stacked.reshape(&x_shape);
        }
        // Fallback: unfused candle ops (CPU / non-CUDA builds / non-F16)
        match norm_type {
            NormType::RMSNorm => rms_norm_gpu(x, weight, eps),
            NormType::LayerNorm => layer_norm_gpu(x, weight, bias, eps),
        }
    }

    /// F16 matrix multiply: x[B, K] × W[N, K]ᵀ → [B, N].
    ///
    /// When B=1, x is F16, and CUDA (and NVE_NO_FUSED is unset), dispatches to
    /// the fused warp-shuffle matvec kernel. Falls back to candle matmul for
    /// prefill (B>1), non-CUDA, or non-F16.
    fn mm_f16(x: &CT, w: &CT) -> candle_core::Result<CT> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && x.dtype() == candle_core::DType::F16
            && x.dim(0).map(|d| d == 1).unwrap_or(false)
            && std::env::var("NVE_NO_FUSED").is_err()
        {
            let (n, k) = w.dims2()?;
            return fused_matvec(x, w, n, k);
        }
        x.matmul(&w.t()?)
    }

    // ── W4A16 weight types ────────────────────────────────────────────────────

    /// GPU-resident packed 4-bit weights for one projection matrix.
    ///
    /// Stores Q4_0-format weights in a memory-efficient layout:
    /// - `nibbles` [N, K/2]: 2 INT4 per byte, sequential pairs.
    /// - `scales`  [N, K/32]: one F32 scale per 32-element block.
    /// - `awq_scales` [K]: optional per-column AWQ input scaling (F32).
    pub struct GpuW4 {
        pub nibbles:    CT,           // [N, K/2] U8
        pub scales:     CT,           // [N, K/32] F32
        pub awq_scales: Option<CT>,   // [K] F32 — None if no AWQ
        pub n:          usize,
        pub k:          usize,
    }

    /// A GPU weight matrix: either F16 (full precision) or W4 (4-bit quantized).
    pub enum GpuWeight {
        /// Full F16 weight [N, K] — used for Compact / Q8 / non-CUDA sources.
        F16(CT),
        /// Packed 4-bit weight — used for Q4_0 sources on CUDA.
        W4(GpuW4),
    }

    impl GpuWeight {
        /// Compute x @ Wᵀ for any batch size.
        ///
        /// - Decode (x: [1, K]) on CUDA: uses the fused W4A16 or F16 matvec kernel.
        /// - Prefill (x: [seq, K]) or fallback: W4 dequantizes to F16 first, then matmul.
        pub fn matmul_x(&self, x: &CT) -> candle_core::Result<CT> {
            match self {
                GpuWeight::F16(w) => mm_f16(x, w),
                GpuWeight::W4(w4) => {
                    // Decode path: single-token W4A16 fused kernel
                    #[cfg(feature = "cuda")]
                    if x.device().is_cuda()
                        && x.dim(0).map(|d| d == 1).unwrap_or(false)
                        && std::env::var("NVE_NO_FUSED").is_err()
                    {
                        return fused_matvec_w4a16(
                            x, &w4.nibbles, &w4.scales,
                            w4.awq_scales.as_ref(), w4.n, w4.k,
                        );
                    }
                    // Prefill path: dequantize to F16 then batched matmul
                    let f16w = self.dequant_to_f16(x.device())?;
                    x.matmul(&f16w.t()?)
                }
            }
        }

        /// Dequantize W4 packed weights to a full F16 tensor [N, K].
        /// Used for the prefill path and any non-CUDA fallback.
        fn dequant_to_f16(&self, _device: &CandleDev) -> candle_core::Result<CT> {
            match self {
                GpuWeight::F16(w) => Ok(w.clone()),
                GpuWeight::W4(w4) => {
                    #[cfg(feature = "cuda")]
                    {
                        return dequant_w4a16(
                            &w4.nibbles, &w4.scales,
                            w4.awq_scales.as_ref(), w4.n, w4.k,
                        );
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = (w4, device);
                        candle_core::bail!("W4 dequant requires --features cuda")
                    }
                }
            }
        }
    }

    /// Apply RoPE rotation to Q and K tensors.
    /// q: [num_heads, head_dim], k: [num_kv_heads, head_dim]
    /// rope_cos/sin: [max_seq, half_dim]
    fn apply_rope_decode(
        q: CT,
        k: CT,
        pos: usize,
        rope_cos: &CT,
        rope_sin: &CT,
        half_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> candle_core::Result<(CT, CT)> {
        let cos_row = rope_cos.narrow(0, pos, 1)?.squeeze(0)?; // [half_dim]
        let sin_row = rope_sin.narrow(0, pos, 1)?.squeeze(0)?;

        // Fast path: single fused kernel (in-place) on CUDA F16.
        // 7×num_heads separate kernel launches → 1 parallel kernel.
        #[cfg(feature = "cuda")]
        if q.device().is_cuda()
            && q.dtype() == candle_core::DType::F16
            && std::env::var("NVE_NO_FUSED").is_err()
        {
            // The fused kernel modifies q/k in-place.  We need contiguous tensors
            // so the kernel sees linear [num_heads, head_dim] memory.
            let q_c = q.contiguous()?;
            let k_c = k.contiguous()?;
            fused_rope_decode(&q_c, &k_c, &cos_row, &sin_row, num_heads, num_kv_heads, head_dim)?;
            return Ok((q_c, k_c));
        }

        // Fallback: unfused candle ops (CPU / Metal / non-F16)
        let rotate = |x: CT| -> candle_core::Result<CT> {
            let nh = x.dim(0)?;
            let x0 = x.narrow(1, 0, half_dim)?;
            let x1 = x.narrow(1, half_dim, half_dim)?;
            let c = cos_row.unsqueeze(0)?.broadcast_as((nh, half_dim))?;
            let s = sin_row.unsqueeze(0)?.broadcast_as((nh, half_dim))?;
            let r0 = x0.mul(&c)?.sub(&x1.mul(&s)?)?;
            let r1 = x0.mul(&s)?.add(&x1.mul(&c)?)?;
            CT::cat(&[&r0, &r1], 1)
        };

        Ok((rotate(q)?, rotate(k)?))
    }

    /// Apply RoPE to each token position in a prefill batch.
    /// q: [seq_len, num_heads, head_dim], k: [seq_len, num_kv_heads, head_dim]
    fn apply_rope_prefill(
        q: CT,
        k: CT,
        start_pos: usize,
        rope_cos: &CT,
        rope_sin: &CT,
        half_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> candle_core::Result<(CT, CT)> {
        let seq_len = q.dim(0)?;

        // Fast path: single fused kernel on CUDA F16.
        // Passes the full cos/sin table [max_seq, half_dim]; kernel indexes by position.
        #[cfg(feature = "cuda")]
        if q.device().is_cuda()
            && q.dtype() == candle_core::DType::F16
            && std::env::var("NVE_NO_FUSED").is_err()
        {
            let q_c = q.contiguous()?;
            let k_c = k.contiguous()?;
            fused_rope_prefill(
                &q_c, &k_c, rope_cos, rope_sin,
                start_pos, num_heads, num_kv_heads, head_dim,
            )?;
            return Ok((q_c, k_c));
        }

        // Fallback: unfused candle ops (CPU / Metal / non-F16)
        // Build a [seq_len, half_dim] cos/sin slice for positions [start..start+seq_len]
        let cos_slice = rope_cos.narrow(0, start_pos, seq_len)?; // [seq_len, half_dim]
        let sin_slice = rope_sin.narrow(0, start_pos, seq_len)?;

        // Broadcast to [seq_len, num_heads, half_dim]
        let cos_q = cos_slice.unsqueeze(1)?.broadcast_as((seq_len, num_heads, half_dim))?;
        let sin_q = sin_slice.unsqueeze(1)?.broadcast_as((seq_len, num_heads, half_dim))?;
        let cos_k = cos_slice.unsqueeze(1)?.broadcast_as((seq_len, num_kv_heads, half_dim))?;
        let sin_k = sin_slice.unsqueeze(1)?.broadcast_as((seq_len, num_kv_heads, half_dim))?;

        let rotate = |x: CT, c: &CT, s: &CT| -> candle_core::Result<CT> {
            let x0 = x.narrow(2, 0, half_dim)?;
            let x1 = x.narrow(2, half_dim, half_dim)?;
            let r0 = x0.mul(c)?.sub(&x1.mul(s)?)?;
            let r1 = x0.mul(s)?.add(&x1.mul(c)?)?;
            CT::cat(&[&r0, &r1], 2)
        };

        Ok((rotate(q, &cos_q, &sin_q)?, rotate(k, &cos_k, &sin_k)?))
    }

    // ── Structs ───────────────────────────────────────────────────────────────

    /// GPU-resident weights for a single transformer layer.
    ///
    /// Projection matrices use `GpuWeight`, which is either:
    /// - `F16`: full-precision upload (for Compact / Q8 / CPU weights)
    /// - `W4`: packed 4-bit Q4_0 (for Q4 quantized weights on CUDA)
    ///
    /// Norm weights stay as F16 `CT` — they are small and not bandwidth-bottlenecked.
    pub struct GpuLayerWeights {
        /// Attention projections [out_dim, in_dim] — F16 or W4
        pub q_proj: GpuWeight,
        pub k_proj: GpuWeight,
        pub v_proj: GpuWeight,
        pub o_proj: GpuWeight,
        /// FFN projections — F16 or W4
        pub gate_proj: Option<GpuWeight>,
        pub up_proj:   GpuWeight,
        pub down_proj: GpuWeight,
        /// Norm weights: [hidden_size] — always F16
        pub input_norm: CT,
        pub post_attn_norm: CT,
        /// Optional norm biases
        pub input_norm_bias: Option<CT>,
        pub post_attn_norm_bias: Option<CT>,
        /// Optional attention biases: [out_dim]
        pub q_bias: Option<CT>,
        pub k_bias: Option<CT>,
        pub v_bias: Option<CT>,
        pub o_bias: Option<CT>,
    }

    /// Upload layer weights from CPU (`GenericBlockWeights`) to GPU.
    ///
    /// Projection matrices are uploaded as `GpuWeight::W4` when the source is Q4_0
    /// on a CUDA device, saving ~4× VRAM bandwidth during decode. All other formats
    /// are dequantized and uploaded as F16 (`GpuWeight::F16`).
    pub fn upload_layer_weights(
        layer: &GenericBlockWeights,
        device: &CandleDev,
    ) -> candle_core::Result<GpuLayerWeights> {
        let proj = |w: &WeightStorage| ws_to_gpu(w, device);

        let norm_vec = |t: &crate::tensor::Tensor| -> candle_core::Result<CT> {
            // Norm weights must be F16 on GPU — fused CUDA kernels (fused_rms_norm,
            // fused_layer_norm) call cuda_dev_ptr() which asserts F16 dtype.
            CT::from_slice(t.data(), (t.numel(),), device)?.to_dtype(DType::F16)
        };
        let norm_vec_opt = |t: Option<&crate::tensor::Tensor>| -> candle_core::Result<Option<CT>> {
            t.map(|v| norm_vec(v)).transpose()
        };
        let bias_vec = |t: &crate::tensor::Tensor| -> candle_core::Result<CT> {
            // Biases must be F16 to be compatible with F16 activations on GPU.
            CT::from_slice(t.data(), (t.numel(),), device)?.to_dtype(DType::F16)
        };

        Ok(GpuLayerWeights {
            q_proj:    proj(&layer.q_proj)?,
            k_proj:    proj(&layer.k_proj)?,
            v_proj:    proj(&layer.v_proj)?,
            o_proj:    proj(&layer.o_proj)?,
            gate_proj: layer.gate_proj.as_ref().map(|g| proj(g)).transpose()?,
            up_proj:   proj(&layer.up_proj)?,
            down_proj: proj(&layer.down_proj)?,
            input_norm:          norm_vec(&layer.input_norm)?,
            post_attn_norm:      norm_vec(&layer.post_attn_norm)?,
            input_norm_bias:     norm_vec_opt(layer.input_norm_bias.as_ref())?,
            post_attn_norm_bias: norm_vec_opt(layer.post_attn_norm_bias.as_ref())?,
            q_bias: layer.q_bias.as_ref().map(|b| bias_vec(b)).transpose()?,
            k_bias: layer.k_bias.as_ref().map(|b| bias_vec(b)).transpose()?,
            v_bias: layer.v_bias.as_ref().map(|b| bias_vec(b)).transpose()?,
            o_bias: layer.o_bias.as_ref().map(|b| bias_vec(b)).transpose()?,
        })
    }

    /// GPU KV cache: stores accumulated K/V tensors per hot layer.
    ///
    /// Layout per layer: [num_kv_heads, seq_len, head_dim]
    /// Grows by 1 token each decode step via `CT::cat`.
    pub struct GpuKvCache {
        k_cache: Vec<Option<CT>>,
        v_cache: Vec<Option<CT>>,
        pub num_layers: usize,
    }

    impl GpuKvCache {
        pub fn new(num_layers: usize) -> Self {
            GpuKvCache {
                k_cache: vec![None; num_layers],
                v_cache: vec![None; num_layers],
                num_layers,
            }
        }

        pub fn reset(&mut self) {
            for i in 0..self.num_layers {
                self.k_cache[i] = None;
                self.v_cache[i] = None;
            }
        }

        /// Append a new K/V slice for layer `layer_idx`.
        /// new_k/v shape: [num_kv_heads, 1, head_dim] (decode) or [num_kv_heads, seq_len, head_dim] (prefill).
        pub fn append(
            &mut self,
            layer_idx: usize,
            new_k: CT,
            new_v: CT,
        ) -> candle_core::Result<()> {
            self.k_cache[layer_idx] = Some(match self.k_cache[layer_idx].take() {
                None => new_k,
                Some(existing) => CT::cat(&[&existing, &new_k], 1)?,
            });
            self.v_cache[layer_idx] = Some(match self.v_cache[layer_idx].take() {
                None => new_v,
                Some(existing) => CT::cat(&[&existing, &new_v], 1)?,
            });
            Ok(())
        }

        /// Get the full cached K/V pair for a layer.
        pub fn get(&self, layer_idx: usize) -> Option<(&CT, &CT)> {
            match (&self.k_cache[layer_idx], &self.v_cache[layer_idx]) {
                (Some(k), Some(v)) => Some((k, v)),
                _ => None,
            }
        }
    }

    /// Full GPU inference state: lives for the lifetime of the PagedModel.
    pub struct GpuInferenceState {
        /// candle device handle.
        pub device: CandleDev,
        /// K/V cache for hot GPU layers.
        pub kv_cache: GpuKvCache,
        /// Per-layer GPU weights. `None` for non-hot (CPU) layers.
        pub layer_weights: Vec<Option<GpuLayerWeights>>,
        /// Optional captured CUDA decode graph (eliminates kernel-launch overhead).
        #[cfg(feature = "cuda")]
        pub decode_graph: Option<crate::decode_graph::StaticDecodeGraph>,
        /// RoPE cosine table on GPU: [max_seq, half_dim]
        pub rope_cos: Option<CT>,
        /// RoPE sine table on GPU: [max_seq, half_dim]
        pub rope_sin: Option<CT>,
        pub head_dim: usize,
        pub num_heads: usize,
        pub num_kv_heads: usize,
        pub ffn_type: FfnType,
        pub norm_type: NormType,
        pub norm_eps: f64,
        pub num_layers: usize,
    }

    impl GpuInferenceState {
        /// Reset KV cache (call after each generation sequence).
        pub fn reset(&mut self) {
            self.kv_cache.reset();
            #[cfg(feature = "cuda")]
            if let Some(ref mut g) = self.decode_graph { g.reset_kv(); }
        }

        /// Capture the full multi-layer decode pass into a CUDA graph.
        ///
        /// After this call, `replay_decode_graph` can be used instead of the
        /// per-layer `gpu_layer_forward_decode` loop for single-token decode.
        ///
        /// Requirements: all active layers must have W4 weights; head_dim must
        /// be divisible by 32; NVE_NO_FUSED and NVE_NO_FLASH must be unset.
        #[cfg(feature = "cuda")]
        pub fn build_decode_graph(&mut self, max_seq_len: usize) -> candle_core::Result<()> {
            use crate::decode_graph::StaticDecodeGraph;
            let graph = StaticDecodeGraph::capture(self, max_seq_len)?;
            self.decode_graph = Some(graph);
            Ok(())
        }

        /// Replay the captured decode graph for one token position.
        ///
        /// Replaces the per-layer `gpu_layer_forward_decode` loop when a graph
        /// has been built with `build_decode_graph`.
        ///
        /// `input`: [hidden] or [1, hidden] F16 on CUDA.
        /// Returns: [hidden] F16 on CUDA — the last-layer hidden state.
        #[cfg(feature = "cuda")]
        pub fn replay_decode_graph(&mut self, input: &CT, pos: usize) -> candle_core::Result<CT> {
            self.decode_graph.as_mut()
                .expect("replay_decode_graph called without build_decode_graph")
                .replay(input, pos)
        }

        /// True if a CUDA decode graph has been built and is ready to replay.
        #[cfg(feature = "cuda")]
        pub fn has_decode_graph(&self) -> bool {
            self.decode_graph.as_ref().map(|_| true).unwrap_or(false)
        }
    }

    /// Build the GPU inference state for the given hot layers.
    ///
    /// `layers` is the full `PagedModel::layers` array; `hot_set[i]` indicates
    /// whether layer `i` should be resident on GPU.
    pub fn build_gpu_state(
        device: CandleDev,
        layers: &[Option<GenericBlockWeights>],
        hot_set: &[bool],
        rope_freqs: Option<&RopeFreqs>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_type: FfnType,
        norm_type: NormType,
        norm_eps: f64,
    ) -> candle_core::Result<GpuInferenceState> {
        let num_layers = layers.len();
        let mut layer_weights: Vec<Option<GpuLayerWeights>> = Vec::with_capacity(num_layers);

        for (i, layer_opt) in layers.iter().enumerate() {
            if hot_set[i] {
                if let Some(layer) = layer_opt {
                    log::info!("[GPU] Uploading layer {} weights to device", i);
                    layer_weights.push(Some(upload_layer_weights(layer, &device)?));
                } else {
                    // Layer marked hot but not loaded yet — skip for now.
                    // init_gpu() should be called after ensure_all_hot_loaded().
                    layer_weights.push(None);
                }
            } else {
                layer_weights.push(None);
            }
        }

        // Upload RoPE tables to GPU as BF16 (same dtype as weights).
        let (rope_cos, rope_sin) = if let Some(rope) = rope_freqs {
            let half = rope.half_dim();
            let max_seq = rope.max_seq();
            let cos = CT::from_slice(rope.cos_data(), (max_seq, half), &device)?
                .to_dtype(DType::F16)?;
            let sin = CT::from_slice(rope.sin_data(), (max_seq, half), &device)?
                .to_dtype(DType::F16)?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };

        Ok(GpuInferenceState {
            kv_cache: GpuKvCache::new(num_layers),
            layer_weights,
            rope_cos,
            rope_sin,
            device,
            head_dim,
            num_heads,
            num_kv_heads,
            ffn_type,
            norm_type,
            norm_eps,
            num_layers,
            #[cfg(feature = "cuda")]
            decode_graph: None,
        })
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    /// GPU decode attention for a single token position.
    ///
    /// normed: [1, hidden_size] on GPU
    /// Returns: [1, hidden_size] attention output (before o_proj bias) on GPU
    fn gpu_attn_decode(
        normed: &CT,
        w: &GpuLayerWeights,
        kv_cache: &mut GpuKvCache,
        layer_idx: usize,
        pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_cos: &CT,
        rope_sin: &CT,
        use_w4a8: bool,
    ) -> candle_core::Result<CT> {
        // ── Q/K/V projections ─────────────────────────────────────────────────
        // W4A8 fast path: quantize input once, use dp4a for all three projections.
        // W4A16 fallback: fused single kernel (3 projections in 1 launch).
        #[cfg(feature = "cuda")]
        let (q, k, v) = if normed.device().is_cuda()
            && std::env::var("NVE_NO_FUSED").is_err()
        {
            if let (GpuWeight::W4(wq), GpuWeight::W4(wk), GpuWeight::W4(wv))
                = (&w.q_proj, &w.k_proj, &w.v_proj)
            {
                if use_w4a8 {
                    // W4A8: quantize normed once → Q8, then dp4a matvec for Q/K/V
                    let (xq, xq_sc) = quantize_f16_q8(normed)?;
                    let qo = matvec_w4a8(&xq, &xq_sc, &wq.nibbles, &wq.scales, wq.n, wq.k)?;
                    let ko = matvec_w4a8(&xq, &xq_sc, &wk.nibbles, &wk.scales, wk.n, wk.k)?;
                    let vo = matvec_w4a8(&xq, &xq_sc, &wv.nibbles, &wv.scales, wv.n, wv.k)?;
                    (qo, ko, vo)
                } else {
                    // W4A16 fallback (env override)
                    let (qo, ko, vo) = fused_qkv_matvec_w4a16(
                        normed,
                        &wq.nibbles, &wq.scales, wq.awq_scales.as_ref(), wq.n,
                        &wk.nibbles, &wk.scales, wk.awq_scales.as_ref(), wk.n,
                        &wv.nibbles, &wv.scales, wv.awq_scales.as_ref(), wv.n,
                        wq.k,
                    )?;
                    (qo, ko, vo)
                }
            } else {
                (
                    w.q_proj.matmul_x(normed)?,
                    w.k_proj.matmul_x(normed)?,
                    w.v_proj.matmul_x(normed)?,
                )
            }
        } else {
            (
                w.q_proj.matmul_x(normed)?,
                w.k_proj.matmul_x(normed)?,
                w.v_proj.matmul_x(normed)?,
            )
        };

        #[cfg(not(feature = "cuda"))]
        let (q, k, v) = (
            w.q_proj.matmul_x(normed)?,
            w.k_proj.matmul_x(normed)?,
            w.v_proj.matmul_x(normed)?,
        );

        // Optional biases
        let q = if let Some(ref b) = w.q_bias { q.broadcast_add(b)? } else { q };
        let k = if let Some(ref b) = w.k_bias { k.broadcast_add(b)? } else { k };
        let v = if let Some(ref b) = w.v_bias { v.broadcast_add(b)? } else { v };

        // Reshape to per-head layout
        let q = q.reshape((num_heads, head_dim))?;       // [nh, hd]
        let k = k.reshape((num_kv_heads, head_dim))?;    // [nkv, hd]
        let v = v.reshape((num_kv_heads, head_dim))?;

        // RoPE
        let half_dim = head_dim / 2;
        let (q, k) = apply_rope_decode(
            q, k, pos, rope_cos, rope_sin,
            half_dim, num_heads, num_kv_heads, head_dim,
        )?;

        // Append to KV cache: layout → [num_kv_heads, seq_len, head_dim]
        kv_cache.append(layer_idx, k.unsqueeze(1)?, v.unsqueeze(1)?)?;

        let (k_full, v_full) = kv_cache.get(layer_idx).unwrap();
        let seq_len = k_full.dim(1)?;

        // ── Attention ─────────────────────────────────────────────────────────
        // Fast path: fused flash decode kernel when on CUDA F16 and head_dim is
        // supported (divisible by 32, ≤ 512).  GQA-native — no expansion copy.
        #[cfg(feature = "cuda")]
        if q.device().is_cuda()
            && q.dtype() == candle_core::DType::F16
            && head_dim % 32 == 0 && head_dim <= 512
            && std::env::var("NVE_NO_FUSED").is_err()
            && std::env::var("NVE_NO_FLASH").is_err()
        {
            let scale = (head_dim as f32).powf(-0.5);
            let attn = flash_decode_f16(
                &q.contiguous()?,
                k_full, v_full,
                num_heads, num_kv_heads, scale,
            )?;
            let attn = attn.reshape((1, num_heads * head_dim))?;
            // W4A8 o_proj when available
            if let GpuWeight::W4(wo) = &w.o_proj {
                if use_w4a8 {
                    let (xq, xq_sc) = quantize_f16_q8(&attn)?;
                    return matvec_w4a8(&xq, &xq_sc, &wo.nibbles, &wo.scales, wo.n, wo.k);
                }
            }
            return w.o_proj.matmul_x(&attn);
        }

        // Fallback: unfused cuBLAS path (CPU / Metal / unsupported head_dim).
        // GQA expansion: [num_kv_heads, seq_len, head_dim] → [num_heads, seq_len, head_dim]
        let groups = num_heads / num_kv_heads;
        let k_exp = if groups > 1 {
            k_full
                .unsqueeze(1)?
                .expand((num_kv_heads, groups, seq_len, head_dim))?
                .contiguous()?
                .reshape((num_heads, seq_len, head_dim))?
        } else {
            k_full.clone()
        };
        let v_exp = if groups > 1 {
            v_full
                .unsqueeze(1)?
                .expand((num_kv_heads, groups, seq_len, head_dim))?
                .contiguous()?
                .reshape((num_heads, seq_len, head_dim))?
        } else {
            v_full.clone()
        };

        let scale = (head_dim as f64).powf(-0.5);
        let q3     = q.unsqueeze(1)?;
        let scores = q3.matmul(&k_exp.t()?.contiguous()?)?.affine(scale, 0.0)?;
        let scores = softmax_last_dim(&scores)?;
        let attn   = scores.matmul(&v_exp.contiguous()?)?;
        let attn   = attn.reshape((1, num_heads * head_dim))?;
        w.o_proj.matmul_x(&attn)
    }

    /// GPU prefill attention for a batch of tokens.
    ///
    /// normed: [seq_len, hidden_size] on GPU
    fn gpu_attn_prefill(
        normed: &CT,
        w: &GpuLayerWeights,
        kv_cache: &mut GpuKvCache,
        layer_idx: usize,
        start_pos: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_cos: &CT,
        rope_sin: &CT,
    ) -> candle_core::Result<CT> {
        let seq_len = normed.dim(0)?;

        // Batch projections: [seq_len, d_q/kv]
        // matmul_x handles batch>1: W4 dequantizes to F16 first, F16 uses cuBLAS.
        let q = w.q_proj.matmul_x(normed)?;
        let k = w.k_proj.matmul_x(normed)?;
        let v = w.v_proj.matmul_x(normed)?;

        let q = if let Some(ref b) = w.q_bias { q.broadcast_add(b)? } else { q };
        let k = if let Some(ref b) = w.k_bias { k.broadcast_add(b)? } else { k };
        let v = if let Some(ref b) = w.v_bias { v.broadcast_add(b)? } else { v };

        // Reshape to [seq_len, num_heads, head_dim]
        let q = q.reshape((seq_len, num_heads, head_dim))?;
        let k = k.reshape((seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((seq_len, num_kv_heads, head_dim))?;

        // RoPE (applied across all positions at once)
        let half_dim = head_dim / 2;
        let (q, k) = apply_rope_prefill(q, k, start_pos, rope_cos, rope_sin, half_dim, num_heads, num_kv_heads, head_dim)?;

        // Reshape K/V for cache: [num_kv_heads, seq_len, head_dim]
        // transpose → non-contiguous; make contiguous before storing in cache.
        let k_for_cache = k.transpose(0, 1)?.contiguous()?; // [num_kv_heads, seq_len, head_dim]
        let v_for_cache = v.transpose(0, 1)?.contiguous()?;
        kv_cache.append(layer_idx, k_for_cache, v_for_cache)?;

        let (k_full, v_full) = kv_cache.get(layer_idx).unwrap();
        let total_len = k_full.dim(1)?; // start_pos + seq_len

        // ── Attention kernel ──────────────────────────────────────────────────
        // Two paths, compile-time selected:
        //
        // [flash-attn]     Flash Attention v2 — fused, O(1) VRAM.
        //                  Requires sm_80+ (Ampere/Ada/Hopper). Handles GQA natively.
        //                  Build: cargo build --release --features cuda,flash-attn
        //
        // [default]        Unfused scaled-dot-product attention.
        //                  Works on all GPUs (including T4/sm_75).
        //                  GQA expanded manually before matmul.

        #[cfg(feature = "flash-attn")]
        let attn = {
            // FA2 shape: [batch, seq, heads, head_dim]
            // q: [sl, nh, hd] → [1, sl, nh, hd]
            let q_fa = q.unsqueeze(0)?;
            // k/v from cache: [nkv, total_len, hd] → [1, total_len, nkv, hd]
            let k_fa = k_full.transpose(0, 1)?.contiguous()?.unsqueeze(0)?;
            let v_fa = v_full.transpose(0, 1)?.contiguous()?.unsqueeze(0)?;
            let scale = (head_dim as f64).powf(-0.5) as f32;
            // FA2 returns [1, sl, nh, hd]; causal=true handles the mask
            let out = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true)?;
            // [1, sl, nh, hd] → [sl, nh*hd]
            out.squeeze(0)?.contiguous()?.reshape((seq_len, num_heads * head_dim))?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn = {
            // Unfused SDPA (sm_75-compatible). Expand K/V for GQA, then matmul.
            // q: [sl, nh, hd] → [nh, sl, hd]
            let q_t = q.transpose(0, 1)?.contiguous()?;

            let groups = num_heads / num_kv_heads;
            let k_exp = if groups > 1 {
                k_full
                    .unsqueeze(1)?
                    .expand((num_kv_heads, groups, total_len, head_dim))?
                    .contiguous()?
                    .reshape((num_heads, total_len, head_dim))?
            } else {
                k_full.clone()
            };
            let v_exp = if groups > 1 {
                v_full
                    .unsqueeze(1)?
                    .expand((num_kv_heads, groups, total_len, head_dim))?
                    .contiguous()?
                    .reshape((num_heads, total_len, head_dim))?
            } else {
                v_full.clone()
            };

            let scale = (head_dim as f64).powf(-0.5);
            let scores = q_t.matmul(&k_exp.t()?.contiguous()?)?.affine(scale, 0.0)?;

            // Causal mask: upper triangle = -inf; cast to scores dtype (F16 on T4)
            let mask = causal_mask(seq_len, total_len, start_pos, &k_full.device())?
                .to_dtype(scores.dtype())?;
            let scores = scores.broadcast_add(&mask)?;
            let scores = softmax_last_dim(&scores)?;

            // [nh, sl, total_len] @ [nh, total_len, hd] → [nh, sl, hd]
            let attn_raw = scores.matmul(&v_exp.contiguous()?)?;
            attn_raw.transpose(0, 1)?.contiguous()?.reshape((seq_len, num_heads * head_dim))?
        };

        // Output projection
        w.o_proj.matmul_x(&attn)
    }

    /// Build a causal (lower-triangular) mask for prefill attention.
    /// Returns a [1, seq_len, total_len] tensor with 0 for attended positions
    /// and -inf for masked (future) positions.
    fn causal_mask(
        seq_len: usize,
        total_len: usize,
        start_pos: usize,
        device: &CandleDev,
    ) -> candle_core::Result<CT> {
        // mask[s, t] = 0 if t <= start_pos + s else -inf
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * total_len];
        for s in 0..seq_len {
            let cutoff = start_pos + s + 1; // can attend to positions 0..cutoff
            for t in 0..cutoff.min(total_len) {
                mask_data[s * total_len + t] = 0.0;
            }
        }
        CT::from_slice(&mask_data, (1, seq_len, total_len), device)
    }

    // ── FFN ───────────────────────────────────────────────────────────────────

    fn gpu_ffn(normed: &CT, w: &GpuLayerWeights, ffn_type: FfnType, use_w4a8: bool) -> candle_core::Result<CT> {
        // W4A8 fast path: on CUDA with W4 weights and use_w4a8 set.
        // Quantize input once for gate+up, then re-quantize the activation output
        // for down_proj. This saves 1 quantization call vs quantizing 3× separately.
        #[cfg(feature = "cuda")]
        if normed.device().is_cuda()
            && std::env::var("NVE_NO_FUSED").is_err()
            && use_w4a8
            && normed.dim(0).map(|d| d == 1).unwrap_or(false)
        {
            let use_w4a8 = match ffn_type {
                FfnType::SwiGLU | FfnType::GeGLU => {
                    matches!(
                        (&w.gate_proj, &w.up_proj, &w.down_proj),
                        (Some(GpuWeight::W4(_)), GpuWeight::W4(_), GpuWeight::W4(_))
                    )
                }
                FfnType::GELU | FfnType::ReLU => {
                    matches!((&w.up_proj, &w.down_proj), (GpuWeight::W4(_), GpuWeight::W4(_)))
                }
            };

            if use_w4a8 {
                let (xq, xq_sc) = quantize_f16_q8(normed)?;
                return match ffn_type {
                    FfnType::SwiGLU => {
                        let wg = match w.gate_proj.as_ref().unwrap() { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wu = match &w.up_proj   { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wd = match &w.down_proj { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let gate = matvec_w4a8(&xq, &xq_sc, &wg.nibbles, &wg.scales, wg.n, wg.k)?.silu()?;
                        let up   = matvec_w4a8(&xq, &xq_sc, &wu.nibbles, &wu.scales, wu.n, wu.k)?;
                        let mid  = gate.mul(&up)?;
                        let (xq2, xs2) = quantize_f16_q8(&mid)?;
                        matvec_w4a8(&xq2, &xs2, &wd.nibbles, &wd.scales, wd.n, wd.k)
                    }
                    FfnType::GeGLU => {
                        let wg = match w.gate_proj.as_ref().unwrap() { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wu = match &w.up_proj   { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wd = match &w.down_proj { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let gate = matvec_w4a8(&xq, &xq_sc, &wg.nibbles, &wg.scales, wg.n, wg.k)?.gelu_erf()?;
                        let up   = matvec_w4a8(&xq, &xq_sc, &wu.nibbles, &wu.scales, wu.n, wu.k)?;
                        let mid  = gate.mul(&up)?;
                        let (xq2, xs2) = quantize_f16_q8(&mid)?;
                        matvec_w4a8(&xq2, &xs2, &wd.nibbles, &wd.scales, wd.n, wd.k)
                    }
                    FfnType::GELU => {
                        let wu = match &w.up_proj   { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wd = match &w.down_proj { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let mid = matvec_w4a8(&xq, &xq_sc, &wu.nibbles, &wu.scales, wu.n, wu.k)?.gelu_erf()?;
                        let (xq2, xs2) = quantize_f16_q8(&mid)?;
                        matvec_w4a8(&xq2, &xs2, &wd.nibbles, &wd.scales, wd.n, wd.k)
                    }
                    FfnType::ReLU => {
                        let wu = match &w.up_proj   { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let wd = match &w.down_proj { GpuWeight::W4(x) => x, _ => unreachable!() };
                        let mid = matvec_w4a8(&xq, &xq_sc, &wu.nibbles, &wu.scales, wu.n, wu.k)?.relu()?;
                        let (xq2, xs2) = quantize_f16_q8(&mid)?;
                        matvec_w4a8(&xq2, &xs2, &wd.nibbles, &wd.scales, wd.n, wd.k)
                    }
                };
            }
        }

        // Fallback: matmul_x() dispatches to W4A16 or F16 fused kernel for decode (batch=1),
        // or dequant+matmul for prefill (batch>1).
        match ffn_type {
            FfnType::SwiGLU => {
                let gate = w.gate_proj.as_ref().unwrap().matmul_x(normed)?.silu()?;
                let up   = w.up_proj.matmul_x(normed)?;
                w.down_proj.matmul_x(&gate.mul(&up)?)
            }
            FfnType::GeGLU => {
                let gate = w.gate_proj.as_ref().unwrap().matmul_x(normed)?.gelu_erf()?;
                let up   = w.up_proj.matmul_x(normed)?;
                w.down_proj.matmul_x(&gate.mul(&up)?)
            }
            FfnType::GELU => {
                w.down_proj.matmul_x(&w.up_proj.matmul_x(normed)?.gelu_erf()?)
            }
            FfnType::ReLU => {
                w.down_proj.matmul_x(&w.up_proj.matmul_x(normed)?.relu()?)
            }
        }
    }

    // ── Full layer forwards ───────────────────────────────────────────────────

    /// GPU decode forward for a single token through one transformer layer.
    ///
    /// hidden: [1, hidden_size] F32 on GPU
    /// Returns: [1, hidden_size] F32 on GPU
    pub fn gpu_layer_forward_decode(
        hidden: CT,
        layer_idx: usize,
        state: &mut GpuInferenceState,
        pos: usize,
        parallel_attn_ffn: bool,
        layer_importance: f64,
    ) -> candle_core::Result<CT> {
        // Determine precision for this layer.
        // NVE_NO_W4A8=1 → always W4A16.
        // Otherwise: use W4A8 when layer_importance < NVE_W4A8_THRESHOLD (default 0.7).
        // No profiling done (importance=0.0) → 0.0 < 0.7 → W4A8 (same as before).
        let use_w4a8 = if std::env::var("NVE_NO_W4A8").is_ok() {
            false
        } else {
            let threshold: f64 = std::env::var("NVE_W4A8_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7);
            layer_importance < threshold
        };

        let w = state.layer_weights[layer_idx]
            .as_ref()
            .expect("GPU weights not initialized for this layer");

        // Attention sublayer
        let normed = apply_norm_gpu(
            &hidden,
            &w.input_norm,
            w.input_norm_bias.as_ref(),
            state.norm_type,
            state.norm_eps,
        )?;

        let attn_out = gpu_attn_decode(
            &normed,
            w,
            &mut state.kv_cache,
            layer_idx,
            pos,
            state.num_heads,
            state.num_kv_heads,
            state.head_dim,
            state.rope_cos.as_ref().unwrap(),
            state.rope_sin.as_ref().unwrap(),
            use_w4a8,
        )?;

        let attn_out = if let Some(ref b) = w.o_bias {
            attn_out.broadcast_add(b)?
        } else {
            attn_out
        };

        if parallel_attn_ffn {
            // Parallel attention + FFN (GPT-NeoX / Falcon style)
            let ffn_normed = apply_norm_gpu(
                &hidden,
                &w.post_attn_norm,
                w.post_attn_norm_bias.as_ref(),
                state.norm_type,
                state.norm_eps,
            )?;
            let ffn_out = gpu_ffn(&ffn_normed, w, state.ffn_type, use_w4a8)?;
            hidden.add(&attn_out)?.add(&ffn_out)
        } else {
            // Sequential: attention then FFN
            let h = hidden.add(&attn_out)?;
            let normed2 = apply_norm_gpu(
                &h,
                &w.post_attn_norm,
                w.post_attn_norm_bias.as_ref(),
                state.norm_type,
                state.norm_eps,
            )?;
            let ffn_out = gpu_ffn(&normed2, w, state.ffn_type, use_w4a8)?;
            h.add(&ffn_out)
        }
    }

    /// GPU prefill forward for all prompt tokens through one transformer layer.
    ///
    /// hidden: [seq_len, hidden_size] F32 on GPU
    /// Returns: [seq_len, hidden_size] F32 on GPU
    pub fn gpu_layer_forward_prefill(
        hidden: CT,
        layer_idx: usize,
        state: &mut GpuInferenceState,
        start_pos: usize,
        parallel_attn_ffn: bool,
    ) -> candle_core::Result<CT> {
        let w = state.layer_weights[layer_idx]
            .as_ref()
            .expect("GPU weights not initialized for this layer");

        // Batch norm
        let normed = apply_norm_gpu(
            &hidden,
            &w.input_norm,
            w.input_norm_bias.as_ref(),
            state.norm_type,
            state.norm_eps,
        )?;

        let seq_len = hidden.dim(0)?;
        let attn_out = gpu_attn_prefill(
            &normed,
            w,
            &mut state.kv_cache,
            layer_idx,
            start_pos,
            state.num_heads,
            state.num_kv_heads,
            state.head_dim,
            state.rope_cos.as_ref().unwrap(),
            state.rope_sin.as_ref().unwrap(),
        )?;

        let attn_out = if let Some(ref b) = w.o_bias {
            // Broadcast bias across seq_len
            let b_2d = b.unsqueeze(0)?.broadcast_as((seq_len, b.dim(0)?))?;
            attn_out.add(&b_2d)?
        } else {
            attn_out
        };

        if parallel_attn_ffn {
            let ffn_normed = apply_norm_gpu(
                &hidden,
                &w.post_attn_norm,
                w.post_attn_norm_bias.as_ref(),
                state.norm_type,
                state.norm_eps,
            )?;
            // Prefill: seq_len > 1, gpu_ffn's batch-size guard prevents W4A8 anyway.
            let ffn_out = gpu_ffn(&ffn_normed, w, state.ffn_type, false)?;
            hidden.add(&attn_out)?.add(&ffn_out)
        } else {
            let h = hidden.add(&attn_out)?;
            let normed2 = apply_norm_gpu(
                &h,
                &w.post_attn_norm,
                w.post_attn_norm_bias.as_ref(),
                state.norm_type,
                state.norm_eps,
            )?;
            let ffn_out = gpu_ffn(&normed2, w, state.ffn_type, false)?;
            h.add(&ffn_out)
        }
    }

    /// Convert a GPU tensor [1, hidden] or [seq, hidden] to a CPU Vec<f32>.
    pub fn gpu_to_cpu_vec(t: CT) -> candle_core::Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
    }
}
