//! CUDA graph capture for single-token (decode) inference.
//!
//! ## What this solves
//!
//! A single decode step launches ~200 CUDA kernels sequentially.  Each launch
//! incurs ~3 µs of CPU-side overhead → ~0.6 ms/token wasted on kernel dispatch.
//! Additionally `GpuKvCache` uses `CT::cat` which allocates + frees tensors on
//! every step (32 allocs for 16 layers × K+V).
//!
//! `StaticDecodeGraph` eliminates both overheads:
//!
//! 1. **Static KV cache** — pre-allocated `[num_kv_heads, max_seq_len, head_dim]`
//!    per layer; writes use the scatter-write kernel `nve_kv_cache_write_f16`.
//!
//! 2. **CUDA graph** — the full multi-layer decode pass is captured once into a
//!    `cudaGraphExec_t`.  Each step updates two pinned host scalars (pos, seq_len)
//!    and calls `cudaGraphLaunch`; all 200+ kernels fire in one driver call.
//!
//! ## Requirements
//!
//! - `--features cuda`
//! - All projection weights must be `GpuWeight::W4` (Q4_0 quantised)
//! - `NVE_NO_FUSED` must be unset
//! - `NVE_NO_FLASH` must be unset
//! - `head_dim % 32 == 0` and `head_dim / 32 ≤ 16` (flash decode constraint)
//!
//! ## Usage
//!
//! ```rust
//! // After building GpuInferenceState with all W4 weights:
//! state.build_decode_graph(max_seq_len)?;
//!
//! // Decode loop (replaces per-layer gpu_layer_forward_decode):
//! for step in 0..n_tokens {
//!     let out = state.replay_decode_graph(&input_embedding, step)?;
//!     state.reset_decode_graph_kv();  // only if you want to reset KV cache
//! }
//! ```

#[cfg(feature = "cuda")]
pub use cuda_graph::*;

#[cfg(feature = "cuda")]
mod cuda_graph {
    use candle_core::{DType, Result, Tensor as CT};
    use crate::cuda_kernels::{
        add_inplace, device_copy, flash_decode_dyn_into, fused_matvec_w4a16_into,
        fused_qkv_matvec_w4a16_into, fused_rms_norm_into, graph_begin_capture,
        graph_d_pos_ptr, graph_d_seqlen_ptr, graph_destroy, graph_end_capture,
        graph_init, graph_launch, graph_ready, kv_cache_write, rope_decode_dyn_inplace,
        silu_mul_inplace,
    };
    use crate::gpu_layer::{GpuInferenceState, GpuWeight};
    use crate::arch::{FfnType, NormType};

    // ── Per-layer intermediate buffers ────────────────────────────────────────

    /// Pre-allocated scratch tensors for one transformer layer.
    /// All shapes are flat (1D) — the kernels work on raw pointers.
    struct LayerBufs {
        normed:  CT,   // [hidden]           — after input RMSNorm
        q_buf:   CT,   // [num_heads*head_dim]
        k_buf:   CT,   // [num_kv_heads*head_dim]
        v_buf:   CT,   // [num_kv_heads*head_dim]
        attn:    CT,   // [num_heads*head_dim] — flash decode output
        o_out:   CT,   // [hidden]           — o_proj output
        normed2: CT,   // [hidden]           — after post-attn RMSNorm
        gate:    CT,   // [intermediate]
        up:      CT,   // [intermediate]
        down:    CT,   // [hidden]           — down_proj output
    }

    impl LayerBufs {
        fn new(
            device:       &candle_core::Device,
            hidden:       usize,
            intermediate: usize,
            num_heads:    usize,
            num_kv_heads: usize,
            head_dim:     usize,
        ) -> Result<Self> {
            let z = |n: usize| CT::zeros((n,), DType::F16, device);
            Ok(Self {
                normed:  z(hidden)?,
                q_buf:   z(num_heads * head_dim)?,
                k_buf:   z(num_kv_heads * head_dim)?,
                v_buf:   z(num_kv_heads * head_dim)?,
                attn:    z(num_heads * head_dim)?,
                o_out:   z(hidden)?,
                normed2: z(hidden)?,
                gate:    z(intermediate)?,
                up:      z(intermediate)?,
                down:    z(hidden)?,
            })
        }
    }

    // ── Static decode graph ───────────────────────────────────────────────────

    /// Captured CUDA graph for multi-layer decode with static KV cache.
    ///
    /// Call `capture()` once after all weights are loaded, then `replay()` each
    /// decode step instead of the per-layer `gpu_layer_forward_decode` loop.
    pub struct StaticDecodeGraph {
        /// Shared hidden-state buffer [hidden]: input and output of the graph.
        hidden_buf:   CT,
        layer_bufs:   Vec<LayerBufs>,
        /// Static KV caches: [num_kv_heads, max_seq_len, head_dim] per layer.
        k_caches:     Vec<CT>,
        v_caches:     Vec<CT>,
        /// Logical fill levels (not stored on GPU; the GPU just has static buffers).
        kv_seq_lens:  Vec<usize>,

        pub hidden:       usize,
        pub intermediate: usize,
        pub num_heads:    usize,
        pub num_kv_heads: usize,
        pub head_dim:     usize,
        pub num_layers:   usize,
        pub max_seq_len:  usize,
    }

    impl Drop for StaticDecodeGraph {
        fn drop(&mut self) {
            graph_destroy();
        }
    }

    impl StaticDecodeGraph {
        // ── Construction ──────────────────────────────────────────────────────

        /// Capture the decode graph from `state`.
        ///
        /// Preconditions (panics otherwise):
        /// - All active layers have W4 weights for Q/K/V/O/gate/up/down.
        /// - `NVE_NO_FUSED` and `NVE_NO_FLASH` are unset.
        /// - `state.head_dim % 32 == 0` and `state.head_dim / 32 ≤ 16`.
        pub fn capture(
            state:       &GpuInferenceState,
            max_seq_len: usize,
        ) -> Result<Self> {
            let dev          = &state.device;
            let hidden       = state.num_heads * state.head_dim;
            let num_kv_heads = state.num_kv_heads;
            let num_heads    = state.num_heads;
            let head_dim     = state.head_dim;
            let num_layers   = state.num_layers;

            // Determine intermediate dimension from gate_proj shape.
            let intermediate = {
                let l0 = state.layer_weights.iter()
                    .find_map(|l| l.as_ref())
                    .expect("no GPU layers loaded");
                match &l0.gate_proj {
                    Some(GpuWeight::W4(w)) => w.n,
                    Some(GpuWeight::F16(t)) => t.dim(0)?,
                    None => candle_core::bail!("StaticDecodeGraph: gate_proj required (SwiGLU)"),
                }
            };

            if std::env::var("NVE_NO_FUSED").is_ok() {
                candle_core::bail!("StaticDecodeGraph requires NVE_NO_FUSED to be unset");
            }
            if std::env::var("NVE_NO_FLASH").is_ok() {
                candle_core::bail!("StaticDecodeGraph requires NVE_NO_FLASH to be unset");
            }
            if head_dim % 32 != 0 || head_dim / 32 > 16 {
                candle_core::bail!("head_dim must be divisible by 32 and ≤ 512 for graph capture");
            }
            if state.norm_type != NormType::RMSNorm {
                candle_core::bail!("StaticDecodeGraph: only RMSNorm is supported");
            }

            // ── Pre-allocate buffers ───────────────────────────────────────────
            let hidden_buf = CT::zeros((hidden,), DType::F16, dev)?;

            let mut layer_bufs = Vec::with_capacity(num_layers);
            let mut k_caches   = Vec::with_capacity(num_layers);
            let mut v_caches   = Vec::with_capacity(num_layers);

            for _ in 0..num_layers {
                layer_bufs.push(LayerBufs::new(dev, hidden, intermediate, num_heads, num_kv_heads, head_dim)?);
                k_caches.push(CT::zeros((num_kv_heads, max_seq_len, head_dim), DType::F16, dev)?);
                v_caches.push(CT::zeros((num_kv_heads, max_seq_len, head_dim), DType::F16, dev)?);
            }

            let kv_seq_lens = vec![0usize; num_layers];

            let rope_cos = state.rope_cos.as_ref()
                .expect("StaticDecodeGraph: rope_cos required");
            let rope_sin = state.rope_sin.as_ref()
                .expect("StaticDecodeGraph: rope_sin required");

            let scale = (head_dim as f32).powf(-0.5);
            let norm_eps = state.norm_eps;

            // ── CUDA graph capture ─────────────────────────────────────────────
            // graph_init() MUST be called before graph_d_pos_ptr() / graph_d_seqlen_ptr()
            // so that g_dev_pos / g_dev_seqlen are allocated.
            graph_init();
            let d_pos     = graph_d_pos_ptr();
            let d_seqlen  = graph_d_seqlen_ptr();

            graph_begin_capture();  // records pinned-memcpy nodes + sets thread-local stream

            for li in 0..num_layers {
                let w = state.layer_weights[li].as_ref()
                    .expect("GPU layer weights missing during graph capture");
                let bufs  = &layer_bufs[li];
                let k_c   = &k_caches[li];
                let v_c   = &v_caches[li];

                // 1. Input RMSNorm: normed = RMSNorm(hidden_buf, norm_w)
                // to_dtype(F16) is a no-op if already F16; converts F32 otherwise.
                let norm_w_flat = w.input_norm.flatten_all()?.to_dtype(DType::F16)?;
                fused_rms_norm_into(&hidden_buf, &norm_w_flat, &bufs.normed, norm_eps)?;

                // 2. Q/K/V projections (fused if all W4, else separate)
                match (&w.q_proj, &w.k_proj, &w.v_proj) {
                    (GpuWeight::W4(wq), GpuWeight::W4(wk), GpuWeight::W4(wv)) => {
                        fused_qkv_matvec_w4a16_into(
                            &bufs.normed,
                            &wq.nibbles, &wq.scales, wq.awq_scales.as_ref(), wq.n,
                            &wk.nibbles, &wk.scales, wk.awq_scales.as_ref(), wk.n,
                            &wv.nibbles, &wv.scales, wv.awq_scales.as_ref(), wv.n,
                            wq.k,
                            &bufs.q_buf, &bufs.k_buf, &bufs.v_buf,
                        )?;
                    }
                    _ => {
                        // F16 path — fallback (less common in production W4 models)
                        match &w.q_proj {
                            GpuWeight::F16(wq) => {
                                let (n, k) = wq.dims2()?;
                                fused_matvec_into_flat(&bufs.normed, wq, &bufs.q_buf, n, k)?;
                            }
                            GpuWeight::W4(wq) => {
                                fused_matvec_w4a16_into(&bufs.normed, &wq.nibbles, &wq.scales, wq.awq_scales.as_ref(), &bufs.q_buf, wq.n, wq.k)?;
                            }
                        }
                        match &w.k_proj {
                            GpuWeight::F16(wk) => {
                                let (n, k) = wk.dims2()?;
                                fused_matvec_into_flat(&bufs.normed, wk, &bufs.k_buf, n, k)?;
                            }
                            GpuWeight::W4(wk) => {
                                fused_matvec_w4a16_into(&bufs.normed, &wk.nibbles, &wk.scales, wk.awq_scales.as_ref(), &bufs.k_buf, wk.n, wk.k)?;
                            }
                        }
                        match &w.v_proj {
                            GpuWeight::F16(wv) => {
                                let (n, k) = wv.dims2()?;
                                fused_matvec_into_flat(&bufs.normed, wv, &bufs.v_buf, n, k)?;
                            }
                            GpuWeight::W4(wv) => {
                                fused_matvec_w4a16_into(&bufs.normed, &wv.nibbles, &wv.scales, wv.awq_scales.as_ref(), &bufs.v_buf, wv.n, wv.k)?;
                            }
                        }
                    }
                }

                // 3. Dynamic RoPE (reads pos from device scalar)
                rope_decode_dyn_inplace(
                    &bufs.q_buf, &bufs.k_buf,
                    rope_cos, rope_sin,
                    num_heads, num_kv_heads, head_dim, d_pos,
                )?;

                // 4. KV cache scatter-write at *d_pos
                kv_cache_write(
                    &bufs.k_buf, k_c,
                    &bufs.v_buf, v_c,
                    d_pos, num_kv_heads, head_dim, max_seq_len,
                )?;

                // 5. Dynamic flash decode (reads seq_len from device scalar)
                flash_decode_dyn_into(
                    &bufs.q_buf, k_c, v_c, &bufs.attn,
                    num_heads, num_kv_heads, head_dim, scale,
                    max_seq_len, d_seqlen,
                )?;

                // 6. o_proj
                match &w.o_proj {
                    GpuWeight::W4(wo) => {
                        fused_matvec_w4a16_into(&bufs.attn, &wo.nibbles, &wo.scales, wo.awq_scales.as_ref(), &bufs.o_out, wo.n, wo.k)?;
                    }
                    GpuWeight::F16(wo) => {
                        let (n, k) = wo.dims2()?;
                        fused_matvec_into_flat(&bufs.attn, wo, &bufs.o_out, n, k)?;
                    }
                }

                // 7. Residual: hidden_buf += o_out
                add_inplace(&hidden_buf, &bufs.o_out)?;

                // 8. Post-attention RMSNorm
                let norm2_w = w.post_attn_norm.flatten_all()?.to_dtype(DType::F16)?;
                fused_rms_norm_into(&hidden_buf, &norm2_w, &bufs.normed2, norm_eps)?;

                // 9. FFN — only SwiGLU supported in graph mode
                if state.ffn_type != FfnType::SwiGLU {
                    candle_core::bail!("StaticDecodeGraph: only SwiGLU FFN is supported");
                }

                let gate_w = w.gate_proj.as_ref()
                    .expect("SwiGLU requires gate_proj");
                match gate_w {
                    GpuWeight::W4(wg) => {
                        fused_matvec_w4a16_into(&bufs.normed2, &wg.nibbles, &wg.scales, wg.awq_scales.as_ref(), &bufs.gate, wg.n, wg.k)?;
                    }
                    GpuWeight::F16(wg) => {
                        let (n, k) = wg.dims2()?;
                        fused_matvec_into_flat(&bufs.normed2, wg, &bufs.gate, n, k)?;
                    }
                }
                match &w.up_proj {
                    GpuWeight::W4(wu) => {
                        fused_matvec_w4a16_into(&bufs.normed2, &wu.nibbles, &wu.scales, wu.awq_scales.as_ref(), &bufs.up, wu.n, wu.k)?;
                    }
                    GpuWeight::F16(wu) => {
                        let (n, k) = wu.dims2()?;
                        fused_matvec_into_flat(&bufs.normed2, wu, &bufs.up, n, k)?;
                    }
                }

                // 10. SiLU-Mul: gate = silu(gate) * up
                silu_mul_inplace(&bufs.gate, &bufs.up)?;

                // 11. down_proj into down buf
                match &w.down_proj {
                    GpuWeight::W4(wd) => {
                        fused_matvec_w4a16_into(&bufs.gate, &wd.nibbles, &wd.scales, wd.awq_scales.as_ref(), &bufs.down, wd.n, wd.k)?;
                    }
                    GpuWeight::F16(wd) => {
                        let (n, k) = wd.dims2()?;
                        fused_matvec_into_flat(&bufs.gate, wd, &bufs.down, n, k)?;
                    }
                }

                // 12. Residual: hidden_buf += down
                add_inplace(&hidden_buf, &bufs.down)?;
            }

            graph_end_capture();

            if !graph_ready() {
                candle_core::bail!("StaticDecodeGraph: graph instantiation failed");
            }

            log::info!(
                "[graph] Captured {}-layer decode graph ({} heads, {}/{} KV, hd={}, max_seq={})",
                num_layers, num_heads, num_kv_heads, num_heads, head_dim, max_seq_len,
            );

            Ok(Self {
                hidden_buf,
                layer_bufs,
                k_caches,
                v_caches,
                kv_seq_lens,
                hidden,
                intermediate,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                max_seq_len,
            })
        }

        // ── Replay ────────────────────────────────────────────────────────────

        /// Replay the captured decode graph for one token.
        ///
        /// `input`: [hidden] or [1, hidden] F16 on the same CUDA device.
        /// `pos`:   current token position (0-indexed).
        ///
        /// Updates the static KV cache at position `pos` then runs all layers.
        /// Returns the hidden state after the last layer (same device memory as
        /// `hidden_buf` — the caller must consume it before the next replay).
        pub fn replay(&mut self, input: &CT, pos: usize) -> Result<CT> {
            // D2D copy: flatten input into hidden_buf (both F16, same device).
            let input_flat = if input.dims().len() > 1 {
                input.flatten_all()?
            } else {
                input.clone()
            };
            device_copy(&self.hidden_buf, &input_flat)?;

            // seq_len after this step's KV write = pos + 1
            let seq_len = pos + 1;
            for li in 0..self.num_layers {
                self.kv_seq_lens[li] = seq_len.min(self.max_seq_len);
            }

            // Launch the graph: memcpy pos→d_pos and seq_len→d_seqlen are the
            // first nodes, followed by all layer kernels.
            graph_launch(pos, seq_len.min(self.max_seq_len));

            // Return a view of the output buffer.  The caller should consume this
            // (e.g. run lm_head) before the next replay overwrites it.
            Ok(self.hidden_buf.clone())
        }

        /// Reset the logical KV cache fill levels (call after each generation).
        /// This does NOT zero GPU memory — old values will be overwritten on the
        /// next pass through each position.
        pub fn reset_kv(&mut self) {
            for sl in &mut self.kv_seq_lens { *sl = 0; }
        }

        /// Current fill level for a given layer's KV cache.
        pub fn kv_seq_len(&self, layer: usize) -> usize {
            self.kv_seq_lens[layer]
        }
    }

    // ── Internal helper ───────────────────────────────────────────────────────

    /// F16 matvec using flat tensors (bypasses shape checks, works on 1D buffers).
    fn fused_matvec_into_flat(
        x:   &CT,
        w:   &CT,
        out: &CT,
        n:   usize,
        k:   usize,
    ) -> Result<()> {
        use crate::cuda_kernels::fused_matvec_into;
        fused_matvec_into(x, w, out, n, k)
    }
}
