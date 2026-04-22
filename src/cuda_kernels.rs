//! NVE custom CUDA kernels — Rust FFI bindings and safe wrappers.
//!
//! Provides fused replacements for:
//! - `rms_norm`      — single-pass RMSNorm (7 candle ops → 1 kernel)
//! - `layer_norm`    — single-pass LayerNorm (6 candle ops → 1 kernel)
//! - `rope_decode`   — fused in-place RoPE for single-token decode (7N ops → 1 kernel)
//! - `rope_prefill`  — fused in-place RoPE for prefill batch
//!
//! All ops take candle F16 tensors and delegate to CUDA kernels compiled from
//! `cuda/nve_kernels.cu` via the static library built in `build.rs`.
//!
//! Only available when built with `--features cuda`.
//!
//! ```
//! CUDA_COMPUTE_CAP=75 cargo build --release --features cuda
//! ```

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::{DType, Result, Tensor};
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use std::cell::Cell;

    // ── Per-thread stream override for CUDA graph capture ─────────────────────
    //
    // During graph capture all kernels must execute on the capture stream
    // (not the CUDA default/legacy stream, which cannot be captured in global
    // capture mode).  Set this before calling `nve_graph_begin_capture()` and
    // clear it afterwards.  All kernel wrappers call `stream()` instead of
    // hard-coding `null_mut()`.

    thread_local! {
        static GRAPH_STREAM: Cell<*mut std::ffi::c_void> =
            Cell::new(std::ptr::null_mut());
    }

    /// Active CUDA stream: capture stream during graph build, null otherwise.
    pub fn stream() -> *mut std::ffi::c_void {
        GRAPH_STREAM.with(|s| s.get())
    }

    /// Activate the given capture stream for this thread.
    pub fn set_graph_capture_stream(s: *mut std::ffi::c_void) {
        GRAPH_STREAM.with(|c| c.set(s));
    }

    /// Deactivate: revert to CUDA default stream.
    pub fn clear_graph_capture_stream() {
        GRAPH_STREAM.with(|c| c.set(std::ptr::null_mut()));
    }

    // ── FFI declarations — must match cuda/nve_kernels.cu host launchers ──────

    extern "C" {
        /// Fused RMSNorm: out[i] = x[i] / rms(x) * weight[i]
        fn nve_rms_norm_f16(
            x:      *const u16,
            weight: *const u16,
            out:    *mut u16,
            n:      i32,
            eps:    f32,
            stream: *mut std::ffi::c_void, // cudaStream_t; null → default stream
        );

        /// Fused LayerNorm: out[i] = (x[i] - mean) / std * weight[i] + bias[i]
        fn nve_layer_norm_f16(
            x:      *const u16,
            weight: *const u16,
            bias:   *const u16, // may be null
            out:    *mut u16,
            n:      i32,
            eps:    f32,
            stream: *mut std::ffi::c_void,
        );

        /// Fused in-place RoPE for single-token decode.
        /// q/k are modified in-place.
        fn nve_rope_f16_decode(
            q:           *mut u16,   // [num_heads, head_dim]
            k:           *mut u16,   // [num_kv_heads, head_dim]
            cos:         *const u16, // [half_dim]
            sin:         *const u16, // [half_dim]
            num_heads:   i32,
            num_kv_heads: i32,
            head_dim:    i32,
            stream:      *mut std::ffi::c_void,
        );

        /// Fused F16 matvec: x[1,K] × W[N,K]^T → out[1,N].
        /// One warp per output row; warp-shuffle reduction.  Batch=1 only.
        fn nve_matvec_f16(
            x:      *const u16,   // [1, K]
            w:      *const u16,   // [N, K]
            out:    *mut u16,     // [1, N]
            n:      i32,
            k:      i32,
            stream: *mut std::ffi::c_void,
        );

        /// Fused in-place RoPE for prefill batch.
        fn nve_rope_f16_prefill(
            q:           *mut u16,
            k:           *mut u16,
            cos:         *const u16, // [max_seq, half_dim]
            sin:         *const u16,
            seq_len:     i32,
            num_heads:   i32,
            num_kv_heads: i32,
            head_dim:    i32,
            start_pos:   i32,
            stream:      *mut std::ffi::c_void,
        );

        /// W4A16 matvec: x[1,K] F16 × W_packed[N,K] INT4 → out[1,N] F16.
        /// nibbles: [N, K/2] packed Q4_0 nibbles (2 INT4 per byte, sequential pairs).
        /// scales:  [N, K/32] per-block f32 scales.
        /// awq:     [K] f32 AWQ input column scales, or null for no AWQ.
        fn nve_matvec_w4a16(
            x:       *const u16,  // [1, K] F16
            nibbles: *const u8,   // [N, K/2] packed nibbles
            scales:  *const f32,  // [N, K/32] per-block scales
            awq:     *const f32,  // [K] AWQ input scales (may be null)
            out:     *mut u16,    // [1, N] F16
            n:       i32,
            k:       i32,
            stream:  *mut std::ffi::c_void,
        );

        /// W4A16 dequantize: W_packed[N,K] INT4 → W_f16[N,K] F16.
        /// Used by the prefill path to convert packed INT4 to F16 before batched matmul.
        fn nve_dequant_w4a16(
            nibbles: *const u8,   // [N, K/2]
            scales:  *const f32,  // [N, K/32]
            awq:     *const f32,  // [K] or null
            out:     *mut u16,    // [N, K] F16
            n:       i32,
            k:       i32,
            stream:  *mut std::ffi::c_void,
        );

        /// Flash decode attention: fused Q@K^T + online-softmax + scores@V.
        /// One warp per head; GQA-native (kv_head = head / groups).
        /// q:        [num_heads, head_dim]             F16
        /// k_cache:  [num_kv_heads, seq_len, head_dim] F16
        /// v_cache:  [num_kv_heads, seq_len, head_dim] F16
        /// out:      [num_heads, head_dim]              F16
        fn nve_flash_decode_f16(
            q:           *const u16,
            k_cache:     *const u16,
            v_cache:     *const u16,
            out:         *mut u16,
            num_heads:   i32,
            num_kv_heads: i32,
            seq_len:     i32,
            head_dim:    i32,
            scale:       f32,
            stream:      *mut std::ffi::c_void,
        );

        /// Fused Q/K/V W4A16 matvec: three projections in one kernel launch.
        /// x:      [K]      F16 input
        /// {q,k,v}_nib:  [N{q,k,v}, K/2]  U8 packed nibbles
        /// {q,k,v}_sc:   [N{q,k,v}, K/32] F32 per-block scales
        /// {q,k,v}_awq:  [K] F32 AWQ scales (may be null)
        /// {q,k,v}_out:  [N{q,k,v}] F16 outputs
        #[allow(clippy::too_many_arguments)]
        fn nve_qkv_matvec_w4a16(
            x:      *const u16,
            q_nib:  *const u8,  q_sc: *const f32, q_awq: *const f32, nq: i32,
            k_nib:  *const u8,  k_sc: *const f32, k_awq: *const f32, nk: i32,
            v_nib:  *const u8,  v_sc: *const f32, v_awq: *const f32, nv: i32,
            q_out:  *mut u16,
            k_out:  *mut u16,
            v_out:  *mut u16,
            k_dim:  i32,
            stream: *mut std::ffi::c_void,
        );

        // ── Graph-capture-safe helpers (new kernels) ──────────────────────────

        /// Scatter-write K+V into static pre-allocated cache at position *d_pos.
        fn nve_kv_cache_write_f16(
            k_new:        *const u16,
            k_cache:      *mut u16,
            v_new:        *const u16,
            v_cache:      *mut u16,
            d_pos:        *const i32,
            num_kv_heads: i32,
            head_dim:     i32,
            max_seq_len:  i32,
            stream:       *mut std::ffi::c_void,
        );

        /// In-place element-wise add: dst[i] += src[i]
        fn nve_add_inplace_f16(
            dst:    *mut u16,
            src:    *const u16,
            n:      i32,
            stream: *mut std::ffi::c_void,
        );

        /// Fused SiLU-Mul: gate[i] = silu(gate[i]) * up[i]  (in-place on gate)
        fn nve_silu_mul_f16(
            gate:   *mut u16,
            up:     *const u16,
            n:      i32,
            stream: *mut std::ffi::c_void,
        );

        /// RoPE decode reading pos from device scalar; takes full [max_seq, half_dim] table.
        fn nve_rope_f16_decode_dyn(
            q:           *mut u16,
            k:           *mut u16,
            cos_tab:     *const u16,
            sin_tab:     *const u16,
            num_heads:   i32,
            num_kv_heads: i32,
            head_dim:    i32,
            d_pos:       *const i32,
            stream:      *mut std::ffi::c_void,
        );

        /// Flash decode reading seq_len from device scalar; k/v_cache are the
        /// full static buffers [num_kv_heads, max_seq_len, head_dim].
        #[allow(clippy::too_many_arguments)]
        fn nve_flash_decode_dyn(
            q:           *const u16,
            k_cache:     *const u16,
            v_cache:     *const u16,
            out:         *mut u16,
            num_heads:   i32,
            num_kv_heads: i32,
            d_seq_len:   *const i32,
            head_dim:    i32,
            scale:       f32,
            max_seq_len: i32,
            stream:      *mut std::ffi::c_void,
        );

        /// Device-to-device async copy.
        fn nve_d2d_copy(
            dst:    *mut std::ffi::c_void,
            src:    *const std::ffi::c_void,
            bytes:  usize,
            stream: *mut std::ffi::c_void,
        );

        // ── W4A8 (dp4a) kernels ───────────────────────────────────────────────

        /// Quantize F16 activation vector to Q8: one group of 32 per CUDA block.
        /// x:      [K] F16 on CUDA
        /// xq:     [K] I8 output (stored as U8 in Rust, same bits)
        /// scales: [K/32] F32 per-group scales
        fn nve_quantize_f16_q8(
            x:      *const u16,
            xq:     *mut u8,   // int8_t* in C — same bits
            scales: *mut f32,
            k:      i32,
            stream: *mut std::ffi::c_void,
        );

        /// W4A8 fused decode matvec using dp4a: xq[K] × W_int4[N,K]ᵀ → out[N] F16.
        /// 4 warps per output row; bias-corrected dp4a for unsigned nibbles.
        fn nve_matvec_w4a8(
            xq:        *const u8,  // int8_t* in C — Q8 activations
            xq_scales: *const f32, // [K/32] per-group activation scales
            nibbles:   *const u8,  // [N, K/2] packed W4 nibbles (llama.cpp GPU format)
            wt_scales: *const f32, // [N, K/32] per-block weight scales
            out:       *mut u16,   // [N] F16 output
            n:         i32,
            k:         i32,
            stream:    *mut std::ffi::c_void,
        );

        // ── CUDA graph management ─────────────────────────────────────────────

        fn nve_graph_init();
        fn nve_graph_begin_capture();
        fn nve_graph_end_capture();
        fn nve_graph_launch(pos: i32, seq_len: i32, stream: *mut std::ffi::c_void);
        fn nve_graph_ready() -> i32;
        fn nve_graph_destroy();
        fn nve_graph_d_pos()    -> *const i32;
        fn nve_graph_d_seqlen() -> *const i32;
        fn nve_graph_capture_stream() -> *mut std::ffi::c_void;
    }

    // ── Helpers: extract raw CUDA device pointers from candle Tensors ────────
    //
    // Returns the CUdeviceptr (GPU address) as u64, accounting for the tensor's
    // start offset (for slices / narrowed tensors).
    //
    // Safety: The returned pointer is valid as long as `t` is alive and not
    // mutated by another CUDA op on a different stream.  The caller must ensure
    // the kernel has completed before the Tensor is dropped.

    fn cuda_dev_ptr(t: &Tensor) -> Result<u64> {
        assert_eq!(t.dtype(), DType::F16, "cuda_dev_ptr: tensor must be F16");
        let (storage, layout) = t.storage_and_layout();
        let offset_bytes = layout.start_offset() as u64 * 2; // 2 bytes per f16
        match &*storage {
            candle_core::Storage::Cuda(s) => {
                let slice = s.as_cuda_slice::<half::f16>()?;
                Ok(*slice.device_ptr() + offset_bytes)
            }
            _ => candle_core::bail!("cuda_dev_ptr: tensor not on CUDA device"),
        }
    }

    /// Extract a raw CUDA pointer from a U8 candle Tensor (packed nibbles).
    fn cuda_dev_ptr_u8(t: &Tensor) -> Result<u64> {
        assert_eq!(t.dtype(), DType::U8, "cuda_dev_ptr_u8: tensor must be U8");
        let (storage, layout) = t.storage_and_layout();
        let offset_bytes = layout.start_offset() as u64; // 1 byte per u8
        match &*storage {
            candle_core::Storage::Cuda(s) => {
                let slice = s.as_cuda_slice::<u8>()?;
                Ok(*slice.device_ptr() + offset_bytes)
            }
            _ => candle_core::bail!("cuda_dev_ptr_u8: tensor not on CUDA device"),
        }
    }

    /// Extract a raw CUDA pointer from an F32 candle Tensor (scales / AWQ).
    fn cuda_dev_ptr_f32(t: &Tensor) -> Result<u64> {
        assert_eq!(t.dtype(), DType::F32, "cuda_dev_ptr_f32: tensor must be F32");
        let (storage, layout) = t.storage_and_layout();
        let offset_bytes = layout.start_offset() as u64 * 4; // 4 bytes per f32
        match &*storage {
            candle_core::Storage::Cuda(s) => {
                let slice = s.as_cuda_slice::<f32>()?;
                Ok(*slice.device_ptr() + offset_bytes)
            }
            _ => candle_core::bail!("cuda_dev_ptr_f32: tensor not on CUDA device"),
        }
    }

    // ── Public fused-kernel wrappers ──────────────────────────────────────────

    /// Fused single-pass RMSNorm on a [hidden] F16 GPU tensor.
    ///
    /// Replaces candle's 7-op sequence (to_f32 → sqr → mean → affine → sqrt →
    /// broadcast_div → broadcast_mul → to_f16) with a single CUDA kernel.
    ///
    /// `x`:      [hidden] F16 on CUDA
    /// `weight`: [hidden] F16 on CUDA
    /// Returns:  [hidden] F16 on CUDA
    pub fn fused_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let n = x.elem_count() as i32;
        let out = x.zeros_like()?;

        let x_ptr = cuda_dev_ptr(x)? as *const u16;
        let w_ptr = cuda_dev_ptr(weight)? as *const u16;
        let o_ptr = cuda_dev_ptr(&out)? as *mut u16;

        // Safety: x, weight, out are distinct GPU allocations; kernel is launch-async
        // on the default CUDA stream.  out is freshly allocated and has no readers.
        unsafe {
            nve_rms_norm_f16(x_ptr, w_ptr, o_ptr, n, eps as f32, stream());
        }
        Ok(out)
    }

    /// Fused single-pass LayerNorm on a [hidden] F16 GPU tensor.
    ///
    /// `x`:      [hidden] F16 on CUDA
    /// `weight`: [hidden] F16 on CUDA
    /// `bias`:   [hidden] F16 on CUDA, or None
    /// Returns:  [hidden] F16 on CUDA
    pub fn fused_layer_norm(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Tensor> {
        let n = x.elem_count() as i32;
        let out = x.zeros_like()?;

        let x_ptr = cuda_dev_ptr(x)? as *const u16;
        let w_ptr = cuda_dev_ptr(weight)? as *const u16;
        let b_ptr = bias
            .map(|b| cuda_dev_ptr(b).map(|p| p as *const u16))
            .transpose()?
            .unwrap_or(std::ptr::null());
        let o_ptr = cuda_dev_ptr(&out)? as *mut u16;

        unsafe {
            nve_layer_norm_f16(x_ptr, w_ptr, b_ptr, o_ptr, n, eps as f32, stream());
        }
        Ok(out)
    }

    /// Fused in-place RoPE for single-token decode.
    ///
    /// Rotates Q and K in-place using precomputed cos/sin tables.
    /// Replaces 7×num_heads candle ops with a single kernel covering all heads.
    ///
    /// `q`:       [num_heads, head_dim]     F16 on CUDA — modified in-place
    /// `k`:       [num_kv_heads, head_dim]  F16 on CUDA — modified in-place
    /// `cos_row`: [half_dim]                F16 on CUDA — cos for current position
    /// `sin_row`: [half_dim]                F16 on CUDA — sin for current position
    pub fn fused_rope_decode(
        q:           &Tensor,
        k:           &Tensor,
        cos_row:     &Tensor,
        sin_row:     &Tensor,
        num_heads:   usize,
        num_kv_heads: usize,
        head_dim:    usize,
    ) -> Result<()> {
        let q_ptr = cuda_dev_ptr(q)? as *mut u16;
        let k_ptr = cuda_dev_ptr(k)? as *mut u16;
        let c_ptr = cuda_dev_ptr(cos_row)? as *const u16;
        let s_ptr = cuda_dev_ptr(sin_row)? as *const u16;

        // Safety: q and k are the only current mutable users; cos/sin are read-only.
        unsafe {
            nve_rope_f16_decode(
                q_ptr, k_ptr, c_ptr, s_ptr,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                stream(),
            );
        }
        Ok(())
    }

    /// Fused F16 matvec for single-token decode: x[1,K] × W[N,K]ᵀ → [1,N].
    ///
    /// Drop-in for `x.matmul(&w.t()?)` when batch=1 on CUDA F16.
    /// Each output element is computed by one warp using shuffle reduction,
    /// achieving ~81% of T4's 320 GB/s memory bandwidth.
    ///
    /// `x`: [1, K] F16 on CUDA
    /// `w`: [N, K] F16 on CUDA
    /// Returns: [1, N] F16 on CUDA
    pub fn fused_matvec(x: &Tensor, w: &Tensor, n: usize, k: usize) -> Result<Tensor> {
        let out = Tensor::zeros((1, n), DType::F16, x.device())?;
        let x_ptr = cuda_dev_ptr(x)? as *const u16;
        let w_ptr = cuda_dev_ptr(w)? as *const u16;
        let o_ptr = cuda_dev_ptr(&out)? as *mut u16;
        unsafe {
            nve_matvec_f16(x_ptr, w_ptr, o_ptr, n as i32, k as i32, stream());
        }
        Ok(out)
    }

    /// W4A16 fused matvec for single-token decode: x[1,K] × W_int4[N,K]ᵀ → [1,N].
    ///
    /// Dequantizes 4-bit packed weights on-the-fly in CUDA registers using per-block
    /// F32 scales (Q4_0 format, group_size=32). Optional per-column AWQ scaling of x.
    /// Achieves ~4× memory-bandwidth reduction vs F16 — ideal for decode (BW-bottlenecked).
    ///
    /// `x`:          [1, K]     F16 on CUDA
    /// `nibbles`:    [N, K/2]   U8 on CUDA — packed Q4_0 nibbles (2 INT4 per byte)
    /// `scales`:     [N, K/32]  F32 on CUDA — per-block dequantization scales
    /// `awq_scales`: [K]        F32 on CUDA — per-column AWQ input scales (or None)
    /// Returns: [1, N] F16 on CUDA
    pub fn fused_matvec_w4a16(
        x:          &Tensor,
        nibbles:    &Tensor,
        scales:     &Tensor,
        awq_scales: Option<&Tensor>,
        n:          usize,
        k:          usize,
    ) -> Result<Tensor> {
        let out = Tensor::zeros((1, n), DType::F16, x.device())?;

        let x_ptr  = cuda_dev_ptr(x)? as *const u16;
        let nb_ptr = cuda_dev_ptr_u8(nibbles)? as *const u8;
        let sc_ptr = cuda_dev_ptr_f32(scales)? as *const f32;
        let aw_ptr = awq_scales
            .map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?
            .unwrap_or(std::ptr::null());
        let o_ptr  = cuda_dev_ptr(&out)? as *mut u16;

        unsafe {
            nve_matvec_w4a16(x_ptr, nb_ptr, sc_ptr, aw_ptr, o_ptr, n as i32, k as i32, stream());
        }
        Ok(out)
    }

    /// Dequantize W4 packed weights to a full F16 tensor for prefill matmul.
    ///
    /// `nibbles`: [N, K/2]  U8 on CUDA
    /// `scales`:  [N, K/32] F32 on CUDA
    /// `awq`:     [K]       F32 on CUDA (or None)
    /// Returns:   [N, K]    F16 on CUDA
    pub fn dequant_w4a16(
        nibbles:    &Tensor,
        scales:     &Tensor,
        awq_scales: Option<&Tensor>,
        n:          usize,
        k:          usize,
    ) -> Result<Tensor> {
        let out = Tensor::zeros((n, k), DType::F16, nibbles.device())?;

        let nb_ptr = cuda_dev_ptr_u8(nibbles)? as *const u8;
        let sc_ptr = cuda_dev_ptr_f32(scales)? as *const f32;
        let aw_ptr = awq_scales
            .map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?
            .unwrap_or(std::ptr::null());
        let o_ptr  = cuda_dev_ptr(&out)? as *mut u16;

        unsafe {
            nve_dequant_w4a16(nb_ptr, sc_ptr, aw_ptr, o_ptr, n as i32, k as i32, stream());
        }
        Ok(out)
    }

    /// Quantize an F16 activation vector to Q8 (int8, group_size=32).
    ///
    /// One CUDA block per group of 32 elements.  Uses absmax scaling so that
    /// the maximum magnitude maps to ±127.  The output `xq` uses U8 storage
    /// in Rust (same bytes; the C kernel writes `int8_t` values).
    ///
    /// `x`:      [K] (or [1, K]) F16 on CUDA — flattened before call
    /// Returns:  (xq: [K] U8, xq_scales: [K/32] F32)  both on CUDA
    pub fn quantize_f16_q8(x: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = x.elem_count();
        assert!(k % 32 == 0, "quantize_f16_q8: K must be multiple of 32");
        let xq     = Tensor::zeros(k,      DType::U8,  x.device())?;
        let scales = Tensor::zeros(k / 32, DType::F32, x.device())?;

        let x_flat = x.flatten_all()?;
        let x_ptr  = cuda_dev_ptr(&x_flat)?     as *const u16;
        let xq_ptr = cuda_dev_ptr_u8(&xq)?      as *mut u8;
        let sc_ptr = cuda_dev_ptr_f32(&scales)? as *mut f32;

        unsafe {
            nve_quantize_f16_q8(x_ptr, xq_ptr, sc_ptr, k as i32, stream());
        }
        Ok((xq, scales))
    }

    /// W4A8 decode matvec using dp4a: xq[K] × W_int4[N,K]ᵀ → [1,N] F16.
    ///
    /// Activations are pre-quantized by `quantize_f16_q8`.
    /// 4 warps per output row; bias-corrected for unsigned [0,15] nibbles.
    ///
    /// `xq`:        [K] U8 (int8 values) on CUDA
    /// `xq_scales`: [K/32] F32 on CUDA — per-group activation scales
    /// `nibbles`:   [N, K/2] U8 on CUDA — W4 packed nibbles (llama.cpp GPU format)
    /// `wt_scales`: [N, K/32] F32 on CUDA — per-block weight scales
    /// Returns:     [1, N] F16 on CUDA
    pub fn matvec_w4a8(
        xq:        &Tensor,
        xq_scales: &Tensor,
        nibbles:   &Tensor,
        wt_scales: &Tensor,
        n:         usize,
        k:         usize,
    ) -> Result<Tensor> {
        let out = Tensor::zeros((1, n), DType::F16, xq.device())?;

        // xq is U8 but we pass it as *const u8 (= int8_t* in C — same bits)
        let xq_ptr = cuda_dev_ptr_u8(xq)? as *const u8;
        let xs_ptr = cuda_dev_ptr_f32(xq_scales)? as *const f32;
        let nb_ptr = cuda_dev_ptr_u8(nibbles)? as *const u8;
        let ws_ptr = cuda_dev_ptr_f32(wt_scales)? as *const f32;
        let o_ptr  = cuda_dev_ptr(&out)? as *mut u16;

        unsafe {
            nve_matvec_w4a8(xq_ptr, xs_ptr, nb_ptr, ws_ptr, o_ptr, n as i32, k as i32, stream());
        }
        Ok(out)
    }

    /// Flash decode attention: fused Q·K^T + online softmax + scores·V in one kernel.
    ///
    /// Replaces: GQA expansion + cuBLAS Q@K^T + softmax ops + cuBLAS scores@V
    /// with a single warp-per-head kernel.  GQA-native (kv_head = head / groups).
    ///
    /// `q`:        [num_heads, head_dim]             F16 on CUDA (contiguous)
    /// `k_cache`:  [num_kv_heads, seq_len, head_dim] F16 on CUDA (contiguous)
    /// `v_cache`:  [num_kv_heads, seq_len, head_dim] F16 on CUDA (contiguous)
    /// Returns:    [num_heads, head_dim]              F16 on CUDA
    ///
    /// Constraints: head_dim must be divisible by 32 and ≤ 512.
    pub fn flash_decode_f16(
        q:           &Tensor,
        k_cache:     &Tensor,
        v_cache:     &Tensor,
        num_heads:   usize,
        num_kv_heads: usize,
        scale:       f32,
    ) -> Result<Tensor> {
        let head_dim = q.dim(1)?;
        let seq_len  = k_cache.dim(1)?;
        let out = Tensor::zeros((num_heads, head_dim), DType::F16, q.device())?;

        let q_ptr = cuda_dev_ptr(q)?       as *const u16;
        let k_ptr = cuda_dev_ptr(k_cache)? as *const u16;
        let v_ptr = cuda_dev_ptr(v_cache)? as *const u16;
        let o_ptr = cuda_dev_ptr(&out)?    as *mut u16;

        unsafe {
            nve_flash_decode_f16(
                q_ptr, k_ptr, v_ptr, o_ptr,
                num_heads as i32, num_kv_heads as i32,
                seq_len as i32, head_dim as i32,
                scale, stream(),
            );
        }
        Ok(out)
    }

    /// Fused Q/K/V W4A16 matvec: three projections in one kernel launch.
    ///
    /// Computes q_out = x @ Q^T, k_out = x @ K^T, v_out = x @ V^T simultaneously.
    /// Saves 2 kernel launches per layer vs separate calls.
    ///
    /// `x`:      [1, K] F16 on CUDA
    /// Returns:  ([1, Nq], [1, Nk], [1, Nv]) F16 on CUDA
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_matvec_w4a16(
        x:       &Tensor,
        q_nib:   &Tensor, q_sc: &Tensor, q_awq: Option<&Tensor>, nq: usize,
        k_nib:   &Tensor, k_sc: &Tensor, k_awq: Option<&Tensor>, nk: usize,
        v_nib:   &Tensor, v_sc: &Tensor, v_awq: Option<&Tensor>, nv: usize,
        k_dim:   usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = x.device();
        let q_out = Tensor::zeros((1, nq), DType::F16, dev)?;
        let k_out = Tensor::zeros((1, nk), DType::F16, dev)?;
        let v_out = Tensor::zeros((1, nv), DType::F16, dev)?;

        let x_ptr  = cuda_dev_ptr(x)?       as *const u16;
        let qn_ptr = cuda_dev_ptr_u8(q_nib)? as *const u8;
        let qs_ptr = cuda_dev_ptr_f32(q_sc)? as *const f32;
        let qa_ptr = q_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let kn_ptr = cuda_dev_ptr_u8(k_nib)? as *const u8;
        let ks_ptr = cuda_dev_ptr_f32(k_sc)? as *const f32;
        let ka_ptr = k_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let vn_ptr = cuda_dev_ptr_u8(v_nib)? as *const u8;
        let vs_ptr = cuda_dev_ptr_f32(v_sc)? as *const f32;
        let va_ptr = v_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let qo_ptr = cuda_dev_ptr(&q_out)? as *mut u16;
        let ko_ptr = cuda_dev_ptr(&k_out)? as *mut u16;
        let vo_ptr = cuda_dev_ptr(&v_out)? as *mut u16;

        unsafe {
            nve_qkv_matvec_w4a16(
                x_ptr,
                qn_ptr, qs_ptr, qa_ptr, nq as i32,
                kn_ptr, ks_ptr, ka_ptr, nk as i32,
                vn_ptr, vs_ptr, va_ptr, nv as i32,
                qo_ptr, ko_ptr, vo_ptr,
                k_dim as i32, stream(),
            );
        }
        Ok((q_out, k_out, v_out))
    }

    /// Fused in-place RoPE for a prefill batch.
    ///
    /// `q`:        [seq_len, num_heads, head_dim]    F16 on CUDA — modified in-place
    /// `k`:        [seq_len, num_kv_heads, head_dim] F16 on CUDA — modified in-place
    /// `rope_cos`: [max_seq, half_dim]               F16 on CUDA
    /// `rope_sin`: [max_seq, half_dim]               F16 on CUDA
    pub fn fused_rope_prefill(
        q:           &Tensor,
        k:           &Tensor,
        rope_cos:    &Tensor,
        rope_sin:    &Tensor,
        start_pos:   usize,
        num_heads:   usize,
        num_kv_heads: usize,
        head_dim:    usize,
    ) -> Result<()> {
        let seq_len = q.dim(0)?;
        let q_ptr = cuda_dev_ptr(q)? as *mut u16;
        let k_ptr = cuda_dev_ptr(k)? as *mut u16;
        let c_ptr = cuda_dev_ptr(rope_cos)? as *const u16;
        let s_ptr = cuda_dev_ptr(rope_sin)? as *const u16;

        unsafe {
            nve_rope_f16_prefill(
                q_ptr, k_ptr, c_ptr, s_ptr,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                start_pos as i32,
                stream(),
            );
        }
        Ok(())
    }

    // ── Graph-capture-safe `_into` variants ──────────────────────────────────
    //
    // These wrappers write into a CALLER-SUPPLIED pre-allocated tensor instead
    // of allocating a new one.  Required for CUDA graph capture (no cudaMalloc
    // during the capture window).

    /// RMSNorm into a pre-allocated output tensor.
    pub fn fused_rms_norm_into(x: &Tensor, weight: &Tensor, out: &Tensor, eps: f64) -> Result<()> {
        let n     = x.elem_count() as i32;
        let x_ptr = cuda_dev_ptr(x)?     as *const u16;
        let w_ptr = cuda_dev_ptr(weight)? as *const u16;
        let o_ptr = cuda_dev_ptr(out)?    as *mut u16;
        unsafe { nve_rms_norm_f16(x_ptr, w_ptr, o_ptr, n, eps as f32, stream()); }
        Ok(())
    }

    /// F16 matvec into a pre-allocated output tensor.
    pub fn fused_matvec_into(x: &Tensor, w: &Tensor, out: &Tensor, n: usize, k: usize) -> Result<()> {
        let x_ptr = cuda_dev_ptr(x)?   as *const u16;
        let w_ptr = cuda_dev_ptr(w)?   as *const u16;
        let o_ptr = cuda_dev_ptr(out)? as *mut u16;
        unsafe { nve_matvec_f16(x_ptr, w_ptr, o_ptr, n as i32, k as i32, stream()); }
        Ok(())
    }

    /// W4A16 matvec into a pre-allocated output tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_matvec_w4a16_into(
        x:          &Tensor,
        nibbles:    &Tensor,
        scales:     &Tensor,
        awq_scales: Option<&Tensor>,
        out:        &Tensor,
        n:          usize,
        k:          usize,
    ) -> Result<()> {
        let x_ptr  = cuda_dev_ptr(x)?       as *const u16;
        let nb_ptr = cuda_dev_ptr_u8(nibbles)? as *const u8;
        let sc_ptr = cuda_dev_ptr_f32(scales)? as *const f32;
        let aw_ptr = awq_scales
            .map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let o_ptr  = cuda_dev_ptr(out)? as *mut u16;
        unsafe { nve_matvec_w4a16(x_ptr, nb_ptr, sc_ptr, aw_ptr, o_ptr, n as i32, k as i32, stream()); }
        Ok(())
    }

    /// Fused Q/K/V W4A16 into pre-allocated output tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_qkv_matvec_w4a16_into(
        x:       &Tensor,
        q_nib:   &Tensor, q_sc: &Tensor, q_awq: Option<&Tensor>, nq: usize,
        k_nib:   &Tensor, k_sc: &Tensor, k_awq: Option<&Tensor>, nk: usize,
        v_nib:   &Tensor, v_sc: &Tensor, v_awq: Option<&Tensor>, nv: usize,
        k_dim:   usize,
        q_out:   &Tensor,
        k_out:   &Tensor,
        v_out:   &Tensor,
    ) -> Result<()> {
        let x_ptr  = cuda_dev_ptr(x)?         as *const u16;
        let qn_ptr = cuda_dev_ptr_u8(q_nib)?  as *const u8;
        let qs_ptr = cuda_dev_ptr_f32(q_sc)?  as *const f32;
        let qa_ptr = q_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let kn_ptr = cuda_dev_ptr_u8(k_nib)?  as *const u8;
        let ks_ptr = cuda_dev_ptr_f32(k_sc)?  as *const f32;
        let ka_ptr = k_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let vn_ptr = cuda_dev_ptr_u8(v_nib)?  as *const u8;
        let vs_ptr = cuda_dev_ptr_f32(v_sc)?  as *const f32;
        let va_ptr = v_awq.map(|a| cuda_dev_ptr_f32(a).map(|p| p as *const f32))
            .transpose()?.unwrap_or(std::ptr::null());
        let qo_ptr = cuda_dev_ptr(q_out)? as *mut u16;
        let ko_ptr = cuda_dev_ptr(k_out)? as *mut u16;
        let vo_ptr = cuda_dev_ptr(v_out)? as *mut u16;
        unsafe {
            nve_qkv_matvec_w4a16(
                x_ptr,
                qn_ptr, qs_ptr, qa_ptr, nq as i32,
                kn_ptr, ks_ptr, ka_ptr, nk as i32,
                vn_ptr, vs_ptr, va_ptr, nv as i32,
                qo_ptr, ko_ptr, vo_ptr,
                k_dim as i32, stream(),
            );
        }
        Ok(())
    }

    // ── New graph-safe operation wrappers ─────────────────────────────────────

    /// Scatter-write K+V into static pre-allocated cache at position *d_pos.
    /// k_new/v_new: [num_kv_heads, head_dim]
    /// k_cache/v_cache: [num_kv_heads, max_seq_len, head_dim]
    pub fn kv_cache_write(
        k_new:        &Tensor,
        k_cache:      &Tensor,
        v_new:        &Tensor,
        v_cache:      &Tensor,
        d_pos:        *const i32,   // device scalar from nve_graph_d_pos()
        num_kv_heads: usize,
        head_dim:     usize,
        max_seq_len:  usize,
    ) -> Result<()> {
        let kn_ptr = cuda_dev_ptr(k_new)?   as *const u16;
        let kc_ptr = cuda_dev_ptr(k_cache)? as *mut u16;
        let vn_ptr = cuda_dev_ptr(v_new)?   as *const u16;
        let vc_ptr = cuda_dev_ptr(v_cache)? as *mut u16;
        unsafe {
            nve_kv_cache_write_f16(
                kn_ptr, kc_ptr, vn_ptr, vc_ptr, d_pos,
                num_kv_heads as i32, head_dim as i32, max_seq_len as i32,
                stream(),
            );
        }
        Ok(())
    }

    /// In-place residual add: dst[i] += src[i]
    pub fn add_inplace(dst: &Tensor, src: &Tensor) -> Result<()> {
        let n     = dst.elem_count() as i32;
        let d_ptr = cuda_dev_ptr(dst)? as *mut u16;
        let s_ptr = cuda_dev_ptr(src)? as *const u16;
        unsafe { nve_add_inplace_f16(d_ptr, s_ptr, n, stream()); }
        Ok(())
    }

    /// Fused SiLU-Mul in-place: gate[i] = silu(gate[i]) * up[i]
    pub fn silu_mul_inplace(gate: &Tensor, up: &Tensor) -> Result<()> {
        let n     = gate.elem_count() as i32;
        let g_ptr = cuda_dev_ptr(gate)? as *mut u16;
        let u_ptr = cuda_dev_ptr(up)?   as *const u16;
        unsafe { nve_silu_mul_f16(g_ptr, u_ptr, n, stream()); }
        Ok(())
    }

    /// Dynamic RoPE decode using device scalar pos and full cos/sin tables.
    /// Modifies q/k in-place.
    pub fn rope_decode_dyn_inplace(
        q:           &Tensor,
        k:           &Tensor,
        cos_tab:     &Tensor,
        sin_tab:     &Tensor,
        num_heads:   usize,
        num_kv_heads: usize,
        head_dim:    usize,
        d_pos:       *const i32,
    ) -> Result<()> {
        let q_ptr = cuda_dev_ptr(q)?       as *mut u16;
        let k_ptr = cuda_dev_ptr(k)?       as *mut u16;
        let c_ptr = cuda_dev_ptr(cos_tab)? as *const u16;
        let s_ptr = cuda_dev_ptr(sin_tab)? as *const u16;
        unsafe {
            nve_rope_f16_decode_dyn(
                q_ptr, k_ptr, c_ptr, s_ptr,
                num_heads as i32, num_kv_heads as i32, head_dim as i32,
                d_pos, stream(),
            );
        }
        Ok(())
    }

    /// Dynamic flash decode into a pre-allocated output tensor.
    /// k_cache/v_cache: full static [num_kv_heads, max_seq_len, head_dim] buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decode_dyn_into(
        q:           &Tensor,
        k_cache:     &Tensor,
        v_cache:     &Tensor,
        out:         &Tensor,
        num_heads:   usize,
        num_kv_heads: usize,
        head_dim:    usize,
        scale:       f32,
        max_seq_len: usize,
        d_seq_len:   *const i32,
    ) -> Result<()> {
        let q_ptr = cuda_dev_ptr(q)?       as *const u16;
        let k_ptr = cuda_dev_ptr(k_cache)? as *const u16;
        let v_ptr = cuda_dev_ptr(v_cache)? as *const u16;
        let o_ptr = cuda_dev_ptr(out)?     as *mut u16;
        unsafe {
            nve_flash_decode_dyn(
                q_ptr, k_ptr, v_ptr, o_ptr,
                num_heads as i32, num_kv_heads as i32,
                d_seq_len, head_dim as i32, scale,
                max_seq_len as i32, stream(),
            );
        }
        Ok(())
    }

    /// Device-to-device copy on the active stream.
    pub fn device_copy(dst: &Tensor, src: &Tensor) -> Result<()> {
        let bytes = dst.elem_count() * dst.dtype().size_in_bytes();
        let d_ptr = cuda_dev_ptr(dst)? as *mut std::ffi::c_void;
        let s_ptr = cuda_dev_ptr(src)? as *const std::ffi::c_void;
        unsafe { nve_d2d_copy(d_ptr, s_ptr, bytes, stream()); }
        Ok(())
    }

    // ── CUDA graph management wrappers ────────────────────────────────────────

    /// Initialise graph infrastructure (call once before first use).
    pub fn graph_init() {
        unsafe { nve_graph_init(); }
    }

    /// Open the capture window.  All subsequent kernel launches on the capture
    /// stream (returned by `graph_capture_stream_raw()`) will be recorded.
    pub fn graph_begin_capture() {
        let cap_stream = unsafe { nve_graph_capture_stream() };
        set_graph_capture_stream(cap_stream);
        unsafe { nve_graph_begin_capture(); }
    }

    /// Close the capture window and instantiate the executable graph.
    pub fn graph_end_capture() {
        unsafe { nve_graph_end_capture(); }
        clear_graph_capture_stream();
    }

    /// Replay the captured graph: update pos/seq_len scalars then launch.
    pub fn graph_launch(pos: usize, seq_len: usize) {
        unsafe { nve_graph_launch(pos as i32, seq_len as i32, std::ptr::null_mut()); }
    }

    /// True if the graph has been successfully instantiated.
    pub fn graph_ready() -> bool {
        unsafe { nve_graph_ready() != 0 }
    }

    /// Destroy graph resources.
    pub fn graph_destroy() {
        unsafe { nve_graph_destroy(); }
    }

    /// Raw device pointer to the pos device scalar (for kernel calls during capture).
    pub fn graph_d_pos_ptr() -> *const i32 {
        unsafe { nve_graph_d_pos() }
    }

    /// Raw device pointer to the seq_len device scalar.
    pub fn graph_d_seqlen_ptr() -> *const i32 {
        unsafe { nve_graph_d_seqlen() }
    }
}
