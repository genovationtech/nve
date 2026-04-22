/*
 * NVE Custom CUDA Kernels
 * =======================
 * Each kernel has:
 *   1. __global__ device kernel
 *   2. extern "C" host launcher called from Rust FFI
 *
 * All F16 I/O, F32 accumulation for numerical stability.
 * Targets sm_75+ (T4/Turing).
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ═══════════════════════════════════════════════════════════════════════
// RMSNorm
// Replaces: to_f32 → sqr → mean_keepdim → affine(1,eps)→sqrt → broadcast_div → broadcast_mul → to_f16
// That's 7 candle ops → 1 kernel launch.
//
// Block: up to 1024 threads, one block per row.
// Shared: float[blockDim.x] for parallel sum-of-squares reduction.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_rms_norm_f16(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half*       __restrict__ out,
    int n,
    float eps
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Accumulate sum-of-squares
    float ss = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float v = __half2float(x[i]);
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)n + eps);

    // Write normalized + scaled output
    for (int i = tid; i < n; i += stride) {
        float v = __half2float(x[i]) * scale * __half2float(weight[i]);
        out[i] = __float2half(v);
    }
}

// Host launcher — called from Rust via extern "C"
extern "C" void nve_rms_norm_f16(
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t*       out,
    int n,
    float eps,
    cudaStream_t stream
) {
    int block = (n < 1024) ? n : 1024;
    size_t shared = block * sizeof(float);
    _nve_rms_norm_f16<<<1, block, shared, stream>>>(
        (const half*)x, (const half*)weight, (half*)out, n, eps
    );
}

// ═══════════════════════════════════════════════════════════════════════
// LayerNorm (with optional bias)
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_layer_norm_f16(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ bias,   // may be NULL
    half*       __restrict__ out,
    int n,
    float eps
) {
    extern __shared__ float sdata[];  // [0..blk) = sum, [blk..2blk) = sum_sq
    int tid = threadIdx.x;
    float* ssum   = sdata;
    float* ssumsq = sdata + blockDim.x;

    float lsum = 0.0f, lss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(x[i]);
        lsum += v; lss += v * v;
    }
    ssum[tid] = lsum; ssumsq[tid] = lss;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { ssum[tid] += ssum[tid+s]; ssumsq[tid] += ssumsq[tid+s]; }
        __syncthreads();
    }

    float mean  = ssum[0] / (float)n;
    float var   = ssumsq[0] / (float)n - mean * mean;
    float istd  = rsqrtf(var + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        float v = (__half2float(x[i]) - mean) * istd;
        float w = __half2float(weight[i]);
        float b = bias ? __half2float(bias[i]) : 0.0f;
        out[i]  = __float2half(v * w + b);
    }
}

extern "C" void nve_layer_norm_f16(
    const uint16_t* x,
    const uint16_t* weight,
    const uint16_t* bias,
    uint16_t*       out,
    int n,
    float eps,
    cudaStream_t stream
) {
    int block = (n < 512) ? n : 512;
    size_t shared = 2 * block * sizeof(float);
    _nve_layer_norm_f16<<<1, block, shared, stream>>>(
        (const half*)x, (const half*)weight, (const half*)bias, (half*)out, n, eps
    );
}

// ═══════════════════════════════════════════════════════════════════════
// RoPE — decode (single position, in-place)
// Replaces: narrow×2 + mul×2 + sub + add + cat per head × num_heads
// = 7 × num_heads ops → 1 kernel.
//
// Each thread rotates one (head, half_dim_element) pair for Q or K.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_rope_f16_decode(
    half* __restrict__ q,           // [num_heads, head_dim]
    half* __restrict__ k,           // [num_kv_heads, head_dim]
    const half* __restrict__ cos,   // [half_dim]
    const half* __restrict__ sin,   // [half_dim]
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int half_dim = head_dim / 2;
    int total_q  = num_heads * half_dim;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_q) {
        int head = idx / half_dim;
        int elem = idx % half_dim;
        half* ptr = q + head * head_dim;
        float c = __half2float(cos[elem]);
        float s = __half2float(sin[elem]);
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]          = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    } else {
        int k_idx = idx - total_q;
        if (k_idx >= num_kv_heads * half_dim) return;
        int head = k_idx / half_dim;
        int elem = k_idx % half_dim;
        half* ptr = k + head * head_dim;
        float c = __half2float(cos[elem]);
        float s = __half2float(sin[elem]);
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]          = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    }
}

extern "C" void nve_rope_f16_decode(
    uint16_t*       q,
    uint16_t*       k,
    const uint16_t* cos,
    const uint16_t* sin,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total = (num_heads + num_kv_heads) * half_dim;
    int block = 256;
    int grid  = (total + block - 1) / block;
    _nve_rope_f16_decode<<<grid, block, 0, stream>>>(
        (half*)q, (half*)k, (const half*)cos, (const half*)sin,
        num_heads, num_kv_heads, head_dim
    );
}

// ═══════════════════════════════════════════════════════════════════════
// RoPE — prefill (batch of positions, in-place)
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_rope_f16_prefill(
    half* __restrict__ q,
    half* __restrict__ k,
    const half* __restrict__ cos,   // [max_seq, half_dim]
    const half* __restrict__ sin,
    int seq_len, int num_heads, int num_kv_heads, int head_dim, int start_pos
) {
    int half_dim = head_dim / 2;
    int total_q  = seq_len * num_heads * half_dim;
    int total_k  = seq_len * num_kv_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_q) {
        int elem = idx % half_dim;
        int head = (idx / half_dim) % num_heads;
        int tok  = idx / (half_dim * num_heads);
        int pos  = start_pos + tok;
        float c  = __half2float(cos[pos * half_dim + elem]);
        float s  = __half2float(sin[pos * half_dim + elem]);
        half* ptr = q + (tok * num_heads + head) * head_dim;
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]          = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    } else if (idx < total_q + total_k) {
        int ki   = idx - total_q;
        int elem = ki % half_dim;
        int head = (ki / half_dim) % num_kv_heads;
        int tok  = ki / (half_dim * num_kv_heads);
        int pos  = start_pos + tok;
        float c  = __half2float(cos[pos * half_dim + elem]);
        float s  = __half2float(sin[pos * half_dim + elem]);
        half* ptr = k + (tok * num_kv_heads + head) * head_dim;
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]          = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    }
}

extern "C" void nve_rope_f16_prefill(
    uint16_t*       q,
    uint16_t*       k,
    const uint16_t* cos,
    const uint16_t* sin,
    int seq_len, int num_heads, int num_kv_heads, int head_dim, int start_pos,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total = seq_len * (num_heads + num_kv_heads) * half_dim;
    int block = 256;
    int grid  = (total + block - 1) / block;
    _nve_rope_f16_prefill<<<grid, block, 0, stream>>>(
        (half*)q, (half*)k, (const half*)cos, (const half*)sin,
        seq_len, num_heads, num_kv_heads, head_dim, start_pos
    );
}

// ═══════════════════════════════════════════════════════════════════════
// F16 Matvec: x[1,K] × W[N,K]^T → out[1,N]
// Optimised for single-token decode (batch=1).
// Uses half2 vectorised loads and warp shuffle reduction.
// Each warp computes one output element.
// ═══════════════════════════════════════════════════════════════════════

#define WARP 32

__global__ void _nve_matvec_f16(
    const half* __restrict__ x,
    const half* __restrict__ W,
    half*       __restrict__ out,
    int N, int K
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    const half2* x2 = (const half2*)x;
    const half2* w2 = (const half2*)(W + (long long)row * K);
    int K2 = K / 2;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < K2; i += WARP) {
        half2 xi = x2[i];
        half2 wi = w2[i];
        acc += (float)xi.x * (float)wi.x + (float)xi.y * (float)wi.y;
    }

    // Warp reduction
    for (int mask = WARP >> 1; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    if (threadIdx.x == 0) out[row] = __float2half(acc);
}

extern "C" void nve_matvec_f16(
    const uint16_t* x,
    const uint16_t* W,
    uint16_t*       out,
    int N, int K,
    cudaStream_t stream
) {
    // blockDim = (WARP, rows_per_block); each warp handles one row
    int rows_per_block = 8;
    dim3 block(WARP, rows_per_block);
    dim3 grid((N + rows_per_block - 1) / rows_per_block);
    _nve_matvec_f16<<<grid, block, 0, stream>>>(
        (const half*)x, (const half*)W, (half*)out, N, K
    );
}

// ═══════════════════════════════════════════════════════════════════════
// W4A16 Matvec: x[1,K] F16 × W_q4[N,K] INT4 → out[1,N] F16
//
// Weight format (Q4_0, sequential nibble pairs):
//   nibbles[row, blk*16 + pos/2]: byte = low nibble (pos even) | high nibble (pos odd)
//   dequant: val = (nibble - 8) * scales[row, blk]
//   AWQ: optional per-column f32 scale applied to x
//
// Design: one warp per output row; each thread covers one column per block.
// Thread t covers columns: t, t+32, t+64, ..., t+(num_blocks-1)*32.
// x reads land in L2 (T4: 4MB L2, x is ≤29KB for 8B model — always fits).
// Weight nibbles: __ldg routes through read-only L1 texture cache.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_matvec_w4a16(
    const half*    __restrict__ x,        // [K] F16
    const uint8_t* __restrict__ nibbles,  // [N, K/2] packed nibbles
    const float*   __restrict__ scales,   // [N, K/32] per-block F32 scales
    const float*   __restrict__ awq,      // [K] F32 AWQ input scales, or NULL
    half*          __restrict__ out,      // [N] F16
    int N, int K
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int tid        = threadIdx.x;   // lane, 0..31
    int num_blocks = K / 32;

    const uint8_t* row_nib = nibbles + (long long)row * (K / 2);
    const float*   row_sc  = scales  + (long long)row * num_blocks;

    float acc = 0.0f;

    for (int blk = 0; blk < num_blocks; blk++) {
        float scale = row_sc[blk];
        int   col   = blk * 32 + tid;

        // x: load through L2 (always cached — x is ≤29KB, L2 is 4MB on T4)
        float xi = __half2float(x[col]);
        if (awq) xi *= awq[col];

        // GPU (llama.cpp) Q4_0 format: qs[i] = {elem i, elem i+16}
        // tid 0..15 → first half (low nibble of qs[tid])
        // tid 16..31 → second half (high nibble of qs[tid-16])
        uint8_t bv  = __ldg(&row_nib[blk * 16 + (tid & 15)]);
        int     nib = (tid < 16) ? (bv & 0xF) : (bv >> 4);

        acc += xi * (float)(nib - 8) * scale;
    }

    // Warp-shuffle reduction
    for (int mask = WARP >> 1; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    if (tid == 0) out[row] = __float2half(acc);
}

extern "C" void nve_matvec_w4a16(
    const uint16_t* x,
    const uint8_t*  nibbles,
    const float*    scales,
    const float*    awq,      // may be NULL
    uint16_t*       out,
    int N, int K,
    cudaStream_t stream
) {
    int rows_per_block = 8;
    dim3 block(WARP, rows_per_block);
    dim3 grid((N + rows_per_block - 1) / rows_per_block);
    _nve_matvec_w4a16<<<grid, block, 0, stream>>>(
        (const half*)x, nibbles, scales, awq, (half*)out, N, K
    );
}

// ═══════════════════════════════════════════════════════════════════════
// W4A16 Dequantize: W_q4[N,K] INT4 → W_f16[N,K] F16 (for prefill matmul)
//
// Each CUDA thread dequantizes one element.
// Used by the prefill path to convert packed weights to F16 on-the-fly.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_dequant_w4a16(
    const uint8_t* __restrict__ nibbles,  // [N, K/2]
    const float*   __restrict__ scales,   // [N, K/32]
    const float*   __restrict__ awq,      // [K] F32 or NULL
    half*          __restrict__ out,      // [N, K]
    int N, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= N || col >= K) return;

    int   blk        = col / 32;
    int   pos        = col % 32;  // 0..31
    int   num_blocks = K / 32;
    // GPU format: qs[i] = {elem i, elem i+16} — pos 0..15 in low nibble, 16..31 in high
    uint8_t bv       = nibbles[row * (K / 2) + blk * 16 + (pos & 15)];
    int   nibble     = (pos < 16) ? (bv & 0xF) : (bv >> 4);
    float scale      = scales[row * num_blocks + blk];
    float val        = (float)(nibble - 8) * scale;
    if (awq) val    *= awq[col];
    out[row * K + col] = __float2half(val);
}

extern "C" void nve_dequant_w4a16(
    const uint8_t*  nibbles,
    const float*    scales,
    const float*    awq,
    uint16_t*       out,
    int N, int K,
    cudaStream_t stream
) {
    dim3 block(256, 1);
    dim3 grid((K + 255) / 256, N);
    _nve_dequant_w4a16<<<grid, block, 0, stream>>>(
        nibbles, scales, awq, (half*)out, N, K
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Flash Decode Attention
// ======================
// Fused single-query attention for decode (batch=1).
// Replaces: Q@K^T + softmax + scores@V (4-6 kernel launches + GQA expansion)
//
// GQA-native: kv_head = head / groups — no expansion tensor materialised.
//
// Block structure: (WARP, FD_WARPS) — FD_WARPS warps per head block.
//   Grid: (num_heads,) — one block per head.
//   Total warps on T4: num_heads × FD_WARPS (e.g. 32 × 8 = 256 warps).
//   T4 can schedule 32 warps/SM × 40 SMs = 1280 warps simultaneously.
//   256 warps → ~6 warps/SM → memory latency hidden by warp interleaving.
//
// Each warp processes a disjoint chunk of KV positions with online softmax.
// Partial (max, sum, out) stored in shared memory; warp 0 merges.
//
// Shared memory: FD_WARPS × (2 + head_dim) floats.
//   FD_WARPS=8, head_dim=128 → 8×130×4 = 4160 B  (well under 48KB limit)
//
// Constraints: head_dim divisible by WARP, head_dim ≤ MAX_HD_PER_THREAD×WARP.
// MAX_HD_PER_THREAD=16 → head_dim ≤ 512 (covers all standard Llama architectures).
// ═══════════════════════════════════════════════════════════════════════

#define MAX_HD_PER_THREAD 16
#define FD_WARPS          8   // warps per flash-decode block (tune for occupancy)

__global__ void _nve_flash_decode_f16(
    const half* __restrict__ q,        // [num_heads, head_dim]
    const half* __restrict__ k_cache,  // [num_kv_heads, seq_len, head_dim]
    const half* __restrict__ v_cache,  // [num_kv_heads, seq_len, head_dim]
    half*       __restrict__ out,      // [num_heads, head_dim]
    int num_kv_heads, int seq_len, int head_dim,
    float scale, int groups            // groups = num_heads / num_kv_heads
) {
    // Shared memory: FD_WARPS × (2 + head_dim) floats
    // Layout per warp: [m, s, acc_0 … acc_{head_dim-1}]
    extern __shared__ float smem[];

    int h     = blockIdx.x;      // head index (0..num_heads-1)
    int wid   = threadIdx.y;     // warp index within block (0..FD_WARPS-1)
    int tid   = threadIdx.x;     // lane (0..WARP-1)
    int kv_h  = h / groups;      // KV cache head (GQA-aware)
    int epw   = head_dim / WARP; // output elements owned by this lane

    const half* q_h = q       + h    * head_dim;
    const half* k_h = k_cache + (long long)kv_h * seq_len * head_dim;
    const half* v_h = v_cache + (long long)kv_h * seq_len * head_dim;

    // Load Q into registers (re-used across all positions in this warp's chunk).
    float qreg[MAX_HD_PER_THREAD];
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        qreg[i] = __half2float(q_h[tid + i * WARP]);

    // This warp's position range.
    int chunk = (seq_len + FD_WARPS - 1) / FD_WARPS;
    int t_start = wid * chunk;
    int t_end   = min(t_start + chunk, seq_len);

    // Per-warp online-softmax state + output accumulators.
    float acc[MAX_HD_PER_THREAD] = {0.0f};
    float m = -1e20f, s = 0.0f;

    for (int t = t_start; t < t_end; t++) {
        const half* k_t = k_h + t * head_dim;
        const half* v_t = v_h + t * head_dim;

        // ── dot product Q · K[t] ─────────────────────────────────────────
        float dot = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            dot += qreg[i] * __half2float(k_t[tid + i * WARP]);
        #pragma unroll
        for (int mask = WARP >> 1; mask > 0; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        dot *= scale;   // broadcast: all lanes see the same value

        // ── online softmax update ────────────────────────────────────────
        float new_m = fmaxf(m, dot);
        float corr  = expf(m - new_m);
        float e     = expf(dot - new_m);
        #pragma unroll 4
        for (int i = 0; i < epw; i++) acc[i] *= corr;
        s = s * corr + e;
        m = new_m;

        // ── accumulate V[t] ──────────────────────────────────────────────
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            acc[i] += e * __half2float(v_t[tid + i * WARP]);
    }

    // ── Store this warp's partial result in shared memory ─────────────────
    // smem layout: wid × (2 + head_dim): [m, s, acc[0..head_dim-1]]
    float* sw = smem + wid * (2 + head_dim);
    if (tid == 0) { sw[0] = m; sw[1] = s; }
    // Store acc: thread tid stores elements tid, tid+WARP, … (no bank conflicts)
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        sw[2 + tid + i * WARP] = acc[i];

    __syncthreads();

    // ── Warp 0 merges all partial results ─────────────────────────────────
    if (wid != 0) return;

    // Read own partial (already stored, read back cleanly post-sync).
    float merged_m = smem[0];          // broadcast read — no bank conflict
    float merged_s = smem[1];
    float merged_acc[MAX_HD_PER_THREAD];
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        merged_acc[i] = smem[2 + tid + i * WARP];

    // Merge warps 1..FD_WARPS-1 into warp 0's running state.
    #pragma unroll
    for (int w = 1; w < FD_WARPS; w++) {
        float* sw_w = smem + w * (2 + head_dim);
        float  w_m  = sw_w[0];          // broadcast
        float  w_s  = sw_w[1];          // broadcast
        float new_m = fmaxf(merged_m, w_m);
        float c0    = expf(merged_m - new_m);
        float c1    = expf(w_m - new_m);
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            merged_acc[i] = merged_acc[i] * c0 + sw_w[2 + tid + i * WARP] * c1;
        merged_s = merged_s * c0 + w_s * c1;
        merged_m = new_m;
    }

    // Normalise and write output.
    float inv_s = (merged_s > 0.0f) ? (1.0f / merged_s) : 0.0f;
    half* o_h = out + h * head_dim;
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        o_h[tid + i * WARP] = __float2half(merged_acc[i] * inv_s);
}

extern "C" void nve_flash_decode_f16(
    const uint16_t* q,        // [num_heads, head_dim]
    const uint16_t* k_cache,  // [num_kv_heads, seq_len, head_dim]
    const uint16_t* v_cache,  // [num_kv_heads, seq_len, head_dim]
    uint16_t*       out,      // [num_heads, head_dim]
    int num_heads, int num_kv_heads, int seq_len, int head_dim, float scale,
    cudaStream_t stream
) {
    if (head_dim % WARP != 0 || head_dim / WARP > MAX_HD_PER_THREAD || seq_len == 0) return;
    int groups = num_heads / num_kv_heads;
    int smem_bytes = FD_WARPS * (2 + head_dim) * sizeof(float);
    dim3 block(WARP, FD_WARPS);
    dim3 grid(num_heads);
    _nve_flash_decode_f16<<<grid, block, smem_bytes, stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache, (half*)out,
        num_kv_heads, seq_len, head_dim, scale, groups
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Fused Q/K/V W4A16 Matvec
// =========================
// Computes all three projections (Q, K, V) in a single kernel launch.
// Saves 2 kernel launches per transformer layer vs separate Q, K, V calls.
//
// Layout: 1D grid covering all Nq + Nk + Nv output rows consecutively.
// Rows [0, Nq)       → Q output
// Rows [Nq, Nq+Nk)   → K output
// Rows [Nq+Nk, Nq+Nk+Nv) → V output
//
// Each warp handles one output row (same as the scalar W4A16 kernel).
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_qkv_matvec_w4a16(
    const half*    __restrict__ x,      // [K] F16 input
    // Q weight buffers
    const uint8_t* __restrict__ q_nib,  // [Nq, K/2] packed nibbles
    const float*   __restrict__ q_sc,   // [Nq, K/32] scales
    const float*   __restrict__ q_awq,  // [K] AWQ or NULL
    int Nq,
    // K weight buffers
    const uint8_t* __restrict__ k_nib,
    const float*   __restrict__ k_sc,
    const float*   __restrict__ k_awq,
    int Nk,
    // V weight buffers
    const uint8_t* __restrict__ v_nib,
    const float*   __restrict__ v_sc,
    const float*   __restrict__ v_awq,
    int Nv,
    // Outputs
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int K
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;  // lane, 0..WARP-1

    // Map combined row index to the appropriate weight matrix.
    const uint8_t* nib_ptr;
    const float*   sc_ptr;
    const float*   awq_ptr;
    half*          out_ptr;
    int            local_row;

    int Nqk = Nq + Nk;
    if (row < Nq) {
        nib_ptr = q_nib; sc_ptr = q_sc; awq_ptr = q_awq;
        out_ptr = q_out; local_row = row;
    } else if (row < Nqk) {
        nib_ptr = k_nib; sc_ptr = k_sc; awq_ptr = k_awq;
        out_ptr = k_out; local_row = row - Nq;
    } else {
        nib_ptr = v_nib; sc_ptr = v_sc; awq_ptr = v_awq;
        out_ptr = v_out; local_row = row - Nqk;
    }

    if (row >= Nqk + Nv) return;

    int num_blks = K / 32;
    const uint8_t* row_nib = nib_ptr + (long long)local_row * (K / 2);
    const float*   row_sc  = sc_ptr  + (long long)local_row * num_blks;

    float acc = 0.0f;

    for (int blk = 0; blk < num_blks; blk++) {
        float scale = row_sc[blk];
        int   col   = blk * 32 + tid;

        float xi = __half2float(x[col]);
        if (awq_ptr) xi *= awq_ptr[col];

        // GPU (llama.cpp) Q4_0 format: qs[i] = {elem i, elem i+16}
        uint8_t bv      = __ldg(&row_nib[blk * 16 + (tid & 15)]);
        int     nib_val = (tid < 16) ? (bv & 0xF) : (bv >> 4);

        acc += xi * (float)(nib_val - 8) * scale;
    }

    for (int mask = WARP >> 1; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    if (tid == 0) out_ptr[local_row] = __float2half(acc);
}

extern "C" void nve_qkv_matvec_w4a16(
    const uint16_t* x,
    const uint8_t* q_nib, const float* q_sc, const float* q_awq, int Nq,
    const uint8_t* k_nib, const float* k_sc, const float* k_awq, int Nk,
    const uint8_t* v_nib, const float* v_sc, const float* v_awq, int Nv,
    uint16_t* q_out, uint16_t* k_out, uint16_t* v_out,
    int K,
    cudaStream_t stream
) {
    int total_N = Nq + Nk + Nv;
    int rows_per_block = 8;
    dim3 block(WARP, rows_per_block);
    dim3 grid((total_N + rows_per_block - 1) / rows_per_block);
    _nve_qkv_matvec_w4a16<<<grid, block, 0, stream>>>(
        (const half*)x,
        q_nib, q_sc, q_awq, Nq,
        k_nib, k_sc, k_awq, Nk,
        v_nib, v_sc, v_awq, Nv,
        (half*)q_out, (half*)k_out, (half*)v_out, K
    );
}

// ═══════════════════════════════════════════════════════════════════════
// F16 → Q8 Activation Quantization
// =================================
// Converts an F16 activation vector x[K] to int8 xq[K] with per-group-32
// F32 scales.  Run once before W4A8 matmuls; result reused for Q/K/V.
//
// Grid: (K/32, 1)  Block: (32, 1)
// Each block: one 32-element group. Shared mem: 32 floats for max-abs.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_quantize_f16_q8(
    const half*  __restrict__ x,       // [K] F16 input
    int8_t*      __restrict__ xq,      // [K] int8 output
    float*       __restrict__ scales,  // [K/32] float32 output
    int K
) {
    int group = blockIdx.x;
    int tid   = threadIdx.x;  // 0..31
    int idx   = group * 32 + tid;

    float val = (idx < K) ? __half2float(x[idx]) : 0.0f;

    // Parallel max-abs reduction in shared memory
    __shared__ float smax[32];
    smax[tid] = fabsf(val);
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }

    float amax  = smax[0];
    float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;

    if (tid == 0) scales[group] = scale;

    // Quantize and clamp to [-127, 127]  (avoid -128 for clean bias correction)
    float inv_scale = 1.0f / scale;
    int   qi = __float2int_rn(val * inv_scale);
    if (qi < -127) qi = -127;
    if (qi >  127) qi =  127;
    if (idx < K) xq[idx] = (int8_t)qi;
}

extern "C" void nve_quantize_f16_q8(
    const uint16_t* x,
    int8_t*         xq,
    float*          scales,
    int K,
    cudaStream_t stream
) {
    dim3 grid(K / 32);
    dim3 block(32);
    _nve_quantize_f16_q8<<<grid, block, 0, stream>>>((const half*)x, xq, scales, K);
}

// ═══════════════════════════════════════════════════════════════════════
// W4A8 Matvec with dp4a  (llama.cpp-style)
// =========================================
// Multiplies F16-x (pre-quantized to Q8) by W4 matrix using __dp4a.
//
// Weight format: GPU (llama.cpp) Q4_0
//   qs[i] = {element i in low nibble, element i+16 in high nibble}
// Activation format: sequential int8, pre-quantized by nve_quantize_f16_q8.
//
// Design (matches llama.cpp mmvq for Q4_0 on Turing / MMVQ_PARAMETERS_GENERIC):
//   nwarps = 4 warps per output row
//   VDR    = 2  (each thread handles 2 consecutive int32s per outer iteration)
//   QI4_0  = 4  (4 int32s = 16 bytes per Q4_0 block)
//   blocks_per_iter = VDR * nwarps * 32 / QI4_0 = 2*4*32/4 = 64
//   → for K=2048: 64 blocks total, outer loop runs exactly once per thread
//   → for K=8192: loop runs 4 times
//
// Grid: (N, 1)  Block: (32, 4) = 128 threads, 1 output row per block.
// Shared: float tmp[3][32] for warp reduction (384 bytes).
//
// Bias correction: raw nibbles [0,15] used in dp4a; the +8 offset is
// corrected per block: sumi_corrected = sumi - 8 * sum_q, where sum_q is
// computed using dp4a with a constant all-ones vector.
// ═══════════════════════════════════════════════════════════════════════

#define W4A8_NWARPS 4
#define W4A8_VDR    2
#define W4A8_QI4    4   // int32s per Q4_0 block (16 bytes / 4)

// Load a 32-bit integer from a 2-byte aligned pointer at int32 index i32.
static __device__ __forceinline__ int get_int_b2(const void* __restrict__ x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int v  = (int)x16[2*i32 + 0];
    v     |= (int)x16[2*i32 + 1] << 16;
    return v;
}

__global__ void _nve_matvec_w4a8(
    const int8_t*  __restrict__ xq,        // [K] Q8 activations
    const float*   __restrict__ xq_scales, // [K/32] activation block scales
    const uint8_t* __restrict__ nibbles,   // [N, K/2] W4 nibbles (llama.cpp format)
    const float*   __restrict__ wt_scales, // [N, K/32] weight block scales
    half*          __restrict__ out,       // [N] F16 output
    int N, int K
) {
    const int row       = blockIdx.x;
    const int tid       = 32 * threadIdx.y + threadIdx.x;  // 0..127
    const int num_blks  = K / 32;
    const int kbx_start = tid / (W4A8_QI4 / W4A8_VDR);      // = tid/2
    const int iqs       = W4A8_VDR * (tid % (W4A8_QI4 / W4A8_VDR)); // 0 or 2
    const int blks_iter = W4A8_VDR * W4A8_NWARPS * 32 / W4A8_QI4;   // = 64

    if (row >= N) return;

    const uint8_t* row_nib = nibbles + (long long)row * (K / 2);

    float acc = 0.0f;

    for (int kbx = kbx_start; kbx < num_blks; kbx += blks_iter) {
        // Load VDR=2 int32s of Q4_0 nibbles (8 bytes = 16 nibbles)
        int v0 = get_int_b2(row_nib + kbx * 16, iqs + 0);
        int v1 = get_int_b2(row_nib + kbx * 16, iqs + 1);

        // Load corresponding Q8 activations.
        // Nibble int32 #j = {elem j, elem j+16, elem j+1, elem j+17, ...}
        // (4 sequential first-half elems in lower nibbles, 4 second-half in upper nibbles)
        // Nibble int32 at index (iqs+j) corresponds to activations:
        //   first-half  acts[(iqs+j)*4 .. (iqs+j)*4+3]  — byte offset (iqs+j)*4
        //   second-half acts[(iqs+j)*4+16 .. (iqs+j)*4+19] — byte offset (iqs+j)*4+16
        const int8_t* xq_blk = xq + kbx * 32;
        int u0 = *(const int*)(xq_blk + (iqs + 0) * 4);           // first-half acts for v0
        int u1 = *(const int*)(xq_blk + (iqs + 0) * 4 + 16);      // second-half acts for v0
        int u2 = *(const int*)(xq_blk + (iqs + 1) * 4);           // first-half acts for v1
        int u3 = *(const int*)(xq_blk + (iqs + 1) * 4 + 16);      // second-half acts for v1

        // dp4a dot product using raw nibbles [0,15]
        const int vi0 = (v0 >> 0) & 0x0F0F0F0F;  // lower nibbles of v0
        const int vi1 = (v0 >> 4) & 0x0F0F0F0F;  // upper nibbles of v0
        const int vi2 = (v1 >> 0) & 0x0F0F0F0F;
        const int vi3 = (v1 >> 4) & 0x0F0F0F0F;

        int sumi = 0;
        sumi = __dp4a(vi0, u0, sumi);
        sumi = __dp4a(vi1, u1, sumi);
        sumi = __dp4a(vi2, u2, sumi);
        sumi = __dp4a(vi3, u3, sumi);

        // Bias correction: raw nibbles are [0,15], real weights are [nibble-8].
        // sumi_corrected = sumi - 8 * sum_q
        // sum_q computed via dp4a with constant 1-vector
        int sum_q = 0;
        sum_q = __dp4a(u0, 0x01010101, sum_q);
        sum_q = __dp4a(u1, 0x01010101, sum_q);
        sum_q = __dp4a(u2, 0x01010101, sum_q);
        sum_q = __dp4a(u3, 0x01010101, sum_q);

        float block_result = wt_scales[row * num_blks + kbx] *
                             xq_scales[kbx] *
                             (float)(sumi - 8 * sum_q);
        acc += block_result;
    }

    // Multi-warp reduction: warps 1-3 store to shared, warp 0 reduces.
    __shared__ float tmp_shared[W4A8_NWARPS - 1][32];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = acc;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int w = 0; w < W4A8_NWARPS - 1; ++w) {
            acc += tmp_shared[w][threadIdx.x];
        }
        // Warp-shuffle reduction to scalar
        for (int mask = 16; mask > 0; mask >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, mask);

        if (threadIdx.x == 0) out[row] = __float2half(acc);
    }
}

extern "C" void nve_matvec_w4a8(
    const int8_t*   xq,
    const float*    xq_scales,
    const uint8_t*  nibbles,
    const float*    wt_scales,
    uint16_t*       out,
    int N, int K,
    cudaStream_t stream
) {
    dim3 block(32, W4A8_NWARPS);
    dim3 grid(N);
    _nve_matvec_w4a8<<<grid, block, 0, stream>>>(
        xq, xq_scales, nibbles, wt_scales, (half*)out, N, K
    );
}

// ═══════════════════════════════════════════════════════════════════════
// KV Cache Scatter Write (graph-capture-safe)
// ===========================================
// Writes a single new K/V token into a pre-allocated static KV cache at
// position *d_pos (device scalar).  Eliminates CT::cat which allocates
// a new tensor (+ cudaFree of old) on every decode step.
//
// k_new  [num_kv_heads, head_dim]                F16 — new token K
// k_cache[num_kv_heads, max_seq_len, head_dim]   F16 — static cache
// Writes: k_cache[kv_h, *d_pos, :] = k_new[kv_h, :]  (same for V)
// One block per kv_head; blockDim.x threads cover head_dim elements.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_kv_cache_write_f16(
    const half* __restrict__ k_new,
    half*       __restrict__ k_cache,
    const half* __restrict__ v_new,
    half*       __restrict__ v_cache,
    const int*  __restrict__ d_pos,
    int head_dim,
    int max_seq_len
) {
    int kv_h = blockIdx.x;
    int tid  = threadIdx.x;
    int pos  = *d_pos;

    long long cache_base = (long long)kv_h * max_seq_len * head_dim + (long long)pos * head_dim;
    long long new_base   = (long long)kv_h * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        k_cache[cache_base + i] = k_new[new_base + i];
        v_cache[cache_base + i] = v_new[new_base + i];
    }
}

extern "C" void nve_kv_cache_write_f16(
    const uint16_t* k_new,   uint16_t* k_cache,
    const uint16_t* v_new,   uint16_t* v_cache,
    const int*      d_pos,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    cudaStream_t stream
) {
    int block = (head_dim < 256) ? head_dim : 256;
    _nve_kv_cache_write_f16<<<num_kv_heads, block, 0, stream>>>(
        (const half*)k_new, (half*)k_cache,
        (const half*)v_new, (half*)v_cache,
        d_pos, head_dim, max_seq_len
    );
}

// ═══════════════════════════════════════════════════════════════════════
// In-place Residual Add: dst[i] += src[i]
// Fuses the two hidden-state accumulations per transformer layer
// (attn output residual + FFN output residual) into single kernels.
// Saves 2 × num_layers tensor allocations per decode step.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_add_inplace_f16(
    half* __restrict__ dst, const half* __restrict__ src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

extern "C" void nve_add_inplace_f16(
    uint16_t* dst, const uint16_t* src, int n, cudaStream_t stream
) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    _nve_add_inplace_f16<<<grid, block, 0, stream>>>((half*)dst, (const half*)src, n);
}

// ═══════════════════════════════════════════════════════════════════════
// Fused SiLU-Mul: gate[i] = silu(gate[i]) * up[i]
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
// Overwrites gate in-place; eliminates 3 separate candle ops
// (.silu() + .mul() + allocation) per FFN layer.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_silu_mul_f16(
    half* __restrict__ gate, const half* __restrict__ up, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        gate[i] = __float2half(g / (1.0f + expf(-g)) * u);
    }
}

extern "C" void nve_silu_mul_f16(
    uint16_t* gate, const uint16_t* up, int n, cudaStream_t stream
) {
    int block = 256;
    int grid  = (n + block - 1) / block;
    _nve_silu_mul_f16<<<grid, block, 0, stream>>>((half*)gate, (const half*)up, n);
}

// ═══════════════════════════════════════════════════════════════════════
// Dynamic RoPE Decode (graph-capture-safe)
// ========================================
// Like nve_rope_f16_decode but takes the FULL cos/sin table
// [max_seq, half_dim] plus a device scalar *d_pos.  The kernel indexes
// into the table at runtime, so the captured graph replays correctly
// for any position without graph re-capture.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_rope_f16_decode_dyn(
    half*       __restrict__ q,        // [num_heads, head_dim]
    half*       __restrict__ k,        // [num_kv_heads, head_dim]
    const half* __restrict__ cos_tab,  // [max_seq, half_dim]
    const half* __restrict__ sin_tab,  // [max_seq, half_dim]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ d_pos
) {
    int half_dim = head_dim / 2;
    int pos      = *d_pos;
    const half* cos = cos_tab + (long long)pos * half_dim;
    const half* sin = sin_tab + (long long)pos * half_dim;

    int total_q = num_heads * half_dim;
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_q) {
        int head = idx / half_dim;
        int elem = idx % half_dim;
        half* ptr = q + head * head_dim;
        float c  = __half2float(cos[elem]);
        float s  = __half2float(sin[elem]);
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]            = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    } else {
        int ki   = idx - total_q;
        if (ki >= num_kv_heads * half_dim) return;
        int head = ki / half_dim;
        int elem = ki % half_dim;
        half* ptr = k + head * head_dim;
        float c  = __half2float(cos[elem]);
        float s  = __half2float(sin[elem]);
        float x0 = __half2float(ptr[elem]);
        float x1 = __half2float(ptr[elem + half_dim]);
        ptr[elem]            = __float2half(x0 * c - x1 * s);
        ptr[elem + half_dim] = __float2half(x0 * s + x1 * c);
    }
}

extern "C" void nve_rope_f16_decode_dyn(
    uint16_t* q, uint16_t* k,
    const uint16_t* cos_tab, const uint16_t* sin_tab,
    int num_heads, int num_kv_heads, int head_dim,
    const int* d_pos,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total    = (num_heads + num_kv_heads) * half_dim;
    int block    = 256;
    int grid     = (total + block - 1) / block;
    _nve_rope_f16_decode_dyn<<<grid, block, 0, stream>>>(
        (half*)q, (half*)k,
        (const half*)cos_tab, (const half*)sin_tab,
        num_heads, num_kv_heads, head_dim, d_pos
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Dynamic Flash Decode (graph-capture-safe)
// ==========================================
// Like _nve_flash_decode_f16 but reads seq_len from a device scalar
// *d_seq_len.  k_cache/v_cache point to the FULL static pre-allocated
// buffer; the kernel only reads the first *d_seq_len positions.
// ═══════════════════════════════════════════════════════════════════════

__global__ void _nve_flash_decode_dyn(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,  // [num_kv_heads, max_seq_len, head_dim]
    const half* __restrict__ v_cache,  // [num_kv_heads, max_seq_len, head_dim]
    half*       __restrict__ out,
    int num_kv_heads,
    const int* __restrict__ d_seq_len, // device scalar — valid tokens in cache
    int head_dim,
    float scale,
    int groups,
    int max_seq_len                    // stride for cache indexing
) {
    int seq_len = *d_seq_len;
    if (seq_len <= 0) return;

    extern __shared__ float smem[];

    int h    = blockIdx.x;
    int wid  = threadIdx.y;
    int tid  = threadIdx.x;
    int kv_h = h / groups;
    int epw  = head_dim / WARP;

    const half* q_h = q       + (long long)h    * head_dim;
    const half* k_h = k_cache + (long long)kv_h * max_seq_len * head_dim;
    const half* v_h = v_cache + (long long)kv_h * max_seq_len * head_dim;

    float qreg[MAX_HD_PER_THREAD];
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        qreg[i] = __half2float(q_h[tid + i * WARP]);

    int chunk   = (seq_len + FD_WARPS - 1) / FD_WARPS;
    int t_start = wid * chunk;
    int t_end   = min(t_start + chunk, seq_len);

    float acc[MAX_HD_PER_THREAD] = {0.0f};
    float m = -1e20f, s = 0.0f;

    for (int t = t_start; t < t_end; t++) {
        const half* k_t = k_h + (long long)t * head_dim;
        const half* v_t = v_h + (long long)t * head_dim;

        float dot = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            dot += qreg[i] * __half2float(k_t[tid + i * WARP]);
        #pragma unroll
        for (int mask = WARP >> 1; mask > 0; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        dot *= scale;

        float new_m = fmaxf(m, dot);
        float corr  = expf(m - new_m);
        float e     = expf(dot - new_m);
        #pragma unroll 4
        for (int i = 0; i < epw; i++) acc[i] *= corr;
        s = s * corr + e;
        m = new_m;
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            acc[i] += e * __half2float(v_t[tid + i * WARP]);
    }

    float* sw = smem + wid * (2 + head_dim);
    if (tid == 0) { sw[0] = m; sw[1] = s; }
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        sw[2 + tid + i * WARP] = acc[i];

    __syncthreads();
    if (wid != 0) return;

    float merged_m = smem[0];
    float merged_s = smem[1];
    float merged_acc[MAX_HD_PER_THREAD];
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        merged_acc[i] = smem[2 + tid + i * WARP];

    #pragma unroll
    for (int w = 1; w < FD_WARPS; w++) {
        float* sw_w = smem + w * (2 + head_dim);
        float w_m   = sw_w[0];
        float w_s   = sw_w[1];
        float new_m = fmaxf(merged_m, w_m);
        float c0    = expf(merged_m - new_m);
        float c1    = expf(w_m - new_m);
        #pragma unroll 4
        for (int i = 0; i < epw; i++)
            merged_acc[i] = merged_acc[i] * c0 + sw_w[2 + tid + i * WARP] * c1;
        merged_s = merged_s * c0 + w_s * c1;
        merged_m = new_m;
    }

    float inv_s = (merged_s > 0.0f) ? (1.0f / merged_s) : 0.0f;
    half* o_h = out + (long long)h * head_dim;
    #pragma unroll 4
    for (int i = 0; i < epw; i++)
        o_h[tid + i * WARP] = __float2half(merged_acc[i] * inv_s);
}

extern "C" void nve_flash_decode_dyn(
    const uint16_t* q,
    const uint16_t* k_cache,
    const uint16_t* v_cache,
    uint16_t*       out,
    int num_heads, int num_kv_heads,
    const int*      d_seq_len,
    int head_dim, float scale,
    int max_seq_len,
    cudaStream_t stream
) {
    if (head_dim % WARP != 0 || head_dim / WARP > MAX_HD_PER_THREAD) return;
    int groups     = num_heads / num_kv_heads;
    int smem_bytes = FD_WARPS * (2 + head_dim) * sizeof(float);
    dim3 block(WARP, FD_WARPS);
    dim3 grid(num_heads);
    _nve_flash_decode_dyn<<<grid, block, smem_bytes, stream>>>(
        (const half*)q, (const half*)k_cache, (const half*)v_cache, (half*)out,
        num_kv_heads, d_seq_len, head_dim, scale, groups, max_seq_len
    );
}

// ═══════════════════════════════════════════════════════════════════════
// D2D Memory Copy Helper
// ═══════════════════════════════════════════════════════════════════════

extern "C" void nve_d2d_copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// ═══════════════════════════════════════════════════════════════════════
// CUDA Graph Management
// =====================
// Implements a static decode graph that eliminates ~200 kernel-launch
// overheads per token (~0.6 ms/token on T4) by capturing the full
// multi-layer decode pass into a single CUDA graph and replaying it.
//
// Dynamic parameters (pos, seq_len) are handled via pinned host memory:
//  - g_pin_pos / g_pin_seqlen: pinned host scalars updated before replay
//  - g_dev_pos / g_dev_seqlen: device scalars read by kernels
//  - The captured graph includes cudaMemcpyAsync nodes that copy from
//    the fixed pinned addresses → device scalars.  Updating the pinned
//    values before nve_graph_launch causes the new values to flow in.
//
// Protocol:
//  1. nve_graph_init()              — once at startup
//  2. nve_graph_begin_capture()     — open CUDA graph capture window
//  3. Call decode kernels on nve_graph_capture_stream() for all layers
//  4. nve_graph_end_capture()       — close + instantiate executable graph
//  5. Per step: update input buffer, then nve_graph_launch(pos, seq_len, stream)
//  6. nve_graph_destroy()           — cleanup
// ═══════════════════════════════════════════════════════════════════════

static cudaGraph_t     g_graph      = NULL;
static cudaGraphExec_t g_exec       = NULL;
static cudaStream_t    g_cap_stream = NULL;
static int*            g_pin_pos    = NULL;
static int*            g_pin_seqlen = NULL;
static int*            g_dev_pos    = NULL;
static int*            g_dev_seqlen = NULL;
static int             g_ready      = 0;

extern "C" {

void nve_graph_init(void) {
    if (g_cap_stream) return;  // idempotent
    cudaStreamCreate(&g_cap_stream);
    cudaMallocHost((void**)&g_pin_pos,    sizeof(int));
    cudaMallocHost((void**)&g_pin_seqlen, sizeof(int));
    cudaMalloc((void**)&g_dev_pos,    sizeof(int));
    cudaMalloc((void**)&g_dev_seqlen, sizeof(int));
    *g_pin_pos = 0; *g_pin_seqlen = 1;
    g_ready = 0;
}

const int* nve_graph_d_pos(void)    { return g_dev_pos; }
const int* nve_graph_d_seqlen(void) { return g_dev_seqlen; }
cudaStream_t nve_graph_capture_stream(void) { return g_cap_stream; }

void nve_graph_begin_capture(void) {
    g_ready = 0;
    cudaStreamSynchronize(g_cap_stream);
    cudaStreamBeginCapture(g_cap_stream, cudaStreamCaptureModeGlobal);
    // Record the scalar memcpy nodes first — these will fire at the start of every replay.
    cudaMemcpyAsync(g_dev_pos,    g_pin_pos,    sizeof(int), cudaMemcpyHostToDevice, g_cap_stream);
    cudaMemcpyAsync(g_dev_seqlen, g_pin_seqlen, sizeof(int), cudaMemcpyHostToDevice, g_cap_stream);
}

void nve_graph_end_capture(void) {
    if (g_exec) { cudaGraphExecDestroy(g_exec); g_exec = NULL; }
    cudaStreamEndCapture(g_cap_stream, &g_graph);
    cudaError_t err = cudaGraphInstantiate(&g_exec, g_graph, NULL, NULL, 0);
    cudaGraphDestroy(g_graph);
    g_graph = NULL;
    g_ready = (err == cudaSuccess) ? 1 : 0;
}

// Update pinned scalars then replay the captured graph on `stream`.
// The graph's embedded memcpy nodes read *g_pin_pos and *g_pin_seqlen
// at launch time, so they get the new values.
// Launch the captured graph on `stream`.  Pass NULL to use the CUDA legacy
// default stream, which serialises with any preceding cudaMemcpyAsync(NULL)
// D2D input-copy — ensuring the input buffer is ready before the graph runs.
void nve_graph_launch(int pos, int seq_len, cudaStream_t stream) {
    if (!g_ready || !g_exec) return;
    *g_pin_pos    = pos;
    *g_pin_seqlen = seq_len;
    cudaGraphLaunch(g_exec, stream);  // NULL = legacy default stream
}

int          nve_graph_ready(void) { return g_ready; }

void nve_graph_destroy(void) {
    g_ready = 0;
    if (g_exec)       { cudaGraphExecDestroy(g_exec);    g_exec       = NULL; }
    if (g_cap_stream) { cudaStreamDestroy(g_cap_stream); g_cap_stream = NULL; }
    if (g_pin_pos)    { cudaFreeHost(g_pin_pos);         g_pin_pos    = NULL; }
    if (g_pin_seqlen) { cudaFreeHost(g_pin_seqlen);      g_pin_seqlen = NULL; }
    if (g_dev_pos)    { cudaFree(g_dev_pos);             g_dev_pos    = NULL; }
    if (g_dev_seqlen) { cudaFree(g_dev_seqlen);          g_dev_seqlen = NULL; }
}

} // extern "C"

// ═══════════════════════════════════════════════════════════════════════
// Standalone microbenchmark entry point
// Benchmarks each kernel and prints throughput, called by modal_microbench.py
// ═══════════════════════════════════════════════════════════════════════

extern "C" void nve_run_microbench(
    int hidden,         // e.g. 2048 for 1B model
    int intermediate,   // e.g. 8192 for 1B model
    int num_heads,      // e.g. 32
    int num_kv_heads,   // e.g. 8  (GQA)
    int head_dim,       // e.g. 64
    int iters           // e.g. 1000
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate F16 buffers
    half *x, *w_norm, *out_norm, *q, *k, *cos_buf, *sin_buf;
    half *W_proj, *out_proj;

    cudaMalloc(&x,        hidden * sizeof(half));
    cudaMalloc(&w_norm,   hidden * sizeof(half));
    cudaMalloc(&out_norm, hidden * sizeof(half));
    cudaMalloc(&q,        num_heads * head_dim * sizeof(half));
    cudaMalloc(&k,        num_kv_heads * head_dim * sizeof(half));
    cudaMalloc(&cos_buf,  (head_dim/2) * sizeof(half));
    cudaMalloc(&sin_buf,  (head_dim/2) * sizeof(half));
    cudaMalloc(&W_proj,   (long long)intermediate * hidden * sizeof(half));
    cudaMalloc(&out_proj, intermediate * sizeof(half));

    // Warm up
    for (int i = 0; i < 10; i++) {
        nve_rms_norm_f16((uint16_t*)x, (uint16_t*)w_norm, (uint16_t*)out_norm, hidden, 1e-5f, stream);
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    float ms;

    // ── RMSNorm ──────────────────────────────────────────────────────────
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        nve_rms_norm_f16((uint16_t*)x, (uint16_t*)w_norm, (uint16_t*)out_norm, hidden, 1e-5f, stream);
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[RMSNorm]   hidden=%d  %d iters  %.3f us/call  -->  %.0f calls/s\n",
           hidden, iters, ms*1000.0f/iters, iters*1000.0f/ms);

    // ── RoPE decode ──────────────────────────────────────────────────────
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        nve_rope_f16_decode((uint16_t*)q, (uint16_t*)k,
                            (uint16_t*)cos_buf, (uint16_t*)sin_buf,
                            num_heads, num_kv_heads, head_dim, stream);
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[RoPE dec]  nh=%d nkv=%d hd=%d  %d iters  %.3f us/call  -->  %.0f calls/s\n",
           num_heads, num_kv_heads, head_dim, iters, ms*1000.0f/iters, iters*1000.0f/ms);

    // ── Matvec [1,H] × [I,H]^T ───────────────────────────────────────────
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++)
        nve_matvec_f16((uint16_t*)x, (uint16_t*)W_proj, (uint16_t*)out_proj,
                       intermediate, hidden, stream);
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float gflops = 2.0f * hidden * intermediate / 1e9f;
    printf("[Matvec]    [1,%d] x [%d,%d]^T  %d iters  %.3f us/call  %.1f GFLOPS/s\n",
           hidden, intermediate, hidden, iters, ms*1000.0f/iters,
           gflops / (ms/1000.0f/iters));

    // ── Simulated decode throughput ───────────────────────────────────────
    // One decode token = 2×RMSNorm + RoPE + 7×matmul (Q,K,V,O,gate,up,down)
    // Measure one synthetic layer pass
    cudaEventRecord(t0, stream);
    for (int i = 0; i < iters; i++) {
        nve_rms_norm_f16((uint16_t*)x, (uint16_t*)w_norm, (uint16_t*)out_norm, hidden, 1e-5f, stream);
        nve_rope_f16_decode((uint16_t*)q, (uint16_t*)k,
                            (uint16_t*)cos_buf, (uint16_t*)sin_buf,
                            num_heads, num_kv_heads, head_dim, stream);
        // 7 matvec ops (Q,K,V,O,gate,up,down — simplified, same sizes for now)
        for (int j = 0; j < 7; j++)
            nve_matvec_f16((uint16_t*)x, (uint16_t*)W_proj, (uint16_t*)out_proj,
                           intermediate, hidden, stream);
        nve_rms_norm_f16((uint16_t*)x, (uint16_t*)w_norm, (uint16_t*)out_norm, hidden, 1e-5f, stream);
    }
    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    // tok/s = 1 / (n_layers × seconds_per_layer)
    printf("[1-layer]   %d iters  %.3f ms/layer  --> projected %.0f tok/s (16-layer 1B, naive)\n",
           iters, ms/iters, 1000.0f / (16.0f * ms/iters));
    // Realistic: use actual 1B weight sizes (Q/K/V/O/gate/up/down at true dims)
    // hidden=2048, kv_heads=8, head_dim=64, intermediate=8192
    float real_layer_ms = (2*9.882f + 4.524f) / 1000.0f   // norms + rope (us→ms)
                        + (2048.0f*2048 + 512*2048 + 512*2048 + 2048*2048  // Q,K,V,O
                           + 8192.0f*2048 + 8192*2048 + 2048*8192)         // gate,up,down
                          * 2.0f / (320.0f * 1024 * 1024 * 1024 / 1000.0f); // MB/BW_GBps
    printf("[realistic] bandwidth-limited 1B layer: %.3f ms/layer --> %.0f tok/s\n",
           real_layer_ms, 1000.0f / (16.0f * real_layer_ms));

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(stream);
    cudaFree(x); cudaFree(w_norm); cudaFree(out_norm);
    cudaFree(q); cudaFree(k); cudaFree(cos_buf); cudaFree(sin_buf);
    cudaFree(W_proj); cudaFree(out_proj);
}
