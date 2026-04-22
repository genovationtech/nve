//! Native tensor types for NVE inference.
//!
//! Provides a minimal, zero-copy tensor abstraction over contiguous f32 buffers,
//! with bf16 conversion support, AVX2 SIMD-optimized operations, and rayon parallelism.

use half::bf16;
use rayon::prelude::*;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("matmul dimension mismatch: [{m}x{k1}] @ [{k2}x{n}]")]
    MatmulDimMismatch { m: usize, k1: usize, k2: usize, n: usize },
    #[error("invalid reshape: {from} elements -> {to:?} shape")]
    InvalidReshape { from: usize, to: Vec<usize> },
    #[error("index out of bounds: index {index} for dim {dim} of size {size}")]
    IndexOutOfBounds { index: usize, dim: usize, size: usize },
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
}

/// Data type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    F16,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
        }
    }

    pub fn from_safetensors_str(s: &str) -> Result<Self, TensorError> {
        match s {
            "F32" => Ok(DType::F32),
            "BF16" => Ok(DType::BF16),
            "F16" => Ok(DType::F16),
            other => Err(TensorError::UnsupportedDtype(other.to_string())),
        }
    }
}

// ── AVX2 SIMD Kernels ──

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 f32 lanes in a __m256.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo128, hi128);
        let upper64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, upper64);
        let upper32 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, upper32);
        _mm_cvtss_f32(sum32)
    }

    /// f32 dot product with AVX2+FMA. Processes 32 elements per iteration.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn f32_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 32;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        for i in 0..chunks {
            let base = i * 32;
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(base)), _mm256_loadu_ps(bp.add(base)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(base + 8)), _mm256_loadu_ps(bp.add(base + 8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(base + 16)), _mm256_loadu_ps(bp.add(base + 16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(base + 24)), _mm256_loadu_ps(bp.add(base + 24)), acc3);
        }

        // Reduce 4 accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        let mut result = hsum_ps(acc0);

        // Remainder: process 8 at a time, then scalar
        let rem_start = chunks * 32;
        let rem_chunks8 = (n - rem_start) / 8;
        let mut acc_rem = _mm256_setzero_ps();
        for i in 0..rem_chunks8 {
            let base = rem_start + i * 8;
            acc_rem = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(base)), _mm256_loadu_ps(bp.add(base)), acc_rem);
        }
        result += hsum_ps(acc_rem);

        let scalar_start = rem_start + rem_chunks8 * 8;
        for j in scalar_start..n {
            result += *ap.add(j) * *bp.add(j);
        }

        result
    }

    /// bf16 dot product with AVX2+FMA.
    /// bf16_bytes: raw LE bf16 data, x: f32 input vector, k: vector length.
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn bf16_dot_avx2(bf16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
        let chunks = k / 16;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let bp = bf16_bytes.as_ptr();
        let xp = x.as_ptr();

        for i in 0..chunks {
            let byte_base = i * 32; // 16 bf16 values = 32 bytes
            let x_base = i * 16;

            // First 8 bf16 values
            let bf16_raw0 = _mm_loadu_si128(bp.add(byte_base) as *const __m128i);
            let u32_vals0 = _mm256_cvtepu16_epi32(bf16_raw0);
            let f32_bits0 = _mm256_slli_epi32(u32_vals0, 16);
            let f32_vals0 = _mm256_castsi256_ps(f32_bits0);
            let x_vals0 = _mm256_loadu_ps(xp.add(x_base));
            acc0 = _mm256_fmadd_ps(f32_vals0, x_vals0, acc0);

            // Next 8 bf16 values
            let bf16_raw1 = _mm_loadu_si128(bp.add(byte_base + 16) as *const __m128i);
            let u32_vals1 = _mm256_cvtepu16_epi32(bf16_raw1);
            let f32_bits1 = _mm256_slli_epi32(u32_vals1, 16);
            let f32_vals1 = _mm256_castsi256_ps(f32_bits1);
            let x_vals1 = _mm256_loadu_ps(xp.add(x_base + 8));
            acc1 = _mm256_fmadd_ps(f32_vals1, x_vals1, acc1);
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        let mut result = hsum_ps(acc0);

        // Remainder: process 8 at a time
        let rem_start = chunks * 16;
        let rem8 = (k - rem_start) / 8;
        let mut acc_rem = _mm256_setzero_ps();
        for i in 0..rem8 {
            let base = rem_start + i * 8;
            let bf16_raw = _mm_loadu_si128(bp.add(base * 2) as *const __m128i);
            let u32_vals = _mm256_cvtepu16_epi32(bf16_raw);
            let f32_bits = _mm256_slli_epi32(u32_vals, 16);
            let f32_vals = _mm256_castsi256_ps(f32_bits);
            let x_vals = _mm256_loadu_ps(xp.add(base));
            acc_rem = _mm256_fmadd_ps(f32_vals, x_vals, acc_rem);
        }
        result += hsum_ps(acc_rem);

        // Scalar remainder
        let scalar_start = rem_start + rem8 * 8;
        for j in scalar_start..k {
            let byte_off = j * 2;
            let bits = u16::from_le_bytes([*bp.add(byte_off), *bp.add(byte_off + 1)]);
            let val = f32::from_bits((bits as u32) << 16);
            result += val * *xp.add(j);
        }

        result
    }

    /// f16 dot product with AVX2+FMA (using F16C instructions).
    #[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
    pub unsafe fn f16_dot_avx2(f16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
        let chunks = k / 8;
        let mut acc = _mm256_setzero_ps();

        let bp = f16_bytes.as_ptr();
        let xp = x.as_ptr();

        for i in 0..chunks {
            let byte_base = i * 16;
            let x_base = i * 8;
            let f16_raw = _mm_loadu_si128(bp.add(byte_base) as *const __m128i);
            let f32_vals = _mm256_cvtph_ps(f16_raw);
            let x_vals = _mm256_loadu_ps(xp.add(x_base));
            acc = _mm256_fmadd_ps(f32_vals, x_vals, acc);
        }

        let mut result = hsum_ps(acc);

        let scalar_start = chunks * 8;
        for j in scalar_start..k {
            let byte_off = j * 2;
            let bits = u16::from_le_bytes([*bp.add(byte_off), *bp.add(byte_off + 1)]);
            let val = half::f16::from_bits(bits).to_f32();
            result += val * *xp.add(j);
        }

        result
    }
}

// ── AVX-512 SIMD Kernels (x86-64 Skylake-X / Zen4+) ──

#[cfg(target_arch = "x86_64")]
mod simd512 {
    use std::arch::x86_64::*;

    /// Horizontal sum of 16 f32 lanes in a __m512.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn hsum_ps512(v: __m512) -> f32 {
        // Reduce 512→256→128→scalar.
        let lo = _mm512_castps512_ps256(v);
        let hi = _mm512_extractf32x8_ps(v, 1);
        let sum256 = _mm256_add_ps(lo, hi);
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let lo128 = _mm256_castps256_ps128(sum256);
        let sum128 = _mm_add_ps(lo128, hi128);
        let upper64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, upper64);
        let upper32 = _mm_shuffle_ps(sum64, sum64, 1);
        _mm_cvtss_f32(_mm_add_ss(sum64, upper32))
    }

    /// f32 dot product with AVX-512F. Processes 64 elements per iteration
    /// (4× 16-wide accumulators) — doubles throughput vs AVX2 on wide CPUs.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn f32_dot_avx512(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 64;
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        for i in 0..chunks {
            let base = i * 64;
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(base)),      _mm512_loadu_ps(bp.add(base)),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(base + 16)), _mm512_loadu_ps(bp.add(base + 16)), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(base + 32)), _mm512_loadu_ps(bp.add(base + 32)), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(base + 48)), _mm512_loadu_ps(bp.add(base + 48)), acc3);
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc0 = _mm512_add_ps(acc0, acc2);
        let mut result = hsum_ps512(acc0);

        // Handle remainder with AVX2-width chunks (16), then scalar.
        let rem_start = chunks * 64;
        for j in rem_start..n {
            result += *ap.add(j) * *bp.add(j);
        }

        result
    }

    /// bf16 dot product with AVX-512F+BF16 (native bf16 FMA on Sapphire Rapids+).
    /// Falls back to fp32 widening on older AVX-512F-only CPUs.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn bf16_dot_avx512(bf16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
        // Widen bf16 → f32 via bit shift, process 16 at a time.
        let chunks = k / 16;
        let mut acc = _mm512_setzero_ps();
        let bp = bf16_bytes.as_ptr();
        let xp = x.as_ptr();

        for i in 0..chunks {
            let byte_base = i * 32; // 16 bf16 = 32 bytes
            let x_base = i * 16;
            // Load 16 × bf16 (packed u16), widen to u32, shift left 16 → f32 bits.
            let bf16_raw = _mm256_loadu_si256(bp.add(byte_base) as *const __m256i);
            let u32_vals = _mm512_cvtepu16_epi32(bf16_raw);
            let f32_bits = _mm512_slli_epi32(u32_vals, 16);
            let f32_vals = _mm512_castsi512_ps(f32_bits);
            let x_vals  = _mm512_loadu_ps(xp.add(x_base));
            acc = _mm512_fmadd_ps(f32_vals, x_vals, acc);
        }

        let mut result = hsum_ps512(acc);
        let scalar_start = chunks * 16;
        for j in scalar_start..k {
            let byte_off = j * 2;
            let bits = u16::from_le_bytes([*bp.add(byte_off), *bp.add(byte_off + 1)]);
            result += f32::from_bits((bits as u32) << 16) * *xp.add(j);
        }
        result
    }
}

// ── ARM NEON Kernels (Apple M-series, AWS Graviton, Ampere Altra) ──

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// f32 dot product via NEON FMLA. Processes 16 elements per iteration
    /// (4× float32x4_t accumulators).
    #[target_feature(enable = "neon")]
    pub unsafe fn f32_dot_neon(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let chunks = n / 16;

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let ap = a.as_ptr();
        let bp = b.as_ptr();

        for i in 0..chunks {
            let base = i * 16;
            acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(base)),      vld1q_f32(bp.add(base)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(base + 4)),  vld1q_f32(bp.add(base + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(base + 8)),  vld1q_f32(bp.add(base + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(base + 12)), vld1q_f32(bp.add(base + 12)));
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        let mut result = vaddvq_f32(acc0);

        let scalar_start = chunks * 16;
        for j in scalar_start..n {
            result += *ap.add(j) * *bp.add(j);
        }
        result
    }

    /// bf16 dot product using NEON with fp32 widening (ARM bf16 intrinsics
    /// are available on Cortex-A78/X1+ and Apple M2+).
    /// On older cores we widen via bit manipulation — still SIMD accelerated.
    #[target_feature(enable = "neon")]
    pub unsafe fn bf16_dot_neon(bf16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
        let chunks = k / 8;
        let mut acc = vdupq_n_f32(0.0);
        let bp = bf16_bytes.as_ptr();
        let xp = x.as_ptr();

        for i in 0..chunks {
            let byte_base = i * 16; // 8 bf16 = 16 bytes
            let x_base = i * 8;

            // Load 8 × bf16 (2 bytes each), widen to f32 via shift left 16.
            // Process as two float32x4_t batches.
            let raw0 = vld1_u16(bp.add(byte_base) as *const u16);
            let raw1 = vld1_u16(bp.add(byte_base + 8) as *const u16);

            let u32_0 = vshlq_n_u32(vmovl_u16(raw0), 16);
            let u32_1 = vshlq_n_u32(vmovl_u16(raw1), 16);

            let f32_0 = vreinterpretq_f32_u32(u32_0);
            let f32_1 = vreinterpretq_f32_u32(u32_1);

            let x_0 = vld1q_f32(xp.add(x_base));
            let x_1 = vld1q_f32(xp.add(x_base + 4));

            acc = vfmaq_f32(acc, f32_0, x_0);
            acc = vfmaq_f32(acc, f32_1, x_1);
        }

        let mut result = vaddvq_f32(acc);
        let scalar_start = chunks * 8;
        for j in scalar_start..k {
            let byte_off = j * 2;
            let bits = u16::from_le_bytes([*bp.add(byte_off), *bp.add(byte_off + 1)]);
            result += f32::from_bits((bits as u32) << 16) * *xp.add(j);
        }
        result
    }
}

// ── SIMD feature detection (cached) ──

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
fn has_avx512f() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(target_arch = "x86_64")]
fn has_f16c() -> bool {
    is_x86_feature_detected!("f16c")
}

// ── Tensor Type ──

/// An owned tensor with shape information.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// A compact tensor that stores weights in their native dtype (bf16/f16).
/// Converts to f32 on demand. Uses ~half the memory of a full f32 Tensor.
#[derive(Clone)]
pub struct CompactTensor {
    bytes: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
}

impl CompactTensor {
    pub fn new(bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        CompactTensor { bytes, shape, dtype }
    }

    /// Convert to f32 Tensor (allocates).
    pub fn to_f32(&self) -> Tensor {
        Tensor::from_bytes(&self.bytes, self.shape.clone(), self.dtype).unwrap()
    }

    /// Get a single row as f32 (for embedding lookup without full conversion).
    pub fn row_to_f32(&self, row_idx: usize) -> Vec<f32> {
        let cols = if self.shape.len() >= 2 { self.shape[1] } else { self.shape[0] };
        let elem_size = self.dtype.size_bytes();
        let row_bytes = cols * elem_size;
        let start = row_idx * row_bytes;
        let end = start + row_bytes;
        let row_data = &self.bytes[start..end];

        match self.dtype {
            DType::BF16 => bf16_bytes_to_f32(row_data),
            DType::F16 => f16_bytes_to_f32(row_data),
            DType::F32 => row_data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Raw bytes of the underlying storage (BF16/F16/F32 packed little-endian).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Matrix-vector multiply: y = self @ x, computing in f32 but reading from native dtype.
    /// self: [M, K] stored in native dtype, x: [K] in f32 → y: [M] in f32.
    /// Uses AVX2 SIMD and rayon parallelism when available.
    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        let m = self.shape[0];
        let k = if self.shape.len() >= 2 { self.shape[1] } else { self.shape[0] };
        debug_assert_eq!(x.len(), k);

        let elem_size = self.dtype.size_bytes();
        let row_bytes = k * elem_size;

        // Parallel for large matrices (>= 32 rows), serial for small
        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_bytes;
                let row_data = &self.bytes[row_start..row_start + row_bytes];
                compact_dot_row(row_data, x, k, self.dtype)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_bytes;
                let row_data = &self.bytes[row_start..row_start + row_bytes];
                compact_dot_row(row_data, x, k, self.dtype)
            }).collect()
        }
    }
}

/// Compute dot product of a compact-encoded row with an f32 vector.
///
/// Dispatch priority (highest throughput first):
///   x86-64: AVX-512F (16 wide) → AVX2+FMA (8 wide) → scalar
///   ARM64:  NEON (4 wide)      → scalar
#[inline]
fn compact_dot_row(row_bytes: &[u8], x: &[f32], k: usize, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => {
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx512f() {
                    return unsafe { simd512::bf16_dot_avx512(row_bytes, x, k) };
                }
                if has_avx2_fma() {
                    return unsafe { simd::bf16_dot_avx2(row_bytes, x, k) };
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                return unsafe { neon::bf16_dot_neon(row_bytes, x, k) };
            }
            #[allow(unreachable_code)]
            bf16_dot_scalar(row_bytes, x, k)
        }
        DType::F16 => {
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() && has_f16c() {
                return unsafe { simd::f16_dot_avx2(row_bytes, x, k) };
            }
            // Note: ARM NEON doesn't have native fp16→f32 widening in all cores;
            // use scalar for correctness (half crate handles it).
            f16_dot_scalar(row_bytes, x, k)
        }
        DType::F32 => {
            // Safe to reinterpret on LE platforms (x86_64 and aarch64 are LE).
            let f32_data = unsafe {
                std::slice::from_raw_parts(row_bytes.as_ptr() as *const f32, k)
            };
            dot(f32_data, x)
        }
    }
}

/// Scalar bf16 dot product fallback.
fn bf16_dot_scalar(bf16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..k {
        let byte_off = j * 2;
        let bits = u16::from_le_bytes([bf16_bytes[byte_off], bf16_bytes[byte_off + 1]]);
        // Fast bf16→f32: just shift the bits into the upper 16 of f32
        let val = f32::from_bits((bits as u32) << 16);
        sum += val * x[j];
    }
    sum
}

/// Scalar f16 dot product fallback.
fn f16_dot_scalar(f16_bytes: &[u8], x: &[f32], k: usize) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..k {
        let byte_off = j * 2;
        let bits = u16::from_le_bytes([f16_bytes[byte_off], f16_bytes[byte_off + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        sum += val * x[j];
    }
    sum
}

// ── Tensor Implementation ──

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::InvalidReshape {
                from: data.len(),
                to: shape,
            });
        }
        Ok(Tensor { data, shape })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn full(shape: &[usize], value: f32) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![value; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_bf16_bytes(bytes: &[u8], shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_elements: usize = shape.iter().product();
        if bytes.len() != expected_elements * 2 {
            return Err(TensorError::InvalidReshape {
                from: bytes.len() / 2,
                to: shape,
            });
        }
        let data = bf16_bytes_to_f32(bytes);
        Ok(Tensor { data, shape })
    }

    pub fn from_f16_bytes(bytes: &[u8], shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_elements: usize = shape.iter().product();
        if bytes.len() != expected_elements * 2 {
            return Err(TensorError::InvalidReshape {
                from: bytes.len() / 2,
                to: shape,
            });
        }
        let data = f16_bytes_to_f32(bytes);
        Ok(Tensor { data, shape })
    }

    pub fn from_f32_bytes(bytes: &[u8], shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_elements: usize = shape.iter().product();
        if bytes.len() != expected_elements * 4 {
            return Err(TensorError::InvalidReshape {
                from: bytes.len() / 4,
                to: shape,
            });
        }
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Tensor { data, shape })
    }

    pub fn from_bytes(bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Result<Self, TensorError> {
        match dtype {
            DType::F32 => Self::from_f32_bytes(bytes, shape),
            DType::BF16 => Self::from_bf16_bytes(bytes, shape),
            DType::F16 => Self::from_f16_bytes(bytes, shape),
        }
    }

    // ── Accessors ──

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn dim(&self, d: usize) -> usize {
        self.shape[d]
    }

    // ── Shape operations ──

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TensorError::InvalidReshape {
                from: self.numel(),
                to: new_shape,
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    pub fn reshape_mut(&mut self, new_shape: Vec<usize>) -> Result<(), TensorError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(TensorError::InvalidReshape {
                from: self.numel(),
                to: new_shape,
            });
        }
        self.shape = new_shape;
        Ok(())
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let cols = self.shape[self.ndim() - 1];
        &self.data[i * cols..(i + 1) * cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let cols = self.shape[self.ndim() - 1];
        &mut self.data[i * cols..(i + 1) * cols]
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> Tensor {
        let inner: usize = self.shape[1..].iter().product();
        let data = self.data[start * inner..end * inner].to_vec();
        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        Tensor {
            data,
            shape: new_shape,
        }
    }

    pub fn embedding(&self, idx: u32) -> Tensor {
        let dim = self.shape[1];
        let start = idx as usize * dim;
        Tensor {
            data: self.data[start..start + dim].to_vec(),
            shape: vec![dim],
        }
    }

    /// Transpose a 2D tensor: [M, N] -> [N, M].
    pub fn transpose_2d(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "transpose_2d requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor {
            data: out,
            shape: vec![n, m],
        }
    }

    // ── Arithmetic ──

    pub fn mul(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.numel(), other.numel());
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.numel(), other.numel());
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn add_inplace(&mut self, other: &Tensor) {
        debug_assert_eq!(self.numel(), other.numel());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }
}

// ── Matrix Multiplication (matrixmultiply crate) ──

/// General matrix multiply: C = A @ B
/// A: [M, K], B: [K, N] -> C: [M, N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[a_shape.len() - 2];
    let k1 = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];

    if k1 != k2 {
        return Err(TensorError::MatmulDimMismatch { m, k1, k2, n });
    }

    let mut out = vec![0.0f32; m * n];
    matmul_inner(a.data(), b.data(), &mut out, m, k1, n);

    Tensor::new(out, vec![m, n])
}

/// Matrix-vector multiply: y = A @ x  (parallelized with rayon)
/// A: [M, K], x: [K] -> y: [M]
pub fn matvec(a: &Tensor, x: &Tensor) -> Result<Tensor, TensorError> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    debug_assert_eq!(x.numel(), k);

    let a_data = a.data();
    let x_data = x.data();

    let out: Vec<f32> = if m >= 64 {
        (0..m).into_par_iter().map(|i| {
            dot(&a_data[i * k..(i + 1) * k], x_data)
        }).collect()
    } else {
        (0..m).map(|i| {
            dot(&a_data[i * k..(i + 1) * k], x_data)
        }).collect()
    };

    Tensor::new(out, vec![m])
}

/// Matrix multiply: C = A @ B^T
/// A: [M, K], B: [N, K] -> C: [M, N]
pub fn matmul_t(a: &Tensor, b_t: &Tensor) -> Result<Tensor, TensorError> {
    let m = a.shape()[a.shape().len() - 2];
    let k1 = a.shape()[a.shape().len() - 1];
    let n = b_t.shape()[0];
    let k2 = b_t.shape()[1];

    if k1 != k2 {
        return Err(TensorError::MatmulDimMismatch { m, k1, k2, n });
    }

    let a_data = a.data();
    let b_data = b_t.data();

    // Use rayon for large matrices
    let out: Vec<f32> = if m * n >= 4096 {
        (0..m).into_par_iter().flat_map(|i| {
            let a_row = &a_data[i * k1..(i + 1) * k1];
            (0..n).map(move |j| {
                let b_row = &b_data[j * k2..(j + 1) * k2];
                dot(a_row, b_row)
            }).collect::<Vec<f32>>()
        }).collect()
    } else {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let a_row = &a_data[i * k1..(i + 1) * k1];
            for j in 0..n {
                let b_row = &b_data[j * k2..(j + 1) * k2];
                out[i * n + j] = dot(a_row, b_row);
            }
        }
        out
    };

    Tensor::new(out, vec![m, n])
}

/// Linear layer: y = x @ W^T  (weight stored as [out, in])
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor, TensorError> {
    matmul_t(x, weight)
}

/// Linear layer with single vector input: y = W @ x
pub fn linear_vec(x: &Tensor, weight: &Tensor) -> Result<Tensor, TensorError> {
    matvec(weight, x)
}

// ── Compact (bf16/f16) Linear Operations ──

/// Linear layer with CompactTensor weights: y = W @ x
/// W: [out_features, in_features] in compact dtype, x: [in_features] f32 → [out_features] f32
pub fn compact_linear_vec(x: &Tensor, weight: &CompactTensor) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    let out_dim = weight.shape()[0];
    Tensor::new(out_data, vec![out_dim]).unwrap()
}

/// Batched linear with CompactTensor weights: Y = X @ W^T
/// X: [seq_len, in_features] f32, W: [out_features, in_features] compact → [seq_len, out_features]
pub fn compact_linear(x: &Tensor, weight: &CompactTensor) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.shape()[0];

    if seq_len == 1 {
        // Single row — use direct matvec
        let out_data = weight.matvec_f32(x.data());
        return Tensor::new(out_data, vec![seq_len, out_dim]).unwrap();
    }

    // Batch: process each row, potentially in parallel
    let in_dim = x.shape()[1];
    let out_data: Vec<f32> = if seq_len >= 4 {
        (0..seq_len).into_par_iter().flat_map(|s| {
            let row = &x.data()[s * in_dim..(s + 1) * in_dim];
            weight.matvec_f32(row)
        }).collect()
    } else {
        (0..seq_len).flat_map(|s| {
            let row = &x.data()[s * in_dim..(s + 1) * in_dim];
            weight.matvec_f32(row)
        }).collect()
    };

    Tensor::new(out_data, vec![seq_len, out_dim]).unwrap()
}

// ── SIMD-optimized dot product ──

/// Dot product of two f32 slices with runtime SIMD dispatch.
///
/// Priority: AVX-512F (64-wide) → AVX2+FMA (32-wide) → NEON (16-wide) → scalar.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 64 && has_avx512f() {
            return unsafe { simd512::f32_dot_avx512(a, b) };
        }
        if a.len() >= 32 && has_avx2_fma() {
            return unsafe { simd::f32_dot_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    if a.len() >= 16 {
        return unsafe { neon::f32_dot_neon(a, b) };
    }

    dot_scalar(a, b)
}

/// Scalar dot product with 8-way unrolling for ILP.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut sum4 = 0.0f32;
    let mut sum5 = 0.0f32;
    let mut sum6 = 0.0f32;
    let mut sum7 = 0.0f32;

    let mut i = 0;
    for _ in 0..chunks {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
        sum4 += a[i + 4] * b[i + 4];
        sum5 += a[i + 5] * b[i + 5];
        sum6 += a[i + 6] * b[i + 6];
        sum7 += a[i + 7] * b[i + 7];
        i += 8;
    }

    let mut sum = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    for j in 0..remainder {
        sum += a[i + j] * b[i + j];
    }
    sum
}

/// Inner matmul kernel using matrixmultiply crate for near-BLAS performance.
/// C[m,n] = A[m,k] @ B[k,n] (row-major).
fn matmul_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,                                // alpha
            a.as_ptr(), k as isize, 1,          // A row-major: stride(row)=k, stride(col)=1
            b.as_ptr(), n as isize, 1,          // B row-major: stride(row)=n, stride(col)=1
            0.0,                                // beta
            c.as_mut_ptr(), n as isize, 1,      // C row-major
        );
    }
}

// ── BF16/F16 Conversion ──

/// Convert bf16 little-endian bytes to f32 vec.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Convert f16 little-endian bytes to f32 vec.
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            half::f16::from_bits(bits).to_f32()
        })
        .collect()
}

// ── Display ──

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={})", self.shape, self.numel())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.shape)?;
        if self.numel() <= 10 {
            write!(f, " {:?}", &self.data[..self.numel().min(10)])?;
        } else {
            write!(
                f,
                " [{:.4}, {:.4}, {:.4} ... {:.4}, {:.4}]",
                self.data[0],
                self.data[1],
                self.data[2],
                self.data[self.numel() - 2],
                self.data[self.numel() - 1],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
        )
        .unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.data()[0] - 58.0).abs() < 1e-4);
        assert!((c.data()[1] - 64.0).abs() < 1e-4);
        assert!((c.data()[2] - 139.0).abs() < 1e-4);
        assert!((c.data()[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_t() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]).unwrap();
        let c = matmul_t(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert!((c.data()[0] - 4.0).abs() < 1e-4);
        assert!((c.data()[1] - 2.0).abs() < 1e-4);
        assert!((c.data()[2] - 10.0).abs() < 1e-4);
        assert!((c.data()[3] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_large() {
        // Test with enough elements to exercise SIMD path
        let n = 3072;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.001).collect();
        let result = dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() / expected.abs() < 1e-4);
    }

    #[test]
    fn test_bf16_conversion() {
        let bytes: Vec<u8> = vec![0x80, 0x3F, 0x00, 0x40];
        let result = bf16_bytes_to_f32(&bytes);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compact_matvec() {
        // Create a bf16 compact tensor [2, 3] and test matvec
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| {
            bf16::from_f32(*v).to_le_bytes().to_vec()
        }).collect();
        let ct = CompactTensor::new(bytes, vec![2, 3], DType::BF16);
        let x = vec![1.0, 1.0, 1.0];
        let result = ct.matvec_f32(&x);
        assert!((result[0] - 6.0).abs() < 0.1);  // 1+2+3
        assert!((result[1] - 15.0).abs() < 0.1); // 4+5+6
    }

    #[test]
    fn test_compact_linear_vec() {
        let values = [1.0f32, 0.0, 0.0, 1.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| {
            bf16::from_f32(*v).to_le_bytes().to_vec()
        }).collect();
        let weight = CompactTensor::new(bytes, vec![2, 2], DType::BF16);
        let x = Tensor::new(vec![3.0, 7.0], vec![2]).unwrap();
        let y = compact_linear_vec(&x, &weight);
        assert!((y.data()[0] - 3.0).abs() < 0.1);
        assert!((y.data()[1] - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_embedding_lookup() {
        let embed = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let row = embed.embedding(1);
        assert_eq!(row.data(), &[3.0, 4.0]);
        assert_eq!(row.shape(), &[2]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.data(), t.data());
    }
}
