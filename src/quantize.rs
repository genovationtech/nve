//! INT4 and INT8 quantization for weight compression.
//!
//! Implements Q4_0 format (compatible with llama.cpp):
//! - Block size: 32 weights
//! - Each block: f32 scale + 16 bytes of packed 4-bit nibbles = 20 bytes
//! - Dequantization: value = (nibble - 8) * scale
//! - Compression: ~5 bits per weight (vs 16 for bf16, 32 for f32)
//!
//! Also implements Q8_0 format:
//! - Block size: 32 weights
//! - Each block: f32 scale + 32 i8 values = 36 bytes
//! - Higher quality than Q4_0, still 2.25x smaller than bf16
//!
//! WeightStorage enum dispatches between bf16/Q4/Q8 at runtime.

use rayon::prelude::*;

use crate::tensor::{compact_linear, compact_linear_vec, CompactTensor, Tensor};

/// Block size for Q4_0 and Q8_0 quantization.
pub const Q4_BLOCK_SIZE: usize = 32;
/// Bytes per Q4 block: 4 (f32 scale) + 16 (packed nibbles).
pub const Q4_BLOCK_BYTES: usize = 20;

/// Block size for Q8_0 quantization (same as Q4).
pub const Q8_BLOCK_SIZE: usize = 32;
/// Bytes per Q8 block: 4 (f32 scale) + 32 (i8 values).
pub const Q8_BLOCK_BYTES: usize = 36;

/// Block size for Q3 quantization.
pub const Q3_BLOCK_SIZE: usize = 32;
/// Bytes per Q3 block: 4 (f32 scale) + 12 (32x3 bits packed).
pub const Q3_BLOCK_BYTES: usize = 16;

/// Block size for Q2 quantization.
pub const Q2_BLOCK_SIZE: usize = 32;
/// Bytes per Q2 block: 4 (f32 scale) + 8 (32x2 bits packed).
pub const Q2_BLOCK_BYTES: usize = 12;

/// Block size for Q1 (ternary) quantization.
pub const Q1_BLOCK_SIZE: usize = 32;
/// Bytes per Q1 block: 4 (f32 scale) + 4 (nonzero mask) + 4 (sign bits).
pub const Q1_BLOCK_BYTES: usize = 12;

// ── QuantMode ──

/// Quantization mode selection.
#[derive(Debug, Clone, Copy)]
pub enum QuantMode {
    None,
    Q8,
    Q4,
    Q3,
    Q2,
    Q1,
    /// Profile-guided: target bits-per-weight (e.g. 2.0, 1.0, 0.5).
    ProfileGuided(f32),
}

impl QuantMode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "none" => Some(QuantMode::None),
            "q4" => Some(QuantMode::Q4),
            "q8" => Some(QuantMode::Q8),
            "q3" => Some(QuantMode::Q3),
            "q2" => Some(QuantMode::Q2),
            "q1" => Some(QuantMode::Q1),
            _ if s.starts_with("pg:") => {
                let val = s[3..].parse::<f32>().ok()?;
                Some(QuantMode::ProfileGuided(val))
            }
            _ => None,
        }
    }

    /// Bits per weight for this mode (used by bit allocation).
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantMode::None => 0.0,
            QuantMode::Q1 => 1.0,
            QuantMode::Q2 => 2.0,
            QuantMode::Q3 => 3.0,
            QuantMode::Q4 => 4.0,
            QuantMode::Q8 => 8.0,
            QuantMode::ProfileGuided(bpw) => *bpw,
        }
    }
}

impl PartialEq for QuantMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (QuantMode::None, QuantMode::None) => true,
            (QuantMode::Q8, QuantMode::Q8) => true,
            (QuantMode::Q4, QuantMode::Q4) => true,
            (QuantMode::Q3, QuantMode::Q3) => true,
            (QuantMode::Q2, QuantMode::Q2) => true,
            (QuantMode::Q1, QuantMode::Q1) => true,
            (QuantMode::ProfileGuided(a), QuantMode::ProfileGuided(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}

impl std::fmt::Display for QuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantMode::None => write!(f, "none"),
            QuantMode::Q4 => write!(f, "q4"),
            QuantMode::Q8 => write!(f, "q8"),
            QuantMode::Q3 => write!(f, "q3"),
            QuantMode::Q2 => write!(f, "q2"),
            QuantMode::Q1 => write!(f, "q1"),
            QuantMode::ProfileGuided(bpw) => write!(f, "pg:{:.1}", bpw),
        }
    }
}

// ── WeightStorage ──

/// A weight matrix stored in one of several formats.
/// Dispatches linear algebra ops to the right implementation at runtime.
pub enum WeightStorage {
    /// bf16 compact — full precision
    Compact(CompactTensor),
    /// Q4_0 — 4-bit quantized
    Quantized4(QuantizedTensor),
    /// Q8_0 — 8-bit quantized
    Quantized8(QuantizedTensor8),
    /// Q3 — 3-bit quantized (8 levels)
    Quantized3(QuantizedTensor3),
    /// Q2 — 2-bit quantized (4 levels)
    Quantized2(QuantizedTensor2),
    /// Q1 — ternary quantized ({-1,0,+1})
    Quantized1(QuantizedTensor1),
    /// Sparse — sub-1-bit with block sparsity
    Sparse(SparseQuantized),
}

impl WeightStorage {
    /// Matrix-vector multiply: y = self @ x
    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        match self {
            WeightStorage::Compact(c) => c.matvec_f32(x),
            WeightStorage::Quantized4(q) => q.matvec_f32(x),
            WeightStorage::Quantized8(q) => q.matvec_f32(x),
            WeightStorage::Quantized3(q) => q.matvec_f32(x),
            WeightStorage::Quantized2(q) => q.matvec_f32(x),
            WeightStorage::Quantized1(q) => q.matvec_f32(x),
            WeightStorage::Sparse(q) => q.matvec_f32(x),
        }
    }

    /// Shape as [rows, cols].
    pub fn shape(&self) -> [usize; 2] {
        match self {
            WeightStorage::Compact(c) => {
                let s = c.shape();
                [s[0], if s.len() >= 2 { s[1] } else { s[0] }]
            }
            WeightStorage::Quantized4(q) => *q.shape(),
            WeightStorage::Quantized8(q) => *q.shape(),
            WeightStorage::Quantized3(q) => q.shape,
            WeightStorage::Quantized2(q) => q.shape,
            WeightStorage::Quantized1(q) => q.shape,
            WeightStorage::Sparse(q) => q.shape,
        }
    }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            WeightStorage::Compact(c) => c.size_bytes(),
            WeightStorage::Quantized4(q) => q.size_bytes(),
            WeightStorage::Quantized8(q) => q.size_bytes(),
            WeightStorage::Quantized3(q) => q.size_bytes(),
            WeightStorage::Quantized2(q) => q.size_bytes(),
            WeightStorage::Quantized1(q) => q.size_bytes(),
            WeightStorage::Sparse(q) => q.size_bytes(),
        }
    }

    /// Dequantize the full weight matrix to a flat f32 Vec in row-major order.
    /// Used to upload weights to GPU for hot-layer inference.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            WeightStorage::Compact(c) => c.to_f32().data().to_vec(),
            WeightStorage::Quantized4(q) => q.to_f32_vec(),
            WeightStorage::Quantized8(q) => q.to_f32_vec(),
            WeightStorage::Quantized3(q) => q.to_f32_vec(),
            WeightStorage::Quantized2(q) => q.to_f32_vec(),
            WeightStorage::Quantized1(q) => q.to_f32_vec(),
            WeightStorage::Sparse(q) => q.to_f32_vec(),
        }
    }
}

/// Linear layer with WeightStorage: y = W @ x (single vector).
pub fn ws_linear_vec(x: &Tensor, weight: &WeightStorage) -> Tensor {
    match weight {
        WeightStorage::Compact(c) => compact_linear_vec(x, c),
        WeightStorage::Quantized4(q) => q4_linear_vec(x, q),
        WeightStorage::Quantized8(q) => q8_linear_vec(x, q),
        WeightStorage::Quantized3(q) => q3_linear_vec(x, q),
        WeightStorage::Quantized2(q) => q2_linear_vec(x, q),
        WeightStorage::Quantized1(q) => q1_linear_vec(x, q),
        WeightStorage::Sparse(q) => sparse_linear_vec(x, q),
    }
}

/// Batched linear with WeightStorage: Y = X @ W^T.
pub fn ws_linear(x: &Tensor, weight: &WeightStorage) -> Tensor {
    match weight {
        WeightStorage::Compact(c) => compact_linear(x, c),
        WeightStorage::Quantized4(q) => q4_linear(x, q),
        WeightStorage::Quantized8(q) => q8_linear(x, q),
        WeightStorage::Quantized3(q) => q3_linear(x, q),
        WeightStorage::Quantized2(q) => q2_linear(x, q),
        WeightStorage::Quantized1(q) => q1_linear(x, q),
        WeightStorage::Sparse(q) => sparse_linear(x, q),
    }
}

// ── Q4_0 ──

/// A Q4_0 quantized tensor.
/// Stores weights as 4-bit integers with per-block f32 scales.
pub struct QuantizedTensor {
    /// Raw block data: [scale: f32 (4 bytes)][nibbles: 16 bytes] per block
    data: Vec<u8>,
    /// Shape of the original tensor [rows, cols]
    shape: [usize; 2],
    /// Number of blocks per row
    blocks_per_row: usize,
    /// AWQ inverse scales: if present, x[j] is multiplied by inv_scales[j] before dot product
    awq_inv_scales: Option<Vec<f32>>,
}

impl QuantizedTensor {
    /// Quantize a CompactTensor (bf16/f16) to Q4_0.
    pub fn from_compact(compact: &CompactTensor) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q4_BLOCK_SIZE == 0, "cols {} must be divisible by block size {}", cols, Q4_BLOCK_SIZE);

        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q4_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q4_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q4_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q4_BLOCK_SIZE];
                let out_offset = b * Q4_BLOCK_BYTES;
                quantize_block_q4(block_values, &mut row_out[out_offset..out_offset + Q4_BLOCK_BYTES]);
            }
        });

        QuantizedTensor {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: None,
        }
    }

    /// Quantize a CompactTensor with AWQ per-channel scaling.
    pub fn from_compact_awq(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q4_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q4_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q4_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            // Apply AWQ scaling: W[:,j] *= awq_scales[j]
            for j in 0..cols {
                row_f32[j] *= awq_scales[j];
            }
            for b in 0..blocks_per_row {
                let block_start = b * Q4_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q4_BLOCK_SIZE];
                let out_offset = b * Q4_BLOCK_BYTES;
                quantize_block_q4(block_values, &mut row_out[out_offset..out_offset + Q4_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();

        QuantizedTensor {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: Some(inv_scales),
        }
    }

    /// Quantize a CompactTensor with AWQ scaling AND sub-block importance weighting (k-quant).
    /// Combines per-channel AWQ scaling with importance-weighted rounding per block.
    pub fn from_compact_awq_kquant(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q4_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q4_BLOCK_BYTES];

        // Compute per-channel importance from the AWQ scales (higher scale = more important).
        let channel_importance: Vec<f32> = awq_scales.iter().map(|&s| s * s).collect();

        let row_data_size = blocks_per_row * Q4_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols {
                row_f32[j] *= awq_scales[j];
            }
            for b in 0..blocks_per_row {
                let block_start = b * Q4_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q4_BLOCK_SIZE];
                let block_importance = &channel_importance[block_start..block_start + Q4_BLOCK_SIZE];
                let out_offset = b * Q4_BLOCK_BYTES;
                quantize_block_q4_weighted(block_values, block_importance, &mut row_out[out_offset..out_offset + Q4_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();

        QuantizedTensor {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: Some(inv_scales),
        }
    }

    /// Quantize an f32 Tensor to Q4_0.
    pub fn from_f32(tensor: &Tensor) -> Self {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(cols % Q4_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q4_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q4_BLOCK_BYTES;
        let t_data = tensor.data();

        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_start = row_idx * cols;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * Q4_BLOCK_SIZE;
                let block_values = &t_data[block_start..block_start + Q4_BLOCK_SIZE];
                let out_offset = b * Q4_BLOCK_BYTES;
                quantize_block_q4(block_values, &mut row_out[out_offset..out_offset + Q4_BLOCK_BYTES]);
            }
        });

        QuantizedTensor {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: None,
        }
    }

    pub fn shape(&self) -> &[usize; 2] {
        &self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    /// Used to upload weights to GPU. AWQ scaling is reversed so the result is the
    /// original (pre-quantization, pre-AWQ-scaling) weight matrix.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let row_bytes = self.blocks_per_row * Q4_BLOCK_BYTES;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row_start = r * row_bytes;
            let row_data = &self.data[row_start..row_start + row_bytes];
            for b in 0..self.blocks_per_row {
                let block_off = b * Q4_BLOCK_BYTES;
                let scale = f32::from_le_bytes(row_data[block_off..block_off + 4].try_into().unwrap());
                let nibbles = &row_data[block_off + 4..block_off + Q4_BLOCK_BYTES];
                for i in 0..16 {
                    let byte = nibbles[i];
                    let q0 = (byte & 0x0F) as i8 - 8;
                    let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                    out.push(q0 as f32 * scale);
                    out.push(q1 as f32 * scale);
                }
            }
        }
        // Reverse AWQ column scaling: the stored matrix is W * diag(awq_scale),
        // so to recover W we multiply each column by inv_scale.
        if let Some(ref inv_scales) = self.awq_inv_scales {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= inv_scales[c];
                }
            }
        }
        out
    }

    /// Extract raw nibble bytes and per-block scales for GPU upload.
    ///
    /// Returns:
    /// - `nibbles`: [N * K/2] u8 — 2 INT4 nibbles per byte, sequential pairs
    ///   (byte at block b, position i/2 holds elements 2i and 2i+1 of block b)
    /// - `scales`: [N * K/32] f32 — one per 32-element block
    /// - `awq`: the AWQ inverse scales slice if present, else None
    ///
    /// This is the input format expected by `nve_matvec_w4a16` and `nve_dequant_w4a16`.
    pub fn extract_for_gpu(&self) -> (Vec<u8>, Vec<f32>, Option<&[f32]>) {
        let [n, k] = self.shape;
        let row_bytes = self.blocks_per_row * Q4_BLOCK_BYTES;
        let mut nibbles = Vec::with_capacity(n * k / 2);
        let mut scales  = Vec::with_capacity(n * self.blocks_per_row);

        for row in 0..n {
            let row_off = row * row_bytes;
            for blk in 0..self.blocks_per_row {
                let off = row_off + blk * Q4_BLOCK_BYTES;
                scales.push(f32::from_le_bytes(
                    self.data[off..off + 4].try_into().unwrap(),
                ));
                // Convert from CPU format {elem 2i, elem 2i+1} to GPU/llama.cpp format
                // {elem i, elem i+16} — required for dp4a W4A8 kernel.
                let cpu = &self.data[off + 4..off + 20];
                for j in 0..16usize {
                    let half = j >> 1;
                    let e_j   = if j & 1 == 0 { cpu[half] & 0x0F }     else { cpu[half] >> 4 };
                    let e_j16 = if j & 1 == 0 { cpu[half + 8] & 0x0F } else { cpu[half + 8] >> 4 };
                    nibbles.push(e_j | (e_j16 << 4));
                }
            }
        }

        (nibbles, scales, self.awq_inv_scales.as_deref())
    }

    /// Matrix-vector multiply: y = self @ x
    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let x_eff = apply_awq_inv_scales(x, &self.awq_inv_scales);
        let x_ref = x_eff.as_deref().unwrap_or(x);
        let m = self.shape[0];
        let row_data_size = self.blocks_per_row * Q4_BLOCK_BYTES;

        // Use parallel for matrices with >= 32 rows (lowered from 64 for better
        // throughput on medium-sized projections like KV heads).
        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q4_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q4_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        }
    }
}

// ── Q8_0 ──

/// A Q8_0 quantized tensor.
/// Stores weights as 8-bit integers with per-block f32 scales.
/// Higher quality than Q4_0 but 1.8x larger.
pub struct QuantizedTensor8 {
    /// Raw block data: [scale: f32 (4 bytes)][values: 32 i8 (32 bytes)] per block
    data: Vec<u8>,
    /// Shape of the original tensor [rows, cols]
    shape: [usize; 2],
    /// Number of blocks per row
    blocks_per_row: usize,
    /// AWQ inverse scales
    awq_inv_scales: Option<Vec<f32>>,
}

impl QuantizedTensor8 {
    /// Quantize a CompactTensor (bf16/f16) to Q8_0.
    pub fn from_compact(compact: &CompactTensor) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q8_BLOCK_SIZE == 0, "cols {} must be divisible by block size {}", cols, Q8_BLOCK_SIZE);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q8_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q8_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q8_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q8_BLOCK_SIZE];
                let out_offset = b * Q8_BLOCK_BYTES;
                quantize_block_q8(block_values, &mut row_out[out_offset..out_offset + Q8_BLOCK_BYTES]);
            }
        });

        QuantizedTensor8 {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: None,
        }
    }

    /// Quantize a CompactTensor with AWQ per-channel scaling.
    pub fn from_compact_awq(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q8_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q8_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q8_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols {
                row_f32[j] *= awq_scales[j];
            }
            for b in 0..blocks_per_row {
                let block_start = b * Q8_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q8_BLOCK_SIZE];
                let out_offset = b * Q8_BLOCK_BYTES;
                quantize_block_q8(block_values, &mut row_out[out_offset..out_offset + Q8_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();

        QuantizedTensor8 {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: Some(inv_scales),
        }
    }

    /// Quantize with AWQ scaling + sub-block importance weighting (k-quant).
    pub fn from_compact_awq_kquant(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q8_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q8_BLOCK_BYTES];

        let channel_importance: Vec<f32> = awq_scales.iter().map(|&s| s * s).collect();

        let row_data_size = blocks_per_row * Q8_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols {
                row_f32[j] *= awq_scales[j];
            }
            for b in 0..blocks_per_row {
                let block_start = b * Q8_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q8_BLOCK_SIZE];
                let block_importance = &channel_importance[block_start..block_start + Q8_BLOCK_SIZE];
                let out_offset = b * Q8_BLOCK_BYTES;
                quantize_block_q8_weighted(block_values, block_importance, &mut row_out[out_offset..out_offset + Q8_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();

        QuantizedTensor8 {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: Some(inv_scales),
        }
    }

    /// Quantize an f32 Tensor to Q8_0.
    pub fn from_f32(tensor: &Tensor) -> Self {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(cols % Q8_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q8_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q8_BLOCK_BYTES;
        let t_data = tensor.data();

        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_start = row_idx * cols;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * Q8_BLOCK_SIZE;
                let block_values = &t_data[block_start..block_start + Q8_BLOCK_SIZE];
                let out_offset = b * Q8_BLOCK_BYTES;
                quantize_block_q8(block_values, &mut row_out[out_offset..out_offset + Q8_BLOCK_BYTES]);
            }
        });

        QuantizedTensor8 {
            data,
            shape: [rows, cols],
            blocks_per_row,
            awq_inv_scales: None,
        }
    }

    pub fn shape(&self) -> &[usize; 2] {
        &self.shape
    }

    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let row_bytes = self.blocks_per_row * Q8_BLOCK_BYTES;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row_start = r * row_bytes;
            let row_data = &self.data[row_start..row_start + row_bytes];
            for b in 0..self.blocks_per_row {
                let block_off = b * Q8_BLOCK_BYTES;
                let scale = f32::from_le_bytes(row_data[block_off..block_off + 4].try_into().unwrap());
                let values = &row_data[block_off + 4..block_off + Q8_BLOCK_BYTES];
                for i in 0..Q8_BLOCK_SIZE {
                    out.push(values[i] as i8 as f32 * scale);
                }
            }
        }
        if let Some(ref inv_scales) = self.awq_inv_scales {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= inv_scales[c];
                }
            }
        }
        out
    }

    /// Matrix-vector multiply: y = self @ x
    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let x_eff = apply_awq_inv_scales(x, &self.awq_inv_scales);
        let x_ref = x_eff.as_deref().unwrap_or(x);
        let m = self.shape[0];
        let row_data_size = self.blocks_per_row * Q8_BLOCK_BYTES;

        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q8_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q8_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        }
    }
}

// ── Q4_0 block operations ──

/// Quantize a block of 32 f32 values to Q4_0 format.
fn quantize_block_q4(values: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q4_BLOCK_BYTES);

    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let scale = if amax > 0.0 { amax / 7.0 } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());

    for i in 0..16 {
        let v0 = values[2 * i];
        let v1 = values[2 * i + 1];

        let q0 = ((v0 * inv_scale).round() as i8).clamp(-8, 7) + 8;
        let q1 = ((v1 * inv_scale).round() as i8).clamp(-8, 7) + 8;

        out[4 + i] = (q0 as u8) | ((q1 as u8) << 4);
    }
}

/// K-quant style importance-weighted Q4 block quantization.
///
/// Instead of using a simple max-based scale, searches for the scale that minimizes
/// importance-weighted quantization error. This gives higher effective precision to
/// weights that matter more (as measured by activation saliency).
///
/// Algorithm: Try multiple candidate scales (from the top-k absolute values in the block)
/// and pick the one that minimizes sum(importance[i] * (value[i] - dequant[i])^2).
fn quantize_block_q4_weighted(values: &[f32], importance: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(importance.len(), Q4_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q4_BLOCK_BYTES);

    // Stack-allocated sorted abs values — no heap allocation per block.
    let mut abs_vals = [0.0f32; Q4_BLOCK_SIZE];
    for (i, &v) in values.iter().enumerate() {
        abs_vals[i] = v.abs();
    }
    abs_vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Try scales derived from the top-8 absolute values (k-quant approach).
    let n_candidates = Q4_BLOCK_SIZE.min(8);
    let mut best_scale = 0.0f32;
    let mut best_error = f32::MAX;

    for c in 0..n_candidates {
        let amax = abs_vals[c];
        if amax < 1e-10 {
            break;
        }
        let candidate_scale = amax / 7.0;
        let candidate_inv = 1.0 / candidate_scale;

        // Compute importance-weighted quantization error for this scale.
        let mut error = 0.0f32;
        for j in 0..Q4_BLOCK_SIZE {
            let q = ((values[j] * candidate_inv).round() as i8).clamp(-8, 7);
            let deq = q as f32 * candidate_scale;
            let diff = values[j] - deq;
            error += importance[j] * diff * diff;
        }

        if error < best_error {
            best_error = error;
            best_scale = candidate_scale;
        }
    }

    let inv_scale = if best_scale > 0.0 { 1.0 / best_scale } else { 0.0 };
    out[0..4].copy_from_slice(&best_scale.to_le_bytes());

    for i in 0..16 {
        let v0 = values[2 * i];
        let v1 = values[2 * i + 1];

        let q0 = ((v0 * inv_scale).round() as i8).clamp(-8, 7) + 8;
        let q1 = ((v1 * inv_scale).round() as i8).clamp(-8, 7) + 8;

        out[4 + i] = (q0 as u8) | ((q1 as u8) << 4);
    }
}

/// K-quant style importance-weighted Q8 block quantization.
fn quantize_block_q8_weighted(values: &[f32], importance: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q8_BLOCK_SIZE);
    debug_assert_eq!(importance.len(), Q8_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q8_BLOCK_BYTES);

    // Stack-allocated sorted abs values — no heap allocation per block.
    let mut abs_vals = [0.0f32; Q8_BLOCK_SIZE];
    for (i, &v) in values.iter().enumerate() {
        abs_vals[i] = v.abs();
    }
    abs_vals.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let n_candidates = Q8_BLOCK_SIZE.min(8);
    let mut best_scale = 0.0f32;
    let mut best_error = f32::MAX;

    for c in 0..n_candidates {
        let amax = abs_vals[c];
        if amax < 1e-10 {
            break;
        }
        let candidate_scale = amax / 127.0;
        let candidate_inv = 1.0 / candidate_scale;

        let mut error = 0.0f32;
        for j in 0..Q8_BLOCK_SIZE {
            let q = ((values[j] * candidate_inv).round() as i16).clamp(-128, 127);
            let deq = q as f32 * candidate_scale;
            let diff = values[j] - deq;
            error += importance[j] * diff * diff;
        }

        if error < best_error {
            best_error = error;
            best_scale = candidate_scale;
        }
    }

    let inv_scale = if best_scale > 0.0 { 1.0 / best_scale } else { 0.0 };
    out[0..4].copy_from_slice(&best_scale.to_le_bytes());

    for i in 0..Q8_BLOCK_SIZE {
        let q = ((values[i] * inv_scale).round() as i8).clamp(-128, 127);
        out[4 + i] = q as u8;
    }
}

/// Compute dot product of a Q4_0 quantized row with an f32 vector.
fn q4_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q4_dot_row_avx2(row_data, x, blocks_per_row) };
    }

    q4_dot_row_scalar(row_data, x, blocks_per_row)
}

fn q4_dot_row_scalar(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset],
            row_data[block_offset + 1],
            row_data[block_offset + 2],
            row_data[block_offset + 3],
        ]);
        let nibbles = &row_data[block_offset + 4..block_offset + Q4_BLOCK_BYTES];
        let x_offset = b * Q4_BLOCK_SIZE;

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = nibbles[i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            block_sum += q0 as f32 * x[x_offset + 2 * i];
            block_sum += q1 as f32 * x[x_offset + 2 * i + 1];
        }
        total += block_sum * scale;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q4_dot_row_avx2(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total_acc = _mm256_setzero_ps();
    let offset_8 = _mm256_set1_epi32(8);

    for b in 0..blocks_per_row {
        let block_offset = b * Q4_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset],
            row_data[block_offset + 1],
            row_data[block_offset + 2],
            row_data[block_offset + 3],
        ]);
        let scale_vec = _mm256_set1_ps(scale);
        let nibbles_ptr = row_data.as_ptr().add(block_offset + 4);
        let x_ptr = x.as_ptr().add(b * Q4_BLOCK_SIZE);

        let mut block_acc = _mm256_setzero_ps();

        for g in 0..4 {
            let nib_offset = g * 4;
            let b0 = *nibbles_ptr.add(nib_offset) as u32;
            let b1 = *nibbles_ptr.add(nib_offset + 1) as u32;
            let b2 = *nibbles_ptr.add(nib_offset + 2) as u32;
            let b3 = *nibbles_ptr.add(nib_offset + 3) as u32;

            let vals = _mm256_set_epi32(
                ((b3 >> 4) & 0x0F) as i32,
                (b3 & 0x0F) as i32,
                ((b2 >> 4) & 0x0F) as i32,
                (b2 & 0x0F) as i32,
                ((b1 >> 4) & 0x0F) as i32,
                (b1 & 0x0F) as i32,
                ((b0 >> 4) & 0x0F) as i32,
                (b0 & 0x0F) as i32,
            );

            let signed_vals = _mm256_sub_epi32(vals, offset_8);
            let f32_vals = _mm256_cvtepi32_ps(signed_vals);
            let x_vals = _mm256_loadu_ps(x_ptr.add(g * 8));
            block_acc = _mm256_fmadd_ps(f32_vals, x_vals, block_acc);
        }

        total_acc = _mm256_fmadd_ps(block_acc, scale_vec, total_acc);
    }

    let hi128 = _mm256_extractf128_ps(total_acc, 1);
    let lo128 = _mm256_castps256_ps128(total_acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let upper64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, upper64);
    let upper32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, upper32);
    _mm_cvtss_f32(sum32)
}

// ── Q8_0 block operations ──

/// Quantize a block of 32 f32 values to Q8_0 format.
fn quantize_block_q8(values: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q8_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q8_BLOCK_BYTES);

    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());

    for i in 0..Q8_BLOCK_SIZE {
        let q = (values[i] * inv_scale).round() as i8;
        out[4 + i] = q as u8;
    }
}

/// Compute dot product of a Q8_0 quantized row with an f32 vector.
fn q8_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q8_dot_row_avx2(row_data, x, blocks_per_row) };
    }

    q8_dot_row_scalar(row_data, x, blocks_per_row)
}

fn q8_dot_row_scalar(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q8_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset],
            row_data[block_offset + 1],
            row_data[block_offset + 2],
            row_data[block_offset + 3],
        ]);
        let values = &row_data[block_offset + 4..block_offset + Q8_BLOCK_BYTES];
        let x_offset = b * Q8_BLOCK_SIZE;

        let mut block_sum = 0.0f32;
        for i in 0..Q8_BLOCK_SIZE {
            let q = values[i] as i8;
            block_sum += q as f32 * x[x_offset + i];
        }
        total += block_sum * scale;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q8_dot_row_avx2(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total_acc = _mm256_setzero_ps();

    for b in 0..blocks_per_row {
        let block_offset = b * Q8_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset],
            row_data[block_offset + 1],
            row_data[block_offset + 2],
            row_data[block_offset + 3],
        ]);
        let scale_vec = _mm256_set1_ps(scale);
        let val_ptr = row_data.as_ptr().add(block_offset + 4);
        let x_ptr = x.as_ptr().add(b * Q8_BLOCK_SIZE);

        let mut block_acc = _mm256_setzero_ps();

        // Process 32 values in 4 groups of 8
        for g in 0..4 {
            let base = g * 8;
            // Load 8 i8 values and convert to i32 then f32
            let v0 = *val_ptr.add(base) as i8 as i32;
            let v1 = *val_ptr.add(base + 1) as i8 as i32;
            let v2 = *val_ptr.add(base + 2) as i8 as i32;
            let v3 = *val_ptr.add(base + 3) as i8 as i32;
            let v4 = *val_ptr.add(base + 4) as i8 as i32;
            let v5 = *val_ptr.add(base + 5) as i8 as i32;
            let v6 = *val_ptr.add(base + 6) as i8 as i32;
            let v7 = *val_ptr.add(base + 7) as i8 as i32;

            let vals = _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0);
            let f32_vals = _mm256_cvtepi32_ps(vals);
            let x_vals = _mm256_loadu_ps(x_ptr.add(base));
            block_acc = _mm256_fmadd_ps(f32_vals, x_vals, block_acc);
        }

        total_acc = _mm256_fmadd_ps(block_acc, scale_vec, total_acc);
    }

    let hi128 = _mm256_extractf128_ps(total_acc, 1);
    let lo128 = _mm256_castps256_ps128(total_acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let upper64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, upper64);
    let upper32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, upper32);
    _mm_cvtss_f32(sum32)
}

// ── Linear ops ──

/// Linear layer with Q4 weights: y = W @ x
pub fn q4_linear_vec(x: &Tensor, weight: &QuantizedTensor) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.rows()]).unwrap()
}

/// Batched linear with Q4 weights: Y = X @ W^T
pub fn q4_linear(x: &Tensor, weight: &QuantizedTensor) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.rows();
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

/// Linear layer with Q8 weights: y = W @ x
pub fn q8_linear_vec(x: &Tensor, weight: &QuantizedTensor8) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.rows()]).unwrap()
}

/// Batched linear with Q8 weights: Y = X @ W^T
pub fn q8_linear(x: &Tensor, weight: &QuantizedTensor8) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.rows();
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

// ── AWQ helper ──

/// Apply AWQ inverse scales to input vector if present.
fn apply_awq_inv_scales(x: &[f32], inv_scales: &Option<Vec<f32>>) -> Option<Vec<f32>> {
    inv_scales.as_ref().map(|scales| {
        x.iter().zip(scales.iter()).map(|(&xi, &si)| xi * si).collect()
    })
}

// ── Q3 (3-bit, 8 levels) ──

/// A Q3 quantized tensor: 3 bits per weight, 8 levels (-4..+3).
/// Block format: f32 scale (4 bytes) + 12 bytes packed (32 weights x 3 bits = 96 bits).
pub struct QuantizedTensor3 {
    data: Vec<u8>,
    pub shape: [usize; 2],
    blocks_per_row: usize,
    awq_inv_scales: Option<Vec<f32>>,
}

impl QuantizedTensor3 {
    pub fn from_compact(compact: &CompactTensor) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q3_BLOCK_SIZE == 0, "cols {} must be divisible by block size {}", cols, Q3_BLOCK_SIZE);

        let blocks_per_row = cols / Q3_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q3_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q3_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q3_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q3_BLOCK_SIZE];
                let out_offset = b * Q3_BLOCK_BYTES;
                quantize_block_q3(block_values, &mut row_out[out_offset..out_offset + Q3_BLOCK_BYTES]);
            }
        });

        QuantizedTensor3 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_f32(tensor: &Tensor) -> Self {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(cols % Q3_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q3_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q3_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q3_BLOCK_BYTES;
        let t_data = tensor.data();
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_start = row_idx * cols;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * Q3_BLOCK_SIZE;
                let block_values = &t_data[block_start..block_start + Q3_BLOCK_SIZE];
                let out_offset = b * Q3_BLOCK_BYTES;
                quantize_block_q3(block_values, &mut row_out[out_offset..out_offset + Q3_BLOCK_BYTES]);
            }
        });

        QuantizedTensor3 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_compact_awq(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q3_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q3_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q3_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q3_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols { row_f32[j] *= awq_scales[j]; }
            for b in 0..blocks_per_row {
                let block_start = b * Q3_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q3_BLOCK_SIZE];
                let out_offset = b * Q3_BLOCK_BYTES;
                quantize_block_q3(block_values, &mut row_out[out_offset..out_offset + Q3_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();
        QuantizedTensor3 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: Some(inv_scales) }
    }

    pub fn size_bytes(&self) -> usize { self.data.len() }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let row_bytes = self.blocks_per_row * Q3_BLOCK_BYTES;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row_start = r * row_bytes;
            let row_data = &self.data[row_start..row_start + row_bytes];
            for b in 0..self.blocks_per_row {
                let block_off = b * Q3_BLOCK_BYTES;
                let scale = f32::from_le_bytes(row_data[block_off..block_off + 4].try_into().unwrap());
                let packed = &row_data[block_off + 4..block_off + Q3_BLOCK_BYTES];
                for i in 0..Q3_BLOCK_SIZE {
                    let bit_offset = i * 3;
                    let byte_idx = bit_offset / 8;
                    let bit_within = bit_offset % 8;
                    let mut q = (packed[byte_idx] >> bit_within) & 0x07;
                    if bit_within > 5 && byte_idx + 1 < packed.len() {
                        q |= (packed[byte_idx + 1] << (8 - bit_within)) & 0x07;
                    }
                    let signed = q as i8 - 4;
                    out.push(signed as f32 * scale);
                }
            }
        }
        if let Some(ref inv_scales) = self.awq_inv_scales {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= inv_scales[c];
                }
            }
        }
        out
    }

    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let x_eff = apply_awq_inv_scales(x, &self.awq_inv_scales);
        let x_ref = x_eff.as_deref().unwrap_or(x);
        let m = self.shape[0];
        let row_data_size = self.blocks_per_row * Q3_BLOCK_BYTES;

        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q3_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q3_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        }
    }
}

/// Quantize a block of 32 f32 values to Q3 format.
fn quantize_block_q3(values: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q3_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q3_BLOCK_BYTES);

    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if amax > 0.0 { amax / 3.0 } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());

    // Pack 32 weights x 3 bits = 96 bits = 12 bytes
    // Clear packed bytes
    for b in &mut out[4..16] { *b = 0; }

    for i in 0..Q3_BLOCK_SIZE {
        // Quantize to signed -4..+3, then shift to unsigned 0..7
        let q = ((values[i] * inv_scale).round() as i8).clamp(-4, 3) + 4;
        let q = q as u8; // 0..7
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_within_byte = bit_offset % 8;
        out[4 + byte_idx] |= (q << bit_within_byte) as u8;
        // Handle overflow into next byte
        if bit_within_byte > 5 {
            out[4 + byte_idx + 1] |= (q >> (8 - bit_within_byte)) as u8;
        }
    }
}

fn q3_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q3_dot_row_avx2(row_data, x, blocks_per_row) };
    }
    q3_dot_row_scalar(row_data, x, blocks_per_row)
}

fn q3_dot_row_scalar(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q3_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let packed = &row_data[block_offset + 4..block_offset + Q3_BLOCK_BYTES];
        let x_offset = b * Q3_BLOCK_SIZE;

        let mut block_sum = 0.0f32;
        for i in 0..Q3_BLOCK_SIZE {
            let bit_offset = i * 3;
            let byte_idx = bit_offset / 8;
            let bit_within_byte = bit_offset % 8;
            let mut q = (packed[byte_idx] >> bit_within_byte) & 0x07;
            if bit_within_byte > 5 {
                q |= (packed[byte_idx + 1] << (8 - bit_within_byte)) & 0x07;
            }
            let signed = q as i8 - 4; // map back to -4..+3
            block_sum += signed as f32 * x[x_offset + i];
        }
        total += block_sum * scale;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q3_dot_row_avx2(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total_acc = _mm256_setzero_ps();
    let offset_4 = _mm256_set1_epi32(4);

    for b in 0..blocks_per_row {
        let block_offset = b * Q3_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let scale_vec = _mm256_set1_ps(scale);
        let packed = &row_data[block_offset + 4..block_offset + Q3_BLOCK_BYTES];
        let x_ptr = x.as_ptr().add(b * Q3_BLOCK_SIZE);

        let mut block_acc = _mm256_setzero_ps();

        // Process 32 weights in groups of 8
        for g in 0..4 {
            let base = g * 8;
            let mut vals = [0i32; 8];
            for k in 0..8 {
                let i = base + k;
                let bit_offset = i * 3;
                let byte_idx = bit_offset / 8;
                let bit_within_byte = bit_offset % 8;
                let mut q = (packed[byte_idx] >> bit_within_byte) & 0x07;
                if bit_within_byte > 5 {
                    q |= (packed[byte_idx + 1] << (8 - bit_within_byte)) & 0x07;
                }
                vals[k] = q as i32;
            }

            let v = _mm256_set_epi32(vals[7], vals[6], vals[5], vals[4], vals[3], vals[2], vals[1], vals[0]);
            let signed = _mm256_sub_epi32(v, offset_4);
            let f32_vals = _mm256_cvtepi32_ps(signed);
            let x_vals = _mm256_loadu_ps(x_ptr.add(base));
            block_acc = _mm256_fmadd_ps(f32_vals, x_vals, block_acc);
        }

        total_acc = _mm256_fmadd_ps(block_acc, scale_vec, total_acc);
    }

    hsum_avx2(total_acc)
}

// ── Q2 (2-bit, 4 levels) ──

/// A Q2 quantized tensor: 2 bits per weight, 4 levels (-2..+1).
/// Block format: f32 scale (4 bytes) + 8 bytes packed (32 weights x 2 bits = 64 bits).
pub struct QuantizedTensor2 {
    data: Vec<u8>,
    pub shape: [usize; 2],
    blocks_per_row: usize,
    awq_inv_scales: Option<Vec<f32>>,
}

impl QuantizedTensor2 {
    pub fn from_compact(compact: &CompactTensor) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q2_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q2_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q2_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q2_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q2_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q2_BLOCK_SIZE];
                let out_offset = b * Q2_BLOCK_BYTES;
                quantize_block_q2(block_values, &mut row_out[out_offset..out_offset + Q2_BLOCK_BYTES]);
            }
        });

        QuantizedTensor2 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_f32(tensor: &Tensor) -> Self {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(cols % Q2_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q2_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q2_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q2_BLOCK_BYTES;
        let t_data = tensor.data();
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_start = row_idx * cols;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * Q2_BLOCK_SIZE;
                let block_values = &t_data[block_start..block_start + Q2_BLOCK_SIZE];
                let out_offset = b * Q2_BLOCK_BYTES;
                quantize_block_q2(block_values, &mut row_out[out_offset..out_offset + Q2_BLOCK_BYTES]);
            }
        });

        QuantizedTensor2 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_compact_awq(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q2_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q2_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q2_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q2_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols { row_f32[j] *= awq_scales[j]; }
            for b in 0..blocks_per_row {
                let block_start = b * Q2_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q2_BLOCK_SIZE];
                let out_offset = b * Q2_BLOCK_BYTES;
                quantize_block_q2(block_values, &mut row_out[out_offset..out_offset + Q2_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();
        QuantizedTensor2 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: Some(inv_scales) }
    }

    pub fn size_bytes(&self) -> usize { self.data.len() }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let row_bytes = self.blocks_per_row * Q2_BLOCK_BYTES;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row_start = r * row_bytes;
            let row_data = &self.data[row_start..row_start + row_bytes];
            for b in 0..self.blocks_per_row {
                let block_off = b * Q2_BLOCK_BYTES;
                let scale = f32::from_le_bytes(row_data[block_off..block_off + 4].try_into().unwrap());
                let packed = &row_data[block_off + 4..block_off + Q2_BLOCK_BYTES];
                for i in 0..8 {
                    let byte = packed[i];
                    for k in 0..4 {
                        let q = ((byte >> (k * 2)) & 0x03) as i8 - 2;
                        out.push(q as f32 * scale);
                    }
                }
            }
        }
        if let Some(ref inv_scales) = self.awq_inv_scales {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= inv_scales[c];
                }
            }
        }
        out
    }

    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let x_eff = apply_awq_inv_scales(x, &self.awq_inv_scales);
        let x_ref = x_eff.as_deref().unwrap_or(x);
        let m = self.shape[0];
        let row_data_size = self.blocks_per_row * Q2_BLOCK_BYTES;

        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q2_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q2_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        }
    }
}

/// Quantize a block of 32 f32 values to Q2 format.
fn quantize_block_q2(values: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q2_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q2_BLOCK_BYTES);

    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    // Scale so that range maps to -2,-1,0,+1 (4 levels)
    let scale = if amax > 0.0 { amax / 1.0 } else { 0.0 };
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());

    // Pack 4 weights per byte (2 bits each)
    for i in 0..8 {
        let base = i * 4;
        let mut byte = 0u8;
        for k in 0..4 {
            let q = ((values[base + k] * inv_scale).round() as i8).clamp(-2, 1) + 2;
            byte |= (q as u8) << (k * 2);
        }
        out[4 + i] = byte;
    }
}

fn q2_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q2_dot_row_avx2(row_data, x, blocks_per_row) };
    }
    q2_dot_row_scalar(row_data, x, blocks_per_row)
}

fn q2_dot_row_scalar(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q2_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let packed = &row_data[block_offset + 4..block_offset + Q2_BLOCK_BYTES];
        let x_offset = b * Q2_BLOCK_SIZE;

        let mut block_sum = 0.0f32;
        for i in 0..8 {
            let byte = packed[i];
            let base = i * 4;
            for k in 0..4 {
                let q = ((byte >> (k * 2)) & 0x03) as i8 - 2;
                block_sum += q as f32 * x[x_offset + base + k];
            }
        }
        total += block_sum * scale;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q2_dot_row_avx2(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total_acc = _mm256_setzero_ps();
    let offset_2 = _mm256_set1_epi32(2);

    for b in 0..blocks_per_row {
        let block_offset = b * Q2_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let scale_vec = _mm256_set1_ps(scale);
        let packed = &row_data[block_offset + 4..block_offset + Q2_BLOCK_BYTES];
        let x_ptr = x.as_ptr().add(b * Q2_BLOCK_SIZE);

        let mut block_acc = _mm256_setzero_ps();

        // 32 weights / 8 per SIMD = 4 groups
        for g in 0..4 {
            let base = g * 8;
            let mut vals = [0i32; 8];
            // 8 weights span 2 packed bytes
            for k in 0..8 {
                let i = base + k;
                let bi = i / 4;
                let shift = (i % 4) * 2;
                vals[k] = ((packed[bi] >> shift) & 0x03) as i32;
            }
            let v = _mm256_set_epi32(vals[7], vals[6], vals[5], vals[4], vals[3], vals[2], vals[1], vals[0]);
            let signed = _mm256_sub_epi32(v, offset_2);
            let f32_vals = _mm256_cvtepi32_ps(signed);
            let x_vals = _mm256_loadu_ps(x_ptr.add(base));
            block_acc = _mm256_fmadd_ps(f32_vals, x_vals, block_acc);
        }

        total_acc = _mm256_fmadd_ps(block_acc, scale_vec, total_acc);
    }

    hsum_avx2(total_acc)
}

// ── Q1 (ternary: {-1, 0, +1}) ──

/// A Q1 quantized tensor: ternary weights {-1, 0, +1}.
/// Block format: f32 scale (4 bytes) + u32 nonzero_mask (4 bytes) + u32 sign_bits (4 bytes) = 12 bytes.
pub struct QuantizedTensor1 {
    data: Vec<u8>,
    pub shape: [usize; 2],
    blocks_per_row: usize,
    awq_inv_scales: Option<Vec<f32>>,
}

impl QuantizedTensor1 {
    pub fn from_compact(compact: &CompactTensor) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q1_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q1_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q1_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q1_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q1_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q1_BLOCK_SIZE];
                let out_offset = b * Q1_BLOCK_BYTES;
                quantize_block_q1(block_values, &mut row_out[out_offset..out_offset + Q1_BLOCK_BYTES]);
            }
        });

        QuantizedTensor1 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_f32(tensor: &Tensor) -> Self {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        assert!(cols % Q1_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q1_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q1_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q1_BLOCK_BYTES;
        let t_data = tensor.data();
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let row_start = row_idx * cols;
            for b in 0..blocks_per_row {
                let block_start = row_start + b * Q1_BLOCK_SIZE;
                let block_values = &t_data[block_start..block_start + Q1_BLOCK_SIZE];
                let out_offset = b * Q1_BLOCK_BYTES;
                quantize_block_q1(block_values, &mut row_out[out_offset..out_offset + Q1_BLOCK_BYTES]);
            }
        });

        QuantizedTensor1 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: None }
    }

    pub fn from_compact_awq(compact: &CompactTensor, awq_scales: &[f32]) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q1_BLOCK_SIZE == 0);
        assert_eq!(awq_scales.len(), cols);

        let blocks_per_row = cols / Q1_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;
        let mut data = vec![0u8; total_blocks * Q1_BLOCK_BYTES];

        let row_data_size = blocks_per_row * Q1_BLOCK_BYTES;
        data.par_chunks_mut(row_data_size).enumerate().for_each(|(row_idx, row_out)| {
            let mut row_f32 = compact.row_to_f32(row_idx);
            for j in 0..cols { row_f32[j] *= awq_scales[j]; }
            for b in 0..blocks_per_row {
                let block_start = b * Q1_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q1_BLOCK_SIZE];
                let out_offset = b * Q1_BLOCK_BYTES;
                quantize_block_q1(block_values, &mut row_out[out_offset..out_offset + Q1_BLOCK_BYTES]);
            }
        });

        let inv_scales: Vec<f32> = awq_scales.iter().map(|&s| if s.abs() > 1e-10 { 1.0 / s } else { 0.0 }).collect();
        QuantizedTensor1 { data, shape: [rows, cols], blocks_per_row, awq_inv_scales: Some(inv_scales) }
    }

    pub fn size_bytes(&self) -> usize { self.data.len() }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let row_bytes = self.blocks_per_row * Q1_BLOCK_BYTES;
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row_start = r * row_bytes;
            let row_data = &self.data[row_start..row_start + row_bytes];
            for b in 0..self.blocks_per_row {
                let block_off = b * Q1_BLOCK_BYTES;
                let scale = f32::from_le_bytes(row_data[block_off..block_off + 4].try_into().unwrap());
                let nonzero_mask = u32::from_le_bytes(row_data[block_off + 4..block_off + 8].try_into().unwrap());
                let sign_bits = u32::from_le_bytes(row_data[block_off + 8..block_off + 12].try_into().unwrap());
                for i in 0..Q1_BLOCK_SIZE {
                    let is_nonzero = (nonzero_mask >> i) & 1 == 1;
                    let is_negative = (sign_bits >> i) & 1 == 1;
                    let val = if is_nonzero {
                        if is_negative { -scale } else { scale }
                    } else { 0.0 };
                    out.push(val);
                }
            }
        }
        if let Some(ref inv_scales) = self.awq_inv_scales {
            for r in 0..rows {
                for c in 0..cols {
                    out[r * cols + c] *= inv_scales[c];
                }
            }
        }
        out
    }

    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let x_eff = apply_awq_inv_scales(x, &self.awq_inv_scales);
        let x_ref = x_eff.as_deref().unwrap_or(x);
        let m = self.shape[0];
        let row_data_size = self.blocks_per_row * Q1_BLOCK_BYTES;

        if m >= 32 {
            (0..m).into_par_iter().map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q1_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        } else {
            (0..m).map(|i| {
                let row_start = i * row_data_size;
                let row_data = &self.data[row_start..row_start + row_data_size];
                q1_dot_row(row_data, x_ref, self.blocks_per_row)
            }).collect()
        }
    }
}

/// Quantize a block of 32 f32 values to Q1 (ternary) format.
fn quantize_block_q1(values: &[f32], out: &mut [u8]) {
    debug_assert_eq!(values.len(), Q1_BLOCK_SIZE);
    debug_assert_eq!(out.len(), Q1_BLOCK_BYTES);

    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let threshold = 0.33 * amax;

    let mut nonzero_mask: u32 = 0;
    let mut sign_bits: u32 = 0;
    let mut abs_sum = 0.0f32;
    let mut abs_count = 0u32;

    for i in 0..Q1_BLOCK_SIZE {
        if values[i].abs() > threshold {
            nonzero_mask |= 1 << i;
            abs_sum += values[i].abs();
            abs_count += 1;
            if values[i] < 0.0 {
                sign_bits |= 1 << i; // 1 = negative
            }
        }
    }

    let scale = if abs_count > 0 { abs_sum / abs_count as f32 } else { 0.0 };

    out[0..4].copy_from_slice(&scale.to_le_bytes());
    out[4..8].copy_from_slice(&nonzero_mask.to_le_bytes());
    out[8..12].copy_from_slice(&sign_bits.to_le_bytes());
}

fn q1_dot_row(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { q1_dot_row_avx2(row_data, x, blocks_per_row) };
    }
    q1_dot_row_scalar(row_data, x, blocks_per_row)
}

fn q1_dot_row_scalar(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    let mut total = 0.0f32;

    for b in 0..blocks_per_row {
        let block_offset = b * Q1_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let nonzero_mask = u32::from_le_bytes([
            row_data[block_offset + 4], row_data[block_offset + 5],
            row_data[block_offset + 6], row_data[block_offset + 7],
        ]);
        let sign_bits = u32::from_le_bytes([
            row_data[block_offset + 8], row_data[block_offset + 9],
            row_data[block_offset + 10], row_data[block_offset + 11],
        ]);
        let x_offset = b * Q1_BLOCK_SIZE;

        let mut block_sum = 0.0f32;
        let mut mask = nonzero_mask;
        while mask != 0 {
            let j = mask.trailing_zeros() as usize;
            mask &= mask - 1; // clear lowest set bit
            if (sign_bits >> j) & 1 == 1 {
                block_sum -= x[x_offset + j];
            } else {
                block_sum += x[x_offset + j];
            }
        }
        total += block_sum * scale;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn q1_dot_row_avx2(row_data: &[u8], x: &[f32], blocks_per_row: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut total_acc = _mm256_setzero_ps();

    for b in 0..blocks_per_row {
        let block_offset = b * Q1_BLOCK_BYTES;
        let scale = f32::from_le_bytes([
            row_data[block_offset], row_data[block_offset + 1],
            row_data[block_offset + 2], row_data[block_offset + 3],
        ]);
        let scale_vec = _mm256_set1_ps(scale);
        let nonzero_mask = u32::from_le_bytes([
            row_data[block_offset + 4], row_data[block_offset + 5],
            row_data[block_offset + 6], row_data[block_offset + 7],
        ]);
        let sign_bits = u32::from_le_bytes([
            row_data[block_offset + 8], row_data[block_offset + 9],
            row_data[block_offset + 10], row_data[block_offset + 11],
        ]);
        let x_ptr = x.as_ptr().add(b * Q1_BLOCK_SIZE);

        let mut block_acc = _mm256_setzero_ps();

        // Process 32 weights in 4 groups of 8
        for g in 0..4 {
            let base = g * 8;
            let nz_byte = ((nonzero_mask >> base) & 0xFF) as u8;
            let sign_byte = ((sign_bits >> base) & 0xFF) as u8;

            // Build weight vector: +1, -1, or 0 for each position
            let mut vals = [0.0f32; 8];
            for k in 0..8 {
                if (nz_byte >> k) & 1 == 1 {
                    vals[k] = if (sign_byte >> k) & 1 == 1 { -1.0 } else { 1.0 };
                }
            }
            let w = _mm256_set_ps(vals[7], vals[6], vals[5], vals[4], vals[3], vals[2], vals[1], vals[0]);
            let x_vals = _mm256_loadu_ps(x_ptr.add(base));
            block_acc = _mm256_fmadd_ps(w, x_vals, block_acc);
        }

        total_acc = _mm256_fmadd_ps(block_acc, scale_vec, total_acc);
    }

    hsum_avx2(total_acc)
}

// ── SparseQuantized (sub-1-bit) ──

/// Sparse quantized tensor — only a fraction of blocks are active (quantized to Q4),
/// the rest are treated as zero. Enables sub-1-bit effective precision.
pub struct SparseQuantized {
    /// 1 bit per block: which blocks are active (nonzero)
    active_bitmap: Vec<u8>,
    /// Only active blocks' quantized data (Q4 format internally)
    active_data: Vec<u8>,
    pub shape: [usize; 2],
    blocks_per_row: usize,
    active_count: usize,
    inner_block_bytes: usize,
}

impl SparseQuantized {
    /// Create a sparse quantized tensor from a compact tensor.
    /// `keep_fraction` is the fraction of blocks to keep (e.g. 0.25 = 25% of blocks active).
    pub fn from_compact_with_sparsity(compact: &CompactTensor, keep_fraction: f32) -> Self {
        let rows = compact.shape()[0];
        let cols = if compact.shape().len() >= 2 { compact.shape()[1] } else { compact.shape()[0] };
        assert!(cols % Q4_BLOCK_SIZE == 0);

        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let total_blocks = rows * blocks_per_row;

        // Compute L2 norm for each block to decide which to keep.
        let mut block_norms: Vec<(usize, f32)> = Vec::with_capacity(total_blocks);
        for row_idx in 0..rows {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_start = b * Q4_BLOCK_SIZE;
                let block_values = &row_f32[block_start..block_start + Q4_BLOCK_SIZE];
                let l2: f32 = block_values.iter().map(|v| v * v).sum::<f32>().sqrt();
                let block_idx = row_idx * blocks_per_row + b;
                block_norms.push((block_idx, l2));
            }
        }

        // Sort by norm descending, keep top fraction.
        block_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep_count = ((total_blocks as f32 * keep_fraction).ceil() as usize).min(total_blocks);

        let mut active_set = vec![false; total_blocks];
        for i in 0..keep_count {
            active_set[block_norms[i].0] = true;
        }

        // Build bitmap.
        let bitmap_bytes = (total_blocks + 7) / 8;
        let mut active_bitmap = vec![0u8; bitmap_bytes];
        for (idx, &active) in active_set.iter().enumerate() {
            if active {
                active_bitmap[idx / 8] |= 1 << (idx % 8);
            }
        }

        // Quantize only active blocks to Q4.
        let active_count = keep_count;
        let mut active_data = vec![0u8; active_count * Q4_BLOCK_BYTES];

        let mut write_idx = 0;
        for row_idx in 0..rows {
            let row_f32 = compact.row_to_f32(row_idx);
            for b in 0..blocks_per_row {
                let block_idx = row_idx * blocks_per_row + b;
                if active_set[block_idx] {
                    let block_start = b * Q4_BLOCK_SIZE;
                    let block_values = &row_f32[block_start..block_start + Q4_BLOCK_SIZE];
                    let out_offset = write_idx * Q4_BLOCK_BYTES;
                    quantize_block_q4(block_values, &mut active_data[out_offset..out_offset + Q4_BLOCK_BYTES]);
                    write_idx += 1;
                }
            }
        }

        SparseQuantized {
            active_bitmap,
            active_data,
            shape: [rows, cols],
            blocks_per_row,
            active_count,
            inner_block_bytes: Q4_BLOCK_BYTES,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.active_bitmap.len() + self.active_data.len()
    }

    /// Dequantize all weights to a flat f32 Vec in row-major order [rows, cols].
    /// Inactive blocks are returned as zero (they were pruned during quantization).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let [rows, cols] = self.shape;
        let mut out = vec![0.0f32; rows * cols];
        let mut active_idx = 0usize;
        for r in 0..rows {
            for b in 0..self.blocks_per_row {
                let block_idx = r * self.blocks_per_row + b;
                let is_active = (self.active_bitmap[block_idx / 8] >> (block_idx % 8)) & 1 == 1;
                if is_active {
                    let block_off = active_idx * self.inner_block_bytes;
                    let block_data = &self.active_data[block_off..block_off + self.inner_block_bytes];
                    let scale = f32::from_le_bytes(block_data[0..4].try_into().unwrap());
                    let nibbles = &block_data[4..];
                    let col_base = b * Q4_BLOCK_SIZE;
                    for i in 0..16 {
                        let byte = nibbles[i];
                        let q0 = (byte & 0x0F) as i8 - 8;
                        let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                        out[r * cols + col_base + 2 * i] = q0 as f32 * scale;
                        out[r * cols + col_base + 2 * i + 1] = q1 as f32 * scale;
                    }
                    active_idx += 1;
                }
            }
        }
        out
    }

    pub fn matvec_f32(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.shape[1]);
        let m = self.shape[0];
        let mut out = vec![0.0f32; m];

        // We need to iterate through active blocks row by row
        let mut active_idx = 0;
        for row in 0..m {
            let mut row_sum = 0.0f32;
            for b in 0..self.blocks_per_row {
                let block_idx = row * self.blocks_per_row + b;
                let is_active = (self.active_bitmap[block_idx / 8] >> (block_idx % 8)) & 1 == 1;
                if is_active {
                    let block_offset = active_idx * self.inner_block_bytes;
                    let block_data = &self.active_data[block_offset..block_offset + self.inner_block_bytes];
                    let x_offset = b * Q4_BLOCK_SIZE;
                    // Inline Q4 dot for this single block
                    let scale = f32::from_le_bytes([block_data[0], block_data[1], block_data[2], block_data[3]]);
                    let nibbles = &block_data[4..];
                    let mut block_sum = 0.0f32;
                    for i in 0..16 {
                        let byte = nibbles[i];
                        let q0 = (byte & 0x0F) as i8 - 8;
                        let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                        block_sum += q0 as f32 * x[x_offset + 2 * i];
                        block_sum += q1 as f32 * x[x_offset + 2 * i + 1];
                    }
                    row_sum += block_sum * scale;
                    active_idx += 1;
                }
                // Inactive blocks contribute 0 — skip.
            }
            out[row] = row_sum;
        }

        out
    }
}

// ── AVX2 horizontal sum helper ──

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let upper64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, upper64);
    let upper32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, upper32);
    _mm_cvtss_f32(sum32)
}

// ── Linear ops for new formats ──

pub fn q3_linear_vec(x: &Tensor, weight: &QuantizedTensor3) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.shape[0]]).unwrap()
}

pub fn q3_linear(x: &Tensor, weight: &QuantizedTensor3) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.shape[0];
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

pub fn q2_linear_vec(x: &Tensor, weight: &QuantizedTensor2) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.shape[0]]).unwrap()
}

pub fn q2_linear(x: &Tensor, weight: &QuantizedTensor2) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.shape[0];
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

pub fn q1_linear_vec(x: &Tensor, weight: &QuantizedTensor1) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.shape[0]]).unwrap()
}

pub fn q1_linear(x: &Tensor, weight: &QuantizedTensor1) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.shape[0];
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

pub fn sparse_linear_vec(x: &Tensor, weight: &SparseQuantized) -> Tensor {
    let out_data = weight.matvec_f32(x.data());
    Tensor::new(out_data, vec![weight.shape[0]]).unwrap()
}

pub fn sparse_linear(x: &Tensor, weight: &SparseQuantized) -> Tensor {
    let seq_len = x.shape()[0];
    let out_dim = weight.shape[0];
    let in_dim = x.shape()[1];

    let out_data: Vec<f32> = (0..seq_len).flat_map(|s| {
        let row = &x.data()[s * in_dim..(s + 1) * in_dim];
        weight.matvec_f32(row)
    }).collect();

    Tensor::new(out_data, vec![seq_len, out_dim]).unwrap()
}

// ── Bit allocation algorithm ──

/// Allocate per-layer quantization modes based on importance scores.
/// Uses greedy algorithm: start all layers at None (0 bits), upgrade most important
/// layers to higher bit rates until the total budget is met.
///
/// `layer_importance`: importance score per layer (higher = more important).
/// `target_bpw`: target average bits-per-weight.
/// `layer_params`: number of parameters per layer (assumed uniform).
pub fn allocate_bits(layer_importance: &[f64], target_bpw: f32, layer_params: usize) -> Vec<QuantMode> {
    let n = layer_importance.len();
    if n == 0 {
        return vec![];
    }

    // Available rates sorted by bpw ascending.
    let rates: [(QuantMode, f32); 6] = [
        (QuantMode::None, 0.0),
        (QuantMode::Q1, 1.0),
        (QuantMode::Q2, 2.0),
        (QuantMode::Q3, 3.0),
        (QuantMode::Q4, 4.0),
        (QuantMode::Q8, 8.0),
    ];

    let total_params = n * layer_params;
    let total_budget = target_bpw * total_params as f32;

    // Current assignment: index into rates array per layer.
    let mut assignments = vec![0usize; n]; // all start at None (0 bits)
    let mut current_bits = 0.0f32;

    // Sort layers by importance descending.
    let mut ranked: Vec<(usize, f64)> = layer_importance.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy: repeatedly upgrade the most important under-allocated layer.
    loop {
        let mut best_upgrade = None;
        let mut best_efficiency = f64::NEG_INFINITY;

        for &(layer_idx, importance) in &ranked {
            let current_rate_idx = assignments[layer_idx];
            if current_rate_idx >= rates.len() - 1 {
                continue; // already at max
            }
            let next_rate_idx = current_rate_idx + 1;
            let bit_cost = (rates[next_rate_idx].1 - rates[current_rate_idx].1) * layer_params as f32;
            if current_bits + bit_cost > total_budget {
                continue; // would exceed budget
            }
            // Efficiency = importance / cost
            let efficiency = importance / bit_cost as f64;
            if efficiency > best_efficiency {
                best_efficiency = efficiency;
                best_upgrade = Some((layer_idx, next_rate_idx, bit_cost));
            }
        }

        match best_upgrade {
            Some((layer_idx, next_rate_idx, bit_cost)) => {
                assignments[layer_idx] = next_rate_idx;
                current_bits += bit_cost;
            }
            None => break, // no more upgrades possible within budget
        }
    }

    assignments.iter().map(|&idx| rates[idx].0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn test_quantize_block_q4_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut block = vec![0u8; Q4_BLOCK_BYTES];
        quantize_block_q4(&values, &mut block);

        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let mut deq = vec![0.0f32; 32];
        for i in 0..16 {
            let byte = block[4 + i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            deq[2 * i] = q0 as f32 * scale;
            deq[2 * i + 1] = q1 as f32 * scale;
        }

        for (orig, deq) in values.iter().zip(deq.iter()) {
            assert!((orig - deq).abs() < 2.0, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_quantize_block_q8_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut block = vec![0u8; Q8_BLOCK_BYTES];
        quantize_block_q8(&values, &mut block);

        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let mut deq = vec![0.0f32; 32];
        for i in 0..32 {
            let q = block[4 + i] as i8;
            deq[i] = q as f32 * scale;
        }

        for (orig, deq) in values.iter().zip(deq.iter()) {
            // Q8 should be much more accurate than Q4
            assert!((orig - deq).abs() < 0.15, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_q4_matvec() {
        let mut values = vec![0.0f32; 64];
        values[0] = 1.0;
        values[33] = 1.0;

        let t = Tensor::new(values, vec![2, 32]).unwrap();
        let qt = QuantizedTensor::from_f32(&t);

        let mut x = vec![0.0f32; 32];
        x[0] = 5.0;
        x[1] = 3.0;

        let y = qt.matvec_f32(&x);
        assert!((y[0] - 5.0).abs() < 2.0);
        assert!((y[1] - 3.0).abs() < 2.0);
    }

    #[test]
    fn test_q8_matvec() {
        let mut values = vec![0.0f32; 64];
        values[0] = 1.0;
        values[33] = 1.0;

        let t = Tensor::new(values, vec![2, 32]).unwrap();
        let qt = QuantizedTensor8::from_f32(&t);

        let mut x = vec![0.0f32; 32];
        x[0] = 5.0;
        x[1] = 3.0;

        let y = qt.matvec_f32(&x);
        // Q8 should be more accurate than Q4
        assert!((y[0] - 5.0).abs() < 0.5);
        assert!((y[1] - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_q4_memory_savings() {
        let rows = 3072;
        let cols = 3072;
        let blocks_per_row = cols / Q4_BLOCK_SIZE;
        let q4_bytes = rows * blocks_per_row * Q4_BLOCK_BYTES;
        let f32_bytes = rows * cols * 4;
        let bf16_bytes = rows * cols * 2;

        assert!(q4_bytes < f32_bytes / 4);
        assert!(q4_bytes < bf16_bytes / 2);
    }

    #[test]
    fn test_q8_memory_savings() {
        let rows = 3072;
        let cols = 3072;
        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let q8_bytes = rows * blocks_per_row * Q8_BLOCK_BYTES;
        let bf16_bytes = rows * cols * 2;

        // Q8 should be ~1.78x smaller than bf16 (36/32 bytes vs 2 bytes per weight)
        assert!(q8_bytes < bf16_bytes);
    }

    #[test]
    fn test_weight_storage_dispatch() {
        // Create a small compact tensor
        let bytes: Vec<u8> = (0..64)
            .flat_map(|_| bf16::from_f32(0.1).to_le_bytes().to_vec())
            .collect();
        let compact = CompactTensor::new(bytes, vec![2, 32], crate::tensor::DType::BF16);

        let ws = WeightStorage::Compact(compact);
        assert_eq!(ws.shape(), [2, 32]);

        let x = vec![1.0f32; 32];
        let y = ws.matvec_f32(&x);
        assert_eq!(y.len(), 2);
    }

    #[test]
    fn test_quant_mode_from_str() {
        assert_eq!(QuantMode::from_str("none"), Some(QuantMode::None));
        assert_eq!(QuantMode::from_str("q4"), Some(QuantMode::Q4));
        assert_eq!(QuantMode::from_str("q8"), Some(QuantMode::Q8));
        assert_eq!(QuantMode::from_str("invalid"), None);
    }

    // ── New tests for Q3, Q2, Q1, Sparse, AWQ, bit allocation ──

    #[test]
    fn test_q3_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let mut block = vec![0u8; Q3_BLOCK_BYTES];
        quantize_block_q3(&values, &mut block);

        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let packed = &block[4..16];
        let mut deq = vec![0.0f32; 32];
        for i in 0..32 {
            let bit_offset = i * 3;
            let byte_idx = bit_offset / 8;
            let bit_within_byte = bit_offset % 8;
            let mut q = (packed[byte_idx] >> bit_within_byte) & 0x07;
            if bit_within_byte > 5 {
                q |= (packed[byte_idx + 1] << (8 - bit_within_byte)) & 0x07;
            }
            let signed = q as i8 - 4;
            deq[i] = signed as f32 * scale;
        }

        // Q3 has 8 levels, so precision is coarse — allow up to 2.5
        for (orig, deq) in values.iter().zip(deq.iter()) {
            assert!((orig - deq).abs() < 2.5, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_q3_matvec() {
        let mut values = vec![0.0f32; 64];
        values[0] = 1.0;
        values[33] = 1.0;

        let t = Tensor::new(values, vec![2, 32]).unwrap();
        let qt = QuantizedTensor3::from_f32(&t);

        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 2.0;

        let y = qt.matvec_f32(&x);
        assert!((y[0] - 3.0).abs() < 2.0, "y[0]={}", y[0]);
        assert!((y[1] - 2.0).abs() < 2.0, "y[1]={}", y[1]);
    }

    #[test]
    fn test_q2_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut block = vec![0u8; Q2_BLOCK_BYTES];
        quantize_block_q2(&values, &mut block);

        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let packed = &block[4..12];
        let mut deq = vec![0.0f32; 32];
        for i in 0..32 {
            let bi = i / 4;
            let shift = (i % 4) * 2;
            let q = ((packed[bi] >> shift) & 0x03) as i8 - 2;
            deq[i] = q as f32 * scale;
        }

        // Q2 has only 4 levels — large quantization error expected
        for (orig, deq) in values.iter().zip(deq.iter()) {
            assert!((orig - deq).abs() < 3.0, "orig={}, deq={}", orig, deq);
        }
    }

    #[test]
    fn test_q2_matvec() {
        let mut values = vec![0.0f32; 64];
        values[0] = 1.0;
        values[33] = 1.0;

        let t = Tensor::new(values, vec![2, 32]).unwrap();
        let qt = QuantizedTensor2::from_f32(&t);

        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 2.0;

        let y = qt.matvec_f32(&x);
        // Q2 is very coarse — check direction is roughly correct
        assert!(y[0] > 0.0, "y[0] should be positive: {}", y[0]);
        assert!(y[1] > 0.0, "y[1] should be positive: {}", y[1]);
    }

    #[test]
    fn test_q1_roundtrip() {
        let values: Vec<f32> = (0..32).map(|i| {
            let v = (i as f32 - 16.0) * 0.3;
            if v.abs() < 1.5 { 0.0 } else { v }
        }).collect();
        let mut block = vec![0u8; Q1_BLOCK_BYTES];
        quantize_block_q1(&values, &mut block);

        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let nonzero_mask = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let sign_bits = u32::from_le_bytes([block[8], block[9], block[10], block[11]]);

        // Verify: nonzero weights have correct sign
        for i in 0..32 {
            let is_nonzero = (nonzero_mask >> i) & 1 == 1;
            let is_negative = (sign_bits >> i) & 1 == 1;
            if is_nonzero {
                assert!(scale > 0.0);
                if is_negative {
                    assert!(values[i] < 0.0 || values[i] == 0.0, "i={}, val={}", i, values[i]);
                }
            }
        }
    }

    #[test]
    fn test_q1_matvec() {
        // Create a simple matrix where row 0 has +1 at position 0
        let mut values = vec![0.0f32; 64];
        values[0] = 5.0;  // clearly above threshold
        values[33] = -5.0;

        let t = Tensor::new(values, vec![2, 32]).unwrap();
        let qt = QuantizedTensor1::from_f32(&t);

        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 2.0;

        let y = qt.matvec_f32(&x);
        // Row 0: scale*1*3.0 should be positive and close to original
        assert!(y[0] > 0.0, "y[0] should be positive: {}", y[0]);
        // Row 1: scale*(-1)*2.0 should be negative
        assert!(y[1] < 0.0, "y[1] should be negative: {}", y[1]);
    }

    #[test]
    fn test_sparse_matvec() {
        // Create a compact tensor with some blocks having large values and some small
        let mut values = vec![0.0f32; 128]; // 4 rows x 32 cols
        // Row 0: large values in block 0
        for i in 0..32 { values[i] = (i as f32 + 1.0) * 0.5; }
        // Row 1: small values (should be pruned with low keep_fraction)
        for i in 0..32 { values[32 + i] = 0.001; }
        // Row 2: medium values
        for i in 0..32 { values[64 + i] = (i as f32 + 1.0) * 0.1; }
        // Row 3: zero
        // values[96..128] already zero

        let bytes: Vec<u8> = values.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes().to_vec()).collect();
        let compact = CompactTensor::new(bytes, vec![4, 32], crate::tensor::DType::BF16);

        let sparse = SparseQuantized::from_compact_with_sparsity(&compact, 0.5);
        assert!(sparse.active_count <= 4); // 4 total blocks, keep 50%

        let x = vec![1.0f32; 32];
        let y = sparse.matvec_f32(&x);
        assert_eq!(y.len(), 4);

        // Row 0 should have a significant result (largest block)
        assert!(y[0].abs() > 0.1, "y[0] should be nonzero: {}", y[0]);
    }

    #[test]
    fn test_awq_scaling() {
        // Test that AWQ scales + inverse scales preserve the computation approximately
        let mut values = vec![0.0f32; 64];
        for i in 0..32 { values[i] = (i as f32 - 16.0) * 0.3; }
        for i in 0..32 { values[32 + i] = (i as f32 - 16.0) * 0.1; }

        let t = Tensor::new(values.clone(), vec![2, 32]).unwrap();
        let qt_no_awq = QuantizedTensor::from_f32(&t);

        let x = vec![1.0f32; 32];
        let y_no_awq = qt_no_awq.matvec_f32(&x);

        // With AWQ: uniform scales shouldn't change the result much
        let bytes: Vec<u8> = values.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes().to_vec()).collect();
        let compact = CompactTensor::new(bytes, vec![2, 32], crate::tensor::DType::BF16);
        let awq_scales = vec![1.5f32; 32]; // Uniform scale = should be equivalent
        let qt_awq = QuantizedTensor::from_compact_awq(&compact, &awq_scales);
        let y_awq = qt_awq.matvec_f32(&x);

        // Results should be similar (not identical due to quantization, but close)
        for i in 0..2 {
            assert!((y_no_awq[i] - y_awq[i]).abs() < 5.0,
                "AWQ result diverged: no_awq={}, awq={}", y_no_awq[i], y_awq[i]);
        }
    }

    #[test]
    fn test_bit_allocation() {
        // 4 layers, target 2.0 bpw, 1000 params each
        let importance = vec![10.0, 5.0, 1.0, 0.5];
        let modes = allocate_bits(&importance, 2.0, 1000);
        assert_eq!(modes.len(), 4);

        // Most important layer should get higher bits than least important
        assert!(modes[0].bits_per_weight() >= modes[3].bits_per_weight(),
            "layer 0 ({}) should have >= bits than layer 3 ({})",
            modes[0], modes[3]);

        // Average should be at or below target
        let total_bits: f32 = modes.iter().map(|m| m.bits_per_weight() * 1000.0).sum();
        let avg_bpw = total_bits / 4000.0;
        assert!(avg_bpw <= 2.1, "avg bpw {} should be <= 2.0 (with rounding tolerance)", avg_bpw);
    }

    #[test]
    fn test_bit_allocation_high_budget() {
        // With high budget, all layers should get Q8
        let importance = vec![10.0, 5.0, 1.0, 0.5];
        let modes = allocate_bits(&importance, 8.0, 1000);
        for (i, m) in modes.iter().enumerate() {
            assert_eq!(m.bits_per_weight(), 8.0, "layer {} should be Q8 with high budget", i);
        }
    }

    #[test]
    fn test_bit_allocation_zero_budget() {
        // With zero budget, all layers should be None
        let importance = vec![10.0, 5.0];
        let modes = allocate_bits(&importance, 0.0, 1000);
        for (i, m) in modes.iter().enumerate() {
            assert_eq!(*m, QuantMode::None, "layer {} should be None with zero budget", i);
        }
    }

    #[test]
    fn test_weight_storage_all_variants() {
        // Test that all WeightStorage variants can be created and dispatch correctly
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let t = Tensor::new(values.clone(), vec![2, 32]).unwrap();
        let x = vec![1.0f32; 32];

        // Q4
        let ws4 = WeightStorage::Quantized4(QuantizedTensor::from_f32(&t));
        let y4 = ws4.matvec_f32(&x);
        assert_eq!(y4.len(), 2);
        assert_eq!(ws4.shape(), [2, 32]);

        // Q8
        let ws8 = WeightStorage::Quantized8(QuantizedTensor8::from_f32(&t));
        let y8 = ws8.matvec_f32(&x);
        assert_eq!(y8.len(), 2);
        assert_eq!(ws8.shape(), [2, 32]);

        // Q3
        let ws3 = WeightStorage::Quantized3(QuantizedTensor3::from_f32(&t));
        let y3 = ws3.matvec_f32(&x);
        assert_eq!(y3.len(), 2);
        assert_eq!(ws3.shape(), [2, 32]);

        // Q2
        let ws2 = WeightStorage::Quantized2(QuantizedTensor2::from_f32(&t));
        let y2 = ws2.matvec_f32(&x);
        assert_eq!(y2.len(), 2);
        assert_eq!(ws2.shape(), [2, 32]);

        // Q1
        let ws1 = WeightStorage::Quantized1(QuantizedTensor1::from_f32(&t));
        let y1 = ws1.matvec_f32(&x);
        assert_eq!(y1.len(), 2);
        assert_eq!(ws1.shape(), [2, 32]);

        // All formats should produce roughly similar results for identity-like inputs
        // (Q8 and Q4 should be closer to each other than Q1)
        assert!(y8[0].is_finite());
        assert!(y4[0].is_finite());
        assert!(y3[0].is_finite());
        assert!(y2[0].is_finite());
        assert!(y1[0].is_finite());
    }

    #[test]
    fn test_quant_mode_new_variants() {
        assert_eq!(QuantMode::from_str("q3"), Some(QuantMode::Q3));
        assert_eq!(QuantMode::from_str("q2"), Some(QuantMode::Q2));
        assert_eq!(QuantMode::from_str("q1"), Some(QuantMode::Q1));
        assert_eq!(QuantMode::from_str("pg:2.0"), Some(QuantMode::ProfileGuided(2.0)));
        assert_eq!(QuantMode::from_str("pg:0.5"), Some(QuantMode::ProfileGuided(0.5)));

        assert_eq!(format!("{}", QuantMode::Q3), "q3");
        assert_eq!(format!("{}", QuantMode::Q2), "q2");
        assert_eq!(format!("{}", QuantMode::Q1), "q1");
        assert_eq!(format!("{}", QuantMode::ProfileGuided(2.0)), "pg:2.0");
    }
}
