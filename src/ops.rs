//! Core neural network operations for transformer inference.
//!
//! All operations work on f32 tensors. Optimized with rayon parallelism
//! and SIMD-aware dot products for cache-friendly access patterns.
//! Supports both f32 Tensor and CompactTensor (bf16) weight variants.

use crate::tensor::{compact_linear, compact_linear_vec, CompactTensor, Tensor};

/// RMS Layer Normalization.
///
/// y_i = (x_i / sqrt(mean(x²) + eps)) * weight_i
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let data = x.data();
    let w = weight.data();
    let n = data.len();

    let mut sum_sq = 0.0f64;
    for &val in data {
        sum_sq += (val as f64) * (val as f64);
    }
    let rms = ((sum_sq / n as f64) + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = (data[i] as f64 * inv_rms * w[i] as f64) as f32;
    }

    Tensor::new(out, x.shape().to_vec()).unwrap()
}

/// RMS norm applied to each row of a [seq_len, hidden_size] tensor.
pub fn rms_norm_batch(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let data = x.data();
    let w = weight.data();

    let mut out = vec![0.0f32; seq_len * hidden];

    for s in 0..seq_len {
        let row = &data[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        let mut sum_sq = 0.0f64;
        for &val in row {
            sum_sq += (val as f64) * (val as f64);
        }
        let rms = ((sum_sq / hidden as f64) + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..hidden {
            out_row[i] = (row[i] as f64 * inv_rms * w[i] as f64) as f32;
        }
    }

    Tensor::new(out, vec![seq_len, hidden]).unwrap()
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU element-wise to a tensor.
pub fn silu_tensor(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|&v| silu(v)).collect();
    Tensor::new(data, x.shape().to_vec()).unwrap()
}

/// SwiGLU FFN with f32 Tensor weights (legacy).
pub fn swiglu_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Tensor {
    use crate::tensor::{linear, linear_vec};

    if x.ndim() == 1 {
        let gate = linear_vec(x, gate_weight).unwrap();
        let up = linear_vec(x, up_weight).unwrap();

        let gate_data = gate.data();
        let up_data = up.data();
        let activated: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();

        linear_vec(&intermediate, down_weight).unwrap()
    } else {
        let gate = linear(x, gate_weight).unwrap();
        let up = linear(x, up_weight).unwrap();

        let gate_data = gate.data();
        let up_data = up.data();
        let activated: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();

        linear(&intermediate, down_weight).unwrap()
    }
}

/// SwiGLU FFN with CompactTensor (bf16) weights.
/// ffn(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
pub fn compact_swiglu_ffn(
    x: &Tensor,
    gate_weight: &CompactTensor,
    up_weight: &CompactTensor,
    down_weight: &CompactTensor,
) -> Tensor {
    if x.ndim() == 1 {
        let gate = compact_linear_vec(x, gate_weight);
        let up = compact_linear_vec(x, up_weight);

        let gate_data = gate.data();
        let up_data = up.data();
        let activated: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();

        compact_linear_vec(&intermediate, down_weight)
    } else {
        let gate = compact_linear(x, gate_weight);
        let up = compact_linear(x, up_weight);

        let gate_data = gate.data();
        let up_data = up.data();
        let activated: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();

        compact_linear(&intermediate, down_weight)
    }
}

/// Softmax over the last dimension.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

/// Precomputed RoPE frequency table.
pub struct RopeFreqs {
    cos: Vec<f32>,
    sin: Vec<f32>,
    head_dim: usize,
    max_seq_len: usize,
}

impl RopeFreqs {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        scaling: Option<&crate::config::RopeScalingConfig>,
    ) -> Self {
        let half_dim = head_dim / 2;

        let mut inv_freqs: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        if let Some(sc) = scaling {
            if sc.rope_type == "llama3" {
                let factor = sc.factor;
                let low_freq = sc.low_freq_factor;
                let high_freq = sc.high_freq_factor;
                let old_context = sc.original_max_position_embeddings as f64;

                let low_freq_wavelen = old_context / low_freq;
                let high_freq_wavelen = old_context / high_freq;

                for freq in inv_freqs.iter_mut() {
                    let wavelen = 2.0 * std::f64::consts::PI / *freq;
                    if wavelen > low_freq_wavelen {
                        *freq /= factor;
                    } else if wavelen > high_freq_wavelen {
                        let smooth = (old_context / wavelen - low_freq) / (high_freq - low_freq);
                        *freq = (1.0 - smooth) * (*freq / factor) + smooth * *freq;
                    }
                }
            }
        }

        let total = max_seq_len * half_dim;
        let mut cos = vec![0.0f32; total];
        let mut sin = vec![0.0f32; total];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f64 * inv_freqs[i];
                cos[pos * half_dim + i] = angle.cos() as f32;
                sin[pos * half_dim + i] = angle.sin() as f32;
            }
        }

        RopeFreqs {
            cos,
            sin,
            head_dim,
            max_seq_len,
        }
    }

    pub fn apply(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        debug_assert!(pos < self.max_seq_len);
        let half = self.head_dim / 2;
        let base = pos * half;

        for i in 0..half {
            let cos_val = self.cos[base + i];
            let sin_val = self.sin[base + i];
            let q0 = q[i];
            let q1 = q[i + half];
            q[i] = q0 * cos_val - q1 * sin_val;
            q[i + half] = q0 * sin_val + q1 * cos_val;
        }

        for i in 0..half {
            let cos_val = self.cos[base + i];
            let sin_val = self.sin[base + i];
            let k0 = k[i];
            let k1 = k[i + half];
            k[i] = k0 * cos_val - k1 * sin_val;
            k[i + half] = k0 * sin_val + k1 * cos_val;
        }
    }

    /// Raw cos table: [max_seq_len * half_dim] in row-major order.
    pub fn cos_data(&self) -> &[f32] { &self.cos }

    /// Raw sin table: [max_seq_len * half_dim] in row-major order.
    pub fn sin_data(&self) -> &[f32] { &self.sin }

    /// Half of the head dimension (cos/sin table width).
    pub fn half_dim(&self) -> usize { self.head_dim / 2 }

    /// Maximum pre-computed sequence length.
    pub fn max_seq(&self) -> usize { self.max_seq_len }

    pub fn apply_q(&self, q: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        let base = pos * half;

        for i in 0..half {
            let cos_val = self.cos[base + i];
            let sin_val = self.sin[base + i];
            let q0 = q[i];
            let q1 = q[i + half];
            q[i] = q0 * cos_val - q1 * sin_val;
            q[i + half] = q0 * sin_val + q1 * cos_val;
        }
    }

    pub fn apply_k(&self, k: &mut [f32], pos: usize) {
        self.apply_q(k, pos);
    }
}

/// Layer Normalization.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Tensor {
    let data = x.data();
    let w = weight.data();
    let n = data.len();

    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&v| {
        let d = v as f64 - mean;
        d * d
    }).sum::<f64>() / n as f64;
    let inv_std = 1.0 / (var + eps).sqrt();

    let mut out = vec![0.0f32; n];
    if let Some(bias) = bias {
        let b = bias.data();
        for i in 0..n {
            out[i] = ((data[i] as f64 - mean) * inv_std * w[i] as f64 + b[i] as f64) as f32;
        }
    } else {
        for i in 0..n {
            out[i] = ((data[i] as f64 - mean) * inv_std * w[i] as f64) as f32;
        }
    }

    Tensor::new(out, x.shape().to_vec()).unwrap()
}

/// Layer norm applied to each row of a [seq_len, hidden_size] tensor.
pub fn layer_norm_batch(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Tensor {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let data = x.data();
    let w = weight.data();

    let mut out = vec![0.0f32; seq_len * hidden];

    for s in 0..seq_len {
        let row = &data[s * hidden..(s + 1) * hidden];
        let out_row = &mut out[s * hidden..(s + 1) * hidden];

        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var: f64 = row.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d
        }).sum::<f64>() / hidden as f64;
        let inv_std = 1.0 / (var + eps).sqrt();

        if let Some(bias) = bias {
            let b = bias.data();
            for i in 0..hidden {
                out_row[i] = ((row[i] as f64 - mean) * inv_std * w[i] as f64 + b[i] as f64) as f32;
            }
        } else {
            for i in 0..hidden {
                out_row[i] = ((row[i] as f64 - mean) * inv_std * w[i] as f64) as f32;
            }
        }
    }

    Tensor::new(out, vec![seq_len, hidden]).unwrap()
}

/// GELU activation (approximate).
#[inline]
pub fn gelu(x: f32) -> f32 {
    let c = 0.7978845608f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Standard 2-layer FFN with GELU.
pub fn gelu_ffn(
    x: &Tensor,
    up_weight: &Tensor,
    up_bias: Option<&Tensor>,
    down_weight: &Tensor,
    down_bias: Option<&Tensor>,
) -> Tensor {
    use crate::tensor::{linear, linear_vec};

    if x.ndim() == 1 {
        let mut up = linear_vec(x, up_weight).unwrap();
        if let Some(bias) = up_bias {
            up.add_inplace(bias);
        }
        let activated: Vec<f32> = up.data().iter().map(|&v| gelu(v)).collect();
        let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
        let mut out = linear_vec(&act, down_weight).unwrap();
        if let Some(bias) = down_bias {
            out.add_inplace(bias);
        }
        out
    } else {
        let mut up = linear(x, up_weight).unwrap();
        if let Some(bias) = up_bias {
            let data = up.data_mut();
            let hidden = bias.numel();
            let seq_len = data.len() / hidden;
            for s in 0..seq_len {
                for i in 0..hidden {
                    data[s * hidden + i] += bias.data()[i];
                }
            }
        }
        let activated: Vec<f32> = up.data().iter().map(|&v| gelu(v)).collect();
        let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
        let mut out = linear(&act, down_weight).unwrap();
        if let Some(bias) = down_bias {
            let data = out.data_mut();
            let hidden = bias.numel();
            let seq_len = data.len() / hidden;
            for s in 0..seq_len {
                for i in 0..hidden {
                    data[s * hidden + i] += bias.data()[i];
                }
            }
        }
        out
    }
}

/// GeGLU feed-forward network.
pub fn geglu_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Tensor {
    use crate::tensor::{linear, linear_vec};

    if x.ndim() == 1 {
        let gate = linear_vec(x, gate_weight).unwrap();
        let up = linear_vec(x, up_weight).unwrap();
        let activated: Vec<f32> = gate.data().iter()
            .zip(up.data().iter())
            .map(|(&g, &u)| gelu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();
        linear_vec(&intermediate, down_weight).unwrap()
    } else {
        let gate = linear(x, gate_weight).unwrap();
        let up = linear(x, up_weight).unwrap();
        let activated: Vec<f32> = gate.data().iter()
            .zip(up.data().iter())
            .map(|(&g, &u)| gelu(g) * u)
            .collect();
        let intermediate = Tensor::new(activated, gate.shape().to_vec()).unwrap();
        linear(&intermediate, down_weight).unwrap()
    }
}

/// Standard 2-layer FFN with ReLU.
pub fn relu_ffn(
    x: &Tensor,
    up_weight: &Tensor,
    up_bias: Option<&Tensor>,
    down_weight: &Tensor,
    down_bias: Option<&Tensor>,
) -> Tensor {
    use crate::tensor::{linear, linear_vec};

    if x.ndim() == 1 {
        let mut up = linear_vec(x, up_weight).unwrap();
        if let Some(bias) = up_bias {
            up.add_inplace(bias);
        }
        let activated: Vec<f32> = up.data().iter().map(|&v| v.max(0.0)).collect();
        let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
        let mut out = linear_vec(&act, down_weight).unwrap();
        if let Some(bias) = down_bias {
            out.add_inplace(bias);
        }
        out
    } else {
        let mut up = linear(x, up_weight).unwrap();
        if let Some(bias) = up_bias {
            let data = up.data_mut();
            let hidden = bias.numel();
            let seq_len = data.len() / hidden;
            for s in 0..seq_len {
                for i in 0..hidden {
                    data[s * hidden + i] += bias.data()[i];
                }
            }
        }
        let activated: Vec<f32> = up.data().iter().map(|&v| v.max(0.0)).collect();
        let act = Tensor::new(activated, up.shape().to_vec()).unwrap();
        let mut out = linear(&act, down_weight).unwrap();
        if let Some(bias) = down_bias {
            let data = out.data_mut();
            let hidden = bias.numel();
            let seq_len = data.len() / hidden;
            for s in 0..seq_len {
                for i in 0..hidden {
                    data[s * hidden + i] += bias.data()[i];
                }
            }
        }
        out
    }
}

pub fn argmax(data: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Top-p (nucleus) sampling.
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();

    let max_val = probs.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in probs.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for (_, v) in probs.iter_mut() {
        *v /= sum;
    }

    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let probs = &probs[..cutoff];

    let total: f32 = probs.iter().map(|(_, p)| p).sum();

    let mut rng_val = rand::random::<f32>() * total;
    for &(idx, p) in probs {
        rng_val -= p;
        if rng_val <= 0.0 {
            return idx;
        }
    }

    probs[probs.len() - 1].0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let y = rms_norm(&x, &w, 1e-5);
        let rms = (7.5f64 + 1e-5).sqrt();
        assert!((y.data()[0] as f64 - 1.0 / rms).abs() < 1e-4);
        assert!((y.data()[1] as f64 - 2.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7311).abs() < 1e-3);
        assert!((silu(10.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    fn test_rope_basic() {
        let freqs = RopeFreqs::new(8, 128, 10000.0, None);
        let mut q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        freqs.apply(&mut q, &mut k, 0);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[4] - 0.0).abs() < 1e-6);

        let mut q2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut k2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        freqs.apply(&mut q2, &mut k2, 10);
        assert!((q2[0] - 1.0).abs() > 0.01);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -3.0, -2.0]), 0);
    }
}
