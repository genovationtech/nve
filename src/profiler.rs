use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Monte Carlo Activation Profiler (MCAP).
///
/// Estimates weight importance via sampling: run diverse prompts through the model,
/// track activation magnitudes per weight, and compute importance as the Monte Carlo
/// expectation of activation.
///
/// Î(Wᵢ) = (1/N) Σₖ activationₖ(Wᵢ)

/// A single activation observation for a weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSample {
    pub weight_id: u64,
    pub magnitude: f64,
    pub prompt_domain: Option<String>,
}

/// Accumulated importance statistics for a single weight.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeightImportance {
    pub weight_id: u64,
    pub total_activation: f64,
    pub sample_count: u64,
    pub max_activation: f64,
    pub min_activation: f64,
    /// Per-domain activation sums for distribution-aware profiling.
    pub domain_activations: HashMap<String, f64>,
    pub domain_counts: HashMap<String, u64>,
}

impl WeightImportance {
    pub fn new(weight_id: u64) -> Self {
        WeightImportance {
            weight_id,
            total_activation: 0.0,
            sample_count: 0,
            max_activation: f64::NEG_INFINITY,
            min_activation: f64::INFINITY,
            domain_activations: HashMap::new(),
            domain_counts: HashMap::new(),
        }
    }

    pub fn record(&mut self, sample: &ActivationSample) {
        self.total_activation += sample.magnitude;
        self.sample_count += 1;
        if sample.magnitude > self.max_activation {
            self.max_activation = sample.magnitude;
        }
        if sample.magnitude < self.min_activation {
            self.min_activation = sample.magnitude;
        }

        if let Some(ref domain) = sample.prompt_domain {
            *self.domain_activations.entry(domain.clone()).or_insert(0.0) += sample.magnitude;
            *self.domain_counts.entry(domain.clone()).or_insert(0) += 1;
        }
    }

    /// Monte Carlo estimate of importance: E[activation].
    pub fn importance(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.total_activation / self.sample_count as f64
    }

    /// Importance for a specific domain.
    pub fn domain_importance(&self, domain: &str) -> f64 {
        let total = self.domain_activations.get(domain).copied().unwrap_or(0.0);
        let count = self.domain_counts.get(domain).copied().unwrap_or(0);
        if count == 0 {
            return 0.0;
        }
        total / count as f64
    }

    /// Variance of the activation distribution (for confidence estimation).
    pub fn activation_variance(&self) -> f64 {
        // We'd need sum-of-squares for true variance; approximate with range for now.
        if self.sample_count < 2 {
            return 0.0;
        }
        let range = self.max_activation - self.min_activation;
        // Rough approximation: variance ≈ (range/4)² for normal-ish distributions.
        (range / 4.0).powi(2)
    }
}

/// Configuration for Monte Carlo profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Number of prompts to sample per profiling round.
    pub samples_per_round: usize,
    /// Minimum samples before importance estimates are considered stable.
    pub min_samples_for_stability: u64,
    /// Exponential moving average decay for online updates (0 = no decay, 1 = full decay).
    pub ema_decay: f64,
    /// Importance threshold below which weights are considered cold.
    pub cold_threshold: f64,
    /// Importance threshold above which weights are considered hot.
    pub hot_threshold: f64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        ProfilerConfig {
            samples_per_round: 100,
            min_samples_for_stability: 50,
            ema_decay: 0.01,
            cold_threshold: 0.1,
            hot_threshold: 0.7,
        }
    }
}

/// The profiler accumulates activation samples and computes importance rankings.
pub struct Profiler {
    config: ProfilerConfig,
    weights: HashMap<u64, WeightImportance>,
    total_rounds: u64,
}

impl Profiler {
    pub fn new(config: ProfilerConfig) -> Self {
        Profiler {
            config,
            weights: HashMap::new(),
            total_rounds: 0,
        }
    }

    /// Record a batch of activation samples from a single forward pass.
    pub fn record_batch(&mut self, samples: Vec<ActivationSample>) {
        for sample in &samples {
            self.weights
                .entry(sample.weight_id)
                .or_insert_with(|| WeightImportance::new(sample.weight_id))
                .record(sample);
        }
    }

    /// Record a single activation sample.
    pub fn record(&mut self, sample: ActivationSample) {
        self.weights
            .entry(sample.weight_id)
            .or_insert_with(|| WeightImportance::new(sample.weight_id))
            .record(&sample);
    }

    /// Mark that a profiling round is complete.
    pub fn finish_round(&mut self) {
        self.total_rounds += 1;
    }

    /// Get importance ranking of all profiled weights, sorted descending.
    pub fn importance_ranking(&self) -> Vec<(u64, f64)> {
        let mut ranking: Vec<(u64, f64)> = self
            .weights
            .values()
            .map(|w| (w.weight_id, w.importance()))
            .collect();
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranking
    }

    /// Partition weights into hot/warm/cold based on importance percentiles.
    pub fn partition(&self, hot_fraction: f64, warm_fraction: f64) -> TierPartition {
        let ranking = self.importance_ranking();
        let n = ranking.len();

        let hot_cutoff = (n as f64 * hot_fraction).ceil() as usize;
        let warm_cutoff = hot_cutoff + (n as f64 * warm_fraction).ceil() as usize;

        TierPartition {
            hot: ranking[..hot_cutoff.min(n)]
                .iter()
                .map(|&(id, _)| id)
                .collect(),
            warm: ranking[hot_cutoff.min(n)..warm_cutoff.min(n)]
                .iter()
                .map(|&(id, _)| id)
                .collect(),
            cold: ranking[warm_cutoff.min(n)..]
                .iter()
                .map(|&(id, _)| id)
                .collect(),
        }
    }

    /// Domain-specific importance ranking.
    pub fn domain_ranking(&self, domain: &str) -> Vec<(u64, f64)> {
        let mut ranking: Vec<(u64, f64)> = self
            .weights
            .values()
            .map(|w| (w.weight_id, w.domain_importance(domain)))
            .collect();
        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranking
    }

    /// Check if profiling is considered stable (enough samples collected).
    pub fn is_stable(&self) -> bool {
        self.weights.values().all(|w| w.sample_count >= self.config.min_samples_for_stability)
    }

    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    pub fn total_rounds(&self) -> u64 {
        self.total_rounds
    }

    pub fn get_importance(&self, weight_id: u64) -> Option<f64> {
        self.weights.get(&weight_id).map(|w| w.importance())
    }

    pub fn get_weight_stats(&self, weight_id: u64) -> Option<&WeightImportance> {
        self.weights.get(&weight_id)
    }
}

/// Result of partitioning weights into tier assignments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierPartition {
    pub hot: Vec<u64>,
    pub warm: Vec<u64>,
    pub cold: Vec<u64>,
}

/// Generate a diverse prompt distribution for profiling.
/// In production this would sample from real prompt datasets;
/// here we define the domain taxonomy and sampling interface.
#[derive(Debug, Clone)]
pub struct PromptDistribution {
    pub domains: Vec<PromptDomain>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDomain {
    pub name: String,
    pub weight: f64,
    pub prompts: Vec<String>,
}

impl PromptDistribution {
    pub fn new(domains: Vec<PromptDomain>) -> Self {
        PromptDistribution { domains }
    }

    /// Sample a batch of prompts according to domain weights.
    pub fn sample(&self, count: usize) -> Vec<(String, String)> {
        let mut rng = rand::thread_rng();
        let total_weight: f64 = self.domains.iter().map(|d| d.weight).sum();
        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            let mut pick = rng.gen::<f64>() * total_weight;
            for domain in &self.domains {
                pick -= domain.weight;
                if pick <= 0.0 {
                    if let Some(prompt) = domain.prompts.choose(&mut rng) {
                        result.push((domain.name.clone(), prompt.clone()));
                    }
                    break;
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_calculation() {
        let mut profiler = Profiler::new(ProfilerConfig::default());

        // Weight 1: high activation.
        for _ in 0..100 {
            profiler.record(ActivationSample {
                weight_id: 1,
                magnitude: 0.9,
                prompt_domain: Some("math".into()),
            });
        }

        // Weight 2: low activation.
        for _ in 0..100 {
            profiler.record(ActivationSample {
                weight_id: 2,
                magnitude: 0.1,
                prompt_domain: Some("math".into()),
            });
        }

        let ranking = profiler.importance_ranking();
        assert_eq!(ranking[0].0, 1); // Weight 1 should rank higher.
        assert!(ranking[0].1 > ranking[1].1);
    }

    #[test]
    fn test_partition() {
        let mut profiler = Profiler::new(ProfilerConfig::default());

        for i in 0..100 {
            profiler.record(ActivationSample {
                weight_id: i,
                magnitude: i as f64 / 100.0,
                prompt_domain: None,
            });
        }

        let partition = profiler.partition(0.2, 0.3);
        assert_eq!(partition.hot.len(), 20);
        assert_eq!(partition.warm.len(), 30);
        assert_eq!(partition.cold.len(), 50);
    }

    #[test]
    fn test_domain_importance() {
        let mut profiler = Profiler::new(ProfilerConfig::default());

        // Weight activates strongly for math, weakly for chat.
        for _ in 0..50 {
            profiler.record(ActivationSample {
                weight_id: 1,
                magnitude: 0.95,
                prompt_domain: Some("math".into()),
            });
            profiler.record(ActivationSample {
                weight_id: 1,
                magnitude: 0.05,
                prompt_domain: Some("chat".into()),
            });
        }

        let stats = profiler.get_weight_stats(1).unwrap();
        assert!(stats.domain_importance("math") > 0.9);
        assert!(stats.domain_importance("chat") < 0.1);
    }
}
