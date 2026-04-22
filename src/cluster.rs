use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Co-activation weight clustering.
///
/// Groups weights that tend to activate together into clusters (latent feature groups).
/// W = ⋃ₖ Wₖ where each Wₖ is a coherent computational unit.
///
/// This enables loading entire clusters at once rather than individual weights,
/// reducing paging overhead and improving cache locality.

/// A cluster of co-activated weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightCluster {
    pub id: u32,
    pub weight_ids: Vec<u64>,
    pub mean_importance: f64,
    /// Domains where this cluster is most active.
    pub dominant_domains: Vec<String>,
    /// Internal cohesion score (higher = weights in this cluster correlate more).
    pub cohesion: f64,
}

/// Tracks co-activation patterns between weight pairs.
pub struct CoActivationTracker {
    /// co_activation[(a, b)] = number of times both a and b were active in the same pass.
    joint_counts: HashMap<(u64, u64), u64>,
    /// individual_counts[a] = number of times a was active.
    individual_counts: HashMap<u64, u64>,
    /// Total number of forward passes observed.
    total_passes: u64,
    /// Only track co-activation for the top-N weights (by activation frequency)
    /// to keep memory bounded.
    max_tracked_weights: usize,
}

impl CoActivationTracker {
    pub fn new(max_tracked_weights: usize) -> Self {
        CoActivationTracker {
            joint_counts: HashMap::new(),
            individual_counts: HashMap::new(),
            total_passes: 0,
            max_tracked_weights,
        }
    }

    /// Record which weights were active in a single forward pass.
    /// `active_weights` should contain only weights whose activation exceeded a threshold.
    pub fn record_pass(&mut self, active_weights: &[u64]) {
        self.total_passes += 1;

        for &w in active_weights {
            *self.individual_counts.entry(w).or_insert(0) += 1;
        }

        // Record pairwise co-activations (only for manageable set sizes).
        let n = active_weights.len().min(self.max_tracked_weights);
        for i in 0..n {
            for j in (i + 1)..n {
                let key = if active_weights[i] < active_weights[j] {
                    (active_weights[i], active_weights[j])
                } else {
                    (active_weights[j], active_weights[i])
                };
                *self.joint_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Compute pointwise mutual information between two weights.
    /// PMI(a,b) = log(P(a,b) / (P(a) * P(b)))
    /// High PMI = strongly co-activated.
    pub fn pmi(&self, a: u64, b: u64) -> f64 {
        if self.total_passes == 0 {
            return 0.0;
        }

        let key = if a < b { (a, b) } else { (b, a) };
        let joint = *self.joint_counts.get(&key).unwrap_or(&0) as f64;
        let count_a = *self.individual_counts.get(&a).unwrap_or(&0) as f64;
        let count_b = *self.individual_counts.get(&b).unwrap_or(&0) as f64;
        let n = self.total_passes as f64;

        if joint == 0.0 || count_a == 0.0 || count_b == 0.0 {
            return 0.0;
        }

        let p_ab = joint / n;
        let p_a = count_a / n;
        let p_b = count_b / n;

        (p_ab / (p_a * p_b)).ln()
    }

    pub fn total_passes(&self) -> u64 {
        self.total_passes
    }

    pub fn tracked_weights(&self) -> usize {
        self.individual_counts.len()
    }
}

/// Simple greedy clustering based on co-activation PMI.
///
/// Algorithm:
/// 1. Build PMI adjacency for all tracked weight pairs
/// 2. Greedily merge the highest-PMI pairs into clusters
/// 3. Stop when PMI drops below threshold
pub struct Clusterer {
    min_pmi_threshold: f64,
    max_cluster_size: usize,
}

impl Clusterer {
    pub fn new(min_pmi_threshold: f64, max_cluster_size: usize) -> Self {
        Clusterer {
            min_pmi_threshold,
            max_cluster_size,
        }
    }

    /// Build clusters from co-activation data.
    pub fn cluster(
        &self,
        tracker: &CoActivationTracker,
        importances: &HashMap<u64, f64>,
    ) -> Vec<WeightCluster> {
        // Collect all weight IDs.
        let weight_ids: Vec<u64> = importances.keys().copied().collect();
        let mut assigned: HashMap<u64, u32> = HashMap::new();
        let mut clusters: Vec<Vec<u64>> = Vec::new();

        // Sort pairs by PMI descending (greedy approach).
        let mut pairs: Vec<((u64, u64), f64)> = Vec::new();
        for i in 0..weight_ids.len() {
            for j in (i + 1)..weight_ids.len() {
                let pmi = tracker.pmi(weight_ids[i], weight_ids[j]);
                if pmi >= self.min_pmi_threshold {
                    pairs.push(((weight_ids[i], weight_ids[j]), pmi));
                }
            }
        }
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for ((a, b), _pmi) in &pairs {
            let cluster_a = assigned.get(a).copied();
            let cluster_b = assigned.get(b).copied();

            match (cluster_a, cluster_b) {
                (None, None) => {
                    // Create new cluster with both.
                    let cid = clusters.len() as u32;
                    clusters.push(vec![*a, *b]);
                    assigned.insert(*a, cid);
                    assigned.insert(*b, cid);
                }
                (Some(ca), None) => {
                    if clusters[ca as usize].len() < self.max_cluster_size {
                        clusters[ca as usize].push(*b);
                        assigned.insert(*b, ca);
                    }
                }
                (None, Some(cb)) => {
                    if clusters[cb as usize].len() < self.max_cluster_size {
                        clusters[cb as usize].push(*a);
                        assigned.insert(*a, cb);
                    }
                }
                (Some(_), Some(_)) => {
                    // Both already assigned — skip (no merge for simplicity).
                }
            }
        }

        // Create singleton clusters for unassigned weights.
        for &wid in &weight_ids {
            if !assigned.contains_key(&wid) {
                let cid = clusters.len() as u32;
                clusters.push(vec![wid]);
                assigned.insert(wid, cid);
            }
        }

        // Build WeightCluster structs.
        clusters
            .into_iter()
            .enumerate()
            .map(|(i, members)| {
                let mean_imp = if members.is_empty() {
                    0.0
                } else {
                    members
                        .iter()
                        .map(|id| importances.get(id).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        / members.len() as f64
                };

                WeightCluster {
                    id: i as u32,
                    weight_ids: members,
                    mean_importance: mean_imp,
                    dominant_domains: Vec::new(), // Filled in by caller with profiler data.
                    cohesion: 0.0,                // Could compute avg intra-cluster PMI.
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_co_activation_tracking() {
        let mut tracker = CoActivationTracker::new(100);

        // Weights 1 and 2 always co-activate.
        for _ in 0..100 {
            tracker.record_pass(&[1, 2]);
        }

        // Weight 3 activates independently.
        for _ in 0..100 {
            tracker.record_pass(&[3]);
        }

        let pmi_12 = tracker.pmi(1, 2);
        let pmi_13 = tracker.pmi(1, 3);

        assert!(pmi_12 > 0.0, "co-activated weights should have positive PMI");
        assert!(
            pmi_13 <= 0.0 || pmi_13.abs() < f64::EPSILON,
            "non-co-activated weights should have zero PMI"
        );
    }

    #[test]
    fn test_clustering() {
        let mut tracker = CoActivationTracker::new(100);

        // Create two clear clusters: {1,2,3} and {4,5,6}.
        for _ in 0..200 {
            tracker.record_pass(&[1, 2, 3]);
            tracker.record_pass(&[4, 5, 6]);
        }

        let mut importances = HashMap::new();
        for i in 1..=6 {
            importances.insert(i, 0.5);
        }

        let clusterer = Clusterer::new(0.1, 10);
        let clusters = clusterer.cluster(&tracker, &importances);

        // Should produce at least 2 clusters.
        assert!(clusters.len() >= 2);

        // Find which cluster weight 1 is in.
        let cluster_of_1 = clusters
            .iter()
            .find(|c| c.weight_ids.contains(&1))
            .unwrap();

        // Weight 2 and 3 should be in the same cluster as 1.
        assert!(cluster_of_1.weight_ids.contains(&2));
        assert!(cluster_of_1.weight_ids.contains(&3));

        // Weight 4 should NOT be in the same cluster.
        assert!(!cluster_of_1.weight_ids.contains(&4));
    }
}
