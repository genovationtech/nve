use std::collections::HashMap;

use crate::cluster::WeightCluster;
use crate::profiler::TierPartition;
use crate::tier::{TierLevel, TierManager, WeightBlock};

use log::{debug, info};
use serde::{Deserialize, Serialize};

/// Virtual weight pager — the core runtime component.
///
/// Manages dynamic loading/unloading of weight clusters across memory tiers
/// based on predicted activation patterns, without deleting any weights.

/// Configuration for the pager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagerConfig {
    /// Fraction of weights to keep on GPU.
    pub gpu_fraction: f64,
    /// Fraction of weights to keep in RAM.
    pub ram_fraction: f64,
    /// Number of recent access timestamps to track per cluster (for LRU).
    pub access_history_depth: usize,
    /// Enable prefetching: predict next clusters and load ahead.
    pub enable_prefetch: bool,
    /// How many clusters to prefetch.
    pub prefetch_count: usize,
}

impl Default for PagerConfig {
    fn default() -> Self {
        PagerConfig {
            gpu_fraction: 0.2,
            ram_fraction: 0.3,
            enable_prefetch: true,
            prefetch_count: 3,
            access_history_depth: 64,
        }
    }
}

impl PagerConfig {
    /// Compute tier fractions from byte budgets and total model size.
    ///
    /// # Arguments
    /// * `model_bytes` - Total model size in bytes
    /// * `gpu_budget_bytes` - Available GPU VRAM budget for weights
    /// * `ram_budget_bytes` - Available system RAM budget for weights
    pub fn from_budget(model_bytes: usize, gpu_budget_bytes: usize, ram_budget_bytes: usize) -> Self {
        let model = model_bytes.max(1) as f64;
        let gpu_frac = (gpu_budget_bytes as f64 / model).min(1.0);
        let ram_frac = (ram_budget_bytes as f64 / model).min(1.0 - gpu_frac);

        PagerConfig {
            gpu_fraction: gpu_frac,
            ram_fraction: ram_frac,
            ..Default::default()
        }
    }
}

/// Access record for a cluster.
#[derive(Debug, Clone)]
struct ClusterAccess {
    #[allow(dead_code)]
    cluster_id: u32,
    access_count: u64,
    last_access_tick: u64,
}

/// The pager orchestrates tier placement and runtime migration.
pub struct Pager {
    #[allow(dead_code)]
    config: PagerConfig,
    tier_manager: TierManager,
    /// Cluster ID → list of WeightBlock IDs in that cluster.
    cluster_blocks: HashMap<u32, Vec<u64>>,
    /// Access tracking for clusters (for LRU / frequency-based eviction).
    cluster_access: HashMap<u32, ClusterAccess>,
    /// Monotonic tick counter for ordering accesses.
    tick: u64,
    /// Page fault counter (cluster was needed but not on GPU).
    page_faults: u64,
    /// Page hit counter.
    page_hits: u64,
}

impl Pager {
    pub fn new(config: PagerConfig, tier_manager: TierManager) -> Self {
        Pager {
            config,
            tier_manager,
            cluster_blocks: HashMap::new(),
            cluster_access: HashMap::new(),
            tick: 0,
            page_faults: 0,
            page_hits: 0,
        }
    }

    /// Initialize placement based on profiler partition.
    pub fn initialize_placement(
        &mut self,
        clusters: &[WeightCluster],
        blocks: &[WeightBlock],
        partition: &TierPartition,
    ) {
        // Build block lookup.
        let block_map: HashMap<u64, &WeightBlock> = blocks.iter().map(|b| (b.id, b)).collect();

        // Map cluster → blocks.
        for cluster in clusters {
            self.cluster_blocks
                .insert(cluster.id, cluster.weight_ids.clone());
        }

        // Place hot weights on GPU.
        for &wid in &partition.hot {
            if let Some(block) = block_map.get(&wid) {
                let _ = self.tier_manager.place(block, TierLevel::Gpu);
            }
        }

        // Place warm weights in RAM.
        for &wid in &partition.warm {
            if let Some(block) = block_map.get(&wid) {
                let _ = self.tier_manager.place(block, TierLevel::Ram);
            }
        }

        // Place cold weights on SSD.
        for &wid in &partition.cold {
            if let Some(block) = block_map.get(&wid) {
                let _ = self.tier_manager.place(block, TierLevel::Ssd);
            }
        }

        info!(
            "Placement initialized: GPU={:.1}%, RAM={:.1}%, SSD={:.1}%",
            self.tier_manager.utilization(TierLevel::Gpu) * 100.0,
            self.tier_manager.utilization(TierLevel::Ram) * 100.0,
            self.tier_manager.utilization(TierLevel::Ssd) * 100.0,
        );
    }

    /// Called when a cluster is accessed at runtime.
    /// Returns the tier the cluster's weights are currently on.
    pub fn access_cluster(&mut self, cluster_id: u32) -> TierLevel {
        self.tick += 1;

        let access = self
            .cluster_access
            .entry(cluster_id)
            .or_insert(ClusterAccess {
                cluster_id,
                access_count: 0,
                last_access_tick: 0,
            });
        access.access_count += 1;
        access.last_access_tick = self.tick;

        // Check what tier the first block of this cluster is on.
        let tier = self
            .cluster_blocks
            .get(&cluster_id)
            .and_then(|blocks| blocks.first())
            .and_then(|&bid| self.tier_manager.get_tier(bid))
            .unwrap_or(TierLevel::Ssd);

        if tier == TierLevel::Gpu {
            self.page_hits += 1;
        } else {
            self.page_faults += 1;
        }

        tier
    }

    /// Attempt to promote a cluster's blocks to a faster tier.
    pub fn promote_cluster(
        &mut self,
        cluster_id: u32,
        blocks: &[WeightBlock],
    ) -> Vec<(u64, Option<TierLevel>)> {
        let block_map: HashMap<u64, &WeightBlock> = blocks.iter().map(|b| (b.id, b)).collect();
        let mut results = Vec::new();

        if let Some(block_ids) = self.cluster_blocks.get(&cluster_id).cloned() {
            for bid in block_ids {
                if let Some(block) = block_map.get(&bid) {
                    match self.tier_manager.promote(block) {
                        Ok(new_tier) => results.push((bid, new_tier)),
                        Err(e) => {
                            debug!("Could not promote block {}: {}", bid, e);
                            results.push((bid, None));
                        }
                    }
                }
            }
        }

        results
    }

    /// Evict the least-recently-used cluster from GPU to make room.
    pub fn evict_lru_from_gpu(&mut self, blocks: &[WeightBlock]) -> Option<u32> {
        let block_map: HashMap<u64, &WeightBlock> = blocks.iter().map(|b| (b.id, b)).collect();

        // Find the cluster on GPU with the oldest last access.
        let gpu_blocks = self.tier_manager.blocks_at(TierLevel::Gpu);
        let mut cluster_last_access: HashMap<u32, u64> = HashMap::new();

        for &bid in &gpu_blocks {
            for (&cid, cblocks) in &self.cluster_blocks {
                if cblocks.contains(&bid) {
                    let last = self
                        .cluster_access
                        .get(&cid)
                        .map(|a| a.last_access_tick)
                        .unwrap_or(0);
                    cluster_last_access
                        .entry(cid)
                        .and_modify(|v| *v = (*v).min(last))
                        .or_insert(last);
                }
            }
        }

        // Find LRU cluster.
        let lru_cluster = cluster_last_access
            .iter()
            .min_by_key(|(_, &tick)| tick)
            .map(|(&cid, _)| cid);

        if let Some(cid) = lru_cluster {
            if let Some(block_ids) = self.cluster_blocks.get(&cid).cloned() {
                for bid in block_ids {
                    if let Some(block) = block_map.get(&bid) {
                        let _ = self.tier_manager.demote(block);
                    }
                }
            }
        }

        lru_cluster
    }

    pub fn page_fault_rate(&self) -> f64 {
        let total = self.page_hits + self.page_faults;
        if total == 0 {
            return 0.0;
        }
        self.page_faults as f64 / total as f64
    }

    pub fn page_faults(&self) -> u64 {
        self.page_faults
    }

    pub fn page_hits(&self) -> u64 {
        self.page_hits
    }

    pub fn tier_utilization(&self, level: TierLevel) -> f64 {
        self.tier_manager.utilization(level)
    }

    pub fn migration_count(&self) -> u64 {
        self.tier_manager.migration_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier::TierConfig;
    use std::path::PathBuf;

    fn setup() -> (Pager, Vec<WeightBlock>) {
        let tier_config = TierConfig::new(
            4000,  // GPU: 4KB
            16000, // RAM: 16KB
            64000, // SSD: 64KB
            PathBuf::from("/tmp/nve_test_pager"),
        );
        let tier_mgr = TierManager::new(tier_config);
        let pager = Pager::new(PagerConfig::default(), tier_mgr);

        let blocks: Vec<WeightBlock> = (0..20)
            .map(|i| WeightBlock {
                id: i,
                layer_index: i as usize / 4,
                offset: (i as usize % 4) * 256,
                size_bytes: 256,
                importance: (20 - i) as f64 / 20.0,
            })
            .collect();

        (pager, blocks)
    }

    #[test]
    fn test_initial_placement() {
        let (mut pager, blocks) = setup();

        let clusters = vec![
            WeightCluster {
                id: 0,
                weight_ids: vec![0, 1, 2, 3],
                mean_importance: 0.9,
                dominant_domains: vec!["math".into()],
                cohesion: 0.8,
            },
            WeightCluster {
                id: 1,
                weight_ids: vec![4, 5, 6, 7],
                mean_importance: 0.7,
                dominant_domains: vec!["code".into()],
                cohesion: 0.7,
            },
        ];

        let partition = TierPartition {
            hot: vec![0, 1, 2, 3],
            warm: vec![4, 5, 6, 7],
            cold: vec![],
        };

        pager.initialize_placement(&clusters, &blocks, &partition);

        // Hot cluster should be on GPU.
        let tier = pager.access_cluster(0);
        assert_eq!(tier, TierLevel::Gpu);
        assert_eq!(pager.page_hits(), 1);
        assert_eq!(pager.page_faults(), 0);
    }

    #[test]
    fn test_page_fault_on_cold_access() {
        let (mut pager, blocks) = setup();

        let clusters = vec![WeightCluster {
            id: 0,
            weight_ids: vec![0, 1],
            mean_importance: 0.1,
            dominant_domains: vec![],
            cohesion: 0.5,
        }];

        let partition = TierPartition {
            hot: vec![],
            warm: vec![],
            cold: vec![0, 1],
        };

        pager.initialize_placement(&clusters, &blocks, &partition);

        let tier = pager.access_cluster(0);
        assert_eq!(tier, TierLevel::Ssd);
        assert_eq!(pager.page_faults(), 1);
    }
}
