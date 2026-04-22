use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TierError {
    #[error("weight block {0} not found in any tier")]
    BlockNotFound(u64),
    #[error("tier {0:?} is full (capacity: {1} bytes, used: {2} bytes)")]
    TierFull(TierLevel, usize, usize),
    #[error("IO error during tier operation: {0}")]
    Io(#[from] std::io::Error),
}

/// Memory tier levels ordered by access speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TierLevel {
    /// GPU VRAM — fastest, most constrained.
    Gpu,
    /// System RAM — moderate speed, larger capacity.
    Ram,
    /// SSD/NVMe — slowest, largest capacity.
    Ssd,
}

impl TierLevel {
    pub fn rank(&self) -> u8 {
        match self {
            TierLevel::Gpu => 0,
            TierLevel::Ram => 1,
            TierLevel::Ssd => 2,
        }
    }

    pub fn slower(&self) -> Option<TierLevel> {
        match self {
            TierLevel::Gpu => Some(TierLevel::Ram),
            TierLevel::Ram => Some(TierLevel::Ssd),
            TierLevel::Ssd => None,
        }
    }

    pub fn faster(&self) -> Option<TierLevel> {
        match self {
            TierLevel::Gpu => None,
            TierLevel::Ram => Some(TierLevel::Gpu),
            TierLevel::Ssd => Some(TierLevel::Ram),
        }
    }
}

/// Configuration for a single memory tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierSpec {
    pub level: TierLevel,
    pub capacity_bytes: usize,
    /// For SSD tier: path to the backing store directory.
    pub backing_path: Option<PathBuf>,
}

/// Configuration for the complete 3-tier memory hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    pub gpu: TierSpec,
    pub ram: TierSpec,
    pub ssd: TierSpec,
}

impl TierConfig {
    pub fn new(gpu_bytes: usize, ram_bytes: usize, ssd_bytes: usize, ssd_path: PathBuf) -> Self {
        TierConfig {
            gpu: TierSpec {
                level: TierLevel::Gpu,
                capacity_bytes: gpu_bytes,
                backing_path: None,
            },
            ram: TierSpec {
                level: TierLevel::Ram,
                capacity_bytes: ram_bytes,
                backing_path: None,
            },
            ssd: TierSpec {
                level: TierLevel::Ssd,
                capacity_bytes: ssd_bytes,
                backing_path: Some(ssd_path),
            },
        }
    }

    pub fn spec(&self, level: TierLevel) -> &TierSpec {
        match level {
            TierLevel::Gpu => &self.gpu,
            TierLevel::Ram => &self.ram,
            TierLevel::Ssd => &self.ssd,
        }
    }
}

/// A block of weights that can be placed in a tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBlock {
    pub id: u64,
    pub layer_index: usize,
    pub offset: usize,
    pub size_bytes: usize,
    pub importance: f64,
}

/// Tracks where each weight block currently resides and manages migrations.
pub struct TierManager {
    config: TierConfig,
    placements: HashMap<u64, TierLevel>,
    tier_usage: HashMap<TierLevel, usize>,
    migration_count: u64,
}

impl TierManager {
    pub fn new(config: TierConfig) -> Self {
        let mut tier_usage = HashMap::new();
        tier_usage.insert(TierLevel::Gpu, 0);
        tier_usage.insert(TierLevel::Ram, 0);
        tier_usage.insert(TierLevel::Ssd, 0);

        TierManager {
            config,
            placements: HashMap::new(),
            tier_usage,
            migration_count: 0,
        }
    }

    /// Place a weight block into the best available tier based on importance.
    pub fn place(&mut self, block: &WeightBlock, target: TierLevel) -> Result<TierLevel, TierError> {
        let cap = self.config.spec(target).capacity_bytes;
        let used = *self.tier_usage.get(&target).unwrap_or(&0);

        if used + block.size_bytes <= cap {
            self.placements.insert(block.id, target);
            *self.tier_usage.entry(target).or_insert(0) += block.size_bytes;
            return Ok(target);
        }

        // Fall through to a slower tier.
        if let Some(slower) = target.slower() {
            return self.place(block, slower);
        }

        Err(TierError::TierFull(target, cap, used))
    }

    /// Migrate a block to a faster tier (promote) if space is available.
    pub fn promote(&mut self, block: &WeightBlock) -> Result<Option<TierLevel>, TierError> {
        let current = self
            .placements
            .get(&block.id)
            .copied()
            .ok_or(TierError::BlockNotFound(block.id))?;

        let faster = match current.faster() {
            Some(f) => f,
            None => return Ok(None), // Already on GPU.
        };

        let cap = self.config.spec(faster).capacity_bytes;
        let used = *self.tier_usage.get(&faster).unwrap_or(&0);

        if used + block.size_bytes <= cap {
            // Remove from current tier.
            *self.tier_usage.entry(current).or_insert(0) -= block.size_bytes;
            // Place in faster tier.
            *self.tier_usage.entry(faster).or_insert(0) += block.size_bytes;
            self.placements.insert(block.id, faster);
            self.migration_count += 1;
            Ok(Some(faster))
        } else {
            Ok(None) // No space in faster tier.
        }
    }

    /// Demote a block to a slower tier.
    pub fn demote(&mut self, block: &WeightBlock) -> Result<Option<TierLevel>, TierError> {
        let current = self
            .placements
            .get(&block.id)
            .copied()
            .ok_or(TierError::BlockNotFound(block.id))?;

        let slower = match current.slower() {
            Some(s) => s,
            None => return Ok(None), // Already on SSD.
        };

        let cap = self.config.spec(slower).capacity_bytes;
        let used = *self.tier_usage.get(&slower).unwrap_or(&0);

        if used + block.size_bytes <= cap {
            *self.tier_usage.entry(current).or_insert(0) -= block.size_bytes;
            *self.tier_usage.entry(slower).or_insert(0) += block.size_bytes;
            self.placements.insert(block.id, slower);
            self.migration_count += 1;
            Ok(Some(slower))
        } else {
            Err(TierError::TierFull(slower, cap, used))
        }
    }

    pub fn get_tier(&self, block_id: u64) -> Option<TierLevel> {
        self.placements.get(&block_id).copied()
    }

    pub fn migration_count(&self) -> u64 {
        self.migration_count
    }

    pub fn usage(&self, level: TierLevel) -> usize {
        *self.tier_usage.get(&level).unwrap_or(&0)
    }

    pub fn capacity(&self, level: TierLevel) -> usize {
        self.config.spec(level).capacity_bytes
    }

    pub fn utilization(&self, level: TierLevel) -> f64 {
        let cap = self.capacity(level);
        if cap == 0 {
            return 0.0;
        }
        self.usage(level) as f64 / cap as f64
    }

    /// Return all block IDs currently placed at the given tier.
    pub fn blocks_at(&self, level: TierLevel) -> Vec<u64> {
        self.placements
            .iter()
            .filter(|(_, &v)| v == level)
            .map(|(&k, _)| k)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_config() -> TierConfig {
        TierConfig::new(1000, 5000, 50000, PathBuf::from("/tmp/nve_test"))
    }

    #[test]
    fn test_place_and_promote() {
        let mut mgr = TierManager::new(test_config());
        let block = WeightBlock {
            id: 1,
            layer_index: 0,
            offset: 0,
            size_bytes: 500,
            importance: 0.9,
        };

        // Place on RAM first.
        let tier = mgr.place(&block, TierLevel::Ram).unwrap();
        assert_eq!(tier, TierLevel::Ram);

        // Promote to GPU.
        let result = mgr.promote(&block).unwrap();
        assert_eq!(result, Some(TierLevel::Gpu));
        assert_eq!(mgr.get_tier(1), Some(TierLevel::Gpu));
    }

    #[test]
    fn test_overflow_to_slower_tier() {
        let mut mgr = TierManager::new(test_config());
        // Fill GPU completely.
        let big_block = WeightBlock {
            id: 1,
            layer_index: 0,
            offset: 0,
            size_bytes: 1000,
            importance: 0.95,
        };
        mgr.place(&big_block, TierLevel::Gpu).unwrap();

        // Next block should overflow to RAM.
        let overflow = WeightBlock {
            id: 2,
            layer_index: 1,
            offset: 0,
            size_bytes: 100,
            importance: 0.5,
        };
        let tier = mgr.place(&overflow, TierLevel::Gpu).unwrap();
        assert_eq!(tier, TierLevel::Ram);
    }
}
