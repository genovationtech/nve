//! # Neural Virtualization Engine (NVE)
//!
//! Monte Carlo Guided Virtual Weight Paging for Neural Networks.
//!
//! NVE uses Monte Carlo sampling to estimate weight importance distributions,
//! then virtualizes weights across memory tiers (GPU / RAM / SSD) — dynamically
//! loading weight clusters based on inferred activation patterns.
//!
//! Supports loading and serving any HuggingFace transformer model.
//!
//! ## Core modules
//!
//! - [`arch`] — Architecture detection and unified configuration
//! - [`generic_model`] — Architecture-generic transformer inference
//! - [`hub`] — HuggingFace Hub model download
//! - [`weight_map`] — Per-architecture weight name mapping
//! - [`profiler`] — Monte Carlo Activation Profiler (MCAP)
//! - [`cluster`] — Co-activation weight clustering
//! - [`tier`] — Memory tier management
//! - [`pager`] — Virtual weight paging runtime
//! - [`tensor`] — Native f32 tensor types with SIMD ops
//! - [`safetensors`] — Zero-copy model weight loading
//! - [`config`] — Llama model configuration (legacy)
//! - [`tokenizer`] — BPE tokenizer
//! - [`ops`] — Neural network ops (RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, etc.)
//! - [`attention`] — Grouped-query attention with KV cache
//! - [`model`] — Llama inference engine (legacy)
//! - [`paged_model`] — NVE tier-aware paged inference
//! - [`benchmark`] — Performance benchmarking

pub mod arch;
pub mod generic_model;
pub mod hub;
pub mod weight_map;

pub mod cluster;
pub mod pager;
pub mod profiler;
pub mod tier;

pub mod tensor;
pub mod safetensors;
pub mod config;
pub mod tokenizer;
pub mod ops;
pub mod attention;
pub mod model;
pub mod paged_model;
pub mod benchmark;
pub mod quantize;
pub mod abc_benchmark;
pub mod importance_cache;
pub mod device;
pub mod gpu_layer;
pub mod cuda_kernels;
pub mod decode_graph;

use std::collections::HashMap;
use std::path::PathBuf;

use cluster::{Clusterer, CoActivationTracker, WeightCluster};
use pager::{Pager, PagerConfig};
use profiler::{ActivationSample, Profiler, ProfilerConfig};
use tier::{TierConfig, TierLevel, TierManager, WeightBlock};

use serde::{Deserialize, Serialize};

/// Top-level engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub profiler: ProfilerConfig,
    pub pager: PagerConfig,
    pub tier: TierConfig,
    pub cluster_pmi_threshold: f64,
    pub cluster_max_size: usize,
    pub co_activation_threshold: f64,
    pub max_tracked_weights: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            profiler: ProfilerConfig::default(),
            pager: PagerConfig::default(),
            tier: TierConfig::new(
                4 * 1024 * 1024 * 1024,  // 4 GB GPU
                16 * 1024 * 1024 * 1024, // 16 GB RAM
                128 * 1024 * 1024 * 1024, // 128 GB SSD
                PathBuf::from("/tmp/nve_weights"),
            ),
            cluster_pmi_threshold: 0.5,
            cluster_max_size: 256,
            co_activation_threshold: 0.3,
            max_tracked_weights: 10000,
        }
    }
}

/// The NVE engine — main entry point.
///
/// Lifecycle:
/// 1. Create engine with config
/// 2. Register weight blocks
/// 3. Run profiling rounds with activation samples
/// 4. Build clusters and initial placement
/// 5. Run inference with automatic paging
pub struct Engine {
    config: EngineConfig,
    profiler: Profiler,
    co_tracker: CoActivationTracker,
    clusters: Vec<WeightCluster>,
    blocks: Vec<WeightBlock>,
    pager: Option<Pager>,
    state: EngineState,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EngineState {
    /// Accepting weight registrations.
    Setup,
    /// Running profiling passes.
    Profiling,
    /// Clusters built, placement done, ready for inference.
    Ready,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        let profiler = Profiler::new(config.profiler.clone());
        let co_tracker = CoActivationTracker::new(config.max_tracked_weights);

        Engine {
            config,
            profiler,
            co_tracker,
            clusters: Vec::new(),
            blocks: Vec::new(),
            pager: None,
            state: EngineState::Setup,
        }
    }

    /// Register a weight block with the engine.
    pub fn register_block(&mut self, block: WeightBlock) {
        assert_eq!(self.state, EngineState::Setup, "Cannot register blocks after profiling has started");
        self.blocks.push(block);
    }

    /// Register multiple weight blocks.
    pub fn register_blocks(&mut self, blocks: Vec<WeightBlock>) {
        for block in blocks {
            self.register_block(block);
        }
    }

    /// Transition to profiling state.
    pub fn start_profiling(&mut self) {
        self.state = EngineState::Profiling;
    }

    /// Feed activation samples from a single forward pass.
    pub fn record_activations(
        &mut self,
        samples: Vec<ActivationSample>,
        activation_threshold: Option<f64>,
    ) {
        assert_eq!(self.state, EngineState::Profiling, "Must be in profiling state");

        let threshold = activation_threshold.unwrap_or(self.config.co_activation_threshold);

        // Track which weights were active for co-activation.
        let active: Vec<u64> = samples
            .iter()
            .filter(|s| s.magnitude >= threshold)
            .map(|s| s.weight_id)
            .collect();
        self.co_tracker.record_pass(&active);

        // Record for importance profiling.
        self.profiler.record_batch(samples);
    }

    /// Finish a profiling round.
    pub fn finish_profiling_round(&mut self) {
        self.profiler.finish_round();
    }

    /// Build clusters and initialize weight placement.
    /// Call this after profiling is complete.
    pub fn build(&mut self) {
        assert_eq!(self.state, EngineState::Profiling, "Must finish profiling first");

        // Build importance map.
        let importances: HashMap<u64, f64> = self
            .blocks
            .iter()
            .map(|b| {
                let imp = self.profiler.get_importance(b.id).unwrap_or(0.0);
                (b.id, imp)
            })
            .collect();

        // Cluster weights by co-activation.
        let clusterer = Clusterer::new(
            self.config.cluster_pmi_threshold,
            self.config.cluster_max_size,
        );
        self.clusters = clusterer.cluster(&self.co_tracker, &importances);

        // Partition into tiers.
        let partition = self.profiler.partition(
            self.config.pager.gpu_fraction,
            self.config.pager.ram_fraction,
        );

        // Initialize the pager.
        let tier_mgr = TierManager::new(self.config.tier.clone());
        let mut pager = Pager::new(self.config.pager.clone(), tier_mgr);
        pager.initialize_placement(&self.clusters, &self.blocks, &partition);
        self.pager = Some(pager);

        self.state = EngineState::Ready;
    }

    /// Access a cluster at runtime — returns which tier it's on.
    pub fn access_cluster(&mut self, cluster_id: u32) -> TierLevel {
        assert_eq!(self.state, EngineState::Ready, "Engine not ready");
        self.pager.as_mut().unwrap().access_cluster(cluster_id)
    }

    /// Promote a cluster to a faster tier.
    pub fn promote_cluster(&mut self, cluster_id: u32) -> Vec<(u64, Option<TierLevel>)> {
        assert_eq!(self.state, EngineState::Ready);
        self.pager
            .as_mut()
            .unwrap()
            .promote_cluster(cluster_id, &self.blocks)
    }

    /// Evict the least recently used cluster from GPU.
    pub fn evict_lru(&mut self) -> Option<u32> {
        assert_eq!(self.state, EngineState::Ready);
        self.pager
            .as_mut()
            .unwrap()
            .evict_lru_from_gpu(&self.blocks)
    }

    // ── Getters ──

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn profiling_rounds(&self) -> u64 {
        self.profiler.total_rounds()
    }

    pub fn is_ready(&self) -> bool {
        self.state == EngineState::Ready
    }

    pub fn page_fault_rate(&self) -> f64 {
        self.pager.as_ref().map(|p| p.page_fault_rate()).unwrap_or(0.0)
    }

    pub fn migration_count(&self) -> u64 {
        self.pager.as_ref().map(|p| p.migration_count()).unwrap_or(0)
    }

    pub fn importance_ranking(&self) -> Vec<(u64, f64)> {
        self.profiler.importance_ranking()
    }

    pub fn clusters(&self) -> &[WeightCluster] {
        &self.clusters
    }

    pub fn tier_utilization(&self, level: TierLevel) -> f64 {
        self.pager
            .as_ref()
            .map(|p| p.tier_utilization(level))
            .unwrap_or(0.0)
    }
}

// ── C FFI for Python bindings ──

/// Opaque handle for FFI.
pub type EngineHandle = *mut Engine;

/// Create a new engine with default config. Caller must free with `nve_engine_free`.
#[no_mangle]
pub extern "C" fn nve_engine_new() -> EngineHandle {
    let engine = Box::new(Engine::new(EngineConfig::default()));
    Box::into_raw(engine)
}

/// Free an engine.
///
/// # Safety
/// `handle` must be a valid pointer returned by `nve_engine_new`.
#[no_mangle]
pub unsafe extern "C" fn nve_engine_free(handle: EngineHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Register a weight block.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_register_block(
    handle: EngineHandle,
    id: u64,
    layer_index: usize,
    offset: usize,
    size_bytes: usize,
    importance: f64,
) {
    let engine = &mut *handle;
    engine.register_block(WeightBlock {
        id,
        layer_index,
        offset,
        size_bytes,
        importance,
    });
}

/// Start profiling mode.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_start_profiling(handle: EngineHandle) {
    let engine = &mut *handle;
    engine.start_profiling();
}

/// Record a single activation sample.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_record_activation(
    handle: EngineHandle,
    weight_id: u64,
    magnitude: f64,
) {
    let engine = &mut *handle;
    engine.record_activations(
        vec![ActivationSample {
            weight_id,
            magnitude,
            prompt_domain: None,
        }],
        None,
    );
}

/// Build clusters and initialize placement.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_build(handle: EngineHandle) {
    let engine = &mut *handle;
    engine.build();
}

/// Get the page fault rate.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_page_fault_rate(handle: EngineHandle) -> f64 {
    let engine = &*handle;
    engine.page_fault_rate()
}

/// Get the number of clusters.
///
/// # Safety
/// `handle` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn nve_cluster_count(handle: EngineHandle) -> usize {
    let engine = &*handle;
    engine.cluster_count()
}

// ── C FFI for GenericModel (fast Rust inference) ──

use std::ffi::CStr;
use std::os::raw::c_char;
use generic_model::GenericModel;
use tokenizer::Tokenizer;

/// Opaque handle for a loaded model.
pub type ModelHandle = *mut (GenericModel, Tokenizer);

/// Load a model from a directory path. Returns null on failure.
///
/// # Safety
/// `model_dir` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn nve_model_load(model_dir: *const c_char) -> ModelHandle {
    let c_str = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let path = std::path::Path::new(c_str);

    let model = match GenericModel::from_dir(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[NVE] Failed to load model: {}", e);
            return std::ptr::null_mut();
        }
    };

    let tokenizer = match Tokenizer::from_model_dir(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[NVE] Failed to load tokenizer: {}", e);
            return std::ptr::null_mut();
        }
    };

    Box::into_raw(Box::new((model, tokenizer)))
}

/// Free a loaded model.
///
/// # Safety
/// `handle` must be a valid pointer from `nve_model_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn nve_model_free(handle: ModelHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Result struct returned by nve_model_generate.
/// Caller must free `tokens` with nve_tokens_free and `text` with nve_text_free.
#[repr(C)]
pub struct NveGenerateResult {
    pub tokens: *mut u32,
    pub num_tokens: usize,
    pub text: *mut c_char,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_sec: f64,
    pub prompt_tokens: usize,
}

/// Generate text from a prompt. Returns result struct.
///
/// # Safety
/// `handle` must be a valid model pointer. `prompt` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nve_model_generate(
    handle: ModelHandle,
    prompt: *const c_char,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> NveGenerateResult {
    let empty = NveGenerateResult {
        tokens: std::ptr::null_mut(),
        num_tokens: 0,
        text: std::ptr::null_mut(),
        prefill_time_ms: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
        tokens_per_sec: 0.0,
        prompt_tokens: 0,
    };

    if handle.is_null() {
        return empty;
    }

    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return empty,
    };

    let (model, tokenizer) = &mut *handle;
    model.reset();

    let prompt_tokens = tokenizer.encode_with_bos(prompt_str);
    if prompt_tokens.is_empty() {
        return empty;
    }

    let result = model.generate(&prompt_tokens, max_new_tokens, temperature, top_p);

    // Decode all tokens (prompt + generated).
    let all_tokens: Vec<u32> = prompt_tokens.iter().copied()
        .chain(result.tokens.iter().copied())
        .collect();
    let text = tokenizer.decode(&all_tokens);

    // Allocate C-compatible outputs.
    let mut token_buf = result.tokens.into_boxed_slice();
    let tokens_ptr = token_buf.as_mut_ptr();
    let num_tokens = token_buf.len();
    std::mem::forget(token_buf);

    let c_text = match std::ffi::CString::new(text) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    };

    NveGenerateResult {
        tokens: tokens_ptr,
        num_tokens,
        text: c_text,
        prefill_time_ms: result.prefill_time_ms,
        decode_time_ms: result.decode_time_ms,
        total_time_ms: result.total_time_ms,
        tokens_per_sec: result.tokens_per_sec,
        prompt_tokens: result.prompt_tokens,
    }
}

/// Free token array returned by nve_model_generate.
///
/// # Safety
/// `ptr` must be from NveGenerateResult.tokens, or null.
#[no_mangle]
pub unsafe extern "C" fn nve_tokens_free(ptr: *mut u32, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Free text string returned by nve_model_generate.
///
/// # Safety
/// `ptr` must be from NveGenerateResult.text, or null.
#[no_mangle]
pub unsafe extern "C" fn nve_text_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(std::ffi::CString::from_raw(ptr));
    }
}

// ── C FFI for PagedModel ──────────────────────────────────────────────────────

/// Opaque handle: `Box<(PagedModel, Tokenizer)>` raw pointer.
pub type PagedModelHandle = *mut (paged_model::PagedModel, tokenizer::Tokenizer);

/// Load a paged model. Returns null on failure.
///
/// # Safety
/// `model_dir`, `quant_mode`, and `device` must be valid null-terminated UTF-8 strings.
///
/// `device` controls where hot-tier weights are placed:
///   - `"auto"` / null  → best available device (CUDA → HIP → Metal → Vulkan → CPU)
///   - `"cpu"`          → CPU-only (default when no GPU feature compiled in)
///   - `"cuda:N"`       → NVIDIA GPU N (requires `--features cuda`)
///   - `"hip:N"`        → AMD GPU N via ROCm (requires `--features hip`)
///   - `"metal"`        → Apple GPU (requires `--features metal`)
///   - `"vulkan:N"`     → Vulkan GPU N (requires `--features vulkan`)
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_load(
    model_dir: *const c_char,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: *const c_char,
    device: *const c_char,
) -> PagedModelHandle {
    let dir_str = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let qmode_str = if quant_mode.is_null() {
        "none"
    } else {
        match CStr::from_ptr(quant_mode).to_str() {
            Ok(s) => s,
            Err(_) => "none",
        }
    };
    let device_str = if device.is_null() {
        "auto"
    } else {
        match CStr::from_ptr(device).to_str() {
            Ok(s) => s,
            Err(_) => "auto",
        }
    };

    // Resolve and log the device. The resolved device is available for future
    // GPU dispatch wiring (paged_model does CPU inference today; GPU dispatch
    // is gated on candle feature flags being compiled in).
    let resolved_device = crate::device::Device::resolve(device_str);
    if resolved_device.is_gpu() {
        log::info!(
            "[NVE paged] Using device: {} (GPU hot-tier dispatch pending candle integration)",
            resolved_device
        );
    } else {
        log::debug!("[NVE paged] Using device: {}", resolved_device);
    }

    let path = std::path::Path::new(dir_str);
    let qmode = quantize::QuantMode::from_str(qmode_str).unwrap_or(quantize::QuantMode::None);

    let cfg = paged_model::PagedConfig {
        hot_budget_bytes: hot_budget_mb * 1024 * 1024,
        warm_budget_bytes: warm_budget_mb * 1024 * 1024,
        prefetch_ahead: 2,
        profile_activations: false,
        hot_only_mode: false,
        active_layers: None,
        quant_mode: qmode,
        device: device_str.to_string(),
    };

    let paged = match paged_model::PagedModel::from_dir(path, cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[NVE paged] Failed to load model: {}", e);
            return std::ptr::null_mut();
        }
    };
    let tok = match tokenizer::Tokenizer::from_model_dir(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[NVE paged] Failed to load tokenizer: {}", e);
            return std::ptr::null_mut();
        }
    };

    Box::into_raw(Box::new((paged, tok)))
}

/// Free a paged model handle.
///
/// # Safety
/// `handle` must be from `nve_paged_model_load`, or null.
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_free(handle: PagedModelHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Generate text with the paged model.
///
/// # Safety
/// `handle` must be valid. `prompt` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_generate(
    handle: PagedModelHandle,
    prompt: *const c_char,
    max_new_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> NveGenerateResult {
    let empty = NveGenerateResult {
        tokens: std::ptr::null_mut(),
        num_tokens: 0,
        text: std::ptr::null_mut(),
        prefill_time_ms: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
        tokens_per_sec: 0.0,
        prompt_tokens: 0,
    };
    if handle.is_null() {
        return empty;
    }
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return empty,
    };

    let (paged, tok) = &mut *handle;
    paged.reset();
    let prompt_tokens = tok.encode_with_bos(prompt_str);
    if prompt_tokens.is_empty() {
        return empty;
    }

    let result = match paged.generate(&prompt_tokens, max_new_tokens, temperature, top_p) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[NVE paged] generate error: {}", e);
            return empty;
        }
    };

    let all_tokens: Vec<u32> = prompt_tokens.iter().copied()
        .chain(result.tokens.iter().copied())
        .collect();
    let text = tok.decode(&all_tokens);

    let mut token_buf = result.tokens.into_boxed_slice();
    let tokens_ptr = token_buf.as_mut_ptr();
    let num_tokens = token_buf.len();
    std::mem::forget(token_buf);

    let c_text = match std::ffi::CString::new(text) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    };

    NveGenerateResult {
        tokens: tokens_ptr,
        num_tokens,
        text: c_text,
        prefill_time_ms: result.prefill_time_ms,
        decode_time_ms: result.decode_time_ms,
        total_time_ms: result.total_time_ms,
        tokens_per_sec: result.tokens_per_sec,
        prompt_tokens: result.prompt_tokens,
    }
}

/// Return a memory report string for the paged model. Caller frees with `nve_text_free`.
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_memory_report(handle: PagedModelHandle) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    let (paged, _) = &*handle;
    let report = paged.memory_report();
    match std::ffi::CString::new(report) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Profile layer importance. `probe_prompt` is tokenized and used as the probe.
/// `active_layers` = 0 means auto-compute from budget.
/// Returns true on success.
///
/// # Safety
/// `handle` must be valid. `probe_prompt` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_profile(
    handle: PagedModelHandle,
    probe_prompt: *const c_char,
    active_layers: usize,
) -> bool {
    if handle.is_null() {
        return false;
    }
    let prompt_str = if probe_prompt.is_null() {
        "The quick brown fox jumps over the lazy dog"
    } else {
        match CStr::from_ptr(probe_prompt).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let (paged, tok) = &mut *handle;
    let probe_tokens = tok.encode_with_bos(prompt_str);
    let capped: Vec<u32> = probe_tokens.into_iter().take(32).collect();

    if paged.profile_layer_importance(&capped).is_err() {
        return false;
    }

    let count = if active_layers == 0 {
        let layer_size = paged.layer_size_bytes();
        let budget_bytes = paged.num_layers() * layer_size; // use all layers as fallback
        (budget_bytes / layer_size.max(1)).min(paged.num_layers())
    } else {
        active_layers
    };
    paged.apply_profiled_hot_only(count);
    // Initialize GPU inference for hot layers (no-op on CPU-only builds).
    if paged.init_gpu().is_err() {
        log::warn!("[NVE paged] GPU init failed — continuing with CPU inference");
    }
    true
}

/// Reset the paged model's KV cache and position counter.
///
/// # Safety
/// `handle` must be valid.
#[no_mangle]
pub unsafe extern "C" fn nve_paged_model_reset(handle: PagedModelHandle) {
    if !handle.is_null() {
        let (paged, _) = &mut *handle;
        paged.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_engine_lifecycle() {
        let mut engine = Engine::new(EngineConfig {
            tier: TierConfig::new(
                4096,
                16384,
                65536,
                PathBuf::from("/tmp/nve_test_engine"),
            ),
            ..Default::default()
        });

        // Register weight blocks.
        for i in 0..20 {
            engine.register_block(WeightBlock {
                id: i,
                layer_index: i as usize / 4,
                offset: (i as usize % 4) * 64,
                size_bytes: 64,
                importance: 0.0,
            });
        }

        // Profile.
        engine.start_profiling();

        for round in 0..10 {
            let mut samples = Vec::new();
            for wid in 0..20u64 {
                // Weights 0-4 always activate strongly (math cluster).
                // Weights 5-9 activate moderately (code cluster).
                // Weights 10-19 rarely activate.
                let mag = if wid < 5 {
                    0.9
                } else if wid < 10 {
                    0.5
                } else {
                    0.05
                };
                samples.push(ActivationSample {
                    weight_id: wid,
                    magnitude: mag,
                    prompt_domain: Some("test".into()),
                });
            }
            engine.record_activations(samples, None);
            engine.finish_profiling_round();
        }

        // Build.
        engine.build();
        assert!(engine.is_ready());
        assert!(engine.cluster_count() > 0);

        // Access clusters.
        let _tier = engine.access_cluster(0);

        // Check rankings.
        let ranking = engine.importance_ranking();
        assert!(!ranking.is_empty());
        // Top-ranked weights should be the high-activation ones.
        assert!(ranking[0].1 > ranking.last().unwrap().1);
    }
}
