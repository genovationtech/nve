//! ABC Benchmark — Detailed comparison of NVE inference strategies.
//!
//! Runs four configurations on actual models and produces detailed reports:
//!
//! - **Baseline**: Full model, bf16, all layers (reference)
//! - **A — Quantization Only**: Full model loaded, uniform Q4 applied to all layers
//! - **B — Profiled Hot-Only (bf16)**: Streaming profiler selects best layers, no quantization
//! - **C — Profiled + Quantization**: Profile-guided mixed-precision with AWQ saliency
//!
//! Metrics collected per configuration:
//! - Latency: prefill time, decode ms/token
//! - Throughput: tokens/sec
//! - Memory: peak RSS, loaded layers, compression ratio
//! - Quality: output text, repetition score, unique n-gram ratio
//! - Paging stats (B/C): page faults, layer loads, profiling time

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::benchmark::PeakMemoryTracker;
use crate::generic_model::GenericModel;
use crate::model::{GenerationResult, ModelError};
use crate::paged_model::{PagedConfig, PagedModel, PagingStats, ScoringComparison};
use crate::quantize::QuantMode;
use crate::tokenizer::Tokenizer;

// ── Configuration ───────────────────────────────────────────────────────────

/// Which ABC configurations to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbcConfig {
    Baseline,
    /// A: Quantization only (uniform Q4, all layers, no profiling)
    QuantOnly,
    /// B: Profiled hot-only (bf16, best layers selected by importance)
    ProfiledHotOnly,
    /// C: Profiled + quantization (profile-guided mixed-precision + AWQ)
    ProfiledQuantized,
}

impl AbcConfig {
    pub fn label(&self) -> &'static str {
        match self {
            AbcConfig::Baseline => "Baseline (bf16, all layers)",
            AbcConfig::QuantOnly => "A: Quantization Only (Q4, all layers)",
            AbcConfig::ProfiledHotOnly => "B: Profiled Hot-Only (bf16, best layers)",
            AbcConfig::ProfiledQuantized => "C: Profiled + Quantized (PG + AWQ)",
        }
    }

    pub fn short(&self) -> &'static str {
        match self {
            AbcConfig::Baseline => "baseline",
            AbcConfig::QuantOnly => "A_quant_only",
            AbcConfig::ProfiledHotOnly => "B_profiled_hot",
            AbcConfig::ProfiledQuantized => "C_profiled_quant",
        }
    }

    pub fn all() -> Vec<AbcConfig> {
        vec![
            AbcConfig::Baseline,
            AbcConfig::QuantOnly,
            AbcConfig::ProfiledHotOnly,
            AbcConfig::ProfiledQuantized,
        ]
    }
}

/// Per-mode minimum acceptable task accuracy.  If a configuration falls
/// below its threshold the benchmark prints a FAIL annotation.
#[derive(Debug, Clone)]
pub struct QualityBudget {
    /// Minimum absolute pass-rate for hot-only mode (config B).
    pub hot_only_min_task_accuracy: f64,
    /// Minimum absolute pass-rate for profiled+quantized mode (config C).
    pub quant_min_task_accuracy: f64,
}

impl Default for QualityBudget {
    fn default() -> Self {
        QualityBudget {
            hot_only_min_task_accuracy: 0.5,
            quant_min_task_accuracy: 0.4,
        }
    }
}

/// Parameters for the ABC benchmark suite.
#[derive(Debug, Clone)]
pub struct AbcBenchmarkParams {
    /// Prompts to test across all configurations.
    pub prompts: Vec<String>,
    /// Max tokens to generate per prompt.
    pub max_new_tokens: usize,
    /// Temperature (0 = greedy for reproducibility).
    pub temperature: f32,
    /// Top-p for nucleus sampling.
    pub top_p: f32,
    /// Hot memory budget in bytes (for B and C).
    pub hot_budget_bytes: usize,
    /// Warm memory budget in bytes (for B and C).
    pub warm_budget_bytes: usize,
    /// Target bits-per-weight for config C (default 2.0).
    pub target_bpw: f32,
    /// Which configurations to run.
    pub configs: Vec<AbcConfig>,
    /// Override number of active layers for B/C (None = auto from budget).
    pub active_layers: Option<usize>,
    /// Run a split-half stability check during profiling.  Profiles the token
    /// corpus in two halves and reports top-k layer overlap.  Adds ~2× profiling
    /// time; disable for fast runs.
    pub stability_check: bool,
    /// Acceptance thresholds for task accuracy per mode.
    pub quality_budget: QualityBudget,
    /// Compute device for GPU-accelerated profiling matmuls.
    /// Passed directly to `PagedConfig::device`.  "auto" selects the best
    /// available device (CUDA → HIP → Metal → CPU).
    pub device: String,
    /// Path to a portable importance-profile JSON (a plain `[f64, …]` array).
    /// When set, configs B and C skip the profiling forward-pass entirely and use
    /// these pre-computed layer-importance scores instead.  Enables the
    /// "profile once on a large machine, run optimally on a small device"
    /// workflow.  Generate with `nve abc-test --save-profile`.
    pub profile_path: Option<std::path::PathBuf>,
}

impl Default for AbcBenchmarkParams {
    fn default() -> Self {
        AbcBenchmarkParams {
            prompts: vec![
                "The theory of general relativity explains that".into(),
                "The three branches of the United States government are".into(),
                "Photosynthesis is the process by which plants".into(),
                "def fibonacci(n):".into(),
                "In machine learning, gradient descent is used to".into(),
            ],
            max_new_tokens: 50,
            temperature: 0.0,
            top_p: 1.0,
            hot_budget_bytes: 512 * 1024 * 1024,
            warm_budget_bytes: 1536 * 1024 * 1024,
            target_bpw: 2.0,
            configs: AbcConfig::all(),
            active_layers: None,
            stability_check: false,
            quality_budget: QualityBudget::default(),
            device: "auto".to_string(),
            profile_path: None,
        }
    }
}

// ── Result types ────────────────────────────────────────────────────────────

/// A single task-evaluation item: a prompt with a known expected answer substring.
#[derive(Debug, Clone)]
pub struct TaskItem {
    pub category: &'static str,
    pub prompt: &'static str,
    /// Case-insensitive substring that must appear in the generated text to pass.
    pub expected: &'static str,
}

/// Fixed internal task suite covering QA, reasoning, coding, and summarization.
/// All prompts use greedy decoding (temperature = 0) for reproducibility.
pub const DEFAULT_TASK_SUITE: &[TaskItem] = &[
    // Factual QA
    TaskItem { category: "qa",          prompt: "The capital of France is",                                      expected: "paris"    },
    TaskItem { category: "qa",          prompt: "Water is composed of hydrogen and",                             expected: "oxygen"   },
    TaskItem { category: "qa",          prompt: "The largest planet in the solar system is",                     expected: "jupiter"  },
    // Simple reasoning
    TaskItem { category: "reasoning",   prompt: "If today is Monday, tomorrow is",                               expected: "tuesday"  },
    TaskItem { category: "reasoning",   prompt: "A square has four equal sides. A shape with four equal sides and four right angles is a", expected: "square" },
    // Coding completion
    TaskItem { category: "coding",      prompt: "def add(a, b):\n    return a",                                  expected: "+"        },
    TaskItem { category: "coding",      prompt: "# Python: list of squares 0-4\nsquares = [x**2 for x in",      expected: "range"    },
    // Summarization keyword
    TaskItem { category: "summarization", prompt: "The main benefit of regular exercise is improved",            expected: "health"   },
];

/// Result for a single task-evaluation item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub category: String,
    pub prompt: String,
    pub expected: String,
    pub generated: String,
    pub passed: bool,
}

/// Quality metrics for generated text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Ratio of unique bigrams to total bigrams (higher = less repetitive).
    pub unique_bigram_ratio: f64,
    /// Ratio of unique trigrams to total trigrams.
    pub unique_trigram_ratio: f64,
    /// Longest repeated consecutive token run.
    pub max_repeat_run: usize,
    /// Total generated tokens.
    pub token_count: usize,
    /// Whether the output ended naturally (EOS) vs hitting max tokens.
    pub hit_eos: bool,
}

/// Result for a single prompt under a single configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbcPromptResult {
    pub prompt: String,
    pub generated_text: String,
    pub generated_tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub num_generated: usize,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_sec: f64,
    pub ms_per_token: f64,
    pub quality: QualityMetrics,
}

/// Paging-specific stats for configs B and C.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbcPagingInfo {
    pub page_hits: u64,
    pub page_faults: u64,
    pub fault_rate_pct: f64,
    pub layers_loaded: u64,
    pub layers_evicted: u64,
    pub load_time_ms: f64,
    pub active_layers: usize,
    pub total_layers: usize,
    pub profiling_time_ms: f64,
    pub layer_importance_scores: Vec<f64>,
    pub layer_quant_assignments: Option<Vec<String>>,
    /// Fraction of top-k layers that matched between two prompt halves.
    /// None if stability_check was disabled or corpus was too small.
    pub layer_stability_top_k_overlap: Option<f64>,
    /// Side-by-side scorer comparison from the definitive profile pass.
    pub scorer_comparison: Option<ScoringComparison>,
}

/// Full result for one ABC configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbcConfigResult {
    pub config: String,
    pub label: String,
    pub results: Vec<AbcPromptResult>,
    pub summary: AbcSummary,
    pub paging_info: Option<AbcPagingInfo>,
    /// Results from the fixed task evaluation suite.
    pub task_results: Vec<TaskResult>,
    /// Fraction of task items that passed (0.0 – 1.0).
    pub task_accuracy: f64,
}

/// Summary statistics for one configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbcSummary {
    pub avg_prefill_ms: f64,
    pub avg_decode_ms_per_token: f64,
    pub avg_tokens_per_sec: f64,
    pub total_tokens_generated: usize,
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
    pub avg_unique_bigram_ratio: f64,
    pub avg_unique_trigram_ratio: f64,
    pub avg_max_repeat_run: f64,
}

/// The full ABC benchmark report.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbcReport {
    pub model_path: String,
    pub model_arch: String,
    pub model_params: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub timestamp: String,
    pub params: AbcReportParams,
    pub configurations: Vec<AbcConfigResult>,
    pub comparison: AbcComparison,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AbcReportParams {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub hot_budget_mb: usize,
    pub warm_budget_mb: usize,
    pub target_bpw: f32,
    pub num_prompts: usize,
    /// Resolved device string used for profiling matmuls (e.g. "cuda:0", "cpu").
    pub device: String,
}

/// Cross-configuration comparison.
#[derive(Debug, Serialize, Deserialize)]
pub struct AbcComparison {
    /// Speedup relative to baseline (tok/s ratio).
    pub speedup_vs_baseline: Vec<SpeedupEntry>,
    /// Memory savings relative to baseline.
    pub memory_savings_pct: Vec<MemorySavingsEntry>,
    /// Quality relative to baseline.
    pub quality_vs_baseline: Vec<QualityCompEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpeedupEntry {
    pub config: String,
    pub speedup: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemorySavingsEntry {
    pub config: String,
    pub savings_pct: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityCompEntry {
    pub config: String,
    /// Average unique bigram ratio relative to baseline (1.0 = same quality).
    pub bigram_ratio_vs_baseline: f64,
    pub trigram_ratio_vs_baseline: f64,
    /// Task accuracy ratio relative to baseline (1.0 = same accuracy).
    pub task_accuracy_vs_baseline: f64,
}

// ── Quality scoring ─────────────────────────────────────────────────────────

fn compute_quality(tokens: &[u32], max_new_tokens: usize) -> QualityMetrics {
    let n = tokens.len();

    // Unique bigram ratio
    let bigrams: Vec<(u32, u32)> = tokens.windows(2).map(|w| (w[0], w[1])).collect();
    let unique_bigrams: HashSet<(u32, u32)> = bigrams.iter().cloned().collect();
    let unique_bigram_ratio = if bigrams.is_empty() {
        1.0
    } else {
        unique_bigrams.len() as f64 / bigrams.len() as f64
    };

    // Unique trigram ratio
    let trigrams: Vec<(u32, u32, u32)> = tokens.windows(3).map(|w| (w[0], w[1], w[2])).collect();
    let unique_trigrams: HashSet<(u32, u32, u32)> = trigrams.iter().cloned().collect();
    let unique_trigram_ratio = if trigrams.is_empty() {
        1.0
    } else {
        unique_trigrams.len() as f64 / trigrams.len() as f64
    };

    // Max consecutive repeat run
    let mut max_run = 0usize;
    let mut current_run = 1usize;
    for i in 1..n {
        if tokens[i] == tokens[i - 1] {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 1;
        }
    }
    if n <= 1 {
        max_run = n;
    }

    let hit_eos = n < max_new_tokens;

    QualityMetrics {
        unique_bigram_ratio,
        unique_trigram_ratio,
        max_repeat_run: max_run,
        token_count: n,
        hit_eos,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Fraction of the top-k layers (by score) that appear in both score vectors.
/// k is clamped to [1, len].  Returns 1.0 if k == 0 or scores are empty.
fn top_k_overlap(scores_a: &[f64], scores_b: &[f64], k: usize) -> f64 {
    if scores_a.is_empty() || k == 0 { return 1.0; }
    let k = k.min(scores_a.len());
    let top_k = |scores: &[f64]| -> HashSet<usize> {
        let mut ranked: Vec<(usize, f64)> = scores.iter().enumerate()
            .map(|(i, &s)| (i, s)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.iter().take(k).map(|&(i, _)| i).collect()
    };
    let set_a = top_k(scores_a);
    let set_b = top_k(scores_b);
    set_a.intersection(&set_b).count() as f64 / k as f64
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let cut = max_len.saturating_sub(3);
        let end = s.char_indices().nth(cut).map(|(i, _)| i).unwrap_or(s.len());
        format!("{}...", &s[..end])
    }
}

/// Run the fixed task suite against any model via a generate closure.
/// `generate` must reset the model's KV state on each call.
/// Returns (per-item results, overall pass rate 0.0–1.0).
fn eval_tasks(
    tasks: &[TaskItem],
    tokenizer: &Tokenizer,
    generate: &mut dyn FnMut(&[u32]) -> String,
) -> (Vec<TaskResult>, f64) {
    let mut task_results = Vec::with_capacity(tasks.len());
    let mut passed = 0usize;
    for task in tasks {
        let tokens = tokenizer.encode_with_bos(task.prompt);
        let generated = generate(&tokens);
        let p = generated.to_lowercase().contains(task.expected);
        if p { passed += 1; }
        task_results.push(TaskResult {
            category: task.category.to_string(),
            prompt: task.prompt.to_string(),
            expected: task.expected.to_string(),
            generated,
            passed: p,
        });
    }
    let accuracy = if tasks.is_empty() { 1.0 } else { passed as f64 / tasks.len() as f64 };
    (task_results, accuracy)
}

// ── Individual config runners ───────────────────────────────────────────────

fn run_baseline(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    params: &AbcBenchmarkParams,
) -> Result<AbcConfigResult, ModelError> {
    println!("\n{}", "─".repeat(64));
    println!("  BASELINE: Full model, bf16, all layers");
    println!("{}\n", "─".repeat(64));

    let mut model = GenericModel::from_dir(model_dir)
        .map_err(|e| ModelError::WeightNotFound(format!("{}", e)))?;

    let mem_tracker = PeakMemoryTracker::start();
    let mut results = Vec::new();

    for (i, prompt) in params.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, params.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, params.max_new_tokens, params.temperature, params.top_p);
        let text = tokenizer.decode(&gen.tokens);
        let quality = compute_quality(&gen.tokens, params.max_new_tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s, bigram_uniq={:.2}", gen.tokens.len(), gen.tokens_per_sec, quality.unique_bigram_ratio);

        results.push(AbcPromptResult {
            prompt: prompt.clone(),
            generated_text: text,
            generated_tokens: gen.tokens.clone(),
            prompt_tokens: gen.prompt_tokens,
            num_generated: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
            quality,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);

    let (task_results, task_accuracy) = eval_tasks(
        DEFAULT_TASK_SUITE, tokenizer,
        &mut |toks| {
            model.reset();
            let gen = model.generate(toks, 20, 0.0, 1.0);
            tokenizer.decode(&gen.tokens)
        },
    );
    println!("  Task accuracy: {}/{} ({:.0}%)",
        task_results.iter().filter(|r| r.passed).count(), task_results.len(), task_accuracy * 100.0);

    Ok(AbcConfigResult {
        config: AbcConfig::Baseline.short().to_string(),
        label: AbcConfig::Baseline.label().to_string(),
        results,
        summary,
        paging_info: None,
        task_results,
        task_accuracy,
    })
}

fn run_quant_only(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    params: &AbcBenchmarkParams,
) -> Result<AbcConfigResult, ModelError> {
    println!("\n{}", "─".repeat(64));
    println!("  CONFIG A: Quantization Only (uniform Q4, all layers)");
    println!("{}\n", "─".repeat(64));

    let mut model = GenericModel::from_dir(model_dir)
        .map_err(|e| ModelError::WeightNotFound(format!("{}", e)))?;

    // Apply uniform Q4 quantization to all layers.
    println!("  Quantizing {} layers to Q4...", model.layers.len());
    let quant_start = Instant::now();
    for layer in &mut model.layers {
        layer.quantize_q4();
    }
    let quant_time = quant_start.elapsed();
    println!("  Quantization complete in {:.1} ms", quant_time.as_secs_f64() * 1000.0);

    let mem_tracker = PeakMemoryTracker::start();
    let mut results = Vec::new();

    for (i, prompt) in params.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, params.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, params.max_new_tokens, params.temperature, params.top_p);
        let text = tokenizer.decode(&gen.tokens);
        let quality = compute_quality(&gen.tokens, params.max_new_tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s, bigram_uniq={:.2}", gen.tokens.len(), gen.tokens_per_sec, quality.unique_bigram_ratio);

        results.push(AbcPromptResult {
            prompt: prompt.clone(),
            generated_text: text,
            generated_tokens: gen.tokens.clone(),
            prompt_tokens: gen.prompt_tokens,
            num_generated: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
            quality,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);

    let (task_results, task_accuracy) = eval_tasks(
        DEFAULT_TASK_SUITE, tokenizer,
        &mut |toks| {
            model.reset();
            let gen = model.generate(toks, 20, 0.0, 1.0);
            tokenizer.decode(&gen.tokens)
        },
    );
    println!("  Task accuracy: {}/{} ({:.0}%)",
        task_results.iter().filter(|r| r.passed).count(), task_results.len(), task_accuracy * 100.0);

    Ok(AbcConfigResult {
        config: AbcConfig::QuantOnly.short().to_string(),
        label: AbcConfig::QuantOnly.label().to_string(),
        results,
        summary,
        paging_info: None,
        task_results,
        task_accuracy,
    })
}

/// Try to load pre-computed layer-importance scores.
///
/// Resolution order:
/// 1. Explicit `profile_path` (portable JSON array `[f64, …]`)
/// 2. Automatic on-disk cache (`~/.cache/nve/importance/<hash>.json`)
/// 3. Returns `None` → caller should run a fresh profiling pass.
fn load_importance_scores(
    model_dir: &Path,
    profile_path: Option<&std::path::Path>,
    num_layers: usize,
) -> Option<Vec<f64>> {
    if let Some(path) = profile_path {
        match std::fs::read_to_string(path) {
            Ok(contents) => {
                match serde_json::from_str::<Vec<f64>>(&contents) {
                    Ok(scores) if scores.len() == num_layers => {
                        println!("  [profile] Loaded {} scores from {:?}", scores.len(), path);
                        return Some(scores);
                    }
                    Ok(scores) => {
                        eprintln!(
                            "  [profile] Warning: file has {} scores, model has {} layers — ignoring",
                            scores.len(), num_layers
                        );
                    }
                    Err(e) => {
                        eprintln!("  [profile] Warning: could not parse {:?}: {}", path, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  [profile] Warning: could not read {:?}: {}", path, e);
            }
        }
    }
    // Automatic cache (saved by previous unconstrained run on same machine)
    crate::importance_cache::ImportanceCache::load(model_dir, num_layers)
}

fn run_profiled_hot_only(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    params: &AbcBenchmarkParams,
) -> Result<AbcConfigResult, ModelError> {
    println!("\n{}", "─".repeat(64));
    println!("  CONFIG B: Profiled Hot-Only (bf16, importance-selected layers)");
    println!("{}\n", "─".repeat(64));

    let paged_config = PagedConfig {
        hot_budget_bytes: params.hot_budget_bytes,
        warm_budget_bytes: params.warm_budget_bytes,
        prefetch_ahead: 2,
        profile_activations: false,
        hot_only_mode: false, // Will switch after profiling
        active_layers: None,
        quant_mode: QuantMode::None,
        device: params.device.clone(),
    };

    let mut model = PagedModel::from_dir(model_dir, paged_config)?;

    // Compute active layer count from budget (before profiling — layer_size_bytes
    // is independent of the profile pass).
    let layer_size = model.layer_size_bytes();
    let total_budget = params.hot_budget_bytes + params.warm_budget_bytes;
    let active_count = if let Some(n) = params.active_layers {
        n
    } else {
        (total_budget / layer_size.max(1)).min(model.num_layers())
    };

    // ── Importance scores: pre-loaded or freshly profiled ───────────────────
    let (scores, profiling_time_ms, stability_overlap, scorer_cmp_opt) =
        if let Some(pre) = load_importance_scores(
            model_dir,
            params.profile_path.as_deref(),
            model.num_layers(),
        ) {
            // Use externally supplied scores — skip the forward profiling pass.
            // This is the "profile on large machine, run on small device" path.
            println!("  Skipping profiling pass — using pre-loaded importance scores");
            model.inject_importance_scores(pre.clone());
            (pre, 0.0_f64, None, None)
        } else {
            // No cached/external scores: run the full profiling pass.
            let profile_tokens: Vec<u32> = params.prompts.iter()
                .flat_map(|p| tokenizer.encode_with_bos(p))
                .take(128)
                .collect();

            println!("  Profiling layer importance ({} tokens across {} prompts)...",
                profile_tokens.len(), params.prompts.len());

            let stability_overlap = if params.stability_check && profile_tokens.len() >= 2 {
                let half = profile_tokens.len() / 2;
                let sa = model.profile_layer_importance(&profile_tokens[..half])?;
                let sb = model.profile_layer_importance(&profile_tokens[half..])?;
                Some(top_k_overlap(&sa, &sb, active_count))
            } else {
                None
            };

            let t0 = Instant::now();
            let (sc, cmp) = model.profile_layer_importance_detailed(&profile_tokens)?;
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

            if let Some(overlap) = stability_overlap {
                println!("  Layer stability: {:.1}% top-{} overlap (split-half)",
                    overlap * 100.0, active_count);
            }
            println!("  Profiling complete in {:.0} ms", elapsed);
            println!("  Scorer comparison  (τ = Kendall's τ vs proxy, top-k = top-{} overlap):",
                active_count);
            println!("    FFN-only:      τ={:.3}  top-k={:.0}%  (drops Q/V projections)",
                cmp.tau_ffn_vs_proxy, cmp.topk_ffn_vs_proxy * 100.0);
            println!("    Attn-proxy:    τ={:.3}  top-k={:.0}%  (Q/V projections only)",
                cmp.tau_attn_vs_proxy, cmp.topk_attn_vs_proxy * 100.0);
            println!("    Input-L2:      τ={:.3}  top-k={:.0}%  (zero extra compute)",
                cmp.tau_input_vs_proxy, cmp.topk_input_vs_proxy * 100.0);
            println!("    Attn proxy overhead: {:.1} ms / {:.1} ms total ({:.1}%)",
                cmp.attn_proxy_cost_ms, cmp.total_time_ms,
                cmp.attn_proxy_fraction * 100.0);

            (sc, elapsed, stability_overlap, Some(cmp))
        };

    model.apply_profiled_hot_only(active_count);
    println!("  Active layers: {}/{}", active_count, model.num_layers());

    // Initialize GPU inference if device is a GPU (no-op on CPU-only builds).
    model.init_gpu()?;

    let mem_tracker = PeakMemoryTracker::start();
    let mut results = Vec::new();

    for (i, prompt) in params.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, params.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, params.max_new_tokens, params.temperature, params.top_p)?;
        let text = tokenizer.decode(&gen.tokens);
        let quality = compute_quality(&gen.tokens, params.max_new_tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s, bigram_uniq={:.2}", gen.tokens.len(), gen.tokens_per_sec, quality.unique_bigram_ratio);

        results.push(AbcPromptResult {
            prompt: prompt.clone(),
            generated_text: text,
            generated_tokens: gen.tokens.clone(),
            prompt_tokens: gen.prompt_tokens,
            num_generated: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
            quality,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);

    let (task_results, task_accuracy) = eval_tasks(
        DEFAULT_TASK_SUITE, tokenizer,
        &mut |toks| {
            model.reset();
            match model.generate(toks, 20, 0.0, 1.0) {
                Ok(gen) => tokenizer.decode(&gen.tokens),
                Err(_) => String::new(),
            }
        },
    );
    println!("  Task accuracy: {}/{} ({:.0}%)",
        task_results.iter().filter(|r| r.passed).count(), task_results.len(), task_accuracy * 100.0);

    let stats = model.stats();

    let paging_info = Some(AbcPagingInfo {
        page_hits: stats.page_hits,
        page_faults: stats.page_faults,
        fault_rate_pct: stats.fault_rate() * 100.0,
        layers_loaded: stats.layers_loaded,
        layers_evicted: stats.layers_evicted,
        load_time_ms: stats.total_load_time_ms,
        active_layers: active_count,
        total_layers: model.num_layers(),
        profiling_time_ms,
        layer_importance_scores: scores,
        layer_quant_assignments: None,
        layer_stability_top_k_overlap: stability_overlap,
        scorer_comparison: scorer_cmp_opt,
    });

    Ok(AbcConfigResult {
        config: AbcConfig::ProfiledHotOnly.short().to_string(),
        label: AbcConfig::ProfiledHotOnly.label().to_string(),
        results,
        summary,
        paging_info,
        task_results,
        task_accuracy,
    })
}

fn run_profiled_quantized(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    params: &AbcBenchmarkParams,
) -> Result<AbcConfigResult, ModelError> {
    println!("\n{}", "─".repeat(64));
    println!("  CONFIG C: Profiled + Quantized (PG {:.1} bpw + AWQ)", params.target_bpw);
    println!("{}\n", "─".repeat(64));

    let paged_config = PagedConfig {
        hot_budget_bytes: params.hot_budget_bytes,
        warm_budget_bytes: params.warm_budget_bytes,
        prefetch_ahead: 2,
        profile_activations: false,
        hot_only_mode: false,
        active_layers: None,
        quant_mode: QuantMode::ProfileGuided(params.target_bpw),
        device: params.device.clone(),
    };

    let mut model = PagedModel::from_dir(model_dir, paged_config)?;

    // Compute active layer count from budget.
    let layer_size = model.layer_size_bytes();
    let total_budget = params.hot_budget_bytes + params.warm_budget_bytes;
    let active_count = if let Some(n) = params.active_layers {
        n
    } else {
        (total_budget / layer_size.max(1)).min(model.num_layers())
    };

    // ── Importance scores + AWQ saliency ────────────────────────────────────
    // Config C always needs AWQ channel saliency (requires a fresh forward pass).
    // If pre-computed layer rankings are available we inject them AFTER the
    // profiling pass so that bit-allocation uses the imported priority order while
    // AWQ still has real activation data.
    let profile_tokens: Vec<u32> = params.prompts.iter()
        .flat_map(|p| tokenizer.encode_with_bos(p))
        .take(128)
        .collect();

    println!("  Profiling layer importance + AWQ saliency ({} tokens across {} prompts)...",
        profile_tokens.len(), params.prompts.len());

    let stability_overlap = if params.stability_check && profile_tokens.len() >= 2 {
        let half = profile_tokens.len() / 2;
        let sa = model.profile_layer_importance(&profile_tokens[..half])?;
        let sb = model.profile_layer_importance(&profile_tokens[half..])?;
        Some(top_k_overlap(&sa, &sb, active_count))
    } else {
        None
    };

    let t0 = Instant::now();
    let (mut scores, scorer_cmp) = model.profile_layer_importance_detailed(&profile_tokens)?;
    let profiling_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Override layer-ranking scores with pre-computed profile if supplied.
    // AWQ saliency collected above is preserved — only the importance ordering changes.
    if let Some(pre) = load_importance_scores(
        model_dir,
        params.profile_path.as_deref(),
        model.num_layers(),
    ) {
        println!("  Overriding layer ranking with pre-loaded profile scores");
        model.inject_importance_scores(pre.clone());
        scores = pre;
    }

    if let Some(overlap) = stability_overlap {
        println!("  Layer stability: {:.1}% top-{} overlap (split-half)",
            overlap * 100.0, active_count);
    }
    println!("  Profiling complete in {:.0} ms", profiling_time_ms);
    println!("  Scorer comparison  (τ = Kendall's τ vs proxy, top-k = top-{} overlap):",
        active_count);
    println!("    FFN-only:      τ={:.3}  top-k={:.0}%  (drops Q/V projections)",
        scorer_cmp.tau_ffn_vs_proxy, scorer_cmp.topk_ffn_vs_proxy * 100.0);
    println!("    Attn-proxy:    τ={:.3}  top-k={:.0}%  (Q/V projections only)",
        scorer_cmp.tau_attn_vs_proxy, scorer_cmp.topk_attn_vs_proxy * 100.0);
    println!("    Input-L2:      τ={:.3}  top-k={:.0}%  (zero extra compute)",
        scorer_cmp.tau_input_vs_proxy, scorer_cmp.topk_input_vs_proxy * 100.0);
    println!("    Attn proxy overhead: {:.1} ms / {:.1} ms total ({:.1}%)",
        scorer_cmp.attn_proxy_cost_ms, scorer_cmp.total_time_ms,
        scorer_cmp.attn_proxy_fraction * 100.0);

    model.apply_profiled_hot_only(active_count);
    println!("  Active layers: {}/{}", active_count, model.num_layers());

    // Initialize GPU inference if device is a GPU (no-op on CPU-only builds).
    model.init_gpu()?;

    // Collect quant assignments for the report.
    let quant_assignments: Option<Vec<String>> = model.layer_quant_assignment_labels();

    let mem_tracker = PeakMemoryTracker::start();
    let mut results = Vec::new();

    for (i, prompt) in params.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, params.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, params.max_new_tokens, params.temperature, params.top_p)?;
        let text = tokenizer.decode(&gen.tokens);
        let quality = compute_quality(&gen.tokens, params.max_new_tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s, bigram_uniq={:.2}", gen.tokens.len(), gen.tokens_per_sec, quality.unique_bigram_ratio);

        results.push(AbcPromptResult {
            prompt: prompt.clone(),
            generated_text: text,
            generated_tokens: gen.tokens.clone(),
            prompt_tokens: gen.prompt_tokens,
            num_generated: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
            quality,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);

    let (task_results, task_accuracy) = eval_tasks(
        DEFAULT_TASK_SUITE, tokenizer,
        &mut |toks| {
            model.reset();
            match model.generate(toks, 20, 0.0, 1.0) {
                Ok(gen) => tokenizer.decode(&gen.tokens),
                Err(_) => String::new(),
            }
        },
    );
    println!("  Task accuracy: {}/{} ({:.0}%)",
        task_results.iter().filter(|r| r.passed).count(), task_results.len(), task_accuracy * 100.0);

    let stats = model.stats();

    let paging_info = Some(AbcPagingInfo {
        page_hits: stats.page_hits,
        page_faults: stats.page_faults,
        fault_rate_pct: stats.fault_rate() * 100.0,
        layers_loaded: stats.layers_loaded,
        layers_evicted: stats.layers_evicted,
        load_time_ms: stats.total_load_time_ms,
        active_layers: active_count,
        total_layers: model.num_layers(),
        profiling_time_ms,
        layer_importance_scores: scores,
        layer_quant_assignments: quant_assignments,
        layer_stability_top_k_overlap: stability_overlap,
        scorer_comparison: Some(scorer_cmp),
    });

    Ok(AbcConfigResult {
        config: AbcConfig::ProfiledQuantized.short().to_string(),
        label: AbcConfig::ProfiledQuantized.label().to_string(),
        results,
        summary,
        paging_info,
        task_results,
        task_accuracy,
    })
}

// ── Summary computation ─────────────────────────────────────────────────────

fn compute_summary(results: &[AbcPromptResult], peak_memory_mb: f64) -> AbcSummary {
    let n = results.len() as f64;
    if n == 0.0 {
        return AbcSummary {
            avg_prefill_ms: 0.0,
            avg_decode_ms_per_token: 0.0,
            avg_tokens_per_sec: 0.0,
            total_tokens_generated: 0,
            total_time_ms: 0.0,
            peak_memory_mb: 0.0,
            avg_unique_bigram_ratio: 0.0,
            avg_unique_trigram_ratio: 0.0,
            avg_max_repeat_run: 0.0,
        };
    }

    AbcSummary {
        avg_prefill_ms: results.iter().map(|r| r.prefill_time_ms).sum::<f64>() / n,
        avg_decode_ms_per_token: results.iter().map(|r| r.ms_per_token).sum::<f64>() / n,
        avg_tokens_per_sec: results.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n,
        total_tokens_generated: results.iter().map(|r| r.num_generated).sum(),
        total_time_ms: results.iter().map(|r| r.total_time_ms).sum(),
        peak_memory_mb,
        avg_unique_bigram_ratio: results.iter().map(|r| r.quality.unique_bigram_ratio).sum::<f64>() / n,
        avg_unique_trigram_ratio: results.iter().map(|r| r.quality.unique_trigram_ratio).sum::<f64>() / n,
        avg_max_repeat_run: results.iter().map(|r| r.quality.max_repeat_run as f64).sum::<f64>() / n,
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

fn compute_comparison(configs: &[AbcConfigResult]) -> AbcComparison {
    // If no baseline is present, use the first config as reference.
    let baseline = configs.iter().find(|c| c.config == "baseline")
        .or_else(|| configs.first());

    let base_tps = baseline.map(|b| b.summary.avg_tokens_per_sec).unwrap_or(1.0);
    let base_mem = baseline.map(|b| b.summary.peak_memory_mb).filter(|&m| m > 0.0).unwrap_or(1.0);
    let base_bigram = baseline.map(|b| b.summary.avg_unique_bigram_ratio).filter(|&b| b > 0.0).unwrap_or(1.0);
    let base_trigram = baseline.map(|b| b.summary.avg_unique_trigram_ratio).filter(|&t| t > 0.0).unwrap_or(1.0);
    let base_task = baseline.map(|b| b.task_accuracy).filter(|&t| t > 0.0).unwrap_or(1.0);
    let baseline_config = baseline.map(|b| b.config.as_str()).unwrap_or("baseline");

    let mut speedup = Vec::new();
    let mut memory = Vec::new();
    let mut quality = Vec::new();

    for cfg in configs {
        if cfg.config == baseline_config {
            continue;
        }
        speedup.push(SpeedupEntry {
            config: cfg.config.clone(),
            speedup: if base_tps > 0.0 { cfg.summary.avg_tokens_per_sec / base_tps } else { 0.0 },
        });
        memory.push(MemorySavingsEntry {
            config: cfg.config.clone(),
            savings_pct: if base_mem > 0.0 {
                (1.0 - cfg.summary.peak_memory_mb / base_mem) * 100.0
            } else {
                0.0
            },
        });
        quality.push(QualityCompEntry {
            config: cfg.config.clone(),
            bigram_ratio_vs_baseline: if base_bigram > 0.0 {
                cfg.summary.avg_unique_bigram_ratio / base_bigram
            } else {
                0.0
            },
            trigram_ratio_vs_baseline: if base_trigram > 0.0 {
                cfg.summary.avg_unique_trigram_ratio / base_trigram
            } else {
                0.0
            },
            task_accuracy_vs_baseline: if base_task > 0.0 {
                cfg.task_accuracy / base_task
            } else {
                0.0
            },
        });
    }

    AbcComparison {
        speedup_vs_baseline: speedup,
        memory_savings_pct: memory,
        quality_vs_baseline: quality,
    }
}

// ── Main entry point ────────────────────────────────────────────────────────

/// Run the full ABC benchmark suite.
pub fn run_abc_benchmark(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    params: &AbcBenchmarkParams,
) -> Result<AbcReport, ModelError> {
    // Detect model info.
    let config = crate::arch::UnifiedConfig::from_model_dir(model_dir)
        .map_err(|e| ModelError::WeightNotFound(format!("config: {}", e)))?;

    let resolved_device = crate::device::Device::resolve(&params.device);
    println!("\n{}", "=".repeat(64));
    println!("  NVE ABC BENCHMARK");
    println!("  Model: {} ({}L, {}d)", config.arch, config.num_hidden_layers, config.hidden_size);
    println!("  Est. params: {:.1}B", config.estimated_params() as f64 / 1e9);
    println!("  Profiling device: {}", resolved_device);
    println!("  Prompts: {}, Tokens: {}, Temp: {}", params.prompts.len(), params.max_new_tokens, params.temperature);
    println!("  Budget: {} MB hot + {} MB warm", params.hot_budget_bytes / 1024 / 1024, params.warm_budget_bytes / 1024 / 1024);
    println!("  Target bpw (config C): {:.1}", params.target_bpw);
    println!("{}\n", "=".repeat(64));

    let mut configurations = Vec::new();

    for abc in &params.configs {
        let result = match abc {
            AbcConfig::Baseline => run_baseline(model_dir, tokenizer, params)?,
            AbcConfig::QuantOnly => run_quant_only(model_dir, tokenizer, params)?,
            AbcConfig::ProfiledHotOnly => run_profiled_hot_only(model_dir, tokenizer, params)?,
            AbcConfig::ProfiledQuantized => run_profiled_quantized(model_dir, tokenizer, params)?,
        };
        configurations.push(result);
    }

    let comparison = compute_comparison(&configurations);

    // Print comparison table.
    print_abc_comparison(&configurations, &comparison, &params.quality_budget);

    // Print per-prompt output comparison.
    print_output_comparison(&configurations, &params.prompts);

    let report = AbcReport {
        model_path: model_dir.display().to_string(),
        model_arch: config.arch.to_string(),
        model_params: format!("{:.1}B", config.estimated_params() as f64 / 1e9),
        num_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        timestamp: chrono_timestamp(),
        params: AbcReportParams {
            max_new_tokens: params.max_new_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            hot_budget_mb: params.hot_budget_bytes / 1024 / 1024,
            warm_budget_mb: params.warm_budget_bytes / 1024 / 1024,
            target_bpw: params.target_bpw,
            num_prompts: params.prompts.len(),
            device: resolved_device.to_string(),
        },
        configurations,
        comparison,
    };

    Ok(report)
}

// ── Pretty printing ─────────────────────────────────────────────────────────

fn print_abc_comparison(configs: &[AbcConfigResult], comparison: &AbcComparison, budget: &QualityBudget) {
    println!("\n{}", "=".repeat(90));
    println!("  ABC BENCHMARK COMPARISON");
    println!("{}\n", "=".repeat(90));

    // Header
    print!("{:<36}", "");
    for cfg in configs {
        print!("{:>13}", short_label(&cfg.config));
    }
    println!();

    print!("{:<36}", "");
    for _ in configs {
        print!("{:>13}", "────────────");
    }
    println!();

    // Throughput
    print!("{:<36}", "Avg throughput (tok/s)");
    for cfg in configs {
        print!("{:>12.1}x", cfg.summary.avg_tokens_per_sec);
    }
    println!();

    // Prefill
    print!("{:<36}", "Avg prefill (ms)");
    for cfg in configs {
        print!("{:>11.1}ms", cfg.summary.avg_prefill_ms);
    }
    println!();

    // Decode per token
    print!("{:<36}", "Avg decode (ms/tok)");
    for cfg in configs {
        print!("{:>11.1}ms", cfg.summary.avg_decode_ms_per_token);
    }
    println!();

    // Peak memory
    print!("{:<36}", "Peak memory (MB)");
    for cfg in configs {
        print!("{:>11.0}MB", cfg.summary.peak_memory_mb);
    }
    println!();

    // Quality: unique bigram ratio
    print!("{:<36}", "Unique bigram ratio");
    for cfg in configs {
        print!("{:>13.3}", cfg.summary.avg_unique_bigram_ratio);
    }
    println!();

    // Quality: unique trigram ratio
    print!("{:<36}", "Unique trigram ratio");
    for cfg in configs {
        print!("{:>13.3}", cfg.summary.avg_unique_trigram_ratio);
    }
    println!();

    // Max repeat run
    print!("{:<36}", "Avg max repeat run");
    for cfg in configs {
        print!("{:>13.1}", cfg.summary.avg_max_repeat_run);
    }
    println!();

    // Task accuracy
    print!("{:<36}", "Task accuracy (pass rate)");
    for cfg in configs {
        let flag = match cfg.config.as_str() {
            "B_profiled_hot"    if cfg.task_accuracy < budget.hot_only_min_task_accuracy  => " !",
            "C_profiled_quant"  if cfg.task_accuracy < budget.quant_min_task_accuracy     => " !",
            _ => "",
        };
        print!("{:>11.0}%{:<2}", cfg.task_accuracy * 100.0, flag);
    }
    println!();

    // Total tokens
    print!("{:<36}", "Total tokens generated");
    for cfg in configs {
        print!("{:>13}", cfg.summary.total_tokens_generated);
    }
    println!();

    // Paging info for B and C
    println!();
    for cfg in configs {
        if let Some(ref pi) = cfg.paging_info {
            println!("  {} paging:", short_label(&cfg.config));
            println!("    Active layers:    {}/{}", pi.active_layers, pi.total_layers);
            println!("    Profiling time:   {:.0} ms", pi.profiling_time_ms);
            println!("    Page faults:      {} ({:.1}%)", pi.page_faults, pi.fault_rate_pct);
            println!("    Layer load time:  {:.0} ms", pi.load_time_ms);
            if let Some(ref assignments) = pi.layer_quant_assignments {
                let counts = count_quant_assignments(assignments);
                println!("    Bit allocation:   {}", counts);
            }
            if let Some(overlap) = pi.layer_stability_top_k_overlap {
                println!("    Layer stability:  {:.1}% top-{} overlap (split-half)",
                    overlap * 100.0, pi.active_layers);
            }
        }
    }

    // Relative comparison
    if !comparison.speedup_vs_baseline.is_empty() {
        println!();
        println!("  Relative to baseline:");
        for s in &comparison.speedup_vs_baseline {
            let mem = comparison.memory_savings_pct.iter().find(|m| m.config == s.config);
            let qual = comparison.quality_vs_baseline.iter().find(|q| q.config == s.config);
            println!(
                "    {:<20} speed={:.2}x  memory={:+.1}%  bigram={:.2}x  task_acc={:.2}x",
                short_label(&s.config),
                s.speedup,
                mem.map(|m| m.savings_pct).unwrap_or(0.0),
                qual.map(|q| q.bigram_ratio_vs_baseline).unwrap_or(0.0),
                qual.map(|q| q.task_accuracy_vs_baseline).unwrap_or(0.0),
            );
        }
    }

    // Threshold violation summary
    let mut violations: Vec<String> = Vec::new();
    for cfg in configs {
        let threshold = match cfg.config.as_str() {
            "B_profiled_hot"   => Some(("hot-only",   budget.hot_only_min_task_accuracy)),
            "C_profiled_quant" => Some(("quant",       budget.quant_min_task_accuracy)),
            _ => None,
        };
        if let Some((mode, min_acc)) = threshold {
            if cfg.task_accuracy < min_acc {
                violations.push(format!(
                    "FAIL  {} task accuracy {:.0}% < {} threshold {:.0}%",
                    short_label(&cfg.config), cfg.task_accuracy * 100.0,
                    mode, min_acc * 100.0
                ));
            }
        }
    }
    if !violations.is_empty() {
        println!();
        println!("  Quality budget violations:");
        for v in &violations {
            println!("    {}", v);
        }
    }
}

fn print_output_comparison(configs: &[AbcConfigResult], prompts: &[String]) {
    println!("\n{}", "=".repeat(90));
    println!("  OUTPUT COMPARISON (per prompt)");
    println!("{}\n", "=".repeat(90));

    for (pi, prompt) in prompts.iter().enumerate() {
        println!("  Prompt {}: \"{}\"", pi + 1, truncate(prompt, 60));
        println!();
        for cfg in configs {
            if pi < cfg.results.len() {
                let r = &cfg.results[pi];
                let out = truncate(&r.generated_text, 120);
                println!("    {:<20} | {}", short_label(&cfg.config), out);
            }
        }
        println!();
    }
}

fn short_label(config: &str) -> &str {
    match config {
        "baseline" => "Baseline",
        "A_quant_only" => "A:Quant",
        "B_profiled_hot" => "B:Profile",
        "C_profiled_quant" => "C:PG+AWQ",
        other => other,
    }
}

fn count_quant_assignments(assignments: &[String]) -> String {
    let mut counts = std::collections::HashMap::new();
    for a in assignments {
        *counts.entry(a.as_str()).or_insert(0usize) += 1;
    }
    let mut parts: Vec<String> = counts.iter().map(|(k, v)| format!("{}x{}", v, k)).collect();
    parts.sort();
    parts.join(", ")
}

fn chrono_timestamp() -> String {
    // Simple timestamp without chrono dependency.
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO-ish: just use epoch for now (avoids adding chrono dep).
    format!("{}", secs)
}
