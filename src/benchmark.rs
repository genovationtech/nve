//! Benchmark harness for NVE inference.
//!
//! Runs latency, throughput, and memory benchmarks comparing:
//! 1. Baseline: all weights resident in RAM
//! 2. NVE paged: tiered weight loading with LRU eviction

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::model::{LlamaModel, ModelError};
use crate::paged_model::{PagedConfig, PagedModel, PagingStats};
use crate::tokenizer::Tokenizer;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Prompts to benchmark with.
    pub prompts: Vec<String>,
    /// Max tokens to generate per prompt.
    pub max_new_tokens: usize,
    /// Temperature for sampling.
    pub temperature: f32,
    /// Top-p for nucleus sampling.
    pub top_p: f32,
    /// Number of warmup runs before measuring.
    pub warmup_runs: usize,
    /// Paged model config for NVE benchmark.
    pub paged_config: PagedConfig,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            prompts: vec![
                "The capital of France is".into(),
                "Explain quantum computing in simple terms:".into(),
                "Write a Python function to sort a list:".into(),
            ],
            max_new_tokens: 50,
            temperature: 0.0, // Greedy for reproducibility
            top_p: 1.0,
            warmup_runs: 0,
            paged_config: PagedConfig::default(),
        }
    }
}

/// Results from a single benchmark run.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub prompt: String,
    pub generated_text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time_ms: f64,
    pub decode_time_ms: f64,
    pub total_time_ms: f64,
    pub tokens_per_sec: f64,
    pub ms_per_token: f64,
}

/// Full benchmark report.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub model_name: String,
    pub mode: String,
    pub results: Vec<BenchmarkResult>,
    pub summary: BenchmarkSummary,
    pub paging_stats: Option<PagingStatsReport>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub avg_prefill_ms: f64,
    pub avg_decode_ms_per_token: f64,
    pub avg_tokens_per_sec: f64,
    pub total_tokens: usize,
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PagingStatsReport {
    pub page_hits: u64,
    pub page_faults: u64,
    pub fault_rate: f64,
    pub layers_loaded: u64,
    pub layers_evicted: u64,
    pub total_load_time_ms: f64,
    pub prefetch_hits: u64,
}

impl From<&PagingStats> for PagingStatsReport {
    fn from(s: &PagingStats) -> Self {
        PagingStatsReport {
            page_hits: s.page_hits,
            page_faults: s.page_faults,
            fault_rate: s.fault_rate(),
            layers_loaded: s.layers_loaded,
            layers_evicted: s.layers_evicted,
            total_load_time_ms: s.total_load_time_ms,
            prefetch_hits: s.prefetch_hits,
        }
    }
}

/// Run benchmark on the baseline (all-resident) model.
pub fn benchmark_baseline(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    config: &BenchmarkConfig,
) -> Result<BenchmarkReport, ModelError> {
    println!("\n=== Baseline Benchmark (all weights resident) ===\n");

    let mut model = LlamaModel::from_dir(model_dir)?;
    let mut results = Vec::new();

    let mem_tracker = PeakMemoryTracker::start();

    for (i, prompt) in config.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, config.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, config.max_new_tokens, config.temperature, config.top_p);
        let text = tokenizer.decode(&gen.tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s", gen.tokens.len(), gen.tokens_per_sec);

        results.push(BenchmarkResult {
            prompt: prompt.clone(),
            generated_text: text,
            prompt_tokens: gen.prompt_tokens,
            generated_tokens: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);

    Ok(BenchmarkReport {
        model_name: format!("Llama (baseline)"),
        mode: "baseline".into(),
        results,
        summary,
        paging_stats: None,
    })
}

/// Run benchmark on the NVE paged model.
pub fn benchmark_paged(
    model_dir: &Path,
    tokenizer: &Tokenizer,
    config: &BenchmarkConfig,
) -> Result<BenchmarkReport, ModelError> {
    println!("\n=== NVE Paged Benchmark (tiered weight loading) ===\n");

    let mut model = PagedModel::from_dir(model_dir, config.paged_config.clone())?;
    let mut results = Vec::new();

    let mem_tracker = PeakMemoryTracker::start();

    for (i, prompt) in config.prompts.iter().enumerate() {
        model.reset();
        let tokens = tokenizer.encode_with_bos(prompt);

        println!("  [{}/{}] \"{}\"", i + 1, config.prompts.len(), truncate(prompt, 50));

        let gen = model.generate(&tokens, config.max_new_tokens, config.temperature, config.top_p)?;
        let text = tokenizer.decode(&gen.tokens);
        let decode_tokens = gen.tokens.len().max(1);

        println!("    -> {} tokens, {:.1} tok/s", gen.tokens.len(), gen.tokens_per_sec);
        println!("    {}", model.memory_report());

        results.push(BenchmarkResult {
            prompt: prompt.clone(),
            generated_text: text,
            prompt_tokens: gen.prompt_tokens,
            generated_tokens: gen.tokens.len(),
            prefill_time_ms: gen.prefill_time_ms,
            decode_time_ms: gen.decode_time_ms,
            total_time_ms: gen.total_time_ms,
            tokens_per_sec: gen.tokens_per_sec,
            ms_per_token: gen.decode_time_ms / decode_tokens as f64,
        });
    }

    let peak_memory_mb = mem_tracker.stop_and_get_peak();
    let summary = compute_summary(&results, peak_memory_mb);
    let paging_stats = Some(PagingStatsReport::from(model.stats()));

    Ok(BenchmarkReport {
        model_name: format!("Llama (NVE paged)"),
        mode: "paged".into(),
        results,
        summary,
        paging_stats,
    })
}

fn compute_summary(results: &[BenchmarkResult], peak_memory_mb: f64) -> BenchmarkSummary {
    let n = results.len() as f64;
    let total_tokens: usize = results.iter().map(|r| r.generated_tokens).sum();
    let total_time: f64 = results.iter().map(|r| r.total_time_ms).sum();

    BenchmarkSummary {
        avg_prefill_ms: results.iter().map(|r| r.prefill_time_ms).sum::<f64>() / n,
        avg_decode_ms_per_token: results.iter().map(|r| r.ms_per_token).sum::<f64>() / n,
        avg_tokens_per_sec: results.iter().map(|r| r.tokens_per_sec).sum::<f64>() / n,
        total_tokens,
        total_time_ms: total_time,
        peak_memory_mb,
    }
}

/// Print a comparison of two benchmark reports.
pub fn print_comparison(baseline: &BenchmarkReport, paged: &BenchmarkReport) {
    println!("\n{}", "=".repeat(60));
    println!("  BENCHMARK COMPARISON");
    println!("{}\n", "=".repeat(60));

    println!("{:<30} {:>12} {:>12}", "", "Baseline", "NVE Paged");
    println!("{:-<30} {:->12} {:->12}", "", "", "");

    println!(
        "{:<30} {:>10.1} ms {:>10.1} ms",
        "Avg prefill",
        baseline.summary.avg_prefill_ms,
        paged.summary.avg_prefill_ms,
    );
    println!(
        "{:<30} {:>10.1} ms {:>10.1} ms",
        "Avg decode (per token)",
        baseline.summary.avg_decode_ms_per_token,
        paged.summary.avg_decode_ms_per_token,
    );
    println!(
        "{:<30} {:>10.1}/s {:>10.1}/s",
        "Avg throughput",
        baseline.summary.avg_tokens_per_sec,
        paged.summary.avg_tokens_per_sec,
    );
    println!(
        "{:<30} {:>10.1} MB {:>10.1} MB",
        "Peak memory",
        baseline.summary.peak_memory_mb,
        paged.summary.peak_memory_mb,
    );

    if let Some(stats) = &paged.paging_stats {
        println!();
        println!("  NVE Paging Stats:");
        println!("    Page hits:      {}", stats.page_hits);
        println!("    Page faults:    {}", stats.page_faults);
        println!("    Fault rate:     {:.1}%", stats.fault_rate * 100.0);
        println!("    Layers loaded:  {}", stats.layers_loaded);
        println!("    Layers evicted: {}", stats.layers_evicted);
        println!("    Load time:      {:.1} ms", stats.total_load_time_ms);
        println!("    Prefetch hits:  {}", stats.prefetch_hits);
    }

    let speedup = if paged.summary.avg_tokens_per_sec > 0.0 {
        baseline.summary.avg_tokens_per_sec / paged.summary.avg_tokens_per_sec
    } else {
        0.0
    };
    let mem_savings = if baseline.summary.peak_memory_mb > 0.0 {
        (1.0 - paged.summary.peak_memory_mb / baseline.summary.peak_memory_mb) * 100.0
    } else {
        0.0
    };

    println!();
    println!("  Speed ratio: {:.2}x (baseline/paged)", speedup);
    println!("  Memory savings: {:.1}%", mem_savings);
}

/// Get current RSS in MB (Linux).
fn get_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }
    0.0
}

/// Tracks true peak RSS by polling /proc/self/status every 10 ms in a
/// background thread.  Before/after snapshots miss spikes during weight
/// loading; this catches them.
pub struct PeakMemoryTracker {
    peak_mb: Arc<Mutex<f64>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakMemoryTracker {
    /// Start polling immediately.
    pub fn start() -> Self {
        let initial = get_rss_mb();
        let peak_mb = Arc::new(Mutex::new(initial));
        let stop = Arc::new(AtomicBool::new(false));
        let peak_clone = Arc::clone(&peak_mb);
        let stop_clone = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                let rss = get_rss_mb();
                let mut guard = peak_clone.lock().unwrap();
                if rss > *guard { *guard = rss; }
                drop(guard);
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        PeakMemoryTracker { peak_mb, stop, handle: Some(handle) }
    }

    /// Stop the polling thread and return the highest RSS seen.
    pub fn stop_and_get_peak(mut self) -> f64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() { let _ = h.join(); }
        // One final reading after the thread exits.
        let rss = get_rss_mb();
        let mut guard = self.peak_mb.lock().unwrap();
        if rss > *guard { *guard = rss; }
        *guard
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
