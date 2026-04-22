//! NVE CLI — Serve any HuggingFace model with virtual weight paging.

use std::io::{self, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod abc_benchmark;
mod arch;
mod importance_cache;
mod attention;
mod benchmark;
mod cli;
mod cluster;
mod config;
mod device;
mod gpu_layer;
mod cuda_kernels;
mod decode_graph;
mod generic_model;
mod hub;
mod model;
mod ops;
mod paged_model;
mod pager;
mod profiler;
mod quantize;
mod safetensors;
mod tensor;
mod tier;
mod tokenizer;
mod weight_map;

use config::LlamaConfig;
use generic_model::GenericModel;
use model::LlamaModel;
use paged_model::{PagedConfig, PagedModel};
use quantize::QuantMode;
use tokenizer::Tokenizer;

use cli::config::NveConfig;

#[derive(Parser)]
#[command(name = "nve")]
#[command(about = "NVE — Serve any HuggingFace model with virtual weight paging")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat with a model (primary interface)
    Chat {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: Option<String>,

        /// Hot memory budget in MB (always-resident RAM)
        #[arg(long)]
        hot_budget_mb: Option<usize>,

        /// Warm memory budget in MB (LRU-cached RAM)
        #[arg(long)]
        warm_budget_mb: Option<usize>,

        /// Auto-detect available memory and compute budgets (uses 80% of available RAM)
        #[arg(long)]
        auto_budget: bool,

        /// Hot-only mode: skip layers that don't fit in memory
        #[arg(long)]
        hot_only: bool,

        /// Profile layer importance before hot-only inference
        #[arg(long)]
        profile: bool,

        /// Number of transformer layers to keep active
        #[arg(long)]
        active_layers: Option<usize>,

        /// Quantize weights: none, q4, q8, q3, q2, q1 (default: none)
        #[arg(long)]
        quantize: Option<String>,

        /// Maximum tokens to generate per turn
        #[arg(short = 'n', long, default_value = "512")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Top-p nucleus sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,
    },

    /// Manage NVE configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Run text generation with any HuggingFace model
    Generate {
        /// Path to model directory OR HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B)
        #[arg(short, long)]
        model: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "100")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Top-p nucleus sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Use NVE paged inference (works with any supported architecture)
        #[arg(long)]
        paged: bool,

        /// Hot-only mode: skip layers that don't fit in memory instead of paging from SSD.
        /// Trades quality for speed — runs only the most important layers.
        #[arg(long)]
        hot_only: bool,

        /// Number of transformer layers to keep active (default: fit as many as budget allows).
        /// Lower = faster but lower quality. Works with --paged and --hot-only.
        #[arg(long)]
        active_layers: Option<usize>,

        /// Auto-detect available memory and compute budgets (uses 80% of available RAM)
        #[arg(long)]
        auto_budget: bool,

        /// Hot memory budget in MB (for paged mode; overridden by --auto-budget)
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB (for paged mode; overridden by --auto-budget)
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Profile layer importance before hot-only inference.
        /// Runs a quick profiling pass to select the most important layers
        /// instead of using evenly-spaced selection.
        #[arg(long)]
        profile: bool,

        /// Quantize weights after loading: none, q4, q8 (default: none)
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Force legacy Llama-only code path
        #[arg(long)]
        legacy: bool,

        /// Compute device: auto, cpu, cuda:0, hip:0, metal, vulkan:0
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Run benchmark comparing baseline vs NVE paged inference
    Benchmark {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Max tokens per prompt
        #[arg(short = 'n', long, default_value = "50")]
        max_tokens: usize,

        /// Auto-detect available memory and compute budgets (uses 80% of available RAM)
        #[arg(long)]
        auto_budget: bool,

        /// Hot memory budget in MB for paged mode (overridden by --auto-budget)
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB for paged mode (overridden by --auto-budget)
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Quantize weights after loading: none, q4, q8 (default: none)
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Save results to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show model information (works with any HuggingFace model)
    Info {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,
    },

    /// Download a model from HuggingFace Hub
    Download {
        /// HuggingFace model ID (e.g. meta-llama/Llama-3.2-1B)
        #[arg()]
        model_id: String,

        /// Directory to download to
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run ABC benchmark: compare quantization-only vs profiled hot-only vs profiled+quantized
    AbcTest {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Max tokens per prompt
        #[arg(short = 'n', long, default_value = "50")]
        max_tokens: usize,

        /// Auto-detect available memory and compute budgets (uses 80% of available RAM)
        #[arg(long)]
        auto_budget: bool,

        /// Hot memory budget in MB (overridden by --auto-budget)
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB (overridden by --auto-budget)
        #[arg(long, default_value = "1536")]
        warm_budget_mb: usize,

        /// Target bits-per-weight for config C (profile-guided quantization)
        #[arg(long, default_value = "2.0")]
        target_bpw: f32,

        /// Override number of active layers for configs B and C
        #[arg(long)]
        active_layers: Option<usize>,

        /// Only run specific configs: baseline,a,b,c (comma-separated)
        #[arg(long)]
        configs: Option<String>,

        /// Custom prompts (comma-separated, overrides defaults)
        #[arg(long)]
        prompts: Option<String>,

        /// Save results to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Compute device for GPU-accelerated profiling matmuls.
        /// "auto" picks the best available device (CUDA → HIP → Metal → CPU).
        /// Only takes effect when compiled with --features cuda/hip/metal.
        #[arg(long, default_value = "auto")]
        device: String,

        /// Path to a portable importance-profile JSON (plain `[f64, …]` array).
        /// Configs B and C skip their profiling forward-pass and use these
        /// pre-computed layer-importance scores instead.
        /// Enables "profile on large machine, run on small device" workflow.
        #[arg(long)]
        profile_from: Option<PathBuf>,

        /// Save the importance scores produced by config B to a portable profile
        /// JSON file at this path.  Use with --profile-from on a smaller machine.
        #[arg(long)]
        save_profile: Option<PathBuf>,
    },

    /// Benchmark fused CUDA kernels using random weights (no model download required).
    ///
    /// Allocates random F16 weight matrices matching the given model-size template,
    /// runs `iters` decode forward passes on GPU, and reports tok/s.
    /// Designed for 5-minute CI validation of custom CUDA kernel integration.
    ///
    /// Only available when compiled with --features cuda.
    BenchRandom {
        /// Compute device: cuda:0, cpu (cuda:0 recommended)
        #[arg(long, default_value = "cuda:0")]
        device: String,

        /// Model architecture size to emulate: 1b, 3b, 8b
        #[arg(long, default_value = "1b")]
        model_size: String,

        /// Number of decode iterations to time
        #[arg(long, default_value = "2000")]
        iters: usize,

        /// Disable fused CUDA kernels (benchmark candle unfused baseline instead).
        /// Compares the same binary with/without fused ops for A/B measurement.
        #[arg(long)]
        no_fused: bool,

        /// Weight format for random weights: f16 (default) or w4 (Q4_0 packed INT4).
        /// Use w4 to benchmark the W4A16 decode kernel vs the F16 kernel.
        #[arg(long, default_value = "f16")]
        quantize: String,

        /// Capture the decode pass as a CUDA graph after warmup and replay it.
        /// Eliminates per-step kernel-launch overhead (~200 launches × 3 µs).
        /// Requires --quantize w4 and CUDA feature.
        #[arg(long)]
        graph: bool,
    },

    /// Score a text sequence: compute per-token NLL and aggregate perplexity.
    ///
    /// Prints: PPL, mean NLL, and the NLL of each token.
    /// Use --output to save JSON results for downstream evaluation scripts.
    Perplexity {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Text to score (the full sequence)
        #[arg(short, long)]
        text: String,

        /// Hot memory budget in MB (for paged mode)
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB (for paged mode)
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Auto-detect available memory budgets (uses 80% of available RAM)
        #[arg(long)]
        auto_budget: bool,

        /// Quantize weights after loading: none, q4, q8 (default: none)
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Compute device: auto, cpu, cuda:0
        #[arg(long, default_value = "auto")]
        device: String,

        /// Save full result to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Score multiple candidate completions for a shared context (HellaSwag-style).
    ///
    /// Prints normalized log-likelihood for each candidate and marks the winner.
    /// Candidates are separated by '\n' in the --candidates argument.
    ScoreCompletions {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Shared context prefix
        #[arg(long)]
        context: String,

        /// Newline-separated candidate completions
        #[arg(long)]
        candidates: String,

        /// Hot memory budget in MB
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Auto-detect available memory budgets
        #[arg(long)]
        auto_budget: bool,

        /// Quantize weights: none, q4, q8 (default: none)
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Compute device: auto, cpu, cuda:0
        #[arg(long, default_value = "auto")]
        device: String,

        /// Save result to JSON
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Batch perplexity: load model once, score many sequences from a JSON file.
    ///
    /// Input JSON: array of strings.
    /// Output JSON: {"results": [{"ppl": f64, "mean_nll": f64, "n_tokens": usize}], "ppl_overall": f64}
    BatchPerplexity {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Path to input JSON file (array of text strings)
        #[arg(long)]
        texts_file: PathBuf,

        /// Hot memory budget in MB
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Quantize weights: none, q4, q8
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Compute device: auto, cpu, cuda:0
        #[arg(long, default_value = "auto")]
        device: String,

        /// Save results to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Batch HellaSwag: load model once, score many examples from a JSON file.
    ///
    /// Input JSON: array of {"ctx": str, "endings": [str, str, str, str], "label": int}.
    /// Output JSON: {"accuracy": f64, "correct": usize, "total": usize, "predictions": [...]}
    BatchHellaswag {
        /// Path to model directory OR HuggingFace model ID
        #[arg(short, long)]
        model: String,

        /// Path to input JSON file (array of HellaSwag examples)
        #[arg(long)]
        examples_file: PathBuf,

        /// Hot memory budget in MB
        #[arg(long, default_value = "512")]
        hot_budget_mb: usize,

        /// Warm memory budget in MB
        #[arg(long, default_value = "2048")]
        warm_budget_mb: usize,

        /// Quantize weights: none, q4, q8
        #[arg(long, default_value = "none")]
        quantize: String,

        /// Compute device: auto, cpu, cuda:0
        #[arg(long, default_value = "auto")]
        device: String,

        /// Save results to JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// List supported architectures
    Architectures,

    /// List available compute devices (CPU, CUDA, HIP, Metal, Vulkan)
    Devices,

    /// Start the NVE HTTP inference server (requires Python SDK)
    Serve {
        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8000")]
        port: u16,

        /// Model to pre-load at startup
        #[arg(short, long)]
        model: Option<String>,

        /// Number of inference worker threads
        #[arg(short, long, default_value = "1")]
        workers: usize,

        /// Compute device for inference: auto, cpu, cuda:0, hip:0, metal, vulkan:0
        /// Defaults to auto (picks the GPU with the most free VRAM, falls back to CPU).
        #[arg(long, default_value = "auto")]
        device: String,

        /// Log level (debug, info, warning, error)
        #[arg(long, default_value = "info")]
        log_level: String,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Save your HuggingFace access token
    SetToken {
        /// Your HuggingFace access token (starts with hf_)
        token: String,
    },
    /// Display current configuration
    Show,
    /// Reset configuration to defaults
    Reset,
}

fn main() {
    // Limit Rayon to 1 thread to prevent parallel allocator pressure during
    // AWQ quantization on memory-constrained machines (≤4 GB RAM).
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .ok();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    let cli = Cli::parse();
    let nve_config = NveConfig::load();

    let result = match cli.command {
        Commands::Chat {
            model,
            hot_budget_mb,
            warm_budget_mb,
            auto_budget,
            hot_only,
            profile,
            active_layers,
            quantize,
            max_tokens,
            temperature,
            top_p,
        } => {
            let model_id = model.unwrap_or_default();
            // If model is empty, prompt the user.
            let model_id = if model_id.is_empty() {
                print!("Enter model ID or path: ");
                io::stdout().flush().unwrap_or(());
                let mut buf = String::new();
                io::stdin().read_line(&mut buf).unwrap_or(0);
                buf.trim().to_string()
            } else {
                model_id
            };

            cli::chat::run_chat(
                &model_id,
                &nve_config,
                hot_budget_mb,
                warm_budget_mb,
                auto_budget,
                hot_only,
                profile,
                active_layers,
                quantize,
                max_tokens,
                temperature,
                top_p,
            )
        }

        Commands::Config { action } => match action {
            ConfigAction::SetToken { token } => {
                let mut cfg = NveConfig::load();
                cfg.hf_token = Some(token.clone());
                cfg.save().map(|_| {
                    println!(
                        "HF token saved to {:?}",
                        NveConfig::config_path()
                    );
                })
            }
            ConfigAction::Show => {
                let cfg = NveConfig::load();
                println!("NVE Configuration ({})", NveConfig::config_path().display());
                println!();
                let token_display = cfg.hf_token.as_deref().map(|t| {
                    if t.len() > 8 { format!("{}…", &t[..8]) } else { "set".into() }
                }).unwrap_or_else(|| "(not set)".into());
                println!("  hf_token:       {}", token_display);
                println!("  hot_budget_mb:  {}", cfg.hot_budget_mb);
                println!("  warm_budget_mb: {}", cfg.warm_budget_mb);
                println!("  auto_budget:    {}", cfg.auto_budget);
                println!("  quantize:       {}", cfg.quantize);
                println!("  temperature:    {}", cfg.temperature);
                println!("  top_p:          {}", cfg.top_p);
                println!("  max_tokens:     {}", cfg.max_tokens);
                Ok(())
            }
            ConfigAction::Reset => NveConfig::reset().map(|_| {
                println!("Configuration reset to defaults.");
            }),
        },

        Commands::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            paged,
            hot_only,
            active_layers,
            auto_budget,
            hot_budget_mb,
            warm_budget_mb,
            profile,
            quantize,
            legacy,
            device,
        } => {
            let resolved_device = device::Device::resolve(&device);
            log::info!("Generate using device: {}", resolved_device);
            let (hot, warm) = if auto_budget {
                auto_detect_budgets()
            } else {
                (hot_budget_mb, warm_budget_mb)
            };
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize value '{}'. Use: none, q4, q8, q3, q2, q1, pg:X.X", quantize);
                std::process::exit(1);
            });
            let is_pg = matches!(quant_mode, QuantMode::ProfileGuided(_));
            if is_pg && !paged && !hot_only {
                eprintln!("Profile-guided quantization (pg:X.X) requires --paged mode");
                std::process::exit(1);
            }
            let profile = profile || is_pg;
            let paged = paged || hot_only || is_pg;
            run_generate(
                &model, &prompt, max_tokens, temperature, top_p,
                paged, hot_only, active_layers, hot, warm, profile, quant_mode, legacy,
                nve_config.hf_token_resolved().as_deref(),
            )
        },

        Commands::Benchmark {
            model,
            max_tokens,
            auto_budget,
            hot_budget_mb,
            warm_budget_mb,
            quantize,
            output,
        } => {
            let (hot, warm) = if auto_budget {
                auto_detect_budgets()
            } else {
                (hot_budget_mb, warm_budget_mb)
            };
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize value '{}'. Use: none, q4, q8, q3, q2, q1, pg:X.X", quantize);
                std::process::exit(1);
            });
            run_benchmark(
                &model, max_tokens, hot, warm, quant_mode, output,
                nve_config.hf_token_resolved().as_deref(),
            )
        },

        Commands::AbcTest {
            model,
            max_tokens,
            auto_budget,
            hot_budget_mb,
            warm_budget_mb,
            target_bpw,
            active_layers,
            configs,
            prompts,
            output,
            device,
            profile_from,
            save_profile,
        } => {
            let (hot, warm) = if auto_budget {
                auto_detect_budgets()
            } else {
                (hot_budget_mb, warm_budget_mb)
            };
            run_abc_test(
                &model, max_tokens, hot, warm, target_bpw, active_layers, configs, prompts,
                output, &device, nve_config.hf_token_resolved().as_deref(),
                profile_from, save_profile,
            )
        },

        Commands::Info { model } => run_info(&model, nve_config.hf_token_resolved().as_deref()),
        Commands::Download { model_id, output } => {
            run_download(&model_id, output, nve_config.hf_token_resolved().as_deref())
        }
        Commands::BenchRandom { device, model_size, iters, no_fused, quantize, graph } => {
            run_bench_random(&device, &model_size, iters, no_fused, &quantize, graph)
        }
        Commands::Perplexity {
            model, text, hot_budget_mb, warm_budget_mb, auto_budget, quantize, device, output,
        } => {
            let (hot, warm) = if auto_budget {
                auto_detect_budgets()
            } else {
                (hot_budget_mb, warm_budget_mb)
            };
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize '{}'. Use: none, q4, q8", quantize);
                std::process::exit(1);
            });
            run_perplexity(
                &model, &text, hot, warm, quant_mode, &device, output,
                nve_config.hf_token_resolved().as_deref(),
            )
        }

        Commands::ScoreCompletions {
            model, context, candidates, hot_budget_mb, warm_budget_mb, auto_budget,
            quantize, device, output,
        } => {
            let (hot, warm) = if auto_budget {
                auto_detect_budgets()
            } else {
                (hot_budget_mb, warm_budget_mb)
            };
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize '{}'. Use: none, q4, q8", quantize);
                std::process::exit(1);
            });
            run_score_completions(
                &model, &context, &candidates, hot, warm, quant_mode, &device, output,
                nve_config.hf_token_resolved().as_deref(),
            )
        }

        Commands::BatchPerplexity {
            model, texts_file, hot_budget_mb, warm_budget_mb, quantize, device, output,
        } => {
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize '{}'. Use: none, q4, q8", quantize);
                std::process::exit(1);
            });
            run_batch_perplexity(
                &model, &texts_file, hot_budget_mb, warm_budget_mb, quant_mode, &device, output,
                nve_config.hf_token_resolved().as_deref(),
            )
        }

        Commands::BatchHellaswag {
            model, examples_file, hot_budget_mb, warm_budget_mb, quantize, device, output,
        } => {
            let quant_mode = QuantMode::from_str(&quantize).unwrap_or_else(|| {
                eprintln!("Invalid --quantize '{}'. Use: none, q4, q8", quantize);
                std::process::exit(1);
            });
            run_batch_hellaswag(
                &model, &examples_file, hot_budget_mb, warm_budget_mb, quant_mode, &device, output,
                nve_config.hf_token_resolved().as_deref(),
            )
        }

        Commands::Architectures => run_list_architectures(),
        Commands::Devices => run_list_devices(),
        Commands::Serve { host, port, model, workers, device, log_level } => {
            run_serve(&host, port, model.as_deref(), workers, &device, &log_level, &nve_config)
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Auto-detect available system memory and return (hot_budget_mb, warm_budget_mb).
/// Uses 80% of available RAM, split 25% hot / 75% warm.
fn auto_detect_budgets() -> (usize, usize) {
    let mut available_mb: usize = 4096; // fallback

    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
        for line in contents.lines() {
            if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        available_mb = kb / 1024;
                    }
                }
                break;
            }
        }
    }

    let budget_mb = (available_mb as f64 * 0.8) as usize;
    let hot = budget_mb / 4;
    let warm = budget_mb - hot;

    log::info!(
        "Auto budget: {available_mb} MB available → hot={hot} MB, warm={warm} MB (80% of available)"
    );
    (hot, warm)
}

fn resolve_model(model: &str, hf_token: Option<&str>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = std::path::Path::new(model);
    if path.is_dir() && path.join("config.json").exists() {
        Ok(path.to_path_buf())
    } else {
        Ok(hub::resolve_model_path(model, None, hf_token)?)
    }
}

fn run_generate(
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    paged: bool,
    hot_only: bool,
    active_layers: Option<usize>,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    profile: bool,
    quant_mode: QuantMode,
    legacy: bool,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
    let tokens = tokenizer.encode_with_bos(prompt);

    println!("Prompt: {}", prompt);
    println!("Tokens: {} chars, {} tokens\n", prompt.len(), tokens.len());

    if legacy {
        return run_generate_legacy(
            &model_dir, &tokenizer, &tokens, prompt,
            max_tokens, temperature, top_p, paged, hot_budget_mb, warm_budget_mb,
        );
    }

    if paged {
        let paged_config = PagedConfig {
            hot_budget_bytes: hot_budget_mb * 1024 * 1024,
            warm_budget_bytes: warm_budget_mb * 1024 * 1024,
            prefetch_ahead: 2,
            profile_activations: false,
            hot_only_mode: hot_only && !profile,
            active_layers: if profile { None } else { active_layers },
            quant_mode,
            device: "auto".to_string(),
        };

        if quant_mode != QuantMode::None {
            log::info!("Quantization mode: {} (quantize-on-load)", quant_mode);
        }

        let mut model = PagedModel::from_dir(&model_dir, paged_config)?;

        if profile {
            // Run importance profiling pass (works with or without --hot-only).
            // Saves scores to ~/.cache/nve/importance/<model_key>.json.
            const MAX_PROFILE_TOKENS: usize = 64;
            let profile_tokens = &tokens[..tokens.len().min(MAX_PROFILE_TOKENS)];
            model.profile_layer_importance(profile_tokens)?;

            if hot_only {
                let layer_size = model.layer_size_bytes();
                let total_budget = hot_budget_mb * 1024 * 1024 + warm_budget_mb * 1024 * 1024;
                let active_count = if let Some(n) = active_layers {
                    n
                } else {
                    (total_budget / layer_size.max(1)).min(model.num_layers())
                };
                model.apply_profiled_hot_only(active_count);
            }
        } else {
            // Try loading cached importance scores for mixed-precision W4A8/16 dispatch.
            // Normalize raw activation magnitudes to [0,1] so NVE_W4A8_THRESHOLD is
            // always a fraction: 0.7 → top 30% of layers use W4A16, rest use W4A8.
            let num_layers = model.num_layers();
            if let Some(scores) = crate::importance_cache::ImportanceCache::load(&model_dir, num_layers) {
                let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                let normalized: Vec<f64> = if range > 1e-9 {
                    scores.iter().map(|&s| (s - min) / range).collect()
                } else {
                    vec![0.5; scores.len()]
                };
                log::info!(
                    "Loaded importance cache: {} layers, min={:.3} max={:.3} (normalized to [0,1])",
                    normalized.len(), min, max
                );
                model.inject_importance_scores(normalized);
            }
        }

        let gen = model.generate(&tokens, max_tokens, temperature, top_p)?;

        print!("{}", prompt);
        let text = tokenizer.decode(&gen.tokens);
        println!("{}", text);

        println!("\n--- Stats ---");
        gen.display();
        println!("{}", model.memory_report());
    } else {
        let mut gen_model = GenericModel::from_dir(&model_dir)?;

        print!("{}", prompt);
        io::stdout().flush()?;

        let gen = gen_model.generate(&tokens, max_tokens, temperature, top_p);

        for &tok in &gen.tokens {
            print!("{}", tokenizer.decode_token(tok));
            io::stdout().flush()?;
        }
        println!();

        println!("\n--- Stats ---");
        gen.display();
    }

    Ok(())
}

fn run_generate_legacy(
    model_dir: &PathBuf,
    tokenizer: &Tokenizer,
    tokens: &[u32],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    paged: bool,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if paged {
        let paged_config = PagedConfig {
            hot_budget_bytes: hot_budget_mb * 1024 * 1024,
            warm_budget_bytes: warm_budget_mb * 1024 * 1024,
            prefetch_ahead: 2,
            profile_activations: false,
            hot_only_mode: false,
            active_layers: None,
            quant_mode: QuantMode::None,
            device: "auto".to_string(),
        };

        let mut model = PagedModel::from_dir(model_dir, paged_config)?;
        let gen = model.generate(tokens, max_tokens, temperature, top_p)?;

        print!("{}", prompt);
        let text = tokenizer.decode(&gen.tokens);
        println!("{}", text);

        println!("\n--- Stats ---");
        gen.display();
        println!("{}", model.memory_report());
    } else {
        let mut model = LlamaModel::from_dir(model_dir)?;

        print!("{}", prompt);
        io::stdout().flush()?;

        let gen = model.generate(tokens, max_tokens, temperature, top_p);

        for &tok in &gen.tokens {
            print!("{}", tokenizer.decode_token(tok));
            io::stdout().flush()?;
        }
        println!();

        println!("\n--- Stats ---");
        gen.display();
    }

    Ok(())
}

fn run_benchmark(
    model: &str,
    max_tokens: usize,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;

    let bench_config = benchmark::BenchmarkConfig {
        prompts: vec![
            "The meaning of life is".into(),
            "def fibonacci(n):".into(),
            "Explain the theory of relativity in simple terms:".into(),
            "Once upon a time in a land far away".into(),
            "The three most important things in programming are".into(),
        ],
        max_new_tokens: max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        warmup_runs: 0,
        paged_config: PagedConfig {
            hot_budget_bytes: hot_budget_mb * 1024 * 1024,
            warm_budget_bytes: warm_budget_mb * 1024 * 1024,
            prefetch_ahead: 2,
            profile_activations: false,
            hot_only_mode: false,
            active_layers: None,
            quant_mode,
            device: "auto".to_string(),
        },
    };

    let baseline = benchmark::benchmark_baseline(&model_dir, &tokenizer, &bench_config)?;
    let paged = benchmark::benchmark_paged(&model_dir, &tokenizer, &bench_config)?;
    benchmark::print_comparison(&baseline, &paged);

    if let Some(output_path) = output {
        let report = serde_json::json!({
            "baseline": baseline,
            "paged": paged,
        });
        std::fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;
        println!("\nResults saved to {:?}", output_path);
    }

    Ok(())
}

fn run_abc_test(
    model: &str,
    max_tokens: usize,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    target_bpw: f32,
    active_layers: Option<usize>,
    configs_str: Option<String>,
    prompts_str: Option<String>,
    output: Option<PathBuf>,
    device: &str,
    hf_token: Option<&str>,
    profile_from: Option<PathBuf>,
    save_profile: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use abc_benchmark::{AbcBenchmarkParams, AbcConfig, QualityBudget};

    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;

    let configs = if let Some(ref s) = configs_str {
        let mut cfgs = Vec::new();
        for part in s.split(',') {
            match part.trim().to_lowercase().as_str() {
                "baseline" | "base" => cfgs.push(AbcConfig::Baseline),
                "a" | "quant" => cfgs.push(AbcConfig::QuantOnly),
                "b" | "profile" | "hot" => cfgs.push(AbcConfig::ProfiledHotOnly),
                "c" | "pg" | "combined" => cfgs.push(AbcConfig::ProfiledQuantized),
                "all" => { cfgs = AbcConfig::all(); break; },
                other => {
                    eprintln!("Unknown config '{}'. Use: baseline, a, b, c, all", other);
                    std::process::exit(1);
                }
            }
        }
        cfgs
    } else {
        AbcConfig::all()
    };

    let prompts = if let Some(ref s) = prompts_str {
        s.split("||").map(|p| p.trim().to_string()).collect()
    } else {
        AbcBenchmarkParams::default().prompts
    };

    let params = AbcBenchmarkParams {
        prompts,
        max_new_tokens: max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        hot_budget_bytes: hot_budget_mb * 1024 * 1024,
        warm_budget_bytes: warm_budget_mb * 1024 * 1024,
        target_bpw,
        configs,
        active_layers,
        stability_check: true,
        quality_budget: QualityBudget::default(),
        device: device.to_string(),
        profile_path: profile_from,
    };

    let report = abc_benchmark::run_abc_benchmark(&model_dir, &tokenizer, &params)?;

    // Optionally export the importance scores produced by config B.
    if let Some(save_path) = save_profile {
        let scores_opt = report.configurations.iter()
            .find(|c| c.config == "B_profiled_hot")
            .and_then(|c| c.paging_info.as_ref())
            .map(|p| &p.layer_importance_scores);
        if let Some(scores) = scores_opt {
            let json = serde_json::to_string_pretty(scores)?;
            std::fs::write(&save_path, &json)?;
            println!("Profile saved → {:?}", save_path);
        } else {
            eprintln!("Warning: config B not found in report — profile not saved");
        }
    }

    if let Some(output_path) = output {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&output_path, &json)?;
        println!("\nResults saved to {:?}", output_path);
    }

    Ok(())
}

fn run_info(model: &str, hf_token: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model(model, hf_token)?;

    match arch::UnifiedConfig::from_model_dir(&model_dir) {
        Ok(config) => {
            println!("=== Model Information ===\n");
            println!("  Architecture:       {}", config.arch);
            println!("  Hidden size:        {}", config.hidden_size);
            println!("  Layers:             {}", config.num_hidden_layers);
            println!("  Attention heads:    {}", config.num_attention_heads);
            println!("  KV heads:           {}", config.num_key_value_heads);
            println!("  Head dim:           {}", config.head_dim());
            println!("  Intermediate:       {}", config.intermediate_size);
            println!("  Vocab size:         {}", config.vocab_size);
            println!("  Max positions:      {}", config.max_position_embeddings);
            println!("  Norm type:          {:?}", config.norm_type);
            println!("  Position encoding:  {:?}", config.pos_encoding);
            println!("  FFN type:           {:?}", config.ffn_type);
            println!("  RoPE theta:         {}", config.rope_theta);
            println!("  Tied embeddings:    {}", config.tie_word_embeddings);
            println!("  Attention bias:     {}", config.attn_bias);
            println!("  Parallel attn+ffn:  {}", config.parallel_attn_ffn);
            println!("  Est. parameters:    {:.1}B", config.estimated_params() as f64 / 1e9);
            println!("  Est. size (bf16):   {:.1} GB", config.estimated_size_bytes() as f64 / 1024.0 / 1024.0 / 1024.0);
        }
        Err(_) => {
            let config = LlamaConfig::from_model_dir(&model_dir)?;
            println!("=== Model Information (Llama) ===\n");
            println!("  Hidden size:        {}", config.hidden_size);
            println!("  Layers:             {}", config.num_hidden_layers);
            println!("  Attention heads:    {}", config.num_attention_heads);
            println!("  KV heads:           {}", config.num_key_value_heads);
            println!("  Head dim:           {}", config.head_dim());
            println!("  Intermediate:       {}", config.intermediate_size);
            println!("  Vocab size:         {}", config.vocab_size);
        }
    }

    let safetensors_exists = model_dir.join("model.safetensors").exists()
        || model_dir.join("model.safetensors.index.json").exists();
    let tokenizer_exists = model_dir.join("tokenizer.json").exists();
    let config_exists = model_dir.join("config.json").exists();

    println!("\n  Files:");
    println!("    config.json:       {}", if config_exists { "found" } else { "MISSING" });
    println!("    tokenizer.json:    {}", if tokenizer_exists { "found" } else { "MISSING" });
    println!("    model.safetensors: {}", if safetensors_exists { "found" } else { "MISSING" });

    if safetensors_exists {
        let weights = safetensors::ModelWeights::load(&model_dir)?;
        println!("    Tensors:           {}", weights.len());
        println!("    Total size:        {:.1} GB", weights.total_size_bytes() as f64 / 1024.0 / 1024.0 / 1024.0);
    }

    Ok(())
}

fn run_download(
    model_id: &str,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dest = if let Some(out) = output {
        std::fs::create_dir_all(&out)?;
        let _id = hub::ModelId::parse(model_id)?;
        hub::resolve_model_path(model_id, Some(&out), hf_token)?
    } else {
        hub::resolve_model_path(model_id, None, hf_token)?
    };
    println!("Model downloaded to: {:?}", dest);
    Ok(())
}

fn run_list_architectures() -> Result<(), Box<dyn std::error::Error>> {
    println!("Supported architectures:");
    println!();
    println!("  Architecture    model_type          FFN       Norm       Position");
    println!("  ─────────────   ─────────────────   ───────   ────────   ────────");
    println!("  Llama           llama               SwiGLU    RMSNorm    RoPE");
    println!("  Mistral         mistral             SwiGLU    RMSNorm    RoPE");
    println!("  Qwen2           qwen2               SwiGLU    RMSNorm    RoPE");
    println!("  Phi-3           phi3/phi            SwiGLU    RMSNorm    RoPE");
    println!("  Gemma           gemma               GeGLU     RMSNorm    RoPE");
    println!("  Gemma2          gemma2              GeGLU     RMSNorm    RoPE");
    println!("  GPT-NeoX        gpt_neox            GELU      LayerNorm  RoPE");
    println!("  GPT-2           gpt2                GELU      LayerNorm  Learned");
    println!("  Falcon          falcon              GELU      LayerNorm  RoPE");
    println!("  StableLM        stablelm            SwiGLU    LayerNorm  RoPE");
    println!("  StarCoder2      starcoder2          GELU      LayerNorm  RoPE");
    println!("  InternLM2       internlm2           SwiGLU    RMSNorm    RoPE");
    println!("  OLMo            olmo                SwiGLU    LayerNorm  RoPE");
    println!("  DeepSeek        deepseek            SwiGLU    RMSNorm    RoPE");
    println!();
    println!("Usage:");
    println!("  nve generate -m meta-llama/Llama-3.2-1B -p \"Hello world\"");
    println!("  nve generate -m microsoft/phi-2 -p \"Write a poem\"");
    println!("  nve generate -m ./my-local-model -p \"Testing\"");
    println!("  nve info -m google/gemma-2b");
    println!("  nve download mistralai/Mistral-7B-v0.1");
    Ok(())
}

fn run_serve(
    host: &str,
    port: u16,
    model: Option<&str>,
    workers: usize,
    device: &str,
    log_level: &str,
    config: &NveConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve the requested device through the Rust device abstraction so that
    // unknown device strings get a clear warning before Python even starts.
    let resolved = device::Device::resolve(device);
    println!("Starting NVE server on {}:{} (device: {})", host, port, resolved);
    println!("Run `pip install -e ./python` if not already installed.\n");

    let mut cmd = std::process::Command::new("python3");
    cmd.arg("-m")
        .arg("nve.serve")
        .arg("--host").arg(host)
        .arg("--port").arg(port.to_string())
        .arg("--workers").arg(workers.to_string())
        .arg("--log-level").arg(log_level);

    // Pass resolved device to Python server via NVE_DEVICE env var.
    // Python DeviceManager reads NVE_DEVICE for its preferred_device.
    if !resolved.is_cpu() || device != "auto" {
        cmd.env("NVE_DEVICE", resolved.name());
    }

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }
    if let Some(ref token) = config.hf_token_resolved() {
        cmd.env("HF_TOKEN", token);
    }

    let status = cmd.status()?;
    if !status.success() {
        return Err(format!("Server exited with status: {}", status).into());
    }
    Ok(())
}

fn run_list_devices() -> Result<(), Box<dyn std::error::Error>> {
    let devices = device::enumerate_devices();
    println!("NVE — Available Compute Devices");
    println!("{}", "─".repeat(50));
    for d in &devices {
        let vram = match d.vram_bytes {
            Some(b) => format!("{:.1} GB VRAM", b as f64 / 1024f64.powi(3)),
            None => "unified memory".to_string(),
        };
        let free = match d.free_vram_bytes {
            Some(b) => format!(" ({:.1} GB free)", b as f64 / 1024f64.powi(3)),
            None => String::new(),
        };
        let caps = {
            let mut c = vec!["fp16"];
            if d.supports_bf16 { c.push("bf16"); }
            c.join(", ")
        };
        println!(
            "  {:12} {:30} {}{}  [{}]",
            d.device.name(), d.name, vram, free, caps
        );
    }
    println!("{}", "─".repeat(50));
    println!("\nBuild flags for GPU support:");
    println!("  NVIDIA CUDA:  cargo build --release --features cuda");
    println!("  AMD ROCm:     cargo build --release --features hip");
    println!("  Apple Metal:  cargo build --release --features metal");
    println!("  Vulkan:       cargo build --release --features vulkan");
    println!("  Intel MKL:    cargo build --release --features mkl");
    Ok(())
}

/// Benchmark fused vs unfused CUDA kernels using random synthetic weights.
///
/// No model download required. Creates random F16 weight matrices directly on
/// the GPU, constructs a GpuInferenceState, and runs `iters` decode forward
/// passes through all transformer layers.  Reports tok/s.
///
/// Pass `--no-fused` to disable the custom CUDA kernel path and benchmark the
/// candle unfused baseline in the same binary via `NVE_NO_FUSED=1`.
fn run_bench_random(
    device_str: &str,
    model_size: &str,
    iters: usize,
    no_fused: bool,
    quantize: &str,
    use_graph: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
    return run_bench_random_gpu(device_str, model_size, iters, no_fused, quantize, use_graph)
        .map_err(|e| e.into());

    #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
    {
        eprintln!("bench-random requires a GPU feature. Rebuild with:");
        eprintln!("  CUDA_COMPUTE_CAP=75 cargo build --release --features cuda");
        std::process::exit(1);
    }
}

#[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
fn run_bench_random_gpu(
    device_str: &str,
    model_size: &str,
    iters: usize,
    no_fused: bool,
    quantize: &str,
    use_graph: bool,
) -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor as CT};
    use gpu_layer::{gpu_layer_forward_decode, GpuInferenceState, GpuKvCache, GpuLayerWeights, GpuWeight};
    use crate::arch::{FfnType, NormType};

    // ── Model-size templates ──────────────────────────────────────────────────
    let (hidden, intermediate, num_heads, num_kv_heads, head_dim, num_layers) =
        match model_size {
            "3b" => (3072usize, 8192usize, 24usize, 8usize, 128usize, 28usize),
            "8b" => (4096, 14336, 32, 8, 128, 32),
            _    => (2048, 8192,  32, 8, 64,  16),  // 1b (default)
        };

    if no_fused {
        // Checked at runtime in apply_norm_gpu and apply_rope_decode to bypass
        // the fused kernel fast-path and fall through to unfused candle ops.
        std::env::set_var("NVE_NO_FUSED", "1");
    }

    println!("NVE bench-random");
    println!("  model_size : {}", model_size);
    println!("  hidden     : {}, intermediate: {}", hidden, intermediate);
    println!("  heads      : {}/{}, head_dim: {}", num_heads, num_kv_heads, head_dim);
    println!("  layers     : {}", num_layers);
    println!("  iters      : {}", iters);
    println!("  fused      : {}", !no_fused);
    println!("  quantize   : {}", quantize);
    println!("  graph      : {}", use_graph);

    // ── Candle device ─────────────────────────────────────────────────────────
    let device: Device = {
        #[cfg(feature = "cuda")]
        {
            if device_str.starts_with("cuda") {
                let idx = device_str
                    .strip_prefix("cuda:")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0usize);
                Device::new_cuda(idx)?
            } else {
                Device::Cpu
            }
        }
        #[cfg(not(feature = "cuda"))]
        { Device::Cpu }
    };

    println!("  device     : {:?}", device);

    // ── Random weight helpers ─────────────────────────────────────────────────
    // Small constant scale so activations don't diverge across 16-32 layers.
    let use_w4 = quantize == "w4";

    // F16 projection: randn → F16 GpuWeight::F16
    let f16_proj = |rows: usize, cols: usize| -> candle_core::Result<GpuWeight> {
        let t = CT::randn(0f32, 1.0, (rows, cols), &device)?
            .affine(0.01, 0.0)?
            .to_dtype(DType::F16)?;
        Ok(GpuWeight::F16(t))
    };

    // W4 projection: random nibbles + unit scales → GpuWeight::W4
    // Nibble values are uniformly random (0-15); scale = 0.01 so magnitudes match F16.
    let w4_proj = |rows: usize, cols: usize| -> candle_core::Result<GpuWeight> {
        use gpu_layer::GpuW4;
        assert!(cols % 32 == 0, "cols must be divisible by Q4 block_size=32");
        let blocks_per_row = cols / 32;
        // Random nibbles: uniform [0, 255] → cast to U8 (2 INT4 per byte)
        let nibbles_t = CT::rand::<_, f32>(0.0, 255.0, (rows, cols / 2), &device)?
            .to_dtype(DType::U8)?;
        // Unit scales: constant 0.01 so dequant gives values ~same magnitude as F16 randn
        let scales_t  = CT::full(0.01f32, (rows, blocks_per_row), &device)?;
        Ok(GpuWeight::W4(GpuW4 {
            nibbles:    nibbles_t,
            scales:     scales_t,
            awq_scales: None,
            n: rows,
            k: cols,
        }))
    };

    let make_proj = |rows: usize, cols: usize| -> candle_core::Result<GpuWeight> {
        if use_w4 { w4_proj(rows, cols) } else { f16_proj(rows, cols) }
    };

    let norm_w = |n: usize| CT::ones((n,), DType::F16, &device);

    // ── Build per-layer GpuLayerWeights directly ──────────────────────────────
    let mut layer_weights: Vec<Option<GpuLayerWeights>> = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        layer_weights.push(Some(GpuLayerWeights {
            q_proj:              make_proj(num_heads * head_dim, hidden)?,
            k_proj:              make_proj(num_kv_heads * head_dim, hidden)?,
            v_proj:              make_proj(num_kv_heads * head_dim, hidden)?,
            o_proj:              make_proj(hidden, num_heads * head_dim)?,
            gate_proj:           Some(make_proj(intermediate, hidden)?),
            up_proj:             make_proj(intermediate, hidden)?,
            down_proj:           make_proj(hidden, intermediate)?,
            input_norm:          norm_w(hidden)?,
            post_attn_norm:      norm_w(hidden)?,
            input_norm_bias:     None,
            post_attn_norm_bias: None,
            q_bias:              None,
            k_bias:              None,
            v_bias:              None,
            o_bias:              None,
        }));
    }

    // ── RoPE tables ───────────────────────────────────────────────────────────
    let max_seq = 2048usize;
    let half_dim = head_dim / 2;
    // Precompute cos/sin: theta_i = 1 / 10000^(2i/head_dim)
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0f32 / (10000f32.powf(2.0 * i as f32 / head_dim as f32)))
        .collect();
    let mut cos_data = vec![0f32; max_seq * half_dim];
    let mut sin_data = vec![0f32; max_seq * half_dim];
    for pos in 0..max_seq {
        for i in 0..half_dim {
            let angle = pos as f32 * freqs[i];
            cos_data[pos * half_dim + i] = angle.cos();
            sin_data[pos * half_dim + i] = angle.sin();
        }
    }
    let rope_cos = CT::from_slice(&cos_data, (max_seq, half_dim), &device)?
        .to_dtype(DType::F16)?;
    let rope_sin = CT::from_slice(&sin_data, (max_seq, half_dim), &device)?
        .to_dtype(DType::F16)?;

    // ── Build GpuInferenceState ───────────────────────────────────────────────
    let mut state = GpuInferenceState {
        device:       device.clone(),
        kv_cache:     GpuKvCache::new(num_layers),
        layer_weights,
        rope_cos:     Some(rope_cos),
        rope_sin:     Some(rope_sin),
        head_dim,
        num_heads,
        num_kv_heads,
        ffn_type:     FfnType::SwiGLU,
        norm_type:    NormType::RMSNorm,
        norm_eps:     1e-5,
        num_layers,
        #[cfg(feature = "cuda")]
        decode_graph: None,
    };

    let h0 = CT::randn(0f32, 1.0, (1, hidden), &device)?.to_dtype(DType::F16)?;

    // ── Warmup (5 tokens via standard path, not timed) ────────────────────────
    for step in 0..5usize {
        let mut h = h0.clone();
        for li in 0..num_layers {
            h = gpu_layer_forward_decode(h, li, &mut state, step, false, 0.0)?;
        }
        let _ = h.flatten_all()?.to_vec1::<half::f16>()?; // force GPU sync
        state.reset();
    }

    // ── Optionally capture CUDA graph (after warmup so kernels are JIT-compiled) ──
    #[cfg(feature = "cuda")]
    if use_graph {
        println!("Capturing CUDA decode graph …");
        state.build_decode_graph(max_seq)?;
        println!("Graph captured. Warming up graph (3 replays) …");
        for _ in 0..3usize {
            let _ = state.replay_decode_graph(&h0, 0)?
                .flatten_all()?.to_vec1::<half::f16>()?;
        }
        println!("Graph warm. Starting timed loop.");
    }

    // ── Timed decode loop ─────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();

    #[cfg(feature = "cuda")]
    if use_graph && state.has_decode_graph() {
        // Always replay at pos=0 so flash_decode_dyn sees seq_len=1 each step,
        // matching the no-graph path which resets the KV cache before every step.
        // This ensures a like-for-like comparison of kernel-launch overhead.
        for step in 0..iters {
            let h = state.replay_decode_graph(&h0, 0)?;
            if step + 1 == iters {
                let _ = h.flatten_all()?.to_vec1::<half::f16>()?;
            }
        }
    } else {
        for step in 0..iters {
            let mut h = h0.clone();
            for li in 0..num_layers {
                h = gpu_layer_forward_decode(h, li, &mut state, step, false, 0.0)?;
            }
            if step + 1 == iters {
                let _ = h.flatten_all()?.to_vec1::<half::f16>()?;
            }
            state.reset();
        }
    }

    #[cfg(not(feature = "cuda"))]
    for step in 0..iters {
        let mut h = h0.clone();
        for li in 0..num_layers {
            h = gpu_layer_forward_decode(h, li, &mut state, step, false, 0.0)?;
        }
        if step + 1 == iters {
            let _ = h.flatten_all()?.to_vec1::<half::f16>()?;
        }
        state.reset();
    }

    let elapsed_s = t0.elapsed().as_secs_f64();
    let ms_per_tok = elapsed_s * 1000.0 / iters as f64;
    let tok_per_s  = iters as f64 / elapsed_s;

    println!("---");
    println!("total_time  : {:.2}s", elapsed_s);
    println!("ms_per_tok  : {:.3} ms", ms_per_tok);
    println!("tok_per_s   : {:.1}", tok_per_s);
    println!("---");

    Ok(())
}

fn make_paged_model(
    model_dir: &std::path::Path,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    device: &str,
) -> Result<PagedModel, Box<dyn std::error::Error>> {
    let paged_config = PagedConfig {
        hot_budget_bytes: hot_budget_mb * 1024 * 1024,
        warm_budget_bytes: warm_budget_mb * 1024 * 1024,
        prefetch_ahead: 2,
        profile_activations: false,
        hot_only_mode: false,
        active_layers: None,
        quant_mode,
        device: device.to_string(),
    };
    let mut model = PagedModel::from_dir(model_dir, paged_config)?;

    // Load cached importance scores and normalize to [0,1] for W4A8/W4A16 dispatch.
    // Mirrors the normalization in run_generate so NVE_W4A8_THRESHOLD=0.7 means
    // "layers whose normalized importance >= 0.7 stay at W4A16".
    let num_layers = model.num_layers();
    if let Some(scores) = crate::importance_cache::ImportanceCache::load(model_dir, num_layers) {
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let normalized: Vec<f64> = if range > 1e-9 {
            scores.iter().map(|&s| (s - min) / range).collect()
        } else {
            vec![0.5; scores.len()]
        };
        model.inject_importance_scores(normalized);
    }

    // Initialize GPU inference if a GPU device is available.
    // Uploads hot-tier layer weights to VRAM for decode-path kernel dispatch.
    model.init_gpu().map_err(|e| format!("GPU init: {}", e))?;

    Ok(model)
}

fn run_perplexity(
    model: &str,
    text: &str,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    device: &str,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
    let tokens = tokenizer.encode_with_bos(text);

    println!("Perplexity evaluation");
    println!("  Model:  {}", model_dir.display());
    println!("  Tokens: {}", tokens.len());
    println!("  Quant:  {}", quant_mode);

    let mut paged = make_paged_model(&model_dir, hot_budget_mb, warm_budget_mb, quant_mode, device)?;

    let t0 = Instant::now();
    let nll = paged.score_sequence_fast(&tokens)?;
    let elapsed = t0.elapsed().as_secs_f64();

    if nll.is_empty() {
        eprintln!("Sequence too short to score (need >= 2 tokens)");
        std::process::exit(1);
    }

    let mean_nll = nll.iter().sum::<f64>() / nll.len() as f64;
    let ppl = mean_nll.exp();

    println!("\nResults:");
    println!("  Tokens scored : {}", nll.len());
    println!("  Mean NLL      : {:.4}", mean_nll);
    println!("  Perplexity    : {:.2}", ppl);
    println!("  Time          : {:.2}s ({:.1} tok/s)", elapsed, nll.len() as f64 / elapsed);

    if let Some(out_path) = output {
        let result = serde_json::json!({
            "model": model,
            "quant": format!("{}", quant_mode),
            "tokens": tokens.len(),
            "mean_nll": mean_nll,
            "perplexity": ppl,
            "nll": nll,
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
        println!("  Saved → {}", out_path.display());
    }

    Ok(())
}

fn run_score_completions(
    model: &str,
    context: &str,
    candidates_str: &str,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    device: &str,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
    let candidates: Vec<&str> = candidates_str.split('\n').filter(|s| !s.is_empty()).collect();

    if candidates.is_empty() {
        eprintln!("No candidates provided");
        std::process::exit(1);
    }

    println!("Scoring {} candidates for context:", candidates.len());
    println!("  {:?}", context);

    let mut paged = make_paged_model(&model_dir, hot_budget_mb, warm_budget_mb, quant_mode, device)?;

    let ctx_tokens = tokenizer.encode_with_bos(context);
    // boundary: full_nll[boundary..] gives NLL values only for the completion tokens.
    let boundary = ctx_tokens.len().saturating_sub(1);

    let mut scores = Vec::new();
    for (i, cand) in candidates.iter().enumerate() {
        let full_text = format!("{}{}", context, cand);
        // Score the full sequence so the model sees the complete context.
        // Then extract only the completion portion of the NLL array.
        let full_tokens = tokenizer.encode_with_bos(&full_text);
        let full_nll = paged.score_sequence(&full_tokens)?;

        if full_nll.len() <= boundary {
            scores.push(f64::NEG_INFINITY);
            continue;
        }
        let ending_nll = &full_nll[boundary..];
        let mean_nll = if ending_nll.is_empty() { f64::INFINITY } else {
            ending_nll.iter().sum::<f64>() / ending_nll.len() as f64
        };
        let norm_score = -mean_nll; // higher is better

        println!("  [{}] score={:.4}  {:?}", i, norm_score, cand);
        scores.push(norm_score);
    }

    let best_idx = scores.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    println!("\nBest candidate: [{}] {:?}", best_idx, candidates[best_idx]);

    if let Some(out_path) = output {
        let result = serde_json::json!({
            "model": model,
            "quant": format!("{}", quant_mode),
            "context": context,
            "candidates": candidates,
            "scores": scores,
            "best_idx": best_idx,
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&result)?)?;
        println!("Saved → {}", out_path.display());
    }

    Ok(())
}

fn run_batch_perplexity(
    model: &str,
    texts_file: &PathBuf,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    device: &str,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let texts_json = std::fs::read_to_string(texts_file)?;
    let texts: Vec<String> = serde_json::from_str(&texts_json)?;
    let n = texts.len();

    println!("BatchPerplexity: {} sequences, model={}", n, model);

    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
    let mut paged = make_paged_model(&model_dir, hot_budget_mb, warm_budget_mb, quant_mode, device)?;

    let mut results = Vec::with_capacity(n);
    let mut all_nll: Vec<f64> = Vec::new();
    let mut failed = 0usize;
    let t_start = Instant::now();

    for (i, text) in texts.iter().enumerate() {
        let tokens = tokenizer.encode_with_bos(text);
        match paged.score_sequence(&tokens) {
            Ok(nll) if !nll.is_empty() => {
                let mean_nll = nll.iter().sum::<f64>() / nll.len() as f64;
                let ppl = mean_nll.exp();
                results.push(serde_json::json!({
                    "seq_idx": i,
                    "ppl": ppl,
                    "mean_nll": mean_nll,
                    "n_tokens": nll.len(),
                }));
                all_nll.extend_from_slice(&nll);
            }
            Ok(_) => {
                failed += 1;
                results.push(serde_json::json!({"seq_idx": i, "failed": true, "reason": "too_short"}));
            }
            Err(e) => {
                failed += 1;
                results.push(serde_json::json!({"seq_idx": i, "failed": true, "reason": format!("{}", e)}));
            }
        }

        if (i + 1) % 10 == 0 || i + 1 == n {
            let running_ppl = if all_nll.is_empty() { f64::NAN } else {
                (all_nll.iter().sum::<f64>() / all_nll.len() as f64).exp()
            };
            let elapsed = t_start.elapsed().as_secs_f64();
            println!("  [PPL] {}/{} sequences done, running PPL={:.2}, elapsed={:.1}s",
                     i + 1, n, running_ppl, elapsed);
            io::stdout().flush().ok();
        }
    }

    let overall_ppl = if all_nll.is_empty() { f64::NAN } else {
        (all_nll.iter().sum::<f64>() / all_nll.len() as f64).exp()
    };
    println!("  [PPL] Final: PPL={:.2}, tokens={}, failed={}", overall_ppl, all_nll.len(), failed);

    if let Some(out_path) = output {
        let out = serde_json::json!({
            "model": model,
            "quant": format!("{}", quant_mode),
            "n_sequences": n,
            "failed": failed,
            "n_tokens": all_nll.len(),
            "ppl_overall": overall_ppl,
            "results": results,
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
        println!("  Saved → {}", out_path.display());
    }

    Ok(())
}

fn run_batch_hellaswag(
    model: &str,
    examples_file: &PathBuf,
    hot_budget_mb: usize,
    warm_budget_mb: usize,
    quant_mode: QuantMode,
    device: &str,
    output: Option<PathBuf>,
    hf_token: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let ex_json = std::fs::read_to_string(examples_file)?;
    let examples: Vec<serde_json::Value> = serde_json::from_str(&ex_json)?;
    let n = examples.len();

    println!("BatchHellaSwag: {} examples, model={}", n, model);

    let model_dir = resolve_model(model, hf_token)?;
    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;
    let mut paged = make_paged_model(&model_dir, hot_budget_mb, warm_budget_mb, quant_mode, device)?;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut failed = 0usize;
    let mut predictions = Vec::with_capacity(n);
    let t_start = Instant::now();

    for (i, ex) in examples.iter().enumerate() {
        let ctx = match ex["ctx"].as_str() {
            Some(s) => s.to_string(),
            None => { failed += 1; continue; }
        };
        let endings = match ex["endings"].as_array() {
            Some(arr) => arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect::<Vec<_>>(),
            None => { failed += 1; continue; }
        };
        let label = ex["label"].as_i64().unwrap_or(-1) as usize;

        if endings.len() < 2 {
            failed += 1;
            continue;
        }

        let ctx_tokens = tokenizer.encode_with_bos(&ctx);
        // boundary = number of context tokens including BOS, minus 1.
        // full_nll[boundary] = NLL(first_ending_token | full_context).
        let boundary = ctx_tokens.len().saturating_sub(1);

        let mut scores = Vec::new();
        let mut any_err = false;
        for ending in &endings {
            let full_text = format!("{}{}", ctx, ending);
            // Score the FULL sequence (context + ending) so the model sees the
            // complete context when computing ending log-probs.  Then slice
            // full_nll at [boundary..] to get only the ending token NLLs.
            let full_tokens = tokenizer.encode_with_bos(&full_text);
            match paged.score_sequence(&full_tokens) {
                Ok(full_nll) if full_nll.len() > boundary => {
                    let ending_nll = &full_nll[boundary..];
                    let mean_nll = ending_nll.iter().sum::<f64>() / ending_nll.len() as f64;
                    scores.push(-mean_nll);
                }
                Ok(_) => scores.push(f64::NEG_INFINITY),
                Err(_) => { any_err = true; scores.push(f64::NEG_INFINITY); }
            }
        }

        if any_err {
            failed += 1;
            continue;
        }

        let predicted = scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if predicted == label { correct += 1; }
        total += 1;
        predictions.push(serde_json::json!({
            "idx": i, "predicted": predicted, "label": label, "correct": predicted == label,
        }));

        if (i + 1) % 20 == 0 || i + 1 == n {
            let acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
            let elapsed = t_start.elapsed().as_secs_f64();
            println!("  [HellaSwag] {}/{} done, acc={:.3}, elapsed={:.1}s",
                     i + 1, n, acc, elapsed);
            io::stdout().flush().ok();
        }
    }

    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    println!("  [HellaSwag] Final: acc={:.3} ({}/{} correct, {} failed)", accuracy, correct, total, failed);

    if let Some(out_path) = output {
        let out = serde_json::json!({
            "model": model,
            "quant": format!("{}", quant_mode),
            "n_examples": n,
            "correct": correct,
            "total": total,
            "failed": failed,
            "accuracy": accuracy,
            "predictions": predictions,
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
        println!("  Saved → {}", out_path.display());
    }

    Ok(())
}
