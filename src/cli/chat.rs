//! Interactive chat REPL for NVE.

use std::io::{self, Write};
use std::path::PathBuf;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::generic_model::GenericModel;
use crate::model::GenerationResult;
use crate::paged_model::{PagedConfig, PagedModel, PagingStats};
use crate::quantize::QuantMode;
use crate::tokenizer::Tokenizer;

use super::banner::{print_banner, print_model_header};
use super::config::NveConfig;
use super::display::{print_gen_stats, print_memory_bar, print_paging_stats, print_section, print_tier_assignment};

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatMode {
    Baseline,
    Paged,
    HotOnly,
    ProfiledHotOnly,
}

impl std::fmt::Display for ChatMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatMode::Baseline => write!(f, "baseline"),
            ChatMode::Paged => write!(f, "paged"),
            ChatMode::HotOnly => write!(f, "hot-only"),
            ChatMode::ProfiledHotOnly => write!(f, "profiled-hot-only"),
        }
    }
}

enum LoadedModel {
    Generic(GenericModel),
    Paged(PagedModel),
}

impl LoadedModel {
    fn generate(
        &mut self,
        tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        match self {
            LoadedModel::Generic(m) => Ok(m.generate(tokens, max_tokens, temperature, top_p)),
            LoadedModel::Paged(m) => Ok(m.generate(tokens, max_tokens, temperature, top_p)?),
        }
    }

    fn paging_stats(&self) -> Option<&PagingStats> {
        match self {
            LoadedModel::Paged(m) => Some(m.stats()),
            _ => None,
        }
    }

    fn memory_report(&self) -> Option<String> {
        match self {
            LoadedModel::Paged(m) => Some(m.memory_report()),
            _ => None,
        }
    }

    fn layer_counts(&self) -> Option<(usize, usize, usize, usize)> {
        match self {
            LoadedModel::Paged(m) => {
                use crate::paged_model::LayerTier;
                let tiers = m.layer_tiers();
                let hot = tiers.iter().filter(|t| matches!(t, LayerTier::Hot)).count();
                let warm = tiers.iter().filter(|t| matches!(t, LayerTier::Warm)).count();
                let cold = tiers.iter().filter(|t| matches!(t, LayerTier::Cold)).count();
                Some((hot, warm, cold, tiers.len()))
            }
            _ => None,
        }
    }
}

/// Result returned by the memory configuration wizard.
struct WizardResult {
    hot_mb: usize,
    warm_mb: usize,
    quant_mode: QuantMode,
    hot_only: bool,
    profile: bool,
    active_layers: Option<usize>,
}

// ─── Memory wizard ────────────────────────────────────────────────────────────

fn read_line_default(prompt: &str, default: &str) -> String {
    print!("{} [{}]: ", prompt, default);
    io::stdout().flush().unwrap_or(());
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap_or(0);
    let trimmed = buf.trim().to_string();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed
    }
}

fn read_choice(prompt: &str, valid: &[&str]) -> String {
    loop {
        print!("{}: ", prompt);
        io::stdout().flush().unwrap_or(());
        let mut buf = String::new();
        io::stdin().read_line(&mut buf).unwrap_or(0);
        let choice = buf.trim().to_string();
        if valid.contains(&choice.as_str()) {
            return choice;
        }
        println!("  Please enter one of: {}", valid.join(", "));
    }
}

fn available_memory_mb() -> usize {
    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
        for line in contents.lines() {
            if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<usize>() {
                        return kb / 1024;
                    }
                }
                break;
            }
        }
    }
    4096 // fallback
}

fn memory_wizard(model_name: &str) -> WizardResult {
    let available_mb = available_memory_mb();
    let available_gb = available_mb as f64 / 1024.0;

    println!();
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│  {}  │", format!("Memory Configuration for: {:37}", model_name).bold());
    println!("│  {}  │", format!("System RAM Available: {:.1} GB{:19}", available_gb, "").cyan());
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│  Inference Mode:                                            │");
    println!("│    [1] auto      — 80% RAM split 25% hot / 75% warm        │");
    println!("│    [2] hot-only  — profile then keep top N layers in RAM   │");
    println!("│    [3] paged     — hot + warm + cold (disk fallback)        │");
    println!("│    [4] custom    — set budgets manually                     │");
    println!("└─────────────────────────────────────────────────────────────┘");

    let choice = read_choice("Select [1-4]", &["1", "2", "3", "4"]);

    match choice.as_str() {
        "1" => {
            // Auto: 80% of available, 25% hot / 75% warm.
            let budget_mb = (available_mb as f64 * 0.8) as usize;
            let hot_mb = budget_mb / 4;
            let warm_mb = budget_mb - hot_mb;
            println!(
                "  Auto budget → {} hot, {} warm",
                format!("{} MB", hot_mb).bright_red(),
                format!("{} MB", warm_mb).bright_yellow()
            );
            WizardResult {
                hot_mb,
                warm_mb,
                quant_mode: QuantMode::None,
                hot_only: false,
                profile: false,
                active_layers: None,
            }
        }
        "2" => {
            // Hot-only with profiling.
            let budget_mb = (available_mb as f64 * 0.8) as usize;
            let hot_mb = budget_mb / 4;
            let warm_mb = budget_mb - hot_mb;
            WizardResult {
                hot_mb,
                warm_mb,
                quant_mode: QuantMode::None,
                hot_only: true,
                profile: true,
                active_layers: None,
            }
        }
        "3" => {
            // Paged.
            let budget_mb = (available_mb as f64 * 0.8) as usize;
            let hot_mb = budget_mb / 4;
            let warm_mb = budget_mb - hot_mb;
            println!(
                "  Paged mode → {} hot, {} warm",
                format!("{} MB", hot_mb).bright_red(),
                format!("{} MB", warm_mb).bright_yellow()
            );
            WizardResult {
                hot_mb,
                warm_mb,
                quant_mode: QuantMode::None,
                hot_only: false,
                profile: false,
                active_layers: None,
            }
        }
        _ => {
            // Custom.
            let hot_str = read_line_default("HOT  tier budget  (always-resident RAM, MB)", "512");
            let warm_str = read_line_default("WARM tier budget  (LRU-cached RAM, MB)", "2048");
            let quant_str =
                read_line_default("Quantization [none/q4/q8/q3/q2/q1]", "none");
            let profile_str = read_line_default("Profile layer importance?", "Y");
            let hot_only_str = read_line_default("Hot-only mode (skip cold layers)?", "N");
            let active_str = read_line_default("Active layers  (blank = auto from budget)", "");

            let hot_mb = hot_str.parse::<usize>().unwrap_or(512);
            let warm_mb = warm_str.parse::<usize>().unwrap_or(2048);
            let quant_mode = QuantMode::from_str(&quant_str).unwrap_or(QuantMode::None);
            let profile = !profile_str.to_lowercase().starts_with('n');
            let hot_only = hot_only_str.to_lowercase().starts_with('y');
            let active_layers = active_str.parse::<usize>().ok();

            WizardResult {
                hot_mb,
                warm_mb,
                quant_mode,
                hot_only,
                profile,
                active_layers,
            }
        }
    }
}

// ─── Model loading ────────────────────────────────────────────────────────────

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn load_model(
    model_dir: &PathBuf,
    mode: ChatMode,
    hot_mb: usize,
    warm_mb: usize,
    quant_mode: QuantMode,
    active_layers: Option<usize>,
    profile: bool,
    tokenizer: &Tokenizer,
) -> Result<LoadedModel, Box<dyn std::error::Error>> {
    match mode {
        ChatMode::Baseline => {
            let pb = spinner("Loading model weights…");
            let model = GenericModel::from_dir(model_dir)?;
            pb.finish_and_clear();
            println!("  {} Model loaded (baseline).", "✓".bold().green());
            Ok(LoadedModel::Generic(model))
        }
        ChatMode::Paged | ChatMode::HotOnly | ChatMode::ProfiledHotOnly => {
            let paged_config = PagedConfig {
                hot_budget_bytes: hot_mb * 1024 * 1024,
                warm_budget_bytes: warm_mb * 1024 * 1024,
                prefetch_ahead: 2,
                profile_activations: false,
                hot_only_mode: mode == ChatMode::HotOnly,
                active_layers: if profile { None } else { active_layers },
                quant_mode,
                device: "auto".to_string(),
            };

            let pb = spinner("Loading paged model…");
            let mut paged = PagedModel::from_dir(model_dir, paged_config)?;
            pb.finish_and_clear();

            if profile {
                let num_layers = paged.num_layers();
                let layer_size = paged.layer_size_bytes();
                let total_budget = (hot_mb + warm_mb) * 1024 * 1024;
                let active_count = active_layers.unwrap_or_else(|| {
                    (total_budget / layer_size.max(1)).min(num_layers)
                });

                // Try loading from importance cache first.
                let cache_hit = crate::importance_cache::ImportanceCache::load(model_dir, num_layers);
                if let Some(scores) = cache_hit {
                    paged.apply_importance_scores(&scores, active_count);
                    println!(
                        "  {} Importance cache loaded — {} active layers (skipped profiling).",
                        "✓".bold().green(),
                        active_count
                    );
                } else {
                    let pb = spinner("Profiling layer importance…");
                    let probe_tokens = tokenizer.encode_with_bos("The quick brown fox");
                    let capped: Vec<u32> = probe_tokens.into_iter().take(32).collect();
                    paged.profile_layer_importance(&capped)?;
                    paged.apply_profiled_hot_only(active_count);
                    pb.finish_and_clear();
                    println!(
                        "  {} Profile done — {} active layers.",
                        "✓".bold().green(),
                        active_count
                    );
                }
            }

            // Initialize GPU inference for hot layers (no-op on CPU-only builds or CPU device).
            if let Err(e) = paged.init_gpu() {
                eprintln!("  [warn] GPU init failed, using CPU inference: {}", e);
            }

            println!(
                "  {} Model loaded ({}).",
                "✓".bold().green(),
                mode
            );
            Ok(LoadedModel::Paged(paged))
        }
    }
}

// ─── REPL helpers ─────────────────────────────────────────────────────────────

fn print_help() {
    println!("{}", "  NVE Chat Commands:".cyan().bold());
    let cmds = [
        ("/help",                      "List commands"),
        ("/model <id>",                "Switch model (triggers reload)"),
        ("/memory",                    "Print memory report"),
        ("/profile",                   "Re-run layer profiling"),
        ("/stats",                     "Show last generation stats"),
        ("/config",                    "Print active config"),
        ("/clear",                     "Clear conversation history"),
        ("/hot <mb>",                  "Change hot budget (triggers reload)"),
        ("/warm <mb>",                 "Change warm budget (triggers reload)"),
        ("/quant <mode>",              "Change quantization (triggers reload)"),
        ("/mode <baseline|paged|hot-only>", "Switch inference mode (triggers reload)"),
        ("/exit  /quit",               "Exit"),
    ];
    for (cmd, desc) in &cmds {
        println!("    {:40} {}", cmd.bright_cyan(), desc);
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn run_chat(
    model: &str,
    config: &NveConfig,
    hot_mb: Option<usize>,
    warm_mb: Option<usize>,
    auto_budget: bool,
    hot_only: bool,
    profile: bool,
    active_layers: Option<usize>,
    quantize: Option<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    print_banner();

    // ── Resolve model path ──────────────────────────────────────────────────
    let hf_token = config.hf_token_resolved();
    let model_dir = {
        let path = std::path::Path::new(model);
        if path.is_dir() && path.join("config.json").exists() {
            path.to_path_buf()
        } else {
            let pb = spinner(&format!("Resolving model '{}'…", model));
            let dir = crate::hub::resolve_model_path(model, None, hf_token.as_deref())?;
            pb.finish_and_clear();
            dir
        }
    };

    let model_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model)
        .to_string();

    let tokenizer = Tokenizer::from_model_dir(&model_dir)?;

    // ── Memory wizard or use provided flags ────────────────────────────────
    let (resolved_hot, resolved_warm, quant_mode, use_hot_only, use_profile, resolved_active) =
        if auto_budget {
            let avail = available_memory_mb();
            let budget = (avail as f64 * 0.8) as usize;
            let h = budget / 4;
            let w = budget - h;
            let qm = quantize
                .as_deref()
                .and_then(QuantMode::from_str)
                .unwrap_or(QuantMode::None);
            (h, w, qm, hot_only, profile, active_layers)
        } else if hot_mb.is_some() && warm_mb.is_some() {
            let qm = quantize
                .as_deref()
                .and_then(QuantMode::from_str)
                .unwrap_or(QuantMode::None);
            (
                hot_mb.unwrap(),
                warm_mb.unwrap(),
                qm,
                hot_only,
                profile,
                active_layers,
            )
        } else {
            let wiz = memory_wizard(&model_name);
            let qm = if wiz.quant_mode != QuantMode::None {
                wiz.quant_mode
            } else {
                quantize
                    .as_deref()
                    .and_then(QuantMode::from_str)
                    .unwrap_or(QuantMode::None)
            };
            (
                wiz.hot_mb,
                wiz.warm_mb,
                qm,
                wiz.hot_only || hot_only,
                wiz.profile || profile,
                wiz.active_layers.or(active_layers),
            )
        };

    // ── Determine chat mode ─────────────────────────────────────────────────
    let mut mode = if use_hot_only && use_profile {
        ChatMode::ProfiledHotOnly
    } else if use_hot_only {
        ChatMode::HotOnly
    } else if resolved_hot > 0 || resolved_warm > 0 {
        ChatMode::Paged
    } else {
        ChatMode::Baseline
    };

    let mut current_hot = resolved_hot;
    let mut current_warm = resolved_warm;
    let mut current_quant = quant_mode;
    let mut current_active = resolved_active;

    // ── Load model ──────────────────────────────────────────────────────────
    println!();
    let mut loaded = load_model(
        &model_dir,
        mode,
        current_hot,
        current_warm,
        current_quant,
        current_active,
        use_profile,
        &tokenizer,
    )?;

    print_model_header(&model_name, &current_quant.to_string(), &mode.to_string());
    println!();
    println!("  Type a message to chat. Use {} to see available commands.", "/help".cyan());
    println!();

    // ── Session state ───────────────────────────────────────────────────────
    let mut history: Vec<(String, String)> = Vec::new();
    let mut last_gen: Option<GenerationResult> = None;
    let mut current_max_tokens = max_tokens;
    let mut current_temperature = temperature;
    let mut current_top_p = top_p;
    let mut reload_needed = false;
    let mut current_model_id = model.to_string();
    let mut current_model_dir = model_dir;

    let mut rl = DefaultEditor::new()?;

    loop {
        // Reload model if settings changed.
        if reload_needed {
            println!();
            let pb = spinner("Reloading model with new settings…");
            let new_dir = {
                let path = std::path::Path::new(&current_model_id);
                if path.is_dir() && path.join("config.json").exists() {
                    path.to_path_buf()
                } else {
                    let token = config.hf_token_resolved();
                    crate::hub::resolve_model_path(&current_model_id, None, token.as_deref())?
                }
            };
            pb.finish_and_clear();
            current_model_dir = new_dir;
            let new_tokenizer = Tokenizer::from_model_dir(&current_model_dir)?;
            loaded = load_model(
                &current_model_dir,
                mode,
                current_hot,
                current_warm,
                current_quant,
                current_active,
                matches!(mode, ChatMode::ProfiledHotOnly),
                &new_tokenizer,
            )?;
            history.clear();
            reload_needed = false;
            print_model_header(&current_model_id, &current_quant.to_string(), &mode.to_string());
            println!();
        }

        let prompt_str = "nve> ".bright_cyan().bold().to_string();
        let line = match rl.readline(&prompt_str) {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("Input error: {}", e);
                break;
            }
        };

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(&input);

        // ── Slash commands ─────────────────────────────────────────────────
        if input.starts_with('/') {
            let parts: Vec<&str> = input.splitn(2, ' ').collect();
            let cmd = parts[0];
            let arg = parts.get(1).copied().unwrap_or("").trim();

            match cmd {
                "/exit" | "/quit" => break,

                "/help" => print_help(),

                "/clear" => {
                    history.clear();
                    println!("  Conversation history cleared.");
                }

                "/config" => {
                    print_section("Active Configuration");
                    println!("    Model:       {}", current_model_id.cyan());
                    println!("    Mode:        {}", mode.to_string().cyan());
                    println!("    Quantize:    {}", current_quant.to_string().cyan());
                    println!("    Hot budget:  {} MB", current_hot);
                    println!("    Warm budget: {} MB", current_warm);
                    println!("    Max tokens:  {}", current_max_tokens);
                    println!("    Temperature: {}", current_temperature);
                    println!("    Top-p:       {}", current_top_p);
                }

                "/memory" => {
                    print_section("Memory Report");
                    if let Some(report) = loaded.memory_report() {
                        println!("{}", report);
                    }
                    if let Some((hot, warm, cold, total)) = loaded.layer_counts() {
                        let hot_used = (hot * current_hot) / total.max(1);
                        print_memory_bar(hot_used, current_hot, 0, current_warm);
                        print_tier_assignment(hot, warm, cold, total);
                    } else {
                        println!("  (baseline mode — no paged memory stats)");
                    }
                }

                "/stats" => {
                    if let Some(ref gen) = last_gen {
                        print_section("Last Generation Stats");
                        print_gen_stats(gen);
                        if let Some(stats) = loaded.paging_stats() {
                            print_paging_stats(stats);
                        }
                    } else {
                        println!("  No generation stats yet.");
                    }
                }

                "/profile" => {
                    if let LoadedModel::Paged(ref mut paged) = loaded {
                        // Invalidate cache so fresh scores are stored.
                        crate::importance_cache::ImportanceCache::invalidate(&current_model_dir);
                        let pb = spinner("Profiling layer importance…");
                        let probe = tokenizer.encode_with_bos("The quick brown fox");
                        let capped: Vec<u32> = probe.into_iter().take(32).collect();
                        paged.profile_layer_importance(&capped)?;
                        let layer_size = paged.layer_size_bytes();
                        let budget = (current_hot + current_warm) * 1024 * 1024;
                        let count = current_active.unwrap_or_else(|| {
                            (budget / layer_size.max(1)).min(paged.num_layers())
                        });
                        paged.apply_profiled_hot_only(count);
                        let _ = paged.init_gpu(); // re-init GPU state with updated hot set
                        pb.finish_and_clear();
                        println!("  {} Profiling complete — {} active layers (cache updated).", "✓".bold().green(), count);
                    } else {
                        println!("  {} /profile only works in paged/hot-only modes.", "!".yellow());
                    }
                }

                "/model" => {
                    if arg.is_empty() {
                        println!("  Usage: /model <model-id-or-path>");
                    } else {
                        current_model_id = arg.to_string();
                        reload_needed = true;
                        println!("  Model set to '{}'. Reloading…", current_model_id);
                    }
                }

                "/hot" => {
                    if let Ok(mb) = arg.parse::<usize>() {
                        current_hot = mb;
                        reload_needed = true;
                        println!("  Hot budget set to {} MB. Reloading…", mb);
                    } else {
                        println!("  Usage: /hot <mb>");
                    }
                }

                "/warm" => {
                    if let Ok(mb) = arg.parse::<usize>() {
                        current_warm = mb;
                        reload_needed = true;
                        println!("  Warm budget set to {} MB. Reloading…", mb);
                    } else {
                        println!("  Usage: /warm <mb>");
                    }
                }

                "/quant" => {
                    if let Some(qm) = QuantMode::from_str(arg) {
                        current_quant = qm;
                        reload_needed = true;
                        println!("  Quantization set to '{}'. Reloading…", arg);
                    } else {
                        println!("  Unknown quant mode '{}'. Use: none, q4, q8, q3, q2, q1", arg);
                    }
                }

                "/mode" => {
                    let new_mode = match arg {
                        "baseline" => Some(ChatMode::Baseline),
                        "paged" => Some(ChatMode::Paged),
                        "hot-only" => Some(ChatMode::HotOnly),
                        "profiled-hot-only" => Some(ChatMode::ProfiledHotOnly),
                        _ => None,
                    };
                    if let Some(m) = new_mode {
                        mode = m;
                        reload_needed = true;
                        println!("  Mode set to '{}'. Reloading…", m);
                    } else {
                        println!("  Unknown mode '{}'. Use: baseline, paged, hot-only, profiled-hot-only", arg);
                    }
                }

                other => {
                    println!("  Unknown command '{}'. Type {} for help.", other, "/help".cyan());
                }
            }

            continue;
        }

        // ── Regular chat message ───────────────────────────────────────────
        // Build the full prompt from conversation history.
        let mut full_prompt = String::new();
        for (user_msg, assistant_msg) in &history {
            full_prompt.push_str(&format!("User: {}\nAssistant: {}\n", user_msg, assistant_msg));
        }
        full_prompt.push_str(&format!("User: {}\nAssistant:", &input));

        let tokens = tokenizer.encode_with_bos(&full_prompt);

        // Stream the assistant prefix.
        print!("{} ", "Assistant:".bold().green());
        io::stdout().flush()?;

        let gen = match loaded.generate(&tokens, current_max_tokens, current_temperature, current_top_p) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("{} {}", "Error:".bold().red(), e);
                continue;
            }
        };

        let response_text = tokenizer.decode(&gen.tokens);

        // Print token-by-token for the streaming feel.
        for word in response_text.split_inclusive(|c: char| c.is_whitespace()) {
            print!("{}", word);
            io::stdout().flush()?;
        }
        println!("\n");

        // Update history and last stats.
        history.push((input.clone(), response_text.trim().to_string()));
        last_gen = Some(gen);
    }

    println!("\n  Goodbye!");
    Ok(())
}
