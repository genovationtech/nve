//! Colored terminal output utilities.

use colored::Colorize;

use crate::model::GenerationResult;
use crate::paged_model::PagingStats;

// ─── Internal helpers ────────────────────────────────────────────────────────

fn make_bar(used: usize, total: usize, width: usize) -> String {
    let ratio = if total > 0 {
        (used as f64 / total as f64).min(1.0)
    } else {
        0.0
    };
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn mb_label(mb: usize) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{} MB", mb)
    }
}

// ─── Public display functions ─────────────────────────────────────────────────

/// Print a two-tier memory bar.
///
/// ```text
/// HOT  [████████░░]  512/512 MB  │  WARM  [██░░░░░░░░]  1.2/2.0 GB
/// ```
pub fn print_memory_bar(
    hot_used_mb: usize,
    hot_total_mb: usize,
    warm_used_mb: usize,
    warm_total_mb: usize,
) {
    let hot_bar = make_bar(hot_used_mb, hot_total_mb, 10);
    let warm_bar = make_bar(warm_used_mb, warm_total_mb, 10);

    let hot_label = format!(
        "HOT  [{}]  {}/{}",
        hot_bar,
        mb_label(hot_used_mb),
        mb_label(hot_total_mb)
    );
    let warm_label = format!(
        "WARM  [{}]  {}/{}",
        warm_bar,
        mb_label(warm_used_mb),
        mb_label(warm_total_mb)
    );

    println!("  {}  │  {}", hot_label.bright_red(), warm_label.bright_yellow());
}

/// Print a layer tier breakdown.
pub fn print_tier_assignment(num_hot: usize, num_warm: usize, num_cold: usize, total: usize) {
    println!("  Layer tiers ({} total):", total);
    println!("    {}  {} layers", "HOT ".bright_red(), num_hot);
    println!("    {}  {} layers", "WARM".bright_yellow(), num_warm);
    println!("    {}  {} layers", "COLD".bright_blue(), num_cold);
}

/// Print generation statistics with color.
pub fn print_gen_stats(result: &GenerationResult) {
    println!("  {}", "Generation stats:".cyan());
    println!(
        "    Prompt tokens:    {}",
        result.prompt_tokens.to_string().cyan()
    );
    println!(
        "    Generated tokens: {}",
        result.tokens.len().to_string().cyan()
    );
    println!(
        "    Prefill time:     {:.1} ms",
        result.prefill_time_ms
    );
    println!(
        "    Decode time:      {:.1} ms",
        result.decode_time_ms
    );
    println!(
        "    Total time:       {:.1} ms",
        result.total_time_ms
    );
    println!(
        "    Decode speed:     {}",
        format!("{:.1} tok/s", result.tokens_per_sec).cyan()
    );
}

/// Print paging statistics.
pub fn print_paging_stats(stats: &PagingStats) {
    let hit_rate = if stats.page_hits + stats.page_faults > 0 {
        stats.page_hits as f64 / (stats.page_hits + stats.page_faults) as f64 * 100.0
    } else {
        100.0
    };
    println!("  {}", "Paging stats:".cyan());
    println!("    Hit rate:      {:.1}%", hit_rate);
    println!("    Page faults:   {}", stats.page_faults);
    println!("    Layers loaded: {}", stats.layers_loaded);
    println!(
        "    Load time:     {:.1} ms",
        stats.total_load_time_ms
    );
    println!("    Prefetch hits: {}", stats.prefetch_hits);
}

/// Print a styled section header.
pub fn print_section(title: &str) {
    println!("\n  {}", format!("── {} ──", title).cyan().bold());
}
