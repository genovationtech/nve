#!/usr/bin/env bash
# NVE Comparison Test: Llama 3.2 1B vs 3B
# RAM budget capped at 500MB total (125 hot + 375 warm)
# Topic: General knowledge
set -euo pipefail

NVE="$(dirname "$0")/../target/release/nve"
HOT_MB=125
WARM_MB=375
MAX_TOKENS=80
TEMP=0.0

MODEL_1B="meta-llama/Llama-3.2-1B"
MODEL_3B="meta-llama/Llama-3.2-3B"

PROMPTS=(
  "The theory of general relativity explains that"
  "The three branches of the United States government are"
  "Photosynthesis is the process by which plants"
  "The French Revolution began in 1789 because"
  "Water boils at 100 degrees Celsius because"
)

OUTDIR="/tmp/nve_compare_1b_3b"
mkdir -p "$OUTDIR"

echo "================================================================="
echo "  NVE COMPARISON: Llama 3.2 1B vs 3B"
echo "  RAM cap: ${HOT_MB}MB hot + ${WARM_MB}MB warm = 500MB total"
echo "  Max tokens: $MAX_TOKENS | Temperature: $TEMP (greedy)"
echo "================================================================="
echo ""

run_model() {
  local label="$1"
  local model="$2"
  local outfile="$3"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  MODEL: $label"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  # --- Info ---
  echo "[info] Model details:"
  "$NVE" info -m "$model" 2>&1 || true
  echo ""

  # --- Benchmark (baseline vs paged) ---
  echo "[benchmark] Running NVE benchmark..."
  "$NVE" benchmark -m "$model" -n "$MAX_TOKENS" \
    --hot-budget-mb "$HOT_MB" --warm-budget-mb "$WARM_MB" \
    --output "$outfile" 2>&1
  echo ""

  # --- Generate on each general prompt (paged mode, 500MB cap) ---
  echo "[generate] General knowledge prompts (paged, 500MB cap):"
  echo ""
  for prompt in "${PROMPTS[@]}"; do
    echo "  Prompt: \"$prompt\""
    echo "  ---"
    "$NVE" generate -m "$model" \
      --paged \
      --hot-budget-mb "$HOT_MB" --warm-budget-mb "$WARM_MB" \
      -p "$prompt" -n "$MAX_TOKENS" -t "$TEMP" --top-p 1.0 2>&1 | \
      sed 's/^/  /'
    echo ""
    echo ""
  done
}

# Run 1B
run_model "Llama 3.2 1B" "$MODEL_1B" "$OUTDIR/bench_1b.json" 2>&1 | tee "$OUTDIR/log_1b.txt"

echo ""
echo ""

# Run 3B
run_model "Llama 3.2 3B" "$MODEL_3B" "$OUTDIR/bench_3b.json" 2>&1 | tee "$OUTDIR/log_3b.txt"

echo ""
echo "================================================================="
echo "  COMPARISON SUMMARY"
echo "================================================================="
echo ""
echo "Results saved to:"
echo "  $OUTDIR/bench_1b.json"
echo "  $OUTDIR/bench_3b.json"
echo "  $OUTDIR/log_1b.txt"
echo "  $OUTDIR/log_3b.txt"
echo ""

# Print side-by-side summary from JSON if jq available
if command -v jq &>/dev/null && [ -f "$OUTDIR/bench_1b.json" ] && [ -f "$OUTDIR/bench_3b.json" ]; then
  echo "  METRIC                          1B              3B"
  echo "  ─────────────────────────────── ─────────────── ───────────────"

  b1_tps=$(jq -r '.baseline.summary.avg_tokens_per_sec // "N/A"' "$OUTDIR/bench_1b.json")
  b3_tps=$(jq -r '.baseline.summary.avg_tokens_per_sec // "N/A"' "$OUTDIR/bench_3b.json")
  echo "  Baseline tok/s                  $b1_tps         $b3_tps"

  p1_tps=$(jq -r '.paged.summary.avg_tokens_per_sec // "N/A"' "$OUTDIR/bench_1b.json")
  p3_tps=$(jq -r '.paged.summary.avg_tokens_per_sec // "N/A"' "$OUTDIR/bench_3b.json")
  echo "  Paged tok/s (500MB cap)         $p1_tps         $p3_tps"

  b1_mem=$(jq -r '.baseline.summary.peak_memory_mb // "N/A"' "$OUTDIR/bench_1b.json")
  b3_mem=$(jq -r '.baseline.summary.peak_memory_mb // "N/A"' "$OUTDIR/bench_3b.json")
  echo "  Baseline peak RAM (MB)          $b1_mem         $b3_mem"

  p1_mem=$(jq -r '.paged.summary.peak_memory_mb // "N/A"' "$OUTDIR/bench_1b.json")
  p3_mem=$(jq -r '.paged.summary.peak_memory_mb // "N/A"' "$OUTDIR/bench_3b.json")
  echo "  Paged peak RAM (MB)             $p1_mem         $p3_mem"

  p1_fr=$(jq -r '.paged.paging_stats.fault_rate // "N/A"' "$OUTDIR/bench_1b.json")
  p3_fr=$(jq -r '.paged.paging_stats.fault_rate // "N/A"' "$OUTDIR/bench_3b.json")
  echo "  Page fault rate                 $p1_fr          $p3_fr"

  echo ""
fi

echo "Done."
