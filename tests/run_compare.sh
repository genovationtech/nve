#!/usr/bin/env bash
# NVE Paged Comparison: Llama 3.2 1B vs 3B — 500MB RAM cap
# Runs paged inference only (no baseline) to stay within memory limits
set -uo pipefail

NVE="$(cd "$(dirname "$0")/.." && pwd)/target/release/nve"
HOT_MB=125
WARM_MB=375
MAX_TOKENS=80
TEMP=0.0
OUTDIR="/tmp/nve_compare_1b_3b"
mkdir -p "$OUTDIR"

PROMPTS=(
  "The theory of general relativity explains that"
  "The three branches of the United States government are"
  "Photosynthesis is the process by which plants"
  "The French Revolution began in 1789 because"
  "Water boils at 100 degrees Celsius because"
)

run_paged_test() {
  local label="$1"
  local model="$2"
  local logfile="$3"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $label — PAGED MODE (${HOT_MB}+${WARM_MB}=500MB cap)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    echo "── Prompt $((i+1))/${#PROMPTS[@]}: \"$prompt\""
    echo ""

    "$NVE" generate -m "$model" --paged \
      --hot-budget-mb "$HOT_MB" --warm-budget-mb "$WARM_MB" \
      -p "$prompt" -n "$MAX_TOKENS" -t "$TEMP" --top-p 1.0 2>&1

    echo ""
    # Check RSS after generation
    rss_kb=$(awk '/VmRSS/{print $2}' /proc/self/status 2>/dev/null || echo "?")
    echo "  [RSS after generate: ${rss_kb} KB]"
    echo ""
  done
}

echo "================================================================="
echo "  NVE COMPARISON: Llama 3.2 1B vs 3B"
echo "  RAM cap: ${HOT_MB}MB hot + ${WARM_MB}MB warm = 500MB total"
echo "  Max tokens: $MAX_TOKENS | Greedy decoding (temp=0)"
echo "  Prompts: ${#PROMPTS[@]} general knowledge"
echo "================================================================="

# Test 1: Llama 1B
run_paged_test "LLAMA 3.2 1B" "meta-llama/Llama-3.2-1B" "$OUTDIR/log_1b.txt"

echo ""
echo ""

# Test 2: Llama 3B
run_paged_test "LLAMA 3.2 3B" "meta-llama/Llama-3.2-3B" "$OUTDIR/log_3b.txt"

echo ""
echo "================================================================="
echo "  TESTS COMPLETE"
echo "================================================================="
