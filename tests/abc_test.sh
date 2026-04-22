#!/usr/bin/env bash
# NVE ABC Test Matrix — Hot-Only Focus
#
# A = Hot-only evenly-spaced, NO quantization (bf16)
# B = Hot-only profiled, NO quantization (bf16)
# C = Hot-only profiled + uniform Q4
# D = Profile-guided pg:2.0 (profiler + AWQ + mixed bits)
# E = Profile-guided pg:1.0 (extreme compression)
#
# Tests on: Llama 3.2 1B (16L) and Llama 3.2 3B (28L)
# Budget: 1 GB | Greedy decoding | 30 tokens

set -uo pipefail

export HF_TOKEN="${HF_TOKEN:-${HF_TOKEN:?HF_TOKEN not set}}"
NVE="./target/release/nve"
HOT=250
WARM=774
TOKENS=30

MODELS=("meta-llama/Llama-3.2-1B" "meta-llama/Llama-3.2-3B")
MODEL_NAMES=("1B" "3B")

PROMPTS=(
  "The theory of general relativity explains that"
  "The three branches of the United States government are"
  "Photosynthesis is the process by which plants"
  "Shakespeare wrote Hamlet to explore themes of"
)

run_test() {
  local label="$1"
  shift
  echo "  [$label]"
  output=$($NVE "$@" 2>&1)
  # Extract generated text (the line that starts with the prompt)
  text=$(echo "$output" | grep -E "^The |^Photo|^Shake" | head -1)
  speed=$(echo "$output" | grep "Decode speed" | head -1 | sed 's/.*: //')
  layers=$(echo "$output" | grep -E "Profiled hot|Hot-only mode" | head -1 | sed 's/.*\] //')
  assignment=$(echo "$output" | grep "Layer  0:" | tail -1 | sed 's/.*Layer  0: /L0:/')

  if [ -n "$text" ]; then
    # Strip the prompt from output to show only generated part
    echo "    Output: ${text:${#prompt}:100}"
  else
    echo "    Output: (empty or non-standard)"
  fi
  echo "    Speed: $speed"
  if [ -n "$layers" ]; then
    echo "    Config: $layers"
  fi
}

echo "================================================================="
echo "  NVE ABC TEST MATRIX"
echo "  Budget: ${HOT}+${WARM}=1024 MB | Tokens: $TOKENS | Greedy"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================="

for m_idx in "${!MODELS[@]}"; do
  model="${MODELS[$m_idx]}"
  name="${MODEL_NAMES[$m_idx]}"

  echo ""
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║  MODEL: Llama 3.2 $name                                      ║"
  echo "╚═══════════════════════════════════════════════════════════════╝"

  for p_idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$p_idx]}"
    echo ""
    echo "── Prompt $((p_idx+1)): \"${prompt:0:55}\""

    # A: Hot-only evenly-spaced, bf16
    run_test "A: Evenly-spaced bf16" \
      generate -m "$model" --paged --hot-only \
      --hot-budget-mb $HOT --warm-budget-mb $WARM \
      -p "$prompt" -n $TOKENS -t 0.0 --top-p 1.0

    # B: Hot-only profiled, bf16
    run_test "B: Profiled bf16" \
      generate -m "$model" --paged --hot-only --profile \
      --hot-budget-mb $HOT --warm-budget-mb $WARM \
      -p "$prompt" -n $TOKENS -t 0.0 --top-p 1.0

    # C: Hot-only profiled + uniform Q4
    run_test "C: Profiled + Q4" \
      generate -m "$model" --paged --hot-only --profile \
      --quantize q4 \
      --hot-budget-mb $HOT --warm-budget-mb $WARM \
      -p "$prompt" -n $TOKENS -t 0.0 --top-p 1.0

    # D: Profile-guided pg:2.0
    run_test "D: PG 2.0 bpw" \
      generate -m "$model" --paged --hot-only --profile \
      --quantize pg:2.0 \
      --hot-budget-mb $HOT --warm-budget-mb $WARM \
      -p "$prompt" -n $TOKENS -t 0.0 --top-p 1.0

    # E: Profile-guided pg:1.0
    run_test "E: PG 1.0 bpw" \
      generate -m "$model" --paged --hot-only --profile \
      --quantize pg:1.0 \
      --hot-budget-mb $HOT --warm-budget-mb $WARM \
      -p "$prompt" -n $TOKENS -t 0.0 --top-p 1.0

    echo ""
  done
done

echo "================================================================="
echo "  ABC TEST COMPLETE"
echo "================================================================="
