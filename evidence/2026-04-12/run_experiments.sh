#!/usr/bin/env bash
# NVE Research Evidence Collection — Experiment Runner
# Models: Llama-3.2-1B (1.2B), Llama-3.2-3B (3.2B), Qwen2.5-0.5B (0.5B)
# Runs: Full ABC on each model + bpw sweep + layer sweep on 1B
# Output: evidence/experiments/*.json

set -euo pipefail
cd /mnt/ex1/apps/general-agent/nve

NVE=./target/release/nve
OUTDIR=./evidence/experiments

LLAMA1B=/home/ai/.cache/nve/models/meta-llama--Llama-3.2-1B
LLAMA3B=/home/ai/.cache/nve/models/meta-llama--Llama-3.2-3B
QWEN=.hf_cache/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987

echo "=== NVE Evidence Collection ==="
echo "Date: $(date)"
echo "Available RAM: $(free -m | awk '/Mem:/{print $7}') MB"
echo "Models:"
echo "  Llama-3.2-1B : $LLAMA1B"
echo "  Llama-3.2-3B : $LLAMA3B"
echo "  Qwen2.5-0.5B : $QWEN"
echo ""

# ── 1. Llama-3.2-1B full ABC ───────────────────────────────────────────────
# 1B layer = ~68 MB. With 500 MB hot + 1000 MB warm → ~22 layers fit → all 16 active
echo "[1/5] Llama-3.2-1B full ABC benchmark (all 4 configs)..."
$NVE abc-test \
  -m "$LLAMA1B" \
  --hot-budget-mb 500 \
  --warm-budget-mb 1000 \
  --target-bpw 2.0 \
  -n 40 \
  -o "$OUTDIR/llama1b_abc_full.json" \
  2>&1 | tee "$OUTDIR/llama1b_abc_full.log"
echo "  -> Saved llama1b_abc_full.json"

# ── 2. Llama-3.2-3B B+C only (3B won't fit in RAM for baseline/A) ─────────
# 3B layer = ~192 MB. 700 hot + 900 warm = 1600 MB → ~8 layers fit
echo ""
echo "[2/5] Llama-3.2-3B B+C configs (budget: 700 MB hot + 900 MB warm)..."
$NVE abc-test \
  -m "$LLAMA3B" \
  --hot-budget-mb 700 \
  --warm-budget-mb 900 \
  --target-bpw 2.0 \
  --configs b,c \
  -n 40 \
  -o "$OUTDIR/llama3b_abc_full.json" \
  2>&1 | tee "$OUTDIR/llama3b_abc_full.log"
echo "  -> Saved llama3b_abc_full.json"

# ── 3. Qwen2.5-0.5B full ABC ──────────────────────────────────────────────
# 0.5B, 24 layers, ~34 MB/layer → all 24 fit easily in 500 hot + 600 warm
echo ""
echo "[3/5] Qwen2.5-0.5B full ABC benchmark (all 4 configs)..."
$NVE abc-test \
  -m "$QWEN" \
  --hot-budget-mb 500 \
  --warm-budget-mb 600 \
  --target-bpw 2.0 \
  -n 40 \
  -o "$OUTDIR/qwen_abc_full.json" \
  2>&1 | tee "$OUTDIR/qwen_abc_full.log"
echo "  -> Saved qwen_abc_full.json"

# ── 4. Llama-1B bits-per-weight sweep (Config C only) ─────────────────────
# Vary bpw to map the quality-compression tradeoff curve
echo ""
echo "[4/5] Llama-1B bpw sweep (0.5 → 4.0 bpw, Config C)..."
for BPW in 0.5 1.0 1.5 2.0 3.0 4.0; do
  OUTFILE="$OUTDIR/llama1b_bpw_${BPW//./_}.json"
  echo "  bpw=$BPW ..."
  $NVE abc-test \
    -m "$LLAMA1B" \
    --hot-budget-mb 500 \
    --warm-budget-mb 1000 \
    --target-bpw "$BPW" \
    --configs c \
    -n 40 \
    -o "$OUTFILE" \
    2>&1 | tee "${OUTFILE%.json}.log"
  echo "    -> $OUTFILE"
done

# ── 5. Llama-1B active-layer sweep (Config B only) ─────────────────────────
# Vary active layers to find the quality cliff
echo ""
echo "[5/5] Llama-1B active-layer sweep (2 → 16 layers, Config B)..."
for N in 2 4 6 8 10 12 14 16; do
  OUTFILE="$OUTDIR/llama1b_layers_${N}.json"
  echo "  active_layers=$N ..."
  $NVE abc-test \
    -m "$LLAMA1B" \
    --hot-budget-mb 500 \
    --warm-budget-mb 1000 \
    --active-layers "$N" \
    --configs b \
    -n 40 \
    -o "$OUTFILE" \
    2>&1 | tee "${OUTFILE%.json}.log"
  echo "    -> $OUTFILE"
done

echo ""
echo "=== All experiments complete. ==="
echo "Regenerate figures: cd /mnt/ex1/apps/general-agent/nve && python3 evidence/visualize.py"
