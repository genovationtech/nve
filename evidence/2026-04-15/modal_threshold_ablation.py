#!/usr/bin/env python3
"""
NVE Threshold Ablation — Section 4.7
=====================================
Measures WikiText-2 PPL across NVE_W4A8_THRESHOLD values to map the
quality-vs-coverage curve: how many layers need W4A16 protection, and
what is the quality cost of each choice?

For Llama-3.2-1B with MCAP normalized scores:
  - Layer 1 (outlier): normalized score = 1.0
  - All other 15 layers: normalized scores cluster 0.0–0.3

Thresholds tested (normalized [0,1]):
  all_w4a8   NVE_W4A8_THRESHOLD=2.0  →  0/16 W4A16
  t=0.30     threshold=0.30          →  layers with norm_score >= 0.30 → W4A16
  t=0.10     threshold=0.10          →  layers with norm_score >= 0.10 → W4A16
  t=0.05     threshold=0.05          →  layers with norm_score >= 0.05 → W4A16
  all_w4a16  NVE_NO_W4A8=1           →  16/16 W4A16

Previously measured:
  t=0.70 (MCAP Mixed)   → 1/16 W4A16,  PPL = 17.70
  all_w4a8              → 0/16 W4A16,  PPL = 17.71

Usage:
    modal run evidence/modal_threshold_ablation.py
"""

import os
import modal
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root

app       = modal.App("nve-threshold-ablation")
model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})

BUILD_CMD = (
    "bash -c 'set -o pipefail && "
    "cd /nve-src && "
    "/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=compute_75 -code=sm_75 "
    "-Xcompiler -fPIC -shared -o /usr/local/lib/libnve_kernels.so cuda/nve_kernels.cu 2>&1 && "
    "ldconfig && "
    "NVE_KERNELS_PREBUILT=/usr/local/lib CUDA_COMPUTE_CAP=75 CUDA_PATH=/usr/local/cuda "
    "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C link-arg=-L/usr/local/lib -C link-arg=-lnve_kernels\" "
    "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -20 && "
    "cp target/release/nve /usr/local/bin/nve-cuda && "
    "chmod +x /usr/local/bin/nve-cuda && "
    "echo \"Build OK: $(nve-cuda --version 2>&1 | head -1)\"'"
)

nve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install(
        "curl", "pkg-config", "libssl-dev", "ca-certificates",
        "build-essential", "cmake", "git",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "pip install huggingface_hub[hf_xfer] tqdm datasets",
    )
    .add_local_dir(
        str(HERE), "/nve-src", copy=True,
        ignore=["target/", ".git/", "docs/", "evidence/figures/", "evidence/figures_paper/", "reports/", "**/__pycache__/", "*.pyc"],
    )
    .run_commands(BUILD_CMD)
)

CALIBRATION_PROMPTS = [
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to compute Fibonacci numbers:",
    "What caused the French Revolution?",
    "Describe the process of photosynthesis:",
    "Solve the integral of x^2 from 0 to 1:",
    "What is the difference between a virus and a bacterium?",
    "Write a haiku about artificial intelligence:",
    "Explain the concept of supply and demand:",
    "What are the main themes in Shakespeare's Hamlet?",
    "Describe how a computer CPU works:",
    "What is the Pythagorean theorem and how is it used?",
    "Explain the water cycle in nature:",
]

# New thresholds to measure. We already have t=0.7 (Mixed=17.70) and t=2.0 (W4A8=17.71).
THRESHOLDS = [
    ("all_w4a8",  {"NVE_W4A8_THRESHOLD": "2.0"}),  # 0/16 W4A16 — baseline
    ("t=0.30",    {"NVE_W4A8_THRESHOLD": "0.30"}),
    ("t=0.10",    {"NVE_W4A8_THRESHOLD": "0.10"}),
    ("t=0.05",    {"NVE_W4A8_THRESHOLD": "0.05"}),
    ("all_w4a16", {"NVE_NO_W4A8": "1"}),            # 16/16 W4A16 — ceiling
]


def get_wikitext2_sequences(n: int = 20, seq_len: int = 256) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    words = text.split()
    seqs = []
    for start in range(0, len(words) - seq_len, seq_len):
        seqs.append(" ".join(words[start:start + seq_len]))
        if len(seqs) >= n:
            break
    return seqs


def run_ppl(model_dir: str, sequences: list[str], env_overrides: dict) -> dict:
    import subprocess, os, json, tempfile

    env = os.environ.copy()
    env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        texts_path = f.name
        json.dump(sequences, f)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    r = subprocess.run(
        ["nve-cuda", "batch-perplexity",
         "-m", model_dir,
         "--texts-file", texts_path,
         "--hot-budget-mb", "14000",
         "--warm-budget-mb", "14000",
         "--quantize", "q4",
         "--output", out_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env
    )
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:300]}")
        return {"ppl": float("nan"), "n_tokens": 0, "n_w4a16": -1}

    try:
        data = json.loads(Path(out_path).read_text())
        return {"ppl": data.get("ppl_overall", float("nan")),
                "n_tokens": data.get("n_tokens", 0)}
    except Exception as e:
        print(f"  parse error: {e}")
        return {"ppl": float("nan"), "n_tokens": 0}
    finally:
        for p in [texts_path, out_path]:
            try: os.unlink(p)
            except: pass


@app.function(
    image=nve_image,
    gpu="T4",
    cpu=8.0,
    memory=32768,
    timeout=7200,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_ablation() -> dict:
    import os, json, subprocess, glob as _glob

    model_dir = "/models/llama1b"
    # Weights should already be cached from earlier runs.
    from huggingface_hub import snapshot_download
    marker = Path(model_dir) / ".downloaded"
    if not marker.exists():
        print("[download] Llama-3.2-1B ...")
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-1B",
            local_dir=model_dir,
            token=os.environ["HF_TOKEN"],
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()

    # ── MCAP Profiling ────────────────────────────────────────────────────────
    cache_dir = Path("/root/.cache/nve/importance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in _glob.glob(str(cache_dir / "*.json")):
        os.remove(f)

    print("\n" + "="*60)
    print("MCAP Profiling — Llama-3.2-1B")
    print("="*60)
    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        r = subprocess.run(
            ["nve-cuda", "generate",
             "-m", model_dir, "-p", prompt, "-n", "5",
             "--temperature", "0", "--paged",
             "--hot-budget-mb", "14000", "--warm-budget-mb", "14000",
             "--quantize", "q4", "--profile"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if r.returncode != 0:
            print(f"  [profile] prompt {i+1} failed: {r.stderr[:100]}")
        else:
            print(f"  [profile] prompt {i+1}/{len(CALIBRATION_PROMPTS)} done")

    # Read cache and compute normalized scores for each threshold.
    normalized = []
    cache_files = list(cache_dir.glob("*.json"))
    print(f"  Cache: {len(cache_files)} file(s)")
    if cache_files:
        cache = json.loads(cache_files[0].read_text())
        scores = cache.get("scores", [])
        if scores:
            mn, mx = min(scores), max(scores)
            rng = mx - mn
            normalized = [(s - mn) / rng if rng > 1e-9 else 0.5 for s in scores]
            print(f"  Scores min={mn:.3f} max={mx:.3f} mean={sum(scores)/len(scores):.3f}")
            for t_label, t_env in THRESHOLDS:
                if "NVE_W4A8_THRESHOLD" in t_env:
                    thr = float(t_env["NVE_W4A8_THRESHOLD"])
                    n16 = sum(1 for s in normalized if s >= thr)
                    print(f"  [{t_label}] threshold={thr:.2f} → {n16}/{len(normalized)} W4A16 layers")

    # ── WikiText-2 Sequences ──────────────────────────────────────────────────
    seqs = get_wikitext2_sequences(n=20, seq_len=256)
    print(f"\n  {len(seqs)} sequences × 256 tokens")

    # ── PPL per threshold ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Threshold Ablation — WikiText-2 PPL")
    print("="*60)

    results = {}
    for label, env_ovr in THRESHOLDS:
        # Count W4A16 layers for this threshold.
        if "NVE_W4A8_THRESHOLD" in env_ovr and normalized:
            thr = float(env_ovr["NVE_W4A8_THRESHOLD"])
            n16 = sum(1 for s in normalized if s >= thr)
        elif "NVE_NO_W4A8" in env_ovr:
            n16 = len(normalized) if normalized else 16
        else:
            n16 = 0

        print(f"\n[{label}]  ({n16}/16 W4A16 layers)")
        r = run_ppl(model_dir, seqs, env_ovr)
        r["n_w4a16"] = n16
        results[label] = r
        print(f"  PPL = {r['ppl']:.2f}")

    return {"normalized_scores": normalized, "thresholds": results}


@app.local_entrypoint()
def main():
    import json
    print("NVE Threshold Ablation — Llama-3.2-1B")
    print("="*60)

    result = run_ablation.remote()
    thresholds = result.get("thresholds", {})

    # Add previously measured values.
    known = {
        "t=0.70 (MCAP Mixed)": {"ppl": 17.70, "n_w4a16": 1},
        "all_w4a8 (measured)": {"ppl": 17.71, "n_w4a16": 0},
        "w4a16 baseline":      {"ppl": 17.70, "n_w4a16": 16},
    }

    print("\n" + "="*60)
    print("ABLATION RESULTS — Llama-3.2-1B (+ previously measured)")
    print("="*60)
    print(f"  {'Config':<22} {'W4A16 layers':>14} {'PPL':>8}")
    print("  " + "-"*46)

    all_rows = list(thresholds.items()) + list(known.items())
    for label, r in all_rows:
        ppl = r.get("ppl", float("nan"))
        n16 = r.get("n_w4a16", "?")
        n16_str = f"{n16}/16" if isinstance(n16, int) else str(n16)
        print(f"  {label:<22} {n16_str:>14} {ppl:>8.2f}")

    out_path = Path(str(HERE)) / "evidence" / "experiments" / "threshold_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")
