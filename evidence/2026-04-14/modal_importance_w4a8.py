#!/usr/bin/env python3
"""
NVE Importance-Guided Mixed-Precision Benchmark
================================================
Tests quality and throughput of three precision strategies on Llama 1B / 3B:

  1. Uniform W4A8   — all layers use dp4a INT8 activations (NVE_W4A8_THRESHOLD=1.0)
  2. Uniform W4A16  — all layers use F16 activations      (NVE_NO_W4A8=1)
  3. Mixed W4A8/16  — profile layer importance first, then:
                       high-importance layers → W4A16
                       low-importance  layers → W4A8
                       (NVE_W4A8_THRESHOLD=0.7, default)

The profile pass runs `nve-cuda generate --profile` on 8 calibration prompts,
which populates ~/.cache/nve/importance/<model_key>.json.  Subsequent generate
calls load the cache and apply per-layer precision automatically.

Usage:
    modal run evidence/modal_importance_w4a8.py
    modal run evidence/modal_importance_w4a8.py --model llama1b
    modal run evidence/modal_importance_w4a8.py --model llama1b,llama3b
"""

import os
import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app       = modal.App("nve-importance-w4a8")
model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})

MODELS = {
    "llama1b": {
        "hf_id":    "meta-llama/Llama-3.2-1B",
        "local_dir": "/models/llama1b",
        "label":    "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":    "meta-llama/Llama-3.2-3B",
        "local_dir": "/models/llama3b",
        "label":    "Llama-3.2-3B",
    },
}

TASK_SUITE = [
    {"prompt": "The capital of France is",                                              "expected": "paris"},
    {"prompt": "Water is composed of hydrogen and",                                     "expected": "oxygen"},
    {"prompt": "The largest planet in the solar system is",                             "expected": "jupiter"},
    {"prompt": "If today is Monday, tomorrow is",                                       "expected": "tuesday"},
    {"prompt": "A square has four equal sides. A shape with four equal sides and four right angles is a", "expected": "square"},
    {"prompt": "def add(a, b):\n    return a",                                          "expected": "+"},
    {"prompt": "# Python: list of squares 0-4\nsquares = [x**2 for x in",             "expected": "range"},
    {"prompt": "The main benefit of regular exercise is improved",                      "expected": "health"},
]

# Calibration prompts for the profiling pass (different from task suite).
CALIBRATION_PROMPTS = [
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to compute Fibonacci numbers:",
    "The French Revolution began in",
    "In machine learning, gradient descent is",
    "The capital of Japan is",
    "def binary_search(arr, target):",
    "Water molecules consist of",
    "The speed of light in vacuum is approximately",
    "In 1969, humans first landed on",
    "A transformer neural network uses attention to",
    "The mitochondria is the powerhouse of",
    "SELECT * FROM users WHERE",
]

BUILD_CMD = (
    "bash -c 'set -o pipefail && "
    "cd /nve-src && "
    "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
    "-arch=compute_75 -code=sm_75 -Xcompiler -fPIC "
    "-shared -o /usr/local/lib/libnve_kernels.so cuda/nve_kernels.cu 2>&1 && "
    "ldconfig && "
    "NVE_KERNELS_PREBUILT=/usr/local/lib "
    "CUDA_COMPUTE_CAP=75 "
    "CUDA_PATH=/usr/local/cuda "
    "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C link-arg=-L/usr/local/lib -C link-arg=-lnve_kernels\" "
    "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -20 && "
    "cp target/release/nve /usr/local/bin/nve-cuda && "
    "chmod +x /usr/local/bin/nve-cuda && "
    "echo \"Build OK: $(nve-cuda --version 2>&1 | head -1)\"'"
)

nve_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "pkg-config", "libssl-dev", "ca-certificates", "build-essential", "cmake", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "pip install huggingface_hub[hf_xfer] tqdm",
    )
    .add_local_dir(str(HERE), "/nve-src", copy=True,
                   ignore=["target/", ".git/", "evidence/figures/", "evidence/figures_paper/", "reports/"])
    .run_commands(BUILD_CMD)
)


@app.function(
    image=nve_image,
    gpu="T4",
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_importance_bench(model_name: str) -> dict:
    import subprocess, time, os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    cfg = MODELS[model_name]

    # Download weights if not cached.
    marker = Path(cfg["local_dir"]) / ".downloaded"
    if not marker.exists():
        print(f"[download] {cfg['hf_id']} ...")
        snapshot_download(
            repo_id=cfg["hf_id"],
            local_dir=cfg["local_dir"],
            token=os.environ["HF_TOKEN"],
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()

    cache_dir = Path("/root/.cache/nve/importance")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def base_cmd(prompt, n=40):
        return [
            "nve-cuda", "generate",
            "-m", cfg["local_dir"],
            "-p", prompt,
            "-n", str(n),
            "--temperature", "0",
            "--paged",
            "--hot-budget-mb", "14000",
            "--warm-budget-mb", "14000",
            "--quantize", "q4",
        ]

    def run_task(task, label, env):
        cmd = base_cmd(task["prompt"])
        t0 = time.time()
        r  = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        dt = time.time() - t0
        combined = r.stdout + r.stderr
        ok  = task["expected"].lower() in combined.lower()
        tps = 40.0 / dt if dt > 0 else 0.0
        if r.returncode != 0:
            print(f"  [{label}] rc={r.returncode} stderr={r.stderr[:200]!r}")
        return {"passed": ok, "tps": tps, "rc": r.returncode}

    def run_suite(label, env_overrides=None):
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        results = [run_task(t, label, env) for t in TASK_SUITE]
        for res, task in zip(results, TASK_SUITE):
            marker = "PASS" if res["passed"] else "FAIL"
            print(f"  [{label}] [{marker}] {task['prompt'][:40]!r}")
        acc = sum(r["passed"] for r in results) / len(results)
        print(f"  [{label}] acc={acc:.0%}")
        return {"accuracy": acc, "tasks": results}

    # ── Step 1: Profile layer importance ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Profiling layer importance — {cfg['label']}")
    print('='*60)

    # Invalidate any stale cache by removing it first.
    import glob as _glob
    for f in _glob.glob("/root/.cache/nve/importance/*.json"):
        os.remove(f)

    # Run calibration prompts to populate importance cache.
    # --profile without --hot-only runs profiling and saves cache, then generates normally.
    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        cmd = base_cmd(prompt, n=5) + ["--profile"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"  [profile] prompt {i} failed: {r.stderr[:200]}")
        else:
            print(f"  [profile] prompt {i+1}/{len(CALIBRATION_PROMPTS)} done")

    # Check cache was written.
    cache_files = list(cache_dir.glob("*.json"))
    print(f"  Importance cache: {len(cache_files)} file(s) written")
    if cache_files:
        import json
        cache = json.loads(cache_files[0].read_text())
        scores = cache.get("scores", [])
        if scores:
            mn, mx = min(scores), max(scores)
            print(f"  Scores (raw): min={mn:.3f}  max={mx:.3f}  mean={sum(scores)/len(scores):.3f}")
            # Normalize to [0,1] — same as Rust does before threshold comparison.
            norm = [(s - mn) / (mx - mn) if mx > mn else 0.5 for s in scores]
            threshold = 0.7
            n_w4a16 = sum(1 for s in norm if s >= threshold)
            print(f"  Layers → W4A16 (normalized importance >= {threshold}): {n_w4a16}/{len(scores)}")
            print(f"  Layers → W4A8  (normalized importance <  {threshold}): {len(scores)-n_w4a16}/{len(scores)}")

    # ── Step 2: Quality comparison ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Quality tests — {cfg['label']}")
    print('='*60)

    # Uniform W4A8 (threshold=1.0 → all layers use W4A8).
    print("\n[Uniform W4A8]")
    r_w4a8 = run_suite("W4A8 uniform", {"NVE_W4A8_THRESHOLD": "1.0"})

    # Uniform W4A16.
    print("\n[Uniform W4A16]")
    r_w4a16 = run_suite("W4A16 uniform", {"NVE_NO_W4A8": "1"})

    # Mixed precision: importance cache on disk, nve loads it automatically.
    # High-importance layers (score >= NVE_W4A8_THRESHOLD=0.7) use W4A16.
    # Low-importance layers use W4A8. NVE_W4A8_THRESHOLD unset → default 0.7.
    print("\n[Mixed W4A8/16 (importance-guided, threshold=0.7)]")
    r_mixed = run_suite("Mixed W4A8/16")  # no env overrides: loads cache, uses default threshold

    return {
        "model":   model_name,
        "label":   cfg["label"],
        "w4a8":    r_w4a8,
        "w4a16":   r_w4a16,
        "mixed":   r_mixed,
        "cache":   {"files": len(cache_files), "scores": scores if cache_files else []},
    }


@app.local_entrypoint()
def main(model: str = "llama1b,llama3b"):
    import json
    from pathlib import Path

    model_names = [m.strip() for m in model.split(",")]
    print(f"NVE Importance-Guided Mixed-Precision — {model_names}")
    print("="*70)

    handles = {m: run_importance_bench.spawn(m) for m in model_names}
    results = {m: h.get() for m, h in handles.items()}

    print("\n" + "="*70)
    print(f"{'Model':<18} {'Strategy':<22} {'Accuracy':>9}")
    print("─"*55)
    for m in model_names:
        r = results[m]
        print(f"  {r['label']:<16} {'W4A8 (uniform)':<22} {r['w4a8']['accuracy']:>8.0%}")
        print(f"  {'':16} {'W4A16 (uniform)':<22} {r['w4a16']['accuracy']:>8.0%}")
        print(f"  {'':16} {'Mixed W4A8/16':<22} {r['mixed']['accuracy']:>8.0%}")

        if r["cache"]["scores"]:
            sc = r["cache"]["scores"]
            mn, mx = min(sc), max(sc)
            norm = [(s - mn) / (mx - mn) if mx > mn else 0.5 for s in sc]
            threshold = 0.7
            n16 = sum(1 for s in norm if s >= threshold)
            print(f"  {'':16} → {n16}/{len(sc)} layers W4A16, {len(sc)-n16}/{len(sc)} W4A8")
        print()

    out_dir = Path(str(HERE)) / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "importance_w4a8.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out_path}")
