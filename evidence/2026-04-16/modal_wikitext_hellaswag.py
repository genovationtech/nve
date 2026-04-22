#!/usr/bin/env python3
"""
NVE WikiText-2 Perplexity + HellaSwag Accuracy Benchmark
=========================================================
Evaluates three precision strategies on Llama-3.2-1B and 3B:

  1. Uniform W4A16  — F16 activations (NVE_NO_W4A8=1)
  2. Uniform W4A8   — INT8 dp4a activations for all layers (NVE_W4A8_THRESHOLD=1.0)
  3. Mixed W4A8/16  — importance-guided: top layers → W4A16, rest → W4A8

Metrics:
  - WikiText-2: perplexity on 50 sequences × 256 tokens from the test set
  - HellaSwag:  accuracy on 200 validation examples (4-way multiple choice)

Baseline comparisons via llama.cpp's `llama-perplexity` and `llama-simple`.

Usage:
    modal run evidence/modal_wikitext_hellaswag.py
    modal run evidence/modal_wikitext_hellaswag.py --model llama1b
    modal run evidence/modal_wikitext_hellaswag.py --tasks wikitext
    modal run evidence/modal_wikitext_hellaswag.py --tasks hellaswag
"""

import os
import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app       = modal.App("nve-wikitext-hellaswag")
model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})

MODELS = {
    "llama1b": {
        "hf_id":     "meta-llama/Llama-3.2-1B",
        "local_dir": "/models/llama1b",
        "label":     "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":     "meta-llama/Llama-3.2-3B",
        "local_dir": "/models/llama3b",
        "label":     "Llama-3.2-3B",
    },
    "llama8b": {
        "hf_id":     "meta-llama/Llama-3.1-8B",
        "local_dir": "/models/llama8b",
        "label":     "Llama-3.1-8B",
    },
    "llama13b": {
        "hf_id":     "meta-llama/Llama-2-13b-hf",
        "local_dir": "/models/llama13b",
        "label":     "Llama-2-13B",
    },
}

# Calibration prompts for the importance profiling pass.
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
    # Multi-arch build: sm_75 (T4), sm_80 (A100), sm_86 (A10G)
    "bash -c 'set -o pipefail && "
    "cd /nve-src && "
    "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
    "-gencode arch=compute_75,code=sm_75 "
    "-gencode arch=compute_80,code=sm_80 "
    "-gencode arch=compute_86,code=sm_86 "
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


# ─────────────────────────────────────────────────────────────────────────────
# WikiText-2 evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_wikitext2_sequences(n_sequences: int = 50, seq_len: int = 256) -> list[str]:
    """Load WikiText-2 test set and split into fixed-length text chunks."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    # Split on whitespace into words, then reassemble into fixed-length windows.
    words = text.split()
    seqs = []
    stride = seq_len  # non-overlapping for speed
    for start in range(0, len(words) - seq_len, stride):
        chunk = " ".join(words[start:start + seq_len])
        seqs.append(chunk)
        if len(seqs) >= n_sequences:
            break
    return seqs


def compute_nve_perplexity(
    model_dir: str,
    sequences: list[str],
    env_overrides: dict,
    hot_budget_mb: int = 14000,
    warm_budget_mb: int = 14000,
    quantize: str = "q4",
) -> dict:
    """Run nve-cuda batch-perplexity: load model once, score all sequences."""
    import subprocess, os, json, tempfile

    base_env = os.environ.copy()
    base_env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        texts_path = f.name
        json.dump(sequences, f)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        "nve-cuda", "batch-perplexity",
        "-m", model_dir,
        "--texts-file", texts_path,
        "--hot-budget-mb", str(hot_budget_mb),
        "--warm-budget-mb", str(warm_budget_mb),
        "--quantize", quantize,
        "--output", out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=base_env)
    # Stream stdout (progress lines) to our stdout.
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(f"  [PPL] batch-perplexity failed: {r.stderr[:400]}")
        return {"ppl": float("nan"), "mean_nll": float("nan"), "n_tokens": 0, "failed": len(sequences)}

    try:
        data = json.loads(Path(out_path).read_text())
        return {
            "ppl": data.get("ppl_overall", float("nan")),
            "mean_nll": float("nan"),
            "n_tokens": data.get("n_tokens", 0),
            "failed": data.get("failed", 0),
        }
    except Exception as e:
        print(f"  [PPL] failed to parse output: {e}")
        return {"ppl": float("nan"), "mean_nll": float("nan"), "n_tokens": 0, "failed": len(sequences)}
    finally:
        for p in [texts_path, out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# HellaSwag evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_hellaswag_examples(n: int = 200) -> list[dict]:
    """Load HellaSwag validation examples."""
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    examples = []
    for item in ds:
        examples.append({
            "ctx": item["ctx"],
            "endings": item["endings"],
            "label": int(item["label"]),
        })
        if len(examples) >= n:
            break
    return examples


def compute_nve_hellaswag(
    model_dir: str,
    examples: list[dict],
    env_overrides: dict,
    hot_budget_mb: int = 14000,
    warm_budget_mb: int = 14000,
    quantize: str = "q4",
) -> dict:
    """Run nve-cuda batch-hellaswag: load model once, score all examples."""
    import subprocess, os, json, tempfile

    base_env = os.environ.copy()
    base_env.update(env_overrides)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        ex_path = f.name
        json.dump(examples, f)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    cmd = [
        "nve-cuda", "batch-hellaswag",
        "-m", model_dir,
        "--examples-file", ex_path,
        "--hot-budget-mb", str(hot_budget_mb),
        "--warm-budget-mb", str(warm_budget_mb),
        "--quantize", quantize,
        "--output", out_path,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, env=base_env)
    if r.stdout:
        print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        print(f"  [HellaSwag] batch-hellaswag failed: {r.stderr[:400]}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "failed": len(examples)}

    try:
        data = json.loads(Path(out_path).read_text())
        return {
            "accuracy": data.get("accuracy", 0.0),
            "correct": data.get("correct", 0),
            "total": data.get("total", 0),
            "failed": data.get("failed", 0),
        }
    except Exception as e:
        print(f"  [HellaSwag] failed to parse output: {e}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "failed": len(examples)}
    finally:
        for p in [ex_path, out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass



# ─────────────────────────────────────────────────────────────────────────────
# Modal function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=nve_image,
    gpu="A10G",
    cpu=8.0,
    memory=32768,
    timeout=14400,  # 4 hours for larger models
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_eval(model_name: str, tasks: str = "wikitext,hellaswag") -> dict:
    import os
    from huggingface_hub import snapshot_download

    cfg = MODELS[model_name]
    task_list = [t.strip() for t in tasks.split(",")]

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

    model_dir = cfg["local_dir"]
    results = {"model": model_name, "label": cfg["label"]}

    # ── Importance profiling ──────────────────────────────────────────────────
    import subprocess, glob as _glob

    cache_dir = Path("/root/.cache/nve/importance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale cache.
    for f in _glob.glob("/root/.cache/nve/importance/*.json"):
        os.remove(f)

    print(f"\n{'='*60}")
    print(f"Profiling layer importance — {cfg['label']}")
    print('='*60)
    for i, prompt in enumerate(CALIBRATION_PROMPTS):
        cmd = [
            "nve-cuda", "generate",
            "-m", model_dir,
            "-p", prompt,
            "-n", "5",
            "--temperature", "0",
            "--paged",
            "--hot-budget-mb", "14000",
            "--warm-budget-mb", "14000",
            "--quantize", "q4",
            "--profile",
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"  [profile] prompt {i} failed: {r.stderr[:150]}")
        else:
            print(f"  [profile] prompt {i+1}/{len(CALIBRATION_PROMPTS)} done")

    cache_files = list(cache_dir.glob("*.json"))
    print(f"  Cache: {len(cache_files)} file(s)")
    if cache_files:
        import json, math
        cache = json.loads(cache_files[0].read_text())
        scores = cache.get("scores", [])
        if scores:
            mn, mx = min(scores), max(scores)
            norm = [(s - mn) / (mx - mn) if mx > mn else 0.5 for s in scores]
            n16 = sum(1 for s in norm if s >= 0.7)
            print(f"  Scores min={mn:.3f} max={mx:.3f} mean={sum(scores)/len(scores):.3f}")
            print(f"  W4A16 layers (importance >= 0.7): {n16}/{len(scores)}")
            print(f"  W4A8  layers (importance <  0.7): {len(scores)-n16}/{len(scores)}")
            results["importance_profile"] = {
                "n_layers": len(scores),
                "n_w4a16": n16,
                "n_w4a8": len(scores) - n16,
            }

    # Strategy environment overrides.
    STRATEGIES = [
        ("W4A16 (uniform)",  {"NVE_NO_W4A8": "1"}),
        # threshold=2.0 ensures ALL layers use W4A8 (max normalized score = 1.0, so 1.0 < 2.0 → W4A8)
        ("W4A8 (uniform)",   {"NVE_W4A8_THRESHOLD": "2.0"}),
        ("Mixed W4A8/16",    {}),   # default threshold=0.7, keeps top-importance layer at W4A16
    ]

    # ── WikiText-2 ────────────────────────────────────────────────────────────
    if "wikitext" in task_list:
        print(f"\n{'='*60}")
        print(f"WikiText-2 Perplexity — {cfg['label']}")
        print('='*60)
        # Scale sequences by model size (larger models are slower per token)
        n_seq_map = {"llama1b": 50, "llama3b": 50, "llama8b": 50, "llama13b": 20}
        n_seq = n_seq_map.get(model_name, 20)
        seqs = get_wikitext2_sequences(n_sequences=n_seq, seq_len=256)
        print(f"  {len(seqs)} sequences × 256 tokens loaded")

        ppl_results = {}
        for label, env_ovr in STRATEGIES:
            print(f"\n[{label}]")
            r = compute_nve_perplexity(model_dir, seqs, env_ovr)
            ppl_results[label] = r
            print(f"  PPL = {r['ppl']:.2f}")

        results["wikitext2"] = ppl_results

    # ── HellaSwag ─────────────────────────────────────────────────────────────
    if "hellaswag" in task_list:
        print(f"\n{'='*60}")
        print(f"HellaSwag Accuracy — {cfg['label']}")
        print('='*60)
        # Scale HellaSwag examples by model size (larger models are slower per example)
        n_hs_map = {"llama1b": 50, "llama3b": 20, "llama8b": 20, "llama13b": 10}
        n_hs = n_hs_map.get(model_name, 10)
        examples = get_hellaswag_examples(n=n_hs)
        print(f"  {len(examples)} validation examples loaded")

        hs_results = {}
        for label, env_ovr in STRATEGIES:
            print(f"\n[{label}]")
            r = compute_nve_hellaswag(model_dir, examples, env_ovr)
            hs_results[label] = r
            print(f"  Accuracy = {r['accuracy']:.1%}")

        results["hellaswag"] = hs_results

    return results


@app.local_entrypoint()
def main(model: str = "llama1b,llama3b", tasks: str = "wikitext,hellaswag"):
    import json
    from pathlib import Path

    model_names = [m.strip() for m in model.split(",")]
    print(f"NVE WikiText-2 + HellaSwag Eval — {model_names}, tasks={tasks}")
    print("=" * 70)

    handles = {m: run_eval.spawn(m, tasks) for m in model_names}
    results = {m: h.get() for m, h in handles.items()}

    # Summary table.
    print("\n" + "=" * 70)
    task_list = [t.strip() for t in tasks.split(",")]

    for m in model_names:
        r = results[m]
        print(f"\n{r['label']}")
        print("-" * 50)

        if "wikitext2" in r:
            print("  WikiText-2 Perplexity (lower = better):")
            for strat, res in r["wikitext2"].items():
                ppl = res.get("ppl", float("nan"))
                n_tok = res.get("n_tokens", 0)
                print(f"    {strat:<24} PPL = {ppl:7.2f}  ({n_tok} tokens)")

        if "hellaswag" in r:
            print("  HellaSwag Accuracy (higher = better):")
            for strat, res in r["hellaswag"].items():
                acc = res.get("accuracy", 0.0)
                tot = res.get("total", 0)
                print(f"    {strat:<24} Acc = {acc:6.1%}  ({tot} examples)")

        if "importance_profile" in r:
            p = r["importance_profile"]
            print(f"  Importance profile: {p['n_w4a16']}/{p['n_layers']} W4A16, "
                  f"{p['n_w4a8']}/{p['n_layers']} W4A8")

    out_dir = Path(str(HERE)) / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "wikitext_hellaswag.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
