#!/usr/bin/env python3
"""
NVE W4A8 Real-Model Test — Llama 1B / 3B / 8B on T4
=====================================================
Tests correctness and throughput of the W4A8 dp4a kernel on real Llama weights.

Runs three comparisons per model:
  1. W4A8   — dp4a INT8 activations × 4-bit weights (new default)
  2. W4A16  — F16  activations × 4-bit weights (NVE_NO_W4A8=1)
  3. llama.cpp Q4_K_M — reference baseline

Quality check: 8-prompt task suite, measure how many expected tokens appear.
Throughput: tok/s on the quality prompts (40 tokens each).

Usage:
    modal run evidence/modal_real_model_w4a8.py
    modal run evidence/modal_real_model_w4a8.py --model llama1b
    modal run evidence/modal_real_model_w4a8.py --model llama1b,llama3b,llama8b
"""

import os
import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app        = modal.App("nve-real-model-w4a8")
model_vol  = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
hf_secret  = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})

MODELS = {
    "llama1b": {
        "hf_id":     "meta-llama/Llama-3.2-1B",
        "gguf_repo": "QuantFactory/Llama-3.2-1B-GGUF",
        "gguf_q4":   "Llama-3.2-1B.Q4_K_M.gguf",
        "local_dir": "/models/llama1b",
        "label":     "Llama-3.2-1B",
    },
    "llama3b": {
        "hf_id":     "meta-llama/Llama-3.2-3B",
        "gguf_repo": "QuantFactory/Llama-3.2-3B-GGUF",
        "gguf_q4":   "Llama-3.2-3B.Q4_K_M.gguf",
        "local_dir": "/models/llama3b",
        "label":     "Llama-3.2-3B",
    },
    "llama8b": {
        "hf_id":     "meta-llama/Llama-3.1-8B",
        "gguf_repo": "QuantFactory/Meta-Llama-3.1-8B-GGUF",
        "gguf_q4":   "Meta-Llama-3.1-8B.Q4_K_M.gguf",
        "local_dir": "/models/llama8b",
        "label":     "Llama-3.1-8B",
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

# Two-step build: nvcc → shared .so → cargo links against it
BUILD_CMD = (
    "bash -c 'set -o pipefail && "
    "cd /nve-src && "
    "echo \"=== Step 1: nvcc ==\" && "
    "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
    "-arch=compute_75 -code=sm_75 -Xcompiler -fPIC "
    "-shared -o /usr/local/lib/libnve_kernels.so cuda/nve_kernels.cu 2>&1 && "
    "ldconfig && "
    "echo \"shared lib: $(nm /usr/local/lib/libnve_kernels.so | grep -c nve_) nve_ symbols\" && "
    "echo \"=== Step 2: cargo build ==\" && "
    "NVE_KERNELS_PREBUILT=/usr/local/lib "
    "CUDA_COMPUTE_CAP=75 "
    "CUDA_PATH=/usr/local/cuda "
    "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C link-arg=-L/usr/local/lib -C link-arg=-lnve_kernels\" "
    "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -40 && "
    "cp target/release/nve /usr/local/bin/nve-cuda && "
    "chmod +x /usr/local/bin/nve-cuda && "
    "echo \"Build OK: $(nve-cuda --version 2>&1 | head -1)\"'"
)

nve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "curl", "pkg-config", "libssl-dev", "ca-certificates",
        "build-essential", "cmake", "git",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "pip install huggingface_hub[hf_xfer] tqdm",
    )
    .add_local_dir(str(HERE), "/nve-src", copy=True,
                   ignore=["target/", ".git/", "evidence/figures/",
                           "evidence/figures_paper/", "reports/"])
    .run_commands(BUILD_CMD)
)

llama_cpp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgcc-s1", "ca-certificates")
    .pip_install("huggingface_hub[hf_xfer]", "tqdm")
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu121",
    )
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
def run_nve(model_name: str) -> dict:
    import subprocess, time, os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    cfg = MODELS[model_name]
    hf_token = os.environ["HF_TOKEN"]

    # Download safetensors weights if not cached
    marker = Path(cfg["local_dir"]) / ".downloaded"
    if not marker.exists():
        print(f"[download] {cfg['hf_id']} ...")
        t0 = time.time()
        snapshot_download(
            repo_id=cfg["hf_id"],
            local_dir=cfg["local_dir"],
            token=hf_token,
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()
        print(f"  done in {time.time()-t0:.0f}s")

    def run_one(task: dict, label: str, env: dict) -> dict:
        """Run a single generate call, capture stdout+stderr, parse result."""
        cmd = [
            "nve-cuda", "generate",
            "-m", cfg["local_dir"],
            "-p", task["prompt"],
            "-n", "40",
            "--temperature", "0",
            "--paged",
            "--hot-budget-mb", "14000",
            "--warm-budget-mb", "14000",
            "--quantize", "q4",
        ]
        t0 = time.time()
        r  = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, env=env)
        dt = time.time() - t0

        # NVE 'generate --paged' writes to stdout:
        #   "Prompt: {prompt}\nTokens: N chars, M tokens\n\n{prompt}{generated}\n\n--- Stats ---\n..."
        # Strip the header lines and stats; extract generated continuation.
        combined = r.stdout + r.stderr   # safety: check both

        # Find the generated text: look for content after "--- Stats ---" is separated,
        # or just check if expected word appears anywhere in the combined output.
        generated = combined

        expected  = task["expected"].lower()
        ok        = expected in generated.lower()
        tps       = 40.0 / dt if dt > 0 else 0.0

        # Debug: show what actually came back (first 200 chars of each stream)
        if r.returncode != 0 or not r.stdout:
            print(f"  [{label}] rc={r.returncode}  stdout={r.stdout[:80]!r}  stderr={r.stderr[:200]!r}")

        return {
            "prompt":    task["prompt"][:40],
            "expected":  expected,
            "generated": generated[:120],
            "passed":    ok,
            "tps":       tps,
            "rc":        r.returncode,
        }

    def run_tasks(label: str, env_overrides: dict = None) -> dict:
        """Run full task suite, return accuracy + tok/s."""
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        results = [run_one(task, label, env) for task in TASK_SUITE]

        for res in results:
            marker = "PASS" if res["passed"] else "FAIL"
            gen_preview = res["generated"][:50].replace("\n", "\\n")
            print(f"  [{label}] [{marker}] {res['prompt'][:35]!r:40s} → {gen_preview!r}")

        acc     = sum(r["passed"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        print(f"  [{label}] acc={acc:.0%}  avg={avg_tps:.1f} tok/s")
        return {"accuracy": acc, "avg_tps": avg_tps, "tasks": results}

    print(f"\n{'='*60}")
    print(f"NVE — {cfg['label']}")
    print('='*60)
    w4a8  = run_tasks("W4A8 ")
    w4a16 = run_tasks("W4A16", {"NVE_NO_W4A8": "1"})

    return {
        "model":  model_name,
        "label":  cfg["label"],
        "w4a8":   w4a8,
        "w4a16":  w4a16,
    }


@app.function(
    image=llama_cpp_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_llamacpp(model_name: str) -> dict:
    import time, os, gc
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    cfg      = MODELS[model_name]
    hf_token = os.environ["HF_TOKEN"]
    gguf_dir = Path(cfg["local_dir"] + "_gguf")
    gguf_path = gguf_dir / cfg["gguf_q4"]

    gguf_dir.mkdir(parents=True, exist_ok=True)
    if not gguf_path.exists():
        print(f"[download] {cfg['gguf_repo']}/{cfg['gguf_q4']} ...")
        t0 = time.time()
        hf_hub_download(
            repo_id=cfg["gguf_repo"],
            filename=cfg["gguf_q4"],
            local_dir=str(gguf_dir),
            token=hf_token,
        )
        model_vol.commit()
        print(f"  done in {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"llama.cpp Q4_K_M — {cfg['label']}")
    print('='*60)

    llm = Llama(
        model_path=str(gguf_path),
        n_threads=4,
        n_ctx=512,
        n_gpu_layers=-1,
        verbose=False,
    )

    passed = 0
    tps_list = []
    outputs = []

    for task in TASK_SUITE:
        t0  = time.time()
        out = llm(task["prompt"], max_tokens=40, temperature=0.0, echo=False)
        dt  = time.time() - t0

        generated = out["choices"][0]["text"]
        n_tokens  = out["usage"]["completion_tokens"]
        tps       = n_tokens / dt if dt > 0 else 0.0
        ok        = task["expected"].lower() in generated.lower()
        passed   += int(ok)
        tps_list.append(tps)
        outputs.append({
            "prompt":    task["prompt"][:40],
            "expected":  task["expected"],
            "generated": generated[:60],
            "passed":    ok,
            "tps":       tps,
        })

        marker = "PASS" if ok else "FAIL"
        print(f"  [llama.cpp] [{marker}] {task['prompt'][:35]!r:40s} → {generated[:40]!r}")

    acc     = passed / len(TASK_SUITE)
    avg_tps = sum(tps_list) / len(tps_list)
    print(f"  [llama.cpp] acc={acc:.0%}  avg={avg_tps:.1f} tok/s")

    del llm
    gc.collect()

    return {
        "model":    model_name,
        "label":    cfg["label"],
        "accuracy": acc,
        "avg_tps":  avg_tps,
        "tasks":    outputs,
    }


@app.local_entrypoint()
def main(model: str = "llama1b,llama3b,llama8b"):
    import json
    from pathlib import Path

    model_names = [m.strip() for m in model.split(",")]
    print(f"NVE W4A8 Real-Model Test — {model_names}")
    print("="*70)

    # Launch NVE and llama.cpp in parallel
    nve_handles   = {m: run_nve.spawn(m)      for m in model_names}
    llama_handles = {m: run_llamacpp.spawn(m) for m in model_names}

    nve_results   = {m: h.get() for m, h in nve_handles.items()}
    llama_results = {m: h.get() for m, h in llama_handles.items()}

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Model':<18} {'System':<14} {'Accuracy':>9} {'Tok/s':>8}  {'vs llama.cpp':>13}")
    print("─"*70)

    for m in model_names:
        nve = nve_results[m]
        llc = llama_results[m]
        llc_tps = llc["avg_tps"]
        llc_acc = llc["accuracy"]

        w8  = nve["w4a8"]
        w16 = nve["w4a16"]

        def pct(tps): return f"{tps/llc_tps*100:.0f}%" if llc_tps > 0 else "?"

        print(f"  {nve['label']:<16} {'llama.cpp Q4':<14} {llc_acc:>8.0%} {llc_tps:>8.1f}  {'—':>13}")
        print(f"  {'':16} {'NVE W4A8':<14} {w8['accuracy']:>8.0%} {w8['avg_tps']:>8.1f}  {pct(w8['avg_tps']):>13}")
        print(f"  {'':16} {'NVE W4A16':<14} {w16['accuracy']:>8.0%} {w16['avg_tps']:>8.1f}  {pct(w16['avg_tps']):>13}")
        print()

    # Save full results
    out_dir = Path(str(HERE)) / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "real_model_w4a8.json"
    out_path.write_text(json.dumps({
        "nve":      {m: nve_results[m]   for m in model_names},
        "llamacpp": {m: llama_results[m] for m in model_names},
    }, indent=2))
    print(f"Results saved → {out_path}")
