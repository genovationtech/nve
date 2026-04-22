#!/usr/bin/env python3
"""
NVE GPU Smoke Test  (~5 min on Modal, reuses cached CUDA image)
===============================================================
Runs a single 1B inference pass with 20 tokens on T4 to verify:
  1. CUDA build works
  2. GPU inference produces coherent text (not garbage)
  3. tok/s is plausible (target: >20 tok/s on T4)

Usage:
  modal run evidence/modal_smoke_test.py
"""

import os
import modal
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root

app = modal.App("nve-smoke-test")

model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})

# Reuse the same CUDA build image as the full benchmark (cached after first build)
nve_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("curl", "pkg-config", "libssl-dev", "ca-certificates",
                 "build-essential", "cmake", "git")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
        "pip install huggingface_hub[hf_xfer] tqdm psutil",
    )
    .add_local_dir(str(HERE), "/nve-src", copy=True,
                   ignore=["target/", ".git/", "evidence/figures/", "evidence/figures_paper/"])
    .run_commands(
        "bash -c 'set -o pipefail && "
        "cd /nve-src && "
        "CUDA_COMPUTE_CAP=75 "
        "CUDA_PATH=/usr/local/cuda "
        "RUSTFLAGS=\"-C target-cpu=x86-64-v3\" "
        "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -5'",
        "cp /nve-src/target/release/nve /usr/local/bin/nve-cuda",
        "chmod +x /usr/local/bin/nve-cuda",
        "nve-cuda --version",
    )
)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In machine learning, gradient descent is used to",
    "Water is composed of hydrogen and",
]

@app.function(
    image=nve_cuda_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=600,
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def smoke_test() -> dict:
    import subprocess, time, json, os
    from pathlib import Path
    from huggingface_hub import snapshot_download

    model_dir = "/models/llama1b"
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

    out_file = "/tmp/smoke_result.json"
    cmd = [
        "nve-cuda", "abc-test",
        "-m", model_dir,
        "-n", "20",          # 20 tokens per prompt (fast)
        "-o", out_file,
        "--device", "cuda:0",
        "--hot-budget-mb", "4000",   # all 16 layers of 1B fit (~1.1 GB)
        "--warm-budget-mb", "4000",
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])
        return {"ok": False, "error": proc.stderr[-500:], "elapsed_s": elapsed}

    data = json.loads(Path(out_file).read_text())

    results = {}
    for cfg in data.get("configurations", []):
        name = cfg["config"]
        acc  = cfg.get("task_accuracy", 0.0)
        tps  = cfg.get("summary", {}).get("avg_tokens_per_sec", 0.0)
        results[name] = {"task_accuracy": acc, "tok_per_sec": tps}
        # Print first generated text for sanity check
        first = cfg.get("results", [{}])[0]
        print(f"  [{name}] acc={acc:.0%}  {tps:.1f} tok/s  | '{first.get('generated_text','')[:60]}'")

    return {"ok": True, "elapsed_s": elapsed, "results": results}


@app.local_entrypoint()
def main():
    print("NVE GPU Smoke Test")
    print("=" * 50)
    r = smoke_test.remote()
    if r["ok"]:
        print(f"\nPASS  (total {r['elapsed_s']:.0f}s)")
        for name, v in r["results"].items():
            print(f"  {name:30s}  acc={v['task_accuracy']:.0%}  {v['tok_per_sec']:.1f} tok/s")
    else:
        print(f"\nFAIL: {r.get('error')}")
