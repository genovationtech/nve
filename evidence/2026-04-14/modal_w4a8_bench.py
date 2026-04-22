#!/usr/bin/env python3
"""
NVE W4A8 (dp4a) vs W4A16 Benchmark
====================================
Compares decode throughput of:

  1. w4a16  — current baseline: F16 activations × 4-bit weights (W4A16 fused kernel)
  2. w4a8   — new: Q8 activations × 4-bit weights (dp4a, bias-corrected)

Expected: ~15-25% speedup from dp4a INT8 throughput on T4.
Target: reach 150.8 tok/s (llama.cpp Q4_0 reference).

Usage:
    modal run evidence/modal_w4a8_bench.py
    modal run evidence/modal_w4a8_bench.py --iters 2000 --model-size 3b
"""

import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app = modal.App("nve-w4a8-bench")

w4a8_image = (
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
    )
    .add_local_dir(str(HERE), "/nve-src", copy=True,
                   ignore=["target/", ".git/", "evidence/figures/",
                           "evidence/figures_paper/", "reports/"])
    .run_commands(
        "bash -c 'set -o pipefail && "
        "cd /nve-src && "
        "echo \"=== Step 1: nvcc → shared lib ==\" && "
        "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
        "-arch=compute_75 -code=sm_75 -Xcompiler -fPIC "
        "-shared -o /usr/local/lib/libnve_kernels.so cuda/nve_kernels.cu 2>&1 && "
        "ldconfig && "
        "echo \"shared lib OK: $(nm /usr/local/lib/libnve_kernels.so | grep -c nve_) nve_ symbols\" && "
        "echo \"=== Step 2: cargo build ==\" && "
        "NVE_KERNELS_PREBUILT=/usr/local/lib "
        "CUDA_COMPUTE_CAP=75 "
        "CUDA_PATH=/usr/local/cuda "
        "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C link-arg=-L/usr/local/lib -C link-arg=-lnve_kernels\" "
        "$HOME/.cargo/bin/cargo build --release --features cuda 2>&1 | tail -80 && "
        "cp target/release/nve /usr/local/bin/nve-cuda && "
        "chmod +x /usr/local/bin/nve-cuda && "
        "echo \"Build OK: $(nve-cuda --version 2>&1 | head -1)\"'",
    )
)


@app.function(
    image=w4a8_image,
    gpu="T4",
    cpu=4.0,
    memory=8192,
    timeout=600,
)
def run_bench(model_size: str = "1b", iters: int = 2000) -> dict:
    import subprocess, time, os, re

    def bench(label: str, env_overrides: dict = None) -> dict:
        cmd = [
            "nve-cuda", "bench-random",
            "--device", "cuda:0",
            "--model-size", model_size,
            "--iters", str(iters),
            "--quantize", "w4",
        ]
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        print(f"\n>>> Running {label} ({iters} iters, {model_size}) …")
        t0 = time.time()
        r  = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        out = r.stdout + (("\nSTDERR: " + r.stderr[-2000:]) if r.returncode != 0 else "")
        print(out[:3000])

        m   = re.search(r"tok_per_s\s*:\s*([\d.]+)", out)
        m2  = re.search(r"ms_per_tok\s*:\s*([\d.]+)", out)
        return {
            "label":   label,
            "tok_s":   float(m.group(1))  if m  else 0.0,
            "ms_tok":  float(m2.group(1)) if m2 else 0.0,
            "elapsed": dt,
        }

    # W4A16 baseline: disable W4A8 path
    w4a16 = bench("W4A16 baseline", {"NVE_NO_W4A8": "1"})
    # W4A8 dp4a: default (no override)
    w4a8  = bench("W4A8 dp4a")

    def speedup(a, b):
        return a["tok_s"] / b["tok_s"] if b["tok_s"] > 0 else 0.0

    LLAMACPP_REF = {"1b": 150.8, "3b": 70.9, "8b": 30.8}.get(model_size, 150.8)

    print("\n" + "=" * 70)
    print(f"RESULTS — bench-random {model_size} / T4 / {iters} iters")
    print("=" * 70)
    print(f"  W4A16 (F16 acts)     : {w4a16['tok_s']:6.1f} tok/s  ({w4a16['ms_tok']:.3f} ms/tok)  [baseline]")
    print(f"  W4A8  (dp4a Q8 acts) : {w4a8['tok_s']:6.1f} tok/s  ({w4a8['ms_tok']:.3f} ms/tok)  [{speedup(w4a8, w4a16):.3f}× speedup]")
    print(f"  llama.cpp Q4_0       : {LLAMACPP_REF:6.1f} tok/s  [reference]")
    print(f"  W4A8 vs llama.cpp    : {w4a8['tok_s'] / LLAMACPP_REF * 100:.1f}%")
    print("=" * 70)

    return {
        "model_size":              model_size,
        "iters":                   iters,
        "w4a16_tok_s":             w4a16["tok_s"],
        "w4a8_tok_s":              w4a8["tok_s"],
        "w4a8_speedup":            speedup(w4a8, w4a16),
        "w4a8_vs_llamacpp_pct":    w4a8["tok_s"] / LLAMACPP_REF * 100,
    }


@app.local_entrypoint()
def main(model_size: str = "1b", iters: int = 2000):
    print(f"NVE W4A8 dp4a Benchmark — model={model_size}, iters={iters}")
    print("=" * 70)
    result = run_bench.remote(model_size, iters)
    print("\n--- Summary ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
