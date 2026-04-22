#!/usr/bin/env python3
"""
NVE CUDA Graph Capture Benchmark
=================================
Measures the impact of CUDA graph capture on decode throughput by running
three configurations back-to-back on the same T4:

  1. fused_w4_flash  — W4A16 + flash decode + fused QKV (prior best, no graph)
  2. graph_w4        — same kernels, captured into a CUDA graph and replayed

Expected improvement: ~0.6 ms/token eliminated kernel-launch overhead
→ from 124 tok/s to ~136-142 tok/s (~90-94% of llama.cpp Q4_0 at 150.8 tok/s).

Additionally eliminates CT::cat KV-cache allocations (32/step for 16-layer 1B):
→ further ~0.3 ms/token savings → potential ~148 tok/s (98% of llama.cpp).

Usage:
    modal run evidence/modal_graph_bench.py
    modal run evidence/modal_graph_bench.py --iters 1000 --model-size 1b
"""

import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app = modal.App("nve-graph-bench")

graph_bench_image = (
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
        # Build NVE with CUDA + new graph-capture kernels.
        # Uses the same two-step nvcc→shared lib → cargo build approach.
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
    image=graph_bench_image,
    gpu="T4",
    cpu=4.0,
    memory=8192,
    timeout=600,
)
def run_bench(model_size: str = "1b", iters: int = 2000) -> dict:
    """
    Run bench-random in three modes:
      1. fused W4+flash          — prior best (W4A16 + flash decode + fused QKV)
      2. graph W4+flash          — same kernels, CUDA graph capture + replay
    """
    import subprocess, time, os, re

    def bench(label: str, extra_args: list = None) -> dict:
        cmd = [
            "nve-cuda", "bench-random",
            "--device", "cuda:0",
            "--model-size", model_size,
            "--iters", str(iters),
            "--quantize", "w4",
        ] + (extra_args or [])

        print(f"\n>>> Running {label} ({iters} iters, {model_size}) …")
        t0 = time.time()
        r  = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        out = r.stdout + (("\nSTDERR: " + r.stderr[-1000:]) if r.returncode != 0 else "")
        print(out[:2000])

        m  = re.search(r"tok_per_s\s*:\s*([\d.]+)", out)
        m2 = re.search(r"ms_per_tok\s*:\s*([\d.]+)", out)
        return {
            "label":   label,
            "tok_s":   float(m.group(1))  if m  else 0.0,
            "ms_tok":  float(m2.group(1)) if m2 else 0.0,
            "elapsed": dt,
        }

    no_graph = bench("fused_w4+flash (no graph)")
    with_graph = bench("fused_w4+flash + CUDA graph", ["--graph"])

    def speedup(a, b):
        return a["tok_s"] / b["tok_s"] if b["tok_s"] > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"RESULTS — bench-random {model_size} / T4 / {iters} iters")
    print("=" * 70)
    print(f"  no graph (W4+flash)  : {no_graph['tok_s']:6.1f} tok/s  ({no_graph['ms_tok']:.3f} ms/tok)  [baseline]")
    print(f"  CUDA graph replay    : {with_graph['tok_s']:6.1f} tok/s  ({with_graph['ms_tok']:.3f} ms/tok)  [{speedup(with_graph, no_graph):.3f}× speedup]")
    print(f"  (llama.cpp Q4_0      :  150.8 tok/s  — reference)")
    print(f"  graph vs llama.cpp   : {with_graph['tok_s'] / 150.8 * 100:.1f}%")
    print("=" * 70)

    return {
        "model_size":            model_size,
        "iters":                 iters,
        "no_graph_tok_s":        no_graph["tok_s"],
        "graph_tok_s":           with_graph["tok_s"],
        "graph_speedup":         speedup(with_graph, no_graph),
        "graph_vs_llamacpp_pct": with_graph["tok_s"] / 150.8 * 100,
    }


@app.local_entrypoint()
def main(model_size: str = "1b", iters: int = 2000):
    print(f"NVE CUDA Graph Capture Benchmark — model={model_size}, iters={iters}")
    print("=" * 70)
    result = run_bench.remote(model_size, iters)
    print("\n--- Summary ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
