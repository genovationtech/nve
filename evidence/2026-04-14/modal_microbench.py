#!/usr/bin/env python3
"""
NVE CUDA Kernel Microbenchmark  (~3 min on Modal, no model download)
======================================================================
Compiles cuda/nve_kernels.cu + cuda/bench_main.cu on a T4 and reports
per-kernel throughput and projected tok/s for 1B-model decode dims.

Usage:
    modal run evidence/modal_microbench.py
    modal run evidence/modal_microbench.py --hidden 4096   # 8B dims
"""

import modal
from pathlib import Path

app  = modal.App("nve-microbench")
HERE = Path(__file__).parent.parent  # nve/ root

# Minimal CUDA devel image — just nvcc + cuda headers, no Rust / HF
bench_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
    .add_local_dir(str(HERE / "cuda"), "/nve-cuda-src", copy=True)
    .run_commands(
        # Compile kernels + driver into a single binary
        "nvcc -O3 --use_fast_math "
        "  -arch=compute_75 -code=sm_75 "
        "  /nve-cuda-src/nve_kernels.cu "
        "  /nve-cuda-src/bench_main.cu "
        "  -o /usr/local/bin/nve_bench && "
        "chmod +x /usr/local/bin/nve_bench && "
        "echo 'Build OK: nve_bench ready'"
    )
)


@app.function(
    image=bench_image,
    gpu="T4",
    cpu=2.0,
    memory=4096,
    timeout=300,
)
def run_microbench(
    hidden: int       = 2048,    # 1B: 2048 / 3B: 3072 / 8B: 4096
    intermediate: int = 8192,    # 1B: 8192 / 3B: 8192 / 8B: 14336
    num_heads: int    = 32,
    num_kv_heads: int = 8,
    head_dim: int     = 64,
    iters: int        = 2000,
) -> str:
    import subprocess, time

    cmd = [
        "nve_bench",
        str(hidden), str(intermediate),
        str(num_heads), str(num_kv_heads),
        str(head_dim), str(iters),
    ]
    t0  = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    dt  = time.time() - t0

    result = out.stdout
    if out.returncode != 0:
        result += "\nSTDERR: " + out.stderr[-2000:]
    result += f"\n[total elapsed: {dt:.1f}s]"
    return result


@app.local_entrypoint()
def main(
    hidden: int       = 2048,
    intermediate: int = 8192,
    num_heads: int    = 32,
    num_kv_heads: int = 8,
    head_dim: int     = 64,
    iters: int        = 2000,
):
    print("NVE CUDA Kernel Microbenchmark")
    print("=" * 60)
    result = run_microbench.remote(
        hidden, intermediate, num_heads, num_kv_heads, head_dim, iters
    )
    print(result)
