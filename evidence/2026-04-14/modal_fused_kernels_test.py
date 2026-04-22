#!/usr/bin/env python3
"""
NVE Fused CUDA Kernels Integration Test — W4A16 + Flash Decode + Fused QKV
============================================================================
Builds NVE with --features cuda on a T4 and runs four configurations:

  1. unfused F16      — candle per-op baseline (NVE_NO_FUSED=1)
  2. fused F16        — fused RMSNorm + RoPE + F16 matvec kernels
  3. fused W4A16      — W4A16 decode kernel only (prior milestone)
  4. fused W4A16+opt  — W4A16 + flash decode attention + fused QKV projection

New kernels in config 4:
  • nve_flash_decode_f16  — Q@K^T + online-softmax + scores@V in one warp-per-head pass
                            GQA-native: kv_head = head/groups (no expansion tensor)
  • nve_qkv_matvec_w4a16 — Q, K, V projections fused into one kernel launch

No model download — all tensors are random (F16 or packed INT4).
~8-12 minutes total on first run (Rust compilation cached after that).

Usage:
    modal run evidence/modal_fused_kernels_test.py
    modal run evidence/modal_fused_kernels_test.py --iters 200 --model-size 1b
    modal run evidence/modal_fused_kernels_test.py --model-size 3b
"""

import modal
from pathlib import Path
import re

HERE = Path(__file__).parent.parent  # nve/ root

app = modal.App("nve-fused-kernels-test")

# ── Build image: Rust + CUDA devel (same as gpu_benchmark) ───────────────────
fused_test_image = (
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
        # Build NVE with CUDA + custom kernels (build.rs compiles cuda/nve_kernels.cu).
        # CUDA_COMPUTE_CAP=75 → T4 sm_75; suppresses nvidia-smi probe during image build.
        # Step 1: compile CUDA kernels as a shared library (.so) so lld can link it.
        #   Static CUDA archives (.a) contain CUDA fat-binary ELF sections that lld
        #   cannot resolve; a shared library produced by nvcc is lld-compatible.
        # Step 2: cargo build with NVE_KERNELS_PREBUILT pointing to the .so dir.
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
    image=fused_test_image,
    gpu="T4",
    cpu=4.0,
    memory=8192,
    timeout=600,
)
def run_bench(model_size: str = "1b", iters: int = 2000) -> dict:
    """
    Run bench-random in four modes:
      1. unfused F16      — candle ops baseline
      2. fused F16        — fused RMSNorm + RoPE + F16 matvec
      3. fused W4A16      — W4A16 decode (INT4 weights, prior milestone)
      4. fused W4A16+opt  — W4A16 + flash decode + fused QKV projection
    """
    import subprocess, time, os, re

    def bench(label: str, no_fused: bool = False, quantize: str = "f16",
              no_flash: bool = False) -> dict:
        cmd = [
            "nve-cuda", "bench-random",
            "--device", "cuda:0",
            "--model-size", model_size,
            "--iters", str(iters),
            "--quantize", quantize,
        ]
        env = dict(os.environ)
        if no_fused:
            cmd.append("--no-fused")
        if no_flash:
            env["NVE_NO_FLASH"] = "1"

        print(f"\n>>> Running {label} ({iters} iters, {model_size}, quantize={quantize}) ...")
        t0 = time.time()
        r  = subprocess.run(cmd, capture_output=True, text=True, env=env)
        dt = time.time() - t0
        out = r.stdout + (("\nSTDERR: " + r.stderr[-800:]) if r.returncode != 0 else "")
        print(out)

        m  = re.search(r"tok_per_s\s*:\s*([\d.]+)", out)
        m2 = re.search(r"ms_per_tok\s*:\s*([\d.]+)", out)
        return {
            "label":   label,
            "tok_s":   float(m.group(1))  if m  else 0.0,
            "ms_tok":  float(m2.group(1)) if m2 else 0.0,
            "elapsed": dt,
        }

    unfused_f16   = bench("unfused_f16",         no_fused=True,  quantize="f16")
    fused_f16     = bench("fused_f16",           no_fused=False, quantize="f16")
    fused_w4_only = bench("fused_w4_no_flash",   no_fused=False, quantize="w4", no_flash=True)
    fused_w4_full = bench("fused_w4a16+flash",   no_fused=False, quantize="w4", no_flash=False)

    def speedup(a, b):
        return a["tok_s"] / b["tok_s"] if b["tok_s"] > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"RESULTS — bench-random {model_size} / T4 / {iters} iters")
    print("=" * 70)
    print(f"  unfused F16          : {unfused_f16['tok_s']:6.1f} tok/s  ({unfused_f16['ms_tok']:.3f} ms/tok)  [baseline]")
    print(f"  fused F16            : {fused_f16['tok_s']:6.1f} tok/s  ({fused_f16['ms_tok']:.3f} ms/tok)  [{speedup(fused_f16, unfused_f16):.2f}× vs unfused]")
    print(f"  fused W4 (no flash)  : {fused_w4_only['tok_s']:6.1f} tok/s  ({fused_w4_only['ms_tok']:.3f} ms/tok)  [{speedup(fused_w4_only, fused_f16):.2f}× vs F16]")
    print(f"  fused W4+flash+QKV   : {fused_w4_full['tok_s']:6.1f} tok/s  ({fused_w4_full['ms_tok']:.3f} ms/tok)  [{speedup(fused_w4_full, fused_w4_only):.2f}× vs no-flash / {speedup(fused_w4_full, fused_f16):.2f}× vs F16]")
    print("  (llama.cpp Q4_0      :  150.8 tok/s  — reference)")
    print("=" * 70)

    return {
        "model_size":              model_size,
        "iters":                   iters,
        "unfused_f16_tok_s":       unfused_f16["tok_s"],
        "fused_f16_tok_s":         fused_f16["tok_s"],
        "fused_w4_noflash_tok_s":  fused_w4_only["tok_s"],
        "fused_w4_flash_tok_s":    fused_w4_full["tok_s"],
        "flash_vs_noflash":        speedup(fused_w4_full, fused_w4_only),
        "w4_full_vs_f16":          speedup(fused_w4_full, fused_f16),
        "vs_llamacpp_pct":         fused_w4_full["tok_s"] / 150.8 * 100,
    }


@app.local_entrypoint()
def main(model_size: str = "1b", iters: int = 2000):
    print(f"NVE W4A16 Kernel Test — model={model_size}, iters={iters}")
    print("=" * 70)
    result = run_bench.remote(model_size, iters)
    print("\n--- Summary ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
