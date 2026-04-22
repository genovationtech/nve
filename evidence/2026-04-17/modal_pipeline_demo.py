#!/usr/bin/env python3
"""
MCAP Pipeline-Simplification Demonstration
===========================================

Demonstrates the central deployment claim: *one model, one profile, N devices*.

Phase 1: compute an MCAP profile for Llama-3.2-1B on one T4 container,
         save to a shared volume as profile.json.

Phase 2: spin up three T4 containers in parallel, each with a different
         memory budget (2 GB / 4 GB / 14 GB). Each container loads the
         SAME unmodified weights and the SAME profile.json, runs the
         abc-test 8-task suite, and reports:
           - profiling_time_ms  (expected: 0 — profile injected)
           - profile_sha256     (expected: identical across all three)
           - task accuracy
           - peak memory
           - tok/s

The three containers share nothing except the weights and the profile JSON;
they do not coordinate at runtime. If profile portability holds, all three
run correctly with profiling_time_ms=0.

Usage:
  modal run evidence/modal_pipeline_demo.py
"""

import os
import modal
from pathlib import Path

HERE = Path(__file__).parent.parent

app = modal.App("mcap-pipeline-demo")

model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)
profile_vol = modal.Volume.from_name("mcap-profile-demo", create_if_missing=True)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})

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
                   ignore=["target/", ".git/", "evidence/figures/",
                           "evidence/figures_paper/", "paper/", "reports/"])
    .run_commands(
        # Build CUDA kernels as a shared .so (lld-friendly), multi-arch for T4 + A10G + PTX JIT
        "mkdir -p /nve-prebuilt && "
        "/usr/local/cuda/bin/nvcc -O3 --use_fast_math "
        "-gencode=arch=compute_75,code=sm_75 "   # T4 (Turing)
        "-gencode=arch=compute_86,code=sm_86 "   # A10G (Ampere)
        "-gencode=arch=compute_86,code=compute_86 "  # PTX fallback for future archs
        "-Xcompiler -fPIC "
        "-shared /nve-src/cuda/nve_kernels.cu "
        "-o /nve-prebuilt/libnve_kernels.so",
        # Sanity-check that the symbols we need are exported
        "nm -D /nve-prebuilt/libnve_kernels.so | grep nve_matvec_f16 || "
        "(echo 'ERROR: nve_matvec_f16 missing from .so' && nm -D /nve-prebuilt/libnve_kernels.so | head -40 && false)",
        # Build the Rust binary, forcing GNU ld with --no-as-needed so the .so link isn't elided
        "bash -c 'cd /nve-src && "
        "CUDA_COMPUTE_CAP=75 CUDA_PATH=/usr/local/cuda "
        "NVE_KERNELS_PREBUILT=/nve-prebuilt "
        "LD_LIBRARY_PATH=/nve-prebuilt:$LD_LIBRARY_PATH "
        "RUSTFLAGS=\"-C target-cpu=x86-64-v3 -C linker=gcc "
        "-C link-arg=-Wl,--no-as-needed "
        "-C link-arg=-L/nve-prebuilt "
        "-C link-arg=-lnve_kernels "
        "-C link-arg=-Wl,-rpath,/nve-prebuilt\" "
        "$HOME/.cargo/bin/cargo build --release --features cuda > /tmp/build.log 2>&1 || "
        "(echo \"BUILD FAIL TAIL:\" && tail -80 /tmp/build.log && false)'",
        "cp /nve-src/target/release/nve /usr/local/bin/nve-cuda",
        "chmod +x /usr/local/bin/nve-cuda",
        "echo /nve-prebuilt >> /etc/ld.so.conf.d/nve.conf && ldconfig",
    )
)


def ensure_model(model_dir: str):
    """Download Llama-3.2-1B if not already cached on the volume."""
    import os
    from huggingface_hub import snapshot_download
    marker = Path(model_dir) / ".downloaded"
    if not marker.exists():
        print(f"[download] Llama-3.2-1B -> {model_dir}")
        snapshot_download(
            repo_id="meta-llama/Llama-3.2-1B",
            local_dir=model_dir,
            token=os.environ["HF_TOKEN"],
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()


@app.function(
    image=nve_cuda_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=900,
    volumes={"/models": model_vol, "/profiles": profile_vol},
    secrets=[hf_secret],
)
def phase1_compute_profile() -> dict:
    """Phase 1: compute MCAP profile, save to shared volume."""
    import subprocess, time, json, hashlib
    from pathlib import Path

    model_dir = "/models/llama1b"
    profile_path = "/profiles/llama1b_profile.json"
    out_file = "/tmp/phase1_result.json"

    ensure_model(model_dir)

    # Unconstrained run with --save-profile
    cmd = [
        "nve-cuda", "abc-test",
        "-m", model_dir,
        "-n", "8",
        "-o", out_file,
        "--device", "cuda:0",
        "--hot-budget-mb", "4000",
        "--warm-budget-mb", "4000",
        "--save-profile", profile_path,
        "--configs", "b",  # hot-only requires profiling; this is the run that emits the .json
    ]
    print("Phase 1:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:])
        return {"ok": False, "error": proc.stderr[-500:]}

    profile_vol.commit()
    profile_bytes = Path(profile_path).read_bytes()
    profile_hash = hashlib.sha256(profile_bytes).hexdigest()

    data = json.loads(Path(out_file).read_text())
    baseline_profiling = None
    for cfg in data.get("configurations", []):
        pi = cfg.get("profiling_info") or {}
        if pi:
            baseline_profiling = pi.get("profiling_time_ms", 0.0)
            break

    return {
        "ok": True,
        "elapsed_s": elapsed,
        "profile_path": profile_path,
        "profile_bytes": len(profile_bytes),
        "profile_sha256": profile_hash,
        "phase1_profiling_time_ms": baseline_profiling,
    }


def _run_phase2_common(budget_mb: int, gpu_name: str) -> dict:
    """Shared phase-2 body. Invoked from multiple gpu-specialized wrappers below."""
    import subprocess, time, json, hashlib
    from pathlib import Path

    model_dir = "/models/llama1b"
    profile_path = "/profiles/llama1b_profile.json"
    ensure_model(model_dir)

    profile_vol.reload()
    profile_bytes = Path(profile_path).read_bytes()
    profile_hash = hashlib.sha256(profile_bytes).hexdigest()

    out_file = f"/tmp/phase2_{gpu_name}_{budget_mb}.json"
    cmd = [
        "nve-cuda", "abc-test",
        "-m", model_dir,
        "-n", "8",
        "-o", out_file,
        "--device", "cuda:0",
        "--hot-budget-mb", str(budget_mb),
        "--warm-budget-mb", str(min(budget_mb * 2, 8000)),
        "--profile-from", profile_path,
        "--configs", "b,c",
    ]
    print(f"[{gpu_name} budget={budget_mb}MB]", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1200:])
        return {"ok": False, "gpu": gpu_name, "budget_mb": budget_mb, "error": proc.stderr[-500:]}

    data = json.loads(Path(out_file).read_text())
    results = []
    for cfg in data.get("configurations", []):
        pi = cfg.get("profiling_info") or {}
        summary = cfg.get("summary") or {}
        results.append({
            "config": cfg.get("config"),
            "task_accuracy": cfg.get("task_accuracy"),
            "tok_per_sec": summary.get("avg_tokens_per_sec"),
            "profiling_time_ms": pi.get("profiling_time_ms"),
            "peak_memory_mb": summary.get("peak_memory_mb"),
        })
    return {
        "ok": True,
        "gpu": gpu_name,
        "budget_mb": budget_mb,
        "elapsed_s": elapsed,
        "profile_sha256": profile_hash,
        "results": results,
    }


@app.function(
    image=nve_cuda_image,
    gpu="T4",
    cpu=4.0,
    memory=16384,
    timeout=900,
    volumes={"/models": model_vol, "/profiles": profile_vol},
    secrets=[hf_secret],
)
def phase2_run_t4(budget_mb: int) -> dict:
    """Phase 2 on T4 (sm_75 / Turing)."""
    return _run_phase2_common(budget_mb, "T4")


@app.function(
    image=nve_cuda_image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=900,
    volumes={"/models": model_vol, "/profiles": profile_vol},
    secrets=[hf_secret],
)
def phase2_run_a10g(budget_mb: int) -> dict:
    """Phase 2 on A10G (sm_86 / Ampere) -- same profile, different silicon."""
    return _run_phase2_common(budget_mb, "A10G")


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("=== Phase 1: compute MCAP profile ===")
    p1 = phase1_compute_profile.remote()
    print(json.dumps(p1, indent=2))
    if not p1.get("ok"):
        raise SystemExit("Phase 1 failed")

    print("\n=== Phase 2: heterogeneous silicon, different budgets ===")
    # Mix of T4 (Turing sm_75) and A10G (Ampere sm_86) at different budgets.
    # Same profile artifact, different silicon classes.
    t4_budgets = [2000, 4000, 14000]
    a10g_budgets = [2000, 4000, 14000]
    t4_results = list(phase2_run_t4.map(t4_budgets))
    a10g_results = list(phase2_run_a10g.map(a10g_budgets))
    results = t4_results + a10g_results

    print("\n=== SUMMARY ===")
    print(f"Profile computed:   {p1['profile_bytes']} bytes")
    print(f"Profile sha256:     {p1['profile_sha256'][:16]}...\n")

    print(f"{'GPU':>5} | {'Budget':>7} | {'Config':>16} | {'Profiling':>10} | {'Accuracy':>9} | {'tok/s':>7} | {'sha-match':>9}")
    print("-" * 90)
    for r in results:
        if not r.get("ok"):
            print(f"{r.get('gpu','?'):>5} | {r.get('budget_mb','?'):>5}MB | ERROR: {r.get('error','?')[:50]}")
            continue
        sha_match = "yes" if r["profile_sha256"] == p1["profile_sha256"] else "NO"
        for cr in r["results"]:
            acc = cr.get("task_accuracy")
            tps = cr.get("tok_per_sec")
            ptm = cr.get("profiling_time_ms")
            print(f"{r['gpu']:>5} | {r['budget_mb']:>5}MB | {cr.get('config','?'):>16} | "
                  f"{ptm if ptm is not None else '?':>10} | "
                  f"{acc if acc is not None else '?':>9} | "
                  f"{tps if tps is not None else '?':>7.2f} | "
                  f"{sha_match:>9}")

    # Save full results
    out_json = {
        "phase1": p1,
        "phase2": results,
    }
    Path("/tmp/pipeline_demo_results.json").write_text(json.dumps(out_json, indent=2))
    print("\nFull results saved to /tmp/pipeline_demo_results.json (container-side)")
    # Also save locally via print marker for easy extraction
    print("\n=== RESULTS_JSON_BEGIN ===")
    print(json.dumps(out_json))
    print("=== RESULTS_JSON_END ===")
