#!/usr/bin/env python3
"""
NVE 3B + 8B Experiment Runner
===============================
Runs full ABC benchmark (all 4 configs) on Modal cloud for:
  • Llama-3.2-3B  (3.2 B params, 28 layers, ~192 MB/layer)
  • Llama-3.1-8B  (8.0 B params, 32 layers, ~500 MB/layer)

Uses 32 GB container to handle 8B baseline (bf16 ≈ 16 GB weights).

Experiments per model:
  1. Full ABC  — baseline / A (Q4) / B (profiled hot) / C (PG+AWQ 2.0 bpw)
  2. BPW sweep — Config C only, bpw 0.5→4.0 (3B only; 8B too slow at low bpw)
  3. Layer sweep — Config B, N=2..max active layers (3B only)

Usage:
  modal run evidence/modal_3b_8b.py
  modal run evidence/modal_3b_8b.py --model llama3b
  modal run evidence/modal_3b_8b.py --model llama8b
"""

import modal
import json
import os
from pathlib import Path

HERE = Path(__file__).parent.parent  # nve/ root

app = modal.App("nve-3b-8b")

model_vol = modal.Volume.from_name("nve-model-weights", create_if_missing=True)

# Build NVE from source inside a CUDA-enabled container so the binary can
# dispatch AWQ profiling matmuls to the GPU via candle-core.
# The image is cached after the first build; subsequent runs reuse it.
nve_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "ca-certificates", "curl", "build-essential", "pkg-config",
        "libssl-dev",
    )
    # Install Rust stable toolchain.
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable",
    )
    # Copy NVE source into the image and build with CUDA feature.
    .add_local_dir(str(HERE), "/build/nve", copy=True)
    .run_commands(
        # CUDA_PATH is set by the nvidia/cuda base image.
        "bash -c 'source $HOME/.cargo/env && "
        "cd /build/nve && "
        "CUDA_PATH=/usr/local/cuda "
        "cargo build --release --features cuda 2>&1 | tail -5'",
        "cp /build/nve/target/release/nve /usr/local/bin/nve",
        "chmod +x /usr/local/bin/nve && nve --version",
    )
    .pip_install("huggingface_hub[hf_xfer]", "tqdm")
)

hf_secret = modal.Secret.from_dict({
    "HF_TOKEN": os.environ["HF_TOKEN"]
})

MODELS = {
    "llama3b": {
        "repo_id":   "meta-llama/Llama-3.2-3B",
        "local_dir": "/models/llama3b",
        "layers":    28,
        "label":     "Llama-3.2-3B",
        "bpw_sweep": False,   # ~6 hrs at 3B scale — skipped
        "layer_sweep": [],    # skipped for speed
    },
    "llama8b": {
        "repo_id":   "meta-llama/Llama-3.1-8B",
        "local_dir": "/models/llama8b",
        "layers":    32,
        "label":     "Llama-3.1-8B",
        "bpw_sweep": False,   # too slow at low bpw; just full ABC
        "layer_sweep": [],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Remote function: 32 GB for 8B baseline headroom
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=nve_image,
    gpu="A10G",            # 24 GB VRAM — GPU used for AWQ profiling matmuls
    cpu=4.0,
    memory=32768,          # 32 GB RAM — 8B bf16 ≈ 16 GB + overhead
    timeout=2 * 3600,      # 2 h — GPU makes profiling + quantization much faster
    volumes={"/models": model_vol},
    secrets=[hf_secret],
)
def run_model_experiments(model_key: str):
    import subprocess, time, json
    from pathlib import Path
    from huggingface_hub import snapshot_download

    cfg     = MODELS[model_key]
    hf_tok  = os.environ["HF_TOKEN"]
    outdir  = Path("/results"); outdir.mkdir(exist_ok=True)

    # ── Download model (cached across runs via Modal volume) ──────────────────
    marker = Path(cfg["local_dir"]) / ".downloaded"
    if marker.exists():
        print(f"[cache] {cfg['label']} already in volume")
    else:
        print(f"[download] {cfg['repo_id']} ...")
        t0 = time.time()
        snapshot_download(
            repo_id=cfg["repo_id"],
            local_dir=cfg["local_dir"],
            token=hf_tok,
            ignore_patterns=["*.bin", "original/*"],
        )
        marker.touch()
        model_vol.commit()
        print(f"  done in {time.time()-t0:.0f}s")

    results = {}

    def probe_model(mdir):
        """Quick sanity check: can NVE load the model config?"""
        r = subprocess.run(
            ["nve", "abc-test", "-m", mdir, "--configs", "baseline",
             "-n", "1", "--auto-budget"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print(f"  [probe FAILED exit={r.returncode}]")
            print("  STDOUT:", r.stdout[-500:] if r.stdout else "(empty)")
            print("  STDERR:", r.stderr[-500:] if r.stderr else "(empty)")
            return False
        print(f"  [probe OK]")
        return True

    def run(label, args, out_file):
        out_path = outdir / out_file
        cmd = ["nve", "abc-test"] + args + ["-o", str(out_path)]
        print(f"\n{'='*60}\n[{label}]\n{'='*60}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"  ERROR exit={proc.returncode} after {elapsed:.0f}s")
            return None
        if out_path.exists():
            with open(out_path) as f:
                data = json.load(f)
            for c in data.get("configurations", []):
                acc = c.get("task_accuracy", "?")
                tps = c.get("summary", {}).get("avg_tokens_per_sec", "?")
                acc_s = f"{acc:.0%}" if isinstance(acc, float) else str(acc)
                tps_s = f"{tps:.1f}" if isinstance(tps, float) else str(tps)
                print(f"  {c['config']:25s}  acc={acc_s}  {tps_s} tok/s")
            print(f"  → {out_file}  ({elapsed:.0f}s elapsed)")
            return data
        return None

    mdir = cfg["local_dir"]

    # ── 1. Full ABC ──────────────────────────────────────────────────────────
    print(f"\n\n>>> {cfg['label']} Full ABC (all 4 configs)")
    print(f"  Probing model load...")
    if not probe_model(mdir):
        print(f"  SKIP: {cfg['label']} probe failed — model may be unsupported")
        return {k: json.dumps(v) for k, v in results.items()}
    key = f"{model_key}_abc_clean"
    d = run(key, ["-m", mdir, "--auto-budget", "--target-bpw", "2.0", "-n", "40"], f"{key}.json")
    if d: results[key] = d

    # ── 2. BPW sweep (Config C only, optional per model) ─────────────────────
    if cfg["bpw_sweep"]:
        print(f"\n\n>>> {cfg['label']} BPW sweep (Config C, 0.5→4.0)")
        for bpw in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            k = f"{model_key}_bpw_{str(bpw).replace('.','_')}_clean"
            d = run(k, ["-m", mdir, "--auto-budget", "--target-bpw", str(bpw),
                        "--configs", "c", "-n", "40"], f"{k}.json")
            if d: results[k] = d

    # ── 3. Layer sweep (Config B, optional) ──────────────────────────────────
    if cfg["layer_sweep"]:
        print(f"\n\n>>> {cfg['label']} Layer sweep (Config B, N=active layers)")
        total = cfg["layers"]
        for n in cfg["layer_sweep"]:
            k = f"{model_key}_layers_{n}_clean"
            d = run(k, ["-m", mdir, "--auto-budget", "--target-bpw", "2.0",
                        "--configs", "b", "--active-layers", str(n),
                        "-n", "40"], f"{k}.json")
            if d: results[k] = d

    return {k: json.dumps(v) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Local entry point
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "both"):
    """
    --model  llama3b | llama8b | both  (default: both)
    """
    to_run = list(MODELS.keys()) if model == "both" else [model]
    print(f"NVE 3B+8B Benchmark  |  models: {to_run}")
    print("32 GB container, 4 CPU, 8 h timeout\n")

    out_dir = HERE / "evidence" / "experiments"
    out_dir.mkdir(exist_ok=True)

    all_saved = []
    handles = {m: run_model_experiments.spawn(m) for m in to_run}
    for mkey, handle in handles.items():
        print(f"\n[waiting for {mkey}...]")
        results_serialized = handle.get()
        for name, json_str in results_serialized.items():
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json_str)
            all_saved.append((name, json.loads(json_str)))
            print(f"  Saved → {out_path.name}")

    # Print summary table
    print(f"\n{'='*82}")
    print(f"{'Experiment':<38} {'Config':<26} {'Acc':>6} {'Tok/s':>8}")
    print("-" * 82)
    for name, data in all_saved:
        for c in data.get("configurations", []):
            acc = c.get("task_accuracy")
            tps = c.get("summary", {}).get("avg_tokens_per_sec")
            acc_s = f"{acc:.0%}" if isinstance(acc, float) else "?"
            tps_s = f"{tps:.1f}" if isinstance(tps, float) else "?"
            print(f"  {name:<36} {c['config']:<26} {acc_s:>6} {tps_s:>8}")

    print(f"\n{len(all_saved)} result files written to evidence/experiments/")
    print("Regenerate figures: python3 evidence/visualize_paper.py")
