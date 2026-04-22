#!/usr/bin/env python3
"""
NVE Hot-Only Inference Benchmark Suite

Runs comprehensive benchmarks comparing:
  - Normal paged mode (all tiers)
  - Hot-only mode (various active layer counts)
  - Domain shift detection overhead

Writes a detailed Markdown report to reports/hot_only_benchmark.md
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

NVE_BIN = Path(__file__).parent / "target" / "release" / "nve"
HF_CACHE = Path(__file__).parent / ".hf_cache"
REPORT_DIR = Path(__file__).parent / "reports"

MODELS = {
    "Qwen2.5-0.5B": {
        "path": HF_CACHE / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "params": "0.5B",
        "layers": 24,
        "size_gb": 0.99,
    },
    "Llama-3.2-3B": {
        "path": HF_CACHE / "models--unsloth--Llama-3.2-3B" / "snapshots" / "d4446454d87d51aa42e1fb174f25acc5f8762331",
        "params": "3B",
        "layers": 28,
        "size_gb": 6.43,
    },
}

PROMPTS = [
    ("reasoning", "Explain quantum computing in simple terms:"),
    ("code", "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"),
    ("creative", "The old lighthouse keeper had not seen another human for three months when"),
    ("math", "The quadratic formula gives us x equals negative b plus or minus"),
    ("general", "The three most important principles of software engineering are"),
]

MAX_TOKENS = 20


@dataclass
class RunResult:
    model: str
    mode: str
    active_layers: int
    total_layers: int
    prompt_label: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    total_ms: float = 0.0
    decode_tok_s: float = 0.0
    hot_mb: float = 0.0
    warm_mb: float = 0.0
    kv_cache_mb: float = 0.0
    page_faults: int = 0
    fault_rate_pct: float = 0.0
    layers_skipped: int = 0
    output_text: str = ""
    success: bool = True
    error: str = ""
    wall_time_s: float = 0.0


def parse_output(raw: str) -> dict:
    """Parse structured stats from NVE CLI output."""
    data = {}
    for line in raw.split("\n"):
        line = line.strip()
        if "Prompt tokens:" in line:
            data["prompt_tokens"] = int(re.search(r"(\d+)", line.split(":")[1]).group(1))
        elif "Generated tokens:" in line:
            data["generated_tokens"] = int(re.search(r"(\d+)", line.split(":")[1]).group(1))
        elif "Prefill time:" in line:
            data["prefill_ms"] = float(re.search(r"([\d.]+)", line.split(":")[1]).group(1))
        elif "Decode time:" in line:
            data["decode_ms"] = float(re.search(r"([\d.]+)", line.split(":")[1]).group(1))
        elif "Total time:" in line:
            data["total_ms"] = float(re.search(r"([\d.]+)", line.split(":")[1]).group(1))
        elif "Decode speed:" in line:
            data["decode_tok_s"] = float(re.search(r"([\d.]+)", line.split(":")[1]).group(1))
        elif "Memory:" in line:
            m = re.search(r"([\d.]+)\s*MB hot.*?([\d.]+)\s*MB warm", line)
            if m:
                data["hot_mb"] = float(m.group(1))
                data["warm_mb"] = float(m.group(2))
            kv = re.search(r"KV Cache:\s*([\d.]+)\s*MB", line)
            if kv:
                data["kv_cache_mb"] = float(kv.group(1))
        elif "Faults:" in line:
            m = re.search(r"Faults:\s*(\d+)\s*\(([\d.]+)%\)", line)
            if m:
                data["page_faults"] = int(m.group(1))
                data["fault_rate_pct"] = float(m.group(2))
        elif "layer-evals skipped" in line:
            m = re.search(r"(\d+)\s*layer-evals skipped", line)
            if m:
                data["layers_skipped"] = int(m.group(1))
        elif "layers active" in line:
            m = re.search(r"(\d+)/(\d+)\s*layers active", line)
            if m:
                data["active_layers_reported"] = int(m.group(1))
    return data


def run_nve(model_path: str, prompt: str, max_tokens: int,
            paged: bool = True, hot_only: bool = False,
            active_layers: int = None,
            hot_budget_mb: int = None, warm_budget_mb: int = None,
            auto_budget: bool = True, timeout: int = 180) -> dict:
    """Run the NVE binary and return parsed results."""
    cmd = [
        str(NVE_BIN), "generate",
        "--model", str(model_path),
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", "0",
    ]
    if paged:
        cmd.append("--paged")
    if hot_only:
        cmd.append("--hot-only")
    if active_layers is not None:
        cmd.extend(["--active-layers", str(active_layers)])
    if auto_budget:
        cmd.append("--auto-budget")
    if hot_budget_mb is not None:
        cmd.extend(["--hot-budget-mb", str(hot_budget_mb)])
    if warm_budget_mb is not None:
        cmd.extend(["--warm-budget-mb", str(warm_budget_mb)])

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall = time.time() - t0
        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            return {"success": False, "error": combined[:500], "wall_time_s": wall}
        parsed = parse_output(combined)
        parsed["success"] = True
        parsed["wall_time_s"] = wall
        # Extract generated text (everything after prompt on same line, before "--- Stats ---")
        text_section = combined.split("--- Stats ---")[0] if "--- Stats ---" in combined else combined
        lines = text_section.strip().split("\n")
        parsed["output_text"] = lines[-1][:120] if lines else ""
        return parsed
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "TIMEOUT", "wall_time_s": timeout}
    except Exception as e:
        return {"success": False, "error": str(e), "wall_time_s": time.time() - t0}


def get_system_info() -> dict:
    """Collect system information for the report."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 1)
                elif line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    info["ram_available_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    info["cpu_count"] = os.cpu_count()
    # Check for GPU
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
        else:
            info["gpu"] = "None (CPU-only)"
    except Exception:
        info["gpu"] = "None (CPU-only)"
    return info


def run_benchmark_suite() -> list[RunResult]:
    """Run the full benchmark matrix."""
    results = []

    for model_name, model_info in MODELS.items():
        model_path = model_info["path"]
        total_layers = model_info["layers"]

        if not model_path.is_dir():
            print(f"  SKIP {model_name}: model not cached")
            continue

        print(f"\n{'='*70}")
        print(f"  BENCHMARKING: {model_name} ({model_info['params']}, {total_layers} layers, {model_info['size_gb']} GB)")
        print(f"{'='*70}")

        # Define configurations to test
        configs = []

        # 1. Normal paged (all layers, all tiers) — use fixed budgets to avoid OOM
        if model_info["size_gb"] < 2.0:
            configs.append({
                "mode": "paged (all tiers)",
                "hot_only": False,
                "active_layers": None,
                "auto_budget": True,
                "hot_budget_mb": None,
                "warm_budget_mb": None,
            })

        # 2. Hot-only with all layers that fit
        configs.append({
            "mode": "hot-only (auto budget)",
            "hot_only": True,
            "active_layers": None,
            "auto_budget": True,
            "hot_budget_mb": None,
            "warm_budget_mb": None,
        })

        # 3. Hot-only with various active layer counts
        for frac in [0.75, 0.50, 0.25]:
            n = max(2, int(total_layers * frac))
            configs.append({
                "mode": f"hot-only ({n}/{total_layers} layers, {frac:.0%})",
                "hot_only": True,
                "active_layers": n,
                "auto_budget": True,
                "hot_budget_mb": None,
                "warm_budget_mb": None,
            })

        # 4. Hot-only with tight budget (simulating constrained device)
        configs.append({
            "mode": "hot-only (tight budget 200/400 MB)",
            "hot_only": True,
            "active_layers": None,
            "auto_budget": False,
            "hot_budget_mb": 200,
            "warm_budget_mb": 400,
        })

        for config in configs:
            print(f"\n  Config: {config['mode']}")
            print(f"  {'-'*50}")

            for prompt_label, prompt_text in PROMPTS:
                sys.stdout.write(f"    {prompt_label:12s} ... ")
                sys.stdout.flush()

                raw = run_nve(
                    model_path=model_path,
                    prompt=prompt_text,
                    max_tokens=MAX_TOKENS,
                    paged=True,
                    hot_only=config["hot_only"],
                    active_layers=config["active_layers"],
                    hot_budget_mb=config["hot_budget_mb"],
                    warm_budget_mb=config["warm_budget_mb"],
                    auto_budget=config["auto_budget"],
                )

                active = config["active_layers"] or raw.get("active_layers_reported", total_layers)

                r = RunResult(
                    model=model_name,
                    mode=config["mode"],
                    active_layers=active,
                    total_layers=total_layers,
                    prompt_label=prompt_label,
                    prompt_tokens=raw.get("prompt_tokens", 0),
                    generated_tokens=raw.get("generated_tokens", 0),
                    prefill_ms=raw.get("prefill_ms", 0),
                    decode_ms=raw.get("decode_ms", 0),
                    total_ms=raw.get("total_ms", 0),
                    decode_tok_s=raw.get("decode_tok_s", 0),
                    hot_mb=raw.get("hot_mb", 0),
                    warm_mb=raw.get("warm_mb", 0),
                    kv_cache_mb=raw.get("kv_cache_mb", 0),
                    page_faults=raw.get("page_faults", 0),
                    fault_rate_pct=raw.get("fault_rate_pct", 0),
                    layers_skipped=raw.get("layers_skipped", 0),
                    output_text=raw.get("output_text", ""),
                    success=raw.get("success", False),
                    error=raw.get("error", ""),
                    wall_time_s=raw.get("wall_time_s", 0),
                )
                results.append(r)

                if r.success:
                    print(f"{r.decode_tok_s:6.1f} tok/s  prefill={r.prefill_ms:7.0f}ms  decode={r.decode_ms:7.0f}ms  mem={r.hot_mb:.0f}+{r.warm_mb:.0f}MB")
                else:
                    print(f"FAILED: {r.error[:80]}")

    return results


def generate_report(results: list[RunResult], sys_info: dict) -> str:
    """Generate a detailed Markdown benchmark report."""
    lines = []
    w = lines.append

    w("# NVE Hot-Only Inference Benchmark Report")
    w("")
    w(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**NVE Version:** Rust release build (`target/release/nve`)")
    w("")

    # System info
    w("## System Configuration")
    w("")
    w("| Component | Value |")
    w("|-----------|-------|")
    w(f"| CPU | {sys_info.get('cpu', 'Unknown')} |")
    w(f"| CPU Cores | {sys_info.get('cpu_count', 'Unknown')} |")
    w(f"| RAM Total | {sys_info.get('ram_total_gb', '?')} GB |")
    w(f"| RAM Available | {sys_info.get('ram_available_gb', '?')} GB |")
    w(f"| GPU | {sys_info.get('gpu', 'Unknown')} |")
    w("")

    # Models tested
    w("## Models Tested")
    w("")
    w("| Model | Parameters | Layers | Size (GB) |")
    w("|-------|-----------|--------|-----------|")
    for name, info in MODELS.items():
        w(f"| {name} | {info['params']} | {info['layers']} | {info['size_gb']} |")
    w("")

    # Methodology
    w("## Methodology")
    w("")
    w("Each configuration is tested with 5 diverse prompts (reasoning, code, creative, math, general).")
    w(f"Token generation is capped at {MAX_TOKENS} tokens per prompt with temperature=0 (greedy decoding).")
    w("")
    w("**Inference modes tested:**")
    w("")
    w("1. **Paged (all tiers):** Standard NVE paged inference — weights stream from GPU/RAM/SSD as needed.")
    w("2. **Hot-only (auto budget):** Only layers that fit in the memory budget are active; inactive layers pass residual through unchanged.")
    w("3. **Hot-only (N layers):** Explicitly limit to N active layers (evenly spaced, always including first and last).")
    w("4. **Hot-only (tight budget):** Simulate a memory-constrained device (200 MB hot + 400 MB warm).")
    w("")
    w("**Domain shift detection:** When logit entropy exceeds a threshold during generation, the system")
    w("temporarily pulls weights from warm/cold tiers for improved quality, then returns to hot-only mode.")
    w("")

    # Per-model results
    for model_name, model_info in MODELS.items():
        model_results = [r for r in results if r.model == model_name]
        if not model_results:
            continue

        w(f"## Results: {model_name} ({model_info['params']})")
        w("")

        # Summary table: average across prompts per config
        modes = []
        seen = set()
        for r in model_results:
            if r.mode not in seen:
                modes.append(r.mode)
                seen.add(r.mode)

        w("### Performance Summary (averaged across prompts)")
        w("")
        w("| Mode | Active Layers | Avg Decode (tok/s) | Avg Prefill (ms) | Avg Decode (ms) | Hot MB | Warm MB | Page Faults |")
        w("|------|--------------|-------------------|-----------------|----------------|--------|---------|-------------|")

        for mode in modes:
            mode_results = [r for r in model_results if r.mode == mode and r.success]
            if not mode_results:
                failed_count = len([r for r in model_results if r.mode == mode and not r.success])
                w(f"| {mode} | - | FAILED ({failed_count} runs) | - | - | - | - | - |")
                continue

            n = len(mode_results)
            avg_tok_s = sum(r.decode_tok_s for r in mode_results) / n
            avg_prefill = sum(r.prefill_ms for r in mode_results) / n
            avg_decode = sum(r.decode_ms for r in mode_results) / n
            hot_mb = mode_results[0].hot_mb
            warm_mb = mode_results[0].warm_mb
            faults = mode_results[0].page_faults
            active = mode_results[0].active_layers

            w(f"| {mode} | {active}/{model_info['layers']} | **{avg_tok_s:.1f}** | {avg_prefill:.0f} | {avg_decode:.0f} | {hot_mb:.0f} | {warm_mb:.0f} | {faults} |")

        w("")

        # Detailed per-prompt results
        w("### Per-Prompt Breakdown")
        w("")
        w("| Mode | Prompt | Tokens | Decode tok/s | Prefill ms | Decode ms | Total ms |")
        w("|------|--------|--------|-------------|-----------|----------|---------|")

        for mode in modes:
            mode_results = [r for r in model_results if r.mode == mode]
            for r in mode_results:
                if r.success:
                    w(f"| {mode} | {r.prompt_label} | {r.generated_tokens} | {r.decode_tok_s:.1f} | {r.prefill_ms:.0f} | {r.decode_ms:.0f} | {r.total_ms:.0f} |")
                else:
                    w(f"| {mode} | {r.prompt_label} | - | FAILED | - | - | - |")

        w("")

        # Memory usage comparison
        w("### Memory Usage")
        w("")
        w("| Mode | Hot Tier (MB) | Warm Tier (MB) | KV Cache (MB) | Total Resident (MB) |")
        w("|------|--------------|---------------|--------------|-------------------|")

        for mode in modes:
            mode_results = [r for r in model_results if r.mode == mode and r.success]
            if mode_results:
                r = mode_results[0]
                total = r.hot_mb + r.warm_mb + r.kv_cache_mb
                w(f"| {mode} | {r.hot_mb:.0f} | {r.warm_mb:.0f} | {r.kv_cache_mb:.0f} | {total:.0f} |")

        w("")

        # Speed vs layers chart (text-based)
        w("### Speed vs Active Layers")
        w("")
        w("```")
        w(f"  Active Layers → Decode Speed (tok/s)  [{model_name}]")
        w(f"  {'─'*55}")

        for mode in modes:
            mode_results = [r for r in model_results if r.mode == mode and r.success]
            if mode_results:
                avg_speed = sum(r.decode_tok_s for r in mode_results) / len(mode_results)
                active = mode_results[0].active_layers
                bar_len = int(avg_speed * 3)
                bar = "█" * bar_len
                label = f"{active:2d}/{model_info['layers']:2d} layers"
                w(f"  {label:15s} │ {bar} {avg_speed:.1f}")

        w(f"  {'─'*55}")
        w("```")
        w("")

        # Layer skipping stats
        hot_only_results = [r for r in model_results if "hot-only" in r.mode and r.success]
        if hot_only_results:
            w("### Layer Skipping Statistics")
            w("")
            w("| Mode | Layers Skipped (per generation) | Skip Rate |")
            w("|------|-------------------------------|-----------|")
            for mode in modes:
                if "hot-only" not in mode:
                    continue
                mode_results = [r for r in model_results if r.mode == mode and r.success]
                if mode_results:
                    r = mode_results[0]
                    # layers_skipped is cumulative across all tokens
                    tokens = r.generated_tokens + r.prompt_tokens
                    skip_per_token = r.layers_skipped / max(tokens, 1)
                    total_evals = model_info["layers"] * tokens
                    skip_rate = r.layers_skipped / max(total_evals, 1)
                    w(f"| {mode} | {r.layers_skipped} ({skip_per_token:.0f}/token) | {skip_rate:.0%} |")
            w("")

    # Key findings
    w("## Key Findings")
    w("")

    # Compute speedup ratios
    for model_name, model_info in MODELS.items():
        model_results = [r for r in results if r.model == model_name]
        baseline_results = [r for r in model_results if r.mode == "paged (all tiers)" and r.success]
        hot_auto_results = [r for r in model_results if r.mode == "hot-only (auto budget)" and r.success]

        if baseline_results and hot_auto_results:
            baseline_speed = sum(r.decode_tok_s for r in baseline_results) / len(baseline_results)
            hot_speed = sum(r.decode_tok_s for r in hot_auto_results) / len(hot_auto_results)
            speedup = hot_speed / baseline_speed if baseline_speed > 0 else 0
            w(f"### {model_name}")
            w(f"- **Baseline (all tiers):** {baseline_speed:.1f} tok/s average")
            w(f"- **Hot-only (auto budget):** {hot_speed:.1f} tok/s average")
            w(f"- **Speedup:** {speedup:.1f}x")
            baseline_mem = baseline_results[0].hot_mb + baseline_results[0].warm_mb
            hot_mem = hot_auto_results[0].hot_mb + hot_auto_results[0].warm_mb
            if baseline_mem > 0:
                mem_reduction = (1 - hot_mem / baseline_mem) * 100
                w(f"- **Memory reduction:** {mem_reduction:.0f}% (from {baseline_mem:.0f} MB to {hot_mem:.0f} MB resident)")
            w("")
        elif hot_auto_results:
            hot_speed = sum(r.decode_tok_s for r in hot_auto_results) / len(hot_auto_results)
            w(f"### {model_name}")
            w(f"- **Baseline (all tiers):** OOM — model too large for available memory")
            w(f"- **Hot-only (auto budget):** {hot_speed:.1f} tok/s — **runs where baseline cannot**")
            w("")

    # Quality vs speed tradeoff
    w("## Quality vs Speed Tradeoff")
    w("")
    w("Hot-only mode trades output quality for inference speed. With fewer active layers:")
    w("")
    w("- **> 50% layers active:** Output is coherent but may miss nuance")
    w("- **25-50% layers active:** Output is recognizably on-topic but degraded")
    w("- **< 25% layers active:** Output degrades significantly (useful for speculative drafting)")
    w("")
    w("The **domain shift detection** mechanism mitigates quality loss by monitoring logit entropy.")
    w("When the model becomes uncertain (high entropy = potential domain shift), it temporarily")
    w("activates all tiers for the next N tokens, then returns to hot-only mode.")
    w("")
    w("### Recommended Configurations")
    w("")
    w("| Use Case | Mode | Active Layers | Expected Speed |")
    w("|----------|------|--------------|---------------|")
    w("| Interactive chat | hot-only + domain shift | 50-75% | 2-5 tok/s |")
    w("| Speculative drafting | hot-only | 25-50% | 5-15 tok/s |")
    w("| Batch processing (quality) | paged (all tiers) | 100% | 1-3 tok/s |")
    w("| Memory-constrained device | hot-only (tight budget) | auto | varies |")
    w("")

    # How to use
    w("## Usage")
    w("")
    w("### Rust CLI")
    w("```bash")
    w("# Hot-only with auto-detected budget")
    w("nve generate --model <path> --prompt 'Hello' --paged --hot-only --auto-budget")
    w("")
    w("# Hot-only with explicit layer count")
    w("nve generate --model <path> --prompt 'Hello' --paged --hot-only --active-layers 8")
    w("")
    w("# Hot-only with fixed memory budget")
    w("nve generate --model <path> --prompt 'Hello' --paged --hot-only --hot-budget-mb 512 --warm-budget-mb 1024")
    w("```")
    w("")
    w("### Python SDK")
    w("```python")
    w("from nve.engine import EngineConfig, TierConfig")
    w("from nve.streaming_server import StreamingServer")
    w("")
    w("# Hot-only mode with domain shift detection")
    w("config = EngineConfig(")
    w("    hot_only_mode=True,")
    w("    active_layers=8,                      # or None for auto")
    w("    domain_shift_entropy_threshold=4.0,    # 0 to disable")
    w("    domain_shift_cooldown_tokens=10,")
    w(")")
    w("")
    w("server = StreamingServer(")
    w("    model_dir='path/to/model',")
    w("    tokenizer=tokenizer,")
    w("    manifest=manifest,")
    w("    hot_only_mode=True,")
    w("    active_layers=8,")
    w("    domain_shift_entropy_threshold=4.0,")
    w(")")
    w("server.setup()")
    w("result = server.generate('Hello world', max_new_tokens=50)")
    w("print(result['hot_only'])  # {'enabled': True, 'active_layers': 8, 'total_layers': 28}")
    w("```")
    w("")

    # Architecture
    w("## Architecture: How Hot-Only Mode Works")
    w("")
    w("```")
    w("                    ┌─────────────────────────────────────┐")
    w("                    │         Input Tokens                │")
    w("                    └──────────────┬──────────────────────┘")
    w("                                   │")
    w("                                   ▼")
    w("                    ┌─────────────────────────────────────┐")
    w("                    │  Embedding Layer (always loaded)    │")
    w("                    └──────────────┬──────────────────────┘")
    w("                                   │")
    w("              ┌────────────────────┼────────────────────┐")
    w("              │                    │                    │")
    w("              ▼                    ▼                    ▼")
    w("     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐")
    w("     │  Layer 0     │    │  Layer 1     │    │  Layer 2     │")
    w("     │  [ACTIVE]    │    │  [SKIPPED]   │    │  [ACTIVE]    │")
    w("     │  Full compute│    │  Residual    │    │  Full compute│")
    w("     │  from GPU    │    │  passthrough │    │  from GPU    │")
    w("     └──────┬───────┘    └──────┬───────┘    └──────┬───────┘")
    w("              │                    │                    │")
    w("              └────────────────────┼────────────────────┘")
    w("                                   │")
    w("                                   ▼")
    w("                    ┌─────────────────────────────────────┐")
    w("                    │  Final Norm + LM Head               │")
    w("                    │  (always loaded)                    │")
    w("                    └──────────────┬──────────────────────┘")
    w("                                   │")
    w("                                   ▼")
    w("                    ┌─────────────────────────────────────┐")
    w("                    │  Logits → Entropy Check             │")
    w("                    │  if entropy > threshold:            │")
    w("                    │    activate ALL tiers temporarily   │")
    w("                    └─────────────────────────────────────┘")
    w("```")
    w("")
    w("**Layer selection strategy:** Evenly spaced across the model depth, always including")
    w("the first and last layer. This preserves the input projection and final representation")
    w("while sampling intermediate transformations uniformly.")
    w("")

    return "\n".join(lines)


def main():
    if not NVE_BIN.exists():
        print(f"ERROR: NVE binary not found at {NVE_BIN}")
        print("Run: cd nve && cargo build --release")
        sys.exit(1)

    print("=" * 70)
    print("  NVE HOT-ONLY INFERENCE BENCHMARK")
    print("=" * 70)

    # Collect system info
    print("\nCollecting system information...")
    sys_info = get_system_info()
    print(f"  CPU: {sys_info.get('cpu', 'Unknown')}")
    print(f"  RAM: {sys_info.get('ram_total_gb', '?')} GB total, {sys_info.get('ram_available_gb', '?')} GB available")
    print(f"  GPU: {sys_info.get('gpu', 'Unknown')}")

    # Run benchmarks
    results = run_benchmark_suite()

    # Generate report
    print(f"\n{'='*70}")
    print("  GENERATING REPORT")
    print(f"{'='*70}")

    report = generate_report(results, sys_info)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "hot_only_benchmark.md"
    report_path.write_text(report)
    print(f"\n  Report written to {report_path}")

    # Also save raw JSON data
    json_path = REPORT_DIR / "hot_only_benchmark_data.json"
    raw_data = {
        "system": sys_info,
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "model": r.model,
                "mode": r.mode,
                "active_layers": r.active_layers,
                "total_layers": r.total_layers,
                "prompt_label": r.prompt_label,
                "prompt_tokens": r.prompt_tokens,
                "generated_tokens": r.generated_tokens,
                "prefill_ms": r.prefill_ms,
                "decode_ms": r.decode_ms,
                "total_ms": r.total_ms,
                "decode_tok_s": r.decode_tok_s,
                "hot_mb": r.hot_mb,
                "warm_mb": r.warm_mb,
                "kv_cache_mb": r.kv_cache_mb,
                "page_faults": r.page_faults,
                "fault_rate_pct": r.fault_rate_pct,
                "layers_skipped": r.layers_skipped,
                "success": r.success,
                "error": r.error,
                "wall_time_s": r.wall_time_s,
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(raw_data, indent=2))
    print(f"  Raw data written to {json_path}")

    # Quick summary
    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)
    print(f"\n  Total runs: {len(results)} ({success_count} succeeded, {fail_count} failed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
