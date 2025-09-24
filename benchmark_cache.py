"""Benchmark result caching and comparison across runs.

Saves benchmark results to disk and loads them for comparison.
Useful for tracking performance regressions across PyTorch versions or hardware.
"""

import json
import os
import platform
import time
import datetime
import torch


CACHE_DIR = "results"


def get_run_metadata() -> dict:
    """Collect metadata about the current run environment."""
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": None,
        "gpu": None,
    }

    if torch.cuda.is_available():
        meta["device"] = "cuda"
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda_version"] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        meta["gpu_memory_gb"] = props.total_memory / 1024**3
        meta["compute_capability"] = f"{props.major}.{props.minor}"
    elif torch.backends.mps.is_available():
        meta["device"] = "mps"
        meta["gpu"] = "Apple Silicon MPS"
    else:
        meta["device"] = "cpu"
        meta["gpu"] = "CPU"

    return meta


def save_benchmark(name: str, results: list | dict, metadata: dict = None) -> str:
    """Save benchmark results with metadata to JSON.

    Args:
        name: benchmark name (used as filename prefix)
        results: list or dict of results
        metadata: optional extra metadata

    Returns:
        Path to the saved file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    run_meta = get_run_metadata()
    if metadata:
        run_meta.update(metadata)

    data = {
        "metadata": run_meta,
        "results": results,
    }

    # clean up non-serializable values
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if obj == float("inf"):
            return "OOM"
        if obj == float("-inf"):
            return "-inf"
        return obj

    data = clean(data)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CACHE_DIR, f"{name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    # also save as "latest"
    latest_path = os.path.join(CACHE_DIR, f"{name}_latest.json")
    with open(latest_path, "w") as f:
        json.dump(data, f, indent=2)

    return path


def load_benchmark(name: str, use_latest: bool = True) -> dict | None:
    """Load a saved benchmark result.

    Args:
        name: benchmark name
        use_latest: if True, load the _latest file; otherwise most recent timestamped

    Returns:
        dict with metadata and results, or None if not found
    """
    if use_latest:
        path = os.path.join(CACHE_DIR, f"{name}_latest.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # find most recent timestamped file
    import glob
    pattern = os.path.join(CACHE_DIR, f"{name}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    files = [f for f in files if "_latest" not in f]

    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)


def compare_benchmarks(name: str, current_results: list, baseline_path: str = None) -> dict:
    """Compare current results against a saved baseline.

    Args:
        name: benchmark name
        current_results: list of current benchmark results
        baseline_path: optional path to specific baseline file

    Returns:
        dict with comparison statistics
    """
    if baseline_path:
        with open(baseline_path) as f:
            baseline = json.load(f)
    else:
        baseline = load_benchmark(name)

    if baseline is None:
        print(f"No baseline found for '{name}'. Saving current as baseline.")
        save_benchmark(name, current_results)
        return {"status": "no_baseline"}

    baseline_results = baseline["results"]
    comparison = []

    for curr, base in zip(current_results, baseline_results):
        if "seq_len" not in curr or "seq_len" not in base:
            continue
        if curr["seq_len"] != base["seq_len"]:
            continue

        row = {"seq_len": curr["seq_len"]}
        for key in ["sdpa_ms", "vanilla_ms", "speedup"]:
            if key in curr and key in base:
                c_val = curr[key]
                b_val = base[key]
                if isinstance(c_val, (int, float)) and isinstance(b_val, (int, float)):
                    delta = (c_val - b_val) / b_val * 100
                    row[key] = {"current": c_val, "baseline": b_val, "delta_pct": delta}

        comparison.append(row)

    print(f"\nBenchmark Comparison: {name}")
    print(f"Baseline: {baseline['metadata']['timestamp']} | {baseline['metadata']['gpu']}")
    print(f"Current:  {get_run_metadata()['timestamp']} | {get_run_metadata()['gpu']}")
    print()

    if comparison:
        print(f"{'seq_len':>8} | {'SDPA ms (curr)':>15} | {'SDPA ms (base)':>15} | {'delta':>8}")
        print("-" * 55)
        for row in comparison:
            sdpa = row.get("sdpa_ms", {})
            if sdpa:
                sign = "+" if sdpa["delta_pct"] > 0 else ""
                print(f"{row['seq_len']:>8} | {sdpa['current']:>15.2f} | {sdpa['baseline']:>15.2f} | {sign}{sdpa['delta_pct']:>7.1f}%")

    return {"comparison": comparison, "baseline_meta": baseline["metadata"]}


def list_saved_benchmarks() -> list:
    """List all saved benchmark files."""
    if not os.path.exists(CACHE_DIR):
        return []

    import glob
    files = glob.glob(os.path.join(CACHE_DIR, "*.json"))
    results = []
    for f in sorted(files):
        try:
            with open(f) as fp:
                data = json.load(fp)
            meta = data.get("metadata", {})
            results.append({
                "file": os.path.basename(f),
                "timestamp": meta.get("timestamp", "unknown"),
                "device": meta.get("device", "unknown"),
                "gpu": meta.get("gpu", "unknown"),
                "torch": meta.get("torch_version", "unknown"),
            })
        except Exception:
            pass

    return results


if __name__ == "__main__":
    # demo: run a quick benchmark and save it
    from benchmark import benchmark_attention

    print("Running benchmark...")
    results = benchmark_attention(seq_lengths=[256, 512, 1024], repeats=5)

    path = save_benchmark("forward_pass", results)
    print(f"Saved to {path}")

    # compare with baseline if it exists
    compare_benchmarks("forward_pass", results)

    saved = list_saved_benchmarks()
    print(f"\nSaved benchmarks ({len(saved)} files):")
    for s in saved:
        print(f"  {s['file']} | {s['timestamp']} | {s['gpu']}")
