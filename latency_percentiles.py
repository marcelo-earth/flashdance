"""Latency percentile analysis for attention operations.

Mean latency hides outliers. In production:
- p50 (median): typical request latency
- p95: 1 in 20 requests is slower than this
- p99: 1 in 100 requests is slower than this
- p999: rare worst-case latency

For LLM serving, tail latency matters because users notice slow responses.
Flash Attention helps both mean AND tail latency since it's more predictable.
"""

import math
import time
import gc
import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention


def measure_latency_distribution(
    fn,
    *args,
    device: str,
    warmup: int = 10,
    repeats: int = 200,
):
    """Measure detailed latency distribution (p50, p95, p99, p999)."""
    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        fn(*args)
    _sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        _sync()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)

    def percentile(p):
        idx = int(p / 100 * n)
        return times[min(idx, n - 1)]

    mean = sum(times) / n
    std = (sum((t - mean) ** 2 for t in times) / n) ** 0.5

    return {
        "mean": mean,
        "std": std,
        "min": times[0],
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "p999": percentile(99.9) if n >= 100 else percentile(99),
        "max": times[-1],
        "n": n,
    }


def benchmark_latency_percentiles(
    seq_lengths=None,
    batch_size: int = 1,
    n_heads: int = 8,
    head_dim: int = 64,
    device=None,
    repeats: int = 200,
):
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\nLatency Percentile Analysis")
    print(f"Device: {device} | batch={batch_size} | heads={n_heads} | head_dim={head_dim}")
    print(f"({repeats} samples per measurement)\n")

    all_results = []

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        van = measure_latency_distribution(vanilla_attention, q, k, v, device=device, repeats=repeats)
        sdpa = measure_latency_distribution(sdpa_attention, q, k, v, device=device, repeats=repeats)

        print(f"seq_len={seq_len}")
        print(f"  {'':12} {'mean':>8} {'std':>7} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
        print(f"  {'Vanilla':12} {van['mean']:>8.2f} {van['std']:>7.2f} {van['p50']:>8.2f} {van['p95']:>8.2f} {van['p99']:>8.2f} {van['max']:>8.2f} ms")
        print(f"  {'SDPA':12} {sdpa['mean']:>8.2f} {sdpa['std']:>7.2f} {sdpa['p50']:>8.2f} {sdpa['p95']:>8.2f} {sdpa['p99']:>8.2f} {sdpa['max']:>8.2f} ms")
        print(f"  Speedup (p50):  {van['p50']/sdpa['p50']:.2f}x  |  Speedup (p99):  {van['p99']/sdpa['p99']:.2f}x")
        print()

        all_results.append({
            "seq_len": seq_len,
            "vanilla": van,
            "sdpa": sdpa,
        })

        del q, k, v
        gc.collect()

    return all_results


def plot_latency_distribution(results, save_path="plots/latency_dist.png"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    percentiles = ["p50", "p75", "p90", "p95", "p99"]

    for ax, r in zip(axes, results):
        x = list(range(len(percentiles)))
        van_vals = [r["vanilla"][p] for p in percentiles]
        sdpa_vals = [r["sdpa"][p] for p in percentiles]

        ax.plot(x, van_vals, "o-", color="coral", label="Vanilla")
        ax.plot(x, sdpa_vals, "o-", color="steelblue", label="SDPA")
        ax.set_xticks(x)
        ax.set_xticklabels(percentiles)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"seq_len={r['seq_len']}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Latency Percentiles: Vanilla vs SDPA", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = benchmark_latency_percentiles()
    plot_latency_distribution(results)
