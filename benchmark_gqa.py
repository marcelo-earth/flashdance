"""Benchmark GQA vs MHA across different n_kv_heads configurations.

Shows the memory/speed tradeoff of reducing KV heads.
"""

import gc
import time
import torch
import torch.nn.functional as F

from gqa import grouped_query_attention, repeat_kv
from attention import sdpa_attention


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_gqa_configs(
    seq_lengths=None,
    n_heads=8,
    head_dim=64,
    batch_size=4,
    device=None,
    repeats=10,
):
    """Benchmark MHA, GQA (various n_kv), and MQA."""
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    kv_configs = [n_heads, n_heads // 2, n_heads // 4, 1]
    kv_configs = [k for k in kv_configs if k >= 1 and n_heads % k == 0]
    kv_names = {
        n_heads: "MHA",
        n_heads // 2: f"GQA (n_kv={n_heads//2})",
        n_heads // 4: f"GQA (n_kv={n_heads//4})",
        1: "MQA (n_kv=1)",
    }

    print(f"\nGQA vs MHA Benchmark")
    print(f"Device: {device} | n_heads={n_heads} | head_dim={head_dim} | batch={batch_size}")

    all_results = []

    for seq_len in seq_lengths:
        print(f"\n  seq_len={seq_len}")
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        baseline_ms = None

        row = {"seq_len": seq_len}

        for n_kv in kv_configs:
            k = torch.randn(batch_size, n_kv, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, n_kv, seq_len, head_dim, device=device)

            # warmup
            for _ in range(3):
                grouped_query_attention(q, k, v, n_heads=n_heads, n_kv_heads=n_kv)
            _sync(device)

            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                grouped_query_attention(q, k, v, n_heads=n_heads, n_kv_heads=n_kv)
                _sync(device)
                times.append(time.perf_counter() - t0)

            times.sort()
            ms = times[len(times) // 2] * 1000  # median

            if baseline_ms is None:
                baseline_ms = ms

            speedup = baseline_ms / ms
            kv_mem_mb = 2 * n_kv * seq_len * head_dim * 4 / 1024**2  # float32
            name = kv_names.get(n_kv, f"n_kv={n_kv}")

            print(f"    {name:<22}: {ms:7.2f}ms  {speedup:.2f}x  KV cache: {kv_mem_mb:.1f}MB")
            row[name] = {"ms": ms, "speedup": speedup, "kv_mb": kv_mem_mb}

            del k, v
            gc.collect()

        del q
        gc.collect()
        all_results.append(row)

    return all_results


def plot_gqa_comparison(results, save_path="plots/gqa_comparison.png"):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    seq_lens = [r["seq_len"] for r in results]
    configs = [k for k in results[0].keys() if k != "seq_len"]
    colors = ["steelblue", "coral", "seagreen", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, color in zip(configs, colors):
        ms_vals = [r.get(name, {}).get("ms", None) for r in results]
        valid = [(s, m) for s, m in zip(seq_lens, ms_vals) if m is not None]
        if valid:
            xs, ys = zip(*valid)
            axes[0].plot(xs, ys, marker="o", color=color, label=name)

    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Latency (ms, median)")
    axes[0].set_title("GQA Variants: Latency")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for name, color in zip(configs, colors):
        sp_vals = [r.get(name, {}).get("speedup", None) for r in results]
        valid = [(s, sp) for s, sp in zip(seq_lens, sp_vals) if sp is not None]
        if valid:
            xs, ys = zip(*valid)
            axes[1].plot(xs, ys, marker="o", color=color, label=name)

    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Speedup vs MHA")
    axes[1].set_title("GQA Variants: Speedup vs MHA")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Grouped Query Attention: Performance vs Memory Tradeoff", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = benchmark_gqa_configs()
    plot_gqa_comparison(results)
