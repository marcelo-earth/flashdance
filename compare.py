"""Comprehensive comparison of all attention variants.

Runs a full benchmark across all implemented mechanisms:
- Vanilla MHA
- SDPA (Flash Attention / mem-efficient)
- GQA (n_kv = n_heads // 2 and n_heads // 4)
- MQA (n_kv = 1)
- Sliding Window Attention
- ALiBi Attention

Outputs a formatted table and optionally saves plots.
"""

import argparse
import gc
import math
import time
import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention
from gqa import grouped_query_attention, repeat_kv
from sliding_window import sliding_window_attention
from alibi import build_alibi_bias, alibi_attention
from rope import precompute_freqs, apply_rope


def _sync(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def timed(fn, device, warmup=3, repeats=8):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "median_ms": times[len(times) // 2] * 1000,
        "min_ms": times[0] * 1000,
        "p95_ms": times[int(len(times) * 0.95)] * 1000,
    }


def run_comparison(
    seq_lengths=None,
    batch_size=2,
    n_heads=8,
    head_dim=64,
    window_size=128,
    device=None,
    repeats=8,
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

    print(f"\n{'='*70}")
    print(f"Attention Variant Comparison")
    print(f"Device: {device} | batch={batch_size} | heads={n_heads} | head_dim={head_dim}")
    print(f"{'='*70}\n")

    all_results = []

    for seq_len in seq_lengths:
        print(f"seq_len = {seq_len}")

        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        # KV heads for GQA/MQA
        k2 = torch.randn(batch_size, n_heads // 2, seq_len, head_dim, device=device)
        v2 = torch.randn(batch_size, n_heads // 2, seq_len, head_dim, device=device)
        k4 = torch.randn(batch_size, n_heads // 4, seq_len, head_dim, device=device)
        v4 = torch.randn(batch_size, n_heads // 4, seq_len, head_dim, device=device)
        k1 = torch.randn(batch_size, 1, seq_len, head_dim, device=device)
        v1 = torch.randn(batch_size, 1, seq_len, head_dim, device=device)

        # RoPE
        cos, sin = precompute_freqs(head_dim, seq_len, device=device)
        q_rope, k_rope = apply_rope(q, k, cos, sin)

        # ALiBi bias
        alibi_bias = build_alibi_bias(n_heads, seq_len, device=device)

        variants = {
            "Vanilla MHA": lambda: vanilla_attention(q, k, v),
            "SDPA (flash)": lambda: sdpa_attention(q, k, v),
            "SDPA + RoPE": lambda: sdpa_attention(q_rope, k_rope, v),
            f"GQA (n_kv={n_heads//2})": lambda: grouped_query_attention(
                q, k2, v2, n_heads=n_heads, n_kv_heads=n_heads//2
            ),
            f"GQA (n_kv={n_heads//4})": lambda: grouped_query_attention(
                q, k4, v4, n_heads=n_heads, n_kv_heads=n_heads//4
            ),
            "MQA (n_kv=1)": lambda: grouped_query_attention(
                q, k1, v1, n_heads=n_heads, n_kv_heads=1
            ),
            f"SWA (W={window_size})": lambda: sliding_window_attention(
                q, k, v, window_size=window_size
            ),
            "ALiBi": lambda: alibi_attention(q, k, v, alibi_bias=alibi_bias, causal=False),
        }

        row = {"seq_len": seq_len}
        baseline_ms = None

        for name, fn in variants.items():
            try:
                stats = timed(fn, device, repeats=repeats)
                ms = stats["median_ms"]
                if baseline_ms is None:
                    baseline_ms = ms
                speedup = baseline_ms / ms
                row[name] = {"ms": ms, "speedup": speedup}
                prefix = f"  {name:<25}"
                print(f"{prefix}: {ms:8.2f}ms  ({speedup:.2f}x vs Vanilla)")
            except Exception as e:
                row[name] = {"ms": float("inf"), "speedup": 0}
                print(f"  {name:<25}: ERROR ({e})")

        all_results.append(row)

        del q, k, v, k2, v2, k4, v4, k1, v1, q_rope, k_rope, alibi_bias
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    return all_results


def print_summary_table(results):
    """Print a GitHub-friendly markdown table."""
    if not results:
        return

    variant_names = [k for k in results[0].keys() if k != "seq_len"]
    seq_lens = [r["seq_len"] for r in results]

    print("\n## Summary: Median latency (ms)\n")
    header = "| Variant | " + " | ".join(str(s) for s in seq_lens) + " |"
    sep = "|" + "-|" * (len(seq_lens) + 1)
    print(header)
    print(sep)

    for name in variant_names:
        cells = []
        for r in results:
            d = r.get(name, {})
            ms = d.get("ms", float("inf"))
            cells.append(f"{ms:.2f}" if ms != float("inf") else "OOM")
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## Summary: Speedup vs Vanilla MHA\n")
    print(header)
    print(sep)

    for name in variant_names:
        cells = []
        for r in results:
            d = r.get(name, {})
            sp = d.get("speedup", 0)
            cells.append(f"{sp:.2f}x")
        print(f"| {name} | " + " | ".join(cells) + " |")


def plot_comparison(results, save_path="plots/comparison.png"):
    """Plot latency vs sequence length for all variants."""
    try:
        import matplotlib.pyplot as plt
        import os
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    os.makedirs("plots", exist_ok=True)
    variant_names = [k for k in results[0].keys() if k != "seq_len"]
    seq_lens = [r["seq_len"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # latency
    for name in variant_names:
        ms_vals = [r.get(name, {}).get("ms", None) for r in results]
        valid = [(s, m) for s, m in zip(seq_lens, ms_vals) if m is not None and m != float("inf")]
        if valid:
            xs, ys = zip(*valid)
            axes[0].plot(xs, ys, marker="o", label=name)
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency vs Sequence Length")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # speedup
    for name in variant_names:
        sp_vals = [r.get(name, {}).get("speedup", None) for r in results]
        valid = [(s, sp) for s, sp in zip(seq_lens, sp_vals) if sp is not None]
        if valid:
            xs, ys = zip(*valid)
            axes[1].plot(xs, ys, marker="o", label=name)
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Vanilla baseline")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Speedup vs Vanilla MHA")
    axes[1].set_title("Speedup vs Sequence Length")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Attention Variant Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all attention variants")
    parser.add_argument("--seq-len", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    args = parser.parse_args()

    results = run_comparison(
        seq_lengths=args.seq_len,
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        window_size=args.window_size,
        device=args.device,
        repeats=args.repeats,
    )
    print_summary_table(results)

    if args.plot:
        plot_comparison(results)
