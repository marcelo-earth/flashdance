"""Throughput analysis: tokens/second and FLOP utilization.

Latency tells you how long one operation takes.
Throughput tells you how much work you can do per second.
For LLM serving, throughput = tokens generated per second per GPU.

FLOP counting:
- Attention QK^T: 2 * B * H * T * T * D (matmul)
- Attention weights @ V: 2 * B * H * T * T * D
- Total attention FLOPs: ~4 * B * H * T^2 * D = ~4 * B * T^2 * dim

MFU (Model FLOP Utilization) = achieved_flops / peak_hardware_flops
A well-optimized kernel should get 30-70% MFU on modern hardware.
"""

import math
import time
import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention


def attention_flops(batch: int, n_heads: int, seq_len: int, head_dim: int) -> int:
    """Approximate FLOPs for a single attention forward pass.

    Counts:
    - QK^T matmul: 2 * B * H * T * T * D
    - softmax: ~5 * B * H * T * T (exp, sum, divide, multiply -- rough)
    - Weights @ V: 2 * B * H * T * T * D
    """
    matmul_flops = 2 * batch * n_heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch * n_heads * seq_len * seq_len
    return 2 * matmul_flops + softmax_flops


def measure_throughput(
    fn,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    device: str,
    warmup: int = 5,
    repeats: int = 20,
):
    """Measure tokens/sec and TFLOP/s for an attention function."""
    q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device=device)

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # warmup
    for _ in range(warmup):
        fn(q, k, v)
    _sync()

    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(q, k, v)
    _sync()
    elapsed = time.perf_counter() - t0

    total_tokens = batch * seq_len * repeats
    tokens_per_sec = total_tokens / elapsed

    flops_per_call = attention_flops(batch, n_heads, seq_len, head_dim)
    total_flops = flops_per_call * repeats
    tflops_per_sec = total_flops / elapsed / 1e12

    return {
        "tokens_per_sec": tokens_per_sec,
        "tflops_per_sec": tflops_per_sec,
        "ms_per_call": elapsed / repeats * 1000,
        "total_tokens": total_tokens,
    }


def benchmark_throughput(
    seq_lengths=None,
    batch_sizes=None,
    n_heads: int = 8,
    head_dim: int = 64,
    device=None,
):
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dim = n_heads * head_dim
    print(f"\nThroughput Benchmark (tokens/sec)")
    print(f"Device: {device} | n_heads={n_heads} | head_dim={head_dim} | dim={dim}")
    print()

    results = []
    for batch in batch_sizes:
        print(f"Batch size = {batch}")
        print(f"  {'seq_len':>8} | {'Vanilla tok/s':>15} | {'SDPA tok/s':>13} | {'Speedup':>8} | {'SDPA TFLOP/s':>13}")
        print(f"  {'-'*8} | {'-'*15} | {'-'*13} | {'-'*8} | {'-'*13}")

        for seq_len in seq_lengths:
            try:
                v_stats = measure_throughput(vanilla_attention, batch, n_heads, seq_len, head_dim, device)
            except RuntimeError:
                v_stats = {"tokens_per_sec": 0, "tflops_per_sec": 0, "ms_per_call": float("inf")}

            s_stats = measure_throughput(sdpa_attention, batch, n_heads, seq_len, head_dim, device)

            speedup = v_stats["tokens_per_sec"] / s_stats["tokens_per_sec"] if v_stats["tokens_per_sec"] > 0 else 0
            speedup = s_stats["tokens_per_sec"] / max(v_stats["tokens_per_sec"], 1)

            v_tps = f"{v_stats['tokens_per_sec']:,.0f}" if v_stats["tokens_per_sec"] > 0 else "OOM"
            print(f"  {seq_len:>8} | {v_tps:>15} | {s_stats['tokens_per_sec']:>13,.0f} | {speedup:>7.2f}x | {s_stats['tflops_per_sec']:>12.3f}")

            results.append({
                "batch_size": batch,
                "seq_len": seq_len,
                "vanilla_tps": v_stats["tokens_per_sec"],
                "sdpa_tps": s_stats["tokens_per_sec"],
                "sdpa_tflops": s_stats["tflops_per_sec"],
                "speedup": speedup,
            })
        print()

    return results


def plot_throughput(results, save_path="plots/throughput.png"):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    batch_sizes = sorted(set(r["batch_size"] for r in results))
    seq_lens = sorted(set(r["seq_len"] for r in results))
    colors = ["steelblue", "coral", "seagreen", "purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for bs, color in zip(batch_sizes, colors):
        rows = [r for r in results if r["batch_size"] == bs]
        xs = [r["seq_len"] for r in rows]
        sdpa_tps = [r["sdpa_tps"] for r in rows]
        van_tps = [r["vanilla_tps"] for r in rows if r["vanilla_tps"] > 0]
        van_xs = [r["seq_len"] for r in rows if r["vanilla_tps"] > 0]

        axes[0].plot(xs, sdpa_tps, marker="o", color=color, label=f"SDPA bs={bs}")
        if van_tps:
            axes[0].plot(van_xs, van_tps, marker="x", linestyle="--", color=color, alpha=0.6, label=f"Vanilla bs={bs}")

    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_title("Throughput: tokens/sec")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # TFLOP/s
    for bs, color in zip(batch_sizes, colors):
        rows = [r for r in results if r["batch_size"] == bs]
        xs = [r["seq_len"] for r in rows]
        ys = [r["sdpa_tflops"] for r in rows]
        axes[1].plot(xs, ys, marker="o", color=color, label=f"SDPA bs={bs}")

    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("TFLOP/s")
    axes[1].set_title("Compute Throughput (TFLOP/s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Attention Throughput Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = benchmark_throughput()
    plot_throughput(results)
