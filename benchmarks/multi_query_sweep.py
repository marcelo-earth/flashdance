"""MQA / GQA inference speedup sweep.

At inference time (batch=1, generating one token):
- KV cache is the bottleneck, not compute
- Reducing n_kv_heads saves memory bandwidth
- MQA (n_kv=1) is fastest but may hurt quality
- GQA balances quality and speed

This script measures the actual inference-time speedup from GQA.
"""

import gc
import time
import torch

from kv_cache import KVCache, AttentionWithKVCache
from rope import precompute_freqs


def sweep_kv_heads_inference(
    dim: int = 512,
    n_heads: int = 8,
    n_kv_options=None,
    context_len: int = 512,
    n_decode_steps: int = 100,
    device=None,
):
    """Measure decode throughput for different n_kv_heads."""
    if n_kv_options is None:
        n_kv_options = [n_heads, n_heads // 2, n_heads // 4, 2, 1]
        n_kv_options = [k for k in n_kv_options if k >= 1 and n_heads % k == 0]
        n_kv_options = sorted(set(n_kv_options), reverse=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    head_dim = dim // n_heads
    max_len = context_len + n_decode_steps + 10

    print(f"\nInference Throughput: GQA n_kv_heads Sweep")
    print(f"Device: {device} | dim={dim} | n_heads={n_heads} | head_dim={head_dim}")
    print(f"Context: {context_len} tokens | Decode: {n_decode_steps} tokens")
    print()
    print(f"  {'n_kv_heads':>12} | {'name':>12} | {'tok/s':>10} | {'KV cache MB':>12} | {'Speedup':>10}")
    print("  " + "-" * 65)

    results = []
    baseline_tps = None

    for n_kv in n_kv_options:
        layer = AttentionWithKVCache(dim, n_heads, n_kv).to(device)
        cache = KVCache(1, n_kv, max_len, head_dim, device=device)
        cos, sin = precompute_freqs(head_dim, max_len, device=device)

        # prefill
        x_pre = torch.randn(1, context_len, dim, device=device)
        with torch.no_grad():
            layer(x_pre, cos[:context_len], sin[:context_len], cache=cache, layer_idx=0)

        def _sync():
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

        # decode
        t0 = time.perf_counter()
        for step in range(n_decode_steps):
            pos = context_len + step
            x_tok = torch.randn(1, 1, dim, device=device)
            with torch.no_grad():
                layer(x_tok, cos[pos:pos+1], sin[pos:pos+1], cache=cache, layer_idx=0)
        _sync()
        elapsed = time.perf_counter() - t0

        tps = n_decode_steps / elapsed
        kv_mb = cache.memory_mb()

        if baseline_tps is None:
            baseline_tps = tps

        speedup = tps / baseline_tps

        if n_kv == n_heads:
            name = "MHA"
        elif n_kv == 1:
            name = "MQA"
        else:
            name = f"GQA"

        print(f"  {n_kv:>12} | {name:>12} | {tps:>10.1f} | {kv_mb:>12.2f} | {speedup:>9.2f}x")
        results.append({
            "n_kv": n_kv,
            "name": name,
            "tps": tps,
            "kv_mb": kv_mb,
            "speedup": speedup,
        })

        del layer, cache
        gc.collect()

    return results


def plot_kv_sweep(results, save_path="plots/kv_sweep.png"):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n_kvs = [r["n_kv"] for r in results]
    tps = [r["tps"] for r in results]
    kv_mbs = [r["kv_mb"] for r in results]
    speedups = [r["speedup"] for r in results]
    names = [r["name"] for r in results]

    colors = ["steelblue", "coral", "seagreen", "purple", "orange"][:len(results)]

    bars0 = axes[0].bar(names, tps, color=colors, alpha=0.8)
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_title("Decode Throughput by KV Head Count")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars0, tps):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # tradeoff: throughput vs KV cache size
    axes[1].scatter(kv_mbs, tps, c=colors, s=100, zorder=5)
    for r, color in zip(results, colors):
        axes[1].annotate(r["name"], (r["kv_mb"], r["tps"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel("KV Cache Size (MB)")
    axes[1].set_ylabel("Tokens / second")
    axes[1].set_title("Throughput vs KV Cache Size Tradeoff")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("GQA / MQA Inference Sweep", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = sweep_kv_heads_inference()
    plot_kv_sweep(results)
