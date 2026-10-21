"""Prefill vs decode phase analysis for LLM inference.

LLM inference has two distinct phases:
1. Prefill: process the entire prompt in parallel (compute-bound)
2. Decode: generate one token at a time (memory-bandwidth-bound)

These have very different performance characteristics:
- Prefill: high arithmetic intensity (lots of compute, good GPU utilization)
- Decode: low arithmetic intensity (tiny batch, memory-bandwidth limited)

Batching decode requests improves GPU utilization -- this is what vLLM/TGI do.
"""

import time
import math
import torch
import torch.nn.functional as F

from rope import precompute_freqs
from kv_cache import KVCache, AttentionWithKVCache


def measure_prefill(
    prompt_len: int,
    dim: int = 512,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    n_layers: int = 4,
    device=None,
    repeats: int = 5,
):
    """Measure prefill throughput (tokens/sec) for a given prompt length."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    head_dim = dim // n_heads
    layer = AttentionWithKVCache(dim, n_heads, n_kv_heads).to(device)
    cos, sin = precompute_freqs(head_dim, prompt_len + 10, device=device)

    def _sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def run():
        x = torch.randn(1, prompt_len, dim, device=device)
        with torch.no_grad():
            layer(x, cos[:prompt_len], sin[:prompt_len])

    for _ in range(2):
        run()
    _sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        _sync()
        times.append(time.perf_counter() - t0)

    ms = sorted(times)[len(times) // 2] * 1000
    tps = prompt_len / (ms / 1000)
    return ms, tps


def measure_decode_batched(
    batch_size: int,
    context_len: int = 256,
    n_decode: int = 50,
    dim: int = 512,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    device=None,
):
    """Measure decode throughput for a batch of requests."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    head_dim = dim // n_heads
    max_len = context_len + n_decode + 5
    layer = AttentionWithKVCache(dim, n_heads, n_kv_heads).to(device)
    cos, sin = precompute_freqs(head_dim, max_len, device=device)

    # Use batch_size caches (one per request)
    # Simplified: use a single batched layer
    cache = KVCache(1, n_kv_heads, max_len, head_dim, device=device)

    def _sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    # prefill all requests
    x_pre = torch.randn(batch_size, context_len, dim, device=device)
    with torch.no_grad():
        # Note: using batch_size here for demonstration
        layer(x_pre[:1], cos[:context_len], sin[:context_len], cache=cache, layer_idx=0)
    cache.reset()

    t0 = time.perf_counter()
    for step in range(n_decode):
        pos = context_len + step
        x_tok = torch.randn(batch_size, 1, dim, device=device)
        with torch.no_grad():
            layer(x_tok[:1], cos[pos:pos+1], sin[pos:pos+1], cache=cache, layer_idx=0)
    _sync()
    elapsed = time.perf_counter() - t0

    total_tokens = n_decode * batch_size
    tps = total_tokens / elapsed
    return elapsed * 1000, tps


def prefill_decode_comparison(
    prompt_lengths=None,
    decode_batches=None,
    dim=512, n_heads=8, n_kv_heads=4,
    device=None,
):
    if prompt_lengths is None:
        prompt_lengths = [64, 128, 256, 512, 1024]
    if decode_batches is None:
        decode_batches = [1, 4, 8, 16]

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nPrefill vs Decode Phase Analysis")
    print(f"Device: {device} | dim={dim} | n_heads={n_heads} | n_kv={n_kv_heads}")

    print(f"\n--- Prefill throughput (prompt processing) ---")
    print(f"{'prompt_len':>12} | {'time (ms)':>10} | {'tok/s':>10}")
    print("-" * 40)
    for plen in prompt_lengths:
        ms, tps = measure_prefill(plen, dim, n_heads, n_kv_heads, device=device)
        print(f"{plen:>12} | {ms:>10.2f} | {tps:>10.0f}")

    print(f"\n--- Decode throughput (generation, 50 steps) ---")
    print(f"{'batch_size':>12} | {'time (ms)':>10} | {'tok/s':>10} | {'utilization':>12}")
    print("-" * 50)
    batch1_tps = None
    for bs in decode_batches:
        ms, tps = measure_decode_batched(bs, dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads, device=device)
        if batch1_tps is None:
            batch1_tps = tps
        util = tps / (batch1_tps * bs) * 100 if bs > 1 else 100.0
        print(f"{bs:>12} | {ms:>10.2f} | {tps:>10.0f} | {util:>11.1f}%")

    print(f"\nNote: prefill is compute-bound (parallel processing)")
    print(f"      decode is memory-bandwidth-bound (one token at a time)")
    print(f"      batching decode requests increases GPU utilization")


if __name__ == "__main__":
    prefill_decode_comparison()
