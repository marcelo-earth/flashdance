"""Speculative decoding attention benchmark.

Speculative decoding uses a small draft model to generate k tokens,
then verifies all k in a single target model forward pass.
This gives 2-3x inference speedup with identical output distribution.

This module benchmarks the attention cost for:
1. Draft generation (small model, one token at a time)
2. Target verification (full model, k tokens in parallel)
3. The overall speedup from speculation

Reference: https://arxiv.org/abs/2211.17192 (Speculative Decoding)
Also: Medusa (https://arxiv.org/abs/2401.10774) -- multiple draft heads
"""

import math
import time
import torch
import torch.nn.functional as F

from attention import sdpa_attention
from kv_cache import KVCache, AttentionWithKVCache
from rope import precompute_freqs


def simulate_draft_decode(
    draft_dim: int = 256,
    draft_heads: int = 4,
    target_dim: int = 512,
    target_heads: int = 8,
    k_draft: int = 4,
    context_len: int = 256,
    n_steps: int = 50,
    device=None,
):
    """Simulate the attention cost of speculative decoding.

    Compares:
    - Standard autoregressive decode (target model, 1 token at a time)
    - Speculative decode (draft k tokens, verify with target in one pass)
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    draft_head_dim = draft_dim // draft_heads
    target_head_dim = target_dim // target_heads
    max_len = context_len + n_steps * k_draft + 10

    draft_cos, draft_sin = precompute_freqs(draft_head_dim, max_len, device=device)
    target_cos, target_sin = precompute_freqs(target_head_dim, max_len, device=device)

    draft_layer = AttentionWithKVCache(draft_dim, draft_heads, draft_heads // 2).to(device)
    target_layer = AttentionWithKVCache(target_dim, target_heads, target_heads // 2).to(device)

    # --- Baseline: standard autoregressive (target only, 1 token at a time) ---
    cache = KVCache(1, target_heads // 2, max_len, target_head_dim, device=device)
    # prefill
    x_pre = torch.randn(1, context_len, target_dim, device=device)
    with torch.no_grad():
        target_layer(x_pre, target_cos[:context_len], target_sin[:context_len], cache=cache, layer_idx=0)

    t0 = time.perf_counter()
    total_tokens_ar = n_steps
    for step in range(n_steps):
        pos = context_len + step
        x_tok = torch.randn(1, 1, target_dim, device=device)
        with torch.no_grad():
            target_layer(x_tok, target_cos[pos:pos+1], target_sin[pos:pos+1], cache=cache, layer_idx=0)
    if device == "mps":
        torch.mps.synchronize()
    ar_time = time.perf_counter() - t0
    ar_tps = total_tokens_ar / ar_time

    # --- Speculative: draft k tokens, then verify with target ---
    # Assume ~k_draft * accept_rate tokens accepted per step (typical accept rate ~0.8)
    accept_rate = 0.8

    draft_cache = KVCache(1, draft_heads // 2, max_len, draft_head_dim, device=device)
    target_cache2 = KVCache(1, target_heads // 2, max_len, target_head_dim, device=device)

    # prefill both caches
    x_pre_draft = torch.randn(1, context_len, draft_dim, device=device)
    x_pre_target = torch.randn(1, context_len, target_dim, device=device)
    with torch.no_grad():
        draft_layer(x_pre_draft, draft_cos[:context_len], draft_sin[:context_len], cache=draft_cache, layer_idx=0)
        target_layer(x_pre_target, target_cos[:context_len], target_sin[:context_len], cache=target_cache2, layer_idx=0)

    t0 = time.perf_counter()
    spec_steps = n_steps // k_draft  # number of speculative steps
    total_tokens_spec = 0
    pos = context_len

    for _ in range(spec_steps):
        # draft: generate k tokens autoregressively (cheap small model)
        for j in range(k_draft):
            x_d = torch.randn(1, 1, draft_dim, device=device)
            with torch.no_grad():
                draft_layer(x_d, draft_cos[pos+j:pos+j+1], draft_sin[pos+j:pos+j+1],
                           cache=draft_cache, layer_idx=0)

        # target: verify k tokens in a single forward pass (parallel)
        x_verify = torch.randn(1, k_draft, target_dim, device=device)
        with torch.no_grad():
            target_layer(x_verify, target_cos[pos:pos+k_draft], target_sin[pos:pos+k_draft],
                        cache=target_cache2, layer_idx=0)

        # accept ~k_draft * accept_rate tokens
        accepted = max(1, int(k_draft * accept_rate))
        total_tokens_spec += accepted
        pos += accepted

    if device == "mps":
        torch.mps.synchronize()
    spec_time = time.perf_counter() - t0
    spec_tps = total_tokens_spec / spec_time

    print(f"\nSpeculative Decoding Simulation")
    print(f"Device: {device} | draft_dim={draft_dim} | target_dim={target_dim}")
    print(f"k_draft={k_draft} | context={context_len} | accept_rate={accept_rate}")
    print(f"\n  Autoregressive:  {ar_tps:>8.1f} tok/s  ({ar_time*1000:.1f}ms total)")
    print(f"  Speculative:     {spec_tps:>8.1f} tok/s  ({spec_time*1000:.1f}ms total)")
    print(f"  Speedup:         {spec_tps/ar_tps:>7.2f}x")

    return {
        "ar_tps": ar_tps,
        "spec_tps": spec_tps,
        "speedup": spec_tps / ar_tps,
        "k_draft": k_draft,
        "accept_rate": accept_rate,
    }


def sweep_k_draft(k_values=None, context_len=256, n_steps=40, device=None):
    """Show how speedup varies with number of draft tokens k."""
    if k_values is None:
        k_values = [1, 2, 4, 8]

    print(f"\nSpeculative Decoding: Speedup vs k_draft")
    print(f"{'k_draft':>8} | {'Speedup':>10}")
    print("-" * 25)

    results = []
    for k in k_values:
        r = simulate_draft_decode(k_draft=k, context_len=context_len, n_steps=max(k * 10, 40), device=device)
        results.append(r)
        print(f"{k:>8} | {r['speedup']:>9.2f}x")

    return results


if __name__ == "__main__":
    simulate_draft_decode()
    sweep_k_draft()
