"""KV Cache for autoregressive inference.

During generation, we don't want to recompute K,V for every token.
Instead we cache them and only compute attention for the new token against
the full cached sequence. This is what makes inference fast.

Memory cost: 2 * n_layers * n_kv_heads * max_seq_len * head_dim * bytes_per_element
Example: LLaMA-7B needs ~1GB for 2K context in float16.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import precompute_freqs, apply_rope
from gqa import repeat_kv


class KVCache:
    """Static KV cache (pre-allocated, no dynamic growth)."""

    def __init__(self, n_layers: int, n_kv_heads: int, max_seq_len: int, head_dim: int, dtype=torch.float16, device=None):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or "cpu"

        # pre-allocate K and V for all layers
        shape = (n_layers, n_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=self.device)
        self.pos = 0  # current position (number of tokens filled)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Write new K,V at current position and return the full cache slice.

        Args:
            layer_idx: which transformer layer
            k, v: (batch=1, n_kv_heads, new_tokens, head_dim) -- new tokens only

        Returns:
            k_full, v_full: (batch=1, n_kv_heads, pos+new_tokens, head_dim)
        """
        new_len = k.shape[2]
        self.k_cache[layer_idx, :, self.pos:self.pos + new_len, :] = k[0]
        self.v_cache[layer_idx, :, self.pos:self.pos + new_len, :] = v[0]
        k_full = self.k_cache[layer_idx, :, :self.pos + new_len, :].unsqueeze(0)
        v_full = self.v_cache[layer_idx, :, :self.pos + new_len, :].unsqueeze(0)
        return k_full, v_full

    def advance(self, n_tokens: int = 1):
        self.pos += n_tokens

    def reset(self):
        self.pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def memory_bytes(self) -> int:
        return self.k_cache.nelement() * self.k_cache.element_size() * 2

    def memory_mb(self) -> float:
        return self.memory_bytes() / 1024**2


class AttentionWithKVCache(nn.Module):
    """Attention that supports both prefill and cached autoregressive decode."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin, cache: KVCache = None, layer_idx: int = 0, causal=True):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
            cache.advance(T)
            # decoding: not causal (we already mask via the cache position)
            causal = False

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k, v = repeat_kv(k, v, n_rep)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


def benchmark_kv_cache(
    n_gen_tokens: int = 128,
    prefill_len: int = 128,
    dim: int = 512,
    n_heads: int = 8,
    n_kv_heads: int = 2,
    n_layers: int = 4,
    device=None,
):
    """Compare decode speed with vs without KV cache."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    head_dim = dim // n_heads
    max_seq = prefill_len + n_gen_tokens + 10

    layer = AttentionWithKVCache(dim, n_heads, n_kv_heads).to(device)

    print(f"\nKV Cache Benchmark")
    print(f"Device: {device} | dim={dim} | n_heads={n_heads} | n_kv_heads={n_kv_heads}")
    print(f"Prefill: {prefill_len} tokens | Generate: {n_gen_tokens} tokens")

    # --- WITHOUT cache: recompute everything each step ---
    cos_full, sin_full = precompute_freqs(head_dim, max_seq, device=device)
    x_prefill = torch.randn(1, prefill_len, dim, device=device)

    t0 = time.perf_counter()
    for step in range(n_gen_tokens):
        total_len = prefill_len + step + 1
        x_full = torch.randn(1, total_len, dim, device=device)
        cos = cos_full[:total_len]
        sin = sin_full[:total_len]
        with torch.no_grad():
            layer(x_full, cos, sin, cache=None, causal=True)
    if device == "mps":
        torch.mps.synchronize()
    no_cache_time = time.perf_counter() - t0
    no_cache_tps = n_gen_tokens / no_cache_time

    # --- WITH cache ---
    cache = KVCache(n_layers=1, n_kv_heads=n_kv_heads, max_seq_len=max_seq, head_dim=head_dim, device=device)

    t0 = time.perf_counter()
    # prefill
    cos_p = cos_full[:prefill_len]
    sin_p = sin_full[:prefill_len]
    with torch.no_grad():
        layer(x_prefill, cos_p, sin_p, cache=cache, layer_idx=0)

    for step in range(n_gen_tokens):
        pos = prefill_len + step
        x_tok = torch.randn(1, 1, dim, device=device)
        cos = cos_full[pos:pos + 1]
        sin = sin_full[pos:pos + 1]
        with torch.no_grad():
            layer(x_tok, cos, sin, cache=cache, layer_idx=0)
    if device == "mps":
        torch.mps.synchronize()
    with_cache_time = time.perf_counter() - t0
    with_cache_tps = n_gen_tokens / with_cache_time

    print(f"\n  Without KV cache: {no_cache_time*1000:.1f}ms  ({no_cache_tps:.1f} tok/s)")
    print(f"  With    KV cache: {with_cache_time*1000:.1f}ms  ({with_cache_tps:.1f} tok/s)")
    print(f"  Speedup:          {no_cache_time / with_cache_time:.2f}x")
    print(f"  Cache memory:     {cache.memory_mb():.2f} MB")

    return {
        "no_cache_ms": no_cache_time * 1000,
        "with_cache_ms": with_cache_time * 1000,
        "speedup": no_cache_time / with_cache_time,
        "cache_mb": cache.memory_mb(),
        "no_cache_tps": no_cache_tps,
        "with_cache_tps": with_cache_tps,
    }


if __name__ == "__main__":
    result = benchmark_kv_cache()
