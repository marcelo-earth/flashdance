"""Grouped Query Attention (GQA) and Multi-Query Attention (MQA).

GQA is used by LLaMA 2 70B, LLaMA 3, Mistral, Gemma, DeepSeek-V2.
MQA is a special case of GQA where n_kv_heads=1.

Key insight: GQA reduces KV cache size from O(n_heads) to O(n_kv_heads),
which is critical for long-context inference throughput.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rope, precompute_freqs


def repeat_kv(k, v, n_rep: int):
    """Expand KV heads to match Q heads by repeating.

    Args:
        k, v: (batch, n_kv_heads, seq_len, head_dim)
        n_rep: how many times to repeat each KV head (= n_heads // n_kv_heads)

    Returns:
        k, v: (batch, n_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return k, v
    B, H_kv, T, D = k.shape
    k = k.unsqueeze(2).expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)
    v = v.unsqueeze(2).expand(B, H_kv, n_rep, T, D).reshape(B, H_kv * n_rep, T, D)
    return k, v


def grouped_query_attention(q, k, v, n_heads: int, n_kv_heads: int, causal=True):
    """Grouped Query Attention (GQA).

    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        v: (batch, n_kv_heads, seq_len, head_dim)
        n_heads: number of query heads
        n_kv_heads: number of key/value heads (must divide n_heads)
        causal: apply causal mask

    Returns:
        output: (batch, n_heads, seq_len, head_dim)
    """
    assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    n_rep = n_heads // n_kv_heads
    k_exp, v_exp = repeat_kv(k, v, n_rep)
    return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=causal)


def multi_query_attention(q, k, v, causal=True):
    """Multi-Query Attention (MQA) -- special case of GQA with n_kv_heads=1.

    Used by PaLM, Falcon, StarCoder. Maximally reduces KV cache.
    """
    n_heads = q.shape[1]
    return grouped_query_attention(q, k, v, n_heads, n_kv_heads=1, causal=causal)


class GroupedQueryAttention(nn.Module):
    """Full GQA module with linear projections.

    This is the architecture used in LLaMA 2 70B and LLaMA 3.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None, causal=True):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rope(q, k, cos, sin)

        out = grouped_query_attention(q, k, v, self.n_heads, self.n_kv_heads, causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

    def kv_cache_size_bytes(self, seq_len: int, dtype=torch.float16) -> int:
        """Estimate KV cache memory in bytes for this attention layer."""
        bytes_per_elem = torch.finfo(dtype).bits // 8
        # K cache + V cache
        return 2 * self.n_kv_heads * seq_len * self.head_dim * bytes_per_elem


def kv_cache_memory_comparison(
    dim=512,
    n_heads=8,
    seq_len=4096,
    dtype=torch.float16,
):
    """Compare KV cache memory for MHA, GQA (n_kv=2), MQA."""
    bytes_per_elem = torch.finfo(dtype).bits // 8
    head_dim = dim // n_heads

    configs = {
        "MHA (n_kv=8)": n_heads,
        "GQA (n_kv=4)": 4,
        "GQA (n_kv=2)": 2,
        "MQA (n_kv=1)": 1,
    }

    print(f"\nKV Cache Memory Comparison")
    print(f"dim={dim}, n_heads={n_heads}, head_dim={head_dim}, seq_len={seq_len}, dtype={dtype}")
    print("-" * 55)

    results = {}
    for name, n_kv in configs.items():
        # K + V, each: (n_kv_heads, seq_len, head_dim)
        kb = 2 * n_kv * seq_len * head_dim * bytes_per_elem
        mb = kb / 1024**2
        reduction = configs["MHA (n_kv=8)"] / n_kv
        results[name] = {"mb": mb, "reduction": reduction, "n_kv_heads": n_kv}
        print(f"  {name:20s}: {mb:.2f} MB  ({reduction:.1f}x reduction vs MHA)")

    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # test GQA
    B, T, D = 2, 128, 512
    n_heads, n_kv_heads = 8, 2

    x = torch.randn(B, T, D, device=device)
    gqa = GroupedQueryAttention(D, n_heads, n_kv_heads).to(device)
    out = gqa(x)
    assert out.shape == (B, T, D)
    print(f"GQA output shape: {out.shape}  ✓")

    head_dim = D // n_heads
    cos, sin = precompute_freqs(head_dim, T, device=device)
    out_rope = gqa(x, cos=cos, sin=sin)
    assert out_rope.shape == (B, T, D)
    print(f"GQA + RoPE output shape: {out_rope.shape}  ✓")

    kv_cache_memory_comparison()
