"""Vanilla vs Flash Attention implementations for benchmarking."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def vanilla_attention(q, k, v, causal=True, dropout_p=0.0):
    """Standard scaled dot-product attention. Materializes full NxN matrix.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        causal: apply causal mask
        dropout_p: dropout probability on attention weights

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # full NxN attention matrix -- this is what flash attention avoids
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)

    if dropout_p > 0.0 and q.requires_grad:
        attn = F.dropout(attn, p=dropout_p)

    out = torch.matmul(attn, v)
    return out


def vanilla_attention_with_scores(q, k, v, causal=True):
    """Same as vanilla_attention but also returns the attention weights.

    Useful for visualization and debugging.
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


def sdpa_attention(q, k, v, causal=True):
    """PyTorch's scaled_dot_product_attention (uses flash attention when available).

    This calls into the optimized C++ kernels:
    - Flash Attention (if available and supported)
    - Memory-efficient attention (xformers-style)
    - Math fallback

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        causal: apply causal mask

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with selectable backend."""

    def __init__(self, dim, n_heads, backend="vanilla"):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        if backend == "vanilla":
            self.attn_fn = vanilla_attention
        elif backend == "sdpa":
            self.attn_fn = sdpa_attention
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def forward(self, x, causal=True):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (batch, heads, seq_len, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = self.attn_fn(q, k, v, causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


def check_backends():
    """Check which SDPA backends are available on this hardware.

    Returns a dict with keys: flash, mem_efficient, math
    """
    backends = {}

    # check flash attention
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            backends["flash"] = True
    except Exception:
        backends["flash"] = False

    # check memory efficient
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            backends["mem_efficient"] = True
    except Exception:
        backends["mem_efficient"] = False

    backends["math"] = True  # always available
    return backends
