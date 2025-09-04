"""ALiBi (Attention with Linear Biases) -- train short, test long.

Instead of adding positional embeddings to tokens, ALiBi adds a
linear bias to attention scores based on the distance between positions.

bias(i, j) = -m * |i - j|

where m is a head-specific slope. This gives each head a different
"recency bias" -- some heads prefer very recent tokens, others look
further back.

Key advantage: trained at context 1024, works at inference time 2048+
without any positional encoding or fine-tuning.

Reference: https://arxiv.org/abs/2108.12409 (Train Short, Test Long)
Used by: BLOOM (176B), MPT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each head.

    The slopes follow a geometric sequence: 2^(-8/n_heads) for the first head,
    2^(-8*2/n_heads) for the second, etc.

    Returns:
        slopes: (n_heads,)
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        # for non-power-of-2 heads, interpolate
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power)
        extra = get_slopes_power_of_2(2 * closest_power)
        extra = extra[0::2]  # take every other
        slopes = slopes + extra[:n_heads - closest_power]

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(n_heads: int, seq_len: int, device=None) -> torch.Tensor:
    """Build the ALiBi bias matrix for attention.

    Returns:
        bias: (1, n_heads, seq_len, seq_len)
        Causal version: lower triangle has the actual biases, upper = -inf
    """
    slopes = get_alibi_slopes(n_heads).to(device)  # (n_heads,)

    # relative distances: for position i attending to position j, distance = i - j
    positions = torch.arange(seq_len, device=device)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)

    # causal: positions in the future get -inf
    causal_mask = distances < 0  # upper triangle

    # ALiBi bias = -slope * |distance|
    distances = distances.float().abs()
    bias = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)  # (n_heads, T, T)
    bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

    return bias.unsqueeze(0)  # (1, n_heads, T, T)


def alibi_attention(q, k, v, alibi_bias=None, causal=True):
    """Scaled dot-product attention with ALiBi position bias.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        alibi_bias: (1, heads, seq_len, seq_len) -- from build_alibi_bias
        causal: if True, also apply causal mask (use False if alibi_bias is already causal)

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

    if alibi_bias is not None:
        attn = attn + alibi_bias[:, :, :T, :T]
    elif causal:
        # plain causal mask if no ALiBi
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi position bias (BLOOM-style)."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        # register slopes as buffer (not a parameter)
        self.register_buffer("slopes", get_alibi_slopes(n_heads))
        self._bias_cache = {}

    def _get_bias(self, seq_len: int, device) -> torch.Tensor:
        if seq_len not in self._bias_cache:
            self._bias_cache[seq_len] = build_alibi_bias(self.n_heads, seq_len, device=device)
        return self._bias_cache[seq_len]

    def forward(self, x, causal=True):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        bias = self._get_bias(T, x.device)
        out = alibi_attention(q, k, v, alibi_bias=bias, causal=False)  # bias already handles causality
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


def compare_position_encodings(seq_len=128, n_heads=8, head_dim=64):
    """Visualize how RoPE vs ALiBi encode positions differently."""
    import matplotlib.pyplot as plt
    import numpy as np

    # ALiBi slopes
    slopes = get_alibi_slopes(n_heads).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ALiBi bias for head 0 (strongest recency bias)
    positions = np.arange(seq_len)
    distances = np.abs(positions[:, None] - positions[None, :])

    alibi_h0 = -slopes[0] * distances
    im0 = axes[0].imshow(alibi_h0, cmap="RdBu_r", vmin=-slopes[0] * seq_len, vmax=0)
    axes[0].set_title(f"ALiBi bias (head 0, slope={slopes[0]:.4f})")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im0, ax=axes[0])

    # ALiBi bias for last head (weakest recency bias)
    alibi_last = -slopes[-1] * distances
    im1 = axes[1].imshow(alibi_last, cmap="RdBu_r", vmin=-slopes[-1] * seq_len, vmax=0)
    axes[1].set_title(f"ALiBi bias (head {n_heads-1}, slope={slopes[-1]:.6f})")
    axes[1].set_xlabel("Key position")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle("ALiBi: each head has a different recency decay rate", fontsize=13)
    plt.tight_layout()

    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/alibi_slopes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/alibi_slopes.png")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    B, T, D = 2, 64, 256
    n_heads = 8

    model = ALiBiAttention(D, n_heads).to(device)
    x = torch.randn(B, T, D, device=device)
    out = model(x)
    assert out.shape == (B, T, D)
    print(f"ALiBi attention output: {out.shape}  ✓")

    slopes = get_alibi_slopes(n_heads)
    print(f"Slopes: {slopes.tolist()}")
    print(f"Slope range: {slopes.min():.6f} to {slopes.max():.4f}")
    print(f"(head 0 = strongest local bias, head {n_heads-1} = weakest/most global)")

    compare_position_encodings(seq_len=64, n_heads=n_heads)
