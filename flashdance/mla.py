"""Multi-head Latent Attention (MLA) -- DeepSeek-V2/V3 innovation.

Standard MHA stores KV cache: 2 * n_heads * head_dim * seq_len per layer.
MLA compresses KV into a low-rank latent vector c_kv of dimension d_c << n_heads * head_dim.

At inference, only c_kv is cached (93% memory reduction in DeepSeek-V2).
The full K,V are reconstructed from c_kv on the fly.

Reference: https://arxiv.org/abs/2405.04434 (DeepSeek-V2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import precompute_freqs, apply_rope_single


class MultiHeadLatentAttention(nn.Module):
    """MLA as described in DeepSeek-V2.

    Key design:
    - Compress KV into latent c_kv of dim d_c (much smaller than n_heads * head_dim)
    - Cache c_kv instead of K, V
    - Decompress back to K, V for attention computation
    - Q also optionally compressed through latent c_q

    For RoPE: a separate rope-specific K (k_rope) is kept uncompressed
    because RoPE is position-dependent and can't be absorbed into the latent.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_c: int,       # KV latent dimension (e.g., 512 for DeepSeek-V2 vs 128*64=8192 for MHA)
        d_c_q: int = None,  # Q latent dimension (if None, no Q compression)
        d_rope: int = 64,   # dimension for decoupled RoPE
    ):
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.d_c = d_c
        self.d_c_q = d_c_q
        self.d_rope = d_rope

        # KV compression
        self.kv_compress = nn.Linear(dim, d_c, bias=False)
        self.k_decompress = nn.Linear(d_c, n_heads * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(d_c, n_heads * self.head_dim, bias=False)

        # separate RoPE key (position-sensitive, can't be compressed)
        self.k_rope_proj = nn.Linear(dim, d_rope, bias=False)

        if d_c_q is not None:
            # Q compression (optional, reduces activation memory during training)
            self.q_compress = nn.Linear(dim, d_c_q, bias=False)
            self.q_decompress = nn.Linear(d_c_q, n_heads * self.head_dim, bias=False)
        else:
            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)

        # RoPE for Q
        self.q_rope_proj = nn.Linear(dim, d_rope, bias=False)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None, causal=True):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # --- KV path ---
        c_kv = self.kv_compress(x)                             # (B, T, d_c)  ← only this is cached
        k = self.k_decompress(c_kv).view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        v = self.v_decompress(c_kv).view(B, T, H, D).transpose(1, 2)

        # --- Q path ---
        if self.d_c_q is not None:
            c_q = self.q_compress(x)
            q = self.q_decompress(c_q).view(B, T, H, D).transpose(1, 2)
        else:
            q = self.wq(x).view(B, T, H, D).transpose(1, 2)

        # --- Decoupled RoPE ---
        if cos is not None and sin is not None:
            # apply rope to rope-specific subspace, then concat with non-rope dims
            k_rope = self.k_rope_proj(x)                            # (B, T, d_rope)
            q_rope = self.q_rope_proj(x)                            # (B, T, d_rope)

            k_rope = k_rope.view(B, T, 1, self.d_rope).transpose(1, 2).expand(B, H, T, self.d_rope)
            q_rope = q_rope.view(B, T, 1, self.d_rope).transpose(1, 2).expand(B, H, T, self.d_rope)

            # only rotate the rope portion -- simplified version
            cos_r = cos[:, :self.d_rope // 2]
            sin_r = sin[:, :self.d_rope // 2]

            q_rope = apply_rope_single(q_rope, cos_r, sin_r)
            k_rope = apply_rope_single(k_rope, cos_r, sin_r)

            # concatenate rope dims to q, k
            rope_dim = min(self.d_rope, D)
            q = torch.cat([q[..., :-rope_dim], q_rope[..., :rope_dim]], dim=-1)
            k = torch.cat([k[..., :-rope_dim], k_rope[..., :rope_dim]], dim=-1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)

    def kv_cache_size_bytes(self, seq_len: int, dtype=torch.float16) -> int:
        """MLA only caches c_kv, not full K,V."""
        bpe = torch.finfo(dtype).bits // 8
        return self.d_c * seq_len * bpe

    def mha_kv_cache_size_bytes(self, seq_len: int, dtype=torch.float16) -> int:
        """Equivalent MHA would cache this much."""
        bpe = torch.finfo(dtype).bits // 8
        return 2 * self.n_heads * self.head_dim * seq_len * bpe


def compare_kv_cache_sizes(
    dim: int = 5120,      # DeepSeek-V2 model dim
    n_heads: int = 128,
    d_c: int = 512,       # MLA latent dim (DeepSeek-V2 uses 512)
    seq_len: int = 4096,
    n_layers: int = 60,   # DeepSeek-V2 has 60 layers
    dtype=torch.float16,
):
    """Compare total KV cache for MHA vs MLA at DeepSeek-V2 scale."""
    bpe = torch.finfo(dtype).bits // 8
    head_dim = dim // n_heads

    mha_per_layer = 2 * n_heads * head_dim * seq_len * bpe
    mla_per_layer = d_c * seq_len * bpe

    mha_total_gb = mha_per_layer * n_layers / 1024**3
    mla_total_gb = mla_per_layer * n_layers / 1024**3

    print(f"\nKV Cache: MHA vs MLA (DeepSeek-V2 scale)")
    print(f"dim={dim}, n_heads={n_heads}, head_dim={head_dim}, seq_len={seq_len}, n_layers={n_layers}")
    print(f"-" * 60)
    print(f"  MHA KV cache: {mha_total_gb:.2f} GB")
    print(f"  MLA KV cache: {mla_total_gb:.2f} GB  (d_c={d_c})")
    print(f"  Reduction:    {mha_total_gb / mla_total_gb:.1f}x  ({100*(1-mla_total_gb/mha_total_gb):.0f}% savings)")

    return {
        "mha_gb": mha_total_gb,
        "mla_gb": mla_total_gb,
        "reduction": mha_total_gb / mla_total_gb,
    }


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dim, n_heads, d_c = 512, 8, 64
    head_dim = dim // n_heads

    mla = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c).to(device)
    B, T = 2, 64
    x = torch.randn(B, T, dim, device=device)

    cos, sin = precompute_freqs(head_dim, T, device=device)
    out = mla(x, cos=cos, sin=sin)
    assert out.shape == (B, T, dim), f"Expected {(B, T, dim)}, got {out.shape}"
    print(f"MLA output: {out.shape}  ✓")

    mha_cache = mla.mha_kv_cache_size_bytes(T) / 1024
    mla_cache = mla.kv_cache_size_bytes(T) / 1024
    print(f"KV cache (T={T}): MHA={mha_cache:.1f}KB, MLA={mla_cache:.1f}KB, reduction={mha_cache/mla_cache:.1f}x")

    compare_kv_cache_sizes()
