"""Cross-attention for encoder-decoder models (T5, Whisper, etc.).

In cross-attention:
- Q comes from the decoder
- K, V come from the encoder output
- seq_len of Q (decoder) != seq_len of KV (encoder)

This is different from self-attention where Q, K, V all have the same seq_len.

Used by: T5, BART, Whisper, vision-language models (ViT + LLM)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_attention(q, k, v, causal=False):
    """Cross-attention between Q (decoder) and K,V (encoder).

    Args:
        q: (batch, n_heads, tgt_len, head_dim)   -- decoder queries
        k: (batch, n_heads, src_len, head_dim)   -- encoder keys
        v: (batch, n_heads, src_len, head_dim)   -- encoder values
        causal: usually False for cross-attention

    Returns:
        output: (batch, n_heads, tgt_len, head_dim)
    """
    # Note: tgt_len != src_len, SDPA handles this correctly
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def cross_attention_with_mask(q, k, v, src_key_padding_mask=None):
    """Cross-attention with optional source padding mask.

    Args:
        q: (batch, n_heads, tgt_len, head_dim)
        k, v: (batch, n_heads, src_len, head_dim)
        src_key_padding_mask: (batch, src_len) bool mask, True = ignore

    Returns:
        output: (batch, n_heads, tgt_len, head_dim)
    """
    B, H, T, D = q.shape
    S = k.shape[2]
    scale = 1.0 / math.sqrt(D)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, S)

    if src_key_padding_mask is not None:
        # expand mask: (B, S) -> (B, 1, 1, S)
        mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)
    # handle all-masked rows (padding-only sequences)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


class CrossAttention(nn.Module):
    """Cross-attention module for encoder-decoder architectures."""

    def __init__(self, dim: int, n_heads: int, encoder_dim: int = None, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        if encoder_dim is None:
            encoder_dim = dim

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(encoder_dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(encoder_dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_key_padding_mask=None):
        """
        Args:
            x: (batch, tgt_len, dim)                    -- decoder hidden states
            encoder_out: (batch, src_len, encoder_dim)  -- encoder outputs
            src_key_padding_mask: (batch, src_len)      -- True = ignore

        Returns:
            output: (batch, tgt_len, dim)
        """
        B, T, _ = x.shape
        S = encoder_out.shape[1]
        H, D = self.n_heads, self.head_dim

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(encoder_out).view(B, S, H, D).transpose(1, 2)
        v = self.wv(encoder_out).view(B, S, H, D).transpose(1, 2)

        if src_key_padding_mask is not None:
            out = cross_attention_with_mask(q, k, v, src_key_padding_mask)
        else:
            out = cross_attention(q, k, v)

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, dim: int, n_heads: int, encoder_dim: int = None, ffn_dim: int = None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * dim
        if encoder_dim is None:
            encoder_dim = dim

        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = CrossAttention(dim, n_heads, encoder_dim=encoder_dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x, encoder_out, src_key_padding_mask=None, tgt_mask=None):
        """Pre-norm transformer decoder layer (as in modern LLMs)."""
        # self-attention with causal mask
        h = self.norm1(x)
        sa_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask)
        x = x + sa_out

        # cross-attention to encoder
        h = self.norm2(x)
        ca_out = self.cross_attn(h, encoder_out, src_key_padding_mask=src_key_padding_mask)
        x = x + ca_out

        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


def benchmark_cross_vs_self(
    src_len: int = 512,
    tgt_lengths=None,
    batch_size: int = 2,
    n_heads: int = 8,
    head_dim: int = 64,
    device=None,
):
    """Compare cross-attention (varying tgt_len) vs self-attention latency."""
    import time
    if tgt_lengths is None:
        tgt_lengths = [64, 128, 256, 512]

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nCross-attention vs Self-attention Benchmark")
    print(f"Device: {device} | src_len={src_len} | batch={batch_size}")
    print(f"{'tgt_len':>8} | {'Cross-attn (ms)':>16} | {'Self-attn (ms)':>15} | {'Ratio':>8}")
    print("-" * 58)

    for tgt_len in tgt_lengths:
        q = torch.randn(batch_size, n_heads, tgt_len, head_dim, device=device)
        k_enc = torch.randn(batch_size, n_heads, src_len, head_dim, device=device)
        v_enc = torch.randn(batch_size, n_heads, src_len, head_dim, device=device)
        k_dec = torch.randn(batch_size, n_heads, tgt_len, head_dim, device=device)
        v_dec = torch.randn(batch_size, n_heads, tgt_len, head_dim, device=device)

        N = 10
        for _ in range(3):
            cross_attention(q, k_enc, v_enc)
            F.scaled_dot_product_attention(q, k_dec, v_dec, is_causal=True)

        def _sync():
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            cross_attention(q, k_enc, v_enc)
        _sync()
        cross_ms = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            F.scaled_dot_product_attention(q, k_dec, v_dec, is_causal=True)
        _sync()
        self_ms = (time.perf_counter() - t0) / N * 1000

        print(f"{tgt_len:>8} | {cross_ms:>16.2f} | {self_ms:>15.2f} | {cross_ms/self_ms:>7.2f}x")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # encoder-decoder: src=512 tokens, tgt=64 tokens
    B, dim, n_heads = 2, 256, 4
    src_len, tgt_len = 512, 64
    encoder_dim = 512

    decoder = CrossAttention(dim=dim, n_heads=n_heads, encoder_dim=encoder_dim).to(device)
    x = torch.randn(B, tgt_len, dim, device=device)
    enc = torch.randn(B, src_len, encoder_dim, device=device)
    # padding: last 100 encoder tokens are padding
    pad_mask = torch.zeros(B, src_len, dtype=torch.bool, device=device)
    pad_mask[:, -100:] = True

    out = decoder(x, enc, src_key_padding_mask=pad_mask)
    assert out.shape == (B, tgt_len, dim)
    print(f"Cross-attention output: {out.shape}  ✓")

    benchmark_cross_vs_self()
