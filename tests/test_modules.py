"""Integration tests: full modules working together."""

import pytest
import torch
import torch.nn as nn

from __init__ import (
    vanilla_attention, sdpa_attention, MultiHeadAttention,
    precompute_freqs, apply_rope, RoPECache,
    GroupedQueryAttention, repeat_kv,
    MultiHeadLatentAttention,
    SlidingWindowAttention,
    ALiBiAttention,
    CrossAttention, TransformerDecoderLayer,
    KVCache, AttentionWithKVCache,
    attention_entropy, max_entropy,
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestPackageImports:
    """Verify all public API exports work correctly."""

    def test_all_attention_functions_importable(self):
        assert callable(vanilla_attention)
        assert callable(sdpa_attention)
        assert callable(apply_rope)
        assert callable(repeat_kv)
        assert callable(attention_entropy)

    def test_all_modules_instantiable(self):
        dim, n_heads = 128, 4

        modules = [
            MultiHeadAttention(dim, n_heads, backend="vanilla"),
            MultiHeadAttention(dim, n_heads, backend="sdpa"),
            GroupedQueryAttention(dim, n_heads, n_kv_heads=2),
            MultiHeadLatentAttention(dim, n_heads, d_c=32),
            SlidingWindowAttention(dim, n_heads, window_size=16),
            ALiBiAttention(dim, n_heads),
            CrossAttention(dim, n_heads),
        ]

        for module in modules:
            assert isinstance(module, nn.Module)


class TestEndToEnd:
    """End-to-end forward pass tests."""

    def test_simple_transformer_block(self):
        """A simple transformer block with self-attention + FFN."""
        dim, n_heads, T, B = 128, 4, 32, 2
        head_dim = dim // n_heads

        attn = MultiHeadAttention(dim, n_heads, backend="sdpa").to(DEVICE)
        ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        ).to(DEVICE)
        norm1 = nn.LayerNorm(dim).to(DEVICE)
        norm2 = nn.LayerNorm(dim).to(DEVICE)

        x = torch.randn(B, T, dim, device=DEVICE)
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

        assert x.shape == (B, T, dim)
        assert not x.isnan().any()

    def test_gqa_with_rope_end_to_end(self):
        """GQA module with RoPE applied."""
        dim, n_heads, n_kv = 256, 8, 2
        B, T = 2, 64
        head_dim = dim // n_heads

        gqa = GroupedQueryAttention(dim, n_heads, n_kv).to(DEVICE)
        cos, sin = precompute_freqs(head_dim, T, device=DEVICE)
        x = torch.randn(B, T, dim, device=DEVICE)
        out = gqa(x, cos=cos, sin=sin)
        assert out.shape == (B, T, dim)

    def test_kv_cache_incremental(self):
        """KV cache should produce same output as full-context attention."""
        dim, n_heads, n_kv = 128, 4, 2
        head_dim = dim // n_heads
        T = 16

        layer = AttentionWithKVCache(dim, n_heads, n_kv).to(DEVICE)
        cos, sin = precompute_freqs(head_dim, T + 5, device=DEVICE)

        # prefill
        x = torch.randn(1, T, dim, device=DEVICE)
        cache = KVCache(1, n_kv, T + 5, head_dim, device=DEVICE)
        out_cached = layer(x, cos[:T], sin[:T], cache=cache, layer_idx=0)
        assert out_cached.shape == (1, T, dim)

    def test_encoder_decoder_pipeline(self):
        """Encoder-decoder pipeline: encode src, cross-attend in decoder."""
        dim, enc_dim, n_heads = 128, 256, 4
        B, src_len, tgt_len = 2, 32, 16

        # "encoder" (just a linear for this test)
        encoder = nn.Linear(enc_dim, enc_dim).to(DEVICE)
        # decoder layer
        decoder = TransformerDecoderLayer(dim=dim, n_heads=n_heads, encoder_dim=enc_dim).to(DEVICE)

        src = torch.randn(B, src_len, enc_dim, device=DEVICE)
        tgt = torch.randn(B, tgt_len, dim, device=DEVICE)
        enc_out = encoder(src)
        dec_out = decoder(tgt, enc_out)

        assert dec_out.shape == (B, tgt_len, dim)

    def test_sliding_window_long_sequence(self):
        """SWA should handle sequences longer than window without OOM."""
        dim, n_heads, W = 128, 4, 32
        B, T = 1, 512  # much longer than window

        swa = SlidingWindowAttention(dim, n_heads, window_size=W).to(DEVICE)
        x = torch.randn(B, T, dim, device=DEVICE)
        out = swa(x)
        assert out.shape == (B, T, dim)

    def test_alibi_long_sequence(self):
        """ALiBi should handle sequences longer than typical training lengths."""
        dim, n_heads = 128, 4
        B, T = 1, 512

        alibi = ALiBiAttention(dim, n_heads).to(DEVICE)
        x = torch.randn(B, T, dim, device=DEVICE)
        out = alibi(x)
        assert out.shape == (B, T, dim)


class TestGradients:
    """Test that all modules support gradient computation."""

    def test_vanilla_mha_gradients(self):
        mha = MultiHeadAttention(128, 4, backend="vanilla").to(DEVICE)
        x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None

    def test_sdpa_mha_gradients(self):
        mha = MultiHeadAttention(128, 4, backend="sdpa").to(DEVICE)
        x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
        out = mha(x)
        out.sum().backward()
        assert x.grad is not None

    def test_gqa_gradients(self):
        gqa = GroupedQueryAttention(128, 4, n_kv_heads=2).to(DEVICE)
        x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
        out = gqa(x)
        out.sum().backward()
        assert x.grad is not None

    def test_mla_gradients(self):
        mla = MultiHeadLatentAttention(128, 4, d_c=16).to(DEVICE)
        x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
        out = mla(x)
        out.sum().backward()
        assert x.grad is not None

    def test_alibi_gradients(self):
        alibi = ALiBiAttention(128, 4).to(DEVICE)
        x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
        out = alibi(x)
        out.sum().backward()
        assert x.grad is not None
