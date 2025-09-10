"""Tests for GQA, KV cache, and MLA."""

import pytest
import torch

from kv_cache import KVCache, AttentionWithKVCache
from mla import MultiHeadLatentAttention, compare_kv_cache_sizes
from rope import precompute_freqs


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestKVCache:
    def test_cache_initialization(self):
        cache = KVCache(n_layers=2, n_kv_heads=4, max_seq_len=128, head_dim=64, device=DEVICE)
        assert cache.pos == 0
        assert cache.k_cache.shape == (2, 4, 128, 64)

    def test_cache_update_advances_position(self):
        cache = KVCache(n_layers=1, n_kv_heads=2, max_seq_len=64, head_dim=32, device=DEVICE)
        k = torch.randn(1, 2, 4, 32, device=DEVICE)
        v = torch.randn(1, 2, 4, 32, device=DEVICE)
        k_full, v_full = cache.update(0, k, v)
        cache.advance(4)
        assert cache.pos == 4
        assert k_full.shape == (1, 2, 4, 32)

    def test_cache_accumulates(self):
        """KV cache should grow with each decode step."""
        cache = KVCache(n_layers=1, n_kv_heads=2, max_seq_len=32, head_dim=16, device=DEVICE)
        for step in range(5):
            k = torch.randn(1, 2, 1, 16, device=DEVICE)
            v = torch.randn(1, 2, 1, 16, device=DEVICE)
            k_full, _ = cache.update(0, k, v)
            cache.advance(1)
            assert k_full.shape[2] == step + 1, f"Cache length should be {step+1}"

    def test_cache_reset(self):
        cache = KVCache(n_layers=1, n_kv_heads=2, max_seq_len=32, head_dim=16, device=DEVICE)
        cache.pos = 10
        cache.reset()
        assert cache.pos == 0
        assert cache.k_cache.abs().sum() == 0

    def test_cache_memory(self):
        cache = KVCache(n_layers=4, n_kv_heads=4, max_seq_len=512, head_dim=64, device=DEVICE)
        mem = cache.memory_mb()
        assert mem > 0
        # should be 4 * 4 * 512 * 64 * 2 * 2 bytes (float16) / 1e6
        expected_mb = 4 * 4 * 512 * 64 * 2 * 2 / 1024**2
        assert abs(mem - expected_mb) < 0.1

    def test_attention_with_cache_shapes(self):
        dim, n_heads, n_kv = 128, 4, 2
        head_dim = dim // n_heads
        layer = AttentionWithKVCache(dim, n_heads, n_kv).to(DEVICE)
        cache = KVCache(1, n_kv, 64, head_dim, device=DEVICE)
        cos, sin = precompute_freqs(head_dim, 64, device=DEVICE)

        # prefill
        x = torch.randn(1, 16, dim, device=DEVICE)
        out = layer(x, cos[:16], sin[:16], cache=cache, layer_idx=0)
        assert out.shape == (1, 16, dim)

        # decode
        x_tok = torch.randn(1, 1, dim, device=DEVICE)
        out_tok = layer(x_tok, cos[16:17], sin[16:17], cache=cache, layer_idx=0)
        assert out_tok.shape == (1, 1, dim)


class TestMLAMemory:
    def test_mla_cache_smaller_than_mha(self):
        for d_c_ratio in [0.1, 0.25, 0.5]:
            dim, n_heads = 512, 8
            d_c = int(n_heads * (dim // n_heads) * d_c_ratio)
            model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c)
            seq_len = 2048
            assert model.kv_cache_size_bytes(seq_len) < model.mha_kv_cache_size_bytes(seq_len)

    def test_compare_kv_cache_sizes_returns_reduction(self):
        result = compare_kv_cache_sizes(
            dim=1024, n_heads=16, d_c=128, seq_len=512, n_layers=4
        )
        assert result["reduction"] > 1.0
        assert result["mla_gb"] < result["mha_gb"]

    def test_mla_forward_no_rope(self):
        dim, n_heads, d_c = 256, 4, 32
        model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c).to(DEVICE)
        x = torch.randn(2, 16, dim, device=DEVICE)
        out = model(x)  # no rope
        assert out.shape == (2, 16, dim)

    def test_mla_forward_with_rope(self):
        dim, n_heads, d_c = 256, 4, 32
        head_dim = dim // n_heads
        model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c).to(DEVICE)
        x = torch.randn(2, 16, dim, device=DEVICE)
        cos, sin = precompute_freqs(head_dim, 16, device=DEVICE)
        out = model(x, cos=cos, sin=sin)
        assert out.shape == (2, 16, dim)

    def test_mla_with_q_compression(self):
        dim, n_heads, d_c, d_c_q = 256, 4, 32, 16
        model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c, d_c_q=d_c_q).to(DEVICE)
        x = torch.randn(2, 16, dim, device=DEVICE)
        out = model(x)
        assert out.shape == (2, 16, dim)
