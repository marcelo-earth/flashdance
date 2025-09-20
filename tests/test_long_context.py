"""Tests for long context modules, speculative decoding, and prefill/decode."""

import math
import pytest
import torch
import torch.nn.functional as F

from long_context import ntk_scaled_rope, pi_scaled_rope
from rope import precompute_freqs, apply_rope
from speculative import simulate_draft_decode
from cross_attention import TransformerDecoderLayer


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestRoPEContextExtension:
    def test_ntk_rope_shape(self):
        cos, sin = ntk_scaled_rope(64, 512, device=DEVICE)
        assert cos.shape == (512, 32)
        assert sin.shape == (512, 32)

    def test_pi_rope_shape(self):
        cos, sin = pi_scaled_rope(64, 512, train_len=256, device=DEVICE)
        assert cos.shape == (512, 32)

    def test_ntk_scale_1_equals_standard_rope(self):
        """NTK with scale=1 should be identical to standard RoPE."""
        D, T = 32, 64
        cos_std, sin_std = precompute_freqs(D, T, device=DEVICE)
        cos_ntk, sin_ntk = ntk_scaled_rope(D, T, scale=1.0, device=DEVICE)
        assert torch.allclose(cos_std, cos_ntk, atol=1e-5)
        assert torch.allclose(sin_std, sin_ntk, atol=1e-5)

    def test_pi_within_training_range_no_scaling(self):
        """PI with seq_len <= train_len should not scale positions."""
        D, T, train = 32, 64, 128  # T < train_len
        cos_std, sin_std = precompute_freqs(D, T, device=DEVICE)
        cos_pi, sin_pi = pi_scaled_rope(D, T, train_len=train, device=DEVICE)
        assert torch.allclose(cos_std, cos_pi, atol=1e-5)

    def test_pi_beyond_training_range_scales(self):
        """PI with seq_len > train_len should scale positions down."""
        D, T, train = 32, 256, 64  # T > train_len
        cos_std, sin_std = precompute_freqs(D, T, device=DEVICE)
        cos_pi, sin_pi = pi_scaled_rope(D, T, train_len=train, device=DEVICE)
        # positions are scaled down, so they should differ from unscaled
        assert not torch.allclose(cos_std[:T], cos_pi, atol=1e-3)

    def test_ntk_rope_preserves_norm(self):
        """NTK-scaled RoPE should still be an isometry (preserve norms)."""
        B, H, T, D = 1, 4, 64, 64
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        cos, sin = ntk_scaled_rope(D, T, scale=2.0, device=DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        # norm should be preserved (RoPE is always an isometry)
        assert torch.allclose(q.norm(), q_rot.norm(), atol=1e-2)


class TestSpeculativeDecoding:
    def test_returns_speedup(self):
        """Speculative decoding simulation should return a speedup metric."""
        result = simulate_draft_decode(
            draft_dim=128, target_dim=256,
            k_draft=2, context_len=64, n_steps=10,
            device=DEVICE
        )
        assert "speedup" in result
        assert result["speedup"] > 0

    def test_speedup_positive(self):
        """Speedup should be > 0 (speculative decode shouldn't be infinitely slow)."""
        result = simulate_draft_decode(
            draft_dim=128, target_dim=256,
            k_draft=4, context_len=64, n_steps=20,
            device=DEVICE
        )
        assert result["spec_tps"] > 0
        assert result["ar_tps"] > 0

    def test_k_draft_params_stored(self):
        result = simulate_draft_decode(k_draft=3, context_len=32, n_steps=6, device=DEVICE)
        assert result["k_draft"] == 3
        assert "accept_rate" in result


class TestTransformerDecoderLayer:
    def test_output_shape(self):
        dim, n_heads, enc_dim = 128, 4, 256
        tgt_len, src_len = 16, 32
        B = 2

        layer = TransformerDecoderLayer(dim=dim, n_heads=n_heads, encoder_dim=enc_dim).to(DEVICE)
        x = torch.randn(B, tgt_len, dim, device=DEVICE)
        enc = torch.randn(B, src_len, enc_dim, device=DEVICE)
        out = layer(x, enc)
        assert out.shape == (B, tgt_len, dim)

    def test_with_padding_mask(self):
        dim, n_heads, enc_dim = 128, 4, 256
        tgt_len, src_len = 16, 32
        B = 2

        layer = TransformerDecoderLayer(dim=dim, n_heads=n_heads, encoder_dim=enc_dim).to(DEVICE)
        x = torch.randn(B, tgt_len, dim, device=DEVICE)
        enc = torch.randn(B, src_len, enc_dim, device=DEVICE)
        pad_mask = torch.zeros(B, src_len, dtype=torch.bool, device=DEVICE)
        pad_mask[:, -8:] = True  # last 8 tokens are padding

        out = layer(x, enc, src_key_padding_mask=pad_mask)
        assert out.shape == (B, tgt_len, dim)
        assert not out.isnan().any()

    def test_residual_connection(self):
        """Input and output should differ (residual updates, not replaces)."""
        dim, n_heads, enc_dim = 64, 2, 128
        layer = TransformerDecoderLayer(dim=dim, n_heads=n_heads, encoder_dim=enc_dim).to(DEVICE)
        x = torch.randn(2, 8, dim, device=DEVICE)
        enc = torch.randn(2, 16, enc_dim, device=DEVICE)
        out = layer(x, enc)
        assert not torch.allclose(x, out), "Residual connection should change the output"
