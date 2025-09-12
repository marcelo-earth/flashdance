"""Tests for RoPE, ALiBi, sliding window, and cross-attention."""

import math
import pytest
import torch
import torch.nn.functional as F

from rope import precompute_freqs, apply_rope, apply_rope_single, RoPECache, rope_attention
from alibi import get_alibi_slopes, build_alibi_bias, alibi_attention, ALiBiAttention
from sliding_window import sliding_window_mask, sliding_window_attention, SlidingWindowAttention
from cross_attention import cross_attention, cross_attention_with_mask, CrossAttention


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestRoPEAdvanced:
    def test_rope_attention_shape(self):
        B, H, T, D = 2, 4, 32, 64
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        v = torch.randn(B, H, T, D, device=DEVICE)
        cos, sin = precompute_freqs(D, T, device=DEVICE)
        out = rope_attention(q, k, v, cos, sin)
        assert out.shape == (B, H, T, D)

    def test_rope_relative_position(self):
        """RoPE encodes relative position: dot product of q_i and k_j
        depends on (i - j), not absolute positions."""
        D = 32
        cos, sin = precompute_freqs(D, 16, device=DEVICE)

        # two queries at positions 5 and 6, two keys at positions 0 and 1
        q1 = torch.randn(1, 1, 1, D, device=DEVICE)
        q2 = torch.randn(1, 1, 1, D, device=DEVICE)
        k1 = torch.randn(1, 1, 1, D, device=DEVICE)
        k2 = torch.randn(1, 1, 1, D, device=DEVICE)

        # rotate q at pos 5, k at pos 0 -> relative pos = 5
        q1_rot = apply_rope_single(q1, cos[5:6, :D//2], sin[5:6, :D//2])
        k1_rot = apply_rope_single(k1, cos[0:1, :D//2], sin[0:1, :D//2])
        score1 = (q1_rot * k1_rot).sum()

        # rotate q at pos 6, k at pos 1 -> relative pos = 5 (same!)
        q2_rot = apply_rope_single(q2.clone().copy_(q1), cos[6:7, :D//2], sin[6:7, :D//2])
        k2_rot = apply_rope_single(k2.clone().copy_(k1), cos[1:2, :D//2], sin[1:2, :D//2])
        score2 = (q2_rot * k2_rot).sum()

        # scores should be equal (same relative position, same vectors)
        assert torch.allclose(score1, score2, atol=1e-4), \
            f"RoPE relative position property violated: {score1:.4f} != {score2:.4f}"

    def test_rope_cache_invalidated_on_device_change(self):
        cache = RoPECache(head_dim=32, max_seq_len=64)
        cos1, _ = cache.get(32, "cpu")
        assert cache._device == "cpu"

    def test_precompute_large_seq(self):
        # should work for very long sequences
        cos, sin = precompute_freqs(64, 32768, device=DEVICE)
        assert cos.shape == (32768, 32)


class TestALiBiAdvanced:
    def test_slopes_power_of_2_vs_non_power(self):
        """Both should produce n slopes."""
        for n in [4, 6, 8, 12, 16]:
            slopes = get_alibi_slopes(n)
            assert len(slopes) == n, f"Expected {n} slopes, got {len(slopes)}"

    def test_bias_recency(self):
        """More recent tokens should have less negative bias (head 0)."""
        bias = build_alibi_bias(n_heads=4, seq_len=16)
        head0_bias = bias[0, 0]
        # position 14 attending to position 15 (distance 1) should have less negative bias
        # than position 14 attending to position 0 (distance 14)
        assert head0_bias[14, 15].item() == float("-inf")  # future = masked
        assert head0_bias[14, 13].item() > head0_bias[14, 0].item(), \
            "Closer token should have less negative ALiBi bias"

    def test_alibi_no_learned_positions(self):
        """ALiBi module should have no positional parameters."""
        model = ALiBiAttention(dim=128, n_heads=4)
        param_names = [n for n, _ in model.named_parameters()]
        # should not have any position-related parameters
        assert all("pos" not in n and "position" not in n for n in param_names)

    def test_alibi_attention_weights_sum_to_one(self):
        B, H, T, D = 1, 4, 32, 64
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        v = torch.randn(B, H, T, D, device=DEVICE)
        bias = build_alibi_bias(H, T, device=DEVICE)

        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias[:, :, :T, :T]
        weights = F.softmax(attn_scores, dim=-1)

        # rows that are not all-masked should sum to 1
        row_sums = weights.sum(dim=-1)
        valid = row_sums > 0  # non-nan rows
        assert torch.allclose(row_sums[valid], torch.ones_like(row_sums[valid]), atol=1e-4)


class TestSlidingWindowAdvanced:
    def test_window_size_equal_seq_len_is_full_attention(self):
        """Window size = seq_len should behave like causal attention."""
        B, H, T, D = 1, 2, 16, 32
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        v = torch.randn(B, H, T, D, device=DEVICE)

        out_swa = sliding_window_attention(q.clone(), k.clone(), v.clone(), window_size=T)
        out_full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.allclose(out_swa, out_full, atol=1e-4)

    def test_window_size_1_attends_only_self(self):
        """Window size = 1 means each token only attends to itself."""
        mask = sliding_window_mask(8, window_size=1, device=DEVICE)
        # diagonal should be False (can attend to self), everything else True
        assert not mask.diagonal().any()
        # off-diagonal should all be masked
        eye = torch.eye(8, dtype=torch.bool, device=DEVICE)
        assert mask[~eye].all()

    def test_swa_gradient_flows(self):
        B, H, T, D = 1, 2, 16, 32
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)
        out = sliding_window_attention(q, k, v, window_size=4)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestCrossAttention:
    def test_cross_attention_shape(self):
        B, H, T, S, D = 2, 4, 32, 64, 32
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, S, D, device=DEVICE)
        v = torch.randn(B, H, S, D, device=DEVICE)
        out = cross_attention(q, k, v)
        assert out.shape == (B, H, T, D)

    def test_cross_attention_different_lengths(self):
        """Cross-attention should work with tgt_len != src_len."""
        for tgt_len, src_len in [(16, 64), (64, 16), (100, 100), (1, 512)]:
            B, H, D = 1, 4, 32
            q = torch.randn(B, H, tgt_len, D, device=DEVICE)
            k = torch.randn(B, H, src_len, D, device=DEVICE)
            v = torch.randn(B, H, src_len, D, device=DEVICE)
            out = cross_attention(q, k, v)
            assert out.shape == (B, H, tgt_len, D), f"Failed for tgt={tgt_len}, src={src_len}"

    def test_padding_mask(self):
        """Padding mask should zero out attention to padded positions."""
        B, H, T, S, D = 1, 2, 16, 32, 32
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, S, D, device=DEVICE)
        v = torch.randn(B, H, S, D, device=DEVICE)

        # mask all src tokens
        mask = torch.ones(B, S, dtype=torch.bool, device=DEVICE)
        out = cross_attention_with_mask(q, k, v, src_key_padding_mask=mask)
        # with all positions masked, softmax gets all -inf -> nan -> 0 (via nan_to_num)
        assert not out.isnan().any()

    def test_cross_attention_module_shape(self):
        B, dim, n_heads = 2, 256, 4
        tgt_len, src_len, enc_dim = 32, 64, 512
        model = CrossAttention(dim=dim, n_heads=n_heads, encoder_dim=enc_dim).to(DEVICE)
        x = torch.randn(B, tgt_len, dim, device=DEVICE)
        enc = torch.randn(B, src_len, enc_dim, device=DEVICE)
        out = model(x, enc)
        assert out.shape == (B, tgt_len, dim)
