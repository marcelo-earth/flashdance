"""Tests for attention implementations."""

import math
import pytest
import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention, vanilla_attention_with_scores, MultiHeadAttention
from rope import precompute_freqs, apply_rope, rotate_half, RoPECache
from gqa import grouped_query_attention, multi_query_attention, repeat_kv, GroupedQueryAttention
from sliding_window import sliding_window_mask, sliding_window_attention, SlidingWindowAttention
from alibi import get_alibi_slopes, build_alibi_bias, alibi_attention, ALiBiAttention
from mla import MultiHeadLatentAttention


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── helpers ────────────────────────────────────────────────────────────────

def make_qkv(B=2, H=4, T=32, D=64, device=DEVICE, requires_grad=False):
    q = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad)
    k = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad)
    v = torch.randn(B, H, T, D, device=device, requires_grad=requires_grad)
    return q, k, v


# ─── vanilla attention ───────────────────────────────────────────────────────

class TestVanillaAttention:
    def test_output_shape(self):
        q, k, v = make_qkv()
        out = vanilla_attention(q, k, v)
        assert out.shape == q.shape

    def test_causal_mask_zeros_upper_triangle(self):
        """First token should not attend to future tokens."""
        q, k, v = make_qkv(B=1, H=1, T=8, D=16)
        _, weights = vanilla_attention_with_scores(q, k, v, causal=True)
        # upper triangle (excluding diagonal) should be zero after softmax
        upper = weights[0, 0].triu(diagonal=1)
        assert upper.max().item() < 1e-6, "Causal mask leaks future tokens"

    def test_bidirectional_no_zeros(self):
        """Without causal mask, all weights should be positive."""
        torch.manual_seed(0)
        q, k, v = make_qkv(B=1, H=1, T=8, D=16)
        _, weights = vanilla_attention_with_scores(q, k, v, causal=False)
        assert (weights > 0).all()

    def test_attention_weights_sum_to_one(self):
        q, k, v = make_qkv(B=1, H=2, T=16, D=32)
        _, weights = vanilla_attention_with_scores(q, k, v, causal=True)
        row_sums = weights.sum(dim=-1)  # (B, H, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_scale_factor(self):
        """Attention scale should be 1/sqrt(head_dim)."""
        # if we scale q manually, result should be the same as auto-scaling
        B, H, T, D = 1, 1, 4, 64
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        out1 = vanilla_attention(q, k, v)
        # manually scale q
        q_scaled = q / math.sqrt(D)
        attn = torch.matmul(q_scaled, k.transpose(-2, -1))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        out2 = torch.matmul(F.softmax(attn, dim=-1), v)
        assert torch.allclose(out1, out2, atol=1e-5)


# ─── SDPA attention ─────────────────────────────────────────────────────────

class TestSDPAAttention:
    def test_output_shape(self):
        q, k, v = make_qkv()
        out = sdpa_attention(q, k, v)
        assert out.shape == q.shape

    def test_close_to_vanilla(self):
        """SDPA and vanilla should produce very close results."""
        torch.manual_seed(42)
        q, k, v = make_qkv(T=64)
        out_van = vanilla_attention(q.float(), k.float(), v.float())
        out_sdpa = sdpa_attention(q.float(), k.float(), v.float())
        assert torch.allclose(out_van, out_sdpa, atol=1e-4), \
            f"Max diff: {(out_van - out_sdpa).abs().max():.2e}"


# ─── MultiHeadAttention module ───────────────────────────────────────────────

class TestMultiHeadAttention:
    def test_vanilla_shape(self):
        mha = MultiHeadAttention(dim=256, n_heads=4, backend="vanilla").to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        out = mha(x)
        assert out.shape == x.shape

    def test_sdpa_shape(self):
        mha = MultiHeadAttention(dim=256, n_heads=4, backend="sdpa").to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        out = mha(x)
        assert out.shape == x.shape

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(dim=256, n_heads=4, backend="invalid")

    def test_dim_divisible_by_heads(self):
        with pytest.raises(AssertionError):
            MultiHeadAttention(dim=255, n_heads=4, backend="vanilla")


# ─── RoPE ────────────────────────────────────────────────────────────────────

class TestRoPE:
    def test_precompute_shape(self):
        cos, sin = precompute_freqs(64, 128, device=DEVICE)
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)

    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 16, 64, device=DEVICE)
        assert rotate_half(x).shape == x.shape

    def test_rope_preserves_norm(self):
        """RoPE is an isometry -- it should not change vector norms."""
        q, k, _ = make_qkv(T=64, D=64)
        cos, sin = precompute_freqs(64, 64, device=DEVICE)
        q_rot, k_rot = apply_rope(q, k, cos, sin)
        assert torch.allclose(q.norm(), q_rot.norm(), atol=1e-3), \
            f"Norm changed: {q.norm():.4f} -> {q_rot.norm():.4f}"

    def test_rope_changes_vectors(self):
        """RoPE should actually rotate the vectors."""
        q, k, _ = make_qkv(T=32, D=32)
        cos, sin = precompute_freqs(32, 32, device=DEVICE)
        q_rot, _ = apply_rope(q, k, cos, sin)
        assert not torch.allclose(q, q_rot), "RoPE did not change the vectors"

    def test_rope_cache(self):
        cache = RoPECache(head_dim=64, max_seq_len=512)
        cos1, sin1 = cache.get(128, DEVICE)
        cos2, sin2 = cache.get(128, DEVICE)
        assert cos1 is cos2, "Cache should return the same tensors"

    def test_rope_different_positions_differ(self):
        """Different sequence positions should have different rotations."""
        _, T, D = 1, 16, 32
        cos, sin = precompute_freqs(D, T, device=DEVICE)
        # first and last position should differ
        assert not torch.allclose(cos[0], cos[-1])


# ─── GQA ─────────────────────────────────────────────────────────────────────

class TestGQA:
    def test_repeat_kv_shape(self):
        B, H_kv, T, D = 2, 2, 32, 64
        k = torch.randn(B, H_kv, T, D, device=DEVICE)
        v = torch.randn(B, H_kv, T, D, device=DEVICE)
        k_exp, v_exp = repeat_kv(k, v, n_rep=4)
        assert k_exp.shape == (B, H_kv * 4, T, D)
        assert v_exp.shape == (B, H_kv * 4, T, D)

    def test_repeat_kv_n_rep_1(self):
        B, H, T, D = 2, 4, 32, 64
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        k_out, v_out = repeat_kv(k, v, n_rep=1)
        assert k_out is k  # should be the same tensor

    def test_gqa_output_shape(self):
        B, T, D = 2, 64, 256
        n_heads, n_kv_heads = 8, 2
        model = GroupedQueryAttention(D, n_heads, n_kv_heads).to(DEVICE)
        x = torch.randn(B, T, D, device=DEVICE)
        out = model(x)
        assert out.shape == (B, T, D)

    def test_mha_is_special_case_of_gqa(self):
        """GQA with n_kv_heads == n_heads should be equivalent to MHA."""
        B, H, T, D = 1, 4, 32, 64
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, H, T, D, device=DEVICE)
        v = torch.randn(B, H, T, D, device=DEVICE)
        out_gqa = grouped_query_attention(q, k, v, n_heads=H, n_kv_heads=H)
        out_sdpa = sdpa_attention(q, k, v)
        assert torch.allclose(out_gqa, out_sdpa, atol=1e-5)

    def test_mqa_is_special_case_of_gqa(self):
        """MQA (n_kv=1) should match grouped_query_attention with n_kv=1."""
        B, H, T, D = 1, 4, 32, 64
        q = torch.randn(B, H, T, D, device=DEVICE)
        k = torch.randn(B, 1, T, D, device=DEVICE)
        v = torch.randn(B, 1, T, D, device=DEVICE)
        out_mqa = multi_query_attention(q, k, v)
        out_gqa = grouped_query_attention(q, k, v, n_heads=H, n_kv_heads=1)
        assert torch.allclose(out_mqa, out_gqa, atol=1e-5)

    def test_invalid_head_ratio(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(256, n_heads=8, n_kv_heads=3)


# ─── Sliding Window ──────────────────────────────────────────────────────────

class TestSlidingWindow:
    def test_mask_shape(self):
        mask = sliding_window_mask(32, window_size=8)
        assert mask.shape == (32, 32)

    def test_mask_causal(self):
        """Future positions must always be masked."""
        mask = sliding_window_mask(16, window_size=4, device=DEVICE)
        # upper triangle (excluding diagonal) should be all True (masked)
        upper = torch.triu(mask, diagonal=1)
        assert upper.all(), "Future tokens should be masked"

    def test_mask_window(self):
        """Tokens beyond window_size in the past should be masked."""
        W = 4
        mask = sliding_window_mask(16, window_size=W, device=DEVICE)
        # position 10 should not see position 0 (distance = 10 > W)
        assert mask[10, 0].item() == True
        # position 10 should see position 8 (distance = 2 < W)
        assert mask[10, 8].item() == False

    def test_swa_output_shape(self):
        q, k, v = make_qkv(T=64)
        out = sliding_window_attention(q, k, v, window_size=16)
        assert out.shape == q.shape

    def test_swa_module_shape(self):
        model = SlidingWindowAttention(dim=256, n_heads=4, window_size=16).to(DEVICE)
        x = torch.randn(2, 64, 256, device=DEVICE)
        out = model(x)
        assert out.shape == x.shape


# ─── ALiBi ───────────────────────────────────────────────────────────────────

class TestALiBi:
    def test_slopes_length(self):
        for n in [4, 8, 16]:
            slopes = get_alibi_slopes(n)
            assert len(slopes) == n

    def test_slopes_decreasing(self):
        """Slopes should be decreasing (head 0 has strongest local bias)."""
        slopes = get_alibi_slopes(8)
        assert (slopes[:-1] >= slopes[1:]).all(), "Slopes should be non-increasing"

    def test_slopes_positive(self):
        slopes = get_alibi_slopes(8)
        assert (slopes > 0).all()

    def test_bias_shape(self):
        bias = build_alibi_bias(n_heads=8, seq_len=32)
        assert bias.shape == (1, 8, 32, 32)

    def test_bias_causal(self):
        """Upper triangle of bias should be -inf (causal)."""
        bias = build_alibi_bias(n_heads=4, seq_len=8)
        upper = bias[0, 0].triu(diagonal=1)
        assert (upper == float("-inf")).all()

    def test_alibi_attention_shape(self):
        B, H, T, D = 2, 4, 32, 64
        q, k, v = make_qkv(B=B, H=H, T=T, D=D)
        bias = build_alibi_bias(H, T, device=DEVICE)
        out = alibi_attention(q, k, v, alibi_bias=bias)
        assert out.shape == q.shape

    def test_alibi_module_shape(self):
        model = ALiBiAttention(dim=256, n_heads=4).to(DEVICE)
        x = torch.randn(2, 32, 256, device=DEVICE)
        out = model(x)
        assert out.shape == x.shape


# ─── MLA ─────────────────────────────────────────────────────────────────────

class TestMLA:
    def test_output_shape(self):
        dim, n_heads, d_c = 256, 4, 32
        model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c).to(DEVICE)
        B, T = 2, 32
        x = torch.randn(B, T, dim, device=DEVICE)
        head_dim = dim // n_heads
        cos, sin = precompute_freqs(head_dim, T, device=DEVICE)
        out = model(x, cos=cos, sin=sin)
        assert out.shape == (B, T, dim)

    def test_kv_cache_size_smaller(self):
        """MLA KV cache should be smaller than MHA KV cache."""
        dim, n_heads, d_c = 512, 8, 64
        model = MultiHeadLatentAttention(dim=dim, n_heads=n_heads, d_c=d_c)
        seq_len = 1024
        mha_bytes = model.mha_kv_cache_size_bytes(seq_len)
        mla_bytes = model.kv_cache_size_bytes(seq_len)
        assert mla_bytes < mha_bytes, "MLA cache should be smaller than MHA"
        # d_c << n_heads * head_dim means significant savings
        assert mla_bytes / mha_bytes < 0.5, "MLA should save at least 50% memory"
