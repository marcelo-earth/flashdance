"""Rotary Position Embeddings (RoPE) -- used by LLaMA, Mistral, DeepSeek."""

import math
import torch


def precompute_freqs(head_dim: int, seq_len: int, base: float = 10000.0, device=None):
    """Precompute the cos/sin rotation matrices for RoPE.

    Returns:
        cos, sin: (seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    # theta_i = base^(-2i/d) for i in [0, half)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, half)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rotate_half(x):
    """Split last dim in half and rotate: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (batch, heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim // 2) -- from precompute_freqs

    Returns:
        q_rot, k_rot: same shape as input
    """
    # expand cos/sin to (1, 1, seq_len, head_dim // 2) then cat to head_dim
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # tile to full head_dim: [cos, cos]
    cos = torch.cat([cos, cos], dim=-1)  # (1, 1, T, head_dim)
    sin = torch.cat([sin, sin], dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def apply_rope_single(x, cos, sin):
    """Apply RoPE to a single q or k tensor."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return x * cos + rotate_half(x) * sin


class RoPECache:
    """Cache cos/sin tables to avoid recomputation."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._cos = None
        self._sin = None
        self._device = None

    def get(self, seq_len: int, device):
        if self._cos is None or self._device != device or seq_len > self._cos.shape[0]:
            max_len = max(seq_len, self.max_seq_len)
            cos, sin = precompute_freqs(self.head_dim, max_len, self.base, device)
            self._cos = cos
            self._sin = sin
            self._device = device
        return self._cos[:seq_len], self._sin[:seq_len]


def rope_attention(q, k, v, cos, sin, causal=True):
    """Scaled dot-product attention with RoPE applied to q and k.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim // 2)
        causal: apply causal mask

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    import torch.nn.functional as F

    q_rot, k_rot = apply_rope(q, k, cos, sin)
    return F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=causal)


if __name__ == "__main__":
    # quick sanity check
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B, H, T, D = 2, 8, 128, 64

    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)

    cos, sin = precompute_freqs(D, T, device=device)
    q_rot, k_rot = apply_rope(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # RoPE should change the vectors
    assert not torch.allclose(q_rot, q)

    print(f"RoPE sanity check passed on {device}")
    print(f"q norm before: {q.norm():.4f}, after: {q_rot.norm():.4f}")
    print("(RoPE is an isometry -- norms should be identical)")
