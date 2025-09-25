"""Mini transformer: put it all together.

A complete decoder-only transformer with:
- GQA (grouped query attention)
- RoPE positional embeddings
- RMSNorm (as used by LLaMA, Mistral, DeepSeek)
- SwiGLU FFN (as used by LLaMA)
- Optional sliding window attention

This is not a production model, but a clean educational implementation
that combines all the attention techniques from this project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from rope import precompute_freqs, apply_rope
from gqa import GroupedQueryAttention, repeat_kv
from kv_cache import KVCache


@dataclass
class MiniTransformerConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA
    ffn_dim: int = 1344          # ~2.6x dim (SwiGLU uses less than 4x)
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self):
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by LLaMA, Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN: FFN(x) = (xW1 * sigmoid(xW1)) * xW3 @ W2.

    Used by LLaMA, PaLM, Mistral. More parameter-efficient than GELU FFN.
    Usually ffn_dim = 2/3 * 4 * dim to keep params same as vanilla FFN.
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)  # gate
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block: pre-norm, GQA, SwiGLU FFN."""

    def __init__(self, cfg: MiniTransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.attn = GroupedQueryAttention(
            dim=cfg.dim,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout,
        )
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_dim, dropout=cfg.dropout)

    def forward(self, x, cos, sin, causal=True):
        x = x + self.attn(self.norm1(x), cos=cos, sin=sin, causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """Decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU.

    Architecture inspired by LLaMA 2 / Mistral.
    """

    def __init__(self, cfg: MiniTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim)

        if cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight  # weight tying
        else:
            self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # precompute RoPE
        self._rope_cos = None
        self._rope_sin = None

    def _get_rope(self, seq_len: int, device):
        if self._rope_cos is None or self._rope_cos.shape[0] < seq_len:
            cos, sin = precompute_freqs(self.cfg.head_dim, self.cfg.max_seq_len,
                                        self.cfg.rope_base, device)
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:seq_len].to(device), self._rope_sin[:seq_len].to(device)

    def forward(self, input_ids: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) -- token IDs
            causal: apply causal mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)  # (B, T, dim)

        cos, sin = self._get_rope(T, input_ids.device)

        for layer in self.layers:
            x = layer(x, cos, sin, causal=causal)

        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self) -> dict:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.embed.parameters())
        attn = sum(p.numel() for layer in self.layers for p in layer.attn.parameters())
        ffn = sum(p.numel() for layer in self.layers for p in layer.ffn.parameters())
        norms = sum(p.numel() for layer in self.layers
                   for p in list(layer.norm1.parameters()) + list(layer.norm2.parameters()))

        return {
            "total": total,
            "embedding": embed,
            "attention": attn,
            "ffn": ffn,
            "norms": norms,
            "total_B": total / 1e9,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple top-k sampling generation (no KV cache for clarity)."""
        for _ in range(max_new_tokens):
            logits = self(input_ids)[:, -1, :]  # last token logits
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_model_configs():
    """Standard model size configs inspired by LLaMA / Mistral."""
    return {
        "mini (1M)": MiniTransformerConfig(
            vocab_size=4096, dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
            ffn_dim=336, max_seq_len=512,
        ),
        "small (10M)": MiniTransformerConfig(
            vocab_size=16000, dim=256, n_layers=4, n_heads=8, n_kv_heads=2,
            ffn_dim=672, max_seq_len=1024,
        ),
        "medium (85M)": MiniTransformerConfig(
            vocab_size=32000, dim=512, n_layers=8, n_heads=8, n_kv_heads=2,
            ffn_dim=1344, max_seq_len=2048,
        ),
    }


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    for name, cfg in create_model_configs().items():
        model = MiniTransformer(cfg).to(device)
        params = model.count_params()

        print(f"\n{name}")
        print(f"  Total params: {params['total']:,} ({params['total_B']:.3f}B)")
        print(f"  Embedding: {params['embedding']:,}")
        print(f"  Attention: {params['attention']:,}")
        print(f"  FFN:       {params['ffn']:,}")

        # forward pass
        B, T = 2, min(64, cfg.max_seq_len)
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        logits = model(ids)
        assert logits.shape == (B, T, cfg.vocab_size)
        print(f"  Forward pass: {logits.shape}  ✓")

        # quick generation test
        seed_ids = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
        out_ids = model.generate(seed_ids, max_new_tokens=5)
        assert out_ids.shape[1] == 9
        print(f"  Generation: {out_ids.shape}  ✓")
