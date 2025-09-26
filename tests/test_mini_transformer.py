"""Tests for MiniTransformer and its components."""

import pytest
import torch
import torch.nn as nn

from mini_transformer import (
    MiniTransformerConfig,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    MiniTransformer,
    create_model_configs,
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        assert norm(x).shape == x.shape

    def test_rms_normalization(self):
        """RMSNorm should normalize the RMS to ~1."""
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64) * 10  # large values
        out = norm(x)
        # RMS of normalized output should be close to 1 (if weight=1)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert (rms - 1.0).abs().max() < 0.1

    def test_learnable_weight(self):
        norm = RMSNorm(32)
        assert norm.weight.shape == (32,)
        assert (norm.weight == 1.0).all()  # initialized to ones

    def test_no_bias(self):
        """RMSNorm should not have a bias parameter (unlike LayerNorm)."""
        norm = RMSNorm(32)
        param_names = [n for n, _ in norm.named_parameters()]
        assert "bias" not in param_names


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(dim=128, ffn_dim=336)
        x = torch.randn(2, 16, 128)
        assert ffn(x).shape == x.shape

    def test_three_linear_layers(self):
        """SwiGLU should have exactly 3 linear layers."""
        ffn = SwiGLU(64, 168)
        linears = [m for m in ffn.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 3

    def test_no_bias(self):
        """SwiGLU linear layers should have no bias (like LLaMA)."""
        ffn = SwiGLU(64, 168)
        for linear in [ffn.w1, ffn.w2, ffn.w3]:
            assert linear.bias is None

    def test_parameter_count(self):
        """Parameter count should be 3 * dim * ffn_dim."""
        dim, ffn_dim = 128, 336
        ffn = SwiGLU(dim, ffn_dim)
        params = sum(p.numel() for p in ffn.parameters())
        expected = 3 * dim * ffn_dim
        assert params == expected


class TestTransformerBlock:
    def test_output_shape(self):
        cfg = MiniTransformerConfig(dim=128, n_heads=4, n_kv_heads=2)
        block = TransformerBlock(cfg).to(DEVICE)
        B, T = 2, 16
        x = torch.randn(B, T, cfg.dim, device=DEVICE)
        from rope import precompute_freqs
        cos, sin = precompute_freqs(cfg.head_dim, T, device=DEVICE)
        out = block(x, cos, sin)
        assert out.shape == (B, T, cfg.dim)

    def test_residual_changes_input(self):
        """Residual connection should produce output != input."""
        cfg = MiniTransformerConfig(dim=64, n_heads=2, n_kv_heads=1)
        block = TransformerBlock(cfg)
        x = torch.randn(1, 8, cfg.dim)
        from rope import precompute_freqs
        cos, sin = precompute_freqs(cfg.head_dim, 8)
        out = block(x, cos, sin)
        assert not torch.allclose(x, out)

    def test_gradient_flows(self):
        cfg = MiniTransformerConfig(dim=64, n_heads=2, n_kv_heads=1)
        block = TransformerBlock(cfg)
        x = torch.randn(1, 8, cfg.dim, requires_grad=True)
        from rope import precompute_freqs
        cos, sin = precompute_freqs(cfg.head_dim, 8)
        out = block(x, cos, sin)
        out.sum().backward()
        assert x.grad is not None


class TestMiniTransformer:
    def test_forward_shape(self):
        cfg = MiniTransformerConfig(
            vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
            ffn_dim=168, max_seq_len=128
        )
        model = MiniTransformer(cfg).to(DEVICE)
        B, T = 2, 16
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
        logits = model(ids)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_weight_tying(self):
        """Embedding and lm_head should share weights when tie_embeddings=True."""
        cfg = MiniTransformerConfig(vocab_size=256, dim=64, n_layers=1, n_heads=2,
                                   n_kv_heads=1, ffn_dim=168, tie_embeddings=True)
        model = MiniTransformer(cfg)
        assert model.embed.weight is model.lm_head.weight

    def test_no_weight_tying(self):
        cfg = MiniTransformerConfig(vocab_size=256, dim=64, n_layers=1, n_heads=2,
                                   n_kv_heads=1, ffn_dim=168, tie_embeddings=False)
        model = MiniTransformer(cfg)
        assert model.embed.weight is not model.lm_head.weight

    def test_count_params(self):
        cfg = MiniTransformerConfig(vocab_size=256, dim=64, n_layers=2, n_heads=2,
                                   n_kv_heads=1, ffn_dim=168)
        model = MiniTransformer(cfg)
        params = model.count_params()
        assert params["total"] > 0
        assert params["attention"] > 0
        assert params["ffn"] > 0
        # total should equal sum of components (+/- ties)
        total_check = params["embedding"] + params["attention"] + params["ffn"] + params["norms"]
        # approximately equal (lm_head may be tied)
        assert abs(params["total"] - total_check) / params["total"] < 0.5

    def test_generate_length(self):
        cfg = MiniTransformerConfig(vocab_size=256, dim=64, n_layers=1, n_heads=2,
                                   n_kv_heads=1, ffn_dim=168, max_seq_len=64)
        model = MiniTransformer(cfg).to(DEVICE)
        seed = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)
        out = model.generate(seed, max_new_tokens=8)
        assert out.shape == (1, 12)  # 4 seed + 8 generated

    def test_causal_mask(self):
        """With causal=True, token at position i should not affect position j < i."""
        cfg = MiniTransformerConfig(vocab_size=256, dim=64, n_layers=1, n_heads=2,
                                   n_kv_heads=1, ffn_dim=168, max_seq_len=32)
        model = MiniTransformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))

        with torch.no_grad():
            logits1 = model(ids)
            # change last token
            ids_modified = ids.clone()
            ids_modified[0, -1] = (ids[0, -1] + 1) % cfg.vocab_size
            logits2 = model(ids_modified)

        # logits for first 7 positions should be identical (causal mask)
        assert torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-5), \
            "Causal mask violated: earlier positions affected by later token"

    def test_all_model_configs(self):
        """All preset configs should instantiate and do a forward pass."""
        configs = create_model_configs()
        for name, cfg in configs.items():
            model = MiniTransformer(cfg).to(DEVICE)
            B, T = 1, min(8, cfg.max_seq_len)
            ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
            with torch.no_grad():
                logits = model(ids)
            assert logits.shape == (B, T, cfg.vocab_size), f"Failed for {name}"
