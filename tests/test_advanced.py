"""Tests for advanced modules: speculative decoding, throughput, entropy, IO."""

import math
import pytest
import torch

from entropy import attention_entropy, max_entropy, detect_attention_sinks
from attention_io import vanilla_attention_io_analysis, flash_attention_io_analysis
from memory_analysis import estimate_parameter_memory, activation_memory_table
from attention_score_analysis import score_statistics, temperature_sweep


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class TestEntropy:
    def test_entropy_shape(self):
        B, H, T = 2, 4, 32
        weights = torch.ones(B, H, T, T) / T  # uniform
        ent = attention_entropy(weights)
        assert ent.shape == (B, H, T)

    def test_uniform_max_entropy(self):
        """Uniform attention should achieve maximum entropy."""
        T = 16
        weights = torch.ones(1, 1, T, T) / T
        ent = attention_entropy(weights)
        expected = max_entropy(T)
        assert abs(ent.mean().item() - expected) < 0.01, \
            f"Uniform entropy {ent.mean():.3f} != max {expected:.3f}"

    def test_one_hot_zero_entropy(self):
        """One-hot attention should have ~0 entropy."""
        T = 16
        weights = torch.zeros(1, 1, T, T)
        weights[0, 0, :, 0] = 1.0  # all attention on token 0
        ent = attention_entropy(weights)
        assert ent.mean().item() < 0.01

    def test_max_entropy_formula(self):
        assert abs(max_entropy(1) - 0.0) < 1e-6
        assert abs(max_entropy(math.e) - 1.0) < 0.01  # log(e) = 1
        assert max_entropy(100) > max_entropy(10)

    def test_sink_detection_returns_correct_keys(self):
        B, H, T = 2, 4, 32
        weights = torch.ones(B, H, T, T) / T
        sinks = detect_attention_sinks(weights)
        assert "sink_positions" in sinks
        assert "sink_frequencies" in sinks
        assert "pct_heads_with_sink_at_0" in sinks


class TestIOAnalysis:
    def test_vanilla_io_increases_with_seq_len(self):
        """Vanilla attention IO should grow quadratically with seq_len."""
        r1 = vanilla_attention_io_analysis(1, 4, 256, 64)
        r2 = vanilla_attention_io_analysis(1, 4, 512, 64)
        # IO grows roughly as T^2 for the attention matrix
        assert r2["total_io_gb"] > r1["total_io_gb"]
        ratio = r2["attn_matrix_gb"] / r1["attn_matrix_gb"]
        assert abs(ratio - 4.0) < 0.5, f"Expected ~4x ratio, got {ratio:.2f}"

    def test_flash_attn_no_matrix_in_hbm(self):
        """Flash attention should have 0 attention matrix in HBM."""
        r = flash_attention_io_analysis(2, 8, 1024, 64)
        assert r["attn_matrix_gb"] == 0.0

    def test_flash_less_io_than_vanilla(self):
        """Flash attention should use less total IO for long sequences."""
        for T in [512, 1024, 2048]:
            van = vanilla_attention_io_analysis(2, 8, T, 64)
            fa = flash_attention_io_analysis(2, 8, T, 64)
            assert fa["total_io_gb"] < van["total_io_gb"], \
                f"Flash should use less IO at seq_len={T}"

    def test_arithmetic_intensity_grows_with_seq_len(self):
        """Flash attention AI should grow as seq_len increases (more compute-bound)."""
        r1 = flash_attention_io_analysis(1, 4, 256, 64)
        r2 = flash_attention_io_analysis(1, 4, 1024, 64)
        # more work per byte as seq grows
        assert r2["arithmetic_intensity"] >= r1["arithmetic_intensity"]

    def test_same_flops_both_methods(self):
        """Both vanilla and flash attention do the same amount of compute."""
        for T in [256, 512, 1024]:
            van = vanilla_attention_io_analysis(2, 4, T, 64)
            fa = flash_attention_io_analysis(2, 4, T, 64)
            assert van["flops"] == fa["flops"], \
                f"FLOPs should match at seq_len={T}"


class TestMemoryAnalysis:
    def test_parameter_memory_scales_with_layers(self):
        m1 = estimate_parameter_memory(512, 8, n_layers=1)
        m12 = estimate_parameter_memory(512, 8, n_layers=12)
        assert abs(m12["total_params"] / m1["total_params"] - 12) < 0.1

    def test_grad_equals_param_memory(self):
        m = estimate_parameter_memory(512, 8, n_layers=4)
        assert abs(m["grad_mb"] - m["param_mb"]) < 0.01

    def test_adam_double_param(self):
        m = estimate_parameter_memory(512, 8, n_layers=1)
        # Adam states should be 2x param size
        assert abs(m["adam_states_mb"] / m["param_mb"] - 2.0) < 0.01

    def test_activation_table_returns_results(self):
        results = activation_memory_table(seq_lengths=[128, 256])
        assert len(results) == 2
        for r in results:
            assert "seq_len" in r
            assert "vanilla_attn_mb" in r
            assert "total_vanilla_mb" in r

    def test_flash_zero_attn_matrix(self):
        results = activation_memory_table(seq_lengths=[512])
        assert results[0]["flash_attn_mb"] == 0.0

    def test_vanilla_attn_scales_quadratically(self):
        results = activation_memory_table(seq_lengths=[256, 512])
        r1, r2 = results[0], results[1]
        ratio = r2["vanilla_attn_mb"] / r1["vanilla_attn_mb"]
        # should be 4x (2^2) since seq_len doubled
        assert abs(ratio - 4.0) < 0.1


class TestScoreAnalysis:
    def test_score_statistics_returns_all_keys(self):
        scores = torch.randn(2, 4, 32, 32)
        stats = score_statistics(scores)
        for key in ["mean", "std", "min", "max", "q25", "q75"]:
            assert key in stats

    def test_scaling_reduces_std(self):
        """Dividing by sqrt(d) should reduce std."""
        import math
        D = 64
        scores = torch.randn(1, 1, 32, 32) * math.sqrt(D)
        scaled = scores / math.sqrt(D)
        assert scores.std().item() > scaled.std().item()

    def test_temperature_sweep_no_errors(self):
        """Temperature sweep should complete without errors."""
        try:
            temperature_sweep(seq_len=16, n_heads=2, head_dim=16, temperatures=[0.5, 1.0, 2.0])
        except Exception as e:
            pytest.fail(f"temperature_sweep raised {e}")
