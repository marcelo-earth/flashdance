"""Long context attention: challenges and solutions.

As context grows beyond training length, attention faces two problems:
1. Memory: vanilla attention needs T^2 memory
2. Quality: position encodings trained on short sequences may not generalize

Solutions:
- Flash Attention: solves the memory problem (IO-efficient)
- RoPE with NTK scaling / YaRN: extends effective context
- Sliding Window Attention: trades quality for memory (Mistral)
- ALiBi: generalizes well beyond training length

This module benchmarks attention at very long sequences and shows
which variants can handle them efficiently.
"""

import math
import gc
import time
import torch
import torch.nn.functional as F

from rope import precompute_freqs, apply_rope
from sliding_window import sliding_window_attention
from alibi import build_alibi_bias, alibi_attention


def ntk_scaled_rope(head_dim: int, seq_len: int, base: float = 10000.0, scale: float = 1.0, device=None):
    """RoPE with NTK-aware scaling for long context extension.

    When extending context beyond training length T_train to T_test:
    scale = (T_test / T_train) ** (head_dim / (head_dim - 2))
    New base = old_base * scale

    Reference: "Extending Context Window of Large Language Models via Positional Interpolation"
    """
    scaled_base = base * scale
    half = head_dim // 2
    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def pi_scaled_rope(head_dim: int, seq_len: int, train_len: int = 2048, base: float = 10000.0, device=None):
    """Position Interpolation (PI) for RoPE.

    Instead of using positions [0, ..., T-1], use [0, ..., T-1] * (T_train / T_test).
    This maps long sequences back to the training range.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    # scale positions down
    scale = train_len / seq_len if seq_len > train_len else 1.0
    positions = torch.arange(seq_len, device=device).float() * scale
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def benchmark_long_context(
    seq_lengths=None,
    batch_size: int = 1,
    n_heads: int = 8,
    head_dim: int = 64,
    window_size: int = 512,
    device=None,
    repeats: int = 3,
):
    """Benchmark attention variants at long sequence lengths."""
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def timed(fn, *args):
        for _ in range(1):
            fn(*args)
        _sync()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(*args)
            _sync()
            times.append(time.perf_counter() - t0)
        return sorted(times)[len(times) // 2] * 1000

    print(f"\nLong Context Benchmark")
    print(f"Device: {device} | batch={batch_size} | heads={n_heads} | head_dim={head_dim}")
    print(f"(window_size={window_size} for SWA, NTK scale computed per seq_len)")
    print()
    print(f"{'seq_len':>8} | {'SDPA (ms)':>10} | {'RoPE+SDPA':>10} | {'NTK-RoPE':>10} | {'SWA (ms)':>10} | {'ALiBi (ms)':>11}")
    print("-" * 75)

    results = []

    for seq_len in seq_lengths:
        try:
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

            # standard SDPA
            sdpa_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

            # RoPE + SDPA
            cos, sin = precompute_freqs(head_dim, seq_len, device=device)
            q_r, k_r = apply_rope(q, k, cos, sin)
            rope_ms = timed(lambda: F.scaled_dot_product_attention(q_r, k_r, v, is_causal=True))

            # NTK-scaled RoPE (for context extension)
            train_len = 2048
            ntk_scale = (seq_len / train_len) ** (head_dim / (head_dim - 2)) if seq_len > train_len else 1.0
            cos_ntk, sin_ntk = ntk_scaled_rope(head_dim, seq_len, scale=ntk_scale, device=device)
            q_ntk, k_ntk = apply_rope(q, k, cos_ntk, sin_ntk)
            ntk_ms = timed(lambda: F.scaled_dot_product_attention(q_ntk, k_ntk, v, is_causal=True))

            # Sliding Window
            swa_ms = timed(lambda: sliding_window_attention(q, k, v, window_size=window_size))

            # ALiBi
            alibi_bias = build_alibi_bias(n_heads, seq_len, device=device)
            alibi_ms = timed(lambda: alibi_attention(q, k, v, alibi_bias=alibi_bias, causal=False))

            print(f"{seq_len:>8} | {sdpa_ms:>10.1f} | {rope_ms:>10.1f} | {ntk_ms:>10.1f} | {swa_ms:>10.1f} | {alibi_ms:>11.1f}")

            results.append({
                "seq_len": seq_len,
                "sdpa_ms": sdpa_ms,
                "rope_ms": rope_ms,
                "ntk_ms": ntk_ms,
                "swa_ms": swa_ms,
                "alibi_ms": alibi_ms,
            })

            del q, k, v, q_r, k_r, q_ntk, k_ntk, alibi_bias
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:>8} | OOM")
            else:
                raise

    return results


def context_extension_strategies():
    """Explain strategies for extending LLM context beyond training length."""
    strategies = {
        "Position Interpolation (PI)": {
            "idea": "Scale positions to fit training range: pos_new = pos * (T_train / T_test)",
            "tradeoff": "Slight quality drop, needs fine-tuning",
            "used_by": "LLaMA 2 Long, Code Llama",
        },
        "NTK-aware scaling": {
            "idea": "Scale RoPE base: new_base = old_base * scale^(d/(d-2))",
            "tradeoff": "No fine-tuning needed (dynamic), moderate quality",
            "used_by": "Many community models",
        },
        "YaRN": {
            "idea": "Combination of PI + NTK + magnitude scaling",
            "tradeoff": "Best quality, minimal fine-tuning",
            "used_by": "Mistral 7B (extended), CodeBooga",
        },
        "Sliding Window (SWA)": {
            "idea": "Each token only attends to W nearest tokens",
            "tradeoff": "O(n*W) memory, loses long-range dependencies",
            "used_by": "Mistral 7B, Falcon",
        },
        "ALiBi": {
            "idea": "Train with linear bias, generalizes to long context for free",
            "tradeoff": "Must train with ALiBi from scratch",
            "used_by": "BLOOM, MPT, MPT-30B-chat",
        },
    }

    print("\nContext Extension Strategies:")
    print("=" * 70)
    for name, info in strategies.items():
        print(f"\n{name}:")
        for k, v in info.items():
            print(f"  {k:<12}: {v}")


if __name__ == "__main__":
    results = benchmark_long_context()
    context_extension_strategies()
