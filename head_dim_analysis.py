"""Head dimension analysis: why 64 and 128 are common choices.

Head dimension (d_head = d_model / n_heads) affects:
1. Computational efficiency (hardware prefers powers of 2)
2. Flash Attention compatibility (must be <= 256 on most GPUs)
3. Representation capacity per head
4. Memory per head in KV cache

Common configs in production models:
- GPT-2: d_head = 64 (dim=768, heads=12)
- LLaMA-7B: d_head = 128 (dim=4096, heads=32)
- LLaMA-70B: d_head = 128 (dim=8192, heads=64)
- DeepSeek-V2: d_head = 128 (dim=5120, heads=128 via MLA)
- Mistral-7B: d_head = 128 (dim=4096, heads=32)

Empirically, d_head=64 and d_head=128 are the sweet spots.
"""

import math
import time
import gc
import torch
import torch.nn.functional as F


def benchmark_head_dims_detailed(
    head_dims=None,
    seq_lengths=None,
    n_heads_total_dim: int = 512,  # n_heads * head_dim = constant model dim
    batch_size: int = 4,
    device=None,
    repeats: int = 15,
):
    """Sweep head_dim while keeping total model dim constant.

    This shows the actual throughput impact of head_dim choice.
    """
    if head_dims is None:
        head_dims = [32, 64, 128, 256]
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024]

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

    print(f"\nHead Dimension Analysis")
    print(f"Device: {device} | total_dim={n_heads_total_dim} | batch={batch_size}")
    print(f"(n_heads = total_dim / head_dim, all configs have same total params)")
    print()

    results = []

    for seq_len in seq_lengths:
        print(f"seq_len={seq_len}")
        print(f"  {'head_dim':>10} | {'n_heads':>8} | {'SDPA (ms)':>10} | {'Vanilla (ms)':>13} | {'Speedup':>8}")
        print(f"  {'-'*10} | {'-'*8} | {'-'*10} | {'-'*13} | {'-'*8}")

        for hd in head_dims:
            if n_heads_total_dim % hd != 0:
                continue
            n_heads = n_heads_total_dim // hd

            q = torch.randn(batch_size, n_heads, seq_len, hd, device=device)
            k = torch.randn(batch_size, n_heads, seq_len, hd, device=device)
            v = torch.randn(batch_size, n_heads, seq_len, hd, device=device)

            # SDPA
            for _ in range(3):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            _sync()

            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
                _sync()
                times.append(time.perf_counter() - t0)
            times.sort()
            sdpa_ms = times[len(times) // 2] * 1000

            # Vanilla
            from attention import vanilla_attention
            for _ in range(3):
                vanilla_attention(q, k, v)
            _sync()

            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                vanilla_attention(q, k, v)
                _sync()
                times.append(time.perf_counter() - t0)
            times.sort()
            vanilla_ms = times[len(times) // 2] * 1000
            speedup = vanilla_ms / sdpa_ms

            print(f"  {hd:>10} | {n_heads:>8} | {sdpa_ms:>10.2f} | {vanilla_ms:>13.2f} | {speedup:>7.2f}x")
            results.append({
                "seq_len": seq_len,
                "head_dim": hd,
                "n_heads": n_heads,
                "sdpa_ms": sdpa_ms,
                "vanilla_ms": vanilla_ms,
                "speedup": speedup,
            })

            del q, k, v
            gc.collect()
        print()

    return results


def model_config_comparison():
    """Compare KV cache and parameter counts for common model configs."""
    print("\nCommon Model Configurations: Head Dimension Analysis")
    print("-" * 80)
    print(f"{'Model':>20} | {'dim':>6} | {'n_heads':>8} | {'head_dim':>9} | {'KV/layer (seq=4K, fp16)':>24}")
    print("-" * 80)

    configs = [
        ("GPT-2 small", 768, 12),
        ("GPT-2 large", 1280, 20),
        ("GPT-2 XL", 1600, 25),
        ("LLaMA-7B", 4096, 32),
        ("LLaMA-13B", 5120, 40),
        ("LLaMA-70B", 8192, 64),
        ("Mistral-7B", 4096, 32),
        ("DeepSeek-V2 attn", 5120, 128),
    ]

    seq_len = 4096
    bpe = 2  # float16

    for name, dim, n_heads in configs:
        head_dim = dim // n_heads
        # KV cache per layer: 2 * n_heads * seq_len * head_dim * bpe
        kv_mb = 2 * n_heads * seq_len * head_dim * bpe / 1024**2
        print(f"{name:>20} | {dim:>6} | {n_heads:>8} | {head_dim:>9} | {kv_mb:>22.1f} MB")


if __name__ == "__main__":
    benchmark_head_dims_detailed()
    model_config_comparison()
