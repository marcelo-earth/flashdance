"""Memory analysis: peak usage, activations, and gradient memory.

Memory breakdown for a transformer layer during training:
1. Parameters: weight matrices (Q, K, V, O projections)
2. Activations: intermediate tensors saved for backward pass
3. Gradients: same size as parameters
4. Optimizer states: Adam needs 2x parameter memory (m and v)

Flash Attention reduces activation memory significantly:
- Vanilla: stores full (B, H, T, T) attention matrix for backward
- Flash: recomputes attention on the fly in backward (no T^2 storage)
"""

import gc
import math
import torch
import torch.nn as nn

from attention import vanilla_attention, sdpa_attention, MultiHeadAttention


def estimate_parameter_memory(dim: int, n_heads: int, n_layers: int = 1, dtype=torch.float32) -> dict:
    """Estimate parameter memory for multi-head attention layers."""
    bpe = torch.finfo(dtype).bits // 8
    # QKV projection: 3 * dim * dim
    # Output projection: dim * dim
    params_per_layer = 4 * dim * dim
    total_params = params_per_layer * n_layers

    return {
        "params_per_layer": params_per_layer,
        "total_params": total_params,
        "param_mb": total_params * bpe / 1024**2,
        "grad_mb": total_params * bpe / 1024**2,  # same size
        "adam_states_mb": total_params * bpe * 2 / 1024**2,  # m + v
        "total_training_mb": total_params * bpe * 4 / 1024**2,  # params + grads + adam
    }


def measure_activation_memory(
    seq_len: int,
    batch_size: int,
    n_heads: int,
    head_dim: int,
    device: str,
    use_sdpa: bool = True,
) -> dict:
    """Measure actual peak memory for forward+backward pass."""
    if not torch.cuda.is_available():
        # estimate analytically for non-CUDA
        B, H, T, D = batch_size, n_heads, seq_len, head_dim
        bytes_per = 4  # float32

        qkv_act = 3 * B * H * T * D * bytes_per
        if use_sdpa:
            # flash attention: no T^2 matrix stored
            attn_act = 0
        else:
            # vanilla: stores full attention matrix + softmax output
            attn_act = 2 * B * H * T * T * bytes_per
        out_act = B * H * T * D * bytes_per

        total = qkv_act + attn_act + out_act
        return {
            "activation_mb": total / 1024**2,
            "attn_matrix_mb": attn_act / 1024**2,
            "estimated": True,
        }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)

    before = torch.cuda.memory_allocated()

    if use_sdpa:
        out = sdpa_attention(q, k, v)
    else:
        out = vanilla_attention(q, k, v)

    after_fwd = torch.cuda.memory_allocated()
    out.sum().backward()
    peak = torch.cuda.max_memory_allocated()

    return {
        "fwd_activation_mb": (after_fwd - before) / 1024**2,
        "peak_mb": peak / 1024**2,
        "estimated": False,
    }


def activation_memory_table(
    seq_lengths=None,
    batch_size: int = 4,
    n_heads: int = 8,
    head_dim: int = 64,
):
    """Compare analytical activation memory for vanilla vs SDPA/Flash."""
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    B, H, D = batch_size, n_heads, head_dim
    bpe = 4  # float32 bytes

    print(f"\nActivation Memory Analysis (analytical estimate)")
    print(f"Batch={B} | Heads={H} | HeadDim={D}")
    print()
    print(f"{'seq_len':>8} | {'QKV acts (MB)':>14} | {'Vanilla attn (MB)':>18} | {'Flash attn (MB)':>16} | {'Reduction':>10}")
    print("-" * 80)

    results = []
    for T in seq_lengths:
        qkv_mb = 3 * B * H * T * D * bpe / 1024**2
        van_attn_mb = 2 * B * H * T * T * bpe / 1024**2  # stores attention matrix
        flash_attn_mb = 0  # no T^2 storage

        reduction = (qkv_mb + van_attn_mb) / (qkv_mb + flash_attn_mb + 1e-6) if flash_attn_mb == 0 else 1.0
        # more useful: memory saved
        mem_saved = van_attn_mb
        total_van = qkv_mb + van_attn_mb
        total_flash = qkv_mb

        print(f"{T:>8} | {qkv_mb:>14.1f} | {van_attn_mb:>18.1f} | {flash_attn_mb:>16.1f} | {total_van/total_flash:>9.2f}x")
        results.append({
            "seq_len": T,
            "qkv_mb": qkv_mb,
            "vanilla_attn_mb": van_attn_mb,
            "flash_attn_mb": flash_attn_mb,
            "total_vanilla_mb": total_van,
            "total_flash_mb": total_flash,
            "reduction": total_van / total_flash,
        })

    return results


def plot_memory_analysis(results, save_path="plots/memory_analysis.png"):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    seq_lens = [r["seq_len"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(seq_lens, [r["total_vanilla_mb"] for r in results], "o-", color="coral", label="Vanilla MHA")
    axes[0].plot(seq_lens, [r["total_flash_mb"] for r in results], "o-", color="steelblue", label="Flash Attention")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("Activation memory (MB)")
    axes[0].set_title("Forward Pass Activation Memory")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].plot(seq_lens, [r["vanilla_attn_mb"] for r in results], "o-", color="coral", label="Attn matrix (vanilla)")
    axes[1].plot(seq_lens, [0] * len(seq_lens), "o-", color="steelblue", label="Attn matrix (flash = 0)")
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("Attention matrix memory (MB)")
    axes[1].set_title("Memory Saved by Flash Attention\n(no T² matrix in HBM)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Memory Analysis: Vanilla vs Flash Attention", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = activation_memory_table()
    plot_memory_analysis(results)

    # also estimate model memory at scale
    print("\n--- Model Memory Estimates ---")
    for dim, n_heads, n_layers, name in [
        (768, 12, 12, "GPT-2 (117M)"),
        (1024, 16, 24, "GPT-2 XL (1.5B)"),
        (4096, 32, 32, "LLaMA-7B"),
        (5120, 40, 40, "LLaMA-13B"),
    ]:
        est = estimate_parameter_memory(dim, n_heads, n_layers)
        print(f"\n{name}:")
        print(f"  Parameters:    {est['total_params']/1e9:.2f}B params  ({est['param_mb']:.0f} MB fp32)")
        print(f"  Training total: ~{est['total_training_mb']:.0f} MB fp32 (params+grads+Adam)")
