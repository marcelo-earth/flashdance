"""IO-bound vs compute-bound analysis for attention.

Flash Attention's key insight: vanilla attention is memory-bandwidth bound,
not compute bound. By tiling operations to stay in SRAM, Flash Attention
reduces HBM reads/writes and achieves better hardware utilization.

This module analyzes the theoretical IO cost of attention operations
and estimates when Flash Attention should help the most.

Key concepts:
- Arithmetic Intensity (AI) = FLOPs / bytes_accessed
- Memory-bound ops: AI < hardware roofline threshold
- Compute-bound ops: AI > hardware roofline threshold
- Roofline = peak_flops / peak_bandwidth

Typical GPU rooflines:
- A100: 312 TFLOP/s FP16, 2000 GB/s HBM -> roofline = 156 FLOP/byte
- RTX 3090: 35.6 TFLOP/s, 936 GB/s -> roofline = 38 FLOP/byte
- Apple M2 Pro: ~7 TFLOP/s, 200 GB/s -> roofline = 35 FLOP/byte
"""

import math
import torch


def vanilla_attention_io_analysis(
    batch: int, n_heads: int, seq_len: int, head_dim: int, dtype_bytes: int = 2
):
    """Compute theoretical IO bytes for vanilla attention.

    Vanilla attention reads Q, K, V and writes the full NxN attention matrix.
    """
    B, H, T, D = batch, n_heads, seq_len, head_dim
    elem = dtype_bytes

    # reads: Q (B*H*T*D), K (B*H*T*D), V (B*H*T*D)
    reads_qkv = 3 * B * H * T * D * elem

    # writes: attention matrix (B*H*T*T), output (B*H*T*D)
    write_attn = B * H * T * T * elem
    write_out = B * H * T * D * elem

    # re-reads attn matrix for softmax
    read_attn = B * H * T * T * elem

    total_bytes = reads_qkv + write_attn + write_out + read_attn

    # FLOPs: QK^T + softmax + weights*V
    flops = 4 * B * H * T * T * D

    ai = flops / total_bytes
    return {
        "reads_gb": reads_qkv / 1e9,
        "attn_matrix_gb": (write_attn + read_attn) / 1e9,
        "total_io_gb": total_bytes / 1e9,
        "flops": flops,
        "arithmetic_intensity": ai,
    }


def flash_attention_io_analysis(
    batch: int, n_heads: int, seq_len: int, head_dim: int,
    block_size: int = 64, dtype_bytes: int = 2
):
    """Compute theoretical IO bytes for Flash Attention.

    Flash Attention tiles the computation to avoid materializing the full NxN
    attention matrix in HBM. It only reads/writes O(T) blocks at a time.
    """
    B, H, T, D = batch, n_heads, seq_len, head_dim
    elem = dtype_bytes
    num_blocks = math.ceil(T / block_size)

    # Q is read once per block pair: O(T^2 / block_size^2) * block_size * D * H * B
    # but each element of Q is read num_blocks times (once per K block)
    reads_q = B * H * T * D * num_blocks * elem  # Q tiled
    reads_kv = 2 * B * H * T * D * num_blocks * elem  # K,V tiled

    # no full attention matrix in HBM (only in SRAM)
    # only the output is written
    write_out = B * H * T * D * elem

    total_bytes = reads_q + reads_kv + write_out

    flops = 4 * B * H * T * T * D  # same FLOPs

    ai = flops / total_bytes
    return {
        "reads_gb": (reads_q + reads_kv) / 1e9,
        "attn_matrix_gb": 0.0,  # never in HBM!
        "total_io_gb": total_bytes / 1e9,
        "flops": flops,
        "arithmetic_intensity": ai,
        "num_blocks": num_blocks,
    }


def io_comparison_table(
    seq_lengths=None,
    batch=4, n_heads=8, head_dim=64,
    hardware="A100",
):
    """Print a comparison of vanilla vs flash attention IO characteristics."""
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 4096, 8192]

    # hardware rooflines (FLOP/byte)
    rooflines = {
        "A100": 156,         # 312 TFLOP/s FP16 / 2000 GB/s
        "RTX 3090": 38,      # 35.6 TFLOP/s / 936 GB/s
        "RTX 4090": 82,      # 165.2 TFLOP/s / 1008 GB/s (BF16)
        "M2 Pro": 35,        # ~7 TFLOP/s / 200 GB/s (rough)
        "M3 Max": 60,        # ~14 TFLOP/s / 300 GB/s (rough)
    }
    roofline = rooflines.get(hardware, 100)

    print(f"\nIO Analysis: Vanilla vs Flash Attention")
    print(f"Hardware: {hardware} (roofline = {roofline} FLOP/byte)")
    print(f"Batch={batch} | Heads={n_heads} | HeadDim={head_dim}")
    print()
    print(f"{'seq_len':>8} | {'Van IO(GB)':>11} | {'FA IO(GB)':>10} | {'IO reduction':>13} | "
          f"{'Van AI':>8} | {'FA AI':>8} | {'Regime':>12}")
    print("-" * 85)

    results = []
    for T in seq_lengths:
        van = vanilla_attention_io_analysis(batch, n_heads, T, head_dim)
        fa = flash_attention_io_analysis(batch, n_heads, T, head_dim)

        io_reduction = van["total_io_gb"] / fa["total_io_gb"]
        van_ai = van["arithmetic_intensity"]
        fa_ai = fa["arithmetic_intensity"]
        van_regime = "compute" if van_ai > roofline else "memory-bnd"
        fa_regime = "compute" if fa_ai > roofline else "memory-bnd"

        print(f"{T:>8} | {van['total_io_gb']:>11.3f} | {fa['total_io_gb']:>10.3f} | "
              f"{io_reduction:>12.1f}x | {van_ai:>8.1f} | {fa_ai:>8.1f} | {fa_regime:>12}")

        results.append({
            "seq_len": T,
            "van_io_gb": van["total_io_gb"],
            "fa_io_gb": fa["total_io_gb"],
            "io_reduction": io_reduction,
            "van_ai": van_ai,
            "fa_ai": fa_ai,
        })

    return results


def plot_io_analysis(results, save_path="plots/io_analysis.png"):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    seq_lens = [r["seq_len"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # IO bytes
    axes[0].plot(seq_lens, [r["van_io_gb"] for r in results], "o-", label="Vanilla", color="coral")
    axes[0].plot(seq_lens, [r["fa_io_gb"] for r in results], "o-", label="Flash Attn", color="steelblue")
    axes[0].set_xlabel("Sequence length")
    axes[0].set_ylabel("HBM IO (GB)")
    axes[0].set_title("Memory Bandwidth: Vanilla vs Flash")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IO reduction
    axes[1].plot(seq_lens, [r["io_reduction"] for r in results], "o-", color="seagreen")
    axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Sequence length")
    axes[1].set_ylabel("IO reduction factor")
    axes[1].set_title("Flash Attention IO Reduction")
    axes[1].grid(True, alpha=0.3)

    # Arithmetic Intensity
    axes[2].plot(seq_lens, [r["van_ai"] for r in results], "o-", label="Vanilla", color="coral")
    axes[2].plot(seq_lens, [r["fa_ai"] for r in results], "o-", label="Flash Attn", color="steelblue")
    axes[2].axhline(y=100, color="purple", linestyle="--", alpha=0.7, label="A100 roofline (~156)")
    axes[2].axhline(y=38, color="orange", linestyle="--", alpha=0.7, label="RTX 3090 roofline (~38)")
    axes[2].set_xlabel("Sequence length")
    axes[2].set_ylabel("Arithmetic Intensity (FLOP/byte)")
    axes[2].set_title("Arithmetic Intensity\n(above roofline = compute-bound)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Flash Attention IO Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = io_comparison_table()
    plot_io_analysis(results)
