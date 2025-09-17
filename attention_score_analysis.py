"""Attention score distribution analysis.

Before softmax, attention scores follow an approximately normal distribution.
After softmax + causal masking, the distribution is very different.

This module analyzes:
- Score distribution before/after scaling
- Effect of temperature on attention concentration
- Score statistics across layers/heads
- Why softmax temperature matters for generation quality
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


def score_statistics(scores: torch.Tensor) -> dict:
    """Compute statistics of attention scores (pre-softmax)."""
    flat = scores.flatten()
    finite_mask = flat.isfinite()
    finite = flat[finite_mask]

    return {
        "mean": finite.mean().item(),
        "std": finite.std().item(),
        "min": finite.min().item(),
        "max": finite.max().item(),
        "q25": finite.quantile(0.25).item(),
        "q75": finite.quantile(0.75).item(),
    }


def analyze_score_distribution(
    seq_len: int = 128,
    n_heads: int = 8,
    head_dim: int = 64,
    batch_size: int = 2,
    device=None,
    seed: int = 42,
):
    """Analyze attention score distributions before and after scaling."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(seed)
    B, H, T, D = batch_size, n_heads, seq_len, head_dim

    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)

    # unscaled scores
    raw = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)

    # scaled (by 1/sqrt(D))
    scale = 1.0 / math.sqrt(D)
    scaled = raw * scale

    # apply causal mask
    mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    scaled_masked = scaled.masked_fill(mask, float("-inf"))

    raw_stats = score_statistics(raw)
    scaled_stats = score_statistics(scaled)

    print(f"\nAttention Score Distribution Analysis")
    print(f"B={B}, H={H}, T={T}, D={D}, scale=1/sqrt({D})={scale:.4f}")
    print()
    print(f"{'Metric':>10} | {'Raw (unscaled)':>15} | {'Scaled (1/sqrt(D))':>18}")
    print("-" * 50)
    for key in ["mean", "std", "min", "max"]:
        print(f"{key:>10} | {raw_stats[key]:>15.4f} | {scaled_stats[key]:>18.4f}")

    print(f"\nKey insight: scaling reduces std from ~{raw_stats['std']:.1f} to ~{scaled_stats['std']:.1f}")
    print(f"Without scaling (std={raw_stats['std']:.1f}), softmax would be too peaked (one-hot).")
    print(f"With scaling (std~=1), softmax produces smoother distributions.")

    return {"raw": raw_stats, "scaled": scaled_stats}


def temperature_sweep(
    seq_len: int = 64,
    n_heads: int = 4,
    head_dim: int = 32,
    temperatures=None,
    device=None,
):
    """Show how temperature (inverse scale) affects attention concentration."""
    if temperatures is None:
        temperatures = [0.25, 0.5, 1.0, 2.0, 4.0]

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(42)
    B, H, T, D = 1, n_heads, seq_len, head_dim
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)

    print(f"\nTemperature Sweep (how scale affects attention concentration)")
    print(f"seq_len={T}, head_dim={D}, base scale=1/sqrt({D})={1/math.sqrt(D):.3f}")
    print()
    print(f"{'temp':>6} | {'eff_scale':>10} | {'mean_entropy':>13} | {'max_weight':>11} | {'interpretation':>20}")
    print("-" * 75)

    raw = torch.matmul(q, k.transpose(-2, -1))
    base_scale = 1.0 / math.sqrt(D)

    for temp in temperatures:
        # temperature * base_scale is the effective scale factor
        scale = temp * base_scale
        scores = raw * scale
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # entropy per row
        eps = 1e-9
        ent = -(weights * torch.log(weights + eps)).sum(dim=-1)  # (B, H, T)
        mean_ent = ent.mean().item()
        max_ent = math.log(T)
        max_weight = weights.max().item()

        if temp < 0.5:
            interp = "very peaked"
        elif temp < 1.0:
            interp = "peaked"
        elif temp == 1.0:
            interp = "standard"
        elif temp < 3.0:
            interp = "more uniform"
        else:
            interp = "very uniform"

        print(f"{temp:>6.2f} | {scale:>10.4f} | {mean_ent:>13.3f} | {max_weight:>11.4f} | {interp:>20}")

    print(f"\nMax possible entropy (uniform): {max_ent:.3f}")


def plot_score_distributions(seq_len=64, n_heads=4, head_dim=32, save_path="plots/score_dist.png"):
    """Plot histograms of attention scores and weights."""
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    torch.manual_seed(42)
    B, H, T, D = 1, n_heads, seq_len, head_dim
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    scale = 1.0 / math.sqrt(D)
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores * scale
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scaled_masked = scaled_scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scaled_masked, dim=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # raw score distribution
    raw_flat = raw_scores.flatten().detach().numpy()
    axes[0].hist(raw_flat, bins=50, color="coral", alpha=0.7, edgecolor="white")
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Raw scores (std={raw_flat.std():.2f})\nWould make softmax too peaked!")
    axes[0].set_xlabel("Score value")
    axes[0].set_ylabel("Count")

    # scaled score distribution
    scaled_flat = scaled_scores.flatten().detach().numpy()
    axes[1].hist(scaled_flat, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_title(f"Scaled scores ÷√{D} (std={scaled_flat.std():.2f})\nMuch better for softmax!")
    axes[1].set_xlabel("Score value")

    # attention weight distribution (after softmax)
    # only look at non-zero weights (non-masked)
    weights_flat = weights.flatten().detach().numpy()
    weights_flat = weights_flat[weights_flat > 1e-6]  # filter near-zero
    axes[2].hist(weights_flat, bins=50, color="seagreen", alpha=0.7, edgecolor="white")
    uniform = 1.0 / T
    axes[2].axvline(x=uniform, color="red", linestyle="--", alpha=0.7, label=f"Uniform (1/{T}={uniform:.3f})")
    axes[2].set_title("Attention weights (post-softmax)\nMost weight on a few tokens")
    axes[2].set_xlabel("Attention weight")
    axes[2].legend()

    plt.suptitle("Attention Score Distributions: Why Scaling Matters", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    analyze_score_distribution()
    temperature_sweep()
    plot_score_distributions()
