"""Visualize attention patterns to understand what flash attention computes."""

import torch
import matplotlib.pyplot as plt
import numpy as np

from attention import vanilla_attention_with_scores


def plot_attention_map(attn_weights, head_idx=0, layer_label="", save_path=None):
    """Plot a single attention head's weights as a heatmap.

    Args:
        attn_weights: (batch, heads, seq_len, seq_len)
        head_idx: which head to visualize
        layer_label: label for the plot title
        save_path: optional path to save the figure
    """
    weights = attn_weights[0, head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"Attention weights {layer_label}(head {head_idx})")
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_heads(attn_weights, max_heads=8, save_path=None):
    """Plot attention maps for all heads in a grid."""
    n_heads = min(attn_weights.shape[1], max_heads)
    cols = 4
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i in range(n_heads):
        weights = attn_weights[0, i].detach().cpu().numpy()
        axes[i].imshow(weights, cmap="viridis", aspect="auto")
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key")
        axes[i].set_ylabel("Query")

    # hide unused axes
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Attention patterns per head (causal mask visible)", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_causal_vs_bidirectional(seq_len=32, n_heads=4, head_dim=64):
    """Show the difference between causal and bidirectional attention."""
    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)

    _, attn_causal = vanilla_attention_with_scores(q, k, v, causal=True)
    _, attn_bidir = vanilla_attention_with_scores(q, k, v, causal=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(attn_causal[0, 0].detach().numpy(), cmap="viridis", aspect="auto")
    axes[0].set_title("Causal (decoder-style)")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")

    axes[1].imshow(attn_bidir[0, 0].detach().numpy(), cmap="viridis", aspect="auto")
    axes[1].set_title("Bidirectional (encoder-style)")
    axes[1].set_xlabel("Key position")
    axes[1].set_ylabel("Query position")

    plt.suptitle("Causal mask creates the triangular pattern", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/causal_vs_bidir.png", dpi=150, bbox_inches="tight")
    plt.show()


def demo_attention_patterns(seq_len=64, n_heads=8, head_dim=64):
    """Generate and visualize random attention patterns."""
    torch.manual_seed(0)
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)

    _, attn = vanilla_attention_with_scores(q, k, v, causal=True)
    plot_all_heads(attn, save_path="plots/attention_heads.png")


if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    plot_causal_vs_bidirectional()
    demo_attention_patterns()
