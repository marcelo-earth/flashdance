"""Advanced attention visualizations.

Extends visualize.py with:
- GQA pattern visualization (shared KV across head groups)
- Attention entropy heatmaps
- RoPE vs ALiBi position bias comparison
- Sliding window mask visualization
- Speedup heatmap (seq_len x head_dim)
"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from attention import vanilla_attention_with_scores
from gqa import repeat_kv
from rope import precompute_freqs, apply_rope
from alibi import build_alibi_bias, get_alibi_slopes
from sliding_window import sliding_window_mask
from entropy import attention_entropy


def plot_gqa_pattern(n_heads=8, n_kv_heads=2, seq_len=32, head_dim=32, save_path="plots/gqa_pattern.png"):
    """Visualize how GQA shares KV heads across query head groups."""
    os.makedirs("plots", exist_ok=True)

    torch.manual_seed(42)
    B = 1
    n_rep = n_heads // n_kv_heads

    q = torch.randn(B, n_heads, seq_len, head_dim)
    k = torch.randn(B, n_kv_heads, seq_len, head_dim)
    v = torch.randn(B, n_kv_heads, seq_len, head_dim)

    # expand KV
    k_exp, v_exp = repeat_kv(k, v, n_rep)

    # get attention weights for each query head
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = []
    for h in range(n_heads):
        q_h = q[:, h:h+1, :, :]
        k_h = k_exp[:, h:h+1, :, :]
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        attn_weights.append(w[0, 0].detach().numpy())

    # group headers by shared KV
    colors = ["steelblue", "coral", "seagreen", "purple"]
    kv_colors = {kv_idx: colors[kv_idx % len(colors)] for kv_idx in range(n_kv_heads)}

    fig, axes = plt.subplots(n_kv_heads, n_rep, figsize=(4 * n_rep, 4 * n_kv_heads))
    if n_kv_heads == 1:
        axes = [axes]
    if n_rep == 1:
        axes = [[ax] for ax in axes]

    for kv_idx in range(n_kv_heads):
        for rep_idx in range(n_rep):
            head_idx = kv_idx * n_rep + rep_idx
            ax = axes[kv_idx][rep_idx]
            im = ax.imshow(attn_weights[head_idx], cmap="Blues", aspect="auto", vmin=0, vmax=0.3)
            ax.set_title(f"Q-head {head_idx}\n(shares KV-head {kv_idx})",
                        color=kv_colors[kv_idx], fontweight="bold")
            ax.set_xlabel("Key pos")
            ax.set_ylabel("Query pos")

        # add colored border to group
        for rep_idx in range(n_rep):
            for spine in axes[kv_idx][rep_idx].spines.values():
                spine.set_edgecolor(kv_colors[kv_idx])
                spine.set_linewidth(3)

    plt.suptitle(f"GQA: {n_heads} query heads sharing {n_kv_heads} KV heads\n"
                 f"(same color = same KV head)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_position_encoding_comparison(seq_len=64, n_heads=4, head_dim=32, save_path="plots/pos_enc_comparison.png"):
    """Compare RoPE and ALiBi positional biases visually."""
    os.makedirs("plots", exist_ok=True)

    torch.manual_seed(0)
    B = 1
    q = torch.randn(B, n_heads, seq_len, head_dim)
    k = torch.randn(B, n_heads, seq_len, head_dim)

    scale = 1.0 / math.sqrt(head_dim)

    # No positional encoding
    raw_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # RoPE
    cos, sin = precompute_freqs(head_dim, seq_len)
    q_r, k_r = apply_rope(q, k, cos, sin)
    rope_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale

    # ALiBi
    alibi_bias = build_alibi_bias(n_heads, seq_len)
    alibi_scores = raw_scores + alibi_bias[:, :, :seq_len, :seq_len]

    # apply causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    raw_scores = raw_scores.masked_fill(mask, float("-inf"))
    rope_scores = rope_scores.masked_fill(mask, float("-inf"))

    raw_w = F.softmax(raw_scores, dim=-1)[0, 0].detach().numpy()
    rope_w = F.softmax(rope_scores, dim=-1)[0, 0].detach().numpy()
    alibi_w = F.softmax(alibi_scores, dim=-1)[0, 0].detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = 0.3

    for ax, w, title in [
        (axes[0], raw_w, "No position encoding\n(same q,k for all positions)"),
        (axes[1], rope_w, "RoPE\n(rotation in embedding space)"),
        (axes[2], alibi_w, "ALiBi\n(linear distance penalty)"),
    ]:
        im = ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax)

    plt.suptitle("Position Encoding Comparison: Attention Patterns (head 0)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_sliding_window_visual(seq_len=32, window_sizes=None, save_path="plots/sliding_window_masks.png"):
    """Visualize sliding window masks at different window sizes."""
    os.makedirs("plots", exist_ok=True)

    if window_sizes is None:
        window_sizes = [4, 8, 16, seq_len]

    fig, axes = plt.subplots(1, len(window_sizes), figsize=(5 * len(window_sizes), 5))
    if len(window_sizes) == 1:
        axes = [axes]

    for ax, W in zip(axes, window_sizes):
        mask = sliding_window_mask(seq_len, W).numpy()
        # show: white = can attend, gray = masked
        vis = (~mask).astype(float)
        ax.imshow(vis, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        label = "Full attention" if W == seq_len else f"Window={W}"
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    plt.suptitle("Sliding Window Attention Masks\n(blue=can attend, white=masked)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_entropy_heatmap(seq_len=128, n_heads=8, head_dim=64, save_path="plots/entropy_heatmap.png"):
    """Heatmap of attention entropy across heads and query positions."""
    os.makedirs("plots", exist_ok=True)

    torch.manual_seed(0)
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)

    _, weights = vanilla_attention_with_scores(q, k, v, causal=True)
    ent = attention_entropy(weights)[0].detach().numpy()  # (n_heads, seq_len)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(ent, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Query position")
    ax.set_ylabel("Attention head")
    ax.set_title("Attention Entropy per Head and Position\n(higher = more uniform, lower = more concentrated)")
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_gqa_pattern()
    plot_position_encoding_comparison()
    plot_sliding_window_visual()
    plot_entropy_heatmap()
