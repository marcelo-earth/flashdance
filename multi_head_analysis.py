"""Multi-head attention analysis: what do different heads learn?

Research shows different attention heads specialize:
- Some heads attend to syntactic relationships (subject-verb)
- Some attend to positional neighbors ("next word" heads)
- Some do global attention (attend to many tokens)
- Some are "attention sinks" (token 0 gets heavy weight)

This module provides tools to analyze head specialization.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from attention import vanilla_attention_with_scores
from entropy import attention_entropy


def classify_head(weights: torch.Tensor, pos_window: int = 3) -> str:
    """Classify an attention head's behavior based on its weight patterns.

    Args:
        weights: (seq_len, seq_len) -- attention weights for one head

    Returns:
        str: 'local', 'global', 'sink', or 'mixed'
    """
    T = weights.shape[-1]
    weights_np = weights.detach().cpu().numpy()

    # entropy (how spread out is attention?)
    eps = 1e-9
    ent = -(weights_np * np.log(weights_np + eps)).sum(axis=-1).mean()
    max_ent = math.log(T)
    relative_ent = ent / max_ent

    # how much attention goes to nearby tokens?
    local_weight = 0.0
    count = 0
    for i in range(T):
        local_start = max(0, i - pos_window)
        local_weight += weights_np[i, local_start:i + 1].sum()
        count += 1
    local_weight /= count

    # how much does token 0 receive?
    sink_weight = weights_np[:, 0].mean()
    uniform = 1.0 / T

    if sink_weight > 5 * uniform:
        return "sink"
    elif local_weight > 0.7:
        return "local"
    elif relative_ent > 0.7:
        return "global"
    else:
        return "mixed"


def analyze_head_specialization(
    seq_len: int = 128,
    n_heads: int = 8,
    head_dim: int = 64,
    n_samples: int = 10,
    device=None,
    seed: int = 0,
):
    """Analyze head specialization across multiple random inputs."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(seed)
    head_types = {h: [] for h in range(n_heads)}
    entropies = {h: [] for h in range(n_heads)}

    for _ in range(n_samples):
        q = torch.randn(1, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(1, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(1, n_heads, seq_len, head_dim, device=device)

        _, weights = vanilla_attention_with_scores(q, k, v, causal=True)

        for h in range(n_heads):
            w_h = weights[0, h]  # (T, T)
            head_type = classify_head(w_h)
            head_types[h].append(head_type)

            eps = 1e-9
            ent = -(w_h * torch.log(w_h + eps)).sum(dim=-1).mean().item()
            entropies[h].append(ent)

    # summarize
    print(f"\nHead Specialization Analysis")
    print(f"seq_len={seq_len} | n_heads={n_heads} | head_dim={head_dim} | n_samples={n_samples}")
    print()
    print(f"{'Head':>6} | {'Mean entropy':>14} | {'Most common type':>18} | {'Type distribution':>30}")
    print("-" * 80)

    results = []
    for h in range(n_heads):
        mean_ent = np.mean(entropies[h])
        types = head_types[h]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        most_common = max(type_counts, key=type_counts.get)
        dist_str = "  ".join(f"{t}:{c}" for t, c in sorted(type_counts.items()))
        print(f"{h:>6} | {mean_ent:>14.3f} | {most_common:>18} | {dist_str}")
        results.append({
            "head": h,
            "mean_entropy": mean_ent,
            "most_common_type": most_common,
            "type_distribution": type_counts,
        })

    return results


def head_diversity_score(weights: torch.Tensor) -> float:
    """Measure diversity between heads (how different are they?).

    If all heads attend to the same places, they're redundant.
    Higher diversity = more useful multi-head attention.

    Returns:
        float: diversity score (0 = identical heads, 1 = maximally diverse)
    """
    B, H, T, _ = weights.shape
    # flatten each head's weight distribution
    heads = weights[0].view(H, -1)  # (H, T*T)

    # compute pairwise cosine similarity between heads
    heads_norm = F.normalize(heads, dim=-1)
    sim = torch.matmul(heads_norm, heads_norm.T)  # (H, H)

    # diversity = 1 - mean off-diagonal similarity
    mask = ~torch.eye(H, dtype=torch.bool)
    diversity = 1.0 - sim[mask].mean().item()
    return diversity


def plot_head_analysis(n_heads=8, seq_len=64, head_dim=32, save_path="plots/head_analysis.png"):
    """Visualize head specialization patterns."""
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)
    except ImportError:
        return

    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)

    _, weights = vanilla_attention_with_scores(q, k, v, causal=True)
    diversity = head_diversity_score(weights)

    n_cols = 4
    n_rows = math.ceil(n_heads / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    type_colors = {"local": "Blues", "global": "Greens", "sink": "Reds", "mixed": "Purples"}

    for h in range(n_heads):
        w = weights[0, h].detach().numpy()
        head_type = classify_head(weights[0, h])
        cmap = type_colors.get(head_type, "viridis")

        axes[h].imshow(w, cmap=cmap, aspect="auto", vmin=0, vmax=w.max())
        axes[h].set_title(f"Head {h} [{head_type}]")
        axes[h].set_xlabel("Key")
        axes[h].set_ylabel("Query")

    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Head Specialization (seq={seq_len}, d_head={head_dim})\n"
                 f"Head diversity score: {diversity:.3f}  (local=blue, global=green, sink=red, mixed=purple)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = analyze_head_specialization()

    # diversity test
    torch.manual_seed(42)
    q = torch.randn(1, 8, 64, 32)
    k = torch.randn(1, 8, 64, 32)
    v = torch.randn(1, 8, 64, 32)
    _, weights = vanilla_attention_with_scores(q, k, v, causal=True)
    div = head_diversity_score(weights)
    print(f"\nHead diversity score: {div:.3f}  (1.0 = maximally diverse)")

    plot_head_analysis()
