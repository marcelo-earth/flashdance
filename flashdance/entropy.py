"""Attention entropy analysis.

Entropy measures how "spread out" attention is:
- Low entropy (near 0): attention is concentrated on few tokens (sharp/spiky)
- High entropy (near log(T)): attention is uniform over all tokens

Research findings:
- Early layers tend to have higher entropy (global patterns)
- Later layers tend to have lower entropy (task-specific focus)
- "Attention sinks" -- token 0 often gets disproportionate weight
- Models trained with ALiBi show different entropy profiles than RoPE models
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from attention import vanilla_attention_with_scores
from alibi import build_alibi_bias, alibi_attention
from rope import precompute_freqs, apply_rope


def attention_entropy(weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute per-head entropy of attention weight distributions.

    Args:
        weights: (batch, heads, seq_len, seq_len) -- attention weights (softmax'd)

    Returns:
        entropy: (batch, heads, seq_len) -- per-query-position entropy
    """
    # H(p) = -sum(p * log(p))
    log_weights = torch.log(weights + eps)
    entropy = -(weights * log_weights).sum(dim=-1)  # (B, H, T)
    return entropy


def max_entropy(seq_len: int) -> float:
    """Maximum possible entropy for a distribution over seq_len tokens (uniform)."""
    return math.log(seq_len)


def analyze_attention_entropy(
    seq_len: int = 128,
    n_heads: int = 8,
    head_dim: int = 64,
    batch_size: int = 1,
    device=None,
    seed: int = 42,
):
    """Compare entropy profiles of vanilla MHA vs ALiBi vs RoPE attention."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(seed)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    max_ent = max_entropy(seq_len)
    results = {}

    # --- Vanilla MHA ---
    _, vanilla_weights = vanilla_attention_with_scores(q, k, v, causal=True)
    vanilla_ent = attention_entropy(vanilla_weights)  # (B, H, T)
    results["Vanilla MHA"] = vanilla_ent.mean(dim=(0, 2)).cpu()  # (H,)

    # --- ALiBi ---
    scale = 1.0 / math.sqrt(head_dim)
    alibi_bias = build_alibi_bias(n_heads, seq_len, device=device)
    attn_alibi = torch.matmul(q, k.transpose(-2, -1)) * scale + alibi_bias[:, :, :seq_len, :seq_len]
    # apply causal (already in alibi_bias)
    alibi_w = F.softmax(attn_alibi, dim=-1)
    alibi_ent = attention_entropy(alibi_w)
    results["ALiBi"] = alibi_ent.mean(dim=(0, 2)).cpu()

    # --- RoPE ---
    cos, sin = precompute_freqs(head_dim, seq_len, device=device)
    q_r, k_r = apply_rope(q, k, cos, sin)
    _, rope_weights = vanilla_attention_with_scores(q_r, k_r, v, causal=True)
    rope_ent = attention_entropy(rope_weights)
    results["RoPE"] = rope_ent.mean(dim=(0, 2)).cpu()

    print(f"\nAttention Entropy Analysis (seq_len={seq_len}, max_entropy={max_ent:.2f})")
    print(f"{'Method':<15} {'Mean':>8} {'Min':>8} {'Max':>8} {'%Max':>8}")
    print("-" * 55)
    for name, ent in results.items():
        mean_e = ent.mean().item()
        min_e = ent.min().item()
        max_e = ent.max().item()
        pct = mean_e / max_ent * 100
        print(f"{name:<15} {mean_e:>8.3f} {min_e:>8.3f} {max_e:>8.3f} {pct:>7.1f}%")

    return results


def detect_attention_sinks(
    weights: torch.Tensor,
    threshold_multiplier: float = 3.0,
) -> dict:
    """Detect attention sink tokens (tokens that receive disproportionate weight).

    An attention sink is a token position that receives much more attention
    than the average across positions.

    Args:
        weights: (batch, heads, seq_len, seq_len) -- softmax attention weights

    Returns:
        dict with sink token positions and statistics
    """
    # average attention received per position across all queries
    received = weights.mean(dim=2)  # (B, H, T) -- avg weight token j receives
    avg_weight = 1.0 / weights.shape[-1]  # uniform baseline

    # a sink is where received > threshold * uniform
    threshold = avg_weight * threshold_multiplier
    is_sink = (received > threshold)  # (B, H, T)

    # which positions are most commonly sinks?
    sink_freq = is_sink.float().mean(dim=(0, 1))  # (T,) -- fraction of (batch, head) pairs

    top_sinks = sink_freq.topk(min(5, len(sink_freq)))

    return {
        "sink_positions": top_sinks.indices.tolist(),
        "sink_frequencies": top_sinks.values.tolist(),
        "pct_heads_with_sink_at_0": is_sink[:, :, 0].float().mean().item(),
        "mean_received_at_0": received[:, :, 0].mean().item(),
        "mean_received_avg": avg_weight,
    }


def plot_entropy_analysis(seq_len=256, n_heads=8, head_dim=64, save_path="plots/entropy.png"):
    """Plot entropy profiles and attention sink analysis."""
    try:
        import matplotlib.pyplot as plt
        import os
    except ImportError:
        print("matplotlib not available")
        return

    os.makedirs("plots", exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results = analyze_attention_entropy(seq_len=seq_len, n_heads=n_heads, head_dim=head_dim, device=device)
    max_ent = max_entropy(seq_len)

    # also get attention weights for sink analysis
    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(1, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(1, n_heads, seq_len, head_dim, device=device)
    _, weights = vanilla_attention_with_scores(q, k, v, causal=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # entropy per head
    colors = ["steelblue", "coral", "seagreen"]
    for (name, ent), color in zip(results.items(), colors):
        axes[0].bar(range(n_heads), ent.numpy(), alpha=0.7, label=name, color=color)
    axes[0].axhline(y=max_ent, color="black", linestyle="--", alpha=0.5, label=f"Max entropy ({max_ent:.2f})")
    axes[0].set_xlabel("Head index")
    axes[0].set_ylabel("Mean entropy (nats)")
    axes[0].set_title("Attention Entropy per Head")
    axes[0].legend(fontsize=8)

    # attention received by each position (sink detection)
    received = weights[0].mean(dim=1).detach().cpu().numpy()  # (H, T) -> mean over queries
    avg_received = received.mean(axis=0)  # (T,)
    uniform = 1.0 / seq_len

    axes[1].plot(avg_received, alpha=0.8, color="steelblue")
    axes[1].axhline(y=uniform, color="red", linestyle="--", alpha=0.7, label=f"Uniform ({uniform:.4f})")
    axes[1].axhline(y=uniform * 3, color="orange", linestyle="--", alpha=0.7, label="Sink threshold (3x)")
    axes[1].set_xlabel("Token position")
    axes[1].set_ylabel("Avg attention received")
    axes[1].set_title("Attention Sink Detection")
    axes[1].legend(fontsize=9)

    # entropy vs position (does entropy change over sequence length?)
    torch.manual_seed(42)
    _, w = vanilla_attention_with_scores(q, k, v, causal=True)
    ent_by_pos = attention_entropy(w)[0].mean(dim=0).detach().cpu().numpy()  # (T,)
    axes[2].plot(ent_by_pos, alpha=0.8, color="steelblue")
    axes[2].axhline(y=max_ent, color="black", linestyle="--", alpha=0.5, label=f"Max entropy")
    axes[2].set_xlabel("Query position")
    axes[2].set_ylabel("Entropy (nats)")
    axes[2].set_title("Entropy by Query Position\n(early tokens have less to attend to)")
    axes[2].legend()

    plt.suptitle("Attention Entropy Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    results = analyze_attention_entropy(seq_len=256)

    # sink analysis
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    q = torch.randn(4, 8, 256, 64, device=device)
    k = torch.randn(4, 8, 256, 64, device=device)
    v = torch.randn(4, 8, 256, 64, device=device)
    _, w = vanilla_attention_with_scores(q, k, v, causal=True)
    sinks = detect_attention_sinks(w)
    print(f"\nAttention Sink Analysis:")
    print(f"  Top sink positions:    {sinks['sink_positions']}")
    print(f"  % heads with sink@0:   {sinks['pct_heads_with_sink_at_0']*100:.1f}%")
    print(f"  Mean weight at pos 0:  {sinks['mean_received_at_0']:.4f} (uniform={sinks['mean_received_avg']:.4f})")

    plot_entropy_analysis()
