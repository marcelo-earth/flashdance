"""Sliding Window Attention (SWA) -- used by Mistral 7B.

Instead of attending to all past tokens (O(n^2) memory),
each token only attends to the W nearest tokens.

- Memory: O(n * W) instead of O(n^2)
- At W=4096, seq_len=32K: 8x memory reduction over full attention
- Mistral uses W=4096 with 32K context

For long sequences with local structure (code, text), SWA hurts quality
minimally while giving massive memory savings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sliding_window_mask(seq_len: int, window_size: int, device=None) -> torch.Tensor:
    """Create a causal sliding window attention mask.

    Positions outside the window are masked to -inf.

    Returns:
        mask: (seq_len, seq_len) bool tensor, True = masked (ignored)
    """
    # start with full causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    # also mask tokens that are too far in the past
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        if start > 0:
            mask[i, :start] = True
    return mask


def sliding_window_attention(q, k, v, window_size: int, causal=True):
    """Attention with sliding window mask.

    Args:
        q, k, v: (batch, heads, seq_len, head_dim)
        window_size: number of tokens each position can attend to
        causal: also apply causal mask

    Returns:
        output: (batch, heads, seq_len, head_dim)
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

    # build mask: positions outside the sliding window
    mask = sliding_window_mask(T, window_size, device=q.device)  # (T, T)

    if not causal:
        # symmetric sliding window (for encoders)
        mask_sym = mask & mask.T
        # actually for bidirectional, we want: |i - j| >= window_size
        mask = torch.zeros(T, T, device=q.device, dtype=torch.bool)
        for i in range(T):
            for j in range(T):
                if abs(i - j) >= window_size:
                    mask[i, j] = True

    attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(attn, dim=-1)
    # replace nan (all-masked rows) with 0
    attn = torch.nan_to_num(attn, nan=0.0)

    return torch.matmul(attn, v)


class SlidingWindowAttention(nn.Module):
    """Mistral-style sliding window attention module."""

    def __init__(self, dim: int, n_heads: int, window_size: int = 4096, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal=True):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = sliding_window_attention(q, k, v, window_size=self.window_size, causal=causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


def benchmark_sliding_window(
    seq_lengths=None,
    window_sizes=None,
    batch_size=2,
    n_heads=8,
    head_dim=64,
    device=None,
    repeats=5,
):
    """Compare full attention vs sliding window at various sequence lengths."""
    import time
    import gc

    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]
    if window_sizes is None:
        window_sizes = [128, 256, 512]

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nSliding Window Attention Benchmark")
    print(f"Device: {device} | batch={batch_size} | heads={n_heads} | head_dim={head_dim}")

    results = []
    for seq_len in seq_lengths:
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        # full attention baseline
        def run_full():
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # warmup
        for _ in range(2):
            run_full()

        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_full()
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
        full_ms = sum(times) / len(times) * 1000

        row = {"seq_len": seq_len, "full_ms": full_ms}

        for W in window_sizes:
            if W >= seq_len:
                continue

            def run_swa():
                return sliding_window_attention(q, k, v, window_size=W)

            for _ in range(2):
                run_swa()

            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                run_swa()
                if device == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - t0)
            swa_ms = sum(times) / len(times) * 1000
            speedup = full_ms / swa_ms
            row[f"swa_w{W}_ms"] = swa_ms
            row[f"swa_w{W}_speedup"] = speedup

        results.append(row)
        del q, k, v
        gc.collect()

    # print table
    print(f"\n{'Seq':>6} | {'Full(ms)':>10} | " + " | ".join(f"W={W} (ms/speedup)" for W in window_sizes if W < seq_lengths[-1]))
    print("-" * 80)
    for r in results:
        row_str = f"{r['seq_len']:>6} | {r['full_ms']:>10.2f} | "
        for W in window_sizes:
            key = f"swa_w{W}_ms"
            if key in r:
                row_str += f"{r[key]:>8.2f}ms ({r[f'swa_w{W}_speedup']:.2f}x) | "
        print(row_str)

    return results


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # sanity: check output shape
    B, H, T, D = 2, 4, 64, 32
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)

    out = sliding_window_attention(q, k, v, window_size=8)
    assert out.shape == (B, H, T, D)
    print(f"SWA output shape: {out.shape}  ✓")

    # verify masking: token 0 and token 20 should not interact with W=8
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    mask = sliding_window_mask(T, window_size=8, device=device)
    # position 20 should not see position 0 (distance = 20 >= window = 8)
    assert mask[20, 0].item() == True, "Position 0 should be masked for position 20"
    # position 20 should see position 15 (distance = 5 < window = 8)
    assert mask[20, 15].item() == False, "Position 15 should be visible for position 20"
    print(f"Mask sanity check passed  ✓")

    benchmark_sliding_window(seq_lengths=[256, 512, 1024])
