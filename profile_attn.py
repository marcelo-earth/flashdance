"""Profile attention with torch.profiler to see where time goes."""

import torch
from torch.profiler import profile, ProfilerActivity

from attention import vanilla_attention, sdpa_attention


def profile_attention(seq_len=1024, batch_size=4, n_heads=8, head_dim=64, device=None):
    """Run torch profiler on both attention implementations."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling vanilla attention (seq_len={seq_len})...")
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(5):
            vanilla_attention(q, k, v)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print(f"\nProfiling SDPA attention (seq_len={seq_len})...")
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(5):
            sdpa_attention(q, k, v)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


def profile_backward(seq_len=512, batch_size=4, n_heads=8, head_dim=64, device=None):
    """Profile backward pass for both implementations."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    def run_vanilla():
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        out = vanilla_attention(q, k, v)
        out.sum().backward()

    def run_sdpa():
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        out = sdpa_attention(q, k, v)
        out.sum().backward()

    print(f"\nProfiling vanilla backward (seq_len={seq_len})...")
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(3):
            run_vanilla()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print(f"\nProfiling SDPA backward (seq_len={seq_len})...")
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(3):
            run_sdpa()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    profile_attention(seq_len=args.seq_len, device=args.device)
    if args.backward:
        profile_backward(seq_len=args.seq_len, device=args.device)
