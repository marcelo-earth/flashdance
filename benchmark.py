"""Benchmark vanilla vs SDPA attention across sequence lengths."""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention


def measure_time(fn, *args, warmup=3, repeats=10, **kwargs):
    """Measure average execution time of a function."""
    # warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def measure_memory(fn, *args, **kwargs):
    """Measure peak memory usage of a function (CUDA only)."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0, "allocated_mb": 0}

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    before = torch.cuda.memory_allocated()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    return {
        "peak_mb": (peak - before) / 1024**2,
        "allocated_mb": peak / 1024**2,
    }


def benchmark_attention(
    seq_lengths=None,
    batch_size=4,
    n_heads=8,
    head_dim=64,
    device=None,
    repeats=10,
):
    """Run full benchmark suite."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Batch: {batch_size}, Heads: {n_heads}, Head dim: {head_dim}")
    print(f"Sequence lengths: {seq_lengths}")
    print()

    results = []

    for seq_len in seq_lengths:
        print(f"--- seq_len={seq_len} ---")

        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        # vanilla attention
        try:
            vanilla_time = measure_time(vanilla_attention, q, k, v, repeats=repeats)
            vanilla_mem = measure_memory(vanilla_attention, q, k, v)
            print(f"  Vanilla:  {vanilla_time['mean']*1000:8.2f}ms  (peak: {vanilla_mem['peak_mb']:.1f}MB)")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Vanilla:  OOM at seq_len={seq_len}")
                vanilla_time = {"mean": float("inf"), "min": float("inf"), "max": float("inf"), "std": 0}
                vanilla_mem = {"peak_mb": float("inf"), "allocated_mb": float("inf")}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

        # SDPA (flash/mem-efficient)
        sdpa_time = measure_time(sdpa_attention, q, k, v, repeats=repeats)
        sdpa_mem = measure_memory(sdpa_attention, q, k, v)
        print(f"  SDPA:     {sdpa_time['mean']*1000:8.2f}ms  (peak: {sdpa_mem['peak_mb']:.1f}MB)")

        speedup = vanilla_time["mean"] / sdpa_time["mean"] if sdpa_time["mean"] > 0 else 0
        print(f"  Speedup:  {speedup:.2f}x")

        results.append({
            "seq_len": seq_len,
            "vanilla_ms": vanilla_time["mean"] * 1000,
            "sdpa_ms": sdpa_time["mean"] * 1000,
            "speedup": speedup,
            "vanilla_peak_mb": vanilla_mem["peak_mb"],
            "sdpa_peak_mb": sdpa_mem["peak_mb"],
        })

        del q, k, v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def benchmark_backward(
    seq_lengths=None,
    batch_size=4,
    n_heads=8,
    head_dim=64,
    device=None,
    repeats=5,
):
    """Benchmark backward pass (gradient computation)."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nBackward pass benchmark")
    print(f"Device: {device}")
    print()

    results = []

    for seq_len in seq_lengths:
        print(f"--- seq_len={seq_len} ---")

        def run_vanilla():
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            out = vanilla_attention(q, k, v)
            loss = out.sum()
            loss.backward()

        def run_sdpa():
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, requires_grad=True)
            out = sdpa_attention(q, k, v)
            loss = out.sum()
            loss.backward()

        try:
            vanilla_time = measure_time(run_vanilla, warmup=2, repeats=repeats)
            print(f"  Vanilla backward:  {vanilla_time['mean']*1000:8.2f}ms")
        except RuntimeError:
            vanilla_time = {"mean": float("inf")}
            print(f"  Vanilla backward:  OOM")

        sdpa_time = measure_time(run_sdpa, warmup=2, repeats=repeats)
        print(f"  SDPA backward:     {sdpa_time['mean']*1000:8.2f}ms")

        speedup = vanilla_time["mean"] / sdpa_time["mean"] if sdpa_time["mean"] > 0 else 0
        print(f"  Speedup:           {speedup:.2f}x")

        results.append({
            "seq_len": seq_len,
            "vanilla_backward_ms": vanilla_time["mean"] * 1000,
            "sdpa_backward_ms": sdpa_time["mean"] * 1000,
            "backward_speedup": speedup,
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def print_results(results):
    """Print results as a formatted table."""
    try:
        from tabulate import tabulate
        headers = ["Seq Len", "Vanilla (ms)", "SDPA (ms)", "Speedup", "Vanilla Mem (MB)", "SDPA Mem (MB)"]
        rows = []
        for r in results:
            rows.append([
                r["seq_len"],
                f"{r['vanilla_ms']:.2f}" if r["vanilla_ms"] != float("inf") else "OOM",
                f"{r['sdpa_ms']:.2f}",
                f"{r['speedup']:.2f}x",
                f"{r['vanilla_peak_mb']:.1f}" if r["vanilla_peak_mb"] != float("inf") else "OOM",
                f"{r['sdpa_peak_mb']:.1f}",
            ])
        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        for r in results:
            print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward", action="store_true", help="Also benchmark backward pass")
    args = parser.parse_args()

    results = benchmark_attention(
        seq_lengths=args.seq_len,
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        device=args.device,
        repeats=args.repeats,
    )
    print_results(results)

    if args.backward:
        backward_results = benchmark_backward(
            seq_lengths=args.seq_len,
            batch_size=args.batch_size,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            device=args.device,
            repeats=min(args.repeats, 5),
        )
