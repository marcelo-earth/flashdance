"""Extended dtype benchmark: FP32, FP16, BF16 across all attention variants.

BF16 vs FP16:
- BF16: same exponent range as FP32 (8 bits), less precision (7 mantissa bits)
  -> Less risk of overflow, great for training
- FP16: more precision (10 mantissa bits), smaller exponent range (5 bits)
  -> Better for inference on older GPUs (pre-Ampere)
- FP32: full precision, 2x memory vs FP16/BF16

Modern hardware (A100, H100, Apple M-series):
- Tensor cores work natively with BF16
- Flash Attention requires FP16 or BF16
"""

import time
import gc
import torch
import torch.nn.functional as F

from attention import vanilla_attention, sdpa_attention


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_dtypes_full(
    seq_lengths=None,
    batch_size: int = 4,
    n_heads: int = 8,
    head_dim: int = 64,
    device=None,
    repeats: int = 20,
):
    """Benchmark all dtypes x all attention variants."""
    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048]

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # available dtypes per device
    if device == "cuda":
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
    elif device == "mps":
        dtypes = [torch.float32, torch.float16]
    else:
        dtypes = [torch.float32]

    dtype_names = {
        torch.float32: "FP32",
        torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }

    print(f"\nDtype Benchmark: FP32 vs FP16 vs BF16")
    print(f"Device: {device} | batch={batch_size} | heads={n_heads} | head_dim={head_dim}")
    print()

    all_results = []

    for seq_len in seq_lengths:
        print(f"seq_len={seq_len}")
        print(f"  {'dtype':>6} | {'Vanilla (ms)':>13} | {'SDPA (ms)':>11} | {'Speedup':>8} | {'Memory ratio':>13}")
        print(f"  {'-'*60}")

        fp32_sdpa_ms = None
        for dtype in dtypes:
            q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device).to(dtype)
            k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device).to(dtype)
            v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device).to(dtype)

            # vanilla
            try:
                for _ in range(3):
                    vanilla_attention(q, k, v)
                _sync(device)
                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    vanilla_attention(q, k, v)
                    _sync(device)
                    times.append(time.perf_counter() - t0)
                times.sort()
                van_ms = times[len(times) // 2] * 1000
            except Exception:
                van_ms = float("inf")

            # sdpa
            for _ in range(3):
                sdpa_attention(q, k, v)
            _sync(device)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                sdpa_attention(q, k, v)
                _sync(device)
                times.append(time.perf_counter() - t0)
            times.sort()
            sdpa_ms = times[len(times) // 2] * 1000

            if fp32_sdpa_ms is None:
                fp32_sdpa_ms = sdpa_ms

            speedup_vs_vanilla = van_ms / sdpa_ms if van_ms != float("inf") else 0
            mem_ratio = 4 / (dtype_names[dtype] == "FP32" and 4 or 2)  # FP32=4bytes, others=2

            dname = dtype_names[dtype]
            van_str = f"{van_ms:13.2f}" if van_ms != float("inf") else "         OOM"
            print(f"  {dname:>6} | {van_str} | {sdpa_ms:>11.2f} | {speedup_vs_vanilla:>7.2f}x | {fp32_sdpa_ms/sdpa_ms:>12.2f}x")

            all_results.append({
                "seq_len": seq_len,
                "dtype": dname,
                "vanilla_ms": van_ms,
                "sdpa_ms": sdpa_ms,
                "speedup_vs_vanilla": speedup_vs_vanilla,
                "speedup_vs_fp32": fp32_sdpa_ms / sdpa_ms,
            })

            del q, k, v
            gc.collect()

        fp32_sdpa_ms = None
        print()

    return all_results


def numerical_precision_comparison(
    seq_len: int = 128,
    n_heads: int = 4,
    head_dim: int = 32,
    device=None,
    n_trials: int = 10,
):
    """Compare numerical precision across dtypes."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    dtypes = [torch.float32]
    if device == "cuda":
        dtypes += [torch.float16, torch.bfloat16]
    elif device == "mps":
        dtypes += [torch.float16]

    dtype_names = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}

    print(f"\nNumerical Precision: max absolute error vs FP64 reference")
    print(f"seq_len={seq_len} | n_heads={n_heads} | head_dim={head_dim}")
    print(f"{'dtype':>6} | {'vanilla vs ref':>15} | {'sdpa vs ref':>13} | {'vanilla vs sdpa':>16}")
    print("-" * 60)

    errors = {}
    for _ in range(n_trials):
        q64 = torch.randn(1, n_heads, seq_len, head_dim)
        k64 = torch.randn(1, n_heads, seq_len, head_dim)
        v64 = torch.randn(1, n_heads, seq_len, head_dim)

        ref = vanilla_attention(q64.double(), k64.double(), v64.double()).float()

        for dtype in dtypes:
            q = q64.to(dtype).to(device)
            k = k64.to(dtype).to(device)
            v = v64.to(dtype).to(device)

            out_van = vanilla_attention(q, k, v).float().cpu()
            out_sdpa = sdpa_attention(q, k, v).float().cpu()

            key = dtype_names[dtype]
            if key not in errors:
                errors[key] = {"van_vs_ref": [], "sdpa_vs_ref": [], "van_vs_sdpa": []}
            errors[key]["van_vs_ref"].append((out_van - ref).abs().max().item())
            errors[key]["sdpa_vs_ref"].append((out_sdpa - ref).abs().max().item())
            errors[key]["van_vs_sdpa"].append((out_van - out_sdpa).abs().max().item())

    import statistics
    for dname, errs in errors.items():
        van_err = statistics.mean(errs["van_vs_ref"])
        sdpa_err = statistics.mean(errs["sdpa_vs_ref"])
        diff_err = statistics.mean(errs["van_vs_sdpa"])
        print(f"{dname:>6} | {van_err:>15.2e} | {sdpa_err:>13.2e} | {diff_err:>16.2e}")

    return errors


if __name__ == "__main__":
    benchmark_dtypes_full()
    numerical_precision_comparison()
