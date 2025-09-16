"""SDPA backend selection and benchmarking.

PyTorch's scaled_dot_product_attention (SDPA) has three backends:
1. Flash Attention -- fastest, requires head_dim <= 128, FP16/BF16
2. Memory-efficient -- xformers-style, more flexible
3. Math -- fallback for CPU and unsupported configs

This module lets you force specific backends and compare their performance.
Useful for understanding which backend is actually being used.
"""

import torch
import torch.nn.functional as F
import time


def detect_sdpa_backends(device: str = "cuda"):
    """Detect which SDPA backends are available."""
    if not torch.cuda.is_available():
        print(f"CUDA not available. On {device}, SDPA uses the math backend.")
        return {"math": True, "flash": False, "mem_efficient": False, "available": False}

    backends = {}

    # Flash Attention
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            q = torch.randn(1, 4, 16, 64, device=device, dtype=torch.float16)
            F.scaled_dot_product_attention(q, q, q)
            backends["flash"] = True
    except Exception as e:
        backends["flash"] = False
        backends["flash_error"] = str(e)

    # Memory-efficient
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            q = torch.randn(1, 4, 16, 64, device=device, dtype=torch.float16)
            F.scaled_dot_product_attention(q, q, q)
            backends["mem_efficient"] = True
    except Exception as e:
        backends["mem_efficient"] = False

    backends["math"] = True  # always available
    backends["available"] = True
    return backends


def benchmark_sdpa_backends(
    seq_len: int = 1024,
    batch_size: int = 4,
    n_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda",
    dtype=torch.float16,
    repeats: int = 50,
):
    """Benchmark each SDPA backend separately (CUDA only)."""
    if not torch.cuda.is_available():
        print("CUDA not available -- can't isolate SDPA backends. Showing SDPA default only.")
        return _benchmark_single(seq_len, batch_size, n_heads, head_dim,
                                 device="mps" if torch.backends.mps.is_available() else "cpu",
                                 dtype=torch.float32, repeats=repeats)

    backends_to_test = [
        ("Flash Attention", {"enable_flash": True, "enable_math": False, "enable_mem_efficient": False}),
        ("Mem-efficient", {"enable_flash": False, "enable_math": False, "enable_mem_efficient": True}),
        ("Math (fallback)", {"enable_flash": False, "enable_math": True, "enable_mem_efficient": False}),
    ]

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    print(f"\nSDPA Backend Benchmark")
    print(f"seq_len={seq_len} | batch={batch_size} | heads={n_heads} | head_dim={head_dim} | dtype={dtype}")
    print()

    results = {}
    for name, backend_kwargs in backends_to_test:
        try:
            with torch.backends.cuda.sdp_kernel(**backend_kwargs):
                # warmup
                for _ in range(5):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                torch.cuda.synchronize()

                times = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)

                times.sort()
                ms = times[len(times) // 2] * 1000
                results[name] = ms
                print(f"  {name:<25}: {ms:.2f}ms (median)")
        except Exception as e:
            print(f"  {name:<25}: UNAVAILABLE ({e})")
            results[name] = None

    return results


def _benchmark_single(seq_len, batch_size, n_heads, head_dim, device, dtype, repeats):
    """Fallback for non-CUDA: just benchmark default SDPA."""
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)

    def _sync():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(5):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _sync()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        _sync()
        times.append(time.perf_counter() - t0)

    times.sort()
    ms = times[len(times) // 2] * 1000
    print(f"\nSDPA (default backend) on {device}: {ms:.2f}ms  (seq={seq_len})")
    return {"default": ms}


def get_active_backend_name() -> str:
    """Try to detect which SDPA backend is currently active."""
    if not torch.cuda.is_available():
        return "math (CPU/MPS fallback)"
    # check flash attn availability
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            q = torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.float16)
            F.scaled_dot_product_attention(q, q, q)
        return "flash_attention"
    except Exception:
        pass
    return "mem_efficient or math"


def sdpa_constraints_table():
    """Print constraints for each SDPA backend."""
    print("\nSDPA Backend Constraints:")
    print("-" * 70)
    constraints = {
        "Flash Attention": {
            "dtype": "FP16, BF16",
            "head_dim": "<= 256 (optimal: 64, 128)",
            "hardware": "A100, H100, RTX 30xx/40xx",
            "causal": "Yes",
            "notes": "Fastest for large seq_len",
        },
        "Memory-efficient": {
            "dtype": "FP16, BF16, FP32",
            "head_dim": "Any",
            "hardware": "Any CUDA GPU",
            "causal": "Yes",
            "notes": "Good fallback, uses xformers-style tiling",
        },
        "Math": {
            "dtype": "Any",
            "head_dim": "Any",
            "hardware": "CPU, MPS, any CUDA",
            "causal": "Yes",
            "notes": "Reference implementation, slowest",
        },
    }

    for backend, info in constraints.items():
        print(f"\n  [{backend}]")
        for k, v in info.items():
            print(f"    {k:<12}: {v}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    backends = detect_sdpa_backends(device)
    print(f"SDPA backends on {device}:")
    for name, available in backends.items():
        if not name.endswith("_error"):
            status = "✓ available" if available else "✗ unavailable"
            print(f"  {name:<20}: {status}")

    sdpa_constraints_table()
    benchmark_sdpa_backends(device=device, dtype=torch.float32 if device != "cuda" else torch.float16)
