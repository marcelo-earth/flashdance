"""Run all benchmarks and generate a full HTML/markdown report.

Usage:
    python run_all.py                    # quick mode
    python run_all.py --full             # full benchmark suite
    python run_all.py --plots            # save all plots
    python run_all.py --full --plots     # everything
"""

import argparse
import json
import os
import sys
import time
import datetime
import torch


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def get_device_info():
    info = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "device": None,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        mem = torch.cuda.get_device_properties(0).total_memory
        info["gpu_memory_gb"] = mem / 1024**3
    elif torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu"] = "Apple Silicon (MPS)"
    else:
        info["device"] = "cpu"
        info["gpu"] = "CPU only"
    return info


def run_suite(quick=True, save_plots=False):
    start_time = time.time()
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device_info": get_device_info(),
        "results": {},
    }

    device = report["device_info"]["device"]
    print(f"\nFlashDance Benchmark Suite")
    print(f"Device: {device} | PyTorch {report['device_info']['torch_version']}")
    print(f"GPU: {report['device_info']['gpu']}")

    seq_lens = [256, 512, 1024] if quick else [128, 256, 512, 1024, 2048, 4096]
    batch = 2 if quick else 4
    repeats = 5 if quick else 15

    # ── 1. Main attention benchmark ──────────────────────────────────────────
    section("1. Forward Pass: Vanilla vs SDPA")
    from benchmark import benchmark_attention
    fwd_results = benchmark_attention(
        seq_lengths=seq_lens, batch_size=batch, repeats=repeats
    )
    report["results"]["forward"] = fwd_results

    # ── 2. GQA benchmark ─────────────────────────────────────────────────────
    section("2. GQA / MHA / MQA Comparison")
    from benchmark_gqa import benchmark_gqa_configs
    gqa_results = benchmark_gqa_configs(
        seq_lengths=seq_lens, batch_size=batch, repeats=repeats
    )
    report["results"]["gqa"] = gqa_results

    # ── 3. Sliding window ────────────────────────────────────────────────────
    section("3. Sliding Window Attention")
    from sliding_window import benchmark_sliding_window
    swa_results = benchmark_sliding_window(
        seq_lengths=seq_lens, batch_size=batch, repeats=repeats
    )
    report["results"]["sliding_window"] = swa_results

    # ── 4. KV cache speedup ──────────────────────────────────────────────────
    section("4. KV Cache: Inference Speedup")
    from kv_cache import benchmark_kv_cache
    kv_result = benchmark_kv_cache(
        prefill_len=min(seq_lens), n_gen_tokens=64, dim=512, n_heads=8, n_kv_heads=2
    )
    report["results"]["kv_cache"] = kv_result

    # ── 5. IO analysis ───────────────────────────────────────────────────────
    section("5. Flash Attention IO Analysis")
    from attention_io import io_comparison_table
    io_results = io_comparison_table(seq_lengths=seq_lens)
    report["results"]["io_analysis"] = io_results

    # ── 6. Throughput ────────────────────────────────────────────────────────
    section("6. Throughput (tokens/sec)")
    from throughput import benchmark_throughput
    tput_results = benchmark_throughput(
        seq_lengths=seq_lens, batch_sizes=[1, 4], repeats=repeats
    )
    report["results"]["throughput"] = tput_results

    # ── 7. Memory analysis ───────────────────────────────────────────────────
    section("7. Activation Memory Analysis")
    from memory_analysis import activation_memory_table
    mem_results = activation_memory_table(seq_lengths=seq_lens)
    report["results"]["memory"] = mem_results

    # ── 8. Entropy ───────────────────────────────────────────────────────────
    section("8. Attention Entropy")
    from entropy import analyze_attention_entropy
    ent_results = analyze_attention_entropy(seq_len=seq_lens[0] if quick else 256)
    report["results"]["entropy"] = {k: v.tolist() for k, v in ent_results.items()}

    # ── Plots ─────────────────────────────────────────────────────────────────
    if save_plots:
        section("Generating Plots")
        try:
            from visualize import plot_causal_vs_bidirectional, demo_attention_patterns
            plot_causal_vs_bidirectional()
            demo_attention_patterns()
        except Exception as e:
            print(f"  visualize.py: {e}")

        try:
            from visualize_advanced import (
                plot_gqa_pattern, plot_position_encoding_comparison,
                plot_sliding_window_visual, plot_entropy_heatmap
            )
            plot_gqa_pattern()
            plot_position_encoding_comparison()
            plot_sliding_window_visual()
            plot_entropy_heatmap()
        except Exception as e:
            print(f"  visualize_advanced.py: {e}")

        try:
            from entropy import plot_entropy_analysis
            plot_entropy_analysis()
        except Exception as e:
            print(f"  entropy plots: {e}")

        try:
            from attention_io import plot_io_analysis
            plot_io_analysis(io_results)
        except Exception as e:
            print(f"  io analysis plot: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    report["elapsed_seconds"] = elapsed

    section("Summary")
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\nKey findings:")

    # extract speedups
    if fwd_results:
        best_speedup = max(r.get("speedup", 0) for r in fwd_results)
        best_seq = max(fwd_results, key=lambda r: r.get("speedup", 0))["seq_len"]
        print(f"  SDPA speedup: {best_speedup:.2f}x at seq_len={best_seq}")

    if kv_result:
        print(f"  KV cache speedup: {kv_result['speedup']:.2f}x")
        print(f"  KV cache memory: {kv_result['cache_mb']:.1f} MB")

    if io_results:
        max_io_red = max(r["io_reduction"] for r in io_results)
        print(f"  Max IO reduction (Flash): {max_io_red:.1f}x at seq_len={io_results[-1]['seq_len']}")

    # save report
    os.makedirs("results", exist_ok=True)
    report_path = "results/benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all FlashDance benchmarks")
    parser.add_argument("--full", action="store_true", help="Full benchmark (more seq_lens, more repeats)")
    parser.add_argument("--plots", action="store_true", help="Save all plots")
    args = parser.parse_args()

    run_suite(quick=not args.full, save_plots=args.plots)
