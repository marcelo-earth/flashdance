"""Vanilla vs SDPA forward timing with a design that supports a confidence interval.

benchmark.py times vanilla then SDPA in a fixed order inside one process, so its
repeats are not independent and the order can favor one method. Here:

- each block runs in a fresh process: blocks are the independent replicates
- condition order inside a block is shuffled from a recorded seed
- inner repeats only sharpen each block's median; they are not extra samples
- the speedup per seq_len is exp(mean log-ratio) over blocks, with a paired t CI

Usage:
    python stats_bench.py --blocks 6 --seq-len 512 1024 2048
    python stats_bench.py --analyze results/stats_bench_<stamp>.csv
"""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["block", "position", "seq_len", "method", "median_ms", "device", "torch"]


def run_block(order, batch_size, n_heads, head_dim, warmup, inner):
    """Time each (seq_len, method) in `order` once. Runs inside a fresh process."""
    import torch
    from attention import vanilla_attention, sdpa_attention

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    fns = {"vanilla": vanilla_attention, "sdpa": sdpa_attention}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    rows = []
    for position, (seq_len, method) in enumerate(order):
        gen = torch.Generator().manual_seed(seq_len)
        q, k, v = (torch.randn(batch_size, n_heads, seq_len, head_dim, generator=gen).to(device) for _ in range(3))
        fn = fns[method]
        for _ in range(warmup):
            fn(q, k, v)
        sync()
        times = []
        for _ in range(inner):
            start = time.perf_counter()
            fn(q, k, v)
            sync()
            times.append(time.perf_counter() - start)
        rows.append({
            "position": position,
            "seq_len": seq_len,
            "method": method,
            "median_ms": float(np.median(times)) * 1000,
            "device": device,
            "torch": torch.__version__,
        })
        del q, k, v
    return rows


def run(args):
    rng = np.random.default_rng(args.seed)
    conditions = [(s, m) for s in args.seq_len for m in ("vanilla", "sdpa")]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(HERE, "results", f"stats_bench_{stamp}.csv")
    meta = {
        "seed": args.seed, "blocks": args.blocks, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "n_heads": args.n_heads, "head_dim": args.head_dim,
        "warmup": args.warmup, "inner": args.inner, "machine": platform.platform(),
        "processor": platform.processor(), "python": platform.python_version(),
    }
    with open(path.replace(".csv", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for block in range(args.blocks):
            order = [conditions[i] for i in rng.permutation(len(conditions))]
            payload = json.dumps({
                "order": order, "batch_size": args.batch_size, "n_heads": args.n_heads,
                "head_dim": args.head_dim, "warmup": args.warmup, "inner": args.inner,
            })
            out = subprocess.run([sys.executable, __file__, "--worker", payload],
                                 cwd=HERE, capture_output=True, text=True, check=True)
            for row in json.loads(out.stdout.strip().splitlines()[-1]):
                writer.writerow({"block": block, **row})
            f.flush()
            print(f"block {block + 1}/{args.blocks} done")
    print(f"saved {path}")
    analyze(path)


def analyze(path):
    from scipy import stats

    with open(path) as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(int(r["seq_len"]), {}).setdefault(int(r["block"]), {})[r["method"]] = r
    print(f"\n{path}")
    print(f"{'seq_len':>7} {'n':>3} {'vanilla_ms':>10} {'sdpa_ms':>8} {'speedup':>8} {'95% CI':>17} {'sd_log':>7} {'order_gap':>9}")
    summary = []
    for seq_len in sorted(by):
        pairs = [b for b in by[seq_len].values() if {"vanilla", "sdpa"} <= b.keys()]
        van = np.array([float(b["vanilla"]["median_ms"]) for b in pairs])
        sdp = np.array([float(b["sdpa"]["median_ms"]) for b in pairs])
        d = np.log(van) - np.log(sdp)
        n = len(d)
        mean, sd = d.mean(), d.std(ddof=1)
        half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        # Order check: does the log-ratio shift when vanilla ran first?
        first = np.array([int(b["vanilla"]["position"]) < int(b["sdpa"]["position"]) for b in pairs])
        gap = d[first].mean() - d[~first].mean() if first.any() and (~first).any() else float("nan")
        print(f"{seq_len:>7} {n:>3} {np.median(van):>10.3f} {np.median(sdp):>8.3f} {math.exp(mean):>7.2f}x "
              f"[{math.exp(mean - half):.2f}, {math.exp(mean + half):.2f}] {sd:>7.3f} {gap:>9.3f}")
        summary.append({"seq_len": seq_len, "n": n, "speedup": math.exp(mean),
                        "ci_low": math.exp(mean - half), "ci_high": math.exp(mean + half), "sd_log": sd})
    print("\nsd_log: SD of log(vanilla/sdpa) across blocks; the input for statistical-power.")
    print("order_gap: mean log-ratio when vanilla ran first minus when it ran second.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seq-len", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--analyze", help="analyze an existing CSV instead of running")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        p = json.loads(args.worker)
        rows = run_block([tuple(c) for c in p["order"]], p["batch_size"], p["n_heads"],
                         p["head_dim"], p["warmup"], p["inner"])
        print(json.dumps(rows))
    elif args.analyze:
        analyze(args.analyze)
    else:
        run(args)


if __name__ == "__main__":
    main()
