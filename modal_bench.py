"""Run benchmarks.stats_bench on a Modal GPU (CUDA arm of the forward experiment).

Usage:
    FD_RUN=cuda-fwd-a100-pilot modal run modal_bench.py --gpu a100 --blocks 3
    FD_RUN=cuda-fwd-h100-pilot modal run modal_bench.py --gpu h100 --blocks 3
    FD_RUN=cuda-fwd-a100-001 modal run modal_bench.py --gpu a100 --blocks 15 --seed 20261009

FD_RUN tags the app for experiment-hub's costs.py. Use a new value per run.
Defaults follow the pre-registered protocol: bf16, flash backend forced (raises
instead of falling back), batch 4, 8 heads, head dim 64, warmup 3, inner 30.
The CSV and meta land in results/ named after FD_RUN.
"""

import json
import os
import re
from argparse import Namespace
from pathlib import Path

import modal

RUN_ID = os.environ.get("FD_RUN", "untitled")
if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", RUN_ID):
    raise ValueError("FD_RUN must be 1 to 96 letters, digits, hyphens or underscores.")

# Timeouts are the spend cap: about $1.5 on an A100 and $2.5 on an H100 at list price.
TIMEOUT_SECONDS = {"a100": 40 * 60, "h100": 35 * 60}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy", "scipy")
    .add_local_python_source("flashdance", "benchmarks")
)

app = modal.App("flashdance-bench", image=image,
                tags={"project": "flashdance", "run": RUN_ID})


def _bench(cfg: dict) -> dict:
    import torch
    from benchmarks import stats_bench

    results = Path(stats_bench.HERE) / "results"
    before = set(results.glob("stats_bench_*.csv")) if results.exists() else set()
    stats_bench.run(Namespace(max_load=0, max_wait=0, **cfg))
    (csv_path,) = set(results.glob("stats_bench_*.csv")) - before
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text())
    meta.update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, torch=torch.__version__)
    return {"csv": csv_path.read_text(), "meta": meta}


@app.function(gpu="A100-40GB", timeout=TIMEOUT_SECONDS["a100"], retries=0, max_containers=1)
def bench_a100(cfg: dict) -> dict:
    return _bench(cfg)


@app.function(gpu="H100", timeout=TIMEOUT_SECONDS["h100"], retries=0, max_containers=1)
def bench_h100(cfg: dict) -> dict:
    return _bench(cfg)


@app.local_entrypoint()
def main(gpu: str = "a100", blocks: int = 3, seq_len: str = "512,1024,2048,4096,8192",
         seed: int = 20261009, dtype: str = "bfloat16", sdpa_backend: str = "flash"):
    cfg = {"blocks": blocks, "seq_len": [int(s) for s in seq_len.split(",")],
           "batch_size": 4, "n_heads": 8, "head_dim": 64, "warmup": 3, "inner": 30,
           "dtype": dtype, "sdpa_backend": sdpa_backend, "seed": seed}
    fn = {"a100": bench_a100, "h100": bench_h100}[gpu]
    out = fn.remote(cfg)
    out["meta"]["run"] = RUN_ID  # FD_RUN is only set locally, not in the container

    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    csv_path = results / f"stats_bench_{RUN_ID}.csv"
    csv_path.write_text(out["csv"])
    csv_path.with_suffix(".meta.json").write_text(json.dumps(out["meta"], indent=2))
    print(f"saved {csv_path} ({out['meta']['gpu']}, torch {out['meta']['torch']})")

    from benchmarks.stats_bench import analyze
    analyze(str(csv_path))
