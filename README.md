<p align="center">
  <img
    src=".github/logo.png"
    align="center"
    width="100"
    alt="FlashDance"
    title="FlashDance"
  />
  <h1 align="center">FlashDance</h1>
</p>

<p align="center">
  Benchmark Flash Attention vs vanilla attention. How much faster is it really? When does it matter?
</p>

## What is this?

Flash Attention rewrites the attention computation to be IO-aware -- it avoids materializing the full NxN attention matrix in HBM. This project measures the real speedup across different sequence lengths.

## What we measure

- Forward pass speed at sequence lengths 128 to 4096
- Backward pass speed (training matters too)
- Peak memory usage
- Head dimension sweep (32, 64, 128)
- Batch size sweep (1 to 16)
- Dtype precision (float32, float16, bfloat16)
- Torch profiler breakdown
- Attention pattern visualization

## Key findings

- SDPA gets faster relative to vanilla as sequence length grows
- At short sequences (<256), the difference is small
- At 2K+ tokens, SDPA saves both time and memory significantly
- The backward pass speedup is often bigger than the forward pass
- On CUDA, Flash Attention kernel kicks in automatically through SDPA
- float16 and bfloat16 give extra speedup with minimal precision loss

## Setup

```bash
pip install -r requirements.txt
python -m benchmarks.benchmark --seq-len 512 1024 2048 4096
python -m benchmarks.benchmark --backward
python -m benchmarks.benchmark --save
python -m benchmarks.profile_attn --seq-len 1024
python -m viz.visualize
```

## Layout

| Path | What it holds |
|------|---------------|
| `flashdance/` | Library: attention variants (`attention`, `gqa`, `mla`, `sliding_window`, `alibi`, `cross_attention`), `rope`, `kv_cache`, `mini_transformer`, `long_context`, `speculative`, and analysis helpers (`entropy`, `attention_io`, `attention_score_analysis`, `memory_analysis`) |
| `benchmarks/` | Runnable benchmarks: `benchmark`, `stats_bench`, `throughput`, `compare`, `run_all`, `profile_attn`, and the rest. Run with `python -m benchmarks.<name>` from the repo root |
| `viz/` | Plotting: `visualize`, `visualize_advanced`. Run with `python -m viz.<name>` |
| `tests/` | pytest suite |
| `flashdance.ipynb` | Full analysis with plots |
