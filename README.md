# FlashDance

Benchmark Flash Attention vs vanilla attention. How much faster is it really? When does it matter?

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
python benchmark.py --seq-len 512 1024 2048 4096
python benchmark.py --backward
python benchmark.py --save
python profile_attn.py --seq-len 1024
python visualize.py
```

## Files

| File | What it does |
|------|-------------|
| `attention.py` | Vanilla and SDPA attention + dtype comparison |
| `benchmark.py` | Speed, memory, head dim, and batch size benchmarks |
| `profile_attn.py` | Torch profiler for attention ops |
| `visualize.py` | Attention pattern heatmaps |
| `flashdance.ipynb` | Full analysis with plots |
