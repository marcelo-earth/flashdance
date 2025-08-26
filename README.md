# FlashDance

Benchmark Flash Attention vs vanilla attention. How much faster is it really? When does it matter?

## What is this?

Flash Attention rewrites the attention computation to be IO-aware -- it avoids materializing the full NxN attention matrix in HBM. This project measures the real speedup across different sequence lengths.

## Setup

```bash
pip install -r requirements.txt
python benchmark.py --seq-len 2048
```

## Files

| File | What it does |
|------|-------------|
| `attention.py` | Vanilla and flash attention implementations |
| `benchmark.py` | Run speed and memory benchmarks |
| `flashdance.ipynb` | Full analysis with plots |
