"""FlashDance: attention benchmarking and analysis toolkit.

A comprehensive library for understanding and benchmarking attention mechanisms
used in modern LLMs: Flash Attention, GQA, MLA, RoPE, ALiBi, and more.

Quick start:
    from flashdance.attention import vanilla_attention, sdpa_attention
    from flashdance.rope import apply_rope, precompute_freqs
    from flashdance.gqa import GroupedQueryAttention
    from flashdance.mla import MultiHeadLatentAttention
    from flashdance.sliding_window import SlidingWindowAttention
    from flashdance.alibi import ALiBiAttention
"""

from .attention import (
    vanilla_attention,
    vanilla_attention_with_scores,
    sdpa_attention,
    MultiHeadAttention,
    compare_dtypes,
    check_backends,
)

from .rope import (
    precompute_freqs,
    apply_rope,
    apply_rope_single,
    rotate_half,
    RoPECache,
    rope_attention,
)

from .gqa import (
    repeat_kv,
    grouped_query_attention,
    multi_query_attention,
    GroupedQueryAttention,
    kv_cache_memory_comparison,
)

from .mla import (
    MultiHeadLatentAttention,
    compare_kv_cache_sizes,
)

from .sliding_window import (
    sliding_window_mask,
    sliding_window_attention,
    SlidingWindowAttention,
)

from .alibi import (
    get_alibi_slopes,
    build_alibi_bias,
    alibi_attention,
    ALiBiAttention,
)

from .cross_attention import (
    cross_attention,
    cross_attention_with_mask,
    CrossAttention,
    TransformerDecoderLayer,
)

from .kv_cache import (
    KVCache,
    AttentionWithKVCache,
)

from .entropy import (
    attention_entropy,
    max_entropy,
    detect_attention_sinks,
)

__version__ = "0.2.0"
__all__ = [
    # vanilla / SDPA
    "vanilla_attention",
    "vanilla_attention_with_scores",
    "sdpa_attention",
    "MultiHeadAttention",
    "compare_dtypes",
    "check_backends",
    # RoPE
    "precompute_freqs",
    "apply_rope",
    "apply_rope_single",
    "rotate_half",
    "RoPECache",
    "rope_attention",
    # GQA / MQA
    "repeat_kv",
    "grouped_query_attention",
    "multi_query_attention",
    "GroupedQueryAttention",
    "kv_cache_memory_comparison",
    # MLA
    "MultiHeadLatentAttention",
    "compare_kv_cache_sizes",
    # Sliding Window
    "sliding_window_mask",
    "sliding_window_attention",
    "SlidingWindowAttention",
    # ALiBi
    "get_alibi_slopes",
    "build_alibi_bias",
    "alibi_attention",
    "ALiBiAttention",
    # Cross-attention
    "cross_attention",
    "cross_attention_with_mask",
    "CrossAttention",
    "TransformerDecoderLayer",
    # KV Cache
    "KVCache",
    "AttentionWithKVCache",
    # Entropy / Analysis
    "attention_entropy",
    "max_entropy",
    "detect_attention_sinks",
]
