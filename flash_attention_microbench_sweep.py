"""
FlashAttention Microbenchmark: Llama 3.1 8B Decode Attention
Sweep over KV cache lengths: [512, 1024, 2048, 4096]
Measures both attention kernel time and host->device transfer time

Replaces FlashInfer with flash-attn due to installation constraints.
"""

import torch
from flash_attn import flash_attn_with_kvcache

# Llama 3.1 8B architecture
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

KV_LENS = [512, 1024, 2048, 4096]

WARMUP_ITERS = 50
BENCH_ITERS = 1000
TRANSFER_ITERS = 100


def benchmark_attention(kv_len: int) -> float:
    """Benchmark FlashAttention decode attention kernel. Returns avg time in µs."""
    device = torch.device("cuda")
    dtype = torch.float16

    # FlashAttention expects:
    #   q: (batch, seqlen_q, nheads, headdim)
    #   k_cache: (batch, seqlen_k, nheads_k, headdim)
    #   v_cache: (batch, seqlen_k, nheads_k, headdim)
    # For decode: seqlen_q = 1

    batch_size = 1
    q = torch.randn(
        batch_size, 1, NUM_QUERY_HEADS, HEAD_DIM, device=device, dtype=dtype
    )
    k_cache = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
    )

    # cache_seqlens tells FlashAttention how much of the cache is valid
    cache_seqlens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event.record()
    for _ in range(BENCH_ITERS):
        _ = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True
        )
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms / BENCH_ITERS) * 1000

    return avg_time_us


def benchmark_attention_batched(kv_len: int, batch_size: int) -> float:
    """Benchmark batched decode attention. Returns avg time in µs."""
    device = torch.device("cuda")
    dtype = torch.float16

    q = torch.randn(
        batch_size, 1, NUM_QUERY_HEADS, HEAD_DIM, device=device, dtype=dtype
    )
    k_cache = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
    )
    v_cache = torch.randn(
        batch_size, kv_len, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype
    )

    cache_seqlens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True
        )
    torch.cuda.synchronize()

    # Benchmark
    start_event.record()
    for _ in range(BENCH_ITERS):
        _ = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True
        )
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms / BENCH_ITERS) * 1000

    return avg_time_us


def benchmark_transfer(kv_len: int) -> tuple[float, float]:
    """Benchmark host->device transfer. Returns (pinned_us, pageable_us)."""
    device = torch.device("cuda")
    dtype = torch.float16

    # Separate K and V tensors to match FlashAttention layout
    # Shape: (batch=1, seq_len, num_kv_heads, head_dim)
    k_cache_pinned = torch.randn(
        1, kv_len, NUM_KV_HEADS, HEAD_DIM, device="cpu", dtype=dtype, pin_memory=True
    )
    v_cache_pinned = torch.randn(
        1, kv_len, NUM_KV_HEADS, HEAD_DIM, device="cpu", dtype=dtype, pin_memory=True
    )

    k_cache_paged = torch.randn(
        1, kv_len, NUM_KV_HEADS, HEAD_DIM, device="cpu", dtype=dtype, pin_memory=False
    )
    v_cache_paged = torch.randn(
        1, kv_len, NUM_KV_HEADS, HEAD_DIM, device="cpu", dtype=dtype, pin_memory=False
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup pinned
    for _ in range(10):
        _ = k_cache_pinned.to(device, non_blocking=False)
        _ = v_cache_pinned.to(device, non_blocking=False)
        torch.cuda.synchronize()

    # Benchmark pinned
    start_event.record()
    for _ in range(TRANSFER_ITERS):
        _ = k_cache_pinned.to(device, non_blocking=False)
        _ = v_cache_pinned.to(device, non_blocking=False)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    pinned_us = (start_event.elapsed_time(end_event) / TRANSFER_ITERS) * 1000

    # Warmup pageable
    for _ in range(10):
        _ = k_cache_paged.to(device, non_blocking=False)
        _ = v_cache_paged.to(device, non_blocking=False)
        torch.cuda.synchronize()

    # Benchmark pageable
    start_event.record()
    for _ in range(TRANSFER_ITERS):
        _ = k_cache_paged.to(device, non_blocking=False)
        _ = v_cache_paged.to(device, non_blocking=False)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    paged_us = (start_event.elapsed_time(end_event) / TRANSFER_ITERS) * 1000

    return pinned_us, paged_us


def compute_metrics(kv_len: int, attention_us: float, transfer_us: float):
    """Compute derived metrics."""
    # K + V cache size (both separate tensors now)
    kv_bytes = 2 * kv_len * NUM_KV_HEADS * HEAD_DIM * 2  # 2 for K+V, 2 for fp16

    # Attention metrics
    # FLOPs: QK^T = 2*seq*heads*dim, softmax ≈ 5*seq*heads, AV = 2*seq*heads*dim
    # Simplified: ~4 * kv_len * num_query_heads * head_dim
    total_flops = 4 * kv_len * NUM_QUERY_HEADS * HEAD_DIM
    attention_bw = (kv_bytes / (attention_us * 1e-6)) / 1e9
    attention_tflops = (total_flops / (attention_us * 1e-6)) / 1e12

    # Transfer metrics
    transfer_bw = (kv_bytes / (transfer_us * 1e-6)) / 1e9

    return {
        "kv_bytes": kv_bytes,
        "attention_bw": attention_bw,
        "attention_tflops": attention_tflops,
        "transfer_bw": transfer_bw,
        "ratio": transfer_us / attention_us,
    }


def main():
    print("=" * 80)
    print("FlashAttention Microbenchmark: Llama 3.1 8B Decode Attention")
    print("=" * 80)
    print(
        f"Config: {NUM_QUERY_HEADS} Q heads, {NUM_KV_HEADS} KV heads, {HEAD_DIM} head dim (GQA)"
    )
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    for kv_len in KV_LENS:
        print(f"Benchmarking KV length = {kv_len}...")

        attention_us = benchmark_attention(kv_len)
        pinned_us, paged_us = benchmark_transfer(kv_len)
        metrics = compute_metrics(kv_len, attention_us, pinned_us)

        results.append(
            {
                "kv_len": kv_len,
                "kv_kb": metrics["kv_bytes"] / 1024,
                "attention_us": attention_us,
                "pinned_us": pinned_us,
                "paged_us": paged_us,
                "attention_bw": metrics["attention_bw"],
                "transfer_bw": metrics["transfer_bw"],
                "ratio": metrics["ratio"],
            }
        )

    # Print results table
    print("\n" + "=" * 80)
    print("Results (batch_size=1)")
    print("=" * 80)

    # Header
    print(
        f"{'KV Len':>8} | {'KV Size':>10} | {'Attn (µs)':>10} | {'Xfer (µs)':>10} | "
        f"{'Attn BW':>10} | {'Xfer BW':>10} | {'Xfer/Attn':>10}"
    )
    print(
        f"{'':>8} | {'(KB)':>10} | {'':>10} | {'(pinned)':>10} | "
        f"{'(GB/s)':>10} | {'(GB/s)':>10} | {'ratio':>10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['kv_len']:>8} | {r['kv_kb']:>10.1f} | {r['attention_us']:>10.2f} | "
            f"{r['pinned_us']:>10.2f} | {r['attention_bw']:>10.1f} | "
            f"{r['transfer_bw']:>10.1f} | {r['ratio']:>10.1f}x"
        )

    # Batched benchmarks
    print("\n" + "=" * 80)
    print("Batched Decode (showing how attention time scales)")
    print("=" * 80)

    batch_sizes = [1, 4, 8, 16, 32]
    fixed_kv_len = 2048

    print(f"\nKV Length = {fixed_kv_len}")
    print(f"{'Batch':>8} | {'Attn (µs)':>12} | {'µs/request':>12} | {'Throughput':>12}")
    print("-" * 50)

    for bs in batch_sizes:
        attn_us = benchmark_attention_batched(fixed_kv_len, bs)
        per_req = attn_us / bs
        throughput = bs / (attn_us * 1e-6)  # requests per second
        print(
            f"{bs:>8} | {attn_us:>12.2f} | {per_req:>12.2f} | {throughput:>12.0f} req/s"
        )

    # Analysis
    print("\n" + "=" * 80)
    print("Analysis for Multi-Tier KV Cache")
    print("=" * 80)

    print("\nPinned vs Pageable transfer:")
    for r in results:
        print(
            f"  KV={r['kv_len']:>4}: pinned={r['pinned_us']:.1f}µs, "
            f"pageable={r['paged_us']:.1f}µs, "
            f"speedup={r['paged_us']/r['pinned_us']:.2f}x"
        )

    print("\nKey insight:")
    print(
        "  Transfer/Attention ratio > 1 means transfer CANNOT be hidden by decode attention alone"
    )
    print(
        "  This validates the need for chunked-prefill overlap (warm prefill) for tiered KV cache"
    )

    print("\nReference bandwidths:")
    print("  A100 HBM:     2,039 GB/s")
    print("  PCIe 4.0 x16:    32 GB/s")
    print("  PCIe 5.0 x16:    64 GB/s")
    print("  NVLink (A100):  600 GB/s")


if __name__ == "__main__":
    main()
