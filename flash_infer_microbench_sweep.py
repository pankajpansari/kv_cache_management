"""
FlashInfer Microbenchmark: Llama 3.1 8B Decode Attention
Sweep over KV cache lengths: [512, 1024, 2048, 4096]
Measures both attention kernel time and host->device transfer time
"""

import torch
import flashinfer

# Llama 3.1 8B architecture
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

KV_LENS = [512, 1024, 2048, 4096]

WARMUP_ITERS = 50
BENCH_ITERS = 1000
TRANSFER_ITERS = 100


def benchmark_attention(kv_len: int) -> float:
    """Benchmark FlashInfer decode attention kernel. Returns avg time in µs."""
    device = torch.device("cuda")
    dtype = torch.float16
    
    q = torch.randn(NUM_QUERY_HEADS, HEAD_DIM, device=device, dtype=dtype)
    kv_cache = torch.randn(kv_len, 2, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = flashinfer.single_decode_with_kv_cache(q, k_cache, v_cache, use_tensor_cores=True)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event.record()
    for _ in range(BENCH_ITERS):
        _ = flashinfer.single_decode_with_kv_cache(q, k_cache, v_cache, use_tensor_cores=True)
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_us = (total_time_ms / BENCH_ITERS) * 1000
    
    return avg_time_us


def benchmark_transfer(kv_len: int) -> tuple[float, float]:
    """Benchmark host->device transfer. Returns (pinned_us, pageable_us)."""
    device = torch.device("cuda")
    dtype = torch.float16
    
    kv_cache_pinned = torch.randn(
        kv_len, 2, NUM_KV_HEADS, HEAD_DIM,
        device="cpu", dtype=dtype, pin_memory=True
    )
    kv_cache_paged = torch.randn(
        kv_len, 2, NUM_KV_HEADS, HEAD_DIM,
        device="cpu", dtype=dtype, pin_memory=False
    )
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup pinned
    for _ in range(10):
        _ = kv_cache_pinned.to(device, non_blocking=False)
        torch.cuda.synchronize()
    
    # Benchmark pinned
    start_event.record()
    for _ in range(TRANSFER_ITERS):
        _ = kv_cache_pinned.to(device, non_blocking=False)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    pinned_us = (start_event.elapsed_time(end_event) / TRANSFER_ITERS) * 1000
    
    # Warmup pageable
    for _ in range(10):
        _ = kv_cache_paged.to(device, non_blocking=False)
        torch.cuda.synchronize()
    
    # Benchmark pageable
    start_event.record()
    for _ in range(TRANSFER_ITERS):
        _ = kv_cache_paged.to(device, non_blocking=False)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    paged_us = (start_event.elapsed_time(end_event) / TRANSFER_ITERS) * 1000
    
    return pinned_us, paged_us


def compute_metrics(kv_len: int, attention_us: float, transfer_us: float):
    """Compute derived metrics."""
    kv_bytes = kv_len * 2 * NUM_KV_HEADS * HEAD_DIM * 2
    
    # Attention metrics
    total_flops = 2 * 2 * kv_len * NUM_QUERY_HEADS * HEAD_DIM  # QK + AV
    attention_bw = (kv_bytes / (attention_us * 1e-6)) / 1e9
    attention_tflops = (total_flops / (attention_us * 1e-6)) / 1e12
    
    # Transfer metrics
    transfer_bw = (kv_bytes / (transfer_us * 1e-6)) / 1e9
    
    return {
        "kv_bytes": kv_bytes,
        "attention_bw": attention_bw,
        "transfer_bw": transfer_bw,
        "ratio": transfer_us / attention_us,
    }


def main():
    print("="*80)
    print("FlashInfer Microbenchmark: Llama 3.1 8B Decode Attention")
    print("="*80)
    print(f"Config: {NUM_QUERY_HEADS} Q heads, {NUM_KV_HEADS} KV heads, {HEAD_DIM} head dim (GQA)")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    results = []
    
    for kv_len in KV_LENS:
        print(f"Benchmarking KV length = {kv_len}...")
        
        attention_us = benchmark_attention(kv_len)
        pinned_us, paged_us = benchmark_transfer(kv_len)
        metrics = compute_metrics(kv_len, attention_us, pinned_us)
        
        results.append({
            "kv_len": kv_len,
            "kv_kb": metrics["kv_bytes"] / 1024,
            "attention_us": attention_us,
            "pinned_us": pinned_us,
            "paged_us": paged_us,
            "attention_bw": metrics["attention_bw"],
            "transfer_bw": metrics["transfer_bw"],
            "ratio": metrics["ratio"],
        })
    
    # Print results table
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    
    # Header
    print(f"{'KV Len':>8} | {'KV Size':>10} | {'Attn (µs)':>10} | {'Xfer (µs)':>10} | "
          f"{'Attn BW':>10} | {'Xfer BW':>10} | {'Xfer/Attn':>10}")
    print(f"{'':>8} | {'(KB)':>10} | {'':>10} | {'(pinned)':>10} | "
          f"{'(GB/s)':>10} | {'(GB/s)':>10} | {'ratio':>10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['kv_len']:>8} | {r['kv_kb']:>10.1f} | {r['attention_us']:>10.2f} | "
              f"{r['pinned_us']:>10.2f} | {r['attention_bw']:>10.1f} | "
              f"{r['transfer_bw']:>10.1f} | {r['ratio']:>10.1f}x")
    
    # Analysis
    print("\n" + "="*80)
    print("Analysis for Multi-Tier KV Cache")
    print("="*80)
    
    print("\nPinned vs Pageable transfer:")
    for r in results:
        paged_r = next(x for x in results if x['kv_len'] == r['kv_len'])
        print(f"  KV={r['kv_len']:>4}: pinned={r['pinned_us']:.1f}µs, "
              f"pageable={paged_r['paged_us']:.1f}µs")
    
    print("\nKey insight:")
    print("  Transfer/Attention ratio > 1 means transfer CANNOT be hidden by decode attention")
    print("  This is why chunked-prefill overlap (warm prefill) is necessary for tiered KV cache")
    
    print("\nReference bandwidths:")
    print("  A100 HBM:     2,039 GB/s")
    print("  PCIe 4.0 x16:    32 GB/s")
    print("  PCIe 5.0 x16:    64 GB/s")


if __name__ == "__main__":
    main()