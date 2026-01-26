#!/usr/bin/env python3
"""
H2D Transfer Analysis for KV Cache Management Research
Measures host-to-device bandwidth and calculates how much data
can be transferred within given time budgets from CSV.
"""

import torch
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime


def benchmark_h2d_bandwidth(size_gb: float = 1.0, warmup: int = 3, iters: int = 10) -> dict:
    """
    Benchmark host-to-device transfer bandwidth.
    
    Args:
        size_gb: Size of data to transfer in GB
        warmup: Number of warmup iterations
        iters: Number of measurement iterations
    
    Returns:
        Dict with bandwidth stats
    """
    size_bytes = int(size_gb * 1024**3)
    num_elements = size_bytes // 2  # FP16 = 2 bytes per element
    
    # Allocate pinned host memory (faster transfers)
    host_tensor = torch.empty(num_elements, dtype=torch.float16, pin_memory=True)
    
    # Warmup
    for _ in range(warmup):
        device_tensor = host_tensor.to('cuda', non_blocking=False)
        torch.cuda.synchronize()
        del device_tensor
        torch.cuda.empty_cache()
    
    # Measure
    times_ms = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        device_tensor = host_tensor.to('cuda', non_blocking=False)
        end.record()
        
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        
        del device_tensor
        torch.cuda.empty_cache()
    
    avg_time_ms = sum(times_ms) / len(times_ms)
    bandwidth_gbps = size_gb / (avg_time_ms / 1000)
    
    del host_tensor
    
    return {
        'bandwidth_gbps': bandwidth_gbps,
        'avg_time_ms': avg_time_ms,
        'min_time_ms': min(times_ms),
        'max_time_ms': max(times_ms),
        'std_time_ms': (sum((t - avg_time_ms)**2 for t in times_ms) / len(times_ms))**0.5,
        'size_gb': size_gb,
        'iters': iters,
        'gpu_name': torch.cuda.get_device_name(0),
        'all_times_ms': times_ms
    }


def analyze_csv(csv_path: str, bandwidth_gbps: float, output_path: str = None) -> pd.DataFrame:
    """
    Read CSV and calculate transferable data for each (batch_size, seq_len) setting.
    
    Args:
        csv_path: Path to input CSV
        bandwidth_gbps: Measured H2D bandwidth in GB/s
        output_path: Optional path to save results
    
    Returns:
        DataFrame with analysis results
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['batch_size', 'seq_len', 'total_ms']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")
    
    # Calculate transferable data
    df['time_sec'] = df['total_ms'] / 1000.0
    df['transferable_gb'] = df['time_sec'] * bandwidth_gbps
    df['transferable_mb'] = df['transferable_gb'] * 1024
    
    # Also compute in terms of KV cache entries (assuming FP16, 2 bytes per element)
    # KV cache size per token per layer = 2 * num_heads * head_dim * 2 bytes
    # For Llama 3.1 8B: 2 * 8 * 128 * 2 = 4096 bytes = 4 KB per token per layer (32 layers total)
    # So ~128 KB per token for full KV cache
    kv_per_token_kb = 128  # Llama 3.1 8B approximate
    df['transferable_tokens_kv'] = (df['transferable_mb'] * 1024) / kv_per_token_kb
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='H2D Transfer Analysis for KV Cache Research')
    parser.add_argument('csv_path', type=str, help='Path to CSV with batch_size, seq_len, total_ms columns')
    parser.add_argument('--benchmark-size', type=float, default=1.0, help='Size in GB for bandwidth benchmark (default: 1.0)')
    parser.add_argument('--bandwidth', type=float, default=None, help='Skip benchmark, use this bandwidth (GB/s)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (default: input_h2d_analysis.csv)')
    parser.add_argument('--iters', type=int, default=10, help='Benchmark iterations (default: 10)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = Path(args.csv_path).stem
    
    print("=" * 60)
    print("H2D Transfer Analysis for KV Cache Management")
    print("=" * 60)
    
    # Get bandwidth
    benchmark_info = {}
    if args.bandwidth:
        bandwidth = args.bandwidth
        benchmark_info = {'bandwidth_gbps': bandwidth, 'source': 'user_provided'}
        print(f"\nUsing provided bandwidth: {bandwidth:.2f} GB/s")
    else:
        print(f"\nBenchmarking H2D bandwidth with {args.benchmark_size} GB transfer...")
        print(f"  Iterations: {args.iters}")
        
        benchmark_info = benchmark_h2d_bandwidth(
            size_gb=args.benchmark_size,
            iters=args.iters
        )
        benchmark_info['source'] = 'measured'
        bandwidth = benchmark_info['bandwidth_gbps']
        
        print(f"\n  Results:")
        print(f"    GPU: {benchmark_info['gpu_name']}")
        print(f"    Average time: {benchmark_info['avg_time_ms']:.2f} ms")
        print(f"    Bandwidth: {bandwidth:.2f} GB/s")
        print(f"    Min/Max: {benchmark_info['min_time_ms']:.2f} / {benchmark_info['max_time_ms']:.2f} ms")
        
        # Save benchmark results to separate CSV
        benchmark_csv = f"{input_stem}_h2d_benchmark_{timestamp}.csv"
        benchmark_df = pd.DataFrame([{
            'gpu_name': benchmark_info['gpu_name'],
            'bandwidth_gbps': bandwidth,
            'avg_time_ms': benchmark_info['avg_time_ms'],
            'min_time_ms': benchmark_info['min_time_ms'],
            'max_time_ms': benchmark_info['max_time_ms'],
            'std_time_ms': benchmark_info['std_time_ms'],
            'transfer_size_gb': benchmark_info['size_gb'],
            'iterations': benchmark_info['iters'],
            'timestamp': timestamp
        }])
        benchmark_df.to_csv(benchmark_csv, index=False)
        print(f"\n  Benchmark saved to: {benchmark_csv}")
    
    # Analyze CSV
    print(f"\nAnalyzing: {args.csv_path}")
    output_path = args.output or f"{input_stem}_h2d_analysis_{timestamp}.csv"
    
    df = analyze_csv(args.csv_path, bandwidth, output_path)
    
    # Add metadata columns for plotting convenience
    df['bandwidth_gbps'] = bandwidth
    df['timestamp'] = timestamp
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\n{'batch_size':>10} {'seq_len':>10} {'total_ms':>10} {'xfer_gb':>10} {'xfer_mb':>10} {'kv_tokens':>12}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{int(row['batch_size']):>10} {int(row['seq_len']):>10} {row['total_ms']:>10.2f} "
              f"{row['transferable_gb']:>10.4f} {row['transferable_mb']:>10.2f} {row['transferable_tokens_kv']:>12.0f}")
    
    print("\n" + "=" * 60)
    print(f"Bandwidth used: {bandwidth:.2f} GB/s")
    print(f"KV tokens assumes Llama 3.1 8B (~128 KB per token full KV)")
    print(f"\nOutput CSV: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
