#!/usr/bin/env python3
"""
Decode timing benchmark for Llama 3.1 8B layer.
Measures linear vs attention time breakdown across batch sizes.
Handles OOM gracefully for large context lengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import argparse
import gc
from transformers import AutoConfig


class RMSNorm(nn.Module):
    """Llama-style RMSNorm."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with random weights."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        
        # Attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Norms
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        # GQA groups
        self.num_kv_groups = self.num_heads // self.num_kv_heads
    
    def forward_timed(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> dict:
        """
        Decode forward with component timing.
        Skips RoPE (negligible element-wise ops for timing purposes).
        """
        batch_size = hidden_states.shape[0]
        
        # Events for timing
        events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for name in ['norm1', 'qkv', 'attn', 'o_proj', 'norm2', 'mlp']}
        
        # === Input LayerNorm ===
        events['norm1'][0].record()
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        events['norm1'][1].record()
        
        # === Q, K, V Projections ===
        events['qkv'][0].record()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        events['qkv'][1].record()
        
        # Reshape for attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Concat with cache (skip RoPE - negligible for timing)
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
        
        # Expand KV for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # === Attention Computation ===
        events['attn'][0].record()
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        events['attn'][1].record()
        
        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        
        # === O Projection ===
        events['o_proj'][0].record()
        attn_output = self.o_proj(attn_output)
        events['o_proj'][1].record()
        
        hidden_states = residual + attn_output
        
        # === Post-attention LayerNorm ===
        events['norm2'][0].record()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        events['norm2'][1].record()
        
        # === MLP (SwiGLU) ===
        events['mlp'][0].record()
        mlp_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        events['mlp'][1].record()
        
        hidden_states = residual + mlp_output
        
        # Synchronize and collect timings
        torch.cuda.synchronize()
        
        timings = {name: events[name][0].elapsed_time(events[name][1]) for name in events}
        
        return {
            'attn_linear_ms': timings['qkv'] + timings['o_proj'],
            'attention_ms': timings['attn'],
            'mlp_linear_ms': timings['mlp'],
            'other_ms': timings['norm1'] + timings['norm2'],
        }


def clear_gpu_memory():
    """Aggressively clear GPU memory after OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def estimate_memory_gb(batch_size: int, seq_len: int, config, dtype: torch.dtype) -> float:
    """Estimate memory required for KV cache in GB."""
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    head_dim = config.hidden_size // config.num_attention_heads
    
    # KV cache: 2 * batch * kv_heads * seq_len * head_dim
    kv_cache_bytes = 2 * batch_size * config.num_key_value_heads * seq_len * head_dim * bytes_per_element
    
    # Attention weights during computation: batch * num_heads * 1 * seq_len (for Q @ K^T)
    # After GQA expansion: batch * num_heads * seq_len * head_dim for expanded K, V
    expanded_kv_bytes = 2 * batch_size * config.num_attention_heads * seq_len * head_dim * bytes_per_element
    
    # Attention scores: batch * num_heads * 1 * seq_len
    attn_scores_bytes = batch_size * config.num_attention_heads * seq_len * bytes_per_element
    
    total_bytes = kv_cache_bytes + expanded_kv_bytes + attn_scores_bytes
    return total_bytes / (1024 ** 3)


def run_single_config(
    layer: nn.Module,
    config,
    batch_size: int,
    seq_len: int,
    num_warmup: int,
    num_runs: int,
    dtype: torch.dtype,
    device: str
) -> dict:
    """
    Run benchmark for a single configuration.
    Returns results dict with timing data or OOM marker.
    """
    head_dim = config.hidden_size // config.num_attention_heads
    
    result = {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'status': 'OK',
    }
    
    try:
        # Create inputs
        hidden_states = torch.randn(
            batch_size, 1, config.hidden_size,
            device=device, dtype=dtype
        )
        k_cache = torch.randn(
            batch_size, config.num_key_value_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, config.num_key_value_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer.forward_timed(hidden_states, k_cache, v_cache)
        
        # Benchmark
        all_timings = {k: [] for k in ['attn_linear_ms', 'attention_ms', 'mlp_linear_ms', 'other_ms']}
        
        with torch.no_grad():
            for _ in range(num_runs):
                timings = layer.forward_timed(hidden_states, k_cache, v_cache)
                for k, v in timings.items():
                    all_timings[k].append(v)
        
        # Compute statistics
        for k, v in all_timings.items():
            result[f'{k}_mean'] = sum(v) / len(v)
            result[f'{k}_std'] = torch.tensor(v).std().item()
        
        # Aggregates
        result['linear_total_ms'] = result['attn_linear_ms_mean'] + result['mlp_linear_ms_mean']
        result['total_ms'] = (result['attn_linear_ms_mean'] + result['attention_ms_mean'] +
                              result['mlp_linear_ms_mean'] + result['other_ms_mean'])
        result['linear_pct'] = 100 * result['linear_total_ms'] / result['total_ms']
        result['attention_pct'] = 100 * result['attention_ms_mean'] / result['total_ms']
        result['other_pct'] = 100 * result['other_ms_mean'] / result['total_ms']
        
        # Clean up tensors explicitly
        del hidden_states, k_cache, v_cache
        clear_gpu_memory()
        
    except torch.cuda.OutOfMemoryError as e:
        # Handle OOM gracefully
        result['status'] = 'OOM'
        
        # Fill timing fields with NaN to maintain CSV structure
        for suffix in ['_mean', '_std']:
            for key in ['attn_linear_ms', 'attention_ms', 'mlp_linear_ms', 'other_ms']:
                result[f'{key}{suffix}'] = float('nan')
        
        result['linear_total_ms'] = float('nan')
        result['total_ms'] = float('nan')
        result['linear_pct'] = float('nan')
        result['attention_pct'] = float('nan')
        result['other_pct'] = float('nan')
        
        # Estimate what memory would have been needed
        result['estimated_mem_gb'] = estimate_memory_gb(batch_size, seq_len, config, dtype)
        
        # Clear GPU memory for subsequent runs
        clear_gpu_memory()
        
        print(f"  OOM at batch_size={batch_size}, seq_len={seq_len} "
              f"(estimated ~{result['estimated_mem_gb']:.1f} GB needed)")
    
    return result


def run_benchmark(
    model_name: str,
    batch_sizes: list,
    seq_lengths: list,
    num_warmup: int = 10,
    num_runs: int = 50,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda'
) -> list:
    """Run decode timing benchmark with OOM handling."""
    
    print(f"Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=True)
    
    print(f"Architecture: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, intermediate={config.intermediate_size}")
    
    # Report GPU memory
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_mem_total:.1f} GB total")
    
    # Create layer with random weights
    layer = LlamaDecoderLayer(config).to(device=device, dtype=dtype)
    layer.eval()
    
    results = []
    oom_configs = []
    
    # Sort seq_lengths ascending so we can potentially skip larger ones after OOM
    seq_lengths_sorted = sorted(seq_lengths)
    
    for seq_len in seq_lengths_sorted:
        # Track if this seq_len OOMed for all batch sizes
        seq_len_all_oom = True
        
        for batch_size in batch_sizes:
            print(f"Running: batch_size={batch_size}, seq_len={seq_len}")
            
            # Estimate memory before attempting
            est_mem = estimate_memory_gb(batch_size, seq_len, config, dtype)
            print(f"  Estimated memory: ~{est_mem:.1f} GB")
            
            result = run_single_config(
                layer=layer,
                config=config,
                batch_size=batch_size,
                seq_len=seq_len,
                num_warmup=num_warmup,
                num_runs=num_runs,
                dtype=dtype,
                device=device
            )
            
            results.append(result)
            
            if result['status'] == 'OK':
                seq_len_all_oom = False
                print(f"  Total: {result['total_ms']:.3f}ms | "
                      f"Linear: {result['linear_pct']:.1f}% | "
                      f"Attention: {result['attention_pct']:.1f}% | "
                      f"Other: {result['other_pct']:.1f}%")
            else:
                oom_configs.append((batch_size, seq_len))
    
    # Summary of OOM cases
    if oom_configs:
        print(f"\n⚠️  OOM occurred for {len(oom_configs)} configurations:")
        for bs, sl in oom_configs:
            print(f"    batch_size={bs}, seq_len={sl}")
    
    return results


def save_results(results: list, filename: str):
    """Save results to CSV, including OOM markers."""
    if not results:
        return
    
    # Ensure consistent fieldnames across all results
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())
    
    # Define preferred order
    ordered_fields = [
        'batch_size', 'seq_len', 'status',
        'total_ms', 'linear_total_ms', 'linear_pct', 'attention_pct', 'other_pct',
        'attn_linear_ms_mean', 'attn_linear_ms_std',
        'attention_ms_mean', 'attention_ms_std',
        'mlp_linear_ms_mean', 'mlp_linear_ms_std',
        'other_ms_mean', 'other_ms_std',
        'estimated_mem_gb'
    ]
    
    # Add any fields that might be missing from ordered list
    fieldnames = [f for f in ordered_fields if f in all_fields]
    fieldnames.extend([f for f in sorted(all_fields) if f not in fieldnames])
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Decode timing benchmark for Llama 3.1 8B')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B',
                        help='HuggingFace model name (for config only)')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help='Batch sizes to test')
    parser.add_argument('--seq-lengths', type=int, nargs='+',
                        default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                        help='KV cache sequence lengths to test')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--runs', type=int, default=50,
                        help='Number of measurement iterations')
    parser.add_argument('--output', type=str, default='decode_timing_results.csv',
                        help='Output CSV filename')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                        help='Data type')
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Config from: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print("-" * 60)
    
    results = run_benchmark(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_warmup=args.warmup,
        num_runs=args.runs,
        dtype=dtype
    )
    
    save_results(results, args.output)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Batch':<8} {'Seq':<10} {'Status':<8} {'Total(ms)':<12} {'Linear%':<10} {'Attn%':<10} {'Other%':<10}")
    print("-" * 100)
    for r in results:
        if r['status'] == 'OK':
            print(f"{r['batch_size']:<8} {r['seq_len']:<10} {r['status']:<8} {r['total_ms']:<12.3f} "
                  f"{r['linear_pct']:<10.1f} {r['attention_pct']:<10.1f} {r['other_pct']:<10.1f}")
        else:
            est_mem = r.get('estimated_mem_gb', 0)
            print(f"{r['batch_size']:<8} {r['seq_len']:<10} {r['status']:<8} {'N/A':<12} "
                  f"{'N/A':<10} {'N/A':<10} (est. {est_mem:.1f} GB)")
    
    # Print statistics
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    oom_count = sum(1 for r in results if r['status'] == 'OOM')
    print(f"\nTotal: {len(results)} configs | OK: {ok_count} | OOM: {oom_count}")


if __name__ == '__main__':
    main()