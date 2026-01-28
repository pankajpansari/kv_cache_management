#!/usr/bin/env python3
"""
Decode timing benchmark for Llama 3.1 8B layer using FlashAttention.
Measures linear vs attention time breakdown across batch sizes.

FlashAttention benefits for decode:
  - Native GQA support without expand/repeat_interleave overhead
  - Fused kernel with memory-efficient attention
  - flash_attn_with_kvcache API optimized for incremental decoding
"""

import argparse
import csv
import gc
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

# FlashAttention import
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    print("Warning: FlashAttention not available, will use manual attention")


class AttentionBackend(Enum):
    MANUAL = "manual"
    FLASH_ATTN = "flash_attn"


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


@dataclass
class AttentionConfig:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    
    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads


def manual_decode_attention(
    q: torch.Tensor,        # [batch, 1, num_heads, head_dim]
    k_cache: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """Manual attention with GQA expansion (baseline)."""
    # Transpose: [batch, heads, seq, dim]
    q = q.transpose(1, 2)
    k = k_cache.transpose(1, 2)
    v = v_cache.transpose(1, 2)
    
    # Expand KV for GQA - this is the inefficient part that FlashAttention avoids
    k = k.repeat_interleave(config.num_kv_groups, dim=1)
    v = v.repeat_interleave(config.num_kv_groups, dim=1)
    
    scale = 1.0 / (config.head_dim ** 0.5)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)
    
    return output.transpose(1, 2)


def flash_attn_decode_attention(
    q: torch.Tensor,        # [batch, 1, num_heads, head_dim]
    k_cache: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """
    FlashAttention decode using kv cache API.
    
    flash_attn_with_kvcache handles GQA natively - no expansion needed.
    Expected shapes:
      q: [batch, seqlen_q, num_heads, head_dim]
      k_cache, v_cache: [batch, seqlen_k, num_kv_heads, head_dim]
    """
    output = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        softmax_scale=1.0 / (config.head_dim ** 0.5),
    )
    return output


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer with configurable attention backend."""
    
    def __init__(self, config, backend: AttentionBackend = AttentionBackend.FLASH_ATTN):
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
        
        # Attention config
        self.attn_config = AttentionConfig(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim
        )
        
        # Select backend
        self.backend = backend
        self._validate_backend()
        
    def _validate_backend(self):
        if self.backend == AttentionBackend.FLASH_ATTN and not FLASH_ATTN_AVAILABLE:
            print("Warning: FlashAttention requested but not available, falling back to manual")
            self.backend = AttentionBackend.MANUAL
    
    def _attention(self, q, k_cache, v_cache):
        if self.backend == AttentionBackend.FLASH_ATTN:
            return flash_attn_decode_attention(q, k_cache, v_cache, self.attn_config)
        else:
            return manual_decode_attention(q, k_cache, v_cache, self.attn_config)
    
    def forward_timed(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> dict:
        """
        Decode forward with component timing.
        
        KV cache format: [batch, seq_len, num_kv_heads, head_dim]
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
        _ = self.k_proj(hidden_states)  # Would be appended to cache in real inference
        _ = self.v_proj(hidden_states)
        events['qkv'][1].record()
        
        # Reshape Q: [batch, 1, num_heads, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        
        # === Attention Computation ===
        events['attn'][0].record()
        attn_output = self._attention(q, k_cache, v_cache)
        events['attn'][1].record()
        
        # Reshape: [batch, 1, num_heads, head_dim] -> [batch, 1, hidden_size]
        attn_output = attn_output.reshape(batch_size, 1, -1)
        
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
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def estimate_memory_gb(batch_size: int, seq_len: int, config, dtype: torch.dtype,
                       backend: AttentionBackend) -> float:
    """
    Estimate memory required for KV cache.
    Note: FlashAttention doesn't need GQA expansion, so uses less memory.
    """
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Base KV cache: 2 * batch * kv_heads * seq_len * head_dim
    kv_cache_bytes = 2 * batch_size * config.num_key_value_heads * seq_len * head_dim * bytes_per_element
    
    # Manual attention needs GQA expansion + attention scores
    if backend == AttentionBackend.MANUAL:
        # Expanded KV: 2 * batch * num_heads * seq_len * head_dim
        expanded_kv_bytes = 2 * batch_size * config.num_attention_heads * seq_len * head_dim * bytes_per_element
        # Attention scores: batch * num_heads * 1 * seq_len
        attn_scores_bytes = batch_size * config.num_attention_heads * seq_len * bytes_per_element
        kv_cache_bytes += expanded_kv_bytes + attn_scores_bytes
    
    return kv_cache_bytes / (1024 ** 3)


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
    """Run benchmark for a single configuration."""
    head_dim = config.hidden_size // config.num_attention_heads

    result = {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'status': 'OK',
        'backend': layer.backend.value,
    }
    
    try:
        # Create inputs
        hidden_states = torch.randn(
            batch_size, 1, config.hidden_size,
            device=device, dtype=dtype
        )
        
        # KV cache: [batch, seq_len, num_kv_heads, head_dim]
        k_cache = torch.randn(
            batch_size, seq_len, config.num_key_value_heads, head_dim,
            device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, seq_len, config.num_key_value_heads, head_dim,
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
        
        del hidden_states, k_cache, v_cache
        clear_gpu_memory()
        
    except torch.cuda.OutOfMemoryError:
        result['status'] = 'OOM'
        for suffix in ['_mean', '_std']:
            for key in ['attn_linear_ms', 'attention_ms', 'mlp_linear_ms', 'other_ms']:
                result[f'{key}{suffix}'] = float('nan')
        
        result['linear_total_ms'] = float('nan')
        result['total_ms'] = float('nan')
        result['linear_pct'] = float('nan')
        result['attention_pct'] = float('nan')
        result['other_pct'] = float('nan')
        result['estimated_mem_gb'] = estimate_memory_gb(batch_size, seq_len, config, dtype, layer.backend)
        
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
    device: str = 'cuda',
    backend: AttentionBackend = AttentionBackend.FLASH_ATTN
) -> list:
    """Run decode timing benchmark."""
    
    print(f"Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=True)
    
    print(f"Architecture: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, intermediate={config.intermediate_size}")
    
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_mem_total:.1f} GB total")
    
    # Create layer
    layer = LlamaDecoderLayer(config, backend=backend).to(device=device, dtype=dtype)
    layer.eval()
    
    print(f"Attention backend: {layer.backend.value}")
    
    results = []
    oom_configs = []
    
    seq_lengths_sorted = sorted(seq_lengths)
    
    for seq_len in seq_lengths_sorted:
        for batch_size in batch_sizes:
            print(f"Running: batch_size={batch_size}, seq_len={seq_len}")
            
            est_mem = estimate_memory_gb(batch_size, seq_len, config, dtype, layer.backend)
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
                print(f"  Total: {result['total_ms']:.3f}ms | "
                      f"Linear: {result['linear_pct']:.1f}% | "
                      f"Attention: {result['attention_pct']:.1f}% | "
                      f"Other: {result['other_pct']:.1f}%")
            else:
                oom_configs.append((batch_size, seq_len))
    
    if oom_configs:
        print(f"\n⚠️  OOM occurred for {len(oom_configs)} configurations:")
        for bs, sl in oom_configs:
            print(f"    batch_size={bs}, seq_len={sl}")
    
    return results


def save_results(results: list, filename: str):
    """Save results to CSV."""
    if not results:
        return
    
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())
    
    ordered_fields = [
        'batch_size', 'seq_len', 'status', 'backend',
        'total_ms', 'linear_total_ms', 'linear_pct', 'attention_pct', 'other_pct',
        'attn_linear_ms_mean', 'attn_linear_ms_std',
        'attention_ms_mean', 'attention_ms_std',
        'mlp_linear_ms_mean', 'mlp_linear_ms_std',
        'other_ms_mean', 'other_ms_std',
        'estimated_mem_gb'
    ]
    
    fieldnames = [f for f in ordered_fields if f in all_fields]
    fieldnames.extend([f for f in sorted(all_fields) if f not in fieldnames])
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Decode timing benchmark with FlashAttention')
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
    parser.add_argument('--output', type=str, default='decode_timing_flash_attn.csv',
                        help='Output CSV filename')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                        help='Data type')
    parser.add_argument('--backend', type=str, default='flash_attn',
                        choices=['manual', 'flash_attn', 'all'],
                        help='Attention backend (or "all" to compare both)')
    
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    print(f"Config from: {args.model}")
    print(f"Dtype: {args.dtype}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print("-" * 60)
    
    # Determine backends to run
    if args.backend == 'all':
        backends = [AttentionBackend.MANUAL]
        if FLASH_ATTN_AVAILABLE:
            backends.append(AttentionBackend.FLASH_ATTN)
    else:
        backends = [AttentionBackend(args.backend)]
    
    all_results = []
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {backend.value}")
        print('='*60)
        
        results = run_benchmark(
            model_name=args.model,
            batch_sizes=args.batch_sizes,
            seq_lengths=args.seq_lengths,
            num_warmup=args.warmup,
            num_runs=args.runs,
            dtype=dtype,
            backend=backend
        )
        all_results.extend(results)
    
    save_results(all_results, args.output)
    
    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"{'Backend':<12} {'Batch':<8} {'Seq':<10} {'Status':<8} {'Total(ms)':<12} {'Linear%':<10} {'Attn%':<10} {'Other%':<10}")
    print("-" * 110)
    for r in all_results:
        if r['status'] == 'OK':
            print(f"{r['backend']:<12} {r['batch_size']:<8} {r['seq_len']:<10} {r['status']:<8} {r['total_ms']:<12.3f} "
                  f"{r['linear_pct']:<10.1f} {r['attention_pct']:<10.1f} {r['other_pct']:<10.1f}")
        else:
            est_mem = r.get('estimated_mem_gb', 0)
            print(f"{r['backend']:<12} {r['batch_size']:<8} {r['seq_len']:<10} {r['status']:<8} {'N/A':<12} "
                  f"{'N/A':<10} {'N/A':<10} (est. {est_mem:.1f} GB)")
    
    ok_count = sum(1 for r in all_results if r['status'] == 'OK')
    oom_count = sum(1 for r in all_results if r['status'] == 'OOM')
    print(f"\nTotal: {len(all_results)} configs | OK: {ok_count} | OOM: {oom_count}")


if __name__ == '__main__':
    main()