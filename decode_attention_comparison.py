#!/usr/bin/env python3
"""
Decode timing benchmark comparing attention backends:
  - FlashInfer (best for decode - dedicated decode kernels, native GQA)
  - FlashAttention (better for prefill, but works for decode)
  - Manual PyTorch (baseline reference)

Key insights for decode attention:
1. FlashInfer is preferred because:
   - Has BatchDecodeWithPagedKVCache optimized for single-query attention
   - Native GQA support without memory-wasting repeat_interleave
   - Designed for serving with variable-length batches
   
2. FlashAttention is optimized for prefill (long Q sequences) but
   can be used for decode via flash_attn_with_kvcache

3. Manual attention requires GQA expansion which wastes memory
"""

import argparse
import csv
import gc
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

# Backend availability
FLASHINFER_AVAILABLE = False
FLASH_ATTN_AVAILABLE = False

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass


class AttentionBackend(Enum):
    MANUAL = "manual"
    FLASHINFER = "flashinfer"
    FLASH_ATTN = "flash_attn"


class RMSNorm(nn.Module):
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
    q: torch.Tensor,      # [batch, 1, num_heads, head_dim]
    k_cache: torch.Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """Manual attention with GQA expansion."""
    batch_size = q.shape[0]
    
    # Transpose: [batch, heads, seq, dim]
    q = q.transpose(1, 2)
    k = k_cache.transpose(1, 2)
    v = v_cache.transpose(1, 2)
    
    # Expand KV for GQA (this is the inefficient part)
    k = k.repeat_interleave(config.num_kv_groups, dim=1)
    v = v.repeat_interleave(config.num_kv_groups, dim=1)
    
    scale = 1.0 / (config.head_dim ** 0.5)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.matmul(attn_weights, v)
    
    return output.transpose(1, 2)


def flashinfer_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """FlashInfer decode attention - native GQA, no expansion needed."""
    # q: [batch, 1, num_heads, head_dim]
    # k/v_cache: [batch, seq_len, num_kv_heads, head_dim]
    
    q_squeezed = q.squeeze(1)  # [batch, num_heads, head_dim]
    
    # FlashInfer handles GQA internally - much more efficient!
    output = flashinfer.batch_decode_with_padded_kv_cache(
        q_squeezed,
        k_cache,
        v_cache,
        kv_layout="NHD",
    )
    
    return output.unsqueeze(1)


def flash_attn_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    config: AttentionConfig,
) -> torch.Tensor:
    """FlashAttention decode using kv cache API."""
    # flash_attn_with_kvcache expects:
    # q: [batch, seqlen_q, num_heads, head_dim]
    # k_cache, v_cache: [batch, seqlen_k, num_kv_heads, head_dim]
    
    output = flash_attn_with_kvcache(
        q=q,  # [batch, 1, num_heads, head_dim]
        k_cache=k_cache,
        v_cache=v_cache,
        softmax_scale=1.0 / (config.head_dim ** 0.5),
    )
    
    return output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, backend: AttentionBackend = AttentionBackend.FLASHINFER):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # MLP
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
        if self.backend == AttentionBackend.FLASHINFER and not FLASHINFER_AVAILABLE:
            print("Warning: FlashInfer requested but not available, falling back to manual")
            self.backend = AttentionBackend.MANUAL
        elif self.backend == AttentionBackend.FLASH_ATTN and not FLASH_ATTN_AVAILABLE:
            print("Warning: FlashAttention requested but not available, falling back to manual")
            self.backend = AttentionBackend.MANUAL
    
    def _attention(self, q, k_cache, v_cache):
        if self.backend == AttentionBackend.FLASHINFER:
            return flashinfer_decode_attention(q, k_cache, v_cache, self.attn_config)
        elif self.backend == AttentionBackend.FLASH_ATTN:
            return flash_attn_decode_attention(q, k_cache, v_cache, self.attn_config)
        else:
            return manual_decode_attention(q, k_cache, v_cache, self.attn_config)
    
    def forward_timed(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> dict:
        """Decode forward with component timing."""
        batch_size = hidden_states.shape[0]
        
        events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for name in ['norm1', 'qkv', 'attn', 'o_proj', 'norm2', 'mlp']}
        
        # Input LayerNorm
        events['norm1'][0].record()
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        events['norm1'][1].record()
        
        # Q, K, V Projections
        events['qkv'][0].record()
        q = self.q_proj(hidden_states)
        _ = self.k_proj(hidden_states)  # Would be appended to cache
        _ = self.v_proj(hidden_states)
        events['qkv'][1].record()
        
        # Reshape Q: [batch, 1, num_heads, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        
        # Attention
        events['attn'][0].record()
        attn_output = self._attention(q, k_cache, v_cache)
        events['attn'][1].record()
        
        attn_output = attn_output.reshape(batch_size, 1, -1)
        
        # O Projection
        events['o_proj'][0].record()
        attn_output = self.o_proj(attn_output)
        events['o_proj'][1].record()
        
        hidden_states = residual + attn_output
        
        # Post-attention LayerNorm
        events['norm2'][0].record()
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        events['norm2'][1].record()
        
        # MLP
        events['mlp'][0].record()
        mlp_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        events['mlp'][1].record()
        
        hidden_states = residual + mlp_output
        
        torch.cuda.synchronize()
        timings = {name: events[name][0].elapsed_time(events[name][1]) for name in events}
        
        return {
            'attn_linear_ms': timings['qkv'] + timings['o_proj'],
            'attention_ms': timings['attn'],
            'mlp_linear_ms': timings['mlp'],
            'other_ms': timings['norm1'] + timings['norm2'],
        }


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def estimate_memory_gb(batch_size: int, seq_len: int, config, dtype: torch.dtype, 
                       backend: AttentionBackend) -> float:
    """Estimate memory - note FlashInfer doesn't need GQA expansion."""
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Base KV cache
    kv_cache_bytes = 2 * batch_size * config.num_key_value_heads * seq_len * head_dim * bytes_per_element
    
    # Only manual attention needs GQA expansion
    if backend == AttentionBackend.MANUAL:
        expanded_kv_bytes = 2 * batch_size * config.num_attention_heads * seq_len * head_dim * bytes_per_element
        attn_scores_bytes = batch_size * config.num_attention_heads * seq_len * bytes_per_element
        kv_cache_bytes += expanded_kv_bytes + attn_scores_bytes
    
    return kv_cache_bytes / (1024 ** 3)


def run_single_config(layer, config, batch_size, seq_len, num_warmup, num_runs, dtype, device):
    head_dim = config.hidden_size // config.num_attention_heads
    
    result = {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'status': 'OK',
        'backend': layer.backend.value,
    }
    
    try:
        hidden_states = torch.randn(batch_size, 1, config.hidden_size, device=device, dtype=dtype)
        k_cache = torch.randn(batch_size, seq_len, config.num_key_value_heads, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn(batch_size, seq_len, config.num_key_value_heads, head_dim, device=device, dtype=dtype)
        
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
        
        for k, v in all_timings.items():
            result[f'{k}_mean'] = sum(v) / len(v)
            result[f'{k}_std'] = torch.tensor(v).std().item()
        
        result['linear_total_ms'] = result['attn_linear_ms_mean'] + result['mlp_linear_ms_mean']
        result['total_ms'] = sum(result[f'{k}_mean'] for k in ['attn_linear_ms', 'attention_ms', 'mlp_linear_ms', 'other_ms'])
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
        print(f"  OOM at batch={batch_size}, seq={seq_len} (est. ~{result['estimated_mem_gb']:.1f} GB)")
    
    return result


def run_benchmark(model_name, batch_sizes, seq_lengths, num_warmup, num_runs, dtype, device, backend):
    print(f"Loading config: {model_name}")
    config = AutoConfig.from_pretrained(model_name, token=True)
    
    print(f"Architecture: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    layer = LlamaDecoderLayer(config, backend=backend).to(device=device, dtype=dtype)
    layer.eval()
    
    print(f"Backend: {layer.backend.value}")
    
    results = []
    for seq_len in sorted(seq_lengths):
        for batch_size in batch_sizes:
            print(f"Running: batch={batch_size}, seq={seq_len}")
            result = run_single_config(layer, config, batch_size, seq_len, num_warmup, num_runs, dtype, device)
            results.append(result)
            
            if result['status'] == 'OK':
                print(f"  Total: {result['total_ms']:.3f}ms | Linear: {result['linear_pct']:.1f}% | Attn: {result['attention_pct']:.1f}%")
    
    return results


def save_results(results, filename):
    if not results:
        return
    
    ordered_fields = [
        'batch_size', 'seq_len', 'status', 'backend',
        'total_ms', 'linear_total_ms', 'linear_pct', 'attention_pct', 'other_pct',
        'attention_ms_mean', 'attention_ms_std',
    ]
    
    all_fields = set()
    for r in results:
        all_fields.update(r.keys())
    
    fieldnames = [f for f in ordered_fields if f in all_fields]
    fieldnames.extend([f for f in sorted(all_fields) if f not in fieldnames])
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Decode benchmark comparing attention backends')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[1024, 4096, 16384, 65536])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--output', type=str, default='decode_timing_comparison.csv')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--backend', type=str, default='flashinfer',
                        choices=['manual', 'flashinfer', 'flash_attn', 'all'],
                        help='Attention backend (or "all" to compare)')
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"FlashInfer available: {FLASHINFER_AVAILABLE}")
    print(f"FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    print("-" * 60)
    
    if args.backend == 'all':
        backends = [AttentionBackend.MANUAL]
        if FLASHINFER_AVAILABLE:
            backends.append(AttentionBackend.FLASHINFER)
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
            device='cuda',
            backend=backend
        )
        all_results.extend(results)
    
    save_results(all_results, args.output)
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Backend':<12} {'Batch':<8} {'Seq':<10} {'Total(ms)':<12} {'Attn(ms)':<12} {'Attn%':<10}")
    print("-" * 100)
    for r in all_results:
        if r['status'] == 'OK':
            print(f"{r['backend']:<12} {r['batch_size']:<8} {r['seq_len']:<10} "
                  f"{r['total_ms']:<12.3f} {r['attention_ms_mean']:<12.3f} {r['attention_pct']:<10.1f}")


if __name__ == '__main__':
    main()