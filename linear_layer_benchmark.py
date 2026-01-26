#!/usr/bin/env python3
"""
Linear layer execution time vs number of tokens for Llama 3.1 8B on A100.
Reproduces Sarathi-Serve Figure 6 style experiment (single GPU, no TP).
"""

import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt

# Llama 3.1 8B config
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128

class LinearLayers(nn.Module):
    """All linear layers in one Llama decoder layer."""
    def __init__(self):
        super().__init__()
        # Attention projections
        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        # MLP (SwiGLU)
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
    
    def forward(self, x):
        # Attention linears
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)  # Using q as proxy for attn output shape
        # MLP linears
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return down

def benchmark_linear(num_tokens_list, num_layers=32, num_warmup=10, num_runs=100, dtype=torch.float16):
    """Benchmark linear layer time for full model (all layers)."""
    device = 'cuda'
    
    # Create one layer, we'll multiply time by num_layers
    layer = LinearLayers().to(device=device, dtype=dtype)
    layer.eval()
    
    results = []
    
    for num_tokens in num_tokens_list:
        x = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start.record()
                _ = layer(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        layer_time_ms = sum(times) / len(times)
        total_time_ms = layer_time_ms * num_layers  # Scale to full model
        
        results.append({
            'num_tokens': num_tokens,
            'per_layer_ms': layer_time_ms,
            'total_ms': total_time_ms,
        })
        print(f"tokens={num_tokens:>4}: per_layer={layer_time_ms:.3f}ms, total={total_time_ms:.2f}ms")
    
    return results

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Llama 3.1 8B: hidden={HIDDEN_SIZE}, intermediate={INTERMEDIATE_SIZE}")
    print(f"Layers: 32, No tensor parallelism")
    print("-" * 50)
    
    # Token counts similar to figure (64 to 4096)
    num_tokens_list = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    
    results = benchmark_linear(num_tokens_list)
    
    # Save CSV
    with open('linear_layer_timing.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved to linear_layer_timing.csv")
    
    # Plot
    tokens = [r['num_tokens'] for r in results]
    times = [r['total_ms'] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(tokens, times, 'b-o', linewidth=2, markersize=6, label='TP-1 (no parallelism)')
    plt.xlabel('Number of tokens')
    plt.ylabel('Time (ms)')
    plt.title('Linear Layer Execution Time - Llama 3.1 8B on A100')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('linear_layer_timing.png', dpi=150)
    plt.show()
    
    print("\nSaved plot to linear_layer_timing.png")

if __name__ == '__main__':
    main()