#!/usr/bin/env python3
"""
Linear layer execution time vs number of tokens for Llama 3.1 8B on A100.
Reproduces Sarathi-Serve Figure 6 style experiment (single GPU, no TP).
"""

import torch
import torch.nn as nn
import csv

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
    layer = LinearLayers().to(device=device, dtype=dtype)
    layer.eval()
    
    # === GLOBAL WARMUP: cycle through ALL configs first ===
    print("Global warmup pass...")
    with torch.no_grad():
        for _ in range(3):  # 3 full passes
            for num_tokens in num_tokens_list:
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
                for _ in range(5):
                    _ = layer(x)
    torch.cuda.synchronize()
    print("Global warmup done.\n")
    
    results = []
    
    for num_tokens in num_tokens_list:
        x = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
        
        # Per-config warmup (generous)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = layer(x)
        torch.cuda.synchronize()
        
        # Benchmark with fresh events each run
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = layer(x)
                end.record()
                
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        
        # Remove outliers (first few might still be unstable)
        times = sorted(times)[5:-5]  # trim 5 from each end
        
        layer_time_ms = sum(times) / len(times)
        std_ms = torch.tensor(times).std().item()
        total_time_ms = layer_time_ms * num_layers
        
        results.append({
            'num_tokens': num_tokens,
            'per_layer_ms': layer_time_ms,
            'std_ms': std_ms,
            'total_ms': total_time_ms,
        })
        print(f"tokens={num_tokens:>4}: {total_time_ms:.2f}ms (±{std_ms*num_layers:.2f}ms)")
    
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
    

if __name__ == '__main__':
    main()