#!/usr/bin/env python3
"""
Linear layer timing with fresh weights per config to avoid L2 caching.
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

def create_layer(device, dtype):
    """Create fresh layer with new random weights."""
    layer = nn.ModuleList([
        nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False),      # q
        nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False),   # k
        nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False),   # v
        nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False),      # o
        nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False),         # gate
        nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False),         # up
        nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False),         # down
    ]).to(device=device, dtype=dtype)
    layer.eval()
    return layer

def forward_layer(layer, x):
    q = layer[0](x)
    k = layer[1](x)
    v = layer[2](x)
    o = layer[3](q)
    gate = layer[4](x)
    up = layer[5](x)
    down = layer[6](gate * up)
    return down

def benchmark_linear(num_tokens_list, num_layers=32, num_warmup=10, num_runs=50, dtype=torch.float16):
    device = 'cuda'
    results = []
    
    for num_tokens in num_tokens_list:
        times = []
        
        for run in range(num_warmup + num_runs):
            # Fresh layer each run - no L2 cache reuse
            layer = create_layer(device, dtype)
            x = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
            
            torch.cuda.synchronize()
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                start.record()
                _ = forward_layer(layer, x)
                end.record()
            
            torch.cuda.synchronize()
            
            if run >= num_warmup:  # skip warmup runs
                times.append(start.elapsed_time(end))
            
            del layer  # free memory
            torch.cuda.empty_cache()
        
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
    num_tokens_list = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024]
    
    results = benchmark_linear(num_tokens_list)
    
    # Save CSV
    with open('linear_layer_timing.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved to linear_layer_timing.csv")
    

if __name__ == '__main__':
    main()