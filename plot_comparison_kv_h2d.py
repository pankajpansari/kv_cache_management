import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Model config
num_kv_heads = 8
head_dim = 128

def kv_cache_gb(seq_len, batch_size):
    size_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * 2
    return size_bytes / (1024 ** 3)

def print_comparison(seq_lengths, df):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print("\n" + "=" * 80)
    print("KV Cache Size vs Transferable GB Comparison")
    print("=" * 80)
    
    for seq_len in seq_lengths:
        print(f"\n{'seq_len = ' + str(seq_len):^80}")
        print("-" * 80)
        print(f"{'Batch Size':>10} | {'KV Cache (GB)':>14} | {'Transferable (GB)':>17} | {'Ratio (T/KV)':>12}")
        print("-" * 80)
        
        for bs in batch_sizes:
            kv_size = kv_cache_gb(seq_len, bs)
            row = df[(df['batch_size'] == bs) & (df['seq_len'] == seq_len)]
            if not row.empty:
                transfer_size = row['transferable_gb'].values[0]
                ratio = transfer_size / kv_size if kv_size > 0 else 0
                print(f"{bs:>10} | {kv_size:>14.4f} | {transfer_size:>17.4f} | {ratio:>11.2%}")
            else:
                print(f"{bs:>10} | {kv_size:>14.4f} | {'N/A':>17} | {'N/A':>12}")
        
        print("-" * 80)

def create_plot(seq_lengths, df):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    
    bar_width = 0.35
    x = np.arange(len(batch_sizes))
    
    for ax, seq_len in zip(axes, seq_lengths):
        kv_sizes = [kv_cache_gb(seq_len, bs) for bs in batch_sizes]
        
        # Get transferable_gb from CSV for each batch_size at this seq_len
        transfer_sizes = []
        for bs in batch_sizes:
            row = df[(df['batch_size'] == bs) & (df['seq_len'] == seq_len)]
            if not row.empty:
                transfer_sizes.append(row['transferable_gb'].values[0])
            else:
                transfer_sizes.append(0)
        
        ax.bar(x - bar_width/2, kv_sizes, bar_width, color='steelblue', label='KV Cache')
        ax.bar(x + bar_width/2, transfer_sizes, bar_width, color='orange', label='Transferable')
        ax.axhline(y=64, color='red', linewidth=2.5, linestyle='-', label='A100 80GB')
        
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel('Batch Size')
        ax.set_title(f'seq_len = {seq_len}')
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('Memory (GB)')
    axes[0].legend()
    fig.suptitle('Llama 3.1 8B KV Cache Size vs Transferable (FP16)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    df = pd.read_csv(sys.argv[1])
    
    seq_lengths_sets = [[128, 1024, 2048, 4096], [16384, 32768, 65536, 131072]]
    for seq_len_set in seq_lengths_sets:
        print_comparison(seq_len_set, df)
        create_plot(seq_len_set, df)