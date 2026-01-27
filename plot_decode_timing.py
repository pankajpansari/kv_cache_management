#!/usr/bin/env python3
"""
Visualize decode timing benchmark results.
Run locally after scp'ing the CSV from runpod.

Usage: python plot_decode_timing.py decode_timing_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def plot_percentage_breakdown(df: pd.DataFrame, seq_len: int, output_file: str = None):
    """
    Create a percentage breakdown chart, marking OOM results.
    """
    data = df[df['seq_len'] == seq_len].copy()
    data = data.sort_values('batch_size')
    
    batch_sizes = data['batch_size'].values
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(batch_sizes))
    width = 0.6
    labels_added = set()

    for i, (_, row) in enumerate(data.iterrows()):
        if row['status'] == 'OK':
            # Calculate percentages if not directly in the column, 
            # or use row['linear_pct'] etc. if they exist.
            linear = row.get('linear_pct', (row['linear_total_ms'] / row['total_ms'] * 100))
            attn = row.get('attention_pct', (row['attention_ms_mean'] / row['total_ms'] * 100))
            other = 100 - linear - attn

            ax.bar(x[i], linear, width, color='#4c72b0', edgecolor='black', 
                   linewidth=0.5, hatch='\\\\', label='linear' if 'linear' not in labels_added else "")
            ax.bar(x[i], attn, width, bottom=linear, color='#dfdfe3', 
                   edgecolor='black', linewidth=0.5, label='attention' if 'attn' not in labels_added else "")
            ax.bar(x[i], other, width, bottom=linear + attn, color='#55a868', 
                   edgecolor='black', linewidth=0.5, hatch='...', label='others' if 'others' not in labels_added else "")
            
            labels_added.update(['linear', 'attn', 'others'])
        else:
            # Marker for OOM
            ax.text(x[i], 40, 'OOM', color='red', fontweight='bold', ha='center', va='top')

    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Decode Time Percentage Breakdown - Llama 3.1 8B (1 layer)\nSeq Length = {seq_len}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_seqlen(df: pd.DataFrame, output_file: str = None):
    """
    Create subplots for different sequence lengths, marking OOM results with a red cross.
    """
    seq_lengths = sorted(df['seq_len'].unique())[4:8]
    n_plots = len(seq_lengths)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=False)
    if n_plots == 1:
        axes = [axes]
    
    for ax, seq_len in zip(axes, seq_lengths):
        data = df[df['seq_len'] == seq_len].copy()
        data = data.sort_values('batch_size')
        
        batch_sizes = data['batch_size'].values
        x = np.arange(len(batch_sizes))
        width = 0.6
        labels_added = set()

        for i, (_, row) in enumerate(data.iterrows()):
            if row['status'] == 'OK':
                linear_time = row['linear_total_ms']
                attn_time = row['attention_ms_mean']
                other_time = row['other_ms_mean']
                
                ax.bar(x[i], linear_time, width, color='#4c72b0', edgecolor='black', 
                       linewidth=0.5, label='linear' if 'linear' not in labels_added else "")
                ax.bar(x[i], attn_time, width, bottom=linear_time, color='#dfdfe3', 
                       edgecolor='black', linewidth=0.5, label='attention' if 'attn' not in labels_added else "")
                ax.bar(x[i], other_time, width, bottom=linear_time + attn_time, color='#55a868', 
                       edgecolor='black', linewidth=0.5, label='others' if 'others' not in labels_added else "")
                
                labels_added.update(['linear', 'attn', 'others'])
            else:
                # Visualize OOM configuration
                ax.scatter(x[i], 0.5, marker='x', color='red', s=80, zorder=3)
        
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_title(f'Seq Len = {seq_len}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes, rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('Time (ms)', fontsize=11)
    if any(labels_added):
        axes[-1].legend(loc='upper left')
    
    fig.suptitle('Decode Time Breakdown - Llama 3.1 8B (1 layer)', fontsize=14, y=1.05)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

def print_summary(df: pd.DataFrame):
    """Print a text summary of results."""
    print("\n" + "=" * 75)
    print("SUMMARY: Decode Time Breakdown")
    print("=" * 75)
    
    for seq_len in sorted(df['seq_len'].unique()):
        print(f"\nSequence Length: {seq_len}")
        print("-" * 60)
        data = df[df['seq_len'] == seq_len].sort_values('batch_size')
        print(f"{'Batch':<8} {'Status':<8} {'Total(ms)':<12} {'Linear%':<10} {'Attn%':<10}")
        for _, row in data.iterrows():
            if row['status'] == 'OK':
                print(f"{row['batch_size']:<8} {row['status']:<8} {row['total_ms']:<12.3f} "
                      f"{row['linear_pct']:<10.1f} {row['attention_pct']:<10.1f}")
            else:
                print(f"{row['batch_size']:<8} {row['status']:<8} {'N/A':<12} {'N/A':<10} {'N/A':<10}")

def main():
    parser = argparse.ArgumentParser(description='Visualize decode timing results')
    parser.add_argument('csv_file', type=str, help='Path to CSV results file')
    parser.add_argument('--output', type=str, default=None, help='Output file for plot')
    
    args = parser.parse_args()
    with open(args.csv_file, 'r') as f:
        lines = [line.rstrip().rstrip(',') for line in f]
    df = pd.read_csv(args.csv_file)
    
    plot_multi_seqlen(df, args.output)
    print_summary(df)

if __name__ == '__main__':
    main()
