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
    Create a percentage breakdown chart.
    """
    data = df[df['seq_len'] == seq_len].copy()
    data = data.sort_values('batch_size')
    
    batch_sizes = data['batch_size'].values
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(batch_sizes))
    width = 0.6

    # p1: A deep professional blue with diagonal stripes
    p1 = ax.bar(x, linear_times, width, label='linear', 
                color='#4c72b0', edgecolor='black', linewidth=0.5, hatch='\\\\')

    # p2: A neutral, solid light gray (serves as a visual break)
    p2 = ax.bar(x, attn_times, width, bottom=linear_times, label='attention',
                color='#dfdfe3', edgecolor='black', linewidth=0.5)

    # p3: A vibrant sage green with a dotted texture
    p3 = ax.bar(x, other_times, width, bottom=linear_times + attn_times, label='others',
                color='#55a868', edgecolor='black', linewidth=0.5, hatch='...')    
    
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
        print(f"Saved plot to {output_file}")
    
    plt.show()


def plot_multi_seqlen(df: pd.DataFrame, output_file: str = None):
    """
    Create subplots for different sequence lengths.
    """
    seq_lengths = sorted(df['seq_len'].unique())
    n_plots = len(seq_lengths)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]
    
    for ax, seq_len in zip(axes, seq_lengths):
        data = df[df['seq_len'] == seq_len].copy()
        data = data.sort_values('batch_size')
        
        batch_sizes = data['batch_size'].values
        linear_times = data['linear_total_ms'].values
        attn_times = data['attention_ms_mean'].values
        other_times = data['other_ms_mean'].values
        
        x = np.arange(len(batch_sizes))
        width = 0.6

        # p1: A deep professional blue with diagonal stripes
        p1 = ax.bar(x, linear_times, width, label='linear', 
                    color='#4c72b0', edgecolor='black', linewidth=0.5, hatch='\\\\')

        # p2: A neutral, solid light gray (serves as a visual break)
        p2 = ax.bar(x, attn_times, width, bottom=linear_times, label='attention',
                    color='#dfdfe3', edgecolor='black', linewidth=0.5)

        # p3: A vibrant sage green with a dotted texture
        p3 = ax.bar(x, other_times, width, bottom=linear_times + attn_times, label='others',
                    color='#55a868', edgecolor='black', linewidth=0.5, hatch='...') 
        
        ax.set_xlabel('Batch Size', fontsize=11)
        ax.set_title(f'Seq Len = {seq_len}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes, rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('Time (ms)', fontsize=11)
    axes[-1].legend(loc='upper left')
    
    fig.suptitle('Decode Time Breakdown - Llama 3.1 8B (1 layer)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print a text summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: Decode Time Breakdown")
    print("=" * 70)
    
    for seq_len in sorted(df['seq_len'].unique()):
        print(f"\nSequence Length: {seq_len}")
        print("-" * 50)
        data = df[df['seq_len'] == seq_len].sort_values('batch_size')
        print(f"{'Batch':<8} {'Total(ms)':<12} {'Linear%':<10} {'Attn%':<10}")
        for _, row in data.iterrows():
            print(f"{row['batch_size']:<8} {row['total_ms']:<12.3f} "
                  f"{row['linear_pct']:<10.1f} {row['attention_pct']:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize decode timing results')
    parser.add_argument('csv_file', type=str, help='Path to CSV results file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for plot (png, pdf, etc.)')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} rows from {args.csv_file}")
    
    # Plot all sequence lengths
    plot_multi_seqlen(df, args.output)
    
    print_summary(df)


if __name__ == '__main__':
    main()