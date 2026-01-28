import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Model config
num_kv_heads = 8
head_dim = 128


def kv_cache_gb(seq_len, batch_size):
    size_bytes = 2 * num_kv_heads * head_dim * seq_len * batch_size * 2
    return size_bytes / (1024**3)


def print_comparison(seq_lengths, df):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("\n" + "=" * 80)
    print("KV Cache Size vs Transferable GB Comparison")
    print("=" * 80)

    for seq_len in seq_lengths:
        print(f"\n{'seq_len = ' + str(seq_len):^80}")
        print("-" * 80)
        print(
            f"{'Batch Size':>10} | {'KV Cache (GB)':>14} | {'Transferable (GB)':>17} | {'Ratio (T/KV)':>12}"
        )
        print("-" * 80)

        for bs in batch_sizes:
            kv_size = kv_cache_gb(seq_len, bs)
            row = df[(df["batch_size"] == bs) & (df["seq_len"] == seq_len)]
            if not row.empty:
                transfer_size = row["transferable_gb"].values[0]
                ratio = transfer_size / kv_size if kv_size > 0 else 0
                print(
                    f"{bs:>10} | {kv_size:>14.4f} | {transfer_size:>17.4f} | {ratio:>11.2%}"
                )
            else:
                print(f"{bs:>10} | {kv_size:>14.4f} | {'N/A':>17} | {'N/A':>12}")

        print("-" * 80)


def create_plot(seq_lengths, df, image_filename):
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
            row = df[(df["batch_size"] == bs) & (df["seq_len"] == seq_len)]
            if not row.empty:
                transfer_sizes.append(row["transferable_gb"].values[0])
            else:
                transfer_sizes.append(0)

        ratios = [0] * len(batch_sizes)
        for i in range(len(batch_sizes)):
            if transfer_sizes[i] > 0:
                ratios[i] = transfer_sizes[i] / kv_sizes[i]

        ratios = np.array(ratios)
        special_mask = ratios == 0
        normal_mask = ratios != 0

        ax.plot(x, ratios, color="gray", linestyle="--", alpha=0.5)

        ax.scatter(
            x[normal_mask], ratios[normal_mask], color="blue", marker="o", label="Valid"
        )
        ax.scatter(
            x[special_mask],
            ratios[special_mask],
            color="red",
            marker="x",
            s=100,
            label="OOM",
        )

        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel("Batch Size")
        ax.set_title(f"seq_len = {seq_len}")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("H2D Transfer Size / KV Cache Size")
    fig.suptitle("Llama 3.1 8B KV Cache Size vs Transferable (FP16)", fontsize=12)
    plt.tight_layout()
    plt.savefig(image_filename)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    seq_lengths_sets = [[128, 1024, 2048, 4096], [16384, 32768, 65536, 131072]]
    print_comparison(seq_lengths_sets[0], df)
    create_plot(seq_lengths_sets[0], df, "results/comparison_kv_h2d_1.png")
    print_comparison(seq_lengths_sets[1], df)
    create_plot(seq_lengths_sets[1], df, "results/comparison_kv_h2d_2.png")
