import matplotlib.pyplot as plt

# Model config
num_layers = 32
num_kv_heads = 8 
head_dim = 128 

def kv_cache_gb(seq_len, batch_size):
    size_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * 2
    return size_bytes / (1024 ** 3)

def create_plot(seq_lengths):
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for ax, seq_len in zip(axes, seq_lengths):
        sizes = [kv_cache_gb(seq_len, bs) for bs in batch_sizes]
        ax.bar(range(len(batch_sizes)), sizes, color='steelblue')
        ax.axhline(y=64, color='red', linewidth=2.5, linestyle='-', label='A100 80GB')
        ax.set_xticks(range(len(batch_sizes)))
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel('Batch Size')
        ax.set_title(f'seq_len = {seq_len}')
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('KV Cache Size (GB)')
    fig.suptitle('Llama 3.1 8B KV Cache Size (FP16)', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    seq_lengths = [[128, 1024, 2048, 4096], [16384, 32768, 65536, 131072]]
    for seq_len_set in seq_lengths:
        create_plot(seq_len_set)