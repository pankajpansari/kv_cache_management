import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('linear_layer_timing.csv')
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
