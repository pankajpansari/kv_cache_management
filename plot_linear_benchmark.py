import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv('linear_layer_timing.csv', header = 0)
# Plot
tokens = results['num_tokens']
times = results['total_ms']

plt.figure(figsize=(8, 5))
plt.plot(tokens, times, 'b-o', linewidth=2, markersize=6)
plt.xscale('log', base=2)
plt.xticks(tokens, tokens)
plt.xlabel('Number of tokens')
plt.ylabel('Time (ms)')
plt.title('Linear Layer Execution Time - Llama 3.1 8B on A100')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('linear_layer_timing.png', dpi=150)
plt.show()

print("\nSaved plot to linear_layer_timing.png")
