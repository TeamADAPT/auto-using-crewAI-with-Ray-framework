import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read and prepare data
df = pd.read_csv('../benchmark_results_20241202-222830.csv')
df = df[df['num_iterations'].notna()]  # Remove rows without iteration numbers

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot for Normal execution
sequential_data = df[df['execution_type'] == 'sequential']
sns.boxplot(data=sequential_data, x='num_iterations', y='execution_time', ax=ax1, color='lightblue')
ax1.set_title('Normal Execution Time Distribution', fontsize=12, pad=10)
ax1.set_xlabel('Number of Iterations', fontsize=10)
ax1.set_ylabel('Execution Time (seconds)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.3)

# Plot for Ray execution
distributed_data = df[df['execution_type'] == 'distributed']
sns.boxplot(data=distributed_data, x='num_iterations', y='execution_time', ax=ax2, color='lightgreen')
ax2.set_title('Ray Execution Time Distribution', fontsize=12, pad=10)
ax2.set_xlabel('Number of Iterations', fontsize=10)
ax2.set_ylabel('Execution Time (seconds)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.savefig('separate_execution_time_boxplots.png', dpi=300, bbox_inches='tight')

# Print statistics
stats = df.groupby(['execution_type', 'num_iterations'])['execution_time'].describe().round(2)
print("\nExecution Time Statistics:")
print(stats)