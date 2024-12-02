import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_benchmark_results(csv_file):
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return None
        
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set figure style
    plt.style.use('default')
    
    # Create figure and axis with larger size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create boxplot
    bp = df[df['num_iterations'].notna()].boxplot(
        column='execution_time', 
        by='num_iterations',
        ax=ax,
        medianprops=dict(color="red", linewidth=1.5),
        meanprops=dict(color="green", linewidth=1.5),
        showmeans=True
    )
    
    # Customize plot
    ax.set_title('Benchmark Results: Sequential Execution Time Distribution', 
                fontsize=14, pad=20)
    ax.set_xlabel('Number of Iterations', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Calculate and add summary statistics
    stats = df.groupby('num_iterations')['execution_time'].agg(
        ['mean', 'min', 'max', 'std']
    ).round(2)
    
    # Format statistics text
    stats_text = "Summary Statistics:\n"
    for idx, row in stats.iterrows():
        stats_text += f"\n{idx:.0f} iterations:\n"
        stats_text += f"Mean: {row['mean']}s\n"
        stats_text += f"Min: {row['min']}s\n"
        stats_text += f"Max: {row['max']}s\n"
        stats_text += f"Std: {row['std']}s\n"
    
    # Add statistics text to plot
    plt.figtext(1.02, 0.5, stats_text, fontsize=10, va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('benchmark_results_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'benchmark_results_plot.png'")
    
    return stats

if __name__ == "__main__":
    csv_file = '/Users/fei/Documents/UCSC/24fall/CSE293/final project/auto/benchmark_results_20241202-062331.csv'
    try:
        stats = plot_benchmark_results(csv_file)
        if stats is not None:
            print("\nStatistical Summary:")
            print(stats)
    except Exception as e:
        print(f"Error: {str(e)}")