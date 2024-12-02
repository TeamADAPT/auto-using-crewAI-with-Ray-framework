import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import json
import os

class MetricsCollector:
    """Collects and analyzes performance metrics"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical metrics for the results"""
        stats = {}
        for exec_type in ['sequential', 'distributed']:
            type_data = data[data['execution_type'] == exec_type]
            stats[exec_type] = {
                'mean_time': type_data['execution_time'].mean(),
                'median_time': type_data['execution_time'].median(),
                'std_dev': type_data['execution_time'].std(),
                'min_time': type_data['execution_time'].min(),
                'max_time': type_data['execution_time'].max(),
                'mean_memory': type_data['memory_usage'].mean(),
                'mean_cpu': type_data['cpu_usage'].mean()
            }
        return stats
    
    def plot_performance_comparison(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """Generate performance comparison plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Execution time comparison
        data.boxplot(column='execution_time', by='execution_type', ax=ax1)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        
        # Resource usage
        data.groupby('execution_type')[['memory_usage', 'cpu_usage']].mean().plot(
            kind='bar', ax=ax2
        )
        ax2.set_title('Resource Usage Comparison')
        ax2.set_ylabel('Usage (%)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_results(self, data: pd.DataFrame, stats: Dict):
        """Save benchmark results and statistics"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        csv_path = os.path.join(self.output_dir, f'benchmark_results_{timestamp}.csv')
        data.to_csv(csv_path, index=False)
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, f'benchmark_stats_{timestamp}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Generate and save plots
        plot_path = os.path.join(self.output_dir, f'benchmark_plots_{timestamp}.png')
        self.plot_performance_comparison(data, plot_path)
        
        return {
            'csv_path': csv_path,
            'stats_path': stats_path,
            'plot_path': plot_path
        }
    
    def calculate_scaling_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate how well the system scales with distributed execution"""
        sequential_mean = data[data['execution_type'] == 'sequential']['execution_time'].mean()
        distributed_mean = data[data['execution_type'] == 'distributed']['execution_time'].mean()
        return (sequential_mean / distributed_mean) / data['tasks_completed'].iloc[0]