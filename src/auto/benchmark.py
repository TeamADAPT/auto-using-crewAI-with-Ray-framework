import time
import ray
import pandas as pd
from typing import List, Dict, Tuple
from crewai import Agent, Task, Crew
from .ray_executor import RayExecutor

class BenchmarkMetrics:
    def __init__(self):
        self.metrics = {
            'execution_time': [],
            'tasks_completed': [],
            'execution_type': [],
            'timestamp': [],
            'memory_usage': [],
            'cpu_usage': []
        }
    
    def add_metric(self, exec_time: float, tasks: int, exec_type: str,
                  memory: float, cpu: float):
        self.metrics['execution_time'].append(exec_time)
        self.metrics['tasks_completed'].append(tasks)
        self.metrics['execution_type'].append(exec_type)
        self.metrics['timestamp'].append(time.time())
        self.metrics['memory_usage'].append(memory)
        self.metrics['cpu_usage'].append(cpu)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

class Benchmark:
    def __init__(self):
        self.metrics = BenchmarkMetrics()
        self.ray_executor = RayExecutor()
    
    def run_sequential(self, agents: List[Dict], tasks: List[Task]) -> Tuple[List[str], float]:
        """Run tasks sequentially and measure performance"""
        start_time = time.time()
        results = []
        
        crew_agents = []
        for agent_config in agents:
            agent = Agent(**agent_config)
            crew_agents.append(agent)
            
        crew = Crew(
            agents=crew_agents,
            tasks=tasks,
            verbose=True
        )
        
        results = crew.kickoff()
        exec_time = time.time() - start_time
        return results, exec_time
    
    def run_distributed(self, agents: List[Dict], tasks: List[Task]) -> Tuple[List[str], float]:
        """Run tasks in parallel using Ray and measure performance"""
        start_time = time.time()
        ray_agents = self.ray_executor.create_agents(agents)
        results = self.ray_executor.execute_tasks(ray_agents, tasks)
        exec_time = time.time() - start_time
        return results, exec_time
    
    def compare_performance(self, agents: List[Dict], tasks: List[Task], iterations: int = 3):
        """Compare sequential vs distributed performance"""
        for _ in range(iterations):
            # Sequential execution
            try:
                _, seq_time = self.run_sequential(agents, tasks)
                self.metrics.add_metric(
                    seq_time, len(tasks), 'sequential',
                    ray.runtime_context.get_runtime_context().get_memory_usage(),
                    ray.runtime_context.get_runtime_context().get_cpu_usage()
                )
            except Exception as e:
                print(f"Error in sequential execution: {e}")
                continue
                
            # Distributed execution
            try:
                _, dist_time = self.run_distributed(agents, tasks)
                self.metrics.add_metric(
                    dist_time, len(tasks), 'distributed',
                    ray.runtime_context.get_runtime_context().get_memory_usage(),
                    ray.runtime_context.get_runtime_context().get_cpu_usage()
                )
            except Exception as e:
                print(f"Error in distributed execution: {e}")
                continue
        
        return self.metrics.to_dataframe()

def run():
    """CLI entry point for running benchmarks"""
    from .utils.metrics import MetricsCollector
    
    # Initialize benchmark and metrics collector
    benchmark = Benchmark()
    metrics_collector = MetricsCollector()
    
    # Define benchmark configurations
    agent_configs = [
        {
            "role": "Researcher",
            "goal": "Research and gather information",
            "backstory": "Expert at finding and collecting information",
            "allow_delegation": True,
            "verbose": True,
            "tools": []  # Add any specific tools your agents need
        },
        {
            "role": "Analyst",
            "goal": "Analyze gathered information",
            "backstory": "Skilled data analyst with attention to detail",
            "allow_delegation": True,
            "verbose": True,
            "tools": []
        },
        {
            "role": "Writer",
            "goal": "Create comprehensive reports",
            "backstory": "Experienced technical writer",
            "allow_delegation": True,
            "verbose": True,
            "tools": []
        }
    ]
    
    tasks = [
        Task(
            description="Gather market research data for tech sector",
            expected_output="Comprehensive market data report",
        ),
        Task(
            description="Analyze market trends and competition",
            expected_output="Detailed analysis report",
        ),
        Task(
            description="Create executive summary of findings",
            expected_output="Executive summary document",
        )
    ]
    
    # Run benchmarks with different configurations
    print("Running benchmarks...")
    
    # Test with different numbers of iterations
    iterations = [3, 5, 10]
    all_results = pd.DataFrame()
    
    for n_iter in iterations:
        print(f"\nRunning benchmark with {n_iter} iterations...")
        results_df = benchmark.compare_performance(agent_configs, tasks, iterations=n_iter)
        if not results_df.empty:
            results_df['num_iterations'] = n_iter
            all_results = pd.concat([all_results, results_df])
    
    if all_results.empty:
        print("No successful benchmark runs completed.")
        return
        
    # Calculate statistics and generate visualizations
    stats = metrics_collector.calculate_statistics(all_results)
    scaling_efficiency = metrics_collector.calculate_scaling_efficiency(all_results)
    
    # Save results
    save_paths = metrics_collector.save_results(all_results, stats)
    
    # Print summary
    print("\nBenchmark Results Summary:")
    print("-" * 50)
    print(f"Scaling Efficiency: {scaling_efficiency:.2f}")
    print("\nMean Execution Times:")
    print(f"Sequential: {stats['sequential']['mean_time']:.2f} seconds")
    print(f"Distributed: {stats['distributed']['mean_time']:.2f} seconds")
    print("\nResource Usage (average):")
    print(f"Sequential - Memory: {stats['sequential']['mean_memory']:.1f}%, CPU: {stats['sequential']['mean_cpu']:.1f}%")
    print(f"Distributed - Memory: {stats['distributed']['mean_memory']:.1f}%, CPU: {stats['distributed']['mean_cpu']:.1f}%")
    
    print("\nDetailed results saved to:")
    for key, path in save_paths.items():
        print(f"{key}: {path}")