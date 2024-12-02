import time
import ray
import pandas as pd
import os
from typing import List, Dict, Tuple
from crewai import Agent, Task, Crew
from .ray_executor import RayExecutor
from langchain_openai import ChatOpenAI

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
        
        # Configure default LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
        )
    
    def create_agents_and_tasks(self, agent_configs: List[Dict]) -> Tuple[List[Agent], List[Task]]:
        """Create agents and tasks with proper assignments"""
        # Add LLM configuration to each agent
        agents = []
        for config in agent_configs:
            config['llm'] = self.llm  # Add LLM configuration
            agents.append(Agent(**config))
        
        # Create tasks with assigned agents
        tasks = [
            Task(
                description="Gather market research data for tech sector",
                expected_output="Comprehensive market data report",
                agent=agents[0]  # Assign to Researcher
            ),
            Task(
                description="Analyze market trends and competition",
                expected_output="Detailed analysis report",
                agent=agents[1]  # Assign to Analyst
            ),
            Task(
                description="Create executive summary of findings",
                expected_output="Executive summary document",
                agent=agents[2]  # Assign to Writer
            )
        ]
        
        return agents, tasks
    
    def run_sequential(self, agent_configs: List[Dict]) -> Tuple[List[str], float]:
        """Run tasks sequentially and measure performance"""
        start_time = time.time()
        
        # Create agents and tasks
        agents, tasks = self.create_agents_and_tasks(agent_configs)
        
        # Create and run crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
        
        results = crew.kickoff()
        exec_time = time.time() - start_time
        return results, exec_time
    
    def run_distributed(self, agent_configs: List[Dict]) -> Tuple[List[str], float]:
        """Run tasks in parallel using Ray and measure performance"""
        start_time = time.time()
        ray_agents = self.ray_executor.create_agents(agent_configs)
        _, tasks = self.create_agents_and_tasks(agent_configs)
        results = self.ray_executor.execute_tasks(ray_agents, tasks)
        exec_time = time.time() - start_time
        return results, exec_time
    
    def compare_performance(self, agent_configs: List[Dict], iterations: int = 3):
        """Compare sequential vs distributed performance"""
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            
            # Sequential execution
            try:
                print("Running sequential execution...")
                _, seq_time = self.run_sequential(agent_configs)
                self.metrics.add_metric(
                    seq_time, len(agent_configs), 'sequential',
                    ray.runtime_context.get_runtime_context().get_memory_usage(),
                    ray.runtime_context.get_runtime_context().get_cpu_usage()
                )
                print(f"Sequential execution completed in {seq_time:.2f} seconds")
            except Exception as e:
                print(f"Error in sequential execution: {str(e)}")
                continue
                
            # Distributed execution
            try:
                print("Running distributed execution...")
                _, dist_time = self.run_distributed(agent_configs)
                self.metrics.add_metric(
                    dist_time, len(agent_configs), 'distributed',
                    ray.runtime_context.get_runtime_context().get_memory_usage(),
                    ray.runtime_context.get_runtime_context().get_cpu_usage()
                )
                print(f"Distributed execution completed in {dist_time:.2f} seconds")
            except Exception as e:
                print(f"Error in distributed execution: {str(e)}")
                continue
        
        return self.metrics.to_dataframe()

def run():
    """CLI entry point for running benchmarks"""
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key first:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
        
    from .utils.metrics import MetricsCollector
    
    # Initialize benchmark and metrics collector
    benchmark = Benchmark()
    metrics_collector = MetricsCollector()
    
    # Define benchmark configurations with more specific agent parameters
    agent_configs = [
        {
            "role": "Researcher",
            "goal": "Research and gather information",
            "backstory": "Expert at finding and collecting information",
            "allow_delegation": True,
            "verbose": True,
            "max_iterations": 1  # Limit iterations for benchmarking
        },
        {
            "role": "Analyst",
            "goal": "Analyze gathered information",
            "backstory": "Skilled data analyst with attention to detail",
            "allow_delegation": True,
            "verbose": True,
            "max_iterations": 1
        },
        {
            "role": "Writer",
            "goal": "Create comprehensive reports",
            "backstory": "Experienced technical writer",
            "allow_delegation": True,
            "verbose": True,
            "max_iterations": 1
        }
    ]
    
    # Run benchmarks with different configurations
    print("Running benchmarks...")
    
    # Start with a single iteration for testing
    print("\nRunning initial test with 1 iteration...")
    test_results = benchmark.compare_performance(agent_configs, iterations=1)
    
    if test_results.empty:
        print("Initial test failed. Please check the configuration and try again.")
        return
        
    # If initial test succeeds, run full benchmark
    iterations = [3, 5, 10]
    all_results = test_results
    
    for n_iter in iterations:
        print(f"\nRunning benchmark with {n_iter} iterations...")
        results_df = benchmark.compare_performance(agent_configs, iterations=n_iter)
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