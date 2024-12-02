import time
import ray
import pandas as pd
import os
from typing import List, Dict, Tuple
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

class BenchmarkMetrics:
    def __init__(self):
        self.metrics = {
            'execution_time': [],
            'tasks_completed': [],
            'execution_type': [],
            'timestamp': [],
            'cpu_percent': []  # Simplified metrics
        }
    
    def add_metric(self, exec_time: float, tasks: int, exec_type: str):
        """Add metrics without trying to access unavailable memory data"""
        self.metrics['execution_time'].append(exec_time)
        self.metrics['tasks_completed'].append(tasks)
        self.metrics['execution_type'].append(exec_type)
        self.metrics['timestamp'].append(time.time())
        # Get available CPU metrics from Ray
        try:
            resources = ray.available_resources()
            cpu_usage = 1.0 - (resources.get('CPU', 0) / ray.cluster_resources()['CPU'])
            self.metrics['cpu_percent'].append(cpu_usage * 100)
        except Exception:
            self.metrics['cpu_percent'].append(0.0)
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics)

@ray.remote
class RayAgent:
    """Ray Actor wrapper for CrewAI agents"""
    def __init__(self, agent_config: Dict):
        # Configure OpenAI for distributed agent
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            request_timeout=30,
            max_retries=3
        )
        agent_config['llm'] = self.llm
        self.agent = Agent(**agent_config)
        
    def execute_task(self, task: Task) -> str:
        return self.agent.execute(task)

class RayExecutor:
    """Handles distributed execution of CrewAI workflows using Ray"""
    
    def __init__(self):
        if not ray.is_initialized():
            ray.init()
            
    def create_agents(self, agent_configs: List[Dict]) -> List[RayAgent]:
        """Create Ray agent actors from configurations"""
        return [RayAgent.remote(config) for config in agent_configs]
    
    def execute_tasks(self, agents: List[RayAgent], tasks: List[Task]) -> List[str]:
        """Execute tasks in parallel using Ray"""
        futures = []
        for agent, task in zip(agents, tasks):
            futures.append(agent.execute_task.remote(task))
        return ray.get(futures)
    
    def shutdown(self):
        """Clean up Ray resources"""
        if ray.is_initialized():
            ray.shutdown()

class Benchmark:
    def __init__(self):
        self.metrics = BenchmarkMetrics()
        self.ray_executor = RayExecutor()
        
        # Configure OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            request_timeout=30,
            max_retries=3
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
        """Run tasks in parallel using Ray"""
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
                    seq_time, 
                    len(agent_configs), 
                    'sequential'
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
                    dist_time, 
                    len(agent_configs), 
                    'distributed'
                )
                print(f"Distributed execution completed in {dist_time:.2f} seconds")
            except Exception as e:
                print(f"Error in distributed execution: {str(e)}")
                continue
        
        return self.metrics.to_dataframe()

def run():
    """CLI entry point for running benchmarks"""
    # Initialize benchmark and metrics collector
    benchmark = Benchmark()
    
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
        
    # Calculate and print summary statistics
    sequential_times = all_results[all_results['execution_type'] == 'sequential']['execution_time']
    distributed_times = all_results[all_results['execution_type'] == 'distributed']['execution_time']
    
    print("\nBenchmark Results Summary:")
    print("-" * 50)
    print("\nExecution Times (seconds):")
    print(f"Sequential  - Mean: {sequential_times.mean():.2f}, Min: {sequential_times.min():.2f}, Max: {sequential_times.max():.2f}")
    print(f"Distributed - Mean: {distributed_times.mean():.2f}, Min: {distributed_times.min():.2f}, Max: {distributed_times.max():.2f}")
    
    speedup = sequential_times.mean() / distributed_times.mean()
    print(f"\nAverage Speedup: {speedup:.2f}x")
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"benchmark_results_{timestamp}.csv"
    all_results.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    run()