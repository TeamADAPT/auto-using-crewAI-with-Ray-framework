import time
import ray
import pandas as pd
import os
from typing import List, Dict, Tuple
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    def __init__(self):
        self.metrics = {
            'execution_time': [],
            'tasks_completed': [],
            'execution_type': [],
            'timestamp': [],
            'cpu_percent': []
        }
    
    def add_metric(self, exec_time: float, tasks: int, exec_type: str):
        self.metrics['execution_time'].append(exec_time)
        self.metrics['tasks_completed'].append(tasks)
        self.metrics['execution_type'].append(exec_type)
        self.metrics['timestamp'].append(time.time())
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
        # Initialize OpenAI client inside the Ray actor
        self.llm = ChatOpenAI(
            model=os.getenv('MODEL', 'gpt-4'),
            temperature=0.7,
            request_timeout=30,
            max_retries=3
        )
        # Create agent with initialized LLM
        agent_config['llm'] = self.llm
        self.agent = Agent(**agent_config)
        
    def execute_task(self, task_data: Dict) -> str:
        """Execute task from serializable data"""
        task = Task(
            description=task_data['description'],
            expected_output=task_data['expected_output']
        )
        logger.info(f"Executing task: {task.description}")
        return self.agent.execute_task(task)

class RayExecutor:
    def __init__(self):
        if not ray.is_initialized():
            ray.init()
            
    def create_agents(self, agent_configs: List[Dict]) -> List[RayAgent]:
        return [RayAgent.remote(config) for config in agent_configs]
    
    def execute_tasks(self, agents: List[RayAgent], tasks: List[Task]) -> List[str]:
        futures = []
        for agent, task in zip(agents, tasks):
            # Convert task to serializable format
            task_data = {
                'description': task.description,
                'expected_output': task.expected_output
            }
            futures.append(agent.execute_task.remote(task_data))
        return ray.get(futures)
    
    def shutdown(self):
        if ray.is_initialized():
            ray.shutdown()

class Benchmark:
    def __init__(self):
        self.metrics = BenchmarkMetrics()
        self.ray_executor = RayExecutor()
        
        # Initialize OpenAI LLM for sequential execution
        self.llm = ChatOpenAI(
            model=os.getenv('MODEL', 'gpt-4'),
            temperature=0.7,
            request_timeout=30,
            max_retries=3
        )
    
    def create_agents_and_tasks(self, agent_configs: List[Dict]) -> Tuple[List[Agent], List[Task]]:
        agents = []
        for config in agent_configs:
            config_copy = config.copy()  # Create a copy to avoid modifying original
            config_copy['llm'] = self.llm
            agents.append(Agent(**config_copy))
        
        tasks = [
            Task(
                description="Gather market research data for tech sector",
                expected_output="Comprehensive market data report",
                agent=agents[0]
            ),
            Task(
                description="Analyze market trends and competition",
                expected_output="Detailed analysis report",
                agent=agents[1]
            ),
            Task(
                description="Create executive summary of findings",
                expected_output="Executive summary document",
                agent=agents[2]
            )
        ]
        
        return agents, tasks
    
    def run_sequential(self, agent_configs: List[Dict]) -> Tuple[List[str], float]:
        start_time = time.time()
        agents, tasks = self.create_agents_and_tasks(agent_configs)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True
        )
        
        results = crew.kickoff()
        exec_time = time.time() - start_time
        return results, exec_time
    
    def run_distributed(self, agent_configs: List[Dict]) -> Tuple[List[str], float]:
        start_time = time.time()
        
        # Create fresh agent configs without LLM
        clean_configs = [{k: v for k, v in config.items() if k != 'llm'} 
                        for config in agent_configs]
        
        ray_agents = self.ray_executor.create_agents(clean_configs)
        _, tasks = self.create_agents_and_tasks(agent_configs)
        results = self.ray_executor.execute_tasks(ray_agents, tasks)
        exec_time = time.time() - start_time
        return results, exec_time
    
    def compare_performance(self, agent_configs: List[Dict], iterations: int = 3):
        for i in range(iterations):
            logger.info(f"\nIteration {i+1}/{iterations}")
            
            try:
                logger.info("Running sequential execution...")
                _, seq_time = self.run_sequential(agent_configs)
                self.metrics.add_metric(seq_time, len(agent_configs), 'sequential')
                logger.info(f"Sequential execution completed in {seq_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in sequential execution: {str(e)}")
                continue
                
            try:
                logger.info("Running distributed execution...")
                _, dist_time = self.run_distributed(agent_configs)
                self.metrics.add_metric(dist_time, len(agent_configs), 'distributed')
                logger.info(f"Distributed execution completed in {dist_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error in distributed execution: {str(e)}")
                continue
        
        return self.metrics.to_dataframe()

def run():
    """CLI entry point for running benchmarks"""
    # Initialize benchmark
    benchmark = Benchmark()
    
    # Agent configurations
    agent_configs = [
        {
            "role": "Researcher",
            "goal": "Research and gather information",
            "backstory": "Expert at finding and collecting information",
            "allow_delegation": True,
            "verbose": True,
            "max_iterations": 1
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
    
    logger.info("Running benchmarks...")
    
    # Initial test
    logger.info("\nRunning initial test with 1 iteration...")
    test_results = benchmark.compare_performance(agent_configs, iterations=1)
    
    if test_results.empty:
        logger.error("Initial test failed. Please check the configuration and try again.")
        return
        
    # Full benchmark
    iterations = [3, 5, 10]
    all_results = test_results
    
    for n_iter in iterations:
        logger.info(f"\nRunning benchmark with {n_iter} iterations...")
        results_df = benchmark.compare_performance(agent_configs, iterations=n_iter)
        if not results_df.empty:
            results_df['num_iterations'] = n_iter
            all_results = pd.concat([all_results, results_df])
    
    if all_results.empty:
        logger.error("No successful benchmark runs completed.")
        return
        
    # Calculate statistics
    sequential_times = all_results[all_results['execution_type'] == 'sequential']['execution_time']
    distributed_times = all_results[all_results['execution_type'] == 'distributed']['execution_time']
    
    logger.info("\nBenchmark Results Summary:")
    logger.info("-" * 50)
    logger.info("\nExecution Times (seconds):")
    logger.info(f"Sequential  - Mean: {sequential_times.mean():.2f}, Min: {sequential_times.min():.2f}, Max: {sequential_times.max():.2f}")
    
    if not distributed_times.empty:
        logger.info(f"Distributed - Mean: {distributed_times.mean():.2f}, Min: {distributed_times.min():.2f}, Max: {distributed_times.max():.2f}")
        speedup = sequential_times.mean() / distributed_times.mean()
        logger.info(f"\nAverage Speedup: {speedup:.2f}x")
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = f"benchmark_results_{timestamp}.csv"
    all_results.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    run()