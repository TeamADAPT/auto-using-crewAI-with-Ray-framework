import ray
from crewai import Agent, Task, Crew
from typing import List, Dict
import time

@ray.remote
class RayAgent:
    """Ray Actor wrapper for CrewAI agents"""
    def __init__(self, agent_config: Dict):
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