from crewai import Agent, Task, Crew
from typing import List
from .ray_executor import RayExecutor
import ray

class DataAnalysisDemo:
    """Demo application showing distributed data analysis with CrewAI and Ray"""
    
    def __init__(self):
        self.ray_executor = RayExecutor()
        
    def create_analysis_agents(self) -> List[dict]:
        """Create a team of data analysis agents"""
        agents = [
            {
                "role": "Data Collector",
                "goal": "Collect and prepare data for analysis",
                "backstory": "Expert at gathering and cleaning data from various sources",
                "allow_delegation": True
            },
            {
                "role": "Data Analyzer",
                "goal": "Perform in-depth analysis of the prepared data",
                "backstory": "Experienced data scientist with strong analytical skills",
                "allow_delegation": True
            },
            {
                "role": "Report Generator",
                "goal": "Create comprehensive reports from analysis results",
                "backstory": "Skilled at creating clear and insightful reports",
                "allow_delegation": True
            }
        ]
        return agents
    
    def create_analysis_tasks(self) -> List[Task]:
        """Create data analysis tasks"""
        tasks = [
            Task(
                description="Collect and clean the latest market data",
                expected_output="Clean dataset ready for analysis",
            ),
            Task(
                description="Analyze market trends and identify patterns",
                expected_output="Detailed analysis of market trends",
            ),
            Task(
                description="Generate executive summary report",
                expected_output="Executive summary with key findings",
            )
        ]
        return tasks
    
    def run_analysis(self):
        """Execute the data analysis workflow"""
        agents = self.create_analysis_agents()
        tasks = self.create_analysis_tasks()
        
        # Execute tasks using Ray
        ray_agents = self.ray_executor.create_agents(agents)
        results = self.ray_executor.execute_tasks(ray_agents, tasks)
        
        return results

def run():
    """CLI entry point for running the demo"""
    demo = DataAnalysisDemo()
    results = demo.run_analysis()
    for i, result in enumerate(results):
        print(f"\nTask {i+1} Result:")
        print(result)