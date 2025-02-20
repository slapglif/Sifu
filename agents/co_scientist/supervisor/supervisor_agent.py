"""Supervisor agent for orchestrating the AI co-scientist system."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
import networkx as nx

from ..base_agent import BaseAgent, AgentState

class ResearchGoal(BaseModel):
    """Research goal specification."""
    goal: str = Field(description="The research goal/question")
    domain: str = Field(description="Research domain")
    constraints: List[str] = Field(default_factory=list, description="Research constraints")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Research preferences")

class ResearchPlan(BaseModel):
    """Research plan configuration."""
    goal: ResearchGoal
    tasks: List[Dict[str, Any]] = Field(default_factory=list, description="List of research tasks")
    agent_assignments: Dict[str, List[str]] = Field(default_factory=dict, description="Map of tasks to agent IDs")
    dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Task dependencies")

class SupervisorState(AgentState):
    """Supervisor agent state."""
    research_goal: Optional[ResearchGoal] = None
    research_plan: Optional[ResearchPlan] = None
    active_agents: Dict[str, AgentState] = Field(default_factory=dict)
    task_queue: List[Dict[str, Any]] = Field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    context_memory: Dict[str, Any] = Field(default_factory=dict)

class SupervisorAgent(BaseAgent):
    """Supervisor agent for orchestrating the AI co-scientist system."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str = "supervisor",
        system_prompt: Optional[str] = None
    ):
        """Initialize the supervisor agent."""
        if system_prompt is None:
            system_prompt = """You are the supervisor agent responsible for orchestrating the AI co-scientist system.
Your role is to:
1. Parse and understand research goals
2. Create and manage research plans
3. Assign tasks to specialized agents
4. Monitor progress and handle failures
5. Manage system resources and context memory
6. Ensure high-quality research output

Follow these guidelines:
- Break down complex research goals into manageable tasks
- Assign tasks to the most suitable specialized agents
- Track dependencies and ensure proper task ordering
- Maintain context and state across the system
- Handle errors and adapt plans as needed
- Optimize resource utilization
- Ensure research quality and novelty

Your output must be a JSON object with the following structure:
{
    "goal": {
        "goal": "research goal statement",
        "domain": "research domain",
        "constraints": ["constraint1", "constraint2"],
        "preferences": {"key1": "value1", "key2": "value2"}
    },
    "tasks": [
        {
            "id": "task1",
            "name": "task name",
            "description": "task description",
            "expected_duration": "duration estimate"
        }
    ],
    "agent_assignments": {
        "task1": ["agent1", "agent2"],
        "task2": ["agent3"]
    },
    "dependencies": {
        "task2": ["task1"],
        "task3": ["task1", "task2"]
    }
}"""

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="supervisor",
            system_prompt=system_prompt,
            output_parser=PydanticOutputParser(pydantic_object=ResearchPlan)
        )
        
        # Initialize supervisor-specific state
        self.state = SupervisorState(
            agent_id=agent_id,
            agent_type="supervisor",
            active_agents={},
            task_queue=[],
            completed_tasks=[],
            context_memory={}
        )
        
        # Initialize agent graph
        self.agent_graph = nx.DiGraph()
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialized agent with the supervisor."""
        self.state.active_agents[agent.state.agent_id] = agent.state
        self.agent_graph.add_node(
            agent.state.agent_id,
            type=agent.state.agent_type,
            state=agent.state.dict()
        )
        
    def set_research_goal(self, goal: ResearchGoal) -> None:
        """Set the current research goal."""
        self.state.research_goal = goal
        self.update_memory("current_goal", goal.dict())
        
    async def create_research_plan(self, context: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        """Create a research plan for the current goal.
        
        Args:
            context: Optional additional context for plan creation
        """
        if not self.state.research_goal:
            raise ValueError("No research goal set")
            
        # Generate plan using LLM
        result = await self.arun({
            "goal": self.state.research_goal.dict(),
            "available_agents": [
                {
                    "id": agent_id,
                    "type": state.agent_type,
                    "capabilities": list(state.tools.keys())
                }
                for agent_id, state in self.state.active_agents.items()
            ],
            "context": context or {}
        })
        
        # Create plan object
        if isinstance(result, dict):
            plan = ResearchPlan(**result)
        else:
            # If result is already a ResearchPlan (from output parser), use it directly
            plan = result
            
        self.state.research_plan = plan
        return plan
        
    def assign_tasks(self) -> None:
        """Assign tasks to agents based on the research plan."""
        if not self.state.research_plan:
            raise ValueError("No research plan available")
            
        # Clear existing queue
        self.state.task_queue = []
        
        # Add tasks to queue based on dependencies
        for task in self.state.research_plan.tasks:
            deps = self.state.research_plan.dependencies.get(task["id"], [])
            if not deps or all(d in self.state.completed_tasks for d in deps):
                self.state.task_queue.append(task)
                
    async def execute_next_task(self) -> Optional[Dict[str, Any]]:
        """Execute the next task in the queue."""
        if not self.state.task_queue:
            return None
            
        task = self.state.task_queue.pop(0)
        agent_ids = self.state.research_plan.agent_assignments.get(task["id"], [])
        
        results = []
        for agent_id in agent_ids:
            if agent_id in self.state.active_agents:
                agent = self.state.active_agents[agent_id]
                try:
                    # Get agent instance and execute task
                    agent_instance = self._get_agent_instance(agent_id)
                    if agent_instance:
                        result = await agent_instance.arun(task)
                        results.append(result)
                except Exception as e:
                    self.update_memory(f"error_{task['id']}", str(e))
                    return {"task_id": task["id"], "error": str(e)}
                    
        # Mark task as completed
        self.state.completed_tasks.append(task["id"])
        
        # Update context memory
        self.update_memory(f"result_{task['id']}", results)
        
        return {
            "task_id": task["id"],
            "results": results
        }
        
    def _get_agent_instance(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent instance by ID."""
        # This would be implemented to return actual agent instances
        # For now it's a placeholder
        return None
        
    def get_research_status(self) -> Dict[str, Any]:
        """Get current research status."""
        return {
            "goal": self.state.research_goal.dict() if self.state.research_goal else None,
            "plan": self.state.research_plan.dict() if self.state.research_plan else None,
            "completed_tasks": len(self.state.completed_tasks),
            "pending_tasks": len(self.state.task_queue),
            "active_agents": len(self.state.active_agents),
            "context_memory_size": len(self.state.context_memory)
        }
        
    def save_context(self) -> None:
        """Save current context to persistent storage."""
        # This would be implemented to save context to disk/database
        pass
        
    def load_context(self) -> None:
        """Load context from persistent storage."""
        # This would be implemented to load context from disk/database
        pass 