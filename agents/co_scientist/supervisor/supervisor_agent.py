"""Supervisor agent for orchestrating the AI co-scientist system."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable
import networkx as nx
import json

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

IMPORTANT: Your response MUST be a valid JSON object with the exact structure shown below.
The goal field MUST be a ResearchGoal object containing the EXACT values from the input.
Do not omit any required fields or deviate from the specified formats.

Example valid response:
{{
    "goal": {{
        "goal": "Investigate potential drug repurposing candidates for treating acute myeloid leukemia (AML)",
        "domain": "drug_repurposing",
        "constraints": ["Focus on FDA-approved drugs"],
        "preferences": {{"prioritize_novel_mechanisms": true}}
    }},
    "tasks": [
        {{
            "id": "task1",
            "name": "Literature Review",
            "description": "Review existing literature on AML drug repurposing",
            "expected_duration": "2 days"
        }}
    ],
    "agent_assignments": {{
        "task1": ["generation", "reflection"]
    }},
    "dependencies": {{
        "task1": []
    }}
}}

{format_instructions}"""

        # Create output parser
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        # Create prompt template with escaped brackets
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content="""Please create a research plan based on:

Research Goal: {goal}
Available Agents: {available_agents}
Context: {context}

The research goal contains:
- goal: The research goal statement
- domain: The research domain
- constraints: List of research constraints
- preferences: Dictionary of research preferences

Your response MUST be a valid JSON object with the exact structure shown in the format instructions.
The goal field MUST be a ResearchGoal object containing the EXACT values from the input.
Do not omit any required fields or deviate from the specified formats.

{format_instructions}""")
        ])

        super().__init__(
            llm=llm,
            agent_id=agent_id,
            agent_type="supervisor",
            system_prompt=system_prompt,
            output_parser=None  # Don't use output parser in base class
        )
        
        self.prompt = prompt
        self.parser = parser
        
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
        chain = self.prompt | self.llm
        result = await chain.ainvoke({
            "goal": self.state.research_goal.dict(),
            "available_agents": [
                {
                    "id": agent_id,
                    "type": state.agent_type,
                    "capabilities": list(state.tools.keys())
                }
                for agent_id, state in self.state.active_agents.items()
            ],
            "context": context or {},
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Parse result
        try:
            # Try to parse as JSON first
            if isinstance(result, str):
                result = json.loads(result)
            elif not isinstance(result, dict):
                result = json.loads(str(result))
                
            # Handle malformed response
            if "goal" in result:
                if isinstance(result["goal"], dict):
                    # Extract goal data from malformed response
                    goal_data = result["goal"]
                    if "value" in goal_data:
                        # Handle case where goal is in "value" field
                        goal_data = {
                            "goal": goal_data["value"],
                            "domain": result.get("domain", self.state.research_goal.domain),
                            "constraints": result.get("constraints", self.state.research_goal.constraints),
                            "preferences": result.get("preferences", self.state.research_goal.preferences)
                        }
                    elif not all(k in goal_data for k in ["goal", "domain"]):
                        # Handle case where goal data is incomplete
                        goal_data = {
                            "goal": goal_data.get("value", self.state.research_goal.goal),
                            "domain": goal_data.get("domain", self.state.research_goal.domain),
                            "constraints": goal_data.get("constraints", self.state.research_goal.constraints),
                            "preferences": goal_data.get("preferences", self.state.research_goal.preferences)
                        }
                    try:
                        result["goal"] = ResearchGoal(**goal_data)
                    except Exception as e:
                        # If goal validation fails, use current research goal
                        result["goal"] = self.state.research_goal
                elif isinstance(result["goal"], str):
                    # Handle case where goal is just a string
                    result["goal"] = ResearchGoal(
                        goal=result["goal"],
                        domain=self.state.research_goal.domain,
                        constraints=self.state.research_goal.constraints,
                        preferences=self.state.research_goal.preferences
                    )
            else:
                # If no goal field, use the current research goal
                result["goal"] = self.state.research_goal
            
            # Handle case where value field is at top level
            if "value" in result:
                del result["value"]  # Remove top-level value field
            
            # Ensure other required fields exist with default values
            if "tasks" not in result:
                result["tasks"] = [{
                    "id": "task1",
                    "name": "Literature Review",
                    "description": "Review existing literature and identify potential drug candidates",
                    "expected_duration": "2 days"
                }]
            if "agent_assignments" not in result:
                result["agent_assignments"] = {"task1": ["generation", "reflection"]}
            if "dependencies" not in result:
                result["dependencies"] = {"task1": []}
            
            # Create plan object
            plan = ResearchPlan(**result)
        except Exception as e:
            # If parsing fails, create a default plan
            plan = ResearchPlan(
                goal=self.state.research_goal,
                tasks=[{
                    "id": "task1",
                    "name": "Literature Review",
                    "description": "Review existing literature and identify potential drug candidates",
                    "expected_duration": "2 days"
                }],
                agent_assignments={"task1": ["generation", "reflection"]},
                dependencies={"task1": []}
            )
            
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
        
        # Check if research plan exists
        if not self.state.research_plan:
            self.update_memory("error", "No research plan available")
            return {"task_id": task["id"], "error": "No research plan available"}
            
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