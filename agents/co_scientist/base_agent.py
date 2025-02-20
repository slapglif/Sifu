"""Base agent class for AI co-scientist system."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableSerializable

class AgentState(BaseModel):
    """Base state for all agents."""
    agent_id: str = Field(description="Unique identifier for this agent")
    agent_type: str = Field(description="Type of agent (e.g. generation, reflection, etc)")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Agent's memory/context")
    tools: Dict[str, Any] = Field(default_factory=dict, description="Tools available to this agent")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Agent performance metrics")

class BaseAgent:
    """Base agent class with common functionality."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        agent_id: str,
        agent_type: str,
        system_prompt: str,
        output_parser: Optional[PydanticOutputParser] = None
    ):
        """Initialize the base agent.
        
        Args:
            llm: The language model to use
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g. generation, reflection, etc)
            system_prompt: System prompt for this agent
            output_parser: Optional output parser for structured outputs
        """
        self.llm = llm
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            memory={},
            tools={},
            metrics={}
        )
        self.system_prompt = system_prompt
        self.output_parser = output_parser
        
        # Create base prompt template
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="{input}")
        ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        
        # Create chain
        self.chain = self.prompt | self.llm
        if output_parser:
            self.chain = self.chain | output_parser
            
    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool with this agent."""
        self.state.tools[name] = tool
        
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's memory."""
        self.state.memory[key] = value
        
    def get_memory(self, key: str) -> Optional[Any]:
        """Get value from agent's memory."""
        return self.state.memory.get(key)
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update agent's performance metrics."""
        self.state.metrics.update(metrics)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent's performance metrics."""
        return self.state.metrics
        
    async def arun(self, input_data: Dict[str, Any]) -> Any:
        """Run the agent's chain asynchronously."""
        try:
            from rich.console import Console
            console = Console()
            with console.status(f"[bold cyan]{self.state.agent_type}[/bold cyan] agent thinking..."):
                result = await self.chain.ainvoke(input_data)
            return result
        except Exception as e:
            raise RuntimeError(f"Agent {self.state.agent_id} failed: {str(e)}")
            
    def get_state(self) -> AgentState:
        """Get the current agent state."""
        return self.state 