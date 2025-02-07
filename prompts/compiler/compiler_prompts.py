"""Compiler prompts for LLM compiler system."""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class Task(BaseModel):
    """Schema for a task in the execution plan."""
    idx: int = Field(..., description="Task index")
    tool: str = Field(..., description="Tool to use")
    args: Dict[str, Any] = Field(..., description="Tool arguments")
    dependencies: List[int] = Field(default_factory=list, description="Task dependencies")

class Plan(BaseModel):
    """Schema for an execution plan."""
    tasks: List[Task] = Field(..., description="List of tasks to execute")
    thought: str = Field(..., description="Planner's reasoning")

class TaskResult(BaseModel):
    """Schema for task execution results."""
    task_id: int = Field(..., description="Task ID")
    result: Any = Field(..., description="Task result")
    error: Optional[str] = Field(None, description="Error if any")

class JoinDecision(BaseModel):
    """Schema for join decisions."""
    complete: bool = Field(..., description="Whether execution is complete")
    thought: str = Field(..., description="Joiner's reasoning")
    replan: bool = Field(..., description="Whether replanning is needed")
    feedback: Optional[str] = Field(None, description="Feedback for replanning")

class CompilerState(TypedDict):
    """State for LLM compiler workflow."""
    content: str
    plan: Optional[Plan]
    results: List[TaskResult]
    join_decision: Optional[JoinDecision]
    final_result: Optional[Any]

def get_plan_generation_prompt() -> ChatPromptTemplate:
    """Get the prompt template for plan generation."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    
    system_template = """You are a planning expert that generates execution plans.
Your plans should be efficient and well-organized.

You must respond with a JSON object that has two fields:
1. "tasks": A list of task objects, each with:
   - "idx": Task index number
   - "tool": Tool name to use
   - "args": Object with tool arguments
   - "dependencies": List of task indices this depends on
2. "thought": String explaining your planning logic

{format_instructions}

Remember to:
1. Break down complex tasks
2. Consider dependencies
3. Choose appropriate tools
4. Explain your planning logic
5. Format response as valid JSON"""

    human_template = """Please generate an execution plan for this state:
{{state}}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_task_execution_prompt() -> ChatPromptTemplate:
    """Get the prompt template for task execution."""
    parser = PydanticOutputParser(pydantic_object=TaskResult)
    
    system_template = """You are a task execution expert that carries out planned tasks.
Your executions should be precise and reliable.

{format_instructions}

Remember to:
1. Follow task specifications
2. Use tools correctly
3. Handle errors gracefully
4. Return complete results"""

    human_template = """Please execute this task:
{task}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_join_decision_prompt() -> ChatPromptTemplate:
    """Get the prompt template for join decisions."""
    parser = PydanticOutputParser(pydantic_object=JoinDecision)
    
    system_template = """You are a decision-making expert that evaluates execution results.
Your decisions should be based on clear evidence and reasoning.

{format_instructions}

Remember to:
1. Evaluate task results
2. Consider overall progress
3. Identify completion criteria
4. Provide replanning feedback"""

    human_template = """Please evaluate this execution state:
{state}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 