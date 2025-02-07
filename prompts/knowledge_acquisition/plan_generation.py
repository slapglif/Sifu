"""Plan generation prompts."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Task(BaseModel):
    """Schema for a task in the execution plan"""
    idx: int = Field(description="Task index")
    tool: str = Field(description="Tool to use")
    args: Dict[str, Any] = Field(description="Tool arguments")
    dependencies: List[int] = Field(default_factory=list, description="Task dependencies")

class Plan(BaseModel):
    """Schema for an execution plan"""
    tasks: List[Task] = Field(description="List of tasks to execute")
    thought: str = Field(description="Planner's reasoning")

def get_plan_generation_prompt() -> ChatPromptTemplate:
    """Get the plan generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Generate a plan to acquire and process knowledge from the given content.
{format_instructions}

Available tools:
- extract_knowledge: Extract structured knowledge from text
- generate_embeddings: Generate embeddings for text
- update_graph: Update knowledge graph
- create_documents: Create final documents

IMPORTANT:
1. Each task must have a unique idx
2. Dependencies must refer to valid task indices
3. Tool names must match exactly
4. All tasks must have required args

Example response:
{{
    "tasks": [
        {{
            "idx": 0,
            "tool": "extract_knowledge",
            "args": {{"text": "sample text"}},
            "dependencies": []
        }},
        {{
            "idx": 1,
            "tool": "generate_embeddings",
            "args": {{"text": "sample text"}},
            "dependencies": [0]
        }}
    ],
    "thought": "First extract knowledge, then generate embeddings"
}}"""

    human_template = """Generate a plan to process this content:

{content}

Remember:
1. Each task must have a unique idx
2. Dependencies must refer to valid task indices
3. Tool names must match exactly
4. All tasks must have required args
5. Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 