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

IMPORTANT: State variables MUST be referenced using {state[variable_name]} format, e.g. {state[domain_name]}.
NEVER use {state} directly - always use {state[variable_name]} for state variables.

Available tools:
- extract_knowledge: Extract structured knowledge from text
  args: {{"text": "{state[content]}"}}
- generate_embeddings: Generate embeddings for text
  args: {{"text": "{state[content]}"}}
- update_graph: Update knowledge graph
  args: {{"knowledge": {state[knowledge]}}}
- create_documents: Create final documents
  args: {{"content": {state[content]}, "metadata": {state[metadata]}}}

CRITICAL:
1. Each task must have a unique idx starting from 0
2. Dependencies must refer to valid task indices
3. Tool names must match exactly
4. All tasks must have required args
5. State variables must use {state[variable_name]} format
6. Never use {state} directly

Example response:
{{
    "tasks": [
        {{
            "idx": 0,
            "tool": "extract_knowledge",
            "args": {{"text": "{state[content]}"}},
            "dependencies": []
        }},
        {{
            "idx": 1,
            "tool": "generate_embeddings",
            "args": {{"text": "{state[content]}"}},
            "dependencies": [0]
        }}
    ],
    "thought": "First extract knowledge from content, then generate embeddings"
}}"""

    human_template = """Current state:
{state}

Generate a plan to process the content based on the current state.

Remember:
1. Return ONLY a valid JSON object
2. Include both tasks and thought fields
3. Use {state[variable_name]} format for state variables
4. Make task idx start from 0 and increment sequentially
5. Ensure dependencies refer to valid task IDs
6. Include a clear thought explaining your plan
7. NEVER use {state} directly"""

    messages = [
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]

    return ChatPromptTemplate(messages=messages) 