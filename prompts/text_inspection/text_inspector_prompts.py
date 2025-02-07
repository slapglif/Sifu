"""Text inspector prompts for text analysis workflow."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Plan(BaseModel):
    """Schema for text inspection plan"""
    tasks: List[Dict[str, Any]] = Field(description="List of tasks to execute")
    thought: str = Field(description="Explanation of the plan")

class JoinDecision(BaseModel):
    """Schema for join decisions"""
    complete: bool = Field(description="Whether execution is complete")
    thought: str = Field(description="Joiner's reasoning")
    replan: bool = Field(description="Whether replanning is needed")
    feedback: Optional[str] = Field(None, description="Feedback for replanning if needed")

def get_plan_generation_prompt() -> tuple[ChatPromptTemplate, PydanticOutputParser]:
    """Get the plan generation prompt template and parser."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Generate a plan to analyze the given text.
{{{{format_instructions}}}}

IMPORTANT:
1. All fields are required
2. tasks must be an array of task objects with:
   - idx: unique integer index
   - tool: one of [analyze_text, identify_segments, extract_entities, identify_relationships]
   - args: object with required arguments
   - dependencies: array of task indices this depends on
3. thought must explain your planning reasoning
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes"""

    human_template = """Generate a plan to analyze this text:

{{content}}

Remember:
1. Each task must have a unique idx
2. Dependencies must refer to valid task indices
3. Tool names must match exactly
4. All tasks must have required args

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    return prompt, parser

def get_join_decision_prompt() -> tuple[ChatPromptTemplate, PydanticOutputParser]:
    """Get the join decision prompt template and parser."""
    parser = PydanticOutputParser(pydantic_object=JoinDecision)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Analyze the results and decide whether to complete or replan.
{{{{format_instructions}}}}

IMPORTANT:
1. All fields are required except feedback
2. complete and replan must be boolean values
3. thought must explain your decision
4. feedback is optional and only needed for replanning
5. Do not include any text before or after the JSON
6. Use proper JSON formatting with double quotes"""

    human_template = """Analyze these results:

Plan: {{plan}}
Results: {{results}}

Decide whether to:
1. Complete - if all tasks succeeded
2. Replan - if some tasks failed but can be retried
3. Fail - if critical tasks failed

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    return prompt, parser 