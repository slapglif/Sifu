"""QA planning prompts."""

from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Plan(BaseModel):
    """Schema for QA plan"""
    tasks: List[Dict[str, Any]] = Field(description="List of tasks to execute")
    thought: str = Field(description="Explanation of the plan")

def get_qa_plan_prompt() -> Tuple[ChatPromptTemplate, PydanticOutputParser]:
    """Get the QA plan generation prompt template and parser."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Generate a plan to answer the question.
{{{{format_instructions}}}}

IMPORTANT:
1. All fields are required
2. tasks must be an array of task objects with:
   - idx: unique integer index
   - tool: one of [retrieve_context, generate_answer, validate_answer, identify_gaps]
   - args: object with required arguments
   - dependencies: array of task indices this depends on
3. thought must explain your planning reasoning
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes

Available tools:
- retrieve_context: Retrieve relevant context from knowledge base
- generate_answer: Generate answer using context
- validate_answer: Validate generated answer
- identify_gaps: Identify knowledge gaps"""

    human_template = """Generate a plan to answer this question:

{{question}}

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