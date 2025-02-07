"""Visual QA join decision prompts."""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class JoinDecision(BaseModel):
    """Schema for join decisions"""
    complete: bool = Field(description="Whether execution is complete")
    thought: str = Field(description="Joiner's reasoning")
    replan: bool = Field(description="Whether replanning is needed")
    feedback: Optional[str] = Field(None, description="Feedback for replanning if needed")

def get_join_decision_prompt() -> ChatPromptTemplate:
    """Get the join decision prompt template."""
    parser = PydanticOutputParser(pydantic_object=JoinDecision)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Analyze the results and decide whether to complete or replan.
{format_instructions}

IMPORTANT RULES:
1. All fields are required except feedback
2. complete and replan must be boolean values
3. thought must explain your decision
4. feedback is optional and only needed for replanning
5. Do not include any text before or after the JSON"""

    human_template = """Analyze these results:

Plan: {plan}
Results: {results}

Decide whether to:
1. Complete - if all tasks succeeded
2. Replan - if some tasks failed but can be retried
3. Fail - if critical tasks failed

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 