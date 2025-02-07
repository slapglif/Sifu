"""Hypothesis generation prompts."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Hypothesis(BaseModel):
    """Schema for generated hypotheses"""
    statement: str = Field(description="A clear, testable hypothesis statement")
    reasoning: str = Field(description="Detailed reasoning explaining why this hypothesis is plausible")
    evidence: List[str] = Field(description="Specific evidence points that support this hypothesis")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")

def get_hypothesis_generation_prompt() -> ChatPromptTemplate:
    """Get the hypothesis generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Hypothesis)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a hypothesis generation expert. Generate insightful hypotheses from patterns.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Generate hypotheses that explain relationships, predict outcomes, or suggest underlying mechanisms
3. Evidence must include specific supporting points
4. Validation status must be one of: pending, processed, failed"""

    human_template = """Generate insightful hypotheses based on these patterns:

{patterns}

Focus on explaining relationships, predicting outcomes, or suggesting underlying mechanisms.
Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 