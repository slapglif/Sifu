"""Relationship inference prompts."""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Relationship(BaseModel):
    """Schema for inferred relationships"""
    source: str = Field(description="Source entity or concept")
    relation: Literal["is_a", "has_part", "related_to"] = Field(description="Type of relationship")
    target: str = Field(description="Target entity or concept")

def get_relationship_inference_prompt() -> ChatPromptTemplate:
    """Get the relationship inference prompt template."""
    parser = PydanticOutputParser(pydantic_object=Relationship)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a relationship inference expert. Identify meaningful relationships between concepts.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The relation field MUST be EXACTLY one of: is_a, has_part, related_to
3. Focus on identifying meaningful relationships that reveal domain structure"""

    human_template = """Infer meaningful relationships from these hypotheses:

{{hypotheses}}

Focus on identifying relationships that reveal the domain's structure.
Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 