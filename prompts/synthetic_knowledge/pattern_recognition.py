"""Pattern recognition prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Pattern(BaseModel):
    """Schema for identified patterns"""
    pattern_type: str = Field(description="Type of pattern (e.g., trend, relationship, behavior, concept)")
    description: str = Field(description="Detailed description of the pattern and its significance")
    supporting_evidence: List[str] = Field(description="Specific examples and evidence from the text that support this pattern")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

def get_pattern_recognition_prompt() -> ChatPromptTemplate:
    """Get the pattern recognition prompt template."""
    parser = PydanticOutputParser(pydantic_object=Pattern)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a pattern recognition expert. Identify meaningful patterns in text.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Focus on identifying meaningful, high-level patterns that reveal insights about the domain
3. Supporting evidence must include specific examples from the text"""

    human_template = """Identify meaningful patterns in this text:

{{content}}

Focus on identifying patterns that reveal important insights about the domain.
Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 