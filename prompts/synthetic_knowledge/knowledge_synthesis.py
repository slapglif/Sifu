"""Knowledge synthesis prompts."""
from typing import List, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .pattern_recognition import Pattern
from .hypothesis_generation import Hypothesis
from .relationship_inference import Relationship

class SourceMetadata(BaseModel):
    """Schema for source metadata"""
    source_type: Literal["text", "pdf", "web"] = Field(description="Type of source")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    domain_relevance: float = Field(description="Domain relevance score between 0.0 and 1.0", ge=0.0, le=1.0)
    timestamp: str = Field(description="ISO format timestamp with timezone")
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")

class SyntheticKnowledge(BaseModel):
    """Schema for synthetic knowledge"""
    content: str = Field(description="A clear, comprehensive synthesis of the key insights and findings")
    patterns: List[Pattern] = Field(description="Identified patterns")
    hypotheses: List[Hypothesis] = Field(description="Generated hypotheses")
    relationships: List[Relationship] = Field(description="Inferred relationships")
    confidence: float = Field(description="Overall confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")
    metadata: SourceMetadata = Field(description="Source metadata")

def get_knowledge_synthesis_prompt() -> ChatPromptTemplate:
    """Get the knowledge synthesis prompt template."""
    parser = PydanticOutputParser(pydantic_object=SyntheticKnowledge)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a knowledge synthesis expert. Synthesize coherent knowledge from patterns, hypotheses, and relationships.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. All confidence fields must be numbers between 0.0 and 1.0
3. Validation status must be one of: pending, processed, failed
4. Source type must be one of: text, pdf, web
5. All relationship types must be EXACTLY one of: is_a, has_part, related_to
6. Timestamp must be in ISO format with timezone
7. All arrays must be non-empty
8. Focus on synthesizing a coherent understanding of the domain"""

    human_template = """Synthesize knowledge from:

Patterns: {{patterns}}
Hypotheses: {{hypotheses}}
Relationships: {{relationships}}

Focus on creating a coherent understanding of the domain.
Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 