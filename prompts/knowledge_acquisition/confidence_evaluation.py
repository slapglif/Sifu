"""Confidence evaluation prompts."""
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class ConfidenceFactors(BaseModel):
    """Schema for confidence evaluation factors"""
    content_quality: float = Field(description="Quality and coherence of content", ge=0.0, le=1.0)
    entity_confidence: float = Field(description="Confidence in entity extraction", ge=0.0, le=1.0)
    relationship_validity: float = Field(description="Validity of relationships", ge=0.0, le=1.0)
    source_reliability: float = Field(description="Reliability of the source", ge=0.0, le=1.0)
    context_relevance: float = Field(description="Relevance to the domain", ge=0.0, le=1.0)

class ConfidenceEvaluation(BaseModel):
    """Schema for confidence evaluation"""
    confidence: float = Field(description="Overall confidence score", ge=0.0, le=1.0)
    reasoning: str = Field(description="Detailed explanation of the confidence score")
    factors: ConfidenceFactors = Field(description="Breakdown of confidence factors")

def get_confidence_evaluation_prompt() -> ChatPromptTemplate:
    """Get the confidence evaluation prompt template."""
    parser = PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a confidence evaluation expert. Analyze the extracted knowledge and provide a confidence score.
{format_instructions}

Consider these factors when evaluating confidence:
1. Quality and coherence of extracted content
2. Number and relevance of extracted entities
3. Validity and logic of relationships
4. Source type and reliability
5. Domain relevance and context

Provide detailed reasoning for your confidence assessment."""

    human_template = """Evaluate confidence for this extracted knowledge:

Content: {content}

Entities: {entities}

Relationships: {relationships}

Source Type: {source_type}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 