"""Synthetic knowledge generation prompts."""
from typing import List, Dict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Pattern(BaseModel):
    """Schema for recognized patterns"""
    pattern_type: str = Field(description="Type of pattern")
    description: str = Field(description="Detailed description")
    supporting_evidence: List[str] = Field(description="Specific evidence")
    confidence: float = Field(description="Confidence in pattern", ge=0.0, le=1.0)

class Hypothesis(BaseModel):
    """Schema for generated hypotheses"""
    statement: str = Field(description="Clear hypothesis statement")
    reasoning: str = Field(description="Detailed reasoning")
    evidence: List[str] = Field(description="Supporting evidence")
    confidence: float = Field(description="Confidence in hypothesis", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")

class Relationship(BaseModel):
    """Schema for inferred relationships"""
    source: str = Field(description="Source entity")
    relation: Literal["is_a", "has_part", "related_to"] = Field(description="Type of relationship")
    target: str = Field(description="Target entity")

class SyntheticKnowledge(BaseModel):
    """Schema for synthesized knowledge"""
    content: str = Field(description="Clear, comprehensive synthesis")
    patterns: List[Pattern] = Field(description="Recognized patterns")
    hypotheses: List[Hypothesis] = Field(description="Generated hypotheses")
    relationships: List[Relationship] = Field(description="Inferred relationships")
    confidence: float = Field(description="Overall confidence", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")
    metadata: Dict = Field(description="Additional metadata")

def get_pattern_recognition_prompt() -> ChatPromptTemplate:
    """Get the pattern recognition prompt template."""
    parser = PydanticOutputParser(pydantic_object=Pattern)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a pattern recognition expert. Identify meaningful patterns in the data.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The confidence field must be a number between 0.0 and 1.0
3. Focus on identifying meaningful, non-obvious patterns
4. Provide specific evidence to support each pattern"""

    human_template = """Identify patterns in this data:

{data}"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

def get_hypothesis_generation_prompt() -> ChatPromptTemplate:
    """Get the hypothesis generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Hypothesis)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a hypothesis generation expert. Generate meaningful hypotheses based on patterns.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The confidence field must be a number between 0.0 and 1.0
3. validation_status must be one of: pending, processed, failed
4. Focus on generating testable, meaningful hypotheses
5. Provide clear reasoning and evidence"""

    human_template = """Generate hypotheses based on these patterns:

{patterns}"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

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

{hypotheses}

Focus on identifying relationships that reveal the domain's structure."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

def get_knowledge_synthesis_prompt() -> ChatPromptTemplate:
    """Get the knowledge synthesis prompt template."""
    parser = PydanticOutputParser(pydantic_object=SyntheticKnowledge)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a knowledge synthesis expert. Synthesize coherent knowledge from patterns, hypotheses, and relationships.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. All confidence fields must be numbers between 0.0 and 1.0
3. validation_status must be one of: pending, processed, failed
4. All relationship types must be EXACTLY one of: is_a, has_part, related_to
5. Focus on synthesizing a coherent understanding of the domain"""

    human_template = """Synthesize knowledge from:

Patterns: {patterns}
Hypotheses: {hypotheses}
Relationships: {relationships}

Focus on creating a coherent understanding of the domain."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 