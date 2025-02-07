"""Knowledge generation prompts for synthetic knowledge creation."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class Pattern(BaseModel):
    """Schema for identified pattern."""
    name: str = Field(..., description="Name of the pattern")
    description: str = Field(..., description="Description of the pattern")
    examples: List[str] = Field(..., description="Examples of the pattern")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

class Hypothesis(BaseModel):
    """Schema for generated hypothesis."""
    statement: str = Field(..., description="The hypothesis statement")
    evidence: List[str] = Field(..., description="Supporting evidence")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

class Relationship(BaseModel):
    """Schema for inferred relationship."""
    source: str = Field(..., description="Source concept/entity")
    target: str = Field(..., description="Target concept/entity")
    type: str = Field(..., description="Type of relationship")
    description: str = Field(..., description="Description of the relationship")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

class SyntheticKnowledge(BaseModel):
    """Schema for synthetic knowledge."""
    patterns: List[Pattern] = Field(default_factory=list, description="Identified patterns")
    hypotheses: List[Hypothesis] = Field(default_factory=list, description="Generated hypotheses")
    relationships: List[Relationship] = Field(default_factory=list, description="Inferred relationships")
    summary: str = Field(..., description="Summary of the synthetic knowledge")

def get_pattern_recognition_prompt() -> ChatPromptTemplate:
    """Get the prompt template for pattern recognition."""
    parser = PydanticOutputParser(pydantic_object=Pattern)
    
    system_template = """You are a pattern recognition expert that identifies meaningful patterns in information.
Your responses should be detailed and well-supported with specific examples.

{format_instructions}

Remember to:
1. Look for recurring themes and structures
2. Identify both obvious and subtle patterns
3. Provide clear examples of each pattern
4. Express confidence based on evidence strength"""

    human_template = """Please identify patterns related to this topic:
{topic}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_hypothesis_generation_prompt() -> ChatPromptTemplate:
    """Get the prompt template for hypothesis generation."""
    parser = PydanticOutputParser(pydantic_object=Hypothesis)
    
    system_template = """You are a hypothesis generation expert that formulates testable explanations.
Your hypotheses should be clear, falsifiable, and supported by evidence.

{format_instructions}

Remember to:
1. State hypotheses clearly and concisely
2. Provide supporting evidence
3. Consider alternative explanations
4. Express confidence based on evidence strength"""

    human_template = """Please generate hypotheses about this topic:
{topic}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_relationship_inference_prompt() -> ChatPromptTemplate:
    """Get the prompt template for relationship inference."""
    parser = PydanticOutputParser(pydantic_object=Relationship)
    
    system_template = """You are a relationship inference expert that identifies connections between concepts.
Your inferences should be logical and well-supported with evidence.

{format_instructions}

Remember to:
1. Identify meaningful connections
2. Specify relationship types clearly
3. Explain the nature of relationships
4. Express confidence based on evidence strength"""

    human_template = """Please infer relationships related to this topic:
{topic}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt

def get_knowledge_generation_prompt() -> ChatPromptTemplate:
    """Get the prompt template for knowledge generation."""
    parser = PydanticOutputParser(pydantic_object=SyntheticKnowledge)
    
    system_template = """You are a knowledge synthesis expert that generates synthetic knowledge.
Your synthesis should combine patterns, hypotheses, and relationships into coherent knowledge.

{format_instructions}

Remember to:
1. Integrate identified patterns
2. Connect related hypotheses
3. Map relationships between concepts
4. Provide a clear knowledge summary"""

    human_template = """Please generate synthetic knowledge about this topic:
{topic}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 