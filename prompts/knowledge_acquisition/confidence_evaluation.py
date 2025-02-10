"""Confidence evaluation prompts."""
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class ConfidenceFactors(BaseModel):
    """Schema for confidence evaluation factors"""
    content_quality: float = Field(default=0.5, description="Quality of content", ge=0.0, le=1.0)
    entity_relevance: float = Field(default=0.5, description="Relevance of entities", ge=0.0, le=1.0)
    relationship_validity: float = Field(default=0.5, description="Validity of relationships", ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.5, description="Reliability of source", ge=0.0, le=1.0)
    context_relevance: float = Field(default=0.5, description="Relevance of context", ge=0.0, le=1.0)
    overall: float = Field(default=0.5, description="Overall confidence score", ge=0.0, le=1.0)

class ConfidenceEvaluation(BaseModel):
    """Schema for confidence evaluation"""
    confidence: float = Field(description="Overall confidence score", ge=0.0, le=1.0)
    factors: ConfidenceFactors = Field(description="Detailed confidence factors")
    reasoning: str = Field(description="Reasoning behind confidence evaluation")

def get_confidence_evaluation_prompt() -> ChatPromptTemplate:
    """Get the confidence evaluation prompt template."""
    parser = PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an expert at evaluating confidence in extracted knowledge.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. All confidence scores MUST be numbers between 0.0 and 1.0
4. The factors object MUST include all required fields
5. All strings MUST be properly escaped if they contain special characters
6. Do not include any text before or after the JSON object
7. Do not include any explanations or notes
8. The response should look exactly like this:
{{
    "confidence": 0.85,
    "factors": {{
        "content_quality": 0.9,
        "entity_relevance": 0.8,
        "relationship_validity": 0.85,
        "source_reliability": 0.9,
        "context_relevance": 0.8,
        "overall": 0.85
    }},
    "reasoning": "The content is well-structured and coherent. Entities are relevant to the domain and relationships are logically valid. The source appears reliable and the context is appropriate."
}}

EVALUATION GUIDELINES:
1. Content Quality:
   - Clarity and coherence
   - Completeness of information
   - Technical accuracy
   - Writing quality

2. Entity Relevance:
   - Domain specificity
   - Technical accuracy
   - Coverage of key concepts
   - Proper naming/terminology

3. Relationship Validity:
   - Logical connections
   - Proper directionality
   - Appropriate relationship types
   - Consistency with domain knowledge

4. Source Reliability:
   - Author expertise
   - Publication venue
   - Citation quality
   - Peer review status

5. Context Relevance:
   - Domain alignment
   - Temporal relevance
   - Scope appropriateness
   - Target audience match"""

    human_template = """Evaluate confidence for this content:

Content: {content}
Entities: {entities}
Relationships: {relationships}
Source Type: {source_type}

Remember:
1. Return ONLY a valid JSON object
2. Include all required confidence factors
3. Use proper JSON formatting
4. Do not include any text before or after the JSON"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 