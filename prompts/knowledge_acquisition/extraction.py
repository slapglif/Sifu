"""Knowledge extraction prompts."""
from typing import List, Dict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Relationship(BaseModel):
    """Schema for knowledge relationships"""
    source: str = Field(description="Source entity or concept")
    relation: Literal["is_a", "has_part", "related_to"] = Field(description="Type of relationship")
    target: str = Field(description="Target entity or concept")
    domain: str = Field(default="knowledge", description="Domain this relationship belongs to")

class SourceMetadata(BaseModel):
    """Schema for source metadata"""
    source_type: Literal["text", "pdf", "web"] = Field(description="Type of source")
    confidence_score: float = Field(description="Confidence in source reliability", ge=0.0, le=1.0)
    domain_relevance: float = Field(description="Relevance to current domain", ge=0.0, le=1.0)
    timestamp: str = Field(description="When the source was processed")
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")
    domain: str = Field(default="knowledge", description="Domain this source belongs to")

class ExtractedKnowledge(BaseModel):
    """Schema for extracted knowledge"""
    content: str = Field(description="A clear, comprehensive summary of the key knowledge extracted from the text")
    entities: List[str] = Field(description="List of important concepts, terms, and entities mentioned in the text")
    relationships: List[Relationship] = Field(description="List of relationships between entities")
    confidence: float = Field(description="Overall confidence in the extraction", ge=0.0, le=1.0)
    metadata: SourceMetadata = Field(description="Source metadata")

def get_knowledge_extraction_prompt() -> ChatPromptTemplate:
    """Get the knowledge extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=ExtractedKnowledge)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a knowledge extraction expert. Extract structured knowledge from text.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. confidence must be a number between 0.0 and 1.0
3. entities must be a non-empty array of meaningful entities from the text
4. Each entity should be a specific, meaningful term or concept
5. relationships must be an array (can be empty) of valid relationship objects
6. Each relationship must connect two entities from the entities list
7. All relationship types must be EXACTLY one of: is_a, has_part, related_to
8. source_type must be one of: text, pdf, web
9. validation_status must be one of: pending, processed, failed
10. timestamp must be in ISO format with timezone
11. Focus on extracting meaningful knowledge that captures the key insights
12. The content field should provide a clear, comprehensive summary"""

    human_template = """Extract structured knowledge from this text:

{{text}}

Focus on identifying:
1. Key concepts and entities
2. Relationships between entities
3. Main insights and knowledge
4. Important patterns or trends

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt