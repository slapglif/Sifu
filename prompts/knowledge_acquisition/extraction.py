"""Knowledge extraction prompts."""
from typing import List, Dict, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Relationship(BaseModel):
    """Schema for knowledge relationships"""
    source: str = Field(description="Source entity or concept")
    relation: Literal[
        # Methodology relationships
        "uses", "applies", "implements",
        # Performance relationships
        "improves", "outperforms", "achieves",
        # Component relationships
        "contains", "consists_of", "part_of",
        # Comparison relationships
        "better_than", "similar_to", "different_from",
        # Causal relationships
        "leads_to", "causes", "affects",
        # Temporal relationships
        "precedes", "follows", "concurrent_with",
        # Legacy relationships
        "is_a", "has_part", "related_to"
    ] = Field(description="Type of relationship")
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
    relationships: List[Relationship] = Field(default_factory=list, description="List of relationships between entities")
    confidence: float = Field(description="Overall confidence in the extraction", ge=0.0, le=1.0)
    metadata: SourceMetadata = Field(description="Source metadata")

class KeyTermsResponse(BaseModel):
    """Schema for key terms extraction response"""
    terms: List[str] = Field(description="List of key search terms")

def get_knowledge_extraction_prompt() -> ChatPromptTemplate:
    """Get the knowledge extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=ExtractedKnowledge)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Extract knowledge from text, with special handling for academic papers and technical content. Return in JSON format.
{format_instructions}

GUIDELINES:
1. Extract meaningful knowledge, focusing on:
   - Research objectives and goals
   - Methodologies and approaches
   - Key findings and results
   - Technical concepts and terminology
   - Experimental setups and configurations
   - Metrics and measurements
   - Conclusions and implications

2. For entities, identify:
   - Technical terms and concepts
   - Methods and algorithms
   - Tools and frameworks
   - Metrics and measurements
   - Research domains and fields
   - Components and systems
   - Datasets and benchmarks

3. For relationships, capture:
   - Methodology relationships (uses, applies, implements)
   - Performance relationships (improves, outperforms, achieves)
   - Component relationships (contains, consists_of, part_of)
   - Comparison relationships (better_than, similar_to, different_from)
   - Causal relationships (leads_to, causes, affects)
   - Temporal relationships (precedes, follows, concurrent_with)

4. Evaluate confidence based on:
   - Clarity of presentation
   - Experimental validation
   - Statistical significance
   - Reproducibility of results
   - Citation of related work
   - Methodology rigor

5. Include metadata about:
   - Paper type (research, survey, case study)
   - Domain relevance
   - Publication venue
   - Research context
   - Validation approach

Example response:
{{
    "content": "The paper presents a novel approach to domain adaptation for test case generation using CodeT5. The method improves line coverage by 49.9% compared to baselines.",
    "entities": [
        "CodeT5",
        "domain adaptation",
        "test case generation",
        "line coverage",
        "automated testing",
        "machine learning"
    ],
    "relationships": [
        {{
            "source": "domain adaptation",
            "relation": "improves",
            "target": "line coverage"
        }},
        {{
            "source": "CodeT5",
            "relation": "used_for",
            "target": "test case generation"
        }}
    ],
    "confidence": 0.85,
    "metadata": {{
        "source_type": "text",
        "confidence_score": 0.85,
        "domain_relevance": 0.9,
        "timestamp": "2024-02-09T11:42:32.000Z",
        "validation_status": "processed"
    }}
}}"""

    human_template = """Extract knowledge from this text:

{text}

Remember:
1. Return a valid JSON object
2. Include any knowledge you find
3. Use proper JSON formatting"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt

def get_key_terms_prompt() -> ChatPromptTemplate:
    """Get the key terms extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=KeyTermsResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Extract 3-5 key search terms from the text.
{format_instructions}

IMPORTANT:
1. The terms field is required and must be an array of strings
2. Each term should be specific and focused on a single concept
3. Terms should be meaningful for web search
4. Do not include any text before or after the JSON
5. Use proper JSON formatting with double quotes

Example response:
{{
    "terms": [
        "machine learning algorithms",
        "neural network architectures",
        "deep learning frameworks"
    ]
}}"""

    human_template = """Extract key search terms from this text:

{text}

Remember:
1. Return 3-5 specific, focused terms
2. Make terms suitable for web search
3. Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt