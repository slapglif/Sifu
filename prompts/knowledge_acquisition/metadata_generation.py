"""Metadata generation prompts."""
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class MetadataResponse(BaseModel):
    """Schema for metadata generation response"""
    metadata: Dict[str, Any] = Field(description="Generated metadata")

def get_metadata_generation_prompt() -> ChatPromptTemplate:
    """Get the metadata generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=MetadataResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an expert at generating metadata for content.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. The metadata field MUST be an object with specific fields
4. All strings MUST be properly escaped if they contain special characters
5. Do not include any text before or after the JSON object
6. Do not include any explanations or notes
7. The response should look exactly like this:
{{
    "metadata": {{
        "source_type": "text",
        "confidence_score": 0.85,
        "domain_relevance": 0.9,
        "timestamp": "2024-02-09T11:42:32.000Z",
        "validation_status": "processed",
        "domain": "machine_learning"
    }}
}}

GUIDELINES for metadata generation:
1. Analyze content to determine:
   - Source type (text, pdf, web)
   - Domain relevance
   - Confidence in reliability
   - Validation status
   - Domain classification

2. Consider factors like:
   - Content quality and coherence
   - Technical depth
   - Citation of sources
   - Author expertise
   - Publication context

3. Ensure metadata is:
   - Accurate and objective
   - Well-structured
   - Complete
   - Consistent"""

    human_template = """Generate metadata for this text:

{text}

Remember:
1. Return ONLY a valid JSON object
2. Include all required metadata fields
3. Use proper JSON formatting
4. Do not include any text before or after the JSON"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 