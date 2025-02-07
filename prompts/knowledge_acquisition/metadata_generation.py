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
    
    system_template = """Generate metadata for the given text. Return in JSON format.
{{{{format_instructions}}}}

IMPORTANT:
1. The metadata field is required and must be an object
2. source_type must be one of: text, pdf, web
3. confidence_score and domain_relevance must be numbers between 0.0 and 1.0
4. timestamp must be in ISO format with timezone (e.g. "2024-02-07T12:00:00Z")
5. validation_status must be one of: pending, processed, failed
6. domain must be "knowledge"
7. Do not include any text before or after the JSON
8. Use proper JSON formatting with double quotes
9. All fields are required
10. The response must be wrapped in a metadata object"""

    human_template = """Generate metadata for this text:

{{content}}

Remember:
1. Return metadata wrapped in a metadata object
2. Use proper JSON formatting
3. Follow the schema exactly
4. All fields are required
5. Use current timestamp in ISO format with timezone
6. Output ONLY a valid JSON object following the format instructions."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 