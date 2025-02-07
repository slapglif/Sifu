"""Entity extraction prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class EntityResponse(BaseModel):
    """Schema for entity extraction response"""
    entities: List[str] = Field(description="List of extracted entities")

def get_entity_extraction_prompt() -> ChatPromptTemplate:
    """Get the entity extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=EntityResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Extract entities from the given text. Return in JSON format.
{{{{format_instructions}}}}

IMPORTANT:
1. The entities field is required and must be an array of strings
2. Each entity should be a meaningful term or concept
3. Do not include any text before or after the JSON
4. Use proper JSON formatting with double quotes"""

    human_template = """Extract entities from this text:

{{content}}

Output ONLY a valid JSON object following the format instructions."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 