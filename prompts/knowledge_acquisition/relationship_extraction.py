"""Relationship extraction prompts."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class RelationshipResponse(BaseModel):
    """Schema for relationship extraction response"""
    relationships: List[Dict[str, Any]] = Field(description="List of extracted relationships")

def get_relationship_extraction_prompt() -> ChatPromptTemplate:
    """Get the relationship extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=RelationshipResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Extract meaningful relationships between concepts from the text. Return in JSON format.
{{{{format_instructions}}}}

IMPORTANT:
1. The relationships field is required and must be a non-empty array
2. Each relationship must have source, relation, target fields
3. relation must be EXACTLY one of: is_a, has_part, related_to
4. Focus on meaningful relationships that reveal:
   - Hierarchical relationships (is_a) - e.g. "AI is_a tool"
   - Compositional relationships (has_part) - e.g. "AI has_part limitations"
   - General associations (related_to) - e.g. "AI related_to productivity"
5. Extract relationships between:
   - Concepts and their categories
   - Components and their wholes
   - Related ideas or concepts
6. Do not include any text before or after the JSON
7. Use proper JSON formatting with double quotes
8. Return at least 3 meaningful relationships"""

    human_template = """Extract meaningful relationships from this text:

{{content}}

Remember:
1. Return at least 3 meaningful relationships
2. Each relation must be EXACTLY one of: is_a, has_part, related_to
3. Focus on meaningful relationships that reveal the structure and connections
4. Consider hierarchical, compositional, and associative relationships
5. Output ONLY a valid JSON object following the format instructions."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 