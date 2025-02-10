"""Entity extraction prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class EntityResponse(BaseModel):
    """Schema for entity extraction response"""
    entities: List[str] = Field(
        description="List of extracted entities",
        default_factory=list
    )

def get_entity_extraction_prompt() -> ChatPromptTemplate:
    """Get the entity extraction prompt template."""
    parser = PydanticOutputParser(pydantic_object=EntityResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an expert at extracting meaningful entities from text.
{format_instructions}

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object
2. The JSON MUST match the schema exactly
3. The entities field MUST be an array of strings
4. Each entity MUST be properly escaped if it contains special characters
5. Do not include any text before or after the JSON object
6. Do not include any explanations or notes
7. The response should look exactly like this:
{{
    "entities": [
        "Machine Learning",
        "Neural Networks",
        "TensorFlow"
    ]
}}

GUIDELINES for entity extraction:
1. Extract meaningful terms, concepts, and entities:
   - Technical concepts and terminology
   - Domain-specific terms
   - Named entities (people, organizations, products)
   - Key processes or methodologies
   - Important tools or technologies
   - Core principles or theories
2. Include both specific and general concepts
3. Clean and normalize entity text:
   - Remove unnecessary punctuation
   - Standardize capitalization
   - Keep acronyms in uppercase
4. Return as many entities as you can find (aim for at least 5-10)
5. If no clear entities, extract key themes or topics
6. Ensure each entity is self-contained and meaningful
7. Avoid overly generic terms unless they're domain-specific"""

    human_template = """Extract entities from this text:

{content}

Remember:
1. Return ONLY a valid JSON object
2. Include all entities you find
3. Use proper JSON formatting
4. Do not include any text before or after the JSON"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 