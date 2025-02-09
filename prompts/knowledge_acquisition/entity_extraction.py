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
    
    system_template = """Extract entities from the given text. You MUST return ONLY a valid JSON object.

{format_instructions}

CRITICAL JSON FORMATTING RULES:
1. Return ONLY the JSON object - no other text before or after
2. Use ONLY double quotes (") for strings and property names
3. The entities field MUST be a non-empty array of strings
4. Each entity should be a meaningful term or concept
5. Do not include any explanatory text or comments
6. No trailing commas
7. No extra whitespace
8. No single quotes
9. No unescaped characters

EXTRACTION GUIDELINES:
1. Focus on extracting:
   - Key concepts and terms
   - Named entities (people, organizations, products)
   - Technical terms and domain-specific vocabulary
   - Important topics and themes
2. Entities should be specific and meaningful
3. Return at least 3-5 entities
4. Clean and normalize entity text

Example valid response:
{
    "entities": [
        "machine learning",
        "neural networks",
        "deep learning",
        "artificial intelligence",
        "data science"
    ]
}"""

    human_template = """Extract entities from this text:

{content}

Remember:
1. Return ONLY a valid JSON object
2. Include at least 3-5 meaningful entities
3. Use proper JSON formatting
4. No text before or after the JSON"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 