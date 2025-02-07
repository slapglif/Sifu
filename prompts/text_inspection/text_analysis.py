"""Text analysis prompt for analyzing text content."""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class TextSegment(BaseModel):
    """Schema for text segment."""
    content: str = Field(..., description="The text content of the segment")
    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata about the segment")

class Entity(BaseModel):
    """Schema for named entity."""
    text: str = Field(..., description="The entity text")
    title: str = Field(..., description="The entity title/type")

class TextAnalysis(BaseModel):
    """Schema for text analysis results."""
    content: str = Field(..., description="The original text content")
    segments: List[TextSegment] = Field(..., description="Text segments identified")
    key_points: List[str] = Field(..., description="Key points extracted from text")
    entities: List[Entity] = Field(..., description="Named entities found in text")
    relationships: List[str] = Field(..., description="Relationships between concepts")
    summary: str = Field(..., description="Overall summary of the text")

def get_text_analysis_prompt() -> Tuple[ChatPromptTemplate, PydanticOutputParser]:
    """Get the prompt template and parser for text analysis."""
    parser = PydanticOutputParser(pydantic_object=TextAnalysis)
    
    system_template = """You are a text analysis assistant that helps analyze and extract structured information from text.
Your responses should be detailed and well-organized, capturing the key information and relationships in the text.

{format_instructions}

Remember to:
1. Break text into logical segments
2. Identify key points and themes
3. Extract named entities and concepts
4. Map relationships between concepts
5. Provide a concise summary"""

    human_template = """Please analyze this text:
{text}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt, parser 