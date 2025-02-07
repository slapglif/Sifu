"""Visual element detection prompts."""
from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Region(BaseModel):
    """Schema for visual region"""
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    width: int = Field(description="Width of region")
    height: int = Field(description="Height of region")
    content: str = Field(description="Region content description")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

class VisualElement(BaseModel):
    """Schema for detected visual elements"""
    element_type: str = Field(description="Type of visual element")
    description: str = Field(description="Detailed description of the element")
    attributes: Dict[str, str] = Field(description="Element attributes")
    region: Optional[Region] = Field(None, description="Region information")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

def get_element_detection_prompt() -> ChatPromptTemplate:
    """Get the visual element detection prompt template."""
    parser = PydanticOutputParser(pydantic_object=VisualElement)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a visual element detection expert. Detect and describe visual elements in the image.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Confidence scores must be between 0.0 and 1.0
3. Region coordinates must be valid integers
4. Provide detailed descriptions of elements"""

    human_template = """Detect and describe visual elements in this image:

Image (base64): {{image}}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 