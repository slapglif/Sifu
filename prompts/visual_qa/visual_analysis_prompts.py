"""Visual analysis prompts."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class VisualAnalysisResult(BaseModel):
    """Schema for visual analysis results"""
    description: str = Field(description="Detailed description of the image")
    objects: List[str] = Field(description="List of identified objects")
    attributes: Dict[str, str] = Field(description="Visual attributes")
    relationships: List[str] = Field(description="Spatial relationships")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

def get_visual_analysis_prompt() -> ChatPromptTemplate:
    """Get the visual analysis prompt template."""
    parser = PydanticOutputParser(pydantic_object=VisualAnalysisResult)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Analyze the given image and provide a detailed description. Return in JSON format.
{{{{format_instructions}}}}

IMPORTANT:
1. All fields are required
2. description must be detailed and comprehensive
3. objects must be a non-empty array of identified objects
4. attributes must include visual properties like color, texture, lighting
5. relationships must describe spatial arrangements
6. confidence must be between 0.0 and 1.0
7. Do not include any text before or after the JSON
8. Use proper JSON formatting with double quotes"""

    human_template = """Analyze this image:

Image (base64): {{image}}
Question: {{question}}

Focus on:
1. Providing a detailed description
2. Identifying key objects and elements
3. Describing visual attributes
4. Explaining spatial relationships

Output ONLY a valid JSON object following the format instructions."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 