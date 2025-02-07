"""Scene analysis prompts."""
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class VisualAttributes(BaseModel):
    """Schema for visual attributes"""
    lighting: str = Field(description="Lighting description")
    composition: str = Field(description="Composition description")
    style: str = Field(description="Style description")

class SceneAnalysis(BaseModel):
    """Schema for scene analysis"""
    scene_description: str = Field(description="Overall description of the scene")
    key_objects: List[str] = Field(description="Important objects in scene")
    spatial_relationships: List[str] = Field(description="Relationships between objects")
    visual_attributes: VisualAttributes = Field(description="Visual attributes of the scene")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

def get_scene_analysis_prompt() -> ChatPromptTemplate:
    """Get the scene analysis prompt template."""
    parser = PydanticOutputParser(pydantic_object=SceneAnalysis)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a scene analysis expert. Analyze the overall scene and relationships between elements.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Confidence score must be between 0.0 and 1.0
3. Provide detailed descriptions of scene elements
4. Include all important spatial relationships"""

    human_template = """Analyze this scene:

Image (base64): {{image}}
Detected Elements: {{elements}}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 