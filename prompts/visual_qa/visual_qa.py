"""Visual QA prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class VisualAnswer(BaseModel):
    """Schema for visual QA answers"""
    answer: str = Field(description="Detailed answer to the question")
    visual_evidence: List[str] = Field(description="Visual evidence points")
    context: str = Field(description="Additional context if needed")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

def get_visual_qa_prompt() -> ChatPromptTemplate:
    """Get the visual QA prompt template."""
    parser = PydanticOutputParser(pydantic_object=VisualAnswer)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a visual question answering expert. Answer questions about images based on analysis.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Confidence score must be between 0.0 and 1.0
3. Provide detailed answers with visual evidence
4. Include relevant context from the scene"""

    human_template = """Answer this question about the image:

Question: {{question}}
Image (base64): {{image}}
Scene Description: {{scene_description}}
Key Objects: {{key_objects}}
Spatial Relationships: {{spatial_relationships}}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 