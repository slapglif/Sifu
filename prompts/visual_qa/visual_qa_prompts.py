"""Visual QA prompts for answering questions about images."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class VisualEvidence(BaseModel):
    """Schema for visual evidence"""
    region: Optional[Dict[str, int]] = Field(None, description="Region coordinates")
    description: str = Field(description="Description of the evidence")
    confidence: float = Field(description="Confidence in this evidence", ge=0.0, le=1.0)

class VisualAnswer(BaseModel):
    """Schema for visual QA answers"""
    answer: str = Field(description="Answer to the visual question")
    visual_evidence: List[VisualEvidence] = Field(description="Visual evidence supporting the answer")
    context: str = Field(description="Context from scene analysis")
    confidence: float = Field(description="Overall confidence in answer", ge=0.0, le=1.0)

def get_visual_qa_prompt() -> ChatPromptTemplate:
    """Get the visual QA prompt template."""
    parser = PydanticOutputParser(pydantic_object=VisualAnswer)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a visual question answering expert. Answer questions about images using scene analysis.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. Confidence scores must be between 0.0 and 1.0
3. Provide detailed visual evidence
4. Answer must be clear and concise
5. Context should explain reasoning"""

    human_template = """Answer this question about the image:

Question: {question}
Image (base64): {image}
Scene Description: {scene_description}
Key Objects: {key_objects}
Spatial Relationships: {spatial_relationships}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 