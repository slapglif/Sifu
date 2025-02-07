"""Visual analysis prompt for analyzing images and answering questions."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class VisualAnalysisResult(BaseModel):
    """Result of visual analysis."""
    answer: str = Field(..., description="The answer to the question")
    visual_evidence: List[str] = Field(..., description="Visual evidence supporting the answer")
    context: str = Field(..., description="Context and reasoning behind the answer")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

def get_visual_analysis_prompt() -> ChatPromptTemplate:
    """Get the prompt template for visual analysis."""
    parser = PydanticOutputParser(pydantic_object=VisualAnalysisResult)
    
    system_template = """You are a visual analysis assistant that helps analyze images and answer questions about them.
Your responses should be detailed and well-reasoned, supported by specific visual evidence from the image.

{format_instructions}

Remember to:
1. Carefully examine all visual details in the image
2. Provide specific visual evidence to support your answer
3. Explain your reasoning clearly
4. Express your confidence based on the clarity and completeness of the visual evidence"""

    human_template = """Here is an image encoded in base64:
{image}

Please analyze this image to answer the following question:
{question}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 