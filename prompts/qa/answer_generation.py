"""Answer generation prompts."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Answer(BaseModel):
    """Schema for generated answers"""
    answer: str = Field(description="Your detailed answer")
    sources: List[str] = Field(description="List of sources used")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of how you arrived at the answer")
    validation_status: Literal["pending", "validated", "failed"] = Field(description="Validation status")

def get_answer_generation_prompt() -> ChatPromptTemplate:
    """Get the answer generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Answer)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an answer generation expert. Generate comprehensive answers based on the provided context.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The confidence field must be a number between 0.0 and 1.0
3. Validation status must be one of: pending, validated, failed
4. Provide detailed reasoning to support your answer"""

    human_template = """Answer this question based on the provided context:

Question: {question}
Context: {context}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 