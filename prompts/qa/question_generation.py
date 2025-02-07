"""Question generation prompts."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Question(BaseModel):
    """Schema for generated questions"""
    question: str = Field(description="The generated question")
    topic: str = Field(description="The topic this question relates to")
    difficulty: float = Field(description="Difficulty score between 0.0 and 1.0", ge=0.0, le=1.0)
    type: Literal["general", "factual", "conceptual", "analytical", "error"] = Field(description="Type of question")
    context: str = Field(description="Context that prompted this question")

def get_question_generation_prompt() -> ChatPromptTemplate:
    """Get the question generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Question)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a question generation expert. Generate specific, focused questions based on the content.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The difficulty field must be a number between 0.0 and 1.0
3. Question type must be one of: general, factual, conceptual, analytical, error
4. Generate questions that test understanding and critical thinking"""

    human_template = """Generate questions based on this context and topic:

Context: {context}
Topic: {topic}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 