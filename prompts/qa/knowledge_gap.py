"""Knowledge gap identification prompts."""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class KnowledgeGap(BaseModel):
    """Schema for identified knowledge gaps"""
    topic: str = Field(description="Topic where knowledge is missing")
    context: str = Field(description="Context around the knowledge gap")
    priority: float = Field(description="Priority score between 0.0 and 1.0", ge=0.0, le=1.0)
    suggested_questions: List[str] = Field(description="Questions to fill the gap")

def get_knowledge_gap_prompt() -> ChatPromptTemplate:
    """Get the knowledge gap identification prompt template."""
    parser = PydanticOutputParser(pydantic_object=KnowledgeGap)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a knowledge gap identification expert. Identify gaps in knowledge based on questions and answers.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The priority field must be a number between 0.0 and 1.0
3. Suggested questions should help fill identified knowledge gaps"""

    human_template = """Identify knowledge gaps based on:

Question: {question}
Answer: {answer}
Confidence: {confidence}
Context: {context}

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 