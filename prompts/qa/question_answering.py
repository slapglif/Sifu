"""Question answering system prompts."""
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class Answer(BaseModel):
    """Schema for generated answers"""
    answer: str = Field(description="Detailed answer to the question")
    sources: List[str] = Field(description="List of sources used")
    confidence: float = Field(description="Confidence in the answer", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of how the answer was derived")
    validation_status: str = Field(description="Must be one of: pending, validated, failed")

class KnowledgeGap(BaseModel):
    """Schema for knowledge gaps"""
    topic: str = Field(description="Topic where knowledge is missing")
    context: str = Field(description="Context around the knowledge gap")
    priority: float = Field(description="Priority score for filling this gap", ge=0.0, le=1.0)
    suggested_questions: List[str] = Field(description="Questions to help fill the gap")

class Question(BaseModel):
    """Schema for generated questions"""
    question: str = Field(description="The generated question")
    topic: str = Field(description="The topic this question relates to")
    difficulty: float = Field(description="Difficulty score between 0.0 and 1.0", ge=0.0, le=1.0)
    type: Literal["general", "factual", "conceptual", "analytical", "error"] = Field(description="Type of question")
    context: str = Field(description="Context that prompted this question")

def get_answer_generation_prompt() -> ChatPromptTemplate:
    """Get the answer generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Answer)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are an answer generation expert. Generate comprehensive answers based on the provided context.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The confidence field must be a number between 0.0 and 1.0
3. validation_status must be one of: pending, validated, failed
4. Provide detailed reasoning to support your answer"""

    human_template = """Answer this question based on the provided context:

Question: {{question}}
Context: {{context}}"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

def get_knowledge_gap_prompt() -> ChatPromptTemplate:
    """Get the knowledge gap prompt template."""
    parser = PydanticOutputParser(pydantic_object=KnowledgeGap)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a knowledge gap identification expert. Identify gaps in knowledge based on questions and answers.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The priority field must be a number between 0.0 and 1.0
3. Suggested questions should help fill identified knowledge gaps"""

    human_template = """Identify knowledge gaps based on this Q&A:

Question: {{question}}
Answer: {{answer}}
Context: {{context}}"""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

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

    human_template = """Answer this question based on the provided context:

Question: {question}
Context: {context}

Output ONLY a valid JSON object following the format instructions."""

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ]) 