"""Reformulation prompts for reformulating conversation outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class ReformulatedAnswer(BaseModel):
    """Schema for reformulated answer"""
    answer: str = Field(description="The reformulated answer")
    confidence: float = Field(description="Confidence in the reformulation", ge=0.0, le=1.0)
    reasoning: str = Field(description="Reasoning behind the reformulation")

def get_reformulation_prompt() -> tuple[ChatPromptTemplate, PydanticOutputParser]:
    """Get the reformulation prompt template and parser."""
    parser = PydanticOutputParser(pydantic_object=ReformulatedAnswer)
    format_instructions = parser.get_format_instructions()
    
    system_template = """You are a reformulation expert that converts conversation outputs into clear, concise answers.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. The answer field must be as concise as possible while being complete
3. The confidence field must be between 0.0 and 1.0
4. The reasoning field must explain your reformulation process
5. Format numbers and lists according to the original request
6. Do not include units unless specifically requested
7. Do not use articles or abbreviations unless specified
8. Do not include final punctuation

You must respond in the following format:
{{"answer": "your concise answer here",
  "confidence": 0.95,
  "reasoning": "your reasoning here"}}"""

    human_template = """Earlier you were asked:
{original_task}

Here is the conversation transcript:
{conversation}

Reformulate a clear, concise answer following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    return prompt, parser 