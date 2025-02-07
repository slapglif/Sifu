"""Question answering prompts."""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

class QAResponse(BaseModel):
    """Schema for QA responses"""
    answer: str = Field(description="Detailed answer to the question")
    sources: List[str] = Field(description="List of sources used")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of reasoning")

def get_qa_prompt() -> ChatPromptTemplate:
    """Get the QA prompt template."""
    parser = PydanticOutputParser(pydantic_object=QAResponse)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Answer the given question based on the provided context. Return in JSON format.
{format_instructions}

IMPORTANT:
1. All fields are required
2. answer must be detailed and comprehensive
3. sources must list all used references
4. confidence must be between 0.0 and 1.0
5. reasoning must explain your thought process
6. Do not include any text before or after the JSON
7. Use proper JSON formatting with double quotes

Example response:
{{
    "answer": "Paris is the capital of France",
    "sources": ["World Geography Database"],
    "confidence": 0.95,
    "reasoning": "This is a well-established geographical fact"
}}"""

    human_template = """Answer this question:

Question: {question}
Context: {context}

Focus on:
1. Providing a detailed answer
2. Citing relevant sources
3. Explaining your reasoning
4. Assessing your confidence

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])
    
    prompt = prompt.partial(format_instructions=format_instructions)
    return prompt 