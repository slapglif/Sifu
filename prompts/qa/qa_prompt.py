"""Question answering prompt for answering questions with context."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class QAResponse(BaseModel):
    """Schema for QA response."""
    answer: str = Field(..., description="The answer to the question")
    sources: List[str] = Field(..., description="Sources supporting the answer")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(..., description="Reasoning behind the answer")

def get_qa_prompt() -> ChatPromptTemplate:
    """Get the prompt template for question answering."""
    parser = PydanticOutputParser(pydantic_object=QAResponse)
    
    system_template = """You are a question answering assistant that helps answer questions using provided context.
Your responses should be accurate and well-supported by the context.

{format_instructions}

Remember to:
1. Use only information from the provided context
2. Cite specific sources from the context
3. Explain your reasoning clearly
4. Express confidence based on context relevance"""

    human_template = """Please answer this question using the provided context:

Question: {question}

Context: {context}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 