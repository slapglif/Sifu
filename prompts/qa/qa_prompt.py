"""Question answering prompt for answering questions with context."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class QAResponse(BaseModel):
    """Response from QA system."""
    answer: str = Field(description="The answer text")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the answer")

def get_qa_prompt() -> ChatPromptTemplate:
    """Get the prompt template for question answering."""
    parser = PydanticOutputParser(pydantic_object=QAResponse)
    
    system_template = """You are a question answering assistant that helps answer questions using provided context.
Your responses should be accurate and well-supported by the context.

{format_instructions}

CRITICAL: You MUST respond with ONLY a valid JSON object, no other text or explanation.
DO NOT wrap the JSON in markdown code blocks or any other formatting.
DO NOT include ```json or ``` markers.
DO NOT include any text before or after the JSON.

Your response should look exactly like this:
{
    "answer": "The detailed answer to the question",
    "confidence": 0.8,
    "sources": ["Source 1", "Source 2"],
    "reasoning": "Clear explanation of how the answer was derived from the context"
}

Remember to:
1. Use only information from the provided context
2. Cite specific sources from the context
3. Explain your reasoning clearly
4. Express confidence based on context relevance
5. Always include all required fields in the JSON response
6. Use proper JSON formatting with double quotes
7. Ensure all strings are properly escaped"""

    human_template = """Please answer this question using the provided context:

Question: {question}

Context: {context}

Remember to respond with ONLY a valid JSON object."""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 