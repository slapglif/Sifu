from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger

# Configure logging
logger.add("model_behavior.log", rotation="100 MB")

class SimpleResponse(BaseModel):
    """Test simple response format"""
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")

def test_model_behavior():
    """Test different prompting strategies and response formats"""
    
    # Initialize LLM with json format
    llm = ChatOllama(
        model="MFDoom/deepseek-r1-tool-calling:1.5b",
        temperature=0.7,
        format="json",
        mirostat=2,
        mirostat_eta=0.1,
        mirostat_tau=5.0
    )

    
    # Initialize parser
    parser = PydanticOutputParser(pydantic_object=SimpleResponse)
    
    logger.info("Starting model behavior tests")
    
    # Test 1: Using parser format instructions
    logger.info("Test 1: Using parser format instructions")
    prompt = PromptTemplate(
        template="Answer the user question.\n{format_instructions}\n\nQuestion: {question}",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"question": "What is the capital of France?"})
        logger.info(f"Parsed response: {result}")
    except Exception as e:
        logger.error(f"Error in test 1: {e}")
    
    # Test 2: Using explicit schema
    logger.info("\nTest 2: Using explicit schema")
    prompt = PromptTemplate(
        template="""Answer the user question. Your response must be a valid JSON object with these fields:
- answer (string): Your answer to the question
- confidence (number): Your confidence between 0 and 1

Question: {question}""",
        input_variables=["question"]
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"question": "What is the capital of France?"})
        logger.info(f"Parsed response: {result}")
    except Exception as e:
        logger.error(f"Error in test 2: {e}")

if __name__ == "__main__":
    test_model_behavior() 