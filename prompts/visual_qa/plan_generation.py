"""Visual QA plan generation prompts."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Plan(BaseModel):
    """Schema for visual QA plan"""
    tasks: List[Dict[str, Any]] = Field(description="List of tasks to execute")
    thought: str = Field(description="Explanation of the plan")

def get_plan_generation_prompt() -> ChatPromptTemplate:
    """Get the plan generation prompt template."""
    parser = PydanticOutputParser(pydantic_object=Plan)
    format_instructions = parser.get_format_instructions()
    
    system_template = """Generate a plan to analyze the image and answer the question.
{format_instructions}

IMPORTANT RULES:
1. All fields are required
2. tasks must be an array of task objects with:
   - idx: unique integer index
   - tool: one of [detect_elements, analyze_scene, answer_question]
   - args: object with required arguments
   - dependencies: array of task indices this depends on
3. thought must explain your planning reasoning
4. Do not include any text before or after the JSON

Available tools:
- detect_elements: Detect visual elements in image
- analyze_scene: Analyze scene composition
- answer_question: Answer specific question about image"""

    human_template = """Generate a plan to analyze this image and answer this question:

Image Path: {image_path}
Question: {question}

Remember:
1. Each task must have a unique idx
2. Dependencies must refer to valid task indices
3. Tool names must match exactly
4. All tasks must have required args

Output ONLY a valid JSON object following the format instructions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template)
    ])
    
    return prompt 