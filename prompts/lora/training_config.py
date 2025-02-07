"""LoRA training configuration prompt."""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

class TrainingConfig(BaseModel):
    """Schema for LoRA training configuration."""
    model_name: str = Field(..., description="Base model name")
    r: int = Field(..., description="LoRA attention dimension", ge=1, le=256)
    lora_alpha: int = Field(..., description="LoRA alpha parameter", ge=1)
    target_modules: List[str] = Field(..., description="Target modules for LoRA")
    lora_dropout: float = Field(..., description="LoRA dropout rate", ge=0.0, le=1.0)
    bias: str = Field(..., description="LoRA bias type", pattern="^(none|all|lora_only)$")
    learning_rate: float = Field(..., description="Learning rate", gt=0.0)
    num_epochs: int = Field(..., description="Number of training epochs", ge=1)
    batch_size: int = Field(..., description="Training batch size", ge=1)

def get_training_config_prompt() -> ChatPromptTemplate:
    """Get the prompt template for generating LoRA training configuration."""
    parser = PydanticOutputParser(pydantic_object=TrainingConfig)
    
    system_template = """You are a LoRA training expert that helps configure training parameters.
Your configurations should be optimized for the specific task while being computationally efficient.

{format_instructions}

Remember to:
1. Choose appropriate model for the task
2. Set reasonable LoRA parameters
3. Configure training hyperparameters
4. Consider computational constraints"""

    human_template = """Please generate a LoRA training configuration for this task:
{task}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_template),
        HumanMessage(content=human_template)
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    return prompt 