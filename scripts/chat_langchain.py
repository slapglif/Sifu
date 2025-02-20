"""LangChain-based chat model implementation with structured output."""

from typing import Any, Dict, List, Optional, Type
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field, SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
import os

class ChatLangChain(BaseChatModel):
    """LangChain-based chat model with structured output support."""
    
    model: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.1)
    api_key: SecretStr
    pydantic_schema: Optional[Type[BaseModel]] = Field(default=None)
    format: Optional[str] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    
    def __init__(self, **kwargs):
        """Initialize with API key and optional parameters."""
        super().__init__(**kwargs)
        
        # Initialize base model
        model_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "api_key": SecretStr(self.api_key.get_secret_value()),
            "convert_system_message_to_human": False,  # Don't convert system messages
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        }
        
        # Add format and response_format if specified
        if self.format:
            model_kwargs["format"] = self.format
            # If format is 'json', ensure response_format is set appropriately
            if self.format == 'json' and not self.response_format:
                model_kwargs["response_format"] = {"type": "json_object"}
        if self.response_format:
            model_kwargs["response_format"] = self.response_format
            
        self._model = ChatGoogleGenerativeAI(**model_kwargs)
        
        # If pydantic_schema is provided, bind it to the model and ensure JSON format
        if self.pydantic_schema:
            self._model = self._model.bind(functions=[self.pydantic_schema])
            # When using pydantic schema, always force JSON format
            if not self.format:
                self._model = self._model.bind(format='json')
                self._model = self._model.bind(response_format={"type": "json_object"})
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Convert messages to format expected by model."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # For JSON format, ensure system message includes format requirement
                if self.format == 'json' or self.pydantic_schema:
                    content = msg.content
                    if "CRITICAL:" not in content:
                        content = f"{content}\n\nCRITICAL: You MUST respond with ONLY a valid JSON object. DO NOT include any text before or after the JSON. DO NOT wrap the JSON in code blocks."
                    converted.append(HumanMessage(content=f"System: {content}"))
                else:
                    converted.append(HumanMessage(content=f"System: {msg.content}"))
            else:
                converted.append(msg)
        return converted
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously."""
        converted_messages = self._convert_messages(messages)
        response = self._model.invoke(converted_messages)
        return ChatResult(generations=[ChatGeneration(message=response)])
        
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        converted_messages = self._convert_messages(messages)
        response = await self._model.ainvoke(converted_messages)
        return ChatResult(generations=[ChatGeneration(message=response)])
        
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat-langchain"
        
    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], **kwargs) -> "ChatLangChain":
        """Create ChatLangChain instance from a Pydantic model."""
        return cls(
            pydantic_schema=model,
            format='json',  # Always use JSON format with Pydantic models
            response_format={"type": "json_object"},
            **kwargs
        ) 