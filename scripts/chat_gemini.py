"""Custom ChatGemini implementation with native function calling for JSON output."""

from typing import Any, Dict, List, Optional, Union, Mapping, Type, Literal, cast
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel, Field, SecretStr, create_model, PrivateAttr
import google.generativeai as genai  # type: ignore
from google.ai import generativelanguage as glm
from google.ai.generativelanguage_v1beta import types
from google.ai.generativelanguage_v1beta.types import Schema
import json
import os

FunctionMode = Literal["AUTO", "ANY", "NONE", "FUNCTION_CALL"]

class ChatGemini(BaseChatModel):
    """Custom ChatGemini implementation that uses function calling for JSON output."""
    
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=1.0)
    top_k: int = Field(default=1)
    max_tokens: Optional[int] = Field(default=None)
    api_key: SecretStr
    format: str = Field(default="text")  # Can be "text" or "json"
    schema: Optional[Dict[str, Any]] = Field(default=None)  # Schema for JSON output
    pydantic_schema: Optional[Type[BaseModel]] = Field(default=None)  # Pydantic model for output parsing
    function_mode: FunctionMode = Field(default="AUTO")  # Function calling mode
    allowed_functions: Optional[List[str]] = Field(default=None)  # Allowed function names
    
    _client: Any = PrivateAttr()  # Private attribute for the Gemini client
    
    def __init__(self, **kwargs):
        """Initialize with API key and optional parameters."""
        super().__init__(**kwargs)
        
        # Configure API key
        api_key = self.api_key.get_secret_value()
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)  # type: ignore
        
        # Initialize model
        self._client = genai.GenerativeModel(  # type: ignore
            model_name=self.model,
            generation_config=glm.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_tokens,
            )
        )
        
        # If pydantic_schema is provided, convert it to JSON schema
        if self.pydantic_schema:
            self.schema = self.pydantic_schema.model_json_schema()
            self.function_mode = "FUNCTION_CALL"  # Always use FUNCTION_CALL mode with schema
        
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a prompt string."""
        prompt_parts = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
            elif isinstance(message, ChatMessage):
                role = message.role.capitalize()
                prompt_parts.append(f"{role}: {message.content}")
                
        return "\n".join(prompt_parts)
        
    def _create_function_declarations(self, schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create function declarations for JSON output."""
        if not schema:
            return []
        
        def process_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Process schema to handle type fields that are lists."""
            processed = schema.copy()
            
            # Set default type if not specified
            if "type" not in processed:
                if "properties" in processed:
                    processed["type"] = "OBJECT"
                elif "items" in processed:
                    processed["type"] = "ARRAY"
                else:
                    processed["type"] = "STRING"
            else:
                # Convert string type to enum
                type_map = {
                    "object": "OBJECT",
                    "array": "ARRAY",
                    "string": "STRING",
                    "integer": "INTEGER",
                    "boolean": "BOOLEAN",
                    "number": "NUMBER"
                }
                processed["type"] = type_map[processed["type"].lower()]
            
            # Handle array items
            if processed.get("type") == "ARRAY" and "items" in processed:
                if isinstance(processed["items"], dict):
                    processed["items"] = process_schema(processed["items"])
                    if not processed["items"].get("type"):
                        if processed["items"].get("properties"):
                            processed["items"]["type"] = "OBJECT"
                        else:
                            processed["items"]["type"] = "STRING"
                    
                    # Ensure array items have properties if object type
                    if processed["items"].get("type") == "OBJECT":
                        if "properties" not in processed["items"]:
                            processed["items"]["properties"] = {"value": {"type": "STRING"}}
                        else:
                            # Process nested properties
                            for prop_name, prop_schema in processed["items"]["properties"].items():
                                if isinstance(prop_schema, dict):
                                    processed["items"]["properties"][prop_name] = process_schema(prop_schema)
                                else:
                                    processed["items"]["properties"][prop_name] = {"type": "STRING"}
                    
                    # Ensure nested arrays have items
                    if processed["items"].get("type") == "ARRAY":
                        if "items" not in processed["items"]:
                            processed["items"]["items"] = {"type": "STRING"}
                        else:
                            processed["items"]["items"] = process_schema(processed["items"]["items"])
                else:
                    processed["items"] = {"type": "STRING"}
            
            # Handle object properties
            if processed.get("type") == "OBJECT" and "properties" in processed:
                processed_properties = {}
                for k, v in processed["properties"].items():
                    if isinstance(v, dict):
                        processed_properties[k] = process_schema(v)
                    else:
                        processed_properties[k] = {"type": "STRING"}
                processed["properties"] = processed_properties
                
                # Ensure object properties are non-empty
                if not processed["properties"]:
                    processed["properties"] = {"value": {"type": "STRING"}}
                
                # Process nested objects in properties
                for prop_name, prop_schema in processed["properties"].items():
                    if isinstance(prop_schema, dict):
                        processed_schema = process_schema(prop_schema)
                        processed["properties"][prop_name] = processed_schema
                        if processed_schema.get("type") == "OBJECT" and "properties" not in processed_schema:
                            processed_schema["properties"] = {"value": {"type": "STRING"}}
                        elif processed_schema.get("type") == "ARRAY" and "items" in processed_schema:
                            items = processed_schema["items"]
                            if isinstance(items, dict):
                                if not items.get("type"):
                                    if items.get("properties"):
                                        items["type"] = "OBJECT"
                                    else:
                                        items["type"] = "STRING"
                                
                                if items.get("type") == "OBJECT" and "properties" not in items:
                                    items["properties"] = {"value": {"type": "STRING"}}
                                elif items.get("type") == "ARRAY" and "items" not in items:
                                    items["items"] = {"type": "STRING"}
            
            return processed
        
        # Create a function declaration that enforces the schema
        processed_schema = process_schema(schema)
        
        # Ensure root schema has type specified
        if "type" not in processed_schema:
            processed_schema["type"] = "OBJECT"
            if "properties" not in processed_schema:
                processed_schema["properties"] = {"value": {"type": "STRING"}}
        
        return [{
            "name": "generate_structured_response",
            "description": "Generate a structured response according to the provided schema",
            "parameters": processed_schema
        }]
        
    def _create_tool_config(self) -> Dict[str, Any]:
        """Create tool config with function calling settings."""
        config: Dict[str, Any] = {
            "function_calling_config": {
                "mode": "ANY"
            }
        }
        
        if self.function_mode == "ANY" and self.allowed_functions:
            config["function_calling_config"]["allowed_function_names"] = list(self.allowed_functions)
            
        return config
        
    def _dict_to_schema_proto(self, schema_dict: Dict[str, Any]) -> Schema:
        """Recursively convert a JSON schema (dict) into a Schema proto object."""
        schema_proto = Schema()
        if "type" in schema_dict:
            t = schema_dict["type"]
            if isinstance(t, str):
                t = t.upper()
            enum_type = getattr(Schema, "Type", None)
            if enum_type is not None:
                try:
                    converted_t = enum_type.Value(t)
                except Exception as e:
                    converted_t = t
                schema_proto.type = converted_t
            else:
                schema_proto.type = t
            # For primitive types, return early
            if t in ["STRING", "INTEGER", "BOOLEAN", "NUMBER"]:
                return schema_proto
        if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
            for key, value in schema_dict["properties"].items():
                child_schema = self._dict_to_schema_proto(value)
                schema_proto.properties[key] = child_schema
        if "items" in schema_dict and isinstance(schema_dict["items"], dict):
            schema_proto.items = self._dict_to_schema_proto(schema_dict["items"])
        if "required" in schema_dict and isinstance(schema_dict["required"], list):
            schema_proto.required.extend(schema_dict["required"])
        return schema_proto
        
    def _convert_schema_types_to_upper(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert all schema types to uppercase."""
        type_map = {
            "object": "OBJECT",
            "array": "ARRAY",
            "string": "STRING",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "number": "NUMBER"
        }
        
        if not isinstance(schema, dict):
            return schema
        
        result = schema.copy()
        
        # Convert type if present
        if "type" in result and isinstance(result["type"], str):
            result["type"] = type_map[result["type"].lower()]
        
        # Handle properties
        if "properties" in result and isinstance(result["properties"], dict):
            result["properties"] = {
                k: self._convert_schema_types_to_upper(v) 
                for k, v in result["properties"].items()
            }
        
        # Handle array items
        if "items" in result and isinstance(result["items"], dict):
            result["items"] = self._convert_schema_types_to_upper(result["items"])
        
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously."""
        raise NotImplementedError("ChatGemini only supports async generation")
        
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            
            # If format is json and schema is provided, use function calling
            if self.format == "json" and (self.schema or self.pydantic_schema):
                # Create function declarations for JSON output
                function_declarations = self._create_function_declarations(self.schema)
                
                # Determine a non-empty response schema and wrap it if needed
                schema_for_response = {}
                if self.schema:
                    schema_for_response = self.schema
                elif self.pydantic_schema:
                    schema_for_response = self.pydantic_schema.schema()

                # Ensure schema has type and properties
                if "type" not in schema_for_response:
                    schema_for_response = {"type": "OBJECT", "properties": schema_for_response}
                
                # Convert all schema types to uppercase
                schema_for_response = self._convert_schema_types_to_upper(schema_for_response)

                # Ensure that properties is non-empty; if empty, set a default
                if not schema_for_response.get("properties") or len(schema_for_response.get("properties", {})) == 0:
                    schema_for_response["properties"] = {"result": {"type": "STRING"}}
                
                # Set up generation config with function calling mode
                generation_config = types.GenerationConfig(
                    temperature=self.temperature,
                    candidate_count=1,
                )

                # Create tool config for function calling
                tool_config = types.ToolConfig(
                    function_declarations=function_declarations,
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY"
                    )
                )

                response = await self._client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    tool_config=tool_config
                )
                
                # Extract JSON from function call
                if response.candidates[0].content.parts[0].function_call:
                    function_call = response.candidates[0].content.parts[0].function_call
                    json_str = json.dumps(function_call.args)
                else:
                    json_str = response.text
                
                if self.pydantic_schema:
                    try:
                        parsed = self.pydantic_schema.model_validate_json(json_str)
                        json_str = parsed.model_dump_json()
                    except Exception as e:
                        raise ValueError(f"Failed to parse response with Pydantic schema: {str(e)}")
                ai_message = AIMessage(content=json_str)
                
            else:
                # Regular text generation
                response = await self._client.generate_content_async(prompt)
                ai_message = AIMessage(content=response.text)
            
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        except Exception as e:
            raise ValueError(f"Failed to generate response: {str(e)}")
        
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat-gemini"
        
    @classmethod
    def from_pydantic(cls, model: Type[BaseModel], **kwargs) -> "ChatGemini":
        """Create ChatGemini instance from a Pydantic model."""
        return cls(
            format="json",
            pydantic_schema=model,
            schema=model.schema(),
            **kwargs
        ) 