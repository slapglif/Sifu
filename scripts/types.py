from typing import Dict, List, Optional, TypedDict, Any, Union
from pydantic import BaseModel
from langchain_core.documents import Document
from prompts.compiler.compiler_prompts import Plan, TaskResult, JoinDecision

class CompilerState(TypedDict, total=False):
    """Compiler state with optional fields"""
    # Required fields
    content: str
    error: Optional[str]
    feedback: Optional[str]
    
    # Optional fields
    domain_name: str
    plan: Optional[Plan]  # Use Plan type directly
    results: List[TaskResult]  # Use TaskResult type directly
    join_decision: Optional[JoinDecision]
    final_result: Optional[Dict[str, Any]]
    knowledge_sources: List[Union[Document, Dict[str, Any]]]
    synthetic_knowledge: List[Dict[str, Any]]
    serializable_knowledge: List[Dict[str, Any]]
    training_examples: List[Dict[str, Any]]
    model_metrics: Dict[str, Any]

class SystemState(BaseModel):
    """System state"""
    domain_name: str
    knowledge_sources: List[Dict[str, Any]] = []
    generated_questions: List[Dict[str, Any]] = []
    synthetic_knowledge: List[Dict[str, Any]] = []
    training_examples: List[Dict[str, Any]] = []
    model_metrics: Dict[str, Any] = {} 