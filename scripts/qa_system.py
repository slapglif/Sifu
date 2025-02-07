from typing import List, Dict, Optional, Any, Literal, cast
from datetime import datetime
from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from loguru import logger
from langchain_ollama import ChatOllama
import json
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from scripts.logging_config import log_error_with_traceback
import asyncio

from prompts.qa import (
    QUESTION_GENERATION_SYSTEM,
    QUESTION_GENERATION_HUMAN,
    ANSWER_GENERATION_SYSTEM,
    ANSWER_GENERATION_HUMAN,
    KNOWLEDGE_GAP_SYSTEM,
    KNOWLEDGE_GAP_HUMAN
)

# Initialize console
console = Console()

class Question(BaseModel):
    """Schema for generated questions"""
    question: str = Field(description="The generated question")
    topic: str = Field(description="The topic this question relates to")
    difficulty: float = Field(description="Difficulty score between 0.0 and 1.0", ge=0.0, le=1.0)
    type: Literal["general", "factual", "conceptual", "analytical", "error"] = Field(description="Type of question")
    context: str = Field(description="Context that prompted this question")

    @validator("question")
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()

    @validator("topic")
    def validate_topic(cls, v):
        if not v or not v.strip():
            raise ValueError("topic cannot be empty")
        return v.strip()

    @validator("context")
    def validate_context(cls, v):
        if not v or not v.strip():
            raise ValueError("context cannot be empty")
        return v.strip()

class Answer(BaseModel):
    """Schema for generated answers"""
    answer: str = Field(description="The detailed answer")
    sources: List[str] = Field(description="List of sources used")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of how the answer was derived")
    validation_status: Literal["pending", "validated", "failed"] = Field(description="Status of answer validation")

    @validator("answer")
    def validate_answer(cls, v):
        if not v or not v.strip():
            raise ValueError("answer cannot be empty")
        return v.strip()

    @validator("reasoning")
    def validate_reasoning(cls, v):
        if not v or not v.strip():
            raise ValueError("reasoning cannot be empty")
        return v.strip()

class KnowledgeGap(BaseModel):
    """Schema for identified knowledge gaps"""
    topic: str = Field(description="Topic where knowledge is missing")
    context: str = Field(description="Context around the knowledge gap")
    priority: float = Field(description="Priority score between 0.0 and 1.0", ge=0.0, le=1.0)
    suggested_questions: List[str] = Field(description="Questions to fill the gap")

    @validator("topic")
    def validate_topic(cls, v):
        if not v or not v.strip():
            raise ValueError("topic cannot be empty")
        return v.strip()

    @validator("context")
    def validate_context(cls, v):
        if not v or not v.strip():
            raise ValueError("context cannot be empty")
        return v.strip()

    @validator("suggested_questions")
    def validate_questions(cls, v):
        if not v:
            raise ValueError("suggested_questions cannot be empty")
        return [q.strip() for q in v if q and q.strip()]

class QAState(BaseModel):
    """Schema for QA workflow state"""
    question: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[Dict] = None
    confidence: float = 0.0
    gaps: List[Dict] = []

class QASystem:
    """Question answering system"""
    def __init__(self, graph: Neo4jGraph, llm: Optional[Any] = None, model: str = "MFDoom/deepseek-r1-tool-calling:1.5b", temperature: float = 0.7):
        """Initialize the QA system"""
        self.graph = graph
        self.llm = llm if llm is not None else ChatOllama(model=model, temperature=temperature, format="json", mirostat=2, mirostat_eta=0.1, mirostat_tau=5.0)
        self.question_workflow = self._create_question_workflow()
        self.answer_workflow = self._create_answer_workflow()
        self.current_topic = ""
        self.current_context = ""
        self.num_questions = 5

    def _create_question_workflow(self) -> RunnableSequence:
        """Create the question generation workflow"""
        # Question generation chain
        workflow = cast(RunnableSequence, (
            {
                "topic": lambda x: self.current_topic,
                "num_questions": lambda x: self.num_questions,
                "context": lambda x: self.current_context
            }
            | ChatPromptTemplate.from_messages([
                SystemMessage(content=QUESTION_GENERATION_SYSTEM),
                HumanMessage(content=QUESTION_GENERATION_HUMAN)
            ])
            | self.llm
            | RunnableLambda(self._parse_questions)
        ))
        
        return workflow

    def _create_answer_workflow(self) -> RunnableSequence:
        """Create the answer generation workflow"""
        # Context retrieval chain
        retrieval_chain = RunnableLambda(self._retrieve_context)
        
        # Answer generation chain
        answer_chain = cast(RunnableSequence, (
            {
                "question": lambda x: x["question"],
                "context": lambda x: x["context"]
            }
            | ChatPromptTemplate.from_messages([
                SystemMessage(content=ANSWER_GENERATION_SYSTEM),
                HumanMessage(content=ANSWER_GENERATION_HUMAN)
            ])
            | self.llm
            | RunnableLambda(self._parse_answer)
        ))
        
        # Combine chains
        workflow = cast(RunnableSequence, (
            retrieval_chain
            | answer_chain
        ))
        
        return workflow

    def _retrieve_context(self, state: Dict) -> Dict:
        """Retrieve context from knowledge graph"""
        question = state.get("question")
        if not question:
            return state
        
        results = self.graph.query(f"""
            MATCH (d:Document)
            WHERE d.content CONTAINS $question
            RETURN d.content as content
            LIMIT 5
        """, {"question": question})
        
        context = "\n".join([r["content"] for r in results])
        return {**state, "context": context}

    def _validate_answer(self, state: Dict) -> Dict:
        """Validate the generated answer"""
        answer = state.get("answer", {})
        if not answer:
            return {**state, "confidence": 0.0}
        
        # Simple validation based on confidence and reasoning
        confidence = answer.get("confidence", 0.0)
        reasoning = answer.get("reasoning", "")
        
        if confidence >= 0.7 and reasoning.strip():
            answer["validation_status"] = "validated"
        else:
            answer["validation_status"] = "failed"
            confidence = max(0.0, confidence - 0.2)  # Penalize confidence
        
        return {**state, "answer": answer, "confidence": confidence}

    async def process_question(self, question: str) -> Dict:
        """Process a question through the workflow"""
        try:
            # Initialize state
            initial_state = {"question": question}
            
            # Run workflow
            try:
                final_state = await self.answer_workflow.ainvoke(initial_state)
                
                # Print stats
                if final_state and isinstance(final_state, dict):
                    console.print(Panel.fit(
                        f"""
                        Question: {question}
                        Answer Length: {len(str(final_state.get('answer', '')))} chars
                        Sources Used: {len(final_state.get('sources', []))}
                        Confidence: {final_state.get('confidence', 0.0):.2f}
                        Validation Status: {final_state.get('validation_status', 'unknown')}
                        """,
                        title="QA Results"
                    ))
                
                return final_state if isinstance(final_state, dict) else {
                    "answer": "Error: Unexpected workflow result",
                    "sources": [],
                    "confidence": 0.0,
                    "reasoning": "Workflow returned invalid format",
                    "validation_status": "failed"
                }
            except Exception as e:
                log_error_with_traceback(e, "Error running QA workflow")
                return {
                    "answer": f"Error running workflow: {str(e)}",
                    "sources": [],
                    "confidence": 0.0,
                    "reasoning": "Workflow error",
                    "validation_status": "failed"
                }
            
        except Exception as e:
            log_error_with_traceback(e, "Error processing question")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "reasoning": "Processing error",
                "validation_status": "failed"
            }

    async def generate_questions(self, topic: str, num_questions: int = 5) -> List[Question]:
        """Generate questions about a topic"""
        try:
            # Set current state
            self.current_topic = topic
            self.current_context = ""
            self.num_questions = num_questions
            
            # Query knowledge graph for context
            context_results = await asyncio.to_thread(
                self.graph.query,
                f"""
                MATCH (d:Document)
                WHERE d.content CONTAINS $topic
                RETURN d.content as content
                LIMIT 5
                """,
                {"topic": topic}
            )
            
            # Format context
            self.current_context = "\n".join([r["content"] for r in context_results])
            
            # Initialize state
            initial_state = {
                "topic": topic,
                "num_questions": num_questions,
                "context": self.current_context
            }
            
            # Run workflow
            try:
                result = await self.question_workflow.ainvoke(initial_state)
                if isinstance(result, list):
                    return result
                return [
                    Question(
                        question="Error: Unexpected workflow result",
                        topic=topic,
                        difficulty=0.0,
                        type="error",
                        context="Workflow error"
                    )
                ]
            except Exception as e:
                logger.error(f"Error running workflow: {e}")
                return [
                    Question(
                        question="Error occurred while running workflow",
                        topic=topic,
                        difficulty=0.0,
                        type="error",
                        context=f"Error: {str(e)}"
                    )
                ]
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return [
                Question(
                    question="Error occurred while generating questions",
                    topic=topic,
                    difficulty=0.0,
                    type="error",
                    context=f"Error: {str(e)}"
                )
            ]

    def _parse_questions(self, response: Any) -> List[Question]:
        """Parse LLM response into list of Question objects"""
        try:
            # Get the raw text
            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "messages"):
                text = response.messages[-1].content
            else:
                text = str(response)
            
            # Parse as JSON
            if isinstance(text, str):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # If not JSON, try to extract questions from text
                    questions = []
                    for line in text.split("\n"):
                        if "?" in line:
                            questions.append({
                                "question": line.strip(),
                                "topic": self.current_topic,
                                "difficulty": 0.5,
                                "type": "general",
                                "context": "Extracted from non-JSON response"
                            })
                    data = questions
            else:
                data = text
            
            # Ensure we have a list
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = [data]
                else:
                    data = [{
                        "question": str(data),
                        "topic": self.current_topic,
                        "difficulty": 0.5,
                        "type": "general",
                        "context": "Non-list response"
                    }]
            
            # Process each question
            result = []
            for item in data:
                if not isinstance(item, dict):
                    item = {
                        "question": str(item),
                        "topic": self.current_topic,
                        "difficulty": 0.5,
                        "type": "general",
                        "context": "Non-dictionary item"
                    }
                
                # Fill in missing fields with defaults
                item.setdefault("question", "No question provided")
                item.setdefault("topic", self.current_topic)
                item.setdefault("difficulty", 0.5)
                item.setdefault("type", "general")
                item.setdefault("context", "No context provided")
                
                # Create Question object
                try:
                    question = Question(
                        question=item["question"],
                        topic=item["topic"],
                        difficulty=float(item["difficulty"]),
                        type=item["type"],
                        context=item["context"]
                    )
                    result.append(question)
                except Exception as e:
                    logger.warning(f"Error creating Question object: {e}")
                    continue
            
            return result if result else [
                Question(
                    question="No valid questions could be parsed",
                    topic=self.current_topic,
                    difficulty=0.0,
                    type="error",
                    context="Failed to parse any valid questions from response"
                )
            ]
            
        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            return [
                Question(
                    question="Error parsing model response",
                    topic=self.current_topic,
                    difficulty=0.0,
                    type="error",
                    context=f"Error: {str(e)}"
                )
            ]

    def _parse_answer(self, response: Any) -> Dict:
        """Parse LLM response into Answer object"""
        try:
            # Get the raw text
            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "messages"):
                text = response.messages[-1].content
            else:
                text = str(response)
            
            # Parse as JSON
            if isinstance(text, str):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # If not JSON, try to extract answer from text
                    data = {
                        "answer": text.strip(),
                        "sources": [],
                        "confidence": 0.5,
                        "reasoning": "Extracted from non-JSON response",
                        "validation_status": "pending"
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "answer": str(data),
                    "sources": [],
                    "confidence": 0.5,
                    "reasoning": "Non-dictionary response",
                    "validation_status": "pending"
                }
            
            # Fill in missing fields with defaults
            data.setdefault("answer", "No answer provided")
            data.setdefault("sources", [])
            data.setdefault("confidence", 0.5)
            data.setdefault("reasoning", "No reasoning provided")
            data.setdefault("validation_status", "pending")
            
            return {
                "answer": data["answer"],
                "sources": data["sources"],
                "confidence": float(data["confidence"]),
                "reasoning": data["reasoning"],
                "validation_status": data["validation_status"]
            }
            
        except Exception as e:
            logger.error(f"Error parsing answer: {e}")
            return {
                "answer": "Error parsing model response",
                "sources": [],
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "validation_status": "failed"
            }

    def _parse_gap(self, response: Any) -> Dict:
        """Parse LLM response into KnowledgeGap object"""
        try:
            # Get the raw text
            if hasattr(response, "content"):
                text = response.content
            elif hasattr(response, "messages"):
                text = response.messages[-1].content
            else:
                text = str(response)
            
            # Parse as JSON
            if isinstance(text, str):
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # If not JSON, try to extract gap from text
                    data = {
                        "topic": self.current_topic,
                        "context": text.strip(),
                        "priority": 0.5,
                        "suggested_questions": []
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "topic": self.current_topic,
                    "context": str(data),
                    "priority": 0.5,
                    "suggested_questions": []
                }
            
            # Fill in missing fields with defaults
            data.setdefault("topic", self.current_topic)
            data.setdefault("context", "No context provided")
            data.setdefault("priority", 0.5)
            data.setdefault("suggested_questions", [])
            
            return {
                "topic": data["topic"],
                "context": data["context"],
                "priority": float(data["priority"]),
                "suggested_questions": data["suggested_questions"]
            }
            
        except Exception as e:
            logger.error(f"Error parsing knowledge gap: {e}")
            return {
                "topic": self.current_topic,
                "context": f"Error parsing model response: {str(e)}",
                "priority": 0.0,
                "suggested_questions": []
            } 