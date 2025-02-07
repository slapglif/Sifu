from typing import List, Dict, Optional, Any, Tuple, cast, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from langchain_ollama import ChatOllama
import json

from scripts.models import ExtractedKnowledge, Relationship, SourceMetadata
from scripts.logging_config import log_error_with_traceback, log_info_with_context, log_warning_with_context

from prompts.synthetic_knowledge.generation import (
    PATTERN_RECOGNITION_SYSTEM,
    PATTERN_RECOGNITION_HUMAN,
    HYPOTHESIS_GENERATION_SYSTEM,
    HYPOTHESIS_GENERATION_HUMAN,
    RELATIONSHIP_INFERENCE_SYSTEM,
    RELATIONSHIP_INFERENCE_HUMAN,
    KNOWLEDGE_SYNTHESIS_SYSTEM,
    KNOWLEDGE_SYNTHESIS_HUMAN
)

class Pattern(BaseModel):
    """Schema for identified patterns"""
    pattern_type: str = Field(description="Type of pattern identified")
    description: str = Field(description="Description of the pattern")
    supporting_evidence: List[str] = Field(description="Evidence supporting this pattern")
    confidence: float = Field(description="Confidence in pattern", ge=0.0, le=1.0)

class Hypothesis(BaseModel):
    """Schema for generated hypotheses"""
    statement: str = Field(description="The hypothesis statement")
    reasoning: str = Field(description="Reasoning behind the hypothesis")
    evidence: List[str] = Field(description="Supporting evidence")
    confidence: float = Field(description="Confidence in hypothesis", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")

class SyntheticKnowledge(BaseModel):
    """Schema for synthetic knowledge"""
    content: str = Field(description="Synthesized knowledge content")
    patterns: List[Pattern] = Field(description="Identified patterns")
    hypotheses: List[Hypothesis] = Field(description="Generated hypotheses")
    relationships: List[Relationship] = Field(description="Inferred relationships")
    confidence: float = Field(description="Overall confidence", ge=0.0, le=1.0)
    validation_status: Literal["pending", "processed", "failed"] = Field(description="Validation status")
    metadata: Optional[SourceMetadata] = Field(None, description="Source metadata")

class SynthesisState(BaseModel):
    """Schema for synthesis workflow state"""
    input_documents: List[Document] = Field(description="Input documents")
    identified_patterns: List[Pattern] = Field(default_factory=list, description="Identified patterns")
    generated_hypotheses: List[Hypothesis] = Field(default_factory=list, description="Generated hypotheses")
    inferred_relationships: List[Relationship] = Field(default_factory=list, description="Inferred relationships")
    synthetic_knowledge: Optional[SyntheticKnowledge] = Field(None, description="Generated synthetic knowledge")

console = Console()

class SyntheticKnowledgeGenerator:
    """Generate synthetic knowledge from documents"""
    def __init__(self, graph: Neo4jGraph, llm: Optional[Any] = None, model: str = "MFDoom/deepseek-r1-tool-calling:1.5b", temperature: float = 0.7):
        """Initialize the generator"""
        self.graph = graph
        self.llm = llm if llm is not None else ChatOllama(model=model, temperature=temperature, format="json", mirostat=2, mirostat_eta=0.1, mirostat_tau=5.0)
        self.state_graph = self._create_workflow()
        self.console = Console()

    def _create_workflow(self) -> StateGraph:
        """Create the knowledge synthesis workflow"""
        # Pattern recognition node
        async def recognize_patterns(state: Dict) -> Dict:
            """Recognize patterns in the current state"""
            try:
                log_info_with_context("Starting pattern recognition", "Knowledge Generation")
                
                if isinstance(state, dict):
                    current_state = SynthesisState(**state)
                else:
                    current_state = state
                
                parser = PydanticOutputParser(pydantic_object=Pattern)
                format_instructions = parser.get_format_instructions()
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=PATTERN_RECOGNITION_SYSTEM + "\n\nFormat Instructions:\n" + format_instructions),
                    HumanMessage(content=PATTERN_RECOGNITION_HUMAN.format(content="\n".join([doc.page_content for doc in current_state.input_documents])))
                ])
                
                chain = prompt | self.llm | RunnableLambda(self._parse_pattern)
                
                try:
                    pattern = await chain.ainvoke({})
                    
                    if isinstance(state, dict):
                        state['identified_patterns'] = state.get('identified_patterns', []) + [pattern]
                    else:
                        state.identified_patterns.append(pattern)
                    
                    log_info_with_context(f"Identified pattern: {pattern.pattern_type}", "Knowledge Generation")
                except Exception as e:
                    log_warning_with_context(f"Error detecting patterns: {e}", "Knowledge Generation")
                    # Return default pattern on error
                    pattern = Pattern(
                        pattern_type="error",
                        description="Failed to detect pattern",
                        supporting_evidence=["Error occurred during pattern detection"],
                        confidence=0.0
                    )
                    if isinstance(state, dict):
                        state['identified_patterns'] = state.get('identified_patterns', []) + [pattern]
                    else:
                        state.identified_patterns.append(pattern)
                
                return state
            except Exception as e:
                log_error_with_traceback(e, "Error in recognize_patterns")
                return state

        # Hypothesis generation node
        async def generate_hypotheses(state: Dict) -> Dict:
            """Generate hypotheses from patterns"""
            try:
                log_info_with_context("Starting hypothesis generation", "Knowledge Generation")
                
                if isinstance(state, dict):
                    current_state = SynthesisState(**state)
                else:
                    current_state = state
                
                parser = PydanticOutputParser(pydantic_object=Hypothesis)
                format_instructions = parser.get_format_instructions()
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=HYPOTHESIS_GENERATION_SYSTEM + "\n\nFormat Instructions:\n" + format_instructions),
                    HumanMessage(content=HYPOTHESIS_GENERATION_HUMAN.format(patterns=json.dumps([p.dict() for p in current_state.identified_patterns], indent=2)))
                ])
                
                chain = prompt | self.llm | RunnableLambda(self._parse_hypothesis)
                
                try:
                    hypothesis = await chain.ainvoke({})
                    
                    if isinstance(state, dict):
                        state['generated_hypotheses'] = state.get('generated_hypotheses', []) + [hypothesis]
                    else:
                        state.generated_hypotheses.append(hypothesis)
                    
                    log_info_with_context(f"Generated hypothesis: {hypothesis.statement}", "Knowledge Generation")
                except Exception as e:
                    log_warning_with_context(f"Error generating hypotheses: {e}", "Knowledge Generation")
                    # Return default hypothesis on error
                    hypothesis = Hypothesis(
                        statement="Failed to generate hypothesis",
                        reasoning="Error occurred during hypothesis generation",
                        evidence=["Error occurred"],
                        confidence=0.0,
                        validation_status="failed"
                    )
                    if isinstance(state, dict):
                        state['generated_hypotheses'] = state.get('generated_hypotheses', []) + [hypothesis]
                    else:
                        state.generated_hypotheses.append(hypothesis)
                
                return state
            except Exception as e:
                log_error_with_traceback(e, "Error in generate_hypotheses")
                return state

        # Relationship inference node
        async def infer_relationships(state: Dict) -> Dict:
            """Infer relationships from hypotheses"""
            try:
                log_info_with_context("Starting relationship inference", "Knowledge Generation")
                
                if isinstance(state, dict):
                    current_state = SynthesisState(**state)
                else:
                    current_state = state
                
                parser = PydanticOutputParser(pydantic_object=Relationship)
                format_instructions = parser.get_format_instructions()
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=RELATIONSHIP_INFERENCE_SYSTEM + "\n\nFormat Instructions:\n" + format_instructions),
                    HumanMessage(content=RELATIONSHIP_INFERENCE_HUMAN.format(hypotheses=json.dumps([h.dict() for h in current_state.generated_hypotheses], indent=2)))
                ])
                
                chain = prompt | self.llm | RunnableLambda(self._parse_relationship)
                
                try:
                    relationship = await chain.ainvoke({})
                    
                    if isinstance(state, dict):
                        state['inferred_relationships'] = state.get('inferred_relationships', []) + [relationship]
                    else:
                        state.inferred_relationships.append(relationship)
                    
                    log_info_with_context(f"Inferred relationship: {relationship.relation}", "Knowledge Generation")
                except Exception as e:
                    log_warning_with_context(f"Error inferring relationships: {e}", "Knowledge Generation")
                    # Return default relationship on error
                    relationship = Relationship(
                        source="error",
                        relation="is_a",
                        target="inference_failure",
                        domain="error"
                    )
                    if isinstance(state, dict):
                        state['inferred_relationships'] = state.get('inferred_relationships', []) + [relationship]
                    else:
                        state.inferred_relationships.append(relationship)
                
                return state
            except Exception as e:
                log_error_with_traceback(e, "Error in infer_relationships")
                return state

        # Knowledge synthesis node
        async def synthesize_knowledge(state: Dict) -> Dict:
            """Synthesize knowledge from the current state"""
            try:
                log_info_with_context("Starting knowledge synthesis", "Knowledge Generation")
                
                if isinstance(state, dict):
                    current_state = SynthesisState(**state)
                else:
                    current_state = state
                
                parser = PydanticOutputParser(pydantic_object=SyntheticKnowledge)
                format_instructions = parser.get_format_instructions()
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=KNOWLEDGE_SYNTHESIS_SYSTEM + "\n\nFormat Instructions:\n" + format_instructions),
                    HumanMessage(content=KNOWLEDGE_SYNTHESIS_HUMAN.format(
                        patterns=json.dumps([p.dict() for p in current_state.identified_patterns], indent=2),
                        hypotheses=json.dumps([h.dict() for h in current_state.generated_hypotheses], indent=2),
                        relationships=json.dumps([r.dict() for r in current_state.inferred_relationships], indent=2)
                    ))
                ])
                
                chain = prompt | self.llm | RunnableLambda(self._parse_synthetic_knowledge)
                
                try:
                    knowledge = await chain.ainvoke({})
                    
                    if isinstance(state, dict):
                        state['synthetic_knowledge'] = knowledge
                    else:
                        state.synthetic_knowledge = knowledge
                    
                    log_info_with_context("Generated synthetic knowledge", "Knowledge Generation")
                except Exception as e:
                    log_warning_with_context(f"Error synthesizing knowledge: {e}", "Knowledge Generation")
                    # Return default knowledge on error
                    knowledge = SyntheticKnowledge(
                        content="Error occurred during knowledge synthesis",
                        patterns=[],
                        hypotheses=[],
                        relationships=[],
                        confidence=0.0,
                        validation_status="failed",
                        metadata=SourceMetadata(
                            source_type="text",
                            confidence_score=0.0,
                            domain_relevance=0.0,
                            timestamp=datetime.now().isoformat(),
                            validation_status="failed",
                            domain="error"
                        )
                    )
                    if isinstance(state, dict):
                        state['synthetic_knowledge'] = knowledge
                    else:
                        state.synthetic_knowledge = knowledge
                
                return state
            except Exception as e:
                log_error_with_traceback(e, "Error in synthesize_knowledge")
                return state

        # Create workflow graph
        workflow = StateGraph(SynthesisState)

        # Add nodes
        workflow.add_node("recognize_patterns", recognize_patterns)
        workflow.add_node("generate_hypotheses", generate_hypotheses)
        workflow.add_node("infer_relationships", infer_relationships)
        workflow.add_node("synthesize_knowledge", synthesize_knowledge)

        # Add edges
        workflow.add_edge("recognize_patterns", "generate_hypotheses")
        workflow.add_edge("generate_hypotheses", "infer_relationships")
        workflow.add_edge("infer_relationships", "synthesize_knowledge")

        # Set entry point
        workflow.set_entry_point("recognize_patterns")

        return workflow

    def _parse_pattern(self, response: Any) -> Pattern:
        """Parse LLM response into Pattern object"""
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
                    # If not JSON, try to extract pattern from text
                    data = {
                        "pattern_type": "unknown",
                        "description": text.strip(),
                        "supporting_evidence": [],
                        "confidence": 0.5
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "pattern_type": "unknown",
                    "description": str(data),
                    "supporting_evidence": [],
                    "confidence": 0.5
                }
            
            # Fill in missing fields with defaults
            data.setdefault("pattern_type", "unknown")
            data.setdefault("description", "No description provided")
            data.setdefault("supporting_evidence", [])
            data.setdefault("confidence", 0.5)
            
            # Ensure confidence is a float between 0 and 1
            try:
                data["confidence"] = float(data["confidence"])
            except (ValueError, TypeError):
                data["confidence"] = 0.5
            data["confidence"] = max(0.0, min(1.0, data["confidence"]))
            
            # Create Pattern object
            return Pattern(**data)
            
        except Exception as e:
            logger.error(f"Error parsing pattern: {e}")
            return Pattern(
                pattern_type="error",
                description="Error parsing pattern",
                supporting_evidence=["Error occurred"],
                confidence=0.0
            )

    def _parse_hypothesis(self, response: Any) -> Hypothesis:
        """Parse LLM response into Hypothesis object"""
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
                    # If not JSON, try to extract hypothesis from text
                    data = {
                        "statement": text.strip(),
                        "reasoning": "Extracted from non-JSON response",
                        "evidence": [],
                        "confidence": 0.5,
                        "validation_status": "pending"
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "statement": str(data),
                    "reasoning": "Non-dictionary response",
                    "evidence": [],
                    "confidence": 0.5,
                    "validation_status": "pending"
                }
            
            # Fill in missing fields with defaults
            data.setdefault("statement", "No statement provided")
            data.setdefault("reasoning", "No reasoning provided")
            data.setdefault("evidence", [])
            data.setdefault("confidence", 0.5)
            data.setdefault("validation_status", "pending")
            
            # Ensure confidence is a float between 0 and 1
            try:
                data["confidence"] = float(data["confidence"])
            except (ValueError, TypeError):
                data["confidence"] = 0.5
            data["confidence"] = max(0.0, min(1.0, data["confidence"]))
            
            # Ensure validation_status is valid
            if data["validation_status"] not in ["pending", "processed", "failed"]:
                data["validation_status"] = "pending"
            
            # Create Hypothesis object
            return Hypothesis(**data)
            
        except Exception as e:
            logger.error(f"Error parsing hypothesis: {e}")
            return Hypothesis(
                statement="Error parsing hypothesis",
                reasoning="Error occurred",
                evidence=["Error occurred"],
                confidence=0.0,
                validation_status="failed"
            )

    def _parse_relationship(self, response: Any) -> Relationship:
        """Parse LLM response into Relationship object"""
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
                    # If not JSON, try to extract relationship from text
                    data = {
                        "source": "unknown",
                        "relation": "related_to",
                        "target": text.strip(),
                        "domain": "knowledge"
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "source": "unknown",
                    "relation": "related_to",
                    "target": str(data),
                    "domain": "knowledge"
                }
            
            # Fill in missing fields with defaults
            data.setdefault("source", "unknown")
            data.setdefault("relation", "related_to")
            data.setdefault("target", "unknown")
            data.setdefault("domain", "knowledge")
            
            # Ensure relation is valid
            if not isinstance(data["relation"], str) or data["relation"] not in ["is_a", "has_part", "related_to"]:
                data["relation"] = "related_to"
            
            # Create Relationship object
            return Relationship(
                source=data["source"],
                relation=cast(Literal["is_a", "has_part", "related_to"], data["relation"]),
                target=data["target"],
                domain=data["domain"]
            )
            
        except Exception as e:
            logger.error(f"Error parsing relationship: {e}")
            return Relationship(
                source="error",
                relation="is_a",
                target="parsing_failure",
                domain="error"
            )

    def _parse_synthetic_knowledge(self, response: Any) -> SyntheticKnowledge:
        """Parse LLM response into SyntheticKnowledge object"""
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
                    # If not JSON, try to extract knowledge from text
                    data = {
                        "content": text.strip(),
                        "patterns": [],
                        "hypotheses": [],
                        "relationships": [],
                        "confidence": 0.5,
                        "validation_status": "pending",
                        "metadata": {
                            "source_type": "text",
                            "confidence_score": 0.5,
                            "domain_relevance": 0.5,
                            "timestamp": datetime.now().isoformat(),
                            "validation_status": "pending",
                            "domain": "knowledge"
                        }
                    }
            else:
                data = text
            
            # Ensure we have a dict
            if not isinstance(data, dict):
                data = {
                    "content": str(data),
                    "patterns": [],
                    "hypotheses": [],
                    "relationships": [],
                    "confidence": 0.5,
                    "validation_status": "pending",
                    "metadata": {
                        "source_type": "text",
                        "confidence_score": 0.5,
                        "domain_relevance": 0.5,
                        "timestamp": datetime.now().isoformat(),
                        "validation_status": "pending",
                        "domain": "knowledge"
                    }
                }
            
            # Fill in missing fields with defaults
            data.setdefault("content", "No content provided")
            data.setdefault("patterns", [])
            data.setdefault("hypotheses", [])
            data.setdefault("relationships", [])
            data.setdefault("confidence", 0.5)
            data.setdefault("validation_status", "pending")
            data.setdefault("metadata", {
                "source_type": "text",
                "confidence_score": data["confidence"],
                "domain_relevance": 0.5,
                "timestamp": datetime.now().isoformat(),
                "validation_status": data["validation_status"],
                "domain": "knowledge"
            })
            
            # Ensure confidence is a float between 0 and 1
            try:
                data["confidence"] = float(data["confidence"])
            except (ValueError, TypeError):
                data["confidence"] = 0.5
            data["confidence"] = max(0.0, min(1.0, data["confidence"]))
            
            # Ensure validation_status is valid
            if data["validation_status"] not in ["pending", "processed", "failed"]:
                data["validation_status"] = "pending"
            
            # Parse nested objects
            if isinstance(data["patterns"], list):
                data["patterns"] = [self._parse_pattern(p) for p in data["patterns"]]
            else:
                data["patterns"] = []
            
            if isinstance(data["hypotheses"], list):
                data["hypotheses"] = [self._parse_hypothesis(h) for h in data["hypotheses"]]
            else:
                data["hypotheses"] = []
            
            if isinstance(data["relationships"], list):
                data["relationships"] = [self._parse_relationship(r) for r in data["relationships"]]
            else:
                data["relationships"] = []
            
            # Parse metadata
            if isinstance(data["metadata"], dict):
                metadata = data["metadata"]
                metadata.setdefault("source_type", "text")
                metadata.setdefault("confidence_score", data["confidence"])
                metadata.setdefault("domain_relevance", 0.5)
                metadata.setdefault("timestamp", datetime.now().isoformat())
                metadata.setdefault("validation_status", data["validation_status"])
                metadata.setdefault("domain", "knowledge")
                
                # Ensure metadata values are valid
                if metadata["source_type"] not in ["text", "pdf", "web"]:
                    metadata["source_type"] = "text"
                if metadata["validation_status"] not in ["pending", "processed", "failed"]:
                    metadata["validation_status"] = data["validation_status"]
                try:
                    metadata["confidence_score"] = float(metadata["confidence_score"])
                except (ValueError, TypeError):
                    metadata["confidence_score"] = 0.5
                metadata["confidence_score"] = max(0.0, min(1.0, metadata["confidence_score"]))
                try:
                    metadata["domain_relevance"] = float(metadata["domain_relevance"])
                except (ValueError, TypeError):
                    metadata["domain_relevance"] = 0.5
                metadata["domain_relevance"] = max(0.0, min(1.0, metadata["domain_relevance"]))
                
                data["metadata"] = SourceMetadata(**metadata)
            else:
                data["metadata"] = SourceMetadata(
                    source_type="text",
                    confidence_score=data["confidence"],
                    domain_relevance=0.5,
                    timestamp=datetime.now().isoformat(),
                    validation_status=data["validation_status"],
                    domain="knowledge"
                )
            
            # Create SyntheticKnowledge object
            return SyntheticKnowledge(**data)
            
        except Exception as e:
            logger.error(f"Error parsing synthetic knowledge: {e}")
            return SyntheticKnowledge(
                content="Error parsing synthetic knowledge",
                patterns=[],
                hypotheses=[],
                relationships=[],
                confidence=0.0,
                validation_status="failed",
                metadata=SourceMetadata(
                    source_type="text",
                    confidence_score=0.0,
                    domain_relevance=0.0,
                    timestamp=datetime.now().isoformat(),
                    validation_status="failed",
                    domain="error"
                )
            )

    async def generate_knowledge(self, documents: List[Document]) -> SyntheticKnowledge:
        """Generate synthetic knowledge from documents"""
        try:
            log_info_with_context(f"Starting knowledge generation with {len(documents)} documents", "Knowledge Generation")
            console.print(f"[green]✓[/green] Processing {len(documents)} documents")
            
            # Initialize state
            state = SynthesisState(input_documents=documents, synthetic_knowledge=None)
            
            # Run workflow
            app = self.state_graph.compile()
            result = await app.ainvoke(state)
            
            # Get final knowledge
            if isinstance(result, dict):
                knowledge = result.get('synthetic_knowledge')
            else:
                knowledge = result.synthetic_knowledge
                
            if knowledge is None:
                log_warning_with_context("No synthetic knowledge generated", "Knowledge Generation")
                # Return default knowledge if none was generated
                return SyntheticKnowledge(
                    content="Failed to generate synthetic knowledge",
                    patterns=[],
                    hypotheses=[],
                    relationships=[],
                    confidence=0.0,
                    validation_status="failed",
                    metadata=SourceMetadata(
                        source_type="text",
                        confidence_score=0.0,
                        domain_relevance=0.0,
                        timestamp=datetime.now().isoformat(),
                        validation_status="failed",
                        domain="error"
                    )
                )
            
            # Print detailed stats
            console.print(Panel.fit(
                f"""
                Content Length: {len(knowledge.content)} chars
                Patterns: {len(knowledge.patterns)}
                Hypotheses: {len(knowledge.hypotheses)}
                Relationships: {len(knowledge.relationships)}
                Confidence: {knowledge.confidence:.2f}
                Validation Status: {knowledge.validation_status}
                Domain Relevance: {knowledge.metadata.domain_relevance if knowledge.metadata else 0.0:.2f}
                """,
                title="Knowledge Generation Results"
            ))
                
            return knowledge
            
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge generation")
            raise 