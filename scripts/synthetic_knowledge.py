from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from rich.console import Console


from prompts.synthetic_knowledge import (
    Pattern,
    Hypothesis,
    Relationship,
    get_pattern_recognition_prompt,
    get_hypothesis_generation_prompt,
    get_relationship_inference_prompt
)
from prompts.synthetic_knowledge.knowledge_synthesis import (
    SyntheticKnowledge,
    SourceMetadata,
    get_knowledge_synthesis_prompt
)
from prompts.synthetic_knowledge.join_decision import get_join_decision_prompt

from scripts.logging_config import (
    log_error_with_traceback,
)
from scripts.llm_compiler import LLMCompiler, Task, Plan, TaskResult, JoinDecision, CompilerState
from prompts.knowledge_acquisition import (
    get_confidence_evaluation_prompt,
    ConfidenceEvaluation,
)

console = Console()

class SynthesisState(BaseModel):
    """Schema for synthesis workflow state"""
    input_documents: List[Document] = Field(description="Input documents")
    identified_patterns: List[Pattern] = Field(default_factory=list, description="Identified patterns")
    generated_hypotheses: List[Hypothesis] = Field(default_factory=list, description="Generated hypotheses")
    inferred_relationships: List[Relationship] = Field(default_factory=list, description="Inferred relationships")
    synthetic_knowledge: Optional[Dict[str, Any]] = Field(None, description="Generated synthetic knowledge")

class SyntheticKnowledgeGenerator(LLMCompiler):
    """Synthetic knowledge generation system."""

    def __init__(self, graph: Neo4jGraph, llm):
        """Initialize with graph database and language model."""
        super().__init__(llm)
        self.graph = graph

    async def generate_knowledge(self, documents: List[Document]) -> SyntheticKnowledge:
        """Generate synthetic knowledge from documents."""
        try:
            # Create initial state
            state = {
                "documents": documents,
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }

            # Run LLM compiler workflow
            result = await self.run(state)
            return result

        except Exception as e:
            log_error_with_traceback(e, "Error generating knowledge")
            raise

    async def _generate_plan(self, state: Dict[str, Any]) -> Plan:
        """Generate execution plan."""
        try:
            tasks = []
            task_idx = 0
            
            # Add pattern recognition tasks
            for doc in state.get("input_documents", []):
                tasks.append(Task(
                    idx=task_idx,
                    tool="recognize_patterns",
                    args={"content": doc.page_content},
                    dependencies=[]
                ))
                task_idx += 1
                
            # Add hypothesis generation task
            tasks.append(Task(
                idx=task_idx,
                tool="generate_hypotheses",
                args={"patterns": []},  # Will be filled from pattern recognition results
                dependencies=[t.idx for t in tasks]  # Depends on all pattern recognition tasks
            ))
            task_idx += 1
            
            # Add relationship inference task
            tasks.append(Task(
                idx=task_idx,
                tool="infer_relationships",
                args={"hypotheses": []},  # Will be filled from hypothesis generation results
                dependencies=[task_idx - 1]  # Depends on hypothesis generation task
            ))
            task_idx += 1
            
            # Add knowledge synthesis task
            tasks.append(Task(
                idx=task_idx,
                tool="synthesize_knowledge",
                args={
                    "patterns": [],  # Will be filled from pattern recognition results
                    "hypotheses": [],  # Will be filled from hypothesis generation results
                    "relationships": []  # Will be filled from relationship inference results
                },
                dependencies=[task_idx - 1]  # Depends on relationship inference task
            ))
            
            return Plan(
                tasks=tasks,
                thought="Generated plan to process documents through pattern recognition, hypothesis generation, relationship inference, and knowledge synthesis"
            )
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute knowledge synthesis tasks"""
        try:
            results = []
            for task in tasks:
                try:
                    # Check dependencies
                    deps_met = all(
                        any(r.task_id == dep and not r.error for r in results)
                        for dep in task.dependencies
                    )
                    if not deps_met:
                        continue

                    # Execute task
                    result = None
                    if task.tool == "recognize_patterns":
                        result = await self._recognize_patterns(task.args["content"])
                    elif task.tool == "generate_hypotheses":
                        result = await self._generate_hypotheses(task.args["patterns"])
                    elif task.tool == "infer_relationships":
                        result = await self._infer_relationships(task.args["hypotheses"])
                    elif task.tool == "synthesize_knowledge":
                        result = await self._synthesize_knowledge(
                            task.args["patterns"],
                            task.args["hypotheses"],
                            task.args["relationships"]
                        )

                    results.append(TaskResult(
                        task_id=task.idx,
                        result=result,
                        error=None
                    ))

                except Exception as e:
                    results.append(TaskResult(
                        task_id=task.idx,
                        result=None,
                        error=str(e)
                    ))

            return results

        except Exception as e:
            log_error_with_traceback(e, "Error executing synthesis tasks")
            raise

    async def _make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Decide whether to complete or replan"""
        try:
            # Create join prompt
            plan_json = "{}"
            plan = state.get('plan')
            if plan is not None:
                plan_json = json.dumps(plan.dict() if hasattr(plan, 'dict') else plan, indent=2)

            results_json = "[]"
            results = state.get('results')
            if results:
                results_json = json.dumps([r.dict() if hasattr(r, 'dict') else r for r in results], indent=2)

            prompt = get_join_decision_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
            decision = await chain.ainvoke({
                "plan": plan_json,
                "results": results_json
            })
            return decision

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> SyntheticKnowledge:
        """Generate final synthetic knowledge"""
        try:
            # Combine results into SyntheticKnowledge
            content = ""
            patterns = []
            hypotheses = []
            relationships = []
            confidence = 0.0
            validation_status = "pending"
            count = 0

            # Extract results from tasks
            for result in state.get('results', []):
                if result and result.result:
                    if isinstance(result.result, dict):
                        if 'content' in result.result:
                            content = result.result['content']
                        if 'patterns' in result.result:
                            patterns.extend([Pattern(**p) for p in result.result['patterns']])
                        if 'hypotheses' in result.result:
                            hypotheses.extend([Hypothesis(**h) for h in result.result['hypotheses']])
                        if 'relationships' in result.result:
                            relationships.extend([Relationship(**r) for r in result.result['relationships']])
                        if 'confidence' in result.result:
                            confidence += float(result.result['confidence'])
                            count += 1

            # Average confidence scores
            if count > 0:
                confidence = confidence / count

            # Create source metadata
            metadata = SourceMetadata(
                source_type="text",
                confidence_score=confidence,
                domain_relevance=confidence,
                timestamp=datetime.now().isoformat(),
                validation_status=validation_status
            )

            return SyntheticKnowledge(
                content=content if content else "No content generated",
                patterns=patterns,
                hypotheses=hypotheses,
                relationships=relationships,
                confidence=confidence,
                validation_status=validation_status,
                metadata=metadata
            )

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def _recognize_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Recognize patterns in content."""
        try:
            prompt = get_pattern_recognition_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Pattern)
            result = await chain.ainvoke({"content": content})
            return result.model_dump()
        except Exception as e:
            log_error_with_traceback(e, "Error recognizing patterns")
            raise

    async def _generate_hypotheses(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses from patterns."""
        try:
            prompt = get_hypothesis_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Hypothesis)
            result = await chain.ainvoke({"patterns": json.dumps(patterns, indent=2)})
            return result.model_dump()
        except Exception as e:
            log_error_with_traceback(e, "Error generating hypotheses")
            raise

    async def _infer_relationships(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer relationships from hypotheses."""
        try:
            prompt = get_relationship_inference_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Relationship)
            result = await chain.ainvoke({"hypotheses": json.dumps(hypotheses, indent=2)})
            return result.model_dump()
        except Exception as e:
            log_error_with_traceback(e, "Error inferring relationships")
            raise

    async def _synthesize_knowledge(
        self,
        patterns: List[Dict[str, Any]],
        hypotheses: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize knowledge from patterns, hypotheses, and relationships."""
        try:
            prompt = get_knowledge_synthesis_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=SyntheticKnowledge)
            result = await chain.ainvoke({
                "patterns": json.dumps(patterns, indent=2),
                "hypotheses": json.dumps(hypotheses, indent=2),
                "relationships": json.dumps(relationships, indent=2)
            })
            # Convert SourceMetadata to dict before returning
            result_dict = result.model_dump()
            if result_dict.get("metadata"):
                result_dict["metadata"] = result_dict["metadata"].model_dump()
            return result_dict
        except Exception as e:
            log_error_with_traceback(e, "Error synthesizing knowledge")
            raise

    async def _evaluate_confidence(self, content: str, entities: List[str], relationships: List[Dict]) -> ConfidenceEvaluation:
        """Evaluate confidence using proper evaluation prompts."""
        try:
            prompt = get_confidence_evaluation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
            result = await chain.ainvoke({
                "content": content,
                "entities": entities,
                "relationships": relationships,
                "source_type": "text"
            })
            return result
        except Exception as e:
            log_error_with_traceback(e, "Error evaluating confidence")
            raise

    async def _process_document(self, document: Document) -> Dict[str, Any]:
        """Process a single document."""
        try:
            # Process content
            patterns = await self._recognize_patterns(document.page_content)
            hypotheses = await self._generate_hypotheses(patterns)
            relationships = await self._infer_relationships(hypotheses)
            
            # Evaluate confidence
            confidence_eval = await self._evaluate_confidence(
                document.page_content,
                [p["pattern_type"] for p in patterns],
                relationships
            )
            
            # Create metadata with evaluated confidence
            metadata = {
                "source_type": "text",
                "confidence_score": confidence_eval.confidence,
                "domain_relevance": confidence_eval.factors.context_relevance,
                "timestamp": datetime.now().isoformat(),
                "validation_status": "pending"
            }
            
            # Synthesize knowledge
            knowledge = await self._synthesize_knowledge(patterns, hypotheses, relationships)
            if knowledge:
                knowledge["metadata"] = metadata
                
            return knowledge

        except Exception as e:
            log_error_with_traceback(e, "Error processing document")
            raise

    async def _execute_task(self, task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task."""
        try:
            # Process task
            result = None
            if task.tool == "recognize_patterns":
                result = await self._recognize_patterns(task.args["content"])
            elif task.tool == "generate_hypotheses":
                result = await self._generate_hypotheses(task.args["patterns"])
            elif task.tool == "infer_relationships":
                result = await self._infer_relationships(task.args["hypotheses"])
            elif task.tool == "synthesize_knowledge":
                result = await self._synthesize_knowledge(
                    task.args["patterns"],
                    task.args["hypotheses"],
                    task.args["relationships"]
                )
                if result:
                    # Evaluate confidence
                    confidence_eval = await self._evaluate_confidence(
                        result.get("content", ""),
                        [p["pattern_type"] for p in result.get("patterns", [])],
                        result.get("relationships", [])
                    )
                    result["metadata"] = {
                        "source_type": "text",
                        "confidence_score": confidence_eval.confidence,
                        "domain_relevance": confidence_eval.factors.context_relevance,
                        "timestamp": datetime.now().isoformat(),
                        "validation_status": "pending"
                    }

            return {"task_id": task.idx, "result": result}

        except Exception as e:
            log_error_with_traceback(e, f"Error executing task {task.tool}")
            return {
                "task_id": task.idx,
                "result": None,
                "error": str(e)
            }

    async def _handle_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during execution."""
        try:
            # Create error state with zero confidence
            error_state = state.copy()
            error_state["error"] = str(error)
            error_state["error_metadata"] = {
                "source_type": "text",
                "confidence_score": 0.0,
                "domain_relevance": 0.0,
                "timestamp": datetime.now().isoformat(),
                "validation_status": "failed"
            }
            error_state["status"] = "failed"

            return error_state

        except Exception as e:
            log_error_with_traceback(e, "Error handling error")
            raise

    async def _join_task_results(self, task_results: List[TaskResult], state: Dict[str, Any]) -> JoinDecision:
        """Join task results and decide next steps."""
        try:
            # Process results
            complete = True
            thought = "All tasks completed successfully"
            replan = False
            feedback = None

            for result in task_results:
                if result.error:
                    complete = False
                    thought = f"Task {result.task_id} failed: {result.error}"
                    replan = True
                    feedback = f"Task {result.task_id} needs to be retried"
                    break

                if result.result is None:
                    complete = False
                    thought = f"Task {result.task_id} returned no result"
                    replan = True
                    feedback = f"Task {result.task_id} needs to be retried"
                    break

                # Evaluate confidence for knowledge synthesis results
                if result.result.get("synthetic_knowledge"):
                    confidence_eval = await self._evaluate_confidence(
                        result.result["synthetic_knowledge"].get("content", ""),
                        [p["pattern_type"] for p in result.result["synthetic_knowledge"].get("patterns", [])],
                        result.result["synthetic_knowledge"].get("relationships", [])
                    )
                    result.result["synthetic_knowledge"]["metadata"] = {
                        "source_type": "text",
                        "confidence_score": confidence_eval.confidence,
                        "domain_relevance": confidence_eval.factors.context_relevance,
                        "timestamp": datetime.now().isoformat(),
                        "validation_status": "pending"
                    }

            return JoinDecision(
                complete=complete,
                thought=thought,
                replan=replan,
                feedback=feedback
            )

        except Exception as e:
            log_error_with_traceback(e, "Error joining task results")
            return JoinDecision(
                complete=False,
                thought=f"Error joining results: {str(e)}",
                replan=True,
                feedback="Error occurred during join, retry tasks"
            )

    async def _update_state(self, task_results: List[TaskResult], state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with task results."""
        try:
            # Process results
            new_state = state.copy()
            
            for result in task_results:
                if result.error or result.result is None:
                    continue
                    
                if "patterns" in result.result:
                    new_state["identified_patterns"] = result.result["patterns"]
                elif "hypotheses" in result.result:
                    new_state["generated_hypotheses"] = result.result["hypotheses"]
                elif "relationships" in result.result:
                    new_state["inferred_relationships"] = result.result["relationships"]
                elif "synthetic_knowledge" in result.result:
                    # Evaluate confidence
                    confidence_eval = await self._evaluate_confidence(
                        result.result["synthetic_knowledge"].get("content", ""),
                        [p["pattern_type"] for p in result.result["synthetic_knowledge"].get("patterns", [])],
                        result.result["synthetic_knowledge"].get("relationships", [])
                    )
                    result.result["synthetic_knowledge"]["metadata"] = {
                        "source_type": "text",
                        "confidence_score": confidence_eval.confidence,
                        "domain_relevance": confidence_eval.factors.context_relevance,
                        "timestamp": datetime.now().isoformat(),
                        "validation_status": "pending"
                    }
                    new_state["synthetic_knowledge"] = result.result["synthetic_knowledge"]
            
            return new_state

        except Exception as e:
            log_error_with_traceback(e, "Error updating state")
            raise

    async def _process_results(self, task_results: List[TaskResult]) -> Dict[str, Any]:
        """Process task results."""
        try:
            # Process results
            patterns = []
            hypotheses = []
            relationships = []
            synthetic_knowledge = None

            for result in task_results:
                if result.error or result.result is None:
                    continue

                if "patterns" in result.result:
                    patterns.extend(result.result["patterns"])
                elif "hypotheses" in result.result:
                    hypotheses.extend(result.result["hypotheses"])
                elif "relationships" in result.result:
                    relationships.extend(result.result["relationships"])
                elif "synthetic_knowledge" in result.result:
                    # Evaluate confidence
                    confidence_eval = await self._evaluate_confidence(
                        result.result["synthetic_knowledge"].get("content", ""),
                        [p["pattern_type"] for p in result.result["synthetic_knowledge"].get("patterns", [])],
                        result.result["synthetic_knowledge"].get("relationships", [])
                    )
                    result.result["synthetic_knowledge"]["metadata"] = {
                        "source_type": "text",
                        "confidence_score": confidence_eval.confidence,
                        "domain_relevance": confidence_eval.factors.context_relevance,
                        "timestamp": datetime.now().isoformat(),
                        "validation_status": "pending"
                    }
                    synthetic_knowledge = result.result["synthetic_knowledge"]

            return {
                "patterns": patterns,
                "hypotheses": hypotheses,
                "relationships": relationships,
                "synthetic_knowledge": synthetic_knowledge
            }

        except Exception as e:
            log_error_with_traceback(e, "Error processing results")
            raise

    async def _validate_results(self, task_results: List[TaskResult]) -> Dict[str, Any]:
        """Validate task results."""
        try:
            validation_errors = []
            validation_warnings = []

            for result in task_results:
                if result.error:
                    validation_errors.append(f"Task {result.task_id} failed: {result.error}")
                    continue

                if result.result is None:
                    validation_warnings.append(f"Task {result.task_id} returned no result")
                    continue

                # Validate patterns
                if "patterns" in result.result:
                    if not isinstance(result.result["patterns"], list):
                        validation_errors.append(f"Task {result.task_id}: patterns must be a list")
                    else:
                        for pattern in result.result["patterns"]:
                            if not all(k in pattern for k in ["pattern_type", "description", "supporting_evidence", "confidence"]):
                                validation_errors.append(f"Task {result.task_id}: invalid pattern format")

                # Validate hypotheses
                elif "hypotheses" in result.result:
                    if not isinstance(result.result["hypotheses"], list):
                        validation_errors.append(f"Task {result.task_id}: hypotheses must be a list")
                    else:
                        for hypothesis in result.result["hypotheses"]:
                            if not all(k in hypothesis for k in ["statement", "reasoning", "evidence", "confidence", "validation_status"]):
                                validation_errors.append(f"Task {result.task_id}: invalid hypothesis format")

                # Validate relationships
                elif "relationships" in result.result:
                    if not isinstance(result.result["relationships"], list):
                        validation_errors.append(f"Task {result.task_id}: relationships must be a list")
                    else:
                        for relationship in result.result["relationships"]:
                            if not all(k in relationship for k in ["source", "relation", "target"]):
                                validation_errors.append(f"Task {result.task_id}: invalid relationship format")

                # Validate synthetic knowledge
                elif "synthetic_knowledge" in result.result:
                    knowledge = result.result["synthetic_knowledge"]
                    if not all(k in knowledge for k in ["content", "patterns", "hypotheses", "relationships", "confidence", "validation_status"]):
                        validation_errors.append(f"Task {result.task_id}: invalid synthetic knowledge format")
                    else:
                        # Evaluate confidence for synthetic knowledge
                        confidence_eval = await self._evaluate_confidence(
                            knowledge.get("content", ""),
                            [p["pattern_type"] for p in knowledge.get("patterns", [])],
                            knowledge.get("relationships", [])
                        )
                        knowledge["metadata"] = {
                            "source_type": "text",
                            "confidence_score": confidence_eval.confidence,
                            "domain_relevance": confidence_eval.factors.context_relevance,
                            "timestamp": datetime.now().isoformat(),
                            "validation_status": "pending"
                        }

            return {
                "is_valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "warnings": validation_warnings
            }

        except Exception as e:
            log_error_with_traceback(e, "Error validating results")
            raise 