from typing import List, Dict, Optional, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.graphs import Neo4jGraph
from loguru import logger
from langchain_ollama import ChatOllama
import json
import os
from rich.console import Console
from scripts.logging_config import log_error_with_traceback, log_warning_with_context
from scripts.llm_compiler import LLMCompiler, Task, Plan, TaskResult, JoinDecision, CompilerState
from scripts.chat_langchain import ChatLangChain
from pydantic import SecretStr
from langchain_core.language_models.chat_models import BaseChatModel

from prompts.qa import (
    Question,
    get_question_generation_prompt,
    Answer,
    get_answer_generation_prompt,
    KnowledgeGap,
    get_knowledge_gap_prompt,
    QAResponse,
    get_qa_prompt
)
from prompts.qa.qa_planning import get_qa_plan_prompt
from prompts.qa.join_decision import get_join_decision_prompt

# Initialize console
console = Console()

class QASystem(LLMCompiler):
    """Question answering system."""

    def __init__(self, graph: Neo4jGraph, llm: Optional[BaseChatModel] = None, model: str = "smallthinker", temperature: float = 0.7):
        """Initialize with graph database and language model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable must be set")
            
        llm = llm if llm is not None else ChatLangChain(
            api_key=SecretStr(api_key),
            model="gemini-1.5-flash",
            temperature=temperature,
            pydantic_schema=QAResponse
        )
        super().__init__(llm)
        self.graph = graph
        self.current_topic = ""
        self.current_context = ""
        self.num_questions = 5

    async def generate_questions(self, topic: str, num_questions: int = 5) -> List[Question]:
        """Generate questions about a topic."""
        try:
            # Get prompt and parser
            prompt = get_question_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Question)
            
            # Get context from graph
            context = await self._get_topic_context(topic)
            
            # Generate questions
            result = await chain.ainvoke({
                "topic": topic,
                "context": context,
                "num_questions": num_questions
            })
            
            questions = result.questions if hasattr(result, 'questions') else []
            
            # Validate and filter questions
            validated_questions = []
            for question in questions:
                # Validate question quality
                if await self._validate_question(question, context):
                    validated_questions.append(question)
            
            return validated_questions
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating questions")
            return []

    async def _validate_question(self, question: Question, context: str) -> bool:
        """Validate a generated question."""
        try:
            # Check if question is answerable from context
            answer = await self._generate_answer_task(question.question, context)
            if not answer or not answer.get("answer"):
                log_warning_with_context(f"Question not answerable: {question.question}", "QA")
                return False
            
            # Check confidence
            if answer.get("confidence", 0.0) < 0.7:
                log_warning_with_context(f"Low confidence answer: {question.question}", "QA")
                return False
            
            # Check if question is too simple/generic
            if len(question.question.split()) < 5:
                log_warning_with_context(f"Question too short: {question.question}", "QA")
                return False
            
            # Check if question is relevant to topic
            if question.topic.lower() not in question.question.lower():
                log_warning_with_context(f"Question not relevant to topic: {question.question}", "QA")
                return False
            
            # Check question type distribution
            if not hasattr(self, '_question_types'):
                self._question_types = {
                    'general': 0,
                    'factual': 0,
                    'conceptual': 0,
                    'analytical': 0,
                    'error': 0
                }
            
            # Ensure balanced distribution of question types
            current_count = self._question_types.get(question.type, 0)
            max_per_type = max(2, self.num_questions // 3)
            if current_count >= max_per_type:
                log_warning_with_context(f"Too many questions of type {question.type}", "QA")
                return False
            
            self._question_types[question.type] = current_count + 1
            return True
            
        except Exception as e:
            log_error_with_traceback(e, "Error validating question")
            return False

    async def _get_topic_context(self, topic: str) -> str:
        """Get context for a topic from the graph."""
        try:
            results = self.graph.query(f"""
                MATCH (d:Document)
                WHERE d.content CONTAINS $topic
                RETURN d.content as content
                LIMIT 5
            """, {"topic": topic})
            
            return "\n".join([r["content"] for r in results])
            
        except Exception as e:
            log_error_with_traceback(e, "Error getting topic context")
            return ""

    async def process_qa_chain(self, question: str) -> QAResponse:
        """Process a question through the QA chain."""
        try:
            # Get prompt and parser
            prompt = get_qa_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=QAResponse)
            
            # Get context
            context = await self._retrieve_context_task(question)
            
            # Run QA chain
            result = await chain.ainvoke({
                "question": question,
                "context": context.get("context", ""),
                "sources": context.get("sources", [])
            })
            
            return result
            
        except Exception as e:
            log_error_with_traceback(e, "Error in QA chain")
            return QAResponse(
                answer="Error processing question",
                sources=[],
                confidence=0.0,
                reasoning="QA chain failed"
            )

    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate QA plan"""
        try:
            prompt, parser = get_qa_plan_prompt()
            chain = prompt | self.llm | parser
            plan = await chain.ainvoke({"question": state.get('question', '')})
            return plan

        except Exception as e:
            log_error_with_traceback(e, "Error generating QA plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute QA tasks"""
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
                    if task.tool == "retrieve_context":
                        result = await self._retrieve_context_task(task.args["question"])
                    elif task.tool == "generate_answer":
                        result = await self._generate_answer_task(task.args["question"], task.args["context"])
                    elif task.tool == "validate_answer":
                        result = await self._validate_answer_task(task.args["answer"])
                    elif task.tool == "identify_gaps":
                        result = await self._identify_gaps_task(task.args["question"], task.args["answer"])

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
            log_error_with_traceback(e, "Error executing QA tasks")
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

            prompt, parser = get_join_decision_prompt()
            chain = prompt | self.llm | parser
            decision = await chain.ainvoke({
                "plan": plan_json,
                "results": results_json
            })
            return decision

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> Answer:
        """Generate final answer"""
        try:
            # Combine results into Answer
            answer = ""
            sources = []
            confidence = 0.0
            reasoning = ""
            validation_status = "pending"
            count = 0

            # Extract results from tasks
            for result in state.get('results', []):
                if result and result.result:
                    if isinstance(result.result, dict):
                        if 'answer' in result.result:
                            answer = result.result['answer']
                        if 'sources' in result.result:
                            sources.extend(result.result['sources'])
                        if 'confidence' in result.result:
                            confidence += float(result.result['confidence'])
                            count += 1
                        if 'reasoning' in result.result:
                            reasoning = result.result['reasoning']
                        if 'validation_status' in result.result:
                            validation_status = result.result['validation_status']

            # Average confidence scores
            if count > 0:
                confidence = confidence / count

            return Answer(
                answer=answer,
                sources=sources,
                confidence=confidence,
                reasoning=reasoning,
                validation_status=validation_status
            )

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def _retrieve_context_task(self, question: str) -> Dict[str, Any]:
        """Retrieve context from knowledge graph"""
        try:
            results = self.graph.query(f"""
                MATCH (d:Document)
                WHERE d.content CONTAINS $question
                RETURN d.content as content
                LIMIT 5
            """, {"question": question})
            
            context = "\n".join([r["content"] for r in results])
            return {
                "context": context,
                "sources": [r["content"][:100] + "..." for r in results]
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error retrieving context")
            raise

    async def _generate_answer_task(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer using context"""
        try:
            prompt = get_answer_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Answer)
            result = await chain.ainvoke({
                "question": question,
                "context": context
            })
            return result.model_dump()
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating answer")
            raise

    async def _validate_answer_task(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated answer"""
        try:
            # Extract key components
            answer_text = answer.get("answer", "")
            reasoning = answer.get("reasoning", "")
            sources = answer.get("sources", [])
            
            # Initialize confidence factors
            completeness = 0.0  # How complete is the answer
            relevance = 0.0    # How relevant is it to the question
            support = 0.0      # How well is it supported by sources
            coherence = 0.0    # How coherent/well-structured is it
            
            # Check answer completeness
            if answer_text and len(answer_text.split()) >= 25:
                completeness = 0.8
            elif answer_text and len(answer_text.split()) >= 10:
                completeness = 0.5
            
            # Check relevance via reasoning
            if reasoning and len(reasoning.split()) >= 20:
                relevance = 0.8
            elif reasoning:
                relevance = 0.5
            
            # Check source support
            if sources and len(sources) >= 2:
                support = 0.9
            elif sources:
                support = 0.6
            
            # Check coherence
            if answer_text and "." in answer_text and "," in answer_text:
                coherence = 0.8
            elif answer_text and "." in answer_text:
                coherence = 0.5
            
            # Calculate overall confidence
            confidence = (completeness + relevance + support + coherence) / 4
            
            # Determine validation status
            if confidence >= 0.7:
                validation_status = "validated"
            else:
                validation_status = "failed"
                confidence = max(0.0, confidence - 0.2)  # Penalize confidence
            
            return {
                **answer,
                "validation_status": validation_status,
                "confidence": confidence,
                "validation_metrics": {
                    "completeness": completeness,
                    "relevance": relevance,
                    "support": support,
                    "coherence": coherence
                }
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error validating answer")
            raise

    async def _identify_gaps_task(self, question: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Identify knowledge gaps"""
        try:
            prompt = get_knowledge_gap_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=KnowledgeGap)
            result = await chain.ainvoke({
                "question": question,
                "answer": answer.get('answer', ''),
                "confidence": answer.get('confidence', 0.0),
                "context": answer.get('context', '')
            })
            return result.model_dump()
            
        except Exception as e:
            log_error_with_traceback(e, "Error identifying gaps")
            raise

    async def process_question(self, question: str) -> QAResponse:
        """Process a question through the workflow."""
        try:
            # Create initial state
            state = {
                "question": question,
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }

            # Run LLM compiler workflow
            result = await self.run(state)
            return result if isinstance(result, QAResponse) else QAResponse(
                answer=f"Error processing question: No valid result",
                sources=[],
                confidence=0.0,
                reasoning="Processing failed"
            )

        except Exception as e:
            log_error_with_traceback(e, "Error processing question")
            raise 