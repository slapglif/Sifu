import asyncio
import aiofiles
import json
import os
import sys
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv
from loguru import logger
from scripts.logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    setup_logging,
    create_progress
)
from langchain_core.globals import set_debug
from langchain.output_parsers import PydanticOutputParser
from prompts.compiler.compiler_prompts import get_join_decision_prompt, get_task_execution_prompt
from pathlib import Path
from rich.layout import Layout
from rich.spinner import Spinner
from rich.text import Text
from rich.box import Box
from rich.syntax import Syntax
set_debug(False)
# Load environment variables
load_dotenv(override=True)

# Initialize logging
setup_logging()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Ensure required environment variables
if not os.getenv("OLLAMA_HOST"):
    raise EnvironmentError("OLLAMA_HOST environment variable must be set")

from scripts.knowledge_acquisition import KnowledgeAcquisitionSystem
from scripts.models import KnowledgeAcquisitionConfig, ExtractedKnowledge
from scripts.qa_system import QASystem
from scripts.synthetic_knowledge import SyntheticKnowledgeGenerator
from scripts.lora_training import LoRATrainer, LoRATrainingConfig, TrainingExample
from scripts.llm_compiler import LLMCompiler, Task, Plan, TaskResult, JoinDecision, CompilerState

console = Console()

class SystemState(BaseModel):
    """Overall system state"""
    domain_name: str = Field(description="Current domain name")
    knowledge_sources: List[Document] = Field(default_factory=list)
    generated_questions: List[Dict] = Field(default_factory=list)
    synthetic_knowledge: List[Dict] = Field(default_factory=list)
    training_examples: List[TrainingExample] = Field(default_factory=list)
    model_metrics: Dict = Field(default_factory=dict)

def print_state_summary(state: SystemState):
    """Print a summary of the current system state"""
    table = Table(title="System State Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Details", style="yellow")
    
    table.add_row(
        "Knowledge Sources",
        str(len(state.knowledge_sources)),
        f"Documents loaded and processed"
    )
    table.add_row(
        "Generated Questions", 
        str(len(state.generated_questions)),
        f"Questions generated from sources"
    )
    table.add_row(
        "Synthetic Knowledge",
        str(len(state.synthetic_knowledge)),
        f"Pieces of synthesized knowledge"
    )
    table.add_row(
        "Training Examples",
        str(len(state.training_examples)),
        f"Examples prepared for training"
    )
    
    if state.model_metrics:
        metrics_str = "\n".join(f"{k}: {v}" for k, v in state.model_metrics.items())
        table.add_row("Model Metrics", "✓", metrics_str)
    else:
        table.add_row("Model Metrics", "✗", "No metrics available")
        
    console.print(table)

class ResearchAgent(LLMCompiler):
    """Research agent using LLMCompiler pattern."""
    
    def __init__(self, config_path: str):
        """Initialize the research agent."""
        # Initialize LLM first
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "MFDoom/deepseek-r1-tool-calling:1.5b"),
            format="json",
            temperature=0.7,
            mirostat=2,
            mirostat_eta=0.1,
            mirostat_tau=5.0
        )
        super().__init__(llm)
        
        # Initialize other attributes
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.state = SystemState(domain_name="default")  # Initialize with default
        self.graph: Optional[Neo4jGraph] = None
        self.knowledge_system: Optional[KnowledgeAcquisitionSystem] = None
        self.qa_system: Optional[QASystem] = None
        self.knowledge_generator: Optional[SyntheticKnowledgeGenerator] = None
        self.lora_trainer: Optional[LoRATrainer] = None
        
        log_info_with_context("Research agent initialized", "Research")
        console.print(Panel("[bold green]Research Agent Initialized[/bold green]"))

    def _log_knowledge_stats(self):
        """Log current knowledge statistics"""
        table = Table(title="[bold]Knowledge Statistics[/bold]", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Knowledge Sources", str(len(self.state.knowledge_sources)))
        table.add_row("Generated Questions", str(len(self.state.generated_questions)))
        table.add_row("Synthetic Knowledge", str(len(self.state.synthetic_knowledge)))
        table.add_row("Training Examples", str(len(self.state.training_examples)))
        
        console.print(table)

    def _log_model_metrics(self):
        """Log model training metrics"""
        if self.state.model_metrics:
            table = Table(title="[bold]Model Metrics[/bold]", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in self.state.model_metrics.items():
                table.add_row(str(metric), str(value))
            
            console.print(table)
        else:
            console.print("[yellow]No model metrics available yet[/yellow]")

    async def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            log_info_with_context(f"Loading configuration from {config_path}", "Config")
            async with aiofiles.open(config_path) as f:
                content = await f.read()
                config = json.loads(content)
                console.print(Panel(f"[green]Configuration loaded from {config_path}[/green]"))
                return config
        except Exception as e:
            log_error_with_traceback(e, "Failed to load configuration")
            raise
            
    async def _init_neo4j(self) -> Neo4jGraph:
        """Initialize Neo4j connection"""
        try:
            if not self.config or "neo4j" not in self.config:
                raise ValueError("Neo4j configuration not found")
                
            log_info_with_context("Initializing Neo4j connection", "Database")    
            graph = Neo4jGraph(
                url=self.config["neo4j"]["url"],
                username=self.config["neo4j"]["username"],
                password=self.config["neo4j"]["password"]
            )
            console.print("[green]Neo4j connection established successfully[/green]")
            return graph
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize Neo4j")
            raise

    async def initialize(self) -> "ResearchAgent":
        """Async initialization of all components"""
        try:
            log_info_with_context("Starting initialization", "Research Agent")
            console.print("\n[bold cyan]Initializing Research Agent Components...[/bold cyan]")
            
            # Create progress tracking
            progress = create_progress()
            init_progress = progress.add_task("[cyan]Initialization...", total=5)
            
            # Load config first
            self.config = await self._load_config(self.config_path)
            progress.update(init_progress, advance=1, description="[cyan]Loading configuration...")
            
            # Initialize state
            self.state = SystemState(domain_name=self.config["domain_name"])
            progress.update(init_progress, advance=1, description="[cyan]Initializing state...")
            
            # Initialize core components
            self.graph = await self._init_neo4j()
            progress.update(init_progress, advance=1, description="[cyan]Connecting to Neo4j...")
            
            # Initialize subsystems
            if "knowledge_acquisition" not in self.config:
                raise ValueError("Knowledge acquisition configuration not found")
                
            self.knowledge_system = KnowledgeAcquisitionSystem(
                KnowledgeAcquisitionConfig(**self.config["knowledge_acquisition"])
            )
            await self.knowledge_system.initialize()
            progress.update(init_progress, advance=1, description="[cyan]Setting up knowledge system...")
            
            # Initialize QA and knowledge generator with the same graph type
            if self.graph:
                # Use Any type to bypass Neo4jGraph type mismatch
                graph: Any = self.graph
                self.qa_system = QASystem(graph, self.llm)
                self.knowledge_generator = SyntheticKnowledgeGenerator(graph, self.llm)
            
            if "lora_training" not in self.config:
                raise ValueError("LoRA training configuration not found")
                
            self.lora_trainer = LoRATrainer(
                LoRATrainingConfig(**self.config["lora_training"])
            )
            progress.update(init_progress, advance=1, description="[cyan]Initializing training system...")
            
            log_info_with_context("Initialization complete", "Research Agent")
            console.print(Panel("[bold green]Research Agent Initialization Complete[/bold green]"))
            
            # Log initial state
            self._log_knowledge_stats()
            return self
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize research agent")
            raise
        
    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate research plan."""
        try:
            # Create plan for research workflow
            tasks = []
            task_idx = 0
            
            # Research task
            tasks.append(Task(
                idx=task_idx,
                tool="research_topics",
                args={"domain": self.state.domain_name},  # Use the actual domain name
                dependencies=[]
            ))
            task_idx += 1
            
            # Knowledge synthesis task
            tasks.append(Task(
                idx=task_idx,
                tool="synthesize_knowledge",
                args={"sources": []},  # Will be filled from research results
                dependencies=[task_idx - 1]
            ))
            task_idx += 1
            
            # Training example generation task
            tasks.append(Task(
                idx=task_idx,
                tool="generate_examples",
                args={"knowledge": []},  # Will be filled from synthesis results
                dependencies=[task_idx - 1]
            ))
            task_idx += 1
            
            # Model training task
            tasks.append(Task(
                idx=task_idx,
                tool="train_model",
                args={"examples": []},  # Will be filled from example generation
                dependencies=[task_idx - 1]
            ))
            
            return Plan(
                tasks=tasks,
                thought=f"Generated plan for research workflow in domain '{self.state.domain_name}': research topics, synthesize knowledge, generate examples, and train model"
            )
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating research plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute planned tasks."""
        try:
            log_info_with_context(f"Starting execution of {len(tasks)} tasks", "Execution")
            console.print("\n[bold yellow]Executing Tasks...[/bold yellow]")
            
            results = []
            task_results = {}  # Store results by task ID
            
            # Create progress tracking
            progress = create_progress()
            task_progress = progress.add_task(
                "[yellow]Task Execution[/yellow]",
                total=len(tasks)
            )
            progress.update(task_progress, description="Executing tasks...")
            
            for task in tasks:
                try:
                    # Check dependencies
                    deps_met = all(
                        task_id in task_results and not task_results[task_id].error
                        for task_id in task.dependencies
                    )
                    if not deps_met:
                        log_warning_with_context(f"Dependencies not met for task {task.idx}", "Execution")
                        continue

                    # Update task args with results from dependencies
                    updated_args = task.args.copy()
                    if task.tool == "synthesize_knowledge" and task.dependencies:
                        # Get sources from research task
                        research_result = task_results[task.dependencies[0]]
                        if research_result and research_result.result:
                            updated_args["sources"] = research_result.result.get("knowledge_sources", [])
                    elif task.tool == "generate_examples" and task.dependencies:
                        # Get knowledge from synthesis task
                        synthesis_result = task_results[task.dependencies[0]]
                        if synthesis_result and synthesis_result.result:
                            updated_args["knowledge"] = synthesis_result.result.get("synthetic_knowledge", [])
                    elif task.tool == "train_model" and task.dependencies:
                        # Get examples from example generation task
                        examples_result = task_results[task.dependencies[0]]
                        if examples_result and examples_result.result:
                            updated_args["examples"] = examples_result.result.get("training_examples", [])

                    # Execute task with updated args
                    log_info_with_context(f"Executing task {task.idx}: {task.tool}", "Execution")
                    result = None
                    
                    if task.tool == "research_topics":
                        result = await self._research_topics(updated_args["domain"])
                    elif task.tool == "synthesize_knowledge":
                        result = await self._synthesize_knowledge(updated_args["sources"])
                    elif task.tool == "generate_examples":
                        result = await self._generate_examples(updated_args["knowledge"])
                    elif task.tool == "train_model":
                        result = await self._train_model(updated_args["examples"])
                    
                    task_result = TaskResult(
                        task_id=task.idx,
                        result=result,
                        error=None
                    )
                    
                    # Store result
                    results.append(task_result)
                    task_results[task.idx] = task_result
                    
                    # Log result
                    self._log_task_result(task_result)
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Error executing task {task.idx}")
                    task_result = TaskResult(
                        task_id=task.idx,
                        result=None,
                        error=str(e)
                    )
                    results.append(task_result)
                    task_results[task.idx] = task_result
                    self._log_task_result(task_result)
                
                progress.update(task_progress, advance=1)
                    
            # Ensure progress is cleared
            progress.stop()
            return results
            
        except Exception as e:
            log_error_with_traceback(e, "Error executing tasks")
            raise

    def _log_task_result(self, result: TaskResult):
        """Log task result in a rich format"""
        if result.error:
            console.print(Panel(
                f"[red]Error:[/red] {result.error}",
                title=f"[bold red]Task {result.task_id} Failed[/bold red]",
                border_style="red"
            ))
        else:
            # Only print result if it's not None and has content
            if result.result and (
                isinstance(result.result, dict) and any(result.result.values()) or
                isinstance(result.result, list) and result.result or
                not isinstance(result.result, (dict, list))
            ):
                table = Table(title=f"[bold green]Task {result.task_id} Result[/bold green]", box=box.ROUNDED)
                
                if isinstance(result.result, dict):
                    table.add_column("Key", style="cyan")
                    table.add_column("Value", style="green")
                    for key, value in result.result.items():
                        if isinstance(value, list):
                            table.add_row(str(key), f"{len(value)} items")
                        else:
                            table.add_row(str(key), str(value))
                elif isinstance(result.result, list):
                    table.add_column("Index", style="cyan")
                    table.add_column("Value", style="green")
                    for idx, value in enumerate(result.result):
                        table.add_row(str(idx), str(value))
                else:
                    table.add_column("Result", style="green")
                    table.add_row(str(result.result))
                
                console.print(table)
            else:
                console.print(f"[green]✓ Task {result.task_id} completed[/green]")

    async def _make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Decide whether to complete or replan"""
        try:
            # Format state for LLM
            formatted_state = {
                "plan": state.plan.model_dump() if state.plan else None,
                "results": [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in state.results],
                "current_progress": {
                    "knowledge_sources": len(self.state.knowledge_sources),
                    "synthetic_knowledge": len(self.state.synthetic_knowledge),
                    "training_examples": len(self.state.training_examples),
                    "has_metrics": bool(self.state.model_metrics)
                }
            }

            # Get decision from LLM using join decision prompt
            prompt = get_join_decision_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=JoinDecision)
            result = await chain.ainvoke({
                "state": json.dumps(formatted_state, indent=2)
            })
            return result

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> SystemState:
        """Generate final system state."""
        try:
            # Update state with results
            for result in state.results:
                if result and result.result:
                    if isinstance(result.result, dict):
                        if 'knowledge_sources' in result.result:
                            self.state.knowledge_sources.extend(result.result['knowledge_sources'])
                        if 'synthetic_knowledge' in result.result:
                            self.state.synthetic_knowledge.extend(result.result['synthetic_knowledge'])
                        if 'training_examples' in result.result:
                            self.state.training_examples.extend(result.result['training_examples'])
                        if 'model_metrics' in result.result:
                            self.state.model_metrics.update(result.result['model_metrics'])
            
            return self.state

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def _research_topics(self, domain: str) -> Dict[str, Any]:
        """Research topics from sources"""
        try:
            if not self.knowledge_system:
                raise ValueError("Knowledge system not initialized")
                
            log_info_with_context(f"Starting topic research for domain: {domain}", "Research")
            console.print(f"\n[bold cyan]Researching Topics for Domain: {domain}[/bold cyan]")
            
            all_docs = []
            total_sources = len(self.config.get("knowledge_sources", []))
            progress = create_progress()
            
            # Add research task
            research_task = progress.add_task(
                "[cyan]Processing Sources[/cyan]",
                total=total_sources
            )
            progress.update(research_task, description="Processing knowledge sources...")
            
            # Process each knowledge source
            if "knowledge_sources" not in self.config:
                log_warning_with_context("No knowledge sources configured", "Research")
                return {"knowledge_sources": []}
            
            for idx, source in enumerate(self.config["knowledge_sources"]):
                try:
                    log_info_with_context(f"Processing source {idx+1}/{total_sources}: {source['path']}", "Research")
                    console.print(f"[cyan]Processing source {idx+1}/{total_sources}:[/cyan] {source['path']}")
                    
                    # Initial knowledge extraction
                    docs = await self.knowledge_system.add_source(
                        source["path"],
                        source["type"]
                    )
                    if docs:
                        all_docs.extend(docs)
                        log_info_with_context(f"Extracted {len(docs)} documents from source", "Research")
                        console.print(f"[green]✓ Extracted {len(docs)} documents[/green]")
                        
                        # Use vector store to find related knowledge
                        for doc in docs:
                            try:
                                # Search for related content
                                if self.knowledge_system.vector_store:
                                    similar_docs = await asyncio.to_thread(
                                        self.knowledge_system.vector_store.similarity_search,
                                        doc.page_content,
                                        k=5
                                    )
                                    
                                    # Add unique related documents
                                    added = 0
                                    for similar_doc in similar_docs:
                                        if similar_doc not in all_docs:
                                            all_docs.append(similar_doc)
                                            added += 1
                                    if added > 0:
                                        log_info_with_context(f"Found {added} related documents", "Research")
                                        console.print(f"[green]✓ Found {added} related documents[/green]")
                                
                                if self.qa_system:
                                    # Generate follow-up questions
                                    questions = await self.qa_system.generate_questions(
                                        topic=doc.metadata.get("topic", "general"),
                                        num_questions=5
                                    )
                                    
                                    if questions:
                                        log_info_with_context(f"Generated {len(questions)} follow-up questions", "Research")
                                        console.print(f"[green]✓ Generated {len(questions)} follow-up questions[/green]")
                                        
                                        # Store questions
                                        self.state.generated_questions.extend([
                                            {
                                                "question": q.question,
                                                "source": doc.metadata.get("source", "unknown"),
                                                "topic": doc.metadata.get("topic", "general")
                                            }
                                            for q in questions
                                        ])
                                        
                                        # Research each question
                                        question_progress = create_progress()
                                        question_task = question_progress.add_task(
                                            "[yellow]Researching questions...",
                                            total=len(questions)
                                        )
                                        
                                        for question in questions:
                                            try:
                                                answer = await self.qa_system.process_question(question.question)
                                                if answer and answer.confidence > 0.7:
                                                    # Add new knowledge to stores
                                                    new_doc = Document(
                                                        page_content=answer.answer,
                                                        metadata={
                                                            "source": "qa_research",
                                                            "question": question.question,
                                                            "confidence": answer.confidence,
                                                            "topic": doc.metadata.get("topic", "general"),
                                                            "domain": domain
                                                        }
                                                    )
                                                    all_docs.append(new_doc)
                                                    log_info_with_context(f"Added answer with confidence {answer.confidence}", "Research")
                                                    console.print(f"[green]✓ Added answer with {answer.confidence:.2f} confidence[/green]")
                                                    
                                                    # Search for related content to the answer
                                                    if self.knowledge_system.vector_store:
                                                        answer_related = await asyncio.to_thread(
                                                            self.knowledge_system.vector_store.similarity_search,
                                                            answer.answer,
                                                            k=5
                                                        )
                                                        added = 0
                                                        for rel_doc in answer_related:
                                                            if rel_doc not in all_docs:
                                                                all_docs.append(rel_doc)
                                                                added += 1
                                                        if added > 0:
                                                            log_info_with_context(f"Found {added} documents related to answer", "Research")
                                                            console.print(f"[green]✓ Found {added} related documents[/green]")
                                                            
                                                question_progress.update(question_task, advance=1)
                                            except Exception as e:
                                                log_error_with_traceback(e, f"Error researching question: {question.question}")
                                                console.print(f"[red]✗ Failed to research question: {question.question}[/red]")
                                                continue
                            except Exception as e:
                                log_error_with_traceback(e, f"Error generating questions from document: {e}")
                                console.print("[red]✗ Failed to generate questions from document[/red]")
                                continue
                    
                    # Update progress
                    progress.update(research_task, advance=1)
                    progress.refresh()
                except Exception as e:
                    log_error_with_traceback(e, f"Error processing source {source['path']}")
                    console.print(f"[red]✗ Failed to process source: {source['path']}[/red]")
                    continue
                
            log_info_with_context(f"Research completed with {len(all_docs)} total documents", "Research")
            console.print(Panel(f"[bold green]Research Complete[/bold green]\nTotal Documents: {len(all_docs)}"))
            
            # Save documents to state
            self.state.knowledge_sources.extend(all_docs)
            
            return {"knowledge_sources": all_docs}
                
        except Exception as e:
            log_error_with_traceback(e, "Error in research topics")
            raise

    async def _synthesize_knowledge(self, sources: List[Document]) -> Dict[str, Any]:
        """Synthesize knowledge from sources"""
        try:
            if not self.knowledge_generator:
                raise ValueError("Knowledge generator not initialized")
                
            log_info_with_context("Starting knowledge synthesis", "Synthesis")
            console.print("\n[bold cyan]Synthesizing Knowledge...[/bold cyan]")
                
            if not sources:
                log_warning_with_context("No sources available for synthesis", "Synthesis")
                console.print("[yellow]Warning: No sources available for synthesis[/yellow]")
                return {"synthetic_knowledge": []}
                
            # Create progress tracking
            progress = create_progress()
            synthesis_task = progress.add_task(
                "[cyan]Synthesizing Knowledge[/cyan]",
                total=len(sources)
            )
            progress.update(synthesis_task, description="Processing sources...")
            
            all_knowledge = []
            
            # Process each source
            for source in sources:
                try:
                    log_info_with_context(f"Processing source: {source.metadata.get('source', 'unknown')}", "Synthesis")
                    
                    # Generate synthetic knowledge
                    knowledge = await self.knowledge_generator.generate_knowledge([source])
                    if knowledge:
                        log_info_with_context("Successfully generated synthetic knowledge", "Synthesis")
                        
                        # Add metadata
                        knowledge_dict = knowledge.model_dump()
                        knowledge_dict["source"] = source.metadata.get("source", "unknown")
                        knowledge_dict["topic"] = source.metadata.get("topic", "general")
                        knowledge_dict["domain"] = source.metadata.get("domain", "unknown")
                        
                        # Log synthesis details
                        table = Table(title="[bold]Synthesis Results[/bold]", box=box.ROUNDED)
                        table.add_column("Category", style="cyan")
                        table.add_column("Count", style="green")
                        
                        table.add_row("Patterns", str(len(knowledge.patterns)))
                        table.add_row("Hypotheses", str(len(knowledge.hypotheses)))
                        table.add_row("Relationships", str(len(knowledge.relationships)))
                        
                        console.print(table)
                        
                        # Add to results
                        all_knowledge.append(knowledge_dict)
                        
                except Exception as e:
                    log_error_with_traceback(e, f"Error processing source: {source.metadata.get('source', 'unknown')}")
                    console.print(f"[red]✗ Failed to process source[/red]")
                    continue
                finally:
                    progress.update(synthesis_task, advance=1)
                    
            # Log final results
            log_info_with_context(f"Generated {len(all_knowledge)} pieces of synthetic knowledge", "Synthesis")
            console.print(Panel(f"[bold green]Synthesis Complete[/bold green]\nTotal Knowledge: {len(all_knowledge)}"))
            
            # Save knowledge to state
            self.state.synthetic_knowledge.extend(all_knowledge)
            
            return {"synthetic_knowledge": all_knowledge}
                
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge synthesis")
            raise

    async def _generate_examples(self, knowledge: List[Dict]) -> Dict[str, Any]:
        """Generate training examples from knowledge"""
        try:
            log_info_with_context("Starting example generation", "Training Data")
            console.print("\n[bold cyan]Generating Training Examples...[/bold cyan]")
            
            if not knowledge:
                log_warning_with_context("No knowledge available for example generation", "Training Data")
                console.print("[yellow]Warning: No knowledge available for example generation[/yellow]")
                return {"training_examples": []}
            
            examples = []
                
            # Create progress tracking
            progress = create_progress()
            example_task = progress.add_task(
                "[cyan]Generating Examples[/cyan]",
                total=len(knowledge)
            )
            progress.update(example_task, description="Processing knowledge items...")
                
                # Generate from synthetic knowledge
            for k in knowledge:
                try:
                    log_info_with_context(f"Processing knowledge from {k.get('source', 'unknown')}", "Training Data")
                    
                        # Generate examples for pattern recognition
                    for pattern in k.get("patterns", []):
                            examples.append(TrainingExample(
                                input_text=f"Identify patterns in this context: {pattern['supporting_evidence']}",
                                output_text=pattern["description"],
                            metadata={
                                "type": "pattern_recognition",
                                "source": k.get("source", "unknown"),
                                "topic": k.get("topic", "general"),
                                "domain": k.get("domain", "unknown")
                            },
                                quality_score=pattern["confidence"]
                            ))
                            
                            # Generate examples for hypothesis validation
                        for hypothesis in k.get("hypotheses", []):
                                examples.append(TrainingExample(
                                    input_text=f"Validate this hypothesis: {hypothesis['statement']}\nEvidence: {hypothesis['evidence']}",
                                    output_text=hypothesis["reasoning"],
                                metadata={
                                    "type": "hypothesis_validation",
                                    "source": k.get("source", "unknown"),
                                    "topic": k.get("topic", "general"),
                                    "domain": k.get("domain", "unknown")
                                },
                                    quality_score=hypothesis["confidence"]
                                ))
                                
                        # Generate examples for relationship inference
                        for relationship in k.get("relationships", []):
                            examples.append(TrainingExample(
                                input_text=f"Infer relationship: {relationship['source']} -> {relationship['target']}",
                                output_text=relationship["relation"],
                                metadata={
                                    "type": "relationship_inference",
                                    "source": k.get("source", "unknown"),
                                    "topic": k.get("topic", "general"),
                                    "domain": k.get("domain", "unknown")
                                },
                                quality_score=k.get("confidence", 0.7)
                            ))
                            
                    progress.update(example_task, advance=1)
                except Exception as e:
                    log_error_with_traceback(e, "Error generating examples from knowledge")
                    console.print("[red]✗ Failed to generate examples from knowledge[/red]")
                        continue
                
                # Filter and validate examples
                valid_examples = []
                for example in examples:
                    if example.quality_score >= 0.7 and len(example.input_text.split()) >= 10:
                        valid_examples.append(example)
                
                log_info_with_context(f"Generated {len(valid_examples)} valid training examples", "Training Data")
            
            # Log example statistics
            table = Table(title="[bold]Training Examples[/bold]", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Examples", str(len(examples)))
            table.add_row("Valid Examples", str(len(valid_examples)))
            table.add_row("Average Quality", f"{sum(e.quality_score for e in valid_examples) / len(valid_examples):.2f}" if valid_examples else "N/A")
            
            # Log example types
            example_types = {}
            for example in valid_examples:
                example_type = example.metadata.get("type", "unknown")
                example_types[example_type] = example_types.get(example_type, 0) + 1
            
            for example_type, count in example_types.items():
                table.add_row(f"{example_type} Examples", str(count))
            
            console.print(table)
            
            # Save examples to state
            self.state.training_examples.extend(valid_examples)
            
            return {"training_examples": valid_examples}
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating examples")
            raise

    async def _train_model(self, examples: List[TrainingExample]) -> Dict[str, Any]:
                """Train LoRA adapter"""
        try:
                if not self.lora_trainer:
                    raise ValueError("LoRA trainer not initialized")
                    
            log_info_with_context("Starting model training", "Training")
            console.print("\n[bold cyan]Training Model...[/bold cyan]")
                
            if not examples:
                    log_warning_with_context("No training examples available", "Training")
                console.print("[yellow]Warning: No training examples available[/yellow]")
                return {"model_metrics": {}}
                
            # Create progress tracking
            progress = create_progress()
            train_task = progress.add_task(
                "[cyan]Training Model[/cyan]",
                total=4
            )
            progress.update(train_task, description="Preparing training data...")
                
                    # Prepare datasets
            console.print("[cyan]Preparing datasets...[/cyan]")
            dataset = self.lora_trainer.prepare_training_data(examples)
                    train_size = int(0.8 * len(dataset))
                    train_dataset = dataset.select(range(train_size))
                    eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # Log dataset statistics
            table = Table(title="[bold]Dataset Statistics[/bold]", box=box.ROUNDED)
            table.add_column("Split", style="cyan")
            table.add_column("Size", style="green")
            
            table.add_row("Training", str(len(train_dataset)))
            table.add_row("Evaluation", str(len(eval_dataset)))
            
            console.print(table)
            progress.update(train_task, advance=1)
                    
                    # Train model
            console.print("[cyan]Training model...[/cyan]")
                    metrics = self.lora_trainer.train(
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset
                    )
            progress.update(train_task, advance=1)
                    
                    # Save adapter
            console.print("[cyan]Saving adapter...[/cyan]")
            adapter_path = f"results/{self.state.domain_name}/lora_adapter"
                    self.lora_trainer.save_adapter(
                adapter_path,
                        "domain_adaptation"
                    )
            progress.update(train_task, advance=1)
            
            # Log training metrics
            if hasattr(metrics, "model_dump"):
                metrics_dict = metrics.model_dump()
            else:
                metrics_dict = dict(metrics)
                
            table = Table(title="[bold]Training Metrics[/bold]", box=box.ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for metric, value in metrics_dict.items():
                table.add_row(str(metric), str(value))
                
            console.print(table)
            console.print(f"[green]✓ Adapter saved to {adapter_path}[/green]")
            progress.update(train_task, advance=1)
            
            # Save metrics to state
            self.state.model_metrics.update(metrics_dict)
            
            return {"model_metrics": metrics_dict}
            
        except Exception as e:
            log_error_with_traceback(e, "Error in model training")
            raise
        
    async def run(self):
        """Run the research agent"""
        try:
            console.print("\n[bold blue]Starting Research Agent[/bold blue]")
            
            # Create progress tracking
            progress = create_progress()
            
            # Add tasks for each major stage
            research_task = progress.add_task(
                "[cyan]Topic Research[/cyan]",
                total=100
            )
            synthesis_task = progress.add_task(
                "[yellow]Knowledge Synthesis[/yellow]",
                total=100
            )
            examples_task = progress.add_task(
                "[green]Training Examples[/green]",
                total=100
            )
            training_task = progress.add_task(
                "[magenta]Model Training[/magenta]",
                total=100
            )
            
            # Update descriptions
            progress.update(research_task, description="Researching topics...")
            progress.update(synthesis_task, description="Synthesizing knowledge...")
            progress.update(examples_task, description="Generating examples...")
            progress.update(training_task, description="Training model...")
                
            # Initialize state
            initial_state = CompilerState(
                content=f"Research domain: {self.state.domain_name}\nKnowledge sources: {len(self.config.get('knowledge_sources', []))} sources\nTask: Generate research plan and execute workflow.",
                plan=None,
                results=[],
                join_decision=None,
                final_result=None
            )
            
            log_info_with_context("Starting research workflow", "Research")
            
            # Run LLM compiler workflow
            result = await super().run(dict(initial_state.model_dump()))
            
            # Update progress based on results
            if result and isinstance(result, SystemState):
                # Research progress
                if result.knowledge_sources:
                    log_info_with_context(
                        f"Processed {len(result.knowledge_sources)} knowledge sources",
                        "Research"
                    )
                    progress.update(research_task, completed=100)
                    
                # Synthesis progress
                if result.synthetic_knowledge:
                    log_info_with_context(
                        f"Generated {len(result.synthetic_knowledge)} pieces of synthetic knowledge",
                        "Synthesis"
                    )
                    progress.update(synthesis_task, completed=100)
                
                # Examples progress
                if result.training_examples:
                    log_info_with_context(
                        f"Created {len(result.training_examples)} training examples",
                        "Examples"
                    )
                    progress.update(examples_task, completed=100)
                
                # Training progress
                if result.model_metrics:
                    log_info_with_context(
                        "Model training completed with metrics",
                        "Training"
                    )
                    progress.update(training_task, completed=100)
                
                # Ensure progress display is cleared
                progress.stop()
                
                    # Log final metrics
                if result.model_metrics:
                        console.print("\n[bold green]Training Metrics[/bold green]")
                        metrics_panel = Panel.fit(
                        "\n".join([f"{k}: {v}" for k, v in result.model_metrics.items()]),
                            title="Results",
                            border_style="green"
                        )
                        console.print(metrics_panel)
                        
                    # Print final state summary
                console.print("\n[bold blue]Final State Summary[/bold blue]")
                print_state_summary(result)
                
                # Save results to files
                try:
                    # Create results directory if it doesn't exist
                    results_dir = Path(f"results/{self.state.domain_name}")
                    results_dir.mkdir(parents=True, exist_ok=True)
                    
                    console.print("\n[bold cyan]Saving Results...[/bold cyan]")
                    
                    # Save synthetic knowledge
                    if result.synthetic_knowledge:
                        with open(results_dir / "synthetic_knowledge.json", "w") as f:
                            json.dump(result.synthetic_knowledge, f, indent=2)
                        console.print("[green]✓ Saved synthetic knowledge[/green]")
                            
                    # Save training examples
                    if result.training_examples:
                        with open(results_dir / "training_examples.json", "w") as f:
                            json.dump([e.model_dump() for e in result.training_examples], f, indent=2)
                        console.print("[green]✓ Saved training examples[/green]")
                            
                    # Save model metrics
                    if result.model_metrics:
                        with open(results_dir / "model_metrics.json", "w") as f:
                            json.dump(result.model_metrics, f, indent=2)
                        console.print("[green]✓ Saved model metrics[/green]")
                            
                    log_info_with_context("Results saved successfully", "Research")
                    console.print(Panel("[bold green]All Results Saved Successfully[/bold green]"))
                    
            except Exception as e:
                    log_error_with_traceback(e, "Error saving results")
                    console.print("[red]✗ Failed to save some results[/red]")
            else:
                log_warning_with_context("No valid results produced", "Research")
                console.print("[yellow]Warning: No valid results produced[/yellow]")
                    
        except Exception as e:
            log_error_with_traceback(e, "Error running research agent")
            raise

async def main():
    """Main entry point"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Run research agent")
        parser.add_argument("--config", required=True, help="Path to config file")
        args = parser.parse_args()
        
        # Initialize logging first
        setup_logging()
        log_info_with_context("Starting research agent", "Main")
        
        try:
        # Initialize and run agent
        log_info_with_context("Initializing research agent", "Main")
        agent = ResearchAgent(args.config)
        await agent.initialize()
        
        log_info_with_context("Starting research agent", "Main")
        await agent.run()
        
    except Exception as e:
        log_error_with_traceback(e, "Fatal error in research agent")
            console.print("[red]Research agent failed with error:[/red]")
            console.print(Panel(str(e), title="Error", border_style="red"))
            sys.exit(1)
            
    except Exception as e:
        # Log any errors that occur during startup
        print(f"CRITICAL ERROR: {str(e)}")  # Fallback if logging not initialized
        if 'logger' in globals():
            log_error_with_traceback(e, "Fatal error during startup")
        sys.exit(1)
    finally:
        # Ensure all logs are written
        if 'logger' in globals():
            logger.complete()

if __name__ == "__main__":
    # Set up argparse at module level
    import argparse
    
    try:
    # Run main with asyncio
    asyncio.run(main())
    except KeyboardInterrupt:
        log_info_with_context("Research agent stopped by user", "Main")
        sys.exit(0)
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")  # Fallback if logging not initialized
        if 'logger' in globals():
            log_error_with_traceback(e, "Fatal error in main")
        sys.exit(1)
