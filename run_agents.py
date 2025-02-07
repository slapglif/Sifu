import asyncio
import aiofiles
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, cast, Union
from datetime import datetime
import torch
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from dotenv import load_dotenv
from scripts.logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    setup_logging,
    create_progress
)
from langchain_core.globals import set_debug
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

    async def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            async with aiofiles.open(config_path) as f:
                content = await f.read()
                config = json.loads(content)
                log_info_with_context(f"Loaded configuration from {config_path}", "Initialization")
                return config
        except Exception as e:
            log_error_with_traceback(e, "Failed to load configuration")
            raise
            
    async def _init_neo4j(self) -> Neo4jGraph:
        """Initialize Neo4j connection"""
        try:
            if not self.config or "neo4j" not in self.config:
                raise ValueError("Neo4j configuration not found")
                
            graph = Neo4jGraph(
                url=self.config["neo4j"]["url"],
                username=self.config["neo4j"]["username"],
                password=self.config["neo4j"]["password"]
            )
            log_info_with_context("Neo4j connection initialized", "Initialization")
            return graph
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize Neo4j")
            raise

    async def initialize(self) -> "ResearchAgent":
        """Async initialization of all components"""
        try:
            log_info_with_context("Starting initialization", "Research Agent")
            
            # Load config first
            self.config = await self._load_config(self.config_path)
            
            # Initialize state
            self.state = SystemState(domain_name=self.config["domain_name"])
            
            # Initialize core components
            self.graph = await self._init_neo4j()
            
            # Initialize subsystems
            if "knowledge_acquisition" not in self.config:
                raise ValueError("Knowledge acquisition configuration not found")
                
            self.knowledge_system = KnowledgeAcquisitionSystem(
                KnowledgeAcquisitionConfig(**self.config["knowledge_acquisition"])
            )
            await self.knowledge_system.initialize()
            
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
            
            log_info_with_context("Initialization complete", "Research Agent")
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
                args={"domain": state.get("domain_name", "")},
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
                thought="Generated plan for research workflow including topic research, knowledge synthesis, example generation, and model training"
            )
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating research plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute research tasks."""
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
                    if task.tool == "research_topics":
                        result = await self._research_topics(task.args["domain"])
                    elif task.tool == "synthesize_knowledge":
                        result = await self._synthesize_knowledge(task.args["sources"])
                    elif task.tool == "generate_examples":
                        result = await self._generate_examples(task.args["knowledge"])
                    elif task.tool == "train_model":
                        result = await self._train_model(task.args["examples"])

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
            log_error_with_traceback(e, "Error executing research tasks")
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

            # Simple decision logic
            complete = True
            thought = "All tasks completed successfully"
            replan = False
            feedback = None

            for result in results:
                if result.error:
                    complete = False
                    thought = f"Task {result.task_id} failed: {result.error}"
                    replan = True
                    feedback = f"Task {result.task_id} needs to be retried"
                    break

            return JoinDecision(
                complete=complete,
                thought=thought,
                replan=replan,
                feedback=feedback
            )

        except Exception as e:
            log_error_with_traceback(e, "Error making join decision")
            raise

    async def _generate_final_result(self, state: CompilerState) -> SystemState:
        """Generate final system state."""
        try:
            # Update state with results
            for result in state.get('results', []):
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
                
            all_docs = []
            
            # Process each knowledge source
            if "knowledge_sources" not in self.config:
                log_warning_with_context("No knowledge sources configured", "Research")
                return {"knowledge_sources": []}
                
            for source in self.config["knowledge_sources"]:
                try:
                    log_info_with_context(f"Processing source: {source['path']}", "Research")
                    # Initial knowledge extraction
                    docs = await self.knowledge_system.add_source(
                        source["path"],
                        source["type"]
                    )
                    if docs:
                        all_docs.extend(docs)
                        
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
                                    for similar_doc in similar_docs:
                                        if similar_doc not in all_docs:
                                            all_docs.append(similar_doc)
                                            
                                if self.qa_system:
                                    # Generate follow-up questions
                                    questions = await self.qa_system.generate_questions(
                                        topic=doc.metadata.get("topic", "general"),
                                        num_questions=5
                                    )
                                    
                                    # Research each question
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
                                                        "confidence": answer.confidence
                                                    }
                                                )
                                                all_docs.append(new_doc)
                                                
                                                # Search for related content to the answer
                                                if self.knowledge_system.vector_store:
                                                    answer_related = await asyncio.to_thread(
                                                        self.knowledge_system.vector_store.similarity_search,
                                                        answer.answer,
                                                        k=5
                                                    )
                                                    for rel_doc in answer_related:
                                                        if rel_doc not in all_docs:
                                                            all_docs.append(rel_doc)
                                                            
                                        except Exception as e:
                                            log_error_with_traceback(e, f"Error researching question: {question.question}")
                                            continue
                                            
                            except Exception as e:
                                log_error_with_traceback(e, f"Error generating questions from document: {e}")
                                continue
                                
                except Exception as e:
                    log_error_with_traceback(e, f"Error processing source {source['path']}")
                    continue
                    
            return {"knowledge_sources": all_docs}
            
        except Exception as e:
            log_error_with_traceback(e, "Error in research topics")
            raise

    async def _synthesize_knowledge(self, sources: List[Document]) -> Dict[str, Any]:
        """Synthesize knowledge from sources"""
        try:
            if not self.knowledge_generator:
                raise ValueError("Knowledge generator not initialized")
                
            if not sources:
                log_warning_with_context("No sources available for synthesis", "Synthesis")
                return {"synthetic_knowledge": []}
                
            # Generate synthetic knowledge
            knowledge = await self.knowledge_generator.generate_knowledge(sources)
            if knowledge:
                log_info_with_context("Successfully generated synthetic knowledge", "Synthesis")
                return {"synthetic_knowledge": [knowledge.model_dump()]}
                
            return {"synthetic_knowledge": []}
            
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge synthesis")
            raise

    async def _generate_examples(self, knowledge: List[Dict]) -> Dict[str, Any]:
        """Generate training examples from knowledge"""
        try:
            examples = []
            
            # Generate from synthetic knowledge
            for k in knowledge:
                try:
                    # Generate examples for pattern recognition
                    for pattern in k.get("patterns", []):
                        examples.append(TrainingExample(
                            input_text=f"Identify patterns in this context: {pattern['supporting_evidence']}",
                            output_text=pattern["description"],
                            metadata={"type": "pattern_recognition"},
                            quality_score=pattern["confidence"]
                        ))
                        
                        # Generate examples for hypothesis validation
                        for hypothesis in k.get("hypotheses", []):
                            examples.append(TrainingExample(
                                input_text=f"Validate this hypothesis: {hypothesis['statement']}\nEvidence: {hypothesis['evidence']}",
                                output_text=hypothesis["reasoning"],
                                metadata={"type": "hypothesis_validation"},
                                quality_score=hypothesis["confidence"]
                            ))
                            
                except Exception as e:
                    log_error_with_traceback(e, "Error generating synthesis example")
                    continue
                    
            # Filter and validate examples
            valid_examples = []
            for example in examples:
                if example.quality_score >= 0.7 and len(example.input_text.split()) >= 10:
                    valid_examples.append(example)
                    
            log_info_with_context(f"Generated {len(valid_examples)} valid training examples", "Training Data")
            return {"training_examples": valid_examples}
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating examples")
            raise

    async def _train_model(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Train LoRA adapter"""
        try:
            if not self.lora_trainer:
                raise ValueError("LoRA trainer not initialized")
                
            if not examples:
                log_warning_with_context("No training examples available", "Training")
                return {"model_metrics": {}}
                
            # Prepare datasets
            dataset = self.lora_trainer.prepare_training_data(examples)
            train_size = int(0.8 * len(dataset))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # Train model
            metrics = self.lora_trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            
            # Save adapter
            self.lora_trainer.save_adapter(
                f"results/{self.state.domain_name}/lora_adapter",
                "domain_adaptation"
            )
            
            # Update metrics
            return {"model_metrics": metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)}
            
        except Exception as e:
            log_error_with_traceback(e, "Error in model training")
            raise

    async def run(self):
        """Run the research agent"""
        try:
            console.print("\n[bold blue]Starting Research Agent[/bold blue]")
            
            # Initialize state
            initial_state = {
                "domain_name": self.state.domain_name,
                "plan": None,
                "results": [],
                "join_decision": None,
                "final_result": None
            }
            
            # Run LLM compiler workflow
            result = await super().run(initial_state)
            if result and isinstance(result, SystemState):
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
                print_state_summary(result)
                
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
        
        # Initialize and run agent
        log_info_with_context("Initializing research agent", "Main")
        agent = ResearchAgent(args.config)
        await agent.initialize()
        
        log_info_with_context("Starting research agent", "Main")
        await agent.run()
        
    except Exception as e:
        log_error_with_traceback(e, "Fatal error in research agent")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argparse at module level
    import argparse
    
    # Run main with asyncio
    asyncio.run(main())
