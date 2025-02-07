import asyncio
import aiofiles
import json
import os
import sys
import shutil
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Any
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
set_debug(True)
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

class ResearchAgent:
    def __init__(self, config_path: str):
        """Initialize the research agent"""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}  # Initialize as empty dict
        self.state: Optional[SystemState] = None
        self.graph: Optional[Neo4jGraph] = None
        self.llm: Optional[ChatOllama] = None
        self.knowledge_system: Optional[KnowledgeAcquisitionSystem] = None
        self.qa_system: Optional[QASystem] = None
        self.knowledge_generator: Optional[SyntheticKnowledgeGenerator] = None
        self.lora_trainer: Optional[LoRATrainer] = None
        self.workflow: Optional[StateGraph] = None

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
        
    def _init_llm(self) -> ChatOllama:
        """Initialize the LLM"""
        try:
            if not self.config or "llm" not in self.config:
                raise ValueError("LLM configuration not found")
                
            llm = ChatOllama(
                model=self.config["llm"]["model_name"],
                temperature=self.config["llm"]["temperature"],
                mirostat=2,
                mirostat_eta=0.1,
                mirostat_tau=5.0,
                format="json"
            )
            log_info_with_context("LLM initialized", "Initialization")
            return llm
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize LLM")
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
            self.llm = self._init_llm()
            
            # Initialize subsystems
            if "knowledge_acquisition" not in self.config:
                raise ValueError("Knowledge acquisition configuration not found")
                
            self.knowledge_system = KnowledgeAcquisitionSystem(
                KnowledgeAcquisitionConfig(**self.config["knowledge_acquisition"])
            )
            await self.knowledge_system.initialize()
            
            self.qa_system = QASystem(self.graph, self.llm)
            self.knowledge_generator = SyntheticKnowledgeGenerator(self.graph, self.llm)
            
            if "lora_training" not in self.config:
                raise ValueError("LoRA training configuration not found")
                
            self.lora_trainer = LoRATrainer(
                LoRATrainingConfig(**self.config["lora_training"])
            )
            
            # Create workflow last
            self.workflow = await self._create_workflow()
            
            log_info_with_context("Initialization complete", "Research Agent")
            return self
            
        except Exception as e:
            log_error_with_traceback(e, "Failed to initialize research agent")
            raise
        
    async def _create_workflow(self) -> StateGraph:
        """Create the main workflow"""
        try:
            log_info_with_context("Creating workflow", "Research Agent")
            workflow = StateGraph(SystemState)
            
            # Knowledge acquisition and research node
            async def research_topics(state: Dict[str, Any]) -> Dict[str, Any]:
                """Research topics from sources"""
                if not self.knowledge_system:
                    raise ValueError("Knowledge system not initialized")
                    
                if isinstance(state, SystemState):
                    state = state.model_dump()
                    
                current_state = SystemState(**state)
                
                log_info_with_context(f"Starting research for domain: {current_state.domain_name}", "Research")
                all_docs = []
                
                # Process each knowledge source
                if "knowledge_sources" not in self.config:
                    log_warning_with_context("No knowledge sources configured", "Research")
                    return current_state.model_dump()
                    
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
                            
                            # Use vector search to find related knowledge
                            for doc in docs:
                                try:
                                    # Search for related content
                                    similar_docs = await self.knowledge_system.search(
                                        doc.page_content,
                                        min_confidence=0.7
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
                                                if answer and answer.get("confidence", 0) > 0.7:
                                                    # Add new knowledge to stores
                                                    new_doc = Document(
                                                        page_content=answer["answer"],
                                                        metadata={
                                                            "source": "qa_research",
                                                            "question": question.question,
                                                            "confidence": answer["confidence"]
                                                        }
                                                    )
                                                    all_docs.append(new_doc)
                                                    
                                                    # Search for related content to the answer
                                                    answer_related = await self.knowledge_system.search(
                                                        answer["answer"],
                                                        min_confidence=0.7
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
                
                # Update state with researched documents
                current_state.knowledge_sources = all_docs
                return current_state.model_dump()
                
            workflow.add_node("research_topics", research_topics)
            
            # Knowledge synthesis node
            async def synthesize_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
                """Synthesize knowledge from research"""
                if not self.knowledge_generator:
                    raise ValueError("Knowledge generator not initialized")
                    
                if isinstance(state, SystemState):
                    state = state.model_dump()
                current_state = SystemState(**state)
                
                log_info_with_context("Starting knowledge synthesis", "Synthesis")
                if not current_state.knowledge_sources:
                    log_warning_with_context("No knowledge sources available for synthesis", "Synthesis")
                    return current_state.model_dump()
                
                try:
                    # Generate synthetic knowledge
                    knowledge = await self.knowledge_generator.generate_knowledge(
                        current_state.knowledge_sources
                    )
                    
                    if knowledge:
                        log_info_with_context("Successfully generated synthetic knowledge", "Synthesis")
                        current_state.synthetic_knowledge.append(knowledge.model_dump())
                        
                        # Add synthesized knowledge back to stores
                        new_doc = Document(
                            page_content=knowledge.content,
                            metadata={
                                "source": "synthesis",
                                "confidence": knowledge.confidence,
                                "patterns": [p.model_dump() for p in knowledge.patterns],
                                "hypotheses": [h.model_dump() for h in knowledge.hypotheses]
                            }
                        )
                        current_state.knowledge_sources.append(new_doc)
                        
                except Exception as e:
                    log_error_with_traceback(e, "Error in knowledge synthesis")
                
                return current_state.model_dump()
                
            workflow.add_node("synthesize_knowledge", synthesize_knowledge)
            
            # Training example generation node
            async def generate_training_examples(state: Dict[str, Any]) -> Dict[str, Any]:
                """Generate diverse training examples"""
                if isinstance(state, SystemState):
                    state = state.model_dump()
                current_state = SystemState(**state)
                
                log_info_with_context("Starting training example generation", "Training Data")
                examples = []
                
                # Generate from research questions and answers
                for doc in current_state.knowledge_sources:
                    if doc.metadata.get("source") == "qa_research":
                        try:
                            examples.append(TrainingExample(
                                input_text=doc.metadata["question"],
                                output_text=doc.page_content,
                                metadata={"type": "qa_pair"},
                                quality_score=doc.metadata.get("confidence", 0.7)
                            ))
                        except Exception as e:
                            log_error_with_traceback(e, "Error generating QA example")
                            continue
                
                # Generate from synthetic knowledge
                for knowledge in current_state.synthetic_knowledge:
                    try:
                        # Generate examples for pattern recognition
                        for pattern in knowledge["patterns"]:
                            examples.append(TrainingExample(
                                input_text=f"Identify patterns in this context: {pattern['supporting_evidence']}",
                                output_text=pattern["description"],
                                metadata={"type": "pattern_recognition"},
                                quality_score=pattern["confidence"]
                            ))
                            
                            # Generate examples for hypothesis validation
                            for hypothesis in knowledge["hypotheses"]:
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
                current_state.training_examples.extend(valid_examples)
                return current_state.model_dump()
                
            workflow.add_node("generate_training_examples", generate_training_examples)
            
            # Model training node
            async def train_model(state: Dict[str, Any]) -> Dict[str, Any]:
                """Train LoRA adapter"""
                if not self.lora_trainer:
                    raise ValueError("LoRA trainer not initialized")
                    
                if isinstance(state, SystemState):
                    state = state.model_dump()
                current_state = SystemState(**state)
                
                log_info_with_context("Starting model training", "Training")
                if not current_state.training_examples:
                    log_warning_with_context("No training examples available", "Training")
                    return current_state.model_dump()
                
                try:
                    # Prepare datasets
                    dataset = self.lora_trainer.prepare_training_data(current_state.training_examples)
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
                        f"results/{current_state.domain_name}/lora_adapter",
                        "domain_adaptation"
                    )
                    
                    # Update metrics
                    current_state.model_metrics = metrics.model_dump() if hasattr(metrics, "model_dump") else dict(metrics)
                    log_info_with_context("Model training complete", "Training")
                    
                except Exception as e:
                    log_error_with_traceback(e, "Error during model training")
                
                return current_state.model_dump()
                
            workflow.add_node("train_model", train_model)
            
            # Define edges
            workflow.add_edge("research_topics", "synthesize_knowledge")
            workflow.add_edge("synthesize_knowledge", "generate_training_examples")
            workflow.add_edge("generate_training_examples", "train_model")
            
            # Set entry and end nodes
            workflow.set_entry_point("research_topics")
            workflow.set_finish_point("train_model")
            
            log_info_with_context("Workflow created successfully", "Research Agent")
            return workflow
            
        except Exception as e:
            log_error_with_traceback(e, "Error creating workflow")
            raise
        
    async def run(self):
        """Run the research agent"""
        try:
            console.print("\n[bold blue]Starting Research Agent[/bold blue]")
            
            if not self.state:
                log_warning_with_context("State not initialized", "Research Agent")
                return
                
            # Initialize state
            initial_state = self.state.model_dump()
            
            if not self.workflow:
                log_warning_with_context("Workflow not initialized", "Research Agent")
                return
                
            # Compile and run workflow
            app = self.workflow.compile()
            
            # Get shared progress instance and create task
            progress = create_progress()
            task = progress.add_task("[cyan]Running research agent...", total=None)
            
            try:
                final_state = await app.ainvoke(initial_state)
                
                if final_state and isinstance(final_state, dict):
                    # Log final metrics
                    if "model_metrics" in final_state:
                        console.print("\n[bold green]Training Metrics[/bold green]")
                        metrics_panel = Panel.fit(
                            "\n".join([f"{k}: {v}" for k, v in final_state["model_metrics"].items()]),
                            title="Results",
                            border_style="green"
                        )
                        console.print(metrics_panel)
                        
                    # Print final state summary
                    final_system_state = SystemState(**final_state)
                    print_state_summary(final_system_state)
                    
            except Exception as e:
                log_error_with_traceback(e, "Error in workflow execution")
                raise
                    
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
