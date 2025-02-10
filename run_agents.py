import asyncio
import aiofiles
import json
import os
import sys
from typing import List, Dict, Optional, Any, cast, TypedDict, Tuple
from pydantic import BaseModel, Field, SecretStr
from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv
from scripts.logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    setup_logging,
    create_progress,
    cleanup_progress
)
from langchain_core.globals import set_debug
from langchain.output_parsers import PydanticOutputParser
from prompts.compiler.compiler_prompts import (
    Task, Plan, TaskResult, JoinDecision, CompilerState as BaseCompilerState, get_join_decision_prompt
)
from pathlib import Path
from prompts.knowledge_acquisition.extraction import get_key_terms_prompt, KeyTermsResponse, SourceMetadata
from scripts.text_web_browser import SimpleTextBrowser, web_search
from datetime import datetime
from scripts.chat_langchain import ChatLangChain
from scripts.llm_compiler import LLMCompiler
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from google.auth.credentials import AnonymousCredentials
import uuid
from scripts.models import KnowledgeAcquisitionConfig, ExtractedKnowledge, SourceMetadata
from datasets import Dataset
from scripts.synthetic_knowledge import SyntheticKnowledgeGenerator
from scripts.lora_training import LoRATrainer, LoRATrainingConfig, TrainingExample
from scripts.example_generator import ExampleGenerator
from rich.progress import Progress
import shutil

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

# Set up Google API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable must be set")
os.environ["GOOGLE_API_KEY"] = google_api_key

from scripts.knowledge_acquisition import KnowledgeAcquisitionSystem
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

class CompilerState(BaseCompilerState):
    """Extended compiler state with research-specific fields."""
    knowledge_sources: list
    synthetic_knowledge: list
    training_examples: list
    model_metrics: dict

class ResearchAgent(LLMCompiler):
    """Research agent that uses LLM compiler for execution."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize research agent."""
        # Initialize LLMs
        planning_llm = ChatLangChain(
            model="gemini-1.5-flash",
            temperature=0.7,
            api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
            format="json",
            pydantic_schema=Plan
        )
        
        task_llm = ChatLangChain(
            model="gemini-1.5-flash",
            temperature=0.1,
            api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
            format="json",
            pydantic_schema=TaskResult
        )
        
        # Initialize compiler with planning LLM
        super().__init__(planning_llm)
        
        # Store task execution LLM
        self.task_llm = task_llm
        
        # Store configuration
        self.config = config
        
        # Initialize Neo4j
        self.graph = cast(Any, Neo4jGraph(
            url=config["neo4j"]["url"],
            username=config["neo4j"]["username"],
            password=config["neo4j"]["password"]
        ))
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model='bge-m3', base_url='http://localhost:11434')
        
        # Initialize knowledge acquisition system
        self.knowledge_system = KnowledgeAcquisitionSystem(
            KnowledgeAcquisitionConfig(**config["knowledge_acquisition"])
        )
        
        # Initialize QA system
        self.qa_system = QASystem(self.graph, self.llm)
        
        # Initialize knowledge generator
        self.knowledge_generator = SyntheticKnowledgeGenerator(self.graph, self.llm)
        
        # Initialize example generator
        self.example_generator = ExampleGenerator(config)
        
        # Initialize LoRA trainer
        self.lora_trainer = LoRATrainer(
            LoRATrainingConfig(**config["lora_training"])
        )
        
        # Initialize system state
        self.system_state = {
            "domain_name": config.get("domain_name", "test_domain"),
            "knowledge_sources": [],
            "generated_questions": [],
            "synthetic_knowledge": [],
            "training_examples": [],
            "model_metrics": {}
        }
        
        # Initialize compiler state
        compiler_state = self._get_compiler_state()
        self.state = compiler_state
        
        # Register tools
        self.register_tool("_research_topics", self._research_topics)
        self.register_tool("_synthesize_knowledge", self._synthesize_knowledge)
        self.register_tool("_generate_examples", self._generate_examples)
        self.register_tool("_train_model", self._train_model)
        
        log_info_with_context("Research agent initialized", "Research")
        console.print(Panel("[bold green]Research Agent Initialized[/bold green]"))

    def _get_compiler_state(self) -> CompilerState:
        """Convert SystemState to CompilerState."""
        return CompilerState(
            content="",
            domain_name=self.system_state["domain_name"],
            plan=None,
            results=[],
            join_decision=None,
            final_result=None,
            error=None,
            feedback=None,
            knowledge_sources=list(self.system_state["knowledge_sources"]),
            synthetic_knowledge=list(self.system_state["synthetic_knowledge"]),
            training_examples=list(self.system_state["training_examples"]),
            model_metrics=dict(self.system_state["model_metrics"])
        )

    async def _research_topics(self, domain: str) -> Dict[str, Any]:
        """Research topics for a domain."""
        try:
            log_info_with_context(f"Starting topic research for domain: {domain}", "Research")
            
            # Initialize knowledge acquisition system
            await self.knowledge_system.initialize()
            
            # Perform web searches
            knowledge_sources = []
            search_queries = [
                f"{domain} overview",
                f"{domain} key concepts",
                f"{domain} latest developments",
                f"{domain} research papers"
            ]
            
            # Track all web sources
            web_sources = []
            
            for query in search_queries:
                log_info_with_context(f"Searching for: {query}", "Search")
                results = await web_search(query)
                if results:
                    # Parse the web search results to extract individual sources
                    source_files = [f for f in os.listdir("web") if f.startswith("source_") and f.endswith(".txt")]
                    for source_file in source_files:
                        if source_file not in web_sources:
                            web_sources.append(source_file)
                            source_path = os.path.join("web", source_file)
                            try:
                                with open(source_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    # Split into metadata and content
                                    parts = content.split("---\n")
                                    if len(parts) == 2:
                                        metadata_str, content = parts
                                        # Parse metadata
                                        metadata = {}
                                        for line in metadata_str.strip().split("\n"):
                                            if ": " in line:
                                                key, value = line.split(": ", 1)
                                                metadata[key.lower()] = value
                                        
                                        # Add as knowledge source
                                        knowledge_sources.append({
                                            "content": content.strip(),
                                            "metadata": {
                                                "query": query,
                                                "title": metadata.get("title", ""),
                                                "url": metadata.get("url", ""),
                                                "timestamp": datetime.now().isoformat(),
                                                "source_type": "web",
                                                "source_file": source_file
                                            }
                                        })
                            except Exception as e:
                                log_error_with_traceback(e, f"Error processing source file: {source_file}")
                                continue
            
            # Log research summary
            console.print(Panel(f"""[bold cyan]Research Summary[/bold cyan]
- Domain: {domain}
- Queries performed: {len(search_queries)}
- Sources gathered: {len(knowledge_sources)}
- Total content: {sum(len(source.get('content', '')) for source in knowledge_sources)} chars"""))
            
            return {
                "knowledge_sources": knowledge_sources,
                "thought": f"Successfully gathered {len(knowledge_sources)} knowledge sources from {len(search_queries)} search queries"
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error in topic research")
            raise

    async def _synthesize_knowledge(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize knowledge from sources."""
        try:
            # Initialize progress tracking
            progress = await create_progress()
            synthesis_task = progress.add_task("[cyan]Synthesizing knowledge...", total=len(sources))
            
            # Track synthesized knowledge
            synthetic_knowledge = []
            
            # Process each source
            for source in sources:
                try:
                    # Extract content and metadata
                    content = source.get("content", "").strip()
                    metadata = source.get("metadata", {})
                    
                    if not content:
                        log_warning_with_context("Empty content, skipping source", "Knowledge Synthesis")
                        progress.update(synthesis_task, advance=1)
                        continue
                        
                    # Process source with knowledge system
                    try:
                        knowledge = await self.knowledge_system.process_source(content)
                        if not knowledge:
                            log_warning_with_context("No knowledge extracted, skipping source", "Knowledge Synthesis")
                            progress.update(synthesis_task, advance=1)
                            continue
                            
                        # Add to synthetic knowledge
                        synthetic_knowledge.append(knowledge)
                        
                        # Update progress and log
                        progress.update(synthesis_task, advance=1)
                        console.print(f"[green]✓ Synthesized knowledge from source {metadata.get('source_file', 'unknown')}[/green]")
                        console.print(f"  Title: {metadata.get('title', 'Unknown')}")
                        console.print(f"  Entities found: {len(knowledge.entities)}")
                        console.print(f"  Relationships found: {len(knowledge.relationships)}")
                        console.print(f"  Confidence: {knowledge.confidence:.2f}")
                        
                    except Exception as e:
                        log_error_with_traceback(e, "Error in knowledge extraction")
                        progress.update(synthesis_task, advance=1)
                        continue
                        
                except Exception as e:
                    log_error_with_traceback(e, f"Error synthesizing knowledge from source {source.get('metadata', {}).get('source_file', 'unknown')}")
                    progress.update(synthesis_task, advance=1)
                    continue
            
            # Save synthetic knowledge to disk
            results_dir = os.path.join("results", self.config.get("domain_name", "test_domain"))
            os.makedirs(results_dir, exist_ok=True)
            
            knowledge_path = os.path.join(results_dir, "synthetic_knowledge.json")
            with open(knowledge_path, "w", encoding="utf-8") as f:
                json.dump(
                    [k.model_dump() for k in synthetic_knowledge],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            # Log synthesis summary
            avg_confidence = 0
            total_entities = 0
            total_relationships = 0
            if synthetic_knowledge:
                avg_confidence = sum(k.confidence for k in synthetic_knowledge) / len(synthetic_knowledge)
                total_entities = sum(len(k.entities) for k in synthetic_knowledge)
                total_relationships = sum(len(k.relationships) for k in synthetic_knowledge)
            
            console.print(Panel(f"""[bold cyan]Knowledge Synthesis Summary[/bold cyan]
- Sources processed: {len(sources)}
- Knowledge entries created: {len(synthetic_knowledge)}
- Total entities: {total_entities}
- Total relationships: {total_relationships}
- Average confidence: {avg_confidence:.2f}"""))
            
            # Update state
            self.system_state["synthetic_knowledge"] = synthetic_knowledge
            
            return {
                "synthetic_knowledge": synthetic_knowledge,
                "thought": f"Successfully synthesized knowledge from {len(sources)} sources with {total_entities} entities and {total_relationships} relationships"
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge synthesis")
            return {
                "synthetic_knowledge": [],
                "thought": f"Error synthesizing knowledge: {str(e)}"
            }

    async def _generate_examples(self, knowledge: List[ExtractedKnowledge]) -> Dict[str, Any]:
        """Generate training examples from knowledge."""
        examples = []
        
        with Progress() as progress:
            example_task = progress.add_task("Generating examples...", total=len(knowledge))
            
            # Process each knowledge entry
            for entry in knowledge:
                try:
                    # Extract content and metadata
                    content = entry.content
                    metadata = entry.metadata
                    
                    if not content:
                        log_warning_with_context("Empty content, skipping entry", "Example Generation")
                        progress.update(example_task, advance=1)
                        continue
                        
                    # Generate examples from content
                    examples_result = await self.example_generator.generate_examples(content)
                    if examples_result and examples_result.examples:
                        examples.extend(examples_result.examples)
                        
                    progress.update(example_task, advance=1)
                    
                except Exception as e:
                    log_error_with_traceback(e, "Error generating examples from knowledge entry")
                    progress.update(example_task, advance=1)
                    continue
            
            # Save training examples to disk
            results_dir = os.path.join("results", self.config.get("domain_name", "test_domain"))
            os.makedirs(results_dir, exist_ok=True)
            
            examples_path = os.path.join(results_dir, "training_examples.json")
            with open(examples_path, "w", encoding="utf-8") as f:
                json.dump(
                    [ex.model_dump() for ex in examples],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            
            await cleanup_progress()
            return {
                "examples": examples,
                "thought": f"Successfully generated {len(examples)} examples from {len(knowledge)} knowledge entries"
            }
                
    async def _train_model(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train model on generated examples."""
        try:
            if not examples:
                return {
                    "metrics": {
                        "error": "No training examples provided"
                    },
                    "thought": "Failed to train model: no examples available"
                }
                
            # Initialize progress tracking
            progress = await create_progress()
            
            # Split data
            train_data, eval_data = self._split_data(examples)
            
            # 1. Prepare training data
            prep_task = progress.add_task("[cyan]Preparing training data...", total=100)
            
            # Format examples
            formatted_examples = []
            for i, example in enumerate(train_data):
                try:
                    formatted = await self._format_example(example)
                    if formatted:
                        formatted_examples.append(formatted)
                    progress.update(prep_task, completed=(i * 100) // len(train_data))
                except Exception as e:
                    log_error_with_traceback(e, f"Error formatting example {i}")
                    continue
                
            progress.update(prep_task, completed=100)
            console.print("[green]✓ Training data prepared[/green]")
            
            if not formatted_examples:
                return {
                    "metrics": {
                        "error": "No valid formatted examples"
                    },
                    "thought": "Failed to train model: no valid examples after formatting"
                }
            
            # 2. Configure training
            config_task = progress.add_task("[cyan]Configuring training...", total=100)
            
            # Get LoRA config
            lora_config = await self._get_lora_config(formatted_examples)
            
            progress.update(config_task, completed=100)
            console.print("[green]✓ Training configuration prepared[/green]")
            
            # 3. Train model
            train_task = progress.add_task("[cyan]Training model...", total=100)
            
            # Initialize metrics tracking
            metrics = {
                "loss": [],
                "eval_loss": [],
                "train_samples": len(train_data),
                "eval_samples": len(eval_data),
                "epochs": lora_config.get("num_epochs", 3),
                "learning_rate": lora_config.get("learning_rate", 3e-4),
                "batch_size": lora_config.get("batch_size", 4)
            }
            
            # Train model using LoRA trainer
            try:
                # Convert examples to Datasets
                train_dataset = Dataset.from_dict({
                    "text": [ex["text"] for ex in formatted_examples],
                    "metadata": [ex["metadata"] for ex in formatted_examples]
                })
                eval_dataset = Dataset.from_dict({
                    "text": [ex["text"] for ex in eval_data],
                    "metadata": [ex["metadata"] for ex in eval_data]
                }) if eval_data else None
                
                # Train model
                train_results = self.lora_trainer.train(
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    num_train_epochs=metrics["epochs"],
                    per_device_train_batch_size=metrics["batch_size"],
                    learning_rate=metrics["learning_rate"],
                    weight_decay=0.01,
                    max_grad_norm=1.0
                )
                metrics.update(train_results.model_dump())
                progress.update(train_task, completed=100)
                console.print("[green]✓ Model training completed[/green]")
                
                # Save metrics to disk
                results_dir = os.path.join("results", self.config.get("domain_name", "test_domain"))
                os.makedirs(results_dir, exist_ok=True)
                
                metrics_path = os.path.join(results_dir, "model_metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                
                # Save LoRA adapter
                adapter_dir = os.path.join(results_dir, "lora_adapter")
                os.makedirs(adapter_dir, exist_ok=True)
                
                # Save adapter config
                config_path = os.path.join(adapter_dir, "adapter_config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(lora_config, f, indent=2)
                
                # Copy adapter files
                adapter_files = [
                    "adapter_model.bin",
                    "adapter_config.json",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "vocab.json",
                    "merges.txt"
                ]
                
                for file in adapter_files:
                    src = os.path.join(lora_config["output_dir"], file)
                    if os.path.exists(src):
                        dst = os.path.join(adapter_dir, file)
                        shutil.copy2(src, dst)
                
            except Exception as e:
                log_error_with_traceback(e, "Error during model training")
                metrics["error"] = str(e)
                progress.update(train_task, completed=100)
                console.print("[red]✗ Model training failed[/red]")
            
            await cleanup_progress()
            return {
                "metrics": metrics,
                "thought": f"Successfully trained model on {len(train_data)} examples with {len(eval_data)} validation examples"
            }
                
        except Exception as e:
            log_error_with_traceback(e, "Error in model training")
            await cleanup_progress()
            return {
                "metrics": {
                    "error": str(e)
                },
                "thought": f"Error training model: {str(e)}"
            }

    def _dict_to_compiler_state(self, state_dict: Dict[str, Any]) -> CompilerState:
        """Convert dictionary to CompilerState."""
        return CompilerState(**{
            "domain_name": str(state_dict.get("domain_name", "")),
            "content": str(state_dict.get("content", "")),
            "plan": state_dict.get("plan"),
            "results": list(state_dict.get("results", [])),
            "thought": str(state_dict.get("thought", "")),
            "join_decision": None,
            "final_result": None,
            "error": None,
            "feedback": None,
            "knowledge_sources": [],
            "synthetic_knowledge": [],
            "training_examples": [],
            "model_metrics": {}
        })

    async def run(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Run the research workflow."""
        try:
            # Initialize state if not provided
            if initial_state is None:
                initial_state = {
                    "domain_name": "test_domain",
                    "content": "",
                    "plan": None,
                    "results": [],
                    "thought": "Starting research workflow"
                }
            
            # Convert to CompilerState
            self.state = self._dict_to_compiler_state(initial_state)
            
            log_info_with_context(f"Starting research workflow for domain: {initial_state['domain_name']}", "Research")
            
            # Create execution plan
            tasks = [
                Task(
                    idx=0,
                    tool="_research_topics",  # Use actual method name
                    args={"domain": initial_state['domain_name']},
                    dependencies=[]
                ),
                Task(
                    idx=1,
                    tool="_synthesize_knowledge",  # Use actual method name
                    args={"sources": []},  # Will be filled from research results
                    dependencies=[0]
                ),
                Task(
                    idx=2,
                    tool="_generate_examples",  # Use actual method name
                    args={"knowledge": []},  # Will be filled from synthesis results
                    dependencies=[1]
                ),
                Task(
                    idx=3,
                    tool="_train_model",  # Use actual method name
                    args={"examples": []},  # Will be filled from example generation results
                    dependencies=[2]
                )
            ]
            
            # Create plan
            plan = Plan(tasks=tasks, thought="Execute research workflow in sequence: research -> synthesis -> examples -> training")
            self.state["plan"] = plan
            
            # Execute tasks
            log_info_with_context("Starting execution of 4 tasks", "Execution")
            results = []
            
            for task in tasks:
                try:
                    # Check dependencies
                    deps_met = all(
                        any(r.task_id == dep and not r.error for r in results)
                        for dep in task.dependencies
                    )
                    if not deps_met:
                        log_warning_with_context(f"Dependencies not met for task {task.idx}", "Execution")
                        continue
                    
                    log_info_with_context(f"Executing task {task.idx}: {task.tool}", "Execution")
                    
                    # Update task args based on previous results
                    if task.tool == "_synthesize_knowledge":
                        # Get knowledge sources from research results
                        research_result = next(
                            (r.result for r in results if r.task_id == 0 and not r.error),
                            None
                        )
                        if research_result and "knowledge_sources" in research_result:
                            task.args["sources"] = research_result["knowledge_sources"]
                            log_info_with_context(f"Found {len(task.args['sources'])} sources for synthesis", "Execution")
                    
                    elif task.tool == "_generate_examples":
                        # Get synthetic knowledge from synthesis results
                        synthesis_result = next(
                            (r.result for r in results if r.task_id == 1 and not r.error),
                            None
                        )
                        if synthesis_result and "synthetic_knowledge" in synthesis_result:
                            task.args["knowledge"] = synthesis_result["synthetic_knowledge"]
                            log_info_with_context(f"Found {len(task.args['knowledge'])} knowledge entries for example generation", "Execution")
                    
                    elif task.tool == "_train_model":
                        # Get examples from example generation results
                        examples_result = next(
                            (r.result for r in results if r.task_id == 2 and not r.error),
                            None
                        )
                        if examples_result and "examples" in examples_result:
                            task.args["examples"] = examples_result["examples"]
                            log_info_with_context(f"Found {len(task.args['examples'])} examples for training", "Execution")
                    
                    # Execute task
                    result = await getattr(self, task.tool)(**task.args)
                    
                    # Update system state based on task
                    if task.tool == "_research_topics" and "knowledge_sources" in result:
                        self.system_state["knowledge_sources"] = result["knowledge_sources"]
                    elif task.tool == "_synthesize_knowledge" and "synthetic_knowledge" in result:
                        self.system_state["synthetic_knowledge"] = result["synthetic_knowledge"]
                    elif task.tool == "_generate_examples" and "examples" in result:
                        self.system_state["training_examples"] = result["examples"]
                    elif task.tool == "_train_model" and "metrics" in result:
                        self.system_state["model_metrics"] = result["metrics"]
                    
                    # Add result
                    results.append(TaskResult(
                        task_id=task.idx,
                        result=result,
                        error=None
                    ))
                    
                except Exception as e:
                    log_error_with_traceback(e, f"Error executing task {task.idx}")
                    results.append(TaskResult(
                        task_id=task.idx,
                        result=None,
                        error=str(e)
                    ))
            
            # Update state with results
            self.state["results"] = results
            
            # Make join decision
            log_info_with_context("Making join decision", "Decision")
            join_decision = await self.make_join_decision(self.state)
            self.state["join_decision"] = join_decision
            
            console.print("\nMaking Join Decision...")
            console.print(Panel(f"""[bold]Join Decision[/bold]
Status: {join_decision.complete}
Thought: {join_decision.thought}
Feedback: {join_decision.feedback or 'None'}"""))
            
            # Generate final result
            log_info_with_context("Generating final result", "Compiler")
            final_result = await self._generate_final_result(self.state)
            self.state["final_result"] = final_result
            
            log_info_with_context("Research workflow completed successfully", "Research")
            
        except Exception as e:
            log_error_with_traceback(e, "Error in research workflow")
            raise

    async def _extract_knowledge(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract knowledge from a source."""
        try:
            # Extract key concepts and relationships
            content = source.get("content", "")
            if not content:
                return None
            
            # Use knowledge acquisition system to extract knowledge
            knowledge = await self.knowledge_system.process_source(content)
            if not knowledge:
                return None
            
            # Get entities and relationships from knowledge object
            entities = getattr(knowledge, "entities", []) or []
            knowledge_relationships = getattr(knowledge, "relationships", []) or []
            confidence = getattr(knowledge, "confidence", 0.5)
            
            # Create synthetic knowledge entry
            return {
                "content": content[:500],  # First 500 chars as summary
                "entities": entities,
                "relationships": knowledge_relationships,
                "confidence": confidence,
                "metadata": {
                    "source_type": "synthesis",
                    "timestamp": datetime.now().isoformat(),
                    "source_id": source.get("id", "unknown"),
                    "num_entities": len(entities),
                    "num_relationships": len(knowledge_relationships)
                }
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting knowledge")
            return None

    def _split_data(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into training and evaluation sets."""
        try:
            # If no examples, return empty lists
            if not examples:
                return [], []
            
            # Split into train/eval sets (80/20)
            split_idx = int(len(examples) * 0.8)
            train_data = examples[:split_idx]
            eval_data = examples[split_idx:]
            
            return train_data, eval_data
            
        except Exception as e:
            log_error_with_traceback(e, "Error splitting data")
            return [], []

    async def _format_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format an example for training."""
        try:
            # Validate example has required fields
            if "input_text" in example and "output_text" in example:
                return {
                    "text": f"Input: {example['input_text']}\nOutput: {example['output_text']}",
                    "metadata": {
                        **example.get("metadata", {}),
                        "id": example.get("id", str(uuid.uuid4())),
                        "quality_score": example.get("quality_score", 0.5)
                    }
                }
            return None
            
        except Exception as e:
            log_error_with_traceback(e, "Error formatting example")
            return None

    async def _get_lora_config(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get LoRA configuration for training."""
        try:
            # Get LoRA config from system state
            lora_config = self.config.get("lora_training", {})
            return {
                "model_name": lora_config.get("model_name", "mistral"),
                "r": lora_config.get("r", 8),
                "bias": lora_config.get("bias", "none"),
                "task_type": lora_config.get("task_type", "CAUSAL_LM"),
                "target_modules": lora_config.get("target_modules", ["q_proj", "v_proj"]),
                "inference_mode": lora_config.get("inference_mode", False),
                "lora_alpha": lora_config.get("lora_alpha", 32),
                "lora_dropout": lora_config.get("lora_dropout", 0.1),
                "num_epochs": lora_config.get("num_epochs", 3),
                "learning_rate": lora_config.get("learning_rate", 3e-4),
                "batch_size": lora_config.get("batch_size", 4)
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error getting LoRA config")
            return {}

    async def make_join_decision(self, state: CompilerState) -> JoinDecision:
        """Make join decision based on task results."""
        try:
            log_info_with_context("Making join decision", "Decision")
            console.print("\n[bold green]Making Join Decision...[/bold green]")
            
            # Get task results
            results = state.get("results", [])
            if not results:
                return JoinDecision(
                    complete=False,
                    thought="No task results available",
                    replan=True,
                    feedback="Need to execute tasks first"
                )
                
            # Check each task result
            failed_tasks = []
            successful_tasks = []
            for result in results:
                if result.error:
                    failed_tasks.append(result.task_id)
                elif result.result is not None:
                    successful_tasks.append(result.task_id)
                    
            # Analyze task dependencies
            plan = state.get("plan")
            if plan and plan.tasks:
                tasks_by_id = {task.idx: task for task in plan.tasks}
                
                # Check if any failed task blocks others
                blocking_failures = []
                for task_id in failed_tasks:
                    # Find tasks that depend on this failed task
                    blocked_tasks = [
                        task.idx for task in plan.tasks 
                        if task_id in task.dependencies
                    ]
                    if blocked_tasks:
                        blocking_failures.append((task_id, blocked_tasks))
                
                # If there are blocking failures, we need to replan
                if blocking_failures:
                    feedback = []
                    for failed_id, blocked_ids in blocking_failures:
                        task = tasks_by_id.get(failed_id)
                        if task:
                            feedback.append(f"Task {failed_id} ({task.tool}) failed and blocks tasks {blocked_ids}")
                    return JoinDecision(
                        complete=False,
                        thought=f"Critical task failures blocking dependent tasks",
                        replan=True,
                        feedback="; ".join(feedback)
                    )
                
                # Check if all tasks completed successfully
                all_tasks = set(task.idx for task in plan.tasks)
                if all_tasks.issubset(set(successful_tasks)):
                    return JoinDecision(
                        complete=True,
                        thought="All tasks completed successfully",
                        replan=False,
                        feedback=None
                    )
                
                # Check if remaining tasks can proceed
                remaining_tasks = all_tasks - set(successful_tasks)
                blocked_tasks = set()
                for task_id in remaining_tasks:
                    task = tasks_by_id.get(task_id)
                    if task and any(dep in failed_tasks for dep in task.dependencies):
                        blocked_tasks.add(task_id)
                
                if blocked_tasks:
                    return JoinDecision(
                        complete=False,
                        thought=f"Tasks {blocked_tasks} are blocked by failed dependencies",
                        replan=True,
                        feedback=f"Need to retry failed tasks: {failed_tasks}"
                    )
                
                # If we have remaining tasks but they're not blocked, continue execution
                return JoinDecision(
                    complete=False,
                    thought=f"Tasks {remaining_tasks} still need to be executed",
                    replan=False,
                    feedback=None
                )
            
            # If we have no plan but have results, something went wrong
            return JoinDecision(
                complete=False,
                thought="No execution plan available",
                replan=True,
                feedback="Need to generate execution plan"
            )
            
        except Exception as e:
            log_error_with_traceback(e, "Error in join decision")
            return JoinDecision(
                complete=False,
                thought=f"Error in join decision: {str(e)}",
                replan=True,
                feedback="Error occurred during join decision"
            )

async def main():
    """Main entry point"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Run research agent")
        parser.add_argument("--config", required=True, help="Path to config file")
        args = parser.parse_args()
        
        # Load config
        async with aiofiles.open(args.config) as f:
            config = json.loads(await f.read())
        
        # Initialize and run agent
        log_info_with_context("Initializing research agent", "Main")
        agent = ResearchAgent(config)
        
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
