import asyncio
import aiofiles
import json
import os
import sys
from typing import List, Dict, Optional, Any, cast, TypedDict, Tuple, Protocol, Awaitable, Callable, Union, TypeVar, Coroutine, Sequence
from pydantic import BaseModel, Field, SecretStr, validator, field_validator
from langchain_neo4j import Neo4jGraph
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
    cleanup_progress,
    log_extraction_results,
    console
)
from langchain_core.globals import set_debug
from langchain.output_parsers import PydanticOutputParser
from prompts.compiler.compiler_prompts import (
    Task, Plan, TaskResult, JoinDecision, CompilerState as BaseCompilerState, get_join_decision_prompt
)
from pathlib import Path
from prompts.knowledge_acquisition.extraction import get_key_terms_prompt, KeyTermsResponse, SourceMetadata
from scripts.text_web_browser_fixed import SimpleTextBrowser, web_search
from datetime import datetime
from scripts.chat_langchain import ChatLangChain
from scripts.llm_compiler import LLMCompiler
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from google.auth.credentials import AnonymousCredentials
import uuid
from scripts.models import KnowledgeAcquisitionConfig, ExtractedKnowledge, SourceMetadata, Relationship, DomainConfig, ConfidenceEvaluation, ConfidenceFactors
from datasets import Dataset
from scripts.synthetic_knowledge import SyntheticKnowledgeGenerator
from scripts.lora_training import LoRATrainer, LoRATrainingConfig, TrainingExample
from scripts.example_generator import ExampleGenerator
from rich.progress import Progress
import shutil
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

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

class CompilerError(Exception):
    """Base class for compiler errors"""
    pass

class PlanningError(CompilerError):
    """Error during plan generation"""
    pass

class ExecutionError(CompilerError):
    """Error during task execution"""
    pass

class DecisionError(CompilerError):
    """Error during join decision"""
    pass

class CompilerStateDict(TypedDict):
    """State for compiler workflow."""
    # Required fields
    content: str
    domain_name: str
    results: List[TaskResult]
    knowledge_sources: List[Union[Document, Dict[str, Any]]]
    synthetic_knowledge: List[Dict[str, Any]]
    training_examples: List[Dict[str, Any]]
    model_metrics: Dict[str, Any]
    
    # Optional fields
    error: Optional[str]
    feedback: Optional[str]
    plan: Optional[Plan]
    join_decision: Optional[JoinDecision]

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
        
        # Initialize compiler with planning LLM
        super().__init__(planning_llm)
        
        # Store configuration
        self.config = config
        
        # Initialize Neo4j graph
        self.graph = Neo4jGraph(
            url=config["neo4j"]["url"],
            username=config["neo4j"]["username"],
            password=config["neo4j"]["password"]
        )
        config["graph"] = self.graph
        
        # Initialize state
        initial_state: CompilerStateDict = {
            "content": "",
            "domain_name": config.get("domain_name", ""),
            "plan": None,
            "results": [],
            "join_decision": None,
            "knowledge_sources": [],
            "synthetic_knowledge": [],
            "training_examples": [],
            "model_metrics": {},
            "error": None,
            "feedback": None
        }
        self.state = cast(BaseCompilerState, initial_state)
        
        # Initialize knowledge system
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
        
        # Initialize web browser for search
        self.browser = SimpleTextBrowser()
        
        # Register tools
        self.register_tool("research_topics", self.research_topics)
        self.register_tool("synthesize_knowledge", self.synthesize_knowledge)
        self.register_tool("generate_examples", self.generate_examples)
        self.register_tool("train_model", self.train_model)
        
        log_info_with_context("Research agent initialized", "Research")
        console.print(Panel("[bold green]Research Agent Initialized[/bold green]"))

    async def research_topics(self, domain: str) -> Dict[str, Any]:
        """Research topics in a domain."""
        return await self._research_topics(domain)

    async def synthesize_knowledge(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize knowledge from sources."""
        return await self._synthesize_knowledge(sources)

    async def generate_examples(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Generate examples from knowledge."""
        return await self._generate_examples(knowledge)

    async def train_model(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Train model on examples."""
        return await self._train_model(examples)

    async def _research_topics(self, domain: str) -> Dict[str, Any]:
        """Research topics in a domain."""
        try:
            # Initialize knowledge system with tracking
            log_info_with_context("Initializing knowledge system", "Research")
            await self.knowledge_system.initialize()
            
            # Track embeddings and tokens
            total_tokens = 0
            total_embeddings = 0
            
            # Get sources with rate limiting and tracking
            max_retries = 3
            base_delay = 2.0
            
            log_info_with_context(f"Starting source collection for domain: {domain}", "Research")
            progress = await create_progress()
            source_task = progress.add_task("[cyan]Collecting sources...", total=max_retries)
            
            for attempt in range(max_retries):
                try:
                    sources = await self._get_sources(domain)
                    if sources:
                        break
                    await asyncio.sleep(base_delay * (2 ** attempt))
                    progress.update(source_task, advance=1)
                except Exception as e:
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        if attempt < max_retries - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                    log_error_with_traceback(e, "Error getting sources")
                    return {
                        "knowledge_sources": [],
                        "thought": "Error occurred during research",
                        "metrics": {
                            "total_tokens": total_tokens,
                            "total_embeddings": total_embeddings,
                            "sources_processed": 0,
                            "entities_found": 0,
                            "relationships_created": 0
                        }
                    }
            
            if not sources:
                return {
                    "knowledge_sources": [],
                    "thought": "No sources found",
                    "metrics": {
                        "total_tokens": total_tokens,
                        "total_embeddings": total_embeddings,
                        "sources_processed": 0,
                        "entities_found": 0,
                        "relationships_created": 0
                    }
                }
            
            # Process sources with tracking
            knowledge_sources = []
            total_entities = 0
            total_relationships = 0
            
            process_task = progress.add_task("[cyan]Processing sources...", total=len(sources))
            
            for i, source in enumerate(sources):
                try:
                    # Add delay between processing sources
                    await asyncio.sleep(1.0)
                    
                    # Extract knowledge with tracking
                    log_info_with_context(f"Processing source {i+1}/{len(sources)}", "Research")
                    knowledge = await self._extract_knowledge(source)
                    
                    if knowledge:
                        # Track metrics
                        total_entities += len(knowledge.entities)
                        total_relationships += len(knowledge.relationships)
                        total_embeddings += 1  # Count embeddings generated
                        
                        # Store knowledge with ID
                        knowledge_dict = knowledge.model_dump()
                        knowledge_dict["id"] = i
                        knowledge_sources.append(knowledge_dict)
                        
                        # Update state with knowledge sources
                        self.state["knowledge_sources"] = knowledge_sources
                        
                        # Log progress
                        console.print(f"[green]✓ Processed source {i+1}[/green]")
                        console.print(f"  Entities found: {len(knowledge.entities)}")
                        console.print(f"  Relationships created: {len(knowledge.relationships)}")
                        console.print(f"  Confidence: {knowledge.confidence:.2f}")
                    
                    progress.update(process_task, advance=1)
                    
                except Exception as e:
                    source_metadata = source.get("metadata", {})
                    query = source_metadata.get("query", "unknown")
                    log_error_with_traceback(e, f"Error processing source: {query}")
                    progress.update(process_task, advance=1)
                    continue
            
            # Generate final metrics
            metrics = {
                "total_tokens": total_tokens,
                "total_embeddings": total_embeddings,
                "sources_processed": len(knowledge_sources),
                "entities_found": total_entities,
                "relationships_created": total_relationships
            }
            
            return {
                "knowledge_sources": knowledge_sources,
                "thought": f"Researched {len(knowledge_sources)} sources about {domain}",
                "metrics": metrics
            }
                
        except Exception as e:
            log_error_with_traceback(e, "Error in research topics")
            return {
                "knowledge_sources": [],
                "thought": "Error occurred during research",
                "metrics": {
                    "total_tokens": 0,
                    "total_embeddings": 0,
                    "sources_processed": 0,
                    "entities_found": 0,
                    "relationships_created": 0
                }
            }

    async def _synthesize_knowledge(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize knowledge from sources."""
        try:
            # Initialize progress tracking
            progress = await create_progress()
            synthesis_task = progress.add_task("[cyan]Synthesizing knowledge...", total=len(sources.get("knowledge_sources", [])))
            
            # Track synthesized knowledge
            synthetic_knowledge = []
            
            # Get knowledge sources from input
            knowledge_sources = sources.get("knowledge_sources", [])
            if not knowledge_sources:
                log_warning_with_context("No knowledge sources found in input", "Knowledge Synthesis")
                return {
                    "synthetic_knowledge": [],
                    "thought": "No knowledge sources available for synthesis"
                }
            
            # Process each source
            for source in knowledge_sources:
                try:
                    if isinstance(source, str):
                        # Skip if source is a string
                        log_warning_with_context("Invalid source format (string), skipping", "Knowledge Synthesis")
                        progress.update(synthesis_task, advance=1)
                        continue
                        
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
                            
                        # Add source metadata and ID
                        if isinstance(knowledge, dict):
                            # Get existing metadata
                            knowledge_metadata = knowledge.get("metadata", {})
                            # Convert metadata to dict if it's a Pydantic model
                            if hasattr(knowledge_metadata, "model_dump"):
                                knowledge_metadata = knowledge_metadata.model_dump()
                            elif hasattr(knowledge_metadata, "dict"):
                                knowledge_metadata = knowledge_metadata.dict()
                                
                            # Create new combined metadata
                            combined_metadata = {**knowledge_metadata, **metadata}
                            knowledge["metadata"] = combined_metadata
                            
                            # Add source ID to track relationships
                            knowledge["source_id"] = source.get("id")
                        
                        # Convert relationships to dictionaries
                        if "relationships" in knowledge:
                            knowledge["relationships"] = [
                                {
                                    "source": rel.source,
                                    "relation": rel.relation,
                                    "target": rel.target,
                                    "domain": rel.domain,
                                    "confidence": rel.confidence
                                }
                                for rel in knowledge["relationships"]
                            ]
                        
                        # Add to synthetic knowledge
                        synthetic_knowledge.append(knowledge)
                        
                        # Update progress and log
                        progress.update(synthesis_task, advance=1)
                        console.print(f"[green]✓ Synthesized knowledge from source {metadata.get('source_file', 'unknown')}[/green]")
                        console.print(f"  Title: {metadata.get('title', 'Unknown')}")
                        console.print(f"  Entities found: {len(knowledge.get('entities', []))}")
                        console.print(f"  Relationships found: {len(knowledge.get('relationships', []))}")
                        console.print(f"  Confidence: {knowledge.get('confidence', 0.0):.2f}")
                        
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
                # Convert any remaining Pydantic models to dicts before saving
                serializable_knowledge = []
                for k in synthetic_knowledge:
                    if isinstance(k, dict):
                        # If it's already a dict, just append it
                        serializable_knowledge.append(k)
                    elif hasattr(k, "model_dump"):
                        # If it's a Pydantic v2 model
                        serializable_knowledge.append(k.model_dump())
                    elif hasattr(k, "dict"):
                        # If it's a Pydantic v1 model
                        serializable_knowledge.append(k.dict())
                    else:
                        # If it's something else, try to convert to dict
                        try:
                            serializable_knowledge.append(dict(k))
                        except:
                            # If conversion fails, just append as is
                            serializable_knowledge.append(k)
                json.dump(serializable_knowledge, f, indent=2, ensure_ascii=False)
            
            # Update state with synthetic knowledge
            self.state["synthetic_knowledge"] = synthetic_knowledge
            
            # Store serializable knowledge in system state for example generation
            self.state["synthetic_knowledge"] = serializable_knowledge
            
            # Count total entities and relationships
            total_entities = sum(len(k.get('entities', [])) for k in synthetic_knowledge)
            total_relationships = sum(len(k.get('relationships', [])) for k in synthetic_knowledge)
            
            return {
                "synthetic_knowledge": synthetic_knowledge,
                "thought": f"Successfully synthesized knowledge from {len(sources.get('knowledge_sources', []))} sources with {total_entities} entities and {total_relationships} relationships"
            }
                
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge synthesis")
            return {
                "synthetic_knowledge": [],
                "thought": f"Error synthesizing knowledge: {str(e)}"
            }

    async def _generate_examples(self, synthetic_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training examples from synthetic knowledge."""
        try:
            examples = []
            knowledge_list = synthetic_knowledge.get("synthetic_knowledge", [])
            
            if not knowledge_list:
                return {
                    "training_examples": [],
                    "thought": "No synthetic knowledge available for example generation"
                }
            
            for knowledge in knowledge_list:
                if not isinstance(knowledge, dict):
                    continue
                
                # Generate examples using the knowledge
                example = {
                    "input_text": knowledge.get("content", ""),
                    "output_text": json.dumps({
                        "entities": knowledge.get("entities", []),
                        "relationships": knowledge.get("relationships", [])
                    }),
                    "metadata": knowledge.get("metadata", {})
                }
                examples.append(example)
            
            # Save examples to file
            results_dir = os.path.join("results", self.state["domain_name"])
            os.makedirs(results_dir, exist_ok=True)
            examples_path = os.path.join(results_dir, "training_examples.json")
            with open(examples_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            
            return {
                "training_examples": examples,
                "thought": f"Generated {len(examples)} training examples"
            }
        except Exception as e:
            logger.error(f"Error generating examples: {str(e)}")
            return {"training_examples": [], "thought": f"Error generating examples: {str(e)}"}

    async def _train_model(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Train model on examples."""
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
        finally:
            # Cleanup
            if hasattr(self, 'knowledge_system'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.knowledge_system, '_session', None) or getattr(self.knowledge_system, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing knowledge system session: {str(e)}")
            if hasattr(self, 'qa_system'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.qa_system, '_session', None) or getattr(self.qa_system, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing QA system session: {str(e)}")
            if hasattr(self, 'browser'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.browser, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing browser session: {str(e)}")
            await cleanup_progress()

    async def run(self, initial_state: Dict[str, Any]) -> CompilerStateDict:
        """Run the research agent with the given initial state."""
        try:
            # Initialize state with required fields
            state_dict: CompilerStateDict = {
                "content": initial_state.get("content", ""),
                "domain_name": initial_state.get("domain_name", ""),
                "results": initial_state.get("results", []),
                "knowledge_sources": initial_state.get("knowledge_sources", []),
                "synthetic_knowledge": initial_state.get("synthetic_knowledge", []),
                "training_examples": initial_state.get("training_examples", []),
                "model_metrics": initial_state.get("model_metrics", {}),
                "error": initial_state.get("error"),
                "feedback": initial_state.get("feedback"),
                "plan": initial_state.get("plan"),
                "join_decision": initial_state.get("join_decision")
            }
            
            self.state = cast(BaseCompilerState, state_dict)
            
            # Create plan
            plan = await self.generate_plan(self.state)
            state_dict["plan"] = plan
            self.state = cast(BaseCompilerState, state_dict)
            
            # Execute tasks
            results = await self.execute_tasks(plan.tasks, self.state)
            state_dict["results"] = results
            self.state = cast(BaseCompilerState, state_dict)
            
            # Generate final result
            final_result = await self._generate_final_result(self.state)
            state_dict["join_decision"] = final_result
            self.state = cast(BaseCompilerState, state_dict)
            
            return state_dict
        except Exception as e:
            logger.error(f"Error in research agent: {str(e)}")
            error_state: CompilerStateDict = {
                    "content": "",
                "domain_name": "",
                    "results": [],
                "knowledge_sources": [],
                "synthetic_knowledge": [],
                "training_examples": [],
                "model_metrics": {},
                "error": str(e),
                    "feedback": None,
                "plan": None,
                "join_decision": None
            }
            return error_state
        finally:
            # Cleanup
            if hasattr(self, 'knowledge_system'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.knowledge_system, '_session', None) or getattr(self.knowledge_system, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing knowledge system session: {str(e)}")
            if hasattr(self, 'qa_system'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.qa_system, '_session', None) or getattr(self.qa_system, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing QA system session: {str(e)}")
            if hasattr(self, 'browser'):
                try:
                    # Get session attribute if it exists
                    session = getattr(self.browser, 'session', None)
                    if session and hasattr(session, 'close'):
                        await session.close()
                except Exception as e:
                    logger.error(f"Error closing browser session: {str(e)}")
            await cleanup_progress()

    async def _extract_knowledge(self, source: Dict[str, Any]) -> Optional[ExtractedKnowledge]:
        """Extract knowledge from a source."""
        try:
            # Get content and metadata
            content = source.get("content", "").strip()
            source_metadata = source.get("metadata", {})
            
            if not content:
                log_warning_with_context("Empty content, skipping source", "Knowledge Extraction")
                return None
            
            # Generate QA pairs for content
            log_info_with_context("Generating QA pairs", "Knowledge Extraction")
            questions = await self.qa_system.generate_questions(content, num_questions=5)
            
            # Track QA metrics
            qa_metrics = {
                "questions_generated": len(questions),
                "questions_answered": 0,
                "average_confidence": 0.0
            }
            
            # Process each question
            answers = []
            total_confidence = 0.0
            
            for question in questions:
                try:
                    response = await self.qa_system.process_qa_chain(question.question)
                    if response and response.answer:
                        answers.append({
                            "question": question.question,
                            "answer": response.answer,
                            "confidence": response.confidence,
                            "sources": response.sources
                        })
                        total_confidence += response.confidence
                        qa_metrics["questions_answered"] += 1
                except Exception as e:
                    log_error_with_traceback(e, f"Error processing question: {question.question}")
                    continue
            
            if qa_metrics["questions_answered"] > 0:
                qa_metrics["average_confidence"] = total_confidence / qa_metrics["questions_answered"]
            
            # Process source with knowledge system
            log_info_with_context("Extracting knowledge", "Knowledge Extraction")
            result = await self.knowledge_system.process_source(content)
            if not result or not isinstance(result, dict):
                log_warning_with_context("No knowledge extracted, skipping source", "Knowledge Extraction")
                return None
            
            # Create metadata from dict
            metadata_dict = result.get("metadata", {})
            if isinstance(metadata_dict, dict):
                # Create new metadata object
                metadata = SourceMetadata(
                    source_type=metadata_dict.get("source_type", "text"),
                    confidence_score=metadata_dict.get("confidence_score", 0.8),
                    domain_relevance=metadata_dict.get("domain_relevance", 0.8),
                    timestamp=metadata_dict.get("timestamp", datetime.now().isoformat()),
                    validation_status=metadata_dict.get("validation_status", "pending"),
                    domain=metadata_dict.get("domain", "medical"),
                    qa_metrics=qa_metrics  # Add QA metrics to metadata
                )
                
                # Update with source metadata
                if isinstance(source_metadata, dict):
                    metadata = SourceMetadata(
                        source_type=source_metadata.get("source_type", metadata.source_type),
                        confidence_score=source_metadata.get("confidence_score", metadata.confidence_score),
                        domain_relevance=source_metadata.get("domain_relevance", metadata.domain_relevance),
                        timestamp=source_metadata.get("timestamp", metadata.timestamp),
                        validation_status=source_metadata.get("validation_status", metadata.validation_status),
                        domain=source_metadata.get("domain", metadata.domain),
                        qa_metrics=qa_metrics
                    )
            else:
                metadata = metadata_dict
            
            # Create ExtractedKnowledge
            extracted = ExtractedKnowledge(
                content=result.get("content", ""),
                entities=result.get("entities", []),
                relationships=result.get("relationships", []),
                metadata=metadata,
                confidence=result.get("confidence", 0.5),
                domain=metadata.domain if hasattr(metadata, "domain") else "medical",
                qa_pairs=answers  # Add QA pairs to extracted knowledge
            )
            return extracted
            
        except Exception as e:
            log_error_with_traceback(e, "Error in knowledge extraction")
            return None

    def _split_data(self, examples: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split examples into training and evaluation sets."""
        try:
            # Get examples list from input
            example_list = examples.get("training_examples", [])
            
            # If no examples, return empty lists
            if not example_list:
                return [], []
            
            # Split into train/eval sets (80/20)
            split_idx = int(len(example_list) * 0.8)
            train_data = example_list[:split_idx]
            eval_data = example_list[split_idx:]
            
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

    async def make_join_decision(self, state: BaseCompilerState) -> JoinDecision:
        """Make join decision based on task results."""
        try:
            log_info_with_context("Making join decision", "Decision")
            console.print("\n[bold green]Making Join Decision...[/bold green]")
            
            # Convert state to dictionary for access
            state_dict = cast(CompilerStateDict, dict(state))
            
            # Get task results
            results = state_dict["results"]
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
            plan = state_dict["plan"]
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

    async def _get_sources(self, domain: str) -> List[Dict[str, Any]]:
        """Get sources for a domain."""
        try:
            # Get actual domain name from state if needed
            if domain == "{state}":
                state_dict = self._get_state_dict()
                domain = state_dict["domain_name"]
            
            log_info_with_context(f"Starting topic research for domain: {domain}", "Research")
            
            # Perform searches
            sources = []
            search_queries = [
                f"{domain} overview",
                f"{domain} key concepts",
                f"{domain} latest developments"
            ]
            
            for query in search_queries:
                log_info_with_context(f"Searching for: {query}", "Search")
                results = await web_search(query, self.config)
                if results and "No results found" not in results:
                    sources.append({
                        "content": results,
                        "metadata": {
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                            "source_type": "web_search"
                        }
                    })
            
            return sources
            
        except Exception as e:
            log_error_with_traceback(e, "Error getting sources")
            return []

    def _get_state_dict(self) -> CompilerStateDict:
        """Get current state as a dictionary."""
        return cast(CompilerStateDict, dict(self.state))

async def main():
    """Main entry point."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Run research agent')
        parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
        args = parser.parse_args()
        
        # Load config
        async with aiofiles.open(args.config, 'r') as f:
            config = json.loads(await f.read())
        
        log_info_with_context("Initializing research agent", "Main")
        agent = ResearchAgent(config)
        
        log_info_with_context("Starting research agent", "Main")
        
        # Create initial state
        initial_state = {
            "content": "",
            "domain_name": config.get("domain_name", ""),
            "plan": None,
            "results": [],
            "join_decision": None,
            "final_result": None,
            "error": None,
            "feedback": None,
            "knowledge_sources": [],
            "synthetic_knowledge": [],
            "training_examples": [],
            "model_metrics": {}
        }
        
        # Run agent
        await agent.run(initial_state)
        
    except Exception as e:
        log_error_with_traceback(e, "Fatal error in research agent")
        raise

if __name__ == "__main__":
    # Set up argparse at module level
    import argparse
    
    # Run main with asyncio
    asyncio.run(main())
