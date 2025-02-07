from typing import List, Dict, Optional, Any, TypedDict, Protocol
from datetime import datetime
import json
from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_chroma import Chroma
from langchain_neo4j import Neo4jGraph
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from scripts.models import KnowledgeAcquisitionConfig, ExtractedKnowledge, Relationship, SourceMetadata
from scripts.logging_config import (
    log_error_with_traceback,
    log_warning_with_context,
    log_info_with_context,
    setup_logging,
    create_progress,
    log_extraction_results
)
from scripts.text_web_browser import SimpleTextBrowser, web_search
from scripts.mdconvert import MarkdownConverter
from scripts.visual_qa import VisualAnalyzer
from scripts.text_inspector_tool import TextInspector, TextAnalysis
from scripts.llm_compiler import LLMCompiler, Task, Plan, TaskResult, JoinDecision, CompilerState
import os
from loguru import logger
import asyncio
from prompts.knowledge_acquisition import (
    ExtractedKnowledge,
    Relationship,
    SourceMetadata,
    get_knowledge_extraction_prompt,
    ConfidenceEvaluation,
    get_confidence_evaluation_prompt,
    EntityResponse,
    get_entity_extraction_prompt,
    RelationshipResponse,
    get_relationship_extraction_prompt,
    MetadataResponse,
    get_metadata_generation_prompt,
    Task,
    Plan,
    get_plan_generation_prompt,
    JoinDecision,
    get_join_decision_prompt
)

# Initialize logging and rich console
setup_logging()
console = Console()

# Create shared progress instance
progress = create_progress()

# Initialize document converters
markdown_converter = MarkdownConverter()

class ContentProcessor(Protocol):
    """Protocol for content processors"""
    async def process(self, content: str, **kwargs) -> Dict[str, Any]: ...

class TextInspectorProcessor(ContentProcessor):
    """Text inspector processor using LLMCompiler pattern"""
    def __init__(self, llm):
        self.inspector = TextInspector(llm)
        
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process text content using text inspector"""
        try:
            result = await self.inspector.inspect_text(
                content,
              
            )
            return result.model_dump() if result else {}
        except Exception as e:
            log_error_with_traceback(e, "Error in text inspection")
            return {}

class VisualProcessor:
    """Visual processor using LLMCompiler pattern"""
    def __init__(self, llm):
        self.analyzer = VisualAnalyzer(llm)
        
    async def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """Process visual content"""
        try:
            if "image_path" not in kwargs:
                return {}
                
            # Analyze image content
            analysis = await self.analyzer.analyze_image(
                kwargs["image_path"],
                kwargs.get("question", "Describe this image in detail.")
            )
            
            return {
                "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis,
            }
            
        except Exception as e:
            log_error_with_traceback(e, "Error in visual processing")
            return {}

class DocumentProcessor:
    """Document processor using LLMCompiler pattern"""
    def __init__(self, llm):
        self.markdown_converter = MarkdownConverter()
        self.text_processor = TextInspectorProcessor(llm)
        self.visual_processor = VisualProcessor(llm)
        
    async def process_document(self, source_path: str, source_type: str) -> Document:
        """Process document with appropriate processor"""
        try:
            # Convert document to markdown
            result = self.markdown_converter.convert(source_path)
            
            # Process based on type
            if source_type in ["jpg", "jpeg", "png"]:
                analysis = await self.visual_processor.process(
                    result.text_content,
                    image_path=source_path
                )
            else:
                analysis = await self.text_processor.process(
                    result.text_content,
                    metadata={"source": source_path, "type": source_type}
                )
                
            # Create document with analysis
            return Document(
                page_content=result.text_content,
                metadata={
                    "source": source_path,
                    "type": source_type,
                    "title": result.title,
                    "analysis": analysis
                }
            )
            
        except Exception as e:
            log_error_with_traceback(e, f"Error processing document {source_path}")
            return Document(
                page_content="",
                metadata={
                    "source": source_path,
                    "type": source_type,
                    "error": str(e)
                }
            )

class EmbeddingGenerator(Protocol):
    """Protocol for embedding generators"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

class TaskState(BaseModel):
    """State for task execution"""
    content: str = Field(description="Content to process")
    knowledge: Optional[ExtractedKnowledge] = None
    embeddings: Optional[List[float]] = None
    graph_updates: Optional[bool] = Field(default=None, description="Whether graph updates are complete")
    error: Optional[str] = None

    @validator("content")
    def validate_content(cls, v: Any) -> str:
        if isinstance(v, str):
            return v
        if hasattr(v, "content"):
            return str(v.content)
        return str(v)

class ProcessingResult(TypedDict):
    """Result of processing"""
    success: bool
    error: Optional[str]

class ProcessingState(BaseModel):
    """State for processing workflow"""
    tasks: List[TaskState] = Field(default_factory=list)
    completed: List[TaskState] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def model_dump(self) -> Dict[str, Any]:
        """Convert state to dict"""
        return {
            "tasks": [task.model_dump() for task in self.tasks],
            "completed": [task.dict() for task in self.completed],
            "errors": [err for err in self.errors]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create state from dict"""
        return cls(
            tasks=[task if isinstance(task, TaskState) else TaskState(**task) for task in data.get("tasks", [])],
            completed=[task if isinstance(task, TaskState) else TaskState(**task) for task in data.get("completed", [])],
            errors=data.get("errors", [])
        )

class SearchFilter(TypedDict):
    confidence: Dict[str, float]

class WebSearchResponse(BaseModel):
    """Schema for web search response"""
    urls: List[str] = Field(description="List of relevant URLs")
    snippets: List[str] = Field(description="List of text snippets from search results")
    titles: List[str] = Field(description="List of result titles")

class WebContentResponse(BaseModel):
    """Schema for web content response"""
    content: str = Field(description="Extracted text content")
    url: str = Field(description="Source URL")
    title: Optional[str] = Field(None, description="Page title")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")

class KnowledgeAcquisitionSystem(LLMCompiler):
    """Knowledge acquisition system using LLMCompiler pattern"""
    def __init__(self, config: KnowledgeAcquisitionConfig):
        """Initialize the system"""
        # Initialize LLM first
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "MFDoom/deepseek-r1-tool-calling:1.5b"),
            format="json",
            temperature=0.7
        )
        super().__init__(llm)
        
        # Initialize other components
        self.config = config
        self.embeddings = OllamaEmbeddings(model='bge-m3', base_url='http://localhost:11434')
        self.vector_store = None
        self.graph = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.entity_parser = PydanticOutputParser(pydantic_object=EntityResponse)
        self.relationship_parser = PydanticOutputParser(pydantic_object=RelationshipResponse)
        self.metadata_parser = PydanticOutputParser(pydantic_object=MetadataResponse)

    async def _generate_plan(self, state: CompilerState) -> Plan:
        """Generate knowledge acquisition plan"""
        try:
            prompt = get_plan_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=Plan)
            plan = await chain.ainvoke({
                "content": state.get('content', '')
            })
            return plan

        except Exception as e:
            log_error_with_traceback(e, "Error generating plan")
            raise

    async def _execute_tasks(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute knowledge acquisition tasks"""
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
                    if task.tool == "extract_knowledge":
                        result = await self.process_source(task.args["content"])
                    elif task.tool == "generate_embeddings":
                        result = await self._generate_embeddings(task.args["content"])
                    elif task.tool == "update_graph":
                        await self._add_to_graph(task.args["knowledge"])
                        result = True
                    elif task.tool == "create_documents":
                        result = Document(
                            page_content=task.args["content"],
                            metadata=task.args["metadata"]
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
            log_error_with_traceback(e, "Error executing tasks")
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

    async def _generate_final_result(self, state: CompilerState) -> List[Document]:
        """Generate final documents from results"""
        try:
            documents = []
            for result in state.get('results', []):
                if result and result.result and isinstance(result.result, Document):
                    documents.append(result.result)
            return documents

        except Exception as e:
            log_error_with_traceback(e, "Error generating final result")
            raise

    async def add_source(self, source_path: str, source_type: str) -> List[Document]:
        """Add a knowledge source using the LLMCompiler pattern"""
        try:
            # Load and split source
            chunks = await self._load_chunks(source_path, source_type)
            if not chunks:
                return []

            # Process each chunk
            all_docs = []
            for chunk in chunks:
                # Create initial state with proper typing
                state: Dict[str, Any] = {
                    "content": str(chunk),
                    "plan": None,
                    "results": [],
                    "join_decision": None,
                    "final_result": None
                }

                # Run LLMCompiler workflow
                docs = await self.run(state)
                if docs:
                    all_docs.extend(docs)

            return all_docs

        except Exception as e:
            log_error_with_traceback(e, f"Error adding source {source_path}")
            return []

    async def _add_to_graph(self, knowledge: ExtractedKnowledge) -> None:
        """Add extracted knowledge to Neo4j graph"""
        if not self.graph:
            return
            
        try:
            # Create entity nodes
            for entity in knowledge.entities:
                await asyncio.to_thread(
                    self.graph.query,
                    """
                    MERGE (e:Entity {name: $name})
                    SET e += $properties
                    RETURN e
                    """,
                    {
                        "name": entity,
                        "properties": {
                            "confidence": knowledge.confidence,
                            "source_type": knowledge.metadata.source_type if knowledge.metadata else "unknown",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                )
            
            # Create relationships
            for rel in knowledge.relationships:
                await asyncio.to_thread(
                    self.graph.query,
                    """
                    MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
                    MERGE (s)-[r:RELATES {type: $relation}]->(t)
                    SET r += $properties
                    RETURN r
                    """,
                    {
                        "source": rel.source,
                        "target": rel.target,
                        "relation": rel.relation,
                        "properties": {
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                )
                
        except Exception as e:
            log_error_with_traceback(e, "Error adding to graph database")

    async def process_source(self, content: str) -> ExtractedKnowledge:
        """Process content to extract knowledge"""
        if not self.llm:
            await self.initialize()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                # Create tasks for tracking
                extraction_task = progress.add_task("[cyan]Extracting knowledge...", total=4)
                
                # Run text analysis first
                inspector = TextInspector(self.llm)
                analysis: TextAnalysis = await inspector.inspect_text(content)
                progress.update(extraction_task, advance=1)
                
                # Extract knowledge using knowledge extraction prompt
                prompt = get_knowledge_extraction_prompt()
                chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ExtractedKnowledge)
                initial_knowledge = await chain.ainvoke({
                    "text": content,
                    "analysis": analysis.model_dump() if hasattr(analysis, "model_dump") else analysis
                })
                progress.update(extraction_task, advance=1)
                
                # Run entity extraction, relationship extraction, and metadata generation in parallel
                tasks = [
                    self._extract_entities(content),
                    self._extract_relationships(content),
                    self._generate_metadata(content)
                ]
                
                log_info_with_context("Starting parallel knowledge extraction", "Knowledge Processing")
                entities, relationships, metadata = await asyncio.gather(*tasks)
                progress.update(extraction_task, advance=1)

                # Evaluate confidence with factors
                confidence_eval = await self._evaluate_confidence(content, entities, relationships)
                factors = confidence_eval.factors
                
                # Optionally enrich with web search
                web_context = None
                if self.config.enable_web_search:
                    try:
                        search_results = await web_search(content)
                        if search_results and not search_results.startswith("Error"):
                            web_context = search_results
                    except Exception as e:
                        log_error_with_traceback(e, "Error in web search enrichment")
                
                progress.update(extraction_task, advance=1)

                # Create final knowledge object
                knowledge = ExtractedKnowledge(
                    content=content,
                    entities=entities,
                    relationships=relationships,
                    confidence=confidence_eval.confidence,
                    metadata=SourceMetadata(
                        source_type="text",
                        confidence_score=confidence_eval.confidence,
                        domain_relevance=confidence_eval.factors.context_relevance,
                        timestamp=datetime.now().isoformat(),
                        validation_status="pending",
                        domain="knowledge"
                    )
                )
                
                # Log extraction results
                log_extraction_results(knowledge)
                return knowledge
                
        except Exception as e:
            log_error_with_traceback(e, "Error in process_source")
            raise

    async def _generate_embeddings(self, content: str) -> Optional[List[float]]:
        """Generate embeddings with proper error handling"""
        if not self.embeddings:
            await self.initialize()
            
        try:
            log_info_with_context("Generating embeddings", "Embeddings")
            
            # Use a safe wrapper to handle the None case
            def generate_embeddings(text: str) -> Optional[List[List[float]]]:
                if self.embeddings:
                    return self.embeddings.embed_documents([text])
                return None
                
            embeddings = await asyncio.to_thread(generate_embeddings, content)
            if embeddings and isinstance(embeddings, list):
                log_info_with_context("Successfully generated embeddings", "Embeddings")
                return embeddings[0]
            
            log_warning_with_context("Failed to generate embeddings", "Embeddings")
            return None
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating embeddings")
            return None

    async def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content"""
        await self._ensure_initialized()
        if not self.llm:
            return []
        
        try:
            prompt = get_entity_extraction_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=EntityResponse)
            result = await chain.ainvoke({
                "content": content
            })
            return result.entities
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting entities")
            return []

    async def _extract_relationships(self, content: str) -> List[Relationship]:
        """Extract relationships from content"""
        await self._ensure_initialized()
        if not self.llm:
            return []
        
        try:
            prompt = get_relationship_extraction_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=RelationshipResponse)
            result = await chain.ainvoke({
                "content": content
            })
            
            # Convert to Relationship objects
            relationships = []
            for rel in result.relationships:
                relationships.append(Relationship(
                    source=rel["source"],
                    relation=rel["relation"],
                    target=rel["target"]
                ))
            return relationships
            
        except Exception as e:
            log_error_with_traceback(e, "Error extracting relationships")
            return []

    async def _generate_metadata(self, content: str) -> SourceMetadata:
        """Generate metadata for content"""
        await self._ensure_initialized()
        if not self.llm:
            return self._create_default_metadata()
        
        try:
            prompt = get_metadata_generation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=MetadataResponse)
            result = await chain.ainvoke({
                "content": content
            })
            return result.metadata
            
        except Exception as e:
            log_error_with_traceback(e, "Error generating metadata")
            return self._create_default_metadata()

    def _create_default_metadata(self) -> SourceMetadata:
        """Create default metadata"""
        return SourceMetadata(
            source_type="text",
            confidence_score=0.0,
            domain_relevance=0.0,
            timestamp=datetime.now().isoformat(),
            validation_status="failed",
            domain="knowledge"
        )

    async def _ensure_initialized(self) -> None:
        """Ensure all components are initialized"""
        if not self.llm or not self.embeddings or not self.vector_store or not self.graph:
            log_info_with_context("Initializing missing components", "Initialization")
            await self.initialize()

    async def _load_chunks(self, source_path: str, source_type: str = "text") -> List[str]:
        """Load and chunk content from source"""
        try:
            # Create progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            )
            
            with progress:
                # Create tasks for tracking
                load_task = progress.add_task("[cyan]Loading document...", total=1)
                chunk_task = progress.add_task("[cyan]Splitting into chunks...", total=1)
                
                # Load document based on type
                log_info_with_context(f"Loading document: {source_path}", "Document Loading")
                
                document_text = ""
                if source_type.lower() == "pdf":
                    loader = PyPDFLoader(source_path)
                    pages = await asyncio.to_thread(loader.load)
                    document_text = "\n\n".join(page.page_content for page in pages)
                elif source_type.lower() == "text":
                    loader = TextLoader(source_path)
                    docs = await asyncio.to_thread(loader.load)
                    document_text = "\n\n".join(doc.page_content for doc in docs)
                elif source_type.lower() in ["html", "htm"]:
                    # Use SimpleTextBrowser for web content
                    browser = SimpleTextBrowser()
                    document_text = await browser.visit(source_path)
                else:
                    # Use markdown converter for other types
                    result = await asyncio.to_thread(markdown_converter.convert, source_path, file_extension=source_type)
                    document_text = result.text_content if result else ""
                
                if not document_text:
                    log_warning_with_context("Failed to load document", "Document Loading")
                    return []
                
                progress.update(load_task, advance=1)
                
                # Split into chunks
                chunks = self.text_splitter.split_text(document_text)
                progress.update(chunk_task, advance=1)
                
                # Store in vector store
                if chunks and self.vector_store:
                    try:
                        embeddings = await self._generate_embeddings("\n".join(chunks))
                        if embeddings:
                            await asyncio.to_thread(
                                self.vector_store.add_texts,
                                texts=chunks,
                                embeddings=[embeddings],
                                metadatas=[{"source": source_path, "type": source_type}]
                            )
                    except Exception as e:
                        log_error_with_traceback(e, "Error storing in vector store")
                
                return chunks
                
        except Exception as e:
            log_error_with_traceback(e, f"Error loading chunks from {source_path}")
            return []

    async def initialize(self) -> "KnowledgeAcquisitionSystem":
        """Async initialization of components"""
        try:
            # Initialize vector store
            self.vector_store = Chroma(
                collection_name="knowledge_base",
                embedding_function=self.embeddings,
                persist_directory="./data/vector_store"
            )
            
            # Initialize graph database
            self.graph = Neo4jGraph(
                url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                username=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password")
            )
            
            log_info_with_context("Knowledge acquisition system initialized successfully", "Initialization")
            return self
            
        except Exception as e:
            log_error_with_traceback(e, "Error initializing knowledge acquisition system")
            raise

    async def _evaluate_confidence(self, content: str, entities: List[str], relationships: List[Relationship]) -> ConfidenceEvaluation:
        """Evaluate confidence using proper evaluation prompts."""
        try:
            prompt = get_confidence_evaluation_prompt()
            chain = prompt | self.llm | PydanticOutputParser(pydantic_object=ConfidenceEvaluation)
            result = await chain.ainvoke({
                "content": content,
                "entities": entities,
                "relationships": [r.model_dump() for r in relationships],
                "source_type": "text"
            })
            return result
        except Exception as e:
            log_error_with_traceback(e, "Error evaluating confidence")
            raise